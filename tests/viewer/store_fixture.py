"""Store-layout instances for the viewer's and ``hls follow``'s tests: plans, manifests,
records, chunked traces, leases and event logs, written where and as the orchestrator, the
runners and finalize write them, without running any of them. The records follow the shapes
finalize writes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from hls.agent.transcript import COPY_DIR
from hls.core.trace import Trace
from hls.pipeline.plan import Outcome, Plan
from hls.spec.resources import GrantedResources, ResourceShape
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder, LockRecord
from hls.store.manifest import Manifest, OutputRef, Provenance, Resources, Times, canonical_json
from hls.trace.store import write_trace
from hls.viewer.server import create_app, token_cookie
from tests.pipeline.plans import noop_plan

#: The viewer's session token in these tests.
TOKEN = "test-session-token-" + "0" * 24
#: The port a test client's requests reach: its base URL, ``http://testserver``, names none.
CLIENT_PORT = 80

#: The task every instance belongs to.
TASK = "multitenant_slo"
#: A rollout and a grading of it.
ROLLOUT, GRADING = "20261004T120000Z-a1b2c3", "g-20261005T080000Z-d4e5f6"
#: The stages of a rollout that builds and grades, and of a grading.
ROLLOUT_STAGES = ("provision", "images", "agent_build", "package", "deploy_drive", "verify", "score",
                  "finalize")
GRADING_STAGES = ("provision", "images", "deploy_drive", "verify", "score", "finalize")
#: The task spec the records carry: enough for the timeline's tenant name and SLO.
SPEC = {"id": TASK, "tenants": [{"id": "chat", "data_source": {"name": "sharegpt"},
                                 "workload": {"slo": {"target_ms": 2000}}}]}
#: When the instances were made.
CREATED = 1_790_000_000.0
#: How a sealed instance ended.
SUCCEEDED = Outcome(status="succeeded", reason="every stage published", at=CREATED + 900)
#: The verify and score stages' outputs.
VERDICT = {"aggregate_passed": True, "failure_mode": None, "per_event": {}, "diagnostics": {},
           "per_verifier_results": []}
SCORE = {"value": 1.12, "gate_passed": True, "breakdown": {"slo": {"p99_ms": 1500.0}}}
#: The agent_build outputs the viewer reads, by the name its record gives them (``hls.stages.agent_build``)
#: to their path under the attempt's ``outputs/``, and the contents each is given.
BUILD_OUTPUTS = {"build": "build.json", "brief_provenance": "brief_provenance.json",
                 "source_bundle": "source.bundle", "transcript": "transcript/INDEX.json"}
BRIEF_PROVENANCE = {"sections": [{"block": "goal", "source": "base", "rendered": "Build a server."}]}
BUNDLE = b"# v2 git bundle\n"
#: The role logs the rollout's eval stages left, by the file name finalize records (``<role>.out`` or
#: ``<role>.err``) to the stage that ran the role.
ROLE_LOGS = {"artifact.out": "deploy_drive", "artifact.err": "deploy_drive", "driver.out": "deploy_drive",
             "verify.err": "verify"}
#: The agent's log in the transcript's copy, and what it holds.
AGENT_LOG = f"transcript/{COPY_DIR}/claude-code.txt"
AGENT_LOG_TEXT = b'{"type": "result"}\n'


def rollout_key(rollout: str = ROLLOUT) -> str:
    """The store key of a rollout of ``TASK``."""
    return f"runs/{TASK}/{rollout}"


def grading_key(rollout: str = ROLLOUT, grading: str = GRADING) -> str:
    """The store key of a grading of that rollout."""
    return f"{rollout_key(rollout)}/gradings/{grading}"


def viewer_client(root: Path, **kwargs: Any) -> TestClient:
    """A client of the viewer of *root* that holds the session's token, as a browser does once
    it opened the URL ``hls view`` printed."""
    return TestClient(create_app(root, token=TOKEN, **kwargs), cookies={token_cookie(CLIENT_PORT): TOKEN})


def dumps(doc: Any) -> bytes:
    """*doc* as a record file's bytes."""
    return json.dumps(doc, indent=2).encode()


def write_plan(store: Path, instance: str, stages: tuple[str, ...], *, tmp: Path,
               outcome: Outcome | None = None, revision: int = 1) -> Plan:
    """Write revision *revision* of a plan of *instance*: one stage of each name in *stages*,
    named as its kind, in one allocation, and ended with *outcome* when given. *tmp* is a scratch
    directory the plan's settings name."""
    base = noop_plan(tmp / "planning")
    template = base.stages[0]
    plan = base.model_copy(update={
        "instance": instance, "task": TASK, "created_at": CREATED, "revision": revision, "outcome": outcome,
        "stages": [template.model_copy(update={"name": s, "kind": s, "after": [], "params": {}})
                   for s in stages],
        "allocations": [base.allocations[0].model_copy(update={"stages": list(stages)})]})
    # A reader validates every revision, so the fixture does too.
    plan = Plan.model_validate_json(plan.model_dump_json())
    path = store / InstanceLayout(instance).plan_revision(revision)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan.model_dump_json())
    return plan


def publish(store: Path, instance: str, stage: str, files: dict[str, bytes], *, trace: Trace | None = None,
            refs: dict[str, OutputRef] | None = None) -> Manifest:
    """Write *files*, each by its path under attempt 1's ``outputs/``, and with *trace* a chunked
    trace under ``outputs/trace/``; then publish the stage's manifest naming them, and *refs* as
    given."""
    layout = InstanceLayout(instance)
    outputs: dict[str, OutputRef] = dict(refs or {})
    for rel, data in files.items():
        key = layout.attempt_file(stage, 1, f"outputs/{rel}")
        (store / key).parent.mkdir(parents=True, exist_ok=True)
        (store / key).write_bytes(data)
        outputs[rel] = OutputRef(key=key, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    if trace is not None:
        write_trace(trace, store / layout.attempt(stage, 1) / "outputs" / "trace", ending="drained",
                    driver={"class": "Driver"})
        key = layout.attempt_file(stage, 1, "outputs/trace/END.json")
        data = (store / key).read_bytes()
        outputs["trace"] = OutputRef(key=key, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    manifest = Manifest(
        instance=instance, stage=stage, attempt=1, outputs=outputs, inputs={},
        provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None,
                              image_digests={}, config_sha256="c" * 64),
        resources=Resources(declared=ResourceShape(cpus=1, memory_gib=1, time_sec=60),
                            granted=GrantedResources(gpus=0, cpus=1, memory_gib=1.0, disk_free_gib=1.0)),
        times=Times(submitted_at=CREATED, allocation_started_at=CREATED, started_at=CREATED,
                    ended_at=CREATED + 60))
    path = store / layout.manifest(stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(manifest))
    return manifest


def write_bytes(store: Path, key: str, data: bytes) -> Path:
    """Write *data* at store key *key* and return its path."""
    path = store / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def write_lease(store: Path, instance: str, *, heartbeat_at: float, released: bool = False,
                ttl_sec: float = 120.0) -> None:
    """Write the instance's lease as generation 1, held by an orchestrator on a login node."""
    holder = Holder(host="login01", pid=4242, job_id=None, purpose="orchestrator")
    record = LockRecord(generation=1, holder=holder, acquired_at=CREATED, heartbeat_at=heartbeat_at,
                        ttl_sec=ttl_sec, released=released)
    key = f"{InstanceLayout(instance).lease}/gen-00000001.json"
    write_bytes(store, key, record.model_dump_json().encode())


def append_raw(store: Path, key: str, text: str) -> None:
    """Append *text* to the file at store key *key*, as a writer's append lands."""
    path = store / key
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(text)


def append_events(store: Path, key: str, events: list[dict[str, Any]]) -> None:
    """Append *events* to the log at store key *key*, one JSON line each, as ``EventLog`` writes."""
    append_raw(store, key, "".join(json.dumps(e, sort_keys=True) + "\n" for e in events))


def rollout_record(*, status: str = "ok") -> dict[str, Any]:
    """A rollout's ``rollout.json``: its provenance, its build record and its conditions."""
    return {
        "rollout_id": ROLLOUT, "task_id": TASK, "created_at": CREATED, "finished_at": CREATED + 900,
        "status": status,
        "config": {"task_id": TASK, "split": "test", "spec": SPEC, "image_digests": {},
                   "hls_commit": "a" * 40, "harbor_commit": "b" * 40, "rollout_overrides": {},
                   "reasoning_effort": "max",
                   "artifact": {"kind": "package", "stage": "package", "manifest": None}},
        "build": {"build": {"arm": {"name": "arm-a", "harness": "claude-code", "model": "claude-opus-5"},
                            "ending": "agent_ended"},
                  "outputs": {name: f"{rollout_key()}/stages/agent_build/attempts/1/outputs/{rel}"
                              for name, rel in BUILD_OUTPUTS.items()},
                  "package": {"status": "built"}, "started_at": CREATED + 60, "ended_at": CREATED + 600,
                  "usage": {"n_input_tokens": 1000, "n_output_tokens": 100, "n_cache_tokens": None,
                            "cost_usd": 1.5},
                  "upload_sha256": None},
        "conditions": {"attempts": {}}, "error": None,
    }


def grading_record(eval_id: str, value: float) -> dict[str, Any]:
    """A ``grading.json`` of eval *eval_id* that scored *value*; the rollout's own names the role
    logs of its stages (``ROLE_LOGS``) by file name and store key."""
    logs = {name: f"{rollout_key()}/stages/{stage}/attempts/1/logs/{name}" for name, stage in ROLE_LOGS.items()}
    return {"episode": {"eval_id": eval_id, "triggered_by": "harness" if eval_id == "terminal" else "regrade",
                        "checkpoint": "e" * 64, "started_at": CREATED + 610, "finished_at": CREATED + 890,
                        "trace_path": "", "verdict_summary": {"gate_passed": True},
                        "score": {"value": value, "gate_passed": True},
                        "container_logs": logs if eval_id == "terminal" else {}},
            "provenance": {"task_id": TASK, "spec": SPEC}, "source_rollout": ROLLOUT,
            "mode": "original" if eval_id == "terminal" else "adopt",
            "conditions": {"attempts": {}}}


def index_row(*, grading: str | None = None, **changes: Any) -> dict[str, Any]:
    """An index row of the rollout, or of its grading *grading*, with finalize's columns."""
    row = {"rollout_id": ROLLOUT, "grading_id": grading, "mode": "original" if grading is None else "adopt",
           "created_at": CREATED, "finished_at": CREATED + 900, "status": "ok", "score": 1.12,
           "gate_passed": True, "cost_usd": 1.5, "image_digests": {}, "failed_stage": None, "fault": None,
           "scanned": True,
           "quarantined": [], "quarantined_count": 0, "arm": "arm-a", "model_fingerprint": "fp",
           "accelerator": "H200", "agent_implementer_model": "claude-opus-5"}
    return {**row, **changes}


def write_index_row(store: Path, row: dict[str, Any]) -> None:
    """Write *row* as its instance's index file, ``<rollout>.json`` or ``<rollout>+<grading>.json``."""
    name = row["rollout_id"] + (f"+{row['grading_id']}" if row.get("grading_id") else "")
    write_bytes(store, f"runs/{TASK}/index/{name}.json", (json.dumps(row) + "\n").encode())


def sealed_rollout(store: Path, tmp: Path, *, trace: Trace, grading_trace: Trace) -> None:
    """A rollout that built and graded, with its records and index row, and one grading of it with
    its own. The rollout's drive published *trace* and the grading's *grading_trace*."""
    key = rollout_key()
    write_plan(store, key, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp)
    write_lease(store, key, heartbeat_at=CREATED + 900, released=True)
    publish(store, key, "agent_build", {"build.json": b"{}", "brief_provenance.json": dumps(BRIEF_PROVENANCE),
                                        "source.bundle": BUNDLE, "transcript/INDEX.json": b"{}"})
    # The transcript's copy of the agent's logs lies beside its index; the manifest names the index only.
    write_bytes(store, f"{key}/stages/agent_build/attempts/1/outputs/{AGENT_LOG}", AGENT_LOG_TEXT)
    publish(store, key, "deploy_drive", {"drive.json": b"{}"}, trace=trace)
    publish(store, key, "verify", {"verdict.json": dumps(VERDICT)})
    for name, stage in ROLE_LOGS.items():
        write_bytes(store, f"{key}/stages/{stage}/attempts/1/logs/{name}", f"{name} text\n".encode())
    publish(store, key, "score", {"score.json": dumps(SCORE)})
    publish(store, key, "finalize", {"rollout.json": dumps(rollout_record()),
                                     "grading.json": dumps(grading_record("terminal", 1.12))})
    write_index_row(store, index_row())
    graded = grading_key()
    write_plan(store, graded, GRADING_STAGES, outcome=SUCCEEDED, tmp=tmp)
    publish(store, graded, "deploy_drive", {"drive.json": b"{}"}, trace=grading_trace)
    publish(store, graded, "verify", {"verdict.json": dumps(VERDICT)})
    publish(store, graded, "score", {"score.json": dumps({**SCORE, "value": 0.98})})
    publish(store, graded, "finalize", {"grading.json": dumps(grading_record(GRADING, 0.98))})
    write_index_row(store, index_row(grading=GRADING, score=0.98))
