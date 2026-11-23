"""The viewer on the store's layout: instances listed from their index rows or their plans,
records read from finalize's outputs or ``failed/``, a rollout's gradings as the instances under
it, chunked traces checked against their ``END.json``, keys that lead out of an instance never
followed, and the store's logs streamed until the instance ended. The published layout has its
own module, ``test_published``."""

from __future__ import annotations

import ast
import gzip
import json
import re
import shutil
import time
from pathlib import Path
from typing import get_args

import pytest

from hls.agent.transcript import COPY_DIR
from hls.arms import ADAPTERS
from hls.core.trace import Trace
from hls.obs.record import RolloutStatus
from hls.pipeline.plan import Outcome
from hls.stages import agent_build, deploy_drive
from hls.store.layout import INDEX_DIR, InstanceLayout
from hls.store.manifest import OutputRef
from hls.substrate.base import FaultRecord
from hls.viewer import scanner as scanner_mod
from hls.viewer import server as server_mod
from hls.viewer.scanner import RolloutScanner
from tests.trace.traces import drive_trace
from tests.viewer.published_fixture import published_runs
from tests.viewer.store_fixture import (AGENT_LOG_TEXT, BRIEF_PROVENANCE, BUILD_OUTPUTS, BUNDLE, CREATED, GRADING,
                                        GRADING_STAGES, ROLE_LOGS, ROLLOUT, ROLLOUT_STAGES, SPEC, SUCCEEDED, TASK,
                                        append_events, dumps, grading_record, index_row, publish, rollout_key,
                                        rollout_record, sealed_rollout, viewer_client, write_bytes,
                                        write_index_row, write_lease, write_plan)

#: The viewer page's modules of names: the constants it reads the stage pipeline's files and events
#: by, for a build's and for an eval's.
VIEWER_BUILD_MODULE = Path(__file__).resolve().parents[2] / "apps/viewer/app/lib/build.ts"
VIEWER_EVAL_MODULE = Path(__file__).resolve().parents[2] / "apps/viewer/app/lib/eval-progress.ts"
#: The viewer page's types of the records it is served.
VIEWER_TYPES_MODULE = Path(__file__).resolve().parents[2] / "apps/viewer/app/lib/types.ts"

#: The rollout's drive, and the grading's: a shorter one, so a view of one is never the other's.
TRACE = drive_trace()
GRADING_TRACE = Trace(events=drive_trace().events[:5])


def _sealed(tmp_path: Path) -> tuple[Path, RolloutScanner]:
    store = tmp_path / "store"
    sealed_rollout(store, tmp_path, trace=TRACE, grading_trace=GRADING_TRACE)
    return store, RolloutScanner(store / "runs")


def test_the_viewer_lists_store_rollouts_and_gradings(tmp_path):
    _, scanner = _sealed(tmp_path)
    rows = scanner.list_rollouts()
    # The grading's own index row is not a second rollout.
    assert [(r["rollout_id"], r["layout"]) for r in rows] == [(ROLLOUT, "store")]
    [row] = rows
    assert (row["task_id"], row["status"], row["score"], row["arm"]) == (TASK, "ok", 1.12, "arm-a")
    rec = scanner.get_rollout(TASK, ROLLOUT)
    assert (rec["layout"], rec["instance"]) == ("store", rollout_key())
    assert rec["build"]["usage"]["cost_usd"] == 1.5
    assert [e["eval_id"] for e in rec["evals"]] == ["terminal", GRADING]
    assert [e["score"]["value"] for e in rec["evals"]] == [1.12, 0.98]


def test_a_store_records_build_that_is_not_a_rollout_build_is_served_as_none_with_where_it_fails(tmp_path):
    """A store record's build that does not validate as a ``RolloutBuild`` reaches the page as
    none, with the field locations at which it fails (``build_invalid``), which the page shows as
    the reason; never a value the build holds. A build that validates, and a record that holds
    none, have none."""
    store, scanner = _sealed(tmp_path)
    assert scanner.get_rollout(TASK, ROLLOUT)["build_invalid"] is None
    record = store / rollout_key() / "stages/finalize/attempts/1/outputs/rollout.json"
    build = {**rollout_record()["build"], "usage": "VALUE-ONE", "cost_usd": "VALUE-TWO"}
    del build["ended_at"]
    record.write_bytes(dumps({**rollout_record(), "build": build}))
    client = viewer_client(scanner.runs_dir)
    for rec in (scanner.get_rollout(TASK, ROLLOUT), client.get(f"/api/rollouts/{TASK}/{ROLLOUT}").json()):
        assert rec["build"] is None
        assert rec["build_invalid"] == ["build.cost_usd", "build.ended_at", "build.usage"]
        assert "VALUE" not in json.dumps(rec["build_invalid"])
    record.write_bytes(dumps({**rollout_record(), "build": ["VALUE"]}))
    assert scanner.get_rollout(TASK, ROLLOUT)["build_invalid"] == ["build"]
    record.write_bytes(dumps({**rollout_record(), "build": None}))
    none = scanner.get_rollout(TASK, ROLLOUT)
    assert (none["build"], none["build_invalid"]) == (None, None)


def test_the_viewer_opens_a_failed_instances_records(tmp_path):
    store = tmp_path / "store"
    key = rollout_key()
    fault = FaultRecord(fault_class="contestant", reason="contestant role sandbox wrote past its disk cap",
                        retryable=False)
    write_plan(store, key, ROLLOUT_STAGES, tmp=tmp_path,
               outcome=Outcome(status="failed", reason=f"agent_build: {fault.reason}", at=CREATED + 700,
                               fault=fault))
    write_bytes(store, f"{key}/failed/rollout.json", dumps(rollout_record()))
    write_bytes(store, f"{key}/failed/grading.json", dumps(grading_record("terminal", 0.0)))
    write_index_row(store, index_row(score=0.0, gate_passed=False, failed_stage="agent_build",
                                     fault=fault.reason, scanned=False))
    scanner = RolloutScanner(store / "runs")
    [row] = scanner.list_rollouts()
    assert (row["failed_stage"], row["fault"]) == ("agent_build", fault.reason)
    assert (row["score"], row["scanned"]) == (0.0, False)
    rec = scanner.get_rollout(TASK, ROLLOUT)
    assert rec["status"] == "ok"
    assert [(e["eval_id"], e["score"]["value"]) for e in rec["evals"]] == [("terminal", 0.0)]


def test_the_viewer_reads_a_chunked_trace(tmp_path):
    _, scanner = _sealed(tmp_path)
    summary = scanner.trace_summary(TASK, ROLLOUT, "terminal")
    assert (summary.total, summary.p99_ms) == (len(TRACE.events), 1500.0)
    page = scanner.trace_page(TASK, ROLLOUT, "terminal", limit=100)
    assert [r["request_id"] for r in page.rows] == [ev.request_id for ev in TRACE.events]
    assert scanner.trace_row(TASK, ROLLOUT, "terminal", "s0.0")["prompt"] == "ask s0.0"
    timeline = scanner.timeline(TASK, ROLLOUT, "terminal")
    assert len(timeline.requests) == len(TRACE.events)
    chat = next(t for t in timeline.tenants if t.id == "chat")
    assert (chat.name, chat.slo.kind, chat.slo.target_ms) == ("sharegpt", "latency", 2000)
    # The grading's eval is its own instance's drive, not the rollout's.
    assert scanner.trace_summary(TASK, ROLLOUT, GRADING).total == len(GRADING_TRACE.events)


def test_an_eval_the_instance_does_not_hold_has_no_trace(tmp_path):
    """An eval id names the rollout's own terminal eval or one of its gradings, nothing else:
    an unknown id is not shown the rollout's own drive."""
    _, scanner = _sealed(tmp_path)
    assert scanner.trace_summary(TASK, ROLLOUT, "g-20261006T000000Z-000000") is None
    assert scanner.trace_page(TASK, ROLLOUT, "g-20261006T000000Z-000000").total == 0
    assert scanner.timeline(TASK, ROLLOUT, "g-20261006T000000Z-000000") is None
    assert scanner.trace_summary(TASK, ROLLOUT, "..") is None


def test_an_altered_chunk_reads_as_no_trace(tmp_path):
    """The chunk is replaced by another whole gzip file holding as many events, so only its
    sha256 against ``END.json`` tells it from the one the drive wrote."""
    store, scanner = _sealed(tmp_path)
    chunk = store / rollout_key() / "stages/deploy_drive/attempts/1/outputs/trace/chunk-000000.jsonl.gz"
    lines = gzip.decompress(chunk.read_bytes()).splitlines()
    assert b'"ask s0.0"' in lines[0]
    lines[0] = lines[0].replace(b'"ask s0.0"', b'"FORGED"')
    chunk.write_bytes(gzip.compress(b"\n".join(lines) + b"\n"))
    assert scanner.trace_summary(TASK, ROLLOUT, "terminal") is None
    assert scanner.trace_page(TASK, ROLLOUT, "terminal").total == 0
    assert scanner.timeline(TASK, ROLLOUT, "terminal").requests == []


def test_a_running_store_instance_is_listed_from_its_plan(tmp_path):
    store = tmp_path / "store"
    key = rollout_key()
    write_plan(store, key, ROLLOUT_STAGES, tmp=tmp_path)
    write_lease(store, key, heartbeat_at=time.time())
    publish(store, key, "provision", {"provisioned.json": b"{}"})
    publish(store, key, "images", {"images.json": b"{}"})
    scanner = RolloutScanner(store / "runs")
    [row] = scanner.list_rollouts()
    assert (row["status"], row["phase"]) == ("running", "stage agent_build")
    assert (row["score"], row["layout"]) == (None, "store")
    rec = scanner.get_rollout(TASK, ROLLOUT)
    assert (rec["status"], rec["evals"], rec["build"]) == ("running", [], None)
    assert scanner.instance_ended(TASK, ROLLOUT) is False


def test_a_store_instance_no_orchestrator_holds_is_stalled(tmp_path):
    store = tmp_path / "store"
    key = rollout_key()
    write_plan(store, key, ROLLOUT_STAGES, tmp=tmp_path)
    write_lease(store, key, heartbeat_at=time.time(), released=True)
    [row] = RolloutScanner(store / "runs").list_rollouts()
    assert (row["status"], row["phase"]) == (scanner_mod.STALLED, "stage provision")
    write_lease(store, key, heartbeat_at=time.time() - 1000)
    [row] = RolloutScanner(store / "runs").list_rollouts()
    assert row["status"] == scanner_mod.STALLED


def test_a_void_store_instance_is_listed_void_from_its_plan(tmp_path):
    store = tmp_path / "store"
    key = rollout_key()
    write_plan(store, key, ROLLOUT_STAGES, tmp=tmp_path,
               outcome=Outcome(status="void", at=CREATED + 800,
                               reason="deploy_drive: the node failed; retries exhausted (2 of 2)"))
    write_lease(store, key, heartbeat_at=CREATED + 800, released=True)
    scanner = RolloutScanner(store / "runs")
    [row] = scanner.list_rollouts()
    assert (row["status"], row["phase"], row["finished_at"]) == ("void", None, CREATED + 800)
    rec = scanner.get_rollout(TASK, ROLLOUT)
    assert (rec["status"], rec["reason"], rec["evals"]) == ("void", "deploy_drive: the node failed; retries "
                                                                    "exhausted (2 of 2)", [])
    assert scanner.instance_ended(TASK, ROLLOUT) is True


def test_an_output_key_outside_its_instance_is_not_read(tmp_path):
    store = tmp_path / "store"
    key = rollout_key()
    write_plan(store, key, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    # Another instance's record, which a manifest of this one names.
    other = write_bytes(store, f"runs/{TASK}/other/stages/finalize/attempts/1/outputs/rollout.json",
                        dumps(rollout_record(status="invalid_artifact")))
    ref = OutputRef(key=f"runs/{TASK}/other/stages/finalize/attempts/1/outputs/rollout.json", sha256="0" * 64,
                    size=other.stat().st_size)
    # And a record of a grading under this rollout, an instance of its own though its key lies
    # under this one's.
    nested_key = f"{key}/gradings/{GRADING}/stages/finalize/attempts/1/outputs/grading.json"
    nested = write_bytes(store, nested_key, dumps(grading_record("terminal", 7.7)))
    nested_ref = OutputRef(key=nested_key, sha256="0" * 64, size=nested.stat().st_size)
    publish(store, key, "finalize", {}, refs={"rollout": ref, "grading": nested_ref})
    # And a grading record inside the instance that is a link out of it.
    outside = write_bytes(store, "elsewhere/grading.json", dumps(grading_record("terminal", 9.9)))
    link = store / key / "failed" / "grading.json"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)
    rec = RolloutScanner(store / "runs").get_rollout(TASK, ROLLOUT)
    assert (rec["status"], rec["evals"]) == ("succeeded", [])


def test_a_stores_warmers_are_listed_with_their_evaluations(tmp_path):
    """A package warmer has no records of its own; its gradings are instances under it, whose
    index rows are gradings' rows, so the warmer is listed from its plan."""
    store = tmp_path / "store"
    warmer = f"warmers/{TASK}/w-20261004T100000Z-0a0b0c"
    write_plan(store, warmer, ("provision", "images", "package"), outcome=SUCCEEDED, tmp=tmp_path)
    graded = f"{warmer}/gradings/{GRADING}"
    write_plan(store, graded, ("deploy_drive", "verify", "score", "finalize"), outcome=SUCCEEDED, tmp=tmp_path)
    publish(store, graded, "deploy_drive", {"drive.json": b"{}"}, trace=GRADING_TRACE)
    publish(store, graded, "finalize", {"grading.json": dumps(grading_record(GRADING, 0.5))})
    write_bytes(store, f"warmers/{TASK}/index/w-20261004T100000Z-0a0b0c+{GRADING}.json",
                json.dumps(index_row(grading=GRADING, rollout_id="w-20261004T100000Z-0a0b0c")).encode())
    scanner = RolloutScanner(store / "warmers")
    [row] = scanner.list_rollouts()
    assert (row["rollout_id"], row["status"], row["layout"]) == ("w-20261004T100000Z-0a0b0c", "succeeded", "store")
    rec = scanner.get_rollout(TASK, "w-20261004T100000Z-0a0b0c")
    assert [(e["eval_id"], e["score"]["value"]) for e in rec["evals"]] == [(GRADING, 0.5)]
    assert scanner.trace_summary(TASK, "w-20261004T100000Z-0a0b0c", GRADING).total == len(GRADING_TRACE.events)


def test_the_trajectory_of_a_store_build(tmp_path):
    store, scanner = _sealed(tmp_path)
    trajectory = {"schema_version": "ATIF-v1.2", "steps": [{"step_id": 1, "source": "user"}]}
    write_bytes(store, f"{rollout_key()}/stages/agent_build/attempts/1/outputs/transcript/trajectory.json",
                dumps(trajectory))
    assert scanner.get_trajectory(TASK, ROLLOUT) == trajectory


def test_a_store_builds_files_are_served_by_the_keys_of_its_outputs(tmp_path):
    """The page reads each file of a build by the store key its record names: the files endpoint
    takes the key without the instance's own key, which the record carries."""
    _, scanner = _sealed(tmp_path)
    rec = scanner.get_rollout(TASK, ROLLOUT)
    client = viewer_client(scanner.runs_dir)

    def served(key: str) -> bytes:
        assert key.startswith(f"{rec['instance']}/")
        response = client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/files/{key.removeprefix(rec['instance'] + '/')}")
        assert response.status_code == 200, key
        return response.content

    outputs = rec["build"]["outputs"]
    assert set(outputs) == set(BUILD_OUTPUTS)
    assert json.loads(served(outputs["brief_provenance"])) == BRIEF_PROVENANCE
    assert served(outputs["source_bundle"]) == BUNDLE
    # The transcript's copy of the agent's logs lies beside the index its output names.
    index_dir = outputs["transcript"].rsplit("/", 1)[0]
    assert served(f"{index_dir}/{COPY_DIR}/claude-code.txt") == AGENT_LOG_TEXT


def test_a_store_gradings_role_logs_are_served_by_the_keys_its_episode_names(tmp_path):
    """The page reads each role log of an eval by the store key the grading's episode names it by
    (``<role>.out``, ``<role>.err``), without the instance's own key."""
    _, scanner = _sealed(tmp_path)
    rec = scanner.get_rollout(TASK, ROLLOUT)
    client = viewer_client(scanner.runs_dir)
    logs = rec["evals"][0]["container_logs"]
    assert set(logs) == set(ROLE_LOGS)
    for name, key in logs.items():
        assert key.startswith(f"{rec['instance']}/")
        response = client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/files/{key.removeprefix(rec['instance'] + '/')}")
        assert (response.status_code, response.text) == (200, f"{name} text\n")


def _viewer_constants(module: Path = VIEWER_BUILD_MODULE) -> dict[str, str]:
    """The string constants the viewer module *module* declares, ``export const NAME = "value";``."""
    return dict(re.findall(r'^export const (\w+) = "([^"]*)";', module.read_text(), re.MULTILINE))


def _emitted(module: object) -> dict[str, set[str]]:
    """The fields each event of a literal kind that *module* logs is given, from its
    ``<log>.emit("kind", field=...)`` calls."""
    found: dict[str, set[str]] = {}
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "emit"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            found.setdefault(node.args[0].value, set()).update(k.arg for k in node.keywords if k.arg)
    return found


def test_the_pages_build_names_are_the_stages():
    """The page reads the transcript's copy of the agent's logs by ``TRANSCRIPT_LOGS`` and a running
    build by the events the agent_build stage logs; the names it holds are the stage's."""
    ts = _viewer_constants()
    assert ts["TRANSCRIPT_LOGS"] == COPY_DIR
    emitted = _emitted(agent_build)
    started, finished = ts["BUILD_STARTED"], ts["BUILD_FINISHED"]
    assert started.startswith(ts["BUILD_EVENT_PREFIX"]) and finished.startswith(ts["BUILD_EVENT_PREFIX"])
    assert {started, finished} <= emitted.keys()
    assert ts["BUILD_COST_FIELD"] in emitted[finished]


def test_the_pages_eval_names_are_the_drive_stages():
    """The page follows a running eval by the events the deploy_drive stage logs of its own, the one
    stage of the eval with events that mark its progress; the names it holds are the stage's."""
    ts = _viewer_constants(VIEWER_EVAL_MODULE)
    emitted = _emitted(deploy_drive)
    assert any(kind.startswith(ts["DRIVE_EVENT_PREFIX"]) for kind in emitted)
    assert {ts["ARTIFACT_READY"], ts["DRIVE_RESULT"]} <= emitted.keys()
    assert ts["DRIVE_OUTCOME_FIELD"] in emitted[ts["DRIVE_RESULT"]]


def test_the_pages_statuses_are_every_status_the_viewer_serves_a_record_with():
    """The page types a record's status as ``RolloutStatus``: a sealed record's own
    (``hls.obs.record.RolloutStatus``), a published run never sealed
    (``UNSEALED``), a store instance that ended without a record (``INSTANCE_ENDINGS``), and one
    no orchestrator holds (``STALLED``); a store instance an orchestrator holds is "running", a
    record's own."""
    declared = re.search(r"^export type RolloutStatus =((?:\s*\|\s*\"[^\"]*\")+);",
                         VIEWER_TYPES_MODULE.read_text(), re.MULTILINE)
    assert declared is not None
    page = set(re.findall(r'"([^"]*)"', declared.group(1)))
    served = (set(get_args(RolloutStatus)) | {scanner_mod.UNSEALED, scanner_mod.STALLED}
              | scanner_mod.INSTANCE_ENDINGS)
    assert page == served
    assert "running" in get_args(RolloutStatus)


def test_the_pages_agent_log_is_the_file_harbors_agent_writes_its_stream_to():
    """The page names the agent's log in the copy by the arm's harness and ``AGENT_LOG_SUFFIX``."""
    from harbor.agents.installed.claude_code import STREAM_PATH
    from harbor.agents.installed.codex import Codex

    suffix = _viewer_constants()["AGENT_LOG_SUFFIX"]
    written = {"claude-code": Path(STREAM_PATH).name, "codex": Codex._OUTPUT_FILENAME}
    assert set(written) == set(ADAPTERS)
    assert written == {harness: f"{harness}{suffix}" for harness in written}


def test_the_config_of_a_store_rollout_is_its_records_provenance(tmp_path):
    _, scanner = _sealed(tmp_path)
    assert scanner.get_config(TASK, ROLLOUT)["spec"] == rollout_record()["config"]["spec"]
    client = viewer_client(scanner.runs_dir)
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/config").json()["reasoning_effort"] == "max"


def test_a_published_rollout_has_no_event_stream(tmp_path):
    """Nothing appends to a published rollout's log: its events are among its files,
    and the event stream is a store instance's alone."""
    published_runs(tmp_path, TRACE)
    client = viewer_client(tmp_path)
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/events").status_code == 404
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/files/events.jsonl").status_code == 200


def test_the_store_events_stream_in_the_page_shape_until_the_instance_ended(tmp_path):
    store, scanner = _sealed(tmp_path)
    key = rollout_key()
    append_events(store, f"{key}/orchestrator/gen-00000001/events.jsonl",
                  [{"at": CREATED + 1, "kind": "instance_created", "revision": 1},
                   {"at": CREATED + 9, "kind": "instance_ended", "status": "succeeded"}])
    append_events(store, f"{key}/stages/verify/attempts/1/events.jsonl",
                  [{"at": CREATED + 5, "kind": "role_started", "role": "verify"}])
    body = viewer_client(scanner.runs_dir).get(f"/api/rollouts/{TASK}/{ROLLOUT}/events").text
    sent = [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]
    assert [(e["type"], e["ts"], e["data"]["source"]) for e in sent[:-1]] == [
        ("instance_created", CREATED + 1, "orchestrator gen 1"),
        ("role_started", CREATED + 5, "stage verify attempt 1"),
        ("instance_ended", CREATED + 9, "orchestrator gen 1")]
    assert sent[1]["data"]["role"] == "verify"
    # The page ends its live follow on the old log's last event, which the stream sends last.
    assert (sent[-1]["type"], sent[-1]["data"]) == ("rollout.finished", {"source": "viewer"})


def test_a_store_event_names_its_line_the_same_on_every_connection(tmp_path):
    """Each streamed event carries where its line begins in its log (``offset``), which with its log
    (``data.source``) names it on every connection: a later connection restarts every log and
    sends a line written since by its time, before lines already sent, and the page keeps each
    line once by that name (``use-rollout-events.ts``). The closing event names no line."""
    store, scanner = _sealed(tmp_path)
    key = rollout_key()
    created = {"at": CREATED + 1, "kind": "instance_created", "revision": 1}
    append_events(store, f"{key}/orchestrator/gen-00000001/events.jsonl",
                  [created, {"at": CREATED + 9, "kind": "instance_ended", "status": "succeeded"}])
    client = viewer_client(scanner.runs_dir)

    def connect() -> list[dict]:
        body = client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/events").text
        return [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]

    first = connect()
    assert [(e["data"]["source"], e["offset"]) for e in first[:-1]] == [
        ("orchestrator gen 1", 0), ("orchestrator gen 1", len(json.dumps(created, sort_keys=True)) + 1)]
    assert "offset" not in first[-1]
    append_events(store, f"{key}/stages/verify/attempts/1/events.jsonl",
                  [{"at": CREATED + 5, "kind": "role_started", "role": "verify"}])
    second = connect()
    names = [(e["data"]["source"], e["offset"]) for e in second[:-1]]
    assert names == [("orchestrator gen 1", 0), ("stage verify attempt 1", 0),
                     ("orchestrator gen 1", first[1]["offset"])]
    assert [e for e in second[:-1] if (e["data"]["source"], e["offset"]) not in
            {(f["data"]["source"], f["offset"]) for f in first[:-1]}] == [second[1]]


def test_every_ending_counts_as_done():
    """The rollout list's stream calls a status done when the record is sealed or the instance
    ended without one; a running or stalled instance is not done."""
    assert server_mod._DONE_STATUSES >= {"ok", "invalid_artifact", "void", "cancelled", "succeeded", "failed",
                                         "unsealed"}
    assert not {"running", "stalled"} & server_mod._DONE_STATUSES


def test_the_viewers_store_names_are_the_writers():
    from hls.score.role import SCORE_FILE
    from hls.stages import finalize
    from hls.stages.agent_build import TRANSCRIPT_DIR
    from hls.stages.score import VERDICT_FILE

    ours = (scanner_mod.ROLLOUT_FILE, scanner_mod.GRADING_FILE, scanner_mod.FAILED_DIR)
    assert ours == (finalize.ROLLOUT_FILE, finalize.GRADING_FILE, finalize.FAILED_DIR)
    # The row lies in the task's index directory, which the viewer reads (the layout's INDEX_DIR).
    index, _, _ = finalize.index_key(rollout_key(), "rollout")
    assert index == f"runs/{TASK}/{INDEX_DIR}/{ROLLOUT}.json"
    ours = (scanner_mod.VERDICT_FILE, scanner_mod.SCORE_FILE)
    assert ours == (VERDICT_FILE, SCORE_FILE)
    assert scanner_mod.TRANSCRIPT_DIR == TRANSCRIPT_DIR
    from hls.verify.answers import ANSWERS_AREA

    assert (scanner_mod.QUARANTINE_FILE, scanner_mod.ANSWERS_AREA) == (finalize.QUARANTINE_FILE, ANSWERS_AREA)
    # The eval id of an instance's own grading is the one the rollout pipeline gives it.
    from hls.pipeline.pipelines import ROLLOUT_EVAL

    assert scanner_mod.TERMINAL_EVAL == ROLLOUT_EVAL


# Guards that keep the scanner inside the instance it shows.


def test_an_output_that_links_out_of_its_instance_is_not_read(tmp_path):
    """The manifest names the output at its attempt's own key, but the file there is a link out
    of the instance."""
    store, scanner = _sealed(tmp_path)
    outside = write_bytes(store, "elsewhere/grading.json", dumps(grading_record("terminal", 9.9)))
    record = store / rollout_key() / "stages/finalize/attempts/1/outputs/grading.json"
    record.unlink()
    record.symlink_to(outside)
    assert [e["eval_id"] for e in scanner.get_rollout(TASK, ROLLOUT)["evals"]] == [GRADING]


def test_an_output_its_manifest_does_not_name_is_not_read(tmp_path):
    store = tmp_path / "store"
    key = rollout_key()
    write_plan(store, key, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    publish(store, key, "finalize", {"rollout.json": dumps(rollout_record())})
    # Written where the attempt keeps its outputs, but not among the outputs it published.
    write_bytes(store, f"{key}/stages/finalize/attempts/1/outputs/grading.json",
                dumps(grading_record("terminal", 9.9)))
    assert RolloutScanner(store / "runs").get_rollout(TASK, ROLLOUT)["evals"] == []


def test_a_grading_directory_that_links_out_of_the_rollout_is_not_read(tmp_path):
    store, scanner = _sealed(tmp_path)
    other = f"runs/{TASK}/20261004T130000Z-ffffff/gradings/g-20261005T090000Z-999999"
    write_plan(store, other, GRADING_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    publish(store, other, "deploy_drive", {"drive.json": b"{}"}, trace=TRACE)
    publish(store, other, "finalize", {"grading.json": dumps(grading_record("g-20261005T090000Z-999999", 9.9))})
    (store / rollout_key() / "gradings" / "g-20261005T090000Z-999999").symlink_to(store / other)
    assert [e["eval_id"] for e in scanner.get_rollout(TASK, ROLLOUT)["evals"]] == ["terminal", GRADING]
    assert scanner.trace_summary(TASK, ROLLOUT, "g-20261005T090000Z-999999") is None


def test_a_trajectory_that_links_out_of_its_instance_is_not_read(tmp_path):
    store, scanner = _sealed(tmp_path)
    outside = write_bytes(store, "elsewhere/trajectory.json", dumps({"steps": [{"step_id": 1}]}))
    (store / rollout_key() / "stages/agent_build/attempts/1/outputs/transcript/trajectory.json").symlink_to(outside)
    assert scanner.get_trajectory(TASK, ROLLOUT) is None


def _with_published_layout_files(store: Path, tmp_path: Path) -> None:
    """Files of the published layout, planted where that layout keeps them in the directory of the
    store instance: its ``rollout.json``, ``config.json`` and ``events.jsonl``, a graded eval's
    ``grading.json``, ``trace.json`` and ``verdict.json``, and the build's
    ``build/trajectory.atif.json``."""
    published_runs(tmp_path / "published", TRACE)
    rdir = store / rollout_key()
    shutil.copytree(tmp_path / "published" / TASK / ROLLOUT, rdir, dirs_exist_ok=True)
    # The published eval directory of a grading, too.
    shutil.copytree(rdir / "evals" / "terminal", rdir / "evals" / GRADING)
    write_bytes(store, f"{rollout_key()}/config.json", dumps({"task_id": TASK, "spec": SPEC}))
    write_bytes(store, f"{rollout_key()}/build/trajectory.atif.json", dumps({"steps": [{"step_id": 1}]}))


def _nothing_of_the_published_layout_is_read(store: Path) -> None:
    """The store instance shows only what its own records hold, which here is nothing: no record,
    config, trajectory or trace, and no row with the published record's score."""
    scanner = RolloutScanner(store / "runs")
    assert scanner.get_trajectory(TASK, ROLLOUT) is None
    assert scanner.get_config(TASK, ROLLOUT) is None
    rec = scanner.get_rollout(TASK, ROLLOUT)
    assert (rec["layout"], rec["status"], rec["config"], rec["build"], rec["evals"]) == (
        "store", "succeeded", None, None, [])
    for eval_id in ("terminal", GRADING):
        assert scanner.trace_summary(TASK, ROLLOUT, eval_id) is None
        assert scanner.trace_page(TASK, ROLLOUT, eval_id).total == 0
        assert scanner.timeline(TASK, ROLLOUT, eval_id) is None
    [row] = scanner.list_rollouts()
    assert (row["layout"], row["score"]) == ("store", None)
    client = viewer_client(store / "runs")
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/trajectory").status_code == 404
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/config").status_code == 404
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/evals/terminal/trace/summary").json()["total"] == 0


def test_a_store_instance_with_no_stage_outputs_never_falls_back_to_the_published_layouts_files(tmp_path):
    """The published layout stays viewable, but only as a published rollout: a store instance that
    holds no record, trajectory, config or trace is shown without them, though files of the
    published layout lie where that layout keeps them."""
    store = tmp_path / "store"
    write_plan(store, rollout_key(), ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_lease(store, rollout_key(), heartbeat_at=CREATED + 900, released=True)
    _with_published_layout_files(store, tmp_path)
    _nothing_of_the_published_layout_is_read(store)


def test_a_store_build_with_no_transcript_output_has_no_trajectory_whatever_the_published_layout_holds(tmp_path):
    """The build published, but its manifest names no transcript: the trajectory is absent, and the
    ``build/trajectory.atif.json`` a published rollout keeps its trajectory in is not read for it."""
    store = tmp_path / "store"
    write_plan(store, rollout_key(), ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_lease(store, rollout_key(), heartbeat_at=CREATED + 900, released=True)
    publish(store, rollout_key(), "agent_build", {"build.json": b"{}"})
    _with_published_layout_files(store, tmp_path)
    _nothing_of_the_published_layout_is_read(store)


def test_a_store_instance_whose_plan_cannot_be_read_is_not_shown_from_the_published_layouts_files(tmp_path):
    """The directory holds ``plan/``, so it is a store instance whatever its newest revision holds:
    with that unreadable the viewer has nothing to show of it, and does not take a record, config,
    trajectory or trace from the files of the published layout beside it."""
    store = tmp_path / "store"
    write_plan(store, rollout_key(), ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    (store / InstanceLayout(rollout_key()).plan_revision(1)).write_text("not a plan")
    _with_published_layout_files(store, tmp_path)
    scanner = RolloutScanner(store / "runs")
    assert scanner.get_rollout(TASK, ROLLOUT) is None
    assert scanner.get_config(TASK, ROLLOUT) is None
    assert scanner.get_trajectory(TASK, ROLLOUT) is None
    for eval_id in ("terminal", GRADING):
        assert scanner.trace_summary(TASK, ROLLOUT, eval_id) is None
        assert scanner.trace_page(TASK, ROLLOUT, eval_id).total == 0
    assert scanner.list_rollouts() == []


def test_a_published_rollouts_trajectory_is_still_read_from_its_own_layout(tmp_path):
    """The other side of the rule: a published rollout, which holds no ``plan/``, is shown from the
    files of its layout."""
    published_runs(tmp_path, TRACE)
    unsealed = RolloutScanner(tmp_path)
    assert unsealed.get_trajectory(TASK, "20260901T000000Z-killed") == {"steps": [{"step_id": 1}]}


def test_an_instance_an_orchestrator_still_holds_has_not_ended(tmp_path):
    """The outcome is written before the orchestrator writes its last events and lets go, so
    the instance has ended only once the lease is let go."""
    store = tmp_path / "store"
    key = rollout_key()
    write_plan(store, key, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_lease(store, key, heartbeat_at=time.time())
    scanner = RolloutScanner(store / "runs")
    assert scanner.instance_ended(TASK, ROLLOUT) is False
    write_lease(store, key, heartbeat_at=time.time(), released=True)
    assert scanner.instance_ended(TASK, ROLLOUT) is True


def test_a_rollout_directory_that_links_out_of_the_root_is_refused(tmp_path):
    root = tmp_path / "root"
    published_runs(tmp_path / "elsewhere", TRACE)
    (root / TASK).mkdir(parents=True)
    (root / TASK / "linked").symlink_to(tmp_path / "elsewhere" / TASK / ROLLOUT)
    with pytest.raises(ValueError, match="traversal"):
        RolloutScanner(root).get_rollout(TASK, "linked")
    assert viewer_client(root).get(f"/api/rollouts/{TASK}/linked/files/rollout.json").status_code == 400
