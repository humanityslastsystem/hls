from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from hls.core.poly import dump_spec
from hls.core.verdict import Verdict
from hls.obs.record import GradingRecord, RolloutRecord, index_columns
from hls.pipeline.events import read_events
from hls.pipeline.pipelines import FINALIZE_SHAPE
from hls.recipes import catalog
from hls.pipeline.planner import StageDecl
from hls.pipeline.stage import StageFault
from hls.spec.resources import ResourceShape
from hls.stages.deploy_drive import DeployDriveStage
from hls.stages.finalize import (
    FINALIZE_SEC,
    OUTSIDE_BUNDLE,
    SCAN,
    SHOWN_HITS,
    SHOWN_QUARANTINED,
    FinalizeParams,
    FinalizeStage,
    contestant_facing,
    index_key,
)
from hls.stages.images import KEPT_DISK_GIB, ImagesParams, ImagesStage
from hls.stages.noop import NoopParams, NoopStage
from hls.stages.oracle_fill import OracleFillStage
from hls.stages.package import SUBMISSION_FILES, PackageStage
from hls.stages.provision import ProvisionStage
from hls.stages.score import ScoreStage
from hls.stages.source_scan import SourceScanStage
from hls.stages.verify import VerifyStage
from hls.substrate.secret_scan import (
    CHUNK_BYTES,
    PACKAGE_SCAN_FILE,
    PackageScan,
    PackageScanArea,
    PackageScanFinding,
)
from hls.util import json_safe
from hls.verify.role import EngineRecord, VerifyRecord
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.secret_forms import FORMS
from hls.substrate.secrets import Scrubber
from hls.measured import BASELINE_CREDIT_FILE, defaults_path
from hls.taskref import MeasuredPaths, TaskRef, load_task
from tests.finalize.fixture import t0_trace
from tests.finalize.published import (CPU, CPU_ROLE, GRADING, INSTALL_SCAN, LOGS, NOT_SERVED, ROLLOUT, RUNNER_GIB,
                                      SIF_SHA, TREE_SCAN, context, echoed_trace, make_plan, package_result, publish,
                                      publish_instance, publish_rollout_package, run_and_publish, with_inputs,
                                      with_params, write)
from tests.finalize.stand_in_stage import AgentStandInStage  # importing it registers the stand-ins
from tests.pipeline.fakes import FakeClock, LocalLauncher
from tests.trace.traces import LATENCY_TENANTS, drive_trace, passing_verdict

SECRET = "hf-token-0123456789abcdef"
SECRETS = {"hf_token": SECRET}
#: The fields of an ``index.jsonl`` row a row keeps: all but
#: ``backend``, which a checkpoint that is a SIF digest leaves always null.
OLD_ROW = {"rollout_id", "grading_id", "mode", "created_at", "finished_at", "status", "score", "gate_passed",
           "cost_usd", "image_digests", "model_fingerprint", "accelerator", "agent_implementer_model"}
#: The column a row adds for the arm whose agent built the graded submission (None for one that no
#: agent built).
ARM_COLUMNS = {"arm"}
#: The columns a row adds for an instance a stage's contestant fault ended (None otherwise).
FAILED_COLUMNS = {"failed_stage", "fault"}
#: The columns a row adds for the secret scan: whether the instance's files were scanned, and
#: the contestant's files a value was found in, which are kept out of any export.
SCAN_COLUMNS = {"scanned", "quarantined", "quarantined_count"}
#: The fixture rollout's index row.
ROW = f"runs/fin/index/{ROLLOUT}.json"


def _instance(tmp_path, checkout, module="fin_goodput.task", *, outcome="served", trace=None, extra=None,
              verdict=None, drive_inputs=None, **plan_args):
    """A published fixture instance up to its score, with *extra* files in its images
    stage's outputs, and its plan, whose drive pins *drive_inputs*."""
    plan = with_inputs(make_plan(tmp_path, checkout, module, **plan_args), "deploy_drive", drive_inputs or {})
    store = PosixStore(plan.store_root)
    if outcome == "served":
        trace = trace or drive_trace()
        verdict = verdict or passing_verdict(trace)
    publish_instance(store, plan, outcome=outcome, trace=trace, verdict=verdict, extra=extra,
                     drive_inputs=drive_inputs)
    run_and_publish(store, plan, "score", ScoreStage)
    return plan, store


def _finalize(plan, store, *, attempt=1, secrets=SECRETS):
    ctx = context(store, plan, "finalize", attempt=attempt, secrets=secrets)
    return ctx, FinalizeStage().run(ctx)


def _strict(data: bytes):
    return json.loads(data, parse_constant=lambda c: pytest.fail(f"a record holds {c}"))


def test_a_rollout_is_recorded_in_the_old_shapes(tmp_path, checkout):
    trace = drive_trace(failures=False, tenants=LATENCY_TENANTS)
    plan, store = _instance(tmp_path, checkout, "fin_latency.task", trace=trace)
    ctx, outputs = _finalize(plan, store)
    layout = plan.layout
    # Every output is the attempt's own, as the runner requires; the index row is not one.
    assert outputs == {"grading": ctx.output_key("grading.json"), "rollout": ctx.output_key("rollout.json")}
    score = json.loads(store.read_bytes(layout.attempt_file("score", 1, "outputs/score.json")))
    verdict = json.loads(store.read_bytes(layout.attempt_file("verify", 1, "outputs/verdict.json")))
    drive, _ = read_manifest(store, layout, "deploy_drive")

    grading = GradingRecord.model_validate_json(store.read_bytes(outputs["grading"]))
    written = json.loads(store.read_bytes(outputs["grading"]))
    # The episode and the provenance hold the keys the stage pipeline sets, and none it never sets.
    assert set(written["episode"]) == {"eval_id", "triggered_by", "checkpoint", "started_at", "finished_at",
                                       "trace_path", "verdict_summary", "score", "container_logs"}
    assert set(written["provenance"]) == {"task_id", "split", "spec", "image_digests", "hls_commit",
                                          "harbor_commit", "rollout_overrides", "artifact", "reasoning_effort"}
    ep = grading.episode
    assert (ep.eval_id, ep.triggered_by, ep.checkpoint) == ("terminal", "harness", SIF_SHA)
    assert (ep.started_at, ep.finished_at) == (100.0, 260.0)
    assert ep.trace_path == layout.attempt_file("deploy_drive", 1, "outputs/trace")
    assert drive.outputs["trace"].key == f"{ep.trace_path}/END.json"
    assert ep.verdict_summary == {"gate_passed": score["gate_passed"], "aggregate_passed": True,
                                  "diagnostics": verdict["diagnostics"]}
    assert ep.score == score and score["value"] > 0
    logs = {f"{stage}:{name}": layout.attempt_file(stage, 1, f"logs/{name}")
            for stage, names in {**LOGS, "score": ("score.err", "score.out")}.items() for name in names}
    assert ep.container_logs == {k.split(":")[1]: v for k, v in logs.items()}
    assert (grading.source_rollout, grading.mode) == (ROLLOUT, "original")

    prov = grading.provenance
    ref = TaskRef(module="fin_latency.task")
    task = load_task(ref, calibration=True, credit=True,
                     files=MeasuredPaths(credit=defaults_path(ref, BASELINE_CREDIT_FILE)))
    assert prov.spec == json.loads(json.dumps(json_safe(dump_spec(task))))
    assert prov.spec["scorer"]["baseline_log_credit"] == pytest.approx(1.0)
    assert prov.image_digests == {"sa-base": "1" * 64, "submission": SIF_SHA, "artifact": SIF_SHA,
                                  "driver": "2" * 64, "verify": "3" * 64}
    assert (prov.task_id, prov.split, prov.hls_commit, prov.harbor_commit) == ("fin", "test", "a" * 40, "b" * 40)
    assert prov.artifact == {"kind": "package", "stage": "package", "manifest": None}

    rollout = RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"]))
    assert (rollout.rollout_id, rollout.task_id, rollout.created_at) == (ROLLOUT, "fin", 1000.0)
    assert rollout.status == "ok"
    assert (rollout.build, rollout.error, rollout.config) == (None, None, prov)
    # Both records carry the same harness conditions; no agent built this fixture's artifact.
    assert rollout.conditions == grading.conditions and rollout.conditions.arm is None
    assert rollout.conditions.attempts["deploy_drive"].counted == 1

    row = _strict(store.read_bytes(ROW))
    assert set(row) == OLD_ROW | FAILED_COLUMNS | SCAN_COLUMNS | ARM_COLUMNS
    assert row == {"rollout_id": ROLLOUT, "grading_id": None, "mode": "original", "created_at": 1000.0,
                   "finished_at": rollout.finished_at, "status": "ok", "score": score["value"],
                   "gate_passed": score["gate_passed"], "cost_usd": None, "image_digests": prov.image_digests,
                   "failed_stage": None, "fault": None, "scanned": True, "quarantined": [], "quarantined_count": 0,
                   "arm": None, **index_columns(prov.spec, None)}
    assert row["accelerator"] == "H200"


@pytest.mark.parametrize("outcome", ["invalid_artifact", "crashed", "not_ready"])
def test_a_drive_that_did_not_serve_is_recorded_with_its_zero(tmp_path, checkout, outcome):
    plan, store = _instance(tmp_path, checkout, outcome=outcome)
    _, outputs = _finalize(plan, store)
    ep = GradingRecord.model_validate_json(store.read_bytes(outputs["grading"])).episode
    assert ep.checkpoint == ("" if outcome == "invalid_artifact" else SIF_SHA) and ep.trace_path == ""
    assert ep.score["zero_reasons"] == [outcome] and ep.verdict_summary["gate_passed"] is False
    assert ep.verdict_summary["diagnostics"] == {"reason": NOT_SERVED[outcome]}
    status = "invalid_artifact" if outcome == "invalid_artifact" else "ok"
    assert RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"])).status == status
    assert json.loads(store.read_bytes(ROW))["status"] == status


@pytest.mark.parametrize("artifact, mode", [
    ({"kind": "package"}, "rebuild"),
    ({"kind": "package", "manifest": f"runs/fin/{ROLLOUT}/stages/package/MANIFEST.json"}, "adopt"),
    ({"kind": "image", "image": "measured-0123", "entrypoint": ["serve"]}, "adopt"),
    ({"kind": "published", "sha256": "cd" * 32, "source": "/shared/submission.sif"}, "adopt"),
])
def test_a_grading_is_recorded_under_its_id_with_how_it_got_its_artifact(tmp_path, checkout, artifact, mode):
    pins = None
    if "manifest" in artifact:
        pins = publish_rollout_package(PosixStore(tmp_path / "store"), make_plan(tmp_path, checkout,
                                                                                 "fin_goodput.task", kind="grading"))
    plan, store = _instance(tmp_path, checkout, kind="grading", artifact=artifact, drive_inputs=pins)
    _, outputs = _finalize(plan, store)
    key = f"runs/fin/index/{ROLLOUT}+{GRADING}.json"
    assert sorted(outputs) == ["grading"] and index_key(plan.instance, "grading")[0] == key
    grading = GradingRecord.model_validate_json(store.read_bytes(outputs["grading"]))
    assert (grading.episode.eval_id, grading.episode.triggered_by) == (GRADING, "regrade")
    assert (grading.source_rollout, grading.mode) == (ROLLOUT, mode)
    row = json.loads(store.read_bytes(key))
    assert (row["rollout_id"], row["grading_id"], row["mode"]) == (ROLLOUT, GRADING, mode)


def test_a_grading_whose_eval_is_not_its_grading_id_is_refused(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout, kind="grading")
    plan = with_params(plan, "finalize", eval_id="terminal")
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.retryable, "named by its grading id" in e.value.fault.reason) == (False, True)


@pytest.mark.parametrize("kind", ["rollout", "grading"])
def test_an_instance_that_is_not_of_its_kind_is_refused(tmp_path, checkout, kind):
    plan, store = _instance(tmp_path, checkout, kind=kind)
    plan = with_params(plan, "finalize", kind="grading" if kind == "rollout" else "rollout", eval_id="terminal")
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.retryable, "instance under runs/" in e.value.fault.reason) == (False, True)


def test_every_text_field_is_scrubbed_when_it_is_written(tmp_path, checkout):
    # The value reaches the records through the task's overrides, which no published file holds.
    plan, store = _instance(tmp_path, checkout, overrides={"task.model.base_checkpoint": SECRET})
    _, outputs = _finalize(plan, store)
    for key in [*outputs.values(), ROW]:
        data = store.read_bytes(key)
        assert SECRET.encode() not in data and b"<redacted:hf_token>" in data
    rollout = RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"]))
    assert rollout.config.spec["model"]["base_checkpoint"] == "<redacted:hf_token>"
    assert rollout.config.rollout_overrides == {"task.model.base_checkpoint": "<redacted:hf_token>"}


def test_a_task_with_an_infinite_budget_has_strict_json_records(tmp_path, checkout):
    # Every number a record holds has been through JSON already (score.json, verdict.json,
    # the spec's JSON-mode dump), each writing a non-finite one as null as json_safe does.
    trace = t0_trace(failures=False)
    plan, store = _instance(tmp_path, checkout, "fin_vacuous.task", trace=trace)
    _, outputs = _finalize(plan, store)
    for key in [*outputs.values(), ROW]:
        _strict(store.read_bytes(key))
    spec = _strict(store.read_bytes(outputs["rollout"]))["config"]["spec"]
    assert spec["tenants"][0]["workload"]["slo"]["tpot_ms"] is None


def test_the_index_row_is_one_create_only_file_that_a_second_finalize_refuses(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    _finalize(plan, store)
    row = store.read_bytes(ROW)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, attempt=2)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert f"the index row {ROW} already exists" in fault.reason
    assert store.read_bytes(ROW) == row


PLANTED = {
    "a gzip member": ("blob.jsonl.gz", gzip.compress(b"x" * 1000 + SECRET.encode() + b"y" * 1000)),
    "a gzip file of another name": ("blob.bin", gzip.compress(b'{"token": "' + SECRET.encode() + b'"}')),
    "a read boundary": ("big.bin", b"a" * (CHUNK_BYTES - 5) + SECRET.encode() + b"b" * 100),
    "a read boundary in a gzip stream": ("big.gz", gzip.compress(b"a" * (CHUNK_BYTES - 5) + SECRET.encode())),
}


@pytest.mark.parametrize("where", sorted(PLANTED))
def test_a_secret_in_any_published_file_refuses_the_instance(tmp_path, checkout, where):
    name, data = PLANTED[where]
    plan, store = _instance(tmp_path, checkout, extra={name: data})
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    file = plan.layout.attempt_file("images", 1, f"outputs/{name}")
    assert fault.reason == f"the harness leaked secret values into published files: {file} (hf_token)"
    logs = context(store, plan, "finalize").attempt_dir / "logs"
    assert all(SECRET not in p.read_text() for p in logs.iterdir() if p.suffix in (".out", ".err"))
    assert not store.exists(f"runs/fin/index/{ROLLOUT}.json")


#: A secret the plan grants to a stage that hands its grants to the contestant it runs: one a
#: contestant could hold (``contestant_facing``), as an agent build's route credentials are.
RELAY = "relay-key-0123456789abcdef"
#: The registry with both kinds of secret.
BOTH = {"hf_token": SECRET, "relay": RELAY}


def _granted(plan, stage: str):
    """*plan* with ``relay`` granted to its stage *stage*."""
    return plan.model_copy(update={"stages": [s.model_copy(update={"grants": {"RELAY_KEY": "relay"}})
                                              if s.name == stage else s for s in plan.stages]})


def _facing(plan):
    """*plan* with a stage first that hands its grants to its contestant, as the agent build
    does, and ``relay`` granted to it."""
    agent = plan.stages[0].model_copy(update={"name": "agent", "kind": AgentStandInStage.kind, "params": {},
                                              "after": [], "inputs": {}})
    return _granted(plan.model_copy(update={"stages": [agent, *plan.stages]}), "agent")


def _quarantined(store, outputs) -> dict:
    return json.loads(store.read_bytes(outputs["quarantine"]))


def test_only_a_secret_granted_to_a_stage_that_hands_its_grants_to_its_contestant_can_be_a_contestants():
    plan = _with_package(make_plan(Path("/x"), Path("/c"), "fin_goodput.task"))
    assert contestant_facing(plan, BOTH) == frozenset()
    assert contestant_facing(_facing(plan), BOTH) == {"relay"}
    # A stage's grants reach its harness roles only, never a container, even in a stage that runs
    # a contestant's code; and a stage that runs none hands its grants to no contestant.
    for stage in ("package", "deploy_drive", "score"):
        assert contestant_facing(_granted(plan, stage), BOTH) == frozenset()
    # A name granted to the contestant stays the contestant's when the registry does not hold it:
    # a record written while the registry did names it.
    assert contestant_facing(_facing(plan), {"hf_token": SECRET}) == {"relay"}


@pytest.mark.parametrize("stage", ["package", "deploy_drive"])
def test_a_key_granted_only_to_a_stage_that_runs_contestant_code_refuses_the_instance(tmp_path, checkout, stage):
    # The stage's harness roles held it, and a contestant role's log is no place for it.
    plan, store = _instance(tmp_path, checkout)
    plan = _granted(_with_package(plan), stage)
    key = plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.err")
    store.write_bytes(key, RELAY.encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {key} (relay)"
    assert not store.exists(ROW)


def test_a_key_a_server_echoed_into_a_reply_quarantines_the_trace_and_the_instance_is_scored(tmp_path,
                                                                                            checkout):
    plan, store = _instance(tmp_path, checkout, trace=echoed_trace(RELAY))
    plan = _facing(plan)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert _quarantined(store, outputs) == {"files": {chunk: ["relay"]}}
    assert sorted(outputs) == ["grading", "quarantine", "rollout"]
    row = _strict(store.read_bytes(ROW))
    score = json.loads(store.read_bytes(plan.layout.attempt_file("score", 1, "outputs/score.json")))
    assert (row["scanned"], row["quarantined"], row["quarantined_count"], row["status"]) == (True, [chunk], 1, "ok")
    assert row["score"] == score["value"]
    assert score["value"] > 0


#: One relay URL that two routes' secrets hold, as one relay can serve two routes, and the
#: registry that names it twice.
SHARED = "https://relay.invalid/v1/" + "x" * 16
SHARED_TWICE = {"route_a_url": SHARED, "route_b_url": SHARED, "hf_token": SECRET}


def _facing_route_a(plan):
    """*plan* with the agent stand-in first, granted ``route_a_url`` alone."""
    agent = _facing(plan).stages[0].model_copy(update={"grants": {"ROUTE_URL": "route_a_url"}})
    return plan.model_copy(update={"stages": [agent, *plan.stages]})


def test_a_secret_is_a_contestants_when_a_secret_granted_to_its_contestant_holds_the_same_value():
    plan = make_plan(Path("/x"), Path("/c"), "fin_goodput.task")
    # A scan finds the bytes, and cannot tell which of two names with one value put them there.
    assert contestant_facing(_facing_route_a(plan), SHARED_TWICE) == {"route_a_url", "route_b_url"}
    # Two values are judged each by its own name.
    assert contestant_facing(_facing(plan), BOTH) == {"relay"}
    assert contestant_facing(_facing(plan), {**BOTH, "route_b_url": RELAY}) == {"relay", "route_b_url"}
    # A value no granted secret holds is no contestant's, however many names hold it.
    assert contestant_facing(plan, SHARED_TWICE) == frozenset()


def test_a_value_two_names_hold_is_quarantined_when_one_of_them_was_granted_to_the_contestant(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout, trace=echoed_trace(SHARED))
    _, outputs = _finalize(_facing_route_a(plan), store, secrets=SHARED_TWICE)
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    image = plan.layout.attempt_file("package", 1, "outputs/submission.sif")
    # The file holds the value of both names, the URL's host with it, and the record says so. The
    # package's scan searched neither name, so its image is kept out unscanned.
    assert _quarantined(store, outputs) == {"files": {chunk: ["route_a_url", "route_a_url.host", "route_b_url",
                                                              "route_b_url.host"], image: []}}
    assert _strict(store.read_bytes(ROW))["status"] == "ok"


def test_a_value_only_secrets_no_contestant_holds_share_still_refuses_the_instance(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout, trace=echoed_trace(SHARED))
    with pytest.raises(StageFault) as e:
        _finalize(_facing(plan), store, secrets={**SHARED_TWICE, "relay": RELAY})
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == ("the harness leaked secret values into published files: "
                                    f"{chunk} (route_a_url), {chunk} (route_b_url)")
    assert SHARED not in e.value.fault.reason
    assert not store.exists(ROW)


def test_a_secret_no_contestant_holds_refuses_the_instance_even_in_the_trace(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout, trace=echoed_trace(SECRET))
    plan = _facing(plan)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {chunk} (hf_token)"
    assert not store.exists(ROW)


@pytest.mark.parametrize("log", ["artifact.out", "artifact.err"])
def test_a_key_in_a_contestant_roles_log_quarantines_it(tmp_path, checkout, log):
    plan, store = _instance(tmp_path, checkout)
    key = plan.layout.attempt_file("deploy_drive", 1, f"logs/{log}")
    store.write_bytes(key, b"\x1f\x8b" + RELAY.encode())
    _, outputs = _finalize(_facing(plan), store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {key: ["relay"]}}
    assert _strict(store.read_bytes(ROW))["quarantined"] == [key]


def _with_package(plan):
    """*plan* naming the package stage the fixture published, as an evaluation's plan does."""
    package = plan.stage("score").model_copy(update={"name": "package", "kind": PackageStage.kind, "params": {}})
    return plan.model_copy(update={"stages": [package, *plan.stages]})


@pytest.mark.parametrize("image_file", SUBMISSION_FILES)
def test_the_submission_image_is_read_as_stored_on_every_substrate(tmp_path, checkout, image_file):
    """An image whose bytes are a gzip stream is read as stored, never decompressed, under its name
    on each substrate: the key the stream hides is not found by finalize's scan, and the package's
    scans, which searched every secret, read it."""
    plan, store = _scanned(tmp_path, checkout, image=image_file)
    plan = _with_package(plan)
    _, _, image, _ = _keys(plan, image_file)
    store.write_bytes(image, gzip.compress(RELAY.encode(), mtime=0))
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert "quarantine" not in outputs


@pytest.mark.parametrize("image_file", SUBMISSION_FILES)
def test_a_key_in_the_submission_image_quarantines_it(tmp_path, checkout, image_file):
    plan, store = _instance(tmp_path, checkout)
    plan = _facing(_with_package(plan))
    sif = plan.layout.attempt_file("package", 1, f"outputs/{image_file}")
    store.write_bytes(sif, b"#!/usr/bin/env run-singularity\n" + RELAY.encode())
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {sif: ["relay"]}}


@pytest.mark.parametrize("output", ["package.json", "install-report.json", "source.json"])
def test_a_key_in_a_package_record_quarantines_it(tmp_path, checkout, output):
    plan, store = _instance(tmp_path, checkout)
    plan = _facing(_with_package(plan))
    key = plan.layout.attempt_file("package", 1, f"outputs/{output}")
    store.write_bytes(key, json.dumps({"log_tail": f"setup.py: token {RELAY}"}).encode())
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {key: ["relay"]}}


#: Files the harness wrote in attempts of the stages that run a contestant's code: the package's
#: build, wheel-check and clean-up logs, its integrity record and its events, and the drive's
#: events and the driver's report. Each can hold a name or bytes the contestant chose, such as a
#: wheel's or an installed file's name.
HARNESS_FILES_OF_CONTESTANT_STAGES = [
    ("package", "logs/build.err"), ("package", "logs/wheel-check.out"), ("package", "logs/dry-run-clean.err"),
    ("package", "outputs/integrity.json"), ("package", "events.jsonl"), ("deploy_drive", "events.jsonl"),
    ("deploy_drive", "logs/driver.report.json")]


def _planted(tmp_path, checkout, stage: str, where: str, value: str):
    """A scored instance with a package stage in its plan and ``relay`` granted to its drive,
    whose *stage*'s attempt file *where* records *value* in a path; and that file's key."""
    plan, store = _instance(tmp_path, checkout)
    plan = _facing(_with_package(plan))
    key = plan.layout.attempt_file(stage, 1, where)
    store.write_bytes(key, json.dumps({"path": f"/opt/venv/lib/python3.12/site-packages/{value}.py"}).encode() + b"\n")
    return plan, store, key


@pytest.mark.parametrize("stage, where", HARNESS_FILES_OF_CONTESTANT_STAGES)
def test_a_contestant_facing_key_in_any_file_of_a_stage_that_runs_contestant_code_quarantines_it(
        tmp_path, checkout, stage, where):
    plan, store, key = _planted(tmp_path, checkout, stage, where, RELAY)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {key: ["relay"]}}
    row = _strict(store.read_bytes(ROW))
    assert (row["status"], row["quarantined"]) == ("ok", [key])


@pytest.mark.parametrize("stage, where", HARNESS_FILES_OF_CONTESTANT_STAGES)
def test_a_key_no_contestant_holds_in_a_file_of_a_stage_that_runs_contestant_code_refuses_the_instance(
        tmp_path, checkout, stage, where):
    plan, store, key = _planted(tmp_path, checkout, stage, where, SECRET)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {key} (hf_token)"
    assert not store.exists(ROW)


#: Files of the instance outside every attempt directory, which only the harness writes whatever
#: a stage runs, those beside a contestant-code stage's attempts included.
OUTSIDE_ATTEMPTS = {
    "a plan revision": "plan/rev-00000099.json",
    "a submission's status": "allocations/0/submissions/1/status.json",
    "the runner's job log": "allocations/0/submissions/1/slurm-100.out",
    "an orchestrator's events": "orchestrator/gen-00000001/events.jsonl",
    "beside the drive's manifest": "stages/deploy_drive/notes.json",
    "beside the package's attempts": "stages/package/attempts/notes.json",
    "the failed records": "failed/notes.json",
    "the lease": "lease/gen-00000009.json",
}


@pytest.mark.parametrize("where", sorted(OUTSIDE_ATTEMPTS))
def test_a_contestant_facing_key_outside_every_attempt_directory_refuses_the_instance(tmp_path, checkout, where):
    plan, store = _instance(tmp_path, checkout)
    plan = _facing(_with_package(plan))
    key = f"{plan.instance}/{OUTSIDE_ATTEMPTS[where]}"
    store.write_bytes(key, json.dumps({"held": RELAY}).encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {key} (relay)"
    assert not store.exists(ROW)


#: The verify stage itself, whose class declares which of its files hold the server's replies.
VERIFY = StageDecl("verify", VerifyStage.kind, CPU.model_copy(update={"disk_gib": KEPT_DISK_GIB}),
                   {"task": {"module": "fin_goodput.task", "overrides": {}}, "reference_image": "reference",
                    "reference_shape": CPU_ROLE.model_dump(), "data_copies": {}})


def _verified(tmp_path, checkout, quoted: str, verify_record: VerifyRecord | None = None):
    """A scored instance with the verify stage in its plan, whose verdict quotes *quoted* as a
    server's reply and whose ``verify.json`` is *verify_record* when given, and ``relay`` granted
    to its drive."""
    trace = drive_trace()
    good = passing_verdict(trace)
    verdict = Verdict(passed=good.passed, per_event=good.per_event, failure_mode=good.failure_mode,
                      diagnostics={"0:MetricVerifier": {"rationale": f"predicted='{quoted}'"}})
    plan = _facing(make_plan(tmp_path, checkout, "fin_goodput.task", before=(VERIFY,)))
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, trace=trace, verdict=verdict, verify_record=verify_record)
    run_and_publish(store, plan, "score", ScoreStage)
    return plan, store


def test_a_contestant_facing_key_in_the_verdict_quarantines_it(tmp_path, checkout):
    plan, store = _verified(tmp_path, checkout, RELAY)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    found = _quarantined(store, outputs)["files"]
    assert plan.layout.attempt_file("verify", 1, "outputs/verdict.json") in found
    assert plan.layout.attempt_file("score", 1, "outputs/score.json") in found
    assert _strict(store.read_bytes(ROW))["status"] == "ok"


def test_a_key_no_contestant_holds_in_the_verdict_refuses_the_instance(tmp_path, checkout):
    plan, store = _verified(tmp_path, checkout, SECRET)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    verdict = plan.layout.attempt_file("verify", 1, "outputs/verdict.json")
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith("the harness leaked secret values into published files: ")
    assert f"{verdict} (hf_token)" in e.value.fault.reason
    assert not store.exists(ROW)


#: Files of the verify attempt, each with what is planted in it and whether the verify stage's
#: declarations make a contestant-facing key there the contestant's (quarantined) or the
#: harness's (refused): its replies files and its role's logs hold the server's replies, and
#: verify.json only what the harness ran, published as the stage's record (None: planted after).
VERIFY_FILES = [("outputs/replies.0.MetricVerifier.jsonl.gz", gzip.compress(f'{{"reply": "{RELAY}"}}\n'.encode()),
                 True),
                ("logs/verify.err", f"event 3: the reply said {RELAY}\n".encode(), True),
                ("outputs/verify.json", None, False)]
#: A verify record that quotes the key where only the harness wrote: an engine's stack error.
QUOTING = VerifyRecord(eval_id="terminal", libraries={}, engines=[
    EngineRecord(role="reference", engine="E", stack=None, stack_error=f"RuntimeError: {RELAY}")])


@pytest.mark.parametrize("where, data, contestants", VERIFY_FILES, ids=["replies", "log", "verify.json"])
def test_the_verify_stage_declares_its_replies_and_its_roles_logs_a_contestants_and_nothing_else(
        tmp_path, checkout, where, data, contestants):
    plan, store = _verified(tmp_path, checkout, "A", verify_record=QUOTING if data is None else None)
    key = plan.layout.attempt_file("verify", 1, where)
    if data is not None:
        store.write_bytes(key, data)
    if contestants:
        _, outputs = _finalize(plan, store, secrets=BOTH)
        assert _quarantined(store, outputs) == {"files": {key: ["relay"]}}
        return
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {key} (relay)"
    assert not store.exists(ROW)


@pytest.mark.parametrize("stage, where", [("verify", "events.jsonl"), ("verify", "logs/verify.report.json"),
                                          ("score", "events.jsonl")])
def test_a_contestant_facing_key_in_a_file_the_harness_alone_wrote_refuses_the_instance(tmp_path, checkout, stage,
                                                                                        where):
    # Neither stage runs a contestant's code, and neither declares these files a contestant's.
    plan, store = _verified(tmp_path, checkout, "A")
    key = plan.layout.attempt_file(stage, 1, where)
    store.write_bytes(key, json.dumps({"held": RELAY}).encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {key} (relay)"
    assert not store.exists(ROW)


@pytest.mark.parametrize("log", ["driver.out", "driver.err"])
def test_a_contestant_facing_key_the_driver_echoed_in_its_log_quarantines_it(tmp_path, checkout, log):
    # The driver logs what a failed request's reply said, so its logs hold the server's bytes.
    plan, store = _instance(tmp_path, checkout)
    key = plan.layout.attempt_file("deploy_drive", 1, f"logs/{log}")
    store.write_bytes(key, f"request 7 failed: the reply said {RELAY}\n".encode())
    _, outputs = _finalize(_facing(plan), store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {key: ["relay"]}}


def test_a_key_no_agent_was_granted_in_the_drivers_log_refuses_the_instance(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    key = plan.layout.attempt_file("deploy_drive", 1, "logs/driver.err")
    store.write_bytes(key, f"request 7 failed: the reply said {SECRET}\n".encode())
    with pytest.raises(StageFault) as e:
        _finalize(_facing(plan), store, secrets=BOTH)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {key} (hf_token)"


#: A stage whose outputs under ``transcript/`` an agent wrote.
TRANSCRIPT = StageDecl("agent_build", "transcript_stand_in", CPU)


#: What a stage that declares its contestant outputs published: a file name and its bytes.
DECLARED = [("transcript/agent.log", gzip.compress(RELAY.encode(), mtime=0)[:15]),
            ("transcript/deep/x.txt", b"\x1f\x8b" + RELAY.encode()),
            ("notes.txt", gzip.compress(RELAY.encode(), mtime=0))]


@pytest.mark.parametrize("name, data", DECLARED,
                         ids=["a cut stream read as stored", "a raw file", "a file the harness wrote around it"])
def test_a_stages_declared_contestant_outputs_are_quarantined_and_its_raw_ones_read_as_stored(
        tmp_path, checkout, name, data):
    plan = _facing(make_plan(tmp_path, checkout, "fin_goodput.task", before=(TRANSCRIPT,)))
    store = PosixStore(plan.store_root)
    key = write(store, plan, "agent_build", name, data)
    publish(store, plan, "agent_build", {"written": key})
    trace = drive_trace()
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    run_and_publish(store, plan, "score", ScoreStage)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    if name == "transcript/agent.log":
        assert "quarantine" not in outputs
    else:
        assert _quarantined(store, outputs) == {"files": {key: ["relay"]}}


def test_the_rows_quarantined_list_is_capped_and_counted(tmp_path, checkout):
    plan = _facing(make_plan(tmp_path, checkout, "fin_goodput.task", before=(TRANSCRIPT,)))
    store = PosixStore(plan.store_root)
    keys = {f"transcript/f{i:02d}.txt": write(store, plan, "agent_build", f"transcript/f{i:02d}.txt", RELAY.encode())
            for i in range(SHOWN_QUARANTINED + 5)}
    publish(store, plan, "agent_build", {})
    trace = drive_trace()
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    run_and_publish(store, plan, "score", ScoreStage)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert len(_quarantined(store, outputs)["files"]) == SHOWN_QUARANTINED + 5
    row = _strict(store.read_bytes(ROW))
    assert row["quarantined"] == sorted(keys.values())[:SHOWN_QUARANTINED]
    assert row["quarantined_count"] == SHOWN_QUARANTINED + 5


def test_p1_a_cut_gzip_left_by_a_failed_verify_attempt_does_not_void_the_instance(tmp_path, checkout):
    # The verification core writes each replies file straight to its name; a verify attempt killed
    # while writing one leaves a cut stream in that attempt's outputs.
    plan, store = _instance(tmp_path, checkout)
    cut = plan.layout.attempt_file("verify", 2, "outputs/replies.0.MetricVerifier.jsonl.gz")
    store.write_bytes(cut, gzip.compress(b'{"event": 0}\n' * 10000)[:200])
    ctx, _ = _finalize(plan, store)
    assert store.exists(ROW)
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "scan_cut_streams"]
    assert (event["count"], event["files"]) == (1, [cut])


def test_a_link_left_in_an_attempts_outputs_is_named_as_not_read_and_the_instance_published(tmp_path, checkout):
    """The scan never follows a link, and no export carries one: the instance is published as
    before, and the link is named in an event, not passed over in silence."""
    plan, store = _instance(tmp_path, checkout)
    link = plan.layout.attempt_file("score", 1, "outputs/elsewhere")
    os.symlink(tmp_path / "host.txt", store.path(link))
    ctx, _ = _finalize(plan, store)
    assert store.exists(ROW)
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "scan_unread_entries"]
    assert (event["count"], event["files"]) == (1, [link])


def test_a_secret_before_a_streams_cut_is_found(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    cut = plan.layout.attempt_file("verify", 2, "outputs/replies.0.MetricVerifier.jsonl.gz")
    whole = gzip.compress(SECRET.encode() + b"x" * 200_000, mtime=0)
    store.write_bytes(cut, whole[:len(whole) // 2])
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {cut} (hf_token)"


def test_p2_a_reply_echoed_into_verdict_json_refuses_the_instance(tmp_path, checkout):
    # The verification core records what the server said in verdict.json; hf_token is a
    # secret no contestant is handed, so finding it anywhere is the harness's leak.
    trace = drive_trace()
    good = passing_verdict(trace)
    verdict = Verdict(passed=good.passed, per_event=good.per_event, failure_mode=good.failure_mode,
                      diagnostics={**good.diagnostics, "0:MetricVerifier": {"rationale": f"predicted='{SECRET}'"}})
    plan, store = _instance(tmp_path, checkout, trace=trace, verdict=verdict)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert "files the harness wrote" not in e.value.fault.reason, e.value.fault.reason
    verdict_key = plan.layout.attempt_file("verify", 1, "outputs/verdict.json")
    assert e.value.fault.reason.startswith("the harness leaked secret values into published files: ")
    assert f"{verdict_key} (hf_token)" in e.value.fault.reason


def test_p2b_a_build_log_in_the_package_record_refuses_the_instance(tmp_path, checkout):
    # An invalid package's record carries the tail of the contestant's pip output, which the
    # score then copies into score.json.
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, outcome="invalid_artifact",
                     package=package_result(plan, invalid=True, log_tail=f"setup.py: token {SECRET}"))
    run_and_publish(store, plan, "score", ScoreStage)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert "files the harness wrote" not in e.value.fault.reason, e.value.fault.reason


#: The bytes of the agent's source bundle, a git pack whose compression hides what it holds.
BUNDLE = b"PACK compressed objects"


def _record(*findings: tuple[str, str], read: tuple[str, ...] = ("install",),
            areas: tuple[tuple[str, str, int], ...] = ()) -> bytes:
    """A package scan's record of *read*, listing *findings* (path, secret) and naming *areas*
    (secret, area, count) by count."""
    return PackageScan(secrets=["hf_token", "relay"], read=list(read), forms=list(FORMS),
                       findings=[PackageScanFinding(path=p, secret=s) for p, s in findings],
                       areas=[PackageScanArea(secret=s, area=a, count=c) for s, a, c in areas]
                       ).model_dump_json().encode()


def _scanned(tmp_path, checkout, *, install: bytes | None = INSTALL_SCAN, tree: bytes | None = TREE_SCAN,
             bundle: bytes | None = None, outcome: str = "served",
             build_extra: dict[str, tuple[str, bytes]] | None = None, image: str = "submission.sif"):
    """An instance whose source_scan published *tree* and whose package published *install* as their
    ``package-scan.json`` (none for None), after an agent_build that published *bundle* when given,
    and *build_extra* beside it; the plan grants ``relay`` to a stage that hands it to its
    contestant."""
    plan = _facing(make_plan(tmp_path, checkout, "fin_goodput.task"))
    store = PosixStore(plan.store_root)
    trace = drive_trace() if outcome == "served" else None
    publish_instance(store, plan, outcome=outcome, trace=trace,
                     verdict=passing_verdict(trace) if trace is not None else None, bundle=bundle, tree_scan=tree,
                     install_scan=install, build_extra=build_extra, image=image)
    run_and_publish(store, plan, "score", ScoreStage)
    return plan, store


def _keys(plan, image: str = "submission.sif") -> tuple[str, str, str, str]:
    """The source scan's record, the package's, the image (published as *image*) and the bundle,
    by their store keys."""
    return (plan.layout.attempt_file("source_scan", 1, f"outputs/{PACKAGE_SCAN_FILE}"),
            plan.layout.attempt_file("package", 1, f"outputs/{PACKAGE_SCAN_FILE}"),
            plan.layout.attempt_file("package", 1, f"outputs/{image}"),
            plan.layout.attempt_file("agent_build", 1, "outputs/source.bundle"))


def test_a_contestant_facing_finding_of_a_package_scan_quarantines_and_the_instance_is_scored(tmp_path,
                                                                                             checkout):
    plan, store = _scanned(tmp_path, checkout, install=_record(("/opt/venv/lib/x.py", "relay")),
                           tree=_record(("/workspace/server.py", "relay"), read=("bundle", "tree")))
    _, outputs = _finalize(plan, store, secrets=BOTH)
    tree, install, _, _ = _keys(plan)
    assert _quarantined(store, outputs) == {"files": {f"{tree}::/workspace/server.py": ["relay"],
                                                      f"{install}::/opt/venv/lib/x.py": ["relay"]}}
    assert _strict(store.read_bytes(ROW))["status"] == "ok"


def test_a_finding_of_a_package_scan_no_contestant_could_hold_refuses_the_instance(tmp_path, checkout):
    plan, store = _scanned(tmp_path, checkout, install=_record(("/opt/venv/lib/x.py", "hf_token")))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    _, install, _, _ = _keys(plan)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == ("the harness leaked secret values into published files: "
                                    f"{install}::/opt/venv/lib/x.py (hf_token)")


def test_one_of_our_own_secrets_in_the_contestants_tree_refuses_the_instance(tmp_path, checkout):
    """The source scan's record stands whatever the package then did: a value of a secret no
    contestant was handed, found in the contestant's tree, is our own leak."""
    plan, store = _scanned(tmp_path, checkout, tree=_record(("/workspace/.env", "hf_token"), read=("bundle", "tree")),
                           bundle=BUNDLE)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    tree, _, _, _ = _keys(plan)
    assert f"{tree}::/workspace/.env (hf_token)" in e.value.fault.reason and not store.exists(ROW)


def _in_quarantine(quarantined: dict, entry: str) -> bool:
    """Whether *entry* is kept out of an export by *quarantined*: named, or below an area it names
    (an entry ending with "/")."""
    return any(entry == q or (q.endswith("/") and entry.startswith(q)) for q in quarantined)


def test_a_key_in_more_files_than_the_bound_leaves_none_of_them_exportable(tmp_path, checkout, monkeypatch):
    """The tree's scan lists a secret's files up to its bound, so the record finalize parses stays
    small, and past it records the area and the count: finalize then quarantines the whole area, the
    tree's copy in the image, the image built from it and the bundle the scan read, so no file of
    the tree is exported, listed or not."""
    from hls.package import scan

    monkeypatch.setattr(scan, "FINDINGS_PER_SECRET", 2)
    tree = tmp_path / "tree"
    tree.mkdir()
    names = [f"mod{i:02}.py" for i in range(40)]
    for name in names:
        (tree / name).write_text(f"KEY = {RELAY!r}\n")
    objects = PackageScan(secrets=["hf_token", "relay"], findings=[], read=["bundle"], forms=list(FORMS))
    record = scan.scan(BOTH, ("tree", scan.tree_entries(tree, "/workspace")), extend=objects)
    data = record.model_dump_json().encode()
    assert len(record.findings) == 2 and len(data) < 1000
    plan, store = _scanned(tmp_path, checkout, tree=data, bundle=BUNDLE)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    quarantined = _quarantined(store, outputs)["files"]
    scan_key, _, image, bundle = _keys(plan)
    assert {f"{scan_key}::/workspace/", image, bundle} <= set(quarantined)
    assert all(_in_quarantine(quarantined, f"{scan_key}::/workspace/{name}") for name in names)
    assert all(secrets == ["relay"] for secrets in quarantined.values())
    assert _strict(store.read_bytes(ROW))["status"] == "ok"


def test_an_install_past_the_bound_quarantines_the_venv_and_the_image(tmp_path, checkout):
    plan, store = _scanned(tmp_path, checkout, install=_record(areas=(("relay", "install", 5000),)), bundle=BUNDLE)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    _, install, image, _ = _keys(plan)
    # The bundle holds the tree only, which the record does not say is over the bound.
    assert _quarantined(store, outputs) == {"files": {f"{install}::/opt/venv/": ["relay"], image: ["relay"]}}


def test_a_key_only_in_the_bundles_history_quarantines_the_bundle(tmp_path, checkout):
    """The checkout read every object of the bundle: a key committed and removed later is in no
    file of the tree, and the record names it by count in the bundle's objects, which leaves the
    whole bundle out of any export."""
    plan, store = _scanned(tmp_path, checkout, tree=_record(read=("bundle", "tree"), areas=(("relay", "bundle", 1),)),
                           bundle=BUNDLE)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    _, _, _, bundle = _keys(plan)
    assert _quarantined(store, outputs) == {"files": {bundle: ["relay"]}}


def test_a_key_the_source_scan_found_quarantines_the_bundle_when_the_package_published_nothing(tmp_path,
                                                                                                checkout):
    """A pip step ended on the contestant's own fault, so the package published no manifest: the
    source scan's record, published before the step ran, still names the key in the tree and in the
    bundle's objects, which keeps the file's entry and the bundle out of any export, and the failed
    attempt's image, which no scan read, too."""
    plan, store = _scanned(tmp_path, checkout, tree=_record(("/workspace/.env", "relay"), read=("bundle", "tree"),
                                                            areas=(("relay", "bundle", 1),)),
                           bundle=BUNDLE)
    store.path(plan.layout.manifest("package")).unlink()
    _, outputs = _finalize(plan, store, secrets=BOTH)
    tree, _, image, bundle = _keys(plan)
    assert _quarantined(store, outputs) == {"files": {f"{tree}::/workspace/.env": ["relay"], bundle: ["relay"],
                                                      image: []}}
    assert _strict(store.read_bytes(ROW))["status"] == "ok"


def test_a_key_in_an_over_limit_trees_objects_quarantines_the_bundle(tmp_path, checkout):
    """A tree past the limits is never checked out, and the package is invalid: the checkout read
    the bundle's objects first, so the key is found and the bundle kept out of any export."""
    plan, store = _scanned(tmp_path, checkout, tree=_record(read=("bundle",), areas=(("relay", "bundle", 2),)),
                           bundle=BUNDLE, outcome="invalid_artifact")
    _, outputs = _finalize(plan, store, secrets=BOTH)
    _, _, _, bundle = _keys(plan)
    assert _quarantined(store, outputs) == {"files": {bundle: ["relay"]}}


def test_a_secret_no_contestant_could_hold_past_the_bound_refuses_the_instance(tmp_path, checkout):
    plan, store = _scanned(tmp_path, checkout, tree=_record(read=("bundle", "tree"),
                                                            areas=(("hf_token", "tree", 3000),)))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    scan_key, _, _, _ = _keys(plan)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert f"{scan_key}::/workspace/ (hf_token)" in e.value.fault.reason and not store.exists(ROW)


def test_a_bundle_record_that_is_not_the_one_the_scan_read_refuses_as_infra(tmp_path, checkout):
    plan, store = _scanned(tmp_path, checkout, tree=_record(read=("bundle", "tree"), areas=(("relay", "tree", 3000),)),
                           bundle=BUNDLE)
    key = plan.layout.manifest("agent_build")
    store.write_bytes(key, store.read_bytes(key).replace(b"source.bundle", b"source.bundlf"))
    with pytest.raises(StageFault, match="is not the manifest source_scan read") as e:
        _finalize(plan, store, secrets=BOTH)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)


@pytest.mark.parametrize("data", [b"not json", b'{"secrets": [], "findings": [], "extra": 1}',
                                  b'{"secrets": [], "findings": [{"path": "/x"}]}'])
def test_a_malformed_package_scan_refuses_as_infra(tmp_path, checkout, data):
    plan, store = _scanned(tmp_path, checkout, install=data)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert PACKAGE_SCAN_FILE in e.value.fault.reason and not store.exists(ROW)


def test_the_scan_of_a_package_the_drive_pinned_is_read_too(tmp_path, checkout):
    artifact = {"kind": "package", "manifest": f"runs/fin/{ROLLOUT}/stages/package/MANIFEST.json"}
    grading = make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading")
    pins = publish_rollout_package(PosixStore(grading.store_root), grading,
                                   {PACKAGE_SCAN_FILE: _record(("/opt/venv/key.txt", "relay"))})
    plan, store = _instance(tmp_path, checkout, kind="grading", artifact=artifact, drive_inputs=pins)
    _, outputs = _finalize(_facing(plan), store, secrets=BOTH)
    scan = f"runs/fin/{ROLLOUT}/stages/package/attempts/1/outputs/{PACKAGE_SCAN_FILE}"
    assert _quarantined(store, outputs) == {"files": {f"{scan}::/opt/venv/key.txt": ["relay"]}}


def test_a_pinned_package_no_scan_read_leaves_its_image_quarantined_unscanned(tmp_path, checkout):
    artifact = {"kind": "package", "manifest": f"runs/fin/{ROLLOUT}/stages/package/MANIFEST.json"}
    grading = make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading")
    install = PackageScan(secrets=[], findings=[], read=["tree"], forms=list(FORMS)).model_dump_json().encode()
    pins = publish_rollout_package(PosixStore(grading.store_root), grading, {PACKAGE_SCAN_FILE: install})
    plan, store = _instance(tmp_path, checkout, kind="grading", artifact=artifact, drive_inputs=pins)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    image = f"runs/fin/{ROLLOUT}/stages/package/attempts/1/outputs/submission.sif"
    assert _quarantined(store, outputs) == {"files": {image: []}}


@pytest.mark.parametrize("missing", ["install", "tree", "both"])
def test_a_container_no_published_scan_read_is_quarantined_unscanned(tmp_path, checkout, missing):
    """A package with no record of what its install added, or no source scan of its
    tree, is unscanned, never clean: its image is kept out of any export, and so is a bundle no scan
    of its objects read. No secret is named, since nothing says which it holds."""
    plan, store = _scanned(tmp_path, checkout, install=None if missing in ("install", "both") else INSTALL_SCAN,
                           tree=None if missing in ("tree", "both") else TREE_SCAN, bundle=BUNDLE)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    _, _, image, bundle = _keys(plan)
    expected = {image: []} if missing == "install" else {image: [], bundle: []}
    assert _quarantined(store, outputs) == {"files": expected}
    row = _strict(store.read_bytes(ROW))
    assert row["status"] == "ok" and set(row["quarantined"]) == set(expected)


@pytest.mark.parametrize("image_file", SUBMISSION_FILES)
@pytest.mark.parametrize("searched, covered", [(["hf_token"], False), (["hf_token", "old_key", "relay"], True)],
                         ids=["fewer", "more"])
def test_a_record_covers_its_containers_only_when_it_searched_every_secret_the_site_holds(tmp_path, checkout,
                                                                                         searched, covered,
                                                                                         image_file):
    """The source scan and the package ran where the secrets file held only hf_token; finalize's
    site holds relay too, which nothing searched for: the bundle and the image are unscanned and
    kept out of any export, with no secret named. Records that searched every secret the site holds,
    and more, read them. The image is found by its name on each substrate."""
    tree = PackageScan(secrets=searched, read=["bundle", "tree"], findings=[], forms=list(FORMS)).model_dump_json().encode()
    install = PackageScan(secrets=searched, read=["install"], findings=[], forms=list(FORMS)).model_dump_json().encode()
    plan, store = _scanned(tmp_path, checkout, tree=tree, install=install, bundle=BUNDLE, image=image_file)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    _, _, image, bundle = _keys(plan, image_file)
    if covered:
        assert "quarantine" not in outputs
    else:
        assert _quarantined(store, outputs) == {"files": {image: [], bundle: []}}


@pytest.mark.parametrize("searched", ["fewer", "none"])
def test_a_package_pinned_by_a_grading_on_a_site_of_more_secrets_is_unscanned(tmp_path, checkout, searched):
    """A grading pins a rollout's package whose scans ran where the secrets file held fewer secrets,
    or none, so that they read nothing: on the grading's site, which holds more, the package's image
    is kept out of any export, unscanned."""
    names, tree_read, install_read = (["hf_token"], ["bundle", "tree"], ["install"]) if searched == "fewer" else (
        [], [], [])
    tree = PackageScan(secrets=names, read=tree_read, findings=[], forms=list(FORMS)).model_dump_json().encode()
    install = PackageScan(secrets=names, read=install_read, findings=[], forms=list(FORMS)).model_dump_json().encode()
    artifact = {"kind": "package", "manifest": f"runs/fin/{ROLLOUT}/stages/package/MANIFEST.json"}
    grading = make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading")
    pins = publish_rollout_package(PosixStore(grading.store_root), grading, {PACKAGE_SCAN_FILE: install},
                                   tree_scan=tree)
    plan, store = _instance(tmp_path, checkout, kind="grading", artifact=artifact, drive_inputs=pins)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    image = f"runs/fin/{ROLLOUT}/stages/package/attempts/1/outputs/submission.sif"
    assert _quarantined(store, outputs) == {"files": {image: []}}


def test_one_of_our_own_secrets_only_in_a_named_bundles_objects_refuses_the_instance(tmp_path, checkout):
    """``hls package --bundle`` names a bundle outside the store, so no file here holds what its
    objects hold. One of our own secrets found there still refuses the instance, by its secret
    first, named by the record and the objects it read."""
    plan, store = _scanned(tmp_path, checkout, tree=_record(read=("bundle", "tree"), areas=(("hf_token", "bundle", 1),)))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store, secrets=BOTH)
    scan_key, _, _, _ = _keys(plan)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert f"{scan_key}::{OUTSIDE_BUNDLE} (hf_token)" in e.value.fault.reason and not store.exists(ROW)


def test_a_file_an_agent_build_attempt_left_unpublished_in_its_outputs_is_quarantined_unscanned(tmp_path, checkout):
    """A commit point killed outright can leave its partial bundle, or git's lock of it, beside the
    published outputs, and an attempt that published nothing leaves every output so. No record names
    such a file, and it may hold compressed history: each is kept out of any export, with no secret
    named. The transcript's files, which its published index holds, are left in."""
    index = ("transcript/INDEX.json", b'{"files": [], "skipped": []}')
    plan, store = _scanned(tmp_path, checkout, bundle=BUNDLE, build_extra={"transcript": index})
    # The store takes no key that starts with a dot; the commit point writes the file itself.
    outputs_dir = plan.layout.attempt_file("agent_build", 1, "outputs")
    lock = f"{outputs_dir}/.tmp-source.bundle-9.lock"
    (store.path(outputs_dir) / ".tmp-source.bundle-9.lock").write_bytes(b"PACK half of the history")
    write(store, plan, "agent_build", "transcript/logs/agent.log", b"plain\n")
    write(store, plan, "agent_build", "transcript/trajectory.json", b"{}")
    brief = write(store, plan, "agent_build", "brief.md", b"the brief\n", attempt=2)
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {lock: [], brief: []}}


def test_containers_the_scans_read_are_left_in_and_every_snapshot_or_unpublished_one_is_kept_out(tmp_path,
                                                                                                checkout):
    plan, store = _scanned(tmp_path, checkout, bundle=BUNDLE)
    snapshot = plan.layout.attempt_file("agent_build", 1, "snapshots/main-000001.bundle")
    store.write_bytes(snapshot, b"PACK snapshot")
    left = plan.layout.attempt_file("agent_build", 2, "outputs/source.bundle")
    store.write_bytes(left, b"PACK a failed attempt's bundle")
    image = plan.layout.attempt_file("package", 2, "outputs/submission.sif")
    store.write_bytes(image, b"a failed attempt's image")
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {snapshot: [], left: [], image: []}}


def test_every_file_of_a_stage_the_plan_does_not_name_is_kept_out_unscanned(tmp_path, checkout):
    """Which of a stage's files are compressed containers its kind declares. Nothing declares them
    for a stage directory the plan does not name, which no runner of the plan writes, so finalize
    fails closed: every file of its attempts is kept out of any export, unscanned, and the instance
    is still scored."""
    plan, store = _scanned(tmp_path, checkout, bundle=BUNDLE)
    blob = plan.layout.attempt_file("unnamed", 1, "outputs/blob.bin")
    store.write_bytes(blob, b"PACK something")
    record = plan.layout.attempt_file("unnamed", 2, "status.json")
    store.write_bytes(record, b"{}")
    _, outputs = _finalize(plan, store, secrets=BOTH)
    assert _quarantined(store, outputs) == {"files": {blob: [], record: []}}
    assert _strict(store.read_bytes(ROW))["status"] == "ok"


#: Files of the instance the records do not name, which the scan reads all the same.
ANYWHERE = {
    "a plan revision": "plan/rev-00000001.json",
    "the runner's job log": "allocations/0/submissions/1/slurm-100.out",
    "a submission's status": "allocations/0/submissions/1/status.json",
    "an attempt's events": "stages/images/attempts/1/events.jsonl",
    "a failed attempt's output": "stages/deploy_drive/attempts/2/outputs/drive.json",
    "a failed attempt's log": "stages/verify/attempts/2/logs/verify.err",
    "the package stage's log": "stages/package/attempts/1/logs/dry-run.err",
    "an orchestrator's events": "orchestrator/gen-00000001/events.jsonl",
}


@pytest.mark.parametrize("where", sorted(ANYWHERE))
def test_every_file_under_the_instance_is_scanned(tmp_path, checkout, where):
    plan, store = _instance(tmp_path, checkout)
    key = f"{plan.instance}/{ANYWHERE[where]}"
    store.write_bytes(key, json.dumps({"held": SECRET}).encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {key} (hf_token)"


def test_logs_of_one_name_in_two_stages_are_both_scanned(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    for stage, text in (("deploy_drive", "clean"), ("verify", SECRET)):
        store.write_bytes(plan.layout.attempt_file(stage, 1, "logs/extract-0123456789ab.out"), text.encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    log = plan.layout.attempt_file("verify", 1, "logs/extract-0123456789ab.out")
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {log} (hf_token)"


def test_a_harness_log_that_starts_like_gzip_is_read_as_stored(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    store.write_bytes(plan.layout.attempt_file("package", 1, "logs/dry-run.out"), b"\x1f\x8b cut")
    store.write_bytes(f"{plan.instance}/allocations/0/submissions/1/slurm-100.out", b"\x1f\x8b cut")
    _finalize(plan, store)
    assert store.exists(ROW)


def test_a_rollouts_scan_leaves_its_gradings_to_their_own(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    store.write_bytes(f"{plan.instance}/gradings/g-1/plan/rev-00000001.json", SECRET.encode())
    _finalize(plan, store)
    assert store.exists(ROW)


def test_finalize_refuses_to_run_without_the_registry_the_runner_loads(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    ctx = context(store, plan, "finalize", secrets={}, registry=False)
    with pytest.raises(StageFault) as e:
        FinalizeStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "secrets registry" in e.value.fault.reason
    assert ctx.launcher.launched == [] and not store.exists(ROW)


def test_a_hit_in_a_file_whose_name_is_not_utf8_gives_a_fault_the_runner_can_write(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    outputs = store.path(plan.layout.attempt_file("images", 1, "outputs/images.json")).parent
    (outputs / os.fsdecode(b"leak-\xff.txt")).write_bytes(SECRET.encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    fault = e.value.fault
    shown = plan.layout.attempt_file("images", 1, "outputs/images.json").rsplit("/", 1)[0] + "/leak-\\xff.txt"
    assert fault.reason == f"the harness leaked secret values into published files: {shown} (hf_token)"
    fault.model_dump_json()


def test_a_scan_fault_names_ten_files_and_counts_the_rest(tmp_path, checkout):
    extra = {f"leak-{i:02d}.txt": SECRET.encode() for i in range(SHOWN_HITS + 2)}
    plan, store = _instance(tmp_path, checkout, extra=extra)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    named = [f"{plan.layout.attempt_file('images', 1, f'outputs/leak-{i:02d}.txt')} (hf_token)"
             for i in range(SHOWN_HITS)]
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {', '.join(named)} and 2 more"


def test_a_scan_that_ended_well_without_a_readable_report_is_not_retried(tmp_path, checkout):
    # A scan that says it finished, and wrote no report of what it found: nothing says the files
    # hold no value.
    stub = tmp_path / "stub.py"
    stub.write_text("import sys\nsys.exit(0)\n")
    plan, store = _instance(tmp_path, checkout)
    ctx = context(store, plan, "finalize", secrets=SECRETS,
                  launcher=LocalLauncher(gpus=0, replace_modules={"hls.substrate.secret_scan": stub}))
    with pytest.raises(StageFault) as e:
        FinalizeStage().run(ctx)
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == "the scan ended, and its report of what it found could not be read"
    assert SECRET not in fault.reason and not store.exists(ROW)


def test_a_scan_that_failed_is_retried(tmp_path, checkout):
    stub = tmp_path / "stub.py"
    stub.write_text("import sys\nsys.exit(1)\n")
    plan, store = _instance(tmp_path, checkout)
    ctx = context(store, plan, "finalize", secrets=SECRETS,
                  launcher=LocalLauncher(gpus=0, replace_modules={"hls.substrate.secret_scan": stub}))
    with pytest.raises(StageFault) as e:
        FinalizeStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert not store.exists(ROW)


def test_the_scans_findings_do_not_pass_through_its_capped_logs(tmp_path, checkout):
    # Every role's stdout keeps only its first and last bytes; the findings are a file of the role.
    extra = {f"leak-{i:02d}.txt": SECRET.encode() for i in range(SHOWN_HITS + 2)}
    plan, store = _instance(tmp_path, checkout, extra=extra)
    ctx = context(store, plan, "finalize", secrets=SECRETS, launcher=LocalLauncher(gpus=0, log_keep_bytes=64))
    with pytest.raises(StageFault) as e:
        FinalizeStage().run(ctx)
    assert e.value.fault.reason.endswith(" and 2 more")


def test_a_record_the_scrubber_missed_is_caught_by_the_scan(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout, overrides={"task.model.base_checkpoint": SECRET})
    ctx = context(store, plan, "finalize", secrets=SECRETS)
    ctx.scrubber = Scrubber({})
    with pytest.raises(StageFault) as e:
        FinalizeStage().run(ctx)
    reason = e.value.fault.reason
    assert ctx.output_key("grading.json") in reason and ctx.output_key("rollout.json") in reason
    assert SECRET not in reason and not store.exists(f"runs/fin/index/{ROLLOUT}.json")


def test_a_secret_in_a_log_the_record_names_refuses_the_instance(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/driver.err")
    store.write_bytes(log, f"Authorization: Bearer {SECRET}\n".encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {log} (hf_token)"


@pytest.mark.parametrize("data", [b"\x1f\x8b not a gzip stream", gzip.compress(SECRET.encode(), mtime=0)[:12],
                                  gzip.compress(SECRET.encode(), mtime=0)],
                         ids=["magic bytes alone", "a stream cut short", "a whole stream"])
def test_a_log_is_scanned_as_it_is_stored_never_decompressed(tmp_path, checkout, data):
    # A server's stdout is the contestant's bytes. Read as gzip, two magic bytes or a cut stream
    # would fail every retry of the scan, and a small stream could expand without bound; the
    # log is shown as it is stored, so it is scanned as it is stored.
    plan, store = _instance(tmp_path, checkout)
    store.write_bytes(plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.out"), data)
    _finalize(plan, store)
    assert store.exists(ROW)


def test_a_secret_in_a_harness_log_that_starts_like_gzip_is_found_as_stored(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    log = plan.layout.attempt_file("deploy_drive", 1, "logs/driver.err")
    store.write_bytes(log, b"\x1f\x8b" + SECRET.encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {log} (hf_token)"


def test_the_scan_gets_the_values_only_through_its_env_file(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    ctx, _ = _finalize(plan, store)
    [scan] = [s for s in ctx.launcher.launched if s.name == "secret-scan"]
    assert (scan.owner, scan.shape) == ("harness", CPU_ROLE)
    # A role given secrets logs node-locally; the store gets the scrubbed copy at its report.
    assert not scan.log_dir.is_relative_to(store.path(plan.instance))
    assert (ctx.attempt_dir / "logs" / "secret-scan.out").is_file()
    assert SECRET not in " ".join(scan.argv) and SECRET not in json.dumps(scan.env)
    assert "hf_token" in scan.argv[scan.argv.index("--names") + 1]
    assert scan.env_file is not None and not scan.env_file.exists()


def test_a_scan_that_cannot_read_a_published_file_is_retried(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    store.path(plan.layout.attempt_file("images", 1, "outputs/images.json")).unlink()
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert not store.exists(f"runs/fin/index/{ROLLOUT}.json")


def test_a_site_without_secret_values_skips_the_scan(tmp_path, checkout):
    plan, store = _instance(tmp_path, checkout)
    ctx, _ = _finalize(plan, store, secrets={})
    assert [s.name for s in ctx.launcher.launched] == []
    assert store.exists(ROW)
    events = [json.loads(line) for line in ctx.store.path(
        plan.layout.attempt_file("finalize", 1, "events.jsonl")).read_text().splitlines()]
    assert "secret_scan_skipped" in [e["kind"] for e in events]


def test_the_stage_needs_the_registry_and_holds_no_gpu():
    assert FinalizeStage.needs_secret_registry is True
    params = FinalizeParams(task=TaskRef(module="fin_goodput.task"), kind="rollout")
    FinalizeStage.check(CPU, params, runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="uses no GPU"):
        FinalizeStage.check(ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=8, time_sec=600), params,
                            runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError):
        FinalizeStage.check(CPU.model_copy(update={"memory_gib": RUNNER_GIB}), params, runner_memory_gib=RUNNER_GIB)


def test_index_keys_never_share_a_file_and_refuse_what_is_not_a_run():
    assert index_key("runs/t/r1", "rollout") == ("runs/t/index/r1.json", "r1", None)
    assert index_key("runs/t/r1/gradings/g1", "grading") == ("runs/t/index/r1+g1.json", "r1", "g1")
    # An evaluation, a grading under a cache warmer, goes to the warmers' own index.
    assert index_key("warmers/t/w1/gradings/g1", "grading") == ("warmers/t/index/w1+g1.json", "w1", "g1")
    refused = [("credits/t/c1", "rollout"), ("runs/t/r1", "grading"), ("runs/t/r1/gradings/g1", "rollout"),
               ("runs/t/index", "rollout"), ("warmers/t/w1", "rollout"), ("warmers/t/index/gradings/g1", "grading"),
               ("warmers/t/w+1/gradings/g1", "grading")]
    for instance, kind in refused:
        with pytest.raises(ValueError):
            index_key(instance, kind)


def test_a_rollout_id_with_a_plus_has_no_row_of_its_own():
    # A grading's row joins its ids with "+", so a rollout id holding one could name another's row.
    for instance, kind in (("runs/t/r+1", "rollout"), ("runs/t/r+1/gradings/g1", "grading")):
        with pytest.raises(ValueError, match=r"\+"):
            index_key(instance, kind)


def test_a_name_that_is_no_store_key_has_no_row():
    # "runs/t/.." would put the row of the task's own directory among its rollouts'.
    for instance in ("runs/t/..", "runs/t/r 1", "runs/t/r1\n"):
        with pytest.raises(ValueError, match="is not an instance's root"):
            index_key(instance, "rollout")


def test_each_stage_kind_says_how_many_roles_one_attempt_starts_at_most():
    images = ImagesParams.model_validate({"catalog_task": "fin", "images": [
        d.model_dump(mode="json") for d in (catalog.BASES["cuda"], catalog.SERVER_LAYERS["cuda-setuptools"])]})
    assert ImagesStage.max_roles(images) == 8
    assert PackageStage.max_roles(None) == 14 and DeployDriveStage.max_roles(None) == 9
    assert ProvisionStage.max_roles(None) == 5 and ScoreStage.max_roles(None) == 1
    assert VerifyStage.max_roles(None) == 4 and FinalizeStage.max_roles(None) == 1
    assert SourceScanStage.max_roles(None) == 2
    assert OracleFillStage.max_roles(None) == 5
    noop = NoopParams(gpu_roles=2, shared_role=True, cpu_role=True)
    assert NoopStage.max_roles(noop) == 4


def _quiet_scan(tmp_path: Path) -> dict[str, Path]:
    """A secret scan that reads nothing and finds nothing, in place of ``hls.substrate.secret_scan``
    (``LocalLauncher.replace_modules``): it writes an empty report, as a scan that found nothing
    does."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    stub = tmp_path / "quiet_scan.py"
    stub.write_text("import sys\n\nargs = sys.argv[1:]\nopen(args[args.index('--report') + 1], 'w').close()\n")
    return {"hls.substrate.secret_scan": stub}


def test_finalize_is_planned_for_its_fixed_time():
    """Every plan's finalize stage takes ``FINALIZE_SHAPE``, an hour, which its scan of the instance
    runs within; the stage accepts the shape."""
    assert FINALIZE_SEC == 3600
    assert FINALIZE_SHAPE == ResourceShape(cpus=2, memory_gib=12, time_sec=FINALIZE_SEC)
    FinalizeStage.check(FINALIZE_SHAPE, FinalizeParams(task=TaskRef(module="fin_goodput.task"), kind="rollout"),
                        runner_memory_gib=RUNNER_GIB)


def _timed_scan(tmp_path, checkout, *, start_after: float, run_sec: float):
    """A finalize of the fixture instance whose quiet scan's step starts *start_after* seconds after
    its launch and works *run_sec* seconds, its stage's deadline ``FINALIZE_SEC`` after it begins:
    the context, its clock and the time it began."""
    plan, store = _instance(tmp_path, checkout)
    clock = FakeClock()
    launcher = LocalLauncher(gpus=0, clock=clock, start_after={SCAN: start_after}, run_sec={SCAN: run_sec},
                             replace_modules=_quiet_scan(tmp_path))
    ctx = context(store, plan, "finalize", secrets=SECRETS, launcher=launcher)
    ctx.clock = clock
    begun = clock()
    ctx.deadline = begun + FINALIZE_SEC
    return ctx, clock, begun


def _scan_event(ctx, kind: str) -> dict:
    """The scan role's one event of *kind* in *ctx*'s attempt."""
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == kind and e["role"] == SCAN]
    return event


def test_a_scan_runs_to_the_stages_deadline_however_little_the_instance_holds(tmp_path, checkout):
    """The scan has the whole of the stage's time: a scan of a small instance that reads for 3,500 s
    of finalize's 3,600 ends, and its start leaves the stage's deadline where it was."""
    ctx, clock, begun = _timed_scan(tmp_path, checkout, start_after=0.0, run_sec=3500.0)
    FinalizeStage().run(ctx)
    ended = _scan_event(ctx, "role_ended")
    assert (ended["ending"], ended["exit_code"]) == ("exited", 0)
    assert clock() - begun == pytest.approx(3500.0, abs=2.0)
    assert ctx.deadline == begun + FINALIZE_SEC
    assert ctx.store.exists(ROW)


def test_a_scan_still_running_at_the_stages_deadline_ends_the_stage_as_infra_then(tmp_path, checkout):
    """A scan that reads past the stage's deadline is stopped at that deadline, not before, and the
    stage ends as infra with no record written."""
    ctx, clock, begun = _timed_scan(tmp_path, checkout, start_after=0.0, run_sec=5000.0)
    with pytest.raises(StageFault, match="harness role secret-scan ended time_limit") as e:
        FinalizeStage().run(ctx)
    assert e.value.fault.fault_class == "infra"
    assert clock() - begun == pytest.approx(FINALIZE_SEC, abs=2.0)
    assert ctx.deadline == begun + FINALIZE_SEC
    assert not ctx.store.exists(ROW)


def test_a_scan_whose_step_starts_late_gets_the_stages_time_grown_by_its_wait(tmp_path, checkout):
    """A scan whose step started 700 s after its launch, as a slow controller can, and reads for
    3,300 s ends 4,000 s in, past finalize's 3,600: the wait for the step grew the stage's deadline by
    700 s, so the scheduler's time comes out of none of the scan's."""
    ctx, clock, begun = _timed_scan(tmp_path, checkout, start_after=700.0, run_sec=3300.0)
    FinalizeStage().run(ctx)
    assert _scan_event(ctx, "role_started")["launch_wait_sec"] == pytest.approx(700.0, abs=1.0)
    ended = _scan_event(ctx, "role_ended")
    assert (ended["ending"], ended["exit_code"]) == ("exited", 0)
    assert clock() - begun == pytest.approx(700.0 + 3300.0, abs=2.0)
    assert ctx.deadline == pytest.approx(begun + FINALIZE_SEC + 700.0, abs=1.0)
    assert ctx.store.exists(ROW)


def test_a_scan_whose_step_starts_late_ends_by_the_allocations_end(tmp_path, checkout):
    # The stage's 3,600 s grow by the scan's 700 s wait, but only to the allocation's end, 4,000 s
    # in; the grown deadline would reach 4,300 s.
    ctx, clock, begun = _timed_scan(tmp_path, checkout, start_after=700.0, run_sec=5000.0)
    cap = ctx.deadline_cap = begun + 4000.0
    with pytest.raises(StageFault, match="harness role secret-scan ended time_limit"):
        FinalizeStage().run(ctx)
    assert ctx.deadline == cap and cap - 1.0 <= clock() <= cap + 1.0
