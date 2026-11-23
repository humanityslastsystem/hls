from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest

from hls.core.score import Score
from hls.core.verdict import Verdict
from hls.pipeline.stage import StageFault
from hls.score import role
from hls.spec.resources import ResourceShape
from hls.stages import finalize  # noqa: F401  (registers the finalize stage the plans name)
from hls.stages.score import ScoreParams, ScoreStage
from hls.store.manifest import canonical_json, read_outputs_report
from hls.store.posix import PosixStore
from hls.measured import BASELINE_CREDIT_FILE, defaults_path
from hls.taskref import MeasuredPaths, TaskRef, blob_sha, load_task, read_task
from tests.finalize.fixture import t0_trace
from tests.finalize.published import (
    CPU,
    CPU_ROLE,
    NOT_SERVED,
    RUNNER_GIB,
    context,
    make_plan,
    package_result,
    publish_instance,
)
from tests.pipeline.fakes import LocalLauncher
from tests.trace.traces import LATENCY_TENANTS, drive_trace, passing_verdict

NAN, INF = float("nan"), float("inf")


def _failing(trace) -> Verdict:
    """A verdict that fails the first event, with diagnostics no strict JSON can hold."""
    per_event = {ev.event_key: i != 0 for i, ev in enumerate(trace.events)}
    return Verdict(passed=False, per_event=per_event, failure_mode="correctness",
                   diagnostics={"0:MetricVerifier": {"value": NAN, "bound": INF, "n": 3}})


def _old_score(module: str, trace, verdict) -> bytes:
    """What today's runner writes to ``score.json`` for this task, trace and verdict, scored against
    its default credit."""
    ref = TaskRef(module=module)
    task = load_task(ref, credit=True, files=MeasuredPaths(credit=defaults_path(ref, BASELINE_CREDIT_FILE)))
    score = asyncio.run(task.scorer.score(task.effective_tenants(), trace, verdict, task.hardware))
    return score.model_dump_json(indent=2).encode()


def _score(tmp_path, checkout, module, **instance):
    plan = make_plan(tmp_path, checkout, module)
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, **instance)
    ctx = context(store, plan, "score")
    return ctx, ScoreStage().run(ctx)


CASES = [
    ("fin_goodput.task", drive_trace(failures=True)),
    ("fin_goodput.task", drive_trace(failures=False)),
    ("fin_session.task", drive_trace(failures=True)),
    ("fin_latency.task", drive_trace(failures=False, tenants=LATENCY_TENANTS)),
    ("fin_latency.task", drive_trace(failures=True, tenants=LATENCY_TENANTS)),
    ("fin_t0.task", t0_trace(failures=False)),
    ("fin_t0.task", t0_trace(failures=True)),
    ("fin_single.task", t0_trace(failures=False)),
    ("fin_single.task", t0_trace(failures=True)),
]


@pytest.mark.parametrize("verdict", ["passing", "failing"])
@pytest.mark.parametrize("module, trace", CASES)
def test_a_served_drive_scores_the_bytes_the_old_runner_wrote(tmp_path, checkout, module, trace, verdict):
    v = passing_verdict(trace) if verdict == "passing" else _failing(trace)
    ctx, outputs = _score(tmp_path, checkout, module, trace=trace, verdict=v)
    assert sorted(outputs) == ["score", "score_provenance"]
    assert outputs["score"] == ctx.output_key("score.json")
    assert ctx.store.read_bytes(outputs["score"]) == _old_score(module, trace, v)
    [spec] = ctx.launcher.launched
    assert (spec.name, spec.owner, spec.shape) == ("score", "harness", CPU_ROLE)


def test_the_cases_cover_passing_runs_gated_runs_and_the_credit(tmp_path, checkout):
    scores = [json.loads(_old_score(m, t, passing_verdict(t))) for m, t in CASES]
    assert any(s["gate_passed"] and s["value"] > 0 for s in scores)
    assert any(not s["gate_passed"] and "slo" in s["zero_reasons"] for s in scores)
    latency = json.loads(_old_score(*CASES[3], passing_verdict(CASES[3][1])))
    assert latency["breakdown"]["baseline_log_credit"] == pytest.approx(1.0)


def test_the_score_roles_outputs_are_recorded_by_its_report(tmp_path, checkout):
    trace = drive_trace()
    ctx, outputs = _score(tmp_path, checkout, "fin_goodput.task", trace=trace, verdict=passing_verdict(trace))
    report = read_outputs_report(ctx.outputs_report("score"))
    assert sorted(report) == ["score-provenance.json", "score.json"]
    for name, file in (("score", "score.json"), ("score_provenance", "score-provenance.json")):
        data = ctx.store.read_bytes(outputs[name])
        assert report[file].sha256 == hashlib.sha256(data).hexdigest()
        # The manifest records the role's own digest, read from its report and not from the file.
        assert ctx.output_ref(outputs[name]).sha256 == report[file].sha256
    assert "--outputs-out" in ctx.launcher.launched[0].argv


def test_a_score_role_that_reports_none_of_its_outputs_is_retried(tmp_path, checkout):
    stub = tmp_path / "stub.py"
    stub.write_text("import sys\nfrom pathlib import Path\n"
                    "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
                    "(out / 'score.json').write_text('{}')\n(out / 'score-provenance.json').write_text('{}')\n")
    trace = drive_trace()
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    ctx = context(store, plan, "score", launcher=LocalLauncher(gpus=0, replace_modules={"hls.score.role": stub}))
    with pytest.raises(StageFault) as e:
        ScoreStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "reported no digest of output score.json" in e.value.fault.reason


def test_the_provenance_records_b_the_blob_it_came_from_and_its_campaign(tmp_path, checkout):
    trace = drive_trace(failures=False, tenants=LATENCY_TENANTS)
    ctx, outputs = _score(tmp_path, checkout, "fin_latency.task", trace=trace, verdict=passing_verdict(trace))
    provenance = json.loads(ctx.store.read_bytes(outputs["score_provenance"]))
    path = checkout / "fin_latency" / "defaults" / "baseline.json"
    hashed = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True)
    blob = hashed.stdout.strip()
    credit = json.loads(path.read_text())
    identity = json.dumps(credit["campaign"], sort_keys=True, separators=(",", ":")).encode()
    # The fixture credit records no hardware.
    assert provenance == {"baseline": {"log_credit": pytest.approx(1.0), "blob_sha": blob,
                                       "campaign_id": hashlib.sha256(identity).hexdigest(), "hardware": None}}


def test_a_scorer_without_a_baseline_records_none(tmp_path, checkout):
    trace = drive_trace()
    ctx, outputs = _score(tmp_path, checkout, "fin_goodput.task", trace=trace, verdict=passing_verdict(trace))
    assert json.loads(ctx.store.read_bytes(outputs["score_provenance"])) == {"baseline": None}


@pytest.mark.parametrize("outcome", ["crashed", "not_ready"])
def test_a_drive_that_did_not_serve_scores_a_zero_whose_reason_is_its_outcome(tmp_path, checkout, outcome):
    ctx, outputs = _score(tmp_path, checkout, "fin_latency.task", outcome=outcome)
    want = Score.zero(reasons=(outcome,), diagnostics={"reason": NOT_SERVED[outcome]})
    assert ctx.store.read_bytes(outputs["score"]) == want.model_dump_json(indent=2).encode()


#: The ``score.json`` written for an invalid artifact, without and with the package build's exit status and log.
OLD_INVALID_ARTIFACT_SCORES = {
    None: (b'{\n  "value": 0.0,\n  "achieved": 0.0,\n  "sol": 0.0,\n  "unit": "",\n  "gate_passed": false,\n'
           b'  "zero_reasons": [\n    "invalid_artifact"\n  ],\n  "breakdown": {\n    "invalid_artifact": {\n'
           b'      "message": "pyproject.toml is missing"\n    }\n  }\n}'),
    1: (b'{\n  "value": 0.0,\n  "achieved": 0.0,\n  "sol": 0.0,\n  "unit": "",\n  "gate_passed": false,\n'
        b'  "zero_reasons": [\n    "invalid_artifact"\n  ],\n  "breakdown": {\n    "invalid_artifact": {\n'
        b'      "message": "pyproject.toml is missing",\n      "returncode": 1,\n'
        b'      "log": "ERROR: No matching distribution found for x"\n    }\n  }\n}'),
}


@pytest.mark.parametrize("returncode, log", [(None, None), (1, "ERROR: No matching distribution found for x")])
def test_an_invalid_artifact_scores_what_the_old_runner_recorded(tmp_path, checkout, returncode, log):
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, outcome="invalid_artifact",
                     package=package_result(plan, invalid=True, returncode=returncode, log_tail=log))
    ctx = context(store, plan, "score")
    outputs = ScoreStage().run(ctx)
    assert ctx.store.read_bytes(outputs["score"]) == OLD_INVALID_ARTIFACT_SCORES[returncode]
    assert "--package" in ctx.launcher.launched[0].argv


def test_an_invalid_artifact_whose_package_record_the_drive_did_not_pin_is_refused(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    published = publish_instance(store, plan, outcome="invalid_artifact")
    ctx = context(store, plan, "score")
    key = plan.layout.manifest("deploy_drive")
    ctx.inputs[key] = published["deploy_drive"].model_copy(update={"inputs": {}})
    with pytest.raises(StageFault) as e:
        ScoreStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "does not pin the package manifest" in e.value.fault.reason
    assert ctx.launcher.launched == []


def test_a_package_record_that_changed_after_the_drive_read_it_is_refused(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    published = publish_instance(store, plan, outcome="invalid_artifact")
    changed = package_result(plan, invalid=True, returncode=2)
    store.write_bytes(published["package"].outputs["package"].key, changed)
    with pytest.raises(StageFault) as e:
        ScoreStage().run(context(store, plan, "score"))
    assert (e.value.fault.retryable, "does not match its manifest" in e.value.fault.reason) == (False, True)


@pytest.mark.parametrize("module, what", [("fin_stale.task", "gpu_kind"),
                                          ("fin_nocredit.task", "no baseline.json was given")])
def test_a_baseline_credit_the_task_cannot_use_is_refused_and_not_retried(tmp_path, checkout, module, what):
    trace = drive_trace(failures=False, tenants=LATENCY_TENANTS)
    with pytest.raises(StageFault) as e:
        _score(tmp_path, checkout, module, trace=trace, verdict=passing_verdict(trace))
    fault = e.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith("score refused: BaselineCreditError: ") and what in fault.reason


def test_a_package_manifest_that_is_not_the_one_the_drive_read_is_refused(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    published = publish_instance(store, plan, outcome="invalid_artifact")
    store.path(plan.layout.manifest("package")).write_bytes(
        canonical_json(published["package"].model_copy(update={"attempt": 2})))
    ctx = context(store, plan, "score")
    with pytest.raises(StageFault) as e:
        ScoreStage().run(ctx)
    assert (e.value.fault.retryable, ctx.launcher.launched) == (False, [])
    assert "is not the package manifest the drive read" in e.value.fault.reason


def test_a_trace_end_that_does_not_match_its_manifest_is_refused_before_any_role(tmp_path, checkout):
    trace = drive_trace()
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    end = store.path(plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/END.json"))
    end.write_bytes(end.read_bytes().replace(b'"drained"', b'"wall"'))
    ctx = context(store, plan, "score")
    with pytest.raises(StageFault) as e:
        ScoreStage().run(ctx)
    assert (e.value.fault.retryable, ctx.launcher.launched) == (False, [])
    assert "END.json does not match its manifest" in e.value.fault.reason


def test_a_drive_record_that_does_not_match_its_manifest_is_refused_before_any_role(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    published = publish_instance(store, plan, outcome="crashed")
    key = published["deploy_drive"].outputs["drive"].key
    store.write_bytes(key, store.read_bytes(key).replace(b"status 1", b"status 2"))
    ctx = context(store, plan, "score")
    with pytest.raises(StageFault) as e:
        ScoreStage().run(ctx)
    assert (e.value.fault.retryable, ctx.launcher.launched) == (False, [])


def test_a_summary_that_does_not_match_its_end_fails_the_role_and_is_retried(tmp_path, checkout):
    trace = drive_trace()
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    summary = store.path(plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/summary.jsonl.gz"))
    summary.write_bytes(gzip.compress(gzip.decompress(summary.read_bytes()).replace(b'"chat"', b'"chit"')))
    ctx = context(store, plan, "score")
    with pytest.raises(StageFault) as e:
        ScoreStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "TraceIncomplete" in (ctx.attempt_dir / "logs" / "score.err").read_text()
    assert not ctx.store.path(ctx.output_key("score.json")).exists()


def test_a_role_that_exits_without_its_files_is_retried(tmp_path, checkout, monkeypatch):
    stub = tmp_path / "stub.py"
    stub.write_text("import sys\nsys.exit(0)\n")
    trace = drive_trace()
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    ctx = context(store, plan, "score", launcher=LocalLauncher(gpus=0, replace_modules={"hls.score.role": stub}))
    with pytest.raises(StageFault) as e:
        ScoreStage().run(ctx)
    assert (e.value.fault.retryable, "without writing" in e.value.fault.reason) == (True, True)


def test_the_credit_record_names_the_credit_the_load_parsed_and_reads_no_file(checkout, tmp_path, monkeypatch):
    """The credit a score names is the one the load read and gave the scorer, by the blob id of the
    bytes it parsed, whatever the file holds by the time the record is made; the record reads no
    file of its own."""
    defaults = checkout / "fin_latency" / "defaults" / "baseline.json"
    given = tmp_path / "baseline.json"
    given.write_bytes(defaults.read_bytes())
    loaded = read_task(TaskRef(module="fin_latency.task"), credit=True, files=MeasuredPaths(credit=given))
    # The file now holds the same content in another layout, so other bytes.
    given.write_text(json.dumps(json.loads(defaults.read_text()), indent=3))
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail(f"read {self}") if self.name == "baseline.json"
                        else real(self))
    record = role.credit_record(loaded)
    assert record.blob_sha == loaded.credit.blob != blob_sha(given)
    assert (record.log_credit, record.blob_sha) == (pytest.approx(1.0), blob_sha(defaults))
    identity = json.dumps(loaded.credit.value.campaign.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert record.campaign_id == hashlib.sha256(identity.encode()).hexdigest()
    # A scorer that scores against no baseline was given none, and names none.
    assert role.credit_record(read_task(TaskRef(module="fin_goodput.task"), credit=True)) is None


def test_the_role_refuses_an_invalid_artifact_without_its_package_record(tmp_path, checkout, capsys):
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    published = publish_instance(store, plan, outcome="invalid_artifact")
    ref = tmp_path / "ref.json"
    ref.write_text(TaskRef(module="fin_latency.task").model_dump_json())
    drive = store.path(published["deploy_drive"].outputs["drive"].key).parent
    verdict = store.path(published["verify"].outputs["verdict"].key)
    out = tmp_path / "out"
    out.mkdir()
    report = tmp_path / "outputs.json"
    code = role.main(["--task-ref", str(ref), "--drive", str(drive), "--verdict", str(verdict), "--out", str(out),
                      "--outputs-out", str(report), "--credit", str(checkout / "fin_latency" / "defaults" / "baseline.json")])
    assert code == role.REFUSED_EXIT and list(out.iterdir()) == [] and not report.exists()
    err = capsys.readouterr().err
    assert "hls-score-refused: Refused: an invalid artifact is scored from its package record" in err


def _role_with_package(tmp_path, checkout, package: dict) -> int:
    """Run the score role over an invalid artifact's drive with the package record *package*."""
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    published = publish_instance(store, plan, outcome="invalid_artifact")
    ref, record, out = tmp_path / "ref.json", tmp_path / "package.json", tmp_path / "out"
    ref.write_text(TaskRef(module="fin_latency.task").model_dump_json())
    record.write_text(json.dumps(package))
    out.mkdir()
    return role.main(["--task-ref", str(ref), "--drive", str(store.path(published["deploy_drive"].outputs["drive"].key)
                                                             .parent),
                      "--verdict", str(store.path(published["verify"].outputs["verdict"].key)), "--out", str(out),
                      "--outputs-out", str(tmp_path / "outputs.json"), "--package", str(record),
                      "--credit", str(checkout / "fin_latency" / "defaults" / "baseline.json")])


def test_the_role_scores_an_invalid_artifact_by_the_fields_of_its_package_record_it_reads(tmp_path, checkout):
    """A package record without the fields a score does not read, and with one this code does
    not know, scores the zero its build failure carries."""
    package = {"status": "invalid_artifact", "reason": "pyproject.toml is missing", "returncode": 1,
               "log_tail": "error: no project", "added_since": 1}
    assert _role_with_package(tmp_path, checkout, package) == 0
    score = Score.model_validate_json((tmp_path / "out" / role.SCORE_FILE).read_bytes())
    assert score == Score.zero(reasons=("invalid_artifact",), diagnostics={
        "message": "pyproject.toml is missing", "returncode": 1, "log": "error: no project"})


@pytest.mark.parametrize("package, errors", [
    ({"reason": "pyproject.toml is missing"}, "missing status (required by this version of hls)"),
    ({"status": "invalid_artifact"}, "reason_set_exactly_when_invalid at the top level"),
], ids=["no-status", "invalid-without-its-reason"])
def test_the_role_refuses_a_package_record_it_cannot_read_naming_its_errors(tmp_path, checkout, capsys, package,
                                                                             errors):
    assert _role_with_package(tmp_path, checkout, package) == role.REFUSED_EXIT
    err = capsys.readouterr().err
    assert f"hls-score-refused: Refused: {tmp_path / 'package.json'} is not a package record: {errors}" in err
    assert "validation error" not in err


def test_the_stage_holds_no_gpu_and_leaves_its_role_memory_beside_the_runners():
    params = ScoreParams(task=TaskRef(module="fin_latency.task"))
    ScoreStage.check(CPU, params, runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="uses no GPU"):
        ScoreStage.check(ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=8, time_sec=600), params,
                         runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError):
        ScoreStage.check(CPU.model_copy(update={"memory_gib": RUNNER_GIB}), params, runner_memory_gib=RUNNER_GIB)


def test_the_scores_old_bytes_keep_non_finite_diagnostics_as_null(tmp_path, checkout):
    trace = drive_trace(failures=False)
    ctx, outputs = _score(tmp_path, checkout, "fin_goodput.task", trace=trace, verdict=_failing(trace))
    text = ctx.store.read_bytes(outputs["score"]).decode()
    doc = json.loads(text, parse_constant=lambda c: pytest.fail(f"score.json holds {c}"))
    assert doc["breakdown"]["verdict"]["0:MetricVerifier"] == {"value": None, "bound": None, "n": 3}
    assert not math.isnan(doc["value"])
