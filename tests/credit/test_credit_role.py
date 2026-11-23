"""The credit role: one drive of the baseline, scored with no baseline credit, from the trace's
summary, against the passthrough verdict; and the drives no baseline credit may pool refused."""

from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from hls.core.trace import Trace
from hls.core.verdict import Verdict, passthrough_verdict
from hls.score import credit_role
from hls.score.credit_role import CREDIT_DRIVE_FILE, REFUSED_EXIT, REFUSED_PREFIX, CreditDrive, Refused, measure
from hls.scoring import SloLatencyScorer
from hls.store.manifest import read_outputs_report
from hls.taskref import TaskRef, apply_task_overrides
from hls.trace.store import write_trace
from tests.finalize.fixture import goodput_task, latency_task, t0_single_task, t0_trace
from tests.finalize.published import SIF_SHA, drive_record
from tests.trace.traces import LATENCY_TENANTS, drive_trace

#: A load override of the single-tenant task: its TTFT budget doubled, which halves every
#: request's SLO-normalized latency.
LOOSER = {"task.workload.slo.ttft_ms": 200.0}


def _errored_reply_trace() -> Trace:
    """The latency tenants' clean drive, but the first chat reply, which streamed every token in
    time, carries the driver's error: its content is not trusted, so it earns nothing, though
    the SLO prices its latency."""
    trace = drive_trace(failures=False, tenants=LATENCY_TENANTS)
    first = trace.events[0]
    assert first.tenant_id == "chat" and first.response is not None
    return Trace(events=[first.model_copy(update={"error": "stream reset after the last token"}), *trace.events[1:]])


def _drive_dir(tmp_path: Path, trace: Trace, *, outcome: str = "served") -> Path:
    """A drive stage's outputs directory: ``drive.json`` and, for a served drive, its trace. The
    fixture record's server opened its gates 1 s after it started and was driven from 100.5 to
    199.5."""
    out = tmp_path / "drive"
    out.mkdir()
    end = None
    if outcome == "served":
        end = write_trace(trace, out / "trace", ending="drained", driver={"class": "Driver"}).end
    (out / "drive.json").write_bytes(drive_record(outcome, end))
    return out


def test_the_passthrough_verdict_trusts_content_and_never_transport():
    """Every event the server answered without an error passes; an errored one and an unanswered
    one fail, an unanswered one even with no error recorded, and the run passes so that no
    measurement is gated to zero by the verdict."""
    trace = drive_trace(failures=True)
    errored = _errored_reply_trace().events[0]
    unanswered = [ev for ev in trace.events if ev.response is None]
    assert len(unanswered) == 2 and errored.event_key not in {ev.event_key for ev in unanswered}
    silent = unanswered[0].model_copy(update={"error": None, "failure": None})
    swapped = {errored.event_key: errored, silent.event_key: silent}
    verdict = passthrough_verdict(Trace(events=[swapped.get(ev.event_key, ev) for ev in trace.events]))
    assert verdict.passed
    failing = {ev.event_key for ev in unanswered} | {errored.event_key}
    assert verdict.per_event == {ev.event_key: ev.event_key not in failing for ev in trace.events}


@pytest.mark.parametrize("task, trace", [
    (latency_task(), drive_trace(failures=False, tenants=LATENCY_TENANTS)),
    (latency_task(), _errored_reply_trace()),
    (t0_single_task(), t0_trace(failures=False)),
], ids=["two-tenants", "errored-reply", "single-tenant"])
def test_the_credit_role_scores_a_drive_with_no_baseline_and_the_passthrough_verdict(tmp_path, task, trace):
    record = measure(task, _drive_dir(tmp_path, trace))
    # The task's scorer over the full trace, against the passthrough verdict, with no baseline
    # credit: the summary the role reads scores the same.
    want = asyncio.run(task.scorer.score(task.effective_tenants(), trace, passthrough_verdict(trace), task.hardware))
    assert want.gate_passed and want.breakdown["baseline_log_credit"] is None
    assert record.mean_log_credit == task.scorer.measured_credit(want)
    assert record.n_units == task.scorer.measured_units(want)
    assert record.n_requests == len(trace.events)
    assert record.score == json.loads(want.model_dump_json())
    assert record.image_sha256 == SIF_SHA


def test_an_errored_reply_counts_against_the_drives_attainment(tmp_path):
    """What the passthrough verdict changes: a reply the driver marked errored is no unit met,
    where a verdict that passed every event would count it."""
    trace = _errored_reply_trace()
    record = measure(latency_task(), _drive_dir(tmp_path, trace))
    task = latency_task()
    trusting = Verdict(passed=True, per_event={ev.event_key: True for ev in trace.events})
    trusted = asyncio.run(task.scorer.score(task.effective_tenants(), trace, trusting, task.hardware))
    assert record.score["breakdown"]["slo_attainment"] < trusted.breakdown["slo_attainment"]


def test_a_drive_is_dated_from_its_drivers_start_less_its_readiness_and_lasts_both(tmp_path):
    record = measure(latency_task(), _drive_dir(tmp_path, drive_trace(failures=False, tenants=LATENCY_TENANTS)))
    assert record.started_at == datetime.fromtimestamp(99.5, timezone.utc).isoformat()
    assert record.duration_sec == pytest.approx(1.0 + 99.0)


@pytest.mark.parametrize("task, trace, outcome, match", [
    (goodput_task(), drive_trace(failures=False), "served", "scores against no baseline"),
    (latency_task().model_copy(update={"baseline": None}), drive_trace(failures=False, tenants=LATENCY_TENANTS),
     "served", "declares no baseline server"),
    (latency_task(), Trace(), "crashed", "did not serve"),
    (latency_task(), Trace(), "served", "sent no request"),
    (latency_task(), drive_trace(failures=True, tenants=LATENCY_TENANTS), "served", "scored zero"),
], ids=["no-baseline-scorer", "no-baseline-server", "not-served", "empty", "zero"])
def test_the_credit_role_refuses_a_drive_that_did_not_serve(tmp_path, task, trace, outcome, match):
    """A drive no baseline credit may pool."""
    with pytest.raises(Refused, match=match):
        measure(task, _drive_dir(tmp_path, trace, outcome=outcome))


@pytest.mark.parametrize("missing", ["credit", "unit count"])
def test_a_drive_whose_scorer_measures_no_credit_or_no_unit_count_is_refused(tmp_path, monkeypatch, missing):
    """A drive whose score names no L, or no count of the units drives are pooled by, cannot be
    pooled: the role refuses it, since a retry would score the same drive."""
    monkeypatch.setattr(SloLatencyScorer, "measured_credit" if missing == "credit" else "measured_units",
                        lambda self, score: None)
    with pytest.raises(Refused, match=f"measured no {missing} on the drive"):
        measure(latency_task(), _drive_dir(tmp_path, drive_trace(failures=False, tenants=LATENCY_TENANTS)))


def _ref_file(tmp_path: Path, module: str, overrides: dict | None = None) -> Path:
    path = tmp_path / "task-ref.json"
    path.write_text(TaskRef(module=module, overrides=overrides or {}).model_dump_json())
    return path


def test_the_role_writes_and_reports_its_record(tmp_path, checkout):
    drive = _drive_dir(tmp_path, drive_trace(failures=False, tenants=LATENCY_TENANTS))
    out, report = tmp_path / "out", tmp_path / "outputs.json"
    out.mkdir()
    code = credit_role.main(["--task-ref", str(_ref_file(tmp_path, "cr_latency.task")), "--drive", str(drive),
                             "--out", str(out), "--outputs-out", str(report)])
    assert code == 0
    written = CreditDrive.model_validate_json((out / CREDIT_DRIVE_FILE).read_bytes())
    assert written == measure(latency_task(), drive)
    assert list(read_outputs_report(report)) == [CREDIT_DRIVE_FILE]


def test_the_roles_refusal_exits_3_with_one_reason_line(tmp_path, checkout, capsys):
    drive = _drive_dir(tmp_path, drive_trace(failures=True, tenants=LATENCY_TENANTS))
    out = tmp_path / "out"
    out.mkdir()
    code = credit_role.main(["--task-ref", str(_ref_file(tmp_path, "cr_latency.task")), "--drive", str(drive),
                             "--out", str(out), "--outputs-out", str(tmp_path / "outputs.json")])
    assert code == REFUSED_EXIT
    [line] = [x for x in capsys.readouterr().err.splitlines() if x.startswith(REFUSED_PREFIX)]
    assert "scored zero" in line
    assert not (out / CREDIT_DRIVE_FILE).exists()


def test_the_role_loads_the_task_without_its_baseline_credit(tmp_path, checkout, monkeypatch):
    """The role measures L with no B: it loads the task as its module declares it, never with
    the committed credit, which the module's directory may not even hold."""
    seen = []
    load = credit_role.load_task

    def spy(ref, **kw):
        seen.append(kw)
        return load(ref, **kw)

    monkeypatch.setattr(credit_role, "load_task", spy)
    drive = _drive_dir(tmp_path, drive_trace(failures=False, tenants=LATENCY_TENANTS))
    out = tmp_path / "out"
    out.mkdir()
    assert credit_role.main(["--task-ref", str(_ref_file(tmp_path, "cr_latency.task")), "--drive", str(drive),
                             "--out", str(out), "--outputs-out", str(tmp_path / "outputs.json")]) == 0
    assert seen == [{}]
    written = CreditDrive.model_validate_json((out / CREDIT_DRIVE_FILE).read_bytes())
    assert written.score["breakdown"]["baseline_log_credit"] is None


def test_the_role_scores_the_task_its_overrides_make(tmp_path, checkout):
    """A credit measured under ``--task.*`` overrides is scored against the load they set, never
    the module's own: the role loads the task its reference names, overrides and all."""
    drive = _drive_dir(tmp_path, t0_trace(failures=False))
    out = tmp_path / "out"
    out.mkdir()
    assert credit_role.main(["--task-ref", str(_ref_file(tmp_path, "cr_single.task", LOOSER)), "--drive", str(drive),
                             "--out", str(out), "--outputs-out", str(tmp_path / "outputs.json")]) == 0
    written = CreditDrive.model_validate_json((out / CREDIT_DRIVE_FILE).read_bytes())
    assert written == measure(apply_task_overrides(t0_single_task(), LOOSER), drive)
    authored = measure(t0_single_task(), drive)
    assert written.mean_log_credit == pytest.approx(authored.mean_log_credit + math.log(2))
