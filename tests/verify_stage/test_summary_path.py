"""The verify role decides from a trace's summary whether any request failed. When one did, it
writes the failed verdict from the summary alone and never reads a chunk, so a trace past its
cap is never held whole; every file it writes is the one the whole-trace verification writes for
the same trace. When none did, it reads the whole trace and grades it."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hls.core.faults import HarnessFault
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import Trace, TraceEvent
from hls.spec.tenant import Tenant
from hls.store.manifest import read_outputs_report
from hls.store.posix import PosixStore
from hls.taskref import MeasuredPaths, TaskRef, read_task
from hls.trace.store import SUMMARY_FILE, TraceIncomplete, read_end, read_trace, write_trace
from hls.verify import LetterMatchMetric, MetricVerifier
from hls.verify.answers import StoreAnswers
from hls.verify.base import WiringError
from hls.verify.role import VERIFY_FILE, calibration_blob, verify, verify_stored
from tests.trace.traces import drive_trace
from tests.verify_stage.conftest import default_calibration
from tests.verify_stage.fixture import _gate, _task, _workload, event, gate_trace, metric_task, single_trace

#: The role program's steps with the whole trace read: the task loaded with its calibration, every
#: event read (``read_trace``), ``verify`` writing the verdict and record, every file reported.
WHOLE_TRACE_ROLE = """
import asyncio, sys
from pathlib import Path
from hls.store.manifest import report_outputs
from hls.store.posix import PosixStore
from hls.taskref import MeasuredPaths, TaskRef, read_task
from hls.trace.store import read_trace
from hls.verify.answers import StoreAnswers
from hls.verify.role import calibration_blob, verify
ref_file, trace_dir, out, report, area, calibration = sys.argv[1:7]
ref = TaskRef.model_validate_json(Path(ref_file).read_bytes())
loaded = read_task(ref, calibration=True, files=MeasuredPaths(calibration=Path(calibration) if calibration else None))
answers = (StoreAnswers(PosixStore(Path(area)), gpu_kind="B200", image="a" * 64, writer="g1+verify+1")
           if area else None)
asyncio.run(verify(loaded.task, read_trace(Path(trace_dir)), out=Path(out), eval_id="terminal", answers=answers,
                   calibration=calibration_blob(loaded)))
report_outputs(Path(report), Path(out), *sorted(Path(out).iterdir()))
"""


def _store(trace: Trace, root: Path) -> Path:
    out = root / "trace"
    write_trace(trace, out, ending="aborted", driver={"class": "fixture"})
    return out


def _spoil_chunks(trace_dir: Path) -> None:
    """Replace every chunk with bytes that are no chunk, under its own name, so a reader of the
    chunks fails and a reader of the summary alone does not."""
    for chunk in read_end(trace_dir).chunks:
        (trace_dir / chunk.name).write_bytes(b"not a chunk")
    with pytest.raises(TraceIncomplete, match="chunk"):
        read_trace(trace_dir)


def _files(out: Path) -> dict[str, bytes]:
    """Every file under *out*, by its path under it, to its bytes."""
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.rglob("*")) if p.is_file()}


def _unstamped(request_id: str, *, error: str | None) -> TraceEvent:
    """A request with no tenant stamped, that failed with *error* (None: no response and no error)."""
    return TraceEvent(request_id=request_id, input=Envelope(request_id=request_id, contents=[TextPayload(text="Q?")]),
                      response=None, error=error, t_send=0.0)


def _answered_with_error(tenant: str, request_id: str, error: str) -> TraceEvent:
    """A reply that arrived and was then failed with *error*."""
    ev = event(tenant, request_id, "B", "B")
    return ev.model_copy(update={"error": error})


def _no_response(tenant: str, request_id: str) -> TraceEvent:
    """A request with no response and no error."""
    ev = event(tenant, request_id, "B", "B")
    return ev.model_copy(update={"response": None, "t_recv": None})


def _drive_task():
    """The tenants of ``drive_trace``, each under one pooled token gate and its own letter match."""
    gate = _gate()
    return _task("verify-stage-drive", tenants=[
        Tenant(id=t, workload=_workload(), verifiers=[gate, MetricVerifier(metric=LetterMatchMetric())])
        for t in ("chat", "agent", "plain")])


def _mixed_failures() -> Trace:
    """Every way a request fails, around answered replies: an error and no response, no response
    and no error, a reply with an error, and two requests with no tenant stamped. Error strings
    repeat unevenly, so the census orders them by count."""
    return Trace(events=[event("t0", "r0", "B", "B"), event("t0", "r1", "A", "B", errored=True),
                         _no_response("t0", "r2"), _answered_with_error("t0", "r3", "late: reply after the deadline"),
                         event("t0", "r4", "B", "B"), event("t0", "r5", "B", "B", errored=True),
                         _unstamped("r6", error=None), _unstamped("r7", error="transport")])


def _all_failed() -> Trace:
    return Trace(events=[event("t0", f"r{i}", "B", "B", errored=True) for i in range(3)])


#: Each case: the task, the failing trace, and the fixture module whose task (with its
#: calibration) is used instead of the task, or None.
CASES = {
    "gate": (None, gate_trace(errored=True), "vstage_gate.task"),
    "drive": (_drive_task, drive_trace(failures=True), None),
    "mixed": (metric_task, _mixed_failures(), None),
    "all-failed": (metric_task, _all_failed(), None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_failing_trace_is_verified_from_its_summary_as_the_whole_trace_is(checkout, tmp_path, case):
    build, trace, module = CASES[case]
    if module is not None:
        ref = TaskRef(module=module)
        loaded = read_task(ref, calibration=True,
                           files=MeasuredPaths(calibration=default_calibration(checkout, module)))
        task, calibration = loaded.task, calibration_blob(loaded)
    else:
        task, calibration = build(), None
    trace_dir = _store(trace, tmp_path)
    whole, summary = tmp_path / "whole", tmp_path / "summary"
    whole.mkdir()
    summary.mkdir()
    by_whole = asyncio.run(verify(task, read_trace(trace_dir), out=whole, eval_id="e3", answers=None,
                                  calibration=calibration))
    _spoil_chunks(trace_dir)
    by_summary = asyncio.run(verify_stored(task, trace_dir, out=summary, eval_id="e3", answers=None,
                                           calibration=calibration))
    assert by_summary == by_whole and by_whole.engines == []
    assert _files(summary) == _files(whole)
    assert sorted(_files(whole)) == ["verdict.json", VERIFY_FILE]


def test_a_failing_trace_is_verified_from_its_summary_with_an_answers_area_that_is_never_read(checkout, tmp_path):
    trace = Trace(events=[event("t0", "r0", "B", "B"), event("t0", "r1", "B", "B", errored=True)])
    trace_dir = _store(trace, tmp_path)
    task = read_task(TaskRef(module="vstage_cached.task"), calibration=True,
                     files=MeasuredPaths(calibration=default_calibration(checkout, "vstage_cached.task"))).task
    outs = {}
    for way in ("whole", "summary"):
        area = tmp_path / way / "area"
        answers = StoreAnswers(PosixStore(area), gpu_kind="B200", image="a" * 64, writer="g1+verify+1")
        out = tmp_path / way / "out"
        out.mkdir(parents=True)
        if way == "whole":
            asyncio.run(verify(task, read_trace(trace_dir), out=out, eval_id="terminal", answers=answers))
            _spoil_chunks(trace_dir)
        else:
            asyncio.run(verify_stored(task, trace_dir, out=out, eval_id="terminal", answers=answers))
        assert (answers.read, answers.written) == ({}, {})
        assert not area.exists() or not any(area.rglob("*"))
        outs[way] = _files(out)
    assert outs["summary"] == outs["whole"]


def _role(checkout: Path, module: str, trace_dir: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    """Run the role as a program, handed the default calibration of *module*, as ``test_role``
    does."""
    out.mkdir(parents=True)
    ref = out.parent / "task-ref.json"
    ref.write_text(TaskRef(module=module).model_dump_json())
    calibration = default_calibration(checkout, module)
    given = ["--calibration", str(calibration)] if calibration is not None else []
    env = {**os.environ, "PYTHONPATH": f"{checkout / 'src'}:{checkout}"}
    return subprocess.run([sys.executable, "-m", "hls.verify.role", "--task-ref", str(ref), "--trace", str(trace_dir),
                           "--out", str(out), "--outputs-out", str(out.parent / "outputs.json"), *given, *extra],
                          env=env, cwd=out.parent, capture_output=True, text=True, timeout=120)


def _whole_trace_role(checkout: Path, module: str, trace_dir: Path, out: Path, area: Path | None
                      ) -> subprocess.CompletedProcess:
    """Run ``WHOLE_TRACE_ROLE`` in a process of its own, as the role's program runs."""
    out.mkdir(parents=True)
    ref = out.parent / "task-ref.json"
    ref.write_text(TaskRef(module=module).model_dump_json())
    calibration = default_calibration(checkout, module)
    env = {**os.environ, "PYTHONPATH": f"{checkout / 'src'}:{checkout}"}
    return subprocess.run([sys.executable, "-c", WHOLE_TRACE_ROLE, str(ref), str(trace_dir), str(out),
                           str(out.parent / "outputs.json"), "" if area is None else str(area),
                           "" if calibration is None else str(calibration)], env=env,
                          cwd=out.parent, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize("module, trace, cached", [
    ("vstage_gate.task", gate_trace(errored=True), False),
    ("vstage_cached.task", Trace(events=[*single_trace().events, event("t0", "r2", "B", "B", errored=True)]), True),
], ids=["gate", "cached"])
def test_the_role_writes_and_reports_a_failing_traces_files_without_reading_a_chunk(checkout, tmp_path, module,
                                                                                    trace, cached):
    """The role program, in a process of its own, writes and reports the bytes the whole-trace
    verification writes in one of its own, ``verify.json``'s libraries included: the summary
    path loads no fewer libraries. Its chunks are spoiled first, so the role read none."""
    trace_dir = _store(trace, tmp_path)
    whole, summary = tmp_path / "whole" / "out", tmp_path / "summary" / "out"
    area = (tmp_path / "whole" / "area") if cached else None
    done = _whole_trace_role(checkout, module, trace_dir, whole, area)
    assert done.returncode == 0, done.stderr
    _spoil_chunks(trace_dir)
    extra = ("--answers", str(tmp_path / "summary" / "area"), "--writer", "g1+verify+1", "--gpu-kind", "B200",
             "--reference-image", "a" * 64) if cached else ()
    done = _role(checkout, module, trace_dir, summary, *extra)
    assert done.returncode == 0, done.stderr
    assert _files(summary) == _files(whole)
    assert (summary.parent / "outputs.json").read_bytes() == (whole.parent / "outputs.json").read_bytes()
    report = read_outputs_report(summary.parent / "outputs.json")
    assert report is not None and sorted(report) == ["verdict.json", VERIFY_FILE]
    failed = sum(ev.response is None or ev.error is not None for ev in trace.events)
    assert f"verification skipped: {failed} of {len(trace.events)} requests failed" in done.stderr


def test_a_trace_with_no_failed_request_is_read_whole(checkout, tmp_path):
    trace_dir = _store(single_trace(), tmp_path)
    _spoil_chunks(trace_dir)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(TraceIncomplete, match="chunk"):
        asyncio.run(verify_stored(metric_task(), trace_dir, out=out, eval_id="terminal", answers=None))
    assert list(out.iterdir()) == []


def test_a_failing_trace_whose_summary_does_not_match_its_end_is_refused(checkout, tmp_path):
    trace_dir = _store(gate_trace(errored=True), tmp_path)
    (trace_dir / SUMMARY_FILE).write_bytes(b"not a summary")
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(TraceIncomplete, match="summary"):
        asyncio.run(verify_stored(metric_task(), trace_dir, out=out, eval_id="terminal", answers=None))
    assert list(out.iterdir()) == []


def _split_gate_task():
    return _task("verify-stage-split", tenants=[
        Tenant(id="a", workload=_workload(), verifiers=[_gate(allow_underpowered=True)]),
        Tenant(id="b", workload=_workload(), verifiers=[_gate(allow_underpowered=False)])])


@pytest.mark.parametrize("build, trace, raised", [
    (_split_gate_task, gate_trace(errored=True), WiringError),
    (metric_task, Trace(events=[]), HarnessFault),
], ids=["split-gate", "empty-trace"])
def test_a_refusal_of_the_whole_trace_is_the_summarys_too(checkout, tmp_path, build, trace, raised):
    trace_dir = _store(trace, tmp_path)
    whole, summary = tmp_path / "whole", tmp_path / "summary"
    whole.mkdir()
    summary.mkdir()
    with pytest.raises(raised) as by_whole:
        asyncio.run(verify(build(), read_trace(trace_dir), out=whole, eval_id="terminal", answers=None))
    if read_end(trace_dir).chunks:
        _spoil_chunks(trace_dir)
    with pytest.raises(raised) as by_summary:
        asyncio.run(verify_stored(build(), trace_dir, out=summary, eval_id="terminal", answers=None))
    assert str(by_summary.value) == str(by_whole.value)
    assert list(whole.iterdir()) == list(summary.iterdir()) == []


def test_a_failing_trace_is_not_written_into_an_out_directory_that_holds_a_file(checkout, tmp_path):
    trace_dir = _store(gate_trace(errored=True), tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "verdict.json").write_bytes(b"{}")
    with pytest.raises(RuntimeError, match=r"already holds \['verdict.json'\]"):
        asyncio.run(verify_stored(metric_task(), trace_dir, out=out, eval_id="terminal", answers=None))
    assert sorted(p.name for p in out.iterdir()) == ["verdict.json"]
