"""A trace whose every request was answered is verified from the store. The verify role hands the
verification core a view of the stored trace (``hls.trace.store.StoredTrace``), which reads the
trace's events from the store on each pass and holds none of them; the core does its own
bookkeeping from the trace's summary. Every file the role writes is the one the verification of
the same trace held in memory writes."""

from __future__ import annotations

import asyncio
import tracemalloc
from pathlib import Path

import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace
from hls.taskref import MeasuredPaths, TaskRef, read_task
from hls.trace import store as store_mod
from hls.trace.store import StoredTrace, write_trace
from hls.verify.inprocess import EngineHost
from hls.verify.role import calibration_blob, verify, verify_stored
from hls.verify.run import verify_trace
from hls.spec.tenant import Tenant
from hls.verify import LetterMatchMetric, MetricVerifier
from tests.verify_stage.conftest import default_calibration
from tests.verify_stage.fixture import _task, _workload, event, gate_trace, metric_task, single_trace
from tests.verify_stage.test_summary_path import _files


def _store(trace: Trace, root: Path) -> Path:
    out = root / "trace"
    write_trace(trace, out, ending="drained", driver={"class": "fixture"}, chunk_bytes=2_000)
    return out


def _tenants_task():
    """Tenants a and b, each graded by its own letter match."""
    return _task("verify-stage-tenants", tenants=[
        Tenant(id=t, workload=_workload(), verifiers=[MetricVerifier(metric=LetterMatchMetric())]) for t in "ab"])


#: Each case: the task, the trace whose every request was answered, and the fixture module whose
#: task (with its calibration) is used instead of the task, or None.
CASES = {
    "gate": (None, gate_trace(), "vstage_gate.task"),
    "tenants": (_tenants_task, gate_trace(), None),
    "metric": (metric_task, single_trace(5), None),
}


@pytest.mark.parametrize("case", list(CASES))
def test_a_passing_trace_is_verified_from_the_store_as_it_is_in_memory(checkout, tmp_path, monkeypatch, case):
    build, trace, module = CASES[case]
    if module is not None:
        ref = TaskRef(module=module)
        loaded = read_task(ref, calibration=True,
                           files=MeasuredPaths(calibration=default_calibration(checkout, module)))
        task, calibration = loaded.task, calibration_blob(loaded)
    else:
        task, calibration = build(), None
    trace_dir = _store(trace, tmp_path)
    memory, stored = tmp_path / "memory", tmp_path / "stored"
    memory.mkdir()
    stored.mkdir()
    by_memory = asyncio.run(verify(task, trace, out=memory, eval_id="e1", answers=None, calibration=calibration))

    def refuse(*args, **kwargs):
        raise AssertionError("the verify role read the whole trace into memory")

    monkeypatch.setattr(store_mod, "read_trace", refuse)
    by_store = asyncio.run(verify_stored(task, trace_dir, out=stored, eval_id="e1", answers=None,
                                         calibration=calibration))
    assert by_store == by_memory
    assert _files(stored) == _files(memory)


def test_the_verification_core_holds_no_trace_of_its_own(tmp_path):
    """Forty replies of a MiB each, graded by a letter match, which keeps a few small fields of
    each reply: verifying them from the store holds a few of the replies at most."""
    def big(i: int):
        ev = event("t0", f"r{i}", "B", "B")
        out = Envelope(request_id=ev.request_id, contents=[TextPayload(text="B" + " x" * 2**19, tokens=[42, 43])])
        return ev.model_copy(update={"response": ArtifactResponse(output=out)})

    trace_dir = _store(Trace(events=[big(i) for i in range(40)]), tmp_path)

    async def run():
        async with EngineHost() as host:
            return await verify_trace(metric_task(), StoredTrace(trace_dir), eval_dir=None, host=host, answers=None)

    tracemalloc.start()
    try:
        verdict, results = asyncio.run(run())
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert verdict.passed and len(results[0]) == 40
    assert peak < 10 * 2**20
