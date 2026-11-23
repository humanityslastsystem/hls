"""Per-tenant verify dispatch.

Two-tenant trace: tenant "a"'s events pass its verifiers; tenant "b"'s fail.
Assert:
  - Each tenant's verifiers see ONLY its own slice.
  - Aggregate verdict reflects per-tenant pass/fail (a passed, b failed -> aggregate failed).
"""
from __future__ import annotations

import asyncio
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.tenant import Tenant
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify.base import RolloutMetric
from hls.verify import LetterMatchMetric, MetricVerifier
from hls.verify.inprocess import EngineHost
from hls.verify.run import verify_trace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wl() -> Workload:
    return Workload(
        arrival=AllAtOnce(),
        inter_turn_delay=NoDelay(),
        output_length_cap=Uncapped(),
        slo=NoSLO(),
    )


def _ev(req: str, text: str, tenant_id: str, gt: str | None = None) -> TraceEvent:
    """Build a TraceEvent stamped with `tenant_id` in input.metadata."""
    inp = Envelope(
        request_id=req,
        contents=[TextPayload(text="Q?")],
        metadata={"tenant_id": tenant_id},
    )
    out = Envelope(request_id=req, contents=[TextPayload(text=text)])
    return TraceEvent(
        request_id=req,
        input=inp,
        response=ArtifactResponse(output=out),
        ground_truth=gt,
        t_send=0.0,
        t_recv=1.0,
    )


# A verifier that records which request_ids it actually saw.
_seen_ids: list[str] = []


class _SpyMetric(RolloutMetric):
    """Records request_ids it scores; passes every event."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        for ev in events:
            _seen_ids.append(ev.request_id)
        pe = {ev.request_id: PerEventVerdict(passed=True, score=1.0) for ev in events}
        return MetricResult(value=1.0, per_event=pe)


class _FailMetric(RolloutMetric):
    """Always fails every event."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        pe = {ev.request_id: PerEventVerdict(passed=False, score=0.0) for ev in events}
        return MetricResult(value=0.0, per_event=pe)


def _contract() -> RequestResponseContract:
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
    )


def _two_tenant_task() -> TaskSpec:
    """Task with tenants 'a' (spy/pass) and 'b' (fail)."""
    tenant_a = Tenant(
        id="a",
        workload=_wl(),
        verifiers=[MetricVerifier(metric=_SpyMetric(), min_value=1.0)],
    )
    tenant_b = Tenant(
        id="b",
        workload=_wl(),
        verifiers=[MetricVerifier(metric=_FailMetric(), min_value=1.0)],
    )
    return TaskSpec(
        id="T_multi",
        model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=_contract(),
        tenants=[tenant_a, tenant_b],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
    )


def _two_tenant_trace() -> Trace:
    """Three events: two for tenant 'a', one for tenant 'b'."""
    return Trace(events=[
        _ev("a0", "X", tenant_id="a", gt="X"),
        _ev("a1", "Y", tenant_id="a", gt="Y"),
        _ev("b0", "Z", tenant_id="b", gt="Z"),
    ])


def _run_verify(task: TaskSpec, trace: Trace):
    async def _go():
        async with EngineHost() as host:
            return await verify_trace(task, trace, eval_dir=None, host=host, answers=None)
    return asyncio.run(_go())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tenant_a_verifiers_see_only_a_events():
    """The spy verifier for tenant 'a' must not receive tenant 'b' events."""
    _seen_ids.clear()
    task = _two_tenant_task()
    trace = _two_tenant_trace()
    _run_verify(task, trace)
    # The spy verifier ran for tenant 'a': it should see only a0 and a1.
    assert "a0" in _seen_ids
    assert "a1" in _seen_ids
    assert "b0" not in _seen_ids


def test_aggregate_verdict_fails_when_any_tenant_fails():
    """Tenant 'a' passes its verifier; tenant 'b' fails -> aggregate must be False."""
    task = _two_tenant_task()
    trace = _two_tenant_trace()
    verdict, _ = _run_verify(task, trace)
    assert verdict.passed is False


def test_tenant_a_events_pass_in_per_event():
    """Events owned by tenant 'a' must be True in the aggregate per_event map."""
    task = _two_tenant_task()
    trace = _two_tenant_trace()
    verdict, _ = _run_verify(task, trace)
    # Run-level per_event keys on event_key ("{tenant_id}:{request_id}").
    assert verdict.per_event["a:a0"] is True
    assert verdict.per_event["a:a1"] is True


def test_tenant_b_event_fails_in_per_event():
    """Events owned by tenant 'b' must be False in the aggregate per_event map."""
    task = _two_tenant_task()
    trace = _two_tenant_trace()
    verdict, _ = _run_verify(task, trace)
    assert verdict.per_event["b:b0"] is False


def test_per_verifier_results_marks_foreign_tenant_cells_not_applicable():
    """per_verifier_results is rectangular over ALL events, but each verifier only
    saw its OWN tenant's slice. A foreign-tenant cell must be NOT-APPLICABLE
    (passed=None) -- distinct from an event the verifier saw and failed (passed=False)."""
    task = _two_tenant_task()
    trace = _two_tenant_trace()          # events [a0, a1, b0]; tenant a passes, b fails
    _verdict, pvr = _run_verify(task, trace)

    # Rectangular, column-aligned to trace.events; one row per verifier (tenant order).
    assert len(pvr) == 2 and all(len(row) == len(trace.events) for row in pvr)
    col = {ev.request_id: i for i, ev in enumerate(trace.events)}
    spy_row, fail_row = pvr[0], pvr[1]    # row 0 = tenant 'a' (spy/pass), row 1 = 'b' (fail)

    # Own-tenant cells keep their real verdict.
    assert spy_row[col["a0"]]["passed"] is True
    assert spy_row[col["a1"]]["passed"] is True
    assert fail_row[col["b0"]]["passed"] is False

    # Foreign-tenant cells are NOT-APPLICABLE (None), never False.
    assert spy_row[col["b0"]]["passed"] is None
    assert fail_row[col["a0"]]["passed"] is None
    assert fail_row[col["a1"]]["passed"] is None
    # ...and carry no rationale/diagnostics for an event the verifier never saw.
    na = spy_row[col["b0"]]
    assert na["rationale"] is None and na["diagnostics"] == {}


def test_same_request_id_across_tenants_no_overwrite_in_verify():
    """Two tenants over the same dataset path share request_id "r0". The run-level
    per_event keys on event_key, so a's "r0" (spy/pass) and b's "r0" (fail) are BOTH
    recorded ("a:r0" True, "b:r0" False), where a request_id-keyed map would let one
    overwrite the other. per_verifier_results stays correct too: each verifier's own
    "r0" keeps its real verdict while the OTHER tenant's "r0" is not-applicable."""
    task = _two_tenant_task()            # a = spy/pass, b = fail
    trace = Trace(events=[
        _ev("r0", "X", tenant_id="a", gt="X"),
        _ev("r0", "Z", tenant_id="b", gt="Z"),
    ])
    verdict, pvr = _run_verify(task, trace)
    assert verdict.per_event["a:r0"] is True
    assert verdict.per_event["b:r0"] is False
    # Columns are positional over trace.events; index by (tenant_id, request_id)
    # because the bare request_id "r0" is shared.
    col = {(ev.tenant_id, ev.request_id): i for i, ev in enumerate(trace.events)}
    a_idx, b_idx = col[("a", "r0")], col[("b", "r0")]
    spy_row, fail_row = pvr[0], pvr[1]   # row 0 = tenant 'a', row 1 = 'b'
    assert spy_row[a_idx]["passed"] is True
    assert spy_row[b_idx]["passed"] is None      # foreign cell, not False
    assert fail_row[b_idx]["passed"] is False
    assert fail_row[a_idx]["passed"] is None


class _GuardedMetric(RolloutMetric):
    """Mirrors the shipped metrics (gsm8k/hotpot/coqa): raises on an all-unlabeled
    batch. With an EMPTY batch, ``all([]) == True`` trips the guard -- the crash an
    empty per-tenant slice would trigger."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        from hls.core.faults import HarnessFault
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("guarded metric needs ground-truth; the dataset provides none")
        pe = {ev.request_id: PerEventVerdict(passed=True, score=1.0) for ev in events}
        return MetricResult(value=1.0, per_event=pe)


def _errored_ev(req: str, tenant_id: str) -> TraceEvent:
    """An errored event: no response, error string set (filtered out of the
    tenant's 'good' slice)."""
    inp = Envelope(
        request_id=req,
        contents=[TextPayload(text="Q?")],
        metadata={"tenant_id": tenant_id},
    )
    return TraceEvent(
        request_id=req,
        input=inp,
        response=None,
        error="boom",
        t_send=0.0,
        t_recv=None,
    )


def test_one_fully_errored_tenant_does_not_crash_whole_run():
    """One tenant whose events ALL errored must not blow up the whole run.

    The fast path returns a clean failed_requests verdict without grading
    anything -- no HarnessFault/ZeroDivisionError from an empty slice."""
    tenant_a = Tenant(
        id="a",
        workload=_wl(),
        verifiers=[MetricVerifier(metric=_GuardedMetric(), min_value=1.0)],
    )
    tenant_b = Tenant(
        id="b",
        workload=_wl(),
        verifiers=[MetricVerifier(metric=_GuardedMetric(), min_value=1.0)],
    )
    task = TaskSpec(
        id="T_multi_err",
        model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=_contract(),
        tenants=[tenant_a, tenant_b],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
    )
    trace = Trace(events=[
        _ev("a0", "X", tenant_id="a", gt="X"),
        _ev("a1", "Y", tenant_id="a", gt="Y"),
        _errored_ev("b0", tenant_id="b"),
        _errored_ev("b1", tenant_id="b"),
    ])
    verdict, _ = _run_verify(task, trace)   # must NOT raise
    assert verdict.passed is False
    assert verdict.failure_mode == "failed_requests"
    # Fast path: nothing was graded, every event is unpassed.
    assert all(v is False for v in verdict.per_event.values())


def test_single_tenant_verify_unchanged():
    """Single-tenant task (one 't0' group) reduces to today's path: same verdict."""
    # Use a trivially-passing MetricVerifier so the verdict is True.
    task = TaskSpec(
        id="T_single",
        model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=_contract(),
        workload=_wl(),
        verifiers=[MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
    )
    # Single-tenant events are stamped tenant_id="t0" (as the driver does); the
    # runner reads that slice via Trace.events_for("t0").
    inp = Envelope(request_id="r0", contents=[TextPayload(text="Q?")],
                   metadata={"tenant_id": "t0"})
    out = Envelope(request_id="r0", contents=[TextPayload(text="B")])
    ev = TraceEvent(
        request_id="r0",
        input=inp,
        response=ArtifactResponse(output=out),
        ground_truth="B",
        t_send=0.0,
        t_recv=1.0,
    )
    trace = Trace(events=[ev])
    verdict, results = _run_verify(task, trace)
    assert verdict.passed is True
    # Run-level per_event keys on event_key; single tenant "t0" -> "t0:r0".
    assert verdict.per_event["t0:r0"] is True
    # Single tenant: every event is own -> no not-applicable cells (matrix unchanged).
    assert results[0][0]["passed"] is True
