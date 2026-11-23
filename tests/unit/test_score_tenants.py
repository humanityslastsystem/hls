"""Per-tenant score gating.

Two-tenant scenario: tenant 'a' passes its SLO + verdict (events credited);
tenant 'b' fails its SLO (0 credit).  Total goodput == a's credited events only.
"""
from __future__ import annotations

import asyncio

from hls.core.model import AcceleratorSpec, Hardware
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.core.verdict import Verdict
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit, Unit
from hls.spec.tenant import Tenant
from hls.spec.workload import (
    AllAtOnce, NoDelay, NoSLO, SLO, SLOResult, Uncapped, Workload
)
from hls.verify import LetterMatchMetric, MetricVerifier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hw() -> Hardware:
    return Hardware(accelerator=AcceleratorSpec(kind="B200", count=1))


def _ev(req: str, tenant_id: str, t_send: float = 0.0, t_recv: float = 1.0) -> TraceEvent:
    inp = Envelope(
        request_id=req,
        contents=[TextPayload(text="Q?")],
        metadata={"tenant_id": tenant_id},
    )
    out = Envelope(request_id=req, contents=[TextPayload(text="X")])
    return TraceEvent(
        request_id=req,
        input=inp,
        response=ArtifactResponse(output=out),
        ground_truth="X",
        t_send=t_send,
        t_recv=t_recv,
    )


def _wl(slo: object) -> Workload:
    return Workload(
        arrival=AllAtOnce(),
        inter_turn_delay=NoDelay(),
        output_length_cap=Uncapped(),
        slo=slo,
    )


class _FailingGate(SLO):
    """Minimal binary aggregate-gate SLO: the gate always fails, zeroing the tenant's credit -- keeps the scorer's
    aggregate-gate machinery (any_slo_failure -> zero_reasons=("slo",)) covered."""

    def apply(self, trace, workload) -> SLOResult:
        return SLOResult(aggregate_passed=False, per_event_credit={},
                         diagnostics={"p99_ms": 999.0, "target_ms": 1.0})


def _loose_wl() -> Workload:
    return _wl(NoSLO())                # always passes

def _tight_wl() -> Workload:
    return _wl(_FailingGate())         # aggregate gate fails


def _verifiers():
    return [MetricVerifier(metric=LetterMatchMetric(), min_value=0.0)]


def _scorer() -> GoodputScorer:
    return GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))


def _two_tenant_trace() -> Trace:
    return Trace(events=[
        _ev("a0", tenant_id="a"),
        _ev("a1", tenant_id="a"),
        _ev("b0", tenant_id="b"),
    ])


def _verdict(trace: Trace, ok: bool = True) -> Verdict:
    """A run-level Verdict keyed by event_key (tenant-namespaced), as the Runner
    builds it -- so every event maps to its OWN tenant's pass/fail even when two
    tenants share a request_id. Here every event shares the aggregate outcome `ok`."""
    return Verdict(passed=ok, per_event={ev.event_key: ok for ev in trace.events},
                   diagnostics={})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_tenant_b_slo_failure_zeros_b_credit():
    """Tenant 'a' passes SLO; tenant 'b' fails -> b's events get 0 credit.
    Total goodput comes only from a's 2 events."""
    tenant_a = Tenant(id="a", workload=_loose_wl(), verifiers=_verifiers())
    tenant_b = Tenant(id="b", workload=_tight_wl(), verifiers=_verifiers())
    tenants = [tenant_a, tenant_b]

    trace = _two_tenant_trace()
    verdict = _verdict(trace)
    scorer = _scorer()

    score = asyncio.run(scorer.score(tenants, trace, verdict, _hw()))

    # Gate should still pass (tenant a passes)
    assert score.gate_passed is True
    # achieved is counted / duration; only 2 events credited (a0, a1)
    # duration = max(t_recv) - min(t_send) = 1.0 - 0.0 = 1.0
    # achieved = 2 / 1.0 = 2.0; sol = 1.0; value = 2.0
    assert score.achieved == 2.0


def test_both_tenants_pass_credits_all_events():
    """Both tenants pass SLO + verdict -> all 3 events credited."""
    tenant_a = Tenant(id="a", workload=_loose_wl(), verifiers=_verifiers())
    tenant_b = Tenant(id="b", workload=_loose_wl(), verifiers=_verifiers())
    tenants = [tenant_a, tenant_b]

    trace = _two_tenant_trace()
    verdict = _verdict(trace)
    scorer = _scorer()

    score = asyncio.run(scorer.score(tenants, trace, verdict, _hw()))
    assert score.gate_passed is True
    # All 3 events credited: achieved = 3 / 1.0 = 3.0
    assert score.achieved == 3.0


def test_verdict_fail_zeros_score_regardless_of_slo():
    """A failed aggregate verdict zeroes the score (correctness gate still fires first)."""
    tenant_a = Tenant(id="a", workload=_loose_wl(), verifiers=_verifiers())
    tenant_b = Tenant(id="b", workload=_loose_wl(), verifiers=_verifiers())
    tenants = [tenant_a, tenant_b]

    trace = _two_tenant_trace()
    # Aggregate verdict failed
    verdict = _verdict(trace, ok=False)
    scorer = _scorer()
    score = asyncio.run(scorer.score(tenants, trace, verdict, _hw()))
    assert score.value == 0.0
    assert score.gate_passed is False
    assert "correctness" in score.zero_reasons


def test_same_request_id_across_tenants_does_not_collide():
    """Two tenants over the SAME dataset path produce identical request_ids (both
    "r0"). Credit and verdict are keyed on event_key, so one tenant's result can
    NOT clobber the other's: tenant 'a' passes correctness (its r0 is credited),
    tenant 'b' fails correctness (its r0 is NOT credited) -- despite the shared id.

    With a flat request_id map, looking up either event by "r0" would return ONE
    verdict/credit for BOTH, conflating the two tenants."""
    tenant_a = Tenant(id="a", workload=_loose_wl(), verifiers=_verifiers())
    tenant_b = Tenant(id="b", workload=_loose_wl(), verifiers=_verifiers())
    # Same request_id "r0" in BOTH tenants -- the collision case.
    trace = Trace(events=[
        _ev("r0", tenant_id="a"),
        _ev("r0", tenant_id="b"),
    ])
    # a's r0 passed correctness; b's r0 failed -- keyed by event_key, so the two
    # "r0"s are distinct ("a:r0" vs "b:r0").
    verdict = Verdict(
        passed=True,
        per_event={"a:r0": True, "b:r0": False},
        diagnostics={},
    )
    score = asyncio.run(_scorer().score([tenant_a, tenant_b], trace, verdict, _hw()))
    # Only a's r0 is credited; b's r0 failed correctness -> 0 credit.
    # duration = 1.0; achieved = 1 credited event / 1.0 = 1.0.
    assert score.gate_passed is True
    assert score.achieved == 1.0


def test_single_tenant_score_unchanged():
    """A single-tenant list [Tenant('t0')] scores its one tenant's events in full."""
    from hls.core.payloads import Envelope, TextPayload
    from hls.core.trace import ArtifactResponse

    wl = _loose_wl()
    tenant_t0 = Tenant(id="t0", workload=wl, verifiers=_verifiers())

    # Single-tenant events are stamped tenant_id="t0" (as the driver does).
    inp = Envelope(request_id="r0", contents=[TextPayload(text="Q?")],
                   metadata={"tenant_id": "t0"})
    out = Envelope(request_id="r0", contents=[TextPayload(text="X")])
    ev = TraceEvent(
        request_id="r0",
        input=inp,
        response=ArtifactResponse(output=out),
        ground_truth="X",
        t_send=0.0,
        t_recv=1.0,
    )
    trace = Trace(events=[ev])
    verdict = _verdict(trace)
    scorer = _scorer()
    score = asyncio.run(scorer.score([tenant_t0], trace, verdict, _hw()))
    assert score.gate_passed is True
    assert score.achieved == 1.0


def test_single_tenant_breakdown_slo_is_nested_under_t0():
    """Unified shape: single-tenant ('t0') breakdown['slo'] is nested under the
    tenant id ({"t0": {...diagnostics...}}), same as multi-tenant -- so the viewer
    reads every run the same way."""
    tenant = Tenant(id="t0", workload=_loose_wl(), verifiers=_verifiers())
    trace = Trace(events=[_ev("r0", tenant_id="t0")])
    verdict = _verdict(trace)
    score = asyncio.run(_scorer().score([tenant], trace, verdict, _hw()))
    slo = score.breakdown["slo"]
    assert "t0" in slo, f"expected per-tenant nesting under 't0', got {slo!r}"
    assert slo["t0"], f"tenant diagnostics should be non-empty, got {slo!r}"


def test_single_tenant_gated_zero_breakdown_slo_is_nested():
    """An SLO-gated-zero single-tenant run keeps breakdown['slo'] nested too
    (one shape across pass and gated-zero runs)."""
    tenant = Tenant(id="t0", workload=_tight_wl(), verifiers=_verifiers())
    trace = Trace(events=[_ev("r0", tenant_id="t0")])
    verdict = _verdict(trace)
    score = asyncio.run(_scorer().score([tenant], trace, verdict, _hw()))
    assert score.gate_passed is False
    assert "slo" in score.zero_reasons
    slo = score.breakdown["slo"]
    assert "t0" in slo, f"expected per-tenant nesting under 't0', got {slo!r}"
    assert slo["t0"], f"tenant diagnostics should be non-empty, got {slo!r}"


def test_multi_tenant_breakdown_slo_is_nested_per_tenant():
    """Multi-tenant breakdown['slo'] keeps the per-tenant nested shape
    ({tenant_id: {diagnostics}}) -- now identical in structure to single-tenant."""
    tenant_a = Tenant(id="a", workload=_loose_wl(), verifiers=_verifiers())
    tenant_b = Tenant(id="b", workload=_loose_wl(), verifiers=_verifiers())
    trace = _two_tenant_trace()
    verdict = _verdict(trace)
    score = asyncio.run(_scorer().score([tenant_a, tenant_b], trace, verdict, _hw()))
    slo = score.breakdown["slo"]
    assert "a" in slo and "b" in slo, f"expected per-tenant nesting, got {slo!r}"
    assert slo["a"], f"tenant diagnostics should be non-empty, got {slo!r}"


class _ZeroCountUnit(Unit):
    """A unit whose count_for_event is 0 for a valid, passing event (e.g. an
    empty-but-valid output). Drives counted == 0.0 WITHOUT any SLO failure."""

    name = "zero"

    def count_for_event(self, ev) -> float:
        return 0.0


def test_zero_count_with_passing_slo_is_not_slo_gated():
    """counted == 0.0 with NO SLO failure (here a zero-count unit under a passing
    SLO) must NOT be mislabeled as SLO-gated. The reason is tracked explicitly
    (any_slo_failure), not inferred from the arithmetic of `counted`: the gate stays
    passed and the score is a genuine value=0, not Score.zero(reasons=('slo',))."""
    tenant = Tenant(id="t0", workload=_loose_wl(), verifiers=_verifiers())  # SLO passes
    trace = Trace(events=[_ev("r0", tenant_id="t0")])
    verdict = _verdict(trace)
    scorer = GoodputScorer(unit=_ZeroCountUnit(), sol=ConstantSoL(rps=1.0))
    score = asyncio.run(scorer.score([tenant], trace, verdict, _hw()))
    assert score.gate_passed is True, "no SLO failed -> gate must stay passed"
    assert score.zero_reasons == ()
    assert score.value == 0.0 and score.achieved == 0.0
