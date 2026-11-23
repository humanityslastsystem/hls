import asyncio
import math

import pytest

from hls.core.model import AcceleratorSpec, Hardware
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import (ArtifactResponse, FailureReason, RequestAttempt,
                            RequestFailure, Trace, TraceEvent)
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit, SloLatencyScorer
from hls.core.verdict import Verdict
from hls.spec.tenant import Tenant
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, SLO, SLOResult, StreamingLatencySLO,
                               Uncapped, Workload)


def _ev(rid: str, t_send: float, t_recv: float) -> TraceEvent:
    # Stamp tenant_id="t0" so the single-tenant scorer's events_for("t0") finds it
    # (the driver always stamps it; there is no None fallback).
    env = Envelope(request_id=rid, contents=[TextPayload(text="x")],
                   metadata={"tenant_id": "t0"})
    return TraceEvent(request_id=rid, input=env, output=env, t_send=t_send, t_recv=t_recv)


def _trace(latencies: list[float]) -> Trace:
    evs = []
    t = 0.0
    for i, dt in enumerate(latencies):
        evs.append(_ev(f"r{i}", t, t + dt))
        t += dt
    return Trace(events=evs)


def _hw() -> Hardware:
    return Hardware(accelerator=AcceleratorSpec(kind="B200", count=1))


def _wl() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                    slo=NoSLO())


def _t0(wl: Workload | None = None) -> list[Tenant]:
    """Wrap a workload into a single Tenant("t0") list for score() calls."""
    return [Tenant(id="t0", workload=wl or _wl(), verifiers=[])]


def _verdict(trace: Trace, ok: bool) -> Verdict:
    """A run-level Verdict whose per_event is keyed by event_key (tenant-namespaced),
    as the Runner builds it; here every event shares the aggregate outcome `ok`."""
    return Verdict(passed=ok, per_event={ev.event_key: ok for ev in trace.events}, diagnostics={})


def test_correctness_gate_zeros_score():
    trace = _trace([0.1, 0.1])
    verdict = _verdict(trace, False)
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
    score = asyncio.run(scorer.score(_t0(), trace, verdict, _hw()))
    assert score.value == 0.0
    assert score.gate_passed is False
    assert "correctness" in score.zero_reasons


def test_slo_aggregate_gate_zeros_score():
    trace = _trace([0.1, 0.1])
    verdict = _verdict(trace, True)
    class _FailingGate(SLO):
        """Minimal binary aggregate-gate SLO (the shape the removed P99Latency had)."""

        def apply(self, trace, workload) -> SLOResult:
            return SLOResult(aggregate_passed=False, per_event_credit={}, diagnostics={})

    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                        slo=_FailingGate())   # aggregate gate fails
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
    score = asyncio.run(scorer.score(_t0(workload), trace, verdict, _hw()))
    assert score.value == 0.0
    assert "slo" in score.zero_reasons


def test_throughput_score_basic():
    trace = _trace([0.1, 0.1])
    verdict = _verdict(trace, True)
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=100.0))
    score = asyncio.run(scorer.score(_t0(), trace, verdict, _hw()))
    assert score.gate_passed is True
    assert score.unit == "request"
    assert 0.0 < score.value <= 1.0


def test_scorer_multiplies_correctness_by_slo_credit():
    """The goodput Scorer must AND the per-event correctness verdict with the SLO's
    per-event credit. Two requests both meet a 200ms TTFT bar (credit 1.0); one is
    correct, one is not -- so only one counts: achieved = 1 request / 1.0s duration."""
    def _stream_ev(rid: str) -> TraceEvent:
        env = Envelope(request_id=rid, contents=[TextPayload(text="x")],
                       metadata={"tenant_id": "t0"})
        out = Envelope(request_id=rid, contents=[TextPayload(text="a")])
        return TraceEvent(request_id=rid, input=env,
                          response=ArtifactResponse(output=out, t_first_packet=0.05,
                                                    packet_timestamps=[0.05]),  # TTFT 50ms
                          t_send=0.0, t_recv=1.0)

    e_ok, e_bad = _stream_ev("ok"), _stream_ev("bad")
    trace = Trace(events=[e_ok, e_bad])
    wl = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                  slo=StreamingLatencySLO(ttft_ms=200.0, tpot_ms=50.0))
    verdict = Verdict(passed=True,
                      per_event={e_ok.event_key: True, e_bad.event_key: False}, diagnostics={})
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
    score = asyncio.run(scorer.score(_t0(wl), trace, verdict, _hw()))
    assert score.achieved == 1.0                       # duration 1.0s; only e_ok counted
    assert score.breakdown["slo_attainment"] == 0.5    # attainment ANDs the verdict too


# --------------------------------------------------------------------------
# SloLatencyScorer -- S = exp(mean log(1/r) - baseline), r = max(r_ttft, r_tpot)
#
# The score is the geometric mean of the per-request speedups 1/r, divided by the
# baseline stack's. Every helper below builds single-packet requests against a 100ms
# TTFT budget, so a request's r is exactly what the helper is asked for.
# --------------------------------------------------------------------------

_DRIVE_SEC = 1.0


def _slo_ev(rid: str, *, tenant: str = "t0", t_first: float, ts: list[float],
            t_recv: float = _DRIVE_SEC) -> TraceEvent:
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": tenant})
    out = Envelope(request_id=rid, contents=[TextPayload(text="a")])
    return TraceEvent(request_id=rid, input=inp,
                      response=ArtifactResponse(output=out, t_first_packet=t_first,
                                                packet_timestamps=ts),
                      t_send=0.0, t_recv=t_recv)


def _at_r(rid: str, r: float, *, tenant: str = "t0",
          t_recv: float = _DRIVE_SEC) -> TraceEvent:
    """A single-packet request (TPOT vacuous -> TTFT alone gates) whose SLO-normalized
    latency is exactly `r` against a 100ms TTFT budget."""
    t = 0.1 * r
    return _slo_ev(rid, tenant=tenant, t_first=t, ts=[t], t_recv=t_recv)


def _dead(rid: str, *, tenant: str = "t0", t_send: float = 0.0) -> TraceEvent:
    """A request that errored: no response, so no computable r."""
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": tenant})
    return TraceEvent(request_id=rid, input=inp, response=None,
                      error="timeout: boom", t_send=t_send, t_recv=None)


def _abandoned(rid: str, *, t_send: float, t_abandoned: float, reason: FailureReason,
               due: float | None = None, tenant: str = "t0") -> TraceEvent:
    """A request the harness stopped waiting on, carrying the failure record the driver
    attaches: ``t_abandoned`` is the moment it stopped watching, and ``due`` is the
    schedule's intended send time when that differs from ``t_send``."""
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": tenant})
    return TraceEvent(
        request_id=rid, input=inp, response=None, error=f"{reason.value}: boom",
        t_send=t_send, t_planned_send=due, t_recv=None,
        failure=RequestFailure(reason=reason, t_abandoned=t_abandoned,
                               observed=RequestAttempt()))


def _unsent(rid: str, *, due: float, skipped_at: float, tenant: str = "t0") -> TraceEvent:
    """A request the driver's circuit breaker never sent: it stamps both the send and
    the abandonment at the skip instant, so t_planned_send is the only record of when
    the request was actually due."""
    return _abandoned(rid, t_send=skipped_at, t_abandoned=skipped_at, due=due,
                      tenant=tenant, reason=FailureReason.CIRCUIT_BREAKER)


def _slo_wl(ttft_ms: float = 100.0, tpot_ms: float = 50.0) -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                    slo=StreamingLatencySLO(ttft_ms=ttft_ms, tpot_ms=tpot_ms))


def _slo_score(events, tenants=None, *, baseline=None, verdict=None):
    trace = Trace(events=events)
    tenants = tenants or [Tenant(id="t0", workload=_slo_wl(), verifiers=[])]
    scorer = SloLatencyScorer(baseline_log_credit=baseline)
    return asyncio.run(scorer.score(tenants, trace, verdict or _verdict(trace, True), _hw()))


def test_slo_latency_request_grain_score_survives_the_unit_seam():
    """The scorer means over the SLO's own units, and a request-grain SLO's units are
    its requests one apiece, so a per-request task's Score is untouched by that seam.
    Every number here was read off this same trace before the seam existed.

    Three requests at r = 0.25, 0.5 and 1.0. The geometric mean of 4, 2 and 1 is 2, so
    S = 2 against the SLO itself and exactly 1 against a baseline stack that managed
    r = 0.5 throughout.
    """
    events = [_at_r("a", 0.25), _at_r("b", 0.5), _at_r("c", 1.0)]
    score = _slo_score(events)

    assert score.value == pytest.approx(2.0)
    assert score.achieved == pytest.approx(2.0)
    assert score.sol == 1.0
    assert score.unit == "slo_speedup"
    assert score.gate_passed is True
    assert score.zero_reasons == ()
    assert score.breakdown["mean_log_credit"] == pytest.approx(math.log(2.0))
    assert score.breakdown["baseline_log_credit"] is None
    assert score.breakdown["n_scored"] == 3
    assert score.breakdown["slo_attainment"] == pytest.approx(2 / 3)
    assert score.breakdown["upper_bound"] == pytest.approx(2.0)
    tenant_slo = score.breakdown["slo"]["t0"]
    assert tenant_slo["mean_log_credit"] == pytest.approx(math.log(2.0))
    assert tenant_slo["mean_r"] == pytest.approx(7 / 12)
    assert (tenant_slo["n_met"], tenant_slo["n_total"], tenant_slo["n_missing"]) == (
        2, 3, 0)

    normalized = _slo_score(events, baseline=math.log(2.0))
    assert normalized.value == pytest.approx(1.0)
    assert normalized.sol == pytest.approx(2.0)
    assert normalized.breakdown["upper_bound"] == pytest.approx(1.0)


def test_slo_latency_speedup_is_one_exactly_at_the_slo():
    # r = 1 is the metric's fixed point: log(1/1) = 0, so S = exp(0) = 1. With no baseline
    # measured, a server that lands every request precisely on its budget scores 1.0.
    score = _slo_score([_at_r(f"r{i}", 1.0) for i in range(5)])
    assert score.value == 1.0
    assert score.gate_passed is True
    assert score.breakdown["mean_log_credit"] == 0.0


def test_slo_latency_score_is_the_geometric_mean_of_one_over_r():
    # S = 1/r for a single request: twice as fast as its budget scores 2.0, twice as slow
    # scores 0.5. The two are reciprocal, which is the property the log buys.
    fast = _slo_score([_at_r("f", 0.5)])
    slow = _slo_score([_at_r("s", 2.0)])
    assert fast.value == pytest.approx(2.0)
    assert slow.value == pytest.approx(0.5)
    assert fast.value * slow.value == pytest.approx(1.0)


def test_slo_latency_halving_every_latency_doubles_the_score():
    # The scale the score is read on: a server that halves every request's latency doubles
    # S, wherever it started. No plateau at the SLO, no ceiling above it.
    base = [_at_r(f"b{i}", r) for i, r in enumerate([0.4, 1.0, 2.5])]
    halved = [_at_r(f"h{i}", r / 2) for i, r in enumerate([0.4, 1.0, 2.5])]
    assert _slo_score(halved).value == pytest.approx(2 * _slo_score(base).value)


def test_slo_latency_baseline_credit_normalizes_the_score_to_the_baseline_stack():
    # S = exp(L - B): the artifact's geometric mean of 1/r over the baseline stack's. A
    # baseline that itself ran at r = 0.5 (B = log 2) makes an artifact at r = 0.25 score
    # 2.0 -- twice the baseline -- while the same artifact scores 4.0 unnormalized.
    b = math.log(2.0)
    score = _slo_score([_at_r("a", 0.25)], baseline=b)
    assert score.value == pytest.approx(2.0)
    assert score.achieved == pytest.approx(4.0)          # against the SLO itself
    assert score.sol == pytest.approx(2.0)               # what the baseline stack managed
    assert score.value == pytest.approx(score.achieved / score.sol)
    assert score.breakdown["baseline_log_credit"] == b
    # an artifact that merely matches the baseline scores exactly 1.0
    assert _slo_score([_at_r("m", 0.5)], baseline=b).value == pytest.approx(1.0)


def test_slo_latency_without_a_baseline_scores_against_the_slo_itself():
    # No baseline measured yet -- the mode the baseline measurement runs in. S = exp(L),
    # the speedup against the SLO, and sol is 1.0 because nothing
    # normalized it. Not a fallback: it is how B itself gets measured, from mean_log_credit.
    score = _slo_score([_at_r("a", 0.5), _at_r("b", 0.125)])
    assert score.breakdown["baseline_log_credit"] is None
    assert score.sol == 1.0
    assert score.value == pytest.approx(score.achieved)
    assert score.value == pytest.approx(math.exp((math.log(2.0) + math.log(8.0)) / 2))
    assert score.value == pytest.approx(4.0)             # geometric mean of 2 and 8
    # the number a baseline measurement reads off and averages over its drives
    assert score.breakdown["mean_log_credit"] == pytest.approx(math.log(4.0))


def test_slo_latency_geometric_mean_defeats_buying_the_score_with_a_few_fast_requests():
    # WHY the geometric mean and not an arithmetic mean of 1/r. A "gamed" server parks 90%
    # of requests exactly on their SLO and makes 10% blazing (r = 0.05); a "uniform" server
    # is simply 2x better everywhere. On mean 1/r the gamed server WINS (2.9 vs 2.0) -- it
    # buys the score with a handful of cheap requests. Under the geometric mean it loses,
    # as it must (1.35 vs 2.0).
    uniform = _slo_score([_at_r(f"u{i}", 0.5) for i in range(100)])
    gamed = _slo_score([_at_r(f"g{i}", 1.0) for i in range(90)]
                       + [_at_r(f"g{i}", 0.05) for i in range(90, 100)])
    assert uniform.value == pytest.approx(2.0)
    assert gamed.value == pytest.approx(1.35, abs=0.005)
    assert uniform.value > gamed.value
    # the arithmetic-mean counterfactual this shape exists to defeat, on the same requests:
    assert (0.9 * (1 / 1.0) + 0.1 * (1 / 0.05)) == pytest.approx(2.9)
    assert 2.9 > 2.0


def test_slo_latency_is_request_weighted_and_invariant_to_how_many_requests_ran():
    # The mean, not a sum: ten requests at r = 0.5 in one drive score exactly what one
    # does. The score describes the typical request, so it does not move with the size of
    # the split, and every request carries the same weight regardless of its tenant.
    one = _slo_score([_at_r("a", 0.5)])
    ten = _slo_score([_at_r(f"r{i}", 0.5) for i in range(10)])
    assert ten.value == pytest.approx(one.value)
    assert ten.breakdown["n_scored"] == 10 and one.breakdown["n_scored"] == 1


def test_slo_latency_is_not_a_rate_the_same_requests_over_a_longer_drive_score_the_same():
    # The score is per request, not per second: identical per-request latencies spread over
    # a drive twice as long are worth the same. The arrival process is fixed by the task,
    # so wall-clock length is not something the server chooses.
    quick = _slo_score([_at_r(f"q{i}", 0.5, t_recv=1.0) for i in range(4)])
    slow = _slo_score([_at_r(f"s{i}", 0.5, t_recv=2.0) for i in range(4)])
    assert quick.value == pytest.approx(slow.value)
    assert "duration_sec" not in quick.breakdown


def test_slo_latency_r_is_max_not_sum_a_great_ttft_cannot_rescue_bad_tpot():
    # End to end through the scorer: TTFT ~0 with mean TPOT 1.76x budget must score BELOW
    # 1.0 (r = 1.76 -> S = 1/1.76). A summed/averaged r (0.88) would have made this blown
    # request score above 1.0 -- better than a fully compliant one.
    ev = _slo_ev("v", t_first=0.0, ts=[0.0, 0.088])   # TTFT 0ms; mean TPOT 88ms = 1.76 x 50
    score = _slo_score([ev], [Tenant(id="t0", workload=_slo_wl(ttft_ms=1000.0, tpot_ms=50.0),
                                     verifiers=[])])
    assert score.value == pytest.approx(1 / 1.76)
    assert score.value < 1.0                                  # a sum would have given 1.14
    assert score.value < _slo_score([_at_r("ok", 1.0)]).value  # loses to a merely-compliant one


def test_slo_latency_vacuous_tpot_lets_ttft_alone_gate():
    # < 2 packets -> TPOT undefined -> that axis contributes 0, so r = r_ttft. A one-token
    # response is complete: it must be scored on TTFT, not punished or dropped.
    score = _slo_score([_slo_ev("short", t_first=0.05, ts=[0.05])])   # TTFT 50ms, r 0.5
    assert score.gate_passed is True
    assert score.value == pytest.approx(2.0)


def test_slo_latency_any_failed_request_fails_the_whole_run():
    # THE hard gate. 99 blazing requests + ONE that errored (no response) -> the run FAILS,
    # rather than the failed request being dropped from the mean (which would have made
    # shedding the hardest request a WIN: the 99 alone score 20.0).
    good = [_at_r(f"g{i}", 0.05) for i in range(99)]
    score = _slo_score(good + [_dead("bad")])
    assert score.gate_passed is False
    assert "slo" in score.zero_reasons
    assert score.value == 0.0
    # and the diagnostics say WHICH tenant and how many requests failed
    assert score.breakdown["slo"]["t0"]["n_missing"] == 1


def test_slo_latency_response_that_never_streamed_fails_the_run():
    # Not an "error" upstream (there IS a response), so any_errored never sees it: a
    # response with no first-token timing has no computable r and must fail the run here.
    out = Envelope(request_id="r0", contents=[TextPayload(text="a")])
    inp = Envelope(request_id="r0", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    ev = TraceEvent(request_id="r0", input=inp,
                    response=ArtifactResponse(output=out, t_first_packet=None,
                                              packet_timestamps=None),
                    t_send=0.0, t_recv=1.0)
    score = _slo_score([ev])
    assert score.gate_passed is False and "slo" in score.zero_reasons


def test_slo_latency_correctness_gate_zeros_a_run_whose_latency_was_fine():
    # An incorrect run is zeroed on correctness alone: its latency gate passed, so
    # "correctness" is the only reason. This is also the path the runner's any_errored
    # (transport faults -> verdict.passed False) arrives on, which is why the SLO gate
    # above is a complement, not a duplicate.
    trace = Trace(events=[_at_r("a", 0.5)])
    score = asyncio.run(SloLatencyScorer().score(
        [Tenant(id="t0", workload=_slo_wl(), verifiers=[])], trace, _verdict(trace, False), _hw()))
    assert score.gate_passed is False and score.zero_reasons == ("correctness",)
    # the upper bound survives the correctness gate too, so a zeroed run still says how
    # fast the artifact was serving what it did serve
    assert score.breakdown["upper_bound"] == pytest.approx(2.0)


def test_slo_latency_reports_every_reason_the_run_was_gated_for():
    # A run can be wrong AND slow. Reporting only the layer that happened to be checked
    # first hides the other: this one answered a request incorrectly and never answered
    # the second at all, and both must be on the record.
    trace = Trace(events=[_at_r("ok", 0.5), _dead("bad")])
    score = _slo_score(trace.events, verdict=_verdict(trace, False))
    assert score.value == 0.0 and score.gate_passed is False
    assert sorted(score.zero_reasons) == ["correctness", "slo"]


def test_slo_latency_correctness_zero_still_reports_the_latency_it_measured():
    # The whole point of carrying both sections: a run zeroed for correctness must still
    # say how fast it was, so "was it also missing its SLO?" is answerable from the score
    # rather than by re-deriving it from the trace.
    trace = Trace(events=[_at_r("a", 0.5), _at_r("b", 0.5)])
    score = _slo_score(trace.events, verdict=_verdict(trace, False))
    assert score.zero_reasons == ("correctness",)
    tenant_slo = score.breakdown["slo"]["t0"]
    assert tenant_slo["p99_ttft_ms"] == pytest.approx(50.0)
    assert tenant_slo["mean_r"] == pytest.approx(0.5)
    assert tenant_slo["n_met"] == 2 and tenant_slo["met_frac"] == 1.0
    assert tenant_slo["mean_log_credit"] == pytest.approx(math.log(2.0))


def test_slo_latency_slo_zero_still_reports_its_verdict():
    # The other direction: a run gated on latency alone must still carry what the
    # verifiers said about it, so a passing verdict on a gated run is legible.
    trace = Trace(events=[_at_r("ok", 0.5), _dead("bad")])
    verdict = Verdict(passed=True, per_event={ev.event_key: True for ev in trace.events},
                      diagnostics={"pass_rate": 1.0})
    score = _slo_score(trace.events, verdict=verdict)
    assert score.zero_reasons == ("slo",)
    assert score.breakdown["verdict"] == {"pass_rate": 1.0}
    assert score.breakdown["slo"]["t0"]["n_missing"] == 1


def test_slo_latency_a_passing_run_names_no_reason():
    score = _slo_score([_at_r("a", 0.5)])
    assert score.gate_passed is True and score.zero_reasons == ()
    assert set(score.breakdown) == {"slo", "slo_attainment", "mean_log_credit",
                                    "baseline_log_credit", "n_scored", "upper_bound",
                                    "verdict"}


def test_slo_latency_upper_bound_equals_the_score_on_a_valid_run():
    # With nothing to censor, the upper bound IS the score: every request has a real r.
    score = _slo_score([_at_r("a", 0.5), _at_r("b", 2.0), _at_r("c", 0.25)])
    assert score.breakdown["upper_bound"] == pytest.approx(score.value)


def test_slo_latency_upper_bound_censors_a_failed_request_at_the_drive_end():
    # A zeroed run's only reportable number. The failed request is scored as if its first
    # token arrived when the drive ended (1.0s against a 100ms budget -> r = 10), so the
    # bound is the geometric mean of 2.0 and 0.1: about 0.45. It is a real number, it is
    # strictly below the 2.0 the surviving request alone would have scored, and it is
    # never the score -- value stays 0.0.
    score = _slo_score([_at_r("ok", 0.5, t_recv=1.0), _dead("bad")])
    assert score.value == 0.0 and "slo" in score.zero_reasons
    bound = score.breakdown["upper_bound"]
    assert bound == pytest.approx(math.sqrt(2.0 * 0.1))
    assert 0.0 < bound < 2.0


def test_slo_latency_upper_bound_charges_a_failed_request_with_no_record_to_the_drive_end():
    # With no failure record there is nothing to say when the harness stopped watching,
    # so the run's end is the censoring time: the same failure in a drive that ran four
    # times as long is charged four times the latency, and the bound falls. A request
    # that does carry a record is charged to its own abandonment instead.
    short = _slo_score([_at_r("ok", 0.5, t_recv=1.0), _dead("bad")])
    long = _slo_score([_at_r("ok", 0.5, t_recv=4.0), _dead("bad")])
    assert long.breakdown["upper_bound"] < short.breakdown["upper_bound"]
    assert long.breakdown["upper_bound"] == pytest.approx(math.sqrt(2.0 * 0.025))


def test_slo_latency_upper_bound_survives_a_breaker_aborted_drive():
    # The bound must stay a bound when the driver's circuit breaker gives up. It stamps
    # every unsent request at the skip instant, milliseconds before the drive ends, so
    # charging from t_send would make each one look answered in ~0ms -- worth +20.7 nats
    # apiece, and a dead server would report a bound orders of magnitude above the
    # baseline. Charged from when it was due and floored at its whole budget, an unsent
    # request is worth 0 credit, so the bound cannot exceed what the served part earned.
    served = [_at_r("ok0", 0.5, t_recv=1.0), _at_r("ok1", 0.5, t_recv=1.0)]
    aborted = [_unsent(f"skip{i}", due=1.0, skipped_at=1.0) for i in range(8)]
    score = _slo_score(served + aborted)

    assert score.value == 0.0 and "slo" in score.zero_reasons
    bound = score.breakdown["upper_bound"]
    assert bound == pytest.approx(2.0 ** 0.2)      # two requests at 1/r = 2, eight at 1
    assert bound <= _slo_score(served).value       # the served part alone scored 2.0


def test_slo_latency_upper_bound_charges_an_unsent_request_from_when_it_was_due():
    # An unsent request is charged for the whole time it was outstanding, which starts at
    # its scheduled arrival, not at the instant the breaker skipped it. One due at the
    # start of a drive that ran 1.0s is 10x its 100ms budget, so it costs credit rather
    # than earning it.
    score = _slo_score([_at_r("ok", 0.5, t_recv=1.0),
                        _unsent("skip", due=0.0, skipped_at=1.0)])
    assert score.breakdown["upper_bound"] == pytest.approx(math.sqrt(2.0 * 0.1))


def test_slo_latency_upper_bound_charges_a_timed_out_request_to_its_deadline():
    # A request abandoned at a deadline is charged up to that deadline: 3.0s against a
    # 100ms budget is r = 30, so it costs credit in proportion to how long it really
    # stalled.
    score = _slo_score([_at_r("ok", 0.5, t_recv=1.0),
                        _abandoned("slow", t_send=0.0, t_abandoned=3.0,
                                   reason=FailureReason.DEADLINE)])
    assert score.breakdown["upper_bound"] == pytest.approx(math.sqrt(2.0 / 30.0))


def test_slo_latency_upper_bound_is_not_moved_by_how_long_the_rest_of_the_drive_ran():
    # A connection refused 5ms after the send was outstanding for 5ms only. It
    # floors at r = 1 and contributes no credit, and the bound is the same whether the
    # rest of the drive took one second or forty -- what the other requests did is not
    # this request's fault, in either direction.
    refused = _abandoned("no", t_send=0.0, t_abandoned=0.005,
                         reason=FailureReason.TRANSPORT)
    short = _slo_score([_at_r("ok", 0.5, t_recv=1.0), refused])
    long = _slo_score([_at_r("ok", 0.5, t_recv=40.0), refused])
    assert short.breakdown["upper_bound"] == pytest.approx(math.sqrt(2.0))
    assert long.breakdown["upper_bound"] == pytest.approx(math.sqrt(2.0))


def test_slo_latency_gates_a_drive_that_served_nothing():
    # A run whose trace is empty served nothing, so it must be gated. Unguarded, the mean
    # of no credits is 0.0 and the run scores exp(-B) -- a pass, above zero, for a server
    # that answered no request at all.
    score = _slo_score([])
    assert score.value == 0.0
    assert score.gate_passed is False and "slo" in score.zero_reasons
    assert score.breakdown["upper_bound"] == 0.0
    assert score.breakdown["slo"]["n_requests"] == 0


def test_slo_latency_bounds_a_run_of_undeclared_tenants_at_zero():
    # Every event carries a tenant id the task does not declare, so no SLO scored any of
    # them and the run is gated. The bound must be 0.0, not exp(-B): the run priced no
    # request, and a mean over nothing would report the baseline's own score as an upper
    # bound on a run that measured nothing. Same vacuity the empty trace is gated for.
    score = _slo_score([_at_r("a", 0.5, tenant="ghost"), _at_r("b", 0.5, tenant="ghost")],
                       [Tenant(id="t0", workload=_slo_wl(), verifiers=[])],
                       baseline=math.log(2.0))
    assert score.value == 0.0
    assert score.gate_passed is False and "slo" in score.zero_reasons
    assert score.breakdown["upper_bound"] == 0.0


def test_slo_latency_bounds_a_tenant_with_no_latency_normalizer_at_zero():
    # A NoSLO tenant defines no r, censored or otherwise, so a speedup scorer can price
    # none of its requests -- the same empty bound arrived at from the declared side.
    score = _slo_score([_at_r("a", 0.5), _at_r("b", 0.5)],
                       [Tenant(id="t0", workload=_wl(), verifiers=[])],
                       baseline=math.log(2.0))
    assert score.gate_passed is False
    assert score.breakdown["upper_bound"] == 0.0


def test_goodput_gates_a_drive_that_served_nothing():
    # The same hole on the goodput member: no events means no credit and no throughput,
    # which must read as a gated run rather than a passing zero.
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
    empty = Trace(events=[])
    score = asyncio.run(scorer.score(_t0(), empty, _verdict(empty, True), _hw()))
    assert score.value == 0.0
    assert score.gate_passed is False and score.zero_reasons == ("slo",)
    assert score.breakdown["slo"]["n_requests"] == 0


def test_goodput_reports_every_reason_the_run_was_gated_for():
    # The goodput member collects reasons the same way. The run is incorrect overall
    # while one event passed, and its tenant's SLO zeroed that event's credit: two
    # independent faults, both named.
    class _FailingGate(SLO):
        def apply(self, trace, workload) -> SLOResult:
            return SLOResult(aggregate_passed=False, per_event_credit={},
                             diagnostics={"n_missing": 2})

    trace = _trace([0.1, 0.1])
    verdict = Verdict(passed=False,
                      per_event={ev.event_key: i == 0 for i, ev in enumerate(trace.events)},
                      diagnostics={"pass_rate": 0.5})
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                        output_length_cap=Uncapped(), slo=_FailingGate())
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
    score = asyncio.run(scorer.score(_t0(workload), trace, verdict, _hw()))
    assert score.value == 0.0 and score.gate_passed is False
    assert sorted(score.zero_reasons) == ["correctness", "slo"]
    # and both sides are on the record, not just the one that happened to be checked first
    assert score.breakdown["slo"]["t0"] == {"n_missing": 2}
    assert score.breakdown["verdict"] == {"pass_rate": 0.5}


def test_goodput_correctness_zero_still_reports_the_latency_it_measured():
    trace = _trace([0.1, 0.1])
    scorer = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
    score = asyncio.run(scorer.score(_t0(), trace, _verdict(trace, False), _hw()))
    assert score.zero_reasons == ("correctness",)
    assert "t0" in score.breakdown["slo"]


def test_slo_latency_multi_tenant_pools_requests_over_each_tenants_own_budget():
    # Each tenant normalizes by ITS OWN budget -- the point of the whole metric. Two
    # tenants, same 300ms TTFT latency: for the lax one (1000ms) r = 0.3, for the strict
    # one (300ms) r = 1.0. The two requests pool into one geometric mean, request-weighted.
    lax = Tenant(id="mrcr", workload=_slo_wl(ttft_ms=1000.0), verifiers=[])
    strict = Tenant(id="bfcl", workload=_slo_wl(ttft_ms=300.0), verifiers=[])
    events = [_slo_ev("a", tenant="mrcr", t_first=0.3, ts=[0.3]),
              _slo_ev("a", tenant="bfcl", t_first=0.3, ts=[0.3])]   # same request_id!
    score = _slo_score(events, [lax, strict])
    assert score.value == pytest.approx(math.sqrt((1 / 0.3) * (1 / 1.0)))
    assert score.breakdown["n_scored"] == 2      # request_id "a" collides; event_key does not
    # The score is pooled, but each tenant's own contribution to it is reported beside
    # the SLO's latency diagnostics: mean_r is an arithmetic mean of r and so is not
    # the tenant's term in a geometric-mean score, which mean_log_credit is.
    assert score.breakdown["slo"]["mrcr"]["mean_r"] == pytest.approx(0.3)
    assert score.breakdown["slo"]["mrcr"]["mean_log_credit"] == pytest.approx(math.log(1 / 0.3))
    assert score.breakdown["slo"]["bfcl"]["mean_log_credit"] == pytest.approx(0.0)
    # The score is the request-weighted mean of the two, both tenants serving one request.
    assert score.breakdown["mean_log_credit"] == pytest.approx(math.log(1 / 0.3) / 2)


def test_slo_latency_reports_its_unit_and_diagnostics():
    score = _slo_score([_at_r("a", 0.5), _at_r("b", 2.0)])
    assert score.unit == "slo_speedup"
    assert score.value == pytest.approx(1.0)      # geometric mean of 2.0 and 0.5
    assert score.breakdown["mean_log_credit"] == pytest.approx(0.0)
    assert score.breakdown["n_scored"] == 2
    assert score.breakdown["slo_attainment"] == 0.5   # binary met (r < 1) ANDed w/ verdict
    # the rate's leftovers are gone: this is a per-request score, not credit per second
    assert "sum_log_credit" not in score.breakdown
    assert "duration_sec" not in score.breakdown


def test_slo_latency_orders_identically_to_the_mean_log_credit():
    # The representation contract: Score.value must order exactly as mean log(1/r), and
    # every valid run scores strictly above 0 -- so a gated 0.0 can never out-rank one.
    cases = [[0.25], [0.5, 0.5], [1.0], [2.0, 0.5], [4.0], [0.5] * 10]
    scored = [(_slo_score([_at_r(f"r{i}", r) for i, r in enumerate(rs)]).value,
               sum(math.log(1 / r) for r in rs) / len(rs))
              for rs in cases]
    for value, mean_credit in scored:
        assert value == pytest.approx(math.exp(mean_credit))
        assert value > 0.0
    by_value = sorted(range(len(scored)), key=lambda i: scored[i][0])
    by_metric = sorted(range(len(scored)), key=lambda i: scored[i][1])
    assert by_value == by_metric


def test_slo_latency_baseline_credit_cannot_reorder_two_artifacts():
    # B is a property of the task, not of the artifact: dividing every score by the same
    # constant leaves the ranking alone. Worth pinning, because it is what lets a baseline
    # measured on one day be compared with an artifact scored on another.
    runs = [[0.25], [0.5], [1.0], [2.0]]
    for b in (None, math.log(1.7)):
        values = [_slo_score([_at_r("a", r) for r in rs], baseline=b).value for rs in runs]
        assert values == sorted(values, reverse=True)
