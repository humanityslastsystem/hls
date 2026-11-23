import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import (ArtifactResponse, FailureReason, RequestAttempt,
                            RequestFailure, Trace, TraceEvent)
from hls.spec.workload import (
    Workload, AllAtOnce, ContextLinearLatencySLO, NoDelay, Uncapped,
    NoSLO, PREFILL_TOKENS_EXTRA, RequestLatencySLO, StreamingLatencySLO,
)


def _wl(slo=None):
    # StreamingLatencySLO.apply ignores `workload`; a placeholder SLO is fine here.
    # Workload.slo stays a required field (never None).
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(),
                    slo=slo or NoSLO())


def _apply(slo, trace):
    """Apply an SLO to one tenant's slice of a trace."""
    return slo.apply(trace, _wl())


def _ev(rid, tenant, t_send, t_first, ts):
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": tenant})
    out = Envelope(request_id=rid, contents=[TextPayload(text="a")])
    return TraceEvent(request_id=rid, input=inp,
                      response=ArtifactResponse(output=out, t_first_packet=t_first,
                                                packet_timestamps=ts),
                      t_send=t_send, t_recv=ts[-1])


def test_streaming_r_is_the_binding_axis_not_a_sum():
    # THE non-negotiable rule: r = max(r_ttft, r_tpot), never a sum/mean. A near-perfect
    # TTFT must NOT buy back a TPOT violation. TTFT ~0 (r_ttft 0.0), mean TPOT 88ms =
    # 1.76 x the 50ms budget -> r = 1.76, NOT the 0.88 a 1/2(a+b) sum would give (which
    # would score this blown request BETTER than a fully compliant r = 1.0 one).
    ev = _ev("v", "t0", 0.5, 0.5, [0.5, 0.588])   # TTFT 0ms; mean TPOT 88ms = 1.76 x 50
    res = _apply(StreamingLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    assert res.per_event_ratio["v"] == pytest.approx(1.76)
    assert res.per_event_ratio["v"] > 1.0            # a sum (0.88) would have "passed" it
    assert res.diagnostics["mean_r"] == pytest.approx(1.76)
    assert res.diagnostics["n_met"] == 0             # r >= 1 -> not binary-met


def test_streaming_r_is_one_exactly_at_the_slo():
    # r = 1 is the metric's natural zero point (log(1/1) = 0), so it must be EXACT, not
    # approximate: a request whose TTFT lands precisely on its budget has r == 1.0.
    ev = _ev("v", "t0", 0.0, 0.1, [0.1])     # TTFT 100ms == the 100ms budget
    res = _apply(StreamingLatencySLO(ttft_ms=100, tpot_ms=50), Trace(events=[ev]))
    assert res.per_event_ratio["v"] == 1.0


def test_streaming_r_takes_the_worse_axis_either_way():
    # max() is symmetric: whichever axis is worse sets r. Same SLO, two mirror-image
    # requests -- one TTFT-bound, one TPOT-bound -- both land on their bad axis.
    ttft_bound = _ev("ttft", "t0", 0.0, 0.15, [0.15, 0.16])   # TTFT 1.5x, TPOT 0.2x
    tpot_bound = _ev("tpot", "t0", 0.0, 0.01, [0.01, 0.16])   # TTFT 0.1x, TPOT 3.0x
    res = _apply(StreamingLatencySLO(ttft_ms=100, tpot_ms=50), Trace(events=[ttft_bound, tpot_bound]))
    assert res.per_event_ratio["ttft"] == pytest.approx(1.5)   # not the 0.2 TPOT axis
    assert res.per_event_ratio["tpot"] == pytest.approx(3.0)   # not the 0.1 TTFT axis


def test_streaming_reports_r_per_request():
    # The SLO's job is the per-request normalization, nothing more. Three requests, single
    # packet each (TPOT vacuous -> TTFT alone gates): r = 0.5, 1.5, 2.5.
    a = _ev("a", "t0", 0.0, 0.05, [0.05])    # TTFT 50ms  -> r 0.5
    b = _ev("b", "t0", 0.0, 0.15, [0.15])    # TTFT 150ms -> r 1.5
    c = _ev("c", "t0", 0.0, 0.25, [0.25])    # TTFT 250ms -> r 2.5
    res = _apply(StreamingLatencySLO(ttft_ms=100, tpot_ms=50), Trace(events=[a, b, c]))
    assert res.per_event_ratio == {"a": pytest.approx(0.5), "b": pytest.approx(1.5),
                                   "c": pytest.approx(2.5)}
    assert res.diagnostics["mean_r"] == pytest.approx(1.5)


def test_streaming_no_plateau_faster_than_slo_keeps_scoring_better():
    # The removed graded credit plateaued at 1.0 for every request at/under budget, which
    # against a FIXED open-loop arrival capped the score. r has no plateau: both of these
    # are comfortably inside budget, and the faster one STILL scores strictly better.
    #   fast: TTFT 10/200 = 0.05, TPOT 10/80 = 0.125 -> r = max = 0.125
    #   near: TTFT 190/200 = 0.95, TPOT 10/80 = 0.125 -> r = max = 0.95
    fast = _ev("fast", "t0", 0.0, 0.01, [0.01, 0.02, 0.03])   # TTFT 10ms, TPOT 10ms
    near = _ev("near", "t0", 0.0, 0.19, [0.19, 0.20, 0.21])   # TTFT 190ms, TPOT 10ms
    res = _apply(StreamingLatencySLO(ttft_ms=200, tpot_ms=80), Trace(events=[fast, near]))
    assert res.per_event_ratio["fast"] == pytest.approx(0.125)
    assert res.per_event_ratio["near"] == pytest.approx(0.95)
    assert res.per_event_ratio["fast"] < res.per_event_ratio["near"]   # no ceiling to tie at
    # ...and both are binary-met, so the DIAGNOSTIC does plateau where the metric does not.
    assert res.diagnostics["met_frac"] == 1.0


def test_streaming_no_cutoff_a_disastrous_request_keeps_costing():
    # The removed ramp died at soft_cutoff x budget: past 2x, every request was equally
    # worthless, so a 10x-over request cost no more than a 2x-over one. r is unbounded:
    # 10x over budget contributes 10.0 to the mean.
    ev = _ev("v", "t0", 0.0, 1.0, [1.0])      # TTFT 1000ms = 10 x the 100ms budget
    res = _apply(StreamingLatencySLO(ttft_ms=100, tpot_ms=50), Trace(events=[ev]))
    assert res.per_event_ratio["v"] == pytest.approx(10.0)
    assert res.diagnostics["mean_r"] == pytest.approx(10.0)


def test_streaming_tpot_is_mean_inter_token_not_worst_gap():
    # TPOT is the MEAN inter-token latency = (last-first)/(n-1). A request with one 150ms
    # spike but mean 80ms is r_tpot 0.8 on a 100ms bar (the spike alone doesn't set it);
    # a request whose mean exceeds the bar is r_tpot > 1. Confirms mean, not worst-gap.
    spiky = _ev("spiky", "t0", 0.0, 0.0, [0.0, 0.01, 0.16])  # gaps 10/150ms, mean 80ms
    slow = _ev("slow", "t0", 0.0, 0.0, [0.0, 0.12, 0.24])    # gaps 120/120ms, mean 120ms
    res = _apply(StreamingLatencySLO(ttft_ms=1000, tpot_ms=100), Trace(events=[spiky, slow]))
    assert res.per_event_ratio["spiky"] == pytest.approx(0.8)
    assert res.per_event_ratio["slow"] == pytest.approx(1.2)
    assert res.diagnostics["n_met"] == 1 and res.diagnostics["met_frac"] == 0.5


def test_streaming_vacuous_tpot_lets_ttft_alone_gate():
    # < 2 packets makes TPOT undefined ((last-first)/(n-1) has no gap to average), so that
    # axis contributes 0.0 and r = r_ttft. The request is NOT punished for being short, and
    # a single-token response still cannot hide a slow TTFT.
    short_fast = _ev("short_fast", "t0", 0.0, 0.05, [0.05])   # 1 packet, TTFT 50ms
    short_slow = _ev("short_slow", "t0", 0.0, 0.30, [0.30])   # 1 packet, TTFT 300ms
    res = _apply(StreamingLatencySLO(ttft_ms=100, tpot_ms=50), Trace(events=[short_fast, short_slow]))
    assert res.per_event_ratio["short_fast"] == pytest.approx(0.5)   # TTFT alone
    assert res.per_event_ratio["short_slow"] == pytest.approx(3.0)   # TTFT alone
    assert res.aggregate_passed is True    # a 1-token response is complete, not a failure


def test_streaming_clean_run_passes_the_gate_and_credits_the_met():
    # A run where every request streamed: the gate passes, every request has an r, and the
    # binary met credit (r < 1) rides alongside for the attainment diagnostic.
    tr = Trace(events=[_ev(f"r{i}", "t0", 0.0, 0.1, [0.1, 0.15, 0.2]) for i in range(50)])
    res = _apply(StreamingLatencySLO(ttft_ms=200, tpot_ms=80), tr)
    assert res.aggregate_passed is True
    assert len(res.per_event_ratio) == 50
    assert all(c == 1.0 for c in res.per_event_credit.values())
    assert res.diagnostics["met_frac"] == 1.0
    assert res.diagnostics["n_missing"] == 0


def test_streaming_request_with_no_timing_fails_the_run():
    # A response that never streamed a first token has NO computable r. It must not be
    # dropped from the mean (that would let a server score best by never answering its
    # hardest requests): it gets no ratio entry and FAILS the aggregate gate.
    out = Envelope(request_id="r0", contents=[TextPayload(text="a")])
    inp = Envelope(request_id="r0", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    ev = TraceEvent(request_id="r0", input=inp,
                    response=ArtifactResponse(output=out, t_first_packet=None,
                                              packet_timestamps=None),
                    t_send=0.0, t_recv=1.0)
    res = _apply(StreamingLatencySLO(ttft_ms=200, tpot_ms=80), Trace(events=[ev]))
    assert res.aggregate_passed is False
    assert "r0" not in res.per_event_ratio
    assert res.diagnostics["n_missing"] == 1
    assert res.diagnostics["has_timing"] is False
    assert res.diagnostics["met_frac"] == 0.0


def test_streaming_one_errored_request_fails_the_whole_run():
    # ONE failed request among 49 good ones fails the gate -- the hard aggregate rule.
    # (Its transport fault also fails per event in the verify path; this gate additionally
    # covers a baseline credit's assume-correct drives, which keep verdict.passed.)
    good = [_ev(f"g{i}", "t0", 0.0, 0.1, [0.1, 0.15, 0.2]) for i in range(49)]
    inp = Envelope(request_id="bad", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    bad = TraceEvent(request_id="bad", input=inp, response=None,
                     error="timeout: boom", t_send=0.0, t_recv=None)
    res = _apply(StreamingLatencySLO(ttft_ms=200, tpot_ms=80), Trace(events=good + [bad]))
    assert res.aggregate_passed is False
    assert res.diagnostics["n_missing"] == 1 and res.diagnostics["n_total"] == 50
    assert "bad" not in res.per_event_ratio


def test_streaming_slo_is_a_per_tenant_normalizer():
    # The SLO normalizes rather than marking a bar. Two tenants with identical latency but
    # different budgets (mrcr's 1000ms vs bfcl's 300ms TTFT) land on different r -- which
    # is exactly what makes their requests comparable in one mean.
    ev = _ev("v", "t0", 0.0, 0.3, [0.3])              # TTFT 300ms, single packet
    lax = _apply(StreamingLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    strict = _apply(StreamingLatencySLO(ttft_ms=300, tpot_ms=50), Trace(events=[ev]))
    assert lax.per_event_ratio["v"] == pytest.approx(0.3)     # well inside a 1000ms budget
    assert strict.per_event_ratio["v"] == pytest.approx(1.0)  # exactly at a 300ms budget


def test_streaming_diagnostics_expose_mean_r_and_the_binary_met_split():
    # Diagnostics split cleanly: mean_r is THE metric (continuous), met_frac/n_met the
    # BINARY feasibility signal. Three requests (single packet -> TTFT alone gates):
    #   met   r 0.5 -> binary-met         mid  r 1.5 -> not met        far  r 2.5 -> not met
    met = _ev("met", "t0", 0.0, 0.05, [0.05])
    mid = _ev("mid", "t0", 0.0, 0.15, [0.15])
    far = _ev("far", "t0", 0.0, 0.25, [0.25])
    res = _apply(StreamingLatencySLO(ttft_ms=100, tpot_ms=50), Trace(events=[met, mid, far]))
    assert res.diagnostics["mean_r"] == pytest.approx((0.5 + 1.5 + 2.5) / 3)
    assert res.diagnostics["n_met"] == 1 and res.diagnostics["n_total"] == 3
    assert res.diagnostics["met_frac"] == pytest.approx(1 / 3)
    assert res.diagnostics["ttft_ms"] == 100 and res.diagnostics["tpot_ms"] == 50
    # no graded-credit knob is in the record
    assert "soft_cutoff" not in res.diagnostics and "mean_credit" not in res.diagnostics


def test_streaming_has_no_soft_cutoff_field():
    # There is no plateau/ramp knob, not even a defaulted one: it is absent from the model
    # and from provenance, so no ramp can be reconstructed from a record. A task passing
    # soft_cutoff= is refused, as every keyword that is not a field is.
    from pydantic import ValidationError

    from hls.core.poly import dump_spec

    slo = StreamingLatencySLO(ttft_ms=100, tpot_ms=50)
    assert not hasattr(slo, "soft_cutoff")
    assert "soft_cutoff" not in dump_spec(slo)
    with pytest.raises(ValidationError, match="soft_cutoff"):
        StreamingLatencySLO(ttft_ms=100, tpot_ms=50, soft_cutoff=1.5)


def test_streaming_censors_a_request_with_no_timing_at_the_run_end():
    # A request that never answered has no r, but it can still be priced: the time it was
    # outstanding, taken as its TTFT. It was due at 0.5s and the run ended at 4.5s, so
    # 4000ms against a 200ms budget is r = 20 -- the most generous reading, since the run
    # stopped before it could do worse. The ratio map still omits it.
    out = Envelope(request_id="dead", contents=[TextPayload(text="a")])
    inp = Envelope(request_id="dead", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    dead = TraceEvent(request_id="dead", input=inp,
                      response=ArtifactResponse(output=out, t_first_packet=None,
                                                packet_timestamps=None),
                      t_send=0.5, t_recv=None)
    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    assert "dead" not in _apply(slo, Trace(events=[dead])).per_event_ratio
    assert slo.censored_ratio([dead], 4.5) == pytest.approx(20.0)


def test_streaming_censors_a_response_that_never_streamed_at_its_own_arrival():
    # A response that came back whole, with no per-packet timing, has no r either -- but
    # the harness watched it for exactly as long as it took. Due at 0.5s and answered at
    # 2.5s, it is charged those 2000ms against a 200ms budget: r = 10, not the r = 100 a
    # drive that ran on to 20.5s would charge it. It has no failure record, so nothing
    # but t_recv says when it stopped being outstanding.
    out = Envelope(request_id="whole", contents=[TextPayload(text="a")])
    inp = Envelope(request_id="whole", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    unstreamed = TraceEvent(request_id="whole", input=inp,
                            response=ArtifactResponse(output=out, t_first_packet=None,
                                                      packet_timestamps=None),
                            t_send=0.5, t_recv=2.5)
    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    assert "whole" not in _apply(slo, Trace(events=[unstreamed])).per_event_ratio
    assert unstreamed.failure is None
    assert slo.censored_ratio([unstreamed], 20.5) == pytest.approx(10.0)


def test_streaming_censors_an_unsent_request_from_when_it_was_due():
    # The driver's circuit breaker stamps a request it never sent at the skip instant, so
    # t_send says nothing about how long the request was outstanding. t_planned_send does:
    # this one was due at 0.0 and the run ended at 4.5s, which is 22.5x a 200ms budget --
    # not the 0.5x that stamping alone would suggest.
    inp = Envelope(request_id="skip", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    skipped = TraceEvent(request_id="skip", input=inp, response=None,
                         error="circuit breaker open",
                         t_send=4.4, t_planned_send=0.0, t_recv=None)
    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    assert slo.censored_ratio([skipped], 4.5) == pytest.approx(22.5)


def _failed(rid, *, t_send, t_abandoned, reason, t_planned_send=None, prefill=None):
    """A request the harness gave up on, with the failure record the driver attaches:
    ``t_abandoned`` is the moment it stopped watching. ``prefill`` is the prompt-length
    stamp the driver lifts off the envelope the framework sent."""
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    return TraceEvent(request_id=rid, input=inp, response=None, error="boom",
                      t_send=t_send, t_planned_send=t_planned_send, t_recv=None,
                      extras={} if prefill is None else {PREFILL_TOKENS_EXTRA: prefill},
                      failure=RequestFailure(reason=reason, t_abandoned=t_abandoned,
                                             observed=RequestAttempt()))


def test_streaming_censors_a_timed_out_request_at_its_deadline():
    # A request the harness abandoned at a deadline is charged up to that deadline, not
    # to whenever the rest of the drive finished. Sent at 0.5s, abandoned at 4.5s: 4000ms
    # against a 200ms budget is r = 20, whether the drive ended there or ran on for an
    # hour afterwards.
    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    timed_out = _failed("slow", t_send=0.5, t_abandoned=4.5,
                        reason=FailureReason.DEADLINE)
    assert slo.censored_ratio([timed_out], 4.5) == pytest.approx(20.0)
    assert slo.censored_ratio([timed_out], 3600.0) == pytest.approx(20.0)


def test_streaming_charges_an_early_refusal_only_the_moment_it_was_outstanding():
    # A connection refused 5ms after the send was outstanding for 5ms, so it is charged
    # 5ms and floors at r = 1 -- credit exactly 0. Charging it to the end of a drive that
    # ran four more seconds would have made an instant refusal look like a 20x latency
    # violation, dragging the bound below what the artifact was really reaching for.
    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    refused = _failed("refused", t_send=0.5, t_abandoned=0.505,
                      reason=FailureReason.TRANSPORT)
    assert slo.censored_ratio([refused], 4.5) == 1.0


def test_streaming_never_credits_a_request_that_did_not_answer():
    # The floor that keeps the censored ratio a bound. A request skipped as the drive
    # ended was outstanding for ~0s, and a ratio of ~0 would read as a request answered
    # infinitely fast -- worth up to +20.7 nats of credit. It did not meet its SLO, so it
    # is charged its whole budget: r = 1, credit exactly 0.
    inp = Envelope(request_id="skip", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    skipped = TraceEvent(request_id="skip", input=inp, response=None,
                         error="circuit breaker open",
                         t_send=4.5, t_planned_send=4.5, t_recv=None)
    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    assert slo.censored_ratio([skipped], 4.5) == 1.0


def test_no_slo_defines_no_censored_ratio():
    # NoSLO has no latency normalizer at all, so there is nothing to censor -- a
    # SloLatencyScorer on a NoSLO tenant can score nothing, not even an upper bound.
    ev = _ev("r0", "t0", 0.0, 0.1, [0.1])
    res = NoSLO().apply(Trace(events=[ev]), _wl())
    assert res.per_event_ratio == {}
    assert NoSLO().censored_ratio([ev], 1.0) is None
    assert res.aggregate_passed is True


def _lat_ev(rid, idx, latency_sec, tenant="t0"):
    # t_planned_send carries the send ORDER (idx); latency drives the ratio.
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": tenant})
    out = Envelope(request_id=rid, contents=[TextPayload(text="a")])
    return TraceEvent(request_id=rid, input=inp, response=ArtifactResponse(output=out),
                      t_send=float(idx), t_recv=float(idx) + latency_sec,
                      t_planned_send=float(idx))


# ---- RequestLatencySLO as a latency NORMALIZER -----------------------------
# The unary counterpart of StreamingLatencySLO's ratio. It is what lets a task whose
# contract is a single unary response be scored on SLO-normalized latency.

def test_request_latency_reports_a_ratio_per_request():
    # r = JCT / target, continuous, so a small latency change moves the score.
    slo = RequestLatencySLO(target_ms=1_000.0)
    tr = Trace(events=[_lat_ev("fast", 0, 0.5), _lat_ev("at_bar", 1, 1.0),
                       _lat_ev("slow", 2, 2.0)])
    res = slo.apply(tr, _wl())
    assert res.per_event_ratio["fast"] == pytest.approx(0.5)
    assert res.per_event_ratio["at_bar"] == pytest.approx(1.0)
    assert res.per_event_ratio["slow"] == pytest.approx(2.0)


def test_request_latency_keeps_its_binary_credit_alongside_the_ratio():
    # The credit is what a goodput scorer and the attainment diagnostic read;
    # adding the ratio must not disturb it.
    slo = RequestLatencySLO(target_ms=1_000.0)
    res = slo.apply(Trace(events=[_lat_ev("in", 0, 0.5), _lat_ev("out", 1, 1.5)]), _wl())
    assert res.per_event_credit == {"in": 1.0, "out": 0.0}
    assert res.diagnostics["n_met"] == 1 and res.diagnostics["n_total"] == 2


def _unanswered(rid, *, t_planned, error="timeout"):
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    return TraceEvent(request_id=rid, input=inp, response=None, error=error,
                      t_send=t_planned, t_planned_send=t_planned, t_recv=None)


def test_request_latency_gives_no_ratio_to_a_request_that_never_answered():
    # It must not vanish silently: the scorer fails the run on a missing ratio, which
    # is what stops a server scoring well by dropping its hardest requests.
    res = RequestLatencySLO(target_ms=1_000.0).apply(
        Trace(events=[_unanswered("dead", t_planned=0.0)]), _wl())
    assert "dead" not in res.per_event_ratio
    assert res.per_event_credit["dead"] == 0.0
    assert res.diagnostics["n_missing"] == 1


def test_request_latency_keeps_no_aggregate_cliff():
    # aggregate_passed stays True even with a missing ratio: the bar is per request,
    # and the scorer, not the SLO, fails a run with a request that never answered.
    res = RequestLatencySLO(target_ms=1_000.0).apply(
        Trace(events=[_unanswered("dead", t_planned=0.0)]), _wl())
    assert res.aggregate_passed is True


def test_request_latency_censors_an_unanswered_request_at_the_run_end():
    # Priced for the upper bound only: due at 0.0, the run ended at 4.0, so 4000ms
    # against a 1000ms budget is r = 4.
    dead = _unanswered("dead", t_planned=0.0)
    assert RequestLatencySLO(target_ms=1_000.0).censored_ratio([dead], 4.0) == pytest.approx(4.0)


def test_request_latency_never_credits_a_request_that_did_not_answer():
    # Floored at r = 1. A request skipped as the drive ended was outstanding for ~0s,
    # and an unfloored ratio near 0 would read as one answered infinitely fast.
    skipped = _unanswered("skip", t_planned=4.0, error="circuit breaker open")
    assert RequestLatencySLO(target_ms=1_000.0).censored_ratio([skipped], 4.0) == 1.0
# --- the TTFT budget that grows with the prompt ----------------------------

def _deep(rid, *, t_send, t_first, prefill=None, reported_prompt_tokens=None):
    """A streamed request carrying the harness's count of its own prompt.

    ``reported_prompt_tokens`` is what the server said about the same prompt. It
    rides on the response metadata, where a server's usage figure would, and nothing
    in the budget may read it."""
    metadata = ({} if reported_prompt_tokens is None
                else {"prompt_tokens": reported_prompt_tokens})
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    out = Envelope(request_id=rid, contents=[TextPayload(text="a")], metadata=metadata)
    extras = {} if prefill is None else {PREFILL_TOKENS_EXTRA: prefill}
    return TraceEvent(request_id=rid, input=inp,
                      response=ArtifactResponse(output=out, t_first_packet=t_first,
                                                packet_timestamps=[t_first]),
                      t_send=t_send, t_recv=t_first, extras=extras)


def test_the_flat_slo_is_untouched_by_a_prompt_length():
    # Every task but the one that opted in keeps exactly the SLO it had. A turn at
    # 600k of prompt is priced against the flat 200ms exactly as a turn at none:
    # 400ms is r = 2 either way, and the length in its extras changes nothing.
    deep = _deep("deep", t_send=0.0, t_first=0.4, prefill=600_000)
    shallow = _deep("shallow", t_send=0.0, t_first=0.4)
    flat = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    res = _apply(flat, Trace(events=[deep, shallow]))
    assert res.per_event_ratio == {"deep": pytest.approx(2.0),
                                   "shallow": pytest.approx(2.0)}
    assert res.aggregate_passed is True


def test_the_flat_slo_serializes_exactly_as_it_did():
    # A stored baseline credit refuses a task whose tenants' serialized SLOs have
    # moved, so the flat record is load-bearing for every task that did not opt in.
    # That is why the slope is its own family member and not a defaulted field here.
    from hls.core.poly import dump_spec

    assert dump_spec(StreamingLatencySLO(ttft_ms=500, tpot_ms=50)) == {
        "kind": "StreamingLatencySLO", "ttft_ms": 500.0, "tpot_ms": 50.0}


def test_a_slope_of_zero_is_the_flat_budget_and_asks_for_nothing():
    # What a task declares while its profile is pending. It must behave as the flat
    # budget did, including for a request nothing stamped a length on -- the stamping
    # requirement arrives with the measurement, not before it.
    pending = ContextLinearLatencySLO(ttft_ms=200, tpot_ms=80)
    assert pending.ttft_ms_per_prefill_token == 0.0
    unstamped = _deep("unstamped", t_send=0.0, t_first=0.4)
    assert pending.ttft_budget_ms(unstamped) == 200.0
    res = _apply(pending, Trace(events=[unstamped]))
    assert res.per_event_ratio["unstamped"] == pytest.approx(2.0)
    assert res.aggregate_passed is True


def test_a_declared_slope_widens_the_budget_by_the_prompt_it_prefilled():
    # 200ms flat + 0.001ms per token: a 100k prompt is budgeted 300ms and a 600k one
    # 800ms. Both took 400ms, so the deep turn now beats its budget (r = 0.5) where
    # the shallow one misses (r = 1.33) -- the ordering a flat budget could not see.
    sloped = ContextLinearLatencySLO(ttft_ms=200, tpot_ms=80,
                                      ttft_ms_per_prefill_token=0.001)
    shallow = _deep("shallow", t_send=0.0, t_first=0.4, prefill=100_000)
    deep = _deep("deep", t_send=0.0, t_first=0.4, prefill=600_000)
    assert sloped.ttft_budget_ms(shallow) == pytest.approx(300.0)
    assert sloped.ttft_budget_ms(deep) == pytest.approx(800.0)
    res = _apply(sloped, Trace(events=[shallow, deep]))
    assert res.per_event_ratio["shallow"] == pytest.approx(400.0 / 300.0)
    assert res.per_event_ratio["deep"] == pytest.approx(0.5)


def test_the_budget_reads_the_harness_count_not_what_the_server_reported():
    # The whole reason the count is stamped harness-side. This server claims a
    # 600k-token prompt for what the harness counted at 1k; under a slope that claim
    # would be worth an 800ms budget instead of 201ms, turning r = 1.99 into r = 0.5.
    sloped = ContextLinearLatencySLO(ttft_ms=200, tpot_ms=80,
                                      ttft_ms_per_prefill_token=0.001)
    liar = _deep("liar", t_send=0.0, t_first=0.4, prefill=1_000,
                 reported_prompt_tokens=600_000)
    assert sloped.ttft_budget_ms(liar) == pytest.approx(201.0)
    assert _apply(sloped, Trace(events=[liar])).per_event_ratio["liar"] == \
        pytest.approx(400.0 / 201.0)


def test_a_slope_with_no_prompt_length_prices_nothing_and_fails_the_run():
    # A declared slope and a request nothing stamped a prompt length on is a wiring
    # fault: pricing it against the flat term would be a plausible number computed
    # from a budget the task did not declare. It gets no ratio, which fails the run
    # the way any unpriceable request does, and its own diagnostic so triage does not
    # read it as a server that never answered.
    sloped = ContextLinearLatencySLO(ttft_ms=200, tpot_ms=80,
                                      ttft_ms_per_prefill_token=0.001)
    unstamped = _deep("unstamped", t_send=0.0, t_first=0.4)
    assert sloped.ttft_budget_ms(unstamped) is None
    res = _apply(sloped, Trace(events=[unstamped]))
    assert res.per_event_ratio == {}
    assert res.aggregate_passed is False
    assert res.diagnostics["n_no_budget"] == 1
    assert res.diagnostics["n_no_response"] == 0   # it answered; nobody counted its prompt
    # ...and the flat SLO never looks, so the same request prices fine under it.
    assert _apply(StreamingLatencySLO(ttft_ms=200, tpot_ms=80),
                  Trace(events=[unstamped])).per_event_ratio == {"unstamped": pytest.approx(2.0)}


def test_a_turn_the_server_never_answered_is_counted_apart_from_the_wiring_faults():
    """Three ways a request gets no ratio, and only one of them is the server's.

    ``n_missing`` is the gate, and on its own it does not say which. ``n_no_budget``
    is a task that stamped no prompt length and ``n_impossible_timing`` is clocks that
    disagree -- both harness faults, both fixed in the harness. A request with no
    response at all is the third, and it is the measurement: the server was asked and
    did not answer. Without a count of its own, triage of a zeroed run starts by
    reading the trace, and a refused reply that also carried no prompt length would read
    as the harness's fault for a contract the server broke.
    """
    sloped = ContextLinearLatencySLO(ttft_ms=1400, tpot_ms=20,
                                     ttft_ms_per_prefill_token=0.008)
    refused = _failed("refused", t_send=0.0, t_abandoned=545.6,
                      reason=FailureReason.PROTOCOL, prefill=202_345)
    answered = _deep("ok", t_send=0.0, t_first=0.4, prefill=1_000)

    res = _apply(sloped, Trace(events=[refused, answered]))

    assert res.diagnostics["n_missing"] == 1
    assert res.diagnostics["n_no_response"] == 1
    assert res.diagnostics["n_no_budget"] == 0
    assert res.diagnostics["n_impossible_timing"] == 0
    # The stamp it now carries is what prices it: its own budget, not the flat term.
    budget_ms = 1400.0 + 0.008 * 202_345
    assert sloped.ttft_budget_ms(refused) == pytest.approx(budget_ms)
    assert sloped.censored_ratio([refused], 600.0) == pytest.approx(
        545.6 * 1000.0 / budget_ms)


def test_a_non_numeric_prompt_length_is_no_prompt_length():
    # Extras is a free-form dict, so the stamp is whatever the framework put there.
    # A string or a bool is not a token count and must not reach the arithmetic.
    sloped = ContextLinearLatencySLO(ttft_ms=200, tpot_ms=80,
                                      ttft_ms_per_prefill_token=0.001)
    for junk in ("600000", True, None):
        ev = _deep("junk", t_send=0.0, t_first=0.4, prefill=junk)
        assert sloped.ttft_budget_ms(ev) is None


def test_a_censored_request_is_charged_against_its_own_budget():
    # A deep request that never answered was reaching for a deep budget, so the bound
    # it reports has to use that one: 4000ms outstanding against an 800ms budget is
    # r = 5, not the r = 20 the flat term would charge.
    sloped = ContextLinearLatencySLO(ttft_ms=200, tpot_ms=80,
                                      ttft_ms_per_prefill_token=0.001)
    inp = Envelope(request_id="dead", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    dead = TraceEvent(request_id="dead", input=inp, response=None, error="boom",
                      t_send=0.5, t_recv=None,
                      extras={PREFILL_TOKENS_EXTRA: 600_000})
    assert sloped.censored_ratio([dead], 4.5) == pytest.approx(5.0)
    # With no length stamped there is no own budget, so it falls back to the flat
    # term -- the harsher of the two, which is the safe direction for a bound.
    unstamped = dead.model_copy(update={"extras": {}})
    assert sloped.censored_ratio([unstamped], 4.5) == pytest.approx(20.0)


def test_the_slope_is_in_the_record():
    # A run's provenance and diagnostics have to say which budget priced it, or two
    # runs at different slopes read as the same measurement. The family member's own
    # name carries the rule; the slope carries its size.
    from hls.core.poly import dump_spec

    sloped = ContextLinearLatencySLO(ttft_ms=200, tpot_ms=80,
                                      ttft_ms_per_prefill_token=0.001)
    assert dump_spec(sloped) == {"kind": "ContextLinearLatencySLO", "ttft_ms": 200.0,
                                 "tpot_ms": 80.0, "ttft_ms_per_prefill_token": 0.001}
    res = _apply(sloped, Trace(events=[_deep("d", t_send=0.0, t_first=0.1,
                                             prefill=1_000)]))
    assert res.diagnostics["ttft_ms_per_prefill_token"] == 0.001
    assert "per prompt token" in sloped.describe()
    flat = _apply(StreamingLatencySLO(ttft_ms=200, tpot_ms=80),
                  Trace(events=[_deep("d", t_send=0.0, t_first=0.1)]))
    assert "ttft_ms_per_prefill_token" not in flat.diagnostics


# ---------------------------------------------------------------------------
# Cut requests: one phantom token completing at the cut
# ---------------------------------------------------------------------------


def _cut(rid, *, t_send, t_cut, stamps=(), prefill=None):
    """A request the drive's scheduled stop cut, as the driver records one: the tokens
    that arrived with their arrival times, a ``t_recv`` of the cut, no error and no
    failure."""
    from hls.spec.workload import CUT_EXTRA

    extras = {CUT_EXTRA: True}
    if prefill is not None:
        extras[PREFILL_TOKENS_EXTRA] = prefill
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    out = Envelope(request_id=rid, contents=[TextPayload(
        text="a" * len(stamps), tokens=list(range(len(stamps))) or None)])
    return TraceEvent(
        request_id=rid, input=inp,
        response=ArtifactResponse(output=out,
                                  t_first_packet=stamps[0] if stamps else None,
                                  packet_timestamps=list(stamps) or None),
        t_send=t_send, t_recv=t_cut, extras=extras)


def test_a_cut_request_that_streamed_nothing_is_charged_its_whole_wait():
    """It reached no first token, so its time-to-first-token is everything it waited,
    up to the moment the harness stopped listening."""
    ev = _cut("c", t_send=1.0, t_cut=4.0)
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    assert res.per_event_ratio["c"] == pytest.approx(3.0)
    assert res.aggregate_passed                    # priced, so the gate is not tripped


def test_a_cut_request_with_one_token_is_charged_the_span_to_the_cut():
    """Its time to first token is what was observed, and the phantom gives it one
    inter-token interval: the whole span from that token to the cut."""
    ev = _cut("c", t_send=1.0, t_cut=9.0, stamps=(1.5,))
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    # TTFT 500 ms over a 1,000 ms budget is 0.5; TPOT is 9.0 - 1.5 = 7.5 s over 50 ms.
    assert res.per_event_ratio["c"] == pytest.approx(150.0)


def test_a_request_that_streamed_fast_and_then_hung_is_charged_the_silence():
    """The bound the phantom exists for. A thousand tokens at 20 ms then ten minutes
    of nothing is not a 20 ms decode: the token that never came would have arrived no
    earlier than the cut, so the trailing silence is one more interval."""
    stamps = tuple(0.02 * i for i in range(1000))
    ev = _cut("c", t_send=0.0, t_cut=stamps[-1] + 600.0, stamps=stamps)
    res = _apply(ContextLinearLatencySLO(ttft_ms=1_000_000, tpot_ms=1.0),
                 Trace(events=[ev]))
    # (619.98 - 0) / 1000 tokens = 619.98 ms, against a 1 ms budget.
    assert res.per_event_ratio["c"] == pytest.approx(619.98, rel=1e-6)


def test_a_burst_then_silence_scores_far_worse_than_streaming_to_the_cut():
    """The hole the phantom closes, stated as the comparison it turns on.

    Two requests cut at the same instant: one streams five tokens in a tenth of a
    second and then says nothing for ten seconds, the other streams five hundred at
    an even 20 ms right up to the cut. Priced on the streamed tokens alone -- last
    arrival minus first, over one fewer interval -- both come out at exactly 20 ms per
    token, so the server that stopped working scores as well as the one that did not.
    The phantom separates them by the whole trailing silence."""
    burst = _cut("burst", t_send=0.0, t_cut=10.0,
                 stamps=tuple(0.02 * i for i in range(5)))
    steady = _cut("steady", t_send=0.0, t_cut=10.0,
                  stamps=tuple(0.02 * i for i in range(500)))

    res = _apply(ContextLinearLatencySLO(ttft_ms=1_000_000, tpot_ms=20.0),
                 Trace(events=[burst, steady]))

    assert res.per_event_ratio["steady"] == pytest.approx(1.0)      # 20 ms per token
    assert res.per_event_ratio["burst"] == pytest.approx(100.0)     # 2 s per token
    assert res.per_event_ratio["burst"] > 50 * res.per_event_ratio["steady"]


def test_a_healthy_stream_cut_mid_decode_scores_near_its_true_rate():
    """The other side of the bound: a server streaming steadily when the window closed
    is charged its own rate plus at most the part of one interval it had used."""
    stamps = tuple(0.02 * i for i in range(100))            # 100 tokens at 20 ms
    ev = _cut("c", t_send=0.0, t_cut=stamps[-1] + 0.021, stamps=stamps)
    res = _apply(ContextLinearLatencySLO(ttft_ms=1_000_000, tpot_ms=20.0),
                 Trace(events=[ev]))
    # 20.01 ms per token against a 20 ms budget: within one interval of the truth.
    assert res.per_event_ratio["c"] == pytest.approx(20.01 / 20.0, rel=1e-6)


def test_a_completed_request_is_untouched_by_the_phantom():
    """The phantom is the cut's alone. A reply that ended on its own a long time
    before the drive's window closed keeps the rate it actually streamed at."""
    ev = _ev("v", "t0", 0.0, 0.0, [0.0, 0.02, 0.04])
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=20), Trace(events=[ev]))
    assert res.per_event_ratio["v"] == pytest.approx(1.0)


def test_a_cut_request_is_priced_against_its_own_context_linear_budget():
    sloped = ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50,
                                     ttft_ms_per_prefill_token=0.01)
    ev = _cut("c", t_send=1.0, t_cut=4.0, prefill=100_000)     # budget 2,000 ms
    res = _apply(sloped, Trace(events=[ev]))
    assert res.per_event_ratio["c"] == pytest.approx(1.5)


def test_a_cut_request_with_no_budget_is_still_refused_a_ratio():
    """The stamp requirement is not waived by the cut: pricing an unstamped request
    against the flat term would be a number computed from a rule the task did not
    declare."""
    sloped = ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50,
                                     ttft_ms_per_prefill_token=0.01)
    res = _apply(sloped, Trace(events=[_cut("c", t_send=1.0, t_cut=4.0)]))
    assert "c" not in res.per_event_ratio
    assert not res.aggregate_passed


def test_a_request_that_is_not_cut_and_never_streamed_still_has_no_ratio():
    """The whole-wait pricing is the cut's alone. A reply that reached no first token
    on its own is a server that did not stream, which fails the run."""
    inp = Envelope(request_id="u", contents=[TextPayload(text="q")],
                   metadata={"tenant_id": "t0"})
    out = Envelope(request_id="u", contents=[TextPayload(text="a")])
    ev = TraceEvent(request_id="u", input=inp,
                    response=ArtifactResponse(output=out), t_send=1.0, t_recv=4.0)
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    assert "u" not in res.per_event_ratio
    assert not res.aggregate_passed


# ---------------------------------------------------------------------------
# Timing that cannot be a measurement
# ---------------------------------------------------------------------------


def test_a_first_token_before_the_send_is_refused_a_ratio():
    """A negative time to first token is not a fast server, it is a broken clock, and
    without this check it is the best score the metric can award: r_ttft is negative,
    a reply under two tokens has r_tpot 0.0, and max() of the two is 0.0 -- exactly
    the ratio a request answered instantly would earn.

    The shape here is the one a framework stamping wall-clock send times against
    monotonic packet arrivals produces."""
    ev = _ev("v", "t0", 1_800_000_000.0, 0.5, [0.5])
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    assert "v" not in res.per_event_ratio
    assert not res.aggregate_passed
    assert res.diagnostics["n_impossible_timing"] == 1


def test_a_last_token_before_the_first_is_refused_a_ratio():
    """The other direction the same broken clock produces: a stream whose tokens run
    backwards would be charged a negative inter-token latency, which again beats
    every honest request."""
    ev = _ev("v", "t0", 0.0, 0.1, [0.1, 0.05])
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    assert "v" not in res.per_event_ratio
    assert not res.aggregate_passed
    assert res.diagnostics["n_impossible_timing"] == 1


def test_impossible_timing_is_counted_apart_from_a_missing_budget():
    """Both refusals read identically in ``n_missing``, and they want opposite fixes:
    one is a task that did not stamp its prompt lengths, the other is a harness whose
    clocks disagree."""
    sloped = ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50,
                                     ttft_ms_per_prefill_token=0.01)
    backwards = _deep("back", t_send=1_800_000_000.0, t_first=0.5, prefill=1_000)
    unstamped = _deep("flat", t_send=0.0, t_first=0.5)
    res = _apply(sloped, Trace(events=[backwards, unstamped]))
    assert res.per_event_ratio == {}
    assert res.diagnostics["n_missing"] == 2
    assert res.diagnostics["n_impossible_timing"] == 1
    assert res.diagnostics["n_no_budget"] == 1


def test_a_cut_request_whose_cut_precedes_its_send_is_refused_a_ratio():
    """The cut path reaches the same arithmetic through the phantom token, so it needs
    the same refusal: a cut instant before the send would make the phantom arrive
    before the request left."""
    ev = _cut("c", t_send=1_800_000_000.0, t_cut=4.0)
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50), Trace(events=[ev]))
    assert "c" not in res.per_event_ratio
    assert not res.aggregate_passed
    assert res.diagnostics["n_impossible_timing"] == 1


def test_an_honest_request_is_not_called_impossible():
    """The check has to be silent on every request that is merely slow, or it would
    refuse the traffic it exists to price."""
    slow = _ev("slow", "t0", 0.0, 9.0, [9.0, 9.5, 10.0])
    instant = _ev("instant", "t0", 0.0, 0.0, [0.0, 0.0])
    res = _apply(ContextLinearLatencySLO(ttft_ms=1000, tpot_ms=50),
                 Trace(events=[slow, instant]))
    assert res.diagnostics["n_impossible_timing"] == 0
    assert set(res.per_event_ratio) == {"slow", "instant"}


def test_the_flat_slo_prices_a_cut_request_as_it_always_did():
    """The phantom token and the timing refusals are ContextLinearLatencySLO's. A flat SLO
    keeps the pricing every task declared before them: a cut request that streamed nothing
    reached no first token, so it has no ratio, and its diagnostics carry no new counts."""
    res = _apply(StreamingLatencySLO(ttft_ms=1000, tpot_ms=50),
                 Trace(events=[_cut("c", t_send=1.0, t_cut=4.0)]))
    assert "c" not in res.per_event_ratio
    assert not res.aggregate_passed
    assert "n_impossible_timing" not in res.diagnostics and "n_no_budget" not in res.diagnostics
