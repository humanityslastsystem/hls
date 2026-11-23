"""The sampling gate end to end: contract checks, the three rejections, the wiring.

A fake engine supplies the two capabilities the verifier composes, so these tests drive
the real ``verify`` body -- the partition, the reference item it builds, the batching,
the reduction and the verdict -- against chosen reports and a chosen reference. What the
statistics themselves mean is pinned in ``test_report.py``; here the question is which
check fires, and what the verdict says about it.
"""

import asyncio
import math

import numpy as np
import pytest

from hls.core.capabilities import ReferenceLogprobs, StackReport
from hls.core.decode import Greedy, Sample
from hls.core.payloads import Envelope, TextPayload, TokenLogprob, TopLogprob
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.reference.references import HFReference
from hls.verify import EngineAccessor, ReportFaithfulnessVerifier, WiringError
from hls.verify.calibration import CalibrationError
from hls.verify.report import (MASS_TOL, PAD_ID, TOP_LOGPROBS_KEPT,
                               TOP_LOGPROBS_REQUESTED)
from tests.unit.calibration_fixtures import fixture_workload, make_report_calibration

_STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"},
          "device_name": "B200"}

#: The fixture checkpoint's end-of-turn id and context window, as the fake reference
#: reports them.
_EOT = 9
_MAX_CONTEXT = 4096

#: A distribution over four ids, the report every fixture position carries. The atoms
#: hold 0.9 and the tail 0.1, so a served token outside the atoms is well defined.
_IDS = (1, 2, 3, 4)
_PROBS = (0.4, 0.3, 0.15, 0.05)


def _report_entry(served: int, probs=_PROBS) -> TokenLogprob:
    """One position's report: the served token's log-probability, and a full-width list
    of alternatives so the partition holds.

    The four real atoms are padded out with ids no served token uses, at a
    log-probability low enough to leave the atoms' mass where it is stated."""
    filler = [(100 + i, -30.0 - i) for i in range(TOP_LOGPROBS_REQUESTED - len(_IDS))]
    alts = list(zip(_IDS, [math.log(p) for p in probs])) + filler
    emitted = dict(alts).get(served, math.log(1e-9))
    return TokenLogprob(
        token="t", token_id=served, logprob=emitted,
        top_logprobs=[TopLogprob(token=f"a{i}", token_id=i, logprob=v)
                      for i, v in alts])


def _event(req: str, served: list[int], *, finish_reason: str = "stop",
           max_tokens: int = 1_000, logprobs: list | None = None,
           tokens: list[int] | None = None) -> TraceEvent:
    """One trace event: a reply that served ``served`` and reported each position.

    ``logprobs``/``tokens`` override the report and the emitted ids, which is how the
    contract-violation tests break one thing at a time."""
    entries = [_report_entry(t) for t in served] if logprobs is None else logprobs
    out = Envelope(request_id=req,
                   contents=[TextPayload(text="x",
                                         tokens=served if tokens is None else tokens,
                                         logprobs=entries,
                                         finish_reason=finish_reason)])
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")],
                   metadata={"messages": [{"role": "user", "content": "Q?"}],
                             "tenant_id": "t0", "max_tokens": max_tokens})
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out),
                      ground_truth=None, t_send=0.0, t_recv=1.0)


def _eot_prices(served: list[int], *, cut: bool = False) -> list[float]:
    """The reference's log end-of-turn probability at each served position.

    A reply that ends where the model wanted to end is priced high at its last
    position, so no threshold prices it and the count reads nothing there. ``cut``
    prices that ending like every other position instead, which is the early stop the
    count exists to see: the reply still ends on an end-of-turn id and still reports a
    normal stop, so every contract check and every report statistic passes it."""
    prices = [math.log(1e-6)] * len(served)
    if served and served[-1] == _EOT and not cut:
        prices[-1] = math.log(0.9)
    return prices


def _honest_reply(served: list[int], *, shift: float = 0.0, n_ctx: int = 32,
                  cut: bool = False) -> dict:
    """A reference that agrees with the fixture report, optionally shifted.

    ``shift`` moves every atom's reference log-probability down and the tail's up by the
    same amount, which raises the divergence without changing the report -- the wrong
    distribution the first check exists to see."""
    row = [math.log(p) - shift for p in _PROBS] + [-60.0] * (TOP_LOGPROBS_KEPT - len(_PROBS))
    return {"p_on_q": [list(row) for _ in served],
            "p_tail_lp": [math.log(0.1) + shift for _ in served],
            "p_eot_lp": _eot_prices(served, cut=cut),
            "n_ctx": n_ctx, "eot_ids": [_EOT], "max_context": _MAX_CONTEXT}


class _Proxy:
    """Fake engine proxy for the two capabilities the verifier composes: the reference's
    log-probabilities at the ids each item names, and the stack report.

    ``replies`` is a callable from one batch of items to the replies for it, so a test
    can answer per item. Every batch it was called with is recorded, which is how the
    batching test reads the chunk sizes."""

    def __init__(self, replies, stack=None):
        self._replies = replies
        self._stack = _STACK if stack is None else stack
        self.batches: list[list[Envelope]] = []

    async def reference_logprobs(self, items):
        self.batches.append(list(items))
        return self._replies(items)

    async def stack_report(self):
        return self._stack


def _acc(events, proxy):
    return EngineAccessor(
        artifact_events=events,
        cap_to_provider={ReferenceLogprobs: proxy.reference_logprobs,
                         StackReport: proxy.stack_report},
    )


def _from_items(items):
    """Reference replies that agree with whatever each item served."""
    return [_honest_reply(item.first_of("text").tokens) for item in items]


def _verifier(cal=None, **kwargs) -> ReportFaithfulnessVerifier:
    return ReportFaithfulnessVerifier(
        reference=HFReference(decode=Sample(temperature=1.0)),
        calibration=cal if cal is not None else make_report_calibration(), **kwargs)


def _draw(n: int, seed: int = 3) -> list[int]:
    """``n`` served tokens drawn from the fixture report, so an honest trace's own
    transforms are uniform and its moment is centered."""
    rng = np.random.default_rng(seed)
    cell = rng.choice(len(_IDS) + 1, size=n, p=list(_PROBS) + [0.1])
    return [_IDS[c] if c < len(_IDS) else 500 for c in cell]


def _run(v, events, proxy=None):
    proxy = proxy if proxy is not None else _Proxy(_from_items)
    return asyncio.run(v.verify(_acc(events, proxy)))


# --- a passing trace -----------------------------------------------------------


def test_an_honest_trace_passes_and_publishes_every_number():
    """A server whose reports match the reference and whose tokens are draws from them:
    no rejection, and each check's measurement sits beside the value it was read
    against."""
    served = _draw(600)
    events = [_event("r0", served[:300] + [_EOT]),
              _event("r1", served[300:] + [_EOT])]
    verdict = _run(_verifier(), events)
    d = verdict.diagnostics
    assert verdict.passed is True, d["rejections"]
    assert d["rejections"] == []
    assert verdict.per_event == {}
    assert set(verdict.event_diagnostics) == {"t0:r0", "t0:r1"}
    assert d["n_replies"] == 2 and d["n_tokens"] == 602
    assert d["mean_kl"] == pytest.approx(0.0, abs=1e-9)
    assert d["ceiling"] == pytest.approx(d["tolerance"] * d["kappa"])
    assert d["utilization"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= d["cvm_p"] <= 1.0
    assert d["z_critical"] == pytest.approx(2.5758293035489004)
    # Check 5: the fixture report leaves 0.1 in its tail and the calibration was taken
    # at that rate, so an honest trace sits near the ceiling and inside the limit.
    assert d["outside_rate"] == pytest.approx(d["n_served_outside"] / d["n_tokens"])
    assert d["outside_limit"] == pytest.approx(d["tolerance"] * d["outside_ceiling"])
    assert 0.0 < d["outside_rate"] <= d["outside_limit"]
    assert d["outside_utilization"] == pytest.approx(d["outside_rate"] / d["outside_limit"])
    assert d["n_violations"] == 0 and d["violations"] == []


def test_the_per_event_record_is_the_reply_report():
    """The sidecar the runner persists is each reply's own record, keyed on the event
    key because the gate is pooled and spans tenants."""
    verdict = _run(_verifier(), [_event("r0", _draw(40) + [_EOT])])
    rec = verdict.event_diagnostics["t0:r0"]
    assert rec["event_key"] == "t0:r0" and rec["n_tokens"] == 41
    assert rec["violation"] is None and rec["ends_with_eot"] is True
    assert rec["at_cap"] is False


def test_the_per_event_record_carries_no_per_position_arrays():
    """The record is persisted per event, so the two arrays would put two floats per
    served token in the verdict -- the largest thing in it, and read by nothing: the
    trace is measured once, and the statistics over both are trace-level. Their lengths
    survive as the count beside them: both are one value per served token."""
    verdict = _run(_verifier(), [_event("r0", _draw(40) + [_EOT])])
    rec = verdict.event_diagnostics["t0:r0"]
    assert "u" not in rec and "p_eot_lp" not in rec
    assert rec["n_tokens"] == 41


# --- the three checks ----------------------------------------------------------


def test_divergence_rejects_a_report_far_from_the_reference():
    """The reports are internally consistent and the tokens are honest draws from them,
    so both exact checks pass; what fails is the distribution itself."""
    served = _draw(400)
    events = [_event("r0", served + [_EOT])]

    def replies(items):
        return [_honest_reply(item.first_of("text").tokens, shift=1.5)
                for item in items]

    verdict = _run(_verifier(), events, _Proxy(replies))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["rejections"] == ["divergence"]
    assert d["mean_kl"] > d["ceiling"] and d["utilization"] > 1.0


def test_divergence_exactly_at_the_ceiling_passes():
    """Strict ``>``: a trace whose mean sits exactly on its ceiling is inside it.

    The reference is shifted from the report, so every position diverges by the same
    amount; the ceiling is then set to the mean that produced, which is what makes the
    boundary exact rather than approximately exact."""
    events = [_event("r0", _draw(400) + [_EOT])]

    def replies(items):
        return [_honest_reply(item.first_of("text").tokens, shift=0.5)
                for item in items]

    loose = _run(_verifier(), events, _Proxy(replies))
    measured = loose.diagnostics["mean_kl"]
    assert measured > 0.1
    on_the_line = _verifier(
        make_report_calibration(engines={"vllm": {"mean_kl": measured, "se_kl": 0.0}}),
        tolerance=1.0)
    verdict = _run(on_the_line, events, _Proxy(replies))
    assert verdict.diagnostics["utilization"] == 1.0
    assert "divergence" not in verdict.diagnostics["rejections"]
    # A hair under the same mean, and it fires.
    tight = _verifier(
        make_report_calibration(
            engines={"vllm": {"mean_kl": measured * (1 - 1e-9), "se_kl": 0.0}}),
        tolerance=1.0)
    assert "divergence" in _run(tight, events, _Proxy(replies)).diagnostics["rejections"]


def test_an_unbounded_divergence_rejects():
    """A report that puts mass where the reference puts none diverges without bound.
    That is a verdict, not a number to discard, so the comparison is unguarded."""
    events = [_event("r0", _draw(100) + [_EOT])]

    def replies(items):
        n = len(items[0].first_of("text").tokens)
        # The reference gives the reported atoms all of its mass, so the report's tail
        # is charged against a reference tail of zero.
        row = [math.log(p) for p in _PROBS] + [-60.0] * (TOP_LOGPROBS_KEPT - 4)
        served = items[0].first_of("text").tokens
        return [{"p_on_q": [list(row)] * n, "p_tail_lp": [-math.inf] * n,
                 "p_eot_lp": _eot_prices(served),
                 "n_ctx": 32, "eot_ids": [_EOT], "max_context": _MAX_CONTEXT}]

    verdict = _run(_verifier(), events, _Proxy(replies))
    assert verdict.passed is False
    assert "divergence" in verdict.diagnostics["rejections"]
    assert verdict.diagnostics["mean_kl"] == math.inf


def test_rank_uniformity_rejects_a_server_that_reports_one_thing_and_serves_another():
    """The report is the reference's own distribution, so the divergence is zero; every
    served token is the reported mode. Only the two exact checks can see it, which is
    what they are for."""
    events = [_event("r0", [_IDS[0]] * 400 + [_EOT])]
    verdict = _run(_verifier(), events)
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["mean_kl"] == pytest.approx(0.0, abs=1e-9)
    assert "rank_uniformity" in d["rejections"]
    assert d["cvm_p"] < d["alpha"]


def test_log_score_rejects_the_same_deviation_through_its_own_functional():
    """The moment test reads the same null as the rank test; a greedy server fails both,
    and the verdict names each one that fired."""
    events = [_event("r0", [_IDS[0]] * 400 + [_EOT])]
    verdict = _run(_verifier(), events)
    d = verdict.diagnostics
    assert "log_score" in d["rejections"]
    assert abs(d["z"]) > d["z_critical"]
    assert d["n_moment"] == 401


def _rare_entry(served: int) -> TokenLogprob:
    """One position's report from a server hiding in the partition's tail: the served
    token with its honest log-probability, and alternatives that are all tokens of
    negligible probability, so the served token is never an atom.

    The 21 alternatives sit at distinct log-probabilities from -30 down, so the
    partition keeps 19 of them and every served token falls in a tail of mass one minus
    about 1e-13."""
    alts = [(100 + i, -30.0 - i) for i in range(TOP_LOGPROBS_REQUESTED)]
    honest = dict(zip(_IDS, [math.log(p) for p in _PROBS]))
    return TokenLogprob(
        token="t", token_id=served, logprob=honest.get(served, math.log(1e-9)),
        top_logprobs=[TopLogprob(token=f"a{i}", token_id=i, logprob=v)
                      for i, v in alts])


def _rare_reply(served: list[int]) -> dict:
    """A reference that agrees with ``_rare_entry`` exactly: the same negligible mass
    at the 19 kept atoms, and the rest of the vocabulary in the tail."""
    kept = [-30.0 - i for i in range(TOP_LOGPROBS_KEPT)]
    tail = math.log1p(-sum(math.exp(v) for v in kept))
    return {"p_on_q": [list(kept) for _ in served],
            "p_tail_lp": [tail for _ in served],
            "p_eot_lp": _eot_prices(served),
            "n_ctx": 32, "eot_ids": [_EOT], "max_context": _MAX_CONTEXT}


def test_served_outside_rejects_a_report_of_rare_alternatives():
    """The reporter chooses the partition. A server that reports rare tokens as its
    alternatives, matches the reference on them exactly, and serves the mode at every
    position passes the three report statistics -- the divergence is on atoms of no
    mass, the transform is uniform over a tail that is the whole interval, the moment
    vanishes with the atoms -- and is caught only by how often its served token is
    outside its own atoms: always, against a calibrated rate of a tenth."""
    served = [_IDS[0]] * 400 + [_EOT]
    events = [_event("r0", served, logprobs=[_rare_entry(t) for t in served])]
    proxy = _Proxy(lambda items: [_rare_reply(item.first_of("text").tokens)
                                  for item in items])
    verdict = _run(_verifier(), events, proxy)
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["rejections"] == ["served_outside"]
    assert d["mean_kl"] == pytest.approx(0.0, abs=1e-9)
    assert d["cvm_p"] >= d["alpha"]
    assert abs(d["z"]) <= d["z_critical"]
    assert d["n_served_outside"] == 401 and d["outside_rate"] == pytest.approx(1.0)
    assert d["outside_rate"] > d["outside_limit"]


def test_the_declared_tolerance_widens_the_served_outside_limit():
    """The limit is the calibrated rate times the same tolerance that widens the
    divergence ceiling: an honest trace at a tenth outside, calibrated at a twentieth,
    fails at the default headroom and passes when the headroom covers it."""
    served = _draw(600)
    events = [_event("r0", served[:300] + [_EOT]), _event("r1", served[300:] + [_EOT])]
    cal = make_report_calibration(engines={"vllm": {"outside_rate": 0.04,
                                                    "se_outside": 0.001}})
    tight = _run(_verifier(cal), events)
    assert tight.diagnostics["rejections"] == ["served_outside"]
    assert tight.diagnostics["outside_limit"] == pytest.approx(2.0 * cal.outside_ceiling)
    wide = _run(_verifier(cal, tolerance=3.0), events)
    assert wide.passed is True, wide.diagnostics["rejections"]
    assert wide.diagnostics["outside_limit"] == pytest.approx(3.0 * cal.outside_ceiling)


def test_the_end_of_turn_count_passes_a_server_that_ends_where_the_reference_does():
    """Every reply ends on the end-of-turn id at a position the reference prices at 0.9,
    which no threshold prices, so the count reads nothing and publishes the numbers it
    read nothing from."""
    events = [_event(f"r{i}", _draw(200, seed=i) + [_EOT]) for i in range(3)]
    d = _run(_verifier(), events).diagnostics
    assert "end_of_turn_count" not in d["rejections"]
    eot = d["end_of_turn"]
    assert eot["eps"] == 0.1
    assert d["n_tokens"] == 603
    assert eot["priced_positions"] == 600  # every position but the three endings
    assert eot["observed"] == 0 and eot["p_value"] == 1.0
    assert eot["expected"] == pytest.approx(600 * 1e-6)


def test_the_end_of_turn_count_rejects_replies_cut_where_the_reference_would_not():
    """The deviation the other three checks cannot see at one trace: the replies are
    honest draws from honest reports and end on a real end-of-turn id with a normal
    stop, so nothing about them violates the contract -- they just end where the
    reference gives ending a probability of a millionth."""
    events = [_event(f"r{i}", _draw(200, seed=i) + [_EOT]) for i in range(3)]

    def replies(items):
        return [_honest_reply(item.first_of("text").tokens, cut=True)
                for item in items]

    verdict = _run(_verifier(), events, _Proxy(replies))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["rejections"] == ["end_of_turn_count"]
    eot = d["end_of_turn"]
    assert eot["observed"] == 3 and eot["expected"] < 0.01
    assert eot["p_value"] <= d["alpha"]


def test_the_declared_tolerance_does_not_widen_the_end_of_turn_count():
    """``tolerance`` is the headroom over a measured ceiling, and this check has none:
    its null mean comes from the trace itself, so there is nothing for a multiplier to
    multiply."""
    events = [_event(f"r{i}", _draw(200, seed=i) + [_EOT]) for i in range(3)]

    def replies(items):
        return [_honest_reply(item.first_of("text").tokens, cut=True)
                for item in items]

    # Membership rather than equality: at a tolerance of 1 the served-outside limit is
    # the fixture's own calibrated rate with no headroom, which an honest draw crosses
    # by chance about half the time. That check is allowed to fire here; this one has
    # to, at both ends.
    for tolerance in (1.0, 100.0):
        d = _run(_verifier(tolerance=tolerance), events, _Proxy(replies)).diagnostics
        assert "end_of_turn_count" in d["rejections"], tolerance
        assert d["end_of_turn"]["p_value"] <= d["alpha"]


def test_a_trace_with_no_measured_position_is_rejected():
    """An empty trace makes every check vacuous, so a pass would rest on no evidence at
    all rather than on thin evidence."""
    verdict = _run(_verifier(), [])
    assert verdict.passed is False
    assert verdict.diagnostics["rejections"] == ["no_measured_positions"]


# --- the contract checks -------------------------------------------------------


def _violation(verdict) -> str:
    assert verdict.passed is False
    assert "contract_violations" in verdict.diagnostics["rejections"]
    assert verdict.diagnostics["n_violations"] == 1
    return verdict.diagnostics["violations"][0]["reason"]


def test_a_reply_with_no_report_violates_the_contract():
    ev = _event("r0", [_IDS[0], _EOT], logprobs=None)
    ev.response.output.first_of("text")
    bare = Envelope(request_id="r0",
                    contents=[TextPayload(text="x", tokens=[_IDS[0], _EOT],
                                          finish_reason="stop")])
    ev = TraceEvent(request_id="r0", input=ev.input,
                    response=ArtifactResponse(output=bare), t_send=0.0, t_recv=1.0)
    reason = _violation(_run(_verifier(), [ev]))
    assert "no reported logprobs" in reason


def test_a_report_shorter_than_the_reply_violates_the_contract():
    served = [_IDS[0], _IDS[1], _EOT]
    ev = _event("r0", served, logprobs=[_report_entry(t) for t in served[:2]])
    assert "2 reported positions for 3 served tokens" in _violation(_run(_verifier(), [ev]))


def test_a_position_reporting_another_token_violates_the_contract():
    served = [_IDS[0], _EOT]
    entries = [_report_entry(served[0]), _report_entry(served[1])]
    entries[1] = TokenLogprob(token="t", token_id=77, logprob=-1.0,
                              top_logprobs=entries[1].top_logprobs)
    ev = _event("r0", served, logprobs=entries)
    assert "reports token 77, served 9" in _violation(_run(_verifier(), [ev]))


def test_too_few_alternatives_violates_the_contract():
    served = [_IDS[0], _EOT]
    narrow = TokenLogprob(
        token="t", token_id=served[0], logprob=math.log(_PROBS[0]),
        top_logprobs=[TopLogprob(token="a", token_id=i, logprob=-1.0 - i)
                      for i in range(5)])
    ev = _event("r0", served, logprobs=[narrow, _report_entry(served[1])])
    assert ("5 usable alternatives (5 distinct as reported)"
            in _violation(_run(_verifier(), [ev])))


def test_a_position_over_unit_mass_violates_the_contract():
    """A position cannot hold more than one of probability, so a report saying it does
    is not the distribution the token was drawn from and the reply fails on it."""
    served = _draw(40, seed=3) + [_EOT]
    entries = [_report_entry(t) for t in served]
    entries[-1] = _report_entry(_EOT, probs=(0.5, 0.4, 0.3, 0.2))
    ev = _event("r0", served, logprobs=entries)
    assert _violation(_run(_verifier(), [ev])) == (
        f"position {len(served) - 1} reports 1.4 of probability over its alternatives, "
        f"more than the one a distribution has")


def _deviant_entry(served: int) -> TokenLogprob:
    """One position's report from a server whose distribution is not the reference's,
    stated at just over unit mass.

    The reference's own probabilities, reversed -- so the position diverges from it by a
    lot -- and scaled until the atoms carry more mass than a position has."""
    over = (1.0 + 4 * MASS_TOL) / sum(_PROBS)
    return _report_entry(served, probs=tuple(p * over for p in reversed(_PROBS)))


def test_excess_mass_cannot_hide_the_positions_a_server_deviates_at():
    """A server free to deviate wherever it overstates its mass would be graded only
    where it chose to be: here one reply in two deviates at a twentieth of its
    positions, and states more than unit mass at exactly those. The reply fails the
    contract instead, and the trace fails with it.

    The measurement left behind is the other reply's, and its divergence is nothing --
    which is what the deviating positions were worth to a gate that skipped them."""
    clean = _draw(300, seed=1) + [_EOT]
    hidden = _draw(300, seed=2) + [_EOT]
    deviating = set(range(0, len(hidden), 20))
    entries = [_deviant_entry(t) if i in deviating else _report_entry(t)
               for i, t in enumerate(hidden)]
    verdict = _run(_verifier(), [_event("r0", clean),
                                 _event("r1", hidden, logprobs=entries)])
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["rejections"] == ["contract_violations"]
    assert d["n_violations"] == 1 and d["violations"][0]["event"] == "t0:r1"
    assert "reports 1.0004 of probability" in d["violations"][0]["reason"]
    assert d["n_tokens"] == len(clean)
    assert d["mean_kl"] == pytest.approx(0.0, abs=1e-9)


def test_a_served_token_reported_at_probability_zero_violates_the_contract():
    """The atoms hold the whole of the position's mass and the served token is not one
    of them, so the report gives the token it served no probability at all -- a draw the
    report says could not have happened."""
    served = _draw(20, seed=4) + [_EOT]
    entries = [_report_entry(t) for t in served]
    # Four atoms holding unit mass between them, and a served token outside them whose
    # own entry underflows to zero probability.
    entries[0] = TokenLogprob(
        token="t", token_id=served[0], logprob=-800.0,
        top_logprobs=[TopLogprob(token=f"a{i}", token_id=i, logprob=v)
                      for i, v in [(600 + j, math.log(0.25)) for j in range(4)]
                      + [(700 + j, -30.0 - j)
                         for j in range(TOP_LOGPROBS_REQUESTED - 4)]])
    ev = _event("r0", served, logprobs=entries)
    assert _violation(_run(_verifier(), [ev])) == (
        "position 0 reports the token it served at probability zero")


def test_a_reported_value_that_is_not_a_number_violates_the_contract():
    """A value that is not a number is a verdict the contestant gets, not a grading run
    that ends in an exception: it compares false against every bound the mass arithmetic
    makes, so it would reach the reduction and fail there instead."""
    served = _draw(20, seed=5) + [_EOT]
    entries = [_report_entry(t) for t in served]
    entries[3] = TokenLogprob(
        token="t", token_id=served[3], logprob=float("nan"),
        top_logprobs=entries[3].top_logprobs)
    verdict = _run(_verifier(), [_event("r0", served, logprobs=entries)])
    assert verdict.passed is False
    assert _violation(verdict) == (
        "position 3 reports a log-probability that is not a number: nan")


def test_a_reported_value_above_zero_violates_the_contract():
    """No token holds more probability than the position it sits in, so a positive
    log-probability is not one, whatever mass the row it sits in adds up to."""
    served = _draw(20, seed=6) + [_EOT]
    entries = [_report_entry(t) for t in served]
    alts = list(entries[1].top_logprobs)
    alts[2] = TopLogprob(token="a", token_id=alts[2].token_id, logprob=800.0)
    entries[1] = TokenLogprob(token="t", token_id=served[1],
                              logprob=entries[1].logprob, top_logprobs=alts)
    assert _violation(_run(_verifier(), [_event("r0", served, logprobs=entries)])) == (
        "position 1 reports a log-probability above zero: 800")


def test_a_reply_with_no_token_ids_violates_the_contract():
    inp = Envelope(request_id="r0", contents=[TextPayload(text="Q?")],
                   metadata={"messages": [{"role": "user", "content": "Q?"}],
                             "tenant_id": "t0", "max_tokens": 100})
    out = Envelope(request_id="r0",
                   contents=[TextPayload(text="x", finish_reason="stop")])
    ev = TraceEvent(request_id="r0", input=inp, response=ArtifactResponse(output=out),
                    t_send=0.0, t_recv=1.0)
    assert "no emitted token ids" in _violation(_run(_verifier(), [ev]))


def test_a_reply_served_past_its_cap_violates_the_contract():
    """The request carried the cap and the reply ignored it, which violates the contract:
    grading fails the trace on it rather than scoring the excess."""
    served = _draw(20) + [_EOT]
    ev = _event("r0", served, max_tokens=10)
    assert "past its cap of 10" in _violation(_run(_verifier(), [ev]))


def test_an_empty_reply_violates_the_contract():
    """A reply that served nothing reports nothing and is graded by nothing, so it
    would otherwise be a way out of the gate entirely."""
    ev = _event("r0", [], logprobs=[])
    assert "served no tokens" in _violation(_run(_verifier(), [ev]))


def test_an_id_outside_the_vocabulary_violates_the_contract():
    """The reference closes the reply rather than indexing its embedding table with an
    id it does not have, and the verdict carries the reason it gave."""
    ev = _event("r0", [_IDS[0], _EOT])
    proxy = _Proxy(lambda items: [{"invalid": "served token id 999999 is outside the "
                                              "model's vocabulary of 256000"}])
    assert "outside the model's vocabulary" in _violation(_run(_verifier(), [ev], proxy))


def test_a_normal_stop_without_an_end_of_turn_token_violates_the_contract():
    """A reply that claims to have ended its turn must have ended it in band: stopping
    on an ordinary token serves fewer tokens for the same credit, and every token it did
    serve is a faithful draw, so no statistic over them can see it."""
    ev = _event("r0", _draw(20), finish_reason="stop")
    assert "not one of the checkpoint's end-of-turn ids" in _violation(_run(_verifier(), [ev]))


def test_a_length_stop_short_of_the_cap_violates_the_contract():
    ev = _event("r0", _draw(20), finish_reason="length", max_tokens=8192)
    reason = _violation(_run(_verifier(), [ev]))
    assert "finish_reason='length' after 20 of 8192 tokens" in reason


def test_a_length_stop_at_the_cap_is_the_contract_being_met():
    served = _draw(20)
    ev = _event("r0", served, finish_reason="length", max_tokens=len(served))
    verdict = _run(_verifier(), [ev])
    assert verdict.diagnostics["n_violations"] == 0
    assert verdict.event_diagnostics["t0:r0"]["at_cap"] is True


def test_a_length_stop_with_the_context_exhausted_is_exempt():
    """Running out of context is a fact about the prompt that the reference measured,
    not a claim of the server's, so it is the one exemption from the budget rule."""
    served = _draw(20)
    ev = _event("r0", served, finish_reason="length", max_tokens=8192)
    proxy = _Proxy(lambda items: [
        _honest_reply(item.first_of("text").tokens, n_ctx=_MAX_CONTEXT - len(served))
        for item in items])
    verdict = _run(_verifier(), [ev], proxy)
    assert verdict.diagnostics["n_violations"] == 0


def test_the_exemption_is_measured_at_the_window_the_task_declares():
    """The window a server fills is the one the task told it to serve in, not the one
    the checkpoint could do.

    The reference is built with no ``max_model_len``, so it reports the checkpoint's
    own declared window; the task's servers and the calibration pool's members are
    built at ``TaskSpec.max_context_len``, which the calibration's fingerprint payload
    carries. Here the task declares half the checkpoint's window and the reply fills
    it: reading the checkpoint's would fail the whole trace for an honest server
    obeying the brief."""
    served = _draw(20)
    declared = _MAX_CONTEXT // 2
    cal = make_report_calibration(**fixture_workload(max_context_len=declared))
    ev = _event("r0", served, finish_reason="length", max_tokens=8192)
    proxy = _Proxy(lambda items: [
        _honest_reply(item.first_of("text").tokens, n_ctx=declared - len(served))
        for item in items])

    assert _run(_verifier(cal), [ev], proxy).diagnostics["n_violations"] == 0
    # The contrast: a calibration whose task declared no window reads the same reply
    # against the checkpoint's, which it is nowhere near, and fails the trace on it.
    assert "finish_reason='length'" in _violation(_run(_verifier(), [ev], proxy))


def test_no_declared_and_no_checkpoint_window_gets_no_exemption():
    """An unverifiable exemption is a free pass, so a reply with a window from neither
    source fails closed."""
    served = _draw(20)
    ev = _event("r0", served, finish_reason="length", max_tokens=8192)

    def replies(items):
        out = _honest_reply(items[0].first_of("text").tokens)
        out["max_context"] = None
        return [out]

    assert "finish_reason='length'" in _violation(_run(_verifier(), [ev], _Proxy(replies)))


def test_a_violated_reply_is_recorded_with_its_reason_and_no_measurement():
    ev = _event("r0", _draw(20), finish_reason="stop")
    verdict = _run(_verifier(), [ev])
    rec = verdict.event_diagnostics["t0:r0"]
    assert rec["violation"] is not None
    assert rec["kl_sum"] == 0.0 and rec["n_moment"] == 0
    assert rec["n_tokens"] == 20  # what the reply claimed, for the reader
    assert verdict.diagnostics["n_tokens"] == 0


def test_one_violation_fails_a_trace_that_is_otherwise_clean():
    good = [_event(f"r{i}", _draw(100, seed=i) + [_EOT]) for i in range(3)]
    bad = _event("r9", _draw(20), finish_reason="stop")
    verdict = _run(_verifier(), [*good, bad])
    assert verdict.passed is False
    assert verdict.diagnostics["rejections"] == ["contract_violations"]
    assert verdict.diagnostics["n_violations"] == 1
    assert verdict.diagnostics["n_replies"] == 4


class _OverCapReplies:
    """*n* replies past their cap, each carrying a MiB of text, made afresh on every pass and
    held by nothing but the reader, as a stored trace's view gives them."""

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        for i in range(self.n):
            ev = _event(f"r{i}", _draw(20, seed=i) + [_EOT], max_tokens=10)
            out = ev.response.output
            text = out.contents[0].model_copy(update={"text": "x" * 2**20})
            yield ev.model_copy(update={"response": ArtifactResponse(output=out.model_copy(update={"contents": [text]}))})


def test_the_report_gate_keeps_of_a_refused_reply_only_what_its_record_reads():
    """A refused reply is recorded from its key, what it served, its cap and its finish reason,
    never the event: forty refused replies of a MiB each are not held."""
    import tracemalloc

    _run(_verifier(), _OverCapReplies(1))
    tracemalloc.start()
    try:
        verdict = _run(_verifier(), _OverCapReplies(40))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert verdict.diagnostics["n_violations"] == 40
    shown = verdict.diagnostics["violations"]
    assert shown and all("past its cap of 10" in v["reason"] for v in shown)
    assert peak < 8 * 2**20
    one = next(iter(_OverCapReplies(1)))
    record = _verifier()._violated(one, shown[0]["reason"], set())
    assert verdict.event_diagnostics["t0:r0"] == record.summary_json()


# --- the reference call --------------------------------------------------------


def test_the_reference_call_is_batched():
    """A graded position crosses the wire as twenty floats each way, so the call is
    chunked; the chunking changes no scored number, only the payload size."""
    events = [_event(f"r{i}", _draw(4, seed=i) + [_EOT]) for i in range(600)]
    proxy = _Proxy(_from_items)
    verdict = _run(_verifier(), events, proxy)
    assert [len(b) for b in proxy.batches] == [256, 256, 88]
    assert verdict.diagnostics["n_replies"] == 600


def test_the_item_carries_the_context_the_cap_and_the_partition():
    """What the reference needs to reproduce the position: the conversation, the stamped
    cap, and the ids this reply's own report fixed."""
    served = [_IDS[0], _IDS[1], _EOT]
    proxy = _Proxy(_from_items)
    _run(_verifier(), [_event("r0", served, max_tokens=77)], proxy)
    item = proxy.batches[0][0]
    assert item.metadata["messages"] == [{"role": "user", "content": "Q?"}]
    assert item.metadata["max_tokens"] == 77
    report_ids = item.metadata["report_ids"]
    assert len(report_ids) == len(served)
    assert all(len(row) == TOP_LOGPROBS_KEPT for row in report_ids)
    assert PAD_ID not in report_ids[0]
    assert item.first_of("text").tokens == served


def test_a_short_report_is_counted_and_not_a_violation():
    """A report one alternative narrower than asked for still fixes the partition, so it
    describes the serving stack rather than gating anything."""
    served = [_IDS[0], _EOT]
    filler = [(100 + i, -30.0 - i) for i in range(TOP_LOGPROBS_REQUESTED - 1 - len(_IDS))]
    alts = list(zip(_IDS, [math.log(p) for p in _PROBS])) + filler
    narrow = TokenLogprob(
        token="t", token_id=served[0], logprob=math.log(_PROBS[0]),
        top_logprobs=[TopLogprob(token="a", token_id=i, logprob=v) for i, v in alts])
    ev = _event("r0", served, logprobs=[narrow, _report_entry(served[1])])
    verdict = _run(_verifier(), [ev])
    assert verdict.diagnostics["n_violations"] == 0
    assert verdict.diagnostics["n_short_positions"] == 1


def test_a_short_reference_batch_raises():
    """A reference that answers fewer items than it was asked about cannot be lined up
    with them, which is a harness fault rather than a verdict."""
    events = [_event("r0", [_IDS[0], _EOT]), _event("r1", [_IDS[1], _EOT])]
    proxy = _Proxy(lambda items: _from_items(items)[:1])
    with pytest.raises(ValueError, match="1 replies for 2 items"):
        _run(_verifier(), events, proxy)


# --- the calibration and the instrument ---------------------------------------


def test_a_missing_calibration_is_refused():
    v = ReportFaithfulnessVerifier(reference=HFReference(decode=Sample()))
    with pytest.raises(WiringError, match="calibration is None"):
        _run(v, [_event("r0", [_IDS[0], _EOT])])


def test_a_drifted_stack_is_refused():
    """The ceiling is a claim about one software stack; a reference that is not on it
    cannot reuse the claim."""
    proxy = _Proxy(_from_items, stack={**_STACK, "torch": "2.5.0"})
    with pytest.raises(CalibrationError, match="stack drifted"):
        _run(_verifier(), [_event("r0", [_IDS[0], _EOT])], proxy)


def test_a_device_name_variant_of_the_calibrated_kind_is_accepted():
    """Marketing-name variants of one kind are the same instrument, which is why the
    device name is gated by kind rather than by equality."""
    proxy = _Proxy(_from_items, stack={**_STACK, "device_name": "NVIDIA B200 SXM"})
    verdict = _run(_verifier(), [_event("r0", _draw(50) + [_EOT])], proxy)
    assert verdict.diagnostics["n_violations"] == 0


def test_a_reference_with_other_end_of_turn_ids_is_refused():
    """The end-of-turn count's threshold was chosen for one checkpoint's ids, so a
    reference that names a different set is not the instrument it was chosen on."""
    def replies(items):
        out = _from_items(items)
        for r in out:
            r["eot_ids"] = [_EOT, 77]
        return out

    with pytest.raises(CalibrationError, match="end-of-turn ids"):
        _run(_verifier(), [_event("r0", _draw(20) + [_EOT])], _Proxy(replies))


def test_the_end_of_turn_ids_are_not_checked_when_no_reply_carried_them():
    """A reply refused before the reference was called reports no ids, and neither does
    one the reference closed as unscorable. There is then nothing to compare against,
    and the trace fails on those replies rather than on a mismatch with an empty set."""
    ev = _event("r0", [_IDS[0], _EOT], logprobs=[])
    verdict = _run(_verifier(), [ev])
    assert verdict.diagnostics["rejections"] == ["contract_violations",
                                                 "no_measured_positions"]


def test_other_silicon_is_refused():
    proxy = _Proxy(_from_items, stack={**_STACK, "device_name": "NVIDIA H200"})
    with pytest.raises(CalibrationError, match="GPU-kind specific"):
        _run(_verifier(), [_event("r0", [_IDS[0], _EOT])], proxy)


def test_a_sub_one_tolerance_is_refused():
    """Gating below the clean pool's own upper limit fails a trusted stack for being
    itself."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        _verifier(tolerance=0.5)


def test_the_verifier_reads_the_report_calibration():
    assert ReportFaithfulnessVerifier.calibration_kind == "report"
    assert ReportFaithfulnessVerifier.needs_calibration is True
    assert ReportFaithfulnessVerifier.pooled is True


def test_a_flips_calibration_cannot_be_set_on_it():
    """The two artifacts calibrate different gates, and neither verifier can grade on
    the other's numbers."""
    import pydantic

    from tests.unit.calibration_fixtures import make_calibration
    with pytest.raises(pydantic.ValidationError):
        _verifier(make_calibration())


# --- wiring -------------------------------------------------------------------


def _wire(**proto_kwargs):
    """``check_wiring`` on a contract carrying the given protocol settings."""
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.wire import OpenAIChatProtocol
    from hls.verify.base import check_wiring
    contract = RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
        protocol=OpenAIChatProtocol(**proto_kwargs))
    return lambda v: check_wiring([v], contract)


def test_wiring_requires_the_report_on_the_wire():
    check = _wire()
    with pytest.raises(WiringError, match="return_logprobs"):
        check(_verifier())


def test_wiring_requires_the_full_partition_width():
    check = _wire(return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED - 1)
    with pytest.raises(WiringError, match="one more than the partition keeps"):
        check(_verifier())


def test_wiring_accepts_the_contracted_width():
    _wire(return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED)(_verifier())


def test_wiring_requires_the_gates_declared_width():
    """A gate that declares a wider report needs a contract asking for at least that
    many; the text tasks' 21 is not enough for it."""
    wide = _verifier(report_width=65)
    with pytest.raises(WiringError, match="top_logprobs >= 65"):
        _wire(return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED)(wide)


def test_the_report_width_only_widens():
    """No report narrower than the default has been validated, so the field refuses it."""
    import pydantic
    with pytest.raises(pydantic.ValidationError):
        _verifier(report_width=TOP_LOGPROBS_REQUESTED - 1)


def test_wiring_refuses_a_greedy_reference():
    """The whole measurement is about a distribution something was drawn from; under
    greedy decode there is none, and a faithful server would fail the exact checks at
    every position."""
    check = _wire(return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED)
    v = ReportFaithfulnessVerifier(reference=HFReference(decode=Greedy()),
                                   calibration=make_report_calibration())
    with pytest.raises(WiringError, match="grades a sampled decode"):
        check(v)


def test_uniforms_do_not_depend_on_the_served_tokens(monkeypatch):
    """The verifier's uniforms come from the event key, so a server cannot place its
    served token inside its reported cell to make the transforms look uniform. Pinned
    on the arguments the draw is handed: two traces that differ only in what was served
    must ask for the same uniforms."""
    import hls.verify.verifiers as verifiers
    from hls.verify.report import reply_uniforms

    asked: list[tuple[str, int]] = []

    def spy(event_key, n):
        asked.append((event_key, n))
        return reply_uniforms(event_key, n)

    monkeypatch.setattr(verifiers, "reply_uniforms", spy)
    a = _run(_verifier(), [_event("r0", _draw(60, seed=1) + [_EOT])])
    b = _run(_verifier(), [_event("r0", _draw(60, seed=2) + [_EOT])])
    assert asked == [("t0:r0", 61), ("t0:r0", 61)]
    # The same uniforms over the same trace give the same transforms; a different draw
    # against them gives different ones.
    again = _run(_verifier(), [_event("r0", _draw(60, seed=1) + [_EOT])])
    assert a.diagnostics["cvm_omega2"] == again.diagnostics["cvm_omega2"]
    assert a.diagnostics["cvm_omega2"] != b.diagnostics["cvm_omega2"]
