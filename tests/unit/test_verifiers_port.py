import asyncio

import pydantic
import pytest

from hls.core.capabilities import Generate, StackReport, TeacherForce
from hls.core.decode import Greedy
from hls.core.payloads import Envelope, TextPayload, TokenLogprob, TopLogprob
from hls.core.trace import TraceEvent
from hls.reference.references import HFReference
from hls.verify import (
    EngineAccessor,
    LetterMatchMetric,
    ReferenceRelativeVerifier,
    TokenFaithfulnessVerifier,
    WiringError,
)
from hls.verify.calibration import Calibration, CalibrationError
from tests.unit.calibration_fixtures import make_calibration


def _ref():
    return HFReference(decode=Greedy())


def _resp(out):
    from hls.core.trace import ArtifactResponse

    return ArtifactResponse(output=out)


def _event(req, text, tokens, gt=None, logprobs=None, finish_reason=None,
           max_tokens=1_000_000):
    out = Envelope(request_id=req,
                   contents=[TextPayload(text=text, tokens=tokens, logprobs=logprobs,
                                         finish_reason=finish_reason)])
    inp = Envelope(
        request_id=req,
        contents=[TextPayload(text="Q?")],
        # tenant_id and max_tokens are what the Driver stamps on every real event
        # (single-tenant tasks stamp the synthetic "t0"; the cap is the workload's).
        # The pooled verifier keys per_event on event_key, derived from the tenant,
        # and reads the cap to decide whether a reply reached it.
        metadata={"messages": [{"role": "user", "content": "Q?"}], "tenant_id": "t0",
                  "max_tokens": max_tokens},
    )
    return TraceEvent(request_id=req, input=inp, response=_resp(out), ground_truth=gt, t_send=0.0, t_recv=1.0)


def _lp(emitted: float, tops: list[float]) -> TokenLogprob:
    """One reported per-token logprob entry: the emitted token's logprob and the
    contestant's reported top alternatives at that position."""
    return TokenLogprob(
        token="t", token_id=1, logprob=emitted,
        top_logprobs=[TopLogprob(token=f"a{i}", token_id=10 + i, logprob=v)
                      for i, v in enumerate(tops)])


# --- TokenFaithfulnessVerifier -------------------------------------------------
#
# The calibration fixture pins q_max=0.004; with the default 2x rate_multiplier
# the rate rule fires above flip_rate 0.008 (a flip is gap > 0; strict >). Gap
# SIZES are gated by the flip-count curve check, whose per-level parameters come
# from the same calibration.

_STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"}, "device_name": "B200"}


def _cal(*, levels=(2, 0, 0, 0, 0, 0, 0), exposure=1_000_000, **overrides) -> Calibration:
    """The verifier fixture: one engine at q_max=0.004 with the given per-level event
    counts over ``exposure`` clean tokens. Rates are stated so the mean is exactly
    0.004, which is what makes the rate rule's firing point readable in each test."""
    return make_calibration(
        engines={"eng": {"rates": [0.004, 0.004, 0.004], "events": list(levels),
                         "exposure": exposure}}, **overrides)


class _Proxy:
    """Fake engine proxy providing the two capabilities TokenFaithfulnessVerifier
    composes: TeacherForce (scripted per-token gaps, one list per event) and
    StackReport (the reported stack dict)."""

    def __init__(self, gaps, stack):
        self._gaps = gaps
        self._stack = stack

    async def teacher_force(self, items):
        return self._gaps

    async def stack_report(self):
        return self._stack


def _acc(events, gaps, stack=None):
    proxy = _Proxy(gaps, _STACK if stack is None else stack)
    return EngineAccessor(
        artifact_events=events,
        cap_to_provider={
            TeacherForce: proxy.teacher_force,
            StackReport: proxy.stack_report,
        },
    )


# The gate publishes measurements and ceilings, not comparisons: the verdict carries no
# reject flags, so a test that wants one reads it back the way any consumer does.

def _rate_reject(d) -> bool:
    """Whether the rate rule fired: the measured flip rate over its enforced ceiling."""
    return d["flip_rate"] > d["rate_multiplier"] * d["q_max"]


def _enforced_k(row) -> int:
    """One curve row's operative ceiling: grading is against the most permissive engine."""
    return max(row["ceiling_per_engine"].values())


def _fired_levels(d) -> list[float]:
    """The gap levels whose enforced ceiling the trace's count exceeded."""
    return [row["c"] for row in d["flip_curve"] if row["count"] > _enforced_k(row)]


def _curve_reject(d) -> bool:
    return bool(_fired_levels(d))


def _insufficient(d) -> bool:
    """Whether the trace fell under the rate rule's fail-closed floor."""
    return d["n_tokens"] < d["min_tokens"]


def _underpowered(d) -> bool:
    """Whether it was graded anyway, on the task's declaration."""
    return _insufficient(d) and d["allow_underpowered_trace"]


def _verifier(cal, **kwargs):
    """The fixture verifier. ``allow_underpowered_trace`` by default: these traces are
    tens to thousands of tokens, far under any real floor, because they exercise the
    RULES rather than the floor -- the floor has its own tests below. The curve
    multiplier is pinned at 1 so the curve tests read the calibrated ceilings
    themselves; the shipped default has its own test."""
    kwargs.setdefault("allow_underpowered_trace", True)
    kwargs.setdefault("curve_multiplier", 1.0)
    return TokenFaithfulnessVerifier(reference=_ref(), calibration=cal, **kwargs)


def test_passes_when_all_gaps_zero_and_above_floor():
    """All-zero gaps over enough tokens: no flips -> pass, and the always-present
    rate_ratio is 0.0 (rate_thresh is nonzero)."""
    v = _verifier(_cal())
    events = [_event("r0", "B", [1] * 20)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 20])))
    assert verdict.passed is True
    # No per-event map: the gate is trace-level, so the Runner reads ``passed`` for
    # every event it covers. The per-event detail is in event_diagnostics.
    assert verdict.per_event == {}
    assert set(verdict.event_diagnostics) == {"t0:r0"}
    assert verdict.diagnostics["rate_ratio"] == 0.0
    assert verdict.diagnostics["n_tokens"] == 20 and verdict.diagnostics["n_flips"] == 0


def test_rate_reject_when_flip_rate_over_ceiling():
    """flip_rate = 3x q_max = 0.012 > rate_thresh (0.008): the rate rule fires."""
    v = _verifier(_cal())
    e0 = _event("r0", "B", [1] * 500)
    e1 = _event("r1", "C", [1] * 500)
    gaps = [[0.1] * 12 + [0.0] * 488, [0.0] * 500]  # 12 flips / 1000 tokens
    verdict = asyncio.run(v.verify(_acc([e0, e1], gaps)))
    assert verdict.passed is False
    assert set(verdict.event_diagnostics) == {"t0:r0", "t0:r1"}
    assert _rate_reject(verdict.diagnostics) is True
    assert _curve_reject(verdict.diagnostics) is False
    assert verdict.diagnostics["flip_rate"] == pytest.approx(0.012)


def test_flip_rate_exactly_at_ceiling_passes_strict_gt():
    """flip_rate exactly == rate_thresh (0.008) does NOT fire the rate rule (strict >)."""
    v = _verifier(_cal())
    gaps = [[0.1] * 8 + [0.0] * 992]  # 8 flips / 1000 tokens == 0.008
    events = [_event("r0", "B", [1] * 1000)]
    verdict = asyncio.run(v.verify(_acc(events, gaps)))
    assert verdict.passed is True
    assert verdict.diagnostics["rate_ratio"] == pytest.approx(1.0)


# --- the flip-count curve check --------------------------------------------------


def test_curve_boundary_exactly_k_passes_k_plus_one_rejects():
    """The curve rule is strict >: observing exactly k(c, N) flips with gap >= c is
    the last count inside the clean envelope; k+1 rejects. Counts small enough
    that the RATE rule stays silent, so the curve rejects alone."""
    cal = _cal()
    v = _verifier(cal)
    n = 2000
    k = cal.thresholds(n)[0]["k"]  # level c=0.5
    assert 0 < k < 8  # the boundary is actually exercised (and rate stays under)

    at_k = [[1.5] * k + [0.0] * (n - k)]
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * n)], at_k)))
    assert verdict.passed is True
    assert _curve_reject(verdict.diagnostics) is False

    over = [[1.5] * (k + 1) + [0.0] * (n - k - 1)]
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * n)], over)))
    assert verdict.passed is False
    assert _curve_reject(verdict.diagnostics) is True
    assert _rate_reject(verdict.diagnostics) is False
    # A 1.5-nat flip counts at every level up to 1.5, so all three fire together.
    assert _fired_levels(verdict.diagnostics) == [0.5, 1.0, 1.5]


def test_curve_multiplier_scales_the_ceilings():
    """curve_multiplier scales each level's ENFORCED ceiling to floor(m * k): a
    count that rejects at the calibrated ceilings passes at 2x, and the scaled
    boundary is again strict-> (2k+1 rejects). Diagnostics carry the multiplier
    and the enforced (scaled) k."""
    cal = _cal()
    n = 2000
    k = cal.thresholds(n)[0]["k"]  # level c=0.5, unscaled
    assert 0 < k < 8
    v = _verifier(cal, curve_multiplier=2.0)

    over_unscaled = [[1.5] * (k + 1) + [0.0] * (n - k - 1)]
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * n)], over_unscaled)))
    assert verdict.passed is True
    assert _curve_reject(verdict.diagnostics) is False
    assert verdict.diagnostics["curve_multiplier"] == 2.0
    assert _enforced_k(verdict.diagnostics["flip_curve"][0]) == 2 * k

    over_scaled = [[1.5] * (2 * k + 1) + [0.0] * (n - 2 * k - 1)]
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * n)], over_scaled)))
    assert verdict.passed is False
    assert _curve_reject(verdict.diagnostics) is True
    assert _rate_reject(verdict.diagnostics) is False
    # A 1.5-nat flip counts at every level up to 1.5, so all three fire together.
    assert _fired_levels(verdict.diagnostics) == [0.5, 1.0, 1.5]


def test_curve_multiplier_scales_every_published_ceiling_consistently():
    """The published row stays self-consistent at any multiplier: the multiplier is
    applied to every entry of ``k_per_engine``, and floor is monotone, so the operative
    ceiling an auditor recomputes as their max is the one that actually decided the
    level."""
    cal = make_calibration(engines={
        "loose": {"rates": [0.004, 0.004, 0.004], "exposure": 1_000_000,
                  "events": [50, 0, 0, 0, 0, 0, 0]},
        "tight": {"rates": [0.004, 0.004, 0.004], "exposure": 1_000_000,
                  "events": [2, 0, 0, 0, 0, 0, 0]}})
    n = 2000
    calibrated = cal.thresholds(n)
    assert calibrated[0]["k_per_engine"]["loose"] > calibrated[0]["k_per_engine"]["tight"]

    v = _verifier(cal, curve_multiplier=2.0)
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * n)], [[0.0] * n])))
    rows = verdict.diagnostics["flip_curve"]
    assert _enforced_k(rows[0]) == 2 * calibrated[0]["k"]
    assert rows[0]["ceiling_per_engine"] == {name: 2 * k for name, k
                                             in calibrated[0]["k_per_engine"].items()}


def test_curve_thresholds_recompute_for_the_actual_trace_length():
    """k(c, N) is computed for THIS trace's pooled length: the same absolute count
    that rejects a short trace is inside the envelope of a long one (more exposure
    means more expected clean events at that level)."""
    cal = _cal()
    v = _verifier(cal)
    short_n, long_n = 2000, 500_000
    k_short = cal.thresholds(short_n)[0]["k"]
    k_long = cal.thresholds(long_n)[0]["k"]
    assert k_long > k_short
    count = k_short + 1

    reject = [[1.5] * count + [0.0] * (short_n - count)]
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * short_n)], reject)))
    assert verdict.passed is False and _curve_reject(verdict.diagnostics) is True

    ok = [[1.5] * count + [0.0] * (long_n - count)]
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * long_n)], ok)))
    assert verdict.passed is True and _curve_reject(verdict.diagnostics) is False


def test_curve_level_diagnostics_report_the_calibrated_parameters():
    """Per level, the diagnostics carry the binding engine's event count and lambda_up,
    the threshold k, the observed count, and whether the level fired -- the exact fields
    an audit of a curve rejection needs.

    The fixture gives the lowest level a much higher clean rate than the rest (50 events
    against 0 over the same exposure) so a count can sit above the stricter levels'
    ceilings and below its own, which is the case worth reading diagnostics for."""
    cal = _cal(levels=(50, 0, 0, 0, 0, 0, 0))
    v = _verifier(cal)
    n = 100_000
    levels = list(cal.curve_side.levels_nats)
    ks = [row["k"] for row in cal.thresholds(n)]
    # ks[1] is the ceiling at 1.0 nats, whose clean rate saw no events; one flip past
    # it fires that level and every stricter one the gap also clears, while the
    # lowest level's much larger ceiling absorbs the same count.
    count = ks[1] + 1
    assert count <= ks[0]
    gaps = [[2.5] * count + [0.0] * (n - count)]
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * n)], gaps)))
    assert verdict.passed is False
    curve = verdict.diagnostics["flip_curve"]
    assert [row["c"] for row in curve] == levels
    # A 2.5-nat gap counts at every level up to 2.5 and at none above it.
    assert [row["count"] for row in curve] == [count] * 5 + [0, 0]
    assert _fired_levels(verdict.diagnostics) == [1.0, 1.5, 2.0, 2.5]
    for row, e in zip(curve, (50, 0, 0, 0, 0, 0, 0)):
        assert row["engine"] == "eng"
        assert row["e"] == e
        assert row["lambda_up"] > 0
        assert row["mu"] == pytest.approx(row["lambda_up"] * n)
    # The Bonferroni share is the same for every rung, so it is published once.
    assert verdict.diagnostics["delta_per_rung"] == cal.delta_per_rung


def test_curve_counts_are_inclusive_at_the_level():
    """gap == c counts at level c (>= is inclusive), mirroring the study's
    N_c = #{flips with gap >= c} definition."""
    cal = _cal(levels=(0, 0, 0, 0, 0, 0, 0))
    v = _verifier(cal)
    n = 2000
    k = cal.thresholds(n)[0]["k"]
    gaps = [[0.5] * (k + 1) + [0.0] * (n - k - 1)]  # exactly at the lowest level
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * n)], gaps)))
    assert verdict.diagnostics["flip_curve"][0]["count"] == k + 1
    assert _curve_reject(verdict.diagnostics) is True


def test_curve_rule_is_always_wired():
    """The curve rule's parameters live in the calibration, so there is no "no
    envelope" mode a task can fall into: every graded run reports a full curve."""
    cal = _cal()
    v = _verifier(cal)
    events = [_event("r0", "B", [1] * 1000)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 1000])))
    assert verdict.passed is True
    assert ([row["c"] for row in verdict.diagnostics["flip_curve"]]
            == list(cal.curve_side.levels_nats))
    assert _curve_reject(verdict.diagnostics) is False


# --- fail-closed guards ----------------------------------------------------------


def test_nan_gaps_fail_closed():
    """NaN loses every strict-> comparison, so without a dedicated guard a broken
    reference (the defect class the gate exists to catch) would score a perfect
    trace. Any NaN gap must reject, with the count in diagnostics."""
    v = _verifier(_cal())
    events = [_event("r0", "B", [1] * 20)]
    gaps = [[float("nan")] * 10 + [0.0] * 10]
    verdict = asyncio.run(v.verify(_acc(events, gaps)))
    assert verdict.passed is False
    assert (verdict.diagnostics["n_nan_gaps"] > 0) is True
    assert verdict.diagnostics["n_nan_gaps"] == 10
    # NaN must not leak into max_gap either (max() would drop it order-dependently).
    assert verdict.diagnostics["max_gap"] == 0.0


def test_inf_gap_fails_closed():
    """+inf is the reference's verdict for a served id outside its vocabulary (or
    with zero probability mass) -- malformed evidence like NaN, not a bf16 noise
    event. The curve counts it like any flip, so a dedicated guard fails it
    closed."""
    v = _verifier(_cal())
    events = [_event("r0", "B", [1] * 20)]
    verdict = asyncio.run(v.verify(_acc(events, [[float("inf")] + [0.0] * 19])))
    assert verdict.passed is False
    assert (verdict.diagnostics["n_inf_gaps"] > 0) is True
    assert verdict.diagnostics["n_inf_gaps"] == 1
    assert verdict.diagnostics["max_gap"] == float("inf")


def test_empty_reply_events_fail_closed():
    """An event whose reply carries ZERO teacher-forced tokens is unverifiable (and
    greedy decode always emits at least one token, so a faithful server never
    produces one). Without this guard a server could answer most requests with
    empty token lists, inflate per-request throughput, and stay invisible to the
    rules -- verified: the non-empty event here is perfectly faithful, so the empty one
    is the only thing that can reject the trace."""
    v = _verifier(_cal())
    events = [_event("r0", "", []), _event("r1", "B", [1] * 50)]
    verdict = asyncio.run(v.verify(_acc(events, [[], [0.0] * 50])))
    assert verdict.passed is False
    assert (verdict.diagnostics["n_empty_events"] > 0) is True
    assert verdict.diagnostics["n_empty_events"] == 1
    assert verdict.event_diagnostics["t0:r0"]["n_tokens"] == 0


def test_the_floor_is_derived_from_q_max_and_the_deployed_headroom():
    """The floor is not a knob a task can shrink: it is where the rate rule first meets
    its false-alarm budget, which depends on q_max AND on the headroom the verifier
    deploys -- a tighter headroom leaves less room for Poisson noise, so it needs a
    longer trace."""
    from hls.verify.poisson import rate_rule_min_tokens

    cal = _cal()
    tight = _verifier(cal, rate_multiplier=1.5)
    loose = _verifier(cal, rate_multiplier=3.0)
    assert tight.min_tokens > loose.min_tokens
    assert tight.min_tokens == rate_rule_min_tokens(
        cal.q_max, cal.delta_per_rung, 1.5)


def test_a_trace_under_the_floor_fails_closed():
    v = TokenFaithfulnessVerifier(reference=_ref(), calibration=_cal())
    events = [_event("r0", "B", [1] * 20)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 20])))
    assert verdict.passed is False
    assert _insufficient(verdict.diagnostics) is True
    assert _underpowered(verdict.diagnostics) is False


def test_allow_underpowered_trace_grades_it_and_says_so():
    """A task whose traffic cannot reach the floor declares it once; the verdict then
    carries the weaker claim explicitly rather than looking like a full one."""
    v = _verifier(_cal())          # the fixture sets allow_underpowered_trace
    events = [_event("r0", "B", [1] * 20)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 20])))
    assert verdict.passed is True
    assert _insufficient(verdict.diagnostics) is True
    assert _underpowered(verdict.diagnostics) is True


def test_insufficient_trace_fails_below_the_floor():
    """Fewer than min_tokens pooled tokens is an automatic fail even with zero flips
    (below the floor the rate rule has no false-alarm guarantee)."""
    v = TokenFaithfulnessVerifier(reference=_ref(), calibration=_cal())
    events = [_event("r0", "B", [1] * 20)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 20])))
    assert verdict.passed is False
    assert _insufficient(verdict.diagnostics) is True
    assert _rate_reject(verdict.diagnostics) is False
    assert _curve_reject(verdict.diagnostics) is False


def test_empty_trace_fails_closed_insufficient():
    """Zero events (an empty trace) is fail-closed exactly like a below-floor trace:
    n_tokens 0 -> insufficient_trace, no per-event verdicts, and the always-present
    rate_ratio is still emitted (0.0) rather than crashing on the empty pool."""
    v = TokenFaithfulnessVerifier(reference=_ref(), calibration=_cal())
    verdict = asyncio.run(v.verify(_acc([], [])))
    assert verdict.passed is False
    assert verdict.per_event == {}
    assert verdict.diagnostics["n_tokens"] == 0
    assert _insufficient(verdict.diagnostics) is True
    assert _rate_reject(verdict.diagnostics) is False
    assert verdict.diagnostics["rate_ratio"] == 0.0


# --- emitted-token consistency (check 3, off by default) -------------------------


def test_emit_consistency_off_by_default_ignores_violations():
    """The flag defaults OFF (most contracts do not request logprobs), so even a
    report that contradicts the greedy claim changes nothing -- and the verdict
    says the check was not run."""
    v = _verifier(_cal())
    lps = [_lp(-2.0, [-0.5, -2.0])]  # emitted 1.5 nats below the reported max
    events = [_event("r0", "B", [1] * 20, logprobs=lps)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 20])))
    assert verdict.passed is True
    assert verdict.diagnostics["emit_consistency_checked"] is False
    assert "n_emit_violations" not in verdict.diagnostics


def test_emit_consistency_fires_on_a_reported_violation():
    """A greedy server's emitted token IS its reported argmax; a position whose
    emitted logprob sits below the reported maximum is the contestant's own report
    contradicting its greedy claim."""
    v = _verifier(_cal(), check_emit_consistency=True)
    lps = [_lp(-0.5, [-0.5, -1.0]),          # clean: emitted == reported max
           _lp(-2.0, [-0.5, -2.0])]          # violation: -2.0 < -0.5
    events = [_event("r0", "B", [1, 2], logprobs=lps)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0, 0.0]])))
    assert verdict.passed is False
    # The emit check rejects ON ITS OWN: the trace is under the floor, but this
    # verifier declares that (allow_underpowered_trace), so nothing else fires.
    assert (verdict.diagnostics["n_emit_violations"] + verdict.diagnostics["n_emit_unverifiable"] > 0) is True
    assert verdict.diagnostics["n_emit_violations"] == 1
    assert verdict.diagnostics["emit_violations"] == [
        {"event": "t0:r0", "index": 1, "logprob": -2.0, "reported_max": -0.5}]


def test_emit_consistency_reports_at_most_the_first_five_positions():
    v = _verifier(_cal(), check_emit_consistency=True)
    lps = [_lp(-2.0, [-0.5]) for _ in range(8)]  # 8 violating positions
    events = [_event("r0", "B", [1] * 8, logprobs=lps)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 8])))
    assert verdict.diagnostics["n_emit_violations"] == 8
    assert [x["index"] for x in verdict.diagnostics["emit_violations"]] == [0, 1, 2, 3, 4]


def test_emit_consistency_tolerates_exact_ties():
    """Exact ties are allowed (bf16 top-1 ties are real and benign), with a 1e-9
    tolerance absorbing serializer round-off of one float."""
    v = _verifier(_cal(), check_emit_consistency=True)
    lps = [_lp(-0.5, [-0.5, -0.5]),                # exact tie
           _lp(-0.5 - 5e-10, [-0.5])]              # inside the tolerance
    events = [_event("r0", "B", [1, 2], logprobs=lps)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0, 0.0]])))
    assert verdict.passed is True
    assert verdict.diagnostics["n_emit_violations"] == 0
    assert (verdict.diagnostics["n_emit_violations"] + verdict.diagnostics["n_emit_unverifiable"] > 0) is False


def test_emit_consistency_fails_closed_on_missing_reports():
    """Under the flag, a position (or whole event) without the reported logprob
    data is unverifiable and rejects -- otherwise a violator could simply omit the
    report for its non-greedy positions."""
    v = _verifier(_cal(), check_emit_consistency=True)
    no_report = _event("r0", "B", [1, 2])  # event with no logprobs at all
    no_tops = _event("r1", "C", [3], logprobs=[
        TokenLogprob(token="t", token_id=3, logprob=-0.5, top_logprobs=None)])
    verdict = asyncio.run(v.verify(_acc([no_report, no_tops], [[0.0, 0.0], [0.0]])))
    assert verdict.passed is False
    assert (verdict.diagnostics["n_emit_violations"] + verdict.diagnostics["n_emit_unverifiable"] > 0) is True
    assert verdict.diagnostics["n_emit_unverifiable"] == 2
    assert verdict.diagnostics["n_emit_violations"] == 0


# --- refusals + wiring ------------------------------------------------------------


def test_serving_on_wrong_silicon_refuses():
    """read_calibration's gpu_kind check compares declarations; the verifier checks
    the SILICON actually serving (substrates do not all enforce the declared
    accelerator) -- a reference on a different GPU kind than was calibrated must
    refuse, since a calibration holds for the GPU kind it was taken on."""
    v = _verifier(_cal())          # cal.gpu_kind == "B200"
    events = [_event("r0", "B", [1] * 20)]
    wrong = {**_STACK, "device_name": "NVIDIA H100"}
    with pytest.raises(CalibrationError, match="(?i)gpu-kind|gpu_kind"):
        asyncio.run(v.verify(_acc(events, [[0.0] * 20], stack=wrong)))


def test_stack_drift_ignores_device_name():
    """device_name is descriptive (GPU identity is gpu_kind's gate at load time, and
    marketing-name variants of one kind must not refuse a run): a differing
    device_name alone is NOT drift, while any version difference still is."""
    v = _verifier(_cal())
    events = [_event("r0", "B", [1] * 20)]
    renamed = {**_STACK, "device_name": "NVIDIA B200 NVL"}
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 20], stack=renamed)))
    assert verdict.passed is True


def test_calibration_none_raises_wiring_error():
    """The loader must inject calibration.json before verify; a None calibration is a
    wiring fault, not a silent pass."""
    v = TokenFaithfulnessVerifier(reference=_ref(), calibration=None)
    events = [_event("r0", "B", [1] * 20)]
    with pytest.raises(WiringError, match="(?i)calibration"):
        asyncio.run(v.verify(_acc(events, [[0.0] * 20])))


def test_stack_drift_raises_calibration_error():
    """A reference whose stack versions drifted from the calibrated ones must be
    refused (the calibration's noise floor does not apply)."""
    v = _verifier(_cal())
    drifted = {**_STACK, "torch": "2.5.0"}
    events = [_event("r0", "B", [1] * 20)]
    with pytest.raises(CalibrationError, match="(?i)drift"):
        asyncio.run(v.verify(_acc(events, [[0.0] * 20], stack=drifted)))


def test_missing_tokens_does_not_raise():
    """An event without emitted token ids must not raise, which would take the whole rollout
    down and leave no verdict to publish. It is the artifact's contract violation, so it
    fails the trace instead; the rejection itself is covered below."""
    v = _verifier(_cal())
    events = [_event("r0", "B", None)]
    verdict = asyncio.run(v.verify(_acc(events, [])))
    assert verdict.passed is False


def test_a_conversation_messages_of_empties_is_refused_not_rebuilt():
    """Every entry of the request's message list dropped (none carried a role and a
    content) is a conversation nothing can teacher-force: the wire sent an empty one,
    and the reference renders an empty prompt for it. Rebuilding a turn from the text
    payload here would score the reply against a prompt no server ever saw."""
    v = _verifier(_cal())
    ev = _event("r0", "B", [1] * 20)
    ev = ev.model_copy(update={"input": Envelope(
        request_id="r0", contents=[TextPayload(text="Q?")],
        metadata={**ev.input.metadata, "messages": [{"content": "no role"}]})})
    with pytest.raises(ValueError, match="(?i)messages"):
        asyncio.run(v.verify(_acc([ev], [[0.0] * 20])))


def test_rate_multiplier_zero_rejected_at_construction():
    """A 0 rate_multiplier would ZeroDivide the rate_ratio diagnostic; reject it
    at construction rather than at verify() time."""
    with pytest.raises(pydantic.ValidationError):
        TokenFaithfulnessVerifier(reference=_ref(), rate_multiplier=0.0)


# --- ReferenceRelativeVerifier -------------------------------------------------


class _GenProxy:
    def __init__(self, outs):
        self._outs = outs

    async def generate(self, prompts):
        return self._outs


def test_reference_relative_compares_art_vs_ref():
    v = ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_ref(), tol=0.0)
    events = [_event("r0", "B", None, gt="B"), _event("r1", "A", None, gt="B")]  # art: 1 right
    ref_outs = [
        Envelope(request_id="r0", contents=[TextPayload(text="B")]),
        Envelope(request_id="r1", contents=[TextPayload(text="B")]),
    ]  # ref: 2 right
    acc = EngineAccessor(
        artifact_events=events, cap_to_provider={Generate: _GenProxy(ref_outs).generate}
    )
    verdict = asyncio.run(v.verify(acc))
    assert verdict.passed is False  # art 0.5 < ref 1.0


def test_a_reply_at_the_output_cap_is_still_graded():
    """A reply whose served length reached the request's cap is measured like any
    other, and its flips reject the trace. The cap is the harness's stamp on the
    request, so the count of such replies is a fact about the trace and is reported;
    nothing about the reply's own label enters."""
    v = _verifier(_cal())
    clean = _event("r0", "B", [1] * 1000)
    capped = _event("r1", "C", [1] * 1000, max_tokens=1000)
    gaps = [[0.0] * 1000, [0.5] * 1000]  # every token of the capped reply is a flip
    verdict = asyncio.run(v.verify(_acc([clean, capped], gaps)))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert _rate_reject(d) is True
    assert d["n_tokens"] == 2000 and d["n_flips"] == 1000
    assert d["flip_rate"] == 0.5
    assert d["n_capped_events"] == 1 and d["n_capped_tokens"] == 1000
    assert verdict.event_diagnostics["t0:r1"]["at_cap"] is True
    assert verdict.event_diagnostics["t0:r0"]["at_cap"] is False


def test_the_reported_finish_reason_is_never_read():
    """The finish reason is a value the artifact chooses and nothing checks it, so it
    must not move anything: two traces identical but for the label produce identical
    verdicts, and a three-token reply labelled "length" is neither exempt nor counted
    as capped."""
    v = _verifier(_cal())
    gaps = [[0.0] * 300, [0.4, 0.4, 0.4]]
    labelled = [_event("r0", "B", [1] * 300, finish_reason="stop"),
                _event("r1", "C", [1, 2, 3], finish_reason="length")]
    unlabelled = [_event("r0", "B", [1] * 300), _event("r1", "C", [1, 2, 3])]
    a = asyncio.run(v.verify(_acc(labelled, gaps)))
    b = asyncio.run(v.verify(_acc(unlabelled, gaps)))
    assert a.diagnostics == b.diagnostics
    assert a.event_diagnostics == b.event_diagnostics
    d = a.diagnostics
    assert d["n_tokens"] == 303 and d["n_flips"] == 3
    assert d["n_capped_events"] == 0 and d["n_capped_tokens"] == 0
    # 3/303 = 0.99% over the 0.8% ceiling: the three tokens are what reject it.
    assert a.passed is False and _rate_reject(d) is True


def test_a_reply_padded_with_one_repeated_token_fails():
    """The cheapest way to fake a capped reply is to repeat one token id: it needs no
    real decoding. Teacher-forced against the reference it is a flip at nearly every
    position, so it fails both rules."""
    v = _verifier(_cal())
    pad = _event("r0", "!" * 2000, [999] * 2000, max_tokens=2000)
    verdict = asyncio.run(v.verify(_acc([pad], [[3.0] * 2000])))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert _rate_reject(d) is True and _curve_reject(d) is True
    assert d["n_tokens"] == 2000 and d["n_flips"] == 2000


def test_a_reply_served_past_its_cap_fails_the_trace():
    """A reply longer than the request's stamped max_tokens ignored the cap. The
    trace fails on that fact alone, with the count published, however clean the
    tokens are; calibration refuses a pool member on the same condition."""
    v = _verifier(_cal())
    over = _event("r0", "B", [1] * 1001, max_tokens=1000)
    verdict = asyncio.run(v.verify(_acc([over], [[0.0] * 1001])))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["n_over_cap_events"] == 1
    assert d["n_capped_events"] == 0
    assert _rate_reject(d) is False and _fired_levels(d) == []


def test_a_passing_verdict_names_no_rejections():
    """``rejections`` is empty on a pass, so a reader can tell "nothing fired" from
    "something fired and was not recorded"."""
    v = _verifier(_cal())
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * 300)], [[0.0] * 300])))
    assert verdict.passed is True
    assert verdict.diagnostics["rejections"] == []


def test_rejections_name_every_rule_that_fired():
    """"correctness" covers every gate in this verifier, so the verdict names which
    rules rejected rather than leaving the reader to re-derive them from the numbers."""
    v = _verifier(_cal())
    # One reply over its stamped cap, which is never teacher-forced, and one carrying a flip
    # at every position: two distinct rules.
    over = _event("r0", "B", [1] * 1001, max_tokens=1000)
    flipped = _event("r1", "C", [1] * 300)
    verdict = asyncio.run(v.verify(_acc([over, flipped], [[3.0] * 300])))
    assert verdict.passed is False
    assert set(verdict.diagnostics["rejections"]) >= {"over_cap_replies", "flip_rate"}


def test_a_reply_without_token_ids_fails_the_trace():
    """token_ids is mandatory on the wire contract and a reply without them cannot be
    teacher-forced. That is the artifact's fault, so it fails the trace with the count
    published rather than raising and taking the whole rollout down with it."""
    v = _verifier(_cal())
    events = [_event("r0", "B", [1] * 300), _event("r1", "C", None)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 300])))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["n_untokenized_events"] == 1
    assert d["untokenized_events"] == ["t0:r1"]
    # The tokenized reply is still measured, so the rejection rests on the contract
    # violation and not on an emptied trace.
    assert d["n_tokens"] == 300 and _rate_reject(d) is False


def test_a_trace_with_no_token_ids_anywhere_fails_without_raising():
    """Every reply omitting token_ids leaves nothing to force. The verdict is a
    rejection on no evidence, not an exception."""
    v = _verifier(_cal())
    events = [_event("r0", "B", None), _event("r1", "C", None)]
    verdict = asyncio.run(v.verify(_acc(events, [])))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["n_untokenized_events"] == 2 and d["n_tokens"] == 0


def test_an_all_capped_trace_is_graded_rather_than_emptied():
    """A trace whose every reply reached the output cap is simply graded: the tokens
    are there, so the rules decide on evidence rather than on the absence of it."""
    v = _verifier(_cal())  # the fixture sets allow_underpowered_trace
    events = [_event("r0", "B", [1] * 1000, max_tokens=1000),
              _event("r1", "C", [1] * 1000, max_tokens=1000)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 1000, [0.0] * 1000])))
    d = verdict.diagnostics
    assert verdict.passed is True
    assert (d["n_tokens"] == 0) is False
    assert d["n_tokens"] == 2000 and d["n_capped_events"] == 2


def test_an_event_without_a_stamped_cap_is_refused():
    """The cap decides whether a reply's stop is scored and whether it counts as capped,
    so an event that does not carry the harness's stamp raises instead of being
    graded under a guess in either direction."""
    v = _verifier(_cal())
    out = Envelope(request_id="r0", contents=[TextPayload(text="B", tokens=[1] * 20)])
    inp = Envelope(request_id="r0", contents=[TextPayload(text="Q?")],
                   metadata={"messages": [{"role": "user", "content": "Q?"}],
                             "tenant_id": "t0"})
    ev = TraceEvent(request_id="r0", input=inp, response=_resp(out), t_send=0.0, t_recv=1.0)
    with pytest.raises(ValueError, match="max_tokens"):
        asyncio.run(v.verify(_acc([ev], [[0.0] * 20])))


def test_per_event_records_carry_the_shared_flips_schema():
    """Every event's diagnostics are the per-event record calibration persists for its
    own servings (hls.verify.flips.ReplyFlips): the same fields, so a graded trace and a
    stored serving can be re-read by one reader."""
    from hls.verify.flips import ReplyFlips

    v = _verifier(_cal())
    ev = _event("r0", "B", [1] * 4, max_tokens=4)
    verdict = asyncio.run(v.verify(_acc([ev], [[0.0, 0.6, 2.2, 0.0]])))
    rec = verdict.event_diagnostics["t0:r0"]
    assert rec == {"event_key": "t0:r0", "n_tokens": 4, "n_flips": 2, "max_gap": 2.2,
                   "severe_gaps": [0.6, 2.2], "n_nan": 0, "n_inf": 0, "stop_gap": None,
                   "at_cap": True}
    assert ReplyFlips.from_json(rec).to_json() == rec


def test_the_shipped_curve_multiplier_is_two():
    """The curve rule's default headroom matches the rate rule's: both leave a clean
    stack the same declared room over its own envelope."""
    v = TokenFaithfulnessVerifier(reference=_ref(), calibration=_cal())
    assert v.curve_multiplier == 2.0 and v.rate_multiplier == 2.0


def test_a_graded_trace_with_tokens_does_not_trip_the_zero_token_guard():
    """The guard is scoped to a genuinely empty measurement, so an ordinary short
    trace still grades normally under allow_underpowered_trace."""
    v = _verifier(_cal())
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", [1] * 20)], [[0.0] * 20])))
    assert verdict.passed is True
    assert (verdict.diagnostics["n_tokens"] == 0) is False


def test_the_teacher_force_item_carries_the_turns_tool_definitions():
    """An agentic turn offers tools, and the chat template folds them into the prompt.
    The item the verifier builds must carry them, or the reference would score the
    reply against a context the artifact was never sent and every position would flip."""
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    ev = _event("r0", "hi", [1, 2])
    ev = ev.model_copy(update={"input": ev.input.model_copy(
        update={"metadata": {**ev.input.metadata, "tools": tools}})})
    v = TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))

    item = v._item(ev, v._out_of(ev))

    assert item.metadata["tools"] == tools
    assert item.metadata["messages"] == [{"role": "user", "content": "Q?"}]


def test_a_turn_with_no_tools_leaves_the_key_off_the_item():
    """A turn that offers none must render exactly as a task that never sets the key,
    so the key is absent rather than an empty list."""
    v = TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))
    ev = _event("r0", "hi", [1, 2])

    assert "tools" not in v._item(ev, v._out_of(ev)).metadata

    ev_empty = ev.model_copy(update={"input": ev.input.model_copy(
        update={"metadata": {**ev.input.metadata, "tools": []}})})
    assert "tools" not in v._item(ev_empty, v._out_of(ev_empty)).metadata
