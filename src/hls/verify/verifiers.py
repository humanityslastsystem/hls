"""Provided correctness verifiers.

A ``Verifier`` owns its reference engine config(s) and gates on a ``RolloutMetric``'s
reduced value (``MetricVerifier``), compares it to a reference rollout
(``ReferenceRelativeVerifier``), or runs the calibrated token-level faithfulness gate
(``TokenFaithfulnessVerifier``). New benchmarks add metrics (``hls.verify.metrics``),
not verifiers -- ``MetricVerifier``/``ReferenceRelativeVerifier`` are metric-generic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import ClassVar

from pydantic import Field

from hls.core.capabilities import Generate, ReferenceLogprobs, StackReport, TeacherForce
from hls.core.payloads import Envelope, TextPayload
from hls.core.verdict import PerEventVerdict, Verdict
from hls.reference.engines import continues_of
from hls.reference.references import Reference
from hls.spec.drivers import messages_of
from hls.spec.workload import CUT_EXTRA
from hls.verify.base import (
    ArtifactRollout,
    EngineAccessor,
    RolloutMetric,
    Verifier,
    WiringError,
    as_rollout,
    verdict_from,
)
from hls.verify.calibration import (
    Calibration,
    CalibrationError,
    ReportCalibration,
    device_matches_kind,
)
from hls.verify.flips import PooledFlips, past_cap, score_reply
from hls.verify.poisson import rate_rule_min_tokens
from hls.verify.report import (
    TOP_LOGPROBS_REQUESTED,
    PooledReport,
    ReplyReport,
    ReportRows,
    position_stats,
    reply_report,
    reply_uniforms,
)
# Each gate reduces a reply through its own module's reduction, and the two are the
# same operation on different measurements, so they carry the same name there.
from hls.verify.report import score_reply as score_report_reply


#: Items one reference call carries. Every scored number is per position, so neither
#: this nor the position bound below changes any of them.
_REPORT_BATCH: int = 256

#: Positions one reference call carries. The item count alone does not bound the
#: memory a call holds: a reply's cost is its length, and 256 replies at an 8,192-token
#: cap return 42 million floats, which is over a gigabyte of lists held at once.
#: This cap holds about 0.7 GB live -- the partitions sent plus the
#: log-probabilities returned come to ~2.3 KB per position -- whatever the replies'
#: lengths happen to be.
_REPORT_POSITIONS: int = 300_000


def report_call_is_full(n_items: int, n_positions: int) -> bool:
    """Whether a reference call already holding this many items and positions should be
    sent before another reply joins it.

    Both sides of the gate batch the same way -- the deployed verifier over a graded
    trace, ``hls.calibrate`` over a captured serving -- so the bounds live here once and
    both ask this. Batching changes no scored number: every statistic is per position
    and the reference scores each item independently."""
    return n_items >= _REPORT_BATCH or n_positions >= _REPORT_POSITIONS


def _text_of(env: Envelope | None) -> str | None:
    """The full text of an Envelope's first text payload (the oracle's raw answer)."""
    if env is None:
        return None
    tp = env.first_of("text")
    return tp.text if tp is not None else None


def _was_cut(ev) -> bool:
    """Whether the drive's scheduled stop cut this request mid-stream
    (``hls.spec.workload.CUT_EXTRA``).

    A cut reply's tokens are the server's, so they are graded like any others; its
    ending is the harness's, so the rules that read an ending do not apply to it."""
    return bool(ev.extras.get(CUT_EXTRA))


def _served_tokens(ev) -> list[int]:
    """The ids a reply emitted, or an empty list where it emitted none or carried no
    response at all."""
    tp = ev.response.output.first_of("text") if ev.response is not None else None
    return list(tp.tokens) if tp is not None and tp.tokens else []


def _reference_meta(ev, verifier: str) -> dict:
    """The context a reference needs to reproduce one event's positions: the
    conversation the request carried, the output cap the harness stamped on it, and
    whether the request resumed its final message.

    The cap decides whether a reply reached its budget, which both gates read, so an
    event without one is refused rather than guessed at.

    Tool definitions ride along when the request carried them (an agentic task offers a
    different tool set on different turns). A chat template folds them into the prompt,
    so an item that dropped them would be scored against a context the artifact was
    never sent, and every position would measure the harness instead of the artifact.

    The conversation comes through ``messages_of``, the same selection the wire protocol
    encodes from and the reference renders its context with. Reading the raw list here
    would let an entry those two drop still reach the teacher-force, so the tokens being
    graded and the tokens the reference scores them against could come from different
    messages. An all-dropped list is refused rather than rebuilt from the text payload:
    the wire sent an empty conversation, and scoring the reply against a prompt no
    server saw would be worse than failing."""
    msgs = messages_of(ev.input)
    if not msgs:
        raise ValueError(
            f"{verifier} found no usable message on event {ev.event_key!r}: every entry "
            f"of metadata['messages'] lacked a role or a content, so the wire sent an "
            f"empty conversation")
    cap = ev.input.metadata.get("max_tokens")
    if not isinstance(cap, int) or cap <= 0:
        raise ValueError(
            f"{verifier} needs the request's stamped output cap "
            f"(metadata['max_tokens']) on event {ev.event_key!r}; got {cap!r}. The "
            f"drive loop stamps it on every request (hls.spec.drivers.stamp_request).")
    meta: dict = {"messages": msgs, "max_tokens": cap}
    tools = ev.input.metadata.get("tools")
    if isinstance(tools, list) and tools:
        meta["tools"] = tools
    # A request that resumed its final assistant message was rendered with no generation
    # prompt after it, so a reference that dropped the flag would score every position
    # under two tokens the server was never sent. Read through the engines' own reader,
    # which refuses a continuation whose conversation does not end on an assistant message.
    if continues_of(ev.input):
        meta["continue_final_message"] = True
    return meta


def _normal_two_sided_critical(alpha: float) -> float:
    """The two-sided critical value ``z_{alpha/2}`` of the standard normal.

    From the standard library's own normal quantile rather than a table or a fitted
    approximation, so the deviate's threshold is exact at whatever ``alpha`` a task
    declares."""
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


class MetricVerifier(Verifier):
    """Score the artifact rollout with `metric`; pass when the metric's reduced value
    meets `min_value`. min_value subsumes pass-rate (a bool metric reduces to a pass-rate)
    and mean-F1 (CoQA). No engine."""

    metric: RolloutMetric
    min_value: float = 1.0

    async def verify(self, engine: EngineAccessor) -> Verdict:
        r = self.metric.score(engine[ArtifactRollout])
        return verdict_from(
            r, passed=r.value >= self.min_value, diagnostics={"value": r.value, "min_value": self.min_value}
        )


class ReferenceRelativeVerifier(Verifier):
    """Score the artifact AND a reference rollout with the SAME metric (both against
    the dataset ground truth); pass when the artifact's reduced value clears the
    reference-derived bar.

    Two bar shapes, mutually exclusive:
      - additive (default): ``artifact + tol >= reference`` -- absolute slack in
        metric units, independent of where the reference lands.
      - multiplicative: set ``min_ratio`` and the gate is
        ``artifact >= reference * min_ratio`` -- the artifact must retain at least
        that fraction of the reference's quality, so the slack scales with the
        reference (e.g. ``min_ratio=0.95`` = keep 95% of the oracle's accuracy)."""

    metric: RolloutMetric
    reference: Reference
    tol: float = 0.0
    #: Multiplicative bar: artifact >= reference * min_ratio. Exclusive with ``tol``.
    min_ratio: float | None = None
    requires = frozenset({Generate})

    def model_post_init(self, __context) -> None:
        if self.min_ratio is not None and self.tol != 0.0:
            raise ValueError(
                "ReferenceRelativeVerifier: set either tol (additive) or min_ratio "
                "(multiplicative), not both")
        if self.min_ratio is not None and self.min_ratio <= 0.0:
            raise ValueError(
                f"min_ratio must be > 0 (a 0 bar passes any artifact), got {self.min_ratio}")

    async def verify(self, engine: EngineAccessor) -> Verdict:
        events = engine[ArtifactRollout]
        ref_outs = await engine[Generate]([e.input for e in events])
        art = self.metric.score(events)
        ref = self.metric.score([as_rollout(e, ro) for e, ro in zip(events, ref_outs)])
        # Re-surface the reference's per-event detail (already computed above) nested under
        # a `reference` key, alongside the oracle's raw text, so the viewer can render the
        # per-turn artifact-vs-oracle comparison. Metric-agnostic: `reference` carries
        # whatever the metric emitted for the oracle rollout. The gate is unchanged -- it
        # still compares only the reduced .value.
        answered = {ev.request_id: ro for ev, ro in zip(events, ref_outs)}
        enriched: dict[str, PerEventVerdict] = {}
        for ev in events:
            rid = ev.request_id
            a_pv = art.per_event[rid]
            r_pv = ref.per_event.get(rid)
            ref_diag = {**(r_pv.diagnostics if r_pv is not None else {}),
                        "text": _text_of(answered[rid])}
            enriched[rid] = a_pv.model_copy(
                update={"diagnostics": {**a_pv.diagnostics, "reference": ref_diag}}
            )
        merged = art.model_copy(update={"per_event": enriched})

        if self.min_ratio is not None:
            passed = art.value >= ref.value * self.min_ratio
            bar = {"min_ratio": self.min_ratio}
        else:
            passed = art.value + self.tol >= ref.value
            bar = {"tol": self.tol}
        return verdict_from(
            merged,
            passed=passed,
            diagnostics={"artifact_value": art.value,
                         "reference_value": ref.value,
                         "reference_n": len(events), **bar},
        )


#: Exact-tie tolerance for the emitted-token consistency check: an emitted token's
#: reported log-probability may sit below the reported per-position maximum by at
#: most this much before it counts as a violation. Sized to absorb serializer
#: round-off of one float (the two numbers come from the same reported
#: distribution, so any real gap between them is a claim of non-greedy sampling).
_EMIT_TIE_TOL = 1e-9

#: Events a verdict names of those a rule rejected, before it only counts them: enough for a
#: reader to know what to fix, and bounded whatever a server makes every reply do.
SHOWN_EVENTS = 5


@dataclass(frozen=True)
class _Refusal:
    """What the report gate's record of a refused reply reads of it
    (``ReportFaithfulnessVerifier._violated``), kept in the reply's place."""

    #: The reply's event key.
    event_key: str
    #: The tokens it served.
    n_served: int
    #: The last token id it served; None when it served none.
    last: int | None
    #: The output cap its request carried (``metadata['max_tokens']``), as found.
    cap: object
    #: The finish reason it reported.
    finish_reason: str | None



class TokenFaithfulnessVerifier(Verifier):
    """Token-level faithfulness: the served tokens may not diverge from the pinned
    reference more than the calibrated trusted-engine envelope allows.

    Three deterministic checks, no runtime statistics:

      Check 1 (rate):  flip_rate > rate_multiplier * calibration.q_max      -> fail
      Check 2 (curve): for each gap level c of the calibration's curve side,
                       #{flips with gap >= c} > floor(curve_multiplier * k(c, N)) -> fail
                       where N is the pooled trace length and
                       k(c, N) = min{ k : P(Poisson(lambda_up_c * N) > k)
                                         <= delta_per_rung }
                       from the engine-measured clean rate at that level
                       (lambda_up_c, the Garwood upper bound on the event rate
                       per teacher-forced token), graded against the more
                       permissive engine at each level (see
                       hls.verify.calibration.CurveSide). The curve bounds how
                       many large-gap flips a trace carries, not how large one
                       gap is: the clean pool itself produces rare multi-nat
                       events.
      Check 3 (emit consistency, ``check_emit_consistency``, default off): any
                       emitted token whose serve-time reported log-probability is
                       below the contestant's own reported maximum at that
                       position (exact ties allowed, tolerance ``_EMIT_TIE_TOL``)
                       -> fail. A greedy server's emitted token is its reported
                       argmax, so a violation is the contestant's own report
                       contradicting its greedy claim. Off by default: it needs
                       the contract to request per-token logprobs with top-k
                       alternatives (``OpenAIChatProtocol(return_logprobs=
                       True, top_logprobs=k)``); ``check_wiring`` refuses the
                       flag on a contract that does not, and under the flag any
                       position missing that data fails closed.

    Every served reply is teacher-forced and counted into both rules, whatever the
    artifact says about it; the artifact's ``finish_reason`` is never read. How many
    replies reached the output cap, by served length against the harness-stamped cap,
    is reported as ``n_capped_events``/``n_capped_tokens`` and changes nothing.

    The stop row is the position after the last served token, scored as how much the
    reference preferred continuing over ending the turn. The reference scores it
    unless the reply ended on an end-of-turn id or its served length reached the
    request's ``max_tokens`` (``HFEngine.teacher_force``); a reply that ran to the cap
    is graded as text with no stop row. The cap is the harness's stamp on the request
    and must be present on every event; a missing cap raises rather than exempting
    anything, and a reply served past its cap (``past_cap``) fails the trace
    (``n_over_cap_events``), as this gate's calibration refuses the pool member that
    served one (``hls.calibrate.refuse_past_cap``). Such a reply is the one that is not
    teacher-forced: the trace fails on it whatever its tokens are, so it is counted, the
    first ``SHOWN_EVENTS`` named in ``over_cap_events``, and left out of the measurement
    before the reference sees it, and the reference never holds a reply whose length
    the server chose past the cap.

    Non-terminating generations are a real property of some checkpoints, and a task
    that has them handles them where they can be identified without trusting the
    artifact: the rows its calibration's own serving stack ran to the cap are removed
    from the workload before it is served (``DataSource.removed_rows``, derived from
    the campaign's captures). The list is keyed by row, measured under trusted
    serving, and identical on both sides of the gate because neither side's server
    chooses it.

    plus fail-closed guards:
      - fewer pooled teacher-forced tokens than ``min_tokens`` is an automatic
        fail (``underpowered_trace`` among the verdict's ``rejections``) -- below
        that floor the rate rule has no false-alarm guarantee at the deployed
        headroom. A task whose traffic cannot reach it declares
        ``allow_underpowered_trace``, which grades the trace anyway; the verdict
        publishes the flag beside ``n_tokens`` and ``min_tokens``;
      - zero pooled teacher-forced tokens is an automatic fail that
        ``allow_underpowered_trace`` cannot waive: on an empty trace every rule
        above is vacuous (a 0/0 flip rate, all-zero level counts), so a pass
        would rest on no evidence at all rather than on thin evidence. A run
        that served nothing, or served only empty replies, lands here;
      - any event whose reply carries zero teacher-forced tokens is an automatic
        fail: an empty reply is unverifiable (and greedy decode always emits at
        least one token, so a faithful server never produces one) -- without this,
        a server could answer most requests with empty token lists, inflate its
        per-request throughput, and stay invisible to the rules;
      - any event whose reply carries no token ids at all is an automatic fail:
        token_ids is mandatory on the wire contract, and a reply without them
        cannot be teacher-forced, so omitting them would otherwise exempt a server
        from the gate entirely;
      - any NaN gap is an automatic fail: NaN loses every strict-``>`` comparison,
        so without the guard a numerically broken reference (the defect class the
        gate exists to catch) would score a perfect trace;
      - any +inf gap is an automatic fail: the reference emits +inf for a served
        id outside its vocabulary or with zero probability mass -- malformed
        evidence like NaN, not a bf16 noise event. The curve rule counts it like
        any other flip, so the guard is what keeps it fail-closed.
    A flip is gap > 0 exactly (a served token equal to the reference argmax
    subtracts a tensor element from itself: bitwise 0.0).

    ``calibration`` is injected by the CLI loader from the task folder's
    calibration.json (see hls.verify.calibration); task authors never set it. It
    carries both rules' parameters; the trace-length floor is computed from them and
    ``rate_multiplier`` (``min_tokens``), so the deployed gate has no numbers of its
    own beyond the two tolerance multipliers below.
    The verifier also refuses to run against a reference whose stack versions
    drifted from the calibrated ones (StackReport capability).
    """

    reference: Reference
    calibration: Calibration | None = None  # injected by the loader, never authored
    # gt=0: the multiplier divides the rate_ratio diagnostic -- a 0 multiplier would
    # ZeroDivide there, so it is refused at construction.
    rate_multiplier: float = Field(default=2.0, gt=0)  # tolerance over the rate ceiling
    # Tolerance over the curve ceilings, the counterpart of rate_multiplier for Check 2:
    # the enforced per-level ceiling is floor(curve_multiplier * k(c, N)). The default
    # matches the rate rule's headroom, so both rules leave a clean stack the same
    # declared room over its own envelope. ge=1: a sub-1 multiplier would gate below
    # the clean envelope's own 97.5% upper bounds, i.e. outside the calibration's
    # false-alarm guarantee.
    curve_multiplier: float = Field(default=2.0, ge=1)
    # Grade a trace the rate rule cannot certify, instead of failing it closed. A
    # declaration, not a tunable number: a task whose traffic cannot reach
    # min_tokens (a smoke-scale demo) says so once, and every verdict it produces
    # carries "allow_underpowered_trace": true beside n_tokens and min_tokens, so the
    # weaker claim is never invisible. The curve rule and every fail-closed guard still
    # apply -- including the zero-token reject, which this flag cannot waive: it
    # licenses grading a short trace, never an empty one.
    allow_underpowered_trace: bool = False

    @property
    def min_tokens(self) -> int:
        """The fail-closed trace floor, in pooled teacher-forced tokens.

        Epistemic, not anti-cheat: below it the rate rule cannot fire at its nominal
        false-alarm rate, so a "faithful" verdict would rest on evidence too thin to
        support it. Computed here rather than stored in the calibration because it is a
        function of ``rate_multiplier`` -- a tighter headroom leaves less room for
        Poisson noise and so needs a longer trace -- and one calibration may be graded
        by several verifiers at different headrooms.

        Not scaled to the expected graded length. Shortening a trace enough
        to matter means dropping requests (which fails the run outright) or ending
        replies early, and an early end-of-sequence where the reference wanted to
        continue is a large-gap flip on every truncated reply -- which is exactly what
        the curve rule counts, at ceilings a handful of such events already exceed."""
        if self.calibration is None:
            raise WiringError(
                "TokenFaithfulnessVerifier.min_tokens needs the injected calibration "
                "(it is derived from q_max and the family budget)")
        return rate_rule_min_tokens(
            self.calibration.q_max, self.calibration.delta_per_rung,
            self.rate_multiplier)
    # Check 3's switch. Default off: it can only run on a contract that requests
    # per-token logprobs with top-k alternatives, which most token-level tasks do
    # not (and cannot flip on casually -- the protocol is inside the workload
    # fingerprint, so requesting logprobs stales the task's calibration).
    check_emit_consistency: bool = False
    requires = frozenset({TeacherForce, StackReport})
    needs_calibration: ClassVar[bool] = True
    # Pooled: this gate is computed once over the whole trace, not per tenant.
    #
    # ``q_max`` is a pooled worst-member statistic measured over the calibration's
    # whole traffic mix, so the only measurement it licenses is a pooled one.
    # Applying it per stratum is incoherent in both directions, because intrinsic
    # flip rates differ by stratum, by up to an order of magnitude, not by
    # implementation quality. A single mean applied to each
    # stratum therefore fails the hottest stratum whatever the server does, while
    # leaving the coldest ample headroom to be broken in.
    pooled: ClassVar[bool] = True

    async def verify(self, engine: EngineAccessor) -> Verdict:
        if self.calibration is None:
            raise WiringError(
                "TokenFaithfulnessVerifier.calibration is None: the loader must inject "
                "the task's calibration.json before verify (hls.verify.calibration)")
        cal = self.calibration
        stack = await engine[StackReport]()
        # Drift = any software-stack difference from the calibrated record (torch +
        # the vendor runtime component versions). device_name is excluded: it is
        # descriptive (see StackInfo's docstring), GPU identity is
        # gpu_kind's gate at read_calibration time, and marketing-name variants of
        # one kind ("NVIDIA H200" vs "NVIDIA H200 NVL") must not refuse a run.
        drift = {k: (v, stack.get(k)) for k, v in cal.stack.model_dump().items()
                 if k != "device_name" and stack.get(k) != v}
        if drift:
            raise CalibrationError(
                f"reference stack drifted from calibration: {drift}; re-run hls calibrate")
        # GPU-identity gate on the serving silicon: read_calibration already compared
        # cal.gpu_kind against the task's declaration, but substrates do not all
        # enforce the declared accelerator, so the reference may be serving on
        # different silicon than was calibrated -- and bf16 noise floors are GPU-kind
        # specific. Kind-token match (device_matches_kind), not verbatim equality:
        # marketing variants of one kind ("NVIDIA H200 NVL") must still pass, which
        # is why device_name sits outside the version-drift set above.
        device = stack.get("device_name", "")
        if not device_matches_kind(device, cal.gpu_kind):
            raise CalibrationError(
                f"reference is serving on {device!r} but the calibration was measured on "
                f"gpu_kind={cal.gpu_kind!r}; bf16 noise floors are GPU-kind specific -- "
                f"re-run hls calibrate on the serving GPU or fix the deployment")
        events = engine[ArtifactRollout]
        # Replies without token ids cannot be teacher-forced, so they are held out of the
        # measurement and rejected on their own count below.
        untokenized_events: list[str] = []
        over_cap_events: list[str] = []
        forced: list = []
        items: list[Envelope] = []
        for ev in events:
            out = self._out_of(ev)
            if out is None:
                untokenized_events.append(ev.event_key)
                continue
            item = self._item(ev, out)
            if past_cap(len(item.first_of("text").tokens), item.metadata["max_tokens"]):
                # The request carried max_tokens and the reply ignored it. Calibration
                # refuses a pool member that does this, and the over_cap_replies rule
                # below fails the trace on it, so its pricing could change no verdict. It
                # is checked before the reference sees it rather than after, as the report
                # gates check it: this changes the order of evaluation only, never a rule or
                # a threshold, and a server that ignores the cap does not choose how much
                # the reference must hold.
                over_cap_events.append(ev.event_key)
                continue
            forced.append(ev)
            items.append(item)
        # Uncached: every item here is built from one contestant's own reply, so no two
        # arms ever ask the same question and a stored answer could never be
        # served. The oracle's cache exists for the questions the corpus determines,
        # which these are not.
        gaps_per_item = await engine.uncached(TeacherForce)(items) if items else []
        rate_thresh = self.rate_multiplier * cal.q_max
        # The measurement, shared verbatim with the calibration that produced
        # this artifact (hls.verify.flips): same flip criterion, same nested level
        # counts, same non-finite accounting, same per-reply record. Every event is
        # counted; what the NaN/+inf counts mean here is this verifier's policy --
        # they fail the trace closed.
        flips = PooledFlips(cal.curve_side.levels_nats)
        empty_events: list[str] = []
        event_diagnostics: dict = {}
        n_capped_events = n_capped_tokens = 0
        for ev, item, gaps in zip(forced, items, gaps_per_item):
            # event_key, not request_id: this gate is pooled, and request_id
            # renumbers from 0 within each tenant, so a bare request_id does not
            # name an event here -- matching every other identifier this verifier
            # collects (per_event, event_diagnostics, emit_violations).
            key = ev.event_key
            n_served = len(item.first_of("text").tokens)
            record = score_reply(gaps, event_key=key, n_served=n_served,
                                 at_cap=n_served == item.metadata["max_tokens"])
            flips.add(record)
            if record.at_cap:
                n_capped_events += 1
                n_capped_tokens += record.n_tokens
            if not gaps:
                empty_events.append(key)
            event_diagnostics[key] = record.to_json()
        n_tokens, n_flips = flips.n_tokens, flips.n_flips
        n_nan, n_inf, max_gap = flips.n_nan, flips.n_inf, flips.max_gap
        flip_rate = n_flips / n_tokens if n_tokens else 0.0
        insufficient = n_tokens < self.min_tokens
        rate_reject = flip_rate > rate_thresh

        # Check 2: the flip-count curve, thresholds recomputed for this trace's
        # pooled length, against the nested per-level flip counts.
        counts = flips.level_counts()
        flip_curve: list[dict] = []
        curve_reject = False
        for row, count in zip(cal.thresholds(n_tokens), counts):
            # Published as enforced ceilings under their own name: the artifact's
            # ``k_per_engine`` is the calibrated count before the multiplier, and
            # ``ceiling_per_engine`` here is after it. The operative ceiling is their
            # max; floor is monotone, so scaling before or after the max gives the
            # same number. ``curve_multiplier`` sits beside the rows so a reader knows
            # what was applied; ``delta`` is the same for every row and is published
            # once, beside them.
            enforced = {name: math.floor(self.curve_multiplier * k)
                        for name, k in row["k_per_engine"].items()}
            curve_reject = curve_reject or count > max(enforced.values())
            flip_curve.append({"c": row["c"], "ceiling_per_engine": enforced,
                               "engine": row["engine"], "e": row["e"],
                               "lambda_up": row["lambda_up"], "mu": row["mu"],
                               "count": count})

        # Check 3: emitted-token consistency (only when the task switched it on).
        emit = self._emit_consistency(events) if self.check_emit_consistency else None

        # Fail-closed guards (see the class docstring): unverifiable evidence is a
        # rejection, never a pass. A trace with no teacher-forced tokens at all carries
        # no evidence -- every rule above is vacuous on it (flip_rate is 0/0, every
        # level count is 0) -- so it is rejected unconditionally:
        # allow_underpowered_trace weakens the claim a verdict makes, and can never
        # license a verdict with nothing behind it.
        rejections = [
            name for name, fired in (
                ("underpowered_trace", insufficient and not self.allow_underpowered_trace),
                ("no_teacher_forced_tokens", n_tokens == 0),
                ("flip_rate", rate_reject),
                ("flip_curve", curve_reject),
                ("emit_consistency", emit is not None
                 and bool(emit["n_emit_violations"] or emit["n_emit_unverifiable"])),
                ("empty_replies", bool(empty_events)),
                ("over_cap_replies", bool(over_cap_events)),
                ("untokenized_replies", bool(untokenized_events)),
                ("nan_gaps", bool(n_nan)),
                ("inf_gaps", bool(n_inf)),
            ) if fired
        ]
        passed = not rejections
        return Verdict(
            passed=passed,
            # No per-event map: the gate is trace-level, so it has no per-event opinion
            # and the runner reads ``passed`` for every event it covers. The per-event
            # detail the viewer renders is ``event_diagnostics``, keyed on event_key
            # rather than request_id -- this gate is pooled, so its map spans tenants,
            # and request_id renumbers from 0 within each tenant.
            diagnostics={
                # Which rules rejected, in the order they are documented. Empty on a
                # pass. The numbers below say by how much; this says which.
                "rejections": rejections,
                # The measurement. Every reject below is a comparison between two of
                # these numbers, so the verdict publishes the numbers and not the
                # comparisons.
                "n_tokens": n_tokens, "n_flips": n_flips, "flip_rate": flip_rate,
                "max_gap": max_gap,
                # The ceilings the numbers are read against.
                "q_max": cal.q_max,
                "rate_multiplier": self.rate_multiplier,
                "curve_multiplier": self.curve_multiplier,
                "rate_ratio": flip_rate / rate_thresh,
                "delta_per_rung": cal.delta_per_rung,
                # One row per gap level: the enforced ceiling per engine (the operative
                # one is their max) and this trace's count at that level.
                "flip_curve": flip_curve,
                # The trace-length floor and the declaration that licenses grading
                # below it. n_tokens against min_tokens says whether the floor was met;
                # the flag says what happens when it was not.
                "min_tokens": self.min_tokens,
                "allow_underpowered_trace": self.allow_underpowered_trace,
                # How much of the graded trace reached the output cap, by served length
                # against the stamped cap. Description only; the tokens are inside
                # n_tokens like any other reply's.
                "n_capped_events": n_capped_events,
                "n_capped_tokens": n_capped_tokens,
                # The unverifiable-evidence counts. Any of them above zero fails the
                # trace closed.
                "n_empty_events": len(empty_events),
                "n_nan_gaps": n_nan, "n_inf_gaps": n_inf,
                # Replies that carried no token ids, so nothing could be forced through
                # the reference. Any above zero fails the trace.
                "n_untokenized_events": len(untokenized_events),
                "untokenized_events": untokenized_events[:SHOWN_EVENTS],
                # Replies served past their stamped cap. Any above zero fails the trace. None
                # of them was teacher-forced, so the measurement above leaves them out and
                # event_diagnostics holds no record of them; the first few are named here.
                "n_over_cap_events": len(over_cap_events),
                "over_cap_events": over_cap_events[:SHOWN_EVENTS],
                "emit_consistency_checked": self.check_emit_consistency,
                **(emit or {}),
            },
            event_diagnostics=event_diagnostics,
        )

    def _emit_consistency(self, events) -> dict:
        """Check 3 over the whole trace: every emitted token's serve-time reported
        log-probability must be >= the contestant's reported per-position maximum
        (exact ties allowed, ``_EMIT_TIE_TOL``). Returns the diagnostics dict.

        Fail-closed under the flag: a position (or whole event) that does not carry
        the reported logprob data is unverifiable and counts as a rejection --
        otherwise a violator could omit the report for its non-greedy
        positions. (A task only reaches here with the contract requesting the data;
        check_wiring refuses the flag otherwise.)
        """
        n_positions = n_violations = n_unverifiable = 0
        first_violations: list[dict] = []
        for ev in events:
            tp = ev.response.output.first_of("text")
            entries = tp.logprobs if tp is not None else None
            if not entries:
                n_unverifiable += 1  # whole event reported no logprobs
                continue
            for i, entry in enumerate(entries):
                n_positions += 1
                tops = entry.top_logprobs
                if not tops:
                    n_unverifiable += 1  # position reported no alternatives
                    continue
                reported_max = max(t.logprob for t in tops)
                if entry.logprob < reported_max - _EMIT_TIE_TOL:
                    n_violations += 1
                    if len(first_violations) < SHOWN_EVENTS:
                        first_violations.append({
                            "event": ev.event_key, "index": i,
                            "logprob": entry.logprob, "reported_max": reported_max,
                        })
        return {
            "n_emit_positions": n_positions,
            "n_emit_violations": n_violations,
            "n_emit_unverifiable": n_unverifiable,
            "emit_violations": first_violations,
        }

    def _out_of(self, ev) -> Envelope | None:
        """The artifact's output envelope, or None when it carries no emitted token ids.

        token_ids is mandatory on the wire contract, so omitting them is the artifact's
        fault and fails the trace closed (``untokenized_events``) rather than raising:
        a reply nobody can teacher-force is unverifiable evidence, not a harness error.
        """
        out = ev.response.output
        tp = out.first_of("text")
        return None if tp is None or tp.tokens is None else out

    def _item(self, ev, out: Envelope) -> Envelope:
        """The teacher-force item for one event: the artifact's continuation and the
        context the reference has to reproduce the position (``_reference_meta``)."""
        return Envelope(request_id=ev.request_id, contents=out.contents,
                        metadata=_reference_meta(ev, type(self).__name__))


class ReportFaithfulnessVerifier(Verifier):
    """Faithfulness under a sampled decode: the distribution the server reported must be
    the pinned reference's, and the tokens it served must be draws from what it reported.

    Under sampling there is no right token to compare against, so the artifact is graded
    on its own report -- at every position it emits, the log-probability it assigned to
    the token it served and to that position's most probable alternatives, with ids
    (``OpenAIChatProtocol(return_logprobs=True, top_logprobs=...)``). Four checks read
    that report, and between them they close both halves of the claim. What each one
    measures, and on what null, is in ``hls.verify.report``; here is what each one
    rejects.

      Check 1 (divergence): ``mean_kl > tolerance * calibration.kappa`` -> fail.
                       ``kappa`` is the divergence a trusted implementation of the same
                       checkpoint already shows against the pinned reference, measured
                       by ``hls calibrate``; ``tolerance`` is the declared headroom over
                       it. This is the check that sees a wrong distribution: different
                       quantization, a broken kernel, the wrong weights. One of the two
                       checks with a measured constant.
      Check 2 (rank uniformity): the Cramer-von Mises p-value of the served tokens'
                       randomized rank transforms under their own reports, below
                       ``alpha`` -> fail. Exactly uniform under an honest sampler at any
                       reply length and under any dependence between positions, so this
                       is the check that sees a server reporting one distribution and
                       sampling from another -- truncating, reweighting, or decoding
                       greedily behind a full-softmax report. Nothing is estimated from
                       the data.
      Check 3 (log score): ``|z| > z_{alpha/2}`` -> fail, with
                       ``z = sum(s_q) / sqrt(sum(var_q))`` standard normal under the
                       same null. A second reading of it through a different
                       functional, which catches shifts in the served tokens'
                       typicality that leave the rank distribution close to uniform.
      Check 4 (end-of-turn count): among the positions where the reference gives the
                       checkpoint's end-of-turn ids a probability of at most
                       ``calibration.end_of_turn.eps``, the number of them the server
                       ended a reply at, against the number the reference expects; its
                       exact Poisson-binomial upper tail at or below ``alpha`` -> fail.
                       The only check that does not read the report, and the only one
                       whose evidence is not diluted by the trace's length, which is
                       what makes it the one that sees an early ending: a reply cut
                       short saves every token after it, at a cost the three means
                       above divide by millions of positions. One-sided -- ending fewer
                       turns than expected is not a deviation -- and calibrated only in
                       which positions count, never in what the count is read against.
      Check 5 (served outside): the fraction of served positions whose token is not
                       one of the reported atoms, above
                       ``tolerance * calibration.outside_ceiling`` -> fail. The
                       partition is built from the report, so the reporter chooses
                       what is in it; a server that reports rare tokens as its
                       alternatives and leaves the served token out of them puts
                       every position in the tail, where checks 1 to 3 see nothing
                       (``hls.verify.report``, The partition). A faithful report puts
                       the served token there at the model's own tail mass, which is
                       what ``hls calibrate`` measures on the trusted stacks. The
                       other check with a measured constant, widened by the same
                       tolerance.

    plus the contract checks, which fail the trace closed rather than being scored
    around (``_violation``): a reply with no report, a report that does not cover every
    served token, a position reporting a different token than the one served, a position
    with too few distinct alternatives to fix the partition's threshold, a position
    reporting a value no log-probability takes, a position whose numbers are not a
    distribution (more probability than one, or the token it served at probability
    zero), a served or reported id outside the reference's vocabulary, a reply served
    past the cap the harness stamped, an empty reply, a normal stop that does not end on
    an end-of-turn id (unless ``implied_end_of_turn`` completes it), and a length stop
    that did not serve the request's whole budget while the context still had room.
    Every one of them is the report failing to be evidence about the emission it claims
    to describe, which is the premise all three checks rest on.

    ``calibration`` is injected by the CLI loader from the task folder's
    calibration.json (``hls.verify.calibration.ReportCalibration``); task authors never
    set it. The verifier also refuses to run against a reference whose stack versions
    drifted from the calibrated ones, or on serving silicon of a different kind, for the
    reasons the token-level gate refuses them -- and against one whose end-of-turn ids
    are not the ones check 4's threshold was chosen on.
    """

    reference: Reference
    calibration: ReportCalibration | None = None  # injected by the loader, never authored
    # Headroom over the calibrated ceilings: the enforced ceilings are tolerance * kappa
    # and tolerance * outside_ceiling. ge=1 because a sub-1 multiplier would gate below
    # the clean pool's own upper limits, i.e. fail a trusted stack for being itself.
    # The default leaves a faithful server twice the divergence a trusted one already
    # shows. It widens
    # checks 1 and 5 and nothing else: the other three read exact nulls, which have no
    # ceiling to widen.
    tolerance: float = Field(default=2.0, ge=1)
    # False-alarm rate of each exact check. All three are tests of a null that holds
    # exactly under a faithful sampler, so this is the probability a faithful server
    # fails one of them by chance; at the millions of positions a graded trace carries,
    # the alternative they reject is many orders of magnitude away from it.
    alpha: float = Field(default=0.01, gt=0, lt=1)
    # Alternatives the report carries at every position, and so the rank the partition's
    # keep threshold is counted in (hls.verify.report, The partition). The contract has
    # to ask for at least this many (check_wiring). Check 5 needs a faithful
    # served-outside rate well under one; a modality whose mass past the default width
    # is heavier asks for a wider report here. Widening only: the field refuses a width
    # below the default.
    report_width: int = Field(default=TOP_LOGPROBS_REQUESTED, ge=TOP_LOGPROBS_REQUESTED)
    # Whether a reply that is not cut, reports finish_reason "stop" and whose last served
    # token is no end-of-turn id is scored as if the server had served one more token, the
    # end-of-turn id the reference rates highest at the next position, rather than refused.
    # That position counts in check 4 as a served ending at the reference's price there,
    # and in nothing else, since the server reported no distribution for it. Of the ids,
    # the most probable is the choice most favourable to the server. Left out of the dump
    # when off, so a gate that leaves it off records no such key.
    implied_end_of_turn: bool = Field(default=False, exclude_if=lambda on: not on)
    requires = frozenset({ReferenceLogprobs, StackReport})
    needs_calibration: ClassVar[bool] = True
    calibration_kind: ClassVar[str] = "report"
    # Pooled, for the reason the token gate is pooled: kappa is measured over the
    # calibration's whole traffic mix, so the only measurement it licenses is one over
    # the whole trace. The two exact checks are pooled for a second reason -- their
    # power comes from the position count, and a per-tenant split would spend the
    # false-alarm budget once per stratum.
    pooled: ClassVar[bool] = True

    async def verify(self, engine: EngineAccessor) -> Verdict:
        if self.calibration is None:
            raise WiringError(
                "ReportFaithfulnessVerifier.calibration is None: the loader must inject "
                "the task's calibration.json before verify (hls.verify.calibration)")
        cal = self.calibration
        stack = await engine[StackReport]()
        # Drift and device-kind refusals as the token-level gate makes them: the
        # ceiling is a claim about one software stack on one kind of silicon, and a
        # reference that is neither cannot reuse it. device_name is excluded from the
        # drift set because it is descriptive and marketing-name variants of one kind
        # must not refuse a run; it is gated by kind below instead.
        drift = {k: (v, stack.get(k)) for k, v in cal.stack.model_dump().items()
                 if k != "device_name" and stack.get(k) != v}
        if drift:
            raise CalibrationError(
                f"reference stack drifted from calibration: {drift}; re-run hls calibrate")
        device = stack.get("device_name", "")
        if not device_matches_kind(device, cal.gpu_kind):
            raise CalibrationError(
                f"reference is serving on {device!r} but the calibration was measured on "
                f"gpu_kind={cal.gpu_kind!r}; the numerical noise this ceiling covers is "
                f"GPU-kind specific -- re-run hls calibrate on the serving GPU or fix "
                f"the deployment")

        # A batch at a time, as ``hls.calibrate.score_report_capture`` does it: build
        # each reply's partition (``_rows_for``, which also names the violations that
        # need no reference), and score and reduce the batch as soon as it is full
        # (``report_call_is_full``, bounding items and positions both). Holding every
        # partition and every reference row for a final pass would keep tens of
        # millions of floats alive beside the trace, in the process that has to survive
        # to write the verdict.
        #
        # ``facts`` is what a reply's ending is read against. The context window is the
        # task's own declaration, which is what its servers run in and what the
        # calibration pool's members were built at; the checkpoint's declared window is
        # a different, usually larger number, and reading that one would fail an honest
        # server for filling the window the task told it to use. Taken from the
        # calibration's fingerprint payload, which the loader has already refused if the
        # task moved off it. The end-of-turn ids are the checkpoint's and the same in
        # every reply, so the first batch fixes them.
        facts: dict = {"eot_ids": set(),
                       "max_context": cal.workload.get("max_context_len")}
        # Of a refused reply only what its record reads is kept, never the event:
        # a trace of refused replies is otherwise held whole.
        refused: list[tuple[_Refusal, str]] = []
        scored: list[tuple[str, str | None, ReplyReport, int]] = []
        # Cut replies that streamed nothing. They report nothing and are graded by
        # nothing, and unlike an empty reply the server chose to send, they are not the
        # artifact taking itself out of the gate: the harness cut them before their
        # first token. Recorded so the verdict shows they existed.
        ungraded: list[_Refusal] = []
        batch: list[tuple[object, Envelope, ReportRows]] = []
        batch_positions = 0
        for ev in engine[ArtifactRollout]:
            if _was_cut(ev) and not _served_tokens(ev):
                ungraded.append(self._refusal(ev))
                continue
            built = self._rows_for(ev)
            if isinstance(built, str):
                refused.append((self._refusal(ev), built))
                continue
            rows, item = built
            batch.append((ev, item, rows))
            batch_positions += len(rows.q_ids)
            if report_call_is_full(len(batch), batch_positions):
                scored.extend(await self._score_batch(engine, batch, facts))
                batch, batch_positions = [], 0
        if batch:
            scored.extend(await self._score_batch(engine, batch, facts))
        eot_ids = facts["eot_ids"]

        # The end-of-turn count is a claim about one set of ids, so a reference that
        # names a different set is not the checkpoint the threshold was chosen on --
        # the same refusal the drifted stack and the wrong silicon get, and for the
        # same reason. Asked only when some reply came back with the facts on it: a
        # trace whose every reply was refused before the call, or closed by the
        # reference as unscorable, reports no ids and fails on those replies instead.
        if eot_ids and eot_ids != set(cal.end_of_turn.ids):
            raise CalibrationError(
                f"the reference reports end-of-turn ids {sorted(eot_ids)} but the "
                f"calibration was taken on {cal.end_of_turn.ids}; the threshold its "
                f"end-of-turn count reads was chosen for that checkpoint -- re-run "
                f"`uv run hls calibrate` for this task or fix the deployed reference")

        # The statistics, per reply and pooled. The replies the reference never saw go
        # first, because only the reference knows the checkpoint's end-of-turn ids and
        # their records name whether the reply ended on one.
        # The records go into the pool whole and into the verdict summarized. The pool
        # needs each reply's per-position values -- the uniformity test and the
        # end-of-turn count are over the whole trace at once -- but the verdict is
        # written out, and what it would carry there is two floats per served token
        # that nothing reads back; the statistics over them are in `diagnostics` below.
        pooled = PooledReport()
        event_diagnostics: dict = {}
        violations: list[tuple[str, str]] = []
        n_short = n_implied = 0
        for refusal in ungraded:
            record = self._refused_record(refusal, None, eot_ids)
            pooled.add(record)
            event_diagnostics[refusal.event_key] = record.summary_json()
        for refusal, reason in refused:
            record = self._refused_record(refusal, reason, eot_ids)
            violations.append((refusal.event_key, reason))
            pooled.add(record)
            event_diagnostics[refusal.event_key] = record.summary_json()
        for key, reason, record, short in scored:
            if reason is not None:
                violations.append((key, reason))
            n_short += short
            n_implied += record.implied_eot is not None
            pooled.add(record)
            event_diagnostics[key] = record.summary_json()

        ceiling = self.tolerance * cal.kappa
        mean_kl = pooled.mean_kl
        outside_limit = self.tolerance * cal.outside_ceiling
        outside_rate = pooled.outside_rate
        omega2, cvm_p = pooled.uniformity()
        z = pooled.z
        z_alpha = _normal_two_sided_critical(self.alpha)
        # Check 4 reads the calibration for one number, which positions are cheap
        # enough to count. The declared tolerance does not widen it: the count's null
        # mean comes from the trace itself, so there is no ceiling for a tolerance to
        # multiply.
        eot = pooled.end_of_turn(cal.end_of_turn.eps)
        # Each comparison is strict and unguarded, which is what makes it fail closed in
        # both directions. An infinite mean fires: a report that puts mass where the
        # reference puts none diverges without bound, and that is a verdict rather than
        # a number to discard. A statistic with nothing behind it is NaN, which loses
        # every strict comparison and so fires nothing -- the empty trace is rejected on
        # its own count above instead of resting on a vacuous pass.
        rejections = [
            name for name, fired in (
                # Unverifiable evidence first: the checks below are statements about
                # reports that describe the emission they came with.
                ("contract_violations", bool(violations)),
                ("no_measured_positions", pooled.n_tokens == 0),
                ("divergence", mean_kl > ceiling),
                ("rank_uniformity", cvm_p < self.alpha),
                ("log_score", abs(z) > z_alpha),
                # At or below alpha, where the other two are strictly below it: this
                # tail is a sum of atoms rather than a continuous distribution, so the
                # attainable p-values are a discrete set and the size of a strict test
                # would sit under the declared alpha rather than at it.
                ("end_of_turn_count", eot.p_value <= self.alpha),
                ("served_outside", outside_rate > outside_limit),
            ) if fired
        ]
        return Verdict(
            passed=not rejections,
            # No per-event map: the gate is trace-level, so the runner reads ``passed``
            # for every event it covers. The per-event detail is event_diagnostics,
            # keyed on event_key rather than request_id -- this gate is pooled, so its
            # map spans tenants, and request_id renumbers from 0 within each tenant.
            diagnostics={
                # Which checks rejected, in the order they are documented. Empty on a
                # pass. The numbers below say by how much; this says which.
                "rejections": rejections,
                # Check 1: the measurement, the ceiling it is read against, and how
                # much of the ceiling it used.
                "mean_kl": mean_kl, "se_kl": pooled.se_kl,
                "kappa": cal.kappa, "tolerance": self.tolerance, "ceiling": ceiling,
                "utilization": mean_kl / ceiling if ceiling > 0 else float("nan"),
                # Checks 2 and 3, each with its own critical value beside it.
                "cvm_omega2": omega2, "cvm_p": cvm_p, "alpha": self.alpha,
                "z": z, "z_critical": z_alpha, "n_moment": pooled.n_moment,
                # Check 4: the threshold that chose the positions, how many there
                # were, and the two counts whose comparison the p-value is. Read
                # against the same alpha as the two above.
                "end_of_turn": eot.to_json(),
                # What the statistics were computed over: every served position of every
                # reply that was measured at all.
                "n_replies": pooled.n_replies, "n_tokens": pooled.n_tokens,
                # Check 5: positions whose served token was not one of the reported
                # atoms, as a count and as the rate the limit is read against, with the
                # limit and how much of it was used.
                "n_served_outside": pooled.n_outside,
                "outside_rate": outside_rate,
                "outside_ceiling": cal.outside_ceiling, "outside_limit": outside_limit,
                "outside_utilization": (outside_rate / outside_limit if outside_limit > 0
                                        else float("nan")),
                # Positions whose report arrived narrower than the contract asked for.
                # The partition still holds there, so this describes the serving stack
                # rather than gating anything.
                "n_short_positions": n_short,
                # The contract violations. Any above zero fails the trace; the first
                # few are named so a reader knows what to fix.
                "n_violations": len(violations),
                "violations": [{"event": k, "reason": r} for k, r in violations[:SHOWN_EVENTS]],
                # Normal stops scored with an implied end-of-turn. Present only when the
                # rule is on.
                **({"n_implied_eot": n_implied} if self.implied_end_of_turn else {}),
            },
            event_diagnostics=event_diagnostics,
        )

    def _violated(self, ev, reason: str, eot_ids: set[int]) -> ReplyReport:
        """The record of a reply whose report violates the contract.

        Its measurement counts are zero because it was never measured: the trace fails
        on the violation itself, and a statistic over a report that does not describe
        the emission it came with would mean nothing. What the reply said about itself
        is still recorded, so a reader can see what it claimed."""
        return self._refused_record(self._refusal(ev), reason, eot_ids)

    def _refusal(self, ev) -> _Refusal:
        """What the record of a refused reply reads of *ev* (``_violated``)."""
        tp = ev.response.output.first_of("text") if ev.response is not None else None
        served = tp.tokens if tp is not None and tp.tokens else []
        return _Refusal(event_key=ev.event_key, n_served=len(served), last=served[-1] if served else None,
                        cap=ev.input.metadata.get("max_tokens"), finish_reason=self._finish_reason(ev))

    @staticmethod
    def _refused_record(refusal: _Refusal, reason: str | None, eot_ids: set[int]) -> ReplyReport:
        """``_violated``'s record, from what it reads of the reply (*refusal*).

        ``reason`` is ``None`` for the one reply that is unmeasured without being
        wrong: a cut request that streamed nothing before the harness stopped
        listening."""
        return ReplyReport(
            event_key=refusal.event_key, n_tokens=refusal.n_served, n_outside=0,
            kl_sum=0.0, s_sum=0.0, var_sum=0.0, n_moment=0, u=(),
            p_eot_lp=(), eot_positions=(),
            finish_reason=refusal.finish_reason,
            ends_with_eot=refusal.n_served > 0 and refusal.last in eot_ids,
            at_cap=refusal.n_served > 0 and refusal.n_served == refusal.cap, violation=reason)

    async def _score_batch(self, engine, batch: list[tuple[object, Envelope, ReportRows]],
                           facts: dict) -> list[tuple[str, str | None, ReplyReport, int]]:
        """Ask the reference about one batch and reduce it (``reduce_replies``).

        Returning records rather than rows is the point: the batch's partitions and the
        reference's rows are the largest thing in this verifier, and this is where they
        are released."""
        replies = await engine[ReferenceLogprobs]([item for _ev, item, _rows in batch])
        if len(replies) != len(batch):
            raise ValueError(
                f"the reference returned {len(replies)} replies for {len(batch)} items; "
                f"the report is scored one reply per item")
        return self.reduce_replies(batch, replies, facts)

    def reduce_replies(self, batch: list[tuple[object, Envelope, ReportRows]],
                       replies: list[dict],
                       facts: dict) -> list[tuple[str, str | None, ReplyReport, int]]:
        """One reference call's worth of replies, each reduced to its record.

        ``batch`` holds, per reply, the event, the reference item built from it and its
        partition (``_rows_for``); ``replies`` is the reference's answer to those items,
        in the same order.

        Returns, per reply in the batch's order: its event key, why its report violates
        the contract (``None`` when it does not), the record, and how many of its
        positions reported fewer alternatives than the contract asked for -- zero on a
        violated reply, which enters no sum.

        ``facts`` is the caller's accumulator for what a reply's ending is read against.
        The end-of-turn ids are unioned into it; the context window is filled from the
        reference only if the caller seeded none, because the task's own declaration is
        the window its servers run in and the checkpoint's is a different number.

        Grading and calibration share this, and it is the whole of what they share once
        the reference has answered: the ceiling a trace is graded against was measured
        by ``hls.calibrate.score_report_capture`` through these lines, so a rule added
        here for one of them is a rule the other is already held to. The numeric half
        of the reduction is shared the same way, in ``hls.verify.report``."""
        facts.setdefault("eot_ids", set()).update(
            int(t) for r in replies for t in r.get("eot_ids", ()))
        if facts.get("max_context") is None:
            facts["max_context"] = next((r["max_context"] for r in replies
                                         if r.get("max_context") is not None), None)
        eot_ids = facts["eot_ids"]
        out: list[tuple[str, str | None, ReplyReport, int]] = []
        for (ev, item, rows), reply in zip(batch, replies):
            key = ev.event_key
            served = item.first_of("text").tokens
            ends_with_eot = served[-1] in eot_ids
            reason = reply.get("invalid") or self._ending_violation(
                ev, item, served, reply.get("n_ctx", 0), facts["max_context"],
                ends_with_eot)
            if reason is not None:
                out.append((key, reason, self._violated(ev, reason, eot_ids), 0))
                continue
            implied = None if ends_with_eot else self._implied_eot(ev, reply)
            stats = position_stats(
                served=served, q_ids=rows.q_ids, q_lp=rows.q_lp,
                q_emit=rows.q_emit, p_on_q=reply["p_on_q"],
                p_tail_lp=reply["p_tail_lp"], p_eot_lp=reply["p_eot_lp"],
                eot_ids=eot_ids, v=reply_uniforms(key, len(served)))
            out.append((key, None, score_report_reply(
                stats, event_key=key, n_served=len(served),
                finish_reason=self._finish_reason(ev),
                ends_with_eot=ends_with_eot,
                at_cap=len(served) == item.metadata["max_tokens"],
                count_end_of_turn=not _was_cut(ev), implied_eot=implied), rows.n_short))
        return out

    def _completes_stop(self, ev) -> bool:
        """Whether the implied end-of-turn rule applies to this reply if it did not
        end on an end-of-turn id: the rule is on, the harness did not cut the reply,
        and the server reported a normal stop."""
        return (self.implied_end_of_turn and not _was_cut(ev)
                and self._finish_reason(ev) == "stop")

    def _implied_eot(self, ev, reply: dict) -> tuple[int, float] | None:
        """The end-of-turn id a reply that did not serve one is scored as having
        served next, with the reference's log end-of-turn mass at that position, or
        ``None`` when the rule does not apply.

        The id is the one the reference gives the highest log-probability there. Which
        id it is changes no statistic, since the count prices the position by the mass
        of all of them; it is recorded so a reader can see what was assumed."""
        if not self._completes_stop(ev):
            return None
        lps = reply.get("p_stop_lp")
        if lps is None:
            raise RuntimeError(
                f"event {ev.event_key!r} reported a normal stop without an end-of-turn "
                f"token, and the reference returned no stop row to complete it; the "
                f"item asked for one (metadata['stop_row']), so the engine did not "
                f"honour the request")
        ids = reply["eot_ids"]
        best = max(range(len(ids)), key=lambda i: lps[i])
        return int(ids[best]), float(reply["p_stop_eot_lp"])

    def _rows_for(self, ev) -> tuple[ReportRows, Envelope] | str:
        """One event's partition and its reference item, or the reason its report
        violates the contract.

        The item carries what the reference needs to reproduce the position: the
        conversation the request held, the stamped output cap, the turn's tools, and the
        kept ids per position (``report_ids``), which is what the reference gathers its
        own log-probabilities at."""
        out = ev.response.output if ev.response is not None else None
        tp = out.first_of("text") if out is not None else None
        if tp is None or tp.tokens is None:
            return "the reply carries no emitted token ids"
        if not tp.tokens:
            # An empty reply reports nothing and is graded by nothing. Greedy or
            # sampled, a faithful server always emits at least one token, so this is
            # the artifact exempting itself from the gate.
            return "the reply served no tokens"
        cap = ev.input.metadata.get("max_tokens")
        if isinstance(cap, int) and past_cap(len(tp.tokens), cap):
            # The request carried max_tokens and the reply ignored it: a contract
            # violation, not excess text to score. Grading fails the trace on it; a
            # report calibration records it and leaves it out of the measurement.
            return f"served {len(tp.tokens)} tokens past its cap of {cap}"
        rows = reply_report(tp, tp.tokens, width=self.report_width)
        if isinstance(rows, str):
            return rows
        meta = _reference_meta(ev, type(self).__name__)
        # report_ids is what the reference gathers its own log-probabilities at: the
        # partition this reply's own report fixes, one row per served token.
        meta["report_ids"] = rows.q_ids
        # The position after the last token, for the implied end-of-turn rule. Asked
        # for on every reply the rule could apply to, because whether the last token
        # is an end-of-turn id is known only once the reference names the ids.
        if self._completes_stop(ev):
            meta["stop_row"] = True
        # The reference reads the served ids and the partition, never the report itself,
        # and the report is report_width entries per token: shipping it would make the request
        # body tens of times larger than the reply it fetches.
        contents = [c.model_copy(update={"logprobs": None}) if isinstance(c, TextPayload)
                    else c for c in out.contents]
        return rows, Envelope(request_id=ev.request_id, contents=contents,
                              metadata=meta)

    def _ending_violation(self, ev, item, served, n_ctx: int,
                          max_context: int | None, ends_with_eot: bool) -> str | None:
        """Why the reply's ending breaks the contract, or ``None`` when it does not.

        Two rules, and both exist because a server that stops early serves fewer tokens
        for the same credit while every token it did serve is a faithful draw -- which
        no statistic over those tokens can see. A reply that claims to have ended its
        turn must have ended it in band, on one of the checkpoint's own end-of-turn ids.
        Under ``implied_end_of_turn`` a reply reporting ``stop`` without one passes
        here instead, and the missing token is scored where the same early stop is
        still visible: the end-of-turn count prices ending at the next position
        (``_implied_eot``). A reply reporting any other reason still fails.
        A reply that claims to have run out of budget must have spent the whole budget
        the request stamped, unless the context window ran out first, which is a fact
        about the prompt the reference measured rather than a claim of the server's.
        ``max_context`` is the window the task declares its servers run in; where it
        declares none, the checkpoint's own. A window from neither source gets no
        exemption: an unverifiable one would be a free pass.

        Both rules read a claim the server made about how its reply ended. A reply the
        harness cut mid-stream makes none, since it ends where the harness stopped
        listening, so there is nothing here to check. The exemption is the cut's alone,
        which the driver flags; a server reporting no finish reason of its own is still
        the first violation.

        The cap comes off the ITEM rather than off the request, because the comparison
        below is against the served POSITIONS and the two are not always the same unit.
        For text they are: a position is a token and ``_rows_for`` copies the request's
        cap across unchanged. For a modality whose unit of generation carries several
        positions they are not, and reading the request's number here would compare
        positions against frames and exempt every reply that claimed to run out of
        budget. The item is where a subclass states the cap in the unit its positions
        are counted in."""
        if _was_cut(ev):
            return None
        cap = item.metadata["max_tokens"]
        finish = self._finish_reason(ev)
        if finish == "length":
            if len(served) >= cap:
                return None
            if max_context is not None and n_ctx + len(served) >= max_context:
                return None
            return (f"reports finish_reason='length' after {len(served)} of "
                    f"{cap} tokens, with {n_ctx + len(served)} of "
                    f"{max_context} context tokens used")
        if not ends_with_eot and not self._completes_stop(ev):
            return (f"reports finish_reason={finish!r} but its last token "
                    f"{served[-1]} is not one of the checkpoint's end-of-turn ids")
        return None

    def _finish_reason(self, ev) -> str | None:
        """The reply's own reported finish reason, or ``None`` when it reported none."""
        tp = ev.response.output.first_of("text") if ev.response is not None else None
        return tp.finish_reason if tp is not None else None
