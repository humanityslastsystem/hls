"""Verification abstractions -- the framework for correctness gates.

A ``Verifier`` is a pure pydantic config that owns its reference engine config(s),
declares the capabilities it ``requires``, and composes capability calls in
``verify(self, engine) -> Verdict``. The verification core (``hls.verify.run``) builds the
declared engines, loads them on the role's engine host and hands the verifier an
``EngineAccessor``: ``engine[Cap]`` is the loaded engine's bound async method;
``engine[ArtifactRollout]`` is the trace's events. ``check_wiring`` validates requirements + output-modality
compatibility from config alone, before any GPU.

A ``RolloutMetric`` is the optional pure scoring hook a verifier composes. The
provided concrete metrics live in ``hls.verify.metrics``; the provided concrete
verifiers in ``hls.verify.verifiers``.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Iterable
from typing import Any, ClassVar

from hls.core.capabilities import Capability, Generate, ReferenceLogprobs, TeacherForce
from hls.core.decode import Greedy, Sample
from hls.core.engine import supports
from hls.core.payloads import Envelope
from hls.core.poly import Poly
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.core.verdict import MetricResult, Verdict
from hls.reference.references import Reference
from hls.verify.answers import CORPUS_CAPABILITIES
from hls.verify.report import TOP_LOGPROBS_REQUESTED


class ArtifactRollout:
    """Sentinel key for the artifact rollout outputs (the trace's events)."""


class WiringError(Exception):
    pass


class EngineAccessor:
    """Runtime accessor handed to ``verify``. Built by the verification core."""

    def __init__(
        self, *, artifact_events: Iterable[Any], cap_to_provider: dict[type[Capability], Callable],
        cap_to_uncached: dict[type[Capability], Callable] | None = None,
    ) -> None:
        #: The trace's events this verifier grades (``engine[ArtifactRollout]``): sized, and taken
        #: again on each pass, which for a stored trace reads them from the store again
        #: (``hls.trace.store.StoredEvents``).
        self._events = artifact_events
        self._cap = cap_to_provider
        # Defaults to the cached providers, so on an accessor built without it (tests)
        # ``uncached(key)`` is ``self[key]``.
        self._uncached = cap_to_uncached if cap_to_uncached is not None else cap_to_provider

    def uncached(self, key):
        """The same engine as ``self[key]``, with its answer cache bypassed.

        For a question whose answer depends on the contestant rather than on a corpus
        row: the price of a contestant's reply under teacher forcing, or an oracle
        answer to a prompt that carries the contestant's own earlier replies. No other
        run asks that question, so a stored answer could never be served; a verifier
        asking it says so by asking here.

        The cache wraps only ``hls.verify.answers.CORPUS_CAPABILITIES`` on a reference
        with caching on. For any other capability, or on a reference with caching off,
        this is ``self[key]``. Same deployed engine either way -- only the cache wrapper
        differs -- so reaching for this costs no extra device.
        """
        if key is ArtifactRollout:
            raise WiringError("ArtifactRollout is the trace, not an engine; it has no cache")
        return self._resolve(self._uncached, key)

    def __getitem__(self, key):
        if key is ArtifactRollout:
            return self._events
        return self._resolve(self._cap, key)

    @staticmethod
    def _resolve(providers, key):
        try:
            return providers[key]
        except KeyError:
            # check_wiring validates declared capabilities against `requires`, not which
            # engine[Cap] the body actually calls; that gap is only catchable here.
            raise WiringError(
                f"verify() requested capability {getattr(key, '__name__', key)!r} but no "
                f"declared engine provides it; add the providing Reference to the verifier "
                f"and list the capability in `requires`"
            ) from None


class Verifier(Poly):
    """Binary correctness check. Owns reference config(s); composes capabilities."""

    requires: ClassVar[frozenset] = frozenset()  # frozenset[type[Capability]]; class-level, not a field
    accepts: ClassVar[frozenset[str]] = frozenset({"text"})  # output modalities this verifier can consume
    # True on verifiers whose thresholds come from a task-folder calibration file; the
    # CLI loader injects it and refuses to run without it.
    needs_calibration: ClassVar[bool] = False
    # Which calibration artifact this verifier reads, by the ``kind`` tag the file
    # carries (hls.verify.calibration.CALIBRATION_KINDS). Read only when
    # needs_calibration: the loader validates the file into that artifact's model and
    # refuses one of any other kind, so a task cannot be graded on numbers measured for
    # a different gate. The default is the token-level gate's.
    calibration_kind: ClassVar[str] = "flips"
    # Pooled gates see the whole trace at once instead of one tenant's slice, producing
    # a single verdict shared by every tenant. Default False: a content verifier's
    # ground truth is per-tenant. Set True only when the threshold itself is a pooled
    # statistic, so comparing it against one stratum wouldn't match the calibration.
    pooled: ClassVar[bool] = False
    # True on a verifier that grades the audio codec codes a reply carries
    # (``AudioPayload.codec_codes``). check_wiring then requires a contract whose wire
    # protocol asks for them, so a gate wired to a protocol that never requests codes
    # fails at authoring instead of failing every reply at verify time. The audio
    # counterpart of the token-id check a TeacherForce verifier gets.
    requires_codec_codes: ClassVar[bool] = False

    def engines(self) -> list[Reference]:
        """The reference engine configs this verifier owns (its Reference-typed fields)."""
        out: list[Reference] = []
        for name in type(self).model_fields:
            val = getattr(self, name)
            # A list/tuple containing a Reference is invisible to this scalar collector,
            # so check_wiring may false-pass and the verification core never loads the engine.
            # No verifier uses one today; fail loud rather than defer to recursive
            # flattening until an ensemble verifier actually needs it.
            if isinstance(val, (list, tuple)) and any(isinstance(x, Reference) for x in val):
                raise WiringError(
                    f"{type(self).__name__}.{name} is a {type(val).__name__} containing a "
                    f"Reference; container-of-Reference fields are not supported (declare "
                    f"each Reference as its own scalar field)"
                )
            if isinstance(val, Reference):
                out.append(val)
        return out

    @abstractmethod
    async def verify(self, engine: EngineAccessor) -> Verdict: ...


def capability_providers(verifier: Verifier) -> dict[type[Capability], Reference]:
    """Map each capability to the verifier's engine config that provides it.

    Fails loud on a capability provided by zero or by more than one engine.
    """
    out: dict[type[Capability], Reference] = {}
    for cfg in verifier.engines():
        eng_cls = cfg.engine_cls  # capabilities from config; no engine built here
        for cap in supports(eng_cls):
            if cap in out:
                raise WiringError(
                    f"{type(verifier).__name__}: capability {cap.__name__} provided by more than one engine"
                )
            out[cap] = cfg
    return out


def check_wiring(verifiers: list[Verifier], contract, slo=None) -> None:
    """Validate (pre-deploy, from config alone) that each verifier's requirements
    are satisfied by exactly one declared engine, that it can consume the contract's
    output modalities, and that protocol prerequisites hold.

    ``slo`` (optional): the tenant's SLO. An SLO that declares
    ``requires_token_timing`` needs a streaming contract that emits token ids;
    asserted here so a misconfigured streaming SLO fails loud at authoring
    rather than silently waiving its TPOT half at scoring time."""
    # Fail closed on no verifiers: an empty list makes the verification core's run-level and
    # per-event all(...) vacuously True, crediting full throughput with zero
    # correctness evidence.
    if not verifiers:
        raise WiringError(
            "TaskSpec declares no verifiers: a task with no correctness check would pass "
            "vacuously (all([]) is True). Declare at least one Verifier (fail closed)."
        )
    out_modalities = frozenset(getattr(contract, "output_modalities", frozenset()))
    for v in verifiers:
        provided = capability_providers(v)
        missing = set(v.requires) - set(provided)
        if missing:
            raise WiringError(
                f"{type(v).__name__} requires {{{', '.join(c.__name__ for c in missing)}}} "
                f"but its engines provide {{{', '.join(c.__name__ for c in provided)}}}"
            )
        # The answer cache holds a reference's own answers to corpus rows and nothing
        # else, and the verification core wraps no other capability. A caching reference a
        # verifier requires for anything else is configured to cache what the cache
        # cannot hold.
        for cfg in v.engines():
            if not cfg.cache:
                continue
            uncachable = sorted(cap.__name__ for cap in v.requires
                                if provided[cap] is cfg and cap not in CORPUS_CAPABILITIES)
            if uncachable:
                raise WiringError(
                    f"{type(v).__name__} requires {type(cfg).__name__}, a reference with "
                    f"cache=True, for {', '.join(uncachable)}. The answer cache holds only "
                    f"a reference's answers to corpus rows "
                    f"({', '.join(c.__name__ for c in CORPUS_CAPABILITIES)}); these answers "
                    f"depend on a contestant's reply or describe the stack, so none could "
                    f"be served from it. Set cache=False on this reference."
                )
        # The verifier must accept every output modality the contract emits, or it
        # would silently mis-extract an unreadable modality at verify time instead
        # of failing loud here at authoring.
        unaccepted = out_modalities - v.accepts
        if unaccepted:
            raise WiringError(
                f"{type(v).__name__} accepts output modalities {set(v.accepts)} but the "
                f"contract emits {set(out_modalities)} (cannot consume {set(unaccepted)})"
            )
        # A TeacherForce verifier teacher-forces what the artifact emitted, so the
        # contract's wire protocol must actually surface it. What that is depends on
        # the modality: emitted token ids for text, and the audio codec's codes for a
        # verifier that declares it grades those. Either way the protocol's own
        # capability marker answers it, rejecting a mis-wire at authoring instead of
        # letting every rollout request fail at decode.
        if TeacherForce in v.requires:
            proto = getattr(contract, "protocol", None)
            if v.requires_codec_codes:
                emitted, marker = "audio codec codes", "emits_codec_codes"
            else:
                emitted, marker = "emitted token ids", "emits_token_ids"
            if not getattr(proto, marker, False):
                raise WiringError(
                    f"{type(v).__name__} requires TeacherForce (teacher-forces the "
                    f"artifact's {emitted}); the contract's {type(proto).__name__} "
                    f"does not emit them (WireProtocol.{marker} is False)"
                )
        # A codec-codes verifier grades the codes the reply carries, which only reach
        # the payload when the protocol asked the server for them. Same reasoning as
        # the token-id check above: without this, every reply would arrive with no
        # codes and fail closed, reporting a contestant defect that is really a
        # mis-wired task.
        if v.requires_codec_codes:
            proto = getattr(contract, "protocol", None)
            if not getattr(proto, "emits_codec_codes", False):
                raise WiringError(
                    f"{type(v).__name__} grades the reply's audio codec codes; the "
                    f"contract's {type(proto).__name__} does not ask for them "
                    f"(WireProtocol.emits_codec_codes is False)"
                )
        # A ReferenceLogprobs verifier grades the distribution the artifact reported at
        # each position it served, so the contract has to ask for that report: the
        # served token's id (to line the report up with the emission) and its top-k
        # alternatives, at least as wide as the partition's threshold rank. Asserted
        # here, or every reply would fail closed as unreportable at verify time.
        if ReferenceLogprobs in v.requires:
            proto = getattr(contract, "protocol", None)
            # Which ids those are depends on the modality, the same way the
            # TeacherForce prerequisite above does: emitted token ids for text, and the
            # audio codec's codes for a verifier that declares it grades those.
            if v.requires_codec_codes:
                emitted, marker = "audio codec codes", "emits_codec_codes"
            else:
                emitted, marker = "emitted token ids", "emits_token_ids"
            if not getattr(proto, marker, False):
                raise WiringError(
                    f"{type(v).__name__} requires ReferenceLogprobs (it grades the "
                    f"artifact's reported distribution against the ids it served); the "
                    f"contract's {type(proto).__name__} does not emit {emitted} "
                    f"(WireProtocol.{marker} is False)"
                )
            top = getattr(proto, "top_logprobs", None) or 0
            width = getattr(v, "report_width", TOP_LOGPROBS_REQUESTED)
            if not (getattr(proto, "return_logprobs", False) and top >= width):
                raise WiringError(
                    f"{type(v).__name__} requires ReferenceLogprobs, which grades the "
                    f"artifact's own per-token report; the contract's "
                    f"{type(proto).__name__} requests return_logprobs="
                    f"{getattr(proto, 'return_logprobs', False)!r} with "
                    f"top_logprobs={getattr(proto, 'top_logprobs', None)!r}. It needs "
                    f"return_logprobs=True and top_logprobs >= "
                    f"{width}, the gate's report width, one more than the partition keeps, so "
                    f"the keep threshold is a value that is on the wire whatever the "
                    f"server drew (hls.verify.report)."
                )
            # The whole measurement is about a distribution the server sampled from.
            # Under greedy decode there is none: the report would describe a
            # distribution nothing was drawn from, both exact nulls would be
            # meaningless, and a faithful server would fail the uniformity test at
            # every position.
            decode = getattr(v.reference, "decode", None)
            if not isinstance(decode, Sample):
                raise WiringError(
                    f"{type(v).__name__} grades a sampled decode -- it tests the served "
                    f"tokens for being draws from the distribution the server reported "
                    f"-- but its reference decodes with {decode!r}. Declare "
                    f"Sample(temperature=...)."
                )
        # The mirror of the check above. A Generate verifier scores the artifact
        # against the reference's own output as a fixed bar, so that output has to be
        # fixed. A sampled reference draws instead: the bar moves from one re-grade to
        # the next, and a prompt's reply additionally depends on which prompts shared
        # its batch (hls.reference.engines.HFEngine.generate), so the comparison is
        # against noise. Greedy is required positively, so a future decode strategy is
        # refused until someone shows its output is reproducible.
        if Generate in v.requires:
            decode = getattr(provided[Generate], "decode", None)
            if not isinstance(decode, Greedy):
                raise WiringError(
                    f"{type(v).__name__} requires Generate -- it scores the artifact "
                    f"against the reference's own output as a fixed bar -- but its "
                    f"reference decodes with {decode!r}, whose replies are draws "
                    f"nothing pins, so the same trace passes or fails across "
                    f"re-grades. Declare Greedy()."
                )
        # The emitted-token consistency check compares each emitted token's reported
        # logprob against the contestant's reported per-position maximum, so the wire
        # protocol must actually request per-token logprobs with top-k alternatives.
        # Assert it here, or every position would fail closed as unverifiable at
        # verify time.
        #
        # Flipping this flag changes the workload fingerprint (the contract is inside
        # workload_payload), so enabling it on an already-calibrated task also means
        # recalibrating it.
        if getattr(v, "check_emit_consistency", False):
            proto = getattr(contract, "protocol", None)
            if not (getattr(proto, "return_logprobs", False)
                    and getattr(proto, "top_logprobs", None)):
                raise WiringError(
                    f"{type(v).__name__} has check_emit_consistency=True, which needs "
                    f"the contestant's per-token reported logprobs AND top-k "
                    f"alternatives on the wire; the contract's {type(proto).__name__} "
                    f"does not request them (needs return_logprobs=True and "
                    f"top_logprobs >= 1)"
                )
            # The check compares each emitted token against the contestant's own
            # reported per-position maximum, and which distribution an engine
            # reports under a penalty is the engine's own choice: vLLM reports the
            # raw logprobs (computed before penalties), SGLang the penalized ones.
            # On an engine reporting raw, a faithful server's emitted token is
            # legitimately not its reported argmax, so every position the penalty
            # moved would fail. There is no engine-neutral reading, so refuse.
            if getattr(getattr(v.reference, "decode", None), "penalized", False):
                raise WiringError(
                    f"{type(v).__name__} has check_emit_consistency=True under a "
                    f"decode with frequency_penalty="
                    f"{v.reference.decode.frequency_penalty}; the check is unavailable "
                    f"there. It compares each emitted token to the contestant's own "
                    f"reported per-position maximum, and an engine may report either "
                    f"the raw distribution or the penalized one (vLLM reports raw). "
                    f"On one reporting raw, a faithful server fails at every position "
                    f"the penalty moved. Set check_emit_consistency=False."
                )
    # A streaming-timing SLO needs an http-sse contract whose protocol emits token ids
    # (per-token timing keys off them); assert both so a misconfigured SLO fails loud
    # at authoring rather than silently waiving its TPOT half at scoring.
    if slo is not None and getattr(slo, "requires_token_timing", False):
        if getattr(contract, "transport", None) != "http-sse":
            raise WiringError(
                f"{type(slo).__name__} requires per-token streaming timing; the "
                f"contract must set transport='http-sse'"
            )
        proto = getattr(contract, "protocol", None)
        if not getattr(proto, "emits_token_ids", False):
            raise WiringError(
                f"{type(slo).__name__} requires per-token streaming timing, which keys "
                f"off token ids; the contract's {type(proto).__name__} does not emit "
                f"token ids (WireProtocol.emits_token_ids is False)"
            )


class RolloutMetric(Poly):
    """Pure: scores a batch of artifact rollout events into a reduced MetricResult.

    The metric owns its reduction (mean-F1, accuracy, micro-F1, ...) -> MetricResult.value;
    the verifier only gates or compares that value. No engine access.

    Default score() maps score_one(event) -> PerEventVerdict over the batch and reduces
    by mean, if the subclass defines score_one; override score() for other reductions."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        one = getattr(self, "score_one", None)
        if one is None:
            raise NotImplementedError(
                f"{type(self).__name__} must define score_one(event) or override score(events)"
            )
        pe = {ev.request_id: one(ev) for ev in events}
        vals = [pv.score if pv.score is not None else float(pv.passed) for pv in pe.values()]
        return MetricResult(value=sum(vals) / len(vals), per_event=pe)


def as_rollout(ev: TraceEvent, output: Envelope) -> TraceEvent:
    """Present a reference output as a rollout event so a RolloutMetric scores it the
    same way it scores the artifact."""
    return TraceEvent(
        request_id=ev.request_id,
        input=ev.input,
        response=ArtifactResponse(output=output),
        ground_truth=ev.ground_truth,
        t_send=ev.t_send,
        t_recv=ev.t_recv,
    )


def verdict_from(r: MetricResult, *, passed: bool, diagnostics: dict) -> Verdict:
    """Assemble the run-level Verdict from a metric's per-event detail + the gate result."""
    return Verdict(
        passed=passed,
        per_event={rid: pv.passed for rid, pv in r.per_event.items()},
        diagnostics={**r.diagnostics, **diagnostics},
        event_diagnostics={rid: pv.diagnostics for rid, pv in r.per_event.items()},
        event_rationale={rid: pv.rationale for rid, pv in r.per_event.items() if pv.rationale is not None},
    )
