import pytest

from hls.core.capabilities import Generate, TeacherForce
from hls.spec.wire import WireProtocol
from hls.verify import ArtifactRollout, EngineAccessor, Verifier, check_wiring


class _NoTokenIdProtocol(WireProtocol):
    """A WireProtocol that does NOT emit token ids (``emits_token_ids`` stays False on
    the base): a stand-in for a hypothetical non-OpenAI protocol, used to show
    check_wiring rejects a TeacherForce verifier / streaming-timing SLO wired over it
    up front instead of letting every request fail at decode."""

    def encode_request(self, env, *, stream=False):
        return {}

    def decode_response(self, data, request_id):
        raise NotImplementedError

    def body_schema(self):
        return {}

    def brief_section(self, *, path, streaming=False):
        return ""


def _token_id_less_contract(*, transport: str = "http"):
    from hls.spec.contracts import RequestResponseContract

    return RequestResponseContract(
        path="/v1/chat/completions", transport=transport,
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
        protocol=_NoTokenIdProtocol())


class _FakeProxy:
    async def teacher_force(self, items):
        return [[0.0] for _ in items]


def test_engine_accessor_routes_capability_and_artifact():
    proxy = _FakeProxy()
    acc = EngineAccessor(artifact_events=["e0"], cap_to_provider={TeacherForce: proxy.teacher_force})
    assert acc[ArtifactRollout] == ["e0"]
    call = acc[TeacherForce]
    assert callable(call)


def test_engines_rejects_container_of_reference():
    """A list/tuple field holding any Reference is invisible to the scalar
    collector, so engines() fails loud rather than silently dropping it."""
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify import WiringError

    class _Ensemble(Verifier):
        references: list[HFReference]
        requires = frozenset({TeacherForce})

        async def verify(self, engine):
            ...

    v = _Ensemble(references=[HFReference(decode=Greedy())])
    with pytest.raises(WiringError, match="(?i)container|Reference"):
        v.engines()


def test_check_wiring_passes_when_requires_are_provided():
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference

    class _V(Verifier):
        reference: HFReference
        requires = frozenset({TeacherForce})

        async def verify(self, engine):  # not exercised here
            ...

    v = _V(reference=HFReference(decode=Greedy()))
    check_wiring([v], _contract())   # no raise


def test_check_wiring_rejects_teacher_force_over_token_id_less_protocol():
    """A TeacherForce verifier
    teacher-forces the artifact's exact emitted token ids, so a contract whose
    WireProtocol does not emit them (``emits_token_ids=False``) must be rejected up
    front by check_wiring -- not left to fail per-request at decode. The guarantee
    rides on the protocol's ``emits_token_ids`` capability marker."""
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify import WiringError

    class _TF(Verifier):
        reference: HFReference
        requires = frozenset({TeacherForce})

        async def verify(self, engine):
            ...

    v = _TF(reference=HFReference(decode=Greedy()))
    with pytest.raises(WiringError, match="(?i)token.?id"):
        check_wiring([v], _token_id_less_contract())


def test_check_wiring_rejects_missing_capability():
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference

    class _NeedsGen(Verifier):
        reference: HFReference
        requires = frozenset({Generate, TeacherForce})

        async def verify(self, engine):
            ...

    # HFEngine supports {Generate, TeacherForce, StackReport}, so this passes; instead
    # require a capability nothing provides:
    class _Embed(Generate):  # a fake unprovided capability stand-in
        pass

    class _NeedsEmbed(Verifier):
        reference: HFReference
        requires = frozenset({_Embed})

        async def verify(self, engine):
            ...

    v = _NeedsEmbed(reference=HFReference(decode=Greedy()))
    with pytest.raises(Exception, match="(?i)wiring|capab|provide"):
        check_wiring([v], _contract())


def test_check_wiring_rejects_unaccepted_output_modality():
    """A text-only verifier paired with a contract that emits a non-text output
    modality fails loud at authoring (the reintroduced modality-compat check)."""
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference

    class _TextV(Verifier):              # accepts defaults to {"text"}
        reference: HFReference
        requires = frozenset({TeacherForce})

        async def verify(self, engine):
            ...

    v = _TextV(reference=HFReference(decode=Greedy()))
    with pytest.raises(Exception, match="(?i)modalit"):
        check_wiring([v], _contract(output_modalities=frozenset({"image"})))


def test_check_wiring_passes_when_verifier_accepts_output_modality():
    """A verifier whose `accepts` covers the contract's output modalities wires cleanly."""
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference

    class _ImgV(Verifier):
        reference: HFReference
        requires = frozenset({TeacherForce})
        accepts = frozenset({"text", "image"})

        async def verify(self, engine):
            ...

    v = _ImgV(reference=HFReference(decode=Greedy()))
    check_wiring([v], _contract(output_modalities=frozenset({"image"})))   # no raise


def _logprob_contract():
    """A contract that reports per-token logprobs with alternatives -- what
    check_emit_consistency needs on the wire."""
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.wire import OpenAIChatProtocol

    return RequestResponseContract(
        path="/v1/chat/completions", transport="http",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
        protocol=OpenAIChatProtocol(return_logprobs=True, top_logprobs=20))


def test_check_wiring_rejects_emit_consistency_under_a_frequency_penalty():
    """The emitted-token consistency check compares each emitted token to the
    contestant's own reported per-position maximum, and serving engines report the
    raw distribution rather than the penalized one. Under a frequency penalty a
    faithful server's token is legitimately not its reported argmax, so the pairing
    must fail loud at authoring instead of rejecting every honest position."""
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify import WiringError
    from hls.verify.verifiers import TokenFaithfulnessVerifier

    v = TokenFaithfulnessVerifier(
        reference=HFReference(decode=Greedy(frequency_penalty=0.1)),
        check_emit_consistency=True)
    with pytest.raises(WiringError, match="(?i)frequency_penalty"):
        check_wiring([v], _logprob_contract())


def test_check_wiring_allows_emit_consistency_under_plain_greedy():
    """Plain greedy is unaffected: the contestant's reported argmax is the reference's
    own choice, so the check still wires."""
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify.verifiers import TokenFaithfulnessVerifier

    v = TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()),
                                  check_emit_consistency=True)
    check_wiring([v], _logprob_contract())   # no raise


def test_check_wiring_rejects_a_generate_verifier_over_a_sampled_reference():
    """A Generate verifier scores the artifact against the reference's own output as a
    fixed bar, and under Sample that output is a draw: the bar moves between re-grades,
    and a prompt's reply also depends on which prompts shared its batch. The pairing
    must fail at authoring rather than produce a verdict nothing can reproduce. Keyed
    on `requires`, so a verifier that is not ReferenceRelativeVerifier is refused too."""
    from hls.core.decode import Sample
    from hls.reference.references import HFReference
    from hls.verify import LetterMatchMetric, ReferenceRelativeVerifier, WiringError

    v = ReferenceRelativeVerifier(metric=LetterMatchMetric(),
                                  reference=HFReference(decode=Sample(temperature=1.0)))
    with pytest.raises(WiringError, match="(?i)sample|greedy"):
        check_wiring([v], _contract())

    class _FixedBar(Verifier):
        reference: HFReference
        requires = frozenset({Generate})

        async def verify(self, engine):
            ...

    with pytest.raises(WiringError, match="(?i)sample|greedy"):
        check_wiring([_FixedBar(reference=HFReference(decode=Sample()))], _contract())


def test_check_wiring_allows_a_generate_verifier_over_a_greedy_reference():
    """The fixed bar a Generate verifier needs: greedy output is the same answer every
    re-grade, so the pairing wires."""
    from hls.core.decode import Greedy
    from hls.reference.references import HFReference
    from hls.verify import LetterMatchMetric, ReferenceRelativeVerifier

    v = ReferenceRelativeVerifier(metric=LetterMatchMetric(),
                                  reference=HFReference(decode=Greedy()))
    check_wiring([v], _contract())   # no raise


def _contract(*, output_modalities: frozenset[str] = frozenset({"text"}),
              transport: str = "http"):
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.wire import OpenAIChatProtocol

    return RequestResponseContract(
        path="/v1/chat/completions", transport=transport,
        input_modalities=frozenset({"text"}), output_modalities=output_modalities,
        protocol=OpenAIChatProtocol())


# --- SLO -> contract prerequisite (transport only; token ids are always emitted) ---

class _NoopV(Verifier):
    """Minimal verifier that requires no engine, so check_wiring's verifier loop
    passes cleanly and the SLO prerequisite is what's under test."""
    requires = frozenset()

    async def verify(self, engine):
        ...


def test_check_wiring_streaming_slo_requires_sse_transport():
    """A requires_token_timing SLO over a non-streaming (http) contract fails loud: no
    streaming means no t_first_packet / per-token timing at all."""
    from hls.spec.workload import StreamingLatencySLO
    from hls.verify import WiringError

    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    with pytest.raises(WiringError, match="(?i)http-sse"):
        check_wiring([_NoopV()], _contract(transport="http"), slo)


def test_check_wiring_streaming_slo_requires_token_id_protocol():
    """The transport is http-sse, but the protocol does not emit token ids: per-token
    timing keys off those ids, so the SLO could never measure TPOT and must fail loud
    at authoring (the emits_token_ids half of the streaming prereq)."""
    from hls.spec.workload import StreamingLatencySLO
    from hls.verify import WiringError

    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    with pytest.raises(WiringError, match="(?i)token.?id"):
        check_wiring([_NoopV()], _token_id_less_contract(transport="http-sse"), slo)


def test_check_wiring_streaming_slo_passes_when_satisfied():
    """An http-sse contract satisfies the streaming SLO prerequisite (token ids are
    always emitted)."""
    from hls.spec.workload import StreamingLatencySLO

    slo = StreamingLatencySLO(ttft_ms=200, tpot_ms=80)
    check_wiring([_NoopV()], _contract(transport="http-sse"), slo)  # no raise


def test_check_wiring_latency_slo_imposes_no_streaming_prereq():
    """A latency-only SLO (requires_token_timing=False) wires fine over a plain
    http contract -- the streaming prereq must not fire."""
    from hls.spec.workload import NoSLO

    slo = NoSLO()
    check_wiring([_NoopV()], _contract(transport="http"), slo)  # no raise
