import pytest
from pydantic import BaseModel
from hls.core.payloads import (Envelope, PAYLOAD_REGISTRY, Payload, TextPayload, TokenLogprob,
                               TopLogprob, register_payload)


def test_text_payload_in_registry():
    assert PAYLOAD_REGISTRY["text"] is TextPayload


def test_text_payload_round_trip():
    p = TextPayload(text="hello")
    assert p.text == "hello"
    assert p.modality == "text"
    assert p.tokens is None


def test_envelope_modalities():
    env = Envelope(request_id="r1", contents=[TextPayload(text="hi")])
    assert env.modalities() == frozenset({"text"})
    assert env.first_of("text").text == "hi"
    assert env.first_of("audio") is None


def test_envelope_metadata_default():
    env = Envelope(request_id="r1", contents=[])
    assert env.metadata == {}


def test_envelope_json_roundtrip_preserves_subclass_fields():
    # list[Payload] must serialize/deserialize the concrete
    # TextPayload fields (text/tokens), not just the base `modality`.
    env = Envelope(request_id="r1", contents=[TextPayload(text="hi", tokens=[1, 2])])
    dumped = env.model_dump(mode="json")
    assert dumped["contents"][0]["text"] == "hi"
    assert dumped["contents"][0]["tokens"] == [1, 2]
    back = Envelope.model_validate(dumped)
    assert isinstance(back.first_of("text"), TextPayload)
    assert back.first_of("text").text == "hi"
    assert back.first_of("text").tokens == [1, 2]


def test_register_payload_rejects_duplicate():
    with pytest.raises(ValueError):
        @register_payload("text")
        class _Dup(TextPayload):  # type: ignore
            pass


class _Holder(BaseModel):
    one: Payload
    many: list[Payload]


def test_base_payload_field_round_trips_concrete_subclass():
    # A field declared as the BASE Payload type must (de)serialize the concrete
    # subclass with all its fields -- no SerializeAsAny, no field_validator.
    h = _Holder(one=TextPayload(text="a"), many=[TextPayload(text="b", tokens=[1])])
    dumped = h.model_dump(mode="json")
    assert dumped["one"]["text"] == "a"
    assert dumped["many"][0]["tokens"] == [1]
    back = _Holder.model_validate(dumped)
    assert isinstance(back.one, TextPayload)
    assert back.one.text == "a"
    assert isinstance(back.many[0], TextPayload)
    assert back.many[0].tokens == [1]


def test_base_payload_rejects_unknown_modality():
    with pytest.raises(ValueError):
        _Holder.model_validate({"one": {"modality": "nope"}, "many": []})


def test_text_payload_carries_logprobs_round_trip():
    p = TextPayload(
        text="Hi!",
        tokens=[9906, 999],
        logprobs=[
            TokenLogprob(token="Hi", logprob=-0.04, bytes=[72, 105],
                         top_logprobs=[TopLogprob(token="Hi", logprob=-0.04, bytes=[72, 105])]),
            TokenLogprob(token="!", logprob=-1.21, bytes=[33]),
        ],
        finish_reason="stop",
    )
    env = Envelope(request_id="r1", contents=[p])
    dumped = env.model_dump(mode="json")
    assert dumped["contents"][0]["logprobs"][0]["token"] == "Hi"
    assert dumped["contents"][0]["logprobs"][0]["top_logprobs"][0]["logprob"] == -0.04
    back = Envelope.model_validate(dumped)
    tp = back.first_of("text")
    assert isinstance(tp, TextPayload)
    assert tp.logprobs[1].token == "!"
    assert tp.logprobs[1].top_logprobs is None
    assert tp.finish_reason == "stop"


def test_an_alternative_needs_only_its_id_and_logprob():
    """The wire parser keeps an alternative's id and log-probability and drops its
    text, so both text fields have to be optional for a parsed reply to round-trip."""
    alt = TopLogprob(token_id=9906, logprob=-0.04)
    assert (alt.token, alt.bytes) == (None, None)
    assert TopLogprob.model_validate(alt.model_dump(mode="json")) == alt
