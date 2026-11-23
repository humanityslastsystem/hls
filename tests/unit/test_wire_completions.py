"""OpenAICompletionProtocol: the raw-prompt completions codec, unary and streaming."""
import pytest

from hls.core.faults import ArtifactProtocolError
from hls.core.payloads import Envelope, TextPayload
from hls.spec.drivers import message_content, messages_of
from hls.spec.wire import OpenAIChatProtocol, OpenAICompletionProtocol, WireProtocol


def _env(text="<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>", **meta):
    return Envelope(request_id="r1", contents=[TextPayload(text=text)], metadata=meta)


def test_contract_fields_default_to_chat_with_no_pinned_stop_ids():
    assert WireProtocol.conditioning == "chat"
    assert OpenAIChatProtocol().conditioning == "chat"
    assert OpenAIChatProtocol().stop_token_ids is None
    assert OpenAICompletionProtocol().conditioning == "raw"
    assert OpenAICompletionProtocol.emits_token_ids is True


def test_encode_request_sends_a_raw_prompt_and_asks_for_token_ids():
    body = OpenAICompletionProtocol().encode_request(_env("P", max_tokens=8))
    assert body == {"model": "served-model", "prompt": "P", "max_tokens": 8,
                    "return_token_ids": True}


def test_encode_request_puts_pinned_stop_ids_on_the_wire():
    p = OpenAICompletionProtocol(stop_token_ids=(151643, 151662))
    assert p.encode_request(_env("P"))["stop_token_ids"] == [151643, 151662]
    # The chat protocol carries the same field the same way when a task pins it.
    c = OpenAIChatProtocol(stop_token_ids=(7,))
    assert c.encode_request(_env("P"))["stop_token_ids"] == [7]
    assert "stop_token_ids" not in OpenAIChatProtocol().encode_request(_env("P"))


def test_an_empty_stop_token_id_tuple_is_refused():
    """None and a non-empty set are the only meaningful states: an empty tuple would
    leave the engines with no end-of-turn id at all and put `stop_token_ids: []` on the
    wire, while reading as a declaration."""
    for cls in (OpenAICompletionProtocol, OpenAIChatProtocol):
        with pytest.raises(ValueError, match="non-empty tuple"):
            cls(stop_token_ids=())


def test_a_negative_stop_token_id_is_refused():
    """No emitted token can ever equal it, so the reference would silently never
    terminate on it and every reply would report `length`."""
    for cls in (OpenAICompletionProtocol, OpenAIChatProtocol):
        with pytest.raises(ValueError, match="non-negative"):
            cls(stop_token_ids=(-1, 2))


def test_a_repeated_stop_token_id_is_refused():
    """Duplicates change nothing about where a turn ends and are dumped verbatim onto
    the wire, so they can only be an authoring mistake."""
    for cls in (OpenAICompletionProtocol, OpenAIChatProtocol):
        with pytest.raises(ValueError, match="must not repeat"):
            cls(stop_token_ids=(7, 9, 7))


def test_a_bool_stop_token_id_is_refused():
    """`bool` is a subclass of `int`, so pydantic would store `True` as token id 1 and a
    mistyped flag would silently become a terminator."""
    for cls in (OpenAICompletionProtocol, OpenAIChatProtocol):
        with pytest.raises(ValueError, match="not a bool|got a bool"):
            cls(stop_token_ids=(True,))


@pytest.mark.parametrize("ids", [{True, 5}, frozenset({True, 5}), iter([True, 5]),
                                 [True, 5], (True, 5)])
def test_a_bool_survives_no_iterable_spelling_of_the_ids(ids):
    """A set literal and a generator are ordinary ways to write the ids, and a guard
    that only looked at lists and tuples let `True` through as token id 1."""
    with pytest.raises(ValueError, match="got a bool"):
        OpenAICompletionProtocol(stop_token_ids=ids)


def test_a_set_of_ids_still_validates():
    """Materialising the iterable must not refuse the spellings that were always fine."""
    assert set(OpenAICompletionProtocol(stop_token_ids={7, 9}).stop_token_ids) == {7, 9}
    assert OpenAICompletionProtocol(stop_token_ids=iter([7, 9])).stop_token_ids == (7, 9)
    with pytest.raises(ValueError):
        OpenAICompletionProtocol(stop_token_ids=7)


def test_encode_request_takes_the_prompt_from_a_prerendered_single_turn():
    """A session step pre-renders its own single user turn in metadata['messages']
    (the driver stamps the same shape); the raw prompt is that turn's content."""
    env = _env("fallback", messages=[{"role": "user", "content": "STEP"}])
    assert OpenAICompletionProtocol().encode_request(env)["prompt"] == "STEP"


@pytest.mark.parametrize("msgs,prompt", [
    ([{"role": "user", "content": "GOOD"}, {"content": "NOROLE"}], "GOOD"),
    ([{"role": "user", "content": "GOOD"}, {"role": "user"}], "GOOD"),
    ([{"content": "NOROLE"}], ""),
    ([{"role": "user", "content": "GOOD"}, "not a dict"], "GOOD"),
])
def test_the_raw_prompt_is_the_message_the_oracle_teacher_forces(msgs, prompt):
    """Under raw conditioning the oracle replays `messages_of(env)[-1]`, so the wire has
    to pick from that same filtered list. Reading the unfiltered metadata would send the
    malformed trailing entry -- a prompt the reference never saw, scoring every token as
    a deviation."""
    env = _env("fallback", messages=msgs)
    assert OpenAICompletionProtocol().encode_request(env)["prompt"] == prompt
    kept = messages_of(env)
    assert (message_content(kept[-1]) if kept else "") == prompt


def test_encode_request_stream_adds_stream_options():
    body = OpenAICompletionProtocol().encode_request(_env("P"), stream=True)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}


def test_decode_response_reads_text_token_ids_and_usage():
    data = {"object": "text_completion",
            "choices": [{"text": "mid", "index": 0, "finish_reason": "stop",
                         "token_ids": [1, 2, 151662]}],
            "usage": {"completion_tokens": 3}}
    env = OpenAICompletionProtocol().decode_response(data, "r1")
    tp = env.first_of("text")
    assert (tp.text, tp.tokens, tp.finish_reason) == ("mid", [1, 2, 151662], "stop")
    assert env.metadata["completion_tokens"] == 3


def test_decode_response_accepts_an_empty_text_with_one_token():
    """A reply whose first token is a terminator: empty text, one id."""
    env = OpenAICompletionProtocol().decode_response(
        {"choices": [{"text": "", "finish_reason": "stop", "token_ids": [151662]}]}, "r1")
    assert env.first_of("text").tokens == [151662]


@pytest.mark.parametrize("bad", [
    {}, {"choices": []}, {"choices": ["x"]},
    {"choices": [{"text": "a", "token_ids": [1]}]},            # no finish_reason
    {"choices": [{"text": "a", "finish_reason": "stop"}]},      # no token_ids
    {"choices": [{"text": "a", "finish_reason": "stop", "token_ids": "1"}]},
])
def test_decode_response_rejects_malformed_bodies_as_protocol_errors(bad):
    with pytest.raises(ArtifactProtocolError):
        OpenAICompletionProtocol().decode_response(bad, "r1")


def _sse():
    return [
        (1.0, {"choices": [{"text": "mi", "token_ids": [1], "finish_reason": None}]}),
        (1.2, {"choices": [{"text": "d", "token_ids": [2, 151662], "finish_reason": "stop"}]}),
        (1.3, {"choices": [], "usage": {"completion_tokens": 3}}),
    ]


def test_decode_stream_folds_text_chunks_and_times_tokens():
    resp = OpenAICompletionProtocol().decode_stream(_sse(), "r1")
    tp = resp.output.first_of("text")
    assert (tp.text, tp.tokens, tp.finish_reason) == ("mid", [1, 2, 151662], "stop")
    assert resp.output.metadata["completion_tokens"] == 3
    assert resp.t_first_packet == 1.0
    assert resp.packet_timestamps == [1.0, 1.2, 1.2]


def test_decode_stream_text_without_token_ids_raises():
    chunks = [(1.0, {"choices": [{"text": "mi", "finish_reason": "stop"}]})]
    with pytest.raises(ArtifactProtocolError, match="(?i)token_ids"):
        OpenAICompletionProtocol().decode_stream(chunks, "r1")


def test_decode_stream_without_finish_reason_raises():
    chunks = [(1.0, {"choices": [{"text": "mi", "token_ids": [1], "finish_reason": None}]})]
    with pytest.raises(ArtifactProtocolError, match="finish_reason"):
        OpenAICompletionProtocol().decode_stream(chunks, "r1")


def test_decode_stream_reports_the_servers_error_frame():
    chunks = [(1.0, {"choices": [{"text": "mi", "token_ids": [1], "finish_reason": None}]}),
              (1.1, {"error": {"message": "CUDA out of memory"}})]
    with pytest.raises(ArtifactProtocolError, match="CUDA out of memory"):
        OpenAICompletionProtocol().decode_stream(chunks, "r1")


def test_decode_partial_keeps_what_a_broken_stream_sent():
    chunks = [(1.0, {"choices": [{"text": "mi", "token_ids": [1], "finish_reason": None}]}),
              (1.1, {"choices": [{"text": "d", "finish_reason": None}]})]
    partial = OpenAICompletionProtocol().decode_partial(chunks)
    assert (partial.token_ids, partial.text, partial.finish_reason) == ([1], "mi", None)
    assert OpenAICompletionProtocol().decode_partial([]) is None


def test_probe_request_is_completions_shaped_with_the_served_model():
    body = OpenAICompletionProtocol(model="m").probe_request()
    assert body == {"model": "m", "prompt": "_", "max_tokens": 1}


def test_body_schema_request_matches_encode_request():
    sample = Envelope(request_id="example", contents=[TextPayload(text="<prompt>")])
    for p in (OpenAICompletionProtocol(), OpenAICompletionProtocol(stop_token_ids=(1, 2))):
        assert p.body_schema()["request"] == p.encode_request(sample)
    choice = OpenAICompletionProtocol().body_schema()["response"]["choices"][0]
    assert choice["token_ids"] == [1234, 5678] and "text" in choice and "message" not in choice


def test_brief_section_states_raw_prompt_streaming_and_the_terminator_rule():
    p = OpenAICompletionProtocol(stop_token_ids=(151643, 151662))
    text = p.brief_section(path="/v1/completions", streaming=True)
    assert "POST /v1/completions" in text
    assert "verbatim" in text and "chat template" in text
    assert "[DONE]" in text and "text_completion" in text
    assert "151662" in text
    assert "one token, not zero" in text
    assert "empty `token_ids` array is unverifiable" in text
    plain = OpenAICompletionProtocol().brief_section(path="/v1/completions")
    assert "stop_token_ids" not in plain and "SSE" not in plain


def test_chat_protocol_dump_carries_the_new_field_as_null():
    """The field rides every protocol's provenance."""
    from hls.core.poly import dump_spec
    rec = dump_spec(OpenAIChatProtocol())
    assert rec["stop_token_ids"] is None
    assert "conditioning" not in rec        # a ClassVar, not a field
    assert dump_spec(OpenAICompletionProtocol(stop_token_ids=(1,)))["stop_token_ids"] == [1]
