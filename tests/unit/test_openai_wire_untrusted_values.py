"""What the OpenAI wire keeps of the values a reply sends, and the bound it puts on the
token ids.

A reply is the contestant's output, so nothing in it may end the run as a harness error;
a malformed reply fails its own request. Each case below would end the run if unchecked:

- The optional ``gpu``, ``usage`` and ``hls_context`` objects go into the envelope's
  metadata. The trace write refuses a value nested 254 deep there.
- A streamed frame's ``finish_reason``, ``token_ids`` entries and text. A frame that fails
  its decode is still recorded through ``decode_partial``, so its values reach the trace
  write, and its text reaches the failure path.
- A ``usage`` that is not an object, and a unary ``message`` or ``logprobs`` of the wrong
  JSON type, raise TypeError or AttributeError if read as the right one.
- A token id outside 64 bits overflows the int64 arrays the report gate packs ids into.
"""

from __future__ import annotations

import pytest

from hls.core.faults import ArtifactProtocolError
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import (ArtifactResponse, FailureReason, RequestAttempt,
                            RequestFailure, Trace, TraceEvent)
from hls.spec.wire import OpenAIChatProtocol, OpenAICompletionProtocol
from hls.verify.report import TOP_LOGPROBS_REQUESTED

#: The shallowest metadata value the trace write refuses, found by bisecting each of the
#: three fields on both decode paths: 253 levels write and 254 raise.
TOO_DEEP = 254

#: The same for a value in a failed request's partial output, which sits one level
#: shallower in the trace: 254 levels write and 255 raise.
TOO_DEEP_IN_PARTIAL = 255

_INPUT = Envelope(request_id="r1", contents=[TextPayload(text="Q?")],
                  metadata={"tenant_id": "t0", "max_tokens": 64})


def _nested(depth: int) -> object:
    value: object = 1
    for _ in range(depth):
        value = {"a": value}
    return value


def _reply(*, stream: bool, **extra) -> dict:
    """A conforming one-token reply, plus ``extra`` top-level fields."""
    choice = {"finish_reason": "stop", "token_ids": [5]}
    choice.update({"delta": {"content": "a"}} if stream else {"message": {"content": "a"}})
    return {"choices": [choice], **extra}


def _decode(reply: dict, *, stream: bool, proto=None) -> ArtifactResponse:
    proto = proto or OpenAIChatProtocol()
    if stream:
        return proto.decode_stream([(0.1, reply)], "r1")
    return ArtifactResponse(output=proto.decode_response(reply, "r1"))


def _trace_json(event: TraceEvent) -> str:
    """The trace write, as the runner does it for trace.json."""
    return Trace(events=[event]).model_dump_json()


def _served(resp: ArtifactResponse) -> TraceEvent:
    return TraceEvent(request_id="r1", input=_INPUT, response=resp, t_send=0.0,
                      t_recv=1.0)


# ---- values beside the tokens ----------------------------------------------

_DEEP = {
    "gpu": {"gpu": _nested(TOO_DEEP)},
    "usage": {"usage": {"completion_tokens": _nested(TOO_DEEP)}},
    "hls_context": {"hls_context": {"max_num_seqs": _nested(TOO_DEEP)}},
}


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("field", list(_DEEP))
def test_a_deeply_nested_value_beside_the_tokens_cannot_stop_the_trace_write(field, stream):
    resp = _decode(_reply(stream=stream, **_DEEP[field]), stream=stream)

    _trace_json(_served(resp))
    meta = resp.output.metadata
    assert "completion_tokens" not in meta and "max_num_seqs" not in meta
    assert meta.get("gpu", {}) == {}


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_the_gpu_snapshot_keeps_its_scalar_entries(stream):
    """The snapshot a server stamps is a flat map of numbers, which is what the
    report-only serving metrics read. A nested entry is dropped, not the snapshot."""
    gpu = {"utilization_pct": 73.0, "memory_used_mb": 41000.0, "name": "H100",
           "busy": True, "fan": None, "per_device": [{"u": 1}], "nested": {"a": 1}}

    meta = _decode(_reply(stream=stream, gpu=gpu), stream=stream).output.metadata

    assert meta["gpu"] == {"utilization_pct": 73.0, "memory_used_mb": 41000.0,
                           "name": "H100", "busy": True, "fan": None}


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("bad", ["8", 8.0, True, None, [8]],
                         ids=["string", "float", "bool", "null", "list"])
def test_usage_counts_are_kept_only_as_integers(bad, stream):
    usage = {"completion_tokens": 2, "prompt_tokens": bad}

    meta = _decode(_reply(stream=stream, usage=usage), stream=stream).output.metadata

    assert meta == {"completion_tokens": 2}


def test_the_stream_keeps_the_last_integer_count_of_each_kind_in_a_fixed_order():
    """Counts can arrive on different frames. Each kind keeps its last integer value; a
    later null does not erase an earlier count. They are written in one fixed order,
    whatever order the frames sent them in."""
    frames = [
        {"choices": [{"delta": {"content": "a"}, "token_ids": [5], "finish_reason": None}],
         "usage": {"total_tokens": 99, "completion_tokens": 3}},
        {"choices": [], "usage": {"completion_tokens": None, "prompt_tokens": 9}},
        {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"total_tokens": 12}},
    ]

    meta = OpenAIChatProtocol().decode_stream(
        [(0.1 * (i + 1), f) for i, f in enumerate(frames)], "r1").output.metadata

    assert list(meta.items()) == [("completion_tokens", 3), ("prompt_tokens", 9),
                                  ("total_tokens", 12)]


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("usage", [["completion_tokens"], "completion_tokens", 7],
                         ids=["list", "string", "number"])
def test_a_usage_that_is_not_an_object_is_ignored(usage, stream):
    """Membership tests pass on a list or a string that names a count, and indexing one
    by a key raises TypeError, which must not escape the decode."""
    meta = _decode(_reply(stream=stream, usage=usage), stream=stream).output.metadata

    assert "completion_tokens" not in meta


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_the_serving_context_keeps_only_scalar_settings(stream):
    ctx = {"max_context_len": 4096, "enable_prefix_caching": True, "max_num_seqs": [1]}

    meta = _decode(_reply(stream=stream, hls_context=ctx), stream=stream).output.metadata

    assert meta["max_context_len"] == 4096
    assert meta["enable_prefix_caching"] is True
    assert "max_num_seqs" not in meta


# ---- what a failed stream leaves in the trace ------------------------------

def _choice(**over) -> dict:
    """The second frame's one choice: the text "b" and the id 6, before ``over``."""
    return {"choices": [{"delta": {"content": "b"}, "token_ids": [6], **over}]}


_BAD_FRAME = {
    "a nested finish_reason": _choice(finish_reason=_nested(TOO_DEEP_IN_PARTIAL)),
    "a nested token_ids entry": _choice(token_ids=[6, _nested(TOO_DEEP_IN_PARTIAL)]),
    "token_ids that are a string": _choice(token_ids="123"),
    "text that is not a string": _choice(delta={"content": ["b"]}),
    "logprobs that do not parse": _choice(logprobs={"content": [{"token": "b",
                                                                 "logprob": -0.1}]}),
    "choices that are an object": {"choices": {"a": 1}},
    "choices that are a number": {"choices": 7},
    "choices that are true": {"choices": True},
}


@pytest.mark.parametrize("label", list(_BAD_FRAME))
def test_a_bad_stream_frame_leaves_a_failure_record_the_trace_can_write(label):
    """The frame fails its request. What the failure record keeps is the reply up to
    the frame before it, nothing of the bad frame itself, so the trace still writes."""
    proto = OpenAIChatProtocol()
    frames = [(0.1, {"choices": [{"delta": {"content": "a"}, "token_ids": [5]}]}),
              (0.2, _BAD_FRAME[label])]

    with pytest.raises(ArtifactProtocolError):
        proto.decode_stream(frames, "r1")
    attempt = RequestAttempt(partial=proto.decode_partial(frames))

    assert (attempt.partial.token_ids, attempt.partial.text,
            attempt.partial.finish_reason) == ([5], "a", None)
    _trace_json(TraceEvent(
        request_id="r1", input=_INPUT, t_send=0.0, error="protocol",
        failure=RequestFailure(reason=FailureReason.PROTOCOL, t_abandoned=1.0,
                               observed=attempt)))


@pytest.mark.parametrize("choices", [{"a": 1}, 7, True, "x", {}, 0, False],
                         ids=["object", "number", "true", "string", "empty object",
                              "zero", "false"])
def test_stream_choices_that_are_not_a_list_fail_their_own_request(choices):
    """Refused even when falsy: a frame's choices are a list or absent."""
    with pytest.raises(ArtifactProtocolError, match="choices is not a list"):
        OpenAIChatProtocol().decode_stream(
            [(0.1, {"choices": [{"delta": {"content": "a"}, "token_ids": [5]}]}),
             (0.2, {"choices": choices}),
             (0.3, {"choices": [{"delta": {}, "finish_reason": "stop"}]})], "r1")


def test_token_ids_sent_as_a_string_are_refused_not_split_into_digits():
    """A string of digits is not a list of ids. Folded as one, it would read as one id per
    character."""
    with pytest.raises(ArtifactProtocolError, match="not a list of integers"):
        OpenAIChatProtocol().decode_stream(
            [(0.1, {"choices": [{"delta": {"content": "a"}, "token_ids": "123",
                                 "finish_reason": "stop"}]})], "r1")


_NOT_A_STRING = {"an object": {"a": 1}, "a list": [1], "an empty list": [],
                 "an empty object": {}, "zero": 0, "false": False}


@pytest.mark.parametrize("value", list(_NOT_A_STRING.values()), ids=list(_NOT_A_STRING))
@pytest.mark.parametrize("proto", [OpenAIChatProtocol(), OpenAICompletionProtocol()],
                         ids=["chat", "completions"])
def test_a_streamed_text_that_is_not_a_string_fails_its_own_request(proto, value):
    """Refused even when it is falsy: a frame's text is a string or null. The frame
    after it would otherwise finish the reply as if nothing were wrong."""
    text = ({"delta": {"content": value}} if isinstance(proto, OpenAIChatProtocol)
            else {"text": value})
    finish = ({"delta": {}} if isinstance(proto, OpenAIChatProtocol) else {"text": ""})
    with pytest.raises(ArtifactProtocolError, match="not a string"):
        proto.decode_stream([(0.1, {"choices": [{**text, "token_ids": [5]}]}),
                             (0.2, {"choices": [{**finish, "finish_reason": "stop"}]})],
                            "r1")


@pytest.mark.parametrize("value", list(_NOT_A_STRING.values()), ids=list(_NOT_A_STRING))
def test_a_streamed_finish_reason_that_is_not_a_string_fails_its_own_request(value):
    with pytest.raises(ArtifactProtocolError, match="finish_reason is not a string"):
        OpenAIChatProtocol().decode_stream(
            [(0.1, {"choices": [{"delta": {"content": "a"}, "token_ids": [5],
                                 "finish_reason": value}]}),
             (0.2, {"choices": [{"delta": {}, "finish_reason": "stop"}]})], "r1")


def test_an_empty_finish_reason_does_not_end_the_stream():
    """Only a non-empty reason marks the reply finished. A stream whose one reason is
    empty never said it finished, so it fails as a truncated stream."""
    with pytest.raises(ArtifactProtocolError, match="without finish_reason"):
        OpenAIChatProtocol().decode_stream(
            [(0.1, {"choices": [{"delta": {"content": "a"}, "token_ids": [5],
                                 "finish_reason": ""}]})], "r1")


# ---- unary replies of the wrong JSON type -----------------------------------

def _entry(**over) -> dict:
    return {"token": "a", "token_id": 5, "logprob": -0.1, **over}


_WRONG_TYPE = {
    "message is a string": {"message": "hi"},
    "message is a list": {"message": ["hi"]},
    "logprobs is a list": {"logprobs": [1]},
    "logprobs is a string": {"logprobs": "x"},
    "a content entry is not an object": {"logprobs": {"content": [1]}},
    "a top_logprobs entry is not an object": {"logprobs": {"content": [
        _entry(top_logprobs=[1])]}},
    "top_logprobs is an object": {"logprobs": {"content": [_entry(top_logprobs={"x": 1})]}},
}


@pytest.mark.parametrize("label", list(_WRONG_TYPE))
def test_a_unary_reply_of_the_wrong_json_type_fails_its_own_request(label):
    """Each would raise AttributeError, which no fault class covers. The logprobs are parsed
    whether or not the request asked for them, so a protocol that asks for none is
    exposed to the same shapes."""
    reply = {"choices": [{"finish_reason": "stop", "token_ids": [5],
                          "message": {"content": "a"}, **_WRONG_TYPE[label]}]}
    for proto in (OpenAIChatProtocol(),
                  OpenAIChatProtocol(return_logprobs=True,
                                     top_logprobs=TOP_LOGPROBS_REQUESTED)):
        with pytest.raises(ArtifactProtocolError):
            proto.decode_response(reply, "r1")


# ---- the token ids ---------------------------------------------------------

#: Alternative ids that are neither the served token nor out of range.
_ALTS = [100 + k for k in range(TOP_LOGPROBS_REQUESTED - 1)]


def _reported(tid: int, alt_ids: list[int]) -> dict:
    """One logprob entry for token ``tid``, whose alternatives carry ``alt_ids``."""
    return {"content": [_entry(token_id=tid, top_logprobs=[
        {"token": "b", "token_id": a, "logprob": -0.1 if a == tid else -9.0}
        for a in alt_ids])]}


def _reporting():
    return OpenAIChatProtocol(return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED)


def _with_ids(stream: bool, *, tokens, logprobs=None) -> dict:
    reply = _reply(stream=stream)
    reply["choices"][0]["token_ids"] = tokens
    if logprobs is not None:
        reply["choices"][0]["logprobs"] = logprobs
    return reply


_TOP, _BOTTOM = 2**63 - 1, -2**63


def _cases(bad: int) -> dict:
    """Where one out-of-range id can sit, each time after an id that is in range."""
    return {
        "token_ids": dict(tokens=[5, bad]),
        "a logprob entry's token_id": dict(tokens=[5], logprobs=_reported(bad, [5, *_ALTS])),
        "an alternative's token_id": dict(tokens=[5],
                                          logprobs=_reported(5, [5, *_ALTS[:-1], bad])),
    }


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("where", list(_cases(0)))
@pytest.mark.parametrize("bad", [_TOP + 1, _BOTTOM - 1], ids=["2**63", "-2**63-1"])
def test_a_token_id_outside_64_bits_fails_its_own_request(bad, where, stream):
    """The report gate packs served and reported ids into int64 arrays, so an id outside
    64 bits that decoded would raise OverflowError in verify, which nothing catches."""
    with pytest.raises(ArtifactProtocolError, match="outside 64 bits"):
        _decode(_with_ids(stream, **_cases(bad)[where]), stream=stream, proto=_reporting())


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("edge", [_TOP, _BOTTOM], ids=["2**63-1", "-2**63"])
def test_a_token_id_at_either_end_of_64_bits_is_accepted(edge, stream):
    """The bound is the int64 range and nothing narrower. Whether a negative id can be
    a token is the gate's question, which it answers by scoring it."""
    resp = _decode(_with_ids(stream, tokens=[5, edge],
                             logprobs=_reported(edge, [edge, *_ALTS])),
                   stream=stream, proto=_reporting())

    tp = resp.output.first_of("text")
    assert tp.tokens == [5, edge]
    assert tp.logprobs[0].token_id == edge


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_token_ids_read_as_the_payload_reads_them(stream):
    """A numeric string or an integral float reads as the integer it names. The bound
    applies to that integer."""
    resp = _decode(_with_ids(stream, tokens=["5", 6.0]), stream=stream)
    assert resp.output.first_of("text").tokens == [5, 6]

    with pytest.raises(ArtifactProtocolError, match="outside 64 bits"):
        _decode(_with_ids(stream, tokens=["5", str(_TOP + 1)]), stream=stream)


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_the_completions_wire_bounds_its_token_ids_too(stream):
    reply = {"choices": [{"finish_reason": "stop", "text": "a", "token_ids": [5, _TOP + 1]}]}
    with pytest.raises(ArtifactProtocolError, match="outside 64 bits"):
        _decode(reply, stream=stream, proto=OpenAICompletionProtocol())
