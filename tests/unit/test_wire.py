import json

import pytest
from pydantic import ValidationError

from hls.core.faults import ArtifactProtocolError
from hls.core.payloads import Envelope, TextPayload
from hls.spec.wire import OpenAIChatProtocol, OpenAICompletionProtocol


def _env(text="Hi"):
    return Envelope(request_id="r1", contents=[TextPayload(text=text)])


def test_encode_request_always_asks_for_token_ids():
    # token_ids is mandatory on the response, and stock vLLM emits choices[].token_ids ONLY
    # when the request asks per-request -- so return_token_ids rides on EVERY request body,
    # unconditionally, even with all toggles off.
    body = OpenAIChatProtocol().encode_request(_env("Hi"))
    assert body == {"model": "served-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "return_token_ids": True}


def test_encode_request_emits_logprobs_flags_only_when_toggled():
    p = OpenAIChatProtocol(return_logprobs=True, top_logprobs=5)
    body = p.encode_request(_env("Hi"))
    # Exact key set: the logprobs toggles plus the always-on return_token_ids ask.
    assert set(body.keys()) == {"model", "messages", "logprobs", "top_logprobs",
                                "return_token_ids"}
    assert body["logprobs"] is True
    assert body["top_logprobs"] == 5


def test_encode_request_stream_adds_stream_options():
    body = OpenAIChatProtocol().encode_request(_env("Hi"), stream=True)
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["return_token_ids"] is True          # the streaming request asks too


def test_streaming_request_field_is_what_makes_a_stock_vllm_stream_conformant():
    """The streaming request carries return_token_ids
    (stock vLLM emits choices[].token_ids only when asked per-request), and a stream WITH
    token_ids decodes while the same stream WITHOUT faults on the mandatory check -- so the
    request field is exactly what separates a conformant exchange from the fault."""
    p = OpenAIChatProtocol()
    assert p.encode_request(_env("Hi"), stream=True)["return_token_ids"] is True
    with_ids = [(0.0, {"choices": [{"delta": {"content": "hello"}, "token_ids": [1, 2],
                                    "finish_reason": "stop"}]})]
    assert p.decode_stream(with_ids, "r1").output.first_of("text").tokens == [1, 2]
    without = [(0.0, {"choices": [{"delta": {"content": "hello"},
                                   "finish_reason": "stop"}]})]
    with pytest.raises(ArtifactProtocolError, match="without token_ids"):
        p.decode_stream(without, "r1")


def test_encode_request_forwards_max_tokens_from_metadata():
    # The driver stamps the workload's per-request decoding budget on the envelope
    # metadata; the wire protocol must forward it so the artifact and oracle run
    # under identical caps. See OpenAIChatProtocol.encode_request.
    env = Envelope(request_id="r1", contents=[TextPayload(text="Hi")],
                   metadata={"max_tokens": 4})
    body = OpenAIChatProtocol().encode_request(env)
    assert body["max_tokens"] == 4


def test_encode_request_forwards_tools_from_metadata():
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    env = Envelope(request_id="r1", contents=[TextPayload(text="Hi")],
                   metadata={"tools": tools})
    body = OpenAIChatProtocol().encode_request(env)
    assert body["tools"] == tools


def test_encode_request_omits_max_tokens_when_metadata_absent_or_invalid():
    # No metadata: no max_tokens key.
    body = OpenAIChatProtocol().encode_request(_env("Hi"))
    assert "max_tokens" not in body
    # Non-positive / wrong type: silently dropped.
    for bad in (0, -1, None, "4"):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Hi")],
                       metadata={"max_tokens": bad})
        assert "max_tokens" not in OpenAIChatProtocol().encode_request(env)


def test_decode_response_parses_content_tokens_logprobs_and_usage():
    p = OpenAIChatProtocol(return_logprobs=True)
    data = {
        "id": "chatcmpl-abc",
        "choices": [{
            "message": {"role": "assistant", "content": "Hi!"},
            "finish_reason": "stop",
            "token_ids": [9906, 999],
            "logprobs": {"content": [
                {"token": "Hi", "token_id": 9906, "logprob": -0.04, "bytes": [72, 105],
                 "top_logprobs": [{"token": "Hi", "token_id": 9906, "logprob": -0.04, "bytes": [72, 105]}]},
                {"token": "!", "token_id": 999, "logprob": -1.21, "bytes": [33]},
            ]},
        }],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
        "hls_context": {
            "max_context_len": 131072,
            "request_max_tokens": 2048,
            "enable_prefix_caching": True,
            "enforce_eager": False,
            "max_num_seqs": 32,
            "tensor_parallel_size": 2,
        },
    }
    env = p.decode_response(data, request_id="r1")
    tp = env.first_of("text")
    assert tp.tokens == [9906, 999]
    assert tp.finish_reason == "stop"
    assert tp.logprobs[0].token_id == 9906
    assert tp.logprobs[0].top_logprobs[0].token_id == 9906
    assert env.metadata["prompt_tokens"] == 8
    assert env.metadata["completion_tokens"] == 2
    assert env.metadata["total_tokens"] == 10
    assert env.metadata["max_context_len"] == 131072
    assert env.metadata["request_max_tokens"] == 2048
    assert env.metadata["enable_prefix_caching"] is True
    assert env.metadata["enforce_eager"] is False
    assert env.metadata["max_num_seqs"] == 32
    assert env.metadata["tensor_parallel_size"] == 2


def test_decode_response_passes_through_gpu_telemetry():
    # Optional, non-standard per-response GPU snapshot rides into metadata so a
    # report-only serving metric can reduce mean/peak GPU util.
    data = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop",
                         "token_ids": [1, 2]}],
            "gpu": {"utilization_pct": 73.0, "memory_used_mb": 41000.0}}
    env = OpenAIChatProtocol().decode_response(data, request_id="r1")
    assert env.metadata["gpu"] == {"utilization_pct": 73.0, "memory_used_mb": 41000.0}


def test_decode_response_ignores_non_dict_gpu():
    data = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop",
                         "token_ids": [1, 2]}],
            "gpu": "n/a"}
    env = OpenAIChatProtocol().decode_response(data, request_id="r1")
    assert "gpu" not in env.metadata


def test_decode_stream_passes_through_gpu_on_final_chunk():
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906],
                            "finish_reason": "stop"}]}),
        (1.1, {"choices": [], "usage": {"prompt_tokens": 8,
                                         "completion_tokens": 1,
                                         "total_tokens": 9},
               "hls_context": {"max_context_len": 32768,
                               "request_max_tokens": 256,
                               "enable_prefix_caching": True,
                               "enforce_eager": False,
                               "max_num_seqs": 64,
                               "tensor_parallel_size": 4},
               "gpu": {"utilization_pct": 80.0, "memory_used_mb": 5.0}}),
    ]
    resp = OpenAIChatProtocol().decode_stream(chunks, "r1")
    assert resp.output.metadata["prompt_tokens"] == 8
    assert resp.output.metadata["completion_tokens"] == 1
    assert resp.output.metadata["total_tokens"] == 9
    assert resp.output.metadata["max_context_len"] == 32768
    assert resp.output.metadata["request_max_tokens"] == 256
    assert resp.output.metadata["enable_prefix_caching"] is True
    assert resp.output.metadata["enforce_eager"] is False
    assert resp.output.metadata["max_num_seqs"] == 64
    assert resp.output.metadata["tensor_parallel_size"] == 4
    assert resp.output.metadata["gpu"] == {"utilization_pct": 80.0, "memory_used_mb": 5.0}


def test_decode_response_requires_token_ids():
    # token_ids is mandatory now -- a body without it hard-fails.
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_response(
            {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]},
            request_id="r1")


def test_decode_response_minimal_with_token_ids_decodes():
    env = OpenAIChatProtocol().decode_response(
        {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop",
                      "token_ids": [1, 2]}]},
        request_id="r1")
    tp = env.first_of("text")
    assert tp.text == "ok"
    assert tp.tokens == [1, 2]
    assert tp.logprobs is None


def test_decode_response_logprob_missing_token_id_raises():
    p = OpenAIChatProtocol(return_logprobs=True)
    bad = {"choices": [{"message": {"content": "B"}, "finish_reason": "stop",
                        "token_ids": [66],
                        "logprobs": {"content": [{"token": "B", "logprob": -0.1}]}}]}
    with pytest.raises(ArtifactProtocolError):
        p.decode_response(bad, request_id="r1")


def test_top_logprobs_requires_return_logprobs():
    with pytest.raises(ValueError):
        OpenAIChatProtocol(top_logprobs=5)               # return_logprobs is False


def test_top_logprobs_range():
    """21 is in range: the report contract asks for that many alternatives, and the
    serving stacks allow it even though OpenAI's own API stops at 20."""
    assert OpenAIChatProtocol(return_logprobs=True, top_logprobs=21).top_logprobs == 21
    with pytest.raises(ValueError):
        OpenAIChatProtocol(return_logprobs=True, top_logprobs=33)
    with pytest.raises(ValueError):
        OpenAIChatProtocol(return_logprobs=True, top_logprobs=-1)


def test_alternatives_keep_only_their_id_and_logprob():
    """The parser drops each alternative's decoded text and bytes. A reply reports up
    to 21 alternatives per generated token and the trace is dumped whole, so the text
    would cost gigabytes for a field verification never reads."""
    p = OpenAIChatProtocol(return_logprobs=True, top_logprobs=2)
    data = {
        "choices": [{
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
            "token_ids": [9906],
            "logprobs": {"content": [
                {"token": "Hi", "token_id": 9906, "logprob": -0.04, "bytes": [72, 105],
                 "top_logprobs": [
                     {"token": "Hi", "token_id": 9906, "logprob": -0.04, "bytes": [72, 105]},
                     {"token": "Yo", "token_id": 1234, "logprob": -2.5},
                 ]},
            ]},
        }],
    }
    entry = p.decode_response(data, request_id="r1").first_of("text").logprobs[0]

    assert (entry.token, entry.bytes) == ("Hi", [72, 105])   # the served token keeps its text
    assert [(t.token_id, t.logprob) for t in entry.top_logprobs] == [(9906, -0.04), (1234, -2.5)]
    assert all(t.token is None and t.bytes is None for t in entry.top_logprobs)


def test_a_stream_reassembles_one_report_entry_per_token():
    """The shape the sampling starters emit: each content chunk carries its own tokens
    and the logprob entries for exactly those tokens, so the reassembled reply has one
    entry per served token, in order."""
    p = OpenAIChatProtocol(return_logprobs=True, top_logprobs=1)

    def chunk(text, token_id, finish=None):
        entry = {"token": text, "token_id": token_id, "logprob": -0.5,
                 "top_logprobs": [{"token_id": token_id, "logprob": -0.5}]}
        return {"choices": [{"index": 0, "delta": {"content": text},
                             "token_ids": [token_id],
                             "logprobs": {"content": [entry]},
                             "finish_reason": finish}]}

    resp = p.decode_stream(
        [(0.0, chunk("Hi", 9906)), (0.1, chunk("!", 999)),
         (0.2, {"choices": [{"index": 0, "delta": {}, "token_ids": None,
                             "finish_reason": "stop"}]})],
        request_id="r1")

    tp = resp.output.first_of("text")
    assert tp.tokens == [9906, 999]
    assert [e.token_id for e in tp.logprobs] == [9906, 999]
    assert [e.top_logprobs[0].token_id for e in tp.logprobs] == [9906, 999]


def test_an_alternative_without_a_decoded_token_still_parses():
    """A server may send an alternative as an id and a log-probability alone; the
    brief makes only the id mandatory there."""
    p = OpenAIChatProtocol(return_logprobs=True, top_logprobs=1)
    data = {
        "choices": [{
            "message": {"role": "assistant", "content": "Hi"},
            "finish_reason": "stop",
            "token_ids": [9906],
            "logprobs": {"content": [
                {"token": "Hi", "token_id": 9906, "logprob": -0.04,
                 "top_logprobs": [{"token_id": 9906, "logprob": -0.04}]},
            ]},
        }],
    }
    [alt] = p.decode_response(data, request_id="r1").first_of("text").logprobs[0].top_logprobs
    assert (alt.token_id, alt.logprob) == (9906, -0.04)


def test_body_schema_request_matches_encode_request_across_toggles():
    # Drift guard: the authored example request must equal the real codec output.
    sample = Envelope(request_id="example", contents=[TextPayload(text="<prompt>")])
    combos = [
        OpenAIChatProtocol(),
        OpenAIChatProtocol(return_logprobs=True),
        OpenAIChatProtocol(return_logprobs=True, top_logprobs=5),
    ]
    for p in combos:
        assert p.body_schema()["request"] == p.encode_request(sample)


def test_body_schema_response_carries_token_ids_and_logprob_token_id():
    schema = OpenAIChatProtocol(return_logprobs=True, top_logprobs=5).body_schema()
    choice = schema["response"]["choices"][0]
    assert choice["token_ids"] == [1234, 5678]
    entry = choice["logprobs"]["content"][0]
    assert entry["token_id"] == 1234
    assert entry["top_logprobs"][0]["token_id"] == 1234


def test_a_sample_that_shows_every_alternative_is_not_annotated():
    # Nothing to reconcile when the sample is complete, so the line stays plain.
    section = OpenAIChatProtocol(return_logprobs=True, top_logprobs=2).brief_section(
        path="/v1/chat/completions")
    assert "\nResponse:\n" in section
    assert "abbreviated" not in section


def test_probe_request_is_minimal_openai_body_with_the_served_model():
    # D14: the readiness probe body comes from the contract's WireProtocol (a minimal OpenAI
    # chat body with max_tokens=1). The model is the SERVED model name (same as the drive),
    # so stock vLLM answers 200 rather than a model-mismatch 404 the readiness gate rejects.
    body = OpenAIChatProtocol(model="served-model").probe_request()
    assert body["max_tokens"] == 1
    assert body["messages"] == [{"role": "user", "content": "_"}]
    assert body["model"] == "served-model"
    assert OpenAIChatProtocol(model="my-model").probe_request()["model"] == "my-model"


def test_brief_section_renders_endpoint_and_conditional_prose():
    p = OpenAIChatProtocol(return_logprobs=True, top_logprobs=5)
    text = p.brief_section(path="/v1/chat/completions", streaming=False)
    assert "POST /v1/chat/completions" in text
    assert "token_ids" in text
    assert "token_id" in text
    assert "top 5" in text
    assert "fails verification" in text
    assert "SSE" not in text


def test_brief_section_always_requires_token_ids_but_omits_logprobs_when_off():
    text = OpenAIChatProtocol().brief_section(path="/v1/chat/completions")
    assert "token_ids" in text                 # mandatory, always present
    assert "logprobs" not in text              # feature off


def test_brief_section_includes_streaming_when_enabled():
    text = OpenAIChatProtocol().brief_section(
        path="/v1/chat/completions", streaming=True)
    assert "SSE" in text or "stream" in text
    assert "[DONE]" in text


def test_brief_section_explains_pinned_stop_token_ids():
    """A pinned set rides every chat request body and appears in the brief's example
    JSON, so the brief has to state the rules that come with it: what ends generation,
    that the terminator counts as an emitted token, and that an empty token_ids array
    fails."""
    text = OpenAIChatProtocol(stop_token_ids=(151643, 151662)).brief_section(
        path="/v1/chat/completions")
    assert "stop_token_ids" in text and "151662" in text
    assert "generation ends when the" in text
    assert "one token, not zero" in text
    assert "empty `token_ids` array is unverifiable" in text


def test_brief_section_says_nothing_about_stop_ids_when_none_are_pinned():
    text = OpenAIChatProtocol().brief_section(path="/v1/chat/completions")
    assert "stop_token_ids" not in text


def test_the_logprobs_hook_is_an_instance_method_on_base_and_override():
    """``decode_response`` and ``_fold_stream`` call ``self._parse_logprobs``. The chat
    override must not be a staticmethod, which resolves only through the descriptor: a
    ``super()`` call from a further subclass would pass the logprobs dict as ``self``."""
    import inspect

    from hls.spec.wire import _OpenAIProtocol

    class _Sub(OpenAIChatProtocol):
        def _parse_logprobs(self, lp):
            return super()._parse_logprobs(lp)

    parsed = _Sub(return_logprobs=True)._parse_logprobs(
        {"content": [{"token": "a", "token_id": 5, "logprob": -0.5}]})
    assert parsed is not None and parsed[0].token_id == 5
    assert (inspect.signature(_OpenAIProtocol._parse_logprobs)
            == inspect.signature(OpenAIChatProtocol._parse_logprobs))


def _sse_chunks():
    return [
        (1.0, {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
        (1.1, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906],
                            "logprobs": {"content": [{"token": "Hi", "token_id": 9906, "logprob": -0.04, "bytes": [72, 105]}]},
                            "finish_reason": None}]}),
        (1.2, {"choices": [{"delta": {"content": "!"}, "token_ids": [999],
                            "logprobs": {"content": [{"token": "!", "token_id": 999, "logprob": -1.21, "bytes": [33]}]},
                            "finish_reason": "stop"}]}),
        (1.3, {"choices": [], "usage": {"completion_tokens": 2}}),
    ]


def test_decode_stream_reassembles_to_same_envelope_as_unary():
    p = OpenAIChatProtocol(return_logprobs=True)
    resp = p.decode_stream(_sse_chunks(), request_id="r1")
    tp = resp.output.first_of("text")
    assert tp.text == "Hi!"
    assert tp.tokens == [9906, 999]
    assert [lp.token_id for lp in tp.logprobs] == [9906, 999]
    assert tp.finish_reason == "stop"
    assert resp.output.metadata["completion_tokens"] == 2


def test_decode_stream_content_without_token_ids_raises():
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}),
        (1.1, {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ]
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_stream(chunks, "r1")


def test_decode_stream_partial_token_ids_raises():
    # token_ids present on the FIRST content chunk but DROPPED on a later content
    # chunk. An aggregate guard ("some chunk carried ids") would let this through,
    # decoding text "Hi!" with tokens=[9906] -- a tokens list shorter than /
    # misaligned with the emitted text that the token-level verifier would silently
    # mis-score. The per-chunk mandate must reject the exact offending chunk instead.
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906],
                            "finish_reason": None}]}),
        (1.1, {"choices": [{"delta": {"content": "!"},  # content, but no token_ids
                            "finish_reason": None}]}),
        (1.2, {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ]
    with pytest.raises(ArtifactProtocolError, match="(?i)token_ids"):
        OpenAIChatProtocol().decode_stream(chunks, "r1")


def test_decode_stream_empty_token_ids_on_content_chunk_raises():
    # An empty token_ids array on a content-bearing delta is as non-conformant as an
    # absent one -- the content it carries has no ids to attribute.
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [],
                            "finish_reason": "stop"}]}),
    ]
    with pytest.raises(ArtifactProtocolError, match="(?i)token_ids"):
        OpenAIChatProtocol().decode_stream(chunks, "r1")


def test_decode_stream_captures_timing():
    resp = OpenAIChatProtocol().decode_stream(_sse_chunks(), "r1")
    assert resp.t_first_packet == 1.1                    # first content chunk
    assert resp.packet_timestamps == [1.1, 1.2]          # one per emitted token


def test_decode_response_missing_choices_raises_protocol_error():
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_response({}, request_id="r1")
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_response({"choices": []}, request_id="r1")


def test_decode_response_missing_finish_reason_raises_protocol_error():
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_response(
            {"choices": [{"message": {"content": "B"}}]}, request_id="r1")


def test_decode_response_malformed_logprobs_raises_protocol_error():
    # token_ids present (bypasses the mandatory check) and the logprob entry has a
    # valid token_id, but is missing the required 'logprob' key -- must surface as
    # ArtifactProtocolError, not a raw KeyError/ValidationError.
    p = OpenAIChatProtocol(return_logprobs=True)
    bad = {"choices": [{"message": {"content": "B"}, "finish_reason": "stop",
                        "token_ids": [66],
                        "logprobs": {"content": [{"token": "B", "token_id": 66}]}}]}  # missing 'logprob'
    with pytest.raises(ArtifactProtocolError):
        p.decode_response(bad, request_id="r1")


def test_decode_stream_truncated_without_finish_reason_raises():
    chunks = [(1.0, {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]})]
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_stream(chunks, "r1")


def test_decode_stream_empty_raises():
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_stream([], "r1")


def test_decode_stream_clean_still_decodes():
    resp = OpenAIChatProtocol().decode_stream(_sse_chunks(), "r1")
    assert resp.output.first_of("text").finish_reason == "stop"


def test_decode_stream_malformed_logprobs_raises_protocol_error():
    # A well-terminated stream whose logprobs carry a valid token_id but are missing
    # the required "logprob" key -- _parse_logprobs raises KeyError on e["logprob"],
    # which must become ArtifactProtocolError. Mirrors
    # test_decode_response_malformed_logprobs_raises_protocol_error for the stream path.
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906], "finish_reason": None,
                            "logprobs": {"content": [{"token": "Hi", "token_id": 9906}]}}]}),
        (1.1, {"choices": [{"delta": {}, "finish_reason": "stop",
                            "logprobs": None}]}),
    ]
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol(return_logprobs=True).decode_stream(chunks, "r1")


def test_decode_stream_non_dict_choice_raises_protocol_error():
    # A stream where choices[0] is a string instead of a dict -- ch.get(...) raises AttributeError.
    # Must be converted to ArtifactProtocolError, not allowed to escape raw.
    chunks = [
        (1.0, {"choices": ["not-a-dict"]}),
        (1.1, {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
    ]
    with pytest.raises(ArtifactProtocolError):
        OpenAIChatProtocol().decode_stream(chunks, "r1")


def test_decode_stream_reports_the_servers_own_error_message():
    """A server dying mid-stream emits {"error": ...} then [DONE] on the already-committed
    200. That frame has no choices, so skipping it would report the fault as
    "truncated or empty" and discard the only account of the cause."""
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"},
                            "token_ids": [9906], "finish_reason": None}]}),
        (1.1, {"error": {"message": "torch.OutOfMemoryError: CUDA out of memory"}}),
    ]
    p = OpenAIChatProtocol()
    with pytest.raises(ArtifactProtocolError, match="CUDA out of memory"):
        p.decode_stream(chunks, request_id="r1")


def test_decode_stream_keeps_the_generic_message_when_no_error_frame():
    """A genuinely truncated stream with nothing to report keeps the original wording."""
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"},
                            "token_ids": [9906], "finish_reason": None}]}),
    ]
    p = OpenAIChatProtocol()
    with pytest.raises(ArtifactProtocolError, match="truncated or empty"):
        p.decode_stream(chunks, request_id="r1")


# ---------------------------------------------------------------------------
# decode_partial: what a stream that never completed managed to send
# ---------------------------------------------------------------------------


def test_decode_partial_reports_the_tokens_a_broken_stream_sent():
    """The record a failed request leaves behind. decode_stream refuses this stream (no
    finish_reason), and the tokens it did carry are the difference between "the server
    answered nothing" and "the server was 2 tokens in"."""
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"},
                            "token_ids": [9906], "finish_reason": None}]}),
        (1.1, {"choices": [{"delta": {"content": "!"},
                            "token_ids": [999], "finish_reason": None}]}),
    ]
    partial = OpenAIChatProtocol().decode_partial(chunks)
    assert partial.token_ids == [9906, 999]
    assert partial.text == "Hi!"
    assert partial.finish_reason is None


def test_decode_partial_of_an_empty_stream_is_none():
    """A server that sent nothing back has no partial output -- reported as absent
    rather than as an empty reply, which is a different observation."""
    assert OpenAIChatProtocol().decode_partial([]) is None


def test_decode_partial_keeps_everything_before_a_non_conformant_frame():
    """The fold raises on the second frame (content with no token_ids). A diagnostic
    record must still show the first frame's tokens: the point of the record is how far
    the request got, and the fault string already names the violation."""
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"},
                            "token_ids": [9906], "finish_reason": None}]}),
        (1.1, {"choices": [{"delta": {"content": "!"}, "finish_reason": None}]}),
    ]
    partial = OpenAIChatProtocol().decode_partial(chunks)
    assert partial.token_ids == [9906]
    assert partial.text == "Hi"


def test_decode_partial_reports_a_finish_reason_the_stream_did_send():
    """A stream that finished its reply and then broke before [DONE] is a different
    fault from one that never finished, so the reported finish_reason is kept."""
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"},
                            "token_ids": [9906], "finish_reason": "stop"}]}),
    ]
    assert OpenAIChatProtocol().decode_partial(chunks).finish_reason == "stop"


def test_decode_partial_never_raises_on_a_stream_that_is_garbage_throughout():
    """It runs while a failure is already being recorded; raising here would replace
    the fault under diagnosis with one about the diagnosis."""
    chunks = [(1.0, {"choices": "not-a-list-of-objects"}), (1.1, 5)]
    assert OpenAIChatProtocol().decode_partial(chunks) is None


def test_decode_partial_reports_the_logprobs_a_broken_stream_sent():
    """The report is what the token-level gate grades, so a partial reply that is scored (a
    request the scheduled stop cut) needs its per-position report, not only its ids."""
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906],
                            "logprobs": {"content": [{"token": "Hi", "token_id": 9906, "logprob": -0.5}]},
                            "finish_reason": None}]}),
        (1.1, {"choices": [{"delta": {"content": "!"}, "token_ids": [999],
                            "logprobs": {"content": [{"token": "!", "token_id": 999, "logprob": -1.5}]},
                            "finish_reason": None}]}),
    ]
    partial = OpenAIChatProtocol(return_logprobs=True).decode_partial(chunks)
    assert [lp.token_id for lp in partial.logprobs] == [9906, 999]
    assert [lp.logprob for lp in partial.logprobs] == [-0.5, -1.5]


def test_decode_partial_reports_one_arrival_time_per_token():
    """A cut request is scored on the rate it was streaming at, which needs the arrival time
    of each token rather than of each frame: one frame can carry several."""
    chunks = [
        (1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906], "finish_reason": None}]}),
        (1.5, {"choices": [{"delta": {"content": " there"}, "token_ids": [1, 2], "finish_reason": None}]}),
    ]
    partial = OpenAIChatProtocol().decode_partial(chunks)
    assert partial.packet_timestamps == [1.0, 1.5, 1.5]


def test_decode_partial_carries_no_logprobs_when_the_contract_asked_for_none():
    chunks = [(1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906], "finish_reason": None}]})]
    assert OpenAIChatProtocol().decode_partial(chunks).logprobs is None


def test_a_failed_events_record_of_a_partial_is_its_ids_text_and_finish_reason():
    """The report and the arrival times are read by the cut alone, which records them on the
    event's response. A failed event's record of its partial stays the ids, the text and the
    finish reason, the same for every task whether or not it declares a scheduled stop."""
    from hls.core.trace import FailureReason, RequestAttempt, RequestFailure, TraceEvent

    chunks = [(1.0, {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906],
                                  "logprobs": {"content": [{"token": "Hi", "token_id": 9906, "logprob": -0.5}]},
                                  "finish_reason": None}]})]
    partial = OpenAIChatProtocol(return_logprobs=True).decode_partial(chunks)
    ev = TraceEvent(request_id="r", input=Envelope(request_id="r"), t_send=0.0, error="transport: x",
                    failure=RequestFailure(reason=FailureReason.TRANSPORT, t_abandoned=2.0,
                                           observed=RequestAttempt(n_frames=1, partial=partial)))
    assert ev.model_dump(mode="python")["failure"]["observed"]["partial"] == {
        "token_ids": [9906], "text": "Hi", "finish_reason": None}


def test_decode_partial_defaults_to_none_for_a_protocol_that_does_not_stream():
    from hls.spec.wire import WireProtocol

    class _Unary(WireProtocol):
        def encode_request(self, env, *, stream=False): return {}
        def decode_response(self, data, request_id): raise NotImplementedError
        def body_schema(self): return {}
        def brief_section(self, *, path, streaming=False): return ""

    assert _Unary().decode_partial([(1.0, {"choices": []})]) is None


def test_decode_stream_rejects_a_frame_that_is_not_an_object():
    """`data: 5` is valid JSON and must not reach `chunk.get` as an int, raising a bare
    AttributeError that escapes the fault taxonomy: one malformed frame would invalidate
    the whole run instead of faulting the one request."""
    with pytest.raises(ArtifactProtocolError, match="not an object"):
        OpenAIChatProtocol().decode_stream([(1.0, 5)], "r1")


#: A tool-calling turn as an agentic task's framework leaves it in the history: the
#: model's own structured call, its reasoning, and the result in the tool role.
_TOOL_TURN = [
    {"role": "user", "content": "who founded it"},
    {"role": "assistant", "content": "", "reasoning_content": "I need evidence.",
     "reasoning": "I need evidence.",
     "tool_calls": [{"id": "call_0", "type": "function",
                     "function": {"name": "web_search",
                                  "arguments": {"query": "founder"}}}]},
    {"role": "tool", "tool_call_id": "call_0", "name": "web_search",
     "content": '{"documents": []}'},
]


def test_encode_request_passes_through_the_chat_fields_a_template_reads():
    """A chat template renders tool calls, tool results and reasoning, so a request
    that dropped them would render a different prompt than the history it was built
    from."""
    env = Envelope(request_id="r1", contents=[TextPayload(text="who founded it")],
                   metadata={"messages": _TOOL_TURN})
    assert OpenAIChatProtocol().encode_request(env)["messages"] == _TOOL_TURN


def test_encode_request_adds_no_chat_fields_to_a_plain_turn():
    env = Envelope(request_id="r1", contents=[TextPayload(text="Hi")],
                   metadata={"messages": [{"role": "user", "content": "Hi"}]})
    assert OpenAIChatProtocol().encode_request(env)["messages"] == [
        {"role": "user", "content": "Hi"}]


def test_chat_message_keeps_a_null_content_and_stringifies_other_values():
    """``content: null`` is the OpenAI shape for an assistant turn that only calls a
    tool, so it survives as null rather than as the text "None"."""
    from hls.spec.drivers import chat_message
    assert chat_message({"role": "assistant", "content": None})["content"] is None
    assert chat_message({"role": "assistant", "content": 12})["content"] == "12"


def test_chat_message_carries_both_reasoning_spellings():
    """A chat template reads ``reasoning`` or ``reasoning_content``, and vLLM's chat
    parser keeps only ``reasoning``, so both have to reach the server."""
    from hls.spec.drivers import CHAT_MESSAGE_FIELDS
    assert "reasoning" in CHAT_MESSAGE_FIELDS
    assert "reasoning_content" in CHAT_MESSAGE_FIELDS


def test_the_chat_message_fields_are_the_ones_the_published_deepresearch_sandbox_repeats():
    """The self-test the published deepresearch sandbox ships (the imported image
    ``deepresearch-sandbox``) repeats this tuple in its ``/app/tasks/deepresearch/selftest.py``,
    since the sandbox has no harness to import it from. That image is never rebuilt, so the
    driver's list stays the one it holds."""
    from hls.spec.drivers import CHAT_MESSAGE_FIELDS
    assert CHAT_MESSAGE_FIELDS == ("reasoning", "reasoning_content", "tool_calls", "tool_call_id", "name")


# --- tool definitions and tool_choice --------------------------------------

def _tool():
    return {"type": "function", "function": {
        "name": "bash", "description": "Execute a bash command",
        "parameters": {"type": "object",
                       "properties": {"command": {"type": "string"}},
                       "required": ["command"]}}}


def _tool_env(tools):
    return Envelope(request_id="r1", contents=[TextPayload(text="Hi")],
                    metadata={"tools": tools})


def _sending(**fields) -> OpenAIChatProtocol:
    return OpenAIChatProtocol(sends_tools=True, tool_call_example="x", **fields)


def test_a_turn_carrying_tools_also_carries_tool_choice_none():
    # Leaving tool_choice off is what breaks: vLLM defaults it to "auto" whenever
    # tools are present and then refuses the request unless it was launched with
    # --enable-auto-tool-choice and a --tool-call-parser; "none" is accepted and still
    # renders the tool preamble into the prompt.
    body = _sending().encode_request(_tool_env([_tool()]))
    assert body["tools"] == [_tool()]
    assert body["tool_choice"] == "none"


def test_a_protocol_that_declares_no_tools_sends_them_as_it_did():
    # A task that sends tools without declaring sends_tools keeps the request it had:
    # the tools, and no tool_choice.
    body = OpenAIChatProtocol().encode_request(_tool_env([_tool()]))
    assert body["tools"] == [_tool()]
    assert "tool_choice" not in body


def test_a_turn_with_no_tools_carries_neither_key():
    # A non-agentic task must not start sending a key it never sent before.
    body = OpenAIChatProtocol().encode_request(_env("Hi"))
    assert "tools" not in body and "tool_choice" not in body


def test_an_empty_tools_list_carries_neither_key():
    body = _sending().encode_request(_tool_env([]))
    assert "tools" not in body and "tool_choice" not in body


def test_sends_tools_stays_out_of_the_serialized_contract():
    # It changes what the brief says and nothing about the traffic, so it must not
    # reach the calibration fingerprint -- which hashes dump_spec(contract). Without
    # the exclusion, declaring it on one task stales every committed calibration in
    # the repository, including tasks that send no tools at all.
    from hls.core.poly import dump_spec
    assert dump_spec(OpenAIChatProtocol(sends_tools=True, tool_call_example="x")) == \
        dump_spec(OpenAIChatProtocol(sends_tools=False))


def test_the_brief_tells_an_agentic_contestant_not_to_parse():
    brief = OpenAIChatProtocol(
        sends_tools=True, tool_call_example="thinking</think><tool_call>x</tool_call>",
    ).brief_section(path="/v1/chat/completions")
    assert "must be the exact text" in brief
    assert "Do NOT run a reasoning parser or" in brief
    assert 'tool_choice: "none"' in brief
    # And the sample request shows both keys, so a contestant builds against the
    # shape that actually arrives rather than meeting it in a graded run.
    assert '"tool_choice": "none"' in brief
    assert '"tools"' in brief


def test_the_brief_of_a_task_that_sends_no_tools_is_as_it_was():
    # The content rule and the tool keys are stated only where a task declares that it
    # sends tools.
    brief = OpenAIChatProtocol().brief_section(path="/v1/chat/completions")
    assert "must be the exact text" not in brief
    assert '"tool_choice": "none"' not in brief


def test_the_sample_reply_shows_an_actual_tool_call_living_in_content():
    # Seeing a call sit inside `content` is what stops someone reaching for a tool
    # parser, and a concrete reply does that better than a prose placeholder.
    brief = OpenAIChatProtocol(
        sends_tools=True,
        tool_call_example="thinking</think>prose<tool_call>x</tool_call>",
    ).brief_section(path="/p")
    assert "thinking</think>prose<tool_call>x</tool_call>" in brief
    assert "the whole generation, verbatim" in brief


def test_a_task_that_sends_no_tools_gets_a_plain_answer_sample():
    # The sample must not teach a shape the task never sends.
    brief = OpenAIChatProtocol().brief_section(path="/p")
    assert "<tool_call>" not in brief


def test_sending_tools_without_an_example_is_a_construction_error():
    # A construction error rather than a fallback to some default dialect. Call syntax
    # is the checkpoint's, so any built-in example is wrong for some task -- and a
    # task that silently inherited another checkpoint's markup would ship a brief
    # teaching a syntax its model never emits, with nothing downstream to notice.
    with pytest.raises(ValidationError, match="tool_call_example"):
        OpenAIChatProtocol(sends_tools=True)


def test_an_example_without_sends_tools_is_allowed():
    # Harmless and unused: the sample only reads it when tools are actually sent, so
    # a task mid-edit is not blocked by the order it sets two fields in.
    assert OpenAIChatProtocol(tool_call_example="x").tool_call_example == "x"


def test_the_example_stays_out_of_the_serialized_contract():
    # Prose in a brief, not traffic; same reasoning as sends_tools.
    from hls.core.poly import dump_spec
    assert dump_spec(OpenAIChatProtocol(sends_tools=True, tool_call_example="a")) == \
        dump_spec(OpenAIChatProtocol(sends_tools=True, tool_call_example="b"))


def test_the_tool_rule_is_its_own_paragraph():
    brief = OpenAIChatProtocol(sends_tools=True, tool_call_example="x").brief_section(path="/p")
    assert "This is mandatory on every response.\n\nThe same faithfulness applies to the text" in brief


def test_an_abbreviated_sample_says_how_many_alternatives_a_reply_carries():
    """The prose demands 21 alternatives at every position and the sample shows two, so
    the line above the sample says which is the rule; the count it states is the
    sample's own."""
    brief = OpenAIChatProtocol(return_logprobs=True, top_logprobs=21,
                               states_sample_abbreviation=True).brief_section(path="/p")
    assert "the top 21 alternatives at that\nposition" in brief
    assert ("Response (each `top_logprobs` array is abbreviated to 2 entries below; a conforming reply "
            "carries 21 at every position):\n```json") in brief
    sample = json.loads(brief.split("```json\n")[2].split("```")[0])
    assert len(sample["choices"][0]["logprobs"]["content"][0]["top_logprobs"]) == 2


def test_a_sample_is_said_to_be_abbreviated_only_where_the_task_opts_in():
    # A protocol that does not opt in renders the plain heading.
    brief = OpenAIChatProtocol(return_logprobs=True, top_logprobs=21).brief_section(path="/p")
    assert "abbreviated" not in brief and "\nResponse:\n```json" in brief


@pytest.mark.parametrize("fields", [{}, {"return_logprobs": True}, {"return_logprobs": True, "top_logprobs": 2}])
def test_a_sample_that_is_not_abbreviated_cannot_be_said_to_be(fields):
    with pytest.raises(ValidationError, match="states_sample_abbreviation needs a sample shorter"):
        OpenAIChatProtocol(states_sample_abbreviation=True, **fields)


def test_the_abbreviation_note_stays_out_of_the_serialized_contract():
    from hls.core.poly import dump_spec
    assert dump_spec(OpenAIChatProtocol(return_logprobs=True, top_logprobs=21, states_sample_abbreviation=True)) == \
        dump_spec(OpenAIChatProtocol(return_logprobs=True, top_logprobs=21))


# --- continuing a reply the harness cut ------------------------------------

def _env_with(meta: dict) -> Envelope:
    """One request, with whatever metadata the case needs."""
    return Envelope(request_id="r0", contents=[TextPayload(text="hi")],
                    metadata={"messages": [{"role": "user", "content": "hi"}], **meta})


def test_an_ordinary_request_asks_for_a_fresh_turn():
    """The control, and the reason the flag is per request rather than per protocol.

    Every turn but a continuation starts a new assistant turn, so neither field may
    appear by default. A protocol-level setting would make every request a
    continuation and the model would never begin a reply.
    """
    body = OpenAIChatProtocol(model="m").encode_request(_env_with({}))
    assert "continue_final_message" not in body
    assert "add_generation_prompt" not in body


def test_a_continuation_asks_the_server_to_resume_the_last_message():
    """Both fields, at the top level of the body, which is the spelling vLLM and
    SGLang read. `add_generation_prompt: false` is not optional beside it: left out,
    the server appends a fresh generation prompt after the assistant message and the
    model starts a new reply instead of finishing the one it was handed.
    """
    body = OpenAIChatProtocol(model="m").encode_request(
        _env_with({"continue_final_message": True}))
    assert body["continue_final_message"] is True
    assert body["add_generation_prompt"] is False


def test_a_falsey_continuation_flag_is_not_a_continuation():
    """A metadata key carrying False, or nothing, is an ordinary turn. Worth pinning
    because the driver stamps metadata from several places and an absent key and a
    False one must not diverge."""
    for value in (False, None, 0, ""):
        body = OpenAIChatProtocol(model="m").encode_request(
            _env_with({"continue_final_message": value}))
        assert "continue_final_message" not in body, value
        assert "add_generation_prompt" not in body, value


def test_a_continuation_still_carries_everything_else():
    """The flag adds to the request rather than replacing it: same messages, same
    tools and tool_choice, same cap. A continuation is the same turn, resumed."""
    plain = OpenAIChatProtocol(model="m").encode_request(
        _env_with({"tools": [{"type": "function"}], "max_tokens": 16}))
    cont = OpenAIChatProtocol(model="m").encode_request(
        _env_with({"tools": [{"type": "function"}], "max_tokens": 16,
                   "continue_final_message": True}))
    assert {k: v for k, v in cont.items()
            if k not in ("continue_final_message", "add_generation_prompt")} == plain


def test_the_completions_protocol_has_no_continuation_flag():
    """Raw conditioning needs none: a raw prompt is the model input verbatim, so a
    continuation is the prompt with more text on the end and nothing tells the server
    anything. Pinned so the field is not copied there by symmetry."""
    body = OpenAICompletionProtocol(model="m").encode_request(
        _env_with({"continue_final_message": True}))
    assert "continue_final_message" not in body
    assert "add_generation_prompt" not in body
