from __future__ import annotations

import pytest

from tasks.deepresearch.tests import baseline_sources

cfg = baseline_sources.serving_config()


def test_serving_defaults_match_high_performance_vllm_baseline(monkeypatch):
    for name in (
        "HLS_MAX_CONTEXT_LEN",
        "HLS_MAX_NEW_TOKENS",
        "HLS_ENABLE_PREFIX_CACHING",
        "HLS_ENFORCE_EAGER",
        "HLS_MAX_NUM_SEQS",
        "HLS_TP_SIZE",
        "HLS_GPU_MEMORY_UTILIZATION",
    ):
        monkeypatch.delenv(name, raising=False)

    assert cfg.max_context_len() == 131072
    assert cfg.default_max_new_tokens() == 8192
    assert cfg.enable_prefix_caching() is True
    assert cfg.enforce_eager() is False
    # None, not a number: unset leaves vLLM's own concurrent-sequence ceiling, so the
    # benchmark's saturation point is the hardware's rather than one we picked.
    assert cfg.max_num_seqs() is None
    assert cfg.tensor_parallel_size() == 1
    assert cfg.gpu_memory_utilization() == 0.92


def test_serving_env_overrides_are_validated(monkeypatch):
    monkeypatch.setenv("HLS_MAX_CONTEXT_LEN", "262144")
    monkeypatch.setenv("HLS_MAX_NEW_TOKENS", "262144")
    monkeypatch.setenv("HLS_ENABLE_PREFIX_CACHING", "0")
    monkeypatch.setenv("HLS_ENFORCE_EAGER", "yes")
    monkeypatch.setenv("HLS_MAX_NUM_SEQS", "64")
    monkeypatch.setenv("HLS_TP_SIZE", "8")
    monkeypatch.setenv("HLS_GPU_MEMORY_UTILIZATION", "0.86")

    assert cfg.max_context_len() == 262144
    assert cfg.default_max_new_tokens() == 262144
    assert cfg.enable_prefix_caching() is False
    assert cfg.enforce_eager() is True
    assert cfg.max_num_seqs() == 64
    assert cfg.tensor_parallel_size() == 8
    assert cfg.gpu_memory_utilization() == 0.86


def test_serving_env_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("HLS_ENABLE_PREFIX_CACHING", "maybe")
    with pytest.raises(ValueError, match="HLS_ENABLE_PREFIX_CACHING"):
        cfg.enable_prefix_caching()

    monkeypatch.setenv("HLS_MAX_CONTEXT_LEN", "8192")
    with pytest.raises(ValueError, match="HLS_MAX_CONTEXT_LEN"):
        cfg.max_context_len()

    monkeypatch.setenv("HLS_MAX_NUM_SEQS", "0")
    with pytest.raises(ValueError, match="HLS_MAX_NUM_SEQS"):
        cfg.max_num_seqs()

    monkeypatch.setenv("HLS_MAX_NEW_TOKENS", "524288")
    with pytest.raises(ValueError, match="HLS_MAX_NEW_TOKENS"):
        cfg.default_max_new_tokens()

    monkeypatch.setenv("HLS_TP_SIZE", "0")
    with pytest.raises(ValueError, match="HLS_TP_SIZE"):
        cfg.tensor_parallel_size()

    monkeypatch.setenv("HLS_GPU_MEMORY_UTILIZATION", "1.5")
    with pytest.raises(ValueError, match="HLS_GPU_MEMORY_UTILIZATION"):
        cfg.gpu_memory_utilization()


def test_serving_response_telemetry_matches_baseline_servers():
    assert cfg.response_usage(prompt_tokens=123, completion_tokens=45) == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }
    assert cfg.hls_context(max_ctx=131072, request_max_tokens=8192) == {
        "max_context_len": 131072,
        "request_max_tokens": 8192,
    }
    assert cfg.hls_context(
        max_ctx=131072,
        request_max_tokens=8192,
        prefix_caching=True,
        eager=False,
        num_seqs=32,
        tensor_parallel=4,
        gpu_memory_util=0.86,
    ) == {
        "max_context_len": 131072,
        "request_max_tokens": 8192,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "max_num_seqs": 32,
        "tensor_parallel_size": 4,
        "gpu_memory_utilization": 0.86,
    }


def test_request_max_tokens_validates_request_override(monkeypatch):
    monkeypatch.setenv("HLS_MAX_NEW_TOKENS", "32768")

    assert cfg.request_max_tokens(None) == 32768
    assert cfg.request_max_tokens("") == 32768
    assert cfg.request_max_tokens(0) == 32768
    assert cfg.request_max_tokens("256") == 256
    assert cfg.request_max_tokens(1) == 1
    assert cfg.request_max_tokens(262_144) == 262_144

    with pytest.raises(ValueError, match="request max_tokens"):
        cfg.request_max_tokens("not-an-int")
    with pytest.raises(ValueError, match="between 1 and 262144"):
        cfg.request_max_tokens(-1)
    with pytest.raises(ValueError, match="between 1 and 262144"):
        cfg.request_max_tokens(262_145)


def test_request_tools_preserves_explicit_tool_free_turns():
    default_tools = [{"type": "function", "function": {"name": "web_search"}}]

    assert cfg.request_tools(None, default_tools) == default_tools
    assert cfg.request_tools("bad", default_tools) == default_tools
    assert cfg.request_tools([], default_tools) == []
    assert cfg.request_tools([{"name": "custom"}], default_tools) == [
        {"name": "custom"}]


def test_finish_reason_preserves_provider_reason_and_infers_length():
    assert cfg.finish_reason(
        completion_tokens=10,
        request_max_tokens=100,
        provider_finish_reason="stop",
    ) == "stop"
    assert cfg.finish_reason(
        completion_tokens=10,
        request_max_tokens=100,
        provider_finish_reason="length",
    ) == "length"
    assert cfg.finish_reason(completion_tokens=100, request_max_tokens=100) == "length"
    assert cfg.finish_reason(completion_tokens=99, request_max_tokens=100) == "stop"


def test_chat_template_kwargs_arg_defaults_to_the_module_constant():
    default = {"enable_thinking": True}
    resolved = cfg.chat_template_kwargs_arg(None, default)

    assert resolved == default
    resolved["enable_thinking"] = False
    assert default == {"enable_thinking": True}


def test_chat_template_kwargs_arg_parses_a_json_object():
    assert cfg.chat_template_kwargs_arg('{"enable_thinking": true}', {}) == {
        "enable_thinking": True}
    assert cfg.chat_template_kwargs_arg("{}", {"enable_thinking": True}) == {}


@pytest.mark.parametrize("value", [
    '["enable_thinking"]',              # a JSON array, not an object
    "enable_thinking=true",             # not JSON at all
    '{"enable_thinking": {"on": 1}}',   # a value the chat template cannot take
    '{"enable_thinking": null}',
])
def test_chat_template_kwargs_arg_rejects_shapes_a_template_cannot_take(value):
    with pytest.raises(ValueError) as excinfo:
        cfg.chat_template_kwargs_arg(value, {})
    assert "--chat-template-kwargs" in str(excinfo.value)


def test_the_decode_both_servers_share_is_a_draw_at_temperature_one():
    """The reference scores the report against the model's distribution at this
    temperature and partitions it into this many alternatives, so both numbers are
    one constant each and both servers read them here."""
    assert cfg.TEMPERATURE == 1.0
    assert cfg.TOP_LOGPROBS == 21


def test_a_logprob_entry_carries_the_served_token_and_its_alternatives():
    """The shape the wire parser reads: the served token with its text, id, log-
    probability and utf-8 bytes, and the alternatives with an id and a log-probability
    each. The harness drops an alternative's text, so bytes there would be megabytes a
    reply nothing reads."""
    text = cfg.TokenText(lambda tid: f"<{tid}>")

    entry = cfg.logprob_entry(7, -0.5, [(3, -0.1), (7, -0.5)], text)

    assert entry == {
        "token": "<7>", "token_id": 7, "logprob": -0.5, "bytes": [60, 55, 62],
        "top_logprobs": [{"token": "<3>", "token_id": 3, "logprob": -0.1},
                         {"token": "<7>", "token_id": 7, "logprob": -0.5}],
    }


def test_token_text_decodes_each_id_once():
    """Every position reports 22 ids, so the same ids are rendered over and over; the
    cache bounds that work by the vocabulary rather than by the reply length."""
    calls: list[int] = []

    def _decode(token_id: int) -> str:
        calls.append(token_id)
        return f"<{token_id}>"

    text = cfg.TokenText(_decode)

    assert [text(4), text(4), text(9)] == ["<4>", "<4>", "<9>"]
    assert calls == [4, 9]
