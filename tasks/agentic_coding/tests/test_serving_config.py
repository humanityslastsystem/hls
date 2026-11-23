"""The baseline server's knobs, and their agreement with the task declaration.

The artifact image carries ``serving_config.py`` and the server but not ``task.py``, so
the serving defaults restate the task's numbers instead of importing them. The first
section below is what keeps the copies honest: it compares the two modules rather than
asserting a literal in each, so a retarget that moves one and forgets the other fails
here.
"""
from __future__ import annotations

import pytest

from tasks.agentic_coding import serving_config as cfg


@pytest.fixture(scope="module")
def task():
    import tasks.agentic_coding.task as m
    return m


_SERVING_ENV = (
    "HLS_MAX_CONTEXT_LEN",
    "HLS_MAX_NEW_TOKENS",
    "HLS_ENABLE_PREFIX_CACHING",
    "HLS_ENFORCE_EAGER",
    "HLS_MAX_NUM_SEQS",
    "HLS_TP_SIZE",
    "HLS_GPU_MEMORY_UTILIZATION",
    "HLS_KV_CPU_OFFLOAD_BYTES",
)


@pytest.fixture
def unset_serving_env(monkeypatch):
    """A launch that passes no overrides, which is how a graded run launches."""
    for name in _SERVING_ENV:
        monkeypatch.delenv(name, raising=False)


# --- the two modules describe one deployment -------------------------------

def test_the_server_is_built_at_the_window_the_task_declares(task, unset_serving_env):
    # A calibration is fingerprinted to the window, so a baseline built at another one
    # would be measured against a ceiling fitted to different traffic.
    assert cfg.DEFAULT_MAX_CONTEXT_LEN == task.MAX_CONTEXT_LEN
    assert cfg.max_context_len() == task.MAX_CONTEXT_LEN
    assert cfg.engine_args()["max_model_len"] == task.MAX_CONTEXT_LEN


def test_an_explicit_window_override_may_reach_the_declared_one(task, monkeypatch):
    # env_int returns its default unchecked, so a ceiling tightened below the window
    # would pass every other test here and refuse only a launch that set the variable.
    monkeypatch.setenv("HLS_MAX_CONTEXT_LEN", str(task.MAX_CONTEXT_LEN))
    assert cfg.max_context_len() == task.MAX_CONTEXT_LEN


def test_the_servers_default_cap_covers_a_graded_turn(task, unset_serving_env):
    """The server's own default is for a request that carries no cap of its own.

    The workload has two caps -- the thinking cap and the continuation's budget -- and
    a request stamps whichever applies, so the server's default is not either of them.
    What it must be is large enough that a probe arriving without a cap is not cut
    shorter than a graded turn would be."""
    assert cfg.DEFAULT_MAX_NEW_TOKENS >= task.THINK_CAP
    assert cfg.default_max_new_tokens() >= task.THINK_CAP
    assert cfg.default_max_new_tokens() >= task.REPLY_BUDGET


def test_no_request_may_ask_for_more_tokens_than_the_window_holds(task):
    assert cfg.MAX_CONFIGURABLE_TOKENS == task.MAX_CONTEXT_LEN
    assert cfg.request_max_tokens(task.MAX_CONTEXT_LEN) == task.MAX_CONTEXT_LEN
    with pytest.raises(ValueError, match="request max_tokens"):
        cfg.request_max_tokens(task.MAX_CONTEXT_LEN + 1)


def test_the_engine_shards_across_every_accelerator_the_task_declares(
        task, unset_serving_env):
    assert cfg.DEFAULT_TENSOR_PARALLEL_SIZE == task.HARDWARE.accelerator.count
    assert cfg.engine_args()["tensor_parallel_size"] == task.HARDWARE.accelerator.count


def test_the_report_is_as_wide_as_the_contract_asks_for(task, unset_serving_env):
    # vLLM refuses a request asking for more alternatives than max_logprobs, and its
    # own default stops at 20, so the contract's 21 has to be raised at bring-up.
    assert cfg.TOP_LOGPROBS == task.CONTRACT.protocol.top_logprobs
    assert cfg.engine_args()["max_logprobs"] == task.CONTRACT.protocol.top_logprobs


def test_the_sampler_runs_at_the_temperature_the_task_decodes_at(task):
    assert cfg.TEMPERATURE == task.DECODE.temperature


# --- the launch ------------------------------------------------------------

def test_the_launch_pins_no_memory_fraction(unset_serving_env):
    # A contestant's server runs under vLLM's own default unless it changes it. The
    # flag is absent rather than set to a value guessed to match that default.
    assert cfg.gpu_memory_utilization() is None
    assert "gpu_memory_utilization" not in cfg.engine_args()


def test_a_diagnostic_run_can_still_pin_the_memory_fraction(monkeypatch):
    monkeypatch.setenv("HLS_GPU_MEMORY_UTILIZATION", "0.86")

    assert cfg.gpu_memory_utilization() == 0.86
    assert cfg.engine_args()["gpu_memory_utilization"] == 0.86


def test_the_launch_declares_no_pipeline_stage_and_no_second_node(unset_serving_env):
    # vLLM gates pipeline parallelism off for this checkpoint, and the task declares one
    # node, so neither a pipeline degree nor a rendezvous setting is passed at all.
    args = cfg.engine_args()
    for absent in ("pipeline_parallel_size", "distributed_executor_backend",
                   "nnodes", "node_rank", "master_addr", "master_port"):
        assert absent not in args, f"{absent} is a multi-node knob; this is one node"


def test_experts_are_sharded_across_the_ranks(unset_serving_env):
    assert cfg.engine_args()["enable_expert_parallel"] is True


def test_prefix_caching_is_on_for_the_baseline(unset_serving_env):
    # The trajectory re-sends its whole transcript every turn. (Off only on the
    # reference, where a cached position would produce no logits to report.)
    assert cfg.enable_prefix_caching() is True
    assert cfg.engine_args()["enable_prefix_caching"] is True


def test_the_launch_captures_cuda_graphs(unset_serving_env):
    assert cfg.enforce_eager() is False
    assert cfg.engine_args()["enforce_eager"] is False


def test_the_in_flight_ceiling_is_above_the_harnesss_own_cap(task, unset_serving_env):
    # The harness admits at most MAX_CONCURRENT_LLM submits, so this does not bind; it
    # must not be the lower of the two or the baseline would queue where a contestant
    # would not.
    assert cfg.max_num_seqs() >= task.MAX_CONCURRENT_LLM
    assert cfg.engine_args()["max_num_seqs"] == cfg.max_num_seqs()


def test_the_baseline_launches_without_kv_offload(unset_serving_env):
    # vLLM's in-process offload connector asserts one hash unit divides every KV group's
    # block size, and this checkpoint's groups are 640, 64 and 4 tokens: the engine ends
    # on its warm-up request with the default unit and at 64. The shipped launch carries
    # no kv_transfer_config, and no prefix_match_unit either, a knob only the connector
    # reads.
    assert cfg.DEFAULT_KV_CPU_OFFLOAD_BYTES == 0
    assert cfg.kv_cpu_offload_bytes() == 0
    assert cfg.kv_transfer_config_spec() is None
    args = cfg.engine_args()
    assert "kv_transfer_config" not in args
    assert "prefix_match_unit" not in args
    assert args["enable_prefix_caching"] is True


def test_kv_offload_can_be_enabled_by_budget(monkeypatch):
    # The knob stays for a build whose connector handles hybrid groups. The value is a
    # plain mapping the server wraps in a KVTransferConfig; this module reads the launch
    # without vLLM.
    monkeypatch.setenv("HLS_KV_CPU_OFFLOAD_BYTES", str(512 * 1024**3))
    assert cfg.kv_cpu_offload_bytes() == 512 * 1024**3
    assert cfg.engine_args()["kv_transfer_config"] == {
        "kv_connector": "SimpleCPUOffloadConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": 512 * 1024**3},
    }


def test_kv_offload_budget_can_be_retuned(monkeypatch):
    monkeypatch.setenv("HLS_KV_CPU_OFFLOAD_BYTES", str(128 * (1024**3)))
    spec = cfg.engine_args()["kv_transfer_config"]
    assert spec["kv_role"] == "kv_both"
    assert spec["kv_connector_extra_config"]["cpu_bytes_to_use"] == 128 * (1024**3)


def test_kv_offload_rejects_a_budget_past_the_host_ram_ceiling(monkeypatch):
    monkeypatch.setenv(
        "HLS_KV_CPU_OFFLOAD_BYTES", str(cfg.MAX_KV_CPU_OFFLOAD_BYTES + 1))
    with pytest.raises(ValueError, match="HLS_KV_CPU_OFFLOAD_BYTES"):
        cfg.kv_cpu_offload_bytes()


# --- environment overrides -------------------------------------------------

def test_serving_env_overrides_are_applied(monkeypatch):
    monkeypatch.setenv("HLS_MAX_CONTEXT_LEN", "131072")
    monkeypatch.setenv("HLS_MAX_NEW_TOKENS", "8192")
    monkeypatch.setenv("HLS_ENABLE_PREFIX_CACHING", "0")
    monkeypatch.setenv("HLS_ENFORCE_EAGER", "yes")
    monkeypatch.setenv("HLS_MAX_NUM_SEQS", "64")
    monkeypatch.setenv("HLS_TP_SIZE", "4")
    monkeypatch.setenv("HLS_KV_CPU_OFFLOAD_BYTES", str(64 * (1024**3)))

    assert cfg.max_context_len() == 131072
    assert cfg.default_max_new_tokens() == 8192
    assert cfg.enable_prefix_caching() is False
    assert cfg.enforce_eager() is True
    assert cfg.max_num_seqs() == 64
    assert cfg.tensor_parallel_size() == 4
    assert cfg.kv_cpu_offload_bytes() == 64 * (1024**3)


def test_serving_env_rejects_invalid_values(monkeypatch):
    monkeypatch.setenv("HLS_ENABLE_PREFIX_CACHING", "maybe")
    with pytest.raises(ValueError, match="HLS_ENABLE_PREFIX_CACHING"):
        cfg.enable_prefix_caching()

    monkeypatch.setenv("HLS_MAX_CONTEXT_LEN", "8192")
    with pytest.raises(ValueError, match="HLS_MAX_CONTEXT_LEN"):
        cfg.max_context_len()

    # Above the declared window: the engine would refuse to allocate it, and the
    # calibration fingerprint would not describe the run.
    monkeypatch.setenv("HLS_MAX_CONTEXT_LEN", str(cfg.DEFAULT_MAX_CONTEXT_LEN + 1))
    with pytest.raises(ValueError, match="HLS_MAX_CONTEXT_LEN"):
        cfg.max_context_len()

    monkeypatch.setenv("HLS_MAX_NUM_SEQS", "0")
    with pytest.raises(ValueError, match="HLS_MAX_NUM_SEQS"):
        cfg.max_num_seqs()

    monkeypatch.setenv("HLS_TP_SIZE", "0")
    with pytest.raises(ValueError, match="HLS_TP_SIZE"):
        cfg.tensor_parallel_size()

    monkeypatch.setenv("HLS_GPU_MEMORY_UTILIZATION", "1.5")
    with pytest.raises(ValueError, match="HLS_GPU_MEMORY_UTILIZATION"):
        cfg.gpu_memory_utilization()


def test_request_max_tokens_validates_a_requests_own_cap(monkeypatch):
    monkeypatch.setenv("HLS_MAX_NEW_TOKENS", "4096")

    # Missing, empty and zero all fall back to the server default, for probes and
    # health checks that carry no cap.
    assert cfg.request_max_tokens(None) == 4096
    assert cfg.request_max_tokens("") == 4096
    assert cfg.request_max_tokens(0) == 4096
    assert cfg.request_max_tokens("256") == 256
    assert cfg.request_max_tokens(1) == 1

    with pytest.raises(ValueError, match="request max_tokens"):
        cfg.request_max_tokens("not-an-int")
    with pytest.raises(ValueError, match="request max_tokens"):
        cfg.request_max_tokens(-1)


# --- the reply's shape -----------------------------------------------------

def test_serving_response_telemetry_matches_the_baseline_server(unset_serving_env):
    assert cfg.response_usage(prompt_tokens=123, completion_tokens=45) == {
        "prompt_tokens": 123,
        "completion_tokens": 45,
        "total_tokens": 168,
    }
    # An unset knob is reported as absent, the way it is launched.
    assert cfg.hls_context(max_ctx=1_048_576, request_max_tokens=4096,
                           gpu_memory_util=cfg.gpu_memory_utilization()) == {
        "max_context_len": 1_048_576,
        "request_max_tokens": 4096,
    }
    assert cfg.hls_context(
        max_ctx=1_048_576,
        request_max_tokens=4096,
        prefix_caching=True,
        eager=False,
        num_seqs=32,
        tensor_parallel=8,
        gpu_memory_util=0.86,
        kv_cpu_offload_bytes=512 * (1024**3),
    ) == {
        "max_context_len": 1_048_576,
        "request_max_tokens": 4096,
        "enable_prefix_caching": True,
        "enforce_eager": False,
        "max_num_seqs": 32,
        "tensor_parallel_size": 8,
        "gpu_memory_utilization": 0.86,
        "kv_cpu_offload_bytes": 512 * (1024**3),
    }
    # Off is recorded as absent, the way the server passes it when offload is disabled.
    assert "kv_cpu_offload_bytes" not in cfg.hls_context(
        max_ctx=1_048_576, request_max_tokens=4096, kv_cpu_offload_bytes=None)


def test_finish_reason_preserves_the_engines_reason_and_infers_length():
    assert cfg.finish_reason(
        completion_tokens=10, request_max_tokens=100,
        provider_finish_reason="stop") == "stop"
    assert cfg.finish_reason(
        completion_tokens=10, request_max_tokens=100,
        provider_finish_reason="length") == "length"
    assert cfg.finish_reason(completion_tokens=100, request_max_tokens=100) == "length"
    assert cfg.finish_reason(completion_tokens=99, request_max_tokens=100) == "stop"


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


# --- the chat-template pin -------------------------------------------------

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
