"""Runtime serving knobs for the agentic_coding vLLM baseline.

Kept separate from ``vllm_server.py`` so unit tests can validate the benchmark's
serving defaults without importing vLLM on CPU-only developer machines.

The numbers below restate the task's declaration rather than importing it: the artifact
image carries this module and the server, and not ``task.py``. What keeps the two in
step is ``tests/test_serving_config.py``, which compares them.
"""

from __future__ import annotations

import json
import os


#: Context window the engine is built at, matching the task's ``MAX_CONTEXT_LEN``. A
#: calibration is fingerprinted to the window, so a server built at another one is
#: measuring a different workload.
DEFAULT_MAX_CONTEXT_LEN = 1_048_576
#: Per-turn output cap when a request does not carry its own. Workload requests always
#: carry one; this covers probes.
DEFAULT_MAX_NEW_TOKENS = 131072
#: Largest generation cap a request or the environment may ask for. The window, because
#: no reply can be longer than the context it is generated into.
MAX_CONFIGURABLE_TOKENS = DEFAULT_MAX_CONTEXT_LEN
#: On. A coding trajectory re-sends its whole transcript every turn, so this is the
#: largest lever the baseline has. (The reference turns it off, for an unrelated reason:
#: a cached position produces no logits and the report capture would come up short.)
DEFAULT_ENABLE_PREFIX_CACHING = True
#: Off, so the engine captures CUDA graphs. Eager is a debugging setting.
DEFAULT_ENFORCE_EAGER = False
#: Concurrent sequences the engine admits. Above the harness's own in-flight cap
#: (``MAX_CONCURRENT_LLM``), so today it is headroom rather than the binding limit.
DEFAULT_MAX_NUM_SEQS = 32
#: Devices one engine shards its weights across: every card on the node the task
#: declares. They share NVLink, so no collective leaves the box. It is also the only
#: degree available -- vLLM gates pipeline parallelism off for this checkpoint, whose
#: model class claims ``SupportsPP`` without implementing
#: ``make_empty_intermediate_tensors``, so a second stage cannot allocate what it would
#: be handed. The engine is therefore launched with no pipeline degree at all.
DEFAULT_TENSOR_PARALLEL_SIZE = 8
#: None leaves vLLM's own memory fraction in place. A contestant's server runs under
#: that default unless it changes it, so a baseline pinned higher would be holding KV
#: space the thing it is compared against was never given. Set
#: ``HLS_GPU_MEMORY_UTILIZATION`` to pin one for a diagnostic run.
DEFAULT_GPU_MEMORY_UTILIZATION: float | None = None

#: Host RAM the engine may spill evicted KV blocks to, in bytes, counted server-wide
#: across the tensor-parallel ranks (vLLM divides it by the world size). 0 launches with
#: no offload at all.
#:
#: Off for the baseline. vLLM's in-process connector (``SimpleCPUOffloadConnector``)
#: cannot serve this checkpoint on the pinned build: the KV cache has groups of three
#: block sizes (640 tokens for attention, 64 for the DeepSeek-V3.2 sparse-indexer layers,
#: 4 for the linear-attention state), the connector stores every positionally stable
#: group, and it asserts each group's block size is a multiple of one hash unit
#: (``kv_cache_utils.resolve_block_hashes``). The default unit is 640 and fails on the
#: 64 group; ``prefix_match_unit=64`` fails on the 4 group; the only unit that divides all
#: three is 4, which would hash a 1,048,576-token prompt at 262,144 boundaries. Both
#: failures end the engine on its warm-up request. The operating point does not need the
#: offload either: GPU KV is 6.8% used there with no preemption, and decode compute sets
#: the latency.
#:
#: The knob stays so a run on a build whose connector handles hybrid groups can turn it
#: on: set ``HLS_KV_CPU_OFFLOAD_BYTES`` to a budget in bytes.
DEFAULT_KV_CPU_OFFLOAD_BYTES = 0
#: The largest offload budget the environment may ask for, leaving the node's ~1.9 TiB of
#: host RAM headroom for the OS, the weights load and the sandbox trajectories.
MAX_KV_CPU_OFFLOAD_BYTES = 1536 * (1024**3)
#: vLLM's registered in-process host-RAM offload connector: a direct GPU-to-host copy
#: inside the one engine, with no daemon and no second node.
KV_CPU_OFFLOAD_CONNECTOR = "SimpleCPUOffloadConnector"


CHAT_TEMPLATE_KWARGS_FLAG = "--chat-template-kwargs"

#: Alternatives the report carries at every generated position. The task's contract
#: asks for this many, both servers report this many, and the verifier partitions the
#: reported distribution assuming it.
TOP_LOGPROBS = 21

#: The temperature both servers sample at. The reference scores the report against the
#: model's distribution at this temperature, so a server that cools or heats it fails.
TEMPERATURE = 1.0


class TokenText:
    """Decoded text per vocabulary id, cached for the life of the process.

    Every generated position reports its own token plus ``TOP_LOGPROBS``
    alternatives, so the same handful of ids is decoded over and over. The cache is
    bounded by the vocabulary.
    """

    def __init__(self, decode) -> None:
        #: Renders one vocabulary id as the text OpenAI's ``token`` field carries.
        self._decode = decode
        self._cache: dict[int, str] = {}

    def __call__(self, token_id: int) -> str:
        text = self._cache.get(token_id)
        if text is None:
            text = self._decode(token_id)
            self._cache[token_id] = text
        return text


def logprob_entry(token_id: int, logprob: float,
                  top: list[tuple[int, float]], text: TokenText) -> dict:
    """One OpenAI ``logprobs.content[]`` entry for one generated position.

    ``top`` holds the position's most probable ids as ``(token_id, logprob)`` pairs,
    most probable first, whether or not the served token is among them. The served
    token carries its utf-8 ``bytes`` and the alternatives do not: the harness keeps
    an alternative's id and log-probability and drops its text, so bytes there would
    be megabytes per reply that nothing reads.
    """
    served = text(token_id)
    return {
        "token": served,
        "token_id": token_id,
        "logprob": logprob,
        "bytes": list(served.encode("utf-8")),
        "top_logprobs": [{"token": text(alt_id), "token_id": alt_id, "logprob": alt_lp}
                         for alt_id, alt_lp in top],
    }


def chat_template_kwargs_arg(value: str | None, default: dict) -> dict:
    """Resolve the chat-template arguments one server launch renders prompts with.

    A task module's ``Baseline.serve_command`` carries the task's
    ``ModelSpec.chat_template_kwargs`` pin as this flag, so one server file serves
    any pinned model: the harness mounts that task's weights at ``$HLS_WEIGHTS_DIR``
    and the launch says how to render for them. ``value`` is the flag's JSON text,
    or None when the launch omits it, in which case the server's module constant
    applies.

    JSON object keys are strings by construction, so only the values are checked.
    They are the scalars a chat template takes as keyword arguments.
    """
    if value is None:
        return dict(default)
    try:
        parsed = json.loads(value)
    except ValueError as exc:
        raise ValueError(
            f"{CHAT_TEMPLATE_KWARGS_FLAG} must be JSON, got {value!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{CHAT_TEMPLATE_KWARGS_FLAG} must be a JSON object, got {value!r}")
    for key, item in parsed.items():
        if not isinstance(item, (bool, int, float, str)):
            raise ValueError(
                f"{CHAT_TEMPLATE_KWARGS_FLAG} value for {key!r} must be a boolean, "
                f"number or string, got {item!r}")
    return parsed


def env_bool(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    norm = raw.strip().casefold()
    if norm in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if norm in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def env_int(name: str, *, default: int, min_value: int, max_value: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < min_value or value > max_value:
        raise ValueError(
            f"{name} must be between {min_value} and {max_value}, got {value}")
    return value


def env_float(
    name: str,
    *,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if value < min_value or value > max_value:
        raise ValueError(
            f"{name} must be between {min_value} and {max_value}, got {value}")
    return value


def max_context_len() -> int:
    return env_int(
        "HLS_MAX_CONTEXT_LEN",
        default=DEFAULT_MAX_CONTEXT_LEN,
        min_value=32_768,
        max_value=DEFAULT_MAX_CONTEXT_LEN,
    )


def default_max_new_tokens() -> int:
    return env_int(
        "HLS_MAX_NEW_TOKENS",
        default=DEFAULT_MAX_NEW_TOKENS,
        min_value=256,
        max_value=MAX_CONFIGURABLE_TOKENS,
    )


def request_max_tokens(value: object) -> int:
    """Resolve one request's generation cap.

    Workload requests normally carry ``max_tokens``. Forced-final repair turns
    may reduce that cap below the server default to preserve context headroom, so
    request-level values allow ``1..MAX_CONFIGURABLE_TOKENS``. Missing/empty/zero
    values fall back to ``HLS_MAX_NEW_TOKENS`` for manual probes and health checks.
    """
    if value is None or value == "":
        return default_max_new_tokens()
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"request max_tokens must be an integer, got {value!r}") from exc
    if resolved == 0:
        return default_max_new_tokens()
    if resolved < 1 or resolved > MAX_CONFIGURABLE_TOKENS:
        raise ValueError(
            "request max_tokens must be between "
            f"1 and {MAX_CONFIGURABLE_TOKENS}, got {resolved}")
    return resolved


def enable_prefix_caching() -> bool:
    return env_bool(
        "HLS_ENABLE_PREFIX_CACHING",
        default=DEFAULT_ENABLE_PREFIX_CACHING,
    )


def enforce_eager() -> bool:
    return env_bool("HLS_ENFORCE_EAGER", default=DEFAULT_ENFORCE_EAGER)


def max_num_seqs() -> int:
    return env_int(
        "HLS_MAX_NUM_SEQS",
        default=DEFAULT_MAX_NUM_SEQS,
        min_value=1,
        max_value=256,
    )


def tensor_parallel_size() -> int:
    return env_int(
        "HLS_TP_SIZE",
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        min_value=1,
        max_value=64,
    )


def gpu_memory_utilization() -> float | None:
    """Fraction of each card vLLM may claim, or None to leave the engine's own.

    See ``DEFAULT_GPU_MEMORY_UTILIZATION``: unset means the flag is not passed at all,
    so the baseline runs on the same memory budget a contestant gets for free.
    """
    if os.environ.get("HLS_GPU_MEMORY_UTILIZATION", "") == "":
        return DEFAULT_GPU_MEMORY_UTILIZATION
    return env_float(
        "HLS_GPU_MEMORY_UTILIZATION",
        default=0.0,        # unreachable: the empty case returned above
        min_value=0.1,
        max_value=0.98,
    )


def kv_cpu_offload_bytes() -> int:
    """Host-RAM KV-offload budget in bytes, server-wide, or 0 for no offload.

    See ``DEFAULT_KV_CPU_OFFLOAD_BYTES``.
    """
    return env_int(
        "HLS_KV_CPU_OFFLOAD_BYTES",
        default=DEFAULT_KV_CPU_OFFLOAD_BYTES,
        min_value=0,
        max_value=MAX_KV_CPU_OFFLOAD_BYTES,
    )


def kv_transfer_config_spec() -> dict | None:
    """The engine's KV-offload settings as a plain mapping, or None when offload is off.

    A mapping rather than a vLLM ``KVTransferConfig`` so this module imports on a host
    with no vLLM; ``vllm_server`` builds the object from it. Keys are ``KVTransferConfig``
    field names. ``kv_both`` because the one engine both writes evicted blocks and reads
    them back, and ``cpu_bytes_to_use`` is the server-wide budget the connector shards
    across the ranks.
    """
    budget = kv_cpu_offload_bytes()
    if budget <= 0:
        return None
    return {
        "kv_connector": KV_CPU_OFFLOAD_CONNECTOR,
        "kv_role": "kv_both",
        "kv_connector_extra_config": {"cpu_bytes_to_use": budget},
    }


def engine_args() -> dict:
    """Every vLLM engine setting this baseline launches with, bar the weights source.

    Assembled here rather than in ``vllm_server`` so a test can read the launch on a
    host with no GPU and no vLLM. Keys are ``AsyncEngineArgs`` field names. A knob left
    to the engine's own default is absent from the mapping rather than present at a
    value guessed to match it.

    One key, ``kv_transfer_config``, carries a plain mapping rather than the
    ``KVTransferConfig`` object the field takes; ``vllm_server`` builds the object, which
    keeps this module importable without vLLM.
    """
    args: dict = {
        # Hopper computes this FP8 checkpoint's activations and its KV in bf16.
        "dtype": "bfloat16",
        "tensor_parallel_size": tensor_parallel_size(),
        # Experts sharded across the same ranks rather than replicated on each. A
        # throughput choice on a 320B-A18B mixture, not a memory one.
        "enable_expert_parallel": True,
        "enable_prefix_caching": enable_prefix_caching(),
        "max_model_len": max_context_len(),
        "enforce_eager": enforce_eager(),
        "trust_remote_code": True,
        "max_num_seqs": max_num_seqs(),
        # vLLM refuses a request asking for more alternatives than this, and its own
        # default stops at OpenAI's 20. The report contract asks for 21.
        "max_logprobs": TOP_LOGPROBS,
    }
    util = gpu_memory_utilization()
    if util is not None:
        args["gpu_memory_utilization"] = util
    offload = kv_transfer_config_spec()
    if offload is not None:
        args["kv_transfer_config"] = offload
    return args


def response_usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def hls_context(
    max_ctx: int,
    request_max_tokens: int,
    *,
    prefix_caching: bool | None = None,
    eager: bool | None = None,
    num_seqs: int | None = None,
    tensor_parallel: int | None = None,
    gpu_memory_util: float | None = None,
    kv_cpu_offload_bytes: int | None = None,
) -> dict:
    context = {
        "max_context_len": max_ctx,
        "request_max_tokens": request_max_tokens,
    }
    if prefix_caching is not None:
        context["enable_prefix_caching"] = prefix_caching
    if eager is not None:
        context["enforce_eager"] = eager
    if num_seqs is not None:
        context["max_num_seqs"] = num_seqs
    if tensor_parallel is not None:
        context["tensor_parallel_size"] = tensor_parallel
    if gpu_memory_util is not None:
        context["gpu_memory_utilization"] = gpu_memory_util
    if kv_cpu_offload_bytes is not None:
        context["kv_cpu_offload_bytes"] = kv_cpu_offload_bytes
    return context


def finish_reason(
    completion_tokens: int,
    request_max_tokens: int,
    *,
    provider_finish_reason: object = None,
) -> str:
    if provider_finish_reason:
        return str(provider_finish_reason)
    if request_max_tokens > 0 and completion_tokens >= request_max_tokens:
        return "length"
    return "stop"
