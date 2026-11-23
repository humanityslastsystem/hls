"""Runtime serving knobs for the deepresearch vLLM baseline.

Kept separate from ``vllm_server.py`` so unit tests can validate the benchmark's
serving defaults without importing vLLM on CPU-only developer machines.
"""

from __future__ import annotations

import json
import os


DEFAULT_MAX_CONTEXT_LEN = 131072
DEFAULT_MAX_NEW_TOKENS = 8192
MAX_CONFIGURABLE_TOKENS = 262_144
DEFAULT_ENABLE_PREFIX_CACHING = True
DEFAULT_ENFORCE_EAGER = False
#: None leaves vLLM's own concurrent-sequence default in place: a ceiling set here would
#: be what a load ladder measures instead of the hardware.
DEFAULT_MAX_NUM_SEQS = None
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_GPU_MEMORY_UTILIZATION = 0.92


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
        max_value=262_144,
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


def max_num_seqs() -> int | None:
    """The engine's concurrent-sequence ceiling, or None to leave vLLM's own default.

    None is the default because the benchmark measures the hardware, and a ceiling we
    pick is a number the load ladder would find instead of the saturation point. Set
    ``HLS_MAX_NUM_SEQS`` to pin one for a diagnostic run.
    """
    if os.environ.get("HLS_MAX_NUM_SEQS", "") == "":
        return DEFAULT_MAX_NUM_SEQS
    return env_int(
        "HLS_MAX_NUM_SEQS",
        default=0,          # unreachable: the empty case returned above
        min_value=1,
        max_value=4096,
    )


def tensor_parallel_size() -> int:
    return env_int(
        "HLS_TP_SIZE",
        default=DEFAULT_TENSOR_PARALLEL_SIZE,
        min_value=1,
        max_value=64,
    )


def gpu_memory_utilization() -> float:
    return env_float(
        "HLS_GPU_MEMORY_UTILIZATION",
        default=DEFAULT_GPU_MEMORY_UTILIZATION,
        min_value=0.1,
        max_value=0.98,
    )


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
    return context


def request_tools(tools: object, default_tools: list[dict]) -> list[dict]:
    """Preserve an explicit empty tool list for forced-final turns.

    The harness sends ``tools=[]`` when the next model call must answer without
    another search. Missing/non-list tools still get the default search
    schema for probes and manual calls.
    """
    if isinstance(tools, list):
        return tools
    return default_tools


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
