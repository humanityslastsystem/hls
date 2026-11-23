"""Artifact server for deepresearch -- the Gemma 4 deep-research baseline (vLLM).

This MIRRORS the baseline inference approach in
https://github.com/RulinShao/elastic-inference, which serves the deep-research
agent with **vLLM**, a **long (128K) context**, and
the model's **language-model-only** path (the elastic-inference serve scripts:
``vllm serve <model> --max-model-len 131072 --enable-prefix-caching
--language-model-only``). The elastic-inference scheduler then fans many agentic
HTTP requests across vLLM workers; HLS reproduces that fan-out with ``Poisson``
arrival + the concurrent ``TrajectorySessionDriver``.

ENGINE: ``AsyncLLMEngine`` (not the offline ``vllm.LLM`` batch class), so the
single engine admits many in-flight requests with continuous batching AND
exposes a per-request async generator that yields incremental outputs -- the
token-by-token stream the ``http-sse`` contract needs to record per-step
TTFT/ITL.

STREAMING + TOKEN IDS: when ``stream`` is set the endpoint emits an OpenAI SSE
stream of ``chat.completion.chunk`` objects, one per newly decoded token, each
carrying its ``token_ids`` + a prefix-diffed text delta, terminated by
``data: [DONE]``. The harness reassembles the full ``choices[].token_ids``, which is what the
token-level faithfulness gate teacher-forces, so the gate reads the same ids under
streaming as unary. A non-stream request returns one unary completion (readiness
probe).

Per-turn semantics (Sec.9 trust invariant): each POST is ONE LLM call. The
harness-side ``DeepResearchFramework`` owns the multi-turn loop -- it parses the
model's native ``web_search`` tool calls and terminal ``\\boxed{}`` answers,
then returns the result as a ``tool`` role message -- and owns the corpus browser
tool. The artifact NEVER executes a tool.

DECODING: one draw per position from the model's whole next-token distribution at
temperature 1 -- no top-p, no top-k, no min-p, no penalties, no logit bias. Not a
default to be flipped casually: the task gates on the distribution the server
reports, scored against a reference sampling the same way, so a server that
truncates or cools its sampler fails the gate by construction.

REPORT: every response carries OpenAI ``logprobs.content[]``, one entry per
generated token, holding that token's own log-probability and the 21 most probable
alternatives at that position, each with its integer vocabulary id. This is what the
gate reads, so it is as much part of the contract as the text. A lightweight
``_GpuSampler`` stamps a GPU utilization/memory snapshot onto each response for
the report-only serving metric.

MODEL: the served checkpoint NAME is the single ``MODEL_ID`` seam (also baked
into the image via ``.model(MODEL_ID)`` in ``task.py``); swap it to the user's
deep-research-tuned checkpoint without touching the loop. Weights load from the
mounted ``$HLS_WEIGHTS_DIR`` snapshot when provisioned, else the baked name.
"""

from __future__ import annotations
import argparse
import json
import os

# Force the V0 async engine path: AsyncLLMEngine.generate's streaming async
# generator (incremental RequestOutput per step) is the API this server relies on.
os.environ.setdefault("VLLM_USE_V1", "0")
# The engine batches as vLLM does by default: the batch-invariant flag is cleared,
# whatever the environment holds.
os.environ.pop("VLLM_BATCH_INVARIANT", None)

import shutil
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from tasks.deepresearch.serving_config import (
    CHAT_TEMPLATE_KWARGS_FLAG,
    TEMPERATURE,
    TOP_LOGPROBS,
    TokenText,
    chat_template_kwargs_arg,
    enable_prefix_caching,
    enforce_eager,
    finish_reason,
    gpu_memory_utilization,
    hls_context,
    logprob_entry,
    max_context_len,
    max_num_seqs,
    request_max_tokens,
    request_tools,
    response_usage,
    tensor_parallel_size,
)

#: The task's checkpoint, loaded by name when no weights are mounted.
MODEL_ID = "google/gemma-4-31B-it"
#: The chat-template arguments a launch renders prompts with when it carries no
#: --chat-template-kwargs: none. The task's launch passes its pin with that flag.
CHAT_TEMPLATE_KWARGS: dict = {}

# Deep-research turns interleave long reasoning with search bursts. The harness
# stamps the workload max_tokens onto each request so artifact and reference
# match; this fallback covers probes/manual calls. Override with
# HLS_MAX_NEW_TOKENS when launching the server outside HLS.

#: The task's search tool, as the research loop defines it: what the warmup at load and a
#: request that carries no tools render with.
_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web. Returns numbered results that can be opened by id.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "topn": {"type": "integer", "description": "Maximum number of results."},
            },
            "required": ["query"],
        },
    },
}


class _Message(BaseModel):
    """One history message. Extra fields are kept because a chat template renders
    more than role and content: an agentic turn carries the model's own tool calls,
    the reasoning it wrote before them, and the id and tool name a tool result
    answers with."""
    model_config = ConfigDict(extra="allow")
    role: str
    content: str


class _ChatRequest(BaseModel):
    model_config = ConfigDict(extra="allow", protected_namespaces=())
    model: str = "served-model"
    messages: list[_Message]


# chat_template_kwargs holds the launch's effective pin: the --chat-template-kwargs
# value when the launch carries one, else the module constant. token_text renders
# vocabulary ids for the report and is built with the tokenizer at load.
_STATE: dict = {"ready": False, "engine": None, "tokenizer": None, "token_text": None,
                "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)}


# --- GPU utilization sampler -------------------------------------------------
def _read_nvidia_smi() -> dict | None:
    """One ``nvidia-smi`` reading aggregated across visible GPUs:
    mean utilization (%) + total memory used (MiB). ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    utils: list[float] = []
    mems: list[float] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            utils.append(float(parts[0]))
            mems.append(float(parts[1]))
        except ValueError:
            continue
    if not utils:
        return None
    return {"utilization_pct": sum(utils) / len(utils), "memory_used_mb": sum(mems)}


class _GpuSampler:
    """Background ``nvidia-smi`` sampler. Holds the most recent snapshot; a no-op
    (``snapshot() is None``) when ``nvidia-smi`` is absent (CPU/test hosts)."""

    def __init__(self, interval_sec: float = 0.5) -> None:
        self.interval_sec = interval_sec
        self._latest: dict | None = None
        self._lock = threading.Lock()
        self._enabled = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._enabled:
            return
        self._sample_once()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        while True:
            time.sleep(self.interval_sec)
            self._sample_once()

    def _sample_once(self) -> None:
        snap = _read_nvidia_smi()
        if snap is not None:
            with self._lock:
                self._latest = snap

    def snapshot(self) -> dict | None:
        with self._lock:
            return dict(self._latest) if self._latest else None


_GPU = _GpuSampler()


def _weights_src() -> str:
    wdir = os.environ.get("HLS_WEIGHTS_DIR")
    if wdir and Path(wdir).is_dir():
        return wdir
    return MODEL_ID


#: The concurrent-sequence ceiling the engine actually runs under, stamped by
#: `_build_engine` and reported on every response. Read back off the engine arguments
#: rather than off our own config, so a run that leaves vLLM's default records the
#: number vLLM chose instead of a null.
_RESOLVED_MAX_NUM_SEQS: int | None = None


def _build_engine() -> AsyncLLMEngine:
    global _RESOLVED_MAX_NUM_SEQS
    seqs = max_num_seqs()
    args = AsyncEngineArgs(
        model=_weights_src(),
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size(),
        # Mirrors elastic-inference's vLLM serving baseline. HLS_ENABLE_PREFIX_CACHING=0
        # turns it off for a launch without changing the benchmark default.
        enable_prefix_caching=enable_prefix_caching(),
        max_model_len=max_context_len(),
        enforce_eager=enforce_eager(),
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization(),
        # vLLM refuses a request asking for more alternatives than this, and its own
        # default stops at OpenAI's 20. The report contract asks for 21.
        max_logprobs=TOP_LOGPROBS,
        # Only when pinned: passing the field at all replaces vLLM's default, and the
        # benchmark wants the engine's own scheduling ceiling unless asked otherwise.
        **({} if seqs is None else {"max_num_seqs": seqs}),
    )
    _RESOLVED_MAX_NUM_SEQS = args.max_num_seqs
    return AsyncLLMEngine.from_engine_args(args)


def _request_max_tokens(req: _ChatRequest) -> int:
    return request_max_tokens(getattr(req, "max_tokens", None))


def _sampling(max_new: int) -> SamplingParams:
    """The task's decode: one draw per position from the whole distribution.

    Every knob that would reshape that distribution is pinned to its neutral value
    rather than left at a default, because the gate scores the reported distribution
    against a reference that applies none of them.
    """
    return SamplingParams(
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=-1,                   # vLLM's "no truncation"
        min_p=0.0,
        repetition_penalty=1.0,
        frequency_penalty=0.0,
        # Unseeded: the gate tests the served tokens for being draws from the
        # reported distribution, which a seed neither helps nor hurts, and per-request
        # seeding costs vLLM a private generator.
        seed=None,
        logprobs=TOP_LOGPROBS,
        max_tokens=max_new,
        # Keep special tokens: the loop reads the model's end-of-turn and
        # reasoning/tool-call delimiters out of the reply text (Gemma's <turn|>,
        # <|channel>/<channel|>, <|tool_call>/<tool_call|>).
        skip_special_tokens=False,
    )


def _top_alternatives(step: dict) -> list[tuple[int, float]]:
    """One position's most probable ids, most probable first.

    ``step`` is what vLLM returns per position: the top ``TOP_LOGPROBS`` ids, plus
    the sampled one when the sample fell outside them. So the sampled token is the
    least probable entry exactly when it does not belong in the list, and taking the
    most probable ``TOP_LOGPROBS`` drops it then and keeps it otherwise -- the report
    never claims a token was among the most probable because the server happened to
    draw it. Ties break on the id so two identical distributions report alike.
    """
    ranked = sorted(step.items(), key=lambda kv: (-kv[1].logprob, kv[0]))
    return [(token_id, entry.logprob) for token_id, entry in ranked[:TOP_LOGPROBS]]


def _report(token_ids: list[int], steps: list[dict] | None) -> dict:
    """The reply's OpenAI ``logprobs`` object: one entry per token, in order.

    ``steps`` is ``RequestOutput.outputs[0].logprobs``, one dict of
    ``token_id -> Logprob`` per position, aligned with ``token_ids``.
    """
    if steps is None or len(steps) != len(token_ids):
        raise RuntimeError(
            f"vLLM returned {0 if steps is None else len(steps)} logprob positions "
            f"for {len(token_ids)} served tokens; the report must cover every token.")
    text = _STATE["token_text"]
    content = []
    for token_id, step in zip(token_ids, steps):
        served = step.get(token_id)
        if served is None:
            raise RuntimeError(
                f"vLLM reported no log-probability for the token it sampled ({token_id})")
        content.append(logprob_entry(token_id, served.logprob,
                                     _top_alternatives(step), text))
    return {"content": content}


def _prompt(req: _ChatRequest) -> str:
    tok = _STATE["tokenizer"]
    messages = [m.model_dump() for m in req.messages]
    tools = request_tools(getattr(req, "tools", None), [_SEARCH_TOOL])
    return tok.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        **_STATE["chat_template_kwargs"])


def _prompt_token_count(prompt: str) -> int:
    tok = _STATE["tokenizer"]
    return len(tok(prompt, add_special_tokens=False).input_ids)


def _hls_context(max_ctx: int, request_max_tokens: int) -> dict:
    return hls_context(
        max_ctx,
        request_max_tokens,
        prefix_caching=enable_prefix_caching(),
        eager=enforce_eager(),
        num_seqs=_RESOLVED_MAX_NUM_SEQS,
        tensor_parallel=tensor_parallel_size(),
        gpu_memory_util=gpu_memory_utilization(),
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Build the engine inside the running event loop so the background batching
    # loop binds to uvicorn's loop. uvicorn does not accept connections until
    # startup returns, so the readiness probe retries during the (slow) load.
    tok = AutoTokenizer.from_pretrained(_weights_src(), trust_remote_code=True)
    engine = _build_engine()
    warm_prompt = tok.apply_chat_template(
        [{"role": "user", "content": "_"}], tools=[_SEARCH_TOOL],
        tokenize=False, add_generation_prompt=True,
        **_STATE["chat_template_kwargs"])
    async for _ in engine.generate(warm_prompt, _sampling(16), "warmup-0"):
        pass
    _STATE.update(
        tokenizer=tok, engine=engine, ready=True,
        token_text=TokenText(lambda tid: tok.decode([tid], skip_special_tokens=False)))
    _GPU.start()
    print(f"[vllm-artifact] model {_weights_src()} loaded (AsyncLLMEngine, continuous batching)",
          flush=True)
    yield


def _build_app() -> FastAPI:
    app = FastAPI(lifespan=_lifespan)

    @app.get("/health")
    def health():
        if _STATE["ready"]:
            return {"status": "ok"}
        return Response(status_code=503, content="loading")

    @app.post("/v1/chat/completions")
    async def chat(req: _ChatRequest):
        max_new = _request_max_tokens(req)
        prompt = _prompt(req)
        prompt_tokens = _prompt_token_count(prompt)
        mcl = max_context_len()
        if getattr(req, "stream", False):
            return _stream_response(req, prompt, max_new, prompt_tokens, mcl)
        return await _unary_response(req, prompt, max_new, prompt_tokens, mcl)

    return app


async def _unary_response(req: _ChatRequest, prompt: str, max_new: int,
                          prompt_tokens: int, max_context_len: int) -> dict:
    engine: AsyncLLMEngine = _STATE["engine"]
    final = None
    async for out in engine.generate(prompt, _sampling(max_new), f"req-{uuid.uuid4().hex}"):
        final = out
    completion = final.outputs[0]
    token_ids = list(completion.token_ids)
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": completion.text},
                     "finish_reason": finish_reason(
                         len(token_ids),
                         max_new,
                         provider_finish_reason=completion.finish_reason,
                     ),
                     "token_ids": token_ids,
                     "logprobs": _report(token_ids, completion.logprobs)}],
        "usage": response_usage(prompt_tokens, len(token_ids)),
        "hls_context": _hls_context(max_context_len, max_new),
    }
    gpu = _GPU.snapshot()
    if gpu is not None:
        body["gpu"] = gpu
    return body


def _stream_response(req: _ChatRequest, prompt: str, max_new: int,
                     prompt_tokens: int, max_context_len: int) -> StreamingResponse:
    """SSE: one chunk per newly decoded token (token_ids + prefix-diffed text delta
    + that token's logprob entry), then a terminal chunk with finish_reason + usage
    + GPU snapshot.

    The harness concatenates the per-chunk ``token_ids`` and ``logprobs.content``
    across the stream, so the reassembled reply carries the same report as the unary
    path: a chunk's entries cover exactly that chunk's tokens, in order.
    """
    engine: AsyncLLMEngine = _STATE["engine"]
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    def _chunk(delta: dict, token_ids, finish_reason, logprobs=None):
        choice: dict = {"index": 0, "delta": delta, "token_ids": token_ids,
                        "finish_reason": finish_reason}
        if logprobs is not None:
            choice["logprobs"] = logprobs
        return {"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                "model": req.model, "choices": [choice]}

    async def _events():
        prev_ids: list[int] = []
        prev_text = ""
        first = True
        provider_finish = None
        async for out in engine.generate(prompt, _sampling(max_new),
                                         f"req-{uuid.uuid4().hex}"):
            comp = out.outputs[0]
            provider_finish = comp.finish_reason
            cur_ids = list(comp.token_ids)
            new_ids = cur_ids[len(prev_ids):]
            if not new_ids:
                continue
            # comp.logprobs accumulates alongside comp.token_ids, so the new tokens'
            # positions are the tail of it.
            new_steps = (comp.logprobs or [])[len(prev_ids):]
            delta_text = comp.text[len(prev_text):]
            prev_ids = cur_ids
            prev_text = comp.text
            delta = ({"role": "assistant", "content": delta_text} if first
                     else {"content": delta_text})
            first = False
            yield (f"data: {json.dumps(_chunk(delta, new_ids, None, _report(new_ids, new_steps)))}"
                   "\n\n")
        final = _chunk(
            {},
            None,
            finish_reason(
                len(prev_ids),
                max_new,
                provider_finish_reason=provider_finish,
            ),
        )
        final["usage"] = response_usage(prompt_tokens, len(prev_ids))
        final["hls_context"] = _hls_context(max_context_len, max_new)
        gpu = _GPU.snapshot()
        if gpu is not None:
            final["gpu"] = gpu
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_events(), media_type="text/event-stream")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    ap.add_argument(CHAT_TEMPLATE_KWARGS_FLAG, type=str, default=None,
                    help="JSON object of chat-template arguments to render every "
                         "prompt with; defaults to CHAT_TEMPLATE_KWARGS.")
    args = ap.parse_args()
    _STATE["chat_template_kwargs"] = chat_template_kwargs_arg(
        args.chat_template_kwargs, CHAT_TEMPLATE_KWARGS)
    uvicorn.run(_build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
