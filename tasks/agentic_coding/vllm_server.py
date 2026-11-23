"""Artifact server for agentic_coding -- the mini-SWE-agent coding baseline (vLLM).

ENGINE: ``AsyncLLMEngine``, so one engine admits many in-flight trajectories with
continuous batching and exposes a per-request async generator -- the token-by-token
stream the ``http-sse`` contract needs to record per-step TTFT and inter-token
latency.

STREAMING + TOKEN IDS: when ``stream`` is set the endpoint emits an OpenAI SSE stream
of ``chat.completion.chunk`` objects, one per newly decoded token, each carrying its
``token_ids`` and a prefix-diffed text delta, terminated by ``data: [DONE]``. The
harness reassembles the full ``choices[].token_ids``. vLLM's own OpenAI server reports
alternatives as token *strings*; the gate needs integer vocabulary ids, which is why
this server exists rather than a stock launch.

Per-turn semantics (Sec.9 trust invariant): each POST is ONE LLM call. The
harness-side ``MiniSweFramework`` owns the multi-turn loop -- it parses the single
fenced bash command out of the reply and runs it in the trajectory's sandbox -- and
the artifact NEVER executes a command.

DECODING: one draw per position from the model's whole next-token distribution at
temperature 1 -- no top-p, no top-k, no min-p, no penalties, no logit bias. The task
gates on the distribution the server reports, scored against a reference sampling the
same way, so a server that truncates or cools its sampler fails by construction.

NO STOP STRINGS: the reply ends where the model ends its turn. A stop string would
cut replies at a delimiter the reference never sees, and the gate's end-of-turn count
reads exactly where turns ended, so introducing one would corrupt the statistic the
check is built on.

PROMPT: rendered here through the checkpoint's chat template and handed to the engine
as token ids, so the engine serves exactly the render the gate teacher-forces the reply
under. A request carrying ``continue_final_message`` resumes its final assistant
message rather than opening a new turn; see ``_ChatRequest``.

TOOL DEFINITIONS, AND NOTHING PARSED OUT OF THE REPLY: every turn carries the bash
tool on the request, and this server renders it through the chat template exactly as
the reference does, so the prompt the gate replays is the prompt the model saw. It
never extracts a tool call from the generation. `tool_choice` arrives as "none" on
every request and that is what it means: describe the tools in the prompt, leave the
model's call in `content`. The harness reads the call out of the text, and it refuses
a reply whose `content` is not what its own `token_ids` detokenize to -- so a launch
that added --tool-call-parser or --reasoning-parser would fail the run rather than
silently serve trajectories that can never act.

REPORT: every response carries OpenAI ``logprobs.content[]``, one entry per generated
token, holding that token's own log-probability and the 21 most probable alternatives
at that position, each with its integer vocabulary id. This is what the gate reads, so
it is as much part of the contract as the text.

MODEL: the served checkpoint name is the single ``MODEL_ID`` seam. Weights load from
the mounted ``$HLS_WEIGHTS_DIR`` snapshot when provisioned, else the baked name.

PARALLELISM: one node of eight NVLink-connected cards, so one engine shards across all
eight and no collective leaves the box. Every engine setting is assembled in
``serving_config.engine_args`` so a test can read the launch without vLLM installed.
"""

from __future__ import annotations
import argparse
import json
import os

# Batch invariance is left to whatever the launch sets. Without it this stack's
# numerics depend on batch composition, which lands on calibration rather than here:
# the ceiling the gate enforces is whatever `hls calibrate` measures under this task's
# own traffic on this stack. Re-check it if the vLLM version or the attention backend
# changes.

import shutil
import subprocess
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.config import KVTransferConfig

from tasks.agentic_coding.serving_config import (
    CHAT_TEMPLATE_KWARGS_FLAG,
    TEMPERATURE,
    TOP_LOGPROBS,
    TokenText,
    chat_template_kwargs_arg,
    enable_prefix_caching,
    enforce_eager,
    engine_args,
    finish_reason,
    gpu_memory_utilization,
    hls_context,
    kv_cpu_offload_bytes,
    logprob_entry,
    max_context_len,
    max_num_seqs,
    request_max_tokens,
    response_usage,
    tensor_parallel_size,
)

# The served checkpoint. Already FP8 on the Hub (the BF16 variant carries the
# suffix), so the reference pins the same weights the artifact serves.
MODEL_ID = "zai-org/GLM-5.3-Flash"
#: The task's chat-template pin (ModelSpec.chat_template_kwargs); the gate replays every
#: prompt rendered with it, so this must equal the task's value.
#: A launch that serves another task's model overrides it with --chat-template-kwargs.
CHAT_TEMPLATE_KWARGS: dict = {}

# The harness stamps the workload's max_tokens onto every request, so artifact and
# reference cap alike; serving_config's default covers probes and manual calls.
# Override it with HLS_MAX_NEW_TOKENS when launching outside HLS.

# No stop strings: a reply ends where the model ends its turn. See the module
# docstring -- the gate's end-of-turn count reads exactly where turns ended.
_STOP_STRINGS: tuple[str, ...] = ()


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
    #: Tool definitions this turn offers. The chat template folds them into the
    #: prompt, so a turn that carried them and a turn that did not are different
    #: prompts; dropping them here would serve a prompt the reference never renders.
    tools: list[dict] | None = None
    #: Always "none" from this harness, and honoured by doing nothing: the definitions
    #: belong in the prompt, and the model's call stays in `content` where the loop
    #: reads it. No tool call is ever extracted server-side.
    tool_choice: str | None = None
    #: Resume the final assistant message instead of opening a new turn. The harness
    #: sets it on the second request of a turn whose thought ran to its cap: the cut
    #: reply comes back as the last message, ending in ``</think>``, and the model goes
    #: on from there. The gate teacher-forces what is served under a render that ends
    #: at that text, so dropping this field would serve the model a fresh generation
    #: prompt the reference never renders, and every resumed position would diverge.
    continue_final_message: bool = False
    #: Whether the render ends in a generation prompt. Unset means the opposite of
    #: ``continue_final_message``: a new turn opens unless a message is resumed.
    add_generation_prompt: bool | None = None


class _BadRequest(ValueError):
    """A request this server refuses with 400 before rendering anything."""


def _adds_generation_prompt(req: _ChatRequest) -> bool:
    """Whether this request's prompt ends in a generation prompt.

    Resuming the final message and opening a new turn after it are two different
    prompts, so a request asking for both is refused, as vLLM refuses it."""
    if req.add_generation_prompt is None:
        return not req.continue_final_message
    if req.add_generation_prompt and req.continue_final_message:
        raise _BadRequest("continue_final_message and add_generation_prompt cannot "
                          "both be true: a prompt either resumes the final message or "
                          "opens a new turn after it")
    return req.add_generation_prompt


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


def _build_engine() -> AsyncLLMEngine:
    args = engine_args()
    spec = args.pop("kv_transfer_config", None)
    if spec is not None:
        # serving_config keeps this a plain mapping so it imports without vLLM; the
        # engine takes a KVTransferConfig object built from it.
        args["kv_transfer_config"] = KVTransferConfig(**spec)
    return AsyncLLMEngine.from_engine_args(
        AsyncEngineArgs(model=_weights_src(), **args))


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
        stop=list(_STOP_STRINGS),
        include_stop_str_in_output=True,
        # Keep special tokens: the loop strips the reasoning span out of the reply
        # text itself, so it has to be able to see the delimiters. Dropping them here
        # would also make the served text disagree with the ids the gate replays.
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


def _prompt(req: _ChatRequest) -> list[int]:
    """The prompt ids the engine is fed for this request.

    Tokenized here without adding special tokens, because the render already carries
    them, and handed over as ids so the engine cannot tokenize the text differently.
    Raises ``_BadRequest`` before rendering a request that contradicts itself."""
    add_generation_prompt = _adds_generation_prompt(req)
    tok = _STATE["tokenizer"]
    messages = [m.model_dump() for m in req.messages]
    # `tools` only when the turn carries them: passing an empty list is not the same
    # as passing none, because the template renders a tools preamble either way and a
    # prompt with an empty <tools> block is one the reference never produces.
    kwargs = dict(_STATE["chat_template_kwargs"])
    if req.tools:
        kwargs["tools"] = req.tools
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=req.continue_final_message,
        **kwargs)
    return tok(text, add_special_tokens=False).input_ids


def _hls_context(max_ctx: int, request_max_tokens: int) -> dict:
    return hls_context(
        max_ctx,
        request_max_tokens,
        prefix_caching=enable_prefix_caching(),
        eager=enforce_eager(),
        num_seqs=max_num_seqs(),
        tensor_parallel=tensor_parallel_size(),
        gpu_memory_util=gpu_memory_utilization(),
        kv_cpu_offload_bytes=kv_cpu_offload_bytes() or None,
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Build the engine inside the running event loop so the background batching
    # loop binds to uvicorn's loop. uvicorn does not accept connections until
    # startup returns, so the readiness probe retries during the (slow) load.
    tok = AutoTokenizer.from_pretrained(_weights_src(), trust_remote_code=True)
    engine = _build_engine()
    # Warmed on ids, the way every request is fed, so an engine that refuses a token
    # prompt fails here rather than on the first graded turn.
    warm_ids = tok(tok.apply_chat_template(
        [{"role": "user", "content": "_"}],
        tokenize=False, add_generation_prompt=True,
        **_STATE["chat_template_kwargs"]), add_special_tokens=False).input_ids
    async for _ in engine.generate({"prompt_token_ids": warm_ids}, _sampling(16),
                                   "warmup-0"):
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
        try:
            prompt_ids = _prompt(req)
        except _BadRequest as exc:
            return JSONResponse(status_code=400, content={"error": {
                "message": str(exc), "type": "BadRequestError", "code": 400}})
        mcl = max_context_len()
        if getattr(req, "stream", False):
            return _stream_response(req, prompt_ids, max_new, mcl)
        return await _unary_response(req, prompt_ids, max_new, mcl)

    return app


async def _unary_response(req: _ChatRequest, prompt_ids: list[int], max_new: int,
                          max_context_len: int) -> dict:
    engine: AsyncLLMEngine = _STATE["engine"]
    final = None
    async for out in engine.generate({"prompt_token_ids": prompt_ids}, _sampling(max_new),
                                     f"req-{uuid.uuid4().hex}"):
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
        "usage": response_usage(len(prompt_ids), len(token_ids)),
        "hls_context": _hls_context(max_context_len, max_new),
    }
    gpu = _GPU.snapshot()
    if gpu is not None:
        body["gpu"] = gpu
    return body


def _stream_response(req: _ChatRequest, prompt_ids: list[int], max_new: int,
                     max_context_len: int) -> StreamingResponse:
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
        async for out in engine.generate({"prompt_token_ids": prompt_ids},
                                         _sampling(max_new), f"req-{uuid.uuid4().hex}"):
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
        final["usage"] = response_usage(len(prompt_ids), len(prev_ids))
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
