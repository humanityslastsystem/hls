"""Artifact server for deepresearch -- the Gemma 4 deep-research baseline (naive HF).

A dependency-light alternative to ``vllm_server.py`` (the elastic-inference
mirror). Loads the model with plain ``transformers`` and samples with ``generate``
-- no continuous batching, no prefix caching -- so it is the "naive" floor the
implementer agent (or the vLLM baseline) is meant to beat on serving efficiency.
Same OpenAI surface + per-turn trust semantics as ``vllm_server.py``.

DECODING: one draw per position from the model's whole next-token distribution at
temperature 1 -- no top-p, no top-k, no min-p, no penalties. The task gates on the
distribution the server reports, scored against a reference sampling the same way,
so a server that truncates or cools its sampler fails the gate by construction.

REPORT: every response carries OpenAI ``logprobs.content[]``, one entry per
generated token, holding that token's own log-probability and the 21 most probable
alternatives at that position, each with its integer vocabulary id.
``_ReportStreamer`` below reads them off the same rows ``generate`` samples from.

STREAMING (http-sse): when the request sets ``stream`` the server emits an
OpenAI SSE stream of ``chat.completion.chunk`` objects, one per generated token,
each carrying its ``token_ids``, a prefix-diffed text delta and that token's
logprob entry, terminated by ``data: [DONE]``. This lets the harness measure
per-step TTFT/ITL while the reassembled ids and report are the ones the gate reads.
A non-stream request still returns one unary completion (the substrate readiness
probe).

Swap the artifact image entrypoint in ``task.py`` between this and
``vllm_server.py`` depending on whether you want the naive or the
prefix-caching baseline. ``MODEL_ID`` is the single model seam (see task.py). A
lightweight ``_GpuSampler`` stamps a GPU snapshot onto each response for the
report-only serving metric.
"""

from __future__ import annotations
import argparse
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList,
)
from transformers.generation.streamers import BaseStreamer

from tasks.deepresearch.serving_config import (
    CHAT_TEMPLATE_KWARGS_FLAG,
    TEMPERATURE,
    TOP_LOGPROBS,
    TokenText,
    chat_template_kwargs_arg,
    finish_reason,
    hls_context,
    logprob_entry,
    max_context_len,
    request_max_tokens,
    request_tools,
    response_usage,
)

#: The task's checkpoint, loaded by name when no weights are mounted.
MODEL_ID = "google/gemma-4-31B-it"
#: The chat-template arguments a launch renders prompts with when it carries no
#: --chat-template-kwargs: none. The task's launch passes its pin with that flag.
CHAT_TEMPLATE_KWARGS: dict = {}

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
_STATE: dict = {"ready": False, "tokenizer": None, "model": None, "device": "cpu",
                "token_text": None,
                "chat_template_kwargs": dict(CHAT_TEMPLATE_KWARGS)}
_GEN_LOCK = threading.Lock()
_LOG_PROGRESS_EVERY = int(os.environ.get("HLS_GENERATION_PROGRESS_INTERVAL", "256"))


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


class _ReportStreamer(LogitsProcessor, BaseStreamer):
    """Pairs each sampled token with the distribution it was drawn from.

    ``generate`` runs its logits processors, samples, then hands the token to the
    streamer, so one object on both seams sees a position's distribution and then
    the token that came out of it. It records the log-softmax row as a processor,
    keeps it until ``put`` names the token, and enqueues the OpenAI logprob entry
    beside the id. The queue makes the streaming path work: one thread generates,
    another yields SSE frames.

    The row this sees is the row ``generate`` samples from because this server
    reshapes the distribution in no way (see ``_generate_kwargs``). A server that
    added a warper would have to record after it, since transformers appends its
    sampling warpers behind any caller-supplied processor.

    The first ``put`` carries the prompt and is skipped; ``end`` enqueues a sentinel
    that stops iteration.
    """

    def __init__(self) -> None:
        self._q: queue.Queue = queue.Queue()
        self._pending = None
        self._prompt_seen = False
        self._ended = False

    def __call__(self, input_ids, scores):
        # fp32: the row is log-softmaxed and then indexed, and a bf16 row's ties
        # would make the top alternatives depend on rounding.
        self._pending = torch.log_softmax(scores[0].float(), dim=-1)
        return scores

    def put(self, value) -> None:
        ids = value.tolist() if hasattr(value, "tolist") else value
        if isinstance(ids, list) and ids and isinstance(ids[0], list):
            ids = ids[0]
        if not self._prompt_seen:
            self._prompt_seen = True
            return
        if isinstance(ids, int):
            ids = [ids]
        for tid in ids:
            self._q.put((int(tid), self._entry(int(tid))))

    def _entry(self, token_id: int) -> dict:
        """The logprob entry for the token just sampled, from the row it came from."""
        row, self._pending = self._pending, None
        if row is None:
            raise RuntimeError(
                f"token {token_id} arrived with no recorded distribution; generate "
                "sampled a position whose logits this processor never saw.")
        top = torch.topk(row, TOP_LOGPROBS)
        alternatives = list(zip(top.indices.tolist(), top.values.tolist()))
        return logprob_entry(token_id, row[token_id].item(), alternatives,
                             _STATE["token_text"])

    def end(self) -> None:
        if not self._ended:
            self._ended = True
            self._q.put(None)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            yield item


def _weights_src() -> str:
    wdir = os.environ.get("HLS_WEIGHTS_DIR")
    if wdir and Path(wdir).is_dir():
        return wdir
    return MODEL_ID


def _load_model() -> None:
    if torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
    src = _weights_src()
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    # The baseline serves the model's language-model path (text-only); load
    # the causal-LM head. trust_remote_code mirrors the elastic-inference serve
    # scripts' loader.
    mdl = AutoModelForCausalLM.from_pretrained(src, dtype=dtype, trust_remote_code=True)
    mdl.to(device).eval()
    _STATE.update(
        tokenizer=tok, model=mdl, device=device, ready=True,
        token_text=TokenText(lambda tid: tok.decode([tid], skip_special_tokens=False)))
    _GPU.start()
    print(f"[artifact] model {src} loaded on {device} ({dtype})", flush=True)


def _prompt_inputs(req: _ChatRequest):
    tok = _STATE["tokenizer"]
    dev = _STATE["device"]
    max_new = request_max_tokens(getattr(req, "max_tokens", None))
    messages = [m.model_dump() for m in req.messages]
    tools = request_tools(getattr(req, "tools", None), [_SEARCH_TOOL])
    prompt = tok.apply_chat_template(
        messages, tools=tools,
        tokenize=False, add_generation_prompt=True,
        **_STATE["chat_template_kwargs"])
    inputs = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    return inputs, max_new, prompt_tokens, max_context_len()


def _generate_kwargs(inputs, max_new: int, report: _ReportStreamer) -> dict:
    """The decode both response paths run: one draw per position from the whole
    distribution, with ``report`` wired in on both of its seams.

    Every knob that would reshape the distribution is passed explicitly rather than
    left to the checkpoint's own ``generation_config``, which ships non-neutral
    sampling defaults for some models and would otherwise apply silently.
    ``min_p=None`` is how transformers says "no minimum-probability cutoff".
    """
    tok = _STATE["tokenizer"]
    return dict(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=TEMPERATURE,
        top_p=1.0,
        top_k=0,                 # transformers' "keep the whole vocabulary"
        min_p=None,
        repetition_penalty=1.0,
        pad_token_id=tok.eos_token_id,
        logits_processor=LogitsProcessorList([report]),
        streamer=report,
    )


def _build_app() -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    def health():
        if _STATE["ready"]:
            return {"status": "ok"}
        return Response(status_code=503, content="loading")

    @app.post("/v1/chat/completions")
    def chat(req: _ChatRequest):
        inputs, max_new, prompt_tokens, max_ctx = _prompt_inputs(req)
        if getattr(req, "stream", False):
            return _stream_response(req, inputs, max_new, prompt_tokens, max_ctx)
        return _unary_response(req, inputs, max_new, prompt_tokens, max_ctx)

    return app


def _unary_response(
    req: _ChatRequest,
    inputs,
    max_new: int,
    prompt_tokens: int,
    max_ctx: int,
) -> dict:
    tok = _STATE["tokenizer"]
    mdl = _STATE["model"]
    started = time.time()
    print(
        f"[artifact] unary request prompt_tokens={prompt_tokens} max_new={max_new} "
        f"max_context={max_ctx}",
        flush=True,
    )
    report = _ReportStreamer()
    with _GEN_LOCK, torch.no_grad():
        out = mdl.generate(**_generate_kwargs(inputs, max_new, report))
    new_ids = out[0, inputs["input_ids"].shape[-1]:].tolist()
    # generate calls end() as it returns, so the queue is complete and draining it
    # here never blocks.
    content = [entry for _, entry in report]
    if len(content) != len(new_ids):
        raise RuntimeError(
            f"reported {len(content)} positions for {len(new_ids)} served tokens; "
            "the report must cover every token.")
    # Keep special tokens: the loop reads the model's end-of-turn and
    # reasoning/tool-call delimiters out of this text (Gemma's <turn|>,
    # <|channel>/<channel|>, <|tool_call>/<tool_call|>).
    decoded = tok.decode(new_ids, skip_special_tokens=False)
    print(
        f"[artifact] unary complete completion_tokens={len(new_ids)} "
        f"elapsed_sec={time.time() - started:.3f}",
        flush=True,
    )
    body = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": decoded},
                     "finish_reason": finish_reason(len(new_ids), max_new),
                     "token_ids": new_ids,
                     "logprobs": {"content": content}}],
        "usage": response_usage(prompt_tokens, len(new_ids)),
        "hls_context": hls_context(max_ctx, max_new),
    }
    gpu = _GPU.snapshot()
    if gpu is not None:
        body["gpu"] = gpu
    return body


def _stream_response(
    req: _ChatRequest,
    inputs,
    max_new: int,
    prompt_tokens: int,
    max_ctx: int,
) -> StreamingResponse:
    """SSE: one chunk per generated token (token_ids + text delta + that token's
    logprob entry), then a terminal chunk with finish_reason + usage + GPU snapshot.

    The harness concatenates the per-chunk ``token_ids`` and ``logprobs.content``
    across the stream, so the reassembled reply carries the same report as the unary
    path.
    """
    tok = _STATE["tokenizer"]
    mdl = _STATE["model"]
    report = _ReportStreamer()
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    started = time.time()
    print(
        f"[artifact] stream request prompt_tokens={prompt_tokens} max_new={max_new} "
        f"max_context={max_ctx}",
        flush=True,
    )

    def _run_generate() -> None:
        try:
            with _GEN_LOCK, torch.no_grad():
                mdl.generate(**_generate_kwargs(inputs, max_new, report))
        finally:
            report.end()

    def _chunk(delta: dict, token_ids, finish_reason, logprobs=None):
        choice: dict = {"index": 0, "delta": delta, "token_ids": token_ids,
                        "finish_reason": finish_reason}
        if logprobs is not None:
            choice["logprobs"] = logprobs
        return {"id": cmpl_id, "object": "chat.completion.chunk", "created": created,
                "model": req.model, "choices": [choice]}

    def _events():
        thread = threading.Thread(target=_run_generate, daemon=True)
        thread.start()
        ids_so_far: list[int] = []
        prev_text = ""
        first = True
        for tid, entry in report:
            ids_so_far.append(tid)
            if _LOG_PROGRESS_EVERY > 0 and len(ids_so_far) % _LOG_PROGRESS_EVERY == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"[artifact] stream progress completion_tokens={len(ids_so_far)} "
                    f"elapsed_sec={elapsed:.3f} tok_per_sec={len(ids_so_far) / elapsed:.3f}",
                    flush=True,
                )
            # Keep special tokens, as the unary path does.
            full = tok.decode(ids_so_far, skip_special_tokens=False)
            delta_text = full[len(prev_text):]
            prev_text = full
            delta = {"content": delta_text}
            if first:
                delta = {"role": "assistant", "content": delta_text}
                first = False
            yield (f"data: {json.dumps(_chunk(delta, [tid], None, {'content': [entry]}))}"
                   "\n\n")
        thread.join()
        final = _chunk({}, None, finish_reason(len(ids_so_far), max_new))
        elapsed = max(time.time() - started, 1e-6)
        print(
            f"[artifact] stream complete completion_tokens={len(ids_so_far)} "
            f"elapsed_sec={elapsed:.3f} tok_per_sec={len(ids_so_far) / elapsed:.3f}",
            flush=True,
        )
        final["usage"] = response_usage(prompt_tokens, len(ids_so_far))
        final["hls_context"] = hls_context(max_ctx, max_new)
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
    _load_model()
    uvicorn.run(_build_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
