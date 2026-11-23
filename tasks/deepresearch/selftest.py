#!/usr/bin/env python3
"""Drive the deepresearch research loop against a server you are building.

Runs dev questions through the same loop the grader runs, with the development
retrieval bundle, and prints what your server saw: calls per trajectory, prompt and
completion sizes, per-call latency, makespan, and exact match against the dev answers.
A trajectory whose request fails is counted and reported, not fatal to the rest.
Every request body is the grader's, byte for byte.

    python selftest.py --server http://localhost:8000 --n 8 --concurrency 8

Reads the mounted dev split at /data/tenants/t0/dev/rows.json and the retrieval
bundle at /data/tools/browsecomp, or wherever HLS_TOOLS_DIR points (override with
--rows and --tools-dir). Retrieval runs on this machine's GPU when it has 40 GB free,
else on CPU (--device forces one). At grading time retrieval runs in a separate
harness container on its own GPU; your server never shares a device with it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # /app, so `tasks.` resolves
from tasks.deepresearch.corpus_browser import CorpusBrowserTool  # noqa: E402
from tasks.deepresearch.limits import MAX_NEW_TOKENS  # noqa: E402
from tasks.deepresearch.research_loop import (  # noqa: E402
    CHAT_FORMAT, DEEPRESEARCH_SYSTEM, FIND_TOOL_NAME, OPEN_TOOL_NAME, SEARCH_TOOL_NAME,
    ResearchLoop,
)
from tasks.deepresearch.retriever import CorpusRetriever, bundle_dir  # noqa: E402

MODEL = "served-model"
MAX_TURNS = 100
MAX_SEARCH_TURNS = 80
# Provisioning mounts a tenant's split at /data/tenants/<id>/<split>/rows.json, and a
# single-tenant task is tenant t0 (hls.core.seeds.DEFAULT_TENANT_ID, not imported here:
# this file runs without the harness).
TENANTS_DIR = Path("/data/tenants")
DEFAULT_TENANT_ID = "t0"
DEFAULT_ROWS = str(TENANTS_DIR / DEFAULT_TENANT_ID / "dev" / "rows.json")
DEFAULT_TOOLS_DIR = str(bundle_dir())
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b")
_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def graded_loop(chat_format: str) -> ResearchLoop:
    """The loop configured as the graded run configures it, for one chat format.

    The chat format is a command-line argument because this file runs in the build
    sandbox, where the task's model spec is not readable; the brief publishes the
    served model's format.
    """
    if chat_format != CHAT_FORMAT:
        raise ValueError(f"the research loop reads chat_format {CHAT_FORMAT!r}, not {chat_format!r}")
    return ResearchLoop(finalize_after_search_turns=MAX_SEARCH_TURNS)


def resolve_rows(rows: str | None, tenants_dir: Path = TENANTS_DIR) -> Path:
    """The mounted dev split, or the path given on the command line.

    A single-tenant task's rows are tenant t0's. A task whose one tenant carries
    another id is found by glob; more than one has to be named with --rows.
    """
    if rows:
        return Path(rows)
    default = tenants_dir / DEFAULT_TENANT_ID / "dev" / "rows.json"
    if default.exists():
        return default
    matches = sorted(tenants_dir.glob("*/dev/rows.json"))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f"no dev rows: {default} is absent and {tenants_dir}/*/dev/rows.json matched "
        f"{len(matches)} files; name the file with --rows")


#: The fields beyond role and content that a chat template reads off a message: the
#: model's own tool calls, the reasoning it wrote before them (spelled both ways, as
#: templates and servers differ on which they read), and the id and tool name a tool
#: result answers with. The driver sends them, so this file sends them. Repeated from
#: ``hls.spec.drivers.CHAT_MESSAGE_FIELDS`` rather than imported, because this file runs
#: in the build sandbox without the harness; a test pins the two lists together.
CHAT_MESSAGE_FIELDS = ("reasoning", "reasoning_content", "tool_calls", "tool_call_id",
                       "name")
#: The alternatives the driver asks for at every generated position, beside the token's own
#: log-probability. Repeated from the task's contract for the same reason; a test holds this
#: file's body to the driver's.
TOP_LOGPROBS = 21


def chat_message(m) -> dict:
    """One history message the way the driver puts it on the wire."""
    content = m["content"]
    out = {"role": str(m["role"]),
           "content": content if content is None or isinstance(content, str)
           else str(content)}
    out.update({k: m[k] for k in CHAT_MESSAGE_FIELDS if k in m})
    return out


def build_request(messages, tools, max_tokens, model: str = MODEL) -> dict:
    """The driver's chat-completion body (OpenAIChatProtocol.encode_request, streaming)."""
    body = {"model": model, "messages": [chat_message(m) for m in messages]}
    if isinstance(tools, list) and tools:
        body["tools"] = tools
    if isinstance(max_tokens, int) and max_tokens > 0:
        body["max_tokens"] = max_tokens
    body["return_token_ids"] = True
    body["logprobs"] = True
    body["top_logprobs"] = TOP_LOGPROBS
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    return body


async def stream_completion(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    """POST one request and fold its SSE stream the way the driver does."""
    t0 = time.monotonic()
    first = None
    parts: list[str] = []
    n_tokens = 0
    finish = None
    missing_ids = False
    server_error = None
    usage: dict = {}
    ctx: dict = {}
    async with client.stream("POST", url, json=body, timeout=None) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if first is None:
                first = time.monotonic()
            usage.update(chunk.get("usage") or {})
            if isinstance(chunk.get("hls_context"), dict):
                ctx = chunk["hls_context"]
            choices = chunk.get("choices") or []
            if not choices:
                # A frame with no choices carries no tokens, but it may carry the
                # server's own account of what went wrong: a server that dies
                # mid-stream sends it on the already committed 200 and nothing else.
                err = chunk.get("error")
                if err is not None:
                    server_error = err.get("message") if isinstance(err, dict) else err
                continue
            for choice in choices:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    parts.append(delta["content"])
                    # The driver rejects a content delta that carries no ids, so a
                    # stream that drops them anywhere fails the run.
                    if not choice.get("token_ids"):
                        missing_ids = True
                if choice.get("token_ids"):
                    n_tokens += len(choice["token_ids"])
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    metadata = {"finish_reason": finish,
                "completion_tokens": usage.get("completion_tokens", n_tokens),
                "prompt_tokens": usage.get("prompt_tokens"),
                "max_tokens": body.get("max_tokens")}
    for key in ("request_max_tokens", "max_context_len"):
        if key in ctx:
            metadata[key] = ctx[key]
    if server_error is not None:
        metadata["error"] = server_error
    return {"text": "".join(parts), "metadata": metadata,
            "latency": time.monotonic() - t0,
            "ttfp": (first - t0) if first is not None else None,
            "protocol_ok": (finish is not None and n_tokens > 0 and not missing_ids
                            and server_error is None)}


def _normalize(text: str) -> str:
    """SQuAD-style normalization: lowercase, drop articles, punctuation to spaces,
    collapse whitespace. The rule the graded answer metric scores with."""
    out = str(text).lower()
    out = _ARTICLES_RE.sub(" ", out)
    out = _PUNCT_RE.sub(" ", out)
    return _SPACE_RE.sub(" ", out).strip()


async def run_trajectory(client, url, question, gold, tool: CorpusBrowserTool, *,
                         loop: ResearchLoop, max_tokens: int, max_turns: int,
                         session_id: str) -> dict:
    """Run one question to an answer and record what the server was asked for."""
    turns: list[dict] = []
    tool_calls: list[str] = []
    server_errors: list[str] = []
    protocol_failures = 0

    async def complete(request):
        nonlocal protocol_failures
        # The driver stamps the cap per request: the loop's own smaller cap on a
        # forced-final turn when it set one, else the workload's output cap.
        out = await stream_completion(client, url, build_request(
            request["metadata"]["messages"], request["metadata"].get("tools"),
            request["metadata"].get("max_tokens", max_tokens)))
        if not out["protocol_ok"]:
            protocol_failures += 1
        reported = out["metadata"].get("error")
        if reported is not None and str(reported) not in server_errors:
            server_errors.append(str(reported))
        return out

    async def run_tool(name, args):
        tool_calls.append(name)
        fn = {SEARCH_TOOL_NAME: tool.search, OPEN_TOOL_NAME: tool.open, FIND_TOOL_NAME: tool.find}[name]
        return await fn(**args)

    async def emit(turn):
        turns.append(turn)

    t0 = time.monotonic()
    seed_text = f"Question: {question}\n"
    await loop.run(session_id=session_id, seed_text=seed_text,
                   seed_messages=[{"role": "system", "content": DEEPRESEARCH_SYSTEM},
                                  {"role": "user", "content": seed_text}],
                   base_metadata={"tenant_id": "selftest", "adapter_id": None,
                                  "session_id": session_id},
                   complete=complete, run_tool=run_tool, emit=emit, max_turns=max_turns)
    answer = (turns[-1]["extras"].get("final_answer") if turns else None) or ""
    prompt_tokens = [t["completion"]["metadata"].get("prompt_tokens") or 0 for t in turns]
    return {"calls": len(turns), "searches": tool_calls.count(SEARCH_TOOL_NAME),
            "opens": tool_calls.count(OPEN_TOOL_NAME),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": [t["completion"]["metadata"].get("completion_tokens") or 0 for t in turns],
            "latencies": [t["completion"]["latency"] for t in turns],
            "makespan": time.monotonic() - t0,
            "terminal_prompt_tokens": prompt_tokens[-1] if prompt_tokens else 0,
            "answer": answer, "gold": gold, "protocol_failures": protocol_failures,
            "server_errors": server_errors}


def _error_record(gold, exc: Exception, makespan: float) -> dict:
    """A trajectory that never finished, in the shape summarize reads."""
    return {"calls": 0, "searches": 0, "opens": 0, "prompt_tokens": [],
            "completion_tokens": [], "latencies": [], "makespan": makespan,
            "terminal_prompt_tokens": 0, "answer": "", "gold": gold,
            "protocol_failures": 0, "server_errors": [],
            "error": f"{type(exc).__name__}: {exc}"}


async def guarded_trajectory(client, url, question, gold, tool: CorpusBrowserTool, *,
                             loop: ResearchLoop, max_tokens: int, max_turns: int,
                             session_id: str) -> dict:
    """One trajectory, recording a failure instead of raising it.

    A server that dies partway through one question would otherwise take every other
    trajectory's record down with it.
    """
    t0 = time.monotonic()
    try:
        return await run_trajectory(client, url, question, gold, tool, loop=loop,
                                    max_tokens=max_tokens, max_turns=max_turns,
                                    session_id=session_id)
    except Exception as exc:
        return _error_record(gold, exc, time.monotonic() - t0)


def _quantile(values: list, p: float):
    """Quantile of an already sorted list, interpolating between neighbours."""
    if not values:
        return 0
    pos = p * (len(values) - 1)
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)


def _dist(values) -> dict:
    """p50, p95 and max of one quantity, ignoring what the server never reported."""
    clean = sorted(v for v in values if v is not None)
    return {"p50": _quantile(clean, 0.5), "p95": _quantile(clean, 0.95),
            "max": clean[-1] if clean else 0}


def _matches(answer, gold) -> bool:
    """The graded rule: normalized equality, and an empty answer never counts."""
    normalized = _normalize(answer)
    return normalized != "" and normalized == _normalize(gold)


def summarize(records: list[dict]) -> dict:
    """Reduce per-trajectory records to the numbers a serving report quotes.

    ``n`` counts every trajectory attempted and ``errors`` those that raised before an
    answer. Every other quantity describes only the trajectories that finished: a
    request that never came back is a serving fault, and scoring it as a wrong answer
    would report it as an accuracy loss instead.
    """
    done = [r for r in records if not r.get("error")]
    per_call_prompt = [v for r in done for v in r["prompt_tokens"]]
    per_call_completion = [v for r in done for v in r["completion_tokens"]]
    per_call_latency = [v for r in done for v in r["latencies"]]
    return {"n": len(records),
            "errors": len(records) - len(done),
            "exact_match": (sum(_matches(r["answer"], r["gold"]) for r in done)
                            / len(done)) if done else 0.0,
            "protocol_failures": sum(r["protocol_failures"] for r in done),
            "calls": _dist([r["calls"] for r in done]),
            "searches": _dist([r["searches"] for r in done]),
            "opens": _dist([r["opens"] for r in done]),
            "prompt_tokens": _dist(per_call_prompt),
            "completion_tokens": _dist(per_call_completion),
            "terminal_prompt_tokens": _dist([r["terminal_prompt_tokens"] for r in done]),
            "latency_sec": _dist(per_call_latency),
            "makespan": _dist([r["makespan"] for r in done])}


async def _main_async(args) -> int:
    rows_path = resolve_rows(args.rows)
    rows = json.loads(rows_path.read_text())
    print(f"rows: {len(rows)} dev questions from {rows_path}", flush=True)
    rng = random.Random(args.seed)
    picked = rng.sample(rows, min(args.n, len(rows)))
    tool = CorpusBrowserTool(retriever=CorpusRetriever(Path(args.tools_dir),
                                                       device=None if args.device == "auto" else args.device))
    t0 = time.monotonic()
    await tool.warmup()
    print(f"retriever: {tool.retriever.n_docs} docs on {tool.retriever.device} "
          f"(loaded in {time.monotonic() - t0:.0f}s)", flush=True)
    loop = graded_loop(args.chat_format)
    sem = asyncio.Semaphore(args.concurrency)
    url = args.server.rstrip("/") + "/v1/chat/completions"

    async def one(client, i, row):
        async with sem:
            rec = await guarded_trajectory(
                client, url, row["query"], row["answer"], tool, loop=loop,
                max_tokens=args.max_tokens, max_turns=args.max_turns, session_id=f"selftest.{i}")
            if rec.get("error"):
                print(f"[{i}] FAILED after {rec['makespan']:.0f}s: {rec['error']}", flush=True)
            else:
                print(f"[{i}] calls={rec['calls']} searches={rec['searches']} opens={rec['opens']} "
                      f"terminal_prompt={rec['terminal_prompt_tokens']} makespan={rec['makespan']:.0f}s "
                      f"answer={rec['answer']!r} gold={rec['gold']!r}", flush=True)
                if rec["server_errors"]:
                    print(f"[{i}] server reported: {'; '.join(rec['server_errors'])}", flush=True)
            return rec

    async with httpx.AsyncClient() as client:
        records = await asyncio.gather(*(one(client, i, row) for i, row in enumerate(picked)))
    summary = summarize(records)
    print(json.dumps(summary, indent=2))
    # Nothing finished, so there is nothing to read: fail rather than print an
    # exact match of 0.0 over an empty set and exit clean.
    return 1 if records and summary["errors"] == len(records) else 0


def _at_least_one(value: str) -> int:
    """An int argument that has to be 1 or more (argparse has no minimum)."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {parsed}")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """The command line. Separate from ``main`` so a test can read the defaults."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--concurrency", type=_at_least_one, default=8)
    p.add_argument("--rows", default=None,
                   help=f"dev split to read (default: {DEFAULT_ROWS})")
    p.add_argument("--tools-dir", default=DEFAULT_TOOLS_DIR,
                   help=f"retrieval bundle (default: {DEFAULT_TOOLS_DIR})")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--max-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--max-turns", type=int, default=MAX_TURNS)
    p.add_argument("--chat-format", choices=[CHAT_FORMAT], default=CHAT_FORMAT,
                   help="the served model's chat format, which decides how replies "
                        f"are read (default: {CHAT_FORMAT})")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
