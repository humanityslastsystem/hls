"""The sandbox runner sends what the driver sends and reads what the driver reads."""
from __future__ import annotations

import asyncio
import json
import shlex
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.core.seeds import DEFAULT_TENANT_ID
from hls.core.trace import ArtifactResponse
from hls.spec.drivers import HarnessContext
from hls.spec.tools import LocalCallableToolRunner
from tasks.deepresearch import selftest as st
from tasks.deepresearch.metric import normalize_answer
from tasks.deepresearch.research_loop import CHAT_FORMAT, DEEPRESEARCH_SYSTEM
from tasks.deepresearch import task as t
from tests.unit.brief_task import render_brief


def test_request_body_is_the_drivers_body_byte_for_byte():
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    tools = [{"type": "function", "function": {"name": "web_search"}}]
    env = Envelope(request_id="r", contents=[TextPayload(text="q")],
                   metadata={"messages": messages, "tools": tools, "max_tokens": 123})
    expected = t.CONTRACT.protocol.encode_request(env, stream=True)
    assert st.build_request(messages, tools, 123) == expected
    assert json.dumps(st.build_request(messages, tools, 123), sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_request_body_carries_a_tool_calling_turn_the_way_the_driver_does():
    """A Gemma turn's history carries the model's structured call, the result in the
    tool role and the reasoning; the self-test has to send those too, or a builder
    tests against a prompt the graded run never sends."""
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "reasoning_content": "I need evidence.",
         "reasoning": "I need evidence.",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": "web_search",
                                      "arguments": {"query": "founder"}}}]},
        {"role": "tool", "tool_call_id": "call_0", "name": "web_search",
         "content": '{"documents": []}'},
    ]
    env = Envelope(request_id="r", contents=[TextPayload(text="q")],
                   metadata={"messages": messages, "max_tokens": 123})
    expected = t.CONTRACT.protocol.encode_request(env, stream=True)
    assert st.build_request(messages, None, 123) == expected
    assert expected["messages"] == messages


def test_the_self_test_and_the_driver_whitelist_the_same_fields():
    """The two lists are separate because this file runs without the harness; a
    difference between them is a difference between the bodies they send."""
    from hls.spec.drivers import CHAT_MESSAGE_FIELDS
    assert st.CHAT_MESSAGE_FIELDS == CHAT_MESSAGE_FIELDS


def test_constants_match_the_task():
    assert st.MODEL == t.CONTRACT.protocol.model
    assert st.MAX_NEW_TOKENS == t.MAX_NEW_TOKENS
    assert st.MAX_TURNS == t.MAX_TURNS
    assert st.MAX_SEARCH_TURNS == t.MAX_SEARCH_TURNS
    graded = st.graded_loop(t.MODEL.chat_format)
    assert graded.finalize_after_search_turns == t.AGENT.framework.finalize_after_search_turns
    assert t.AGENT.framework.chat_format == CHAT_FORMAT
    assert st.DEFAULT_TENANT_ID == DEFAULT_TENANT_ID
    assert st.DEFAULT_ROWS == f"/data/tenants/{DEFAULT_TENANT_ID}/dev/rows.json"


def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})


def test_stream_completion_folds_chunks_like_the_driver():
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}, "token_ids": [1, 2]}], "hls_context": {"request_max_tokens": 50}},
        {"choices": [{"delta": {"content": "lo"}, "token_ids": [3], "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
    ]
    transport = httpx.MockTransport(lambda req: _sse(chunks))

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await st.stream_completion(client, "http://s/v1/chat/completions",
                                              st.build_request([{"role": "user", "content": "q"}], [], 50))

    out = asyncio.run(go())
    assert out["text"] == "Hello" and out["protocol_ok"]
    assert out["metadata"]["completion_tokens"] == 3 and out["metadata"]["prompt_tokens"] == 7
    assert out["metadata"]["request_max_tokens"] == 50 and out["metadata"]["finish_reason"] == "stop"
    assert out["latency"] >= 0 and out["ttfp"] is not None


def test_summarize_reports_the_benchmark_report_quantities():
    records = [
        {"calls": 3, "searches": 2, "opens": 1, "prompt_tokens": [100, 200, 300],
         "completion_tokens": [10, 20, 30], "latencies": [0.1, 0.2, 0.3], "makespan": 2.0,
         "terminal_prompt_tokens": 300, "answer": "Paris", "gold": "Paris", "protocol_failures": 0},
        {"calls": 1, "searches": 0, "opens": 0, "prompt_tokens": [50], "completion_tokens": [5],
         "latencies": [0.5], "makespan": 0.5, "terminal_prompt_tokens": 50, "answer": "Rome",
         "gold": "Paris", "protocol_failures": 1},
    ]
    s = st.summarize(records)
    assert s["n"] == 2 and s["exact_match"] == 0.5 and s["protocol_failures"] == 1
    assert s["calls"]["p50"] == 2 and s["prompt_tokens"]["p95"] >= 200
    assert s["makespan"]["p50"] >= 0.5


def test_one_failed_trajectory_does_not_lose_the_others(monkeypatch):
    good = {"calls": 2, "searches": 1, "opens": 0, "prompt_tokens": [10, 20],
            "completion_tokens": [3, 4], "latencies": [0.1, 0.2], "makespan": 1.0,
            "terminal_prompt_tokens": 20, "answer": "Paris", "gold": "Paris",
            "protocol_failures": 0}

    async def fake_run_trajectory(client, url, question, gold, tool, **kwargs):
        if question == "boom":
            raise httpx.ReadError("server went away")
        return good

    monkeypatch.setattr(st, "run_trajectory", fake_run_trajectory)

    async def go():
        return [await st.guarded_trajectory(None, "http://s/v1/chat/completions", q,
                                            "Paris", None, loop=None, max_tokens=8,
                                            max_turns=2, session_id=f"selftest.{i}")
                for i, q in enumerate(("boom", "capital of france"))]

    failed, ok = asyncio.run(go())
    assert failed["error"] == "ReadError: server went away"
    assert failed["calls"] == 0 and failed["answer"] == "" and failed["gold"] == "Paris"
    assert ok == good

    s = st.summarize([failed, ok])
    assert s["n"] == 2 and s["errors"] == 1
    assert s["exact_match"] == 1.0                  # over the trajectory that finished
    assert s["calls"]["p50"] == 2 and s["prompt_tokens"]["max"] == 20


def test_answers_are_graded_the_way_the_metric_grades_them():
    for text in ("the Eiffel Tower", "co-founder", "  A  b.c ", "An Apple", "THE the",
                 "Jean-Luc Picard, Capt.", "1,024", "", "a"):
        assert st._normalize(text) == normalize_answer(text), text
    assert st._matches("The Eiffel Tower", "eiffel tower")
    assert not st._matches("", "")                  # an empty answer never counts


def test_rows_default_to_the_default_tenants_dev_mount(tmp_path):
    rows = tmp_path / DEFAULT_TENANT_ID / "dev" / "rows.json"
    rows.parent.mkdir(parents=True)
    rows.write_text("[]")
    assert st.resolve_rows(None, tenants_dir=tmp_path) == rows
    assert st.resolve_rows("/somewhere/else.json") == Path("/somewhere/else.json")


def test_a_single_differently_named_tenant_is_found_and_ambiguity_is_reported(tmp_path):
    only = tmp_path / "tenant_a" / "dev" / "rows.json"
    only.parent.mkdir(parents=True)
    only.write_text("[]")
    assert st.resolve_rows(None, tenants_dir=tmp_path) == only

    second = tmp_path / "tenant_b" / "dev" / "rows.json"
    second.parent.mkdir(parents=True)
    second.write_text("[]")
    with pytest.raises(FileNotFoundError) as excinfo:
        st.resolve_rows(None, tenants_dir=tmp_path)
    message = str(excinfo.value)
    assert str(tmp_path / DEFAULT_TENANT_ID / "dev" / "rows.json") in message
    assert f"{tmp_path}/*/dev/rows.json" in message


def test_a_server_error_frame_is_kept_and_fails_the_protocol():
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}, "token_ids": [1, 2]}]},
        {"choices": [], "error": {"message": "CUDA out of memory"}},
    ]
    transport = httpx.MockTransport(lambda req: _sse(chunks))

    async def go():
        async with httpx.AsyncClient(transport=transport) as client:
            return await st.stream_completion(client, "http://s/v1/chat/completions",
                                              st.build_request([{"role": "user", "content": "q"}], [], 50))

    out = asyncio.run(go())
    assert out["metadata"]["error"] == "CUDA out of memory"
    assert out["protocol_ok"] is False
    assert out["text"] == "Hel"                     # what arrived before the failure


class _StubBrowser:
    """A corpus browser that answers one canned page, recording how it was called."""

    def __init__(self):
        self.calls = []

    async def search(self, **kwargs):
        self.calls.append(("search", kwargs))
        page = '[1]\nL1: Searched for "capital of france"\nL2: Paris is the capital.'
        return {"query": kwargs.get("query"), "documents": [page], "raw_text": page}

    async def open(self, **kwargs):
        raise AssertionError("open was not part of this trajectory")

    async def find(self, **kwargs):
        raise AssertionError("find was not part of this trajectory")


def _scripted_sse(text, prompt_tokens, completion_tokens):
    chunks = [
        {"choices": [{"delta": {"content": text}, "token_ids": list(range(completion_tokens))}]},
        {"choices": [{"delta": {}, "token_ids": [], "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": prompt_tokens,
                                  "completion_tokens": completion_tokens}},
    ]
    return _sse(chunks)


def test_a_whole_trajectory_searches_then_answers():
    replies = [
        '<|tool_call>call:web_search{query:<|"|>capital of france<|"|>}<tool_call|>',
        "The evidence is consistent. \\boxed{Paris}",
    ]
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        index = len(bodies) - 1
        return _scripted_sse(replies[index], prompt_tokens=100 * (index + 1),
                             completion_tokens=7 + index)

    browser = _StubBrowser()

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await st.run_trajectory(
                client, "http://s/v1/chat/completions", "What is the capital of France?",
                "Paris", browser, loop=st.graded_loop(CHAT_FORMAT),
                max_tokens=st.MAX_NEW_TOKENS, max_turns=8, session_id="selftest.0")

    record = asyncio.run(go())

    assert browser.calls == [("search", {"query": "capital of france",
                                         "_session_id": "selftest.0"})]
    assert len(bodies) == 2
    result = next(m for m in bodies[1]["messages"] if m["role"] == "tool")
    assert result["tool_call_id"] == bodies[1]["messages"][2]["tool_calls"][0]["id"]
    assert "Paris is the capital." in result["content"]
    assert [b["max_tokens"] for b in bodies] == [st.MAX_NEW_TOKENS, st.MAX_NEW_TOKENS]
    assert len(bodies[0]["tools"]) == 3
    assert record["calls"] == 2 and record["searches"] == 1 and record["opens"] == 0
    assert record["answer"] == "Paris"
    assert record["prompt_tokens"] == [100, 200]        # from usage, not counted locally
    assert record["completion_tokens"] == [7, 8]
    assert record["terminal_prompt_tokens"] == 200
    assert record["protocol_failures"] == 0 and record["server_errors"] == []
    assert st.summarize([record])["exact_match"] == 1.0


def test_chat_format_is_the_loops_one_format():
    parser = st.build_parser()
    assert parser.parse_args(["--server", "http://s"]).chat_format == CHAT_FORMAT == t.MODEL.chat_format
    assert parser.parse_args(["--server", "http://s", "--chat-format", "gemma"]).chat_format == "gemma"
    with pytest.raises(SystemExit):
        parser.parse_args(["--server", "http://s", "--chat-format", "chatml"])
    with pytest.raises(ValueError, match="reads chat_format 'gemma', not 'chatml'"):
        st.graded_loop("chatml")


def test_the_default_output_cap_is_the_tasks():
    """One number: the task's workload and the self-test read it from the same module."""
    from tasks.deepresearch import limits
    assert st.MAX_NEW_TOKENS is limits.MAX_NEW_TOKENS is t.MAX_NEW_TOKENS
    assert st.build_parser().parse_args(["--server", "http://s"]).max_tokens \
        == t.TASK.workload.output_length_cap.max_tokens()


def _wire(body: dict) -> bytes:
    """A body as httpx, which the driver and the self-test both post with, puts it on the wire."""
    return httpx.Request("POST", "http://s/v1/chat/completions", json=body).content


def test_a_trajectory_sends_the_drivers_bytes_for_the_same_row_and_settings(tmp_path):
    """The same dev row and the same replies, once through the self-test at its defaults and once
    through the task's framework, protocol and cap as the driver runs them: every request body is
    the same bytes, key order included."""
    from hls.spec.datasets import SOURCE_INDEX, Split

    rows = tmp_path / DEFAULT_TENANT_ID / "dev" / "rows.json"
    rows.parent.mkdir(parents=True)
    rows.write_text(json.dumps([{"query": "What is the capital of France?", "answer": "Paris", SOURCE_INDEX: 7}]))
    replies = [
        '<|tool_call>call:web_search{query:<|"|>capital of france<|"|>}<tool_call|>',
        "The evidence is consistent. \\boxed{Paris}",
    ]

    # The self-test, at the defaults the brief's command leaves it.
    args = st.build_parser().parse_args(["--server", "http://s"])
    [row] = json.loads(st.resolve_rows(None, tenants_dir=tmp_path).read_text())
    tested: list[bytes] = []

    def handler(request):
        tested.append(request.content)
        return _scripted_sse(replies[len(tested) - 1], prompt_tokens=100 * len(tested), completion_tokens=7)

    async def selftest():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await st.run_trajectory(
                client, "http://s/v1/chat/completions", row["query"], row["answer"], _StubBrowser(),
                loop=st.graded_loop(args.chat_format), max_tokens=args.max_tokens, max_turns=args.max_turns,
                session_id="selftest.0")

    record = asyncio.run(selftest())
    assert record["answer"] == "Paris" and record["protocol_failures"] == 0

    # The driver: the dataset's seed for that row, the task's framework, and each request encoded
    # by the task's protocol under the workload's cap, which the trajectory driver stamps on a turn
    # that asks for none of its own.
    cap = t.TASK.workload.output_length_cap.max_tokens()
    [item] = list(t.DATASET.items_at(tmp_path / DEFAULT_TENANT_ID, Split.DEV))
    driven: list[bytes] = []

    async def llm_call(env: Envelope) -> ArtifactResponse:
        asked = env.metadata.get("max_tokens")
        stamped = {**env.metadata, "max_tokens": asked if isinstance(asked, int) and asked > 0 else cap}
        body = t.CONTRACT.protocol.encode_request(env.model_copy(update={"metadata": stamped}), stream=True)
        driven.append(_wire(body))
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text=replies[len(driven) - 1])],
            metadata={"finish_reason": "stop", "completion_tokens": 7, "prompt_tokens": 100 * len(driven),
                      "max_tokens": body["max_tokens"]}))

    browser = _StubBrowser()
    runner = LocalCallableToolRunner(
        specs=[t.BROWSER_SEARCH_TOOL_SPEC, t.BROWSER_OPEN_TOOL_SPEC, t.BROWSER_FIND_TOOL_SPEC],
        callables={"search": browser.search, "open": browser.open, "find": browser.find})
    events = []

    async def on_event(event):
        events.append(event)

    asyncio.run(t.AGENT.framework.run_trajectory(
        seed_item=item, llm_call=llm_call, tool_runner=runner, on_event=on_event,
        ctx=HarnessContext(tenant_id=DEFAULT_TENANT_ID, t_send=0.0, max_output_tokens=cap,
                           max_context_len=t.TASK.max_context_len,
                           chat_template_kwargs=t.MODEL.chat_template_kwargs),
        max_turns=t.MAX_TURNS))

    assert events[-1].extras["final_answer"] == "Paris"
    assert len(tested) == len(driven) == 2
    assert tested == driven
    first = json.loads(tested[0])
    assert list(first) == ["model", "messages", "tools", "max_tokens", "return_token_ids", "logprobs",
                           "top_logprobs", "stream", "stream_options"]
    assert (first["max_tokens"], first["top_logprobs"]) == (cap, t.CONTRACT.protocol.top_logprobs)
    assert first["messages"][0] == {"role": "system", "content": DEEPRESEARCH_SYSTEM}


#: A stand-in for the sandbox's interpreter: it imports the self-test from the directory given as
#: its first argument, laid out as the sandbox's /app, and prints the arguments it parsed.
_PARSE = """
import json, sys
sys.path.insert(0, sys.argv[1])
from tasks.deepresearch import selftest
assert selftest.__file__ == sys.argv[1] + "/tasks/deepresearch/selftest.py", selftest.__file__
print(json.dumps(vars(selftest.build_parser().parse_args(json.loads(sys.argv[2])))))
"""


def test_the_briefs_command_line_runs_against_the_files_the_sandbox_holds(tmp_path):
    """The files the recipe copies, at the paths it copies them to under a stand-in for the
    sandbox's root, with nothing else of the checkout importable: the self-test imports, and its
    parser takes the command the brief gives the agent, word for word."""
    sandbox = next(d for d in t.IMAGES.images if d.name == "deepresearch-sandbox")
    task_dir = Path(t.__file__).resolve().parent
    for inside, source in sandbox.files.items():
        target = tmp_path / inside.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((task_dir / source).read_bytes())
    held = sorted(p.relative_to(tmp_path / "app").as_posix() for p in (tmp_path / "app").rglob("*") if p.is_file())
    assert held == sorted(f"tasks/deepresearch/{name}" for name in t.SANDBOX_FILES)

    brief, _ = render_brief(t.TASK, t.ARTIFACT.brief)
    [command] = [line.strip() for line in brief.splitlines()
                 if "selftest.py" in line and line.strip().startswith("python ")]
    words = shlex.split(command)
    assert words[:2] == ["python", f"{t.SANDBOX_DIR}/selftest.py"]

    app = str(tmp_path / "app")
    # -I: no PYTHONPATH and no working directory on the path, so `tasks` is the stand-in's alone.
    parsed = subprocess.run([sys.executable, "-I", "-c", _PARSE, app, json.dumps(words[2:])],
                            capture_output=True, text=True, cwd=tmp_path)
    assert parsed.returncode == 0, parsed.stderr
    got = json.loads(parsed.stdout)
    assert (got["server"], got["n"], got["concurrency"], got["chat_format"]) == (
        "http://localhost:8000", 8, 8, t.MODEL.chat_format)
    assert got["max_tokens"] == t.MAX_NEW_TOKENS and got["device"] == "auto"
    # The file itself, started the way the brief starts it, finds its siblings from its own path.
    usage = subprocess.run([sys.executable, "-I", f"{app}/tasks/deepresearch/selftest.py", "--help"],
                           capture_output=True, text=True, cwd=tmp_path)
    assert usage.returncode == 0 and "--chat-format" in usage.stdout, usage.stderr
    assert "--device {auto,cuda,cpu}" in usage.stdout
