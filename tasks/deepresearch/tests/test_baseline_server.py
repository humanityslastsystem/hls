"""The measured baseline's server, ``/app/server.py`` of its image: what it loads, warms up and
renders with when a launch or a request names nothing is the task's own.

The engine is absent here, so the server loads against a stand-in ``vllm`` module whose
``SamplingParams`` keeps what it was given; the tokenizer and the engine of a warmup are
stand-ins that record their calls. Everything else is the server's own code."""
from __future__ import annotations

import ast
import asyncio
import os
from types import ModuleType, SimpleNamespace

import pytest

from tasks.deepresearch import task as t
from tasks.deepresearch.research_loop import BROWSER_TOOLS
from tasks.deepresearch.tests import baseline_sources


class _SamplingParams:
    """Keeps the arguments the server samples with."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _Tokenizer:
    """Records each prompt it is asked to render and renders a fixed one."""

    def __init__(self) -> None:
        self.rendered: list[dict] = []

    def apply_chat_template(self, messages, **kwargs) -> str:
        self.rendered.append({"messages": messages, **kwargs})
        return "<prompt>"

    def decode(self, ids, skip_special_tokens=False) -> str:
        return "".join(f"<{i}>" for i in ids)


class _Engine:
    """Records each generation it is asked for and yields nothing."""

    def __init__(self) -> None:
        self.generated: list[tuple] = []

    async def generate(self, prompt, sampling, request_id):
        self.generated.append((prompt, sampling, request_id))
        return
        yield  # pragma: no cover  (makes this an async generator)


@pytest.fixture
def server(monkeypatch):
    """The image's server module, loaded with the engine stood in for. The two variables it
    sets or clears when it loads are put back after the test."""
    for name in ("VLLM_USE_V1", "VLLM_BATCH_INVARIANT"):
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    engine = ModuleType("vllm")
    engine.AsyncEngineArgs = SimpleNamespace
    engine.AsyncLLMEngine = SimpleNamespace
    engine.SamplingParams = _SamplingParams
    module, _ = baseline_sources.server("server.py", {"vllm": engine})
    return module


def _warm_up(server, monkeypatch) -> tuple[_Tokenizer, _Engine]:
    """Run the server's own load: its lifespan, with the tokenizer and the engine stood in for."""
    tokenizer, engine = _Tokenizer(), _Engine()
    monkeypatch.setattr(server, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer))
    monkeypatch.setattr(server, "_build_engine", lambda: engine)
    monkeypatch.setattr(server._GPU, "start", lambda: None)

    async def load() -> None:
        async with server._lifespan(None):
            pass

    asyncio.run(load())
    return tokenizer, engine


def _constant(file: str, name: str):
    """The value ``/app/<file>`` assigns *name* at its top level, read from its source."""
    tree = ast.parse((baseline_sources.APP / file).read_text())
    [value] = [node.value for node in tree.body if isinstance(node, ast.Assign)
               and [getattr(target, "id", None) for target in node.targets] == [name]]
    return ast.literal_eval(value)


@pytest.mark.parametrize("file", ["server.py", "server_hf.py"])
def test_a_servers_search_tool_and_checkpoint_are_the_tasks(file):
    """Both servers of the image: the tool is the research loop's search tool, the checkpoint
    the task's model."""
    assert BROWSER_TOOLS[0]["function"]["name"] == "web_search"
    assert _constant(file, "_SEARCH_TOOL") == BROWSER_TOOLS[0]
    assert _constant(file, "MODEL_ID") == t.MODEL_ID


def test_the_warmup_prompt_renders_with_the_tasks_tool(server, monkeypatch):
    """The prompt the server generates from when it loads carries the task's search tool, and
    is rendered with the launch's chat-template arguments."""
    monkeypatch.setitem(server._STATE, "chat_template_kwargs", dict(t.MODEL.chat_template_kwargs))
    tokenizer, engine = _warm_up(server, monkeypatch)
    [warmup] = tokenizer.rendered
    assert warmup["tools"] == [BROWSER_TOOLS[0]] == [server._SEARCH_TOOL]
    assert warmup["messages"] == [{"role": "user", "content": "_"}]
    assert warmup["enable_thinking"] is True and warmup["add_generation_prompt"] is True
    [(prompt, sampling, request_id)] = engine.generated
    assert (prompt, request_id, sampling.kwargs["max_tokens"]) == ("<prompt>", "warmup-0", 16)
    assert server._STATE["ready"] is True


def test_a_request_with_no_tools_renders_with_the_tasks_tool(server, monkeypatch):
    """The harness's readiness probe carries no tools; a graded turn carries its own, and a
    forced-final turn an empty list, which the server keeps."""
    tokenizer, _ = _warm_up(server, monkeypatch)
    probe = t.CONTRACT.protocol.probe_request()
    assert "tools" not in probe and probe["max_tokens"] == 1
    server._prompt(server._ChatRequest.model_validate(probe))
    assert tokenizer.rendered[-1]["tools"] == [BROWSER_TOOLS[0]]
    assert server._request_max_tokens(server._ChatRequest.model_validate(probe)) == 1

    turn = {"model": "served-model", "messages": [{"role": "user", "content": "q"}]}
    server._prompt(server._ChatRequest.model_validate({**turn, "tools": BROWSER_TOOLS}))
    assert tokenizer.rendered[-1]["tools"] == BROWSER_TOOLS
    server._prompt(server._ChatRequest.model_validate({**turn, "tools": []}))
    assert tokenizer.rendered[-1]["tools"] == []


def test_what_the_server_falls_back_to_is_the_tasks(server, monkeypatch, tmp_path):
    """Its checkpoint by name is the task's model, and its output cap for a request that names
    none is the task's cap. The task's launch reaches neither: the harness mounts the weights
    and stamps a cap on every request."""
    assert server.MODEL_ID == t.MODEL_ID == t.MODEL.base_checkpoint
    monkeypatch.delenv("HLS_WEIGHTS_DIR", raising=False)
    assert server._weights_src() == t.MODEL_ID
    monkeypatch.setenv("HLS_WEIGHTS_DIR", str(tmp_path))
    assert server._weights_src() == str(tmp_path)

    monkeypatch.delenv("HLS_MAX_NEW_TOKENS", raising=False)
    no_cap = server._ChatRequest.model_validate({"messages": [{"role": "user", "content": "q"}]})
    assert server._request_max_tokens(no_cap) == t.MAX_NEW_TOKENS == t.TASK.workload.output_length_cap.max_tokens()


def test_the_tasks_launch_names_the_chat_template_arguments(server):
    """The server's own are none; the task's serve command passes the model's pin with the flag
    the server takes, so every prompt of a launch renders with it."""
    assert server.CHAT_TEMPLATE_KWARGS == {}
    flag, value = t.BASELINE.serve_command
    assert flag == server.CHAT_TEMPLATE_KWARGS_FLAG
    assert server.chat_template_kwargs_arg(value, server.CHAT_TEMPLATE_KWARGS) == dict(t.MODEL.chat_template_kwargs)


def test_the_server_samples_the_tasks_decode(server):
    """One draw from the whole distribution at the task's temperature, with the report the
    contract asks for and the model's control tokens kept in the reply."""
    sampling = server._sampling(64).kwargs
    assert sampling["temperature"] == t.DECODE.temperature == 1.0
    assert (sampling["top_p"], sampling["top_k"], sampling["min_p"]) == (1.0, -1, 0.0)
    assert sampling["logprobs"] == t.CONTRACT.protocol.top_logprobs == 21
    assert sampling["max_tokens"] == 64 and sampling["skip_special_tokens"] is False
    # No text ends a reply: one ends at the model's end-of-turn token or at the cap.
    assert not {"stop", "stop_token_ids", "include_stop_str_in_output"} & set(sampling)
