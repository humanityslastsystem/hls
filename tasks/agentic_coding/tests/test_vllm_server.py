"""The prompt the baseline server hands its engine, read through its own HTTP handler.

The gate teacher-forces every served reply under the prompt the contract renders, so
the prompt this server feeds its engine has to be that render, token for token. A
continuation is where the two can part: the harness resumes a cut reply by sending it
back as a final assistant message with ``continue_final_message: true`` and
``add_generation_prompt: false``, and a server that drops those fields renders a fresh
turn after it instead.

The bodies posted here come from the task's own wire encoder, so they are the bodies a
graded run sends. The server imports vLLM at module level, so a stand-in replaces it
and a fake engine records what it was handed. The tokenizer is the checkpoint's own,
because the render is a claim about its template, and the tests that need it skip where
no snapshot is cached.
"""
from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.reference.engines import chat_prompt_ids
from tasks.agentic_coding.framework import prompt_tokenizer
from tasks.agentic_coding.mini_swe_loop import BASH_TOOL, THINK_CLOSE
from tasks.agentic_coding.tests.test_continuation_render import _snapshot

SERVER = "tasks.agentic_coding.vllm_server"
MESSAGES = [{"role": "user", "content": "Do the thing."}]
#: The one id the fake engine serves, with a log-probability for the report.
SERVED_ID = 13


class _FakeEngine:
    """Records each prompt it is handed and serves one token for it."""

    def __init__(self) -> None:
        self.prompts: list = []

    async def generate(self, prompt, sampling_params, request_id):
        self.prompts.append(prompt)
        completion = types.SimpleNamespace(
            token_ids=[SERVED_ID], text="x", finish_reason="stop",
            logprobs=[{SERVED_ID: types.SimpleNamespace(logprob=-0.5)}])
        yield types.SimpleNamespace(outputs=[completion])


class _NoTokenizer:
    """Fails any use, for a request that must be refused before it is rendered."""

    def __getattr__(self, name):
        raise AssertionError(f"the tokenizer was used ({name}) for a refused request")

    def __call__(self, *args, **kwargs):
        raise AssertionError("the tokenizer was called for a refused request")


def _fake_vllm() -> dict[str, types.ModuleType]:
    vllm = types.ModuleType("vllm")
    vllm.AsyncEngineArgs = lambda **kw: kw
    vllm.AsyncLLMEngine = object
    vllm.SamplingParams = lambda **kw: types.SimpleNamespace(**kw)
    config = types.ModuleType("vllm.config")
    config.KVTransferConfig = lambda **kw: kw
    return {"vllm": vllm, "vllm.config": config}


@pytest.fixture
def server(monkeypatch):
    """The server module, imported fresh against the stand-in vLLM."""
    for name, module in _fake_vllm().items():
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules.pop(SERVER, None)
    yield importlib.import_module(SERVER)
    sys.modules.pop(SERVER, None)


@pytest.fixture
def tokenizer(monkeypatch):
    """The checkpoint's tokenizer, through the loader the task ships."""
    snap = _snapshot()
    if snap is None:
        pytest.skip("no local snapshot of the checkpoint; this claim is about its template")
    monkeypatch.setenv("HLS_WEIGHTS_DIR", snap)
    tok = prompt_tokenizer()
    if tok is None:
        pytest.skip(f"{snap} holds no loadable tokenizer")
    return tok


def _client(server, tokenizer, engine):
    """A client on the server's app, with the loaded state its lifespan would set.
    Not entered as a context manager, so the lifespan (which loads the real engine)
    never runs."""
    from fastapi.testclient import TestClient
    from tasks.agentic_coding.serving_config import TokenText
    server._STATE.update(ready=True, engine=engine, tokenizer=tokenizer,
                         token_text=TokenText(lambda tid: "x"), chat_template_kwargs={})
    return TestClient(server._build_app())


def _wire_body(messages, *, continues: bool, stream: bool = False) -> dict:
    """The request body a graded run sends for this turn: the task's own encoder over
    the metadata the loop puts on the request."""
    from tasks.agentic_coding.task import CONTRACT
    metadata = {"messages": messages, "tools": [BASH_TOOL], "max_tokens": 64}
    if continues:
        metadata["continue_final_message"] = True
    env = Envelope(request_id="s0.1", contents=[TextPayload(text=messages[-1]["content"])],
                   metadata=metadata)
    return CONTRACT.protocol.encode_request(env, stream=stream)


def _prompt_tokens(response, stream: bool) -> int:
    """The ``usage.prompt_tokens`` a reply reports, from either response shape."""
    assert response.status_code == 200, response.text
    if not stream:
        return response.json()["usage"]["prompt_tokens"]
    chunks = [json.loads(line[len("data: "):]) for line in response.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    return chunks[-1]["usage"]["prompt_tokens"]


def _contract_ids(tokenizer, messages, *, continues: bool) -> list[int]:
    """The prompt ids the reference renders for this turn."""
    return chat_prompt_ids(tokenizer, messages, [BASH_TOOL], {}, continues)


CARRIED = {
    "thought cut": "I am thinki" + THINK_CLOSE,
    "visible half cut": "thinking" + THINK_CLOSE + "cat <<EO",
}


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
@pytest.mark.parametrize("carried", list(CARRIED.values()), ids=list(CARRIED))
def test_a_continuation_is_served_the_contract_render(server, tokenizer, carried, stream):
    """A resumed reply is fed to the engine as the contract's continuation render: the
    conversation, then the carried text, and nothing after it.

    A server that ignores the two fields appends a generation prompt after the carried
    text, so the model opens a fresh thought where the reference expects it to go on,
    and every position of the second phase is scored against a prompt it never saw."""
    engine = _FakeEngine()
    messages = MESSAGES + [{"role": "assistant", "content": carried}]
    body = _wire_body(messages, continues=True, stream=stream)
    assert body["continue_final_message"] is True
    assert body["add_generation_prompt"] is False

    response = _client(server, tokenizer, engine).post("/v1/chat/completions", json=body)

    expected = _contract_ids(tokenizer, messages, continues=True)
    assert _prompt_tokens(response, stream) == len(expected)
    assert engine.prompts == [{"prompt_token_ids": expected}]


def test_continue_final_message_alone_resumes_the_message(server, tokenizer):
    """``add_generation_prompt`` defaults to the opposite of ``continue_final_message``,
    so a body that names only the first still resumes the message."""
    engine = _FakeEngine()
    messages = MESSAGES + [{"role": "assistant", "content": "I am thinking" + THINK_CLOSE}]
    body = _wire_body(messages, continues=True)
    del body["add_generation_prompt"]

    response = _client(server, tokenizer, engine).post("/v1/chat/completions", json=body)

    expected = _contract_ids(tokenizer, messages, continues=True)
    assert _prompt_tokens(response, stream=False) == len(expected)
    assert engine.prompts == [{"prompt_token_ids": expected}]


@pytest.mark.parametrize("stream", [False, True], ids=["unary", "stream"])
def test_a_first_turn_ends_in_the_generation_prompt(server, tokenizer, stream):
    """A turn that is not a continuation is rendered ready to generate, as before."""
    engine = _FakeEngine()
    body = _wire_body(MESSAGES, continues=False, stream=stream)
    assert "continue_final_message" not in body

    response = _client(server, tokenizer, engine).post("/v1/chat/completions", json=body)

    expected = _contract_ids(tokenizer, MESSAGES, continues=False)
    assert _prompt_tokens(response, stream) == len(expected)
    assert engine.prompts == [{"prompt_token_ids": expected}]


def test_resuming_a_message_and_opening_a_new_one_is_refused(server):
    """Both flags true asks for two different prompts at once. vLLM answers 400, and so
    does this server, before anything is rendered or served."""
    engine = _FakeEngine()
    messages = MESSAGES + [{"role": "assistant", "content": "I am thinking" + THINK_CLOSE}]
    body = _wire_body(messages, continues=True)
    body["add_generation_prompt"] = True

    response = _client(server, _NoTokenizer(), engine).post("/v1/chat/completions",
                                                            json=body)

    assert response.status_code == 400, response.text
    assert "continue_final_message" in response.json()["error"]["message"]
    assert engine.prompts == []
