"""Tests for ``ClockedServing`` -- the engine half of a calibration serving.

A serving is the harness's OWN drive loop, running inside a pool member's process and
submitting each request to that engine's asynchronous serving API as it comes due.
These tests exercise that seam with a fake engine: no GPU, no vllm/sglang install, no
container. What they pin is the part that is ours -- that the real ``Driver`` is what
clocks the traffic, that a conversation is rendered through the model's own chat
template, that emitted token ids come back, and that a failed generation becomes a
faulted event rather than a lost serving. The vendor bindings themselves
(``AsyncLLM.generate``, ``Engine.async_generate``) can only be exercised on the GPU
images that carry those packages.
"""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from hls.core.dataitem import DatasetItem
from hls.core.decode import Greedy, Sample
from hls.core.model import ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.reference.engines import (
    _GREEDY_SAMPLING,
    ClockedServing,
    chat_prompt_ids,
    chat_prompt_text,
    _EngineClient,
    _sampling_for,
)
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.datasets import Dataset, Split
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import load_mounted_sessions
from hls.spec.tenant import Tenant
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import (DeterministicArrival, GammaInterTurn, NoDelay, NoSLO,
                               Poisson, ShortOutputCap, Workload)
from hls.verify import LetterMatchMetric, MetricVerifier
from tests.support.virtual_clock import virtual_clock
from tests.unit.test_trajectory_driver import _agent


class _FakeTokenizer:
    """A tokenizer double: renders a conversation the way a chat template does and
    tokenizes by word, so a test can assert on what the engine was actually fed."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt,
                            tools=None, **template_kwargs):
        assert tokenize is False and add_generation_prompt is True
        turns = " ".join(f"<{m['role']}>{m['content']}" for m in messages)
        rendered = f"<bos> {turns} <assistant>"
        # A real template varies its output on the tools and on the pin; rendering both
        # is how a test sees what reached it. tools is named rather than swept into
        # template_kwargs so the double catches a caller that confuses the two.
        if tools is not None:
            rendered += f" tools={len(tools)}"
        if template_kwargs:
            rendered += " " + " ".join(f"{k}={v}"
                                       for k, v in sorted(template_kwargs.items()))
        return rendered

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return type("Encoded", (), {"input_ids": [len(w) for w in text.split()]})()


class _FakeEngine(ClockedServing):
    """A pool member whose async API is a dict lookup: records every prompt it was
    handed and replies with a fixed token count."""

    def __init__(self, *, tokens: int = 3, fail_on: str | None = None,
                 max_model_len: int | None = None):
        self._tokenizer = _FakeTokenizer()
        self.model_spec = ModelSpec(base_checkpoint="m", arch="TX")
        self.tokens = tokens
        self.max_model_len = max_model_len
        #: request_id whose generation raises (None = never).
        self.fail_on = fail_on
        self.loads = 0
        #: one (prompt_ids, max_new_tokens, tools) per served request, in service order.
        self.served: list[tuple[list[int], int | None, list[dict] | None]] = []

    async def _serve_load(self) -> None:
        self.loads += 1

    async def _serve_one(self, prompt: Envelope):
        from hls.reference.engines import _max_new_tokens, _messages_of, tools_of
        if prompt.request_id == self.fail_on:
            raise RuntimeError("engine core died")
        ids = chat_prompt_ids(self._tokenizer, _messages_of(prompt))
        self.served.append((ids, _max_new_tokens([prompt]), tools_of(prompt)))
        # No report: this member stands in for a greedy serving, whose gate reads none.
        return "reply", "stop", list(range(self.tokens)), None, len(ids)


class _TwoTurnDataset(Dataset):
    """Two sessions: one with two turns, one with a single turn."""
    name = "twoturn"
    version = "v0"

    def items_at(self, mount_dir, split):
        for c, n_turns in ((0, 2), (1, 1)):
            yield DatasetItem(
                request_id=f"c{c}",
                turns=tuple(Envelope(request_id=f"c{c}.{k}",
                                     contents=[TextPayload(text=f"u{c}.{k}")])
                            for k in range(n_turns)),
                ground_truth_turns=tuple(f"g{c}.{k}" for k in range(n_turns)))


class _SixTwoTurnDataset(Dataset):
    """Six two-turn sessions: enough concurrent multi-turn traffic that a stochastic
    workload's drawn arrivals and think-times are visible in the trace."""
    name = "sixtwoturn"
    version = "v0"

    def items_at(self, mount_dir, split):
        for c in range(6):
            yield DatasetItem(
                request_id=f"s{c}",
                turns=tuple(Envelope(request_id=f"s{c}.{k}",
                                     contents=[TextPayload(text=f"u{c}.{k}")])
                            for k in range(2)),
                ground_truth_turns=("g0", "g1"))


def _sessions(history_char_budget: int | None = None) -> list[tuple]:
    """``[(tenant, items)]`` as a calibration serving receives them, on an arrival
    process fast enough that the test does not wait on a clock."""
    tenant = Tenant(
        id="t0",
        data_source=DataSource(dataset=_TwoTurnDataset(),
                               split=SplitSpec(dev=Count(value=1), test=Count(value=2))),
        workload=Workload(arrival=DeterministicArrival(gap=0.0),
                          inter_turn_delay=NoDelay(),
                          output_length_cap=ShortOutputCap(cap_tokens=17), slo=NoSLO(),
                          history_char_budget=history_char_budget),
        verifiers=[MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)])
    return load_mounted_sessions([tenant], Split.TEST)


def _stochastic_sessions() -> list[tuple]:
    """``[(tenant, items)]`` on a workload whose timing is DRAWN: Poisson arrivals and
    Gamma think-times, both seeded (mean gap ~8ms, mean think ~40ms)."""
    tenant = Tenant(
        id="t0",
        data_source=DataSource(dataset=_SixTwoTurnDataset(),
                               split=SplitSpec(dev=Count(value=1), test=Count(value=6))),
        workload=Workload(arrival=Poisson(rate_per_sec=120.0),
                          inter_turn_delay=GammaInterTurn(shape=2.0, scale=0.02,
                                                          max_gap=0.2),
                          output_length_cap=ShortOutputCap(cap_tokens=17), slo=NoSLO()),
        verifiers=[MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)])
    return load_mounted_sessions([tenant], Split.TEST)


def _timing_signature(trace) -> tuple[list[float], list[float]]:
    """What a serving's SEED decides, read back off its trace:
    ``(arrival_offsets, think_gaps)``.

    ``arrival_offsets`` are the drawn session arrivals, recovered from the events'
    ``t_planned_send`` (stamped on each session's turn 0) relative to the earliest.
    ``think_gaps`` are each session's realized inter-turn wait (turn 1's send minus
    turn 0's receive)."""
    first = {ev.session_id: ev for ev in trace.events if ev.turn_index == 0}
    later = {ev.session_id: ev for ev in trace.events if ev.turn_index == 1}
    t_zero = min(ev.t_planned_send for ev in first.values())
    sids = sorted(first)
    return ([first[s].t_planned_send - t_zero for s in sids],
            [later[s].t_send - first[s].t_recv for s in sids])


def _scorer() -> GoodputScorer:
    return GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))


# ---- chat rendering ---------------------------------------------------------

def test_chat_rendering_is_the_models_own_template_tokenized_without_specials():
    """The teacher-forced context assumes a compliant server's rendering, so the served
    prompt must be the model's own chat template with the generation prompt appended.
    Ids, not text, because handing a serving stack the rendered string makes it
    re-tokenize and prepend a second BOS."""
    tok = _FakeTokenizer()
    messages = [{"role": "user", "content": "hello"}]
    assert chat_prompt_text(tok, messages) == "<bos> <user>hello <assistant>"
    assert chat_prompt_ids(tok, messages) == [5, 11, 11]


# ---- the chat-template pin --------------------------------------------------

class _RecordedSamplingParams:
    """Stands in for ``vllm.SamplingParams``, keeping whatever the engine passed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _RecordedLLM:
    """Stands in for the synchronous ``vllm.LLM``: records every ``chat`` call and
    replies with one output per conversation in the batch."""

    def __init__(self):
        #: one dict of keyword arguments per chat call, in call order.
        self.calls: list[dict] = []

    def chat(self, *, messages, sampling_params, use_tqdm, **kwargs):
        self.calls.append({"messages": messages, "sampling_params": sampling_params,
                           **kwargs})
        out = type("Out", (), {"text": "r", "finish_reason": "stop", "token_ids": [1]})()
        return [type("Res", (), {"outputs": [out]})() for _ in messages]


class _RecordedAsyncLLM:
    """Stands in for vLLM's ``AsyncLLM``: records the prompt it was handed and yields
    one finished cumulative output.

    The output carries a one-position report in vLLM's shape (a ``{token_id: Logprob}``
    per served token) so the fake also serves a sampled decode, which asks for one."""

    def __init__(self):
        #: one prompt dict per served request, in service order.
        self.prompts: list[dict] = []
        #: the sampling parameters of each served request, in the same order.
        self.params: list[_RecordedSamplingParams] = []

    async def generate(self, *, request_id, prompt, sampling_params):
        self.prompts.append(prompt)
        self.params.append(sampling_params)
        step = {1: type("Logprob", (), {"logprob": -0.1, "decoded_token": "r"})()}
        out = type("Out", (), {"text": "r", "finish_reason": "stop", "token_ids": [1],
                               "logprobs": [step]})()
        yield type("Res", (), {"outputs": [out]})()


class _RecordingTokenizer(_FakeTokenizer):
    """A ``_FakeTokenizer`` that keeps every rendered prompt it was asked to tokenize,
    so a test can assert on the text a path rendered rather than on its ids."""

    def __init__(self):
        #: one rendered prompt per tokenize call, in call order.
        self.texts: list[str] = []

    def __call__(self, text, *, add_special_tokens):
        self.texts.append(text)
        return super().__call__(text, add_special_tokens=add_special_tokens)


class _RecordedSyncSGLangEngine:
    """Stands in for the blocking ``sglang.Engine.generate``: records the ids it was
    handed and replies in the shape ``return_logprob=True`` produces."""

    def __init__(self):
        #: the ids of the batch it was handed.
        self.input_ids: list[list[int]] = []

    def generate(self, *, input_ids, sampling_params, return_logprob):
        self.input_ids = input_ids
        return [{"text": "r",
                 "meta_info": {"finish_reason": {"type": "stop"},
                               "output_token_logprobs": [(-0.1, 7, "r")]}}
                for _ in input_ids]


class _RecordedSGLangEngine:
    """Stands in for ``sglang.Engine``: records the ids it was handed and replies in
    the shape ``return_logprob=True`` produces."""

    def __init__(self):
        #: one id list per served request, in service order.
        self.ids: list[list[int]] = []
        #: the sampling parameters of each served request, in the same order.
        self.params: list[dict] = []

    async def async_generate(self, *, input_ids, sampling_params, return_logprob,
                             **kwargs):
        self.ids.append(input_ids)
        self.params.append(sampling_params)
        # ``output_top_logprobs`` is one alternatives row per served token; a sampled
        # decode asks for it and the reply is read through ``_sglang_positions``.
        return {"text": "r",
                "meta_info": {"finish_reason": {"type": "stop"},
                              "output_token_logprobs": [(-0.1, 7, "r")],
                              "output_top_logprobs": [[(-0.1, 7, "r")]]}}


@pytest.fixture
def vllm_stub(monkeypatch):
    """A ``vllm`` module carrying only ``SamplingParams``, which is all the paths under
    test import. vllm is a GPU-image dependency and is not installed in this suite's
    environment; monkeypatch restores sys.modules afterwards."""
    mod = types.ModuleType("vllm")
    mod.SamplingParams = _RecordedSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", mod)
    return mod


@pytest.fixture
def sglang_stub(monkeypatch):
    """An ``sglang`` module whose ``Engine`` records its construction kwargs, so a test
    can drive ``SGLangEngine._load`` on a host with neither sglang nor the weights.
    Yields that list. The tokenizer loader is stubbed for the same reason: ``_load``
    builds one before the engine and the fixture models have no checkpoint behind
    them."""
    from transformers import AutoTokenizer

    built: list[dict] = []
    mod = types.ModuleType("sglang")
    mod.Engine = lambda **kwargs: built.append(kwargs) or _RecordedSGLangEngine()
    monkeypatch.setitem(sys.modules, "sglang", mod)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        classmethod(lambda cls, *a, **k: _FakeTokenizer()))
    return built


def _pinned_engine(cls, pin: dict, *, decode=None, **kwargs):
    """One reference engine on a model whose chat-template pin is ``pin``."""
    return cls(model_spec=ModelSpec(base_checkpoint="m", arch="TX",
                                    chat_template_kwargs=pin),
               decode=decode or Greedy(), **kwargs)


def test_chat_prompt_helpers_pass_the_pin_to_the_template():
    """A template switch (Gemma 4's ``enable_thinking``) changes the rendered prompt and
    so the tokens a faithful reply carries, so it has to reach apply_chat_template."""
    tok = _FakeTokenizer()
    messages = [{"role": "user", "content": "hello"}]
    assert chat_prompt_text(tok, messages) == "<bos> <user>hello <assistant>"
    assert (chat_prompt_text(tok, messages, template_kwargs={"enable_thinking": True})
            == "<bos> <user>hello <assistant> enable_thinking=True")
    assert (chat_prompt_ids(tok, messages, template_kwargs={"enable_thinking": True})
            == [5, 11, 11, 20])


def test_generate_uniq_merges_the_pin_with_the_turns_tools(vllm_stub):
    """vLLM's own chat rendering takes one ``chat_template_kwargs`` dict, which has to
    carry the turn's tools and the model's pin together: the batch is grouped by tools,
    while the pin is the same for every conversation this engine ever renders."""
    from hls.reference.engines import VLLMEngine

    engine = _pinned_engine(VLLMEngine, {"enable_thinking": True})
    engine._llm = llm = _RecordedLLM()
    tools = [{"name": "search"}]
    msgs = [{"role": "user", "content": "hello"}]
    engine._generate_uniq(["k0"], [msgs], [tools], 8)
    assert llm.calls[0]["chat_template_kwargs"] == {"enable_thinking": True,
                                                   "tools": tools}


def test_generate_uniq_passes_the_pin_when_a_turn_has_no_tools(vllm_stub):
    """A toolless turn still renders under the pin, which is the model's, not the
    turn's."""
    from hls.reference.engines import VLLMEngine

    engine = _pinned_engine(VLLMEngine, {"enable_thinking": True})
    engine._llm = llm = _RecordedLLM()
    engine._generate_uniq(["k0"], [[{"role": "user", "content": "hello"}]], [None], 8)
    assert llm.calls[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_generate_uniq_sends_nothing_when_there_is_no_pin_and_no_tools(vllm_stub):
    """An unpinned model renders with the template's defaults, so no empty
    ``chat_template_kwargs`` is sent."""
    from hls.reference.engines import VLLMEngine

    engine = _pinned_engine(VLLMEngine, {})
    engine._llm = llm = _RecordedLLM()
    engine._generate_uniq(["k0"], [[{"role": "user", "content": "hello"}]], [None], 8)
    assert "chat_template_kwargs" not in llm.calls[0]


def test_clocked_paths_render_prompt_ids_with_the_pin(vllm_stub):
    """Both pool members render chat themselves and feed the async engine ids, so the
    pin has to reach that render too: a member serving an unpinned prompt would be
    scored against an oracle that saw a different conversation."""
    from hls.reference.engines import SGLangEngine, VLLMEngine

    pin = {"enable_thinking": True}
    messages = [{"role": "user", "content": "hello"}]
    expected = chat_prompt_ids(_FakeTokenizer(), messages, template_kwargs=pin)
    assert expected != chat_prompt_ids(_FakeTokenizer(), messages)
    env = Envelope(request_id="r0", metadata={"messages": messages})

    vllm_engine = _pinned_engine(VLLMEngine, pin)
    vllm_engine._tokenizer = _FakeTokenizer()
    vllm_engine._async_llm = async_llm = _RecordedAsyncLLM()
    asyncio.run(vllm_engine._serve_one(env))
    assert async_llm.prompts == [{"prompt_token_ids": expected}]

    sglang_engine = _pinned_engine(SGLangEngine, pin)
    sglang_engine._tokenizer = _FakeTokenizer()
    sglang_engine._engine = recorded = _RecordedSGLangEngine()
    asyncio.run(sglang_engine._serve_one(env))
    assert recorded.ids == [expected]


def test_sglang_generate_renders_the_prompt_with_the_pin():
    """The synchronous oracle renders chat through our own tokenizer and feeds sglang
    the ids, so the pin has to reach that render the way it reaches the clocked one.
    Pre-setting _engine makes _load return early, so sglang is never imported."""
    from hls.reference.engines import SGLangEngine

    engine = _pinned_engine(SGLangEngine, {"enable_thinking": True})
    engine._tokenizer = tokenizer = _RecordingTokenizer()
    engine._engine = recorded = _RecordedSyncSGLangEngine()
    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hello"}]})

    out = asyncio.run(engine.generate([env]))
    assert tokenizer.texts == ["<bos> <user>hello <assistant> enable_thinking=True"]
    assert recorded.input_ids == [[5, 11, 11, 20]]
    assert out[0].first_of("text").tokens == [7]


def test_vllm_oracle_leaves_the_decode_stock(vllm_stub):
    """The oracle's reply text is scored by the metric verifiers against a contestant's
    reply, which a stock server decodes with its special tokens stripped. So the
    synchronous path leaves skip_special_tokens alone and gets that same decode, unlike
    the clocked path (see _reference_reply)."""
    from hls.reference.engines import VLLMEngine

    engine = _pinned_engine(VLLMEngine, {})
    engine._llm = llm = _RecordedLLM()
    engine._generate_uniq(["k0"], [[{"role": "user", "content": "hello"}]], [None], 8)
    assert "skip_special_tokens" not in llm.calls[0]["sampling_params"].kwargs


def test_clocked_paths_keep_control_tokens_in_the_reply_text(vllm_stub):
    """A clocked serving stands in for a contestant server, whose reply an agentic
    framework parses for tool-call and reasoning delimiters. For a model that spells
    those as special tokens a stripping decode would leave nothing to parse, so both
    members keep them."""
    from hls.reference.engines import SGLangEngine, VLLMEngine

    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hello"}]})

    vllm_engine = _pinned_engine(VLLMEngine, {})
    vllm_engine._tokenizer = _FakeTokenizer()
    vllm_engine._async_llm = async_llm = _RecordedAsyncLLM()
    asyncio.run(vllm_engine._serve_one(env))
    assert async_llm.params[0].kwargs["skip_special_tokens"] is False

    sglang_engine = _pinned_engine(SGLangEngine, {})
    sglang_engine._tokenizer = _FakeTokenizer()
    sglang_engine._engine = recorded = _RecordedSGLangEngine()
    asyncio.run(sglang_engine._serve_one(env))
    assert recorded.params[0]["skip_special_tokens"] is False


# ---- the sampling pin -------------------------------------------------------

def test_every_sampling_field_is_pinned_explicitly():
    """A checkpoint's generation_config ships sampling defaults (Qwen's carries a 1.05
    repetition penalty) and a stack applies them to fields a request leaves unset.
    Greedy is what the gate is defined against, so each field is set explicitly rather
    than left to whatever the stack's default happens to be."""
    assert _GREEDY_SAMPLING["temperature"] == 0.0
    assert _GREEDY_SAMPLING["repetition_penalty"] == 1.0
    assert _GREEDY_SAMPLING["presence_penalty"] == _GREEDY_SAMPLING["frequency_penalty"] == 0.0
    assert _GREEDY_SAMPLING["top_p"] == 1.0 and _GREEDY_SAMPLING["top_k"] == -1
    # The output cap is deliberately NOT in the pin: it is the workload's, stamped per
    # request, and baking a value here would silently override it. Both spellings, since
    # the one pin is spread into vLLM's SamplingParams and SGLang's params dict alike.
    assert "max_tokens" not in _GREEDY_SAMPLING
    assert "max_new_tokens" not in _GREEDY_SAMPLING


def test_a_sampled_decode_pins_its_own_truncation_off(monkeypatch):
    """Under greedy either spelling of "disabled" would do, since both stacks override
    top_k once the temperature is 0. A sampled decode is the case where neither does,
    so the value is the graded distribution's own "no truncation" pin and is set here
    rather than inherited from the greedy dict."""
    monkeypatch.setitem(_GREEDY_SAMPLING, "top_k", 20)
    assert _sampling_for(Sample(temperature=1.0))["top_k"] == -1
    # Both stacks take -1: sglang expands it to the whole vocabulary and rejects 0,
    # vLLM reads it as "consider all tokens".
    assert _sampling_for(Sample(temperature=1.0))["min_p"] == 0.0


def test_a_sampled_decode_pins_no_draw():
    """The decode carries no seed, and the absence is deliberate rather than an
    oversight: the report gate reads the distribution the server reported at each
    position instead of replaying a reference draw, and sglang only acts on a
    per-request seed when the whole engine is built for deterministic inference --
    which swaps its sampling and attention backends, so a seeded pool member would
    measure a different kernel set than its unseeded neighbours."""
    assert not hasattr(Sample(temperature=1.0), "seed")
    assert Sample(temperature=1.0).fingerprint() == "sample-t1.0"


def test_neither_stack_is_sent_a_draw_seed(vllm_stub):
    """Guards the absence at the wire: a seed reintroduced on the decode must not
    reach a serving under either stack's spelling without this failing first."""
    from hls.reference.engines import SGLangEngine, VLLMEngine

    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hello"}]})

    vllm = _pinned_engine(VLLMEngine, {}, decode=Sample(temperature=1.0))
    vllm._tokenizer = _FakeTokenizer()
    vllm._async_llm = async_llm = _RecordedAsyncLLM()
    asyncio.run(vllm._serve_one(env))
    assert "seed" not in async_llm.params[0].kwargs

    sgl = _pinned_engine(SGLangEngine, {}, decode=Sample(temperature=1.0))
    sgl._tokenizer = _FakeTokenizer()
    sgl._engine = recorded = _RecordedSGLangEngine()
    asyncio.run(sgl._serve_one(env))
    assert not {"seed", "sampling_seed"} & set(recorded.params[0])


# ---- the request adapter ----------------------------------------------------

def test_engine_client_returns_the_emitted_ids_and_finish_reason():
    """The drive only needs ``submit``; what comes back must carry the exact ids the
    engine emitted, since that is what the pinned reference teacher-forces."""
    engine = _FakeEngine(tokens=4)
    env = Envelope(request_id="r0", metadata={"messages": [{"role": "user",
                                                            "content": "hi"}]})
    resp = asyncio.run(_EngineClient(engine).submit(env))
    tp = resp.output.first_of("text")
    assert tp.tokens == [0, 1, 2, 3] and tp.finish_reason == "stop"
    assert resp.output.request_id == "r0"


def test_engine_client_reply_reports_what_the_wire_would_have_decoded():
    """A pool member stands in for a graded server, and an agentic framework decides
    when a trajectory must stop and answer, and whether a reply was cut at its cap, by
    reading the serving facts off every reply. So a member's reply has to carry the
    same fields, under the same names and with the same values, that the wire protocol
    decodes off a compliant server's answer to the same request."""
    engine = _FakeEngine(tokens=4, max_model_len=2048)
    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hi there"}],
                             "max_tokens": 64})
    served = asyncio.run(_EngineClient(engine).submit(env)).output
    prompt_tokens = len(engine.served[0][0])

    wire = OpenAIChatProtocol().decode_response(
        {"choices": [{"message": {"content": "reply"}, "finish_reason": "stop",
                      "token_ids": [0, 1, 2, 3]}],
         "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 4,
                   "total_tokens": prompt_tokens + 4},
         "hls_context": {"max_context_len": 2048, "request_max_tokens": 64}},
        "r0")

    assert served.metadata == {"source": "reference", **wire.metadata}
    assert wire.metadata["prompt_tokens"] == prompt_tokens > 0


def test_engine_client_reports_no_window_when_the_engine_declares_none():
    """An engine built at no context window declares none, the way a reply whose server
    sent no serving configuration carries none: the framework's context guards test for
    the field and leave the trajectory alone when it is absent. Same for a turn that
    asks for no output cap."""
    engine = _FakeEngine(tokens=2)
    env = Envelope(request_id="r0", metadata={"messages": [{"role": "user",
                                                            "content": "hi"}]})
    meta = asyncio.run(_EngineClient(engine).submit(env)).output.metadata
    assert "max_context_len" not in meta and "request_max_tokens" not in meta
    assert meta["completion_tokens"] == 2


def test_engine_client_turns_a_failed_generation_into_a_faulted_request():
    """A generation that raises must become a FAULTED event, not an exception that
    kills the serving mid-flight: the drive records it, and the calibration then
    refuses the whole serving on any fault (a member that could not serve its own
    task's load has not produced a measurement)."""
    from hls.core.faults import ArtifactFault

    engine = _FakeEngine(fail_on="r0")
    env = Envelope(request_id="r0", metadata={"messages": [{"role": "user",
                                                            "content": "hi"}]})
    with pytest.raises(ArtifactFault, match="engine core died"):
        asyncio.run(_EngineClient(engine).submit(env))


# ---- the serving itself -----------------------------------------------------

def test_a_serving_is_driven_only_with_the_grace_its_caller_names():
    """The serving builds its driver with its caller's grace; it has none of its own."""
    with pytest.raises(TypeError, match="request_timeout_sec"):
        asyncio.run(_FakeEngine(tokens=5).serve_traffic(_sessions(), seed=7))


def test_serve_traffic_drives_every_turn_through_the_real_driver():
    """The acceptance bar: one ``serve_traffic`` call runs the harness's own drive loop
    against the engine's API and returns a trace with one event per served turn --
    session history stitched, output cap stamped, ids attached."""
    engine = _FakeEngine(tokens=5)
    trace = asyncio.run(engine.serve_traffic(_sessions(), seed=7, request_timeout_sec=REQUEST_TIMEOUT_SEC))

    assert engine.loads == 1
    assert [ev.request_id for ev in trace.events] == ["c0.0", "c0.1", "c1.0"]
    assert all(ev.error is None for ev in trace.events)
    assert all(ev.response.output.first_of("text").tokens == [0, 1, 2, 3, 4]
               for ev in trace.events)
    # The workload's output cap reached the engine on every request.
    assert [cap for _, cap, _ in engine.served] == [17, 17, 17]
    # ...and the second turn of session c0 carried the stitched conversation: its user
    # turn, the engine's own reply, then the follow-up.
    second_turn = next(ev for ev in trace.events if ev.request_id == "c0.1")
    assert [m["role"] for m in second_turn.input.metadata["messages"]] == [
        "user", "assistant", "user"]
    assert second_turn.input.metadata["messages"][1]["content"] == "reply"


def test_a_serving_ends_a_session_on_the_same_context_budget_a_graded_run_does():
    """A calibration serving measures the request set a graded run will serve, so the
    tenant's context budget has to end its sessions here too. Session c0's second turn
    carries its 4-character question, the engine's 5-character reply and the 4-character
    follow-up (13 > 5), so it is not served; the single-turn session is untouched."""
    assert [ev.request_id for ev in
            asyncio.run(_FakeEngine().serve_traffic(_sessions(), seed=7,
                                                    request_timeout_sec=REQUEST_TIMEOUT_SEC)).events] == [
        "c0.0", "c0.1", "c1.0"]
    trace = asyncio.run(_FakeEngine().serve_traffic(_sessions(5), seed=7,
                                                    request_timeout_sec=REQUEST_TIMEOUT_SEC))
    assert [ev.request_id for ev in trace.events] == ["c0.0", "c1.0"]
    assert all(ev.error is None for ev in trace.events)


def test_serve_traffic_subsamples_sessions_when_bounded():
    """The tractability knob reaches the drive: a bounded serving drives fewer whole
    sessions at the same arrival rate."""
    engine = _FakeEngine()
    trace = asyncio.run(engine.serve_traffic(_sessions(), seed=7, max_sessions=1,
                                             request_timeout_sec=REQUEST_TIMEOUT_SEC))
    assert [ev.session_id for ev in trace.events] == ["c0", "c0"]


def test_serve_traffic_records_a_failed_generation_as_a_faulted_event():
    """A member that fails one request produces a trace with that event faulted, which
    is what lets ``hls.calibrate`` refuse the serving by name and count."""
    engine = _FakeEngine(fail_on="c1.0")
    trace = asyncio.run(engine.serve_traffic(_sessions(), seed=7, request_timeout_sec=REQUEST_TIMEOUT_SEC))
    faulted = [ev for ev in trace.events if ev.error]
    assert [ev.request_id for ev in faulted] == ["c1.0"]
    assert "engine core died" in faulted[0].error
    assert faulted[0].response is None


def test_serve_traffic_drives_an_agentic_task_with_its_own_framework():
    """A serving of an agentic task is the task's trajectories, not one seed request
    per item: every LLM call the framework makes reaches the engine, carrying the
    tools the framework offered and the history it stitched."""
    engine = _FakeEngine(tokens=4)
    trace = asyncio.run(engine.serve_traffic(_sessions(), seed=7, agent=_agent(),
                                             request_timeout_sec=REQUEST_TIMEOUT_SEC))
    assert sorted(ev.request_id for ev in trace.events) == ["c0.0", "c0.1", "c1.0", "c1.1"]
    assert all(ev.error is None for ev in trace.events)
    assert all(ev.response.output.first_of("text").tokens == [0, 1, 2, 3]
               for ev in trace.events)
    tools = [t for _, _, t in engine.served]
    assert all(t and t[0]["function"]["name"] == "lookup" for t in tools)
    # Every call was capped by the tenant's workload, as a graded run's would be.
    assert all(cap == 17 for _, cap, _ in engine.served)


def test_serve_traffic_without_an_agent_still_drives_scripted_turns():
    trace = asyncio.run(_FakeEngine().serve_traffic(_sessions(), seed=7, agent=None,
                                                    request_timeout_sec=REQUEST_TIMEOUT_SEC))
    assert [ev.request_id for ev in trace.events] == ["c0.0", "c0.1", "c1.0"]


def _serve(seed: int):
    """One stochastic serving at ``seed``, as its ``(arrival_offsets, think_gaps)``
    signature plus the request set it drove.

    The serving runs on a virtual clock (``tests.support.virtual_clock``), so its waits
    take exactly the time the schedule gives them however loaded the machine is, and a
    repeated seed repeats every stamp."""
    engine = _FakeEngine()
    with virtual_clock() as runner:
        trace = runner.run(engine.serve_traffic(_stochastic_sessions(), seed=seed,
                                                request_timeout_sec=REQUEST_TIMEOUT_SEC))
    assert all(ev.error is None for ev in trace.events)
    return _timing_signature(trace), sorted(ev.request_id for ev in trace.events)


def test_the_serving_clock_is_the_drivers_own():
    """Not a reimplementation: the seed a serving is clocked with reaches the same
    scheduling code a graded run uses. Under a DRAWN workload (Poisson arrivals, Gamma
    think-times) two seeds must therefore produce different timing -- both the arrival
    schedule and the think-times -- while driving the same request SET, and one seed
    must reproduce its timing exactly. A serving that dropped the seed on the floor
    would pass none of the three.

    The claim is on the drawn TIMING, not on the order events happen to land in: with
    gaps this short, asserting an order would be asserting the outcome of a wall-clock
    race between sessions."""
    (offsets_a, thinks_a), requests_a = _serve(11)
    (offsets_b, thinks_b), requests_b = _serve(12)
    (offsets_a2, thinks_a2), requests_a2 = _serve(11)

    # The workload is the same request set at every seed; only its timing is drawn.
    assert requests_a == requests_b == requests_a2
    assert requests_a == [f"s{c}.{k}" for c in range(6) for k in range(2)]

    # A different seed draws a different schedule AND different think-times. The
    # margins are milliseconds, and exactly 0 for both if the seed never reached the
    # draw.
    assert max(abs(x - y) for x, y in zip(offsets_a, offsets_b)) > 5e-3
    assert max(abs(x - y) for x, y in zip(thinks_a, thinks_b)) > 5e-3

    # The same seed reproduces it exactly: on the virtual clock nothing but the draw
    # moves a stamp.
    assert offsets_a == offsets_a2
    assert thinks_a == thinks_a2


def test_clocked_serving_is_the_pool_membership_criterion():
    """Pool members implement ClockedServing; the reference instruments do not -- an
    engine that cannot take one request at a time on a clock cannot be a member, and
    the capability is what says so."""
    from hls.core.capabilities import ServeTraffic
    from hls.reference.engines import HFEngine, SGLangEngine, VLLMEngine

    assert issubclass(VLLMEngine, ServeTraffic)
    assert issubclass(SGLangEngine, ServeTraffic)
    assert not issubclass(HFEngine, ServeTraffic)


