"""The reference renders a continued turn the way the driver rendered it.

An agentic loop that cuts a reply at a cap hands the text back as a trailing assistant
message with ``continue_final_message`` set, and the server resumes that message: no
generation prompt after it. The reference teacher-forces the served tokens under a
prompt it renders itself, so the flag has to travel from the trace event into the
reference item, through the calibration capture store, and into every render the
reference engines do. Dropped anywhere, the reference scores the reply under a fresh
generation prompt the server was never sent, and every position measures the harness.
The calibration capture's half is tested with calibration.

The tokenizer here is a real transformers tokenizer over a character vocabulary with a
template shaped like GLM's: an assistant message is its role tag and its content with
nothing after it, and the generation prompt is the assistant tag and an opening thought
tag. So the continued render and the fresh one differ by exactly those two ids, as they
do on the task's checkpoint.
"""
from __future__ import annotations

import asyncio
import types

import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent

_SPECIALS = ["[gMASK]", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
             "<think>", "</think>"]
_TEMPLATE = ("[gMASK]{% if tools %}<|system|>tools={{ tools | length }}{% endif %}"
             "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}{% endfor %}"
             "{% if add_generation_prompt %}<|assistant|><think>{% endif %}")
_TOOLS = [{"type": "function", "function": {"name": "bash"}}]
_FIRST_TURN = [{"role": "user", "content": "Fix it."}]
_CONTINUED = _FIRST_TURN + [{"role": "assistant", "content": "I am thinki</think>"}]


@pytest.fixture(scope="module")
def tokenizer():
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    chars = [chr(i) for i in range(32, 127)]
    vocab = {"[UNK]": 0, **{c: i + 1 for i, c in enumerate(chars)}}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = tokenizers.pre_tokenizers.Split(
        tokenizers.Regex("."), behavior="isolated")
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    tok.add_special_tokens({"additional_special_tokens": _SPECIALS})
    tok.chat_template = _TEMPLATE
    return tok


def _ids(tokenizer, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def _driver_ids(tokenizer, messages, continued: bool) -> list[int]:
    """The prompt ids the driver counts for a turn, through the call the task
    framework makes (``tasks/agentic_coding/framework.py``, ``_count_prompt_ids``)."""
    from hls.reference.engines import chat_prompt_ids
    return chat_prompt_ids(tokenizer, messages, _TOOLS, {}, continued)


def _event(messages, *, continued: bool, served=(40, 41, 42)) -> TraceEvent:
    meta = {"messages": messages, "tools": _TOOLS, "tenant_id": "t0",
            "max_tokens": 1_000}
    if continued:
        meta["continue_final_message"] = True
    req = "s.1" if continued else "s.0"
    inp = Envelope(request_id=req, contents=[TextPayload(text="x")], metadata=meta)
    out = Envelope(request_id=req, contents=[TextPayload(text="abc", tokens=list(served))])
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out),
                      ground_truth=None, t_send=0.0, t_recv=1.0)


def _item(ev: TraceEvent, **extra) -> Envelope:
    """The reference item the flip gate builds for ``ev``, plus ``extra`` metadata."""
    from hls.verify.verifiers import _reference_meta
    meta = {**_reference_meta(ev, "test"), **extra}
    return Envelope(request_id=ev.request_id, contents=ev.response.output.contents,
                    metadata=meta)


# --- the fixture template has the property the tests rest on ---------------------


def test_the_fixture_template_differs_from_a_fresh_turn_by_the_generation_prompt(tokenizer):
    continued = _driver_ids(tokenizer, _CONTINUED, True)
    fresh = _driver_ids(tokenizer, _CONTINUED, False)
    assert continued == _ids(tokenizer, "[gMASK]<|system|>tools=1<|user|>Fix it."
                                        "<|assistant|>I am thinki</think>")
    assert fresh == continued + _ids(tokenizer, "<|assistant|><think>")
    assert len(fresh) - len(continued) == 2


# --- the flag travels into the reference item ------------------------------------


def test_the_reference_item_carries_the_continuation():
    from hls.verify.verifiers import _reference_meta

    assert _reference_meta(_event(_CONTINUED, continued=True), "t")[
        "continue_final_message"] is True
    assert "continue_final_message" not in _reference_meta(
        _event(_FIRST_TURN, continued=False), "t")


def test_a_continuation_that_resumes_no_assistant_message_is_refused():
    # The wire would ask the server to resume a user message, which no loop sends:
    # a wiring fault, refused before any reference render could hide it.
    from hls.verify.verifiers import _reference_meta

    with pytest.raises(ValueError, match="assistant"):
        _reference_meta(_event(_FIRST_TURN, continued=True), "t")


def test_the_report_gate_item_carries_the_continuation():
    from hls.core.decode import Sample
    from hls.core.payloads import TokenLogprob, TopLogprob
    from hls.reference.references import HFReference
    from hls.verify import ReportFaithfulnessVerifier
    from hls.verify.report import TOP_LOGPROBS_REQUESTED
    from tests.unit.calibration_fixtures import make_report_calibration

    def entry(served):
        alts = [TopLogprob(token="a", token_id=i, logprob=-3.0 - i)
                for i in range(TOP_LOGPROBS_REQUESTED)]
        return TokenLogprob(token="t", token_id=served, logprob=-3.0 - served,
                            top_logprobs=alts)

    v = ReportFaithfulnessVerifier(reference=HFReference(decode=Sample(temperature=1.0)),
                                   calibration=make_report_calibration())
    for messages, continued in ((_CONTINUED, True), (_FIRST_TURN, False)):
        ev = _event(messages, continued=continued, served=(1, 2))
        tp = ev.response.output.contents[0]
        ev.response.output.contents[0] = tp.model_copy(
            update={"logprobs": [entry(1), entry(2)], "finish_reason": "stop"})
        built = v._rows_for(ev)
        assert not isinstance(built, str), built
        _rows, item = built
        assert bool(item.metadata.get("continue_final_message")) is continued


# --- the vLLM reference renders what the driver rendered -------------------------


def _vllm_engine(tokenizer):
    from hls.core.decode import Sample
    from hls.core.model import ModelSpec
    from hls.reference.engines import VLLMEngine

    eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                     decode=Sample(temperature=1.0))
    eng._tokenizer = tokenizer
    eng._llm = types.SimpleNamespace(collective_rpc=lambda *a, **k: None)
    eng._load = lambda: None
    eng._eos_ids = lambda: {7}
    eng._vocab_size = lambda: 1_000
    eng._declared_max_context = lambda: 4_096
    seen: list[list[int]] = []

    def score_one(prompt_ids, continuation, report_ids, eot_column, stop_row=False):
        seen.append(list(prompt_ids))
        n = len(continuation)
        return {"p_on_q": [[0.0]] * n, "p_tail_lp": [0.0] * n, "p_eot_lp": [0.0] * n}

    eng._score_one = score_one
    return eng, seen


@pytest.mark.parametrize("messages,continued", [(_CONTINUED, True), (_FIRST_TURN, False)])
def test_the_vllm_reference_renders_the_prompt_the_driver_rendered(tokenizer, messages,
                                                                   continued):
    eng, seen = _vllm_engine(tokenizer)
    ev = _event(messages, continued=continued)
    item = _item(ev, report_ids=[[40], [41], [42]])
    reply, = asyncio.run(eng.reference_logprobs([item]))
    assert seen == [_driver_ids(tokenizer, messages, continued)]
    assert reply["n_ctx"] == len(seen[0])
    ends_fresh = seen[0][-2:] == _ids(tokenizer, "<|assistant|><think>")
    assert ends_fresh is not continued


# --- the HF reference renders what the driver rendered ---------------------------


def _hf_engine(tokenizer, decode):
    torch = pytest.importorskip("torch")
    from hls.core.model import ModelSpec
    from hls.reference.engines import HFEngine

    vocab = 1_000
    eng = HFEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"), decode=decode)
    fed: list[list[int]] = []

    class _Config:
        final_logit_softcapping = None

        def get_text_config(self, decoder=None, encoder=None):
            return self

    class _Model:
        config = _Config()
        generation_config = types.SimpleNamespace(eos_token_id=[7])

        def model(self, ids, past_key_values=None, use_cache=True):
            fed.append(ids[0].tolist())
            n = ids.shape[1]
            return types.SimpleNamespace(last_hidden_state=torch.zeros(1, n, vocab),
                                         past_key_values=None)

        def lm_head(self, h):
            return h

        def get_input_embeddings(self):
            return types.SimpleNamespace(num_embeddings=vocab)

    eng._model, eng._tokenizer, eng._device = _Model(), tokenizer, "cpu"
    eng._load = lambda: None
    eng._eos_ids = lambda: {7}
    return eng, fed


@pytest.mark.parametrize("messages,continued", [(_CONTINUED, True), (_FIRST_TURN, False)])
def test_the_hf_teacher_force_renders_the_prompt_the_driver_rendered(tokenizer, messages,
                                                                     continued):
    from hls.core.decode import Greedy

    eng, fed = _hf_engine(tokenizer, Greedy())
    ev = _event(messages, continued=continued)
    asyncio.run(eng.teacher_force([_item(ev)]))
    full = [t for chunk in fed for t in chunk]
    assert full == _driver_ids(tokenizer, messages, continued) + [40, 41, 42]


@pytest.mark.parametrize("messages,continued", [(_CONTINUED, True), (_FIRST_TURN, False)])
def test_the_hf_report_reference_renders_the_prompt_the_driver_rendered(tokenizer, messages,
                                                                        continued):
    from hls.core.decode import Sample

    eng, fed = _hf_engine(tokenizer, Sample(temperature=1.0))
    ev = _event(messages, continued=continued)
    reply, = asyncio.run(eng.reference_logprobs(
        [_item(ev, report_ids=[[40], [41], [42]])]))
    full = [t for chunk in fed for t in chunk]
    driver = _driver_ids(tokenizer, messages, continued)
    assert full == driver + [40, 41, 42]
    assert reply["n_ctx"] == len(driver)


def test_an_engine_refuses_a_continuation_that_resumes_no_assistant_message():
    from hls.reference.engines import continues_of

    env = Envelope(request_id="r", contents=[TextPayload(text="q")],
                   metadata={"messages": _FIRST_TURN, "continue_final_message": True})
    with pytest.raises(ValueError, match="assistant"):
        continues_of(env)
