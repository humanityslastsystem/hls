"""Engines built from a task read the contract's conditioning and stop ids."""
import types

import pytest

from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.reference.engines import (HFEngine, _check_pinned_stop_ids,
                                   contract_conditioning)
from hls.reference.references import HFReference, SGLangReference, VLLMReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OpenAIChatProtocol, OpenAICompletionProtocol
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier

RAW = OpenAICompletionProtocol(stop_token_ids=(7, 9))
CHAT = OpenAIChatProtocol()


def _task(protocol, verifiers=None):
    return TaskSpec(
        id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="H200", count=1)),
        contract=RequestResponseContract(
            path="/v1/completions", input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"}), protocol=protocol),
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()),
        verifiers=verifiers or [MetricVerifier(metric=LetterMatchMetric())],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


class _NoTemplateTokenizer:
    """Raises if a chat template is ever applied; tokenizes by word."""
    eos_token_id = 2

    def apply_chat_template(self, *a, **k):
        raise AssertionError("chat template applied under raw conditioning")

    def __call__(self, text, add_special_tokens=False, **k):
        assert add_special_tokens is False
        return types.SimpleNamespace(input_ids=[len(w) for w in text.split()])


def test_contract_conditioning_reads_the_protocol():
    assert contract_conditioning(_task(RAW)) == {"conditioning": "raw", "stop_token_ids": (7, 9)}
    assert contract_conditioning(_task(CHAT)) == {"conditioning": "chat", "stop_token_ids": None}


@pytest.mark.parametrize("ref", [HFReference(decode=Greedy()), VLLMReference(decode=Greedy()),
                                 SGLangReference(decode=Greedy())])
def test_build_engine_threads_the_contract(ref):
    raw = ref.build_engine(_task(RAW))
    assert (raw.conditioning, raw.stop_token_ids) == ("raw", (7, 9))
    chat = ref.build_engine(_task(CHAT))
    assert (chat.conditioning, chat.stop_token_ids) == ("chat", None)


def test_engine_refuses_an_unknown_conditioning():
    with pytest.raises(ValueError, match="conditioning"):
        HFEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"), decode=Greedy(),
                 conditioning="bogus")


def test_raw_context_is_the_final_message_verbatim_and_never_templated():
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    eng._tokenizer = _NoTemplateTokenizer()
    msgs = [{"role": "user", "content": "<|fim_prefix|>a b<|fim_suffix|>c<|fim_middle|>"}]
    assert eng._render_context(msgs) == msgs[0]["content"]
    assert eng._context_ids(msgs) == [len(w) for w in msgs[0]["content"].split()]


def test_chat_context_still_goes_through_the_template():
    class _Tok:
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
            return "<user>" + messages[-1]["content"] + "<assistant>"

        def __call__(self, text, add_special_tokens=False, **k):
            return types.SimpleNamespace(input_ids=[1, 2, 3])
    eng = HFReference(decode=Greedy()).build_engine(_task(CHAT))
    eng._tokenizer = _Tok()
    assert eng._render_context([{"role": "user", "content": "q"}]) == "<user>q<assistant>"
    assert eng._context_ids([{"role": "user", "content": "q"}]) == [1, 2, 3]


def test_pinned_stop_ids_replace_the_checkpoints_eos_set():
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    eng._tokenizer = _NoTemplateTokenizer()                       # eos 2
    eng._model = types.SimpleNamespace(
        generation_config=types.SimpleNamespace(eos_token_id=[2, 3]))
    assert eng._eos_ids() == {7, 9}                               # not a union with {2, 3}
    chat = HFReference(decode=Greedy()).build_engine(_task(CHAT))
    chat._tokenizer, chat._model = eng._tokenizer, eng._model
    assert chat._eos_ids() == {2, 3}


def test_calibrate_engines_are_built_from_the_same_contract():
    """A calibration pool is pinned to the decode its task's calibrated verifier
    declares, so the task carries a TokenFaithfulnessVerifier; what is asserted is that
    the conditioning reaches every engine the pool builds."""
    from hls.calibrate import member_engines, pinned_reference
    from hls.verify import TokenFaithfulnessVerifier

    task = _task(RAW, verifiers=[
        TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))])
    for build in member_engines(task).values():
        eng = build()
        assert (eng.conditioning, eng.stop_token_ids) == ("raw", (7, 9))
    pinned = pinned_reference(task)
    assert (pinned.conditioning, pinned.stop_token_ids) == ("raw", (7, 9))


def test_reference_fingerprint_folds_the_protocol():
    ref = HFReference(decode=Greedy())
    assert ref.fingerprint(_task(CHAT)) != ref.fingerprint(_task(RAW))
    assert ref.fingerprint(_task(RAW)) != ref.fingerprint(
        _task(OpenAICompletionProtocol(stop_token_ids=(7,))))
    assert ref.fingerprint(_task(RAW)) == ref.fingerprint(_task(RAW))


# --- the stop row under pinned ids (real tensors; skipped where torch is absent) ---

def _stub_engine(logprob_rows, *, protocol, vocab=16):
    """HFEngine with the model/tokenizer stubbed, built from a task whose protocol pins
    the stop ids. Row i of ``logprob_rows`` is the logits for position i; the context
    is one token. Mirrors tests/unit/test_references.py::_stub_engine."""
    torch = pytest.importorskip("torch")
    eng = HFReference(decode=Greedy()).build_engine(_task(protocol))
    rows = torch.tensor(logprob_rows, dtype=torch.float32)

    class _Config:
        """Declares no attention structure and no logit softcapping, so the scoring pass
        builds transformers' stock cache and scores the stub's rows as they stand. A
        text-only config answers ``get_text_config`` with itself."""

        final_logit_softcapping = None

        def get_text_config(self, decoder=None, encoder=None):
            return self

    class _Model:
        def model(self, ids, past_key_values=None, use_cache=True):
            n = ids.shape[1]
            taken = getattr(self, "_seen", 0)
            self._seen = taken + n
            return type("O", (), {"last_hidden_state": rows[taken:taken + n].unsqueeze(0),
                                  "past_key_values": None})()

        def lm_head(self, h):
            return h

        def get_input_embeddings(self):
            return type("E", (), {"num_embeddings": vocab})()

        config = _Config()
        generation_config = type("G", (), {"eos_token_id": [2]})()   # the checkpoint's own

    class _Tok(_NoTemplateTokenizer):
        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            class _Enc(dict):
                def to(self, _device):
                    return self
            return _Enc(input_ids=torch.tensor([[0]]))

    eng._model, eng._tokenizer, eng._device = _Model(), _Tok(), "cpu"
    eng._load = lambda: None
    return eng


def _rows(n):
    return [[0.0] * 16 for _ in range(n)]


def test_a_reply_ending_on_a_pinned_terminator_scores_no_stop_row():
    rows = _rows(3)
    rows[0][1] = 5.0
    rows[1][9] = 5.0        # served token 9 is a pinned terminator and the argmax
    rows[2][4] = 40.0       # would-be stop row: enormous if scored
    eng = _stub_engine(rows, protocol=RAW)
    out = eng._teacher_force_prepared(
        [([{"role": "user", "content": "p"}], None, [1, 9], True)])
    assert out[0] == pytest.approx([0.0, 0.0], abs=1e-5)


def test_a_reply_that_is_only_a_terminator_scores_one_position_not_zero():
    rows = _rows(2)
    rows[0][9] = 5.0
    eng = _stub_engine(rows, protocol=RAW)
    out = eng._teacher_force_prepared(
        [([{"role": "user", "content": "p"}], None, [9], True)])
    assert out[0] == pytest.approx([0.0], abs=1e-5)


def test_the_stop_row_is_scored_against_the_pinned_ids_not_the_checkpoints():
    rows = _rows(2)
    rows[0][1] = 5.0
    rows[1][2] = 5.0        # the checkpoint's eos (2) is the argmax, but it is not pinned
    rows[1][7] = 4.0        # best pinned terminator is 1 nat behind
    eng = _stub_engine(rows, protocol=RAW)
    out = eng._teacher_force_prepared(
        [([{"role": "user", "content": "p"}], None, [1], True)])
    assert out[0][1] == pytest.approx(1.0, abs=1e-5)


def test_hf_generate_stops_on_the_pinned_ids(monkeypatch):
    """model.generate only stops on the ids it is told; the pinned set must reach it."""
    torch = pytest.importorskip("torch")
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    seen = {}

    class _Model:
        generation_config = types.SimpleNamespace(eos_token_id=2)

        def generate(self, **kw):
            seen.update(kw)
            return torch.tensor([[0, 5, 9, 9]])                    # ctx + 5, 9, pad

    class _Tok(_NoTemplateTokenizer):
        padding_side = "left"
        pad_token_id = 2
        pad_token = "x"

        def __call__(self, texts, return_tensors=None, padding=True, add_special_tokens=False):
            class _Enc(dict):
                def to(self, _d):
                    return self
            return _Enc(input_ids=torch.tensor([[0]] * len(texts)))

        def decode(self, ids, skip_special_tokens=True):
            return "t"

    eng._model, eng._tokenizer, eng._device = _Model(), _Tok(), "cpu"
    monkeypatch.setattr(eng, "_sdpa_ctx", lambda: __import__("contextlib").nullcontext())
    decoded, finish, tokens = eng._generate_uniq(["p"], 8)
    assert sorted(seen["eos_token_id"]) == [7, 9]
    assert finish["p"] == "stop" and tokens["p"] == [5, 9]


def test_vllm_raw_conditioning_generates_from_ids_not_chat(monkeypatch):
    import sys
    calls = {}

    class FakeSamplingParams:
        def __init__(self, **kw):
            calls["sp"] = kw

    class FakeLLM:
        def __init__(self, **kw):
            pass

        def chat(self, **kw):
            raise AssertionError("LLM.chat used under raw conditioning")

        def generate(self, prompts, *, sampling_params, use_tqdm):
            calls["prompts"] = prompts
            out = types.SimpleNamespace(text="m", finish_reason="stop", token_ids=[4, 9])
            return [types.SimpleNamespace(outputs=[out]) for _ in prompts]

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    eng = VLLMReference(decode=Greedy()).build_engine(_task(RAW))
    eng._llm = FakeLLM()
    eng._tokenizer = _NoTemplateTokenizer()
    out = eng._generate_uniq(["k"], [[{"role": "user", "content": "a bb ccc"}]], [None], 8)
    assert calls["prompts"] == [{"prompt_token_ids": [1, 2, 3]}]
    assert calls["sp"]["stop_token_ids"] == [7, 9] and calls["sp"]["max_tokens"] == 8
    # No ignore_eos: sglang nests its stop_token_ids check inside that flag, so setting
    # it would turn the pinned ids off on one stack and not the other.
    assert "ignore_eos" not in calls["sp"]
    assert out["k"] == ("m", "stop", [4, 9])


def test_sglang_raw_conditioning_feeds_ids_and_stop_ids():
    calls = {}

    class FakeEngine:
        def generate(self, input_ids, sampling_params, return_logprob):
            calls.update(input_ids=input_ids, sp=sampling_params)
            return [{"text": "m", "meta_info": {"output_token_logprobs": [(0.0, 4, "m")],
                                                "finish_reason": {"type": "stop"}}}]

    eng = SGLangReference(decode=Greedy()).build_engine(_task(RAW))
    eng._engine, eng._tokenizer = FakeEngine(), _NoTemplateTokenizer()
    out = eng._generate_uniq(["a bb"], 8)
    assert calls["input_ids"] == [[1, 2]]
    assert calls["sp"]["stop_token_ids"] == [7, 9]
    assert "ignore_eos" not in calls["sp"]       # it would disable the ids just set
    assert out["a bb"] == ("m", "stop", [4])


def test_an_unpinned_contract_sets_no_stop_fields(monkeypatch):
    """No pinned ids means no pin: neither stack is told anything about terminators, so
    both keep ending turns on the checkpoint's own eos, as the pinned reference does."""
    import sys
    calls = {}

    class FakeSamplingParams:
        def __init__(self, **kw):
            calls["vllm"] = kw

    class FakeLLM:
        def chat(self, *, messages, sampling_params, use_tqdm):
            out = types.SimpleNamespace(text="m", finish_reason="stop", token_ids=[4])
            return [types.SimpleNamespace(outputs=[out]) for _ in messages]

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=FakeLLM, SamplingParams=FakeSamplingParams))
    vllm_engine = VLLMReference(decode=Greedy()).build_engine(_task(CHAT))
    vllm_engine._llm = FakeLLM()
    vllm_engine._generate_uniq(["k"], [[{"role": "user", "content": "q"}]], [None], 8)
    assert "stop_token_ids" not in calls["vllm"] and "ignore_eos" not in calls["vllm"]

    class FakeEngine:
        def generate(self, input_ids, sampling_params, return_logprob):
            calls["sglang"] = sampling_params
            return [{"text": "m", "meta_info": {"output_token_logprobs": [(0.0, 4, "m")],
                                                "finish_reason": {"type": "stop"}}}]

    sglang_engine = SGLangReference(decode=Greedy()).build_engine(_task(CHAT))
    sglang_engine._engine, sglang_engine._tokenizer = FakeEngine(), _NoTemplateTokenizer()
    sglang_engine._generate_uniq(["a bb"], 8)
    assert "stop_token_ids" not in calls["sglang"] and "ignore_eos" not in calls["sglang"]


# --- the clocked path calibration drives (_serve_one on each serving stack) ---

def _raw_prompt(text="a bb ccc", cap=8):
    """One request envelope carrying a pre-rendered raw prompt and its output cap."""
    return Envelope(request_id="r1", contents=[TextPayload(text=text)],
                    metadata={"messages": [{"role": "user", "content": text}],
                              "max_tokens": cap})


async def test_vllm_serve_one_sends_ids_and_the_pinned_stop_ids(monkeypatch):
    """The clocked path conditions the same way the batch path does: the prompt reaches
    the engine as _context_ids, never as templated text, and the pinned terminators ride
    along without ignore_eos."""
    import sys
    calls = {}

    class FakeSamplingParams:
        def __init__(self, **kw):
            calls["sp"] = kw

    class FakeAsyncLLM:
        async def generate(self, *, request_id, prompt, sampling_params):
            calls["prompt"] = prompt
            out = types.SimpleNamespace(text="m", finish_reason="stop", token_ids=[4, 9])
            yield types.SimpleNamespace(outputs=[out])

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        SamplingParams=FakeSamplingParams))
    eng = VLLMReference(decode=Greedy()).build_engine(_task(RAW))
    eng._async_llm, eng._tokenizer = FakeAsyncLLM(), _NoTemplateTokenizer()
    # The report is None under a greedy decode, which is graded on the tokens
    # themselves; 3 is the length of the rendered prompt, which the reply reports the
    # way a graded server reports its own prompt_tokens.
    assert await eng._serve_one(_raw_prompt()) == ("m", "stop", [4, 9], None, 3)
    assert calls["prompt"] == {"prompt_token_ids": [1, 2, 3]}
    assert calls["sp"]["stop_token_ids"] == [7, 9] and calls["sp"]["max_tokens"] == 8
    assert "ignore_eos" not in calls["sp"]


async def test_sglang_serve_one_sends_ids_and_the_pinned_stop_ids():
    """As above on sglang, where ignore_eos would have disabled stop_token_ids outright
    and let every reply run to the output cap."""
    calls = {}

    class FakeEngine:
        async def async_generate(self, input_ids, sampling_params, return_logprob):
            calls.update(input_ids=input_ids, sp=sampling_params)
            return {"text": "m", "meta_info": {"output_token_logprobs": [(0.0, 4, "m")],
                                               "finish_reason": {"type": "stop"}}}

    eng = SGLangReference(decode=Greedy()).build_engine(_task(RAW))
    eng._engine, eng._tokenizer = FakeEngine(), _NoTemplateTokenizer()
    assert await eng._serve_one(_raw_prompt()) == ("m", "stop", [4], None, 3)
    assert calls["input_ids"] == [1, 2, 3]
    assert calls["sp"]["stop_token_ids"] == [7, 9] and calls["sp"]["max_new_tokens"] == 8
    assert "ignore_eos" not in calls["sp"]


# --- the pinned set must cover the checkpoint's own terminators ---

def test_a_pinned_set_missing_a_checkpoint_terminator_is_refused():
    """A serving stack can only add stop ids to the checkpoint's own, so a pinned set
    that drops one would have the server stopping where the reference reads an ordinary
    token. The helper names the ids the contract has to add."""
    with pytest.raises(RuntimeError, match=r"omits the checkpoint's own end-of-turn"):
        _check_pinned_stop_ids({2, 3}, (7, 9))
    with pytest.raises(RuntimeError, match=r"\[3\]"):
        _check_pinned_stop_ids({2, 3}, (2, 7))
    # A covering set, and an unpinned contract, both pass.
    assert _check_pinned_stop_ids({2, 3}, (2, 3, 7)) is None
    assert _check_pinned_stop_ids({2, 3}, None) is None


def _checkpoint_transformers(tokenizer_eos=2, generation_eos=(2, 3)):
    """A transformers stand-in answering the two reads the stop-id guard makes: the
    tokenizer's eos and the generation config's. Neither needs a built model, which is
    what lets every engine run the guard from its own ``_load``."""
    tf = types.ModuleType("transformers")
    tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: types.SimpleNamespace(eos_token_id=tokenizer_eos))
    tf.GenerationConfig = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: types.SimpleNamespace(
            eos_token_id=list(generation_eos)))
    tf.AutoConfig = types.SimpleNamespace(from_pretrained=lambda src, **kw: (
        types.SimpleNamespace(architectures=["FauxForCausalLM"], model_type="faux",
                              vision_config=None)))
    return tf


COVERING = OpenAICompletionProtocol(stop_token_ids=(2, 3, 7, 9))
CHAT_COVERING = OpenAIChatProtocol(stop_token_ids=(2, 3, 7, 9))


def test_hf_load_refuses_a_pinned_set_that_drops_the_checkpoints_eos(monkeypatch):
    """The guard runs off the checkpoint's own config files, so it refuses before the
    weight load rather than after it."""
    import sys

    built = []

    class FakeModel:
        generation_config = types.SimpleNamespace(eos_token_id=[2, 3])

        def named_modules(self):
            return [("", self)]

        def to(self, device):
            return self

        def eval(self):
            return self

    # A clean load report: nothing missing, nothing shape-mismatched.
    clean = {"missing_keys": set(), "unexpected_keys": set(), "mismatched_keys": set()}
    tf = _checkpoint_transformers()
    # A models namespace with no submodules: the kernel pin looks for this model_type's
    # modeling module and treats an absent one as nothing to pin.
    tf.models = types.ModuleType("transformers.models")
    tf.AutoModelForCausalLM = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: (built.append(src) or FakeModel(), clean))
    fake_torch = types.SimpleNamespace(
        bfloat16="bf16", float32="fp32",
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(enable_cudnn_sdp=lambda flag: None)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "transformers.models", tf.models)

    # The provisioned snapshot the engine loads from; the fake library reads nothing.
    from hls.reference.engines import HFEngine
    monkeypatch.setattr(HFEngine, "_weights_source", lambda self: "/w")
    # The checkpoint ends on 2 and 3; the contract pins 7 and 9.
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    with pytest.raises(RuntimeError, match=r"omits the checkpoint's own end-of-turn"):
        eng._load()
    assert built == []
    # The same load with a covering set gets through to a usable model.
    covering = HFReference(decode=Greedy()).build_engine(_task(COVERING))
    covering._load()
    assert built and covering._eos_ids() == {2, 3, 7, 9}


def test_vllm_load_refuses_a_pinned_set_that_drops_the_checkpoints_eos(monkeypatch):
    """A task's oracle can be vLLM (a `VLLMReference`), so the guard cannot sit on
    HFEngine alone: this engine feeds the pinned ids to its stack and would otherwise
    stop in a different place than the reference with nothing raised."""
    import sys
    from hls.reference.engines import VLLMEngine

    built = []
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=lambda **kw: built.append(kw) or object()))
    monkeypatch.setitem(sys.modules, "transformers", _checkpoint_transformers())
    monkeypatch.setattr(VLLMEngine, "_weights_source", lambda self: "/w")

    eng = VLLMReference(decode=Greedy()).build_engine(_task(RAW))
    with pytest.raises(RuntimeError, match=r"omits the checkpoint's own end-of-turn"):
        eng._load()
    assert built == []
    VLLMReference(decode=Greedy()).build_engine(_task(COVERING))._load()
    assert len(built) == 1


def test_vllm_serve_build_refuses_the_same_set(monkeypatch):
    """The clocked-serving face builds its engine on its own path, so it checks there
    too -- a calibration pool member is graded against the same pinned set."""
    import sys
    from hls.reference.engines import VLLMEngine

    class FakeAsyncLLM:
        @staticmethod
        def from_engine_args(args):
            raise AssertionError("built the async engine despite a bad pinned set")

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils",
                        types.SimpleNamespace(AsyncEngineArgs=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm",
                        types.SimpleNamespace(AsyncLLM=FakeAsyncLLM))
    monkeypatch.setitem(sys.modules, "transformers", _checkpoint_transformers())
    monkeypatch.setattr(VLLMEngine, "_weights_source", lambda self: "/w")

    eng = VLLMReference(decode=Greedy()).build_engine(_task(RAW))
    eng.clocked_serving = True
    with pytest.raises(RuntimeError, match=r"omits the checkpoint's own end-of-turn"):
        eng._load()


def test_sglang_load_refuses_a_pinned_set_that_drops_the_checkpoints_eos(monkeypatch):
    """As above on sglang, the other engine that can serve a task's declared load."""
    import sys
    from hls.reference.engines import SGLangEngine

    built = []
    monkeypatch.setitem(sys.modules, "sglang", types.SimpleNamespace(
        Engine=lambda **kw: built.append(kw) or object()))
    monkeypatch.setitem(sys.modules, "transformers", _checkpoint_transformers())
    monkeypatch.setattr(SGLangEngine, "_weights_source", lambda self: "/w")

    eng = SGLangReference(decode=Greedy()).build_engine(_task(RAW))
    with pytest.raises(RuntimeError, match=r"omits the checkpoint's own end-of-turn"):
        eng._load()
    assert built == []
    SGLangReference(decode=Greedy()).build_engine(_task(COVERING))._load()
    assert len(built) == 1


def test_a_pinned_load_reads_the_tokenizer_only_once(monkeypatch):
    """The guard and the engine must share one tokenizer rather than each building an
    identical one. Counting AutoTokenizer reads across a pinned load, on every engine
    that needs a tokenizer of its own, catches a second build."""
    import sys
    from hls.reference.engines import SGLangEngine, VLLMEngine

    tf = _checkpoint_transformers()
    reads: list[str] = []
    tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: reads.append(src) or types.SimpleNamespace(eos_token_id=2))
    monkeypatch.setitem(sys.modules, "transformers", tf)

    monkeypatch.setitem(sys.modules, "sglang", types.SimpleNamespace(
        Engine=lambda **kw: object()))
    monkeypatch.setattr(SGLangEngine, "_weights_source", lambda self: "/w")
    reads.clear()
    SGLangReference(decode=Greedy()).build_engine(_task(COVERING))._load()
    assert reads == ["/w"]

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kw: object()))
    monkeypatch.setattr(VLLMEngine, "_weights_source", lambda self: "/w")
    reads.clear()
    VLLMReference(decode=Greedy()).build_engine(_task(COVERING))._load()
    assert reads == ["/w"]  # raw conditioning: one read, stored and reused by the guard.

    # Chat conditioning keeps no tokenizer of its own, so the guard's read here is a
    # throwaway -- still one, not the two a naive fix would leave it at.
    reads.clear()
    chat_eng = VLLMReference(decode=Greedy()).build_engine(_task(CHAT_COVERING))
    chat_eng._load()
    assert reads == ["/w"] and chat_eng._tokenizer is None


def test_an_unpinned_contract_reads_no_checkpoint_files(monkeypatch):
    """The guard costs nothing when a contract pins nothing: with no declared set there
    is nothing the checkpoint's own ids could contradict."""
    import sys
    from hls.reference.engines import SGLangEngine

    tf = _checkpoint_transformers()
    tf.GenerationConfig = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: pytest.fail("read the generation config"))
    monkeypatch.setitem(sys.modules, "sglang", types.SimpleNamespace(
        Engine=lambda **kw: object()))
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setattr(SGLangEngine, "_weights_source", lambda self: "/w")
    SGLangReference(decode=Greedy()).build_engine(_task(CHAT))._load()


def test_vllm_loads_a_tokenizer_only_under_raw_conditioning(monkeypatch):
    """Under chat conditioning LLM.chat templates and tokenizes inside vLLM and nothing
    reads this engine's tokenizer, so the oracle deploy must not block on loading one."""
    import sys
    from hls.reference.engines import VLLMEngine

    loaded = []
    tf = _checkpoint_transformers()
    tf.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda src, **kw: loaded.append(src) or "tok")
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=lambda **kw: object()))
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setattr(VLLMEngine, "_weights_source", lambda self: "/w")

    chat = VLLMReference(decode=Greedy()).build_engine(_task(CHAT))
    chat._load()
    assert loaded == [] and chat._tokenizer is None
    raw = VLLMReference(decode=Greedy()).build_engine(_task(OpenAICompletionProtocol()))
    raw._load()
    assert loaded == ["/w"] and raw._tokenizer == "tok"


def test_stop_sampling_belongs_to_the_serving_engines_only():
    """Its value is a vLLM SamplingParams / SGLang sampling-dict fragment. HFEngine ends
    a turn through generate(eos_token_id=...) and could never pass it, so it does not
    inherit it."""
    from hls.reference.engines import ReferenceEngine, SGLangEngine, VLLMEngine

    assert not hasattr(ReferenceEngine, "_stop_sampling")
    assert not hasattr(HFEngine, "_stop_sampling")
    assert hasattr(VLLMEngine, "_stop_sampling")
    assert hasattr(SGLangEngine, "_stop_sampling")
    # _render_context / _context_ids stay shared: all three engines render a context.
    for attr in ("_render_context", "_context_ids"):
        assert hasattr(ReferenceEngine, attr)


def test_wire_and_engine_read_a_null_message_content_the_same_way():
    """A pre-rendered conversation comes from a dataset's own JSON, which may carry a
    null ``content``. Under raw conditioning the oracle reproduces the exact bytes the
    server was sent, so the wire cannot send "" while the reference forces "None"."""
    from hls.reference.engines import _messages_of

    env = Envelope(request_id="r1", contents=[TextPayload(text="fallback")],
                   metadata={"messages": [{"role": "user", "content": None}]})
    on_the_wire = OpenAICompletionProtocol().encode_request(env)["prompt"]
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    eng._tokenizer = _NoTemplateTokenizer()
    assert on_the_wire == ""
    assert eng._render_context(_messages_of(env)) == on_the_wire
    # The chat pair reads it the same way, from the same helper: the null survives
    # verbatim, because that is how the OpenAI schema spells a turn carrying no text.
    assert (OpenAIChatProtocol().encode_request(env)["messages"] == _messages_of(env)
            == [{"role": "user", "content": None}])


def _tf_item(messages, *, tokens=(1, 2), cap=8):
    """One teacher_force item: the conversation the request carried, the continuation
    ids and the stamped output cap, shaped as TokenFaithfulnessVerifier._item does."""
    return Envelope(request_id="r1",
                    contents=[TextPayload(text="cont", tokens=list(tokens))],
                    metadata={"messages": list(messages), "max_tokens": cap})


async def _prepared_messages(engine, item):
    """The context ``teacher_force`` prepares for *item*, with the model load and the
    forward passes stubbed out -- the preparation loop runs before either."""
    seen: list = []
    engine._load = lambda: None
    engine._teacher_force_prepared = lambda prepared: (
        seen.extend(m for m, *_ in prepared) or [[] for _ in prepared])
    await engine.teacher_force([item])
    return seen[0]


async def test_teacher_force_drops_the_messages_the_wire_dropped():
    """A pre-rendered conversation comes from a dataset's own JSON and may carry an
    entry with no ``content``. The wire drops it; teacher_force has to render the same
    list, or a faithful reply is scored against a context it was never sent."""
    from hls.reference.engines import _messages_of

    item = _tf_item([{"role": "user", "content": "A"}, {"role": "assistant"}])
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    eng._tokenizer = _NoTemplateTokenizer()
    prepared = await _prepared_messages(eng, item)
    assert prepared == _messages_of(item) == [{"role": "user", "content": "A"}]
    # Raw conditioning renders the last message, so the dropped entry decided the whole
    # prompt: unfiltered it would have been the assistant's forced "".
    assert (eng._render_context(prepared)
            == OpenAICompletionProtocol().encode_request(item)["prompt"] == "A")


async def test_teacher_force_does_not_raise_on_a_message_with_no_role():
    """The other malformed shape: a missing ``role`` must not raise KeyError out of the
    preparation loop, which is no ArtifactFault and would take the verification pass down."""
    item = _tf_item([{"role": "user", "content": "A"}, {"content": "B"}])
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    assert await _prepared_messages(eng, item) == [{"role": "user", "content": "A"}]


def test_raw_context_of_an_empty_conversation_is_the_empty_prompt():
    """Every entry malformed leaves no message. The wire sends "" for that list, so the
    reference renders "" rather than raising on the missing final message."""
    eng = HFReference(decode=Greedy()).build_engine(_task(RAW))
    eng._tokenizer = _NoTemplateTokenizer()
    env = Envelope(request_id="r1", contents=[TextPayload(text="x")],
                   metadata={"messages": [{"role": "user"}]})
    assert eng._render_context([]) == OpenAICompletionProtocol().encode_request(env)["prompt"] == ""


def test_contract_conditioning_refuses_a_contract_with_no_wire_protocol():
    """A bare ArtifactContract is legal (TaskSpec.contract is typed as the base) and says
    nothing about how the prompt reached the server; defaulting it to chat would build a
    whole calibration pool on the guess."""
    from hls.spec.contracts import ArtifactContract

    bare = ArtifactContract(path="/v1/x", input_modalities=frozenset({"text"}),
                            output_modalities=frozenset({"text"}))
    with pytest.raises(TypeError, match="declares no wire protocol"):
        contract_conditioning(types.SimpleNamespace(id="T", contract=bare))


def test_declared_eos_ids_reads_sub_config_terminators_on_the_config_json_fallback(monkeypatch):
    """A checkpoint with no generation_config.json can keep its eos id in a sub-config
    (``text_config`` on a vision-language layout). transformers' own fallback passes
    ``_from_model_config=True``, which is what reaches those sub-configs; without it the
    pre-load guard sees fewer terminators than the loaded model ends on and lets a
    narrowing pin through. Measured on Qwen/Qwen3.5-9B: eos None without the flag,
    248044 with it, against tokenizer eos 248046."""
    import sys
    from hls.reference.engines import _declared_eos_ids

    def from_pretrained(src, *, config_file_name=None, local_files_only=False, **kw):
        if config_file_name is None:
            raise OSError("no generation_config.json")
        return types.SimpleNamespace(
            eos_token_id=248044 if kw.get("_from_model_config") else None)

    tf = types.ModuleType("transformers")
    tf.GenerationConfig = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", tf)
    tok = types.SimpleNamespace(eos_token_id=248046)
    assert _declared_eos_ids(tok, "Qwen/Qwen3.5-9B") == {248044, 248046}
    # And the guard reading it then refuses a pin covering only the tokenizer's id.
    with pytest.raises(RuntimeError, match=r"\[248044\]"):
        _check_pinned_stop_ids(_declared_eos_ids(tok, "Qwen/Qwen3.5-9B"), (248046,))
