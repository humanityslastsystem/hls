import asyncio
import sys
import types

import pytest

from hls.core.capabilities import Generate, ReferenceLogprobs, StackReport, TeacherForce
from hls.core.decode import Greedy
from hls.core.engine import supports
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.reference.engines import HFEngine
from hls.reference.references import HFReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier


def _model():
    return ModelSpec(base_checkpoint="m", arch="TX")


def _task():
    return TaskSpec(
        id="T",
        model=_model(),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=RequestResponseContract(
            path="/v1/chat/completions",
            input_modalities=frozenset({"text"}),
            output_modalities=frozenset({"text"})),
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()),
        verifiers=[MetricVerifier(metric=LetterMatchMetric())],  # stub: a task needs >=1 verifier
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
    )


def _ref():
    return HFReference(decode=Greedy())


def test_hfreference_is_pure_config_and_serializes():
    from hls.core.poly import dump_spec

    rec = dump_spec(_ref())
    assert rec["kind"] == "HFReference"
    assert "max_new_tokens" not in rec    # field removed


def test_build_engine_returns_capable_engine():
    task = _task()
    eng = _ref().build_engine(task)
    assert isinstance(eng, HFEngine)
    assert supports(HFEngine) == frozenset(
        {Generate, TeacherForce, ReferenceLogprobs, StackReport})


def test_build_engine_uses_task_model():
    """HFReference.build_engine(task) uses task.model as the model_spec."""
    task = _task()
    eng = _ref().build_engine(task)
    assert eng.model_spec is task.model


def test_reference_fingerprint_preserved():
    # fingerprint(task) folds model + decode identity.
    task = _task()
    fp = _ref().fingerprint(task)
    assert isinstance(fp, str) and fp


def test_engine_resources_request_one_gpu():
    from hls.core.engine import ResourceRequest
    task = _task()
    eng = _ref().build_engine(task)
    assert eng.resources == ResourceRequest(gpu_count=1, gpu_kind=task.hardware.accelerator.kind)


def test_messages_of_prefers_metadata_messages():
    from hls.reference.engines import _messages_of
    from hls.core.payloads import Envelope, TextPayload
    env = Envelope(request_id="r1",
                   contents=[TextPayload(text="ignored when messages present")],
                   metadata={"messages": [{"role": "user", "content": "hi"}]})
    assert _messages_of(env) == [{"role": "user", "content": "hi"}]


def test_messages_of_falls_back_to_single_text_turn():
    from hls.reference.engines import _messages_of
    from hls.core.payloads import Envelope, TextPayload
    env = Envelope(request_id="r2", contents=[TextPayload(text="just this")])
    assert _messages_of(env) == [{"role": "user", "content": "just this"}]


def test_max_new_tokens_picks_max_positive_cap_else_none():
    from hls.reference.engines import _max_new_tokens
    from hls.core.payloads import Envelope, TextPayload
    def mk(cap):
        md = {"max_tokens": cap} if cap is not None else {}
        return Envelope(request_id="x", contents=[TextPayload(text="t")], metadata=md)
    assert _max_new_tokens([mk(4), mk(8), mk(None)]) == 8
    assert _max_new_tokens([mk(None), mk(0), mk(-1)]) is None
    assert _max_new_tokens([]) is None


def test_build_engine_threads_task_accelerator_kind():
    """The oracle slot requests the task's accelerator kind -- a pragmatic HARNESS
    default, not task semantics (oracle output must be accelerator-invariant for
    task-level metrics). gpu_kind=None maps to Modal's "any" and can land a card
    too small for the model (see ReferenceEngine.__init__)."""
    task = _task()
    eng = _ref().build_engine(task)
    assert eng.gpu_kind == "B200"
    assert eng.resources.gpu_kind == "B200"


def test_hf_finish_reason_stop_on_eos_else_length():
    # HF generate exposes no finish_reason; the engine derives it from the row:
    # any EOS id (multi-EOS models stop on the generation config's list, and
    # early-stopped rows are padded -- with EOS itself when pad is unset) means
    # a natural stop; a row with none was cut by the max_new_tokens cap.
    from hls.reference.engines import _hf_finish_reason
    assert _hf_finish_reason([5, 7, 2], {2}) == "stop"
    assert _hf_finish_reason([5, 7, 2, 2, 2], {2}) == "stop"   # eos-padded row
    assert _hf_finish_reason([5, 7, 9], {2, 128009}) == "length"
    assert _hf_finish_reason([5, 128009], {2, 128009}) == "stop"  # cfg-list EOS


def test_reference_reply_stamps_source_and_finish_reason():
    from hls.reference.engines import _reference_reply
    env = _reference_reply("r1", "hello", "stop")
    assert env.request_id == "r1"
    assert env.metadata == {"source": "reference"}
    tp = env.first_of("text")
    assert tp.text == "hello" and tp.finish_reason == "stop"
    assert tp.tokens is None                      # no ids attached -> None


def test_reference_reply_carries_emitted_tokens():
    # generate stamps the reference's emitted ids so a downstream teacher_force can
    # score the TRUE continuation (the whole point of tokens on replies).
    from hls.reference.engines import _reference_reply
    env = _reference_reply("r1", "hello", "stop", tokens=[3, 4, 5])
    assert env.first_of("text").tokens == [3, 4, 5]


def test_emitted_tokens_strips_trailing_eos_padding():
    # generate right-pads early-finished rows (with EOS, since pad_token is set to
    # eos), so the emitted sequence on a natural stop is up to and INCLUDING the first
    # EOS; a length stop (no EOS in the row) keeps the whole row. Mirrors the EOS test
    # in _hf_finish_reason so tokens and finish_reason agree.
    from hls.reference.engines import _emitted_tokens
    assert _emitted_tokens([5, 7, 2, 2, 2], {2}) == [5, 7, 2]     # eos-padded row
    assert _emitted_tokens([5, 7, 2], {2}) == [5, 7, 2]           # exact-fit natural stop
    assert _emitted_tokens([5, 128009], {2, 128009}) == [5, 128009]  # cfg-list EOS
    assert _emitted_tokens([5, 7, 9], {2}) == [5, 7, 9]           # length stop -> whole row
    assert _emitted_tokens([], {2}) == []


def test_hfengine_rejects_unknown_sdpa_backend():
    # The instrument pin is validated at CONSTRUCTION (pure string check, no torch),
    # so a bad reference config fails on the torch-free config host, before any deploy.
    with pytest.raises(ValueError, match="sdpa_backend"):
        HFEngine(model_spec=_model(), decode=Greedy(), sdpa_backend="bogus")


def test_hfengine_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="dtype"):
        HFEngine(model_spec=_model(), decode=Greedy(), dtype="float16")


def test_hfengine_default_pins_are_math_and_auto():
    # The reference contract: math kernel, auto (bf16) precision.
    eng = HFEngine(model_spec=_model(), decode=Greedy())
    assert eng.sdpa_backend == "math"
    assert eng.dtype == "auto"


def test_hfengine_accepts_calibrate_pool_alternates():
    # flash and the float32 anchor are legal ONLY so hls calibrate can build the
    # alternate-kernel / fp32 pool members; construction must accept them. "efficient"
    # is refused at construction: the mem-efficient kernel rejects GQA on every current
    # torch, so it cannot run any modern checkpoint.
    for backend in ("math", "flash"):
        assert HFEngine(model_spec=_model(), decode=Greedy(),
                        sdpa_backend=backend).sdpa_backend == backend
    with pytest.raises(ValueError, match="sdpa_backend"):
        HFEngine(model_spec=_model(), decode=Greedy(), sdpa_backend="efficient")
    assert HFEngine(model_spec=_model(), decode=Greedy(), dtype="float32").dtype == "float32"


def test_build_engine_uses_default_instrument_pins():
    # HFReference.build_engine passes no instrument kwargs -- a task's reference is
    # ALWAYS the ("math","auto") instrument; the alternates are never task semantics.
    eng = _ref().build_engine(_task())
    assert eng.sdpa_backend == "math"
    assert eng.dtype == "auto"


def test_load_requires_cuda_and_never_falls_back_to_cpu(monkeypatch):
    # NO CPU fallback: a CPU forward is a different numeric instrument, so a missing
    # CUDA device is a loud RuntimeError, never a silent CPU load. Inject a torch whose
    # cuda.is_available() is False so the guard fires before any real torch/transformers
    # import (this box is torch-free).
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    eng = HFEngine(model_spec=_model(), decode=Greedy())
    with pytest.raises(RuntimeError, match="different instrument"):
        eng._load()


def test_stack_report_refuses_non_cuda_torch(monkeypatch):
    """stack_report must refuse loudly on a torch build with no CUDA runtime
    (CPU-only, or ROCm -- where torch.version.cuda is None) instead of writing
    missing or misleading component versions into a calibration stack record."""
    fake_torch = types.SimpleNamespace(
        __version__="0.0.0-test",
        cuda=types.SimpleNamespace(is_available=lambda: False,
                                   get_device_name=lambda idx: "none"),
        version=types.SimpleNamespace(cuda=None),
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: None)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    eng = HFEngine(model_spec=_model(), decode=Greedy())
    with pytest.raises(RuntimeError, match="NVIDIA CUDA stacks only"):
        asyncio.run(eng.stack_report())


def test_stack_report_is_the_calibration_stacks_shape_with_every_leaf_a_plain_str(monkeypatch):
    """The token gate compares this dict, but for the device name, with the ``stack`` of each
    committed calibration, whose versions are strings: a leaf of another type (cuDNN reports an
    int) never equals its calibrated value, so every token-gated verify would refuse as a drifted
    stack. Exact types, never isinstance: torch's version is a ``TorchVersion``, a str subclass
    that compares as a version, and the report is recorded as plain data."""
    class TorchVersion(str):
        """Stands in for ``torch.torch_version.TorchVersion``."""

    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        __version__=TorchVersion("2.11.0+cu130"),
        cuda=types.SimpleNamespace(is_available=lambda: True,
                                   get_device_name=lambda idx: TorchVersion("NVIDIA B200")),
        version=types.SimpleNamespace(cuda=TorchVersion("13.0")),
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: 91002))))
    report = asyncio.run(HFEngine(model_spec=_model(), decode=Greedy()).stack_report())
    assert report == {"torch": "2.11.0+cu130", "runtime": {"cuda": "13.0", "cudnn": "91002"},
                      "device_name": "NVIDIA B200"}
    leaves = [report["torch"], report["device_name"], *report["runtime"].values()]
    assert all(type(v) is str for v in leaves), [type(v) for v in leaves]


# --- instrument guards: the checkpoint really is the checkpoint -------------------
# (HFEngine._load's two refusals. Auto-class resolution is pure python by design, so it
# is exercised directly here without torch; the weight-coverage refusal reads what the
# loader reports about the load, so it is exercised through a fake transformers below.
# The fused-kernel pin has its own module and tests: hls.reference.kernel_pins.)


def test_auto_class_resolves_from_the_checkpoints_own_config():
    """A multimodal checkpoint stores its body under model.language_model.* and must be
    built through the image-text auto-class. Getting this wrong does not raise -- it builds
    a differently-shaped tree and loads nothing into it -- which is why the resolution is
    read off the checkpoint, never assumed.

    The signal is a declared vision_config AND a ...ForConditionalGeneration architecture,
    not the name suffix alone: ~30 text-only encoder-decoder families (T5, BART, Pegasus,
    ...) share that suffix and fit neither branch."""
    from hls.reference.engines import _auto_class_name

    def cfg(architectures, **kw):
        return types.SimpleNamespace(architectures=architectures, **kw)

    assert _auto_class_name(cfg(["Qwen3_5ForConditionalGeneration"],
                                vision_config={"depth": 27})) == "AutoModelForImageTextToText"
    assert _auto_class_name(cfg(["Qwen2ForCausalLM"])) == "AutoModelForCausalLM"
    # Text-only encoder-decoder: NOT routed to the image-text class on its name.
    assert _auto_class_name(cfg(["T5ForConditionalGeneration"])) == "AutoModelForCausalLM"
    # No declaration at all: the text-only class is the safe default, and the
    # weight-coverage guard is what catches it if that was wrong.
    assert _auto_class_name(cfg(None)) == "AutoModelForCausalLM"
    assert _auto_class_name(cfg([])) == "AutoModelForCausalLM"


def _fake_transformers(architectures, model_type="faux", vision_config=None):
    """A transformers stand-in whose auto-classes build a model the caller controls."""
    tf = types.ModuleType("transformers")
    # The fused-kernel pin imports transformers.models.<model_type>.modeling_<model_type>.
    # A stand-in parent that is not a package makes that lookup miss the same way an
    # unknown family does, so the pin reports nothing to pin instead of raising.
    tf.models = types.ModuleType("transformers.models")
    tf.AutoConfig = types.SimpleNamespace(from_pretrained=lambda src, **kw: (
        types.SimpleNamespace(architectures=architectures, model_type=model_type,
                              vision_config=vision_config)))
    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda src, **kw: object())
    return tf


def _fake_torch():
    return types.SimpleNamespace(
        bfloat16="bf16", float32="fp32",
        cuda=types.SimpleNamespace(is_available=lambda: True),
        backends=types.SimpleNamespace(
            cuda=types.SimpleNamespace(enable_cudnn_sdp=lambda flag: None)),
    )


class _FakeModel:
    """The little of a built model that _load touches after from_pretrained."""

    def named_modules(self):
        return [("", self)]

    def to(self, device):
        return self

    def eval(self):
        return self


def _fake_load(monkeypatch, loading_info, model=None):
    """Install a fake torch and transformers whose image-text auto-class builds ``model``
    and reports ``loading_info``. Returns the list of kwargs it was called with."""
    built = []

    def from_pretrained(src, **kw):
        built.append(kw)
        # The guard's whole evidence: without this the loader returns the model alone.
        assert kw.get("output_loading_info") is True
        return (model if model is not None else _FakeModel()), loading_info

    tf = _fake_transformers(["FauxForConditionalGeneration"], vision_config={"depth": 4})
    tf.AutoModelForImageTextToText = types.SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "transformers.models", tf.models)
    # The provisioned snapshot the engine loads from; the fake library reads nothing.
    monkeypatch.setattr(HFEngine, "_weights_source", lambda self: "/w")
    return built


def test_load_refuses_parameters_the_checkpoint_did_not_supply(monkeypatch):
    """The random-initialization footgun end to end: the auto-class resolved for the
    checkpoint builds a tree the checkpoint does not populate, the loader reports those
    parameters missing, and the load must refuse loudly rather than hand the gate a
    randomly-initialized reference."""
    built = _fake_load(monkeypatch, {"missing_keys": {"model.layers.0.attn.weight",
                                                      "model.layers.0.mlp.weight"},
                                     "unexpected_keys": set(), "mismatched_keys": set()})

    eng = HFEngine(model_spec=_model(), decode=Greedy())
    with pytest.raises(RuntimeError, match="randomly initialized") as err:
        eng._load()
    assert "model.layers.0.attn.weight" in str(err.value)
    assert "model.layers.0.mlp.weight" in str(err.value)
    # It got as far as building through the class the architectures named, with the
    # instrument's own pins.
    assert built and built[0]["attn_implementation"] == "sdpa"


def test_load_refuses_a_parameter_whose_stored_shape_did_not_fit(monkeypatch):
    """A mismatched key was re-initialized too, so it is as fatal as a missing one. The
    loader reports it as the name followed by the two shapes."""
    _fake_load(monkeypatch, {"missing_keys": set(), "unexpected_keys": set(),
                             "mismatched_keys": {("lm_head.weight", (32000, 4096),
                                                  (32064, 4096))}})

    eng = HFEngine(model_spec=_model(), decode=Greedy())
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        eng._load()


def test_load_accepts_checkpoint_keys_the_model_does_not_hold(monkeypatch):
    """Unexpected keys are inert -- a multi-token-prediction head transformers does not
    instantiate is not part of the served model either -- so they do not block the load.
    Names the loader rewrote or fused on the way in are reported as filled, which is why
    the guard reads the loader's report instead of the checkpoint's file names."""
    model = _FakeModel()
    _fake_load(monkeypatch, {"missing_keys": set(), "mismatched_keys": set(),
                             "unexpected_keys": {"model.mtp.layers.0.weight"}}, model)

    eng = HFEngine(model_spec=_model(), decode=Greedy())
    eng._load()
    assert eng._model is model


class _PlacedModel(_FakeModel):
    """A built model that records the device the load moves it to."""

    def __init__(self) -> None:
        self.device: str | None = None

    def to(self, device):
        self.device = device
        return self


@pytest.mark.parametrize("index, device", [(None, "cuda"), (0, "cuda:0"), (1, "cuda:1")])
def test_load_moves_the_model_to_the_gpu_the_engine_was_placed_on(monkeypatch, index, device):
    # Every tensor the engine makes after its load goes to ``_device`` too, so an engine placed
    # on GPU 1 computes there while another computes on GPU 0.
    model = _PlacedModel()
    _fake_load(monkeypatch, {"missing_keys": set(), "mismatched_keys": set(), "unexpected_keys": set()},
               model)
    eng = HFEngine(model_spec=_model(), decode=Greedy())
    eng.device_index = index
    eng._load()
    assert (eng._device, model.device) == (device, device)


def test_the_stack_report_names_the_gpu_the_engine_was_placed_on(monkeypatch):
    names = {0: "NVIDIA H200", 1: "NVIDIA H200 (the second)"}
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(
        __version__="2.11.0", version=types.SimpleNamespace(cuda="13.0"),
        backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(version=lambda: 91900)),
        cuda=types.SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: names[index])))
    eng = HFEngine(model_spec=_model(), decode=Greedy())
    assert asyncio.run(eng.stack_report())["device_name"] == "NVIDIA H200"
    eng.device_index = 1
    assert asyncio.run(eng.stack_report())["device_name"] == "NVIDIA H200 (the second)"




def test_messages_of_passes_through_the_chat_fields_a_template_reads():
    """The oracle renders the artifact's own conversation, so a tool-calling turn
    keeps the fields the chat template reads: the structured call, the result's tool
    role and id, and the reasoning."""
    from hls.reference.engines import _messages_of
    from hls.core.payloads import Envelope, TextPayload
    messages = [
        {"role": "user", "content": "who founded it"},
        {"role": "assistant", "content": "", "reasoning_content": "I need evidence.",
         "reasoning": "I need evidence.",
         "tool_calls": [{"id": "call_0", "type": "function",
                         "function": {"name": "web_search",
                                      "arguments": {"query": "founder"}}}]},
        {"role": "tool", "tool_call_id": "call_0", "name": "web_search",
         "content": '{"documents": []}'},
    ]
    env = Envelope(request_id="r3", contents=[TextPayload(text="who founded it")],
                   metadata={"messages": messages})
    assert _messages_of(env) == messages
    plain = Envelope(request_id="r4", contents=[TextPayload(text="hi")],
                     metadata={"messages": [{"role": "user", "content": "hi"}]})
    assert _messages_of(plain) == [{"role": "user", "content": "hi"}]


def _sliding_config(**over):
    """A synthetic decoder config in the shape a sliding-window checkpoint declares:
    60 layers, every sixth reaching the whole sequence and the rest bounded by a
    1,024-token window, sharing no keys or values. Attribute access is all the cache
    builder reads, so a namespace stands in for a real config. ``get_text_config``
    answers with the config itself, which is what a text-only config does."""
    fields = {"layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 10,
              "sliding_window": 1024, "num_kv_shared_layers": 0,
              "num_hidden_layers": 60}
    config = types.SimpleNamespace(**{**fields, **over})
    config.get_text_config = lambda decoder=None, encoder=None: config
    return config


def test_kv_cache_windows_every_layer_the_config_declares_sliding():
    """The declared structure, layer for layer: a windowed cache layer wherever the
    config bounds attention and a full-length one everywhere else."""
    pytest.importorskip("torch")
    from transformers.cache_utils import (DynamicLayer,
                                          DynamicSlidingWindowLayer)
    from hls.reference.engines import _kv_cache

    config = _sliding_config()
    cache = _kv_cache(config)
    kinds = [type(layer) for layer in cache.layers]
    assert kinds.count(DynamicSlidingWindowLayer) == 50
    assert kinds.count(DynamicLayer) == 10
    assert cache.is_sliding == [t == "sliding_attention" for t in config.layer_types]
    assert {layer.sliding_window for layer in cache.layers
            if isinstance(layer, DynamicSlidingWindowLayer)} == {1024}


def test_transformers_builds_no_layers_when_no_keys_and_values_are_shared():
    """Why the helper above exists. transformers trims the layer-type list by the
    number of layers sharing keys and values, and a checkpoint that shares none
    declares 0, where ``list[:-0]`` is the empty list. The constructor then builds no
    layers at all and replicates a full-length one per layer instead."""
    pytest.importorskip("torch")
    from transformers.cache_utils import DynamicCache, DynamicLayer

    stock = DynamicCache(config=_sliding_config())
    assert stock.layers == []
    assert stock.layer_class_to_replicate is DynamicLayer


def test_kv_cache_without_declared_layer_types_is_the_stock_dynamic_cache():
    """A config that declares no attention structure gets transformers' own cache,
    which lazily replicates a full-length layer per layer."""
    pytest.importorskip("torch")
    from transformers.cache_utils import DynamicCache, DynamicLayer
    from hls.reference.engines import _kv_cache

    cache = _kv_cache(types.SimpleNamespace(num_hidden_layers=4))
    assert isinstance(cache, DynamicCache)
    assert cache.layers == []
    assert cache.layer_class_to_replicate is DynamicLayer


def test_kv_cache_drops_the_trailing_layers_that_share_keys_and_values():
    """A checkpoint whose last layers reuse an earlier layer's keys and values needs no
    cache for them, so they are trimmed -- what transformers' own slice intends and
    achieves whenever the count is positive."""
    pytest.importorskip("torch")
    from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer
    from hls.reference.engines import _kv_cache

    cache = _kv_cache(_sliding_config(num_kv_shared_layers=6))
    kinds = [type(layer) for layer in cache.layers]
    assert len(kinds) == 54
    assert kinds.count(DynamicSlidingWindowLayer) == 45
    assert kinds.count(DynamicLayer) == 9


def test_kv_cache_refuses_a_bounded_layer_with_no_declared_window():
    """A config cannot both bound a layer's reach and decline to say by how much. The
    cache is unbuildable then, and caching that layer at full length instead is the
    failure this helper exists to prevent, so it raises."""
    pytest.importorskip("torch")
    from hls.reference.engines import _kv_cache

    with pytest.raises(ValueError, match="bounded"):
        _kv_cache(_sliding_config(sliding_window=None))


def test_final_logit_softcap_reads_the_text_sub_config():
    """The cap comes off the text half of the config, where a vision-language
    checkpoint carries it and where the model's own forward reads it; a checkpoint
    declaring none reports none."""
    from hls.reference.engines import _final_logit_softcap

    composite = types.SimpleNamespace(
        get_text_config=lambda **kw: types.SimpleNamespace(final_logit_softcapping=30.0))
    assert _final_logit_softcap(composite) == 30.0
    assert _final_logit_softcap(types.SimpleNamespace(
        get_text_config=lambda **kw: types.SimpleNamespace(
            final_logit_softcapping=None))) is None
    # A text-only config answers with itself, so the same read reaches the cap.
    text_only = _sliding_config(final_logit_softcapping=30.0)
    assert _final_logit_softcap(text_only) == 30.0


def test_softcapped_compresses_a_block_the_way_the_model_does():
    """One block of logits off a real output head, capped: exactly
    ``tanh(raw / cap) * cap``, and returned untouched when there is no cap."""
    torch = pytest.importorskip("torch")
    from hls.reference.engines import _softcapped

    torch.manual_seed(0)
    head = torch.nn.Linear(4, 11, bias=False)
    # Scaled so the raw logits run well past the cap, which is where tanh bites.
    with torch.no_grad():
        head.weight.mul_(40.0)
    raw = head(torch.randn(6, 4)).float()
    assert raw.abs().max() > 30.0

    assert torch.equal(_softcapped(raw, 30.0), torch.tanh(raw / 30.0) * 30.0)
    assert _softcapped(raw, 30.0).abs().max() < 30.0
    assert _softcapped(raw, None) is raw


class _StubBatch(dict):
    """A tokenizer batch that answers ``.to(device)`` with itself (CPU only)."""

    def to(self, device):
        return self


class _StubTokenizer:
    """The two tokenizer calls teacher forcing makes: render a conversation, then
    tokenize it. The rendered text is ignored and the context is always the same two
    ids, which is all the scoring path needs from a prompt."""

    eos_token_id = 7

    def apply_chat_template(self, messages, **kwargs):
        return "ctx"

    def __call__(self, text, return_tensors=None, add_special_tokens=None):
        import torch
        return _StubBatch(input_ids=torch.tensor([[1, 2]]))


def _stub_engine(frequency_penalty, softcap):
    """An HFEngine wired to a tiny CPU model: an embedding for hidden states, a
    linear output head scaled so its logits run past a cap, and a config declaring
    *softcap*. Enough to run the real scoring loop without weights or a GPU."""
    import torch
    from hls.reference.engines import HFEngine

    torch.manual_seed(0)
    embed = torch.nn.Embedding(8, 4)
    head = torch.nn.Linear(4, 8, bias=False)
    with torch.no_grad():
        head.weight.mul_(60.0)

    def body(ids, past_key_values=None, use_cache=False):
        return types.SimpleNamespace(last_hidden_state=embed(ids),
                                     past_key_values=past_key_values)

    config = types.SimpleNamespace(final_logit_softcapping=softcap, num_hidden_layers=2)
    config.get_text_config = lambda decoder=None, encoder=None: config
    engine = HFEngine(model_spec=_model(), decode=Greedy(frequency_penalty=frequency_penalty))
    engine._device = "cpu"
    engine._tokenizer = _StubTokenizer()
    engine._model = types.SimpleNamespace(
        model=body, lm_head=head, config=config,
        get_input_embeddings=lambda: embed,
        generation_config=types.SimpleNamespace(eos_token_id=7))
    return engine, embed, head


def test_teacher_forcing_penalizes_the_capped_logits_not_the_raw_ones():
    """The order of cap and penalty. A serving stack subtracts its frequency penalty from
    softcapped logits, so the oracle has to cap first; tanh is not linear, so the other
    order is a different distribution and can hand a faithful server a flip it never
    made."""
    torch = pytest.importorskip("torch")
    from hls.reference.engines import _softcapped

    continuation = [3, 5, 3, 3]
    penalty = 0.1
    engine, embed, head = _stub_engine(penalty, 30.0)
    prepared = [([{"role": "user", "content": "hi"}], None, continuation, False)]
    got = engine._teacher_force_prepared(prepared)[0]

    # The same arithmetic spelled out: hidden rows from the context's last position on,
    # the head, the cap, then the penalty times how often each id appeared earlier.
    ids = torch.tensor([[1, 2, *continuation]])
    hidden = embed(ids)[0][1:]
    raw = head(hidden).float()
    counts = torch.zeros(raw.shape[0], 8)
    for j in range(1, raw.shape[0]):
        counts[j] = counts[j - 1]
        counts[j, continuation[j - 1]] += 1
    capped_then_penalized = _softcapped(raw, 30.0) - penalty * counts
    other_order = _softcapped(raw - penalty * counts, 30.0)
    expected = torch.log_softmax(capped_then_penalized, dim=-1)
    wrong = torch.log_softmax(other_order, dim=-1)

    want = [(expected[j].max() - expected[j, t]).item()
            for j, t in enumerate(continuation)]
    assert got == pytest.approx(want, abs=1e-5)
    # The two orders really do score differently, so the assertion above has teeth.
    other = [(wrong[j].max() - wrong[j, t]).item() for j, t in enumerate(continuation)]
    assert other != pytest.approx(want, abs=1e-5)


def test_teacher_forcing_without_a_declared_cap_scores_the_raw_logits():
    """A checkpoint declaring no softcapping is scored on the output head's own logits."""
    torch = pytest.importorskip("torch")

    continuation = [3, 5, 3]
    engine, embed, head = _stub_engine(0.0, None)
    prepared = [([{"role": "user", "content": "hi"}], None, continuation, False)]
    got = engine._teacher_force_prepared(prepared)[0]

    hidden = embed(torch.tensor([[1, 2, *continuation]]))[0][1:]
    lp = torch.log_softmax(head(hidden).float(), dim=-1)
    want = [(lp[j].max() - lp[j, t]).item() for j, t in enumerate(continuation)]
    assert got == pytest.approx(want, abs=1e-5)
