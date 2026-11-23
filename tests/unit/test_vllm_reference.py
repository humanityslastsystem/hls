"""Unit tests for the vLLM reference surface (config + wiring). CPU-only:
vllm/torch are imported only inside VLLMEngine._load, so nothing here loads them."""
from hls.core.poly import dump_spec


def test_vllm_reference_knob_defaults():
    # The tuning knobs live FLAT on VLLMReference (no separate params class).
    from hls.reference.references import VLLMReference
    from hls.core.decode import Greedy
    r = VLLMReference(decode=Greedy())
    assert r.gpu_memory_utilization == 0.9
    assert r.max_model_len is None
    assert r.dtype == "auto"
    assert r.enforce_eager is False
    assert r.batch_invariant is False


def test_vllm_reference_is_frozen():
    import pytest
    from hls.reference.references import VLLMReference
    from hls.core.decode import Greedy
    r = VLLMReference(decode=Greedy())
    with pytest.raises(Exception):
        r.dtype = "float16"  # frozen config rejects mutation


def test_vllm_reference_serializes_knobs_flat():
    from hls.reference.references import VLLMReference
    from hls.core.decode import Greedy
    rec = dump_spec(VLLMReference(decode=Greedy(), max_model_len=4096, dtype="bfloat16"))
    assert rec["max_model_len"] == 4096
    assert rec["dtype"] == "bfloat16"
    assert "params" not in rec


def _vllm_engine():
    from hls.reference.engines import VLLMEngine
    from hls.core.decode import Greedy
    from hls.core.model import ModelSpec
    return VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                      decode=Greedy())


def test_vllm_engine_generates_serves_and_scores_reports_but_never_teacher_forces():
    """Two serving faces and one scoring face: this engine can produce a rollout
    (Generate), take the task's clocked traffic on its async API (ServeTraffic), and score
    a served report under a sampled decode (ReferenceLogprobs), which makes it the report
    gate's pinned reference. It cannot teacher-force for the greedy token-level gate. It
    reports its stack, so every allocation a campaign gives it can be checked against the
    declared accelerator."""
    from hls.core.engine import supports
    from hls.core.capabilities import (Generate, ReferenceLogprobs, ServeTraffic, ServingVersion,
                                        StackReport, TeacherForce)
    from hls.reference.engines import VLLMEngine
    caps = supports(VLLMEngine)
    assert caps == frozenset({Generate, ServeTraffic, ServingVersion, StackReport, ReferenceLogprobs})
    assert TeacherForce not in caps


def test_vllm_engine_requests_one_gpu():
    from hls.core.engine import ResourceRequest
    eng = _vllm_engine()
    assert eng.resources == ResourceRequest(gpu_count=1, gpu_kind=eng.gpu_kind)


def test_vllm_generate_empty_is_noop_without_loading():
    import asyncio
    eng = _vllm_engine()
    assert asyncio.run(eng.generate([])) == []   # returns before _load (no vllm import)


def test_vllm_generate_rejects_sample_decode():
    import asyncio
    import pytest
    from hls.reference.engines import VLLMEngine
    from hls.core.decode import Sample
    from hls.core.model import ModelSpec
    from hls.core.payloads import Envelope, TextPayload
    eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                     decode=Sample(temperature=0.7))
    item = Envelope(request_id="r", contents=[TextPayload(text="hi")])
    with pytest.raises(NotImplementedError):
        asyncio.run(eng.generate([item]))   # raises before _load


def test_vllm_generate_passes_max_tokens_explicitly(monkeypatch):
    # vLLM's SamplingParams defaults max_tokens=16, so the engine must pass the cap
    # EXPLICITLY: None means unlimited (the _max_new_tokens no-cap contract), and
    # relying on the constructor default would silently truncate uncapped runs.
    # The oracle also decodes under the module's shared greedy pin -- the same one the
    # clocked servings use -- so an oracle continuation and the member decode it is
    # compared against are produced under identical sampling.
    import asyncio
    import sys
    import types
    from hls.core.payloads import Envelope, TextPayload
    from hls.reference.engines import _GREEDY_SAMPLING

    captured = {"max_tokens": "<omitted>"}       # sentinel distinguishes None from omitted

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(SamplingParams=FakeSamplingParams))

    class FakeLLM:
        def chat(self, *, messages, sampling_params, use_tqdm):
            return [types.SimpleNamespace(
                        outputs=[types.SimpleNamespace(text="ok", finish_reason="stop",
                                                       token_ids=(1, 2, 3))])
                    for _ in messages]

    eng = _vllm_engine()
    eng._llm = FakeLLM()   # pre-loaded: _load returns early, never imports real vllm
    uncapped = [Envelope(request_id="r1", contents=[TextPayload(text="hi")])]
    out = asyncio.run(eng.generate(uncapped))    # no metadata['max_tokens'] anywhere
    assert captured["max_tokens"] is None        # explicit None -> unlimited, not the 16 default
    assert captured == {"max_tokens": None, **_GREEDY_SAMPLING}   # every pinned field reached it
    assert out[0].first_of("text").text == "ok"
    assert out[0].first_of("text").tokens == [1, 2, 3]   # vLLM's token_ids threaded onto the reply

    capped = [Envelope(request_id="r2", contents=[TextPayload(text="hi")],
                       metadata={"max_tokens": 7})]
    asyncio.run(eng.generate(capped))
    assert captured["max_tokens"] == 7           # a stamped cap still flows through


def test_vllm_generate_passes_the_declared_frequency_penalty(monkeypatch):
    # A task whose decode declares a frequency penalty must have vLLM apply it, or the
    # served tokens would not be the ones the penalized reference scores them against.
    # Every other pinned field stays as it is.
    import asyncio
    import sys
    import types
    from hls.core.decode import Greedy
    from hls.core.model import ModelSpec
    from hls.core.payloads import Envelope, TextPayload
    from hls.reference.engines import VLLMEngine, _GREEDY_SAMPLING

    captured = {}

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(SamplingParams=FakeSamplingParams))

    class FakeLLM:
        def chat(self, *, messages, sampling_params, use_tqdm):
            return [types.SimpleNamespace(
                        outputs=[types.SimpleNamespace(text="ok", finish_reason="stop",
                                                       token_ids=(1, 2))])
                    for _ in messages]

    eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                     decode=Greedy(frequency_penalty=0.1))
    eng._llm = FakeLLM()
    asyncio.run(eng.generate([Envelope(request_id="r1",
                                       contents=[TextPayload(text="hi")])]))
    assert captured == {"max_tokens": None, **_GREEDY_SAMPLING,
                        "frequency_penalty": 0.1}


def test_vllm_generate_propagates_real_finish_reason(monkeypatch):
    # The reply carries vLLM's REAL finish_reason per result -- no hardcoded value
    # ("length" or "stop") survives a mixed batch.
    import asyncio
    import sys
    import types
    from hls.core.payloads import Envelope, TextPayload

    monkeypatch.setitem(
        sys.modules, "vllm",
        types.SimpleNamespace(SamplingParams=lambda **kw: None))

    class FakeLLM:
        def chat(self, *, messages, sampling_params, use_tqdm):
            reasons = ["stop", "length"]
            return [types.SimpleNamespace(
                        outputs=[types.SimpleNamespace(text=f"t{i}",
                                                       finish_reason=reasons[i],
                                                       token_ids=(i, i + 1))])
                    for i, _ in enumerate(messages)]

    eng = _vllm_engine()
    eng._llm = FakeLLM()
    prompts = [Envelope(request_id="r1", contents=[TextPayload(text="hi")]),
               Envelope(request_id="r2", contents=[TextPayload(text="bye")])]
    out = asyncio.run(eng.generate(prompts))
    assert [o.first_of("text").finish_reason for o in out] == ["stop", "length"]
    assert [o.first_of("text").tokens for o in out] == [[0, 1], [1, 2]]


def test_vllm_generate_forwards_tools_to_chat_template(monkeypatch):
    import asyncio
    import sys
    import types
    from hls.core.payloads import Envelope, TextPayload

    monkeypatch.setitem(
        sys.modules, "vllm",
        types.SimpleNamespace(SamplingParams=lambda **kw: None))
    captured = {}

    class FakeLLM:
        def chat(self, *, messages, sampling_params, use_tqdm, chat_template_kwargs=None):
            captured["chat_template_kwargs"] = chat_template_kwargs
            return [types.SimpleNamespace(
                outputs=[types.SimpleNamespace(text="ok", finish_reason="stop",
                                               token_ids=[1, 2])])
                for _ in messages]

    tools = [{"type": "function", "function": {"name": "web_search"}}]
    eng = _vllm_engine()
    eng._llm = FakeLLM()
    prompt = Envelope(request_id="r", contents=[TextPayload(text="hi")],
                      metadata={"tools": tools})

    asyncio.run(eng.generate([prompt]))

    assert captured["chat_template_kwargs"] == {"tools": tools}


def _vllm_engine_bi(batch_invariant):
    from hls.reference.engines import VLLMEngine
    from hls.core.decode import Greedy
    from hls.core.model import ModelSpec
    return VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                      decode=Greedy(), batch_invariant=batch_invariant)


def test_vllm_batch_invariant_sets_env(monkeypatch):
    import os
    # Sandbox os.environ: _apply_env sets the var DIRECTLY (not via monkeypatch.setenv),
    # so without this the mutation leaks into the rest of the pytest process. Replacing
    # os.environ with a copy makes monkeypatch revert the whole dict on teardown.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    _vllm_engine_bi(True)._apply_env()            # env side effect only; no vllm import
    assert os.environ["VLLM_BATCH_INVARIANT"] == "1"


def test_vllm_batch_invariant_off_leaves_env(monkeypatch):
    import os
    monkeypatch.setattr(os, "environ", os.environ.copy())  # sandbox: no real-env mutation
    os.environ.pop("VLLM_BATCH_INVARIANT", None)           # start clean in the sandbox
    _vllm_engine_bi(False)._apply_env()
    assert "VLLM_BATCH_INVARIANT" not in os.environ


def _task():
    from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.taskspec import TaskSpec
    from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, Uncapped, Workload)
    from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
    from hls.verify import LetterMatchMetric, MetricVerifier
    return TaskSpec(
        id="T",
        model=ModelSpec(base_checkpoint="Qwen/Qwen2.5-0.5B-Instruct", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="H200", count=1)),
        contract=RequestResponseContract(
            path="/v1/chat/completions",
            input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"})),
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()),
        verifiers=[MetricVerifier(metric=LetterMatchMetric())],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def test_vllm_reference_builds_vllm_engine_with_knobs():
    from hls.reference.references import VLLMReference
    from hls.reference.engines import VLLMEngine
    from hls.core.decode import Greedy
    ref = VLLMReference(decode=Greedy(), max_model_len=2048)
    task = _task()
    eng = ref.build_engine(task)
    assert isinstance(eng, VLLMEngine)
    assert eng.model_spec is task.model
    assert eng.decode is ref.decode
    assert eng.max_model_len == 2048


def test_vllm_reference_engine_cls_capabilities_from_config():
    from hls.reference.references import VLLMReference
    from hls.reference.engines import VLLMEngine
    from hls.core.engine import supports
    from hls.core.capabilities import Generate, TeacherForce
    from hls.core.decode import Greedy
    ref = VLLMReference(decode=Greedy())
    assert ref.engine_cls is VLLMEngine
    assert Generate in supports(ref.engine_cls)
    assert TeacherForce not in supports(ref.engine_cls)


def test_vllm_reference_serializes_kind_and_knobs():
    from hls.reference.references import VLLMReference
    from hls.core.decode import Greedy
    rec = dump_spec(VLLMReference(decode=Greedy(), dtype="bfloat16"))
    assert rec["kind"] == "VLLMReference"
    assert rec["dtype"] == "bfloat16"
    assert rec["decode"]["kind"] == "Greedy"


def _contract():
    from hls.spec.contracts import RequestResponseContract
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}))


def test_vllm_reference_wires_for_metric_verifier():
    from hls.verify import ReferenceRelativeVerifier, LetterMatchMetric, check_wiring
    from hls.reference.references import VLLMReference
    from hls.core.decode import Greedy
    v = ReferenceRelativeVerifier(metric=LetterMatchMetric(),
                                  reference=VLLMReference(decode=Greedy()), tol=0.0)
    check_wiring([v], _contract())   # provides Generate -> no raise


def test_vllm_reference_fails_wiring_for_teacher_force_verifier():
    import pytest
    from hls.verify import TokenFaithfulnessVerifier, check_wiring, WiringError
    from hls.reference.references import VLLMReference
    from hls.core.decode import Greedy
    v = TokenFaithfulnessVerifier(reference=VLLMReference(decode=Greedy()))
    with pytest.raises(WiringError):
        check_wiring([v], _contract())   # requires TeacherForce, vLLM doesn't provide it


def test_sample_task_default_oracle_is_hf():
    from tasks.sample_task.task import TASK
    from hls.reference.references import HFReference
    refs = TASK.verifiers[0].engines()
    assert isinstance(refs[0], HFReference)


def test_sample_task_serves_the_small_qwen_model():
    from tasks.sample_task.task import TASK
    assert "Qwen2.5-0.5B" in TASK.model.base_checkpoint


def test_vllm_build_engine_threads_task_accelerator_kind():
    """Same pragmatic default as HFReference: the oracle slot requests the task's
    accelerator kind (harness config, not task semantics)."""
    from hls.core.decode import Greedy
    from hls.reference.references import VLLMReference

    task = _task()
    eng = VLLMReference(decode=Greedy()).build_engine(task)
    assert eng.gpu_kind == "H200"
    assert eng.resources.gpu_kind == "H200"


def test_serve_mode_warmup_builds_the_async_engine_and_only_it(monkeypatch):
    """A clocked-serving member's warmup (_load) must construct the ASYNC engine,
    never the oracle's synchronous LLM: the two cannot coexist on one card, and a
    warmup that builds the wrong one leaves it holding the GPU while serve_traffic
    fails to construct the right one."""
    import sys
    import types
    from hls.core.decode import Greedy
    from hls.reference.engines import VLLMEngine
    from hls.core.model import ModelSpec

    built = {"async": 0, "sync": 0}

    class FakeAsyncLLM:
        @staticmethod
        def from_engine_args(args):
            built["async"] += 1
            return object()

    def fake_llm(**kw):
        built["sync"] += 1
        return object()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=fake_llm))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils",
                        types.SimpleNamespace(AsyncEngineArgs=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm",
                        types.SimpleNamespace(AsyncLLM=FakeAsyncLLM))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: object())))

    eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                     decode=Greedy(), clocked_serving=True)
    monkeypatch.setattr(VLLMEngine, "_weights_source", lambda self: "/w")

    eng._load()          # the RPC warmup hook
    eng._load()          # idempotent
    assert built == {"async": 1, "sync": 0}


def test_a_vllm_engine_reports_its_async_engine_errored():
    # The engine host asks it once it loaded; AsyncLLM.errored is vLLM's own report that its engine
    # core died or its output handler ended. The synchronous LLM has none.
    import types

    from hls.reference.engines import SGLangEngine, VLLMEngine

    assert VLLMEngine.runs_processes and SGLangEngine.runs_processes
    serving = _vllm_engine()
    assert serving._health() is None
    serving._async_llm = types.SimpleNamespace(errored=False)
    assert serving._health() is None
    serving._async_llm = types.SimpleNamespace(errored=True)
    assert serving._health() == ("vLLM's async engine reports it errored: its engine core died or its output "
                                 "handler ended")
    oracle = _vllm_engine()
    oracle._llm = object()
    assert oracle._health() is None


def test_unload_shuts_down_the_engine_core_of_whichever_engine_was_built():
    """vLLM's engine core runs in a process of its own, which dropping the engine does
    not end. The synchronous ``LLM`` has no shutdown of its own, so the engine shuts its
    engine core client down; ``AsyncLLM`` has one."""
    import types

    stopped: list[str] = []
    oracle = _vllm_engine()
    oracle._llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace(
        engine_core=types.SimpleNamespace(shutdown=lambda: stopped.append("oracle"))))
    oracle._unload()
    serving = _vllm_engine()
    serving._async_llm = types.SimpleNamespace(shutdown=lambda: stopped.append("serving"))
    serving._unload()
    _vllm_engine()._unload()      # nothing built: nothing to stop
    assert stopped == ["oracle", "serving"]


def test_serve_mode_generate_refuses_instead_of_building_a_second_engine():
    """The oracle path on a clocked-serving member must raise loudly: building the
    synchronous LLM beside the async engine is exactly the double-build that OOMs."""
    import pytest
    from hls.core.decode import Greedy
    from hls.reference.engines import VLLMEngine
    from hls.core.model import ModelSpec

    eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                     decode=Greedy(), clocked_serving=True)
    import asyncio
    with pytest.raises(RuntimeError, match="clocked serving"):
        asyncio.run(eng.generate([object()]))  # type: ignore[list-item]


# ---- the report a sampled clocked serving carries ---------------------------

class _RecordedParams:
    """Stands in for ``vllm.SamplingParams``, keeping whatever the engine passed."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _ReportTokenizer:
    """The smallest tokenizer the chat helpers need: renders a prompt and encodes it
    to one id per word."""

    def apply_chat_template(self, messages, **kw):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, *, add_special_tokens):
        import types
        assert add_special_tokens is False
        return types.SimpleNamespace(input_ids=[len(w) for w in text.split()])


def _serving_engine(decode, *, logprobs=None, token_ids=(7,)):
    """A clocked-serving VLLMEngine whose async engine is a double, plus the double."""
    import types
    from hls.core.model import ModelSpec
    from hls.reference.engines import VLLMEngine

    class FakeAsyncLLM:
        def __init__(self):
            #: the sampling parameters of each served request, in service order.
            self.params = []

        async def generate(self, *, request_id, prompt, sampling_params):
            self.params.append(sampling_params)
            out = types.SimpleNamespace(text="r", finish_reason="stop",
                                        token_ids=list(token_ids), logprobs=logprobs)
            yield types.SimpleNamespace(outputs=[out])

    eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                     decode=decode, clocked_serving=True)
    eng._tokenizer = _ReportTokenizer()
    eng._async_llm = FakeAsyncLLM()
    return eng, eng._async_llm


def _lp(logprob, token="t"):
    """One entry of vLLM's per-position ``{token_id: Logprob}`` map."""
    import types
    return types.SimpleNamespace(logprob=logprob, decoded_token=token)


def test_a_sampled_serving_requests_the_contracted_report_width(monkeypatch):
    """The gate grades the distribution the server reported at every position it
    served, so a pool member serving a sampled decode has to ask its stack for one --
    at the width the contract asks a contestant for."""
    import asyncio
    import sys
    import types

    from hls.core.decode import Greedy, Sample
    from hls.core.payloads import Envelope
    from hls.verify.report import TOP_LOGPROBS_REQUESTED

    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(SamplingParams=_RecordedParams))
    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hello"}]})

    steps = [{7: _lp(-0.5), 8: _lp(-1.0)}]
    eng, async_llm = _serving_engine(Sample(temperature=1.0), logprobs=steps)
    asyncio.run(eng._serve_one(env))
    assert async_llm.params[0].kwargs["logprobs"] == TOP_LOGPROBS_REQUESTED

    # A greedy member asks for nothing: its gate reads no report, and vLLM's own
    # default for "report nothing" is None.
    eng, async_llm = _serving_engine(Greedy())
    asyncio.run(eng._serve_one(env))
    assert async_llm.params[0].kwargs["logprobs"] is None


def test_a_sampled_serving_raises_the_engines_logprob_ceiling(monkeypatch):
    """vLLM caps what a request may ask for at ``max_logprobs`` (default 20), below the
    contracted width -- so a member that did not raise it at bring-up would have every
    request refused at drive time instead."""
    import sys
    import types

    from hls.core.decode import Greedy, Sample
    from hls.core.model import ModelSpec
    from hls.reference.engines import VLLMEngine
    from hls.verify.report import TOP_LOGPROBS_REQUESTED

    seen: list[dict] = []

    class FakeAsyncLLM:
        @staticmethod
        def from_engine_args(args):
            seen.append(args)
            return object()

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=object()))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils",
                        types.SimpleNamespace(AsyncEngineArgs=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, "vllm.v1", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm",
                        types.SimpleNamespace(AsyncLLM=FakeAsyncLLM))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **kw: object())))
    monkeypatch.setattr(VLLMEngine, "_weights_source", lambda self: "/w")

    for decode, expected in ((Sample(temperature=1.0), TOP_LOGPROBS_REQUESTED),
                             (Greedy(), None)):
        eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                         decode=decode, clocked_serving=True)
        eng._serve_build()
        assert seen[-1].get("max_logprobs") == expected


def test_a_sampled_serving_reports_the_served_token_and_its_top_alternatives(monkeypatch):
    """vLLM appends the token it sampled to that position's top-k, so an uncut row is
    one entry wider exactly when the draw fell outside the top-k. Ranking and cutting
    to the contracted width drops the appended entry again precisely in that case, so
    the report never claims a token was among the most probable because the server
    happened to draw it -- while the served token's own log-probability survives on the
    entry either way."""
    import asyncio
    import sys
    import types

    from hls.core.decode import Sample
    from hls.core.payloads import Envelope
    from hls.verify.report import TOP_LOGPROBS_REQUESTED

    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(SamplingParams=_RecordedParams))
    # Position 0: the draw (id 500) is the least probable of the 22 entries, i.e. it
    # was appended. Position 1: the draw (id 0) is the most probable.
    outside = {500: _lp(-20.0)} | {i: _lp(-1.0 - i / 100) for i in range(21)}
    inside = {i: _lp(-1.0 - i / 100) for i in range(21)}
    eng, _ = _serving_engine(Sample(temperature=1.0),
                             logprobs=[outside, inside], token_ids=(500, 0))
    _text, _finish, tokens, report, _prompt = asyncio.run(eng._serve_one(Envelope(
        request_id="r0",
        metadata={"messages": [{"role": "user", "content": "hello"}]})))

    assert tokens == [500, 0]
    assert [e.token_id for e in report] == [500, 0]
    assert [e.logprob for e in report] == [-20.0, -1.0]
    for entry in report:
        assert len(entry.top_logprobs) == TOP_LOGPROBS_REQUESTED
        # Most probable first, ties broken on the id.
        assert [a.token_id for a in entry.top_logprobs] == list(range(21))
    # The appended entry is gone from position 0's alternatives; the drawn token at
    # position 1 belongs there and stays.
    assert 500 not in [a.token_id for a in report[0].top_logprobs]
    assert 0 in [a.token_id for a in report[1].top_logprobs]


def test_a_report_missing_the_sampled_tokens_own_logprob_raises(monkeypatch):
    """The served token's reported log-probability is one of the two numbers the
    measurement reads, so a stack that omits it must fail loudly rather than have the
    reply scored as if those positions were never served."""
    import asyncio
    import sys
    import types

    import pytest

    from hls.core.decode import Sample
    from hls.core.payloads import Envelope

    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(SamplingParams=_RecordedParams))
    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hello"}]})

    eng, _ = _serving_engine(Sample(temperature=1.0), logprobs=[{8: _lp(-1.0)}])
    with pytest.raises(RuntimeError,
                       match="no log-probability for the token it sampled"):
        asyncio.run(eng._serve_one(env))

    eng, _ = _serving_engine(Sample(temperature=1.0), logprobs=[], token_ids=(7,))
    with pytest.raises(RuntimeError, match="must cover every token"):
        asyncio.run(eng._serve_one(env))


# ---- how the report capture is reached in the worker ------------------------

class _ReportLLM:
    """A stand-in for vLLM's LLM, recording how the report path calls it.

    ``collective_rpc`` mirrors vLLM's dispatch closely enough to catch the calling
    convention: a string names a method on the worker, and anything else is invoked
    as ``partial(target, worker)``, which is what makes a worker-less callable raise.
    """

    def __init__(self, rows=None, workers=1):
        #: What each worker's take returns.
        self.rows = rows
        #: How many workers the engine is sharded over.
        self.workers = workers
        #: (method, args) in call order.
        self.calls = []
        #: The prompt ids the scored request carried.
        self.prompt = None

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls.append((method, args))
        if not isinstance(method, str):
            raise TypeError(
                f"{getattr(method, '__name__', method)}() takes "
                f"{method.__code__.co_argcount} positional arguments but "
                f"{method.__code__.co_argcount + 1} were given")
        if method == "hls_take_report":
            return [self.rows] * self.workers
        return [None] * self.workers

    def generate(self, prompt, sampling_params=None, use_tqdm=False):
        self.prompt = prompt["prompt_token_ids"]
        return []


def _report_engine(monkeypatch, llm):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "vllm",
                        types.SimpleNamespace(SamplingParams=lambda **kw: None))
    eng = _vllm_engine()
    eng._llm = llm
    return eng


def _capture_rows(n):
    return {"p_on_q": [[-1.0]] * n, "p_tail_lp": [-2.0] * n, "p_eot_lp": [-30.0] * n}


def test_the_report_capture_is_reached_by_method_name(monkeypatch):
    """vLLM invokes an rpc target as ``partial(target, worker)`` and refuses to ship a
    function object to a worker process at all unless insecure serialization is on.
    Both are avoided by naming a method the worker carries."""
    llm = _ReportLLM(rows=_capture_rows(2))
    eng = _report_engine(monkeypatch, llm)
    eng._score_one([1, 2, 3], [4, 5], [[4], [5]], [0])
    assert [m for m, _ in llm.calls] == ["hls_arm_report", "hls_take_report"]
    assert all(isinstance(m, str) for m, _ in llm.calls)


def test_the_capture_is_armed_with_the_replys_reported_ids(monkeypatch):
    # No request id: LLM.generate mints its own and takes none, so the arming call
    # carries the reply itself and the worker binds the slot to whatever arrives.
    llm = _ReportLLM(rows=_capture_rows(2))
    eng = _report_engine(monkeypatch, llm)
    eng._score_one([1, 2, 3], [4, 5], [[4, 9], [5, 9]], [0, 7])
    assert llm.calls[0] == ("hls_arm_report", ([[4, 9], [5, 9]], [0, 7]))
    assert llm.prompt == [1, 2, 3, 4, 5]                  # context then continuation


def test_a_reply_scored_end_to_end_returns_its_rows(monkeypatch):
    llm = _ReportLLM(rows=_capture_rows(2))
    eng = _report_engine(monkeypatch, llm)
    assert eng._score_one([1], [4, 5], [[4], [5]], [0]) == _capture_rows(2)


def test_no_rows_at_all_is_not_blamed_on_the_prefix_cache(monkeypatch):
    """Zero rows means the wrap never ran, which prefix caching cannot cause: the
    reply's own positions are the last of the prompt and a cache hit stops before
    them. Naming the cache here sends the reader to the wrong place."""
    import pytest
    llm = _ReportLLM(rows=_capture_rows(0))
    eng = _report_engine(monkeypatch, llm)
    with pytest.raises(RuntimeError, match="no request reached the prompt-logprob"):
        eng._score_one([1], [4, 5], [[4], [5]], [0])


def test_some_rows_missing_points_at_the_prefix_cache(monkeypatch):
    # A short count is the symptom a cache read does produce: the positions vLLM did
    # not prefill produced no logits.
    import pytest
    llm = _ReportLLM(rows=_capture_rows(1))
    eng = _report_engine(monkeypatch, llm)
    with pytest.raises(RuntimeError, match="enable_prefix_caching=False"):
        eng._score_one([1], [4, 5], [[4], [5]], [0])


def test_a_capture_failure_is_raised_on_the_host(monkeypatch):
    # The hook cannot raise on the engine loop without killing the engine, so it
    # hands the reason back and the host raises it here.
    import pytest
    llm = _ReportLLM(rows={"error": "RuntimeError: two requests at once"})
    eng = _report_engine(monkeypatch, llm)
    with pytest.raises(RuntimeError, match="two requests at once"):
        eng._score_one([1], [4, 5], [[4], [5]], [0])


def test_a_worker_that_was_never_armed_is_reported(monkeypatch):
    import pytest
    llm = _ReportLLM(rows=None)
    eng = _report_engine(monkeypatch, llm)
    with pytest.raises(RuntimeError, match="never received the arming call"):
        eng._score_one([1], [4, 5], [[4], [5]], [0])


def test_a_failing_shard_is_reported_even_beside_a_succeeding_one(monkeypatch):
    # Every rank runs the wrap; one of them failing must not be masked by another's
    # rows, because the reference would then grade on a partial shard.
    import pytest

    class Mixed(_ReportLLM):
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            if method == "hls_take_report":
                return [_capture_rows(2), {"error": "RuntimeError: shard 1 failed"}]
            return [None, None]

    eng = _report_engine(monkeypatch, Mixed())
    with pytest.raises(RuntimeError, match="shard 1 failed"):
        eng._score_one([1], [4, 5], [[4], [5]], [0])


def test_the_engine_mixes_the_report_methods_into_its_workers(monkeypatch):
    """The methods are only reachable by name because vLLM injects this class into the
    worker at construction; without it the rpc raises 'not implemented'."""
    import sys
    import types

    from hls.reference.vllm_report import WORKER_EXTENSION_CLS

    captured = {}

    def FakeLLM(**kwargs):
        captured.update(kwargs)
        return object()

    from hls.core.decode import Sample
    from hls.core.model import ModelSpec
    from hls.reference.engines import VLLMEngine

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(LLM=FakeLLM))
    eng = VLLMEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"), decode=Sample(temperature=1.0))
    monkeypatch.setattr(eng, "_weights_source", lambda: "m")
    eng._load()
    assert captured["worker_extension_cls"] == WORKER_EXTENSION_CLS
    # The capture reads every position it is asked about, so nothing may be served
    # from a cache instead of recomputed.
    assert captured["enable_prefix_caching"] is False


def test_the_mixin_name_resolves_to_the_class_vllm_will_import():
    # vLLM resolves the extension by qualified name in the worker process; a rename
    # here would fail there, where the error is much harder to read.
    import importlib

    from hls.reference.vllm_report import WORKER_EXTENSION_CLS, ReportWorker
    module, _, name = WORKER_EXTENSION_CLS.rpartition(".")
    assert getattr(importlib.import_module(module), name) is ReportWorker
