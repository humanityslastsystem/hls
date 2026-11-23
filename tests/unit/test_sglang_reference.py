"""Unit tests for the SGLang reference surface (config + wiring). CPU-only:
sglang is imported only inside SGLangEngine._load, so nothing here loads it --
sglang is NOT installed in the test environment (unlike vllm's mocked-import tests,
the absence itself is exercised directly). The real Engine().generate() call shape
needs a GPU node where sglang is installed."""
from hls.core.poly import dump_spec


def test_sglang_reference_knob_defaults():
    # The tuning knobs live FLAT on SGLangReference (mirrors VLLMReference), but
    # SGLang gets only the two it genuinely needs -- no speculative knobs (YAGNI).
    from hls.reference.references import SGLangReference
    from hls.core.decode import Greedy
    r = SGLangReference(decode=Greedy())
    assert r.mem_fraction_static == 0.85
    assert r.max_model_len is None


def test_sglang_reference_is_frozen():
    import pytest
    from hls.reference.references import SGLangReference
    from hls.core.decode import Greedy
    r = SGLangReference(decode=Greedy())
    with pytest.raises(Exception):
        r.mem_fraction_static = 0.5   # frozen config rejects mutation


def test_sglang_reference_serializes_knobs_flat():
    from hls.reference.references import SGLangReference
    from hls.core.decode import Greedy
    rec = dump_spec(SGLangReference(decode=Greedy(), max_model_len=4096, mem_fraction_static=0.7))
    assert rec["max_model_len"] == 4096
    assert rec["mem_fraction_static"] == 0.7
    assert "params" not in rec


def test_sglang_reference_serializes_kind_and_knobs():
    from hls.reference.references import SGLangReference
    from hls.core.decode import Greedy
    rec = dump_spec(SGLangReference(decode=Greedy()))
    assert rec["kind"] == "SGLangReference"
    assert rec["decode"]["kind"] == "Greedy"


def _sglang_engine():
    from hls.reference.engines import SGLangEngine
    from hls.core.decode import Greedy
    from hls.core.model import ModelSpec
    return SGLangEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                        decode=Greedy())


def test_sglang_engine_generates_and_serves_but_never_teacher_forces():
    """Two serving faces, no scoring one: this engine can produce a rollout (Generate)
    and take the task's clocked traffic on its async API (ServeTraffic), but it cannot
    teacher-force -- which is why it can be a calibration pool MEMBER and never the
    pinned reference the members are measured against. It DOES report its stack, so
    every allocation it is given can be checked against the declared
    accelerator."""
    from hls.core.engine import supports
    from hls.core.capabilities import (Generate, ServeTraffic, ServingVersion, StackReport,
                                        TeacherForce)
    from hls.reference.engines import SGLangEngine
    caps = supports(SGLangEngine)
    assert caps == frozenset({Generate, ServeTraffic, ServingVersion, StackReport})
    assert TeacherForce not in caps


def test_sglang_engine_requests_one_gpu():
    from hls.core.engine import ResourceRequest
    eng = _sglang_engine()
    assert eng.resources == ResourceRequest(gpu_count=1, gpu_kind=eng.gpu_kind)


def test_sglang_generate_empty_is_noop_without_loading():
    import asyncio
    eng = _sglang_engine()
    assert asyncio.run(eng.generate([])) == []   # returns before _load (no sglang import)


def test_sglang_generate_rejects_sample_decode():
    import asyncio
    import pytest
    from hls.reference.engines import SGLangEngine
    from hls.core.decode import Sample
    from hls.core.model import ModelSpec
    from hls.core.payloads import Envelope, TextPayload
    eng = SGLangEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                       decode=Sample(temperature=0.7))
    item = Envelope(request_id="r", contents=[TextPayload(text="hi")])
    with pytest.raises(NotImplementedError):
        asyncio.run(eng.generate([item]))   # raises before _load


def test_sglang_load_raises_with_install_hint_when_package_absent():
    # sglang is genuinely absent from this environment, so this exercises the real
    # ImportError path: a missing optional dependency must fail loud with the exact
    # fix, not a bare ModuleNotFoundError.
    import pytest
    eng = _sglang_engine()
    with pytest.raises(RuntimeError, match=r"uv pip install sglang"):
        eng._load()


class _FakeContext:
    """A ZMQ context that records each ``destroy``: the linger it was given, and the thread
    that called it."""

    def __init__(self, log: list) -> None:
        self.log = log

    def destroy(self, linger=None) -> None:
        import threading
        self.log.append(("destroy", self, linger, threading.current_thread()))


class _FakeSocket:
    def __init__(self, context: _FakeContext) -> None:
        self.context = context


class _FakeTokenizerManager:
    """The tokenizer manager's two sockets on one context, and the loop and tasks a serving's
    first request starts (none before one)."""

    def __init__(self, log: list, loop=None, tasks=()) -> None:
        self.context = _FakeContext(log)
        self.recv_from_detokenizer = _FakeSocket(self.context)
        self.send_to_scheduler = _FakeSocket(self.context)
        self.event_loop = loop
        self.asyncio_tasks = set(tasks)


class _FakeSGLangEngine:
    """Registers its shutdown to run at interpreter exit on construction, as
    ``sglang.Engine`` does, and records each shutdown in ``log``. It holds SGLang's sockets in
    this process: its own RPC socket, which its shutdown closes and drops as SGLang's does, and
    the tokenizer manager's."""

    def __init__(self, fail: bool = False, loop=None, tasks=()) -> None:
        import atexit
        self.fail = fail
        self.shutdowns = 0
        self.log: list = []
        self.send_to_rpc = _FakeSocket(_FakeContext(self.log))
        self.rpc_context = self.send_to_rpc.context
        self.tokenizer_manager = _FakeTokenizerManager(self.log, loop, tasks)
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        self.shutdowns += 1
        self.log.append(("shutdown",))
        self.send_to_rpc = None
        if self.fail:
            raise RuntimeError("scheduler would not stop")


def _unhooked(monkeypatch) -> list:
    """Record what ``atexit.unregister`` is asked to drop, and drop it."""
    import atexit
    unhooked: list = []
    real = atexit.unregister

    def unregister(func):
        unhooked.append(func)
        real(func)

    monkeypatch.setattr(atexit, "unregister", unregister)
    return unhooked


def test_sglang_unload_shuts_the_engine_down_and_drops_its_exit_hook(monkeypatch):
    # sglang.Engine pins itself to interpreter exit, so dropping it never ends its
    # scheduler process, which keeps the GPU; the engine shuts it down instead, and
    # unhooks it so the dead engine is not kept until exit and shut down twice.
    import atexit
    unhooked = _unhooked(monkeypatch)
    eng = _sglang_engine()
    fake = eng._engine = _FakeSGLangEngine()
    try:
        eng._unload()
        # atexit matches a hook by ==, which a bound method meets when it is the same
        # object's same method.
        assert fake.shutdowns == 1 and unhooked == [fake.shutdown]
    finally:
        atexit.unregister(fake.shutdown)
    _sglang_engine()._unload()    # no engine built: nothing to stop


def test_a_failed_sglang_shutdown_keeps_its_exit_hook(monkeypatch):
    # The scheduler may still hold the GPU, so interpreter exit tries again; the sockets are
    # closed all the same.
    import atexit
    import pytest
    unhooked = _unhooked(monkeypatch)
    eng = _sglang_engine()
    fake = eng._engine = _FakeSGLangEngine(fail=True)
    try:
        with pytest.raises(RuntimeError, match="would not stop"):
            eng._unload()
        assert unhooked == []
        assert [entry[0] for entry in fake.log] == ["shutdown", "destroy", "destroy"]
    finally:
        atexit.unregister(fake.shutdown)


def test_sglang_unload_closes_its_sockets_with_linger_0_once_the_engine_is_shut_down():
    # SGLang's shutdown kills its scheduler and detokenizer and leaves the tokenizer manager's
    # sockets open with ZMQ's default linger, which waits for ever: whatever collected them then
    # terminated their context, which waited for ever on messages for the peers the shutdown
    # killed. Closed here with linger 0, nothing is left to wait on.
    import atexit
    import threading
    eng = _sglang_engine()
    fake = eng._engine = _FakeSGLangEngine()
    try:
        eng._unload()
    finally:
        atexit.unregister(fake.shutdown)
    here = threading.current_thread()
    assert fake.log == [("shutdown",), ("destroy", fake.rpc_context, 0, here),
                        ("destroy", fake.tokenizer_manager.context, 0, here)]


def test_sglang_unload_closes_the_tokenizer_managers_sockets_on_the_loop_that_serves_them():
    # A serving's requests registered the tokenizer manager's sockets with the loop they ran on,
    # where its tasks wait on them: the tasks are cancelled first, since SGLang ends the whole
    # process when a task finds its socket closed, and the sockets are closed on that loop's
    # thread, the one that uses them.
    import asyncio
    import atexit
    import threading

    eng = _sglang_engine()
    seen: dict = {}

    async def serve() -> None:
        handle_loop = asyncio.ensure_future(asyncio.Event().wait())
        await asyncio.sleep(0)
        fake = eng._engine = _FakeSGLangEngine(loop=asyncio.get_running_loop(), tasks=[handle_loop])
        seen["fake"], seen["task"] = fake, handle_loop
        try:
            # Off the loop, as the engine host runs every stop.
            await asyncio.to_thread(eng._unload)
        finally:
            atexit.unregister(fake.shutdown)
        await asyncio.sleep(0)

    asyncio.run(serve())
    fake = seen["fake"]
    main = threading.main_thread()
    [rpc, manager] = [entry for entry in fake.log if entry[0] == "destroy"]
    assert rpc[1] is fake.rpc_context and rpc[3] is not main
    assert manager[1:] == (fake.tokenizer_manager.context, 0, main)
    assert seen["task"].cancelled()


def test_sglang_unload_terminates_a_real_context_whose_peer_is_gone():
    # pyzmq's own sockets: a message queued for a peer that is gone keeps a context with the
    # default linger from terminating; the unload leaves nothing for a collector to wait on.
    import asyncio
    import threading

    import pytest
    zmq = pytest.importorskip("zmq")
    zmq_asyncio = pytest.importorskip("zmq.asyncio")

    class _Manager:
        def __init__(self) -> None:
            context = zmq_asyncio.Context(2)
            self.recv_from_detokenizer = context.socket(zmq.PULL)
            self.send_to_scheduler = context.socket(zmq.PUSH)
            # Nobody listens there: the message waits for a peer that never comes.
            self.send_to_scheduler.connect("tcp://127.0.0.1:9")
            self.send_to_scheduler.send(b"a request for a scheduler that is gone", zmq.NOBLOCK)
            self.event_loop = None
            self.asyncio_tasks = set()

    class _Engine:
        def __init__(self) -> None:
            self.send_to_rpc = zmq.Context(2).socket(zmq.DEALER)
            self.send_to_rpc.connect("tcp://127.0.0.1:9")
            self.send_to_rpc.send(b"an rpc for a scheduler that is gone", zmq.NOBLOCK)
            self.tokenizer_manager = _Manager()

        def shutdown(self) -> None:
            # SGLang's own shutdown closes its RPC socket with linger 0 and drops it.
            self.send_to_rpc.close(linger=0)
            self.send_to_rpc = None

    async def build() -> _Engine:
        # An asyncio socket sends from within a running loop.
        return _Engine()

    eng = _sglang_engine()
    fake = eng._engine = asyncio.run(build())
    contexts = [fake.send_to_rpc.context, fake.tokenizer_manager.send_to_scheduler.context]
    # Terminating a context waits for its sockets' queued messages unless they linger 0, so an
    # unload that returns with both contexts closed applied linger 0.
    worker = threading.Thread(target=eng._unload, daemon=True)
    worker.start()
    worker.join(10)
    assert not worker.is_alive(), "the unload waited on a message for a peer that is gone"
    assert [context.closed for context in contexts] == [True, True]


def test_sglang_generate_carries_tokens_and_finish_reason():
    # Exercises OUR parsing of the offline Engine API's documented output shape:
    # Engine.generate() has no plain token-ids field (sglang), so
    # return_logprob=True's output_token_logprobs -- a list of (logprob, token_id,
    # token_text) per emitted token, in order -- is the documented way to recover
    # the exact emitted ids. Pre-loading _engine/_tokenizer (mirrors VLLMEngine's
    # `eng._llm = FakeLLM()` pattern) means _load, hence the real sglang import,
    # never runs.
    import asyncio
    from hls.core.payloads import Envelope, TextPayload

    import types

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<prompt:{messages[0]['content']}>"

        def __call__(self, text, add_special_tokens):
            # The engine must tokenize the rendered text itself with NO special
            # tokens (the template already carries BOS; a server-side re-tokenize
            # would prepend a second one and shift the whole context).
            assert add_special_tokens is False
            return types.SimpleNamespace(input_ids=[len(text), 7])

    captured = {}

    class FakeSGLangHandle:
        def generate(self, input_ids, sampling_params, return_logprob):
            captured["input_ids"] = input_ids
            captured["sampling_params"] = sampling_params
            captured["return_logprob"] = return_logprob
            return [{"text": f"reply-{i}",
                     "meta_info": {"finish_reason": {"type": "stop"},
                                   "output_token_logprobs": [(-0.1, 10 + i, "a"),
                                                             (-0.2, 20 + i, "b")]}}
                    for i, _ in enumerate(input_ids)]

    eng = _sglang_engine()
    eng._engine = FakeSGLangHandle()   # pre-loaded: _load returns early, never imports sglang
    eng._tokenizer = FakeTokenizer()

    prompts = [Envelope(request_id="r1", contents=[TextPayload(text="hi")]),
               Envelope(request_id="r2", contents=[TextPayload(text="bye")])]
    out = asyncio.run(eng.generate(prompts))

    assert captured["return_logprob"] is True
    # Every sampling field is pinned explicitly, not left to the stack's own defaults:
    # a checkpoint's generation_config supplies unset fields, and its repetition
    # penalty alone would move percent-scale mass off the reference's argmax.
    assert captured["sampling_params"] == {
        "max_new_tokens": None, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}
    # Prompts arrive as OUR tokenizer's ids of the rendered templates, in order.
    assert captured["input_ids"] == [[len("<prompt:hi>"), 7], [len("<prompt:bye>"), 7]]
    assert [o.first_of("text").tokens for o in out] == [[10, 20], [11, 21]]
    assert all(o.first_of("text").finish_reason == "stop" for o in out)


def test_sglang_generate_passes_the_declared_frequency_penalty():
    # A task whose decode declares a frequency penalty must have sglang apply it, or
    # the served tokens would not be the ones the penalized reference scores them
    # against. Every other pinned field stays as it is.
    import asyncio
    import types

    from hls.core.decode import Greedy
    from hls.core.model import ModelSpec
    from hls.core.payloads import Envelope, TextPayload
    from hls.reference.engines import SGLangEngine

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return "<prompt>"

        def __call__(self, text, add_special_tokens):
            return types.SimpleNamespace(input_ids=[1, 2])

    captured = {}

    class FakeSGLangHandle:
        def generate(self, input_ids, sampling_params, return_logprob):
            captured["sampling_params"] = sampling_params
            return [{"text": "reply",
                     "meta_info": {"finish_reason": {"type": "stop"},
                                   "output_token_logprobs": [(-0.1, 10, "a")]}}
                    for _ in input_ids]

    eng = SGLangEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                       decode=Greedy(frequency_penalty=0.1))
    eng._engine = FakeSGLangHandle()
    eng._tokenizer = FakeTokenizer()
    asyncio.run(eng.generate([Envelope(request_id="r1",
                                       contents=[TextPayload(text="hi")])]))

    assert captured["sampling_params"] == {
        "max_new_tokens": None, "temperature": 0.0, "top_p": 1.0, "top_k": -1,
        "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.1}


def test_sglang_generate_raises_loud_when_finish_reason_missing():
    # A reply that drops the documented "finish_reason" key must raise, not
    # silently degrade to finish_reason=None -- direct indexing (matching
    # output_token_logprobs) means a missing/malformed key propagates a real
    # KeyError instead of being swallowed.
    import asyncio
    import types

    import pytest
    from hls.core.payloads import Envelope, TextPayload

    class FakeTokenizer:
        @staticmethod
        def apply_chat_template(messages, tokenize, add_generation_prompt):
            return f"<prompt:{messages[0]['content']}>"

        def __call__(self, text, add_special_tokens):
            return types.SimpleNamespace(input_ids=[len(text)])

    class FakeSGLangHandleMissingFinishReason:
        def generate(self, input_ids, sampling_params, return_logprob):
            return [{"text": "reply",
                     "meta_info": {"output_token_logprobs": [(-0.1, 10, "a")]}}
                    for _ in input_ids]   # no "finish_reason" key at all

    eng = _sglang_engine()
    eng._engine = FakeSGLangHandleMissingFinishReason()
    eng._tokenizer = FakeTokenizer()

    prompts = [Envelope(request_id="r1", contents=[TextPayload(text="hi")])]
    with pytest.raises(KeyError):
        asyncio.run(eng.generate(prompts))


def _task():
    from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.taskspec import TaskSpec
    from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
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


def test_sglang_reference_builds_sglang_engine_with_knobs():
    from hls.reference.references import SGLangReference
    from hls.reference.engines import SGLangEngine
    from hls.core.decode import Greedy
    ref = SGLangReference(decode=Greedy(), max_model_len=2048)
    task = _task()
    eng = ref.build_engine(task)
    assert isinstance(eng, SGLangEngine)
    assert eng.model_spec is task.model
    assert eng.decode is ref.decode
    assert eng.max_model_len == 2048


def test_sglang_build_engine_threads_task_accelerator_kind():
    """Same pragmatic default as HFReference/VLLMReference: the oracle slot
    requests the task's accelerator kind (harness config, not task semantics)."""
    from hls.core.decode import Greedy
    from hls.reference.references import SGLangReference
    task = _task()
    eng = SGLangReference(decode=Greedy()).build_engine(task)
    assert eng.gpu_kind == "H200"
    assert eng.resources.gpu_kind == "H200"


def test_sglang_reference_engine_cls_capabilities_from_config():
    from hls.reference.references import SGLangReference
    from hls.reference.engines import SGLangEngine
    from hls.core.engine import supports
    from hls.core.capabilities import Generate, TeacherForce
    from hls.core.decode import Greedy
    ref = SGLangReference(decode=Greedy())
    assert ref.engine_cls is SGLangEngine
    assert Generate in supports(ref.engine_cls)
    assert TeacherForce not in supports(ref.engine_cls)


def _contract():
    from hls.spec.contracts import RequestResponseContract
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}))


def test_sglang_reference_wires_for_metric_verifier():
    from hls.verify import ReferenceRelativeVerifier, LetterMatchMetric, check_wiring
    from hls.reference.references import SGLangReference
    from hls.core.decode import Greedy
    v = ReferenceRelativeVerifier(metric=LetterMatchMetric(),
                                  reference=SGLangReference(decode=Greedy()), tol=0.0)
    check_wiring([v], _contract())   # provides Generate -> no raise


def test_sglang_reference_fails_wiring_for_teacher_force_verifier():
    # TokenFaithfulnessVerifier requires TeacherForce + StackReport; SGLangEngine is
    # a Generate-only pool member and provides neither, so wiring must name BOTH
    # missing capabilities.
    import pytest
    from hls.verify import TokenFaithfulnessVerifier, check_wiring, WiringError
    from hls.reference.references import SGLangReference
    from hls.core.decode import Greedy
    v = TokenFaithfulnessVerifier(reference=SGLangReference(decode=Greedy()))
    with pytest.raises(WiringError) as exc_info:
        check_wiring([v], _contract())
    msg = str(exc_info.value)
    assert "TeacherForce" in msg
    assert "StackReport" in msg


# ---- the report a sampled clocked serving carries ---------------------------

class _ReportTokenizer:
    """The smallest tokenizer the chat helpers need: renders a prompt and encodes it
    to one id per word."""

    def apply_chat_template(self, messages, **kw):
        return " ".join(m["content"] for m in messages)

    def __call__(self, text, *, add_special_tokens):
        import types
        assert add_special_tokens is False
        return types.SimpleNamespace(input_ids=[len(w) for w in text.split()])


#: The field names sglang's own ``SamplingParams`` accepts, of those this harness sends
#: or could send. The struct raises ``TypeError: Unexpected keyword argument`` on
#: anything else, so the double below holds the same line: a request parameter passed as
#: a sampling parameter has to fail in this suite rather than fault every request of a
#: real serving. Read off sglang 0.5.16's
#: ``sglang/srt/sampling/sampling_params.py``.
_SGLANG_SAMPLING_FIELDS = frozenset({
    "max_new_tokens", "min_new_tokens", "stop", "stop_token_ids",
    "skip_special_tokens", "spaces_between_special_tokens", "temperature", "top_p",
    "top_k", "min_p", "frequency_penalty", "presence_penalty", "repetition_penalty",
    "ignore_eos", "n", "json_schema", "regex", "ebnf", "structural_tag",
    "no_stop_trim", "custom_params", "stream_interval", "logit_bias",
})


def _serving_engine(decode, *, served, tops=None):
    """A clocked-serving SGLangEngine whose async engine is a double, plus the double.

    ``served`` is ``meta_info['output_token_logprobs']`` -- one ``(logprob, token_id,
    text)`` per served token -- and ``tops`` the matching
    ``meta_info['output_top_logprobs']``, one list of the same triples per position."""
    from hls.core.model import ModelSpec
    from hls.reference.engines import SGLangEngine

    class FakeEngine:
        """Stands in for ``sglang.Engine``, with ``async_generate``'s real signature:
        the report width is a parameter of the call, beside ``return_logprob``, and the
        sampling params are a dict the engine hands to its own ``SamplingParams``."""

        def __init__(self):
            #: the sampling parameters of each served request, in service order.
            self.params = []
            #: the report width asked for per request, 0 when none was.
            self.widths = []

        async def async_generate(self, *, input_ids, sampling_params, return_logprob,
                                 top_logprobs_num=0):
            unexpected = set(sampling_params) - _SGLANG_SAMPLING_FIELDS
            if unexpected:
                raise TypeError(
                    f"Unexpected keyword argument {sorted(unexpected)[0]!r}")
            self.params.append(sampling_params)
            self.widths.append(top_logprobs_num)
            meta = {"finish_reason": {"type": "stop"},
                    "output_token_logprobs": served}
            if tops is not None:
                meta["output_top_logprobs"] = tops
            return {"text": "r", "meta_info": meta}

    eng = SGLangEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"),
                       decode=decode)
    eng._tokenizer = _ReportTokenizer()
    eng._engine = FakeEngine()
    return eng, eng._engine


def test_a_sampled_serving_requests_the_contracted_report_width():
    """The gate grades the distribution the server reported at every position it
    served, so a pool member serving a sampled decode has to ask its stack for one --
    at the width the contract asks a contestant for, and as a parameter of the call
    rather than a sampling param, which this stack's SamplingParams refuses. A greedy
    member asks for no alternatives at all: it already passes return_logprob to recover
    the served ids, and its gate reads no distribution."""
    import asyncio

    from hls.core.decode import Greedy, Sample
    from hls.core.payloads import Envelope
    from hls.verify.report import TOP_LOGPROBS_REQUESTED

    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hello"}]})
    served = [(-0.5, 7, "a")]
    tops = [[(-0.5, 7, "a"), (-1.0, 8, "b")]]

    eng, engine = _serving_engine(Sample(temperature=1.0), served=served, tops=tops)
    asyncio.run(eng._serve_one(env))
    assert engine.widths == [TOP_LOGPROBS_REQUESTED]
    assert "top_logprobs_num" not in engine.params[0]

    eng, engine = _serving_engine(Greedy(), served=served)
    asyncio.run(eng._serve_one(env))
    assert engine.widths == [0]


def test_the_sampled_serving_pins_every_truncation_knob():
    """The distribution graded is the model's own softmax over the whole vocabulary, so
    every knob that would narrow it is sent explicitly rather than left to the
    checkpoint's generation_config."""
    import asyncio

    from hls.core.decode import Sample
    from hls.core.payloads import Envelope

    eng, engine = _serving_engine(Sample(temperature=1.0), served=[(-0.5, 7, "a")],
                                  tops=[[(-0.5, 7, "a")]])
    asyncio.run(eng._serve_one(Envelope(
        request_id="r0",
        metadata={"messages": [{"role": "user", "content": "hello"}]})))
    params = engine.params[0]
    assert params["temperature"] == 1.0
    assert (params["top_p"], params["top_k"], params["min_p"]) == (1.0, -1, 0.0)
    assert params["repetition_penalty"] == 1.0
    assert params["presence_penalty"] == params["frequency_penalty"] == 0.0


def test_a_sampled_serving_reports_its_true_top_k():
    """This stack sends its true top-k, which need not contain the token it drew, so
    the report can name a served token that is not one of its own alternatives -- which
    is exactly the case the partition's tail atom exists for."""
    import asyncio

    from hls.core.decode import Sample
    from hls.core.payloads import Envelope

    served = [(-20.0, 500, "z"), (-1.0, 0, "a")]
    tops = [[(-1.0 - i / 100, i, "x") for i in range(3)],
            [(-1.0 - i / 100, i, "x") for i in range(3)]]
    eng, _ = _serving_engine(Sample(temperature=1.0), served=served, tops=tops)
    _text, finish, tokens, report, _prompt = asyncio.run(eng._serve_one(Envelope(
        request_id="r0",
        metadata={"messages": [{"role": "user", "content": "hello"}]})))

    assert (finish, tokens) == ("stop", [500, 0])
    assert [e.token_id for e in report] == [500, 0]
    assert [e.logprob for e in report] == [-20.0, -1.0]
    assert [a.token_id for a in report[0].top_logprobs] == [0, 1, 2]
    # The drawn token at position 0 is not among its own alternatives.
    assert 500 not in [a.token_id for a in report[0].top_logprobs]


def test_a_report_that_misses_a_position_raises():
    """Direct indexing on the documented reply shape, as the served-ids path beside it
    does: a schema change has to raise here rather than silently degrade the report."""
    import asyncio

    import pytest

    from hls.core.decode import Sample
    from hls.core.payloads import Envelope

    env = Envelope(request_id="r0",
                   metadata={"messages": [{"role": "user", "content": "hello"}]})
    eng, _ = _serving_engine(Sample(temperature=1.0),
                             served=[(-0.5, 7, "a"), (-0.5, 8, "b")],
                             tops=[[(-0.5, 7, "a")]])
    with pytest.raises(RuntimeError, match="must cover every token"):
        asyncio.run(eng._serve_one(env))

    eng, _ = _serving_engine(Sample(temperature=1.0), served=[(-0.5, 7, "a")])
    with pytest.raises(KeyError, match="output_top_logprobs"):
        asyncio.run(eng._serve_one(env))
