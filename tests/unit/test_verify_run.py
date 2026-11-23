"""The verification core, called directly: ``hls.verify.run.verify_trace`` and
``write_verdict``.

The main scenario is the one shape that exercises every rule the verify stage must keep:
a pooled gate declared on several tenants (one verdict, aliased
after its first index), per-tenant verifiers on different references (evicted, and each
loaded on its first use, one at a time, on the engine host the caller hands in), a failing
gate, a tenant with no events (it shifts no index), emission order across tenants,
foreign-tenant matrix cells, per-event sidecars and a non-finite diagnostic. A single-tenant
task asking its reference to generate exercises the reference answers on the store. The
token gate's own tests run it on a fake teacher of its declared engine class."""

from __future__ import annotations

import asyncio
import gzip
import inspect
import json
import math
from typing import ClassVar

import pytest

from hls.core.capabilities import Generate, StackReport, TeacherForce
from hls.core.decode import Greedy
from hls.core.engine import Engine, ResourceRequest
from hls.core.faults import HarnessFault
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.core.verdict import Verdict
from hls.reference.engines import HFEngine
from hls.reference.references import HFReference, Reference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.store.posix import PosixStore
from hls.verify import EngineAccessor, LetterMatchMetric, MetricVerifier, TokenFaithfulnessVerifier
from hls.verify.answers import StoreAnswers
from hls.verify.base import ArtifactRollout, Verifier, WiringError
from hls.verify.flips import ReplyFlips
from hls.verify.inprocess import EngineHost, Release
from hls.verify.run import verify_trace, write_verdict
from tests.unit.calibration_fixtures import make_calibration

_LOG: list[tuple] = []


class _Engine(Engine, TeacherForce, Generate):
    """A reference engine that logs its load, its stop and every call."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        _LOG.append(("acquire", self.tag))

    def _unload(self) -> None:
        _LOG.append(("release", self.tag))

    async def teacher_force(self, items):
        _LOG.append(("teacher_force", self.tag))
        return [[0.0] for _ in items]

    async def generate(self, prompts):
        _LOG.append(("generate", self.tag, [p.request_id for p in prompts]))
        return [Envelope(request_id=p.request_id, contents=[TextPayload(text=f"{self.tag}:{p.request_id}")])
                for p in prompts]


class _Ref(Reference):
    engine_cls: ClassVar[type] = _Engine
    tag: str

    def build_engine(self, task):
        return _Engine(self.tag)


class _WrongRef(_Ref):
    """Builds something other than the engine class it declares."""

    def build_engine(self, task):
        return object()


class _LoadingHost(EngineHost):
    """The engine host, recording the tag of each engine it loads, in load order."""

    def __init__(self) -> None:
        super().__init__()
        #: The tags of the engines loaded, in load order.
        self.loaded: list[str] = []

    async def acquire(self, engine: Engine) -> tuple[Engine, Release]:
        loaded = await super().acquire(engine)
        self.loaded.append(engine.tag)
        return loaded


class _PerTenant(Verifier):
    """Grades its own tenant's slice event by event, keyed by request_id, teacher-forcing
    each event unless ``ask`` is off; fails the events named in ``fail``."""

    reference: _Ref
    fail: tuple[str, ...] = ()
    ask: bool = True

    async def verify(self, engine) -> Verdict:
        _LOG.append(("verify", self.reference.tag))
        events = engine[ArtifactRollout]
        if self.ask:
            for ev in events:
                await engine[TeacherForce]([ev])
        per_event = {ev.request_id: ev.request_id not in self.fail for ev in events}
        diagnostics = {"rejections": ["planted"]} if self.fail else {"n": len(events)}
        return Verdict(passed=all(per_event.values()), per_event=per_event, diagnostics=diagnostics,
                       event_rationale={k: f"r-{k}" for k in per_event},
                       event_diagnostics={k: {"ok": v} for k, v in per_event.items()})


class _Pooled(Verifier):
    """A pooled gate: every tenant's events as one population, keyed by event_key, with
    a non-finite number in its diagnostics."""

    pooled: ClassVar[bool] = True
    reference: _Ref

    async def verify(self, engine) -> Verdict:
        events = engine[ArtifactRollout]
        _LOG.append(("pooled", [ev.event_key for ev in events]))
        await engine[TeacherForce](list(events))
        return Verdict(passed=True, per_event={ev.event_key: True for ev in events},
                       diagnostics={"se": float("nan"), "n": len(events)},
                       event_diagnostics={ev.event_key: {"n_tokens": 2} for ev in events})


class _Gen(Verifier):
    """Asks its reference to answer every event's prompt, through the answer cache or, with
    ``bypass``, around it, and passes with the answers in its diagnostics."""

    reference: _Ref
    bypass: bool = False

    async def verify(self, engine) -> Verdict:
        events = engine[ArtifactRollout]
        generate = engine.uncached(Generate) if self.bypass else engine[Generate]
        answers = await generate([ev.input for ev in events])
        return Verdict(passed=True, diagnostics={"answers": {a.request_id: a.contents[0].text for a in answers}})


@pytest.fixture(autouse=True)
def _clear_log():
    _LOG.clear()


def _wl() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(), slo=NoSLO())


def _ev(tenant: str, req: str, *, errored: bool = False) -> TraceEvent:
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")], metadata={"tenant_id": tenant})
    if errored:
        return TraceEvent(request_id=req, input=inp, response=None, error="transport", t_send=0.0)
    out = Envelope(request_id=req, contents=[TextPayload(text="A")])
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0)


def _contract() -> RequestResponseContract:
    return RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                   output_modalities=frozenset({"text"}))


def _spec(tenants: list[Tenant]) -> TaskSpec:
    return TaskSpec(id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
                    hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
                    contract=_contract(), tenants=tenants,
                    scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def _task() -> TaskSpec:
    gate = _Pooled(reference=_Ref(tag="pooled", decode=Greedy()))
    ref_a, ref_b = _Ref(tag="a", decode=Greedy()), _Ref(tag="b", decode=Greedy())
    return _spec([
        Tenant(id="t0", workload=_wl(), verifiers=[gate, _PerTenant(reference=ref_a)]),
        Tenant(id="t1", workload=_wl(), verifiers=[gate, _PerTenant(reference=ref_b, fail=("r0",))]),
        Tenant(id="t2", workload=_wl(), verifiers=[gate, _PerTenant(reference=ref_a)]),
        Tenant(id="t3", workload=_wl(), verifiers=[gate]),
    ])


# Emission order interleaves tenants; t2 has no events.
_ORDER = [("t1", "r0"), ("t0", "r0"), ("t3", "r0"), ("t0", "r1"), ("t1", "r1")]


def _trace() -> Trace:
    return Trace(events=[_ev(t, r) for t, r in _ORDER])


def _gen_task(*, tag: str = "g", cache: bool = True, bypass: bool = False) -> TaskSpec:
    ref = _Ref(tag=tag, decode=Greedy(), cache=cache)
    return _spec([Tenant(id="t0", workload=_wl(), verifiers=[_Gen(reference=ref, bypass=bypass)])])


def _gen_trace() -> Trace:
    return Trace(events=[_ev("t0", "r0"), _ev("t0", "r1")])


def _verify(tmp_path, trace: Trace, task: TaskSpec | None = None, *, answers: StoreAnswers | None = None):
    """``verify_trace`` on a recording host; returns the verdict, the matrix and the tags of
    the engines the host loaded, in load order."""
    host = _LoadingHost()

    async def go():
        async with host:
            verdict, results = await verify_trace(task or _task(), trace, eval_dir=tmp_path, host=host,
                                                  answers=answers)
        assert host.outstanding == 0
        return verdict, results

    verdict, results = asyncio.run(go())
    return verdict, results, host.loaded


def _records(path):
    with gzip.open(path, "rt") as fh:
        return [json.loads(line) for line in fh]


def test_indices_aliases_and_failed_gates_follow_the_tenants_that_ran(tmp_path):
    verdict, _results, _loaded = _verify(tmp_path, _trace())
    d = verdict.diagnostics
    assert [k for k in d if k != "failed_gates"] == [
        "0:_Pooled", "1:_PerTenant", "2:_Pooled", "3:_PerTenant", "4:_Pooled"]
    assert d["0:_Pooled"]["n"] == 5 and math.isnan(d["0:_Pooled"]["se"])
    alias = {"pooled_alias_of": 0, "events_file": "replies.0._Pooled.jsonl.gz"}
    assert d["2:_Pooled"] == alias and d["4:_Pooled"] == alias
    assert d["1:_PerTenant"] == {"n": 2}
    assert d["failed_gates"] == [{"verifier": "_PerTenant", "tenant": "t1", "key": "3:_PerTenant",
                                  "rejections": ["planted"]}]
    assert (verdict.passed, verdict.failure_mode) == (False, "correctness")
    # The run-level map keys on event_key, in tenant order then each tenant's slice order.
    assert verdict.per_event == {"t0:r0": True, "t0:r1": True, "t1:r0": False, "t1:r1": True, "t3:r0": True}


def test_the_matrix_is_a_row_per_verifier_that_ran_and_a_column_per_event_in_trace_order(tmp_path):
    _verdict, results, _loaded = _verify(tmp_path, _trace())
    assert len(results) == 5 and all(len(row) == len(_ORDER) for row in results)
    own = [[cell["passed"] is not None for cell in row] for row in results]
    #        t1:r0  t0:r0  t3:r0  t0:r1  t1:r1
    assert own == [
        [False, True, False, True, False],    # t0 pooled
        [False, True, False, True, False],    # t0 per-tenant
        [True, False, False, False, True],    # t1 pooled
        [True, False, False, False, True],    # t1 per-tenant
        [False, False, True, False, False],   # t3 pooled
    ]
    assert results[3][0] == {"passed": False, "score": None, "scores": {}, "rationale": "r-r0",
                             "diagnostics": {"ok": False}}
    assert results[0][1]["diagnostics"] == {"n_tokens": 2} and results[0][1]["rationale"] is None
    assert results[0][0] == {"passed": None, "score": None, "scores": {}, "rationale": None, "diagnostics": {}}


def test_a_pooled_gate_runs_once_over_every_tenant_in_emission_order(tmp_path):
    _verify(tmp_path, _trace())
    pooled = [entry for entry in _LOG if entry[0] == "pooled"]
    assert pooled == [("pooled", [f"{t}:{r}" for t, r in _ORDER])]


def test_engines_load_one_set_at_a_time_on_the_host(tmp_path):
    _verdict, _results, loaded = _verify(tmp_path, _trace())
    assert loaded == ["pooled", "a", "b"]
    lifecycle = [e[:2] for e in _LOG if e[0] in ("acquire", "release")]
    assert lifecycle == [("acquire", "pooled"), ("release", "pooled"), ("acquire", "a"), ("release", "a"),
                         ("acquire", "b"), ("release", "b")]
    # A live engine answers every later call without a second load.
    assert [e[1] for e in _LOG if e[0] == "teacher_force"] == ["pooled", "a", "a", "b", "b"]


def test_verify_trace_loads_every_engine_through_the_host_it_is_given(tmp_path):
    task = _spec([Tenant(id="t0", workload=_wl(), verifiers=[
        _PerTenant(reference=_Ref(tag="first", decode=Greedy())),
        _PerTenant(reference=_Ref(tag="second", decode=Greedy()))])])
    _verdict, _results, loaded = _verify(tmp_path, _gen_trace(), task)
    assert loaded == ["first", "second"]
    assert [e[:2] for e in _LOG if e[0] in ("acquire", "release")] == [
        ("acquire", "first"), ("release", "first"), ("acquire", "second"), ("release", "second")]


def test_an_engine_loads_on_its_first_use_and_never_when_unasked(tmp_path):
    task = _spec([Tenant(id="t0", workload=_wl(), verifiers=[
        _PerTenant(reference=_Ref(tag="a", decode=Greedy()), ask=False),
        _PerTenant(reference=_Ref(tag="b", decode=Greedy()))])])
    _verdict, _results, loaded = _verify(tmp_path, _gen_trace(), task)
    assert loaded == ["b"]
    assert _LOG == [("verify", "a"), ("verify", "b"), ("acquire", "b"), ("teacher_force", "b"),
                    ("teacher_force", "b"), ("release", "b")]


def test_an_engine_of_another_class_than_declared_is_a_wiring_error(tmp_path):
    task = _spec([Tenant(id="t0", workload=_wl(), verifiers=[
        _PerTenant(reference=_WrongRef(tag="w", decode=Greedy()))])])
    with pytest.raises(WiringError, match="_WrongRef.build_engine returned object, not the declared engine_cls _Engine"):
        _verify(tmp_path, _gen_trace(), task)


def test_sidecars_are_written_under_the_first_index_in_each_verdicts_own_order(tmp_path):
    _verify(tmp_path, _trace())
    assert sorted(p.name for p in tmp_path.glob("replies.*")) == [
        "replies.0._Pooled.jsonl.gz", "replies.1._PerTenant.jsonl.gz", "replies.3._PerTenant.jsonl.gz"]
    assert [r["event_key"] for r in _records(tmp_path / "replies.0._Pooled.jsonl.gz")] == [
        f"{t}:{r}" for t, r in _ORDER]
    # A per-tenant verdict keys its own maps by request_id, and the record says so.
    assert _records(tmp_path / "replies.3._PerTenant.jsonl.gz") == [
        {"event_key": "r0", "ok": False}, {"event_key": "r1", "ok": True}]


def test_no_eval_dir_writes_no_sidecar(tmp_path):
    async def go():
        async with EngineHost() as host:
            return await verify_trace(_task(), _trace(), eval_dir=None, host=host, answers=None)

    asyncio.run(go())
    assert list(tmp_path.iterdir()) == []


def test_any_failed_request_skips_every_verifier_and_every_load(tmp_path):
    trace = Trace(events=[_ev("t0", "r0"), _ev("t1", "r0", errored=True)])
    verdict, results, loaded = _verify(tmp_path, trace)
    assert verdict.failure_mode == "failed_requests"
    assert verdict.per_event == {"t0:r0": False, "t1:r0": False}
    assert verdict.diagnostics == {"reason": "verification skipped: 1 of 2 requests failed",
                                   "failed_requests": {"count": 1, "total": 2, "errors": {"transport": 1}}}
    assert len(results) == 7 and all(cell["passed"] is False for row in results for cell in row)
    assert loaded == [] and _LOG == [] and list(tmp_path.iterdir()) == []


def test_an_empty_trace_is_a_harness_fault(tmp_path):
    with pytest.raises(HarnessFault, match="zero items"):
        _verify(tmp_path, Trace(events=[]))


def _answers(tmp_path, writer: str) -> StoreAnswers:
    """A verify's writer on the task's answers area under *tmp_path*."""
    area = tmp_path / "answers"
    area.mkdir(exist_ok=True)
    return StoreAnswers(PosixStore(area), gpu_kind="H200", image="a" * 64, writer=writer)


def test_a_caching_reference_serves_stored_answers_without_loading(tmp_path):
    cold, _results, cold_loaded = _verify(tmp_path, _gen_trace(), _gen_task(),
                                          answers=_answers(tmp_path, "g1+verify+1"))
    assert cold_loaded == ["g"]
    assert [e for e in _LOG if e[0] == "generate"] == [("generate", "g", ["r0", "r1"])]
    _LOG.clear()
    warm, _results, warm_loaded = _verify(tmp_path, _gen_trace(), _gen_task(),
                                          answers=_answers(tmp_path, "g2+verify+1"))
    assert warm_loaded == [] and _LOG == []
    assert warm.diagnostics == cold.diagnostics == {"0:_Gen": {"answers": {"r0": "g:r0", "r1": "g:r1"}}}


@pytest.mark.parametrize("task, warm", [
    pytest.param(lambda: _gen_task(bypass=True), True, id="uncached-ask"),
    pytest.param(lambda: _gen_task(cache=False), True, id="caching-off"),
    pytest.param(lambda: _gen_task(), False, id="no-cache-given"),
    pytest.param(lambda: _gen_task(tag="h"), True, id="another-reference"),
    pytest.param(lambda: _spec([Tenant(id="t0", workload=_wl(), verifiers=[
        _PerTenant(reference=_Ref(tag="g", decode=Greedy(), cache=True))])]), True, id="not-a-corpus-capability"),
])
def test_only_a_caching_reference_asked_a_corpus_question_reads_the_cache(tmp_path, task, warm):
    _verify(tmp_path, _gen_trace(), _gen_task(), answers=_answers(tmp_path, "g1+verify+1"))
    _LOG.clear()
    _verdict, _results, loaded = _verify(tmp_path, _gen_trace(), task(),
                                         answers=_answers(tmp_path, "g2+verify+1") if warm else None)
    assert len(loaded) == 1
    assert [e[0] for e in _LOG if e[0] in ("generate", "teacher_force")] in (
        ["generate"], ["teacher_force", "teacher_force"])


def test_write_verdict_writes_the_viewers_verdict_json(tmp_path):
    verdict = Verdict(passed=False, per_event={"t0:r0": False}, failure_mode="correctness",
                      diagnostics={"0:X": {"se": float("inf"), "n": 1}})
    results = [[{"passed": False, "score": None, "scores": {}, "rationale": None, "diagnostics": {}}]]
    write_verdict(tmp_path, verdict, results)
    assert (tmp_path / "verdict.json").read_text() == (
        '{\n  "aggregate_passed": false,\n  "failure_mode": "correctness",\n  "per_event": {\n'
        '    "t0:r0": false\n  },\n  "diagnostics": {\n    "0:X": {\n      "se": null,\n      "n": 1\n'
        '    }\n  },\n  "per_verifier_results": [\n    [\n      {\n        "passed": false,\n'
        '        "score": null,\n        "scores": {},\n        "rationale": null,\n'
        '        "diagnostics": {}\n      }\n    ]\n  ]\n}')


class _Fixed(Verifier):
    """A verifier with no engine that returns the verdict it was given: ``passed``, its
    ``per_event`` keyed by request_id (empty for a verdict on the whole trace) and its
    per-event diagnostics."""

    passed: bool
    per_event: dict[str, bool] = {}
    event_diagnostics: dict[str, dict] = {}

    async def verify(self, engine) -> Verdict:
        return Verdict(passed=self.passed, per_event=self.per_event, event_diagnostics=self.event_diagnostics)


def _one_tenant(*verifiers: Verifier) -> TaskSpec:
    return _spec([Tenant(id="t0", workload=_wl(), verifiers=list(verifiers))])


def test_one_failing_verifier_fails_its_tenant_and_the_run_whatever_its_events_say(tmp_path):
    task = _one_tenant(_Fixed(passed=True), _Fixed(passed=False, per_event={"r0": True, "r1": True}))
    verdict, _results, _loaded = _verify(tmp_path, _gen_trace(), task)
    assert (verdict.passed, verdict.failure_mode) == (False, "correctness")
    assert verdict.per_event == {"t0:r0": True, "t0:r1": True}


def test_an_event_a_verifier_left_out_of_its_per_event_map_counts_as_failed(tmp_path):
    task = _one_tenant(_Fixed(passed=True, per_event={"r0": True}))
    verdict, results, _loaded = _verify(tmp_path, _gen_trace(), task)
    assert verdict.per_event == {"t0:r0": True, "t0:r1": False}
    assert [cell["passed"] for cell in results[0]] == [True, False]


@pytest.mark.parametrize("passed", [True, False])
def test_a_verdict_on_the_whole_trace_gives_every_event_it_covers_that_verdict(tmp_path, passed):
    verdict, results, _loaded = _verify(tmp_path, _gen_trace(), _one_tenant(_Fixed(passed=passed)))
    assert verdict.per_event == {"t0:r0": passed, "t0:r1": passed}
    assert [cell["passed"] for cell in results[0]] == [passed, passed]


def test_a_non_finite_number_in_a_sidecar_is_written_as_null(tmp_path):
    diagnostics = {"r0": {"lp": float("nan")}, "r1": {"lp": float("-inf")}}
    _verify(tmp_path, _gen_trace(), _one_tenant(_Fixed(passed=True, event_diagnostics=diagnostics)))
    with gzip.open(tmp_path / "replies.0._Fixed.jsonl.gz", "rt") as fh:
        lines = fh.read().splitlines()
    records = [json.loads(line, parse_constant=lambda c: pytest.fail(f"a sidecar holds {c}")) for line in lines]
    assert records == [{"event_key": "r0", "lp": None}, {"event_key": "r1", "lp": None}]


@pytest.mark.parametrize("name", ["host", "answers"])
def test_every_caller_names_the_host_and_the_answers(name):
    # No default: a caller that left the answers out would ask every question of the engine,
    # and one that left the host out could not load an engine at all.
    parameter = inspect.signature(verify_trace).parameters[name]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY and parameter.default is inspect.Parameter.empty


# --- the token gate and the structural rules, on a single-tenant task -------------------

_STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"}, "device_name": "B200"}


def _single_task(verifiers: list[Verifier] | None = None) -> TaskSpec:
    """A task declaring one workload and its verifiers itself, so its one tenant is ``t0``."""
    return TaskSpec(
        id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=_contract(), workload=_wl(),
        verifiers=verifiers or [MetricVerifier(metric=LetterMatchMetric())],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


class _ZeroGapEngine(HFEngine):
    """The token gate's declared engine class, loading nothing, finding no gap."""

    def _load(self) -> None:
        pass

    async def teacher_force(self, items):
        return [[0.0] * 12 for _ in items]

    async def stack_report(self):
        return _STACK


class _FakeTeacherEngine(HFEngine):
    """The token gate's declared engine class, loading nothing, reporting one flip of gap 0.7
    in every two-token item."""

    def _load(self) -> None:
        pass

    async def teacher_force(self, items):
        return [[0.0, 0.7] for _ in items]

    async def stack_report(self):
        return _STACK


def _token_event(tenant_id: str, request_id: str) -> TraceEvent:
    """One two-token reply attributed to ``tenant_id``."""
    out = Envelope(request_id=request_id, contents=[TextPayload(text="B", tokens=[42, 43])])
    inp = Envelope(request_id=request_id, contents=[TextPayload(text="Q?")],
                   metadata={"messages": [{"role": "user", "content": "Q?"}],
                             "tenant_id": tenant_id, "max_tokens": 2})
    return TraceEvent(request_id=request_id, input=inp, response=ArtifactResponse(output=out),
                      t_send=0.0, t_recv=1.0)


def _verify_on_host(task: TaskSpec, trace: Trace, eval_dir=None):
    async def go():
        async with EngineHost() as host:
            return await verify_trace(task, trace, eval_dir=eval_dir, host=host, answers=None)
    return asyncio.run(go())


def test_token_faithfulness_verify_end_to_end_with_fake_engine(monkeypatch):
    # The verifier composes teacher_force and stack_report through an accessor over an engine
    # the host loaded, without a GPU.
    task = _single_task()
    monkeypatch.setattr(HFReference, "build_engine", lambda self, t: _ZeroGapEngine(model_spec=t.model,
                                                                                    decode=Greedy()))
    out = Envelope(request_id="r0", contents=[TextPayload(text="B", tokens=[42] * 12)])
    inp = Envelope(request_id="r0", contents=[TextPayload(text="Q?")],
                   metadata={"messages": [{"role": "user", "content": "Q?"}], "max_tokens": 1_000_000})
    ev = TraceEvent(request_id="r0", input=inp, response=ArtifactResponse(output=out),
                    ground_truth=None, t_send=0.0, t_recv=1.0)
    # allow_underpowered_trace: this fixture's 12-token trace is far under any real floor,
    # because the test exercises the verify() composition, not the floor.
    v = TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()), calibration=make_calibration(),
                                  allow_underpowered_trace=True)

    async def go():
        async with EngineHost() as host:
            proxy, _release = await host.acquire(v.engines()[0].build_engine(task))
            acc = EngineAccessor(artifact_events=[ev], cap_to_provider={
                TeacherForce: proxy.teacher_force, StackReport: proxy.stack_report})
            return await v.verify(acc)

    assert asyncio.run(go()).passed is True


def test_verify_fails_loud_on_engine_class_skew(monkeypatch):
    """The core checks the built engine is an instance of the declared ``engine_cls``, so the
    capability dispatch cannot read a different engine than the wiring validated. The verifier
    is given its calibration so it gets as far as reaching for the engine: the core loads on
    first use, so the skew surfaces when the capability is called, before anything loads."""

    class _WrongEng:                       # not an instance of HFReference.engine_cls
        pass

    monkeypatch.setattr(HFReference, "build_engine", lambda self, t: _WrongEng())
    task = _single_task([TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()),
                                                   calibration=make_calibration())])
    trace = Trace(events=[_token_event("t0", "r0")])
    with pytest.raises(WiringError, match="(?i)engine_cls|HFEngine"):
        _verify_on_host(task, trace)


def test_verify_writes_each_verifiers_per_event_records_beside_the_verdict(monkeypatch, tmp_path):
    """Every verifier with per-event diagnostics gets a sidecar beside verdict.json,
    replies.<index>.<Verifier>.jsonl.gz, one record per graded event, and for the token gate
    that record is the schema calibration stores per serving (hls.verify.flips.ReplyFlips),
    event key included."""
    monkeypatch.setattr(HFReference, "build_engine",
                        lambda self, t: _FakeTeacherEngine(model_spec=t.model, decode=Greedy()))
    task = _single_task([TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()),
                                                   calibration=make_calibration(),
                                                   allow_underpowered_trace=True)])
    verdict, _results = _verify_on_host(task, Trace(events=[_token_event("t0", "r0")]), eval_dir=tmp_path)
    assert "0:TokenFaithfulnessVerifier" in verdict.diagnostics

    records = _records(tmp_path / "replies.0.TokenFaithfulnessVerifier.jsonl.gz")
    assert records == [{"event_key": "t0:r0", "n_tokens": 2, "n_flips": 1, "max_gap": 0.7,
                        "severe_gaps": [0.7], "n_nan": 0, "n_inf": 0, "stop_gap": None,
                        "at_cap": True}]
    assert ReplyFlips.from_json(records[0]).event_key == "t0:r0"


def test_verify_records_a_pooled_gate_once_and_aliases_the_other_tenants(monkeypatch, tmp_path):
    """Tenants sharing one pooled verifier share one verdict. verdict.json carries its
    diagnostics and its per-event sidecar under the first (tenant, verifier) index; every
    later index points back to that entry instead of repeating it, so the record shows
    one gate covering all the tenants."""
    monkeypatch.setattr(HFReference, "build_engine",
                        lambda self, t: _FakeTeacherEngine(model_spec=t.model, decode=Greedy()))
    gate = TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()), calibration=make_calibration(),
                                     allow_underpowered_trace=True)
    task = _spec([Tenant(id=tid, workload=_wl(), verifiers=[gate]) for tid in ("t0", "t1", "t2")])
    trace = Trace(events=[_token_event("t0", "r0"), _token_event("t1", "r1"), _token_event("t2", "r2")])
    verdict, results = _verify_on_host(task, trace, eval_dir=tmp_path)

    name = "TokenFaithfulnessVerifier"
    # The aggregate also carries run-level keys (failed_gates); the per-verifier entries
    # are the "<idx>:<ClassName>" ones.
    verifier_keys = {k for k in verdict.diagnostics if ":" in k and k.split(":", 1)[0].isdigit()}
    assert verifier_keys == {f"0:{name}", f"1:{name}", f"2:{name}"}
    first = verdict.diagnostics[f"0:{name}"]
    assert first and "pooled_alias_of" not in first          # the real diagnostics
    alias = {"pooled_alias_of": 0, "events_file": f"replies.0.{name}.jsonl.gz"}
    assert verdict.diagnostics[f"1:{name}"] == alias
    assert verdict.diagnostics[f"2:{name}"] == alias
    # One sidecar, holding every tenant's events, and nothing under the alias indices.
    assert sorted(p.name for p in tmp_path.glob("replies.*")) == [f"replies.0.{name}.jsonl.gz"]
    keys = [r["event_key"] for r in _records(tmp_path / f"replies.0.{name}.jsonl.gz")]
    assert sorted(keys) == ["t0:r0", "t1:r1", "t2:r2"]
    # The matrix is unchanged by the aliasing: a row per (tenant, verifier) pair, each
    # graded (a bool) only at its own tenant's event and not-applicable elsewhere.
    assert len(results) == 3
    assert [[cell["passed"] is not None for cell in row] for row in results] == [
        [True, False, False], [False, True, False], [False, False, True]]


def _labelled(req, text, gt=None, errored=False):
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")], metadata={"tenant_id": "t0"})
    if errored:
        return TraceEvent(request_id=req, input=inp, response=None, error="transport",
                          ground_truth=gt, t_send=0.0, t_recv=None)
    out = Envelope(request_id=req, contents=[TextPayload(text=text)])
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


def test_empty_trace_is_invalid():
    with pytest.raises(HarnessFault, match="zero items"):
        _verify_on_host(_single_task(), Trace(events=[]))


def test_all_errored_fails_and_skips_verifier_run():
    # A label metric would raise if it ran; the all-errored short-circuit returns before any
    # verifier or engine work, so we get a clean failed verdict instead.
    verdict, results = _verify_on_host(_single_task(), Trace(events=[
        _labelled("r0", "", errored=True), _labelled("r1", "", errored=True)]))
    assert verdict.passed is False
    # Run-level per_event keys on event_key (single tenant "t0").
    assert verdict.per_event == {"t0:r0": False, "t0:r1": False}
    assert len(results) == 1 and len(results[0]) == 2          # one verifier, two events
    assert all(r["passed"] is False for r in results[0])


def test_partial_error_hard_fails_even_when_scored_events_pass():
    task = _single_task([MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)])
    verdict, _ = _verify_on_host(task, Trace(events=[_labelled("r0", "B", gt="B"),
                                                     _labelled("r1", "", errored=True)]))
    assert verdict.passed is False                 # r1 errored -> hard fail
    # The fast path skips grading entirely: the correct r0 is never verified.
    assert verdict.per_event == {"t0:r0": False, "t0:r1": False}
    assert verdict.failure_mode == "failed_requests"


def test_unlabeled_dataset_with_label_metric_is_invalid():
    with pytest.raises(HarnessFault, match="ground-truth labels"):
        _verify_on_host(_single_task(), Trace(events=[_labelled("r0", "B", gt=None),
                                                      _labelled("r1", "A", gt=None)]))


class _OracleEngine(Engine, Generate):
    """A Generate engine that answers one fixed letter to every prompt."""

    def __init__(self, letter: str = "B") -> None:
        self.tag = f"oracle-{letter}"
        self._letter = letter

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=0)

    async def generate(self, prompts):
        return [Envelope(request_id=p.request_id, contents=[TextPayload(text=self._letter)]) for p in prompts]


class _OracleRef(Reference):
    """A reference whose engine is ``_OracleEngine``, as ``HFReference`` builds ``HFEngine``."""

    decode: Greedy = Greedy()
    letter: str = "B"
    engine_cls: ClassVar[type] = _OracleEngine

    def build_engine(self, task):
        return _OracleEngine(letter=self.letter)


def test_a_dataset_verifier_passes_when_its_rate_clears_its_minimum():
    """A verifier with no reference passes when its value clears ``min_value``: the artifact
    answers B and the label is B, so 1.0 clears 0.5."""
    verdict, _ = _verify_on_host(_single_task([MetricVerifier(metric=LetterMatchMetric(), min_value=0.5)]),
                                 Trace(events=[_labelled("r0", "B", gt="B")]))
    assert verdict.passed is True


def test_a_reference_relative_verifier_passes_an_artifact_as_good_as_its_reference():
    """Artifact and reference both answer B and the label is B, so both score 1.0 and the
    artifact's is within tol 0 of the reference's. The verdict holds one per-event entry per
    event, the verifier's values under its index-prefixed key, and each event's letters; the
    reference's engine is loaded once, on the host the caller gives."""
    from hls.verify import ReferenceRelativeVerifier

    task = _single_task([ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_OracleRef(letter="B"),
                                                   tol=0.0)])
    events = [_labelled("r0", "B", gt="B")]
    host = _LoadingHost()

    async def go():
        async with host:
            return await verify_trace(task, Trace(events=events), eval_dir=None, host=host, answers=None)

    verdict, results = asyncio.run(go())
    assert verdict.passed is True
    assert verdict.per_event == {ev.event_key: True for ev in events}
    diag = verdict.diagnostics["0:ReferenceRelativeVerifier"]
    assert (diag["artifact_value"], diag["reference_value"], diag["tol"]) == (1.0, 1.0, 0.0)
    [[cell]] = results
    assert {"predicted", "expected"} <= set(cell["diagnostics"]) and cell["diagnostics"]["predicted"] == "B"
    assert host.loaded == ["oracle-B"]


def test_two_verifiers_of_one_class_keep_their_own_diagnostics():
    """A verifier's diagnostics are keyed by its index, so two of one class do not overwrite
    each other's."""
    task = _single_task([MetricVerifier(metric=LetterMatchMetric(), min_value=1.0),
                         MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)])
    verdict, _ = _verify_on_host(task, Trace(events=[_labelled("r0", "B", gt="B")]))
    assert {k for k in verdict.diagnostics if k.split(":", 1)[0].isdigit()} == {"0:MetricVerifier",
                                                                                "1:MetricVerifier"}


def test_a_verifiers_replies_file_is_the_same_bytes_for_the_same_records(tmp_path, monkeypatch):
    """Written twice, at different times, the per-event records of the same verdict are the same
    file, so two verifications of one trace agree byte for byte."""
    import time

    from hls.core.verdict import Verdict
    from hls.verify.run import _write_event_records

    vd = Verdict(passed=True, per_event={"r0": True}, event_diagnostics={"r0": {"gap": 0.5}})
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _write_event_records(tmp_path / "a", 0, "V", vd)
    monkeypatch.setattr(time, "time", lambda: 2_000_000_000.0)
    _write_event_records(tmp_path / "b", 0, "V", vd)
    name = "replies.0.V.jsonl.gz"
    assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
