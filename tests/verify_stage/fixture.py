"""Tasks for the verify stage's tests, on fake reference engines that load and answer in the
verify role's own process, with no model and no GPU.

``FakeTeacherEngine`` is the token gate's declared engine class, loading nothing on the
role's engine host: one flip of gap 0.7 in every two-token item, and the stack the fixture
calibrations are measured on. Every
fake engine refuses a call made before its ``_load`` ran, so a verdict shows the engines were
loaded when they answered.

``conftest.py`` writes the task modules the tests name into a checkout of their own, each
beside the calibration it needs, so the role finds them the way it finds a real task."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import os
from typing import ClassVar

from hls.core.capabilities import Generate, StackReport, TeacherForce
from hls.core.decode import Greedy
from hls.core.engine import Engine, ResourceRequest
from hls.core.faults import HarnessFault
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.core.verdict import Verdict
from hls.reference.engines import HFEngine, ReferenceEngine
from hls.reference.references import HFReference, Reference
from hls.prompt import JinjaPromptTemplate
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.datasets import HFDataset
from hls.spec.agent import AgentFramework, AgentSpec
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier, ReferenceRelativeVerifier, TokenFaithfulnessVerifier
from hls.verify.base import ArtifactRollout, EngineAccessor, Verifier
from hls.verify.metrics.asr import WERMetric
from hls.verify.calibration import CalibrationError

#: What the fake teacher reports from ``stack_report``: the stack ``make_calibration`` records.
STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"}, "device_name": "B200"}


def _check_loaded(engine: ReferenceEngine) -> None:
    if engine._model != "loaded":
        raise RuntimeError(f"{type(engine).__name__} was called before it was loaded")


class FakeTeacherEngine(HFEngine):
    """The token gate's engine, loading nothing."""

    def _load(self) -> None:
        self._model = "loaded"

    async def teacher_force(self, items):
        _check_loaded(self)
        return [[0.0, 0.7] for _ in items]

    async def stack_report(self) -> dict:
        return copy.deepcopy(STACK)


class ForeignSiliconEngine(FakeTeacherEngine):
    """The fake teacher on an H100: the calibrated stack on other silicon."""

    async def stack_report(self) -> dict:
        return {**copy.deepcopy(STACK), "device_name": "NVIDIA H100 80GB HBM3"}


class WideTeacherEngine(FakeTeacherEngine):
    """The fake teacher as one engine over two GPUs, as a tensor-parallel engine asks."""

    @property
    def resources(self) -> ResourceRequest:
        return dataclasses.replace(super().resources, gpu_count=2)


class FakeReference(HFReference):
    engine_cls: ClassVar[type[Engine]] = FakeTeacherEngine


class WideReference(HFReference):
    engine_cls: ClassVar[type[Engine]] = WideTeacherEngine


class ForeignSiliconReference(HFReference):
    engine_cls: ClassVar[type[Engine]] = ForeignSiliconEngine


class FakeGenerator(ReferenceEngine, Generate):
    """A ``Generate`` engine that answers "B" to every prompt."""

    def _load(self) -> None:
        self._model = "loaded"

    async def generate(self, prompts):
        _check_loaded(self)
        return [Envelope(request_id=p.request_id, contents=[TextPayload(text="B")]) for p in prompts]


class FakeForcer(ReferenceEngine, TeacherForce):
    """A ``TeacherForce`` engine that finds no gap."""

    def _load(self) -> None:
        self._model = "loaded"

    async def teacher_force(self, items):
        _check_loaded(self)
        return [[0.0] for _ in items]


class LoadRefusingGenerator(FakeGenerator):
    """The generator, with a load that raises one of the verify role's refusal types, as an
    engine's own check at load time could."""

    def _load(self) -> None:
        raise CalibrationError("the engine's load raised a refusal's type")


class StopFailingForcer(FakeForcer):
    """The forcer, with a stop that raises one of the verify role's refusal types."""

    def _unload(self) -> None:
        raise HarnessFault("the engine's stop raised a refusal's type")


class StopFailingGenerator(FakeGenerator):
    """The generator, with a stop that raises one of the verify role's refusal types."""

    def _unload(self) -> None:
        raise CalibrationError("the generator's stop raised a refusal's type")


class BrokenStackForcer(FakeForcer, StackReport):
    """The forcer, with a stack report that fails, which no verifier of its task asks for."""

    async def stack_report(self) -> dict:
        raise RuntimeError("no stack on this engine")


class _Built(Reference):
    """A reference that builds its ``engine_cls`` for the task's model and GPU kind."""

    def build_engine(self, task):
        return self.engine_cls(model_spec=task.model, decode=self.decode, gpu_kind=task.hardware.accelerator.kind)


class GeneratorReference(_Built):
    engine_cls: ClassVar[type[Engine]] = FakeGenerator


class ForcerReference(_Built):
    engine_cls: ClassVar[type[Engine]] = FakeForcer


class LoadRefusingGeneratorReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = LoadRefusingGenerator


class StopFailingGeneratorReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = StopFailingGenerator


class StopFailingForcerReference(ForcerReference):
    engine_cls: ClassVar[type[Engine]] = StopFailingForcer


class BrokenStackForcerReference(ForcerReference):
    engine_cls: ClassVar[type[Engine]] = BrokenStackForcer


class PlacedGenerator(FakeGenerator):
    """The generator, placing itself on a GPU of its own, as an HF engine does."""

    placeable: ClassVar[bool] = True


class PlacedForcer(FakeForcer):
    """The forcer, placing itself on a GPU of its own."""

    placeable: ClassVar[bool] = True


class PlacedGeneratorReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = PlacedGenerator


class PlacedForcerReference(ForcerReference):
    engine_cls: ClassVar[type[Engine]] = PlacedForcer


class PairVerifier(Verifier):
    """A verifier with two engines, both live while it runs. It passes when both answered
    every event."""

    requires: ClassVar[frozenset] = frozenset({Generate, TeacherForce})
    generator: GeneratorReference
    forcer: ForcerReference

    async def verify(self, engine: EngineAccessor) -> Verdict:
        events = engine[ArtifactRollout]
        generated = await engine[Generate]([ev.input for ev in events])
        forced = await engine[TeacherForce]([ev.input for ev in events])
        ok = len(generated) == len(forced) == len(events)
        return Verdict(passed=ok, per_event={ev.request_id: ok for ev in events},
                       diagnostics={"generated": len(generated), "forced": len(forced)})


#: A module no other code of the role imports, and the distribution that installs it.
LAZY_MODULE, LAZY_DISTRIBUTION = "markdown_it", "markdown-it-py"


class ImportingVerifier(MetricVerifier):
    """The letter match, importing ``LAZY_MODULE`` as it verifies, as a verifier that loads a
    library on first use does."""

    async def verify(self, engine: EngineAccessor) -> Verdict:
        importlib.import_module(LAZY_MODULE)
        return await super().verify(engine)


MODEL = ModelSpec(base_checkpoint="m", arch="TX")
CONTRACT = RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                   output_modalities=frozenset({"text"}), protocol=OpenAIChatProtocol())
SCORER = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))


def _workload() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(), slo=NoSLO())


def _task(task_id: str, **parts) -> TaskSpec:
    return TaskSpec(id=task_id, model=MODEL, hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
                    contract=CONTRACT, scorer=SCORER, **parts)


def _gate(reference: type[HFReference] = FakeReference, *, allow_underpowered: bool = True):
    return TokenFaithfulnessVerifier(reference=reference(decode=Greedy()), allow_underpowered_trace=allow_underpowered)


def gate_task(reference: type[HFReference] = FakeReference) -> TaskSpec:
    """Tenants a and b under one pooled token gate on *reference*, each with its own letter
    match."""
    gate = _gate(reference)
    return _task("verify-stage-gate", tenants=[
        Tenant(id=t, workload=_workload(), verifiers=[gate, MetricVerifier(metric=LetterMatchMetric())])
        for t in ("a", "b")])


def split_gate_task() -> TaskSpec:
    """Two tenants whose pooled token gates are configured differently."""
    return _task("verify-stage-split", tenants=[
        Tenant(id="a", workload=_workload(), verifiers=[_gate(allow_underpowered=True)]),
        Tenant(id="b", workload=_workload(), verifiers=[_gate(allow_underpowered=False)])])


def pair_task(generator: type[GeneratorReference] = GeneratorReference,
              forcer: type[ForcerReference] = ForcerReference) -> TaskSpec:
    """One verifier with two engines, built from *generator* and *forcer*."""
    return _task("verify-stage-pair", workload=_workload(), verifiers=[
        PairVerifier(generator=generator(decode=Greedy()), forcer=forcer(decode=Greedy()))])


class AskVerifier(Verifier):
    """A verifier that asks its reference to answer every event's prompt, and passes when each
    answer is "B"."""

    requires: ClassVar[frozenset] = frozenset({Generate})
    reference: GeneratorReference

    async def verify(self, engine: EngineAccessor) -> Verdict:
        events = engine[ArtifactRollout]
        answers = await engine[Generate]([ev.input for ev in events])
        ok = [a.first_of("text").text == "B" for a in answers]
        return Verdict(passed=all(ok), per_event={ev.request_id: o for ev, o in zip(events, ok, strict=True)},
                       diagnostics={"answered": len(answers)})


class IdleFramework(AgentFramework):
    """An agent framework no test drives: enough for a task to declare the agentic seam."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        raise NotImplementedError("the fixture never drives a trajectory")


def _lookup(query: str) -> dict:
    return {"text": query}


class WeightsGenerator(FakeGenerator):
    """The fake generator, whose load needs the task model's weights mounted where the verify
    stage mounts them, as a real reference engine's does: it raises without them."""

    def _load(self) -> None:
        weights = os.environ.get("HLS_WEIGHTS_DIR")
        if not weights or not os.path.isdir(weights):
            raise RuntimeError(f"no weights mounted at $HLS_WEIGHTS_DIR ({weights!r})")
        super()._load()


class WeightsGeneratorReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = WeightsGenerator


def cached_task(turns: float = 1.0, *, agent: bool = False,
                reference: type[GeneratorReference] = GeneratorReference) -> TaskSpec:
    """One verifier whose *reference* caches its corpus answers, on a workload of *turns* turns
    per session; with *agent*, a task that declares an agent (``TaskSpec.agent``), driven one
    trajectory per row."""
    seam = None
    if agent:
        runner = LocalCallableToolRunner(specs=[ToolSpec(name="lookup", description="d", json_schema={})],
                                         callables={"lookup": _lookup})
        seam = AgentSpec(framework=IdleFramework(), tool_runner=runner, max_turns=4)
    return _task("verify-stage-cached", workload=_workload().model_copy(update={"mean_turns_per_session": turns}),
                 verifiers=[AskVerifier(reference=reference(decode=Greedy(), cache=True))], agent=seam)


def metric_task() -> TaskSpec:
    """A letter match and no engine."""
    return _task("verify-stage-metric", workload=_workload(), verifiers=[MetricVerifier(metric=LetterMatchMetric())])


def importing_task() -> TaskSpec:
    """The letter match of ``ImportingVerifier`` and no engine."""
    return _task("verify-stage-importing", workload=_workload(),
                 verifiers=[ImportingVerifier(metric=LetterMatchMetric())])


def event(tenant: str, request_id: str, answer: str, truth: str, *, errored: bool = False) -> TraceEvent:
    """A two-token reply *answer* of *tenant* to a question whose answer is *truth*; with
    *errored*, the request failed instead."""
    inp = Envelope(request_id=request_id, contents=[TextPayload(text="Q?")],
                   metadata={"messages": [{"role": "user", "content": "Q?"}], "tenant_id": tenant, "max_tokens": 2})
    if errored:
        return TraceEvent(request_id=request_id, input=inp, response=None, error="transport", ground_truth=truth,
                          t_send=0.0, t_recv=None)
    out = Envelope(request_id=request_id, contents=[TextPayload(text=answer, tokens=[42, 43])])
    return TraceEvent(request_id=request_id, input=inp, response=ArtifactResponse(output=out), ground_truth=truth,
                      t_send=0.0, t_recv=1.0)


def gate_trace(*, errored: bool = False) -> Trace:
    """Two replies of each of tenants a and b, in completion order; b's are wrong. With
    *errored*, b's second request failed."""
    return Trace(events=[event("a", "r0", "B", "B"), event("b", "r0", "A", "C"), event("a", "r1", "C", "C"),
                         event("b", "r1", "D", "A", errored=errored)])


def single_trace(n: int = 2) -> Trace:
    """*n* right answers of the single tenant t0."""
    return Trace(events=[event("t0", f"r{i}", "B", "B") for i in range(n)])


# ---------------------------------------------------------------------------
# Tasks the oracle fill fills: a corpus of test rows behind a caching reference
# ---------------------------------------------------------------------------


class FlakyGenerator(FakeGenerator):
    """The fake generator, which answers the first call made to it in a process and raises on
    the next: a fill that dies after its first chunk."""

    calls: ClassVar[int] = 0

    async def generate(self, prompts):
        type(self).calls += 1
        if type(self).calls > 1:
            raise RuntimeError("the flaky generator died")
        return await super().generate(prompts)


class FlakyGeneratorReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = FlakyGenerator


def _corpus(*, multi_turn: bool = False) -> DataSource:
    """A test split of the provision tests' shape: rows ``{"q": ...}`` whose prompt is their q."""
    if multi_turn:
        dataset = HFDataset(path="corp", revision="r1", source_split="test", id_column="q", context_col="ctx",
                            questions_col="qs", answers_col="as", answers_text_key=None)
    else:
        dataset = HFDataset(path="corp", revision="r1", source_split="test", id_column="q",
                            prompt=JinjaPromptTemplate(template="{{q}}"), target=JinjaPromptTemplate(template="B"))
    return DataSource(dataset=dataset, split=SplitSpec(dev=Count(value=1), test=Count(value=3)))


def fill_task(reference: type[GeneratorReference] = GeneratorReference, *, cache: bool = True,
              multi_turn: bool = False) -> TaskSpec:
    """One tenant's test corpus, verified by asking *reference*, which caches its answers when
    *cache*."""
    return _task("verify-stage-fill", workload=_workload(), data_source=_corpus(multi_turn=multi_turn),
                 verifiers=[AskVerifier(reference=reference(decode=Greedy(), cache=cache))])


def dev_reference_task() -> TaskSpec:
    """``fill_task``'s corpus, whose data source declares a dev reference published as
    ``oracle_dev.json``, gated relative to its caching reference with ``WERMetric``: the fake
    generator answers "B", every row's target."""
    source = _corpus().model_copy(update={"measured_dev_reference": "oracle_dev.json"})
    gate = ReferenceRelativeVerifier(metric=WERMetric(),
                                     reference=GeneratorReference(decode=Greedy(), cache=True))
    return _task("verify-stage-devref", workload=_workload(), data_source=source, verifiers=[gate])


def mixed_fill_task() -> TaskSpec:
    """Tenant asr, a single-turn corpus verified against a caching reference, beside tenant chat,
    a corpus of two turns per session whose reference does not cache."""
    chat = _workload().model_copy(update={"mean_turns_per_session": 2.0})
    return _task("verify-stage-mixed", tenants=[
        Tenant(id="asr", data_source=_corpus(), workload=_workload(),
               verifiers=[AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=True))]),
        Tenant(id="chat", data_source=_corpus(multi_turn=True), workload=chat,
               verifiers=[AskVerifier(reference=GeneratorReference(decode=Greedy(), cache=False))])])
