"""Reference split: HFReference is the config (Poly); HFEngine is the runtime
(an Engine, Generate+TeacherForce). The engine -- built via
``HFReference.build_engine(task)`` -- is what carries the capabilities and the GPU it asks
for."""
import asyncio

from hls.core.capabilities import Generate, TeacherForce
from hls.core.decode import Greedy
from hls.core.engine import Engine, ResourceRequest, supports
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.reference.engines import HFEngine
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier


def _task(model: ModelSpec | None = None) -> TaskSpec:
    m = model or ModelSpec(base_checkpoint="m", arch="TX")
    return TaskSpec(
        id="T",
        model=m,
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


class _EchoEngine(Engine, Generate):
    """Fake reference engine: counts batch sizes so the ONE-batched-call contract
    can be asserted without loading a real model."""

    def __init__(self):
        self.batch_sizes = []

    @property
    def resources(self):
        return ResourceRequest(gpu_count=1)

    async def generate(self, prompts):
        self.batch_sizes.append(len(prompts))
        return [Envelope(request_id=p.request_id,
                         contents=[TextPayload(text="B")]) for p in prompts]


def test_reference_engine_is_an_engine_with_resources():
    """The reference RUNTIME (HFEngine) is an Engine asking for one GPU, of the kind it was
    built for, and no host memory of its own."""
    eng = HFEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"), decode=Greedy())
    assert isinstance(eng, Engine)
    assert eng.resources == ResourceRequest(gpu_count=1)
    placed = HFEngine(model_spec=ModelSpec(base_checkpoint="m", arch="TX"), decode=Greedy(), gpu_kind="H200")
    assert placed.resources == ResourceRequest(gpu_count=1, gpu_kind="H200")


def test_hfengine_provides_generate_and_teacher_force():
    """HFEngine gains its capabilities from its MRO (Generate + TeacherForce)."""
    caps = supports(HFEngine)
    assert Generate in caps
    assert TeacherForce in caps


def test_build_engine_produces_hfengine():
    from hls.reference.references import HFReference
    task = _task()
    cfg = HFReference(decode=Greedy())
    eng = cfg.build_engine(task)
    assert isinstance(eng, HFEngine)
    assert eng.model_spec is task.model


def test_generate_takes_and_returns_a_batch():
    eng = _EchoEngine()
    prompts = [Envelope(request_id=str(i), contents=[TextPayload(text="q")]) for i in range(3)]
    outs = asyncio.run(eng.generate(prompts))
    assert len(outs) == 3
    assert eng.batch_sizes == [3]      # ONE batched call, not 3
