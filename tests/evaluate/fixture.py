"""Tasks for the ``hls grade`` tests.

``evaluate_task()`` is what the whole path runs on one machine: one tenant over the provision
tests' 20-row corpus (the fake provision role serves every corpus as it), the chat contract
the deploy_drive tests' fake server speaks, a reference-relative letter match on the verify
tests' fake generator, which answers "B" as the fake server does and as every row expects,
and a goodput scorer. Its hardware, 2 GiB of storage among it (the artifact's disk cap), is
small enough for the in-process Slurm to grant. Its measured baseline starts the fake server
(``BASELINE_ENTRYPOINT``), and its baseline's serve command is the server's serving mode, so
``--measured baseline`` warms the baseline's engine cache and serves it.

``conftest.py`` writes each variant into a checkout of the tests' own, and declares the task's
images (``image_set``), its measured baseline's start among them, as a real task's module
declares them."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import ClassVar

from hls.artifact.artifact import (AgentProducedArtifact, AgentProducerSpec, Budget, EvaluationAllocation,
                                   HighestAvailable)
from hls.core.decode import Greedy
from hls.core.engine import Engine, ResourceRequest
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.drive.bounds import VerifyPeak, bounds_for
from hls.egress.policy import AccessList, NetworkPolicy
from hls.prompt import JinjaPromptTemplate
from hls.recipes.catalog import ImageSet
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.agent import ToolAssets
from hls.spec.baseline import Baseline
from hls.spec.datasets import HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric
from hls.verify.verifiers import ReferenceRelativeVerifier
from tests.unit.test_trajectory_driver import _agent
from tests.verify_stage.fixture import CONTRACT, FakeGenerator, ForcerReference, GeneratorReference, PairVerifier

#: The task's id, which the tests add to the image catalog.
TASK_ID = "evaluate_task"
#: The task model's commit on the fake hub.
SHA = "0123456789abcdef0123456789abcdef01234567"
#: The corpus's commit, which the fake provision role serves the corpus at.
DATA_SHA = "89abcdef0123456789abcdef0123456789abcdef"
#: The task model's files on the fake hub.
FILES = {"config.json": b'{"model_type": "tx"}', "model.safetensors": b"\x01" * 4096}
#: The command the measured baseline starts with: the fake server.
BASELINE_ENTRYPOINT = ["python", "/workspace/server.py"]
#: The variable the measured baseline starts with.
BASELINE_ENV = {"FAKE_SERVER_FLAVOR": "baseline"}
#: The serve command the task's baseline declares: the fake server serving.
BASELINE_SERVE = ["--mode", "serve"]
#: What one drive of the task may make the harness hold: its drives are kilobytes, none published,
#: so its trace gets the least cap; its verifiers grade kilobytes too, so a byte held per byte of
#: trace stands for what they keep.
DRIVE_BOUNDS = bounds_for(CONTRACT, trace_bytes=0, held_reply_bytes=0, trace_request_bytes=2**20,
                          verify_peak=VerifyPeak(base_bytes=0, bytes_per_byte=1.0))
#: The drive wall the agent variant declares.
AGENT_WALL_SEC = 900.0
#: The runtime policy and the port the network variant declares.
NETWORK = NetworkPolicy(domains=AccessList(mode="allow", entries=["example.org"]))
NET_PORT = 9100
#: The serve command an agent-produced artifact of the task declares.
ARTIFACT_SERVE = ["--workers", "2"]
#: The evaluation allocation the allocated variants declare: the artifact's CPUs, memory and
#: writable cap, and the driver's memory beside them.
ALLOCATION = EvaluationAllocation(cpus=3, job_memory_gib=10, writable_gib=5, driver_memory_gib=3)
#: The verify stage's time the variant with a declared verify time takes: five hours, neither
#: the two-hour default nor any other stage's time.
VERIFY_TIME_SEC = 5 * 3600
#: The verify role's memory the variant with a declared verify memory takes: 24 GiB, neither the
#: reference job's nor any other role's.
VERIFY_MEMORY_GIB = 24


class GpuAssets(ToolAssets):
    """A tool bundle that encodes on a GPU and loads 40 GiB."""

    name = "gpu-tool"
    memory_mb = 40 * 1024
    gpu = True

    def identity(self) -> str:
        return "gpu-tool-v1"

    def hub_revisions(self) -> list[tuple[str, str]]:
        return []

    def build(self, variant, *, dev_row_ids, test_row_ids, out_dir, hf_token=None):
        (out_dir / "manifest.json").write_text(variant)


class WideGenerator(FakeGenerator):
    """The fake generator as one engine over two GPUs that loads 200 GiB on the host."""

    @property
    def resources(self) -> ResourceRequest:
        return dataclasses.replace(super().resources, gpu_count=2, memory_mb=200 * 1024)


class WideGeneratorReference(GeneratorReference):
    engine_cls: ClassVar[type[Engine]] = WideGenerator


def evaluate_task(*, agent=None, verifiers=None, hardware: Hardware | None = None,
                  network: NetworkPolicy | None = None, port: int | None = None) -> TaskSpec:
    """The whole path's task, with *agent*, *verifiers*, *hardware*, the runtime *network*
    policy or the contract's *port* replacing its own."""
    # Every row's answer is "B", which the fake server and the fake generator both give.
    dataset = HFDataset(path="corp", revision=DATA_SHA, source_split="test", id_column="q",
                        prompt=JinjaPromptTemplate(template="{{q}}"), target=JinjaPromptTemplate(template="B"))
    source = DataSource(dataset=dataset, split=SplitSpec(dev=Count(value=5), test=Count(value=4)))
    reference = ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=GeneratorReference(decode=Greedy()))
    return TaskSpec(
        id=TASK_ID, model=ModelSpec(base_checkpoint="org/model", revision=SHA, arch="TX"),
        hardware=hardware or Hardware(accelerator=AcceleratorSpec(kind="B200", count=1), cpu_count=2, memory_gb=4,
                                      storage_gb=2),
        contract=CONTRACT if port is None else CONTRACT.model_copy(update={"port": port}), network=network,
        data_source=source, seed=1,
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(), slo=NoSLO()),
        verifiers=verifiers if verifiers is not None else [reference],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
        baseline=Baseline(image="hls-baseline-evaluate-task", serve_command=BASELINE_SERVE), agent=agent,
        drive_bounds=DRIVE_BOUNDS)


def image_set(roles: Sequence[str] = ("baseline",)) -> ImageSet:
    """The task's images: on sample_task's layer, a server image for each of *roles*, its
    measured baseline, when it declares one, starting with ``BASELINE_ENTRYPOINT`` and
    ``BASELINE_ENV``."""
    from tests.recipes.declared import image_set

    return image_set(TASK_ID, roles, entrypoint=BASELINE_ENTRYPOINT, runtime_env=BASELINE_ENV)


def unbounded_task() -> TaskSpec:
    """The task with no drive bounds, which cannot be evaluated."""
    return evaluate_task().model_copy(update={"drive_bounds": None})


def agent_task() -> TaskSpec:
    """The task with an agent seam whose wall is ``AGENT_WALL_SEC`` and whose tool encodes on
    a GPU."""
    return evaluate_task(agent=dataclasses.replace(_agent(), drive_timeout_sec=AGENT_WALL_SEC, assets=GpuAssets()))


def served_artifact() -> AgentProducedArtifact:
    """An agent-produced artifact whose server takes ``ARTIFACT_SERVE`` after its port."""
    return AgentProducedArtifact(producer=AgentProducerSpec(hard_budget=Budget(wallclock_sec=60),
                                                            reasoning_effort=HighestAvailable()),
                                 serve_command=ARTIFACT_SERVE)


def allocated_artifact(*, driver_cpus: int | None = None, verify_time_sec: int | None = None,
                       verify_memory_gib: int | None = None) -> AgentProducedArtifact:
    """An agent-produced artifact whose evaluations take ``ALLOCATION``, its driver on CPUs of
    its own when *driver_cpus* is given, its verify stage *verify_time_sec* when given, and its
    verify role *verify_memory_gib* when given."""
    allocation = ALLOCATION.model_copy(update={"driver_cpus": driver_cpus, "verify_time_sec": verify_time_sec,
                                               "verify_memory_gib": verify_memory_gib})
    return AgentProducedArtifact(producer=AgentProducerSpec(hard_budget=Budget(wallclock_sec=60),
                                                            reasoning_effort=HighestAvailable(),
                                                            evaluation_allocation=allocation))


def net_task() -> TaskSpec:
    """The task with the runtime policy ``NETWORK`` and its server on ``NET_PORT``."""
    return evaluate_task(network=NETWORK, port=NET_PORT)


def wide_task() -> TaskSpec:
    """The task verified against the wide generator."""
    return evaluate_task(verifiers=[ReferenceRelativeVerifier(metric=LetterMatchMetric(),
                                                              reference=WideGeneratorReference(decode=Greedy()))])


def cached_task() -> TaskSpec:
    """The task verified against the fake generator, which caches its answers."""
    return evaluate_task(verifiers=[ReferenceRelativeVerifier(
        metric=LetterMatchMetric(), reference=GeneratorReference(decode=Greedy(), cache=True))])


def pair_task() -> TaskSpec:
    """The task under a verifier that holds two engines at once."""
    return evaluate_task(verifiers=[PairVerifier(generator=GeneratorReference(decode=Greedy()),
                                                 forcer=ForcerReference(decode=Greedy()))])
