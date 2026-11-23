"""What the local scheduler's tests share: a host given as facts, the request the scheduler reads,
batch scripts that do one thing, and a wait for a condition."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hls.spec.resources import ResourceShape
from hls.substrate.local.inventory import CoreFact, GpuFact, HostFacts, HostInventory
from hls.substrate.local.records import StateDir
from hls.substrate.local.scheduler import LocalScheduler


class FakeInventory(HostInventory):
    """A host of *gpus* H200s, *numa* NUMA nodes of *cores_per_node* cores of *threads* CPUs each,
    and *memory_gib* of memory. Its nvidia-smi is *nvidia_smi*: ``true``, which lists no process
    computing on any of them."""

    def __init__(self, *, gpus: int = 4, numa: int = 2, cores_per_node: int = 4, threads: int = 1,
                 memory_gib: float = 64.0, nvidia_smi: Path | None = Path("/bin/true")) -> None:
        cores = []
        for node in range(numa):
            for core in range(cores_per_node):
                first = (node * cores_per_node + core) * threads
                cores.append(CoreFact(numa=node, cpus=tuple(range(first, first + threads))))
        self.facts = dict(host="testhost", gpus=tuple(GpuFact(uuid=f"GPU-{i:08x}-fake", name="NVIDIA H200",
                                                              kind="H200") for i in range(gpus)),
                          nvidia_smi=nvidia_smi, driver=None, cores=tuple(cores), memory_gib=memory_gib)

    def read(self, *, kept_memory_gib: float) -> HostFacts:
        return HostFacts(**self.facts, kept_memory_gib=kept_memory_gib)


@dataclass(frozen=True)
class Request:
    """The submission the local scheduler reads: ``SubmitRequest`` as the factory reshapes it,
    with the shape, job name and output directory in place of sbatch arguments."""

    script: Path
    shape: ResourceShape
    job_name: str
    output_dir: Path
    tag: str
    after: str | None = None
    singleton: bool = False


def scheduler(tmp_path: Path, *, inventory: HostInventory | None = None, kill_wait_sec: float = 1.0,
              poll_sec: float = 0.05, kept_memory_gib: float = 4.0) -> LocalScheduler:
    return LocalScheduler(state_dir=tmp_path / "state", inventory=inventory or FakeInventory(),
                          kept_memory_gib=kept_memory_gib, python=Path(sys.executable), stopper="none", stopper_options={},
                          kill_wait_sec=kill_wait_sec, poll_sec=poll_sec)


def script(tmp_path: Path, name: str, body: str) -> Path:
    """A bash batch script *name* that runs *body*."""
    path = tmp_path / f"{name}.sh"
    path.write_text(f"#!/bin/bash\n{body}\n")
    return path


def shape(*, gpus: int = 0, cpus: int = 1, memory_gib: int = 1, time_sec: int = 600) -> ResourceShape:
    return ResourceShape(gpus=gpus, gpu_kind="H200" if gpus else None, cpus=cpus, memory_gib=memory_gib,
                         time_sec=time_sec)


def request(tmp_path: Path, body: str = "exit 0", *, tag: str = "t", name: str = "hls-test",
            after: str | None = None, singleton: bool = False, **kw) -> Request:
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    return Request(script=script(tmp_path, tag.replace(":", "_").replace("/", "_"), body), shape=shape(**kw),
                   job_name=name, output_dir=out, tag=tag, after=after, singleton=singleton)


def wait_for(condition: Callable[[], object], timeout: float = 30.0, what: str = "the condition") -> object:
    """Poll *condition* until it is truthy and return it; fail after *timeout* seconds."""
    deadline = time.monotonic() + timeout
    while True:
        got = condition()
        if got:
            return got
        if time.monotonic() >= deadline:
            raise AssertionError(f"{what} did not hold within {timeout}s")
        time.sleep(0.02)


def record(sched: LocalScheduler, job_id: str):
    with StateDir(sched.state.root).locked(exclusive=False):
        return sched.state.read(job_id)
