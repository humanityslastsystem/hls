"""Measuring what Slurm granted: the allocation's resources from the batch step, and a
role's slice from inside its step.

The GPU count is the number of ``/dev/nvidia<N>`` device nodes this process can open,
since the device cgroup is what isolates GPUs; nvidia-smi supplies their UUIDs and product
names. CPUs come from the process's affinity, the CPU model from ``/proc/cpuinfo``, memory and
swap limits from the cgroup's files, and free disk from the node-local scratch directory."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from hls.spec.resources import GrantedResources, gpu_kind_of
from hls.substrate.cgroup import read_limit

_GIB = 2**30


@dataclass(frozen=True)
class Gpu:
    """One GPU as nvidia-smi lists it."""

    #: Its UUID, "GPU-...".
    uuid: str
    #: Its product name, such as "NVIDIA H200".
    name: str


@dataclass(frozen=True)
class Measured:
    """A grant as measured, with what could not be reconciled."""

    #: The measured resources.
    granted: GrantedResources
    #: UUIDs of the GPUs, in nvidia-smi's order.
    gpu_uuids: list[str]
    #: The swap limit in bytes; None when unlimited.
    swap_max_bytes: int | None
    #: Facts that disagree with each other, such as nvidia-smi listing a GPU whose device
    #: node does not open. Each one is a grant mismatch.
    problems: list[str] = field(default_factory=list)


def openable_gpu_devices(dev: Path = Path("/dev")) -> list[str]:
    """Names of the ``nvidia<N>`` device nodes this process may open."""
    out = []
    for node in sorted(dev.glob("nvidia[0-9]*")):
        if not node.name[len("nvidia"):].isdecimal():
            continue
        try:
            os.close(os.open(node, os.O_RDONLY))
        except OSError:
            continue
        out.append(node.name)
    return out


def gpus_computing(nvidia_smi: Path,
                   run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> set[str] | None:
    """The UUIDs of the GPUs on which the program *nvidia_smi* lists a compute process, whoever runs
    it; None when it fails or cannot be run."""
    try:
        p = run([str(nvidia_smi), "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if p.returncode != 0:
        return None
    return {line.partition(",")[0].strip() for line in p.stdout.splitlines() if line.strip()}


def list_gpus(run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> list[Gpu] | None:
    """The GPUs nvidia-smi lists for this process; [] when there is none; None when
    nvidia-smi failed for another reason."""
    try:
        p = run(["nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader"], capture_output=True,
                text=True, timeout=60)
    except FileNotFoundError:
        return []
    except subprocess.TimeoutExpired:
        return None
    if p.returncode != 0:
        return [] if "No devices were found" in p.stdout + p.stderr else None
    gpus = []
    for line in p.stdout.splitlines():
        if line.strip():
            uuid, _, name = line.partition(",")
            gpus.append(Gpu(uuid.strip(), name.strip()))
    return gpus


def _mem_total_bytes(meminfo: Path) -> int:
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError(f"{meminfo} has no MemTotal")


def _joined(names: set[str]) -> str | None:
    """The distinct *names* sorted and joined by "; "; None for none."""
    return "; ".join(sorted(names)) or None


def cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str | None:
    """The node's CPU model, every distinct ``model name`` of *cpuinfo* sorted and joined by
    "; "; None for a node that names none (some ARM kernels do not)."""
    names = set()
    for line in cpuinfo.read_text(errors="replace").splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "model name" and value.strip():
            names.add(" ".join(value.split()))
    return _joined(names)


def measure(*, limit_dir: Path, scratch: Path, dev: Path = Path("/dev"), meminfo: Path = Path("/proc/meminfo"),
            cpuinfo: Path = Path("/proc/cpuinfo"), run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
            cpus: Callable[[], int] = lambda: len(os.sched_getaffinity(0))) -> Measured:
    """Measure the grant whose memory limit is on the cgroup *limit_dir*: the job's
    directory for the allocation, the step's for a role."""
    problems = []
    devices = openable_gpu_devices(dev)
    # With no device node open there is nothing to identify, and nvidia-smi may fail to
    # start at all in a step that was given no GPU.
    gpus = list_gpus(run) if devices else []
    if gpus is None:
        problems.append("nvidia-smi failed, so the GPUs cannot be identified")
        gpus = []
    elif len(gpus) != len(devices):
        problems.append(f"nvidia-smi lists {len(gpus)} GPUs but {len(devices)} device nodes open")
    kinds = {gpu_kind_of(g.name) for g in gpus}
    kind = kinds.pop() if len(kinds) == 1 else None
    memory = read_limit(limit_dir / "memory.max")
    if memory is None:
        problems.append(f"{limit_dir} has no memory limit")
        memory = _mem_total_bytes(meminfo)
    swap_file = limit_dir / "memory.swap.max"
    swap = read_limit(swap_file) if swap_file.exists() else None
    granted = GrantedResources(
        gpus=len(devices),
        gpu_kind=kind if devices else None,
        gpu_model=_joined({g.name for g in gpus}) if devices else None,
        cpus=cpus(),
        cpu_model=cpu_model(cpuinfo),
        memory_gib=memory / _GIB,
        disk_free_gib=shutil.disk_usage(scratch).free / _GIB,
    )
    return Measured(granted=granted, gpu_uuids=[g.uuid for g in gpus], swap_max_bytes=swap, problems=problems)
