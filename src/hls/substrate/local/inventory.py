"""What the host holds for allocations: its GPUs by UUID and kind, its CPUs as cores on NUMA nodes,
and its memory.

The scheduler reads the inventory once when it is made and again for every dry run, and writes it
to the state directory (``HostFacts``), where the supervisors read it. ``SysInventory`` reads the
real host: nvidia-smi for the GPUs, ``/sys`` for the CPUs and ``/proc/meminfo`` for the memory, and
on a host with GPUs the NVIDIA driver's files that every GPU role is given
(``hls.substrate.docker.nvidia``).
GPUs are named by UUID only, since a host's device minors need not follow nvidia-smi's index
order."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hls.spec.resources import gpu_kind_of
from hls.substrate.base import NVIDIA_GPUS, ONLINE_CPUS, parse_cpulist
from hls.substrate.docker.nvidia import DriverUnusable, NvidiaDriver, read_driver
from hls.substrate.grant import list_gpus

_GIB = 2**30


class InventoryError(RuntimeError):
    """The host's inventory could not be read, or its sources disagree."""


class GpuFact(BaseModel):
    """One GPU of the host."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Its UUID, "GPU-...", as nvidia-smi lists it.
    uuid: str
    #: Its product name, such as "NVIDIA H200".
    name: str
    #: The kind its name names (``hls.spec.resources.GPU_KINDS``); None for a GPU of no known kind,
    #: which no allocation is given.
    kind: str | None


class CoreFact(BaseModel):
    """One physical core of the host: the online CPUs (hardware threads) it holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The NUMA node the core lies on.
    numa: int = Field(ge=0)
    #: Its online CPUs, ascending.
    cpus: tuple[int, ...] = Field(min_length=1)


class HostFacts(BaseModel):
    """The host as the scheduler places allocations on it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The host's name.
    host: str
    #: Its GPUs, in nvidia-smi's order.
    gpus: tuple[GpuFact, ...]
    #: The nvidia-smi on the submitting process's PATH, by its absolute path, which lists the
    #: processes computing on the GPUs for the scheduler's passes, whose environment holds no PATH
    #: of the user's; None when there is none, and then no GPU is free.
    nvidia_smi: Path | None
    #: The NVIDIA driver's files every GPU role is given; None on a host with no GPUs.
    driver: NvidiaDriver | None
    #: Its cores, by their first CPU.
    cores: tuple[CoreFact, ...]
    #: Its memory, ``MemTotal``, in GiB.
    memory_gib: float = Field(gt=0)
    #: The memory no allocation is given, for the host's own processes, the orchestrator and the
    #: supervisors, in GiB.
    kept_memory_gib: float = Field(ge=0)

    @model_validator(mode="after")
    def _distinct(self) -> HostFacts:
        uuids = [g.uuid for g in self.gpus]
        if len(set(uuids)) != len(uuids):
            raise ValueError("a GPU UUID is listed twice")
        cpus = [c for core in self.cores for c in core.cpus]
        if len(set(cpus)) != len(cpus):
            raise ValueError("a CPU lies on two cores")
        return self

    @property
    def allocatable_memory_gib(self) -> float:
        """The memory allocations may hold between them."""
        return self.memory_gib - self.kept_memory_gib


class HostInventory(ABC):
    """Reads what the host holds."""

    @abstractmethod
    def read(self, *, kept_memory_gib: float) -> HostFacts:
        """The host's facts, with *kept_memory_gib* kept back from allocations. Raise
        InventoryError when they cannot be read or disagree."""


class SysInventory(HostInventory):
    """The real host: GPUs from nvidia-smi, cross-checked against the driver's list of GPUs, CPUs
    and cores from ``/sys``, memory from ``/proc/meminfo``, all under *root*; *which* finds the
    nvidia-smi on this process's PATH that the scheduler's passes run, and the driver's programs.
    A host with GPUs whose driver's files a GPU role could not be given is refused."""

    def __init__(self, *, root: Path = Path("/"),
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 which: Callable[[str], str | None] = shutil.which) -> None:
        #: The root of the host's file system.
        self.root = root
        self._run = run
        self._which = which

    def read(self, *, kept_memory_gib: float) -> HostFacts:
        found = self._which("nvidia-smi")
        gpus = tuple(self._gpus())
        driver = None
        if gpus:
            try:
                driver = read_driver(proc=self.root / "proc", run=self._run, which=self._which)
            except DriverUnusable as e:
                raise InventoryError(f"the host's NVIDIA driver cannot be given to a GPU role: {e}") from None
        return HostFacts(host=socket.gethostname().split(".", 1)[0], gpus=gpus,
                         nvidia_smi=None if found is None else Path(found).resolve(), driver=driver,
                         cores=tuple(read_cores(self.root)), memory_gib=read_memory_gib(self.root),
                         kept_memory_gib=kept_memory_gib)

    def _gpus(self) -> list[GpuFact]:
        """The GPUs the driver lists, each named by nvidia-smi. A host whose driver lists none, or
        has no driver loaded, has none, and nvidia-smi, which fails without a driver, is not asked."""
        try:
            listed = len(os.listdir(self.root / NVIDIA_GPUS))
        except FileNotFoundError:
            listed = 0
        except OSError as e:
            raise InventoryError(f"{self.root / NVIDIA_GPUS} cannot be listed: {e}") from e
        if listed == 0:
            return []
        gpus = list_gpus(self._run)
        if gpus is None:
            raise InventoryError("nvidia-smi failed, so the GPUs cannot be identified")
        if len(gpus) != listed:
            raise InventoryError(f"nvidia-smi lists {len(gpus)} GPUs but the driver lists {listed}")
        return [GpuFact(uuid=g.uuid, name=g.name, kind=gpu_kind_of(g.name)) for g in gpus]


def read_cores(root: Path) -> list[CoreFact]:
    """The host's cores under *root*: each online CPU grouped with the online CPUs of its core
    (``topology/core_cpus_list``, or ``thread_siblings_list`` on kernels before 5.8), on the NUMA
    node whose ``cpulist`` holds it, node 0 on a kernel without NUMA nodes. Raise InventoryError
    when a file cannot be read or a CPU lies on no node or on two."""
    cpu_dir = root / "sys/devices/system/cpu"
    try:
        online = parse_cpulist((root / ONLINE_CPUS).read_text())
        node_of = _numa_nodes(root / "sys/devices/system/node")
        cores: dict[tuple[int, ...], set[int]] = {}
        for cpu in sorted(online):
            topology = cpu_dir / f"cpu{cpu}" / "topology"
            path = topology / "core_cpus_list"
            if not path.exists():
                path = topology / "thread_siblings_list"
            siblings = tuple(sorted(parse_cpulist(path.read_text()) & online))
            cores.setdefault(siblings, set()).add(cpu)
    except (OSError, ValueError) as e:
        raise InventoryError(f"the CPU topology under {root} cannot be read: {e}") from e
    out = []
    for siblings in sorted(cores, key=min):
        nodes = {node_of.get(c, 0 if not node_of else None) for c in siblings}
        if None in nodes or len(nodes) != 1:
            raise InventoryError(f"CPUs {list(siblings)} of one core do not lie on one NUMA node")
        out.append(CoreFact(numa=nodes.pop(), cpus=siblings))
    return out


def _numa_nodes(node_dir: Path) -> dict[int, int]:
    """Each CPU's NUMA node from ``node<N>/cpulist``; empty on a kernel without NUMA nodes."""
    if not node_dir.is_dir():
        return {}
    out: dict[int, int] = {}
    for entry in node_dir.iterdir():
        number = entry.name[len("node"):]
        if not entry.name.startswith("node") or not number.isdecimal():
            continue
        for cpu in parse_cpulist((entry / "cpulist").read_text()):
            if cpu in out:
                raise ValueError(f"CPU {cpu} lies on NUMA nodes {out[cpu]} and {number}")
            out[cpu] = int(number)
    return out


def read_memory_gib(root: Path) -> float:
    """``MemTotal`` of ``/proc/meminfo`` under *root*, in GiB. Raise InventoryError when it cannot
    be read."""
    path = root / "proc/meminfo"
    try:
        for line in path.read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / _GIB
    except (OSError, ValueError, IndexError) as e:
        raise InventoryError(f"{path} cannot be read: {e}") from e
    raise InventoryError(f"{path} has no MemTotal")

