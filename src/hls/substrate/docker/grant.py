"""Measuring a role container's grant from the host, while its gate waits.

A container role runs its own image, which need not hold the step wrapper, so the launcher
measures what the container was given from outside it, through its first process (``pid``, as
the host numbers it), and checks it the way the step wrapper checks a step
(``hls.substrate.role_exec.check_role``):
- GPUs: the ``nvidia<N>`` device nodes in the container's ``/dev`` (``/proc/<pid>/root/dev``),
  named by the host's ``nvidia-smi``, which lists each GPU's UUID with its PCI bus id, and by the
  driver's ``/proc/driver/nvidia/gpus/<bus id>/information``, which gives the GPU's device node
  minor number (``gpus_by_minor``). They must be exactly the GPUs whose device nodes the
  container was created with.
- CPUs: the process's affinity, which its cpuset sets.
- Memory, swap and the pids limit: the files of the container's cgroup, found through
  ``/proc/<pid>/cgroup``. The memory limit must hold no swap, and the pids limit must be the
  one the container was created with.
- Disk: the free space of the job's node-local scratch, which holds the container's workdir."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hls.spec.resources import GrantedResources, gpu_kind_of
from hls.substrate.cgroup import CGROUP_ROOT, read_limit
from hls.substrate.grant import Gpu, Measured

_GIB = 2**30
#: Seconds ``nvidia-smi`` may take to list the host's GPUs.
NVIDIA_SMI_TIMEOUT_SEC = 60


@dataclass(frozen=True)
class ContainerGrant:
    """What a container was given, measured from the host."""

    #: The measured slice and every fact that disagrees with what the container was created with.
    measured: Measured
    #: The container's cgroup directory on the host, which holds its limits.
    cgroup_dir: Path
    #: The CPUs its first process may run on, ascending.
    cpus: list[int] = field(default_factory=list)


def container_cgroup(pid: int, *, proc: Path = Path("/proc"), cgroup_root: Path = CGROUP_ROOT) -> Path:
    """The cgroup v2 directory of process *pid* on the host. Raise RuntimeError when the process
    has no cgroup v2 entry or the directory holds no memory limit file."""
    try:
        text = (proc / str(pid) / "cgroup").read_text()
    except OSError as e:
        raise RuntimeError(f"cannot read the cgroup of process {pid}: {e}") from None
    for line in text.splitlines():
        if line.startswith("0::/"):
            parts = [p for p in line[3:].split("/") if p]
            if ".." in parts or not parts:
                break
            directory = cgroup_root.joinpath(*parts)
            if not (directory / "memory.max").is_file():
                raise RuntimeError(f"the cgroup {directory} of process {pid} has no memory.max")
            return directory
    raise RuntimeError(f"process {pid} has no cgroup v2 entry of its own")


def device_minors(pid: int, *, proc: Path = Path("/proc")) -> list[int]:
    """The minor numbers of the ``nvidia<N>`` device nodes in the ``/dev`` of process *pid*'s
    root, ascending."""
    dev = proc / str(pid) / "root" / "dev"
    found = []
    try:
        names = os.listdir(dev)
    except OSError as e:
        raise RuntimeError(f"cannot list /dev of process {pid}: {e}") from None
    for name in names:
        number = name[len("nvidia"):]
        if name.startswith("nvidia") and number.isdecimal():
            found.append(int(number))
    return sorted(found)


class GpusUnknown(Exception):
    """The host's GPUs cannot be told by their device nodes; the message says why, in one line."""


def driver_gpu_dir(bus_id: str) -> str:
    """The name of the driver's directory of the GPU at PCI bus id *bus_id*, as nvidia-smi prints it
    (``00000000:03:00.0``): the same address with a four-digit domain, in lowercase
    (``0000:03:00.0``). Raise ValueError for anything but a PCI address."""
    domain, sep, rest = bus_id.strip().lower().partition(":")
    if not sep or not re.fullmatch(r"[0-9a-f]{1,8}", domain) or not re.fullmatch(r"[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]",
                                                                                   rest):
        raise ValueError(f"{bus_id!r} is not a PCI bus id")
    return f"{int(domain, 16):04x}:{rest}"


def device_minor(information: str) -> int:
    """The ``Device Minor`` of a GPU's ``information`` file, the minor number of its ``nvidia<N>``
    device node. Raise ValueError when the file gives none."""
    for line in information.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip() == "Device Minor" and value.strip().isdecimal():
            return int(value.strip())
    raise ValueError("it gives no Device Minor")


def gpus_by_minor(run: Callable[..., subprocess.CompletedProcess] = subprocess.run, *,
                  proc: Path = Path("/proc")) -> dict[int, Gpu]:
    """The host's GPUs by the minor number of their device nodes: each GPU ``nvidia-smi`` lists by
    UUID, name and PCI bus id, fields every driver's ``--query-gpu`` takes, with the minor number
    the driver gives in its ``information`` file under ``<proc>/driver/nvidia/gpus/<bus id>``. {}
    when nvidia-smi lists no GPU or is not installed. Raise GpusUnknown, saying why in one line,
    when nvidia-smi fails or prints anything else, or the driver gives no minor number for a GPU it
    lists."""
    try:
        p = run(["nvidia-smi", "--query-gpu=uuid,name,pci.bus_id", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SEC)
    except FileNotFoundError:
        return {}
    except subprocess.TimeoutExpired:
        raise GpusUnknown(f"nvidia-smi did not answer within {NVIDIA_SMI_TIMEOUT_SEC} s") from None
    if p.returncode != 0:
        if "No devices were found" in p.stdout + p.stderr:
            return {}
        said = next((line.strip() for line in (p.stderr + p.stdout).splitlines() if line.strip()), "")
        raise GpusUnknown(f"nvidia-smi exited {p.returncode}" + (f": {said}" if said else ""))
    out = {}
    for line in p.stdout.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        try:
            if len(fields) != 3:
                raise ValueError(line)
            directory = proc / "driver" / "nvidia" / "gpus" / driver_gpu_dir(fields[2])
        except ValueError:
            raise GpusUnknown(f"nvidia-smi listed {line.strip()!r}, not a UUID, a name and a PCI bus id") from None
        try:
            minor = device_minor((directory / "information").read_text())
        except OSError as e:
            raise GpusUnknown(f"the driver's {directory}/information of GPU {fields[0]} cannot be read: "
                              f"{e.strerror}") from None
        except ValueError as e:
            raise GpusUnknown(f"the driver's {directory}/information of GPU {fields[0]}: {e}") from None
        out[minor] = Gpu(uuid=fields[0], name=fields[1])
    return out


def measure_container(pid: int, *, requested_gpus: Sequence[str], pids_limit: int, scratch: Path,
                      proc: Path = Path("/proc"), cgroup_root: Path = CGROUP_ROOT,
                      run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                      affinity: Callable[[int], set[int]] = os.sched_getaffinity) -> ContainerGrant:
    """Measure the grant of the container whose first process is *pid* on the host. It was
    created with the device nodes of the GPUs *requested_gpus* (UUIDs) and the pids limit *pids_limit*.
    Raise RuntimeError when the process or its cgroup cannot be read, as when the container
    already ended."""
    problems: list[str] = []
    cgroup_dir = container_cgroup(pid, proc=proc, cgroup_root=cgroup_root)
    try:
        cpus = sorted(affinity(pid))
    except OSError as e:
        raise RuntimeError(f"cannot read the CPU affinity of process {pid}: {e}") from None
    minors = device_minors(pid, proc=proc)
    gpus: list[Gpu] = []
    identified = True
    if minors:
        try:
            table = gpus_by_minor(run, proc=proc)
        except GpusUnknown as e:
            problems.append(f"the container's GPUs cannot be identified: {e}")
            identified = False
        else:
            unknown = [m for m in minors if m not in table]
            if unknown:
                problems.append(f"the container has device nodes nvidia{unknown} that nvidia-smi does not list")
                identified = False
            gpus = [table[m] for m in minors if m in table]
    seen = sorted(g.uuid for g in gpus)
    if identified and seen != sorted(requested_gpus):
        problems.append(f"the container's device nodes are GPUs {seen}, and it was created with those of "
                        f"{sorted(requested_gpus)}")
    try:
        memory = read_limit(cgroup_dir / "memory.max")
        swap_file = cgroup_dir / "memory.swap.max"
        swap = read_limit(swap_file) if swap_file.exists() else None
        pids = read_limit(cgroup_dir / "pids.max")
    except (OSError, ValueError) as e:
        raise RuntimeError(f"cannot read the limits of cgroup {cgroup_dir}: {e}") from None
    if memory is None:
        problems.append(f"{cgroup_dir} has no memory limit")
        memory = 0
    if swap != 0:
        problems.append(f"{cgroup_dir} lets the container swap ({'no limit' if swap is None else f'{swap} bytes'})")
    if pids != pids_limit:
        problems.append(f"{cgroup_dir} holds the container to {pids or 'no'} tasks, not {pids_limit}")
    kinds = {gpu_kind_of(g.name) for g in gpus}
    kind = kinds.pop() if len(kinds) == 1 else None
    granted = GrantedResources(
        gpus=len(minors),
        gpu_kind=kind if minors else None,
        cpus=max(1, len(cpus)),
        memory_gib=max(memory, 1) / _GIB,
        disk_free_gib=shutil.disk_usage(scratch).free / _GIB,
    )
    measured = Measured(granted=granted, gpu_uuids=[g.uuid for g in gpus], swap_max_bytes=swap, problems=problems)
    return ContainerGrant(measured=measured, cgroup_dir=cgroup_dir, cpus=cpus)
