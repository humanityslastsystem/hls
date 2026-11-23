"""Reading a Slurm job's and step's cgroup v2 files, and stopping every process of a step.

Slurm places each step at
``/sys/fs/cgroup/system.slice/slurmstepd.scope/job_<id>/step_<n>/user/task_<i>``. The job's
memory limit is on ``job_<id>``, and a step's memory limit and cpuset are on
``step_<n>/user``. The job's own processes can read every step's files, so the batch step
can sample each role's memory, and can write none of them."""

from __future__ import annotations

import logging
import os
import signal
from pathlib import Path

log = logging.getLogger(__name__)

#: Where cgroup v2 is mounted.
CGROUP_ROOT = Path("/sys/fs/cgroup")


def own_cgroup(proc_cgroup: Path = Path("/proc/self/cgroup")) -> str:
    """This process's cgroup v2 path, such as ``/system.slice/.../step_0/user/task_0``.
    Raise RuntimeError when the host has no cgroup v2 entry."""
    lines = proc_cgroup.read_text().splitlines()
    for line in lines:
        if line.startswith("0::"):
            return line[3:]
    found = f"only cgroup v1 entries such as {lines[0]!r}" if lines else "no entry at all"
    raise RuntimeError(f"{proc_cgroup} has no cgroup v2 entry ('0::'), {found}")


def _parts(cgroup: str) -> list[str]:
    return [p for p in cgroup.split("/") if p]


def job_dir(cgroup: str, root: Path = CGROUP_ROOT) -> Path:
    """The ``job_<id>`` directory that contains *cgroup*, where the job's limits are."""
    parts = _parts(cgroup)
    for i, part in enumerate(parts):
        if part.startswith("job_"):
            return root.joinpath(*parts[: i + 1])
    raise RuntimeError(f"cgroup {cgroup} is not inside a Slurm job")


def step_dir(cgroup: str, root: Path = CGROUP_ROOT) -> Path:
    """The ``step_<n>/user`` directory that contains *cgroup*, where the step's limits are."""
    parts = _parts(cgroup)
    for i, part in enumerate(parts):
        if part.startswith("step_") and i + 1 < len(parts) and parts[i + 1] == "user":
            return root.joinpath(*parts[: i + 2])
    raise RuntimeError(f"cgroup {cgroup} is not inside a Slurm step")


def read_limit(path: Path) -> int | None:
    """A limit file such as ``memory.max`` in bytes; None for ``max`` (unlimited)."""
    text = path.read_text().strip()
    return None if text == "max" else int(text)


def read_int(path: Path) -> int:
    """A single-number file such as ``memory.current``."""
    return int(path.read_text().split()[0])


def read_keyed(path: Path) -> dict[str, int]:
    """A ``<key> <value>`` file such as ``memory.events`` or ``memory.stat``."""
    out: dict[str, int] = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(" ")
        if value.strip().isdecimal():
            out[key] = int(value)
    return out


def pids_limit_hits(cgroup_dir: Path) -> int | None:
    """How often a step's pids limit was reached: the ``max`` count of ``pids.events`` in the
    step's cgroup *cgroup_dir* (``step_<n>/user``) or its parent (``step_<n>``), whichever is
    higher, since a site may set the limit on either; None when neither file can be read. A step
    cgroup is made for its step, so its counts start at 0."""
    found = []
    for directory in (cgroup_dir, cgroup_dir.parent):
        try:
            count = read_keyed(directory / "pids.events").get("max")
        except (OSError, ValueError):
            continue
        if count is not None:
            found.append(count)
    return max(found) if found else None


def pids(directory: Path) -> list[int]:
    """Every process in *directory* and the cgroups below it."""
    found: list[int] = []
    for procs in sorted(directory.rglob("cgroup.procs")):
        try:
            found += [int(p) for p in procs.read_text().split()]
        except OSError:
            continue
    return found


def kill_all(directory: Path, sig: int = signal.SIGKILL) -> int:
    """Send *sig* to every process in *directory*'s cgroups and return how many were
    signalled. A process that already exited, or that this user may not signal, is skipped
    and does not count; the rest are still signalled."""
    sent = 0
    for pid in pids(directory):
        try:
            os.kill(pid, sig)
            sent += 1
        except ProcessLookupError:
            continue
        except PermissionError:
            log.warning("not permitted to signal process %d in %s", pid, directory)
    return sent
