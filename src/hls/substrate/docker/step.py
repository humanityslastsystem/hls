"""What a harness role's step wrapper (``hls.substrate.role_exec``) reads of its step inside a
Docker role container: its step id and the cgroup that holds its limits.

The Docker launcher (``hls.substrate.docker.launcher``) names this substrate in every harness role
container's environment (``hls.substrate.factory.SUBSTRATE_VAR``) and sets ``STEP_ID_VAR`` to the
step's id. The container has a private cgroup namespace, so the cgroup path a process in it reads
in ``/proc/self/cgroup`` is relative to the container's own cgroup, the root of the cgroup v2
mount it sees; the role's handle rewrites that directory in the grant report to the container's
cgroup on the host, which the runner's memory watch reads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

#: The variable that holds the step's id.
STEP_ID_VAR = "HLS_STEP_ID"
#: Where a role container sees its own cgroup: the root of its private cgroup namespace.
STEP_CGROUP = Path("/sys/fs/cgroup")
#: A step id, ``<job>.<step>``, the form ``RoleReport.step`` has on every substrate.
STEP_ID = re.compile(r"[0-9]+\.[0-9]+")


def step_id(environ: Mapping[str, str]) -> str:
    """The step id the launcher set in ``STEP_ID_VAR`` of *environ*. Raise RuntimeError when it
    is unset or not ``<job>.<step>``."""
    value = environ.get(STEP_ID_VAR)
    if value is None:
        raise RuntimeError(f"{STEP_ID_VAR} is not set, so this is not a step a Docker launcher started")
    if not STEP_ID.fullmatch(value):
        raise RuntimeError(f"{STEP_ID_VAR} is {value!r}, not <job>.<step>")
    return value


def runtime_reads(name: str) -> bool:
    """False for every *name*: the daemon creates a container from its configuration alone and
    reads nothing of the environment a step starts in."""
    return False


def runtime_setting(name: str) -> bool:
    """False for every *name*: no variable of a step's environment is a setting of Docker's or
    one it copies into a container."""
    return False


def step_cgroup(cgroup: str, root: Path) -> Path:
    """The directory of the cgroup that holds a container's limits, from the cgroup v2 path
    *cgroup* of a process in it (``hls.substrate.cgroup.own_cgroup``) and *root*, where cgroup v2
    is mounted: *root* joined with that path, *root* itself for ``/``, the container's own cgroup
    under a private cgroup namespace. Raise RuntimeError for a path that leaves *root* and for a
    directory with no ``memory.max``, which a host's root cgroup never has."""
    parts = [p for p in cgroup.split("/") if p]
    if not cgroup.startswith("/") or any(p in (".", "..") for p in parts):
        raise RuntimeError(f"cgroup {cgroup!r} is not an absolute cgroup path")
    directory = root.joinpath(*parts)
    if not (directory / "memory.max").is_file():
        raise RuntimeError(f"{directory} has no memory.max, so it is not a container's own cgroup")
    return directory
