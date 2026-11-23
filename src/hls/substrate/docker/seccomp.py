"""The seccomp profile of a Docker container whose confinement is ``spawner``
(``hls.substrate.container.Confinement``): the agent build's sandbox, whose spawner makes the
commands' namespaces (``hls.agent.spawner``).

It is Docker's default profile, ``docker-default-seccomp.json`` beside this file, byte for byte as
moby/profiles publishes it at ``seccomp/v0.2.3`` (the profile Docker 29.5 builds in), with exactly
two rules added, each for steps of the spawner:

- ``unshare`` with no namespace flag but ``CLONE_NEWUSER``, ``CLONE_NEWNS`` and ``CLONE_NEWPID``:
  step 1, a user namespace that maps the container's ids to the host user's and a mount namespace
  of its own, and step 3, the commands' PID namespace. Docker's default allows ``unshare`` only to
  a container that holds ``CAP_SYS_ADMIN``, which no hls container holds. The cgroup, IPC,
  network, time and UTS flags stay refused, so no command makes a network namespace of its own
  behind the seal.
- ``mount``: step 2's private propagation of the new mount namespace, its tmpfs on ``/dev/shm``
  and the empty read-only tmpfs over each hidden path, and step 3's ``/proc`` of the new PID
  namespace. The container holds no capability, so the kernel allows a mount only inside the user
  namespace the spawner made and owns.

Every other call is as Docker's default has it: ``clone`` with a namespace flag, ``clone3``,
``setns``, ``umount2``, ``pivot_root`` and the new mount calls stay refused. Forking the reaper
and each command is a ``clone`` without a namespace flag, which the default allows."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

#: Docker's default profile as moby/profiles publishes it, unchanged.
DEFAULT_PROFILE = Path(__file__).with_name("docker-default-seccomp.json")
#: The namespace flags the spawner never asks ``unshare`` for: time, cgroup, UTS, IPC and network.
OTHER_NAMESPACES = 0x00000080 | 0x02000000 | 0x04000000 | 0x08000000 | 0x40000000
#: The rules the spawner's profile adds to Docker's default, each with the steps that need it.
SPAWNER_RULES: tuple[dict[str, Any], ...] = (
    {"names": ["unshare"], "action": "SCMP_ACT_ALLOW",
     "args": [{"index": 0, "value": OTHER_NAMESPACES, "valueTwo": 0, "op": "SCMP_CMP_MASKED_EQ"}],
     "comment": "hls spawner steps 1 and 3: its user, mount and PID namespaces, and no other"},
    {"names": ["mount"], "action": "SCMP_ACT_ALLOW",
     "comment": "hls spawner steps 2 and 3: private propagation, /dev/shm, the hidden paths, its own /proc"},
)


@functools.cache
def spawner_profile() -> str:
    """The profile, as the JSON text the Engine API takes after ``seccomp=``."""
    profile = json.loads(DEFAULT_PROFILE.read_text())
    profile["syscalls"] = [*profile["syscalls"], *SPAWNER_RULES]
    return json.dumps(profile, separators=(",", ":"))
