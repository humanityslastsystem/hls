"""The versions of the tools the substrate's rules depend on, as the node a runner runs on has
them: Slurm (the launcher's step options and states), apptainer (the runtime's mount set), pasta
(the network seal's options) and the kernel. The runner
measures them with its grant and records them in its allocation status; the harness-conditions
table of each record names them per job. A version that cannot be read is recorded as None and
never refuses a run."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

#: Seconds each tool may take to print its version. ``--version`` reads nothing past the binary
#: and answers at once on a healthy node; a tool that does not answer by then is recorded as None.
VERSION_TIMEOUT_SEC = 10


class ToolVersions(BaseModel):
    """The tools of one node, each as the first line its ``--version`` prints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Slurm's, from ``srun --version`` on the allocation's PATH, such as "slurm 23.02.7"; None
    #: when it cannot be read.
    slurm: str | None
    #: apptainer's, from ``apptainer --version`` on the allocation's PATH, such as "apptainer
    #: version 1.3.4-1.el8"; None when it cannot be read.
    apptainer: str | None
    #: The pasta the network seal runs: ``site.pasta_path``, else ``pasta`` on the allocation's
    #: PATH, such as "pasta 2024_09_06.6b38f07"; None when it cannot be read.
    pasta: str | None
    #: The kernel's release (``os.uname().release``), such as "4.18.0-513.24.1.el8_9.x86_64".
    kernel: str
    #: On a Docker host, the Docker Engine's version as its daemon reports it, such as "29.5.2".
    docker: str | None = None
    #: On a Docker host, the step image's cache key (``hls.substrate.docker.step_image.step_key``),
    #: which names the recipe of the image every harness role and gateway runs; the store's entry of
    #: that key fixes its archive.
    step_image: str | None = None
    #: On a Docker host, the NVIDIA driver's, whose files every GPU role is given
    #: (``hls.substrate.docker.nvidia``), from ``/proc/driver/nvidia/version``, such as "570.211.01".
    nvidia_driver: str | None = None

    @model_serializer(mode="wrap")
    def _without_unmeasured_hosts_tools(self, handler: SerializerFunctionWrapHandler) -> dict:
        """The versions as a record holds them: a field only a Docker host has is left out when it
        was not measured, so a Slurm node's record holds exactly the Slurm fields."""
        data = handler(self)
        for name in DOCKER_FIELDS:
            if data.get(name) is None:
                data.pop(name, None)
        return data


#: The fields of ``ToolVersions`` that only a Docker host measures.
DOCKER_FIELDS = ("docker", "step_image", "nvidia_driver")


def tool_version(argv: Sequence[str], run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> str | None:
    """The first non-blank line *argv* prints, stripped; None when it cannot be started, does not
    end within ``VERSION_TIMEOUT_SEC``, exits nonzero or prints nothing."""
    try:
        done = run(list(argv), capture_output=True, text=True, timeout=VERSION_TIMEOUT_SEC)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return next((s for line in done.stdout.splitlines() if (s := line.strip())), None)


def measure_tools(pasta_path: Path | None,
                  run: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> ToolVersions:
    """The versions of this node's tools, pasta's being *pasta_path*'s (``site.pasta_path``) when
    it is given, as the seal runs it, else the one on PATH."""
    return ToolVersions(slurm=tool_version(["srun", "--version"], run),
                        apptainer=tool_version(["apptainer", "--version"], run),
                        pasta=tool_version([str(pasta_path) if pasta_path is not None else "pasta", "--version"], run),
                        kernel=os.uname().release)
