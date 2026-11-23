"""A stand-in for subprocess.run that answers Slurm commands from a script of replies."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from hls.spec.resources import ResourceShape
from hls.substrate.base import SubmitRequest
from hls.substrate.slurm.resources import SlurmPlacement

Reply = subprocess.CompletedProcess | Exception | Callable[[list[str]], subprocess.CompletedProcess]


def done(stdout: str = "", stderr: str = "", rc: int = 0) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], rc, stdout, stderr)


@dataclass
class FakeRun:
    """Replies to commands by tool name, in order; records every call."""

    replies: dict[str, list[Reply]] = field(default_factory=dict)
    calls: list[list[str]] = field(default_factory=list)
    kwargs: list[dict] = field(default_factory=list)

    def on(self, tool: str, *replies: Reply) -> FakeRun:
        self.replies.setdefault(tool, []).extend(replies)
        return self

    def __call__(self, argv: list[str], **kwargs) -> subprocess.CompletedProcess:
        self.calls.append(argv)
        self.kwargs.append(kwargs)
        tool = argv[0].rsplit("/", 1)[-1]
        queue = self.replies.get(tool)
        if not queue:
            raise AssertionError(f"unexpected command {argv}")
        reply = queue.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            return reply(argv)
        return reply

    def argv_of(self, tool: str) -> list[list[str]]:
        return [c for c in self.calls if c[0].rsplit("/", 1)[-1] == tool]


#: The allocation ``slurm_request`` asks for: one H200, four CPUs and 16 GiB for 30 minutes.
ONE_GPU = ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=16, time_sec=1800)


def slurm_request(tag: str, *, job_name: str = "x", after: str | None = None, singleton: bool = False,
                  exclude: Sequence[str] = (), script: Path = Path("/b")) -> SubmitRequest:
    """A request to submit ``ONE_GPU`` on the test site's GPU partition, under its QOS, with its
    output in ``/o``; *exclude* names nodes it must not land on."""
    from tests.substrate.test_slurm_resources import SITE

    return SubmitRequest(script=script, shape=ONE_GPU, job_name=job_name, output_dir=Path("/o"),
                         placement=SlurmPlacement(partition=None, qos=None, socket_tasks=None, exclude=list(exclude)),
                         site=SITE, tag=tag, after=after, singleton=singleton)
