"""Slurm's command-line tools run as local subprocesses, and how to tell a refusal from a
controller that did not answer.

The orchestrator runs on a login node of the cluster it schedules on and the runner runs
inside an allocation, so every Slurm call is a local process; there is no ssh transport.
A call that timed out, or failed with one of the messages a client prints when it cannot
reach slurmctld, says nothing about any job, and callers treat the job as alive."""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

#: Seconds each tool may take. sbatch gets the longest: under controller load it can take
#: minutes to return, and a submission that is cut short must be looked up by its tag.
DEFAULT_TIMEOUT_SEC: dict[str, float] = {
    "sbatch": 300.0,
    "squeue": 60.0,
    "sacct": 60.0,
    "scontrol": 60.0,
    "scancel": 60.0,
}
# What a Slurm client prints when it could not reach the controller or lost its answer.
_UNREACHABLE = re.compile(
    r"socket timed out|unable to contact slurm controller|zero bytes were transmitted|connection refused"
    r"|connection timed out|temporarily unavailable|communication connection failure|slurm_receive_msg"
    r"|unexpected missing socket error",
    re.IGNORECASE,
)
# The tools whose output carries times. They are run with times printed as epoch seconds.
# sbatch is not one of them: it passes SLURM_* variables on to the job, so every role would
# inherit the setting.
_QUERY_TOOLS = ("squeue", "sacct", "scontrol")
# Environment prefixes that change what a Slurm tool does without appearing in its argv.
_SLURM_INPUT_PREFIXES = ("SBATCH_", "SALLOC_", "SRUN_", "SQUEUE_", "SACCT_", "SCANCEL_", "SCONTROL_", "SLURM_")


@dataclass(frozen=True)
class CommandResult:
    """One finished Slurm command."""

    argv: tuple[str, ...]
    #: Exit status; None when the command timed out.
    returncode: int | None
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        """True when the command exited with status 0."""
        return self.returncode == 0

    @property
    def unreachable(self) -> bool:
        """True when the controller did not answer, so the call proves nothing: the command
        timed out, was killed by a signal, or printed a contact failure."""
        rc = self.returncode
        return rc is None or rc < 0 or (rc != 0 and bool(_UNREACHABLE.search(self.stderr)))


def slurm_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* without the variables that would silently add options to a Slurm command, such
    as ``SBATCH_EXCLUSIVE`` or a ``SLURM_*`` job request inherited from an enclosing job.
    ``SLURM_CONF`` stays, since it only says where the cluster's configuration is."""
    return {k: v for k, v in env.items() if k == "SLURM_CONF" or not k.startswith(_SLURM_INPUT_PREFIXES)}


def _text(data: bytes | str | None) -> str:
    if data is None:
        return ""
    return data if isinstance(data, str) else data.decode(errors="replace")


class SlurmCommands:
    """Runs Slurm tools with a timeout each and a clean environment. squeue, sacct and
    scontrol also get ``SLURM_TIME_FORMAT=%s``, so the times they print are epoch seconds."""

    def __init__(
        self,
        bin_dir: Path | None = None,
        *,
        timeout_sec: Mapping[str, float] | None = None,
        run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        #: Directory holding the tools; None looks them up on PATH.
        self.bin_dir = bin_dir
        #: Tool name to its timeout.
        self.timeout_sec = {**DEFAULT_TIMEOUT_SEC, **(timeout_sec or {})}
        #: Runs one process; tests replace it.
        self._run = run

    def __call__(self, tool: str, *args: str) -> CommandResult:
        """Run *tool* with *args*. Never raises for a nonzero exit or a timeout; a missing
        tool raises FileNotFoundError."""
        exe = str(self.bin_dir / tool) if self.bin_dir is not None else tool
        argv = (exe, *args)
        env = slurm_env(os.environ)
        if tool in _QUERY_TOOLS:
            env["SLURM_TIME_FORMAT"] = "%s"
        try:
            p = self._run(list(argv), capture_output=True, text=True, timeout=self.timeout_sec[tool], env=env)
        except subprocess.TimeoutExpired as e:
            return CommandResult(argv, None, _text(e.stdout), _text(e.stderr))
        return CommandResult(argv, p.returncode, p.stdout, p.stderr)
