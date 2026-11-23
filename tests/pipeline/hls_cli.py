"""``hls <command>`` as the command line runs it, its instance driven on a test's scheduler."""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.pipeline import commands
from hls.substrate.base import Scheduler


def run_hls(scheduler: Scheduler, args: Sequence[str]) -> tuple[int, list[str]]:
    """The exit status of ``hls <args>`` and every line it printed, its refusals on stderr among
    them: the command line's own parsing, planning and ``commands.start``, which is given
    *scheduler* in place of the site's Slurm. An exception the command raises, but its exit, is
    raised here with its traceback. The console's logging setup is left out: it turns propagation
    off on the hls loggers, which would keep their records from caplog in every later test."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli, "configure_logging", lambda verbose=False: None)
        mp.setattr(commands, "start", functools.partial(commands.start, scheduler=scheduler))
        result = CliRunner().invoke(cli.app, list(args), catch_exceptions=False)
    return result.exit_code, result.output.splitlines()


def flags(overrides: Mapping[str, str]) -> list[str]:
    """*overrides* as the ``--<dotted>=<value>`` words of a command line."""
    return [f"--{key}={value}" for key, value in overrides.items()]
