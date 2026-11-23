"""The commands that hold an orchestrator, and ``hls follow``, start numpy's thread pools at one
thread each, since they run on shared login nodes where every thread counts against the user's
process limit; the runner keeps its pools. Each case starts the command in its own process and
stops it where numpy is first imported, which is where OpenBLAS reads its size and starts its
threads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

#: The pool sizes the commands must set: OpenBLAS's (numpy's wheels), OpenMP's and MKL's.
POOLS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
SRC = Path(__file__).resolve().parents[2] / "src"

# Runs `hls <words>`, or `python -m <module> <words>` for words `-m <module> ...`, and at numpy's
# first import prints the pool variables as numpy found them and the process's thread count once
# numpy has loaded, then exits.
_PROBE = r"""
import importlib.abc
import json
import os
import runpy
import sys

POOLS = json.loads(sys.argv[1])
WORDS = sys.argv[2:]


class StopAtNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name != "numpy":
            return None
        sys.meta_path.remove(self)
        seen = {pool: os.environ.get(pool) for pool in POOLS}
        import numpy  # noqa: F401
        print(json.dumps({"pools": seen, "threads": len(os.listdir("/proc/self/task"))}), flush=True)
        os._exit(0)


sys.meta_path.insert(0, StopAtNumpy())
if WORDS[0] == "-m":
    sys.argv = WORDS[1:]
    runpy.run_module(WORDS[1], run_name="__main__", alter_sys=True)
else:
    from hls.cli import app

    app(WORDS, prog_name="hls")
"""

#: Each command that holds an orchestrator on a login node, and ``hls follow``, which watches one
#: as long as it runs, with words that parse. Every one reaches numpy before it reads anything.
LOGIN_COMMANDS = {
    "provision": ["provision", "tasks.sample_task.task"],
    "images build": ["images", "build", "sample_task"],
    "calibrate": ["calibrate", "tasks.sample_task.task"],
    "package": ["package", "tasks.sample_task.task", "--bundle", "/nonexistent/bundle"],
    "baseline-credit": ["baseline-credit", "tasks.sample_task.task", "--drives", "2"],
    "run": ["run", "tasks.sample_task.task", "--arm", "some-arm"],
    "grade": ["grade", "tasks.sample_task.task"],
    "oracle fill": ["oracle", "fill", "tasks.sample_task.task"],
    "resume": ["resume", "runs/sample_task/none"],
    "cancel": ["cancel", "runs/sample_task/none"],
    "export": ["export", "runs/sample_task/none", "--to", "/nonexistent/export"],
    "follow": ["follow", "runs/sample_task/none"],
}


def _at_numpy(words: list[str], cwd: Path, **pools: str) -> dict:
    """What the probe saw when the process running *words* on this checkout's code first imported
    numpy, from *cwd*, with the pool variables set to *pools* and every other one unset."""
    env = {k: v for k, v in os.environ.items() if k not in POOLS} | pools | {"PYTHONPATH": str(SRC)}
    done = subprocess.run([sys.executable, "-c", _PROBE, json.dumps(POOLS), *words], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout, f"numpy was never imported:\n{done.stdout}\n{done.stderr}"
    return json.loads(done.stdout.splitlines()[-1])


@pytest.mark.parametrize("command", sorted(LOGIN_COMMANDS))
def test_login_command_starts_numpy_with_one_thread_pools(command, tmp_path):
    seen = _at_numpy(LOGIN_COMMANDS[command], tmp_path)
    assert seen["pools"] == dict.fromkeys(POOLS, "1")
    assert seen["threads"] == 1


def test_a_pool_size_the_operator_set_is_kept(tmp_path):
    seen = _at_numpy(LOGIN_COMMANDS["run"], tmp_path, OPENBLAS_NUM_THREADS="3", OMP_NUM_THREADS="")
    assert seen["pools"] == {"OPENBLAS_NUM_THREADS": "3", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


@pytest.mark.parametrize("words", [
    ["-m", "hls.substrate.runner", "/nonexistent/plan", "--allocation", "0", "--submission", "1"],
    ["runner", "/nonexistent/plan", "--allocation", "0", "--submission", "1"],
], ids=["batch-script", "hls-runner"])
def test_the_runner_keeps_its_pools(words, tmp_path):
    """The batch script starts the runner as a module (``hls.substrate.slurm.batch``); people use
    ``hls runner``. Neither caps a pool: the runner runs verifier and scoring math in its process."""
    assert _at_numpy(words, tmp_path)["pools"] == dict.fromkeys(POOLS)
