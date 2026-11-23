from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hls.substrate.slurm.commands import CommandResult, SlurmCommands, slurm_env
from tests.substrate.slurm_fakes import FakeRun, done


@pytest.mark.parametrize("rc, stderr, unreachable", [
    (None, "", True),
    (-9, "", True),
    (1, "sbatch: error: Batch job submission failed: Socket timed out on send/recv operation", True),
    (1, "squeue: error: Unable to contact slurm controller (connect failure)", True),
    (1, "slurm_load_jobs error: Zero Bytes were transmitted or received", True),
    (1, "sbatch: error: Batch job submission failed: Invalid account or account/partition combination", False),
    (1, "scontrol: error: Job dependency problem", False),
    (0, "", False),
])
def test_unreachable_is_a_timeout_or_a_contact_failure(rc, stderr, unreachable):
    assert CommandResult(("x",), rc, "", stderr).unreachable is unreachable


def test_a_timeout_returns_no_exit_status(tmp_path):
    run = FakeRun().on("sbatch", subprocess.TimeoutExpired(["sbatch"], 300, output=b"", stderr=b"slow"))
    r = SlurmCommands(run=run)("sbatch", "--parsable", "x.sh")
    assert r.returncode is None and r.unreachable and r.stderr == "slow"


def test_commands_use_bin_dir_timeouts_and_a_clean_environment(monkeypatch):
    monkeypatch.setenv("SBATCH_EXCLUSIVE", "1")
    monkeypatch.setenv("SLURM_GPUS", "8")
    monkeypatch.setenv("SLURM_CONF", "/etc/slurm/slurm.conf")
    run = FakeRun().on("squeue", done("1|RUNNING|None|t\n"))
    cmds = SlurmCommands(Path("/opt/slurm/bin"), timeout_sec={"squeue": 5.0}, run=run)
    r = cmds("squeue", "--me")
    assert r.ok and run.calls == [["/opt/slurm/bin/squeue", "--me"]]
    env = run.kwargs[0]["env"]
    assert "SBATCH_EXCLUSIVE" not in env and "SLURM_GPUS" not in env and env["SLURM_CONF"] == "/etc/slurm/slurm.conf"
    assert run.kwargs[0]["timeout"] == 5.0


def test_slurm_env_keeps_everything_else():
    assert slurm_env({"PATH": "/usr/bin", "SRUN_CPUS_PER_TASK": "4", "HOME": "/h", "SLURM_TIME_FORMAT": "x"}) == {
        "PATH": "/usr/bin", "HOME": "/h"}


@pytest.mark.parametrize("tool, epoch", [
    ("squeue", True), ("sacct", True), ("scontrol", True), ("sbatch", False), ("scancel", False),
])
def test_only_the_query_tools_print_times_as_epoch_seconds(monkeypatch, tool, epoch):
    # sbatch passes SLURM_* on to the job, so the setting must stay out of its environment.
    monkeypatch.setenv("SLURM_TIME_FORMAT", "relative")
    run = FakeRun().on(tool, done())
    SlurmCommands(run=run)(tool, "x")
    env = run.kwargs[0]["env"]
    assert (env.get("SLURM_TIME_FORMAT") == "%s") is epoch
    assert epoch or "SLURM_TIME_FORMAT" not in env
