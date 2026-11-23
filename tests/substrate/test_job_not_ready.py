from __future__ import annotations

import pytest

from hls.substrate.base import SchedulerUnavailable
from hls.substrate.slurm.commands import SlurmCommands
from hls.substrate.slurm.scheduler import job_not_ready
from tests.substrate.slurm_fakes import FakeRun, done


def _show(state: str, reason: str, name: str = "hls-x") -> str:
    # As scontrol prints a job that preempts other jobs, its batch script already running.
    return (f"JobId=1234567 JobName={name}\n   UserId=user(1000) GroupId=user(1000)\n"
            f"   Priority=1 Nice=0 Account=acct QOS=normal\n   JobState={state} Reason={reason} Dependency=(null)\n"
            f"   StartTime=1759300040 EndTime=1759303640 Deadline=N/A\n"
            f"   Comment=hls:JobState=PENDING Reason=Prolog\n")


@pytest.mark.parametrize("state, reason, name, why", [
    ("RUNNING", "None", "a Reason=Prolog", None),
    ("RUNNING", "None", "x JobState=PENDING", None),
    ("RUNNING", "Prolog", "x JobState=RUNNING Reason=None", "Reason=Prolog"),
    ("CONFIGURING", "None", "JobState=RUNNING", "JobState=CONFIGURING"),
])
def test_only_the_state_line_says_whether_a_job_may_start_steps(state, reason, name, why):
    # The job name comes first, and the operator's name could otherwise speak for the state.
    run = FakeRun().on("scontrol", done(_show(state, reason, name)))
    assert job_not_ready(SlurmCommands(run=run), "1234567") == why


@pytest.mark.parametrize("state, reason, why", [
    ("RUNNING", "None", None),
    ("RUNNING", "Prolog", "Reason=Prolog"),
    ("CONFIGURING", "None", "JobState=CONFIGURING"),
    ("PENDING", "Priority", "JobState=PENDING"),
])
def test_a_job_may_start_steps_once_it_runs_outside_its_prolog(state, reason, why):
    run = FakeRun().on("scontrol", done(_show(state, reason)))
    assert job_not_ready(SlurmCommands(run=run), "1234567") == why
    assert run.calls == [["scontrol", "show", "job", "1234567"]]


@pytest.mark.parametrize("reply", [
    done(stderr="scontrol: error: Unable to contact slurm controller (connect failure)", rc=1),
    done("JobId=1234567 JobName=hls-x\n"),
])
def test_a_controller_that_does_not_say_is_unavailable(reply):
    with pytest.raises(SchedulerUnavailable):
        job_not_ready(SlurmCommands(run=FakeRun().on("scontrol", reply)), "1234567")


def test_a_malformed_job_id_never_reaches_scontrol():
    with pytest.raises(ValueError):
        job_not_ready(SlurmCommands(run=FakeRun()), "--all")
