from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from hls.substrate.base import DependencyRefused, JobState, Refused, Scheduler, SchedulerUnavailable, SubmitUnknown
from hls.substrate.slurm.commands import SlurmCommands
from hls.substrate.slurm.resources import placement_args
from hls.substrate.slurm.scheduler import (NO_SBATCH, SlurmScheduler, check_job_id, ended_jobs, job_end_time,
                                           normalize_state, parse_slurm_time)
from tests.substrate.slurm_fakes import ONE_GPU, FakeRun, done, slurm_request
from tests.substrate.test_slurm_resources import SITE

REQ = slurm_request("hls:runs/t/r/0/1", script=Path("/s/batch.sh"))
#: The allocation's own arguments of ``REQ``, which sbatch is given first: its output goes to
#: ``slurm-%j.out`` in the request's output directory.
ARGS = placement_args(ONE_GPU, SITE, REQ.placement, job_name="x", output=Path("/o/slurm-%j.out"))
TIMEOUT = subprocess.TimeoutExpired(["sbatch"], 300)
# Slurm times as the tools print them under SLURM_TIME_FORMAT=%s.
T_NOW = 1790935200
T_EARLIER = 1790931600
T_ESTIMATE = 1790942400


def _sched(run: FakeRun) -> SlurmScheduler:
    return SlurmScheduler(SlurmCommands(run=run))


def test_submit_returns_the_id_and_tags_the_job():
    run = FakeRun().on("sbatch", done("1234110;alpha\n"))
    assert _sched(run).submit(REQ) == "1234110"
    assert run.calls == [["sbatch", "--parsable", *ARGS, "--comment=hls:runs/t/r/0/1", "/s/batch.sh"]]
    assert ARGS[0] == "--job-name=x" and {"--gpus=h200:1", "--output=/o/slurm-%j.out"} <= set(ARGS)


def test_submit_runs_exactly_the_command_line_it_names_for_the_record():
    """The orchestrator records the line ``submit_argv`` names, so ``submit`` must run that line."""
    assert _sched(FakeRun()).submit_argv(REQ) == ("sbatch", "--parsable", *ARGS, "--comment=hls:runs/t/r/0/1",
                                                  "/s/batch.sh")
    retry = slurm_request("t", after="6", singleton=True, exclude=("n1", "n2"))
    for request in (REQ, retry):
        run = FakeRun().on("sbatch", done("7\n"))
        sched = _sched(run)
        sched.submit(request)
        assert run.calls == [list(sched.submit_argv(request))]
    assert sched.submit_argv(retry)[-5:] == ("--exclude=n1,n2", "--comment=t", "--dependency=afterok:6,singleton",
                                             "--kill-on-invalid-dep=no", "/b")


def test_a_dependent_never_gets_killed_by_the_site():
    run = FakeRun().on("sbatch", done("7\n"))
    _sched(run).submit(slurm_request("t", after="6"))
    assert run.calls[0][-4:] == ["--comment=t", "--dependency=afterok:6", "--kill-on-invalid-dep=no", "/b"]


def test_a_singleton_job_waits_for_its_name_and_never_gets_killed_by_the_site():
    run = FakeRun().on("sbatch", done("7\n"))
    _sched(run).submit(slurm_request("t", singleton=True))
    assert run.calls[0][-3:] == ["--comment=t", "--dependency=singleton", "/b"]
    run = FakeRun().on("sbatch", done("8\n"))
    _sched(run).submit(slurm_request("t", after="6", singleton=True))
    assert run.calls[0][-4:] == ["--comment=t", "--dependency=afterok:6,singleton", "--kill-on-invalid-dep=no",
                                 "/b"]
    run = FakeRun().on("sbatch", done(stderr="msg\n"))
    _sched(run).test_only(slurm_request("t", singleton=True))
    assert "--dependency=singleton" in run.calls[0]


@pytest.mark.parametrize("reply, error", [
    (done(stderr="sbatch: error: Batch job submission failed: Invalid account", rc=1), Refused),
    (done(stderr="sbatch: error: Batch job submission failed: Job dependency problem", rc=1), DependencyRefused),
    (done(stderr="sbatch: error: Batch job submission failed: Socket timed out on send/recv operation", rc=1),
     SubmitUnknown),
    (TIMEOUT, SubmitUnknown),
    (done(rc=-9), SubmitUnknown),
    (done("garbage\n"), SubmitUnknown),
])
def test_a_lost_answer_is_unknown_and_a_refusal_is_refused(reply, error):
    with pytest.raises(Exception) as caught:
        _sched(FakeRun().on("sbatch", reply)).submit(REQ)
    assert type(caught.value) is error


def test_a_timed_out_submission_says_so():
    with pytest.raises(SubmitUnknown, match="sbatch timed out"):
        _sched(FakeRun().on("sbatch", TIMEOUT)).submit(REQ)
    with pytest.raises(SubmitUnknown, match="sbatch exited -9"):
        _sched(FakeRun().on("sbatch", done(rc=-9))).submit(REQ)


def test_test_only_returns_the_estimate():
    msg = "sbatch: Job 1 to start at 2026-10-02T10:00:00 using 4 processors on nodes node11 in partition gpu-b"
    run = FakeRun().on("sbatch", done(stderr=msg + "\n"))
    assert _sched(run).test_only(REQ) == msg
    assert run.calls[0][:2] == ["sbatch", "--test-only"]
    with pytest.raises(Refused):
        _sched(FakeRun().on("sbatch", done(stderr="error: invalid partition", rc=1))).test_only(REQ)
    with pytest.raises(DependencyRefused):
        _sched(FakeRun().on("sbatch", done(stderr="sbatch: error: Job dependency problem", rc=1))).test_only(REQ)
    with pytest.raises(SchedulerUnavailable):
        _sched(FakeRun().on("sbatch", TIMEOUT)).test_only(REQ)


def test_a_host_without_sbatch_is_refused_in_one_line_before_anything_is_submitted():
    missing = FileNotFoundError(2, "No such file or directory", "sbatch")
    with pytest.raises(Refused) as caught:
        _sched(FakeRun().on("sbatch", missing)).test_only(REQ)
    assert str(caught.value) == NO_SBATCH and "\n" not in NO_SBATCH
    assert 'site.substrate = "docker"' in NO_SBATCH


SQUEUE = "\n".join([
    f"1234110|RUNNING|None|{T_NOW}|hls:runs/t/r/0/1",
    "1234111|PENDING|DependencyNeverSatisfied|N/A|hls:runs/t/r/1/1",
    f"1234112|COMPLETING|None|{T_EARLIER}|someone else's | comment",
    f"1234113_4|RUNNING|None|{T_EARLIER}|array",
    f"1234114|PENDING|Resources|{T_ESTIMATE}|hls:runs/t/r/2/1",
    f"1234115|RUNNING|(null)|{T_EARLIER}|(null)",
]) + "\n"
SQUEUE_CALL = ["squeue", "--me", "--all", "--noheader", "--format=%i|%T|%r|%S|%k"]


def test_a_tag_lookup_reads_the_comment_from_squeue_only():
    run = FakeRun().on("squeue", done(SQUEUE))
    assert _sched(run).find_by_tags(["hls:runs/t/r/1/1"])["hls:runs/t/r/1/1"] == ["1234111"]
    assert run.calls == [SQUEUE_CALL]


def test_a_tag_lookup_is_empty_when_the_controller_holds_no_such_job():
    run = FakeRun().on("squeue", done(SQUEUE))
    assert _sched(run).find_by_tags(["hls:gone"])["hls:gone"] == []
    assert run.calls == [SQUEUE_CALL]
    assert _sched(FakeRun().on("squeue", done(""))).find_by_tags(["hls:x"])["hls:x"] == []


def test_a_tag_lookup_sorts_ids_numerically():
    out = f"10|RUNNING|None|{T_NOW}|hls:dup\n9|PENDING|Resources|N/A|hls:dup\n100|RUNNING|None|{T_NOW}|other\n"
    assert _sched(FakeRun().on("squeue", done(out))).find_by_tags(["hls:dup"])["hls:dup"] == ["9", "10"]


@pytest.mark.parametrize("reply", [
    TIMEOUT,
    done(stderr="squeue: error: Unable to contact slurm controller (connect failure)", rc=1),
    done(stderr="squeue: error: something else", rc=1),
])
def test_a_tag_lookup_is_none_when_squeue_did_not_answer(reply):
    assert _sched(FakeRun().on("squeue", reply)).find_by_tags(["hls:x"])["hls:x"] is None


def test_states_merge_queue_and_accounting_and_unknown_is_alive():
    run = (FakeRun().on("squeue", done(SQUEUE))
           .on("sacct", done(f"1234000|CANCELLED by 1000|0:15|{T_EARLIER}\n"
                             "1234001|NODE_FAIL|0:0|Unknown\n"
                             f"1234002|RUNNING|0:0|{T_NOW}\n"
                             f"1234003|PENDING|0:0|{T_ESTIMATE}\n")))
    ids = ["1234110", "1234111", "1234112", "1234114", "1234115", "1234000", "1234001", "1234002",
           "1234003", "1234999"]
    got = _sched(run).states(ids)
    assert run.calls[0] == SQUEUE_CALL
    assert got["1234110"].state is JobState.RUNNING and got["1234110"].started_at == float(T_NOW)
    assert got["1234111"].started_at is None and got["1234001"].started_at is None
    assert got["1234111"].reason == "DependencyNeverSatisfied"
    assert got["1234112"].state is JobState.RUNNING
    assert got["1234115"].state is JobState.RUNNING and got["1234115"].reason is None
    assert got["1234000"].state is JobState.CANCELLED and got["1234000"].exit_code == "0:15"
    assert got["1234000"].started_at == float(T_EARLIER)
    assert got["1234001"].state is JobState.NODE_FAIL and got["1234001"].exit_code == "0:0"
    assert got["1234999"].state is JobState.UNKNOWN and not got["1234999"].state.terminal
    assert run.argv_of("sacct")[0][4] == "--jobs=1234000,1234001,1234002,1234003,1234999"


def test_ended_jobs_are_those_the_controller_reports_in_a_final_state_asked_by_exact_id():
    """A runner sweeps the node scratch of jobs that have ended: those squeue or sacct report in a
    final state. A queued, running or unknown job counts as alive, and only the ids asked for are
    asked about."""
    run = (FakeRun().on("squeue", done(SQUEUE))
           .on("sacct", done(f"1234000|CANCELLED by 1000|0:15|{T_EARLIER}\n"
                             "1234001|NODE_FAIL|0:0|Unknown\n"
                             f"1234002|RUNNING|0:0|{T_NOW}\n")))
    ids = ["1234110", "1234000", "1234001", "1234002", "1234999"]
    assert ended_jobs(_sched(run), ids) == {"1234000", "1234001"}
    assert run.argv_of("sacct")[0][4] == "--jobs=1234000,1234001,1234002,1234999"


def test_a_pending_job_has_no_start_and_a_live_job_has_no_exit_code():
    run = (FakeRun().on("squeue", done(SQUEUE))
           .on("sacct", done(f"1234002|RUNNING|0:0|{T_NOW}\n1234003|PENDING|0:0|{T_ESTIMATE}\n")))
    got = _sched(run).states(["1234114", "1234002", "1234003"])
    assert got["1234114"].state is JobState.PENDING and got["1234114"].started_at is None
    assert got["1234114"].reason == "Resources"
    assert got["1234002"].state is JobState.RUNNING and got["1234002"].exit_code is None
    assert got["1234002"].started_at == float(T_NOW)
    assert got["1234003"].state is JobState.PENDING and got["1234003"].exit_code is None
    assert got["1234003"].started_at is None


def test_states_use_accounting_when_squeue_is_down():
    run = (FakeRun().on("squeue", TIMEOUT)
           .on("sacct", done(f"5|RUNNING|0:0|{T_EARLIER}\n")))
    got = _sched(run).states(["5", "6"])
    assert got["5"].state is JobState.RUNNING and got["6"].state is JobState.UNKNOWN
    assert got["5"].exit_code is None


@pytest.mark.parametrize("text, seconds", [
    (str(T_NOW), float(T_NOW)), (f" {T_NOW}\n", float(T_NOW)), ("Unknown", None), ("None", None), ("N/A", None),
    ("", None), ("2026-10-02T10:00:00", None),
])
def test_parse_slurm_time_reads_epoch_seconds(text, seconds):
    assert parse_slurm_time(text) == seconds


@pytest.mark.parametrize("raw, state", [
    ("CANCELLED by 1000", JobState.CANCELLED), ("COMPLETED", JobState.COMPLETED), ("FAILED", JobState.FAILED),
    ("OUT_OF_MEMORY", JobState.OUT_OF_MEMORY), ("PREEMPTED", JobState.PREEMPTED), ("CONFIGURING", JobState.RUNNING),
    ("REQUEUED", JobState.PENDING), ("CANCELLED+", JobState.CANCELLED), ("WEIRD", JobState.UNKNOWN),
])
def test_normalize_state(raw, state):
    assert normalize_state(raw) is state


def test_rewire():
    run = FakeRun().on("scontrol", done())
    _sched(run).rewire("11", "12")
    assert run.calls == [["scontrol", "update", "JobId=11", "Dependency=afterok:12"]]


@pytest.mark.parametrize("reply, error", [
    (done(stderr="Job dependency problem for job 11", rc=1), DependencyRefused),
    (done(stderr="slurm_update error: Invalid account", rc=1), Refused),
    (TIMEOUT, SchedulerUnavailable),
    (done(stderr="slurm_update error: Unable to contact slurm controller", rc=1), SchedulerUnavailable),
])
def test_rewire_failures(reply, error):
    with pytest.raises(Exception) as caught:
        _sched(FakeRun().on("scontrol", reply)).rewire("11", "12")
    assert type(caught.value) is error


def test_cancel_names_one_id_and_nothing_else():
    run = FakeRun().on("scancel", done())
    _sched(run).cancel("1234110")
    assert run.calls == [["scancel", "1234110"]]


@pytest.mark.parametrize("bad", ["-u", "--name=hls-reference", "1234110 1234111", "", "4101_1"])
def test_job_ids_are_plain_numbers(bad):
    with pytest.raises(ValueError):
        check_job_id(bad)
    with pytest.raises(ValueError):
        _sched(FakeRun()).cancel(bad)


def test_cancelling_a_finished_job_counts_as_done_and_a_lost_answer_does_not():
    gone = FakeRun().on("scancel", done(stderr="scancel: error: Invalid job id specified", rc=1))
    _sched(gone).cancel("5")
    with pytest.raises(SchedulerUnavailable):
        _sched(FakeRun().on("scancel", TIMEOUT)).cancel("5")
    contact = done(stderr="scancel: error: Unable to contact slurm controller (connect failure)", rc=1)
    with pytest.raises(SchedulerUnavailable):
        _sched(FakeRun().on("scancel", contact)).cancel("5")


def test_a_refused_cancel_is_refused_not_unavailable():
    denied = done(stderr="scancel: error: Access/permission denied for job 5", rc=1)
    with pytest.raises(Exception) as caught:
        _sched(FakeRun().on("scancel", denied)).cancel("5")
    assert type(caught.value) is Refused


def test_job_end_time():
    out = (f"JobId=5 JobName=x\n   StartTime={T_NOW} EndTime={T_NOW + 1800} Deadline=N/A\n")
    run = FakeRun().on("scontrol", done(out))
    assert job_end_time(SlurmCommands(run=run), "5") == float(T_NOW + 1800)
    assert job_end_time(SlurmCommands(run=FakeRun().on("scontrol", TIMEOUT)), "5") is None
    unset = FakeRun().on("scontrol", done("JobId=5 JobName=x\n   StartTime=Unknown EndTime=Unknown Deadline=N/A\n"))
    assert job_end_time(SlurmCommands(run=unset), "5") is None


def test_only_the_start_time_line_says_when_a_job_ends():
    # The job name comes first, and the operator's name could otherwise speak for the end.
    out = (f"JobId=5 JobName=x EndTime={T_NOW + 99}\n   StartTime={T_NOW} EndTime={T_NOW + 1800} Deadline=N/A\n"
           f"   Comment=EndTime={T_NOW + 7}\n")
    assert job_end_time(SlurmCommands(run=FakeRun().on("scontrol", done(out))), "5") == float(T_NOW + 1800)


def test_find_by_tags_asks_squeue_once_for_every_tag():
    out = f"10|RUNNING|None|{T_NOW}|hls:a\n9|PENDING|Resources|N/A|hls:a\n11|RUNNING|None|{T_NOW}|hls:b\n"
    run = FakeRun().on("squeue", done(out))
    got = _sched(run).find_by_tags(["hls:a", "hls:b", "hls:none", "hls:a"])
    assert got == {"hls:a": ["9", "10"], "hls:b": ["11"], "hls:none": []}
    assert run.calls == [SQUEUE_CALL]
    assert _sched(FakeRun()).find_by_tags([]) == {}


def test_find_by_tags_matches_a_tag_exactly_not_as_a_prefix_or_a_part():
    out = (f"10|RUNNING|None|{T_NOW}|hls:r/0/1\n11|RUNNING|None|{T_NOW}|hls:r/0/10\n"
           f"12|RUNNING|None|{T_NOW}|xhls:r/0/1\n")
    got = _sched(FakeRun().on("squeue", done(out))).find_by_tags(["hls:r/0/1", "hls:r/0/10", "hls:r/0/100"])
    assert got == {"hls:r/0/1": ["10"], "hls:r/0/10": ["11"], "hls:r/0/100": []}


def test_find_by_tags_is_none_for_every_tag_when_squeue_did_not_answer():
    run = FakeRun().on("squeue", TIMEOUT)
    assert _sched(run).find_by_tags(["hls:a", "hls:b"]) == {"hls:a": None, "hls:b": None}
    assert run.calls == [SQUEUE_CALL]


def test_tags_are_looked_up_in_one_batch_only():
    # One API per operation: a scheduler answers every tag from one lookup.
    class Batch(Scheduler):
        def find_by_tags(self, tags):
            return {tag: [] for tag in tags}

        def submit(self, request): ...
        def submit_argv(self, request): ...
        def test_only(self, request): ...
        def states(self, job_ids): ...
        def rewire(self, job_id, after): ...
        def cancel(self, job_id): ...

    assert Batch().find_by_tags(["a"]) == {"a": []}
    assert not hasattr(Scheduler, "find_by_tag") and not hasattr(SlurmScheduler, "find_by_tag")

    class NoLookup(Scheduler):
        def submit(self, request): ...
        def submit_argv(self, request): ...
        def test_only(self, request): ...
        def states(self, job_ids): ...
        def rewire(self, job_id, after): ...
        def cancel(self, job_id): ...

    with pytest.raises(TypeError, match="find_by_tags"):
        NoLookup()
