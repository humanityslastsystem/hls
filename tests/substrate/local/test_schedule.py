"""The queue pass on records written by hand, with this process standing in for every live
supervisor: who gets a reservation, in what order, and what a gone supervisor leaves."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

from hls.substrate.base import JobState
from hls.substrate.local.records import JobRecord, ProcessId, StateDir, reported_state
from hls.substrate.local.schedule import run_pass
from hls.substrate.local.stoppers import stopper
from tests.substrate.local.helpers import FakeInventory, shape

NOW = 1_000_000.0


def _state(tmp_path: Path, **inventory) -> StateDir:
    """A state directory whose host is a ``FakeInventory`` of *inventory*."""
    state = StateDir(tmp_path / "state")
    with state.locked():
        state.write_host(FakeInventory(**inventory).read(kept_memory_gib=0))
    return state


def _gone() -> ProcessId:
    p = subprocess.Popen(["sleep", "30"])
    ident = ProcessId.of(p.pid)
    p.kill()
    p.wait()
    return ident


def _job(state: StateDir, job_id: int, *, supervisor: ProcessId | None = None, after: str | None = None,
         name: str = "j", singleton: bool = False, **kw) -> JobRecord:
    record = JobRecord(job_id=str(job_id), tag=f"t{job_id}", name=name, script=Path("/x.sh"), output=Path("/x.out"),
                       shape=shape(**kw), after=after, singleton=singleton, stopper="none", stopper_options={}, kill_wait_sec=10,
                       poll_sec=1, token=uuid.uuid4().hex, submitted_at=NOW,
                       supervisor=supervisor or ProcessId.of(os.getpid()))
    with state.locked():
        state.write(record)
    return record


def _pass(state: StateDir, now: float = NOW, in_use=lambda: set()) -> dict[str, JobRecord]:
    """A pass, *in_use* the GPUs a process computes on as nvidia-smi would list them."""
    with state.locked():
        run_pass(state, now=now, stoppers=stopper, in_use=in_use)
        return {r.job_id: r for r in state.active()}


def _gpu(i: int) -> str:
    """The UUID of the fake host's GPU *i* (``FakeInventory``)."""
    return f"GPU-{i:08x}-fake"


def _end(state: StateDir, job_id: int, final: str = "COMPLETED") -> None:
    with state.locked():
        record = state.read(str(job_id))
        state.write(record.model_copy(update={"state": final, "released": True, "ended_at": NOW}))
        state.write_reservations(state.active())


def test_jobs_start_in_submission_order_and_short_ones_fill_in_behind_a_blocked_one(tmp_path):
    state = _state(tmp_path, gpus=4)
    _job(state, 1, gpus=2, time_sec=1000)
    _job(state, 2, gpus=4, time_sec=1000)
    _job(state, 3, gpus=1, time_sec=2000)
    _job(state, 4, gpus=1, time_sec=500)
    jobs = _pass(state)
    assert [jobs[j].state for j in "1234"] == ["RUNNING", "PENDING", "PENDING", "RUNNING"]
    assert jobs["2"].reason == "waiting for GPUs: 2 of the host's 4 H200 GPUs are free and it asks for 4"
    assert jobs["3"].reason == "Priority"
    assert jobs["1"].end_time == NOW + 1000 and jobs["1"].started_at == NOW
    assert set(state.reservations()) == {"1", "4"}
    _end(state, 1)
    _end(state, 4)
    jobs = _pass(state)
    assert jobs["2"].state == "RUNNING" and jobs["3"].state == "PENDING"


def test_a_job_that_never_fits_holds_nothing_back_and_the_next_blocked_one_does(tmp_path):
    state = _state(tmp_path, gpus=4, memory_gib=64)
    _job(state, 1, gpus=2, time_sec=1000)
    _job(state, 2, gpus=4, memory_gib=100, time_sec=10)  # more memory than the host gives
    _job(state, 3, gpus=4, time_sec=1000)
    _job(state, 4, gpus=1, time_sec=5000)
    jobs = _pass(state)
    assert [jobs[j].state for j in "1234"] == ["RUNNING", "PENDING", "PENDING", "PENDING"]
    assert jobs["4"].reason == "Priority"


def test_dependencies_and_singletons_hold_a_job_without_holding_back_others(tmp_path):
    state = _state(tmp_path, gpus=4)
    _job(state, 1, gpus=1)
    _job(state, 2, gpus=1, after="1")
    _job(state, 3, gpus=1, name="one", singleton=True)
    _job(state, 4, gpus=1, name="one", singleton=True)
    _job(state, 5, gpus=1, name="one")
    jobs = _pass(state)
    assert {j: jobs[j].state for j in "12345"} == {"1": "RUNNING", "2": "PENDING", "3": "RUNNING",
                                                   "4": "PENDING", "5": "RUNNING"}
    assert jobs["2"].reason == "Dependency" and jobs["4"].reason == "Dependency"
    _end(state, 1, "FAILED")
    jobs = _pass(state)
    assert jobs["2"].state == "PENDING" and jobs["2"].reason == "DependencyNeverSatisfied"


def test_a_card_with_a_process_hls_did_not_start_is_never_granted_and_free_ones_are_taken(tmp_path):
    state = _state(tmp_path, gpus=4)
    _job(state, 1, gpus=2)
    jobs = _pass(state, in_use=lambda: {_gpu(0), _gpu(2)})
    assert jobs["1"].state == "RUNNING" and jobs["1"].reservation.gpus == (_gpu(1), _gpu(3))


def test_with_processes_hls_did_not_start_on_cards_0_and_1_jobs_take_the_first_free_cards(tmp_path):
    """The shared host's case: another user's processes sit on cards 0 and 1 of eight. A 1-GPU and a
    2-GPU job take the first cards free of them, 2, then 3 and 4."""
    state = _state(tmp_path, gpus=8, cores_per_node=8)
    _job(state, 1, gpus=1)
    _job(state, 2, gpus=2)
    jobs = _pass(state, in_use=lambda: {_gpu(0), _gpu(1)})
    assert jobs["1"].reservation.gpus == (_gpu(2),)
    assert jobs["2"].reservation.gpus == (_gpu(3), _gpu(4))


def test_a_pass_lists_the_gpus_processes_with_the_nvidia_smi_the_host_facts_name(tmp_path):
    """A supervisor's pass has no PATH of the user's: it runs the nvidia-smi the host's facts name,
    here one that lists a process on card 0; with none named no card is free."""
    smi = tmp_path / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\necho '{_gpu(0)}, 4242'\n")
    smi.chmod(0o755)
    state = _state(tmp_path, gpus=2, nvidia_smi=None)
    _job(state, 1, gpus=1)
    with state.locked():
        run_pass(state, now=NOW, stoppers=stopper)
        assert state.read("1").state == "PENDING" and "cannot be read" in state.read("1").reason
        state.write_host(FakeInventory(gpus=2, nvidia_smi=smi).read(kept_memory_gib=0))
        run_pass(state, now=NOW, stoppers=stopper)
        assert state.read("1").reservation.gpus == (_gpu(1),)


def test_a_job_asking_for_more_cards_than_are_free_waits_and_takes_no_foreign_card(tmp_path):
    state = _state(tmp_path, gpus=4)
    _job(state, 1, gpus=3)
    jobs = _pass(state, in_use=lambda: {_gpu(0), _gpu(1)})
    assert jobs["1"].state == "PENDING" and jobs["1"].reservation is None and state.reservations() == {}
    assert jobs["1"].reason == ("waiting for GPUs: 2 of the host's 4 H200 GPUs are free and it asks for 3; "
                                "2 hold a process hls did not start")


def test_a_job_waits_while_too_few_cards_are_free_and_starts_on_the_pass_after_one_frees(tmp_path):
    """No card is free while processes hls did not start sit on both: the job stays queued, its reason
    saying how many are free, with no time limit, and the pass after one frees starts it there."""
    state = _state(tmp_path, gpus=2)
    _job(state, 1, gpus=1)
    for later in (0, 600, 86_400):
        jobs = _pass(state, now=NOW + later, in_use=lambda: {_gpu(0), _gpu(1)})
        assert jobs["1"].state == "PENDING"
        assert jobs["1"].reason == ("waiting for GPUs: 0 of the host's 2 H200 GPUs are free and it asks for 1; "
                                    "2 hold a process hls did not start")
    assert reported_state(jobs["1"]) is JobState.PENDING
    jobs = _pass(state, in_use=lambda: {_gpu(1)})
    assert jobs["1"].state == "RUNNING" and jobs["1"].reservation.gpus == (_gpu(0),)


def test_a_card_another_job_holds_is_busy_through_the_reservations_and_not_counted_as_foreign(tmp_path):
    state = _state(tmp_path, gpus=2)
    _job(state, 1, gpus=1, time_sec=1000)
    _job(state, 2, gpus=1, time_sec=1000)
    jobs = _pass(state, in_use=lambda: {_gpu(1)})
    assert jobs["1"].reservation.gpus == (_gpu(0),) and jobs["2"].state == "PENDING"
    assert jobs["2"].reason == ("waiting for GPUs: 0 of the host's 2 H200 GPUs are free and it asks for 1; "
                                "1 hold a process hls did not start")


def test_gpus_whose_processes_cannot_be_read_are_none_free_and_a_job_without_gpus_still_starts(tmp_path):
    state = _state(tmp_path, gpus=2)
    _job(state, 1, gpus=1)
    _job(state, 2, gpus=0, cpus=1)
    jobs = _pass(state, in_use=lambda: None)
    assert jobs["1"].state == "PENDING" and "cannot be read (nvidia-smi failed)" in jobs["1"].reason
    assert jobs["2"].state == "RUNNING"


def test_the_gpus_in_use_are_read_only_when_a_pending_job_wants_gpus(tmp_path):
    state = _state(tmp_path, gpus=2)
    _job(state, 1, gpus=0, cpus=1)
    asked = []
    _pass(state, in_use=lambda: asked.append(1) or set())
    assert asked == []


def test_a_gone_supervisor_ends_its_job_and_its_reservation_is_given_back(tmp_path):
    state = _state(tmp_path, gpus=2)
    gone = _gone()
    _job(state, 1, gpus=2, supervisor=gone)
    with state.locked():
        record = state.read("1")
    assert reported_state(record) is JobState.NODE_FAIL
    _job(state, 2, gpus=2)
    jobs = _pass(state)
    ended = state.read("1")
    assert ended.state == "NODE_FAIL" and "supervisor" in ended.note
    assert jobs["2"].state == "RUNNING"
    # Held by a job whose supervisor is gone between the reservation and the start.
    _end(state, 2)
    _job(state, 3, gpus=2)
    jobs = _pass(state)
    with state.locked():
        state.write(jobs["3"].model_copy(update={"supervisor": _gone()}))
    _job(state, 4, gpus=2)
    jobs = _pass(state)
    assert state.read("3").state == "NODE_FAIL" and state.read("3").released
    assert jobs["4"].state == "RUNNING" and set(state.reservations()) == {"4"}


def test_a_cancelled_job_whose_supervisor_is_gone_reads_cancelled(tmp_path):
    state = _state(tmp_path)
    record = _job(state, 1, gpus=1, supervisor=_gone())
    with state.locked():
        state.write(record.model_copy(update={"cancel_requested_at": NOW}))
    _pass(state)
    assert state.read("1").state == "CANCELLED"
