"""The local scheduler with real supervisors and batch scripts, on a host given as facts."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from pathlib import Path

import pytest

from hls.substrate.base import DependencyRefused, JobState, Refused, SchedulerUnavailable
from hls.substrate.local import scheduler as local
from hls.substrate.local.records import RECLAIM_BACKOFF_SEC, ProcessId, StateDir
from hls.substrate.local.schedule import TOKEN_VAR, run_pass
from hls.substrate.local.stoppers import stopper
from hls.substrate.local.supervisor import END_TIME_VAR, JOB_ID_VAR, STATE_VAR, output_name
from tests.substrate.local.helpers import FakeInventory, record, request, scheduler, wait_for


def _state(sched, job_id):
    return sched.states([job_id])[job_id].state


def _ended(sched, job_id, timeout=30.0):
    return wait_for(lambda: _state(sched, job_id).terminal and _state(sched, job_id), timeout,
                    f"job {job_id} ending")


def _output(req, job_id) -> str:
    return (req.output_dir / output_name(job_id)).read_text()


def test_a_job_runs_its_script_with_the_job_variables_alone_and_completes(tmp_path):
    sched = scheduler(tmp_path)
    req = request(tmp_path, "env | sort; echo cpus=$(nproc)", cpus=2)
    job = sched.submit(req)
    assert job.isdecimal()
    assert _ended(sched, job) is JobState.COMPLETED
    info = sched.states([job])[job]
    assert info.exit_code == "0:0" and info.started_at is not None
    names = {line.split("=", 1)[0] for line in _output(req, job).splitlines() if "=" in line}
    # bash adds its own (PWD, SHLVL, _); nothing comes from the submitting process.
    assert names - {"PWD", "SHLVL", "_", "cpus"} == {JOB_ID_VAR, END_TIME_VAR, STATE_VAR, TOKEN_VAR}
    assert "cpus=2" in _output(req, job)
    rec = record(sched, job)
    assert rec.released and rec.reservation is not None and len(rec.reservation.cpus) == 2
    assert StateDir(sched.state.root).reservations() == {}


def test_exit_codes_and_signals_are_recorded_as_sacct_prints_them(tmp_path):
    sched = scheduler(tmp_path)
    failed = sched.submit(request(tmp_path, "exit 3", tag="a"))
    killed = sched.submit(request(tmp_path, "kill -9 $$", tag="b"))
    assert _ended(sched, failed) is JobState.FAILED and _ended(sched, killed) is JobState.FAILED
    states = sched.states([failed, killed])
    assert states[failed].exit_code == "3:0" and states[killed].exit_code == "0:9"


def test_the_end_time_is_enforced_with_sigterm_then_sigkill(tmp_path):
    sched = scheduler(tmp_path, kill_wait_sec=0.5)
    req = request(tmp_path, "trap 'echo got-term' TERM; while :; do sleep 0.05; done", time_sec=1)
    job = sched.submit(req)
    assert _ended(sched, job) is JobState.TIMEOUT
    rec = record(sched, job)
    assert rec.exit_code == "0:9" and "got-term" in _output(req, job)
    assert rec.ended_at - rec.started_at < 4
    assert local.job_end_time(sched.state, job) == rec.end_time == rec.started_at + 1


def test_the_runner_reads_its_end_time_both_ways_and_is_always_ready(tmp_path):
    sched = scheduler(tmp_path)
    req = request(tmp_path, "echo end=$HLS_JOB_END_TIME", time_sec=100)
    job = sched.submit(req)
    _ended(sched, job)
    env_end = local.env_end_time({END_TIME_VAR: _output(req, job).strip().split("=")[1]})
    assert env_end == int(record(sched, job).end_time)
    assert local.job_not_ready(job) is None
    assert local.job_id({JOB_ID_VAR: job}) == job and local.state_dir({STATE_VAR: "/x"}) == Path("/x")
    with pytest.raises(RuntimeError):
        local.job_cgroup()
    assert local.ended_jobs(sched.state, [job, "999999"]) == {job}


def test_a_dependent_starts_after_its_parent_completes(tmp_path):
    sched = scheduler(tmp_path)
    gate = tmp_path / "gate"
    parent = sched.submit(request(tmp_path, f"while [ ! -e {gate} ]; do sleep 0.02; done", tag="p"))
    child = sched.submit(request(tmp_path, "exit 0", tag="c", after=parent))
    wait_for(lambda: _state(sched, parent) is JobState.RUNNING)
    info = sched.states([child])[child]
    assert info.state is JobState.PENDING and info.reason == "Dependency"
    gate.touch()
    assert _ended(sched, parent) is JobState.COMPLETED
    assert _ended(sched, child) is JobState.COMPLETED
    assert record(sched, child).started_at >= record(sched, parent).ended_at


def test_a_dependent_of_a_failed_job_pends_until_rewired(tmp_path):
    sched = scheduler(tmp_path)
    parent = sched.submit(request(tmp_path, "exit 1", tag="p"))
    child = sched.submit(request(tmp_path, "exit 0", tag="c", after=parent))
    assert _ended(sched, parent) is JobState.FAILED
    wait_for(lambda: sched.states([child])[child].reason == "DependencyNeverSatisfied")
    time.sleep(0.3)
    assert _state(sched, child) is JobState.PENDING
    retry = sched.submit(request(tmp_path, "exit 0", tag="r"))
    sched.rewire(child, retry)
    assert _ended(sched, child) is JobState.COMPLETED


def test_submit_and_rewire_refuse_unknown_and_circular_dependencies(tmp_path):
    sched = scheduler(tmp_path)
    with pytest.raises(DependencyRefused):
        sched.submit(request(tmp_path, tag="a", after="12345"))
    gate = tmp_path / "gate"
    parent = sched.submit(request(tmp_path, f"while [ ! -e {gate} ]; do sleep 0.02; done", tag="p"))
    child = sched.submit(request(tmp_path, tag="c", after=parent))
    grandchild = sched.submit(request(tmp_path, tag="g", after=child))
    with pytest.raises(DependencyRefused):
        sched.rewire(child, "12345")
    with pytest.raises(DependencyRefused):
        sched.rewire(child, grandchild)
    with pytest.raises(DependencyRefused):
        sched.rewire(child, child)
    wait_for(lambda: _state(sched, parent) is JobState.RUNNING)
    with pytest.raises(Refused, match="is not pending"):
        sched.rewire(parent, child)
    with pytest.raises(Refused, match="not known"):
        sched.rewire("12345", parent)
    gate.touch()
    assert _ended(sched, grandchild) is JobState.COMPLETED


def test_find_by_tags_lists_jobs_that_have_not_ended(tmp_path):
    sched = scheduler(tmp_path)
    gate = tmp_path / "gate"
    running = sched.submit(request(tmp_path, f"while [ ! -e {gate} ]; do sleep 0.02; done", tag="hls:i/0/1"))
    done = sched.submit(request(tmp_path, "exit 0", tag="hls:i/1/1"))
    _ended(sched, done)
    assert sched.find_by_tags(["hls:i/0/1", "hls:i/1/1", "none"]) == {
        "hls:i/0/1": [running], "hls:i/1/1": [], "none": []}
    gate.touch()
    _ended(sched, running)
    assert sched.find_by_tags([]) == {}


def test_an_unreadable_state_directory_answers_nothing(tmp_path):
    sched = scheduler(tmp_path)
    job = sched.submit(request(tmp_path))
    _ended(sched, job)
    os.chmod(sched.state.root, 0o750)
    try:
        assert sched.find_by_tags(["t"]) == {"t": None}
        assert sched.states([job]) == {job: local.JobInfo(job, JobState.UNKNOWN)}
        with pytest.raises(SchedulerUnavailable):
            sched.cancel(job)
        with pytest.raises(SchedulerUnavailable):
            sched.rewire(job, job)
    finally:
        os.chmod(sched.state.root, 0o700)


def test_test_only_checks_the_request_against_the_empty_host(tmp_path):
    sched = scheduler(tmp_path, inventory=FakeInventory(gpus=2, numa=2, cores_per_node=4, memory_gib=64))
    answer = sched.test_only(request(tmp_path, gpus=2, cpus=8, memory_gib=60))
    assert "2 of 2 H200 GPUs" in answer and "8 of 8 CPUs" in answer
    for kw, said in ((dict(gpus=3), "3 H200 GPUs"), (dict(cpus=9), "9 CPUs"), (dict(memory_gib=61), "61 GiB")):
        with pytest.raises(Refused, match=said):
            sched.test_only(request(tmp_path, **kw))
        with pytest.raises(Refused, match=said):
            sched.submit(request(tmp_path, **kw))
    with pytest.raises(DependencyRefused):
        sched.test_only(request(tmp_path, after="777"))
    assert not list((sched.state.root / "active").iterdir())


def test_a_script_without_an_interpreter_line_is_refused(tmp_path):
    sched = scheduler(tmp_path)
    req = request(tmp_path)
    req.script.write_text("exit 0\n")
    with pytest.raises(Refused, match="#!"):
        sched.submit(req)


def test_cancel_of_a_pending_job_ends_it_at_once(tmp_path):
    sched = scheduler(tmp_path)
    gate = tmp_path / "gate"
    parent = sched.submit(request(tmp_path, f"while [ ! -e {gate} ]; do sleep 0.02; done", tag="p"))
    child = sched.submit(request(tmp_path, tag="c", after=parent))
    sched.cancel(child)
    assert _state(sched, child) is JobState.CANCELLED
    supervisor = record(sched, child).supervisor
    wait_for(lambda: supervisor.alive() is False, what="the supervisor's exit")
    gate.touch()
    _ended(sched, parent)
    assert record(sched, child).started_at is None
    sched.cancel(child)
    sched.cancel("424242")


def test_cancel_of_a_running_job_stops_its_script_and_releases(tmp_path):
    sched = scheduler(tmp_path, kill_wait_sec=5)
    req = request(tmp_path, "trap 'echo got-term; exit 0' TERM; while :; do sleep 0.02; done", gpus=2)
    job = sched.submit(req)
    wait_for(lambda: record(sched, job).runner is not None)
    sched.cancel(job)
    assert _ended(sched, job) is JobState.CANCELLED
    assert "got-term" in _output(req, job) and record(sched, job).exit_code == "0:0"
    assert StateDir(sched.state.root).reservations() == {}


def test_singleton_jobs_of_one_name_run_one_at_a_time(tmp_path):
    sched = scheduler(tmp_path)
    log = tmp_path / "log"
    body = f"echo start $HLS_JOB_ID >> {log}; sleep 0.3; echo end $HLS_JOB_ID >> {log}"
    jobs = [sched.submit(request(tmp_path, body, tag=f"s{i}", name="one", singleton=True)) for i in range(3)]
    other = sched.submit(request(tmp_path, f"echo other >> {log}", tag="o", name="two", singleton=True))
    for job in [*jobs, other]:
        assert _ended(sched, job) is JobState.COMPLETED
    lines = [line for line in log.read_text().splitlines() if line != "other"]
    assert lines == [f"{what} {job}" for job in jobs for what in ("start", "end")]


def test_two_jobs_never_share_a_gpu_or_a_core(tmp_path):
    sched = scheduler(tmp_path, inventory=FakeInventory(gpus=4, numa=2, cores_per_node=4, threads=2))
    gate = tmp_path / "gate"
    body = f"while [ ! -e {gate} ]; do sleep 0.02; done"
    a = sched.submit(request(tmp_path, body, tag="a", gpus=2, cpus=3))
    b = sched.submit(request(tmp_path, body, tag="b", gpus=2, cpus=4))
    c = sched.submit(request(tmp_path, body, tag="c", gpus=1, cpus=1))
    wait_for(lambda: _state(sched, a) is _state(sched, b) is JobState.RUNNING)
    held = StateDir(sched.state.root).reservations()
    assert set(held) == {a, b} and _state(sched, c) is JobState.PENDING
    ra, rb = held[a], held[b]
    assert not set(ra.gpus) & set(rb.gpus) and not set(ra.held_cpus) & set(rb.held_cpus)
    # Three CPUs of two-thread cores take two whole cores on one NUMA node; the fourth CPU idles.
    assert ra.cpus == (0, 1, 2) and ra.held_cpus == (0, 1, 2, 3)
    # Four CPUs fit in what node 0 has left, the fullest node that holds them.
    assert rb.cpus == rb.held_cpus == (4, 5, 6, 7)
    assert all(u.startswith("GPU-") for u in (*ra.gpus, *rb.gpus))
    gate.touch()
    for job in (a, b, c):
        assert _ended(sched, job) is JobState.COMPLETED


def test_a_large_job_is_not_starved_by_small_ones(tmp_path):
    sched = scheduler(tmp_path, inventory=FakeInventory(gpus=4))
    gate = tmp_path / "gate"
    body = f"while [ ! -e {gate} ]; do sleep 0.02; done"
    first = sched.submit(request(tmp_path, body, tag="first", gpus=2, time_sec=600))
    wait_for(lambda: _state(sched, first) is JobState.RUNNING)
    large = sched.submit(request(tmp_path, body, tag="large", gpus=4, time_sec=600))
    long_small = sched.submit(request(tmp_path, body, tag="long", gpus=1, time_sec=1200))
    short_small = sched.submit(request(tmp_path, "exit 0", tag="short", gpus=1, time_sec=60))
    # The short job ends before the large one can start, so it runs beside the first; the long
    # one would hold the large one back, so it waits behind it.
    assert _ended(sched, short_small) is JobState.COMPLETED
    infos = sched.states([large, long_small])
    assert infos[large].reason == "waiting for GPUs: 2 of the host's 4 H200 GPUs are free and it asks for 4"
    assert infos[long_small].reason == "Priority"
    gate.touch()
    for job in (first, large, long_small):
        assert _ended(sched, job) is JobState.COMPLETED
    assert record(sched, long_small).started_at >= record(sched, large).started_at


# Races


def test_two_submits_at_once_get_two_ids_and_never_both_hold_the_host(tmp_path):
    sched = scheduler(tmp_path, inventory=FakeInventory(gpus=4))
    gate = tmp_path / "gate"
    body = f"while [ ! -e {gate} ]; do sleep 0.02; done"
    ids: list[str] = []
    barrier = threading.Barrier(4)

    def submit(i: int) -> None:
        req = request(tmp_path, body, tag=f"r{i}", gpus=4)
        barrier.wait()
        ids.append(sched.submit(req))

    threads = [threading.Thread(target=submit, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 4
    wait_for(lambda: any(_state(sched, j) is JobState.RUNNING for j in ids))
    time.sleep(0.3)
    states = sched.states(ids)
    assert sum(info.state is JobState.RUNNING for info in states.values()) == 1
    assert len(StateDir(sched.state.root).reservations()) == 1
    gate.touch()
    for job in ids:
        assert _ended(sched, job) is JobState.COMPLETED


@pytest.mark.parametrize("delay", [0.0, 0.05, 0.15, 0.3, 0.5])
def test_a_cancel_during_start_ends_cancelled_and_leaves_nothing_running(tmp_path, delay):
    sched = scheduler(tmp_path, kill_wait_sec=1)
    marker = tmp_path / "ran"
    req = request(tmp_path, f"echo $$ > {marker}; exec sleep 30")
    job = sched.submit(req)
    time.sleep(delay)
    sched.cancel(job)
    assert _ended(sched, job) is JobState.CANCELLED
    rec = record(sched, job)
    wait_for(lambda: rec.supervisor.alive() is False, what="the supervisor's exit")
    rec = record(sched, job)
    assert rec.released and StateDir(sched.state.root).reservations() == {}
    if marker.exists():
        pid = int(marker.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_a_supervisor_killed_between_reserving_and_starting_releases_on_the_next_pass(tmp_path):
    sched = scheduler(tmp_path, inventory=FakeInventory(gpus=2))
    gate = tmp_path / "gate"
    first = sched.submit(request(tmp_path, f"while [ ! -e {gate} ]; do sleep 0.02; done", tag="a", gpus=2))
    wait_for(lambda: record(sched, first).runner is not None)
    # Rewind the record to the moment after the pass gave it the GPUs and before its script ran,
    # then kill its supervisor there.
    state = StateDir(sched.state.root)
    rec = record(sched, first)
    os.kill(rec.supervisor.pid, signal.SIGKILL)
    wait_for(lambda: rec.supervisor.alive() is False)
    wait_for(lambda: rec.runner.alive() is False, what="the script's parent-death kill")
    with state.locked():
        state.write(rec.model_copy(update={"runner": None}))
    assert _state(sched, first) is JobState.NODE_FAIL
    second = sched.submit(request(tmp_path, "exit 0", tag="b", gpus=2))
    assert _ended(sched, second) is JobState.COMPLETED
    rec = record(sched, first)
    assert rec.state == "NODE_FAIL" and rec.released and "supervisor" in rec.note


def test_a_supervisor_killed_while_its_script_runs_takes_the_script_and_its_children_with_it(tmp_path):
    sched = scheduler(tmp_path, inventory=FakeInventory(gpus=2))
    child_pid = tmp_path / "child"
    # The child leaves the script's process group and so escapes a group kill; its token does not.
    body = (f"setsid bash -c 'echo $$ > {child_pid}; exec sleep 60' & "
            "while :; do sleep 0.02; done")
    first = sched.submit(request(tmp_path, body, tag="a", gpus=2))
    wait_for(child_pid.exists)
    wait_for(lambda: child_pid.read_text().strip())
    rec = record(sched, first)
    os.kill(rec.supervisor.pid, signal.SIGKILL)
    assert _ended(sched, first) is JobState.NODE_FAIL
    # The polls reclaim it, with no other job on the host.
    wait_for(lambda: sched.states([first]) and record(sched, first).released, what="the reclaim")
    assert StateDir(sched.state.root).reservations() == {}
    second = sched.submit(request(tmp_path, "exit 0", tag="b", gpus=2))
    assert _ended(sched, second) is JobState.COMPLETED
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid.read_text()), 0)
    assert record(sched, first).released


def test_a_reclaim_whose_stopper_fails_keeps_the_reservation(tmp_path, monkeypatch):
    sched = scheduler(tmp_path, inventory=FakeInventory(gpus=2))
    first = sched.submit(request(tmp_path, "sleep 60", tag="a", gpus=2))
    wait_for(lambda: record(sched, first).runner is not None)
    rec = record(sched, first)
    os.kill(rec.supervisor.pid, signal.SIGKILL)
    wait_for(lambda: rec.supervisor.alive() is False)
    calls: list[str] = []

    class Failing:
        def stop(self, job_id: str) -> None:
            calls.append(job_id)
            raise RuntimeError("the daemon did not answer")

    state = StateDir(sched.state.root)
    now = time.time()
    with state.locked():
        run_pass(state, now=now, stoppers=lambda name, options: Failing())
    assert calls == [first] and set(state.reservations()) == {first}
    assert not record(sched, first).released
    # Something of it may still run, so it reads RUNNING, as Slurm's COMPLETING does.
    assert local.job_states(state, [first])[first].state is JobState.RUNNING
    with state.locked():
        run_pass(state, now=now + RECLAIM_BACKOFF_SEC - 1, stoppers=lambda name, options: Failing())
    assert calls == [first]
    with state.locked():
        run_pass(state, now=now + RECLAIM_BACKOFF_SEC, stoppers=stopper)
    assert state.reservations() == {} and record(sched, first).released
    assert local.job_states(state, [first])[first].state is JobState.NODE_FAIL


def test_a_cancel_whose_supervisor_is_already_gone_still_reads_cancelled(tmp_path):
    sched = scheduler(tmp_path)
    job = sched.submit(request(tmp_path, "sleep 60"))
    wait_for(lambda: record(sched, job).runner is not None)
    rec = record(sched, job)
    os.kill(rec.supervisor.pid, signal.SIGSTOP)
    try:
        sched.cancel(job)
        os.kill(rec.supervisor.pid, signal.SIGKILL)
        wait_for(lambda: rec.supervisor.alive() is False)
    finally:
        try:
            os.kill(rec.supervisor.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
    assert _state(sched, job) is JobState.CANCELLED


def test_no_secret_reaches_a_record_or_a_command_line(tmp_path, monkeypatch):
    secret = "sk-test-0123456789abcdef"
    monkeypatch.setenv("SOME_API_KEY", secret)
    sched = scheduler(tmp_path)
    gate = tmp_path / "gate"
    req = request(tmp_path, f"while [ ! -e {gate} ]; do sleep 0.02; done; cat /proc/$$/environ | tr '\\0' '\\n'")
    assert secret not in " ".join(sched.submit_argv(req))
    job = sched.submit(req)
    wait_for(lambda: record(sched, job).runner is not None)
    rec = record(sched, job)
    for pid in (rec.supervisor.pid, rec.runner.pid):
        assert secret.encode() not in Path(f"/proc/{pid}/environ").read_bytes()
        assert secret.encode() not in Path(f"/proc/{pid}/cmdline").read_bytes()
    gate.touch()
    _ended(sched, job)
    assert secret not in _output(req, job)
    for path in sched.state.root.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


def test_the_state_directory_must_be_this_users_alone(tmp_path):
    root = tmp_path / "state"
    root.mkdir(mode=0o755)
    os.chmod(root, 0o755)
    with pytest.raises(local.StateError):
        scheduler(tmp_path)


def test_a_record_holds_what_the_brief_asks(tmp_path):
    sched = scheduler(tmp_path)
    req = request(tmp_path, "exit 0", tag="hls:x/0/1", name="hls-x", gpus=1, cpus=2, memory_gib=3, time_sec=90)
    job = sched.submit(req)
    _ended(sched, job)
    raw = json.loads((sched.state.root / "ended" / f"{job}.json").read_text())
    assert raw["job_id"] == job and raw["tag"] == "hls:x/0/1" and raw["name"] == "hls-x"
    assert raw["shape"]["gpus"] == 1 and raw["shape"]["gpu_kind"] == "H200" and raw["shape"]["time_sec"] == 90
    assert raw["reservation"]["gpus"] == ["GPU-00000000-fake"] and raw["reservation"]["memory_gib"] == 3
    assert raw["supervisor"]["pid"] > 0 and raw["end_time"] == raw["started_at"] + 90
    assert raw["state"] == "COMPLETED" and raw["exit_code"] == "0:0" and raw["after"] is None
    assert isinstance(ProcessId.model_validate(raw["runner"]), ProcessId)


def test_a_process_that_left_the_group_and_the_environment_is_killed_at_the_end(tmp_path):
    sched = scheduler(tmp_path)
    pid_file = tmp_path / "escaped"
    body = (f"setsid env -i /bin/sh -c 'echo $$ > {pid_file}; exec sleep 300' </dev/null >/dev/null 2>&1 & "
            f"while [ ! -s {pid_file} ]; do sleep 0.02; done; exit 0")
    job = sched.submit(request(tmp_path, body))
    assert _ended(sched, job) is JobState.COMPLETED
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
    assert record(sched, job).released
