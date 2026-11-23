from __future__ import annotations

import threading
import time

import pytest

import hls.substrate.runner as runner_mod
from hls.substrate.base import SchedulerUnavailable
from hls.substrate.runner import EXIT_FAILED, EXIT_PUBLISHED, EXIT_TERMINATED, JobEnd
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.pipeline.test_runner import (OTHER, _events, _main, _no_attempts, _pin, _record_runners, _runner, _status,
                                        _store, _sub)
from tests.substrate.slurm_fakes import FakeRun, done


class _Controller:
    """Answers ``job_not_ready`` from a script, then says the job is ready; notes when the
    first step was started relative to the answers."""

    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.asked = 0

    def __call__(self):
        self.asked += 1
        if not self.answers:
            return None
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def _run_within(runner, seconds: float) -> int:
    """``runner.run()``, failing the test instead of hanging when it has not returned in
    *seconds*: a wait that never gives up is then ended as a SIGTERM ends it."""
    result: list[int] = []
    thread = threading.Thread(target=lambda: result.append(runner.run()), daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        runner.terminate()
        thread.join(30)
        pytest.fail(f"the runner was still waiting after {seconds}s")
    return result[0]


class _Watching(LocalLauncher):
    """Records how often the controller had been asked when each role started."""

    def __init__(self, controller: _Controller) -> None:
        super().__init__(gpus=2)
        self.controller = controller
        self.asked_at_launch: list[int] = []

    def launch(self, spec, *, share_gpus_of=None):
        self.asked_at_launch.append(self.controller.asked)
        return super().launch(spec, share_gpus_of=share_gpus_of)


def test_the_first_step_waits_until_the_job_has_left_its_prolog(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "PROLOG_POLL_SEC", 0.01)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = _Controller("Reason=Prolog", "Reason=Prolog", "JobState=CONFIGURING")
    launcher = _Watching(controller)
    assert _runner(plan, launcher=launcher, job_not_ready=controller).run() == EXIT_PUBLISHED
    assert controller.asked == 4 and min(launcher.asked_at_launch) == 4
    kinds = [e for e in _events(plan) if e["kind"] in ("job_not_ready", "job_ready")]
    assert [(e["kind"], e.get("why")) for e in kinds] == [("job_not_ready", "Reason=Prolog"), ("job_ready", None)]


def test_a_job_ready_at_once_is_asked_once(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = _Controller()
    assert _runner(plan, job_not_ready=controller).run() == EXIT_PUBLISHED
    assert controller.asked == 1
    assert not [e for e in _events(plan) if e["kind"] in ("job_not_ready", "job_ready", "job_state_unknown")]


def test_a_controller_that_does_not_answer_is_not_waited_for(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = _Controller(SchedulerUnavailable("scontrol timed out"))
    assert _runner(plan, job_not_ready=controller).run() == EXIT_PUBLISHED
    [event] = [e for e in _events(plan) if e["kind"] == "job_state_unknown"]
    assert event["error"] == "scontrol timed out" and controller.asked == 1


def test_a_job_still_in_its_prolog_when_the_wait_ends_starts_no_step(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "PROLOG_POLL_SEC", 0.01)
    monkeypatch.setattr(runner_mod, "PROLOG_WAIT_SEC", 0.2)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    launcher = LocalLauncher(gpus=2)
    assert _run_within(_runner(plan, launcher=launcher, job_not_ready=lambda: "Reason=Prolog"), 30) == EXIT_FAILED
    fault = _status(plan).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "still Reason=Prolog" in fault.reason
    assert launcher.launched == [] and _no_attempts(plan, "noop_a")


def test_the_controller_is_asked_every_poll_interval_not_in_a_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "PROLOG_POLL_SEC", 0.05)
    monkeypatch.setattr(runner_mod, "PROLOG_WAIT_SEC", 0.5)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = _Controller(*["Reason=Prolog"] * 10_000)
    assert _run_within(_runner(plan, job_not_ready=controller), 30) == EXIT_FAILED
    assert 5 <= controller.asked <= 12


def test_a_wall_clock_step_during_the_wait_does_not_cut_it_short(tmp_path, monkeypatch):
    # The wait is timed on the monotonic clock: a wall clock stepped forward past the whole
    # wait, as a time sync can, leaves the job its time to leave the Prolog.
    monkeypatch.setattr(runner_mod, "PROLOG_POLL_SEC", 0.01)
    monkeypatch.setattr(runner_mod, "PROLOG_WAIT_SEC", 30.0)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    step = [0.0]

    def controller():
        step[0] += 3600.0
        return "Reason=Prolog" if step[0] <= 2 * 3600.0 else None

    runner = _runner(plan, job_not_ready=controller, clock=lambda: time.time() + step[0],
                     job_end=JobEnd(controller=time.time() + 86400 * 2, env=None))
    assert _run_within(runner, 60) == EXIT_PUBLISHED
    [ready] = [e for e in _events(plan) if e["kind"] == "job_ready"]
    assert ready["waited_sec"] < 30


def test_a_scontrol_that_cannot_be_run_is_not_waited_for(tmp_path):
    # SlurmCommands raises FileNotFoundError for a node without scontrol, as read_job_end takes it.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = _Controller(FileNotFoundError("scontrol"))
    assert _runner(plan, job_not_ready=controller).run() == EXIT_PUBLISHED
    [event] = [e for e in _events(plan) if e["kind"] == "job_state_unknown"]
    assert event["error"] == "scontrol" and controller.asked == 1


def test_a_failing_pinned_input_does_not_wait_behind_the_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "PROLOG_POLL_SEC", 0.01)
    monkeypatch.setattr(runner_mod, "PROLOG_WAIT_SEC", 0.2)
    plan = _pin(noop_plan(tmp_path), {OTHER: "0" * 64})
    plan = _store(plan, _sub(plan))
    controller = _Controller("Reason=Prolog")
    assert _runner(plan, job_not_ready=controller).run() == EXIT_FAILED
    assert "not published" in _status(plan).fault.reason
    assert controller.asked == 0 and _no_attempts(plan, "noop_a")


def test_a_sigterm_during_the_wait_ends_it(tmp_path, monkeypatch):
    monkeypatch.setattr(runner_mod, "PROLOG_POLL_SEC", 0.01)
    monkeypatch.setattr(runner_mod, "PROLOG_WAIT_SEC", 30.0)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    launcher = LocalLauncher(gpus=2)
    runner = _runner(plan, launcher=launcher, job_not_ready=lambda: "Reason=Prolog")
    threading.Timer(0.3, runner.terminate).start()
    started = time.monotonic()
    assert runner.run() == EXIT_TERMINATED
    assert time.monotonic() - started < 10
    assert launcher.launched == [] and _status(plan).state == "terminated"


def test_the_runners_entry_point_asks_the_controller_about_its_own_job(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    made = _record_runners(monkeypatch)
    scontrol = FakeRun().on("scontrol", done("JobId=100 JobName=x\n   JobState=RUNNING Reason=Prolog Dependency=(null)\n"))
    assert _main(plan, monkeypatch, lambda: "/job_100/step_batch", scontrol=scontrol) == 0
    [kwargs] = made
    assert kwargs["job_not_ready"]() == "Reason=Prolog"
    assert scontrol.calls == [["scontrol", "show", "job", "100"]]


def test_a_runner_with_every_stage_published_asks_no_controller(tmp_path, monkeypatch):
    # It starts no step, so the Prolog cannot cost it one.
    monkeypatch.setattr(runner_mod, "PROLOG_POLL_SEC", 0.01)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert _runner(plan).run() == EXIT_PUBLISHED
    plan = _store(plan, _sub(plan).model_copy(update={"status": "absent"}), _sub(plan, 2, job_id="101"))
    controller = _Controller("Reason=Prolog")
    assert _runner(plan, n=2, job_id="101", job_not_ready=controller).run() == EXIT_PUBLISHED
    assert controller.asked == 0
