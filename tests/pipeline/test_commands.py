from __future__ import annotations

import dataclasses
import re
import sys
import textwrap
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from hls import cli
from hls.pipeline import commands
from hls.pipeline import follow as follow_mod
from hls.pipeline.events import read_events
from hls.pipeline.orchestrator import WITHHELD, Orchestrator
from hls.pipeline.plan import AllocationStatus, AttemptStatus, Outcome, Plan, PlanStore, Toolchain
from hls.settings import SettingsError
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder, instance_lease
from hls.store.posix import PosixStore
from hls.substrate.base import JobState, Refused
from hls.substrate.slurm.scheduler import submit_argv
from tests.pipeline.plans import CODE, noop_plan
from tests.pipeline.test_end_to_end_local import LocalSlurm, bounded_sleep, contestant_fault_in


def _config(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "hls.toml"
    path.write_text(textwrap.dedent(f"""
        [site]
        partition = "gpu"
        account = "acct"
        store_root = "{tmp_path / 'store'}"
        node_scratch = "{tmp_path / 'node'}"
        secrets_path = "/x"
        [run]
        heartbeat_sec = 1
        lease_ttl_sec = 30
        poll_sec = 1
        allocation_overhead_sec = 360
        allocation_dead_after_sec = 10
        """))
    return path


def test_only_site_and_run_overrides_are_accepted():
    assert commands.settings_overrides({"run.gpu_policy": "split"}) == {"run.gpu_policy": "split"}
    with pytest.raises(ValueError, match="substrate.apptainer.x, task.y"):
        commands.settings_overrides({"task.y": "1", "substrate.apptainer.x": "2"})


def test_checkout_root_is_the_repository_holding_this_package():
    assert (commands.checkout_root() / "src" / "hls" / "__init__.py").exists()


def test_start_drives_a_planned_instance_to_its_end(tmp_path, monkeypatch):
    code, lines, slurm, _ = _local_start(tmp_path, monkeypatch, two_stage=True, overrides={"run.gpu_policy": "max"})
    assert code == 0
    assert lines[0].startswith("instance runs/noop/")
    # 26 min of stages and overhead, and 12.5 min for their 5 roles' steps to start.
    assert lines[1] == "allocation 0: noop_a, noop_b; 2 x H200, 8 CPUs, 12 GiB, 38 min"
    assert lines[2].startswith("warning: allocation 0 holds 2 GPUs; stage noop_a uses 1")
    assert lines[-1].endswith(": succeeded: every stage published")
    assert "--job-name=hls-test-local" in submit_argv(slurm.jobs["100"].request)


class SimTime:
    """Time that moves only when something sleeps, so a command that waits out a lease's TTL
    takes no real time. ``on_sleep`` runs at each sleep, as a live orchestrator renewing its
    lease would. More than *calls* sleeps fail the test."""

    def __init__(self, calls: int = 50) -> None:
        self.now = time.time()
        self.calls = calls
        self.slept: list[float] = []
        self.on_sleep: list[Callable[[], None]] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        assert len(self.slept) <= self.calls, f"slept more than {self.calls} times"
        self.now += seconds
        for f in self.on_sleep:
            f()


LOGIN = Holder(host="login", pid=1, job_id=None, purpose="orchestrator")


def _instance(tmp_path: Path, slurm: LocalSlurm, *, clock: Callable[[], float] = time.time,
              holder: Holder = LOGIN) -> Orchestrator:
    plan = noop_plan(tmp_path, gpu_policy="split")
    orch = Orchestrator.create(plan, slurm, holder=holder, clock=clock, sleep=lambda s: None)
    orch.step()
    orch.step()
    return orch


def test_status_gives_the_local_schedulers_reason_a_job_waits(tmp_path, monkeypatch):
    """The local scheduler says in words why a job waits, such as how many GPUs are free, and
    ``hls status`` gives it; Slurm's line, whose reasons are codes, stays as it was."""
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    states = slurm.states
    why = "waiting for GPUs: 0 of the host's 8 H200 GPUs are free and it asks for 1; another user computes on 8"

    def waiting(ids):
        return {j: dataclasses.replace(info, reason=why) for j, info in states(ids).items()}

    monkeypatch.setattr(slurm, "states", waiting)
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert "  submission 1: job 100, submitted, Slurm PENDING\n" in text + "\n" and why not in text
    monkeypatch.setitem(commands._STATUS_SCHEDULERS, "slurm", ("local scheduler", True))
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert f"  submission 1: job 100, submitted, local scheduler PENDING ({why})" in text


def test_status_shows_the_plan_the_jobs_and_the_stages(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert "instance runs/noop/r1 (pipeline noop2), plan revision" in text
    assert "outcome: still running" in text
    assert "lease: generation 1, orchestrator on login pid 1" in text
    assert "  submission 1: job 100, submitted, Slurm PENDING" in text
    assert "  submission 1: job 101, submitted, Slurm PENDING, after job 100" in text
    assert "stage noop_a: not published; attempts: none" in text


def _killed_mid_stage(orch: Orchestrator, slurm: LocalSlurm, job_id: str, *, k: int = 0) -> None:
    """Leave the statuses a runner of *job_id* writes in stage noop_a when it is SIGKILLed, and end its job."""
    plan = orch.plans.latest()
    store = orch.store
    attempt = plan.layout.claim_attempt(store, "noop_a")
    store.write_bytes(plan.layout.attempt_file("noop_a", attempt, "status.json"), AttemptStatus(
        stage="noop_a", attempt=attempt, allocation=k, submission=1, job_id=job_id, state="running", started_at=1.0,
        ended_at=None, fault=None, manifest_sha256=None, no_retry=None).model_dump_json().encode())
    store.write_bytes(plan.layout.allocation_status(k, 1), AllocationStatus(
        instance=plan.instance, allocation=k, submission=1, job_id=job_id, node="node06", state="running",
        started_at=1.0, heartbeat_at=1.0, end_time=None, stage="noop_a", attempt=attempt, grant=None,
        tool_versions=None, swap_max_bytes=None, memory_enforced_by=None, fault=None,
        ended_at=None).model_dump_json().encode())
    slurm.start(job_id, at=1.0)
    slurm.finish(job_id, JobState.FAILED)


def test_status_says_a_runner_and_attempt_still_marked_running_are_abandoned_once_the_job_ended(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    _killed_mid_stage(orch, slurm, "100")
    before = sorted(p.name for p in (tmp_path / "store").rglob("*"))
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert "  submission 1: job 100, submitted, Slurm FAILED; runner abandoned (job 100 ended FAILED) on node06 " \
           "in stage noop_a, heartbeat " in text
    assert "stage noop_a: not published; attempts: 1 abandoned (job 100 ended FAILED)" in text
    assert sorted(p.name for p in (tmp_path / "store").rglob("*")) == before


def test_status_asks_about_the_job_of_an_attempt_whose_submission_has_no_job_id(tmp_path):
    # The sbatch answer was lost, so the plan has no job id, and the job ran and died meanwhile.
    slurm = LocalSlurm()
    slurm.submit_fault = "lose_answer"
    orch = Orchestrator.create(noop_plan(tmp_path), slurm, holder=LOGIN, sleep=lambda s: None)
    orch.step()
    assert orch.plans.latest().allocation(0).submissions[0].job_id is None
    _killed_mid_stage(orch, slurm, "100")
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert "attempts: 1 abandoned (job 100 ended FAILED)" in text


def test_cancel_without_a_live_orchestrator_takes_the_lease_and_cancels_by_id(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    orch.handle.release()
    lines: list[str] = []
    assert commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep()) == 0
    assert sorted(slurm.cancels()) == ["100", "101"]
    assert slurm.jobs["100"].state is JobState.CANCELLED
    assert PlanStore(orch.store, orch.layout).latest().outcome.status == "cancelled"
    assert lines[-1].endswith("cancelled: cancelled by hls cancel")


def test_cancel_takes_over_without_a_usable_secrets_file_and_records_no_scheduler_answer(tmp_path, caplog):
    """An orphaned instance's jobs are cancelled even when its secrets file cannot be read, as after
    a chmod by mistake. The orchestrator that cancels could scrub nothing, so it records job ids and
    states, and any answer of the scheduler only as withheld."""
    secrets = tmp_path / "secrets.toml"
    secrets.write_text('[secrets]\nnamed = "hls-noop"\n')
    secrets.chmod(0o600)
    slurm = LocalSlurm()
    orch = Orchestrator.create(noop_plan(tmp_path, gpu_policy="split", site={"secrets_path": str(secrets)}), slurm,
                               holder=LOGIN, sleep=lambda s: None)
    orch.step()
    orch.step()
    orch.handle.release()
    secrets.chmod(0o644)
    answer = "scancel: error: Kill job error on job id 100: the answer of a cancel"
    real, refused = slurm.cancel, set()

    def refused_once(job_id):
        if job_id not in refused:
            refused.add(job_id)
            raise Refused(answer)
        real(job_id)

    slurm.cancel = refused_once
    lines: list[str] = []
    assert commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep()) == 0
    assert sorted(slurm.cancels()) == ["100", "101"] and refused == {"100", "101"}
    assert PlanStore(orch.store, orch.layout).latest().outcome.status == "cancelled"
    records = [p.read_text() for p in orch.store.path(orch.layout.root).rglob("*") if p.is_file()]
    assert not any("the answer of a cancel" in text for text in records)
    assert any(WITHHELD in text for text in records)
    assert "secrets.toml" in caplog.text


def test_cancel_with_a_live_orchestrator_waits_for_it_to_cancel(tmp_path):
    slurm = LocalSlurm()
    orch = Orchestrator.create(noop_plan(tmp_path, gpu_policy="split"), slurm,
                               holder=Holder(host="login", pid=1, job_id=None, purpose="orchestrator"),
                               sleep=lambda s: time.sleep(0.05))
    orch.step()
    orch.step()
    live = threading.Thread(target=orch.run)
    live.start()
    lines: list[str] = []
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep())
    live.join(30)
    assert code == 0 and not live.is_alive()
    assert PosixStore(orch.store.root).exists(InstanceLayout("runs/noop/r1").cancel_request)
    assert sorted(slurm.cancels()) == ["100", "101"]
    assert PlanStore(orch.store, orch.layout).latest().outcome.status == "cancelled"
    # Either the live orchestrator was seen finishing, or the lease freed first and this command took it.
    assert lines[-1].endswith(": cancelled by hls cancel")


def test_cancel_names_the_live_orchestrator_that_carried_it_out(tmp_path):
    sim = SimTime()
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm, clock=sim.clock)

    def one_more_pass():
        _, done, _ = orch.step()
        if done:
            orch.handle.release()

    # Each wait of the command gives the live orchestrator one pass; it sees the request.
    sim.on_sleep.append(one_more_pass)
    lines: list[str] = []
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 0 and sorted(slurm.cancels()) == ["100", "101"]
    assert re.fullmatch(r"waiting for orchestrator on login pid 1 \(heartbeat \d+ s ago\), up to 32 s", lines[0])
    assert lines[1:] == ["runs/noop/r1 cancelled by orchestrator on login pid 1: cancelled by hls cancel"]


def test_cancel_names_the_orchestrator_that_held_the_lease_when_it_was_carried_out(tmp_path):
    sim = SimTime()
    slurm = LocalSlurm()
    first = _instance(tmp_path, slurm, clock=sim.clock)
    second: list[Orchestrator] = []

    def hand_over_then_cancel():
        if not second:
            first.handle.release()
            second.append(Orchestrator.attach(
                first.store, first.layout, slurm, holder=Holder(host="second", pid=2, job_id=None,
                                                                purpose="orchestrator"),
                clock=sim.clock, sleep=lambda s: None))
            return
        _, done, _ = second[0].step()
        if done:
            second[0].handle.release()

    sim.on_sleep.append(hand_over_then_cancel)
    lines: list[str] = []
    code = commands.cancel(first.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 0
    assert lines[-1] == "runs/noop/r1 cancelled by orchestrator on second pid 2: cancelled by hls cancel"


def test_cancel_of_a_cancelled_instance_whose_job_ends_while_it_looks_says_it_had_ended(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    orch.plans.write(_ended(orch.plans.latest(), "cancelled", **LIVE), orch.handle)
    orch.handle.release()
    slurm.start("100", at=time.time())
    states = slurm.states
    asked: list[int] = []

    def job_ends_after_the_first_look(job_ids):
        asked.append(1)
        if len(asked) > 1:
            slurm.jobs["100"].state = JobState.CANCELLED
        return states(job_ids)

    slurm.states = job_ends_after_the_first_look
    lines: list[str] = []
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep())
    assert code == 0 and lines == ["runs/noop/r1 already ended: cancelled: x"]
    assert PosixStore(orch.store.root).exists(InstanceLayout("runs/noop/r1").cancel_request)


def test_cancel_says_so_when_a_live_orchestrator_does_not_confirm(tmp_path):
    sim = SimTime()
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm, clock=sim.clock)
    # The holder stays alive and never acts on the request.
    sim.on_sleep.append(orch.handle.renew)
    lines: list[str] = []
    start = sim.now
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 3
    assert re.fullmatch(r"cancel requested, not confirmed: orchestrator on login pid 1 still holds the lease "
                        r"\(heartbeat \d+ s ago\)", lines[-1])
    # It says once that it waits, and for how long.
    assert len(lines) == 2 and re.fullmatch(
        r"waiting for orchestrator on login pid 1 \(heartbeat \d+ s ago\), up to 32 s", lines[0])
    # It waits the lease TTL (30 s) and two polls (1 s each) before it gives up.
    assert 32 <= sim.now - start < 34
    assert PosixStore(orch.store.root).exists(InstanceLayout("runs/noop/r1").cancel_request)
    assert slurm.cancels() == []


def test_cancel_of_a_dead_orchestrators_lease_waits_it_out_then_cancels(tmp_path):
    sim = SimTime()
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm, clock=sim.clock)
    lines: list[str] = []
    start = sim.now
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 0 and sorted(slurm.cancels()) == ["100", "101"]
    assert sim.now - start >= 30
    assert lines[-1].endswith("cancelled: cancelled by hls cancel")


def test_resume_gives_up_on_a_holder_that_keeps_its_lease(tmp_path):
    sim = SimTime()
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm, clock=sim.clock)
    sim.on_sleep.append(orch.handle.renew)
    lines: list[str] = []
    start = sim.now
    code = commands.resume(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 2
    assert re.fullmatch(r"runs/noop/r1 is held by orchestrator on login pid 1 \(heartbeat \d+ s ago\); "
                        r"not resumed", lines[-1])
    assert 30 <= sim.now - start < 32
    assert PlanStore(orch.store, orch.layout).latest().lease_generation == 1


def test_resume_gives_up_at_once_on_a_holder_that_is_alive_on_this_host(tmp_path):
    sim = SimTime()
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm, clock=sim.clock, holder=Holder.here("orchestrator"))
    lines: list[str] = []
    code = commands.resume(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 2 and sim.slept == []
    assert len(lines) == 1 and lines[0].startswith("runs/noop/r1 is held by orchestrator on ")
    assert lines[0].endswith("; not resumed")


def test_cancel_keeps_waiting_on_a_holder_that_is_alive_on_this_host(tmp_path):
    sim = SimTime()
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm, clock=sim.clock, holder=Holder.here("orchestrator"))
    sim.on_sleep.append(orch.handle.renew)
    lines: list[str] = []
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 3 and len(sim.slept) >= 30
    assert lines[0].startswith("waiting for orchestrator on ") and "not confirmed" in lines[-1]


def test_resume_waits_out_a_dead_holders_lease_and_takes_over(tmp_path, monkeypatch):
    sim = SimTime()
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm, clock=sim.clock)
    states = slurm.states

    def states_after_the_queue_runs(job_ids):
        slurm.tick()
        return states(job_ids)

    monkeypatch.setattr(slurm, "states", states_after_the_queue_runs)
    lines: list[str] = []
    start = sim.now
    code = commands.resume(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=sim.sleep, clock=sim.clock)
    assert code == 0 and lines[1] == "attached to runs/noop/r1 as lease generation 2"
    assert re.fullmatch(r"waiting for orchestrator on login pid 1 \(heartbeat \d+ s ago\), up to 30 s", lines[0])
    assert sim.now - start >= 30
    assert len(slurm.submits()) == 2


def test_the_cli_routes_to_the_commands(tmp_path, monkeypatch):
    # configure_logging turns propagation off on the hls loggers, which would break caplog in
    # every later test of the session.
    monkeypatch.setattr(cli, "configure_logging", lambda verbose=False: None)
    seen = {}
    started = _started(monkeypatch, 0)
    monkeypatch.setattr(commands, "plan_provision", lambda **kw: seen.update(kw) or "provision plan")
    # Overrides follow the task module, which is the one positional argument.
    result = CliRunner().invoke(cli.app, ["provision", "tasks.multitenant_slo.task", "--rows-only", "--job-name",
                                          "warm-x", "--config", str(tmp_path / "hls.toml"), "--site.partition", "l40"])
    assert result.exit_code == 0, result.output
    assert {k: v for k, v in seen.items() if k != "echo"} == {
        "config": tmp_path / "hls.toml", "overrides": {"site.partition": "l40"},
        "task_module": "tasks.multitenant_slo.task", "weights": False, "job_name": "warm-x"}
    assert started == [("provision plan", {"site.partition": "l40"})]
    ran = {}
    monkeypatch.setattr("hls.substrate.runner.main", lambda argv: ran.setdefault("argv", argv) and 0)
    result = CliRunner().invoke(cli.app, ["runner", "/s/runs/noop/r1/plan", "--allocation", "0", "--submission", "2"])
    assert ran["argv"] == ["/s/runs/noop/r1/plan", "--allocation", "0", "--submission", "2"]


def _started(monkeypatch, code: int) -> list[tuple[object, dict[str, str]]]:
    """Every plan the CLI hands ``commands.start``, with its overrides; each start returns *code*."""
    started: list[tuple[object, dict[str, str]]] = []
    monkeypatch.setattr(commands, "start", lambda plan, overrides, **kw: started.append((plan, overrides)) or code)
    return started


def test_an_error_while_a_planned_instance_is_driven_keeps_its_traceback(tmp_path, monkeypatch, capsys):
    # A ValueError of the drive is a bug to debug, not a refusal of what was asked.
    def broken(*args, **kwargs):
        raise ValueError("an orchestrator bug")

    monkeypatch.setattr(commands.Orchestrator, "create", broken)
    with pytest.raises(ValueError, match="^an orchestrator bug$"):
        cli._plan_and_start("some-command", lambda: noop_plan(tmp_path), {})
    assert "hls some-command" not in capsys.readouterr().err


def _local_start(tmp_path, monkeypatch, *, overrides: Mapping[str, str] | None = None, stage_time_sec: int = 600,
                 store_root: Path | None = None, **pipeline: object):
    """A no-op instance planned by ``commands.plan_noop`` and driven by ``commands.start`` over the
    local Slurm, which runs each job as soon as it is submitted, each role sleeping 0 s and each
    stage given *stage_time_sec*. *pipeline* are ``plan_noop``'s other arguments, over the
    one-stage pipeline on one GPU; *overrides* are the plan's, and with *store_root* the store is
    there by one. Returns the exit status, the printed lines, the scheduler and the final plan."""
    monkeypatch.setattr(commands, "code_state", lambda path: CODE)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    slurm = LocalSlurm()
    submit = slurm.submit

    def submit_and_run(request):
        job_id = submit(request)
        slurm.tick()
        return job_id

    monkeypatch.setattr(slurm, "submit", submit_and_run)
    overrides = {**(overrides or {}), **({"site.store_root": str(store_root)} if store_root is not None else {})}
    planned = commands.plan_noop(**{
        "config": _config(tmp_path), "overrides": overrides, "two_stage": False, "gpus": 1, "gpu_kind": "H200",
        "sleep_sec": 0, "stage_time_sec": stage_time_sec, "fail_attempts": 0, "job_name": "hls-test-local",
        **pipeline})
    lines: list[str] = []
    code = commands.start(planned, overrides, scheduler=slurm, echo=lines.append)
    plan = PlanStore(PosixStore(store_root or tmp_path / "store"),
                     InstanceLayout(lines[0].removeprefix("instance "))).latest()
    return code, lines, slurm, plan


def test_planning_a_noop_refuses_an_override_outside_site_and_run(tmp_path):
    with pytest.raises(ValueError,
                       match=r"takes only --site\.\*, --run\.\* and --resources\.\* overrides, not task\.y"):
        commands.plan_noop(config=_config(tmp_path), overrides={"task.y": "1"}, two_stage=False, gpus=1,
                           gpu_kind="H200", sleep_sec=0, stage_time_sec=600, fail_attempts=0, job_name="x")


def test_a_one_stage_noop_plan_carries_every_argument_and_override(tmp_path, monkeypatch):
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, gpus=2, fail_attempts=1,
                                            overrides={"run.poll_sec": "2"}, stage_time_sec=300)
    stage = plan.stage("noop")
    assert code == 0 and plan.pipeline == "noop" and plan.settings.run.poll_sec == 2
    assert (stage.shape.gpus, stage.shape.time_sec) == (2, 300)
    assert stage.params["fail_attempts"] == 1
    assert not stage.params["contestant_role"] and stage.params["batch_hog_gib"] == 0
    assert plan.layout.attempts(PosixStore(plan.store_root), "noop") == [1, 2]


def test_a_two_stage_noop_plan_names_its_pipeline_and_fails_noop_a_on_purpose(tmp_path, monkeypatch):
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, two_stage=True, fail_attempts=1,
                                            stage_time_sec=450)
    assert code == 0 and plan.pipeline == "noop2" and plan.stage("noop_a").shape.time_sec == 450
    assert plan.layout.attempts(PosixStore(plan.store_root), "noop_a") == [1, 2]


def test_start_exits_1_when_the_instance_does_not_succeed(tmp_path, monkeypatch):
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, two_stage=True, fail_attempts=9)
    assert code == 1 and plan.outcome.status == "void" and ": void: " in lines[-1]


def test_an_instance_that_failed_on_a_contestant_fault_exits_1_and_says_whose_fault(tmp_path, monkeypatch):
    contestant_fault_in("noop_a", monkeypatch)
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, two_stage=True)
    assert code == 1 and plan.outcome.status == "failed"
    assert lines[-1].endswith(": failed: noop_a: contestant role server ran out of memory")
    text = commands.status_text(plan.instance, config=_config(tmp_path), scheduler=slurm)
    assert ("outcome: failed (a contestant fault: scored, never relaunched) at "
            in text) and text.split("outcome: ")[1].splitlines()[0].endswith(
        ": noop_a: contestant role server ran out of memory")
    assert "  failed at stage noop_a, a contestant fault: contestant role server ran out of memory" in text
    said: list[str] = []
    assert commands.resume(plan.instance, config=_config(tmp_path), scheduler=slurm, echo=said.append,
                           sleep=bounded_sleep()) == 1
    assert said == [f"{plan.instance} already ended: failed: noop_a: contestant role server ran out of memory"]


def test_status_of_a_finished_instance_reports_warnings_failures_attempts_and_runners(tmp_path, monkeypatch):
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, two_stage=True, fail_attempts=1)
    text = commands.status_text(plan.instance, config=_config(tmp_path), scheduler=slurm)
    assert "outcome: succeeded at " in text and text.split("outcome: ")[1].splitlines()[0].endswith(
        ": every stage published")
    assert ", released" in text.split("lease: ")[1].splitlines()[0]
    assert "warning: allocation 0 holds 2 GPUs; stage noop_a uses 1" in text
    assert "retry of allocation 0" in text and "  failed at stage noop_a, an infra fault: " in text
    assert "runner failed on local" in text and "runner succeeded on local" in text
    assert "Slurm COMPLETED" in text
    assert "stage noop_a: published; attempts: 1 failed (" in text and ", 2 succeeded" in text
    assert "stage noop_b: published; attempts: 1 succeeded" in text


def test_status_shows_the_last_error_of_a_submission(tmp_path):
    slurm = LocalSlurm()
    slurm.submit_fault = "lose_answer"
    orch = Orchestrator.create(noop_plan(tmp_path), slurm,
                               holder=Holder(host="login", pid=1, job_id=None, purpose="orchestrator"),
                               sleep=lambda s: None)
    orch.step()
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert "last error: sbatch timed out" in text


def test_resume_takes_a_dead_orchestrators_instance_to_its_end_without_resubmitting(tmp_path, monkeypatch):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    orch.handle.release()
    states = slurm.states

    def states_after_the_queue_runs(job_ids):
        slurm.tick()
        return states(job_ids)

    monkeypatch.setattr(slurm, "states", states_after_the_queue_runs)
    lines: list[str] = []
    assert commands.resume(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep()) == 0
    assert lines[0] == "attached to runs/noop/r1 as lease generation 2"
    assert lines[-1].endswith(": succeeded: every stage published")
    assert len(slurm.submits()) == 2
    lease = instance_lease(orch.store, orch.layout, ttl_sec=30).current()
    assert lease.generation == 2 and lease.holder.purpose == "orchestrator" and lease.released


def test_a_planning_command_exits_with_the_status_its_start_returns():
    started = []
    with pytest.raises(typer.Exit) as ended:
        cli._plan_and_start("some-command", lambda: "a plan", {"site.account": "other"},
                            start=lambda plan, overrides, echo: started.append((plan, overrides)) or 1)
    assert ended.value.exit_code == 1 and started == [("a plan", {"site.account": "other"})]


def test_a_refusal_while_planning_is_one_line_with_no_traceback(tmp_path, capsys):
    def plan() -> Plan:
        return commands.plan_noop(config=tmp_path / "hls.toml", overrides={"task.y": "1"}, two_stage=False, gpus=1,
                                  gpu_kind="H200", sleep_sec=0, stage_time_sec=600, fail_attempts=0, job_name="x")

    with pytest.raises(typer.Exit) as refused:
        cli._plan_and_start("some-command", plan, {"task.y": "1"})
    assert refused.value.exit_code == 1
    assert capsys.readouterr().err == ("hls some-command: this command takes only --site.*, --run.* and "
                                       "--resources.* overrides, not task.y\n")


def test_the_no_op_pipeline_has_no_command():
    result = CliRunner().invoke(cli.app, ["noop", "--help"])
    assert result.exit_code == 2 and "No such command" in result.output
    assert "noop" not in CliRunner().invoke(cli.app, ["--help"]).output
    assert not hasattr(cli, "noop_command")


def test_the_cli_routes_resume_status_and_cancel_with_their_config_and_store(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "configure_logging", lambda verbose=False: None)
    calls = []
    monkeypatch.setattr(commands, "resume", lambda instance, *, config, store, echo: calls.append(
        ("resume", instance, config, store)) or 0)
    monkeypatch.setattr(commands, "status", lambda instance, *, config, store, echo: calls.append(
        ("status", instance, config, store)) or 0)
    monkeypatch.setattr(commands, "cancel", lambda instance, *, config, store, echo: calls.append(
        ("cancel", instance, config, store)) or 3)
    resume = CliRunner().invoke(cli.app, ["resume", "runs/noop/r1", "--config", "a.toml", "--store", "/r"])
    status = CliRunner().invoke(cli.app, ["status", "runs/noop/r1", "--config", "b.toml", "--store", "/s"])
    cancelled = CliRunner().invoke(cli.app, ["cancel", "runs/noop/r1", "--config", "c.toml", "--store", "/t"])
    assert (resume.exit_code, status.exit_code, cancelled.exit_code) == (0, 0, 3)
    # Without the flag the store is the one hls.toml names, which the CLI reads before the command.
    config = _config(tmp_path)
    CliRunner().invoke(cli.app, ["status", "runs/noop/r1", "--config", str(config)])
    assert calls == [("resume", "runs/noop/r1", Path("a.toml"), Path("/r")),
                     ("status", "runs/noop/r1", Path("b.toml"), Path("/s")),
                     ("cancel", "runs/noop/r1", Path("c.toml"), Path("/t")),
                     ("status", "runs/noop/r1", config, tmp_path / "store")]


@pytest.mark.parametrize("command", ["resume", "status", "cancel", "follow"])
@pytest.mark.parametrize("extra, says", [
    ('not_a_setting = "hunter2-value"\n', "unknown setting run.not_a_setting"),
    ('account = "hunter2-value\n', "not valid TOML (at line 14, column 25)"),
], ids=["invalid", "not-toml"])
def test_the_cli_refuses_an_invalid_config_in_one_line_naming_the_command(tmp_path, monkeypatch, command,
                                                                          extra, says):
    # Without --store the command reads hls.toml for the store's root; a value of the file is
    # never shown.
    monkeypatch.setattr(cli, "configure_logging", lambda verbose=False: None)
    config = _config(tmp_path)
    config.write_text(config.read_text() + extra)
    result = CliRunner().invoke(cli.app, [command, "runs/noop/r1", "--config", str(config)])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.exception
    assert result.stdout == ""
    assert result.stderr == f"hls {command}: {config}: {says}\n"


@pytest.mark.parametrize("command", ["resume", "status", "cancel", "follow"])
def test_an_error_raised_after_the_store_opens_keeps_its_traceback(tmp_path, monkeypatch, command):
    # Only reading hls.toml for the store's root is a refusal. A SettingsError the command raises
    # once the store is open, while it drives or reads the instance, is a fault.
    monkeypatch.setattr(cli, "configure_logging", lambda verbose=False: None)
    opened: list[Path] = []

    def act(instance, *, store, echo, config=None):
        opened.append(store.root if isinstance(store, PosixStore) else commands.open_store(config, store).root)
        raise SettingsError("raised while the instance is driven")

    monkeypatch.setattr(follow_mod if command == "follow" else commands, command, act)
    result = CliRunner().invoke(cli.app, [command, "runs/noop/r1", "--config", str(_config(tmp_path))])
    assert opened == [tmp_path / "store"]
    assert type(result.exception) is SettingsError and str(result.exception) == "raised while the instance is driven"
    assert result.exc_info[2] is not None and result.stderr == ""


def test_the_cli_status_of_an_unknown_instance_exits_2(tmp_path):
    result = CliRunner().invoke(cli.app, ["status", "runs/noop/typo", "--config", str(_config(tmp_path))])
    assert result.exit_code == 2 and f"no such instance under {tmp_path / 'store'}" in result.output


def _ended(plan: Plan, outcome: str = "void", **first: object) -> Plan:
    """*plan* ended with *outcome*, every submission cancelled but the first, which *first* changes."""
    allocations = [a.model_copy(update={"submissions": [x.model_copy(update={"status": "cancelled"})
                                                        for x in a.submissions]}) for a in plan.allocations]
    changed = allocations[0].submissions[0].model_copy(update=first)
    allocations[0] = allocations[0].model_copy(update={"submissions": [changed]})
    return plan.model_copy(update={"allocations": allocations, "outcome": Outcome(status=outcome, reason="x", at=1.0)})


LIVE = {"status": "submitted"}


@pytest.mark.parametrize("status, fields, job, settled", [
    ("void", {}, None, True),
    ("void", LIVE, JobState.PENDING, False),
    ("void", LIVE, JobState.RUNNING, False),
    ("void", LIVE, JobState.FAILED, True),
    ("void", {**LIVE, "job_id": "999"}, None, False),
    ("succeeded", LIVE, JobState.RUNNING, True),
    ("void", {"status": "cancel_intended"}, JobState.FAILED, False),
    ("void", {"strays": ["555"]}, None, False),
    ("void", {"status": "intended", "job_id": None}, None, False),
    ("void", {"status": "absent", "job_id": None, "missing_since": 4990.0}, None, False),
    ("void", {"status": "absent", "job_id": None, "missing_since": 4700.0}, None, True),
], ids=["all cancelled", "live job pending", "live job running", "live job failed", "live job unknown",
        "succeeded with a job running", "cancel not finished", "stray", "sbatch unanswered", "tag may turn up",
        "tag looked up long enough"])
def test_an_ended_instance_is_settled_when_nothing_of_it_is_outstanding(tmp_path, status, fields, job, settled):
    slurm = LocalSlurm()
    plan = _instance(tmp_path, slurm).plans.latest()
    if job is not None:
        slurm.jobs["100"].state = job
    assert commands.settled(_ended(plan, status, **fields), slurm, now=5000.0) is settled


def test_an_instance_still_running_is_not_settled(tmp_path):
    slurm = LocalSlurm()
    assert not commands.settled(_instance(tmp_path, slurm).plans.latest(), slurm, now=5000.0)


def _void_with_a_running_job(tmp_path: Path, slurm: LocalSlurm) -> Orchestrator:
    """An instance that ended void because sbatch refused allocation 1, while job 100 of
    allocation 0 still runs; the orchestrator is gone before the pass that would cancel it."""
    orch = Orchestrator.create(noop_plan(tmp_path, gpu_policy="split"), slurm, holder=LOGIN, sleep=lambda s: None)
    orch.step()
    slurm.start("100", at=time.time())
    slurm.submit_fault = "refuse"
    orch.step()
    assert orch.plans.latest().outcome.status == "void"
    orch.handle.release()
    return orch


def test_cancel_of_a_void_instance_whose_job_still_runs_takes_the_lease_and_cancels_it(tmp_path):
    slurm = LocalSlurm()
    orch = _void_with_a_running_job(tmp_path, slurm)
    lines: list[str] = []
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep())
    assert code == 1 and "100" in slurm.cancels() and slurm.jobs["100"].state is JobState.CANCELLED
    assert lines[-1].startswith("runs/noop/r1: void: ")


def test_resume_of_a_void_instance_whose_job_still_runs_takes_the_lease_and_cancels_it(tmp_path):
    slurm = LocalSlurm()
    orch = _void_with_a_running_job(tmp_path, slurm)
    lines: list[str] = []
    code = commands.resume(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep())
    assert code == 1 and "100" in slurm.cancels() and slurm.jobs["100"].state is JobState.CANCELLED
    assert lines[0] == "attached to runs/noop/r1 as lease generation 2"


def test_cancel_and_resume_of_a_succeeded_instance_write_nothing(tmp_path, monkeypatch):
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, two_stage=True)
    store, layout = PosixStore(tmp_path / "store"), InstanceLayout(plan.instance)
    lease = instance_lease(store, layout, ttl_sec=30).current()
    said: list[str] = []
    assert commands.cancel(plan.instance, config=_config(tmp_path), scheduler=slurm, echo=said.append,
                           sleep=bounded_sleep()) == 1
    assert commands.resume(plan.instance, config=_config(tmp_path), scheduler=slurm, echo=said.append,
                           sleep=bounded_sleep()) == 0
    assert said == [f"{plan.instance} already ended: succeeded: every stage published"] * 2
    assert not store.exists(layout.cancel_request)
    assert PlanStore(store, layout).latest() == plan
    assert instance_lease(store, layout, ttl_sec=30).current() == lease


def test_cancel_of_a_cancelled_instance_succeeds_and_resume_of_a_void_one_does_not(tmp_path, monkeypatch):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    orch.handle.release()
    config = _config(tmp_path)
    assert commands.cancel(orch.layout.root, config=config, scheduler=slurm, echo=lambda line: None,
                           sleep=bounded_sleep()) == 0
    before = PlanStore(orch.store, orch.layout).latest()
    said: list[str] = []
    assert commands.cancel(orch.layout.root, config=config, scheduler=slurm, echo=said.append,
                           sleep=bounded_sleep()) == 0
    assert said == ["runs/noop/r1 already ended: cancelled: cancelled by hls cancel"]
    assert PlanStore(orch.store, orch.layout).latest() == before
    _, _, voided_slurm, voided = _local_start(tmp_path / "void", monkeypatch, two_stage=True, fail_attempts=9)
    said.clear()
    assert commands.resume(voided.instance, config=_config(tmp_path / "void"), scheduler=voided_slurm,
                           echo=said.append, sleep=bounded_sleep()) == 1
    assert said[0].startswith(f"{voided.instance} already ended: void: ")


def test_cancel_of_an_ended_instance_with_a_cancel_unfinished_takes_the_lease_and_finishes_it(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    orch.plans.write(_ended(orch.plans.latest(), status="cancel_intended"), orch.handle)
    orch.handle.release()
    lines: list[str] = []
    code = commands.cancel(orch.layout.root, config=_config(tmp_path), scheduler=slurm, echo=lines.append,
                           sleep=bounded_sleep())
    assert "100" in slurm.cancels() and slurm.jobs["100"].state is JobState.CANCELLED
    assert code == 1 and lines[-1].startswith("runs/noop/r1: void: ")
    assert orch.plans.latest().lease_generation == 2


def test_commands_name_an_instance_the_store_does_not_hold_and_write_nothing(tmp_path):
    slurm = LocalSlurm()
    config = _config(tmp_path)
    said: list[str] = []
    expected = f"no such instance under {tmp_path / 'store'}"
    assert commands.cancel("runs/noop/typo", config=config, scheduler=slurm, echo=said.append,
                           sleep=bounded_sleep()) == 2
    assert commands.resume("runs/noop/typo", config=config, scheduler=slurm, echo=said.append,
                           sleep=bounded_sleep()) == 2
    assert commands.status("runs/noop/typo", config=config, scheduler=slurm, echo=said.append) == 2
    assert said == [expected] * 3
    assert not (tmp_path / "store" / "runs" / "noop" / "typo").exists()
    with pytest.raises(commands.NoSuchInstance, match=expected):
        commands.status_text("runs/noop/typo", config=config, scheduler=slurm)


def test_status_and_cancel_take_the_store_from_the_flag_not_hls_toml(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    orch.handle.release()
    nothing = tmp_path / "no-such-hls.toml"
    text = commands.status_text(orch.layout.root, config=nothing, store=tmp_path / "store", scheduler=slurm)
    assert "instance runs/noop/r1" in text
    assert commands.cancel(orch.layout.root, config=nothing, store=tmp_path / "store", scheduler=slurm,
                           echo=lambda line: None, sleep=bounded_sleep()) == 0


def test_start_names_the_store_it_was_given_by_an_override(tmp_path, monkeypatch):
    other = tmp_path / "other"
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, two_stage=True, store_root=other)
    assert code == 0 and plan.store_root == other
    assert lines[1] == f"store root {other}; resume, status and cancel need --store {other}"
    _, quiet, _, _ = _local_start(tmp_path / "again", monkeypatch, two_stage=True)
    assert not any(line.startswith("store root") for line in quiet)


def test_status_shows_strays_a_rewire_in_flight_and_a_tag_that_found_nothing(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    plan = orch.plans.latest()
    first = plan.allocation(0).submissions[0].model_copy(
        update={"strays": ["555", "556"], "rewire_to": "777", "rewire_refusals": 2})
    second = plan.allocation(1).submissions[0].model_copy(
        update={"status": "intended", "job_id": None, "missing_since": 1234.0})
    changed = plan.with_allocation(plan.allocation(0).model_copy(update={"submissions": [first]}))
    changed = changed.with_allocation(plan.allocation(1).model_copy(update={"submissions": [second]}))
    orch.plans.write(changed, orch.handle)
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert "; strays: 555, 556" in text
    assert "; rewire to job 777 intended, 2 refusals" in text
    assert "; no job under its tag since " + datetime.fromtimestamp(1234.0).strftime("%Y-%m-%d %H:%M:%S") in text


def test_status_says_when_a_status_file_does_not_parse(tmp_path):
    slurm = LocalSlurm()
    orch = _instance(tmp_path, slurm)
    orch.store.write_bytes(orch.layout.allocation_status(0, 1), b'{"state": "runn')
    attempt = orch.layout.claim_attempt(orch.store, "noop_a")
    orch.store.write_bytes(orch.layout.attempt_file("noop_a", attempt, "status.json"), b"not json")
    text = commands.status_text(orch.layout.root, config=_config(tmp_path), scheduler=slurm)
    assert ("  submission 1: job 100, submitted, Slurm PENDING; runner status unreadable (json_invalid at the top "
            "level)") in text
    assert "stage noop_a: not published; attempts: 1 status unreadable (json_invalid at the top level)" in text


def test_a_noop_instance_runs_its_role_as_the_contestants_and_holds_batch_memory(tmp_path, monkeypatch):
    monkeypatch.setattr("hls.stages.noop.BATCH_HOG_DELAY_SEC", 0.0)
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, contestant=True, batch_hog_gib=1)
    stage = plan.stage("noop")
    assert code == 0 and stage.params["contestant_role"] and stage.params["batch_hog_gib"] == 1


def test_a_noop_instance_can_run_a_harness_cpu_role_beside_the_contestant_role(tmp_path, monkeypatch):
    code, lines, slurm, plan = _local_start(tmp_path, monkeypatch, contestant=True, with_cpu_role=True)
    stage = plan.stage("noop")
    assert code == 0 and stage.params["cpu_role"] and stage.params["gpu_roles"] == 1
    assert (stage.shape.cpus, stage.shape.memory_gib) == (4, 8)
    store = PosixStore(plan.store_root)
    attempt = plan.layout.attempt_file("noop", 1, "events.jsonl")
    started = [e for e in read_events(store.path(attempt)) if e["kind"] == "role_started"]
    assert [e["role"] for e in started] == ["gpu0", "cpu"]
