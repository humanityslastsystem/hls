from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hls.pipeline import stage as stage_module
from hls.stages import noop as noop_module
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.plan import Plan
from hls.pipeline.stage import StageContext, StageFault, run_stage, stage_fault
from hls.spec.resources import ResourceShape
from hls.stages.noop import NoopParams, NoopStage, isolation_problems
from hls.store.manifest import ReportedOutput, read_outputs_report
from hls.store.posix import PosixStore
from hls.substrate import noop_role
from hls.substrate.base import FaultRecord
from hls.substrate.noop_role import NoopRoleReport
from hls.substrate.faults import RoleOutcome, classify_role
from hls.substrate.memory_watch import JobOoms, MemorySample, MemoryWatch
from hls.substrate.slurm.launcher import srun_argv
from tests.pipeline.fakes import LocalLauncher, fake_uuid
from tests.pipeline.plans import noop_plan

GIB = 2**30
#: A role's slice, and the commands the fake launcher runs as given (it overrides them for
#: roles named in its knobs).
SLICE = ResourceShape(cpus=1, memory_gib=1, time_sec=60)
GPU = ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=1, time_sec=60)
QUICK = [sys.executable, "-c", "pass"]
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
DEAF = [sys.executable, "-c", "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"]
EXIT_3 = [sys.executable, "-c", "import sys; sys.exit(3)"]


def _ctx(plan: Plan, stage: str, launcher: LocalLauncher, *, attempt: int = 1, deadline_in: float = 60.0,
         watch: MemoryWatch | None = None) -> StageContext:
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file(stage, attempt, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage(stage), attempt=attempt, launcher=launcher,
                        watch=watch, events=events, inputs={}, deadline=time.time() + deadline_in,
                        scratch=Path(plan.settings.site.node_scratch) / "hls-100", job_id="100")


def _with_params(plan: Plan, stage: str, **params) -> Plan:
    """*plan* with *params* set on *stage*, past what the pipeline's declaration allows."""
    changed = plan.stage(stage).model_copy(update={"params": {**plan.stage(stage).params, **params}})
    return plan.model_copy(update={"stages": [changed if s.name == stage else s for s in plan.stages]})


class _Scripted:
    """A stage whose run is *body*."""

    kind = "scripted"
    params_model = NoopParams

    def __init__(self, body) -> None:
        self.body = body

    @classmethod
    def check(cls, shape, params, *, runner_memory_gib) -> None:
        pass

    def run(self, ctx):
        return self.body(ctx)


def _ended(ctx: StageContext) -> list[tuple[str, str]]:
    """(role, ending) of every role_ended event, sorted."""
    events = read_events(ctx.attempt_dir / "events.jsonl")
    return sorted((e["role"], e["ending"]) for e in events if e["kind"] == "role_ended")


def test_noop_b_runs_isolated_roles_and_names_its_outputs(tmp_path):
    plan = noop_plan(tmp_path)
    ctx = _ctx(plan, "noop_b", LocalLauncher(gpus=2))
    outputs = run_stage(NoopStage(), ctx)
    assert set(outputs) == {"gpu0", "gpu1", "cpu", "shared", "summary"}
    summary = json.loads(ctx.store.read_bytes(outputs["summary"]))
    assert summary["gpu0"]["cuda_uuids"] == [fake_uuid(0)] and summary["gpu1"]["cuda_uuids"] == [fake_uuid(1)]
    assert summary["shared"]["cuda_uuids"] == [fake_uuid(1)] and summary["cpu"]["cuda_uuids"] == []
    kinds = [e["kind"] for e in read_events(ctx.attempt_dir / "events.jsonl")]
    assert kinds.count("role_started") == 4 and kinds.count("role_ended") == 4


def test_a_shared_role_on_the_wrong_gpu_is_refused(tmp_path):
    plan = noop_plan(tmp_path)
    with pytest.raises(StageFault, match="shared sees") as caught:
        NoopStage().run(_ctx(plan, "noop_b", LocalLauncher(gpus=2, wrong_share=True)))
    assert not caught.value.fault.retryable


def test_planted_failures_fail_early_attempts_only(tmp_path):
    plan = _with_params(noop_plan(tmp_path), "noop_a", fail_attempts=1)
    with pytest.raises(StageFault, match="planted fault on attempt 1 of 1") as caught:
        NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2), attempt=1))
    assert caught.value.fault.retryable and caught.value.fault.fault_class == "infra"
    assert set(NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2), attempt=2))) == {"gpu0", "summary"}


def test_a_role_with_the_wrong_grant_fails_the_stage_as_infra(tmp_path):
    plan = noop_plan(tmp_path)
    with pytest.raises(StageFault, match="role gpu0: grant mismatch") as caught:
        NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2, mismatch=("gpu0",))))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence["roles"] == "gpu0" and fault.evidence["gpu0.detail"] == "gpus: declared 1, granted 2"


def test_roles_past_the_deadline_are_killed(tmp_path):
    plan = _with_params(noop_plan(tmp_path), "noop_a", sleep_sec=60)
    started = time.time()
    with pytest.raises(StageFault, match="ended time_limit"):
        NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2), deadline_in=1.0))
    assert time.time() - started < 15


def test_a_role_over_its_memory_is_killed_where_swap_is_unlimited(tmp_path):
    plan = _with_params(noop_plan(tmp_path), "noop_a", sleep_sec=60)
    launcher = LocalLauncher(gpus=2, cgroup_root=tmp_path / "cg")
    watch = MemoryWatch(enforce=True)
    ctx = _ctx(plan, "noop_a", launcher, watch=watch)
    cg = tmp_path / "cg" / "cg-gpu0"
    cg.mkdir(parents=True)
    for name, value in {"memory.current": 2 * GIB, "memory.swap.current": GIB}.items():
        (cg / name).write_text(f"{value}\n")
    (cg / "memory.stat").write_text("anon 2147483648\nfile 0\nshmem 0\n")
    (cg / "memory.events").write_text("oom_kill 0\n")
    threading.Timer(0.5, watch.sample_once).start()
    with pytest.raises(StageFault, match="harness role gpu0 ran out of memory") as caught:
        NoopStage().run(ctx)
    assert "over 2 GiB" in caught.value.fault.evidence["gpu0.detail"]


def test_a_contestant_role_exiting_97_is_a_result_not_a_grant_mismatch(tmp_path):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2, exit_97=("contest",)))
    ctx.start_role("contest", QUICK, SLICE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("exited", 97)
    assert classify_role(outcome) is None


def test_a_role_that_never_reports_is_killed_at_once(tmp_path):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2, no_report=("gpu0",)))
    handle = ctx.start_role("gpu0", SLEEP, SLICE, owner="harness")
    assert handle.wait(5.0) is not None
    [event] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_launch_failed"]
    assert event["role"] == "gpu0" and "no report" in event["error"]


@pytest.mark.parametrize("deadline_in", [60.0, -1.0], ids=["before the deadline", "past the deadline"])
def test_a_role_that_never_reports_ends_launch_failed(tmp_path, deadline_in):
    plan = noop_plan(tmp_path)
    started = time.time()
    with pytest.raises(StageFault, match="role gpu0: launch failed") as caught:
        NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2, no_report=("gpu0",)), deadline_in=deadline_in))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert time.time() - started < 15


def test_a_step_the_scheduler_ended_is_infra_and_retryable_for_any_owner(tmp_path):
    plan = noop_plan(tmp_path)
    with pytest.raises(StageFault, match="role gpu0: scheduler ended") as caught:
        NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2, scheduler_ended=("gpu0",))))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    ctx = _ctx(plan, "noop_a", LocalLauncher(gpus=2, scheduler_ended=("contest",)))
    ctx.start_role("contest", QUICK, SLICE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("scheduler_ended", 1)
    fault = classify_role(outcome)
    assert fault is not None and fault.fault_class == "infra" and fault.retryable


class _QuietWatch:
    """A memory watch whose last sample of every step saw no OOM kill, in a job whose own limit
    was never reached."""

    def watch(self, step, cgroup_dir, limit_bytes, kill, *, role) -> None:
        pass

    def last(self, step) -> MemorySample:
        return MemorySample(at=0.0, current=GIB, swap_current=0, reclaimable=0, oom_kills=0)

    def breach(self, step) -> MemorySample | None:
        return None

    #: No job cgroup is watched, so no role step lies under one.
    job_dir = None

    def job_ooms(self) -> JobOoms:
        return JobOoms(own=0, below=0, steps={})


def test_a_step_that_ended_out_of_memory_is_an_oom_whoever_owns_the_role(tmp_path):
    # Where the kernel enforces memory, a role can be OOM-killed and its step end before the
    # watch samples the kill: only accounting has it (step state OUT_OF_MEMORY, exit 0:125).
    plan = noop_plan(tmp_path)
    with pytest.raises(StageFault, match="harness role gpu0 ran out of memory") as caught:
        NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2, out_of_memory=("gpu0",)), watch=_QuietWatch()))
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence["gpu0.oom_kills"] == "1" and "OUT_OF_MEMORY" in fault.evidence["gpu0.detail"]
    # The same kill of a contestant role is infra while nothing says its own limit was reached:
    # this job records no limit reached below it.
    ctx = _ctx(plan, "noop_a", LocalLauncher(gpus=2, out_of_memory=("contest",)), watch=_QuietWatch())
    ctx.start_role("contest", QUICK, SLICE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code, outcome.oom_kills) == ("exited", 1, 1)
    fault = classify_role(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "contestant role contest was OOM-killed and its own memory limit was not reached"


def test_stopping_terminates_every_role_before_waiting_on_any(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher)
    for name in ("first", "second", "third"):
        ctx.start_role(name, SLEEP, SLICE, owner="harness")
    ctx.stop_roles()
    outcomes = ctx.wait_roles()
    assert [o.ending for o in outcomes] == ["stopped"] * 3
    acts = [(what, name) for what, name in launcher.calls if what in ("terminate", "wait")]
    assert acts[:3] == [("terminate", "first"), ("terminate", "second"), ("terminate", "third")]
    assert all(what == "wait" for what, _ in acts[3:])


def test_stopping_gives_every_role_one_shared_grace_period(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_module, "STOP_GRACE_SEC", 1.0)
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher)
    for name in ("a", "b", "c"):
        ctx.start_role(name, DEAF, SLICE, owner="harness")
    time.sleep(0.5)
    ctx.stop_roles()
    started = time.time()
    outcomes = ctx.wait_roles()
    assert time.time() - started < 2.5
    assert [o.ending for o in outcomes] == ["stopped"] * 3
    assert all(h.poll() is not None for h in launcher.handles)
    assert sorted(launcher.calls.count(("terminate", name)) for name in ("a", "b", "c")) == [1, 1, 1]


def test_stopping_signals_every_role_in_one_launcher_call_within_the_shared_grace(tmp_path, monkeypatch):
    # Slurm signals a step with scancel, which can take its whole 10 s timeout while the
    # controller is cancelling the job. One call for every role keeps the stop
    # within one such call and the one grace period the roles share.
    monkeypatch.setattr(stage_module, "STOP_GRACE_SEC", 1.0)
    launcher = LocalLauncher(gpus=2, signal_sec=1.0)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher)
    for name in ("a", "b", "c"):
        ctx.start_role(name, DEAF, SLICE, owner="harness")
    time.sleep(0.5)
    ctx.stop_roles()
    started = time.time()
    outcomes = ctx.wait_roles()
    assert time.time() - started < 2.5
    assert [o.ending for o in outcomes] == ["stopped"] * 3
    assert launcher.terminations == [["a", "b", "c"]]


def test_stopping_does_not_wait_on_the_scheduler_about_a_role_that_ended_first(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_module, "STOP_GRACE_SEC", 1.0)
    launcher = LocalLauncher(gpus=2, scheduler_ended=("a",), scheduler_query_sec=3.0)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher)
    # Role a ends on its own, as when Slurm signals every step of a cancelled job.
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    ctx.start_role("b", DEAF, SLICE, owner="harness")
    ctx.start_role("c", DEAF, SLICE, owner="harness")
    time.sleep(0.5)
    assert launcher.handles[0].poll() == 1
    ctx.stop_roles()
    started = time.time()
    outcomes = ctx.wait_roles()
    assert time.time() - started < 2.5
    assert [(o.name, o.ending) for o in outcomes] == [("a", "stopped"), ("b", "stopped"), ("c", "stopped")]
    assert [h.asked for h in launcher.handles] == [0, 0, 0]
    assert launcher.calls.count(("terminate", "b")) == launcher.calls.count(("terminate", "c")) == 1


class _SlowWatch:
    """A memory watch whose reading takes two seconds, so reporting a role takes that long."""

    def watch(self, step, cgroup_dir, limit_bytes, kill, *, role) -> None:
        pass

    def last(self, step) -> MemorySample | None:
        time.sleep(2.0)
        return None

    def breach(self, step) -> MemorySample | None:
        return None

    #: No job cgroup is watched, so no role step lies under one.
    job_dir = None

    def job_ooms(self) -> JobOoms:
        return JobOoms(own=0, below=0, steps={})


def test_stopping_signals_every_role_before_it_reports_any(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher, watch=_SlowWatch())
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    ctx.start_role("b", SLEEP, SLICE, owner="harness")
    time.sleep(0.5)
    ctx.stop_roles()
    waiting = threading.Thread(target=ctx.wait_roles)
    waiting.start()
    time.sleep(1.0)
    # Role a is still being reported, and role b has been signalled already.
    assert launcher.handles[1].poll() is not None
    waiting.join(30)


@pytest.mark.parametrize("wait_after", [0.0, 2.0], ids=["waited on at once", "waited on after the deadline"])
def test_a_role_that_ended_before_the_deadline_is_not_a_time_limit_when_another_runs_past_it(tmp_path, wait_after):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2), deadline_in=1.5)
    ctx.start_role("server", SLEEP, SLICE, owner="contestant")
    ctx.start_role("driver", [sys.executable, "-c", "import time; time.sleep(0.5)"], SLICE, owner="harness")
    time.sleep(wait_after)
    outcomes = {o.name: o for o in ctx.wait_roles()}
    assert (outcomes["driver"].ending, outcomes["driver"].exit_code) == ("exited", 0)
    assert outcomes["server"].ending == "time_limit"


def test_a_wait_with_a_command_limit_kills_what_still_runs_that_long_after_its_start(tmp_path):
    """A stage can hold a role's command to a limit of its own, short of the stage's deadline:
    what still runs that long after its command started is killed and ends time_limit, and what
    ended first stands. A role on the host runs its command from its step's start."""
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2), deadline_in=60.0)
    ctx.start_role("server", SLEEP, SLICE, owner="contestant")
    ctx.start_role("driver", [sys.executable, "-c", "pass"], SLICE, owner="harness")
    outcomes = {o.name: o for o in ctx.wait_roles(command_sec=1.5)}
    began = ctx.command_started_at("server")
    assert began is not None and 1.5 <= ctx.clock() - began < 15
    assert outcomes["server"].ending == "time_limit"
    assert (outcomes["driver"].ending, outcomes["driver"].exit_code) == ("exited", 0)


def test_a_command_limit_counts_for_each_role_from_its_own_start(tmp_path):
    """Roles waited for together each get the whole limit from their own command's start: the one
    that reaches it is killed, and one started two seconds later, still running then, runs on
    and ends as its own command does."""
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2), deadline_in=60.0)
    ctx.start_role("early", SLEEP, SLICE, owner="contestant")
    early = ctx.command_started_at("early")
    assert early is not None
    while ctx.clock() < early + 2.0:
        time.sleep(0.05)
    ctx.start_role("late", [sys.executable, "-c", "import time; time.sleep(1.5)"], SLICE, owner="contestant")
    outcomes = {o.name: o for o in ctx.wait_roles(command_sec=3.0)}
    assert outcomes["early"].ending == "time_limit"
    assert (outcomes["late"].ending, outcomes["late"].exit_code) == ("exited", 0)


def test_a_command_limit_past_the_deadline_still_ends_at_the_deadline(tmp_path):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2), deadline_in=1.5)
    ctx.start_role("server", SLEEP, SLICE, owner="contestant")
    started = time.time()
    [outcome] = ctx.wait_roles(command_sec=600)
    assert outcome.ending == "time_limit" and time.time() - started < 15


def test_the_grant_report_wait_stops_at_the_stage_deadline(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher, deadline_in=10.0)
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    [timeout] = launcher.handles[0].report_timeouts
    assert 1.0 <= timeout <= 10.0


def test_a_role_ending_with_a_fault_stops_the_others_at_once(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)
    ctx.start_role("server", SLEEP, SLICE, owner="contestant")
    ctx.start_role("driver", EXIT_3, SLICE, owner="harness")
    started = time.time()
    outcomes = ctx.wait_roles()
    assert time.time() - started < 10
    assert [(o.name, o.ending) for o in outcomes] == [("server", "stopped"), ("driver", "exited")]
    fault = stage_fault(outcomes)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence["roles"] == "driver"


def test_a_role_that_exits_0_while_the_stage_is_stopping_is_stopped_not_exited(tmp_path):
    # Under a job-wide SIGTERM a graceful exit can be a cut-short run, so it must not count as success.
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2))
    ctx.start_role("driver", QUICK, SLICE, owner="contestant")
    time.sleep(0.5)
    ctx.stop_roles()
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("stopped", 0)
    fault = stage_fault([outcome])
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)


class _BreachedWatch:
    """A memory watch that says every step was killed for memory."""

    def watch(self, step, cgroup_dir, limit_bytes, kill, *, role) -> None:
        pass

    def last(self, step) -> MemorySample | None:
        return None

    def breach(self, step) -> MemorySample:
        return MemorySample(at=0.0, current=3 * GIB, swap_current=0, reclaimable=0, oom_kills=0)

    #: No job cgroup is watched, so no role step lies under one.
    job_dir = None

    def job_ooms(self) -> JobOoms:
        return JobOoms(own=0, below=0, steps={})


def test_a_memory_breach_outranks_a_step_the_scheduler_ended(tmp_path):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2, scheduler_ended=("gpu0",)), watch=_BreachedWatch())
    ctx.start_role("gpu0", QUICK, SLICE, owner="harness")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("memory_limit", 1)


def test_a_contestant_out_of_memory_with_its_driver_stopped_is_the_contestants_fault(tmp_path):
    launcher = LocalLauncher(gpus=2, cgroup_root=tmp_path / "cg")
    watch = MemoryWatch(enforce=True)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher, watch=watch)
    cg = tmp_path / "cg" / "cg-server"
    cg.mkdir(parents=True)
    for name, value in {"memory.current": 2 * GIB, "memory.swap.current": 0}.items():
        (cg / name).write_text(f"{value}\n")
    (cg / "memory.stat").write_text("anon 2147483648\nfile 0\nshmem 0\n")
    (cg / "memory.events").write_text("oom_kill 0\n")
    ctx.start_role("server", SLEEP, SLICE, owner="contestant")
    ctx.start_role("driver", SLEEP, SLICE, owner="harness")
    threading.Timer(0.5, watch.sample_once).start()
    started = time.time()
    outcomes = ctx.wait_roles()
    assert time.time() - started < 10
    assert [(o.name, o.ending) for o in outcomes] == [("server", "memory_limit"), ("driver", "stopped")]
    fault = stage_fault(outcomes)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("contestant", False)


def _outcome(name, owner, ending, code=1, **kw) -> RoleOutcome:
    return RoleOutcome(name, owner, code, ending, **kw)


@pytest.mark.parametrize("outcomes, expected, roles", [
    ([_outcome("server", "contestant", "memory_limit", -9), _outcome("driver", "harness", "exited", 1)],
     ("infra", True), "server,driver"),
    ([_outcome("server", "contestant", "memory_limit", -9), _outcome("driver", "harness", "exited", 0)],
     ("contestant", False), "server"),
    ([_outcome("a", "contestant", "memory_limit", -9), _outcome("b", "contestant", "memory_limit", -9)],
     ("contestant", False), "a,b"),
    ([_outcome("a", "harness", "grant_mismatch", 97), _outcome("b", "harness", "exited", 1)],
     ("infra", True), "a,b"),
    ([_outcome("a", "harness", "exited", 1), _outcome("b", "harness", "stopped", -15)], ("infra", True), "a"),
    ([_outcome("a", "harness", "stopped", -15), _outcome("b", "harness", "stopped", -15)],
     ("infra", True), "a,b"),
    ([_outcome("a", "contestant", "exited", 1), _outcome("b", "harness", "exited", 0)], None, ""),
], ids=["contestant OOM and harness crash", "contestant OOM alone", "two contestant faults",
        "two infra faults", "a stopped role adds nothing to a fault", "only stopped roles", "no fault"])
def test_the_stage_fault_is_decided_from_every_roles_fault(outcomes, expected, roles):
    fault = stage_fault(outcomes)
    if expected is None:
        assert fault is None
        return
    assert fault is not None and (fault.fault_class, fault.retryable) == expected
    assert fault.evidence["roles"] == roles


def test_one_infra_fault_that_cannot_be_retried_makes_the_stage_fault_not_retryable(monkeypatch):
    monkeypatch.setattr(stage_module, "classify_role", lambda o: FaultRecord(
        fault_class="infra", reason=o.name, retryable=o.name != "a"))
    fault = stage_fault([_outcome("a", "harness", "exited"), _outcome("b", "harness", "exited")])
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", False)


def test_a_shared_role_whose_target_failed_is_an_infra_fault_and_no_role_outlives_it(tmp_path):
    plan = _with_params(noop_plan(tmp_path), "noop_b", sleep_sec=30)
    launcher = LocalLauncher(gpus=2, no_report=("gpu1",))
    ctx = _ctx(plan, "noop_b", launcher)
    started = time.time()
    with pytest.raises(StageFault, match="role shared cannot share the GPUs of gpu1") as caught:
        run_stage(NoopStage(), ctx)
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", True)
    assert all(h.poll() is not None for h in launcher.handles)
    assert [s.name for s in launcher.launched] == ["gpu0", "gpu1", "cpu"]
    assert time.time() - started < 15


def test_a_shared_role_whose_target_has_a_grant_mismatch_is_refused(tmp_path):
    plan = _with_params(noop_plan(tmp_path), "noop_b", sleep_sec=30)
    launcher = LocalLauncher(gpus=2, mismatch=("gpu1",))
    with pytest.raises(StageFault, match="role shared cannot share the GPUs of gpu1.*gpus: declared 1") as caught:
        run_stage(NoopStage(), _ctx(plan, "noop_b", launcher))
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", True)
    assert all(h.poll() is not None for h in launcher.handles)


def test_a_stage_that_raises_has_its_roles_reaped_and_the_error_becomes_an_infra_fault(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)

    def body(ctx):
        ctx.start_role("a", SLEEP, SLICE, owner="harness")
        ctx.start_role("b", SLEEP, SLICE, owner="harness")
        raise ValueError("stage bug")

    with pytest.raises(StageFault) as caught:
        run_stage(_Scripted(body), ctx)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence == {"exception": "ValueError", "message": "stage bug"}
    assert all(h.poll() is not None for h in launcher.handles)
    assert _ended(ctx) == [("a", "stopped"), ("b", "stopped")]


@pytest.mark.parametrize("owner, command, fault", [
    ("harness", EXIT_3, ("infra", True, "harness role a ended exited with status 3")),
    ("contestant", EXIT_3, None),
    ("harness", QUICK, None),
], ids=["harness crash", "contestant exit", "clean exit"])
def test_a_stage_that_returns_without_waiting_for_its_roles_is_judged_by_how_they_ended(tmp_path, owner, command,
                                                                                         fault):
    # The roles ended before the stage returned, so close() reports them; their faults count.
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2))

    def body(ctx):
        assert ctx.start_role("a", command, SLICE, owner=owner).wait(10.0) is not None
        return {"x": ctx.write_output("x", b"x")}

    if fault is None:
        assert set(run_stage(_Scripted(body), ctx)) == {"x"}
        return
    with pytest.raises(StageFault) as caught:
        run_stage(_Scripted(body), ctx)
    got = caught.value.fault
    assert (got.fault_class, got.retryable, got.reason) == fault
    assert got.evidence["roles"] == "a"


def test_a_stage_fault_leaves_run_stage_as_it_was_raised(tmp_path):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2))
    raised = StageFault(FaultRecord(fault_class="contestant", reason="the contestant's", retryable=False))

    def body(ctx):
        raise raised

    with pytest.raises(StageFault) as caught:
        run_stage(_Scripted(body), ctx)
    assert caught.value is raised


def _failing_close(ctx: StageContext, monkeypatch) -> None:
    def close() -> None:
        raise OSError("cannot reap")

    monkeypatch.setattr(ctx, "close", close)


def test_a_close_that_fails_after_a_clean_stage_is_an_infra_fault(tmp_path, monkeypatch):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2))
    _failing_close(ctx, monkeypatch)
    with pytest.raises(StageFault) as caught:
        run_stage(_Scripted(lambda ctx: {}), ctx)
    fault = caught.value.fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence == {"exception": "OSError", "message": "cannot reap"}
    assert isinstance(caught.value.__cause__, OSError)


def test_a_close_that_fails_leaves_the_stage_fault_that_was_propagating(tmp_path, monkeypatch):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2))
    _failing_close(ctx, monkeypatch)
    raised = StageFault(FaultRecord(fault_class="contestant", reason="the contestant's", retryable=False,
                                    evidence={"kept": "yes"}))

    def body(ctx):
        raise raised

    with pytest.raises(StageFault) as caught:
        run_stage(_Scripted(body), ctx)
    assert caught.value is raised
    assert raised.fault.fault_class == "contestant" and not raised.fault.retryable
    assert raised.fault.evidence == {"kept": "yes", "close_exception": "OSError", "close_message": "cannot reap"}


def test_a_close_that_fails_after_another_exception_is_noted_on_its_fault(tmp_path, monkeypatch):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2))
    _failing_close(ctx, monkeypatch)

    def body(ctx):
        raise ValueError("stage bug")

    with pytest.raises(StageFault) as caught:
        run_stage(_Scripted(body), ctx)
    assert caught.value.fault.evidence == {"exception": "ValueError", "message": "stage bug",
                                           "close_exception": "OSError", "close_message": "cannot reap"}


def test_a_stage_that_returns_with_a_role_running_is_a_fault_that_is_not_retryable(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)

    def body(ctx):
        ctx.start_role("leftover", SLEEP, SLICE, owner="harness")
        return {}

    with pytest.raises(StageFault, match="leftover") as caught:
        run_stage(_Scripted(body), ctx)
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", False)
    assert all(h.poll() is not None for h in launcher.handles)
    assert _ended(ctx) == [("leftover", "stopped")]


def test_a_role_started_after_stop_roles_is_refused(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)
    ctx.start_role("a", SLEEP, SLICE, owner="harness")
    ctx.stop_roles()
    with pytest.raises(StageFault, match="stopping") as caught:
        ctx.start_role("b", SLEEP, SLICE, owner="harness")
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", True)
    assert [s.name for s in launcher.launched] == ["a"]
    ctx.close()
    assert launcher.handles[0].poll() is not None


def test_close_reports_only_the_roles_wait_roles_did_not(tmp_path):
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2), deadline_in=1.5)
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    ctx.start_role("b", SLEEP, SLICE, owner="harness")
    ctx.wait_roles()
    ctx.start_role("late", SLEEP, SLICE, owner="harness")
    assert [(o.name, o.ending) for o in ctx.close()] == [("late", "stopped")]
    assert ctx.close() == []
    assert _ended(ctx) == [("a", "exited"), ("b", "time_limit"), ("late", "stopped")]


def test_a_role_of_another_gpu_kind_is_refused(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)
    other = ResourceShape(gpus=1, gpu_kind="H100", cpus=1, memory_gib=1, time_sec=60)
    with pytest.raises(StageFault, match="gpu kind") as caught:
        ctx.start_role("x", SLEEP, other, owner="harness")
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", False)
    assert launcher.launched == []


#: The stage noop_a has 1 GPU, 4 CPUs and 6 GiB, of which its roles leave the runner 4.
@pytest.mark.parametrize("first, second, match", [
    (GPU, GPU, "need 2 GPUs"),
    (ResourceShape(cpus=3, memory_gib=1, time_sec=60), ResourceShape(cpus=2, memory_gib=1, time_sec=60),
     "need 5 CPUs"),
    (ResourceShape(cpus=1, memory_gib=1, time_sec=60), ResourceShape(cpus=1, memory_gib=2, time_sec=60),
     "the roles need 3 GiB and the runner 4 GiB; the stage declares 6"),
], ids=["gpus", "cpus", "memory"])
def test_a_role_past_the_stages_resources_is_refused(tmp_path, first, second, match):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)
    ctx.start_role("first", SLEEP, first, owner="harness")
    with pytest.raises(StageFault, match=match) as caught:
        ctx.start_role("second", SLEEP, second, owner="harness")
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", False)
    assert [s.name for s in launcher.launched] == ["first"]
    ctx.close()


def test_a_shared_role_does_not_count_against_gpus_or_cpus_but_does_against_memory(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)
    main = ctx.start_role("gpu0", SLEEP, ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=1, time_sec=60),
                          owner="harness")
    ctx.start_role("shared", SLEEP, ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=1, time_sec=60),
                   share_gpus_of=main, owner="harness")
    with pytest.raises(StageFault, match="the roles need 3 GiB and the runner 4 GiB"):
        ctx.start_role("shared2", SLEEP, ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=1, time_sec=60),
                       share_gpus_of=main, owner="harness")
    ctx.close()


def test_a_role_that_shares_anothers_memory_is_counted_for_nothing_but_must_fit_alone(tmp_path):
    """``share_memory_of`` oversubscribes one role's memory on another's: the stage's budget does
    not count it, its own memory must fit the stage beside the runner, and only an overlap step
    may set it. A role that does not set it is counted as before."""
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)
    main = ctx.start_role("gpu0", SLEEP, ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=1, time_sec=60),
                          owner="harness")
    with pytest.raises(ValueError, match="only an overlap step shares another role's memory"):
        ctx.start_role("alone", SLEEP, ResourceShape(cpus=1, memory_gib=1, time_sec=60), share_memory_of=main,
                       owner="harness")
    with pytest.raises(StageFault, match="shares another's memory asks for 3 GiB"):
        ctx.start_role("big", SLEEP, ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=3, time_sec=60),
                       share_gpus_of=main, share_memory_of=main, owner="harness")
    # 2 GiB on top of the main role's 1 and the runner's 4 would pass the stage's 6.
    ctx.start_role("oversub", SLEEP, ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=2, time_sec=60),
                   share_gpus_of=main, share_memory_of=main, owner="harness")
    # The oversubscribed role is not counted: a shared role of 1 GiB still fits.
    ctx.start_role("shared", SLEEP, ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=1, time_sec=60),
                   share_gpus_of=main, owner="harness")
    assert [s.name for s in launcher.launched] == ["gpu0", "oversub", "shared"]
    ctx.close()


def test_a_role_that_ended_gives_its_resources_back(tmp_path):
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher)
    big = ResourceShape(cpus=4, memory_gib=2, time_sec=60)
    first = ctx.start_role("first", QUICK, big, owner="harness")
    assert first.wait(10.0) == 0
    ctx.start_role("second", QUICK, big, owner="harness")
    ctx.close()


def test_a_gpu_role_with_no_view_of_its_gpus_is_a_retryable_infra_fault(tmp_path):
    plan = noop_plan(tmp_path)
    with pytest.raises(StageFault, match="gpu0") as caught:
        NoopStage().run(_ctx(plan, "noop_a", LocalLauncher(gpus=2, no_view=("gpu0",))))
    assert (caught.value.fault.fault_class, caught.value.fault.retryable) == ("infra", True)


def test_noop_role_reports_before_it_holds_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(noop_role, "cuda_uuids", lambda: ["GPU-x"])
    out = tmp_path / "gpu0.json"
    report_file = tmp_path / "outputs.json"
    assert noop_role.main(["--out", str(out), "--outputs-out", str(report_file), "--sleep-sec", "0"]) == 0
    data = out.read_bytes()
    assert read_outputs_report(report_file) == {
        out.name: ReportedOutput(sha256=hashlib.sha256(data).hexdigest(), size=len(data))}
    report = noop_role.NoopRoleReport.model_validate_json(out.read_bytes())
    assert report.cuda_uuids == ["GPU-x"]
    assert noop_role.hold_memory(0) == []


def test_cuda_uuids_without_a_driver_is_none(monkeypatch):
    def no_lib(name):
        raise OSError(name)

    monkeypatch.setattr(noop_role.ctypes, "CDLL", no_lib)
    assert noop_role.cuda_uuids() is None


class _Driver:
    """A stand-in for libcuda whose calls return the given statuses."""

    def __init__(self, *, init=0, count=2, count_status=0, device_status=0, uuid_status=0) -> None:
        self.init, self.count, self.count_status = init, count, count_status
        self.device_status, self.uuid_status = device_status, uuid_status

    def cuInit(self, flags):
        return self.init

    def cuDeviceGetCount(self, ref):
        ref._obj.value = self.count
        return self.count_status

    def cuDeviceGet(self, ref, i):
        ref._obj.value = i
        return self.device_status

    def cuDeviceGetUuid(self, raw, dev):
        for k in range(16):
            raw[k] = dev.value * 16 + k
        return self.uuid_status


def test_cuda_uuids_reads_every_device(monkeypatch):
    monkeypatch.setattr(noop_role.ctypes, "CDLL", lambda name: _Driver(count=2))
    assert noop_role.cuda_uuids() == ["GPU-00010203-0405-0607-0809-0a0b0c0d0e0f",
                                      "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f"]
    monkeypatch.setattr(noop_role.ctypes, "CDLL", lambda name: _Driver(count=0))
    assert noop_role.cuda_uuids() == []


@pytest.mark.parametrize("driver", [
    _Driver(init=100), _Driver(count_status=3), _Driver(device_status=3), _Driver(uuid_status=3),
], ids=["cuInit", "cuDeviceGetCount", "cuDeviceGet", "cuDeviceGetUuid"])
def test_cuda_uuids_is_none_when_the_driver_cannot_say(monkeypatch, driver):
    monkeypatch.setattr(noop_role.ctypes, "CDLL", lambda name: driver)
    assert noop_role.cuda_uuids() is None


def _seen(**uuids) -> dict[str, NoopRoleReport]:
    return {name: NoopRoleReport(host="h", cuda_uuids=u, cuda_visible_devices=None) for name, u in uuids.items()}


A, B = fake_uuid(0), fake_uuid(1)
TWO_ROLES = NoopParams(gpu_roles=2, shared_role=True, cpu_role=True)


def test_isolation_problems_is_empty_for_isolated_roles():
    assert isolation_problems(_seen(gpu0=[A], gpu1=[B], cpu=[], shared=[B]), TWO_ROLES, 1) == []
    assert isolation_problems(_seen(gpu0=[A], gpu1=[B], cpu=None, shared=[B]), TWO_ROLES, 1) == []


@pytest.mark.parametrize("seen, per, match", [
    (_seen(gpu0=[A], gpu1=[A], cpu=[], shared=[A]), 1, f"gpu1 and gpu0 both see {A}"),
    (_seen(gpu0=[A, B], gpu1=[B], cpu=[], shared=[B]), 1, r"gpu0 sees GPUs .*, not 1"),
    (_seen(gpu0=[], gpu1=[B], cpu=[], shared=[B]), 1, r"gpu0 sees GPUs \[\], not 1"),
    (_seen(gpu0=None, gpu1=[B], cpu=[], shared=[B]), 1, "gpu0 sees GPUs None, not 1"),
    (_seen(gpu0=[A], gpu1=[B], cpu=[A], shared=[B]), 1, "cpu sees GPUs"),
    (_seen(gpu0=[A], gpu1=[B], cpu=[], shared=[A]), 1, "shared sees"),
], ids=["overlap", "too many", "too few", "no view", "cpu role sees a GPU", "shared role on the wrong GPU"])
def test_isolation_problems_names_each_way_roles_are_not_isolated(seen, per, match):
    problems = isolation_problems(seen, TWO_ROLES, per)
    assert len(problems) == 1 and re.search(match, problems[0])


def _job_cgroup(tmp_path, ooms: int, *, below: int = 0, batch: int = 0, local: bool = True, events: bool = True):
    """A fake ``job_<id>`` cgroup whose own limit was reached *ooms* times and the limits of the
    cgroups below it *below* times, *batch* of them the batch step's own. *local* and *events*
    leave out ``memory.events.local`` and ``memory.events``."""
    job = tmp_path / "job_1"
    (job / "step_batch").mkdir(parents=True, exist_ok=True)
    for name, there in (("memory.events.local", local), ("memory.events", events)):
        if there:
            count = ooms if name.endswith("local") else ooms + below
            (job / name).write_text(f"low 0\nhigh 0\nmax 0\noom {count}\noom_kill 0\noom_group_kill 0\n")
        else:
            (job / name).unlink(missing_ok=True)
    (job / "step_batch" / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom {batch}\noom_kill 0\n")
    return job


def _step_cgroup(tmp_path, name: str, *, oom: int | None, oom_kill: int):
    """Fake memory files for role *name*'s step as the local launcher places them; *oom* None
    leaves the ``oom`` key out of ``memory.events``."""
    step = tmp_path / "cg" / f"cg-{name}"
    step.mkdir(parents=True, exist_ok=True)
    for file, value in {"memory.current": GIB // 2, "memory.swap.current": 0}.items():
        (step / file).write_text(f"{value}\n")
    (step / "memory.stat").write_text("anon 536870912\nfile 0\nshmem 0\n")
    own = "" if oom is None else f"oom {oom}\n"
    (step / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\n{own}oom_kill {oom_kill}\n")
    return step


#: What the job's counters do during the stage for each cause of a kill: (the job's own limit,
#: the limits below it, of which the batch step's own). A node out of memory and a limit above
#: the job (slurmstepd.scope, system.slice) both leave every count of the job alone.
_CAUSES = {"own limit": (0, 1, 0), "job limit": (1, 0, 0), "node-wide": (0, 0, 0),
           "limit above the job": (0, 0, 0), "batch step's limit": (0, 1, 1)}


@pytest.mark.parametrize("owner", ["contestant", "harness"])
@pytest.mark.parametrize("cause, step_oom, expected", [
    ("own limit", 1, ("contestant", False)),
    ("job limit", 0, ("infra", True)),
    ("node-wide", 0, ("infra", True)),
    ("limit above the job", 0, ("infra", True)),
])
def test_a_kill_a_sample_saw_is_the_contestants_only_when_its_steps_own_limit_was_reached(tmp_path, owner, cause,
                                                                                          step_oom, expected):
    # The kernel counts in a step's oom_kill a kill by any OOM killer, and picks a contestant
    # process (oom_score_adj 1000) first whoever's limit was reached. Only the step's own oom
    # count, read in the same sample, says that the step's own limit was.
    job = _job_cgroup(tmp_path, ooms=3)
    if cause == "limit above the job":
        (tmp_path / "memory.events.local").write_text("oom 1\n")
    _step_cgroup(tmp_path, "role", oom=step_oom, oom_kill=1)
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2, cgroup_root=tmp_path / "cg"), watch=watch)
    own, below, batch = _CAUSES[cause]
    _job_cgroup(tmp_path, ooms=3 + own, below=below, batch=batch)
    ctx.start_role("role", EXIT_3, SLICE, owner=owner)
    watch.sample_once()
    [outcome] = ctx.wait_roles()
    assert (outcome.oom_kills, outcome.own_oom, outcome.job_oom) == (1, step_oom == 1, own == 1)
    fault = stage_fault([outcome])
    want = expected if owner == "contestant" else ("infra", True)
    assert fault is not None and (fault.fault_class, fault.retryable) == want
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert ended["step_ooms"] == step_oom and ended["own_oom"] is (step_oom == 1)
    assert (ended["job_ooms_at_start"], ended["job_ooms"]) == (
        {"own": 3, "below": 0, "steps": {"step_batch": 0}},
        {"own": 3 + own, "below": below, "steps": {"step_batch": batch}})


def _step_events(job, n: int, oom: int, *, kills: int = 0) -> None:
    """Write the step-level ``memory.events`` of step <n> of the fake *job*."""
    step = job / f"step_{n}"
    (step / "user").mkdir(parents=True, exist_ok=True)
    (step / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom {oom}\noom_kill {kills}\n")


def _user_sample(job, n: int, oom: int, kills: int) -> None:
    """Write what the memory watch samples in step <n>'s ``user`` cgroup of the fake *job*."""
    user = job / f"step_{n}" / "user"
    user.mkdir(parents=True, exist_ok=True)
    for file, value in {"memory.current": GIB // 2, "memory.swap.current": 0}.items():
        (user / file).write_text(f"{value}\n")
    (user / "memory.stat").write_text("anon 536870912\nfile 0\nshmem 0\n")
    (user / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom {oom}\noom_kill {kills}\n")


def _gone(job, step: str) -> None:
    """Remove step cgroup *step* of the fake *job*, as Slurm does once the step ends."""
    shutil.rmtree(job / step)


#: How the job's counters and step cgroups change between the stage's start and the server's
#: report, with the server's own-limit kill seen in accounting only, and the attribution due.
#: The server is step 2 behind a driver (step 1) unless the case runs it alone; an earlier
#: stage's step 0 is there at the start.
_SCENES = {
    "its step still there, own count 1": (True, lambda job: (_step_events(job, 2, 1, kills=1),), True),
    "its step still there, own count 0": (True, lambda job: (_step_events(job, 2, 0, kills=1),), False),
    "its step gone, the other steps there and unchanged": (True, lambda job: (_gone(job, "step_2"),), True),
    "its step gone, the driver's step gone": (True, lambda job: (_gone(job, "step_2"), _gone(job, "step_1")), None),
    "its step gone, the driver's own limit reached": (
        True, lambda job: (_gone(job, "step_2"), _step_events(job, 1, 1)), None),
    "its step gone, the batch step's own limit reached": (
        True, lambda job: (_gone(job, "step_2"), (job / "step_batch" / "memory.events").write_text("oom 1\n")), None),
    "its step gone, an earlier stage's step gone since the start": (
        False, lambda job: (_gone(job, "step_2"), _gone(job, "step_0")), None),
    "its step gone, a step nobody started there now": (
        True, lambda job: (_gone(job, "step_2"), _step_events(job, 9, 0)), None),
    "its step gone, no limit reached below the job": (True, lambda job: (_gone(job, "step_2"),), False),
}


@pytest.mark.parametrize("scene", list(_SCENES))
def test_a_kill_only_accounting_saw_is_charged_by_whatever_still_shows_whose_limit_was_reached(tmp_path, scene):
    # Where the step's cgroup is gone within a second of its main process's kill, before a
    # sample. Then the role's own count is read where the step still is; once it is gone, the
    # job's count of limits reached below it names the role only when every other step of the
    # stage, and every step there at its start, is still there with its count unchanged.
    driver, change, expected = _SCENES[scene]
    job = _job_cgroup(tmp_path, ooms=0)
    _step_events(job, 0, 0)
    launcher = LocalLauncher(gpus=2, steps_under=job, out_of_memory=("server",))
    if not driver:
        launcher.step_ids = 1
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher, watch=watch)
    if driver:
        ctx.start_role("driver", SLEEP, SLICE, owner="harness")
        _step_events(job, 1, 0)
    ctx.start_role("server", QUICK, SLICE, owner="contestant")
    _step_events(job, 2, 0)
    assert ctx._roles["server"].wait(10.0) == 1
    _job_cgroup(tmp_path, ooms=0, below=0 if scene.endswith("no limit reached below the job") else 1,
                batch=1 if "batch step" in scene else 0)
    change(job)
    ctx._report_role("server", timed_out=False)
    outcome = ctx._outcomes["server"]
    ctx.close()
    assert (outcome.oom_kills, outcome.own_oom) == (1, expected)
    fault = classify_role(outcome)
    assert fault.fault_class == ("contestant" if expected else "infra")


def test_a_kill_a_sample_saw_decides_a_stage_of_two_role_steps_once_both_are_gone(tmp_path):
    # The driver's step is gone too, so the job's counters cannot name the server; the sample
    # that saw the server's own limit reached still does.
    job = _job_cgroup(tmp_path, ooms=0)
    launcher = LocalLauncher(gpus=2, steps_under=job)
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher, watch=watch)
    ctx.start_role("driver", QUICK, SLICE, owner="harness")
    ctx.start_role("server", EXIT_3, SLICE, owner="contestant")
    _step_events(job, 1, 0)
    _user_sample(job, 2, oom=1, kills=1)
    _step_events(job, 2, 1, kills=1)
    watch.sample_once()
    assert ctx._roles["server"].wait(10.0) == 3 and ctx._roles["driver"].wait(10.0) == 0
    _job_cgroup(tmp_path, ooms=0, below=1)
    _gone(job, "step_1")
    _gone(job, "step_2")
    outcome = {o.name: o for o in ctx.wait_roles()}["server"]
    assert (outcome.oom_kills, outcome.own_oom) == (1, True)
    assert classify_role(outcome).fault_class == "contestant"


def test_a_step_nobody_here_started_between_the_roles_leaves_the_kill_unattributed(tmp_path):
    # A step id between two of the stage's own that no role holds is a step that came and went
    # unseen; its own count is unknown.
    job = _job_cgroup(tmp_path, ooms=0)
    launcher = LocalLauncher(gpus=2, steps_under=job, out_of_memory=("server",))
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher, watch=watch)
    ctx.start_role("driver", SLEEP, SLICE, owner="harness")
    _step_events(job, 1, 0)
    launcher.step_ids += 1
    ctx.start_role("server", QUICK, SLICE, owner="contestant")
    assert ctx._roles["server"].wait(10.0) == 1
    _job_cgroup(tmp_path, ooms=0, below=1)
    _gone(job, "step_3")
    ctx._report_role("server", timed_out=False)
    outcome = ctx._outcomes["server"]
    ctx.close()
    assert outcome.own_oom is None and classify_role(outcome).fault_class == "infra"


def test_a_role_of_the_stage_whose_step_is_not_known_leaves_the_kill_unattributed(tmp_path):
    # The driver never reported, so which step is its, and whether its limit was reached, is
    # unknown.
    job = _job_cgroup(tmp_path, ooms=0)
    launcher = LocalLauncher(gpus=2, steps_under=job, out_of_memory=("server",), no_report=("driver",))
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", launcher, watch=watch)
    ctx.start_role("driver", SLEEP, SLICE, owner="harness")
    ctx.start_role("server", QUICK, SLICE, owner="contestant")
    assert ctx._roles["server"].wait(10.0) == 1
    _job_cgroup(tmp_path, ooms=0, below=1)
    ctx._report_role("server", timed_out=False)
    outcome = ctx._outcomes["server"]
    ctx.close()
    assert outcome.own_oom is None and classify_role(outcome).fault_class == "infra"


def test_role_ended_records_every_step_count_at_the_start_and_at_the_report(tmp_path):
    job = _job_cgroup(tmp_path, ooms=0)
    launcher = LocalLauncher(gpus=2, steps_under=job, out_of_memory=("server",))
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", launcher, watch=watch)
    ctx.start_role("server", QUICK, SLICE, owner="contestant")
    _step_events(job, 1, 0)
    assert ctx._roles["server"].wait(10.0) == 1
    _job_cgroup(tmp_path, ooms=0, below=1)
    _gone(job, "step_1")
    ctx._report_role("server", timed_out=False)
    ctx.close()
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert ended["job_ooms_at_start"] == {"own": 0, "below": 0, "steps": {"step_batch": 0}}
    assert ended["job_ooms"] == {"own": 0, "below": 1, "steps": {"step_batch": 0}}
    assert ended["own_oom"] is True


@pytest.mark.parametrize("missing", ["job memory.events.local", "job memory.events", "step oom key", "no job cgroup"])
def test_a_kill_with_a_count_missing_is_infra(tmp_path, missing):
    job = _job_cgroup(tmp_path, ooms=0, local=missing != "job memory.events.local",
                      events=missing != "job memory.events")
    _step_cgroup(tmp_path, "role", oom=None if missing == "step oom key" else 1, oom_kill=1)
    watch = MemoryWatch(enforce=False, job_dir=None if missing == "no job cgroup" else job)
    ctx = _ctx(noop_plan(tmp_path), "noop_a", LocalLauncher(gpus=2, cgroup_root=tmp_path / "cg"), watch=watch)
    if missing != "step oom key":
        _job_cgroup(tmp_path, ooms=0, below=1, local=missing != "job memory.events.local",
                    events=missing != "job memory.events")
    ctx.start_role("role", EXIT_3, SLICE, owner="contestant")
    watch.sample_once()
    [outcome] = ctx.wait_roles()
    fault = classify_role(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)


def test_the_contestant_fault_decided_so_far_needs_every_role_that_ended_reported(tmp_path):
    # The runner writes this fault in its records at a SIGTERM; a role that ended but is not
    # reported yet may still turn the stage's fault into infra.
    job = _job_cgroup(tmp_path, ooms=0)
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", LocalLauncher(gpus=2, steps_under=job, out_of_memory=("server",)),
               watch=watch)
    assert ctx.contestant_fault() is None
    _job_cgroup(tmp_path, ooms=0, below=1)
    ctx.start_role("server", QUICK, SLICE, owner="contestant")
    ctx.wait_roles()
    assert ctx.contestant_fault().reason == "contestant role server ran out of memory"
    driver = ctx.start_role("driver", EXIT_3, SLICE, owner="harness")
    assert driver.wait(10.0) == 3
    assert ctx.contestant_fault() is None
    ctx.wait_roles()
    assert ctx.contestant_fault() is None
    ctx.close()


def _exit_3_once(path) -> list[str]:
    """A command that waits until *path* exists, then exits 3."""
    return [sys.executable, "-c",
            f"import os, sys, time\nwhile not os.path.exists({str(path)!r}): time.sleep(0.02)\nsys.exit(3)"]


def test_a_decided_contestant_fault_waits_for_the_other_roles_to_be_signalled(tmp_path):
    # The stage has reported the contestant's own-limit OOM but not yet signalled the driver,
    # which still runs. If the driver then ends on its own, it is a harness fault and the stage is
    # infra, so the fault a SIGTERM's early records would carry must not be contestant yet.
    job = _job_cgroup(tmp_path, ooms=0)
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", LocalLauncher(gpus=2, cgroup_root=tmp_path / "cg"), watch=watch)
    _step_cgroup(tmp_path, "server", oom=1, oom_kill=1)
    trigger = tmp_path / "go"
    ctx.start_role("server", EXIT_3, SLICE, owner="contestant")
    ctx.start_role("driver", _exit_3_once(trigger), SLICE, owner="harness")
    watch.sample_once()
    assert ctx._roles["server"].wait(10.0) == 3
    ctx._report_role("server", timed_out=False)
    early = ctx.contestant_fault()
    trigger.write_text("")
    assert ctx._roles["driver"].wait(10.0) == 3
    outcomes = ctx.wait_roles()
    final = stage_fault([ctx._outcomes["server"], *outcomes])
    ctx.close()
    assert final.fault_class == "infra" and early is None


def test_a_decided_contestant_fault_stands_once_every_other_role_is_signalled(tmp_path, monkeypatch):
    # A signalled role can only read "stopped", which adds nothing to another role's fault.
    monkeypatch.setattr(stage_module, "STOP_GRACE_SEC", 1.0)
    job = _job_cgroup(tmp_path, ooms=0)
    watch = MemoryWatch(enforce=False, job_dir=job)
    ctx = _ctx(noop_plan(tmp_path), "noop_b", LocalLauncher(gpus=2, cgroup_root=tmp_path / "cg"), watch=watch)
    _step_cgroup(tmp_path, "server", oom=1, oom_kill=1)
    ctx.start_role("server", EXIT_3, SLICE, owner="contestant")
    ctx.start_role("driver", DEAF, SLICE, owner="harness")
    watch.sample_once()
    assert ctx._roles["server"].wait(10.0) == 3
    ctx._report_role("server", timed_out=False)
    ctx._terminate_all()
    decided = ctx.contestant_fault()
    outcomes = ctx.close()
    assert decided is not None and decided.reason == "contestant role server ran out of memory"
    assert [o.ending for o in outcomes] == ["stopped"]
    assert stage_fault([ctx._outcomes["server"], *outcomes]) == decided


def test_the_contestant_role_is_the_first_and_its_step_wrapper_starts_it_with_oom_score_adj_1000(tmp_path):
    plan = _with_params(noop_plan(tmp_path), "noop_b", contestant_role=True)
    launcher = LocalLauncher(gpus=2)
    ctx = _ctx(plan, "noop_b", launcher)
    NoopStage().run(ctx)
    owners = {spec.name: spec.owner for spec in launcher.launched}
    assert owners == {"gpu0": "contestant", "gpu1": "harness", "cpu": "harness", "shared": "harness"}
    ceiling = ctx.process_ceiling().ceiling
    for spec in launcher.launched:
        # The role asks for nothing itself; on Slurm the wrapper sets both for a contestant's role:
        # its score, and the stage's process ceiling (``StageContext.process_ceiling``).
        assert "--oom-score-adj" not in spec.argv and "--nproc" not in spec.argv
        argv = srun_argv(spec, python="p", report=Path("/r"), scratch=Path("/s"), shared=None, allocation_gpus=2)
        wrapper = argv[:argv.index("--")]
        contestant = spec.owner == "contestant"
        assert (wrapper[-4:] == ["--oom-score-adj", "1000", "--nproc", str(ceiling)]) == contestant
        assert ("--oom-score-adj" in wrapper) == ("--nproc" in wrapper) == contestant


def test_batch_memory_is_held_by_a_process_of_its_own_while_the_roles_run(tmp_path, monkeypatch):
    monkeypatch.setattr(noop_module, "BATCH_HOG_DELAY_SEC", 0.0)
    plan = _with_params(noop_plan(tmp_path), "noop_a", batch_hog_gib=1, sleep_sec=2)
    ctx = _ctx(plan, "noop_a", LocalLauncher(gpus=2))
    started = []
    real = subprocess.Popen

    def popen(argv, **kw):
        started.append(real(argv, **kw))
        return started[-1]

    monkeypatch.setattr(noop_module.subprocess, "Popen", popen)
    NoopStage().run(ctx)
    [hog] = [proc for proc in started if str(ctx.attempt_dir / "logs" / "batch-hog.json") in proc.args]
    # It holds its memory for the stage's whole time, so it was still there to be killed when
    # the roles had ended.
    assert hog.args[hog.args.index("--hog-gib") + 1] == "1" and hog.returncode == -9


def test_a_role_that_ends_while_another_is_reported_past_the_deadline_is_not_a_time_limit(tmp_path):
    # Role b ends on its own before the deadline, while the report of role a waits on the
    # scheduler past the deadline; b was last looked at while it still ran. Only the guard that
    # charges a time limit to the roles still running at the deadline keeps b "exited".
    plan = noop_plan(tmp_path)
    launcher = LocalLauncher(gpus=2, scheduler_query_sec=2.0)
    ctx = _ctx(plan, "noop_b", launcher, deadline_in=1.2)
    ctx.start_role("b", [sys.executable, "-c", "import time; time.sleep(0.5)"], SLICE, owner="harness")
    ctx.start_role("a", EXIT_3, SLICE, owner="harness")
    outcomes = {o.name: o for o in ctx.wait_roles()}
    ctx.close()
    assert (outcomes["b"].ending, outcomes["b"].exit_code) == ("exited", 0)
