"""The two waits between a role's launch and its grant report: for its step to start, from the
launch request (``STEP_START_TIMEOUT_SEC``), cut short by the allocation's end, then for its
report, from the step's start (``REPORT_TIMEOUT_SEC``), cut short by the stage's deadline; and
the stage's deadline, which grows by each role's wait for its step to start, up to the
allocation's end. The steps here wait on a fake clock, so a wait of half an hour takes none."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

from hls.pipeline.events import EventLog, read_events
from hls.pipeline.stage import REPORT_TIMEOUT_SEC, STEP_START_TIMEOUT_SEC, StageContext, stage_fault
from hls.spec.resources import ResourceShape
from hls.store.posix import PosixStore
from tests.pipeline.fakes import FakeClock, LocalLauncher
from tests.pipeline.plans import noop_plan

SLICE = ResourceShape(cpus=1, memory_gib=1, time_sec=60)
QUICK = [sys.executable, "-c", "pass"]
#: What a launcher on Slurm says of a step srun is still waiting for.
WAITING = {"slurm_steps": "none", "srun_said": "srun: Job 100 step creation temporarily disabled, retrying "
                                                "(Requested nodes are busy)"}


def _ctx(tmp_path: Path, launcher: LocalLauncher, clock, *, deadline_in: float,
         cap_in: float | None = None) -> StageContext:
    plan = noop_plan(tmp_path)
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("noop_a", 1, "events.jsonl")), clock=clock)
    return StageContext(store=store, plan=plan, stage=plan.stage("noop_a"), attempt=1, launcher=launcher,
                        watch=None, events=events, inputs={}, deadline=clock() + deadline_in, clock=clock,
                        deadline_cap=None if cap_in is None else clock() + cap_in,
                        scratch=Path(plan.settings.site.node_scratch) / "hls-100", job_id="100")


def _events(ctx: StageContext, kind: str) -> list[dict]:
    return [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == kind]


def test_a_step_that_starts_600_s_after_its_launch_still_runs_and_reports(tmp_path):
    clock = FakeClock()
    ctx = _ctx(tmp_path, LocalLauncher(gpus=2, clock=clock, start_after={"late": 600.0}), clock,
               deadline_in=3 * 3600.0)
    asked = clock()
    ctx.start_role("late", QUICK, SLICE, owner="harness")
    assert clock() - asked == pytest.approx(600.0, abs=1.0)
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("exited", 0) and stage_fault([outcome]) is None
    [started] = _events(ctx, "role_started")
    assert started["launch_wait_sec"] == pytest.approx(600.0, abs=1.0)
    assert started["report_wait_sec"] == pytest.approx(0.0, abs=1.0)
    assert _events(ctx, "role_launch_failed") == []


def test_a_step_that_never_starts_is_infra_once_the_start_wait_ends_with_slurms_word_on_it(tmp_path):
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"stuck": math.inf}, launch_evidence=WAITING)
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=3 * STEP_START_TIMEOUT_SEC)
    asked = clock()
    ctx.start_role("stuck", QUICK, SLICE, owner="harness")
    assert clock() - asked == pytest.approx(STEP_START_TIMEOUT_SEC, abs=1.0)
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "launch_failed"
    fault = stage_fault([outcome])
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    detail = fault.evidence["stuck.detail"]
    assert f"did not start within {STEP_START_TIMEOUT_SEC:.0f} s of its launch" in detail
    assert "slurm_steps: none" in detail and "Requested nodes are busy" in detail
    [failed] = _events(ctx, "role_launch_failed")
    assert failed["launch_wait_sec"] == pytest.approx(STEP_START_TIMEOUT_SEC, abs=1.0)
    assert failed["report_wait_sec"] is None and failed["evidence"] == WAITING
    assert _events(ctx, "role_started") == []


def test_a_step_that_started_but_never_reports_is_infra_once_the_report_wait_from_its_start_ends(tmp_path):
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"mute": 100.0}, unreported=("mute",),
                             launch_evidence={"slurm_steps": "100.3 RUNNING", "srun_said": "nothing"})
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=3 * STEP_START_TIMEOUT_SEC)
    asked = clock()
    ctx.start_role("mute", QUICK, SLICE, owner="harness")
    assert clock() - asked == pytest.approx(100.0 + REPORT_TIMEOUT_SEC, abs=2.0)
    [outcome] = ctx.wait_roles()
    fault = stage_fault([outcome])
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    detail = fault.evidence["mute.detail"]
    assert f"did not report its grant within {REPORT_TIMEOUT_SEC:.0f} s of its start" in detail
    assert "slurm_steps: 100.3 RUNNING" in detail
    [failed] = _events(ctx, "role_launch_failed")
    assert failed["launch_wait_sec"] == pytest.approx(100.0, abs=1.0)
    assert failed["report_wait_sec"] == pytest.approx(REPORT_TIMEOUT_SEC, abs=1.0)


def test_a_stage_whose_roles_start_late_gets_its_deadline_extended_by_exactly_their_waits(tmp_path):
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"a": 300.0, "b": 450.0})
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=1000.0)
    planned = ctx.deadline
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    assert ctx.deadline == pytest.approx(planned + 300.0, abs=1.0)
    ctx.start_role("b", QUICK, SLICE, owner="harness")
    assert ctx.deadline == pytest.approx(planned + 750.0, abs=1.0)
    assert ctx.launch_waits() == pytest.approx({"a": 300.0, "b": 450.0}, abs=1.0)
    assert [(o.ending, o.exit_code) for o in ctx.wait_roles()] == [("exited", 0), ("exited", 0)]
    assert [e["deadline"] for e in _events(ctx, "role_started")] == pytest.approx([planned + 300, planned + 750],
                                                                                    abs=1.0)


def test_a_role_that_starts_at_once_extends_the_deadline_by_nothing(tmp_path):
    clock = FakeClock()
    ctx = _ctx(tmp_path, LocalLauncher(gpus=2, clock=clock), clock, deadline_in=1000.0)
    planned = ctx.deadline
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    assert ctx.deadline == planned and ctx.launch_waits() == {"a": 0.0}


def test_the_deadline_grows_no_further_than_the_allocations_end_and_never_shrinks(tmp_path):
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"a": 300.0, "b": 10.0})
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=1000.0, cap_in=1100.0)
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    assert ctx.deadline == ctx.deadline_cap
    # A stage whose deadline already lies past the cap keeps it.
    ctx.deadline = ctx.deadline_cap + 50.0
    ctx.start_role("b", QUICK, SLICE, owner="harness")
    assert ctx.deadline == ctx.deadline_cap + 50.0


def test_the_stage_deadline_does_not_end_a_start_wait(tmp_path):
    clock = FakeClock()
    ctx = _ctx(tmp_path, LocalLauncher(gpus=2, clock=clock, start_after={"late": 600.0}), clock, deadline_in=120.0)
    planned = ctx.deadline
    ctx.start_role("late", QUICK, SLICE, owner="harness")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("exited", 0)
    assert ctx.deadline == pytest.approx(planned + 600.0, abs=1.0)


def test_the_allocations_end_ends_a_start_wait(tmp_path):
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"stuck": math.inf}, launch_evidence=WAITING)
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=60.0, cap_in=120.0)
    ctx.start_role("stuck", QUICK, SLICE, owner="harness")
    assert clock() == ctx.deadline_cap
    fault = stage_fault(ctx.wait_roles())
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert "did not start within 120 s of its launch (the allocation's end came first)" in fault.evidence[
        "stuck.detail"]


def test_the_extended_deadline_ends_a_report_wait(tmp_path):
    # Started 50 s late with 120 s of stage time, it has until the deadline grown by 50 s.
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"mute": 50.0}, unreported=("mute",))
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=120.0)
    asked = clock()
    ctx.start_role("mute", QUICK, SLICE, owner="harness")
    assert clock() == pytest.approx(asked + 170.0, abs=1.0) and ctx.deadline == pytest.approx(asked + 170.0, abs=1.0)
    fault = stage_fault(ctx.wait_roles())
    assert fault is not None and "within 120 s of its start (the stage's deadline came first)" in fault.evidence[
        "mute.detail"]


def test_a_launch_past_the_allocations_end_looks_once_and_waits_for_nothing(tmp_path):
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"slow": 5.0})
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=-1.0, cap_in=-1.0)
    asked = clock()
    ctx.start_role("slow", QUICK, SLICE, owner="harness")
    assert clock() == asked
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "launch_failed"


def test_a_role_that_starts_at_once_records_both_waits(tmp_path):
    ctx = _ctx(tmp_path, LocalLauncher(gpus=2), time.time, deadline_in=60.0)
    ctx.start_role("quick", QUICK, SLICE, owner="harness")
    ctx.wait_roles()
    [started] = _events(ctx, "role_started")
    assert 0.0 <= started["launch_wait_sec"] < 5.0 and 0.0 <= started["report_wait_sec"] < 5.0


def test_a_roles_step_may_run_until_the_allocations_end_whatever_its_declared_time(tmp_path):
    # Its own Slurm time limit never ends it before the stage's deadline, however far that grew.
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock)
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=100.0, cap_in=5000.5)
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    assert launcher.launched[-1].shape == SLICE.model_copy(update={"time_sec": 5001})
    # With no allocation's end to keep to, a step has its declared time.
    bare = LocalLauncher(gpus=2, clock=clock)
    _ctx(tmp_path / "bare", bare, clock, deadline_in=100.0).start_role("a", QUICK, SLICE, owner="harness")
    assert bare.launched[-1].shape == SLICE


def test_a_launch_at_or_past_the_allocations_end_still_asks_slurm_for_a_second(tmp_path):
    # A time limit of none or less is no limit Slurm takes; the shape's copy is not validated.
    clock = FakeClock()
    launcher = LocalLauncher(gpus=2, clock=clock)
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=100.0, cap_in=-10.0)
    ctx.start_role("a", QUICK, SLICE, owner="harness")
    assert launcher.launched[-1].shape.time_sec == 1


def test_a_step_that_starts_late_in_its_start_wait_still_gets_the_whole_report_wait(tmp_path):
    # Started 100 s before the start wait would end, it still has REPORT_TIMEOUT_SEC to report.
    clock = FakeClock()
    late = STEP_START_TIMEOUT_SEC - 100.0
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"late": late}, unreported=("late",))
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=3 * STEP_START_TIMEOUT_SEC)
    asked = clock()
    ctx.start_role("late", QUICK, SLICE, owner="harness")
    assert clock() - asked == pytest.approx(late + REPORT_TIMEOUT_SEC, abs=2.0)
    [failed] = _events(ctx, "role_launch_failed")
    assert failed["report_wait_sec"] == pytest.approx(REPORT_TIMEOUT_SEC, abs=1.0)


class _StopsAfter(FakeClock):
    """A clock that tells its stage context to stop once *after* seconds have passed on it, as
    the runner does on a SIGTERM."""

    def __init__(self, after: float) -> None:
        super().__init__()
        #: When the stop comes, on this clock.
        self.stop_at = self.now + after
        #: The context told to stop; None until the test sets it.
        self.ctx: StageContext | None = None

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        if self.ctx is not None and self.now >= self.stop_at:
            self.ctx.stopping = True


def test_a_stop_during_the_start_wait_asks_the_scheduler_nothing(tmp_path):
    clock = _StopsAfter(30.0)
    launcher = LocalLauncher(gpus=2, clock=clock, start_after={"stuck": math.inf}, launch_evidence=WAITING)
    ctx = _ctx(tmp_path, launcher, clock, deadline_in=3 * STEP_START_TIMEOUT_SEC)
    clock.ctx = ctx
    ctx.start_role("stuck", QUICK, SLICE, owner="harness")
    [failed] = _events(ctx, "role_launch_failed")
    assert "the stage is stopping" in failed["error"] and failed["evidence"] == {}
    assert launcher.handles[0].evidence_asked == 0
    assert failed["launch_wait_sec"] == pytest.approx(30.0, abs=1.0)
