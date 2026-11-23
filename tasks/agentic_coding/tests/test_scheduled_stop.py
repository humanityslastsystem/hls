"""The task's end of the drive's scheduled stop.

The harness bounds the drive in wall-clock time and raises into the loop's own model
call when the bound passes. The loop has to name that ending, or a trajectory the
harness stopped cleanly is filed as one that crashed -- which is what the diagnostics
would then report for every session still running at the end of the window.
"""

from __future__ import annotations

import pytest

from hls.spec.agent import DriveWallClockReached

from tasks.agentic_coding import metric as metric_mod
from tasks.agentic_coding import task as task_mod
from tasks.agentic_coding.mini_swe_loop import (
    EXIT_STATUS_TO_STOP_REASON, STOP_WALL_CLOCK,
)


def test_the_task_declares_its_window():
    """The two numbers the brief states are the two the drive runs on, so a reader
    cannot find one in the brief and another in the drive."""
    assert task_mod.AGENT.wall_clock_sec == task_mod.DRIVE_WALL_CLOCK_SEC == 45 * 60
    assert task_mod.AGENT.cut_grace_sec == task_mod.DRIVE_CUT_GRACE_SEC == 10 * 60


def test_the_task_declares_a_stall_budget():
    assert task_mod.AGENT.stall_timeout_sec == task_mod.STALL_TIMEOUT_SEC == 120


def test_the_brief_states_both_rules_in_full():
    """Contestants are held to the phantom token and the stall, so both are published:
    a rule a contestant cannot read is one they cannot diagnose a failure against."""
    from tests.unit.brief_task import render_brief

    brief, _ = render_brief(task_mod.TASK, task_mod.ARTIFACT.brief)
    # Read with the line wrapping flattened: what has to hold is the wording, not
    # where the paragraph happens to break.
    flat = " ".join(brief.split())

    assert "as if one more token completed at the cut instant" in flat
    assert "charged the whole wait as TTFT" in flat
    assert "is charged the span from its first token to the cut over n intervals" in flat
    assert (f"first token, {task_mod.STALL_TIMEOUT_SEC} seconds without a further token "
            "fails the request") in flat
    assert "Time to first token has no limit" in flat


def test_the_drive_wall_sits_behind_the_cut():
    """``drive_timeout_sec`` records the remaining work as faults, so one firing before
    the cut would file a drive that stopped on schedule as a failed one."""
    stop = task_mod.AGENT.scheduled_stop()
    assert stop is not None
    timeout = task_mod.AGENT.drive_timeout_sec
    assert timeout is None or timeout > stop.cut_at(0.0)


def test_the_backstop_is_derived_rather_than_declared():
    """Three numbers that are one relationship is how a drift goes unnoticed. The task
    declares the window and the grace; the backstop is those two plus the margin, and
    declaring it here would be a third to keep in step with them."""
    from hls.spec.agent import BACKSTOP_MARGIN_SEC
    from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
    from hls.spec.drivers import driver_for

    assert task_mod.AGENT.drive_timeout_sec is None

    planned_end = task_mod.DRIVE_WALL_CLOCK_SEC + task_mod.DRIVE_CUT_GRACE_SEC
    assert driver_for(task_mod.AGENT, request_timeout_sec=REQUEST_TIMEOUT_SEC).drive_timeout_sec == (
        planned_end + BACKSTOP_MARGIN_SEC)


def test_the_backstop_leaves_the_cut_room_to_finish():
    """The margin is not slack. The cut instant is when the last stream is cancelled,
    and the drive still has to price each cut request, unwind a container per session
    and flush the trace after it -- all past the clock the stop is measured against."""
    from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
    from hls.spec.drivers import driver_for

    planned_end = task_mod.DRIVE_WALL_CLOCK_SEC + task_mod.DRIVE_CUT_GRACE_SEC
    assert driver_for(task_mod.AGENT, request_timeout_sec=REQUEST_TIMEOUT_SEC).drive_timeout_sec > planned_end


def test_the_stage_plans_the_drive_on_the_backstop_the_driver_keeps():
    """The drive stage holds the wall the planner names, so the planner, the driver and the
    drive's record name one number: the backstop, not the open wall of a task that declares
    none."""
    from hls.pipeline.pipelines import OPEN_DRIVE_SEC, REQUEST_TIMEOUT_SEC, drive_wall_sec
    from hls.spec.drivers import driver_for

    wall = drive_wall_sec(task_mod.TASK)
    assert wall == task_mod.AGENT.drive_wall_sec() < OPEN_DRIVE_SEC
    driver = driver_for(task_mod.AGENT, request_timeout_sec=REQUEST_TIMEOUT_SEC, drive_timeout_sec=wall)
    assert driver.drive_timeout_sec == wall
    assert driver.scheduled_stop == task_mod.AGENT.scheduled_stop()
    assert driver.stall_timeout_sec == task_mod.STALL_TIMEOUT_SEC


def test_the_loop_names_the_harness_exception_the_wall_clock_stop():
    """The loop reads an ending off the exit status upstream stamps, which for an
    uncaught exception is its class name. So the harness's own class name is the key
    this map needs, and this is what holds the two together across a boundary the loop
    cannot import across."""
    assert EXIT_STATUS_TO_STOP_REASON[DriveWallClockReached.__name__] == STOP_WALL_CLOCK


def test_a_trajectory_the_wall_clock_ended_is_not_reported_as_a_crash():
    """Without the entry above it falls through to the error ending, which says the
    trajectory raised on its way out."""
    from tasks.agentic_coding.mini_swe_loop import STOP_ERROR

    assert STOP_WALL_CLOCK != STOP_ERROR
    assert EXIT_STATUS_TO_STOP_REASON.get("SomeOtherError") is None


def _shape(stop_reasons: list[str]):
    """The trajectory-shape result over one one-turn session per stop reason."""
    from hls.core.payloads import Envelope, TextPayload
    from hls.core.trace import ArtifactResponse, TraceEvent

    events = []
    for i, reason in enumerate(stop_reasons):
        inp = Envelope(request_id=f"s{i}.0", contents=[TextPayload(text="q")],
                       metadata={"tenant_id": "t0"})
        out = Envelope(request_id=f"s{i}.0", contents=[TextPayload(text="a")])
        events.append(TraceEvent(
            request_id=f"s{i}.0", session_id=f"s{i}", turn_index=0, input=inp,
            response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0,
            extras={"stop_reason": reason, "commands_run": 1, "session_turns": 1}))
    return metric_mod.TrajectoryShapeMetric().score(events)


def test_the_diagnostics_count_the_trajectories_the_wall_clock_stopped():
    """How many sessions were still working when the window closed is what says
    whether the drive measured whole trajectories or a slice of them."""
    d = _shape([STOP_WALL_CLOCK, STOP_WALL_CLOCK, "submitted"]).diagnostics
    assert d["stopped_on_the_wall_clock"] == 2
    assert d["submitted"] == 1


def test_a_wall_clock_stop_is_not_counted_as_an_error():
    d = _shape([STOP_WALL_CLOCK]).diagnostics
    assert d["stop_reasons"] == {STOP_WALL_CLOCK: 1}
    assert d["submitted"] == 0


def test_the_harness_exception_is_an_exception_the_agent_loop_can_see():
    """Upstream's agent loop catches ``Exception`` to stamp an exit status. A
    BaseException would go straight past it and the ending would never be named."""
    assert issubclass(DriveWallClockReached, Exception)
    with pytest.raises(Exception):
        raise DriveWallClockReached("x")
