"""The drive must stop at its own wall clock, not at the orchestrator's.

``request_timeout_sec`` is a per-request floor: ``max(deadline.t - t_send, grace)``
re-arms the full grace for every turn a session sends, and a session's turns are sent
serially, so a trajectory workload has no bound on total drive time. A drive that outlasts its
Slurm wall is killed mid-drive and produces no trace and no score, not even the upper bound
the scorer reports for a run that could not keep up.

These tests pin the total-drive cut-off: the drive stops on its own terms, the turns it
never drove are recorded as faults so the SLO gate zeroes the run, and a fast drive
never sees it.
"""

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest

from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload
from hls.core.model import AcceleratorSpec, Hardware
from hls.core.trace import ArtifactResponse, FailureReason, Trace, TraceEvent
from hls.core.verdict import Verdict
from hls.scoring import SloLatencyScorer
from hls.spec.datasets import Dataset, Split
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import WALL_ERROR, Driver, _Deadline, _DriveWall, harness_now
from hls.spec.tenant import Tenant
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, TrajectoryMakespanSLO,
                               Uncapped, Workload)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from tests.support.drive import BoundDrive


class _NTurnDataset(Dataset):
    """One session of N turns -- the shape the wall exists for. A session's turns are
    sent serially (each waits for the previous reply), so N slow turns take N times as
    long however many sessions run at once, and the per-request grace re-arms on every
    one of them."""
    name = "stub-wall"
    version = "v0"
    N: ClassVar[int] = 1

    def items_at(self, mount_dir, split):
        turns = tuple(Envelope(request_id=f"r0.{k}", contents=[TextPayload(text="Q?")])
                      for k in range(self.N))
        yield DatasetItem(request_id="r0", turns=turns,
                          ground_truth_turns=("A",) * self.N)


def _n_turn_dataset(n):
    return type("_Ds", (_NTurnDataset,), {"N": n})()


def _serial_wl(slo=None):
    """One session at a time, no think time: turns land in a deterministic order."""
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(), slo=slo or NoSLO())


def _engine(*, dataset, drive_timeout_sec, slo=None, request_timeout_sec=REQUEST_TIMEOUT_SEC) -> BoundDrive:
    source = DataSource(dataset=dataset, split=SplitSpec(dev=Count(value=1), test=Count(value=1)))
    return BoundDrive(Driver(drive_timeout_sec=drive_timeout_sec, request_timeout_sec=request_timeout_sec),
                      [Tenant(id="t0", workload=_serial_wl(slo), verifiers=[], data_source=source)])


class _SlowClient:
    """A server that answers correctly but far slower than the drive can afford."""

    def __init__(self, delay_sec: float):
        self.delay_sec = delay_sec
        self.calls = 0

    async def submit(self, env: Envelope, *, attempt=None) -> ArtifactResponse:
        self.calls += 1
        await asyncio.sleep(self.delay_sec)
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text="A")]))


def test_drive_wall_stops_a_slow_drive_and_faults_the_undriven_tail():
    """A server slow enough to outlast the wall stops being driven at the wall, and
    every turn it never reached is recorded as a fault -- so the trace is complete and
    the SLO gate zeroes the run instead of the drive running until Slurm kills it."""
    eng = _engine(dataset=_n_turn_dataset(20), drive_timeout_sec=0.5)
    slow = _SlowClient(delay_sec=0.1)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(slow, events.append, Split.DEV))

    # It stopped on its own terms, well short of the 20 * 0.1s the session needs.
    assert slow.calls < 20
    # The trace is still complete: one event per turn, so nothing vanishes from the mean.
    assert len(events) == 20
    # Every turn that did not answer names the wall -- both the one cut in flight and
    # the tail that was never sent. A failure census that blamed the per-request grace
    # for a wall cut would send the next operator after the wrong number.
    failed = [ev for ev in events if ev.response is None]
    assert failed, "expected the drive to stop short"
    assert all("drive wall" in ev.error for ev in failed), \
        [ev.error for ev in failed if "drive wall" not in ev.error]
    assert len(events) - len(failed) < slow.calls + 1


def test_a_wall_cut_names_one_failure_reason_for_the_whole_drive():
    """One cause, one reason. The wall cuts the request in flight and skips every turn
    behind it; the cut request took the deadline path and the skipped tail took the
    breaker's, so the same event landed under two reasons. A census keyed on reason
    then splits the wall in half and cannot tell a real breaker trip -- a server that
    stopped answering -- from a drive that never kept up."""
    eng = _engine(dataset=_n_turn_dataset(20), drive_timeout_sec=0.5)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_SlowClient(delay_sec=0.1), events.append, Split.DEV))

    walled = [ev for ev in events if "drive wall" in (ev.error or "")]
    assert len(walled) > 1, "expected a cut request and a skipped tail behind it"
    assert {ev.failure.reason for ev in walled} == {FailureReason.DEADLINE}, \
        sorted(str(ev.failure.reason) for ev in walled)


def test_drive_wall_does_not_fire_on_a_drive_that_keeps_up():
    """A server that finishes inside the wall drives every request, with no faults."""
    eng = _engine(dataset=_n_turn_dataset(5), drive_timeout_sec=30.0)
    fast = _SlowClient(delay_sec=0.0)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(fast, events.append, Split.DEV))

    assert fast.calls == 5
    assert len(events) == 5
    assert all(ev.error is None for ev in events)


def test_drive_wall_cuts_a_request_still_in_flight_at_the_wall():
    """A single request that would outlive the wall is cut at the wall, not left to run
    out the much longer per-request grace."""
    eng = _engine(dataset=_n_turn_dataset(1), drive_timeout_sec=0.3,
                  request_timeout_sec=3600.0)          # the grace must not win
    stuck = _SlowClient(delay_sec=30.0)
    events: list[TraceEvent] = []

    async def _run():
        return await eng.drive(stuck, events.append, Split.DEV)

    started = asyncio.run(_timed(_run()))
    assert started < 5.0, "the drive outlived its wall"
    assert len(events) == 1
    assert events[0].response is None and events[0].error


def test_the_deadline_clamps_to_the_wall_the_drive_opened():
    """One wall per drive, held once. The deadline is handed the drive's own
    ``_DriveWall`` rather than a copy of its instant, and answers with the wall's time
    and the wall's fault string: a per-request grace, however large, cannot outlast the
    drive, and a request filed under the grace it never reached sends the next operator
    after a bound that had thousands of seconds left."""
    now = harness_now()
    wall = _DriveWall(t=now + 0.25)
    deadline = _Deadline(t=now + 3600.0, grace=3600.0, error="grace", wall=wall)

    allowed, error = deadline.allowance(now)
    assert allowed == pytest.approx(0.25) and error == WALL_ERROR
    # Past the wall there is nothing left to grant, whatever the grace says.
    assert deadline.allowance(now + 1.0) == (0.0, WALL_ERROR)
    # No wall: the grace stands, which is the unbounded drive below.
    assert _Deadline(t=now + 10.0, grace=5.0, error="grace").allowance(now) == \
        (pytest.approx(10.0), "grace")


def test_drive_wall_unset_leaves_the_drive_unbounded():
    """With no wall configured the per-request grace alone bounds the drive, so nothing
    is cut."""
    eng = _engine(dataset=_n_turn_dataset(4), drive_timeout_sec=None)
    fast = _SlowClient(delay_sec=0.0)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(fast, events.append, Split.DEV))

    assert fast.calls == 4
    assert all(ev.error is None for ev in events)


def test_a_walled_drive_scores_zero_carrying_the_upper_bound_it_reached_for():
    """The point of stopping at the wall: the run is scored rather than killed. A drive
    the wall cut zeroes on the SLO gate -- it did not finish -- but the record still
    carries ``upper_bound``, the score it would have earned had every unfinished
    trajectory answered the moment the harness stopped watching it. A drive killed by
    the orchestrator's own Slurm wall writes neither number."""
    eng = _engine(dataset=_n_turn_dataset(20), drive_timeout_sec=0.5,
                  slo=TrajectoryMakespanSLO(target_sec=10.0))
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_SlowClient(delay_sec=0.1), events.append, Split.DEV))

    trace = Trace(events=events)
    tenants = [SimpleNamespace(id="t0", workload=eng.tenants[0].workload)]
    verdict = Verdict(passed=True, per_event={ev.event_key: True for ev in events})
    score = asyncio.run(SloLatencyScorer().score(
        tenants, trace, verdict,
        Hardware(accelerator=AcceleratorSpec(kind="H200", count=1))))

    assert score.value == 0.0, "a drive that did not finish must not earn a score"
    assert score.zero_reasons == ("slo",)
    # The diagnosis survives the zero: a positive bound on what it was reaching for.
    assert score.breakdown["upper_bound"] > 0.0


async def _timed(coro) -> float:
    """Run ``coro`` and return its wall-clock duration in seconds."""
    import time
    t0 = time.monotonic()
    await coro
    return time.monotonic() - t0
