"""An artifact fault must become a failed event, not escape the trajectory drive.

``ArtifactFault`` documents itself as "caught at the boundary -> failed TraceEvent",
and the base ``Driver`` honours it in ``_submit_or_fault``. ``TrajectorySessionDriver``
must too, or a contestant server that crashed, ran out of KV pool or stopped answering
inside the 10,800 s request cap would take the whole rollout down: no trace, no
verdict, no score, and no ``upper_bound`` -- the number that exists precisely to say
what a run that could not keep up was reaching for.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from types import SimpleNamespace

from hls.core.dataitem import DatasetItem
from hls.core.faults import ArtifactTransportError
from hls.core.model import AcceleratorSpec, Hardware
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.core.verdict import Verdict
from hls.scoring import SloLatencyScorer
from hls.spec.agent import AgentFramework
from hls.spec.datasets import Dataset, Split
from hls.spec.drivers import TrajectorySessionDriver
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.workload import (AllAtOnce, NoDelay, TrajectoryMakespanSLO, Uncapped,
                               Workload)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from tests.support.drive import BoundDrive


class _ListDataset(Dataset):
    name = "stub-fault"
    version = "v0"
    n: int

    def items_at(self, mount_dir, split):
        for i in range(self.n):
            env = Envelope(request_id=f"r{i}", contents=[TextPayload(text="Q?")])
            yield DatasetItem(request_id=f"r{i}", turns=(env,), ground_truth_turns=("A",))

    def size(self, split):
        return self.n


class _DiesOnTurn:
    """A server that answers a few turns, then faults the way a crashed or wedged
    contestant does: the same ``ArtifactFault`` the 10,800 s request cap raises."""

    def __init__(self, ok_turns: int):
        self.ok_turns = ok_turns
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        if self.calls > self.ok_turns:
            raise ArtifactTransportError("request exceeded 10800.0s")
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text="A")]))


class _HangsThenFaults:
    """A server that takes the request, holds it, and only then faults -- the shape of
    a wedged contestant reaching the 10,800 s request cap. The hold is the time the
    trajectory actually spent, and the only record of it is the turn's send time."""

    def __init__(self, hang_sec: float):
        self.hang_sec = hang_sec
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        await asyncio.sleep(self.hang_sec)
        raise ArtifactTransportError("request exceeded 10800.0s")


class _ManyTurnFramework(AgentFramework):
    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx,
                             max_turns):
        for k in range(max_turns):
            t_send = time.monotonic()
            env = Envelope(request_id=f"{seed_item.request_id}.{k}",
                           contents=[TextPayload(text="Q?")],
                           metadata={"turn_index": k,
                                     "session_id": seed_item.request_id,
                                     # Production stamps this from HarnessContext;
                                     # Trace.events_for has no None fallback.
                                     "tenant_id": ctx.tenant_id})
            resp = await llm_call(env)
            await on_event(TraceEvent(
                request_id=f"{seed_item.request_id}.{k}",
                session_id=seed_item.request_id, turn_index=k,
                input=env, response=resp, ground_truth=None,
                t_send=t_send, t_recv=time.monotonic()))


def _driver(*, n_items, max_turns, target_sec: float = 10.0) -> BoundDrive:
    driver = TrajectorySessionDriver(
        framework=_ManyTurnFramework(),
        tool_runner=LocalCallableToolRunner(specs=[ToolSpec(name="search", description="d")],
                                            callables={"search": lambda: {}}),
        max_turns=max_turns, max_concurrent_llm=None, request_timeout_sec=REQUEST_TIMEOUT_SEC)
    tenants = [SimpleNamespace(
        id="t0",
        data_source=SimpleNamespace(dataset=_ListDataset(n=n_items), removed_rows=()),
        workload=Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(),
                          slo=TrajectoryMakespanSLO(target_sec=target_sec)))]
    return BoundDrive(driver, tenants, seed=42)


def _drive(drv: BoundDrive, client):
    events: list[TraceEvent] = []
    asyncio.run(drv.drive(client, events.append, Split.DEV))
    return events


def test_an_artifact_fault_is_recorded_not_raised():
    """The drive returns a complete trace with the fault recorded on the turn that
    hit it, instead of letting the exception end the rollout."""
    drv = _driver(n_items=1, max_turns=6)

    events = _drive(drv, _DiesOnTurn(ok_turns=2))

    assert len(events) >= 3, "the faulted turn is missing from the trace"
    failed = [ev for ev in events if ev.response is None]
    assert failed, "the artifact fault produced no failed event"
    assert any("transport" in (ev.error or "") for ev in failed), \
        [ev.error for ev in failed]


def test_a_faulted_trajectory_still_scores_with_an_upper_bound():
    """The point of recording it: the run is gated, and the record still carries the
    score it was reaching for. A fault that escapes the drive produces neither."""
    drv = _driver(n_items=2, max_turns=6)

    events = _drive(drv, _DiesOnTurn(ok_turns=3))

    trace = Trace(events=events)
    tenants = [SimpleNamespace(id="t0", workload=drv.tenants[0].workload)]
    verdict = Verdict(passed=True, per_event={ev.event_key: True for ev in events})
    score = asyncio.run(SloLatencyScorer().score(
        tenants, trace, verdict,
        Hardware(accelerator=AcceleratorSpec(kind="H200", count=1))))

    assert score.value == 0.0
    assert score.zero_reasons == ("slo",)
    assert score.breakdown["upper_bound"] > 0.0
    assert math.isfinite(score.breakdown["upper_bound"])


def test_a_faulted_turn_is_stamped_with_when_it_was_sent():
    """``t_send`` is the moment the request went out, not the moment the harness gave
    up on it. The driver takes the send time before the submit and holds it; an event
    stamped at the fault reports a turn that took no time at all, and the elapsed time
    of a failure has nowhere else to live -- ``t_recv`` stays None."""
    drv = _driver(n_items=1, max_turns=6)

    events = _drive(drv, _HangsThenFaults(hang_sec=0.3))

    [ev] = events
    assert ev.response is None and ev.failure is not None
    waited = ev.failure.t_abandoned - ev.t_send
    assert waited >= 0.25, f"a turn that waited 0.3s was recorded as {waited:.3f}s"


def test_a_trajectory_that_hung_is_priced_on_the_time_it_hung():
    """What that send time is for. ``censored_ratio`` charges a trajectory with no
    measurable makespan the span from when its first turn was due to when the harness
    stopped watching its last. Stamp both ends at the fault and the span is zero, so
    the trajectory is charged the floor of r = 1 -- the most lenient price there is,
    handed to the run that spent the longest -- and ``upper_bound`` is overstated."""
    drv = _driver(n_items=1, max_turns=6, target_sec=0.1)

    events = _drive(drv, _HangsThenFaults(hang_sec=0.3))

    trace = Trace(events=events)
    slo = drv.tenants[0].workload.slo
    priced = slo.censored_ratio(slo.units(trace)["r0"], trace.t_end())
    assert priced > 2.0, f"0.3s against a 0.1s target was priced at r = {priced}"


def test_a_healthy_trajectory_drive_records_no_faults():
    """The healthy path is unchanged: nothing faults, nothing is recorded."""
    drv = _driver(n_items=2, max_turns=3)

    events = _drive(drv, _DiesOnTurn(ok_turns=1000))

    assert len(events) == 6
    assert all(ev.response is not None and ev.error is None for ev in events)


# --- a dead sandbox drops its own session, never the whole drive ------------
# The server here is healthy; it is the served agent's own container that runs out of memory
# or wedges, which a framework reports as SandboxGone. One such session must not end a drive,
# in calibration or in grading: the driver drops it, keeps driving the rest, and logs the drop.

class _SandboxDiesForOne(AgentFramework):
    """A framework whose sandbox dies on one named session after ``ok_turns`` served turns
    (``die_session=None`` dies for every session). It emits the turns it served and then raises
    ``SandboxGone``. Every session it does not kill runs to its budget."""

    def __init__(self, die_session: str | None, ok_turns: int):
        self.die_session = die_session
        self.ok_turns = ok_turns

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        from hls.core.faults import SANDBOX_GONE, SandboxGone

        dies = self.die_session is None or seed_item.request_id == self.die_session
        for k in range(self.ok_turns if dies else max_turns):
            t_send = time.monotonic()
            env = Envelope(request_id=f"{seed_item.request_id}.{k}", contents=[TextPayload(text="Q?")],
                           metadata={"turn_index": k, "session_id": seed_item.request_id,
                                     "tenant_id": ctx.tenant_id})
            resp = await llm_call(env)
            await on_event(TraceEvent(
                request_id=f"{seed_item.request_id}.{k}", session_id=seed_item.request_id, turn_index=k,
                input=env, response=resp, ground_truth=None, t_send=t_send, t_recv=time.monotonic()))
        if dies:
            raise SandboxGone(f"{SANDBOX_GONE} session='{seed_item.request_id}' restarts=5 restart_budget=5 "
                              "last_error=MemoryLimitExceededError")


def _sessions(events) -> dict[str, list]:
    by_session: dict[str, list] = {}
    for ev in events:
        by_session.setdefault(ev.session_id, []).append(ev)
    return by_session


def test_a_sandbox_gone_session_is_dropped_and_the_drive_keeps_driving(caplog):
    """One session's sandbox dies: the drive does not raise, the other sessions run to their
    budget, the dropped session keeps the turns it served, and no fault is recorded for it."""
    drv = _driver(n_items=3, max_turns=4)
    drv.driver.framework = _SandboxDiesForOne(die_session="r1", ok_turns=2)

    with caplog.at_level(logging.WARNING, logger="hls.spec.drivers"):
        events = _drive(drv, _DiesOnTurn(ok_turns=1000))

    by_session = _sessions(events)
    assert {sid: len(evs) for sid, evs in by_session.items()} == {"r0": 4, "r1": 2, "r2": 4}
    assert all(ev.response is not None and ev.error is None for ev in events)
    assert any("r1" in r.getMessage() and "dropped" in r.getMessage() for r in caplog.records)
    # What the drive role writes into the trace's END.json: the session, its tenant and why.
    [dropped] = drv.driver.dropped_sessions
    assert (dropped.session_id, dropped.tenant_id) == ("r1", "t0")
    assert dropped.reason.startswith("sandbox gone: session='r1' restarts=5")


def test_a_sandbox_that_never_started_fails_the_drive_rather_than_dropping_its_session():
    """A container that did not start in time is the harness's failure, so the drive fails as
    infrastructure. Dropping the session would score the run on fewer sessions than it was asked
    for, for a reason that has nothing to do with the server."""
    import pytest

    from hls.core.faults import SandboxNotStarted

    class _NeverStarts(AgentFramework):
        async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
            if seed_item.request_id == "r1":
                raise SandboxNotStarted(f"sandbox not started: session='{seed_item.request_id}' did not start "
                                        "within 600s; the container was released")
            await _ManyTurnFramework().run_trajectory(seed_item, llm_call, tool_runner, on_event, ctx, max_turns)

    drv = _driver(n_items=3, max_turns=2)
    drv.driver.framework = _NeverStarts()
    with pytest.raises(SandboxNotStarted, match="r1"):
        _drive(drv, _DiesOnTurn(ok_turns=1000))
    assert drv.driver.dropped_sessions == ()


def test_a_sandbox_gone_under_a_wall_and_a_scheduled_stop_is_dropped_too():
    """The trajectory is awaited under the wall, or under a scheduled stop inside the wall; the
    drop holds on each."""
    from hls.spec.agent import ScheduledStop

    for wall, stop in ((600.0, None), (1000.0, ScheduledStop(wall_clock_sec=30.0, cut_grace_sec=5.0))):
        drv = _driver(n_items=2, max_turns=3)
        drv.driver.framework = _SandboxDiesForOne(die_session="r0", ok_turns=1)
        drv.driver.drive_timeout_sec, drv.driver.scheduled_stop = wall, stop

        events = _drive(drv, _DiesOnTurn(ok_turns=1000))

        assert {sid: len(evs) for sid, evs in _sessions(events).items()} == {"r0": 1, "r1": 3}
        assert all(ev.error is None for ev in events)


def test_every_sandbox_gone_still_returns_a_trace(caplog):
    """There is no cap on drops: a model that exhausts every sandbox yields a trace of what was
    served, with the count of drops in the log."""
    drv = _driver(n_items=3, max_turns=4)
    drv.driver.framework = _SandboxDiesForOne(die_session=None, ok_turns=1)

    with caplog.at_level(logging.WARNING, logger="hls.spec.drivers"):
        events = _drive(drv, _DiesOnTurn(ok_turns=1000))

    assert {sid: len(evs) for sid, evs in _sessions(events).items()} == {"r0": 1, "r1": 1, "r2": 1}
    assert all(ev.response is not None and ev.error is None for ev in events)
    assert any("3 of 3 session(s) dropped" in r.getMessage() for r in caplog.records)
    assert sorted(d.session_id for d in drv.driver.dropped_sessions) == ["r0", "r1", "r2"]


def test_a_long_drop_reason_is_cut_to_what_the_trace_keeps():
    from hls.core.faults import SandboxGone
    from hls.core.trace import DROPPED_REASON_MAX_CHARS

    class _LongReason(AgentFramework):
        async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
            raise SandboxGone("sandbox gone: " + "x" * 5000)

    drv = _driver(n_items=1, max_turns=1)
    drv.driver.framework = _LongReason()
    _drive(drv, _DiesOnTurn(ok_turns=1000))
    [dropped] = drv.driver.dropped_sessions
    assert len(dropped.reason) == DROPPED_REASON_MAX_CHARS and dropped.reason.endswith("...")


def test_a_drive_that_dropped_nothing_records_no_drop():
    drv = _driver(n_items=2, max_turns=2)
    _drive(drv, _DiesOnTurn(ok_turns=1000))
    assert drv.driver.dropped_sessions == ()


def test_a_harness_fault_that_is_not_a_dead_sandbox_still_ends_the_drive():
    """Only ``SandboxGone`` is absorbed per session; any other harness fault is ours and still
    ends the drive, so it is never scored as a shorter trajectory."""
    import pytest

    from hls.core.faults import HarnessFault

    class _Broken(AgentFramework):
        async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
            raise HarnessFault("the corpus row names no task")

    drv = _driver(n_items=2, max_turns=1)
    drv.driver.framework = _Broken()
    with pytest.raises(HarnessFault, match="names no task"):
        _drive(drv, _DiesOnTurn(ok_turns=1000))
