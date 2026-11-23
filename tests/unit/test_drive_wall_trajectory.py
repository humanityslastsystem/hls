"""The drive wall must bound the trajectory driver, which is the one graded runs use.

`driver_for` returns `TrajectorySessionDriver` for any task with an agent seam, and that
subclass owns its own `_drive`/`_drive_sessions`. A wall implemented only in the base
`Driver` loop is dead code for every agentic task, whose drive then runs past the wall
until something else stops it, with no trace and no score.

These tests drive the real subclass. A wall test that instantiates a bare `Driver`
proves nothing about a graded run.
"""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.spec.agent import AgentFramework
from hls.spec.datasets import Dataset, Split
from hls.spec.drivers import TrajectorySessionDriver
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.workload import (AllAtOnce, ArrivalProcess, DeterministicArrival, NoDelay,
                               NoSLO, SLO, TrajectoryMakespanSLO, Uncapped, Workload)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from tests.support.drive import BoundDrive


class _ListDataset(Dataset):
    name = "stub-wall-traj"
    version = "v0"
    n: int

    def items_at(self, mount_dir, split):
        for i in range(self.n):
            env = Envelope(request_id=f"r{i}", contents=[TextPayload(text="Q?")])
            yield DatasetItem(request_id=f"r{i}", turns=(env,), ground_truth_turns=("A",))

    def size(self, split):
        return self.n


class _SlowClient:
    """A server that answers, but slower than the drive can afford."""

    def __init__(self, delay_sec: float):
        self.delay_sec = delay_sec
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        await asyncio.sleep(self.delay_sec)
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text="A")]))


class _ManyTurnFramework(AgentFramework):
    """A trajectory that keeps taking turns until its budget runs out, like a real
    agent does. Each turn is one artifact call, so a slow server makes the whole
    trajectory outlast any fixed bound."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx,
                             max_turns):
        for k in range(max_turns):
            t_send = time.monotonic()
            env = Envelope(request_id=f"{seed_item.request_id}.{k}",
                           contents=[TextPayload(text="Q?")],
                           metadata={"turn_index": k,
                                     "session_id": seed_item.request_id})
            resp = await llm_call(env)
            await on_event(TraceEvent(
                request_id=f"{seed_item.request_id}.{k}",
                session_id=seed_item.request_id, turn_index=k,
                input=env, response=resp, ground_truth=None,
                t_send=t_send, t_recv=time.monotonic()))


class _StampingFramework(_ManyTurnFramework):
    """Stamps ``tenant_id``/``session_id`` on the envelope it sends, the way a real
    framework does (``DeepResearchFramework`` builds a ``base_metadata`` from
    ``ctx``). The driver never stamps the trajectory loop's envelopes itself, so this
    is where a trajectory task's events get the tenant they are scored under."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx,
                             max_turns):
        base = {"tenant_id": ctx.tenant_id,
                "session_id": seed_item.request_id}
        for k in range(max_turns):
            t_send = time.monotonic()
            env = Envelope(request_id=f"{seed_item.request_id}.{k}",
                           contents=[TextPayload(text="Q?")],
                           metadata={**base, "turn_index": k})
            resp = await llm_call(env)
            await on_event(TraceEvent(
                request_id=f"{seed_item.request_id}.{k}",
                session_id=seed_item.request_id, turn_index=k,
                input=env, response=resp, ground_truth=None,
                t_send=t_send, t_recv=time.monotonic()))


class _StashesItsLlmCall(_StampingFramework):
    """Keeps the ``llm_call`` the driver handed it, so a test can read the loop's own
    per-turn maps out of the closure (``_driver_map``). They are drive-local by design
    and no event carries them, so this is the only way to see what one held on to."""

    def __init__(self):
        self.llm_call = None

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx,
                             max_turns):
        self.llm_call = llm_call
        await super().run_trajectory(seed_item, llm_call, tool_runner, on_event, ctx,
                                     max_turns)


def _driver_map(llm_call, name: str) -> dict:
    """One of the trajectory loop's per-turn maps, by the name it is bound under.

    ``llm_call`` is the per-tenant wrapper the framework is handed; the maps live in
    the ``_llm_call`` closure behind it. Reading them by name means a rename fails the
    test loudly rather than passing on the wrong cell."""
    inner = llm_call.__closure__[
        llm_call.__code__.co_freevars.index("_llm_call")].cell_contents
    return inner.__closure__[inner.__code__.co_freevars.index(name)].cell_contents


def _tools():
    return LocalCallableToolRunner(specs=[ToolSpec(name="search", description="d")],
                                   callables={"search": lambda: {}})


def _driver(*, n_items, max_turns, drive_timeout_sec,
            framework: AgentFramework | None = None,
            slo: SLO | None = None, arrival: ArrivalProcess | None = None,
            max_concurrent_llm: int | None = None) -> BoundDrive:
    driver = TrajectorySessionDriver(framework=framework or _ManyTurnFramework(), tool_runner=_tools(),
                                     request_timeout_sec=REQUEST_TIMEOUT_SEC,
                                     max_turns=max_turns, max_concurrent_llm=max_concurrent_llm,
                                     drive_timeout_sec=drive_timeout_sec)
    tenants = [SimpleNamespace(
        id="t0",
        data_source=SimpleNamespace(dataset=_ListDataset(n=n_items), removed_rows=()),
        workload=Workload(arrival=arrival or AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=slo or NoSLO()))]
    return BoundDrive(driver, tenants, seed=42)


def _drive(drv: BoundDrive, client):
    events: list[TraceEvent] = []
    t0 = time.monotonic()
    asyncio.run(drv.drive(client, events.append, Split.DEV))
    return events, time.monotonic() - t0


def test_the_wall_stops_a_slow_trajectory_drive():
    """A trajectory whose turns are slower than the wall allows is cut at the wall,
    and the drive returns instead of running until something else kills it."""
    drv = _driver(n_items=1, max_turns=40, drive_timeout_sec=0.5)
    slow = _SlowClient(delay_sec=0.1)

    events, elapsed = _drive(drv, slow)

    assert elapsed < 3.0, f"drive ran {elapsed:.1f}s past a 0.5s wall"
    assert slow.calls < 40, "the wall did not stop the drive taking new turns"


def test_a_walled_trajectory_leaves_an_unanswered_turn_so_the_gate_sees_it():
    """A cut trajectory must carry a turn with no response. Its makespan is then not
    measurable, the SLO gate zeroes the run, and the scorer prices it for
    ``upper_bound``. Without it a truncated trajectory looks like a fast one."""
    drv = _driver(n_items=1, max_turns=40, drive_timeout_sec=0.5)

    events, _ = _drive(drv, _SlowClient(delay_sec=0.1))

    unanswered = [ev for ev in events if ev.response is None]
    assert unanswered, "a wall-cut trajectory reported no unanswered turn"
    assert all("drive wall" in (ev.error or "") for ev in unanswered), \
        [ev.error for ev in unanswered]


def test_the_wall_does_not_touch_a_trajectory_drive_that_keeps_up():
    """A drive that finishes inside the wall is untouched: every turn answered."""
    drv = _driver(n_items=2, max_turns=3, drive_timeout_sec=30.0)
    fast = _SlowClient(delay_sec=0.0)

    events, _ = _drive(drv, fast)

    assert fast.calls == 6
    assert all(ev.response is not None and ev.error is None for ev in events)


def test_a_wall_fault_carries_the_tenant_that_owns_the_trajectory():
    """The fault the wall records must land in its tenant's slice.

    ``Trace.events_for`` is a total lookup on ``input.metadata["tenant_id"]`` with no
    None fallback, and the driver records the raw dataset envelope, which the
    trajectory loop never stamps. An unstamped fault is in no tenant's slice at all."""
    drv = _driver(n_items=1, max_turns=40, drive_timeout_sec=0.5,
                  framework=_StampingFramework())

    events, _ = _drive(drv, _SlowClient(delay_sec=0.1))

    cut = [ev for ev in events if ev.response is None]
    assert cut, "a wall-cut trajectory reported no unanswered turn"
    assert [ev.tenant_id for ev in cut] == ["t0"] * len(cut), \
        [(ev.request_id, ev.tenant_id) for ev in cut]
    assert len(Trace(events=events).events_for("t0")) == len(events), \
        "a wall fault fell outside every tenant slice"


def test_a_walled_trajectory_is_censored_rather_than_priced_as_a_fast_finish():
    """The cut trajectory must reach ``censored_ratio``, which is what the wall's
    ``upper_bound`` is built from.

    With the fault outside the tenant's slice, that slice holds only the turns that
    answered: the SLO measures a makespan over them, the trajectory looks complete,
    and it earns a fast ratio instead. The run still zeroes, but on the bound the
    scorer reports the cut trajectory as one that finished early."""
    slo = TrajectoryMakespanSLO(target_sec=60.0)
    drv = _driver(n_items=1, max_turns=40, drive_timeout_sec=0.5,
                  framework=_StampingFramework(), slo=slo)

    events, _ = _drive(drv, _SlowClient(delay_sec=0.1))

    trace = Trace(events=events)
    t_slice = Trace(events=trace.events_for("t0"))
    result = slo.apply(t_slice, drv.tenants[0].workload)
    assert not result.aggregate_passed, "the cut trajectory passed its own SLO gate"
    units = slo.units(t_slice)
    assert set(units) == {"r0"}
    assert result.per_event_ratio.get("r0") is None, \
        f"the cut trajectory was credited a measured ratio {result.per_event_ratio}"
    assert slo.censored_ratio(units["r0"], trace.t_end()) is not None


class _HoldsAnEventOverATool(_ManyTurnFramework):
    """Holds a turn's event until the turn is over, and flushes it from a ``finally``.

    ``ResearchLoop`` does exactly this: a turn's event is built, the tool runs, and a
    ``finally`` emits whatever is still held. The wall cancels the trajectory wherever
    it is parked, so the held event is emitted during the cancellation -- after the
    driver's clock has already expired and before it records the cut."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx,
                             max_turns):
        base = {"tenant_id": ctx.tenant_id,
                "session_id": seed_item.request_id}
        held = None
        try:
            for k in range(max_turns):
                if held is not None:
                    await on_event(held)
                    held = None
                t_send = time.monotonic()
                env = Envelope(request_id=f"{seed_item.request_id}.{k}",
                               contents=[TextPayload(text="Q?")],
                               metadata={**base, "turn_index": k})
                resp = await llm_call(env)
                held = TraceEvent(
                    request_id=env.request_id, session_id=seed_item.request_id,
                    turn_index=k, input=env, response=resp, ground_truth=None,
                    t_send=t_send, t_recv=time.monotonic())
                await tool_runner.run("search", {})
                await asyncio.sleep(10.0)      # the tool wait the wall lands in
        finally:
            if held is not None:
                await on_event(held)


def test_a_trajectory_cut_while_holding_an_event_does_not_collide_with_its_fault():
    """The wall's fault must be the turn after the last one emitted, not a second
    event for a turn already in the trace.

    The driver numbers the fault from ``turns_seen``, which the held event updates on
    its way out during cancellation. A duplicate request id would offer one turn's
    served tokens to a per-call verifier twice, and would put an answered turn and a
    fault under the same key."""
    drv = _driver(n_items=1, max_turns=40, drive_timeout_sec=0.5,
                  framework=_HoldsAnEventOverATool())

    events, _ = _drive(drv, _SlowClient(delay_sec=0.0))

    ids = [ev.request_id for ev in events]
    assert len(ids) == len(set(ids)), ids
    answered = [ev.request_id for ev in events if ev.response is not None]
    cut = [ev for ev in events if ev.response is None]
    assert answered == ["r0.0"], ids
    assert [ev.request_id for ev in cut] == ["r0.1"], ids
    assert all(ev.tenant_id == "t0" for ev in events), \
        [(ev.request_id, ev.tenant_id) for ev in events]


def test_a_trajectory_the_wall_never_let_start_carries_its_tenant_too():
    """The same stamp on the other ``_record_wall_fault`` caller: a session whose
    arrival falls past the wall is recorded without being opened.

    And it is recorded at the wall, not at the arrival it was still waiting for. The
    second session is due at t=2s under a 0.3s wall, so a drive that waits out its
    schedule ends nearly seven times later than the bound it was given."""
    drv = _driver(n_items=2, max_turns=2, drive_timeout_sec=0.3,
                  framework=_StampingFramework(),
                  arrival=DeterministicArrival(gap=2.0))

    events, elapsed = _drive(drv, _SlowClient(delay_sec=0.0))

    never_started = [ev for ev in events if ev.request_id.startswith("r1")]
    assert never_started, "the late session left no fault behind"
    assert all(ev.tenant_id == "t0" and ev.response is None
               for ev in never_started), \
        [(ev.request_id, ev.tenant_id) for ev in never_started]
    assert elapsed < 1.0, \
        f"the drive ran {elapsed:.1f}s, waiting out an arrival past its 0.3s wall"
    # The same bound read off the record instead of the clock, which a slow machine
    # cannot move: the harness gave up on this trajectory before the arrival it was
    # owed. A drive that sleeps to the arrival first stamps the fault there, so
    # t_send lands at or after t_planned_send however fast the host is.
    [late] = [ev for ev in never_started if ev.t_planned_send is not None]
    assert late.t_send < late.t_planned_send, (
        f"the fault was stamped {late.t_send - late.t_planned_send:+.2f}s from the "
        "arrival it was owed, so the drive waited that arrival out")


def test_a_wall_fault_names_the_same_turn_on_the_event_and_its_input():
    """``ev.request_id`` and ``ev.input.request_id`` must be the same turn.

    The envelope the wall records is the raw dataset one, whose id is the session's;
    the event is numbered ``<session>.<turn>``. Answered turns and the artifact-fault
    path carry the envelope the framework built, so the two agree there. The driver's
    own ``stamped_inputs``/``submitted`` maps key off ``input.request_id``, and the
    verifiers read ``ev.input``."""
    drv = _driver(n_items=2, max_turns=40, drive_timeout_sec=0.4,
                  framework=_StampingFramework(),
                  arrival=DeterministicArrival(gap=1.0))

    events, _ = _drive(drv, _SlowClient(delay_sec=0.1))

    cut = [ev for ev in events if ev.response is None]
    assert cut, "a wall-cut trajectory reported no unanswered turn"
    assert [(ev.request_id, ev.input.request_id) for ev in cut
            if ev.request_id != ev.input.request_id] == []


def test_a_turn_cut_while_it_waits_for_an_llm_slot_leaves_nothing_behind():
    """A turn registers its envelope and its send time before admission control, and
    whatever ends it has to drop both again.

    With ``max_concurrent_llm`` set, a trajectory can be cancelled while queued for a
    slot -- before the submit runs at all, so neither of the submit's own fault paths
    fires. The entry left behind is then the last thing holding that turn's envelope,
    which on a trajectory task is the whole accumulated conversation, and it holds it
    for the rest of the drive."""
    fw = _StashesItsLlmCall()
    drv = _driver(n_items=2, max_turns=40, drive_timeout_sec=0.3, framework=fw,
                  max_concurrent_llm=1)

    _drive(drv, _SlowClient(delay_sec=0.1))

    assert _driver_map(fw.llm_call, "stamped_inputs") == {}
    assert _driver_map(fw.llm_call, "submitted") == {}


def test_no_wall_configured_leaves_the_trajectory_drive_unbounded():
    """``None`` sets no wall, so a task that sets none drives every turn."""
    drv = _driver(n_items=1, max_turns=4, drive_timeout_sec=None)
    fast = _SlowClient(delay_sec=0.0)

    events, _ = _drive(drv, fast)

    assert fast.calls == 4
    assert all(ev.error is None for ev in events)
