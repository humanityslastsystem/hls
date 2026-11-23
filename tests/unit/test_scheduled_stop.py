"""The opt-in scheduled stop: a trajectory drive bounded in wall-clock time without the bound
being a failure.

A task declares ``wall_clock_sec`` and ``cut_grace_sec`` on its agent seam. Past the first, no
request is issued and each session ends on a clean stop reason. At the sum of the two, every
open stream is cut and its session's sandbox dies with it; a cut request is scored on what it
streamed and verified on the tokens it sent. Neither is a fault, which is what separates this
from ``drive_timeout_sec``, the backstop behind it, whose faults zero the run.

The drives here run on a virtual clock (``tests.support.virtual_clock``), so every instant
they assert on is the one the schedule made, whatever the machine is doing.
"""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace

import pytest

from hls.core.dataitem import DatasetItem
from hls.core.faults import ArtifactStallError
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, FailureReason, PartialOutput, Trace, TraceEvent
from hls.pipeline.pipelines import OPEN_DRIVE_SEC, REQUEST_TIMEOUT_SEC, drive_wall, drive_wall_sec
from hls.recipes.catalog import task_keys, task_module
from hls.spec import drivers
from hls.spec.agent import (BACKSTOP_MARGIN_SEC, AgentFramework, AgentSpec, DriveWallClockReached,
                            ScheduledStop)
from hls.spec.datasets import Dataset, Split
from hls.spec.drivers import TrajectorySessionDriver, driver_for
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.workload import (CUT_EXTRA, MAX_TOKEN_GAP_EXTRA, PREFILL_TOKENS_EXTRA, AllAtOnce,
                               ArrivalProcess, DeterministicArrival, NoDelay, NoSLO, Uncapped, Workload)
from tests.support.drive import BoundDrive
from tests.support.virtual_clock import virtual_clock


def _tools():
    return LocalCallableToolRunner(specs=[ToolSpec(name="search", description="d")],
                                   callables={"search": lambda: {}})


class _NoopFramework(AgentFramework):
    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        return None


def _agent(**kwargs) -> AgentSpec:
    return AgentSpec(framework=_NoopFramework(), tool_runner=_tools(), **kwargs)


def _built(agent, **kwargs):
    return driver_for(agent, request_timeout_sec=REQUEST_TIMEOUT_SEC, **kwargs)


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------


def test_an_agent_seam_declares_no_scheduled_stop_by_default():
    """Opt-in: the fields exist on every agent seam and mean nothing until a task sets them."""
    agent = _agent()
    assert agent.wall_clock_sec is None
    assert agent.cut_grace_sec is None
    assert agent.stall_timeout_sec is None
    assert agent.scheduled_stop() is None
    assert agent.drive_wall_sec() is None


def test_a_scheduled_stop_is_the_pair_or_neither():
    """Half a declaration has no reading: a stop with no grace would cut the streams it just
    stopped issuing behind, and a grace with no stop bounds nothing."""
    with pytest.raises(ValueError, match="cut_grace_sec"):
        _agent(wall_clock_sec=2700.0)
    with pytest.raises(ValueError, match="wall_clock_sec"):
        _agent(cut_grace_sec=600.0)


def test_a_scheduled_stop_refuses_times_that_bound_nothing():
    with pytest.raises(ValueError, match="wall_clock_sec"):
        _agent(wall_clock_sec=0.0, cut_grace_sec=600.0)
    with pytest.raises(ValueError, match="cut_grace_sec"):
        _agent(wall_clock_sec=2700.0, cut_grace_sec=-1.0)
    with pytest.raises(ValueError, match="stall_timeout_sec"):
        _agent(stall_timeout_sec=0.0)


def test_the_stop_and_the_cut_are_offsets_from_the_drives_own_start():
    stop = ScheduledStop(wall_clock_sec=2700.0, cut_grace_sec=600.0)
    assert stop.stop_at(1000.0) == 3700.0
    assert stop.cut_at(1000.0) == 4300.0
    assert stop.backstop_timeout_sec() == 3300.0 + BACKSTOP_MARGIN_SEC


def test_the_agent_identity_of_a_task_without_a_scheduled_stop_is_unchanged():
    """The calibration fingerprint hashes this payload, so a new field in it would stale every
    stored calibration of an agentic task that never asked for a scheduled stop."""
    from hls.verify.calibration import agent_payload

    assert agent_payload(_agent()) == {
        "framework": f"{_NoopFramework.__module__}.{_NoopFramework.__qualname__}",
        "framework_config": {},
        "tools": [{"name": "search", "description": "d", "json_schema": {}}],
        "max_turns": 1,
    }


def test_a_declared_stall_budget_leaves_the_agent_identity_alone():
    """A failure rule, not a traffic rule: under a server that does not stall it changes no
    prompt the reference sees."""
    from hls.verify.calibration import agent_payload

    assert agent_payload(_agent(stall_timeout_sec=120.0)) == agent_payload(_agent())


def test_a_declared_scheduled_stop_enters_the_agent_identity():
    """It decides how many turns of each trajectory exist to be served."""
    from hls.verify.calibration import agent_payload

    payload = agent_payload(_agent(wall_clock_sec=2700.0, cut_grace_sec=600.0))
    assert payload["wall_clock_sec"] == 2700.0
    assert payload["cut_grace_sec"] == 600.0


# ---------------------------------------------------------------------------
# The wall the stop runs behind, as the stage plans it and the driver keeps it
# ---------------------------------------------------------------------------


def test_the_driver_carries_the_declared_stop_and_stall_budget():
    drv = _built(_agent(wall_clock_sec=30.0, cut_grace_sec=5.0, stall_timeout_sec=120.0))
    assert drv.scheduled_stop == ScheduledStop(wall_clock_sec=30.0, cut_grace_sec=5.0)
    assert drv.stall_timeout_sec == 120.0


def test_a_task_with_no_agent_seam_has_no_stall_budget():
    assert _built(None).stall_timeout_sec is None


def test_a_declared_stop_supplies_the_drive_wall():
    """A task that declares a stop and no wall gets the stop's backstop, so the backstop is a
    function of the window and the grace rather than a third number to keep in step."""
    agent = _agent(wall_clock_sec=30.0, cut_grace_sec=5.0)
    assert agent.drive_wall_sec() == 35.0 + BACKSTOP_MARGIN_SEC
    assert _built(agent).drive_timeout_sec == 35.0 + BACKSTOP_MARGIN_SEC


def test_a_wall_that_would_fire_before_the_cut_is_refused():
    """The inversion: a backstop at or below the planned ending fires first, so every
    trajectory the stop meant to cut cleanly is recorded as a fault and the run zeroes."""
    with pytest.raises(ValueError, match="drive_timeout_sec"):
        _agent(wall_clock_sec=30.0, cut_grace_sec=5.0, drive_timeout_sec=1.0)
    with pytest.raises(ValueError, match="drive_timeout_sec"):
        _agent(wall_clock_sec=30.0, cut_grace_sec=5.0, drive_timeout_sec=35.0)


def test_a_wall_inside_the_margin_is_lifted_to_it_for_the_stage_and_the_driver_alike():
    """A wall past the planned ending is the task's to set, and one inside the margin is
    lifted to it. The stage plans the drive from ``drive_wall_sec`` and the driver keeps the
    same number, so the stage never holds less time than the drive takes."""
    agent = _agent(wall_clock_sec=30.0, cut_grace_sec=5.0, drive_timeout_sec=36.0)
    assert agent.drive_timeout_sec == 36.0
    assert agent.drive_wall_sec() == 35.0 + BACKSTOP_MARGIN_SEC
    assert _built(agent).drive_timeout_sec == agent.drive_wall_sec()


def test_a_wall_past_the_margin_is_the_tasks_own():
    agent = _agent(wall_clock_sec=30.0, cut_grace_sec=5.0, drive_timeout_sec=100_000.0)
    assert agent.drive_wall_sec() == 100_000.0
    assert _built(agent, drive_timeout_sec=100_000.0).drive_timeout_sec == 100_000.0


def test_a_stage_wall_before_the_stops_backstop_is_refused():
    """The stage passes the wall it planned. One before the backstop would fire during the
    drive's own planned ending, so the driver refuses it rather than record that ending as a
    failure or run past the time the stage holds."""
    agent = _agent(wall_clock_sec=30.0, cut_grace_sec=5.0)
    with pytest.raises(ValueError, match="scheduled stop"):
        _built(agent, drive_timeout_sec=36.0)


def test_an_agent_with_no_scheduled_stop_keeps_the_wall_it_was_given():
    drv = _built(_agent(drive_timeout_sec=42.0))
    assert drv.drive_timeout_sec == 42.0
    assert drv.scheduled_stop is None
    assert _built(_agent(drive_timeout_sec=42.0), drive_timeout_sec=7.0).drive_timeout_sec == 7.0


def _graded_tasks():
    return [(name, importlib.import_module(task_module(name)).TASK) for name in task_keys()]


@pytest.mark.parametrize(("name", "task"), _graded_tasks(), ids=lambda v: v if isinstance(v, str) else "")
def test_every_task_without_a_stop_keeps_the_wall_it_had(name, task):
    """The stage takes its wall from ``AgentSpec.drive_wall_sec``. For a task that declares no
    scheduled stop that is its agent seam's ``drive_timeout_sec``, as it was before the stop
    existed, or ``OPEN_DRIVE_SEC`` with no seam or no wall; and the driver it builds with that
    wall has no stop and no stall budget. Only a task that declares a stop moves, to the
    stop's backstop."""
    agent = task.agent
    before = agent.drive_timeout_sec if agent is not None else None
    stop = agent.scheduled_stop() if agent is not None else None
    driver = _built(agent, drive_timeout_sec=drive_wall_sec(task))
    if stop is None:
        assert drive_wall(task) == before
        assert drive_wall_sec(task) == (float(before) if before is not None else float(OPEN_DRIVE_SEC))
        assert driver.stall_timeout_sec is None
        assert getattr(driver, "scheduled_stop", None) is None
    else:
        assert drive_wall(task) == max(stop.backstop_timeout_sec(), before or 0.0)
        assert driver.scheduled_stop == stop
    assert driver.drive_timeout_sec == drive_wall_sec(task)


def test_the_tasks_that_declare_a_stop_are_the_ones_expected():
    """Every task in the checkout but agentic_coding declares neither a stop nor a stall
    budget, so the test above pins their walls to what they were."""
    declaring = sorted(name for name, task in _graded_tasks()
                       if task.agent is not None
                       and (task.agent.scheduled_stop() is not None or task.agent.stall_timeout_sec is not None))
    assert declaring == ["agentic_coding"]


# ---------------------------------------------------------------------------
# Driving: stop issuing at T, cut at T+G
# ---------------------------------------------------------------------------


class _ListDataset(Dataset):
    name = "stub-scheduled-stop"
    version = "v0"
    n: int

    def items_at(self, mount_dir, split):
        for i in range(self.n):
            env = Envelope(request_id=f"r{i}", contents=[TextPayload(text="Q?")])
            yield DatasetItem(request_id=f"r{i}", turns=(env,), ground_truth_turns=("A",))

    def size(self, split):
        return self.n


def _now() -> float:
    """The driver's clock, which the virtual clock moves."""
    return drivers.harness_now()


class _SlowClient:
    """A server that answers whole replies, slowly."""

    def __init__(self, delay_sec: float):
        self.delay_sec = delay_sec
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        await asyncio.sleep(self.delay_sec)
        return ArtifactResponse(output=Envelope(request_id=env.request_id, contents=[TextPayload(text="A")]))


class _StreamsThenStalls:
    """Streams ``n_tokens`` and then never finishes, writing the attempt the way the HTTP client
    does: frames noted as they land, the partial decoded on the way out of a submit the caller
    cancelled."""

    def __init__(self, n_tokens: int, gap_sec: float = 0.01):
        self.n_tokens = n_tokens
        self.gap_sec = gap_sec
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        stamps: list[float] = []
        try:
            for _ in range(self.n_tokens):
                await asyncio.sleep(self.gap_sec)
                stamps.append(_now())
                if attempt is not None:
                    attempt.note_frame(stamps[-1])
            await asyncio.sleep(3600)
        except BaseException:
            if attempt is not None and stamps:
                attempt.partial = PartialOutput(
                    token_ids=list(range(len(stamps))), text="x" * len(stamps), finish_reason=None,
                    logprobs=None, packet_timestamps=stamps)
            raise


def _base_metadata(ctx, seed_item) -> dict:
    return {"tenant_id": ctx.tenant_id, "session_id": seed_item.request_id}


class _CountingFramework(AgentFramework):
    """Takes turns until its budget runs out, stamping each envelope the way a real framework
    does, and records which stop reason ended it."""

    def __init__(self):
        self.stop_reasons: list[str] = []

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        try:
            for k in range(max_turns):
                t_send = _now()
                env = Envelope(request_id=f"{seed_item.request_id}.{k}", contents=[TextPayload(text="Q?")],
                               metadata={**_base_metadata(ctx, seed_item), "turn_index": k})
                resp = await llm_call(env)
                await on_event(TraceEvent(
                    request_id=env.request_id, session_id=seed_item.request_id, turn_index=k, input=env,
                    response=resp, ground_truth=None, t_send=t_send, t_recv=_now()))
        except DriveWallClockReached:
            self.stop_reasons.append("wall_clock")


def _driver(*, n_items, max_turns, wall_clock_sec, cut_grace_sec, framework: AgentFramework | None = None,
            arrival: ArrivalProcess | None = None, drive_timeout_sec: float | None = None,
            max_concurrent_llm: int | None = None, stall_timeout_sec: float | None = None) -> BoundDrive:
    driver = TrajectorySessionDriver(
        framework=framework or _CountingFramework(), tool_runner=_tools(), max_turns=max_turns,
        max_concurrent_llm=max_concurrent_llm, request_timeout_sec=REQUEST_TIMEOUT_SEC,
        drive_timeout_sec=drive_timeout_sec,
        scheduled_stop=ScheduledStop(wall_clock_sec=wall_clock_sec, cut_grace_sec=cut_grace_sec),
        stall_timeout_sec=stall_timeout_sec)
    tenants = [SimpleNamespace(
        id="t0", data_source=SimpleNamespace(dataset=_ListDataset(n=n_items), removed_rows=()),
        workload=Workload(arrival=arrival or AllAtOnce(), inter_turn_delay=NoDelay(),
                          output_length_cap=Uncapped(), slo=NoSLO()))]
    return BoundDrive(driver, tenants, seed=42)


def _drive(drive: BoundDrive, client) -> tuple[list[TraceEvent], float]:
    """The drive's events and how long it took, on a virtual clock whose time 0 is the drive's
    start."""
    events: list[TraceEvent] = []
    with virtual_clock() as runner:
        runner.run(drive.drive(client, events.append, Split.DEV))
        return events, runner.get_loop().time()


def test_no_request_is_issued_past_the_wall_clock():
    """Turns of 0.02 s from t=0: the last one is sent at 0.30 s, the stop, and the next one
    comes up at 0.32 s and is never sent."""
    fw = _CountingFramework()
    slow = _SlowClient(delay_sec=0.02)

    events, elapsed = _drive(_driver(n_items=1, max_turns=1000, wall_clock_sec=0.3, cut_grace_sec=5.0,
                                     framework=fw), slow)

    assert slow.calls == len(events) == pytest.approx(0.3 / 0.02, abs=1)
    assert elapsed < 0.3 + 2 * 0.02
    assert fw.stop_reasons == ["wall_clock"]


def test_the_scheduled_stop_records_no_fault():
    """The whole point: the drive is bounded and nothing about the bound reads as the server
    failing. A failed request zeroes the run."""
    events, _ = _drive(_driver(n_items=2, max_turns=1000, wall_clock_sec=0.3, cut_grace_sec=5.0),
                       _SlowClient(delay_sec=0.02))

    assert events
    assert [ev.request_id for ev in events if ev.error or ev.failure] == []


def test_turns_answered_before_the_stop_are_kept_whole():
    fast = _SlowClient(delay_sec=0.0)

    events, _ = _drive(_driver(n_items=1, max_turns=3, wall_clock_sec=30.0, cut_grace_sec=5.0), fast)

    assert fast.calls == 3
    assert all(ev.response is not None and ev.error is None for ev in events)


def test_a_session_whose_arrival_is_past_the_stop_is_never_opened():
    """It would open a sandbox, find the clock already past, and close it again; it has no
    request to issue. The drive does not wait out its arrival either."""
    events, elapsed = _drive(_driver(n_items=2, max_turns=2, wall_clock_sec=0.2, cut_grace_sec=5.0,
                                     arrival=DeterministicArrival(gap=2.0)), _SlowClient(delay_sec=0.0))

    assert {ev.session_id for ev in events} == {"r0"}
    assert elapsed == pytest.approx(0.2)
    assert [ev.request_id for ev in events if ev.error] == []


def test_a_task_without_a_stop_drives_as_before():
    """The control: the same trajectories with no stop run every turn, and neither the cut nor
    the gap is stamped on anything."""
    driver = TrajectorySessionDriver(framework=_CountingFramework(), tool_runner=_tools(), max_turns=40,
                                     max_concurrent_llm=None, request_timeout_sec=REQUEST_TIMEOUT_SEC)
    drive = BoundDrive(driver, _driver(n_items=1, max_turns=1, wall_clock_sec=1.0, cut_grace_sec=1.0).tenants,
                       seed=42)

    events, _ = _drive(drive, _SlowClient(delay_sec=0.1))

    assert len(events) == 40
    assert all(ev.extras == {} for ev in events)


# ---------------------------------------------------------------------------
# The cut
# ---------------------------------------------------------------------------


class _OneTurnFramework(AgentFramework):
    """Sends one turn and waits for it, so a request is open when the cut lands. It emits
    nothing itself: the turn never returns, so the only event this trajectory can produce is
    the driver's own record of the cut.

    ``prefill`` is the prompt length a framework counts before it sends, carried on the
    envelope as a real one carries it; ``None`` stands for a framework that counts nothing."""

    def __init__(self, prefill: int | None = None):
        self.prefill = prefill

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        metadata = {**_base_metadata(ctx, seed_item), "turn_index": 0}
        if self.prefill is not None:
            metadata[PREFILL_TOKENS_EXTRA] = self.prefill
        await llm_call(Envelope(request_id=f"{seed_item.request_id}.0", contents=[TextPayload(text="Q?")],
                                metadata=metadata))


def _cut_events(*, n_tokens, gap_sec=0.01, wall_clock_sec=0.05, cut_grace_sec=0.6, prefill=None,
                stall_timeout_sec=None):
    """Drive one trajectory whose single turn is still streaming when the cut lands."""
    drive = _driver(n_items=1, max_turns=1, wall_clock_sec=wall_clock_sec, cut_grace_sec=cut_grace_sec,
                    framework=_OneTurnFramework(prefill=prefill), stall_timeout_sec=stall_timeout_sec)
    events, _ = _drive(drive, _StreamsThenStalls(n_tokens=n_tokens, gap_sec=gap_sec))
    return events


def test_a_cut_request_is_recorded_and_flagged():
    [ev] = _cut_events(n_tokens=3)

    assert ev.extras[CUT_EXTRA] is True
    assert ev.session_id == "r0" and ev.turn_index == 0


def test_a_cut_request_is_not_a_failed_one():
    """A failed request zeroes the run. The cut is the harness's own bound, so none of the
    fields that make an event a failure may be set."""
    [ev] = _cut_events(n_tokens=3)

    assert ev.error is None
    assert ev.failure is None
    assert ev.response is not None


def test_a_cut_request_carries_the_tokens_it_streamed():
    [ev] = _cut_events(n_tokens=3)

    payload = ev.response.output.first_of("text")
    assert payload.tokens == [0, 1, 2]
    assert payload.text == "xxx"
    assert payload.finish_reason is None


def test_a_cut_request_carries_its_token_times_and_the_cut_instant():
    """The times the latency SLO reads: each token's arrival, and a ``t_recv`` of the cut."""
    [ev] = _cut_events(n_tokens=3, gap_sec=0.01, wall_clock_sec=0.05, cut_grace_sec=0.6)

    assert ev.response.packet_timestamps == pytest.approx([0.01, 0.02, 0.03])
    assert ev.response.t_first_packet == pytest.approx(0.01)
    assert ev.t_send == pytest.approx(0.0)
    assert ev.t_recv == pytest.approx(0.65)


def test_a_cut_request_that_streamed_nothing_still_records_its_wait():
    """A server that answered nothing before the cut is the case the stop must not lose:
    without the event the session looks like one that ended early."""
    [ev] = _cut_events(n_tokens=0)

    payload = ev.response.output.first_of("text")
    assert payload.tokens is None
    assert ev.response.t_first_packet is None
    assert ev.response.packet_timestamps is None
    assert ev.t_recv == pytest.approx(0.65)


def test_a_cut_request_lands_in_its_tenants_slice():
    """``Trace.events_for`` is a total lookup on the stamped tenant, so an event that carries
    none is scored by nobody."""
    [ev] = _cut_events(n_tokens=2)

    assert ev.tenant_id == "t0"
    assert len(Trace(events=[ev]).events_for("t0")) == 1


def test_a_cut_request_carries_the_prompt_length_its_framework_counted():
    """A context-linear TTFT budget needs the length, and a cut turn never reaches the
    framework's own recorder, so the envelope is where the driver reads it."""
    [ev] = _cut_events(n_tokens=2, prefill=123_456)

    assert ev.extras[PREFILL_TOKENS_EXTRA] == 123_456


def test_a_cut_request_from_a_framework_that_counted_nothing_carries_no_length():
    """A framework that stamped nothing leaves the event unstamped, and a sloped SLO refuses
    it a budget rather than pricing it against the flat term."""
    [ev] = _cut_events(n_tokens=2)

    assert PREFILL_TOKENS_EXTRA not in ev.extras


def test_a_cut_request_records_its_longest_token_gap_under_a_stall_budget():
    [ev] = _cut_events(n_tokens=3, gap_sec=0.01, stall_timeout_sec=120.0)
    assert ev.extras[MAX_TOKEN_GAP_EXTRA] == pytest.approx(0.01)
    [plain] = _cut_events(n_tokens=3, gap_sec=0.01)
    assert MAX_TOKEN_GAP_EXTRA not in plain.extras


class _RefusesTheReply(_OneTurnFramework):
    """A framework whose reply check refuses whatever it is shown, and remembers what it was
    shown."""

    def __init__(self):
        super().__init__()
        self.seen: list[tuple] = []

    def check_reply(self, text, tokens):
        self.seen.append((text, tuple(tokens or ())))
        return "the reply's content is not what its token_ids say"


def test_a_cut_reply_is_checked_by_the_framework_that_would_have_read_it():
    """A framework validates its own replies inside the call the driver hands it, and a cut
    turn never returns from that call, so the driver asks on its behalf. Otherwise the one
    reply nobody checks is the one the server knows will not be checked."""
    fw = _RefusesTheReply()
    drive = _driver(n_items=1, max_turns=1, wall_clock_sec=0.05, cut_grace_sec=0.6, framework=fw)

    events, _ = _drive(drive, _StreamsThenStalls(n_tokens=3))

    assert fw.seen == [("xxx", (0, 1, 2))]
    assert len(events) == 1


def test_a_cut_reply_the_framework_refuses_is_a_failed_request():
    """The same weight the check carries on an answered reply, under the same error form."""
    drive = _driver(n_items=1, max_turns=1, wall_clock_sec=0.05, cut_grace_sec=0.6,
                    framework=_RefusesTheReply())

    [ev], _ = _drive(drive, _StreamsThenStalls(n_tokens=3))

    assert ev.failure is not None and ev.failure.reason is FailureReason.PROTOCOL
    assert ev.error == "protocol: the reply's content is not what its token_ids say"
    assert ev.failure.observed.partial.token_ids == [0, 1, 2]
    assert CUT_EXTRA not in ev.extras


def test_a_cut_request_is_priced_by_the_sloped_slo_that_wanted_it_cut():
    """The driver's cut record against the latency rule agentic_coding declares. Its cut is
    the drive's designed ending, so a cut request that cannot be priced is a run that scores
    nothing however well the server served."""
    from tasks.agentic_coding.task import WORKLOAD

    slo = WORKLOAD.slo
    assert slo.ttft_ms_per_prefill_token > 0

    [ev] = _cut_events(n_tokens=3, prefill=60_000)
    res = slo.apply(Trace(events=[ev]), WORKLOAD)

    assert res.diagnostics["n_no_budget"] == 0
    assert res.aggregate_passed
    assert slo.ttft_budget_ms(ev) > slo.ttft_ms
    assert res.per_event_ratio[ev.request_id] > 0


def test_a_turn_still_queueing_for_an_admission_slot_is_not_recorded_as_a_cut():
    """Only the server's own requests are cut. A turn the cut cancels while it waits for an
    admission slot was never posted, so there is no latency of the server's to record. Two
    sessions against one slot, so the second is still queueing when the cut lands."""
    drive = _driver(n_items=2, max_turns=1, wall_clock_sec=0.05, cut_grace_sec=0.6,
                    framework=_OneTurnFramework(prefill=1_000), max_concurrent_llm=1)

    events, _ = _drive(drive, _StreamsThenStalls(n_tokens=2))

    assert [ev.session_id for ev in events] == ["r0"]


def test_the_queue_for_a_slot_is_still_charged_to_a_turn_that_was_sent():
    """Waiting for an admission slot is the contestant's own throughput, so a turn that waited
    and then went out is charged that wait: its send time is taken before the queue."""
    drive = _driver(n_items=2, max_turns=1, wall_clock_sec=30.0, cut_grace_sec=5.0, max_concurrent_llm=1)

    events, _ = _drive(drive, _SlowClient(delay_sec=0.2))

    first, second = sorted(events, key=lambda e: e.t_recv)
    assert second.t_send < first.t_recv


class _HoldsTheTurnBeforeLast(AgentFramework):
    """Emits each turn's event only once the next turn has answered, and flushes whatever is
    still held from a ``finally``, as a recorder that stamps a later command result onto a
    turn does. While turn k is in flight turn k-1 is unemitted, and it is emitted during the
    cancellation, after the driver has recorded the cut. Like agentic_coding's loop, it names
    each turn in its request id alone, with no ``turn_index`` metadata."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        held = None
        try:
            for k in range(max_turns):
                env = Envelope(request_id=f"{seed_item.request_id}.{k}", contents=[TextPayload(text="Q?")],
                               metadata=_base_metadata(ctx, seed_item))
                t_send = _now()
                resp = await llm_call(env)
                if held is not None:
                    await on_event(held)
                held = TraceEvent(request_id=env.request_id, session_id=seed_item.request_id, turn_index=k,
                                  input=env, response=resp, ground_truth=None, t_send=t_send, t_recv=_now())
        finally:
            if held is not None:
                await on_event(held)


class _AnswersThenStalls:
    """Answers the first request whole, then streams a little and never finishes, so the cut
    lands on turn 1 while turn 0 is still held."""

    def __init__(self):
        self.stalling = _StreamsThenStalls(n_tokens=2)
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        if self.calls == 1:
            return ArtifactResponse(output=Envelope(request_id=env.request_id, contents=[TextPayload(text="A")]))
        return await self.stalling.submit(env, attempt=attempt)


def test_a_cut_turn_does_not_take_the_id_of_a_turn_still_being_held():
    """The cut event is numbered from the request that was cut, not from the turns emitted so
    far. A duplicate id would put an answered turn and a cut one under one key and offer one
    turn's served tokens to the report gate twice."""
    drive = _driver(n_items=1, max_turns=40, wall_clock_sec=0.4, cut_grace_sec=0.4,
                    framework=_HoldsTheTurnBeforeLast())

    events, elapsed = _drive(drive, _AnswersThenStalls())

    assert elapsed == pytest.approx(0.8)
    assert sorted((ev.request_id, ev.turn_index) for ev in events) == [("r0.0", 0), ("r0.1", 1)]
    assert [(ev.request_id, ev.input.request_id) for ev in events if ev.request_id != ev.input.request_id] == []


class _CleanupHangs(AgentFramework):
    """A trajectory whose own teardown never returns. Cancelling it runs the framework's
    cleanup and the driver waits for it; a coding task tears a container down there."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        env = Envelope(request_id=f"{seed_item.request_id}.0", contents=[TextPayload(text="Q?")],
                       metadata={**_base_metadata(ctx, seed_item), "turn_index": 0})
        try:
            await llm_call(env)
        finally:
            await asyncio.sleep(3600)


def test_the_drive_wall_still_backstops_a_cut_that_cannot_finish():
    """``drive_timeout_sec`` keeps its fault semantics behind the scheduled stop. If the cut
    itself wedges, the drive ends at the wall anyway, with the trajectory recorded as the
    wall's fault on the turn after its cut."""
    drive = _driver(n_items=1, max_turns=1, wall_clock_sec=0.05, cut_grace_sec=0.2,
                    framework=_CleanupHangs(), drive_timeout_sec=1.0)

    events, elapsed = _drive(drive, _StreamsThenStalls(n_tokens=1))

    assert elapsed == pytest.approx(1.0)
    cut, wall = events
    assert cut.extras[CUT_EXTRA] is True and cut.request_id == "r0.0"
    assert wall.error == drivers.WALL_ERROR and wall.request_id == "r0.1"


# ---------------------------------------------------------------------------
# The stall and the longest gap
# ---------------------------------------------------------------------------


class _StallsAfterATokenClient:
    """A server that delivers one token and then holds the connection open, as the HTTP client
    reports it: an ``ArtifactStallError`` with the partial attached."""

    async def submit(self, env, *, attempt=None):
        if attempt is not None:
            attempt.note_frame(_now())
            attempt.partial = PartialOutput(token_ids=[7], text="x", finish_reason=None,
                                            packet_timestamps=[_now()])
        raise ArtifactStallError("no token for 120s after the stream had delivered 1 of them")


def test_a_stalled_turn_is_a_failed_request_of_its_own_class():
    """Not a cut and not a transport error. A drive is zeroed by a failed request, so what the
    failure census says it was is the only account of why."""
    drive = _driver(n_items=1, max_turns=4, wall_clock_sec=30.0, cut_grace_sec=5.0,
                    framework=_OneTurnFramework(), stall_timeout_sec=120.0)

    [ev], _ = _drive(drive, _StallsAfterATokenClient())

    assert ev.error is not None and ev.error.startswith("stall: ")
    assert ev.failure.reason is FailureReason.STALLED
    assert not ev.extras.get(CUT_EXTRA)


class _GappyClient:
    """Answers each turn with three tokens, 0.25 s and then 0.05 s apart."""

    async def submit(self, env, *, attempt=None):
        return ArtifactResponse(output=Envelope(request_id=env.request_id,
                                                contents=[TextPayload(text="abc", tokens=[1, 2, 3])]),
                                t_first_packet=1.0, packet_timestamps=[1.0, 1.25, 1.30])


def test_an_answered_turn_records_the_longest_gap_between_its_tokens():
    """The number a stall budget is chosen against: it has to sit above the gaps a healthy
    server already shows, and the reference serving this traffic is the source for those."""
    drive = _driver(n_items=1, max_turns=1, wall_clock_sec=30.0, cut_grace_sec=5.0, stall_timeout_sec=120.0)

    events, _ = _drive(drive, _GappyClient())

    assert [ev.extras[MAX_TOKEN_GAP_EXTRA] for ev in events] == [pytest.approx(0.25)]


def test_a_drive_without_a_stall_budget_records_no_gap():
    """The gap is recorded only where a budget is declared, so a task without one records
    exactly what it did before."""
    drive = _driver(n_items=1, max_turns=1, wall_clock_sec=30.0, cut_grace_sec=5.0)

    events, _ = _drive(drive, _GappyClient())

    assert [MAX_TOKEN_GAP_EXTRA in ev.extras for ev in events] == [False]


def test_a_reply_that_timed_no_tokens_records_no_gap():
    """A whole reply, as an in-process engine returns one, carries no token times, so it has
    no gap to report; 0.0 would claim a measurement nobody took. A cut that streamed nothing
    is the same."""
    drive = _driver(n_items=1, max_turns=1, wall_clock_sec=30.0, cut_grace_sec=5.0, stall_timeout_sec=120.0)
    events, _ = _drive(drive, _SlowClient(delay_sec=0.0))
    assert [MAX_TOKEN_GAP_EXTRA in ev.extras for ev in events] == [False]
    [cut] = _cut_events(n_tokens=0, stall_timeout_sec=120.0)
    assert MAX_TOKEN_GAP_EXTRA not in cut.extras


def test_the_longest_gap_is_zero_under_two_tokens():
    assert drivers.max_token_gap_sec(None) == 0.0
    assert drivers.max_token_gap_sec([3.0]) == 0.0
    assert drivers.max_token_gap_sec([1.0, 1.0, 1.5, 1.6]) == pytest.approx(0.5)


def test_a_trajectory_that_ends_on_its_own_before_the_cut_records_no_cut_event():
    events, _ = _drive(_driver(n_items=1, max_turns=2, wall_clock_sec=30.0, cut_grace_sec=5.0),
                       _SlowClient(delay_sec=0.0))

    assert len(events) == 2
    assert [ev.request_id for ev in events if ev.extras.get(CUT_EXTRA)] == []


def test_the_stall_budget_reaches_the_client_the_drive_builds(monkeypatch):
    """The transport is the only place that sees a stream arriving, so ``Driver.run`` hands
    the budget to the client it builds; a driver whose task declares none hands it None."""
    built: list = []

    class _Client(_SlowClient):
        async def aclose(self):
            return None

    def make(contract, base_url, request_timeout_sec=None, stall_timeout_sec=None):
        built.append(stall_timeout_sec)
        return _Client(delay_sec=0.0)

    monkeypatch.setattr("hls.artifact.artifact_client.make_artifact_client", make)
    tenants = _driver(n_items=1, max_turns=1, wall_clock_sec=1.0, cut_grace_sec=1.0).tenants
    for agent in (_agent(wall_clock_sec=30.0, cut_grace_sec=5.0, stall_timeout_sec=120.0), _agent()):
        driver = _built(agent)
        asyncio.run(driver.run("http://127.0.0.1:1", split="dev", tenants=tenants, contract=None, seed=0,
                               on_event=lambda ev: None))
    assert built == [120.0, None]
