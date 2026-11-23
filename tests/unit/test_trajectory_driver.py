"""Unit tests for TrajectorySessionDriver -- the agentic loop dispatcher.

The driver is the base ``Driver``'s subclass, built whole by ``driver_for`` from the
task's agent seam (framework, tool_runner, max_turns, max_concurrent_llm); ``run`` calls
``_drive`` with the drive's tenants and seed and a SYNC on_event. These tests exercise
``_drive`` directly with a sync collector.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.core.engine import ResourceRequest
from hls.spec.agent import AgentFramework, AgentSpec
from hls.spec.datasets import Dataset, Split
from hls.spec import drivers
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import (
    Driver, TrajectorySessionDriver, driver_for, load_mounted_sessions,
)
from hls.spec.tenant import Tenant
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.workload import (
    AllAtOnce, ConstantInterTurn, DeterministicArrival, GammaInterTurn, NoDelay,
    NoSLO, Poisson, ShortOutputCap, Uncapped, Workload,
)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from tests.support.drive import BoundDrive
from tests.support.virtual_clock import virtual_clock


class _ListDataset(Dataset):
    """Plain N-item dataset stub. ``items_at`` is the reader the drive loop uses
    (every tenant reads its own mount), so that is what the stub provides."""
    name = "stub"
    version = "v0"
    n: int

    def items_at(self, mount_dir, split):
        for i in range(self.n):
            env = Envelope(request_id=f"r{i}", contents=[TextPayload(text="Q?")])
            yield DatasetItem(request_id=f"r{i}", turns=(env,), ground_truth_turns=("A",))

    def size(self, split):
        return self.n


class _EchoClient:
    async def submit(self, env, *, attempt=None):
        return ArtifactResponse(
            output=Envelope(request_id=env.request_id,
                            contents=[TextPayload(text="A")]))


class _RecordingFramework(AgentFramework):
    """Records the args its run_trajectory was called with."""
    def __init__(self):
        self.calls: list[dict] = []
    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event,
                             ctx, max_turns):
        self.calls.append({
            "seed_id": seed_item.request_id,
            "max_turns": max_turns,
            "tool_runner": tool_runner,
            "ctx_tenant": ctx.tenant_id,
            "ctx_window": ctx.max_context_len,
            "ctx_pin": dict(ctx.chat_template_kwargs),
        })
        t_send = time.monotonic()
        resp = await llm_call(seed_item.turns[0])
        await on_event(TraceEvent(
            request_id=f"{seed_item.request_id}.0",
            session_id=seed_item.request_id, turn_index=0,
            input=seed_item.turns[0], response=resp,
            ground_truth=None, t_send=t_send, t_recv=time.monotonic(),
            extras={"final_answer": "stub"}))


def _tools():
    return LocalCallableToolRunner(
        specs=[ToolSpec(name="search", description="d")],
        callables={"search": lambda: {}})


def _driver(framework, dataset, workload, *, max_turns=7, max_concurrent_llm=None,
            tool_runner=None) -> BoundDrive:
    """A trajectory driver over *framework*, driven at seed 42 over one tenant "t0" holding
    *dataset* and *workload*."""
    driver = TrajectorySessionDriver(framework=framework, request_timeout_sec=REQUEST_TIMEOUT_SEC,
                                     tool_runner=tool_runner if tool_runner is not None else _tools(),
                                     max_turns=max_turns, max_concurrent_llm=max_concurrent_llm)
    tenants = [
        SimpleNamespace(
            id="t0",
            data_source=SimpleNamespace(dataset=dataset, removed_rows=()),
            workload=workload,
        )
    ]
    return BoundDrive(driver, tenants, seed=42)


def _workload(arrival=None):
    return Workload(arrival=arrival or AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(), slo=NoSLO())


def _drive(drive: BoundDrive, client=None):
    events: list[TraceEvent] = []
    def on_event(ev): events.append(ev)
    asyncio.run(drive.drive(client or _EchoClient(), on_event, Split.DEV))
    return events


def test_trajectory_driver_delegates_to_framework():
    framework = _RecordingFramework()
    drv = _driver(framework, _ListDataset(n=1), _workload(), max_turns=7)

    events = _drive(drv)

    assert len(framework.calls) == 1
    call = framework.calls[0]
    assert call["seed_id"] == "r0"
    assert call["max_turns"] == 7      # the stashed per-trajectory budget
    assert call["tool_runner"] is drv.driver.tool_runner
    assert call["ctx_tenant"] == "t0"
    assert len(events) == 1


def test_every_trajectory_is_handed_the_tasks_window_and_pin():
    """``driver_for`` takes the task's context window and its model's chat-template pin, and
    the driver stamps both on every trajectory's ``HarnessContext``: the framework object is
    shared by every copy of a TaskSpec, so it carries neither."""
    framework = _RecordingFramework()
    agent = AgentSpec(framework=framework, tool_runner=_tools(), max_turns=7)
    drv = driver_for(agent, request_timeout_sec=REQUEST_TIMEOUT_SEC, max_context_len=32_768,
                     chat_template_kwargs={"enable_thinking": True})
    _drive(BoundDrive(drv, _driver(framework, _ListDataset(n=2), _workload()).tenants, seed=42))
    assert [(c["ctx_window"], c["ctx_pin"]) for c in framework.calls] == [
        (32_768, {"enable_thinking": True})] * 2
    framework.calls.clear()
    _drive(_driver(framework, _ListDataset(n=1), _workload()))
    assert [(c["ctx_window"], c["ctx_pin"]) for c in framework.calls] == [(None, {})]


def test_a_trajectory_driver_without_tool_assets_asks_for_no_gpu_and_no_memory():
    """A trajectory driver whose tools load nothing is sized like the base driver: no GPU and no
    memory beyond the driver role's own shape."""
    drv = _driver(_RecordingFramework(), _ListDataset(n=1), _workload())
    assert drv.driver.resources == Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC).resources
    assert drv.driver.resources == ResourceRequest(gpu_count=0)


# ---- concurrent dispatch (Poisson arrival) -------------------------------

class _SlowFramework(AgentFramework):
    """Framework that sleeps to expose concurrency overlap."""
    def __init__(self, sleep_sec: float):
        self.sleep_sec = sleep_sec
        self.entered: list[float] = []
        self.exited: list[float] = []
        self._t0 = None

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event,
                             ctx, max_turns):
        if self._t0 is None:
            self._t0 = time.monotonic()
        self.entered.append(time.monotonic() - self._t0)
        await asyncio.sleep(self.sleep_sec)
        self.exited.append(time.monotonic() - self._t0)
        await on_event(TraceEvent(
            request_id=f"{seed_item.request_id}.0",
            session_id=seed_item.request_id, turn_index=0,
            input=seed_item.turns[0],
            response=ArtifactResponse(output=seed_item.turns[0]),
            ground_truth=None, t_send=0.0, t_recv=0.1,
            extras={"final_answer": "stub"}))


def test_poisson_arrival_runs_trajectories_concurrently():
    """With Poisson at high rate, multiple trajectories overlap (entered before
    others exited). Each trajectory sleeps 100ms; with rate=1000/sec the
    inter-arrival is ~1ms so most trajectories start before any finish."""
    framework = _SlowFramework(sleep_sec=0.1)
    drv = _driver(framework, _ListDataset(n=5),
                  _workload(arrival=Poisson(rate_per_sec=1000.0)), max_turns=2)

    _drive(drv)

    first_exit = framework.exited[0]
    last_entry = max(framework.entered)
    assert last_entry < first_exit, (
        f"expected overlap: last entry {last_entry:.3f}s, first exit {first_exit:.3f}s")


class _ToolOverlapFramework(AgentFramework):
    """One LLM call then a slow tool wait; tracks concurrent tool phases."""
    def __init__(self, tool_sleep_sec: float):
        self.tool_sleep_sec = tool_sleep_sec
        self.tool_in_flight = 0
        self.max_tool_in_flight = 0

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event,
                             ctx, max_turns):
        t_send = time.monotonic()
        await llm_call(seed_item.turns[0])
        t_recv = time.monotonic()
        self.tool_in_flight += 1
        self.max_tool_in_flight = max(self.max_tool_in_flight, self.tool_in_flight)
        await asyncio.sleep(self.tool_sleep_sec)
        self.tool_in_flight -= 1
        await on_event(TraceEvent(
            request_id=f"{seed_item.request_id}.0",
            session_id=seed_item.request_id, turn_index=0,
            input=seed_item.turns[0],
            response=ArtifactResponse(output=seed_item.turns[0]),
            ground_truth=None, t_send=t_send, t_recv=t_recv,
            extras={"final_answer": "stub"}))


def test_max_concurrent_llm_allows_tool_overlap():
    """LLM-only cap lets more trajectories wait on tools than the LLM limit."""
    framework = _ToolOverlapFramework(tool_sleep_sec=0.08)
    drv = _driver(framework, _ListDataset(n=5),
                  _workload(arrival=Poisson(rate_per_sec=1000.0)),
                  max_turns=2, max_concurrent_llm=2)

    _drive(drv)

    assert framework.max_tool_in_flight >= 3, (
        f"expected >=3 concurrent tool waits with llm cap=2; "
        f"saw {framework.max_tool_in_flight}")


def test_max_concurrent_llm_caps_llm_calls():
    """At most max_concurrent_llm artifact submits overlap."""
    class _TrackingClient:
        def __init__(self):
            self.in_flight = 0
            self.max_in_flight = 0

        async def submit(self, env, *, attempt=None):
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            await asyncio.sleep(0.04)
            self.in_flight -= 1
            return ArtifactResponse(
                output=Envelope(request_id=env.request_id,
                                contents=[TextPayload(text="A")]))

    framework = _ToolOverlapFramework(tool_sleep_sec=0.06)
    client = _TrackingClient()
    drv = _driver(framework, _ListDataset(n=5),
                  _workload(arrival=Poisson(rate_per_sec=1000.0)),
                  max_turns=2, max_concurrent_llm=2)

    _drive(drv, client=client)

    assert client.max_in_flight <= 2, (
        f"max_concurrent_llm=2 violated: saw {client.max_in_flight} in flight")


def test_trajectory_driver_stamps_max_tokens_on_llm_call():
    """Agentic LLM submits must carry the workload cap (same as the base Driver)."""
    from hls.spec.workload import ShortOutputCap

    class _CapturingClient:
        def __init__(self):
            self.max_tokens: list[int] = []

        async def submit(self, env, *, attempt=None):
            self.max_tokens.append(env.metadata.get("max_tokens"))
            return ArtifactResponse(
                output=Envelope(request_id=env.request_id,
                                contents=[TextPayload(text="A")]))

    workload = Workload(
        arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
        output_length_cap=ShortOutputCap(cap_tokens=9),
        slo=NoSLO(),
    )
    client = _CapturingClient()
    drv = _driver(_RecordingFramework(), _ListDataset(n=1), workload, max_turns=3)
    events = _drive(drv, client=client)
    assert client.max_tokens == [9]
    assert events[0].input.metadata["max_tokens"] == 9


def test_trajectory_driver_preserves_framework_max_tokens_override():
    """Framework repair turns can request a smaller cap than the workload cap."""
    from hls.spec.workload import ShortOutputCap

    class _CapOverrideFramework(AgentFramework):
        async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event,
                                 ctx, max_turns):
            env = seed_item.turns[0].model_copy(
                update={"metadata": {"max_tokens": 5}})
            t_send = time.monotonic()
            resp = await llm_call(env)
            await on_event(TraceEvent(
                request_id=f"{seed_item.request_id}.0",
                session_id=seed_item.request_id, turn_index=0,
                input=env, response=resp,
                ground_truth=None, t_send=t_send, t_recv=time.monotonic(),
                extras={"final_answer": "stub"}))

    class _CapturingClient:
        def __init__(self):
            self.max_tokens: list[int] = []

        async def submit(self, env, *, attempt=None):
            self.max_tokens.append(env.metadata.get("max_tokens"))
            return ArtifactResponse(
                output=Envelope(request_id=env.request_id,
                                contents=[TextPayload(text="A")]))

    workload = Workload(
        arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
        output_length_cap=ShortOutputCap(cap_tokens=99),
        slo=NoSLO(),
    )
    client = _CapturingClient()
    drv = _driver(_CapOverrideFramework(), _ListDataset(n=1), workload, max_turns=3)
    events = _drive(drv, client=client)
    assert client.max_tokens == [5]
    assert events[0].input.metadata["max_tokens"] == 5


def test_the_recorded_input_keeps_the_turns_tool_definitions():
    """The trace event records the envelope actually sent, tools included: the
    token-level gate teacher-forces against it, so a turn's tools have to survive the
    driver's max_tokens stamp."""
    tools = [{"type": "function", "function": {"name": "web_search"}}]

    class _ToolOfferingFramework(AgentFramework):
        async def run_trajectory(self, *, seed_item, llm_call, tool_runner,
                                 on_event, ctx, max_turns):
            env = seed_item.turns[0].model_copy(update={"metadata": {"tools": tools}})
            resp = await llm_call(env)
            await on_event(TraceEvent(
                request_id=env.request_id, session_id=seed_item.request_id,
                turn_index=0, input=env, response=resp, t_send=ctx.t_send))

    drv = _driver(_ToolOfferingFramework(), _ListDataset(n=1), _workload(), max_turns=1)

    events = _drive(drv)

    assert events[0].input.metadata["tools"] == tools
    assert events[0].input.metadata["max_tokens"] is not None


# ---- the driver factory --------------------------------------------------

class _TwoCallFramework(AgentFramework):
    """Two LLM calls per item: the first asks the tool, the second answers.

    Numbers its turns and names its session on every request, and takes its own send
    and receive stamps around each ``llm_call``, as the real deep-research loop does,
    on the driver's clock (``harness_now``, the monotonic clock that loop reads).
    ``sent`` and ``returned`` hold ``(session_id, turn_index, time)``; the send stamp is
    the one the driver re-stamps, so a test can compare the two."""

    def __init__(self):
        self.sent: list[tuple[str, int, float]] = []
        self.returned: list[tuple[str, int, float]] = []

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx,
                             max_turns):
        sid = seed_item.request_id
        messages = [{"role": "user", "content": seed_item.turns[0].first_of("text").text}]
        for turn in range(2):
            env = Envelope(request_id=f"{sid}.{turn}",
                           contents=[TextPayload(text=messages[-1]["content"])],
                           metadata={"messages": list(messages),
                                     "session_id": sid, "turn_index": turn,
                                     "tools": [{"type": "function",
                                                "function": {"name": "lookup"}}]})
            t_send = drivers.harness_now()
            self.sent.append((sid, turn, t_send))
            resp = await llm_call(env)
            t_recv = drivers.harness_now()
            self.returned.append((sid, turn, t_recv))
            reply = resp.output.first_of("text").text
            await on_event(TraceEvent(request_id=env.request_id, session_id=sid,
                                      turn_index=turn, input=env, response=resp,
                                      t_send=t_send, t_recv=t_recv))
            if turn == 0:
                result = await tool_runner.run("lookup", {"q": reply})
                messages += [{"role": "assistant", "content": reply},
                             {"role": "user", "content": result["text"]}]


def _agent() -> AgentSpec:
    """The agent seam a task declares: one framework, one tool, a turn budget.

    Imported by the clocked-serving tests, so a calibration serving and a graded run
    are exercised against the same agent."""
    runner = LocalCallableToolRunner(
        specs=[ToolSpec(name="lookup", description="d", json_schema={})],
        callables={"lookup": lambda q: {"text": f"found {q}"}})
    return AgentSpec(framework=_TwoCallFramework(), tool_runner=runner, max_turns=4,
                     max_concurrent_llm=2)


class _RecordingClient:
    """Echo client that keeps every envelope it was handed, and when.

    ``at`` maps a request id to the time, on the driver's clock, the driver actually
    submitted it, which is what a think-time test has to measure: the framework stamps
    its own send time before handing the call to the driver, so only the client sees the
    wait."""

    def __init__(self):
        self.seen: list[Envelope] = []
        self.at: dict[str, float] = {}

    async def submit(self, env, *, attempt=None):
        self.seen.append(env)
        self.at[env.request_id] = drivers.harness_now()
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id,
            contents=[TextPayload(text="reply", tokens=[1, 2, 3])]))


class _TwoSessionDataset(Dataset):
    """Two single-turn sessions, ``c0`` and ``c1``."""
    name = "twosession"
    version = "v0"

    def items_at(self, mount_dir, split):
        for c in range(2):
            env = Envelope(request_id=f"c{c}.0", contents=[TextPayload(text=f"q{c}")])
            yield DatasetItem(request_id=f"c{c}", turns=(env,),
                              ground_truth_turns=(f"g{c}",))


def _sessions(inter_turn_delay=None) -> list[tuple]:
    """``[(tenant, items)]`` already materialized, as a calibration serving gets them."""
    tenant = Tenant(
        id="t0",
        data_source=DataSource(dataset=_TwoSessionDataset(),
                               split=SplitSpec(dev=Count(value=1), test=Count(value=2))),
        workload=Workload(arrival=DeterministicArrival(gap=0.0),
                          inter_turn_delay=inter_turn_delay or NoDelay(),
                          output_length_cap=ShortOutputCap(cap_tokens=17), slo=NoSLO()),
        verifiers=[])
    return load_mounted_sessions([tenant], Split.TEST)


def test_driver_for_selects_the_trajectory_driver_only_for_an_agentic_task():
    plain = driver_for(None, request_timeout_sec=3.0)
    assert type(plain) is Driver and plain.request_timeout_sec == 3.0
    agent = _agent()
    traj = driver_for(agent, request_timeout_sec=3.0)
    assert isinstance(traj, TrajectorySessionDriver)
    assert traj.framework is agent.framework
    assert traj.tool_runner is agent.tool_runner
    assert traj.max_turns == 4 and traj.max_concurrent_llm == 2
    assert traj.request_timeout_sec == 3.0


def test_the_stage_s_drive_wall_beats_the_agent_s_own():
    """A drive wall the stage names wins over the one the task's agent declares."""
    agent = replace(_agent(), drive_timeout_sec=21600.0)

    walled = driver_for(agent, request_timeout_sec=REQUEST_TIMEOUT_SEC, drive_timeout_sec=600.0)
    assert walled.drive_timeout_sec == 600.0


def test_trajectory_drive_sessions_runs_every_call_of_every_trajectory():
    driver = driver_for(_agent(), request_timeout_sec=REQUEST_TIMEOUT_SEC)
    client = _RecordingClient()
    events: list[TraceEvent] = []
    asyncio.run(driver._drive_sessions(client, events.append, _sessions(),
                                       drive_seed=7))
    assert sorted(ev.request_id for ev in events) == ["c0.0", "c0.1", "c1.0", "c1.1"]
    # The stamped cap reached the client on every call, and the second call carried
    # the tool result the framework appended.
    assert all(isinstance(e.metadata["max_tokens"], int) for e in client.seen)
    second = next(e for e in client.seen if e.request_id == "c0.1")
    assert second.metadata["messages"][-1]["content"] == "found reply"
    assert second.metadata["tools"][0]["function"]["name"] == "lookup"


def test_trajectory_drive_sessions_bounds_sessions_not_calls():
    """``max_sessions`` is the knob that bounds a trajectory serving: it drops whole
    trajectories, and the one that runs makes every call the model asks for."""
    driver = driver_for(_agent(), request_timeout_sec=REQUEST_TIMEOUT_SEC)
    events: list[TraceEvent] = []
    asyncio.run(driver._drive_sessions(_RecordingClient(), events.append, _sessions(),
                                       drive_seed=7, max_sessions=1))
    assert len(events) == 2 and len({ev.session_id for ev in events}) == 1


# ---- inter-turn think time ------------------------------------------------

def _turn_gaps(events) -> dict[str, float]:
    """Per session, the recorded gap between turn 0 being received and turn 1 being
    sent -- which is where think time belongs."""
    received = {(ev.session_id, ev.turn_index): ev.t_recv for ev in events}
    return {ev.session_id: ev.t_send - received[(ev.session_id, 0)]
            for ev in events if ev.turn_index == 1}


def test_the_driver_waits_the_inter_turn_delay_before_a_follow_up_turn():
    """The workload's think time lands in the gap between one turn coming back and the
    next going out. The first turn of a trajectory is not delayed: where it starts is
    the arrival schedule's business.

    On a virtual clock (``tests.support.virtual_clock``), so every stamp is what the
    schedule made it however loaded the machine is: the drive starts at 0."""
    events: list[TraceEvent] = []
    with virtual_clock() as runner:
        runner.run(driver_for(_agent(), request_timeout_sec=REQUEST_TIMEOUT_SEC)._drive_sessions(
            _RecordingClient(), events.append,
            _sessions(ConstantInterTurn(gap=0.05)), drive_seed=7))
    assert len(events) == 4                          # two sessions, two turns each
    first_sends = [ev.t_send for ev in events if ev.turn_index == 0]
    assert first_sends == [0.0, 0.0]                 # turn 0 leaves at its arrival
    gaps = _turn_gaps(events)
    assert gaps == {"c0": pytest.approx(0.05), "c1": pytest.approx(0.05)}


def test_the_inter_turn_draw_is_deterministic_per_session_and_turn():
    """The same (seed, session, turn) draws the same wait, so replaying a seed
    reproduces the trajectory's timing."""
    delay = GammaInterTurn(shape=2.0, scale=1.0, max_gap=15.0)
    assert delay.delay(1, seed=(7, "c0")) == delay.delay(1, seed=(7, "c0"))
    assert delay.delay(1, seed=(7, "c0")) != delay.delay(1, seed=(7, "c1"))


def test_the_driver_draws_the_gamma_delay_per_session_and_never_for_turn_zero(
        monkeypatch):
    """Which draw the driver asks for is the claim under test: one per follow-up turn,
    keyed by the drive seed and the session that turn belongs to."""
    drawn: list[tuple] = []

    def recording(self, turn, seed):
        drawn.append((turn, seed))
        return 0.0

    monkeypatch.setattr(GammaInterTurn, "delay", recording)
    asyncio.run(driver_for(_agent(), request_timeout_sec=REQUEST_TIMEOUT_SEC)._drive_sessions(
        _RecordingClient(), lambda ev: None,
        _sessions(GammaInterTurn(shape=2.0, scale=1.0, max_gap=15.0)),
        drive_seed=7))
    assert sorted(drawn) == [(1, (7, "c0")), (1, (7, "c1"))]


def test_no_delay_leaves_trajectory_timing_untouched():
    """NoDelay draws 0.0, so a workload that declares no think time pays nothing for
    the seam."""
    events: list[TraceEvent] = []
    asyncio.run(driver_for(_agent(), request_timeout_sec=REQUEST_TIMEOUT_SEC)._drive_sessions(
        _RecordingClient(), events.append, _sessions(), drive_seed=7))
    assert max(_turn_gaps(events).values()) < 0.05


def test_the_event_records_the_submit_time_not_the_frameworks_earlier_stamp():
    """The framework stamps its send before the driver thinks and queues for admission,
    so the driver re-stamps at the submit. The follow-up turn's recorded span is then
    shorter than the think time that preceded it, which is the whole point: waiting is
    not something the artifact was asked to do."""
    agent, client, events = _agent(), _RecordingClient(), []
    asyncio.run(driver_for(agent, request_timeout_sec=REQUEST_TIMEOUT_SEC)._drive_sessions(
        client, events.append, _sessions(ConstantInterTurn(gap=0.05)),
        drive_seed=7))
    framework_sent = {(sid, turn): at for sid, turn, at in agent.framework.sent}
    for ev in events:
        own = framework_sent[(ev.session_id, ev.turn_index)]
        assert ev.t_send >= own                     # never earlier than the framework's
        assert ev.t_send <= client.at[ev.request_id]  # and no later than the submit
    follow_ups = [ev for ev in events if ev.turn_index == 1]
    assert len(follow_ups) == 2
    assert max(ev.t_recv - ev.t_send for ev in follow_ups) < 0.05
    # Without the re-stamp each of those spans would have contained the whole wait.
    assert min(ev.t_send - framework_sent[(ev.session_id, 1)] for ev in follow_ups) >= 0.05

    # The re-stamp follows the envelope, not the event id, so a framework free to
    # number its events some other way still gets the harness-recorded send time.
    class _RenumberingFramework(AgentFramework):
        async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event,
                                 ctx, max_turns):
            env = seed_item.turns[0]
            resp = await llm_call(env)
            await on_event(TraceEvent(
                request_id="renumbered", session_id=seed_item.request_id,
                turn_index=0, input=env, response=resp,
                t_send=0.0, t_recv=time.monotonic()))

    before = time.monotonic()
    renumbered = _drive(_driver(_RenumberingFramework(), _ListDataset(n=1), _workload()))
    assert renumbered[0].request_id == "renumbered"
    assert renumbered[0].t_send >= before


def test_waiting_for_an_llm_slot_still_counts_toward_a_turns_latency():
    """Think time is the only thing a turn's span excludes. Queueing for an LLM slot is
    the contestant's own throughput showing up, so it stays inside the span: the send is
    recorded before admission control, not after it."""

    class _QueueingClient:
        """Holds each request for 50 ms and records when it took it."""

        def __init__(self):
            self.at: dict[str, float] = {}

        async def submit(self, env, *, attempt=None):
            self.at[env.request_id] = time.monotonic()
            await asyncio.sleep(0.05)
            return ArtifactResponse(output=Envelope(
                request_id=env.request_id,
                contents=[TextPayload(text="reply", tokens=[1, 2, 3])]))

    client, events = _QueueingClient(), []
    # One slot, two sessions at once.
    driver = driver_for(replace(_agent(), max_concurrent_llm=1), request_timeout_sec=REQUEST_TIMEOUT_SEC)
    asyncio.run(driver._drive_sessions(client, events.append, _sessions(),
                                       drive_seed=7))
    first_turns = [ev for ev in events if ev.turn_index == 0]
    assert len(first_turns) == 2
    queued = max(client.at[ev.request_id] - ev.t_send for ev in first_turns)
    assert queued >= 0.05, (
        "the session that waited for the slot should carry that wait in its span")


def test_every_turn_of_a_trajectory_is_re_stamped_not_just_the_first():
    """``_aev`` re-stamps each event with the envelope actually sent, and consumes the
    entry as it does. Consuming it must not leave a later turn unstamped: the verifier
    teacher-forces the recorded input, so a turn missing the driver's ``max_tokens``
    would be replayed under a contract the artifact never served.

    Several sessions at once, so one trajectory's consumed entry cannot be another's.
    """

    class _ManyTurns(AgentFramework):
        async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event,
                                 ctx, max_turns):
            for k in range(max_turns):
                t_send = time.monotonic()
                env = Envelope(request_id=f"{seed_item.request_id}.{k}",
                               contents=[TextPayload(text="Q?")],
                               metadata={"turn_index": k,
                                         "session_id": seed_item.request_id})
                resp = await llm_call(env)
                await on_event(TraceEvent(
                    request_id=env.request_id, session_id=seed_item.request_id,
                    turn_index=k, input=env, response=resp, ground_truth=None,
                    t_send=t_send, t_recv=time.monotonic()))

    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                        output_length_cap=ShortOutputCap(cap_tokens=9), slo=NoSLO())
    drv = _driver(_ManyTurns(), _ListDataset(n=3), workload, max_turns=4)

    events = _drive(drv)

    assert len(events) == 12
    assert [ev.input.metadata.get("max_tokens") for ev in events] == [9] * 12, \
        [(ev.request_id, ev.input.metadata.get("max_tokens")) for ev in events]
    # The driver's t_send, taken after think time, not the framework's earlier one.
    assert all(ev.t_recv is not None and ev.t_send <= ev.t_recv for ev in events)


# ---------------------------------------------------------------------------
# An artifact fault the framework raises about a reply the submit path accepted
# ---------------------------------------------------------------------------


class _RefusesTheSecondReply(AgentFramework):
    """Serves two turns, emits the first, and refuses the second the way a framework that
    checks a reply's text against its tokens does: an ``ArtifactFault`` raised inside the
    framework, about a reply the transport already returned.

    ``prefill`` is the prompt length the framework stamps on each request it sends; None
    leaves it unstamped. ``evidence`` is what the fault says about the reply it refused."""

    def __init__(self, prefill: int | None = None, evidence: dict | None = None):
        self.prefill = prefill
        self.evidence = evidence

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        from hls.core.faults import ArtifactProtocolError
        from hls.spec.workload import PREFILL_TOKENS_EXTRA

        base = {"tenant_id": ctx.tenant_id, "session_id": seed_item.request_id}
        if self.prefill is not None:
            base[PREFILL_TOKENS_EXTRA] = self.prefill
        for k in range(2):
            env = Envelope(request_id=f"{seed_item.request_id}.{k}", contents=[TextPayload(text="Q?")],
                           metadata={**base, "turn_index": k})
            t_send = time.monotonic()
            resp = await llm_call(env)
            if k == 1:
                raise ArtifactProtocolError("the reply's `content` is not what its own `token_ids` "
                                            "detokenize to", extras=self.evidence)
            await on_event(TraceEvent(
                request_id=env.request_id, session_id=seed_item.request_id, turn_index=k, input=env,
                response=resp, ground_truth=None, t_send=t_send, t_recv=time.monotonic()))


def _faults(events):
    return [ev for ev in events if ev.failure is not None]


def test_a_framework_raised_artifact_fault_zeroes_the_run_instead_of_ending_the_drive():
    """A contract violation the submit path could not see is still a measurement: it is
    recorded on the turn that broke and unwinds one trajectory, as the transport's own faults
    do, rather than ending the drive with no trace."""
    from hls.core.trace import FailureReason

    events = _drive(_driver(_RefusesTheSecondReply(), _ListDataset(n=2), _workload()))

    faults = _faults(events)
    assert {(ev.session_id, ev.turn_index) for ev in faults} == {("r0", 1), ("r1", 1)}
    assert all(ev.failure.reason is FailureReason.PROTOCOL for ev in faults)
    assert all(ev.error.startswith("protocol: ") and "token_ids" in ev.error for ev in faults)
    answered = [ev for ev in events if ev.failure is None]
    assert {(ev.session_id, ev.turn_index) for ev in answered} == {("r0", 0), ("r1", 0)}
    assert all(ev.tenant_id == "t0" for ev in faults)


def test_a_framework_raised_fault_is_recorded_under_a_wall_and_a_scheduled_stop_too():
    """The trajectory is awaited under no bound, the wall, or the stop inside the wall; the
    fault is recorded the same way in each, with the send time of the turn that broke."""
    from hls.core.trace import FailureReason
    from hls.spec.agent import ScheduledStop

    bounds = ((None, None), (600.0, None), (1000.0, ScheduledStop(wall_clock_sec=30.0, cut_grace_sec=5.0)))
    for wall, stop in bounds:
        drive = _driver(_RefusesTheSecondReply(), _ListDataset(n=1), _workload())
        drive.driver.drive_timeout_sec, drive.driver.scheduled_stop = wall, stop

        [fault] = _faults(_drive(drive))

        assert (fault.session_id, fault.turn_index) == ("r0", 1)
        assert fault.failure.reason is FailureReason.PROTOCOL
        assert fault.t_send < fault.failure.t_abandoned


def test_a_framework_raised_fault_carries_the_refused_turns_prompt_length():
    """``censored_ratio`` prices an unanswered turn against the budget its prompt length buys,
    and the flat term alone without one. The fault is recorded on the envelope the framework
    sent for the refused turn, which carries the length."""
    from hls.spec.workload import PREFILL_TOKENS_EXTRA, ContextLinearLatencySLO

    [fault] = _faults(_drive(_driver(_RefusesTheSecondReply(prefill=61_234), _ListDataset(n=1), _workload())))

    assert fault.input.request_id == "r0.1" and fault.input.metadata[PREFILL_TOKENS_EXTRA] == 61_234
    assert fault.extras == {PREFILL_TOKENS_EXTRA: 61_234}
    slo = ContextLinearLatencySLO(ttft_ms=1400.0, tpot_ms=20.0, ttft_ms_per_prefill_token=0.008)
    assert slo.ttft_budget_ms(fault) == pytest.approx(1400.0 + 0.008 * 61_234)


def test_a_faults_own_evidence_reaches_the_event_it_is_recorded_on():
    """A failed event has no response, and the refused reply is kept nowhere else, so the
    fault's own ``extras`` go on the event. The harness's prompt length wins a collision."""
    from hls.spec.workload import PREFILL_TOKENS_EXTRA

    evidence = {"agentic_coding.refused_reply": {"chars": 114_769, "tokens": 32_768}, PREFILL_TOKENS_EXTRA: 1}
    [fault] = _faults(_drive(_driver(_RefusesTheSecondReply(prefill=61_234, evidence=evidence),
                                     _ListDataset(n=1), _workload())))

    assert fault.extras["agentic_coding.refused_reply"] == {"chars": 114_769, "tokens": 32_768}
    assert fault.extras[PREFILL_TOKENS_EXTRA] == 61_234


def test_a_fault_on_a_turn_whose_prompt_was_never_counted_carries_no_stamp():
    """A length nobody counted is not invented: the event has none, and a sloped SLO refuses
    it a budget as documented."""
    from hls.spec.workload import ContextLinearLatencySLO

    [fault] = _faults(_drive(_driver(_RefusesTheSecondReply(), _ListDataset(n=1), _workload())))

    assert fault.extras == {}
    slo = ContextLinearLatencySLO(ttft_ms=1400.0, tpot_ms=20.0, ttft_ms_per_prefill_token=0.008)
    assert slo.ttft_budget_ms(fault) is None


def test_a_transport_fault_on_one_trajectory_leaves_the_others_driving():
    """A request that fails on its connection (refused, reset, dropped) fails that request and
    unwinds its own trajectory; the drive goes on and every other session is served."""
    from hls.core.faults import ArtifactTransportError

    class _ResetsSessionR1:
        async def submit(self, env, *, attempt=None):
            if env.request_id.startswith("r1"):
                raise ArtifactTransportError("ReadError('connection reset by peer')")
            return ArtifactResponse(output=Envelope(request_id=env.request_id, contents=[TextPayload(text="A")]))

    events = _drive(_driver(_RecordingFramework(), _ListDataset(n=3), _workload()), _ResetsSessionR1())

    assert {ev.session_id: ev.error for ev in events} == {
        "r0": None, "r1": "transport: ReadError('connection reset by peer')", "r2": None}


# ---------------------------------------------------------------------------
# A fault under a framework that holds each turn's event back by one turn
# ---------------------------------------------------------------------------


class _HoldsEachTurn(AgentFramework):
    """Emits turn k-1's event only once turn k's reply is in, and the held turn from a
    ``finally``, as agentic_coding's recorder does. Like that loop it names each turn in its
    request id alone, with no ``turn_index`` metadata. Each request carries its own prompt length,
    1,000 times its turn number plus one. ``refuse_turn`` raises an ``ArtifactFault`` about that
    turn's reply after the submit returned it, before the held turn is emitted. With
    ``flush_on_error`` False, a held turn is not emitted when the trajectory raises."""

    def __init__(self, turns: int, refuse_turn: int | None = None, flush_on_error: bool = True):
        self.turns = turns
        self.refuse_turn = refuse_turn
        self.flush_on_error = flush_on_error

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        from hls.core.faults import ArtifactProtocolError
        from hls.spec.workload import PREFILL_TOKENS_EXTRA

        held = None
        raised = False
        try:
            for k in range(self.turns):
                env = Envelope(request_id=f"{seed_item.request_id}.{k}", contents=[TextPayload(text="Q?")],
                               metadata={"tenant_id": ctx.tenant_id, "session_id": seed_item.request_id,
                                         PREFILL_TOKENS_EXTRA: 1000 * (k + 1)})
                t_send = time.monotonic()
                resp = await llm_call(env)
                if k == self.refuse_turn:
                    raised = True
                    raise ArtifactProtocolError("the reply's `content` is not what its own `token_ids` say")
                if held is not None:
                    await on_event(held)
                held = TraceEvent(request_id=env.request_id, session_id=seed_item.request_id, turn_index=k,
                                  input=env, response=resp, ground_truth=None, t_send=t_send,
                                  t_recv=time.monotonic())
        finally:
            if held is not None and (self.flush_on_error or not raised):
                await on_event(held)


class _FailsOnCall:
    """Answers every request but the ``n``-th, which fails on its connection."""

    def __init__(self, n: int):
        self.n = n
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        from hls.core.faults import ArtifactTransportError

        self.calls += 1
        await asyncio.sleep(0)
        if self.calls == self.n:
            raise ArtifactTransportError("ReadError('connection reset by peer')")
        return ArtifactResponse(output=Envelope(request_id=env.request_id, contents=[TextPayload(text="A")]))


def _capped_workload():
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=ShortOutputCap(cap_tokens=9),
                    slo=NoSLO())


def _by_turn(events) -> dict[int, TraceEvent]:
    turns = [ev.turn_index for ev in events]
    assert len(turns) == len(set(turns)), [(ev.request_id, ev.turn_index) for ev in events]
    return {ev.turn_index: ev for ev in events}


def test_a_failed_request_under_a_held_turn_is_recorded_on_its_own_turn():
    """Turn 2 fails while turn 1's event is still held. The fault is turn 2's: its own id, its
    own input and its own prompt length, which is what ``censored_ratio`` prices its wait
    against. Turn 1 is emitted after it, with the envelope the server was sent."""
    from hls.spec.workload import PREFILL_TOKENS_EXTRA

    events = _drive(_driver(_HoldsEachTurn(turns=4), _ListDataset(n=1), _capped_workload()), _FailsOnCall(3))

    by_turn = _by_turn(events)
    assert sorted(by_turn) == [0, 1, 2]
    assert all(ev.request_id == ev.input.request_id == f"r0.{k}" for k, ev in by_turn.items())
    failed = by_turn[2]
    assert failed.error == "transport: ReadError('connection reset by peer')"
    assert failed.extras[PREFILL_TOKENS_EXTRA] == 3000
    for k in (0, 1):
        assert by_turn[k].error is None and by_turn[k].input.metadata["max_tokens"] == 9


def test_a_reply_refused_under_a_held_turn_is_recorded_on_its_own_turn():
    """The framework refuses turn 2's reply after the submit returned it, and turn 1 is still
    held. The fault is recorded on the request whose reply was refused, as it was sent and when
    it was sent, and turn 1 keeps its own sent envelope and send time."""
    from hls.core.trace import FailureReason
    from hls.spec.workload import PREFILL_TOKENS_EXTRA

    events = _drive(_driver(_HoldsEachTurn(turns=4, refuse_turn=2), _ListDataset(n=1), _capped_workload()))

    by_turn = _by_turn(events)
    assert sorted(by_turn) == [0, 1, 2]
    assert all(ev.request_id == ev.input.request_id == f"r0.{k}" for k, ev in by_turn.items())
    refused = by_turn[2]
    assert refused.failure.reason is FailureReason.PROTOCOL
    assert refused.extras[PREFILL_TOKENS_EXTRA] == 3000
    assert refused.input.metadata["max_tokens"] == 9
    assert refused.t_send < refused.failure.t_abandoned
    for k in (0, 1):
        assert by_turn[k].error is None and by_turn[k].input.metadata["max_tokens"] == 9


def test_a_refused_reply_is_recorded_on_its_own_turn_when_the_held_turn_is_never_emitted():
    """A framework that drops its held turn when it raises leaves turn 1 unemitted, so the turns
    emitted say turn 1 is next. The refused reply is still turn 2's, the request the session sent
    last, and the fault carries turn 2's prompt length."""
    from hls.spec.workload import PREFILL_TOKENS_EXTRA

    events = _drive(_driver(_HoldsEachTurn(turns=4, refuse_turn=2, flush_on_error=False), _ListDataset(n=1),
                            _capped_workload()))

    by_turn = _by_turn(events)
    assert sorted(by_turn) == [0, 2]
    assert by_turn[2].request_id == by_turn[2].input.request_id == "r0.2"
    assert by_turn[2].extras[PREFILL_TOKENS_EXTRA] == 3000
