import asyncio
from typing import ClassVar

import pytest
from hls.core.faults import ArtifactTransportError, DatasetError
from hls.spec.contracts import RequestResponseContract
from hls.core.dataitem import DatasetItem
from hls.spec.datasets import Dataset, Split
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import (Driver, HarnessContext, _Deadline, _DriveWall, _WALL_GRACES,
                              WALL_ERROR, harness_now)
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import FailureReason, TraceEvent, ArtifactResponse
from hls.spec.tenant import Tenant
from hls.spec.workload import (AllAtOnce, ConstantInterTurn, NoDelay, NoSLO, ShortOutputCap,
                               Uncapped, Workload)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from tests.support.drive import BoundDrive
from tests.support.virtual_clock import virtual_clock


class _OneItemDataset(Dataset):
    name = "stub"
    version = "v0"
    def items_at(self, mount_dir, split):
        env = Envelope(request_id="r0", contents=[TextPayload(text="Q?")])
        yield DatasetItem(request_id="r0", turns=(env,),
                          ground_truth_turns=("A",))


class _EchoClient:
    """Stand-in ArtifactClient. Echoes input as the response with text='A'."""
    async def submit(self, env: Envelope, *, attempt=None) -> ArtifactResponse:
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id,
            contents=[TextPayload(text="A")],
            metadata={"source": "echo"}))


def _dummy_ds(dataset):
    """Wrap a test-double Dataset in a minimal DataSource for use in Tenant.data_source.

    The SplitSpec fields are only used at provisioning time; the driver only calls
    tenant.data_source.dataset.items_at(mount_dir, split) at drive time.
    In-memory test doubles define items_at directly and ignore mount_dir, so the
    dummy SplitSpec is never consulted."""
    if dataset is None:
        return None
    return DataSource(dataset=dataset, split=SplitSpec(dev=Count(value=1), test=Count(value=1)))


def _engine(*, dataset, seed=0, workload=None, **driver_args) -> BoundDrive:
    """A Driver built with *driver_args*, driven at *seed* over one Tenant("t0") holding
    *dataset* and *workload*. The SplitSpec is a dummy -- it is only used at provisioning
    time, not at drive time."""
    tenant = Tenant(id="t0", data_source=_dummy_ds(dataset), workload=workload, verifiers=[])
    return BoundDrive(Driver(**{"request_timeout_sec": REQUEST_TIMEOUT_SEC, **driver_args}), [tenant], seed)


def test_single_turn_dispatch_emits_one_event():
    ds = _OneItemDataset()
    eng = _engine(dataset=ds, seed=42, workload=_wl())

    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_EchoClient(), events.append, Split.DEV))

    assert len(events) == 1
    ev = events[0]
    assert ev.input.first_of("text").text == "Q?"
    assert ev.response.output.first_of("text").text == "A"
    assert ev.ground_truth == "A"
    assert ev.t_recv >= ev.t_send


def test_harness_context_required_for_dispatch():
    ctx = HarnessContext(tenant_id="t0", t_send=0.0)
    assert ctx.tenant_id == "t0"


def test_a_harness_context_names_no_adapter():
    """A task serves its base model, so the context a request is sent under has no adapter."""
    import dataclasses

    assert "adapter_id" not in {f.name for f in dataclasses.fields(HarnessContext)}


def test_driver_stamps_max_tokens_from_workload_output_cap():
    """The workload's OutputLengthCap is the source of truth for per-request
    decoding budget. The driver must stamp it on the envelope so the wire
    protocol forwards it to the artifact AND the oracle honors it."""
    ds = _OneItemDataset()
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                        output_length_cap=ShortOutputCap(cap_tokens=7),
                        slo=NoSLO())
    eng = _engine(dataset=ds, seed=42, workload=workload)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_EchoClient(), events.append, Split.DEV))
    assert events[0].input.metadata["max_tokens"] == 7


def test_driver_stamps_default_cap_when_uncapped():
    """Uncapped resolves to the 1M default, and the driver ALWAYS stamps max_tokens so
    the artifact and the reference oracle decode under the same budget."""
    from hls.spec.workload import DEFAULT_MAX_TOKENS
    ds = _OneItemDataset()
    eng = _engine(dataset=ds, seed=42, workload=_wl())
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_EchoClient(), events.append, Split.DEV))
    assert events[0].input.metadata["max_tokens"] == DEFAULT_MAX_TOKENS


def _rrc():
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}))


def _wl():
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                    slo=NoSLO())


class _TwoItemDataset(_OneItemDataset):
    def items_at(self, mount_dir, split):
        for rid in ("r0", "r1"):
            env = Envelope(request_id=rid, contents=[TextPayload(text="Q?")])
            yield DatasetItem(request_id=rid, turns=(env,), ground_truth_turns=("A",))


class _FlakyClient:
    """Fails the first submit with an ArtifactFault, succeeds afterwards."""
    def __init__(self):
        self.calls = 0
    async def submit(self, env, *, attempt=None):
        self.calls += 1
        if self.calls == 1:
            raise ArtifactTransportError("artifact returned 503")
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text="A")]))


class _WarmupRecordingClient(_EchoClient):
    """Records call order so we can assert warmup precedes any request send."""
    def __init__(self):
        self.calls: list[str] = []
    async def warmup(self):
        self.calls.append("warmup")
    async def submit(self, env, *, attempt=None):
        self.calls.append("submit")
        return await super().submit(env)


def test_driver_warms_connection_before_dispatch():
    # The driver must warm the shared connection (off the measured clock) BEFORE
    # it sends the first scheduled request, so the first send reuses an
    # established keep-alive connection instead of paying the handshake.
    eng = _engine(dataset=_OneItemDataset(), seed=42, workload=_wl())
    client = _WarmupRecordingClient()
    asyncio.run(eng.drive(client, [].append, Split.DEV))
    assert client.calls, "driver made no calls"
    assert client.calls[0] == "warmup"
    assert "submit" in client.calls


def test_artifact_fault_becomes_failed_event_and_loop_continues():
    eng = _engine(dataset=_TwoItemDataset(), seed=0, workload=_wl())
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_FlakyClient(), events.append, Split.DEV))

    assert len(events) == 2                       # loop did NOT abort on the first failure
    by_rid = {ev.request_id: ev for ev in events}
    failed = by_rid["r0.0"]
    assert failed.response is None
    assert failed.error is not None and "transport" in failed.error
    assert failed.t_recv is None                  # excluded from latency
    assert failed.ground_truth == "A"
    ok = by_rid["r1.0"]
    assert ok.response.output.first_of("text").text == "A"
    assert ok.error is None


def test_all_failed_events_score_a_valid_zero():
    from hls.core.trace import Trace
    from hls.verify import LetterMatchMetric, MetricVerifier, EngineAccessor
    from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
    from hls.core.model import AcceleratorSpec, Hardware

    def _failed_ev(rid):
        env = Envelope(request_id=rid, contents=[TextPayload(text="Q?")])
        return TraceEvent(request_id=rid, input=env, response=None,
                          error="transport: artifact returned 503",
                          ground_truth="B", t_send=0.0, t_recv=None)

    events = [_failed_ev("r0"), _failed_ev("r1")]
    verifier = MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)
    acc = EngineAccessor(artifact_events=events, cap_to_provider={})
    verdict = asyncio.run(verifier.verify(acc))
    assert all(v is False for v in verdict.per_event.values())  # response=None -> fail
    assert verdict.passed is False
    hw = Hardware(accelerator=AcceleratorSpec(kind="B200", count=1))
    t0 = Tenant(id="t0", workload=_wl(), verifiers=[])
    score = asyncio.run(GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)).score(
        [t0], Trace(events=events), verdict, hw))
    assert score.value == 0.0
    assert score.gate_passed is False
    assert "correctness" in score.zero_reasons                  # VALID zero, not invalid


def _noop(ev):  # on_event sink (sync, like the list Driver.run collects into)
    return None


def test_session_empty_turns_raises_dataset_error():
    item = DatasetItem(request_id="bad", turns=(), ground_truth_turns=())
    with pytest.raises(DatasetError):
        driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
        asyncio.run(driver._session(_wl(), 0, 0.0, item, "t0", None, _EchoClient(), _noop))


def test_session_ragged_ground_truth_raises_dataset_error():
    item = DatasetItem(
        request_id="rag",
        turns=(Envelope(request_id="rag.0", contents=[TextPayload(text="u0")]),
               Envelope(request_id="rag.1", contents=[TextPayload(text="u1")])),
        ground_truth_turns=("g0",))               # only one GT for two input turns
    with pytest.raises(DatasetError):
        driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
        asyncio.run(driver._session(_wl(), 0, 0.0, item, "t0", None, _EchoClient(), _noop))


def test_single_turn_request_carries_messages_metadata():
    eng = _engine(dataset=_OneItemDataset(), seed=0, workload=_wl())
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_EchoClient(), events.append, Split.DEV))
    assert events[0].input.metadata["messages"] == [{"role": "user", "content": "Q?"}]


def test_build_engine_from_config():
    """A Driver holds what it is built with and nothing a drive passes it."""
    eng = Driver(request_timeout_sec=5.0)
    assert isinstance(eng, Driver)
    assert eng.request_timeout_sec == 5.0
    assert not {"tenants", "contract", "seed"} & set(vars(eng))


def test_a_driver_is_built_only_with_the_grace_its_caller_names():
    """No driver has a grace of its own: a graded drive's is the drive stage's
    (``hls.pipeline.pipelines.REQUEST_TIMEOUT_SEC``), a calibration serving's its campaign's."""
    from hls.spec.drivers import TrajectorySessionDriver, driver_for

    with pytest.raises(TypeError, match="request_timeout_sec"):
        Driver()
    with pytest.raises(TypeError, match="request_timeout_sec"):
        TrajectorySessionDriver(framework=None, tool_runner=None, max_turns=1, max_concurrent_llm=None)
    with pytest.raises(TypeError, match="request_timeout_sec"):
        driver_for(None)


def test_the_drives_seed_draws_each_tenants_arrival_schedule(monkeypatch):
    """The seed a drive is given, and no other, seeds the arrival schedule: each tenant's is
    drawn at ``tenant_seed(seed, tenant.id)``."""
    from hls.core.seeds import tenant_seed

    drawn: list[int] = []
    real = AllAtOnce.schedule

    def recording(self, n, seed):
        drawn.append(seed)
        return real(self, n, seed)

    monkeypatch.setattr(AllAtOnce, "schedule", recording)
    two = [Tenant(id=tid, data_source=_dummy_ds(_OneItemDataset()), workload=_wl(), verifiers=[])
           for tid in ("t0", "corp-a")]
    events: list[TraceEvent] = []
    driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    asyncio.run(driver._drive(_EchoClient(), events.append, Split.DEV, tenants=two, seed=99))
    assert drawn == [tenant_seed(99, "t0"), tenant_seed(99, "corp-a")] and drawn[0] == 99
    assert len(events) == 2


# ---------------------------------------------------------------------------
# Fail-fast circuit-breaker
# ---------------------------------------------------------------------------


def _serial_wl():
    """All-at-once arrival against a client that answers without ever suspending, so
    each request folds back before the next session is scheduled and the
    circuit-breaker trips deterministically, without wall-clock waits.

    The ordering comes from the client, not from the workload: ``Workload`` has no
    ``max_concurrent`` field, and pydantic drops the kwarg silently."""
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(), slo=NoSLO())


class _NItemDataset(Dataset):
    """N single-turn items r0..r{N-1}. N is a ClassVar the factory overrides (a plain
    class attr would be read as a pydantic model field)."""
    name = "stub-n"
    version = "v0"
    N: ClassVar[int] = 1

    def items_at(self, mount_dir, split):
        for i in range(self.N):
            env = Envelope(request_id=f"r{i}", contents=[TextPayload(text="Q?")])
            yield DatasetItem(request_id=f"r{i}", turns=(env,), ground_truth_turns=("A",))


def _n_item_dataset(n):
    return type("_Ds", (_NItemDataset,), {"N": n})()


class _DeadClient:
    """A crashed contestant server: every request faults (never recovers)."""
    def __init__(self):
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        raise ArtifactTransportError("connection refused (server is dead)")


def test_circuit_breaker_aborts_drive_on_consecutive_faults():
    """A dead server (every request faults) must NOT drive all N requests each up to
    the per-request timeout. After ``fault_abort_threshold`` consecutive faults the
    breaker trips: no more requests are SENT, yet the drive still returns a Trace
    whose remaining requests are marked as faults (so a crashed server scores LOW,
    it does not error the rollout)."""
    eng = _engine(dataset=_n_item_dataset(8), seed=0, workload=_serial_wl(), fault_abort_threshold=3)
    dead = _DeadClient()
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(dead, events.append, Split.DEV))

    # Only the threshold's worth of requests ever hit the (dead) server -- the rest
    # were short-circuited, not sent (this is the fail-fast: no thousands of hangs).
    assert dead.calls == 3
    # The Trace is complete: every item is present, all as faults (none scored good).
    assert len(events) == 8
    assert all(ev.response is None and ev.error for ev in events)
    # The undriven tail carries the circuit-breaker marker, not a real transport error.
    # "aborted:", not "circuit-breaker": the latter also matches the string a request
    # gets when the breaker cut it short mid-flight, which is a different claim.
    aborted = [ev for ev in events if ev.error.startswith("aborted:")]
    assert len(aborted) == 8 - 3


def test_circuit_breaker_does_not_trip_below_threshold():
    """Sporadic faults below the threshold must NOT abort: every request is still
    driven and the healthy responses land (no premature breaker)."""
    eng = _engine(dataset=_n_item_dataset(6), seed=0, workload=_serial_wl(),
                  fault_abort_threshold=50)  # default-ish: never reached here
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_FlakyClient(), events.append, Split.DEV))
    # _FlakyClient faults ONLY the first request; all 6 items are driven.
    assert len(events) == 6
    assert sum(1 for ev in events if ev.error) == 1
    assert sum(1 for ev in events if ev.response is not None) == 5


# ---------------------------------------------------------------------------
# Drive-completion deadline
# ---------------------------------------------------------------------------


class _SlowClient:
    """Echoes after sleeping ``delay`` -- a slow server whose request can outlive
    the drive deadline."""
    def __init__(self, delay):
        self.delay = delay
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        await asyncio.sleep(self.delay)
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text="A")]))


class _TwoTurnDataset(Dataset):
    """One two-turn item (so a session sends turn 1 after turn 0)."""
    name = "stub-2t"
    version = "v0"

    def items_at(self, mount_dir, split):
        env = Envelope(request_id="r0", contents=[TextPayload(text="Q?")])
        yield DatasetItem(request_id="r0", turns=(env, env),
                          ground_truth_turns=("A", "A"))


_DEADLINE_ERR = ("transport: request did not complete by the drive deadline "
                 "(last scheduled arrival + {g}s, min {g}s per request)")


def test_drive_deadline_faults_requests_that_outlive_it():
    """A request slower than (last scheduled arrival + grace) is faulted with the
    deadline string, not a real transport error, and stays out of the latency
    sample (t_recv is None)."""
    eng = _engine(dataset=_OneItemDataset(), workload=_wl(), request_timeout_sec=0.05)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_SlowClient(0.5), events.append, Split.DEV))

    assert len(events) == 1
    ev = events[0]
    assert ev.response is None
    assert ev.error == _DEADLINE_ERR.format(g="0.05")
    assert ev.t_recv is None


def test_followup_turn_past_deadline_still_gets_full_grace():
    """A follow-up turn sent after the drive deadline is still SENT and gets the
    full grace (the floor), never a squeezed or zero allowance."""
    eng = _engine(dataset=_TwoTurnDataset(), workload=_wl(), request_timeout_sec=0.05)
    client = _SlowClient(0.3)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(client, events.append, Split.DEV))

    assert client.calls == 2                       # turn 1 hits the network too
    assert len(events) == 2
    assert all(ev.error == _DEADLINE_ERR.format(g="0.05") for ev in events)


def test_interturn_gap_longer_than_grace_is_not_a_fault():
    """A healthy server whose session thinks longer than the grace between turns
    must not be faulted: the floor gives the late turn its full grace."""
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=ConstantInterTurn(gap=0.15),
                        output_length_cap=Uncapped(), slo=NoSLO())
    eng = _engine(dataset=_TwoTurnDataset(), workload=workload, request_timeout_sec=0.05)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_SlowClient(0.01), events.append, Split.DEV))

    assert len(events) == 2
    assert all(ev.error is None for ev in events)
    assert all(ev.response is not None for ev in events)


def test_drive_clears_client_request_cap():
    """The driver drops the client's per-request cap (clear_request_cap) so a
    queued request can wait out the drive deadline; a client without the method is
    driven fine via the getattr guard."""
    class _ClearCapEcho(_EchoClient):
        def __init__(self):
            self.cleared = False

        def clear_request_cap(self):
            self.cleared = True

    eng = _engine(dataset=_OneItemDataset(), workload=_wl())
    client = _ClearCapEcho()
    asyncio.run(eng.drive(client, _noop, Split.DEV))
    assert client.cleared is True

    # A plain _EchoClient (no clear_request_cap) still drives via the getattr guard.
    events: list[TraceEvent] = []
    eng2 = _engine(dataset=_OneItemDataset(), workload=_wl())
    asyncio.run(eng2.drive(_EchoClient(), events.append, Split.DEV))
    assert len(events) == 1


def test_deadline_anchors_to_last_scheduled_arrival(monkeypatch):
    """The deadline is (last scheduled arrival + grace), NOT per request: the
    early-arriving session has room to spare while the late one, sent near the
    last arrival, misses."""
    monkeypatch.setattr(AllAtOnce, "schedule", lambda self, n, seed: [0.0, 0.4])
    eng = _engine(dataset=_TwoItemDataset(), workload=_wl(), request_timeout_sec=0.1)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_SlowClient(0.25), events.append, Split.DEV))

    assert len(events) == 2
    errored = [ev for ev in events if ev.error]
    assert len(errored) == 1
    assert errored[0].error == _DEADLINE_ERR.format(g="0.1")


def test_the_deadline_anchors_on_the_last_scheduled_arrival():
    """The deadline anchor is the whole schedule's last arrival, wherever the earlier
    arrivals fall."""
    from hls.spec.drivers import _deadline_anchor
    assert _deadline_anchor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]) == 5.0
    assert _deadline_anchor([3.0]) == 3.0


class _FollowUpsFaultClient(_EchoClient):
    """Answers every first turn and faults every follow-up."""

    async def submit(self, env, *, attempt=None):
        if env.metadata["turn_index"] > 0:
            raise ArtifactTransportError("connection refused (server is dead)")
        return await super().submit(env)


def test_inter_turn_think_time_wakes_on_stop():
    """A tripped drive must not sit on inter-turn think-time it will never use: the think sleep
    wakes on the stop event like the arrival wait does. Tenant X (short think) trips the breaker
    WHILE tenant Y is mid a 30s think; the drive must finalize at once -- were the sleep not
    stop-aware it would hang out Y's full delay."""
    def _2t_tenant(tid, gap):
        return Tenant(id=tid, data_source=_dummy_ds(_TwoTurnDataset()),
                      workload=Workload(arrival=AllAtOnce(),
                                        inter_turn_delay=ConstantInterTurn(gap=gap),
                                        output_length_cap=Uncapped(), slo=NoSLO()),
                      verifiers=[])
    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC, fault_abort_threshold=1)
    events: list[TraceEvent] = []

    async def _run():
        await asyncio.wait_for(
            eng._drive(_FollowUpsFaultClient(), events.append, Split.DEV,
                       tenants=[_2t_tenant("x", 0.1), _2t_tenant("y", 30.0)], seed=0), timeout=8.0)

    asyncio.run(_run())               # must NOT time out: Y's 30s think woke on the trip
    by_turn = {(ev.input.metadata["tenant_id"], ev.turn_index): ev.error for ev in events}
    assert by_turn[("x", 0)] is None and by_turn[("y", 0)] is None
    assert by_turn[("x", 1)].startswith("transport:")     # the fault that tripped it
    assert by_turn[("y", 1)].startswith("aborted:")       # woken, and never sent


# ---------------------------------------------------------------------------
# Dispatch time-cap + Workload.trim_rampdown
# ---------------------------------------------------------------------------


def _split(s):
    """Convert a split string to Split enum (as ``Driver.run`` does)."""
    return Split(s)


def _mt_ds():
    """One 3-turn in-memory conversation for time-cap tests."""
    from tests.support.inline_dataset import InlineMultiTurnDataset
    return InlineMultiTurnDataset(conversations=[
        ("c0", [("u1", "g1"), ("u2", "g2"), ("u3", "g3")])])


def test_rampdown_budget_trims_by_the_nominal_timeline_only():
    """rampdown_turn_budget: turn 0 always exists; a follow-up exists iff its nominal
    dispatch (session arrival + cumulative think-time, zero service time) falls before
    the merged schedules' last arrival. Exact arrival at the cutoff trims (>=), and
    tenants without trim_rampdown get no entry (serve everything)."""
    from hls.spec.drivers import rampdown_turn_budget

    trim_wl = Workload(arrival=AllAtOnce(), inter_turn_delay=ConstantInterTurn(gap=1.0),
                       output_length_cap=ShortOutputCap(cap_tokens=8), slo=NoSLO(),
                       trim_rampdown=True)
    plain_wl = Workload(arrival=AllAtOnce(), inter_turn_delay=ConstantInterTurn(gap=1.0),
                        output_length_cap=ShortOutputCap(cap_tokens=8), slo=NoSLO())
    item = next(iter(_mt_ds().items_at(None, _split("dev"))))          # 3 turns
    trim = Tenant(id="trim", data_source=_dummy_ds(_mt_ds()), workload=trim_wl,
                  verifiers=[])
    plain = Tenant(id="plain", data_source=_dummy_ds(_mt_ds()), workload=plain_wl,
                   verifiers=[])
    # Cutoff = plain's arrival at 6.5. trim session 0 (arrival 0): follow-ups at 1, 2
    # -> all 3 turns. Session 1 (arrival 5): turn 1 at 6.0 < 6.5, turn 2 at 7.0 >= 6.5
    # -> 2 turns. Session 2 (arrival 5.5): turn 1 at 6.5 >= 6.5 -> turn 0 only.
    scheduled = [(trim, [item, item, item], [0.0, 5.0, 5.5]),
                 (plain, [item], [6.5])]
    budget = rampdown_turn_budget(scheduled, seed=0)
    assert budget == {("trim", 0): 3, ("trim", 1): 2, ("trim", 2): 1}


def test_session_serves_exactly_the_turn_budget_with_no_fault():
    """_session drives the leading n_turns of the item and records no fault for the
    turns that do not exist in this draw."""
    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    wl = Workload(arrival=AllAtOnce(), inter_turn_delay=ConstantInterTurn(gap=0.0),
                  output_length_cap=ShortOutputCap(cap_tokens=8), slo=NoSLO())
    item = next(iter(_mt_ds().items_at(None, _split("dev"))))
    events = []
    asyncio.run(eng._session(wl, 0, 0.0, item, "t0", 8, _EchoClient(),
                             events.append, seed=0, n_turns=1))
    assert [e.turn_index for e in events] == [0]
    assert all(e.error is None for e in events)
    events.clear()
    asyncio.run(eng._session(wl, 0, 0.0, item, "t0", 8, _EchoClient(),
                             events.append, seed=0, n_turns=None))
    assert [e.turn_index for e in events] == [0, 1, 2]


def test_trim_rampdown_disabled_by_default_is_byte_identical():
    """Default Workload has trim_rampdown=False; a multi-turn AllAtOnce drive still
    emits every turn (no time-cap wired)."""
    wl = Workload(arrival=AllAtOnce(), inter_turn_delay=ConstantInterTurn(gap=0.0),
                  output_length_cap=ShortOutputCap(cap_tokens=8), slo=NoSLO())
    assert wl.trim_rampdown is False
    tenants = [Tenant(id="t0", data_source=DataSource(
        dataset=_mt_ds(), split=SplitSpec(dev=Count(value=1), test=Count(value=1))),
        workload=wl, verifiers=[])]
    events = []
    driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    asyncio.run(driver._drive(_EchoClient(), events.append, _split("dev"), tenants=tenants, seed=0))
    assert [e.turn_index for e in events] == [0, 1, 2]


def test_trim_rampdown_across_a_real_schedule():
    """With trim_rampdown=True and a spread arrival, follow-up turns whose NOMINAL
    dispatch falls past the last arrival are excluded from the draw; turn-0 arrivals
    all fire. No faults."""
    from hls.spec.workload import DeterministicArrival
    from tests.support.inline_dataset import InlineMultiTurnDataset
    # 4 sessions, arrivals at 0/0.1/0.2/0.3s; each is 2-turn with a 10s think-time,
    # so every follow-up nominally lands far past the last arrival (0.3s) -> all
    # excluded at draw time.
    convs = [(f"c{i}", [("u1", "g1"), ("u2", "g2")]) for i in range(4)]
    ds = InlineMultiTurnDataset(conversations=convs)
    wl = Workload(arrival=DeterministicArrival(gap=0.1),
                  inter_turn_delay=ConstantInterTurn(gap=10.0),
                  output_length_cap=ShortOutputCap(cap_tokens=8), slo=NoSLO(),
                  trim_rampdown=True)
    tenants = [Tenant(id="t0", data_source=DataSource(
        dataset=ds, split=SplitSpec(dev=Count(value=4), test=Count(value=4))),
        workload=wl, verifiers=[])]
    events = []
    driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    asyncio.run(driver._drive(_EchoClient(), events.append, _split("dev"), tenants=tenants, seed=0))
    assert sorted(e.turn_index for e in events) == [0, 0, 0, 0]   # every turn-0, no follow-ups
    assert all(e.error is None for e in events)


def test_trim_membership_is_independent_of_artifact_latency():
    """The decoupling the draw-time cut exists for: a slow artifact must serve the SAME
    request set as a fast one -- only later. A serve-time cut would shed the slow drive's
    follow-ups (service time pushes their dispatch past the cutoff); under draw-time
    membership the turn multiset is identical."""
    from hls.spec.workload import DeterministicArrival
    from tests.support.inline_dataset import InlineMultiTurnDataset

    class _SlowEcho(_EchoClient):
        async def submit(self, env, *, attempt=None):
            await asyncio.sleep(0.25)
            return await super().submit(env)

    def turn_multiset(client):
        convs = [(f"c{i}", [("u1", "g1"), ("u2", "g2")]) for i in range(2)]
        ds = InlineMultiTurnDataset(conversations=convs)
        # Arrivals at 0 / 0.5 s; think-time 0.2 s. Nominal follow-ups: session 0 at
        # 0.2 (< 0.5, kept), session 1 at 0.7 (>= 0.5, excluded) -- regardless of how
        # long the artifact takes to answer.
        wl = Workload(arrival=DeterministicArrival(gap=0.5),
                      inter_turn_delay=ConstantInterTurn(gap=0.2),
                      output_length_cap=ShortOutputCap(cap_tokens=8), slo=NoSLO(),
                      trim_rampdown=True)
        tenants = [Tenant(id="t0", data_source=DataSource(
            dataset=ds, split=SplitSpec(dev=Count(value=2), test=Count(value=2))),
            workload=wl, verifiers=[])]
        events = []
        driver = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
        asyncio.run(driver._drive(client, events.append, _split("dev"), tenants=tenants, seed=0))
        return sorted((e.session_id, e.turn_index) for e in events)

    fast = turn_multiset(_EchoClient())
    slow = turn_multiset(_SlowEcho())
    assert fast == slow
    assert [t for _, t in fast].count(1) == 1     # exactly one kept follow-up


# ---- context budget: a full conversation ENDS the session (no windowing) --------

class _SizedTurnsDataset(Dataset):
    """One session of ``n_turns`` user turns, each exactly ``turn_chars`` characters,
    so a test can place a conversation on either side of a character budget."""
    name: ClassVar[str] = "stub-sized"
    version: ClassVar[str] = "v0"

    n_turns: int = 4
    turn_chars: int = 100

    def items_at(self, mount_dir, split):
        turns = tuple(
            Envelope(request_id=f"r0.{k}",
                     contents=[TextPayload(text=str(k) * self.turn_chars)])
            for k in range(self.n_turns))
        yield DatasetItem(request_id="r0", turns=turns,
                          ground_truth_turns=tuple("A" for _ in turns))


class _SizedEcho:
    """An artifact whose every reply is ``reply_chars`` characters, and which records
    the conversation each request carried -- so a test can assert on the prompts as
    sent, not on a reconstruction of them."""

    def __init__(self, reply_chars: int) -> None:
        self.reply_chars = reply_chars
        self.sent: list[list[dict]] = []

    async def submit(self, env: Envelope, *, attempt=None) -> ArtifactResponse:
        self.sent.append(env.metadata["messages"])
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id,
            contents=[TextPayload(text="a" * self.reply_chars)]))


def _drive_with_budget(*, budget, n_turns=4, turn_chars=100, reply_chars=100):
    """Drive one session of sized turns against a sized artifact under ``budget``.
    Returns ``(client, events)``."""
    wl = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                  output_length_cap=Uncapped(), slo=NoSLO(),
                  history_char_budget=budget)
    eng = _engine(dataset=_SizedTurnsDataset(n_turns=n_turns, turn_chars=turn_chars),
                  workload=wl)
    client = _SizedEcho(reply_chars)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(client, events.append, Split.DEV))
    return client, events


def test_a_session_under_its_context_budget_serves_every_turn():
    """The budget is a capacity, not a per-request policy: a conversation that fits
    is driven to its last turn and nothing is trimmed."""
    client, events = _drive_with_budget(budget=100_000)
    assert [ev.turn_index for ev in events] == [0, 1, 2, 3]
    assert all(ev.error is None for ev in events)
    # The 4th request carries the whole conversation: 4 user turns + 3 replies.
    assert len(client.sent[-1]) == 7


def test_a_session_ends_at_the_first_turn_that_would_overflow_its_budget():
    """A conversation that does not fit is OVER -- the overflowing turn is never
    sent and neither is any turn after it. Turn k's request carries k+1 user turns of
    100 chars and k replies of 100, so it measures 100*(2k+1): turns 0 and 1 fit a
    350-character budget, turn 2 (500) does not."""
    client, events = _drive_with_budget(budget=350)
    assert [ev.turn_index for ev in events] == [0, 1]
    assert [len(m) for m in client.sent] == [1, 3]     # nothing sent for turns 2, 3


def test_turns_left_unserved_by_the_budget_are_not_failures():
    """An unserved turn produces NO event, exactly like a turn trim_rampdown excluded:
    it is not a fault, not an SLO miss, and leaves no half-recorded request behind."""
    _, events = _drive_with_budget(budget=350)
    assert len(events) == 2
    assert all(ev.error is None and ev.response is not None for ev in events)
    assert all(ev.t_recv is not None for ev in events)          # in the latency sample
    # Only the session's arrival is planned; a follow-up carries no planned send.
    assert [ev.t_planned_send is not None for ev in events] == [True, False]


def test_every_prompt_a_session_sends_extends_the_previous_one():
    """No prompt is ever a rewritten shape: each turn's conversation is the previous
    turn's plus the reply and the new question, so the served prefix only grows. This
    is what a sliding window would have broken."""
    for budget in (100_000, 350):
        client, _ = _drive_with_budget(budget=budget)
        for earlier, later in zip(client.sent, client.sent[1:]):
            assert later[:len(earlier)] == earlier
            assert len(later) == len(earlier) + 2      # the reply, then the next question
        assert all(m[0]["role"] == "user" for m in client.sent)


def test_a_first_turn_over_the_budget_is_a_corpus_defect_and_raises():
    """An opening turn is corpus text alone -- no generated reply is in it -- so a
    session that cannot serve even its first turn means the corpus does not fit the
    tenant. That is raised, not silently served as an empty session."""
    with pytest.raises(DatasetError, match="FIRST turn is already"):
        _drive_with_budget(budget=50)


def test_no_budget_serves_every_turn_whatever_the_replies():
    """None means no cap: the same session with replies far larger than any budget
    still serves every turn."""
    client, events = _drive_with_budget(budget=None, reply_chars=10_000)
    assert [ev.turn_index for ev in events] == [0, 1, 2, 3]
    assert len(client.sent[-1]) == 7


def test_history_chars_counts_a_falsy_content_value():
    """history_chars reads content through message_content, the same rule the wire
    protocols use, so a present-but-falsy value like 0 counts as one character rather
    than being treated as empty."""
    from hls.spec.drivers import history_chars

    assert history_chars([{"role": "user", "content": 0}]) == 1


# ---------------------------------------------------------------------------
# The failed event's diagnostic record (TraceEvent.failure)
# ---------------------------------------------------------------------------


class _PartialThenBreakClient:
    """A server that streams some tokens and then breaks the connection, the way the
    HTTP client leaves an attempt behind: frames noted, partial decoded, then raise."""

    def __init__(self, fault=None, status_code=None):
        self.fault = fault or ArtifactTransportError("connection reset")
        self.status_code = status_code

    async def submit(self, env, *, attempt=None):
        from hls.core.trace import PartialOutput
        if attempt is not None:
            attempt.note_frame(harness_now())
            attempt.note_frame(harness_now())
            attempt.partial = PartialOutput(token_ids=[7, 8], text="ab",
                                            finish_reason=None)
            attempt.status_code = self.status_code
        raise self.fault


class _PartialThenStallClient:
    """Streams some tokens, then never finishes -- the drive deadline abandons it."""

    async def submit(self, env, *, attempt=None):
        from hls.core.trace import PartialOutput
        if attempt is not None:
            attempt.note_frame(harness_now())
            attempt.partial = PartialOutput(token_ids=[7], text="a", finish_reason=None)
        await asyncio.sleep(3600)


def _one_failure(client, *, grace=REQUEST_TIMEOUT_SEC):
    """Drive one single-turn request against ``client`` and return its event."""
    eng = _engine(dataset=_OneItemDataset(), workload=_wl(), request_timeout_sec=grace)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(client, events.append, Split.DEV))
    assert len(events) == 1
    return events[0]


def test_a_successful_event_carries_no_failure_record():
    """The success shape is unchanged: everything downstream keys on response/error,
    and a failure record on a success would be a second, contradictory account."""
    ev = _one_failure(_EchoClient())
    assert ev.error is None
    assert ev.failure is None
    assert ev.response is not None


def test_a_served_rejection_is_recorded_as_a_server_error_with_its_status():
    from hls.core.trace import FailureReason

    ev = _one_failure(_PartialThenBreakClient(status_code=503))
    assert ev.failure.reason == FailureReason.SERVER_ERROR
    assert ev.failure.observed.status_code == 503


def test_a_dropped_connection_is_recorded_as_transport_with_its_partial_output():
    """A request that got two tokens out reads differently from one that got none, though
    both fault as "transport"."""
    from hls.core.trace import FailureReason

    ev = _one_failure(_PartialThenBreakClient())
    assert ev.failure.reason == FailureReason.TRANSPORT
    assert ev.failure.observed.n_frames == 2
    assert ev.failure.observed.t_first_frame is not None
    assert ev.failure.observed.partial.token_ids == [7, 8]
    assert ev.failure.observed.partial.text == "ab"
    assert ev.failure.observed.partial.finish_reason is None


def test_a_non_conformant_response_is_recorded_as_protocol():
    from hls.core.faults import ArtifactProtocolError
    from hls.core.trace import FailureReason

    ev = _one_failure(_PartialThenBreakClient(
        fault=ArtifactProtocolError("stream ended without finish_reason")))
    assert ev.failure.reason == FailureReason.PROTOCOL
    assert ev.failure.observed.partial.token_ids == [7, 8]


def test_a_deadline_miss_keeps_the_partial_output_of_the_cancelled_request():
    """The mode with no other evidence at all: the submit is cancelled, so this record
    is the only thing that says the server had started answering."""
    from hls.core.trace import FailureReason

    ev = _one_failure(_PartialThenStallClient(), grace=0.05)
    assert ev.failure.reason == FailureReason.DEADLINE
    assert ev.failure.observed.n_frames == 1
    assert ev.failure.observed.partial.token_ids == [7]


def test_a_failure_records_when_the_harness_gave_up_not_when_it_gave_an_answer():
    """t_recv stays None so the event is out of the latency sample; t_abandoned is
    where the elapsed time of a failure lives instead."""
    ev = _one_failure(_PartialThenStallClient(), grace=0.05)
    assert ev.t_recv is None
    assert ev.failure.t_abandoned - ev.t_send >= 0.05


def test_a_request_the_breaker_skipped_says_so_and_observed_nothing():
    """Distinguishable from a request that was sent and failed: no frames, no status,
    and its own reason -- so a census cannot mistake breaker fallout for real faults."""
    from hls.core.trace import FailureReason

    eng = _engine(dataset=_n_item_dataset(4), seed=0, workload=_serial_wl(), fault_abort_threshold=1)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_DeadClient(), events.append, Split.DEV))

    reasons = [ev.failure.reason for ev in events]
    assert reasons[0] == FailureReason.TRANSPORT
    assert set(reasons[1:]) == {FailureReason.CIRCUIT_BREAKER}
    skipped = events[-1]
    assert skipped.failure.observed.n_frames == 0
    assert skipped.failure.observed.t_first_frame is None
    assert skipped.failure.observed.status_code is None
    assert skipped.failure.observed.partial is None


def test_every_failed_event_carries_both_the_fault_string_and_the_record():
    """``error`` stays the single home of the verbatim fault string (the verify census
    groups on it); the record is what the string cannot say."""
    eng = _engine(dataset=_n_item_dataset(3), seed=0, workload=_serial_wl(), fault_abort_threshold=2)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_DeadClient(), events.append, Split.DEV))

    assert all(ev.error and ev.failure is not None for ev in events)
    assert all(ev.response is None for ev in events)


def test_the_failure_record_round_trips_through_the_persisted_trace():
    """The record has to survive ``trace.model_dump_json`` -- the run's only artifact
    once the containers are gone -- not just live in memory."""
    from hls.core.trace import FailureReason, Trace

    ev = _one_failure(_PartialThenBreakClient(status_code=500))
    reloaded = Trace.model_validate_json(Trace(events=[ev]).model_dump_json())
    failure = reloaded.events[0].failure
    assert failure.reason == FailureReason.SERVER_ERROR
    assert failure.observed.status_code == 500
    assert failure.observed.partial.token_ids == [7, 8]
    assert failure.t_abandoned == ev.failure.t_abandoned


class _ManyTurnDataset(Dataset):
    """One item with eight turns, so a session's turns chain far enough to outrun any
    anchor if each is allowed a fresh grace."""
    name = "stub-8t"
    version = "v0"

    def items_at(self, mount_dir, split):
        env = Envelope(request_id="r0", contents=[TextPayload(text="Q?")])
        yield DatasetItem(request_id="r0", turns=(env,) * 8,
                          ground_truth_turns=("A",) * 8)


def test_a_chain_of_slow_turns_cannot_outrun_the_drive_wall():
    """The bound the per-request floor does not give on its own.

    Every turn is slower than the grace, so each claims a full grace and the next is
    sent later still. Left to the floor alone the drive runs turns x grace -- which on a
    real multi-turn task is hours, and the job's own wall clock kills the run before it
    can be scored at all. The wall stops it, and every remaining request is faulted, so
    the trace stays complete and the gated scorer still prices them.

    Two fault strings, not one: a turn already in flight misses its own per-request
    deadline, while the turns behind it are never sent and carry the wall's string. The
    census reads them apart on purpose -- a server that never kept up is not a server
    that stopped answering."""
    eng = _engine(dataset=_ManyTurnDataset(), workload=_wl(), request_timeout_sec=0.05)
    events: list[TraceEvent] = []

    # On a virtual clock (tests.support.virtual_clock), so the drive's length is what
    # its bounds made it however loaded the machine is. Its clock starts at 0.
    with virtual_clock() as runner:
        runner.run(eng.drive(_SlowClient(0.3), events.append, Split.DEV))
        elapsed = runner.get_loop().time()

    assert len(events) == 8
    assert all(ev.error in (_DEADLINE_ERR.format(g="0.05"), WALL_ERROR)
               for ev in events), [ev.error for ev in events]
    assert any(ev.error == WALL_ERROR for ev in events), (
        "nothing was cut by the wall, so this drive did not exercise it")
    # Eight turns each claiming the full 0.05 s floor would take 0.4 s; the wall ends
    # the drive _WALL_GRACES graces past the last arrival, which is at 0.
    assert elapsed == pytest.approx(_WALL_GRACES * 0.05), f"drive ran {elapsed:.3f}s"


def test_the_wall_leaves_the_per_request_floor_intact_inside_it():
    """A request sent before the wall still gets its full grace, so the wall bounds the
    drive without shortening any individual request's allowance.

    The allowance also names which bound cut it, so a request the wall ended is not
    filed under the grace it never reached."""
    d = _Deadline(t=100.0, grace=10.0, error="x", wall=_DriveWall(t=140.0))

    assert d.allowance(0.0) == (100.0, "x")   # sent early: waits out the whole deadline
    assert d.allowance(95.0) == (10.0, "x")   # sent near the deadline: the floor
    assert d.allowance(120.0) == (10.0, "x")  # sent past it, inside the wall: still the floor
    assert d.allowance(135.0) == (5.0, WALL_ERROR)   # squeezed only by the wall itself
    assert d.allowance(145.0) == (0.0, WALL_ERROR)   # past the wall: fails immediately


# ---------------------------------------------------------------------------
# Abandoning requests that are in flight when the breaker trips
# ---------------------------------------------------------------------------


class _OneHangsClient:
    """A server that stops answering: the first request never returns, the rest fault.

    The shape of a real drive's worst case -- connections dropped for most requests
    while one is left hanging -- reduced to the two behaviours that matter."""

    def __init__(self):
        self.calls = 0
        self.hung = asyncio.Event()           # set once the first request is stuck
        self.released = asyncio.Event()       # never set: the request never returns

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        if self.calls == 1:
            self.hung.set()
            await self.released.wait()
        raise ArtifactTransportError("connection refused (server is dead)")


def test_a_request_in_flight_when_the_breaker_trips_is_abandoned_at_once():
    """The breaker gates the next send, so a request already outstanding must not keep
    its own allowance: past the drive deadline that allowance is the full grace
    floor, and a drive that had given up could not return until the last one expired,
    possibly after the run's remaining wall clock.

    The grace here is 30 s against a drive that should finish in milliseconds, so the
    elapsed assertion fails if anything waits it out."""
    eng = _engine(dataset=_n_item_dataset(6), seed=0, workload=_wl(), fault_abort_threshold=3,
                  request_timeout_sec=30.0)
    client = _OneHangsClient()
    events: list[TraceEvent] = []

    t0 = harness_now()
    asyncio.run(eng.drive(client, events.append, Split.DEV))
    elapsed = harness_now() - t0

    assert client.hung.is_set(), "no request was ever left outstanding, so nothing " \
                                 "could have been abandoned"
    assert elapsed < 5.0, f"drive waited out the abandoned request ({elapsed:.1f}s)"
    abandoned = [ev for ev in events if ev.error and "abandoned" in ev.error]
    assert len(abandoned) == 1, "the hung request was not recorded as abandoned"
    assert abandoned[0].failure.reason is FailureReason.CIRCUIT_BREAKER
    assert len(events) == 6


def test_an_abandoned_request_is_told_apart_from_one_that_was_never_sent():
    """Both are the breaker's doing and share a reason code, but only one was asked of
    the server. The failure census groups on the error string, so merging them would
    report requests the harness never sent as evidence about the artifact."""
    eng = _engine(dataset=_n_item_dataset(6), seed=0, workload=_wl(), fault_abort_threshold=3,
                  request_timeout_sec=30.0)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_OneHangsClient(), events.append, Split.DEV))

    strings = {ev.error for ev in events if ev.error}
    abandoned = {s for s in strings if "abandoned" in s}
    skipped = {s for s in strings if s.startswith("aborted:")}
    assert len(abandoned) == 1, "the request cut short mid-flight has no string"
    assert len(skipped) == 1, "the requests never sent have no string"
    assert abandoned.isdisjoint(skipped), "one string is covering both conditions"


def test_an_abandoned_request_is_stamped_when_the_harness_gave_up():
    """SLO.censored_ratio charges an unanswered request from when it was due to its
    t_abandoned. Waiting out the allowance would bill the server for the harness's own
    delay and drag the run's reported upper bound down with it."""
    eng = _engine(dataset=_n_item_dataset(6), seed=0, workload=_wl(), fault_abort_threshold=3,
                  request_timeout_sec=30.0)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_OneHangsClient(), events.append, Split.DEV))

    ev = next(e for e in events if e.error and "abandoned" in e.error)
    assert ev.failure.t_abandoned - ev.t_send < 5.0


class _MostlyHangClient:
    """Most sessions hang; one refuses. The hangs expire on their own allowance, and
    it is those deadline misses that eventually trip the breaker -- so every hung
    request ran its full course and none was cut short."""

    def __init__(self, n_hang: int = 5):
        self.calls = 0
        self.n_hang = n_hang
        self.released = asyncio.Event()       # never set

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        if self.calls <= self.n_hang:
            await self.released.wait()
        raise ArtifactTransportError("connection refused")


def test_a_natural_deadline_miss_after_the_trip_is_not_called_an_abandonment():
    """Deciding from the drive-wide `aborted` flag relabels every later deadline miss.

    Here five identical requests all expire on the same 5 s allowance; the breaker
    trips partway through folding them back, on those very misses. Which of them were
    still outstanding at that instant is an artefact of fold-back order, and none of
    them was shortened by anything. Recording some as abandoned would claim the
    harness stopped waiting on requests that in fact ran their full course, and would
    invert the upper-bound argument for them: censored_ratio charges to t_abandoned,
    which for these is the honest end of their own allowance."""
    eng = _engine(dataset=_n_item_dataset(6), seed=0, workload=_wl(), fault_abort_threshold=3,
                  request_timeout_sec=5.0)
    events: list[TraceEvent] = []
    asyncio.run(eng.drive(_MostlyHangClient(), events.append, Split.DEV))

    timed_out = [ev for ev in events
                 if ev.failure and ev.failure.reason is FailureReason.DEADLINE]
    assert len(timed_out) == 5, "the hung requests did not all reach their deadline"
    # Every one held its whole allowance, so none of them was cut short.
    assert all(ev.failure.t_abandoned - ev.t_send > 4.5 for ev in timed_out)
    assert not any(ev.error and "abandoned" in ev.error for ev in events), \
        "a request that ran its full allowance was recorded as abandoned"


class _NeverAnswersClient:
    """A server that accepts every request and answers none, so no submit ever folds
    back and the fault-run breaker never moves."""

    def __init__(self):
        self.calls = 0

    async def submit(self, env, *, attempt=None):
        self.calls += 1
        await asyncio.Event().wait()


def test_a_server_that_goes_silent_is_abandoned_at_the_stall_threshold(monkeypatch):
    """The stall trip reaches the same collapse as a fault-run trip.

    Every request is sent at once and none comes back. Without the collapse the trip
    only stops future sends, and the drive waits each request's 30 s allowance out,
    which in production is 10,800 s past the last arrival."""
    import functools

    from hls.spec import drivers
    monkeypatch.setattr(drivers, "_DriveState",
                        functools.partial(drivers._DriveState, stall_abort_sec=0.5))
    eng = _engine(dataset=_n_item_dataset(6), seed=0, workload=_wl(), fault_abort_threshold=3,
                  request_timeout_sec=30.0)
    client = _NeverAnswersClient()
    events: list[TraceEvent] = []

    t0 = harness_now()
    asyncio.run(eng.drive(client, events.append, Split.DEV))
    elapsed = harness_now() - t0

    assert client.calls == 6
    # The watchdog polls once a second at the shortest, so the trip lands near 1 s.
    assert elapsed < 5.0, f"drive waited out the silent requests ({elapsed:.1f}s)"
    assert len(events) == 6
    assert all(ev.error and "abandoned" in ev.error for ev in events), \
        [ev.error for ev in events]
    assert all(ev.failure.reason is FailureReason.CIRCUIT_BREAKER for ev in events)


def test_collapsing_an_already_expired_window_reports_that_it_changed_nothing():
    """`_collapse` is what decides whether a request was cut short or ran its course,
    so it has to answer for a window whose deadline has already passed: bringing that
    one 'forward' takes nothing from it, and saying otherwise is what relabels a
    natural deadline miss as an abandonment."""
    from hls.spec.drivers import _collapse

    async def _probe():
        async with asyncio.timeout(60.0) as ample:
            first = _collapse(ample)                  # genuinely shortens it
        loop = asyncio.get_running_loop()
        async with asyncio.timeout(None) as spent:
            spent.reschedule(loop.time() - 1.0)       # already past its deadline
            second = _collapse(spent)
            spent.reschedule(loop.time() + 60.0)      # don't fire on the way out
        return first, second

    shortened, already_gone = asyncio.run(_probe())
    assert shortened is True
    assert already_gone is False


def test_the_breakers_census_excludes_the_requests_it_abandoned(caplog, monkeypatch):
    """The abort line reports "%d/%d completed submits faulted". Folding the requests
    the breaker itself cancelled back into those counters moves them after the line
    that already published them, and can emit fresh "request fault #N" lines for
    requests the harness killed rather than the server."""
    import hls.spec.drivers as drivers_mod
    states: list = []

    class _Recorded(drivers_mod._DriveState):
        def __init__(self, **kw):
            super().__init__(**kw)
            states.append(self)

    monkeypatch.setattr(drivers_mod, "_DriveState", _Recorded)

    eng = _engine(dataset=_n_item_dataset(6), seed=0, workload=_wl(), fault_abort_threshold=3,
                  request_timeout_sec=30.0)
    events: list[TraceEvent] = []
    with caplog.at_level("WARNING", logger="hls.spec.drivers"):
        asyncio.run(eng.drive(_OneHangsClient(), events.append, Split.DEV))

    lines = [r.getMessage() for r in caplog.records]
    abort_at = next(i for i, m in enumerate(lines) if "aborting drive" in m)
    after = [m for m in lines[abort_at + 1:] if "request fault #" in m]
    assert not after, f"faults logged for harness-cancelled requests: {after}"

    # The counters themselves, not just the logging: the abort line publishes them,
    # so anything added afterwards makes driver.log disagree with the trace.
    assert states, "the drive built no state to inspect"
    st = states[-1]
    abandoned = [ev for ev in events if ev.error and "abandoned" in ev.error]
    server_faults = [ev for ev in events
                     if ev.error and not ev.error.startswith(("aborted:", "abandoned:"))]
    assert abandoned, "guard assumption: something was abandoned"
    assert st._faults_total == len(server_faults), (
        f"breaker counted {st._faults_total} faults for {len(server_faults)} server "
        f"faults; it folded in {len(abandoned)} request(s) the harness cancelled")
