"""Verdict.failure_mode: "correctness" vs "failed_requests" labeling.

A run zeroed by errored (timed-out) requests while every verifier gate passed must
not say its answers were wrong. These tests pin the split:

- ANY errored event -> the fast path: verification (and its reference-engine
  deploys) is skipped outright, failure_mode="failed_requests", with the verbatim
  error-string census in diagnostics["failed_requests"] and a "verification
  skipped" reason -- grading survivors cannot change a hard-gated zero;
- a failing verifier gate (all requests
  succeeded)                            -> failure_mode="correctness";
- hand-built Verdicts without the field -> scorers fall back to "correctness".

Also pins the driver-side relabeling: the circuit-breaker logs measurements
(fault counts, t= stamps on the trace's clock, the verbatim fault string),
never a "server appears dead" diagnosis -- an alive-but-overloaded
server trips the breaker exactly like a dead one.
"""
from __future__ import annotations

import asyncio
import logging

from hls.core.dataitem import DatasetItem
from hls.core.faults import ArtifactTransportError
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.core.verdict import Verdict
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit, SloLatencyScorer
from hls.spec import drivers
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import Dataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import _DriveState
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify.base import ArtifactRollout, Verifier
from hls.verify.inprocess import EngineHost
from hls.verify.run import verify_trace
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC

_TIMEOUT_ERR = "transport: request exceeded 600.0s"


class _StaticVerifier(Verifier):
    """Engine-less verifier with a fixed gate outcome (per-event mirrors it)."""

    passes: bool = True

    async def verify(self, engine) -> Verdict:
        events = engine[ArtifactRollout]
        return Verdict(passed=self.passes,
                       per_event={ev.request_id: self.passes for ev in events})


def _wl() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(), slo=NoSLO())


def _contract() -> RequestResponseContract:
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}))


def _good_ev(req: str, tenant_id: str) -> TraceEvent:
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")],
                   metadata={"tenant_id": tenant_id})
    out = Envelope(request_id=req, contents=[TextPayload(text="A")])
    return TraceEvent(request_id=req, input=inp,
                      response=ArtifactResponse(output=out), ground_truth="A",
                      t_send=0.0, t_recv=1.0)


def _bad_ev(req: str, tenant_id: str, error: str = _TIMEOUT_ERR) -> TraceEvent:
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")],
                   metadata={"tenant_id": tenant_id})
    return TraceEvent(request_id=req, input=inp, response=None, error=error,
                      ground_truth="A", t_send=0.0, t_recv=None)


def _task(verifier: Verifier) -> TaskSpec:
    return TaskSpec(
        id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=_contract(),
        tenants=[Tenant(id="t0", workload=_wl(), verifiers=[verifier])],
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def _run_verify(task: TaskSpec, trace: Trace) -> Verdict:
    """``verify_trace``, the verification the verify role runs, on an engine host: these
    verifiers load no engine."""

    async def _go():
        async with EngineHost() as host:
            verdict, _results = await verify_trace(task, trace, eval_dir=None, host=host, answers=None)
            return verdict

    return asyncio.run(_go())


def _hw() -> Hardware:
    return Hardware(accelerator=AcceleratorSpec(kind="B200", count=1))


# ---------------------------------------------------------------------------
# verify_trace: failure_mode + the error census
# ---------------------------------------------------------------------------

def test_any_failed_request_skips_verification_and_labels_failed_requests():
    task = _task(_StaticVerifier(passes=True))
    trace = Trace(events=[_good_ev("r0", "t0"), _bad_ev("r1", "t0"),
                          _bad_ev("r2", "t0")])
    verdict = _run_verify(task, trace)
    assert verdict.passed is False
    assert verdict.failure_mode == "failed_requests"
    census = verdict.diagnostics["failed_requests"]
    # The verbatim error string is the record of the concrete failure mode.
    assert census == {"count": 2, "total": 3, "errors": {_TIMEOUT_ERR: 2}}
    # The fast path never grades the survivors: no verifier diagnostics, all
    # events unpassed, and the reason says so.
    assert verdict.diagnostics["reason"] == "verification skipped: 2 of 3 requests failed"
    assert "0:_StaticVerifier" not in verdict.diagnostics
    assert verdict.per_event == {"t0:r0": False, "t0:r1": False, "t0:r2": False}


def test_verifier_gate_failure_labels_correctness():
    task = _task(_StaticVerifier(passes=False))
    trace = Trace(events=[_good_ev("r0", "t0")])
    verdict = _run_verify(task, trace)
    assert verdict.passed is False
    assert verdict.failure_mode == "correctness"
    assert "failed_requests" not in verdict.diagnostics


def test_fast_path_beats_a_failing_verifier_it_never_runs():
    # A failing verifier cannot relabel a faulted run: it is never invoked.
    task = _task(_StaticVerifier(passes=False))
    trace = Trace(events=[_good_ev("r0", "t0"), _bad_ev("r1", "t0")])
    verdict = _run_verify(task, trace)
    assert verdict.failure_mode == "failed_requests"
    assert verdict.diagnostics["failed_requests"]["count"] == 1
    assert "0:_StaticVerifier" not in verdict.diagnostics


def test_all_errored_labels_failed_requests():
    task = _task(_StaticVerifier(passes=True))
    trace = Trace(events=[_bad_ev("r0", "t0"), _bad_ev("r1", "t0")])
    verdict = _run_verify(task, trace)
    assert verdict.passed is False
    assert verdict.failure_mode == "failed_requests"
    assert verdict.diagnostics["reason"] == "verification skipped: 2 of 2 requests failed"
    assert verdict.diagnostics["failed_requests"]["errors"] == {_TIMEOUT_ERR: 2}


def test_passing_run_has_no_failure_mode():
    task = _task(_StaticVerifier(passes=True))
    verdict = _run_verify(task, Trace(events=[_good_ev("r0", "t0")]))
    assert verdict.passed is True
    assert verdict.failure_mode is None


# ---------------------------------------------------------------------------
# Scorers: failure_mode -> zero_reasons (with its fallback)
# ---------------------------------------------------------------------------

def test_scorers_surface_failed_requests_as_a_zero_reason():
    trace = Trace(events=[_bad_ev("r0", "t0")])
    verdict = Verdict(passed=False, failure_mode="failed_requests",
                      per_event={"t0:r0.0": False},
                      diagnostics={"failed_requests": {
                          "count": 1, "total": 1, "errors": {_TIMEOUT_ERR: 1}}})
    tenant = Tenant(id="t0", workload=_wl(), verifiers=[])
    for scorer in (GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
                   SloLatencyScorer()):
        score = asyncio.run(scorer.score([tenant], trace, verdict, _hw()))
        assert score.value == 0.0 and score.gate_passed is False
        assert "failed_requests" in score.zero_reasons
        # The census rides into the Score so score.json names the mode too.
        assert score.breakdown["verdict"]["failed_requests"]["errors"] == {
            _TIMEOUT_ERR: 1}


def test_scorers_default_to_correctness_without_failure_mode():
    trace = Trace(events=[_good_ev("r0", "t0")])
    verdict = Verdict(passed=False, per_event={"t0:r0.0": False})
    tenant = Tenant(id="t0", workload=_wl(), verifiers=[])
    for scorer in (GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)),
                   SloLatencyScorer()):
        score = asyncio.run(scorer.score([tenant], trace, verdict, _hw()))
        assert "correctness" in score.zero_reasons


# ---------------------------------------------------------------------------
# _DriveState: measured fault/trip logging, no "server appears dead" diagnosis
# ---------------------------------------------------------------------------

def test_breaker_logs_measurements_not_a_diagnosis(caplog):
    state = _DriveState(fault_threshold=3, total_requests=10)
    with caplog.at_level(logging.WARNING, logger="hls.spec.drivers"):
        state.record_result(faulted=False)                  # a success, then...
        for _ in range(3):                                  # ...the fatal run
            state.record_result(faulted=True, error=_TIMEOUT_ERR)
    assert state.aborted is True
    text = caplog.text
    # First fault is logged with its t= stamp and verbatim error string.
    assert "request fault #1 at t=" in text
    assert _TIMEOUT_ERR in text
    # The trip states measurements: the consecutive run, the fault census, the
    # age of the last success, the last fault string -- and no cause guess.
    assert "3 consecutive request faults" in text
    assert "3/4 completed submits faulted" in text
    assert "last success" in text
    assert "server appears dead" not in text


def test_breaker_trip_without_any_success_says_so(caplog):
    state = _DriveState(fault_threshold=2, total_requests=5)
    with caplog.at_level(logging.WARNING, logger="hls.spec.drivers"):
        state.record_result(faulted=True, error=_TIMEOUT_ERR)
        state.record_result(faulted=True, error=_TIMEOUT_ERR)
    assert state.aborted is True
    assert "no successful response ever received" in caplog.text


# ---------------------------------------------------------------------------
# _DriveState: the stall watchdog, for the server that answers nothing at all
# ---------------------------------------------------------------------------

def _drive_state_on(clock, monkeypatch, **kw):
    """A _DriveState reading a clock the test advances by hand."""
    monkeypatch.setattr(drivers, "harness_now", lambda: clock[0])
    return _DriveState(**kw)


def test_a_server_that_answers_nothing_trips_the_watchdog(caplog, monkeypatch):
    """The gap the fault counter cannot see.

    `record_result` only fires on a COMPLETED submit, so a server that accepts the
    connection and never answers keeps the consecutive run at zero forever. A
    drive could then keep firing at a server that had stopped answering, with the breaker
    reading zero faults throughout.
    """
    clock = [1000.0]
    state = _drive_state_on(clock, monkeypatch, fault_threshold=50,
                            total_requests=5000, stall_abort_sec=900.0)

    for _ in range(50):                      # 50 sent, none ever completes
        assert state.gate_send() == "send"
    assert state.aborted is False, "not yet -- no time has passed"

    clock[0] += 900.0
    with caplog.at_level(logging.WARNING, logger="hls.spec.drivers"):
        assert state.gate_send() == "abort"

    assert state.aborted is True and state.stop.is_set()
    assert "nothing has completed for 900s" in caplog.text
    assert "50 request(s) outstanding" in caplog.text


def test_a_slow_server_that_keeps_answering_is_never_aborted(monkeypatch):
    """The false positive that would corrupt a score, so it is pinned explicitly.

    A server far slower than its pin deserves its zero on failed requests, not an abort.
    One that answers about one request every 6 s is 150x inside this window. A drive that
    aborted it would discard a real result.
    """
    clock = [1000.0]
    state = _drive_state_on(clock, monkeypatch, fault_threshold=50,
                            total_requests=5000, stall_abort_sec=900.0)

    for _ in range(400):
        assert state.gate_send() == "send", "a slow but answering server must not abort"
        clock[0] += 6.0                      # one completion every 6 s
        state.record_result(faulted=False)

    assert state.aborted is False and not state.stop.is_set()


def test_a_deep_backlog_alone_does_not_abort(monkeypatch):
    """Outstanding count alone is not evidence of death.

    A server slower than its pin accumulates arbitrarily many in flight and is still
    answering; that is a thing to SCORE, not to abort. Only silence decides.
    """
    clock = [1000.0]
    state = _drive_state_on(clock, monkeypatch, fault_threshold=50,
                            total_requests=5000, stall_abort_sec=900.0)

    for i in range(1000):                    # 1,000 outstanding, far past the threshold
        assert state.gate_send() == "send"
        clock[0] += 1.0
        if i % 100 == 0:                     # but it answers now and then
            state.record_result(faulted=False)

    assert state.aborted is False


def test_faults_that_come_back_clear_the_watchdog(monkeypatch):
    """A fault is a RESPONSE. The watchdog asks whether the server answers, not whether
    it answers correctly -- a run of returned faults is the other breaker's business."""
    clock = [1000.0]
    state = _drive_state_on(clock, monkeypatch, fault_threshold=50,
                            total_requests=5000, stall_abort_sec=900.0)

    for _ in range(40):
        state.gate_send()
        clock[0] += 60.0                     # well past the window between answers
        state.record_result(faulted=True, error=_TIMEOUT_ERR)

    assert state.aborted is False, "returned faults are answers; they clear the stall"


def test_the_watchdog_can_be_disabled(monkeypatch):
    """`stall_abort_sec <= 0` turns the watchdog off."""
    clock = [1000.0]
    state = _drive_state_on(clock, monkeypatch, fault_threshold=50,
                            total_requests=5000, stall_abort_sec=0.0)

    for _ in range(50):
        state.gate_send()
    clock[0] += 100_000.0
    assert state.gate_send() == "send" and state.aborted is False


def test_the_watchdog_fires_with_no_sends_happening(monkeypatch):
    """The reason the watchdog is a task and not just a check inside `gate_send`.

    `gate_send` runs once per request, so it only ticks while arrivals are still going
    out. A corpus can offer everything sooner than the silence being watched for, and a
    server that goes quiet after the last arrival leaves no send to notice it, so a
    send-time check alone would never fire.
    """
    clock = [1000.0]
    state = _drive_state_on(clock, monkeypatch, fault_threshold=50,
                            total_requests=5000, stall_abort_sec=900.0)

    for _ in range(60):                  # the whole corpus goes out, nothing comes back
        state.gate_send()
    assert state.aborted is False, "not yet -- no time has passed"

    clock[0] += 900.0                    # silence, with no further sends at all
    assert state.trip_if_stalled() is True
    assert state.aborted is True and state.stop.is_set()


def test_trip_if_stalled_is_idempotent(monkeypatch):
    """The watchdog polls, so it asks repeatedly; only the first ask may trip."""
    clock = [1000.0]
    state = _drive_state_on(clock, monkeypatch, fault_threshold=50,
                            total_requests=5000, stall_abort_sec=900.0)
    for _ in range(60):
        state.gate_send()
    clock[0] += 900.0

    assert state.trip_if_stalled() is True
    assert state.trip_if_stalled() is False, "a tripped breaker must not re-trip"


# ---------------------------------------------------------------------------
# The breaker's threshold is the one the driver was built with
# ---------------------------------------------------------------------------

class _RefusingClient:
    """An artifact client whose every request faults at once, as a dead server's does."""

    async def submit(self, env, *, attempt=None):
        raise ArtifactTransportError("connection refused")

    async def aclose(self) -> None:
        return None


class _TwoQuestions(Dataset):
    """Two single-turn questions, r0 and r1."""

    def items_at(self, mount_dir, split):
        for i in range(2):
            env = Envelope(request_id=f"r{i}", contents=[TextPayload(text="Q?")])
            yield DatasetItem(request_id=f"r{i}", turns=(env,), ground_truth_turns=("A",))


def test_the_breakers_threshold_is_not_read_from_the_environment(monkeypatch):
    """A drive role starts with a clean environment and nothing sets the threshold but the
    driver's constructor (50 by default), so a drive never reads it from the environment: with
    HLS_FAULT_ABORT_THRESHOLD=1 set, two faults in a row trip nothing, and both requests are
    sent and recorded as the server's faults."""
    monkeypatch.setenv("HLS_FAULT_ABORT_THRESHOLD", "1")
    monkeypatch.setattr("hls.artifact.artifact_client.make_artifact_client",
                        lambda contract, url, request_timeout_sec=None, stall_timeout_sec=None: _RefusingClient())
    tenant = Tenant(id="t0", workload=_wl(), verifiers=[],
                    data_source=DataSource(dataset=_TwoQuestions(),
                                           split=SplitSpec(dev=Count(value=1), test=Count(value=2))))
    driver = drivers.Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    events: list = []
    asyncio.run(driver.run("http://127.0.0.1:1", split="test", tenants=[tenant], contract=_contract(), seed=0,
                           on_event=events.append))
    assert [ev.error for ev in events] == ["transport: connection refused"] * 2
