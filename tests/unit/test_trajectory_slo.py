"""Unit tests for TrajectoryMakespanSLO + TrajectorySessionUnit.

The SLO normalizes a trajectory's makespan against a target and defines the unit a
``SloLatencyScorer`` means over; the unit is what a ``GoodputScorer`` counts.
"""

from __future__ import annotations
import asyncio
import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from hls.core.model import AcceleratorSpec, Hardware
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import (
    ArtifactResponse, FailureReason, RequestAttempt, RequestFailure, Trace, TraceEvent,
)
from hls.core.verdict import Verdict
from hls.scoring import (
    ConstantSoL, GoodputScorer, SloLatencyScorer, TrajectorySessionUnit,
)
from hls.spec.workload import (
    AllAtOnce, NoDelay, StreamingLatencySLO, TrajectoryMakespanSLO, Uncapped, Workload,
)


def _ev(sid: str, turn: int, t_send: float, t_recv: float) -> TraceEvent:
    env = Envelope(
        request_id=f"{sid}.{turn}",
        contents=[TextPayload(text="x")],
        metadata={"tenant_id": "t0"},
    )
    resp = ArtifactResponse(output=env)
    return TraceEvent(request_id=f"{sid}.{turn}", session_id=sid, turn_index=turn,
                      input=env, response=resp, t_send=t_send, t_recv=t_recv)


def _unanswered(sid: str, turn: int, *, t_send: float, t_abandoned: float,
                due: float | None = None) -> TraceEvent:
    """A turn the harness stopped waiting on: no response, and a failure record whose
    ``t_abandoned`` is the moment it stopped watching."""
    env = Envelope(request_id=f"{sid}.{turn}", contents=[TextPayload(text="x")],
                   metadata={"tenant_id": "t0"})
    return TraceEvent(
        request_id=f"{sid}.{turn}", session_id=sid, turn_index=turn, input=env,
        response=None, error="deadline: boom", t_send=t_send, t_planned_send=due,
        t_recv=None,
        failure=RequestFailure(reason=FailureReason.DEADLINE, t_abandoned=t_abandoned,
                               observed=RequestAttempt()))


def _turns(sid: str, n: int, makespan: float) -> list[TraceEvent]:
    """``n`` back-to-back answered turns spanning exactly ``makespan`` seconds."""
    step = makespan / n
    return [_ev(sid, k, t_send=k * step, t_recv=(k + 1) * step) for k in range(n)]


def _workload(target_sec: float = 10.0) -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(),
                    slo=TrajectoryMakespanSLO(target_sec=target_sec))


def _tenants(target_sec: float = 10.0):
    return [SimpleNamespace(id="t0", workload=_workload(target_sec))]


def _hw() -> Hardware:
    return Hardware(accelerator=AcceleratorSpec(kind="B200", count=1))


def _verdict(trace: Trace) -> Verdict:
    return Verdict(passed=True,
                   per_event={ev.event_key: True for ev in trace.events})


# ---- the grain ------------------------------------------------------------

def test_units_group_a_trace_at_the_slos_own_grain():
    """A session-grain SLO's unit is the trajectory; a request-grain SLO's is one
    event. The scorer reads this instead of branching on the grain itself."""
    trace = Trace(events=_turns("s0", 2, 4.0) + _turns("s1", 1, 3.0))

    trajectory = TrajectoryMakespanSLO(target_sec=10.0)
    assert trajectory.grain == "session"
    assert {sid: len(evs) for sid, evs in trajectory.units(trace).items()} == {
        "s0": 2, "s1": 1}

    request = StreamingLatencySLO(ttft_ms=100.0, tpot_ms=50.0)
    assert request.grain == "request"
    assert {rid: len(evs) for rid, evs in request.units(trace).items()} == {
        "s0.0": 1, "s0.1": 1, "s1.0": 1}


# ---- TrajectoryMakespanSLO ------------------------------------------------

def test_apply_returns_one_ratio_per_trajectory_keyed_by_session_id():
    trace = Trace(events=[
        _ev("s0", 0, t_send=0.0, t_recv=3.0),
        _ev("s0", 1, t_send=3.0, t_recv=7.0),
        _ev("s1", 0, t_send=5.0, t_recv=8.0),
    ])
    res = TrajectoryMakespanSLO(target_sec=10.0).apply(trace, _workload())

    # s0 makespan = 7-0 = 7; s1 = 8-5 = 3. Both normalize against the 10 s target.
    assert res.per_event_ratio == {"s0": 0.7, "s1": 0.3}
    assert res.diagnostics["p99_makespan_sec"] == 7.0
    assert res.diagnostics["n_sessions"] == 2


def test_the_gate_is_computability_not_the_makespan_target():
    """Every request must have answered, which is what makes a makespan measurable. A
    trajectory over target is scored on how far over, not zeroed."""
    over = Trace(events=[
        _ev("s_ok", 0, 0.0, 5.0),
        _ev("s_slow", 0, 0.0, 12.0),
        _ev("s_slow", 1, 5.0, 12.0),
    ])
    res = TrajectoryMakespanSLO(target_sec=10.0).apply(over, _workload())

    assert res.aggregate_passed                       # p99 is 12 s, and that is fine
    assert res.per_event_ratio == {"s_ok": 0.5, "s_slow": 1.2}
    assert res.diagnostics["p99_makespan_sec"] == 12.0
    assert res.diagnostics["met_frac"] == 0.5
    # The binary reading survives as a goodput scorer's credit, per request.
    assert res.per_event_credit == {"s_ok.0": 1.0, "s_slow.0": 0.0, "s_slow.1": 0.0}


def test_a_trajectory_with_an_unanswered_request_has_no_ratio_and_fails_the_gate():
    trace = Trace(events=[
        _ev("s_ok", 0, 0.0, 5.0),
        _ev("s_dead", 0, 0.0, 2.0),
        _unanswered("s_dead", 1, t_send=2.0, t_abandoned=9.0),
    ])
    res = TrajectoryMakespanSLO(target_sec=10.0).apply(trace, _workload())

    assert res.aggregate_passed is False
    assert res.diagnostics["n_missing"] == 1
    assert res.per_event_ratio == {"s_ok": 0.5}
    assert res.per_event_credit["s_dead.0"] == 0.0


def test_empty_trace_is_a_trivial_pass():
    res = TrajectoryMakespanSLO(target_sec=10.0).apply(Trace(events=[]), _workload())
    assert res.aggregate_passed
    assert res.diagnostics["n_sessions"] == 0
    assert res.per_event_ratio == {}


def test_censored_ratio_spans_the_whole_trajectory():
    """An unfinished trajectory is charged from when its first turn was due to the
    moment the harness stopped watching its last: due at 1.0, abandoned at 9.0, which
    is 8 s against a 4 s target."""
    first = _ev("s", 0, t_send=1.2, t_recv=3.0)
    # t_planned_send is what says when the trajectory was due; the driver stamps
    # t_send late for a turn its circuit breaker held back.
    events = [first.model_copy(update={"t_planned_send": 1.0}),
              _unanswered("s", 1, t_send=3.0, t_abandoned=9.0)]
    slo = TrajectoryMakespanSLO(target_sec=4.0)

    assert slo.censored_ratio(events, t_end=100.0) == pytest.approx(2.0)


def test_censored_ratio_floors_at_one():
    """A trajectory that did not finish did not meet its target, so it earns at most
    what one landing exactly on the target earns, which is nothing."""
    events = [_unanswered("s", 0, t_send=0.0, t_abandoned=0.5, due=0.0)]
    slo = TrajectoryMakespanSLO(target_sec=10.0)

    assert slo.censored_ratio(events, t_end=90.0) == 1.0


def test_censored_ratio_charges_to_the_run_end_when_nothing_else_is_known():
    env = Envelope(request_id="s.0", contents=[TextPayload(text="x")],
                   metadata={"tenant_id": "t0"})
    pending = TraceEvent(request_id="s.0", session_id="s", turn_index=0, input=env,
                         response=None, error="boom", t_send=1.0, t_recv=None)
    slo = TrajectoryMakespanSLO(target_sec=4.0)

    assert slo.censored_ratio([pending], t_end=9.0) == pytest.approx(2.0)


def test_a_non_positive_target_is_refused_at_construction():
    """The target is the divisor every ratio is normalized by, so a zero or negative
    one is a config error, not a run-time one: caught when the task is built rather
    than as a ZeroDivisionError inside apply, after the GPUs have been spent."""
    for bad in (0.0, -1.0):
        with pytest.raises(ValidationError):
            TrajectoryMakespanSLO(target_sec=bad)


def test_describe_states_the_target_per_trajectory():
    assert TrajectoryMakespanSLO(target_sec=1200.0).describe() == (
        "makespan < 1200 s (per trajectory)")


# ---- TrajectorySessionUnit ------------------------------------------------

def test_trajectory_unit_counts_one_per_session():
    unit = TrajectorySessionUnit()
    assert unit.count_for_event(_ev("s0", 0, 0.0, 1.0)) == 1.0
    assert unit.count_for_event(_ev("s0", 1, 1.0, 2.0)) == 0.0
    assert unit.count_for_event(_ev("s0", 2, 2.0, 3.0)) == 0.0


def test_trajectory_unit_name():
    assert TrajectorySessionUnit().name == "trajectory"


# ---- composes with the goodput scorer -------------------------------------

def test_goodput_credits_all_trajectories_when_makespan_holds():
    trace = Trace(events=[
        _ev("a", 0, 0.0, 4.0),
        _ev("b", 0, 0.0, 6.0),
    ])
    scorer = GoodputScorer(unit=TrajectorySessionUnit(), sol=ConstantSoL(rps=1.0))

    score = asyncio.run(scorer.score(_tenants(), trace, _verdict(trace), _hw()))

    assert score.gate_passed
    assert score.unit == "trajectory"
    # 2 trajectories counted over a duration of 6 sec -> 0.333 traj/sec.
    assert abs(score.achieved - (2.0 / 6.0)) < 1e-9


def test_goodput_counts_only_trajectories_inside_the_makespan_target():
    """The target zeroes a busting trajectory's credit rather than the whole run: the
    SLO's own gate is computability, and every request here answered."""
    trace = Trace(events=[
        _ev("good", 0, 0.0, 5.0),
        _ev("bad", 0, 0.0, 15.0),
    ])
    scorer = GoodputScorer(unit=TrajectorySessionUnit(), sol=ConstantSoL(rps=1.0))

    score = asyncio.run(scorer.score(_tenants(), trace, _verdict(trace), _hw()))

    assert score.gate_passed is True
    assert score.achieved == pytest.approx(1.0 / 15.0)   # only "good" counted
    assert score.breakdown["slo_attainment"] == 0.5


# ---- composes with the speedup scorer -------------------------------------

def _speedup(events, *, target_sec: float = 10.0, baseline=None):
    trace = Trace(events=events)
    scorer = SloLatencyScorer(baseline_log_credit=baseline)
    return asyncio.run(scorer.score(_tenants(target_sec), trace, _verdict(trace),
                                    _hw()))


def test_speedup_is_the_geometric_mean_over_trajectories():
    """Two trajectories at r = 0.25 and r = 0.5: the geometric mean of 4 and 2 is
    sqrt(8), which is what the run scores against the target itself."""
    score = _speedup(_turns("fast", 3, 2.5) + _turns("slower", 2, 5.0))

    assert score.unit == "slo_speedup"
    assert score.gate_passed is True
    assert score.breakdown["n_scored"] == 2          # trajectories, not the 5 requests
    assert score.breakdown["mean_log_credit"] == pytest.approx(
        (math.log(4.0) + math.log(2.0)) / 2)
    assert score.value == pytest.approx(math.sqrt(8.0))


def test_a_deep_and_a_shallow_trajectory_carry_the_same_weight():
    """One term per trajectory, whatever its turn count: a serving stack does not
    choose how many turns a research question takes. A request-weighted mean would give
    the 30-turn trajectory six times the pull and score 3.62 instead."""
    score = _speedup(_turns("deep", 30, 2.5) + _turns("shallow", 5, 5.0))

    assert score.value == pytest.approx(math.sqrt(8.0))       # 2.83
    turn_weighted = math.exp((30 * math.log(4.0) + 5 * math.log(2.0)) / 35)
    assert turn_weighted == pytest.approx(3.6229, abs=1e-4)
    assert abs(score.value - turn_weighted) > 0.5


def test_speedup_normalizes_against_the_baseline_stacks_own_credit():
    """S = exp(L - B). A baseline stack that itself ran at r = 0.5 (B = log 2) halves
    the score of an artifact whose own geometric mean is 4."""
    events = _turns("a", 2, 2.5) + _turns("b", 2, 2.5)   # both r = 0.25
    score = _speedup(events, baseline=math.log(2.0))

    assert score.achieved == pytest.approx(4.0)
    assert score.sol == pytest.approx(2.0)
    assert score.value == pytest.approx(2.0)


def test_speedup_zeroes_a_run_whose_trajectory_never_finished():
    """An unanswered request fails the run, and its trajectory is priced for the bound
    only: charged the 9 s it was outstanding against the 10 s target, so r floors at 1
    and it contributes no credit."""
    events = _turns("ok", 2, 2.5) + [
        _ev("dead", 0, t_send=0.0, t_recv=2.0),
        _unanswered("dead", 1, t_send=2.0, t_abandoned=9.0, due=0.0),
    ]
    score = _speedup(events)

    assert score.value == 0.0
    assert score.zero_reasons == ("slo",)
    assert score.breakdown["upper_bound"] == pytest.approx(
        math.exp((math.log(4.0) + 0.0) / 2))
