"""The trajectory-shape diagnostic: what it counts, and that it never gates."""
from __future__ import annotations

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent

from tasks.agentic_coding.metric import (
    ReportOnlyTrajectoryShapeVerifier, TrajectoryShapeMetric,
)
from tasks.agentic_coding.mini_swe_loop import (
    STOP_EMPTY_REPLY, STOP_FORMAT_REPAIR_EXHAUSTED, STOP_STEP_LIMIT, STOP_SUBMITTED,
)


def event(session: str, turn: int, **extras) -> TraceEvent:
    env = Envelope(request_id=f"{session}.{turn}",
                   contents=[TextPayload(text="x")], metadata={"tenant_id": "t0"})
    return TraceEvent(request_id=f"{session}.{turn}", session_id=session,
                      turn_index=turn, input=env, t_send=0.0, t_recv=1.0,
                      extras=extras)


def trajectory(session: str, *, turns: int, commands: int, stop: str):
    """A session's events, with the verdict on the last one as the loop stamps it."""
    evs = [event(session, i, commands_run=min(i + 1, commands)) for i in range(turns - 1)]
    evs.append(event(session, turns - 1, commands_run=commands, stop_reason=stop,
                     session_turns=turns))
    return evs


class Engine:
    def __init__(self, events):
        self.events = events

    def __getitem__(self, key):
        return self.events


def test_it_counts_how_each_trajectory_ended():
    events = (trajectory("a", turns=3, commands=2, stop=STOP_SUBMITTED)
              + trajectory("b", turns=5, commands=4, stop=STOP_STEP_LIMIT)
              + trajectory("c", turns=2, commands=0,
                           stop=STOP_FORMAT_REPAIR_EXHAUSTED))
    d = TrajectoryShapeMetric().score(events).diagnostics
    assert d["sessions"] == 3
    assert d["submitted"] == 1
    assert d["hit_step_limit"] == 1
    assert d["format_repair_exhausted"] == 1


def test_a_model_that_would_not_answer_is_counted_apart_from_a_silent_server():
    # The distinction a reader needs is whose failure it was. A model that never
    # produced a parseable command is the model; a completion carrying no tokens at
    # all is the server, and reporting the first as the second would call a starved
    # artifact a broken one.
    events = (trajectory("a", turns=2, commands=1, stop=STOP_FORMAT_REPAIR_EXHAUSTED)
              + trajectory("b", turns=2, commands=1, stop=STOP_FORMAT_REPAIR_EXHAUSTED)
              + trajectory("c", turns=1, commands=0, stop=STOP_EMPTY_REPLY))
    d = TrajectoryShapeMetric().score(events).diagnostics
    assert d["format_repair_exhausted"] == 2
    assert d["empty_reply"] == 1


def test_the_value_is_the_share_of_trajectories_that_ran_a_command():
    # The weakest property the traffic must have: a run where nothing executed is not
    # a coding-agent load whatever its tokens looked like.
    events = (trajectory("a", turns=3, commands=2, stop=STOP_SUBMITTED)
              + trajectory("b", turns=2, commands=0,
                           stop=STOP_FORMAT_REPAIR_EXHAUSTED))
    result = TrajectoryShapeMetric().score(events)
    assert result.value == 0.5
    assert result.diagnostics["ran_a_command"] == 1


def test_means_are_per_trajectory_not_per_turn():
    events = (trajectory("a", turns=4, commands=3, stop=STOP_SUBMITTED)
              + trajectory("b", turns=2, commands=1, stop=STOP_SUBMITTED))
    d = TrajectoryShapeMetric().score(events).diagnostics
    assert d["mean_commands"] == 2.0
    assert d["mean_turns"] == 3.0


def test_an_empty_trace_reduces_to_zero_rather_than_dividing_by_zero():
    result = TrajectoryShapeMetric().score([])
    assert result.value == 0.0 and result.per_event == {}


def test_a_session_with_no_stamped_verdict_is_named_unknown():
    # Defensive: a framework that emitted turns without the loop's final stamp.
    events = [event("a", 0, commands_run=1)]
    d = TrajectoryShapeMetric().score(events).diagnostics
    assert d["stop_reasons"] == {"unknown": 1}


def test_events_out_of_order_still_read_the_last_turn():
    evs = trajectory("a", turns=3, commands=2, stop=STOP_SUBMITTED)
    d = TrajectoryShapeMetric().score(list(reversed(evs))).diagnostics
    assert d["submitted"] == 1 and d["mean_turns"] == 3.0


async def test_the_verifier_reports_and_never_gates():
    events = trajectory("a", turns=2, commands=0,
                        stop=STOP_FORMAT_REPAIR_EXHAUSTED)
    verdict = await ReportOnlyTrajectoryShapeVerifier().verify(Engine(events))
    # Every trajectory failed to do anything, and the run still passes: this is a
    # diagnostic, and the gate is the report-KL verifier.
    assert verdict.passed is True
    assert all(verdict.per_event.values())
    assert verdict.diagnostics["ran_a_command_rate"] == 0.0
    assert verdict.diagnostics["trajectory_shape"]["format_repair_exhausted"] == 1


def test_turns_that_tried_to_call_and_produced_nothing_are_counted_apart():
    # A parser bug and a model that would not act look identical in a stop reason, and
    # only one of them is our fault. This count is what makes the difference visible
    # without anyone re-reading raw completions.
    events = [event("a", 0, commands_run=0, unparsed_tool_call=True),
              event("a", 1, commands_run=0, unparsed_tool_call=False),
              event("a", 2, commands_run=0, unparsed_tool_call=True,
                    stop_reason=STOP_FORMAT_REPAIR_EXHAUSTED, session_turns=3)]
    d = TrajectoryShapeMetric().score(events).diagnostics
    assert d["unparsed_tool_calls"] == 2
    assert d["format_repair_exhausted"] == 1


def test_a_clean_run_reports_no_parser_suspicion():
    events = trajectory("a", turns=2, commands=1, stop=STOP_SUBMITTED)
    assert TrajectoryShapeMetric().score(events).diagnostics["unparsed_tool_calls"] == 0
