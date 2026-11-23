"""Report-only diagnostics for the coding-agent trajectories.

Nothing here gates. Correctness is the report-KL verifier's job, and whether the task
was solved is not measured at all: Terminal-Bench scores a task by running its
verifier in a second container built from the task's own tests, which this harness
does not build or run. What these metrics answer is whether the
traffic was the traffic the task intended -- did trajectories take turns, did commands
execute, did the model produce parseable actions -- so a run that served plausible
tokens for the wrong reason is visible in the record.

``ReportOnlyMetricVerifier`` and ``TrajectoryServingMetrics`` are reused from
``tasks.deepresearch.metric``; they are task-neutral reductions over the trace, and a
second copy would be a second thing to keep in step.
"""
from __future__ import annotations

from collections import Counter

from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict, Verdict
from hls.verify.base import ArtifactRollout, EngineAccessor, RolloutMetric, Verifier

from tasks.deepresearch.metric import (  # noqa: F401  re-exported so task.py has one import site
    ReportOnlyMetricVerifier, TrajectoryServingMetrics,
)

from tasks.agentic_coding.mini_swe_loop import (
    NO_COMMAND_RUNAWAY, STOP_EMPTY_REPLY, STOP_FORMAT_REPAIR_EXHAUSTED,
    STOP_NO_COMMAND_REPEATED, STOP_STEP_LIMIT, STOP_SUBMITTED, STOP_WALL_CLOCK,
)


def _sessions(events: list[TraceEvent]) -> dict[str, list[TraceEvent]]:
    """Trace events grouped into trajectories, in arrival order within each."""
    grouped: dict[str, list[TraceEvent]] = {}
    for ev in events:
        grouped.setdefault(ev.session_id or ev.request_id, []).append(ev)
    for evs in grouped.values():
        evs.sort(key=lambda e: e.turn_index)
    return grouped


class TrajectoryShapeMetric(RolloutMetric):
    """How the trajectories ran: turns taken, commands executed, and how each ended.

    The reduced value is the fraction of trajectories that ran at least one command,
    which is the weakest property the traffic has to have. A run where most
    trajectories never executed anything is not measuring a coding agent's serving
    load, whatever its tokens looked like -- most often because the model could not
    produce the action format, which ``stop_reasons`` names directly.
    """

    def score(self, events: list[TraceEvent]) -> MetricResult:
        sessions = _sessions(events)
        if not sessions:
            return MetricResult(value=0.0, per_event={})
        # Turns whose reply wrote a tool-call delimiter and still yielded no action.
        # Counted across every turn rather than per trajectory, and reported on its own
        # rather than folded into the format-error total, because it does not mean the
        # same thing: a reply with no delimiter is a model that did not act, and a
        # reply that tried and produced nothing is the PARSER under suspicion. A run
        # with a non-zero count here needs the parser looked at, and saying so in the
        # diagnostics is what stops that being found by re-reading raw completions.
        unparsed = sum(1 for ev in events
                       if (ev.extras or {}).get("unparsed_tool_call"))
        stops: Counter[str] = Counter()
        #: Turns that spent a repair budget by running the output cap out, summed over
        #: the trajectories the budget ended. Read off each session's last event,
        #: which carries the breakdown of the run that ended it.
        runaways = 0
        commands: list[int] = []
        turns: list[int] = []
        per_event: dict[str, PerEventVerdict] = {}
        ran_any = 0
        for sid, evs in sessions.items():
            last = evs[-1]
            stop = str(last.extras.get("stop_reason") or "unknown")
            stops[stop] += 1
            runaways += int((last.extras.get("no_command_causes")
                             or {}).get(NO_COMMAND_RUNAWAY, 0))
            n_commands = int(last.extras.get("commands_run") or 0)
            commands.append(n_commands)
            turns.append(int(last.extras.get("session_turns") or len(evs)))
            ran = n_commands > 0
            ran_any += int(ran)
            # Report-only: the per-event flag is never read as a gate, and the
            # rationale is what a reader opens the record for.
            per_event[last.event_key] = PerEventVerdict(
                passed=True, score=float(ran),
                rationale=f"session {sid}: {stop}, {n_commands} command(s)")
        n = len(sessions)
        return MetricResult(value=ran_any / n, per_event=per_event, diagnostics={
            "sessions": n,
            "ran_a_command": ran_any,
            "submitted": stops.get(STOP_SUBMITTED, 0),
            "hit_step_limit": stops.get(STOP_STEP_LIMIT, 0),
            # Sessions still working when the drive's window closed. Not a failure of
            # anything; it says how much of this drive measured whole trajectories and
            # how much measured a slice of one.
            "stopped_on_the_wall_clock": stops.get(STOP_WALL_CLOCK, 0),
            # The repair budget ran out, and these two say on what. The first is the
            # parser: at least one reply in the run that ended the trajectory wrote a
            # delimiter this parser could not read, and anything but zero here means
            # go and look at it. The second is the model: no reply in the run carried
            # a call at all, so the parser was never asked.
            #
            # Two numbers, because one total reads as the parser's fault when it is
            # the model's.
            "format_repair_exhausted": stops.get(STOP_FORMAT_REPAIR_EXHAUSTED, 0),
            "no_command_repeated": stops.get(STOP_NO_COMMAND_REPEATED, 0),
            # Of the turns that spent a budget, the ones that ran the output cap out
            # without writing a call. A runaway is a decoding problem and the lever
            # for it is the sampler, which is a different fix from either of the two
            # above.
            "budget_spent_on_runaways": runaways,
            # Kept apart from the line above because it is the one stop that means the
            # server, not the model: a completion that carried no tokens at all.
            "empty_reply": stops.get(STOP_EMPTY_REPLY, 0),
            # Turns that tried to call and produced nothing. Not a model failure: see
            # the comment where it is counted. Anything but zero, and read it before
            # reading the rest of this run.
            "unparsed_tool_calls": unparsed,
            "stop_reasons": dict(stops),
            "mean_commands": sum(commands) / n,
            "mean_turns": sum(turns) / n,
        })


class ReportOnlyTrajectoryShapeVerifier(Verifier):
    """``TrajectoryShapeMetric`` surfaced into the run record, never gating.

    A separate class rather than ``ReportOnlyMetricVerifier(metric=...)`` only
    because the shape metric's ``details`` are the point of it, and they ride into
    the run-level diagnostics from here."""

    requires = frozenset()

    async def verify(self, engine: EngineAccessor) -> Verdict:
        result = TrajectoryShapeMetric().score(engine[ArtifactRollout])
        return Verdict(
            passed=True,
            per_event={k: True for k in result.per_event},
            diagnostics={"trajectory_shape": dict(result.diagnostics),
                         "ran_a_command_rate": result.value},
            event_rationale={k: v.rationale for k, v in result.per_event.items()
                             if v.rationale})


__all__ = ["ReportOnlyMetricVerifier", "ReportOnlyTrajectoryShapeVerifier",
           "TrajectoryServingMetrics", "TrajectoryShapeMetric"]
