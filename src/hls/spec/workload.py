"""Workload -- open-loop, two timing levels (the config plane).

Session arrival is open-loop: arrival.schedule(n, seed) gives absolute session-
start offsets, fired regardless of completions. inter_turn_delay is the one
reactive point, waited after the prior turn's response. Concurrency is not a
knob; it is the artifact's observed lever. Both the identity implementations
(AllAtOnce, NoDelay) and the concrete family subclasses (Poisson, Normal, ...)
live in this file.
"""
from __future__ import annotations

import logging
import math
import random
from abc import abstractmethod
from itertools import accumulate
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field, model_validator

from hls.core.poly import Poly, Spec
from hls.core.score import SLOResult

if TYPE_CHECKING:
    from hls.core.trace import Trace, TraceEvent

log = logging.getLogger(__name__)


# ---- ArrivalProcess ------------------------------------------------------

class ArrivalProcess(Poly):
    """When each session starts (L1).

    `schedule(n, seed)` returns absolute session-start offsets (sec from t=0),
    fired open-loop (regardless of completions). Subclass per traffic shape
    (AllAtOnce = every session starts at t0; Poisson = a rate).
    """

    @abstractmethod
    def schedule(self, n: int, seed: int) -> list[float]:
        """Absolute session-start offsets from t0 (open-loop; not gaps)."""

    def describe(self) -> str:
        """Human-readable label for use in the implementer brief."""
        return type(self).__name__.lower()

    @abstractmethod
    def mean_rate(self) -> float | None:
        """Mean sessions per second this process schedules, or None for a process
        with no rate (every session at once). The number the brief states, next
        to ``describe()``; the unit is sessions, since ``schedule`` places
        sessions."""

    @abstractmethod
    def schedule_span_sec(self, n: int, *, sigmas: float = 0.0) -> float:
        """How long after the drive starts the ``n``-th session is due, allowing
        ``sigmas`` standard deviations of headroom on the draw.

        What a bounded issuing window has to cover. A drive with a scheduled stop
        (``AgentSpec.wall_clock_sec``) opens no session that arrives after it, and a
        session never opened is one the run silently did not measure, so the arrival
        process, the session count and the window are one relationship rather than
        three numbers. This is the side of it a task can compute before it runs.

        ``sigmas`` at 0 is the mean, which leaves about half the draws over any window
        it exactly fits. A window is chosen against a margin instead; 3 puts the draw
        inside it on all but about one run in a thousand for the session counts these
        tasks use. A family whose gaps do not vary ignores it."""


class AllAtOnce(ArrivalProcess):
    """The identity arrival: every session starts at t0."""

    def schedule(self, n: int, seed: int) -> list[float]:
        return [0.0] * n

    def describe(self) -> str:
        return "all sessions dispatched at once (offline batch)"

    def mean_rate(self) -> float | None:
        return None

    def schedule_span_sec(self, n: int, *, sigmas: float = 0.0) -> float:
        """Nothing waits: every session is due at t0, however many there are."""
        return 0.0


# ---- InterTurnDelay ------------------------------------------------------

class InterTurnDelay(Poly):
    """The one reactive timing point (L1).

    `delay(turn, seed)` returns the seconds to wait before a follow-up turn,
    after the prior turn's response. Subclass per think-time shape (NoDelay =
    no wait).
    """

    @abstractmethod
    def delay(self, turn: int, seed: Any) -> float:
        """Seconds to wait before follow-up `turn`, after the prior response."""

    def describe(self) -> str:
        """Human-readable label for use in the implementer brief."""
        return type(self).__name__.lower()


class NoDelay(InterTurnDelay):
    """The identity think-time: no inter-turn wait."""

    def delay(self, turn: int, seed: Any) -> float:
        return 0.0

    def describe(self) -> str:
        return "no inter-turn delay"


# ---- Concrete arrival families (gap-defined; schedule = cumulative sum) -----
class _GapArrival(ArrivalProcess):
    """ArrivalProcess defined by i.i.d. inter-request gaps: schedule[i] =
    sum(gaps[:i]), so session 0 starts at t=0 and gaps[k] is the wait between
    sessions k and k+1."""

    def _gaps(self, n: int, seed: int) -> list[float]:
        raise NotImplementedError

    def _gap_moments(self) -> tuple[float, float]:
        """(mean, standard deviation) of one gap, in seconds.

        Declared rather than measured from a draw, so ``schedule_span_sec`` is exact
        and costs nothing. A family whose gaps are fixed reports a deviation of 0."""
        raise NotImplementedError

    def schedule(self, n: int, seed: int) -> list[float]:
        gaps = self._gaps(n, seed)
        # accumulate(..., initial=0.0) prepends t=0 and yields n+1 values; drop
        # the trailing total to get length n ([] for n == 0).
        return list(accumulate(gaps[:n], initial=0.0))[:n]

    def schedule_span_sec(self, n: int, *, sigmas: float = 0.0) -> float:
        """The sum of the ``n - 1`` gaps that place ``n`` sessions, with headroom.

        Session 0 leaves at t0, so ``n`` sessions cost ``n - 1`` gaps. The gaps are
        i.i.d., so the sum has mean ``(n-1) * mean`` and deviation
        ``sqrt(n-1) * sd``."""
        if n <= 1:
            return 0.0
        mean, sd = self._gap_moments()
        return (n - 1) * mean + sigmas * math.sqrt(n - 1) * sd


class DeterministicArrival(_GapArrival):
    """Fixed-cadence arrivals: every gap is the same. gap >= 0."""
    gap: float

    @model_validator(mode="after")
    def _validate(self) -> "DeterministicArrival":
        if self.gap < 0:
            raise ValueError(f"gap must be >= 0, got {self.gap}")
        return self

    def _gaps(self, n: int, seed: int) -> list[float]:
        return [self.gap] * n

    def _gap_moments(self) -> tuple[float, float]:
        """Every gap is the same, so the schedule has no spread."""
        return (self.gap, 0.0)

    def describe(self) -> str:
        return f"deterministic arrivals every {self.gap:g}s"

    def mean_rate(self) -> float | None:
        return 1.0 / self.gap


class Poisson(_GapArrival):
    """Exp(1/rate) memoryless arrivals (mean gap = 1/rate)."""
    rate_per_sec: float

    @model_validator(mode="after")
    def _validate(self) -> "Poisson":
        if self.rate_per_sec <= 0.0:
            raise ValueError(f"rate_per_sec must be > 0, got {self.rate_per_sec}")
        return self

    def _gaps(self, n: int, seed: int) -> list[float]:
        rng = random.Random(seed)
        return [rng.expovariate(self.rate_per_sec) for _ in range(n)]

    def _gap_moments(self) -> tuple[float, float]:
        """Exp(rate): mean and deviation are both ``1 / rate``."""
        return (1.0 / self.rate_per_sec, 1.0 / self.rate_per_sec)

    @classmethod
    def covering(cls, n_sessions: int, window_sec: float,
                 *, sigmas: float = 3.0) -> "Poisson":
        """The slowest Poisson process that still places ``n_sessions`` inside
        ``window_sec``, with ``sigmas`` of headroom on the draw.

        The inverse of ``schedule_span_sec``, and how a task with a bounded issuing
        window declares its rate instead of picking one. A rate chosen independently
        of the session count and the window can schedule sessions past the stop. They
        are never opened, and the run scores part of its declared corpus reporting
        nothing unusual.

        Slowest, because the span is what has to fit and a faster process only
        compresses it. What stops a task simply asking for a very fast one is the
        other side: every session that has arrived holds its sandbox open, so arrivals
        far ahead of completions buy disk rather than coverage."""
        if n_sessions < 1:
            raise ValueError(f"n_sessions must be >= 1, got {n_sessions}")
        if window_sec <= 0.0:
            raise ValueError(f"window_sec must be > 0, got {window_sec}")
        if n_sessions == 1:
            # One session leaves at t0 and waits for nothing, so no rate is implied.
            # Return the fastest reading rather than dividing by zero.
            return cls(rate_per_sec=1.0 / window_sec)
        gaps = n_sessions - 1
        return cls(rate_per_sec=(gaps + sigmas * math.sqrt(gaps)) / window_sec)

    def describe(self) -> str:
        return "poisson arrivals"

    def mean_rate(self) -> float | None:
        return self.rate_per_sec


class Gamma(_GapArrival):
    """Gamma(shape, scale) gaps -- burstier than Poisson for shape<1. mean = shape*scale."""
    shape: float
    scale: float

    @model_validator(mode="after")
    def _validate(self) -> "Gamma":
        if self.shape <= 0.0:
            raise ValueError(f"shape must be > 0, got {self.shape}")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        return self

    def _gaps(self, n: int, seed: int) -> list[float]:
        rng = random.Random(seed)
        return [rng.gammavariate(self.shape, self.scale) for _ in range(n)]

    def _gap_moments(self) -> tuple[float, float]:
        return (self.shape * self.scale, math.sqrt(self.shape) * self.scale)

    def describe(self) -> str:
        return f"gamma arrivals (shape={self.shape:g}, scale={self.scale:g})"

    def mean_rate(self) -> float | None:
        return 1.0 / (self.shape * self.scale)


class Uniform(_GapArrival):
    """U(low, high) bounded gaps. low >= 0, high > low."""
    low: float
    high: float

    @model_validator(mode="after")
    def _validate(self) -> "Uniform":
        if self.low < 0.0:
            raise ValueError(f"low must be >= 0, got {self.low}")
        if self.high <= self.low:
            raise ValueError(f"high must be > low, got high={self.high}, low={self.low}")
        return self

    def _gaps(self, n: int, seed: int) -> list[float]:
        rng = random.Random(seed)
        return [rng.uniform(self.low, self.high) for _ in range(n)]

    def _gap_moments(self) -> tuple[float, float]:
        return ((self.low + self.high) / 2.0,
                (self.high - self.low) / math.sqrt(12.0))

    def describe(self) -> str:
        return f"uniform arrivals in [{self.low:g}, {self.high:g}]s"

    def mean_rate(self) -> float | None:
        return 2.0 / (self.low + self.high)


class Normal(_GapArrival):
    """max(0, N(mean, std)) clipped at zero. mean >= 0, std >= 0."""
    mean: float
    std: float

    @model_validator(mode="after")
    def _validate(self) -> "Normal":
        if self.mean < 0.0:
            raise ValueError(f"mean must be >= 0, got {self.mean}")
        if self.std < 0.0:
            raise ValueError(f"std must be >= 0, got {self.std}")
        return self

    def _gaps(self, n: int, seed: int) -> list[float]:
        rng = random.Random(seed)
        return [max(0.0, rng.gauss(self.mean, self.std)) for _ in range(n)]

    def _gap_moments(self) -> tuple[float, float]:
        """The unclipped gap's moments. Clipping at zero shortens the mean and
        narrows the spread, so this reads slightly wide, which is the safe direction
        for a span a window has to cover."""
        return (self.mean, self.std)

    def describe(self) -> str:
        return f"normal arrivals (mean={self.mean:g}, std={self.std:g})"

    def mean_rate(self) -> float | None:
        """The rate of the unclipped gap; clipping at zero shortens the mean gap by
        an amount that is negligible unless ``std`` rivals ``mean``."""
        return 1.0 / self.mean if self.mean > 0.0 else None


class ConstantInterTurn(InterTurnDelay):
    """Fixed think-time before every follow-up turn (seconds). gap >= 0.
    ConstantInterTurn(gap=0.0) is equivalent to NoDelay."""
    gap: float

    @model_validator(mode="after")
    def _validate(self) -> "ConstantInterTurn":
        if self.gap < 0:
            raise ValueError(f"gap must be >= 0, got {self.gap}")
        return self

    def delay(self, turn: int, seed: Any) -> float:
        return self.gap

    def describe(self) -> str:
        return f"constant {self.gap:g}s inter-turn delay"


class GammaInterTurn(InterTurnDelay):
    """Truncated Gamma think-time before every follow-up turn (seconds).

    Grounds inter-turn latency in production traces (Bailian reuse-time / KVCache
    ATC'25): heavy-tailed, median-anchored. ``shape`` comes from the measured
    coefficient of variation (CV = 1/sqrt(shape)); ``scale`` is pinned so the
    realized median matches the trace; ``max_gap`` truncates the tail so a
    session's walk-time stays bounded. All three must be > 0.

    Deterministic: each (session, turn) draws from ``Random(f"{seed}-{turn}")`` --
    a string, since the driver's tuple seed ``(seed, s)`` isn't accepted directly
    by ``Random`` and this keeps the draw independent of PYTHONHASHSEED."""
    shape: float
    scale: float
    max_gap: float

    @model_validator(mode="after")
    def _validate(self) -> "GammaInterTurn":
        if self.shape <= 0.0:
            raise ValueError(f"shape must be > 0, got {self.shape}")
        if self.scale <= 0.0:
            raise ValueError(f"scale must be > 0, got {self.scale}")
        if self.max_gap <= 0.0:
            raise ValueError(f"max_gap must be > 0, got {self.max_gap}")
        return self

    def delay(self, turn: int, seed: Any) -> float:
        g = random.Random(f"{seed}-{turn}").gammavariate(self.shape, self.scale)
        return min(g, self.max_gap)

    def describe(self) -> str:
        return (f"gamma think-time (shape={self.shape:g}, scale={self.scale:g}, "
                f"cap {self.max_gap:g}s)")


# ---- OutputLengthCap -----------------------------------------------------

#: Default per-response token ceiling when a task sets no explicit cap. A run
#: always sends a concrete max_tokens so artifact and oracle decode under the
#: same budget and the oracle's length stays reproducible across model bumps;
#: 1M is effectively unbounded for normal tasks.
DEFAULT_MAX_TOKENS = 1_000_000


class OutputLengthCap(Poly):
    """The max output length the workload permits (L1).

    `max_tokens()` always returns a concrete int: artifact and reference decode
    under the same budget, defaulting to ``DEFAULT_MAX_TOKENS`` when the task
    imposes none. Subclass per policy (Uncapped, ShortOutputCap).
    """

    @abstractmethod
    def max_tokens(self) -> int: ...


class Uncapped(OutputLengthCap):
    """No task-imposed cap: falls back to ``DEFAULT_MAX_TOKENS`` (1M),
    effectively unbounded since generation stops at EOS well before it."""

    def max_tokens(self) -> int:
        return DEFAULT_MAX_TOKENS


class ShortOutputCap(OutputLengthCap):
    """A task-imposed cap on how long a reply may run."""

    #: The cap, in tokens. Stamped on every request as ``max_tokens``, so it bounds
    #: both the artifact's reply and the reference's.
    cap_tokens: int = 16

    def max_tokens(self) -> int:
        return self.cap_tokens


# ---- SLO -----------------------------------------------------------------

class SLO(Poly):
    """The service-level objective the score is derived from (L1).

    `apply` turns a Trace into an SLOResult: an aggregate pass/fail gate, per-event
    throughput credit (what a goodput Scorer multiplies in), and per-event
    SLO-normalized latency ratios (what a SloLatencyScorer means over). Subclass per
    objective (StreamingLatencySLO).
    """

    # An SLO that needs per-token streaming timing sets this True; check_wiring
    # then requires transport='http-sse' + protocol.emits_token_ids. Class-level,
    # not a field.
    requires_token_timing: ClassVar[bool] = False

    #: The unit ``per_event_ratio`` is keyed by, and the unit a ``SloLatencyScorer``
    #: weights equally. ``"request"`` keys by ``request_id``, ``"session"`` by
    #: ``session_id``. Class-level, not a field: it is a property of the objective's
    #: shape, not something a task tunes.
    grain: ClassVar[Literal["request", "session"]] = "request"

    @abstractmethod
    def apply(self, trace: "Trace", workload: "Workload") -> SLOResult:
        """Score one tenant's slice."""

    def units(self, trace: "Trace") -> "dict[str, list[TraceEvent]]":
        """The trace's units at this SLO's grain, keyed as ``per_event_ratio`` is, so
        a scorer means over them without branching on the grain itself: one event per
        unit at request grain, a session's whole event list per unit at session
        grain."""
        if self.grain == "session":
            return trace.by_session()
        return {ev.request_id: [ev] for ev in trace.events}

    def censored_ratio(self, unit_events: "list[TraceEvent]",
                       t_end: float) -> "float | None":
        """The SLO-normalized latency to charge one unit when it produced none to
        measure, censored at ``t_end`` -- the harness time the whole run ended
        (``Trace.t_end()`` of the full trace, not of one tenant's slice).
        ``unit_events`` is that unit's events, as ``units`` groups them.

        None when the SLO defines no ratio to censor, which is every SLO that is not a
        latency normalizer. A scorer asks only about the units ``apply`` gave no ratio,
        and only to report a bound on a run it has already gated; the answer is never
        part of a score."""
        return None

    def describe(self) -> str:
        """Human-readable label for use in the implementer brief."""
        return type(self).__name__.lower()


class NoSLO(SLO):
    """No latency objective: the gate always passes and every request earns full
    credit, so goodput is bounded by correctness alone. Defines no per-event ratio,
    censored or otherwise, so it is meaningless to a ``SloLatencyScorer``. Use for a
    task that doesn't constrain latency; a task that does should use
    ``StreamingLatencySLO``."""

    def describe(self) -> str:
        return "none (no latency SLO)"

    def apply(self, trace: "Trace", workload: "Workload") -> SLOResult:
        return SLOResult(
            aggregate_passed=True,
            per_event_credit={ev.request_id: 1.0 for ev in trace.events},
            diagnostics={"n": len(trace.events)},
        )


class TrajectoryMakespanSLO(SLO):
    """Per-trajectory SLO-normalized latency: a normalizer, not a pass/fail bar.

    A trajectory is every TraceEvent sharing a ``session_id`` -- one agentic task end
    to end, however many LLM calls it took. Its makespan is
    ``max(t_recv) - min(t_send)``, and::

        r = makespan / target_sec        # lower is better, one term per trajectory

    The grain is the trajectory, so ``per_event_ratio`` is keyed by session id and a
    ``SloLatencyScorer`` weights each trajectory once. Weighting requests instead
    would weight a trajectory by its turn count, and a serving stack does not choose
    how many turns an agentic task takes.

    ``target_sec`` normalizes and nothing else. It cancels out of the speedup score,
    since the contestant and the baseline stack divide by the same number, so its one
    lasting role is the attainment diagnostic: the fraction of trajectories that came
    in under it. It is never a gate. A trajectory over target is scored on how far
    over, not zeroed.

    Hard gate: every request must have answered. A request with no response, or one
    with no ``t_recv``, leaves its trajectory with no measurable makespan, so that
    trajectory gets no ratio and ``aggregate_passed`` is False. A failed request must
    never be silently dropped, or a server that sheds load would score best.

    Such a trajectory can still be priced, by ``censored_ratio``, so a zeroed run can
    report an upper bound on the score it was reaching for. That is never part of a
    score.

    The binary "met" notion (r < 1) survives only as per-event credit for a goodput
    scorer and the ``n_met``/``met_frac`` diagnostics; it never feeds r.
    """

    grain: ClassVar[Literal["request", "session"]] = "session"

    #: The normalizer, in seconds. Strictly positive: every ratio divides by it, so a
    #: zero or negative target is refused here rather than at the first division.
    target_sec: float = Field(gt=0.0)

    def describe(self) -> str:
        return f"makespan < {self.target_sec:.0f} s (per trajectory)"

    def apply(self, trace: "Trace", workload: "Workload") -> SLOResult:
        sessions = trace.by_session()
        makespans: dict[str, float] = {}   # session id -> makespan, in seconds
        n_missing = 0                      # requests that never answered
        for sid, events in sessions.items():
            unanswered = [ev for ev in events
                          if ev.response is None or ev.t_recv is None]
            n_missing += len(unanswered)
            if unanswered:
                # No last response, so the trajectory's makespan is not measurable.
                continue
            makespans[sid] = max(max(ev.t_recv for ev in events)
                                 - min(ev.t_send for ev in events), 0.0)

        ratios = {sid: m / self.target_sec for sid, m in makespans.items()}
        credit: dict[str, float] = {}
        for sid, events in sessions.items():
            met = sid in ratios and ratios[sid] < 1.0
            for ev in events:
                credit[ev.request_id] = 1.0 if met else 0.0

        n_met = sum(1 for r in ratios.values() if r < 1.0)
        return SLOResult(
            # Every request must have produced a response the harness timed.
            aggregate_passed=(n_missing == 0),
            per_event_ratio=ratios,
            # Binary met/not-met, for a goodput scorer's credit. Keyed by request, so
            # every turn of a trajectory carries its trajectory's outcome.
            per_event_credit=credit,
            diagnostics={
                # The attainment reading of the target, which is not a gate.
                "p99_makespan_sec": _p99(list(makespans.values())),
                "target_sec": self.target_sec,
                "n_sessions": len(sessions),
                "n_met": n_met,
                "met_frac": (n_met / len(sessions)) if sessions else 0.0,
                "n_missing": n_missing,
            },
        )

    def censored_ratio(self, unit_events: "list[TraceEvent]",
                       t_end: float) -> "float | None":
        """What a trajectory with no measurable makespan is charged: the span from when
        its first request was due to the moment the harness stopped watching its last.

        Due is ``t_planned_send``, the open-loop schedule's intended send time, rather
        than ``t_send``, which the driver stamps at the skip instant for a request its
        circuit breaker never sent. Stopped watching is per request, the latest over
        the trajectory's own events (see ``_t_stopped_watching``), so a trajectory that
        timed out is charged up to its deadline rather than however long the rest of
        the drive happened to run.

        Floored at r = 1: a trajectory that did not finish did not meet its target, so
        it earns at most what one landing exactly on it earns, which is nothing."""
        t_start = min(ev.t_planned_send if ev.t_planned_send is not None else ev.t_send
                      for ev in unit_events)
        t_stop = max(_t_stopped_watching(ev, t_end) for ev in unit_events)
        return max(1.0, (t_stop - t_start) / self.target_sec)


def _p99(values: list[float]) -> float:
    """The 99th percentile of ``values``, by nearest rank; 0.0 over nothing."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, int(round(0.99 * (len(ordered) - 1))))]


def p99_ms_of(values_sec: list[float]) -> float:
    """The 99th percentile of ``values_sec``, in milliseconds."""
    return _p99([v * 1000.0 for v in values_sec])


def _t_stopped_watching(ev: "TraceEvent", t_end: float) -> float:
    """When the harness stopped watching ``ev``, in order: its own
    ``failure.t_abandoned`` when it has a failure record, its ``t_recv`` when it
    answered but produced no timing, and the end of the run when neither is known.

    So a request that timed out is charged up to its deadline and one that answered at
    ten seconds is charged those ten seconds, rather than however long the rest of the
    drive happened to run. Used by the censored ratios, never by a score."""
    if ev.failure is not None:
        return ev.failure.t_abandoned
    if ev.t_recv is not None:
        return ev.t_recv
    return t_end


#: ``TraceEvent.extras`` key carrying how long a request's prompt was, in tokens, as
#: the harness counted it. Read by ``ContextLinearLatencySLO``, and stamped by
#: whichever framework built the prompt.
#:
#: The count must be the harness's own, never the server's reported
#: ``usage.prompt_tokens``: a slope turns prompt length into budget, so a server that
#: over-reported its prompt would buy itself a longer deadline.
PREFILL_TOKENS_EXTRA = "prefill_tokens"

#: ``TraceEvent.extras`` key flagging, as ``True``, a request the drive's scheduled
#: stop cut while it was still streaming (``hls.spec.agent.ScheduledStop``). Absent on
#: every other event.
#:
#: A cut request is not a failed one: it carries no ``error`` and no ``failure``, and
#: the event holds the tokens it did stream with a ``t_recv`` of the cut instant. This
#: flag is what tells the readers of those fields that the reply stops where the
#: harness stopped it rather than where the server did -- the latency SLO prices a cut
#: request that never reached a first token on its whole wait, and the report gate
#: skips it in the end-of-turn count because it did not end.
CUT_EXTRA = "cut"

#: ``TraceEvent.extras`` key carrying the longest a request's stream went between two
#: tokens, in seconds. Report-only: nothing scored or gated reads it.
#:
#: It is what a stall budget is chosen against (``AgentSpec.stall_timeout_sec``). The
#: budget has to sit well above the gaps a healthy server already shows under load, and
#: the only honest source for those is the reference serving the task's own traffic.
MAX_TOKEN_GAP_EXTRA = "max_token_gap_sec"


class StreamingLatencySLO(SLO):
    """Per-request SLO-normalized latency: a per-tenant normalizer, not a pass/fail bar.

    Each request's latency is divided by its tenant's budget, so different tenants'
    TTFT/TPOT targets become comparable on one scale, and the two axes collapse to
    the more stringent one::

        r_ttft = TTFT_ms / ttft_ms
        r_tpot = mean_TPOT_ms / tpot_ms      # < 2 tokens -> 0.0 (TTFT alone gates)
        r      = max(r_ttft, r_tpot)         # lower is better

    Uses max, not a sum: a sum would let a tiny TTFT buy back a blown TPOT, so a
    request well over its TPOT budget could outscore a fully compliant one.
    Conjunctive semantics per DistServe (Zhong et al., OSDI 2024): a request is
    constrained by the more stringent of TTFT and TPOT.

    r is continuous, with no plateau or cutoff: r=0.5 beats r=1.0 beats r=2.0. A
    credit that plateaued at full credit for any request at or under budget would,
    under a fixed open-loop arrival, cap every compliant implementation's score at
    the offered load.

    r is the normalization only; the objective built on it is
    ``hls.scoring.SloLatencyScorer``'s ``mean(log(1/r))``. This class owns one
    thing: what a request's latency is worth relative to its own tenant's budget.

    Hard gate: a request with no computable r (no response, or no first-token
    timing) gets no ratio entry and trips ``aggregate_passed=False``, failing the
    run. A failed request must never be silently dropped, or a server that sheds
    load would score best. (Transport faults also fail per event in the verify
    path; this gate additionally covers the assume-correct drives of a baseline
    credit and the no-timing case.)

    Such a request can still be priced, by ``censored_ratio``: its latency is taken to
    run from when it was due to the moment the harness stopped watching it, and is at
    least its whole TTFT budget. That is the most generous reading of a request that
    never answered -- it is charged only for time it was observably outstanding --
    while still crediting it nothing, since it did not meet its SLO. It exists so a
    zeroed run can report an upper bound on the score it was reaching for, and it is
    never part of the score itself.

    The binary "met" notion (r < 1) survives only as per-event credit for a
    goodput scorer and the ``n_met``/``met_frac`` diagnostics; it never feeds r.

    Needs a streaming contract that emits token ids -- check_wiring enforces
    transport='http-sse' + protocol.emits_token_ids at authoring time."""

    requires_token_timing: ClassVar[bool] = True

    ttft_ms: float
    tpot_ms: float

    def describe(self) -> str:
        return f"TTFT < {self.ttft_ms:.0f}ms, TPOT < {self.tpot_ms:.0f}ms (per request)"

    def _ratios(self, ev: "TraceEvent") -> "tuple[float, float] | None":
        """(r_ttft, r_tpot) = (latency / SLO) per axis, or None with no first-token
        timing (unary path / errored). r_tpot uses mean inter-token latency
        ((last-first)/(n-1)) / tpot_ms, or 0.0 under 2 tokens (TTFT alone then gates)."""
        resp = ev.response
        if resp is None or resp.t_first_packet is None:
            return None
        r_ttft = (resp.t_first_packet - ev.t_send) * 1000.0 / self.ttft_ms
        ts = resp.packet_timestamps
        if ts and len(ts) >= 2:
            mean_tpot_ms = (ts[-1] - ts[0]) / (len(ts) - 1) * 1000.0
            r_tpot = mean_tpot_ms / self.tpot_ms
        else:
            r_tpot = 0.0
        return (r_ttft, r_tpot)

    def _r(self, ev: "TraceEvent") -> "float | None":
        """SLO-normalized latency r = max(r_ttft, r_tpot), never a sum. None for a
        request with no computable ratio; ``apply`` fails the run on it rather than
        letting it vanish from the mean."""
        ratios = self._ratios(ev)
        return None if ratios is None else max(ratios)

    def censored_ratio(self, unit_events: "list[TraceEvent]",
                       t_end: float) -> "float | None":
        """What a request with no measured latency is charged: the time it was
        observably outstanding, against the TTFT budget. The grain is the request, so
        ``unit_events`` holds exactly one.

        It runs from when the request was due to the moment the harness stopped
        watching it. Due is ``t_planned_send``, the open-loop schedule's intended send
        time, rather than ``t_send``, which the driver stamps at the skip instant for a
        request its circuit breaker never sent. Stopped watching is
        ``_t_stopped_watching``: its deadline for a request that timed out, its arrival
        for one that answered without streaming.

        Floored at r = 1: a request that did not answer did not meet its SLO, so it
        earns at most what one landing exactly on its budget earns, which is nothing."""
        [ev] = unit_events
        t_start = ev.t_planned_send if ev.t_planned_send is not None else ev.t_send
        t_stop = _t_stopped_watching(ev, t_end)
        return max(1.0, (t_stop - t_start) * 1000.0 / self.ttft_ms)

    def apply(self, trace: "Trace", workload: "Workload") -> SLOResult:
        ratios = {ev.request_id: r for ev in trace.events
                  if (r := self._r(ev)) is not None}
        n_total = len(trace.events)
        n_missing = n_total - len(ratios)          # requests with no computable r
        n_met = sum(1 for r in ratios.values() if r < 1.0)   # binary diagnostic
        # Mean over requests that have an r; a run with any missing r already fails
        # via the gate below, so this stays a readable diagnostic.
        mean_r = (sum(ratios.values()) / len(ratios)) if ratios else 0.0
        return SLOResult(
            # Every request must have produced a complete streaming response.
            aggregate_passed=(n_missing == 0),
            per_event_ratio=ratios,
            # Binary met/not-met, for a goodput scorer's credit. Requests with no r
            # are absent (the gate above already fails their run).
            per_event_credit={rid: (1.0 if r < 1.0 else 0.0) for rid, r in ratios.items()},
            diagnostics={"p99_ttft_ms": p99_ms_of(trace.ttft_sec()),
                         "p99_tpot_ms": p99_ms_of(trace.tpot_sec()),  # per-request mean TPOT
                         "ttft_ms": self.ttft_ms, "tpot_ms": self.tpot_ms,
                         "mean_r": mean_r,  # per-tenant metric; 1.0 = exactly at SLO
                         "n_met": n_met, "n_total": n_total,
                         "met_frac": (n_met / n_total) if n_total else 0.0,
                         "n_missing": n_missing,
                         "has_timing": bool(trace.ttft_sec())})


class RequestLatencySLO(SLO):
    """Per-request unary latency bar: a request is credited iff its OWN JCT
    (``t_recv - t_send``) is under ``target_ms``; an errored or unanswered request is
    not credited. No aggregate cliff -- ``aggregate_passed`` is always True. The bar is
    per request, and a request that never answered is the scorer's to refuse (below),
    so the SLO has no run-level rule of its own to add.

    It also normalizes: ``per_event_ratio`` reports ``r = JCT / target_ms``, the unary
    counterpart of ``StreamingLatencySLO``'s ``max(r_ttft, r_tpot)``. That is what a
    ``SloLatencyScorer`` reads, so a task whose contract is a single unary response can
    be scored on SLO-normalized latency alongside a streaming one. A request with no
    measured JCT gets no ratio; the scorer fails the run on it rather than letting it
    vanish from the mean, so nothing is gained by never answering."""

    target_ms: float

    def describe(self) -> str:
        return f"JCT < {self.target_ms:.0f}ms (per request)"

    def _meets(self, ev: "TraceEvent") -> bool:
        if ev.response is None or ev.error or ev.t_recv is None:
            return False
        return (ev.t_recv - ev.t_send) * 1000.0 < self.target_ms

    def _r(self, ev: "TraceEvent") -> "float | None":
        """SLO-normalized latency ``JCT / target_ms``. None for a request that never
        produced a JCT to measure (errored, or still outstanding when the drive ended)."""
        if ev.response is None or ev.error or ev.t_recv is None:
            return None
        return (ev.t_recv - ev.t_send) * 1000.0 / self.target_ms

    def censored_ratio(self, unit_events: "list[TraceEvent]",
                       t_end: float) -> "float | None":
        """What a request with no measured JCT is charged: the time it was observably
        outstanding, against the same budget. The grain is the request, so
        ``unit_events`` holds exactly one.

        Mirrors ``StreamingLatencySLO``: due at ``t_planned_send`` (the open-loop
        schedule's intent, not the instant a circuit breaker stamped a request it never
        sent), and watched until its own ``failure.t_abandoned``, else ``t_recv``, else
        the end of the run. Floored at r = 1, since a request that did not answer did
        not meet its SLO and so can earn no credit."""
        [ev] = unit_events
        t_start = ev.t_planned_send if ev.t_planned_send is not None else ev.t_send
        if ev.failure is not None:
            t_stop = ev.failure.t_abandoned
        elif ev.t_recv is not None:
            t_stop = ev.t_recv
        else:
            t_stop = t_end
        return max(1.0, (t_stop - t_start) * 1000.0 / self.target_ms)

    def apply(self, trace: "Trace", workload: "Workload") -> SLOResult:
        credit = {ev.request_id: (1.0 if self._meets(ev) else 0.0) for ev in trace.events}
        ratios = {ev.request_id: r for ev in trace.events
                  if (r := self._r(ev)) is not None}
        n_total = len(trace.events)
        n_met = sum(1 for v in credit.values() if v > 0)
        return SLOResult(
            aggregate_passed=True,
            per_event_ratio=ratios,
            per_event_credit=credit,
            diagnostics={"p99_ms": p99_ms_of(trace.latencies_sec()),
                         "target_ms": self.target_ms, "n_met": n_met, "n_total": n_total,
                         "n_missing": n_total - len(ratios),
                         "met_frac": (n_met / n_total) if n_total else 0.0},
        )


class ContextLinearLatencySLO(StreamingLatencySLO):
    """``StreamingLatencySLO`` with a TTFT budget that grows with the prompt::

        budget_ms = ttft_ms + ttft_ms_per_prefill_token * prefill_tokens

    For traffic whose prompts span orders of magnitude, which is every long-horizon
    agent: a session's first turn prefills a few thousand tokens and its last prefills
    several hundred thousand. Prefill time is roughly linear in prompt length, so a
    flat deadline that the first turn clears easily is one no stack can meet at the
    last, and the deep turns would all fail for a reason that says nothing about how
    well the artifact serves. TPOT keeps a flat budget, since decode cost per token
    need not grow with the cache and whether it does is a property of the checkpoint.

    A separate family member rather than a slope on the flat one. The rule a run was
    priced under is part of its record, and every stored baseline credit compares its
    tenants' serialized SLOs against the task's current ones; a task that wants a flat
    budget keeps exactly the record it had.

    What it requires of the task: the framework must stamp each request's prompt
    length into ``TraceEvent.extras[PREFILL_TOKENS_EXTRA]``, counted harness-side. A
    request with no such stamp cannot be budgeted, gets no ratio, and fails the run --
    loudly, because pricing it against the flat term would be a plausible number
    computed from a rule the task did not declare."""

    #: Milliseconds of TTFT budget per token of prompt, on top of ``ttft_ms``.
    #:
    #: A serving measurement, not a policy choice: it is the baseline stack's own
    #: prefill rate on the task's hardware, so it comes from a profile of that stack
    #: across the prompt lengths the traffic reaches. 0 leaves the flat budget, and is
    #: what a task declares while its profile is still pending -- a slope of zero also
    #: excuses the framework from stamping anything, so the requirement above arrives
    #: with the measurement.
    ttft_ms_per_prefill_token: float = 0.0

    def describe(self) -> str:
        return (f"TTFT < {self.ttft_ms:.0f}ms + "
                f"{self.ttft_ms_per_prefill_token:.4g}ms per prompt token, "
                f"TPOT < {self.tpot_ms:.0f}ms (per request)")

    def ttft_budget_ms(self, ev: "TraceEvent") -> "float | None":
        """The flat term plus the slope times this request's prompt length, or None
        where a slope is declared and nothing stamped one."""
        if not self.ttft_ms_per_prefill_token:
            return self.ttft_ms
        prefill = ev.extras.get(PREFILL_TOKENS_EXTRA)
        if not isinstance(prefill, (int, float)) or isinstance(prefill, bool):
            return None
        return self.ttft_ms + self.ttft_ms_per_prefill_token * float(prefill)

    def budget_diagnostics(self) -> dict:
        """What the run's record has to say about how the TTFT budget was computed."""
        return {"ttft_ms_per_prefill_token": self.ttft_ms_per_prefill_token}

    def _stamps(self, ev: "TraceEvent") -> "tuple[float | None, list[float]] | None":
        """(first-token time, token arrival times) as this SLO prices them, or None for
        an event with no response at all.

        The one place the cut's phantom token is added, so the ratios and the
        impossibility check below read the same timing. ``t_first`` is None for a
        request that reached no first token and was not cut."""
        resp = ev.response
        if resp is None:
            return None
        ts = list(resp.packet_timestamps or ())
        t_first = resp.t_first_packet
        if bool(ev.extras.get(CUT_EXTRA)) and ev.t_recv is not None:
            ts.append(ev.t_recv)
            if t_first is None:
                t_first = ev.t_recv
        return (t_first, ts)

    def impossible_timing(self, ev: "TraceEvent") -> "str | None":
        """Why this request's timing cannot be a measurement, or None if it can.

        A token that arrived before the request was sent, or a last token before the
        first, is not a fast server. It is a harness whose clocks disagree, and it must
        never reach the arithmetic below: a negative TTFT on a reply of under two
        tokens yields max(negative, 0.0) = 0.0, which is the best ratio the metric can
        award. So the impossible measurement would score better than every honest
        request, and the run would read as excellent rather than broken.

        Refused rather than clamped, and counted under its own name in the
        diagnostics, because clamping to r = 1 would turn a wiring fault into a
        plausible number nobody goes looking for."""
        stamps = self._stamps(ev)
        if stamps is None:
            return None
        t_first, ts = stamps
        if t_first is not None and t_first < ev.t_send:
            return (f"first token at {t_first!r} precedes the send at {ev.t_send!r}, "
                    f"a TTFT of {(t_first - ev.t_send) * 1000.0:.1f}ms")
        if len(ts) >= 2 and ts[-1] < ts[0]:
            return (f"last token at {ts[-1]!r} precedes the first at {ts[0]!r}, "
                    f"a span of {(ts[-1] - ts[0]) * 1000.0:.1f}ms")
        return None

    def _ratios(self, ev: "TraceEvent") -> "tuple[float, float] | None":
        """(r_ttft, r_tpot) against this request's own TTFT budget, or None with no
        first-token timing (unary path / errored) and none with no TTFT budget (a slope
        was declared and the request carries no prompt length). r_tpot uses mean
        inter-token latency ((last-first)/(n-1)) / tpot_ms, or 0.0 under 2 tokens
        (TTFT alone then gates).

        A request the scheduled stop cut (``CUT_EXTRA``) is priced as though one more
        token had completed at the cut instant. Its ``n`` streamed tokens plus that
        phantom then go through the rule above unchanged: with nothing streamed the
        phantom is the first token, so the whole wait is the time to first token and
        there is no rate to measure; with ``n`` tokens the mean inter-token latency is
        the span from the first token to the cut over ``n`` intervals.

        The phantom is a bound, not a guess. The token that never came would have
        arrived no earlier than the cut, so every latency this yields is a lower bound
        on the true one and the request's contribution to the score an upper bound. It
        is also what stops a request being credited for a fast start it did not
        sustain: a server that streams a thousand tokens and then goes silent for ten
        minutes is charged that silence rather than the rate it managed before it.

        The phantom exists only here. Nothing adds it to the reply's tokens, so the
        token-level gate grades exactly what the server sent.

        A cut request still needs a budget like any other: the prompt-length stamp is
        a wiring rule, not a latency one.

        Timing that cannot be a measurement gets no ratio either; see
        ``impossible_timing``."""
        stamps = self._stamps(ev)
        if stamps is None:
            return None
        t_first, ts = stamps
        if t_first is None:
            return None
        why = self.impossible_timing(ev)
        if why is not None:
            log.error("SLO: request %s is refused a ratio, its timing is impossible: "
                      "%s", ev.request_id, why)
            return None
        budget_ms = self.ttft_budget_ms(ev)
        if budget_ms is None:
            return None
        r_ttft = (t_first - ev.t_send) * 1000.0 / budget_ms
        if len(ts) >= 2:
            mean_tpot_ms = (ts[-1] - ts[0]) / (len(ts) - 1) * 1000.0
            r_tpot = mean_tpot_ms / self.tpot_ms
        else:
            r_tpot = 0.0
        return (r_ttft, r_tpot)

    def censored_ratio(self, unit_events: "list[TraceEvent]",
                       t_end: float) -> "float | None":
        """``StreamingLatencySLO.censored_ratio`` against this request's own budget
        where its prompt length is known, and against the flat term where it is not. A
        request that never reached the server may carry no length, and the flat term is
        the harsher reading of the two, which is the right direction for a number that
        only ever reports what a zeroed run was reaching for."""
        [ev] = unit_events
        t_start = ev.t_planned_send if ev.t_planned_send is not None else ev.t_send
        t_stop = _t_stopped_watching(ev, t_end)
        budget_ms = self.ttft_budget_ms(ev) or self.ttft_ms
        return max(1.0, (t_stop - t_start) * 1000.0 / budget_ms)

    def apply(self, trace: "Trace", workload: "Workload") -> SLOResult:
        """``StreamingLatencySLO.apply`` on this SLO's ratios, with the budget rule and the
        three reasons a request has no ratio counted apart: no budget (a wiring fault),
        impossible timing (clocks that disagree) and no response (the server's own)."""
        result = super().apply(trace, workload)
        diagnostics = {**result.diagnostics, **self.budget_diagnostics(),
                       "n_no_budget": sum(1 for ev in trace.events if self.ttft_budget_ms(ev) is None),
                       "n_impossible_timing": sum(1 for ev in trace.events
                                                  if self.impossible_timing(ev) is not None),
                       "n_no_response": sum(1 for ev in trace.events if ev.response is None)}
        return result.model_copy(update={"diagnostics": diagnostics})


# ---- Workload ------------------------------------------------------------

class Workload(Spec):
    """How one tenant's traffic is offered: when requests arrive, how long a reply may
    run, how deep a live session may go, and what the run is held to."""

    #: When each session arrives, drawn at draw time from the task seed.
    arrival: ArrivalProcess
    #: The think-time between one turn's reply and the next turn of the same session.
    inter_turn_delay: InterTurnDelay
    #: The cap stamped on every request as ``max_tokens``.
    output_length_cap: OutputLengthCap
    #: The service-level objective the score is derived from.
    slo: SLO
    #: Mean turns per session of the drawn workload, the factor between the arrival
    #: process's session rate and the offered request rate. 1 for a single-turn
    #: tenant. Declared by the task, which derives its session rate from the same
    #: constant, so the request rate the brief states is the one the task set.
    mean_turns_per_session: float = 1.0
    #: Context capacity of a live multi-turn session, in characters. The first turn
    #: whose conversation would exceed this is not sent, and the session ends there;
    #: later turns produce no event, like a turn ``trim_rampdown`` excludes. Needed
    #: by envelope-limited tenants whose history grows with the model's own replies,
    #: since an uncapped deep session eventually pushes a prompt past the serving
    #: context and hard-fails.
    #:
    #: Ends the session rather than trimming old turns, matching what a real client
    #: does (trimming requires summarization, which would inject harness-authored
    #: text into the prompt); keeps every prompt an append-only extension of the
    #: previous turn's.
    #:
    #: The cutoff depends on the replies the artifact generated. A session whose
    #: first turn alone exceeds the budget raises, since that is a corpus/tenant
    #: mismatch rather than a runtime event.
    #:
    #: Measured in characters, not tokens, since the driver never tokenizes; the
    #: task derives the budget from its serving envelope at a conservative
    #: chars-per-token floor. None = no cap.
    history_char_budget: int | None = None
    #: When True, excludes follow-up turns whose nominal dispatch time (session
    #: arrival plus cumulative think-time, assuming zero service time) falls past
    #: the last scheduled session arrival, so trace duration tracks the arrival
    #: span instead of trailing off with the last sessions' deep turns. Membership
    #: is decided at draw time (drivers.rampdown_turn_budget), not from the wall
    #: clock, so every contestant serves an identical request set. Single-turn
    #: tenants are unaffected.
    trim_rampdown: bool = False

    def prompt_fields(self) -> dict:
        """Brief-friendly strings for the ## Workload section of the implementer brief.

        ``sessions`` is the arrival process with its session rate; ``requests`` is the
        turns-per-session factor and the request rate it yields, None when the process
        has no rate and the tenant is single-turn (offline batch); ``think_time`` is the
        inter-turn delay, None for a single-turn tenant, which has no turn to wait
        before. ``request_rate`` is the number behind ``requests``, for the brief's
        aggregate over tenants."""
        cap = self.output_length_cap.max_tokens()
        rate = self.arrival.mean_rate()
        turns = self.mean_turns_per_session
        request_rate = None if rate is None else rate * turns
        sessions = self.arrival.describe()
        if rate is not None:
            sessions += f", {rate:.4g} sessions/s"
        turns_text = f"{turns:.4g} turn{'' if turns == 1 else 's'}/session"
        if request_rate is not None:
            requests = f"{turns_text}, {request_rate:.4g} requests/s"
        elif turns != 1:
            requests = turns_text
        else:
            requests = None
        return {
            "sessions": sessions,
            "requests": requests,
            "request_rate": request_rate,
            "think_time": (None if turns == 1 else self.inter_turn_delay.describe()),
            # Carries its own unit, because the two branches do not share one: a cap
            # reads "256 tokens" and an absent one "uncapped (<= 1000000 tokens)", which
            # already ends in the word.
            "output_cap": (f"uncapped (<= {DEFAULT_MAX_TOKENS} tokens)"
                           if cap == DEFAULT_MAX_TOKENS else f"{cap} tokens"),
            "slo": self.slo.describe(),
        }
