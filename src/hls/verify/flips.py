"""What the token-level gate measures, defined once so both sides agree.

Take every token position's gap -- the served token's log-probability minus
the reference's greedy argmax's, so 0 means agreement and positive means the
served token lost -- and reduce the whole trace to a handful of counters:
positions seen, positions that disagreed (flips), the worst disagreement, how
many disagreements cleared each rung of the severity grid, and how many gaps
were non-finite.

Calibration measures a trusted engine's servings to find the clean envelope
(``hls.calibrate.score_draw``); the deployed verifier measures a contestant's
trace the same way to grade against it
(``hls.verify.verifiers.TokenFaithfulnessVerifier``). Both sides must compute
the identical summary, or they would compare numbers that do not mean the
same thing -- so the reduction is defined here, once, and both sides call it.

Semantics:

  - A flip is a strictly positive gap (``gap > 0``). A served token equal to
    the reference's greedy argmax subtracts a tensor element from itself and
    yields bitwise 0.0, so an exact tie -- a different id at an identical
    log-probability -- is not a flip.
  - The severity grid is a set of rungs in nats; the count at a rung is the
    number of flips whose gap is at least that rung, so counts are nested and
    non-increasing along the grid. Rungs come from the half-nat lattice
    starting at ``SEVERITY_FLOOR_NATS`` (``candidate_rungs``); which of them
    the calibration keeps is decided from the pool's own counts
    (``hls.calibrate.pool_grid``): a rung whose clean count equals the rung
    below it has the same ceiling and adds nothing.
  - A reply's record keeps every gap at or above the floor (``severe_gaps``)
    rather than counts on a grid, so the counts for any grid that starts at
    the floor are a file read.
  - Non-finite gaps are counted, never silently dropped. NaN loses every
    ``>`` comparison, so an unnoticed NaN would report a perfect trace for a
    numerically broken reference; +inf means the served id was outside the
    reference's vocabulary. This module only counts them; what to do about
    them is the caller's policy -- calibration refuses the campaign, the
    verifier fails the trace closed.
  - Which replies to count is the caller's choice, but both callers count
    every reply they are handed: nothing a served reply reports about itself
    keeps it out of the measurement. A reply served past the cap the harness
    stamped on its request (``past_cap``) is measured by neither side:
    calibration refuses the pool member that served it
    (``hls.calibrate.refuse_past_cap``), and the verifier fails the trace on
    it, each before the reference is asked to teacher-force it.

``score_reply`` turns one reply's gaps into a ``ReplyFlips`` record;
``PooledFlips`` sums records over a measurement and reads the counts off a
grid.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

#: Lowest rung of every severity grid, in nats. Gaps below it are the benign band two
#: correct bf16 stacks already disagree in; the grid measures from its shoulder.
SEVERITY_FLOOR_NATS: float = 0.5

#: Spacing between rungs, in nats.
SEVERITY_STEP_NATS: float = 0.5


def candidate_rungs(max_gap: float) -> list[float]:
    """The half-nat lattice a grid is chosen from: rungs from the floor, in steps, up to
    one step above ``max_gap`` rounded up, so a pool's worst gap always has a rung it
    reaches and one it does not."""
    if not math.isfinite(max_gap):
        raise ValueError(f"a severity grid needs a finite worst gap, got {max_gap}")
    top = max(SEVERITY_FLOOR_NATS,
              math.ceil(round(max_gap / SEVERITY_STEP_NATS, 9)) * SEVERITY_STEP_NATS)
    n = int(round((top - SEVERITY_FLOOR_NATS) / SEVERITY_STEP_NATS)) + 2
    return [round(SEVERITY_FLOOR_NATS + i * SEVERITY_STEP_NATS, 9) for i in range(n)]


@dataclass(frozen=True)
class ReplyFlips:
    """One served reply's measurement, in the form both sides persist per event.

    Grid-free: the grid is read from the calibration artifact at grading time and
    chosen from the pool at calibration time, so the record stores the gaps
    themselves where they matter and the counts are derived (``level_counts``).

    Written one record per line into ``<draw>.replies.jsonl.gz`` under a calibration
    serving's identity header, and into the graded run's per-reply sidecar."""

    #: ``hls.core.trace.event_key`` of the request.
    event_key: str
    #: Positions counted: the served length, plus one when a stop row was scored.
    n_tokens: int
    #: Positions with a strictly positive, non-NaN gap.
    n_flips: int
    #: Worst non-NaN gap, in nats (0.0 for an all-agree reply; +inf if one occurred).
    max_gap: float
    #: Every gap at or above ``SEVERITY_FLOOR_NATS``, ascending, +inf included.
    severe_gaps: tuple[float, ...]
    #: NaN positions.
    n_nan: int
    #: +/-inf positions.
    n_inf: int
    #: The stop row's gap, or ``None`` when the reference scored none: the reply ended
    #: on an end-of-turn id, or its served length reached the request's cap.
    stop_gap: float | None
    #: Whether the served length reached the ``max_tokens`` the harness stamped.
    at_cap: bool

    def level_counts(self, levels: Sequence[float]) -> list[int]:
        """Flips at or above each rung of ``levels`` (ascending), nested."""
        n = len(self.severe_gaps)
        return [n - bisect_left(self.severe_gaps, c) for c in levels]

    def to_json(self) -> dict:
        """Plain-JSON form, one key per field."""
        return {"event_key": self.event_key, "n_tokens": self.n_tokens,
                "n_flips": self.n_flips, "max_gap": self.max_gap,
                "severe_gaps": list(self.severe_gaps), "n_nan": self.n_nan,
                "n_inf": self.n_inf, "stop_gap": self.stop_gap, "at_cap": self.at_cap}

    @classmethod
    def from_json(cls, d: dict) -> "ReplyFlips":
        """Inverse of ``to_json``."""
        return cls(event_key=d["event_key"], n_tokens=d["n_tokens"], n_flips=d["n_flips"],
                   max_gap=d["max_gap"], severe_gaps=tuple(d["severe_gaps"]),
                   n_nan=d["n_nan"], n_inf=d["n_inf"], stop_gap=d["stop_gap"],
                   at_cap=d["at_cap"])


def past_cap(n_served: int, cap: int) -> bool:
    """Whether a reply that served *n_served* units went past *cap*, the output cap the harness
    stamped on its request in the same unit (tokens for text, frames for speech). Such a reply
    ignored the cap it was sent. Each of these asks this one test: every faithfulness gate
    fails the trace on such a reply, a token-level calibration refuses the pool member that
    served it (``hls.calibrate.refuse_past_cap``), a report calibration records it as a
    contract violation and leaves it out of the measurement
    (``ReportFaithfulnessVerifier._rows_for``), and ``hls capped-requests`` refuses a member
    whose captures hold one (``hls.calibrate.capped_rows``)."""
    return n_served > cap


def score_reply(gaps: Sequence[float], *, event_key: str, n_served: int,
                at_cap: bool) -> ReplyFlips:
    """Reduce one reply's per-position gaps to its ``ReplyFlips``.

    ``n_served`` is the reply's served token count, ``at_cap`` whether that count
    reached the request's stamped cap. The reference returns one gap per served token
    plus, when it scored the reply's stop, one more; any other length is a malformed
    reference reply and raises ``ValueError``, since a per-position measurement cannot
    be read off gaps that do not line up with the positions."""
    if len(gaps) not in (n_served, n_served + 1):
        raise ValueError(
            f"event {event_key!r}: the reference returned {len(gaps)} gaps for "
            f"{n_served} served tokens; expected {n_served} (no stop row) or "
            f"{n_served + 1} (stop row scored)")
    flips = n_nan = n_inf = 0
    worst = 0.0
    severe: list[float] = []
    for g in gaps:
        if math.isnan(g):
            n_nan += 1
            continue
        if math.isinf(g):
            n_inf += 1
        if g > worst:
            worst = g
        if g > 0:
            flips += 1
            if g >= SEVERITY_FLOOR_NATS:
                severe.append(g)
    severe.sort()
    return ReplyFlips(event_key=event_key, n_tokens=len(gaps), n_flips=flips,
                      max_gap=worst, severe_gaps=tuple(severe), n_nan=n_nan, n_inf=n_inf,
                      stop_gap=gaps[-1] if len(gaps) == n_served + 1 else None,
                      at_cap=at_cap)


class PooledFlips:
    """Sum of ``ReplyFlips`` records over one measurement, read against a grid.

    Construct with the severity grid (ascending nats, e.g.
    ``Calibration.curve_side.levels_nats``), :meth:`add` each reply's record, then
    read the totals and :meth:`level_counts`.

    Attributes:
        levels:     the ascending severity grid, in nats.
        n_tokens:   positions counted over every added reply, non-finite included --
                    they are positions the server emitted and the reference scored.
        n_flips:    positions with a strictly positive, non-NaN gap.
        max_gap:    the largest non-NaN gap seen, in nats (0.0 when nothing was added).
        n_nan:      NaN positions.
        n_inf:      +/-inf positions.
        n_replies:  records added.
    """

    def __init__(self, levels: Sequence[float]) -> None:
        self.levels = list(levels)
        self.n_tokens = 0
        self.n_flips = 0
        self.max_gap = 0.0
        self.n_nan = 0
        self.n_inf = 0
        self.n_replies = 0
        self._counts = [0] * len(self.levels)

    def add(self, record: ReplyFlips) -> None:
        """Count one reply's record into the totals."""
        self.n_tokens += record.n_tokens
        self.n_flips += record.n_flips
        self.n_nan += record.n_nan
        self.n_inf += record.n_inf
        self.n_replies += 1
        if record.max_gap > self.max_gap:
            self.max_gap = record.max_gap
        for i, n in enumerate(record.level_counts(self.levels)):
            self._counts[i] += n

    def extend(self, records: Iterable[ReplyFlips]) -> None:
        """``add`` every record."""
        for r in records:
            self.add(r)

    def level_counts(self) -> list[int]:
        """Flips at or above each rung, parallel to ``levels`` (non-increasing)."""
        return list(self._counts)

    def top_level(self, worst_gap: float) -> int:
        """The index of the highest rung a reply with this worst gap reached, or
        ``-1`` when it stayed below the first. Because the rungs are nested, the
        reply is a site at every rung ``0..top_level``."""
        return bisect_right(self.levels, worst_gap) - 1
