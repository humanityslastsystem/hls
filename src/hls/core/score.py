"""L0 -- performance score types: SLOResult, Score.

SLOResult: output of SLO.apply(trace, workload) -- two-tier (gate + credit)
Score:     final number produced by Scorer.score()

The correctness result types (Verdict, PerEventVerdict, MetricResult) live in
``hls.core.verdict``.
"""

from __future__ import annotations
from collections.abc import Collection
from typing import Any, Mapping
from pydantic import BaseModel, ConfigDict, Field


class SLOResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    aggregate_passed: bool
    #: Per-request throughput credit in [0, 1] (higher is better), keyed by request id
    #: within the tenant's slice. ``hls.scoring.Scorer`` multiplies this into its unit count.
    per_event_credit: Mapping[str, float] = Field(default_factory=dict)
    #: SLO-normalized latency ``r`` (lower is better; 1.0 = exactly at the SLO), keyed by
    #: the unit the SLO's ``grain`` names: request id at request grain, session id at
    #: session grain. ``SloLatencyScorer`` means log(1/r) over these. Empty when the SLO
    #: defines no such ratio (``NoSLO``); otherwise a missing key means ``r`` could not
    #: be computed (no response), and ``aggregate_passed`` is False in that case. A
    #: scorer reporting a bound on such a run prices those units with
    #: ``SLO.censored_ratio``.
    per_event_ratio: Mapping[str, float] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class Score(BaseModel):
    model_config = ConfigDict(frozen=True)

    #: ``achieved / sol``. Higher is better on both scorers, and the scale differs: the
    #: goodput scorer reports achieved throughput against a roofline, in [0, 1];
    #: ``SloLatencyScorer`` reports a speedup over the baseline serving stack, where 1.0
    #: is the baseline and there is no ceiling. Positive for every valid run. 0.0 means
    #: the run was gated, and ``gate_passed``/``zero_reasons`` say why.
    value: float
    #: What this artifact measured, before normalization: throughput in ``unit``/sec for
    #: the goodput scorer, the geometric mean of 1/r for ``SloLatencyScorer``.
    achieved: float = 0.0
    #: What ``achieved`` is normalized against: the roofline for the goodput scorer, the
    #: baseline stack's own geometric mean of 1/r for ``SloLatencyScorer``.
    sol: float = 0.0
    unit: str = ""
    gate_passed: bool = True
    #: Every gate layer the run failed, in the order the scorer checked them; empty on a
    #: run that was not gated. Always a collection, never a single reason or None: a run
    #: can be wrong *and* miss its SLO, and reporting one of those hides the other.
    zero_reasons: tuple[str, ...] = ()
    breakdown: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def zero(cls, *, reasons: Collection[str], diagnostics: dict[str, Any] | None = None,
             breakdown: dict[str, Any] | None = None) -> "Score":
        """A gated zero naming every layer that failed. `breakdown` carries the named
        sections a gated run still reports (each gate's diagnostics, the scorer's upper
        bound) and has the same shape as a passing run's, so the viewer can always read
        score.breakdown.slo regardless of outcome.

        `diagnostics` is the shorthand for a zero with a single reason and one section to
        report: it nests under the section that reason belongs to ("correctness" and
        "failed_requests" -> "verdict", "slo" -> "slo", anything else under its own name).
        A zero with several reasons names its sections in `breakdown` itself.

        `reasons` is a collection of reason NAMES; a bare string is refused, since a
        string collects its own characters."""
        if isinstance(reasons, str):
            # A string satisfies `Collection[str]` as a collection of its characters, so
            # `reasons="slo"` would file three one-character reasons into score.json and
            # every index row read off it. Never what a caller means.
            raise TypeError(
                f"reasons={reasons!r} is a string, which is a collection of its "
                f"characters: this would record {len(reasons)} one-character reasons. "
                f"Pass a collection of reason names, e.g. ({reasons!r},).")
        entries = dict(breakdown or {})
        if diagnostics is not None:
            if len(reasons) != 1:
                raise ValueError(
                    f"`diagnostics` names no section with {len(reasons)} reasons "
                    f"({tuple(reasons)}); pass the sections in `breakdown` instead")
            reason = next(iter(reasons))
            entries[{"correctness": "verdict", "failed_requests": "verdict",
                     "slo": "slo"}.get(reason, reason)] = diagnostics
        return cls(
            value=0.0,
            achieved=0.0,
            sol=0.0,
            unit="",
            gate_passed=False,
            zero_reasons=tuple(reasons),
            breakdown=entries,
        )
