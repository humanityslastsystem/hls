"""L0 -- correctness result types: Verdict, PerEventVerdict, MetricResult.

Verdict:         run-level correctness, built by verify_trace from each verifier's verify()
PerEventVerdict: per-event detail a metric produces for one rollout event (engine-plane)
MetricResult:    a metric's batch output -- the reduced value + per-event detail

These are the L0 types of the verification domain (``hls.verify``). The performance
score types (Score, SLOResult) live in ``hls.core.score``.

``passthrough_verdict`` is the assume-correct verdict a drive of the task's own reference
server is scored against: the baseline credit's drives.
"""

from __future__ import annotations
from typing import Any, Mapping
from pydantic import BaseModel, ConfigDict, Field


class Verdict(BaseModel):
    model_config = ConfigDict(frozen=True)

    passed: bool
    #: Per-event pass/fail. A per-verifier Verdict keys this by `request_id` (unique
    #: within one tenant's slice). The run-level aggregate Verdict and any pooled
    #: verifier's Verdict (graded across every tenant's events) key by
    #: `TraceEvent.event_key` (`"{tenant_id}:{request_id}"`) instead, since request_id
    #: alone is not unique across tenants; the same convention applies to
    #: `event_diagnostics` and `event_rationale` below. Empty on a verifier that grades
    #: the trace as a whole: it has no per-event opinion, and verify_trace reads `passed`
    #: for every event it covers.
    per_event: Mapping[str, bool] = Field(default_factory=dict)
    #: Why `passed` is False, set only on the run-level aggregate Verdict:
    #: "correctness" (a verifier gate failed) or "failed_requests" (every gate passed
    #: but some request errored, e.g. a driver timeout -- see
    #: `diagnostics["failed_requests"]` for the error census). None when passed, or on
    #: a per-verifier verdict, which never sees errored events.
    failure_mode: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    #: Per-event detail keyed like `per_event`, for the viewer's per-turn eval display.
    #: Content is metric-specific (e.g. {predicted, expected} for letter/contains).
    #: ReferenceRelativeVerifier additionally nests the oracle's per-event detail under
    #: a `reference` key, for the artifact-vs-oracle view.
    event_diagnostics: Mapping[str, dict] = Field(default_factory=dict)
    #: Per-event human-readable rationale, keyed like ``per_event`` (optional).
    event_rationale: Mapping[str, str] = Field(default_factory=dict)


def passthrough_verdict(trace) -> Verdict:
    """A correctness-passes Verdict for assume-correct scoring of *trace*.

    The baseline is the task's own reference serving system, so its outputs are trusted by
    construction and only the timing behavior (throughput, SLO) is measured. Trust covers
    content only, never transport: an errored request (fault, timeout -- ``response=None`` or
    ``error`` set) produced no output to trust, so it earns per-event False here exactly as
    the real verify path forces it; otherwise an overloaded baseline that sheds load by
    fast-failing would earn the most credit. ``passed`` stays True: a measurement must never
    gate a drive's Score to zero on errors (per-event credit is the only lever)."""
    return Verdict(
        passed=True,
        per_event={ev.event_key: ev.response is not None and ev.error is None for ev in trace.events},
    )


class PerEventVerdict(BaseModel):
    """Per-event detail produced by a metric for one rollout event.

    The per-event sibling of the run-level Verdict. `score` is the per-event
    magnitude (e.g. a turn's F1); None for a pass/fail metric."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    score: float | None = None
    rationale: str | None = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class MetricResult(BaseModel):
    """A metric's batch output: the reduced value (the metric owns the reduction)
    plus per-event detail keyed by request_id. The verifier gates/compares `value`
    and assembles the run-level Verdict from `per_event`."""

    model_config = ConfigDict(frozen=True)

    value: float
    per_event: Mapping[str, PerEventVerdict] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
