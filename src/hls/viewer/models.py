"""Pydantic response models for the HLS viewer API.

These are API-facing DTOs: they contain no hls.* imports and no business logic.
"""

from __future__ import annotations

from typing import Annotated, Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class PaginatedResponse(BaseModel, Generic[T]):
    """Generic paginated list wrapper."""

    items: list[T]
    total: int
    page: int
    page_size: int
    total_pages: int


class FileInfo(BaseModel):
    """Information about a single file or directory entry inside a rollout dir."""

    path: str  # relative path from the rollout dir root
    name: str  # file/dir name
    is_dir: bool
    size: int | None = None  # bytes; None for directories


class RolloutSummary(BaseModel):
    """Summary row for the rollouts list.

    Required fields: task_id, rollout_id.
    All others are optional/best-effort, filled from the index + full record.
    ``extra="allow"`` lets the JSON scanner pass arbitrary enrichment fields
    through without schema churn.
    """

    model_config = ConfigDict(extra="allow")

    task_id: str
    rollout_id: str
    created_at: float | None = None
    finished_at: float | None = None
    status: str | None = None
    score: float | None = None
    gate_passed: bool | None = None
    cost_usd: float | None = None
    image_digests: dict[str, str] = Field(default_factory=dict)
    # Enriched columns (best-effort; None when the source record can't be read)
    agent_implementer_model: str | None = None
    model_fingerprint: str | None = None
    accelerator: str | None = None
    backend: str | None = None
    phase: str | None = None
    #: The layout the row came from: "store" for a store instance, None for a published rollout.
    layout: str | None = None
    #: The arm that built a store rollout.
    arm: str | None = None
    #: The stage a failed store instance ended on.
    failed_stage: str | None = None
    #: The reason of the fault that ended a failed store instance.
    fault: str | None = None
    #: How many files a store instance's secret scan quarantined.
    quarantined_count: int | None = None


class _VerifierSummaryBase(BaseModel):
    """Identity and outcome shared by every per-verifier summary. The concrete
    per-kind subclasses below carry only the diagnostics their verifier actually
    emits; ``VerifierSummary`` is their union, discriminated on ``kind``."""

    name: str  # class name (part after colon in the diagnostics key)
    gate_passed: bool | None = None


class TokenLevelSummary(_VerifierSummaryBase):
    """TokenFaithfulnessVerifier's gate diagnostics (rate rule + flip-count curve)."""

    kind: Literal["token_level"] = "token_level"
    flip_rate: float | None = None  # fraction of teacher-forced tokens with gap > 0
    rate_ratio: float | None = None  # flip_rate / (rate_multiplier * calibration.q_max)
    # Gap levels (nats) whose flip-count ceiling the trace exceeded, derived from the
    # verdict's per-level rows; [] = curve checked and clean, None = the record carries
    # no curve rows.
    curve_fired_levels: list[float] | None = None
    n_tokens: int | None = None  # total teacher-forced tokens pooled across the trace


class TaskRelativeSummary(_VerifierSummaryBase):
    """ReferenceRelativeVerifier's artifact-vs-reference metric diagnostics."""

    kind: Literal["task_relative"] = "task_relative"
    artifact_value: float | None = None
    reference_value: float | None = None
    tol: float | None = None


class TaskAbsoluteSummary(_VerifierSummaryBase):
    """MetricVerifier's absolute-threshold metric diagnostics."""

    kind: Literal["task_absolute"] = "task_absolute"
    artifact_value: float | None = None
    min_value: float | None = None


class GenericVerifierSummary(_VerifierSummaryBase):
    """Any verifier without a dedicated summary shape: identity and outcome only."""

    kind: Literal["generic"] = "generic"


VerifierSummary = Annotated[
    TokenLevelSummary | TaskRelativeSummary | TaskAbsoluteSummary | GenericVerifierSummary,
    Field(discriminator="kind"),
]


class TraceSummary(BaseModel):
    """Aggregate statistics for one eval's trace.json (windowed summary card)."""

    total: int
    token_match: int | None = None  # count of per_event True values
    token_match_total: int | None = None  # len(per_event)
    correct: int | None = None  # exact-match vs ground_truth
    graded: int | None = None  # events with non-empty ground_truth
    p99_ms: float | None = None  # latency p99 in milliseconds
    # Verifier-kind fields -- populated from verdict.json diagnostics
    verifier_kind: str | None = None  # "token_level" | "task_relative" | "task_absolute" | "generic"
    gate_passed: bool | None = None
    artifact_value: float | None = None
    reference_value: float | None = None
    min_value: float | None = None
    tol: float | None = None
    verifiers: list[VerifierSummary] = Field(default_factory=list)


class TracePage(BaseModel):
    """Windowed page of per-turn trace rows, with optional filter applied."""

    offset: int
    limit: int
    total: int  # total after filtering, before slicing
    rows: list[dict[str, Any]]


class TimelineSLO(BaseModel):
    """Per-tenant SLO target, best-effort from config (None when unparseable)."""

    kind: str  # "streaming" | "latency"
    ttft_ms: float | None = None
    tpot_ms: float | None = None
    target_ms: float | None = None


class TimelineTenant(BaseModel):
    """One tenant lane-group: id (from the trace), best-effort name + SLO target."""

    id: str
    name: str | None = None
    req_n: int
    slo: TimelineSLO | None = None


class TimelinePayload(BaseModel):
    """Compact per-request timeline for one eval. Times are seconds relative to t0.

    ``requests`` are plain dicts (like ``TracePage.rows``) to keep 30k-row
    construction fast; each has keys:
    ``ti`` (index into ``tenants``), ``sid`` (session id), ``turn``,
    ``send``, ``recv``, ``plan``, ``ttft``, ``tlast`` (all seconds from t0;
    recv/plan/ttft/tlast may be null), ``ntok`` (output token count or null),
    ``err`` (bool). ``tlast`` is the last streamed packet's timestamp (the same
    instant StreamingLatencySLO._meets uses for TPOT) -- distinct from ``recv``,
    which is stamped after the SSE stream's trailing `data: [DONE]` and so runs
    strictly >= the last packet.
    """

    t0: float
    duration: float
    tenants: list[TimelineTenant]
    requests: list[dict[str, Any]]
