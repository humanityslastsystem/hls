"""obs record schema. Large blobs (Trace, full trajectory) live in their own
files; the record embeds small high-value fields and references the big ones by
relative path so rollout.json stays a readable manifest."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Literal, get_args

from pydantic import BaseModel, ConfigDict, Field

from hls.obs.conditions import HarnessConditions

#: How a rollout ended. "invalid_artifact" is a scored-zero run: the submission was
#: invalid or did not build, which is the contestant's fault. "built" is a build-only
#: rollout (``hls run --build-only``) whose submission built: it records no score until a
#: grading of it does. "error" is accepted on read and never written: the stage pipeline
#: records a failed instance as a scored zero.
RolloutStatus = Literal[
    "running", "ok", "built", "error", "invalid_artifact"
]

#: Every status a sealed record can carry. Readers asking "is this rollout finished?"
#: derive the answer from here instead of listing statuses, so a status added above
#: cannot leave one of them tailing a rollout that already ended.
TERMINAL_STATUSES: frozenset[str] = frozenset(get_args(RolloutStatus)) - {"running"}


def index_columns(spec: Mapping[str, Any] | None, arm_model: str | None) -> dict[str, Any]:
    """The columns a stage pipeline's index row reads from its record (``hls.stages.finalize``):
    the task model's fingerprint and the accelerator kind from the task's ``dump_spec``, and the
    model of the arm whose agent built the graded submission, None for one no agent built. There
    is no ``backend`` column: a checkpoint that is a SIF digest would leave it always null."""
    model = (spec or {}).get("model")
    hardware = (spec or {}).get("hardware")
    accelerator = hardware.get("accelerator") if isinstance(hardware, dict) else None
    return {"model_fingerprint": model.get("fingerprint") if isinstance(model, dict) else None,
            "accelerator": accelerator.get("kind") if isinstance(accelerator, dict) else None,
            "agent_implementer_model": arm_model}


class ExceptionInfo(BaseModel):
    """The exception a rollout's record may hold as the only account of its failure; the viewer
    shows it. The stage pipeline records none: a failed instance is a scored zero whose reason and
    evidence the score holds."""

    model_config = ConfigDict(extra="forbid")

    #: The exception's class name, e.g. ``RuntimeError``.
    type: str
    #: ``str(exc)``.
    message: str
    #: The formatted traceback.
    traceback: str
    #: A build failure's installer output, which tells a defect in the submission apart from a
    #: dependency that moved; None for any other exception.
    log: str | None = None


class RolloutProvenance(BaseModel):
    """What one instance ran, enough to know it and run it again, as finalize records it
    (``hls.stages.finalize``).

    The arena config is the serialized TaskSpec tree (``spec`` = ``dump_spec(task)``),
    the single source of truth. Model fingerprint, wire protocol, accelerator,
    workload, dataset split sizes, oracle identity, transport, and so on all live in
    ``spec`` and are read straight out of the dict (e.g.
    ``spec["model"]["fingerprint"]``), not hand-copied here. Provenance is
    write-only: nothing reconstructs the typed tree.

    Only the facts that are not the spec's are fields of their own: the split, the image
    digests, the commits, the override map, the artifact served and the reasoning effort. A key
    it does not hold is refused: finalize is its one writer, and the viewer reads published
    records as JSON, never through this model."""

    model_config = ConfigDict(extra="forbid")

    #: The task's id, also the spec's.
    task_id: str
    #: The dataset split the drive evaluated, which finalize sets (``test`` for a build-only
    #: rollout, which drove nothing).
    split: str = "test"
    #: ``dump_spec`` of the task as the roles rebuilt it, calibration and baseline credit included.
    spec: dict[str, Any] = Field(default_factory=dict)
    #: The SHA-256 of every image the instance's stages launched, by role.
    image_digests: dict[str, str] = Field(default_factory=dict)
    #: The hls commit the instance ran.
    hls_commit: str | None = None
    #: The harbor submodule's commit the instance ran.
    harbor_commit: str | None = None
    #: The task's overrides, by dotted ``task.*`` path, as ``--task.*`` gave them.
    rollout_overrides: dict[str, Any] = Field(default_factory=dict)
    #: The artifact the drive served, dumped whole with its ``kind``; empty for a build-only
    #: rollout, which drove nothing.
    artifact: dict[str, Any] = Field(default_factory=dict)
    #: The provider reasoning-effort level the build launched its harness at: the effort of the
    #: arm record that built the submission (``hls.arms.resolve_arm``, the arm's own or the task
    #: producer's); None for a submission no agent built or an unset effort.
    reasoning_effort: str | None = None


class RolloutBuild(BaseModel):
    """A rollout's build as the stage pipeline records it (``hls.stages.finalize``): the
    agent_build stage's ``build.json`` as the stage wrote it, the store key of each of its outputs,
    and the package stage's ``package.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``build.json`` as written: the arm, the budgets, how the agent's part ended, the commit
    #: point's record and tries.
    build: dict[str, Any]
    #: Each output of the agent_build manifest, by name (``source_bundle`` or ``no_submission``,
    #: ``transcript``, ``brief``, ``brief_provenance``, ``agent_env``, ``build``), to its store key.
    outputs: dict[str, str]
    #: ``package.json`` as written; None for a rollout whose package stage did not publish.
    package: dict[str, Any] | None
    #: When the agent's clock started (POSIX time, the runner's), from ``build.json``; None when it
    #: records none.
    started_at: float | None
    #: When the agent's part ended; None when ``build.json`` records none.
    ended_at: float | None
    #: The agent's usage as harbor reported it (tokens and cost), from ``build.json``'s
    #: extraction; None when it records none.
    usage: dict[str, Any] | None
    #: The tree digest of the folder the task uploaded into the agent's workspace
    #: (``hls.agent.uploads.tree_digest``), from ``build.json``; None when it records none.
    upload_sha256: str | None

    @property
    def cost_usd(self) -> float | None:
        """The agent's cost in USD as harbor reported it in ``usage``; None without a finite one."""
        return finite_number((self.usage or {}).get("cost_usd"))


def finite_number(value: Any) -> float | None:
    """*value* as a float when it is a finite int or float and not a bool, else None: what a number
    a harness file records may be."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


class EvalEpisode(BaseModel):
    """One eval of the artifact, carried by the GradingRecord that records it.
    `triggered_by` leaves room for agent-requested intermediate evals beside the
    end-of-rollout "terminal" one."""

    model_config = ConfigDict(extra="forbid")

    #: The episode's id; ``terminal`` is the end-of-rollout eval.
    eval_id: str
    #: who started it: "harness" (a run's own eval), "agent" (reserved, not yet
    #: emitted), or "regrade" (a later grading of an existing artifact).
    triggered_by: Literal["harness", "agent", "regrade"] = "harness"
    #: The digest of the artifact image the drive served; empty for an artifact that never ran.
    checkpoint: str
    #: When the drive stage started, or a failed instance's failed stage (POSIX time).
    started_at: float
    #: When the score stage ended, or a failed instance's failed stage; None when it recorded no end.
    finished_at: float | None = None
    #: The store key of the trace's directory; empty when the drive did not serve.
    trace_path: str
    #: ``gate_passed``, ``aggregate_passed`` and the verifiers' ``diagnostics``.
    verdict_summary: dict[str, Any] = Field(default_factory=dict)
    #: The serialized Score.
    score: dict[str, Any] = Field(default_factory=dict)
    #: The store key of each role log of the drive, verify and score attempts, by file name.
    container_logs: dict[str, str] = Field(default_factory=dict)


class GradingRecord(BaseModel):
    """One grading of one artifact, written to ``evals/<id>/grading.json``.

    A rollout's artifact can be graded many times: to correct a scoring bug, to apply
    an updated scorer, or to average several gradings. Each grading records the
    provenance that produced it, so two gradings that disagree can be told apart --
    a different ``hls_commit`` means the code changed, a different ``provenance.spec``
    means the scorer or its calibration did."""

    model_config = ConfigDict(extra="forbid")

    #: The graded episode: its score, verdict, logs and trace.
    episode: EvalEpisode
    #: What graded it, not what built it.
    provenance: RolloutProvenance
    #: The rollout id whose artifact was graded.
    source_rollout: str
    #: "original" for the grading a run performs on its own artifact; "adopt" when a
    #: committed image was re-deployed; "rebuild" when the image was rebuilt from the
    #: submitted source.
    mode: Literal["original", "adopt", "rebuild"] = "original"
    #: How the grading ran.
    conditions: HarnessConditions
    #: ``package.json`` as written by the package stage of a grading that packaged the rollout's
    #: source again (``--artifact.rebuild``); None for a grading of a package it did not build.
    package: dict[str, Any] | None = None


class RolloutRecord(BaseModel):
    """The manifest of one rollout, ``rollout.json``, which finalize writes: small fields inline,
    big blobs by store key."""

    model_config = ConfigDict(extra="forbid")

    #: The rollout's id (``hls.store.layout.new_rollout_id``).
    rollout_id: str
    #: The task's id.
    task_id: str
    #: When the rollout's plan was made (POSIX time).
    created_at: float
    #: When finalize sealed the record, or a failed instance's failed stage ended.
    finished_at: float | None = None
    #: How the rollout ended.
    status: RolloutStatus = "running"
    #: What the rollout ran.
    config: RolloutProvenance
    #: The build phase as the stage pipeline records it (``RolloutBuild``); None for an artifact no
    #: agent built.
    build: RolloutBuild | None = None
    #: How the rollout ran.
    conditions: HarnessConditions
    #: The failure of a rollout whose record is the only account of it, which the viewer shows;
    #: the stage pipeline records a failed instance as a scored zero instead.
    error: ExceptionInfo | None = None
