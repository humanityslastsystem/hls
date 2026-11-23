"""Artifact: the contestant, represented as pure config.

An ``Artifact`` is a declarative, reproducible ``Poly`` node serialized whole into
provenance at ``config.artifact`` via ``dump_spec``, overridable by ``--artifact.*``.
It holds no runtime: no ``produce()``, no ``config()`` projection, no secrets or
per-machine host paths (the producer's ``upload_dir``/``skills_dir`` folders are
relative to the task module's directory, not the machine). The agent_build stage runs
the agent this config describes, and the package stage builds the repository it commits
into the image a grading serves.

Its one kind is ``AgentProducedArtifact``: a submission an implementer agent builds.
"""

from __future__ import annotations
import logging
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from hls.core.poly import Poly, Spec
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import AccessList, NetworkPolicy
from hls.prompt import ImplementerBrief
from hls.store.digest import Sha256Hex

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Producer config: declarative knobs of the build engine, no runtime or secrets
# ---------------------------------------------------------------------------

class Budget(Spec):
    """A resource budget over wall-clock time and turns.

    Used both as the hard budget (harness-enforced, invisible to the agent) and the
    soft budget (surfaced to the agent via the `budget` MCP server). Each dimension is
    independently optional; None means no bound on that dimension. A budget has no cost
    bound: no runner keeps one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Wall-clock seconds since the agent started.
    wallclock_sec: int | None = None
    #: Agent turns (assistant messages).
    turns: int | None = None


def _default_producer_network() -> NetworkPolicy:
    """Default producer egress: sealed. Raw internet is air-gapped (empty domains
    allow-list); the agent_build stage opens the arm's model route host in the sandbox's
    seal (``hls.stages.agent_build.seal_policy``), so the agent still runs. The package index
    stays reachable through the build seal's proxy, which refuses the harness deny
    list (``hls.egress.denied_packages``). A task that passes its own ``network=``
    must deny that list too (see tasks/multitenant_slo).
    """
    return NetworkPolicy(
        domains=AccessList(mode="allow"),
        packages=AccessList(mode="deny", entries=list(DENIED_PACKAGES)))


class HighestAvailable(Spec):
    """The highest reasoning-effort level the provider offers for the harness and
    model a run launches with, resolved when an arm with no effort of its own is
    planned (``hls.arms.resolve_effort``) and recorded in the arm's record as the
    provider level it became. TOML: ``reasoning_effort = { kind = "highest_available" }``."""

    kind: Literal["highest_available"] = "highest_available"


class ProviderLevel(Spec):
    """An explicit provider level, handed to the harness as written (claude-code
    ``--effort <level>``, codex ``model_reasoning_effort=<level>``); the harness
    validates it against its own choices. TOML:
    ``reasoning_effort = { kind = "provider", level = "xhigh" }``."""

    kind: Literal["provider"] = "provider"
    level: str


#: The two forms a producer's reasoning effort takes, told apart by ``kind``.
ReasoningEffort = Annotated[HighestAvailable | ProviderLevel, Field(discriminator="kind")]


class EvaluationAllocation(Spec):
    """What a grading of the task's artifact holds (``hls grade``, ``hls run``), in place of
    the brief's hardware: the artifact's CPUs, memory and writable disk, with the task's
    GPUs, the load driver's memory and CPUs beside them, and
    the time and memory its verify stage may take. The planner adds the runner's reserve and the extracted
    images on top. None of it is the task's workload: no fingerprint reads it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: CPUs of the artifact's server.
    cpus: int = Field(gt=0)
    #: Memory of the job, in GiB, where the server shares it with the harness: the server gets
    #: it less the runner's reserve (``server_memory_gib``).
    #: None when ``step_memory_gib`` is set.
    job_memory_gib: int | None = Field(default=None, gt=0)
    #: Memory of the server's own Slurm step, in GiB, where it has one apart from the
    #: harness (deepresearch): the server gets it whole. None when ``job_memory_gib`` is set.
    step_memory_gib: int | None = Field(default=None, gt=0)
    #: GiB the artifact may write to node-local disk: its workdir and engine cache together.
    writable_gib: int = Field(gt=0)
    #: Memory of the driver, in GiB; a tool bundle that loads more raises it.
    driver_memory_gib: int = Field(gt=0)
    #: CPUs of the driver's own; None runs the driver on the artifact's CPUs.
    driver_cpus: int | None = Field(default=None, gt=0)
    #: True oversubscribes the driver's memory on the server's: the server gets the job's memory
    #: less the runner's reserve, the driver keeps its own step limit (``driver_memory_gib``), and
    #: the stage asks for no more than the job. Only a driver on the artifact's CPUs
    #: (``driver_cpus`` None) in the server's job may. Left out of the dump when false.
    driver_memory_shared: bool = Field(default=False, exclude_if=lambda shared: not shared)
    #: Seconds the verify role may take after its stage's staging, which the planner adds for the
    #: reference image's extraction and the copies of its data (``hls.pipeline.pipelines.staging_sec``):
    #: the reference engines' load and every verifier over the drive's replies. None takes the
    #: planner's two hours (``REFERENCE_SHAPE``). A task declares it when two hours are less than
    #: twice its verify time: twice that time, rounded up to the half hour.
    verify_time_sec: int | None = Field(default=None, gt=0)
    #: Memory of the verify role, in GiB, beside which the planner adds the runner's reserve: what
    #: the reference engines hold on the host while they grade the drive's replies. None takes
    #: the planner's (``REFERENCE_SHAPE``'s less the runner's reserve). The planner gives the role
    #: more when the engines ask for more or the task's verify peak at its trace cap is more
    #: (``hls.pipeline.pipelines.reference_shape``). A task declares it when its reference is far
    #: smaller than the planner's figure, so that a grading fits a smaller machine.
    verify_memory_gib: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _one_memory(self) -> EvaluationAllocation:
        if (self.job_memory_gib is None) == (self.step_memory_gib is None):
            raise ValueError("an evaluation allocation names exactly one of job_memory_gib and step_memory_gib: the "
                             "job the server shares with the harness, or the server's own step")
        if self.driver_memory_shared and (self.driver_cpus is not None or self.job_memory_gib is None):
            raise ValueError("only a driver on the server's CPUs in the server's job (driver_cpus None, "
                             "job_memory_gib set) shares the server's memory")
        return self

    def server_memory_gib(self, runner_memory_gib: int) -> int:
        """The artifact's server's memory, in GiB, when the runner keeps *runner_memory_gib*
        (``run.runner_memory_gib``): the job's less that reserve, or the server's
        own step whole. Raise ValueError when the reserve leaves the
        server none."""
        if self.step_memory_gib is not None:
            return self.step_memory_gib
        assert self.job_memory_gib is not None
        memory = self.job_memory_gib - runner_memory_gib
        if memory < 1:
            raise ValueError(f"the job's {self.job_memory_gib} GiB leaves the server no memory beside the "
                             f"runner's {runner_memory_gib} GiB")
        return memory


class BuildAllocation(Spec):
    """The allocation a build of the task (``hls run``) is sized to equal, in place of the
    brief's hardware. Its GPUs are the task's own (``TaskSpec.hardware``). The planner
    gives the build's sandbox every CPU of it, its
    memory less the runner's and a harness role's, and as much writable space as it has memory
    (``hls.pipeline.pipelines.sandbox_shape``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: CPUs of the allocation.
    cpus: int = Field(ge=1)
    #: Memory of the allocation, in GiB.
    memory_gib: int = Field(ge=1)


class AgentProducerSpec(Spec):
    """Declarative knobs of the agent build of a task's submission.

    The arm names the harness and the model (``hls.arms``), and the submission is always a
    git repository the package stage builds; runtime and secret values are the agent_build
    stage's, never the task's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # The effort an arm that sets none of its own launches at. Every agentic task authors
    # one, because harness defaults are neither uniform nor visible: claude-code emits no
    # --effort flag when unset and silently takes the CLI's built-in level, while codex
    # defaults to "high" -- leaving it implicit would let two arms differ in effort while
    # looking like a clean model comparison.
    #
    # Typed rather than a bare level name: ``HighestAvailable`` asks for the provider's
    # top level and is resolved through the arm's harness adapter's ``top_effort``
    # (``hls.arms.resolve_effort``), while a ``ProviderLevel`` is passed through as written
    # and checked against the levels the harness accepts (the levels are harness- and
    # model-specific).
    reasoning_effort: ReasoningEffort

    # Task-provided folders uploaded into the build sandbox verbatim (e.g. CUDA docs,
    # reference PDFs), merged at ``upload_dest``: each absolute or relative to the task
    # module's directory, resolved and copied there by the agent_build stage when the build
    # starts. No path may be held by two of them. Setting this requires ``upload_brief`` --
    # there is no default explanation text.
    upload_dir: tuple[str, ...] | None = Field(default=None, min_length=1)
    #: The tree digest of the view ``upload_dir`` makes merged (``hls.agent.uploads.merged_digest``:
    #: every file's path, permission bits and sha256; for one folder its ``tree_digest``), which
    #: the build checks before it copies the folders and records in ``build.json``. Required with
    #: ``upload_dir``: a folder may be no part of the commit (too large to commit), so this is
    #: what names the material a contestant gets.
    upload_sha256: Sha256Hex | None = None
    # Where ``upload_dir`` lands inside the sandbox. Defaults to the neutral
    # ``/task/files`` staging path; a task sets this when the files should live in
    # the agent's working tree instead (e.g. ``/workspace/docs``). The brief's
    # uploads section reflects whatever this resolves to, so text and upload stay
    # in sync.
    upload_dest: str | None = None
    # The task's explanation of the uploaded files, rendered verbatim as its own
    # brief section after Build Instructions (including the heading, if any -- the
    # harness only appends where the files live). Required with ``upload_dir``,
    # rejected without it.
    upload_brief: str | None = None

    # Task-provided folder of agent skills: either a single skill dir (contains
    # SKILL.md) or a root whose child dirs each contain SKILL.md (harbor's
    # resolve_skills semantics). Absolute or relative to the task module's
    # directory; each harness copies it into its own native skills location.
    skills_dir: str | None = None

    # Enforced by the harness; the agent never sees it. wallclock_sec is required --
    # a build must always have a wall-clock bound. turns maps to --max-turns; breaching
    # either interrupts the build and collects gracefully.
    hard_budget: Budget

    # Surfaced to the agent via the `budget` MCP server so it can pace itself. None
    # means no budget server is registered and no brief section is added. Independent
    # of hard_budget.
    soft_budget: Budget | None = None

    # What a grading of the artifact holds (hls grade, hls run). None serves the artifact on
    # the brief's hardware.
    evaluation_allocation: EvaluationAllocation | None = None

    # What a build of the artifact holds (hls run). None builds on the brief's hardware.
    build_allocation: BuildAllocation | None = None

    @model_validator(mode="after")
    def _check_budgets(self) -> "AgentProducerSpec":
        if self.hard_budget.wallclock_sec is None:
            raise ValueError("hard_budget.wallclock_sec is required (a build must be time-bounded)")
        if self.hard_budget.wallclock_sec <= 0:
            raise ValueError("hard_budget.wallclock_sec must be > 0 (a build must be time-bounded)")
        if self.soft_budget is not None:
            for dim in ("wallclock_sec", "turns"):
                soft, hard = getattr(self.soft_budget, dim), getattr(self.hard_budget, dim)
                if soft is not None and hard is not None and soft >= hard:
                    logger.warning("soft_budget.%s (%s) >= hard_budget.%s (%s); the soft "
                                   "gauge will never bind before the hard stop", dim, soft, dim, hard)
        return self

    @model_validator(mode="after")
    def _check_uploads(self) -> "AgentProducerSpec":
        # Caught at spec construction so a mismatch never costs a sandbox launch.
        if self.upload_dir is not None and not (self.upload_brief or "").strip():
            raise ValueError(
                "upload_brief is required when upload_dir is set: the brief's "
                "uploaded-files section has no default -- the task must "
                "explain the uploaded files to the agent")
        if self.upload_brief is not None and self.upload_dir is None:
            raise ValueError(
                "upload_brief is set but upload_dir is not -- the brief section "
                "describes an uploaded folder, so it cannot appear without one")
        if self.upload_dir is not None and self.upload_sha256 is None:
            raise ValueError(
                "upload_sha256 is required when upload_dir is set: the folder's tree digest "
                "(hls.agent.uploads.tree_digest) names the material the contestant is handed")
        if self.upload_sha256 is not None and self.upload_dir is None:
            raise ValueError("upload_sha256 is set but upload_dir is not: there is no folder it is the digest of")
        return self

    #: The submission's server file, relative to the root of the repository the agent commits.
    server_rel_path: str = "server.py"
    # Egress policy for the build sandbox. Defaults to sealed (see
    # _default_producer_network); a task layers its own policy by passing an explicit
    # ``network=`` (see tasks/multitenant_slo). Setting network=None here means fully
    # open egress -- the opposite of TaskSpec.network, where None means air-gapped at
    # runtime.
    network: NetworkPolicy | None = Field(default_factory=_default_producer_network)
    # Run the task's pinned reference beside the build sandbox, asleep on the same GPUs, so
    # the agent can have token sequences scored over a unix socket
    # (hls.reference.service_harness). A plan refuses a task with no baseline image or no
    # weights for it; a service that then does not start, load or answer is recorded and the
    # build goes on without it (hls.stages.reference_service). Left out of the dump when
    # off, so producers declared before the field keep their records.
    reference_service: bool = Field(default=False, exclude_if=lambda on: not on)


# ---------------------------------------------------------------------------
# Artifact (the contestant) -- pure config Poly and its kind
# ---------------------------------------------------------------------------

class Artifact(Poly):
    """The contestant: pure config, serialized to provenance. No produce(), no config().

    ``kind`` is the concrete class name (via ``Poly``)."""

    #: Extra launch args the submission's server is started with; host and port derive from the
    #: task's contract.
    serve_command: list[str] = Field(default_factory=list)


class AgentProducedArtifact(Artifact):
    """A submission an implementer agent builds, and the brief the agent is given."""

    #: How the build runs: its budgets, uploads, network policy and effort.
    producer: AgentProducerSpec
    #: The brief the agent_build stage renders with every section it states: the default
    #: brief, or a task's child template of it.
    brief: ImplementerBrief = Field(default_factory=ImplementerBrief)
