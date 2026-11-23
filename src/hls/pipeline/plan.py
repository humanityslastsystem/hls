"""What the orchestrator and the runners exchange through the store: the plan, its
revisions, and the status records the runners write.

The plan is a directory of revisions, each a whole plan in its own create-only file. Only
the orchestrator holding the instance's lease writes revisions, and each revision records
that lease's generation. To write, the holder confirms its lease, reads the latest revision
and creates the next one. A holder whose lease was taken over gets LockLost and must stop:
from the lease check, from a newer generation in the latest revision, or from a second lease
check after it lost the create. That is the fencing a lease alone does not give: a stale
orchestrator cannot overwrite what the new one wrote. PlanConflict is for a change under
the same lease generation, which the caller handles by reading the plan again.

The runner of each allocation writes one status file per submission, and one per stage
attempt; each file has exactly one writer."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from hls.measured import MeasuredFiles
from hls.pipeline.checkout import CodeState
from hls.settings import RetryPolicy, Settings, StageName
from hls.spec.resources import GrantedResources, HardwareRecord, ResourceShape
from hls.store.base import AlreadyExists, NotFound, Store, StoreError
from hls.store.digest import Sha256Hex
from hls.store.layout import InstanceLayout
from hls.store.lock import LockHandle, LockLost
from hls.store.manifest import ManifestKey, canonical_json
from hls.substrate.base import FaultRecord
from hls.substrate.factory import Placement
from hls.substrate.versions import ToolVersions

_REVISION_FILE = re.compile(r"rev-(\d{8})\.json")


class Toolchain(BaseModel):
    """Where the allocation finds its programs, as resolved on the login node."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Interpreter of the checkout's environment; the batch script runs the runner with it.
    python: Path
    #: PATH for the batch script and role steps: the Slurm, git and nvidia-smi directories,
    #: then the system's.
    path: str
    #: HOME for the batch script.
    home: Path


class StagePlan(BaseModel):
    """One stage as planned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Stage name, unique in the instance.
    name: StageName
    #: Registry name of the stage's implementation.
    kind: str
    #: The declared shape, after any ``[resources]`` override.
    shape: ResourceShape
    #: True when ``[resources.<task>.<stage>]`` replaced the task's declaration.
    shape_overridden: bool
    #: Parameters for the implementation, already checked against its parameter model.
    params: dict[str, Any]
    #: Earlier stages of this instance whose manifests the stage reads.
    after: list[StageName]
    #: Manifests of other instances the stage reads, pinned by key and digest when planned.
    inputs: dict[ManifestKey, Sha256Hex]
    #: The stage's retry policy, with its count resolved.
    retry: RetryPolicy
    #: Secrets granted to the stage (grantee ``stage.<name>`` in the secrets file), by the
    #: variable each is delivered as, to the secret's name. Names only: the runner reads the
    #: values when the stage starts.
    grants: dict[str, str] = Field(default_factory=dict)


class Submission(BaseModel):
    """One sbatch of an allocation. Written as an intent before sbatch runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Number within the allocation, from 1.
    n: int = Field(ge=1)
    #: The job's unique ``--comment``.
    tag: str
    #: When the intent was recorded.
    intended_at: float
    #: The job id it waits for (afterok), as submitted or as last rewired; None for none.
    after_job: str | None
    #: "intended" before sbatch has answered, "submitted" once the job id is known,
    #: "absent" when no job exists for it, "cancel_intended" and "cancelled" around a cancel.
    status: Literal["intended", "submitted", "absent", "cancel_intended", "cancelled"]
    #: The job id once known.
    job_id: str | None = None
    #: The command line the scheduler was given for it (``Scheduler.submit_argv``), every value of
    #: the site's secrets file replaced; None until it ran.
    argv: list[str] | None = None
    #: When the job id became known.
    submitted_at: float | None = None
    #: While the job id is unknown: when a tag lookup first answered that no job carries the
    #: tag. The orchestrator counts the submission as absent only once that answer is a full
    #: absence window old, however long sbatch itself took.
    missing_since: float | None = None
    #: A rewire to wait for this job id instead is intended and not yet confirmed.
    rewire_to: str | None = None
    #: Rewires to ``rewire_to`` that scontrol refused for a reason other than the dependency;
    #: the orchestrator gives the job up after a few.
    rewire_refusals: int = Field(default=0, ge=0)
    #: Other jobs found under this submission's tag. Each is cancelled until it is seen ended.
    strays: list[str] = Field(default_factory=list)
    #: The last scheduler error about this submission, for people.
    error: str | None = None

    @property
    def live(self) -> bool:
        """Whether this is the allocation's current submission."""
        return self.status in ("intended", "submitted")


class Failure(BaseModel):
    """An allocation that ended without publishing all its stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: When the orchestrator saw it.
    at: float
    #: The first stage it did not publish.
    stage: StageName
    #: Why.
    fault: FaultRecord


class RecordsRefused(Exception):
    """A failed instance's records cannot be written (the orchestrator's ``on_failed`` writer,
    ``hls.stages.finalize.failed_records``). The writer has replaced every value of the secrets
    file in the message, so it may be recorded. It lives here, below both, because a stage
    module may not import the orchestrator, which imports the planner and so every stage."""


class Allocation(BaseModel):
    """One Slurm job's worth of stages."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Index in the plan, from 0.
    index: int = Field(ge=0)
    #: Stages it runs, in order.
    stages: list[StageName]
    #: What sbatch asks for, after the site's rules.
    request: ResourceShape
    #: Where the substrate places it beside its request.
    placement: Placement
    #: The allocation whose success this one waits for; None for none.
    depends_on: int | None
    #: Allocations every stage of which must be published before this one is submitted. It is
    #: submitted with no Slurm dependency on them, so it can wait for several (a fan-in), and for
    #: one whose job Slurm has already forgotten.
    waits_for: list[int] = Field(default_factory=list)
    #: Submitted with Slurm's ``singleton`` dependency, so it starts only once no other job of the
    #: user with the plan's job name runs: cache warmers run one at a time on a cluster.
    singleton: bool = False
    #: The allocation whose failed stage this one retries; None for a planned allocation.
    retry_of: int | None = None
    #: Every sbatch made for it, oldest first.
    submissions: list[Submission] = Field(default_factory=list)
    #: Set once it ended without publishing all its stages.
    failure: Failure | None = None

    def current(self) -> Submission | None:
        """The live submission, if any."""
        live = [s for s in self.submissions if s.live]
        return live[-1] if live else None


class Outcome(BaseModel):
    """How the instance ended."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: "succeeded" when every stage published. "failed" when a stage failed with a contestant
    #: fault: that is the contestant's scored result, and the instance is never retried or
    #: relaunched. "void" when an infra fault was not retried (not retryable, retries disabled
    #: or used up) or the scheduler refused the instance; a void instance is not the
    #: contestant's result and is relaunched. "cancelled" after ``hls cancel``.
    status: Literal["succeeded", "failed", "void", "cancelled"]
    #: Why, for people and the run record.
    reason: str
    #: When it was decided.
    at: float
    #: The fault of the stage that ended it: the contestant fault of a "failed" instance, the
    #: infra fault of a "void" one when a failed stage decided it; None otherwise.
    fault: FaultRecord | None = None
    #: True for a "failed" instance whose records (its zero score among them) the orchestrator
    #: that ended it left unwritten, as one that only cancels and could not read the secrets file
    #: does; the next orchestrator that takes it over with the file readable writes them and
    #: clears this. False for every other.
    records_left: bool = False

    @model_validator(mode="after")
    def _fault_matches_status(self) -> Self:
        if self.records_left and self.status != "failed":
            raise ValueError("only a failed instance has records to leave")
        if self.status == "failed" and (self.fault is None or self.fault.fault_class != "contestant"):
            raise ValueError("a failed instance carries the contestant fault that ended it")
        if self.status == "void" and self.fault is not None and self.fault.fault_class != "infra":
            raise ValueError("a contestant fault ends the instance failed, not void")
        if self.status in ("succeeded", "cancelled") and self.fault is not None:
            raise ValueError("only a failed or void instance carries a fault")
        return self


class Plan(BaseModel):
    """One revision of an instance's plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Instance root key, such as ``runs/sample_task/<id>``.
    instance: str
    #: Store root on the shared filesystem.
    store_root: Path
    #: Task the instance belongs to.
    task: str
    #: Pipeline it runs.
    pipeline: str
    #: Slurm job name of its allocations.
    job_name: str
    #: When the first revision was made.
    created_at: float
    #: This revision's number, from 1.
    revision: int = Field(ge=1)
    #: Lease generation of the orchestrator that wrote this revision.
    lease_generation: int = Field(ge=1)
    #: The checkout the instance runs from.
    code: CodeState
    #: Programs the allocations run.
    toolchain: Toolchain
    #: The resolved settings.
    settings: Settings
    #: Every stage, in order.
    stages: list[StagePlan]
    #: Every allocation, planned or added for a retry.
    allocations: list[Allocation]
    #: Planner warnings, such as GPUs idle under the max policy, and why an allocation the policy
    #: grouped was split into several that the site's QOS limits hold
    #: (``hls.pipeline.planner.policy_layout``).
    warnings: list[str]
    #: The registered grower (``hls.pipeline.growth``) the orchestrator asks for more stages once
    #: every stage of the plan has published; None for a plan that never grows.
    grower: str | None = None
    #: The measured files its roles load, each pinned where the plan found it (``hls.measured``);
    #: none for a plan whose roles load none.
    measured: MeasuredFiles = MeasuredFiles()
    #: Set once the instance ended.
    outcome: Outcome | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        """Check what later code indexes by: allocations by position, stages by name, and
        each allocation's submissions by number. A retry points back to the allocation it
        retries. A dependency may point forward, because a retry allocation is added at the
        end and the allocation that waited for the failed one is rewired to it; it must still
        name another allocation of the plan. ``waits_for`` names other allocations of the plan,
        each once, and no allocation waits on itself through ``depends_on`` and ``waits_for``
        together. A singleton allocation depends on no other, since its afterok would be
        rewired without it. An allocation on another partition asks for no GPUs: only a run uses
        GPUs, and a run uses the GPU partition."""
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names are not unique")
        for i, alloc in enumerate(self.allocations):
            if alloc.index != i:
                raise ValueError(f"allocation {i} has index {alloc.index}")
            for stage in alloc.stages:
                if stage not in names:
                    raise ValueError(f"allocation {i} runs {stage}, which is not a stage of the plan")
            problem = alloc.placement.request_problem(alloc.request)
            if problem is not None:
                raise ValueError(f"allocation {i} {problem}")
            if alloc.depends_on is not None and not (0 <= alloc.depends_on < len(self.allocations)
                                                     and alloc.depends_on != i):
                raise ValueError(f"allocation {i} depends_on {alloc.depends_on}, which is not another allocation "
                                 f"of the plan")
            for w in alloc.waits_for:
                if not (0 <= w < len(self.allocations) and w != i):
                    raise ValueError(f"allocation {i} waits_for {w}, which is not another allocation of the plan")
            if len(set(alloc.waits_for)) != len(alloc.waits_for):
                raise ValueError(f"allocation {i} waits_for {alloc.waits_for}, which names an allocation twice")
            if alloc.singleton and alloc.depends_on is not None:
                raise ValueError(f"allocation {i} runs one at a time with the site's other warmers (singleton) and "
                                 f"depends_on {alloc.depends_on}; such an allocation waits on no other")
            if alloc.retry_of is not None and not 0 <= alloc.retry_of < i:
                raise ValueError(f"allocation {i} retry_of {alloc.retry_of}, which is not an earlier allocation")
            for position, sub in enumerate(alloc.submissions, start=1):
                if sub.n != position:
                    raise ValueError(f"allocation {i} submission {position} is numbered {sub.n}")
            live = sum(sub.live for sub in alloc.submissions)
            if live > 1:
                raise ValueError(f"allocation {i} has {live} live submissions")
        for i in range(len(self.allocations)):
            through = _waits_on_itself(self.allocations, i)
            if through is not None:
                raise ValueError(f"allocation {i} waits on itself through allocation {through}")
        return self

    @property
    def layout(self) -> InstanceLayout:
        """The instance's store layout."""
        return InstanceLayout(self.instance)

    @property
    def plan_dir(self) -> Path:
        """Absolute path of the revisions directory, the runner's argument."""
        return self.store_root / self.layout.plan

    def stage(self, name: str) -> StagePlan:
        """The stage called *name*; KeyError if the plan has none."""
        for s in self.stages:
            if s.name == name:
                return s
        raise KeyError(name)

    def allocation(self, k: int) -> Allocation:
        """Allocation number *k*; IndexError if there is none (a negative k does not count
        from the end)."""
        if not 0 <= k < len(self.allocations):
            raise IndexError(f"the plan has no allocation {k}")
        return self.allocations[k]

    def with_allocation(self, alloc: Allocation) -> Plan:
        """This plan with *alloc* in place of the allocation of the same index, or added."""
        allocs = list(self.allocations)
        if alloc.index < len(allocs):
            allocs[alloc.index] = alloc
        elif alloc.index == len(allocs):
            allocs.append(alloc)
        else:
            raise ValueError(f"allocation index {alloc.index} skips {len(allocs)}")
        return self.model_copy(update={"allocations": allocs})


def _waited(alloc: Allocation) -> list[int]:
    """The allocations *alloc* waits for: its afterok predecessor and those it waits to publish."""
    return ([alloc.depends_on] if alloc.depends_on is not None else []) + list(alloc.waits_for)


def _waits_on_itself(allocations: list[Allocation], start: int) -> int | None:
    """The allocation through which allocation *start* waits on itself, by ``depends_on`` and
    ``waits_for`` together; None when it does not. Two allocations that wait for the same one
    are no cycle."""
    seen: set[int] = set()
    pending = [(start, k) for k in _waited(allocations[start])]
    while pending:
        via, k = pending.pop()
        if k == start:
            return via
        if k in seen:
            continue
        seen.add(k)
        pending += [(k, w) for w in _waited(allocations[k])]
    return None


def revision_numbers(names: Iterable[str]) -> list[int]:
    """The ascending revision numbers among the file names in a plan directory. Other names
    are ignored."""
    return sorted(int(m.group(1)) for name in names if (m := _REVISION_FILE.fullmatch(name)))


def submission_tag(instance: str, k: int, n: int) -> str:
    """The ``--comment`` of submission *n* of allocation *k*; unique across instances."""
    return f"hls:{instance}/{k}/{n}"


class PlanConflict(StoreError):
    """The plan changed since the caller read it; reread and decide again."""


class PlanMismatch(StoreError):
    """A plan is not the one its place in the store says: it belongs to another instance, or
    a revision file holds another revision number."""


class InstanceExists(StoreError):
    """The instance already has a plan."""


class PlanUnreadable(StoreError):
    """A stored plan revision that this version of hls cannot read: one another version wrote,
    whose settings or stages this one does not have, or a file that is not JSON."""


def _unreadable(key: str, data: bytes, error: ValidationError) -> str:
    """Why the plan revision *key*, whose bytes are *data*, does not load (*error*): the first
    field this version refuses, and the hls commit that wrote the plan when its text names one. It
    quotes no value of the plan, which may hold a path or a name the user would not print."""
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"]) or "the plan"
    try:
        written = json.loads(data).get("code", {}).get("hls_commit")
    except (ValueError, AttributeError):
        return f"{key} is not a plan: it is not a JSON object ({first['msg']})"
    by = f"hls commit {written}" if isinstance(written, str) else "another version of hls"
    more = f", and {error.error_count() - 1} more" if error.error_count() > 1 else ""
    return (f"{key} was written by another version of hls ({by}) and does not load under this one: "
            f"{field}: {first['msg']}{more}. Resume, follow or inspect it with a checkout of that version")


class PlanStore:
    """The revisions of one instance's plan."""

    def __init__(self, store: Store, layout: InstanceLayout) -> None:
        self.store = store
        #: The instance.
        self.layout = layout

    def revisions(self) -> list[int]:
        """The numbers of the revisions written so far, ascending."""
        return revision_numbers(self.store.list_children(self.layout.plan))

    def read(self, n: int) -> Plan:
        """Revision *n*. Raise NotFound if there is none, PlanUnreadable if this version of hls
        cannot read it (``_unreadable``), and PlanMismatch if the file holds another revision or
        another instance's plan."""
        key = self.layout.plan_revision(n)
        data = self.store.read_bytes(key)
        try:
            plan = Plan.model_validate_json(data)
        except ValidationError as e:
            raise PlanUnreadable(_unreadable(key, data, e)) from None
        if plan.revision != n or plan.instance != self.layout.root:
            raise PlanMismatch(f"{self.layout.plan_revision(n)} holds revision {plan.revision} and belongs to "
                               f"{plan.instance}")
        return plan

    def latest(self) -> Plan | None:
        """The newest revision, or None when no plan was written."""
        revs = self.revisions()
        return self.read(revs[-1]) if revs else None

    def _create(self, plan: Plan) -> Plan:
        checked = Plan.model_validate(plan.model_dump())
        if checked.instance != self.layout.root:
            raise PlanMismatch(f"a plan of {checked.instance} cannot be written to {self.layout.root}")
        self.store.create(self.layout.plan_revision(checked.revision), canonical_json(checked))
        return checked

    def create(self, plan: Plan, handle: LockHandle) -> Plan:
        """Write revision 1 as the holder of *handle*. Raise InstanceExists if there is one."""
        handle.confirm()
        try:
            return self._create(plan.model_copy(update={"revision": 1, "lease_generation": handle.generation}))
        except AlreadyExists as e:
            raise InstanceExists(self.layout.root) from e

    def write(self, plan: Plan, handle: LockHandle) -> Plan:
        """Write *plan* as the revision after the one it was derived from, as the holder of
        *handle*, and return it as written. Raise LockLost if a newer lease generation has
        written, and PlanConflict if another revision came first under the same generation.
        Raise NotFound if there is no plan to derive from."""
        handle.confirm()
        latest = self.latest()
        if latest is None:
            raise NotFound(f"{self.layout.root} has no plan")
        if latest.lease_generation > handle.generation:
            handle.confirm()
            raise LockLost(f"{self.layout.lease}: generation {latest.lease_generation} wrote revision "
                           f"{latest.revision}; this is generation {handle.generation}")
        if plan.revision != latest.revision:
            raise PlanConflict(f"plan derived from revision {plan.revision}, latest is {latest.revision}")
        update = {"revision": latest.revision + 1, "lease_generation": handle.generation}
        try:
            return self._create(plan.model_copy(update=update))
        except AlreadyExists as e:
            # Another writer took the number. It is a takeover if the lease is gone.
            handle.confirm()
            raise PlanConflict(f"revision {latest.revision + 1} was written first") from e


class AllocationStatus(BaseModel):
    """What a runner reports about its allocation submission, rewritten every heartbeat."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Instance root key.
    instance: str
    #: Allocation index.
    allocation: int = Field(ge=0)
    #: Submission number.
    submission: int = Field(ge=1)
    #: Slurm job id.
    job_id: str
    #: Node it runs on.
    node: str
    #: "starting" through the startup checks, "running" through the stages, then how it
    #: ended: "succeeded", "failed", "refused" (it did no work) or "terminated" (by signal).
    state: Literal["starting", "running", "succeeded", "failed", "refused", "terminated"]
    #: When the runner started.
    started_at: float
    #: The last heartbeat.
    heartbeat_at: float
    #: The job's EndTime the runner keeps to: the controller's record of the job, else
    #: ``SLURM_JOB_END_TIME``; None when neither is known.
    end_time: float | None
    #: The stage running now.
    stage: StageName | None
    #: The attempt of that stage running now, whose status holds the fault it declared any end of
    #: it is (``AttemptStatus.no_retry``), for the orchestrator to read after a runner that died
    #: with no record of its own; None when no stage runs.
    attempt: int | None = Field(ge=1)
    #: The measured grant; None before the grant check.
    grant: GrantedResources | None
    #: The versions of the node's Slurm, apptainer, pasta and kernel, measured with the grant
    #: (``hls.substrate.versions``); None before the grant check.
    tool_versions: ToolVersions | None
    #: The job cgroup's swap limit in bytes; None when unlimited or not yet read.
    swap_max_bytes: int | None
    #: Who enforces memory limits: the kernel where swap is off, the runner otherwise.
    memory_enforced_by: Literal["kernel", "runner"] | None
    #: Why it failed or refused; None otherwise.
    fault: FaultRecord | None
    #: When it ended.
    ended_at: float | None


class AttemptStatus(BaseModel):
    """One stage attempt, written by the runner that made it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: StageName
    attempt: int = Field(ge=1)
    #: Allocation that ran it.
    allocation: int = Field(ge=0)
    #: Submission of that allocation that ran it.
    submission: int = Field(ge=1)
    #: Slurm job id.
    job_id: str
    #: "running", then "succeeded" or "failed".
    state: Literal["running", "succeeded", "failed"]
    #: When it started.
    started_at: float
    #: When it ended; None while it runs.
    ended_at: float | None
    #: Why it failed; None otherwise.
    fault: FaultRecord | None
    #: Digest of the manifest it published; None otherwise.
    manifest_sha256: Sha256Hex | None
    #: The fault, infra and never retried, that any end of the attempt is from the moment its
    #: stage declared it had done work a retry would do again (``StageContext.declare_no_retry``,
    #: the agent build's commit point); None while it declared none. The runner writes it into the
    #: running record then, so a runner killed outright after it, which writes no fault of its own,
    #: still leaves it for ``reconcile.job_fault``.
    no_retry: FaultRecord | None


def attempt_hardware(store: Store, layout: InstanceLayout, stage: str, attempt: int) -> HardwareRecord | None:
    """The hardware record of the allocation that ran attempt *attempt* of *stage* of the instance
    *layout* names, as its runner's grant check measured it: the attempt's
    status names its allocation and submission, whose status holds the grant. None when either
    status is missing or does not parse, the two name different jobs, or the grant check had not
    run."""
    try:
        run = AttemptStatus.model_validate_json(store.read_bytes(layout.attempt_file(stage, attempt, "status.json")))
        status = AllocationStatus.model_validate_json(store.read_bytes(layout.allocation_status(run.allocation,
                                                                                                run.submission)))
    except (NotFound, ValidationError):
        return None
    if status.job_id != run.job_id or status.grant is None:
        return None
    return status.grant.hardware()
