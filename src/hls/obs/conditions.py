"""The harness-conditions table on every rollout and grading record.

A result depends on the harness conditions it was reached under: a retried stage, a budget other
than the task's, the credit it was scored against, the deny list a brief named.
Finalize, and the orchestrator's records of a failed instance, write this table beside the
episode, built from the plan and the store when the instance is recorded:
- every stage's attempts, each with how it ended and its fault, and the attempt whose manifest
  the stage published;
- every allocation, the jobs of its submissions, what it retried and how it failed, the
  versions of the Slurm, apptainer, pasta and kernel each job's runner found on its node, and the
  hardware it measured there;
- the hls and harbor commits, and whether the checkout held uncommitted changes;
- the hardware the scored drive ran on, the hardware the baseline credit's drives ran on, and a
  warning when a field the credit records differs (``hardware_check``);
- the baseline credit the score was computed against, as the score role recorded it, and the
  calibration blob the verifiers were given;
- where each measured file the roles loaded was found, as the plan pinned it (``hls.measured``),
  and which of them are the task's defaults (``measured_note``);
- every image the instance ran, by digest;
- where the scored server's CPUs lay, as its drive recorded it (a site that
  cannot ask for one socket records a split rather than refusing it);
- the arm whose agent built the submission and the budgets it ran under, and whether those are
  not the task's own;
- the rule that build ran under for a sandbox that ends on its own memory, graded on its last
  commit or scored zero."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from hls.arms import ArmRecord
from hls.measured import MeasuredFiles
from hls.pipeline.plan import AllocationStatus, AttemptStatus, Plan, Submission, attempt_hardware
from hls.spec.resources import ArtifactPlacement, BaselineHardware, HardwareRecord
from hls.store.base import NotFound, Store
from hls.store.manifest import Manifest
from hls.substrate.factory import substrate
from hls.substrate.versions import ToolVersions


class AttemptNote(BaseModel):
    """One attempt of a stage, as its status says it ended."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The attempt's number.
    attempt: int
    #: Its status's state ("running", "succeeded" or "failed"); "claimed" when it wrote no status,
    #: "unreadable" when its status does not parse.
    state: str
    #: The class of its fault; None without one.
    fault_class: Literal["infra", "contestant"] | None = None
    #: Why it failed, as the runner recorded it, scrubbed; None without a fault.
    reason: str | None = None
    #: Whether its fault could be retried; None without a fault.
    retryable: bool | None = None


class StageAttempts(BaseModel):
    """Every attempt of one stage, and the one that counted."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The attempt whose manifest the stage published; None when it published none.
    counted: int | None
    #: Every attempt the stage claimed, first first.
    attempts: list[AttemptNote]


class AllocationNote(BaseModel):
    """One allocation of the plan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Its index in the plan.
    index: int
    #: The allocation whose failed stage it retried; None for a planned one.
    retry_of: int | None
    #: The Slurm job of each of its submissions that got one, in order.
    job_ids: list[str]
    #: The nodes it was kept off, each one an earlier allocation's grant check failed on
    #: (``SlurmPlacement.exclude``).
    exclude: list[str]
    #: "<stage>: <reason>" when it ended without publishing every stage; None otherwise.
    failure: str | None
    #: The versions of the Slurm, apptainer, pasta and kernel each submission's job ran on, by the
    #: job's id, as its runner recorded them with its grant (``AllocationStatus.tool_versions``):
    #: every job of ``job_ids``, and a job the plan the record was made from did not know yet whose
    #: runner's status names it. None for a job whose submission has no status of its own: none
    #: written, one that does not parse or another job wrote, or one from before its grant check.
    tool_versions: dict[str, ToolVersions | None]
    #: The hardware each of those jobs ran on, by the job's id, as its runner's grant check
    #: measured it (``GrantedResources.hardware``), read from the same status;
    #: None where its versions are.
    hardware: dict[str, HardwareRecord | None]


class ArmNote(BaseModel):
    """The arm whose agent built the submission."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The arm as the plan resolved it: names only.
    record: ArmRecord
    #: The declared reason it differs from the task's earlier builds of it; None when it does not,
    #: and for the arm of a package a grading grades.
    arm_change: str | None


class BudgetNote(BaseModel):
    """The budgets an agent build ran under."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The wall-clock budget the agent was told, in seconds; None when it was told none.
    soft_sec: int | None
    #: The limit the runner enforced, in seconds.
    hard_sec: int
    #: The turn cap; None for none.
    max_turns: int | None
    #: True when these are not the task's own budgets (``hls run --soft-sec/--hard-sec``); None
    #: when the task's could not be read.
    overridden: bool | None


class HarnessConditions(BaseModel):
    """How an instance ran."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Each stage's attempts, by stage name, in plan order.
    attempts: dict[str, StageAttempts]
    #: Every allocation, planned or added for a retry.
    allocations: list[AllocationNote]
    #: The hls commit the instance ran.
    hls_commit: str
    #: The harbor commit it ran.
    harbor_commit: str
    #: Whether the checkout held uncommitted changes (``CodeState.dirty_diff_sha256``).
    dirty: bool
    #: The hardware the scored drive ran on: the grant of the allocation that ran its counted
    #: attempt (``attempt_hardware``); None for an instance that drove nothing and for a drive whose
    #: allocation's status records none.
    hardware: HardwareRecord | None
    #: The hardware the baseline credit's drives ran on, as the credit records it, blank fields
    #: included; None for a task scored against no baseline credit, a credit that records none,
    #: and before a score.
    baseline_hardware: list[BaselineHardware] | None
    #: How the scored drive's hardware differs from the baseline credit's on the fields the credit
    #: records (``hardware_check``); None when it does not. The grading is scored as on the same
    #: hardware.
    hardware_warning: str | None
    #: The baseline credit the score was computed against, as the score role recorded it (its B,
    #: its blob id, its campaign's id); None for a scorer that scores against none, and before a
    #: score.
    credit: dict[str, Any] | None
    #: The calibration blob the verifiers were given: its blob id, its kind and the workload
    #: fingerprint it was measured for; None for a task that needs none.
    calibration: dict[str, Any] | None
    #: The measured files the roles loaded, as the plan pinned them: where each was found, the
    #: store's measured area or the task's defaults, and its blob id.
    measured: MeasuredFiles
    #: Which of the measured files the run used are the task's defaults (``measured_note``); None
    #: when the run used no default.
    measured_note: str | None
    #: The arm whose agent built the submission; None for one no agent built.
    arm: ArmNote | None
    #: The budgets that build ran under; None for an instance that ran no build.
    budget: BudgetNote | None
    #: The rule that build ran under for a sandbox that ends on its own memory, its plan's
    #: ``run.grade_after_sandbox_oom``: true when such a build is graded on
    #: its last commit; false when it scores zero with nothing graded.
    #: None for an instance that ran no build.
    grade_after_sandbox_oom: bool | None
    #: Every image the instance ran, by role, to its file's digest.
    images: dict[str, str]
    #: Where the scored server's step ran, as its drive recorded it: its CPUs, their NUMA nodes and
    #: whether they were split; None when no drive counted or its step reported nothing.
    placement: ArtifactPlacement | None


def _attempts(store: Store, plan: Plan, stage: str) -> list[AttemptNote]:
    """Every attempt *stage* claimed, with how its status says it ended."""
    layout = plan.layout
    notes = []
    for n in layout.attempts(store, stage):
        try:
            status = AttemptStatus.model_validate_json(store.read_bytes(layout.attempt_file(stage, n, "status.json")))
        except NotFound:
            notes.append(AttemptNote(attempt=n, state="claimed"))
            continue
        except ValidationError:
            notes.append(AttemptNote(attempt=n, state="unreadable"))
            continue
        fault = status.fault
        notes.append(AttemptNote(attempt=n, state=status.state, fault_class=fault.fault_class if fault else None,
                                 reason=fault.reason if fault else None,
                                 retryable=fault.retryable if fault else None))
    return notes


def _submission_tools(store: Store, plan: Plan, k: int,
                      sub: Submission) -> tuple[str, ToolVersions | None, HardwareRecord | None] | None:
    """The job of submission *sub* of allocation *k*, and the tool versions and hardware its
    runner recorded in its status, the status being usable only when it parses and was written
    for this submission by its job, as the orchestrator reads it. The job is the plan's, else,
    while the plan does not know it yet (a runner may start before the orchestrator records its
    job's id), the one the usable status names; None when neither names one. The versions and the
    hardware are None without a usable status, and in one written before the grant check."""
    status: AllocationStatus | None
    try:
        status = AllocationStatus.model_validate_json(store.read_bytes(plan.layout.allocation_status(k, sub.n)))
        substrate(plan.settings.site).check_job_id(status.job_id)
    except (NotFound, ValidationError, ValueError):
        status = None
    if status is not None and ((status.instance, status.allocation, status.submission) != (plan.instance, k, sub.n)
                               or sub.job_id not in (None, status.job_id)):
        status = None
    job = sub.job_id if sub.job_id is not None else status.job_id if status is not None else None
    if job is None:
        return None
    if status is None:
        return job, None, None
    return job, status.tool_versions, status.grant.hardware() if status.grant is not None else None


def measured_note(files: MeasuredFiles) -> str | None:
    """The line a record carries for the task's defaults *files* pins: which files the run used that
    are the task's defaults. None when *files* pins no default."""
    defaults = [pin.path for pin in files.pins().values() if pin.source == "defaults"]
    if not defaults:
        return None
    return f"used the task's default{'s' if len(defaults) > 1 else ''} {' and '.join(defaults)}"


#: How a credit record's hardware reads back (``hls.score.role.CreditRecord.hardware``).
_RECORDS = TypeAdapter(list[BaselineHardware] | None)


def hardware_check(drive: HardwareRecord | None, baseline: list[BaselineHardware] | None) -> str | None:
    """The warning a grading's record carries for the hardware its scored drive ran on, *drive*,
    against the hardware its baseline credit's drives ran on, *baseline*:
    each field the baseline records that differs, the drive's value then the baseline's, against
    a record of the baseline's, every record when the baseline's drives ran on more than one.
    None when no recorded field differs, and when either side records no hardware. It refuses
    and changes nothing: the grading is scored as it would be on the same hardware."""
    if baseline is None or drive is None:
        return None
    differing = [(record, drive.differences(record)) for record in baseline if drive.differences(record)]
    if not differing:
        return None
    if len(baseline) == 1:
        said = ", ".join(differing[0][1])
    else:
        said = f"the baseline's drives ran on {len(baseline)} hardware records; " + "; ".join(
            f"against {record.describe()}: {', '.join(fields)}" for record, fields in differing)
    return (f"warning: this drive ran on other hardware than its baseline credit's drives, so its score compares "
            f"the two: {said}")


def harness_conditions(store: Store, plan: Plan, *, published: Sequence[Manifest],
                       credit: dict[str, Any] | None, calibration: dict[str, Any] | None, arm: ArmNote | None,
                       budget: BudgetNote | None, placement: ArtifactPlacement | None, drive_stage: str | None,
                       grade_after_sandbox_oom: bool | None) -> HarnessConditions:
    """The harness conditions of *plan*'s instance now: its stages' attempts and its allocations
    read from the plan and *store*, the attempt each stage counted from its manifest among
    *published*, the images every manifest of *published* records, the hardware the counted
    attempt of *drive_stage* ran on when it published, the *credit*, *calibration*, *arm*,
    *budget*, server *placement* and build rule *grade_after_sandbox_oom*
    the caller read from the plan and the store, and the drive's hardware checked against the
    credit's (``hardware_check``) for an instance scored against one. Reads only attempt and
    submission statuses, which are small."""
    counted = {m.stage: m.attempt for m in published}
    images: dict[str, str] = {}
    for m in published:
        images |= m.provenance.image_digests
    attempts = {s.name: StageAttempts(counted=counted.get(s.name), attempts=_attempts(store, plan, s.name))
                for s in plan.stages}
    allocations = []
    for a in plan.allocations:
        jobs = [found for s in a.submissions if (found := _submission_tools(store, plan, a.index, s)) is not None]
        allocations.append(AllocationNote(
            index=a.index, retry_of=a.retry_of, job_ids=[s.job_id for s in a.submissions if s.job_id is not None],
            exclude=list(a.placement.exclude), failure=f"{a.failure.stage}: {a.failure.fault.reason}" if a.failure else None,
            tool_versions={job: tools for job, tools, _ in jobs}, hardware={job: hw for job, _, hw in jobs}))
    hardware = (attempt_hardware(store, plan.layout, drive_stage, counted[drive_stage])
                if drive_stage is not None and drive_stage in counted else None)
    baseline = _RECORDS.validate_python(credit.get("hardware")) if credit is not None else None
    warning = hardware_check(hardware, baseline) if credit is not None else None
    return HarnessConditions(attempts=attempts, allocations=allocations, hls_commit=plan.code.hls_commit,
                             harbor_commit=plan.code.harbor_commit, dirty=plan.code.dirty_diff_sha256 is not None,
                             hardware=hardware, baseline_hardware=baseline, hardware_warning=warning,
                             credit=credit, calibration=calibration, measured=plan.measured,
                             measured_note=measured_note(plan.measured), arm=arm, budget=budget,
                             grade_after_sandbox_oom=grade_after_sandbox_oom, images=images, placement=placement)
