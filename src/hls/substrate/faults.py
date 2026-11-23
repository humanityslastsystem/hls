"""Fault classes for the ways a role can end.

A failure in a harness role is an infra fault, including an OOM-killed process in a step
whose main process survived. An OOM in a contestant role is the contestant's fault only when
it is certain that the role's own limit was reached: the runner's memory watch measured the
role's footprint past its declared memory, or the kernel killed a process of the role when its
step's own limit was reached and the job's own limit was not, during the stage. Any other
kernel kill is infra: the node's OOM killer and a limit above or at the job pick their victim
among more processes than the role's, a contestant process first, and leave the same kill in
the role's step. So is a kill when a count it needs cannot be read (``memory_watch``). A
contestant role the watch killed for writing past its declared disk (``disk_limit``) is the
contestant's fault, a harness role's infra, and so is a contestant role whose step reached its
pids limit (``contestant_pids``) within ``OOM_EVIDENCE_MAX_AGE_SEC`` of its end or its stop
(``within_evidence_age``, read by the stage context). A contestant role that exited on its own while the uid's
other tasks alone filled its process pool, counted once its step was gone, was starved by them,
and its end is infra (``starvation_fault``); so is one that failed while others held the
node-local disk its cap still left it (``hls.substrate.memory_watch.disk_starvation``). A role
the stage stopped as it planned
(``planned_stop``, such as a server once its drive is over) ended cleanly, whoever owns it:
only an OOM kill before the stop counts. Any other failure of a contestant role is a result
the stage scores, not a fault. Grant
mismatches, launch failures, roles stopped by a shutdown, steps the scheduler ended (or whose
end it did not confirm) and containers whose runtime failed before their command started are
infra whoever owns the role, and so is a role whose log capper died (``log_lost``), since its
step then wrote into a broken pipe.

The build sandbox, a contestant container whose step also holds harness processes, is judged by
a stricter rule (``classify_build_end``), which ``classify_role`` applies to every outcome that
carries the sandbox's memory split. The stage's planned stop is its only end that is no fault.
Any other end cuts the build short. Writing past its disk cap or reaching its step's pids limit
ends it as the contestant's fault, and an OOM does only when it is certain that the sandbox's
own limit was reached while the harness stayed within its reserve. Any other end is infra, and
one the uid's other tasks starved (``starvation_fault``) says so. Beside the in-build reference
service, whose memory is oversubscribed on the sandbox's, a kernel kill at the job's own limit is
decided by what the service and the runner held (``classify_shared_build_end``).
The rule says whose end it was, not what follows: the agent build stage grades a build that
ended on the contestant's fault on its last commit, and fails it only when nothing is gradable."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from hls.substrate.base import FaultRecord
from hls.substrate.memory_watch import DiskAtEnd, disk_starvation
from hls.substrate.nproc import PoolAtEnd, PoolSample, starvation

#: How a role ended.
Ending = Literal["exited", "grant_mismatch", "memory_limit", "time_limit", "launch_failed", "stopped",
                 "scheduler_ended", "runtime_failed", "planned_stop", "disk_limit", "contestant_pids",
                 "log_lost"]

#: How long before the build sandbox's end its last OOM, its pids limit reached or the runner's
#: memory kill in it may be, and how long before that OOM the split sample read as its evidence
#: may be, for the build rule to name the contestant (``within_evidence_age``).
OOM_EVIDENCE_MAX_AGE_SEC = 30.0
# Endings of the build sandbox that are infra however its memory looked.
_SANDBOX_INFRA = ("grant_mismatch", "launch_failed", "stopped", "scheduler_ended", "runtime_failed", "time_limit",
                  "log_lost")


def within_evidence_age(evidence_at: float | None, end_at: float) -> bool:
    """Whether the contestant's evidence dated *evidence_at* (an OOM, a memory kill, a pids limit
    reached, the split sample before an OOM) may explain an end at *end_at*: it came at most
    ``OOM_EVIDENCE_MAX_AGE_SEC`` before it, inclusive. Evidence first read after the end is the
    end's own, as a count is dated by the read that shows it; undated evidence (None) explains
    nothing. Evidence a role survived longer before says nothing about a later end. The build
    rule (``classify_build_end``), the stage context's reading of every contestant container's
    pids limit and the agent build's evidence all bound their evidence by this one rule."""
    return evidence_at is not None and end_at - evidence_at <= OOM_EVIDENCE_MAX_AGE_SEC


@dataclass(frozen=True)
class SplitEvidence:
    """What the build rule reads of a role whose step holds harness processes beside the
    contestant's, the build sandbox: its last OOM and the memory split sampled before it
    (``hls.agent.memory.split_evidence``). Times are POSIX seconds, sizes bytes."""

    #: The memory of the step its harness may hold.
    reserve_bytes: int
    #: When the stage context saw the role end.
    ended_at: float
    #: When its last OOM was: the later of the step's last kernel OOM kill and the runner's kill of
    #: the step; None when there was neither. A single kill leaves the sandbox running, so it is
    #: none.
    oom_at: float | None = None
    #: Whose that OOM was: the kernel's OOM killer's, or the runner's memory watch's.
    oom_by: Literal["kernel", "runner"] | None = None
    #: When the last split sample before the OOM was read; None when there is no such sample.
    sample_at: float | None = None
    #: The step's footprint in that sample.
    total_bytes: int | None = None
    #: The tmpfs pages charged to the step in it.
    tmpfs_bytes: int | None = None
    #: The resident memory of the harness's processes in it.
    harness_bytes: int | None = None
    #: The resident memory of the contestant's processes in it.
    contestant_bytes: int | None = None
    #: When the runner's memory watch ended one contestant process of the step to bring it back
    #: to its limit: the latest ``hls.agent.memory.PROCESS_KILLS_KEPT`` of those times, oldest
    #: first.
    process_kills: tuple[float, ...] = ()
    #: How many such single kills there were in all.
    process_kill_count: int = 0
    #: Why the watch killed the whole step, a runner's kill over the step's limit like any other:
    #: single kills did not bring it back, no contestant process was left to end, or its processes
    #: could not be told apart. None when it did not.
    step_kill_reason: str | None = None


@dataclass(frozen=True)
class RoleOutcome:
    """How one role ended, with the evidence the fault rules read."""

    name: str
    #: Whose code it ran.
    owner: Literal["harness", "contestant"]
    #: Exit status of the step; None when it never started or was not reaped.
    exit_code: int | None
    #: How it ended.
    ending: Ending
    #: ``oom_kill`` events in the step's cgroup at the last memory sample; at least 1 when the
    #: step ended OUT_OF_MEMORY in the scheduler's accounting.
    oom_kills: int = 0
    #: Anything else worth recording, such as the memory sample that tripped the limit.
    detail: str = ""
    #: Whether the role's own memory limit was reached: its step's own ``oom`` count, read
    #: where the step's cgroup still is when the role is reported, or else in the memory sample
    #: that saw the kill; or, once the step is gone, a limit reached below the job during the
    #: stage while every other step that could have reached its own is still there unchanged
    #: (``StageContext._own_oom``). False when it was not; None when that cannot be told.
    own_oom: bool | None = None
    #: Whether the job's own memory limit was reached between the start of the stage and the
    #: time the role was reported, from the job cgroup's own ``oom`` count; None when that
    #: count could not be read.
    job_oom: bool | None = None
    #: For a contestant step with a process ceiling, the last sample of its uid's process pool
    #: and of its own tasks a walk read before it ended or was stopped (``hls.substrate.nproc``):
    #: context, which decides nothing. None otherwise, or when none was read.
    pool: PoolSample | None = None
    #: For a contestant step with a process ceiling, the uid's tasks counted when it was
    #: reported, its step gone: the others' alone, which decide whether it was starved
    #: (``starvation_fault``), or the count that failed, which leaves it not ruled out
    #: (``PoolAtEnd.others_tasks`` None). None for any other role.
    pool_at_end: PoolAtEnd | None = None
    #: The build rule's evidence, for a role started with a harness reserve (the build sandbox);
    #: None for every other role. With it, ``classify_role`` applies the build rule.
    split: SplitEvidence | None = None
    #: For a contestant step with a disk cap, what its writable directories held and the free
    #: space of their filesystem when it was reported, its step gone, which decide whether others
    #: starved it of disk (``starvation_fault``). None for any other role.
    disk_at_end: DiskAtEnd | None = None


def infra_fault(reason: str, *, retryable: bool = True, **evidence: str) -> FaultRecord:
    """An infra fault with *reason* for people and *evidence* as its supporting facts."""
    return FaultRecord(fault_class="infra", reason=reason, retryable=retryable, evidence=evidence)


def _evidence(outcome: RoleOutcome) -> dict[str, str]:
    """What a fault of *outcome* carries as its supporting facts."""
    evidence = {"ending": outcome.ending, "exit_code": str(outcome.exit_code), "oom_kills": str(outcome.oom_kills)}
    if outcome.detail:
        evidence["detail"] = outcome.detail
    if outcome.pool is not None:
        evidence.update(walk_uid_tasks=str(outcome.pool.uid_tasks), walk_step_tasks=str(outcome.pool.step_tasks),
                        ceiling=str(outcome.pool.ceiling))
    if outcome.pool_at_end is not None:
        at_end = outcome.pool_at_end
        evidence.update(others_tasks=_count(at_end.others_tasks), ceiling=str(at_end.rule.ceiling),
                        accounting=at_end.rule.accounting, node_tasks=_count(at_end.node_tasks),
                        pid_max="unread" if at_end.pid_max is None else str(at_end.pid_max))
        if at_end.rule.accounting == "user_namespace":
            limit = at_end.rule.parent_limit
            evidence["parent_limit"] = "unlimited" if limit is None else str(limit)
    if outcome.disk_at_end is not None:
        disk = outcome.disk_at_end
        evidence.update(disk_cap_bytes=str(disk.limit_bytes),
                        disk_own_bytes="unmeasured" if disk.own_bytes is None else str(disk.own_bytes),
                        disk_free_bytes="unread" if disk.free_bytes is None else str(disk.free_bytes))
    return evidence


def _count(tasks: int | None) -> str:
    """A count of tasks for the evidence: its value, or ``uncounted``."""
    return "uncounted" if tasks is None else str(tasks)


def starvation_fault(outcome: RoleOutcome) -> FaultRecord | None:
    """The infra fault of a contestant role whose end the others' tasks or files may have caused:
    counted when it was reported, its step gone, at a level that counts its tasks they leave it at
    most what ``POOL_FULL`` of its ceiling leaves (``hls.substrate.nproc.starvation``), so its forks
    may have failed for what others started; or they could not be counted then
    (``PoolAtEnd.uncounted``), so that cannot be ruled out. Likewise for its disk: the free space
    of its node-local filesystem then left it at most what ``POOL_FULL`` of the room its cap still
    allowed leaves (``hls.substrate.memory_watch.disk_starvation``), so its writes may have failed
    for what others wrote, or that space could not be read. None otherwise, for a role with
    neither a ceiling nor a disk cap, and for a harness role. It is asked only of a failing end:
    ``classify_role`` asks it of a contestant role that exited with a status other than 0, and a
    stage that scores
    another end as the contestant's failure asks it through
    ``hls.pipeline.stage.scored_end_fault``: a step cut at its time limit, a server stopped not
    ready in time, a server that exited 0 before it was ready or while it was driven. A role
    that exited 0 and succeeded is never retried for its pool."""
    if outcome.owner != "contestant":
        return None
    name, at_end, disk = outcome.name, outcome.pool_at_end, outcome.disk_at_end

    def infra(reason: str) -> FaultRecord:
        return FaultRecord(fault_class="infra", retryable=True, evidence=_evidence(outcome), reason=reason)

    if at_end is not None:
        missing = at_end.uncounted
        if missing is not None:
            return infra(f"contestant role {name} ended, and {missing} could not be counted once its step was "
                         "gone, so starvation cannot be ruled out")
        why = starvation(at_end)
        if why is not None:
            return infra(f"contestant role {name} ended while {why} once its step was gone: the others starved it")
    if disk is None:
        return None
    if disk.free_bytes is None:
        return infra(f"contestant role {name} ended, and its node-local disk could not be read once its step was "
                     "gone, so starvation of it cannot be ruled out")
    why = disk_starvation(disk)
    if why is None:
        return None
    return infra(f"contestant role {name} ended while {why} once its step was gone: the others held its disk")


def classify_role(outcome: RoleOutcome) -> FaultRecord | None:
    """The fault *outcome* is, or None when the role succeeded or its failure is a result."""
    if outcome.split is not None:
        return classify_build_end(outcome)
    evidence = _evidence(outcome)
    if outcome.ending in ("grant_mismatch", "launch_failed", "stopped", "scheduler_ended", "runtime_failed",
                          "log_lost"):
        return FaultRecord(fault_class="infra", reason=f"role {outcome.name}: {outcome.ending.replace('_', ' ')}",
                           retryable=True, evidence=evidence)
    oom = outcome.ending == "memory_limit" or outcome.oom_kills > 0
    if oom:
        evidence["own_oom"] = str(outcome.own_oom)
        evidence["job_oom"] = str(outcome.job_oom)
    clean = outcome.ending == "planned_stop" or (outcome.ending == "exited" and outcome.exit_code == 0)
    if outcome.owner == "harness":
        if clean and not oom:
            return None
        what = "ran out of memory" if oom else f"ended {outcome.ending} with status {outcome.exit_code}"
        return FaultRecord(fault_class="infra", reason=f"harness role {outcome.name} {what}", retryable=True,
                           evidence=evidence)
    if outcome.ending == "disk_limit":
        # How the cap was passed, bytes over it or a tree deeper than the walk goes, is the record's.
        how = f": {outcome.detail}" if outcome.detail else ""
        return FaultRecord(fault_class="contestant", retryable=False, evidence=evidence,
                           reason=f"contestant role {outcome.name} wrote past its disk cap{how}")
    if outcome.ending == "contestant_pids":
        how = f": {outcome.detail}" if outcome.detail else ""
        return FaultRecord(fault_class="contestant", retryable=False, evidence=evidence,
                           reason=f"contestant role {outcome.name} reached its step's process limit{how}")
    if not oom:
        # A clean exit 0 is judged by the stage that scores it, when it scores it as a failure.
        return starvation_fault(outcome) if outcome.ending == "exited" and not clean else None
    if outcome.ending == "memory_limit" or (outcome.own_oom is True and outcome.job_oom is False):
        return FaultRecord(fault_class="contestant", reason=f"contestant role {outcome.name} ran out of memory",
                           retryable=False, evidence=evidence)
    return FaultRecord(fault_class="infra",
                       reason=f"contestant role {outcome.name} was OOM-killed and {_not_own_limit(outcome)}",
                       retryable=True, evidence=evidence)


def _not_own_limit(outcome: RoleOutcome) -> str:
    """Why a kernel OOM kill in *outcome*'s step is not certain to be at its own limit alone."""
    if outcome.job_oom:
        return "the job's own memory limit was reached"
    if outcome.own_oom is False:
        return "its own memory limit was not reached"
    if outcome.own_oom is None:
        return "it is unknown whether its own memory limit was reached"
    return "it is unknown whether the job's own memory limit was reached"


def classify_build_end(outcome: RoleOutcome) -> FaultRecord | None:
    """The fault *outcome*, an end of the build sandbox, is; ``classify_role``
    reads every outcome that carries ``split`` this way.

    The stage's planned stop is the one end that is no fault: the sandbox was alive when the
    build was over, whatever it survived before. Writing past its disk cap is the contestant's
    fault, and so is reaching its step's pids limit (``contestant_pids``), which the stage
    context reads, for every contestant container, only within ``OOM_EVIDENCE_MAX_AGE_SEC`` of
    its end or its stop (``StageContext.pids_hits``, ``within_evidence_age``), as an OOM is read
    here. Any other
    end cuts the build short, and is the contestant's fault only when all of these hold:
    1. the sandbox's own limit certainly tripped, and that OOM is its end's. Its last OOM is the
       runner's own kill of the step where swap is unconstrained (its kills of single processes
       leave the sandbox running and are only recorded), or a kernel OOM kill while the step's
       own limit was reached and the job's was not (``own_oom`` true and ``job_oom`` false, read
       as for any role). It came at most ``OOM_EVIDENCE_MAX_AGE_SEC`` before the end: an OOM the
       sandbox survived says nothing about a later death;
    2. the last split sample taken before that OOM shows the harness within its reserve;
    3. that sample is at most ``OOM_EVIDENCE_MAX_AGE_SEC`` older than the OOM.
    The bounds are inclusive. Anything less certain is infra, and is retried, and so is an end
    that the grant, the launch, a shutdown, the scheduler, the runtime, a dead log capper
    (``log_lost``) or the stage's own deadline caused, whatever the memory says. An end without
    an OOM that the uid's other tasks starved, counted when it was reported (``starvation_fault``),
    is infra saying so; the sandbox's own limits decide before it, as for any contestant role.
    A contestant fault here names whose end it was;
    what follows is the stage's (``hls.stages.agent_build`` grades the build's last commit).
    Raise ValueError for an outcome without ``split``."""
    split = outcome.split
    if split is None:
        raise ValueError(f"role {outcome.name} has no memory split for the build rule to read")
    evidence = {**_evidence(outcome), "own_oom": str(outcome.own_oom), "job_oom": str(outcome.job_oom),
                **{key: str(value) for key, value in asdict(split).items()}}

    def infra(why: str) -> FaultRecord:
        return FaultRecord(fault_class="infra", reason=f"build sandbox {outcome.name} ended early: {why}",
                           retryable=True, evidence=evidence)

    if outcome.ending == "planned_stop":
        return None
    if outcome.ending == "disk_limit":
        # How the cap was passed, bytes over it or a tree deeper than the walk goes, is the record's.
        how = f": {outcome.detail}" if outcome.detail else ""
        return FaultRecord(fault_class="contestant", reason=f"build sandbox {outcome.name} wrote past its disk cap{how}",
                           retryable=False, evidence=evidence)
    if outcome.ending == "contestant_pids":
        how = f": {outcome.detail}" if outcome.detail else ""
        return FaultRecord(fault_class="contestant", retryable=False, evidence=evidence,
                           reason=f"build sandbox {outcome.name} reached its step's process limit{how}")
    if outcome.ending in _SANDBOX_INFRA:
        return infra(outcome.ending.replace("_", " "))
    if split.oom_at is None:
        starved = starvation_fault(outcome)
        if starved is not None:
            return FaultRecord(fault_class="infra", retryable=True, evidence={**evidence, **starved.evidence},
                               reason=f"build sandbox {outcome.name} ended early: {starved.reason}")
        return infra(f"it ended {outcome.ending} with status {outcome.exit_code} and no OOM recorded")
    if split.oom_by == "kernel" and not (outcome.own_oom is True and outcome.job_oom is False):
        return infra(f"the kernel OOM-killed in it and {_not_own_limit(outcome)}")
    if not within_evidence_age(split.oom_at, split.ended_at):
        return infra(f"its last OOM was {split.ended_at - split.oom_at:.0f} s before it ended")
    if split.sample_at is None or split.harness_bytes is None:
        return infra("no memory split was sampled before its last OOM")
    if not within_evidence_age(split.sample_at, split.oom_at):
        return infra(f"the last memory split before its last OOM was {split.oom_at - split.sample_at:.0f} s older "
                     "than it")
    if split.harness_bytes > split.reserve_bytes:
        return infra(f"the harness held {split.harness_bytes} bytes against its {split.reserve_bytes} byte reserve")
    return FaultRecord(fault_class="contestant", reason=f"build sandbox {outcome.name} ran out of memory while the "
                       "harness stayed within its reserve", retryable=False, evidence=evidence)


@dataclass(frozen=True)
class StepMemory:
    """One step's memory as the record shows it, in bytes: the runner's watch's last and highest
    samples of its footprint, and its cgroup's ``memory.peak`` where the kernel keeps the file,
    which counts page cache too. Each is None when it could not be read."""

    #: The footprint of the watch's last sample.
    last_bytes: int | None
    #: The highest footprint the watch sampled. A spike shorter than its interval is not in it.
    highest_bytes: int | None
    #: ``memory.peak``.
    peak_bytes: int | None


@dataclass(frozen=True)
class SharedMemoryEvidence:
    """What the build rule reads of the roles that share the build sandbox's memory: the in-build
    reference service, a harness role whose memory is oversubscribed on the sandbox's
    (``hls.stages.reference_service``), and the runner in the job's batch step. The job's memory
    is the sandbox's, a harness role's and the runner's; the service's own limit comes on top
    within it. So when the job's own limit is reached while the service is within its own limit,
    the runner within its reserve and the harness roles within theirs, the sandbox held more than
    its memory less the service's: the excess is the session's. Each count or footprint the rule
    needs and could not read leaves the end the harness's (``classify_shared_build_end``)."""

    #: Whether the service's role was running when the sandbox's last OOM was seen: started before
    #: it and still running when the sandbox's end was judged.
    service_running: bool
    #: The service's declared memory, its step's own limit.
    service_limit_bytes: int
    #: Its step's own ``oom`` count, the times it reached its own limit, when the sandbox's end
    #: was judged; None when it could not be read.
    service_ooms: int | None
    #: Whether the runner's watch killed the service at its limit.
    service_breach: bool
    #: The service's memory.
    service: StepMemory
    #: The runner's reserve, ``run.runner_memory_gib``, in bytes.
    runner_reserve_bytes: int
    #: The job's batch step's memory, where the runner runs; None when the watch never sampled it.
    runner: StepMemory | None
    #: The sandbox's memory.
    sandbox: StepMemory
    #: The job's own ``oom`` count when the service started and when the sandbox's end was judged;
    #: None when it could not be read.
    job_ooms_at_start: int | None
    job_ooms_at_end: int | None


def classify_shared_build_end(outcome: RoleOutcome, shared: SharedMemoryEvidence) -> FaultRecord | None:
    """The fault *outcome*, an end of the build sandbox beside the in-build reference service, is.

    The build rule decides every end (``classify_build_end``) but a kernel OOM kill at the job's
    own memory limit, which the sandbox shares with the service whose memory is oversubscribed on
    the sandbox's. That kill is the contestant's only when each of these was read, from *shared*
    and the sandbox's memory split, and held:
    - the service stayed within its own limit: its step's own ``oom`` count 0 and no kill of it
      by the runner's watch;
    - the runner stayed within its reserve: its highest sampled footprint at most it;
    - the harness stayed within its reserve, in the sandbox's split sampled at most
      ``OOM_EVIDENCE_MAX_AGE_SEC`` before the OOM, which was at most that long before the end.
    Anything unread or exceeded leaves it infra, retried. A kill with no service running at the
    OOM keeps the build rule's verdict. The contestant's end is graded as any out-of-memory end
    of the sandbox is (``contestant_oom``). The fault's evidence carries *shared*, so a reader
    sees what was read."""
    decided = classify_build_end(outcome)
    split = outcome.split
    # Only the build rule's verdict on a kernel kill at the job's limit is reopened; an end it
    # judges infra for its own sake (the grant, the launch, the runtime and the like) stands.
    if (decided is None or decided.fault_class != "infra" or outcome.ending in _SANDBOX_INFRA or split is None
            or split.oom_at is None or split.oom_by != "kernel" or outcome.job_oom is not True):
        return decided
    evidence = {**decided.evidence, **_shared_evidence(shared)}

    def infra(why: str) -> FaultRecord:
        return FaultRecord(fault_class="infra", reason=f"build sandbox {outcome.name} ended early: the kernel "
                           f"OOM-killed in it at the job's own limit, and {why}", retryable=True, evidence=evidence)

    if not shared.service_running:
        return decided.model_copy(update={"evidence": evidence})
    if shared.service_ooms is None:
        return infra("whether the reference service reached its own limit could not be read")
    if shared.service_ooms > 0 or shared.service_breach:
        return infra(f"the reference service reached its own limit of {shared.service_limit_bytes} bytes")
    runner = shared.runner.highest_bytes if shared.runner is not None else None
    if runner is None:
        return infra("the runner's memory could not be read")
    if runner > shared.runner_reserve_bytes:
        return infra(f"the runner held {runner} bytes against its {shared.runner_reserve_bytes} byte reserve")
    if not within_evidence_age(split.oom_at, split.ended_at):
        return infra(f"its last OOM was {split.ended_at - split.oom_at:.0f} s before it ended")
    if split.sample_at is None or split.harness_bytes is None:
        return infra("no memory split was sampled before its last OOM")
    if not within_evidence_age(split.sample_at, split.oom_at):
        return infra(f"the last memory split before its last OOM was {split.oom_at - split.sample_at:.0f} s older "
                     "than it")
    if split.harness_bytes > split.reserve_bytes:
        return infra(f"the harness held {split.harness_bytes} bytes against its {split.reserve_bytes} byte reserve")
    return FaultRecord(fault_class="contestant", retryable=False, evidence=evidence,
                       reason=f"build sandbox {outcome.name} ran out of the job's memory, which it shares with the "
                              "in-build reference service, while the service stayed within its own limit, the "
                              "runner within its reserve and the harness within its own")


def _shared_evidence(shared: SharedMemoryEvidence) -> dict[str, str]:
    """*shared* as a fault's evidence, each value a string, ``None`` for one not read."""
    flat: dict[str, str] = {}
    for key, value in asdict(shared).items():
        if isinstance(value, dict):
            flat.update({f"shared.{key}.{k}": str(v) for k, v in value.items()})
        else:
            flat[f"shared.{key}"] = str(value)
    return flat
