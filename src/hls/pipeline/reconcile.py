"""The orchestrator's decisions, as pure functions of the plan and what it observed.

``decide`` takes the latest plan revision and an observation of the store and the scheduler.
It returns the next plan, with every intended submission, rewire and cancel recorded in it,
and the effects to carry out once that plan is written. ``record`` folds the effects'
results into the plan after. So every action is in the plan before it happens, and an
orchestrator that dies anywhere in between reconciles from the plan and the
scheduler and never submits twice.

The rules:
- A submission whose sbatch answer was lost is looked up by its tag, and its runner's status
  file names the job once it started. Unknown counts as alive; only lookups that answer
  "none" from the first such answer until ``ABSENT_AFTER_SEC`` later make it absent. Tags of
  absent submissions are looked up on, and a job that turns up under one is cancelled by id.
- An allocation is done when every one of its stages has a manifest, whatever its job's
  state; OUT_OF_MEMORY and the like are evidence, never success or failure on their own.
- An allocation fails when its job has ended without publishing every stage, or when its
  job runs but its heartbeat is older than ``allocation_dead_after_sec``; then its job is
  cancelled by id.
- A failed stage with a contestant fault ends the instance "failed": that is the contestant's
  scored result, and nothing of it runs again. It stands over an
  ``hls cancel`` asked for later, which ends any other instance "cancelled".
- A failed stage with an infra fault, retries left and retries enabled is retried in a new
  allocation that starts at that stage, after the backoff, on the failed allocation's
  partition. Its dependents are repointed to the new allocation, so nothing waits on a
  finished job. Otherwise (the fault is not retryable, retries are disabled or used up) the
  instance is void.
- An allocation that waits for others (``waits_for``) is not submitted until every stage of
  each has published, and is then submitted with no Slurm dependency on them. A failed
  allocation whose retry published the rest counts as published.
- A retry keeps what its failed allocation waited for (``depends_on``, ``waits_for``) and how it
  was submitted (``singleton``). It excludes the nodes its failed allocation excluded and, when
  the runner's grant check failed the stage on a node (its fault's ``check="grant"`` and
  ``node`` evidence), that node too, so it does not land again on a node short of the request.
- A plan that grows (``Plan.grower``): once every stage has published, the orchestrator asks
  its grower (``hls.pipeline.growth.grow``) and the observation carries the answer. What the
  grower adds is appended, and its allocations' submissions recorded, in the same revision; a
  refusal ends the instance void; nothing more ends it succeeded. A growing plan is never
  decided succeeded without the answer.
- A dependent is submitted with ``afterok`` only on a predecessor job that is pending or
  running; a predecessor that already published everything means no dependency at all. A
  pending dependent that waits for a job that ended badly, once its predecessor published
  everything, is cancelled and submitted again without the dependency.
- A rewire that scontrol refuses for a reason other than the dependency is asked again at the
  next pass, and after ``MAX_REWIRE_REFUSALS`` refusals the job is cancelled and submitted
  again. While a rewire is intended, only Slurm's pending reason says the dependent is stuck,
  because the rewire may have been applied and its answer lost.
- An sbatch refusal that is not about the dependency voids the instance at once (``record``),
  since asking again would only repeat it. Three submissions of one allocation in a row that
  were lost, or refused about their dependency, void it too.
- A void, failed or cancelled instance cancels every live job of it by id, on every pass until
  none is left, including dependents that would otherwise pend forever as
  DependencyNeverSatisfied.
- When the instance succeeds, jobs of it that still wait in the queue are cancelled by id and
  running ones are left to end.
- An instance is finished when no cancel is outstanding, no job of it is unaccounted for (a
  state unknown, an sbatch answer unresolved, a stray not seen ended), and the tags of
  submissions declared absent were looked up for one more window."""

from __future__ import annotations

import signal
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from hls.pipeline.growth import GrowthComplete, GrowthRefused, PlanGrowth
from hls.pipeline.plan import Allocation, AllocationStatus, Failure, Outcome, Plan, Submission, submission_tag
from hls.pipeline.planner import allocation_request, socket_tasks
from hls.settings import SiteSettings
from hls.substrate.base import FaultRecord, JobInfo, JobState
from hls.substrate.factory import substrate

#: Seconds a submission whose sbatch answer was lost must stay unfound by its tag, from the
#: first lookup that found none, before it counts as never submitted.
ABSENT_AFTER_SEC = 120.0
#: Submissions of one allocation, refused or lost one after the other, that void the instance.
MAX_ABSENT_IN_A_ROW = 3
#: Rewires refused for a reason other than the dependency, after which the job is submitted again.
MAX_REWIRE_REFUSALS = 3
_ALIVE = (JobState.PENDING, JobState.RUNNING)
_JOB_ENDINGS = {
    JobState.NODE_FAIL: "the node failed",
    JobState.PREEMPTED: "the job was preempted",
    JobState.TIMEOUT: "the job reached its time limit",
    JobState.DEADLINE: "the job reached its deadline",
    JobState.BOOT_FAIL: "the node failed to boot",
    JobState.CANCELLED: "the job was cancelled outside hls",
    JobState.OUT_OF_MEMORY: "the batch step ran out of memory",
    JobState.COMPLETED: "the job completed without publishing every stage",
}


@dataclass(frozen=True)
class Observation:
    """What the orchestrator saw in one poll."""

    #: When.
    now: float
    #: Stages with a published manifest.
    published: frozenset[str]
    #: The scheduler's view of every job id recorded in the plan.
    jobs: Mapping[str, JobInfo]
    #: Tag lookups for submissions recorded without a job id; None where the scheduler did
    #: not answer.
    tags: Mapping[str, list[str] | None] = field(default_factory=dict)
    #: The runner status of each (allocation, submission) that has written one.
    statuses: Mapping[tuple[int, int], AllocationStatus] = field(default_factory=dict)
    #: Whether ``hls cancel`` asked for the instance to stop.
    cancel_requested: bool = False
    #: What the plan's grower adds now that every stage has published (``hls.pipeline.growth.grow``);
    #: None when it was not asked: the plan names no grower, a stage is still unpublished, or the
    #: instance ended or was asked to stop.
    growth: PlanGrowth | GrowthRefused | GrowthComplete | None = None
    #: For each (allocation, submission) whose job has ended, the fault, infra and never retried,
    #: that the attempt its runner's status names declared any end of it is
    #: (``AttemptStatus.no_retry``), when that attempt's stage has not published.
    declared: Mapping[tuple[int, int], FaultRecord] = field(default_factory=dict)

    def job(self, job_id: str) -> JobInfo:
        return self.jobs.get(job_id, JobInfo(job_id, JobState.UNKNOWN))


@dataclass(frozen=True)
class Submit:
    """Run sbatch for submission *n* of allocation *allocation*, as recorded in the plan."""

    allocation: int
    n: int


@dataclass(frozen=True)
class Rewire:
    """Make the pending job of submission *n* of *allocation* wait for *to_job*."""

    allocation: int
    n: int
    to_job: str


@dataclass(frozen=True)
class Cancel:
    """Cancel the job of submission *n* of *allocation*, by its recorded id."""

    allocation: int
    n: int
    job_id: str


Effect = Submit | Rewire | Cancel


@dataclass(frozen=True)
class Submitted:
    """What sbatch said. *job_id* when it answered with one; otherwise *refused* says whether
    it refused (no job exists) or its answer was lost (the job may exist), and *dependency*
    whether the refusal was about the dependency."""

    allocation: int
    n: int
    job_id: str | None
    refused: bool = False
    #: The refusal was a DependencyRefused: the predecessor ended meanwhile.
    dependency: bool = False
    error: str | None = None
    #: The command line sbatch was given, as recorded (``Submission.argv``); None when the
    #: submission failed before its line was known.
    argv: tuple[str, ...] | None = None


@dataclass(frozen=True)
class Rewired:
    """What scontrol said: *ok*, refused (the job cannot be repointed, or *dependency*: the
    job to wait for is not known) or unknown."""

    allocation: int
    n: int
    to_job: str
    ok: bool
    refused: bool = False
    #: The refusal was a DependencyRefused: the job to wait for is no longer known.
    dependency: bool = False
    error: str | None = None


@dataclass(frozen=True)
class Cancelled:
    """What scancel said: *ok*, or the controller did not answer. *job_id* is set when the job
    is not the submission's own, a stray that carried its tag."""

    allocation: int
    n: int
    ok: bool
    error: str | None = None
    job_id: str | None = None


Result = Submitted | Rewired | Cancelled


@dataclass(frozen=True)
class Decision:
    """The next plan, the effects to carry out after writing it, and events to log."""

    #: The plan with every intent recorded.
    plan: Plan
    #: Effects, in order.
    effects: list[Effect]
    #: (kind, fields) pairs for the orchestrator's event log.
    events: list[tuple[str, dict]]


class _Draft:
    """A plan being changed, with the effects and events the changes call for."""

    def __init__(self, plan: Plan, now: float) -> None:
        self.plan = plan
        self.now = now
        self.effects: list[Effect] = []
        self.events: list[tuple[str, dict]] = []

    def put(self, alloc: Allocation) -> Allocation:
        self.plan = self.plan.with_allocation(alloc)
        return alloc

    def put_sub(self, k: int, sub: Submission) -> None:
        alloc = self.plan.allocation(k)
        subs = [sub if s.n == sub.n else s for s in alloc.submissions]
        if all(s.n != sub.n for s in alloc.submissions):
            subs.append(sub)
        self.put(alloc.model_copy(update={"submissions": subs}))

    def event(self, kind: str, **fields) -> None:
        self.events.append((kind, fields))

    def decision(self) -> Decision:
        return Decision(self.plan, self.effects, self.events)


def _done(alloc: Allocation, published: frozenset[str]) -> bool:
    return all(s in published for s in alloc.stages)


def _runner_failed(exit_code: str | None) -> str:
    """What a FAILED job's exit code, "<status>:<signal>" as sacct prints it, says of how its
    runner ended."""
    status, _, number = (exit_code or "").partition(":")
    if number.isdecimal() and int(number):
        try:
            return f"the runner was killed by signal {number} ({signal.Signals(int(number)).name})"
        except ValueError:
            return f"the runner was killed by signal {number}"
    if status.isdecimal() and int(status):
        return f"the runner exited with status {status}"
    return "the job ended FAILED" + (f" with exit code {exit_code}" if exit_code else "")


def job_fault(info: JobInfo, status: AllocationStatus | None, declared: FaultRecord | None) -> FaultRecord:
    """Why an allocation job that ended without publishing every stage failed. When its runner
    wrote a fault, that fault, whoever it charges: a stage that failed with a contestant fault
    leaves the contestant's. Otherwise, when its running attempt had declared the fault any end
    of it is (*declared*, ``AttemptStatus.no_retry``: the agent build past its commit point), that
    fault, never retried: a runner killed outright after the declaration, by the job's OOM kill or
    a node failure, must not have its stage's work, a whole agent session, run again. Otherwise
    an infra fault from what the job's final state and exit code say."""
    evidence = {"job_state": info.state.value, "exit_code": info.exit_code or ""}
    if status is not None and status.fault is not None:
        return status.fault.model_copy(update={"evidence": {**status.fault.evidence, **evidence}})
    if declared is not None:
        return declared.model_copy(update={"evidence": {**declared.evidence, **evidence}})
    if info.state is JobState.FAILED:
        reason = _runner_failed(info.exit_code)
    else:
        reason = _JOB_ENDINGS.get(info.state, f"the job ended {info.state.value}")
    return FaultRecord(fault_class="infra", reason=f"job {info.job_id}: {reason}", retryable=True, evidence=evidence)


def _jobs_found(obs: Observation, k: int, sub: Submission) -> list[str]:
    """The jobs that belong to a submission whose job id was never recorded: the one its
    runner's status names, then any others the tag lookup found."""
    tagged = obs.tags.get(sub.tag) or []
    status = obs.statuses.get((k, sub.n))
    if status is None:
        return list(tagged)
    return [status.job_id, *(j for j in tagged if j != status.job_id)]


def _resolve_unknown_submissions(d: _Draft, obs: Observation) -> None:
    """Find the job of each submission recorded without one, or note that none carries its tag."""
    for alloc in d.plan.allocations:
        for sub in alloc.submissions:
            if sub.job_id is not None:
                continue
            found = _jobs_found(obs, alloc.index, sub)
            if found:
                first, *strays = found
                # A job under the tag of an absent or cancelled submission turned up late: it is
                # cancelled like any other job nobody wants.
                status = "submitted" if sub.status == "intended" else "cancel_intended"
                d.put_sub(alloc.index, sub.model_copy(update={
                    "job_id": first, "submitted_at": obs.now, "status": status, "missing_since": None,
                    "strays": strays}))
                d.event("job_found", allocation=alloc.index, submission=sub.n, job_id=first, was=sub.status)
                for stray in strays:
                    d.event("stray_job", allocation=alloc.index, submission=sub.n, job_id=stray)
            elif sub.status in ("intended", "cancel_intended") and sub.tag in obs.tags:
                if obs.tags[sub.tag] is None:
                    # No answer: the lookups that found none do not count any more, so an outage
                    # cannot use the window up. It starts again with the next empty answer.
                    if sub.missing_since is not None:
                        d.put_sub(alloc.index, sub.model_copy(update={"missing_since": None}))
                elif sub.missing_since is None:
                    d.put_sub(alloc.index, sub.model_copy(update={"missing_since": obs.now}))
                elif obs.now - sub.missing_since >= ABSENT_AFTER_SEC:
                    gone = "absent" if sub.status == "intended" else "cancelled"
                    note = (f"no job carries tag {sub.tag}, {ABSENT_AFTER_SEC:.0f}s after the first lookup that "
                            "found none")
                    d.put_sub(alloc.index, sub.model_copy(update={
                        "status": gone, "error": f"{sub.error} ({note})" if sub.error else note}))
                    d.event("submission_absent", allocation=alloc.index, submission=sub.n)


def _pending_strays(d: _Draft, obs: Observation) -> None:
    """Cancel by id every stray job recorded on a submission, each until it is seen ended."""
    for alloc in d.plan.allocations:
        for sub in alloc.submissions:
            alive = [j for j in sub.strays if not obs.job(j).state.terminal]
            if alive != sub.strays:
                d.put_sub(alloc.index, sub.model_copy(update={"strays": alive}))
            d.effects.extend(Cancel(alloc.index, sub.n, j) for j in alive)


def _pending_cancels(d: _Draft, obs: Observation) -> None:
    for alloc in d.plan.allocations:
        for sub in alloc.submissions:
            if sub.status != "cancel_intended" or sub.job_id is None:
                continue
            if obs.job(sub.job_id).state.terminal:
                d.put_sub(alloc.index, sub.model_copy(update={"status": "cancelled"}))
            else:
                d.effects.append(Cancel(alloc.index, sub.n, sub.job_id))


def _cancel_everything(d: _Draft, obs: Observation) -> None:
    """Mark every live submission whose job is not over for cancelling, and cancel those with
    an id; the others are looked up by tag first."""
    for alloc in d.plan.allocations:
        for sub in alloc.submissions:
            if not sub.live:
                continue
            if sub.job_id is not None and obs.job(sub.job_id).state.terminal:
                continue
            d.put_sub(alloc.index, sub.model_copy(update={"status": "cancel_intended"}))
            if sub.job_id is not None:
                d.effects.append(Cancel(alloc.index, sub.n, sub.job_id))


def _cancel_queued(d: _Draft, obs: Observation) -> None:
    """After success: cancel by id every job of a live submission that still waits in the queue,
    which would run stages again. Running ones end by themselves."""
    for alloc in d.plan.allocations:
        for sub in alloc.submissions:
            if sub.live and sub.job_id is not None and obs.job(sub.job_id).state is JobState.PENDING:
                d.put_sub(alloc.index, sub.model_copy(update={"status": "cancel_intended"}))
                d.effects.append(Cancel(alloc.index, sub.n, sub.job_id))


def _drop_unsent(d: _Draft) -> None:
    """A submission this decision marked for cancelling was never sent: record it cancelled and
    leave its Submit out."""
    kept: list[Effect] = []
    for effect in d.effects:
        if isinstance(effect, Submit):
            sub = next(s for s in d.plan.allocation(effect.allocation).submissions if s.n == effect.n)
            if sub.status == "cancel_intended":
                d.put_sub(effect.allocation, sub.model_copy(update={
                    "status": "cancelled", "error": "never submitted: the instance ended in the same pass"}))
                d.event("submit_dropped", allocation=effect.allocation, submission=effect.n)
                continue
        kept.append(effect)
    d.effects = kept


def _end(d: _Draft, obs: Observation, status: str, reason: str, fault: FaultRecord | None = None) -> None:
    d.plan = d.plan.model_copy(update={"outcome": Outcome(status=status, reason=reason, at=obs.now, fault=fault)})
    d.event("instance_ended", status=status, reason=reason, fault_class=None if fault is None else fault.fault_class)


def _retries_used(plan: Plan, stage: str) -> int:
    return sum(1 for a in plan.allocations if a.retry_of is not None and a.stages[0] == stage)


def _excluded(alloc: Allocation, fault: FaultRecord, site: SiteSettings) -> list[str]:
    """The nodes a retry of *alloc* must not land on: those *alloc* excluded, then the node the
    runner's grant check found short of the request, which *fault*'s evidence names
    (``check="grant"``, ``node``), each once. A fault of any other kind, or a node name the
    scheduler of *site*'s substrate would not take, adds nothing."""
    excluded = list(alloc.placement.exclude)
    node = fault.evidence.get("node")
    if fault.evidence.get("check") != "grant" or node is None or node in excluded:
        return excluded
    try:
        substrate(site).check_node_name(node)
    except ValueError:
        return excluded
    return [*excluded, node]


def _after_failure(d: _Draft, obs: Observation, alloc: Allocation) -> None:
    """End the instance failed for a contestant fault; for an infra fault, retry *alloc*'s
    failed stage in a new allocation, wait out the backoff, or void."""
    if any(a.retry_of == alloc.index for a in d.plan.allocations):
        return
    failure = alloc.failure
    assert failure is not None
    if failure.fault.fault_class == "contestant":
        _end(d, obs, "failed", f"{failure.stage}: {failure.fault.reason}", failure.fault)
        return
    policy = d.plan.stage(failure.stage).retry
    used = _retries_used(d.plan, failure.stage)
    if not failure.fault.retryable:
        why = "not retryable"
    elif not policy.enabled:
        why = "retries are disabled"
    elif used >= (policy.max or 0):
        why = f"retries exhausted ({used} of {policy.max})"
    else:
        if obs.now < failure.at + policy.backoff_sec:
            return
        stages = alloc.stages[alloc.stages.index(failure.stage):]
        plans = [d.plan.stage(s) for s in stages]
        kinds = {s.shape.gpu_kind for s in d.plan.stages if s.shape.gpu_kind is not None}
        partition = alloc.placement.partition
        tasks = socket_tasks(plans, d.plan.settings, partition=partition)
        request = allocation_request(plans, d.plan.settings, gpu_kind=next(iter(kinds), None),
                                     partition=partition, socket_tasks=tasks)
        # The retry waits for what the failed allocation waited for, and is submitted as it was.
        # It asks for no more time than the allocation it retries, so a QOS holds it, perhaps one
        # of a shorter limit.
        site = d.plan.settings.site
        exclude = _excluded(alloc, failure.fault, site)
        qos = substrate(site).allocation_qos(request.time_sec, site, partition=partition)
        placement = substrate(site).placement(partition=partition, qos=qos, socket_tasks=tasks, exclude=exclude)
        retry = d.put(Allocation(index=len(d.plan.allocations), stages=stages, request=request,
                                 placement=placement, depends_on=alloc.depends_on, waits_for=list(alloc.waits_for),
                                 singleton=alloc.singleton, retry_of=alloc.index))
        for dep in d.plan.allocations:
            if dep.depends_on == alloc.index and dep.index != retry.index:
                d.put(dep.model_copy(update={"depends_on": retry.index}))
        d.event("retry", stage=failure.stage, failed_allocation=alloc.index, allocation=retry.index,
                retry=used + 1, of=policy.max, reason=failure.fault.reason)
        if exclude != alloc.placement.exclude:
            d.event("node_excluded", allocation=retry.index, nodes=exclude, reason=failure.fault.reason)
        return
    _end(d, obs, "void", f"{failure.stage}: {failure.fault.reason}; {why}", failure.fault)


def _submit(d: _Draft, alloc: Allocation, after: str | None) -> None:
    n = len(alloc.submissions) + 1
    sub = Submission(n=n, tag=submission_tag(d.plan.instance, alloc.index, n), intended_at=d.now, after_job=after,
                     status="intended")
    d.put_sub(alloc.index, sub)
    d.effects.append(Submit(alloc.index, n))
    d.event("submit_intended", allocation=alloc.index, submission=n, after_job=after)


def _predecessor_job(d: _Draft, obs: Observation, alloc: Allocation) -> tuple[bool, str | None]:
    """Whether *alloc* may be submitted now, and the job id it must wait for."""
    if alloc.depends_on is None:
        return True, None
    pred = d.plan.allocation(alloc.depends_on)
    if _done(pred, obs.published):
        return True, None
    cur = pred.current()
    if pred.failure is not None or cur is None or cur.job_id is None:
        return False, None
    if obs.job(cur.job_id).state in _ALIVE:
        return True, cur.job_id
    return False, None


def _absent_in_a_row(alloc: Allocation) -> list[Submission]:
    """The submissions at the end of *alloc*'s list that no job ever came from, oldest first."""
    run: list[Submission] = []
    for sub in reversed(alloc.submissions):
        if sub.status != "absent":
            break
        run.append(sub)
    return run[::-1]


def _step(d: _Draft, obs: Observation, alloc: Allocation) -> None:
    """Move one allocation forward."""
    if _done(alloc, obs.published):
        return
    if alloc.failure is not None:
        _after_failure(d, obs, alloc)
        return
    cur = alloc.current()
    if cur is None:
        if any(not _done(d.plan.allocation(w), obs.published) for w in alloc.waits_for):
            return
        ready, after = _predecessor_job(d, obs, alloc)
        if not ready:
            return
        absent = _absent_in_a_row(alloc)
        if len(absent) >= MAX_ABSENT_IN_A_ROW:
            last = f"; last error: {absent[-1].error}" if absent[-1].error else ""
            numbers = ", ".join(str(s.n) for s in absent)
            _end(d, obs, "void", f"allocation {alloc.index}: submissions {numbers} were refused or lost{last}")
            return
        _submit(d, alloc, after)
        return
    if cur.job_id is None:
        return
    info = obs.job(cur.job_id)
    status = obs.statuses.get((alloc.index, cur.n))
    if info.state is JobState.PENDING and alloc.depends_on is not None:
        pred = d.plan.allocation(alloc.depends_on)
        pred_cur = pred.current()
        target = pred_cur.job_id if pred_cur is not None else None
        # A finished predecessor is never a target: Slurm forgets finished jobs within minutes.
        if cur.rewire_refusals >= MAX_REWIRE_REFUSALS:
            # scontrol keeps refusing: submit the job again, waiting for what is right by then.
            d.put_sub(alloc.index, cur.model_copy(update={"status": "cancel_intended"}))
            d.effects.append(Cancel(alloc.index, cur.n, cur.job_id))
            d.event("rewire_given_up", allocation=alloc.index, submission=cur.n, refusals=cur.rewire_refusals)
        elif (target is not None and target != cur.after_job and not _done(pred, obs.published)
                and obs.job(target).state in _ALIVE):
            refusals = cur.rewire_refusals if cur.rewire_to == target else 0
            d.put_sub(alloc.index, cur.model_copy(update={"rewire_to": target, "rewire_refusals": refusals}))
            d.effects.append(Rewire(alloc.index, cur.n, target))
            d.event("rewire_intended", allocation=alloc.index, submission=cur.n, to_job=target)
        elif _done(pred, obs.published) and cur.after_job is not None:
            # The predecessor published everything, so nothing is left to wait for; but the job
            # this one waits for ended badly and Slurm keeps it pending forever. Submit it again
            # without the dependency. While a rewire is intended Slurm may have applied it, so
            # after_job may be stale and only the pending reason is trusted.
            after = obs.job(cur.after_job)
            ended_badly = after.state.terminal and after.state is not JobState.COMPLETED
            if info.reason == "DependencyNeverSatisfied" or (cur.rewire_to is None and ended_badly):
                d.put_sub(alloc.index, cur.model_copy(update={"status": "cancel_intended"}))
                d.effects.append(Cancel(alloc.index, cur.n, cur.job_id))
                d.event("dependency_dropped", allocation=alloc.index, submission=cur.n, after_job=cur.after_job)
        return
    stage = next(s for s in alloc.stages if s not in obs.published)
    if info.state.terminal:
        fault = job_fault(info, status, obs.declared.get((alloc.index, cur.n)))
    elif info.state is JobState.RUNNING:
        last = status.heartbeat_at if status is not None else info.started_at
        silent = None if last is None else obs.now - last
        if silent is None or silent <= d.plan.settings.run.allocation_dead_after_sec:
            return
        fault = FaultRecord(fault_class="infra", retryable=True, reason=f"job {cur.job_id}: no heartbeat for "
                            f"{silent:.0f}s while it ran", evidence={"job_state": info.state.value})
        d.put_sub(alloc.index, cur.model_copy(update={"status": "cancel_intended"}))
        d.effects.append(Cancel(alloc.index, cur.n, cur.job_id))
    else:
        return
    alloc = d.put(d.plan.allocation(alloc.index).model_copy(
        update={"failure": Failure(at=obs.now, stage=stage, fault=fault)}))
    d.event("allocation_failed", allocation=alloc.index, stage=stage, reason=fault.reason)
    _after_failure(d, obs, alloc)


def _contestant_failure(d: _Draft, obs: Observation) -> bool:
    """End the instance failed, and return True, when one of its allocations failed with a
    contestant fault, recorded already or seen now in a job that ended: a contestant's scored
    result stands over a cancel asked for later, such as one asked for while the orchestrator's
    passes failed between deciding it and writing it."""
    for alloc in d.plan.allocations:
        if _done(alloc, obs.published) or any(a.retry_of == alloc.index for a in d.plan.allocations):
            continue
        failure = alloc.failure
        cur = alloc.current()
        ended = cur is not None and cur.job_id is not None and obs.job(cur.job_id).state.terminal
        if failure is None and ended:
            fault = job_fault(obs.job(cur.job_id), obs.statuses.get((alloc.index, cur.n)),
                              obs.declared.get((alloc.index, cur.n)))
            if fault.fault_class == "contestant":
                stage = next(s for s in alloc.stages if s not in obs.published)
                failure = Failure(at=obs.now, stage=stage, fault=fault)
                d.put(alloc.model_copy(update={"failure": failure}))
                d.event("allocation_failed", allocation=alloc.index, stage=stage, reason=fault.reason)
        if failure is not None and failure.fault.fault_class == "contestant":
            _end(d, obs, "failed", f"{failure.stage}: {failure.fault.reason}", failure.fault)
            return True
    return False


def _all_published(d: _Draft, obs: Observation) -> None:
    """Every stage of the plan published. A plan that does not grow ends succeeded. A growing
    plan takes what its grower adds, the stages and allocations appended and the allocations
    submitted in this decision; ends void when its grower refused; ends succeeded when its
    grower adds nothing more. Raise ValueError when its grower was not asked: an observation
    without the answer must never end a growing plan."""
    if d.plan.grower is None:
        _end(d, obs, "succeeded", "every stage published")
        return
    growth = obs.growth
    if growth is None:
        raise ValueError(f"{d.plan.instance} grows by {d.plan.grower!r} and was decided without asking its grower")
    if isinstance(growth, GrowthRefused):
        _end(d, obs, "void", f"growth refused: {growth.reason}")
        return
    if isinstance(growth, GrowthComplete):
        _end(d, obs, "succeeded", "every stage published")
        return
    first = len(d.plan.allocations)
    numbers = [a.index for a in growth.allocations]
    if numbers != list(range(first, first + len(numbers))):
        raise ValueError(f"{d.plan.instance}: the grown allocations are numbered {numbers}, not from {first}")
    d.plan = d.plan.model_copy(update={"stages": [*d.plan.stages, *growth.stages],
                                       "allocations": [*d.plan.allocations, *growth.allocations]})
    d.event("plan_grown", stages=[s.name for s in growth.stages], allocations=numbers, reason=growth.reason)
    for k in numbers:
        _step(d, obs, d.plan.allocation(k))


def decide(plan: Plan, obs: Observation) -> Decision:
    """The next plan and the effects to carry out once it is written."""
    d = _Draft(plan, obs.now)
    _resolve_unknown_submissions(d, obs)
    _pending_strays(d, obs)
    _pending_cancels(d, obs)
    if d.plan.outcome is None:
        if obs.cancel_requested:
            if not _contestant_failure(d, obs):
                _end(d, obs, "cancelled", "cancelled by hls cancel")
        elif all(s.name in obs.published for s in d.plan.stages):
            _all_published(d, obs)
        else:
            # A retry added during the pass is stepped too, so it is submitted at once.
            k = 0
            while k < len(d.plan.allocations) and d.plan.outcome is None:
                _step(d, obs, d.plan.allocation(k))
                k += 1
    if d.plan.outcome is not None:
        if d.plan.outcome.status == "succeeded":
            _cancel_queued(d, obs)
        else:
            # A void or cancelled instance stops every job it still has, on every pass until
            # none is left.
            _cancel_everything(d, obs)
    _drop_unsent(d)
    return d.decision()


def record(plan: Plan, results: Sequence[Result], now: float) -> Plan:
    """*plan* with what the scheduler answered to each effect."""
    d = _Draft(plan, now)
    for r in results:
        sub = next(s for s in d.plan.allocation(r.allocation).submissions if s.n == r.n)
        if isinstance(r, Submitted):
            if r.argv is not None:
                sub = sub.model_copy(update={"argv": list(r.argv)})
            if r.job_id is not None:
                sub = sub.model_copy(update={"status": "submitted", "job_id": r.job_id, "submitted_at": now,
                                             "error": None})
            elif r.refused:
                sub = sub.model_copy(update={"status": "absent", "error": r.error})
                # A refusal about the dependency means the predecessor ended meanwhile, and the
                # next decision resubmits; any other refusal would only repeat.
                if not r.dependency and d.plan.outcome is None:
                    d.plan = d.plan.model_copy(update={"outcome": Outcome(
                        status="void", reason=f"sbatch refused allocation {r.allocation}: {r.error}", at=now)})
            else:
                sub = sub.model_copy(update={"error": r.error})
        elif isinstance(r, Rewired):
            if r.ok:
                sub = sub.model_copy(update={"after_job": r.to_job, "rewire_to": None, "rewire_refusals": 0,
                                             "error": None})
            elif r.refused and r.dependency:
                # The job to wait for is gone: submit this one again, with whatever is right then.
                sub = sub.model_copy(update={"status": "cancel_intended", "rewire_to": None, "rewire_refusals": 0,
                                             "error": r.error})
            elif r.refused:
                # Refused for another reason: the intent stays and the next pass asks again, which
                # changes no plan; the count ends it.
                sub = sub.model_copy(update={"rewire_refusals": sub.rewire_refusals + 1, "error": r.error})
            else:
                sub = sub.model_copy(update={"error": r.error})
        elif r.job_id is not None and r.job_id != sub.job_id:
            # A stray's answer says nothing about the submission's own job.
            continue
        elif r.ok:
            sub = sub.model_copy(update={"status": "cancelled", "error": None})
        else:
            sub = sub.model_copy(update={"error": r.error})
        d.put_sub(r.allocation, sub)
    return d.plan


def _may_still_turn_up(sub: Submission, now: float) -> bool:
    """Whether a job could still appear under the tag of a submission that has none: sbatch may
    have been slow, so its tag is looked up for one more window after the lookups that found
    none ran a full window."""
    return (sub.job_id is None and sub.missing_since is not None
            and now - sub.missing_since < 2 * ABSENT_AFTER_SEC)


def finished(plan: Plan, obs: Observation) -> bool:
    """Whether nothing is left to do: the instance ended and every job of it is accounted for.
    No cancel or stray may be outstanding, no live submission may lack a job id or have a job in
    an unknown state, and unless the instance succeeded no job of it may still run."""
    if plan.outcome is None:
        return False
    for alloc in plan.allocations:
        for s in alloc.submissions:
            if s.status == "cancel_intended" or s.strays or _may_still_turn_up(s, obs.now):
                return False
            if s.live:
                # A live submission without a job id is an sbatch answer nobody resolved.
                state = JobState.UNKNOWN if s.job_id is None else obs.job(s.job_id).state
                if state is JobState.UNKNOWN or (plan.outcome.status != "succeeded" and not state.terminal):
                    return False
    return True
