"""An in-memory Slurm for orchestrator tests, with these behaviors of Slurm's:
dependents of a failed job pend as DependencyNeverSatisfied until rewired or cancelled, and
a dependency on a purged job id is refused."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from hls.substrate.base import (DependencyRefused, JobInfo, JobState, Refused, Scheduler, SchedulerUnavailable,
                                SubmitRequest, SubmitUnknown)
from hls.substrate.slurm.scheduler import submit_argv


@dataclass
class FakeJob:
    job_id: str
    request: SubmitRequest
    after: str | None
    state: JobState = JobState.PENDING
    reason: str | None = None
    started_at: float | None = None
    purged: bool = False


@dataclass
class FakeScheduler(Scheduler):
    jobs: dict[str, FakeJob] = field(default_factory=dict)
    next_id: int = 100
    #: The controller does not answer any call.
    down: bool = False
    #: How the next submit misbehaves: "lose_answer" creates the job then raises
    #: SubmitUnknown, "refuse" raises Refused with *refusal*.
    submit_fault: str | None = None
    refusal: str = "Batch job submission failed: Invalid account or account/partition combination"
    #: Called after a job is created and before submit returns, to simulate a crash there.
    after_create: Callable[[FakeJob], None] | None = None
    test_only_refusal: str | None = None
    #: squeue does not answer tag lookups, though every other call works.
    tags_down: bool = False
    #: When set, every pending job reports this reason instead of the one it earned, as a site
    #: whose pending reasons are not the ones Slurm documents would.
    pending_reason: str | None = None
    calls: list[tuple] = field(default_factory=list)

    # Scheduler

    def submit(self, request: SubmitRequest) -> str:
        self.calls.append(("submit", request.tag, request.after))
        if self.down:
            raise SubmitUnknown("Socket timed out on send/recv operation")
        fault, self.submit_fault = self.submit_fault, None
        if fault == "refuse":
            raise Refused(self.refusal)
        if request.after is not None:
            parent = self.jobs.get(request.after)
            if parent is None or parent.purged:
                raise DependencyRefused("Batch job submission failed: Job dependency problem")
        job = FakeJob(str(self.next_id), request, request.after)
        self.next_id += 1
        self.jobs[job.job_id] = job
        self._wire(job)
        if self.after_create is not None:
            self.after_create(job)
        if fault == "lose_answer":
            raise SubmitUnknown("sbatch timed out")
        return job.job_id

    def submit_argv(self, request: SubmitRequest) -> tuple[str, ...]:
        """The line Slurm's adapter would run for *request*."""
        return submit_argv(request)

    def test_only(self, request: SubmitRequest) -> str:
        self.calls.append(("test_only", request.tag))
        if self.test_only_refusal:
            raise Refused(self.test_only_refusal)
        return "sbatch: Job 1 to start at 2026-10-02T10:00:00"

    def find_by_tags(self, tags: Sequence[str]) -> dict[str, list[str] | None]:
        self.calls.append(("find_by_tags", tuple(tags)))
        return {t: self._lookup(t) for t in dict.fromkeys(tags)}

    def _lookup(self, tag: str) -> list[str] | None:
        if self.down or self.tags_down:
            return None
        return sorted((j.job_id for j in self.jobs.values() if j.request.tag == tag and not j.purged), key=int)

    def states(self, job_ids: Sequence[str]) -> dict[str, JobInfo]:
        self.calls.append(("states", tuple(job_ids)))
        if self.down:
            return {j: JobInfo(j, JobState.UNKNOWN) for j in job_ids}
        out = {}
        for j in job_ids:
            job = self.jobs.get(j)
            if job is None:
                out[j] = JobInfo(j, JobState.UNKNOWN)
                continue
            reason = job.reason
            if self.pending_reason is not None and job.state is JobState.PENDING:
                reason = self.pending_reason
            out[j] = JobInfo(j, job.state, reason=reason, started_at=job.started_at)
        return out

    def rewire(self, job_id: str, after: str) -> None:
        self.calls.append(("rewire", job_id, after))
        if self.down:
            raise SchedulerUnavailable("Socket timed out")
        job = self.jobs[job_id]
        target = self.jobs.get(after)
        if job.state is not JobState.PENDING:
            raise Refused(f"Job {job_id} is no longer pending execution")
        if target is None or target.purged:
            raise DependencyRefused(f"Job dependency problem for job {job_id}")
        job.after = after
        self._wire(job)

    def cancel(self, job_id: str) -> None:
        self.calls.append(("cancel", job_id))
        if self.down:
            raise SchedulerUnavailable("Socket timed out")
        job = self.jobs.get(job_id)
        if job is not None and not job.state.terminal:
            job.state = JobState.CANCELLED

    # test controls

    def ready(self, job_id: str) -> bool:
        """Whether pending job *job_id* may start now: the job it waits for completed, and, for a
        singleton job, no other job of its name runs, as Slurm's ``singleton`` holds it."""
        job = self.jobs[job_id]
        parent = self.jobs.get(job.after) if job.after else None
        if parent is not None and parent.state is not JobState.COMPLETED:
            return False
        if not job.request.singleton:
            return True
        name = _job_name(job.request)
        return not any(other.job_id != job_id and _job_name(other.request) == name
                       and other.state is JobState.RUNNING for other in self.jobs.values())

    def _wire(self, job: FakeJob) -> None:
        parent = self.jobs.get(job.after) if job.after else None
        failed = parent is not None and parent.state.terminal and parent.state is not JobState.COMPLETED
        job.reason = "DependencyNeverSatisfied" if failed else ("Dependency" if parent else None)

    def start(self, job_id: str, at: float = 0.0) -> None:
        job = self.jobs[job_id]
        if job.request.singleton and not self.ready(job_id):
            raise AssertionError(f"job {job_id} is singleton and another job named {_job_name(job.request)} runs")
        job.state, job.reason, job.started_at = JobState.RUNNING, None, at

    def finish(self, job_id: str, state: JobState = JobState.COMPLETED) -> None:
        self.jobs[job_id].state = state
        for other in self.jobs.values():
            if other.after == job_id and other.state is JobState.PENDING:
                self._wire(other)

    def purge(self, job_id: str) -> None:
        self.jobs[job_id].purged = True

    def submits(self) -> list[tuple]:
        return [c for c in self.calls if c[0] == "submit"]

    def cancels(self) -> list[str]:
        return [c[1] for c in self.calls if c[0] == "cancel"]


def _job_name(request: SubmitRequest) -> str:
    """The job name *request* asks for."""
    return request.job_name
