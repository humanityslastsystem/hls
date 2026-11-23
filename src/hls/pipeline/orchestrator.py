"""The orchestrator: one per instance, on a login node of the cluster it schedules on.

It holds the instance's lease for as long as it runs, renewing it in the background from the
moment it takes it, so that no scheduler call, however slow, lets the lease lapse. It drives
the plan through ``reconcile.decide``. Each pass observes the store and the scheduler, writes
the next plan revision with its intents, carries out the effects, and writes their results. It
moves no heavy data and never runs srun. A pass that fails on a store or OS error, such as a
shared filesystem that stalls, is tried again at the next poll, up to ``MAX_FAILED_PASSES`` in a
row; a lost lease ends it at once. When it dies, the allocations keep running; ``hls resume``
takes the lease back, writes a revision under the new lease generation before it acts, and
carries on from the plan and the scheduler. It never relaunches a job that is still alive.

A plan that grows (``Plan.grower``) is asked for more once every stage has published: the
orchestrator calls the grower in ``observe``, on this login node, and ``decide`` appends what it
adds together with the new allocations' submissions, so the revision is written before anything
it adds is submitted. This module imports every grower (``hls.pipeline.growers``); a grower the
plan names that none registers fails the pass, and the instance waits for ``hls resume``. Warmer
allocations marked ``singleton`` are submitted, and checked with ``sbatch --test-only``, with
Slurm's ``singleton`` dependency. Every submission of an allocation, and its preflight, excludes
the nodes the allocation excludes.

Each submission records the command line the scheduler was given (``Scheduler.submit_argv``), in
its ``submitted``, ``submit_refused`` or ``submit_unknown`` event and in the plan's submission, so
what Slurm was asked for, ``--qos``, ``--exclude``, ``--dependency`` and the rest, is on record
when the scheduler's accounting has forgotten it. No secret reaches that line, which comes from a
plan the planner checked for every value of the secrets file. Everything the orchestrator records
is still scrubbed of every value of the site's secrets file as it read the file when it took the
lease: each event, and the scheduler's answers and errors before they enter a plan revision. An
orchestrator that cannot read the file refuses before it takes the lease, unless it takes the
instance over only to cancel it (``hls cancel``), which an unreadable file must not keep its jobs
running for: that one cancels and records job ids and states alone, every scheduler answer and
error withheld (``WITHHELD``).

A decision that ends the instance failed (a stage's contestant fault: the contestant's scored
result) first has the instance's records written by the orchestrator's ``on_failed`` writer
(``hls.stages.finalize.failed_records`` from the commands), and only then the revision, so no
failed instance is without them unless its outcome says so. The writer decides which instances
have records: one whose plan has no finalize stage, such as a probe or preflight-only build, has
none, and the writer says so without reading anything. No writer holds back a cancel. A writer
that fails, such as one that cannot read the secrets file, as an orchestrator that took the
instance over only to cancel it cannot, ends the instance all the same: the revision that ends
it is marked ``Outcome.records_left`` (a ``failed_records_left`` event) and its cancels are
carried out. A writer that refuses is not asked again by the same orchestrator; one that failed
on the store or the OS is, after each pass's effects, until it writes them. The next
orchestrator that takes a marked instance over writes the records once its pass's cancels are
done and clears the mark in a revision; a writer that refuses then is a
``failed_records_refused`` event, and the mark stays for the next. Either event holds the
writer's message only when it is one that holds no value of the secrets file (a
``SecretsError`` or a ``RecordsRefused`` the writer scrubbed), and the exception's type alone
otherwise. The writer creates each record only if it is not there yet, so a pass decided again
after a failed write writes nothing twice. No other instance gets a record written at a
take-over."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path

from pydantic import ValidationError

import hls.pipeline.growers  # noqa: F401  (registers every grower, whichever command started a plan)
from hls.obs.logsetup import scrub_console
from hls.pipeline.events import EventLog
from hls.pipeline.growth import grow
from hls.pipeline.plan import AllocationStatus, AttemptStatus, Outcome, Plan, PlanConflict, PlanStore, RecordsRefused
from hls.pipeline.reconcile import (Cancel, Cancelled, Effect, Observation, Result, Rewire, Rewired, Submit,
                                    Submitted, decide, finished, record)
from hls.store.base import AlreadyExists, NotFound, StoreError
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder, LockHandle, Liveness, LockLost, instance_lease, same_host_liveness
from hls.store.posix import PosixStore
from hls.substrate.base import (DependencyRefused, FaultRecord, JobInfo, Refused, Scheduler, SchedulerError,
                                SchedulerUnavailable, SubmitRequest, SubmitUnknown)
from hls.substrate.cgroup import CGROUP_ROOT, own_cgroup, read_limit
from hls.substrate.factory import substrate
from hls.substrate.secrets import Scrubber, SecretsError
from hls.util import validation_errors

log = logging.getLogger(__name__)

#: Passes between two records of the orchestrator's own memory use.
MEMORY_REPORT_PASSES = 20
#: Passes in a row that may fail on a store or OS error before the orchestrator gives up. A
#: store that stays away longer loses the lease first: it counts as lost 0.75 TTL after its
#: last renewal.
MAX_FAILED_PASSES = 10
#: What an orchestrator that could not read the secrets file records in place of a scheduler's
#: answer or an error, which it could not scrub.
WITHHELD = "withheld: the secrets file could not be read to scrub it"
#: How the void reason of an instance whose allocation ``sbatch --test-only`` refused begins.
PREFLIGHT_REFUSED = "preflight: sbatch --test-only refused allocation"

#: Writes the records of an instance the plan it is given ends failed and returns each one's
#: name and key; {} for an instance that records none, which it says without reading anything,
#: the secrets file included.
FailedRecords = Callable[[PosixStore, Plan], Mapping[str, str]]


def login_memory(*, proc: Path = Path("/proc/self"), cgroup_root: Path = CGROUP_ROOT) -> dict[str, int | None]:
    """This process's resident memory and the tightest memory limit of its cgroup and the
    cgroups above it, in bytes; None for what cannot be read. Login nodes may cap each user's
    processes together. *proc* and *cgroup_root* are
    this process's ``/proc`` directory and where cgroup v2 is mounted."""
    rss = None
    for line in (proc / "status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            rss = int(line.split()[1]) * 1024
    limits: list[int] = []
    try:
        here = cgroup_root.joinpath(*[part for part in own_cgroup(proc / "cgroup").split("/") if part])
        for d in (here, *here.parents):
            if not d.is_relative_to(cgroup_root):
                break
            if (d / "memory.max").exists() and (limit := read_limit(d / "memory.max")) is not None:
                limits.append(limit)
    except (OSError, RuntimeError, ValueError):
        pass
    return {"rss": rss, "limit": min(limits, default=None)}


def request_cancel(store: PosixStore, layout: InstanceLayout, *, at: float) -> None:
    """Ask the orchestrator holding the instance's lease to cancel it. Asking twice is fine."""
    try:
        store.create(layout.cancel_request, json.dumps({"requested_at": at}).encode())
    except AlreadyExists:
        pass


def _hold(handle: LockHandle) -> ExitStack:
    """Start renewing *handle*, a lease just taken, in the background, every quarter of its TTL.
    Closing the returned stack stops the renewals, confirms the lease unless an error is on its
    way out, and lets the lease go."""
    with ExitStack() as stack:
        stack.enter_context(handle)
        stack.enter_context(handle.keep_alive(interval_sec=handle.lock.ttl_sec / 4))
        return stack.pop_all()


def _failed(effect: Effect, error: str) -> Result:
    """The result of an effect that raised something the scheduler adapter did not classify. It
    is recorded like an answer that was lost: the effect may or may not have happened."""
    if isinstance(effect, Submit):
        return Submitted(effect.allocation, effect.n, None, error=error)
    if isinstance(effect, Rewire):
        return Rewired(effect.allocation, effect.n, effect.to_job, ok=False, error=error)
    return Cancelled(effect.allocation, effect.n, ok=False, error=error, job_id=effect.job_id)


def _refusal(error: Exception) -> str:
    """What a record writer's refusal *error* may be recorded as: its message when it is one that
    holds no value of the secrets file (a ``SecretsError``, or a ``RecordsRefused`` the writer
    scrubbed), its type alone otherwise."""
    safe = isinstance(error, (SecretsError, RecordsRefused))
    return f"{type(error).__name__}: {error}" if safe else type(error).__name__


def _definite(result: Result, plan: Plan) -> bool:
    """Whether the scheduler gave a definite answer: a job id, a rewire made, a job cancelled.
    Cancelling a stray is not one: it may take a while to leave the queue, and asking again
    changes nothing."""
    if isinstance(result, Submitted):
        return result.job_id is not None
    if isinstance(result, Cancelled) and result.job_id is not None:
        own = next(s for s in plan.allocation(result.allocation).submissions if s.n == result.n).job_id
        if result.job_id != own:
            return False
    return result.ok


class Orchestrator:
    """Drives one instance while holding its lease. The lease is renewed in the background from
    construction until ``run`` ends or ``close`` is called."""

    def __init__(self, store: PosixStore, layout: InstanceLayout, handle: LockHandle, scheduler: Scheduler, *,
                 poll_sec: float, scrubber: Scrubber | None, clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep, on_failed: FailedRecords | None = None) -> None:
        self.store = store
        #: Replaces every value of the site's secrets file in what it records; None for one that
        #: took an instance over to cancel it and could not read the file, which only cancels and
        #: records ``WITHHELD`` for every answer and error (``attach``).
        self.scrubber = scrubber
        #: Writes a failed instance's records before the revision that ends it; None writes none.
        self.on_failed = on_failed
        #: Whether the writer refused the instance's records, which this orchestrator then does
        #: not ask for again (``_record_failed``, ``_write_left_records``).
        self._refused = False
        #: The instance.
        self.layout = layout
        #: The plan's revisions.
        self.plans = PlanStore(store, layout)
        #: The held lease generation.
        self.handle = handle
        #: Submits and observes jobs.
        self.scheduler = scheduler
        self.clock = clock
        #: Sleep between passes that found nothing to do.
        self.sleep = sleep
        #: Seconds of each such sleep, and before a failed pass is tried again: the plan's
        #: ``run.poll_sec``.
        self.poll_sec = poll_sec
        #: This generation's event log.
        self.events = EventLog(store.path(layout.orchestrator_file(handle.generation, "events.jsonl")), clock=clock,
                               scrub=scrubber.scrub if scrubber is not None else None)
        self._passes = 0
        #: Jobs seen ended, with what the scheduler said. An ended job stays ended, so it is
        #: not asked about again.
        self._ended: dict[str, JobInfo] = {}
        #: Status files already reported as unusable, so each is logged once.
        self._ignored: set[tuple[int, int, str]] = set()
        #: Renews the lease until it is closed, which confirms the lease and lets it go; closing
        #: it also stops scrubbing the console with ``scrubber``, which this orchestrator starts.
        self._held = _hold(handle)
        if scrubber is not None:
            self._held.callback(scrub_console(scrubber.scrub))

    @classmethod
    def create(cls, plan: Plan, scheduler: Scheduler, *, holder: Holder, clock: Callable[[], float] = time.time,
               sleep: Callable[[float], None] = time.sleep, on_failed: FailedRecords | None = None) -> Orchestrator:
        """Take the lease of a new instance, write its first plan and check each allocation
        with the scheduler before anything is submitted. If any of that fails, the lease is let
        go. A secrets file that cannot be read raises SecretsError before the lease is taken."""
        scrubber = Scrubber.from_path(plan.settings.site.secrets_path)
        store = PosixStore(plan.store_root)
        lease = instance_lease(store, plan.layout, ttl_sec=plan.settings.run.lease_ttl_sec, clock=clock)
        orch = cls(store, plan.layout, lease.try_acquire(holder), scheduler, poll_sec=plan.settings.run.poll_sec,
                   scrubber=scrubber, clock=clock, sleep=sleep, on_failed=on_failed)
        with ExitStack() as on_error:
            on_error.push(orch._held)
            written = orch.plans.create(plan, orch.handle)
            orch.events.emit("instance_created", revision=written.revision, warnings=written.warnings,
                             memory=login_memory())
            orch.preflight(written)
            on_error.pop_all()
        return orch

    @classmethod
    def attach(cls, store: PosixStore, layout: InstanceLayout, scheduler: Scheduler, *, holder: Holder,
               alive: Liveness = same_host_liveness, clock: Callable[[], float] = time.time,
               sleep: Callable[[float], None] = time.sleep, on_failed: FailedRecords | None = None,
               cancel_only: bool = False) -> Orchestrator:
        """Take over an existing instance whose orchestrator is gone. Raise LockHeld while a
        live one holds the lease, and SecretsError, before the lease is taken, for a secrets file
        that cannot be read; but *cancel_only*, for an instance whose cancel is requested, takes it
        over all the same, only to cancel it, and records no free text (``WITHHELD``). The takeover
        revision is written before anything else, so the previous holder can no longer write."""
        plans = PlanStore(store, layout)
        latest = plans.latest()
        if latest is None:
            raise NotFound(f"{layout.root} has no plan")
        try:
            scrubber: Scrubber | None = Scrubber.from_path(latest.settings.site.secrets_path)
        except SecretsError as e:
            if not cancel_only:
                raise
            # The message names the path and the problem, never a value.
            log.warning("%s; cancelling with every scheduler answer withheld", e)
            scrubber = None
        lease = instance_lease(store, layout, ttl_sec=latest.settings.run.lease_ttl_sec, clock=clock)
        previous = lease.current()
        orch = cls(store, layout, lease.try_acquire(holder, alive=alive), scheduler,
                   poll_sec=latest.settings.run.poll_sec, scrubber=scrubber, clock=clock, sleep=sleep,
                   on_failed=on_failed)
        with ExitStack() as on_error:
            on_error.push(orch._held)
            for _ in range(3):
                try:
                    written = plans.write(plans.latest() or latest, orch.handle)
                    break
                except PlanConflict:
                    continue
            else:
                raise PlanConflict(f"{layout.root}: could not write the takeover revision")
            orch.events.emit("attached", revision=written.revision, generation=orch.handle.generation,
                             previous=None if previous is None else previous.holder.model_dump(),
                             memory=login_memory(), scrubbed=scrubber is not None)
            on_error.pop_all()
        return orch

    def preflight(self, plan: Plan) -> None:
        """``sbatch --test-only`` for every allocation. A refusal voids the instance before
        anything is submitted."""
        for alloc in plan.allocations:
            key = self.layout.orchestrator_file(self.handle.generation, f"preflight-{alloc.index}.sh")
            self.store.write_bytes(key, self._script(plan, alloc.index, 1).encode())
            request = self._request(plan, alloc.index, 1, script=self.store.path(key),
                                    tag=f"preflight:{plan.instance}", after=None)
            try:
                answer = self.scheduler.test_only(request)
            except Refused as e:
                self.events.emit("preflight_refused", allocation=alloc.index, error=str(e))
                void = plan.model_copy(update={"outcome": Outcome(
                    status="void",
                    reason=f"{PREFLIGHT_REFUSED} {alloc.index}: {self._said(str(e))}",
                    at=self.clock())})
                self.plans.write(void, self.handle)
                return
            except SchedulerUnavailable as e:
                self.events.emit("preflight_unavailable", allocation=alloc.index, error=str(e))
                continue
            self.events.emit("preflight", allocation=alloc.index, answer=answer)

    # effects

    def _script(self, plan: Plan, k: int, n: int) -> str:
        return substrate(plan.settings.site).batch_script(python=plan.toolchain.python, plan_dir=plan.plan_dir, allocation=k,
                                        submission=n, checkout=plan.code.checkout, path=plan.toolchain.path,
                                        home=plan.toolchain.home)

    def _request(self, plan: Plan, k: int, n: int, *, script: Path, tag: str, after: str | None) -> SubmitRequest:
        """The request of submission *n* of allocation *k*, and of its preflight, whose job's output
        goes to the submission's directory."""
        alloc = plan.allocation(k)
        return SubmitRequest(script=script, shape=alloc.request, job_name=plan.job_name,
                             output_dir=self.store.path(self.layout.submission(k, n)), placement=alloc.placement,
                             site=plan.settings.site, tag=tag, after=after, singleton=alloc.singleton)

    def _said(self, text: str) -> str:
        """*text*, an answer or an error to record, scrubbed of every value of the secrets file;
        ``WITHHELD`` when this orchestrator has no scrubber."""
        return self.scrubber.scrub(text) if self.scrubber is not None else WITHHELD

    def _scrubbed(self, result: Result) -> Result:
        """*result* with its error and its command line as they may be recorded (``_said``)."""
        if result.error is not None:
            result = dataclasses.replace(result, error=self._said(result.error))
        if isinstance(result, Submitted) and result.argv is not None:
            result = dataclasses.replace(result, argv=tuple(self._said(arg) for arg in result.argv))
        return result

    def perform(self, plan: Plan, effect: Effect) -> Result:
        """Carry out one effect and say what the scheduler answered, as the scheduler said it. Raise
        RuntimeError for a submission or a rewire by an orchestrator without a scrubber, which only
        cancels."""
        sub = next(s for s in plan.allocation(effect.allocation).submissions if s.n == effect.n)
        if self.scrubber is None and not isinstance(effect, Cancel):
            raise RuntimeError(f"{type(effect).__name__} of allocation {effect.allocation}: this orchestrator could "
                               "not read the secrets file and only cancels")
        if isinstance(effect, Submit):
            key = self.layout.submission_file(effect.allocation, effect.n, "batch.sh")
            self.store.write_bytes(key, self._script(plan, effect.allocation, effect.n).encode())
            request = self._request(plan, effect.allocation, effect.n, script=self.store.path(key), tag=sub.tag,
                                    after=sub.after_job)
            argv = self.scheduler.submit_argv(request)
            try:
                job_id = self.scheduler.submit(request)
            except DependencyRefused as e:
                self.events.emit("submit_refused", allocation=effect.allocation, submission=effect.n, error=str(e),
                                 dependency=True, argv=list(argv))
                return Submitted(effect.allocation, effect.n, None, refused=True, dependency=True, error=str(e),
                                 argv=argv)
            except Refused as e:
                self.events.emit("submit_refused", allocation=effect.allocation, submission=effect.n, error=str(e),
                                 argv=list(argv))
                return Submitted(effect.allocation, effect.n, None, refused=True, error=str(e), argv=argv)
            except SubmitUnknown as e:
                self.events.emit("submit_unknown", allocation=effect.allocation, submission=effect.n, error=str(e),
                                 argv=list(argv))
                return Submitted(effect.allocation, effect.n, None, error=str(e), argv=argv)
            self.events.emit("submitted", allocation=effect.allocation, submission=effect.n, job_id=job_id,
                             after_job=sub.after_job, argv=list(argv))
            return Submitted(effect.allocation, effect.n, job_id, argv=argv)
        if isinstance(effect, Rewire):
            assert sub.job_id is not None
            try:
                self.scheduler.rewire(sub.job_id, effect.to_job)
            except DependencyRefused as e:
                self.events.emit("rewire_refused", job_id=sub.job_id, to_job=effect.to_job, error=str(e),
                                 dependency=True)
                return Rewired(effect.allocation, effect.n, effect.to_job, ok=False, refused=True, dependency=True,
                               error=str(e))
            except Refused as e:
                self.events.emit("rewire_refused", job_id=sub.job_id, to_job=effect.to_job, error=str(e))
                return Rewired(effect.allocation, effect.n, effect.to_job, ok=False, refused=True, error=str(e))
            except SchedulerError as e:
                return Rewired(effect.allocation, effect.n, effect.to_job, ok=False, error=str(e))
            self.events.emit("rewired", job_id=sub.job_id, to_job=effect.to_job)
            return Rewired(effect.allocation, effect.n, effect.to_job, ok=True)
        assert isinstance(effect, Cancel)
        try:
            self.scheduler.cancel(effect.job_id)
        except SchedulerError as e:
            return Cancelled(effect.allocation, effect.n, ok=False, error=str(e), job_id=effect.job_id)
        self.events.emit("cancelled", job_id=effect.job_id)
        return Cancelled(effect.allocation, effect.n, ok=True, job_id=effect.job_id)

    def _perform(self, plan: Plan, effect: Effect) -> Result:
        """``perform``, with a failure it does not classify recorded as the effect's result, so
        one effect that breaks does not lose the answers of the others; either way as it may be
        recorded (``_scrubbed``)."""
        try:
            return self._scrubbed(self.perform(plan, effect))
        except LockLost:
            raise
        except Exception as e:
            log.error("%s of allocation %d submission %d failed: %s", type(effect).__name__, effect.allocation,
                      effect.n, self._said("".join(traceback.format_exception(e))))
            error = self._said(f"{type(e).__name__}: {e}")
            self.events.emit("effect_failed", effect=type(effect).__name__, allocation=effect.allocation,
                             submission=effect.n, error=error)
            return _failed(effect, error)

    # the loop

    def _status_of(self, plan: Plan, k: int, n: int, job_id: str | None) -> AllocationStatus | None:
        """The runner status of submission *n* of allocation *k*, or None when there is none or
        it is not usable: unreadable, written for another submission, or written by a job other
        than the one recorded for this submission."""
        try:
            raw = self.store.read_bytes(self.layout.allocation_status(k, n))
        except NotFound:
            return None
        try:
            status = AllocationStatus.model_validate_json(raw)
        except ValidationError as e:
            return self._ignore_status(k, n, f"it does not parse: {validation_errors(e)}", "unparsed")
        try:
            substrate(plan.settings.site).check_job_id(status.job_id)
        except ValueError:
            return self._ignore_status(k, n, f"it names {status.job_id!r}, which is not a Slurm job id", "id")
        if (status.instance, status.allocation, status.submission) != (plan.instance, k, n):
            return self._ignore_status(k, n, f"it is for {status.instance} allocation {status.allocation} "
                                             f"submission {status.submission}", "other")
        if job_id is not None and status.job_id != job_id:
            return self._ignore_status(k, n, f"it was written by job {status.job_id}, not job {job_id}", "job")
        return status

    def _ignore_status(self, k: int, n: int, reason: str, kind: str) -> None:
        if (k, n, kind) not in self._ignored:
            self._ignored.add((k, n, kind))
            self.events.emit("status_ignored", allocation=k, submission=n, reason=reason)
        return None

    def observe(self, plan: Plan) -> Observation:
        """Read the scheduler and the store. The scheduler is asked first: a job seen ended has
        already published every manifest and written every runner status it ever will, so the
        manifests and statuses read after it cannot miss one. Read before the job's state, a
        runner's final status could still say "running", and the job's exit code would then
        stand in for the runner's own fault, which may not be retryable or not be infra."""
        subs = [(a.index, s) for a in plan.allocations for s in a.submissions]
        wanted = {s.job_id for _, s in subs if s.job_id and (s.live or s.status == "cancel_intended")}
        wanted |= {s.after_job for _, s in subs if s.after_job and s.live}
        wanted |= {j for _, s in subs for j in s.strays}
        asked = sorted(wanted - self._ended.keys())
        fresh = self.scheduler.states(asked) if asked else {}
        self._ended.update({j: info for j, info in fresh.items() if info.state.terminal})
        jobs = {**fresh, **{j: self._ended[j] for j in wanted & self._ended.keys()}}
        unresolved = [s.tag for _, s in subs if s.job_id is None]
        tags = self.scheduler.find_by_tags(unresolved) if unresolved else {}
        published = frozenset(s.name for s in plan.stages if self.store.exists(self.layout.manifest(s.name)))
        statuses = {}
        for k, s in subs:
            if (s.live or s.job_id is None) and plan.allocation(k).failure is None:
                status = self._status_of(plan, k, s.n, s.job_id)
                if status is not None:
                    statuses[(k, s.n)] = status
        declared = {}
        for (k, n), status in statuses.items():
            info = jobs.get(status.job_id)
            if info is not None and info.state.terminal and status.stage not in published:
                fault = self._declared(k, n, status)
                if fault is not None:
                    declared[(k, n)] = fault
        cancel_requested = self.store.exists(self.layout.cancel_request)
        # Asked only once everything planned has published, since a grower reads the published
        # files, and never for an instance asked to stop, which then needs no grower to end.
        growth = None
        if (plan.grower is not None and plan.outcome is None and not cancel_requested
                and all(s.name in published for s in plan.stages)):
            growth = grow(plan, self.store)
        return Observation(now=self.clock(), published=published, jobs=jobs, tags=tags, statuses=statuses,
                           cancel_requested=cancel_requested, growth=growth, declared=declared)

    def _declared(self, k: int, n: int, status: AllocationStatus) -> FaultRecord | None:
        """The fault the attempt runner *status* names as running declared any end of it is
        (``AttemptStatus.no_retry``), for submission *n* of allocation *k*, whose job has ended;
        None when it names none, its status is not there, or the status is not this runner's
        attempt's: written for another allocation, submission or job, or unreadable, which an
        event records."""
        if status.stage is None or status.attempt is None:
            return None
        try:
            raw = self.store.read_bytes(self.layout.attempt_file(status.stage, status.attempt, "status.json"))
        except NotFound:
            return None
        try:
            attempt = AttemptStatus.model_validate_json(raw)
        except ValidationError as e:
            return self._ignore_status(k, n, f"its attempt {status.stage}/{status.attempt}'s status does not parse: "
                                             f"{e.error_count()} errors", "attempt")
        if (attempt.stage, attempt.attempt, attempt.allocation, attempt.submission, attempt.job_id) != (
                status.stage, status.attempt, k, n, status.job_id):
            return self._ignore_status(k, n, f"its attempt {status.stage}/{status.attempt}'s status was written by job "
                                             f"{attempt.job_id} for allocation {attempt.allocation} submission "
                                             f"{attempt.submission}", "attempt")
        return attempt.no_retry

    def _record(self, results: list[Result]) -> Plan:
        """Write the results into the plan and return the latest plan. Results that would change
        nothing, such as the same error again, write no revision."""
        for _ in range(3):
            latest = self.plans.latest()
            assert latest is not None
            recorded = record(latest, results, self.clock())
            if recorded == latest:
                return latest
            try:
                return self.plans.write(recorded, self.handle)
            except PlanConflict:
                continue
        raise PlanConflict(f"{self.layout.root}: could not record {len(results)} results")

    def step(self) -> tuple[Plan, bool, bool]:
        """One pass. Returns the latest plan, whether the instance is finished, and whether
        the pass made progress, in which case the next pass follows at once. Progress is a
        decision that changed the plan, or an effect that got a definite answer; recording
        another error text is neither."""
        self.handle.check()
        before = self.plans.latest()
        assert before is not None
        obs = self.observe(before)
        decision = decide(before, obs)
        plan, progress = before, decision.plan != before
        if progress:
            decided = decision.plan
            if before.outcome is None and decided.outcome is not None and decided.outcome.status == "failed":
                decided = self._record_failed(decided)
            try:
                plan = self.plans.write(decided, self.handle)
            except PlanConflict:
                return self.plans.latest(), False, True
        for kind, fields in decision.events:
            self.events.emit(kind, **fields)
        if decision.effects:
            # Another orchestrator may have taken over since the write: confirm in the store
            # once, and check the lease before each effect.
            self.handle.confirm()
            results: list[Result] = []
            for effect in decision.effects:
                self.handle.check()
                results.append(self._perform(plan, effect))
            plan = self._record(results)
            progress = progress or any(_definite(r, plan) for r in results)
        plan = self._write_left_records(plan)
        self._passes += 1
        if self._passes % MEMORY_REPORT_PASSES == 0:
            self.events.emit("orchestrator_memory", **login_memory())
        return plan, finished(plan, obs), progress

    def _record_failed(self, plan: Plan) -> Plan:
        """Write the records of the failed instance *plan* ends, with ``on_failed``, and return the
        plan to write. A writer that fails other than on a lost lease holds back neither the
        revision nor its cancels: *plan* is returned with its outcome marked ``records_left``,
        after a ``failed_records_left`` event with the error as it may be recorded
        (``_refusal``). A refusal is not asked again; a store or OS error is, after this pass's
        effects (``_write_left_records``)."""
        if self.on_failed is None:
            return plan
        try:
            written = self.on_failed(self.store, plan)
        except LockLost:
            raise
        except Exception as e:
            self._refused = not isinstance(e, (StoreError, OSError))
            error = _refusal(e)
            log.warning("the records of %s could not be written, so they are left: %s", plan.instance, error)
            self.events.emit("failed_records_left", error=error)
            assert plan.outcome is not None
            return plan.model_copy(update={"outcome": plan.outcome.model_copy(update={"records_left": True})})
        if written:
            self.events.emit("failed_records_written", records=dict(written))
        return plan

    def _write_left_records(self, plan: Plan) -> Plan:
        """Write the records left unwritten (``Outcome.records_left``), by an earlier
        orchestrator or by this one when its writer failed on the store, after this pass's
        effects, and return the plan with the mark cleared in a revision. A store or OS error
        fails the pass, which tries again; any other refusal is a ``failed_records_refused``
        event, after which this orchestrator asks no more, and the mark stays for the next. An
        orchestrator without a writer writes none."""
        outcome = plan.outcome
        if outcome is None or not outcome.records_left or self.on_failed is None or self._refused:
            return plan
        try:
            written = self.on_failed(self.store, plan)
        except (LockLost, StoreError, OSError):
            raise
        except Exception as e:
            self._refused = True
            error = _refusal(e)
            log.warning("the records %s left could not be written: %s", plan.instance, error)
            self.events.emit("failed_records_refused", error=error)
            return plan
        self.events.emit("failed_records_written", records=dict(written))
        for _ in range(3):
            latest = self.plans.latest()
            assert latest is not None and latest.outcome is not None
            cleared = latest.model_copy(update={"outcome": latest.outcome.model_copy(update={"records_left": False})})
            try:
                return self.plans.write(cleared, self.handle)
            except PlanConflict:
                continue
        raise PlanConflict(f"{self.layout.root}: could not clear its records' mark")

    def run(self) -> Plan:
        """Pass until the instance is finished; then stop renewing the lease, confirm that it was
        held to the end, and let it go. A pass that raises a store or OS error other than a lost
        lease is logged as a ``pass_failed`` event and tried again after a poll; the error is
        raised once ``MAX_FAILED_PASSES`` passes in a row failed."""
        with self._held:
            failed = 0
            while True:
                try:
                    plan, done, acted = self.step()
                except LockLost:
                    raise
                except (StoreError, OSError) as e:
                    failed += 1
                    self._pass_failed(e, failed)
                    if failed >= MAX_FAILED_PASSES:
                        raise
                    self.sleep(self.poll_sec)
                    continue
                failed = 0
                if done:
                    break
                if not acted:
                    self.sleep(self.poll_sec)
        return plan

    def _pass_failed(self, error: Exception, in_a_row: int) -> None:
        """Record a pass that failed; the event log is on the store too, so its own failure is
        only logged."""
        log.warning("pass %d in a row failed: %s", in_a_row, self._said(f"{type(error).__name__}: {error}"))
        try:
            self.events.emit("pass_failed", error=self._said(f"{type(error).__name__}: {error}"),
                             in_a_row=in_a_row, of=MAX_FAILED_PASSES)
        except OSError:
            log.exception("the pass_failed event could not be written")

    def close(self) -> None:
        """End an orchestrator that will not ``run``: stop renewing the lease and let it go.
        Raise LockLost when the lease was lost meanwhile."""
        self._held.close()
