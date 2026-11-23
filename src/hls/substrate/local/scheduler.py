"""``Scheduler`` on one host with no scheduler: job records in a state directory and a supervisor
per job (``hls.substrate.local.supervisor``).

- ``submit`` runs ``submit_argv``, which records the job, starts its supervisor and prints the
  id, as sbatch does. A refusal that names the dependency is DependencyRefused, any other
  refusal Refused, and anything else, a timeout among them, SubmitUnknown: the job may exist and
  is looked up by its tag.
- ``test_only`` reads the host's inventory again and checks the request against the empty host.
- ``find_by_tags`` lists the jobs that have not ended, as squeue lists queued and running jobs.
- ``states`` runs a pass first, so what a job whose supervisor is gone still holds is reclaimed at
  the next poll, then reads the records. A job pending for a job that ended other than COMPLETED reports
  DependencyNeverSatisfied and pends until it is rewired or cancelled, as a Slurm job submitted
  with ``--kill-on-invalid-dep=no`` does. Records are kept, so a job is never forgotten.
- ``rewire`` changes a pending job's dependency; a job to wait for that is not known, or one
  that would make the job wait for itself, is DependencyRefused.
- ``cancel`` acts on the recorded id alone: a pending job is CANCELLED at once, a running one's
  supervisor gets SIGTERM and stops the batch script. A cancel stands even when the SIGTERM
  finds no supervisor.

Every failure to use the state directory is SchedulerUnavailable; ``states`` then reports
UNKNOWN and ``find_by_tags`` None, as when a controller does not answer.

The runner asks four things outside ``Scheduler``: its job's end time (``job_end_time``, and
``env_end_time`` from its environment), whether it may start steps (always: the batch script
starts only once the job holds its reservation, ``job_not_ready``), which earlier jobs ended
(``ended_jobs``), and its job cgroup, of which there is none (``job_cgroup``)."""

from __future__ import annotations

import logging
import signal
import subprocess
import time
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path

from pydantic import ValidationError

from hls.substrate.base import (DependencyRefused, JobInfo, JobState, Refused, Scheduler, SchedulerUnavailable,
                                SubmitRequest, SubmitUnknown)
from hls.substrate.local.inventory import HostFacts, HostInventory, InventoryError
from hls.substrate.local.placement import Reservation, never_fits
from hls.substrate.local.records import (LOCK_TIMEOUT_SEC, JobRecord, StateDir, StateError, check_job_id,
                                        reported_state)
from hls.substrate.local.schedule import TOKEN_VAR, dependency_reason, run_pass
from hls.substrate.local.stoppers import check_stopper
from hls.substrate.local.stoppers import stopper as stopper_named
from hls.substrate.local.supervisor import (BASE_ENV, END_TIME_VAR, EXIT_DEPENDENCY, EXIT_REFUSED, JOB_ID_VAR,
                                            STATE_VAR)

#: Seconds between the SIGTERM at a job's end time or cancel and the SIGKILL. The runner stops its
#: stage's roles in that time: ``STOP_GRACE_SEC`` (10 s) of SIGTERM, then up to 5 s per kill, then
#: its final records.
KILL_WAIT_SEC = 30.0
#: Seconds between a waiting supervisor's looks at the queue, and a running one's at its record.
POLL_SEC = 1.0
#: Seconds ``submit`` waits for its command: the lock's wait and a Python start, with room.
SUBMIT_TIMEOUT_SEC = LOCK_TIMEOUT_SEC + 30.0

#: Seconds ``states`` waits for the lock to run its pass; a pass someone else runs does the same work.
HOUSEKEEPING_LOCK_SEC = 1.0

_UNAVAILABLE = (StateError, OSError, ValidationError)
log = logging.getLogger(__name__)


def _one_line(text: str) -> str:
    return " ".join(text.split())


class LocalScheduler(Scheduler):
    """The jobs of the state directory *state_dir* on this host. *inventory* is read now and for
    every dry run, with *kept_memory_gib* kept back from allocations. *python* runs the submit
    command and the supervisors; it must import this package. *stopper* names the stopper every
    job's end calls (``hls.substrate.local.stoppers.STOPPERS``), with *stopper_options*, exactly the
    options it takes. Raise StateError when the directory cannot be used, InventoryError when the
    inventory cannot be read, and ValueError for an unknown stopper or options it does not take."""

    def __init__(self, *, state_dir: Path, inventory: HostInventory, kept_memory_gib: float, python: Path,
                 stopper: str, stopper_options: Mapping[str, str], kill_wait_sec: float = KILL_WAIT_SEC,
                 poll_sec: float = POLL_SEC) -> None:
        check_stopper(stopper, stopper_options)
        self.state = StateDir(state_dir)
        self.inventory = inventory
        self.kept_memory_gib = kept_memory_gib
        self.python = python
        self.stopper = stopper
        self.stopper_options = dict(stopper_options)
        self.kill_wait_sec = kill_wait_sec
        self.poll_sec = poll_sec
        self._read_host()

    def _read_host(self) -> HostFacts:
        facts = self.inventory.read(kept_memory_gib=self.kept_memory_gib)
        with self.state.locked():
            self.state.write_host(facts)
        return facts

    def submit_argv(self, request: SubmitRequest) -> tuple[str, ...]:
        shape = request.shape
        return (
            str(self.python), "-I", "-m", "hls.substrate.local.supervisor", "submit",
            f"--state={self.state.root}", f"--script={request.script}", f"--tag={request.tag}",
            f"--job-name={request.job_name}", f"--output-dir={request.output_dir}", f"--gpus={shape.gpus}",
            f"--gpu-kind={shape.gpu_kind or ''}", f"--cpus={shape.cpus}", f"--memory-gib={shape.memory_gib}",
            f"--disk-gib={shape.disk_gib}", f"--time-sec={shape.time_sec}",
            *([f"--after={check_job_id(request.after)}"] if request.after is not None else []),
            *(["--singleton"] if request.singleton else []),
            f"--stopper={self.stopper}",
            *(f"--stopper-option={name}={value}" for name, value in sorted(self.stopper_options.items())),
            f"--kill-wait-sec={self.kill_wait_sec}", f"--poll-sec={self.poll_sec}",
        )

    def submit(self, request: SubmitRequest) -> str:
        argv = self.submit_argv(request)
        try:
            r = subprocess.run(argv, env=BASE_ENV, cwd="/", stdin=subprocess.DEVNULL, capture_output=True,
                               text=True, timeout=SUBMIT_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            raise SubmitUnknown(f"submit did not answer within {SUBMIT_TIMEOUT_SEC:.0f}s") from None
        except OSError as e:
            raise SubmitUnknown(f"submit could not be run: {e}") from e
        why = _one_line(r.stderr) or f"submit exited {r.returncode}"
        if r.returncode == 0 and r.stdout.strip().isdecimal():
            return r.stdout.strip()
        if r.returncode == EXIT_DEPENDENCY:
            raise DependencyRefused(why)
        if r.returncode == EXIT_REFUSED:
            raise Refused(why)
        raise SubmitUnknown(why)

    def test_only(self, request: SubmitRequest) -> str:
        try:
            facts = self._read_host()
            check_stopper(self.stopper, self.stopper_options)
            after = None if request.after is None else self.state.read(check_job_id(request.after))
        except (InventoryError, *_UNAVAILABLE) as e:
            raise SchedulerUnavailable(str(e)) from e
        if request.after is not None and after is None:
            raise DependencyRefused(f"job dependency problem: job {request.after} is not known")
        problems = never_fits(request.shape, facts)
        if problems:
            raise Refused("the host cannot hold the request: " + "; ".join(problems))
        shape = request.shape
        kind = [g for g in facts.gpus if g.kind == shape.gpu_kind] if shape.gpus else []
        cpus = sum(len(c.cpus) for c in facts.cores)
        gpus = f"{shape.gpus} of {len(kind)} {shape.gpu_kind} GPUs" if shape.gpus else "no GPU"
        return (f"host {facts.host} holds it: {gpus}, {shape.cpus} of {cpus} CPUs, {shape.memory_gib} of "
                f"{facts.allocatable_memory_gib:.0f} GiB")

    def find_by_tags(self, tags: Sequence[str]) -> dict[str, list[str] | None]:
        wanted = list(dict.fromkeys(tags))
        if not wanted:
            return {}
        try:
            with self.state.locked(exclusive=False):
                live = [r for r in self.state.active() if not reported_state(r).terminal]
        except _UNAVAILABLE:
            return {tag: None for tag in wanted}
        return {tag: sorted((r.job_id for r in live if r.tag == tag), key=int) for tag in wanted}

    def states(self, job_ids: Sequence[str]) -> dict[str, JobInfo]:
        self._housekeep()
        return job_states(self.state, job_ids)

    def _housekeep(self) -> None:
        """A pass (``hls.substrate.local.schedule.run_pass``), when the lock is had within
        ``HOUSEKEEPING_LOCK_SEC``: it reclaims what a job whose supervisor is gone still holds even
        when no other job waits, at the orchestrator's next poll."""
        try:
            with self.state.locked(timeout_sec=HOUSEKEEPING_LOCK_SEC):
                run_pass(self.state, now=time.time(), stoppers=stopper_named, log=log.info)
        except _UNAVAILABLE as e:
            log.info("no pass of %s: %s", self.state.root, e)

    def rewire(self, job_id: str, after: str) -> None:
        check_job_id(job_id)
        check_job_id(after)
        try:
            with self.state.locked():
                record = self.state.read(job_id)
                if record is None:
                    raise Refused(f"job {job_id} is not known")
                if record.state != "PENDING" or reported_state(record) is not JobState.PENDING:
                    raise Refused(f"job {job_id} is not pending")
                if self.state.read(after) is None:
                    raise DependencyRefused(f"job dependency problem: job {after} is not known")
                if _waits_for(self.state, after, job_id):
                    raise DependencyRefused(f"job dependency problem: job {after} waits for job {job_id}")
                self.state.write(record.model_copy(update={"after": after, "reason": None}))
        except _UNAVAILABLE as e:
            raise SchedulerUnavailable(str(e)) from e

    def cancel(self, job_id: str) -> None:
        check_job_id(job_id)
        try:
            with self.state.locked():
                record = self.state.read(job_id)
                if record is None or reported_state(record).terminal:
                    return
                now = time.time()
                update: dict[str, object] = {"cancel_requested_at": record.cancel_requested_at or now}
                if record.state == "PENDING":
                    update.update(state="CANCELLED", ended_at=now, released=True, note="cancelled while pending")
                self.state.write(record.model_copy(update=update))
                record.supervisor.signal(signal.SIGTERM)
        except _UNAVAILABLE as e:
            raise SchedulerUnavailable(str(e)) from e


def _waits_for(state: StateDir, start: str, target: str) -> bool:
    """Whether job *start*, or a job it waits for, through the chain of ``after``, is *target*."""
    seen: set[str] = set()
    current: str | None = start
    while current is not None and current not in seen:
        if current == target:
            return True
        seen.add(current)
        record = state.read(current)
        current = None if record is None else record.after
    return False


def _info(state: StateDir, record: JobRecord) -> JobInfo:
    reported = reported_state(record)
    reason = None
    if reported is JobState.PENDING:
        parent = None if record.after is None else state.read(record.after)
        reason = dependency_reason(record, parent) or record.reason
    return JobInfo(record.job_id, reported, reason=reason,
                   exit_code=record.exit_code if reported.terminal else None,
                   started_at=None if reported is JobState.PENDING else record.started_at)


def job_states(state: StateDir, job_ids: Sequence[str]) -> dict[str, JobInfo]:
    """``Scheduler.states`` of the jobs of *state*: an entry for every id, UNKNOWN for one with no
    record and for all when the directory cannot be read."""
    ids = [check_job_id(j) for j in dict.fromkeys(job_ids)]
    out = {j: JobInfo(j, JobState.UNKNOWN) for j in ids}
    if not ids:
        return out
    try:
        with state.locked(exclusive=False):
            for j in ids:
                record = state.read(j)
                if record is not None:
                    out[j] = _info(state, record)
    except _UNAVAILABLE:
        return {j: JobInfo(j, JobState.UNKNOWN) for j in ids}
    return out


# What the runner asks.

#: The variables the supervisor sets for the batch script, which no role's declared environment
#: may set.
RESERVED_VARIABLES = frozenset({JOB_ID_VAR, END_TIME_VAR, STATE_VAR, TOKEN_VAR})


def node_name(hostname: str, environ: Mapping[str, str]) -> str:
    """The host's name: *hostname* up to its first dot, as the inventory names it."""
    return hostname.split(".", 1)[0]


def job_id(environ: Mapping[str, str]) -> str:
    """The job id the supervisor gave the batch script."""
    return check_job_id(environ[JOB_ID_VAR])


def state_dir(environ: Mapping[str, str]) -> Path:
    """The state directory of the job the batch script runs in."""
    return Path(environ[STATE_VAR])


def job_end_time(state: StateDir, job_id: str) -> float | None:
    """The end time of job *job_id* from its record, which its supervisor keeps to; None when the
    record cannot be read or the job has not started."""
    try:
        with state.locked(exclusive=False):
            record = state.read(check_job_id(job_id))
    except _UNAVAILABLE:
        return None
    return None if record is None else record.end_time


def env_end_time(environ: Mapping[str, str]) -> float | None:
    """``END_TIME_VAR`` from *environ*; None when it is unset or not integer seconds."""
    raw = environ.get(END_TIME_VAR)
    return float(raw) if raw and raw.isdecimal() else None


def job_not_ready(job_id: str) -> str | None:
    """Always None: a job's batch script starts only once the job holds its reservation."""
    return None


def ended_jobs(state: StateDir, job_ids: Collection[str]) -> set[str]:
    """Which of *job_ids* have ended (``JobState.terminal``); one with no record is UNKNOWN, which
    counts as alive."""
    return {j for j, info in job_states(state, list(job_ids)).items() if info.state.terminal}


def job_cgroup() -> Path:
    """Raise RuntimeError: a local job has no cgroup of its own. The runner then has no job-level
    OOM count."""
    raise RuntimeError("a local job has no job cgroup")


def reservation(state: StateDir, job_id: str) -> Reservation:
    """What job *job_id* holds, from its record. Raise StateError when the job holds nothing."""
    with state.locked(exclusive=False):
        record = state.read(check_job_id(job_id))
    if record is None or not record.holds or record.reservation is None:
        raise StateError(f"job {job_id} holds no reservation")
    return record.reservation

