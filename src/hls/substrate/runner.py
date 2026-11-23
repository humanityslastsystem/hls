"""The runner: one per allocation submission, started by the batch script on the compute node.

    python -m hls.substrate.runner <plan dir> --allocation K --submission N

``hls runner`` is the same entry point for people. The runner holds control state only;
roles move data to the store. In order, it:
1. reads the latest plan revision. When the plan records another job id for this submission,
   it writes nothing under the submission's directory, which that job's runner owns, logs the
   refusal and exits 2. When the instance has ended or this submission is not the
   allocation's current one, it writes a "refused" status and exits 2;
2. writes the submission's status, then rewrites it every heartbeat. It then reads the job's
   EndTime, which the status carries from then on: the controller's record of the job
   (``scontrol show job``, within scontrol's 60 s timeout) first, ``SLURM_JOB_END_TIME`` only
   when the controller does not answer. When both are known and more than 60 s apart, an
   ``end_time_sources_differ`` event names both. Each start reads both again, so a job the
   site requeues keeps to its new EndTime. The status and the SIGTERM handler exist before the
   query, so a cancel while the controller is slow still leaves the final records;
3. removes the node-local scratch (``hls-<job>`` under ``site.node_scratch``) of this user's jobs
   that the controller reports ended, each asked for by its exact id (``ended_jobs``): a job
   killed before its runner's end leaves its own behind. It then makes its own scratch, mode
   0700, refusing a path there that is not a directory of this user's, which anyone may have made
   first in a world-writable scratch. Before the grant it checks that the node's cgroups are
   cgroup v2 as Slurm lays them out (``_check_cgroups``). It checks the grant against the allocation's request, and
   reads the job's swap limit. Where swap is unlimited, its memory watch enforces each role's
   declared memory. It records the versions of the node's Slurm, apptainer, pasta and kernel
   beside the grant (``hls.substrate.versions``);
4. checks the checkout against the plan, and that ``site.nproc_max`` holds every contestant step
   of the stages it has to run on this node (``_check_process_cap``): a node whose size cannot be
   told, or a cap under the floor of the evaluations of a stage's size the node holds, is an
   infra fault that a retry would repeat. When a stage it has to run seals a container, it checks
   that the node can run the network seal (``_check_seal``). It checks that ``site.secrets_path`` lies outside the
   checkout and the store as this node resolves them. When a stage it has to run has grants, or
   its kind sets ``needs_secret_registry``, it reads the site's secrets file and the value of
   every such grant; a stage that needs the registry also gets every value of the file under
   its secret's name. A misplaced, unusable or missing file, a grant of a secret the file does
   not hold, or a grant to a variable named as a secret of the registry, is an infra fault that
   a retry would repeat. It then checks that the pinned inputs of the stages it has to run are
   published with the planned digests. When a stage is left to run, it waits until the
   controller shows the job RUNNING outside its Prolog (``job_not_ready``): a step requested
   while the controller still runs the Prolog may never get a step id, though the batch script
   already runs (after a preemption);
5. runs the allocation's stages in order and skips the published ones. A stage's deadline is its
   start plus its time, and grows by the time its roles' steps take to start, up to the job's
   EndTime less ``END_MARGIN_SEC`` (``StageContext.deadline``). Before each stage it
   checks the checkout again, that the stage can finish before the job's EndTime (an unknown
   EndTime fails the check), and that its inputs are published. Each stage runs as a newly
   claimed attempt with its own status file and event log, with the values of its own grants,
   a scrubber for every value of the secrets file, the cache and the container runtime. A
   stage that succeeds gets its manifest published, with the images and cache entries the
   stage recorded;
6. exits 0 only if every stage of the allocation is published, so ``afterok`` means success.

No exception escapes ``run``: whatever goes wrong becomes a fault in the final status, which
is always written, and when that write fails the exit status is 1. Every fault it records,
every event of its own and its attempts' event logs (role_ended with its OOM counts among
them), and every line of its log (the job's output file in the store), has the secrets file's
values replaced.

On SIGTERM no stage publishes any more, so the running attempt and the allocation have ended
whatever the roles do next. The runner first writes both final records, the attempt failed and
the allocation "terminated", and only then stops its roles, which can take longer than the
site leaves between SIGTERM and SIGKILL (Slurm's KillWait). Once the roles have
stopped it writes both again with what they did, and exits 143. A final status is replaced only
by another final status. The shutdown removes the job's scratch as soon as the stage's roles have
stopped, since the SIGKILL may come before the run's end, where it is removed otherwise.

The fault of both records is the termination's, an infra fault, with one exception: a
contestant fault stands. The early records carry one when the stage had
decided it before the SIGTERM (``StageContext.contestant_fault``), and the final ones when the
stage raised one. Roles still running at the SIGTERM are stopped, which is infra, so a stage
that had decided nothing yet ends infra."""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import socket
import stat
import sys
import threading
import time
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import TextIO

from pydantic import ValidationError

import hls.stages.registry  # noqa: F401  (registers every stage implementation)
from hls.pipeline.checkout import CodeState, code_state
from hls.pipeline.events import EventLog
from hls.pipeline.plan import (Allocation, AllocationStatus, AttemptStatus, Plan, StagePlan, Submission,
                               revision_numbers)
from hls.pipeline.stage import (END_MARGIN_SEC, StageContext, StageFault, most_roles, run_stage, seals_containers,
                                stage_class)
from hls.settings import settings_sha256
from hls.spec.resources import ResourceShape, check_grant
from hls.store.base import check_key
from hls.store.cache import ContentCache
from hls.store.manifest import (AlreadyPublished, InputMismatch, Manifest, Provenance, Resources, Times,
                                publish_manifest, read_manifest, require_inputs)
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord, JobEnd, Launcher, SchedulerUnavailable
from hls.substrate.container import ContainerRuntime, ImageRuntime
from hls.substrate.factory import substrate
from hls.substrate.faults import infra_fault
from hls.substrate.grant import Measured
from hls.substrate.memory_watch import MemoryWatch
from hls.substrate.nproc import node_cap_problem, raise_soft_limit
from hls.substrate.sandbox import remove_extracted
from hls.substrate.secrets import Scrubber, SecretsError, load_secrets, resolve_grants, secrets_path_problem
from hls.substrate.site_checks import SiteProblem
from hls.substrate.versions import ToolVersions
from hls.util import remove_tree, validation_errors

log = logging.getLogger(__name__)

EXIT_PUBLISHED = 0
EXIT_FAILED = 1
EXIT_REFUSED = 2
EXIT_TERMINATED = 143

#: Seconds between the runner's questions whether the job may start its first step, and the
#: longest it asks. A site's Prolog can hold the job for tens of seconds.
PROLOG_POLL_SEC, PROLOG_WAIT_SEC = 2.0, 300.0
#: Seconds between the shutdown's looks at whether the stage's roles have stopped, before it
#: removes the job's scratch.
ROLES_STOPPED_POLL_SEC = 0.2
#: The name of a job's node-local scratch under ``site.node_scratch``: ``hls-<job id>``.
SCRATCH_NAME = re.compile(r"hls-([0-9]+)")


def _terminated() -> FaultRecord:
    return infra_fault("the allocation was terminated", signal="SIGTERM")


def _fault_of(error: Exception, what: str) -> FaultRecord:
    """The fault *error* is: its own when it is a StageFault, otherwise a retryable infra
    fault that names *what* was running and carries the exception, since an error nothing
    foresaw may be transient."""
    if isinstance(error, StageFault):
        return error.fault
    return infra_fault(f"{what} raised {type(error).__name__}: {error}", exception=type(error).__name__,
                  message=str(error))


def read_latest_plan(plan_dir: Path) -> Plan:
    """The latest revision in *plan_dir*, read straight from the directory the runner was
    given, and checked to belong there."""
    numbers = revision_numbers(p.name for p in plan_dir.iterdir())
    if not numbers:
        raise FileNotFoundError(f"{plan_dir} holds no plan revision")
    path = plan_dir / f"rev-{numbers[-1]:08d}.json"
    plan = Plan.model_validate_json(path.read_bytes())
    if plan.revision != numbers[-1]:
        raise ValueError(f"{path} holds revision {plan.revision}, not {numbers[-1]}")
    if plan.plan_dir.resolve() != plan_dir.resolve():
        raise ValueError(f"{path} is the plan of {plan.plan_dir}, not of {plan_dir}")
    return plan


def _ready() -> str | None:
    """A job that may start its steps at once, for runners that ask no controller."""
    return None


def _no_site_problem() -> SiteProblem | None:
    """No problem, for runners that check no node's cgroups."""
    return None


def _no_seal_problems() -> list[SiteProblem]:
    """No problem, for runners that check no node's seal."""
    return []


def _none_ended(job_ids: Collection[str]) -> set[str]:
    """No job ended, for runners that ask no controller."""
    return set()


class Runner:
    """One allocation submission's runner."""

    def __init__(self, plan: Plan, allocation: int, submission: int, *, job_id: str, node: str,
                 read_job_end: Callable[[], JobEnd], measure_grant: Callable[[], Measured],
                 measure_tools: Callable[[], ToolVersions], make_launcher: Callable[[int, list[str], Path], Launcher],
                 node_name: Callable[[str, Mapping[str, str]], str], runtime: ContainerRuntime,
                 images: Callable[[Path], ImageRuntime],
                 checkout_state: Callable[[Path], CodeState] = code_state,
                 clock: Callable[[], float] = time.time, job_cgroup: Path | None = None,
                 job_not_ready: Callable[[], str | None] = _ready,
                 ended_jobs: Callable[[Collection[str]], set[str]] = _none_ended,
                 check_cgroups: Callable[[], SiteProblem | None] = _no_site_problem,
                 check_seal: Callable[[], list[SiteProblem]] = _no_seal_problems) -> None:
        #: The plan revision read at start.
        self.plan = plan
        #: Index of the allocation this runner serves.
        self.k = allocation
        #: Its submission number.
        self.n = submission
        #: The Slurm job id.
        self.job_id = job_id
        self.node = node
        #: The name the scheduler knows a node by, from its host name and the job's environment
        #: (``Substrate.node_name``): what a fault names, so a retry can exclude the node.
        self._node_name = node_name
        #: The job's EndTime as each source gave it, read once the first status is written
        #: (``read_job_end``) and unknown until then; the runner keeps to its ``end_time``. An
        #: unknown EndTime fails the run before its first stage.
        self.job_end = JobEnd(controller=None, env=None)
        self._read_job_end = read_job_end
        #: Why the job may not start a step yet (``job_not_ready``); None once it may. It raises
        #: SchedulerUnavailable or OSError when the controller does not answer.
        self._job_not_ready = job_not_ready
        #: Which of the given job ids the controller reports ended, each asked for by its exact id
        #: (``Substrate.ended_jobs``); it raises when the controller cannot be asked.
        self._ended_jobs = ended_jobs
        #: Why this node's cgroups are not laid out as every role step needs
        #: (``hls.substrate.site_checks.cgroup_problem``), or None.
        self._cgroup_problem = check_cgroups
        #: Why this node could not run the network seal (``Substrate.seal_problems``).
        self._seal_problems = check_seal
        #: The job's ``job_<id>`` cgroup, whose own OOM count tells a contestant role's OOM from
        #: the job's; None where there is none, which leaves every kernel OOM kill of a
        #: contestant role infra.
        self.job_cgroup = job_cgroup
        self._measure_grant = measure_grant
        #: Reads the versions of the node's tools (``hls.substrate.versions.measure_tools``), which
        #: the grant check records.
        self._measure_tools = measure_tools
        self._make_launcher = make_launcher
        self._checkout_state = checkout_state
        self.clock = clock
        self.store = PosixStore(plan.store_root)
        #: The instance's keys.
        self.layout = plan.layout
        #: The submission's event log, each line scrubbed.
        self.events = EventLog(self.store.path(self.layout.submission_file(allocation, submission, "events.jsonl")),
                               clock=clock, scrub=self._scrub)
        #: The grant as measured; None before the grant check.
        self.measured: Measured | None = None
        #: The memory watch, started once the checks pass.
        self.watch: MemoryWatch | None = None
        #: Node-local directory of this job, mode 0700: each role's directory and the node-local
        #: copies of cache entries. Removed when the runner ends, or by a SIGTERM's shutdown once
        #: the roles have stopped.
        self.scratch = plan.settings.site.node_scratch / f"hls-{job_id}"
        #: Held while the scratch is made or removed, so the shutdown and the run's end never both
        #: remove it, and neither while it is made.
        self._scratch_guard = threading.Lock()
        #: True once this runner has made the scratch, or taken this user's own directory there as
        #: it: only then is what lies at its path the runner's to remove.
        self._scratch_made = False
        #: The store's cache, whose locks live as long as an orchestrator lease.
        self.cache = ContentCache(self.store, lock_ttl_sec=plan.settings.run.lease_ttl_sec, clock=clock)
        #: Builds the command lines of container roles.
        self.runtime = runtime
        #: Pins images on the node for containers to run from.
        self.images = images(self.scratch)
        #: Replaces every value of the secrets file in records; empty until the file is read.
        self.scrubber = Scrubber({})
        #: The values of each pending stage's grants, by stage name, then variable, and for a
        #: stage that needs the registry every value of the file by its name; read with the
        #: secrets file, and only when a stage has grants or needs the registry.
        self._stage_secrets: dict[str, dict[str, str]] = {}
        #: The pending stages that need the registry and were handed it, every value of the
        #: secrets file among their secrets.
        self._registry_stages: set[str] = set()
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._beat: threading.Thread | None = None
        #: Set by SIGTERM: no stage publishes from then on.
        self._terminating = False
        #: Set once the final records of a SIGTERM are written: the roles may be stopped.
        self._roles_stopping = False
        #: Writes the final records of a SIGTERM, then stops the roles.
        self._shutdown: threading.Thread | None = None
        #: True once this runner owns the submission's status file: the plan names this job or
        #: none, and the submission is current.
        self._owned = False
        #: Orders the writes of attempt records between the stage's thread and the shutdown.
        self._attempt_guard = threading.Lock()
        #: The status key and running record of the attempt whose final record nobody has taken
        #: on yet; None between attempts and once the stage's thread is publishing.
        self._open_attempt: tuple[str, AttemptStatus] | None = None
        self._ctx: StageContext | None = None
        now = clock()
        self.status = AllocationStatus(
            instance=plan.instance, allocation=allocation, submission=submission, job_id=job_id, node=node,
            state="starting", started_at=now, heartbeat_at=now, end_time=None, stage=None, attempt=None, grant=None,
            tool_versions=None, swap_max_bytes=None, memory_enforced_by=None, fault=None, ended_at=None)

    # status and heartbeat

    def _scrub(self, text: str) -> str:
        """*text* with every value of the secrets file replaced. The scrubber is looked up at
        each call, since the runner replaces its own once it has read the secrets file; every
        event log the runner writes passes each line through this."""
        return self.scrubber.scrub(text)

    def _scrubbed(self, fault: FaultRecord | None) -> FaultRecord | None:
        """*fault* with every value of the secrets file replaced in its reason and evidence,
        as every fault the runner writes is (records are scrubbed when written)."""
        if fault is None:
            return None
        return fault.model_copy(update={"reason": self._scrub(fault.reason),
                                        "evidence": {k: self._scrub(v) for k, v in fault.evidence.items()}})

    def _write_status(self, fields: dict) -> None:
        """Write the status with *fields*, its fault scrubbed, and a fresh heartbeat; the caller
        holds the guard. A final status (one with ``ended_at``) is replaced only by another
        final one: until then only the heartbeat and what was measured change."""
        if self.status.ended_at is not None and fields.get("ended_at") is None:
            fields = {k: v for k, v in fields.items() if k not in ("state", "stage", "attempt", "fault", "ended_at")}
        if "fault" in fields:
            fields = {**fields, "fault": self._scrubbed(fields["fault"])}
        self.status = self.status.model_copy(update={**fields, "heartbeat_at": self.clock()})
        data = self.status.model_dump_json(indent=2).encode()
        self.store.write_bytes(self.layout.allocation_status(self.k, self.n), data)

    def _update(self, **fields) -> None:
        # The write stays under the guard, so a heartbeat cannot land an older status last.
        with self._guard:
            self._write_status(fields)

    def beat(self) -> None:
        """Rewrite the status with a fresh heartbeat."""
        self._update()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.plan.settings.run.heartbeat_sec):
            try:
                self.beat()
            except Exception:
                log.exception("heartbeat write failed")

    def terminate(self) -> None:
        """What SIGTERM does: no stage publishes from now on, and a thread of its own writes
        the final records and then stops the roles. It takes no lock and does not wait, since
        it runs as the signal handler, on the thread that may hold the status lock when the
        signal comes. A second call does nothing."""
        if self._terminating:
            return
        self._terminating = True
        self._shutdown = threading.Thread(target=self._shut_down, name="hls-runner-shutdown", daemon=True)
        self._shutdown.start()

    def _shut_down(self) -> None:
        try:
            self._write_final_records()
        except Exception:
            log.exception("the final records of the SIGTERM could not be written")
        self._roles_stopping = True
        ctx = self._ctx
        if ctx is not None:
            ctx.stop_roles()
            # No role starts once the context stops; the stage's thread stops the running ones.
            while ctx.running_roles():
                time.sleep(ROLES_STOPPED_POLL_SEC)
        self._remove_scratch()

    def _write_final_records(self) -> None:
        """The attempt that is running ends failed and the allocation "terminated", before any
        role is stopped, with the contestant fault the stage has decided by now, or else the
        fault never retried it declared once it passed work a retry would redo
        (``StageContext.declare_no_retry``), or else the termination's. An attempt that is
        publishing, or whose final record is written, is left to the stage's thread, and so is a
        status that is final already. Both records are written under the attempt guard, which a
        declaration's own write takes too (``_declare_no_retry``): a declaration either lands
        before them, and they hold it, or after, and it rewrites them."""
        with self._attempt_guard:
            ctx = self._ctx
            decided = (ctx.contestant_fault() or ctx.no_retry_fault()) if ctx is not None else None
            fault, now = self._scrubbed(decided or _terminated()), self.clock()
            if self._open_attempt is not None:
                key, running = self._open_attempt
                ended = running.model_copy(update={"state": "failed", "ended_at": now, "fault": fault})
                self.store.write_bytes(key, ended.model_dump_json(indent=2).encode())
            with self._guard:
                if self._owned and self.status.ended_at is None:
                    self._write_status({"state": "terminated", "stage": None, "attempt": None, "ended_at": now,
                                        "fault": fault})

    def _declare_no_retry(self, fault: FaultRecord) -> None:
        """Write *fault*, which the running stage just declared any end of its attempt is
        (``StageContext.declare_no_retry``), into the attempt's status at once, scrubbed: its
        running record, which the allocation's status names (``AllocationStatus.attempt``), so the
        orchestrator reads it after a runner killed outright (``reconcile.job_fault``). Once the
        SIGTERM's records are written, with the termination's fault, which is retried, it rewrites
        both with *fault* instead, as they would have been written had the declaration come first.
        An attempt whose final record is the stage thread's already has the fault from the
        context. What the write raises propagates to the stage, which must not go on without it."""
        fault = self._scrubbed(fault)
        assert fault is not None
        with self._attempt_guard:
            if self._open_attempt is None:
                return
            key, running = self._open_attempt
            running = running.model_copy(update={"no_retry": fault})
            self._open_attempt = (key, running)
            if not self._terminating:
                self.store.write_bytes(key, running.model_dump_json(indent=2).encode())
                return
            ctx = self._ctx
            if ctx is not None and ctx.contestant_fault() is not None:
                return
            now = self.clock()
            ended = running.model_copy(update={"state": "failed", "ended_at": now, "fault": fault})
            self.store.write_bytes(key, ended.model_dump_json(indent=2).encode())
            with self._guard:
                if self._owned and self.status.state == "terminated":
                    self._write_status({"state": "terminated", "stage": None, "attempt": None,
                                        "ended_at": self.status.ended_at or now, "fault": fault})

    # refusals

    def _submission(self) -> Submission | None:
        """This runner's submission in the plan; None when the plan has none."""
        if not 0 <= self.k < len(self.plan.allocations):
            return None
        return next((s for s in self.plan.allocation(self.k).submissions if s.n == self.n), None)

    def _foreign_job(self) -> str | None:
        """Why the plan gives this submission to another job, or None. The other job's runner
        owns the submission's directory, so this runner leaves it alone."""
        sub = self._submission()
        if sub is not None and sub.job_id is not None and sub.job_id != self.job_id:
            return f"submission {self.n} of allocation {self.k} is job {sub.job_id}, not {self.job_id}"
        return None

    def _refusal(self) -> str | None:
        """Why this runner must do no work, or None."""
        if self.plan.outcome is not None:
            return f"the instance already ended: {self.plan.outcome.status}"
        if not 0 <= self.k < len(self.plan.allocations):
            return f"the plan has no allocation {self.k}"
        sub = self._submission()
        if sub is None:
            return f"allocation {self.k} has no submission {self.n}"
        if not sub.live:
            return f"submission {self.n} of allocation {self.k} is {sub.status}"
        return None

    # the node-local scratch

    def _sweep_scratch(self) -> None:
        """Remove the node-local scratch of this user's jobs that have ended: each directory under
        ``site.node_scratch`` named ``hls-<id>`` for a job id, owned by this user and not a link,
        whose job the controller reports ended when asked for those exact ids. A job running, or
        one the controller cannot report on, keeps its own, and no job is asked for by user or
        name. Each removal is a ``scratch_swept`` event, one that fails ``scratch_sweep_failed``;
        a controller that cannot be asked leaves them all (``scratch_sweep_skipped``)."""
        root = self.plan.settings.site.node_scratch
        candidates: dict[str, Path] = {}
        try:
            names = os.listdir(root)
        except FileNotFoundError:
            return
        for name in names:
            m = SCRATCH_NAME.fullmatch(name)
            if m is None or m[1] == self.job_id:
                continue
            path = root / name
            try:
                st = os.lstat(path)
            except FileNotFoundError:
                continue
            if stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid():
                candidates[m[1]] = path
        if not candidates:
            return
        try:
            ended = self._ended_jobs(sorted(candidates)) & set(candidates)
        except Exception as e:  # noqa: BLE001 -- a controller that cannot be asked removes nothing
            log.warning("the controller could not be asked which jobs left their scratch: %s", e)
            self.events.emit("scratch_sweep_skipped", error=f"{type(e).__name__}: {e}")
            return
        for job_id in sorted(ended):
            path = candidates[job_id]
            try:
                remove_extracted(path)
            except OSError as e:
                log.warning("the scratch of ended job %s at %s could not be removed: %s", job_id, path, e)
                self.events.emit("scratch_sweep_failed", job_id=job_id, path=str(path), error=str(e))
                continue
            self.events.emit("scratch_swept", job_id=job_id, path=str(path))

    def _make_scratch(self) -> FaultRecord | None:
        """Make the job's node-local scratch, mode 0700, so no other user of the node can read
        what the job keeps under it, whatever modes those have; one this user's directory already
        holds there, as a requeued job's earlier run leaves it, is made 0700 and used. Anything
        else at that path, which anyone may make first in a world-writable scratch, is refused:
        infra, retried as a new job, and left where it is."""
        with self._scratch_guard:
            if self._terminating:
                return _terminated()
            self.scratch.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.mkdir(self.scratch, 0o700)
            except FileExistsError:
                pass
            st = os.lstat(self.scratch)
            if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid():
                return infra_fault(f"the job's scratch {self.scratch} is already there and is not a directory of "
                                   "this user's", check="scratch", node=self._node_name(self.status.node, os.environ))
            os.chmod(self.scratch, 0o700)
            self._scratch_made = True
        return None

    def _remove_scratch(self) -> None:
        """Remove the job's scratch, whatever an extraction cut short left in it, once this runner
        has made it (``_make_scratch``): the shutdown and the run's end hold the guard while they
        do."""
        with self._scratch_guard:
            if not self._scratch_made:
                return
            try:
                remove_extracted(self.scratch)
            except OSError as e:
                log.warning("the job's scratch %s could not be removed whole: %s", self.scratch, e)
                remove_tree(self.scratch, ignore_errors=True)

    # checks

    def _check_grant(self, request: ResourceShape) -> FaultRecord | None:
        measured = self._measure_grant()
        self.measured = measured
        # Recorded whatever the check finds: a version is a fact of the node, never a refusal.
        tools = self._measure_tools()
        swap = measured.swap_max_bytes
        self._update(grant=measured.granted, tool_versions=tools, swap_max_bytes=swap,
                     memory_enforced_by="kernel" if swap == 0 else "runner")
        problems = measured.problems + [f"{m.field}: requested {m.declared}, granted {m.granted}"
                                        for m in check_grant(request, measured.granted)]
        self.events.emit("grant_checked", granted=measured.granted.model_dump(), gpu_uuids=measured.gpu_uuids,
                         swap_max_bytes=swap, tool_versions=tools.model_dump(), problems=problems)
        if problems:
            # The node is named, as Slurm names it, so a retry can go elsewhere: a node short of
            # the request (free disk another user holds, a GPU gone) stays short.
            return infra_fault("grant mismatch: " + "; ".join(problems), check="grant",
                               node=self._node_name(self.status.node, os.environ))
        return None

    def _check_checkout(self) -> FaultRecord | None:
        now = self._checkout_state(self.plan.code.checkout)
        if now == self.plan.code:
            return None
        return infra_fault(f"the checkout changed since planning: planned {self.plan.code.model_dump_json()}, "
                      f"found {now.model_dump_json()}", retryable=False)

    def _check_process_cap(self, launcher: Launcher, stages: list[str]) -> FaultRecord | None:
        """Why a contestant step of one of *stages*, those whose kind runs a contestant's code
        (``runs_contestant_code``), could not start on this node, or None: the node's size
        (``Launcher.node_size``) does not tell how many evaluations of the stage's size it holds,
        or ``site.nproc_max`` is under their floor (``hls.substrate.nproc.node_cap_problem``). A
        plan checks the cap against one evaluation only, as it does not know its nodes; this
        checks it against the node's before any stage runs, so a cap that cannot hold them costs
        the allocation's start and never a stage's hours. Infra, and not retried: the cap is the
        site's setting, and the reason names the value to raise it to. Each stage checks the same
        again with the uid's hard limit when it reads its ceiling (``StageContext.process_ceiling``)."""
        node = launcher.node_size()
        nproc_max = self.plan.settings.site.nproc_max
        for name in stages:
            stage = self.plan.stage(name)
            try:
                cls = stage_class(stage.kind)
            except ValueError:
                # An unknown kind fails as unknown when its stage runs.
                continue
            if not cls.runs_contestant_code:
                continue
            problem = node_cap_problem(node, nproc_max=nproc_max, gpus=stage.shape.gpus, cpus=stage.shape.cpus)
            self.events.emit("process_cap_checked", stage=name, node_gpus=node.gpus, node_cpus=node.cpus,
                             nproc_max=nproc_max, problem=problem)
            if problem is not None:
                return infra_fault(f"no contestant step of stage {name} can run on this node: {problem}",
                                   retryable=False, stage=name)
        return None

    def _site_fault(self, reason: str, *, retryable: bool, check: str) -> FaultRecord:
        """An infra fault of this node's set-up, naming the node as Slurm names it, so a retry can
        go elsewhere."""
        node = self._node_name(self.status.node, os.environ)
        return infra_fault(reason, retryable=retryable, check=check, node=node)

    def _check_cgroups(self) -> FaultRecord | None:
        """Why no role step could read its limits on this node, or None: its cgroups are not
        cgroup v2 as Slurm's cgroup/v2 plugin lays them out (``site_checks.cgroup_problem``).
        Checked before the grant, which reads the job's cgroup."""
        problem = self._cgroup_problem()
        if problem is None:
            return None
        return self._site_fault(problem.reason, retryable=problem.retryable, check="cgroups")

    def _check_seal(self, stages: list[str]) -> FaultRecord | None:
        """Why the network seal of one of *stages*, those that start a sealed container
        (``Stage.seals_containers``), could not run on this node, or None: what the substrate's
        seal needs of the node (``Substrate.seal_problems``; for the pasta seal, pasta, the firewall
        tools and unprivileged user namespaces). Retried only when every problem may pass on
        another node."""
        sealed = []
        for name in stages:
            stage = self.plan.stage(name)
            try:
                if seals_containers(stage.kind, stage.params):
                    sealed.append(name)
            except ValueError:
                # An unknown kind, or parameters its model refuses, fail when the stage runs.
                continue
        if not sealed:
            return None
        problems = self._seal_problems()
        if not problems:
            return None
        return self._site_fault(f"the network seal of stage {', '.join(sealed)} cannot run on this node: "
                                + "; ".join(p.reason for p in problems),
                                retryable=all(p.retryable for p in problems), check="seal")

    def _needs_registry(self, name: str) -> bool:
        """Whether stage *name*'s kind needs every value of the secrets file. An unknown kind
        needs nothing here; its stage fails as unknown when it runs."""
        try:
            cls = stage_class(self.plan.stage(name).kind)
        except ValueError:
            return False
        return cls.needs_secret_registry

    def _load_secrets(self, stages: list[str]) -> FaultRecord | None:
        """Read the site's secrets file and each of *stages*' granted values, when one of them
        has grants or needs the registry; the file is not touched otherwise. A stage that needs
        the registry gets every value of the file under its secret's name beside its grants. A
        ``site.secrets_path`` inside the checkout or the store as this node resolves it, a file
        that cannot be used or is missing, a grant of a secret the file does not hold, or a grant
        to a variable a secret of the registry is also named, is an infra fault a retry would
        repeat."""
        misplaced = secrets_path_problem(self.plan.settings.site.secrets_path, checkout=self.plan.code.checkout,
                                         store_root=self.plan.store_root)
        if misplaced is not None:
            return infra_fault(misplaced, retryable=False)
        granted = [s for s in stages if self.plan.stage(s).grants or self._needs_registry(s)]
        if not granted:
            return None
        try:
            secrets = load_secrets(self.plan.settings.site.secrets_path)
        except SecretsError as e:
            return infra_fault(f"secrets file: {e}", retryable=False)
        self.scrubber = Scrubber.from_secrets(secrets)
        for name in granted:
            try:
                values = resolve_grants(secrets, self.plan.stage(name).grants)
            except SecretsError as e:
                return infra_fault(f"stage {name} grants: {e}", retryable=False, stage=name)
            if self._needs_registry(name):
                clash = sorted(set(values) & set(secrets.secrets))
                if clash:
                    return infra_fault(f"stage {name} needs the secrets registry, whose names its grants also use: "
                                       f"{clash}", retryable=False, stage=name)
                values |= secrets.secrets
                self._registry_stages.add(name)
            self._stage_secrets[name] = values
        return None

    def _deadline_cap(self) -> float | None:
        """The latest a stage's deadline may grow to: the job's EndTime less ``END_MARGIN_SEC``;
        None while the EndTime is unknown, which ``_check_time`` refuses before any stage."""
        end_time = self.job_end.end_time
        return None if end_time is None else end_time - END_MARGIN_SEC

    def _check_time(self, stage: StagePlan) -> FaultRecord | None:
        """A stage that cannot end before the job does is not started, and no stage is when the
        job's end is unknown. It needs its own time, ``END_MARGIN_SEC`` before the job's end, and
        ``site.step_launch_allowance_sec`` for each role it may start (``most_roles``), by which
        its deadline may grow."""
        end_time = self.job_end.end_time
        if end_time is None:
            return infra_fault("EndTime unknown: neither scontrol nor SLURM_JOB_END_TIME gave the job's end")
        left = end_time - self.clock()
        allowance = most_roles(stage.kind, stage.params) * self.plan.settings.site.step_launch_allowance_sec
        need = stage.shape.time_sec + END_MARGIN_SEC + allowance
        if need > left:
            return infra_fault(f"stage {stage.name} needs {need:.0f}s (its time, the runner's margin and its roles' "
                               f"launch allowance) and the allocation ends in {left:.0f}s")
        return None

    # stages

    def _published(self, name: str) -> tuple[Manifest, str] | None:
        """The manifest of stage *name* of this instance and its digest; None when it is not
        published. A manifest that cannot be parsed is a StageFault that is not retryable,
        since reading it again gives the same bytes."""
        try:
            return read_manifest(self.store, self.layout, name)
        except ValidationError as e:
            raise StageFault.infra(f"the manifest of {name} is not a valid manifest: {validation_errors(e)}",
                                   retryable=False, stage=name) from None

    def _pinned(self, pinned: dict[str, str], what: str) -> dict[str, Manifest]:
        """The manifests of other instances that the plan pinned, keyed by manifest key. Raise
        StageFault when one is not published (retryable, since the store may be slow to show
        it), or has other bytes than planned or cannot be parsed (not retryable, since reading
        it again gives the same bytes)."""
        absent = sorted(key for key in pinned if not self.store.exists(key))
        if absent:
            raise StageFault.infra(f"{what}: {', '.join(absent)} not published", retryable=True)
        try:
            return require_inputs(self.store, pinned)
        except InputMismatch as e:
            raise StageFault.infra(f"{what}: {e}", retryable=False) from e

    def _inputs(self, stage_name: str) -> tuple[dict[str, Manifest], dict[str, str]]:
        """The manifests *stage_name* reads, and the digest of each one's key. Manifests of
        this instance are whatever was published; manifests of other instances must match the
        digests pinned in the plan. Raise StageFault for anything missing or wrong."""
        stage = self.plan.stage(stage_name)
        manifests = self._pinned(stage.inputs, f"stage {stage_name} inputs")
        digests = dict(stage.inputs)
        missing = []
        for name in stage.after:
            found = self._published(name)
            if found is None:
                missing.append(name)
                continue
            key = self.layout.manifest(name)
            manifests[key], digests[key] = found
        if missing:
            raise StageFault.infra(f"stage {stage_name} inputs: stages {missing} are not published", retryable=True)
        return manifests, digests

    def _check_outputs(self, name: str, attempt: int, outputs: dict[str, str]) -> None:
        """Raise StageFault unless every output key lies under the attempt's ``outputs/``;
        otherwise the manifest could vouch for a file that is not the attempt's own."""
        prefix = self.layout.attempt(name, attempt) + "/outputs/"

        def outside(key: str) -> bool:
            try:
                check_key(key)
            except ValueError:
                return True
            return not key.startswith(prefix)

        bad = sorted(key for key in outputs.values() if outside(key))
        if bad:
            raise StageFault.infra(f"stage {name} named outputs outside {prefix}: {', '.join(bad)}",
                                   retryable=False, stage=name)

    def _publish(self, ctx: StageContext, outputs: dict[str, str], digests: dict[str, str], *,
                 started_at: float, submitted_at: float) -> str | None:
        """Publish the manifest of the attempt *ctx* ran, with the images and cache entries it
        recorded, and return its digest; None when another attempt's manifest was there
        first, which counts as the stage being published. Each output is recorded by the sha256
        its writer gave and its size (``StageContext.output_ref``): the runner reads no output."""
        name, attempt, stage = ctx.stage.name, ctx.attempt, ctx.stage
        self._check_outputs(name, attempt, outputs)
        manifest = Manifest(
            instance=self.plan.instance, stage=name, attempt=attempt,
            outputs={k: ctx.output_ref(key) for k, key in outputs.items()}, inputs=digests,
            provenance=Provenance(hls_commit=self.plan.code.hls_commit, harbor_commit=self.plan.code.harbor_commit,
                                  dirty_diff_sha256=self.plan.code.dirty_diff_sha256,
                                  image_digests=ctx.image_digests, cache_entries=ctx.cache_entries,
                                  config_sha256=settings_sha256(self.plan.settings)),
            resources=Resources(declared=stage.shape, granted=self.status.grant),
            times=Times(submitted_at=submitted_at, allocation_started_at=self.status.started_at,
                        started_at=started_at, ended_at=self.clock()))
        try:
            return publish_manifest(self.store, self.layout, manifest)
        except AlreadyPublished:
            self.events.emit("stage_published_elsewhere", stage=name, attempt=attempt)
            return None

    def _run_stage(self, name: str, launcher: Launcher, submitted_at: float) -> FaultRecord | None:
        """Run stage *name* as a new attempt and return the fault it ended with, or None.
        Raise StageFault when its inputs are not right, before an attempt is claimed."""
        stage = self.plan.stage(name)
        fault = self._check_checkout() or self._check_time(stage)
        if fault is not None:
            return fault
        inputs, digests = self._inputs(name)
        now = self.clock()
        attempt = self.layout.claim_attempt(self.store, name)
        status = AttemptStatus(stage=name, attempt=attempt, allocation=self.k, submission=self.n, job_id=self.job_id,
                               state="running", started_at=now, ended_at=None, fault=None, manifest_sha256=None,
                               no_retry=None)
        status_key = self.layout.attempt_file(name, attempt, "status.json")
        digest = None
        published = False
        # The fault the stage declared an end of it is once a retry would redo its work.
        no_retry: FaultRecord | None = None
        # From here, whatever goes wrong is the attempt's fault, and its failed status is written.
        try:
            if self._begin_attempt(status_key, status):
                self._update(state="running", stage=name, attempt=attempt)
                self.events.emit("stage_started", stage=name, attempt=attempt)
                events = EventLog(self.store.path(self.layout.attempt_file(name, attempt, "events.jsonl")),
                                  clock=self.clock, scrub=self._scrub)
                ctx = StageContext(store=self.store, plan=self.plan, stage=stage, attempt=attempt,
                                   launcher=launcher, watch=self.watch, events=events, inputs=inputs,
                                   deadline=now + stage.shape.time_sec, clock=self.clock,
                                   deadline_cap=self._deadline_cap(),
                                   secrets=self._stage_secrets.get(name, {}), scrubber=self.scrubber,
                                   cache=self.cache, scratch=self.scratch, job_id=self.job_id,
                                   runtime=self.runtime, images=self.images,
                                   secret_registry=name in self._registry_stages,
                                   on_no_retry=self._declare_no_retry)
                self._ctx = ctx
                # A SIGTERM whose records were written before the context existed could not stop
                # its roles.
                if self._roles_stopping:
                    ctx.stop_roles()
                outputs = run_stage(stage_class(stage.kind)(), ctx)
                if self._may_publish():
                    digest = self._publish(ctx, outputs, digests, started_at=now, submitted_at=submitted_at)
                    published = True
        except Exception as e:
            if not isinstance(e, StageFault):
                log.exception("stage %s failed outside its own code", name)
            fault = _fault_of(e, f"stage {name}")
        finally:
            if self._ctx is not None:
                no_retry = self._ctx.no_retry_fault()
            self._ctx = None
        if self._terminating and not published and (fault is None or fault.fault_class != "contestant"):
            fault = no_retry or _terminated()
        elif no_retry is not None and fault is not None and fault.retryable:
            # Past its point of no retry the attempt's failure, the runner's own among them, is the
            # declared one: a retry would do the stage's work again.
            fault = no_retry.model_copy(update={"evidence": {**no_retry.evidence, "fault": fault.reason}})
        fault = self._scrubbed(fault)
        ended = status.model_copy(update={"state": "failed" if fault else "succeeded", "ended_at": self.clock(),
                                          "fault": fault, "manifest_sha256": digest,
                                          "no_retry": self._scrubbed(no_retry)})
        try:
            with self._attempt_guard:
                self._open_attempt = None
                self.store.write_bytes(status_key, ended.model_dump_json(indent=2).encode())
            self.events.emit("stage_ended", stage=name, attempt=attempt, state=ended.state,
                             fault=None if fault is None else fault.reason)
        except Exception as e:
            # The stage's own fault, and whose it is, stays; the manifest decides success.
            log.exception("the final record of attempt %d of stage %s could not be written", attempt, name)
            if fault is not None:
                noted = {"status_write_exception": type(e).__name__, "status_write_message": str(e)}
                fault = fault.model_copy(update={"evidence": {**fault.evidence, **noted}})
        return fault

    def _begin_attempt(self, key: str, running: AttemptStatus) -> bool:
        """Write the running record of a claimed attempt and return True; or, once a SIGTERM
        came, write nothing and return False, so that the attempt's failed record follows at
        once."""
        with self._attempt_guard:
            if self._terminating:
                return False
            self.store.write_bytes(key, running.model_dump_json(indent=2).encode())
            self._open_attempt = (key, running)
            return True

    def _may_publish(self) -> bool:
        """Whether the attempt whose stage returned may publish: not once a SIGTERM came. From
        a yes on, its final record is the stage thread's alone."""
        with self._attempt_guard:
            if self._terminating:
                return False
            self._open_attempt = None
            return True

    # the run

    def _work(self, alloc: Allocation, sub: Submission) -> FaultRecord | None:
        """Everything from the first status to the last stage: the checks, then the stages in
        order. Return the fault that stopped it, or None. Any exception becomes a fault."""
        try:
            self._update()
            self._beat = threading.Thread(target=self._heartbeat_loop, name="hls-runner-heartbeat", daemon=True)
            self._beat.start()
            # Asked only now that the status is written: the controller can take its whole timeout.
            self.job_end = self._read_job_end()
            self._update(end_time=self.job_end.end_time)
            if self.job_end.sources_differ:
                self.events.emit("end_time_sources_differ", end_time=self.job_end.end_time,
                                 scontrol_end_time=self.job_end.controller, slurm_job_end_time=self.job_end.env)
            if self._terminating:
                return _terminated()
            # Before the grant check, which reads the scratch's free space.
            self._sweep_scratch()
            fault = (self._check_cgroups() or self._check_grant(alloc.request) or self._check_checkout()
                     or self._make_scratch())
            if fault is not None:
                return fault
            assert self.status.grant is not None and self.measured is not None
            pending = [name for name in alloc.stages if self._published(name) is None]
            launcher = self._make_launcher(self.status.grant.gpus, self.measured.gpu_uuids, self.scratch)
            fault = (self._check_process_cap(launcher, pending) or self._check_seal(pending)
                     or self._load_secrets(pending))
            if fault is not None:
                return fault
            self._pinned({key: digest for name in pending for key, digest in self.plan.stage(name).inputs.items()},
                         "pinned inputs")
            if pending:
                fault = self._await_steps()
                if fault is not None:
                    return fault
            self.watch = MemoryWatch(enforce=self.status.memory_enforced_by == "runner", clock=self.clock,
                                     on_breach=lambda step, s, limit: self.events.emit(
                                         "memory_enforced", step=step, footprint=s.footprint, limit=limit,
                                         current=s.current, swap_current=s.swap_current),
                                     on_single_kill=lambda step, s, limit: self.events.emit(
                                         "memory_process_killed", step=step, footprint=s.footprint, limit=limit,
                                         current=s.current, swap_current=s.swap_current),
                                     job_dir=self.job_cgroup)
            self.watch.start()
            for name in alloc.stages:
                if self._terminating:
                    return _terminated()
                if self._published(name) is not None:
                    self.events.emit("stage_already_published", stage=name)
                    continue
                fault = self._run_stage(name, launcher, sub.submitted_at or sub.intended_at)
                if fault is not None:
                    return fault
            return None
        except Exception as e:
            if not isinstance(e, StageFault):
                log.exception("the runner failed")
            return _fault_of(e, "the runner")

    def _await_steps(self) -> FaultRecord | None:
        """Wait until the controller shows the job RUNNING outside its Prolog, asking every
        ``PROLOG_POLL_SEC`` for at most ``PROLOG_WAIT_SEC``. A controller that does not answer
        is not waited for: the batch script runs, so the first step is tried. Return the fault
        that ends the run: a SIGTERM meanwhile, or a job still not ready when the wait ends
        (infra, retryable). The wait is timed on the monotonic clock, so a step of the wall
        clock neither cuts it short nor stretches it."""
        give_up = time.monotonic() + PROLOG_WAIT_SEC
        waiting_since = None
        while True:
            try:
                why = self._job_not_ready()
            except (SchedulerUnavailable, OSError) as e:
                self.events.emit("job_state_unknown", error=str(e))
                return None
            if why is None:
                if waiting_since is not None:
                    self.events.emit("job_ready", waited_sec=round(time.monotonic() - waiting_since, 3))
                return None
            if waiting_since is None:
                waiting_since = time.monotonic()
                self.events.emit("job_not_ready", why=why)
            if self._terminating:
                return _terminated()
            if time.monotonic() >= give_up:
                return infra_fault(f"the job was still {why} after {PROLOG_WAIT_SEC:.0f}s, so no step was started",
                                   retryable=True, why=why)
            time.sleep(PROLOG_POLL_SEC)

    def _stop_background(self) -> None:
        """Stop the memory watch and the heartbeat, and wait for a SIGTERM's records; no
        sample, heartbeat or early record follows."""
        if self.watch is not None:
            self.watch.stop()
        self._stop.set()
        if self._beat is not None:
            self._beat.join(self.plan.settings.run.heartbeat_sec)
        if self._shutdown is not None:
            self._shutdown.join(self.plan.settings.run.heartbeat_sec)

    def _conclude(self, alloc: Allocation, fault: FaultRecord | None) -> int:
        """Write the final status and return the exit status. What is published decides first:
        when every stage is, the allocation succeeded, whatever else happened since."""
        unpublished = []
        for name in alloc.stages:
            try:
                if self._published(name) is None:
                    unpublished.append(name)
            except Exception as e:
                unpublished.append(name)
                fault = fault or _fault_of(e, f"reading the manifest of {name}")
        if not unpublished:
            state, code, fault = "succeeded", EXIT_PUBLISHED, None
        elif self._terminating:
            state, code, fault = "terminated", EXIT_TERMINATED, fault or _terminated()
        else:
            state, code, fault = "failed", EXIT_FAILED, fault or infra_fault(f"stages {unpublished} are not published")
        try:
            self._update(state=state, stage=None, attempt=None, ended_at=self.clock(), fault=fault)
        except Exception:
            log.exception("the final status could not be written")
            return EXIT_FAILED
        return code

    def _refuse(self, reason: str) -> int:
        """Record that this runner did no work, and return the exit status."""
        try:
            self.events.emit("refused", reason=reason)
            self._update(state="refused", fault=infra_fault(reason), ended_at=self.clock())
        except Exception:
            log.exception("the refusal could not be recorded")
            return EXIT_FAILED
        return EXIT_REFUSED

    def run(self) -> int:
        """Do the allocation's work and return the exit status."""
        foreign = self._foreign_job()
        if foreign is not None:
            log.error("refused, and nothing written: %s", foreign)
            return EXIT_REFUSED
        refusal = self._refusal()
        if refusal is not None:
            return self._refuse(refusal)
        with self._guard:
            self._owned = True
        alloc = self.plan.allocation(self.k)
        sub = next(s for s in alloc.submissions if s.n == self.n)
        try:
            try:
                fault = self._work(alloc, sub)
            finally:
                self._stop_background()
            return self._conclude(alloc, fault)
        finally:
            self._remove_scratch()


def _job_cgroup(locate: Callable[[], Path]) -> Path | None:
    """This process's job cgroup as *locate* finds it (``Substrate.job_cgroup``); None when it
    cannot be found."""
    try:
        return locate()
    except (OSError, RuntimeError) as e:
        log.warning("the job's cgroup cannot be found: %s", e)
        return None


#: Format of the runner's log lines.
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ScrubbingFormatter(logging.Formatter):
    """Formats a log record, its traceback included, then replaces every value the scrubber
    knows in the whole text. The scrubber is looked up for each record, because the runner
    replaces its own once it has read the secrets file."""

    def __init__(self, fmt: str, scrubber: Callable[[], Scrubber]) -> None:
        super().__init__(fmt)
        self._scrubber = scrubber

    def format(self, record: logging.LogRecord) -> str:
        return self._scrubber().scrub(super().format(record))


def configure_logging(scrubber: Callable[[], Scrubber], stream: TextIO | None = None) -> logging.Handler:
    """Send every log record of INFO and above, and every warning, to *stream* (stderr when
    None), each scrubbed with the scrubber *scrubber* returns when the record is formatted.
    Return the handler. The job's log in the store is what this writes."""
    handler = logging.StreamHandler(stream)
    handler.setFormatter(ScrubbingFormatter(LOG_FORMAT, scrubber))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    logging.captureWarnings(True)
    return handler


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls runner")
    p.add_argument("plan_dir", type=Path)
    p.add_argument("--allocation", type=int, required=True)
    p.add_argument("--submission", type=int, required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    runner: Runner | None = None
    configure_logging(lambda: runner.scrubber if runner is not None else Scrubber({}))
    plan = read_latest_plan(args.plan_dir)
    on = substrate(plan.settings.site)
    job_id = on.job_id(os.environ)
    log.info("runner for %s allocation %d submission %d in job %s; environment names: %s", plan.instance,
             args.allocation, args.submission, job_id, " ".join(sorted(os.environ)))
    # Before any step: the steps take the raised limit, and a contestant's its own share of it.
    soft, hard = raise_soft_limit()
    log.info("RLIMIT_NPROC found soft %s hard %s; the soft limit is now the hard one", soft, hard)
    scratch_root = plan.settings.site.node_scratch
    job = _job_cgroup(on.job_cgroup)
    runner = Runner(
        plan, args.allocation, args.submission, job_id=job_id, node=socket.gethostname(),
        read_job_end=lambda: on.read_job_end(job_id, os.environ), job_cgroup=job,
        job_not_ready=lambda: on.job_not_ready(job_id),
        ended_jobs=on.ended_jobs,
        check_cgroups=on.cgroup_problem,
        check_seal=lambda: on.seal_problems(os.environ),
        # The substrate finds the job's limits again here, in the grant check, which records why
        # that fails.
        measure_grant=lambda: on.measure_grant(os.environ, scratch_root),
        measure_tools=lambda: on.measure_tools(plan.settings.site.pasta_path),
        make_launcher=lambda gpus, gpu_uuids, scratch: on.launcher(
            python=str(plan.toolchain.python), scratch=scratch, allocation_gpus=gpus,
            allocation_gpu_uuids=gpu_uuids, pasta=plan.settings.site.pasta_path,
            upstream_proxy=plan.settings.site.egress_upstream_proxy),
        node_name=on.node_name, runtime=on.runtime(), images=on.images)
    signal.signal(signal.SIGTERM, lambda signum, frame: runner.terminate())
    code = runner.run()
    log.info("runner exits %d (%s)", code, runner.status.state)
    return code


if __name__ == "__main__":
    sys.exit(main())
