"""The whole path on one machine: the orchestrator plans and submits, an in-memory Slurm
runs each started job's batch script line through the real runner, and roles run as local
processes. Only srun and the cluster are stood in for."""

from __future__ import annotations

import shlex
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

from hls.pipeline import commands
from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.pipelines import noop_two
from hls.pipeline.plan import AllocationStatus, Plan
from hls.pipeline.stage import StageFault
from hls.spec.resources import GrantedResources
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.stages.noop import NoopStage
from hls.substrate.base import FaultRecord, JobState, RoleHandle, RoleSpec
from hls.substrate.factory import substrate
from hls.substrate.grant import Measured
from hls.substrate.runner import EXIT_PUBLISHED, EXIT_TERMINATED, JobEnd, Runner, read_latest_plan
from tests.pipeline.fake_scheduler import FakeJob, FakeScheduler
from tests.pipeline.fakes import LocalLauncher, fake_uuid
from tests.pipeline.plans import VERSIONS, noop_plan


class KillingLauncher(LocalLauncher):
    """Terminates its runner, as scancel's SIGTERM would, once a role of *stage* starts. *kw*
    are LocalLauncher's."""

    def __init__(self, gpus: int, stage: str | None, **kw: Any) -> None:
        super().__init__(gpus=gpus, **kw)
        self.stage = stage
        self.runner: Runner | None = None

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        handle = super().launch(spec, share_gpus_of=share_gpus_of)
        if self.stage is not None and f"/stages/{self.stage}/" in str(spec.log_dir) and self.runner is not None:
            threading.Timer(0.3, self.runner.terminate).start()
        return handle


class LocalSlurm(FakeScheduler):
    """Runs a started job's runner in this process. ``cancel`` of a running job stops its
    runner with SIGTERM's effect, as scancel does, and the job stays CANCELLED. Every job is
    granted what it asked for, and *disk_free_gib* of node-local disk, or with None exactly the
    disk it asked for. With *node_disk*, that disk is the node's whole local disk, which the
    job's scratch uses (LocalLauncher's *node_disk_gib*), so each role's free disk is what the
    files there leave of it. *job_cgroups* gives each job a fake job cgroup,
    ``<job_cgroups>/job_<id>``, whose own limit was never reached; its runner watches it, and its
    roles' steps are laid out under it."""

    #: The launcher of each job's roles; a subclass may name one that plants what a step does.
    launcher_class: ClassVar[type[KillingLauncher]] = KillingLauncher

    def __init__(self, kill: dict[str, str] | None = None, *, replace_modules: dict[str, Path] | None = None,
                 disk_free_gib: float | None = 1000.0, node_disk: bool = False,
                 job_cgroups: Path | None = None) -> None:
        super().__init__()
        #: Free node-local disk every job is granted, in GiB; None for the disk each job asked for.
        self.disk_free_gib = disk_free_gib
        #: Whether the granted disk is the node's whole local disk, which the roles' grants count down.
        self.node_disk = node_disk
        #: Directory of each job's fake job cgroup; None for none.
        self.job_cgroups = job_cgroups
        #: Job id to the stage during which that job is killed.
        self.kill = kill or {}
        #: Role modules the jobs' launchers run a stand-in for (see LocalLauncher).
        self.replace_modules = dict(replace_modules or {})
        #: The runner of each job that is running now, by job id.
        self.runners: dict[str, Runner] = {}

    def _job_cgroup(self, job_id: str) -> Path | None:
        """Job *job_id*'s fake job cgroup, made with no limit reached at it or below it."""
        if self.job_cgroups is None:
            return None
        cgroup = self.job_cgroups / f"job_{job_id}"
        cgroup.mkdir(parents=True, exist_ok=True)
        for name in ("memory.events.local", "memory.events"):
            (cgroup / name).write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
        return cgroup

    def _run(self, job: FakeJob) -> int:
        words = shlex.split(job.request.script.read_text().splitlines()[-1])
        plan_dir, k, n = Path(words[4]), int(words[6]), int(words[8])
        plan = read_latest_plan(plan_dir)
        req = plan.allocation(k).request
        disk = float(req.disk_gib) if self.disk_free_gib is None else self.disk_free_gib
        granted = GrantedResources(gpus=req.gpus, gpu_kind=req.gpu_kind, cpus=req.cpus,
                                   memory_gib=float(req.memory_gib), disk_free_gib=disk)
        measured = Measured(granted=granted, gpu_uuids=[fake_uuid(i) for i in range(req.gpus)], swap_max_bytes=0)
        # The job ends when its time limit does, as the controller records it.
        job_end = JobEnd(controller=time.time() + req.time_sec, env=None)
        cgroup = self._job_cgroup(job.job_id)

        def make_launcher(gpus: int, gpu_uuids: list[str], scratch: Path) -> KillingLauncher:
            node = {"node_disk_gib": disk, "scratch": scratch} if self.node_disk else {}
            launcher = self.launcher_class(req.gpus, self.kill.get(job.job_id), replace_modules=self.replace_modules,
                                           steps_under=cgroup, **node)
            launcher.runner = runner
            launcher.use_plan(plan)
            return launcher

        runner = Runner(plan, k, n, job_id=job.job_id, node="local", read_job_end=lambda: job_end,
                        measure_grant=lambda: measured, measure_tools=lambda: VERSIONS, make_launcher=make_launcher,
                        node_name=substrate(plan.settings.site).node_name, runtime=substrate(plan.settings.site).runtime(),
                        images=substrate(plan.settings.site).images,
                        checkout_state=lambda path: plan.code, job_cgroup=cgroup)
        self.runners[job.job_id] = runner
        try:
            return runner.run()
        finally:
            del self.runners[job.job_id]

    def cancel(self, job_id: str) -> None:
        runner = self.runners.get(job_id)
        super().cancel(job_id)
        if runner is not None:
            runner.terminate()

    def tick(self) -> None:
        """Start and run, to the end, every pending job Slurm would start: its dependency is met,
        and no other job of its name runs when it is a singleton."""
        for job in list(self.jobs.values()):
            if job.state is not JobState.PENDING or not self.ready(job.job_id):
                continue
            self.start(job.job_id, at=time.time())
            code = self._run(job)
            if job.state is JobState.CANCELLED:
                continue
            ending = {EXIT_PUBLISHED: JobState.COMPLETED, EXIT_TERMINATED: JobState.CANCELLED}
            self.finish(job.job_id, ending.get(code, JobState.FAILED))


def bounded_sleep(*, calls: int = 50, seconds: float = 0.1) -> Callable[[float], None]:
    """A sleep for tests of commands that wait: it sleeps *seconds* whatever it is asked for,
    and raises after *calls* calls, so a command that would wait forever fails the test."""
    count = [0]

    def sleep(requested: float) -> None:
        count[0] += 1
        if count[0] > calls:
            raise AssertionError(f"slept more than {calls} times")
        time.sleep(seconds)

    return sleep


def _create(plan: Plan, slurm: LocalSlurm) -> Orchestrator:
    return Orchestrator.create(plan, slurm, holder=Holder(host="login", pid=1, job_id=None, purpose="orchestrator"),
                               sleep=lambda s: None)


def _drive(orch: Orchestrator, slurm: LocalSlurm, passes: int = 60) -> Plan:
    """Pass until the instance ends. Jobs run only after a pass that changed nothing, as on a
    cluster where jobs queue for longer than the orchestrator takes to submit the chain."""
    for _ in range(passes):
        plan, done, acted = orch.step()
        if done:
            return plan
        if not acted:
            slurm.tick()
    raise AssertionError("the instance did not finish")


def _attempts(plan: Plan, stage: str) -> list[int]:
    return plan.layout.attempts(PosixStore(plan.store_root), stage)


def test_max_policy_runs_both_stages_in_one_allocation(tmp_path):
    slurm = LocalSlurm()
    plan = _drive(_create(noop_plan(tmp_path), slurm), slurm)
    assert plan.outcome.status == "succeeded" and len(slurm.submits()) == 1
    assert _attempts(plan, "noop_a") == [1] and _attempts(plan, "noop_b") == [1]


def test_split_policy_chains_two_allocations(tmp_path):
    slurm = LocalSlurm()
    plan = _drive(_create(noop_plan(tmp_path, gpu_policy="split"), slurm), slurm)
    assert plan.outcome.status == "succeeded"
    assert [s[2] for s in slurm.submits()] == [None, "100"]


def test_a_killed_allocation_is_retried_from_the_stage_it_was_in(tmp_path):
    slurm = LocalSlurm(kill={"100": "noop_b"})
    decls = noop_two(gpu_kind="H200", sleep_sec=2, stage_time_sec=600)
    plan = _drive(_create(noop_plan(tmp_path, decls=decls), slurm), slurm)
    assert plan.outcome.status == "succeeded"
    assert plan.allocation(0).failure.stage == "noop_b"
    # The fault is the runner's own record of being terminated, which the orchestrator read from
    # the runner's status file; Slurm's account of the job would read differently.
    assert plan.allocation(0).failure.fault.reason == "the allocation was terminated"
    assert plan.allocation(1).retry_of == 0 and plan.allocation(1).stages == ["noop_b"]
    assert _attempts(plan, "noop_a") == [1] and _attempts(plan, "noop_b") == [1, 2]
    assert read_manifest(PosixStore(plan.store_root), plan.layout, "noop_b")[0].attempt == 2


def test_an_orchestrator_killed_mid_stage_is_reattached_without_relaunching(tmp_path):
    slurm = LocalSlurm()
    # The job is silent for longer than allocation_dead_after_sec unless the orchestrator reads
    # the runner's heartbeats from its status file.
    decls = noop_two(gpu_kind="H200", sleep_sec=3, stage_time_sec=600)
    first = _create(noop_plan(tmp_path, decls=decls, heartbeat_sec=1, allocation_dead_after_sec=3), slurm)
    first.step()
    job = threading.Thread(target=slurm.tick)
    job.start()
    time.sleep(1.0)
    # The first orchestrator is gone; its lease is still on the store.
    again = Orchestrator.attach(first.store, first.layout, slurm,
                                holder=Holder(host="login", pid=2, job_id=None, purpose="orchestrator"),
                                alive=lambda h: False, sleep=lambda s: None)
    plan, done, _ = again.step()
    assert not done and plan.lease_generation == 2
    while job.is_alive():
        again.step()
        time.sleep(0.5)
    plan = _drive(again, slurm)
    assert plan.outcome.status == "succeeded" and len(slurm.submits()) == 1


def test_a_split_retry_repoints_the_pending_dependent(tmp_path):
    slurm = LocalSlurm()
    decls = noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600, fail_attempts=1)
    plan = _drive(_create(noop_plan(tmp_path, decls=decls, gpu_policy="split"), slurm), slurm)
    assert plan.outcome.status == "succeeded"
    assert ("rewire", "101", "102") in slurm.calls
    assert _attempts(plan, "noop_a") == [1, 2]


def test_exhausted_retries_void_and_cancel_the_pending_dependent(tmp_path):
    slurm = LocalSlurm()
    decls = noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600, fail_attempts=5)
    plan = noop_plan(tmp_path, decls=decls, gpu_policy="split", retry={"noop_a": {"max": 1}})
    plan = _drive(_create(plan, slurm), slurm)
    assert plan.outcome.status == "void" and "retries exhausted (1 of 1)" in plan.outcome.reason
    # The reason is noop_a's own fault, from its attempt, not a different break.
    assert "planted fault on attempt 2 of 5" in plan.outcome.reason
    assert _attempts(plan, "noop_a") == [1, 2]
    assert slurm.jobs["101"].state is JobState.CANCELLED and slurm.cancels() == ["101"]


def contestant_fault_in(stage: str, monkeypatch) -> FaultRecord:
    """Make *stage* of the no-op pipelines fail with a contestant fault, as a contestant role
    that ran out of memory does; return the fault."""
    fault = FaultRecord(fault_class="contestant", reason="contestant role server ran out of memory",
                        retryable=False, evidence={"roles": "server"})
    real = NoopStage.run

    def run(self, ctx):
        if ctx.stage.name == stage:
            raise StageFault(fault)
        return real(self, ctx)

    monkeypatch.setattr(NoopStage, "run", run)
    return fault


def test_a_contestant_fault_ends_the_instance_failed_and_runs_nothing_again(tmp_path, monkeypatch):
    slurm = LocalSlurm()
    contestant_fault_in("noop_a", monkeypatch)
    plan = noop_plan(tmp_path, gpu_policy="split", retry={"noop_a": {"max": 2}})
    plan = _drive(_create(plan, slurm), slurm)
    assert (plan.outcome.status, plan.outcome.reason) == ("failed", "noop_a: contestant role server ran out of memory")
    assert plan.outcome.fault.fault_class == "contestant"
    assert _attempts(plan, "noop_a") == [1] and len(plan.allocations) == 2
    assert slurm.jobs["101"].state is JobState.CANCELLED and slurm.cancels() == ["101"]


def test_hls_cancel_mid_stage_stops_the_runner_and_ends_the_instance_cancelled(tmp_path):
    slurm = LocalSlurm()
    # noop_b would run for a minute: the test ends quickly only if cancel stops its roles.
    a, b = noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600)
    first = _create(noop_plan(tmp_path, decls=[a, replace(b, params={**b.params, "sleep_sec": 60})]), slurm)
    first.step()
    # The orchestrator is gone, so hls cancel takes the lease and cancels the job itself.
    first.handle.release()
    job = threading.Thread(target=slurm.tick)
    job.start()
    status_key = first.layout.allocation_status(0, 1)
    store = PosixStore(tmp_path / "store")

    def runner_stage() -> str | None:
        if not store.exists(status_key):
            return None
        return AllocationStatus.model_validate_json(store.read_bytes(status_key)).stage

    deadline = time.time() + 30
    while runner_stage() != "noop_b" and time.time() < deadline:
        time.sleep(0.05)
    assert runner_stage() == "noop_b"
    time.sleep(0.5)
    lines: list[str] = []
    code = commands.cancel(first.layout.root, config=tmp_path / "unused.toml", store=tmp_path / "store",
                           scheduler=slurm, echo=lines.append, sleep=bounded_sleep())
    job.join(20)
    assert not job.is_alive() and code == 0
    plan = first.plans.latest()
    assert plan.outcome.status == "cancelled" and lines[-1].endswith("cancelled: cancelled by hls cancel")
    assert slurm.jobs["100"].state is JobState.CANCELLED and "100" in slurm.cancels()
    assert store.exists(plan.layout.manifest("noop_a")) and not store.exists(plan.layout.manifest("noop_b"))
    assert AllocationStatus.model_validate_json(store.read_bytes(status_key)).state == "terminated"
