"""Allocations a pipeline lays out itself (``AllocationDecl``): side by side, and an allocation
that waits for several to publish (a fan-in), submitted with no Slurm dependency; and cache
warmers that run one at a time through Slurm's ``singleton`` dependency."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hls.pipeline.events import read_events
from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK
from hls.pipeline.plan import Plan, Submission, Toolchain, submission_tag
from hls.pipeline.planner import AllocationDecl, PlanError, StageDecl, build_plan
from hls.pipeline.reconcile import Observation, Submit, Submitted, decide, record
from hls.pipeline.stage import most_roles
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import JobInfo, JobState
from hls.substrate.slurm.scheduler import submit_argv
from tests.drive.test_stage import ARTIFACT_SHAPE
from tests.drive.test_stage import SHAPE as DRIVE_SHAPE
from tests.drive.test_stage import _params as drive_params
from tests.pipeline.fake_scheduler import FakeScheduler
from tests.pipeline.plans import CODE, settings
from tests.pipeline.test_end_to_end_local import LocalSlurm, _drive
from tests.substrate.slurm_fakes import slurm_request

HOLDER = Holder(host="login", pid=1, job_id=None, purpose="orchestrator")
C, F, R = JobState.COMPLETED, JobState.FAILED, JobState.RUNNING
#: A no-op stage with one GPU role: the role's 2 CPUs and 2 GiB, and the runner's 4 GiB.
GPU = ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=6, time_sec=600)
#: A no-op stage with one CPU role.
CPU = ResourceShape(cpus=2, memory_gib=6, time_sec=600)
#: What these tests' settings add to an allocation's stage times: the allocation overhead, with no
#: launch allowance unless a test sets one (``tests.pipeline.plans.settings``).
OVERHEAD_SEC = 360


def _noop(name: str, *, after: tuple[str, ...] = (), gpu: bool = True) -> StageDecl:
    params = {"gpu_roles": 1} if gpu else {"gpu_roles": 0, "cpu_role": True}
    return StageDecl(name, "noop", GPU if gpu else CPU, params, after=after)


#: Two stages side by side, and a third that reads both.
FAN = [_noop("left"), _noop("right"), _noop("join", after=("left", "right"))]
#: Their layout: each side in its own allocation, and the join once both published.
FAN_GROUPS = [AllocationDecl(("left",)), AllocationDecl(("right",)), AllocationDecl(("join",), waits_for=(0, 1))]


def _plan(tmp_path: Path, decls, groups=None, *, job_name: str = "hls-noop", singleton: bool = False,
          site=None, **run) -> Plan:
    return build_plan(decls=decls, task=NOOP_TASK, pipeline="noop", layout=InstanceLayout.rollout(NOOP_TASK, "r1"),
                      settings=settings(tmp_path, site=site, **run), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name=job_name, known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=(), now=1000.0, groups=groups,
                      singleton=singleton)


def _obs(published=(), jobs=None, now: float = 2000.0) -> Observation:
    infos = {j: JobInfo(j, s) for j, s in (jobs or {}).items()}
    return Observation(now=now, published=frozenset(published), jobs=infos)


def _submitted(plan: Plan, ids: dict[int, str]) -> Plan:
    """*plan* with a submitted first submission of each allocation in *ids*, under its job id."""
    for k, job_id in ids.items():
        sub = Submission(n=1, tag=submission_tag(plan.instance, k, 1), intended_at=1000.0, after_job=None,
                         status="submitted", job_id=job_id, submitted_at=1000.0)
        plan = plan.with_allocation(plan.allocation(k).model_copy(update={"submissions": [sub]}))
    return plan


# ---------------------------------------------------------------------------
# Laying allocations out
# ---------------------------------------------------------------------------


def test_two_side_by_side_allocations_are_submitted_in_the_same_pass(tmp_path):
    plan = _plan(tmp_path, FAN, FAN_GROUPS)
    assert [(a.stages, a.depends_on, a.waits_for) for a in plan.allocations] == [
        (["left"], None, []), (["right"], None, []), (["join"], None, [0, 1])]
    assert all(a.request.gpus == 1 and a.request.time_sec == 600 + OVERHEAD_SEC for a in plan.allocations)
    d = decide(plan, _obs())
    assert d.effects == [Submit(0, 1), Submit(1, 1)]
    assert d.plan.allocation(2).submissions == []


def test_an_allocation_waiting_for_two_is_submitted_once_both_published(tmp_path):
    plan = _submitted(_plan(tmp_path, FAN, FAN_GROUPS), {0: "100", 1: "101"})
    one = decide(plan, _obs(published={"left"}, jobs={"100": C, "101": R}))
    assert one.effects == [] and one.plan.allocation(2).submissions == []
    both = decide(plan, _obs(published={"left", "right"}, jobs={"100": C, "101": C}))
    assert both.effects == [Submit(2, 1)]


def test_a_fan_in_is_submitted_with_no_dependency(tmp_path):
    # Slurm forgets a finished job within minutes, and an afterok names one job: the fan-in
    # waits on the published manifests instead, and is submitted with nothing to wait for.
    plan = _submitted(_plan(tmp_path, FAN, FAN_GROUPS), {0: "100", 1: "101"})
    both = decide(plan, _obs(published={"left", "right"}, jobs={"100": C, "101": R}))
    assert both.plan.allocation(2).submissions[0].after_job is None


def test_a_waited_allocation_retried_counts_as_done_when_its_retry_publishes(tmp_path):
    plan = _submitted(_plan(tmp_path, FAN, FAN_GROUPS), {0: "100", 1: "101"})
    failed = decide(plan, _obs(published={"right"}, jobs={"100": F, "101": C}))
    retry = failed.plan.allocation(3)
    assert (retry.stages, retry.retry_of, retry.depends_on, retry.waits_for) == (["left"], 0, None, [])
    assert Submit(3, 1) in failed.effects and all(e.allocation != 2 for e in failed.effects)
    submitted = record(failed.plan, [Submitted(3, 1, "102")], 2001.0)
    # The retry published left, so every stage of allocation 0 is published and the join goes.
    later = decide(submitted, _obs(published={"left", "right"}, jobs={"100": F, "101": C, "102": C}))
    assert later.effects == [Submit(2, 1)]


def test_a_retry_of_a_fan_in_keeps_what_it_waits_for(tmp_path):
    plan = _submitted(_plan(tmp_path, FAN, FAN_GROUPS), {0: "100", 1: "101", 2: "102"})
    failed = decide(plan, _obs(published={"left", "right"}, jobs={"100": C, "101": C, "102": F}))
    retry = failed.plan.allocation(3)
    assert (retry.stages, retry.retry_of, retry.waits_for) == (["join"], 2, [0, 1])
    assert failed.effects == [Submit(3, 1)]


@pytest.mark.parametrize("groups, match", [
    ([AllocationDecl(("left",)), AllocationDecl(("right",))], r"stages \['join'\] are in no allocation"),
    ([AllocationDecl(("left", "right")), AllocationDecl(("right", "join"), waits_for=(0,))],
     "stage right is in allocations 0 and 1"),
    ([AllocationDecl(("right", "left")), AllocationDecl(("join",), waits_for=(0,))],
     r"allocation 0 runs its stages out of declaration order: \['right', 'left'\]"),
    ([AllocationDecl(("left",)), AllocationDecl(("right",)), AllocationDecl(("join",), waits_for=(0, 3))],
     r"allocation 2 waits for \[3\], which do not come before it"),
    ([AllocationDecl(("left",), depends_on=1), AllocationDecl(("right",)),
      AllocationDecl(("join",), waits_for=(0, 1))],
     r"allocation 0 waits for \[1\], which do not come before it"),
    ([AllocationDecl(("left",)), AllocationDecl(("ghost",)), AllocationDecl(("right", "join"), waits_for=(0,))],
     "allocation 1 names stage ghost, which the plan does not declare"),
    ([AllocationDecl(()), AllocationDecl(("left", "right", "join"))], "allocation 0 holds no stage"),
], ids=["missing", "twice", "order", "forward_wait", "forward_depends", "unknown", "empty"])
def test_groups_must_hold_every_stage_once(tmp_path, groups, match):
    with pytest.raises(PlanError, match=match):
        _plan(tmp_path, FAN, groups)


def test_a_stage_reading_another_groups_stage_must_wait_for_it(tmp_path):
    loose = [AllocationDecl(("left",)), AllocationDecl(("right",)), AllocationDecl(("join",), waits_for=(0,))]
    with pytest.raises(PlanError, match="stage join reads stage right, which runs in allocation 1; allocation 2 "
                                        "neither depends on nor waits for it"):
        _plan(tmp_path, FAN, loose)
    # Through another allocation counts: join waits for right's allocation, which depends on left's.
    chained = [AllocationDecl(("left",)), AllocationDecl(("right",), depends_on=0),
               AllocationDecl(("join",), waits_for=(1,))]
    plan = _plan(tmp_path, FAN, chained)
    assert [(a.depends_on, a.waits_for) for a in plan.allocations] == [(None, []), (0, []), (None, [1])]


def test_a_laid_out_allocation_warns_of_its_idle_gpus_naming_the_layout(tmp_path):
    decls = [_noop("prep", gpu=False), _noop("work")]
    [warning] = _plan(tmp_path, decls, [AllocationDecl(("prep", "work"))]).warnings
    assert "stage prep uses 0, so 1 idle" in warning and "the pipeline's layout" in warning
    assert "gpu_policy" not in warning and "min_gpus_per_job" not in warning


def test_without_a_layout_the_policy_still_chains_allocations(tmp_path):
    plan = _plan(tmp_path, [_noop("prep", gpu=False), _noop("work", after=("prep",))], gpu_policy="split")
    assert [(a.stages, a.depends_on, a.waits_for) for a in plan.allocations] == [
        (["prep"], None, []), (["work"], 0, [])]


#: Three drives of a scored server: two side by side, and a third once both have published.
DRIVES = [StageDecl(f"drive_{k}", "deploy_drive", DRIVE_SHAPE, drive_params()) for k in (1, 2, 3)]
DRIVE_GROUPS = [AllocationDecl(("drive_1",)), AllocationDecl(("drive_2",)),
                AllocationDecl(("drive_3",), waits_for=(0, 1))]
#: A site that holds a scored server to one socket and limits a job's time by its QOS.
SOCKETS_AND_LIMITS = {"evaluation_socket_tasks": True, "qos_by_hours": {"short": 0.5, "long": 168},
                      "step_launch_allowance_sec": 150}


def test_laid_out_and_fan_in_allocations_are_sized_as_a_policys_are(tmp_path):
    """Each allocation of a layout, the fan-in among them, asks for its stages' time with the
    overhead and the launch headroom of its roles, is planned under the QOS that holds that time,
    and asks for a scored server's CPUs as two socket tasks, as an allocation the GPU policy
    makes does; sbatch gets the QOS and the two tasks."""
    plan = _plan(tmp_path, DRIVES, DRIVE_GROUPS, site=SOCKETS_AND_LIMITS)
    headroom = 150 * most_roles("deploy_drive", plan.stage("drive_1").params)
    policy = _plan(tmp_path, DRIVES[:1], site=SOCKETS_AND_LIMITS).allocation(0)
    assert policy.request.time_sec == DRIVE_SHAPE.time_sec + OVERHEAD_SEC + headroom > 1800
    for alloc in plan.allocations:
        assert (alloc.request, alloc.placement.qos, alloc.placement.socket_tasks) == (policy.request, "long", ARTIFACT_SHAPE.cpus)
    # The fan-in, submitted once both sides published, asks sbatch for the same.
    sched = FakeScheduler()
    orch = Orchestrator.create(plan, sched, holder=HOLDER, clock=lambda: 10_000.0, sleep=lambda s: None)
    orch.step()
    for name in ("drive_1", "drive_2"):
        orch.store.write_bytes(orch.layout.manifest(name), b"{}")
    for job_id in ("100", "101"):
        sched.finish(job_id)
    orch.step()
    fan_in = sched.jobs["102"].request
    assert fan_in.tag == submission_tag(plan.instance, 2, 1) and fan_in.after is None
    assert {"--qos=long", "--ntasks=2", "--ntasks-per-socket=1",
            f"--cpus-per-task={ARTIFACT_SHAPE.cpus}"} <= set(submit_argv(fan_in))
    # Without the headroom the same stage fits the short limit, on every allocation of the layout.
    quick = _plan(tmp_path, DRIVES, DRIVE_GROUPS, site={**SOCKETS_AND_LIMITS, "step_launch_allowance_sec": 0})
    assert [a.placement.qos for a in quick.allocations] == ["short"] * 3


def test_a_layout_is_refused_an_overhead_below_the_runners_end_margin(tmp_path):
    with pytest.raises(PlanError, match="allocation_overhead_sec is 299s, less than the 300s"):
        _plan(tmp_path, FAN, FAN_GROUPS, allocation_overhead_sec=299)


def test_a_fan_in_runs_after_both_side_by_side_allocations_end_to_end(tmp_path):
    slurm = LocalSlurm()
    orch = Orchestrator.create(_plan(tmp_path, FAN, FAN_GROUPS), slurm, holder=HOLDER, sleep=lambda s: None)
    plan = _drive(orch, slurm)
    assert plan.outcome.status == "succeeded" and len(plan.allocations) == 3
    submits = slurm.submits()
    assert [s[1] for s in submits] == [submission_tag(plan.instance, k, 1) for k in (0, 1, 2)]
    assert [s[2] for s in submits] == [None, None, None]
    # The join was intended only once both sides had published.
    store = PosixStore(plan.store_root)
    sides = [read_manifest(store, plan.layout, name)[0].times.ended_at for name in ("left", "right")]
    assert plan.allocation(2).submissions[0].intended_at >= max(sides)


# ---------------------------------------------------------------------------
# Warmers one at a time
# ---------------------------------------------------------------------------


def test_warmers_are_submitted_singleton_under_one_job_name(tmp_path):
    warm = [_noop("warm", gpu=False)]
    plan = _plan(tmp_path, warm, job_name="hls-warmer", singleton=True)
    assert [a.singleton for a in plan.allocations] == [True]
    with pytest.raises(PlanError, match=r"site\.warmer_job_name \(hls-warmer\); --job-name hls-noop"):
        _plan(tmp_path, warm, singleton=True)
    named = _plan(tmp_path, warm, job_name="w-kl", singleton=True, site={"warmer_job_name": "w-kl"})
    assert named.job_name == "w-kl"
    sched = FakeScheduler()
    orch = Orchestrator.create(plan, sched, holder=HOLDER, sleep=lambda s: None)
    orch.step()
    [job] = sched.jobs.values()
    assert job.request.singleton and "--job-name=hls-warmer" in submit_argv(job.request)


def test_a_warmers_recorded_sbatch_line_carries_its_singleton_dependency(tmp_path):
    plan = _plan(tmp_path, [_noop("warm", gpu=False)], job_name="hls-warmer", singleton=True)
    sched = FakeScheduler()
    orch = Orchestrator.create(plan, sched, holder=HOLDER, sleep=lambda s: None)
    plan, _, _ = orch.step()
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    [line] = [e["argv"] for e in events if e["kind"] == "submitted"]
    assert line == list(sched.submit_argv(sched.jobs["100"].request)) == plan.allocation(0).current().argv
    assert "--dependency=singleton" in line and "--job-name=hls-warmer" in line


def test_a_warmers_preflight_asks_with_its_singleton_dependency(tmp_path):
    class Asked(FakeScheduler):
        def test_only(self, request):
            self.asked = request
            return super().test_only(request)

    sched = Asked()
    Orchestrator.create(_plan(tmp_path, [_noop("warm", gpu=False)], job_name="hls-warmer", singleton=True), sched,
                        holder=HOLDER, sleep=lambda s: None)
    assert sched.asked.singleton and "--job-name=hls-warmer" in submit_argv(sched.asked)


def test_a_plan_that_is_not_a_warmer_is_never_singleton(tmp_path):
    plan = _plan(tmp_path, FAN, FAN_GROUPS)
    sched = FakeScheduler()
    Orchestrator.create(plan, sched, holder=HOLDER, sleep=lambda s: None).step()
    assert [j.request.singleton for j in sched.jobs.values()] == [False, False]


def test_a_singleton_allocation_waits_on_no_other_allocation(tmp_path):
    decls = [_noop("first", gpu=False), _noop("second", gpu=False, after=("first",))]
    chained = [AllocationDecl(("first",)), AllocationDecl(("second",), depends_on=0)]
    with pytest.raises(PlanError, match="allocation 1 runs one at a time with the site's other warmers and depends "
                                        "on allocation 0"):
        _plan(tmp_path, decls, chained, job_name="hls-warmer", singleton=True)
    # Waiting on published manifests holds no Slurm dependency, so it goes with singleton.
    fan_in = [AllocationDecl(("first",)), AllocationDecl(("second",), waits_for=(0,))]
    plan = _plan(tmp_path, decls, fan_in, job_name="hls-warmer", singleton=True)
    assert [(a.singleton, a.waits_for) for a in plan.allocations] == [(True, []), (True, [0])]
    # So the split policy chains a warmer's allocations that way, not by afterok.
    plan = _plan(tmp_path, [_noop("first", gpu=False), _noop("second", after=("first",))], job_name="hls-warmer",
                 singleton=True, gpu_policy="split")
    assert [(a.singleton, a.depends_on, a.waits_for) for a in plan.allocations] == [(True, None, []),
                                                                                     (True, None, [0])]


def test_a_retry_of_a_singleton_allocation_stays_singleton(tmp_path):
    warmer = _plan(tmp_path, [_noop("warm", gpu=False)], job_name="hls-warmer", singleton=True)
    plan = _submitted(warmer, {0: "100"})
    d = decide(plan, _obs(jobs={"100": F}))
    retry = d.plan.allocation(1)
    assert (retry.retry_of, retry.singleton, retry.depends_on) == (0, True, None)
    assert d.effects == [Submit(1, 1)]


def test_the_local_slurm_runs_one_singleton_job_of_a_name_at_a_time():
    slurm = LocalSlurm()
    first = slurm.submit(slurm_request("a", job_name="hls-warmer", singleton=True))
    second = slurm.submit(slurm_request("b", job_name="hls-warmer", singleton=True))
    slurm.start(first)
    assert not slurm.ready(second)
    # A tick starts only jobs that are ready: the second warmer stays queued behind the first.
    slurm.tick()
    assert slurm.jobs[second].state is JobState.PENDING
    with pytest.raises(AssertionError, match="singleton"):
        slurm.start(second)
    slurm.finish(first)
    assert slurm.ready(second)


def test_a_singleton_job_of_another_name_is_not_held():
    sched = FakeScheduler()
    first = sched.submit(slurm_request("a", job_name="hls-warmer", singleton=True))
    other = sched.submit(slurm_request("b", job_name="hls-other", singleton=True))
    plain = sched.submit(slurm_request("c", job_name="hls-warmer"))
    sched.start(first)
    assert sched.ready(other) and sched.ready(plain)
