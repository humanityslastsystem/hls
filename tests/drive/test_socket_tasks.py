"""``site.evaluation_socket_tasks``: an allocation that runs a scored server asks for two tasks
of the server's CPUs with at most one per socket, so the server's step lands on one socket, on
the first submission and on every retry; without the setting the allocation is one task."""

from __future__ import annotations

from pathlib import Path

import pytest

from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.plan import Plan
from hls.pipeline.planner import PlanError, StageDecl, socket_tasks
from hls.pipeline.reconcile import Observation, decide
from hls.pipeline.stage import most_roles
from hls.spec.resources import ResourceShape
from hls.store.lock import Holder
from hls.substrate.base import JobInfo, JobState
from hls.substrate.slurm.scheduler import submit_argv
from tests.drive.test_stage import ARTIFACT_SHAPE, SHAPE, _params
from tests.pipeline.fake_scheduler import FakeScheduler
from tests.pipeline.plans import noop_plan
from tests.pipeline.test_reconcile import _sub

ON = {"evaluation_socket_tasks": True}


def _plan(tmp_path: Path, site: dict | None = None, **run) -> Plan:
    return noop_plan(tmp_path, decls=[StageDecl("deploy_drive", "deploy_drive", SHAPE, _params())], site=site, **run)


def test_a_scored_servers_allocation_asks_for_its_cpus_twice_as_two_tasks(tmp_path):
    [alloc] = _plan(tmp_path, ON).allocations
    assert alloc.placement.socket_tasks == ARTIFACT_SHAPE.cpus
    assert alloc.request.cpus == 2 * ARTIFACT_SHAPE.cpus and alloc.request.cpus >= SHAPE.cpus


def test_without_the_setting_the_allocation_is_one_task_of_the_stages_cpus(tmp_path):
    [alloc] = _plan(tmp_path).allocations
    assert alloc.placement.socket_tasks is None and alloc.request.cpus == SHAPE.cpus


def test_an_allocation_without_a_scored_server_is_one_task_whatever_the_setting(tmp_path):
    plan = noop_plan(tmp_path, site=ON)
    assert all(a.placement.socket_tasks is None for a in plan.allocations)


def test_a_stage_needing_more_cpus_than_the_two_tasks_hold_is_refused(tmp_path):
    plan = _plan(tmp_path, ON)
    big = plan.stage("deploy_drive").model_copy(update={"shape": SHAPE.model_copy(update={"cpus": 5})})
    with pytest.raises(PlanError, match="two tasks"):
        socket_tasks([big], plan.settings, partition=None)


def test_the_submission_asks_slurm_for_two_tasks_one_per_socket(tmp_path):
    sched = FakeScheduler()
    orch = Orchestrator.create(_plan(tmp_path, ON), sched, holder=Holder(host="login", pid=1, job_id=None,
                                                                          purpose="orchestrator"),
                               clock=lambda: 10_000.0, sleep=lambda s: None)
    orch.step()
    [job] = sched.jobs.values()
    args = submit_argv(job.request)
    assert "--ntasks=2" in args and "--ntasks-per-socket=1" in args
    assert f"--cpus-per-task={ARTIFACT_SHAPE.cpus}" in args and "--ntasks=1" not in args


def test_a_retry_of_a_scored_servers_allocation_keeps_its_two_tasks(tmp_path):
    plan = _plan(tmp_path, ON)
    alloc = plan.allocation(0)
    plan = _sub(plan, 0, job_id="100")
    obs = Observation(now=2000.0, published=frozenset(), jobs={"100": JobInfo("100", JobState.FAILED)}, tags={},
                      statuses={}, cancel_requested=False)
    retry = decide(plan, obs).plan.allocation(1)
    assert retry.retry_of == 0
    assert (retry.placement.socket_tasks, retry.request.cpus) == (alloc.placement.socket_tasks, alloc.request.cpus)


def test_an_allocation_on_the_cpu_partition_never_takes_socket_tasks(tmp_path):
    plan = _plan(tmp_path, ON)
    assert socket_tasks(plan.stages, plan.settings, partition="cpu-a") is None


def test_a_plan_whose_scored_servers_differ_in_cpus_is_refused(tmp_path):
    plan = _plan(tmp_path, ON)
    stage = plan.stage("deploy_drive")
    params = dict(stage.params)
    params["artifact_shape"] = ResourceShape(**{**ARTIFACT_SHAPE.model_dump(), "cpus": 1}).model_dump(mode="json")
    other = stage.model_copy(update={"name": "deploy_drive_b", "params": params})
    with pytest.raises(PlanError, match="servers"):
        socket_tasks([stage, other], plan.settings, partition=None)


def test_a_scored_servers_allocation_runs_under_the_qos_its_time_with_headroom_needs(tmp_path):
    """On a site with QOS limits the two-task allocation is planned under the QOS that holds its
    time with the launch headroom in it, sbatch asks for that QOS and the two tasks, and a retry
    keeps both."""
    site = {**ON, "qos_by_hours": {"short": 0.5, "long": 168}, "step_launch_allowance_sec": 150}
    plan = _plan(tmp_path, site)
    alloc = plan.allocation(0)
    headroom = 150 * most_roles("deploy_drive", plan.stage("deploy_drive").params)
    assert alloc.request.time_sec == SHAPE.time_sec + 360 + headroom > 1800
    assert (alloc.placement.qos, alloc.placement.socket_tasks) == ("long", ARTIFACT_SHAPE.cpus)
    # Without the headroom the same stage fits the short limit.
    assert _plan(tmp_path, {**site, "step_launch_allowance_sec": 0}).allocation(0).placement.qos == "short"
    sched = FakeScheduler()
    orch = Orchestrator.create(plan, sched, holder=Holder(host="login", pid=1, job_id=None, purpose="orchestrator"),
                               clock=lambda: 10_000.0, sleep=lambda s: None)
    orch.step()
    [job] = sched.jobs.values()
    assert {"--qos=long", "--ntasks=2", "--ntasks-per-socket=1"} <= set(submit_argv(job.request))
    plan = _sub(plan, 0, job_id="100")
    obs = Observation(now=2000.0, published=frozenset(), jobs={"100": JobInfo("100", JobState.FAILED)}, tags={},
                      statuses={}, cancel_requested=False)
    retry = decide(plan, obs).plan.allocation(1)
    assert (retry.placement.qos, retry.placement.socket_tasks, retry.request.time_sec) == ("long", ARTIFACT_SHAPE.cpus,
                                                                       alloc.request.time_sec)
