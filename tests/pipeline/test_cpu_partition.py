"""Cache warmers on the site's CPU partition: every allocation of a CPU plan goes there, with no
GPU floor, its retries stay there, and a run's plan never does."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK, noop_one
from hls.pipeline.plan import Plan, Submission, Toolchain, submission_tag
from hls.pipeline.planner import PlanError, build_plan
from hls.pipeline.reconcile import Observation, Submit, decide
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.substrate.base import JobInfo, JobState, SubmitRequest
from hls.substrate.slurm.resources import SlurmPlacement
from hls.substrate.slurm.scheduler import submit_argv
from tests.pipeline.fake_scheduler import FakeScheduler
from tests.pipeline.plans import CODE, noop_plan, settings

CPU_SITE = {"cpu_partition": "cpu-a"}
HOLDER = Holder(host="login", pid=1, job_id=None, purpose="orchestrator")


def _plan(tmp_path: Path, *, partition: str = "cpu", site: dict | None = None, gpus: int = 0, **run) -> Plan:
    """A first plan of the one-stage no-op pipeline, on a CPU role unless *gpus* is set."""
    decls = noop_one(gpus=gpus, gpu_kind="H200", sleep_sec=0, stage_time_sec=600)
    return build_plan(decls=decls, task=NOOP_TASK, pipeline="noop", layout=InstanceLayout.rollout(NOOP_TASK, "w1"),
                      settings=settings(tmp_path, site=site, **run), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-warm", known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=(), now=1000.0, partition=partition)


@dataclass
class RecordingScheduler(FakeScheduler):
    """A fake scheduler that also keeps the arguments of every --test-only."""

    checked: list[tuple[str, ...]] = field(default_factory=list)

    def test_only(self, request: SubmitRequest) -> str:
        self.checked.append(submit_argv(request))
        return super().test_only(request)


def test_a_cpu_plan_runs_on_the_cpu_partition_without_the_gpu_floor(tmp_path):
    plan = _plan(tmp_path, site={**CPU_SITE, "min_gpus_per_job": 1})
    assert [a.placement.partition for a in plan.allocations] == ["cpu-a"]
    request = plan.allocation(0).request
    assert (request.gpus, request.gpu_kind, request.cpus) == (0, None, 2)
    assert plan.warnings == []


@pytest.mark.parametrize("site", [CPU_SITE, None])
def test_a_cpu_plan_refuses_a_gpu_stage(tmp_path, site):
    with pytest.raises(PlanError, match=r"CPU stages only; these ask for GPUs: noop \(1\)"):
        _plan(tmp_path, site=site, gpus=1)


def test_without_a_cpu_partition_a_cpu_plan_uses_the_gpu_partition_and_says_so(tmp_path):
    plan = _plan(tmp_path)
    assert [a.placement.partition for a in plan.allocations] == [None]
    assert plan.allocation(0).request.gpus == 0
    assert plan.warnings == ["site.cpu_partition is not set, so this CPU-only pipeline runs on the GPU partition "
                             "gpu-a"]
    # There the GPU partition's floor applies, and a CPU-only pipeline names no GPU kind to ask for.
    with pytest.raises(PlanError, match="at least 1 GPUs per job"):
        _plan(tmp_path, site={"min_gpus_per_job": 1})


def test_a_gpu_floor_without_a_cpu_partition_refuses_a_cpu_plan_naming_the_setting_to_add(tmp_path):
    with pytest.raises(PlanError, match=r"at least 1 GPUs per job.*site\.cpu_partition is not set.*set site\."
                                        r"cpu_partition to a partition that takes jobs with no GPU"):
        _plan(tmp_path, site={"min_gpus_per_job": 1})
    # A run's plan never takes the CPU partition, so its refusal says nothing of one.
    with pytest.raises(PlanError, match="at least 1 GPUs per job") as refused:
        _plan(tmp_path, partition="gpu", site={"min_gpus_per_job": 1})
    assert "cpu_partition" not in str(refused.value)


def test_a_run_plan_never_takes_the_cpu_partition(tmp_path):
    plan = noop_plan(tmp_path, site=CPU_SITE)
    assert [(a.placement.partition, a.request.gpus) for a in plan.allocations] == [(None, 2)]


def test_an_allocation_partition_is_never_empty():
    with pytest.raises(ValidationError, match="at least 1 character"):
        SlurmPlacement(partition="", qos=None, socket_tasks=None, exclude=[])


def test_preflight_and_submission_ask_for_the_allocations_partition(tmp_path):
    sched = RecordingScheduler()
    orch = Orchestrator.create(_plan(tmp_path, site=CPU_SITE), sched, holder=HOLDER, clock=lambda: 10_000.0,
                               sleep=lambda s: None)
    assert len(sched.checked) == 1
    assert "--partition=cpu-a" in sched.checked[0] and "--gpus=0" in sched.checked[0]
    assert not any(a.startswith("--partition=gpu") for a in sched.checked[0])
    plan, _, acted = orch.step()
    assert acted and sched.submits() == [("submit", f"hls:{plan.instance}/0/1", None)]
    args = submit_argv(sched.jobs["100"].request)
    assert "--partition=cpu-a" in args and "--gpus=0" in args


def test_a_retry_of_a_cpu_allocation_stays_on_the_cpu_partition(tmp_path):
    plan = _plan(tmp_path, site={**CPU_SITE, "min_gpus_per_job": 1})
    sub = Submission(n=1, tag=submission_tag(plan.instance, 0, 1), intended_at=1000.0, after_job=None,
                     status="submitted", job_id="100", submitted_at=1000.0)
    plan = plan.with_allocation(plan.allocation(0).model_copy(update={"submissions": [sub]}))
    obs = Observation(now=2000.0, published=frozenset(), jobs={"100": JobInfo("100", JobState.FAILED)}, tags={},
                      statuses={}, cancel_requested=False)
    d = decide(plan, obs)
    retry = d.plan.allocation(1)
    assert (retry.retry_of, retry.placement.partition, retry.request.gpus) == (0, "cpu-a", 0)
    assert d.effects == [Submit(1, 1)]


def test_a_run_and_its_retry_stay_on_the_gpu_partition_and_its_floor_on_a_site_with_a_cpu_partition(tmp_path):
    site = {**CPU_SITE, "min_gpus_per_job": 4}
    sched = RecordingScheduler()
    orch = Orchestrator.create(noop_plan(tmp_path, site=site), sched, holder=HOLDER, clock=lambda: 10_000.0,
                               sleep=lambda s: None)
    assert "--partition=gpu-a" in sched.checked[0] and "--gpus=h200:4" in sched.checked[0]
    plan, _, _ = orch.step()
    args = submit_argv(sched.jobs["100"].request)
    assert "--partition=gpu-a" in args and "--partition=cpu-a" not in args
    sub = Submission(n=1, tag=submission_tag(plan.instance, 0, 1), intended_at=1000.0, after_job=None,
                     status="submitted", job_id="100", submitted_at=1000.0)
    plan = plan.with_allocation(plan.allocation(0).model_copy(update={"submissions": [sub]}))
    obs = Observation(now=2000.0, published=frozenset(), jobs={"100": JobInfo("100", JobState.FAILED)}, tags={},
                      statuses={}, cancel_requested=False)
    retry = decide(plan, obs).plan.allocation(1)
    assert (retry.retry_of, retry.placement.partition, retry.request.gpus) == (0, None, 4)


def test_a_takeover_keeps_a_cpu_plans_partition_and_submits_to_it(tmp_path):
    sched = RecordingScheduler()
    orch = Orchestrator.create(_plan(tmp_path, site=CPU_SITE), sched, holder=HOLDER, clock=lambda: 10_000.0,
                               sleep=lambda s: None)
    before = orch.plans.latest()
    again = Orchestrator.attach(orch.store, orch.layout, sched, holder=HOLDER.model_copy(update={"pid": 2}),
                                alive=lambda h: False, clock=lambda: 10_000.0, sleep=lambda s: None)
    taken = again.plans.latest()
    assert (taken.revision, taken.lease_generation) == (before.revision + 1, 2)
    assert [a.placement.partition for a in taken.allocations] == ["cpu-a"]
    again.step()
    assert "--partition=cpu-a" in submit_argv(sched.jobs["100"].request)


def test_a_plan_cannot_send_gpus_to_another_partition(tmp_path):
    plan = _plan(tmp_path, site=CPU_SITE)
    gpu_request = ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=4, time_sec=60)
    bad = plan.with_allocation(plan.allocation(0).model_copy(update={"request": gpu_request}))
    with pytest.raises(ValueError, match=r"allocation 0 goes to partition cpu-a and asks for 1 GPUs"):
        Plan.model_validate(bad.model_dump())
    # The same request on the GPU partition is a run's allocation.
    on_gpus = plan.allocation(0).placement.model_copy(update={"partition": None})
    ok = plan.with_allocation(plan.allocation(0).model_copy(update={"request": gpu_request, "placement": on_gpus}))
    assert Plan.model_validate(ok.model_dump()).allocation(0).request.gpus == 1


def test_a_cpu_plan_no_qos_holds_is_refused_naming_the_cpu_partition(tmp_path):
    # The stage's 600 s and the allocation's 60 s of overhead: 11 minutes, past a 10-minute QOS.
    with pytest.raises(PlanError) as e:
        _plan(tmp_path, site={**CPU_SITE, "qos_by_hours": {"short": 1 / 6}})
    assert "on partition cpu-a ask" in str(e.value) and "gpu-a" not in str(e.value)
