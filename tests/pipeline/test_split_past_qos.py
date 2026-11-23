"""Allocations the GPU policy groups that no QOS of the site holds: split in stage order into a
chain of allocations that each fit the longest limit, a stage no QOS holds alone refused, and a
pipeline's own layout never split."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import AllocationDecl, AllocationPastQOS, PlanError, StageDecl, build_plan
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.substrate.slurm.scheduler import submit_argv
from tests.pipeline.fake_scheduler import FakeScheduler
from tests.pipeline.plans import CODE, noop_plan, settings
from tests.pipeline.test_end_to_end_local import LocalSlurm, _drive

HOLDER = Holder(host="login", pid=1, job_id=None, purpose="orchestrator")
#: What the tests' settings add to an allocation's stage times: the allocation overhead, with no
#: launch allowance unless a test sets one (``tests.pipeline.plans.settings``).
OVERHEAD_SEC = 360
#: A day, the one QOS limit of some sites.
DAY = {"normal": 24}


def _cpu(name: str, hours: float, *, after: tuple[str, ...] = ()) -> StageDecl:
    """A no-op stage with one CPU role that takes *hours*."""
    return StageDecl(name, "noop", ResourceShape(cpus=2, memory_gib=6, time_sec=round(hours * 3600)),
                     {"gpu_roles": 0, "cpu_role": True}, after=after)


def _gpu(name: str, hours: float, *, gpus: int = 1) -> StageDecl:
    """A no-op stage with one GPU role on *gpus* H200s that takes *hours*."""
    return StageDecl(name, "noop", ResourceShape(gpus=gpus, gpu_kind="H200", cpus=4, memory_gib=6,
                                                 time_sec=round(hours * 3600)), {"gpu_roles": 1})


def _layout(plan: Plan) -> list[tuple[list[str], int | None, list[int]]]:
    """Each allocation's stages, the allocation it depends on and those it waits for."""
    return [(a.stages, a.depends_on, a.waits_for) for a in plan.allocations]


#: Six stages whose 47 h no day holds: the first three fill a day exactly with the overhead, and the
#: last three take 23 h.
SIX = [_cpu("a", 5), _cpu("b", 15), _cpu("c", 24 - OVERHEAD_SEC / 3600 - 20), _cpu("d", 20), _cpu("e", 2),
       _cpu("f", 1)]


def test_an_allocation_no_qos_holds_is_split_in_stage_order_as_a_chain(tmp_path):
    plan = noop_plan(tmp_path, decls=SIX, site={"qos_by_hours": DAY})
    assert _layout(plan) == [(["a", "b", "c"], None, []), (["d", "e", "f"], 0, [])]
    # Each takes as many following stages as the day holds: the first ends at the day exactly.
    assert [a.request.time_sec for a in plan.allocations] == [24 * 3600, 23 * 3600 + OVERHEAD_SEC]
    assert [a.placement.qos for a in plan.allocations] == ["normal", "normal"]
    whole = sum(s.shape.time_sec for s in plan.stages) + OVERHEAD_SEC
    assert plan.warnings == [
        f"the max GPU policy groups stages a to f into one allocation, but it asks for {whole / 3600:.2f} h, more "
        "than any QOS of the site lets a job on partition gpu-a ask (site.qos_by_hours): the longest, normal, "
        "allows 24 h; they run as allocations 0 to 1 instead, as many stages each as that limit holds, each queued "
        "after the one before"]


def test_the_split_counts_each_stages_launch_headroom(tmp_path):
    # 150 s for each stage's one role moves the first boundary: a, b and c no longer fit a day.
    plan = noop_plan(tmp_path, decls=SIX, site={"qos_by_hours": DAY, "step_launch_allowance_sec": 150})
    assert [a.stages for a in plan.allocations] == [["a", "b"], ["c"], ["d", "e", "f"]]
    assert [a.depends_on for a in plan.allocations] == [None, 0, 1]
    assert plan.allocation(2).request.time_sec == 23 * 3600 + OVERHEAD_SEC + 3 * 150


def test_each_piece_takes_the_least_qos_that_holds_it_and_the_split_is_by_the_longest(tmp_path):
    decls = [_cpu("a", 20), _cpu("b", 3), _cpu("c", 2)]
    plan = noop_plan(tmp_path, decls=decls, site={"qos_by_hours": {"short": 4, "normal": 24}})
    assert [(a.stages, a.placement.qos) for a in plan.allocations] == [(["a", "b"], "normal"), (["c"], "short")]
    # The site's own QOS is kept where it holds a piece; the packing is by the longest limit.
    decls = [_cpu("a", 30), _cpu("b", 10), _cpu("c", 10)]
    plan = noop_plan(tmp_path, decls=decls, site={"qos": "normal", "qos_by_hours": {"normal": 24, "long": 48}})
    assert [(a.stages, a.placement.qos) for a in plan.allocations] == [(["a", "b"], "long"), (["c"], "normal")]
    assert "the longest, long, allows 48 h" in plan.warnings[0]


def test_a_stage_no_qos_holds_alone_is_refused_naming_it_its_time_and_the_limit(tmp_path):
    with pytest.raises(AllocationPastQOS) as e:
        noop_plan(tmp_path, decls=[_cpu("a", 2), _cpu("big", 25)], site={"qos_by_hours": DAY})
    assert str(e.value) == ("stage big takes 25.00 h; alone, with the runner's overhead and its launch headroom, it "
                            "asks for 25.10 h, more than any QOS of the site lets a job on partition gpu-a ask "
                            "(site.qos_by_hours): the longest, normal, allows 24 h")
    # The overhead counts: a stage of the day less a minute does not fit it.
    with pytest.raises(AllocationPastQOS, match="^stage edge takes 23.98 h; alone"):
        noop_plan(tmp_path, decls=[_cpu("edge", 24 - 1 / 60)], site={"qos_by_hours": DAY})


def test_a_site_with_no_limits_keeps_the_policys_allocations(tmp_path):
    assert _layout(noop_plan(tmp_path, decls=SIX)) == [(["a", "b", "c", "d", "e", "f"], None, [])]
    assert noop_plan(tmp_path, decls=SIX).warnings == []
    # A limit that holds the whole run changes nothing either.
    plan = noop_plan(tmp_path, decls=SIX, site={"qos_by_hours": {"long": 168}})
    assert _layout(plan) == [(["a", "b", "c", "d", "e", "f"], None, [])] and plan.warnings == []


def test_the_split_policys_allocations_are_split_each_on_its_own(tmp_path):
    decls = [_cpu("a", 10), _cpu("b", 10), _gpu("c", 10), _gpu("d", 10), _gpu("e", 10)]
    plan = noop_plan(tmp_path, decls=decls, gpu_policy="split", site={"qos_by_hours": DAY})
    # The CPU pair fits a day; the GPU stages, a group of their own, are split two and one.
    assert _layout(plan) == [(["a", "b"], None, []), (["c", "d"], 0, []), (["e"], 1, [])]
    [note] = plan.warnings
    assert note.startswith("the split GPU policy groups stages c to e into one allocation") and (
        "allocations 1 to 2 instead" in note)


def test_a_stage_reads_a_stage_of_any_allocation_before_it_in_the_chain(tmp_path):
    decls = [_cpu("a", 20), _cpu("b", 20), _cpu("c", 20, after=("a", "b"))]
    plan = noop_plan(tmp_path, decls=decls, site={"qos_by_hours": DAY})
    assert _layout(plan) == [(["a"], None, []), (["b"], 0, []), (["c"], 1, [])]


def test_each_piece_asks_for_its_own_stages_resources(tmp_path):
    decls = [_gpu("wide", 20, gpus=2), _cpu("narrow", 20)]
    plan = noop_plan(tmp_path, decls=decls, site={"qos_by_hours": DAY})
    assert [(a.request.gpus, a.request.cpus) for a in plan.allocations] == [(2, 4), (0, 2)]
    # The CPU piece idles no GPU, so only the plan's split is noted.
    assert len(plan.warnings) == 1 and plan.warnings[0].startswith("the max GPU policy groups")


def test_the_idle_gpus_the_sites_floor_sets_name_the_floor(tmp_path):
    """A piece of stages below the site's GPU floor holds the floor's GPUs, which no policy
    avoids: its idle GPUs name the floor, while a piece the policy sized still suggests split."""
    decls = [_gpu("wide", 10, gpus=2), _cpu("prep", 1), _cpu("narrow", 20)]
    plan = noop_plan(tmp_path, decls=decls, site={"qos_by_hours": DAY, "min_gpus_per_job": 1})
    assert [(a.stages, a.request.gpus) for a in plan.allocations] == [(["wide", "prep"], 2), (["narrow"], 1)]
    split, prep, narrow = plan.warnings
    assert split.startswith("the max GPU policy groups")
    assert "stage prep uses 0, so 2 idle" in prep and "--run.gpu_policy=split" in prep
    assert "allocation 1 holds 1 GPUs; stage narrow uses 0, so 1 idle" in narrow
    assert "site.min_gpus_per_job = 1 sets the allocation's GPU count" in narrow and "gpu_policy" not in narrow
    # So does one allocation whose largest stage asks for the floor, under max as under split.
    [floor] = noop_plan(tmp_path, decls=[_cpu("prep", 1), _gpu("work", 1)], site={"min_gpus_per_job": 1}).warnings
    assert "stage prep uses 0, so 1 idle" in floor and "min_gpus_per_job" in floor and "gpu_policy" not in floor


def _plan(tmp_path: Path, decls, groups=None, *, singleton: bool = False, site=None) -> Plan:
    """A plan of *decls* laid out as *groups*, or as the GPU policy groups them with None; with
    *singleton*, one that runs one at a time with the site's other warmers."""
    return build_plan(decls=decls, task=NOOP_TASK, pipeline="noop", layout=InstanceLayout.rollout(NOOP_TASK, "r1"),
                      settings=settings(tmp_path, site=site), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-warmer" if singleton else "hls-noop", known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=(), now=1000.0, groups=groups, singleton=singleton)


def test_a_warmers_split_waits_for_each_piece_to_publish(tmp_path):
    # A singleton allocation depends on no other by afterok, so each piece waits for the one
    # before to publish, and runs one at a time with the site's other warmers all the same.
    decls = [_cpu("fill", 20), _cpu("more", 20, after=("fill",)), _cpu("last", 20, after=("fill",))]
    plan = _plan(tmp_path, decls, singleton=True, site={"qos_by_hours": DAY})
    assert _layout(plan) == [(["fill"], None, []), (["more"], None, [0]), (["last"], None, [1])]
    assert [a.singleton for a in plan.allocations] == [True, True, True]


def test_a_laid_out_allocation_no_qos_holds_is_refused_not_split(tmp_path):
    decls = [_cpu("a", 20), _cpu("b", 20), _cpu("join", 1, after=("a", "b"))]
    groups = [AllocationDecl(("a", "b")), AllocationDecl(("join",), depends_on=0)]
    with pytest.raises(AllocationPastQOS) as e:
        _plan(tmp_path, decls, groups, site={"qos_by_hours": DAY})
    assert str(e.value) == ("allocation 0: it asks for 40.10 h, more than any QOS of the site lets a job on partition "
                            "gpu-a ask (site.qos_by_hours): the longest, normal, allows 24 h; the pipeline lays "
                            "this allocation out itself, so the planner does not split it")
    assert isinstance(e.value, PlanError)


def test_the_orchestrator_submits_the_pieces_as_a_chain(tmp_path):
    sched = FakeScheduler()
    plan = noop_plan(tmp_path, decls=SIX, site={"qos_by_hours": DAY})
    orch = Orchestrator.create(plan, sched, holder=HOLDER, clock=lambda: 10_000.0, sleep=lambda s: None)
    orch.step()
    plan, _, _ = orch.step()
    assert [s[2] for s in sched.submits()] == [None, "100"]
    assert [a.current().job_id for a in plan.allocations] == ["100", "101"]
    assert all("--qos=normal" in submit_argv(sched.jobs[j].request) for j in ("100", "101"))
    assert "--time=1440" in submit_argv(sched.jobs["100"].request)


def test_a_split_run_ends_succeeded_end_to_end(tmp_path):
    # Two no-op stages of 600 s each and the overhead: 26 minutes, past an 18-minute QOS that holds
    # each with the overhead.
    slurm = LocalSlurm()
    plan = noop_plan(tmp_path, site={"qos_by_hours": {"short": 0.3}})
    assert [a.stages for a in plan.allocations] == [["noop_a"], ["noop_b"]]
    plan = _drive(Orchestrator.create(plan, slurm, holder=HOLDER, sleep=lambda s: None), slurm)
    assert plan.outcome.status == "succeeded"
    assert [s[2] for s in slurm.submits()] == [None, "100"]
