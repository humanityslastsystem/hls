"""Plans that grow once every stage has published (``hls.pipeline.growth``): the orchestrator asks
the plan's grower, writes what it adds as a revision with the new allocations' submissions, and
ends the instance when the grower has nothing more, or void when it refuses. The growers and the
granted stage kind here are the tests' own."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from hls.pipeline.commands import known_task_ids
from hls.pipeline.events import read_events
from hls.pipeline.grown import GROWN_STAGES, GrownStages, declare_grower, grown_stages
from hls.pipeline.growth import (GROWERS, GrowthComplete, GrowthDecl, GrowthRefused, PlanGrowth,
                                 UnknownGrower, grow, register_grower)
from hls.pipeline.orchestrator import Orchestrator, request_cancel
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK
from hls.pipeline.plan import Plan, PlanStore, Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.reconcile import Observation, decide
from hls.pipeline.stage import Stage, most_roles, register
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.nproc import FLOOR_PER_EVALUATION, nproc_max_problem
from tests.drive.test_stage import ARTIFACT_SHAPE
from tests.drive.test_stage import SHAPE as DRIVE_SHAPE
from tests.drive.test_stage import _params as drive_params
from tests.pipeline.fake_scheduler import FakeScheduler
from tests.pipeline.plans import CODE, settings
from tests.pipeline.test_end_to_end_local import LocalSlurm, _drive
from tests.pipeline.test_orchestrator import Clock, _attach, _publish

HOLDER = Holder(host="login", pid=1, job_id=None, purpose="orchestrator")
#: A no-op stage with one GPU role.
GPU = ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=6, time_sec=600)
#: A no-op stage with one CPU role.
CPU = ResourceShape(cpus=2, memory_gib=6, time_sec=600)
#: What these tests' settings add to an allocation's stage times: the allocation overhead, with no
#: launch allowance unless a test sets one (``tests.pipeline.plans.settings``).
OVERHEAD_SEC = 360
#: Each call of a test grower: its name and the stages the plan held.
ASKED: list[tuple[str, tuple[str, ...]]] = []


declare_grower("test_twice", names=("grown_1", "grown_2"))


@register_grower("test_twice")
def _twice(plan: Plan, store) -> GrowthDecl | None:
    """Adds ``grown_1``, then ``grown_2``, each reading the plan's first stage and in an allocation
    of its own; then nothing more."""
    names = tuple(s.name for s in plan.stages)
    ASKED.append(("test_twice", names))
    grown = [n for n in names if n.startswith("grown_")]
    if len(grown) == 2:
        return None
    name = f"grown_{len(grown) + 1}"
    return GrowthDecl(decls=(StageDecl(name, "noop", GPU, {"gpu_roles": 1}, after=(names[0],)),),
                      groups=((name,),), reason=f"grown {len(grown) + 1} of 2")


declare_grower("test_refuse")


@register_grower("test_refuse")
def _refuse(plan: Plan, store) -> GrowthRefused:
    return GrowthRefused("exposure short after 24 servings")


declare_grower("test_granted", names=("grown_granted",))


@register_grower("test_granted")
def _granted(plan: Plan, store) -> GrowthDecl:
    return GrowthDecl(decls=(StageDecl("grown_granted", "grow_granted_probe", GPU, {}),),
                      groups=(("grown_granted",),), reason="a stage that declares grants")


declare_grower("test_other_kind", names=("grown_b200",))


@register_grower("test_other_kind")
def _other_kind(plan: Plan, store) -> GrowthDecl:
    shape = GPU.model_copy(update={"gpu_kind": "B200"})
    return GrowthDecl(decls=(StageDecl("grown_b200", "noop", shape, {"gpu_roles": 1}),),
                      groups=(("grown_b200",),), reason="another kind")


declare_grower("test_again", names=("first",))


@register_grower("test_again")
def _again(plan: Plan, store) -> GrowthDecl:
    return GrowthDecl(decls=(StageDecl("first", "noop", GPU, {"gpu_roles": 1}),), groups=(("first",),),
                      reason="a name the plan holds")


declare_grower("test_drive", names=("drive_1", "drive_2"))


@register_grower("test_drive")
def _drive_grower(plan: Plan, store) -> GrowthDecl:
    """Adds two drives of a scored server, side by side."""
    names = ("drive_1", "drive_2")
    return GrowthDecl(decls=tuple(StageDecl(n, "deploy_drive", DRIVE_SHAPE, drive_params()) for n in names),
                      groups=tuple((n,) for n in names), reason="two drives")


declare_grower("test_aggregate", names=("aggregate",))


@register_grower("test_aggregate")
def _aggregate(plan: Plan, store) -> GrowthDecl | None:
    """Adds ``aggregate``, as a calibration's grower does once its servings are scored; then nothing
    more."""
    if any(s.name == "aggregate" for s in plan.stages):
        return None
    return GrowthDecl(decls=(StageDecl("aggregate", "noop", GPU, {"gpu_roles": 1}),), groups=(("aggregate",),),
                      reason="every serving scored")


declare_grower("test_undeclared", names=("declared",), patterns=(r"declared_[1-9][0-9]*",))


@register_grower("test_undeclared")
def _undeclared(plan: Plan, store) -> GrowthDecl:
    return GrowthDecl(decls=(StageDecl("undeclared", "noop", GPU, {"gpu_roles": 1}),), groups=(("undeclared",),),
                      reason="a stage it does not declare")


class _NoParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


@register
class GrantedProbe(Stage):
    """A stage kind that declares its grants from its parameters, as agent_build does."""

    kind: ClassVar[str] = "grow_granted_probe"
    params_model: ClassVar[type[BaseModel]] = _NoParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        pass

    @classmethod
    def declared_grants(cls, params: BaseModel, settings) -> dict[str, str]:
        return {"HF_TOKEN": "hf_token"}

    @classmethod
    def max_roles(cls, params: BaseModel) -> int:
        """It starts no role."""
        return 0

    def run(self, ctx) -> dict[str, str]:
        return {}


def _plan(tmp_path: Path, grower: str | None, *, resources=None, patterns=(), site=None) -> Plan:
    return build_plan(decls=[StageDecl("first", "noop", GPU, {"gpu_roles": 1})], task=NOOP_TASK, pipeline="noop",
                      layout=InstanceLayout.rollout(NOOP_TASK, "r1"),
                      settings=settings(tmp_path, resources=resources, site=site), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-noop", known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=known_task_ids(), now=1000.0, grower=grower, known_patterns=patterns)


def _warmer(tmp_path: Path, *, grower: str | None, partition: str = "gpu", singleton: bool = False,
            site=None) -> Plan:
    """A plan of one CPU stage, as a cache warmer's is, on *partition*."""
    return build_plan(decls=[StageDecl("first", "noop", CPU, {"gpu_roles": 0, "cpu_role": True})], task=NOOP_TASK,
                      pipeline="noop", layout=InstanceLayout.rollout(NOOP_TASK, "w1"),
                      settings=settings(tmp_path, site=site), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-warmer", known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=(), now=1000.0, grower=grower, singleton=singleton, partition=partition)


def _orchestrator(tmp_path: Path, grower: str, sched: FakeScheduler, clock: Clock, **kw) -> Orchestrator:
    return Orchestrator.create(_plan(tmp_path, grower, **kw), sched, holder=HOLDER, clock=clock,
                               sleep=lambda s: None)


def _all_published(plan: Plan, growth) -> Observation:
    return Observation(now=2000.0, published=frozenset(s.name for s in plan.stages), jobs={}, growth=growth)


def test_a_plan_records_its_grower_and_an_undeclared_one_is_refused(tmp_path):
    assert _plan(tmp_path, "test_twice").grower == "test_twice"
    assert _plan(tmp_path, None).grower is None
    with pytest.raises(PlanError, match="'no_such_grower', which no imported module declares"):
        _plan(tmp_path, "no_such_grower")


REPO = Path(__file__).resolve().parents[2]


def _fresh(code: str) -> str:
    """The last line *code* prints in a fresh interpreter run from this checkout."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO)])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, env=env,
                         check=True, timeout=120).stdout
    return out.splitlines()[-1]


def test_the_planner_knows_every_growers_stages_without_importing_a_grower():
    # The stage registry the planner imports declares them: a calibration's by the stage module
    # that makes their names.
    code = ("import json, sys\nimport hls.pipeline.planner\nfrom hls.pipeline.grown import grown_stages\n"
            "growers = ('hls.pipeline.growth', 'hls.pipeline.growers', 'hls.pipeline.calibration',\n"
            "           'hls.pipeline.calibration_grower')\n"
            "print(json.dumps([grown_stages('calibration').holds('aggregate'),\n"
            "                  sorted(m for m in sys.modules if m in growers)]))")
    assert json.loads(_fresh(code)) == [True, []]


def test_every_declared_grower_is_registered_where_an_orchestrator_runs():
    # The planner refuses a grower no module declares, and an orchestrator finds every declared one
    # whichever command started it.
    code = ("import json\nimport hls.pipeline.orchestrator\nfrom hls.pipeline.grown import GROWN_STAGES\n"
            "from hls.pipeline.growth import GROWERS\n"
            "print(json.dumps([sorted(GROWN_STAGES), sorted(GROWERS)]))")
    declared, registered = json.loads(_fresh(code))
    assert declared == registered and "calibration" in declared


def _imported(module: str) -> set[str]:
    """Every module the source of *module* imports, at module level or in a function."""
    tree = ast.parse((REPO / "src" / Path(*module.split("."))).with_suffix(".py").read_text())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found |= {node.module, *(f"{node.module}.{a.name}" for a in node.names)}
        elif isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
    return found


def test_growth_plans_through_the_planner_which_imports_no_growth():
    # One direction only: growth plans what a grower adds through the planner, and the planner reads
    # what growers may add from a registry below both. Growth imports no grower.
    assert "hls.pipeline.planner" in _imported("hls.pipeline.growth")
    for module in ("hls.pipeline.planner", "hls.pipeline.grown"):
        assert not _imported(module) & {"hls.pipeline.growth", "hls.pipeline.growers"}, module
    assert not _imported("hls.pipeline.growth") & {"hls.pipeline.growers", "hls.calibrate"}


def test_growth_is_asked_only_when_every_stage_published(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _orchestrator(tmp_path, "test_twice", sched, clock)
    ASKED.clear()
    plan, _, _ = orch.step()
    assert orch.observe(plan).growth is None and ASKED == []
    _publish(plan, "first")
    found = orch.observe(plan).growth
    assert isinstance(found, PlanGrowth) and [s.name for s in found.stages] == ["grown_1"]
    assert [(a.index, a.stages, a.depends_on, a.waits_for) for a in found.allocations] == [
        (1, ["grown_1"], None, [])]
    assert ASKED == [("test_twice", ("first",))]


def test_a_cancelled_instance_does_not_ask_its_grower(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _orchestrator(tmp_path, "test_twice", sched, clock)
    plan, _, _ = orch.step()
    _publish(plan, "first")
    request_cancel(orch.store, orch.layout, at=clock.t)
    ASKED.clear()
    assert orch.observe(plan).growth is None and ASKED == []
    plan, _, _ = orch.step()
    assert plan.outcome.status == "cancelled" and [s.name for s in plan.stages] == ["first"]


def test_an_ended_instance_does_not_ask_its_grower(tmp_path, monkeypatch):
    # The refused growth voids the instance while its job still runs; the passes that cancel the
    # job, here those of an orchestrator that does not register the grower, never ask it again.
    sched, clock = FakeScheduler(), Clock()
    orch = _orchestrator(tmp_path, "test_refuse", sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    _publish(plan, "first")
    sched.down = True
    plan, done, _ = orch.step()
    assert plan.outcome.status == "void" and not done
    sched.down = False
    monkeypatch.delitem(GROWERS, "test_refuse")
    clock.t += 5
    second = _attach(orch, sched, clock)
    plan, done, _ = second.step()
    assert plan.outcome.status == "void" and done and sched.jobs["100"].state.terminal


def test_growth_is_written_before_its_allocations_are_submitted(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _orchestrator(tmp_path, "test_twice", sched, clock)
    plan, _, _ = orch.step()
    _publish(plan, "first")
    seen: list[Plan] = []
    sched.after_create = lambda job: seen.append(PlanStore(orch.store, orch.layout).latest())
    plan, done, acted = orch.step()
    assert acted and not done
    [at_submit] = seen
    assert [s.name for s in at_submit.stages] == ["first", "grown_1"]
    assert at_submit.allocation(1).submissions[0].status == "intended"
    assert plan.allocation(1).current().job_id == "101" and plan.outcome is None
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    [grown] = [e for e in events if e["kind"] == "plan_grown"]
    assert (grown["stages"], grown["allocations"], grown["reason"]) == (["grown_1"], [1], "grown 1 of 2")


def test_a_resumed_orchestrator_grows_the_same(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    first = _orchestrator(tmp_path, "test_twice", sched, clock)
    plan, _, _ = first.step()
    _publish(plan, "first")
    mine = first.observe(plan).growth
    # The first orchestrator dies before it writes what it saw; another takes the instance over.
    clock.t += 5
    second = _attach(first, sched, clock)
    theirs = second.observe(PlanStore(second.store, second.layout).latest()).growth
    assert isinstance(mine, PlanGrowth) and theirs == mine
    plan, _, _ = second.step()
    assert [s.name for s in plan.stages] == ["first", "grown_1"] and len(plan.allocations) == 2
    assert len(sched.submits()) == 2


def test_a_refused_growth_voids_the_instance(tmp_path):
    plan = _plan(tmp_path, "test_refuse")
    refused = grow(plan, PosixStore(plan.store_root))
    assert refused == GrowthRefused("exposure short after 24 servings")
    d = decide(plan, _all_published(plan, refused))
    assert d.plan.outcome.status == "void"
    assert d.plan.outcome.reason == "growth refused: exposure short after 24 servings"
    assert d.effects == []


def test_a_plan_whose_grower_adds_nothing_more_succeeds(tmp_path):
    plan = _plan(tmp_path, "test_twice")
    d = decide(plan, _all_published(plan, GrowthComplete()))
    assert d.plan.outcome.status == "succeeded"


def test_a_growing_plan_is_never_decided_succeeded_without_asking_its_grower(tmp_path):
    plan = _plan(tmp_path, "test_twice")
    with pytest.raises(ValueError, match="without asking its grower"):
        decide(plan, _all_published(plan, None))
    # A plan that names no grower ends as it always did.
    plain = _plan(tmp_path, None)
    assert decide(plain, _all_published(plain, None)).plan.outcome.status == "succeeded"


@pytest.mark.parametrize("grower, reason", [
    ("test_granted", "its stages cannot be planned: grown stages are granted no secret; ['grown_granted'] declare "
                     "grants"),
    ("test_other_kind", "its stages cannot be planned: grown stages use another gpu kind: ['B200', 'H200']; one run "
                        "uses one kind"),
    ("test_again", "its stages cannot be planned: the grower added stages the plan already has: ['first']"),
], ids=["grants", "gpu_kind", "name"])
def test_a_grown_stage_that_cannot_be_planned_is_refused(tmp_path, grower, reason):
    plan = _plan(tmp_path, grower)
    assert grow(plan, PosixStore(plan.store_root)) == GrowthRefused(reason)


def test_a_grown_stage_that_declares_grants_is_refused(tmp_path):
    # A grown stage is planned on the login node from the plan alone: the secrets file is read
    # only when a plan is made, so no grant can be resolved for it.
    plan = _plan(tmp_path, "test_granted")
    refused = grow(plan, PosixStore(plan.store_root))
    assert isinstance(refused, GrowthRefused) and "granted no secret" in refused.reason


def test_a_grown_stage_takes_its_resources_table_from_the_plans_settings(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _orchestrator(tmp_path, "test_twice", sched, clock, resources={"noop": {"grown_1": {"time_sec": 900}}})
    plan, _, _ = orch.step()
    _publish(plan, "first")
    # The plan read back from the store holds the table as JSON, every unset field null.
    found = grow(PlanStore(orch.store, orch.layout).latest(), orch.store)
    [stage] = found.stages
    assert stage.shape == GPU.model_copy(update={"time_sec": 900}) and stage.shape_overridden
    assert found.allocations[0].request.time_sec == 900 + OVERHEAD_SEC


def test_a_grown_allocation_is_sized_as_a_planned_one(tmp_path):
    """A grown allocation asks for its stages' time with the overhead and its roles' launch
    headroom, is planned under the QOS that holds that time, and asks for a scored server's CPUs
    as two socket tasks, as an allocation of a new plan does."""
    site = {"evaluation_socket_tasks": True, "qos_by_hours": {"short": 0.5, "long": 168},
            "step_launch_allowance_sec": 150}
    plan = _plan(tmp_path, "test_drive", site=site)
    found = grow(plan, PosixStore(plan.store_root))
    headroom = 150 * most_roles("deploy_drive", found.stages[0].params)
    assert [a.index for a in found.allocations] == [1, 2]
    for alloc in found.allocations:
        assert alloc.request.time_sec == DRIVE_SHAPE.time_sec + OVERHEAD_SEC + headroom > 1800
        assert (alloc.placement.qos, alloc.placement.socket_tasks, alloc.request.cpus) == ("long", ARTIFACT_SHAPE.cpus,
                                                                       2 * ARTIFACT_SHAPE.cpus)
    # The plan's own allocation, a no-op stage's, fits the short limit.
    assert plan.allocation(0).placement.qos == "short" and plan.allocation(0).placement.socket_tasks is None


def test_a_grown_allocation_is_refused_an_overhead_below_the_runners_end_margin(tmp_path):
    # A plan's settings passed this when it was made; a checkout whose margin grew since refuses it.
    plan = _plan(tmp_path, "test_twice")
    short = plan.settings.model_copy(update={"run": plan.settings.run.model_copy(
        update={"allocation_overhead_sec": 299})})
    refused = grow(plan.model_copy(update={"settings": short}), PosixStore(plan.store_root))
    assert isinstance(refused, GrowthRefused)
    assert "allocation_overhead_sec is 299s, less than the 300s" in refused.reason


def test_a_grown_contestant_stage_is_refused_a_process_cap_no_node_can_hold(tmp_path):
    # Grown stages are planned as a plan's first ones are: a contestant's step under a site cap
    # under one evaluation's floor is refused before it is submitted.
    plan = _plan(tmp_path, "test_drive")
    low = plan.settings.model_copy(update={"site": plan.settings.site.model_copy(
        update={"nproc_max": FLOOR_PER_EVALUATION - 1})})
    refused = grow(plan.model_copy(update={"settings": low}), PosixStore(plan.store_root))
    assert isinstance(refused, GrowthRefused)
    assert nproc_max_problem(FLOOR_PER_EVALUATION - 1, 1) in refused.reason
    enough = plan.settings.model_copy(update={"site": plan.settings.site.model_copy(
        update={"nproc_max": FLOOR_PER_EVALUATION})})
    assert isinstance(grow(plan.model_copy(update={"settings": enough}), PosixStore(plan.store_root)), PlanGrowth)


def test_a_warmer_that_runs_one_at_a_time_may_not_grow(tmp_path):
    # What it grew would not be held behind the site's other warmers.
    assert _warmer(tmp_path, grower=None, singleton=True).allocations[0].singleton
    with pytest.raises(PlanError, match="runs one at a time with the site's other warmers and may not grow by "
                                        "'test_twice'"):
        _warmer(tmp_path, grower="test_twice", singleton=True)


def test_a_cpu_plan_may_not_grow(tmp_path):
    # A grown stage may ask for GPUs, which no allocation on the CPU partition holds.
    site = {"cpu_partition": "cpu-a"}
    assert _warmer(tmp_path, grower=None, partition="cpu", site=site).allocations[0].placement.partition == "cpu-a"
    with pytest.raises(PlanError, match="runs on the CPU partition and may not grow by 'test_twice'"):
        _warmer(tmp_path, grower="test_twice", partition="cpu", site=site)


def test_a_growth_that_is_no_valid_plan_voids_the_instance(tmp_path):
    # A plan another checkout made: a CPU warmer that grows a GPU stage. The grown plan is
    # checked whole, so the instance ends void at once instead of failing every write.
    sched, clock = FakeScheduler(), Clock()
    made = _warmer(tmp_path, grower=None, partition="cpu", site={"cpu_partition": "cpu-a"})
    orch = Orchestrator.create(made.model_copy(update={"grower": "test_twice"}), sched, holder=HOLDER, clock=clock,
                               sleep=lambda s: None)
    plan, _, _ = orch.step()
    _publish(plan, "first")
    refused = grow(plan, orch.store)
    assert isinstance(refused, GrowthRefused) and refused.reason.startswith(
        "its stages cannot be planned: the grown plan is no valid plan: ")
    assert "allocation 1 goes to partition cpu-a and asks for 1 GPUs" in refused.reason
    plan, done, _ = orch.step()
    assert plan.outcome.status == "void" and plan.outcome.reason == f"growth refused: {refused.reason}"
    assert done and PlanStore(orch.store, orch.layout).latest().outcome.status == "void"
    assert [s.name for s in plan.stages] == ["first"] and len(sched.submits()) == 1


def test_a_grown_stage_its_grower_does_not_declare_is_refused(tmp_path):
    # Settings tables are checked against what growers declare, so a grower adds nothing else.
    plan = _plan(tmp_path, "test_undeclared")
    assert grow(plan, PosixStore(plan.store_root)) == GrowthRefused(
        "its stages cannot be planned: the grower added stages it does not declare: ['undeclared']")


def test_the_registry_lists_what_each_grower_adds():
    assert grown_stages("test_twice") == GrownStages(names=frozenset({"grown_1", "grown_2"}))
    assert GROWN_STAGES["test_twice"] is grown_stages("test_twice")
    every = grown_stages()
    assert {"grown_1", "aggregate", "first"} <= every.names and every.holds("declared_7")
    assert not every.holds("declared_0") and not every.holds("verfy")
    with pytest.raises(ValueError, match="test_twice is declared twice"):
        declare_grower("test_twice", names=("grown_3",))
    with pytest.raises(ValueError, match="test_nameless is not declared; declare_grower comes first"):
        register_grower("test_nameless")(lambda plan, store: None)


def test_a_table_for_a_stage_the_plans_grower_adds_is_kept_for_it(tmp_path):
    plan = _plan(tmp_path, "test_aggregate", resources={"noop": {"aggregate": {"time_sec": 900}}})
    assert not any("aggregate" in w for w in plan.warnings)
    [stage] = grow(plan, PosixStore(plan.store_root)).stages
    assert stage.name == "aggregate" and stage.shape.time_sec == 900 and stage.shape_overridden


def test_a_table_for_a_stage_only_another_plans_grower_adds_warns_and_is_ignored(tmp_path):
    plan = _plan(tmp_path, None, resources={"noop": {"aggregate": {"time_sec": 900}}})
    assert [w for w in plan.warnings if "aggregate" in w] == [
        "[resources.noop.aggregate] names stage aggregate, which this plan does not run: the table is ignored"]
    # Another task's table for it is checked and accepted without a word.
    plan = _plan(tmp_path, None, resources={"multitenant_slo": {"aggregate": {"time_sec": 900}}})
    assert not any("aggregate" in w for w in plan.warnings)
    with pytest.raises(PlanError, match=r"resources\.noop\.agregate"):
        _plan(tmp_path, None, resources={"noop": {"agregate": {"time_sec": 900}}})


def test_an_unregistered_grower_fails_the_pass_not_the_instance(tmp_path, monkeypatch):
    sched, clock = FakeScheduler(), Clock()
    orch = _orchestrator(tmp_path, "test_twice", sched, clock)
    plan, _, _ = orch.step()
    _publish(plan, "first")
    monkeypatch.delitem(GROWERS, "test_twice")
    with pytest.raises(UnknownGrower, match="'test_twice', which no imported module registers"):
        orch.step()
    assert PlanStore(orch.store, orch.layout).latest().outcome is None


def test_a_grower_registered_twice_under_one_name_is_refused():
    with pytest.raises(ValueError, match="two growers named test_twice"):
        register_grower("test_twice")(lambda plan, store: None)


def test_a_grown_plan_runs_its_new_stages_end_to_end(tmp_path):
    slurm = LocalSlurm()
    orch = Orchestrator.create(_plan(tmp_path, "test_twice"), slurm, holder=HOLDER, sleep=lambda s: None)
    plan = _drive(orch, slurm)
    assert plan.outcome.status == "succeeded"
    assert [s.name for s in plan.stages] == ["first", "grown_1", "grown_2"]
    assert [a.stages for a in plan.allocations] == [["first"], ["grown_1"], ["grown_2"]]
    assert all(a.depends_on is None and a.waits_for == [] for a in plan.allocations)
    store = PosixStore(plan.store_root)
    assert all(read_manifest(store, plan.layout, name) is not None for name in ("first", "grown_1", "grown_2"))
