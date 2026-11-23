"""Plans that grow after they were submitted (a calibration appends allocations while its
exposure falls short, the one way a plan grows after submission).

A plan may name a grower (``Plan.grower``): a function that the module declaring the pipeline
registers here under that name. Once every stage of the plan has published, the orchestrator
asks it, on the login node, what to add: stages and the new allocations that run them, a
refusal, or nothing more. A grower reads only the plan and small files that harness roles
published, each checked against its manifest, so it moves no heavy data, and its
answer depends on those alone: an orchestrator that takes the instance over asks again and gets
the same answer.

``grow`` turns what a grower declares into the plan's stages and allocations through the
planner, as ``build_plan`` does for a new plan: each stage is checked by its kind and takes the
site's ``[resources]`` table and its retry policy from the plan's own settings; the allocations
are sized under the site's rules (``layout_allocations``: the overhead, the launch headroom, the
QOS, the socket tasks) and numbered after the plan's, and each waits for nothing, since
everything planned before it has published. The orchestrator appends them in the revision that
also holds their intended submissions, so nothing they imply happens before the plan says so
(intent before action).

A grown stage is granted no secret: the secrets file is read only when a plan is made, and a
stage kind that declares its grants is refused. A grower whose stages cannot be planned, or
whose plan with them appended is no valid plan, refuses, and the instance ends void with the
reason. Only a run's plan on the GPU partition grows: the planner refuses a grower on a warmer
that runs one at a time or on the CPU partition.

Every grower is declared (``hls.pipeline.grown.declare_grower``) with the stage names, and the
patterns of the numbered names, it may add, by the module that makes those names, apart from the
function that decides what it adds, which is registered here once it is declared
(``register_grower``). The planner checks a plan's grower and every plan's ``[resources]`` and
``[run.retry]`` tables against the declarations without importing this module or a grower, and
``grow`` refuses a stage its grower did not declare. ``hls.pipeline.growers`` imports every
grower, and the orchestrator, which asks them, imports it, so a plan's grower is found whichever
command started the instance."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import ValidationError

from hls.pipeline.grown import GROWN_STAGES, grown_stages
from hls.pipeline.plan import Allocation, Plan, StagePlan
from hls.pipeline.planner import AllocationDecl, PlanError, StageDecl, layout_allocations, stage_plans
from hls.pipeline.stage import stage_class
from hls.store.base import Store


@dataclass(frozen=True)
class GrowthDecl:
    """What a grower adds."""

    #: The new stages, in order.
    decls: tuple[StageDecl, ...]
    #: Each new allocation's stages, in order. A new allocation waits for nothing.
    groups: tuple[tuple[str, ...], ...]
    #: Why, for the orchestrator's event and for people.
    reason: str


@dataclass(frozen=True)
class GrowthRefused:
    """A grower's refusal: the instance cannot go on, and ends void."""

    #: Why.
    reason: str


@dataclass(frozen=True)
class GrowthComplete:
    """The grower adds nothing more: the instance may end."""


@dataclass(frozen=True)
class PlanGrowth:
    """What a grower adds, as the plan will hold it."""

    #: The new stages.
    stages: tuple[StagePlan, ...]
    #: The new allocations, numbered after the plan's.
    allocations: tuple[Allocation, ...]
    #: Why.
    reason: str


#: A grower: from the plan and the store, which it only reads, to the stages it adds, a
#: refusal, or None when it adds nothing more.
Grower = Callable[[Plan, Store], GrowthDecl | GrowthRefused | None]

#: Every registered grower, by name.
GROWERS: dict[str, Grower] = {}


class UnknownGrower(ValueError):
    """A plan names a grower that no imported module registers."""


def register_grower(name: str) -> Callable[[Grower], Grower]:
    """Decorator that registers a grower under *name*, which must be declared already
    (``hls.pipeline.grown.declare_grower``). A second grower under a name already taken is
    refused."""

    def add(grower: Grower) -> Grower:
        if name not in GROWN_STAGES:
            raise ValueError(f"grower {name} is not declared; declare_grower comes first, so that every plan "
                             "knows what it adds without importing it")
        if name in GROWERS and GROWERS[name] is not grower:
            raise ValueError(f"two growers named {name}")
        GROWERS[name] = grower
        return grower

    return add


def grow(plan: Plan, store: Store) -> PlanGrowth | GrowthRefused | GrowthComplete:
    """What *plan*'s grower adds now, as the plan will hold it: its stages and allocations, its
    refusal (or the reason what it added cannot be planned), or that it adds nothing more. Ask
    only once every stage of *plan* has published. Raise ValueError for a plan that names no
    grower, and UnknownGrower for one whose grower no imported module registers."""
    if plan.grower is None:
        raise ValueError(f"{plan.instance} names no grower")
    grower = GROWERS.get(plan.grower)
    if grower is None:
        raise UnknownGrower(f"{plan.instance} grows by {plan.grower!r}, which no imported module registers; "
                            f"registered: {sorted(GROWERS)}")
    asked = grower(plan, store)
    if asked is None:
        return GrowthComplete()
    if isinstance(asked, GrowthRefused):
        return asked
    try:
        return _planned(plan, asked)
    except PlanError as e:
        return GrowthRefused(f"its stages cannot be planned: {e}")


def _declares_grants(kind: str) -> bool:
    """Whether stage kind *kind* declares its grants; an unknown kind is left to the planner,
    which refuses it."""
    try:
        cls = stage_class(kind)
    except ValueError:
        return False
    return cls.declared_grants is not None


def _planned(plan: Plan, decl: GrowthDecl) -> PlanGrowth:
    """*decl* planned onto *plan*. Raise PlanError for anything that cannot be planned."""
    if not decl.decls:
        raise PlanError("the grower added no stage")
    existing = {s.name for s in plan.stages}
    clash = sorted(d.name for d in decl.decls if d.name in existing)
    if clash:
        raise PlanError(f"the grower added stages the plan already has: {clash}")
    declared = grown_stages(plan.grower)
    undeclared = sorted(d.name for d in decl.decls if not declared.holds(d.name))
    if undeclared:
        raise PlanError(f"the grower added stages it does not declare: {undeclared}")
    granted = sorted(d.name for d in decl.decls if _declares_grants(d.kind))
    if granted:
        raise PlanError(f"grown stages are granted no secret; {granted} declare grants")
    stages = stage_plans(decl.decls, plan.settings, plan.task, None, earlier=existing)
    kinds = {s.shape.gpu_kind for s in (*plan.stages, *stages) if s.shape.gpu_kind is not None}
    if len(kinds) > 1:
        raise PlanError(f"grown stages use another gpu kind: {sorted(kinds)}; one run uses one kind")
    # Every allocation of a plan goes to one partition: a run's to the GPU partition (None).
    partition = plan.allocations[0].placement.partition if plan.allocations else None
    allocations = layout_allocations(stages, [AllocationDecl(stages=g) for g in decl.groups], plan.settings,
                                     gpu_kind=next(iter(kinds), None), partition=partition, job_name=plan.job_name,
                                     first_index=len(plan.allocations))
    # The orchestrator writes the plan with these appended; one it could not write would fail
    # every pass, on every orchestrator, instead of ending the instance.
    grown = plan.model_copy(update={"stages": [*plan.stages, *stages], "allocations": [*plan.allocations, *allocations]})
    try:
        Plan.model_validate(grown.model_dump())
    except ValidationError as e:
        problems = "; ".join(err["msg"].removeprefix("Value error, ") for err in e.errors())
        raise PlanError(f"the grown plan is no valid plan: {problems}") from e
    return PlanGrowth(stages=tuple(stages), allocations=tuple(allocations), reason=decl.reason)
