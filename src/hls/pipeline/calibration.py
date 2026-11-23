"""``hls calibrate``: the calibration pipeline, whose plan grows by the calibration grower
(``hls.pipeline.calibration_grower``).

A calibration instance, ``calibrations/<task>/<id>``, measures the constant a task's calibrated
gate needs and writes the ``calibration.json`` into the store's measured area, where runs of the
task find it (``hls.measured``). Its first plan holds:
- allocation 0: ``provision`` (the task's weights, rows and tool bundles), ``images`` (the
  reference image and each pool member's image the task module declares, each with what it
  builds on), ``instrument`` (the reference's instrument check, which owns the reference's engine
  cache) and a warm stage for each pool member (``warm_<engine>``), which owns the member's
  engine cache: it builds it cold and publishes it, or finds it published and does nothing, so
  every capture after it is seeded from it (``hls.stages.calibration``);
- allocations 1 to K, each once allocation 0 has succeeded: the first round of servings, those
  each engine owes first (eight for the token-level gate, ``servings_per_engine`` for the sampling
  gate), the engines interleaved and the servings dealt round robin over K = ``min(fanout,
  servings)`` allocations, each serving a capture stage (``capture_<engine>_<draw>``) and each
  allocation's captures scored by one score stage (``score_<j>``) after them, as the grower lays
  out every round (``hls.pipeline.calibration_grower``).

Once every planned stage has published, the orchestrator asks the grower, which adds the servings
the exposure counter still owes in further rounds, then the ``aggregate`` that assembles the
calibration. Whichever command drives a calibration to its end, ``hls calibrate`` or ``hls resume``,
says what it measured (``hls.pipeline.commands``).

Every stage that copies the task's data (the instrument's and each warm stage's the reference
role's, each capture and score stage's the calibration role's; ``hls.provision.mounts``) declares
its copies and holds them in its disk (``data_copies``): the first plan sizes them by the
provision record the cache holds (``commands.planned_provision``), which a cold store has none
of, so it refuses the plan naming the ``hls provision`` to run first; the grower sizes a round's
by the record the instance's provision stage published, which those stages copy from. Each such
stage's time holds its staging, the fixed allowance for those copies and the one image it
extracts (``hls.pipeline.pipelines.STAGING_SEC``).

The login node imports the task module, which declares the images, reads the settings, the
``ENTRY.json`` of each image's cache entry, whose presence sizes the
images stage's disk (``commands.cache_holds``), the cache's provision record of the task, and the
secrets file's grant names; it serves and reads nothing else. A
calibration is planned only from a committed checkout (``check_committed_tree``): runs verify
against it, so it must come from the commit its plan records. A task that answers
with audio is refused: no role can drive its members.

``hls capped-requests`` reads a calibration instance's published captures (``capped_requests``)."""

from __future__ import annotations

import math
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hls.calibrate import (DEFAULT_DELTA, DEFAULT_EPS, DEFAULT_MU_PER_MTOK, MIN_SERVINGS, POOL_ENGINE_NAMES, CappedRows, CaptureStore, RowIndex, capped_rows,
                           check_campaign, engine_cache_config, load_sessions, member_engines,
                           member_internet, pinned_reference)
from hls.artifact.artifact import EvaluationAllocation
from hls.pipeline import commands
from hls.pipeline.calibration_grower import member_reserve_gib, provision_record, round_servings, serving_stages
from hls.pipeline.pipelines import (DRIVE_MARGIN_SEC, EVALUATE_IMAGES_CPUS, EVALUATE_PROVISION_SEC,
                                    KNOWN_STAGE_NAMES, READINESS_TIMEOUT_SEC, REFERENCE_SHAPE,
                                    REQUEST_TIMEOUT_SEC, STAGE_NAME_PATTERNS, STAGING_SEC, drive_wall,
                                    images_chain, images_stage, provision_shape, provision_stages,
                                    reference_shape, runner_stores)
from hls.pipeline.plan import Plan, PlanStore
from hls.pipeline.planner import AllocationDecl, AllocationPastQOS, PlanError, StageDecl, build_plan
from hls.pipeline.preflight import check_committed_tree
from hls.provision.model import Provisioned
from hls.provision.mounts import copied_entries, copies_gib
from hls.recipes import catalog
from hls.spec.datasets import Split
from hls.spec.resources import ResourceShape
from hls.spec.taskspec import TaskSpec
from hls.stages.calibration import GROWER, INSTRUMENT, CaptureParams, InstrumentParams, stage_shape, warm_stage
from hls.stages.engine_cache import COLD_LOAD_SEC, ENGINE_CACHE_GIB
from hls.stages.images import KEPT_DISK_GIB, HeldKeys
from hls.store.cache import ContentCache, entry_sha256
from hls.store.digest import sha256_file
from hls.store.layout import InstanceLayout, new_rollout_id
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.factory import RecipeKey, seal_memory_gib, substrate
from hls.taskref import TaskRef, load_task
from hls.util import validation_errors
from hls.verify.calibration import DEFAULT_EOT_EXPECTED_MAX, CalibrationError, exposure_target, workload_fingerprint
from hls.verify.calibration_roles import CampaignSettings

#: The image catalog's measured role of each pool member's image.
MEMBER_ROLES = {engine: f"member-{engine}" for engine in POOL_ENGINE_NAMES}
#: The pool members a calibration serves unless ``--engines`` says otherwise.
DEFAULT_ENGINES = ("vllm", "sglang")
#: Allocations that take servings side by side unless ``--fanout`` says otherwise. With four, a
#: token-level top-up past the floor of servings may overshoot its exposure target by up to three
#: servings per engine.
DEFAULT_FANOUT = 4
#: The most servings an engine may take unless ``--max-servings-per-engine`` says otherwise:
#: three times the eight servings an engine of a token-level calibration takes as its floor.
DEFAULT_MAX_SERVINGS_PER_ENGINE = 24
#: Seconds for the preflight's probes after the engines load: a generous bound.
PREFLIGHT_SEC = 600
#: Seconds a score stage allows for teacher-forcing one serving's capture of a task whose
#: evaluation allocation declares no verify time (``score_per_serving_sec``): half as much again
#: as the 5,208 s a serving of about 1.7M tokens takes, its share of the stage's copies and of the
#: reference's load included.
SCORE_PER_SERVING_SEC = math.ceil(1.5 * 10_416 / 2)


def member_shape(task: TaskSpec, engines: Sequence[str], *, runner_memory_gib: int) -> ResourceShape:
    """A capture role's shape for *task*'s pool members *engines*: the GPUs of the task's kind
    the widest member asks for, the reference role's CPUs, and the larger of the reference role's
    memory, the reference job's less the runner's *runner_memory_gib*, and what a
    member asks for (an agentic task's tool bundle). Engines are built, never loaded."""
    builders = member_engines(task)
    requests = [builders[e]().resources for e in engines]
    memory = max([REFERENCE_SHAPE.memory_gib - runner_memory_gib,
                  *(math.ceil(r.memory_mb / 1024) for r in requests if r.memory_mb)])
    return ResourceShape(gpus=max(r.gpu_count for r in requests), gpu_kind=task.hardware.accelerator.kind,
                         cpus=REFERENCE_SHAPE.cpus, memory_gib=memory, time_sec=REFERENCE_SHAPE.time_sec)


def instrument_sec(campaign: CampaignSettings) -> int:
    """The instrument stage's time: its staging, its copies' and its extraction's
    (``STAGING_SEC``), the bound of each of its two engines' loads
    (``CampaignSettings.load_timeout_sec``) and ``PREFLIGHT_SEC``."""
    return STAGING_SEC + 2 * math.ceil(campaign.load_timeout_sec) + PREFLIGHT_SEC


def warm_sec() -> int:
    """A warm stage's time: its staging, its copies' and its extraction's (``STAGING_SEC``),
    ``COLD_LOAD_SEC`` for the member's cold load and the margin after it, in which the stage
    publishes what the load built. The stage gives the load what its time leaves after the other
    two (``warm_load_sec``), so a ``[resources]`` table that gives it more time gives the load as
    much more."""
    return STAGING_SEC + COLD_LOAD_SEC + DRIVE_MARGIN_SEC


def score_per_serving_sec(allocation: EvaluationAllocation | None) -> int:
    """Seconds a score stage gives each serving of a task whose evaluation allocation is
    *allocation*: the verify time it declares, in which one drive's replies are verified, or
    ``SCORE_PER_SERVING_SEC`` when it declares none or the task has none."""
    declared = allocation.verify_time_sec if allocation is not None else None
    return declared if declared is not None else SCORE_PER_SERVING_SEC


def calibration_stages(*, task_ref: TaskRef, task: Any, catalog_key: str, campaign: CampaignSettings,
                       runner_memory_gib: int, seal_memory_gib: int, held: HeldKeys, recipe_key: RecipeKey,
                       provisioned: Provisioned) -> tuple[list[StageDecl], list[AllocationDecl]]:
    """A calibration's first stages and their allocations, as the module docstring lays them
    out, for *campaign* of *task*, the ``TASK`` of the module *task_ref* names, which the image
    catalog knows as *catalog_key*. Every stage leaves *runner_memory_gib*
    (``run.runner_memory_gib``) to the runner, and a stage whose member is sealed
    *seal_memory_gib* to its seal as well; the provision stage is sized for *task*
    (``provision_shape``). The images are the campaign's, which the task module declares, each
    with what it builds on; the images stage's disk counts the entries *held* says the cache lacks (``pipelines.images_stage``). Each stage
    that copies the task's data declares its copies and holds them in its disk, by what
    *provisioned*, the provision record the cache holds, states of its role's entries
    (``copied_entries``): the reference role's for the instrument and the warm stages, the
    calibration role's for the servings'. Each stage's time holds its staging (``STAGING_SEC``)."""
    images = catalog.task_images(catalog_key)
    chain = images_chain(images, [*campaign.member_images.values(), campaign.reference_image])
    reference = copied_entries(("reference",), provisioned)
    instrument = stage_shape(campaign.instrument_shape(), runner_memory_gib=runner_memory_gib,
                             disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + copies_gib(reference),
                             time_sec=instrument_sec(campaign))
    ref, settings = task_ref.model_dump(mode="json"), campaign.model_dump(mode="json")
    first = [
        *provision_stages(task_ref=task_ref, shape=provision_shape(task, runner_memory_gib=runner_memory_gib,
                                                                   time_sec=EVALUATE_PROVISION_SEC),
                          sandbox_stores=runner_stores(task)),
        images_stage(catalog_task=catalog_key, decls=chain, held=held, recipe_key=recipe_key,
                     cpus=EVALUATE_IMAGES_CPUS),
        StageDecl(name=INSTRUMENT, kind="calibrate_instrument", shape=instrument, after=("provision", "images"),
                  params={"task": ref, "campaign": settings, "data_copies": reference}),
        *(StageDecl(name=warm_stage(e), kind="calibrate_warm",
                    shape=stage_shape(campaign.member_shape,
                                      runner_memory_gib=member_reserve_gib(campaign, runner_memory_gib, seal_memory_gib),
                                      disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + copies_gib(reference),
                                      time_sec=warm_sec()),
                    after=("provision", "images"),
                    params={"task": ref, "campaign": settings, "engine": e, "staging_sec": STAGING_SEC,
                            "margin_sec": DRIVE_MARGIN_SEC, "data_copies": reference}) for e in campaign.engines),
    ]
    servings = round_servings(campaign, {}, {e: 0 for e in campaign.engines})
    decls, groups = serving_stages(task_ref=task_ref, campaign=campaign, servings=servings, first_share=1,
                                   runner_memory_gib=runner_memory_gib, seal_memory_gib=seal_memory_gib,
                                   data_copies=copied_entries(("calibration",), provisioned))
    return [*first, *decls], [AllocationDecl(stages=tuple(d.name for d in first)),
                              *(AllocationDecl(stages=group, depends_on=0) for group in groups)]


def _measured(catalog_key: str, role: str, images: catalog.TaskImages) -> str:
    """The name of *catalog_key*'s image of *role*. Raise PlanError when its module declares
    none."""
    if role not in images.measured:
        raise PlanError(f"{catalog_key} declares no {role} image (IMAGES in its module); a calibration runs on the "
                        "reference and member images its task declares")
    return images.measured[role]


def plan_calibrate(*, config: Path, overrides: Mapping[str, str], task_module: str,
                   engines: Sequence[str] = DEFAULT_ENGINES, fanout: int = DEFAULT_FANOUT,
                   servings_per_engine: int | None = None, sessions_per_draw: int | None = None,
                   eps: float = DEFAULT_EPS, mu_per_mtok: float = DEFAULT_MU_PER_MTOK, delta: float = DEFAULT_DELTA,
                   eot_expected_max: float = DEFAULT_EOT_EXPECTED_MAX,
                   max_servings_per_engine: int = DEFAULT_MAX_SERVINGS_PER_ENGINE, job_name: str,
                   echo: commands.Echo = print) -> Plan:
    """The first plan of a calibration instance of *task_module*'s task, on the site's GPU
    partition, as the module docstring lays it out: the pool members *engines*, *fanout*
    allocations of servings side by side, the tolerances *eps*, *mu_per_mtok*, *delta* and
    *eot_expected_max*, *sessions_per_draw* sessions a serving, *servings_per_engine* servings
    for the sampling gate and at most *max_servings_per_engine* of any engine. Overrides under
    ``site``, ``run`` and ``resources`` reach the settings and ``task.*`` the task reference
    (``commands.split_overrides``). Once planned, one line says what the campaign takes. Raise
    ValueError (a PlanError among them) for what cannot be planned: a checkout with uncommitted
    changes, a campaign the task cannot take (``check_campaign``), a member image the task does
    not declare, settings out of range, and an
    allocation no QOS of the site holds, the refusal naming the smallest --fanout that fits
    (``_fanout_that_fits``), and a cache that lacks the task's data, whose provision record sizes
    the copying stages' disk (``commands.planned_provision``, which names the ``hls provision`` to
    run first)."""
    settings_part, task_part = commands.split_overrides(overrides)
    settings = commands.planning_settings(config, settings_part)
    code = commands.code_state(commands.checkout_root())
    check_committed_tree(code)
    catalog_key = catalog.catalog_task(task_module)
    task_id = commands.task_id_of(task_module)
    task_ref = TaskRef(module=task_module, overrides=task_part)
    task = commands.task_to_plan(task_ref)
    allocation = commands.evaluation_allocation(commands.produced_artifact(task_module))
    try:
        kind, servings = check_campaign(task, engines=engines, servings_per_engine=servings_per_engine)
    except CalibrationError as e:
        raise PlanError(f"{task_id} cannot be calibrated: {' '.join(str(e).split())}") from None
    images = catalog.task_images(catalog_key)
    member_images = {e: _measured(catalog_key, MEMBER_ROLES[e], images) for e in engines}
    # The image a grading's verify stage runs the reference engine on: the task's reference image,
    # else its harness.
    reference = images.reference
    held = commands.cache_holds(settings)
    try:
        campaign = CampaignSettings(
            kind=kind, workload_fingerprint=workload_fingerprint(task), task_seed=task.seed, engines=list(engines),
            member_internet=member_internet(task),
            eps=eps, mu_per_mtok=mu_per_mtok, delta=delta, eot_expected_max=eot_expected_max,
            sessions_per_draw=sessions_per_draw, servings_per_engine=servings,
            max_servings_per_engine=max_servings_per_engine, fanout=fanout, member_images=member_images,
            reference_image=reference,
            member_engine_configs={e: engine_cache_config(member_engines(task)[e]()) for e in engines},
            reference_engine_config=engine_cache_config(pinned_reference(task)),
            member_shape=member_shape(task, engines, runner_memory_gib=settings.run.runner_memory_gib + (
                0 if member_internet(task) else seal_memory_gib(settings.site))),
            reference_shape=reference_shape(task, allocation, runner_memory_gib=settings.run.runner_memory_gib),
            request_timeout_sec=REQUEST_TIMEOUT_SEC, drive_timeout_sec=drive_wall(task),
            load_timeout_sec=READINESS_TIMEOUT_SEC, score_per_serving_sec=score_per_serving_sec(allocation))
    except ValidationError as e:
        raise PlanError(f"{task_id}: the campaign's settings are not valid: {validation_errors(e)}") from None
    layout = InstanceLayout.calibration(task_id, new_rollout_id())
    toolchain, now, known_tasks = commands.toolchain_here(), time.time(), commands.known_task_ids()

    provisioned = commands.planned_provision(settings, task, task_ref, config=config, overrides=overrides)

    def stages(campaign: CampaignSettings) -> tuple[list[StageDecl], list[AllocationDecl]]:
        return calibration_stages(task_ref=task_ref, task=task, catalog_key=catalog_key, campaign=campaign,
                                  runner_memory_gib=settings.run.runner_memory_gib,
                                  seal_memory_gib=seal_memory_gib(settings.site), held=held,
                                  recipe_key=substrate(settings.site).recipe_key,
                                  provisioned=provisioned)

    def planned(decls: list[StageDecl], groups: list[AllocationDecl]) -> Plan:
        return build_plan(decls=decls, task=task_id, pipeline="calibration", layout=layout, settings=settings,
                          code=code, toolchain=toolchain, job_name=job_name, known_stage_names=KNOWN_STAGE_NAMES,
                          known_tasks=known_tasks, known_patterns=STAGE_NAME_PATTERNS, now=now,
                          secrets=secrets, groups=groups, grower=GROWER)

    decls, groups = stages(campaign)
    secrets = commands.site_secrets(settings, echo, decls)
    try:
        plan = planned(decls, groups)
    except AllocationPastQOS as e:
        raise PlanError(f"{e}; {_fanout_that_fits(campaign, lambda c: planned(*stages(c)))}") from None
    after = f"in {len(groups) - 1} allocations after the instrument's"
    if kind == "report":
        echo(f"calibrating the sampling gate on {', '.join(engines)}: {servings} servings per engine, {after}")
    else:
        echo(f"calibrating the token-level gate on {', '.join(engines)}: {MIN_SERVINGS} servings per engine first, "
             f"{after}; each engine serves to {exposure_target(mu_per_mtok):.0f} tokens of exposure, at most "
             f"{max_servings_per_engine} servings")
    return plan


def _fanout_that_fits(campaign: CampaignSettings, planned: Callable[[CampaignSettings], object]) -> str:
    """What to say of the fanout of *campaign*, an allocation of whose first plan no QOS of the
    site holds: the smallest larger fanout with which *planned*, the planner of that plan, fits
    every allocation under a QOS, or that none does. A larger fanout deals the first servings
    over more allocations, so each holds fewer; past their number it changes nothing."""
    first = len(round_servings(campaign, {}, {e: 0 for e in campaign.engines}))
    for fanout in range(campaign.fanout + 1, first + 1):
        try:
            planned(campaign.model_copy(update={"fanout": fanout}))
        except AllocationPastQOS:
            continue
        return f"the smallest --fanout whose allocations a QOS holds is {fanout}"
    return "no --fanout fits"




def _rows(store: PosixStore, plan: Plan, task: TaskSpec) -> RowIndex:
    """The rows *plan*'s servings drove, from the rows entry its provision stage recorded
    (``provision_record``), checked against the digest recorded for the entry."""
    rows = provision_record(store, plan).rows
    if rows is None:
        raise ValueError(f"{plan.instance} provisioned no rows")
    entry = ContentCache(store).lookup("rows", rows.key)
    if entry is None or entry_sha256(entry) != rows.entry_sha256:
        raise ValueError(f"cache entry rows/{rows.key} is not the one {plan.instance} provisioned")
    return RowIndex.from_sessions(load_sessions(task, entry.path, Split.TEST))


def capped_requests(task_module: str, *, instance: str, engine: str, store: PosixStore) -> dict[str, CappedRows]:
    """Per tenant, the sessions *engine* ran to the output cap in the published captures of the
    calibration instance *instance* of *task_module*'s task (``capped_rows``): the removal list
    a task authors as ``DataSource.removed_rows``. The task is the one the instance served, its
    overrides replayed; each capture is read where it was published, after its sha256 is checked
    against its manifest. Raise ValueError for an instance that is no calibration of the task,
    an engine it did not serve, a file that is not its manifest's, and a member that did not
    honour a cap."""
    task_id = commands.task_id_of(task_module)
    layout = InstanceLayout(instance)
    plan = PlanStore(store, layout).latest()
    if plan is None or plan.pipeline != "calibration" or plan.task != task_id:
        raise ValueError(f"{instance} is no calibration of {task_id} in the store at {store.root}")
    params = InstrumentParams.model_validate(plan.stage(INSTRUMENT).params)
    if params.task.module != task_module:
        raise ValueError(f"{instance} calibrated {params.task.module}, not {task_module}")
    if engine not in params.campaign.engines:
        raise ValueError(f"{instance} served no {engine}; its pool is {params.campaign.engines}")
    task = load_task(params.task)
    rows = _rows(store, plan, task)
    with tempfile.TemporaryDirectory(prefix="hls-capped-requests-") as view:
        captures = CaptureStore(Path(view), params.campaign.workload_fingerprint)
        for stage in plan.stages:
            if stage.kind != "calibrate_capture":
                continue
            serving = CaptureParams.model_validate(stage.params)
            found = read_manifest(store, layout, stage.name)
            # A serving whose capture never published holds nothing to read.
            if serving.engine != engine or found is None:
                continue
            ref = found[0].outputs["capture"]
            if sha256_file(store.path(ref.key))[0] != ref.sha256:
                raise ValueError(f"{ref.key} is not the capture its manifest recorded")
            where = captures.capture_path(engine, serving.draw)
            where.parent.mkdir(parents=True, exist_ok=True)
            where.symlink_to(store.path(ref.key))
        try:
            return capped_rows(captures, engine, rows)
        except CalibrationError as e:
            raise ValueError(" ".join(str(e).split())) from None
