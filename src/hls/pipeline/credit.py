"""The baseline credit pipeline, ``hls baseline-credit``.

A credit instance, ``credits/<task>/<id>``, measures B for a task: N fresh drives of the task's
measured baseline image (one unless ``--drives`` says more), each scored with no baseline credit,
pooled into a ``baseline.json`` that the fold writes into the store's measured area, where runs
of the task find it (``hls.measured``). Its stages:
- ``provision`` and ``images``: the task's weights, rows and tool bundles; the harness image the
  driver runs on and the measured baseline, each with what it builds on, on an evaluation's CPUs;
- ``warm_baseline``: the engine cache entry every drive's server starts from, found in the store
  or built once by serving the baseline as its drives serve it (``pipelines.warm_baseline_decl``);
- ``drive_<k>`` for k from 1: a drive stage of the measured baseline (``pipelines.drive_decl``),
  with no reply counted or refused, in the task's evaluation allocation when its module declares
  one and on the brief's hardware otherwise; each is a fresh deploy whose engine cache is a fresh
  checked copy of the warm stage's entry;
- ``credit_<k>``: the drive scored with no baseline credit (``hls.stages.credit``), its role
  reading the drive's summary with the memory a grading's score stage gets (``score_shape``);
- ``fold``: the drives pooled into the credit.
With ``--fanout K`` above 1 the drives run side by side: allocation 0 holds provision, images and
the warm stage, K allocations after it hold the drives and their credits dealt round robin, each
waiting for allocation 0, and the fold's allocation is submitted once all K have published
(``AllocationDecl.waits_for``); one of them that no QOS of the site holds is refused. With K = 1
the stages are declared in order and the GPU policy groups them, into one allocation by default,
split into a chain of them where no QOS holds it (``hls.pipeline.planner.policy_layout``).

The login node imports the task module for its images and its baseline's start
(``commands.measured_artifact``), reads the ``ENTRY.json`` of each image's cache entry, whose presence
sizes the images stage's disk (``commands.cache_holds``), the cache's provision record of the task,
which sizes the drives' copies of its data and without which the plan is refused
(``commands.planned_provision``), and the secrets file's grant names; it
serves and reads nothing else. A credit is planned only from a committed checkout
(``check_committed_tree``): runs score against it, so it must come from the commit its plan
records. The credit records the hardware its drives ran on."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hls.artifact.artifact import EvaluationAllocation
from hls.baseline_credit import BASELINE_CREDIT_FILE, MIN_DRIVES, BaselineCredit, spread_said
from hls.drive.bounds import UnknownTask, drive_bounds
from hls.pipeline import commands
from hls.pipeline.pipelines import (CREDIT_STAGE_NAMES, EVALUATE_IMAGES_CPUS, EVALUATE_PROVISION_SEC,
                                    KNOWN_STAGE_NAMES, STAGE_NAME_PATTERNS, drive_decl, images_chain, images_stage,
                                    provision_shape, provision_stages, runner_stores, score_shape, warm_baseline_decl)
from hls.pipeline.plan import Plan, PlanStore
from hls.pipeline.planner import AllocationDecl, PlanError, StageDecl, build_plan
from hls.pipeline.preflight import check_committed_tree
from hls.provision.model import Provisioned
from hls.recipes import catalog
from hls.settings import RunSettings
from hls.spec.resources import BaselineHardware, ResourceShape
from hls.stages.baseline_warm import WARM_BASELINE
from hls.stages.credit import CREDIT_SUMMARY_FILE, CreditSummary
from hls.stages.deploy_drive import ImageArtifact
from hls.stages.images import HeldKeys
from hls.store.base import Store
from hls.store.layout import InstanceLayout, new_rollout_id
from hls.store.manifest import OutputRef, checked_output, output_record, read_manifest
from hls.store.posix import PosixStore
from hls.substrate.factory import RecipeKey, seal_memory_gib, substrate
from hls.taskref import TaskRef

#: The stage that pools the drives, the last of a credit's fixed stages.
FOLD = CREDIT_STAGE_NAMES[-1]
#: The memory the fold holds beside the runner's, in GiB, rounded up: pooling 16 drives (each
#: record's score 43 KB) and loading the task takes about 23 MiB above the runner's own resident
#: memory.
FOLD_MEMORY_GIB = 1


def fold_shape(runner_memory_gib: int) -> ResourceShape:
    """The fold's shape: one CPU for 15 minutes, since it reads the drives' small records in the
    runner and starts no role, and ``FOLD_MEMORY_GIB`` beside the runner's
    *runner_memory_gib*."""
    return ResourceShape(cpus=1, memory_gib=runner_memory_gib + FOLD_MEMORY_GIB, time_sec=900)


def drive_stage(k: int) -> str:
    """The name of the *k*-th drive, from 1."""
    return f"drive_{k}"


def credit_stage(k: int) -> str:
    """The name of the stage that scores the *k*-th drive."""
    return f"credit_{k}"


def credit_stages(*, task_ref: TaskRef, task: Any, catalog_key: str, artifact: ImageArtifact,
                  serve_command: Sequence[str], drives: int, fanout: int, run: RunSettings, seal_memory_gib: int,
                  allocation: EvaluationAllocation | None, held: HeldKeys, recipe_key: RecipeKey,
                  provisioned: Provisioned) -> tuple[list[StageDecl], list[AllocationDecl] | None]:
    """The stages that measure *task*'s baseline credit, the ``TASK`` of the module *task_ref*
    names, which the image catalog knows as *catalog_key*, with *drives* drives of the measured
    baseline *artifact* run with *serve_command*, each after the warm stage that leaves the engine
    cache entry its server starts from (``pipelines.warm_baseline_decl``), and the allocation layout
    for *fanout*: None for the GPU policy's (K = 1), else allocation 0 for provision, images and
    the warm stage, K for the drives dealt round robin, and one for the fold that waits for all K.
    The images stage's disk counts the entries *held* says the cache lacks
    (``pipelines.images_stage``), and the warm stage's and each drive's the copies of its roles'
    data from *provisioned*, the provision record the cache holds (``pipelines.drive_decl``). Raise
    ValueError for fewer than ``MIN_DRIVES`` drives, a fanout outside 1 to *drives*, an *artifact*
    that is not the measured baseline the task declares, a role's shape that cannot be sized, and a
    baseline served on no GPU, whose engine cache no key names (``pipelines.warm_baseline_decl``)."""
    if drives < MIN_DRIVES:
        raise ValueError(f"--drives {drives}: a baseline credit needs at least {MIN_DRIVES} drive")
    if not 1 <= fanout <= drives:
        raise ValueError(f"--fanout {fanout} must be between 1 and the {drives} drives")
    images = catalog.task_images(catalog_key)
    baseline = images.measured.get("baseline")
    if artifact.image != baseline:
        raise ValueError(f"a baseline credit drives {catalog_key}'s measured baseline {baseline}, not {artifact.image}")
    chain = images_chain(images, [images.harness, artifact.image])
    ref = task_ref.model_dump(mode="json")
    summary = score_shape(task, run.runner_memory_gib)
    provision = provision_shape(task, runner_memory_gib=run.runner_memory_gib, time_sec=EVALUATE_PROVISION_SEC)
    decls = [*provision_stages(task_ref=task_ref, shape=provision, sandbox_stores=runner_stores(task)),
             images_stage(catalog_task=catalog_key, decls=chain, held=held, recipe_key=recipe_key,
                          cpus=EVALUATE_IMAGES_CPUS),
             warm_baseline_decl(task_ref=task_ref, task=task, catalog_key=catalog_key, artifact=artifact,
                                serve_command=serve_command, allocation=allocation, run=run, provisioned=provisioned,
                                after=("provision", "images"))]
    for k in range(1, drives + 1):
        decls += [drive_decl(name=drive_stage(k), task_ref=task_ref, task=task, catalog_key=catalog_key,
                             artifact=artifact, harness_image=images.harness, serve_command=serve_command,
                             allocation=allocation, run=run, seal_memory_gib=seal_memory_gib,
                             provisioned=provisioned, after=("provision", "images", WARM_BASELINE)),
                  StageDecl(name=credit_stage(k), kind="credit_drive", shape=summary,
                            params={"task": ref, "drive_stage": drive_stage(k)}, after=(drive_stage(k),))]
    credits = [credit_stage(k) for k in range(1, drives + 1)]
    decls.append(StageDecl(name=FOLD, kind="credit_fold", shape=fold_shape(run.runner_memory_gib),
                           params={"task": ref, "credit_stages": credits}, after=tuple(credits)))
    if fanout == 1:
        return decls, None
    shares = [[k for k in range(1, drives + 1) if (k - 1) % fanout == j] for j in range(fanout)]
    groups = [AllocationDecl(stages=("provision", "images", WARM_BASELINE)),
              *(AllocationDecl(stages=tuple(s for k in share for s in (drive_stage(k), credit_stage(k))), depends_on=0)
                for share in shares),
              AllocationDecl(stages=(FOLD,), waits_for=tuple(range(1, fanout + 1)))]
    return decls, groups


def plan_credit(*, config: Path, overrides: Mapping[str, str], task_module: str, drives: int = MIN_DRIVES,
                fanout: int = 1, job_name: str, echo: commands.Echo = print) -> Plan:
    """The first plan of a credit instance of *task_module*'s task: *drives* drives of its
    measured baseline, *fanout* allocations of them side by side, then the fold. Overrides of
    the settings reach the settings and ``task.*`` ones the task reference
    (``commands.split_overrides``), so a credit measured under them records their load. Raise
    ValueError (a PlanError among them) for what cannot be planned: a checkout with uncommitted
    changes, a task that declares no drive bounds, a task that declares no baseline server
    or whose scorer scores against no baseline, a task with no data to drive
    (``commands.check_task_has_data``), a cache that lacks the task's data, naming the
    ``hls provision`` to run first (``commands.planned_provision``), a credit stage a resources
    table gives less memory than a drive's summary needs (``commands.check_score_memory``), and
    what ``credit_stages`` refuses."""
    settings_part, task_part = commands.split_overrides(overrides)
    settings = commands.planning_settings(config, settings_part)
    code = commands.code_state(commands.checkout_root())
    check_committed_tree(code)
    catalog_key = catalog.catalog_task(task_module)
    task_id = commands.task_id_of(task_module)
    task_ref = TaskRef(module=task_module, overrides=task_part)
    task = commands.task_to_plan(task_ref)
    try:
        drive_bounds(task)
    except UnknownTask as e:
        raise PlanError(f"{task_id} cannot be driven: {e.args[0]}") from None
    if task.baseline is None:
        raise PlanError(f"{task_id} declares no baseline server, so it has no baseline credit to measure")
    if not task.scorer.scores_against_baseline():
        raise PlanError(f"{task_id}'s scorer {type(task.scorer).__name__} scores against no baseline credit")
    commands.check_task_has_data(task_id, task)
    artifact, serve = commands.measured_artifact(task, "baseline", catalog_key)
    allocation = commands.evaluation_allocation(commands.produced_artifact(task_module))
    decls, groups = credit_stages(task_ref=task_ref, task=task, catalog_key=catalog_key, artifact=artifact,
                                  serve_command=serve, drives=drives, fanout=fanout, run=settings.run,
                                  seal_memory_gib=seal_memory_gib(settings.site),
                                  allocation=allocation, held=commands.cache_holds(settings),
                                  recipe_key=substrate(settings.site).recipe_key,
                                  provisioned=commands.planned_provision(settings, task, task_ref, config=config,
                                                                         overrides=overrides))
    plan = build_plan(decls=decls, task=task_id, pipeline="credit",
                      layout=InstanceLayout.credit(task_id, new_rollout_id()), settings=settings, code=code,
                      toolchain=commands.toolchain_here(), job_name=job_name, known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=commands.known_task_ids(), now=time.time(),
                      secrets=commands.site_secrets(settings, echo, decls), groups=groups,
                      known_patterns=STAGE_NAME_PATTERNS)
    commands.check_score_memory(plan, task)
    return plan


def read_credit(store: Store, plan: Plan) -> tuple[CreditSummary, OutputRef, bytes]:
    """The credit the credit *plan*'s fold published, as its manifest records it: the summary
    (``credit.json``), and the reference and the bytes of the ``baseline.json``, each checked
    against the digest its manifest recorded (``checked_output``, ``output_record``). Raise
    ValueError for a fold that did not publish, an output missing from its manifest or not the file
    its manifest recorded, a summary that does not parse, and a summary that names another
    credit."""
    found = read_manifest(store, plan.layout, FOLD)
    if found is None:
        raise ValueError(f"{plan.instance} succeeded without a published {FOLD} stage")
    manifest = found[0]
    refs = {}
    for name, file in (("credit", CREDIT_SUMMARY_FILE), ("baseline", BASELINE_CREDIT_FILE)):
        ref = manifest.outputs.get(name)
        if ref is None or not ref.key.endswith(f"/{file}"):
            raise ValueError(f"{plan.instance}: its {FOLD} manifest records no {file}")
        refs[name] = ref
    summary, baseline = output_record(store, refs["credit"], CreditSummary), refs["baseline"]
    if summary.baseline_sha256 != baseline.sha256:
        raise ValueError(f"{refs['credit'].key} names the credit {summary.baseline_sha256}, not the "
                         f"{baseline.sha256} its fold published")
    return summary, baseline, checked_output(store, baseline)


def hardware_said(hardware: list[BaselineHardware] | None) -> str:
    """The hardware a credit's drives ran on, for people: the one record, or every record when
    the drives ran on several, or that it is not recorded."""
    if hardware is None:
        return "on hardware its allocations' statuses do not record"
    if len(hardware) == 1:
        return f"on {hardware[0].describe()}"
    return f"on {len(hardware)} different hardware records: {'; '.join(h.describe() for h in hardware)}"


def say_credit(plan: Plan, echo: commands.Echo) -> None:
    """Say what a succeeded credit measured, on which hardware, where its fold published it, and
    the store's measured file of the task that holds it, which runs of the task use. Raise
    ValueError as ``read_credit`` does."""
    summary, baseline, data = read_credit(PosixStore(plan.store_root), plan)
    echo(f"baseline credit: B = {summary.log_credit_mean:.6f} ({spread_said(BaselineCredit.model_validate_json(data))}) "
         f"measured {hardware_said(summary.hardware)}, in {baseline.key}; the store's {summary.measured_key} holds "
         "it, and runs of the task use it")


def run_credit(plan: Plan, overrides: Mapping[str, str], *, echo: commands.Echo = print) -> int:
    """Submit the credit *plan* and drive it to its end (``commands.start``), then, when it
    succeeded, say what it measured (``say_credit``). The exit status is ``commands.start``'s: 0
    when every stage published, else 1."""
    status = commands.start(plan, overrides, echo=echo)
    final = PlanStore(PosixStore(plan.store_root), plan.layout).latest()
    if final is not None and final.outcome is not None and final.outcome.status == "succeeded":
        say_credit(final, echo)
    return status


