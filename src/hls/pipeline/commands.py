"""What the CLI commands do: warm the cache with ``hls provision`` and ``hls images build``,
build a submission image with ``hls package``, fill a
task's reference answers ahead of its gradings with ``hls oracle fill``, build and grade a
submission with ``hls run``, grade an artifact with ``hls grade``, resume an instance, show its
status, cancel it.

These commands read hls.toml with the strict loader and accept ``--site.*``, ``--run.*`` and
``--resources.*`` overrides; ``hls run``, ``hls grade``, ``hls provision`` and ``hls oracle fill``
also take ``--task.*``, which the plan's task reference carries to every role. Overrides apply
when an instance is created. Afterwards everything about an instance comes from its plan, and
hls.toml is read only to find the store, unless ``--store`` names it. Every orchestrator they
start or take over writes a failed instance's records (``hls.stages.finalize.failed_records``)
before the revision that ends the instance. One whose writer cannot write them, as with a secrets
file that cannot be read, which the one ``cancel`` takes over with may be, ends the instance all
the same and leaves them (``Outcome.records_left``); the command says so, and ``resume`` writes
them once the writer can, whether or not the instance has ended.

An instance whose plan has a finalize stage (a rollout or a grading) records its result, and a
failed one is a scored result: a contestant fault scores zero, in ``failed/score.json``. When
such an instance ends succeeded or failed, the command prints its score, or a build-only
rollout's status, and where it is. A probe or preflight-only build (``hls run --probe-only``,
``--preflight-only``) has no finalize stage and is not scored: the command prints its probe.

Exit statuses:
- ``provision``, ``images build``, ``package``, ``oracle fill`` and ``resume``: 0 when
  the instance succeeded; 1 when it ended failed (a contestant fault), void or cancelled. A
  package instance succeeds whether the submission built or is ``invalid_artifact``; its
  ``package.json`` says which. An oracle fill runs no contestant role, so it never ends failed.
- ``run`` and ``grade``, and ``resume`` of a rollout or a grading: 0 when it ended with the
  contestant's result, that is succeeded or failed with its zero recorded; 1 when it ended void or
  cancelled, or failed with its records left. A build-only run succeeds whether its submission
  built or is ``invalid_artifact``; its record says which. A probe or preflight-only run: 0 when
  the probe ran and the sandbox was stopped, 1 when it ended void.
- ``cancel``: 0 when the instance ended cancelled, else 1.
- ``status``: 0 whatever the outcome, which it prints.
- ``resume``, ``status`` and ``cancel``: 2 when the store holds no such instance; ``resume`` also
  when another orchestrator still holds it. ``cancel``: 3 when the request is left but no
  orchestrator confirmed it."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import json
import os
import re
import shlex
import signal
import stat
import sys
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

import hls
from hls.agent.probe import ProbeRecord
from hls.arms import ArmError, ArmRecord, arm_differences, resolve_arm
from hls.artifact.artifact import AgentProducedArtifact, AgentProducerSpec, EvaluationAllocation
from hls.core.score import Score
from hls.drive.bounds import UnknownTask, drive_bounds
from hls.obs.record import GradingRecord, RolloutRecord
from hls.override import parse_scalar
from hls.pipeline.checkout import code_state
from hls.pipeline.orchestrator import Orchestrator, request_cancel
from hls.pipeline.pipelines import (KNOWN_STAGE_NAMES, NOOP_TASK, PROVISION_SEC, STAGE_NAME_PATTERNS,
                                    grading_stages, images_chain, images_stages, noop_one, noop_two,
                                    oracle_fill_stages, package_stages, producer_network, provision_shape,
                                    provision_stages, reference_gpu_kind, rollout_stages, runner_stores,
                                    sandbox_disk_gib, sandbox_images_stage,
                                    score_shape, submission_gpu_kind)
from hls.pipeline.preflight import (arm_lines, check_committed_tree, check_defaults_committed, check_oracle_coverage,
                                    check_skills_folder, check_upload_folder, task_flags)
from hls.pipeline.plan import Allocation, AllocationStatus, AttemptStatus, Plan, PlanStore
from hls.pipeline.planner import PlanError, StageDecl, build_plan, toolchain_here
from hls.pipeline.stage import stage_class
from hls.pipeline.reconcile import Observation, finished
from hls.measured import (BASELINE_CREDIT_FILE, CALIBRATION_FILE, DEV_REFERENCE_FILE, MEASURES, Found, MeasuredFiles,
                          find, measured_files, needed_files, pinned_dev_files, store_key)
from hls.provision.model import Provisioned
from hls.provision.rows import entry_rows
from hls.provision.sandbox_images import SandboxImageSet, image_file, missing_images
from hls.recipes import catalog
from hls.score.role import SCORE_FILE
from hls.settings import Settings, SiteSettings, load_settings
from hls.stages.agent_build import BUILD_OUTPUT, AgentBuildParams, AgentBuildStage
from hls.stages.calibration import AggregateStage
from hls.stages.credit import CreditDriveParams, CreditDriveStage
from hls.stages.deploy_drive import ImageArtifact, PackagedArtifact, PublishedArtifact
from hls.stages.finalize import FinalizeStage, failed_key, failed_records, finalize_params
from hls.stages.images import HeldKeys, held_keys, image_keys
from hls.stages.package import AGENT_BUILD_STAGE, PipLimits, StoredBundle
from hls.stages.provision import NotProvisioned, ProvisionStage, cached_provision
from hls.stages.score import ScoreParams, ScoreStage
from hls.stages.verify import NO_GPU
from hls.store.base import NotFound, Store
from hls.store.cache import CacheCorrupt, ContentCache
from hls.store.digest import CHUNK, check_sha256_hex, sha256_bytes
from hls.store.layout import (INDEX_DIR, ROLLOUTS, WARMERS, InstanceLayout, NotAnInstance, new_grading_id,
                              new_rollout_id, task_index)
from hls.store.lock import Holder, LockHeld, LockRecord, instance_lease, same_host_liveness
from hls.store.manifest import Manifest, OutputRef, output_record, parse_record, read_manifest
from hls.store.posix import PosixStore
from hls.substrate.seal import refuse_interceptor_port
from hls.substrate.base import JobInfo, JobState, Scheduler
from hls.substrate.factory import RecipeKey, seal_memory_gib, substrate
from hls.substrate.secrets import SecretsFile, read_secrets
from hls.taskref import MODULE_PATTERN, TaskRef, check_task_overrides, first_on_import_path, load_task
from hls.util import validation_errors
from hls.verify.calibration_roles import AGGREGATE_FILE, AggregateRecord
from hls.verify.fill import caching_references

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec

#: Override roots that set the settings, through the strict loader.
SETTINGS_ROOTS = ("site", "run", "resources")
#: The override root that sets fields of the task, which the plan's task reference carries.
TASK_ROOT = "task"

Echo = Callable[[str], None]


class NoSuchInstance(NotFound):
    """The store holds no plan for the instance."""


def checkout_root() -> Path:
    """The checkout this hls package runs from: ``<checkout>/src/hls/__init__.py``."""
    return Path(hls.__file__).resolve().parents[2]


def settings_overrides(overrides: Mapping[str, str]) -> dict[str, str]:
    """*overrides* if every one is under ``site``, ``run`` or ``resources``, else raise ValueError."""
    bad = sorted(k for k in overrides if k.split(".", 1)[0] not in SETTINGS_ROOTS)
    if bad:
        raise ValueError(f"this command takes only --site.*, --run.* and --resources.* overrides, not "
                         f"{', '.join(bad)}")
    return dict(overrides)


def split_overrides(overrides: Mapping[str, str]) -> tuple[dict[str, str], dict[str, Any]]:
    """*overrides* split into the settings' (``--site.*``, ``--run.*``, ``--resources.*``) and the
    task's (``--task.*``, each value typed by ``parse_scalar``: an int, a float, a bool or the
    text). Raise ValueError naming any other root, and any task override no task reference takes
    (``hls.taskref.check_task_overrides``), before the command reads anything."""
    roots = (*SETTINGS_ROOTS, TASK_ROOT)
    bad = sorted(k for k in overrides if k.split(".", 1)[0] not in roots)
    if bad:
        raise ValueError(f"this command takes --site.*, --run.*, --resources.* and --task.* overrides, not "
                         f"{', '.join(bad)}")
    settings = {k: v for k, v in overrides.items() if k.split(".", 1)[0] in SETTINGS_ROOTS}
    task = {k: parse_scalar(v) for k, v in overrides.items() if k.split(".", 1)[0] == TASK_ROOT}
    check_task_overrides(task)
    return settings, task


def _scheduler(scheduler: Scheduler | None, site: SiteSettings) -> Scheduler:
    """*scheduler*, else the scheduler of the substrate *site*, a plan's, runs on."""
    return scheduler if scheduler is not None else substrate(site).scheduler()


def check_store_root(root: Path) -> None:
    """Refuse a ``site.store_root`` this user cannot use: one that does not exist and cannot be
    made, or one it cannot write. Made here when it can be, as the orchestrator would make it.
    Raise PlanError naming the setting and the path."""
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise PlanError(f"site.store_root {root} does not exist and cannot be made: "
                        f"{e.strerror or type(e).__name__}") from None
    if not root.is_dir():
        raise PlanError(f"site.store_root {root} is not a directory")
    if not os.access(root, os.W_OK | os.X_OK):
        raise PlanError(f"site.store_root {root} is not writable by this user")


def planning_settings(config: Path, overrides: Mapping[str, str]) -> Settings:
    """The settings a plan is made under: hls.toml at *config* with the settings' *overrides*
    (``load_settings``), its store root checked (``check_store_root``), since planning reads the
    store and the orchestrator writes it."""
    settings = load_settings(config, overrides)
    check_store_root(settings.site.store_root)
    return settings


def store_root(config: Path, store: Path | None) -> Path:
    """The store's root: *store* when given, else the ``site.store_root`` of the hls.toml at
    *config*, whose reading raises SettingsError for a file that is missing or not valid."""
    return store if store is not None else load_settings(config).site.store_root


def open_store(config: Path, store: Path | None) -> PosixStore:
    """The store at ``store_root(config, store)``."""
    return PosixStore(store_root(config, store))


def _latest(store: PosixStore, layout: InstanceLayout) -> Plan:
    """The instance's latest plan. Raise NoSuchInstance when the store holds none."""
    plan = PlanStore(store, layout).latest()
    if plan is None:
        raise NoSuchInstance(f"no such instance under {store.root}")
    return plan


def settled(plan: Plan, scheduler: Scheduler, *, now: float) -> bool:
    """Whether the instance has ended and every job of it is accounted for, as the orchestrator
    judges when it may stop (``reconcile.finished``): no cancel or stray outstanding, no sbatch
    answer unresolved, and unless it succeeded no job still queued or running. Asks the
    scheduler about the jobs of the live submissions; a job it cannot answer for counts as not
    settled; so is an instance whose records an orchestrator left (``Outcome.records_left``),
    which the next one writes. When the instance is not settled, an orchestrator has work to do."""
    if plan.outcome is None or plan.outcome.records_left:
        return False
    ids = sorted(s.job_id for a in plan.allocations for s in a.submissions if s.live and s.job_id is not None)
    jobs = scheduler.states(ids) if ids else {}
    return finished(plan, Observation(now=now, published=frozenset(), jobs=jobs))


def _scored(plan: Plan) -> bool:
    """Whether the instance ended with the contestant's result: it succeeded, or it failed, which
    is the contestant's zero, and its plan has a finalize stage, so the zero is recorded
    (``failed_records``) unless its records were left (``Outcome.records_left``), which it is
    not until they are written. A probe or preflight-only build has no finalize stage and is
    never the contestant's result."""
    assert plan.outcome is not None
    status = plan.outcome.status
    return status == "succeeded" or (status == "failed" and finalize_params(plan) is not None
                                     and not plan.outcome.records_left)


def _exit_status(plan: Plan) -> int:
    """0 when the instance scored (``_scored``), else 1: a failed instance that records no
    score, void and cancelled alike."""
    return 0 if _scored(plan) else 1


def _score(store: PosixStore, plan: Plan) -> tuple[str, Score] | None:
    """The key and record of the ``score.json`` of an instance that ended scored (``_scored``): its
    score stage's output, checked against the stage's manifest, or for a failed instance the zero
    the orchestrator recorded in ``failed/``, which no manifest pins. None for an instance that
    records no score: its plan has no finalize stage, or it is a build-only rollout that succeeded.
    Raise ValueError for one that succeeded without a published score, and UnreadableRecord for a
    score that is not the one its manifest recorded or is not a score."""
    params = finalize_params(plan)
    if params is None:
        return None
    if plan.outcome is not None and plan.outcome.status == "failed":
        key = failed_key(plan.layout, SCORE_FILE)
        return key, parse_record(key, store.read_bytes(key), Score)
    if params.score_stage is None:
        return None
    found = read_manifest(store, plan.layout, params.score_stage)
    if found is None or "score" not in found[0].outputs:
        raise ValueError(f"{plan.instance} succeeded without a published score stage {params.score_stage}")
    ref = found[0].outputs["score"]
    return ref.key, output_record(store, ref, Score)


def _say_score(plan: Plan, echo: Echo) -> None:
    """Say what an instance that ended with the contestant's result (``_scored``) and records it
    holds, and where:
    - a score: "<instance>: score <value>[, zero for <reasons>] (<key>)", from its score stage or,
      for a failed instance, from the zero the orchestrator recorded;
    - a build-only rollout that succeeded: "<instance>: rollout <status> (<key>)", from its record.
    An instance without a finalize stage records nothing to say. A published record is read as its
    manifest recorded it (``output_record``)."""
    if not _scored(plan) or finalize_params(plan) is None:
        return
    store = PosixStore(plan.store_root)
    scored = _score(store, plan)
    if scored is not None:
        key, score = scored
        zero = f", zero for {', '.join(score.zero_reasons)}" if score.zero_reasons else ""
        echo(f"{plan.instance}: score {score.value}{zero} ({key})")
        return
    stage = next(s.name for s in plan.stages if s.kind == FinalizeStage.kind)
    found = read_manifest(store, plan.layout, stage)
    if found is None or "rollout" not in found[0].outputs:
        raise ValueError(f"{plan.instance} succeeded without a published rollout record")
    ref = found[0].outputs["rollout"]
    echo(f"{plan.instance}: rollout {output_record(store, ref, RolloutRecord).status} ({ref.key})")


def _say_hardware(plan: Plan, echo: Echo) -> None:
    """For an instance whose finalize published a grading record, say the record's hardware
    warning (``HarnessConditions.hardware_warning``): its drive ran on other
    hardware than its baseline credit's drives; and which of the measured files it used are the
    task's defaults (``HarnessConditions.measured_note``).
    Nothing when the record has neither. The record is read as its manifest recorded it
    (``output_record``)."""
    stage = next((s.name for s in plan.stages if s.kind == FinalizeStage.kind), None)
    if stage is None:
        return
    store = PosixStore(plan.store_root)
    found = read_manifest(store, plan.layout, stage)
    if found is None or "grading" not in found[0].outputs:
        return
    conditions = output_record(store, found[0].outputs["grading"], GradingRecord).conditions
    for line in (conditions.hardware_warning, conditions.measured_note):
        if line is not None:
            echo(f"{plan.instance}: {line}")


def _say_probe(plan: Plan, echo: Echo) -> None:
    """For a probe or preflight-only build, say the probe's result once it published one: its
    ``all_ok`` and the names of the checks that did not pass, as a stop build records no score.
    A stop build that ended void published no probe (it is evidence in the attempt), so only its
    void reason is said. The probe is read as its manifest recorded it (``output_record``)."""
    stop = next((s for s in plan.stages if s.kind == AgentBuildStage.kind and s.params.get("stop_after")), None)
    if stop is None:
        return
    store = PosixStore(plan.store_root)
    found = read_manifest(store, plan.layout, stop.name)
    if found is None or "probe" not in found[0].outputs:
        return
    probe = output_record(store, found[0].outputs["probe"], ProbeRecord)
    failed = [c.name for c in probe.checks if not c.ok]
    note = f"; failed: {', '.join(failed)}" if failed else ""
    echo(f"{plan.instance}: probe all_ok={str(probe.all_ok).lower()}{note}")


def _say_calibration(plan: Plan, echo: Echo) -> None:
    """For a calibration that succeeded, say what it measured: the gate it calibrates, its ceiling
    and the servings it was assembled from, where its aggregate stage published the
    ``calibration.json``, that the store's measured area holds it (``hls.measured``), and each
    warning its aggregate recorded (``AggregateRecord.warnings``). Nothing for any other instance.
    Raise ValueError for an aggregate that did not publish, or whose record is not the one its
    manifest recorded."""
    assert plan.outcome is not None
    aggregate = next((s for s in plan.stages if s.kind == AggregateStage.kind), None)
    if aggregate is None or plan.outcome.status != "succeeded":
        return
    store = PosixStore(plan.store_root)
    found = read_manifest(store, plan.layout, aggregate.name)
    if found is None:
        raise ValueError(f"{plan.instance} succeeded without a published {aggregate.name} stage")
    manifest = found[0]
    ref = manifest.outputs["aggregate"]
    if not ref.key.endswith(f"/{AGGREGATE_FILE}"):
        raise ValueError(f"{ref.key} is not the {AGGREGATE_FILE} its manifest recorded")
    record = output_record(store, ref, AggregateRecord)
    counts = ", ".join(f"{e} {len(d)} servings" for e, d in sorted(record.servings.items()))
    echo(f"calibration ({record.kind}): ceiling {record.ceiling:.6g} over {counts}, in "
         f"{manifest.outputs['calibration'].key}; the store's {store_key(plan.task, CALIBRATION_FILE)} holds it, "
         "and runs of the task use it")
    for line in record.warnings:
        echo(f"warning: {line}")


def _finish(plan: Plan, echo: Echo) -> int:
    """Say how the instance ended, that its records are left when they are
    (``Outcome.records_left``), the score of one that scored and its record's hardware warning,
    the probe of a stop build and what a calibration measured; return its exit status."""
    assert plan.outcome is not None
    echo(f"{plan.instance}: {plan.outcome.status}: {plan.outcome.reason}")
    if plan.outcome.records_left:
        echo(f"{plan.instance}: its records are left for hls resume; its failed_records events say why")
    _say_score(plan, echo)
    _say_hardware(plan, echo)
    _say_probe(plan, echo)
    _say_calibration(plan, echo)
    return _exit_status(plan)


def _partition_note(plan: Plan, alloc: Allocation) -> str:
    """", partition <name>" for an allocation that does not go to the site's GPU partition."""
    partition = alloc.placement.partition
    if partition is None or partition == plan.settings.site.partition:
        return ""
    return f", partition {partition}"


#: How ``hls status`` names each substrate's scheduler, and whether it gives a pending job's
#: reason: the local scheduler's says in words what the job waits for, such as how many GPUs are
#: free; Slurm's is a code of its own.
_STATUS_SCHEDULERS = {"slurm": ("Slurm", False), "docker": ("local scheduler", True)}


def _resources(plan: Plan, alloc: Allocation) -> str:
    """What *alloc* asks Slurm for, as "1 x H200, 8 CPUs, 64 GiB, 75 min", then the partition
    when it is not the site's GPU partition and the QOS it was planned under, if any."""
    r = alloc.request
    qos = f", qos {alloc.placement.qos}" if alloc.placement.qos is not None else ""
    return (f"{r.gpus} x {r.gpu_kind or 'no GPU'}, {r.cpus} CPUs, {r.memory_gib} GiB, {r.time_sec // 60} min"
            f"{_partition_note(plan, alloc)}{qos}")


def start(plan: Plan, overrides: Mapping[str, str], *, scheduler: Scheduler | None = None, echo: Echo = print) -> int:
    """Print what *plan* asks for, submit it and drive it to its end; the exit status. *overrides*
    are the ones the plan was made with. A ValueError raised here is a fault of the drive, not a
    refusal of what was asked: the refusals all come from planning (``plan_noop``,
    ``plan_provision``)."""
    echo(f"instance {plan.instance}")
    if "site.store_root" in overrides:
        # resume, status and cancel read hls.toml for the store, which this override did not change.
        echo(f"store root {plan.store_root}; resume, status and cancel need --store {plan.store_root}")
    for alloc in plan.allocations:
        echo(f"allocation {alloc.index}: {', '.join(alloc.stages)}; {_resources(plan, alloc)}")
    for w in plan.warnings:
        echo(f"warning: {w}")
    orch = Orchestrator.create(plan, _scheduler(scheduler, plan.settings.site), holder=Holder.here("orchestrator"),
                               on_failed=failed_records)
    return _finish(orch.run(), echo)


def _already_ended(plan: Plan, echo: Echo) -> None:
    assert plan.outcome is not None
    echo(f"{plan.instance} already ended: {plan.outcome.status}: {plan.outcome.reason}")


def _who(record: LockRecord) -> str:
    """The holder of a lease, as "orchestrator on host pid 1"."""
    h = record.holder
    return f"{h.purpose} on {h.host} pid {h.pid}"


def _beat(record: LockRecord, now: float) -> str:
    """When the holder last renewed its lease, as "heartbeat 3 s ago"."""
    return f"heartbeat {now - record.heartbeat_at:.0f} s ago"


@dataclass(frozen=True)
class Settled:
    """An instance that ended with nothing outstanding while a command waited for its lease."""

    #: Its final plan.
    plan: Plan
    #: The lease record of the orchestrator the command was waiting for; None if it waited for none.
    holder: LockRecord | None


def _take_over(store: PosixStore, layout: InstanceLayout, scheduler: Scheduler, holder: Holder, *,
               give_up_at: Callable[[LockRecord], float], stop_if_alive: bool, echo: Echo,
               sleep: Callable[[float], None], clock: Callable[[], float],
               cancel_only: bool = False) -> Orchestrator | Settled:
    """The orchestrator that holds the instance's lease, or the instance's final plan when it
    is settled before the lease is free. While another orchestrator holds the lease, say once
    that this waits, then wait one poll at a time and try again; once the clock passes the time
    *give_up_at* names for the first holder seen, try once more and then raise that holder's
    LockHeld. With *stop_if_alive*, raise at once for a holder that is alive on this host.
    *cancel_only* takes the instance over to cancel it, even with a secrets file that cannot be
    read (``Orchestrator.attach``)."""
    deadline = None
    waiting: LockRecord | None = None
    while True:
        plan = _latest(store, layout)
        if settled(plan, scheduler, now=clock()):
            return Settled(plan, waiting)
        try:
            return Orchestrator.attach(store, layout, scheduler, holder=holder, clock=clock, sleep=sleep,
                                       on_failed=failed_records, cancel_only=cancel_only)
        except LockHeld as held:
            if stop_if_alive and same_host_liveness(held.record.holder) is True:
                raise
            waiting = held.record
            now = clock()
            if deadline is None:
                deadline = give_up_at(held.record)
                echo(f"waiting for {_who(waiting)} ({_beat(waiting, now)}), up to {deadline - now:.0f} s")
            if now > deadline:
                raise
            sleep(min(plan.settings.run.poll_sec, deadline - now + 0.1))


def check_score_memory(plan: Plan, task: Any) -> None:
    """Refuse *plan*, an instance of *task*, when a stage whose role reads a drive's summary has less
    memory than that role needs for the summary a drive's replies can make (``score_shape``), as a
    ``[resources.<task>.<stage>]`` table can give it: the replies could then run it out of memory and
    void the instance. Such a stage is a score stage or a credit's credit stage, each reading the
    summary of the drive stage its parameters name. A plan with neither passes."""
    for stage in plan.stages:
        if stage.kind == ScoreStage.kind:
            drive = plan.stage(ScoreParams.model_validate(stage.params).drive_stage)
        elif stage.kind == CreditDriveStage.kind:
            drive = plan.stage(CreditDriveParams.model_validate(stage.params).drive_stage)
        else:
            continue
        need = score_shape(task, plan.settings.run.runner_memory_gib).memory_gib
        held = stage.shape.memory_gib
        if held < need:
            raise PlanError(f"stage {stage.name}: [resources.{plan.task}.{stage.name}] gives it {held} GiB, less than "
                            f"the {need} GiB its role needs for the summary {drive.name}'s replies can make "
                            "(score_shape); those replies could run it out of memory and void the instance")


def task_to_plan(task_ref: TaskRef) -> TaskSpec:
    """The task *task_ref* names, loaded for a plan (``load_task``) and refused when its contract
    port is one the egress interceptor listens on (``refuse_interceptor_port``). Every plan that
    loads a task loads it here: a run, a grading, a baseline credit, a calibration and the warmers
    that provision or fill for a task, so every plan that serves a task's artifact or reads its
    contract refuses such a port before anything is submitted."""
    task = load_task(task_ref)
    refuse_interceptor_port(task.id, task.contract.port)
    return task


def find_measured(store: PosixStore, *, task_ref: TaskRef, task: Any, task_id: str) -> list[Found]:
    """Each measured file *task*, the task *task_ref* names with its overrides applied, loads,
    looked for in the store's measured area and then among the task's defaults
    (``hls.measured.find``)."""
    return [find(name, task_ref=task_ref, task=task, task_id=task_id, checkout=checkout_root(), store=store)
            for name in needed_files(task)]


def planned_dev_files(settings: Settings, measured: MeasuredFiles) -> dict[str, dict[str, str]]:
    """The files the dev reference *measured* pins publishes beside each tenant's dev rows
    (``hls.measured.pinned_dev_files``), by tenant id, which the stages whose roles see the dev
    split declare the views of (``hls.provision.mounts.copied_entries``)."""
    return pinned_dev_files(measured, checkout=checkout_root(), store_root=settings.site.store_root)


def measuring_command(name: str, task_ref: TaskRef) -> str:
    """The command that measures file *name* of *task_ref*: its module with its task overrides,
    each as a command line gives it (``task_flags``), quoted for a shell."""
    return shlex.join([*MEASURES[name].split(), task_ref.module, *task_flags(task_ref)])


def pinned_measured(store: PosixStore, *, task_ref: TaskRef, task: Any, task_id: str,
                    scored: bool = True) -> tuple[MeasuredFiles, list[str]]:
    """The measured files a plan of *task* pins (``find_measured``), and one line for each, saying
    which file it uses. For a *scored* plan, raise PlanError naming each file none fits, why each
    that was there does not, and the command that measures it; its caller then checks that a
    default is committed (``check_defaults_committed``). A plan that is not scored, a probe's, pins
    what fits and leaves the rest out."""
    found = find_measured(store, task_ref=task_ref, task=task, task_id=task_id)
    if not scored:
        pins = {f.name: f.pin for f in found if f.pin is not None}
        return (MeasuredFiles(credit=pins.get(BASELINE_CREDIT_FILE), calibration=pins.get(CALIBRATION_FILE),
                              dev_reference=pins.get(DEV_REFERENCE_FILE)), [f.said() for f in found])
    missing = [f for f in found if f.pin is None]
    if missing:
        raise PlanError("; ".join(f"{f.said()}: measure it with {measuring_command(f.name, task_ref)}, or let "
                                  "hls run measure it" for f in missing))
    return measured_files(found), [f.said() for f in found]


def check_task_has_data(task_id: str, task: Any) -> None:
    """Refuse a plan that drives a task none of whose tenants declares a data source: its drive
    would send no request. A grading's verify would then fail it for zero items only after every
    stage before it, a run's agent build among them, had run."""
    if all(t.data_source is None for t in task.effective_tenants()):
        raise PlanError(f"{task_id} declares no data source on any tenant, so its drive would send no request; "
                        "declare the task's data_source or a tenant's")


def _warmer_job(settings: Settings, job_name: str) -> tuple[str, bool]:
    """The job name a cache warmer on the CPU partition is submitted under, and whether it waits
    for the site's other warmers: Slurm's ``singleton`` keys on the job name, so a warmer that
    runs one at a time (``site.warmers_one_at_a_time``) takes ``site.warmer_job_name``."""
    if settings.site.warmers_one_at_a_time:
        return settings.site.warmer_job_name, True
    return job_name, False


def plan_noop(*, config: Path, overrides: Mapping[str, str], two_stage: bool, gpus: int, gpu_kind: str,
              sleep_sec: int, stage_time_sec: int, fail_attempts: int, job_name: str,
              contestant: bool = False, batch_hog_gib: int = 0, with_cpu_role: bool = False) -> Plan:
    """The first plan of a no-op instance, the minimal pipeline (``hls.pipeline.pipelines``); nothing
    is submitted. No command makes this plan: the test suite and a site's live tests call this and
    hand the plan to ``start``. *contestant*, *batch_hog_gib* and *with_cpu_role* apply to the
    one-stage pipeline. Raise ValueError for what cannot be planned."""
    settings = planning_settings(config, settings_overrides(overrides))
    if two_stage:
        decls = noop_two(gpu_kind=gpu_kind, sleep_sec=sleep_sec, stage_time_sec=stage_time_sec,
                         fail_attempts=fail_attempts)
    else:
        decls = noop_one(gpus=gpus, gpu_kind=gpu_kind, sleep_sec=sleep_sec, stage_time_sec=stage_time_sec,
                         fail_attempts=fail_attempts, contestant=contestant,
                         batch_hog_gib=batch_hog_gib, with_cpu_role=with_cpu_role)
    layout = InstanceLayout.rollout(NOOP_TASK, new_rollout_id())
    return build_plan(decls=decls, task=NOOP_TASK, pipeline="noop2" if two_stage else "noop", layout=layout,
                      settings=settings, code=code_state(checkout_root()), toolchain=toolchain_here(),
                      job_name=job_name, known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=known_task_ids(), known_patterns=STAGE_NAME_PATTERNS, now=time.time())




def _check_own(root: Path, task_module: str, name: str, origin: str) -> None:
    """Raise ValueError unless the file *origin* of module *name*, which is *task_module* or a
    package on its dotted name, is the checkout's own code: inside the checkout at *root*, and
    outside this interpreter's environment, which a checkout may hold (``<checkout>/.venv``)
    but whose packages are installed libraries."""
    where = Path(origin).resolve()
    what = f"task module {task_module}" if name == task_module else f"package {name} of task module {task_module}"
    if not where.is_relative_to(root):
        raise ValueError(f"{what} is {where}, which is not in the checkout {root}; the allocation imports task "
                         "modules from the checkout")
    environment = Path(sys.prefix).resolve()
    if where.is_relative_to(environment):
        raise ValueError(f"{what} is {where}, which is installed in this interpreter's environment {environment}, "
                         "not the checkout's own code")


def _check_before_import(root: Path, task_module: str) -> None:
    """Check each part of *task_module*'s dotted name that has a file before the part runs.
    Finding a part imports only its parent, which was checked one step before. A part with no
    file of its own, a namespace package or a built-in module, runs no code from outside the
    interpreter."""
    parts = task_module.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        spec = importlib.util.find_spec(name)
        if spec is None:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        if spec.has_location and spec.origin is not None:
            _check_own(root, task_module, name, spec.origin)


def _import_task_module(root: Path, task_module: str) -> ModuleType:
    """*task_module*, imported from the checkout at *root*. Unless it was imported already,
    every part of its name is checked before it runs; its own file is checked once it is
    imported."""
    if not re.fullmatch(MODULE_PATTERN, task_module):
        raise ValueError(f"{task_module!r} is not a dotted module name, such as tasks.multitenant_slo.task")
    with first_on_import_path(root):
        try:
            if task_module not in sys.modules:
                _check_before_import(root, task_module)
            module = importlib.import_module(task_module)
        except ImportError as e:
            raise ValueError(f"cannot import task module {task_module}: {e}") from e
    where = getattr(module, "__file__", None)
    if where is None:
        raise ValueError(f"task module {task_module} has no file: it is a namespace package or a built-in module")
    _check_own(root, task_module, task_module, where)
    return module


def task_id_of(task_module: str) -> str:
    """The id of *task_module*'s ``TASK``, which names its instances. The module must be the
    checkout's own code, since the allocation's roles import it from there, and nothing of it
    runs before that is checked; a ValueError says what is wrong. Importing a task module builds
    its TaskSpec, which loads no model, dataset or hub library and downloads nothing."""
    module = _import_task_module(checkout_root(), task_module)
    task = getattr(module, "TASK", None)
    if task is None:
        raise ValueError(f"task module {task_module} has no TASK")
    task_id = getattr(task, "id", None)
    if not isinstance(task_id, str):
        raise ValueError(f"task module {task_module}: its TASK has no string id")
    return task_id


def known_task_ids() -> frozenset[str]:
    """The tasks whose ids a ``[resources]`` table may be keyed by besides a plan's own
    (``hls.pipeline.planner.check_setting_names``): the no-op task, and the TASK of each task
    directory this checkout holds (``catalog.task_keys``, ``task_id_of``). A plan of another task
    knows that task as its own. A ValueError says which task module cannot be read."""
    modules = (catalog.task_module(t) for t in catalog.task_keys(checkout_root() / "tasks"))
    return frozenset({NOOP_TASK, *(task_id_of(module) for module in modules)})


def cache_holds(settings: Settings) -> HeldKeys:
    """Which of an images stage's keys the site's store's cache holds now (``held_keys``), asked
    by the pipeline about the stage's own images alone: the stage finds those and fills the rest,
    which its disk counts, each holding the image file of the site's substrate. Only each entry's
    ``ENTRY.json`` is read."""
    return functools.partial(held_keys, ContentCache(PosixStore(settings.site.store_root)),
                             image_file=substrate(settings.site).image_file)


def _checked_secrets(settings: Settings, echo: Echo, decls: Sequence[StageDecl],
                     secrets: SecretsFile | None) -> SecretsFile | None:
    """*secrets*, the site's secrets file as ``read_secrets`` found it, for the grant names the plan
    records, with a note when there is none. A missing file is refused with PlanError, before
    anything is submitted, when a stage of *decls* needs every value of it
    (``needs_secret_registry``: the package scans', finalize's): its allocation would only stop as
    the stage started."""
    if secrets is not None:
        return secrets
    path = settings.site.secrets_path
    needing = [d.name for d in decls if stage_class(d.kind).needs_secret_registry]
    if needing:
        raise PlanError(f"stage{'s' if len(needing) > 1 else ''} {', '.join(needing)} "
                        f"need{'' if len(needing) > 1 else 's'} every value of the site's secrets file, and there is "
                        f"none at {path}")
    echo(f"note: no secrets file at {path}; no stage is granted a secret")
    return None


def site_secrets(settings: Settings, echo: Echo, decls: Sequence[StageDecl] = ()) -> SecretsFile | None:
    """The site's secrets file for a plan of *decls* (``read_secrets``, ``_checked_secrets``)."""
    return _checked_secrets(settings, echo, decls, read_secrets(settings.site.secrets_path))


def provision_command_text(task_ref: TaskRef, config: Path, overrides: Mapping[str, str]) -> str:
    """The command that fills the cache a plan of the task *task_ref* names, from *config* with
    *overrides*, reads: ``hls provision`` of its module with the same configuration, the same site
    overrides, which choose the store, each as ``--<key>=<value>``, and its task overrides, which
    choose what is provisioned (``task_flags``), quoted for a shell."""
    site = [f"--{key}={value}" for key, value in sorted(overrides.items()) if key.startswith("site.")]
    return shlex.join(["hls", "provision", task_ref.module, "--config", str(config), *site, *task_flags(task_ref)])


def planned_provision(settings: Settings, task: Any, task_ref: TaskRef, *, config: Path,
                      overrides: Mapping[str, str]) -> Provisioned:
    """The provision record the site's cache holds for *task*, the ``TASK`` *task_ref* names with
    its overrides applied: what its provision stage will publish, each entry with the disk its
    node-local copy takes, which the plan declares as the disk of the stages that copy them
    (``cached_provision``). Raise PlanError, before anything is planned, when the cache lacks an
    entry the task needs, naming each and the command to run first (``provision_command_text``,
    from the plan's *config* and *overrides* and the task's own overrides): the stages' disk on
    the node is sized from what the cache holds, so a cold cache refuses the plan, though the plan
    provisions in its own allocation too. Raise PlanError too when the cache refuses an entry, and
    when a sandbox store the site names lacks what the task's tool runner needs for its corpus
    (``missing_store_entries``), a sandbox image the task's module declares among it, naming the
    same command."""
    module = task_ref.module
    cache = ContentCache(PosixStore(settings.site.store_root))
    try:
        provisioned = cached_provision(task, cache)
    except NotProvisioned as e:
        raise PlanError(f"{module}'s data is not all in the cache, so the plan cannot make room on the node for "
                        f"its copies ({e}): run {provision_command_text(task_ref, config, overrides)} first") from None
    except CacheCorrupt as e:
        raise PlanError(f"{module}'s data in the cache cannot be used: {e}") from None
    missing = missing_store_entries(task, settings, corpus_rows(provisioned, cache),
                                    images=declared_sandbox_images(module))
    if missing:
        raise PlanError(f"{module}'s sandbox stores lack what its corpus needs ({'; '.join(missing)}): run "
                        f"{provision_command_text(task_ref, config, overrides)} first")
    return provisioned


#: How many missing entries of one store a refusal names before it counts the rest.
MISSING_NAMED = 3


def corpus_rows(provisioned: Provisioned, cache: ContentCache) -> list[dict]:
    """Every row of both splits of every tenant the rows entry of *provisioned* holds, as the
    cache holds them; none when it records no rows or the cache lacks the entry."""
    entry = cache.lookup("rows", provisioned.rows.key) if provisioned.rows is not None else None
    return entry_rows(entry.path, provisioned.rows.tenants) if entry is not None else []


def missing_store_entries(task: Any, settings: Settings, rows: Sequence[dict], *,
                          images: SandboxImageSet | None = None) -> list[str]:
    """What the site's sandbox stores lack of what *task*'s tool runner needs for its corpus *rows*
    (``ToolRunner.store_entries``, ``corpus_rows``), one phrase per store naming the first
    ``MISSING_NAMED`` and counting the rest; empty for a task whose runner names no store. A store
    the site does not name is left to the planner, which refuses it by name. An entry that is the
    file of one of *images*, the sandbox images the task's module declares, is there only when the
    store holds that image as ``hls provision`` leaves it (``missing_images``): the one test the
    plan of the stage that pulls them asks, so a store this passes is one that plan leaves alone.
    Any other entry is there when a file or directory of its kind is."""
    runner = task.agent.tool_runner if task.agent is not None else None
    if runner is None or not runner.host_paths:
        return []
    declared: dict[str, str] = {}
    not_held: set[str] = set()
    directory = settings.sandbox_stores.get(images.store) if images is not None else None
    if images is not None and directory is not None:
        declared = {image_file(directory, image.name).name: image.name for image in images.images}
        not_held = set(missing_images(directory, images))
    lacking: dict[str, list[str]] = {}
    for e in runner.store_entries(rows):
        root = settings.sandbox_stores.get(e.store)
        if root is None:
            continue
        target = root / e.path if e.path else root
        if images is not None and e.store == images.store and e.path in declared:
            there = declared[e.path] not in not_held
        else:
            there = target.is_file() if e.kind == "file" else target.is_dir()
        if not there:
            lacking.setdefault(e.store, []).append(e.what)
    phrases = []
    for store, whats in lacking.items():
        named = ", ".join(whats[:MISSING_NAMED]) + (f" and {len(whats) - MISSING_NAMED} more"
                                                    if len(whats) > MISSING_NAMED else "")
        phrases.append(f"{store} lacks {named}")
    return phrases


def declared_sandbox_images(task_module: str) -> SandboxImageSet | None:
    """The sandbox images the imported module *task_module* declares as ``SANDBOX_IMAGES``; None
    when it declares none, or is not imported. Raise ValueError when what it declares is no
    ``SandboxImageSet``."""
    declared = getattr(sys.modules.get(task_module), "SANDBOX_IMAGES", None)
    if declared is not None and not isinstance(declared, SandboxImageSet):
        raise ValueError(f"{task_module}'s SANDBOX_IMAGES is {type(declared).__name__}, not a SandboxImageSet")
    return declared


def missing_sandbox_images(settings: Settings, images: SandboxImageSet) -> list[str]:
    """The images of *images* the site's store does not hold, by name
    (``hls.provision.sandbox_images.missing_images``): every one when the site names no directory
    for the store, which the plan of the stage that would pull them then refuses."""
    directory = settings.sandbox_stores.get(images.store)
    if directory is None:
        return [image.name for image in images.images]
    return missing_images(directory, images)


def plan_provision(*, config: Path, overrides: Mapping[str, str], task_module: str, weights: bool, job_name: str,
                   echo: Echo = print) -> Plan:
    """The first plan of a cache warmer that provisions what *task_module*'s ``TASK`` needs, on
    the site's CPU partition when it has one, one at a time with the site's other warmers
    (``_warmer_job``); nothing is submitted. *weights* False leaves out the task model's weights.
    A task that declares sandbox images (``declared_sandbox_images``) gets the ``sandbox_images``
    stage after ``provision`` when its site store lacks one of them (``missing_sandbox_images``),
    and no such stage when the store holds them all.
    *overrides* hold the task's (``--task.*``) beside the settings': the warmer provisions the task
    they make, so a grading with the same overrides finds what it provisioned. The login node
    only imports the task module for its id, to check the overrides and for the memory its
    provision roles need (``provision_shape``), and reads the secrets file for its grant names;
    every download happens in the allocation. Raise ValueError for what cannot be planned, a
    secrets file that cannot be used and an override that names no field of the task among it."""
    settings_over, task_over = split_overrides(overrides)
    settings = planning_settings(config, settings_over)
    task = task_id_of(task_module)
    layout = InstanceLayout.warmer(task, new_rollout_id())
    task_ref = TaskRef(module=task_module, overrides=task_over)
    # An override that names no field, and a contract port the interceptor listens on, are refused
    # here, before anything is submitted.
    spec = task_to_plan(task_ref)
    shape = provision_shape(spec, runner_memory_gib=settings.run.runner_memory_gib,
                            time_sec=PROVISION_SEC)
    decls = provision_stages(task_ref=task_ref, shape=shape, weights=weights, sandbox_stores=runner_stores(spec))
    images = declared_sandbox_images(task_module)
    if images is not None:
        missing = missing_sandbox_images(settings, images)
        if missing:
            echo(f"sandbox store {images.store} lacks {len(missing)} of {task_module}'s {len(images.images)} "
                 "sandbox images; the plan pulls them")
            decls.append(sandbox_images_stage(images, runner_memory_gib=settings.run.runner_memory_gib))
    job, singleton = _warmer_job(settings, job_name)
    return build_plan(decls=decls, task=task, pipeline="provision", layout=layout, settings=settings,
                      code=code_state(checkout_root()), toolchain=toolchain_here(), job_name=job,
                      known_stage_names=KNOWN_STAGE_NAMES, known_tasks=known_task_ids(),
                      known_patterns=STAGE_NAME_PATTERNS, now=time.time(),
                      secrets=site_secrets(settings, echo, decls), partition="cpu", singleton=singleton)




def plan_images_build(*, config: Path, overrides: Mapping[str, str], task: str, job_name: str,
                      echo: Echo = print) -> Plan:
    """The first plan of a cache warmer that builds every image of *task*, as the image catalog
    names it, on the site's CPU partition when it has one, one at a time with the site's other
    warmers (``_warmer_job``); nothing is submitted. The plan and the warmer's layout name the task
    by the TaskSpec id of its module (``catalog.task_module``), as every other warmer and grading
    of it does. The login node reads only the task module, the lock files and the ``ENTRY.json``
    of each image's cache entry, whose presence sizes the stage's disk (``cache_holds``); the
    builds, copies and hashes happen in the allocation."""
    settings = planning_settings(config, settings_overrides(overrides))
    decls = images_stages(task=task, held=cache_holds(settings), recipe_key=substrate(settings.site).recipe_key)
    task_id = task_id_of(catalog.task_module(task))
    job, singleton = _warmer_job(settings, job_name)
    plan = build_plan(decls=decls, task=task_id, pipeline="images",
                      layout=InstanceLayout.warmer(task_id, new_rollout_id()), settings=settings,
                      code=code_state(checkout_root()), toolchain=toolchain_here(), job_name=job,
                      known_stage_names=KNOWN_STAGE_NAMES, known_tasks=known_task_ids(),
                      known_patterns=STAGE_NAME_PATTERNS, now=time.time(),
                      secrets=site_secrets(settings, echo, decls), partition="cpu", singleton=singleton)
    return plan




def _bundle_path(bundle: Path) -> Path:
    """*bundle* if it is an absolute regular file and not a symlink, else raise ValueError. Only
    its kind is looked at here; ``_copy_bundle`` reads it."""
    if not bundle.is_absolute():
        raise ValueError(f"--bundle {bundle} must be an absolute path")
    if bundle.is_symlink():
        raise ValueError(f"--bundle {bundle} is a symlink; name the bundle file itself")
    if not bundle.is_file():
        raise ValueError(f"--bundle {bundle} is not a regular file")
    return bundle


def _bundle_digest(bundle_sha256: str | None) -> str:
    """The digest the operator states for the bundle, if it is 64 lowercase hex, else raise
    ValueError. The copy ``_copy_bundle`` makes must have it."""
    if bundle_sha256 is None:
        raise ValueError("--bundle needs --sha256: the bundle's sha256 as 64 lowercase hex, which its copy "
                         "in the store must have")
    try:
        return check_sha256_hex(bundle_sha256)
    except ValueError as e:
        raise ValueError(f"--sha256: {e}") from None


@contextmanager
def _signals_interrupt() -> Iterator[None]:
    """While the block runs, SIGTERM and SIGHUP raise KeyboardInterrupt as SIGINT does, so what the
    block cleans up on its way out is cleaned up when the command is ended by any of them."""

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt(f"ended by signal {signal.Signals(signum).name}")

    previous = {signum: signal.signal(signum, interrupt) for signum in (signal.SIGTERM, signal.SIGHUP)}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _too_large(bundle: Path, size: int, cap_gib: int) -> str:
    return (f"--bundle {bundle} holds {size} bytes, more than the {cap_gib} GiB ({cap_gib * 2**30} bytes) the "
            "task's builds may write")


def _copy_bundle(bundle: Path, store: PosixStore, key: str, *, sha256: str, cap_gib: int) -> StoredBundle:
    """Copy the operator's *bundle* to *key* in *store*, hashing it as it is copied, and return
    what the plan records of the copy. The file is opened without following a link and read
    through that one descriptor. It is always copied, wherever it lies, the store among the
    places, so nothing done to the operator's file later reaches the instance. Raise ValueError
    for a file this process cannot open or that is not a regular file, one of more than *cap_gib*
    GiB (the writable disk cap of the task's builds, ``sandbox_disk_gib``), and one whose bytes do
    not have *sha256*. A refused copy and an interrupted one (``_signals_interrupt``) leave
    nothing at *key*, and the store's temporary file goes with them."""
    try:
        fd = os.open(bundle, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        raise ValueError(f"--bundle {bundle} cannot be read: {e.strerror}") from None
    with os.fdopen(fd, "rb") as source:
        st = os.fstat(source.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"--bundle {bundle} is not a regular file")
        if st.st_size > cap_gib * 2**30:
            raise ValueError(_too_large(bundle, st.st_size, cap_gib))
        digest, size = hashlib.sha256(), 0
        with _signals_interrupt(), store.open_write(key, exclusive=True) as copy:
            while chunk := source.read(CHUNK):
                size += len(chunk)
                if size > cap_gib * 2**30:
                    raise ValueError(_too_large(bundle, size, cap_gib))
                digest.update(chunk)
                copy.write(chunk)
            if digest.hexdigest() != sha256:
                raise ValueError(f"--bundle {bundle} has sha256 {digest.hexdigest()}, not the {sha256} --sha256 "
                                 "states")
    return StoredBundle(key=key, size=size)


def produced_artifact(task_module: str) -> AgentProducedArtifact | None:
    """The ``ARTIFACT`` the imported module *task_module* declares when an agent produces it;
    None when the module declares no artifact or another kind."""
    spec = getattr(sys.modules[task_module], "ARTIFACT", None)
    return spec if isinstance(spec, AgentProducedArtifact) else None


def evaluation_allocation(produced: AgentProducedArtifact | None) -> EvaluationAllocation | None:
    """The allocation a drive of the task whose agent-produced artifact is *produced* takes, as
    its producer declares it; None when it declares none or the task has no such artifact, so
    the drive takes the hardware the brief states."""
    return produced.producer.evaluation_allocation if produced is not None else None


def _producer(module: ModuleType, task_module: str) -> AgentProducerSpec:
    """The producer of *task_module*'s ``ARTIFACT``, *module* being the module: an agent-produced
    artifact, whose submission is a git repository, the one kind ``hls package`` and ``hls run``
    take. Raise ValueError for any other artifact."""
    artifact = getattr(module, "ARTIFACT", None)
    if not isinstance(artifact, AgentProducedArtifact):
        raise ValueError(f"{task_module}'s ARTIFACT is {type(artifact).__name__ if artifact is not None else None}, "
                         "not an agent-produced artifact; only a submission an agent built is packaged")
    return artifact.producer


def plan_package(*, config: Path, overrides: Mapping[str, str], task_module: str, bundle: Path,
                 bundle_sha256: str | None, job_name: str, echo: Echo = print) -> Plan:
    """The first plan of an instance that builds the source *bundle* into a submission image
    for *task_module*'s task, on the task's submission base and under its producer's network
    policy (the seal the pip steps run behind, and the package rule), on the site's CPU
    partition when it has one; nothing is submitted. *bundle_sha256* is the bundle's sha256 as its
    operator states it; None, or anything but 64 lowercase hex, is refused before anything is
    planned. Before planning, the bundle is copied into the new instance in the store
    (``InstanceLayout.source_bundle``, ``_copy_bundle``), which refuses a bundle past the writable
    disk cap of the task's builds (``sandbox_disk_gib``) or without the stated digest; the plan
    records the copy's key and size and the digest, and its stages read the copy, on every
    substrate. A plan that is refused after the copy removes it. It runs one at a time with the
    site's other warmers (``_warmer_job``). The login node imports the task module for its id, its
    producer and its GPU kind, and reads the ``ENTRY.json`` of each image's cache entry
    (``cache_holds``). The catalog knows the task by its directory (``catalog.catalog_task``), the
    plan and the store by its id."""
    bundle = _bundle_path(bundle)
    digest = _bundle_digest(bundle_sha256)
    settings = planning_settings(config, settings_overrides(overrides))
    catalog_key = catalog.catalog_task(task_module)
    task_id = task_id_of(task_module)
    module = sys.modules[task_module]
    producer = _producer(module, task_module)
    cap_gib = sandbox_disk_gib(module.TASK, producer)
    layout = InstanceLayout.warmer(task_id, new_rollout_id())
    store = PosixStore(settings.site.store_root)
    stored = _copy_bundle(bundle, store, layout.source_bundle, sha256=digest, cap_gib=cap_gib)
    try:
        decls = package_stages(task=task_id, catalog_key=catalog_key, bundle=stored, bundle_sha256=digest,
                               network=producer_network(producer), gpu_kind=submission_gpu_kind(module.TASK),
                               held=cache_holds(settings), recipe_key=substrate(settings.site).recipe_key,
                               disk_cap_gib=cap_gib, runner_memory_gib=settings.run.runner_memory_gib,
                               seal_memory_gib=seal_memory_gib(settings.site),
                               server_filename=producer.server_rel_path)
        job, singleton = _warmer_job(settings, job_name)
        return build_plan(decls=decls, task=task_id, pipeline="package", layout=layout, settings=settings,
                          code=code_state(checkout_root()), toolchain=toolchain_here(), job_name=job,
                          known_stage_names=KNOWN_STAGE_NAMES, known_tasks=known_task_ids(),
                          known_patterns=STAGE_NAME_PATTERNS, now=time.time(),
                          secrets=site_secrets(settings, echo, decls), partition="cpu", singleton=singleton)
    except BaseException:
        store.delete(layout.source_bundle)
        raise


def _package_manifest(store: PosixStore, task: str, instance: str) -> tuple[str, str]:
    """The key and digest of the published package manifest of *instance*, which must be a
    cache warmer of task *task* (``warmers/<task>/<id>``, as ``hls package`` makes it). Only the
    manifest is read; the drive checks the package record and the image it names."""
    try:
        key = InstanceLayout.parse(instance)
    except NotAnInstance as e:
        raise ValueError(f"--package {instance} is not a cache warmer of {task}: {e.why}") from None
    if (key.area, key.task, key.grading_id) != (WARMERS, task, None):
        raise ValueError(f"--package {instance} is not a cache warmer of {task}; name the {WARMERS}/{task}/<id> "
                         "instance hls package created")
    layout = InstanceLayout(instance)
    try:
        found = _published_manifest(store, layout, "package")
    except ValueError as e:
        raise ValueError(f"--package {instance}: {e}") from None
    if found is None or "package" not in found[0].outputs:
        raise ValueError(f"--package {instance} has no published package stage")
    return layout.manifest("package"), found[1]


def measured_artifact(task: Any, role: str, catalog_key: str) -> tuple[ImageArtifact, list[str]]:
    """*task*'s measured image of *role* as an artifact, the catalog knowing the task as
    *catalog_key*: its name in the images stage's ``images.json`` and the entrypoint and
    runtime variables the task module declares for it (``catalog.Served``); and the serve
    command the task's baseline declares. Raise ValueError for a task that declares no image of
    *role*, a role that is not the baseline (the one measured image a task serves), or a task
    whose spec declares no baseline, whose serve command is then unknown; the refusal names no
    flag, so each caller says which of its own named the role."""
    images = catalog.task_images(catalog_key)
    if role not in images.measured:
        raise ValueError(f"{catalog_key} declares no {role} image; it declares {list(images.measured)}")
    if role != "baseline" or images.baseline is None:
        raise ValueError(f"only the baseline is served as an artifact; {role} is not a server {task.id} declares")
    if task.baseline is None:
        raise ValueError(f"{task.id} declares no baseline (TaskSpec.baseline), so the serve command its measured "
                         "baseline image is served with is unknown")
    served = images.baseline
    artifact = ImageArtifact(image=images.measured[role], entrypoint=list(served.entrypoint),
                             runtime_env=dict(served.runtime_env))
    return artifact, list(task.baseline.serve_command)


def held_reference_sha256(store: PosixStore, images: catalog.TaskImages, *, recipe_key: RecipeKey,
                          image_file: str) -> str | None:
    """The sha256 of the file the cache of *store* holds for the reference image a task declares,
    which its gradings verify on, *images* being the task's images, each keyed by *recipe_key* and
    kept in its entry as *image_file*, as the site's substrate does (``Substrate.recipe_key``,
    ``Substrate.image_file``); None when it declares none (the harness image is its reference) or
    the cache holds no entry of it yet. Only the entry's ``ENTRY.json`` is read."""
    name = images.measured.get("reference")
    if name is None:
        return None
    key = image_keys(images_chain(images, [name]), recipe_key)[name]
    try:
        entry = ContentCache(store).lookup("images", key)
    except CacheCorrupt as e:
        raise ValueError(f"cache entry images/{key}: {e}") from e
    if entry is None:
        return None
    return next((f.sha256 for f in entry.manifest.files if f.path == image_file), None)


def checked_for_grading(store: PosixStore, *, task_ref: TaskRef, task_id: str, task: Any,
                        catalog_key: str, site: SiteSettings) -> list[str]:
    """What a plan that grades *task*, the task *task_ref* names with its overrides applied,
    reports of the oracle answers its verify will read, read for the
    task's reference image in the catalog: raise PlanError when its caching references' answers
    are not all in the store for the GPU kind and the measured reference image it verifies on,
    naming the fill with these overrides that would answer them; else, once the answers are all
    there, a line naming the prompts and their rows, the references' fingerprints, the answers
    files and the instances that wrote them (the fill's among them), or a warning when the task's
    reference image is the harness image or one the cache holds no file of yet, whose file the
    images stage makes. The answers are keyed by the sha256 of the reference image's file the
    cache holds (``held_reference_sha256``)."""
    images = catalog.task_images(catalog_key)
    on = substrate(site)
    reference = held_reference_sha256(store, images, recipe_key=on.recipe_key, image_file=on.image_file)
    gpu_kind = reference_gpu_kind(task) or NO_GPU
    coverage = check_oracle_coverage(store, task_ref=task_ref, task_id=task_id, task=task, gpu_kind=gpu_kind,
                                     reference_sha256=reference)
    if coverage is not None:
        return [f"oracle answers: {coverage.prompts} prompts of rows {coverage.rows} answered for "
                f"references {', '.join(coverage.fingerprints)} in {coverage.files} answers files written by "
                f"{', '.join(coverage.writers)}"]
    if reference is None and caching_references(task):
        if "reference" not in images.measured:
            return [f"warning: {task_id} verifies on a built image, so whether its reference answers are all "
                    "in the store is known only once the images stage made it; run hls oracle fill first"]
        return [f"warning: the cache holds no file of {task_id}'s reference image yet, so whether its reference "
                "answers are all in the store is known only once the images stage made it; run hls oracle fill "
                "first"]
    return []


@dataclass(frozen=True)
class RolloutSource:
    """What a grading of a rollout reads from the rollout."""

    #: The rollout's instance, ``runs/<task>/<id>``.
    instance: str
    #: The task reference the rollout ran: its module and the task overrides it ran under.
    task_ref: TaskRef
    #: The arm whose agent built the rollout's package.
    arm: ArmRecord
    #: The key and digest of the rollout's published package manifest.
    package: tuple[str, str]
    #: The key and digest of the rollout's published agent_build manifest.
    build: tuple[str, str]
    #: For a rebuild, the limits the build's brief stated for each pip step of its package
    #: (``AgentBuildParams.package_pip``), which the package stage that packages its bundle again
    #: holds the steps to; None for a grading of the published package, which runs no pip step.
    package_pip: PipLimits | None


def rollout_source(store: Store, task_id: str, task_module: str, rollout: str, *, rebuild: bool) -> RolloutSource:
    """What a grading of the rollout *rollout* of task *task_id* reads: its latest plan's
    agent_build parameters, the pip limits among them for a *rebuild*, and its published
    agent_build and package manifests. Only the store is read. Raise ValueError for an instance
    that is not a rollout of the task, a rollout the store does not hold, one that built nothing
    or built for another module than *task_module*, one whose agent_build or package stage has
    not published, and for a rebuild of one whose plan states no pip limits (``_StatedPip``)."""
    try:
        key = InstanceLayout.parse(rollout)
    except NotAnInstance as e:
        raise ValueError(f"--rollout {rollout} is not a rollout of {task_id}: {e.why}") from None
    if (key.area, key.task, key.grading_id) != (ROLLOUTS, task_id, None):
        raise ValueError(f"--rollout {rollout} is not a rollout of {task_id}; name the {ROLLOUTS}/{task_id}/<id> "
                         "instance hls run created")
    layout = InstanceLayout(rollout)
    try:
        planned = _planned_build(store, layout)
        if planned is None:
            raise ValueError("the store holds no such rollout")
        key, _, stage = planned
        if stage is None:
            raise ValueError(f"it built nothing: its plan has no {AGENT_BUILD_STAGE} stage")
        try:
            built = _BuiltBy.model_validate(stage.get("params"))
        except ValidationError as e:
            raise ValueError(f"{key} names no task and arm record: {validation_errors(e)}") from None
        if built.task.module != task_module:
            raise ValueError(f"it was built for {built.task.module}, not {task_module}")
        pip = None
        if rebuild:
            try:
                pip = _StatedPip.model_validate(stage.get("params")).package_pip
            except ValidationError as e:
                raise ValueError(f"{key} states no pip limits its build's brief gave the package, which a rebuild "
                                 f"holds its pip steps to: {validation_errors(e)}; grade its published package "
                                 "without --artifact.rebuild") from None
        found: dict[str, tuple[str, str]] = {}
        for name, output in ((AGENT_BUILD_STAGE, BUILD_OUTPUT), ("package", "package")):
            manifest = _published_manifest(store, layout, name)
            if manifest is None or output not in manifest[0].outputs:
                raise ValueError(f"it has no published {name} stage")
            found[name] = layout.manifest(name), manifest[1]
    except ValueError as e:
        raise ValueError(f"--rollout {rollout}: {e}") from None
    return RolloutSource(instance=layout.root, task_ref=built.task, arm=built.arm, package=found["package"],
                         build=found[AGENT_BUILD_STAGE], package_pip=pip)


class _BuiltBy(BaseModel):
    """What a reader of another instance's agent_build stage parameters needs of them: the task
    it built for and the arm it ran. Every other parameter is left unread, so a plan older or newer
    code wrote still names them; a task or an arm that does not parse as today's is refused."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    #: The task the build was for: its module and the task overrides it ran under.
    task: TaskRef
    #: The arm whose agent built.
    arm: ArmRecord


class _StatedPip(BaseModel):
    """What a rebuild reads of another instance's agent_build stage parameters beside
    ``_BuiltBy``: the limits the build's brief stated for each pip step of its package, which the
    grading's package stage holds the steps to. Every other parameter is left unread. A plan
    that states none is refused, since its bundle cannot be packaged again under the limits its
    brief stated."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    #: The limits as the build's plan stated them (``AgentBuildParams.package_pip``).
    package_pip: PipLimits


class _BuiltArm(BaseModel):
    """What the rerun rule reads of an earlier build's ``build.json``: the arm it ran. Every other
    field is left unread, so a record older or newer code wrote, with fields this code does not
    know or lacks, still names its arm; an arm that does not parse as today's is refused, since
    a change of the arm's schema is a change of the arm."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    #: The arm the build ran.
    arm: ArmRecord


def _planned_build(store: Store, layout: InstanceLayout) -> tuple[str, dict[str, Any], dict[str, Any] | None] | None:
    """The latest plan revision of *layout*'s instance read as JSON alone, for the few fields a
    reader of another instance's plan needs: its key, the plan, and its agent_build stage (None for
    a plan with none). None when no plan was written. The plan is not validated as a ``Plan``:
    older or newer code may have written the rest of it otherwise. Raise ValueError naming the
    file for one that is not a JSON object with a list of stages."""
    revisions = PlanStore(store, layout).revisions()
    if not revisions:
        return None
    key = layout.plan_revision(revisions[-1])
    try:
        plan = json.loads(store.read_bytes(key))
    except ValueError:
        raise ValueError(f"{key} is not a plan: it is not JSON") from None
    stages = plan.get("stages") if isinstance(plan, dict) else None
    if not isinstance(stages, list) or not all(isinstance(s, dict) for s in stages):
        raise ValueError(f"{key} is not a plan: it holds no list of stages")
    return key, plan, next((s for s in stages if s.get("kind") == AgentBuildStage.kind), None)


def _published_manifest(store: Store, layout: InstanceLayout, stage: str) -> tuple[Manifest, str] | None:
    """The published manifest of *stage* of *layout*'s instance and its digest (``read_manifest``);
    None when it is not published. Raise ValueError naming the file for one that does not parse,
    by its errors' types and places."""
    try:
        return read_manifest(store, layout, stage)
    except ValidationError as e:
        raise ValueError(f"{layout.manifest(stage)} is not a manifest: {validation_errors(e)}") from None


#: A published image's digest as ``--artifact.image`` names it.
IMAGE_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


def _published(image: str, image_file: Path) -> PublishedArtifact:
    """The published image ``--artifact.image`` *image* names, read from *image_file*. Raise
    ValueError for a name that is not ``sha256:`` and 64 hex digits, and for a file this login
    node does not see; its bytes are checked when the drive stage reads them, not here."""
    named = IMAGE_DIGEST.fullmatch(image)
    if named is None:
        raise ValueError(f"--artifact.image is sha256:<64 hex digits>, the digest of the image file, not {image!r}")
    path = image_file.resolve()
    if not path.is_file():
        raise ValueError(f"--from {image_file}: no such file on this cluster's shared filesystem")
    return PublishedArtifact(sha256=named.group(1), source=str(path))


def plan_grade(*, config: Path, overrides: Mapping[str, str], task_module: str, rollout: str | None = None,
               package: str | None = None, measured: str | None = None, rebuild: bool = False,
               image: str | None = None, image_file: Path | None = None, built_arm: str | None = None,
               job_name: str, echo: Echo = print) -> Plan:
    """The first plan of ``hls grade``: a grading of one artifact of *task_module*'s task, the
    published package of the rollout *rollout* (under it, its task overrides replayed beneath the
    command's own, the arm that built it recorded, so finalize counts that arm's route secrets
    contestant-facing), the published package of the warmer *package*, the task's measured image
    of role *measured* (both under the warmers), or the published submission image *image*
    (``sha256:<digest>``) read from *image_file*, under ``warmers/<task>/image-<digest>``, with
    the arm *built_arm* recorded as the one that built it when it is given. Exactly one is given.
    With *rebuild* (a rollout's grading only) the grading scans and packages the rollout's source
    bundle again at this checkout, pinned by its agent_build manifest's digest, and serves that.
    Its stages run in one allocation on the site's GPU partition (``grading_stages``), and the
    artifact and its driver take the task's evaluation allocation when its module declares one.
    Before anything is planned it is checked: a committed checkout, a task with data, the
    credit and calibration that fit the task (``find_measured``), and the oracle answers its
    verify reads; once planned, the measured files and the answers are printed through *echo*.
    *overrides* are ``--site.*``, ``--run.*``, ``--resources.*`` and ``--task.*``. The login node
    imports the task module and reads the catalog, the settings, the store's manifests and image
    entries' ``ENTRY.json`` (``cache_holds``), the measured files and the secrets file, whose
    values the plan may not hold; it never downloads, builds or loads a model. Raise
    ArmError for a *built_arm* that cannot be resolved, and ValueError (a PlanError among them)
    for what cannot be planned, a task that declares no drive bounds and a score stage's memory
    below its role's (``check_score_memory``) among them."""
    named = [n for n, v in (("rollout", rollout), ("package", package), ("measured", measured), ("image", image))
             if v is not None]
    if len(named) != 1:
        raise ValueError("name exactly one artifact to grade: --rollout INSTANCE, --package INSTANCE, --measured "
                         "ROLE or --artifact.image sha256:<digest>")
    if rebuild and rollout is None:
        raise ValueError("--artifact.rebuild packages a rollout's source bundle again: it needs --rollout")
    if image is not None and image_file is None:
        raise ValueError("--artifact.image grades a file by its digest: name the file with --from PATH")
    if image is None and image_file is not None:
        raise ValueError("--from PATH names the file of --artifact.image, which is not given")
    if image is None and built_arm is not None:
        raise ValueError("--built-arm names the arm that built a published image: it goes with --artifact.image")
    published = _published(image, image_file) if image is not None and image_file is not None else None
    settings_over, task_over = split_overrides(overrides)
    settings = planning_settings(config, settings_over)
    catalog_key = catalog.catalog_task(task_module)
    task_id = task_id_of(task_module)
    module = sys.modules[task_module]
    produced = produced_artifact(task_module)
    code = code_state(checkout_root())
    check_committed_tree(code)
    store = PosixStore(settings.site.store_root)
    built = rollout_source(store, task_id, task_module, rollout, rebuild=rebuild) if rollout is not None else None
    task_ref = TaskRef(module=task_module, overrides={**(built.task_ref.overrides if built else {}), **task_over})
    task = task_to_plan(task_ref)
    try:
        drive_bounds(task)
    except UnknownTask as e:
        raise PlanError(f"{task_id} cannot be graded: {e.args[0]}") from None
    check_task_has_data(task_id, task)
    serve = list(produced.serve_command) if produced is not None else []
    rebuild_from = rebuild_digest = producer = stated_pip = None
    arm = built.arm if built is not None else None
    artifact: PackagedArtifact | ImageArtifact | PublishedArtifact
    if built is not None:
        parent = InstanceLayout(built.instance)
        if rebuild:
            producer = _producer(module, task_module)
            artifact, digest = PackagedArtifact(), None
            rebuild_from, rebuild_digest = built.build
            stated_pip = built.package_pip
        else:
            artifact, digest = PackagedArtifact(manifest=built.package[0]), built.package[1]
    elif package is not None:
        key, digest = _package_manifest(store, task_id, package)
        artifact, parent = PackagedArtifact(manifest=key), InstanceLayout(package)
    elif published is not None:
        artifact, digest = published, None
        parent = InstanceLayout.warmer(task_id, f"image-{artifact.sha256}")
        if built_arm is not None:
            effort = produced.producer.reasoning_effort if produced is not None else None
            arm = resolve_arm(settings, built_arm, task_effort=effort)
    else:
        assert measured is not None
        try:
            artifact, serve = measured_artifact(task, measured, catalog_key)
        except ValueError as e:
            raise ValueError(f"--measured {measured}: {e}") from None
        digest, parent = None, InstanceLayout.warmer(task_id, f"measured-{measured}")
    measured, notes = pinned_measured(store, task_ref=task_ref, task=task, task_id=task_id)
    check_defaults_committed(checkout_root(), measured)
    notes += checked_for_grading(store, task_ref=task_ref, task_id=task_id, task=task, catalog_key=catalog_key,
                                 site=settings.site)
    grading = new_grading_id()
    registry = read_secrets(settings.site.secrets_path)
    decls = grading_stages(task_ref=task_ref, task=task, catalog_key=catalog_key, artifact=artifact,
                           serve_command=serve, package_digest=digest, eval_id=grading,
                           allocation=evaluation_allocation(produced), run=settings.run,
                           seal_memory_gib=seal_memory_gib(settings.site), rebuild_from=rebuild_from,
                           rebuild_digest=rebuild_digest, producer=producer, stated_pip=stated_pip, built_arm=arm,
                           held=cache_holds(settings), recipe_key=substrate(settings.site).recipe_key,
                           provisioned=planned_provision(settings, task, task_ref, config=config, overrides=overrides),
                           dev_files=planned_dev_files(settings, measured))
    plan = build_plan(decls=decls, task=task_id, pipeline="grading", layout=parent.grading(grading),
                      settings=settings, code=code, toolchain=toolchain_here(), job_name=job_name,
                      known_stage_names=KNOWN_STAGE_NAMES, known_tasks=known_task_ids(),
                      known_patterns=STAGE_NAME_PATTERNS, now=time.time(),
                      secrets=_checked_secrets(settings, echo, decls, registry), measured=measured)
    check_score_memory(plan, task)
    for line in notes:
        echo(line)
    return plan




def plan_oracle_fill(*, config: Path, overrides: Mapping[str, str], task_module: str, chunk: int, job_name: str,
                     echo: Echo = print) -> Plan:
    """The first plan of a cache warmer that fills *task_module*'s reference answers to its
    test split into the store, *chunk* rows at a time; nothing is submitted. *overrides* hold the
    task's (``--task.*``) beside the settings': the fill provisions and asks about the task they
    make, so a grading with the same overrides finds its answers. It runs on the
    site's GPU partition under the default max policy, never on the CPU partition: the answers
    are keyed by the GPU kind and the image the task's gradings verify on, so it runs on the
    verify role's shape and image. The login node imports the task module for its id, its
    verifiers and its hardware, builds the caching references' engines to size and check the
    role, reads the catalog (by the task's directory, ``catalog.catalog_task``), the ``ENTRY.json``
    of each image's cache entry (``cache_holds``) and the secrets file's grant names, and loads no
    model. The instance and the answers belong to the task's
    id. When the task declares a dev reference and neither the store nor the task's defaults hold
    one that fits (``hls.measured.find``), the fill produces it too, and one line says so. Raise
    ValueError for what cannot be planned, a PlanError for a task with no reference that caches its
    corpus answers among it."""
    settings_over, task_over = split_overrides(overrides)
    settings = planning_settings(config, settings_over)
    catalog_key = catalog.catalog_task(task_module)
    task_id = task_id_of(task_module)
    task_ref = TaskRef(module=task_module, overrides=task_over)
    task = task_to_plan(task_ref)
    provisioned = planned_provision(settings, task, task_ref, config=config, overrides=overrides)
    store = PosixStore(settings.site.store_root)
    dev = None
    if DEV_REFERENCE_FILE in needed_files(task):
        dev = find(DEV_REFERENCE_FILE, task_ref=task_ref, task=task, task_id=task_id, checkout=checkout_root(),
                   store=store)
    decls = oracle_fill_stages(task_ref=task_ref, task=task, catalog_key=catalog_key,
                               runner_memory_gib=settings.run.runner_memory_gib, held=cache_holds(settings),
                               recipe_key=substrate(settings.site).recipe_key,
                               provisioned=provisioned, chunk=chunk, dev_reference=dev is not None and dev.pin is None)
    if dev is not None:
        echo(dev.said() if dev.pin is not None else f"{dev.said()}: the fill produces it into the store")
    return build_plan(decls=decls, task=task_id, pipeline="oracle_fill",
                      layout=InstanceLayout.warmer(task_id, new_rollout_id()), settings=settings,
                      code=code_state(checkout_root()), toolchain=toolchain_here(), job_name=job_name,
                      known_stage_names=KNOWN_STAGE_NAMES, known_tasks=known_task_ids(),
                      known_patterns=STAGE_NAME_PATTERNS, now=time.time(),
                      secrets=site_secrets(settings, echo, decls))




def earlier_arm(store: Store, task: str, arm: str) -> tuple[str, ArmRecord] | None:
    """The latest earlier build of arm *arm* among task *task*'s rollouts, and the arm it ran.
    The candidates are the rollouts whose index rows (``runs/<task>/index/<rollout>.json``, which
    finalize writes, and the orchestrator for a failed instance) name the arm, and the rollouts
    with no row whose latest plan has not ended and whose agent_build stage, not a stop build's,
    names the arm: builds still running, which count as much as finished ones, so two builds at
    once under one arm name cannot differ unrecorded. The latest is the one whose plan was created
    last (a row records its plan's creation). Nothing typed is read before it is chosen, and of a
    running build's plan only its outcome, its agent_build stage's stop and its arm's name, so no
    record or plan of another arm's build, which older or other code may have written, can block
    this one. A probe or preflight-only build has no finalize and a void build no records, so
    neither has a row, and neither is a build: one runs no agent and builds nothing, the other is
    relaunched and was never the contestant's result. The arm the latest ran is the one its
    published ``build.json`` records, of which only the arm is read, after the file is checked
    against its manifest; for one that publishes none, a failed build (the contestant's zero) or
    a running one, the one its plan's agent_build stage names. None when no build names the arm.
    Only the store is read. Raise ValueError naming the file for a row, or a running plan, that
    does not say which arm it is or when its plan was created, since it may be this arm's latest;
    for the latest's agent_build manifest that does not parse, its ``build.json`` that is not the
    file its manifest recorded or names no arm record, and its plan that names none."""
    index = task_index(ROLLOUTS, task)
    rows = _index_rows(store, index)
    candidates: list[tuple[float, str]] = []
    for rollout, (key, row) in rows.items():
        if row["arm"] == arm:
            candidates.append((_created_at(row, key, arm), rollout))
    for rollout in store.list_children(f"{ROLLOUTS}/{task}"):
        if rollout == INDEX_DIR or rollout in rows:
            continue
        try:
            layout = InstanceLayout.rollout(task, rollout)
        except ValueError:
            continue
        planned = _planned_build(store, layout)
        if planned is None:
            continue
        key, plan, stage = planned
        if plan.get("outcome") is not None or stage is None:
            continue
        params = stage.get("params")
        if isinstance(params, dict) and params.get("stop_after") is not None:
            continue
        named = params.get("arm") if isinstance(params, dict) else None
        if not isinstance(named, dict) or not isinstance(named.get("name"), str):
            raise ValueError(f"{key} is a running plan whose {AGENT_BUILD_STAGE} stage names no arm, so it may be "
                             f"{arm}'s")
        if named["name"] == arm:
            candidates.append((_created_at(plan, key, arm), rollout))
    if not candidates:
        return None
    _, rollout = max(candidates)
    try:
        layout = InstanceLayout.rollout(task, rollout)
    except ValueError as e:
        raise ValueError(f"{index}/{rollout}.json is a row of arm {arm} that names no rollout: {e}") from None
    found = _published_manifest(store, layout, AGENT_BUILD_STAGE)
    if found is not None and BUILD_OUTPUT in found[0].outputs:
        return layout.root, _built_arm(store, found[0].outputs[BUILD_OUTPUT])
    return layout.root, _planned_arm(store, layout)


def _index_rows(store: Store, index: str) -> dict[str, tuple[str, dict[str, Any]]]:
    """Each rollout's index row in the index directory *index*, by rollout id, with its key; a
    grading's row, ``<rollout>+<grading>.json``, names its rollout's build and is left out. Raise
    ValueError naming the row for one that is not a JSON object naming an arm, since it may be the
    latest build of any arm."""
    rows: dict[str, tuple[str, dict[str, Any]]] = {}
    for name in store.list_children(index):
        if "+" in name:
            continue
        key = f"{index}/{name}"
        try:
            row = json.loads(store.read_bytes(key))
        except ValueError:
            raise ValueError(f"{key} is not an index row: it is not JSON") from None
        if not isinstance(row, dict) or "arm" not in row:
            raise ValueError(f"{key} is not an index row: it names no arm, so it may be any arm's")
        rows[name.removesuffix(".json")] = (key, row)
    return rows


def _created_at(record: dict[str, Any], key: str, arm: str) -> float:
    """When the plan of a build of arm *arm* was created, as its row or its plan, *record* at
    *key*, says. Raise ValueError naming the file for one that does not say it."""
    created = record.get("created_at")
    if isinstance(created, bool) or not isinstance(created, (int, float)):
        raise ValueError(f"{key} is a build of arm {arm} that does not say when its plan was created")
    return float(created)


def _planned_arm(store: Store, layout: InstanceLayout) -> ArmRecord:
    """The arm the agent_build stage of *layout*'s latest plan revision names, read from the plan's
    JSON alone (``_planned_build``). Raise ValueError naming the plan for one that names no arm
    record."""
    planned = _planned_build(store, layout)
    if planned is None:
        raise ValueError(f"{layout.root} has an index row and no plan")
    key, _, stage = planned
    if stage is None:
        raise ValueError(f"{key} has no {AGENT_BUILD_STAGE} stage")
    try:
        return _BuiltArm.model_validate(stage.get("params")).arm
    except ValidationError as e:
        raise ValueError(f"{key}: its {AGENT_BUILD_STAGE} stage names no arm record: {validation_errors(e)}") from None


def _built_arm(store: Store, ref: OutputRef) -> ArmRecord:
    """The arm the ``build.json`` that output *ref* holds records, the one field of it read
    (``_BuiltArm``). Raise ValueError for a file that is not the one its manifest recorded, or
    whose arm does not parse."""
    data = store.read_bytes(ref.key)
    if sha256_bytes(data) != ref.sha256:
        raise ValueError(f"{ref.key} does not match its manifest: {sha256_bytes(data)} is not {ref.sha256}")
    try:
        return _BuiltArm.model_validate_json(data).arm
    except ValidationError as e:
        raise ValueError(f"{ref.key} names no arm record: {validation_errors(e)}") from None


def check_rerun_arm(store: Store, task: str, arm: ArmRecord, arm_change: str | None) -> None:
    """Refuse a build of *task* whose resolved *arm* differs from the arm of the task's latest
    earlier build of that arm name, unless *arm_change* gives the reason; refuse a reason
    that is blank or names no difference, since the plan and ``build.json`` record it as one.
    Raise ArmError."""
    if arm_change is not None and not arm_change.strip():
        raise ArmError(f"arm {arm.name}: --arm-change needs the reason the arm differs")
    earlier = earlier_arm(store, task, arm.name)
    if earlier is None:
        if arm_change is not None:
            raise ArmError(f"arm {arm.name}: --arm-change was given, but {task} has no earlier build of this arm for "
                           "it to differ from")
        return
    instance, old = earlier
    differences = arm_differences(old, arm)
    if differences and arm_change is None:
        raise ArmError(f"arm {arm.name} differs from its latest build of {task}, {instance}: "
                       f"{'; '.join(differences)}. A rerun runs the arm it ran; give --arm-change REASON to declare "
                       "the change")
    if not differences and arm_change is not None:
        raise ArmError(f"arm {arm.name}: --arm-change was given, but the arm is the one {instance} ran")


def run_checks(*, settings: Settings, store: PosixStore, code: Any, task_module: str, task: Any, arm: str,
               arm_change: str | None, build_only: bool, stop_after: Literal["probe", "preflight"] | None
               ) -> tuple[ArmRecord, list[str]]:
    """The checks of ``hls run`` that need nothing measured, which ``hls.pipeline.prepare`` makes
    before it measures anything for the run: the arm *arm* resolves against the producer's effort
    and a rerun whose arm differs from the task's latest build of it declares why
    (``check_rerun_arm``); every run's upload folder and skills folder pass the rules the build
    applies (``check_upload_folder``, ``check_skills_folder``); and a scored run, every run but a
    stop build, is of a task with data, from a committed checkout *code*, and, unless
    *build_only*, with drive bounds. Return the resolved arm and the lines to print
    for the folders. Raise ArmError and ValueError (a PlanError among them) as ``plan_run`` says."""
    task_id = task_id_of(task_module)
    producer = _producer(sys.modules[task_module], task_module)
    record = resolve_arm(settings, arm, task_effort=producer.reasoning_effort)
    check_rerun_arm(store, task_id, record, arm_change)
    notes: list[str] = []
    # Every build copies the uploaded folder and the skills into its sandbox, a stop build's
    # too, and the agent_build stage refuses what these checks refuse.
    uploaded = check_upload_folder(checkout_root(), task_module, producer)
    if uploaded is not None:
        notes.append(f"uploads {', '.join(uploaded.paths)} {uploaded.sha256}")
    skills = check_skills_folder(checkout_root(), task_module, producer)
    if skills:
        notes.append(f"skills {', '.join(skills)}")
    if stop_after is None:
        check_task_has_data(task_id, task)
        check_committed_tree(code)
        if not build_only:
            try:
                drive_bounds(task)
            except UnknownTask as e:
                raise PlanError(f"{task_id} cannot be graded: {e.args[0]}") from None
    return record, notes


def plan_run(*, config: Path, overrides: Mapping[str, str], task_module: str, arm: str, arm_change: str | None,
             job_name: str, build_only: bool = False, soft_sec: int | None = None, hard_sec: int | None = None,
             stop_after: Literal["probe", "preflight"] | None = None, probe_oom: bool = False,
             echo: Echo = print) -> Plan:
    """The first plan of ``hls run``: a rollout, ``runs/<task>/<id>``, that builds a submission
    for *task_module*'s task with the agent of arm *arm* (``[arms.<arm>]`` in hls.toml), packages
    it and grades it, in one allocation on the site's GPU partition (``rollout_stages``), sized to
    the task's build allocation when it declares one; the plan warns how many GPUs idle during the
    stages that use fewer. With *build_only* it ends with the package and a record of the
    rollout; a probe or preflight-only build (*stop_after*, with *probe_oom*) stops at the sandbox
    under its own short budget and is not scored. *soft_sec* and *hard_sec*, given together,
    replace the task's wall budgets for this run. The arm resolves against the producer's effort,
    and a rerun whose arm differs from the task's latest build of it is refused unless
    *arm_change* declares why (``check_rerun_arm``). Before anything is planned every run's
    upload folder is checked against the tree digest the task pins and where it lands
    (``check_upload_folder``), and its skills folder against the skill rule
    (``check_skills_folder``), and a scored run, every run but a stop build, is checked too: a
    committed checkout, a task with data, the credit and calibration that fit the task
    (``find_measured``), and for a run that grades, the oracle answers its verify reads. Once
    planned, the arm, the upload folder, the skills, the measured files and the oracle answers are
    printed through *echo*.
    *overrides* are ``--site.*``, ``--run.*``, ``--resources.*`` and ``--task.*``. The login node
    imports the task module and reads the catalog, the settings, the image entries' ``ENTRY.json``
    (``cache_holds``), the cache's provision record of the task, which sizes the copies of its
    data (``planned_provision``), the task's earlier builds, the upload and skills folders' trees,
    the measured files and the secrets file's grant names; it builds, downloads or copies nothing.
    Raise ArmError for an arm that cannot be resolved or a rerun the arm check refuses, and
    ValueError (a PlanError among them) for anything else that cannot be planned, a run but a stop
    build of a task that declares no drive bounds, a cache that
    lacks the task's data (``planned_provision``, which names the ``hls provision`` to run first),
    a collection allowance
    (``run.collect_allowance_sec``) under what a build's collection needs
    (``AgentBuildStage.check``) and a score stage's memory below its role's
    (``check_score_memory``) among them."""
    if stop_after is not None and (soft_sec is not None or hard_sec is not None):
        raise ValueError("a probe or preflight-only build runs its own short budget: --soft-sec and --hard-sec are "
                         "refused with --probe-only or --preflight-only")
    if (soft_sec is None) != (hard_sec is None):
        raise ValueError("give --soft-sec and --hard-sec together, or neither")
    if soft_sec is not None and hard_sec is not None and soft_sec > hard_sec:
        raise ValueError(f"--soft-sec {soft_sec} is over --hard-sec {hard_sec}")
    settings_over, task_over = split_overrides(overrides)
    settings = planning_settings(config, settings_over)
    catalog_key = catalog.catalog_task(task_module)
    task_id = task_id_of(task_module)
    module = sys.modules[task_module]
    producer = _producer(module, task_module)
    task_ref = TaskRef(module=task_module, overrides=task_over)
    task = task_to_plan(task_ref)
    store = PosixStore(settings.site.store_root)
    code = code_state(checkout_root())
    record, notes = run_checks(settings=settings, store=store, code=code, task_module=task_module, task=task, arm=arm,
                               arm_change=arm_change, build_only=build_only, stop_after=stop_after)
    # A stop build renders its brief with what the store or the task's defaults hold, and measures
    # nothing.
    measured, found = pinned_measured(store, task_ref=task_ref, task=task, task_id=task_id,
                                      scored=stop_after is None)
    notes += found
    if stop_after is None:
        check_defaults_committed(checkout_root(), measured)
        if not build_only:
            notes += checked_for_grading(store, task_ref=task_ref, task_id=task_id, task=task,
                                         catalog_key=catalog_key, site=settings.site)
    override = None if soft_sec is None or hard_sec is None else {"soft_sec": soft_sec, "hard_sec": hard_sec}
    registry = read_secrets(settings.site.secrets_path)
    provisioned = planned_provision(settings, task, task_ref, config=config, overrides=overrides)
    decls = rollout_stages(task_ref=task_ref, task=task, producer=producer,
                           serve_command=list(module.ARTIFACT.serve_command), arm=record, arm_change=arm_change,
                           catalog_key=catalog_key, run=settings.run, seal_memory_gib=seal_memory_gib(settings.site),
                           allocation=producer.evaluation_allocation,
                           budgets_override=override, stop_after=stop_after, probe_oom=probe_oom,
                           build_only=build_only, held=cache_holds(settings),
                           recipe_key=substrate(settings.site).recipe_key,
                           provisioned=provisioned, dev_files=planned_dev_files(settings, measured))
    plan = build_plan(decls=decls, task=task_id, pipeline="rollout",
                      layout=InstanceLayout.rollout(task_id, new_rollout_id()), settings=settings, code=code,
                      toolchain=toolchain_here(), job_name=job_name, known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=known_task_ids(), known_patterns=STAGE_NAME_PATTERNS, now=time.time(),
                      secrets=_checked_secrets(settings, echo, decls, registry), measured=measured)
    check_score_memory(plan, task)
    built = AgentBuildParams.model_validate(plan.stage(AGENT_BUILD_STAGE).params)
    for line in [*arm_lines(built.arm, soft_sec=built.soft_sec, hard_sec=built.hard_sec, arm_change=arm_change),
                 *notes]:
        echo(line)
    return plan




def resume(instance: str, *, config: Path, store: Path | None = None, scheduler: Scheduler | None = None,
           echo: Echo = print, sleep: Callable[[float], None] = time.sleep,
           clock: Callable[[], float] = time.time) -> int:
    """Take an instance over from an orchestrator that is gone and drive it to its end.
    A lease whose holder is not provably dead is waited out: until its last heartbeat is a TTL
    old, then tried once more. A holder that is alive on this host is not waited for. If the
    holder is still there, say so and exit 2. An instance that has ended with nothing
    outstanding is reported and left alone; one whose records an orchestrator left is taken over,
    and its records written (``Outcome.records_left``)."""
    root, layout = open_store(config, store), InstanceLayout(instance)
    try:
        site = _latest(root, layout).settings.site
        taken = _take_over(root, layout, _scheduler(scheduler, site), Holder.here("orchestrator"),
                           give_up_at=lambda record: record.heartbeat_at + record.ttl_sec, stop_if_alive=True,
                           echo=echo, sleep=sleep, clock=clock)
    except NoSuchInstance as e:
        echo(str(e))
        return 2
    except LockHeld as held:
        echo(f"{instance} is held by {_who(held.record)} ({_beat(held.record, clock())}); not resumed")
        return 2
    if isinstance(taken, Orchestrator):
        echo(f"attached to {instance} as lease generation {taken.handle.generation}")
        return _finish(taken.run(), echo)
    _already_ended(taken.plan, echo)
    _say_score(taken.plan, echo)
    _say_calibration(taken.plan, echo)
    return _exit_status(taken.plan)


def cancel(instance: str, *, config: Path, store: Path | None = None, scheduler: Scheduler | None = None,
           echo: Echo = print, sleep: Callable[[float], None] = time.sleep,
           clock: Callable[[], float] = time.time) -> int:
    """Cancel an instance's recorded jobs, and say whether that happened. The request is left
    for the orchestrator holding the lease, which cancels at its next poll; this command waits
    for it, or for the lease to free so that it can take it and cancel itself; a holder alive
    on this host is waited for as well, since it will act on the request. It gives up after the
    lease TTL and two polls and exits 3. Exit status 0 only when the instance ended cancelled.
    An instance that has ended with nothing outstanding gets no request. A secrets file that
    cannot be read does not stop the cancel: what takes the instance over records no free text."""
    root, layout = open_store(config, store), InstanceLayout(instance)
    try:
        plan = _latest(root, layout)
        queue = _scheduler(scheduler, plan.settings.site)
        if settled(plan, queue, now=clock()):
            _already_ended(plan, echo)
            return 0 if plan.outcome.status == "cancelled" else 1
        request_cancel(root, layout, at=clock())
        run = plan.settings.run
        limit = clock() + run.lease_ttl_sec + 2 * run.poll_sec
        taken = _take_over(root, layout, queue, Holder.here("hls cancel"), give_up_at=lambda record: limit,
                           stop_if_alive=False, echo=echo, sleep=sleep, clock=clock, cancel_only=True)
    except NoSuchInstance as e:
        echo(str(e))
        return 2
    except LockHeld as held:
        echo(f"cancel requested, not confirmed: {_who(held.record)} still holds the lease "
             f"({_beat(held.record, clock())})")
        return 3
    if isinstance(taken, Orchestrator):
        final = taken.run()
        _finish(final, echo)
    else:
        final = taken.plan
        assert final.outcome is not None
        if taken.holder is not None and final.outcome.status == "cancelled":
            echo(f"{instance} cancelled by {_who(taken.holder)}: {final.outcome.reason}")
        else:
            _already_ended(final, echo)
    assert final.outcome is not None
    return 0 if final.outcome.status == "cancelled" else 1


#: How ``status`` names an outcome whose status alone does not say what follows from it.
_ENDINGS = {"failed": "failed (a contestant fault: scored, never relaunched)",
            "void": "void (not the contestant's result)"}
#: How ``status`` names a fault class.
_FAULT_CLASSES = {"infra": "an infra fault", "contestant": "a contestant fault"}


def _ago(t: float | None, now: float) -> str:
    return "never" if t is None else f"{now - t:.0f}s ago"


def _clock(t: float) -> str:
    return datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")


def _abandoned(state: str, job_id: str, jobs: Mapping[str, JobInfo]) -> str | None:
    """How to show a runner or attempt status that still says it is running although its job
    has ended, as when the runner was killed before it wrote its final status; None for any
    other status. The status file is the runner's own, so it is shown qualified, not changed."""
    info = jobs.get(job_id)
    if state not in ("starting", "running") or info is None or not info.state.terminal:
        return None
    return f"abandoned (job {job_id} ended {info.state.value})"


def _attempt_statuses(root: PosixStore, layout: InstanceLayout,
                      stage: str) -> list[tuple[int, AttemptStatus | str]]:
    """Each attempt of *stage* with its status, or with why it has none to show."""
    attempts: list[tuple[int, AttemptStatus | str]] = []
    for n in layout.attempts(root, stage):
        try:
            raw = root.read_bytes(layout.attempt_file(stage, n, "status.json"))
            attempts.append((n, AttemptStatus.model_validate_json(raw)))
        except NotFound:
            attempts.append((n, "claimed"))
        except ValidationError as e:
            attempts.append((n, f"status unreadable ({validation_errors(e)})"))
    return attempts


def status_text(instance: str, *, config: Path, store: Path | None = None,
                scheduler: Scheduler | None = None) -> str:
    """The instance's plan, lease, allocations, jobs, stages and attempts, for people. Reads
    only: it takes no lease and changes nothing. Raise NoSuchInstance when the store holds no
    plan. A status file that does not parse is named as unreadable, not raised. A runner or
    attempt status that says running while its job has ended reads abandoned."""
    root, layout = open_store(config, store), InstanceLayout(instance)
    plan = _latest(root, layout)
    now = time.time()
    lines = [f"instance {plan.instance} (pipeline {plan.pipeline}), plan revision {plan.revision} "
             f"by lease generation {plan.lease_generation}"]
    if plan.outcome is None:
        lines.append("outcome: still running")
    else:
        ending = _ENDINGS.get(plan.outcome.status, plan.outcome.status)
        lines.append(f"outcome: {ending} at {_clock(plan.outcome.at)}: {plan.outcome.reason}")
        if plan.outcome.records_left:
            lines.append("records: left for hls resume, which writes them; the failed_records events say why")
    lease = instance_lease(root, layout, ttl_sec=plan.settings.run.lease_ttl_sec).current()
    if lease is not None:
        h = lease.holder
        lines.append(f"lease: generation {lease.generation}, {h.purpose} on {h.host} pid {h.pid}, acquired "
                     f"{_clock(lease.acquired_at)}, heartbeat {_ago(lease.heartbeat_at, now)}"
                     + (", released" if lease.released else ""))
    lines += [f"warning: {w}" for w in plan.warnings]
    attempts_of = {stage.name: _attempt_statuses(root, layout, stage.name) for stage in plan.stages}
    ids = [s.job_id for a in plan.allocations for s in a.submissions if s.job_id]
    ids += [st.job_id for attempts in attempts_of.values() for _, st in attempts if isinstance(st, AttemptStatus)]
    ids = list(dict.fromkeys(ids))
    jobs = _scheduler(scheduler, plan.settings.site).states(ids) if ids else {}
    named, says_why = _STATUS_SCHEDULERS[plan.settings.site.substrate]
    for alloc in plan.allocations:
        retry = f", retry of allocation {alloc.retry_of}" if alloc.retry_of is not None else ""
        lines.append(f"allocation {alloc.index}: stages {', '.join(alloc.stages)}; {_resources(plan, alloc)}{retry}")
        for s in alloc.submissions:
            job = f"job {s.job_id}" if s.job_id else "no job id"
            info = jobs.get(s.job_id) if s.job_id else None
            state = "" if info is None else f", {named} {info.state.value}"
            if info is not None and says_why and info.state is JobState.PENDING and info.reason:
                state += f" ({info.reason})"
            after = f", after job {s.after_job}" if s.after_job else ""
            line = f"  submission {s.n}: {job}, {s.status}{state}{after}"
            try:
                raw = root.read_bytes(layout.allocation_status(alloc.index, s.n))
                st = AllocationStatus.model_validate_json(raw)
                state = _abandoned(st.state, st.job_id, jobs) or st.state
                line += (f"; runner {state} on {st.node}" + (f" in stage {st.stage}" if st.stage else "")
                         + f", heartbeat {_ago(st.heartbeat_at, now)}")
            except NotFound:
                pass
            except ValidationError as e:
                line += f"; runner status unreadable ({validation_errors(e)})"
            if s.strays:
                line += f"; strays: {', '.join(s.strays)}"
            if s.rewire_to:
                line += f"; rewire to job {s.rewire_to} intended, {s.rewire_refusals} refusals"
            if s.missing_since is not None:
                line += f"; no job under its tag since {_clock(s.missing_since)}"
            if s.error:
                line += f"; last error: {s.error}"
            lines.append(line)
        if alloc.failure is not None:
            f = alloc.failure.fault
            lines.append(f"  failed at stage {alloc.failure.stage}, {_FAULT_CLASSES[f.fault_class]}: {f.reason}")
    for stage in plan.stages:
        attempts = []
        for n, st in attempts_of[stage.name]:
            if isinstance(st, str):
                attempts.append(f"{n} {st}")
                continue
            state = _abandoned(st.state, st.job_id, jobs) or st.state
            attempts.append(f"{n} {state}" + (f" ({st.fault.reason})" if st.fault else ""))
        published = "published" if root.exists(layout.manifest(stage.name)) else "not published"
        lines.append(f"stage {stage.name}: {published}; attempts: {', '.join(attempts) or 'none'}")
    lines += _provisioned_lines(root, plan)
    return "\n".join(lines)


def _provisioned_lines(store: PosixStore, plan: Plan) -> list[str]:
    """What the instance's provision stage put in the cache, once it published: the task model's
    weights, the rows and each tool bundle, by cache key. A record that is missing, is not the
    file its manifest recorded or does not parse is named as unreadable."""
    stage = next((s for s in plan.stages if s.kind == ProvisionStage.kind), None)
    found = read_manifest(store, plan.layout, stage.name) if stage is not None else None
    if found is None or "provisioned" not in found[0].outputs:
        return []
    ref = found[0].outputs["provisioned"]
    try:
        data = store.read_bytes(ref.key)
    except NotFound:
        return [f"provisioned: unreadable ({ref.key} is missing)"]
    if sha256_bytes(data) != ref.sha256:
        return [f"provisioned: unreadable ({ref.key} is not the file its manifest recorded)"]
    try:
        record = Provisioned.model_validate_json(data)
    except ValidationError as e:
        return [f"provisioned: unreadable ({validation_errors(e)})"]
    lines = []
    if record.weights is not None:
        lines.append(f"provisioned weights: {record.weights.repo} at {record.weights.revision} "
                     f"(cache weights/{record.weights.key})")
    if record.rows is not None:
        lines.append(f"provisioned rows: tenants {', '.join(record.rows.tenants)} (cache rows/{record.rows.key})")
    lines += [f"provisioned tool bundle: {t.name} {t.identity}, {t.variant} (cache tools/{t.key})"
              for t in record.tools]
    return lines


def status(instance: str, *, config: Path, store: Path | None = None, scheduler: Scheduler | None = None,
           echo: Echo = print) -> int:
    """Print ``status_text``. Exit status 0, or 2 when the store holds no such instance."""
    try:
        text = status_text(instance, config=config, store=store, scheduler=scheduler)
    except NoSuchInstance as e:
        echo(str(e))
        return 2
    echo(text)
    return 0
