"""The pipelines the orchestrator can run.

The two no-op pipelines are the minimal pipeline the test suite and a site's live tests run:
the whole path (plan, sbatch, batch script, runner, role steps, manifests) with no real work.
``noop`` is one stage, and ``noop2`` is two stages with different GPU counts, so the two GPU
policies plan it differently. The others are the cache warmers
``provision``, ``images`` (a task's shared images) and ``package`` (a source bundle scanned for
secret values, then built into a submission image on the task's submission base); the grading's
tail, which serves one artifact, drives it, verifies and scores the drive and records it;
``oracle_fill``, a cache warmer on the GPU partition that computes a task's reference answers to
its test split into the store ahead of its gradings; the rollout of ``hls run``, which builds a
submission with an arm's agent, packages it and grades it in one allocation, or with
``--build-only`` ends at the package; and the grading of ``hls grade``, which serves a rollout's
package, a package warmer's, a measured image or a published image, and may scan and package a
rollout's source bundle again. The credit, calibration and export pipelines live in their own
modules (``hls.pipeline.credit``, ``hls.pipeline.calibration``, ``hls.pipeline.export``); their
stage names are declared here, so every hls.toml table can be checked against them. A grading's
drive and a credit's are the same stage (``drive_decl``), and a drive of the measured baseline in
either comes after the same warm stage (``warm_baseline_decl``)."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

import hls.stages.registry  # noqa: F401  (registers every stage implementation)
from hls.arms import ArmRecord
from hls.calibrate import POOL_ENGINE_NAMES
from hls.artifact.artifact import EvaluationAllocation
from hls.drive.bounds import REPLY_MAX_BYTES, drive_bounds, trace_disk_bytes, trace_event_bytes, verify_peak_bytes
from hls.egress.policy import NetworkPolicy
from hls.package.checkout import CHECKOUT_MEMORY_GIB
from hls.pipeline.planner import StageDecl
from hls.provision.model import Provisioned
from hls.provision.mounts import copied_entries, copies_gib
from hls.provision.sandbox_images import SandboxImageSet
from hls.recipes import catalog
from hls.recipes.model import ImageDecl, order, parent_of
from hls.settings import RunSettings
from hls.spec.drivers import driver_for
from hls.spec.resources import GPU_KINDS, ResourceShape, concurrent
from hls.spec.tools import DEVICE
from hls.stages.agent_build import AgentBuildParams, budget_params, collection_sec
from hls.stages.reference_service import ReferenceServiceParams, service_shape
from hls.stages.baseline_warm import WARM_BASELINE
from hls.stages.calibration import INSTRUMENT, warm_stage
from hls.stages.deploy_drive import (SEAL_PROBE_SEC, ImageArtifact, PackagedArtifact, PublishedArtifact,
                                     extracted_disk_gib)
from hls.stages.engine_cache import COLD_LOAD_SEC
from hls.stages.finalize import FINALIZE_SEC
from hls.stages import package as package_stage
from hls.stages import source_scan
from hls.stages.images import KEPT_DISK_GIB, HeldKeys, disk_gib, fills, image_keys
from hls.stages.package import AGENT_BUILD_STAGE, PipLimits, StoredBundle, pip_limits
from hls.stages.score import score_memory_gib
from hls.stages.verify import reference_engines, verifier_gpus
from hls.substrate.factory import RecipeKey
from hls.substrate.seal import INGRESS_WAIT_SEC
from hls.taskref import TaskRef
from hls.verify.fill import DEFAULT_CHUNK

#: The task every no-op instance belongs to; its instances live under ``runs/noop/``.
NOOP_TASK = "noop"


def noop_one(*, gpus: int, gpu_kind: str, sleep_sec: int, stage_time_sec: int,
             fail_attempts: int = 0, contestant: bool = False, batch_hog_gib: int = 0,
             with_cpu_role: bool = False) -> list[StageDecl]:
    """One stage, ``noop``, with one role: a GPU role holding every GPU, or a CPU role. The
    stage declares the role's 2 GiB and the runner's default 4 GiB. *contestant* runs the role
    as the contestant's; *batch_hog_gib* is memory the stage holds in the batch step;
    *with_cpu_role* adds a harness role with no GPU beside the GPU role, and the stage declares
    its 2 CPUs and 2 GiB too."""
    if with_cpu_role and not gpus:
        raise ValueError("a CPU role beside the first role needs a GPU role to be beside")
    extra = 2 if with_cpu_role else 0
    shape = ResourceShape(gpus=gpus, gpu_kind=gpu_kind if gpus else None, cpus=2 + extra, memory_gib=6 + extra,
                          time_sec=stage_time_sec)
    params = {"sleep_sec": sleep_sec, "gpu_roles": 1 if gpus else 0, "cpu_role": with_cpu_role or not gpus,
              "fail_attempts": fail_attempts, "contestant_role": contestant,
              "batch_hog_gib": batch_hog_gib}
    return [StageDecl(name="noop", kind="noop", shape=shape, params=params)]


def noop_two(*, gpu_kind: str, sleep_sec: int, stage_time_sec: int, fail_attempts: int = 0) -> list[StageDecl]:
    """``noop_a`` on one GPU, then ``noop_b`` on two: two exclusive GPU roles, a role sharing
    the second one's GPU, and a role with no GPU. *fail_attempts* applies to ``noop_a``.
    ``noop_a`` declares its role's 2 GiB and the runner's default 4 GiB; ``noop_b`` declares
    12 GiB, its roles' 7 GiB, the runner's 4 and 1 to spare."""
    a = ResourceShape(gpus=1, gpu_kind=gpu_kind, cpus=4, memory_gib=6, time_sec=stage_time_sec)
    b = ResourceShape(gpus=2, gpu_kind=gpu_kind, cpus=8, memory_gib=12, time_sec=stage_time_sec)
    return [
        StageDecl(name="noop_a", kind="noop", shape=a,
                  params={"sleep_sec": sleep_sec, "gpu_roles": 1, "fail_attempts": fail_attempts}),
        StageDecl(name="noop_b", kind="noop", shape=b, after=("noop_a",),
                  params={"sleep_sec": sleep_sec, "gpu_roles": 2, "shared_role": True, "cpu_role": True}),
    ]


#: The provision stage's CPUs: its roles run one at a time, and four CPUs are room for a
#: download's threads and the fill's hashing.
PROVISION_CPUS = 4
#: Memory, in GiB, each role of a provision stage gets when its task declares none
#: (``TaskSpec.provision_memory_gib``): more than a quarter over the 5.55 GiB that Slurm's record
#: of a role step's memory (sacct MaxRSS) reads for a multi-GB weights download. That record counts
#: page cache, so it is above the footprint the runner holds a role to. A task whose peak and a
#: quarter more passes this declares its own.
PROVISION_ROLE_GIB = 12
#: A cache warmer's provision time: four hours, a 60 GiB model's download at 4.5 MB/s, a slow
#: hub link.
PROVISION_SEC = 4 * 3600
#: An evaluation's provision time: two hours, since a cache warmer has usually filled the cache
#: before a task is evaluated.
EVALUATE_PROVISION_SEC = 2 * 3600


def provision_shape(task: Any, *, runner_memory_gib: int, time_sec: int) -> ResourceShape:
    """The provision stage's shape for *task*, the ``TASK`` it provisions, for *time_sec*: a CPU
    stage on the host side, which a run places in its GPU allocation and a cache warmer on a CPU
    partition, of ``PROVISION_CPUS``, and the memory each of its roles gets (the task's
    ``provision_memory_gib``, or ``PROVISION_ROLE_GIB`` when it declares none) with the runner's
    *runner_memory_gib* (``run.runner_memory_gib``) on top. ``[resources.<task>.provision]``
    overrides it."""
    role = task.provision_memory_gib if task.provision_memory_gib is not None else PROVISION_ROLE_GIB
    return ResourceShape(cpus=PROVISION_CPUS, memory_gib=role + runner_memory_gib, time_sec=time_sec)


def runner_stores(task: Any) -> tuple[str, ...]:
    """The site's sandbox stores *task*'s tool runner names (``ToolRunner.host_paths``), each
    once; none for a task without one."""
    paths = task.agent.tool_runner.host_paths if task.agent is not None else ()
    return tuple(dict.fromkeys(p.store for p in paths if p.store != DEVICE))


def provision_stages(*, task_ref: TaskRef, shape: ResourceShape, weights: bool = True,
                     sandbox_stores: Sequence[str] = ()) -> list[StageDecl]:
    """One stage, ``provision``, of *shape* (``provision_shape``), that fills the cache with what
    the task *task_ref* names needs: its module's ``TASK`` with its overrides, which can change its
    rows and tool bundles, and makes what its tool runner needs in the site's *sandbox_stores*
    (``runner_stores``). *weights* False provisions everything but the task model's weights."""
    params: dict[str, Any] = {"task": task_ref.model_dump(mode="json"), "weights": weights}
    # Keyed only for a task whose runner names stores.
    if sandbox_stores:
        params["sandbox_stores"] = list(sandbox_stores)
    return [StageDecl(name="provision", kind="provision", shape=shape, params=params)]


#: The stage of ``hls provision`` that pulls a task's sandbox images into their site store.
SANDBOX_IMAGES_STAGE = "sandbox_images"


def sandbox_images_stage(images: SandboxImageSet, *, runner_memory_gib: int) -> StageDecl:
    """Stage ``sandbox_images``, after ``provision``, which leaves every image of *images*, a
    task's declared set, in its site store, pulling those the store lacks. Its shape is the set's
    fixed allowances, the role's CPUs, memory, node-local disk and time as the task declares them,
    with the runner's *runner_memory_gib* (``run.runner_memory_gib``) on top of the memory.
    ``[resources.<task>.sandbox_images]`` overrides it."""
    shape = ResourceShape(cpus=images.cpus, memory_gib=images.memory_gib + runner_memory_gib,
                          disk_gib=images.disk_gib, time_sec=images.time_sec)
    return StageDecl(name=SANDBOX_IMAGES_STAGE, kind="sandbox_images", shape=shape, after=("provision",),
                     params={"images": images.model_dump(mode="json")})


#: The images stage's CPUs and memory, and its time when it fills no entry; its disk follows from
#: the entries it fills, and its time grows with them (``images_shape``). mksquashfs uses every
#: core; a base's build is given 64 GiB, a chosen bound and not a fit, and the stage also leaves
#: the runner its reserve (``run.runner_memory_gib``, 4 GiB by default). Its own steps beside its
#: fills, an entry's lookup for each image and its records, take seconds; fifteen minutes is a
#: fixed allowance.
IMAGES_SHAPE = ResourceShape(cpus=16, memory_gib=68, time_sec=900)
#: Seconds the images stage gives each entry it fills, a fixed allowance: a build's pull, installs
#: and packing, or an import's copy and check, then the image's copy into the entry, its
#: extraction and its inventory. Half an hour is nearly three times what a base's build with its
#: registry pull takes, about 11 minutes.
IMAGE_FILL_SEC = 1800
#: Seconds a stage allows before its first role starts for what it copies and extracts off the
#: shared filesystem: its roles' data copies (the weights, the rows, a tool bundle) and the image
#: files it extracts. A drive, verify, oracle fill, calibration and agent build stage each add it
#: to their time, a fixed allowance. An hour is about nine times what the staging of a drive
#: that copies 55 GB of weights takes.
STAGING_SEC = 3600


def package_shape(disk_cap_gib: int) -> ResourceShape:
    """The package stage's shape for a bundle of a build whose writable disk cap is *disk_cap_gib*:
    8 CPUs, 36 GiB, memory for the pip steps and the offline build with the runner's reserve on
    top, the disk its floor needs (``hls.stages.package.min_disk_gib``), and its fixed time
    (``hls.stages.package.MIN_TIME_SEC``): the four hours both pip steps may take together, the
    checkout, and two hours for the build, the extractions, the integrity check over both trees
    and the submission image's publish."""
    return ResourceShape(cpus=8, memory_gib=36, disk_gib=package_stage.min_disk_gib(disk_cap_gib),
                         time_sec=package_stage.MIN_TIME_SEC)


def source_scan_shape(disk_cap_gib: int, *, runner_memory_gib: int) -> ResourceShape:
    """The source_scan stage's shape for a bundle of a build whose writable disk cap is
    *disk_cap_gib*: two CPUs; memory for its one role at a time, the checkout's
    (``hls.package.checkout.CHECKOUT_MEMORY_GIB``, far above the tree scan's, 0.03 GiB at the
    tree's bounds), with the runner's *runner_memory_gib* on top; the disk of a
    checkout of a bundle at the cap (``hls.stages.source_scan.min_disk_gib``); and its fixed time
    (``hls.stages.source_scan.MIN_TIME_SEC``)."""
    return ResourceShape(cpus=2, memory_gib=CHECKOUT_MEMORY_GIB + runner_memory_gib,
                         disk_gib=source_scan.min_disk_gib(disk_cap_gib), time_sec=source_scan.MIN_TIME_SEC)


def images_shape(fill_count: int, *, cpus: int = IMAGES_SHAPE.cpus) -> ResourceShape:
    """``IMAGES_SHAPE`` with *cpus*, for an images stage that fills *fill_count* entries: the disk
    it holds at its peak (``disk_gib``), and ``IMAGES_SHAPE``'s time with ``IMAGE_FILL_SEC`` for
    each fill."""
    return IMAGES_SHAPE.model_copy(update={
        "cpus": cpus, "disk_gib": disk_gib(fill_count),
        "time_sec": IMAGES_SHAPE.time_sec + fill_count * IMAGE_FILL_SEC})


def images_stage(*, catalog_task: str, decls: Sequence[ImageDecl], held: HeldKeys,
                 recipe_key: RecipeKey, cpus: int = IMAGES_SHAPE.cpus) -> StageDecl:
    """Stage ``images``, on *cpus* CPUs, which builds *decls* of the catalog's task
    *catalog_task*, each keyed by *recipe_key*, the site's substrate's (``Substrate.recipe_key``).
    *held* says which of its images' keys the store's cache holds (``held_keys``), and is asked
    about those alone: the stage records them as found, and its disk and its time count the
    entries it fills, the rest (``images_shape``). Raise ValueError naming an image whose recipe
    cannot be keyed."""
    keys = image_keys(decls, recipe_key)
    found = sorted(held(set(keys.values())))
    return StageDecl(name="images", kind="images", shape=images_shape(len(fills(keys, found)), cpus=cpus),
                     params={"catalog_task": catalog_task, "images": [d.model_dump(mode="json") for d in decls],
                             "found": found})


def images_stages(*, task: str, held: HeldKeys, recipe_key: RecipeKey) -> list[StageDecl]:
    """One stage, ``images``, that builds every image of catalog task *task*: its built chain
    (base, layer, agent cap) and each image it declares for itself, each keyed by *recipe_key*;
    its disk and its time count the entries *held* says the cache lacks (``images_stage``). Raise
    ValueError when *task* is no task, or declares no images."""
    images = catalog.task_images(task)
    return [images_stage(catalog_task=task, decls=images.decls, held=held, recipe_key=recipe_key)]


def _chain(decls: list[ImageDecl], name: str) -> list[ImageDecl]:
    """Image *name* and every image it builds on, parents first."""
    by_name = {d.name: d for d in decls}
    out: list[ImageDecl] = []
    current: str | None = name
    while current is not None:
        out.insert(0, by_name[current])
        current = parent_of(by_name[current])
    return out


def package_stages(*, task: str, catalog_key: str, bundle: StoredBundle, bundle_sha256: str,
                   network: dict[str, Any], gpu_kind: str | None, held: HeldKeys, recipe_key: RecipeKey,
                   disk_cap_gib: int, runner_memory_gib: int, seal_memory_gib: int,
                   server_filename: str = "server.py") -> list[StageDecl]:
    """``images`` for the submission base of the task the image catalog knows as *catalog_key*
    and what it builds on, its disk and time counting the entries *held* says the cache lacks
    (``images_stage``); ``source_scan``, which reads the source *bundle* (the copy ``hls package``
    made in the instance's store, whose sha256 the operator states as *bundle_sha256*) for every
    secret value and publishes what it found; then ``package``, which builds the bundle into a
    submission image on that base for task *task*, the id ``package.json`` records. Both hold the
    bundle to *disk_cap_gib*, the
    writable disk cap of the task's builds (``sandbox_disk_gib``), and are shaped for a bundle at
    it; the pip steps are held to what the package's shape gives with the runner keeping
    *runner_memory_gib* and their seal *seal_memory_gib* (``pip_limits``). Raise ValueError when *catalog_key*
    is no task, or declares no images."""
    images = catalog.task_images(catalog_key)
    chain = _chain(images.decls, images.submission_base)
    # Each stage's extracted trees go when it closes, so the package stage's own floor
    # (package_shape) holds the two trees it pins, and the allocation the larger of the stages.
    package = package_shape(disk_cap_gib)
    return [
        images_stage(catalog_task=catalog_key, decls=chain, held=held, recipe_key=recipe_key),
        StageDecl(name="source_scan", kind="source_scan",
                  shape=source_scan_shape(disk_cap_gib, runner_memory_gib=runner_memory_gib),
                  params={"bundle": bundle.model_dump(), "bundle_sha256": bundle_sha256,
                          "disk_cap_gib": disk_cap_gib}),
        StageDecl(name="package", kind="package", shape=package, after=("images", "source_scan"),
                  params={"task": task, "submission_base": images.submission_base,
                          "constraints": images.submission_constraints, "images_stage": "images",
                          "network": network, "gpu_kind": gpu_kind, "server_filename": server_filename,
                          "bundle": bundle.model_dump(), "bundle_sha256": bundle_sha256,
                          "disk_cap_gib": disk_cap_gib,
                          "pip": pip_limits(package, runner_memory_gib + seal_memory_gib).model_dump()}),
    ]


#: Seconds each readiness gate of a drive may take, health and then inference: 45 minutes.
#: Engines sharing a node read their weights off the shared filesystem at once, which can stretch
#: a healthy warmup past 15 minutes.
#: A crash still fails at once; this bound only ends a hang.
READINESS_TIMEOUT_SEC = 2700
#: Seconds of the drive stage kept after the drive's wall, for finishing and recording the trace
#: and stopping the artifact: about twice what a driver takes to write a whole 11 GB trace of
#: event JSON beside a busy server. The drive role writes its trace as the drive goes, so what
#: is left at the wall is the events its writer has not written yet, far less. A drive stage
#: adds the time to copy the trace to the store (``TRACE_COPY_SEC``); a calibration's serving
#: writes its capture after the serving, which this covers.
DRIVE_MARGIN_SEC = 1200
#: Seconds a drive stage adds after ``DRIVE_MARGIN_SEC`` for the drive role to publish the trace
#: from its room on the node's disk to the store, a fixed allowance. A copy from node-local disk
#: into the store runs at about 350 MB/s, at which an hour holds over a terabyte. A copy the store
#: slows past the hour ends the stage as infra.
TRACE_COPY_SEC = 3600
#: Seconds the drive stage allows the server's runtime to start its command, which no readiness
#: budget is charged: the stage's own give-up for the seal's start, the phase before the seal
#: starts its payload (``hls.substrate.seal.INGRESS_WAIT_SEC``, an interceptor's bind
#: of up to a minute under load and the image paging in after it). Each drive records its
#: server's own start (``ArtifactRecord``).
RUNTIME_START_SEC = math.ceil(INGRESS_WAIT_SEC)
#: The drive's wall when its task declares none: three hours from the drive's start. Such a drive
#: gets all of it, as a walled one gets its own, or does not start.
OPEN_DRIVE_SEC = 3 * 3600
#: The drive deadline's grace the drive stage gives its driver: every request must complete within
#: this many seconds after the drive's last scheduled arrival, and one sent later gets all of it
#: unless the drive's wall comes first (``hls.spec.drivers._Deadline``). The wall can end a late
#: request before its grace: on an open single-turn drive the wall, at t0 + ``OPEN_DRIVE_SEC``,
#: comes before t0 + the last arrival + this grace, so there every request ends by the wall. A
#: missed request is a failed one, and one failure zeroes the score and skips the correctness gate.
#: Slowness already costs continuously through log(1/r), so it needs no cliff: this bound only
#: stops a hung server from measuring forever.
REQUEST_TIMEOUT_SEC = 10800.0
#: The driver role's shape for a task whose tool assets ask for nothing and that declares no
#: evaluation allocation, for an open drive: its time is the drive stage's after the staging
#: (``drive_time``), which each stage adds for its own copies and images (``STAGING_SEC``).
DRIVER_SHAPE = ResourceShape(cpus=4, memory_gib=16,
                             time_sec=RUNTIME_START_SEC + 2 * READINESS_TIMEOUT_SEC + OPEN_DRIVE_SEC + DRIVE_MARGIN_SEC)
#: The reference job's 8 CPUs and 180 GiB beside its GPU, which the verify role gets
#: less the runner's reserve (``reference_shape``) before what the task's reference engines ask
#: for, unless the task's evaluation allocation declares the role's memory
#: (``EvaluationAllocation.verify_memory_gib``). Its two hours are the verify role's time for a task whose
#: evaluation allocation declares none (``EvaluationAllocation.verify_time_sec``); each stage that
#: runs the role adds its own staging (``STAGING_SEC``).
REFERENCE_SHAPE = ResourceShape(cpus=8, memory_gib=180, time_sec=2 * 3600)
#: The score stage's CPUs and time: one host role over the trace's summary. Its memory here is
#: the role's least (``hls.stages.score.SCORE_BASE_GIB``) with the runner's default reserve
#: beside it; an evaluation gives the role what one reply's summary line can make it hold
#: (``score_shape``).
SCORE_SHAPE = ResourceShape(cpus=2, memory_gib=12, time_sec=1800)
#: The finalize stage's shape: the records and the secret scan of every published file, with
#: 8 GiB, and the runner's default reserve beside them, for its fixed time
#: (``hls.stages.finalize.FINALIZE_SEC``).
FINALIZE_SHAPE = ResourceShape(cpus=2, memory_gib=12, time_sec=FINALIZE_SEC)
#: An evaluation's images stage's CPUs: 8 per GPU, the most some sites let a job ask for per
#: GPU. A cache warmer has usually built the
#: images already (a hit takes seconds), so a build here is only slower. The oracle fill, a
#: GPU job too, takes the same.
EVALUATE_IMAGES_CPUS = 8
#: The stages of a rollout that builds and grades its build, in order.
ROLLOUT_STAGES = ("provision", "images", "agent_build", "source_scan", "package", "deploy_drive", "verify", "score",
                  "finalize")
#: The stages of a grading, in order; ``source_scan`` and ``package`` only for one that packages a
#: rollout's source bundle again, and ``warm_baseline`` only for one of the measured baseline.
GRADING_STAGES = ("provision", "images", "source_scan", "package", WARM_BASELINE, "deploy_drive", "verify", "score",
                  "finalize")
#: The fixed names of a baseline credit's stages, the fold last; its drives are numbered
#: (``STAGE_NAME_PATTERNS``).
CREDIT_STAGE_NAMES = ("provision", "images", WARM_BASELINE, "fold")
#: The names of a calibration's first stages that its grower does not add, a warm stage for each
#: pool member among them; what the grower adds (its servings, its score shares and the
#: aggregate) is declared by ``hls.stages.calibration``, which names it.
CALIBRATION_STAGE_NAMES = ("provision", "images", INSTRUMENT, *(warm_stage(e) for e in POOL_ENGINE_NAMES))
#: The export's one stage.
EXPORT_STAGE_NAMES = ("export",)
#: The numbered stage names a pipeline declares as many of as it is asked for, as regular
#: expressions a whole name must match: a credit's drives and their scores. A grower's numbered
#: names are its declaration's (``hls.pipeline.grown.grown_stages``).
STAGE_NAME_PATTERNS = (r"drive_[1-9][0-9]*", r"credit_[1-9][0-9]*")


#: The oracle fill's time after its staging, which its stage adds (``oracle_fill_stages``):
#: 2 h 30 min, about four times what a cold fill of 2,400 prompts through an ASR reference takes,
#: rounded up to the half hour.
#: A fill cut short keeps every chunk it finished, and its retry, in an allocation that holds the
#: fill alone, asks only for the rest. ``[resources.<task>.oracle_fill]`` sets another time.
ORACLE_FILL_SEC = 9000
#: The stages of an oracle fill, in order.
ORACLE_FILL_STAGES = ("provision", "images", "oracle_fill")


def _gpu_kind(task: Any) -> str:
    kind = task.hardware.accelerator.kind
    if kind not in GPU_KINDS:
        raise ValueError(f"{task.id}'s accelerator {kind} is not a GPU kind the planner knows "
                         f"({', '.join(sorted(GPU_KINDS))})")
    return kind


def drive_wall(task: Any) -> float | None:
    """The task's own drive wall: its agent seam's (``AgentSpec.drive_wall_sec``), which is its
    ``drive_timeout_sec`` or, for a seam with a scheduled stop, the stop's backstop when later;
    None for a task with no agent seam, or whose seam declares neither."""
    return task.agent.drive_wall_sec() if task.agent is not None else None


def drive_wall_sec(task: Any) -> float:
    """The drive's wall: the task's own (``drive_wall``), or ``OPEN_DRIVE_SEC`` when it declares
    none. A drive gets all of it or does not start (``DeployDriveParams.drive_timeout_sec``)."""
    wall = drive_wall(task)
    return float(wall) if wall is not None else float(OPEN_DRIVE_SEC)


def drive_time(task: Any) -> int:
    """The drive stage's time after its staging, which each stage adds for its own copies and
    images (``STAGING_SEC``): the server's runtime's start, two full readiness budgets, the
    drive's wall (``drive_wall_sec``) and the margin after it."""
    return RUNTIME_START_SEC + 2 * READINESS_TIMEOUT_SEC + math.ceil(drive_wall_sec(task)) + DRIVE_MARGIN_SEC


def artifact_shape(task: Any, allocation: EvaluationAllocation | None = None, *,
                   runner_memory_gib: int) -> ResourceShape:
    """The artifact role's shape, for the drive stage's time: the task's accelerators with the
    CPUs and writable disk of its evaluation *allocation*, and the memory it gives the server
    when the runner keeps *runner_memory_gib* (``EvaluationAllocation.server_memory_gib``);
    without one, the hardware the brief states, its CPU count, memory and
    storage. The disk caps what the artifact's workdir and engine cache hold. Raise ValueError
    for an accelerator that is not a GPU kind the planner knows, for an allocation whose job the
    reserve leaves no memory, or, without an allocation, for hardware that states no CPU count,
    memory or storage."""
    hw = task.hardware
    gpus = hw.accelerator.count
    kind = _gpu_kind(task) if gpus else None
    if allocation is not None:
        return ResourceShape(gpus=gpus, gpu_kind=kind, cpus=allocation.cpus,
                             memory_gib=allocation.server_memory_gib(runner_memory_gib),
                             disk_gib=allocation.writable_gib, time_sec=drive_time(task))
    if hw.cpu_count < 1 or hw.memory_gib < 1 or hw.storage_gib < 1:
        raise ValueError(f"{task.id}'s hardware states {hw.cpu_count} CPUs, {hw.memory_gb} GB of memory and "
                         f"{hw.storage_gb} GB of storage; the artifact runs on what the brief states, so all three "
                         "must be set, memory and storage to at least 1 GiB")
    # The hardware states decimal GB; the shape takes whole GiB, rounded down.
    return ResourceShape(gpus=gpus, gpu_kind=kind, cpus=hw.cpu_count, memory_gib=hw.memory_gib,
                         disk_gib=hw.storage_gib, time_sec=drive_time(task))


def driver_overlaps(allocation: EvaluationAllocation | None) -> bool:
    """Whether the driver runs on the artifact's CPUs (``DeployDriveParams.driver_overlaps``):
    under an evaluation *allocation* that gives it none of its own."""
    return allocation is not None and allocation.driver_cpus is None


def driver_memory_shared(allocation: EvaluationAllocation | None) -> bool:
    """Whether the driver's memory is oversubscribed on the artifact's
    (``DeployDriveParams.driver_memory_shared``): under an evaluation *allocation* that says so
    (``EvaluationAllocation.driver_memory_shared``)."""
    return allocation is not None and allocation.driver_memory_shared


def driver_shape(task: Any, allocation: EvaluationAllocation | None = None) -> ResourceShape:
    """The driver role's shape, for the drive stage's time. Its memory is the evaluation
    *allocation*'s ``driver_memory_gib`` (``DRIVER_SHAPE``'s without one), or the loaded tool
    bundle's when that is more, and it takes a GPU of the task's kind when its tool encodes on
    one: the request of the driver ``driver_for`` builds. Its CPUs are the
    allocation's ``driver_cpus``, or the artifact's when it overlaps them (``driver_overlaps``);
    ``DRIVER_SHAPE``'s without an allocation. Raise ValueError for an overlapping driver whose
    tool encodes on a GPU: its step shares the artifact's, which holds the artifact's GPUs."""
    request = driver_for(task.agent, request_timeout_sec=REQUEST_TIMEOUT_SEC,
                         gpu_kind=task.hardware.accelerator.kind).resources
    gpus = request.gpu_count
    base = DRIVER_SHAPE.memory_gib if allocation is None else allocation.driver_memory_gib
    memory = max(base, math.ceil((request.memory_mb or 0) / 1024))
    if allocation is None:
        cpus = DRIVER_SHAPE.cpus
    elif allocation.driver_cpus is not None:
        cpus = allocation.driver_cpus
    elif gpus:
        raise ValueError(f"{task.id}: its driver's tool encodes on a GPU, so the driver needs CPUs of its own; "
                         "its evaluation allocation names none (driver_cpus)")
    else:
        cpus = allocation.cpus
    return ResourceShape(gpus=gpus, gpu_kind=_gpu_kind(task) if gpus else None, cpus=cpus, memory_gib=memory,
                         time_sec=drive_time(task))


def reference_shape(task: Any, allocation: EvaluationAllocation | None = None, *,
                    runner_memory_gib: int) -> ResourceShape:
    """The verify role's shape: ``REFERENCE_SHAPE``'s CPUs, the GPUs of the task's kind the
    verifier that holds the most at once needs (``verifier_gpus``: each engine of a verifier is
    placed on GPUs of its own while it runs), and the verify memory the task's evaluation
    *allocation* declares (``EvaluationAllocation.verify_memory_gib``), without one the reference
    job's memory less the runner's *runner_memory_gib*, or the memory of the verifier whose
    engines need the most together, or the verify role's measured peak over a trace at the task's
    cap and requests (``hls.drive.bounds.verify_peak_bytes``), when either is more, for the verify
    time the *allocation* declares (``REFERENCE_SHAPE``'s without one), which the stage running
    it gives the role after its staging. Engines are built, never loaded."""
    held = [sum(math.ceil((r.memory_mb or 0) / 1024) for r in engines) for _, engines in reference_engines(task)]
    bounds = task.drive_bounds
    peak = ([] if bounds is None else
            [math.ceil(verify_peak_bytes(bounds.verify_peak, trace_event_bytes(bounds.trace_cap_bytes,
                                                                               bounds.trace_request_bytes))
                       / 2**30)])
    role = allocation.verify_memory_gib if allocation is not None else None
    memory = max([role or REFERENCE_SHAPE.memory_gib - runner_memory_gib, *held, *peak])
    declared = allocation.verify_time_sec if allocation is not None else None
    return ResourceShape(gpus=verifier_gpus(task), gpu_kind=reference_gpu_kind(task), cpus=REFERENCE_SHAPE.cpus,
                         memory_gib=memory, time_sec=declared or REFERENCE_SHAPE.time_sec)


def reference_gpu_kind(task: Any) -> str | None:
    """The GPU kind the verify role runs on: the task's, or None when no verifier needs a GPU
    (``verifier_gpus``)."""
    return _gpu_kind(task) if verifier_gpus(task) else None


def images_chain(images: catalog.TaskImages, needed: Sequence[str]) -> list[ImageDecl]:
    """The images *needed* of a task whose images are *images*, each with what it builds on,
    parents first."""
    return order(list({d.name: d for name in needed for d in _chain(images.decls, name)}.values()))


def is_measured_baseline(artifact: PackagedArtifact | ImageArtifact | PublishedArtifact, catalog_key: str) -> bool:
    """Whether *artifact* is the measured baseline of the task the image catalog knows as
    *catalog_key*: the image of the baseline its module declares."""
    return (isinstance(artifact, ImageArtifact)
            and artifact.image == catalog.task_images(catalog_key).measured.get("baseline"))


def baseline_warm_sec() -> int:
    """A measured baseline's warm stage's time: its staging, its copies' and its extraction's
    (``STAGING_SEC``), ``RUNTIME_START_SEC`` for its server's runtime to start its command,
    ``COLD_LOAD_SEC`` for the server's load with an empty engine cache, and the margin after it, in
    which the stage stops the server and publishes its engine cache. The stage gives the load what
    its time leaves after the others (``hls.stages.baseline_warm.load_sec``), so a ``[resources]``
    table that gives it more time gives the load as much more."""
    return STAGING_SEC + RUNTIME_START_SEC + COLD_LOAD_SEC + DRIVE_MARGIN_SEC


def warm_baseline_decl(*, task_ref: TaskRef, task: Any, catalog_key: str, artifact: ImageArtifact,
                       serve_command: Sequence[str], allocation: EvaluationAllocation | None, run: RunSettings,
                       provisioned: Provisioned, after: Sequence[str]) -> StageDecl:
    """The ``baseline_warm`` stage (``WARM_BASELINE``) that owns the engine cache entry every drive of
    *task*'s measured baseline *artifact* starts from, the catalog knowing
    the task as *catalog_key*: it finds the entry, or serves the baseline once, as its drives serve
    it with *serve_command* under the task's runtime network policy, to build it. Its role is the
    drive's artifact role (``artifact_shape``), on the CPUs, memory and disk a drive's server gets,
    so the stage asks for no more than a drive of the baseline; the stage holds it with
    ``run.runner_memory_gib`` beside it and the disk of the image it extracts and of the copies of
    the artifact role's data, by what *provisioned*, the provision record the cache holds, states
    of them (``copied_entries``), which it declares (``data_copies``); its time is
    ``baseline_warm_sec``. It comes *after* the named stages. Raise ValueError for an *artifact* that
    is not the task's measured baseline, and when the artifact's shape cannot be sized or names no
    GPU kind."""
    if not is_measured_baseline(artifact, catalog_key):
        raise ValueError(f"a warm stage builds {catalog_key}'s measured baseline's engine cache, not {artifact.image}'s")
    art = artifact_shape(task, allocation, runner_memory_gib=run.runner_memory_gib)
    if art.gpu_kind is None:
        raise ValueError(f"{task.id}'s measured baseline runs on no GPU, so it has no engine cache to warm")
    data = copied_entries(("artifact",), provisioned)
    shape = art.model_copy(update={"memory_gib": art.memory_gib + run.runner_memory_gib,
                                   "disk_gib": art.disk_gib + KEPT_DISK_GIB + copies_gib(data),
                                   "time_sec": baseline_warm_sec()})
    params = {"task": task_ref.model_dump(mode="json"), "artifact": artifact.model_dump(mode="json"),
              "port": task.contract.port, "serve_command": list(serve_command),
              "network": task.network.model_dump(mode="json") if task.network is not None else None,
              "artifact_shape": art.model_dump(), "runtime_start_sec": RUNTIME_START_SEC, "staging_sec": STAGING_SEC,
              "margin_sec": DRIVE_MARGIN_SEC, "data_copies": data}
    return StageDecl(name=WARM_BASELINE, kind="baseline_warm", shape=shape, params=params, after=tuple(after))


def drive_decl(*, name: str, task_ref: TaskRef, task: Any, catalog_key: str,
               artifact: PackagedArtifact | ImageArtifact | PublishedArtifact, harness_image: str,
               serve_command: Sequence[str], allocation: EvaluationAllocation | None,
               run: RunSettings, seal_memory_gib: int, provisioned: Provisioned, after: Sequence[str],
               inputs: Mapping[str, str] | None = None,
               dev_files: Mapping[str, Mapping[str, str]] | None = None) -> StageDecl:
    """A ``deploy_drive`` stage named *name* that serves *artifact* of *task*, the ``TASK`` of the
    module *task_ref* names, which the image catalog knows as *catalog_key*: in the task's
    evaluation *allocation*, or on the hardware the brief states, under the task's runtime
    network policy, with *serve_command* after ``--host`` and ``--port``, driven on the harness
    image *harness_image* by the task's own driver within its own wall. The driver writes the
    drive's trace on the node's disk as the drive goes and holds it to the cap the task declares
    (``drive_bounds``), the server's replies to ``REPLY_MAX_BYTES`` each, and counts none of a
    drive of the task's measured baseline (``measured_baseline``: the artifact is the catalog's
    baseline image), whose server starts from the engine cache entry its warm stage
    (``WARM_BASELINE``, among *after*) left; any other artifact's starts
    with an empty one. The stage holds both roles and ``run.runner_memory_gib`` beside them, with
    *seal_memory_gib* for the seal of its sealed roles (``hls.substrate.factory.seal_memory_gib``),
    which the artifact's memory leaves as it leaves the runner's, but no memory for a driver the
    allocation oversubscribes on the artifact's (``driver_memory_shared``), and the disk of the
    images it extracts, of the copies of its roles' data (the artifact's and the
    driver's), by what *provisioned*, the provision record the cache holds
    (``hls.stages.provision.cached_provision``), states of them (``copied_entries``), which it
    declares (``data_copies``), and the driver's of the trace's room (``trace_disk_bytes``);
    its time is its roles' (``drive_time``) after its staging (``STAGING_SEC``) and its seal probe
    (``SEAL_PROBE_SEC``), and the margin after the wall grows by the trace's copy to the store
    (``TRACE_COPY_SEC``); it allows the server's runtime ``RUNTIME_START_SEC`` to start its
    command; it comes *after* the named stages and pins the manifests *inputs* names with their
    digests. It records the task's trace cap and request bytes, and what its tool runner needs in
    the driver's container (``ToolRunner.host_paths``), when it names any. Raise ValueError when a role's
    shape cannot be sized (``artifact_shape``, ``driver_shape``), for a drive of the measured
    baseline that does not come after its warm stage, and for a task that declares no drive bounds
    (``UnknownTask``)."""
    baseline = is_measured_baseline(artifact, catalog_key)
    if baseline and WARM_BASELINE not in after:
        raise ValueError(f"a drive of {catalog_key}'s measured baseline starts from the engine cache its warm stage "
                         f"{WARM_BASELINE} leaves, so it comes after it")
    bounds = drive_bounds(task)
    reserve = run.runner_memory_gib + seal_memory_gib
    art = artifact_shape(task, allocation, runner_memory_gib=reserve)
    trace_gib = math.ceil(trace_disk_bytes(bounds.trace_cap_bytes, bounds.trace_request_bytes) / 2**30)
    drv = driver_shape(task, allocation).model_copy(update={"disk_gib": trace_gib})
    overlaps = driver_overlaps(allocation)
    shared = driver_memory_shared(allocation)
    roles = concurrent([art, drv])
    data = copied_entries(("artifact", "driver"), provisioned, dev_files=dev_files)
    staging = STAGING_SEC + math.ceil(SEAL_PROBE_SEC)
    # A driver whose memory is oversubscribed on the server's adds none to the stage.
    memory = (art.memory_gib if shared else roles.memory_gib) + reserve
    shape = roles.model_copy(update={"cpus": art.cpus if overlaps else roles.cpus,
                                     "memory_gib": memory,
                                     "disk_gib": roles.disk_gib + extracted_disk_gib(artifact) + copies_gib(data),
                                     "time_sec": roles.time_sec + staging + TRACE_COPY_SEC})
    params = {"task": task_ref.model_dump(mode="json"), "artifact": artifact.model_dump(mode="json"),
              "harness_image": harness_image, "port": task.contract.port, "serve_command": list(serve_command),
              "network": task.network.model_dump(mode="json") if task.network is not None else None,
              "artifact_shape": art.model_dump(), "driver_shape": drv.model_dump(), "driver_overlaps": overlaps,
              "readiness_timeout_sec": READINESS_TIMEOUT_SEC, "request_timeout_sec": REQUEST_TIMEOUT_SEC,
              "drive_timeout_sec": drive_wall_sec(task), "staging_sec": staging, "runtime_start_sec": RUNTIME_START_SEC,
              "wall_margin_sec": DRIVE_MARGIN_SEC + TRACE_COPY_SEC, "reply_max_bytes": REPLY_MAX_BYTES,
              "trace_cap_bytes": bounds.trace_cap_bytes, "trace_request_bytes": bounds.trace_request_bytes,
              "measured_baseline": baseline, "engine_cache_owner": WARM_BASELINE if baseline else None,
              "data_copies": data}
    # The stage binds these from the site's sandbox stores; a task whose runner names none has no
    # such key.
    paths = task.agent.tool_runner.host_paths if task.agent is not None else ()
    if paths:
        params["sandbox_paths"] = [dataclasses.asdict(p) for p in paths]
    # Keyed only where the allocation oversubscribes the driver's memory.
    if shared:
        params["driver_memory_shared"] = True
    return StageDecl(name=name, kind="deploy_drive", shape=shape, params=params, after=tuple(after),
                     inputs=dict(inputs or {}))


def score_shape(task: Any, runner_memory_gib: int) -> ResourceShape:
    """The score stage's shape for *task*: ``SCORE_SHAPE``'s CPUs and time, and the memory of a
    role that reads the summary of a drive under the task's contract (``score_memory_gib``) with
    *runner_memory_gib* beside it. Raise ValueError for a contract the score stage cannot size."""
    memory = score_memory_gib(task.contract) + runner_memory_gib
    return SCORE_SHAPE.model_copy(update={"memory_gib": memory})


def grading_images(catalog_key: str, artifact: PackagedArtifact | ImageArtifact | PublishedArtifact, *,
                   rebuild: bool = False) -> list[ImageDecl]:
    """The images a grading of an artifact of the task the catalog knows as *catalog_key* runs
    on, each with what it builds on, parents first: the harness image, the reference image
    (``TaskImages.reference``), a measured *artifact*'s image, and with *rebuild* the submission
    base the package stage builds on. A published artifact's file is read by the drive stage
    alone, never imported here. Raise ValueError for a measured *artifact* that is not one of the
    images the task declares for itself."""
    images = catalog.task_images(catalog_key)
    needed = [images.harness, images.reference]
    if isinstance(artifact, ImageArtifact):
        if artifact.image not in images.measured.values():
            raise ValueError(f"the artifact image {artifact.image} is not one of {catalog_key}'s measured images")
        needed.append(artifact.image)
    if rebuild:
        needed.append(images.submission_base)
    return images_chain(images, needed)


def grading_tail(*, task_ref: TaskRef, task: Any, catalog_key: str,
                 artifact: PackagedArtifact | ImageArtifact | PublishedArtifact,
                 serve_command: Sequence[str], package_digest: str | None, eval_id: str,
                 allocation: EvaluationAllocation | None, run: RunSettings, seal_memory_gib: int,
                 kind: Literal["rollout", "grading"],
                 before: Sequence[StageDecl], build_stage: str | None = None, provisioned: Provisioned,
                 package_stage: str | None = None, built_arm: ArmRecord | None = None,
                 dev_files: Mapping[str, Mapping[str, str]] | None = None) -> list[StageDecl]:
    """The stages that grade one artifact of *task*, after the stages *before* (which hold
    ``provision`` and ``images``, and the package stage that built a packaged *artifact* of this
    instance):
    - for the task's measured baseline, ``warm_baseline``: the engine cache entry its drive starts
      from, found or built once (``warm_baseline_decl``);
    - ``deploy_drive``: the artifact in the task's evaluation *allocation*, or on the hardware the
      brief states, under the task's runtime network policy, with *serve_command* after ``--host``
      and ``--port``, driven by the task's own driver within its own wall; the driver holds the
      drive's trace to the cap the task declares and the replies to ``REPLY_MAX_BYTES`` each, and
      counts none of the task's measured baseline's (``drive_decl``). A packaged artifact is
      another instance's package manifest,
      pinned in the stage's inputs by *package_digest*, or this instance's package stage, which
      the drive then comes after; a published one is read off the shared filesystem by the stage
      itself;
    - ``verify`` on the reference image, for the verify time the *allocation* declares after its
      staging (``STAGING_SEC``); and ``score``, with the memory to read the summary of a drive
      (``score_shape``);
    - ``finalize`` of *kind*, its eval *eval_id*, recording *build_stage* and *package_stage*
      when given (and coming after them) and *built_arm*, the arm that built the graded package,
      whose route secrets it counts contestant-facing; for its fixed time (``FINALIZE_SHAPE``).
    The drive and verify stages hold their roles and ``run.runner_memory_gib`` beside them, and the
    disk of the images they extract and of the copies of their roles' data (the artifact's and the
    driver's; the reference's, its weights among them), by what *provisioned*, the provision record
    the cache holds (``hls.stages.provision.cached_provision``), states of them
    (``copied_entries``), which they declare (``data_copies``). Raise ValueError for a packaged
    artifact that names neither a pinned manifest with its digest nor a package stage among
    *before*, when a role's shape cannot be sized (``artifact_shape``, ``driver_shape``), for a
    measured baseline served on no GPU, whose engine cache no key names (``warm_baseline_decl``), and
    for a task that declares no drive bounds (``UnknownTask``)."""
    images = catalog.task_images(catalog_key)
    reference = images.reference
    inputs: dict[str, str] = {}
    drive_after = ["provision", "images"]
    if isinstance(artifact, PackagedArtifact):
        if artifact.manifest is not None:
            if package_digest is None:
                raise ValueError("a grading serves a published package: name its manifest and the manifest's digest")
            inputs[artifact.manifest] = package_digest
        elif artifact.stage not in {d.name for d in before}:
            raise ValueError(f"a grading serves a published package or this instance's {artifact.stage} stage, "
                             "which the plan does not hold: name its manifest and the manifest's digest")
        else:
            drive_after.append(artifact.stage)
    ref = task_ref.model_dump(mode="json")
    verify = reference_shape(task, allocation, runner_memory_gib=run.runner_memory_gib)
    verify_data = copied_entries(("reference",), provisioned)
    verify_shape = verify.model_copy(update={"memory_gib": verify.memory_gib + run.runner_memory_gib,
                                             "disk_gib": verify.disk_gib + KEPT_DISK_GIB + copies_gib(verify_data),
                                             "time_sec": verify.time_sec + STAGING_SEC})
    decls = []
    if is_measured_baseline(artifact, catalog_key):
        assert isinstance(artifact, ImageArtifact)
        decls.append(warm_baseline_decl(task_ref=task_ref, task=task, catalog_key=catalog_key, artifact=artifact,
                                        serve_command=serve_command, allocation=allocation, run=run,
                                        provisioned=provisioned, after=("provision", "images")))
        drive_after.append(WARM_BASELINE)
    decls += [
        drive_decl(name="deploy_drive", task_ref=task_ref, task=task, catalog_key=catalog_key, artifact=artifact,
                   harness_image=images.harness, serve_command=serve_command, allocation=allocation,
                   run=run, seal_memory_gib=seal_memory_gib, provisioned=provisioned, after=drive_after,
                   inputs=inputs, dev_files=dev_files),
        StageDecl(name="verify", kind="verify", shape=verify_shape, after=("provision", "images", "deploy_drive"),
                  params={"task": ref, "reference_image": reference, "reference_shape": verify.model_dump(),
                          "data_copies": verify_data, "eval_id": eval_id}),
        StageDecl(name="score", kind="score", shape=score_shape(task, run.runner_memory_gib),
                  params={"task": ref}, after=("deploy_drive", "verify")),
    ]
    finalize = {"task": ref, "kind": kind, "eval_id": eval_id, "score_stage": "score", "build_stage": build_stage,
                "package_stage": package_stage,
                "built_arm": built_arm.model_dump(mode="json") if built_arm is not None else None}
    recorded = tuple(s for s in (build_stage, package_stage) if s is not None)
    return [*decls, StageDecl(name="finalize", kind="finalize", shape=FINALIZE_SHAPE, params=finalize,
                              after=(*recorded, "deploy_drive", "verify", "score"))]


def grading_stages(*, task_ref: TaskRef, task: Any, catalog_key: str,
                   artifact: PackagedArtifact | ImageArtifact | PublishedArtifact,
                   serve_command: Sequence[str] = (), package_digest: str | None = None, eval_id: str,
                   allocation: EvaluationAllocation | None = None, run: RunSettings, seal_memory_gib: int,
                   held: HeldKeys, recipe_key: RecipeKey,
                   provisioned: Provisioned, rebuild_from: str | None = None, rebuild_digest: str | None = None,
                   producer: Any = None, stated_pip: PipLimits | None = None,
                   built_arm: ArmRecord | None = None,
                   dev_files: Mapping[str, Mapping[str, str]] | None = None) -> list[StageDecl]:
    """The stages of a grading of one artifact of *task*, the ``TASK`` of the module *task_ref*
    names with its overrides applied, which the image catalog knows as *catalog_key*, its eval
    *eval_id*:
    - ``provision``: the task's weights, rows and tool bundles, for the task as *task_ref*'s
      overrides make it;
    - ``images``: the harness and reference images, a measured artifact's image and, for a
      rebuild, the submission base, each with what it builds on (``grading_images``), on an
      evaluation's CPUs, its disk counting the entries *held* says the cache lacks
      (``images_stage``);
    - with *rebuild_from*, another instance's agent_build manifest pinned by *rebuild_digest*,
      ``source_scan``: its bundle read for every secret value at this checkout, then ``package``:
      the bundle (or its record that there is none) into a submission image on the submission
      base, under *producer*'s network, its pip steps held to *stated_pip*, the limits that
      build's brief stated (``AgentBuildParams.package_pip``), which the stage's shape must give
      (``PackageStage.check``), and which the drive then serves;
    - the tail (``grading_tail``) of kind ``grading``, recording *built_arm*, the arm that built
      the graded package, and the copies of its roles' data
      from *provisioned*, the provision record the cache holds.
    A packaged *artifact* is another instance's published package manifest pinned by
    *package_digest*, or with *rebuild_from* this instance's package stage. Raise ValueError when
    neither holds, for a rebuild without its digest, its producer or its stated pip limits, and
    as ``grading_images`` and ``grading_tail`` do."""
    rebuild = rebuild_from is not None
    if rebuild and (rebuild_digest is None or producer is None or stated_pip is None):
        raise ValueError("a rebuild packages another instance's agent_build manifest: name it, its digest, the "
                         "task's producer and the pip limits its brief stated")
    if rebuild and not (isinstance(artifact, PackagedArtifact) and artifact.manifest is None):
        raise ValueError("a rebuild serves this instance's package stage, not another artifact")
    if isinstance(artifact, PackagedArtifact) and artifact.manifest is None and not rebuild:
        raise ValueError("a grading serves a published package: name its manifest and the manifest's digest, or "
                         "package a rollout's bundle again with its agent_build manifest")
    chain = grading_images(catalog_key, artifact, rebuild=rebuild)
    decls = [
        *provision_stages(task_ref=task_ref, shape=provision_shape(task, runner_memory_gib=run.runner_memory_gib,
                                                                   time_sec=EVALUATE_PROVISION_SEC),
                          sandbox_stores=runner_stores(task)),
        images_stage(catalog_task=catalog_key, decls=chain, held=held, recipe_key=recipe_key,
                     cpus=EVALUATE_IMAGES_CPUS),
    ]
    if rebuild:
        assert rebuild_from is not None and rebuild_digest is not None and stated_pip is not None
        images = catalog.task_images(catalog_key)
        # The bundle is scanned at this checkout before the package stage runs any of its code.
        # The bundle was held to the writable disk cap of the task's builds, and is again here.
        cap = sandbox_disk_gib(task, producer)
        decls.append(StageDecl(name="source_scan", kind="source_scan",
                               shape=source_scan_shape(cap, runner_memory_gib=run.runner_memory_gib),
                               inputs={rebuild_from: rebuild_digest},
                               params={"bundle": None, "bundle_sha256": None, "build_manifest": rebuild_from,
                                       "disk_cap_gib": cap}))
        # The pip steps get the limits the build's brief stated, whatever this site's runner keeps;
        # the stage's check refuses a shape that gives less.
        decls.append(StageDecl(
            name="package", kind="package", shape=package_shape(cap),
            after=("images", "source_scan"), inputs={rebuild_from: rebuild_digest},
            params={"task": task.id, "submission_base": images.submission_base,
                    "constraints": images.submission_constraints, "images_stage": "images",
                    "network": producer_network(producer), "gpu_kind": submission_gpu_kind(task),
                    "server_filename": producer.server_rel_path, "bundle": None, "bundle_sha256": None,
                    "build_manifest": rebuild_from, "disk_cap_gib": cap, "pip": stated_pip.model_dump()}))
    tail = grading_tail(task_ref=task_ref, task=task, catalog_key=catalog_key, artifact=artifact,
                        serve_command=serve_command, package_digest=package_digest, eval_id=eval_id,
                        allocation=allocation, run=run, seal_memory_gib=seal_memory_gib, kind="grading",
                        before=decls, provisioned=provisioned,
                        package_stage="package" if rebuild else None, built_arm=built_arm, dev_files=dev_files)
    return [*decls, *tail]


def oracle_fill_stages(*, task_ref: TaskRef, task: Any, catalog_key: str, runner_memory_gib: int, held: HeldKeys,
                       recipe_key: RecipeKey,
                       provisioned: Provisioned, chunk: int = DEFAULT_CHUNK,
                       dev_reference: bool = False) -> list[StageDecl]:
    """The stages that fill the answers area with the reference answers of *task*, the ``TASK``
    of the module *task_ref* names, to its test split, on the image and the silicon its gradings
    verify on, the image catalog knowing the task as *catalog_key*:
    - ``provision``: the task's weights, rows and tool bundles, as an evaluation provisions them;
    - ``images``: the image the verify role runs on (the measured reference image when the
      catalog lists one, else the harness image) and what it builds on, on an evaluation's CPUs,
      its disk counting the entries *held* says the cache lacks (``images_stage``);
    - ``oracle_fill``, *chunk* rows at a time, for ``ORACLE_FILL_SEC`` after its staging
      (``STAGING_SEC``); holding the verify role's shape with *runner_memory_gib*
      (``run.runner_memory_gib``) beside it and the disk of the image it extracts and of the copies
      of its data, by what *provisioned*, the provision record the cache holds, states of them
      (``copied_entries``), which it declares (``data_copies``); with *dev_reference* it also
      produces the task's dev reference into the store's measured area (``hls.verify.dev_reference``)."""
    images = catalog.task_images(catalog_key)
    reference = images.reference
    chain = _chain(images.decls, reference)
    role = reference_shape(task, runner_memory_gib=runner_memory_gib)
    data = copied_entries(("oracle",), provisioned)
    shape = role.model_copy(update={"memory_gib": role.memory_gib + runner_memory_gib,
                                    "disk_gib": role.disk_gib + KEPT_DISK_GIB + copies_gib(data),
                                    "time_sec": ORACLE_FILL_SEC + STAGING_SEC})
    return [
        *provision_stages(task_ref=task_ref, shape=provision_shape(task, runner_memory_gib=runner_memory_gib,
                                                                   time_sec=EVALUATE_PROVISION_SEC),
                          sandbox_stores=runner_stores(task)),
        images_stage(catalog_task=catalog_key, decls=chain, held=held, recipe_key=recipe_key,
                     cpus=EVALUATE_IMAGES_CPUS),
        StageDecl(name="oracle_fill", kind="oracle_fill", shape=shape, after=("provision", "images"),
                  params={"task": task_ref.model_dump(mode="json"), "reference_image": reference,
                          "reference_shape": role.model_dump(), "data_copies": data, "chunk": chunk,
                          "dev_reference": dev_reference}),
    ]


#: The commit point's peak of anonymous memory, the program and the git it runs summed, in GiB,
#: when git reads the bases of the deltas it rebuilds one at a time: git 2.43.0 rebuilding a
#: reachable object stored as a delta, just under ``hls.agent.commit.DELTA_MAX_BYTES``, on a 4 GiB
#: base main does not reach, the largest object the program gives git (``OBJECT_MAX_BYTES``),
#: which it reads whole. Every other shape the bounds let through peaks lower: the most objects
#: (``OBJECTS_MAX``), trees near ``TREE_BYTES_MAX``, blobs git reads whole or streams, and the
#: program's own reads of the deepest delta chains. Anonymous memory is what a role's limit
#: holds: the pack's file pages git maps are reclaimed before an OOM, and the runner's watch
#: leaves them out.
ONE_BASE_PEAK_GIB = 5.003
#: The worst peak of the commit point's anonymous memory, in GiB, over the shapes its bounds let
#: through: git keeps the base it last rebuilt a delta from in its cache while it reads the next
#: delta's base, at any setting of the cache, so three 4 GiB blobs main does not reach, each with
#: a 512 MiB delta main reaches, held two bases and an object at once, 8.335 GiB (git 2.43.0),
#: against ``ONE_BASE_PEAK_GIB`` for one.
COMMIT_PEAK_GIB = 8.335
#: The commit point's shape: one CPU, ``COMMIT_PEAK_GIB`` and a quarter more, rounded up,
#: ceil(8.335 x 1.25) = 11 GiB, no GPU and no disk. ``build_stages`` gives it the build stage's
#: time. It runs once the sandbox has stopped, so the stage's shape holds it already.
COMMIT_SHAPE = ResourceShape(cpus=1, memory_gib=math.ceil(COMMIT_PEAK_GIB * 1.25), time_sec=3600)
#: The shape of each harness role beside a build's sandbox (a snapshot, a copy, the transcript
#: copy): one CPU, no GPU and no disk, and ``ONE_BASE_PEAK_GIB`` and a quarter more, rounded up,
#: ceil(5.003 x 1.25) = 7 GiB: a snapshot runs the commit point's program on the same workspace
#: main, and the copies and the transcript need no more. A snapshot of a repository whose deltas
#: hold two 4 GiB bases at once passes it, up to ``COMMIT_PEAK_GIB``, and its step is held at its
#: memory: it swaps where the node's swap is unlimited or is ended, either only an event,
#: since snapshots are evidence and the commit point bundles main (``COMMIT_SHAPE``).
#: ``build_stages`` gives it the build stage's time. A snapshot overlaps the sandbox, so the
#: sandbox's memory is the allocation less this and the runner's (``sandbox_shape``).
SNAPSHOT_SHAPE = ResourceShape(cpus=1, memory_gib=math.ceil(ONE_BASE_PEAK_GIB * 1.25), time_sec=3600)
#: The extraction role's shape (``hls.agent.extract``): one CPU and 4 GiB, its memory the limit on
#: what harbor's reading of the agent's logs may hold. The reading holds about 3.5 bytes a byte
#: of a codex session's logs, and less of any Claude Code session the transcript copy keeps. A
#: session too large for it ends the role, which the build records and survives. No GPU and no
#: disk; ``build_stages`` gives it the build stage's time. It runs once the sandbox has stopped,
#: so the stage's shape holds it already.
EXTRACT_SHAPE = ResourceShape(cpus=1, memory_gib=4, time_sec=3600)
#: The eval a rollout's own grading records.
ROLLOUT_EVAL = "terminal"
#: The hard limit of a probe or preflight-only build, in seconds: it starts the sandbox and stops,
#: so it gets a short fixed limit, where the task's agent budget would hold the allocation for
#: nothing.
PROBE_HARD_SEC = 1800


def submission_gpu_kind(task: Any) -> str | None:
    """The GPU kind a submission's CUDA extensions are compiled for: *task*'s accelerator when it
    has GPUs of a kind the planner knows, else None."""
    accelerator = task.hardware.accelerator
    return accelerator.kind if accelerator.count > 0 and accelerator.kind in GPU_KINDS else None


def producer_network(producer: Any) -> dict[str, Any]:
    """The producer's ``NetworkPolicy`` as JSON: what an agent's sandbox and a submission's pip
    steps may reach, and the package rule. A producer network of None means open egress
    (``AgentProducerSpec.network``); the seal is then open too, and no package rule applies, which
    the package and agent_build stages' checks refuse when the plan is made: the seal's index
    carries the producer's own deny list."""
    return (producer.network or NetworkPolicy()).model_dump(mode="json")


def sandbox_disk_gib(task: Any, producer: Any) -> int:
    """The writable disk cap of *task*'s build sandbox, in GiB, the limit on what its writable
    directories hold together and on the repository a build submits (its objects inflated and
    main's bundle, ``hls.agent.commit``): as much as the memory of the build allocation the
    producer declares (``build_allocation``), or without one the
    storage of the hardware the brief states (``artifact_shape``)."""
    allocation = producer.build_allocation
    return allocation.memory_gib if allocation is not None else task.hardware.storage_gib


def sandbox_shape(task: Any, producer: Any, time_sec: int, *, runner_memory_gib: int) -> ResourceShape:
    """The build sandbox's shape for *time_sec*, with *task*'s GPUs and its writable disk cap
    (``sandbox_disk_gib``). A producer that declares a build allocation (``build_allocation``)
    sizes the rest: every CPU of it, which the harness roles share;
    and its memory less the runner's (*runner_memory_gib*) and a harness role's
    (``SNAPSHOT_SHAPE``). Without one, the sandbox is the hardware the brief states
    (``artifact_shape``). Raise ValueError for an allocation that leaves the sandbox no memory."""
    shape = artifact_shape(task, runner_memory_gib=runner_memory_gib)
    allocation = producer.build_allocation
    if allocation is not None:
        memory = allocation.memory_gib - runner_memory_gib - SNAPSHOT_SHAPE.memory_gib
        if memory < 1:
            raise ValueError(f"the build allocation's {allocation.memory_gib} GiB leaves the sandbox no memory beside "
                             f"the runner's {runner_memory_gib} GiB and a harness role's {SNAPSHOT_SHAPE.memory_gib}")
        shape = shape.model_copy(update={"cpus": allocation.cpus, "memory_gib": memory})
    return shape.model_copy(update={"disk_gib": sandbox_disk_gib(task, producer), "time_sec": time_sec})


def _within(shape: ResourceShape, producer: Any) -> ResourceShape:
    """*shape*, for a stage of a build, with no more CPUs than the producer's build allocation
    has when it declares one, so that the build's allocation is the one it declares: fewer
    CPUs only slow a stage that runs alone."""
    allocation = producer.build_allocation
    return shape if allocation is None else shape.model_copy(update={"cpus": min(shape.cpus, allocation.cpus)})


def build_stages(*, task_ref: TaskRef, task: Any, producer: Any, arm: ArmRecord, arm_change: str | None,
                 catalog_key: str, runner_memory_gib: int, seal_memory_gib: int, collect_allowance_sec: int,
                 held: HeldKeys, recipe_key: RecipeKey,
                 provisioned: Provisioned,
                 budgets_override: Mapping[str, int] | None = None,
                 stop_after: Literal["probe", "preflight"] | None = None,
                 probe_oom: bool = False,
                 dev_files: Mapping[str, Mapping[str, str]] | None = None) -> list[StageDecl]:
    """The stages of a rollout that builds a submission for *task*, the ``TASK`` of the module
    *task_ref* names, with arm *arm*'s agent, and packages it; *producer* is the module's producer
    spec, and the image catalog knows the task as *catalog_key*:
    - ``provision``: the task's weights, rows and tool bundles, as an evaluation provisions them;
    - ``images``: the agent cap the sandbox runs on and the submission base, each with what it
      builds on, its disk counting the entries *held* says the cache lacks (``images_stage``);
    - ``agent_build``: the arm's agent in its sandbox (``sandbox_shape``), for the producer's
      budgets, under the producer's network policy, its brief stating what a graded server gets
      (``artifact_shape`` of the producer's evaluation allocation, as ``evaluate_stages`` gives
      the server); *arm_change* is the declared reason the arm
      differs from the task's earlier builds of it, if it does. The stage holds the sandbox and
      one harness role (``SNAPSHOT_SHAPE``), which runs on the sandbox's CPUs, *runner_memory_gib*
      (``run.runner_memory_gib``) and the sandbox's seal's *seal_memory_gib* beside them, and the disk of the agent cap's extracted tree and of
      the copies of the agent's data, by what *provisioned*, the provision record the cache holds
      (``hls.stages.provision.cached_provision``), states of them (``copied_entries``), which it
      declares (``data_copies``); the commit point (``COMMIT_SHAPE``) and the extraction
      (``EXTRACT_SHAPE``) run once the sandbox has stopped, within *collect_allowance_sec*
      (``run.collect_allowance_sec``) after the hard limit; before its setup, the stage's staging
      copies the agent's data and extracts the agent cap (``STAGING_SEC``). For a producer that
      runs the reference service (``AgentProducerSpec.reference_service``), the images stage also
      holds the task's measured baseline image, which the service runs on, and the stage holds the
      disk of that image's tree and of the service's workdir; the service's role
      (``hls.stages.reference_service.service_shape``) shares the sandbox's GPUs, CPUs and memory,
      so the stage's memory and the sandbox's are those of a build without it;
    - ``source_scan``: the build's source bundle read for every secret value, every object of it
      and its tree, published before any of the contestant's code runs again;
    - ``package``: the build's source bundle, or its record that there is none, into a
      submission image on the submission base.
    The commit point, the source scan and the package hold the repository and its bundle to the
    sandbox's writable disk cap, and the latter two are shaped for a bundle at it.
    No stage takes more CPUs than the producer's build allocation has (``_within``). The build
    stage's time is its staging and its allowances around the producer's hard limit, or around the
    wall budgets *budgets_override* gives (``soft_sec`` and ``hard_sec``). A probe or
    preflight-only build (*stop_after*, with *probe_oom* for the memory hog) runs ``PROBE_HARD_SEC``
    with no soft budget and no turn cap, on the same allocation, keeps only its short ending after
    it (``hls.stages.agent_build.collection_sec``), and has no source_scan or package stage: it
    publishes no bundle. Raise ValueError when the producer's budgets are ones the runner cannot
    keep (``budget_params``), when the build allocation leaves the sandbox no memory, and when a
    producer that runs the reference service has no baseline image or no provisioned weights."""
    images = catalog.task_images(catalog_key)
    needed = [images.sandbox, images.submission_base]
    service_image = None
    if producer.reference_service:
        # The service runs on the task's measured baseline image, and loads the task model's weights.
        if "baseline" not in images.measured:
            raise ValueError(f"{catalog_key}'s producer runs the reference service, which runs on the task's measured "
                             "baseline image, and its module declares none")
        if provisioned.weights is None:
            raise ValueError(f"{task.id}'s producer runs the reference service, which loads the task model's "
                             "weights, and none are provisioned")
        service_image = images.measured["baseline"]
        needed.append(service_image)
    chain = images_chain(images, needed)
    if stop_after is not None:
        budgets: dict[str, int | None] = {"soft_sec": None, "hard_sec": PROBE_HARD_SEC, "max_turns": None}
    else:
        budgets = {**budget_params(producer), **(budgets_override or {})}
    setup = AgentBuildParams.model_fields["setup_allowance_sec"].default
    data = copied_entries(("agent",), provisioned, dev_files=dev_files)
    hard_sec = budgets["hard_sec"]
    assert hard_sec is not None, "every build has a hard limit"
    time_sec = setup + STAGING_SEC + hard_sec + collection_sec(stop_after, collect_allowance_sec)
    # The sandbox, the package's pip steps and the served artifact are sealed: what they leave
    # holds the seal's memory with the runner's.
    sealed = runner_memory_gib + seal_memory_gib
    sandbox = sandbox_shape(task, producer, time_sec, runner_memory_gib=sealed)
    snapshot = SNAPSHOT_SHAPE.model_copy(update={"time_sec": time_sec})
    commit = COMMIT_SHAPE.model_copy(update={"time_sec": time_sec})
    extract = EXTRACT_SHAPE.model_copy(update={"time_sec": time_sec})
    roles = concurrent([sandbox, snapshot])
    stage = roles.model_copy(update={"cpus": sandbox.cpus, "memory_gib": roles.memory_gib + sealed,
                                     "disk_gib": roles.disk_gib + KEPT_DISK_GIB + copies_gib(data)})
    service = None
    if service_image is not None:
        assert provisioned.weights is not None
        service = ReferenceServiceParams(image=service_image,
                                         shape=service_shape(sandbox, provisioned.weights.disk_bytes))
        # The service shares the sandbox's GPUs, CPUs and memory (``ReferenceServiceParams``); its
        # image's tree and its workdir take disk of their own.
        stage = stage.model_copy(update={"disk_gib": stage.disk_gib + KEPT_DISK_GIB + service.shape.disk_gib})
    network = producer_network(producer)
    # The package stage's shape for a bundle at the sandbox's cap, whose pip limits the brief
    # states, a stop build's too.
    package = _within(package_shape(sandbox.disk_gib), producer)
    pip = pip_limits(package, sealed)
    build = AgentBuildParams(task=task_ref, arm=arm, arm_change=arm_change, sandbox_image=images.sandbox,
                             sandbox_held=catalog.held_versions(images), sandbox_shape=sandbox,
                             served_shape=artifact_shape(task, producer.evaluation_allocation,
                                                         runner_memory_gib=sealed),
                             snapshot_shape=snapshot, commit_shape=commit, extract_shape=extract,
                             setup_allowance_sec=setup, staging_sec=STAGING_SEC,
                             collect_allowance_sec=collect_allowance_sec, network=network, data_copies=data,
                             package_pip=pip, stop_after=stop_after, probe_oom=probe_oom,
                             reference_service=service, **budgets)
    images_decl = images_stage(catalog_task=catalog_key, decls=chain, held=held, recipe_key=recipe_key)
    provision = provision_shape(task, runner_memory_gib=runner_memory_gib, time_sec=EVALUATE_PROVISION_SEC)
    stages = [
        *provision_stages(task_ref=task_ref, shape=_within(provision, producer), sandbox_stores=runner_stores(task)),
        dataclasses.replace(images_decl, shape=_within(images_decl.shape, producer)),
        StageDecl(name="agent_build", kind="agent_build", shape=stage, params=build.model_dump(mode="json"),
                  after=("provision", "images")),
    ]
    # A probe or preflight-only build stops at the sandbox: it has no bundle to scan or package.
    if stop_after is None:
        stages += [
            StageDecl(name="source_scan", kind="source_scan",
                      shape=_within(source_scan_shape(sandbox.disk_gib, runner_memory_gib=runner_memory_gib),
                                    producer),
                      after=("agent_build",),
                      params={"bundle": None, "bundle_sha256": None, "disk_cap_gib": sandbox.disk_gib}),
            StageDecl(name="package", kind="package", shape=package,
                      after=("images", "agent_build", "source_scan"),
                      params={"task": task.id, "submission_base": images.submission_base,
                              "constraints": images.submission_constraints, "images_stage": "images",
                              "network": network, "gpu_kind": submission_gpu_kind(task),
                              "server_filename": producer.server_rel_path, "bundle": None, "bundle_sha256": None,
                              "disk_cap_gib": sandbox.disk_gib, "pip": pip.model_dump()}),
        ]
    return stages


def rollout_stages(*, task_ref: TaskRef, task: Any, producer: Any, serve_command: Sequence[str], arm: ArmRecord,
                   arm_change: str | None, catalog_key: str, run: RunSettings, seal_memory_gib: int,
                   allocation: EvaluationAllocation | None, held: HeldKeys, recipe_key: RecipeKey,
                   provisioned: Provisioned,
                   budgets_override: Mapping[str, int] | None = None,
                   stop_after: Literal["probe", "preflight"] | None = None, probe_oom: bool = False,
                   build_only: bool = False,
                   dev_files: Mapping[str, Mapping[str, str]] | None = None) -> list[StageDecl]:
    """The stages of ``hls run``: a rollout that builds a submission for *task* with arm *arm*'s
    agent and packages it (``build_stages``, with *budgets_override*, *stop_after* and *probe_oom*
    as it takes them, and the collection allowance *run* sets, ``run.collect_allowance_sec``), then
    grades this instance's package (``grading_tail`` of kind ``rollout``, its eval
    ``ROLLOUT_EVAL``, recording the agent_build and package stages), the images stage holding every image both halves run on, capped at the
    build allocation's CPUs (``_within``), its disk counting the entries *held* says the cache
    lacks (``images_stage``). Every stage that copies the task's data counts the copies' disk from
    *provisioned*, the provision record the cache holds.
    *serve_command* is the artifact's, after ``--host`` and ``--port``; *allocation* the task's
    evaluation allocation. With *build_only*, a finalize that records the rollout and grades nothing
    follows the package (``score_stage`` None); a probe or preflight-only build (*stop_after*) has
    neither. Raise ValueError for a build-only run that also stops, and as ``build_stages`` and
    ``grading_tail`` do: a run that grades its build needs the task's drive bounds
    (``UnknownTask``)."""
    if stop_after is not None and build_only:
        raise ValueError("a probe or preflight-only run stops at the sandbox; it cannot also be build-only")
    build = build_stages(task_ref=task_ref, task=task, producer=producer, arm=arm, arm_change=arm_change,
                         catalog_key=catalog_key, runner_memory_gib=run.runner_memory_gib,
                         seal_memory_gib=seal_memory_gib,
                         collect_allowance_sec=run.collect_allowance_sec, held=held, recipe_key=recipe_key,
                         provisioned=provisioned,
                         budgets_override=budgets_override, stop_after=stop_after, probe_oom=probe_oom,
                         dev_files=dev_files)
    if stop_after is not None:
        return build
    ref = task_ref.model_dump(mode="json")
    if build_only:
        params = {"task": ref, "kind": "rollout", "eval_id": ROLLOUT_EVAL, "score_stage": None,
                  "build_stage": AGENT_BUILD_STAGE, "package_stage": "package"}
        return [*build, StageDecl(name="finalize", kind="finalize", shape=FINALIZE_SHAPE, params=params,
                                  after=(AGENT_BUILD_STAGE, "package"))]
    images = catalog.task_images(catalog_key)
    needed = [images.sandbox, images.submission_base, images.harness, images.reference]
    # The in-build reference service runs on the task's measured baseline image, which the build
    # pins from this stage's images (``build_stages``, which refuses a task that declares none).
    if producer.reference_service:
        needed.append(images.measured["baseline"])
    chain = images_chain(images, needed)
    images_decl = images_stage(catalog_task=catalog_key, decls=chain, held=held, recipe_key=recipe_key)
    stages = [dataclasses.replace(images_decl, shape=_within(images_decl.shape, producer)) if d.name == "images" else d
              for d in build]
    tail = grading_tail(task_ref=task_ref, task=task, catalog_key=catalog_key, artifact=PackagedArtifact(),
                        serve_command=serve_command, package_digest=None, eval_id=ROLLOUT_EVAL, allocation=allocation,
                        run=run, seal_memory_gib=seal_memory_gib, kind="rollout", before=stages,
                        provisioned=provisioned,
                        build_stage=AGENT_BUILD_STAGE, package_stage="package", dev_files=dev_files)
    return [*stages, *tail]


#: Stage names of every pipeline, which ``[run.retry]`` tables are checked against. The images
#: and package stages' names are listed, not taken from their declarations, which import task
#: modules: ``hls status``, ``cancel`` and ``resume`` import this module, and a task module left
#: broken by an edit must not stop them. A rollout's, a grading's, an oracle fill's, a
#: credit's, a calibration's and an export's stages, and a provision's sandbox_images stage,
#: are listed too, since only a task or a command's arguments declare them; their numbered
#: stages match ``STAGE_NAME_PATTERNS``.
KNOWN_STAGE_NAMES = frozenset(
    d.name for decls in (noop_one(gpus=1, gpu_kind="H200", sleep_sec=0, stage_time_sec=1),
                         noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=1)) for d in decls) | {
    "images", "source_scan", "package", SANDBOX_IMAGES_STAGE, *ROLLOUT_STAGES, *GRADING_STAGES,
    *ORACLE_FILL_STAGES, *CREDIT_STAGE_NAMES, *CALIBRATION_STAGE_NAMES, *EXPORT_STAGE_NAMES}
