from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.artifact.artifact import EvaluationAllocation
from hls.core.model import AcceleratorSpec, Hardware, InterconnectSpec
from hls.drive.bounds import (REPLY_MAX_BYTES, drive_bounds, trace_disk_bytes, trace_event_bytes,
                              verify_peak_bytes)
from hls.measured import BASELINE_CREDIT_FILE, Found, MeasuredFile, pinned_key
from hls.pipeline import commands
from hls.pipeline.pipelines import (
    DRIVE_MARGIN_SEC,
    DRIVER_SHAPE,
    EVALUATE_IMAGES_CPUS,
    EVALUATE_PROVISION_SEC,
    GRADING_STAGES,
    FINALIZE_SHAPE,
    KNOWN_STAGE_NAMES,
    OPEN_DRIVE_SEC,
    PROVISION_ROLE_GIB,
    READINESS_TIMEOUT_SEC,
    REFERENCE_SHAPE,
    RUNTIME_START_SEC,
    SCORE_SHAPE,
    STAGING_SEC,
    TRACE_COPY_SEC,
    artifact_shape,
    drive_time,
    driver_shape,
    grading_stages,
    images_shape,
    provision_shape,
    reference_shape,
    score_shape,
)
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.preflight import OracleCoverage
from hls.provision.mounts import copied_entries, copies_gib
from hls.pipeline.planner import PlanError, StageDecl, launch_headroom_sec
from hls.provision.model import Provisioned
from hls.provision.sandbox_runtime import RUNTIME_TAG
from hls.recipes import catalog
from hls.recipes.model import order
from hls.settings import RunSettings, SiteSettings, load_settings
from hls.spec.resources import ResourceShape, concurrent
from hls.stages.baseline_warm import BaselineWarmParams
from hls.stages.deploy_drive import SEAL_PROBE_SEC, DeployDriveParams, ImageArtifact, PackagedArtifact
from hls.stages.finalize import FINALIZE_SEC, index_key
from hls.stages.images import IMAGE_FILE, image_keys
from hls.stages.score import score_memory_gib
from hls.store.cache import ContentCache
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.substrate.seal import INGRESS_WAIT_SEC
from hls.taskref import TaskRef, load_task
from hls.recipes.render import recipe_key as def_key
from hls.substrate.apptainer.images import ApptainerImages
from tests.agent.test_hls_run_build import _text
from tests.evaluate import fixture
from tests.evaluate.warmer import WARMER, publish_package
from tests.pipeline.plans import CODE, pin_harbor
from tests.provision.cached import provision_into, provision_module, provision_task
from tests.provision.warmer_config import (SECRETS, fill_sandbox_stores, sandbox_store, task_sandbox_stores,
                                         write_config)
from tests.store.entries import populate

LAYER = catalog.image_set("sample_task").layer
#: The catalog's reader of a task module's images, as the fixtures find it before they declare
#: the fixture task's.
DECLARED_IMAGES = catalog.image_set
STAGES = ["provision", "images", "deploy_drive", "verify", "score", "finalize"]
#: The memory the test site leaves to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4
#: The disk the drive stage of a package holds for the two images it extracts, 40 GiB each and
#: twice the 28 GiB a package adds to its submission image (``extracted_disk_gib``), and verify
#: for its one.
DRIVE_TREES_GIB, VERIFY_TREE_GIB = 136, 40
#: The fixture task's trace cap and the request bytes its drives' traces may hold, as it
#: declares them.
CAP, REQUESTS = fixture.DRIVE_BOUNDS.trace_cap_bytes, fixture.DRIVE_BOUNDS.trace_request_bytes
#: The drive's roles' time for an open drive (``drive_time``): the server's runtime's start, two
#: readiness budgets, the open wall and the margin after it.
OPEN_TIME = RUNTIME_START_SEC + 2 * READINESS_TIMEOUT_SEC + OPEN_DRIVE_SEC + DRIVE_MARGIN_SEC
#: The fixture package warmer's submission image.
SUBMISSION_IMAGE = b"submission image"


#: The sandbox stores the tasks of the checkout name, which the test site names, so a drive of any
#: of them plans.
SANDBOX_STORES = tuple(dict.fromkeys(store for key in catalog.task_keys()
                                     for store in task_sandbox_stores(catalog.task_module(key))))


def _config(tmp_path: Path, resources: str = "", **kw) -> Path:
    """The warmer tests' hls.toml with *resources* (``[resources]`` tables) after it, a secrets
    file, which finalize scans for, and every task's sandbox stores (``SANDBOX_STORES``), unless
    *kw* gives others."""
    path = write_config(tmp_path, **{"secrets": SECRETS, "sandbox_stores": SANDBOX_STORES, **kw})
    path.write_text(path.read_text() + resources)
    return path


def _plan(tmp_path: Path, *, module: str = "ev_task.task", package: str | None = None, measured: str | None = None,
          config: Path | None = None, overrides: dict[str, str] | None = None) -> Plan:
    """The plan of a grading of *module*'s task, its provisioning in the cache and its sandbox
    stores first."""
    provision_module(tmp_path / "store", module)
    stores = task_sandbox_stores(module)
    if stores:
        fill_sandbox_stores({name: sandbox_store(tmp_path, name) for name in stores}, module, tmp_path / "store")
    return commands.plan_grade(config=config or _config(tmp_path), overrides=overrides or {}, task_module=module,
                               package=package, measured=measured, job_name="hls-grade", echo=[].append)


def _packaged(tmp_path: Path, task: str = fixture.TASK_ID) -> str:
    """The package warmer of *task* in the test's store, published on first use."""
    store = PosixStore(tmp_path / "store")
    warmer = InstanceLayout.warmer(task, WARMER)
    if store.exists(warmer.manifest("package")):
        return warmer.root
    return publish_package(store, SUBMISSION_IMAGE, task=task)


def _with_runner(shape: ResourceShape, *, disk_gib: int) -> ResourceShape:
    """*shape* with the runner's memory and *disk_gib* more disk, as a stage of roles declares it."""
    return shape.model_copy(update={"memory_gib": shape.memory_gib + RUNNER_GIB,
                                    "disk_gib": shape.disk_gib + disk_gib})


#: Seconds a site gives each role's step to start when it sets none
#: (``site.step_launch_allowance_sec``), as these tests' sites do.
ALLOWANCE = SiteSettings.model_fields["step_launch_allowance_sec"].default
#: The disk the fixture task's driver holds for its trace's room, in whole GiB: its 1 GiB cap
#: and 1 MiB of requests make 1,074,790,400 bytes of events, which the room holds in 17 chunks
#: and again in the summary at gzip's bound, 2,150,237,314 bytes (``trace_disk_bytes``), so 3 GiB.
TRACE_GIB = 3
#: The staging every drive stage is given before its server starts: the fixed allowance for its
#: copies and extractions, and its seal probe.
DRIVE_STAGING = STAGING_SEC + math.ceil(SEAL_PROBE_SEC)


def _headroom(plan: Plan) -> int:
    """The launch headroom *plan*'s one allocation asks for beside its stages' times."""
    return launch_headroom_sec(plan.stages, ALLOWANCE)


def _task(plan: Plan):
    """The task *plan* grades, as its finalize stage's reference makes it."""
    return load_task(TaskRef.model_validate(plan.stage("finalize").params["task"]))


def _cap(module: str) -> int:
    """The trace cap the task of *module* declares."""
    return drive_bounds(load_task(TaskRef(module=module))).trace_cap_bytes


def test_evaluate_plans_the_whole_path_in_one_gpu_allocation(tmp_path, login, in_catalog):
    package = _packaged(tmp_path)
    plan = _plan(tmp_path, package=package)
    grading = plan.instance.removeprefix(f"{package}/gradings/")
    assert grading.startswith("g-") and "/" not in grading and plan.pipeline == "grading"
    [alloc] = plan.allocations
    # A grading packages nothing unless it rebuilds a rollout's source, and warms no engine cache
    # unless it serves the measured baseline.
    assert alloc.stages == STAGES == [s for s in GRADING_STAGES if s not in ("source_scan", "package",
                                                                             "warm_baseline")]
    assert alloc.placement.partition is None and set(GRADING_STAGES) <= KNOWN_STAGE_NAMES
    task = fixture.evaluate_task()
    # A task with no evaluation allocation serves on the brief's hardware, its storage the
    # artifact's disk cap, each decimal GB figure in whole GiB rounded down (4 GB is 3.7 GiB, 2 GB
    # 1.9); the driver and the reference take the defaults.
    assert (task.hardware.memory_gb, task.hardware.storage_gb) == (4, 2)
    art = ResourceShape(gpus=1, gpu_kind="B200", cpus=2, memory_gib=3, disk_gib=1, time_sec=OPEN_TIME)
    drv = ResourceShape(cpus=4, memory_gib=16, time_sec=OPEN_TIME)
    assert drv == DRIVER_SHAPE
    assert (artifact_shape(task, runner_memory_gib=RUNNER_GIB), driver_shape(task), drive_time(task)) == (
        art, drv, OPEN_TIME)
    # The drive's driver holds its trace on the node's disk at its most, and the stage keeps a fixed
    # allowance after the wall to copy it to the store.
    assert math.ceil(trace_disk_bytes(CAP, REQUESTS) / 2**30) == TRACE_GIB
    traced = drv.model_copy(update={"disk_gib": TRACE_GIB})
    # The time allowed for the server's runtime to start is the seal's start the stage waits for.
    assert RUNTIME_START_SEC == math.ceil(INGRESS_WAIT_SEC) == 300
    # The reference job's 8 CPUs and 180 GiB, the runner's reserve among them.
    ref = ResourceShape(gpus=1, gpu_kind="B200", cpus=8, memory_gib=180 - RUNNER_GIB, time_sec=2 * 3600)
    assert reference_shape(task, runner_memory_gib=RUNNER_GIB) == ref and REFERENCE_SHAPE.memory_gib == 180
    # A task that declares no provision memory gives each provision role the default, with the
    # runner's reserve beside it, for an evaluation's two hours.
    provision = ResourceShape(cpus=4, memory_gib=PROVISION_ROLE_GIB + RUNNER_GIB, time_sec=2 * 3600)
    assert provision_shape(task, runner_memory_gib=RUNNER_GIB, time_sec=EVALUATE_PROVISION_SEC) == provision
    shapes = {s.name: s.shape for s in plan.stages}
    # The drive and verify stages hold their roles, the runner's memory and the trees of the
    # images they extract (two for the drive, one for verify). Finalize has its fixed time.
    assert FINALIZE_SHAPE.time_sec == FINALIZE_SEC
    # The drive's and verify's disk also hold the copies of their roles' data, by what the
    # cache's provision record states of them.
    provisioned = provision_module(tmp_path / "store", "ev_task.task")
    copies = copied_entries(("artifact", "driver"), provisioned)
    ref_copies = copied_entries(("reference",), provisioned)
    assert plan.stage("deploy_drive").params["data_copies"] == copies
    assert plan.stage("verify").params["data_copies"] == ref_copies
    data, ref_data = copies_gib(copies), copies_gib(ref_copies)
    assert data == ref_data == 1
    # The drive's time holds its staging and, after the margin, the trace's copy; verify's holds the
    # staging. The images stage fills the two images the cache lacks, each in its fixed time.
    drive = _with_runner(concurrent([art, traced]), disk_gib=DRIVE_TREES_GIB + data)
    assert shapes == {"provision": provision, "images": images_shape(2, cpus=EVALUATE_IMAGES_CPUS),
                      "deploy_drive": drive.model_copy(update={"time_sec": OPEN_TIME + DRIVE_STAGING + TRACE_COPY_SEC}),
                      "verify": _with_runner(ref, disk_gib=VERIFY_TREE_GIB + ref_data).model_copy(
                          update={"time_sec": ref.time_sec + STAGING_SEC}), "score": SCORE_SHAPE,
                      "finalize": FINALIZE_SHAPE}
    assert shapes["images"].time_sec == 900 + 2 * 1800
    # The drive records the trace cap and request bytes its task declares.
    bounds = drive_bounds(_task(plan))
    assert (plan.stage("deploy_drive").params["trace_cap_bytes"],
            plan.stage("deploy_drive").params["trace_request_bytes"]) == (bounds.trace_cap_bytes,
                                                                          bounds.trace_request_bytes)
    # The images stage fills two entries one at a time, a build and a tree; the drive's two trees,
    # the artifact's disk, the trace's and the copies of its roles' data are more.
    assert (alloc.request.gpus, alloc.request.gpu_kind, alloc.request.cpus, alloc.request.memory_gib,
            alloc.request.disk_gib) == (1, "B200", 8, 180, art.disk_gib + TRACE_GIB + DRIVE_TREES_GIB + data)
    assert alloc.request.time_sec == (7200 + shapes["images"].time_sec + OPEN_TIME + DRIVE_STAGING + TRACE_COPY_SEC
                                      + 7200 + STAGING_SEC + 1800 + FINALIZE_SEC + 360 + _headroom(plan))
    after = {s.name: s.after for s in plan.stages}
    assert after == {"provision": [], "images": [], "deploy_drive": ["provision", "images"],
                     "verify": ["provision", "images", "deploy_drive"], "score": ["deploy_drive", "verify"],
                     "finalize": ["deploy_drive", "verify", "score"]}
    # The package manifest of the warmer is pinned by digest; nothing else of another instance.
    key = InstanceLayout(package).manifest("package")
    digest = hashlib.sha256(PosixStore(tmp_path / "store").read_bytes(key)).hexdigest()
    assert {s.name: s.inputs for s in plan.stages if s.inputs} == {"deploy_drive": {key: digest}}
    ref_task = {"module": "ev_task.task", "overrides": {}}
    images = plan.stage("images").params
    assert ([d["name"] for d in images["images"]], images["catalog_task"]) == (["s-a", LAYER], fixture.TASK_ID)
    assert plan.stage("provision").params == {"task": {"module": "ev_task.task", "overrides": {}}, "weights": True}
    # Images are named as the images stage's images.json names them.
    assert plan.stage("deploy_drive").params == {
        "task": ref_task, "artifact": {"kind": "package", "stage": "package", "manifest": key},
        "harness_image": LAYER, "images_stage": "images", "provision_stage": "provision", "port": 8000,
        "serve_command": [], "network": None, "artifact_shape": art.model_dump(),
        "driver_shape": traced.model_dump(), "driver_overlaps": False,
        "readiness_timeout_sec": READINESS_TIMEOUT_SEC, "request_timeout_sec": 10800.0,
        "drive_timeout_sec": float(OPEN_DRIVE_SEC), "staging_sec": DRIVE_STAGING,
        "runtime_start_sec": RUNTIME_START_SEC, "split": "test",
        "wall_margin_sec": DRIVE_MARGIN_SEC + TRACE_COPY_SEC, "reply_max_bytes": REPLY_MAX_BYTES,
        "trace_cap_bytes": CAP, "trace_request_bytes": REQUESTS, "measured_baseline": False,
        "engine_cache_owner": None,
        "data_copies": copies}
    assert plan.stage("verify").params == {
        "task": ref_task, "reference_image": LAYER, "drive_stage": "deploy_drive", "images_stage": "images",
        "provision_stage": "provision", "reference_shape": ref.model_dump(), "data_copies": ref_copies,
        "eval_id": grading}
    assert plan.stage("score").params == {"task": ref_task, "drive_stage": "deploy_drive", "verify_stage": "verify"}
    # An evaluation records no build and names no building arm, so it counts no secret facing.
    assert plan.stage("finalize").params == {"task": ref_task, "eval_id": grading, "kind": "grading",
                                             "score_stage": "score", "build_stage": None, "package_stage": None,
                                             "built_arm": None}
    # The CPU stages hold the allocation's GPU too, and the plan says what that costs.
    idle = [w for w in plan.warnings if "idle" in w]
    assert [w.split("; stage ")[1].split(" ")[0] for w in idle] == ["provision", "images", "score", "finalize"]
    assert idle[0] == ("allocation 0 holds 1 GPUs; stage provision uses 0, so 1 idle for up to 2.00 h (2.00 "
                       "GPU-hours). --run.gpu_policy=split asks for each stage's own count instead")


def test_the_drive_and_verify_stages_leave_the_runner_the_memory_the_site_sets(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, package=_packaged(tmp_path), overrides={"run.runner_memory_gib": "6"})
    task = fixture.evaluate_task()
    assert plan.stage("deploy_drive").shape.memory_gib == artifact_shape(task, runner_memory_gib=6).memory_gib + 16 + 6
    assert plan.stage("verify").shape.memory_gib == reference_shape(task, runner_memory_gib=6).memory_gib + 6


#: Qwen3-Omni-30B-A3B's bf16 weights, about 70.5 GB: the largest a graded task's drive copies.
OMNI_WEIGHTS_BYTES = 70_500_000_000


def _grading_decls(tmp_path: Path, provisioned: Provisioned | None = None) -> dict[str, StageDecl]:
    """The stages of a grading of the fixture task's pinned package warmer, by name, its data
    copied as *provisioned* states it, else as the task's own provision record does."""
    task = fixture.evaluate_task()
    if provisioned is None:
        provisioned = provision_into(tmp_path / "store", task)
    pinned = InstanceLayout.warmer(fixture.TASK_ID, WARMER).manifest("package")
    decls = grading_stages(task_ref=TaskRef(module="ev_task.task"), task=task, catalog_key=fixture.TASK_ID,
                           artifact=PackagedArtifact(manifest=pinned), package_digest="0" * 64, eval_id="g1",
                           run=RunSettings(), seal_memory_gib=0, held=lambda keys: set(), recipe_key=def_key,
                           provisioned=provisioned)
    return {d.name: d for d in decls}


def test_omni_sized_weights_grow_the_drive_and_verify_disk_and_not_their_staging(tmp_path, in_catalog):
    """The drive and verify stages hold the copies of their roles' data in their disk, and give
    their staging the same fixed allowance whatever the copies hold."""
    task = fixture.evaluate_task()
    provisioned = provision_into(tmp_path / "store", task)
    assert provisioned.weights is not None
    omni = provisioned.model_copy(update={"weights": provisioned.weights.model_copy(
        update={"disk_bytes": OMNI_WEIGHTS_BYTES})})
    stages = _grading_decls(tmp_path, omni)
    drive, verify = stages["deploy_drive"], stages["verify"]
    copies, ref_copies = drive.params["data_copies"], verify.params["data_copies"]
    assert min(sum(copies.values()), sum(ref_copies.values())) >= OMNI_WEIGHTS_BYTES
    assert min(copies_gib(copies), copies_gib(ref_copies)) >= math.ceil(OMNI_WEIGHTS_BYTES / 2**30)
    assert drive.shape.disk_gib == (artifact_shape(task, runner_memory_gib=RUNNER_GIB).disk_gib + TRACE_GIB
                                    + DRIVE_TREES_GIB + copies_gib(copies))
    assert drive.params["staging_sec"] == DRIVE_STAGING
    assert drive.shape.time_sec == drive_time(task) + DRIVE_STAGING + TRACE_COPY_SEC
    role = reference_shape(task, runner_memory_gib=RunSettings().runner_memory_gib)
    assert verify.params["reference_shape"] == role.model_dump()
    assert verify.shape.disk_gib == role.disk_gib + VERIFY_TREE_GIB + copies_gib(ref_copies)
    assert verify.shape.time_sec == role.time_sec + STAGING_SEC


def test_a_drive_keeps_a_fixed_allowance_to_copy_its_trace_to_the_store(tmp_path, in_catalog):
    """The drive role writes the trace into its room on the node's disk as the drive goes and
    publishes it to the store after it: the driver's disk holds the room (``trace_disk_bytes``),
    and the stage's margin after the wall, and so its time, the copy's fixed allowance
    (``TRACE_COPY_SEC``)."""
    task = fixture.evaluate_task()
    drive = _grading_decls(tmp_path)["deploy_drive"]
    params = drive.params
    assert params["wall_margin_sec"] == DRIVE_MARGIN_SEC + TRACE_COPY_SEC
    assert drive.shape.time_sec == drive_time(task) + DRIVE_STAGING + TRACE_COPY_SEC
    # That is the stage's floor: the staging, the runtime's start, two readiness budgets, the wall
    # and the margin after it.
    assert drive.shape.time_sec == (params["staging_sec"] + params["runtime_start_sec"]
                                    + 2 * params["readiness_timeout_sec"] + params["drive_timeout_sec"]
                                    + params["wall_margin_sec"])
    # The driver declares the trace's disk, which the stage holds beside the artifact's, the two
    # trees it extracts and the copies of its roles' data.
    assert params["driver_shape"] == driver_shape(task).model_copy(update={"disk_gib": TRACE_GIB}).model_dump()
    assert drive.shape.disk_gib == (artifact_shape(task, runner_memory_gib=RUNNER_GIB).disk_gib + TRACE_GIB
                                    + DRIVE_TREES_GIB + copies_gib(params["data_copies"]))


def test_a_package_is_served_with_the_serve_command_its_task_module_declares(tmp_path, login, in_catalog):
    drive = _plan(tmp_path, module="ev_art.task", package=_packaged(tmp_path)).stage("deploy_drive")
    assert drive.params["serve_command"] == fixture.ARTIFACT_SERVE


def test_a_task_with_its_own_wall_and_a_gpu_tool_sizes_the_drive_from_them(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, module="ev_agent.task", package=_packaged(tmp_path))
    task = fixture.agent_task()
    time = RUNTIME_START_SEC + 2 * READINESS_TIMEOUT_SEC + math.ceil(fixture.AGENT_WALL_SEC) + DRIVE_MARGIN_SEC
    assert drive_time(task) == time
    assert driver_shape(task) == ResourceShape(gpus=1, gpu_kind="B200", cpus=4, memory_gib=40, time_sec=time)
    drive = plan.stage("deploy_drive")
    assert drive.params["drive_timeout_sec"] == fixture.AGENT_WALL_SEC
    assert (drive.shape.gpus, drive.shape.time_sec) == (2, time + DRIVE_STAGING + TRACE_COPY_SEC)


def test_the_artifact_runs_under_the_tasks_runtime_policy_on_its_contracts_port(tmp_path, login, in_catalog):
    drive = _plan(tmp_path, module="ev_net.task", package=_packaged(tmp_path)).stage("deploy_drive")
    assert (drive.params["network"], drive.params["port"]) == (fixture.NETWORK.model_dump(mode="json"),
                                                               fixture.NET_PORT)


def test_the_verify_role_holds_the_widest_reference_engine(tmp_path, login, in_catalog):
    wide = ResourceShape(gpus=2, gpu_kind="B200", cpus=8, memory_gib=200, time_sec=2 * 3600)
    assert reference_shape(fixture.wide_task(), runner_memory_gib=RUNNER_GIB) == wide
    plan = _plan(tmp_path, module="ev_wide.task", package=_packaged(tmp_path))
    assert plan.stage("verify").shape == _with_runner(
        wide, disk_gib=VERIFY_TREE_GIB + copies_gib(plan.stage("verify").params["data_copies"])).model_copy(
        update={"time_sec": wide.time_sec + STAGING_SEC})
    assert plan.allocations[0].request.gpus == 2


def test_an_evaluation_allocation_sizes_the_artifact_and_its_driver(tmp_path, login, in_catalog):
    """With ``driver_cpus`` None the driver overlaps the artifact's CPUs; with it, the driver
    has CPUs of its own beside them."""
    alloc = fixture.ALLOCATION
    art = ResourceShape(gpus=1, gpu_kind="B200", cpus=3, memory_gib=6, disk_gib=5, time_sec=OPEN_TIME)
    task = fixture.evaluate_task()
    assert artifact_shape(task, alloc, runner_memory_gib=RUNNER_GIB) == art
    plan = _plan(tmp_path, module="ev_alloc.task", package=_packaged(tmp_path))
    drive = plan.stage("deploy_drive")
    drv = ResourceShape(cpus=3, memory_gib=3, time_sec=OPEN_TIME)
    assert driver_shape(task, alloc) == drv
    assert (drive.params["artifact_shape"], drive.params["driver_shape"], drive.params["driver_overlaps"]) == (
        art.model_dump(), drv.model_copy(update={"disk_gib": TRACE_GIB}).model_dump(), True)
    data = copies_gib(drive.params["data_copies"])
    assert drive.shape == ResourceShape(gpus=1, gpu_kind="B200", cpus=3, memory_gib=6 + 3 + RUNNER_GIB,
                                        disk_gib=5 + TRACE_GIB + DRIVE_TREES_GIB + data,
                                        time_sec=OPEN_TIME + DRIVE_STAGING + TRACE_COPY_SEC)
    own = _plan(tmp_path, module="ev_alloc_own.task", package=_packaged(tmp_path)).stage("deploy_drive")
    assert (own.params["driver_shape"]["cpus"], own.params["driver_overlaps"], own.shape.cpus) == (2, False, 5)


def test_a_driver_that_encodes_on_a_gpu_needs_cpus_of_its_own(tmp_path, login, in_catalog):
    # An overlapping driver's step shares the artifact's, so it can hold no GPU of its own.
    with pytest.raises(ValueError, match="its driver's tool encodes on a GPU, so the driver needs CPUs of its own"):
        _plan(tmp_path, module="ev_gpu_alloc.task", package=_packaged(tmp_path))


def test_the_verify_stage_runs_for_the_time_the_tasks_evaluation_allocation_declares(tmp_path, login, in_catalog):
    """The verify role takes the allocation's ``verify_time_sec`` and its stage that after its
    staging, and the allocation asks for it among the stages' times."""
    task, verify_sec = fixture.evaluate_task(), fixture.VERIFY_TIME_SEC
    allocation = fixture.ALLOCATION.model_copy(update={"verify_time_sec": verify_sec})
    ref = ResourceShape(gpus=1, gpu_kind="B200", cpus=8, memory_gib=180 - RUNNER_GIB, time_sec=verify_sec)
    assert reference_shape(task, allocation, runner_memory_gib=RUNNER_GIB) == ref
    plan = _plan(tmp_path, module="ev_alloc_verify.task", package=_packaged(tmp_path))
    verify = plan.stage("verify")
    assert (verify.shape, verify.shape_overridden) == (
        _with_runner(ref, disk_gib=VERIFY_TREE_GIB + copies_gib(verify.params["data_copies"])).model_copy(
            update={"time_sec": verify_sec + STAGING_SEC}), False)
    assert verify.params["reference_shape"] == ref.model_dump()
    [alloc] = plan.allocations
    assert alloc.request.time_sec == (7200 + plan.stage("images").shape.time_sec + OPEN_TIME + DRIVE_STAGING
                                      + TRACE_COPY_SEC + verify_sec + STAGING_SEC + 1800 + FINALIZE_SEC + 360
                                      + _headroom(plan))


def test_the_verify_role_holds_the_memory_the_tasks_evaluation_allocation_declares(tmp_path, login, in_catalog):
    """The verify role takes the allocation's ``verify_memory_gib`` in place of the reference
    job's memory less the runner's reserve, and its stage holds the runner's reserve beside it."""
    task, verify_gib = fixture.evaluate_task(), fixture.VERIFY_MEMORY_GIB
    allocation = fixture.ALLOCATION.model_copy(update={"verify_memory_gib": verify_gib})
    ref = ResourceShape(gpus=1, gpu_kind="B200", cpus=8, memory_gib=verify_gib, time_sec=2 * 3600)
    assert reference_shape(task, allocation, runner_memory_gib=RUNNER_GIB) == ref
    # The reserve is the stage's: a larger one leaves the declared role its memory.
    assert reference_shape(task, allocation, runner_memory_gib=RUNNER_GIB + 2) == ref
    plan = _plan(tmp_path, module="ev_alloc_memory.task", package=_packaged(tmp_path))
    verify = plan.stage("verify")
    assert (verify.shape, verify.shape_overridden) == (
        _with_runner(ref, disk_gib=VERIFY_TREE_GIB + copies_gib(verify.params["data_copies"])).model_copy(
            update={"time_sec": 2 * 3600 + STAGING_SEC}), False)
    assert verify.shape.memory_gib == verify_gib + RUNNER_GIB
    assert verify.params["reference_shape"] == ref.model_dump()


def test_a_declared_verify_memory_under_the_tasks_verify_peak_is_raised_to_it():
    # The role must hold what the task's verifiers keep of a trace at its cap, whatever the
    # allocation declares: 1 GiB of trace and its requests at a byte held per byte, 2 GiB.
    task = fixture.evaluate_task()
    bounds = task.drive_bounds
    peak = math.ceil(verify_peak_bytes(bounds.verify_peak,
                                       trace_event_bytes(bounds.trace_cap_bytes, bounds.trace_request_bytes)) / 2**30)
    assert peak == 2
    under = fixture.ALLOCATION.model_copy(update={"verify_memory_gib": 1})
    assert reference_shape(task, under, runner_memory_gib=RUNNER_GIB).memory_gib == peak


@pytest.mark.parametrize("module", ["ev_task.task", "ev_alloc.task"])
def test_the_verify_role_holds_the_reference_jobs_memory_when_the_task_declares_none(module):
    # No evaluation allocation, or one that names no verify memory.
    assert REFERENCE_SHAPE.memory_gib == 180 and fixture.ALLOCATION.verify_memory_gib is None
    allocation = fixture.ALLOCATION if module == "ev_alloc.task" else None
    assert reference_shape(fixture.evaluate_task(), allocation,
                           runner_memory_gib=RUNNER_GIB).memory_gib == 180 - RUNNER_GIB


@pytest.mark.parametrize("module", ["ev_task.task", "ev_alloc.task"])
def test_the_verify_stage_runs_for_two_hours_when_the_task_declares_no_time(tmp_path, login, in_catalog, module):
    # No evaluation allocation, or one that names no verify time.
    assert REFERENCE_SHAPE.time_sec == 2 * 3600 and fixture.ALLOCATION.verify_time_sec is None
    assert reference_shape(fixture.evaluate_task(), fixture.ALLOCATION,
                           runner_memory_gib=RUNNER_GIB).time_sec == 2 * 3600
    plan = _plan(tmp_path, module=module, package=_packaged(tmp_path))
    verify = plan.stage("verify")
    assert (verify.shape.time_sec, verify.params["reference_shape"]["time_sec"]) == (
        2 * 3600 + STAGING_SEC, 2 * 3600)


@pytest.fixture
def repo_login(monkeypatch, tmp_path, warmed_servers):
    """The login node planning from this repository's own checkout and image catalog, on a
    cluster whose images warmers built every server image (``warmed_servers``). The plan
    names a stand-in checkout, which is no directory, so a developer's own secrets file in this
    one does not refuse it (``build_plan``). The store holds no oracle fill, and the checks that
    refuse a grading without its answers or with an uncommitted blob have tests of their own: here
    they find nothing to check."""
    monkeypatch.setattr(commands, "code_state", lambda path: CODE)
    pin_harbor(monkeypatch, CODE)
    monkeypatch.setattr(commands, "check_defaults_committed", lambda checkout, files: None)
    monkeypatch.setattr(commands, "check_oracle_coverage", lambda store, **kw: None)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))


#: The six tasks the catalog grades, and the allocation each declares: the server's
#: job (the task's GPUs, 8 CPUs, 180 GiB; TTS's 240 GiB build job), which the server gets less
#: the runner's reserve, and the driver's memory beside it; deepresearch's server its own 180 GiB
#: step, and its driver 8 CPUs of its own.
PUBLISHED_ALLOCATIONS = {
    "tasks.multitenant_slo.task": EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180,
                                                       driver_memory_gib=16),
    "tasks.tab_completion.task": EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180,
                                                      driver_memory_gib=16, verify_time_sec=3 * 3600),
    "tasks.multimodal_asr.task": EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180,
                                                      driver_memory_gib=16),
    "tasks.multimodal_omni.task": EvaluationAllocation(cpus=8, job_memory_gib=180, writable_gib=180,
                                                       driver_memory_gib=16),
    "tasks.deepresearch.task": EvaluationAllocation(cpus=8, step_memory_gib=180, writable_gib=180,
                                                    driver_memory_gib=64, driver_cpus=8, verify_time_sec=23400),
    "tasks.multimodal_tts.task": EvaluationAllocation(cpus=8, job_memory_gib=240, writable_gib=240,
                                                      driver_memory_gib=32),
    "tasks.agentic_coding.task": EvaluationAllocation(cpus=64, job_memory_gib=1911, writable_gib=3814,
                                                      driver_memory_gib=768, driver_memory_shared=True),
}


def test_agentic_coding_serves_in_the_whole_job_with_its_driver_oversubscribed(tmp_path, repo_login):
    """The server gets the published job's 1,911 GiB less the runner's reserve; the driver's
    768 GiB, its step's own limit, is oversubscribed on it, so the drive and the allocation ask
    for the job and no more."""
    plan = _published_plan(tmp_path, "tasks.agentic_coding.task", "agentic_coding")
    drive = plan.stage("deploy_drive")
    assert drive.params["artifact_shape"]["memory_gib"] == 1911 - RUNNER_GIB
    assert (drive.params["driver_shape"]["memory_gib"], drive.params["driver_overlaps"],
            drive.params["driver_memory_shared"]) == (768, True, True)
    assert drive.shape.memory_gib == 1911
    [alloc] = plan.allocations
    assert alloc.request.memory_gib == 1911


def test_agentic_codings_build_brief_states_the_memory_its_graded_server_is_given(tmp_path, repo_login):
    """The build brief's serving figure is the build plan's served shape, which must be the
    memory the grading's server step is given."""
    from tests.agent.test_build_reference_service import _agentic_build

    build = _agentic_build(True)["agent_build"]
    drive = _published_plan(tmp_path, "tasks.agentic_coding.task", "agentic_coding").stage("deploy_drive")
    assert build.params["served_shape"]["memory_gib"] == drive.params["artifact_shape"]["memory_gib"]


def test_a_task_whose_driver_shares_no_memory_plans_its_drive_as_before(tmp_path, repo_login):
    # The six other tasks' drives carry no such key and count the driver's memory beside the server's.
    plan = _published_plan(tmp_path, "tasks.multitenant_slo.task", "multitenant_slo")
    assert "driver_memory_shared" not in plan.stage("deploy_drive").params


def test_a_larger_runner_reserve_comes_out_of_the_published_server_and_reference(tmp_path, repo_login):
    """The server's and the reference's memory are the declared jobs' less the runner's reserve,
    derived where they are planned: a site that keeps more for its runner gives the server and
    the reference that much less, and asks Slurm for the same jobs."""
    plan = _published_plan(tmp_path, "tasks.multitenant_slo.task", "multitenant_slo", **{"run.runner_memory_gib": "6"})
    drive = plan.stage("deploy_drive")
    assert drive.params["artifact_shape"]["memory_gib"] == 180 - 6
    assert drive.shape.memory_gib == 196
    assert plan.stage("verify").shape.memory_gib == REFERENCE_SHAPE.memory_gib == 180
    task = importlib.import_module("tasks.multitenant_slo.task").TASK
    assert reference_shape(task, runner_memory_gib=6).memory_gib == 180 - 6


@pytest.mark.parametrize("module", sorted(PUBLISHED_ALLOCATIONS))
def test_each_task_with_published_gradings_declares_their_allocation(module):
    artifact = importlib.import_module(module).ARTIFACT
    assert artifact.producer.evaluation_allocation == PUBLISHED_ALLOCATIONS[module]


#: The hardware each of the seven tasks declares: its H200s, 141 GB each, its host's CPUs,
#: memory in GB and storage in GB, and the interconnect between its GPUs where it names one.
PUBLISHED_HARDWARE = {
    "tasks.multitenant_slo.task": (1, 64, 1024, 2048, None),
    "tasks.tab_completion.task": (1, 64, 1024, 2048, None),
    "tasks.multimodal_asr.task": (1, 64, 512, 2048, None),
    "tasks.multimodal_omni.task": (2, 64, 512, 2048, None),
    "tasks.deepresearch.task": (1, 64, 512, 2048, None),
    "tasks.multimodal_tts.task": (1, 8, 240, 2048, None),
    "tasks.agentic_coding.task": (8, 64, 2011, 4096, "NVLink"),
}


@pytest.mark.parametrize("module", sorted(PUBLISHED_HARDWARE))
def test_each_published_task_declares_its_own_hardware(module):
    """A published task's hardware is its own declaration, so a change of another task's, the
    sample's among them, cannot move it."""
    assert set(PUBLISHED_HARDWARE) == set(PUBLISHED_ALLOCATIONS)
    gpus, cpus, memory_gb, storage_gb, interconnect = PUBLISHED_HARDWARE[module]
    assert importlib.import_module(module).TASK.hardware == Hardware(
        accelerator=AcceleratorSpec(kind="H200", count=gpus, memory_gb_per_device=141),
        interconnect=InterconnectSpec(kind=interconnect) if interconnect else None,
        cpu_count=cpus, memory_gb=memory_gb, storage_gb=storage_gb)
    source = Path(importlib.import_module(module).__file__).read_text()
    assert not re.search(r"tasks\.sample_task\.task import[^)]*\bHARDWARE\b", source)


def _published_plan(tmp_path: Path, module: str, task_id: str, **overrides: str) -> Plan:
    """A grading of *module*'s published task."""
    package = _packaged(tmp_path, task=task_id)
    return _plan(tmp_path, module=module, package=package, overrides=overrides, config=_config(tmp_path))


def test_a_grading_whose_sandbox_stores_lack_what_the_corpus_needs_is_refused_in_one_line(tmp_path, repo_login):
    """What the corpus needs in the stores is checked before anything is planned
    (``commands.missing_store_entries``); the refusal names what is missing and the command to
    run."""
    module = "tasks.agentic_coding.task"
    package = _packaged(tmp_path, task="agentic_coding")
    config = _config(tmp_path)
    provision_module(tmp_path / "store", module)
    stores = {name: sandbox_store(tmp_path, name) for name in task_sandbox_stores(module)}
    fill_sandbox_stores(stores, module, tmp_path / "store")
    (stores["agentic-coding-runtime"] / RUNTIME_TAG / "bin" / "python3").unlink()
    with pytest.raises(PlanError) as e:
        commands.plan_grade(config=config, overrides={}, task_module=module, package=package, job_name="hls-grade",
                            echo=[].append)
    message = str(e.value)
    assert "\n" not in message
    assert "agentic-coding-runtime lacks the exec server's interpreter" in message
    assert message.endswith(f"run hls provision {module} --config {config} first")
    # A store the site names but that is no directory at all is named as such.
    shutil.rmtree(stores["agentic-coding-overlays"])
    with pytest.raises(PlanError, match="agentic-coding-overlays lacks its directory"):
        commands.plan_grade(config=config, overrides={}, task_module=module, package=package, job_name="hls-grade",
                            echo=[].append)


def test_a_grading_is_refused_while_a_sandbox_image_is_not_held_as_provisioning_leaves_it(tmp_path, repo_login):
    """A declared sandbox image is in its store only with its record (``missing_images``), the
    test ``hls provision`` plans its ``sandbox_images`` stage by: a file alone is refused, and the
    refusal names the command whose plan then pulls it."""
    from hls.provision.role import IDS_FILE, resolve
    from hls.provision.rows import tenant_subpath
    from hls.provision.sandbox_images import record_file
    from hls.store.lock import Holder
    from hls.taskref import TaskRef, load_task
    from tasks.agentic_coding.task import SANDBOX_IMAGES
    from tests.store.entries import populate

    module = "tasks.agentic_coding.task"
    package = _packaged(tmp_path, task="agentic_coding")
    config = _config(tmp_path)
    # A rows entry whose test split holds two tasks of the corpus, so the plan asks for their images.
    first, second = SANDBOX_IMAGES.images[:2]
    commands.task_id_of(module)
    resolved = resolve(load_task(TaskRef(module=module)))
    rows = json.dumps([{"task_name": first.name}, {"task_name": second.name}])

    def fill(d: Path) -> None:
        for tenant in resolved.tenants:
            for split in ("dev", "test"):
                path = d / tenant_subpath(tenant) / split / "rows.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rows if split == "test" else "[]")
        (d / IDS_FILE).write_text(json.dumps({"dev": [], "test": [first.name, second.name]}))

    populate(ContentCache(PosixStore(tmp_path / "store")), "rows", resolved.rows_key, fill,
             holder=Holder.here("test provision"))
    provision_module(tmp_path / "store", module)
    stores = {name: sandbox_store(tmp_path, name) for name in task_sandbox_stores(module)}
    fill_sandbox_stores(stores, module, tmp_path / "store")
    settings = load_settings(config, {})

    def grade() -> Plan:
        return commands.plan_grade(config=config, overrides={}, task_module=module, package=package,
                                   job_name="hls-grade", echo=[].append)

    held = set(commands.missing_sandbox_images(settings, SANDBOX_IMAGES))
    assert not {first.name, second.name} & held
    grade()
    record_file(stores[SANDBOX_IMAGES.store], second.name).unlink()
    assert second.name in commands.missing_sandbox_images(settings, SANDBOX_IMAGES)
    with pytest.raises(PlanError) as e:
        grade()
    message = str(e.value)
    assert "\n" not in message and f"agentic-coding-images lacks the image {second.name}.sif" in message
    assert first.name not in message
    assert message.endswith(f"run hls provision {module} --config {config} first")
    # That command's plan carries the stage that pulls it.
    plan = commands.plan_provision(config=config, overrides={}, task_module=module, weights=True,
                                   job_name="hls-provision", echo=[].append)
    assert "sandbox_images" in [s.name for s in plan.stages]


def _request(plan: Plan) -> tuple:
    [alloc] = plan.allocations
    r = alloc.request
    return r.gpus, r.gpu_kind, r.cpus, r.memory_gib, r.disk_gib


def test_multitenant_slo_evaluates_in_its_published_gradings_allocation(tmp_path, repo_login):
    plan = _published_plan(tmp_path, "tasks.multitenant_slo.task", "multitenant_slo")
    drive = plan.stage("deploy_drive")
    trace_gib = PUBLISHED_TRACE_GIB["multitenant_slo"]
    art = ResourceShape(gpus=1, gpu_kind="H200", cpus=8, memory_gib=176, disk_gib=180, time_sec=OPEN_TIME)
    drv = ResourceShape(cpus=8, memory_gib=16, disk_gib=trace_gib, time_sec=OPEN_TIME)
    assert (drive.params["artifact_shape"], drive.params["driver_shape"], drive.params["driver_overlaps"]) == (
        art.model_dump(), drv.model_dump(), True)
    # Its 180 GiB writable cap, the two trees the drive extracts with what a package adds to its
    # image in both its file's copy and its tree, its trace at its most, and the copies of its data.
    provisioned = provision_module(tmp_path / "store", "tasks.multitenant_slo.task")
    assert drive.params["data_copies"] == copied_entries(("artifact", "driver"), provisioned)
    data = copies_gib(drive.params["data_copies"])
    assert drive.shape == ResourceShape(gpus=1, gpu_kind="H200", cpus=8, memory_gib=196,
                                        disk_gib=316 + trace_gib + data,
                                        time_sec=OPEN_TIME + DRIVE_STAGING + TRACE_COPY_SEC)
    verify = plan.stage("verify")
    assert verify.params["data_copies"] == copied_entries(("reference",), provisioned)
    assert verify.shape == ResourceShape(gpus=1, gpu_kind="H200", cpus=8, memory_gib=180,
                                         disk_gib=40 + copies_gib(verify.params["data_copies"]),
                                         time_sec=7200 + STAGING_SEC)
    assert _request(plan) == (1, "H200", 8, 196, 316 + trace_gib + data)


def test_deepresearch_drives_with_its_gpu_retriever_on_cpus_of_its_own(tmp_path, repo_login):
    plan = _published_plan(tmp_path, "tasks.deepresearch.task", "deepresearch")
    drive = plan.stage("deploy_drive")
    time = drive_time(load_task(TaskRef(module="tasks.deepresearch.task")))
    trace_gib = PUBLISHED_TRACE_GIB["deepresearch"]
    drv = ResourceShape(gpus=1, gpu_kind="H200", cpus=8, memory_gib=64, disk_gib=trace_gib, time_sec=time)
    assert (drive.params["driver_shape"], drive.params["driver_overlaps"]) == (drv.model_dump(), False)
    data = copies_gib(drive.params["data_copies"])
    assert drive.shape == ResourceShape(gpus=2, gpu_kind="H200", cpus=16, memory_gib=248,
                                        disk_gib=316 + trace_gib + data,
                                        time_sec=time + DRIVE_STAGING + TRACE_COPY_SEC)
    assert _request(plan) == (2, "H200", 16, 248, 316 + trace_gib + data)


def test_multimodal_tts_serves_in_its_build_jobs_memory(tmp_path, repo_login):
    plan = _published_plan(tmp_path, "tasks.multimodal_tts.task", "multimodal-tts")
    drive = plan.stage("deploy_drive")
    assert (drive.params["artifact_shape"]["memory_gib"], drive.params["driver_shape"]["memory_gib"]) == (236, 32)
    # Within a 283 GiB per-GPU share. Its disk: its 240 GiB writable cap, the two trees, its
    # trace at its most and the copies of its data.
    trace_gib = PUBLISHED_TRACE_GIB["multimodal-tts"]
    assert _request(plan) == (1, "H200", 8, 272, 376 + trace_gib + copies_gib(drive.params["data_copies"]))


#: The score stage's memory each published task's grading plans, in GiB: the runner's 4, and the
#: role's 8 with a quarter more than 80 bytes of the summary for each packet timestamp one reply
#: of ``REPLY_MAX_BYTES`` (268,435,456) can buy, rounded up to the GiB
#: (``hls.stages.score.score_memory_gib``). A token-id stream's packet takes 2 bytes: 134,217,728
#: packets, 12.5 GiB, so 4 + 8 + 13. A TTS frame takes 78: 3,441,480 packets, 0.32 GiB, so
#: 4 + 8 + 1. A unary reply has no packets: 4 + 8.
PUBLISHED_SCORE_GIB = {"multitenant_slo": 25, "tab_completion": 25, "deepresearch": 25, "multimodal-tts": 13,
                       "multimodal-asr": 12, "multimodal-omni": 12, "agentic_coding": 25}
#: The disk in whole GiB each published task's driver holds for its trace's room
#: (``trace_disk_bytes``), from the cap and request bytes the task declares: the events hold the
#: cap and the requests, and the room holds them in the chunks and again in the summary at zlib's
#: bound.
#: - multitenant_slo: 6 GiB and 160 MiB, a room of 13,224,483,662 bytes: 13 GiB;
#: - tab_completion: 6 GiB and 384 MiB, 13,694,389,184 bytes: 13 GiB;
#: - deepresearch: 3 GiB and 160 MiB, 6,780,065,246 bytes: 7 GiB;
#: - multimodal-asr: 3 GiB and 1 GiB, 8,592,557,938 bytes: 9 GiB;
#: - multimodal-omni: 3 GiB and 2 GiB, 10,740,697,410 bytes: 11 GiB;
#: - multimodal-tts: 61 GiB and 4 MiB, 131,044,899,010 bytes: 123 GiB;
#: - agentic_coding: 22 GiB and 640 MiB, 48,601,655,604 bytes: 46 GiB.
PUBLISHED_TRACE_GIB = {"multitenant_slo": 13, "tab_completion": 13, "deepresearch": 7, "multimodal-asr": 9,
                       "multimodal-omni": 11, "multimodal-tts": 123, "agentic_coding": 46}


#: Each graded task module the catalog plans, its id, and the verify
#: stage's time its plan carries: the two-hour default where twice the slowest measured verify
#: fits in it, else the time its allocation declares (tab_completion 3 h, deepresearch 6 h 30 min).
PUBLISHED_VERIFY_SEC = [
    ("tasks.multitenant_slo.task", "multitenant_slo", 2 * 3600),
    ("tasks.tab_completion.task", "tab_completion", 3 * 3600),
    ("tasks.multimodal_asr.task", "multimodal-asr", 2 * 3600),
    ("tasks.multimodal_omni.task", "multimodal-omni", 2 * 3600),
    ("tasks.deepresearch.task", "deepresearch", 23400),
    ("tasks.multimodal_tts.task", "multimodal-tts", 2 * 3600),
    ("tasks.agentic_coding.task", "agentic_coding", 2 * 3600),
]


@pytest.mark.parametrize("module, task_id, verify_sec", PUBLISHED_VERIFY_SEC)
def test_each_published_task_verifies_for_its_declared_time(tmp_path, repo_login, module, task_id, verify_sec):
    plan = _published_plan(tmp_path, module, task_id)
    verify = plan.stage("verify")
    assert (verify.shape.time_sec, verify.params["reference_shape"]["time_sec"]) == (
        verify_sec + STAGING_SEC, verify_sec)
    # The allocation asks for that time among its stages', and finalize has its fixed time.
    [alloc] = plan.allocations
    assert alloc.request.time_sec == sum(s.shape.time_sec for s in plan.stages) + 360 + _headroom(plan)
    assert plan.stage("finalize").shape == FINALIZE_SHAPE


@pytest.mark.parametrize("module, task_id, verify_sec", PUBLISHED_VERIFY_SEC)
def test_each_published_tasks_score_stage_holds_one_replys_summary_inside_the_drives_allocation(
        tmp_path, repo_login, module, task_id, verify_sec):
    plan = _published_plan(tmp_path, module, task_id)
    score = plan.stage("score").shape
    task = load_task(TaskRef(module=module))
    assert score == score_shape(task, RUNNER_GIB)
    assert score.memory_gib == score_memory_gib(task.contract) + RUNNER_GIB
    assert score.memory_gib == PUBLISHED_SCORE_GIB[task_id]
    # Stages run in turn, so the allocation holds the largest; the drive's stays the largest.
    [alloc] = plan.allocations
    assert alloc.request.memory_gib == plan.stage("deploy_drive").shape.memory_gib > score.memory_gib


@pytest.mark.parametrize("module, task_id, verify_sec", PUBLISHED_VERIFY_SEC)
def test_each_published_tasks_driver_holds_its_trace_and_its_drive_the_traces_copy(
        tmp_path, repo_login, module, task_id, verify_sec):
    plan = _published_plan(tmp_path, module, task_id)
    drive = plan.stage("deploy_drive")
    task = load_task(TaskRef(module=module))
    bounds = drive_bounds(task)
    trace_gib = PUBLISHED_TRACE_GIB[task_id]
    assert math.ceil(trace_disk_bytes(bounds.trace_cap_bytes, bounds.trace_request_bytes) / 2**30) == trace_gib
    assert drive.params["driver_shape"]["disk_gib"] == trace_gib
    assert (drive.params["staging_sec"], drive.params["wall_margin_sec"]) == (
        DRIVE_STAGING, DRIVE_MARGIN_SEC + TRACE_COPY_SEC)
    assert drive.shape.time_sec == drive_time(task) + DRIVE_STAGING + TRACE_COPY_SEC


def _score_table(task_id: str, memory_gib: int) -> str:
    """A ``[resources]`` table that gives *task_id*'s score ``SCORE_SHAPE`` with *memory_gib*."""
    fields = {**SCORE_SHAPE.model_dump(exclude_none=True), "memory_gib": memory_gib}
    return f"\n[resources.{task_id}.score]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in fields.items())


def test_a_site_may_not_give_the_score_role_less_than_one_replys_summary_needs(tmp_path, repo_login):
    # Less memory than the summary a contestant's reply can make would let the role run out of it
    # on a trace the driver accepted: an infra fault, retried, then a void.
    package = _packaged(tmp_path, task="multitenant_slo")
    mslo = "tasks.multitenant_slo.task"
    need = score_memory_gib(load_task(TaskRef(module=mslo)).contract) + RUNNER_GIB
    assert need == PUBLISHED_SCORE_GIB["multitenant_slo"] == 25
    short = _config(tmp_path, _score_table("multitenant_slo", 12))
    with pytest.raises(PlanError, match="stage score: \\[resources.multitenant_slo.score\\] gives it 12 GiB, less than "
                                        "the 25 GiB"):
        _plan(tmp_path, module="tasks.multitenant_slo.task", package=package, config=short)
    for memory_gib in (need, 2 * need):
        config = _config(tmp_path, _score_table("multitenant_slo", memory_gib))
        plan = _plan(tmp_path, module="tasks.multitenant_slo.task", package=package, config=config)
        assert (plan.stage("score").shape.memory_gib, plan.stage("score").shape_overridden) == (memory_gib, True)


#: The trace cap three published tasks declare (``hls.drive.bounds.trace_cap``): the larger of 4
#: times the largest trace the task cites and that trace beside the most reply bytes it cites held
#: at once at the protocol's most trace per byte, in whole GiB, or the most its verify role
#: can hold when that is less (``hls.drive.bounds.verify_cap``). multitenant_slo: 64,342,088 bytes
#: beside 462,268,100 at 13.5, 5.87 GiB, so 6; deepresearch: what its verify role holds, 59.25 GiB
#: and 26.3 bytes per byte, fits 3 GiB, under the 12 its cited figures give; multimodal-tts:
#: 29.4 GiB and 2.11 per byte fits 61 GiB, under 98.
@pytest.mark.parametrize("module, task_id, cap, driver_gib", [
    ("tasks.multitenant_slo.task", "multitenant_slo", 6 * 2**30, 16),
    ("tasks.deepresearch.task", "deepresearch", 3 * 2**30, 64),
    ("tasks.multimodal_tts.task", "multimodal-tts", 61 * 2**30, 32),
])
def test_each_evaluation_passes_its_tasks_trace_cap_to_the_driver(tmp_path, repo_login, module, task_id, cap,
                                                                  driver_gib):
    drive = _published_plan(tmp_path, module, task_id).stage("deploy_drive").params
    assert (drive["reply_max_bytes"], drive["trace_cap_bytes"]) == (REPLY_MAX_BYTES, cap)
    assert "drive_budget_bytes" not in drive
    assert cap == _cap(module)
    # The driver's memory limit is its declared memory, the allocation's figure for the driver.
    assert drive["driver_shape"]["memory_gib"] == driver_gib


def test_a_grading_of_a_task_without_drive_bounds_is_refused_when_planned(tmp_path, login, in_catalog):
    with pytest.raises(PlanError, match="evaluate_task cannot be graded: no drive bounds for task 'evaluate_task'"):
        _plan(tmp_path, module="ev_unbounded.task", package=_packaged(tmp_path))


def test_only_a_measured_baseline_evaluation_drives_uncounted(tmp_path, login, in_catalog):
    package = _plan(tmp_path, package=_packaged(tmp_path))
    in_catalog(roles=("baseline",))
    measured = _plan(tmp_path, measured="baseline")
    assert (package.stage("deploy_drive").params["measured_baseline"],
            measured.stage("deploy_drive").params["measured_baseline"]) == (False, True)
    # The baseline's drive still holds its trace to the task's cap.
    assert measured.stage("deploy_drive").params["trace_cap_bytes"] == CAP
    # Only the baseline's server starts from a stored engine cache, the one its warm stage owns;
    # a contestant's starts empty and its grading warms nothing.
    assert (package.stage("deploy_drive").params["engine_cache_owner"],
            measured.stage("deploy_drive").params["engine_cache_owner"]) == (None, "warm_baseline")
    assert "warm_baseline" not in {s.name for s in package.stages}
    warm = measured.stage("warm_baseline")
    assert (warm.kind, warm.after) == ("baseline_warm", ["provision", "images"])
    assert measured.stage("deploy_drive").after == ["provision", "images", "warm_baseline"]
    assert [s.name for s in measured.stages] == ["provision", "images", "warm_baseline", "deploy_drive", "verify",
                                                 "score", "finalize"]
    params = BaselineWarmParams.model_validate(warm.params)
    drive = DeployDriveParams.model_validate(measured.stage("deploy_drive").params)
    assert (params.artifact, params.serve_command, params.artifact_shape, params.network, params.port) == (
        drive.artifact, drive.serve_command, drive.artifact_shape, drive.network, drive.port)


def test_the_drive_enforces_the_trace_cap_the_task_module_declares(tmp_path, repo_login):
    # No task override sets the cap or the request bytes.
    plan = _published_plan(tmp_path, "tasks.multitenant_slo.task", "multitenant_slo")
    assert plan.stage("deploy_drive").params["trace_cap_bytes"] == 6 * 2**30
    for key in ("task.drive_bounds.trace_cap_bytes", "task.drive_bounds.trace_request_bytes"):
        with pytest.raises(ValueError, match=f"overrides may not set {re.escape(key)}"):
            _published_plan(tmp_path, "tasks.multitenant_slo.task", "multitenant_slo", **{key: str(4 * 2**30)})


def test_a_grading_of_a_task_whose_build_stage_has_a_table_warns_and_is_planned(tmp_path, repo_login):
    # One hls.toml serves a task's run and its gradings: the run's agent_build table is no typo in
    # a grading's plan, which does not run that stage and ignores it.
    package = _packaged(tmp_path, task="multitenant_slo")
    table = "\n[resources.multitenant_slo.agent_build]\ntime_sec = 7200\n"
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task", package=package, config=_config(tmp_path, table))
    assert "agent_build" not in [s.name for s in plan.stages]
    assert [w for w in plan.warnings if "agent_build" in w] == [
        "[resources.multitenant_slo.agent_build] names stage agent_build, which this plan does not run: the table is "
        "ignored"]
    typo = "\n[resources.multitenant_slo.agent_biuld]\ntime_sec = 7200\n"
    with pytest.raises(PlanError, match=r"settings name stages that do not exist: resources\.multitenant_slo\."
                                        r"agent_biuld"):
        _plan(tmp_path, module="tasks.multitenant_slo.task", package=package, config=_config(tmp_path, typo))


def test_a_grading_scored_against_its_baseline_is_planned_on_any_site(tmp_path, repo_login):
    """No site is checked against the baseline credit's when a grading is
    planned; its records state the hardware it ran on and compare it with the credit's."""
    provision_module(tmp_path / "store", "tasks.multitenant_slo.task")
    plan = commands.plan_grade(config=_config(tmp_path), overrides={},
                               task_module="tasks.multitenant_slo.task",
                               package=_packaged(tmp_path, task="multitenant_slo"), job_name="hls-grade",
                               echo=[].append)
    assert not {"baseline_site", "site_waiver"} & plan.stage("finalize").params.keys()


#: The file the store holds for ASR's reference image, by its sha256.
REFERENCE_FILE = b"the reference image"
REFERENCE_SHA = hashlib.sha256(REFERENCE_FILE).hexdigest()


@pytest.fixture
def asr_measured(tmp_path, repo_login):
    """The login node planning from this repository's own checkout, with ASR's measured
    baseline and reference in the store's cache, as its images warmer builds them."""
    keys = image_keys(catalog.task_images("multimodal_asr").decls, ApptainerImages.recipe_key)
    cache = ContentCache(PosixStore(tmp_path / "store"))
    for name, data in (("multimodal-asr-baseline", b"the baseline image"), ("multimodal-asr-reference", REFERENCE_FILE)):
        populate(cache, "images", keys[name], lambda d, data=data: (d / IMAGE_FILE).write_bytes(data),
                 holder=Holder.here("test"))


def test_a_task_whose_id_is_not_its_directory_is_read_from_the_catalog_by_its_directory(tmp_path, asr_measured,
                                                                                          monkeypatch):
    """ASR's TaskSpec is ``multimodal-asr`` and its directory, the catalog's name for it,
    ``multimodal_asr``: the catalog is read by the directory, the plan and the store by the id. Its
    caching reference's answers are checked for the GPU kind and the measured reference image its
    verify runs on, and the plan says what the check found."""
    asked = []

    def covered(store, **kw):
        asked.append(kw)
        return OracleCoverage(rows="rows-0123456789abcdef", fingerprints=("fp-a", "fp-b"), prompts=10, files=3,
                              writers=("warmers/multimodal-asr/w0", "warmers/multimodal-asr/w1"))

    monkeypatch.setattr(commands, "check_oracle_coverage", covered)
    lines: list[str] = []
    provision_task(tmp_path / "store", "tasks.multimodal_asr.task")
    plan = commands.plan_grade(config=_config(tmp_path), overrides={}, task_module="tasks.multimodal_asr.task",
                               measured="baseline", job_name="hls-grade", echo=lines.append)
    [kw] = asked
    assert (kw["task_id"], kw["gpu_kind"], kw["reference_sha256"]) == ("multimodal-asr", "H200", REFERENCE_SHA)
    assert kw["task_ref"] == TaskRef(module="tasks.multimodal_asr.task")
    assert ("oracle answers: 10 prompts of rows rows-0123456789abcdef answered for references fp-a, fp-b in 3 "
            "answers files written by warmers/multimodal-asr/w0, warmers/multimodal-asr/w1") in lines
    assert plan.task == "multimodal-asr"
    assert plan.instance.startswith("warmers/multimodal-asr/measured-baseline/gradings/g-")
    images = plan.stage("images").params
    assert {"multimodal-asr-baseline", "multimodal-asr-reference"} <= {d["name"] for d in images["images"]}
    assert images["catalog_task"] == "multimodal_asr"
    assert plan.stage("deploy_drive").params["artifact"]["image"] == "multimodal-asr-baseline"
    assert plan.stage("deploy_drive").params["trace_cap_bytes"] == _cap("tasks.multimodal_asr.task")
    assert plan.stage("verify").params["reference_image"] == "multimodal-asr-reference"
    with pytest.raises(ValueError, match="is not a task module: a task module is tasks.<directory>.task"):
        commands.plan_grade(config=_config(tmp_path), overrides={}, task_module="multimodal_asr.task",
                            measured="baseline", job_name="hls-grade", echo=[].append)
    # The check is given the task the grading verifies, its overrides applied, and the reference
    # that holds them, which a refusal names the fill of. The task's default credit was measured on
    # the task without them, so it does not fit and the store's own credit is pinned.
    pin = MeasuredFile(source="store", path=pinned_key("multimodal-asr", "baseline.json", "e" * 40), blob="e" * 40)
    monkeypatch.setattr(commands, "find_measured", lambda store, **kw: [Found(name=BASELINE_CREDIT_FILE, pin=pin)])
    provision_task(tmp_path / "store", "tasks.multimodal_asr.task", {"task.seed": "7"})
    commands.plan_grade(config=_config(tmp_path), overrides={"task.seed": "7"}, task_module="tasks.multimodal_asr.task",
                        measured="baseline", job_name="hls-grade", echo=[].append)
    kw = asked[-1]
    assert kw["task"].seed == 7 != load_task(TaskRef(module="tasks.multimodal_asr.task")).seed
    assert kw["task_ref"] == TaskRef(module="tasks.multimodal_asr.task", overrides={"task.seed": 7})


def test_an_asr_grading_without_its_oracle_answers_is_refused(tmp_path, asr_measured, monkeypatch):
    def refuse(store, **kw):
        raise PlanError("multimodal-asr: run hls oracle fill tasks.multimodal_asr.task first")

    monkeypatch.setattr(commands, "check_oracle_coverage", refuse)
    with pytest.raises(PlanError, match="multimodal-asr: run hls oracle fill tasks.multimodal_asr.task first"):
        _plan(tmp_path, module="tasks.multimodal_asr.task", measured="baseline")


def _cache_all(tmp_path: Path) -> dict[str, str]:
    """Publish an entry for every image of the fixture task in the test's store; its keys by name."""
    keys = image_keys(order(catalog.task_images(fixture.TASK_ID).decls), ApptainerImages.recipe_key)
    cache = ContentCache(PosixStore(tmp_path / "store"))
    for key in keys.values():
        populate(cache, "images", key, lambda d: (d / IMAGE_FILE).write_bytes(b"image"), holder=Holder.here("test"))
    return keys


def test_a_grading_whose_cache_holds_its_images_asks_for_the_disk_of_its_other_stages(tmp_path, login, in_catalog):
    """The images stage finds every image, so it holds no disk, and the drive stage's two trees and
    the copies of its roles' data size the allocation's disk."""
    package = _packaged(tmp_path)
    keys = _cache_all(tmp_path)
    plan = _plan(tmp_path, package=package)
    images = plan.stage("images")
    assert images.shape == images_shape(0, cpus=EVALUATE_IMAGES_CPUS) and images.shape.disk_gib == 0
    assert images.shape.time_sec == 900
    assert images.params["found"] == sorted(keys[d["name"]] for d in images.params["images"])
    drive = plan.stage("deploy_drive").shape.disk_gib
    data = copies_gib(plan.stage("deploy_drive").params["data_copies"])
    art = artifact_shape(fixture.evaluate_task(), runner_memory_gib=RUNNER_GIB)
    assert drive == art.disk_gib + TRACE_GIB + DRIVE_TREES_GIB + data and plan.allocations[0].request.disk_gib == drive
    # The drive's staging is the same fixed allowance as when the cache lacks its images.
    assert plan.stage("deploy_drive").params["staging_sec"] == DRIVE_STAGING


def test_a_grading_looks_up_the_cache_entries_of_its_own_images_and_no_others(tmp_path, login, in_catalog,
                                                                            monkeypatch):
    """The catalog's agent cap is no image of a grading: the planner never asks the cache for it,
    so whatever the plan found is an image of its images stage."""
    package = _packaged(tmp_path)
    keys = _cache_all(tmp_path)
    asked: list[str] = []
    lookup = ContentCache.lookup

    def recording(self, kind, key):
        if kind == "images":
            asked.append(key)
        return lookup(self, kind, key)

    monkeypatch.setattr(ContentCache, "lookup", recording)
    images = _plan(tmp_path, package=package).stage("images")
    own = sorted(keys[d["name"]] for d in images.params["images"])
    # The images stage asks for each once; no other stage asks.
    assert sorted(asked) == own == images.params["found"] and len(own) < len(keys)


def test_a_grading_serves_a_pinned_package_or_one_of_the_tasks_measured_images(tmp_path, in_catalog):
    ref, task = TaskRef(module="ev_task.task"), fixture.evaluate_task()
    pinned = InstanceLayout.warmer(fixture.TASK_ID, WARMER).manifest("package")
    common = {"task_ref": ref, "task": task, "catalog_key": fixture.TASK_ID, "eval_id": "g1",
              "run": RunSettings(), "seal_memory_gib": 0, "held": lambda keys: set(), "recipe_key": def_key,
              "provisioned": provision_into(tmp_path / "store", task)}
    for artifact, digest in ((PackagedArtifact(), "0" * 64), (PackagedArtifact(manifest=pinned), None)):
        with pytest.raises(ValueError, match="serves a published package"):
            grading_stages(artifact=artifact, package_digest=digest, **common)
    for image in (LAYER, "evaluate-task-sandbox"):
        with pytest.raises(ValueError, match=f"{image} is not one of evaluate_task's measured images"):
            grading_stages(artifact=ImageArtifact(image=image, entrypoint=["x"]), **common)


def test_the_measured_baseline_is_served_with_the_start_the_catalog_declares(tmp_path, login, in_catalog):
    in_catalog(roles=("baseline",))
    plan = _plan(tmp_path, measured="baseline")
    assert plan.instance.startswith(f"warmers/{fixture.TASK_ID}/measured-baseline/gradings/g-")
    drive = plan.stage("deploy_drive")
    assert drive.params["artifact"] == {"kind": "image", "image": "evaluate-task-baseline",
                                        "entrypoint": fixture.BASELINE_ENTRYPOINT, "runtime_env": fixture.BASELINE_ENV}
    assert drive.params["serve_command"] == fixture.BASELINE_SERVE and drive.inputs == {}
    names = [d["name"] for d in plan.stage("images").params["images"]]
    assert names == ["s-a", LAYER, "cuda", "evaluate-task-baseline"]
    # The cluster's images warmer built the baseline, so the stage fills the task's two others.
    assert plan.stage("images").shape == images_shape(2, cpus=EVALUATE_IMAGES_CPUS)


def test_the_measured_reference_is_the_verify_image_when_the_catalog_lists_one(tmp_path, login, in_catalog):
    package = _packaged(tmp_path)
    in_catalog(roles=("baseline", "reference"))
    plan = _plan(tmp_path, package=package)
    assert plan.stage("verify").params["reference_image"] == "evaluate-task-reference"
    assert plan.stage("deploy_drive").params["harness_image"] == LAYER
    assert "evaluate-task-reference" in [d["name"] for d in plan.stage("images").params["images"]]
    in_catalog(roles=("baseline",))
    assert _plan(tmp_path, package=package).stage("verify").params["reference_image"] == LAYER


@pytest.mark.parametrize("roles, role, match", [
    (("baseline",), "sandbox", "evaluate_task declares no sandbox image"),
    (("reference",), "baseline", "evaluate_task declares no baseline image"),
    (("baseline", "member-vllm"), "member-vllm", "only the baseline is served"),
])
def test_measured_refuses_what_the_task_does_not_declare(tmp_path, login, in_catalog, roles, role, match):
    in_catalog(roles=roles)
    with pytest.raises(ValueError, match=rf"^--measured {role}: .*{match}"):
        _plan(tmp_path, measured=role)
    # The reader names no flag of its own: each command that reads a measured image says which
    # of its flags named it.
    task = commands._import_task_module(commands.checkout_root(), "ev_task.task").TASK
    with pytest.raises(ValueError, match=match) as refused:
        commands.measured_artifact(task, role, fixture.TASK_ID)
    assert "--measured" not in str(refused.value)


@pytest.mark.parametrize("package, match", [
    ("runs/evaluate_task/r1", "is not a cache warmer of evaluate_task"),
    (f"warmers/sample_task/{WARMER}", "is not a cache warmer of evaluate_task"),
    ("warmers/evaluate_task/nothing-here", "has no published package stage"),
    (f"warmers/evaluate_task/{WARMER}/gradings/g1", "is not a cache warmer of evaluate_task"),
    # Ids no instance may hold are refused before the store is read.
    ("warmers/evaluate_task/index", "is not a cache warmer of evaluate_task: .*index directory"),
    ("warmers/evaluate_task/w+1", r"is not a cache warmer of evaluate_task: .*holds \+"),
])
def test_package_names_a_published_package_warmer_of_the_task(tmp_path, login, in_catalog, package, match):
    _packaged(tmp_path)
    _packaged(tmp_path, task="sample_task")
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, package=package)


def test_a_package_warmers_manifest_that_does_not_parse_is_refused_naming_its_errors(tmp_path, login, in_catalog):
    package = _packaged(tmp_path)
    manifest = InstanceLayout(package).manifest("package")
    PosixStore(tmp_path / "store").path(manifest).write_bytes(b'{"instance": 7}')
    with pytest.raises(ValueError) as e:
        _plan(tmp_path, package=package)
    assert str(e.value).startswith(f"--package {package}: {manifest} is not a manifest: string_type instance; ")
    assert "validation error" not in str(e.value)


def test_exactly_one_artifact_is_named(tmp_path, login, in_catalog):
    with pytest.raises(ValueError, match="exactly one"):
        _plan(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        _plan(tmp_path, package=_packaged(tmp_path), measured="baseline")


def test_a_verifier_whose_engines_cannot_share_the_role_is_refused_when_planned(tmp_path, login, in_catalog):
    # The fake generator and forcer do not place themselves on GPUs of their own.
    with pytest.raises(PlanError, match="stage verify: ev_pair.task: PairVerifier loads 2 reference engines "
                                        "at once, and these take every GPU the role sees"):
        _plan(tmp_path, module="ev_pair.task", package=_packaged(tmp_path))


@pytest.mark.parametrize("hardware, match", [
    (Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)), "states 0 CPUs, 0 GB of memory and 0 GB of"),
    (Hardware(accelerator=AcceleratorSpec(kind="B200", count=1), cpu_count=2, memory_gb=4),
     "states 2 CPUs, 4 GB of memory and 0 GB of storage"),
    (Hardware(accelerator=AcceleratorSpec(kind="CPU", count=1), cpu_count=2, memory_gb=4, storage_gb=2),
     "CPU is not a GPU kind"),
])
def test_a_task_without_an_evaluation_allocation_serves_on_the_briefs_hardware(hardware, match):
    task = fixture.evaluate_task(hardware=hardware)
    with pytest.raises(ValueError, match=match):
        artifact_shape(task, runner_memory_gib=RUNNER_GIB)
    # An evaluation allocation stands in for the brief's CPUs, memory and storage; the kind is
    # still the task's.
    if hardware.accelerator.kind == "CPU":
        with pytest.raises(ValueError, match=match):
            artifact_shape(task, fixture.ALLOCATION, runner_memory_gib=RUNNER_GIB)
    else:
        assert artifact_shape(task, fixture.ALLOCATION,
                              runner_memory_gib=RUNNER_GIB).disk_gib == fixture.ALLOCATION.writable_gib


def _resources(stage: str, shape: ResourceShape, **changes) -> str:
    """A ``[resources]`` table that gives the fixture task's *stage* *shape* with *changes*."""
    fields = {**shape.model_dump(exclude_none=True), **changes}
    return f"\n[resources.{fixture.TASK_ID}.{stage}]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in fields.items())


@pytest.mark.parametrize("stage, match", [("deploy_drive", "the stage's resources must be the artifact's"),
                                          ("verify", "its resources must be reference_shape's")])
def test_a_resources_override_of_a_stage_that_runs_declared_roles_is_refused(tmp_path, login, in_catalog, stage,
                                                                              match):
    package = _packaged(tmp_path)
    resources = _resources(stage, _plan(tmp_path, package=package).stage(stage).shape, cpus=32)
    with pytest.raises(PlanError, match=f"stage {stage}: .*{match}"):
        _plan(tmp_path, package=package, config=_config(tmp_path, resources))


def test_a_site_sizes_the_drive_and_verify_times_and_the_allocation_follows(tmp_path, login, in_catalog):
    package = _packaged(tmp_path)
    declared = _plan(tmp_path, package=package)
    drive, verify = declared.stage("deploy_drive").shape, declared.stage("verify").shape
    # The drive's floor is the staging, the runtime's start, two readiness budgets, the drive's wall
    # (the open one here) and the margin, which holds the trace's copy to the store: a site's time
    # below it would let ordinary staging cut the wall.
    floor = DRIVE_STAGING + OPEN_TIME + TRACE_COPY_SEC
    assert floor == drive.time_sec
    resources = (_resources("deploy_drive", drive, time_sec=floor + 1800)
                 + _resources("verify", verify, time_sec=3600))
    plan = _plan(tmp_path, package=package, config=_config(tmp_path, resources))
    stages = {s.name: s for s in plan.stages}
    assert [(stages[s].shape.time_sec, stages[s].shape_overridden) for s in ("deploy_drive", "verify", "score")] == [
        (floor + 1800, True), (3600, True), (1800, False)]
    # The parameters keep the declared role shapes; the roles run for the stage's time.
    assert stages["deploy_drive"].params["artifact_shape"] == artifact_shape(fixture.evaluate_task(), runner_memory_gib=RUNNER_GIB).model_dump()
    [alloc] = plan.allocations
    assert alloc.request.time_sec == (7200 + stages["images"].shape.time_sec + floor + 1800 + 3600 + 1800
                                      + FINALIZE_SEC + 360 + _headroom(plan))
    low = _resources("deploy_drive", drive, time_sec=floor - 1)
    with pytest.raises(PlanError, match=f"stage deploy_drive: the stage's time {floor - 1}s does not cover its floor "
                                        f"of {floor}s: the staging"):
        _plan(tmp_path, package=package, config=_config(tmp_path, low))


def test_the_new_stages_take_retry_settings(tmp_path, login, in_catalog):
    overrides = {f"run.retry.{s}.max": "0" for s in STAGES[2:]}
    plan = _plan(tmp_path, package=_packaged(tmp_path), overrides=overrides)
    assert [plan.stage(s).retry.max for s in STAGES[2:]] == [0, 0, 0, 0]


def test_task_overrides_are_carried_and_other_roots_refused(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, package=_packaged(tmp_path), overrides={"task.max_context_len": "2048"})
    refs = [s.params["task"] for s in plan.stages if isinstance(s.params.get("task"), dict)]
    assert len(refs) == 5 and all(ref == {"module": "ev_task.task", "overrides": {"task.max_context_len": 2048}}
                                  for ref in refs)
    with pytest.raises(ValueError, match=r"takes --site\.\*, --run\.\*, --resources\.\* and --task\.\* "
                                         "overrides, not artifact.image"):
        _plan(tmp_path, package=_packaged(tmp_path), overrides={"artifact.image": "x"})


@pytest.mark.parametrize("port", [8888, 8899])
def test_a_grading_of_a_task_on_an_interceptor_port_is_refused_when_planned(tmp_path, login, in_catalog, port):
    with pytest.raises(ValueError, match=f"^{fixture.TASK_ID}'s contract port {port} is refused: the egress "
                                         f"interceptor's [a-z ]+ listens on {port} "):
        _plan(tmp_path, package=_packaged(tmp_path), overrides={"task.contract.port": str(port)})


def test_the_cli_refuses_a_task_on_an_interceptor_port_in_one_line(tmp_path, login, in_catalog, monkeypatch):
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["grade", "ev_task.task", "--package", _packaged(tmp_path), "--config",
                                          str(_config(tmp_path)), "--task.contract.port", "8888"])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
    assert result.stderr == (f"hls grade: {fixture.TASK_ID}'s contract port 8888 is refused: the egress "
                             "interceptor's transparent listener listens on 8888 in the network namespace a sealed "
                             "role's server shares; declare another contract.port\n")


def test_the_cli_refuses_a_measured_grading_of_a_task_with_no_directory_in_one_line(tmp_path, login, in_catalog,
                                                                                     monkeypatch):
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    monkeypatch.setattr(catalog, "image_set", DECLARED_IMAGES)
    result = CliRunner().invoke(cli.app, ["grade", "ev_task.task", "--measured", "baseline", "--config",
                                          str(_config(tmp_path))])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
    assert result.stderr == (f"hls grade: --measured baseline: no task 'evaluate_task': a task is a directory under "
                             f"tasks/ holding task.py; there are {catalog.task_keys()}\n")


def test_an_evaluation_plan_that_would_hold_a_secret_value_is_refused(tmp_path, login, in_catalog):
    # The secrets file reaches the planner, which refuses a plan whose text would hold a value.
    secret = "group-account-0123456789"
    config = _config(tmp_path, secrets=f'[secrets]\naccount = "{secret}"\n')
    with pytest.raises(PlanError, match="settings.site.account holds account") as refused:
        _plan(tmp_path, package=_packaged(tmp_path), config=config, overrides={"site.account": secret})
    assert secret not in str(refused.value)


def test_an_evaluation_is_indexed_apart_from_the_tasks_runs():
    assert index_key("warmers/t/w1/gradings/g1", "grading") == ("warmers/t/index/w1+g1.json", "w1", "g1")
    assert index_key("runs/t/r1/gradings/g1", "grading") == ("runs/t/index/r1+g1.json", "r1", "g1")
    refused = [("warmers/t/w1", "rollout"), ("warmers/t/w1", "grading"), ("warmers/t/index/gradings/g1", "grading"),
               ("credits/t/c1/gradings/g1", "grading"), ("warmers/t/w1/gradings/g1", "rollout")]
    for instance, kind in refused:
        with pytest.raises(ValueError):
            index_key(instance, kind)


@pytest.mark.parametrize("args, package, measured", [
    (["--package", f"warmers/evaluate_task/{WARMER}"], f"warmers/evaluate_task/{WARMER}", None),
    (["--measured", "baseline"], None, "baseline"),
])
def test_the_cli_plans_grade_with_its_artifact_and_starts_it(monkeypatch, args, package, measured):
    seen, started = {}, []

    def plan(**kw):
        seen.update(kw)
        return "the plan"

    monkeypatch.setattr(commands, "plan_grade", plan)
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: started.append((made, overrides)) or 0)
    # The command configures hls logging for a terminal; a test leaves it as pytest set it.
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["grade", "tasks.sample_task.task", *args, "--site.qos", "normal"])
    assert result.exit_code == 0, result.output
    assert (seen["task_module"], seen["package"], seen["measured"], seen["job_name"]) == (
        "tasks.sample_task.task", package, measured, "hls-grade")
    assert (seen["rollout"], seen["rebuild"], seen["image"], seen["image_file"], seen["built_arm"]) == (
        None, False, None, None, None)
    assert "site_waiver" not in seen
    assert seen["overrides"] == {"site.qos": "normal"}
    assert started == [("the plan", {"site.qos": "normal"})]


def test_the_cli_takes_no_site_waiver(monkeypatch):
    """There is no site check to waive. ``--site-waiver`` is no option, so it
    reaches the planner as an override of no settings root, which the planner refuses."""
    seen = {}
    monkeypatch.setattr(commands, "plan_grade", lambda **kw: seen.update(kw) or "the plan")
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: 0)
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    CliRunner().invoke(cli.app, ["grade", "tasks.multitenant_slo.task", "--package", "warmers/x/y",
                                 "--site-waiver", "a reason"])
    assert "site_waiver" not in seen and seen["overrides"] == {"site-waiver": "a reason"}
    with pytest.raises(ValueError, match="not site-waiver$"):
        commands.split_overrides(seen["overrides"])
    assert "--site-waiver" not in _text(CliRunner().invoke(cli.app, ["grade", "--help"]).output)


def test_the_cli_takes_artifact_rebuild_as_a_flag(monkeypatch):
    seen = {}
    monkeypatch.setattr(commands, "plan_grade", lambda **kw: seen.update(kw) or "the plan")
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: 0)
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["grade", "tasks.sample_task.task", "--rollout", "runs/sample_task/r1",
                                          "--artifact.rebuild", "--site.qos", "normal"])
    assert result.exit_code == 0, result.output
    assert (seen["rollout"], seen["rebuild"]) == ("runs/sample_task/r1", True)
    assert seen["overrides"] == {"site.qos": "normal"}


def test_the_cli_refuses_what_cannot_be_planned_in_one_line(monkeypatch):
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["grade", "tasks.sample_task.task"])
    assert result.exit_code == 1
    assert result.output.strip().splitlines() == [
        "hls grade: name exactly one artifact to grade: --rollout INSTANCE, --package INSTANCE, --measured ROLE or "
        "--artifact.image sha256:<digest>"]


PROBE = """
import json, sys
from pathlib import Path
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
import pytest
from tests.pipeline.plans import CODE, pin_harbor
commands.code_state = lambda path: CODE
pin_harbor(pytest.MonkeyPatch(), CODE)
commands.toolchain_here = lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/"))
plan = commands.plan_grade(config=Path(sys.argv[1]), overrides={}, task_module="tasks.sample_task.task",
                           package=sys.argv[2], job_name="x", echo=lambda line: None)
heavy = sorted(m for m in ("torch", "transformers", "vllm", "datasets", "huggingface_hub", "pyarrow")
               if m in sys.modules)
print(json.dumps({"heavy": heavy, "stages": [s.name for s in plan.stages]}))
"""


def _fresh(code: str, *args: str) -> str:
    """The last line *code* prints in a fresh interpreter run from this checkout."""
    root = commands.checkout_root()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, cwd=root, env=env,
                         check=True, timeout=120).stdout
    return out.splitlines()[-1]


def test_planning_an_evaluation_on_the_login_node_loads_nothing_heavy(tmp_path):
    # sample_task's verifier builds an HF reference engine and its driver a dataset driver; the
    # plan builds both to size their roles and loads neither.
    package = _packaged(tmp_path, task="sample_task")
    provision_module(tmp_path / "store", "tasks.sample_task.task")
    assert json.loads(_fresh(PROBE, str(_config(tmp_path)), package)) == {"heavy": [], "stages": STAGES}


def test_importing_the_stages_registers_every_stage_of_an_evaluation():
    # What the runner imports to find a stage by its kind.
    code = ("import json, hls.stages.registry\nfrom hls.pipeline.stage import STAGES\n"
            "print(json.dumps(sorted(STAGES)))")
    assert set(STAGES) <= set(json.loads(_fresh(code)))


@pytest.mark.parametrize("module", ["hls.drive.role", "hls.verify.role", "hls.score.role", "hls.stages.verify",
                                    "hls.stages.score", "hls.stages.finalize", "hls.pipeline.orchestrator"])
def test_a_role_program_or_a_stage_imports_first_without_a_cycle(module):
    # A role program imports no stage, and each stage imports its role program's constants: whichever
    # of them is imported first meets no cycle.
    assert _fresh(f"import {module}\nprint('imported')") == "imported"


def test_an_evaluation_on_a_site_without_its_secrets_file_is_refused_when_planned(tmp_path, login, in_catalog):
    """Finalize scans every file of the instance for every value of the site's secrets file, so on
    a site with none the evaluation's allocation would only stop as it started."""
    with pytest.raises(PlanError) as e:
        _plan(tmp_path, package=_packaged(tmp_path), config=_config(tmp_path, secrets=None))
    assert str(e.value) == (f"stage finalize needs every value of the site's secrets file, and there is none at "
                            f"{tmp_path / 'secrets.toml'}")
