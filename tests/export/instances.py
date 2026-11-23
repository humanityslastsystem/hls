"""Finished instances for the export's tests, on the finalize tests' published fixture
(``tests.finalize.published``): a finalized rollout or grading, a rollout a contestant fault
ended, a package warmer, a calibration's servings, and the export stage's own plan and run."""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from hls.arms import ArmRecord
from hls.calibrate import serving_seed_of
from hls.core.trace import Trace
from hls.pipeline.checkout import CodeState
from hls.pipeline.export import EXPORT_SHAPE
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, STAGE_NAME_PATTERNS
from hls.pipeline.plan import Failure, Outcome, Plan, Toolchain
from hls.pipeline.planner import StageDecl, build_plan
from hls.pipeline.stage import StageContext
from hls.stages.agent_build import AgentBuildStage
from hls.stages.calibration import (SCRATCH_GIB, CaptureStage, ScoreShareStage, capture_stage, score_stage,
                                    stage_shape, warm_stage)
from hls.stages.engine_cache import ENGINE_CACHE_GIB
from hls.stages.export import ExportStage
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.finalize import FinalizeStage, finalize_params, record_failed
from hls.stages.package import AGENT_BUILD_STAGE, PackageStage
from hls.stages.score import ScoreStage
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord
from hls.substrate.secret_forms import FORMS
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan, PackageScanArea, PackageScanFinding
from hls.substrate.secrets import Scrubber
from tests.finalize.published import (INSTALL_SCAN, ROLLOUT, SIF, TASK, TREE_SCAN, context, make_plan,
                                      package_result, publish, publish_instance, publish_rollout_package,
                                      run_and_publish, with_params, with_stages, write)
from tests.calibration import fixture as calibrations
from tests.finalize.test_agent_build_scan import ARM
from tests.finalize.stand_in_stage import AgentStandInStage
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import settings
from tests.trace.traces import drive_trace, passing_verdict

#: A secret no contestant holds, and a route credential an agent was handed.
SECRET, RELAY = "hf-token-0123456789abcdef", "relay-key-0123456789abcdef"
#: The registry with both kinds of secret.
BOTH = {"hf_token": SECRET, "relay": RELAY}
#: The secrets file a site holds them in.
SECRETS_FILE = f'[secrets]\nhf_token = "{SECRET}"\nrelay = "{RELAY}"\n'
#: The fixture task's module every instance runs.
MODULE = "fin_goodput.task"
#: The fixture package warmer's id, and the export's own.
WARMER, EXPORT_ID = "20261003T000000Z-0000bb", "20261004T000000Z-0000cc"
#: The stage that scans the source bundle's objects and its checked-out tree.
SOURCE_SCAN = "source_scan"
#: A source bundle an agent build publishes.
BUNDLE = b"PACK the agent's history"


def facing(plan: Plan) -> Plan:
    """*plan* with a stage first that hands its grants to its contestant, as agent_build does, and
    ``relay`` granted to it."""
    agent = plan.stages[0].model_copy(update={"name": "agent", "kind": AgentStandInStage.kind, "params": {},
                                              "after": [], "inputs": {}, "grants": {"RELAY_KEY": "relay"}})
    return plan.model_copy(update={"stages": [agent, *plan.stages]})


def with_package(plan: Plan) -> Plan:
    """*plan* naming the package stage the fixture publishes, as a rollout's plan does."""
    package = plan.stage("score").model_copy(update={"name": "package", "kind": PackageStage.kind, "params": {},
                                                     "after": [], "inputs": {}})
    return with_stages(plan, package)


def agent_built(plan: Plan) -> Plan:
    """*plan* with a stage that hands its contestant ``relay`` (``facing``), and naming the
    agent_build and package stages the fixture publishes by their own kinds, which declare their
    containers (``Stage.compressed_files``, ``Stage.unpublished_containers``,
    ``Stage.published_dirs``), as a rollout's plan has them."""
    plan = facing(plan)
    model = plan.stage("score")
    build = model.model_copy(update={"name": AGENT_BUILD_STAGE, "kind": AgentBuildStage.kind, "params": {},
                                     "after": [], "inputs": {}})
    package = model.model_copy(update={"name": "package", "kind": PackageStage.kind, "params": {},
                                       "after": [], "inputs": {}})
    return with_stages(plan, build, package)


def scan_record(registry: dict[str, str], read: list[str], findings: list[PackageScanFinding] | None = None,
                areas: list[PackageScanArea] | None = None, forms: tuple[str, ...] = FORMS) -> bytes:
    """A package scan's ``package-scan.json``: it read the areas *read*, searching for every secret
    of *registry* in *forms*, every form by default, and found *findings* and *areas*."""
    return PackageScan(secrets=sorted(registry), read=read, findings=findings or [], areas=areas or [],
                       forms=list(forms)).model_dump_json().encode()


def ended(store: PosixStore, plan: Plan, *, status: str = "succeeded", fault: FaultRecord | None = None,
          stage: str = "deploy_drive") -> Plan:
    """Write *plan* to the store as its first and latest revision, ended *status*: one that ended
    failed at *stage* with *fault*, which its allocation records as the orchestrator does;
    "running" for one that has not ended."""
    if status == "failed":
        assert fault is not None
        failure = Failure(at=1500.0, stage=stage, fault=fault)
        plan = plan.model_copy(update={"allocations": [plan.allocations[0].model_copy(update={"failure": failure}),
                                                       *plan.allocations[1:]]})
    if status != "running":
        plan = plan.model_copy(update={"outcome": Outcome(status=status, reason=f"{status} for the test", at=1500.0,
                                                          fault=fault)})
    store.create(plan.layout.plan_revision(plan.revision), plan.model_dump_json().encode())
    return plan


def finalized(tmp_path: Path, checkout: Path, *, plan: Plan | None = None, trace: Trace | None = None,
              kind: str = "rollout", registry: dict[str, str] | None = None, bundle: bytes | None = None,
              build_extra: dict[str, tuple[str, bytes]] | None = None, tree_scan: bytes | None = TREE_SCAN,
              install_scan: bytes | None = INSTALL_SCAN, score_attempt: int = 1) -> tuple[Plan, PosixStore]:
    """A served instance of the fixture task, scored in attempt *score_attempt*, finalized with
    *registry* as its secrets (``BOTH`` by default), and ended succeeded; *plan* replaces the
    fixture's plan of *kind*. With *bundle*, an agent_build publishes it as its source bundle
    first, and *build_extra* beside it; the source_scan stage publishes *tree_scan* and the
    package *install_scan* as their ``package-scan.json`` (no stage, or none, for None), by
    default records of the fixture's two secrets that found nothing (``publish_instance``)."""
    plan = plan or make_plan(tmp_path, checkout, MODULE, kind=kind)
    store = PosixStore(plan.store_root)
    trace = trace or drive_trace()
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace), bundle=bundle,
                     build_extra=build_extra, tree_scan=tree_scan, install_scan=install_scan)
    run_and_publish(store, plan, "score", ScoreStage, attempt=score_attempt)
    run_and_publish(store, plan, "finalize", FinalizeStage, secrets=registry or BOTH)
    return ended(store, plan), store


def failed(tmp_path: Path, checkout: Path, *, log: str) -> tuple[Plan, PosixStore]:
    """A rollout of the fixture task, with a route granted to its agent, that a contestant fault of
    its drive ended failed: provision, images and package published, the drive's one attempt
    holding *log* as its artifact's stderr and a clean driver log, and the orchestrator's records
    of the failure written. Finalize never ran, and no scan read inside the image."""
    plan = with_package(facing(make_plan(tmp_path, checkout, MODULE)))
    store = PosixStore(plan.store_root)
    publish(store, plan, "provision", {"provisioned": write(store, plan, "provision", "provisioned.json",
                                                            b'{"weights": null}')})
    publish(store, plan, "images", {"images": write(store, plan, "images", "images.json", b'{"sa-base": "built"}')})
    publish(store, plan, "package", {
        "package": write(store, plan, "package", "package.json", package_result(plan, invalid=False)),
        "submission": write(store, plan, "package", "submission.sif", SIF)})
    store.write_bytes(plan.layout.attempt_file("deploy_drive", 1, "logs/artifact.err"), log.encode())
    store.write_bytes(plan.layout.attempt_file("deploy_drive", 1, "logs/driver.err"), b"driver started\n")
    fault = FaultRecord(fault_class="contestant", retryable=False,
                        reason="contestant role artifact ran out of memory at its own limit")
    plan = ended(store, plan, status="failed", fault=fault)
    record_failed(store, plan, finalize_params(plan), scrubber=Scrubber(BOTH))
    return plan, store


def graded_package(tmp_path: Path, checkout: Path, *, built_arm: ArmRecord | None,
                   tree_scan: bytes = TREE_SCAN) -> tuple[Plan, PosixStore]:
    """A grading of the fixture rollout's package, which the agent build of ``ARM`` made (its
    ``build.json`` names the arm) after a source_scan that recorded *tree_scan*, its finalize
    naming *built_arm*, and its drive pinning that package; nothing of the grading published
    yet."""
    pinned = InstanceLayout.rollout(TASK, ROLLOUT).manifest("package")
    plan = make_plan(tmp_path, checkout, MODULE, kind="grading", artifact={"kind": "package", "manifest": pinned})
    store = PosixStore(plan.store_root)
    pins = publish_rollout_package(store, plan, build=json.dumps({"arm": ARM.model_dump(mode="json")}).encode(),
                                   tree_scan=tree_scan)
    stages = [s.model_copy(update={"inputs": {**s.inputs, **pins}}) if s.name == "deploy_drive" else s
              for s in plan.stages]
    plan = plan.model_copy(update={"stages": stages})
    arm = built_arm.model_dump(mode="json") if built_arm is not None else None
    return with_params(plan, "finalize", built_arm=arm), store


def package_warmer(tmp_path: Path, checkout: Path, *, findings: list[PackageScanFinding],
                   warmer: str = WARMER, tree_scan: bytes | None = None) -> tuple[Plan, PosixStore]:
    """A package warmer of the fixture task (``hls package``) of id *warmer*, ended succeeded: its
    source_scan stage scanned the operator's bundle and its tree and found *findings*, or wrote
    *tree_scan* when given, and its package stage, which read source_scan's, published a
    submission image and a scan of the install that found nothing."""
    base = make_plan(tmp_path, checkout, MODULE)
    package = base.stage("score").model_copy(update={"name": "package", "kind": PackageStage.kind, "params": {},
                                                     "after": [], "inputs": {}})
    plan = base.model_copy(update={"instance": InstanceLayout.warmer(TASK, warmer).root, "pipeline": "package",
                                   "stages": [package],
                                   "allocations": [base.allocations[0].model_copy(update={"stages": ["package"]})]})
    store = PosixStore(plan.store_root)
    tree = tree_scan if tree_scan is not None else scan_record(BOTH, ["bundle", "tree"], findings)
    publish(store, plan, SOURCE_SCAN, {"package_scan": write(store, plan, SOURCE_SCAN, PACKAGE_SCAN_FILE, tree)})
    source = plan.layout.manifest(SOURCE_SCAN)
    publish(store, plan, "package", {
        "package": write(store, plan, "package", "package.json", package_result(plan, invalid=False)),
        "submission": write(store, plan, "package", "submission.sif", SIF),
        "package_scan": write(store, plan, "package", PACKAGE_SCAN_FILE, INSTALL_SCAN)},
        inputs={source: sha256_bytes(store.read_bytes(source))})
    return ended(store, plan), store


def calibration(tmp_path: Path, checkout: Path, calibration_id: str, *,
                draws: Sequence[int]) -> tuple[Plan, PosixStore]:
    """A calibration of the calibration tests' token-level task that ended succeeded, with a
    capture stage of its vllm member for each of *draws* and one score stage that scored them all,
    declared as ``hls.pipeline.calibration.serving_stages`` declares them, and one attempt of each
    claimed; the stages before them, whose files hold no gzip, are left out."""
    campaign = calibrations.campaign(engines=["vllm"])
    base = settings(tmp_path)
    disk = KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB
    member, reference = (stage_shape(role, runner_memory_gib=base.run.runner_memory_gib, disk_gib=disk, time_sec=600)
                         for role in (campaign.member_shape, campaign.reference_shape))
    common = {"task": {"module": "cal_flips.task"}, "campaign": campaign.model_dump(mode="json"), "data_copies": {}}
    captures = [StageDecl(capture_stage("vllm", d), CaptureStage.kind, member,
                          {**common, "engine": "vllm", "draw": d, "seed": serving_seed_of(campaign.task_seed, d),
                           "warm_stage": warm_stage("vllm")}) for d in draws]
    score = StageDecl(score_stage(1), ScoreShareStage.kind, reference,
                      {**common, "share": 1, "servings": [["vllm", d] for d in draws]},
                      after=tuple(c.name for c in captures))
    plan = build_plan(decls=[*captures, score], task=calibrations.TASK_ID, pipeline="calibration",
                      layout=InstanceLayout.calibration(calibrations.TASK_ID, calibration_id), settings=base,
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-calibrate", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0,
                      known_patterns=STAGE_NAME_PATTERNS)
    store = PosixStore(plan.store_root)
    for stage in plan.stages:
        store.write_bytes(plan.layout.attempt_file(stage.name, 1, "CLAIM"), b"")
    return ended(store, plan), store


def export_plan(tmp_path: Path, checkout: Path, instances: list[str], destination: Path, *,
                oracle: bool = False, export_id: str = EXPORT_ID) -> Plan:
    """An export's plan of *instances* to *destination*, as ``plan_export`` would make it."""
    decl = StageDecl("export", ExportStage.kind, EXPORT_SHAPE,
                     {"instances": instances, "oracle": oracle, "destination": str(destination)})
    return build_plan(decls=[decl], task=TASK, pipeline="export", layout=InstanceLayout.export(TASK, export_id),
                      settings=settings(tmp_path),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-export", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=3000.0,
                      known_patterns=STAGE_NAME_PATTERNS, partition="cpu")


def run_export(store: PosixStore, plan: Plan, *, registry: dict[str, str] | None = None,
               launcher: LocalLauncher | None = None) -> tuple[StageContext, dict[str, str]]:
    """Run the export stage of *plan* with *registry* as the site's secrets (``BOTH`` by default)."""
    ctx = context(store, plan, "export", secrets=BOTH if registry is None else registry, launcher=launcher)
    return ctx, ExportStage().run(ctx)


def destination(tmp_path: Path, name: str = "export") -> Path:
    """A destination that does not exist yet, in a directory that does."""
    (tmp_path / "out").mkdir(exist_ok=True)
    return tmp_path / "out" / name
