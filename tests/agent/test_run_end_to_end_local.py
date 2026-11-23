"""``hls run`` and ``hls grade`` on one machine: the commands plan and submit, the in-process Slurm
of ``tests/pipeline/test_end_to_end_local.py`` runs the real runner, and every stage runs its
real roles in one allocation per instance, on a node with exactly the disk each allocation asked
for and a fake job cgroup. The build half stands in for what the agent build's tests stand in for (the
hub, pasta, the seal probe's checks, the agent's CLI driven by harbor's real adapter, the
sandbox's container, pip and the offline build) and the grading half for what the evaluate
tests do (the artifact's and the driver's containers, the seal probe's interceptor, the reference
engine). One ``apptainer`` on PATH hands each command to its fake (``tests.agent.run_fakes``).

These tests need harbor at a pin with ``SessionReentry``, as the agent build's tests do."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

import hls.drive.readiness as readiness
from hls.agent import environment, harnesses
from hls.drive.record import DriveRecord
from hls.obs.record import GradingRecord, RolloutRecord
from hls.package.result import PackageResult
from hls.pipeline import stage as stage_module
from hls.pipeline.plan import PlanStore
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages import agent_build, deploy_drive
from hls.stages.finalize import failed_key
from hls.stages.images import IMAGE_FILE, image_keys
from hls.store.cache import ContentCache
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import NodeSize
from hls.substrate.apptainer.images import ApptainerImages
from tests.agent import fake_cli, fake_sandbox, run_fakes
from tests.agent.build_fixture import KEY, TASK_ID, build_config
from tests.agent.test_stage import MappedEnvironment
from tests.drive import fake_container
from tests.drive.interceptor import Interceptor
from tests.drive.test_stage import _free_port
from tests.evaluate import fixture
from tests.evaluate.warmer import publish_package
from tests.package import fake_tools
from tests.package.bundles import PYPROJECT
from tests.package.bundles import SERVER as BUNDLE_SERVER
from tests.package.test_stage import REPORT
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.hls_cli import run_hls
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.provision import fake_hub
from tests.provision.warmer_config import TOKEN
from tests.store.entries import populate
from tests.substrate import fake_pasta
from tests.support import node

FAKE_ROLE = Path(fake_hub.__file__).with_name("fake_role.py")
ROLLOUT = ["provision", "images", "agent_build", "source_scan", "package", "deploy_drive", "verify", "score",
           "finalize"]
LAYER = catalog.image_set("sample_task").layer
CAP = catalog.AGENT_CAPS[LAYER].name
#: The agent cap's image file: an image the sandbox fake extracts, with nothing at /workspace.
CAP_IMAGE = fake_sandbox.image_bytes()
#: The node every job runs on: one GPU and 16 CPUs, the build allocation's size and at least each
#: grading's, which holds one agent build or served artifact and two package stages of their sizes.
JOB_NODE = NodeSize(gpus=1, cpus=16)


@pytest.fixture
def bins(tmp_path, tmp_path_factory, short_dir, monkeypatch, fake_seal):
    """Every fake on PATH behind one ``apptainer``, the fake hub, the build's seal probe answered as
    a working seal answers it (``fake_seal``), the runner's clock moved by the fake CLI, the
    runner's environment clean of what harbor would read, fast readiness polls and the commands'
    node, whose ceiling this host's kernel sets on the build sandbox's real commands
    (``tests.support.node``), on ``JOB_NODE``. Returns (bin dir, the CLI's control directory, the
    package fake's plan, the node's short scratch)."""
    node.require_commands_ceiling()
    for var in harnesses.harness_env_leaks(os.environ):
        monkeypatch.delenv(var)
    base = tmp_path_factory.mktemp("b")
    bin_dir, cli_dir, control = base / "bin", base / "cli", base / "control"
    fake_pasta.install(bin_dir)
    sandbox = fake_sandbox.install(base / "sandbox")
    tools = base / "tools.json"
    fake_tools.install(base / "package", tools)
    container = fake_container.install(base / "container")
    run_fakes.install(bin_dir, sandbox=sandbox, package=base / "package" / "apptainer", container=container,
                      cap=sha256_bytes(CAP_IMAGE)[:12])
    tools.write_text(json.dumps({"dry_run": "ok", "wheel": "ok", "build": "ok", "report": REPORT,
                                 "record": str(base / "tools.log"),
                                 "image_changes": run_fakes.serving_image_changes()}))
    fake_cli.install(cli_dir, control)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{base / 'package'}:{venv}:/usr/bin:/bin")
    # The seal resolves its firewall tools from here, not from the system's /usr/sbin.
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    monkeypatch.setenv("FAKE_SANDBOX_PATH", f"{cli_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_SANDBOX_RECORDS", str(base / "records"))
    monkeypatch.setenv("FAKE_CONTAINER_PATH", f"{venv}:/usr/bin:/bin")
    (base / "pasta.plan").write_text("pass\n" * 32)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(base / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(base / "pasta.log"))
    (base / "counter").write_text(str(_free_port() - 40001))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(base / "counter"))
    for var in ("FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL", "FAKE_PROXY_ERRORS", "FAKE_PROXY_MID_STREAM",
                "FAKE_PROXY_OVERSIZED", "FAKE_PROXY_REFUSED", "FAKE_PROXY_HEARTBEAT_AGE", "FAKE_TREE_GIB"):
        monkeypatch.delenv(var, raising=False)
    hub = tmp_path / "hub"
    fake_hub.make_hub(hub, "org/model", fixture.SHA, fixture.FILES)
    monkeypatch.setenv("FAKE_HUB", str(hub))
    offset = control / "offset"
    monkeypatch.setattr(agent_build, "monotonic",
                        lambda: time.monotonic() + (float(offset.read_text()) if offset.exists() else 0.0))
    monkeypatch.setattr(stage_module, "hard_limit", lambda: node.COMMANDS_HARD)
    monkeypatch.setattr(LocalLauncher, "node_size", lambda self: JOB_NODE if self.node is None else self.node)
    monkeypatch.setattr(environment, "SandboxEnvironment", MappedEnvironment)
    monkeypatch.setattr(MappedEnvironment, "SENT", [])
    monkeypatch.setattr(MappedEnvironment, "TIMEOUTS", [])
    monkeypatch.setattr(agent_build, "_ipv4s", lambda host: [])  # no DNS from a test
    monkeypatch.setattr(readiness, "HEALTH_POLL_SEC", 0.1)
    monkeypatch.setattr(readiness, "INFERENCE_POLL_SEC", 0.1)
    monkeypatch.setattr(deploy_drive, "PORT_POLL_SEC", 0.05)
    (control / "script.json").write_text(json.dumps({"window": 1_000_000, "preflight": {}, "claude": [
        {"commit": BUNDLE_SERVER, "commit_files": {"pyproject.toml": PYPROJECT}, "advance": 900}]}))
    return bin_dir, control, tools, short_dir


def _slurm(tmp_path: Path) -> LocalSlurm:
    """An in-process Slurm whose jobs run as soon as they are submitted, on a node with exactly
    the disk each asked for and a fake job cgroup, the provision role served by the fake hub."""
    slurm = LocalSlurm(replace_modules={"hls.provision.role": FAKE_ROLE}, disk_free_gib=None, node_disk=True,
                       job_cgroups=tmp_path / "cgroups")

    def submit_and_run(request):
        job_id = LocalSlurm.submit(slurm, request)
        slurm.tick()
        return job_id

    slurm.submit = submit_and_run
    return slurm


def _cache_images(store: PosixStore) -> None:
    """The base, the layer (the submission base and the harness image) and the agent cap, as ``hls
    images build`` would have cached them: the base and the layer as tree images. The images
    stage finds every entry and judges none, since an entry is judged only when it is filled, so
    the cap's inventory is never read here. A test that fills the cap through the images stage
    instead must report ``tests.recipes.warmer_site.inventory_for`` as the built image's inventory
    (``FAKE_INVENTORY``, as the images end to end does), which keeps every tool and package the
    brief promises, or the stage refuses the cap."""
    keys = image_keys(order(catalog.task_images(TASK_ID).decls), ApptainerImages.recipe_key)
    cache, holder = ContentCache(store), Holder.here("test")
    for name, data in (("s-a", fake_tools.tree_image(fake_tools.BASE_TREE)),
                       (LAYER, fake_tools.tree_image(fake_tools.BASE_TREE)), (CAP, CAP_IMAGE)):
        populate(cache, "images", keys[name], lambda d, data=data: (d / IMAGE_FILE).write_bytes(data), holder=holder)


def _driven(args: list[str], scratch: Path, tmp_path: Path, config: Path) -> tuple[int, list[str]]:
    """Run ``hls provision`` of the task, which the plan sizes the copies of its data by (a warm
    cache's is a no-op), then ``hls <args>`` (a run or a grading of the task), each with the
    hls.toml at *config* on a fresh in-process Slurm, the latter with the drive's seal probe's
    interceptor; return the exit status and what the command printed."""
    code, provisioned = run_hls(_slurm(tmp_path), ["provision", "bd_task.task", "--job-name",
                                                   "hls-provision-test", "--config", str(config)])
    assert code == 0, provisioned
    status = scratch / "hls-100" / "roles" / deploy_drive.SEAL_PROBE / "seal" / "status.json"
    interceptor = Interceptor(status, deploy_drive.AIR_GAP)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(deploy_drive, "PROBE_DIAL", interceptor.address)
            return run_hls(_slurm(tmp_path), [*args, "--config", str(config)])
    finally:
        interceptor.close()


def _config(tmp_path: Path, bin_dir: Path, scratch: Path) -> Path:
    return build_config(tmp_path, site=f'pasta_path = "{bin_dir / "pasta"}"', scratch=scratch)


def _output(store: PosixStore, instance: str, stage: str, name: str) -> bytes:
    manifest, _ = read_manifest(store, InstanceLayout(instance), stage)
    return store.read_bytes(manifest.outputs[name].key)


def _run(tmp_path: Path, bins, *args: str) -> tuple[int, list[str]]:
    """``hls run bd_task.task --arm arm-a <args>``, after its provision."""
    bin_dir, _, _, scratch = bins
    return _driven(["run", "bd_task.task", "--arm", "arm-a", "--job-name", "hls-run-test", *args], scratch, tmp_path,
                   _config(tmp_path, bin_dir, scratch))


def _grade(tmp_path: Path, bins, *args: str) -> tuple[int, list[str]]:
    """``hls grade bd_task.task <args>``, after its provision."""
    bin_dir, _, _, scratch = bins
    return _driven(["grade", "bd_task.task", "--job-name", "hls-grade-test", *args], scratch, tmp_path,
                   _config(tmp_path, bin_dir, scratch))


def _instance(lines: list[str]) -> str:
    return next(line for line in lines if line.startswith("instance ")).removeprefix("instance ")


def test_hls_run_runs_build_and_grading_in_one_allocation_end_to_end(tmp_path, build_login, build_catalog, bins):
    store = PosixStore(tmp_path / "store")
    _cache_images(store)
    code, lines = _run(tmp_path, bins)
    assert code == 0, lines
    instance = _instance(lines)
    assert instance.startswith(f"runs/{TASK_ID}/")
    assert any(line.startswith(f"allocation 0: {', '.join(ROLLOUT)}; 1 x B200") for line in lines)
    assert f"{instance}: succeeded: every stage published" in lines
    layout = InstanceLayout(instance)
    assert all(read_manifest(store, layout, s) is not None for s in ROLLOUT)
    # The agent's commit was packaged, and the drive served that package's image.
    package = PackageResult.model_validate_json(_output(store, instance, "package", "package"))
    drive = DriveRecord.model_validate_json(_output(store, instance, "deploy_drive", "drive"))
    assert package.status == "built" and drive.outcome == "served"
    assert drive.artifact.image_sha256 == package.sif_sha256
    # The rollout's record holds its build, its package and its own grading.
    record = RolloutRecord.model_validate_json(_output(store, instance, "finalize", "rollout"))
    assert record.status == "ok" and record.build is not None
    assert record.build.build["arm"]["name"] == "arm-a" and record.build.package["status"] == "built"
    grading = GradingRecord.model_validate_json(_output(store, instance, "finalize", "grading"))
    assert (grading.mode, grading.episode.eval_id) == ("original", "terminal")
    score_key = read_manifest(store, layout, "score")[0].outputs["score"].key
    assert lines[-1] == f"{instance}: score {grading.episode.score['value']} ({score_key})"
    row = json.loads(store.read_bytes(f"runs/{TASK_ID}/index/{instance.rsplit('/', 1)[1]}.json"))
    assert (row["arm"], row["status"], row["scanned"], row["grading_id"]) == ("arm-a", "ok", True, None)
    # The arm's credential reached the CLI and no published file; the hub token reached the hub.
    assert [p for p in store.path(instance).rglob("*") if p.is_file() and KEY.encode() in p.read_bytes()] == []
    assert {c["token"] for c in fake_hub.calls(tmp_path / "hub")} == {TOKEN}


def test_a_build_only_rollout_then_its_grading_and_its_rebuild_end_to_end(tmp_path, build_login, build_catalog,
                                                                          bins):
    store = PosixStore(tmp_path / "store")
    _cache_images(store)
    code, lines = _run(tmp_path, bins, "--build-only")
    assert code == 0, lines
    rollout = _instance(lines)
    record_key = read_manifest(store, InstanceLayout(rollout), "finalize")[0].outputs["rollout"].key
    assert lines[-1] == f"{rollout}: rollout built ({record_key})"
    record = RolloutRecord.model_validate_json(store.read_bytes(record_key))
    assert record.status == "built" and record.build.package["status"] == "built"
    row = json.loads(store.read_bytes(f"runs/{TASK_ID}/index/{rollout.rsplit('/', 1)[1]}.json"))
    assert (row["status"], row["score"], row["gate_passed"]) == ("built", None, None)
    package = PackageResult.model_validate_json(_output(store, rollout, "package", "package"))

    # Its grading serves the rollout's own image, pinned by its manifest's digest.
    code, graded = _grade(tmp_path, bins, "--rollout", rollout)
    assert code == 0, graded
    grading = _instance(graded)
    assert grading.startswith(f"{rollout}/gradings/g-")
    drive = DriveRecord.model_validate_json(_output(store, grading, "deploy_drive", "drive"))
    assert drive.outcome == "served" and drive.artifact.image_sha256 == package.sif_sha256
    record = GradingRecord.model_validate_json(_output(store, grading, "finalize", "grading"))
    assert (record.mode, record.source_rollout) == ("adopt", rollout.rsplit("/", 1)[1])
    assert store.exists(f"runs/{TASK_ID}/index/{rollout.rsplit('/', 1)[1]}+{grading.rsplit('/', 1)[1]}.json")

    # With --artifact.rebuild the grading scans and packages the rollout's bundle again, at its own
    # checkout.
    code, rebuilt = _grade(tmp_path, bins, "--rollout", rollout, "--artifact.rebuild")
    assert code == 0, rebuilt
    again = _instance(rebuilt)
    assert read_manifest(store, InstanceLayout(again), "source_scan") is not None
    package_again = PackageResult.model_validate_json(_output(store, again, "package", "package"))
    assert (package_again.status, package_again.commit) == ("built", package.commit)
    record = GradingRecord.model_validate_json(_output(store, again, "finalize", "grading"))
    assert record.mode == "rebuild"


def test_a_grading_whose_server_echoes_the_relay_key_quarantines_and_scores(tmp_path, build_login, build_catalog,
                                                                            bins):
    """The arm held its relay key, so a server it built may send it back; a grading of the
    rollout's package counts the key contestant-facing, quarantines what holds it, and scores. The
    same image served from a package warmer, which no agent built, holds a secret no contestant
    was given: our leak, which refuses the grading."""
    _, _, tools, _ = bins
    plan = json.loads(tools.read_text())
    tools.write_text(json.dumps({**plan, "image_changes": run_fakes.serving_image_changes("echo", reply=KEY)}))
    store = PosixStore(tmp_path / "store")
    _cache_images(store)
    code, lines = _run(tmp_path, bins, "--build-only")
    assert code == 0, lines
    rollout = _instance(lines)

    code, graded = _grade(tmp_path, bins, "--rollout", rollout)
    assert code == 0, graded
    grading = _instance(graded)
    assert f"{grading}: succeeded: every stage published" in graded
    quarantine = json.loads(_output(store, grading, "finalize", "quarantine"))
    drive_files = [k for k in quarantine["files"] if f"{grading}/stages/deploy_drive/" in k]
    assert drive_files and all(names == ["anthropic_key"] for names in quarantine["files"].values())
    row = json.loads(store.read_bytes(f"runs/{TASK_ID}/index/{rollout.rsplit('/', 1)[1]}+"
                                      f"{grading.rsplit('/', 1)[1]}.json"))
    assert row["scanned"] and row["quarantined_count"] == len(quarantine["files"]) and row["score"] is not None

    # The same image from a package warmer: the key is no contestant's, so the grading is void.
    image = _output(store, rollout, "package", "submission")
    warmer = publish_package(store, image, task=TASK_ID)
    code, refused = _grade(tmp_path, bins, "--package", warmer)
    assert code == 1, refused
    failed = _instance(refused)
    ended = PlanStore(store, InstanceLayout(failed)).latest().outcome
    assert ended.status == "void" and "the harness leaked secret values" in ended.reason
    assert KEY not in ended.reason
    assert not store.exists(failed_key(InstanceLayout(failed), "score.json"))

    # The same image as a published file, graded again by its digest. With the arm that built it
    # declared, its relay key is contestant-facing: the drive's files are quarantined and scored.
    published = tmp_path / "published" / "submission.sif"
    published.parent.mkdir()
    published.write_bytes(image)
    digest = sha256_bytes(image)
    code, declared = _grade(tmp_path, bins, "--artifact.image", f"sha256:{digest}", "--from", str(published),
                            "--built-arm", "arm-a")
    assert code == 0, declared
    regrade = _instance(declared)
    assert regrade.startswith(f"warmers/{TASK_ID}/image-{digest}/gradings/g-")
    drive = DriveRecord.model_validate_json(_output(store, regrade, "deploy_drive", "drive"))
    assert drive.outcome == "served" and drive.artifact.image_sha256 == digest
    record = GradingRecord.model_validate_json(_output(store, regrade, "finalize", "grading"))
    assert record.provenance.artifact == {"kind": "published", "sha256": digest, "source": str(published)}
    assert record.provenance.rollout_overrides == {}
    assert json.loads(_output(store, regrade, "finalize", "quarantine"))["files"]

    # Without it, no arm is known, so the key is ours and the grading is void.
    code, undeclared = _grade(tmp_path, bins, "--artifact.image", f"sha256:{digest}", "--from", str(published))
    assert code == 1, undeclared
    failed = _instance(undeclared)
    ended = PlanStore(store, InstanceLayout(failed)).latest().outcome
    assert ended.status == "void" and "the harness leaked secret values" in ended.reason

    # A file with other bytes than the digest names is refused when it is read, once: the
    # grading is void and its drive stage ran one attempt.
    other = tmp_path / "published" / "other.sif"
    other.write_bytes(image + b"\n")
    code, refused = _grade(tmp_path, bins, "--artifact.image", f"sha256:{digest}", "--from", str(other),
                           "--built-arm", "arm-a")
    assert code == 1, refused
    failed = _instance(refused)
    assert PlanStore(store, InstanceLayout(failed)).latest().outcome.status == "void"
    assert InstanceLayout(failed).attempts(store, "deploy_drive") == [1]
