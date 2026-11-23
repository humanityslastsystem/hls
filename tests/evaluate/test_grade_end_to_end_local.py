"""``hls grade`` on one machine: the command plans and submits, the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` runs the real runner, and all six stages run their
real roles in one allocation. Each job's node has exactly the disk its allocation asked for,
which every role's grant check counts its scratch against, and a fake job cgroup the runner
watches. Only srun, the cluster, the hub, pasta and the container runtime are stood in for:
the provision role reads the fake hub, the artifact is the deploy_drive tests' fake server in
a fake container image, the seal probe's refusal is answered by the seal's real interceptor on
the loopback, the driver and the verify role run in fake containers on the harness image, and
the verifier's reference engine is the verify tests' fake generator, loaded in the role's own
process."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

import hls.drive.readiness as readiness
from hls.drive.bounds import trace_disk_bytes
from hls.drive.record import DriveRecord
from hls.obs.record import GradingRecord
from hls.pipeline.plan import PlanStore
from hls.pipeline.planner import launch_headroom_sec
from hls.provision.mounts import copies_gib
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages import deploy_drive
from hls.stages.deploy_drive import PackagedArtifact, extracted_disk_gib
from hls.stages.finalize import failed_key
from hls.stages.images import IMAGE_FILE, image_keys
from hls.store.cache import ContentCache
from hls.store.engine_cache import EngineCacheRecord
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import JobState, RoleHandle, RoleReport, RoleSpec
from hls.substrate.slurm.scheduler import submit_argv
from hls.substrate.apptainer.images import ApptainerImages
from tests.drive import fake_container
from tests.drive.interceptor import Interceptor
from tests.drive.test_stage import SERVER, _free_port, _server_image
from tests.evaluate import fixture
from tests.evaluate.warmer import WARMER, publish_package
from tests.pipeline.hls_cli import flags, run_hls
from tests.pipeline.plans import VERSIONS
from tests.pipeline.test_end_to_end_local import KillingLauncher, LocalSlurm
from tests.provision import fake_hub
from tests.provision.warmer_config import SECRETS, TOKEN
from tests.recipes.warmer_site import local_grading_node, site_config
from tests.store.entries import populate
from tests.substrate import fake_pasta

FAKE_ROLE = Path(fake_hub.__file__).with_name("fake_role.py")
STAGES = ["provision", "images", "deploy_drive", "verify", "score", "finalize"]
# The allocation: the verify stage's CPUs and memory (the reference job's 8 CPUs and
# 180 GiB), one GPU, and its time (``_allocation_min``).
ALLOCATION = "allocation 0: provision, images, deploy_drive, verify, score, finalize; 1 x B200, 8 CPUs, 180 GiB"
#: The package warmer whose submission the kernel kills for its memory.
OOM_WARMER = "20261002T000000Z-0000aa"
#: The package warmer whose submission exits before it is ready.
CRASH_WARMER = "20261002T000000Z-0000ab"


class OomLauncher(KillingLauncher):
    """Ends the artifact role's step OUT_OF_MEMORY, as the kernel does when the server reaches
    its own memory limit while it loads: the step's own cgroup counts one limit reached, and the
    job's cgroup one below it and none of its own. The server's process runs as it would."""

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        handle = super().launch(spec, share_gpus_of=share_gpus_of)
        if spec.name == deploy_drive.ARTIFACT:
            report = RoleReport.model_validate_json((spec.log_dir / f"{spec.name}.report.json").read_text())
            step = Path(report.cgroup_dir).parent
            (step / "memory.events").write_text("low 0\nhigh 0\nmax 1\noom 1\noom_kill 1\n")
            (step.parent / "memory.events").write_text("low 0\nhigh 0\nmax 1\noom 1\noom_kill 1\n")
            handle.step_state = JobState.OUT_OF_MEMORY
        return handle


class OomSlurm(LocalSlurm):
    launcher_class = OomLauncher


@pytest.fixture
def bins(tmp_path, monkeypatch):
    """The fake container runtime and seal on PATH, the fake hub, fast readiness polls and no
    process ceiling (an unlimited hard limit), so no test depends on its host's limit; returns
    the bin dir."""
    bin_dir = tmp_path / "bin"
    fake_container.install(bin_dir)
    fake_pasta.install(bin_dir)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_PATH", f"{venv}:/usr/bin:/bin")
    # The seal resolves its firewall tools from here, not from the system's /usr/sbin.
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    (tmp_path / "pasta.plan").write_text("pass\n" * 16)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    (tmp_path / "counter").write_text(str(_free_port() - 40001))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(tmp_path / "counter"))
    for var in ("FAKE_PROXY_ERRORS", "FAKE_APPTAINER_FAIL", "FAKE_TREE_GIB"):
        monkeypatch.delenv(var, raising=False)
    hub = tmp_path / "hub"
    fake_hub.make_hub(hub, "org/model", fixture.SHA, fixture.FILES)
    monkeypatch.setenv("FAKE_HUB", str(hub))
    monkeypatch.setattr(readiness, "HEALTH_POLL_SEC", 0.1)
    monkeypatch.setattr(readiness, "INFERENCE_POLL_SEC", 0.1)
    monkeypatch.setattr(deploy_drive, "PORT_POLL_SEC", 0.05)
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: None)
    return bin_dir


def provision(tmp_path: Path, config: Path, task_module: str, overrides: dict[str, str] | None = None) -> None:
    """``hls provision`` of *task_module*'s task, with the task's *overrides*, through an in-process
    Slurm of its own, as an operator warms the cache before an evaluation: the plan sizes the
    drive's copies of the task's data by what the cache holds. Its jobs' ids are its own, so
    the evaluation's job is 100 as before."""
    code, lines = run_hls(_slurm(tmp_path), ["provision", task_module, *flags(overrides or {}), "--job-name",
                                             "hls-provision-test", "--config", str(config)])
    assert code == 0, lines


def _slurm(tmp_path: Path, cls: type[LocalSlurm] = LocalSlurm) -> LocalSlurm:
    """An in-process Slurm whose jobs run as soon as they are submitted, on a node with exactly
    the disk each asked for and a fake job cgroup, with the provision role program served by
    the fake hub."""
    slurm = cls(replace_modules={"hls.provision.role": FAKE_ROLE}, disk_free_gib=None, node_disk=True,
                job_cgroups=tmp_path / "cgroups")

    def submit_and_run(request):
        job_id = LocalSlurm.submit(slurm, request)
        slurm.tick()
        return job_id

    slurm.submit = submit_and_run
    return slurm


def _cache_images(store: PosixStore, baseline: bytes) -> None:
    """Publish the task's harness chain and its measured baseline as ``hls images build`` would
    have: the harness layer is an image the fake container runtime can run."""
    keys = image_keys(order(catalog.task_images(fixture.TASK_ID).decls), ApptainerImages.recipe_key)
    cache, holder = ContentCache(store), Holder.here("test")
    layer = catalog.image_set("sample_task").layer
    populate(cache, "images", keys["s-a"], lambda d: (d / IMAGE_FILE).write_bytes(b"s-a image"), holder=holder)
    populate(cache, "images", keys[layer], lambda d: fake_container.make_image(
        d / IMAGE_FILE, {".singularity.d/labels.json": (b"{}", 0o644)}), holder=holder)
    populate(cache, "images", keys["evaluate-task-baseline"], lambda d: (d / IMAGE_FILE).write_bytes(baseline),
             holder=holder)


def _allocation_min(store: PosixStore, instance: str) -> int:
    """The whole minutes the grading *instance*'s one allocation asks for: every stage's time in
    turn as its plan holds it (the drive's after its staging, the copies and images it reads at the
    site's rate), the runner's overhead and the time for the roles' steps to start."""
    plan = PlanStore(store, InstanceLayout(instance)).latest()
    assert plan is not None
    return (sum(s.shape.time_sec for s in plan.stages) + plan.settings.run.allocation_overhead_sec
            + launch_headroom_sec(plan.stages, plan.settings.site.step_launch_allowance_sec)) // 60


def _grade(tmp_path: Path, bin_dir: Path, slurm: LocalSlurm, *, package: str | None = None,
           measured: str | None = None, records: str, task_module: str = "ev_task.task") -> tuple[int, list[str]]:
    """``hls grade <task_module>`` with *package* or *measured*, its containers recorded under
    *records*; the exit status and what it said. The seal probe's refusal check dials the seal's
    interceptor, fresh for each grading, which writes its status where the probe's seal keeps it
    on the job's node."""
    config = site_config(tmp_path, site=f'pasta_path = "{bin_dir / "pasta"}"\n{local_grading_node()}',
                         secrets=SECRETS)
    provision(tmp_path, config, task_module)
    status = tmp_path / "node" / "hls-100" / "roles" / deploy_drive.SEAL_PROBE / "seal" / "status.json"
    interceptor = Interceptor(status, deploy_drive.AIR_GAP)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(deploy_drive, "PROBE_DIAL", interceptor.address)
            mp.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / records))
            artifact = ["--package", package] if package is not None else []
            artifact += ["--measured", measured] if measured is not None else []
            return run_hls(slurm, ["grade", task_module, *artifact, "--job-name", "hls-grade-test",
                                   "--config", str(config)])
    finally:
        interceptor.close()


def _output(store: PosixStore, instance: str, stage: str, name: str) -> bytes:
    manifest, _ = read_manifest(store, InstanceLayout(instance), stage)
    return store.read_bytes(manifest.outputs[name].key)


def _holding(store: PosixStore, root: str, value: str) -> list[Path]:
    return [p for p in store.path(root).rglob("*") if p.is_file() and value.encode() in p.read_bytes()]


def test_hls_grade_runs_the_whole_path_in_one_allocation_warms_the_baseline_and_scores_a_crash_and_an_oom_zero(
        tmp_path, login, in_catalog, bins):
    store = PosixStore(tmp_path / "store")
    baseline = fake_container.make_image(tmp_path / "baseline.sif", {"workspace/server.py": (SERVER.read_bytes(),
                                                                                             0o644)}).read_bytes()
    in_catalog(roles=("baseline",))
    _cache_images(store, baseline)
    package = publish_package(store, _server_image(tmp_path / "submission.sif", "serve").read_bytes())
    slurm = _slurm(tmp_path)

    code, lines = _grade(tmp_path, bins, slurm, package=package, records="records-1")
    assert code == 0, lines
    instance = lines[0].removeprefix("instance ")
    grading = instance.removeprefix(f"{package}/gradings/")
    assert lines[1] == f"{ALLOCATION}, {_allocation_min(store, instance)} min"
    idle = [line for line in lines if line.startswith("warning: allocation 0 holds 1 GPUs")]
    assert [line.split("; stage ")[1].split(" ")[0] for line in idle] == ["provision", "images", "score", "finalize"]
    assert lines[-2] == f"{instance}: succeeded: every stage published"
    [job] = slurm.jobs.values()
    args = submit_argv(job.request)
    assert {"--partition=gpu-a", "--gpus=b200:1", "--cpus-per-task=8", "--mem=180G",
            "--job-name=hls-grade-test"} <= set(args)
    assert all(read_manifest(store, InstanceLayout(instance), s) is not None for s in STAGES)
    drive = DriveRecord.model_validate_json(_output(store, instance, "deploy_drive", "drive"))
    assert (drive.outcome, drive.trace.events) == ("served", 4)
    verified = json.loads(_output(store, instance, "verify", "verify"))
    assert [(e["role"], e["engine"]) for e in verified["engines"]] == [("reference", "FakeGenerator")]
    assert verified["libraries"]["python"] == ".".join(map(str, sys.version_info[:3]))
    record = GradingRecord.model_validate_json(_output(store, instance, "finalize", "grading"))
    assert (record.source_rollout, record.mode, record.episode.eval_id) == (WARMER, "adopt", grading)
    assert record.episode.checkpoint == drive.artifact.image_sha256
    # The tools its job's runner found on the node.
    assert [a.tool_versions for a in record.conditions.allocations] == [{job.job_id: VERSIONS}]
    score = record.episode.score
    assert score["gate_passed"] and score["value"] > 0
    score_key = read_manifest(store, InstanceLayout(instance), "score")[0].outputs["score"].key
    assert lines[-1] == f"{instance}: score {score['value']} ({score_key})"
    row = json.loads(store.read_bytes(f"warmers/{fixture.TASK_ID}/index/{WARMER}+{grading}.json"))
    assert (row["rollout_id"], row["grading_id"], row["status"], row["scanned"]) == (WARMER, grading, "ok", True)
    # The node had exactly the allocation's disk, and every role's grant was checked against what
    # the job's scratch left of it. The cache held every image, so the images stage holds none, and
    # the drive stage's disk, the artifact's, the driver's for the trace it writes on the node, the
    # images it extracts and the copies of its roles' data size the allocation.
    plan = PlanStore(store, InstanceLayout(instance)).latest()
    assert plan.stage("images").shape.disk_gib == 0
    data = copies_gib(plan.stage("deploy_drive").params["data_copies"])
    artifact = plan.stage("deploy_drive").params["artifact_shape"]["disk_gib"]
    # The fixture's 1 GiB cap and 1 MiB of requests: a room of 2,150,237,314 bytes, 3 GiB.
    trace = plan.stage("deploy_drive").params["driver_shape"]["disk_gib"]
    bounds = fixture.DRIVE_BOUNDS
    assert trace == math.ceil(trace_disk_bytes(bounds.trace_cap_bytes, bounds.trace_request_bytes) / 2**30) == 3
    assert plan.allocations[0].request.disk_gib == plan.stage("deploy_drive").shape.disk_gib == (
        artifact + trace + extracted_disk_gib(PackagedArtifact()) + data)
    # The hub token reached the hub through the provision stage's grant, and no file of the
    # evaluation holds it.
    assert {c["token"] for c in fake_hub.calls(tmp_path / "hub")} == {TOKEN}
    assert _holding(store, instance, TOKEN) == []

    # A grading of the measured baseline first warms its engine cache: its warm stage serves the
    # baseline once and publishes what it built, and the drive's server starts from a copy of that
    # entry, which the drive's record names. The packaged artifact's
    # server above started with an empty one.
    assert drive.artifact.engine_cache is None
    code, again = _grade(tmp_path, bins, _slurm(tmp_path), measured="baseline", records="records-2")
    assert code == 0, again
    second = again[0].removeprefix("instance ")
    assert second.startswith(f"warmers/{fixture.TASK_ID}/measured-baseline/gradings/g-")
    assert again[-2] == f"{second}: succeeded: every stage published"
    warm = EngineCacheRecord.model_validate_json(_output(store, second, "warm_baseline", "engine_cache"))
    assert warm.published and warm.entry is not None
    drive = DriveRecord.model_validate_json(_output(store, second, "deploy_drive", "drive"))
    assert (drive.outcome, drive.trace.events) == ("served", 4)
    assert drive.artifact.engine_cache == EngineCacheRecord(key=warm.key, seeded_from=warm.entry)
    records = fake_container.records(tmp_path / "records-2")
    command = [*fixture.BASELINE_ENTRYPOINT, "--host", "0.0.0.0", "--port", "8000", *fixture.BASELINE_SERVE]
    # The warm stage served the baseline as the drive did: the same command and the same variables,
    # the build environment the entry's key digests among them.
    assert records["server"]["command"] == records["artifact"]["command"] == command
    assert records["server"]["env"] == records["artifact"]["env"]
    assert records["artifact"]["env"]["FAKE_SERVER_FLAVOR"] == "baseline"
    assert PlanStore(store, InstanceLayout(second)).latest().stage("deploy_drive").params["measured_baseline"]
    assert store.exists(f"warmers/{fixture.TASK_ID}/index/measured-baseline+{second.rsplit('/', 1)[1]}.json")
    assert _holding(store, f"warmers/{fixture.TASK_ID}/measured-baseline", TOKEN) == []

    # A submission that exits before it is ready: a result the contestant's own, scored zero.
    crash = publish_package(store, _server_image(tmp_path / "crash.sif", "crash").read_bytes(), warmer=CRASH_WARMER)
    code, crashed = _grade(tmp_path, bins, _slurm(tmp_path), package=crash, records="records-4")
    assert code == 0, crashed
    fourth = crashed[0].removeprefix("instance ")
    drive = DriveRecord.model_validate_json(_output(store, fourth, "deploy_drive", "drive"))
    assert drive.outcome == "crashed" and drive.trace is None and drive.artifact.engine_cache is None
    score = json.loads(_output(store, fourth, "score", "score"))
    assert (score["value"], score["zero_reasons"]) == (0.0, ["crashed"])
    assert crashed[-2:] == [f"{fourth}: succeeded: every stage published",
                            f"{fourth}: score 0.0, zero for crashed "
                            f"({read_manifest(store, InstanceLayout(fourth), 'score')[0].outputs['score'].key})"]
    assert "verify" not in fake_container.records(tmp_path / "records-4")

    # The kernel kills a submission that reaches its own memory limit while it loads: the
    # contestant's fault, which ends the instance failed. Nothing after the drive runs, and the
    # orchestrator records the instance's zero and its index row.
    oom = publish_package(store, _server_image(tmp_path / "oom.sif", "crash").read_bytes(), warmer=OOM_WARMER)
    code, third = _grade(tmp_path, bins, _slurm(tmp_path, OomSlurm), package=oom, records="records-3")
    assert code == 0, third
    failed = third[0].removeprefix("instance ")
    plan = PlanStore(store, InstanceLayout(failed)).latest()
    assert (plan.outcome.status, plan.outcome.fault.fault_class) == ("failed", "contestant")
    assert plan.outcome.fault.reason == "contestant role artifact ran out of memory"
    assert [s for s in STAGES if read_manifest(store, InstanceLayout(failed), s) is not None] == ["provision",
                                                                                                    "images"]
    zero_key = failed_key(InstanceLayout(failed), "score.json")
    zero = json.loads(store.read_bytes(zero_key))
    assert (zero["value"], zero["zero_reasons"]) == (0.0, ["contestant_fault"])
    named = zero["breakdown"]["contestant_fault"]
    assert (named["stage"], named["reason"]) == ("deploy_drive", "contestant role artifact ran out of memory")
    # The kill was the artifact's own limit for certain: its step's own count, not the job's.
    evidence = named["evidence"]
    assert (evidence["artifact.own_oom"], evidence["artifact.job_oom"]) == ("True", "False")
    assert third[-1] == f"{failed}: score 0.0, zero for contestant_fault ({zero_key})"
    grading = failed.rsplit("/", 1)[1]
    row = json.loads(store.read_bytes(f"warmers/{fixture.TASK_ID}/index/{OOM_WARMER}+{grading}.json"))
    assert (row["rollout_id"], row["grading_id"], row["score"], row["failed_stage"]) == (
        OOM_WARMER, grading, 0.0, "deploy_drive")
    assert "verify" not in fake_container.records(tmp_path / "records-3")
