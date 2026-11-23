"""The baseline_warm stage on the local launcher, with the deploy_drive tests' fake container and
fake server: the measured baseline served once, as its drives serve it,
builds its engine cache, which the stage publishes as the entry every drive starts from; an entry
the store holds already is found, and nothing is started, copied or extracted."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

import hls.drive.readiness as readiness
import hls.pipeline.stage as stage_mod
from hls.egress.policy import SealPolicy
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.planner import StageDecl
from hls.pipeline.stage import StageContext, StageFault
from hls.provision.model import Provisioned, RowsRef, WeightsRef, weights_key
from hls.provision.mounts import copies_gib
from hls.stages import deploy_drive
from hls.stages.baseline_warm import WARM_BASELINE, BaselineWarmParams, BaselineWarmStage, _Warm, load_sec
from hls.stages.deploy_drive import AIR_GAP, DeployDriveParams, served_cache_key
from hls.stages.engine_cache import engine_build_env
from hls.stages.images import KEPT_DISK_GIB
from hls.store.cache import ContentCache, copy_disk_bytes, entry_sha256
from hls.store.engine_cache import ENGINE_CACHE_FILE, ENGINE_CACHE_KIND, EngineCacheRecord
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.docker.images import DockerImages
from tests.drive import fake_container, fake_server
from tests.drive.copy_order import CallOrder
from tests.drive.test_stage import (ARTIFACT_SHAPE, GRADING_NODE, MEASURED, MODEL, REVISION, ROWS, RUNNER_GIB,
                                    _fill_rows, _free_port, _manifest, _params, _server_image, _sif_of)
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.store.entries import populate
from tests.substrate import fake_pasta

#: The measured baseline: the fake server, which builds its kernel into an empty engine cache in
#: half a second before it listens.
BASELINE = {"kind": "image", "image": "measured", "entrypoint": ["python", "/workspace/server.py", "--mode", "serve"],
            "runtime_env": {"ENGINE_FLAG": "1"}}
SERVE = ["--jit-sec", "0.5"]
#: The copies of the artifact role's data: the weights and the rows, a MiB each.
DATA_COPIES = {f"weights/{weights_key(MODEL, REVISION)}": 2**20, f"rows/{ROWS}": 2**20}
#: The provisioned weights, by the repo and revision the key digests.
WEIGHTS = WeightsRef(repo=MODEL, revision=REVISION, key=weights_key(MODEL, REVISION), entry_sha256="0" * 64,
                     disk_bytes=0)
#: Seconds of the stage's time beside the load: 10 s of staging, 5 s for the runtime's start and
#: a 5 s margin.
STAGING, RUNTIME_START, MARGIN = 10, 5, 5


@pytest.fixture
def server_bins(tmp_path, monkeypatch):
    """The fake container runtime and seal on PATH, fast readiness polls, no process ceiling and
    ``GRADING_NODE``'s per-uid limits, as the drive tests set them; returns the bin dir."""
    monkeypatch.setattr("hls.pipeline.stage.hard_limit", lambda: None)
    monkeypatch.setattr(stage_mod, "node_limits", lambda: GRADING_NODE)
    bin_dir = tmp_path / "bin"
    fake_container.install(bin_dir)
    fake_pasta.install(bin_dir)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_PATH", f"{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / "records"))
    (tmp_path / "pasta.plan").write_text("pass\n" * 4)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    (tmp_path / "counter").write_text(str(_free_port() - 40001))
    monkeypatch.setenv("FAKE_PYTHON_COUNTER", str(tmp_path / "counter"))
    monkeypatch.setattr(readiness, "HEALTH_POLL_SEC", 0.1)
    monkeypatch.setattr(readiness, "INFERENCE_POLL_SEC", 0.1)
    monkeypatch.setattr(deploy_drive, "PORT_POLL_SEC", 0.05)
    return bin_dir


def _warm_params(**changes) -> dict:
    return {"task": {"module": "tasks.sample_task.task"}, "artifact": BASELINE, "port": 8000, "serve_command": SERVE,
            "network": None, "artifact_shape": ARTIFACT_SHAPE.model_dump(), "runtime_start_sec": RUNTIME_START,
            "staging_sec": STAGING, "margin_sec": MARGIN, "data_copies": DATA_COPIES, **changes}


def _shape(load: int = 60):
    """The stage's shape: the artifact role's, the runner's memory beside it, the disk of the image
    it extracts and of its copies, and a load of *load* seconds."""
    return ARTIFACT_SHAPE.model_copy(update={
        "memory_gib": ARTIFACT_SHAPE.memory_gib + RUNNER_GIB,
        "disk_gib": ARTIFACT_SHAPE.disk_gib + KEPT_DISK_GIB + copies_gib(DATA_COPIES),
        "time_sec": STAGING + RUNTIME_START + load + MARGIN})


def _image_sha(entry, image_file: str) -> str:
    return next(f.sha256 for f in entry.manifest.files if f.path == image_file)


def _ctx(tmp_path: Path, bin_dir: Path, *, load: int = 60, docker: bool = False, **changes) -> StageContext:
    """The warm stage's attempt over a published provision and images stage, the measured image
    the fake server's, and the weights and rows in the cache. With *docker* the images are
    Docker's runtime's, whose entries hold its own image file."""
    decl = StageDecl(WARM_BASELINE, "baseline_warm", _shape(load), _warm_params(**changes))
    plan = noop_plan(tmp_path, decls=[decl])
    plan = plan.model_copy(update={"toolchain": plan.toolchain.model_copy(update={"python": bin_dir / "python"})})
    store = PosixStore(plan.store_root)
    cache, holder = ContentCache(store), Holder.here("test")
    scratch = tmp_path / "node" / "hls-100"
    images = (DockerImages(scratch, job_id="100", python=sys.executable, buildkit_image="sha256:" + "b" * 64)
              if docker else ApptainerImages(scratch))
    measured = populate(cache, "images", MEASURED, lambda d: _server_image(d / images.image_file, "serve"),
                        holder=holder).entry
    weights = populate(cache, "weights", weights_key(MODEL, REVISION), lambda d: (d / "config.json").write_text("{}"),
                       holder=holder).entry
    rows = populate(cache, "rows", ROWS, _fill_rows, holder=holder).entry
    provisioned = Provisioned(
        weights=WeightsRef(repo=MODEL, revision=REVISION, key=weights.key, entry_sha256=entry_sha256(weights),
                           disk_bytes=copy_disk_bytes(weights)),
        rows=RowsRef(key=ROWS, entry_sha256=entry_sha256(rows), disk_bytes=copy_disk_bytes(rows), tenants=["t0"]),
        tools=[])
    prov_key = plan.layout.attempt_file("provision", 1, "outputs/provisioned.json")
    store.write_bytes(prov_key, provisioned.model_dump_json().encode())
    images_key = plan.layout.attempt_file("images", 1, "outputs/images.json")
    store.write_bytes(images_key, json.dumps({"measured": {"key": measured.key, "entry_sha256": entry_sha256(measured),
                                                           "sif_sha256": _image_sha(measured, images.image_file)}}).encode())
    inputs = {plan.layout.manifest("provision"): _manifest(plan, store, "provision", {"provisioned": prov_key}),
              plan.layout.manifest("images"): _manifest(plan, store, "images", {"images": images_key})}
    events = EventLog(store.path(plan.layout.attempt_file(WARM_BASELINE, 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage(WARM_BASELINE), attempt=1, launcher=LocalLauncher().use_plan(plan),
                        watch=None, events=events, inputs=inputs, deadline=time.time() + 120, cache=cache,
                        scratch=scratch, job_id="100", runtime=ApptainerRuntime(), images=images)


def _key(ctx: StageContext) -> str:
    """The engine cache key of the warm stage's baseline, as the stage computes it."""
    return served_cache_key(ctx.params, _sif_of(ctx.cache.lookup("images", MEASURED)), WEIGHTS)


@pytest.mark.parametrize("docker", [False, True], ids=["apptainer", "docker"])
def test_the_served_cache_key_is_the_digest_of_the_image_file_the_sites_runtime_keeps(tmp_path, server_bins, docker):
    """A Docker images entry holds ``image.oci.tar``, an Apptainer one ``image.sif``: the warm
    stage keys the baseline's engine cache by the sha256 of the file the site's runtime keeps,
    which its drives pin, so the key is computed for a Docker entry and stays the ``image.sif``'s
    on Slurm."""
    ctx = _ctx(tmp_path, server_bins, docker=docker)
    image_file = DockerImages.image_file if docker else "image.sif"
    measured = ctx.cache.lookup("images", MEASURED)
    assert {f.path for f in measured.manifest.files} == {image_file}
    warm = _Warm(ctx)
    sha = warm.image_sha256(warm.image_entries(), "measured")
    assert sha == _image_sha(measured, image_file)


def _record(ctx: StageContext, outputs: dict[str, str]) -> EngineCacheRecord:
    return EngineCacheRecord.model_validate_json(ctx.store.read_bytes(outputs["engine_cache"]))


def _fault(ctx: StageContext) -> StageFault:
    with pytest.raises(StageFault) as e:
        BaselineWarmStage().run(ctx)
    assert all(h.poll() is not None for h in ctx.launcher.handles), "a role outlived the stage"
    return e.value


def test_a_missing_entry_is_built_once_by_serving_the_baseline_as_its_drives_serve_it(tmp_path, server_bins):
    ctx = _ctx(tmp_path, server_bins)
    key = _key(ctx)
    assert ctx.cache.lookup(ENGINE_CACHE_KIND, key) is None
    order = CallOrder(ctx)
    outputs = BaselineWarmStage().run(ctx)
    assert outputs == {"engine_cache": ctx.output_key(ENGINE_CACHE_FILE)}
    # The baseline built its kernel cold, once, and the stage published what it built, its
    # temporary files dropped, under the key a drive of the same baseline computes.
    entry = ctx.cache.lookup(ENGINE_CACHE_KIND, key)
    assert {f.path for f in entry.manifest.files} == {fake_server.KERNEL, fake_server.KERNEL_OBJECT,
                                                      fake_server.COMPILES, fake_server.LOADS}
    assert (entry.path / fake_server.COMPILES).read_text() == "kernel\n"
    assert entry.manifest.source == {"stage": WARM_BASELINE, "instance": ctx.plan.instance}
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry),
                                                      published=True)
    drive = DeployDriveParams.model_validate(_params(artifact=BASELINE, serve_command=SERVE, measured_baseline=True,
                                                     engine_cache_owner=WARM_BASELINE))
    assert served_cache_key(drive, _sif_of(ctx.cache.lookup("images", MEASURED)), WEIGHTS) == key
    # Its data copied and its image pinned before the server, which ran as the harness's, sealed
    # under the air gap as a drive's server is, with its start, its variables and the build
    # environment the key digests.
    assert order.staged() == ["materialize", "materialize", "node_image", "seed"]
    assert order.roles_before_staged() == []
    [spec] = [s for s in ctx.launcher.launched if s.name == "server"]
    assert (spec.owner, spec.shape.cpus, spec.shape.gpus) == ("harness", ARTIFACT_SHAPE.cpus, ARTIFACT_SHAPE.gpus)
    assert SealPolicy(network=AIR_GAP).model_dump_json() in spec.argv
    seen = fake_container.records(tmp_path / "records")["server"]
    assert seen["command"] == [*BASELINE["entrypoint"], "--host", "0.0.0.0", "--port", "8000", *SERVE]
    assert {name: seen["env"][name] for name in engine_build_env("H200")} == engine_build_env("H200")
    assert seen["env"]["ENGINE_FLAG"] == "1" and seen["in_pasta"] and seen["drop_caps"] == "ALL"
    assert f"{ctx.role_dir('server') / 'engine-cache'}:/engine-cache" in seen["binds"]
    [ready] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "baseline_warm.ready"]
    assert ready["after_command_sec"] >= 0.5 and ready["cpus"] == ARTIFACT_SHAPE.cpus
    # The data's two copies, the extraction, the server and the copy that published its cache.
    assert len(ctx.launcher.launched) == BaselineWarmStage.max_roles(ctx.params) == 5
    assert [s.name for s in ctx.launcher.launched][-2:] == ["server", "fill-engine-cache"]


def test_a_published_entry_is_found_and_nothing_is_started_copied_or_extracted(tmp_path, server_bins):
    ctx = _ctx(tmp_path, server_bins)
    key = _key(ctx)
    entry = populate(ctx.cache, ENGINE_CACHE_KIND, key, lambda d: (d / "built").write_text("x"),
                     holder=Holder.here("an earlier instance")).entry
    outputs = BaselineWarmStage().run(ctx)
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry))
    assert ctx.launcher.launched == [] and ctx.image_digests == {}


def test_a_baseline_not_ready_within_its_load_is_infra_and_retried(tmp_path, server_bins):
    # A build that outlasts the stage's load budget leaves nothing to publish; another node may be
    # faster, so the stage is retried, and its time is the knob.
    ctx = _ctx(tmp_path, server_bins, load=2, serve_command=["--jit-sec", "60"])
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "the measured baseline was not ready within its 2s load"
    assert ctx.cache.lookup(ENGINE_CACHE_KIND, _key(ctx)) is None


def test_a_load_the_allocations_end_cuts_is_infra_and_retried(tmp_path, server_bins, monkeypatch):
    # The allocation's end leaves the load two of its 60 s: the stage's time, never the baseline.
    monkeypatch.setattr(stage_mod.StageContext, "hold_until", lambda self, until: False)
    ctx = _ctx(tmp_path, server_bins, serve_command=["--jit-sec", "60"])
    real = ctx.wait_command_started

    def wait_command_started(name):
        began = real(name)
        if began is not None:
            ctx.deadline = began + 2 + MARGIN
        return began

    ctx.wait_command_started = wait_command_started
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "the stage's time cut the readiness gates to 2s of their 60s budget"
    assert ctx.cache.lookup(ENGINE_CACHE_KIND, _key(ctx)) is None


def test_a_load_that_does_not_fit_before_the_allocations_end_is_not_started(tmp_path, server_bins):
    ctx = _ctx(tmp_path, server_bins)
    ctx.deadline_cap = ctx.clock() + RUNTIME_START + 60 + MARGIN - 10
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == (f"the {RUNTIME_START}s allowed for the server's runtime to start its command, its 60s load "
                            f"and the {MARGIN}s margin after it do not fit before the allocation's end, so the server "
                            "was not started")
    assert fault.evidence["deadline_cap"] == str(ctx.deadline_cap)
    assert "server" not in [s.name for s in ctx.launcher.launched]


def test_a_baseline_that_exits_before_it_is_ready_is_infra_and_retried(tmp_path, server_bins):
    ctx = _ctx(tmp_path, server_bins, artifact={**BASELINE, "entrypoint": ["python", "/workspace/server.py", "--mode",
                                                                           "crash"]})
    fault = _fault(ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert ctx.cache.lookup(ENGINE_CACHE_KIND, _key(ctx)) is None


def test_the_stage_is_the_artifact_role_with_the_runners_memory_and_its_disk():
    params = BaselineWarmParams.model_validate(_warm_params())
    BaselineWarmStage.check(_shape(), params, runner_memory_gib=RUNNER_GIB)
    # A site may give the stage more time, which the load gets.
    longer = _shape(load=600)
    BaselineWarmStage.check(longer, params, runner_memory_gib=RUNNER_GIB)
    assert load_sec(params, longer.time_sec) == 600
    for shape in (_shape().model_copy(update={"cpus": 32}), _shape().model_copy(update={"memory_gib": 99}),
                  _shape().model_copy(update={"disk_gib": 1})):
        with pytest.raises(ValueError, match="resources must be the role's"):
            BaselineWarmStage.check(shape, params, runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="leave the baseline's load no time"):
        BaselineWarmStage.check(_shape(load=0), params, runner_memory_gib=RUNNER_GIB)


def test_only_a_measured_image_on_gpus_is_warmed():
    with pytest.raises(ValueError):
        BaselineWarmParams.model_validate(_warm_params(artifact={"kind": "package"}))
    with pytest.raises(ValueError, match="runs on none"):
        BaselineWarmParams.model_validate(_warm_params(artifact_shape=ARTIFACT_SHAPE.model_copy(
            update={"gpus": 0, "gpu_kind": None}).model_dump()))
