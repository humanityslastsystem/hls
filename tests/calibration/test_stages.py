"""The calibration stages on the local launcher, with the deploy_drive tests' fake container. Each
role runs ``tests.calibration.fake_roles``, the real role program with the fake engines: in a fake
container on its pinned measured image, with the reference's or a serving's data mounted, or on
the plan's interpreter for the aggregate. A role's refusal is infra that is not retried; any other
failure of it, infra that is."""

from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
import math
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from hls.calibrate import CaptureStore, capture_items, load_sessions, score_draw, serving_seed_of
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import EventLog
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK, REQUEST_TIMEOUT_SEC, STAGE_NAME_PATTERNS
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.stage import StageContext, StageFault
from hls.provision.model import Provisioned, RowsRef, ToolRef, WeightsRef, rows_key, tool_key, weights_key
from hls.provision.mounts import copies_gib
from hls.spec.datasets import Split
from hls.spec.resources import ResourceShape
from hls.stages import calibration as stages
from hls.stages.calibration import (SCRATCH_GIB, AggregateStage, CaptureParams, CaptureStage, InstrumentStage,
                                    ScoreShareParams, ScoreShareStage, WarmParams, WarmStage, capture_stage,
                                    member_seal, stage_shape)
from hls.stages import engine_cache
from hls.stages.engine_cache import ENGINE_CACHE_GIB, engine_build_env, engine_cache_key
from hls.egress.policy import SealPolicy
from hls.stages.deploy_drive import AIR_GAP
from hls.stages.engine_cache_dirs import ENGINE_CACHE_VARIABLES
from hls.stages.images import KEPT_DISK_GIB
from hls.store.cache import CacheEntry, ContentCache, copy_disk_bytes, entry_sha256
from hls.store.layout import InstanceLayout
from hls.store.digest import sha256_bytes
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.docker.images import DockerImages
from hls.substrate.faults import RoleOutcome, infra_fault
from hls.substrate.sandbox import sandbox_paths
from hls.substrate.secrets import Scrubber
from hls.verify import calibration_roles as roles
from hls.verify.calibration import CALIBRATION_FILE, PreflightResult, StackInfo
from hls.store.engine_cache import ENGINE_CACHE_FILE, ENGINE_CACHE_KIND, EngineCacheRecord
from tests.calibration import engines, fixture
from tests.drive import fake_container
from tests.drive.copy_order import CallOrder
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import settings
from tests.store.entries import populate
from tests.substrate import fake_pasta
from tests.unit.test_calibrate import _tf_item
from tests.verify_stage.test_stage import _manifest, _sif_of

#: The memory the test plans leave to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4
#: The roles' shape: one B200, four CPUs and 16 GiB.
SHAPE = fixture.ROLE_SHAPE
#: The cache key of each measured image the tests' ``images.json`` names: vllm's pool member's and
#: the reference's, each a fake container image of its own.
IMAGE_KEYS = {"evaluate-task-member-vllm": "member-vllm-0123456789abcdef",
              "evaluate-task-reference": "reference-0123456789abcdef"}
REPO_ID, REVISION = "org/model", "c" * 40
ROWS = rows_key("0123456789abcdef")
#: The partition digest of the tool bundle's dev variant, and none for its full one.
DEV_DIGESTS = {"full": None, "dev": "0123456789ab"}
#: The cache key of each variant of the tasks' one tool bundle.
TOOL_KEYS = {variant: tool_key("tiny", "v1", variant, digest) for variant, digest in DEV_DIGESTS.items()}
#: The campaign of these stages: vllm alone, on ``cal_flips``'s token-level gate, its exposure
#: target 40,000 tokens.
CAMPAIGN = fixture.campaign(engines=["vllm"], mu_per_mtok=math.log(40) * 1e6 / 40_000)
FINGERPRINT = CAMPAIGN.workload_fingerprint
TASK = {"module": "cal_flips.task"}
#: Stage name to output name to its path under the attempt's outputs and its bytes.
Files = Mapping[str, Mapping[str, tuple[str, bytes]]]
#: The copies of a role's data, as the plan declares them (``data_copies``): the weights, the rows
#: and the full tool bundle ``_ctx`` publishes, a MiB each, more than each copy takes, which holds
#: either role's (the reference role copies no rows).
DATA_COPIES = {f"weights/{weights_key(REPO_ID, REVISION)}": 2**20, f"rows/{ROWS}": 2**20,
               f"tools/{TOOL_KEYS['full']}": 2**20}
#: GiB they take on the node.
DATA_GIB = copies_gib(DATA_COPIES)


@pytest.fixture
def container(tmp_path, monkeypatch) -> Path:
    """The fake container runtime and seal on PATH, before this interpreter's directory, each
    seal passing its payload through (``tests.substrate.fake_pasta``, its calls logged in
    ``pasta.log``), and the stages' roles run as ``tests.calibration.fake_roles``; returns the
    directory of the containers' records."""
    bin_dir = tmp_path / "bin"
    fake_container.install(bin_dir)
    fake_pasta.install(bin_dir)
    venv = Path(sys.executable).parent
    monkeypatch.setenv("PATH", f"{bin_dir}:{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_PATH", f"{venv}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / "records"))
    # The seal resolves its firewall tools from here, not from the system's /usr/sbin.
    monkeypatch.setenv("HLS_SEAL_TOOL_PATH", str(bin_dir))
    (tmp_path / "pasta.plan").write_text("pass\n" * 4)
    monkeypatch.setenv("FAKE_PASTA_PLAN", str(tmp_path / "pasta.plan"))
    monkeypatch.setenv("FAKE_PASTA_LOG", str(tmp_path / "pasta.log"))
    monkeypatch.setattr(stages, "ROLE_MODULE", "tests.calibration.fake_roles")
    return tmp_path / "records"


def _decl(name: str, kind: str, role: ResourceShape, disk_gib: int, params: dict, *,
          campaign: roles.CampaignSettings = CAMPAIGN,
          data_copies: dict[str, int] | None = DATA_COPIES) -> StageDecl:
    """A stage of *kind* that runs one role of shape *role*, with *disk_gib* of disk beside its
    copies of the role's data, which it declares (*data_copies*; None for a stage that copies
    none)."""
    copies = {} if data_copies is None else {"data_copies": data_copies}
    shape = stage_shape(role, runner_memory_gib=RUNNER_GIB, disk_gib=disk_gib + copies_gib(data_copies or {}),
                        time_sec=600)
    return StageDecl(name, kind, shape, {"task": TASK, "campaign": campaign.model_dump(mode="json"), **copies,
                                         **params})


def _capture(draw: int = 3, *, campaign: roles.CampaignSettings = CAMPAIGN,
             data_copies: dict[str, int] = DATA_COPIES) -> StageDecl:
    return _decl(capture_stage("vllm", draw), "calibrate_capture", SHAPE,
                 KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB,
                 {"engine": "vllm", "draw": draw, "seed": serving_seed_of(fixture.flips_task().seed, draw),
                  "warm_stage": "warm_vllm"}, campaign=campaign, data_copies=data_copies)


def _score(draws: tuple[int, ...] = (0, 2), *, data_copies: dict[str, int] = DATA_COPIES) -> StageDecl:
    return _decl("score_1", "calibrate_score", SHAPE, KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB,
                 {"share": 1, "servings": [["vllm", d] for d in draws]}, data_copies=data_copies)


def _warm(*, campaign: roles.CampaignSettings = CAMPAIGN, data_copies: dict[str, int] = DATA_COPIES) -> StageDecl:
    """vllm's warm stage, of the 600 s every stage of these tests has: 100 s of copies and 100 s of
    margin leave its load 400 s."""
    return _decl("warm_vllm", "calibrate_warm", SHAPE, KEPT_DISK_GIB + ENGINE_CACHE_GIB,
                 {"engine": "vllm", "staging_sec": 100, "margin_sec": 100}, campaign=campaign,
                 data_copies=data_copies)


def _instrument(*, data_copies: dict[str, int] = DATA_COPIES) -> StageDecl:
    return _decl("instrument", "calibrate_instrument", CAMPAIGN.instrument_shape(), KEPT_DISK_GIB + ENGINE_CACHE_GIB,
                 {}, data_copies=data_copies)


def _aggregate() -> StageDecl:
    shape = ResourceShape(cpus=2, memory_gib=4 + RUNNER_GIB, time_sec=600)
    return StageDecl("aggregate", "calibrate_aggregate", shape,
                     {"task": TASK, "campaign": CAMPAIGN.model_dump(mode="json"), "score_stages": ["score_1"]})


def _plan(tmp_path: Path, checkout: Path, *decls: StageDecl) -> Plan:
    return build_plan(decls=list(decls), task=NOOP_TASK, pipeline="calibration",
                      layout=InstanceLayout.calibration(NOOP_TASK, "c1"), settings=settings(tmp_path),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-calibrate", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(),
                      known_patterns=STAGE_NAME_PATTERNS, now=1000.0)


def _image_sha(entry, image_file: str) -> str:
    return next(f.sha256 for f in entry.manifest.files if f.path == image_file)


def _ctx(tmp_path: Path, checkout: Path, stage: str, *decls: StageDecl, files: Files | None = None,
         docker: bool = False) -> StageContext:
    """The context of *stage*'s attempt in a plan of *decls*, over a published provision and
    images stage, owners of vllm's and the reference's engine caches that left no entry
    (``LEFT_NONE``), a fake container image for each measured name (``IMAGE_KEYS``), the weights,
    the test rows and both variants of a tool bundle in the cache, and *files*, the outputs of
    other published stages. With *docker* the images are Docker's runtime's, whose entries hold
    its own image file."""
    plan = _plan(tmp_path, checkout, *decls)
    store = PosixStore(plan.store_root)
    cache, holder = ContentCache(store), Holder.here("test")
    scratch = tmp_path / "node" / "hls-100"
    images = (DockerImages(scratch, job_id="100", python=sys.executable, buildkit_image="sha256:" + "b" * 64)
              if docker else ApptainerImages(scratch))
    pinned = {}
    for name, image_key in IMAGE_KEYS.items():
        image = populate(cache, "images", image_key, lambda d, name=name: fake_container.make_image(
            d / images.image_file, {".singularity.d/labels.json": (b"{}", 0o644), "image.txt": (name.encode(), 0o644)}),
                         holder=holder).entry
        pinned[name] = {"key": image_key, "entry_sha256": entry_sha256(image),
                        "sif_sha256": _image_sha(image, images.image_file)}
    key = weights_key(REPO_ID, REVISION)
    weights = populate(cache, "weights", key, lambda d: (d / "config.json").write_text("{}"), holder=holder).entry

    def fill_rows(d: Path) -> None:
        for split, data in (("dev", [{"q": "0"}]), ("test", fixture.TEST_ROWS)):
            (d / "tenants" / "t0" / split).mkdir(parents=True)
            (d / "tenants" / "t0" / split / "rows.json").write_text(json.dumps(data))

    entry = populate(cache, "rows", ROWS, fill_rows, holder=holder).entry
    tools = []
    for variant, tkey in TOOL_KEYS.items():
        bundle = populate(cache, "tools", tkey, lambda d, v=variant: (d / "bundle.txt").write_text(v),
                          holder=holder).entry
        tools.append(ToolRef(name="tiny", identity="v1", variant=variant, dev_digest=DEV_DIGESTS[variant], key=tkey,
                             entry_sha256=entry_sha256(bundle), disk_bytes=copy_disk_bytes(bundle)))
    provisioned = Provisioned(weights=WeightsRef(repo=REPO_ID, revision=REVISION, key=key,
                                                 entry_sha256=entry_sha256(weights), disk_bytes=copy_disk_bytes(weights)),
                              rows=RowsRef(key=ROWS, entry_sha256=entry_sha256(entry), tenants=["t0"],
                                           disk_bytes=copy_disk_bytes(entry)), tools=tools)
    published: Files = {
        "provision": {"provisioned": ("provisioned.json", provisioned.model_dump_json().encode())},
        "images": {"images": ("images.json", json.dumps(pinned).encode())},
        **{owner: {"engine_cache": (ENGINE_CACHE_FILE, LEFT_NONE.model_dump_json().encode())}
           for owner in ("warm_vllm", "instrument")},
        **(files or {})}
    inputs = {}
    for name, outputs in published.items():
        keys = {}
        for output, (path, data) in outputs.items():
            keys[output] = plan.layout.attempt_file(name, 1, f"outputs/{path}")
            store.write_bytes(keys[output], data)
        inputs[plan.layout.manifest(name)] = _manifest(plan, store, name, keys)
    events = EventLog(store.path(plan.layout.attempt_file(stage, 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage(stage), attempt=1, launcher=LocalLauncher(gpus=2).use_plan(plan),
                        watch=None, events=events, inputs=inputs, deadline=time.time() + 120, scrubber=Scrubber({}),
                        cache=cache, scratch=scratch, job_id="100", runtime=ApptainerRuntime(), images=images)


def _spec(ctx: StageContext, role: str):
    [spec] = [s for s in ctx.launcher.launched if s.name == role]
    return spec


def _fault(stage: type, ctx: StageContext) -> StageFault:
    with pytest.raises(StageFault) as e:
        stage().run(ctx)
    assert all(h.poll() is not None for h in ctx.launcher.handles), "a role outlived the stage"
    return e.value


def _captures(tmp_path: Path, rows: Path, draws: tuple[int, ...]) -> Files:
    """The fake member's captures of vllm's servings *draws*, as their capture stages published
    them."""
    task = fixture.flips_task()
    sessions = load_sessions(task, rows, Split.TEST)
    made = CaptureStore(tmp_path / "made", FINGERPRINT)
    files = {}
    for draw in draws:
        trace = asyncio.run(engines.FakeMember().serve_traffic(
            sessions, seed=serving_seed_of(task.seed, draw), request_timeout_sec=REQUEST_TIMEOUT_SEC))
        made.write_capture("vllm", draw, serving_seed_of(task.seed, draw), engines.MEMBER_VERSION,
                           capture_items(trace, engine="vllm", draw=draw))
        files[capture_stage("vllm", draw)] = {"capture": (roles.capture_file(FINGERPRINT, "vllm", draw),
                                                          made.capture_path("vllm", draw).read_bytes())}
    return files


def _cache_key(ctx: StageContext, image: str, config: str) -> str:
    """The engine cache key of an engine of *config* on the measured image *image* of
    ``IMAGE_KEYS``, on the roles' B200, on the network its roles run on: a member's sealed as
    ``CAMPAIGN`` seals it, the reference's open."""
    seal = None if image == "evaluate-task-reference" else member_seal(CAMPAIGN)
    return engine_cache_key(_sif_of(ctx.cache.lookup("images", IMAGE_KEYS[image])), "B200", config, seal=seal)


def _engine_cache(ctx: StageContext, key: str, files: Mapping[str, bytes]) -> CacheEntry:
    """Engine cache entry *key* published holding *files*, as an owner of the engine left it."""

    def fill(d: Path) -> None:
        for rel, data in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(data)

    return populate(ctx.cache, ENGINE_CACHE_KIND, key, fill, holder=Holder.here("another stage")).entry


def _owner(ctx: StageContext, stage: str, record: EngineCacheRecord) -> None:
    """Owner *stage* published, with *record* as its ``engine-cache.json``, among *ctx*'s inputs."""
    key = ctx.plan.layout.attempt_file(stage, 1, f"outputs/{ENGINE_CACHE_FILE}")
    ctx.store.write_bytes(key, record.model_dump_json().encode())
    ctx.inputs[ctx.plan.layout.manifest(stage)] = _manifest(ctx.plan, ctx.store, stage, {"engine_cache": key})


def _record(ctx: StageContext, outputs: Mapping[str, str]) -> EngineCacheRecord:
    return EngineCacheRecord.model_validate_json(ctx.store.read_bytes(outputs["engine_cache"]))


#: An owner's record that left no entry.
LEFT_NONE = EngineCacheRecord(key="B200-0123456789abcdef-fedcba9876543210", seeded_from=None,
                              unpublished="the publish failed: the tree was refused")
#: An engine cache the member's kernel was built into, as its warm stage published it: the kernel,
#: the object file it was linked from, and the logs; never the build's temporary files.
MEMBER_BUILT = {engines.KERNEL.format("member"): b"member for NVIDIA B200\n", engines.KERNEL_OBJECT.format("member"):
                b"member", engines.COMPILES: b"member\n", engines.LOADS_LOG: b"member\n"}


# ---- capture ----------------------------------------------------------------

def test_a_capture_runs_its_member_in_a_container_on_the_members_measured_image(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    order = CallOrder(ctx)
    outputs = CaptureStage().run(ctx)
    name = roles.capture_file(FINGERPRINT, "vllm", 3)
    assert outputs == {"capture": ctx.output_key(name), "engine_cache": ctx.output_key(ENGINE_CACHE_FILE)}
    # The image, the weights, the test rows and the full tool bundle are pinned and copied before
    # the role starts. Its owner left no engine cache, so the member starts its own empty one.
    assert order.staged() == ["node_image", "materialize", "materialize", "materialize"]
    assert order.while_running() == [] and order.roles_before_staged() == []
    assert len(ctx.launcher.launched) == 5
    spec = _spec(ctx, "capture")
    assert (spec.owner, spec.shape) == ("harness", SHAPE)
    # The member's measured image, not the reference's.
    sha = _sif_of(ctx.cache.lookup("images", IMAGE_KEYS["evaluate-task-member-vllm"]))
    assert ctx.image_digests == {"capture": sha}
    seen = fake_container.records(container)["capture"]
    root, _ = sandbox_paths(ctx.scratch, sha)
    assert seen["image"] == str(root) and seen["nv"] is True
    rows = ctx.materialize("rows", ROWS, ctx.cache_entries[f"rows/{ROWS}"])
    role = ctx.scratch / "roles" / "capture"
    assert set(seen["binds"]) >= {f"{ctx.plan.code.checkout}:/opt/hls-src:ro",
                                  f"{rows}/tenants/t0/test:/data/tenants/t0/test:ro",
                                  f"{role / 'view'}:/hls/view", f"{role / 'engine-cache'}:/engine-cache",
                                  f"{ctx.outputs_dir}:/hls/out", f"{ctx.outputs_report('capture')}:/hls/outputs.json"}
    seed = serving_seed_of(fixture.flips_task().seed, 3)
    assert seen["command"] == ["python", "-m", "tests.calibration.fake_roles", "capture", "--task-ref",
                               "/hls/task-ref.json", "--campaign", "/hls/campaign.json", "--rows", "/data",
                               "--engine", "vllm", "--draw", "3", "--seed", str(seed), "--view", "/hls/view",
                               "--out", "/hls/out", "--outputs-out", "/hls/outputs.json"]
    env = seen["env"]
    assert (env["PYTHONPATH"], env["TORCH_CUDA_ARCH_LIST"], env["VLLM_CACHE_ROOT"], env["HF_HUB_OFFLINE"]) == (
        "/opt/hls-src/src:/opt/hls-src", "10.0", "/engine-cache/vllm", "1")
    # The role runs with exactly the build environment its engine cache key digests.
    assert {name: env[name] for name in engine_build_env("B200")} == engine_build_env("B200")
    # Sealed with no egress, as a graded artifact is served, and reading the hub offline: the
    # member measures the stack it loads from the provisioned weights, and nothing it could fetch.
    assert spec.argv[:2] == ("bash", "-c") and SealPolicy(network=AIR_GAP).model_dump_json() in spec.argv
    assert len(fake_pasta.calls(tmp_path / "pasta.log")) == 1 and env["NO_PROXY"] == "*"
    with gzip.open(ctx.store.path(outputs["capture"]), "rt") as fh:
        header, *records = [json.loads(line) for line in fh]
    assert header == {"engine": "vllm", "draw": 3, "seed": seed, "serving_version": engines.MEMBER_VERSION}
    assert len(records) == len(fixture.TEST_ROWS)


def test_a_capture_whose_owner_left_no_engine_cache_starts_cold_and_publishes_none(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    outputs = CaptureStage().run(ctx)
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None)
    assert (ctx.role_dir("capture") / "engine-cache" / engines.COMPILES).read_text() == "member\n"
    assert ctx.cache.lookup(ENGINE_CACHE_KIND, key) is None
    assert not [s.name for s in ctx.launcher.launched if s.name.startswith("fill-")]


def test_a_capture_is_seeded_with_the_entry_its_owner_left(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    entry = _engine_cache(ctx, key, MEMBER_BUILT)
    _owner(ctx, "warm_vllm", EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry), published=True))
    order = CallOrder(ctx)
    outputs = CaptureStage().run(ctx)
    assert order.staged() == ["node_image", "materialize", "materialize", "materialize", "seed"]
    # The member found its kernel built and wrote beside it: the seed is a copy of its own.
    cache = ctx.role_dir("capture") / "engine-cache"
    assert (cache / engines.COMPILES).read_text() == "member\n"
    assert (cache / engines.LOADS_LOG).read_text() == "member\nmember\n"
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=entry_sha256(entry))
    assert ctx.cache_entries[f"{ENGINE_CACHE_KIND}/{key}"] == entry_sha256(entry)
    assert (ctx.cache.lookup(ENGINE_CACHE_KIND, key).path / engines.LOADS_LOG).read_text() == "member\n"
    # The extraction, the three copies of its data, the seed's copy and the capture role.
    assert len(ctx.launcher.launched) == CaptureStage.max_roles(ctx.params) == 6


@pytest.mark.parametrize("damage, says", [
    (lambda e: (e.path / engines.COMPILES).write_bytes(b"tampered"), f"is corrupt: changed {engines.COMPILES}"),
    (lambda e: (e.path / "ENTRY.json").write_text("{"), "is corrupt: .* is not a valid entry manifest"),
], ids=["file", "manifest"])
def test_a_corrupt_engine_cache_is_refused_before_the_member_starts(tmp_path, checkout, container, damage, says):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    entry = _engine_cache(ctx, key, MEMBER_BUILT)
    _owner(ctx, "warm_vllm", EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry), published=True))
    damage(entry)
    fault = _fault(CaptureStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert re.search(f"cache entry {ENGINE_CACHE_KIND}/{key} {says}", fault.reason)
    assert "capture" not in [s.name for s in ctx.launcher.launched]


def test_a_capture_is_seeded_from_its_owners_build_and_no_other(tmp_path, checkout, container):
    # Two builds of one engine are not byte for byte the same: an entry of another build under the
    # key, as if removed and published again since the owner left its own, is refused.
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    other = _engine_cache(ctx, key, {**MEMBER_BUILT, engines.KERNEL.format("member"): b"another build"})
    _owner(ctx, "warm_vllm", EngineCacheRecord(key=key, seeded_from=None, entry="f" * 64, published=True))
    fault = _fault(CaptureStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == (f"cache entry {ENGINE_CACHE_KIND}/{key} has ENTRY.json digest {entry_sha256(other)}, not "
                            f"the {'f' * 64} its producer recorded")
    assert "capture" not in [s.name for s in ctx.launcher.launched]


def test_a_capture_whose_owner_keyed_its_entry_otherwise_is_refused(tmp_path, checkout, container):
    # The warm stage and the capture would key one engine differently, and every retry would too.
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    other = f"{key[:-16]}{'0' * 16}"
    entry = _engine_cache(ctx, other, MEMBER_BUILT)
    _owner(ctx, "warm_vllm", EngineCacheRecord(key=other, seeded_from=None, entry=entry_sha256(entry),
                                               published=True))
    fault = _fault(CaptureStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == (f"the engine's owner warm_vllm left the engine cache entry {other}, and this stage keys "
                            f"the engine {key}")
    assert "capture" not in [s.name for s in ctx.launcher.launched]


def test_a_member_whose_tools_reach_the_network_runs_open(tmp_path, checkout, container):
    # An agentic task whose tools call the network itself (no provisioned assets) runs its members
    # on the step's own network; the mount wall still has every engine
    # read the provisioned weights offline.
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3",
               _capture(campaign=CAMPAIGN.model_copy(update={"member_internet": True})))
    CaptureStage().run(ctx)
    spec = _spec(ctx, "capture")
    assert spec.argv[:2] != ("bash", "-c") and SealPolicy(network=AIR_GAP).model_dump_json() not in spec.argv
    assert fake_pasta.calls(tmp_path / "pasta.log") == []
    env = fake_container.records(container)["capture"]["env"]
    assert env["HF_HUB_OFFLINE"] == "1" and "NO_PROXY" not in env


def test_a_roles_refusal_is_infra_that_is_not_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3",
               _capture(campaign=CAMPAIGN.model_copy(update={"workload_fingerprint": "f" * 64})))
    fault = _fault(CaptureStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith("calibration capture refused: InputMismatch: cal_flips.task has workload "
                                   "fingerprint ")


class _Exits1(ApptainerRuntime):
    """Runs each container's command as a program that exits 1."""

    def argv(self, spec, *, env_file, seal_python=None):
        command = ("python", "-c", "import sys; sys.exit(1)")
        return super().argv(dataclasses.replace(spec, command=command), env_file=env_file, seal_python=seal_python)


class _WritesWithoutReport(ApptainerRuntime):
    """Runs each container's command as a program that writes the capture and reports nothing."""

    def argv(self, spec, *, env_file, seal_python=None):
        name = roles.capture_file(FINGERPRINT, "vllm", 3)
        script = ("import pathlib, sys; out, report = map(pathlib.Path, sys.argv[1:]); "
                  f"(out / {name!r}).parent.mkdir(parents=True); (out / {name!r}).write_text('x'); "
                  "report.write_text('{}')")
        command = ("python", "-c", script, "/hls/out", "/hls/outputs.json")
        return super().argv(dataclasses.replace(spec, command=command), env_file=env_file, seal_python=seal_python)


class _ExitsCleanly(ApptainerRuntime):
    """Runs each container's command as a program that writes nothing and exits 0."""

    def argv(self, spec, *, env_file, seal_python=None):
        command = ("python", "-c", "pass")
        return super().argv(dataclasses.replace(spec, command=command), env_file=env_file, seal_python=seal_python)


def test_a_capture_that_failed_otherwise_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    ctx.launcher.runtime = _Exits1()
    fault = _fault(CaptureStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)


def test_a_capture_role_that_does_not_report_its_capture_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    ctx.launcher.runtime = _WritesWithoutReport()
    fault = _fault(CaptureStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == f"role capture reported no digest of output {roles.capture_file(FINGERPRINT, 'vllm', 3)}"


def test_a_capture_role_that_exits_0_without_writing_its_capture_is_retried(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    ctx.launcher.runtime = _ExitsCleanly()
    fault = _fault(CaptureStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == f"the capture role exited 0 without writing {roles.capture_file(FINGERPRINT, 'vllm', 3)}"


@pytest.mark.parametrize(("ending", "exit_code", "oom_kills", "retryable"), [
    ("exited", 3, 0, False), ("exited", 3, 1, True), ("memory_limit", 137, 1, True), ("time_limit", 3, 0, True),
    ("exited", 1, 0, True)],
    ids=["refusal", "refusal-status-after-an-oom", "memory-limit", "refusal-status-after-the-deadline",
         "another-status"])
def test_only_a_clean_exit_3_of_a_role_is_a_refusal(tmp_path, checkout, container, ending, exit_code, oom_kills,
                                                     retryable):
    # A calibration has no contestant, so every ending of a role is infra; only status 3 from a role
    # that ended on its own with no OOM kill is its refusal, which another attempt would repeat.
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    (ctx.attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ctx.attempt_dir / "logs" / "capture.err").write_text(f"{roles.REFUSED_PREFIX}InputMismatch: stale\n")
    with pytest.raises(StageFault) as e:
        stages._Capture(ctx).raise_fault(RoleOutcome("capture", "harness", exit_code, ending, oom_kills))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", retryable)
    assert (e.value.fault.reason == "calibration capture refused: InputMismatch: stale") is not retryable


@pytest.mark.parametrize("line", [
    "EngineDied: vllm-3 (VLLMEngine) died after it loaded: process 7 (EngineCore_DP0) was killed by signal 9 (SIGKILL)",
    "EngineReleaseTimeout: vllm-3's stop did not return within 300 s"], ids=["died", "release-hung"])
def test_a_role_that_gave_up_on_its_engine_is_retried_with_its_reason(tmp_path, checkout, container, line):
    # An engine that died or would not let go is the harness's: another attempt starts it afresh,
    # and the record says why, from the role's own line.
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture())
    (ctx.attempt_dir / "logs").mkdir(parents=True, exist_ok=True)
    (ctx.attempt_dir / "logs" / "capture.err").write_text(f"stacks\n{roles.ENGINE_FAULT_PREFIX}{line}\n")
    with pytest.raises(StageFault) as e:
        stages._Capture(ctx).raise_fault(RoleOutcome("capture", "harness", roles.ENGINE_FAULT_EXIT, "exited", 0))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert e.value.fault.reason == f"calibration capture: {line}"


def test_a_capture_whose_member_does_not_load_within_its_bound_is_refused(tmp_path, checkout, container):
    # A build that deadlocks would hold the stage to its whole time: past the plan's bound for a
    # seeded load the role ends at once, and a retry on the node would start the same build.
    campaign = CAMPAIGN.model_copy(update={"load_timeout_sec": 1.0})
    ctx = _ctx(tmp_path, checkout, "capture_vllm_3", _capture(campaign=campaign))
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    entry = _engine_cache(ctx, key, {**MEMBER_BUILT, engines.HANG: b""})
    _owner(ctx, "warm_vllm", EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry), published=True))
    started = time.monotonic()
    fault = _fault(CaptureStage, ctx).fault
    assert time.monotonic() - started < 60
    # Retried: a load slowed by what else shares the node may pass at another attempt, and one that
    # hangs every time ends the instance void once the retries are spent.
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == "calibration capture: EngineLoadTimeout: vllm-3 did not load within 1 s"


# ---- warm ------------------------------------------------------------------------

def test_a_warm_stage_builds_its_members_engine_cache_cold_and_publishes_it(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "warm_vllm", _warm())
    order = CallOrder(ctx)
    outputs = WarmStage().run(ctx)
    assert outputs == {"engine_cache": ctx.output_key(ENGINE_CACHE_FILE)}
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    entry = ctx.cache.lookup(ENGINE_CACHE_KIND, key)
    assert {f.path for f in entry.manifest.files} == set(MEMBER_BUILT)
    assert entry.manifest.source == {"stage": "warm_vllm", "instance": ctx.plan.instance}
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry),
                                                      published=True)
    # On the member's measured image, with the reference's data: the weights and the full tool
    # bundle, no rows.
    assert order.staged() == ["node_image", "materialize", "materialize", "seed"]
    sha = _sif_of(ctx.cache.lookup("images", IMAGE_KEYS["evaluate-task-member-vllm"]))
    assert ctx.image_digests == {"warm": sha}
    seen = fake_container.records(container)["warm"]
    # Its load is given the stage's time less its copies and its margin.
    assert seen["command"][3:] == ["warm", "--task-ref", "/hls/task-ref.json", "--campaign", "/hls/campaign.json",
                                   "--engine", "vllm", "--load-sec", "400", "--out", "/hls/out", "--outputs-out",
                                   "/hls/outputs.json"]
    assert not any("/data/tenants" in b for b in seen["binds"])
    assert seen["env"]["FLASHINFER_WORKSPACE_BASE"].endswith("/engine-cache")
    # Sealed as the member's captures are, so nothing it fetched lands in the cache they replay.
    spec = _spec(ctx, "warm")
    assert spec.shape == SHAPE and spec.argv[:2] == ("bash", "-c")
    assert SealPolicy(network=AIR_GAP).model_dump_json() in spec.argv
    assert len(fake_pasta.calls(tmp_path / "pasta.log")) == 1 and seen["env"]["NO_PROXY"] == "*"
    assert [s.name for s in ctx.launcher.launched][-2:] == ["warm", "fill-engine-cache"]
    assert len(ctx.launcher.launched) == WarmStage.max_roles(ctx.params) == 5


def test_a_warm_stage_whose_members_tools_reach_the_network_runs_open(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "warm_vllm", _warm(campaign=CAMPAIGN.model_copy(update={"member_internet": True})))
    WarmStage().run(ctx)
    spec = _spec(ctx, "warm")
    assert spec.argv[:2] != ("bash", "-c") and SealPolicy(network=AIR_GAP).model_dump_json() not in spec.argv
    assert fake_pasta.calls(tmp_path / "pasta.log") == []


def test_a_warm_stage_given_more_time_gives_its_load_as_much_more(tmp_path, checkout, container):
    decl = _warm()
    ctx = _ctx(tmp_path, checkout, "warm_vllm", dataclasses.replace(decl, shape=decl.shape.model_copy(
        update={"time_sec": 900})))
    WarmStage().run(ctx)
    command = fake_container.records(container)["warm"]["command"]
    assert command[command.index("--load-sec") + 1] == "700"


def test_on_docker_the_calibration_key_is_the_digest_of_the_image_file_dockers_runtime_keeps(tmp_path, checkout,
                                                                                               container):
    """A Docker images entry holds ``image.oci.tar``, not ``image.sif``: the warm stage keys its
    member's engine cache by that file's sha256, finds the entry published under it and starts
    nothing, as on Slurm, where the key is the ``image.sif``'s."""
    ctx = _ctx(tmp_path, checkout, "warm_vllm", _warm(), docker=True)
    image = ctx.cache.lookup("images", IMAGE_KEYS["evaluate-task-member-vllm"])
    assert {f.path for f in image.manifest.files} == {DockerImages.image_file}
    key = engine_cache_key(_image_sha(image, DockerImages.image_file), "B200", CAMPAIGN.member_engine_configs["vllm"],
                           seal=member_seal(CAMPAIGN))
    entry = _engine_cache(ctx, key, MEMBER_BUILT)
    outputs = WarmStage().run(ctx)
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry))
    assert ctx.launcher.launched == []


def test_a_warm_stage_that_finds_its_members_engine_cache_starts_nothing(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "warm_vllm", _warm())
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    entry = _engine_cache(ctx, key, MEMBER_BUILT)
    outputs = WarmStage().run(ctx)
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry))
    # No extraction, no copy, no role: the stage cost nothing but its look at the store.
    assert ctx.launcher.launched == [] and ctx.image_digests == {}


def test_a_warm_stage_another_stage_beat_to_publishing_leaves_the_winners_entry(tmp_path, checkout, container,
                                                                              monkeypatch):
    # Another calibration's warm stage of the engine publishes first: this one built its own cache
    # cold, and the entry it leaves, which its captures are seeded from, is the winner's, whole.
    ctx = _ctx(tmp_path, checkout, "warm_vllm", _warm())
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    start = stages._ContainerAttempt.start
    won: list[CacheEntry] = []

    def meanwhile(self, *args, **kwargs):
        start(self, *args, **kwargs)
        won.append(_engine_cache(ctx, key, {**MEMBER_BUILT, "winner": b"x"}))

    monkeypatch.setattr(stages._ContainerAttempt, "start", meanwhile)
    outputs = WarmStage().run(ctx)
    [entry] = won
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry))
    assert entry_sha256(ctx.cache.lookup(ENGINE_CACHE_KIND, key)) == entry_sha256(entry)


def test_a_warm_stage_whose_publish_fails_fails(tmp_path, checkout, container, monkeypatch):
    # The entry is all a warm stage is for.
    ctx = _ctx(tmp_path, checkout, "warm_vllm", _warm())

    def refused(*args, **kwargs):
        raise StageFault(infra_fault("the tree for cache entry was refused: link is a symlink", retryable=False))

    monkeypatch.setattr(ctx, "publish", refused)
    fault = _fault(WarmStage, ctx).fault
    assert (fault.fault_class, fault.retryable, fault.reason) == (
        "infra", False, "the tree for cache entry was refused: link is a symlink")


def test_a_warm_stage_refuses_a_corrupt_engine_cache(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "warm_vllm", _warm())
    key = _cache_key(ctx, "evaluate-task-member-vllm", CAMPAIGN.member_engine_configs["vllm"])
    (_engine_cache(ctx, key, MEMBER_BUILT).path / "ENTRY.json").write_text("{")
    fault = _fault(WarmStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith(f"cache entry {ENGINE_CACHE_KIND}/{key} is corrupt: ")
    assert ctx.launcher.launched == []


# ---- score and instrument -------------------------------------------------------

def test_a_score_stage_mounts_each_capture_read_only_and_reports_every_servings_replies(tmp_path, checkout,
                                                                                       container, rows):
    ctx = _ctx(tmp_path, checkout, "score_1", _score(), files=_captures(tmp_path, rows, (0, 2)))
    key = _cache_key(ctx, "evaluate-task-reference", CAMPAIGN.reference_engine_config)
    built = {engines.KERNEL.format("reference-bfloat16"): b"k", engines.COMPILES: b"reference-bfloat16\n"}
    entry = _engine_cache(ctx, key, built)
    _owner(ctx, "instrument", EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry), published=True))
    outputs = ScoreShareStage().run(ctx)
    assert outputs == {"scored": ctx.output_key(roles.SCORED_FILE),
                       **{f"replies_vllm_{d}": ctx.output_key(roles.replies_file(FINGERPRINT, "vllm", d))
                          for d in (0, 2)}, "engine_cache": ctx.output_key(ENGINE_CACHE_FILE)}
    # The pinned reference was seeded with the entry its owner, the instrument, left.
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=entry_sha256(entry))
    assert (ctx.role_dir("score") / "engine-cache" / engines.COMPILES).read_text() == "reference-bfloat16\n"
    seen = fake_container.records(container)["score"]
    captures = {d: ctx.store.path(ctx.inputs[ctx.plan.layout.manifest(capture_stage("vllm", d))].outputs[
        "capture"].key) for d in (0, 2)}
    assert set(seen["binds"]) >= {f"{captures[d]}:/hls/captures/vllm-{d}.capture.jsonl.gz:ro" for d in (0, 2)}
    shas = {d: ctx.inputs[ctx.plan.layout.manifest(capture_stage("vllm", d))].outputs["capture"].sha256
            for d in (0, 2)}
    assert seen["command"][4:] == [
        "--task-ref", "/hls/task-ref.json", "--campaign", "/hls/campaign.json", "--rows", "/data", "--gpus", "1",
        "--share", "1", "--capture", "vllm", "0", shas[0], "/hls/captures/vllm-0.capture.jsonl.gz",
        "--capture", "vllm", "2", shas[2], "/hls/captures/vllm-2.capture.jsonl.gz", "--view", "/hls/view",
        "--out", "/hls/out", "--outputs-out", "/hls/outputs.json"]
    scored = roles.ScoredServings.model_validate_json(ctx.store.read_bytes(outputs["scored"]))
    assert [(s.engine, s.draw) for s in scored.servings] == [("vllm", 0), ("vllm", 2)]
    # On the reference's measured image, where the pinned reference loads.
    assert ctx.image_digests == {"score": _sif_of(ctx.cache.lookup("images", IMAGE_KEYS["evaluate-task-reference"]))}
    # The extraction, a copy of the weights, the test rows and the full tool bundle, the copy of
    # the engine cache and the score role; the captures are mounted where their stages published
    # them.
    assert len(ctx.launcher.launched) == ScoreShareStage.max_roles(ctx.params) == 6


def test_the_instrument_stage_holds_the_anchors_gpu_too(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "instrument", _instrument())
    outputs = InstrumentStage().run(ctx)
    assert outputs == {"instrument": ctx.output_key(roles.INSTRUMENT_FILE),
                       "engine_cache": ctx.output_key(ENGINE_CACHE_FILE)}
    # The reference's owner publishes its engine cache, the anchor's kernel beside the pinned
    # reference's, for every score stage after it.
    key = _cache_key(ctx, "evaluate-task-reference", CAMPAIGN.reference_engine_config)
    entry = ctx.cache.lookup(ENGINE_CACHE_KIND, key)
    assert (entry.path / engines.COMPILES).read_text() == "reference-bfloat16\nreference-float32\n"
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=None, entry=entry_sha256(entry),
                                                      published=True)
    assert _spec(ctx, "instrument").shape == CAMPAIGN.instrument_shape() and CAMPAIGN.instrument_shape().gpus == 2
    seen = fake_container.records(container)["instrument"]
    assert "--gpus" in seen["command"] and seen["command"][seen["command"].index("--gpus") + 1] == "2"
    # The reference mounts no rows.
    assert not any("/data/tenants" in b for b in seen["binds"])
    record = roles.InstrumentRecord.model_validate_json(ctx.store.read_bytes(outputs["instrument"]))
    assert record.preflight is not None and record.preflight.passed
    assert ctx.image_digests == {
        "instrument": _sif_of(ctx.cache.lookup("images", IMAGE_KEYS["evaluate-task-reference"]))}
    # The extraction, a copy of the weights and of the full tool bundle, the instrument role and
    # the copy that publishes its engine cache.
    assert len(ctx.launcher.launched) == InstrumentStage.max_roles(ctx.params) == 5


def test_an_instrument_that_finds_the_references_engine_cache_is_seeded_from_it(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "instrument", _instrument())
    key = _cache_key(ctx, "evaluate-task-reference", CAMPAIGN.reference_engine_config)
    entry = _engine_cache(ctx, key, {engines.COMPILES: b"reference-bfloat16\n"})
    outputs = InstrumentStage().run(ctx)
    assert _record(ctx, outputs) == EngineCacheRecord(key=key, seeded_from=entry_sha256(entry),
                                                      entry=entry_sha256(entry))
    assert not [s.name for s in ctx.launcher.launched if s.name.startswith("fill-")]


def test_an_instruments_failed_publish_is_recorded_and_fails_nothing(tmp_path, checkout, container, monkeypatch):
    # The reference builds next to no kernels, so its score stages start cold at little cost.
    ctx = _ctx(tmp_path, checkout, "instrument", _instrument())
    key = _cache_key(ctx, "evaluate-task-reference", CAMPAIGN.reference_engine_config)

    def refused(*args, **kwargs):
        raise StageFault(infra_fault("the tree was refused: link is a symlink", retryable=False))

    monkeypatch.setattr(ctx, "publish", refused)
    outputs = InstrumentStage().run(ctx)
    assert outputs["instrument"] == ctx.output_key(roles.INSTRUMENT_FILE)
    assert _record(ctx, outputs) == EngineCacheRecord(
        key=key, seeded_from=None, unpublished="the publish failed: the tree was refused: link is a symlink")


def test_a_score_stage_is_seeded_from_its_instruments_build_and_no_other(tmp_path, checkout, container, rows):
    ctx = _ctx(tmp_path, checkout, "score_1", _score(), files=_captures(tmp_path, rows, (0, 2)))
    key = _cache_key(ctx, "evaluate-task-reference", CAMPAIGN.reference_engine_config)
    other = _engine_cache(ctx, key, {engines.COMPILES: b"another build"})
    _owner(ctx, "instrument", EngineCacheRecord(key=key, seeded_from=None, entry="f" * 64, published=True))
    fault = _fault(ScoreShareStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == (f"cache entry {ENGINE_CACHE_KIND}/{key} has ENTRY.json digest {entry_sha256(other)}, not "
                            f"the {'f' * 64} its producer recorded")
    assert "score" not in [s.name for s in ctx.launcher.launched]


# ---- aggregate -----------------------------------------------------------------

def _scored_files(tmp_path: Path, draws: range) -> Files:
    """Score stage ``score_1``'s published outputs and the instrument's, for vllm's servings *draws*
    of 10,000 tokens each."""
    made = CaptureStore(tmp_path / "scored", FINGERPRINT)
    servings, replies = [], {}
    for d in draws:
        made.write_scored(score_draw("vllm", d, serving_seed_of(fixture.flips_task().seed, d), "0.0.0", [_tf_item(10_000)],
                                     [[0.1] * 20 + [0.0] * 9_980]))
        servings.append(roles.ScoredServing(engine="vllm", draw=d, n_tokens=10_000, n_replies=1,
                                            serving_version="0.0.0"))
        replies[f"replies_vllm_{d}"] = (roles.replies_file(FINGERPRINT, "vllm", d),
                                        made.replies_path("vllm", d).read_bytes())
    record = roles.InstrumentRecord(stack=StackInfo(**engines.STACK), preflight=PreflightResult(
        passed=True, flip_rate_vs_fp32=0.0, max_gap_vs_fp32=0.0, n_tokens=24))
    return {"instrument": {"instrument": (roles.INSTRUMENT_FILE, record.model_dump_json().encode())},
            "score_1": {"scored": (roles.SCORED_FILE, roles.ScoredServings(servings=servings).model_dump_json().encode()),
                        **replies},
            **{capture_stage("vllm", d): {"engine_cache": (ENGINE_CACHE_FILE,
                                                           MEMBER_CACHES[d].model_dump_json().encode())}
               for d in draws}}


#: Each capture's record of the engine cache its member ran with: serving 0 started cold, the
#: rest were seeded from the entry their owner left.
MEMBER_CACHES = {d: EngineCacheRecord(key="B200-0123456789abcdef-fedcba9876543210",
                                      seeded_from=None if d == 0 else "e" * 64)
                 for d in range(9)}


def test_the_aggregate_runs_on_the_plans_interpreter_over_the_published_records(tmp_path, checkout, container):
    ctx = _ctx(tmp_path, checkout, "aggregate", _score(tuple(range(8))), _aggregate(),
               files=_scored_files(tmp_path, range(8)))
    outputs = AggregateStage().run(ctx)
    assert outputs == {"calibration": ctx.output_key(CALIBRATION_FILE),
                       "aggregate": ctx.output_key(roles.AGGREGATE_FILE)}
    spec = _spec(ctx, "aggregate")
    assert spec.owner == "harness" and spec.argv[:4] == (str(sys.executable), "-m", "tests.calibration.fake_roles",
                                                         "aggregate")
    replies = ctx.inputs[ctx.plan.layout.manifest("score_1")].outputs["replies_vllm_0"]
    i = spec.argv.index("--serving")
    assert spec.argv[i:i + 6] == ("--serving", "vllm", "0", "10000", replies.sha256, str(ctx.store.path(replies.key)))
    summary = roles.AggregateRecord.model_validate_json(ctx.store.read_bytes(outputs["aggregate"]))
    assert (summary.servings, summary.tokens) == ({"vllm": list(range(8))}, {"vllm": 80_000})
    # The compile products each serving's member replayed, as its capture stage recorded it.
    assert summary.member_caches == {"vllm": {d: MEMBER_CACHES[d] for d in range(8)}}
    assert len(ctx.launcher.launched) == AggregateStage.max_roles(ctx.params) == 1


def test_an_aggregate_that_cannot_read_a_captures_engine_cache_record_is_refused_without_a_role(
        tmp_path, checkout, container):
    files = dict(_scored_files(tmp_path, range(8)))
    del files[capture_stage("vllm", 5)]
    ctx = _ctx(tmp_path, checkout, "aggregate", _score(tuple(range(8))), _aggregate(), files=files)
    fault = _fault(AggregateStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert "capture_vllm_5" in fault.reason and "not among its inputs" in fault.reason
    assert not ctx.launcher.launched


def test_an_aggregate_over_records_of_other_servings_is_refused_without_a_role(tmp_path, checkout, container):
    files = _scored_files(tmp_path, range(8))
    ctx = _ctx(tmp_path, checkout, "aggregate", _score(tuple(range(9))), _aggregate(), files=files)
    fault = _fault(AggregateStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason.startswith("score_1 scored [('vllm', 0)")
    assert not ctx.launcher.launched


def test_an_aggregate_that_leaves_out_a_score_stage_of_the_plan_is_refused_without_a_role(tmp_path, checkout,
                                                                                       container):
    # The candidate is assembled over every serving the instance scored, or not at all.
    second = _decl("score_2", "calibrate_score", SHAPE, KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB,
                   {"share": 2, "servings": [["vllm", 8]]})
    ctx = _ctx(tmp_path, checkout, "aggregate", _score(tuple(range(8))), second, _aggregate(),
               files=_scored_files(tmp_path, range(8)))
    fault = _fault(AggregateStage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == "the aggregate reads score stages ['score_1'], not every score stage of its plan, " \
                           "['score_1', 'score_2']"
    assert not ctx.launcher.launched


# ---- plan time -------------------------------------------------------------------

def test_each_stage_is_held_to_its_roles_shape(tmp_path, checkout):
    roomy = _capture()
    with pytest.raises(PlanError, match="calibrate_capture stage runs one role"):
        _plan(tmp_path, checkout, dataclasses.replace(roomy, shape=roomy.shape.model_copy(update={"disk_gib": 1})))
    one_gpu = _instrument()
    with pytest.raises(PlanError, match="calibrate_instrument stage runs one role"):
        _plan(tmp_path, checkout, dataclasses.replace(one_gpu, shape=stage_shape(
            SHAPE, runner_memory_gib=RUNNER_GIB, disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + DATA_GIB, time_sec=600)))
    # Each copying stage's disk holds its copies of its role's data, as its plan recorded them.
    for stage in (_instrument(), _warm(), _capture(), _score()):
        short = stage.shape.model_copy(update={"disk_gib": stage.shape.disk_gib - DATA_GIB})
        with pytest.raises(PlanError, match=f"{DATA_GIB} GiB for the copies of its data"):
            _plan(tmp_path, checkout, dataclasses.replace(stage, shape=short))
    with pytest.raises(PlanError, match="loads no engine"):
        _plan(tmp_path, checkout, dataclasses.replace(_aggregate(), shape=SHAPE))
    # A site may give a stage another time, which its role then runs for.
    _plan(tmp_path, checkout, dataclasses.replace(roomy, shape=roomy.shape.model_copy(update={"time_sec": 900})))


@pytest.mark.parametrize(("stage", "decl", "role", "copied"), [
    pytest.param(CaptureStage, lambda: _capture(data_copies={}), "capture", (0, 1, 2), id="capture"),
    pytest.param(ScoreShareStage, lambda: _score(data_copies={}), "score", (0, 1, 2), id="score"),
    pytest.param(InstrumentStage, lambda: _instrument(data_copies={}), "instrument", (0, 2), id="instrument"),
    pytest.param(WarmStage, lambda: _warm(data_copies={}), "warm", (0, 2), id="warm"),
])
def test_copies_the_plan_made_no_room_for_are_refused_before_the_role(tmp_path, checkout, container, rows, stage,
                                                                       decl, role, copied):
    """The node's start check passed the allocation's disk, so the stage's copies of its role's
    data must be those the plan declared (``data_copies``): a copy it did not declare is infra
    that is not retried, refused before any copy and before the role, naming each such copy (the
    reference role's, the weights and the full bundle; the calibration role's, the rows too)."""
    planned = decl()
    files = _captures(tmp_path, rows, (0, 2)) if stage is ScoreShareStage else None
    ctx = _ctx(tmp_path, checkout, planned.name, planned, files=files)
    fault = _fault(stage, ctx).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    entries = [list(DATA_COPIES)[i] for i in copied]
    assert fault.reason == (f"the {role} role's data cannot be mounted: the plan made no room on the node for a copy "
                            f"of {', '.join(entries)}: plan the instance again")
    assert not [s for s in ctx.launcher.launched if s.name == role or s.name.startswith("copy-")]


def test_a_captures_seed_is_its_servings():
    params = dict(task=TASK, campaign=CAMPAIGN, engine="vllm", draw=3, warm_stage="warm_vllm",
                  data_copies=DATA_COPIES)
    CaptureParams.model_validate({**params, "seed": serving_seed_of(fixture.flips_task().seed, 3)})
    with pytest.raises(ValueError, match="serving 3 is clocked with seed"):
        CaptureParams.model_validate({**params, "seed": 7})
    with pytest.raises(ValueError, match="not one of the campaign's pool members"):
        CaptureParams.model_validate({**params, "engine": "sglang", "seed": serving_seed_of(fixture.flips_task().seed, 3)})


@pytest.mark.parametrize(("servings", "refusal"), [
    pytest.param([["vllm", 0], ["vllm", 0]], "servings [('vllm', 0), ('vllm', 0)] name a serving twice", id="twice"),
    pytest.param([["vllm", 0], ["sglang", 1]], "servings of ['sglang'], which are not the campaign's pool members",
                 id="stranger"),
    pytest.param([["vllm", -1]], "servings [('vllm', -1)] hold a negative index", id="negative"),
])
def test_a_share_holds_each_serving_of_its_campaign_once(servings, refusal):
    with pytest.raises(ValueError, match=re.escape(refusal)):
        ScoreShareParams.model_validate({"task": TASK, "campaign": CAMPAIGN, "share": 1, "servings": servings,
                                         "data_copies": DATA_COPIES})


def test_an_engine_caches_key_is_its_image_its_gpu_kind_its_config_and_its_build():
    key = engine_cache_key("a" * 64, "B200", "config", seal=None)
    build = json.dumps({"env": engine_build_env("B200"), "seal": None}, sort_keys=True)
    assert key == f"B200-{'a' * 16}-{sha256_bytes(b'config')[:16]}-{sha256_bytes(build.encode())[:16]}"
    sealed = member_seal(CAMPAIGN)
    assert sealed is not None
    assert len({key, engine_cache_key("b" * 64, "B200", "config", seal=None),
                engine_cache_key("a" * 64, "H200", "config", seal=None),
                engine_cache_key("a" * 64, "B200", "other", seal=None),
                engine_cache_key("a" * 64, "B200", "config", seal=sealed)}) == 5


@pytest.mark.parametrize("change", [
    pytest.param({"TVM_FFI_CACHE_DIR": "tvm-ffi"}, id="a-variable-added"),
    pytest.param({"TRITON_CACHE_DIR": "triton-elsewhere"}, id="a-target-moved"),
])
def test_a_change_to_the_cache_variables_changes_the_key(monkeypatch, change):
    # An entry built under other variables holds other products: the warm stage of a cache env that
    # newly sent TVM-FFI's builds into the cache would find the entry built without it, and every
    # seeded capture would compile them again.
    variables = {name: path for name, path in ENGINE_CACHE_VARIABLES.items() if name != "TVM_FFI_CACHE_DIR"}
    monkeypatch.setattr(engine_cache, "ENGINE_CACHE_VARIABLES", variables)
    before = engine_cache_key("a" * 64, "B200", "config", seal=None)
    monkeypatch.setattr(engine_cache, "ENGINE_CACHE_VARIABLES", {**variables, **change})
    assert engine_cache_key("a" * 64, "B200", "config", seal=None) != before


def test_a_gpu_kinds_compute_capability_is_part_of_the_build(monkeypatch):
    before = engine_cache_key("a" * 64, "B200", "config", seal=None)
    monkeypatch.setitem(engine_cache.COMPUTE_CAPABILITY, "B200", "10.0a")
    assert engine_cache_key("a" * 64, "B200", "config", seal=None) != before


def test_a_warm_stage_builds_a_member_of_its_campaign():
    params = {"task": TASK, "campaign": CAMPAIGN, "engine": "vllm", "staging_sec": 100, "margin_sec": 100,
              "data_copies": DATA_COPIES}
    WarmParams.model_validate(params)
    with pytest.raises(ValueError, match=re.escape("engine sglang is not one of the campaign's pool members ['vllm']")):
        WarmParams.model_validate({**params, "engine": "sglang"})


def test_a_warm_stage_whose_time_leaves_its_load_none_is_refused_at_plan_time(tmp_path, checkout):
    decl = _warm()
    with pytest.raises(PlanError, match="leave its member's load no time after its 100s of copies and its 100s margin"):
        _plan(tmp_path, checkout, dataclasses.replace(decl, shape=decl.shape.model_copy(update={"time_sec": 200})))
