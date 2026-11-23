"""``hls calibrate``'s planning: the instrument check first, in
the allocation of the task's data and measured images; each engine's first servings after it,
interleaved and dealt round robin over --fanout allocations, each serving a capture stage and each
allocation's captures scored by one score stage; the plan grown by the calibration grower. The
refusals, before anything is submitted; and the command."""

from __future__ import annotations

import dataclasses
import importlib
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from hls import cli
from hls.calibrate import (DEFAULT_DELTA, DEFAULT_EPS, DEFAULT_MU_PER_MTOK, DEFAULT_REPORT_SERVINGS,
                           MIN_SERVINGS, engine_cache_config, member_engines, pinned_reference,
                           serving_seed_of)
from hls.pipeline import calibration, calibration_grower, commands
from hls.pipeline.grown import grown_stages
from hls.pipeline.pipelines import (DRIVE_MARGIN_SEC, EVALUATE_IMAGES_CPUS, KNOWN_STAGE_NAMES, OPEN_DRIVE_SEC,
                                    PROVISION_ROLE_GIB, READINESS_TIMEOUT_SEC, REFERENCE_SHAPE, REQUEST_TIMEOUT_SEC,
                                    STAGING_SEC, images_shape, reference_shape)
from hls.pipeline.plan import Plan
from hls.provision.mounts import copied_entries, copies_gib
from hls.pipeline.planner import PlanError
from hls.spec.resources import ResourceShape
from hls.stages.calibration import (SCRATCH_GIB, CaptureParams, InstrumentParams, ScoreShareParams, WarmParams,
                                    warm_load_sec)
from hls.stages.engine_cache import COLD_BUILD_SEC, COLD_LOAD_SEC, ENGINE_CACHE_GIB, engine_cache_key
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages.images import IMAGE_FILE, KEPT_DISK_GIB, image_keys
from hls.store.cache import ENTRY_FILE, ContentCache
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.verify.calibration import DEFAULT_EOT_EXPECTED_MAX, exposure_target, workload_fingerprint
from hls.substrate.apptainer.images import ApptainerImages
from tests.calibration import fixture
from tests.provision.cached import provision_task
from tests.provision.warmer_config import write_config
from tests.store.entries import hold_measured, populate
from tests.unit.test_tool_assets import FakeAssets

REPO = Path(__file__).resolve().parents[2]

#: The memory the test site leaves to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4


def limited_config(tmp_path: Path, limits: str) -> Path:
    """``write_config``'s site under QOS ``normal``, with *limits* (a TOML inline table of QOS to
    hours) as the most hours a job may ask under each, as ``{normal = 24, long = 168}``."""
    config = write_config(tmp_path)
    config.write_text(config.read_text().replace("[site]\n", f'[site]\nqos = "normal"\nqos_by_hours = {limits}\n', 1))
    return config


def _plan(tmp_path: Path, *, module: str = "cal_flips.task", engines=("vllm", "sglang"), fanout: int = 4,
          servings_per_engine: int | None = None, sessions_per_draw: int | None = None, eps: float = DEFAULT_EPS,
          delta: float = DEFAULT_DELTA, max_servings: int = calibration.DEFAULT_MAX_SERVINGS_PER_ENGINE,
          echo=None, config: Path | None = None, provisioned: bool = True,
          overrides: dict[str, str] | None = None) -> Plan:
    """``hls calibrate``'s first plan, the task's provisioning in the cache first unless
    *provisioned* is False."""
    if provisioned:
        provision_task(tmp_path / "store", module)
    return calibration.plan_calibrate(
        config=config or write_config(tmp_path), overrides=overrides or {}, task_module=module, engines=list(engines),
        fanout=fanout,
        servings_per_engine=servings_per_engine, sessions_per_draw=sessions_per_draw, eps=eps,
        mu_per_mtok=DEFAULT_MU_PER_MTOK, delta=delta, eot_expected_max=DEFAULT_EOT_EXPECTED_MAX,
        max_servings_per_engine=max_servings, job_name="hls-calibrate", echo=echo if echo is not None else [].append)


def test_a_calibrations_images_stage_asks_the_cache_about_its_own_images_and_holds_disk_for_what_it_fills(
        tmp_path, login, in_catalog, monkeypatch, warmed_servers):
    """Cold, it builds its images, holding disk for them. Once the cache holds them it holds no
    disk, and the planner asked the cache about the calibration's images alone."""
    warmed_servers()
    (tmp_path / "cold").mkdir()
    (tmp_path / "warm").mkdir()
    cold = _plan(tmp_path / "cold").stage("images")
    assert cold.params["found"] == [] and cold.shape.disk_gib > 0
    keys = image_keys(order(catalog.task_images(fixture.TASK_ID).decls), ApptainerImages.recipe_key)
    cache = ContentCache(PosixStore(tmp_path / "warm" / "store"))
    for key in keys.values():
        populate(cache, "images", key, lambda d: (d / IMAGE_FILE).write_bytes(b"image"), holder=Holder.here("test"))
    asked: list[str] = []
    lookup = ContentCache.lookup

    def recording(self, kind, key):
        if kind == "images":
            asked.append(key)
        return lookup(self, kind, key)

    monkeypatch.setattr(ContentCache, "lookup", recording)
    warm = _plan(tmp_path / "warm").stage("images")
    own = sorted(keys[d["name"]] for d in warm.params["images"])
    # Only the images stage asks, once for each of its own images.
    assert sorted(asked) == own == warm.params["found"] and len(own) < len(keys)
    assert warm.shape == images_shape(0, cpus=EVALUATE_IMAGES_CPUS) and warm.shape.disk_gib == 0


def test_the_stages_that_copy_the_tasks_data_hold_the_copies_in_their_disk(tmp_path, login, in_catalog,
                                                                            monkeypatch):
    """The instrument and each warm stage copy the reference role's data (``hls.provision.mounts``),
    each capture and score stage the calibration role's: each declares the disk those copies take,
    by the provision record the plan read from the cache (``copied_entries``), beside the disk of
    its image and its engine cache, and declares those copies for the stage to hold its own to
    (``data_copies``)."""
    provisioned = provision_task(tmp_path / "store", "cal_flips.task")
    sized = {"reference": {"weights/w": 3 * 2**30}, "calibration": {"weights/w": 3 * 2**30, "rows/r": 2 * 2**30}}
    asked: list = []

    def copies(roles, record):
        asked.append(record)
        [role] = roles
        return sized[role]

    monkeypatch.setattr(calibration, "copied_entries", copies)
    plan = _plan(tmp_path)
    assert asked and all(record == provisioned for record in asked)
    served = KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB
    for name, role, room in (("instrument", "reference", KEPT_DISK_GIB + ENGINE_CACHE_GIB),
                             ("warm_vllm", "reference", KEPT_DISK_GIB + ENGINE_CACHE_GIB),
                             ("warm_sglang", "reference", KEPT_DISK_GIB + ENGINE_CACHE_GIB),
                             ("capture_vllm_0", "calibration", served), ("score_1", "calibration", served)):
        stage = plan.stage(name)
        assert (stage.params["data_copies"], stage.shape.disk_gib) == (
            sized[role], room + copies_gib(sized[role])), name


def test_a_calibration_on_a_cold_cache_is_refused_naming_the_command_that_provisions_it(tmp_path, login,
                                                                                       in_catalog):
    """The plan sizes its stages' copies of the task's data from the provision record the cache
    holds, so a store without the task's provisioning refuses the plan before anything runs, though
    the plan provisions in its own allocation too. The refusal names the command to run first, with
    the task's overrides, which choose what is provisioned."""
    config = write_config(tmp_path)
    with pytest.raises(PlanError) as e:
        calibration.plan_calibrate(
            config=config, overrides={"task.seed": "7"}, task_module="cal_flips.task", engines=["vllm"], fanout=1,
            servings_per_engine=None, sessions_per_draw=None, eps=DEFAULT_EPS, mu_per_mtok=DEFAULT_MU_PER_MTOK,
            delta=DEFAULT_DELTA, eot_expected_max=DEFAULT_EOT_EXPECTED_MAX, max_servings_per_engine=24,
            job_name="hls-calibrate", echo=[].append)
    assert str(e.value).endswith(f"run hls provision cal_flips.task --config {config} --task.seed 7 first"), \
        str(e.value)


def test_a_calibration_whose_measured_image_entry_carries_no_mtimes_builds_it_again(
        tmp_path, login, in_catalog, warmed_servers):
    """An entry whose file records carry no times is not one the cache holds, so the plan fills it."""
    warmed_servers()
    store = tmp_path / "store"
    measured = hold_measured(store, fixture.TASK_ID)
    name, key = next(iter(measured.items()))
    mf = PosixStore(store).path(f"cache/images/{key}") / ENTRY_FILE
    manifest = json.loads(mf.read_text())
    mf.write_text(json.dumps({**manifest, "files": [{k: v for k, v in f.items() if k != "mtime_ns"}
                                                    for f in manifest["files"]]}))
    assert ContentCache(PosixStore(store)).lookup("images", key) is None
    assert key not in _plan(tmp_path).stage("images").params["found"]


def _shares(plan: Plan) -> list[list[tuple[str, int]]]:
    """Each score stage's servings, in share order."""
    return [[tuple(s) for s in ScoreShareParams.model_validate(st.params).servings]
            for st in plan.stages if st.kind == "calibrate_score"]


def test_a_flips_calibration_plans_eight_servings_per_engine_and_the_instrument_first(tmp_path, login, in_catalog):
    lines: list[str] = []
    plan = _plan(tmp_path, echo=lines.append)
    task = fixture.flips_task()
    # The copies of the reference role's data (the instrument's and the warm stages') and of the
    # calibration role's (each serving's), by the provision record the cache holds.
    provisioned = provision_task(tmp_path / "store", "cal_flips.task")
    reference, served = (copies_gib(copied_entries((role,), provisioned)) for role in ("reference", "calibration"))
    assert (plan.pipeline, plan.task, plan.grower) == ("calibration", fixture.TASK_ID, "calibration")
    assert plan.instance.startswith(f"calibrations/{fixture.TASK_ID}/") and plan.instance.count("/") == 2
    names = [s.name for s in plan.stages]
    assert names[:3] == ["provision", "images", "instrument"]
    # The first stages' fixed names are every pipeline's; the numbered ones are the grower's
    # declaration's, which holds nothing but what the grower adds.
    assert all(n in KNOWN_STAGE_NAMES or grown_stages("calibration").holds(n) for n in names)
    assert not any(grown_stages("calibration").holds(n) for n in ("provision", "images", "instrument"))
    captures = [CaptureParams.model_validate(s.params) for s in plan.stages if s.kind == "calibrate_capture"]
    for engine in ("vllm", "sglang"):
        mine = [c for c in captures if c.engine == engine]
        assert sorted(c.draw for c in mine) == list(range(MIN_SERVINGS))
        # Each serving is clocked from its own seed, derived from its index.
        assert all(c.seed == serving_seed_of(task.seed, c.draw) for c in mine)
    # The instrument check and each member's warm stage run in the allocation of the data and the
    # images; every serving's allocation starts only once that one has succeeded.
    first, *shares = plan.allocations
    assert (first.stages, first.depends_on) == (["provision", "images", "instrument", "warm_vllm", "warm_sglang"],
                                                None)
    assert len(shares) == 4 and all((a.depends_on, a.waits_for) == (0, []) for a in shares)
    assert [a.stages[-1] for a in shares] == ["score_1", "score_2", "score_3", "score_4"]
    assert sum(len(a.stages) - 1 for a in shares) == 2 * MIN_SERVINGS
    instrument = plan.stage("instrument")
    campaign = InstrumentParams.model_validate(instrument.params).campaign
    assert (campaign.kind, campaign.servings_per_engine, campaign.engines) == ("flips", None, ["vllm", "sglang"])
    assert (campaign.workload_fingerprint, campaign.task_seed) == (workload_fingerprint(task), task.seed)
    assert campaign.member_images == {"vllm": "evaluate-task-member-vllm", "sglang": "evaluate-task-member-sglang"}
    assert campaign.reference_image == "evaluate-task-reference"
    # Each engine's cache is keyed by what it is built with, beside its image and GPU kind.
    assert campaign.member_engine_configs == {e: engine_cache_config(member_engines(task)[e]())
                                              for e in ("vllm", "sglang")}
    assert campaign.reference_engine_config == engine_cache_config(pinned_reference(task))
    assert len({*campaign.member_engine_configs.values(), campaign.reference_engine_config}) == 3
    # The fp32 anchor is up beside the pinned reference: twice the reference role's GPUs, CPUs and
    # memory, and the tree of the image it extracts.
    ref = reference_shape(task, runner_memory_gib=RUNNER_GIB)
    assert instrument.shape == ResourceShape(gpus=2 * ref.gpus, gpu_kind="B200", cpus=2 * ref.cpus,
                                             memory_gib=2 * ref.memory_gib + RUNNER_GIB,
                                             disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + reference,
                                             time_sec=calibration.instrument_sec(campaign))
    assert instrument.after == ["provision", "images"]
    # Every engine loads within its bound, and each stage's time holds it after the fixed staging
    # allowance: a member's warm stage gives its cold build what its time leaves, every other stage
    # a seeded load the readiness budget.
    assert campaign.load_timeout_sec == READINESS_TIMEOUT_SEC
    assert COLD_LOAD_SEC >= COLD_BUILD_SEC
    assert calibration.instrument_sec(campaign) == (STAGING_SEC + 2 * READINESS_TIMEOUT_SEC
                                                    + calibration.PREFLIGHT_SEC)
    for engine in ("vllm", "sglang"):
        warm = plan.stage(f"warm_{engine}")
        assert (warm.kind, warm.after) == ("calibrate_warm", ["provision", "images"])
        params = WarmParams.model_validate(warm.params)
        assert params.data_copies == instrument.params["data_copies"]
        assert warm.shape == ResourceShape(gpus=1, gpu_kind="B200", cpus=8, memory_gib=176 + RUNNER_GIB,
                                           disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + reference,
                                           time_sec=calibration.warm_sec())
        assert (params.engine, params.staging_sec, params.margin_sec) == (engine, STAGING_SEC, DRIVE_MARGIN_SEC)
        assert warm_load_sec(params, warm.shape.time_sec) == COLD_LOAD_SEC
        assert calibration.warm_sec() == STAGING_SEC + COLD_LOAD_SEC + DRIVE_MARGIN_SEC
    # The reference's and each member's measured images; nothing is built.
    assert {d["name"] for d in plan.stage("images").params["images"]} == {
        "cuda", "evaluate-task-member-vllm", "evaluate-task-member-sglang", "evaluate-task-reference"}
    # The fixture task declares no provision memory: each provision role gets the default.
    assert plan.stage("provision").shape == ResourceShape(cpus=4, memory_gib=PROVISION_ROLE_GIB + RUNNER_GIB,
                                                          time_sec=2 * 3600)
    capture = plan.stage("capture_vllm_0")
    assert capture.shape == ResourceShape(gpus=1, gpu_kind="B200", cpus=8, memory_gib=176 + RUNNER_GIB,
                                          disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB + served,
                                          time_sec=calibration_grower.capture_sec(campaign))
    # Room for the largest engine cache measured with its build's temporary files, twice over, and
    # beside it for twice the largest serving's replies, deepresearch's 1.66 GB of logprobs.
    assert (ENGINE_CACHE_GIB, SCRATCH_GIB) == (8, 4)
    # Ten minutes for the preflight's probes, and half as much again as the slowest measured
    # score stage's 10,416 s over two servings.
    assert (calibration.PREFLIGHT_SEC, calibration.SCORE_PER_SERVING_SEC) == (600, 7812)
    assert calibration_grower.capture_sec(campaign) == (STAGING_SEC + READINESS_TIMEOUT_SEC + OPEN_DRIVE_SEC
                                                         + DRIVE_MARGIN_SEC)
    # Its member's warm stage owns the engine cache it is seeded from, and the instrument the
    # reference's, which every score stage is seeded from.
    assert capture.after == ["provision", "images", "warm_vllm"]
    shared = _shares(plan)[0]
    score = plan.stage("score_1")
    assert score.after == ["provision", "images", "instrument", *(f"capture_{e}_{d}" for e, d in shared)]
    assert score.shape.time_sec == calibration_grower.score_sec(campaign, len(shared))
    assert calibration_grower.score_sec(campaign, 4) == (STAGING_SEC + READINESS_TIMEOUT_SEC
                                                         + 4 * calibration.SCORE_PER_SERVING_SEC)
    target = exposure_target(DEFAULT_MU_PER_MTOK)
    assert lines[-1] == (f"calibrating the token-level gate on vllm, sglang: {MIN_SERVINGS} servings per engine "
                         f"first, in 4 allocations after the instrument's; each engine serves to {target:.0f} "
                         "tokens of exposure, at most 24 servings")


@pytest.mark.parametrize("module", ["tasks.multitenant_slo.task", "tasks.tab_completion.task"])
def test_each_serving_is_clocked_from_the_seed_its_published_calibration_recorded(module):
    # The seeds a plan gives its servings are the ones the committed token-level calibrations
    # were measured with, serving by serving, so a calibration planned today drives the same
    # clocks.
    task = importlib.import_module(module).TASK
    recorded = json.loads((REPO / "tasks" / module.split(".")[1] / "defaults" / "calibration.json").read_text())
    seeds = recorded["draw"]["serving_seeds"]
    assert sorted(seeds) == ["sglang", "vllm"] and all(len(s) == MIN_SERVINGS for s in seeds.values())
    for engine_seeds in seeds.values():
        assert engine_seeds == [serving_seed_of(task.seed, i) for i in range(len(engine_seeds))]


def test_a_report_calibration_plans_its_servings_per_engine(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, module="cal_report.task", engines=("vllm",), fanout=2, servings_per_engine=3)
    campaign = InstrumentParams.model_validate(plan.stage("instrument").params).campaign
    assert (campaign.kind, campaign.servings_per_engine) == ("report", 3)
    assert _shares(plan) == [[("vllm", 0), ("vllm", 2)], [("vllm", 1)]]
    # The sampling gate runs no fp32 preflight: the instrument is the reference role alone.
    assert plan.stage("instrument").shape.gpus == reference_shape(fixture.report_task(),
                                                                    runner_memory_gib=RUNNER_GIB).gpus
    lines: list[str] = []
    one = _plan(tmp_path, module="cal_report.task", engines=("vllm",), echo=lines.append)
    campaign = InstrumentParams.model_validate(one.stage("instrument").params).campaign
    assert campaign.servings_per_engine == DEFAULT_REPORT_SERVINGS
    # A fanout above the servings deals one serving to each allocation.
    assert _shares(one) == [[("vllm", 0)]] and len(one.allocations) == 2
    assert lines[-1] == ("calibrating the sampling gate on vllm: 1 servings per engine, in 1 allocations after the "
                         "instrument's")


def test_servings_are_dealt_round_robin_over_the_fanout(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, fanout=3)
    # The engines interleave in name order (sglang 0, vllm 0, sglang 1, ...), and the n-th
    # serving goes to allocation n mod 3, so every allocation mixes the engines.
    s, v = "sglang", "vllm"
    assert _shares(plan) == [[(s, 0), (v, 1), (s, 3), (v, 4), (s, 6), (v, 7)],
                             [(v, 0), (s, 2), (v, 3), (s, 5), (v, 6)],
                             [(s, 1), (v, 2), (s, 4), (v, 5), (s, 7)]]
    assert plan.allocations[1].stages == ["capture_sglang_0", "capture_vllm_1", "capture_sglang_3", "capture_vllm_4",
                                          "capture_sglang_6", "capture_vllm_7", "score_1"]
    campaign = InstrumentParams.model_validate(plan.stage("instrument").params).campaign
    assert [plan.stage(f"score_{j}").shape.time_sec for j in (1, 2, 3)] == [
        calibration_grower.score_sec(campaign, 6), calibration_grower.score_sec(campaign, 5),
        calibration_grower.score_sec(campaign, 5)]


def test_a_share_allocation_past_a_qos_limit_runs_under_a_qos_that_holds_it(tmp_path, login, in_catalog):
    # Sixteen servings at a fanout of 4: each share allocation takes four servings and their
    # scoring, over a day, past the normal QOS, so it is planned under long; the
    # instrument's allocation fits normal.
    plan = _plan(tmp_path, config=limited_config(tmp_path, "{normal = 24, long = 168}"))
    first, *shares = plan.allocations
    assert sum(len(a.stages) - 1 for a in shares) == 2 * MIN_SERVINGS and len(shares) == 4
    assert first.placement.qos == "normal" and first.request.time_sec <= 24 * 3600
    assert all(a.placement.qos == "long" and 24 * 3600 < a.request.time_sec <= 168 * 3600 for a in shares)


def test_a_share_allocation_no_qos_holds_is_refused_naming_the_smallest_fanout_that_fits(tmp_path, login,
                                                                                          in_catalog):
    # Sixteen servings at a fanout of 4 put four in each share allocation, past the one QOS's
    # day; the refusal names the smallest --fanout the planner fits under it, the first that
    # deals fewer servings to each share.
    config = limited_config(tmp_path, "{normal = 24}")

    def fits(fanout: int) -> bool:
        try:
            _plan(tmp_path, config=config, fanout=fanout)
        except PlanError:
            return False
        return True

    smallest = next(f for f in range(5, 2 * MIN_SERVINGS + 1) if fits(f))
    assert not fits(4) and not fits(smallest - 1)
    with pytest.raises(PlanError, match=r"^allocation 1: it asks for [0-9.]+ h, more than any QOS of the site lets a "
                                        r"job on partition gpu-a ask \(site.qos_by_hours\): the longest, normal, "
                                        r"allows 24 h; the pipeline lays this allocation out itself, so the planner "
                                        rf"does not split it; the smallest --fanout whose allocations a QOS holds is "
                                        rf"{smallest}$"):
        _plan(tmp_path, config=config)
    with pytest.raises(PlanError, match=f"the smallest --fanout whose allocations a QOS holds is {smallest}$"):
        _plan(tmp_path, config=config, fanout=smallest - 1)
    shares = max(len(a.stages) - 1 for a in _plan(tmp_path, config=config, fanout=smallest).allocations[1:])
    assert shares < 4


def test_a_refusal_no_fanout_cures_says_so(tmp_path, login, in_catalog):
    # The instrument's allocation alone takes longer than an hour, whatever the fanout.
    with pytest.raises(PlanError, match=r"^allocation 0: .*allows 1 h; the pipeline lays this allocation out itself, "
                                        r"so the planner does not split it; no --fanout fits$"):
        _plan(tmp_path, config=limited_config(tmp_path, "{normal = 1}"))


def test_an_audio_task_is_refused(tmp_path, login, in_catalog):
    with pytest.raises(PlanError) as refused:
        _plan(tmp_path, module="cal_audio.task", engines=("vllm",), provisioned=False)
    message = str(refused.value)
    assert message.startswith(f"{fixture.TASK_ID} cannot be calibrated: task '{fixture.TASK_ID}' answers with audio")
    assert message.endswith("A run of the task uses its default, defaults/calibration.json beside its module, "
                            "which fits the task as its module declares it, without task overrides that change its "
                            "workload.")


def test_a_task_without_measured_members_is_refused(tmp_path, login, in_catalog):
    in_catalog(roles=("reference", "member-vllm"))
    with pytest.raises(PlanError, match=re.escape("evaluate_task declares no member-sglang image")):
        _plan(tmp_path)


def test_a_task_that_declares_no_reference_image_calibrates_on_its_harness(tmp_path, login, in_catalog):
    # As a grading's verify stage runs the reference engine on the harness image then.
    in_catalog(roles=("member-vllm", "member-sglang"))
    campaign = InstrumentParams.model_validate(_plan(tmp_path).stage("instrument").params).campaign
    assert campaign.reference_image == catalog.task_images(fixture.TASK_ID).harness


@pytest.mark.parametrize(("module", "internet"), [("cal_flips.task", False), ("cal_agentic.task", True)])
def test_a_member_is_sealed_unless_its_tools_reach_the_network(tmp_path, login, in_catalog, module, internet):
    # The members serve as a graded artifact is served, sealed with no egress, unless the task's
    # agent runs tools that call the network itself (no provisioned assets).
    campaign = InstrumentParams.model_validate(_plan(tmp_path, module=module).stage("instrument").params).campaign
    assert campaign.member_internet is internet


def test_a_serving_drives_with_the_request_timeout_and_its_tasks_drive_wall(tmp_path, login, in_catalog):
    # Each request of a serving's drive has the graded drive's timeout, and the drive the task's
    # own wall, which the capture stage's time holds in place of an open drive's.
    plan = _plan(tmp_path, module="cal_agentic.task")
    campaign = InstrumentParams.model_validate(plan.stage("instrument").params).campaign
    assert (campaign.request_timeout_sec, campaign.drive_timeout_sec) == (REQUEST_TIMEOUT_SEC, fixture.DRIVE_WALL_SEC)
    walled = plan.stage("capture_vllm_0").shape.time_sec
    open_campaign = InstrumentParams.model_validate(_plan(tmp_path).stage("instrument").params).campaign
    assert (open_campaign.request_timeout_sec, open_campaign.drive_timeout_sec) == (REQUEST_TIMEOUT_SEC, None)
    assert _plan(tmp_path).stage("capture_vllm_0").shape.time_sec - walled == OPEN_DRIVE_SEC - fixture.DRIVE_WALL_SEC


class _HeavyAssets(FakeAssets):
    """A tool bundle whose tools ask for more memory than the reference role has."""

    memory_mb = (REFERENCE_SHAPE.memory_gib - RUNNER_GIB + 24) * 1024


def test_a_member_whose_tools_need_more_memory_is_given_it():
    task = fixture.agentic_task()
    heavy = task.model_copy(update={"agent": dataclasses.replace(task.agent, assets=_HeavyAssets())})
    # The reference role's memory, the reference job's less the runner's reserve.
    role = REFERENCE_SHAPE.memory_gib - RUNNER_GIB
    assert calibration.member_shape(task, ["vllm"], runner_memory_gib=RUNNER_GIB).memory_gib == role
    assert calibration.member_shape(heavy, ["vllm", "sglang"], runner_memory_gib=RUNNER_GIB).memory_gib == role + 24


def test_only_the_engines_asked_for_need_a_measured_image(tmp_path, login, in_catalog):
    in_catalog(roles=("reference", "member-sglang"))
    plan = _plan(tmp_path, engines=("sglang",))
    assert {d["name"] for d in plan.stage("images").params["images"]} == {
        "cuda", "evaluate-task-member-sglang", "evaluate-task-reference"}


@pytest.mark.parametrize("port", [8888, 8899])
def test_a_calibration_of_a_task_on_an_interceptor_port_is_refused(tmp_path, login, in_catalog, port):
    with pytest.raises(ValueError, match=f"^{fixture.TASK_ID}'s contract port {port} is refused: the egress "
                                         f"interceptor's [a-z ]+ listens on {port} "):
        _plan(tmp_path, overrides={"task.contract.port": str(port)})


def test_a_calibration_refuses_a_dirty_checkout(tmp_path, login, in_catalog, monkeypatch):
    dirty = login.model_copy(update={"dirty_diff_sha256": "e" * 64})
    monkeypatch.setattr(commands, "code_state", lambda path: dirty)
    with pytest.raises(PlanError, match="uncommitted changes"):
        _plan(tmp_path)


@pytest.mark.parametrize(("module", "kw", "refusal"), [
    pytest.param("cal_flips.task", {"servings_per_engine": 2}, "does not apply to a token-level calibration",
                 id="flips-servings"),
    pytest.param("cal_plain.task", {}, "declares no calibrated verifier", id="no-gate"),
    pytest.param("cal_flips.task", {"engines": ("hf_pinned",)}, "cannot be calibration pool members",
                 id="batch-engine"),
    pytest.param("cal_flips.task", {"engines": ()}, "at least one pool engine", id="no-engine"),
    pytest.param("cal_report.task", {"engines": ("vllm",), "servings_per_engine": 0},
                 "servings_per_engine must be at least 1, got 0", id="no-servings"),
    pytest.param("cal_reports.task", {"engines": ("vllm",)},
                 "a report calibration measures one ReportFaithfulnessVerifier's ceiling, but this task declares 2; "
                 "declare exactly one", id="two-report-verifiers"),
    pytest.param("cal_chunks.task", {}, "this task's calibrated verifiers declare 2 different teacher-forcing query "
                 "chunks ([512, 1024])", id="two-chunks"),
])
def test_a_campaign_the_task_cannot_take_is_refused(tmp_path, login, in_catalog, module, kw, refusal):
    with pytest.raises(PlanError, match=re.escape(refusal)):
        _plan(tmp_path, module=module, **kw)


@pytest.mark.parametrize(("kw", "field"), [
    pytest.param({"eps": 0.0}, "eps", id="eps"),
    pytest.param({"delta": 1.0}, "delta", id="delta"),
    pytest.param({"fanout": 0}, "fanout", id="fanout"),
    pytest.param({"sessions_per_draw": 0}, "sessions_per_draw", id="sessions"),
    pytest.param({"max_servings": 7}, "max_servings_per_engine", id="bound"),
])
def test_settings_out_of_range_are_refused(tmp_path, login, in_catalog, kw, field):
    with pytest.raises(PlanError, match=r"the campaign's settings are not valid: .*" + re.escape(field)):
        _plan(tmp_path, **kw)


@pytest.mark.parametrize(("overrides", "refusal"), [
    pytest.param({"engines": ["vllm", "vllm"]}, "engines ['vllm', 'vllm'] name a member twice", id="twice"),
    pytest.param({"member_images": {"vllm": "evaluate-task-member-vllm"}},
                 "member_images names ['vllm'], not the engines ['sglang', 'vllm']", id="images"),
    pytest.param({"member_engine_configs": {"vllm": "{}"}},
                 "member_engine_configs names ['vllm'], not the engines ['sglang', 'vllm']", id="configs"),
    pytest.param({"servings_per_engine": 2}, "servings_per_engine is set for the sampling gate and only for it",
                 id="servings"),
    pytest.param({"member_shape": fixture.ROLE_SHAPE.model_copy(update={"gpus": 0, "gpu_kind": None})},
                 "a pool member and the reference each run on a GPU", id="member-gpu"),
    pytest.param({"reference_shape": fixture.ROLE_SHAPE.model_copy(update={"gpus": 0, "gpu_kind": None})},
                 "a pool member and the reference each run on a GPU", id="reference-gpu"),
])
def test_campaign_settings_that_contradict_themselves_are_refused(overrides, refusal):
    with pytest.raises(ValidationError, match=re.escape(refusal)):
        fixture.campaign(**overrides)


def test_a_sampling_gates_settings_name_its_servings():
    with pytest.raises(ValidationError, match="servings_per_engine is set for the sampling gate and only for it"):
        fixture.campaign(fixture.report_task(), servings_per_engine=None)


def test_task_overrides_reach_the_task_and_its_seeds(tmp_path, login, in_catalog):
    provision_task(tmp_path / "store", "cal_flips.task", {"task.seed": "7"})
    plan = calibration.plan_calibrate(
        config=write_config(tmp_path), overrides={"task.seed": "7"}, task_module="cal_flips.task",
        engines=["vllm"], fanout=1, servings_per_engine=None, sessions_per_draw=None, eps=DEFAULT_EPS,
        mu_per_mtok=DEFAULT_MU_PER_MTOK, delta=DEFAULT_DELTA, eot_expected_max=DEFAULT_EOT_EXPECTED_MAX,
        max_servings_per_engine=24, job_name="hls-calibrate", echo=[].append)
    params = InstrumentParams.model_validate(plan.stage("instrument").params)
    assert params.task.overrides == {"task.seed": 7} and params.campaign.task_seed == 7
    seeded = fixture.flips_task().model_copy(update={"seed": 7})
    assert CaptureParams.model_validate(plan.stage("capture_vllm_5").params).seed == serving_seed_of(seeded.seed, 5)


def test_the_cli_plans_a_calibration_and_runs_it(tmp_path, login, in_catalog, monkeypatch):
    seen = {}

    def run(plan, overrides, *, scheduler=None, echo=print):
        seen.update(plan=plan, overrides=overrides)
        return 0

    monkeypatch.setattr(commands, "start", run)
    provision_task(tmp_path / "store", "cal_report.task", {"task.seed": "7"})
    result = CliRunner().invoke(cli.app, [
        "calibrate", "cal_report.task", "--engines", "vllm", "--fanout", "2", "--servings-per-engine", "3",
        "--sessions-per-draw", "2", "--eps", "0.1", "--max-servings-per-engine", "9",
        "--config", str(write_config(tmp_path)), "--task.seed", "7"])
    assert result.exit_code == 0, result.output
    campaign = InstrumentParams.model_validate(seen["plan"].stage("instrument").params).campaign
    assert (campaign.engines, campaign.fanout, campaign.servings_per_engine, campaign.sessions_per_draw) == (
        ["vllm"], 2, 3, 2)
    assert (campaign.eps, campaign.max_servings_per_engine, campaign.task_seed) == (0.1, 9, 7)
    assert (campaign.mu_per_mtok, campaign.delta) == (DEFAULT_MU_PER_MTOK, DEFAULT_DELTA)
    assert seen["overrides"] == {"task.seed": "7"}


def test_the_cli_says_a_refusal_in_one_line_and_exits_1(tmp_path, login, in_catalog):
    result = CliRunner().invoke(cli.app, ["calibrate", "cal_flips.task", "--servings-per-engine", "2",
                                          "--config", str(write_config(tmp_path))])
    assert result.exit_code == 1, result.output
    [line] = result.stderr.splitlines()
    assert line.startswith(f"hls calibrate: {fixture.TASK_ID} cannot be calibrated: servings_per_engine does not "
                           "apply to a token-level calibration")


def test_the_cli_default_pool_is_both_serving_stacks(tmp_path, login, in_catalog, monkeypatch):
    seen = {}
    monkeypatch.setattr(commands, "start", lambda plan, overrides, **kw: seen.update(plan=plan) or 0)
    provision_task(tmp_path / "store", "cal_flips.task")
    result = CliRunner().invoke(cli.app, ["calibrate", "cal_flips.task", "--config", str(write_config(tmp_path))])
    assert result.exit_code == 0, result.output
    campaign = InstrumentParams.model_validate(seen["plan"].stage("instrument").params).campaign
    assert (campaign.engines, campaign.fanout) == (["vllm", "sglang"], calibration.DEFAULT_FANOUT)
    assert campaign.max_servings_per_engine == calibration.DEFAULT_MAX_SERVINGS_PER_ENGINE


#: Tables for a score share and a serving that no first plan of vllm at a fanout of 1 holds.
GROWN_TABLES = {"resources.evaluate_task.score_2.time_sec": "9000",
                "resources.evaluate_task.capture_vllm_12.time_sec": "9000"}


def test_a_table_for_a_stage_the_grower_may_add_passes_through_its_declaration(tmp_path, login, in_catalog):
    provision_task(tmp_path / "store", "cal_flips.task")
    plan = calibration.plan_calibrate(
        config=write_config(tmp_path), overrides=GROWN_TABLES, task_module="cal_flips.task", engines=["vllm"],
        fanout=1, servings_per_engine=None, sessions_per_draw=None, eps=DEFAULT_EPS, mu_per_mtok=DEFAULT_MU_PER_MTOK,
        delta=DEFAULT_DELTA, eot_expected_max=DEFAULT_EOT_EXPECTED_MAX, max_servings_per_engine=24,
        job_name="hls-calibrate", echo=[].append)
    assert not {"score_2", "capture_vllm_12"} & {s.name for s in plan.stages}
    # Planned, and not warned of: the grower may add both, and the tables then apply.
    assert not [w for w in plan.warnings if w.startswith("[resources.")]


def test_a_calibration_table_on_another_plan_of_the_task_is_ignored_with_a_warning(tmp_path, login, in_catalog):
    plan = commands.plan_provision(config=write_config(tmp_path), overrides=GROWN_TABLES, task_module="cal_flips.task",
                                   weights=True, job_name="hls-provision", echo=[].append)
    assert [w for w in plan.warnings if w.startswith("[resources.")] == [
        f"[resources.{fixture.TASK_ID}.{s}] names stage {s}, which this plan does not run: the table is ignored"
        for s in ("score_2", "capture_vllm_12")]
    with pytest.raises(PlanError, match=re.escape(f"resources.{fixture.TASK_ID}.scores_2")):
        commands.plan_provision(config=write_config(tmp_path), task_module="cal_flips.task", weights=True,
                                overrides={f"resources.{fixture.TASK_ID}.scores_2.time_sec": "9000"},
                                job_name="hls-provision", echo=[].append)


def test_each_stages_time_holds_the_bound_of_its_engines_loads():
    """Each stage's time is the fixed staging allowance, its loads' bound rounded up, and what else
    it runs."""
    campaign = fixture.campaign(load_timeout_sec=100.5, drive_timeout_sec=None)
    assert calibration.instrument_sec(campaign) == STAGING_SEC + 2 * 101 + calibration.PREFLIGHT_SEC
    assert calibration_grower.capture_sec(campaign) == STAGING_SEC + 101 + OPEN_DRIVE_SEC + DRIVE_MARGIN_SEC
    assert calibration_grower.score_sec(campaign, 3) == STAGING_SEC + 101 + 3 * calibration.SCORE_PER_SERVING_SEC


def test_a_task_that_declares_its_verify_time_has_each_serving_scored_in_it(tmp_path, login, in_catalog, monkeypatch):
    # deepresearch's verify takes up to 10,888 s, so its evaluation allocation declares
    # 23,400 s for one; a task module declaring its artifact so gives its calibration's
    # reference role that time and each serving a score stage scores as much.
    deepresearch = importlib.import_module("tasks.deepresearch.task")
    declared = commands.evaluation_allocation(commands.produced_artifact("tasks.deepresearch.task"))
    assert declared is not None and declared.verify_time_sec == 23_400
    monkeypatch.setattr(importlib.import_module("cal_flips.task"), "ARTIFACT", deepresearch.ARTIFACT, raising=False)
    plan = _plan(tmp_path)
    campaign = InstrumentParams.model_validate(plan.stage("instrument").params).campaign
    assert (campaign.score_per_serving_sec, campaign.reference_shape.time_sec) == (23_400, 23_400)
    shared = _shares(plan)[0]
    assert plan.stage("score_1").shape.time_sec == STAGING_SEC + READINESS_TIMEOUT_SEC + len(shared) * 23_400
    # A task that declares none has its servings scored in the planner's own figure.
    other = _plan(tmp_path, module="cal_report.task", engines=("vllm",))
    assert InstrumentParams.model_validate(other.stage("instrument").params).campaign.score_per_serving_sec == (
        calibration.SCORE_PER_SERVING_SEC)


def test_a_table_that_gives_a_warm_stage_more_time_gives_its_load_as_much_more(tmp_path, login, in_catalog):
    provision_task(tmp_path / "store", "cal_flips.task")
    plan = calibration.plan_calibrate(
        config=write_config(tmp_path), overrides={f"resources.{fixture.TASK_ID}.warm_vllm.time_sec": "20000"},
        task_module="cal_flips.task", engines=["vllm"], fanout=1, servings_per_engine=None, sessions_per_draw=None,
        eps=DEFAULT_EPS, mu_per_mtok=DEFAULT_MU_PER_MTOK, delta=DEFAULT_DELTA, eot_expected_max=DEFAULT_EOT_EXPECTED_MAX,
        max_servings_per_engine=24, job_name="hls-calibrate", echo=[].append)
    warm = plan.stage("warm_vllm")
    assert warm.shape.time_sec == 20000
    params = WarmParams.model_validate(warm.params)
    assert params.staging_sec == STAGING_SEC
    assert warm_load_sec(params, 20000) == 20000 - STAGING_SEC - DRIVE_MARGIN_SEC


def test_an_engines_cache_config_holds_every_setting_it_was_built_with():
    # mslo's and tab_completion's vLLM members share an image: their caches are apart only by what
    # each member is built with.
    task = fixture.flips_task()
    longer = task.model_copy(update={"max_context_len": 4096})
    other_model = task.model_copy(update={"model": task.model.model_copy(update={"revision": "f" * 40})})
    configs = [engine_cache_config(member_engines(t)[e]()) for t in (task, longer, other_model) for e in ("vllm",
                                                                                                         "sglang")]
    assert len(set(configs)) == 6
    assert engine_cache_config(member_engines(task)["vllm"]()) == configs[0]
    assert len({engine_cache_key("a" * 64, "H200", c, seal=None) for c in configs}) == 6
