"""``hls calibrate`` on one machine: the command plans and submits, the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` runs the real runner for every allocation, and every
stage runs its real role program with fake engines (``tests.calibration.fake_roles``): the
instrument check, each serving's capture and each share's scores in fake containers on the
measured images, and the aggregate on the plan's interpreter. The orchestrator grows the plan
twice: once with the servings the first eight leave short of the exposure target, once with the
aggregate. Only srun, the cluster, the hub and the container runtime are stood in for, with the
stand-ins of the ``hls grade`` end to end."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from hls.measured import store_key
from hls.calibrate import DEFAULT_DELTA, DEFAULT_EPS, serving_seed_of
from hls.pipeline import commands
from hls.pipeline.plan import PlanStore
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages import calibration as stages
from hls.stages.images import IMAGE_FILE, image_keys
from hls.store.cache import ContentCache
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.verify.calibration import CALIBRATION_FILE, Calibration, read_calibration, workload_fingerprint
from hls.verify.calibration_roles import AggregateRecord, EngineCacheRecord
from hls.substrate.apptainer.images import ApptainerImages
from tests.calibration import engines, fixture
from tests.drive import fake_container
from tests.evaluate import test_grade_end_to_end_local as grade_e2e
from tests.evaluate.test_grade_end_to_end_local import _slurm
from tests.pipeline.hls_cli import run_hls
from tests.provision.warmer_config import SECRETS
from tests.recipes.warmer_site import site_config
from tests.store.entries import populate
from tests.substrate import fake_pasta

#: The ``hls grade`` end to end's fake runtime, seal and hub.
bins = grade_e2e.bins
#: Teacher-forced tokens of one serving: the fake member's replies to the test split's sessions.
PER_SERVING = len(fixture.TEST_ROWS) * engines.TOKENS_PER_REPLY
#: The per-token tail tolerance whose exposure target is 49,000 tokens: the first eight servings
#: of 5,000 leave the engine 9,000 short, which two more servings close.
MU = math.log(40) * 1e6 / 49_000


def _image(path: Path, role: bytes) -> bytes:
    """A fake container image the runtime can run, told apart by *role*."""
    return fake_container.make_image(path, {".singularity.d/labels.json": (b"{}", 0o644),
                                            "role.txt": (role, 0o644)}).read_bytes()


def _cache_images(store: PosixStore, files: dict[str, bytes]) -> None:
    """The measured images in the cache as ``hls images build`` would have left them."""
    keys = image_keys(order(catalog.task_images(fixture.TASK_ID).decls), ApptainerImages.recipe_key)
    cache, holder = ContentCache(store), Holder.here("test")
    for name, data in files.items():
        populate(cache, "images", keys[name], lambda d, data=data: (d / IMAGE_FILE).write_bytes(data), holder=holder)


def _output(store: PosixStore, layout: InstanceLayout, stage: str, name: str) -> bytes:
    manifest, _ = read_manifest(store, layout, stage)
    return store.read_bytes(manifest.outputs[name].key)


def test_hls_calibrate_end_to_end_local(tmp_path, login, in_catalog, bins, monkeypatch):
    store = PosixStore(tmp_path / "store")
    member, reference = _image(tmp_path / "member.sif", b"member-vllm"), _image(tmp_path / "reference.sif", b"reference")
    in_catalog(roles=("member-vllm", "reference"))
    _cache_images(store, {"evaluate-task-member-vllm": member, "evaluate-task-reference": reference})
    monkeypatch.setattr(stages, "ROLE_MODULE", "tests.calibration.fake_roles")
    monkeypatch.setenv("FAKE_CONTAINER_RECORDS", str(tmp_path / "records"))
    slurm = _slurm(tmp_path)
    config = site_config(tmp_path, secrets=SECRETS)
    # The plan sizes its stages' copies of the task's data by what the cache holds.
    grade_e2e.provision(tmp_path, config, "cal_flips.task")
    # The gate's precision, false-alarm budget and expected end-of-turn count are the command's
    # defaults.
    code, lines = run_hls(slurm, ["calibrate", "cal_flips.task", "--engines", "vllm", "--fanout", "2",
                                  "--mu-per-mtok", repr(MU), "--max-servings-per-engine", "24",
                                  "--job-name", "hls-calibrate-test", "--config", str(config)])
    assert code == 0, lines
    instance = next(line for line in lines if line.startswith("instance ")).removeprefix("instance ")
    assert instance.startswith(f"calibrations/{fixture.TASK_ID}/")
    assert f"{instance}: succeeded: every stage published" in lines
    layout = InstanceLayout(instance)
    plan = PlanStore(store, layout).latest()
    # The first plan's three allocations, the top-up's two and the aggregate's: six jobs.
    assert [a.stages for a in plan.allocations] == [
        ["provision", "images", "instrument", "warm_vllm"],
        ["capture_vllm_0", "capture_vllm_2", "capture_vllm_4", "capture_vllm_6", "score_1"],
        ["capture_vllm_1", "capture_vllm_3", "capture_vllm_5", "capture_vllm_7", "score_2"],
        ["capture_vllm_8", "score_3"], ["capture_vllm_9", "score_4"], ["aggregate"]]
    assert len(slurm.jobs) == 6
    # Each of the ten servings ran its member sealed, as a graded artifact is served, and so did the
    # member's warm stage, which built the engine cache they replayed.
    assert len(fake_pasta.calls(tmp_path / "pasta.log")) == 11
    assert all(read_manifest(store, layout, s.name) is not None for s in plan.stages)
    cal = Calibration.model_validate_json(_output(store, layout, "aggregate", "calibration"))
    assert (cal.tolerances.eps, cal.tolerances.mu_per_mtok, cal.tolerances.delta) == (DEFAULT_EPS, MU, DEFAULT_DELTA)
    task = fixture.flips_task()
    assert cal.workload_fingerprint == workload_fingerprint(task) and cal.task == "cal_flips.task"
    assert cal.rate_side["vllm"].draws == 10 and cal.curve_side.engines["vllm"].exposure == 10 * PER_SERVING
    assert cal.q_max == pytest.approx(engines.FLIPS / engines.TOKENS_PER_REPLY)
    assert cal.preflight.passed and cal.stack.device_name == engines.STACK["device_name"]
    assert cal.engines["vllm"].version == engines.MEMBER_VERSION
    assert cal.draw.serving_seeds == {"vllm": tuple(serving_seed_of(task.seed, d) for d in range(10))}
    summary = AggregateRecord.model_validate_json(_output(store, layout, "aggregate", "aggregate"))
    assert (summary.servings, summary.tokens) == ({"vllm": list(range(10))}, {"vllm": 10 * PER_SERVING})
    # Each engine's owner in the first allocation published its engine cache, and every stage of
    # it after replayed that one: the instrument the reference's, which seeded every score stage,
    # and the warm stage the member's, which seeded every capture.
    caches = {s.name: EngineCacheRecord.model_validate_json(_output(store, layout, s.name, "engine_cache"))
              for s in plan.stages if s.kind != "calibrate_aggregate" and s.name not in ("provision", "images")}
    assert caches["instrument"].published and caches["warm_vllm"].published
    reference, member = caches["instrument"].entry, caches["warm_vllm"].entry
    assert all(caches[f"score_{j}"] == EngineCacheRecord(key=caches["instrument"].key, seeded_from=reference)
               for j in range(1, 5))
    members = [caches[f"capture_vllm_{d}"] for d in range(10)]
    assert all(r == EngineCacheRecord(key=caches["warm_vllm"].key, seeded_from=member) for r in members)
    assert summary.member_caches == {"vllm": dict(enumerate(members))}
    # The calibration records the hardware of the allocations its servings ran on.
    assert cal.hardware is not None and len(cal.hardware) == 1
    # The aggregate wrote it into the store's measured area too, byte for byte, where a run of the
    # task finds it and the task loads it.
    measured = store_key(fixture.TASK_ID, CALIBRATION_FILE)
    assert store.read_bytes(measured) == _output(store, layout, "aggregate", "calibration")
    assert read_calibration(store.path(measured), task)[0] == cal
    key = read_manifest(store, layout, "aggregate")[0].outputs["calibration"].key
    assert lines[-1] == (f"calibration (flips): ceiling {cal.q_max:.6g} over vllm 10 servings, in {key}; the store's "
                         f"{measured} holds it, and runs of the task use it")
    # hls resume of the ended instance says the same, as hls calibrate did.
    resumed: list[str] = []
    assert commands.resume(instance, config=config, scheduler=slurm, echo=resumed.append) == 0
    assert resumed[0] == f"{instance} already ended: succeeded: every stage published"
    assert resumed[-1:] == lines[-1:]
    # A calibration that did not succeed says nothing of it, though its aggregate published here.
    said: list[str] = []
    commands._say_calibration(plan.model_copy(update={"outcome": plan.outcome.model_copy(update={"status": "void"})}),
                            said.append)
    assert said == []
    # The calibration is said from the record the aggregate's manifest recorded, or not at all.
    ref = read_manifest(store, layout, "aggregate")[0].outputs["aggregate"]
    store.path(ref.key).write_bytes(store.read_bytes(ref.key) + b" ")
    with pytest.raises(ValueError, match=re.escape(f"{ref.key} does not match its manifest")):
        commands.resume(instance, config=config, scheduler=slurm, echo=[].append)
