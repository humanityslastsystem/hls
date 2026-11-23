"""``hls baseline-credit``'s planning: N fresh drives of the measured baseline, each scored, then
the fold, in one allocation or, with a fanout, side by side; the refusals; the command; and no
drive measured in the baseline credit module itself."""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import baseline_credit, cli
from hls.drive.bounds import REPLY_MAX_BYTES, drive_bounds, trace_disk_bytes
from hls.pipeline import commands, credit
from hls.pipeline.pipelines import (DRIVE_MARGIN_SEC, EVALUATE_IMAGES_CPUS, KNOWN_STAGE_NAMES, RUNTIME_START_SEC,
                                    STAGE_NAME_PATTERNS, STAGING_SEC, TRACE_COPY_SEC, artifact_shape,
                                    baseline_warm_sec, driver_shape, images_shape, score_shape)
from hls.pipeline.plan import Plan
from hls.pipeline.planner import PlanError
from hls.recipes import catalog
from hls.recipes.model import order
from hls.settings import load_settings
from hls.spec.resources import concurrent
from hls.stages.baseline_warm import BaselineWarmParams, load_sec
from hls.stages.deploy_drive import PINNED_IMAGES, SEAL_PROBE_SEC, DeployDriveParams, ImageArtifact, served_cache_key
from hls.stages.engine_cache import COLD_LOAD_SEC
from hls.stages import score
from hls.stages.images import IMAGE_FILE, KEPT_DISK_GIB, image_keys
from hls.store.cache import ContentCache
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from tests.credit import fixture
from tests.evaluate import fixture as evaluate
from tests.pipeline.plans import CODE
from tests.provision.cached import provision_into, provision_task
from hls.provision.mounts import copied_entries, copies_gib
from hls.taskref import TaskRef
from hls.recipes.render import recipe_key as def_key
from hls.substrate.apptainer.images import ApptainerImages
from tests.provision.warmer_config import write_config
from tests.recipes.declared import declare
from tests.store.entries import populate

LAYER = catalog.image_set("sample_task").layer
#: The catalog's reader of a task module's images, as the fixtures find it before they declare
#: the fixture task's.
DECLARED_IMAGES = catalog.image_set
#: The memory the test site leaves to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4

#: The credit task over a stream whose packets are as small as multitenant_slo's, so the summary
#: of one reply at ``REPLY_MAX_BYTES`` needs more than the score role's least memory.
STREAM = "cr_stream.task"
#: The credit task's drive bounds, which every variant of it declares: the evaluate tests' task's.
BOUNDS = drive_bounds(fixture.credit_task())
#: The disk in whole GiB a drive's driver holds for that task's trace at its most: its 1 GiB cap
#: and 1 MiB of requests make a room of 2,150,237,314 bytes on disk (``trace_disk_bytes``), so 3.
TRACE_GIB = 3


def _plan(tmp_path: Path, *, module: str = "cr_task.task", drives: int = 3, fanout: int = 1,
          overrides: dict[str, str] | None = None, provisioned: bool = True) -> Plan:
    """A credit's first plan, the task's provisioning, its task overrides applied, in the cache first
    unless *provisioned* is False."""
    if provisioned:
        provision_task(tmp_path / "store", module, overrides)
    return credit.plan_credit(config=write_config(tmp_path), overrides=overrides or {}, task_module=module,
                              drives=drives, fanout=fanout, job_name="hls-credit", echo=[].append)


def test_a_credit_plans_n_fresh_drives_of_the_measured_baseline(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, module=STREAM)
    assert plan.pipeline == "credit" and plan.task == evaluate.TASK_ID
    assert plan.instance.startswith(f"credits/{evaluate.TASK_ID}/") and plan.instance.count("/") == 2
    names = ["provision", "images", "warm_baseline", "drive_1", "credit_1", "drive_2", "credit_2", "drive_3",
             "credit_3", "fold"]
    assert [s.name for s in plan.stages] == names
    assert all(s in KNOWN_STAGE_NAMES or any(re.fullmatch(p, s) for p in STAGE_NAME_PATTERNS) for s in names)
    [alloc] = plan.allocations
    assert alloc.stages == names
    artifact = ImageArtifact(image="evaluate-task-baseline", entrypoint=evaluate.BASELINE_ENTRYPOINT,
                             runtime_env=evaluate.BASELINE_ENV)
    task = fixture.credit_task(stream=True)
    # The driver holds the trace's disk at its most, as a grading's drive does.
    assert math.ceil(trace_disk_bytes(BOUNDS.trace_cap_bytes, BOUNDS.trace_request_bytes) / 2**30) == TRACE_GIB
    art = artifact_shape(task, runner_memory_gib=plan.settings.run.runner_memory_gib)
    drv = driver_shape(task).model_copy(update={"disk_gib": TRACE_GIB})
    roles = concurrent([art, drv])
    # Each drive's disk holds its two image trees and the copies of its roles' data, by the cache's
    # provision record (``copied_entries``), which each declares.
    copies = copied_entries(("artifact", "driver"), provision_into(tmp_path / "store", task))
    # Each drive's time is its roles' after the fixed staging allowance and its seal probe, with
    # the trace's copy to the store after its wall, as a grading's drive has.
    staging = STAGING_SEC + math.ceil(SEAL_PROBE_SEC)
    shape = roles.model_copy(update={"memory_gib": roles.memory_gib + RUNNER_GIB,
                                     "disk_gib": roles.disk_gib + PINNED_IMAGES * KEPT_DISK_GIB + copies_gib(copies),
                                     "time_sec": roles.time_sec + staging + TRACE_COPY_SEC})
    ref = {"module": STREAM, "overrides": {}}
    # The warm stage owns the baseline's engine cache entry: it serves the
    # baseline as each drive serves it, in the artifact role's shape, so it asks for no more of any
    # resource than a drive does, beside its one image, the copies of the artifact role's data and
    # its fixed time.
    warm = plan.stage("warm_baseline")
    warm_params = BaselineWarmParams.model_validate(warm.params)
    warm_copies = copied_entries(("artifact",), provision_into(tmp_path / "store", task))
    assert (warm.kind, warm.after) == ("baseline_warm", ["provision", "images"])
    assert warm.shape == art.model_copy(update={"memory_gib": art.memory_gib + RUNNER_GIB,
                                                "disk_gib": art.disk_gib + KEPT_DISK_GIB + copies_gib(warm_copies),
                                                "time_sec": baseline_warm_sec()})
    assert baseline_warm_sec() == STAGING_SEC + RUNTIME_START_SEC + COLD_LOAD_SEC + DRIVE_MARGIN_SEC
    for resource in ("gpus", "cpus", "memory_gib", "disk_gib"):
        assert getattr(warm.shape, resource) <= getattr(shape, resource), resource
    assert (warm_params.artifact, warm_params.serve_command, warm_params.artifact_shape) == (
        artifact, fixture.BASELINE_SERVE, art)
    assert (warm_params.port, warm_params.network, warm_params.data_copies) == (
        task.contract.port, None, warm_copies)
    assert (warm_params.staging_sec, warm_params.runtime_start_sec, warm_params.margin_sec) == (
        STAGING_SEC, RUNTIME_START_SEC, DRIVE_MARGIN_SEC)
    assert load_sec(warm_params, warm.shape.time_sec) == COLD_LOAD_SEC
    for k in (1, 2, 3):
        stage = plan.stage(f"drive_{k}")
        drive = DeployDriveParams.model_validate(stage.params)
        assert (stage.kind, stage.shape, stage.after) == ("deploy_drive", shape,
                                                         ["provision", "images", "warm_baseline"])
        # Each drive's server starts from the warm stage's entry, under the key the warm stage
        # computes from the same parameters.
        assert drive.engine_cache_owner == "warm_baseline"
        assert served_cache_key(drive, "a" * 64, None) == served_cache_key(warm_params, "a" * 64, None)
        assert (drive.artifact, drive.serve_command, drive.data_copies) == (artifact, fixture.BASELINE_SERVE, copies)
        assert (drive.harness_image, drive.split, drive.staging_sec) == (LAYER, "test", staging)
        # Nothing of a baseline's drive is counted, and the trace cap is the task's own.
        assert drive.measured_baseline and drive.reply_max_bytes == REPLY_MAX_BYTES
        assert (drive.trace_cap_bytes, drive.trace_request_bytes) == (BOUNDS.trace_cap_bytes,
                                                                      BOUNDS.trace_request_bytes)
        assert (drive.driver_shape, drive.wall_margin_sec) == (drv, DRIVE_MARGIN_SEC + TRACE_COPY_SEC)
        assert drive.task.model_dump(mode="json") == ref
        credit_stage = plan.stage(f"credit_{k}")
        assert (credit_stage.kind, credit_stage.after) == ("credit_drive", [f"drive_{k}"])
        assert credit_stage.params == {"task": ref, "drive_stage": f"drive_{k}"}
        # The role reads the drive's summary, as a grading's score role does: a stream of token ids
        # makes it 8 GiB and 13 more for one reply's packets (``score_memory_gib``), the runner's 4
        # beside it.
        summary = score_shape(task, RUNNER_GIB)
        assert summary.memory_gib == 8 + 13 + RUNNER_GIB
        assert credit_stage.shape == summary and summary.memory_gib > score.SCORE_BASE_GIB + RUNNER_GIB
    fold = plan.stage("fold")
    assert (fold.kind, fold.shape, fold.after) == ("credit_fold", credit.fold_shape(RUNNER_GIB),
                                                   ["credit_1", "credit_2", "credit_3"])
    assert (fold.shape.gpus, fold.shape.memory_gib) == (0, RUNNER_GIB + credit.FOLD_MEMORY_GIB)
    assert fold.params == {"task": ref, "credit_stages": ["credit_1", "credit_2", "credit_3"]}
    assert {d["name"] for d in plan.stage("images").params["images"]} == {"s-a", LAYER, "cuda",
                                                                          "evaluate-task-baseline"}
    # A credit runs no verifier, no score of a grading and no finalize.
    assert not {"verify", "score", "finalize"} & {s.kind for s in plan.stages}


def test_a_site_that_leaves_the_runner_more_memory_gives_the_fold_and_the_credit_roles_their_own(
        tmp_path, login, in_catalog):
    """The fold runs in the runner and each credit role beside it, so a larger
    ``run.runner_memory_gib`` grows their stages by as much and never fails the plan."""
    plan = _plan(tmp_path, overrides={"run.runner_memory_gib": "10"})
    assert plan.stage("fold").shape.memory_gib == 10 + credit.FOLD_MEMORY_GIB
    assert plan.stage("credit_1").shape == score_shape(fixture.credit_task(), 10)


def test_a_credit_with_fanout_runs_drives_side_by_side_and_folds_after_all(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, drives=5, fanout=2)
    shape = [(a.stages, a.depends_on, a.waits_for) for a in plan.allocations]
    assert shape == [
        (["provision", "images", "warm_baseline"], None, []),
        (["drive_1", "credit_1", "drive_3", "credit_3", "drive_5", "credit_5"], 0, []),
        (["drive_2", "credit_2", "drive_4", "credit_4"], 0, []),
        (["fold"], None, [1, 2]),
    ]
    # Every drive allocation holds a GPU of the task's kind for its drives, and the first one for
    # the warm stage, which builds the baseline's engine cache once before any drive starts.
    assert {(a.request.gpus, a.request.gpu_kind) for a in plan.allocations[0:3]} == {(1, "B200")}
    assert plan.allocations[3].request.gpus == 0


def test_a_credit_in_turn_is_split_to_the_sites_limit_and_a_side_by_side_one_is_refused(tmp_path, login,
                                                                                        in_catalog):
    """At --fanout 1 the GPU policy groups a credit's stages, and splits the nearly 29 h they take
    in turn, the warm stage's 3.1 h among them, into a chain of allocations that a 12 h QOS holds.
    At --fanout 2 the credit lays its allocations out itself, and one of them that the QOS does not
    hold is refused, not split."""
    limit = {"site.qos_by_hours.normal": "12"}
    plan = _plan(tmp_path, drives=3, overrides=limit)
    assert [(a.stages, a.depends_on) for a in plan.allocations] == [
        (["provision", "images", "warm_baseline"], None), (["drive_1", "credit_1"], 0),
        (["drive_2", "credit_2"], 1), (["drive_3", "credit_3", "fold"], 2)]
    assert all(a.placement.qos == "normal" and a.request.time_sec <= 12 * 3600 for a in plan.allocations)
    with pytest.raises(PlanError, match=r"^allocation 1: it asks for 15\.80 h, .*the longest, normal, allows 12 h; "
                                        "the pipeline lays this allocation out itself, so the planner does not split "
                                        "it$"):
        _plan(tmp_path, drives=3, fanout=2, overrides=limit)


@pytest.mark.parametrize("drives, fanout, match", [
    (0, 1, "--drives 0: a baseline credit needs at least 1 drive"), (3, 0, "--fanout 0"), (3, 4, "--fanout 4"),
], ids=["no-drive", "no-fanout", "fanout-past-drives"])
def test_a_credit_needs_a_drive_and_a_fanout_within_its_drives(tmp_path, login, in_catalog, drives, fanout, match):
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, drives=drives, fanout=fanout)


def test_a_credit_takes_one_drive_unless_told_more(tmp_path, login, in_catalog):
    """``--drives`` defaults to one: the plan drives the baseline once and folds that drive."""
    provision_task(tmp_path / "store", "cr_task.task", None)
    plan = credit.plan_credit(config=write_config(tmp_path), overrides={}, task_module="cr_task.task",
                              job_name="hls-credit", echo=[].append)
    assert [s.name for s in plan.stages if s.kind in ("deploy_drive", "credit_drive")] == ["drive_1", "credit_1"]
    assert plan.stage("fold").params["credit_stages"] == ["credit_1"]


def test_a_credit_refuses_a_dirty_checkout(tmp_path, login, in_catalog, monkeypatch, checkout):
    dirty = CODE.model_copy(update={"checkout": checkout, "dirty_diff_sha256": "c" * 64})
    monkeypatch.setattr(commands, "code_state", lambda path: dirty)
    with pytest.raises(PlanError) as e:
        _plan(tmp_path)
    assert str(checkout) in str(e.value) and "uncommitted changes" in str(e.value)


def test_a_task_without_a_measured_baseline_is_refused(tmp_path, login, in_catalog, monkeypatch):
    declare(monkeypatch, {evaluate.TASK_ID: evaluate.image_set(("reference",))})
    with pytest.raises(ValueError, match="evaluate_task declares no baseline image") as e:
        _plan(tmp_path)
    # The credit takes no --measured flag, so its refusal names none.
    assert "--measured" not in str(e.value)


def test_a_credit_drives_only_the_measured_baseline(tmp_path, in_catalog):
    """A credit is a measurement of the task's measured baseline file: any other image the
    catalog knows is refused, so no drive of it could be pooled or go unmetered."""
    task = fixture.credit_task()
    harness = catalog.task_images(evaluate.TASK_ID).harness
    other = ImageArtifact(image=harness, entrypoint=evaluate.BASELINE_ENTRYPOINT)
    with pytest.raises(ValueError, match=f"measured baseline evaluate-task-baseline, not {harness}"):
        credit.credit_stages(task_ref=TaskRef(module="cr_task.task"), task=task, catalog_key=evaluate.TASK_ID,
                             artifact=other, serve_command=[], drives=2, fanout=1,
                             run=load_settings(write_config(tmp_path)).run, seal_memory_gib=0,
                             allocation=None, held=lambda keys: set(), recipe_key=def_key,
                             provisioned=provision_into(tmp_path / "store", task))


@pytest.mark.parametrize("module, match", [
    ("cr_goodput.task", "scores against no baseline credit"),
    ("cr_nobase.task", "declares no baseline server"),
    ("cr_latency.task", "cannot be driven"),
], ids=["goodput", "no-baseline", "no-drive-bounds"])
def test_a_task_whose_credit_cannot_be_measured_is_refused(tmp_path, login, in_catalog, module, match):
    # Each is refused before the plan reads the cache, so nothing is provisioned.
    with pytest.raises(PlanError, match=match):
        _plan(tmp_path, module=module, provisioned=False)


def test_a_credit_stage_given_less_memory_than_a_drives_summary_is_refused(tmp_path, login, in_catalog):
    """A ``[resources.<task>.credit_<k>]`` table may not take from a credit role the memory its
    drive's summary needs, one reply's line at ``REPLY_MAX_BYTES`` among it: the drive's replies
    would run it out of memory. The refusal names the drive the stage sums."""
    need = score_shape(fixture.credit_task(stream=True), RUNNER_GIB).memory_gib
    with pytest.raises(PlanError, match=rf"stage credit_2: \[resources\.{evaluate.TASK_ID}\.credit_2\] gives it "
                                        rf"{need - 1} GiB, less than the {need} GiB its role needs for the summary "
                                        "drive_2's replies can make"):
        _plan(tmp_path, module=STREAM, overrides={f"resources.{evaluate.TASK_ID}.credit_2.memory_gib": str(need - 1)})


def test_a_credits_images_stage_asks_the_cache_about_its_own_images_and_holds_disk_for_what_it_fills(
        tmp_path, login, in_catalog, monkeypatch, warmed_servers):
    """Cold, it builds its images, holding disk for them. Once the cache holds its images it
    holds no disk, and the planner asked the cache about the credit's images alone."""
    warmed_servers()
    (tmp_path / "cold").mkdir()
    (tmp_path / "warm").mkdir()
    cold = _plan(tmp_path / "cold", module=STREAM).stage("images")
    assert cold.params["found"] == [] and cold.shape.disk_gib > 0
    keys = image_keys(order(catalog.task_images(evaluate.TASK_ID).decls), ApptainerImages.recipe_key)
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
    warm = _plan(tmp_path / "warm", module=STREAM).stage("images")
    own = sorted(keys[d["name"]] for d in warm.params["images"])
    # Only the images stage asks, once for each of its own images.
    assert sorted(asked) == own == warm.params["found"] and len(own) < len(keys)
    assert warm.shape == images_shape(0, cpus=EVALUATE_IMAGES_CPUS) and warm.shape.disk_gib == 0


def test_a_credit_of_a_task_with_no_data_is_refused_when_planned(tmp_path, login, in_catalog, monkeypatch):
    """A task none of whose tenants declares a data source would drive nothing: the credit is
    refused before any image is built or any baseline served."""
    load = commands.load_task
    monkeypatch.setattr(commands, "load_task",
                        lambda ref, **kw: load(ref, **kw).model_copy(update={"data_source": None}))
    with pytest.raises(PlanError, match=f"{evaluate.TASK_ID} declares no data source on any tenant"):
        _plan(tmp_path)


@pytest.mark.parametrize("port", [8888, 8899])
def test_a_credit_of_a_task_on_an_interceptor_port_is_refused_when_planned(tmp_path, login, in_catalog, port):
    with pytest.raises(ValueError, match=f"^{evaluate.TASK_ID}'s contract port {port} is refused: the egress "
                                         f"interceptor's [a-z ]+ listens on {port} "):
        _plan(tmp_path, overrides={"task.contract.port": str(port)})


def test_task_overrides_reach_every_stage_that_loads_the_task(tmp_path, login, in_catalog):
    """A candidate measured under ``--task.*`` overrides records the load they set, which the
    graded task is then held to: the provision of its rows, the drives, their credits and the fold
    all load it so."""
    plan = _plan(tmp_path, overrides={"task.seed": "7"})
    ref = {"module": "cr_task.task", "overrides": {"task.seed": 7}}
    loading = [s for s in plan.stages if s.kind in ("provision", "deploy_drive", "credit_drive", "credit_fold")]
    assert len(loading) == 8 and all(s.params["task"] == ref for s in loading)


def test_the_cli_plans_a_credit_and_runs_it(monkeypatch):
    seen: dict = {}
    ran: list = []
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    monkeypatch.setattr(credit, "plan_credit", lambda **kw: seen.update(kw) or "the plan")
    monkeypatch.setattr(credit, "run_credit", lambda plan, overrides, **kw: ran.append((plan, overrides)) or 0)
    result = CliRunner().invoke(cli.app, ["baseline-credit", "cr_task.task", "--drives", "4", "--fanout", "2",
                                          "--site.qos", "normal"])
    assert result.exit_code == 0, result.output
    assert (seen["task_module"], seen["drives"], seen["fanout"], seen["job_name"]) == ("cr_task.task", 4, 2,
                                                                                      "hls-credit")
    assert ran == [("the plan", {"site.qos": "normal"})]


def test_the_cli_refuses_a_task_with_no_directory_in_one_line(tmp_path, login, in_catalog, monkeypatch):
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    monkeypatch.setattr(catalog, "image_set", DECLARED_IMAGES)
    result = CliRunner().invoke(cli.app, ["baseline-credit", "cr_task.task", "--drives", "3", "--config",
                                          str(write_config(tmp_path))])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
    assert result.stderr == ("hls baseline-credit: no task 'evaluate_task': a task is a directory under tasks/ "
                             f"holding task.py; there are {catalog.task_keys()}\n")


def test_the_cli_refuses_in_one_line(monkeypatch):
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)

    def refuse(**kw):
        raise PlanError("no committed tree")

    monkeypatch.setattr(credit, "plan_credit", refuse)
    result = CliRunner().invoke(cli.app, ["baseline-credit", "cr_task.task", "--drives", "4"])
    assert result.exit_code == 1
    assert "hls baseline-credit: no committed tree" in result.output


def test_the_old_measuring_path_is_gone():
    for name in ("measure_baseline_credit", "load_drives", "_units_in_trace", "check_sizing", "drive_budget",
                 "DEFAULT_DRIVES", "DEFAULT_MAX_DRIVES", "DriveRecord", "DRIVE_FILE", "TRACE_FILE"):
        assert not hasattr(baseline_credit, name), name
