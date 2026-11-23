"""``hls oracle fill`` on the login node: the plan, its refusals and the CLI's routing."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.pipeline import commands
from hls.pipeline.pipelines import (EVALUATE_IMAGES_CPUS, EVALUATE_PROVISION_SEC, KNOWN_STAGE_NAMES,
                                    ORACLE_FILL_SEC, ORACLE_FILL_STAGES, STAGING_SEC, images_shape,
                                    provision_shape, reference_shape)
from hls.pipeline.plan import Outcome, Plan
from hls.pipeline.planner import PlanError, launch_headroom_sec
from hls.provision.mounts import copied_entries, copies_gib
from hls.recipes import catalog
from hls.settings import RunSettings, SiteSettings
from hls.spec.resources import ResourceShape
from hls.taskref import TaskRef, load_task
from hls.verify.fill import DEFAULT_CHUNK
from tests.evaluate import fixture
from tests.evaluate import test_hls_grade as grade_tests
from tests.evaluate.test_hls_grade import _config, _fresh, _resources
from tests.provision.cached import provision_module, provision_task
from tests.provision.warmer_config import write_config
from tests.store.entries import hold_measured

#: ASR planned from this repository, its measured images in the cache (``hls grade``'s tests).
repo_login, asr_measured = grade_tests.repo_login, grade_tests.asr_measured
LAYER = catalog.image_set("sample_task").layer
STAGES = ["provision", "images", "oracle_fill"]
#: The memory the test site leaves to the runner (``run.runner_memory_gib``'s default).
RUNNER_GIB = 4


def _plan(tmp_path, *, module: str = "ev_cached.task", config=None, overrides=None, chunk: int = DEFAULT_CHUNK,
          provisioned: bool = True):
    """The plan of a fill of *module*'s task, with its provisioning in the cache first unless
    *provisioned* is False."""
    if provisioned:
        provision_task(tmp_path / "store", module, overrides)
    return commands.plan_oracle_fill(config=config or _config(tmp_path), overrides=overrides or {},
                                     task_module=module, chunk=chunk, job_name="hls-oracle-fill", echo=[].append)


#: Seconds a site gives each role's step to start when it sets none
#: (``site.step_launch_allowance_sec``), as these tests' sites do.
ALLOWANCE = SiteSettings.model_fields["step_launch_allowance_sec"].default


def _headroom(plan: Plan) -> int:
    """The launch headroom *plan*'s one allocation asks for beside its stages' times."""
    return launch_headroom_sec(plan.stages, ALLOWANCE)


#: GiB the copies of the fixture task's fill data take, as its provision record in the cache
#: states them (``copied_entries``, ``copies_gib``).
DATA_GIB = 1


def _fill_shape(role: ResourceShape, *, runner_gib: int = RUNNER_GIB) -> ResourceShape:
    """The fill stage's shape for a role of shape *role*: the runner's memory, and the image's
    disk and the copies of its data beside it, for ``ORACLE_FILL_SEC`` after its fixed staging."""
    return role.model_copy(update={"memory_gib": role.memory_gib + runner_gib, "disk_gib": role.disk_gib + 40 + DATA_GIB,
                                   "time_sec": ORACLE_FILL_SEC + STAGING_SEC})


def test_a_fill_is_a_gpu_warmer_on_the_verify_roles_shape(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, chunk=32)
    assert plan.instance.startswith(f"warmers/{fixture.TASK_ID}/") and plan.pipeline == "oracle_fill"
    assert "/gradings/" not in plan.instance and plan.task == fixture.TASK_ID
    [alloc] = plan.allocations
    # The test site has a CPU partition; the fill goes to the GPU one all the same.
    assert alloc.stages == STAGES == list(ORACLE_FILL_STAGES) and alloc.placement.partition is None
    assert set(ORACLE_FILL_STAGES) <= KNOWN_STAGE_NAMES
    ref = reference_shape(fixture.cached_task(), runner_memory_gib=plan.settings.run.runner_memory_gib)
    provision = provision_shape(fixture.cached_task(), runner_memory_gib=plan.settings.run.runner_memory_gib,
                                time_sec=EVALUATE_PROVISION_SEC)
    shapes = {s.name: s.shape for s in plan.stages}
    # Provision and images as an evaluation has them, on its 8 CPUs per GPU; the images stage fills
    # the two images the cache lacks, each in its fixed time.
    assert shapes == {"provision": provision, "images": images_shape(2, cpus=EVALUATE_IMAGES_CPUS),
                      "oracle_fill": _fill_shape(ref)}
    assert shapes["images"].time_sec == 900 + 2 * 1800
    # The images stage's one build and one tree are more than the fill's one tree.
    assert (alloc.request.gpus, alloc.request.gpu_kind, alloc.request.cpus, alloc.request.memory_gib,
            alloc.request.disk_gib) == (1, "B200", 8, 180, images_shape(2).disk_gib)
    assert alloc.request.time_sec == (7200 + shapes["images"].time_sec + ORACLE_FILL_SEC + STAGING_SEC + 360
                                      + _headroom(plan))
    assert {s.name: s.after for s in plan.stages} == {"provision": [], "images": [],
                                                      "oracle_fill": ["provision", "images"]}
    images = plan.stage("images").params
    assert ([d["name"] for d in images["images"]], images["catalog_task"]) == (["s-a", LAYER], fixture.TASK_ID)
    assert plan.stage("provision").params == {"task": {"module": "ev_cached.task", "overrides": {}}, "weights": True}
    # The image is named as the images stage's images.json names it.
    assert plan.stage("oracle_fill").params == {
        "task": {"module": "ev_cached.task", "overrides": {}}, "reference_image": LAYER, "images_stage": "images",
        "provision_stage": "provision", "reference_shape": ref.model_dump(),
        "data_copies": copied_entries(("oracle",), provision_module(tmp_path / "store", "ev_cached.task")), "chunk": 32,
        # The task publishes no dev reference, so the fill produces none.
        "dev_reference": False}
    assert copies_gib(plan.stage("oracle_fill").params["data_copies"]) == DATA_GIB
    idle = [w for w in plan.warnings if "idle" in w]
    assert [w.split("; stage ")[1].split(" ")[0] for w in idle] == ["provision", "images"]


#: The slowest fill: an ASR fill of 2,400 prompts, whose oracle_fill stage takes 2,207 s.
SLOWEST_MEASURED_FILL_SEC = 2207


def test_the_fills_time_is_a_margin_over_the_slowest_measured_fill():
    # At least 4x, so a slower node still finishes; not many times more, which the allocation
    # would hold a GPU for through every fill that ends in minutes.
    assert 4 * SLOWEST_MEASURED_FILL_SEC <= ORACLE_FILL_SEC <= 6 * SLOWEST_MEASURED_FILL_SEC


def test_the_default_fills_allocation_is_its_stages_and_the_runners_overhead(tmp_path, login, in_catalog):
    overhead = RunSettings.model_fields["allocation_overhead_sec"].default
    plan = _plan(tmp_path, overrides={"run.allocation_overhead_sec": str(overhead)})
    stages = sum(stage.shape.time_sec for stage in plan.stages)
    assert stages == (EVALUATE_PROVISION_SEC + images_shape(2, cpus=EVALUATE_IMAGES_CPUS).time_sec
                      + ORACLE_FILL_SEC + STAGING_SEC)
    assert plan.allocations[0].request.time_sec == stages + overhead + _headroom(plan)
    # Well inside a QOS that caps a job at one day.
    assert plan.allocations[0].request.time_sec < 24 * 3600


def test_the_fill_leaves_the_runner_the_memory_the_site_sets(tmp_path, login, in_catalog):
    plan = _plan(tmp_path, overrides={"run.runner_memory_gib": "6"})
    assert plan.stage("oracle_fill").shape == _fill_shape(reference_shape(fixture.cached_task(), runner_memory_gib=6),
                                                          runner_gib=6)


def test_a_fill_runs_on_the_measured_reference_image_when_the_catalog_lists_one(tmp_path, login, in_catalog):
    in_catalog(roles=("baseline", "reference"))
    plan = _plan(tmp_path)
    assert plan.stage("oracle_fill").params["reference_image"] == "evaluate-task-reference"
    assert [d["name"] for d in plan.stage("images").params["images"]] == ["cuda", "evaluate-task-reference"]
    # The cluster's images warmer built it, so the stage finds it and fills nothing.
    assert plan.stage("images").shape == images_shape(0, cpus=EVALUATE_IMAGES_CPUS)
    in_catalog(roles=("baseline",))
    assert _plan(tmp_path).stage("oracle_fill").params["reference_image"] == LAYER


def test_a_task_with_nothing_to_fill_is_refused_when_planned(tmp_path, login, in_catalog):
    with pytest.raises(PlanError, match="stage oracle_fill: ev_task.task: no reference caches its answers"):
        _plan(tmp_path, module="ev_task.task")


@pytest.mark.parametrize("port", [8888, 8899])
def test_a_fill_of_a_task_on_an_interceptor_port_is_refused_when_planned(tmp_path, login, in_catalog, port):
    with pytest.raises(ValueError, match=f"^{fixture.TASK_ID}'s contract port {port} is refused: the egress "
                                         f"interceptor's [a-z ]+ listens on {port} "):
        _plan(tmp_path, overrides={"task.contract.port": str(port)})


def test_a_site_sets_the_fills_time_and_no_other_resource(tmp_path, login, in_catalog):
    declared = _plan(tmp_path).stage("oracle_fill").shape
    plan = _plan(tmp_path, config=_config(tmp_path, _resources("oracle_fill", declared, time_sec=7200)))
    stage = plan.stage("oracle_fill")
    assert (stage.shape.time_sec, stage.shape_overridden) == (7200, True)
    assert plan.allocations[0].request.time_sec == (7200 + plan.stage("images").shape.time_sec + 7200 + 360
                                                    + _headroom(plan))
    wider = _resources("oracle_fill", declared, time_sec=3600, cpus=32)
    with pytest.raises(PlanError, match="stage oracle_fill: the oracle_fill stage runs one role"):
        _plan(tmp_path, config=_config(tmp_path, wider))


def test_the_fill_takes_retry_settings_and_the_gradings_task_overrides(tmp_path, login, in_catalog):
    assert _plan(tmp_path, overrides={"run.retry.oracle_fill.max": "0"}).stage("oracle_fill").retry.max == 0
    # The fill provisions and asks about the task a grading with these overrides verifies, and
    # holds the verify role of that task: here on the GPU kind the override names.
    plan = _plan(tmp_path, overrides={"task.data_source.split.test.value": "3",
                                      "task.hardware.accelerator.kind": "H200"})
    ref = {"module": "ev_cached.task", "overrides": {"task.data_source.split.test.value": 3,
                                                     "task.hardware.accelerator.kind": "H200"}}
    assert plan.stage("provision").params["task"] == ref == plan.stage("oracle_fill").params["task"]
    runner = plan.settings.run.runner_memory_gib
    held = reference_shape(load_task(TaskRef.model_validate(ref)), runner_memory_gib=runner)
    assert held.gpu_kind == "H200" != reference_shape(fixture.cached_task(), runner_memory_gib=runner).gpu_kind
    assert plan.stage("oracle_fill").params["reference_shape"] == held.model_dump()
    assert plan.allocations[0].request.gpu_kind == "H200"
    with pytest.raises(ValueError, match="not artifact.image"):
        _plan(tmp_path, overrides={"artifact.image": "x"})


def test_a_fill_records_no_score_so_one_that_did_not_succeed_exits_1(tmp_path, login, in_catalog):
    # Its one role is the harness's, so a fill never ends failed: it succeeds or not.
    plan = _plan(tmp_path)
    for status, code in (("succeeded", 0), ("void", 1), ("cancelled", 1)):
        lines: list[str] = []
        ended = plan.model_copy(update={"outcome": Outcome(status=status, reason="why", at=1.0)})
        assert commands._finish(ended, lines.append) == code
        assert lines == [f"{plan.instance}: {status}: why"]


def test_the_cli_plans_oracle_fill_and_starts_it(monkeypatch):
    seen, started = {}, []

    def plan(**kw):
        seen.update(kw)
        return "the plan"

    monkeypatch.setattr(commands, "plan_oracle_fill", plan)
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: started.append((made, overrides)) or 0)
    # The command configures hls logging for a terminal; a test leaves it as pytest set it.
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["oracle", "fill", "tasks.multimodal_omni.task", "--chunk", "48",
                                          "--site.qos", "normal"])
    assert result.exit_code == 0, result.output
    assert (seen["task_module"], seen["chunk"], seen["job_name"]) == (
        "tasks.multimodal_omni.task", 48, "hls-oracle-fill")
    assert seen["overrides"] == {"site.qos": "normal"}
    assert started == [("the plan", {"site.qos": "normal"})]
    refused = CliRunner().invoke(cli.app, ["oracle", "fill", "tasks.multimodal_omni.task", "--chunk", "0"])
    assert refused.exit_code != 0 and "chunk" in refused.output
    first = CliRunner().invoke(cli.app, ["oracle", "fill", "--site.qos=normal"])
    assert first.exit_code == 1 and "expected the task module first" in first.output
    assert len(started) == 1


def test_the_cli_refuses_what_cannot_be_planned_in_one_line(monkeypatch):
    def plan(**kw):
        raise PlanError("stage oracle_fill: tasks.x.task: no reference caches its answers")

    monkeypatch.setattr(commands, "plan_oracle_fill", plan)
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["oracle", "fill", "tasks.x.task"])
    assert result.exit_code == 1
    assert result.output.strip().splitlines() == [
        "hls oracle fill: stage oracle_fill: tasks.x.task: no reference caches its answers"]


def test_a_task_whose_id_is_not_its_directory_is_filled_under_its_catalog_name(tmp_path, asr_measured):
    """ASR's TaskSpec is ``multimodal-asr`` and its directory, the catalog's name for it,
    ``multimodal_asr``: the fill runs on the reference image the catalog records for the
    directory, and its warmer and answers belong to the id."""
    plan = _plan(tmp_path, module="tasks.multimodal_asr.task")
    assert plan.task == "multimodal-asr" and plan.instance.startswith("warmers/multimodal-asr/")
    assert plan.stage("oracle_fill").params["reference_image"] == "multimodal-asr-reference"
    images = plan.stage("images").params
    # The reference image is a server layer of the task's own, on the cuda base.
    assert ([d["name"] for d in images["images"]], images["catalog_task"]) == (
        ["cuda", "multimodal-asr-reference"], "multimodal_asr")
    with pytest.raises(ValueError, match="is not a task module: a task module is tasks.<directory>.task"):
        _plan(tmp_path, module="multimodal_asr.task", provisioned=False)


FILL_PROBE = """
import json, sys
from pathlib import Path
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from hls.recipes import catalog
from tests.pipeline.plans import CODE
commands.code_state = lambda path: CODE
commands.toolchain_here = lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/"))
plan = commands.plan_oracle_fill(config=Path(sys.argv[1]), overrides={}, task_module="tasks.multimodal_asr.task",
                                 chunk=96, job_name="x", echo=lambda line: None)
heavy = sorted(m for m in ("torch", "transformers", "vllm", "datasets", "huggingface_hub", "pyarrow")
               if m in sys.modules)
print(json.dumps({"heavy": heavy, "stages": [s.name for s in plan.stages]}))
"""


def test_planning_a_fill_on_the_login_node_loads_nothing_heavy(tmp_path):
    # ASR's verifier caches its Whisper oracle's answers; the plan builds the engine to size the
    # role and to check it, and loads no model, dataset or hub library. The cluster's provision
    # and images warmers ran: the task's data is provisioned and its reference image held.
    provision_module(tmp_path / "store", "tasks.multimodal_asr.task")
    hold_measured(tmp_path / "store", "multimodal_asr")
    out = json.loads(_fresh(FILL_PROBE, str(write_config(tmp_path))))
    assert out == {"heavy": [], "stages": STAGES}


def test_importing_the_registry_registers_the_fill():
    # What the runner imports to find a stage by its kind.
    code = ("import json, hls.stages.registry\nfrom hls.pipeline.stage import STAGES\n"
            "print(json.dumps(sorted(STAGES)))")
    assert "oracle_fill" in json.loads(_fresh(code))


@pytest.mark.parametrize("module", ["hls.verify.fill", "hls.stages.oracle_fill"])
def test_the_fill_role_and_its_stage_import_first_without_a_cycle(module):
    assert _fresh(f"import {module}\nprint('imported')") == "imported"
