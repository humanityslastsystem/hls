"""``hls run``'s planning: the build and its grading in one rollout and one allocation, the
build-only and stop forms, the checks a scored run passes before anything is submitted, and how
the command's overrides are routed. The fixtures are the agent build tests' (``conftest.py``,
``build_fixture.py``): the evaluate tests' task with an agent-produced artifact, in a checkout of
the tests' own."""

from __future__ import annotations

import importlib
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.artifact.artifact import EvaluationAllocation
from hls.drive.bounds import UnknownTask, drive_bounds
from hls.measured import BASELINE_CREDIT_FILE, Found, MeasuredFile, MeasuredFiles
from hls.pipeline import commands, pipelines, preflight, prepare
from hls.pipeline.grown import grown_stages
from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.pipelines import (DRIVE_MARGIN_SEC, FINALIZE_SHAPE, KNOWN_STAGE_NAMES, ROLLOUT_STAGES,
                                    STAGE_NAME_PATTERNS, STAGING_SEC, TRACE_COPY_SEC, reference_shape,
                                    rollout_stages, score_shape)
from hls.pipeline.plan import Plan, StagePlan, Toolchain
from hls.pipeline.planner import PlanError, launch_headroom_sec
from hls.provision.mounts import copies_gib
from hls.recipes import catalog
from hls.recipes.model import CONSTRAINTS, order
from hls.settings import RunSettings
from hls.stages.agent_build import AgentBuildParams
from hls.stages.deploy_drive import SEAL_PROBE_SEC
from hls.stages.finalize import FinalizeParams
from hls.stages.images import image_keys
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, load_task
from hls.substrate.slurm.scheduler import submit_argv
from hls.recipes.render import recipe_key as def_key
from hls.substrate.apptainer.images import ApptainerImages
from tests.agent import build_fixture
from tests.agent.build_fixture import KEY, KEY_NAME, TASK_ID, build_config
from tests.agent.test_hls_run_build import _text
from tests.pipeline.plans import CODE, pin_harbor
from tests.provision.cached import provision_task

LAYER = catalog.image_set("sample_task").layer
CAP = catalog.AGENT_CAPS[LAYER].name
#: The trace cap and request bytes the build task declares for its drive.
TRACE_CAP, REQUESTS = (drive_bounds(build_fixture.build_task()).trace_cap_bytes,
                       drive_bounds(build_fixture.build_task()).trace_request_bytes)


def _plan(tmp_path: Path, *, module: str = "bd_task.task", config: Path | None = None,
          overrides: dict[str, str] | None = None, echo=None, **kw) -> Plan:
    """``hls run``'s plan of *module*, the task's provisioning in the cache first."""
    provision_task(tmp_path / "store", module, overrides)
    return commands.plan_run(config=config or build_config(tmp_path), overrides=overrides or {}, task_module=module,
                             arm="arm-a", arm_change=None, job_name="hls-run",
                             echo=echo if echo is not None else [].append, **kw)


def _sbatch(tmp_path: Path, plan: Plan, k: int = 0) -> tuple[str, ...]:
    """The sbatch line of allocation *k*'s first submission, from the request the orchestrator makes."""
    here = SimpleNamespace(store=PosixStore(tmp_path / "store"), layout=plan.layout)
    return submit_argv(Orchestrator._request(here, plan, k, 1, script=Path("/b"), tag="t", after=None))


def _stage_time(plan: Plan, stages: Sequence[StagePlan] | None = None) -> int:
    """The time an allocation of *plan* that runs *stages*, every stage of the plan by default,
    asks for: each stage's in turn, the runner's overhead and the headroom for each role's step to
    start."""
    stages = plan.stages if stages is None else stages
    return (sum(s.shape.time_sec for s in stages) + plan.settings.run.allocation_overhead_sec
            + launch_headroom_sec(stages, plan.settings.site.step_launch_allowance_sec))


def test_a_run_plans_build_and_grading_in_one_allocation(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path)
    assert [s.name for s in plan.stages] == list(ROLLOUT_STAGES)
    assert set(ROLLOUT_STAGES) <= KNOWN_STAGE_NAMES
    assert plan.pipeline == "rollout" and plan.task == TASK_ID
    assert plan.instance.startswith(f"runs/{TASK_ID}/") and plan.instance.count("/") == 2
    [alloc] = plan.allocations
    assert alloc.stages == list(ROLLOUT_STAGES) and alloc.placement.partition is None
    # One images stage holds what both halves run on: the agent cap and the submission base for
    # the build, the harness and reference images (the layer, for this task) for the grading.
    assert [d["name"] for d in plan.stage("images").params["images"]] == ["s-a", LAYER, CAP]
    drive = plan.stage("deploy_drive")
    assert drive.params["artifact"] == {"kind": "package", "stage": "package", "manifest": None}
    assert drive.after == ["provision", "images", "package"] and drive.inputs == {}
    # The image the drive serves is built in this instance, on the submission base its images stage
    # builds too. The drive's staging is the fixed allowance for its copies and extractions and its
    # seal probe, and after the wall it keeps its margin and the trace's copy to the store.
    images = catalog.task_images(TASK_ID)
    assert images.harness == images.submission_base == LAYER
    layer = image_keys(order(images.decls), ApptainerImages.recipe_key)[LAYER]
    assert layer not in plan.stage("images").params["found"]
    assert drive.params["staging_sec"] == STAGING_SEC + math.ceil(SEAL_PROBE_SEC)
    assert drive.params["wall_margin_sec"] == DRIVE_MARGIN_SEC + TRACE_COPY_SEC
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert (final.kind, final.eval_id, final.score_stage, final.build_stage, final.package_stage) == (
        "rollout", "terminal", "score", "agent_build", "package")
    # A rollout's arm is its own build's; its agent_build stage hands its grants to the agent.
    assert final.built_arm is None
    assert plan.stage("finalize").after == ["agent_build", "package", "deploy_drive", "verify", "score"]
    # The build is granted its arm's route by name; no plan holds a value.
    assert KEY_NAME in plan.stage("agent_build").grants.values()
    assert KEY not in plan.model_dump_json()
    # The allocation asks for the largest of each resource and every stage's time in turn, the
    # runner's overhead and the headroom for each role's step to start.
    r = alloc.request
    assert r.memory_gib == max(s.shape.memory_gib for s in plan.stages)
    assert r.cpus == max(s.shape.cpus for s in plan.stages)
    assert r.time_sec == _stage_time(plan)
    # Finalize has its fixed hour, whatever the stages before it leave.
    assert plan.stage("finalize").shape == FINALIZE_SHAPE and FINALIZE_SHAPE.time_sec == 3600
    # The drive holds its trace to the cap and request bytes the task declares.
    assert (drive.params["trace_cap_bytes"], drive.params["trace_request_bytes"]) == (TRACE_CAP, REQUESTS)


def test_a_runs_cpu_stages_take_no_more_cpus_than_its_build_allocation(tmp_path, build_login, build_catalog):
    """A producer that declares a build allocation (3 CPUs here) caps the run's
    CPU stages at it, the images stage that holds both halves' images among them; the grading's
    stages keep their own shapes, and the job asks for the most any stage needs."""
    plan = _plan(tmp_path, module="bd_alloc.task")
    cpus = {s.name: s.shape.cpus for s in plan.stages}
    assert cpus["provision"] <= 3 and cpus["images"] <= 3 and cpus["source_scan"] <= 3 and cpus["package"] <= 3
    assert cpus["agent_build"] == 3
    assert plan.allocations[0].request.cpus == max(cpus.values())


def test_a_task_declaring_a_longer_verify_time_gets_it_in_the_plan(tmp_path, build_login, build_catalog,
                                                                    monkeypatch):
    """The verify role takes the verify time its task declares (TTS's reference takes hours
    cold) and its stage that after its staging, and the allocation asks for it with every other
    stage's time."""
    default = _plan(tmp_path).stage("verify").params["reference_shape"]["time_sec"]
    module = importlib.import_module("bd_task.task")
    allocation = EvaluationAllocation(cpus=2, job_memory_gib=20, writable_gib=2, driver_memory_gib=1,
                                      verify_time_sec=8 * 3600)
    producer = module.ARTIFACT.producer.model_copy(update={"evaluation_allocation": allocation})
    monkeypatch.setattr(module, "ARTIFACT", module.ARTIFACT.model_copy(update={"producer": producer}))
    plan = _plan(tmp_path)
    verify = plan.stage("verify")
    assert default < 8 * 3600 == verify.params["reference_shape"]["time_sec"]
    assert verify.shape.time_sec == 8 * 3600 + STAGING_SEC
    assert plan.allocations[0].request.time_sec == _stage_time(plan)


def test_a_graded_run_collects_its_build_for_the_runs_collection_allowance(tmp_path, build_login, build_catalog):
    """A run that grades its build gives the build the collection ``run.collect_allowance_sec``
    sets, as a build-only run does: its parameters record it and its stage's time grows by it,
    while the grading's stages keep their times."""
    default = _plan(tmp_path)
    plan = _plan(tmp_path, overrides={"run.collect_allowance_sec": "5000"})
    params = AgentBuildParams.model_validate(plan.stage("agent_build").params)
    assert AgentBuildParams.model_validate(default.stage("agent_build").params).collect_allowance_sec == 3600
    assert (params.collect_allowance_sec, params.collection_sec) == (5000, 5000)
    times, default_times = ({s.name: s.shape.time_sec for s in p.stages} for p in (plan, default))
    assert times["agent_build"] - default_times["agent_build"] == 5000 - 3600
    assert {k: v for k, v in times.items() if k != "agent_build"} == {
        k: v for k, v in default_times.items() if k != "agent_build"}


@pytest.fixture
def repo_login(monkeypatch, tmp_path, warmed_servers):
    """The login node planning from this repository, whose ``tasks/`` holds the real tasks, on a
    cluster whose images warmers built every server image (``warmed_servers``). The blob,
    oracle and upload-folder checks have tests of their own, so here they find nothing to load:
    the folder a task uploads is not committed, so no checkout of the tests holds it."""
    monkeypatch.setattr(commands, "code_state", lambda path: CODE)
    pin_harbor(monkeypatch, CODE)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    monkeypatch.setattr(commands, "check_defaults_committed", lambda checkout, files: None)
    monkeypatch.setattr(commands, "check_oracle_coverage", lambda store, **kw: None)
    monkeypatch.setattr(commands, "check_upload_folder", lambda checkout, task_module, producer: None)


def test_a_runs_score_stage_holds_one_replys_summary_and_a_site_may_not_give_it_less(tmp_path, repo_login):
    """The rollout grades its build with the score memory a grading has (``score_shape``), and a
    site's ``[resources]`` table that gives the role less is refused, as for a grading. For
    multitenant_slo's token ids, 2 bytes a packet, that is the role's 8 GiB and 13 more for the
    134,217,728 packets of one reply at ``REPLY_MAX_BYTES`` (``score_memory_gib``), and the
    runner's 4."""
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task")
    task = load_task(TaskRef(module="tasks.multitenant_slo.task"))
    assert plan.stage("score").shape == score_shape(task, plan.settings.run.runner_memory_gib)
    assert plan.stage("score").shape.memory_gib == 8 + 13 + 4
    with pytest.raises(PlanError, match=r"stage score: \[resources.multitenant_slo.score\] gives it 12 GiB, less than "
                                        "the 25 GiB"):
        _plan(tmp_path, module="tasks.multitenant_slo.task",
              overrides={"resources.multitenant_slo.score.memory_gib": "12"})
    # A build-only run scores nothing, so its plan holds no score stage to size.
    assert "score" not in {s.name for s in _plan(tmp_path, module="tasks.multitenant_slo.task",
                                                 build_only=True).stages}


def test_a_run_that_grades_its_build_builds_the_reference_it_verifies_on(tmp_path, repo_login):
    # A run's images stage builds the task's reference image, which its verify stage runs on.
    assert "multitenant-slo-reference" in {d["name"] for d in _plan(
        tmp_path, module="tasks.multitenant_slo.task").stage("images").params["images"]}


#: Bytes of the node-local copy of each of deepresearch's entries the cache's provision record
#: states in the test below: round figures for the model's weights and the corpus rows, and the
#: retriever's two bundles (18 GB full, 16 GB dev).
DR_COPIES = {"weights": 60 * 2**30, "rows": 2 * 2**30, "full": 18 * 10**9, "dev": 16 * 10**9}


def test_a_deepresearch_run_holds_every_entry_its_stages_copy_once(tmp_path, repo_login, monkeypatch):
    """A rollout's stages run in one allocation, and a node-local copy lasts the job: a later
    stage that mounts an entry an earlier one copied reuses its copy (``copies_root``). The build
    copies the weights, the rows and the retriever's dev bundle, the drive the weights, the rows
    and the full bundle, and the verify the weights and the full bundle, each as its parameters
    declare. While the drive runs the build's copies are still on the node, so the allocation holds
    the drive's other disk and the four entries once each: the largest stage's disk alone, the
    drive's, leaves the dev bundle out."""
    planned, seen = commands.planned_provision, []

    def sized(*args, **kw):
        record = planned(*args, **kw)
        record = record.model_copy(update={
            "weights": record.weights.model_copy(update={"disk_bytes": DR_COPIES["weights"]}),
            "rows": record.rows.model_copy(update={"disk_bytes": DR_COPIES["rows"]}),
            "tools": [t.model_copy(update={"disk_bytes": DR_COPIES[t.variant]}) for t in record.tools]})
        seen.append(record)
        return record

    monkeypatch.setattr(commands, "planned_provision", sized)
    plan = _plan(tmp_path, module="tasks.deepresearch.task")
    [alloc] = plan.allocations
    drive = plan.stage("deploy_drive")
    assert max(plan.stages, key=lambda s: s.shape.disk_gib) == drive
    # The dev bundle's 16 GB, which only the build copies, add 15 GiB to the copies the drive holds
    # (94 GiB of copies in all, against the drive's 79).
    assert alloc.request.disk_gib == drive.shape.disk_gib + 15
    [record] = seen
    weights, rows = f"weights/{record.weights.key}", f"rows/{record.rows.key}"
    full, dev = ({f"tools/{t.key}" for t in record.tools if t.variant == v} for v in ("full", "dev"))
    [full], [dev] = full, dev
    sizes = {weights: DR_COPIES["weights"], rows: DR_COPIES["rows"], full: DR_COPIES["full"], dev: DR_COPIES["dev"]}
    declared = {s.name: s.params["data_copies"] for s in plan.stages if "data_copies" in s.params}
    assert declared == {"agent_build": {e: sizes[e] for e in (weights, rows, dev)},
                        "deploy_drive": {e: sizes[e] for e in (weights, rows, full)},
                        "verify": {e: sizes[e] for e in (weights, full)}}
    assert (copies_gib(sizes), copies_gib(declared["deploy_drive"])) == (94, 79)


@pytest.mark.parametrize("module, constraints", [("tasks.deepresearch.task", None),
                                                 ("tasks.multitenant_slo.task", CONSTRAINTS)],
                         ids=["published-base", "built-base"])
def test_a_run_packages_under_the_constraints_file_its_submission_base_holds(tmp_path, repo_login, module,
                                                                           constraints):
    """deepresearch packages on its published sandbox, which holds no constraints file, so its
    offline install names none; a pip install that names a missing file fails before it reads a
    wheel. A task whose submission base hls builds installs under the file every base writes."""
    plan = _plan(tmp_path, module=module)
    assert plan.stage("package").params["constraints"] == constraints


def test_a_run_of_a_twelve_hour_task_is_planned_under_a_long_qos(tmp_path, repo_login):
    site = 'qos = "normal"\nqos_by_hours = { normal = 24, long = 168 }'
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task", config=build_config(tmp_path, site=site))
    [alloc] = plan.allocations
    r = alloc.request
    # One H200 for the build and the grading together, and more than a day of stage time in all:
    # the agent's 12.5 h wall, the package's 7 h, the drive and verify.
    assert (r.gpus, r.gpu_kind) == (1, "H200")
    assert r.time_sec > 24 * 3600 and alloc.placement.qos == "long"
    assert "--qos=long" in _sbatch(tmp_path, plan)


def test_a_run_past_a_day_is_split_into_allocations_that_each_fit_a_day(tmp_path, repo_login):
    """On a site whose every QOS holds a day at most, a 3 h build of multitenant_slo (soft
    10,800 s, hard 12,600 s), packaged and graded, would ask for more than a day in one allocation.
    It runs as a chain of allocations instead, each taking as many stages in turn as a day holds."""
    site = 'qos = "normal"\nqos_by_hours = { normal = 24 }'
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task", config=build_config(tmp_path, site=site),
                 soft_sec=10_800, hard_sec=12_600)
    day = 24 * 3600
    assert _stage_time(plan) > day
    assert [a.stages for a in plan.allocations] == [["provision", "images", "agent_build", "source_scan", "package"],
                                                    ["deploy_drive", "verify", "score", "finalize"]]
    for k, alloc in enumerate(plan.allocations):
        members = [plan.stage(name) for name in alloc.stages]
        assert (alloc.depends_on, alloc.waits_for, alloc.placement.qos) == (k - 1 if k else None, [], "normal")
        assert alloc.request.time_sec == _stage_time(plan, members) <= day
        assert "--qos=normal" in _sbatch(tmp_path, plan, k)
        if k + 1 < len(plan.allocations):
            assert _stage_time(plan, [*members, plan.stage(plan.allocations[k + 1].stages[0])]) > day
        # Each stage's disk holds the copies of the task's data it makes, so an allocation of some
        # of the run's stages is sized right without the others' copies.
        for s in members:
            assert s.shape.disk_gib >= copies_gib(s.params.get("data_copies", {}))
    # The first allocation runs the package while the build's copies are still on the node, so it
    # asks for the package's disk and those copies; the second asks for its largest stage's disk.
    build, package = plan.stage("agent_build"), plan.stage("package")
    assert [a.request.disk_gib for a in plan.allocations] == [
        package.shape.disk_gib + copies_gib(build.params["data_copies"]),
        max(plan.stage(name).shape.disk_gib for name in plan.allocations[1].stages)]
    assert plan.warnings[0].startswith("the max GPU policy groups stages provision to finalize into one allocation, "
                                       "but it asks for ")
    assert plan.warnings[0].endswith("the longest, normal, allows 24 h; they run as allocations 0 to 1 instead, as "
                                     "many stages each as that limit holds, each queued after the one before")


def test_a_run_that_grades_against_a_baseline_is_planned_on_any_site(tmp_path, repo_login):
    """No site is checked against the baseline credit's when a run that
    grades is planned; its records state the hardware it ran on and compare it with the
    credit's."""
    plan = _plan(tmp_path, module="tasks.multitenant_slo.task", config=build_config(tmp_path))
    assert not {"baseline_site", "site_waiver"} & plan.stage("finalize").params.keys()


def test_a_build_only_run_ends_with_package_and_a_recording_finalize(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path, build_only=True)
    assert [s.name for s in plan.stages] == ["provision", "images", "agent_build", "source_scan", "package",
                                             "finalize"]
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert (final.kind, final.eval_id, final.score_stage, final.build_stage, final.package_stage) == (
        "rollout", "terminal", None, "agent_build", "package")
    assert plan.stage("finalize").after == ["agent_build", "package"]
    # Nothing is driven, and finalize has the same fixed hour as a graded run's.
    assert plan.stage("finalize").shape == FINALIZE_SHAPE


@pytest.mark.parametrize("stop", ["probe", "preflight"])
def test_a_probe_only_run_has_no_package_and_no_finalize(tmp_path, build_login, build_catalog, stop):
    plan = _plan(tmp_path, stop_after=stop)
    assert [s.name for s in plan.stages] == ["provision", "images", "agent_build"]
    assert AgentBuildParams.model_validate(plan.stage("agent_build").params).stop_after == stop


def test_a_build_only_run_cannot_also_stop(tmp_path, build_login, build_catalog):
    with pytest.raises(ValueError, match="cannot also be build-only"):
        _plan(tmp_path, build_only=True, stop_after="probe")


def test_a_run_refuses_a_dirty_checkout(tmp_path, build_login, build_catalog, monkeypatch):
    dirty = build_login.model_copy(update={"dirty_diff_sha256": "e" * 64})
    monkeypatch.setattr(commands, "code_state", lambda path: dirty)
    for kw in ({}, {"build_only": True}):
        with pytest.raises(PlanError, match=str(dirty.checkout)):
            _plan(tmp_path, **kw)


def test_a_probe_only_run_from_a_dirty_checkout_is_planned(tmp_path, build_login, build_catalog, monkeypatch):
    dirty = build_login.model_copy(update={"dirty_diff_sha256": "e" * 64})
    monkeypatch.setattr(commands, "code_state", lambda path: dirty)
    plan = _plan(tmp_path, stop_after="probe")
    assert plan.code.dirty_diff_sha256 == "e" * 64


def test_a_run_refuses_an_uncommitted_default(tmp_path, build_login, build_catalog, monkeypatch):
    seen = []

    def refuse(checkout, files):
        seen.append((checkout, files))
        raise PlanError("bd_task/defaults/baseline.json is not committed at HEAD")

    monkeypatch.setattr(commands, "check_defaults_committed", refuse)
    for kw in ({}, {"build_only": True}):
        with pytest.raises(PlanError, match="baseline.json is not committed at HEAD"):
            _plan(tmp_path, **kw)
    assert seen == [(commands.checkout_root(), MeasuredFiles())] * 2
    # A stop build is not scored: it runs from any checkout.
    _plan(tmp_path, stop_after="probe")
    assert len(seen) == 2


def test_a_run_pins_and_prints_the_measured_files_it_loads(tmp_path, build_login, build_catalog, monkeypatch):
    pin = MeasuredFile(source="defaults", path="bd_task/defaults/baseline.json", blob="f" * 40)
    monkeypatch.setattr(commands, "find_measured",
                        lambda store, **kw: [Found(name=BASELINE_CREDIT_FILE, pin=pin)])
    monkeypatch.setattr(commands, "check_defaults_committed", lambda checkout, files: None)
    lines: list[str] = []
    plan = _plan(tmp_path, echo=lines.append)
    params = AgentBuildParams.model_validate(plan.stage("agent_build").params)
    expected = preflight.arm_lines(params.arm, soft_sec=params.soft_sec, hard_sec=params.hard_sec, arm_change=None)
    assert expected and all(line in lines for line in expected)
    assert plan.measured == MeasuredFiles(credit=pin)
    assert f"baseline credit: the task's defaults bd_task/defaults/baseline.json (blob {'f' * 12})" in lines
    assert not any(KEY in line for line in lines)


def test_a_run_that_grades_refuses_a_task_whose_oracle_answers_are_missing(tmp_path, build_login, build_catalog,
                                                                          monkeypatch):
    """A run that grades its build checks the oracle answers its verify reads before anything is
    submitted, for the task with the run's overrides; a build-only run and a stop build verify
    nothing and are planned."""
    asked = []

    def refuse(store, **kw):
        asked.append(kw)
        raise PlanError("evaluate_task: Run hls oracle fill bd_task.task first")

    monkeypatch.setattr(commands, "check_oracle_coverage", refuse)
    with pytest.raises(PlanError, match=r"Run hls oracle fill bd_task\.task first"):
        _plan(tmp_path, overrides={"task.max_context_len": "4096"})
    [kw] = asked
    assert kw["task_ref"] == TaskRef(module="bd_task.task", overrides={"task.max_context_len": 4096})
    assert kw["task"].max_context_len == 4096 and kw["task_id"] == TASK_ID
    _plan(tmp_path, build_only=True)
    _plan(tmp_path, stop_after="probe")
    assert len(asked) == 1


def test_the_cli_refuses_a_dirty_checkout_in_one_line_before_submitting(tmp_path, build_login, build_catalog,
                                                                        monkeypatch):
    dirty = build_login.model_copy(update={"dirty_diff_sha256": "e" * 64})
    monkeypatch.setattr(commands, "code_state", lambda path: dirty)
    monkeypatch.setattr(commands, "start", lambda *args, **kw: pytest.fail("the plan was submitted"))
    result = CliRunner().invoke(cli.app, ["run", "bd_task.task", "--arm", "arm-a", "--config",
                                          str(build_config(tmp_path))])
    assert result.exit_code == 1
    [line] = result.output.strip().splitlines()
    assert line.startswith(f"hls run: the checkout {dirty.checkout} or its harbor submodule has uncommitted changes")


def test_every_run_checks_the_folder_it_uploads_and_prints_it(tmp_path, build_login, build_catalog, monkeypatch):
    """Every build uploads the task's folder into its sandbox, a stop build's too, so every run
    checks it against the digest the task pins before anything is submitted."""
    seen = []

    def check(checkout, task_module, producer):
        seen.append((checkout, task_module))
        return preflight.UploadFolder(paths=("tasks/assets/references",), sha256="d" * 64)

    monkeypatch.setattr(commands, "check_upload_folder", check)
    lines: list[str] = []
    _plan(tmp_path, echo=lines.append)
    assert f"uploads tasks/assets/references {'d' * 64}" in lines
    assert seen == [(commands.checkout_root(), "bd_task.task")]

    def refuse(checkout, task_module, producer):
        raise PlanError("bd_task.task uploads its folder, and its tree digest is not the one the task pins")

    monkeypatch.setattr(commands, "check_upload_folder", refuse)
    for kw in ({}, {"build_only": True}, {"stop_after": "probe"}):
        with pytest.raises(PlanError, match="its tree digest is not the one the task pins"):
            _plan(tmp_path, **kw)


def test_every_run_checks_the_skills_it_copies_and_prints_them(tmp_path, build_login, build_catalog, monkeypatch):
    """Every build copies the task's skills into its sandbox, a stop build's too, so every run
    checks the folder the task names before anything is submitted."""
    seen = []

    def check(checkout, task_module, producer):
        seen.append((checkout, task_module))
        return ["cuda", "nsight"]

    monkeypatch.setattr(commands, "check_skills_folder", check)
    lines: list[str] = []
    _plan(tmp_path, echo=lines.append)
    assert "skills cuda, nsight" in lines
    assert seen == [(commands.checkout_root(), "bd_task.task")]

    def refuse(checkout, task_module, producer):
        raise PlanError("bd_task.task names skills as its skills (producer.skills_dir), and broken holds no SKILL.md")

    monkeypatch.setattr(commands, "check_skills_folder", refuse)
    for kw in ({}, {"build_only": True}, {"stop_after": "probe"}):
        with pytest.raises(PlanError, match="broken holds no SKILL.md"):
            _plan(tmp_path, **kw)


def test_a_run_of_a_task_with_no_data_is_refused_when_planned(tmp_path, build_login, build_catalog, monkeypatch):
    """A task none of whose tenants declares a data source would drive nothing, and verify would
    fail it for zero items only after the agent had built: the run is refused when planned. A
    stop build drives nothing and is planned."""
    load = commands.load_task
    monkeypatch.setattr(commands, "load_task",
                        lambda ref, **kw: load(ref, **kw).model_copy(update={"data_source": None}))
    for kw in ({}, {"build_only": True}):
        with pytest.raises(PlanError, match=f"{TASK_ID} declares no data source on any tenant"):
            _plan(tmp_path, **kw)
    _plan(tmp_path, stop_after="probe")


@pytest.mark.parametrize("port", [8888, 8899])
def test_a_run_of_a_task_on_an_interceptor_port_is_refused_when_planned(tmp_path, build_login, build_catalog, port):
    """The build's sandbox and the drive's server both share the interceptor's network namespace,
    so a run of such a task is refused before anything is built, a stop build's too."""
    for kw in ({}, {"build_only": True}, {"stop_after": "probe"}):
        with pytest.raises(ValueError, match=f"^{TASK_ID}'s contract port {port} is refused: the egress "
                                             f"interceptor's [a-z ]+ listens on {port} "):
            _plan(tmp_path, overrides={"task.contract.port": str(port)}, **kw)


def test_task_overrides_go_to_the_task_ref_and_others_are_refused(tmp_path, build_login, build_catalog):
    plan = _plan(tmp_path, overrides={"task.max_context_len": "4096", "site.qos": "normal"})
    refs = {s.name: s.params["task"] for s in plan.stages if "task" in s.params and isinstance(s.params["task"], dict)}
    assert set(refs) >= {"provision", "agent_build", "deploy_drive", "verify", "score", "finalize"}
    assert all(ref == {"module": "bd_task.task", "overrides": {"task.max_context_len": 4096}}
               for ref in refs.values())
    assert plan.settings.site.qos == "normal"
    for bad in ("artifact.image", "images.build", "substrate.apptainer.x", "env.X"):
        with pytest.raises(ValueError, match=f"takes --site.\\*, --run.\\*, --resources.\\* and --task.\\* overrides, "
                                             f"not {bad}"):
            _plan(tmp_path, overrides={bad: "1"})


def test_a_single_field_resources_override_reaches_the_plan(tmp_path, build_login, build_catalog):
    declared = _plan(tmp_path).stage("verify").shape
    plan = _plan(tmp_path, overrides={f"resources.{TASK_ID}.verify.time_sec": "9000"})
    verify = plan.stage("verify")
    assert verify.shape == declared.model_copy(update={"time_sec": 9000}) and verify.shape_overridden


def test_the_known_names_cover_every_pipeline():
    assert {"agent_build", "package", "deploy_drive", "verify", "score", "finalize", "fold", "instrument",
            "warm_vllm", "warm_sglang", "export", "oracle_fill"} <= KNOWN_STAGE_NAMES
    assert STAGE_NAME_PATTERNS == (r"drive_[1-9][0-9]*", r"credit_[1-9][0-9]*")
    # A calibration's captures, score shares and aggregate are what its grower declares it adds.
    grown = grown_stages("calibration")
    assert all(grown.holds(name) for name in ("aggregate", "capture_vllm_0", "capture_sglang_12", "score_1"))
    assert not any(grown.holds(name) for name in ("capture_vllm_01", "score_0", "capture_trt_0"))


def test_the_verify_role_holds_every_engine_one_verifier_loads_at_once(monkeypatch):
    """A verifier's engines are live together, each on GPUs of its own: the role takes the most
    GPUs one verifier holds and the memory of its engines together, the most over verifiers."""
    engines = [("PairVerifier", [SimpleNamespace(gpu_count=1, memory_mb=200 * 1024),
                                 SimpleNamespace(gpu_count=1, memory_mb=100 * 1024)]),
               ("GateVerifier", [SimpleNamespace(gpu_count=1, memory_mb=250 * 1024)])]
    monkeypatch.setattr(pipelines, "reference_engines", lambda task: engines)
    monkeypatch.setattr(pipelines, "verifier_gpus", lambda task: 2)
    shape = reference_shape(build_fixture.build_task(), runner_memory_gib=RunSettings().runner_memory_gib)
    assert (shape.gpus, shape.gpu_kind, shape.memory_gib) == (2, "B200", 300)


def test_a_run_that_grades_a_task_without_drive_bounds_is_refused(tmp_path, build_login, build_catalog,
                                                                  monkeypatch):
    """A run that grades its build drives it, and a drive needs the task's drive bounds: the
    command refuses a task that declares none before anything is planned, and ``rollout_stages``
    refuses it too, through its drive stage (``UnknownTask``). A build-only run drives nothing, so
    it is planned."""
    arm = AgentBuildParams.model_validate(_plan(tmp_path, build_only=True).stage("agent_build").params).arm
    unbounded = build_fixture.build_task().model_copy(update={"drive_bounds": None})
    common = {"task_ref": TaskRef(module="bd_task.task"), "task": unbounded,
              "producer": build_fixture.artifact().producer, "serve_command": [], "arm": arm, "arm_change": None,
              "catalog_key": TASK_ID, "run": RunSettings(), "seal_memory_gib": 0, "allocation": None,
              "held": lambda keys: set(), "recipe_key": def_key,
              "provisioned": provision_task(tmp_path / "store", "bd_task.task")}
    with pytest.raises(UnknownTask, match=f"no drive bounds for task '{TASK_ID}'"):
        rollout_stages(**common)
    assert "finalize" in {d.name for d in rollout_stages(**common, build_only=True)}
    load = commands.load_task
    monkeypatch.setattr(commands, "load_task",
                        lambda ref, **kw: load(ref, **kw).model_copy(update={"drive_bounds": None}))
    with pytest.raises(PlanError, match=f"{TASK_ID} cannot be graded: no drive bounds for task '{TASK_ID}'"):
        _plan(tmp_path)
    assert "finalize" in {s.name for s in _plan(tmp_path, build_only=True).stages}


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def test_the_cli_plans_a_run_with_its_flags_and_starts_it(monkeypatch):
    seen, started, prepared = {}, [], []

    def plan(**kw):
        seen.update(kw)
        return "the plan"

    monkeypatch.setattr(commands, "plan_run", plan)
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: started.append((made, overrides)) or 0)
    monkeypatch.setattr(prepare, "prepare", lambda **kw: prepared.append(kw) or 0)
    result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a", "--build-only",
                                          "--site.qos", "normal", "--task.max_context_len", "4096"])
    assert result.exit_code == 0, result.output
    assert (seen["task_module"], seen["arm"], seen["build_only"], seen["stop_after"], seen["job_name"]) == (
        "tasks.multitenant_slo.task", "arm-a", True, None, "hls-run")
    assert "site_waiver" not in seen
    assert seen["overrides"] == {"site.qos": "normal", "task.max_context_len": "4096"}
    assert started == [("the plan", {"site.qos": "normal", "task.max_context_len": "4096"})]
    # What the run lacks is made first, with the run's own overrides; a stop build makes nothing.
    [made] = prepared
    assert (made["task_module"], made["arm"], made["arm_change"], made["build_only"], made["overrides"]) == (
        "tasks.multitenant_slo.task", "arm-a", None, True, {"site.qos": "normal", "task.max_context_len": "4096"})
    CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a", "--probe-only"])
    assert len(prepared) == 1 and seen["stop_after"] == "probe"
    # --site-waiver is no option; it reaches the planner as an override of no
    # settings root, which the planner refuses (``split_overrides``).
    CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a", "--site-waiver", "a reason"])
    assert "site_waiver" not in seen and seen["overrides"] == {"site-waiver": "a reason"}
    assert "--site-waiver" not in _text(CliRunner().invoke(cli.app, ["run", "--help"]).output)


@pytest.mark.parametrize("argv, match", [
    (["--build-only", "--probe-only"], "--build-only builds and packages"),
    (["--probe-only", "--preflight-only"], "give at most one of --probe-only and --preflight-only"),
    (["--probe-oom"], "--probe-oom runs only with --probe-only"),
], ids=["build-only-and-stop", "both-stops", "oom-alone"])
def test_the_cli_refuses_flags_that_cannot_go_together(monkeypatch, argv, match):
    monkeypatch.setattr(commands, "plan_run", lambda **kw: pytest.fail("planned"))
    result = CliRunner().invoke(cli.app, ["run", "tasks.multitenant_slo.task", "--arm", "arm-a", *argv])
    assert result.exit_code == 2 and match in _text(result.output)


def test_the_old_commands_are_gone():
    for name in ("build", "evaluate", "install-credit"):
        result = CliRunner().invoke(cli.app, [name, "--help"])
        assert result.exit_code == 2 and "No such command" in result.output
    for name in ("_assemble_run", "_run", "_grade", "_run_and_finalize", "_build_config", "_resolve_latest",
                 "_assemble_calibrate", "_assemble_baseline_credit", "build_command", "evaluate_command"):
        assert not hasattr(cli, name), name
    for name in ("plan_build", "start_build", "plan_evaluate", "start_evaluate"):
        assert not hasattr(commands, name), name
    for name in ("run", "grade", "follow"):
        assert CliRunner().invoke(cli.app, [name, "--help"]).exit_code == 0
