"""``hls grade --rollout``'s planning: a grading of a rollout's published package lives under the
rollout, pins the package by digest, replays the task overrides the rollout ran under, records
the building arm, whose route secrets finalize then counts contestant-facing, and with
``--artifact.rebuild`` scans and packages the rollout's source bundle again at the grading's
checkout. And ``--artifact.image``: a published submission image graded again by the digest of
its file, packaged by nobody."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.arms import ArmRecord
from hls.pipeline import commands
from hls.pipeline.pipelines import STAGING_SEC
from hls.pipeline.plan import Plan
from hls.pipeline.planner import PlanError
from hls.measured import BASELINE_CREDIT_FILE, Found, MeasuredFile, MeasuredFiles
from hls.recipes import catalog
from hls.recipes.model import CONSTRAINTS
from hls.stages.agent_build import AgentBuildParams, arm_route_secrets
from hls.stages.deploy_drive import SEAL_PROBE_SEC
from hls.stages.finalize import FinalizeParams, contestant_facing
from hls.stages.package import PipLimits, pip_limits
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from tests.agent.build_fixture import KEY, KEY_NAME, TASK_ID, build_config, publish_build
from tests.agent.test_hls_run import repo_login  # noqa: F401  (the login node planning from this repository)
from tests.evaluate.warmer import WARMER, publish_package, publish_stage
from tests.provision.warmer_config import TOKEN
from tests.provision.cached import provision_task

LAYER = catalog.image_set("sample_task").layer
#: The site's secrets by name, each with its own value: the arm's route key and the hub's token.
REGISTRY = {KEY_NAME: KEY, "hf_token": TOKEN}
GRADING = ["provision", "images", "deploy_drive", "verify", "score", "finalize"]
#: The fixture packages' submission image.
SUBMISSION_IMAGE = b"submission image"
#: The staging every drive is given before its server starts: the fixed allowance for its copies
#: and extractions, and its seal probe.
DRIVE_STAGING = STAGING_SEC + math.ceil(SEAL_PROBE_SEC)


def _rollout(tmp_path: Path, *, module: str = "bd_task.task", overrides: dict[str, str] | None = None,
             package: bool = True) -> Plan:
    """A build-only rollout of *module*, the fixture task by default, as ``hls run --build-only``
    plans it, with its agent_build published and, when *package*, its package; returns its plan.
    The task's provisioning is in the cache first, as the plan reads it."""
    provision_task(tmp_path / "store", module, overrides)
    plan = commands.plan_run(config=build_config(tmp_path), overrides=overrides or {}, task_module=module,
                             arm="arm-a", arm_change=None, job_name="x", build_only=True, echo=[].append)
    store = PosixStore(tmp_path / "store")
    arm = AgentBuildParams.model_validate(plan.stage("agent_build").params).arm
    publish_build(store, plan, arm, created_at=100.0)
    if package:
        publish_package(store, SUBMISSION_IMAGE, layout=InstanceLayout(plan.instance))
    return plan


def _grade(tmp_path: Path, *, module: str = "bd_task.task", overrides: dict[str, str] | None = None, **kw) -> Plan:
    """``hls grade``'s plan, the task's provisioning in the cache first."""
    provision_task(tmp_path / "store", module, overrides)
    return commands.plan_grade(config=build_config(tmp_path), overrides=overrides or {}, task_module=module,
                               job_name="hls-grade", echo=[].append, **kw)


def test_a_grading_of_a_rollout_lives_under_it_and_pins_its_package(tmp_path, build_login, build_catalog):
    rollout = _rollout(tmp_path)
    plan = _grade(tmp_path, rollout=rollout.instance)
    grading = plan.instance.removeprefix(f"{rollout.instance}/gradings/")
    assert grading.startswith("g-") and "/" not in grading and plan.pipeline == "grading"
    assert [s.name for s in plan.stages] == GRADING
    layout = InstanceLayout(rollout.instance)
    key = layout.manifest("package")
    digest = sha256_bytes(PosixStore(tmp_path / "store").read_bytes(key))
    drive = plan.stage("deploy_drive")
    assert drive.params["artifact"] == {"kind": "package", "stage": "package", "manifest": key}
    assert {s.name: s.inputs for s in plan.stages if s.inputs} == {"deploy_drive": {key: digest}}
    assert drive.params["staging_sec"] == DRIVE_STAGING
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert (final.kind, final.eval_id, final.score_stage, final.build_stage, final.package_stage) == (
        "grading", grading, "score", None, None)


def test_a_grading_of_a_rollout_counts_its_arms_route_secrets_contestant_facing(tmp_path, build_login,
                                                                                  build_catalog):
    rollout = _rollout(tmp_path)
    built = AgentBuildParams.model_validate(rollout.stage("agent_build").params)
    plan = _grade(tmp_path, rollout=rollout.instance)
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert final.built_arm == built.arm
    # Finalize counts the building arm's route secrets the contestant's, though no stage of the
    # grading is granted them.
    facing = contestant_facing(plan, REGISTRY, final.built_arm)
    assert arm_route_secrets(built.arm) <= facing and KEY_NAME in facing and "hf_token" not in facing
    # A grading holds the secrets' names, never a value, and is granted none of them.
    assert KEY not in plan.model_dump_json()
    assert all(KEY_NAME not in s.grants.values() for s in plan.stages)


def test_a_grading_of_a_package_warmer_counts_no_route_secret(tmp_path, build_login, build_catalog):
    store = PosixStore(tmp_path / "store")
    publish_package(store, b"submission image")
    plan = _grade(tmp_path, package=f"warmers/{TASK_ID}/{WARMER}")
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert final.built_arm is None
    assert KEY_NAME not in contestant_facing(plan, REGISTRY, final.built_arm)


def test_a_grading_replays_the_rollouts_task_overrides(tmp_path, build_login, build_catalog, monkeypatch):
    asked = []
    monkeypatch.setattr(commands, "check_oracle_coverage", lambda store, **kw: asked.append(kw))
    rollout = _rollout(tmp_path, overrides={"task.max_context_len": "4096"})
    plan = _grade(tmp_path, rollout=rollout.instance)
    assert plan.stage("deploy_drive").params["task"]["overrides"] == {"task.max_context_len": 4096}
    assert plan.stage("provision").params["task"]["overrides"] == {"task.max_context_len": 4096}
    # The oracle coverage is checked for the task the grading verifies, the rollout's overrides
    # applied, and the reference that holds them.
    kw = asked[-1]
    assert kw["task"].max_context_len == 4096
    assert kw["task_ref"].overrides == {"task.max_context_len": 4096}
    # The command's own --task.* lies over them.
    plan = _grade(tmp_path, rollout=rollout.instance, overrides={"task.max_context_len": "8192"})
    assert plan.stage("verify").params["task"]["overrides"] == {"task.max_context_len": 8192}


def test_a_rebuild_packages_the_rollouts_bundle_pinned_by_digest(tmp_path, build_login, build_catalog):
    rollout = _rollout(tmp_path)
    layout = InstanceLayout(rollout.instance)
    build_key = layout.manifest("agent_build")
    build_digest = sha256_bytes(PosixStore(tmp_path / "store").read_bytes(build_key))
    plan = _grade(tmp_path, rollout=rollout.instance, rebuild=True)
    assert [s.name for s in plan.stages] == ["provision", "images", "source_scan", "package", "deploy_drive", "verify",
                                             "score", "finalize"]
    scan = plan.stage("source_scan")
    assert scan.inputs == {build_key: build_digest} and scan.after == []
    assert (scan.params["bundle"], scan.params["build_manifest"]) == (None, build_key)
    package = plan.stage("package")
    assert package.inputs == {build_key: build_digest} and package.after == ["images", "source_scan"]
    assert (package.params["bundle"], package.params["build_manifest"]) == (None, build_key)
    assert package.params["submission_base"] == LAYER
    assert package.params["constraints"] == CONSTRAINTS
    drive = plan.stage("deploy_drive")
    assert drive.params["artifact"] == {"kind": "package", "stage": "package", "manifest": None}
    assert drive.after == ["provision", "images", "package"] and drive.inputs == {}
    # The image it serves is built in this instance, after the plan; its drive has the same staging.
    assert drive.params["staging_sec"] == DRIVE_STAGING
    assert LAYER in [d["name"] for d in plan.stage("images").params["images"]]
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert (final.package_stage, final.built_arm) == (
        "package", AgentBuildParams.model_validate(rollout.stage("agent_build").params).arm)
    assert plan.stage("finalize").after == ["package", "deploy_drive", "verify", "score"]


@pytest.mark.parametrize("module, constraints", [("tasks.deepresearch.task", None),
                                                 ("tasks.multitenant_slo.task", CONSTRAINTS)],
                         ids=["published-base", "built-base"])
@pytest.mark.usefixtures("repo_login")
def test_a_rebuild_packages_under_the_constraints_file_its_submission_base_holds(tmp_path, module, constraints):
    """A rebuild packages the rollout's bundle again on the task's submission base: deepresearch's
    published sandbox holds no constraints file, so the offline install names none, and a base hls
    builds holds the file every base writes."""
    rollout = _rollout(tmp_path, module=module)
    plan = _grade(tmp_path, module=module, rollout=rollout.instance, rebuild=True)
    assert plan.stage("package").params["constraints"] == constraints


def test_a_rebuild_holds_the_pip_steps_to_the_limits_the_builds_brief_stated(tmp_path, build_login, build_catalog):
    """The build's brief stated what each pip step of its package may use (``package_pip``), from
    the runner's reserve where it was planned. A rebuild graded where the runner keeps another
    reserve holds the steps to those figures, never to what its own shape would give, and a
    grading whose package stage gives less than they is refused before anything runs, unless its
    shape is raised to give them."""
    rollout = _rollout(tmp_path)
    stated = AgentBuildParams.model_validate(rollout.stage("agent_build").params).package_pip
    built = rollout.settings.run.runner_memory_gib
    # A smaller reserve: the shape gives each step more memory, and the steps get what was stated.
    plan = _grade(tmp_path, rollout=rollout.instance, rebuild=True,
                  overrides={"run.runner_memory_gib": str(built - 2)})
    package = plan.stage("package")
    assert pip_limits(package.shape, built - 2).memory_gib == stated.memory_gib + 2
    assert PipLimits.model_validate(package.params["pip"]) == stated
    # A larger one: the shape gives each step 2 GiB less than the brief stated.
    larger = {"run.runner_memory_gib": str(built + 2)}
    with pytest.raises(PlanError) as e:
        _grade(tmp_path, rollout=rollout.instance, rebuild=True, overrides=larger)
    assert str(e.value) == (f"stage package: the plan states {stated.disk_gib} GiB of disk and {stated.memory_gib} "
                            f"GiB of memory for each pip step, and the stage's shape gives {stated.disk_gib} and "
                            f"{stated.memory_gib - 2}")
    # The site's resources give the package stage the memory back, and the steps the stated figures.
    raised = {**larger, f"resources.{TASK_ID}.package.memory_gib": str(package.shape.memory_gib + 4)}
    plan = _grade(tmp_path, rollout=rollout.instance, rebuild=True, overrides=raised)
    assert PipLimits.model_validate(plan.stage("package").params["pip"]) == stated


def test_a_rebuild_of_a_build_whose_plan_states_no_pip_limits_is_refused(tmp_path, build_login, build_catalog):
    """A rollout whose build's plan has no ``package_pip`` states no pip limits: its bundle
    cannot be packaged again under the limits its brief stated, so a rebuild is refused naming
    the plan; its published package is graded all the same."""
    rollout = _rollout(tmp_path)
    store, layout = PosixStore(tmp_path / "store"), InstanceLayout(rollout.instance)
    key = layout.plan_revision(1)
    raw = json.loads(store.read_bytes(key))
    [build] = [s for s in raw["stages"] if s["kind"] == "agent_build"]
    del build["params"]["package_pip"]
    store.path(key).write_text(json.dumps(raw))
    with pytest.raises(ValueError) as e:
        _grade(tmp_path, rollout=rollout.instance, rebuild=True)
    assert str(e.value) == (f"--rollout {rollout.instance}: {key} states no pip limits its build's brief gave the "
                            "package, which a rebuild holds its pip steps to: missing package_pip (required by this "
                            "version of hls); grade its published package without --artifact.rebuild")
    assert [s.name for s in _grade(tmp_path, rollout=rollout.instance).stages] == GRADING


@pytest.mark.parametrize("kw, match", [
    ({}, "name exactly one artifact to grade"),
    ({"rollout": "runs/x/y", "measured": "baseline"}, "name exactly one artifact to grade"),
    ({"package": f"warmers/{TASK_ID}/{WARMER}", "rebuild": True}, "--artifact.rebuild .* needs --rollout"),
    ({"rollout": "runs/sample_task/r1"}, "is not a rollout of evaluate_task"),
    ({"rollout": f"runs/{TASK_ID}/nothing-here"}, "the store holds no such rollout"),
    ({"rollout": f"runs/{TASK_ID}/r1/gradings/g1"}, f"is not a rollout of {TASK_ID}"),
    ({"rollout": f"runs/{TASK_ID}/index"}, f"is not a rollout of {TASK_ID}: .*index directory"),
    ({"rollout": f"runs/{TASK_ID}/r+1"}, rf"is not a rollout of {TASK_ID}: .*holds \+"),
], ids=["none", "two", "rebuild-without-rollout", "another-task", "no-plan", "grading", "index", "plus"])
def test_what_cannot_be_graded_is_refused(tmp_path, build_login, build_catalog, kw, match):
    with pytest.raises(ValueError, match=match):
        _grade(tmp_path, **kw)


def test_a_rollout_with_no_package_or_of_another_module_is_refused(tmp_path, build_login, build_catalog):
    rollout = _rollout(tmp_path, package=False)
    with pytest.raises(ValueError, match="has no published package stage"):
        _grade(tmp_path, rollout=rollout.instance)
    # A package manifest published without the package record is no package to grade either.
    store, layout = PosixStore(tmp_path / "store"), InstanceLayout(rollout.instance)
    sif = layout.attempt_file("package", 1, "outputs/submission.sif")
    store.write_bytes(sif, b"submission image")
    publish_stage(store, layout, "package", {"submission": sif})
    with pytest.raises(ValueError, match="has no published package stage"):
        _grade(tmp_path, rollout=rollout.instance)
    other = _rollout(tmp_path)
    with pytest.raises(ValueError, match="built for bd_task.task, not bd_effort.task"):
        _grade(tmp_path, module="bd_effort.task", rollout=other.instance)


def test_a_rollouts_plan_is_read_for_its_task_and_arm_alone(tmp_path, build_login, build_catalog):
    """A grading reads of the rollout's plan only the task and the arm its agent_build ran: a plan
    another version of hls wrote, with settings this code does not know, is graded all the same, and
    one whose arm does not parse is refused, naming the plan and the arm's errors."""
    rollout = _rollout(tmp_path, overrides={"task.max_context_len": "4096"})
    store, layout = PosixStore(tmp_path / "store"), InstanceLayout(rollout.instance)
    key = layout.plan_revision(1)
    raw = json.loads(store.read_bytes(key))
    raw["settings"]["site"]["no_such_setting"] = 1
    store.path(key).write_text(json.dumps(raw))
    plan = _grade(tmp_path, rollout=rollout.instance)
    assert plan.stage("deploy_drive").params["task"]["overrides"] == {"task.max_context_len": 4096}
    [build] = [s for s in raw["stages"] if s["kind"] == "agent_build"]
    del build["params"]["arm"]["model"]
    store.path(key).write_text(json.dumps(raw))
    with pytest.raises(ValueError) as e:
        _grade(tmp_path, rollout=rollout.instance)
    assert str(e.value) == (f"--rollout {rollout.instance}: {key} names no task and arm record: missing arm.model "
                            "(required by this version of hls)")


def test_a_grading_refuses_a_dirty_checkout(tmp_path, build_login, build_catalog, monkeypatch):
    rollout = _rollout(tmp_path)
    dirty = build_login.model_copy(update={"dirty_diff_sha256": "e" * 64})
    monkeypatch.setattr(commands, "code_state", lambda path: dirty)
    with pytest.raises(PlanError, match=str(dirty.checkout)):
        _grade(tmp_path, rollout=rollout.instance)


def test_a_grading_refuses_an_uncommitted_default_and_pins_what_it_loads(tmp_path, build_login,
                                                                                       build_catalog, monkeypatch):
    rollout = _rollout(tmp_path)
    pin = MeasuredFile(source="defaults", path="bd_task/defaults/baseline.json", blob="f" * 40)
    monkeypatch.setattr(commands, "find_measured", lambda store, **kw: [Found(name=BASELINE_CREDIT_FILE, pin=pin)])
    seen = []

    def refuse(checkout, files):
        seen.append((checkout, files))
        raise PlanError("bd_task/defaults/baseline.json is not committed at HEAD")

    monkeypatch.setattr(commands, "check_defaults_committed", refuse)
    with pytest.raises(PlanError, match="baseline.json is not committed at HEAD"):
        _grade(tmp_path, rollout=rollout.instance)
    assert seen == [(commands.checkout_root(), MeasuredFiles(credit=pin))]
    monkeypatch.setattr(commands, "check_defaults_committed", lambda checkout, files: None)
    lines: list[str] = []
    plan = commands.plan_grade(config=build_config(tmp_path), overrides={}, task_module="bd_task.task",
                               rollout=rollout.instance, job_name="hls-grade", echo=lines.append)
    assert plan.measured == MeasuredFiles(credit=pin)
    assert f"baseline credit: the task's defaults bd_task/defaults/baseline.json (blob {'f' * 12})" in lines


def test_the_cli_refuses_a_grading_from_a_dirty_checkout_in_one_line_before_submitting(tmp_path, build_login,
                                                                                       build_catalog, monkeypatch):
    rollout = _rollout(tmp_path)
    dirty = build_login.model_copy(update={"dirty_diff_sha256": "e" * 64})
    monkeypatch.setattr(commands, "code_state", lambda path: dirty)
    monkeypatch.setattr(commands, "start", lambda *args, **kw: pytest.fail("the plan was submitted"))
    result = CliRunner().invoke(cli.app, ["grade", "bd_task.task", "--rollout", rollout.instance, "--config",
                                          str(build_config(tmp_path))])
    assert result.exit_code == 1
    [line] = result.output.strip().splitlines()
    assert line.startswith(f"hls grade: the checkout {dirty.checkout} or its harbor submodule has uncommitted")


def test_the_arm_a_grading_records_is_the_builds(tmp_path, build_login, build_catalog):
    rollout = _rollout(tmp_path)
    final = FinalizeParams.model_validate(_grade(tmp_path, rollout=rollout.instance).stage("finalize").params)
    assert isinstance(final.built_arm, ArmRecord) and final.built_arm.name == "arm-a"


# ---------------------------------------------------------------------------
# A published image graded again by the digest of its file
# ---------------------------------------------------------------------------


def _published(tmp_path: Path, data: bytes = b"a published submission image") -> tuple[str, Path]:
    """A published submission image on the shared filesystem: its file's digest and the file."""
    path = tmp_path / "published" / "submission.sif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data), path


def test_a_published_image_is_graded_by_its_digest_and_packaged_by_nobody(tmp_path, build_login, build_catalog):
    digest, path = _published(tmp_path)
    plan = _grade(tmp_path, image=f"sha256:{digest}", image_file=path, overrides={"task.max_context_len": "4096"})
    grading = plan.instance.removeprefix(f"warmers/{TASK_ID}/image-{digest}/gradings/")
    assert grading.startswith("g-") and "/" not in grading and plan.pipeline == "grading"
    assert [s.name for s in plan.stages] == GRADING
    drive = plan.stage("deploy_drive")
    assert drive.params["artifact"] == {"kind": "published", "sha256": digest, "source": str(path)}
    assert drive.inputs == {} and drive.after == ["provision", "images"]
    # The images stage touches nothing of the contestant's: the harness and the reference alone.
    images = plan.stage("images").params
    assert [d["name"] for d in images["images"]] == ["s-a", LAYER]
    # An override the rollout ran under is given on the command line and is the plan's.
    assert drive.params["task"]["overrides"] == {"task.max_context_len": 4096}
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert (final.kind, final.eval_id, final.package_stage, final.built_arm) == ("grading", grading, None, None)
    assert KEY_NAME not in contestant_facing(plan, {KEY_NAME: KEY}, final.built_arm)


def test_a_grading_of_a_task_with_no_data_is_refused_when_planned(tmp_path, build_login, build_catalog,
                                                                   monkeypatch):
    digest, path = _published(tmp_path)
    load = commands.load_task
    monkeypatch.setattr(commands, "load_task",
                        lambda ref, **kw: load(ref, **kw).model_copy(update={"data_source": None}))
    with pytest.raises(PlanError, match="declares no data source on any tenant"):
        _grade(tmp_path, image=f"sha256:{digest}", image_file=path)


def test_a_published_images_drive_has_a_packaged_ones_time_whatever_its_files_size(tmp_path, build_login,
                                                                                    build_catalog):
    """The drive extracts the published file within the fixed staging every drive has, so its
    time is a packaged image's drive's, however large the file the login node sees."""
    rollout = _rollout(tmp_path)
    packaged = _grade(tmp_path, rollout=rollout.instance).stage("deploy_drive")
    for size in (1_000_000_000, 9_500_000_000):
        path = tmp_path / "published" / f"{size}.sif"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            # A sparse file of that size.
            f.truncate(size)
        published = _grade(tmp_path, image=f"sha256:{'ab' * 32}", image_file=path).stage("deploy_drive")
        assert published.params["staging_sec"] == packaged.params["staging_sec"] == DRIVE_STAGING
        assert published.shape.time_sec == packaged.shape.time_sec


def test_a_declared_building_arm_makes_its_route_secrets_contestant_facing(tmp_path, build_login, build_catalog):
    digest, path = _published(tmp_path)
    plan = _grade(tmp_path, image=f"sha256:{digest}", image_file=path, built_arm="arm-a")
    final = FinalizeParams.model_validate(plan.stage("finalize").params)
    assert isinstance(final.built_arm, ArmRecord) and final.built_arm.name == "arm-a"
    assert KEY_NAME in contestant_facing(plan, {KEY_NAME: KEY}, final.built_arm)
    assert KEY not in plan.model_dump_json()
    assert all(KEY_NAME not in s.grants.values() for s in plan.stages)


@pytest.mark.parametrize("kw, match", [
    ({"image": "sha256:" + "a" * 64}, "--artifact.image grades a file by its digest: name the file with --from"),
    ({"package": f"warmers/{TASK_ID}/{WARMER}", "image_file": Path("/x.sif")}, "--from PATH names the file of "),
    ({"package": f"warmers/{TASK_ID}/{WARMER}", "built_arm": "arm-a"}, "--built-arm .* goes with --artifact.image"),
    ({"image": "md5:abc", "image_file": Path("/x.sif")}, r"--artifact.image is sha256:<64 hex digits>"),
    ({"image": "sha256:" + "a" * 64, "image_file": Path("/no/such/submission.sif")}, "no such file"),
    ({"image": "sha256:" + "a" * 64, "image_file": Path("/x.sif"), "package": f"warmers/{TASK_ID}/{WARMER}"},
     "name exactly one artifact to grade"),
    ({"image": "sha256:" + "a" * 64, "image_file": Path("/x.sif"), "rebuild": True}, "--artifact.rebuild .* needs"),
], ids=["no-file", "file-alone", "arm-alone", "not-a-digest", "no-such-file", "two", "rebuild"])
def test_what_a_published_image_grading_cannot_be_is_refused(tmp_path, build_login, build_catalog, kw, match):
    with pytest.raises(ValueError, match=match):
        _grade(tmp_path, **kw)


def test_the_cli_takes_a_published_image_its_file_and_its_arm(monkeypatch):
    seen = {}
    monkeypatch.setattr(commands, "plan_grade", lambda **kw: seen.update(kw) or "the plan")
    monkeypatch.setattr(commands, "start", lambda made, overrides, **kw: 0)
    digest = "a" * 64
    result = CliRunner().invoke(cli.app, ["grade", "tasks.tab_completion.task", "--artifact.image", f"sha256:{digest}",
                                          "--from", "/data/published/submission.sif", "--built-arm", "arm-a",
                                          "--task.max_context_len", "4096"])
    assert result.exit_code == 0, result.output
    assert (seen["image"], seen["image_file"], seen["built_arm"]) == (
        f"sha256:{digest}", Path("/data/published/submission.sif"), "arm-a")
    assert (seen["rollout"], seen["package"], seen["measured"]) == (None, None, None)
    assert seen["overrides"] == {"task.max_context_len": "4096"}
