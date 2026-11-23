"""``hls export``'s planning on the login node, the command line, and the whole path on the
in-process Slurm."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.pipeline import commands
from hls.pipeline import export as export_pipeline
from hls.pipeline.export import EXPORT_SHAPE, split_arguments
from hls.pipeline.plan import Toolchain
from hls.pipeline.planner import PlanError
from hls.spec.resources import ResourceShape
from hls.stages.export import EXPORT_FILE, ExportRecord
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from hls.substrate.slurm.scheduler import submit_argv
from tests.export.instances import MODULE, SECRETS_FILE, calibration, ended, finalized
from tests.finalize.published import ROLLOUT, make_plan
from tests.pipeline.hls_cli import run_hls
from tests.pipeline.plans import CODE
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.provision.warmer_config import write_config


@pytest.fixture
def login(monkeypatch, tmp_path, checkout):
    """The login node as the export's planning sees it, planning from *checkout*."""
    code = CODE.model_copy(update={"checkout": checkout})
    monkeypatch.setattr(export_pipeline, "checkout_root", lambda: checkout)
    monkeypatch.setattr(export_pipeline, "code_state", lambda path: code)
    monkeypatch.setattr(export_pipeline, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    return code


def _plan(tmp_path, instances, *, dest=None, oracle=False, overrides=None, secrets=SECRETS_FILE, echo=None):
    config = write_config(tmp_path, secrets=secrets)
    return export_pipeline.plan_export(config=config, overrides=overrides or {}, instances=instances,
                                       destination=dest if dest is not None else tmp_path / "exported",
                                       oracle=oracle, job_name="hls-export", echo=echo or [].append)


def _rollout(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, MODULE)
    store = PosixStore(plan.store_root)
    return ended(store, plan), store


def test_an_export_plans_one_stage_on_the_cpu_partition_for_its_fixed_time(tmp_path, checkout, login):
    """The stage gets ``EXPORT_SHAPE``, an hour, whatever its instances hold: here over a terabyte,
    which the plan states, by a walk of the entries that reads no file."""
    rollout, store = finalized(tmp_path, checkout)
    # A sparse file: its size counts, though the disk holds none of it.
    with open(store.path(rollout.layout.attempt_file("package", 1, "outputs/big.bin")), "wb") as f:
        f.truncate(10**12)
    said: list[str] = []
    plan = _plan(tmp_path, [rollout.instance], echo=said.append)
    assert (plan.task, plan.pipeline) == ("fin", "export")
    assert plan.instance.startswith("exports/fin/") and plan.instance.count("/") == 2
    [stage] = plan.stages
    assert (stage.name, stage.kind) == ("export", "export")
    assert stage.params == {"instances": [rollout.instance], "oracle": False,
                            "destination": str(tmp_path / "exported")}
    assert stage.shape == EXPORT_SHAPE == ResourceShape(cpus=2, memory_gib=12, time_sec=3600)
    assert not stage.shape_overridden
    nbytes = sum(p.lstat().st_size for p in store.path(rollout.instance).rglob("*")
                 if p.is_file() and not p.is_symlink())
    assert 10**12 < nbytes < 10**12 + 2**20
    assert f"export of 1 instance of fin, {nbytes / 2**30:.1f} GiB, to {tmp_path / 'exported'}" in said
    [alloc] = plan.allocations
    assert alloc.placement.partition == "cpu-a" and not alloc.singleton


def test_a_rollout_is_exported_with_its_gradings(tmp_path, checkout, login):
    rollout, store = _rollout(tmp_path, checkout)
    grading = ended(store, make_plan(tmp_path, checkout, MODULE, kind="grading"))
    store.path(f"{rollout.instance}/gradings/g-empty").mkdir(parents=True)
    plan = _plan(tmp_path, [rollout.instance])
    assert plan.stage("export").params["instances"] == [rollout.instance, grading.instance]
    # Named twice, a grading is exported once.
    plan = _plan(tmp_path, [rollout.instance, grading.instance])
    assert plan.stage("export").params["instances"] == [rollout.instance, grading.instance]


def test_a_rollouts_grading_still_running_is_refused(tmp_path, checkout, login):
    rollout, store = _rollout(tmp_path, checkout)
    grading = ended(store, make_plan(tmp_path, checkout, MODULE, kind="grading"), status="running")
    with pytest.raises(ValueError, match=f"{grading.instance} has not ended"):
        _plan(tmp_path, [rollout.instance])


def test_an_export_of_two_tasks_is_refused(tmp_path, checkout, login):
    with pytest.raises(ValueError, match="an export holds one task's instances, and these are of a, b"):
        _plan(tmp_path, ["runs/a/r1", "runs/b/r2"])


@pytest.mark.parametrize("instance", ["exports/fin/x1", "oracle/fin", "runs/fin/r1/stages", "runs/fin",
                                      "warmers/fin/w1/stages/package", "credits/fin/c1/gradings/g1"],
                         ids=["export", "oracle", "inside", "task", "inside-warmer", "credit-grading"])
def test_what_is_not_an_exportable_instance_is_refused(tmp_path, checkout, login, instance):
    with pytest.raises(ValueError, match="is not an instance an export publishes"):
        _plan(tmp_path, [instance])


@pytest.mark.parametrize("instance, why", [("runs/fin/index", "index directory"), ("runs/fin/r+1", r"holds \+"),
                                           ("warmers/fin/index/gradings/g1", "index directory")])
def test_an_id_no_instance_may_hold_is_refused_before_the_store_is_read(tmp_path, checkout, login, instance, why):
    with pytest.raises(ValueError, match=f"is not an instance an export publishes: .*{why}"):
        _plan(tmp_path, [instance])


def test_a_warmer_with_no_package_stage_is_refused(tmp_path, checkout, login):
    base = make_plan(tmp_path, checkout, MODULE, names_published=False)
    warmer = base.model_copy(update={"instance": InstanceLayout.warmer("fin", "w-prov").root, "pipeline": "provision"})
    ended(PosixStore(base.store_root), warmer)
    with pytest.raises(ValueError, match="is a warmer with no package stage"):
        _plan(tmp_path, [warmer.instance])


def test_an_instance_still_running_or_missing_is_refused(tmp_path, checkout, login):
    plan = make_plan(tmp_path, checkout, MODULE)
    ended(PosixStore(plan.store_root), plan, status="running")
    with pytest.raises(ValueError, match="has not ended"):
        _plan(tmp_path, [plan.instance])
    with pytest.raises(ValueError, match=f"no instance runs/fin/{ROLLOUT}-missing"):
        _plan(tmp_path, [f"runs/fin/{ROLLOUT}-missing"])


@pytest.mark.parametrize("where, match", [
    ("store", "lies inside the store"),
    ("checkout", "lies inside the checkout"),
    ("exists", "exists; an export writes a directory of its own"),
    ("relative", "must be an absolute path"),
    ("no-parent", "its parent directory"),
    ("through-a-link", "lies inside the store"),
])
def test_a_destination_outside_the_rules_is_refused(tmp_path, checkout, login, where, match):
    rollout, _ = _rollout(tmp_path, checkout)
    (tmp_path / "into-store").symlink_to(tmp_path / "store")
    dest = {"store": tmp_path / "store" / "out", "checkout": checkout / "out", "exists": tmp_path / "exists",
            "relative": Path("out"), "no-parent": tmp_path / "nowhere" / "out",
            "through-a-link": tmp_path / "into-store" / "out"}[where]
    if where == "exists":
        dest.mkdir()
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, [rollout.instance], dest=dest)


def test_an_export_without_a_secrets_file_is_refused(tmp_path, checkout, login):
    rollout, _ = _rollout(tmp_path, checkout)
    with pytest.raises(PlanError, match="stage export needs every value of the site's secrets file, and there is none"):
        _plan(tmp_path, [rollout.instance], secrets=None)


def test_an_export_with_a_secrets_file_of_no_secret_is_refused(tmp_path, checkout, login):
    rollout, _ = _rollout(tmp_path, checkout)
    with pytest.raises(PlanError, match="holds no secret: an export searches every file for the values"):
        _plan(tmp_path, [rollout.instance], secrets="[secrets]\n")


def test_task_overrides_are_refused(tmp_path, checkout, login):
    rollout, _ = _rollout(tmp_path, checkout)
    with pytest.raises(ValueError, match="an export runs no task's code"):
        _plan(tmp_path, [rollout.instance], overrides={"task.seed": "3"})


def test_oracle_answers_that_are_not_there_are_refused(tmp_path, checkout, login):
    rollout, _ = _rollout(tmp_path, checkout)
    with pytest.raises(ValueError, match="--oracle: fin has no reference answers in the store"):
        _plan(tmp_path, [rollout.instance], oracle=True)


def test_a_calibration_is_planned_for_the_exports_fixed_time(tmp_path, checkout, login):
    """A calibration's captures and scored servings, retried or not, leave the export's time at
    ``EXPORT_SHAPE``'s hour."""
    plan, store = calibration(tmp_path, checkout, "c-1", draws=[0, 1])
    for stage in ("capture_vllm_1", "score_1"):
        store.write_bytes(plan.layout.attempt_file(stage, 2, "CLAIM"), b"")
    [stage] = _plan(tmp_path, [plan.instance]).stages
    assert stage.params["instances"] == [plan.instance]
    assert stage.shape == EXPORT_SHAPE and stage.shape.time_sec == 3600


def test_arguments_split_into_instances_and_overrides_in_either_form():
    assert split_arguments(["runs/t/r1", "--site.partition", "gpu-x", "runs/t/r2", "--run.poll_sec=5"]) == (
        ["runs/t/r1", "runs/t/r2"], ["--site.partition", "gpu-x", "--run.poll_sec=5"])


def test_the_cli_plans_the_named_instances_destination_and_oracle(monkeypatch, tmp_path):
    seen = {}

    def plan_export(**kwargs):
        seen.update(kwargs)
        return "the plan"

    monkeypatch.setattr(export_pipeline, "plan_export", plan_export)
    monkeypatch.setattr(commands, "start", lambda plan, overrides, echo: seen.update(started=(plan, overrides)) or 0)
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["export", "runs/fin/r1", "--site.partition", "gpu-x", "runs/fin/r2",
                                          "--to", str(tmp_path / "out"), "--oracle", "--run.poll_sec=5"])
    assert result.exit_code == 0, result.output
    assert (seen["instances"], seen["destination"], seen["oracle"]) == (["runs/fin/r1", "runs/fin/r2"],
                                                                        tmp_path / "out", True)
    assert seen["overrides"] == {"site.partition": "gpu-x", "run.poll_sec": "5"}
    assert seen["started"] == ("the plan", {"site.partition": "gpu-x", "run.poll_sec": "5"})


def test_the_cli_refuses_in_one_line(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    result = CliRunner().invoke(cli.app, ["export", "runs/a/r1", "runs/b/r2", "--to", str(tmp_path / "out"),
                                          "--config", str(write_config(tmp_path, secrets=SECRETS_FILE))])
    assert result.exit_code == 1
    assert result.output.strip().splitlines() == [
        "hls export: an export holds one task's instances, and these are of a, b"]


def test_hls_export_end_to_end_local(tmp_path, checkout, login):
    rollout, store = finalized(tmp_path, checkout)
    config = write_config(tmp_path, secrets=SECRETS_FILE)
    slurm = LocalSlurm()

    def submit_and_run(request):
        job_id = LocalSlurm.submit(slurm, request)
        slurm.tick()
        return job_id

    slurm.submit = submit_and_run
    dest = tmp_path / "exported"
    code, lines = run_hls(slurm, ["export", rollout.instance, "--to", str(dest), "--job-name", "hls-export-test",
                                  "--config", str(config)])
    assert code == 0, lines
    assert lines[-1].endswith(": succeeded: every stage published")
    [job] = slurm.jobs.values()
    assert "--partition=cpu-a" in submit_argv(job.request)
    record = ExportRecord.model_validate_json((dest / EXPORT_FILE).read_bytes())
    [done] = record.instances
    assert done.instance == rollout.instance and f"runs/fin/index/{ROLLOUT}.json" in {f.path for f in done.copied}
    assert all((dest / f.path).is_file() for f in done.copied)
