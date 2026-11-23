"""A plan whose drive binds the site's sandbox stores (``Stage.sandbox_stores``) is refused unless
the site's ``[sandbox_stores]`` names each, and a drive records its task's tool runner's host
paths only when the runner names some: every other task's drive plans with no such field, and its
plan's settings with no such table."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import pytest

from hls.pipeline.commands import evaluation_allocation, produced_artifact
from hls.pipeline.pipelines import drive_decl
from hls.pipeline.planner import PlanError, StageDecl
from hls.recipes import catalog
from hls.settings import RunSettings, load_settings, resolved
from hls.spec.tools import DEVICE
from hls.stages.deploy_drive import DeployDriveParams, DeployDriveStage, PackagedArtifact
from hls.taskref import TaskRef, load_task
from tasks.agentic_coding.task import TOOL_RUNNER
from tests.drive.test_stage import SHAPE, _params
from tests.pipeline.plans import CODE, noop_plan
from tests.provision.cached import provision_into
from tests.provision.warmer_config import sandbox_store, task_sandbox_stores, write_config

PATHS = TOOL_RUNNER.host_paths
STORES = sorted({p.store for p in PATHS if p.store != DEVICE})
#: Every task of the checkout whose tool runner starts no container.
EXISTING = sorted(set(catalog.task_keys()) - {"agentic_coding"})


def _decl(**params) -> StageDecl:
    return StageDecl("deploy_drive", "deploy_drive", SHAPE, _params(**params))


def _declared() -> list[dict]:
    return [dataclasses.asdict(p) for p in PATHS]


def _named(tmp_path: Path, names) -> dict[str, str]:
    return {name: str(tmp_path / "sandbox-stores" / name) for name in names}


def _site(tmp_path: Path) -> dict[str, str]:
    """A secrets file in a directory of its own, which no store lies in."""
    return {"secrets_path": str(tmp_path / "secrets" / "secrets.toml")}


def test_a_drive_binding_stores_the_site_does_not_name_is_refused_naming_each(tmp_path):
    with pytest.raises(PlanError, match=f"stage deploy_drive binds the sandbox stores {', '.join(STORES)}, which the "
                                        "site does not name"):
        noop_plan(tmp_path, decls=[_decl(sandbox_paths=_declared())])
    with pytest.raises(PlanError, match="binds the sandbox stores agentic-coding-runtime, which"):
        noop_plan(tmp_path, decls=[_decl(sandbox_paths=_declared())], site=_site(tmp_path),
                  sandbox_stores=_named(tmp_path, [s for s in STORES if s != "agentic-coding-runtime"]))


@pytest.mark.parametrize("target, why", [("store", r"the store root \(site\.store_root\)"),
                                         ("secrets", r"the secrets file's directory \(site\.secrets_path\)")])
def test_a_store_that_reaches_the_store_or_the_secrets_through_a_link_is_refused_when_planned(tmp_path, target, why):
    # The file's own check compares the paths as written, so only the planner, which follows the
    # link, sees where the store is.
    stores = _named(tmp_path, STORES)
    (tmp_path / "secrets").mkdir()
    (tmp_path / "store").mkdir()
    (tmp_path / "sandbox-stores").mkdir()
    os.symlink(tmp_path / target, stores["agentic-coding-runtime"])
    with pytest.raises(PlanError, match=rf"sandbox_stores\.agentic-coding-runtime is, holds or lies in {why}, with "
                                        "links followed"):
        noop_plan(tmp_path, decls=[_decl(sandbox_paths=_declared())], sandbox_stores=stores, site=_site(tmp_path))


def test_a_store_in_the_checkout_is_refused_when_planned(tmp_path):
    stores = {**_named(tmp_path, STORES), "agentic-coding-runtime": str(CODE.checkout / "runtime")}
    with pytest.raises(PlanError, match=r"sandbox_stores\.agentic-coding-runtime is, holds or lies in the checkout"):
        noop_plan(tmp_path, decls=[_decl(sandbox_paths=_declared())], sandbox_stores=stores, site=_site(tmp_path))
    # A plan whose stages bind no store is not refused for a store it does not bind.
    assert noop_plan(tmp_path, decls=[_decl()], sandbox_stores=stores, site=_site(tmp_path))


def test_a_drive_whose_stores_the_site_names_is_planned_with_its_paths(tmp_path):
    stores = _named(tmp_path, STORES)
    plan = noop_plan(tmp_path, decls=[_decl(sandbox_paths=_declared())], sandbox_stores=stores, site=_site(tmp_path))
    assert plan.stage("deploy_drive").params["sandbox_paths"] == _declared()
    assert resolved(plan.settings)["sandbox_stores"] == stores
    assert DeployDriveStage.sandbox_stores(DeployDriveParams.model_validate(plan.stage("deploy_drive").params)) == tuple(
        p.store for p in PATHS if p.store != DEVICE)


def test_a_drive_without_host_paths_plans_with_no_such_field_and_its_settings_with_no_such_table(tmp_path):
    plan = noop_plan(tmp_path, decls=[_decl()])
    assert "sandbox_paths" not in plan.stage("deploy_drive").params
    assert "sandbox_stores" not in resolved(plan.settings)
    assert "sandbox_stores" not in plan.model_dump_json() and "sandbox_paths" not in plan.model_dump_json()
    # A site that names stores changes nothing of a drive that binds none.
    named = noop_plan(tmp_path, decls=[_decl()], sandbox_stores=_named(tmp_path, STORES), site=_site(tmp_path))
    assert named.stage("deploy_drive").params == plan.stage("deploy_drive").params


@pytest.mark.parametrize("key", EXISTING)
def test_no_other_tasks_drive_records_host_paths(tmp_path, key):
    module = catalog.task_module(key)
    ref = TaskRef(module=module)
    task = load_task(ref)
    assert task.agent is None or task.agent.tool_runner.host_paths == ()
    decl = drive_decl(name="deploy_drive", task_ref=ref, task=task, catalog_key=key, artifact=PackagedArtifact(),
                      harness_image="harness", serve_command=[], allocation=evaluation_allocation(produced_artifact(module)),
                      run=RunSettings(), seal_memory_gib=0, provisioned=provision_into(tmp_path / "store", task),
                      after=())
    assert "sandbox_paths" not in decl.params
    params = DeployDriveParams.model_validate(decl.params)
    assert params.sandbox_paths == () and DeployDriveStage.sandbox_stores(params) == ()
    # What the plan records of the stage's parameters.
    assert "sandbox_paths" not in params.model_dump(mode="json")


def test_a_drive_of_a_task_whose_runner_starts_containers_records_the_runners_paths(tmp_path):
    # deepresearch's drive, with agentic_coding's tool runner in place of its own: the drive
    # declares the paths of whichever runner the task's agent holds.
    key = "deepresearch"
    ref = TaskRef(module=catalog.task_module(key))
    task = load_task(ref)
    task = task.model_copy(update={"agent": dataclasses.replace(task.agent, tool_runner=TOOL_RUNNER)})
    decl = drive_decl(name="deploy_drive", task_ref=ref, task=task, catalog_key=key, artifact=PackagedArtifact(),
                      harness_image="harness", serve_command=[], allocation=None, run=RunSettings(),
                      seal_memory_gib=0, provisioned=provision_into(tmp_path / "store", task), after=())
    assert decl.params["sandbox_paths"] == _declared()
    assert DeployDriveParams.model_validate(decl.params).sandbox_paths == PATHS


def test_a_test_sites_file_names_the_stores_a_task_needs_beside_its_store(tmp_path):
    module = catalog.task_module("agentic_coding")
    assert task_sandbox_stores(module) == tuple(p.store for p in PATHS if p.store != DEVICE)
    assert all(task_sandbox_stores(catalog.task_module(key)) == () for key in EXISTING)
    settings = load_settings(write_config(tmp_path, sandbox_stores=task_sandbox_stores(module)))
    assert settings.sandbox_stores == {name: sandbox_store(tmp_path, name) for name in STORES}
    assert load_settings(write_config(tmp_path)).sandbox_stores == {}
