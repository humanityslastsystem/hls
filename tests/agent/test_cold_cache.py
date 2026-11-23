"""A cold cache refuses every plan whose stages copy the task's data: ``hls run``, ``hls grade``,
``hls oracle-fill`` and ``hls credit`` each size those copies from the provision record the cache
holds (``commands.planned_provision``), so a store without the task's provisioning refuses the
plan before anything runs, though the plan provisions in its own allocation too. The refusal
names the command to run first, as typed: the module, the configuration, the site overrides,
which choose the store, and the task overrides of the task the plan reads, which choose what is
provisioned; a grading of a rollout reads the task with the overrides the rollout ran under. The
store holds every measured image the plans import, as their images warmer leaves them, so the
task's data is all it lacks; a plan whose measured images the cache lacks is refused too, on a
provisioned store, in the tests of each command. The fixtures are the build tests'
(``conftest.py``), the credit tests' task among them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hls.pipeline import commands, credit
from hls.pipeline.planner import PlanError
from hls.stages.agent_build import AgentBuildParams
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from tests.agent.build_fixture import TASK_ID, build_config, publish_build
from tests.evaluate import fixture as evaluate
from tests.evaluate.warmer import publish_package
from tests.provision.cached import provision_task
from tests.recipes.declared import declare
from tests.store.entries import hold_measured

#: The task overrides the seeded plans read their task with.
SEED = {"task.seed": "7"}
#: A plan of one command: ``plan(config, overrides, task_overrides, tmp_path, monkeypatch)``.
Planner = Callable[[Path, dict[str, str], dict[str, str], Path, pytest.MonkeyPatch], None]


def _run(config: Path, overrides: dict[str, str], task: dict[str, str], tmp_path: Path, monkeypatch) -> None:
    commands.plan_run(config=config, overrides={**overrides, **task}, task_module="bd_task.task", arm="arm-a",
                      arm_change=None, job_name="hls-run", build_only=True, echo=[].append)


def _grade(config: Path, overrides: dict[str, str], task: dict[str, str], tmp_path: Path, monkeypatch) -> None:
    """A grading, with no task overrides of its own, of a build-only rollout that ran under *task*:
    the rollout ran against a store of its own that held the task's data, and its records are in
    the store the grading plans against, which holds none."""
    warm = tmp_path / "warm"
    provision_task(warm, "bd_task.task", task)
    rollout = commands.plan_run(config=config, overrides={**overrides, **task, "site.store_root": str(warm)},
                                task_module="bd_task.task", arm="arm-a", arm_change=None, job_name="hls-run",
                                build_only=True, echo=[].append)
    store = PosixStore(Path(overrides["site.store_root"]))
    arm = AgentBuildParams.model_validate(rollout.stage("agent_build").params).arm
    publish_build(store, rollout, arm, created_at=100.0)
    publish_package(store, b"submission image", layout=InstanceLayout(rollout.instance))
    commands.plan_grade(config=config, overrides=overrides, task_module="bd_task.task", rollout=rollout.instance,
                        job_name="hls-grade", echo=[].append)


def _oracle_fill(config: Path, overrides: dict[str, str], task: dict[str, str], tmp_path: Path, monkeypatch) -> None:
    commands.plan_oracle_fill(config=config, overrides={**overrides, **task}, task_module="bd_task.task", chunk=2,
                              job_name="hls-oracle-fill", echo=[].append)


def _credit(config: Path, overrides: dict[str, str], task: dict[str, str], tmp_path: Path, monkeypatch) -> None:
    """A credit of the credit tests' task, which drives its measured baseline."""
    credit.plan_credit(config=config, overrides={**overrides, **task}, task_module="bd_credit.task", drives=1,
                       fanout=1, job_name="hls-credit", echo=[].append)


def _measured_baseline(tmp_path: Path, monkeypatch) -> None:
    """The fixture task declares a measured baseline and how it starts."""
    declare(monkeypatch, {TASK_ID: evaluate.image_set(("baseline",))})


@pytest.mark.parametrize("task", [{}, SEED], ids=["plain", "seeded"])
@pytest.mark.parametrize("plan, module, measured", [(_run, "bd_task.task", False), (_grade, "bd_task.task", False),
                                                    (_oracle_fill, "bd_task.task", False),
                                                    (_credit, "bd_credit.task", True)],
                         ids=["run", "grade", "oracle_fill", "credit"])
def test_a_plan_on_a_cold_cache_is_refused_naming_the_command_that_provisions_it(tmp_path, build_login, build_catalog,
                                                                                 monkeypatch, plan: Planner,
                                                                                 module: str, measured: bool,
                                                                                 task: dict[str, str]):
    config = build_config(tmp_path)
    cold = tmp_path / "cold"
    if measured:
        _measured_baseline(tmp_path, monkeypatch)
    # Every measured image is imported (the credit's baseline; the build task measures none), and
    # nothing is provisioned.
    images = hold_measured(cold, TASK_ID)
    assert bool(images) == measured
    # The site override chooses the store and is named; the run override is not the provision's.
    overrides = {"site.store_root": str(cold), "run.retry.agent_build.max": "0"}
    with pytest.raises(PlanError) as e:
        plan(config, overrides, task, tmp_path, monkeypatch)
    assert not any((cold / "cache" / kind).exists() for kind in ("weights", "rows", "tools"))
    flags = "".join(f" --{k} {v}" for k, v in task.items())
    command = f"hls provision {module} --config {config} --site.store_root={cold}{flags}"
    assert str(e.value).startswith(f"{module}'s data is not all in the cache"), str(e.value)
    assert str(e.value).endswith(f": run {command} first"), str(e.value)
