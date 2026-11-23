"""The build brief the agent_build stage writes: the submission section for the producer's server
file, the wall budget the arm is told, and the versions the sandbox's base holds (the constraints
of the image catalog's base under the agent cap, and none for a sandbox the catalog does not
build)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hls.prompt import budget_brief_section
from hls.recipes import catalog
from hls.recipes.catalog import ImageSet, held_versions
from hls.recipes.model import LayerImage
from hls.spec.submission import SubmissionContract
from tests.agent import build_task, test_stage
from tests.recipes.declared import declare
from tests.unit.test_prompt import ENDS_GRADED, ENDS_ZERO


def _brief(tmp_path: Path, artifact: Any, **params: Any) -> str:
    """The brief the stage renders for *artifact* with *params*, on a real stage context with the
    commands' process ceiling read first, as the stage's setup reads it."""
    return test_stage.briefing(test_stage._ctx(tmp_path, test_stage._params(**params))).brief(build_task.TASK,
                                                                                               artifact)


@pytest.mark.parametrize("soft_sec", [900, None], ids=["told", "untold"])
def test_the_stage_brief_states_the_producers_server_file_and_the_wall_budget(tmp_path, soft_sec):
    producer = build_task.ARTIFACT.producer.model_copy(update={"server_rel_path": "serve.py"})
    text = _brief(tmp_path, build_task.ARTIFACT.model_copy(update={"producer": producer}), soft_sec=soft_sec)
    assert f"\n## Submission\n{SubmissionContract(server_filename='serve.py').brief_section()}\n" in text
    assert "Write your server as `/workspace/serve.py`" in text
    budget = text.split("\n## Your budget\n", 1)[1]
    if soft_sec is None:
        assert budget.startswith("Your sandbox has ") and "get_budget" not in budget
    else:
        assert budget.startswith(f"{budget_brief_section(soft_sec)}\n\nYour sandbox has ")


@pytest.mark.parametrize("graded", [True, False], ids=["graded", "zero"])
def test_the_stage_brief_states_the_sandbox_oom_rule_its_plan_records(tmp_path, graded):
    """The brief's sandbox section states the rule of the plan's
    ``run.grade_after_sandbox_oom`` for a sandbox that ends on its memory."""
    ctx = test_stage._ctx(tmp_path, test_stage._params(), run={"grade_after_sandbox_oom": graded})
    text = test_stage.briefing(ctx).brief(build_task.TASK, build_task.ARTIFACT)
    stated, other = (ENDS_GRADED, ENDS_ZERO) if graded else (ENDS_ZERO, ENDS_GRADED)
    assert f"the sandbox ends. {stated}" in text and other not in text


def _pins(base: str) -> dict[str, str]:
    return {name: version for name, _, version in (pin.partition("==") for pin in catalog.BASES[base].constraints)}


def test_every_task_on_a_shared_layer_states_its_bases_versions():
    for task in catalog.task_keys():
        images = catalog.task_images(task)
        layer = catalog.image_set(task).layer
        if layer in catalog.LAYERS:
            assert held_versions(images) == _pins(catalog.LAYERS[layer].parent), task
    images = catalog.task_images("sample_task")
    assert list(held_versions(images)) == ["torch", "torchaudio", "torchvision"]


def test_a_tasks_own_sandbox_states_the_versions_its_package_installs_under(monkeypatch):
    own = LayerImage(name="sample-task-sandbox", parent="s0", pip_lock="x.txt", owner="sample_task")
    declare(monkeypatch, {"sample_task": ImageSet(layer=own.name, images=[own])})
    images = catalog.task_images("sample_task")
    assert images.submission_constraints is not None and held_versions(images) == _pins("s0")
    # On the cuda base, which holds nothing, the package installs under no constraints file.
    declare(monkeypatch, {"sample_task": ImageSet(layer=own.name, images=[own.model_copy(update={"parent": "cuda"})])})
    images = catalog.task_images("sample_task")
    assert images.submission_constraints is None and held_versions(images) is None

