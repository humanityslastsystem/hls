"""How each measured baseline image starts: the entrypoint and the runtime variables its task
module declares beside it (``IMAGES.baseline``), as B was measured with them."""

from __future__ import annotations

import importlib

import pytest

from hls.pipeline.commands import measured_artifact
from hls.recipes import catalog
from hls.recipes.catalog import ImageSet, Served
from hls.taskref import TaskRef, load_task

VLLM = (["sh", "-c", 'exec vllm serve "$HLS_WEIGHTS_DIR" "$@"', "vllm-baseline"],
        {"PYTHONPATH": "/opt/hls-src/src:/opt/hls-src", "PYTHONUNBUFFERED": "1"})
EXPECTED = {
    "multitenant_slo": VLLM,
    "tab_completion": VLLM,
    "multimodal_tts": (["python", "/app/server_vllm_omni.py"], {}),
    "multimodal_asr": (["python", "/app/server_vllm.py"], {}),
    "multimodal_omni": (["python", "/app/server_vllm_omni.py"], {}),
    "deepresearch": (["python", "/app/server.py"], {}),
    "agentic_coding": (["python3", "/app/server.py"], {"PYTHONPATH": "/app"}),
}


def test_each_measured_baseline_starts_as_it_was_measured():
    declared = {task: catalog.task_images(task).baseline for task in catalog.task_keys()}
    assert {task: (b.entrypoint, b.runtime_env) for task, b in declared.items() if b is not None} == EXPECTED


def test_a_start_names_no_unknown_field():
    with pytest.raises(ValueError, match="extra"):
        Served(image={"kind": "layer", "name": "t-baseline", "parent": "cuda", "pip_lock": "x.txt", "owner": "t",
                      "server": True}, entrypoint=["python", "/app/server.py"], serve_command=[])


@pytest.mark.parametrize("task", sorted(EXPECTED))
def test_the_measured_artifact_is_served_with_the_start_its_module_declares(task):
    loaded = load_task(TaskRef(module=catalog.task_module(task)))
    artifact, serve = measured_artifact(loaded, "baseline", task)
    assert artifact.image == catalog.task_images(task).measured["baseline"]
    assert (artifact.entrypoint, artifact.runtime_env) == EXPECTED[task]
    assert serve == list(loaded.baseline.serve_command)


def test_a_role_the_task_declares_no_image_of_is_refused():
    loaded = load_task(TaskRef(module=catalog.task_module("multimodal_asr")))
    with pytest.raises(ValueError, match=r"multimodal_asr declares no member-vllm image"):
        measured_artifact(loaded, "member-vllm", "multimodal_asr")
    with pytest.raises(ValueError, match=r"only the baseline is served"):
        measured_artifact(loaded, "reference", "multimodal_asr")


def test_a_measured_baseline_of_a_task_that_declares_no_baseline_is_refused():
    """With no ``TaskSpec.baseline`` there is no serve command the baseline was measured with, and
    serving the image with none would be a server it never was."""
    loaded = load_task(TaskRef(module=catalog.task_module("multimodal_asr")))
    with pytest.raises(ValueError, match=r"declares no baseline \(TaskSpec.baseline\)"):
        measured_artifact(loaded.model_copy(update={"baseline": None}), "baseline", "multimodal_asr")


def test_every_task_module_declares_its_images_beside_its_task():
    for task in catalog.task_keys():
        module = importlib.import_module(catalog.task_module(task))
        assert isinstance(module.IMAGES, ImageSet), task
        assert "images" not in type(module.TASK).model_fields
