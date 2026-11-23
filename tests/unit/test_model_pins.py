"""Every task's served model is declared at a hub commit, and each published task at the commit
it serves.

The provision stage refuses weights at a branch, so a model left at ``main`` would make its task
unprovisionable. The hub's files at each published commit equal the weights the published
task serves, byte for byte."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from hls.provision.model import is_revision
from hls.verify.calibration import workload_payload

#: The model and commit each published task's runs served.
PUBLISHED = {
    "tasks.multimodal_asr.task": ("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1"),
    "tasks.multimodal_tts.task": ("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "0c0e3051f131929182e2c023b9537f8b1c68adfe"),
    "tasks.multimodal_omni.task": ("Qwen/Qwen3-Omni-30B-A3B-Instruct", "26291f793822fb6be9555850f06dfe95f2d7e695"),
    "tasks.multitenant_slo.task": ("Qwen/Qwen3.6-27B", "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"),
    "tasks.tab_completion.task": ("JetBrains/Mellum2-12B-A2.5B-Base", "271755e48ab6b2ed0ef224eaaabe2d25275fb8ee"),
    "tasks.deepresearch.task": ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475"),
}

_TASKS = Path(__file__).resolve().parents[2] / "tasks"
#: Every task module in the repository's task folders.
MODULES = sorted(f"tasks.{p.parent.name}.{p.stem}" for p in _TASKS.glob("*/task*.py"))


@pytest.mark.parametrize("module", sorted(PUBLISHED))
def test_each_published_task_serves_the_commit_its_published_runs_served(module):
    task = importlib.import_module(module).TASK
    assert (task.model.base_checkpoint, task.model.revision) == PUBLISHED[module]
    assert workload_payload(task)["model"]["revision"] == PUBLISHED[module][1]


@pytest.mark.parametrize("module", MODULES)
def test_every_task_declares_its_model_at_a_commit(module):
    model = importlib.import_module(module).TASK.model
    assert is_revision(model.revision), f"{module}: {model.base_checkpoint} at {model.revision!r}"
