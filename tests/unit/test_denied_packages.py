"""Every agentic task's build denies the whole harness package list.

The build seal's index filters package names only when the producer network carries a
deny list, and a task that sets none lets every package install. These tests pin that no
agentic task in the tree is in that state, whether it takes the default producer network
or passes its own.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from hls.artifact.artifact import AgentProducedArtifact, AgentProducerSpec, Budget, HighestAvailable
from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import pep503_normalize

_REPO = Path(__file__).resolve().parents[2]
_TASK_MODULES = sorted(
    f"tasks.{p.parent.name}.{p.stem}" for p in _REPO.glob("tasks/*/task*.py"))


def _agentic_producers() -> dict[str, AgentProducerSpec]:
    producers = {}
    for name in _TASK_MODULES:
        artifact = getattr(importlib.import_module(name), "ARTIFACT", None)
        if isinstance(artifact, AgentProducedArtifact):
            producers[name] = artifact.producer
    return producers


_PRODUCERS = _agentic_producers()


def _denied(producer: AgentProducerSpec) -> set[str]:
    network = producer.network
    assert network is not None, "network=None is open egress"
    assert network.packages is not None, "no package policy: every package installs"
    assert network.packages.mode == "deny"
    return {pep503_normalize(e) for e in network.packages.entries}


def test_discovery_finds_the_agentic_tasks():
    # A glob that silently matched nothing would pass the test below vacuously.
    for name in ("tasks.deepresearch.task", "tasks.tab_completion.task",
                 "tasks.multitenant_slo.task", "tasks.sample_task.task",
                 "tasks.multimodal_asr.task", "tasks.multimodal_omni.task",
                 "tasks.multimodal_tts.task"):
        assert name in _PRODUCERS


@pytest.mark.parametrize("name", sorted(_PRODUCERS))
def test_every_agentic_task_denies_the_whole_harness_list(name):
    missing = {pep503_normalize(p) for p in DENIED_PACKAGES} - _denied(_PRODUCERS[name])
    assert not missing, f"{name} lets these install: {sorted(missing)}"


def test_the_default_producer_network_denies_the_list():
    producer = AgentProducerSpec(reasoning_effort=HighestAvailable(), hard_budget=Budget(wallclock_sec=60))
    assert _denied(producer) == {pep503_normalize(p) for p in DENIED_PACKAGES}


def test_names_are_distinct_after_normalization():
    normalized = [pep503_normalize(p) for p in DENIED_PACKAGES]
    assert len(normalized) == len(set(normalized))


@pytest.mark.parametrize("name", [
    "triton", "tilelang", "nvidia-cutlass", "nvidia-cutlass-dsl", "cuda-python", "torch",
    "transformers",
])
def test_the_toolchain_stays_installable(name):
    assert pep503_normalize(name) not in {pep503_normalize(p) for p in DENIED_PACKAGES}
