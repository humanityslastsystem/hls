"""A stage that seals a container keeps room beside its roles for its seal's memory, which a
Docker site's gateway container holds (``site.docker.gateway_memory_gib``) and a Slurm site's
seal, inside the role's own step, does not; the plan refuses a stage sized without it."""

from __future__ import annotations

from pathlib import Path

import pytest

from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import AccessList, NetworkPolicy
from hls.pipeline.pipelines import package_stages
from hls.pipeline.planner import PlanError, stage_plans
from hls.settings import SLURM_SITE_FIELDS, RunSettings, Settings
from hls.stages.package import StoredBundle
from hls.substrate.factory import seal_memory_gib, substrate
from tests.pipeline.plans import settings

GATEWAY_GIB = 3


def _docker(tmp_path: Path) -> Settings:
    """Settings of a Docker site whose sealed roles' gateways hold ``GATEWAY_GIB``."""
    slurm = settings(tmp_path)
    site = {k: v for k, v in slurm.site.model_dump(mode="json").items() if k not in SLURM_SITE_FIELDS}
    return Settings.model_validate({
        **slurm.model_dump(mode="json"),
        "site": {**site, "substrate": "docker",
                 "docker": {"buildkit_image": "moby/buildkit@sha256:" + "b" * 64, "gateway_memory_gib": GATEWAY_GIB},
                 "local": {"state_dir": str(tmp_path / "state"), "kept_memory_gib": 8}}})


def _package(seal_gib: int, settings: Settings) -> list:
    network = NetworkPolicy(domains=AccessList(mode="allow", entries=[]),
                            packages=AccessList(mode="deny", entries=list(DENIED_PACKAGES))).model_dump(mode="json")
    return package_stages(task="t", catalog_key="sample_task",
                          bundle=StoredBundle(key="warmers/t/x/source.bundle", size=1), bundle_sha256="ab" * 32,
                          network=network, gpu_kind=None, held=lambda keys: set(),
                          recipe_key=substrate(settings.site).recipe_key, disk_cap_gib=180,
                          runner_memory_gib=RunSettings().runner_memory_gib, seal_memory_gib=seal_gib)


def test_a_slurm_sites_seal_holds_nothing_beside_its_role_and_a_docker_sites_its_gateways(tmp_path):
    assert seal_memory_gib(settings(tmp_path).site) == 0
    assert seal_memory_gib(_docker(tmp_path).site) == GATEWAY_GIB


def test_a_sealing_stage_sized_with_its_seals_memory_is_planned(tmp_path):
    docker = _docker(tmp_path)
    planned = {p.name: p for p in stage_plans(_package(GATEWAY_GIB, docker), docker, "t", None)}
    runner = docker.run.runner_memory_gib
    pip = planned["package"].params["pip"]
    assert pip["memory_gib"] == planned["package"].shape.memory_gib - runner - GATEWAY_GIB
    # The same stage on a Slurm site holds the runner's reserve alone.
    slurm = {p.name: p for p in stage_plans(_package(0, settings(tmp_path)), settings(tmp_path), "t", None)}
    assert slurm["package"].params["pip"]["memory_gib"] == pip["memory_gib"] + GATEWAY_GIB


def test_a_sealing_stage_sized_without_its_seals_memory_is_refused_at_plan_time(tmp_path):
    with pytest.raises(PlanError) as refused:
        stage_plans(_package(0, _docker(tmp_path)), _docker(tmp_path), "t", None)
    reason = str(refused.value)
    assert "\n" not in reason and reason.startswith("stage package: the plan states ")
    assert reason.endswith(f"(its seal holds {GATEWAY_GIB} GiB beside its roles on this site, so the runner's "
                           f"{RunSettings().runner_memory_gib} GiB counts as "
                           f"{RunSettings().runner_memory_gib + GATEWAY_GIB})")
