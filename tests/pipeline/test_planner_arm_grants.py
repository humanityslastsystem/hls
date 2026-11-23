"""Grants a stage declares from its own parameters and the settings, as the agent build does from
its arm's route: the stage gets exactly those, each named secret must be in the secrets file, and
the file may not grant the stage as well."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import ClassVar

import pytest
from pydantic import BaseModel, ConfigDict

from hls.arms import adapter_for, route_of
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
from hls.pipeline.plan import Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.stage import Stage, StageContext, register
from hls.settings import Settings
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
from hls.substrate.secrets import SecretsFile
from tests.pipeline import probe_stage  # noqa: F401  (registers the probe stage)
from tests.pipeline.plans import CODE, settings

RELAY_TOKEN, HF = "relay-token-0123456789", "hf-token-0123456789"
#: A relay's base URL, which is a secret: the plan may name its secret, never the URL or its host.
RELAY_URL, RELAY_HOST = "https://relay.example/anthropic", "relay.example"
SHAPE = ResourceShape(cpus=2, memory_gib=4, time_sec=60)


class DeclaredParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Variable to secret name, as an arm's adapter would give them.
    grants: dict[str, str]


@register
class DeclaredStage(Stage):
    """A stage whose grants are its parameters'."""

    kind: ClassVar[str] = "declared_probe"
    params_model: ClassVar[type[BaseModel]] = DeclaredParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        pass

    @classmethod
    def declared_grants(cls, params: DeclaredParams, settings: Settings) -> dict[str, str]:
        return dict(params.grants)

    @classmethod
    def max_roles(cls, params: BaseModel) -> int:
        """It starts no role."""
        return 0

    def run(self, ctx: StageContext) -> dict[str, str]:
        return {}


class ArmParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The name of an arm of the settings.
    arm: str


@register
class ArmGrantStage(Stage):
    """A stage granted its arm's route, as the agent build is."""

    kind: ClassVar[str] = "arm_grant_probe"
    params_model: ClassVar[type[BaseModel]] = ArmParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        pass

    @classmethod
    def declared_grants(cls, params: ArmParams, settings: Settings) -> dict[str, str]:
        if params.arm not in settings.arms:
            raise ValueError(f"no arm named {params.arm!r}")
        arm = settings.arms[params.arm]
        return adapter_for(arm).grants(route_of(settings, arm))

    @classmethod
    def max_roles(cls, params: BaseModel) -> int:
        """It starts no role."""
        return 0

    def run(self, ctx: StageContext) -> dict[str, str]:
        return {}


def _arm_settings(tmp_path: Path) -> Settings:
    routes = {"relay": {"credential": "relay_token", "base_url": "relay_url", "auth": "bearer"}}
    arms = {"k3": {"harness": "claude-code", "model": "example-model", "context_window": 1_000_000, "output_limit": 64_000,
                   "route": "relay"}}
    return Settings.model_validate({**settings(tmp_path).model_dump(), "routes": routes, "arms": arms})


def _plan(tmp_path: Path, declared: dict[str, str], file_grants: dict[str, dict[str, str]] | None = None, *,
          secrets: bool = True, decl: StageDecl | None = None, plan_settings: Settings | None = None,
          relay_url: str = RELAY_URL):
    sf = (SecretsFile(secrets={"relay_token": RELAY_TOKEN, "relay_url": relay_url, "hf_token": HF}, grants=file_grants or {})
          if secrets else None)
    decls = [decl or StageDecl("agent_build", "declared_probe", SHAPE, {"grants": declared}),
             StageDecl("provision", "probe", SHAPE)]
    return build_plan(decls=decls, task="noop", pipeline="probe", layout=InstanceLayout.rollout("noop", "p1"),
                      settings=plan_settings or settings(tmp_path), code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-probe", known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=(), now=1000.0, secrets=sf)


def test_a_stage_gets_exactly_the_grants_it_declares_and_others_keep_the_files(tmp_path):
    plan = _plan(tmp_path, {"ANTHROPIC_AUTH_TOKEN": "relay_token"}, {"stage.provision": {"HF_TOKEN": "hf_token"}})
    assert plan.stage("agent_build").grants == {"ANTHROPIC_AUTH_TOKEN": "relay_token"}
    assert plan.stage("provision").grants == {"HF_TOKEN": "hf_token"}
    text = plan.model_dump_json()
    assert RELAY_TOKEN not in text and HF not in text


def test_a_stage_is_granted_its_arms_route_from_the_settings_and_the_plan_holds_no_url(tmp_path):
    decl = StageDecl("agent_build", "arm_grant_probe", SHAPE, {"arm": "k3"})
    plan = _plan(tmp_path, {}, decl=decl, plan_settings=_arm_settings(tmp_path))
    assert plan.stage("agent_build").grants == {"ANTHROPIC_AUTH_TOKEN": "relay_token", "ANTHROPIC_BASE_URL": "relay_url"}
    text = plan.model_dump_json()
    for leak in (RELAY_TOKEN, RELAY_URL, RELAY_HOST):
        assert leak not in text


def test_grants_a_stage_cannot_declare_are_refused_naming_the_stage(tmp_path):
    decl = StageDecl("agent_build", "arm_grant_probe", SHAPE, {"arm": "arm-a"})
    with pytest.raises(PlanError, match="stage agent_build: no arm named 'arm-a'"):
        _plan(tmp_path, {}, decl=decl, plan_settings=_arm_settings(tmp_path))


def test_a_declared_secret_the_file_does_not_hold_is_refused_by_name(tmp_path):
    with pytest.raises(PlanError, match=r"stage agent_build is granted secrets \['opus_key'\], which the secrets "
                                        r"file has none of") as e:
        _plan(tmp_path, {"ANTHROPIC_API_KEY": "opus_key", "X_TOKEN": "relay_token"})
    assert RELAY_TOKEN not in str(e.value)


def test_declared_grants_without_a_secrets_file_are_refused(tmp_path):
    with pytest.raises(PlanError, match="no secrets file was read"):
        _plan(tmp_path, {"ANTHROPIC_AUTH_TOKEN": "relay_token"}, secrets=False)


def test_a_stage_that_declares_its_grants_may_not_also_be_granted_by_the_file(tmp_path):
    with pytest.raises(PlanError, match=r"the secrets file may not also grant stage\.agent_build \(\['HF_TOKEN'\]\)"):
        _plan(tmp_path, {"ANTHROPIC_AUTH_TOKEN": "relay_token"}, {"stage.agent_build": {"HF_TOKEN": "hf_token"}})


def test_a_declared_grant_may_not_set_a_variable_the_launcher_owns(tmp_path):
    with pytest.raises(PlanError, match=r"stage agent_build: grants set \['PATH'\]"):
        _plan(tmp_path, {"PATH": "relay_token"})


def test_a_stage_that_declares_no_grants_gets_none_from_the_file_either(tmp_path):
    plan = _plan(tmp_path, {})
    assert plan.stage("agent_build").grants == {}


@pytest.mark.parametrize("value", ["http://relay.internal.example/anthropic", "relay.internal.example/anthropic",
                                   "https://[relay.internal.example]/x", "ftp://relay.internal.example/x"])
def test_a_base_url_secret_that_is_not_an_https_url_is_refused_when_planned(tmp_path, value):
    # Planned, the run would otherwise fail only when the agent build starts, after the queue
    # wait and the stages before it in the same allocation.
    decl = StageDecl("agent_build", "arm_grant_probe", SHAPE, {"arm": "k3"})
    with pytest.raises(PlanError, match=r"stage agent_build: the base URL secret relay_url, granted as "
                                        r"ANTHROPIC_BASE_URL, must hold an https URL with a host") as e:
        _plan(tmp_path, {}, decl=decl, plan_settings=_arm_settings(tmp_path), relay_url=value)
    shown = "".join(traceback.format_exception(e.value))
    assert value not in shown and "relay.internal.example" not in shown


@pytest.mark.parametrize("var", ["ANTHROPIC_BASE_URL", "OPENAI_BASE_URL"])
def test_every_harness_base_url_grant_is_checked_and_no_other(tmp_path, var):
    with pytest.raises(PlanError, match=rf"the base URL secret hf_token, granted as {var}, must hold an https URL"):
        _plan(tmp_path, {var: "hf_token"})
    # A credential is not a URL, and nothing asks it to be.
    assert _plan(tmp_path, {"ANTHROPIC_AUTH_TOKEN": "hf_token", var: "relay_url"}).stage("agent_build").grants == {
        "ANTHROPIC_AUTH_TOKEN": "hf_token", var: "relay_url"}
