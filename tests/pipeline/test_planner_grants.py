from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
from hls.pipeline.plan import Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan, toolchain_here
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
from hls.substrate.secrets import SecretsFile
from tests.pipeline import probe_stage  # noqa: F401  (registers the probe stage)
from tests.pipeline.plans import CODE, settings
from tests.substrate import fake_apptainer

TOKEN = "hf-token-0123456789"
SHAPE = ResourceShape(cpus=2, memory_gib=4, time_sec=60)


def _plan(tmp_path: Path, grants: dict[str, dict[str, str]], *, code=CODE):
    secrets = SecretsFile(secrets={"hf_token": TOKEN}, grants=grants)
    decls = [StageDecl("probe", "probe", SHAPE), StageDecl("other", "probe", SHAPE)]
    return build_plan(decls=decls, task="noop", pipeline="probe", layout=InstanceLayout.rollout("noop", "p1"),
                      settings=settings(tmp_path), code=code,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-probe", known_stage_names=KNOWN_STAGE_NAMES | {"probe", "other"}, known_tasks=(),
                      now=1000.0, secrets=secrets)


def test_each_stage_records_the_names_of_its_own_grants(tmp_path):
    plan = _plan(tmp_path, {"stage.probe": {"HF_TOKEN": "hf_token"},
                            "stage.absent": {"X_TOKEN": "hf_token"}})
    assert plan.stage("probe").grants == {"HF_TOKEN": "hf_token"}
    assert plan.stage("other").grants == {}
    assert TOKEN not in plan.model_dump_json()


@pytest.mark.parametrize("var", ["PATH", "HOME", "TMPDIR", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID",
                                 "SRUN_CPUS", "APPTAINERENV_HF_TOKEN", "SINGULARITYENV_X", "APPTAINER_BIND"])
def test_a_grant_may_not_set_a_variable_the_launcher_owns(tmp_path, var):
    with pytest.raises(PlanError, match=f"stage probe: grants set .*{var}"):
        _plan(tmp_path, {"stage.probe": {var: "hf_token"}})


def test_a_checkout_holding_a_secrets_file_copy_is_refused(tmp_path):
    checkout = tmp_path / "co"
    (checkout / "tasks").mkdir(parents=True)
    (checkout / "tasks" / ".secrets.toml.bak-1").write_text("x")
    with pytest.raises(PlanError, match=r"\.secrets\.toml\.bak-1"):
        _plan(tmp_path, {}, code=CODE.model_copy(update={"checkout": checkout}))


def test_a_clean_checkout_plans(tmp_path):
    checkout = tmp_path / "co"
    (checkout / "tasks").mkdir(parents=True)
    plan = _plan(tmp_path, {}, code=CODE.model_copy(update={"checkout": checkout}))
    assert plan.code.checkout == checkout


def _plan_with_secrets_at(tmp_path: Path, secrets_path: Path, checkout: Path):
    decls = [StageDecl("probe", "probe", SHAPE)]
    return build_plan(decls=decls, task="noop", pipeline="probe", layout=InstanceLayout.rollout("noop", "p1"),
                      settings=settings(tmp_path, site={"secrets_path": str(secrets_path)}),
                      code=CODE.model_copy(update={"checkout": checkout}),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-probe", known_stage_names=KNOWN_STAGE_NAMES | {"probe"},
                      known_tasks=(), now=1000.0)


@pytest.mark.parametrize("where, label", [("co/secrets.toml", "the checkout"), ("co/tasks/s.toml", "the checkout"),
                                          ("store/secrets.toml", "the store"), ("store/cache/x", "the store"),
                                          ("linked/secrets.toml", "the store")])
def test_a_secrets_path_inside_the_checkout_or_the_store_is_refused(tmp_path, where, label):
    # Harness roles bind the checkout and may bind the store, so a secrets file there would reach them.
    checkout = tmp_path / "co"
    (checkout / "tasks").mkdir(parents=True)
    (tmp_path / "store" / "cache").mkdir(parents=True)
    (tmp_path / "linked").symlink_to(tmp_path / "store")
    with pytest.raises(PlanError, match=re.escape(f"site.secrets_path {tmp_path / where} is in {label}")):
        _plan_with_secrets_at(tmp_path, tmp_path / where, checkout)


@pytest.mark.parametrize("root", ["store", "checkout"])
def test_a_secrets_path_is_refused_by_its_real_path_inside_a_root_named_through_a_link(tmp_path, root):
    # A site names its store through a link (as many shared-filesystem paths are), and the secrets path by the
    # real directory the link leads to.
    real = tmp_path / "real"
    (real / "tasks").mkdir(parents=True)
    (tmp_path / "via").symlink_to(real)
    site = {"secrets_path": str(real / "secrets.toml")}
    checkout = tmp_path / "co"
    checkout.mkdir()
    if root == "store":
        site["store_root"] = str(tmp_path / "via")
    else:
        checkout = tmp_path / "via"
    decls = [StageDecl("probe", "probe", SHAPE)]
    with pytest.raises(PlanError, match=re.escape(f"site.secrets_path {real / 'secrets.toml'} is in the {root}")):
        build_plan(decls=decls, task="noop", pipeline="probe", layout=InstanceLayout.rollout("noop", "p1"),
                   settings=settings(tmp_path, site=site), code=CODE.model_copy(update={"checkout": checkout}),
                   toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                   job_name="hls-probe", known_stage_names=KNOWN_STAGE_NAMES | {"probe"}, known_tasks=(), now=1000.0)


def test_a_secrets_path_beside_the_checkout_and_the_store_plans(tmp_path):
    checkout = tmp_path / "co"
    checkout.mkdir()
    plan = _plan_with_secrets_at(tmp_path, tmp_path / "co-secrets.toml", checkout)
    assert plan.settings.site.secrets_path == tmp_path / "co-secrets.toml"


def test_the_toolchain_puts_apptainer_on_the_allocations_path(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    fake_apptainer.install(bin_dir)
    monkeypatch.setenv("PATH", f"/usr/bin:/bin:{bin_dir}")
    assert str(bin_dir) in toolchain_here().path.split(":")
