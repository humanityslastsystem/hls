"""``hls.stages.registry`` is the one place a stage is registered: the runner imports it alone to
find a stage by its kind, so the planner imports it too and refuses a plan naming a kind it does
not register, before any job is submitted."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from hls.pipeline import commands
from hls.pipeline.planner import PlanError, StageDecl
from hls.spec.resources import ResourceShape
from tests.pipeline.plans import noop_plan


def _fresh(code: str) -> str:
    """The last line *code* prints in a fresh interpreter run from this checkout."""
    root = commands.checkout_root()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root, env=env,
                         check=True, timeout=120).stdout
    return out.splitlines()[-1]


EVERY_MODULE = """
import importlib, json, pkgutil
import hls.stages, hls.stages.registry
from hls.pipeline.stage import STAGES
registered = sorted(STAGES)
for module in pkgutil.iter_modules(hls.stages.__path__, "hls.stages."):
    importlib.import_module(module.name)
print(json.dumps({"registry": registered, "every_module": sorted(STAGES)}))
"""


def test_every_stage_module_of_the_package_is_in_the_registry():
    # A stage module the registry leaves out would plan wherever it was imported for its
    # constants and be unknown to the runner, which imports the registry alone.
    found = json.loads(_fresh(EVERY_MODULE))
    assert found["registry"] == found["every_module"]
    assert {"agent_build", "deploy_drive", "finalize", "images", "noop", "oracle_fill", "package", "provision",
            "score", "verify"} <= set(found["registry"])


def test_the_stages_package_alone_registers_nothing():
    # A module imported for one stage's constants loads that stage alone.
    assert _fresh("import json, hls.stages\nfrom hls.pipeline.stage import STAGES\nprint(json.dumps(sorted(STAGES)))") \
        == "[]"


PLANNER_ONLY = """
import json, sys
from pathlib import Path
from hls.pipeline.checkout import CodeState
from hls.pipeline.plan import Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.settings import Settings
from hls.spec.resources import ResourceShape
from hls.store.layout import InstanceLayout
settings = Settings.model_validate({"site": {"partition": "gpu", "store_root": "/s",
                                             "node_scratch": "/n", "secrets_path": "/x"}})
code = CodeState(checkout=Path("/c/hls"), hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None)
shape = ResourceShape(cpus=2, memory_gib=6, time_sec=60)

def plan(kind):
    params = {"gpu_roles": 0, "cpu_role": True} if kind == "noop" else {}
    return build_plan(decls=[StageDecl("one", kind, shape, params)], task="noop", pipeline="noop",
                      layout=InstanceLayout.rollout("noop", "r1"), settings=settings, code=code,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/")),
                      job_name="x", known_stage_names={"one"}, known_tasks=(), now=1.0)

plan("noop")
try:
    plan("absent_kind")
    refused = None
except PlanError as e:
    refused = str(e)
print(json.dumps({"pipelines": "hls.pipeline.pipelines" in sys.modules, "refused": refused}))
"""


def test_the_planner_knows_every_registered_kind_as_the_runner_does_and_refuses_any_other():
    # A process that imports the planner alone, as no command does today, knows what the runner
    # knows: the registry's kinds, and nothing it registers by another import.
    found = json.loads(_fresh(PLANNER_ONLY))
    assert found["pipelines"] is False
    assert found["refused"].startswith("stage one: stage kind 'absent_kind' is not registered by "
                                       "hls.stages.registry")


def test_a_plan_naming_an_unregistered_kind_is_refused_before_any_job(tmp_path):
    shape = ResourceShape(cpus=2, memory_gib=6, time_sec=60)
    with pytest.raises(PlanError, match="stage absent: stage kind 'absent_kind' is not registered by "
                                        "hls.stages.registry, which the runner imports to find it"):
        noop_plan(tmp_path, decls=[StageDecl("absent", "absent_kind", shape)])
