from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hls import cli
from hls.pipeline import commands
from hls.pipeline.pipelines import IMAGE_FILL_SEC, IMAGES_SHAPE, images_shape
from hls.recipes import catalog
from hls.recipes.model import order
from hls.settings import load_settings
from hls.stages.images import BUILD_DISK_GIB, IMAGE_FILE, KEPT_DISK_GIB, image_keys
from hls.store.cache import ENTRY_FILE, ContentCache
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.substrate.secrets import SecretsError
from hls.substrate.apptainer.images import ApptainerImages
from tests.store.entries import populate
from tests.provision.warmer_config import write_config
from tests.recipes.warmer_site import (REGISTRY_SECRETS, REGISTRY_SITE, REGISTRY_TOKEN, REGISTRY_USER,
                                       USER_GRANTING_SECRETS, site_config)

def _plan(config: Path, *, task: str = "sample_task", overrides: dict[str, str] | None = None,
          lines: list[str] | None = None):
    return commands.plan_images_build(config=config, overrides=overrides or {}, task=task, job_name="hls-images",
                                      echo=(lines if lines is not None else []).append)


@pytest.fixture
def quiet(monkeypatch):
    """configure_logging turns propagation off on the hls loggers, which would break caplog in
    every later test of the session."""
    monkeypatch.setattr(cli, "configure_logging", lambda verbose=False: None)


def test_hls_images_build_plans_the_tasks_built_chain_on_one_cpu_allocation(tmp_path, login):
    plan = _plan(write_config(tmp_path, floor=1))
    assert plan.instance.startswith("warmers/sample_task/") and plan.pipeline == "images"
    [alloc] = plan.allocations
    assert alloc.stages == ["images"] and alloc.placement.partition == "cpu-a" and alloc.request.gpus == 0
    stage = plan.stage("images")
    names = [d["name"] for d in stage.params["images"]]
    assert names == ["s-a", "s-a-tf-5-13-1", "s-a-tf-5-13-1-agent"]
    # The cache holds none of them: the stage fills all three, one at a time, so its disk holds
    # one build and one tree; its time gives each fill its allowance.
    assert stage.shape == images_shape(3) == IMAGES_SHAPE.model_copy(
        update={"disk_gib": 80, "time_sec": IMAGES_SHAPE.time_sec + 3 * IMAGE_FILL_SEC})
    assert stage.shape.disk_gib == BUILD_DISK_GIB + KEPT_DISK_GIB
    assert stage.params["found"] == [] and alloc.request.disk_gib == 80
    assert stage.params["catalog_task"] == "sample_task"
    assert plan.warnings == []


def test_an_images_warmer_waits_for_the_sites_other_warmers_unless_they_run_at_once(tmp_path, login):
    """Slurm's singleton keys on the job name, so a warmer run one at a time is submitted under the
    site's warmer job name and waits for the others; one run at once keeps its own name."""
    config = write_config(tmp_path, floor=1)
    plan = _plan(config, overrides={"site.warmer_job_name": "hls-warm.kl"})
    assert plan.job_name == "hls-warm.kl" and plan.allocations and all(a.singleton for a in plan.allocations)
    plan = _plan(config, overrides={"site.warmer_job_name": "hls-warm.kl", "site.warmers_one_at_a_time": "false"})
    assert plan.job_name == "hls-images" and not any(a.singleton for a in plan.allocations)


def test_an_images_build_lives_where_every_warmer_of_its_task_does(tmp_path, login):
    """The catalog names ASR by its directory, ``multimodal_asr``; its TaskSpec id,
    ``multimodal-asr``, names its provision and package warmers, and so its images warmer."""
    plan = _plan(write_config(tmp_path), task="multimodal_asr")
    assert plan.task == commands.task_id_of("tasks.multimodal_asr.task") == "multimodal-asr"
    assert plan.instance.startswith("warmers/multimodal-asr/")
    stage = plan.stage("images")
    assert stage.params["catalog_task"] == "multimodal_asr"
    # The built chain, then ASR's server images on the cuda base, which it builds from its locks.
    assert [d["name"] for d in stage.params["images"]] == ["s-a", "s-a-tf-5-9-0", "s-a-tf-5-9-0-agent", "cuda",
                                                           "multimodal-asr-baseline", "multimodal-asr-reference"]


def test_a_deepresearch_images_warmer_lives_under_its_tasks_id(tmp_path, login):
    """The warmer takes the id of the task its directory's module defines, as the task's gradings
    and credit do."""
    plan = _plan(write_config(tmp_path), task="deepresearch")
    assert plan.task == commands.task_id_of("tasks.deepresearch.task") == "deepresearch"
    assert plan.instance.startswith("warmers/deepresearch/")
    assert plan.stage("images").params["catalog_task"] == "deepresearch"


def test_the_images_stage_leaves_the_runner_its_memory(tmp_path, login):
    plan = _plan(write_config(tmp_path))
    stage = plan.stage("images")
    assert stage.shape.memory_gib - plan.settings.run.runner_memory_gib == 64


def test_the_registry_user_name_is_a_plain_setting_and_the_token_reaches_the_plan_by_name_only(tmp_path, login):
    plan = _plan(site_config(tmp_path, site=REGISTRY_SITE, secrets=REGISTRY_SECRETS))
    assert plan.stage("images").grants == {"REGISTRY_PASSWORD": "registry_token"}
    assert plan.settings.site.registry_username == REGISTRY_USER
    assert REGISTRY_TOKEN not in plan.model_dump_json()


def test_a_secrets_file_that_grants_the_registry_user_name_is_refused_pointing_to_the_setting(tmp_path, login):
    with pytest.raises(SecretsError, match="REGISTRY_USERNAME: .*site.registry_username in hls.toml") as info:
        _plan(site_config(tmp_path, site=REGISTRY_SITE, secrets=USER_GRANTING_SECRETS))
    shown = "".join(traceback.format_exception(info.value))
    assert REGISTRY_USER not in shown and REGISTRY_TOKEN not in shown


def test_a_task_the_catalog_does_not_know_is_refused(tmp_path, login):
    with pytest.raises(ValueError, match="no task 'nope'"):
        _plan(write_config(tmp_path), task="nope")


def test_an_images_stages_time_follows_its_count_of_fills_alone(tmp_path, login):
    """Each fill is given ``IMAGE_FILL_SEC``, a fixed allowance, beside the stage's own steps'
    fifteen minutes, whatever the image, and a stage that fills nothing has its own steps' time
    alone."""
    assert (IMAGES_SHAPE.time_sec, IMAGE_FILL_SEC) == (900, 1800)
    config = write_config(tmp_path)
    assert _plan(config).stage("images").shape.time_sec == 900 + 3 * 1800
    _hold(config, "sample_task", "s-a")
    shape = _plan(config).stage("images").shape
    assert shape == images_shape(2) and shape.time_sec == 900 + 2 * 1800
    assert images_shape(0) == IMAGES_SHAPE.model_copy(update={"disk_gib": 0})


def _hold(config: Path, task: str, *names: str) -> dict[str, str]:
    """Publish an entry in the configured store's cache for each image *names* of catalog task
    *task*, as an earlier warmer would have; return every image's key by name."""
    keys = image_keys(order(catalog.task_images(task).decls), ApptainerImages.recipe_key)
    cache = ContentCache(PosixStore(load_settings(config).site.store_root))
    for name in names:
        populate(cache, "images", keys[name], lambda d: (d / IMAGE_FILE).write_bytes(b"image"),
                 holder=Holder.here("test"))
    return keys


def test_an_images_warmer_whose_cache_holds_every_image_asks_for_no_disk(tmp_path, login):
    config = write_config(tmp_path)
    keys = _hold(config, "sample_task", "s-a", "s-a-tf-5-13-1", "s-a-tf-5-13-1-agent")
    plan = _plan(config)
    stage = plan.stage("images")
    assert stage.params["found"] == sorted(keys.values())
    assert stage.shape == images_shape(0) and stage.shape.disk_gib == 0
    assert plan.allocations[0].request.disk_gib == 0


@pytest.mark.parametrize("held", [["s-a"], ["s-a", "s-a-tf-5-13-1"]])
def test_a_warmer_that_fills_any_entry_asks_for_one_build_and_one_tree(tmp_path, login, held):
    config = write_config(tmp_path)
    keys = _hold(config, "sample_task", *held)
    stage = _plan(config).stage("images")
    assert stage.params["found"] == sorted(keys[name] for name in held)
    assert stage.shape.disk_gib == BUILD_DISK_GIB + KEPT_DISK_GIB


def test_a_cold_multitenant_slo_warmer_asks_for_one_build_and_one_tree(tmp_path, login):
    """Seven images, filled one at a time: within the 80 GiB a node has free, where a tree kept for
    each image needed 280."""
    stage = _plan(write_config(tmp_path), task="multitenant_slo").stage("images")
    assert len(stage.params["images"]) == 7 and stage.params["found"] == []
    assert stage.shape.disk_gib == BUILD_DISK_GIB + KEPT_DISK_GIB == 80


def test_a_corrupt_entry_in_the_cache_refuses_the_plan(tmp_path, login):
    config = write_config(tmp_path)
    keys = _hold(config, "sample_task", "s-a")
    store = PosixStore(load_settings(config).site.store_root)
    store.path(f"cache/images/{keys['s-a']}/{ENTRY_FILE}").write_text("not json")
    with pytest.raises(ValueError, match=f"cache entry images/{keys['s-a']}: .* not a valid entry manifest"):
        _plan(config)


def test_only_site_and_run_overrides_are_accepted(tmp_path, login):
    with pytest.raises(ValueError, match="images.recipe"):
        _plan(write_config(tmp_path), overrides={"images.recipe": "x"})


def test_the_keys_the_plan_implies_are_the_keys_the_stage_computes(tmp_path, login):
    plan = _plan(write_config(tmp_path))
    decls = order(catalog.task_images("sample_task").decls)
    keys = image_keys(decls, ApptainerImages.recipe_key)
    assert list(keys) == ["s-a", "s-a-tf-5-13-1", "s-a-tf-5-13-1-agent"]
    assert all(key.startswith(f"{name}-") and len(key) == len(name) + 17 for name, key in keys.items())
    assert [d["name"] for d in plan.stage("images").params["images"]] == list(keys)


def test_the_cli_routes_images_build_with_its_overrides(tmp_path, monkeypatch, quiet):
    seen, started = {}, []
    monkeypatch.setattr(commands, "plan_images_build", lambda **kw: seen.update(kw) or "images plan")
    monkeypatch.setattr(commands, "start", lambda plan, overrides, **kw: started.append((plan, overrides)) or 0)
    result = CliRunner().invoke(cli.app, ["images", "build", "multitenant_slo", "--job-name", "warm",
                                          "--run.poll_sec", "5"])
    assert result.exit_code == 0, result.output
    assert (seen["task"], seen["job_name"]) == ("multitenant_slo", "warm")
    assert seen["overrides"] == {"run.poll_sec": "5"}
    assert started == [("images plan", {"run.poll_sec": "5"})]


@pytest.mark.parametrize("args, says", [
    (["--site.partition", "l40", "sample_task"], "expected the task first, got override flag '--site.partition'"),
    (["nope"], "no task 'nope'"),
])
def test_a_refusal_is_one_line(tmp_path, login, quiet, args, says):
    result = CliRunner().invoke(cli.app, ["images", "build", *args, "--config", str(write_config(tmp_path))])
    assert result.exit_code == 1 and type(result.exception) is SystemExit, result.output
    [line] = result.stderr.splitlines()
    assert line.startswith("hls images build: ") and says in line


PROBE = """
import json, sys
from pathlib import Path
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from tests.pipeline.plans import CODE
commands.code_state = lambda path: CODE
commands.toolchain_here = lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=Path("/"))
plan = commands.plan_images_build(config=Path(sys.argv[1]), overrides={}, task="multimodal_asr", job_name="x",
                                  echo=lambda line: None)
heavy = sorted(m for m in ("torch", "transformers", "datasets", "huggingface_hub", "vllm", "pyarrow")
               if m in sys.modules)
print(json.dumps({"heavy": heavy, "images": [d["name"] for d in plan.stages[0].params["images"]]}))
"""


def test_planning_an_images_build_on_the_login_node_loads_no_heavy_library(tmp_path):
    root = commands.checkout_root()
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(root / "src"), str(root)])}
    out = subprocess.run([sys.executable, "-c", PROBE, str(write_config(tmp_path))],
                         capture_output=True, text=True, cwd=root, env=env, check=True, timeout=120).stdout
    result = json.loads(out.splitlines()[-1])
    # The built chain, then the server images multimodal_asr declares for itself on the cuda base;
    # reading the task module's images loads no heavy library.
    assert result == {"heavy": [], "images": ["s-a", "s-a-tf-5-9-0", "s-a-tf-5-9-0-agent", "cuda",
                                              "multimodal-asr-baseline", "multimodal-asr-reference"]}
