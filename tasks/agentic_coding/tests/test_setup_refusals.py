"""What the harness refuses for this task before anything is planned: a calibration, and a plan
whose sandbox stores lack what the corpus needs."""

from __future__ import annotations

from pathlib import Path

import pytest

from hls.calibrate import CalibrationError, check_campaign
from tasks.agentic_coding import task as task_mod


def test_hls_calibrate_refuses_the_task_in_one_line():
    with pytest.raises(CalibrationError) as e:
        check_campaign(task_mod.TASK, engines=["sglang"], servings_per_engine=None)
    message = str(e.value)
    assert "\n" not in message
    assert ("hls calibrate does not calibrate a task whose gate's reference is a served engine (VLLMReference) "
            "and whose trajectories run in sandboxes") in message


def _settings(tmp_path):
    from hls.settings import load_settings

    stores = {name: tmp_path / name for name in ("agentic-coding-images", "agentic-coding-overlays",
                                                 "agentic-coding-prefixes", "agentic-coding-runtime")}
    (tmp_path / "secrets").mkdir()
    table = "".join(f'{name} = "{path}"\n' for name, path in stores.items())
    config = tmp_path / "hls.toml"
    config.write_text(f'[site]\npartition = "gpu"\nstore_root = "{tmp_path / "store"}"\nnode_scratch = "/tmp"\n'
                      f'secrets_path = "{tmp_path / "secrets" / "s.toml"}"\n[sandbox_stores]\n{table}')
    return load_settings(config, {}), stores


def test_a_plan_names_each_image_prefix_and_state_the_corpus_lacks(tmp_path):
    from hls.pipeline.commands import missing_store_entries

    settings, stores = _settings(tmp_path)
    pin = "ab" * 32
    rows = [{"task_name": f"t{i}"} for i in range(5)] + [{"task_name": "deep", "prefix_digest": pin}]
    assert missing_store_entries(task_mod.TASK, settings, rows) == [
        "agentic-coding-images lacks its directory, the image t0.sif, the image t1.sif and 4 more",
        "agentic-coding-overlays lacks its directory",
        "agentic-coding-prefixes lacks its directory, the prefix of deep (abababababab), the state of deep "
        "(abababababab)",
        "agentic-coding-runtime lacks its directory, the exec server's interpreter"]
    for path in stores.values():
        path.mkdir()
    for name in [f"t{i}" for i in range(5)] + ["deep"]:
        (stores["agentic-coding-images"] / f"{name}.sif").touch()
    (stores["agentic-coding-prefixes"] / "deep.prefix.json").touch()
    from hls.provision.sandbox_runtime import RUNTIME_TAG

    (stores["agentic-coding-runtime"] / RUNTIME_TAG / "bin").mkdir(parents=True)
    (stores["agentic-coding-runtime"] / RUNTIME_TAG / "bin" / "python3").touch()
    assert missing_store_entries(task_mod.TASK, settings, rows) == [
        "agentic-coding-prefixes lacks the state of deep (abababababab)"]
    (stores["agentic-coding-prefixes"] / "deep.state").mkdir()
    assert missing_store_entries(task_mod.TASK, settings, rows) == []


def test_the_shipped_defaults_fit_the_task_as_declared(tmp_path):
    """``hls run`` uses the task's defaults when they fit it (``hls.measured.find``): each is read
    by its own reader, against the task as its module declares it, and found in the defaults."""
    from hls.measured import BASELINE_CREDIT_FILE, CALIBRATION_FILE, find
    from hls.store.posix import PosixStore
    from hls.taskref import TaskRef

    ref = TaskRef(module="tasks.agentic_coding.task")
    checkout = Path(task_mod.__file__).resolve().parents[2]
    for name in (BASELINE_CREDIT_FILE, CALIBRATION_FILE):
        found = find(name, task_ref=ref, task=task_mod.TASK, task_id=task_mod.TASK.id, checkout=checkout,
                     store=PosixStore(tmp_path / "store"))
        assert found.pin is not None and found.pin.source == "defaults", found.said()


def test_a_docker_site_refuses_the_task_in_one_line(tmp_path):
    """The trajectories' sandboxes run under Apptainer on the node, so a stage that binds the
    task's sandbox stores is refused when the plan is made on a Docker site."""
    from hls.pipeline.pipelines import provision_stages, runner_stores
    from hls.pipeline.planner import PlanError, stage_plans
    from hls.spec.resources import ResourceShape
    from hls.taskref import TaskRef
    from tests.pipeline.test_seal_memory_plan import _docker

    docker = _docker(tmp_path)
    stores = runner_stores(task_mod.TASK)
    docker = docker.model_copy(update={"sandbox_stores": {name: tmp_path / "stores" / name for name in stores}})
    decls = provision_stages(task_ref=TaskRef(module="tasks.agentic_coding.task"),
                             shape=ResourceShape(cpus=1, memory_gib=16, time_sec=3600), sandbox_stores=stores)
    with pytest.raises(PlanError) as e:
        stage_plans(decls, docker, "agentic_coding", None)
    assert "\n" not in str(e.value)
    assert str(e.value) == (f"stage provision binds the sandbox stores {', '.join(sorted(stores))}, whose sandboxes "
                            "run under Apptainer on the node, which a docker site does not run")
