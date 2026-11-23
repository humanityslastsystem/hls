"""The task's sandbox images: one per task the corpus carries, each the release's published
environment image by the digest of its manifest, and the plan of ``hls provision`` that pulls
them into the site's store."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from hls.provision.sandbox_images import (ImageRecord, SandboxImageSet, image_file, missing_images, publish,
                                          pull_argv)
from hls.spec.resources import ResourceShape
from hls.stages.sandbox_images import SandboxImagesParams
from tasks.agentic_coding import datasets
from tasks.agentic_coding.datasets import (NEEDS_ACCELERATOR, NO_IMAGE, RELEASE_COMMIT, RELEASE_DIGESTS_FILE,
                                           RELEASE_IMAGES_FILE, TASK_NAME_FIELD, TerminalBenchError, release_images)
from tasks.agentic_coding.task import (DATASET, IMAGE_STORE, SANDBOX_IMAGE_DIR, SANDBOX_IMAGES, TOOL_RUNNER,
                                       USABLE_TASKS, sandbox_image)
from tests.pipeline.plans import CODE
from tests.provision.warmer_config import sandbox_store, task_sandbox_stores, write_config

MODULE = "tasks.agentic_coding.task"
REPOSITORY = "docker.io/harborframework/terminal-bench"


def test_the_set_names_every_task_the_corpus_carries_and_no_other():
    pinned = json.loads(RELEASE_DIGESTS_FILE.read_text())["tasks"]
    names = [image.name for image in SANDBOX_IMAGES.images]
    assert names == sorted(set(pinned) - NO_IMAGE - NEEDS_ACCELERATOR)
    assert len(names) == USABLE_TASKS == DATASET.usable_tasks() == 55
    assert not set(names) & DATASET.excluded()
    assert SANDBOX_IMAGES.store == IMAGE_STORE == "agentic-coding-images"
    # The store is the one the driver binds for the sessions' images.
    assert IMAGE_STORE in {path.store for path in TOOL_RUNNER.host_paths if path.path == SANDBOX_IMAGE_DIR}


def test_every_image_is_the_releases_published_one_by_digest():
    repository, digests = release_images()
    pinned = json.loads(RELEASE_IMAGES_FILE.read_text())
    assert repository == SANDBOX_IMAGES.repository == REPOSITORY
    assert pinned["commit"] == RELEASE_COMMIT and pinned["release"] == datasets.RELEASE_TAG
    # The file pins the image of every task of the release, each by a manifest digest.
    assert sorted(digests) == sorted(json.loads(RELEASE_DIGESTS_FILE.read_text())["tasks"])
    assert len(digests) == datasets.RELEASE_TASK_COUNT
    for image in SANDBOX_IMAGES.images:
        assert image.digest == digests[image.name]
        assert SANDBOX_IMAGES.source(image) == f"docker://{REPOSITORY}@{image.digest}"
    assert len({image.digest for image in SANDBOX_IMAGES.images}) == len(SANDBOX_IMAGES.images)


def test_pins_of_another_commit_are_refused(tmp_path, monkeypatch):
    moved = json.loads(RELEASE_IMAGES_FILE.read_text()) | {"commit": "0" * 40}
    other = tmp_path / "release_images.json"
    other.write_text(json.dumps(moved))
    monkeypatch.setattr(datasets, "RELEASE_IMAGES_FILE", other)
    with pytest.raises(TerminalBenchError, match="pins the images of 000000000000, not of the release at 452bf305c6da"):
        release_images()


def test_the_command_for_one_problem_pulls_its_published_image_by_digest():
    (image,) = [i for i in SANDBOX_IMAGES.images if i.name == "rs-archive-clone"]
    assert pull_argv(Path("/node/pulls/rs-archive-clone/rs-archive-clone.sif"), SANDBOX_IMAGES.source(image)) == [
        "apptainer", "pull", "/node/pulls/rs-archive-clone/rs-archive-clone.sif",
        "docker://docker.io/harborframework/terminal-bench"
        "@sha256:207aa8b9546ba775ea1fc50811e710139fad54a25aa8de401716efdd554b8c20"]


def test_a_session_looks_its_image_up_where_the_store_holds_it(tmp_path):
    for image in SANDBOX_IMAGES.images:
        in_driver = sandbox_image(SANDBOX_IMAGE_DIR, {TASK_NAME_FIELD: image.name})
        assert in_driver == f"{SANDBOX_IMAGE_DIR}/{image_file(tmp_path, image.name).name}"


def test_the_allowances_are_the_tasks_fixed_ones():
    assert (SANDBOX_IMAGES.cpus, SANDBOX_IMAGES.memory_gib, SANDBOX_IMAGES.disk_gib) == (16, 64, 80)
    assert (SANDBOX_IMAGES.time_sec, SANDBOX_IMAGES.pull_timeout_sec, SANDBOX_IMAGES.parallel) == (14400, 3600, 2)
    assert SandboxImageSet.model_validate_json(SANDBOX_IMAGES.model_dump_json()) == SANDBOX_IMAGES


@pytest.fixture
def login(monkeypatch, tmp_path):
    """The login node as the command sees it: a fixed checkout state and this interpreter."""
    monkeypatch.setattr(commands, "code_state", lambda path: CODE)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))


def _plan(config: Path, lines: list[str]):
    return commands.plan_provision(config=config, overrides={}, task_module=MODULE, weights=True,
                                   job_name="hls-provision", echo=lines.append)


def _hold(directory: Path, name: str, source: str) -> None:
    """Put a stand-in file for image *name* in the store at *directory* with its record."""
    directory.mkdir(parents=True, exist_ok=True)
    stand_in = directory.parent / f"{name}.stand-in"
    stand_in.write_bytes(name.encode())
    publish(directory, stand_in, ImageRecord(name=name, source=source, sha256="ab" * 32, size=len(name),
                                             apptainer="apptainer version 1.5.3"))
    stand_in.unlink()


def test_provisioning_plans_the_stage_for_a_store_that_lacks_images_and_not_for_a_full_one(tmp_path, login):
    config = write_config(tmp_path, sandbox_stores=task_sandbox_stores(MODULE))
    store = sandbox_store(tmp_path, IMAGE_STORE)
    lines: list[str] = []
    plan = _plan(config, lines)
    assert [s.name for s in plan.stages] == ["provision", "sandbox_images"]
    stage = plan.stage("sandbox_images")
    assert SandboxImagesParams.model_validate(stage.params).images == SANDBOX_IMAGES
    assert stage.shape == ResourceShape(cpus=16, memory_gib=64 + plan.settings.run.runner_memory_gib, disk_gib=80,
                                        time_sec=4 * 3600)
    assert stage.after == ["provision"] and not stage.shape.gpus
    assert f"sandbox store {IMAGE_STORE} lacks 55 of {MODULE}'s 55 sandbox images; the plan pulls them" in lines
    # All but one held: the stage is still planned, and names the whole set.
    for image in SANDBOX_IMAGES.images[:-1]:
        _hold(store, image.name, SANDBOX_IMAGES.source(image))
    assert missing_images(store, SANDBOX_IMAGES) == [SANDBOX_IMAGES.images[-1].name]
    lines = []
    assert [s.name for s in _plan(config, lines).stages] == ["provision", "sandbox_images"]
    assert f"sandbox store {IMAGE_STORE} lacks 1 of {MODULE}'s 55 sandbox images; the plan pulls them" in lines
    # Every image held: provision alone.
    last = SANDBOX_IMAGES.images[-1]
    _hold(store, last.name, SANDBOX_IMAGES.source(last))
    lines = []
    assert [s.name for s in _plan(config, lines).stages] == ["provision"]
    assert not [line for line in lines if "sandbox" in line]


def test_the_plan_time_check_and_the_provision_plan_ask_the_same_question(tmp_path):
    from hls.pipeline.commands import missing_sandbox_images, missing_store_entries
    from hls.settings import load_settings
    from tasks.agentic_coding.task import TASK

    stores = {name: tmp_path / name for name in task_sandbox_stores(MODULE)}
    (tmp_path / "secrets").mkdir()
    table = "".join(f'{name} = "{path}"\n' for name, path in stores.items())
    config = tmp_path / "hls.toml"
    config.write_text(f'[site]\npartition = "gpu"\nstore_root = "{tmp_path / "store"}"\nnode_scratch = "/tmp"\n'
                      f'secrets_path = "{tmp_path / "secrets" / "s.toml"}"\n[sandbox_stores]\n{table}')
    settings = load_settings(config, {})
    first, second = SANDBOX_IMAGES.images[:2]
    rows = [{TASK_NAME_FIELD: first.name}, {TASK_NAME_FIELD: second.name}, {TASK_NAME_FIELD: "undeclared"}]
    directory = stores[IMAGE_STORE]

    def images_lacking() -> list[str]:
        phrases = missing_store_entries(TASK, settings, rows, images=SANDBOX_IMAGES)
        return [p for p in phrases if p.startswith(f"{IMAGE_STORE} lacks")]

    directory.mkdir()
    assert images_lacking() == [f"{IMAGE_STORE} lacks the image {first.name}.sif, the image {second.name}.sif, "
                                "the image undeclared.sif"]
    # A file alone is not a held image; an image the set does not declare is there by its file.
    for name in (first.name, second.name, "undeclared"):
        image_file(directory, name).touch()
    assert images_lacking() == [f"{IMAGE_STORE} lacks the image {first.name}.sif, the image {second.name}.sif"]
    assert set(missing_sandbox_images(settings, SANDBOX_IMAGES)) >= {first.name, second.name}
    _hold(directory, first.name, SANDBOX_IMAGES.source(first))
    assert images_lacking() == [f"{IMAGE_STORE} lacks the image {second.name}.sif"]
    # A record of another source is not the declared image.
    _hold(directory, second.name, "docker://docker.io/harborframework/terminal-bench@sha256:" + "0" * 64)
    assert images_lacking() == [f"{IMAGE_STORE} lacks the image {second.name}.sif"]
    _hold(directory, second.name, SANDBOX_IMAGES.source(second))
    assert images_lacking() == []
    assert not {first.name, second.name} & set(missing_sandbox_images(settings, SANDBOX_IMAGES))
