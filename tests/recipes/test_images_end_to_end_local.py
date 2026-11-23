"""``hls images build`` on one machine: the command plans and submits, the in-process Slurm of
``tests/pipeline/test_end_to_end_local.py`` runs the real runner, and the images stage runs
its real roles against the fake ``apptainer build`` (builds and extractions) and ``exec``. Only
srun, the cluster and apptainer are stood in for."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.events import read_events
from hls.pipeline.plan import PlanStore
from hls.recipes import catalog
from hls.recipes.model import order
from hls.stages.images import IMAGE_FILE, image_keys
from hls.store.cache import ContentCache, entry_sha256
from hls.store.layout import InstanceLayout
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.slurm.scheduler import submit_argv
from hls.util import remove_tree
from hls.substrate.apptainer.images import ApptainerImages
from tests.pipeline.hls_cli import run_hls
from tests.pipeline.test_end_to_end_local import LocalSlurm
from tests.recipes import fake_build
from tests.recipes.warmer_site import (REGISTRY_SECRETS, REGISTRY_SITE, REGISTRY_TOKEN, REGISTRY_USER, inventory_for,
                                       local_slurm, site_config)
from tests.store.entries import hold_images


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    """The fake apptainer on PATH, a build log, and an inventory that holds S-A's versions."""
    bin_dir = tmp_path / "bin"
    fake_build.install(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    for knob in ("FAKE_BUILD_FAIL", "FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL"):
        monkeypatch.delenv(knob, raising=False)
    log, inventory = tmp_path / "builds.jsonl", tmp_path / "inventory.txt"
    inventory.write_text(inventory_for("s-a"))
    monkeypatch.setenv("FAKE_BUILD_LOG", str(log))
    monkeypatch.setenv("FAKE_INVENTORY", str(inventory))
    monkeypatch.setenv("FAKE_EXTRACT_LOG", str(tmp_path / "extract.log"))
    return log


def _build(tmp_path: Path, slurm: LocalSlurm, *, task: str = "sample_task", secrets: str | None = None,
           site: str = "") -> tuple[int, list[str]]:
    """``hls images build <task>``: the exit status and what it printed."""
    code, lines = run_hls(slurm, ["images", "build", task, "--job-name", "hls-images-test",
                                  "--config", str(site_config(tmp_path, site=site, secrets=secrets))])
    # The site in most of these tests has no secrets file; the command's note about that is
    # not what they check.
    return code, [line for line in lines if not line.startswith("note: no secrets file")]


def _events(store: PosixStore, instance: str) -> list[dict]:
    return read_events(store.path(InstanceLayout(instance).attempt_file("images", 1, "events.jsonl")))


def test_hls_images_build_fills_the_chain_once_and_then_finds_it(tmp_path, login, fakes):
    slurm = local_slurm()
    code, lines = _build(tmp_path, slurm)
    assert code == 0, lines
    instance = lines[0].removeprefix("instance ")
    assert instance.startswith("warmers/sample_task/")
    # The stage's own 15 min and 30 min for each of the three built images it fills: 111 min with
    # the overhead, and 30 min for its 12 roles' steps to start.
    assert lines[1] == "allocation 0: images; 0 x no GPU, 16 CPUs, 68 GiB, 141 min, partition cpu-a"
    assert lines[-1] == f"{instance}: succeeded: every stage published"
    args = submit_argv(slurm.jobs["100"].request)
    assert "--partition=cpu-a" in args and "--gpus=0" in args
    decls = order(catalog.task_images("sample_task").decls)
    keys = image_keys(decls, ApptainerImages.recipe_key)
    froms = [b["from"] for b in fake_build.read_log(fakes)]
    # The base bootstraps from the pinned CUDA image by its digest, each child from the copy of
    # its parent's file that its own build role read from the store.
    assert froms[0] == decls[0].pull_ref
    assert [(Path(f).parent.name, Path(f).name) for f in froms[1:]] == [(f"build-{d.name}", "parent.sif")
                                                                      for d in decls[1:]]
    # Each new image file was extracted once on the node, for its inventory.
    assert len((tmp_path / "extract.log").read_text().splitlines()) == 3
    store = PosixStore(tmp_path / "store")
    cache = ContentCache(store)
    manifest, _ = read_manifest(store, InstanceLayout(instance), "images")
    record = json.loads(store.read_bytes(manifest.outputs["images"].key))
    for name, key in keys.items():
        entry = cache.lookup("images", key)
        sif = next(f.sha256 for f in entry.manifest.files if f.path == IMAGE_FILE)
        assert record[name] == {"key": key, "entry_sha256": entry_sha256(entry), "sif_sha256": sif}
        assert manifest.provenance.image_digests[name] == sif

    code, again = _build(tmp_path, slurm)
    assert code == 0, again
    second = again[0].removeprefix("instance ")
    events = _events(store, second)
    assert [e for e in events if e["kind"] == "role_started"] == []
    assert [e["kind"] for e in events].count("cache_hit") == 3
    # The first plan held one build and one tree; the second found every image, so it holds none.
    disks = [PlanStore(store, InstanceLayout(i)).latest().stage("images").shape.disk_gib for i in (instance, second)]
    assert disks == [80, 0]
    assert len(fake_build.read_log(fakes)) == 3


def test_an_image_the_cache_lost_after_the_plan_fails_the_build_asking_for_a_new_plan(tmp_path, login, fakes,
                                                                                    monkeypatch):
    """The plan found every entry, so it holds no disk for a fill; an entry gone by the stage's turn
    can only end the stage, naming the entry, and a new plan counts the fill."""
    keys = image_keys(order(catalog.task_images("sample_task").decls), ApptainerImages.recipe_key)
    hold_images(tmp_path / "store", keys.values())
    real = commands.cache_holds

    def found_then_lost(settings):
        held = real(settings)

        def asked(asked_keys):
            found = held(asked_keys)
            remove_tree(PosixStore(settings.site.store_root).path(f"cache/images/{keys['s-a-tf-5-13-1-agent']}"))
            return found

        return asked

    monkeypatch.setattr(commands, "cache_holds", found_then_lost)
    code, lines = _build(tmp_path, local_slurm())
    assert code == 1
    assert "the cache no longer holds s-a-tf-5-13-1-agent, which it held when the plan was made" in lines[-1]


def test_a_registry_login_whose_user_name_every_path_holds_logs_in_and_writes_the_token_nowhere(tmp_path, login,
                                                                                                  fakes):
    # The registry user name is the cluster login name, so the checkout aside, every path the
    # roles name (the store, the node's scratch, the secrets file) lies under a directory of that
    # name, as on a cluster.
    site = tmp_path / REGISTRY_USER
    site.mkdir()
    code, lines = _build(site, local_slurm(), secrets=REGISTRY_SECRETS, site=REGISTRY_SITE)
    assert code == 0, lines
    builds = fake_build.read_log(fakes)
    # The base pulls from the registry; a layer or a cap builds from its parent's local copy.
    assert builds[0]["login"] == {"APPTAINER_DOCKER_USERNAME": hashlib.sha256(REGISTRY_USER.encode()).hexdigest(),
                                  "APPTAINER_DOCKER_PASSWORD": hashlib.sha256(REGISTRY_TOKEN.encode()).hexdigest()}
    assert [b["login"] for b in builds[1:]] == [{}, {}]
    assert all(b["registry"] == [] for b in builds)
    assert all(REGISTRY_USER in b["out"] for b in builds)
    secrets = site / "secrets.toml"
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p != secrets and REGISTRY_TOKEN.encode() in p.read_bytes()]
    assert written == []
