from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hls.pipeline.events import EventLog, read_events
from hls.pipeline import commands
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, NOOP_TASK
from hls.pipeline.pipelines import images_stage as plan_images_stage
from hls.pipeline.plan import Plan, Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.stage import StageContext, StageFault, run_stage
from hls.recipes import inventory as inventory_mod
from hls.recipes.inventory import BRIEF_APT, BRIEF_PACKAGES, BRIEF_TOOLS, INVENTORY_TARGET, Inventory
from hls.recipes.model import BaseImage, CapImage, LayerImage, NodePin, UvPin
from hls.recipes.render import recipe_key
from hls.settings import SLURM_SITE_FIELDS, SiteSettings
from hls.spec.resources import ResourceShape
from hls.stages import images as images_stage
from hls.stages.images import IMAGE_FILE, ImagesParams, ImagesStage
from hls.store.cache import ENTRY_FILE, ContentCache, copies_root, entry_sha256
from hls.store.digest import COPY_FAILED_EXIT, DIGEST_MISMATCH_EXIT, reported_sha256
from hls.store.layout import InstanceLayout
from hls.store.lock import GenerationLock, Holder
from hls.store.posix import STAGING, PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.apptainer.images import (BUILD_SHELL, LOGIN_SHIM, REGISTRY_TOKEN_GRANT, REGISTRY_USER_VAR,
                                            BuildMode, build_argv, build_env, build_mode)
from hls.substrate.base import RoleSpec
from hls.substrate.docker.images import DockerImages
from hls.substrate.factory import substrate
from hls.substrate.sandbox import SANDBOXES, sandbox_paths
from hls.substrate.secrets import SecretsFile
from hls.substrate.slurm.launcher import srun_argv, step_env
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import CODE, noop_plan, settings
from tests.recipes import fake_build
from tests.store.entries import entry_reads, populate, recording_reads
from tests.substrate.sandboxes import install_read_log

FIXTURES = Path(__file__).with_name("fixtures")
CUDA = "nvidia/cuda:13.0.1-devel-ubuntu24.04@sha256:" + "7" * 64
UV = UvPin(version="0.11.21", sha256="8c88519b0ef0af9801fcdee419bbb12116bd9e6b18e162ae093c932d8b264050")
BASE = BaseImage(name="s-a", cuda_ref=CUDA, snapshot="20261001T000000Z", apt_lock="apt-base.txt",
                 pip_lock="pip-base.txt", constraints=["torch==2.11.0+cu130", "torchaudio==2.11.0+cu130"],
                 env={"PYTHONUNBUFFERED": "1"})
LAYER = LayerImage(name="s-a-tf", parent="s-a", pip_lock="pip-layer.txt")
CAP = CapImage(name="s-a-tf-agent", parent="s-a-tf", cap="agent", apt_lock="apt-agent.txt",
               npm={"@anthropic-ai/claude-code": "2.1.251", "@openai/codex": "0.146.0"},
               npm_lock="npm-agent.json", node=NodePin(version="22.23.3", sha256="1" * 64))
# Disk for a build and the three images of the longest chain.
SHAPE = ResourceShape(cpus=4, memory_gib=8, disk_gib=160, time_sec=600)
RUNNER_GIB = 4
#: What every fixture image reports: curl and python3, the base's held torch and torchaudio, numpy,
#: and every tool and package the brief promises an agent cap holds, so the cap these tests build
#: keeps the brief's promises, which the stage holds it to.
INVENTORY = "\n".join([
    "apt\tcurl\t8.5.0-2ubuntu10.6", "apt\tpython3\t3.12.3-0ubuntu2", *(f"apt\t{name}\t1.0" for name in BRIEF_APT),
    "pip\ttorch\t2.11.0+cu130", "pip\ttorchaudio\t2.11.0+cu130", "pip\tnumpy\t2.5.3",
    *(f"pip\t{name}\t1.0" for name in BRIEF_PACKAGES if name != "torch"),
    *(f"tool\t{name}\t{name} 1.0" for name in BRIEF_TOOLS), "env\tPATH", ""])
#: The BuildKit image a Docker site builds with, by its digest as a site file names it.
BUILDKIT = "moby/buildkit@sha256:" + "b" * 64
#: The registry user name, a plain site setting.
REGISTRY_USER = "registry-user-01"
#: The ``[site]`` setting that names it.
LOGIN_SITE = {"registry_username": REGISTRY_USER}
#: The registry token, the login's one secret, by the variable the images stage is granted it as.
TOKEN = {"REGISTRY_PASSWORD": "registry-pass-0123456789"}
#: Both, by the variables the login shim reads.
LOGINS = {"REGISTRY_USERNAME": REGISTRY_USER, **TOKEN}
#: The build mode a build on a cluster login node reports: no /etc/subuid range, %post under fakeroot.
ROOT_MAPPED_MODE = BuildMode(user_namespace="root-mapped", post_under_fakeroot_command=True,
                       info=[line.strip() for line in fake_build.MODE_LINES["root-mapped-fakeroot"]])


@pytest.fixture
def locks(monkeypatch):
    """The fixture locks and the uv pin the stage keys and renders these tests' images with."""
    monkeypatch.setattr(images_stage, "LOCKS_DIR", FIXTURES)
    monkeypatch.setattr(images_stage, "UV", UV)


@pytest.fixture
def fakes(tmp_path, monkeypatch, locks):
    """The fake apptainer on PATH, the fixture locks and the uv pin the stage renders with, a
    build log, an extraction log and an inventory every image reports."""
    bin_dir = tmp_path / "bin"
    fake_build.install(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    for knob in ("FAKE_BUILD_FAIL", "FAKE_BUILD_MODE", "FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL",
                 "FAKE_TREE_GIB"):
        monkeypatch.delenv(knob, raising=False)
    log, inventory = tmp_path / "builds.jsonl", tmp_path / "inventory.txt"
    inventory.write_text(INVENTORY)
    monkeypatch.setenv("FAKE_BUILD_LOG", str(log))
    monkeypatch.setenv("FAKE_EXTRACT_LOG", str(tmp_path / "extracts.txt"))
    monkeypatch.setenv("FAKE_INVENTORY", str(inventory))
    monkeypatch.setattr(images_stage, "POLL_SEC", 0.1)
    return log, inventory


def _plan(tmp_path: Path, decls, *, instance: str = "r1", site: dict[str, str] | None = None,
          found: list[str] | None = None,
          shape: ResourceShape = SHAPE) -> Plan:
    """A plan of the images stage building *decls* in *shape*, having found the entries keyed
    *found* in the cache, granted the registry token when *site* sets the registry user name, as
    a site with a registry login must."""
    params = {"catalog_task": "multitenant_slo", "images": [d.model_dump() for d in decls], "found": found or []}
    login = site is not None and "registry_username" in site
    secrets = SecretsFile(secrets={"registry_pass": TOKEN["REGISTRY_PASSWORD"]},
                          grants={"stage.images": {REGISTRY_TOKEN_GRANT: "registry_pass"}}) if login else None
    return noop_plan(tmp_path, decls=[StageDecl("images", "images", shape, params)], instance=instance, site=site,
                     secrets=secrets)


def _disk(gib: int) -> ResourceShape:
    """``SHAPE`` with *gib* of node-local disk."""
    return SHAPE.model_copy(update={"disk_gib": gib})


def _ctx(plan: Plan, launcher: LocalLauncher, tmp_path: Path, *, attempt: int = 1, job: str = "100",
         secrets: dict[str, str] | None = None, deadline_sec: float = 120) -> StageContext:
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("images", attempt, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("images"), attempt=attempt,
                        launcher=launcher.use_plan(plan), watch=None, events=events, inputs={},
                        deadline=time.time() + deadline_sec, secrets=secrets or {},
                        cache=ContentCache(store, lock_ttl_sec=30), scratch=tmp_path / "node" / f"hls-{job}",
                        job_id=job, runtime=ApptainerRuntime(), images=ApptainerImages(tmp_path / "node" / f"hls-{job}"))


def _run(plan: Plan, tmp_path: Path, *, launcher: LocalLauncher | None = None,
         **kw) -> tuple[dict, LocalLauncher, StageContext]:
    launcher = launcher or LocalLauncher(gpus=0)
    ctx = _ctx(plan, launcher, tmp_path, **kw)
    outputs = ImagesStage().run(ctx)
    return json.loads(ctx.store.read_bytes(outputs["images"])), launcher, ctx


def _keys(*decls) -> dict[str, str]:
    keys: dict[str, str] = {}
    for d in decls:
        parent = getattr(d, "parent", None)
        keys[d.name] = recipe_key(d, parent_key=keys.get(parent) if parent else None, locks_dir=FIXTURES,
                                  uv=UV)
    return keys


def _published(plan: Plan) -> list[str]:
    root = PosixStore(plan.store_root).path("cache/images")
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def _meta(plan: Plan, key: str) -> dict:
    return json.loads(PosixStore(plan.store_root).path(f"cache/images/{key}/meta.json").read_text())


def _staging_left(plan: Plan) -> list[Path]:
    root = Path(plan.store_root) / STAGING / "cache" / "images"
    return [p for p in root.rglob("*") if p.is_file()] if root.exists() else []


def _roles(launcher: LocalLauncher) -> list[str]:
    """The roles started, an extraction named by its kind."""
    return ["extract" if s.name.startswith("extract-") else s.name for s in launcher.launched]


def _extracts(tmp_path: Path) -> list[str]:
    log = tmp_path / "extracts.txt"
    return log.read_text().splitlines() if log.exists() else []


def _store_image(plan: Plan, key: str) -> Path:
    """The image file of published entry *key*, in the store."""
    return PosixStore(plan.store_root).path(f"cache/images/{key}") / IMAGE_FILE


#: Paths of image files this process opened while ``_OPENED`` is a list.
_OPENED: list[str] | None = None


def _audit(event: str, args: tuple) -> None:
    if _OPENED is not None and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = os.fsdecode(args[0])
        if path.endswith(".sif"):
            _OPENED.append(path)


sys.addaudithook(_audit)


@pytest.fixture
def opened_images():
    """The image files this test process opens from now until the test ends."""
    global _OPENED
    _OPENED = []
    yield _OPENED
    _OPENED = None


def _apptainer_stand_in(path: Path) -> Path:
    """An ``apptainer`` that writes ``built`` to the image a ``build --fakeroot`` names and logs
    each call next to it."""
    path.write_text('#!/bin/sh\necho "$*" >> "$0.log"\n'
                    '[ "$1 $2" = "build --fakeroot" ] && printf built > "$3"\n')
    path.chmod(0o755)
    return path


def test_a_build_runs_apptainer_build_with_fakeroot_and_its_step_reports_the_images_digest(tmp_path):
    out, def_file, digest = tmp_path / "image.sif", tmp_path / "recipe.def", tmp_path / "image.sha256"
    argv = build_argv(def_file, out, digest=digest)
    assert argv == ["/bin/sh", "-c", BUILD_SHELL, "hls-build", "apptainer", str(out), str(def_file),
                    str(digest), "", "", ""]
    assert '"$1" build --fakeroot "$2" "$3"' in BUILD_SHELL and "--ignore-fakeroot-command" not in BUILD_SHELL
    apptainer = _apptainer_stand_in(tmp_path / "apptainer")
    done = subprocess.run(build_argv(def_file, out, digest=digest, apptainer=str(apptainer)),
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert reported_sha256(digest) == hashlib.sha256(b"built").hexdigest()
    assert build_env(cache_dir=tmp_path / "c", tmp_dir=tmp_path / "t") == {
        "APPTAINER_CACHEDIR": str(tmp_path / "c"), "APPTAINER_TMPDIR": str(tmp_path / "t")}
    for bad in ({"def_file": Path("recipe.def")}, {"out": Path("image.sif")}, {"digest": Path("d")}):
        with pytest.raises(ValueError, match="absolute"):
            build_argv(**{"def_file": def_file, "out": out, "digest": digest, **bad})
    with pytest.raises(ValueError, match="absolute"):
        build_env(cache_dir=Path("c"), tmp_dir=tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        build_env(cache_dir=tmp_path, tmp_dir=Path("t"))


def test_a_build_copies_its_parent_once_and_builds_from_the_copy_only_with_the_entrys_digest(tmp_path):
    """The step reads the parent's file once, into its own directory, hashes that copy, builds
    from it (the ``.def`` names the copy) and removes it whatever happened."""
    out, def_file, digest = tmp_path / "image.sif", tmp_path / "recipe.def", tmp_path / "image.sha256"
    parent, copy = tmp_path / "store" / "image.sif", tmp_path / "role" / "parent.sif"
    parent.parent.mkdir()
    copy.parent.mkdir()
    parent.write_bytes(b"the parent")
    apptainer = tmp_path / "apptainer"
    # It logs its arguments and whether the copy is there while it builds, and exits $FAKE_RC.
    apptainer.write_text(f'#!/bin/sh\necho "$* $([ -f {copy} ] && echo there)" >> "$0.log"\n'
                         'printf built > "$3"\nexit "${FAKE_RC:-0}"\n')
    apptainer.chmod(0o755)
    calls, reads = Path(f"{apptainer}.log"), tmp_path / "reads.txt"
    install_read_log(tmp_path / "bin", reads)

    def build(parent_sha256: str, rc: int = 0) -> subprocess.CompletedProcess:
        argv = build_argv(def_file, out, digest=digest, parent=(parent, parent_sha256, copy),
                          apptainer=str(apptainer))
        env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "FAKE_RC": str(rc)}
        return subprocess.run(argv, capture_output=True, text=True, timeout=30, env=env)

    refused = build("d" * 64)
    assert refused.returncode == DIGEST_MISMATCH_EXIT and not calls.exists() and not digest.exists()
    assert refused.stderr.startswith(f"refused: file {parent} has sha256 ")
    assert ("d" * 64) in refused.stderr and not copy.exists()
    assert reads.read_text().splitlines() == [f"cp {parent}", f"sha256sum < {copy}", f"sha256sum < {parent}"]
    # A copy that lands other bytes than a sound parent holds is the copy's failure, not a refusal.
    appending = tmp_path / "appending"
    appending.mkdir()
    real_cp = shutil.which("cp", path="/usr/bin:/bin")
    (appending / "cp").write_text(f'#!/bin/sh\n{real_cp} "$@" && printf x >> "$3"\n')
    (appending / "cp").chmod(0o755)
    wrong = subprocess.run(build_argv(def_file, out, digest=digest,
                                      parent=(parent, hashlib.sha256(b"the parent").hexdigest(), copy),
                                      apptainer=str(apptainer)),
                           capture_output=True, text=True, timeout=30,
                           env={"PATH": f"{appending}:/usr/bin:/bin"})
    assert wrong.returncode == COPY_FAILED_EXIT and "refused" not in wrong.stderr
    assert not calls.exists() and not digest.exists() and not copy.exists()
    parent.unlink()
    missing = build("d" * 64)
    assert missing.returncode == 74 and not calls.exists() and not copy.exists()
    parent.write_bytes(b"the parent")
    done = build(hashlib.sha256(b"the parent").hexdigest())
    assert done.returncode == 0 and calls.read_text() == f"build --fakeroot {out} {def_file} there\n"
    assert reported_sha256(digest) == hashlib.sha256(b"built").hexdigest() and not copy.exists()
    digest.unlink()
    failed = build(hashlib.sha256(b"the parent").hexdigest(), rc=5)
    assert failed.returncode == 5 and not digest.exists() and not copy.exists()
    with pytest.raises(ValueError, match="not a bare hex"):
        build_argv(def_file, out, digest=digest, parent=(parent, "not hex", copy))
    with pytest.raises(ValueError, match="absolute"):
        build_argv(def_file, out, digest=digest, parent=(parent, "d" * 64, Path("parent.sif")))


def test_a_failed_build_reports_no_digest_even_when_it_wrote_part_of_the_image(tmp_path):
    out, def_file, digest = tmp_path / "image.sif", tmp_path / "recipe.def", tmp_path / "image.sha256"
    apptainer = tmp_path / "apptainer"
    apptainer.write_text('#!/bin/sh\nprintf partial > "$3"\nexit 3\n')
    apptainer.chmod(0o755)
    done = subprocess.run(build_argv(def_file, out, digest=digest, apptainer=str(apptainer)),
                          capture_output=True, text=True, timeout=30)
    assert done.returncode == 3 and out.read_bytes() == b"partial" and not digest.exists()


@pytest.mark.parametrize("text", ["", "not a digest\n", "z" * 64 + "  /x\n"])
def test_a_role_that_reported_no_digest_reads_as_none(tmp_path, text):
    (tmp_path / "d").write_text(text)
    assert reported_sha256(tmp_path / "d") is None and reported_sha256(tmp_path / "absent") is None


@pytest.mark.parametrize("mode, namespace, fakeroot", [
    ("root-mapped-fakeroot", "root-mapped", True), ("root-mapped", "root-mapped", False),
    ("fakeroot-only", "none", True), ("subuid", "unknown", None),
])
def test_the_build_mode_is_read_from_apptainers_info_lines(mode, namespace, fakeroot):
    err = "".join(f"INFO:    {line}\n" for line in [*fake_build.MODE_LINES[mode], "Starting build..."])
    found = build_mode(err)
    assert (found.user_namespace, found.post_under_fakeroot_command) == (namespace, fakeroot)
    assert found.info == [line.strip() for line in fake_build.MODE_LINES[mode]]


def test_a_build_mode_no_info_line_names_is_unknown():
    assert build_mode("") == BuildMode(user_namespace="unknown", post_under_fakeroot_command=None, info=[])


def test_a_cold_chain_fills_base_layer_and_cap_in_order_and_records_their_digests(tmp_path, fakes):
    log, _ = fakes
    plan = _plan(tmp_path, [CAP, LAYER, BASE])
    record, launcher, ctx = _run(plan, tmp_path)
    keys = _keys(BASE, LAYER, CAP)
    assert {name: r["key"] for name, r in record.items()} == keys
    assert _published(plan) == sorted(keys.values())
    # A child builds on the copy its build role made of its parent's file in the store, which
    # the role removed; no entry is copied to the node.
    copies = [str(ctx.scratch / "roles" / f"build-{name}" / "parent.sif")
              for name in ("s-a-tf", "s-a-tf-agent")]
    assert [b["from"] for b in fake_build.read_log(log)] == [BASE.pull_ref, *copies]
    assert not list(ctx.scratch.rglob("parent.sif"))
    assert not (ctx.scratch / ctx.job_id).exists()
    cache = ContentCache(PosixStore(plan.store_root))
    for name, key in keys.items():
        entry = cache.lookup("images", key)
        files = {f.path: f.sha256 for f in entry.manifest.files}
        assert set(files) == {IMAGE_FILE, "recipe.def", "inventory.json", "meta.json"}
        assert ctx.image_digests[name] == record[name]["sif_sha256"] == files[IMAGE_FILE]
        assert record[name]["entry_sha256"] == entry_sha256(entry)
        assert entry.manifest.source == {"image": name, "stage": "images", "instance": plan.instance}
        assert Inventory.model_validate_json((entry.path / "inventory.json").read_bytes()).pip["torch"] == \
            "2.11.0+cu130"
        parent = {"s-a": None, "s-a-tf": "s-a", "s-a-tf-agent": "s-a-tf"}[name]
        kind = {"s-a": "base", "s-a-tf": "layer", "s-a-tf-agent": "cap"}[name]
        assert _meta(plan, key) == {"name": name, "kind": kind, "key": key, "parent": parent,
                                    "parent_key": keys[parent] if parent else None,
                                    "build_mode": ROOT_MAPPED_MODE.model_dump()}
    # Only the parents a child built on were read, each pinned to its entry digest.
    assert ctx.cache_entries == {f"images/{keys['s-a']}": record["s-a"]["entry_sha256"],
                                 f"images/{keys['s-a-tf']}": record["s-a-tf"]["entry_sha256"]}
    assert {s.owner for s in launcher.launched} == {"harness"}
    assert _roles(launcher) == [
        "build-s-a", "extract", "inventory-s-a", "publish-s-a", "build-s-a-tf", "extract", "inventory-s-a-tf",
        "publish-s-a-tf", "build-s-a-tf-agent", "extract", "inventory-s-a-tf-agent", "publish-s-a-tf-agent"]
    # Only the build and pin roles speak to a container daemon.
    assert [s.name.split("-")[0] for s in launcher.launched if s.daemon] == ["build", "extract"] * 3
    assert _staging_left(plan) == []
    kinds = [e["kind"] for e in read_events(ctx.events.path) if e["kind"].startswith(("cache_", "image_"))]
    # Each tree goes once its image is inventoried, before its entry publishes.
    assert kinds == ["image_extracted", "image_removed", "cache_filled"] * 3


def test_the_build_mode_goes_into_meta_and_never_into_the_key(tmp_path, fakes, monkeypatch):
    monkeypatch.setenv("FAKE_BUILD_MODE", "subuid")
    plan = _plan(tmp_path, [BASE])
    record, _, _ = _run(plan, tmp_path)
    assert record["s-a"]["key"] == _keys(BASE)["s-a"]
    assert _meta(plan, record["s-a"]["key"])["build_mode"] == {
        "user_namespace": "unknown", "post_under_fakeroot_command": None, "info": []}


def test_each_build_keeps_its_temporary_files_and_apptainer_cache_in_its_own_directory(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE, LAYER])
    ctx = _ctx(plan, LocalLauncher(gpus=0), tmp_path)
    run_stage(ImagesStage(), ctx)
    for spec in ctx.launcher.launched:
        if spec.name.startswith(("build-", "extract-")):
            role = ctx.scratch / "roles" / spec.name
            assert {Path(v).parent for v in spec.env.values()} == {role}, spec.name
    # The fake apptainer writes into its cache as a pull would; the stage's close took it all.
    assert not list(ctx.scratch.rglob("apptainer-cache")) and not list(ctx.scratch.rglob("cache"))
    assert not list((ctx.scratch / "roles").iterdir())


def test_an_image_file_leaves_its_roles_directory_once_it_is_published(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE, LAYER])
    _, _, ctx = _run(plan, tmp_path)
    for role in ("build-s-a", "build-s-a-tf"):
        own = ctx.scratch / "roles" / role
        assert own.is_dir() and not (own / IMAGE_FILE).exists()


def test_a_build_whose_log_cannot_be_read_keeps_its_image_with_an_unknown_mode(tmp_path, fakes, monkeypatch):
    real = images_stage._Images._wait

    def losing_the_log(self):
        real(self)
        for err in (self.ctx.attempt_dir / "logs").glob("build-*.err"):
            err.unlink()

    monkeypatch.setattr(images_stage._Images, "_wait", losing_the_log)
    plan = _plan(tmp_path, [BASE])
    record, _, _ = _run(plan, tmp_path)
    assert _meta(plan, record["s-a"]["key"])["build_mode"] == {
        "user_namespace": "unknown", "post_under_fakeroot_command": None, "info": []}


def test_the_inventory_runs_in_a_harness_container_on_the_extracted_image(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE])
    record, launcher, ctx = _run(plan, tmp_path)
    spec = next(s for s in launcher.launched if s.name == "inventory-s-a")
    assert spec.owner == "harness" and spec.argv[:2] == ("apptainer", "exec")
    assert f"{inventory_mod.SCRIPT}:{INVENTORY_TARGET}:ro" in spec.argv
    assert spec.argv[-2:] == ("/bin/sh", INVENTORY_TARGET)
    # The image is the extraction of the very file the entry publishes, keyed by its digest, and
    # its tree and manifest go once the inventory is read.
    root, manifest = sandbox_paths(ctx.scratch, record["s-a"]["sif_sha256"])
    assert str(root) in spec.argv and not os.path.lexists(root) and not os.path.lexists(manifest)
    extract = next(s for s in launcher.launched if s.name.startswith("extract-"))
    assert extract.owner == "harness" and extract.argv[:2] == ("/bin/sh", "-c")
    # It extracts the file the build wrote, after checking it has the digest the build reported.
    built = ctx.scratch / "roles" / "build-s-a" / IMAGE_FILE
    assert extract.argv[4:7] == (str(built), record["s-a"]["sif_sha256"], "apptainer")
    assert extract.shape.gpus == 0 and extract.shape.memory_gib == SHAPE.memory_gib - RUNNER_GIB


def _left_on_node(ctx: StageContext) -> list[str]:
    """The extracted trees, their manifests and the image entry copies on the job's node."""
    assert ctx.scratch is not None
    sandboxes, copies = ctx.scratch / SANDBOXES, copies_root(ctx.scratch, ctx.job_id) / "images"
    return sorted(p.name for d in (sandboxes, copies) if d.exists() for p in d.iterdir())


def test_a_later_stage_of_the_job_pins_a_published_image_and_extracts_it_from_the_store(tmp_path, fakes):
    """The images stage's trees live until its context closes. A later stage on the same node
    pins an image by the digest the images stage recorded, and extracts it again, straight from
    the entry's image file in the store."""
    plan = _plan(tmp_path, [BASE, LAYER])
    ctx = _ctx(plan, LocalLauncher(gpus=0), tmp_path)
    record = json.loads(ctx.store.read_bytes(run_stage(ImagesStage(), ctx)["images"]))
    assert len(_extracts(tmp_path)) == 2 and ctx.cache_entries == {f"images/{record['s-a']['key']}":
                                                                    record["s-a"]["entry_sha256"]}
    assert _left_on_node(ctx) == []
    kinds = [e["kind"] for e in read_events(ctx.events.path) if e["kind"].startswith("image_")]
    assert kinds == ["image_extracted", "image_removed"] * 2
    launcher = LocalLauncher(gpus=0)
    later = _ctx(plan, launcher, tmp_path, attempt=2)
    pinned = later.node_image(record["s-a"]["key"], record["s-a"]["entry_sha256"])
    assert pinned.file_sha256 == record["s-a"]["sif_sha256"]
    assert Path(pinned.ref) == sandbox_paths(ctx.scratch, pinned.file_sha256)[0]
    [extract] = launcher.launched
    assert extract.name == f"extract-{pinned.file_sha256[:12]}"
    assert extract.argv[4:6] == (str(_store_image(plan, record["s-a"]["key"])), pinned.file_sha256)
    # It extracted from its own copy of the entry's file.
    assert _extracts(tmp_path)[-1] == str(later.scratch / "roles" / extract.name / IMAGE_FILE)
    assert len(_extracts(tmp_path)) == 3
    assert [e["kind"] for e in read_events(later.events.path) if e["kind"].startswith("image_")] == [
        "image_extracted"]
    assert later.cache_entries == {f"images/{record['s-a']['key']}": record["s-a"]["entry_sha256"]}
    assert _left_on_node(later) == [pinned.file_sha256, f"{pinned.file_sha256}.json"]


def test_a_tampered_image_file_in_the_store_is_refused_before_it_is_extracted(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE])
    record, _, _ = _run(plan, tmp_path)
    stored = _store_image(plan, record["s-a"]["key"])
    stored.write_bytes(stored.read_bytes() + b"x")
    launcher = LocalLauncher(gpus=0)
    later = _ctx(plan, launcher, tmp_path, attempt=2, job="101")
    refusal = f"file {stored} has sha256 .*, not {record['s-a']['sif_sha256']}"
    with pytest.raises(StageFault, match=refusal) as caught:
        later.node_image(record["s-a"]["key"], record["s-a"]["entry_sha256"])
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _roles(launcher) == ["extract"] and len(_extracts(tmp_path)) == 1 and _left_on_node(later) == []


def test_a_layer_whose_parents_file_in_the_store_was_tampered_is_refused_before_it_builds(tmp_path, fakes):
    log, _ = fakes
    plan = _plan(tmp_path, [BASE])
    record, _, _ = _run(plan, tmp_path)
    stored = _store_image(plan, record["s-a"]["key"])
    stored.write_bytes(stored.read_bytes() + b"x")
    builds = len(fake_build.read_log(log))
    plan = _plan(tmp_path, [BASE, LAYER], instance="r2")
    launcher = LocalLauncher(gpus=0)
    refusal = f"file {stored} has sha256 .*, not {record['s-a']['sif_sha256']}"
    with pytest.raises(StageFault, match=refusal) as caught:
        _run(plan, tmp_path, launcher=launcher, job="101")
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert caught.value.fault.evidence["role"] == "build-s-a-tf"
    assert len(fake_build.read_log(log)) == builds and _roles(launcher) == ["build-s-a-tf"]
    assert _published(plan) == [record["s-a"]["key"]] and _staging_left(plan) == []


def _appending_cp(tmp_path: Path, under: Path) -> None:
    """Put first on PATH a ``cp`` that appends a byte to every copy of a file under *under*, as a
    copy that went wrong on the way would."""
    cp = tmp_path / "bin" / "cp"
    real = shutil.which("cp", path="/usr/bin:/bin")
    cp.write_text(f'#!/bin/sh\n{real} "$@" || exit\n'
                  f'case "$2" in {shlex.quote(str(under))}/*) printf x >> "$3";; esac\n')
    cp.chmod(0o755)


def test_a_layers_copy_of_a_sound_parent_gone_wrong_on_the_way_is_retried(tmp_path, fakes):
    """The parent's file in the store matches its entry, so a copy that differs went wrong on the
    way: the build role exits with its own status and no refusal, and a retry copies it again."""
    log, _ = fakes
    plan = _plan(tmp_path, [BASE])
    record, _, _ = _run(plan, tmp_path)
    stored = _store_image(plan, record["s-a"]["key"])
    builds = len(fake_build.read_log(log))
    _appending_cp(tmp_path, stored.parent)
    plan = _plan(tmp_path, [BASE, LAYER], instance="r2")
    with pytest.raises(StageFault, match="harness role build-s-a-tf ended exited with status 75") as caught:
        _run(plan, tmp_path, job="101")
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert hashlib.sha256(stored.read_bytes()).hexdigest() == record["s-a"]["sif_sha256"]
    assert len(fake_build.read_log(log)) == builds and _published(plan) == [record["s-a"]["key"]]


def test_a_child_build_reads_its_parents_file_from_the_store_once_into_its_own_directory(tmp_path, fakes,
                                                                                         monkeypatch):
    log, _ = fakes
    record, _, _ = _run(_plan(tmp_path, [BASE]), tmp_path)
    reads = tmp_path / "reads.txt"
    install_read_log(tmp_path / "read-bin", reads)
    monkeypatch.setenv("PATH", f"{tmp_path / 'read-bin'}:{os.environ['PATH']}")
    plan = _plan(tmp_path, [BASE, LAYER], instance="r2")
    _, _, ctx = _run(plan, tmp_path, job="101")
    stored = _store_image(plan, record["s-a"]["key"])
    copy = ctx.scratch / "roles" / "build-s-a-tf" / "parent.sif"
    lines = reads.read_text().splitlines()
    assert [line for line in lines if line.endswith(str(stored))] == [f"cp {stored}"]
    assert f"sha256sum < {copy}" in lines and not copy.exists()
    # Every role hashes what it reads on its standard input, so no file name is parsed.
    assert all(line.startswith(("cp ", "sha256sum < ")) for line in lines)
    assert fake_build.read_log(log)[-1]["from"] == str(copy)


def test_a_publish_copy_that_lands_other_bytes_is_never_published(tmp_path, fakes):
    """The publish role hashes the copy it wrote, not the file it read: a copy that lands other
    bytes than were inventoried is refused, and the entry is never published."""
    plan = _plan(tmp_path, [BASE])
    cp = tmp_path / "bin" / "cp"
    store = shlex.quote(str(plan.store_root))
    # The destination is the last argument.
    cp.write_text(f'#!/bin/sh\n{shutil.which("cp", path="/usr/bin:/bin")} "$@" || exit\n'
                  'for dest; do :; done\n'
                  f'case "$dest" in {store}/*) printf x >> "$dest";; esac\n')
    cp.chmod(0o755)
    with pytest.raises(StageFault, match="not the .* of the file that was inventoried") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.retryable and _published(plan) == [] and _staging_left(plan) == []


def test_an_images_entry_records_its_image_file_with_the_mtime_its_build_left(tmp_path, fakes, monkeypatch):
    """The publish role copies the build's file into the entry keeping its time, which the entry
    records, as every fill keeps the times of the tree its role left."""
    built_at = 1_700_000_000_123_456_789
    real = images_stage._Images._copy

    def copying_an_old_file(self, name, source, dest):
        os.utime(source, ns=(built_at, built_at))
        return real(self, name, source, dest)

    monkeypatch.setattr(images_stage._Images, "_copy", copying_an_old_file)
    plan = _plan(tmp_path, [BASE])
    record, _, _ = _run(plan, tmp_path)
    entry = ContentCache(PosixStore(plan.store_root)).lookup("images", record["s-a"]["key"])
    [sif] = [f for f in entry.manifest.files if f.path == IMAGE_FILE]
    assert sif.mtime_ns == built_at == os.lstat(entry.path / IMAGE_FILE).st_mtime_ns


def test_the_runner_never_opens_an_image_file_or_reads_any_file_of_an_images_entry(tmp_path, fakes,
                                                                                      opened_images):
    """Every image file is written, hashed, copied and extracted by roles: the stage's own
    process opens none, for a build, a publish or a later stage's pin. The small files it writes
    into an entry itself, the recipe, the inventory and the meta, it reports by the bytes it wrote
    and never reads back."""
    plan = _plan(tmp_path, [BASE, LAYER, CAP])
    ctx = _ctx(plan, LocalLauncher(gpus=0), tmp_path)
    with recording_reads() as reads:
        record = json.loads(ctx.store.read_bytes(run_stage(ImagesStage(), ctx)["images"]))
        later = _ctx(plan, LocalLauncher(gpus=0), tmp_path, attempt=2)
        for name in record:
            later.node_image(record[name]["key"], record[name]["entry_sha256"])
        later.close()
    assert len(_extracts(tmp_path)) == 6 and opened_images == []
    store = PosixStore(plan.store_root)
    assert entry_reads(reads, [store.root / "cache", store.root / STAGING / "cache"]) == []
    for name in record:
        entry = ContentCache(store).lookup("images", record[name]["key"])
        files = {f.path: f.sha256 for f in entry.manifest.files}
        assert all(files[p] == hashlib.sha256((entry.path / p).read_bytes()).hexdigest() for p in files)


def test_a_failed_run_of_the_stage_leaves_no_tree(tmp_path, fakes, monkeypatch):
    by_image = tmp_path / "inventories"
    by_image.mkdir()
    (by_image / "default.txt").write_text(INVENTORY)
    (by_image / "build-s-a-tf.txt").write_text(INVENTORY + "pip\tvllm\t0.22.1\n")
    monkeypatch.setenv("FAKE_INVENTORY", str(by_image))
    plan = _plan(tmp_path, [BASE, LAYER])
    ctx = _ctx(plan, LocalLauncher(gpus=0), tmp_path)
    with pytest.raises(StageFault, match="vllm"):
        run_stage(ImagesStage(), ctx)
    assert len(_extracts(tmp_path)) == 2 and _left_on_node(ctx) == []


def test_a_second_run_finds_every_image_and_starts_no_role(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE, LAYER, CAP])
    first, _, _ = _run(plan, tmp_path)
    second, launcher, ctx = _run(plan, tmp_path, attempt=2, job="101")
    assert second == first and launcher.launched == []
    assert ctx.image_digests == {name: r["sif_sha256"] for name, r in first.items()}
    kinds = [e["kind"] for e in read_events(ctx.attempt_dir / "events.jsonl")]
    assert kinds.count("cache_hit") == 3 and "cache_filled" not in kinds


def test_the_cache_says_which_entries_it_holds_and_refuses_one_with_no_image_file(tmp_path):
    """The planner asks which of an images stage's keys the cache holds, so the stage counts only
    the entries it fills; an entry that holds no image file is no image the stage would find."""
    cache = ContentCache(PosixStore(tmp_path / "store"))
    populate(cache, "images", "held-key", lambda d: (d / IMAGE_FILE).write_bytes(b"x" * 1234),
             holder=Holder.here("test"))
    assert images_stage.held_keys(cache, ["held-key", "lacked-key"], image_file=IMAGE_FILE) == {"held-key"}
    populate(cache, "images", "no-image-key", lambda d: (d / "other.bin").write_bytes(b"x"), holder=Holder.here("test"))
    with pytest.raises(ValueError, match=f"cache entry images/no-image-key holds no {IMAGE_FILE}"):
        images_stage.held_keys(cache, ["no-image-key"], image_file=IMAGE_FILE)


def test_an_entry_whose_records_carry_no_mtimes_is_not_held_and_is_built_again(tmp_path, fakes):
    """An entry whose file records carry no times is not one: the planner counts its
    fill, and the stage builds the image again in its place."""
    log, _ = fakes
    plan = _plan(tmp_path, [BASE])
    first, _, _ = _run(plan, tmp_path)
    key = first["s-a"]["key"]
    store = PosixStore(plan.store_root)
    mf = store.path(f"cache/images/{key}") / ENTRY_FILE
    manifest = json.loads(mf.read_text())
    mf.write_text(json.dumps({**manifest, "files": [{k: v for k, v in f.items() if k != "mtime_ns"}
                                                    for f in manifest["files"]]}))
    cache = ContentCache(store)
    assert images_stage.held_keys(cache, [key], image_file=IMAGE_FILE) == set()
    builds = len(fake_build.read_log(log))
    again, launcher, _ = _run(_plan(tmp_path, [BASE], instance="r2"), tmp_path, job="101")
    assert _roles(launcher) == ["build-s-a", "extract", "inventory-s-a", "publish-s-a"]
    assert len(fake_build.read_log(log)) == builds + 1
    entry = cache.lookup("images", key)
    assert entry is not None and entry_sha256(entry) == again["s-a"]["entry_sha256"] != first["s-a"]["entry_sha256"]
    assert all(f.mtime_ns == os.lstat(entry.path / f.path).st_mtime_ns for f in entry.manifest.files)


def test_a_child_miss_copies_only_its_parent_and_builds_only_itself(tmp_path, fakes):
    log, _ = fakes
    _run(_plan(tmp_path, [BASE, LAYER]), tmp_path)
    builds = len(fake_build.read_log(log))
    record, launcher, ctx = _run(_plan(tmp_path, [BASE, LAYER, CAP], instance="r2"), tmp_path, job="101")
    keys = _keys(BASE, LAYER, CAP)
    assert sorted(ctx.cache_entries) == [f"images/{keys['s-a-tf']}"]
    assert _roles(launcher) == ["build-s-a-tf-agent", "extract", "inventory-s-a-tf-agent",
                                "publish-s-a-tf-agent"]
    assert len(fake_build.read_log(log)) == builds + 1
    assert set(record) == {"s-a", "s-a-tf", "s-a-tf-agent"}


@pytest.mark.parametrize("line, match", [
    ("pip\tvllm\t0.22.1", "vllm"),
    ("pip\tFlash_Attn\t2.8.3", "flash-attn"),
])
def test_a_denied_package_refuses_the_base_and_publishes_nothing(tmp_path, fakes, line, match):
    _, inventory = fakes
    inventory.write_text(INVENTORY + line + "\n")
    plan = _plan(tmp_path, [BASE, LAYER])
    with pytest.raises(StageFault, match=match) as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _published(plan) == [] and _staging_left(plan) == []


BARE = BaseImage(name="cuda", cuda_ref=CUDA, snapshot="20261001T000000Z", apt_lock="apt-base.txt",
                 env={"PYTHONUNBUFFERED": "1"})
SERVER = LayerImage(name="t-baseline", parent="cuda", pip_lock="pip-layer.txt", owner="t", server=True,
                    files={"/app/server.py": "server.py"})


def test_a_server_layer_may_hold_its_serving_framework(tmp_path, fakes, monkeypatch):
    """A task's server layer holds vLLM by design, on the base that holds no torch; the same
    package in a base still refuses it."""
    tasks = tmp_path / "tasks"
    (tasks / "t" / "locks").mkdir(parents=True)
    shutil.copyfile(FIXTURES / "pip-layer.txt", tasks / "t" / "locks" / "pip-layer.txt")
    (tasks / "t" / "server.py").write_text("print('serving')\n")
    monkeypatch.setattr(images_stage, "TASKS_DIR", tasks)
    by_image = tmp_path / "inventories"
    by_image.mkdir()
    (by_image / "default.txt").write_text(INVENTORY)
    for build in ("build-t-baseline", "build-s-a"):
        (by_image / f"{build}.txt").write_text(INVENTORY + "pip\tvllm\t0.22.1\n")
    monkeypatch.setenv("FAKE_INVENTORY", str(by_image))
    plan = _plan(tmp_path, [BARE, SERVER])
    record, _, _ = _run(plan, tmp_path)
    assert set(record) == {"cuda", "t-baseline"}
    assert _meta(plan, record["t-baseline"]["key"])["kind"] == "layer"
    with pytest.raises(StageFault, match="vllm"):
        _run(_plan(tmp_path / "with-a-base", [BASE]), tmp_path / "with-a-base")


def test_an_agent_cap_on_a_server_layer_is_refused_when_the_plan_is_made(tmp_path):
    cap = CAP.model_copy(update={"parent": "t-baseline"})
    params = {"catalog_task": "t", "images": [d.model_dump() for d in (BARE, SERVER, cap)]}
    with pytest.raises(PlanError, match="image s-a-tf-agent builds on server image t-baseline"):
        noop_plan(tmp_path, decls=[StageDecl("images", "images", SHAPE, params)])


def test_a_submission_base_on_a_server_layer_is_refused_when_the_plan_is_made(tmp_path):
    """A package plan's images hold no agent cap: the layer a submission is installed into is
    refused itself when it builds on a server image."""
    own = LayerImage(name="t-own", parent="t-baseline", pip_lock="pip-layer.txt", owner="t")
    params = {"catalog_task": "t", "images": [d.model_dump() for d in (BARE, SERVER, own)]}
    with pytest.raises(PlanError, match="image t-own builds on server image t-baseline"):
        noop_plan(tmp_path, decls=[StageDecl("images", "images", SHAPE, params)])


def test_a_layer_that_moves_a_held_version_is_refused(tmp_path, fakes, monkeypatch):
    by_image = tmp_path / "inventories"
    by_image.mkdir()
    (by_image / "default.txt").write_text(INVENTORY)
    moved = INVENTORY.replace("torch\t2.11.0+cu130", "torch\t2.13.0+cu130")
    (by_image / "build-s-a-tf.txt").write_text(moved)
    monkeypatch.setenv("FAKE_INVENTORY", str(by_image))
    plan = _plan(tmp_path, [BASE, LAYER])
    with pytest.raises(StageFault, match="torch") as caught:
        _run(plan, tmp_path)
    assert not caught.value.fault.retryable
    assert _published(plan) == [_keys(BASE)["s-a"]] and _staging_left(plan) == []


def _without(text: str, start: str) -> str:
    """The inventory *text* without its record that starts with *start*."""
    kept = [line for line in text.split("\n") if not line.startswith(start)]
    assert len(kept) == len(text.split("\n")) - 1, f"no single record starts with {start!r}"
    return "\n".join(kept)


def _inventories(tmp_path: Path, monkeypatch, **by_build: str) -> None:
    """``INVENTORY`` for every image, and *by_build*'s text for the image each build role (``build_s_a_tf``
    for ``build-s-a-tf``) wrote."""
    by_image = tmp_path / "inventories"
    by_image.mkdir()
    (by_image / "default.txt").write_text(INVENTORY)
    for role, text in by_build.items():
        (by_image / f"{role.replace('_', '-')}.txt").write_text(text)
    monkeypatch.setenv("FAKE_INVENTORY", str(by_image))


def test_an_agent_cap_missing_a_promised_tool_is_refused(tmp_path, fakes, monkeypatch):
    _inventories(tmp_path, monkeypatch, build_s_a_tf_agent=_without(INVENTORY, "tool\tnvcc\t"))
    plan = _plan(tmp_path, [BASE, LAYER, CAP])
    with pytest.raises(StageFault, match="image s-a-tf-agent is refused: lacks the tools nvcc, which the brief says "
                                         "are installed") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    # The base and the layer, which the brief does not describe, were published before it.
    assert _published(plan) == sorted(_keys(BASE, LAYER).values()) and _staging_left(plan) == []


@pytest.mark.parametrize("record, named", [
    ("pip\ttriton\t", "lacks the Python packages triton"),
    ("apt\tpython3-dev\t", "lacks the Ubuntu packages python3-dev"),
], ids=["python", "ubuntu"])
def test_an_agent_cap_missing_a_promised_package_is_refused(tmp_path, fakes, monkeypatch, record, named):
    _inventories(tmp_path, monkeypatch, build_s_a_tf_agent=_without(INVENTORY, record))
    plan = _plan(tmp_path, [BASE, LAYER, CAP])
    with pytest.raises(StageFault, match=named) as caught:
        _run(plan, tmp_path)
    assert not caught.value.fault.retryable
    assert _published(plan) == sorted(_keys(BASE, LAYER).values())


def test_a_layer_is_not_held_to_the_briefs_promises(tmp_path, fakes, monkeypatch):
    bare = _without(_without(INVENTORY, "tool\tnvcc\t"), "pip\ttriton\t")
    _inventories(tmp_path, monkeypatch, build_s_a=bare, build_s_a_tf=bare)
    record, _, _ = _run(_plan(tmp_path, [BASE, LAYER]), tmp_path)
    assert set(record) == {"s-a", "s-a-tf"}


def test_a_failed_build_is_a_retryable_infra_fault_and_publishes_nothing(tmp_path, fakes, monkeypatch):
    monkeypatch.setenv("FAKE_BUILD_FAIL", "1")
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="build-s-a") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    # The build role's own fault, not the missing image file after it.
    assert caught.value.fault.evidence["roles"] == "build-s-a"
    assert _published(plan) == [] and _staging_left(plan) == []


def test_a_build_that_writes_no_image_fails_its_role_a_retryable_infra_fault(tmp_path, fakes):
    """An apptainer that succeeds without writing the image leaves the role's sha256sum nothing
    to read, so the role fails."""
    (tmp_path / "bin" / "apptainer").write_text('#!/bin/sh\n[ "$2" = --sandbox ] || exit 0\nexit 9\n')
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="build-s-a") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.retryable and caught.value.fault.evidence["roles"] == "build-s-a"
    assert _published(plan) == [] and _staging_left(plan) == []


def test_a_build_that_reports_no_digest_is_a_retryable_infra_fault(tmp_path, fakes):
    (tmp_path / "bin" / "sha256sum").write_text("#!/bin/sh\nexit 0\n")
    (tmp_path / "bin" / "sha256sum").chmod(0o755)
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="build-s-a reported no sha256") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.retryable and _published(plan) == [] and _staging_left(plan) == []


def test_a_corrupt_published_entry_is_an_infra_fault_that_is_not_retried(tmp_path, fakes):
    _run(_plan(tmp_path, [BASE]), tmp_path)
    key = _keys(BASE)["s-a"]
    plan = _plan(tmp_path, [BASE], instance="r2")
    PosixStore(plan.store_root).path(f"cache/images/{key}/{ENTRY_FILE}").write_text("not json")
    with pytest.raises(StageFault, match=f"cache entry images/{key}") as caught:
        _run(plan, tmp_path, job="101")
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert caught.value.fault.evidence["entry"] == f"images/{key}"


def test_a_published_entry_without_an_image_file_is_an_infra_fault_that_is_not_retried(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE])
    key = _keys(BASE)["s-a"]
    populate(ContentCache(PosixStore(plan.store_root)),
             "images", key, lambda d: (d / "meta.json").write_text("{}"), holder=Holder.here("test"))
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(StageFault, match=f"cache entry images/{key} holds no image.sif") as caught:
        _run(plan, tmp_path, launcher=launcher)
    assert not caught.value.fault.retryable and launcher.launched == []


def test_a_wait_for_another_filler_that_outlasts_the_deadline_is_retried(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE])
    key = _keys(BASE)["s-a"]
    other = Holder(host="other", pid=7, job_id="7", purpose="other filler")
    GenerationLock(PosixStore(plan.store_root), f"cache-locks/images/{key}", ttl_sec=30).try_acquire(other)
    ctx = _ctx(plan, LocalLauncher(gpus=0), tmp_path, deadline_sec=1)
    with pytest.raises(StageFault, match="still filled by other filler") as caught:
        ImagesStage().run(ctx)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert _published(plan) == []
    [wait] = [e for e in read_events(ctx.events.path) if e["kind"] == "cache_wait"]
    assert (wait["entry_kind"], wait["key"], wait["holder_purpose"], wait["holder_job_id"]) == (
        "images", key, "other filler", "7")


def test_a_wait_for_another_filler_ends_when_the_allocation_stops(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE])
    key = _keys(BASE)["s-a"]
    other = Holder(host="other", pid=7, job_id="7", purpose="other filler")
    GenerationLock(PosixStore(plan.store_root), f"cache-locks/images/{key}", ttl_sec=30).try_acquire(other)
    ctx = _ctx(plan, LocalLauncher(gpus=0), tmp_path, deadline_sec=60)
    ctx.stopping = True
    started = time.monotonic()
    with pytest.raises(StageFault, match="stopped while other filler") as caught:
        ImagesStage().run(ctx)
    assert caught.value.fault.retryable and time.monotonic() - started < 30 and ctx.launcher.launched == []


def test_a_failed_extraction_is_a_retryable_infra_fault_and_publishes_nothing(tmp_path, fakes, monkeypatch):
    monkeypatch.setenv("FAKE_EXTRACT_FAIL", "1")
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="extract-") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert _published(plan) == [] and _staging_left(plan) == []
    assert not list((tmp_path / "node" / "hls-100" / "sandboxes").iterdir())


def test_an_inventory_that_is_not_an_inventory_is_refused_and_not_retried(tmp_path, fakes):
    _, inventory = fakes
    inventory.write_text("not an inventory line\n")
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="inventory of s-a") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert "unknown record kind" in caught.value.fault.evidence["inventory"]
    assert _published(plan) == [] and _staging_left(plan) == []


@pytest.mark.parametrize("kind, retryable", [("transient", True), ("deterministic", False)])
def test_a_built_image_whose_inventory_reports_a_failed_probe_is_refused(tmp_path, fakes, kind, retryable):
    _, inventory = fakes
    inventory.write_text(INVENTORY + f"error\t{kind}\tdpkg-query exited 124\n")
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="inventory of s-a") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.retryable is retryable
    assert "dpkg-query exited 124" in caught.value.fault.evidence["inventory"]
    assert _published(plan) == [] and _staging_left(plan) == []


def test_an_inventory_that_cannot_be_read_is_a_retryable_fault(tmp_path, fakes, monkeypatch):
    real = images_stage._Images._wait

    def losing_the_output(self):
        real(self)
        for out in (self.ctx.attempt_dir / "logs").glob("inventory-*.out"):
            out.unlink()

    monkeypatch.setattr(images_stage._Images, "_wait", losing_the_output)
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="inventory of s-a") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.retryable and _published(plan) == []


def test_an_image_published_with_other_bytes_than_were_inventoried_is_refused(tmp_path, fakes, monkeypatch):
    """The entry's image file must be the file the inventory ran on: the publish role's digest of
    its copy is checked against it."""
    real = images_stage._Images._copy

    def tampering(self, name, source, dest):
        if name.startswith("publish-"):
            source.write_bytes(source.read_bytes() + b"x")
        return real(self, name, source, dest)

    monkeypatch.setattr(images_stage._Images, "_copy", tampering)
    plan = _plan(tmp_path, [BASE])
    with pytest.raises(StageFault, match="not the .* of the file that was inventoried") as caught:
        _run(plan, tmp_path)
    assert caught.value.fault.retryable and _published(plan) == [] and _staging_left(plan) == []


def test_a_source_build_without_its_backend_below_is_refused_before_anything_starts(tmp_path, fakes):
    tts = LayerImage(name="s-a-tts", parent="s-a", pip_lock="pip-sdist.txt",
                     source_builds={"sox==1.5.0": ["hatchling"]})
    plan = _plan(tmp_path, [BASE, tts])
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(StageFault, match="sox==1.5.0 of image s-a-tts needs hatchling") as caught:
        _run(plan, tmp_path, launcher=launcher)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert launcher.launched == [] and _published(plan) == []


def test_a_recipe_whose_lock_is_missing_is_refused_when_planned_and_before_anything_starts(tmp_path, fakes,
                                                                                            monkeypatch):
    """The plan keys every image to count the entries the stage fills, so a recipe that cannot be
    keyed refuses the plan; a lock gone after the plan was made refuses the stage."""
    layer = LayerImage(name="s-a-x", parent="s-a", pip_lock="absent.txt")
    with pytest.raises(PlanError, match="stage images: the recipe of image s-a-x cannot be keyed"):
        _plan(tmp_path, [BASE, layer])
    locks = tmp_path / "locks"
    shutil.copytree(FIXTURES, locks)
    shutil.copy(FIXTURES / "pip-layer.txt", locks / "absent.txt")
    monkeypatch.setattr(images_stage, "LOCKS_DIR", locks)
    plan = _plan(tmp_path, [BASE, layer])
    (locks / "absent.txt").unlink()
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(StageFault, match="recipe of image s-a-x") as caught:
        _run(plan, tmp_path, launcher=launcher)
    assert not caught.value.fault.retryable and launcher.launched == [] and _published(plan) == []


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_the_base_build_logs_in_with_the_plain_user_name_and_the_granted_token(tmp_path, fakes):
    log, _ = fakes
    plan = _plan(tmp_path, [BASE, LAYER], site=LOGIN_SITE)
    _, launcher, ctx = _run(plan, tmp_path, secrets=TOKEN)
    builds = fake_build.read_log(log)
    assert builds[0]["login"] == {"APPTAINER_DOCKER_USERNAME": _sha(REGISTRY_USER),
                                  "APPTAINER_DOCKER_PASSWORD": _sha(TOKEN["REGISTRY_PASSWORD"])}
    assert builds[0]["registry"] == [] and builds[1]["login"] == {} and builds[1]["registry"] == []
    base, layer = (s for s in launcher.launched if s.name.startswith("build-"))
    assert base.argv[:7] == ("/bin/sh", "-c", LOGIN_SHIM, "registry-login", "/bin/sh", "-c", BUILD_SHELL)
    assert layer.argv[:3] == ("/bin/sh", "-c", BUILD_SHELL)
    # The user name is the base build's declared variable and the token reaches it only through
    # its env file; the layer's build gets neither, and no other role the user name.
    assert base.env[REGISTRY_USER_VAR] == REGISTRY_USER and base.env_file is not None
    assert layer.env_file is None
    assert [s.name for s in launcher.launched if REGISTRY_USER_VAR in s.env] == ["build-s-a"]
    for spec in launcher.launched:
        assert TOKEN["REGISTRY_PASSWORD"] not in " ".join(spec.argv)
        assert TOKEN["REGISTRY_PASSWORD"] not in spec.env.values()
    assert not list((ctx.scratch / "roles").rglob("*.env"))
    # The base build logged privately; its scrubbed log still gives the build mode.
    assert _meta(plan, _keys(BASE)["s-a"])["build_mode"] == ROOT_MAPPED_MODE.model_dump()


@pytest.mark.parametrize("missing", [REGISTRY_USER_VAR, REGISTRY_TOKEN_GRANT])
def test_the_login_shim_stops_before_the_build_when_a_login_variable_is_missing_or_empty(tmp_path, missing):
    argv = build_argv(tmp_path / "recipe.def", tmp_path / "image.sif", digest=tmp_path / "d",
                      apptainer="false", login=True)
    for env in ({k: v for k, v in LOGINS.items() if k != missing}, {**LOGINS, missing: ""}):
        done = subprocess.run(argv, env={"PATH": "/usr/bin:/bin", **env}, capture_output=True, text=True,
                              timeout=30)
        assert done.returncode != 0 and missing in done.stderr
        assert not any(value and value in done.stderr for value in env.values())
    done = subprocess.run([*argv[:4], "sh", "-c", 'echo "$APPTAINER_DOCKER_USERNAME|$REGISTRY_PASSWORD"'],
                          env={"PATH": "/usr/bin:/bin", **LOGINS}, capture_output=True, text=True, timeout=30)
    assert done.stdout == f"{LOGINS['REGISTRY_USERNAME']}|\n"


def test_every_role_the_stage_starts_passes_the_slurm_launchers_checks(tmp_path, fakes):
    plan = _plan(tmp_path, [BASE, LAYER, CAP], site=LOGIN_SITE)
    _, launcher, _ = _run(plan, tmp_path, secrets=TOKEN)
    kinds = {s.name.split("-")[0] for s in launcher.launched}
    assert kinds == {"build", "extract", "inventory", "publish"}
    for spec in launcher.launched:
        srun_argv(spec, python=sys.executable, report=tmp_path / "r.json", scratch=tmp_path, shared=None,
                  allocation_gpus=0)
        env = step_env({"PATH": "/usr/bin:/bin"}, spec.env, tmpdir=tmp_path)
        assert TOKEN["REGISTRY_PASSWORD"] not in env.values()


def test_the_local_launcher_refuses_a_role_that_slurm_would_refuse(tmp_path):
    spec = RoleSpec(name="r", argv=("true",), shape=ResourceShape(cpus=1, memory_gib=1, time_sec=60),
                    owner="harness", log_dir=tmp_path / "logs", env={"APPTAINER_BIND": "/shared"})
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(ValueError, match="may not set .*APPTAINER_BIND"):
        launcher.launch(spec)
    assert launcher.launched == []


def _granted_plan(tmp_path: Path, grants: dict[str, str], *, site: dict[str, str] | None = None,
                  values: dict[str, str] | None = None) -> Plan:
    """A plan of the images stage under *site* settings, granted *grants* of a secrets file that
    holds the token and *values*."""
    secrets = SecretsFile(secrets={"registry_pass": TOKEN["REGISTRY_PASSWORD"], **(values or {})},
                          grants={"stage.images": grants})
    params = {"catalog_task": "sample_task", "images": [BASE.model_dump()]}
    return build_plan(decls=[StageDecl("images", "images", SHAPE, params)], task=NOOP_TASK, pipeline="noop2",
                      layout=InstanceLayout.rollout(NOOP_TASK, "g1"), settings=settings(tmp_path, site=site),
                      code=CODE,
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-images", known_stage_names=KNOWN_STAGE_NAMES,
                      known_tasks=(), now=1000.0, secrets=secrets)


def test_the_planner_grants_the_registry_token_under_its_own_name_never_apptainers(tmp_path, locks):
    grants = {REGISTRY_TOKEN_GRANT: "registry_pass"}
    plan = _granted_plan(tmp_path, grants, site=LOGIN_SITE)
    assert plan.stage("images").grants == grants and plan.settings.site.registry_username == REGISTRY_USER
    assert TOKEN["REGISTRY_PASSWORD"] not in plan.model_dump_json()
    for var in ("APPTAINER_DOCKER_PASSWORD", "APPTAINER_CACHEDIR"):
        with pytest.raises(PlanError, match=f"grants set .*{var}"):
            _granted_plan(tmp_path, {var: "registry_pass"})


@pytest.mark.parametrize("value", [REGISTRY_USER, "registry-user"])
def test_a_plan_whose_registry_user_name_holds_a_value_of_the_secrets_file_is_refused(tmp_path, locks, value):
    # Every role naming a path that holds the user name would be refused as holding a secret.
    with pytest.raises(PlanError, match="site.registry_username holds the value of secret old_user") as info:
        _granted_plan(tmp_path, {REGISTRY_TOKEN_GRANT: "registry_pass"}, site=LOGIN_SITE,
                      values={"old_user": value})
    assert value not in str(info.value)
    # A secret that holds the user name is no part of it.
    plan = _granted_plan(tmp_path, {REGISTRY_TOKEN_GRANT: "registry_pass"}, site=LOGIN_SITE,
                         values={"other": f"x{REGISTRY_USER}x"})
    assert plan.settings.site.registry_username == REGISTRY_USER


def test_a_cold_chain_builds_on_a_node_with_one_builds_room_and_one_tree(tmp_path, fakes, monkeypatch):
    # Each tree is KEPT_DISK_GIB by its apparent size, and the node has exactly the disk the
    # stage declares: one build and one tree for three images, each tree released once its image
    # is inventoried. Every role's grant is checked as the step wrapper checks it.
    monkeypatch.setenv("FAKE_TREE_GIB", str(images_stage.KEPT_DISK_GIB))
    disk = images_stage.BUILD_DISK_GIB + images_stage.KEPT_DISK_GIB
    plan = _plan(tmp_path, [BASE, LAYER, CAP], shape=_disk(disk))
    launcher = LocalLauncher(gpus=0, node_disk_gib=disk, scratch=tmp_path / "node" / "hls-100")
    record, launcher, ctx = _run(plan, tmp_path, launcher=launcher)
    assert sorted(record) == ["s-a", "s-a-tf", "s-a-tf-agent"]
    assert {s.shape.disk_gib for s in launcher.launched} == {0}
    free = [(h.spec.name.split("-")[0], h.report(1.0).granted.disk_free_gib) for h in launcher.handles]
    builds, extracts = ([f for kind, f in free if kind == k] for k in ("build", "extract"))
    assert len(builds) == len(extracts) == 3
    assert min(builds) >= images_stage.BUILD_DISK_GIB and min(extracts) >= images_stage.KEPT_DISK_GIB
    kinds = [e["kind"] for e in read_events(ctx.events.path) if e["kind"].startswith("image_")]
    assert kinds == ["image_extracted", "image_removed"] * 3 and _left_on_node(ctx) == []


@pytest.mark.parametrize("site, grants, says", [
    (LOGIN_SITE, {}, "site.registry_username is set but stage images is not granted REGISTRY_PASSWORD"),
    (None, {REGISTRY_TOKEN_GRANT: "registry_pass"},
     "stage images is granted REGISTRY_PASSWORD but site.registry_username is not set"),
])
def test_a_registry_login_without_its_user_name_or_its_token_is_refused_when_the_plan_is_made(tmp_path, locks,
                                                                                              site, grants, says):
    """A site that sets only one of the two would queue a job whose images stage cannot log in,
    even when every image is a cache hit."""
    with pytest.raises(PlanError, match=says):
        _granted_plan(tmp_path, grants, site=site)


def test_a_site_without_a_registry_login_plans_an_images_stage_granted_no_token(tmp_path, locks):
    plan = _granted_plan(tmp_path, {})
    assert plan.stage("images").grants == {} and plan.settings.site.registry_username is None


def test_the_login_is_checked_for_each_images_stage_and_no_other(tmp_path, locks):
    """Only an images stage logs in to a registry: each must have both halves, while a site's
    user name beside a plan with no images stage, or another stage granted the token, is no
    half of a login."""
    token = {REGISTRY_TOKEN_GRANT: "registry_pass"}
    params = {"catalog_task": "sample_task", "images": [BASE.model_dump()]}
    decls = [StageDecl("images", "images", SHAPE, params), StageDecl("again", "images", SHAPE, params)]
    granted_first = SecretsFile(secrets={"registry_pass": TOKEN["REGISTRY_PASSWORD"]}, grants={"stage.images": token})
    with pytest.raises(PlanError, match="site.registry_username is set but stage again is not granted"):
        noop_plan(tmp_path, decls=decls, site=LOGIN_SITE, secrets=granted_first)
    noop_granted = SecretsFile(secrets={"registry_pass": TOKEN["REGISTRY_PASSWORD"]}, grants={"stage.noop_a": token})
    assert noop_plan(tmp_path, site=LOGIN_SITE).settings.site.registry_username == REGISTRY_USER
    assert noop_plan(tmp_path, secrets=noop_granted).stage("noop_a").grants == token


def test_roles_naming_paths_under_the_user_name_start_and_one_holding_the_token_is_refused(tmp_path, fakes):
    # The registry user name is the cluster login name, so the store and the node's scratch lie
    # under a directory of that name, as every path under the user's directories does.
    home = tmp_path / REGISTRY_USER
    home.mkdir()
    plan = _plan(home, [BASE, LAYER], site=LOGIN_SITE)
    record, launcher, ctx = _run(plan, home, secrets=TOKEN)
    assert sorted(record) == ["s-a", "s-a-tf"]
    named = {s.name.split("-")[0] for s in launcher.launched if any(REGISTRY_USER in arg for arg in s.argv)}
    assert named == {"build", "extract", "inventory", "publish"}
    shape = ResourceShape(cpus=1, memory_gib=1, time_sec=60)
    for argv, env, where in ((["/bin/echo", TOKEN["REGISTRY_PASSWORD"]], {}, r"argv\[1\]"),
                             (["/bin/true"], {"X": f"a{TOKEN['REGISTRY_PASSWORD']}"}, "env X")):
        with pytest.raises(ValueError, match=f"{where} holds a secret"):
            ctx.start_role("leak", argv, shape, env=env, owner="harness")
    assert "leak" not in [s.name for s in launcher.launched]


def test_the_stage_takes_no_gpu_and_leaves_the_runner_its_memory(locks):
    params = ImagesParams(catalog_task="sample_task", images=[BASE])
    with pytest.raises(ValueError, match="no GPU"):
        ImagesStage.check(ResourceShape(gpus=1, gpu_kind="H200", cpus=4, memory_gib=8, disk_gib=80,
                                        time_sec=60), params, runner_memory_gib=RUNNER_GIB)
    with pytest.raises(ValueError, match="leaves its roles none"):
        ImagesStage.check(ResourceShape(cpus=4, memory_gib=4, disk_gib=160, time_sec=60), params,
                          runner_memory_gib=RUNNER_GIB)


@pytest.mark.parametrize("found, fills, disk", [
    ((), 3, 80), (("s-a",), 2, 80), (("s-a", "s-a-tf"), 1, 80), (("s-a", "s-a-tf", "s-a-tf-agent"), 0, 0),
    # The cache can hold a child without its parent: the parent alone is filled.
    (("s-a-tf", "s-a-tf-agent"), 1, 80),
])
def test_the_disk_is_one_build_and_one_tree_whatever_the_stage_fills_and_none_when_it_finds_all(tmp_path, locks,
                                                                                                found, fills, disk):
    """One fill at a time, each tree released once its image is inventoried: a build's room and a
    tree however many entries the stage fills. An entry found holds nothing on the node, and with
    nothing to fill the stage holds no disk."""
    keys = _keys(BASE, LAYER, CAP)
    params = ImagesParams(catalog_task="sample_task", images=[BASE, LAYER, CAP], found=sorted(keys[n] for n in found))
    entries = "entry" if fills == 1 else "entries"
    if disk:
        with pytest.raises(ValueError, match=f"at least {disk} GiB of node-local disk for the {fills} {entries} it "
                                             f"fills, not {disk - 1}"):
            ImagesStage.check_site(_disk(disk - 1), params, site=settings(tmp_path).site)
    ImagesStage.check_site(_disk(disk), params, site=settings(tmp_path).site)


def test_a_plan_that_found_an_entry_no_image_has_or_one_twice_is_refused(tmp_path, locks):
    key = _keys(BASE)["s-a"]
    stray = ImagesParams(catalog_task="sample_task", images=[BASE], found=["s-a-0000000000000000"])
    with pytest.raises(ValueError, match="found s-a-0000000000000000, which is no image's key"):
        ImagesStage.check_site(_disk(80), stray, site=settings(tmp_path).site)
    with pytest.raises(ValueError, match=f"found lists {key} twice"):
        ImagesParams(catalog_task="sample_task", images=[BASE], found=[key, key])


def test_a_found_parent_puts_no_tree_on_the_node_and_its_childs_build_holds_its_file(tmp_path, fakes, monkeypatch):
    """A child built on a parent the cache holds copies the parent's file into its build
    directory and builds on it there, in the room of one build; the build removes the copy when it
    ends, and nothing extracts the parent. So a found base and a built layer need one build and
    one tree, the layer's, on a node with no more disk than that."""
    first, _, _ = _run(_plan(tmp_path, [BASE]), tmp_path)
    monkeypatch.setenv("FAKE_TREE_GIB", str(images_stage.KEPT_DISK_GIB))
    keys = _keys(BASE, LAYER)
    disk = images_stage.BUILD_DISK_GIB + images_stage.KEPT_DISK_GIB
    plan = _plan(tmp_path, [BASE, LAYER], instance="r2", found=[keys["s-a"]], shape=_disk(disk))
    launcher = LocalLauncher(gpus=0, node_disk_gib=disk, scratch=tmp_path / "node" / "hls-101")
    record, launcher, ctx = _run(plan, tmp_path, launcher=launcher, job="101")
    assert record["s-a"] == first["s-a"] and _roles(launcher) == ["build-s-a-tf", "extract", "inventory-s-a-tf",
                                                                  "publish-s-a-tf"]
    extracted = [e["sif_sha256"] for e in read_events(ctx.events.path) if e["kind"] == "image_extracted"]
    assert extracted == [record["s-a-tf"]["sif_sha256"]]
    assert not list(ctx.scratch.rglob("parent.sif"))
    granted = {h.spec.name.split("-")[0]: h.report(1.0).granted.disk_free_gib for h in launcher.handles}
    assert granted["build"] >= images_stage.BUILD_DISK_GIB and granted["extract"] >= images_stage.KEPT_DISK_GIB


def test_a_build_leaves_only_its_image_file_once_it_ends(tmp_path, fakes):
    """A base's build pulls its layers into apptainer's cache in the build's directory; the cache
    goes when the build ends, so what the build leaves is its image file, which goes once it is
    published."""
    _, _, ctx = _run(_plan(tmp_path, [BASE]), tmp_path)
    own = ctx.scratch / "roles" / "build-s-a"
    assert own.is_dir() and not (own / "cache").exists() and not (own / IMAGE_FILE).exists()


def test_an_entry_the_plan_found_and_the_cache_lost_ends_the_stage_at_its_turn_before_any_fill(tmp_path, fakes):
    """The plan declared no disk, as the cache held both images; filling the base, which it must at
    its turn, would need a build's room and a tree, so the stage ends there, naming the entry, before
    anything starts. A retry would run the same plan, so it is not retried: a new plan counts the
    fills."""
    keys = _keys(BASE, LAYER)
    plan = _plan(tmp_path, [BASE, LAYER], found=sorted(keys.values()), shape=_disk(0))
    launcher = LocalLauncher(gpus=0)
    refusal = "the cache no longer holds s-a, which it held when the plan was made"
    with pytest.raises(StageFault, match=refusal) as caught:
        _run(plan, tmp_path, launcher=launcher)
    fault = caught.value.fault
    assert fault.fault_class == "infra" and not fault.retryable
    assert "1 entry, which need 80 GiB of node-local disk, more than the 0 GiB it declares" in fault.reason
    assert fault.evidence == {"entries": f"images/{keys['s-a']}", "declared_disk_gib": "0", "needed_disk_gib": "80"}
    assert launcher.launched == [] and _published(plan) == [] and _staging_left(plan) == []


def test_an_entry_missing_when_the_stage_starts_and_published_by_its_turn_is_found(tmp_path, fakes, monkeypatch):
    """Another stage publishes the base after this one started and before its turn: the stage finds
    it there and fills nothing, which its declared disk, none, holds."""
    keys = _keys(BASE, LAYER)
    plan = _plan(tmp_path, [BASE, LAYER], found=sorted(keys.values()), shape=_disk(0))
    cache = ContentCache(PosixStore(plan.store_root))
    populate(cache, "images", keys["s-a-tf"], lambda d: (d / IMAGE_FILE).write_bytes(b"layer"),
             holder=Holder.here("test"))
    real, pending = ContentCache.populate, [keys["s-a"]]

    def published_by_its_turn(self, kind, key, fill, **kw):
        if key in pending:
            pending.remove(key)
            populate(self, kind, key, lambda d: (d / IMAGE_FILE).write_bytes(b"base"), holder=Holder.here("other"))
        return real(self, kind, key, fill, **kw)

    monkeypatch.setattr(ContentCache, "populate", published_by_its_turn)
    launcher = LocalLauncher(gpus=0)
    record, _, _ = _run(plan, tmp_path, launcher=launcher)
    assert {name: r["key"] for name, r in record.items()} == keys and launcher.launched == []


def test_of_two_entries_the_plan_found_only_the_lost_one_is_named(tmp_path, fakes):
    keys = _keys(BASE, LAYER)
    plan = _plan(tmp_path, [BASE, LAYER], found=sorted(keys.values()), shape=_disk(0))
    populate(ContentCache(PosixStore(plan.store_root)), "images", keys["s-a-tf"],
             lambda d: (d / IMAGE_FILE).write_bytes(b"layer"), holder=Holder.here("test"))
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(StageFault, match="the cache no longer holds s-a, which it held when the plan was made: the "
                                         "stage would fill 1 entry, which need 80 GiB") as caught:
        _run(plan, tmp_path, launcher=launcher)
    assert caught.value.fault.evidence["entries"] == f"images/{keys['s-a']}" and launcher.launched == []


def test_an_entry_the_plan_found_and_that_is_corrupt_now_is_refused_as_corrupt(tmp_path, fakes):
    """A corrupt entry is not an absent one: the stage refuses it, not retried, rather than counting
    it as a fill its disk cannot hold."""
    _run(_plan(tmp_path, [BASE]), tmp_path)
    key = _keys(BASE)["s-a"]
    plan = _plan(tmp_path, [BASE], instance="r2", found=[key], shape=_disk(0))
    PosixStore(plan.store_root).path(f"cache/images/{key}/{ENTRY_FILE}").write_text("not json")
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(StageFault, match=f"cache entry images/{key}: .* not a valid entry manifest") as caught:
        _run(plan, tmp_path, launcher=launcher, job="101")
    assert not caught.value.fault.retryable and caught.value.fault.evidence == {"entry": f"images/{key}"}
    assert launcher.launched == []


def _evicting(monkeypatch, gone: str) -> None:
    """Make the cache lose entry *gone* at its image's turn, after the stage started."""
    real = ContentCache.populate

    def populate_(self, kind, key, fill, **kw):
        if key == gone:
            shutil.rmtree(self.store.path(f"cache/images/{key}"))
        return real(self, kind, key, fill, **kw)

    monkeypatch.setattr(ContentCache, "populate", populate_)


def test_a_plan_whose_found_key_is_no_images_key_any_more_is_refused_before_anything_starts(tmp_path, fakes,
                                                                                            monkeypatch):
    locks = tmp_path / "locks"
    shutil.copytree(FIXTURES, locks)
    monkeypatch.setattr(images_stage, "LOCKS_DIR", locks)
    key = images_stage.image_keys([BASE], ApptainerImages.recipe_key)["s-a"]
    plan = _plan(tmp_path, [BASE], found=[key], shape=_disk(0))
    with (locks / BASE.pip_lock).open("a") as f:
        f.write("# a lock edited after the plan was made\n")
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(StageFault, match=f"the plan found {key}, which is no image's key now") as caught:
        _run(plan, tmp_path, launcher=launcher)
    assert not caught.value.fault.retryable and launcher.launched == []


def test_a_lost_entry_is_filled_when_the_stages_declared_disk_holds_the_fill(tmp_path, fakes):
    keys = _keys(BASE, LAYER)
    disk = images_stage.BUILD_DISK_GIB + images_stage.KEPT_DISK_GIB
    plan = _plan(tmp_path, [BASE, LAYER], found=sorted(keys.values()), shape=_disk(disk))
    record, _, _ = _run(plan, tmp_path)
    assert {name: r["key"] for name, r in record.items()} == keys and _published(plan) == sorted(keys.values())


def test_an_entry_lost_after_the_stage_started_ends_it_at_that_images_turn(tmp_path, fakes, monkeypatch):
    """Both entries are there when the stage starts, the base's is found at its turn, and the
    layer's is gone by its own: the plan declared no disk, so the stage fills nothing."""
    keys = _keys(BASE, LAYER)
    plan = _plan(tmp_path, [BASE, LAYER], found=sorted(keys.values()), shape=_disk(0))
    cache = ContentCache(PosixStore(plan.store_root))
    for name in ("s-a", "s-a-tf"):
        populate(cache, "images", keys[name], lambda d: (d / IMAGE_FILE).write_bytes(b"image"),
                 holder=Holder.here("test"))
    _evicting(monkeypatch, keys["s-a-tf"])
    launcher = LocalLauncher(gpus=0)
    with pytest.raises(StageFault, match="the cache no longer holds s-a-tf, which it held when the plan was made: "
                                         "the stage would fill 1 entry, which need 80 GiB") as caught:
        _run(plan, tmp_path, launcher=launcher)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _published(plan) == [keys["s-a"]] and launcher.launched == [] and _staging_left(plan) == []


def test_a_contestant_role_never_runs_on_the_host_in_the_images_stage(tmp_path, fakes):
    ctx = _ctx(_plan(tmp_path, [BASE]), LocalLauncher(gpus=0), tmp_path)
    with pytest.raises(ValueError, match="runs only in a container"):
        ctx.start_role("x", ["true"], ResourceShape(cpus=1, memory_gib=1, time_sec=60), owner="contestant")
    assert ctx.launcher.launched == []


@pytest.mark.parametrize("params, match", [
    ({"images": []}, "at least one image"),
    ({"images": [LAYER.model_dump()]}, "unknown parent"),
    ({"images": [BASE.model_dump(), BASE.model_dump()]}, "declared twice"),
    ({"images": [BASE.model_dump()], "sources": {"s-a": "/x/a.sif"}}, "Extra inputs are not permitted"),
    ({"images": [{"kind": "imported", "name": "b"}]}, "imported"),
])
def test_bad_image_lists_are_refused_when_the_plan_is_made(tmp_path, params, match):
    with pytest.raises(PlanError, match=match):
        noop_plan(tmp_path, decls=[StageDecl("images", "images", SHAPE, {"catalog_task": "multitenant_slo",
                                                                         **params})])


def test_a_docker_sites_images_stage_finds_its_entries_by_dockerfile_key_beside_an_apptainer_entry(tmp_path, locks):
    """A plan for a Docker site keys its images as the Docker image runtime does, by the
    Dockerfile, and sizes its images stage by the image file that runtime keeps: the stage finds
    every entry a Docker images stage published, and an Apptainer entry of the same image in the
    same cache, under the ``.def``'s key, is neither found nor refused."""
    slurm = settings(tmp_path)
    site = SiteSettings.model_validate(
        {**{k: v for k, v in slurm.site.model_dump(mode="json").items() if k not in SLURM_SITE_FIELDS},
         "substrate": "docker", "docker": {"buildkit_image": BUILDKIT},
         "local": {"state_dir": str(tmp_path / "state"), "kept_memory_gib": 8}})
    on_docker = slurm.model_copy(update={"site": site})
    decls = [BASE, LAYER, CAP]
    docker_keys = images_stage.image_keys(decls, DockerImages.recipe_key)
    def_keys = images_stage.image_keys(decls, ApptainerImages.recipe_key)
    assert not set(docker_keys.values()) & set(def_keys.values())
    cache = ContentCache(PosixStore(site.store_root))
    for name, key in docker_keys.items():
        populate(cache, "images", key, lambda d, n=name: (d / DockerImages.image_file).write_bytes(n.encode()),
                 holder=Holder.here("test"))
    populate(cache, "images", def_keys["s-a"], lambda d: (d / IMAGE_FILE).write_bytes(b"sif"), holder=Holder.here("test"))
    on = substrate(site)
    assert (on.recipe_key, on.image_file) == (DockerImages.recipe_key, DockerImages.image_file)

    stage = plan_images_stage(catalog_task="multitenant_slo", decls=decls, held=commands.cache_holds(on_docker),
                              recipe_key=on.recipe_key)
    assert stage.params["found"] == sorted(docker_keys.values()) and stage.shape.disk_gib == 0
    params = ImagesParams.model_validate(stage.params)
    ImagesStage.check_site(stage.shape, params, site=site)
    # The .def's key is no key of a Docker site's images.
    with pytest.raises(ValueError, match=f"found {def_keys['s-a']}, which is no image's key"):
        ImagesStage.check_site(stage.shape, params.model_copy(update={"found": [def_keys["s-a"]]}), site=site)

    # The stage, on the Docker image runtime, finds every entry and starts no role.
    planned = _plan(tmp_path, decls)
    images = planned.stage("images").model_copy(update={"params": stage.params, "shape": stage.shape})
    plan = planned.model_copy(update={"settings": on_docker, "stages": [images]})
    store, launcher = PosixStore(plan.store_root), LocalLauncher(gpus=0)
    scratch = tmp_path / "node" / "hls-100"
    ctx = StageContext(store=store, plan=plan, stage=images, attempt=1, launcher=launcher.use_plan(plan), watch=None,
                       events=EventLog(store.path(plan.layout.attempt_file("images", 1, "events.jsonl"))),
                       inputs={}, deadline=time.time() + 120, secrets={}, cache=ContentCache(store, lock_ttl_sec=30),
                       scratch=scratch, job_id="100", runtime=ApptainerRuntime(),
                       images=DockerImages(scratch, job_id="100", python=sys.executable,
                                           buildkit_image=BUILDKIT))
    record = json.loads(store.read_bytes(ImagesStage().run(ctx)["images"]))
    assert {name: r["key"] for name, r in record.items()} == docker_keys and launcher.launched == []

