"""The stage context's pinned images: extracted on the node for the stage that pins them, from
a copy in the extraction role's own directory whose digest the role checked, reused within the
stage through its own memo, removed when its context closes, a tree an earlier stage left reused
only after a check against the manifest written at extraction, and the only images a container
runs from. And the refusals the context reads from its roles."""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import stat
import sys
import threading
import time
from pathlib import Path

import pytest

from hls.substrate.apptainer import images as images_module
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.plan import Plan
from hls.pipeline.stage import Refusal, StageContext, StageFault, run_stage, stage_fault
from hls.spec.resources import ResourceShape
from hls.store.cache import ContentCache, copies_root, entry_sha256
from hls.store.digest import DIGEST_MISMATCH_EXIT
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.base import RoleReport
from hls.substrate.container import ContainerSpec, PinnedImage
from hls.substrate.faults import infra_fault
from hls.substrate.sandbox import IMAGE_FILE, SANDBOXES, TreeManifest, sandbox_paths, tree_problem
from hls.util import remove_tree
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan
from tests.store.entries import populate
from tests.substrate import fake_apptainer
from tests.substrate.sandboxes import install_read_log, stand_in

SHAPE = ResourceShape(cpus=2, memory_gib=2, disk_gib=1, time_sec=60)
IMAGE = b"an image file\n"


@pytest.fixture
def apptainer(tmp_path, monkeypatch):
    """The fake apptainer on PATH, logging each extraction."""
    bin_dir = tmp_path / "bin"
    fake_apptainer.install(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    monkeypatch.setenv("FAKE_EXTRACT_LOG", str(tmp_path / "extracts.txt"))
    for knob in ("FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL", "FAKE_TREE_GIB", "FAKE_TREE_LOCKED", "FAKE_TREE_DATED"):
        monkeypatch.delenv(knob, raising=False)
    return bin_dir


@pytest.fixture
def reads(tmp_path, apptainer, monkeypatch):
    """The log of every file a role's ``cp`` or ``sha256sum`` reads (``install_read_log``)."""
    log = tmp_path / "reads.txt"
    install_read_log(tmp_path / "read-bin", log)
    monkeypatch.setenv("PATH", f"{tmp_path / 'read-bin'}:{os.environ['PATH']}")
    return log


def _ctx(tmp_path: Path, *, attempt: int = 1, job: str = "100", plan: Plan | None = None,
         launcher: LocalLauncher | None = None, images: ApptainerImages | None = None,
         disk_gib: int | None = None) -> StageContext:
    plan = plan or noop_plan(tmp_path)
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("noop_a", attempt, "events.jsonl")))
    stage = plan.stage("noop_a")
    if disk_gib is not None:
        stage = stage.model_copy(update={"shape": stage.shape.model_copy(update={"disk_gib": disk_gib})})
    return StageContext(store=store, plan=plan, stage=stage, attempt=attempt,
                        launcher=(launcher or LocalLauncher(gpus=2)).use_plan(plan), watch=None, events=events,
                        inputs={},
                        deadline=time.time() + 60, cache=ContentCache(store),
                        scratch=tmp_path / "node" / f"hls-{job}", job_id=job,
                        runtime=ApptainerRuntime(),
                        images=images if images is not None else ApptainerImages(tmp_path / "node" / f"hls-{job}"))


def _image(tmp_path: Path, data: bytes = IMAGE) -> tuple[Path, str]:
    path = tmp_path / "node" / "files" / hashlib.sha256(data).hexdigest()[:8] / "image.sif"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    """The digest an image file's producing role would report for it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _image_pin(ctx: StageContext, tmp_path: Path, data: bytes = IMAGE) -> PinnedImage:
    """A new image file holding *data*, pinned in *ctx* by the digest its producer reported."""
    path, digest = _image(tmp_path, data)
    return ctx.pin_file(path, sha256=digest)


def _kinds(ctx: StageContext) -> list[str]:
    return [e["kind"] for e in read_events(ctx.events.path) if e["kind"].startswith("image_")]


def _extracts(tmp_path: Path) -> list[str]:
    """The image file each extraction handed apptainer, in order."""
    log = tmp_path / "extracts.txt"
    return log.read_text().splitlines() if log.exists() else []


def _copy_of(ctx: StageContext, digest: str) -> Path:
    """Where the extraction role of the image whose file has *digest* copies the file."""
    assert ctx.scratch is not None
    return ctx.scratch / "roles" / f"extract-{digest[:12]}" / IMAGE_FILE


def _sandboxes(ctx: StageContext) -> list[str]:
    """What the job's sandboxes directory holds: trees, manifests and staging directories."""
    assert ctx.scratch is not None
    root = ctx.scratch / SANDBOXES
    return sorted(p.name for p in root.iterdir()) if root.exists() else []


def _published(ctx: StageContext, key: str) -> str:
    """Publish an images entry *key* holding ``IMAGE`` and return its ENTRY.json digest."""
    assert ctx.cache is not None
    return entry_sha256(populate(ctx.cache, "images", key, lambda d: (d / IMAGE_FILE).write_bytes(IMAGE),
                                 holder=Holder.here("test")).entry)


def test_an_image_file_is_extracted_once_by_a_harness_role_into_the_jobs_scratch(tmp_path, apptainer):
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    pinned = ctx.pin_file(path, sha256=_sha(path))
    root, manifest = sandbox_paths(ctx.scratch, digest)
    assert (Path(pinned.ref), pinned.file_sha256) == (root, digest)
    assert pinned.check_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    copy = _copy_of(ctx, digest)
    assert (root / ".fake-image").read_bytes() == IMAGE and _extracts(tmp_path) == [str(copy)]
    [role] = ctx.launcher.launched
    assert role.name == f"extract-{digest[:12]}" and role.owner == "harness"
    # The role copies the file into its own directory, checks the copy against the reported
    # digest, extracts it and removes it.
    assert role.argv[:2] == ("/bin/sh", "-c")
    assert role.argv[3:] == ("hls-extract", str(path), digest, "apptainer", role.argv[7], str(copy))
    assert not copy.exists()
    # The stage's CPUs and its memory less the runner's 4 GiB; never its GPU, and no disk.
    assert (role.shape.gpus, role.shape.cpus, role.shape.memory_gib, role.shape.disk_gib) == (0, 4, 2, 0)
    assert set(role.env) == {"APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"}
    # Apptainer's cache and its temporary files are the role's, and go with its directory.
    assert {Path(v).parent for v in role.env.values()} == {ctx.role_dir(role.name)}
    # Only the stage reaches its sandboxes.
    assert stat.S_IMODE((ctx.scratch / SANDBOXES).stat().st_mode) == 0o700
    [event] = [e for e in read_events(ctx.events.path) if e["kind"] == "image_extracted"]
    assert (event["sif_sha256"], event["manifest_sha256"]) == (digest, pinned.check_sha256)
    assert event["sec"] >= 0
    assert _kinds(ctx) == ["image_extracted"]
    assert _sandboxes(ctx) == [digest, f"{digest}.json"], "the staging directory is left"


def test_an_extraction_is_made_removable_before_its_manifest_so_the_jobs_cleanup_removes_it(tmp_path,
                                                                                         apptainer):
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    pinned = ctx.pin_file(path, sha256=_sha(path))
    locked = Path(pinned.ref) / "usr" / "share" / "ro"
    assert stat.S_IMODE(locked.stat().st_mode) == 0o755
    manifest = TreeManifest.model_validate_json(sandbox_paths(ctx.scratch, digest)[1].read_bytes())
    assert next(e.mode for e in manifest.entries if e.path == "usr/share/ro") == locked.stat().st_mode
    # A tree its stage could not remove is left for the end of the job, when the runner removes
    # the job's scratch.
    remove_tree(ctx.scratch, ignore_errors=True)
    assert not ctx.scratch.exists()


def test_an_image_with_directories_its_owner_cannot_enter_is_served_with_fresh_directory_times(
        tmp_path, apptainer, monkeypatch):
    """A contestant's image may hold directories its owner can neither list nor enter (modes 000
    and 300), and every tree carries the times its image was built with, which a node's periodic
    ``systemd-tmpfiles --clean`` reads as age: it removes an empty directory under /tmp whose
    access and modification times are both older than its age. The runner gives every directory
    the time of the extraction once the tree is made removable, so the image is pinned and runs;
    its files keep their times."""
    built = time.time() - 400 * 86400
    monkeypatch.setenv("FAKE_TREE_LOCKED", "1")
    monkeypatch.setenv("FAKE_TREE_DATED", str(built))
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    started = time.time() - 1
    pinned = ctx.pin_file(path, sha256=digest)
    dirs = [Path(pinned.ref), *(p for p in Path(pinned.ref).rglob("*") if p.is_dir() and not p.is_symlink())]
    assert {"opt/sealed", "opt/blind"} <= {p.relative_to(Path(pinned.ref)).as_posix() for p in dirs}
    stale = [p for p in dirs if p.stat().st_mtime < started or p.stat().st_atime < started]
    assert stale == []
    assert (Path(pinned.ref) / "opt" / "sealed" / "f").stat().st_mtime == pytest.approx(built)
    # The manifest was written after, so the tree still matches it.
    assert tree_problem(Path(pinned.ref), sandbox_paths(ctx.scratch, digest)[1], digest)[0] is None
    spec = ContainerSpec(image=pinned, command=("true",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="harness")
    assert stage_fault(ctx.wait_roles()) is None


def test_a_repeat_pin_within_a_stage_reuses_its_tree_through_the_stages_memo(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    first = _image_pin(ctx, tmp_path)
    assert _image_pin(ctx, tmp_path) == first
    expect = _published(ctx, "k1")
    # The entry's copy holds the same bytes, so the stage's tree of that digest serves it too.
    assert ctx.node_image("k1", expect) == first and ctx.node_image("k1", expect) == first
    assert _kinds(ctx) == ["image_extracted"] and len(ctx.launcher.launched) == 1
    assert len(_extracts(tmp_path)) == 1


def test_a_stage_removes_its_trees_when_it_closes_and_a_later_stage_extracts_again(tmp_path, apptainer):
    path, digest = _image(tmp_path)
    first = _ctx(tmp_path)
    pinned = first.pin_file(path, sha256=_sha(path))
    (Path(pinned.ref) / "usr" / "share" / "ro").chmod(0o555)
    first.close()
    root, manifest = sandbox_paths(first.scratch, digest)
    assert not os.path.lexists(root) and not os.path.lexists(manifest) and _sandboxes(first) == []
    assert _kinds(first) == ["image_extracted", "image_removed"]
    [removed] = [e for e in read_events(first.events.path) if e["kind"] == "image_removed"]
    assert removed["sif_sha256"] == digest
    later = _ctx(tmp_path, attempt=2)
    again = later.pin_file(path, sha256=_sha(path))
    assert Path(again.ref) == Path(pinned.ref) and _kinds(later) == ["image_extracted"]
    assert _extracts(tmp_path) == [str(_copy_of(later, digest))] * 2 and len(later.launcher.launched) == 1
    later.close()
    first.close()
    assert _kinds(first).count("image_removed") == 1 and _sandboxes(later) == []


def test_a_stage_releases_a_tree_it_is_done_with_and_a_later_pin_extracts_it_again(tmp_path, apptainer):
    """A released tree and its manifest go at once, and so does the stage's memo of it: pinning
    the same bytes again, by file or by entry, extracts them anew, and closing removes only that."""
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    pinned = ctx.pin_file(path, sha256=_sha(path))
    ctx.release(pinned)
    root, manifest = sandbox_paths(ctx.scratch, digest)
    assert not os.path.lexists(root) and not os.path.lexists(manifest) and _sandboxes(ctx) == []
    assert _kinds(ctx) == ["image_extracted", "image_removed"]
    expect = _published(ctx, "k1")
    entry = ctx.node_image("k1", expect)
    assert Path(entry.ref) == root and root.is_dir() and len(_extracts(tmp_path)) == 2
    ctx.release(entry)
    again = ctx.node_image("k1", expect)
    assert Path(again.ref) == root and root.is_dir() and len(_extracts(tmp_path)) == 3
    ctx.close()
    assert _kinds(ctx) == ["image_extracted", "image_removed"] * 3 and _sandboxes(ctx) == []


@pytest.mark.parametrize("command", ["import time; time.sleep(30)", "import sys; sys.exit(3)"],
                         ids=["running", "ended, not waited for"])
def test_a_tree_is_not_released_while_a_role_the_stage_has_not_waited_for_may_run_from_it(tmp_path, apptainer,
                                                                                         command):
    path, _ = _image(tmp_path)
    ctx = _ctx(tmp_path)
    pinned = ctx.pin_file(path, sha256=_sha(path))
    handle = ctx.start_role("busy", [sys.executable, "-c", command], SHAPE, owner="harness")
    if "exit" in command:
        handle.wait(30)
    with pytest.raises(RuntimeError, match=r"roles \['busy'\] have not been waited for"):
        ctx.release(pinned)
    assert Path(pinned.ref).is_dir() and "image_removed" not in _kinds(ctx)
    ctx.close()
    assert not Path(pinned.ref).exists()


def test_only_a_tree_the_context_holds_pinned_is_released(tmp_path, apptainer):
    path, _ = _image(tmp_path)
    ctx = _ctx(tmp_path)
    pinned = ctx.pin_file(path, sha256=_sha(path))
    ctx.release(pinned)
    with pytest.raises(ValueError, match=f"image {pinned.file_sha256} is not pinned in this stage"):
        ctx.release(pinned)
    other = _ctx(tmp_path, attempt=2).pin_file(path, sha256=_sha(path))
    with pytest.raises(ValueError, match="is not pinned in this stage"):
        ctx.release(other)
    assert Path(other.ref).is_dir()


def test_a_tree_that_cannot_be_released_stays_pinned_and_its_close_removes_it(tmp_path, apptainer, monkeypatch):
    """The stage's disk would not hold the tree it kept, so the release is the stage's fault, and
    retried; the tree stays pinned until the context closes."""
    path, _ = _image(tmp_path)
    ctx = _ctx(tmp_path)
    pinned = ctx.pin_file(path, sha256=_sha(path))
    removal = images_module.remove_extracted

    def stuck(root: Path) -> None:
        raise PermissionError(f"cannot remove {root}")

    monkeypatch.setattr(images_module, "remove_extracted", stuck)
    with pytest.raises(StageFault, match=f"the tree of image {pinned.file_sha256} could not be removed") as caught:
        ctx.release(pinned)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert Path(pinned.ref).is_dir() and ctx.pin_file(path, sha256=_sha(path)) == pinned
    monkeypatch.setattr(images_module, "remove_extracted", removal)
    ctx.close()
    assert not Path(pinned.ref).exists() and _kinds(ctx) == ["image_extracted", "image_removed"]


def test_a_published_image_is_extracted_from_the_store_and_never_copied_to_the_node(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    pinned = ctx.node_image("k1", expect)
    stored = ctx.store.path("cache/images/k1") / IMAGE_FILE
    [role] = ctx.launcher.launched
    assert role.argv[4:6] == (str(stored), pinned.file_sha256)
    assert _extracts(tmp_path) == [str(_copy_of(ctx, pinned.file_sha256))]
    assert not (copies_root(ctx.scratch, ctx.job_id) / "images").exists()
    assert ctx.cache_entries == {"images/k1": expect}
    with pytest.raises(ValueError, match="never copied to the node"):
        ctx.materialize("images", "k1", expect)


def test_a_pin_reads_the_store_file_once_and_extracts_from_its_own_copy(tmp_path, apptainer, reads):
    """The shared filesystem is read once per pin: the extraction role copies the entry's file
    into its own directory, and both the check and apptainer read the copy, which goes after."""
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    pinned = ctx.node_image("k1", expect)
    stored, copy = ctx.store.path("cache/images/k1") / IMAGE_FILE, _copy_of(ctx, pinned.file_sha256)
    assert reads.read_text().splitlines() == [f"cp {stored}", f"sha256sum < {copy}"]
    assert _extracts(tmp_path) == [str(copy)] and not copy.exists()


def test_a_tampered_image_file_in_the_store_is_refused_before_it_is_extracted(tmp_path, apptainer, reads):
    """The copy differs from the record, so the role reads the store's file again, and that one
    differs too: the entry is refused, which a retry would meet again."""
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    stored = ctx.store.path("cache/images/k1") / IMAGE_FILE
    stored.write_bytes(b"other bytes\n")
    with pytest.raises(StageFault, match=f"file {stored} has sha256 {_sha(stored)}, not "
                                         f"{hashlib.sha256(IMAGE).hexdigest()}") as caught:
        ctx.node_image("k1", expect)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert len(ctx.launcher.launched) == 1 and _extracts(tmp_path) == [] and _sandboxes(ctx) == []
    copy = ctx.scratch / "roles" / ctx.launcher.launched[0].name / IMAGE_FILE
    assert reads.read_text().splitlines() == [f"cp {stored}", f"sha256sum < {copy}", f"sha256sum < {stored}"]


def _appending_cp(tmp_path: Path, under: Path) -> Path:
    """A ``cp`` that appends a byte to every copy of a file under *under*, as a copy that went
    wrong on the way would; the directory to put first on PATH."""
    bin_dir = tmp_path / "appending-bin"
    bin_dir.mkdir(exist_ok=True)
    cp = bin_dir / "cp"
    real = shutil.which("cp", path="/usr/bin:/bin")
    cp.write_text(f'#!/bin/sh\n{real} "$@" || exit\n'
                  f'case "$2" in {shlex.quote(str(under))}/*) printf x >> "$3";; esac\n')
    cp.chmod(0o755)
    return bin_dir


def test_an_image_copy_gone_wrong_on_the_way_is_a_retryable_infra_fault(tmp_path, apptainer, monkeypatch):
    """The store's file is sound and only the copy differs: the role exits with its own status
    and no refusal, so a retry copies the file again."""
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    stored = ctx.store.path("cache/images/k1") / IMAGE_FILE
    monkeypatch.setenv("PATH", f"{_appending_cp(tmp_path, ctx.store.root)}:{os.environ['PATH']}")
    with pytest.raises(StageFault, match="harness role extract-.* ended exited with status 75") as caught:
        ctx.node_image("k1", expect)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert _sha(stored) == hashlib.sha256(IMAGE).hexdigest() and _extracts(tmp_path) == []
    assert _sandboxes(ctx) == []


def test_a_repeat_node_image_within_a_stage_asks_the_store_nothing(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    first = ctx.node_image("k1", expect)
    ctx.store.path("cache/images/k1/ENTRY.json").write_text("not json")
    assert ctx.node_image("k1", expect) == first and len(ctx.launcher.launched) == 1


def test_a_stopping_stage_pins_no_published_image(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    ctx.stopping = True
    with pytest.raises(StageFault, match="stopping") as caught:
        ctx.node_image("k1", expect)
    assert caught.value.fault.retryable and ctx.launcher.launched == [] and ctx.cache_entries == {}


def test_a_container_records_the_image_it_runs(tmp_path, apptainer):
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    pinned = ctx.pin_file(path, sha256=digest)
    spec = ContainerSpec(image=pinned, command=("true",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="harness")
    ctx.wait_roles()
    assert ctx.image_digests == {"srv": digest}


def test_a_stage_leaves_the_copies_of_other_entries_for_the_job(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    assert ctx.cache is not None
    found = populate(ctx.cache, "weights", "org+m@" + "a" * 40, lambda d: (d / "w.bin").write_bytes(b"w"),
                     holder=Holder.here("test")).entry
    copy = ctx.materialize("weights", found.key, entry_sha256(found))
    ctx.node_image("k1", _published(ctx, "k1"))
    ctx.close()
    assert (copy / "w.bin").read_bytes() == b"w" and _sandboxes(ctx) == []


class _PinsThen:
    """A stage that pins an image file, starts a container on it that sleeps, and then ends as
    *how* says: it raises a fault of its own, or is stopped while it waits for its role."""

    def __init__(self, path: Path, how: str) -> None:
        self.path = path
        self.how = how
        self.pinned: PinnedImage | None = None

    def run(self, ctx: StageContext) -> dict[str, str]:
        self.pinned = ctx.pin_file(self.path, sha256=_sha(self.path))
        spec = ContainerSpec(image=self.pinned, command=(sys.executable, "-c", "import time; time.sleep(30)"),
                             workdir=ctx.role_dir("srv") / "wd")
        ctx.start_container("srv", spec, SHAPE, owner="harness")
        if self.how == "failed":
            raise StageFault(infra_fault("the stage failed"))
        threading.Timer(0.5, ctx.stop_roles).start()
        fault = stage_fault(ctx.wait_roles())
        if fault is not None:
            raise StageFault(fault)
        return {}


@pytest.mark.parametrize("how", ["failed", "stopped"])
def test_a_stages_trees_are_gone_after_it_fails_or_is_stopped(tmp_path, apptainer, how):
    path, _ = _image(tmp_path)
    ctx = _ctx(tmp_path)
    stage = _PinsThen(path, how)
    with pytest.raises(StageFault):
        run_stage(stage, ctx)
    assert stage.pinned is not None and not os.path.lexists(Path(stage.pinned.ref)) and _sandboxes(ctx) == []


def test_a_tree_an_earlier_stage_left_is_reused_after_its_manifest_check_then_removed(tmp_path, apptainer):
    path, digest = _image(tmp_path)
    # An earlier stage of the job that ended without closing its context.
    left = _ctx(tmp_path).pin_file(path, sha256=_sha(path))
    later = _ctx(tmp_path, attempt=2)
    assert later.pin_file(path, sha256=_sha(path)) == left
    assert later.launcher.launched == [] and len(_extracts(tmp_path)) == 1
    assert _kinds(later) == ["image_reused"]
    [reused] = [e for e in read_events(later.events.path) if e["kind"] == "image_reused"]
    assert (reused["sif_sha256"], reused["manifest_sha256"]) == (digest, left.check_sha256)
    later.close()
    assert not os.path.lexists(Path(left.ref)) and _sandboxes(later) == []
    assert _kinds(later) == ["image_reused", "image_removed"]


def test_a_tree_a_closing_stage_cannot_remove_is_left_then_reused_and_removed_by_the_next(tmp_path, apptainer,
                                                                                        monkeypatch, caplog):
    path, _ = _image(tmp_path)
    first = _ctx(tmp_path)
    pinned = first.pin_file(path, sha256=_sha(path))
    removal = images_module.remove_extracted

    def stuck(root: Path) -> None:
        raise PermissionError(f"cannot remove {root}")

    monkeypatch.setattr(images_module, "remove_extracted", stuck)
    first.close()
    assert Path(pinned.ref).is_dir() and "image_removed" not in _kinds(first)
    assert f"extracted image {Path(pinned.ref)} could not be removed" in caplog.text
    monkeypatch.setattr(images_module, "remove_extracted", removal)
    later = _ctx(tmp_path, attempt=2)
    assert later.pin_file(path, sha256=_sha(path)) == pinned and _kinds(later) == ["image_reused"]
    later.close()
    assert not Path(pinned.ref).exists() and _kinds(later)[-1] == "image_removed"


def test_another_job_on_the_node_extracts_for_itself(tmp_path, apptainer):
    path, _ = _image(tmp_path)
    _ctx(tmp_path).pin_file(path, sha256=_sha(path))
    _ctx(tmp_path, job="101").pin_file(path, sha256=_sha(path))
    assert len(_extracts(tmp_path)) == 2


def _tamper(tamper: str, root: Path, manifest: Path, digest: str) -> None:
    target = root / ".fake-image"
    if tamper == "rewrite":
        target.write_bytes(b"other bytes\n")
    elif tamper == "rewrite_same_size_and_time":
        st = target.stat()
        target.write_bytes(bytes(len(IMAGE)))
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))
    elif tamper == "add":
        (root / "etc" / "planted").write_text("x")
    elif tamper == "remove":
        (root / "bin").rmdir()
    elif tamper == "chmod":
        target.chmod(0o777)
    elif tamper == "relink":
        (root / "link").symlink_to("/etc")
    elif tamper == "manifest_of_another":
        manifest.write_text(manifest.read_text().replace(digest, "f" * 64))
    elif tamper == "manifest_garbage":
        manifest.write_text("{")
    elif tamper == "manifest_gone":
        manifest.unlink()
    else:
        os.rename(root, root.with_name("moved"))
        root.symlink_to(root.with_name("moved"))


@pytest.mark.parametrize("tamper", ["rewrite", "rewrite_same_size_and_time", "add", "remove", "chmod",
                                    "relink", "manifest_of_another", "manifest_garbage", "manifest_gone",
                                    "root_a_link"])
def test_a_tampered_tree_an_earlier_stage_left_is_extracted_again(tmp_path, apptainer, tamper):
    path, digest = _image(tmp_path)
    _ctx(tmp_path).pin_file(path, sha256=_sha(path))
    root, manifest = sandbox_paths(tmp_path / "node" / "hls-100", digest)
    # Past the filesystem clock's tick, so a write moves a change time even where it is coarse.
    time.sleep(0.02)
    _tamper(tamper, root, manifest, digest)
    later = _ctx(tmp_path, attempt=2)
    pinned = later.pin_file(path, sha256=_sha(path))
    assert _kinds(later) == ["image_reextracted", "image_extracted"] and len(_extracts(tmp_path)) == 2
    [event] = [e for e in read_events(later.events.path) if e["kind"] == "image_reextracted"]
    assert event["sif_sha256"] == digest and event["problem"]
    assert Path(pinned.ref) == root and root.is_dir() and not root.is_symlink()
    assert (Path(pinned.ref) / ".fake-image").read_bytes() == IMAGE and not (root / "etc" / "planted").exists()
    assert pinned.check_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert later.pin_file(path, sha256=_sha(path)) == pinned
    assert _kinds(later) == ["image_reextracted", "image_extracted"]


def test_three_images_pin_in_one_stage_on_a_node_with_exactly_the_stages_disk(tmp_path, apptainer,
                                                                              monkeypatch):
    # Each extracted tree is 20 GiB by its apparent size, and the stage declares the 60 the
    # three take; the node has exactly that. Each extraction role's grant is checked as the step
    # wrapper checks it, after the trees before it took their share.
    monkeypatch.setenv("FAKE_TREE_GIB", "20")
    scratch = tmp_path / "node" / "hls-100"
    launcher = LocalLauncher(gpus=2, node_disk_gib=60, scratch=scratch)
    ctx = _ctx(tmp_path, launcher=launcher, disk_gib=60)
    pinned = [_image_pin(ctx, tmp_path, f"image {n}\n".encode()) for n in range(3)]
    assert len({p.ref for p in pinned}) == 3 and _kinds(ctx) == ["image_extracted"] * 3
    for spec in launcher.launched:
        assert spec.shape.disk_gib == 0, spec.name
        report = RoleReport.model_validate_json((spec.log_dir / f"{spec.name}.report.json").read_text())
        assert report.mismatches == [], spec.name
    assert [round(h.report(1.0).granted.disk_free_gib) for h in launcher.handles] == [60, 40, 20]


def test_two_stages_of_a_job_run_on_a_node_whose_disk_fits_the_larger_stage_not_the_sum(tmp_path, apptainer,
                                                                                         monkeypatch):
    # Trees of 10 GiB by their apparent size. The first stage declares 30 for its three; the
    # second declares 20, one tree and the 10 its work role declares. The node has 30: enough for
    # the larger stage, not for the two together, which it needs only if a stage's trees outlive it.
    monkeypatch.setenv("FAKE_TREE_GIB", "10")
    scratch = tmp_path / "node" / "hls-100"
    launcher = LocalLauncher(gpus=2, node_disk_gib=30, scratch=scratch)
    first = _ctx(tmp_path, launcher=launcher, disk_gib=30)
    for n in range(3):
        _image_pin(first, tmp_path, f"image {n}\n".encode())
    first.close()
    second = _ctx(tmp_path, attempt=2, launcher=launcher, disk_gib=20)
    _image_pin(second, tmp_path, b"image 3\n")
    second.start_role("work", [sys.executable, "-c", "pass"],
                      ResourceShape(cpus=2, memory_gib=2, disk_gib=10, time_sec=60), owner="harness")
    [outcome] = second.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0, outcome.detail
    assert round(launcher.handles[-1].report(1.0).granted.disk_free_gib) == 20
    second.close()
    assert not list((scratch / "sandboxes").iterdir())


def test_the_local_launchers_node_disk_refuses_a_role_that_declares_more_than_is_left(tmp_path, apptainer,
                                                                                     monkeypatch):
    monkeypatch.setenv("FAKE_TREE_GIB", "20")
    scratch = tmp_path / "node" / "hls-100"
    ctx = _ctx(tmp_path, launcher=LocalLauncher(gpus=2, node_disk_gib=60, scratch=scratch), disk_gib=60)
    _image_pin(ctx, tmp_path)
    ctx.start_role("big", [sys.executable, "-c", "pass"], ResourceShape(cpus=2, memory_gib=2, disk_gib=41,
                                                                        time_sec=60), owner="harness")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "grant_mismatch" and "disk_gib: declared 41" in (outcome.detail or "")


def test_the_local_launchers_node_disk_goes_with_a_scratch():
    with pytest.raises(ValueError, match="go together"):
        LocalLauncher(gpus=0, node_disk_gib=60)


def test_a_published_image_is_pinned_by_the_digest_its_producer_recorded(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    pinned = ctx.node_image("k1", expect)
    assert pinned.file_sha256 == hashlib.sha256(IMAGE).hexdigest()
    assert ctx.cache_entries == {"images/k1": expect}
    with pytest.raises(StageFault, match="has ENTRY.json digest") as caught:
        _ctx(tmp_path, attempt=2).node_image("k1", "e" * 64)
    assert not caught.value.fault.retryable


def test_an_images_entry_without_an_image_file_is_refused(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    found = populate(ctx.cache, "images", "k2", lambda d: (d / "other.txt").write_bytes(b"x"),
                     holder=Holder.here("test")).entry
    with pytest.raises(StageFault, match="holds no image.sif") as caught:
        ctx.node_image("k2", entry_sha256(found))
    assert not caught.value.fault.retryable and ctx.launcher.launched == []


def test_a_node_local_file_that_is_not_the_reported_image_is_refused_and_retried(tmp_path, apptainer):
    """A file no entry holds yet was written by a role of this attempt, which a retry runs
    again: a mismatch is retried. A store entry's mismatch is not."""
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    with pytest.raises(StageFault, match=f"file {path} has sha256 {digest}, not {'d' * 64}") as caught:
        ctx.pin_file(path, sha256="d" * 64)
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert len(ctx.launcher.launched) == 1 and _extracts(tmp_path) == [] and _sandboxes(ctx) == []


def test_a_published_file_that_is_not_the_named_image_is_refused_and_not_retried(tmp_path, apptainer):
    """A published file the plan names by its digest (``stored``) is read unchanged by a retry,
    so its mismatch is final."""
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    with pytest.raises(StageFault, match=f"file {path} has sha256 {digest}, not {'d' * 64}") as caught:
        ctx.pin_file(path, sha256="d" * 64, contestant=True, stored=True)
    assert caught.value.fault.fault_class == "infra" and not caught.value.fault.retryable
    assert _sandboxes(ctx) == []


def test_an_image_file_that_cannot_be_read_is_a_retryable_fault(tmp_path, apptainer):
    path, digest = _image(tmp_path)
    path.unlink()
    with pytest.raises(StageFault, match=f"extract-{digest[:12]}") as caught:
        _ctx(tmp_path).pin_file(path, sha256=digest)
    assert caught.value.fault.retryable and _extracts(tmp_path) == []


def test_a_refusal_line_counts_only_with_the_refusal_status(tmp_path, apptainer):
    """A role that prints a refusal line and exits otherwise failed in its own way: its fault is
    the role's, and retried."""
    (apptainer / "apptainer").write_text('#!/bin/sh\necho "refused: not this way" >&2\nexit 1\n')
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    with pytest.raises(StageFault, match=f"extract-{digest[:12]}") as caught:
        ctx.pin_file(path, sha256=digest)
    assert caught.value.fault.retryable and caught.value.fault.evidence["roles"] == f"extract-{digest[:12]}"


@pytest.mark.parametrize("code", [DIGEST_MISMATCH_EXIT, 3, 4])
@pytest.mark.parametrize("owner", ["harness", "contestant"])
def test_a_refusal_is_read_only_from_a_harness_role(tmp_path, apptainer, owner, code):
    """A contestant's code chooses its own exit status and stderr, so a ``refused:`` line from it
    counts for nothing and its outcome stays as it is; a harness role's is its refusal."""
    ctx = _ctx(tmp_path)
    argv = ["/bin/sh", "-c", f"echo 'refused: no such thing' >&2; exit {code}"]
    ctx.start_role("r", argv, SHAPE, owner=owner)
    outcomes = ctx.wait_roles()
    found = ctx.refusal("r", outcomes, (code,))
    assert found == (Refusal(exit_code=code, reason="no such thing") if owner == "harness" else None)
    assert ctx.refusal("r", outcomes, (code + 1,)) is None
    [outcome] = outcomes
    assert (outcome.owner, outcome.ending, outcome.exit_code) == (owner, "exited", code)
    # A contestant's failing exit is its result; a harness role's, a retryable fault of its own.
    fault = stage_fault(outcomes)
    if owner == "contestant":
        assert fault is None
    else:
        assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)


def test_a_digest_that_is_not_bare_hex_pins_nothing(tmp_path, apptainer):
    path, _ = _image(tmp_path)
    ctx = _ctx(tmp_path)
    with pytest.raises(ValueError, match="not a bare hex"):
        ctx.pin_file(path, sha256="not hex")
    assert ctx.launcher.launched == [] and _sandboxes(ctx) == []


def test_a_second_extraction_of_an_image_in_a_stage_has_a_role_name_of_its_own(tmp_path, apptainer,
                                                                               monkeypatch):
    """A second extraction after a failed one starts a role of its own, which the faulted stage
    stops, rather than refusing a name the stage has taken."""
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    monkeypatch.setenv("FAKE_EXTRACT_FAIL", "1")
    with pytest.raises(StageFault):
        ctx.pin_file(path, sha256=digest)
    monkeypatch.delenv("FAKE_EXTRACT_FAIL")
    with pytest.raises(StageFault, match=f"extract-{digest[:12]}.2: stopped"):
        ctx.pin_file(path, sha256=digest)
    assert [s.name for s in ctx.launcher.launched] == [f"extract-{digest[:12]}", f"extract-{digest[:12]}.2"]


@pytest.mark.parametrize("how", ["nothing", "part"])
def test_a_failed_extraction_is_a_retryable_fault_and_leaves_nothing(tmp_path, apptainer, monkeypatch, how):
    if how == "nothing":
        monkeypatch.setenv("FAKE_EXTRACT_FAIL", "1")
    else:
        # An extraction that wrote part of its tree, read-only as images have it, and then failed.
        (apptainer / "apptainer").write_text('#!/bin/sh\nmkdir -p "$3/usr" && chmod 555 "$3/usr" && exit 1\n')
    path, digest = _image(tmp_path)
    ctx = _ctx(tmp_path)
    with pytest.raises(StageFault, match=f"extract-{digest[:12]}") as caught:
        ctx.pin_file(path, sha256=_sha(path))
    assert caught.value.fault.fault_class == "infra" and caught.value.fault.retryable
    assert caught.value.fault.evidence["roles"] == f"extract-{digest[:12]}"
    assert _sandboxes(ctx) == []


@pytest.mark.parametrize("wrote", ["nothing", "a file", "a link to a directory"])
def test_an_extraction_that_writes_no_directory_is_a_retryable_fault(tmp_path, apptainer, wrote):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    script = {"nothing": "exit 0", "a file": 'echo x > "$3"',
              "a link to a directory": f'ln -s {elsewhere} "$3"'}[wrote]
    (apptainer / "apptainer").write_text(f"#!/bin/sh\n{script}\n")
    path, _ = _image(tmp_path)
    ctx = _ctx(tmp_path)
    with pytest.raises(StageFault, match="without extracting") as caught:
        ctx.pin_file(path, sha256=_sha(path))
    assert caught.value.fault.retryable and _sandboxes(ctx) == [] and elsewhere.is_dir()


@pytest.mark.parametrize("command", ["import time; time.sleep(30)", "import sys; sys.exit(3)"],
                         ids=["running", "ended, not waited for"])
def test_an_extraction_waits_for_no_role_the_stage_has_not_waited_for(tmp_path, apptainer, command):
    path, _ = _image(tmp_path)
    ctx = _ctx(tmp_path)
    handle = ctx.start_role("busy", [sys.executable, "-c", command], SHAPE, owner="harness")
    if "exit" in command:
        handle.wait(30)
    with pytest.raises(RuntimeError, match=r"roles \['busy'\] have not been waited for"):
        ctx.pin_file(path, sha256=_sha(path))
    assert [s.name for s in ctx.launcher.launched] == ["busy"]
    ctx.close()


def test_a_stage_whose_memory_leaves_the_extraction_none_is_refused(tmp_path, apptainer):
    path, _ = _image(tmp_path)
    plan = noop_plan(tmp_path)
    # A runner that kept the whole of the stage's 6 GiB; the planner refuses such a plan.
    run = plan.settings.run.model_copy(update={"runner_memory_gib": 6})
    plan = plan.model_copy(update={"settings": plan.settings.model_copy(update={"run": run})})
    ctx = _ctx(tmp_path, plan=plan)
    with pytest.raises(StageFault, match="leaves role extract-.* none") as caught:
        ctx.pin_file(path, sha256=_sha(path))
    assert not caught.value.fault.retryable and ctx.launcher.launched == []


@pytest.mark.parametrize("owner", ["harness", "contestant"])
@pytest.mark.parametrize("image", ["stand_in", "earlier_stage", "other_job", "forged"])
def test_a_container_runs_only_from_an_image_this_context_pinned(tmp_path, apptainer, owner, image):
    path, _ = _image(tmp_path)
    earlier = _ctx(tmp_path, attempt=1).pin_file(path, sha256=_sha(path))
    chosen = {"stand_in": stand_in(tmp_path / "anywhere"), "earlier_stage": earlier,
              "other_job": _ctx(tmp_path, job="101").pin_file(path, sha256=_sha(path)),
              "forged": PinnedImage(ref=earlier.ref, file_sha256=earlier.file_sha256,
                                    check_sha256="0" * 64)}[image]
    ctx = _ctx(tmp_path, attempt=2)

    def refused() -> None:
        spec = ContainerSpec(image=chosen, command=("true",), workdir=ctx.role_dir("srv") / "wd")
        with pytest.raises(ValueError, match="is not one this stage pinned"):
            ctx.start_container("srv", spec, SHAPE, owner=owner)
        assert ctx.launcher.launched == []

    refused()
    mine = ctx.pin_file(path, sha256=_sha(path))
    assert mine == earlier
    if image != "earlier_stage":
        refused()
    spec = ContainerSpec(image=mine, command=("true",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner=owner)
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "exited" and outcome.exit_code == 0


def test_a_closed_context_pins_nothing_and_leaves_nothing_on_the_node(tmp_path, apptainer):
    ctx = _ctx(tmp_path)
    expect = _published(ctx, "k1")
    other = _published(ctx, "k2")
    ctx.node_image("k1", expect)
    ctx.close()
    pins = (lambda: ctx.node_image("k1", expect), lambda: ctx.node_image("k2", other),
            lambda: _image_pin(ctx, tmp_path, b"another image\n"))
    for pin in pins:
        with pytest.raises(StageFault, match="stopping"):
            pin()
    assert len(ctx.launcher.launched) == 1 and _sandboxes(ctx) == []
    images = copies_root(ctx.scratch, ctx.job_id) / "images"
    assert not images.exists() or list(images.iterdir()) == []
    assert list((ctx.scratch / "roles").iterdir()) == []


def test_a_context_without_an_image_runtime_pins_nothing(tmp_path, apptainer):
    path, _ = _image(tmp_path)
    plan = noop_plan(tmp_path)
    store = PosixStore(plan.store_root)
    ctx = StageContext(store=store, plan=plan, stage=plan.stage("noop_a"), attempt=1,
                       launcher=LocalLauncher(gpus=2), watch=None, events=EventLog(tmp_path / "events.jsonl"),
                       inputs={}, deadline=time.time() + 60, scratch=tmp_path / "node" / "hls-100")
    with pytest.raises(RuntimeError, match="no image runtime"):
        ctx.pin_file(path, sha256=_sha(path))
