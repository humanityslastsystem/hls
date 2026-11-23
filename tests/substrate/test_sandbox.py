from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import EXTRACT_SHELL, build_env, extract_argv
from hls.store.digest import COPY_FAILED_EXIT, DIGEST_MISMATCH_EXIT
from hls.substrate.container import ContainerSpec, PinnedImage
from hls.substrate.sandbox import (SANDBOXES, TreeManifest, make_removable, remove_extracted, renew_directory_times,
                                   sandbox_paths, scan_tree, tree_problem, write_manifest)
from hls.util import remove_tree
from tests.substrate import fake_apptainer
from tests.substrate.sandboxes import install_read_log, stand_in

SHA = "ab" * 32


def _tree(root: Path) -> Path:
    (root / "usr" / "bin").mkdir(parents=True)
    (root / "usr" / "bin" / "python3").write_bytes(b"#!python\n")
    (root / "etc").mkdir()
    (root / "etc" / "os-release").write_text("ID=ubuntu\n")
    (root / "bin").symlink_to("usr/bin")
    return root


def test_the_sandbox_of_an_image_file_is_keyed_by_its_digest_in_the_jobs_scratch(tmp_path):
    assert sandbox_paths(tmp_path, SHA) == (tmp_path / SANDBOXES / SHA, tmp_path / SANDBOXES / f"{SHA}.json")
    with pytest.raises(ValueError):
        sandbox_paths(tmp_path, "../" + SHA[3:])


def test_a_scan_records_every_entry_and_links_without_following_them(tmp_path):
    root = _tree(tmp_path / "root")
    entries = {e.path: e for e in scan_tree(root)}
    assert list(entries)[0] == "." and set(entries) == {".", "bin", "etc", "etc/os-release", "usr", "usr/bin",
                                                        "usr/bin/python3"}
    assert entries["bin"].target == "usr/bin" and stat.S_ISLNK(entries["bin"].mode)
    assert entries["usr/bin/python3"].size == 9 and entries["usr/bin/python3"].target is None


def test_a_tree_that_matches_its_manifest_has_no_problem_and_names_its_digest(tmp_path):
    root = _tree(tmp_path / "root")
    digest = write_manifest(root, SHA, tmp_path / "m.json")
    assert tree_problem(root, tmp_path / "m.json", SHA) == (None, digest)
    manifest = TreeManifest.model_validate_json((tmp_path / "m.json").read_bytes())
    assert manifest.sif_sha256 == SHA and manifest.entries == scan_tree(root)


def test_a_problem_names_what_changed_was_added_and_was_removed(tmp_path):
    root = _tree(tmp_path / "root")
    write_manifest(root, SHA, tmp_path / "m.json")
    (root / "etc" / "os-release").write_text("ID=other\n")
    (root / "etc" / "planted").write_text("x")
    (root / "usr" / "bin" / "python3").unlink()
    problem, digest = tree_problem(root, tmp_path / "m.json", SHA)
    assert digest is None and problem is not None
    assert "changed ['etc', 'etc/os-release', 'usr/bin']" in problem
    assert "added ['etc/planted']" in problem and "removed ['usr/bin/python3']" in problem


def test_a_problem_names_three_paths_of_a_kind_and_counts_the_rest(tmp_path):
    root = _tree(tmp_path / "root")
    write_manifest(root, SHA, tmp_path / "m.json")
    for n in range(5):
        (root / f"planted-{n}").write_text("x")
    problem = tree_problem(root, tmp_path / "m.json", SHA)[0]
    assert "added ['planted-0', 'planted-1', 'planted-2'] and 2 more" in problem


def test_a_manifest_of_another_image_or_one_that_does_not_parse_is_a_problem(tmp_path):
    root = _tree(tmp_path / "root")
    write_manifest(root, SHA, tmp_path / "m.json")
    assert "is of image file" in tree_problem(root, tmp_path / "m.json", "cd" * 32)[0]
    assert "cannot be read" in tree_problem(root, tmp_path / "absent.json", SHA)[0]
    (tmp_path / "m.json").write_text(json.dumps({"sif_sha256": SHA}))
    assert "cannot be read" in tree_problem(root, tmp_path / "m.json", SHA)[0]
    write_manifest(root, SHA, tmp_path / "m.json")
    remove_extracted(root)
    assert "cannot be walked" in tree_problem(root, tmp_path / "m.json", SHA)[0]


@pytest.fixture
def locked(tmp_path):
    """A tree with locked directories (``_locked``) under *tmp_path*, unlocked again at teardown
    with ``chmod -R`` whatever the test did, so pytest can remove it."""
    yield _locked(tmp_path / "root", tmp_path / "outside")
    subprocess.run(["chmod", "-R", "u+rwx", str(tmp_path)], check=False)


def _locked(root: Path, outside: Path) -> Path:
    """*root* with a directory the owner cannot read, one it cannot write, a link to the
    read-only directory *outside* and a read-only root."""
    _tree(root)
    outside.mkdir(mode=0o500)
    (root / "out").symlink_to(outside)
    (root / "usr" / "share").mkdir()
    (root / "usr" / "share" / "f").write_text("x")
    (root / "etc" / "os-release").chmod(0o444)
    (root / "usr" / "share").chmod(0o000)
    (root / "usr").chmod(0o555)
    root.chmod(0o500)
    return root


def test_a_tree_made_removable_gives_the_owner_every_directory_and_leaves_files_alone(tmp_path, locked):
    root = locked
    make_removable(root)
    for d in (root, root / "usr", root / "usr" / "share"):
        assert stat.S_IMODE(d.stat().st_mode) & 0o700 == 0o700
    assert stat.S_IMODE((root / "usr").stat().st_mode) == 0o755
    assert stat.S_IMODE((tmp_path / "outside").stat().st_mode) == 0o500, "a link was followed"
    assert stat.S_IMODE((root / "etc" / "os-release").stat().st_mode) == 0o444
    remove_tree(root)
    assert not root.exists()


def test_a_tree_with_unreadable_and_read_only_directories_is_removed_whole(tmp_path, locked):
    remove_extracted(locked)
    assert not locked.exists() and (tmp_path / "outside").is_dir()


def test_removing_a_link_removes_the_link_only(tmp_path):
    target = _tree(tmp_path / "target")
    target.chmod(0o500)
    (tmp_path / "link").symlink_to(target)
    remove_extracted(tmp_path / "link")
    assert not os.path.lexists(tmp_path / "link") and (target / "etc" / "os-release").exists()
    assert stat.S_IMODE(target.stat().st_mode) == 0o500, "the link's target was made removable"
    target.chmod(0o700)
    remove_extracted(tmp_path / "absent")


def test_the_runtime_extracts_with_apptainer_build_sandbox_on_exact_paths(tmp_path):
    apptainer = "/opt/apptainer/bin/apptainer"
    image, dest = Path("/shared/store/cache/images/k/image.sif"), Path("/node/hls-1/sandboxes/.x")
    copy = Path("/node/hls-1/roles/extract-x/image.sif")
    assert extract_argv(image, dest, sha256=SHA, copy=copy, apptainer=apptainer) == [
        "/bin/sh", "-c", EXTRACT_SHELL, "hls-extract", str(image), SHA, "/opt/apptainer/bin/apptainer",
        str(dest), str(copy)]
    assert '"$3" build --sandbox "$4" "$5"' in EXTRACT_SHELL
    assert build_env(cache_dir=tmp_path / "c", tmp_dir=tmp_path / "t") == {
        "APPTAINER_CACHEDIR": str(tmp_path / "c"), "APPTAINER_TMPDIR": str(tmp_path / "t")}
    for image, dest, copy in ((Path("image.sif"), Path("/node/s"), Path("/node/c")),
                              (Path("/node/i.sif"), Path("/node/a:b"), Path("/node/c")),
                              (Path("/node/../i.sif"), Path("/node/s"), Path("/node/c")),
                              (Path("/node/i.sif"), Path("/node/s"), Path("c.sif")),
                              (Path("/node/i.sif"), Path("/node/s"), Path("/node/c d"))):
        with pytest.raises(ValueError, match="image file|sandbox|copy"):
            extract_argv(image, dest, sha256=SHA, copy=copy, apptainer=apptainer)
    with pytest.raises(ValueError, match="bare hex"):
        extract_argv(Path("/node/i.sif"), Path("/node/s"), sha256="sha256:" + SHA, copy=Path("/node/c"),
                     apptainer=apptainer)


def test_the_extraction_copies_the_image_file_once_checks_the_copy_and_extracts_from_it(tmp_path):
    """The step reads the file once, into its own directory, hashes that copy, extracts from it
    and removes it whatever happened, so apptainer reads exactly the bytes that were checked."""
    apptainer = tmp_path / "apptainer"
    log = Path(f"{apptainer}.log")
    # It logs its arguments and whether the file it was handed is there, and exits $FAKE_RC.
    apptainer.write_text('#!/bin/sh\necho "$* $([ -f "$4" ] && echo there)" >> "$0.log"\n'
                         'exit "${FAKE_RC:-0}"\n')
    apptainer.chmod(0o755)
    image, dest, copy = tmp_path / "store" / "image.sif", tmp_path / "out", tmp_path / "role" / "image.sif"
    image.parent.mkdir()
    copy.parent.mkdir()
    image.write_bytes(b"an image")
    reads = tmp_path / "reads.txt"
    install_read_log(tmp_path / "bin", reads)

    def extract(sha256: str, rc: int = 0) -> subprocess.CompletedProcess:
        env = {"PATH": f"{tmp_path / 'bin'}:/usr/bin:/bin", "FAKE_RC": str(rc)}
        return subprocess.run(extract_argv(image, dest, sha256=sha256, copy=copy, apptainer=str(apptainer)),
                              capture_output=True, text=True, timeout=30, env=env)

    got = hashlib.sha256(b"an image").hexdigest()
    refused = extract(SHA)
    assert refused.returncode == DIGEST_MISMATCH_EXIT and not log.exists() and not copy.exists()
    assert refused.stderr == f"refused: file {image} has sha256 {got}, not {SHA}\n"
    # The copy differed, so the file was read again, and it differs too: the file is refused.
    assert reads.read_text().splitlines() == [f"cp {image}", f"sha256sum < {copy}", f"sha256sum < {image}"]
    done = extract(got)
    assert done.returncode == 0 and log.read_text() == f"build --sandbox {dest} {copy} there\n"
    assert not copy.exists()
    failed = extract(got, rc=7)
    assert failed.returncode == 7 and not copy.exists()
    # A copy that lands other bytes than a sound file holds is the copy's failure, not a refusal.
    appending = tmp_path / "appending"
    appending.mkdir()
    real_cp = shutil.which("cp", path="/usr/bin:/bin")
    (appending / "cp").write_text(f'#!/bin/sh\n{real_cp} "$@" && printf x >> "$3"\n')
    (appending / "cp").chmod(0o755)
    wrong = subprocess.run(extract_argv(image, dest, sha256=got, copy=copy, apptainer=str(apptainer)), capture_output=True,
                           text=True, timeout=30, env={"PATH": f"{appending}:/usr/bin:/bin", "FAKE_RC": "0"})
    assert wrong.returncode == COPY_FAILED_EXIT and "refused" not in wrong.stderr
    assert wrong.stderr == f"copy failed: the copy of file {image} differs from it\n"
    assert not copy.exists() and log.read_text().count("build --sandbox") == 2
    # A copy gone wrong of a file that differs too: the refusal names the file's own digest.
    both = subprocess.run(extract_argv(image, dest, sha256=SHA, copy=copy, apptainer=str(apptainer)), capture_output=True,
                          text=True, timeout=30, env={"PATH": f"{appending}:/usr/bin:/bin", "FAKE_RC": "0"})
    assert both.returncode == DIGEST_MISMATCH_EXIT
    assert both.stderr == f"refused: file {image} has sha256 {got}, not {SHA}\n"
    image.unlink()
    unreadable = extract(SHA)
    assert unreadable.returncode == 74 and "refused" not in unreadable.stderr and not copy.exists()


def test_renewing_a_trees_directory_times_leaves_its_files_and_what_its_links_point_to(tmp_path):
    """A node's periodic ``systemd-tmpfiles --clean`` removes an empty directory under /tmp whose
    access and modification times are both older than its age, and an extracted tree carries its
    image's times, so every directory gets the current time. Its files keep their times, which
    the clean weighs with their change time, and a link is not followed out of the tree."""
    import time

    old = time.time() - 400 * 86400
    root, outside = tmp_path / "tree", tmp_path / "outside"
    for d in ("tmp", "var/tmp", "usr/share/ro"):
        (root / d).mkdir(parents=True)
    outside.mkdir()
    (root / "usr" / "f").write_text("x")
    (root / "usr" / "link").symlink_to(outside)
    for p in (outside, root / "usr" / "f", root / "usr" / "share" / "ro", root / "usr" / "share", root / "var" / "tmp",
              root / "var", root / "tmp", root / "usr", root):
        os.utime(p, (old, old))
    started = time.time() - 1
    renew_directory_times(root)
    dirs = [root, *(p for p in root.rglob("*") if p.is_dir() and not p.is_symlink())]
    assert len(dirs) == 7
    assert all(p.stat().st_mtime >= started and p.stat().st_atime >= started for p in dirs)
    assert (root / "usr" / "f").stat().st_mtime == pytest.approx(old)
    assert outside.stat().st_mtime == pytest.approx(old)


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
def test_a_failed_reread_of_the_source_after_a_bad_copy_is_neither_refused_nor_a_failed_copy(tmp_path):
    """The copy differs and the source cannot be read again: the step knows neither which is bad,
    so it exits 74, its own failure, which is retried, with no refusal and the copy removed."""
    image, copy = tmp_path / "store" / "image.sif", tmp_path / "role" / "image.sif"
    image.parent.mkdir()
    copy.parent.mkdir()
    image.write_bytes(b"an image")
    got = hashlib.sha256(b"an image").hexdigest()
    hiding = tmp_path / "hiding"
    hiding.mkdir()
    real_cp = shutil.which("cp", path="/usr/bin:/bin")
    (hiding / "cp").write_text(f'#!/bin/sh\n{real_cp} "$@" && printf x >> "$3" && chmod 000 "$2"\n')
    (hiding / "cp").chmod(0o755)
    env = {"PATH": f"{hiding}:/usr/bin:/bin"}
    argv = extract_argv(image, tmp_path / "out", sha256=got, copy=copy, apptainer="/bin/false")
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=30, env=env)
    finally:
        image.chmod(0o644)
    assert done.returncode == 74 and "refused:" not in done.stderr and "copy failed:" not in done.stderr
    assert not copy.exists() and not (tmp_path / "out").exists()


@pytest.mark.parametrize("tool, script", [("cp", 'head -c 3 "$2" > "$3"; exit 1'), ("sha256sum", "exit 1")])
def test_an_extraction_whose_copy_fails_stops_on_its_own_and_is_no_refusal(tmp_path, tool, script):
    """A copy cut short (a full disk) leaves other bytes than the file holds, and a copy that
    cannot be read back gives no digest. Either way the step stops with its own failure, which
    is retried, never with the refusal a store entry's other bytes get, and extracts nothing."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / tool).write_text(f"#!/bin/sh\n{script}\n")
    (bin_dir / tool).chmod(0o755)
    apptainer = tmp_path / "apptainer"
    apptainer.write_text('#!/bin/sh\necho "$*" >> "$0.log"\n')
    apptainer.chmod(0o755)
    image, dest, copy = tmp_path / "image.sif", tmp_path / "out", tmp_path / "role" / "image.sif"
    copy.parent.mkdir()
    image.write_bytes(b"an image")
    argv = extract_argv(image, dest, sha256=hashlib.sha256(b"an image").hexdigest(), copy=copy,
                        apptainer=str(apptainer))
    done = subprocess.run(argv, capture_output=True, text=True, timeout=30,
                          env={"PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert done.returncode == 74 and "refused" not in done.stderr
    assert not copy.exists() and not Path(f"{apptainer}.log").exists()


def test_a_container_runs_from_its_extracted_directory_never_writable(tmp_path):
    spec = ContainerSpec(image=stand_in(Path("/node/hls-1/sandboxes/" + SHA)), command=("true",),
                         workdir=Path("/node/hls-1/roles/r/wd"))
    argv = ApptainerRuntime().argv(spec, env_file="/dev/fd/3")
    assert f"/node/hls-1/sandboxes/{SHA}" in argv
    assert not [a for a in argv if a.startswith(("--writable", "--overlay")) or a == "-w"]


def test_a_pinned_image_carries_bare_hex_digests():
    with pytest.raises(ValueError):
        PinnedImage(ref="/x", file_sha256="nothex", check_sha256=SHA)
    with pytest.raises(ValueError):
        PinnedImage(ref="/x", file_sha256=SHA, check_sha256="sha256:" + SHA)


def test_a_container_spec_takes_only_a_pinned_image(tmp_path):
    with pytest.raises(TypeError, match="PinnedImage"):
        ContainerSpec(image=tmp_path / "i.sif", command=("true",), workdir=tmp_path / "wd")


def _fake(tmp_path: Path, *args: str, **env: str) -> subprocess.CompletedProcess:
    fake = fake_apptainer.install(tmp_path / "bin")
    return subprocess.run([str(fake), *args], capture_output=True, text=True, timeout=30,
                          env={"PATH": os.environ["PATH"], **env})


def test_the_fake_extracts_an_image_file_into_a_new_directory(tmp_path):
    (tmp_path / "i.sif").write_bytes(b"bytes")
    done = _fake(tmp_path, "build", "--sandbox", str(tmp_path / "out"), str(tmp_path / "i.sif"))
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "out" / ".fake-image").read_bytes() == b"bytes"
    again = _fake(tmp_path, "build", "--sandbox", str(tmp_path / "out"), str(tmp_path / "i.sif"))
    assert again.returncode == 2 and "exists" in again.stderr
    missing = _fake(tmp_path, "build", "--sandbox", str(tmp_path / "out2"), str(tmp_path / "absent.sif"))
    assert missing.returncode == 255 and missing.stderr.startswith("FATAL")
    assert not (tmp_path / "out2").exists()
    make_removable(tmp_path / "out")


def test_the_fake_refuses_to_run_an_image_file_or_a_writable_image(tmp_path):
    (tmp_path / "i.sif").write_bytes(b"bytes")
    (tmp_path / "root").mkdir()
    spec = ContainerSpec(image=stand_in(tmp_path / "i.sif"), command=(sys.executable, "-c", "pass"),
                         workdir=tmp_path / "wd")
    fake = fake_apptainer.install(tmp_path / "bin")
    argv = ApptainerRuntime(apptainer=str(fake)).argv(spec, env_file="/dev/fd/3")
    done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert done.returncode == 2 and "not an extracted directory" in done.stderr
    writable = [a if a != str(tmp_path / "i.sif") else str(tmp_path / "root") for a in argv]
    writable.insert(2, "--writable")
    done = subprocess.run(writable, capture_output=True, text=True, timeout=30)
    assert done.returncode == 2 and "--writable" in done.stderr
