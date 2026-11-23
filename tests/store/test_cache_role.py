from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hls.store import cache_role
from hls.store.cache import (ENTRY_FILE, MAX_DEPTH, VERIFIED_FILE, CacheEntry, ContentCache, entry_sha256,
                             read_digests)
from hls.store.cache_role import CopyFailed, Refused, copy_entry, verify_copy
from hls.store.digest import COPY_FAILED_EXIT, DIGEST_MISMATCH_EXIT, sha256_bytes
from hls.store.lock import Holder
from hls.store.posix import PosixStore
from hls.util import remove_tree
from tests.store.entries import entry_reads, populate, recording_reads

FILES = {"config.json": b'{"a": 1}', "shards/w0.bin": b"\x00" * 1000, "shards/w1.bin": b"\x01" * 10}
#: The stand-in for ``python -m hls.store.cache_role`` whose copies land other bytes than they read.
FLIPPING = Path(__file__).with_name("flipping_cache_role.py")


def _entry(tmp_path: Path, files: dict[str, bytes] = FILES) -> CacheEntry:
    def fill(d: Path) -> None:
        for rel, data in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(data)

    return populate(ContentCache(PosixStore(tmp_path / "store")), "weights", "k", fill,
                    holder=Holder.here("test")).entry


def _digests(files: dict[str, bytes], entry: CacheEntry) -> dict[str, str]:
    return {**{p: sha256_bytes(b) for p, b in files.items()}, ENTRY_FILE: entry_sha256(entry)}


def _link_to_a_copy(path: Path) -> None:
    """Replace the file at *path* by a link to a copy of it outside its directory: the same bytes."""
    kept = path.parent.parent / f"kept-{path.name}"
    kept.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(kept)


def _role(*args: str, script: Path | None = None) -> subprocess.CompletedProcess:
    program = [str(script)] if script is not None else ["-m", "hls.store.cache_role"]
    return subprocess.run([sys.executable, *program, *args], capture_output=True, text=True, timeout=60)


def test_a_copy_holds_the_entrys_files_and_its_manifest_and_reports_each_digest(tmp_path):
    e = _entry(tmp_path)
    dest, report = tmp_path / "node" / "copy", tmp_path / "role" / "digests.json"
    dest.parent.mkdir()
    report.parent.mkdir()
    done = _role("copy", "--entry", str(e.path), "--expect", entry_sha256(e), "--dest", str(dest),
                 "--digests-out", str(report))
    assert done.returncode == 0, done.stderr
    assert read_digests(report) == _digests(FILES, e)
    copied = sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file())
    assert copied == sorted([ENTRY_FILE, *FILES])
    assert all((dest / p).read_bytes() == b for p, b in FILES.items())
    assert (dest / ENTRY_FILE).read_bytes() == (e.path / ENTRY_FILE).read_bytes()


def test_a_copy_reads_each_file_of_the_store_once_and_hashes_what_landed(tmp_path):
    """One read of the shared filesystem per file; each digest is of the copy, read back."""
    e = _entry(tmp_path)
    copy = tmp_path / "copy"
    with recording_reads() as reads:
        copy_entry(e.path, entry_sha256(e), copy)
    assert sorted(entry_reads(reads, [e.path]) + [p for p in reads if Path(p).name == ENTRY_FILE]) == sorted(
        [str(e.path / p) for p in FILES] + [str(e.path / ENTRY_FILE), str(copy / ENTRY_FILE)])
    assert sorted(entry_reads(reads, [copy])) == sorted(str(copy / p) for p in FILES)


@pytest.mark.parametrize("damage, says", [
    (lambda e: (e.path / ENTRY_FILE).write_bytes((e.path / ENTRY_FILE).read_bytes() + b"\n"),
     "the entry's ENTRY.json has sha256"),
    (lambda e: (e.path / "shards" / "w0.bin").write_bytes(b"\x02" * 1000), "changed shards/w0.bin"),
    (lambda e: (e.path / "shards" / "w0.bin").write_bytes(b"short"), "changed shards/w0.bin"),
    (lambda e: (e.path / "shards" / "w1.bin").unlink(), "missing shards/w1.bin"),
    (lambda e: (e.path / "extra.bin").write_bytes(b"x"), "unexpected extra.bin"),
    (lambda e: (e.path / "link").symlink_to(e.path / "config.json"), "link is a symlink"),
    (lambda e: os.mkfifo(e.path / "pipe"), "pipe is not a regular file or directory"),
    (lambda e: os.link(e.path / "config.json", e.path.parent / "twin"), "config.json has 2 hard links"),
    (lambda e: (e.path / "shards" / ENTRY_FILE).write_text("x"), "shards/ENTRY.json uses a reserved name"),
    (lambda e: (e.path / "shards" / VERIFIED_FILE).write_text("x"), "uses a reserved name"),
    (lambda e: (e.path / VERIFIED_FILE).write_text("x"), ".verified uses a reserved name"),
    (lambda e: _link_to_a_copy(e.path / ENTRY_FILE), "the entry's ENTRY.json is not a regular file"),
], ids=["manifest", "changed", "resized", "missing", "unexpected", "link", "fifo", "hard link", "reserved",
        "stamp name", "stamp at the top", "manifest link"])
def test_a_copy_refuses_an_entry_that_differs_from_its_manifest(tmp_path, damage, says):
    """Whatever the store holds that its manifest does not say is the entry's fault, which a
    retry would read again: the role refuses it, exiting with its refusal status."""
    e = _entry(tmp_path)
    damage(e)
    dest, report = tmp_path / "copy", tmp_path / "digests.json"
    done = _role("copy", "--entry", str(e.path), "--expect", entry_sha256(e), "--dest", str(dest),
                 "--digests-out", str(report))
    assert done.returncode == DIGEST_MISMATCH_EXIT, done.stderr
    last = done.stderr.strip().splitlines()[-1]
    assert last.startswith("refused: ") and says in last
    assert not report.exists()


def test_a_refusal_is_made_before_any_file_is_copied_when_the_tree_or_the_manifest_is_wrong(tmp_path):
    e = _entry(tmp_path)
    (e.path / "extra.bin").write_bytes(b"x")
    with pytest.raises(Refused, match="unexpected extra.bin"):
        copy_entry(e.path, entry_sha256(e), tmp_path / "copy")
    # Only the manifest was copied, to be checked.
    assert [p.name for p in (tmp_path / "copy").iterdir()] == [ENTRY_FILE]


def test_an_entry_deeper_than_the_limit_is_refused_without_recursing(tmp_path):
    e = _entry(tmp_path)
    p = e.path / "deep"
    for _ in range(sys.getrecursionlimit() + 100):
        p.mkdir()
        p = p / "a"
    with pytest.raises(Refused, match=f"nested deeper than {MAX_DEPTH} levels"):
        copy_entry(e.path, entry_sha256(e), tmp_path / "copy")
    # pytest removes old temporary directories by recursing, which a tree this deep would overflow.
    remove_tree(e.path / "deep")


def test_a_copy_that_lands_other_bytes_than_a_sound_entry_holds_is_a_failure_not_a_refusal(tmp_path,
                                                                                          monkeypatch):
    """The role reads the entry's file again only then, to tell a bad copy, which a retry makes
    anew, from a bad entry, which it would not."""
    e = _entry(tmp_path)
    real = shutil.copyfile

    def flipping(src, dst, **kw):
        real(src, dst, **kw)
        if Path(dst).name == "w0.bin":
            Path(dst).write_bytes(b"\x09" * 1000)
        return dst

    monkeypatch.setattr(cache_role.shutil, "copyfile", flipping)
    with pytest.raises(CopyFailed, match="the copy of shards/w0.bin differs from the entry's file"):
        copy_entry(e.path, entry_sha256(e), tmp_path / "copy")


def test_a_short_copy_of_a_sound_entry_is_a_failure_not_a_refusal(tmp_path, monkeypatch):
    e = _entry(tmp_path)
    real = shutil.copyfile

    def short(src, dst, **kw):
        real(src, dst, **kw)
        if Path(dst).name == "w0.bin":
            os.truncate(dst, 10)
        return dst

    monkeypatch.setattr(cache_role.shutil, "copyfile", short)
    with pytest.raises(CopyFailed, match="shards/w0.bin"):
        copy_entry(e.path, entry_sha256(e), tmp_path / "copy")


def test_a_manifest_copy_that_lands_other_bytes_than_a_sound_entry_holds_is_a_failure(tmp_path, monkeypatch):
    """The entry's manifest follows the rule of every file: its store file is read again only on a
    mismatch, and a sound one makes the mismatch the copy's."""
    e = _entry(tmp_path)
    real = shutil.copyfile

    def flipping(src, dst, **kw):
        real(src, dst, **kw)
        if Path(dst).name == ENTRY_FILE:
            Path(dst).write_bytes(Path(dst).read_bytes() + b" ")
        return dst

    monkeypatch.setattr(cache_role.shutil, "copyfile", flipping)
    with recording_reads() as reads, pytest.raises(CopyFailed, match="the copy of ENTRY.json differs from "
                                                                      "the entry's file"):
        copy_entry(e.path, entry_sha256(e), tmp_path / "copy")
    # The store's manifest was read twice: once to copy it, once to tell the copy was the bad one.
    assert reads.count(str(e.path / ENTRY_FILE)) == 2


@pytest.mark.parametrize("flip_only", [None, ENTRY_FILE], ids=["data files", "manifest"])
def test_a_failed_copy_ends_the_role_with_its_own_status_and_no_refusal_line(tmp_path, monkeypatch,
                                                                              flip_only):
    if flip_only is not None:
        monkeypatch.setenv("FLIP_ONLY", flip_only)
    e = _entry(tmp_path)
    dest, report = tmp_path / "copy", tmp_path / "digests.json"
    done = _role("copy", "--entry", str(e.path), "--expect", entry_sha256(e), "--dest", str(dest),
                 "--digests-out", str(report), script=FLIPPING)
    assert done.returncode == COPY_FAILED_EXIT, done.stderr
    assert "refused:" not in done.stderr and not report.exists()
    assert done.stderr.strip().splitlines()[-1].startswith("copy failed: the copy of ")


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file whatever its mode")
@pytest.mark.parametrize("flip_only", [None, ENTRY_FILE], ids=["data files", "manifest"])
def test_a_failed_reread_of_the_source_is_neither_a_refusal_nor_a_failed_copy(tmp_path, monkeypatch,
                                                                               flip_only):
    """The second read, which tells a bad copy from a bad entry, fails: the role knows neither,
    so it ends with an ordinary failure, which the stage retries, never with a refusal."""
    monkeypatch.setenv("HIDE_SOURCE", "1")
    if flip_only is not None:
        monkeypatch.setenv("FLIP_ONLY", flip_only)
    e = _entry(tmp_path)
    dest, report = tmp_path / "copy", tmp_path / "digests.json"
    try:
        done = _role("copy", "--entry", str(e.path), "--expect", entry_sha256(e), "--dest", str(dest),
                     "--digests-out", str(report), script=FLIPPING)
    finally:
        for p in e.path.rglob("*"):
            p.chmod(0o755 if p.is_dir() else 0o644)
    assert done.returncode not in (0, DIGEST_MISMATCH_EXIT, COPY_FAILED_EXIT), done.stderr
    assert "refused:" not in done.stderr and "copy failed:" not in done.stderr
    assert "PermissionError" in done.stderr
    assert not report.exists()


def test_a_copy_never_writes_into_an_existing_directory(tmp_path):
    e = _entry(tmp_path)
    (tmp_path / "copy").mkdir()
    with pytest.raises(FileExistsError):
        copy_entry(e.path, entry_sha256(e), tmp_path / "copy")


@pytest.mark.parametrize("mode", ["copy", "verify"])
def test_the_report_never_lies_inside_the_copy(tmp_path, mode):
    e = _entry(tmp_path)
    dest = tmp_path / "copy"
    if mode == "verify":
        copy_entry(e.path, entry_sha256(e), dest)
    where = ["--entry", str(e.path), "--dest", str(dest)] if mode == "copy" else ["--copy", str(dest)]
    done = _role(mode, *where, "--expect", entry_sha256(e), "--digests-out", str(dest / "digests.json"))
    assert done.returncode not in (0, DIGEST_MISMATCH_EXIT) and "lies inside" in done.stderr


def test_a_sound_copy_is_verified_and_its_digests_reported(tmp_path):
    e = _entry(tmp_path)
    copy = tmp_path / "copy"
    copy_entry(e.path, entry_sha256(e), copy)
    (copy / VERIFIED_FILE).write_text("{}")
    report = tmp_path / "digests.json"
    done = _role("verify", "--copy", str(copy), "--expect", entry_sha256(e), "--digests-out", str(report))
    assert done.returncode == 0, done.stderr
    assert read_digests(report) == _digests(FILES, e)


@pytest.mark.parametrize("change, says", [
    (lambda c: (c / "shards" / "w0.bin").write_bytes(b"\x02" * 1000), "changed shards/w0.bin"),
    (lambda c: (c / "shards" / "w0.bin").write_bytes(b"short"), "changed shards/w0.bin"),
    (lambda c: (c / "extra.bin").write_bytes(b"x"), "unexpected extra.bin"),
    (lambda c: (c / "config.json").unlink(), "missing config.json"),
    (lambda c: (c / ENTRY_FILE).write_text("{}"), "its ENTRY.json has sha256"),
    (lambda c: (c / ENTRY_FILE).unlink(), "its ENTRY.json is not a regular file"),
    (lambda c: (c / "link").symlink_to(c / "config.json"), "link is a symlink"),
], ids=["changed", "resized", "added", "removed", "manifest", "no manifest", "link"])
def test_verify_refuses_a_copy_that_differs_from_the_manifest(tmp_path, change, says):
    e = _entry(tmp_path)
    copy = tmp_path / "copy"
    copy_entry(e.path, entry_sha256(e), copy)
    change(copy)
    report = tmp_path / "digests.json"
    done = _role("verify", "--copy", str(copy), "--expect", entry_sha256(e), "--digests-out", str(report))
    assert done.returncode == DIGEST_MISMATCH_EXIT, done.stderr
    last = done.stderr.strip().splitlines()[-1]
    assert last.startswith("refused: ") and says in last and not report.exists()


def test_verify_reads_each_file_of_the_copy_once(tmp_path):
    e = _entry(tmp_path)
    copy = tmp_path / "copy"
    copy_entry(e.path, entry_sha256(e), copy)
    with recording_reads() as reads:
        verify_copy(copy, entry_sha256(e))
    assert sorted(reads) == sorted(str(copy / p) for p in [ENTRY_FILE, *FILES])


def test_verify_reads_only_the_copy(tmp_path):
    e = _entry(tmp_path)
    copy = tmp_path / "copy"
    copy_entry(e.path, entry_sha256(e), copy)
    shutil.rmtree(e.path)
    assert verify_copy(copy, entry_sha256(e)) == _digests(FILES, e)


# ---- fill: a node-local tree into an entry's staging -------------------------------------

def _tree(root: Path, files: dict[str, bytes] = FILES) -> Path:
    for rel, data in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return root


def test_a_fill_copies_every_file_of_the_tree_and_reports_each_copys_digest(tmp_path):
    source, dest, report = _tree(tmp_path / "cache"), tmp_path / "staging", tmp_path / "digests.json"
    dest.mkdir()
    done = _role("fill", "--source", str(source), "--dest", str(dest), "--digests-out", str(report))
    assert done.returncode == 0, done.stderr
    assert read_digests(report) == {p: sha256_bytes(b) for p, b in FILES.items()}
    assert sorted(str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()) == sorted(FILES)
    assert all((dest / p).read_bytes() == b for p, b in FILES.items())


@pytest.mark.parametrize("plant, says", [
    (lambda s: (s / "link").symlink_to(s / "config.json"), "link is a symlink"),
    (lambda s: os.link(s / "config.json", s / "hard"), "has 2 hard links"),
    (lambda s: (s / "shards" / ENTRY_FILE).write_text("{}"), f"shards/{ENTRY_FILE} uses a reserved name"),
], ids=["symlink", "hard link", "reserved name"])
def test_a_fill_refuses_a_tree_the_cache_would_not_publish_before_copying_anything(tmp_path, plant, says):
    source, dest, report = _tree(tmp_path / "cache"), tmp_path / "staging", tmp_path / "digests.json"
    plant(source)
    dest.mkdir()
    done = _role("fill", "--source", str(source), "--dest", str(dest), "--digests-out", str(report))
    assert done.returncode == DIGEST_MISMATCH_EXIT, done.stderr
    last = done.stderr.strip().splitlines()[-1]
    assert last.startswith("refused:") and says in last
    assert list(dest.iterdir()) == [] and not report.exists()


def test_a_fill_never_writes_into_a_directory_that_holds_anything(tmp_path):
    source, dest = _tree(tmp_path / "cache"), tmp_path / "staging"
    dest.mkdir()
    (dest / "left").write_text("x")
    with pytest.raises(FileExistsError):
        cache_role.fill_entry(source, dest)


# ---- file times: what a fill keeps and a copy restores ------------------------------------

#: A JIT's build tree: each output newer than its source, while a copy in path order would write
#: the source last. Each file's bytes and its mtime in ns, with digits below the microsecond.
BUILT = {"a/kernel.so": (b"shared object", 1_700_000_300_123_456_789),
         "a/kernel.o": (b"object", 1_700_000_200_123_456_789),
         "z/kernel.cu": (b"source", 1_700_000_100_123_456_789)}
#: Each file of ``BUILT`` with its mtime.
BUILT_MTIMES = {p: t for p, (_, t) in BUILT.items()}


def _built(root: Path) -> Path:
    for rel, (data, mtime_ns) in BUILT.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
        os.utime(root / rel, ns=(mtime_ns, mtime_ns))
    return root


def _mtimes(root: Path, paths) -> dict[str, int]:
    return {p: os.lstat(root / p).st_mtime_ns for p in paths}


def _filled(tmp_path: Path, source: Path) -> CacheEntry:
    """The entry the fill role publishes from *source*, as ``StageContext.publish`` fills one."""
    reported: dict[str, str] = {}
    return ContentCache(PosixStore(tmp_path / "store")).populate(
        "engine-cache", "k", lambda staging: reported.update(cache_role.fill_entry(source, staging)),
        holder=Holder.here("test"), digests=lambda: reported).entry


def test_a_fill_copies_each_file_with_the_mtime_the_tree_has(tmp_path):
    source, dest, report = _built(tmp_path / "cache"), tmp_path / "staging", tmp_path / "digests.json"
    dest.mkdir()
    done = _role("fill", "--source", str(source), "--dest", str(dest), "--digests-out", str(report))
    assert done.returncode == 0, done.stderr
    assert _mtimes(dest, BUILT) == BUILT_MTIMES


def _off_by_one_ns(monkeypatch) -> None:
    """Make every time the role sets land one ns later, as on a filesystem that keeps another."""
    real = os.utime

    def off_by_one(path, *args, ns=None, **kw):
        if ns is not None:
            ns = (ns[0], ns[1] + 1)
        return real(path, *args, ns=ns, **kw)

    monkeypatch.setattr(cache_role.os, "utime", off_by_one)


def test_a_fill_whose_copy_did_not_keep_its_time_is_a_failed_copy(tmp_path, monkeypatch):
    source, dest, report = _built(tmp_path / "cache"), tmp_path / "staging", tmp_path / "digests.json"
    dest.mkdir()
    _off_by_one_ns(monkeypatch)
    with pytest.raises(CopyFailed, match="the copy of a/kernel.o kept mtime 1700000200123456790 ns, not the "
                                         "1700000200123456789 of the tree's file"):
        cache_role.fill_entry(source, dest)
    other = tmp_path / "other"
    other.mkdir()
    assert cache_role.main(["fill", "--source", str(source), "--dest", str(other),
                            "--digests-out", str(report)]) == COPY_FAILED_EXIT
    assert not report.exists()


def test_a_copy_gives_each_file_the_mtime_its_record_holds_not_the_stores_file(tmp_path):
    """The record is pinned by the entry's digest; the store's own file times are never copied."""
    e = _filled(tmp_path, _built(tmp_path / "cache"))
    assert {f.path: f.mtime_ns for f in e.manifest.files} == BUILT_MTIMES
    os.utime(e.path / "z" / "kernel.cu", ns=(1, 1))
    dest, report = tmp_path / "copy", tmp_path / "digests.json"
    done = _role("copy", "--entry", str(e.path), "--expect", entry_sha256(e), "--dest", str(dest),
                 "--digests-out", str(report))
    assert done.returncode == 0, done.stderr
    assert _mtimes(dest, BUILT) == BUILT_MTIMES


def test_a_copy_whose_mtime_did_not_land_is_a_failed_copy(tmp_path, monkeypatch):
    """A destination that keeps another time than the one set is the copy's fault, which a retry
    makes anew; the entry is sound."""
    e = _filled(tmp_path, _built(tmp_path / "cache"))
    _off_by_one_ns(monkeypatch)
    with pytest.raises(CopyFailed, match="a/kernel.o has mtime 1700000200123456790 ns, not the recorded "
                                         "1700000200123456789"):
        copy_entry(e.path, entry_sha256(e), tmp_path / "copy")


def test_verify_refuses_a_copy_whose_mtime_is_not_the_record(tmp_path):
    e = _filled(tmp_path, _built(tmp_path / "cache"))
    copy = tmp_path / "copy"
    copy_entry(e.path, entry_sha256(e), copy)
    os.utime(copy / "a" / "kernel.o", ns=(1, 1_700_000_200_123_456_788))
    report = tmp_path / "digests.json"
    done = _role("verify", "--copy", str(copy), "--expect", entry_sha256(e), "--digests-out", str(report))
    assert done.returncode == DIGEST_MISMATCH_EXIT, done.stderr
    last = done.stderr.strip().splitlines()[-1]
    assert last == "refused: a/kernel.o has mtime 1700000200123456788 ns, not the recorded 1700000200123456789"
    assert not report.exists()


@pytest.mark.skipif(shutil.which("ninja") is None, reason="ninja is not on PATH")
def test_a_ninja_build_filled_into_the_cache_and_copied_out_has_no_work_to_do(tmp_path):
    """A seeded engine cache's JIT builds by ninja, which decides by file times: a copy that
    left its sources newer than its outputs would build again on every load."""
    built = tmp_path / "build"
    built.mkdir()
    (built / "build.ninja").write_text("rule cat\n  command = cat $in > $out\n"
                                       "build a.out: cat z.src\nbuild b.out: cat a.out\n")
    (built / "z.src").write_text("kernel source")
    subprocess.run(["ninja", "-C", str(built)], check=True, capture_output=True, timeout=60)

    def dry_run(d: Path) -> str:
        return subprocess.run(["ninja", "-n", "-C", str(d)], check=True, capture_output=True, text=True,
                              timeout=60).stdout

    assert "no work to do" in dry_run(built)
    e = _filled(tmp_path, built)
    copy = ContentCache(PosixStore(tmp_path / "store")).materialize(
        e, tmp_path / "node", "job-1", copy=lambda d: copy_entry(e.path, entry_sha256(e), d),
        verify=lambda d: verify_copy(d, entry_sha256(e)))
    assert "no work to do" in dry_run(copy)
