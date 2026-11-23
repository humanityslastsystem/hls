from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.store import cache as cache_module
from hls.store import cache_role
from hls.store.cache import (COPY_BLOCK_BYTES, ENTRY_FILE, MAX_DEPTH, VERIFIED_FILE, CacheCorrupt, CacheEntry,
                             ContentCache, EntryManifest, FileRecord, WaitEnded, copies_root, copy_disk_bytes,
                             entry_sha256, hash_tree, read_digests, scan_dir, write_digests, write_file)
from hls.substrate.memory_watch import disk_usage
from hls.store.cache_role import Refused, copy_entry, verify_copy
from hls.store.digest import sha256_bytes
from hls.store.lock import GenerationLock, Holder, LockLost
from hls.store.posix import PosixStore
from hls.util import remove_tree
from tests.store.entries import SPAWNED, UNRESOLVED, entry_reads, launches, populate, recording_reads

HOLDER = Holder(host="h", pid=1, job_id="123", purpose="cache")


def _fill_two_files(d: Path) -> None:
    (d / "config.json").write_text('{"a": 1}')
    (d / "shards").mkdir()
    (d / "shards" / "w0.bin").write_bytes(b"\x00" * 1000)


def _staging_root(tmp_path: Path, kind: str = "weights", key: str = "k") -> Path:
    return tmp_path / ".staging" / "cache" / kind / key


def _publish_by_hand(root: Path, kind: str, key: str) -> None:
    """Publish an entry the way another filler would, without this cache's populate."""
    d = root / "cache" / kind / key
    d.mkdir(parents=True)
    _fill_two_files(d)
    files = scan_dir(d, digests=hash_tree(d))
    m = EntryManifest(kind=kind, key=key, files=files, total_size=sum(f.size for f in files), created_at=1.0)
    (d / ENTRY_FILE).write_text(m.model_dump_json())


@contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    """Fail the block with TimeoutError instead of hanging, e.g. on a read of a FIFO."""

    def expired(signum, frame):
        raise TimeoutError(f"still running after {seconds} s")

    old = signal.signal(signal.SIGALRM, expired)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _nest(root: Path, levels: int) -> None:
    """A chain of *levels* nested directories under *root* with one file at the bottom. Made one
    level at a time, so building it does not recurse."""
    p = root / "deep"
    for _ in range(levels):
        p.mkdir()
        p = p / "a"
    (p.parent / "f").write_bytes(b"x")


def _no_fill(staging: Path) -> None:
    """A fill that must not run: the entry is another filler's, or corrupt."""
    pytest.fail("filled an entry that was not this call's to fill")


def _replace_by_symlink(path: Path) -> None:
    """Move the file out of its directory and leave a symlink to it."""
    kept = path.parent.parent / f"kept-{path.name}"
    path.rename(kept)
    path.symlink_to(kept)


def test_miss_fills_once_then_hits(tmp_path):
    cache = ContentCache(PosixStore(tmp_path))
    calls = []

    def fill(d: Path) -> None:
        calls.append(d)
        _fill_two_files(d)

    first = populate(cache, "weights", "qwen@abc", fill, holder=HOLDER)
    again = populate(cache, "weights", "qwen@abc", fill, holder=HOLDER)
    assert (first.published, again.published) == (True, False)
    e1, e2 = first.entry, again.entry
    assert len(calls) == 1
    assert calls[0].parent == _staging_root(tmp_path, key="qwen@abc")
    assert e1.path == e2.path == tmp_path / "cache" / "weights" / "qwen@abc"
    assert [f.path for f in e1.manifest.files] == ["config.json", "shards/w0.bin"]
    assert e1.manifest.total_size == 1008
    assert (e1.path / ENTRY_FILE).exists()
    assert entry_sha256(e1) == sha256_bytes((e1.path / ENTRY_FILE).read_bytes())


def test_failed_fill_leaves_nothing_and_frees_the_lock(tmp_path):
    store = PosixStore(tmp_path)
    cache = ContentCache(store)

    def bad(d: Path) -> None:
        (d / "partial").write_bytes(b"x")
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError):
        populate(cache, "weights", "k", bad, holder=HOLDER).entry
    assert cache.lookup("weights", "k") is None
    assert list(_staging_root(tmp_path).iterdir()) == []
    assert populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry.manifest.total_size == 1008


@pytest.mark.parametrize("plant, what", [
    (lambda d: (d / "link").symlink_to(d / "config.json"), "symlink"),
    (lambda d: (d / "shards" / "dirlink").symlink_to("/etc"), "symlink"),
    (lambda d: os.mkfifo(d / "pipe"), "not a regular file"),
    (lambda d: (d / ENTRY_FILE).write_text("{}"), "reserved"),
    (lambda d: (d / "shards" / VERIFIED_FILE).write_text("x"), "reserved"),
    (lambda d: (d / "shards" / ENTRY_FILE).mkdir(), "reserved"),
    (lambda d: os.link(d / "config.json", d / "twin"), "2 hard links"),
    (lambda d: _nest(d, MAX_DEPTH + 1), "nested deeper than"),
    (lambda d: _nest(d, 1100), "nested deeper than"),
])
def test_fill_output_that_is_not_plain_files_is_refused(tmp_path, plant, what):
    cache = ContentCache(PosixStore(tmp_path))

    def fill(d: Path) -> None:
        _fill_two_files(d)
        plant(d)

    # The deadline turns a broken special-file guard, which would block reading the FIFO, into a failure.
    with _deadline(20), pytest.raises(CacheCorrupt, match=what):
        populate(cache, "weights", "k", fill, holder=HOLDER).entry
    assert cache.lookup("weights", "k") is None
    assert list(_staging_root(tmp_path).iterdir()) == []


def test_a_tree_exactly_at_the_depth_limit_is_accepted(tmp_path):
    cache = ContentCache(PosixStore(tmp_path))
    e = populate(cache, "weights", "k", lambda d: _nest(d, MAX_DEPTH), holder=HOLDER).entry
    assert len(e.manifest.files) == 1


def test_the_check_sees_the_fill_output_before_anything_is_published(tmp_path):
    cache = ContentCache(PosixStore(tmp_path))
    seen = []

    def check(files: list[FileRecord]) -> None:
        seen.append([(f.path, f.size) for f in files])
        assert cache.lookup("weights", "k") is None
        raise ValueError("source digest differs")

    with pytest.raises(ValueError, match="source digest differs"):
        populate(cache, "weights", "k", _fill_two_files, holder=HOLDER, check=check).entry
    assert seen == [[("config.json", 8), ("shards/w0.bin", 1000)]]
    assert cache.lookup("weights", "k") is None
    assert list(_staging_root(tmp_path).iterdir()) == []
    passed = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER, check=lambda f: None).entry
    assert passed.manifest.total_size == 1008


def _reported_fill(d: Path) -> dict[str, str]:
    """Write the two files as a role would, synced, and return the digests it would report."""
    _fill_two_files(d)
    for path in (d / "config.json", d / "shards" / "w0.bin"):
        fd = os.open(path, os.O_WRONLY)
        os.fsync(fd)
        os.close(fd)
    return {"config.json": sha256_bytes(b'{"a": 1}'), "shards/w0.bin": sha256_bytes(b"\x00" * 1000)}


def test_the_cache_syncs_every_directory_and_its_manifest_before_publish_never_a_file_the_fill_wrote(
        tmp_path, monkeypatch):
    store = PosixStore(tmp_path)
    cache = ContentCache(store)
    events: list[tuple[str, str]] = []
    real_fsync, real_publish = os.fsync, store.publish_dir
    reported: dict[str, str] = {}
    filled = threading.Event()

    def fsync(fd: int) -> None:
        if filled.is_set():
            events.append(("fsync", os.readlink(f"/proc/self/fd/{fd}")))
        real_fsync(fd)

    def publish(src: Path, key: str) -> None:
        events.append(("publish", str(src)))
        real_publish(src, key)

    def fill(d: Path) -> None:
        reported.update(_reported_fill(d))
        filled.set()

    monkeypatch.setattr(cache_module.os, "fsync", fsync)
    monkeypatch.setattr(store, "publish_dir", publish)
    with recording_reads() as reads:
        e = cache.populate("weights", "k", fill, holder=HOLDER, digests=lambda: reported).entry
    at = [kind for kind, _ in events].index("publish")
    staging = Path(events[at][1])
    synced = {Path(p) for kind, p in events[:at] if kind == "fsync" and Path(p).is_relative_to(staging)}
    assert synced == {staging, staging / "shards", staging / ENTRY_FILE}
    assert entry_reads(reads, [tmp_path / ".staging", tmp_path / "cache"]) == []
    # The rename itself is made durable by syncing the directory it landed in.
    assert ("fsync", str(e.path.parent)) in events[at:]
    # This publish created cache/weights and cache, so the names of both are made durable too.
    assert ("fsync", str(tmp_path / "cache")) in events[at:]
    assert ("fsync", str(tmp_path)) in events[at:]


def test_the_cache_records_the_reported_digests_without_opening_any_file_the_fill_wrote(tmp_path):
    """The roles that wrote the files hashed and synced them; the cache records what they
    reported and reads none of them, so a wrong report is recorded as it is (the copy role that
    reads the files later refuses them)."""
    cache = ContentCache(PosixStore(tmp_path))
    reported = {"config.json": sha256_bytes(b"what the role hashed"), "shards/w0.bin": "b" * 64}
    with recording_reads() as reads:
        entry = cache.populate("weights", "k", _fill_two_files, holder=HOLDER, digests=lambda: reported).entry
    assert {f.path: (f.size, f.sha256) for f in entry.manifest.files} == {
        "config.json": (8, reported["config.json"]), "shards/w0.bin": (1000, "b" * 64)}
    assert entry_reads(reads, [tmp_path / "cache", tmp_path / ".staging"]) == []


def test_a_file_the_fill_wrote_with_no_reported_digest_publishes_nothing(tmp_path):
    """The cache hashes nothing: a file whose writer reported no digest is refused, never read."""
    cache = ContentCache(PosixStore(tmp_path))
    with recording_reads() as reads, pytest.raises(CacheCorrupt, match="no digest was reported for shards/"):
        cache.populate("weights", "k", _fill_two_files, holder=HOLDER,
                       digests=lambda: {"config.json": sha256_bytes(b'{"a": 1}')})
    assert entry_reads(reads, [tmp_path / "cache", tmp_path / ".staging"]) == []
    assert cache.lookup("weights", "k") is None and list(_staging_root(tmp_path).iterdir()) == []


@pytest.mark.parametrize("extra, match", [
    ({"absent.bin": "a" * 64}, "reported a digest for absent.bin"),
    ({"shards": "a" * 64}, "reported a digest for shards"),
    ({"config.json": "not hex"}, "not a bare hex"),
])
def test_a_reported_digest_of_no_file_the_fill_wrote_or_of_no_digest_publishes_nothing(tmp_path, extra,
                                                                                        match):
    cache = ContentCache(PosixStore(tmp_path))
    staged: list[Path] = []

    def fill(d: Path) -> None:
        staged.append(d)
        _fill_two_files(d)

    with pytest.raises(CacheCorrupt, match=match):
        cache.populate("weights", "k", fill, holder=HOLDER,
                       digests=lambda: {**hash_tree(staged[-1]), **extra})
    assert cache.lookup("weights", "k") is None and list(_staging_root(tmp_path).iterdir()) == []


def test_hash_tree_hashes_and_syncs_every_regular_file_and_opens_nothing_else(tmp_path, monkeypatch):
    """What a fill role reports: the bytes on disk after its write, each file synced. A link is
    not followed and a FIFO not opened; the cache's scan refuses both."""
    _fill_two_files(tmp_path)
    (tmp_path / "link").symlink_to(tmp_path / "config.json")
    os.mkfifo(tmp_path / "pipe")
    synced: list[str] = []
    real = os.fsync
    monkeypatch.setattr(cache_module.os, "fsync",
                        lambda fd: (synced.append(os.readlink(f"/proc/self/fd/{fd}")), real(fd))[1])
    with _deadline(20):
        got = hash_tree(tmp_path)
    assert got == {"config.json": sha256_bytes(b'{"a": 1}'), "shards/w0.bin": sha256_bytes(b"\x00" * 1000)}
    assert sorted(synced) == [str(tmp_path / "config.json"), str(tmp_path / "shards" / "w0.bin")]


def test_write_file_writes_syncs_and_reports_the_bytes_it_wrote(tmp_path, monkeypatch):
    synced: list[str] = []
    real = os.fsync
    monkeypatch.setattr(cache_module.os, "fsync",
                        lambda fd: (synced.append(os.readlink(f"/proc/self/fd/{fd}")), real(fd))[1])
    with recording_reads() as reads:
        digest = write_file(tmp_path / "meta.json", b"{}")
    assert digest == sha256_bytes(b"{}") and (tmp_path / "meta.json").read_bytes() == b"{}"
    assert synced == [str(tmp_path / "meta.json")] and entry_reads(reads, [tmp_path]) == []
    with pytest.raises(FileExistsError):
        write_file(tmp_path / "meta.json", b"again")


def test_a_roles_digest_report_round_trips_and_a_bad_one_reads_as_none(tmp_path):
    report = tmp_path / "digests.json"
    write_digests(report, {"a/b.bin": "a" * 64, ENTRY_FILE: "b" * 64})
    assert read_digests(report) == {"a/b.bin": "a" * 64, ENTRY_FILE: "b" * 64}
    for bad in ("", "not json", "[]", '{"a": 1}', '{"a": "not hex"}'):
        report.write_text(bad)
        assert read_digests(report) is None
    assert read_digests(tmp_path / "absent.json") is None


def test_a_dead_fills_staging_is_swept_by_the_next_holder(tmp_path):
    left = _staging_root(tmp_path) / "0123abcd"
    (left / "shards").mkdir(parents=True)
    (left / "shards" / "w0.bin").write_bytes(b"half")
    cache = ContentCache(PosixStore(tmp_path))
    populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    assert list(_staging_root(tmp_path).iterdir()) == []
    # Another key's staging is not this holder's to sweep.
    other = _staging_root(tmp_path, key="other") / "fedc"
    other.mkdir(parents=True)
    populate(cache, "weights", "k2", _fill_two_files, holder=HOLDER).entry
    assert other.exists()


def test_a_dead_fills_staging_deeper_than_the_interpreter_recurses_is_swept(tmp_path):
    left = _staging_root(tmp_path) / "0123abcd"
    left.mkdir(parents=True)
    _nest(left, sys.getrecursionlimit() + 100)
    cache = ContentCache(PosixStore(tmp_path))
    assert populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).published
    assert list(_staging_root(tmp_path).iterdir()) == []


@pytest.mark.parametrize("stale_name, outcome", [("0stale", "refused"), ("zstale", "landed")])
def test_a_stale_publish_during_the_sweep_never_leaves_a_partial_entry(tmp_path, monkeypatch, stale_name, outcome):
    """A holder that lost its lock may still rename its staging tree into place while the next
    holder sweeps it. The sweep has moved the tree aside first, so the rename is refused, or it
    landed before the sweep reached the tree and the entry is whole: never half-deleted."""
    store = PosixStore(tmp_path)
    cache = ContentCache(store)
    root = _staging_root(tmp_path)
    stale = root / stale_name
    stale.mkdir(parents=True)
    for i in range(8):
        (stale / f"w{i}.bin").write_bytes(b"x" * 10)
    files = scan_dir(stale, digests=hash_tree(stale))
    manifest = EntryManifest(kind="weights", key="k", files=files, total_size=80, created_at=2.0)
    (stale / ENTRY_FILE).write_text(manifest.model_dump_json())
    # A stray plain file beside it, which sorts between the two stale names.
    (root / "m-stray.txt").write_bytes(b"stray")
    (tmp_path / "cache" / "weights").mkdir(parents=True)
    happened: list[str] = []
    real_unlink = os.unlink

    def unlink(name, *args, dir_fd=None, **kwargs):
        # The first removal inside the staging root: the stale holder's rename lands now.
        if not happened and (dir_fd is not None or str(name).startswith(str(root))):
            try:
                os.rename(stale, tmp_path / "cache" / "weights" / "k")
                happened.append("landed")
            except FileNotFoundError:
                happened.append("refused")
        return real_unlink(name, *args, dir_fd=dir_fd, **kwargs)

    monkeypatch.setattr(cache_module.os, "unlink", unlink)
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    assert happened == [outcome]
    monkeypatch.undo()
    # The entry is whole: a copy of it checks out file by file.
    copy_entry(e.path, entry_sha256(e), tmp_path / "copy")
    assert list(root.iterdir()) == []


def test_a_waiter_never_sweeps_the_holders_staging(tmp_path):
    store = PosixStore(tmp_path)
    started, finish = threading.Event(), threading.Event()
    results: dict[str, object] = {}

    def slow(d: Path) -> None:
        (d / "config.json").write_text('{"a": 1}')
        started.set()
        assert finish.wait(5)
        (d / "shards").mkdir()
        (d / "shards" / "w0.bin").write_bytes(b"\x00" * 1000)

    def holder() -> None:
        try:
            results["holder"] = populate(ContentCache(store), "weights", "k", slow, holder=HOLDER).entry
        except Exception as e:  # noqa: BLE001
            results["holder"] = e

    def waiter() -> None:
        results["waiter"] = populate(ContentCache(store, sleep=lambda s: time.sleep(0.02)), "weights", "k",
                                     _no_fill, holder=HOLDER, poll_sec=0.02).entry

    t1 = threading.Thread(target=holder)
    t1.start()
    assert started.wait(5)
    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.3)
    finish.set()
    t1.join(10)
    t2.join(10)
    assert not isinstance(results["holder"], Exception), results["holder"]
    assert results["waiter"].manifest.total_size == 1008


def test_a_waiter_keeps_waiting_while_the_holder_renews(tmp_path):
    store = PosixStore(tmp_path)
    now = [1000.0]
    held = GenerationLock(store, "cache-locks/weights/k", ttl_sec=120, clock=lambda: now[0]).try_acquire(
        Holder(host="other", pid=7, job_id="7", purpose="other filler"))
    sleeps = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        now[0] += sec
        if len(sleeps) < 2000:
            held.renew()
        else:
            _publish_by_hand(tmp_path, "weights", "k")

    cache = ContentCache(store, clock=lambda: now[0], sleep=sleep)
    e = populate(cache, "weights", "k", _no_fill, holder=HOLDER, poll_sec=5.0).entry
    # Two thousand polls is close to three hours of waiting, longer than any fixed limit
    # a waiter could have had; the holder was renewing throughout.
    assert len(sleeps) == 2000 and now[0] - 1000.0 == 10000.0
    assert e.manifest.total_size == 1008


def test_a_stale_holder_is_taken_over(tmp_path):
    store = PosixStore(tmp_path)
    now = [1000.0]
    GenerationLock(store, "cache-locks/weights/k", ttl_sec=120, clock=lambda: now[0]).try_acquire(
        Holder(host="other", pid=7, job_id="7", purpose="silent filler"))
    sleeps = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        now[0] += sec

    cache = ContentCache(store, clock=lambda: now[0], sleep=sleep)
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER, poll_sec=30.0).entry
    # Taken over once the heartbeat was older than the TTL: after five polls of 30 s.
    assert sleeps == [30.0] * 5
    assert e.manifest.total_size == 1008


OTHER = Holder(host="other", pid=7, job_id="7", purpose="other filler")


def _renewing_holder(store: PosixStore, now: list[float]):
    """Hold the lock of weights/k as OTHER; return the list of sleeps and a sleep function that
    records each, advances the clock *now* and renews the lock."""
    held = GenerationLock(store, "cache-locks/weights/k", ttl_sec=120, clock=lambda: now[0]).try_acquire(OTHER)
    sleeps: list[float] = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        # A waiter that ignored its bound would otherwise wait here for ever.
        assert len(sleeps) < 100, "still waiting after 100 pauses"
        now[0] += sec
        held.renew()

    return sleeps, sleep


def test_a_waiter_gives_up_at_its_deadline_and_never_sleeps_past_it(tmp_path):
    store = PosixStore(tmp_path)
    now = [1000.0]
    sleeps, sleep = _renewing_holder(store, now)
    cache = ContentCache(store, clock=lambda: now[0], sleep=sleep)
    with pytest.raises(WaitEnded, match="weights/k: still filled by other filler on other pid 7 at the deadline") as info:
        populate(cache, "weights", "k", _no_fill, holder=HOLDER, poll_sec=30.0, deadline=1070.0)
    assert sleeps == [30.0, 30.0, 10.0] and now[0] == 1070.0
    # The message names the holder; the error carries nothing else, which no caller would read.
    assert vars(info.value) == {}


def test_a_waiter_stops_waiting_when_told_to_stop(tmp_path):
    store = PosixStore(tmp_path)
    now = [1000.0]
    sleeps, sleep = _renewing_holder(store, now)
    cache = ContentCache(store, clock=lambda: now[0], sleep=sleep)
    with pytest.raises(WaitEnded, match="weights/k: stopped while other filler on other pid 7 fills it"):
        populate(cache, "weights", "k", _no_fill, holder=HOLDER, poll_sec=30.0, deadline=5000.0,
                 stopping=lambda: len(sleeps) == 2)
    assert sleeps == [30.0, 30.0]


def test_a_waiter_names_each_holder_it_waits_for_once(tmp_path):
    store = PosixStore(tmp_path)
    now = [1000.0]
    lock = GenerationLock(store, "cache-locks/weights/k", ttl_sec=120, clock=lambda: now[0])
    held = [lock.try_acquire(OTHER)]
    second = Holder(host="third", pid=9, job_id="9", purpose="third filler")
    seen: list[Holder] = []

    def sleep(sec: float) -> None:
        now[0] += sec
        if len(seen) == 1 and now[0] >= 1100.0:
            held[0].release()
            held[0] = lock.try_acquire(second)
        elif now[0] >= 1200.0:
            _publish_by_hand(tmp_path, "weights", "k")
        else:
            held[0].renew()

    cache = ContentCache(store, clock=lambda: now[0], sleep=sleep)
    got = populate(cache, "weights", "k", _no_fill, holder=HOLDER, poll_sec=10.0, on_wait=seen.append)
    assert seen == [OTHER, second] and not got.published


def _populate(root: str, q) -> None:
    calls = []

    def fill(d: Path) -> None:
        calls.append(1)
        time.sleep(0.3)
        _fill_two_files(d)

    cache = ContentCache(PosixStore(Path(root)), lock_ttl_sec=5.0)
    holder = Holder(host="h", pid=os.getpid(), job_id=None, purpose="race")
    populate(cache, "weights", "race", fill, holder=holder, poll_sec=0.05).entry
    q.put(len(calls))


def test_concurrent_populate_fills_once(tmp_path):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_populate, args=(str(tmp_path), q)) for _ in range(5)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert [p.exitcode for p in procs] == [0] * 5
    assert sorted(q.get(timeout=5) for _ in procs) == [0, 0, 0, 0, 1]


def test_fill_longer_than_ttl_keeps_the_lock(tmp_path):
    cache = ContentCache(PosixStore(tmp_path), lock_ttl_sec=0.6)

    def slow(d: Path) -> None:
        time.sleep(1.2)
        _fill_two_files(d)

    assert populate(cache, "weights", "slow", slow, holder=HOLDER).entry.manifest.total_size == 1008


def test_dead_holder_lock_is_taken_over(tmp_path):
    store = PosixStore(tmp_path)
    GenerationLock(store, "cache-locks/weights/k", ttl_sec=3600).try_acquire(
        Holder(host="h", pid=1, job_id="999", purpose="dead"))
    cache = ContentCache(store, sleep=lambda s: pytest.fail("waited for a dead holder"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER, alive=lambda h: False).entry
    assert e.manifest.total_size == 1008


def test_a_lost_lock_does_not_publish(tmp_path):
    store = PosixStore(tmp_path)
    cache = ContentCache(store)

    def fill_then_lose(d: Path) -> None:
        _fill_two_files(d)
        # another filler takes the lock over while this one is still filling
        GenerationLock(store, "cache-locks/weights/lost", ttl_sec=120).try_acquire(
            Holder(host="h", pid=2, job_id="2", purpose="other"), alive=lambda h: False)

    with pytest.raises(LockLost):
        populate(cache, "weights", "lost", fill_then_lose, holder=HOLDER).entry
    assert cache.lookup("weights", "lost") is None
    assert list(_staging_root(tmp_path, key="lost").iterdir()) == []


def test_a_holder_that_lost_its_lock_reports_the_loss_not_the_error_it_hit(tmp_path):
    store = PosixStore(tmp_path)
    cache = ContentCache(store)

    def fill_lose_then_fail(d: Path) -> None:
        _fill_two_files(d)
        # The next holder took the lock over and swept this staging tree away.
        GenerationLock(store, "cache-locks/weights/k", ttl_sec=120).try_acquire(
            Holder(host="h", pid=2, job_id="2", purpose="other"), alive=lambda h: False)
        raise FileNotFoundError("staging was swept")

    with pytest.raises(LockLost) as raised:
        populate(cache, "weights", "k", fill_lose_then_fail, holder=HOLDER).entry
    assert isinstance(raised.value.__cause__, FileNotFoundError)
    assert cache.lookup("weights", "k") is None


def test_a_lock_lost_after_the_publish_still_returns_the_entry(tmp_path, monkeypatch):
    store = PosixStore(tmp_path)
    cache = ContentCache(store)
    real_publish = store.publish_dir

    def publish_then_lose(src: Path, key: str) -> None:
        real_publish(src, key)
        GenerationLock(store, "cache-locks/weights/k", ttl_sec=120).try_acquire(
            Holder(host="h", pid=2, job_id="2", purpose="other"), alive=lambda h: False)

    monkeypatch.setattr(store, "publish_dir", publish_then_lose)
    got = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER)
    assert got.entry.manifest.total_size == 1008 and got.published


def test_another_fillers_identical_publish_is_returned_as_found(tmp_path, monkeypatch):
    store = PosixStore(tmp_path)
    cache = ContentCache(store)
    real_publish = store.publish_dir

    def beaten(src: Path, key: str) -> None:
        _publish_by_hand(tmp_path, "weights", "k")
        real_publish(src, key)

    monkeypatch.setattr(store, "publish_dir", beaten)
    got = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER)
    assert got.entry.manifest.created_at == 1.0 and not got.published
    assert list(_staging_root(tmp_path).iterdir()) == []


@pytest.mark.parametrize("damage, what", [
    (lambda d: (d / "config.json").write_text("x"), "no ENTRY.json"),
    (lambda d: (d / ENTRY_FILE).write_text("not json"), "not a valid"),
])
def test_a_published_entry_without_a_valid_manifest_is_corrupt(tmp_path, damage, what):
    d = tmp_path / "cache" / "weights" / "k"
    d.mkdir(parents=True)
    damage(d)
    cache = ContentCache(PosixStore(tmp_path))
    with pytest.raises(CacheCorrupt, match=what):
        cache.lookup("weights", "k")
    with pytest.raises(CacheCorrupt, match=what):
        populate(cache, "weights", "k", _no_fill, holder=HOLDER).entry


def test_an_entry_published_under_another_name_is_corrupt(tmp_path):
    _publish_by_hand(tmp_path, "weights", "k")
    (tmp_path / "cache" / "weights" / "k").rename(tmp_path / "cache" / "weights" / "j")
    with pytest.raises(CacheCorrupt, match="weights/k"):
        ContentCache(PosixStore(tmp_path)).lookup("weights", "j")


def test_a_symlinked_manifest_is_corrupt(tmp_path):
    _publish_by_hand(tmp_path, "weights", "k")
    _replace_by_symlink(tmp_path / "cache" / "weights" / "k" / ENTRY_FILE)
    with pytest.raises(CacheCorrupt, match="symlink"):
        ContentCache(PosixStore(tmp_path)).lookup("weights", "k")


def test_a_symlink_at_the_entry_place_is_corrupt_even_when_it_leads_to_a_valid_entry(tmp_path):
    _publish_by_hand(tmp_path, "weights", "k")
    place = tmp_path / "cache" / "weights" / "k"
    elsewhere = tmp_path / "elsewhere"
    place.rename(elsewhere)
    place.symlink_to(elsewhere)
    cache = ContentCache(PosixStore(tmp_path))
    with pytest.raises(CacheCorrupt, match="symlink"):
        cache.lookup("weights", "k")
    with pytest.raises(CacheCorrupt, match="symlink"):
        populate(cache, "weights", "k", _no_fill, holder=HOLDER).entry


def test_a_file_at_the_entry_place_is_corrupt(tmp_path):
    place = tmp_path / "cache" / "weights" / "k"
    place.parent.mkdir(parents=True)
    place.write_text("x")
    with pytest.raises(CacheCorrupt, match="not a directory"):
        ContentCache(PosixStore(tmp_path)).lookup("weights", "k")


def test_the_entry_digest_is_that_of_the_manifest_bytes_lookup_read(tmp_path):
    cache = ContentCache(PosixStore(tmp_path))
    populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    e = cache.lookup("weights", "k")
    read = sha256_bytes((e.path / ENTRY_FILE).read_bytes())
    assert entry_sha256(e) == read
    # Rewriting the file later does not change what this entry was read as.
    (e.path / ENTRY_FILE).write_text("rewritten")
    assert entry_sha256(e) == read
    assert sha256_bytes((e.path / ENTRY_FILE).read_bytes()) != read


class _Roles:
    """The copy and verify roles of ``materialize``, run in this process (``cache_role``), which
    count their runs. A test of the runner's own reads runs them as processes instead."""

    def __init__(self, entry: CacheEntry) -> None:
        self.entry = entry
        #: How many copies and verifications ran, in order.
        self.runs: list[str] = []

    def copy(self, dest: Path) -> dict[str, str]:
        self.runs.append("copy")
        return copy_entry(self.entry.path, entry_sha256(self.entry), dest)

    def verify(self, copy: Path) -> dict[str, str]:
        self.runs.append("verify")
        return verify_copy(copy, entry_sha256(self.entry))


def _materialize(cache: ContentCache, e: CacheEntry, root: Path, job: str,
                 roles: _Roles | None = None) -> Path:
    roles = roles or _Roles(e)
    return cache.materialize(e, root, job, copy=roles.copy, verify=roles.verify)


def test_materialize_copies_verifies_and_reuses(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    local = tmp_path / "node"
    roles = _Roles(e)
    d1 = _materialize(cache, e, local, "job-42", roles)
    assert d1 == local / "job-42" / "weights" / "k" == copies_root(local, "job-42") / "weights" / "k"
    assert (d1 / "shards" / "w0.bin").read_bytes() == b"\x00" * 1000
    assert (d1 / ENTRY_FILE).read_bytes() == (e.path / ENTRY_FILE).read_bytes()
    # A recopy would make new files; the verified copy is reused as it is, with no role.
    inode = (d1 / "shards" / "w0.bin").stat().st_ino
    assert _materialize(cache, e, local, "job-42", roles) == d1
    assert (d1 / "shards" / "w0.bin").stat().st_ino == inode and roles.runs == ["copy"]


def test_a_jobs_node_local_copies_are_made_under_a_directory_only_its_owner_can_enter(tmp_path):
    """The copies of a job (weights, rows, tool bundles) lie under ``<root>/<job>``, made 0700 as
    the role directories are, whatever the modes of what the copy role writes below it."""
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    local = tmp_path / "node"
    local.mkdir()
    (local / "job-7").mkdir(mode=0o755)
    for job in ("job-42", "job-7"):
        _materialize(cache, e, local, job)
        assert stat.S_IMODE(os.lstat(copies_root(local, job)).st_mode) == 0o700


@pytest.mark.parametrize("change, says", [
    (lambda d: (d / "shards" / "w0.bin").write_bytes(b"\x01" * 1000), "changed shards/w0.bin"),
    (lambda d: (d / "extra.bin").write_bytes(b"x"), "unexpected extra.bin"),
    (lambda d: (d / "config.json").unlink(), "missing config.json"),
    (lambda d: (d / ENTRY_FILE).write_text("{}"), "its ENTRY.json has sha256"),
    (lambda d: _replace_by_symlink(d / "config.json"), "config.json is a symlink"),
], ids=["changed", "added", "removed", "manifest", "link"])
def test_a_verified_copy_that_changed_since_is_refused_by_the_verify_role_not_reused(tmp_path, change, says):
    # Something wrote into a copy after it was verified; a later stage of the job must not mount it.
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    change(_materialize(cache, e, tmp_path / "node", "job-1"))
    roles = _Roles(e)
    with pytest.raises(Refused, match=says):
        _materialize(cache, e, tmp_path / "node", "job-1", roles)
    assert roles.runs == ["verify"]


def test_a_copy_whose_entry_manifest_changed_while_it_was_made_is_refused(tmp_path):
    # The copy is of the ENTRY.json the stage pinned, or of none.
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    (e.path / ENTRY_FILE).write_text((e.path / ENTRY_FILE).read_text() + "\n")
    with pytest.raises(Refused, match=f"the entry's ENTRY.json has sha256 .*, not the {entry_sha256(e)} its "
                                      "producer recorded"):
        _materialize(cache, e, tmp_path / "node", "job-1")
    assert list((tmp_path / "node" / "job-1" / "weights").iterdir()) == []


@pytest.mark.parametrize("report, says", [
    (lambda got: {**got, "shards/w0.bin": "c" * 64}, "changed shards/w0.bin"),
    (lambda got: {k: v for k, v in got.items() if k != ENTRY_FILE}, "its ENTRY.json is not the entry's"),
    (lambda got: {k: v for k, v in got.items() if k != "config.json"}, "no digest was reported for config"),
], ids=["wrong digest", "no manifest digest", "no digest"])
def test_a_copy_is_kept_only_when_its_roles_report_matches_the_manifest(tmp_path, report, says):
    """The runner holds the copy's names and sizes, with the digests its role reported, against
    the manifest it pinned: a role that passed a copy it should not have is caught."""
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    roles = _Roles(e)
    with pytest.raises(CacheCorrupt, match=says):
        cache.materialize(e, tmp_path / "node", "job-1", copy=lambda dest: report(roles.copy(dest)),
                          verify=roles.verify)
    assert list((tmp_path / "node" / "job-1" / "weights").iterdir()) == []


def test_a_copy_with_a_file_its_role_did_not_report_is_not_kept(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    roles = _Roles(e)

    def planting(dest: Path) -> dict[str, str]:
        got = roles.copy(dest)
        (dest / "planted.bin").write_bytes(b"x")
        return {**got, "planted.bin": sha256_bytes(b"x")}

    with pytest.raises(CacheCorrupt, match="unexpected planted.bin"):
        cache.materialize(e, tmp_path / "node", "job-1", copy=planting, verify=roles.verify)


def test_a_copy_without_its_manifest_is_not_kept_whatever_its_role_reported(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    roles = _Roles(e)

    def dropping(dest: Path) -> dict[str, str]:
        got = roles.copy(dest)
        (dest / ENTRY_FILE).unlink()
        return got

    with pytest.raises(CacheCorrupt, match="its ENTRY.json is not the entry's"):
        cache.materialize(e, tmp_path / "node", "job-1", copy=dropping, verify=roles.verify)
    assert list((tmp_path / "node" / "job-1" / "weights").iterdir()) == []


def test_a_copy_whose_manifest_is_a_link_is_not_kept_even_to_the_same_bytes(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    roles = _Roles(e)

    def linking(dest: Path) -> dict[str, str]:
        got = roles.copy(dest)
        kept = tmp_path / "kept-ENTRY.json"
        kept.write_bytes((dest / ENTRY_FILE).read_bytes())
        (dest / ENTRY_FILE).unlink()
        (dest / ENTRY_FILE).symlink_to(kept)
        return got

    # The scan refuses the link before the manifest's own check would.
    with pytest.raises(CacheCorrupt, match="ENTRY.json is not a regular file"):
        cache.materialize(e, tmp_path / "node", "job-1", copy=linking, verify=roles.verify)
    assert list((tmp_path / "node" / "job-1" / "weights").iterdir()) == []


def _stamp(copy) -> dict:
    return json.loads((copy / VERIFIED_FILE).read_text())


def test_the_copy_is_stamped_with_the_entry_digest_the_job_and_a_fingerprint(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-1")
    stamp = _stamp(d)
    assert (stamp["entry_sha256"], stamp["job_id"]) == (entry_sha256(e), "job-1")
    assert re.fullmatch(r"[0-9a-f]{64}", stamp["fingerprint"])


def test_materialize_recopies_when_the_marker_holds_another_stamp(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-1")
    (d / "shards" / "w0.bin").write_bytes(b"\x01" * 1000)
    (d / VERIFIED_FILE).write_text("0" * 64)
    roles = _Roles(e)
    assert _materialize(cache, e, tmp_path / "node", "job-1", roles) == d
    assert (d / "shards" / "w0.bin").read_bytes() == b"\x00" * 1000
    assert _stamp(d)["entry_sha256"] == entry_sha256(e) and roles.runs == ["copy"]


def test_a_copy_of_an_entry_filled_again_under_its_key_is_made_afresh(tmp_path):
    # The copy on disk is of the old entry: its stamp names the old ENTRY.json, so it is never
    # taken for a copy of the new one, however fresh its fingerprint.
    store = PosixStore(tmp_path / "store")
    cache = ContentCache(store)
    old = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, old, tmp_path / "node", "job-1")
    shutil.rmtree(old.path)
    new = populate(cache, "weights", "k", lambda s: (s / "w.bin").write_bytes(b"refilled"),
                   holder=HOLDER).entry
    assert _materialize(cache, new, tmp_path / "node", "job-1") == d
    assert sorted(p.name for p in d.iterdir()) == sorted([ENTRY_FILE, VERIFIED_FILE, "w.bin"])
    assert (d / "w.bin").read_bytes() == b"refilled" and _stamp(d)["entry_sha256"] == entry_sha256(new)


def test_a_copy_this_job_verified_is_reused_without_a_role(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-1")
    roles = _Roles(e)
    assert _materialize(cache, e, tmp_path / "node", "job-1", roles) == d
    assert _materialize(cache, e, tmp_path / "node", "job-1", roles) == d
    assert roles.runs == []


def _replace_by_an_equal_file(path: Path) -> None:
    """Replace the file at *path* by a new file, another inode, with its bytes and times."""
    st = os.lstat(path)
    twin = path.with_name(f"{path.name}.twin")
    twin.write_bytes(path.read_bytes())
    os.utime(twin, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(twin, path)


def test_a_file_replaced_by_an_equal_one_makes_the_reuse_run_the_verify_role_once(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-1")
    _replace_by_an_equal_file(d / "config.json")
    roles = _Roles(e)
    assert _materialize(cache, e, tmp_path / "node", "job-1", roles) == d
    # The bytes and times still match, so the copy is stamped again and the next reuse trusts it.
    assert _materialize(cache, e, tmp_path / "node", "job-1", roles) == d and roles.runs == ["verify"]


def test_a_touched_file_is_refused_by_the_verify_role_not_reused(tmp_path):
    # A file's time is part of the copy: a build tool would take the touched file for a new one.
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-1")
    st = (d / "config.json").stat()
    os.utime(d / "config.json", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    roles = _Roles(e)
    with pytest.raises(Refused, match=f"config.json has mtime {st.st_mtime_ns + 1_000_000} ns, not the recorded "
                                      f"{st.st_mtime_ns}"):
        _materialize(cache, e, tmp_path / "node", "job-1", roles)
    assert roles.runs == ["verify"]


def test_a_write_that_keeps_a_files_size_and_mtime_is_still_found(tmp_path):
    # mtime can be set back; ctime cannot.
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-1")
    st = (d / "shards" / "w0.bin").stat()
    (d / "shards" / "w0.bin").write_bytes(b"\x07" * 1000)
    os.utime(d / "shards" / "w0.bin", ns=(st.st_atime_ns, st.st_mtime_ns))
    with pytest.raises(Refused, match="changed shards/w0.bin"):
        _materialize(cache, e, tmp_path / "node", "job-1")


def test_a_stamp_another_job_wrote_is_not_trusted(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-1")
    (d / VERIFIED_FILE).write_text(json.dumps({**_stamp(d), "job_id": "job-0"}))
    roles = _Roles(e)
    assert _materialize(cache, e, tmp_path / "node", "job-1", roles) == d
    assert roles.runs == ["verify"] and _stamp(d)["job_id"] == "job-1"


def test_a_copy_verified_again_is_kept_only_when_its_roles_report_matches_the_manifest(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = _materialize(cache, e, tmp_path / "node", "job-0")
    (d / VERIFIED_FILE).write_text(json.dumps({**_stamp(d), "job_id": "job-x"}))
    roles = _Roles(e)
    with pytest.raises(CacheCorrupt, match="copy of weights/k changed since it was verified: changed config"):
        cache.materialize(e, tmp_path / "node", "job-0", copy=roles.copy,
                          verify=lambda c: {**roles.verify(c), "config.json": "c" * 64})
    assert roles.runs == ["verify"] and _stamp(d)["job_id"] == "job-x"


def test_materialize_refuses_a_symlink_even_when_it_leads_to_the_same_bytes(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    same = tmp_path / "same.bin"
    same.write_bytes(b"\x00" * 1000)
    (e.path / "shards" / "w0.bin").unlink()
    (e.path / "shards" / "w0.bin").symlink_to(same)
    with pytest.raises(Refused, match="symlink"):
        _materialize(cache, e, tmp_path / "node", "job-1")
    assert list((tmp_path / "node" / "job-1" / "weights").iterdir()) == []


def test_materialize_replaces_a_corrupt_copy(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    d = tmp_path / "node" / "job-1" / "weights" / "k"
    d.mkdir(parents=True)
    (d / "config.json").write_text("partial")
    got = _materialize(cache, e, tmp_path / "node", "job-1")
    assert (got / "config.json").read_text() == '{"a": 1}'


def test_materialize_refuses_an_entry_that_no_longer_matches(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    (e.path / "shards" / "w0.bin").write_bytes(b"\x01" * 1000)
    with pytest.raises(Refused, match="changed shards/w0.bin"):
        _materialize(cache, e, tmp_path / "node", "job-1")
    assert list((tmp_path / "node" / "job-1" / "weights").iterdir()) == []


def test_a_published_entry_made_deeper_than_the_limit_is_refused_before_anything_is_copied(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    _nest(e.path / "shards", sys.getrecursionlimit() + 100)
    with pytest.raises(Refused, match=f"nested deeper than {MAX_DEPTH} levels"):
        _materialize(cache, e, tmp_path / "node", "job-1")
    assert list((tmp_path / "node" / "job-1" / "weights").iterdir()) == []
    # pytest removes old temporary directories by recursing, which a tree this deep would overflow.
    remove_tree(e.path / "shards" / "deep")


@launches
def _role_process(mode: str, entry: CacheEntry, path: Path, report: Path) -> dict[str, str]:
    """Run the cache's role program as its own process, as a role step runs it."""
    where = ["--entry", str(entry.path), "--dest", str(path)] if mode == "copy" else ["--copy", str(path)]
    done = subprocess.run([sys.executable, "-m", "hls.store.cache_role", mode, *where, "--expect",
                           entry_sha256(entry), "--digests-out", str(report)], capture_output=True, text=True,
                          timeout=60)
    assert done.returncode == 0, done.stderr
    got = read_digests(report)
    assert got is not None
    return got


def test_the_runner_reads_no_file_of_an_entry_or_of_its_copy(tmp_path):
    """Every byte of an entry is read on the node by a role: the process that copies, reuses and
    verifies copies reads only the entry's manifest and the copy's stamp."""
    store = PosixStore(tmp_path / "store")
    cache = ContentCache(store)
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    reports = tmp_path / "reports"
    reports.mkdir()

    def copy(dest: Path) -> dict[str, str]:
        return _role_process("copy", e, dest, reports / "copy.json")

    def verify(copy_dir: Path) -> dict[str, str]:
        return _role_process("verify", e, copy_dir, reports / "verify.json")

    node = tmp_path / "node"
    with recording_reads() as reads:
        d = cache.materialize(e, node, "job-1", copy=copy, verify=verify)
        assert cache.materialize(e, node, "job-1", copy=copy, verify=verify) == d
        (d / VERIFIED_FILE).write_text(json.dumps({**_stamp(d), "job_id": "job-0"}))
        assert cache.materialize(e, node, "job-1", copy=copy, verify=verify) == d
    assert entry_reads(reads, [store.root / "cache", node]) == []
    assert (d / "shards" / "w0.bin").read_bytes() == b"\x00" * 1000


def test_the_read_audit_sees_reads_under_a_directory_descriptor_and_processes_no_launcher_started(tmp_path):
    """The audit the tests above rely on: a relative open under a directory descriptor, whose
    event names no file from the working directory, and a process started outside a launcher
    are both kept, whatever the roots; a launcher's processes are not."""
    (tmp_path / "w.bin").write_bytes(b"x")
    fd = os.open(tmp_path, os.O_RDONLY)

    @launches
    def launcher() -> None:
        subprocess.run(["true"], check=True)

    try:
        with recording_reads() as reads:
            os.close(os.open("w.bin", os.O_RDONLY, dir_fd=fd))
            subprocess.run(["true"], check=True)
            launcher()
    finally:
        os.close(fd)
    kept = entry_reads(reads, [tmp_path / "elsewhere"])
    assert kept[0] == UNRESOLVED + "w.bin" and kept[1].startswith(SPAWNED + "subprocess.Popen")
    assert len(kept) == 2


@pytest.mark.parametrize("bad", ["", "a/b", ".x", "a b", "../x", "a\n"])
def test_kind_key_and_job_id_must_be_one_segment(tmp_path, bad):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    with pytest.raises(ValueError):
        cache.lookup(bad, "k")
    with pytest.raises(ValueError):
        populate(cache, "weights", bad, _fill_two_files, holder=HOLDER).entry
    e = populate(cache, "weights", "k", _fill_two_files, holder=HOLDER).entry
    with pytest.raises(ValueError):
        _materialize(cache, e, tmp_path / "node", bad)


def test_file_digests_are_bare_hex():
    FileRecord(path="w.bin", size=1, sha256="a" * 64, mtime_ns=0)
    for bad in ("sha256:" + "a" * 64, "a" * 64 + "\n"):
        with pytest.raises(ValidationError):
            FileRecord(path="w.bin", size=1, sha256=bad, mtime_ns=0)


def test_a_file_record_carries_its_mtime():
    with pytest.raises(ValidationError, match="mtime_ns"):
        FileRecord(path="w.bin", size=1, sha256="a" * 64)


def test_the_cache_hashes_nothing_itself():
    """The runner-side module keeps no way to hash a file: digests come from the roles."""
    assert not hasattr(cache_module, "sha256_file") and not hasattr(ContentCache, "verify")
    assert cache_role.copy_entry is copy_entry


# ---- a writable seed --------------------------------------------------------------------

def test_a_seed_is_a_writable_copy_of_the_entry_that_nothing_stamps_or_reuses(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "engine-cache", "k", _fill_two_files, holder=HOLDER).entry
    roles = _Roles(e)
    dest = tmp_path / "node" / "role" / "engine-cache"
    cache.seed(e, dest, copy=roles.copy)
    assert (dest / "shards" / "w0.bin").read_bytes() == b"\x00" * 1000
    assert (dest / ENTRY_FILE).read_bytes() == (e.path / ENTRY_FILE).read_bytes()
    assert not (dest / VERIFIED_FILE).exists() and roles.runs == ["copy"]
    # A role writes into it as into its own.
    (dest / "shards" / "w0.bin").write_bytes(b"tuned")
    (dest / "new.bin").write_bytes(b"compiled")
    assert sorted(p.name for p in dest.parent.iterdir()) == ["engine-cache"]


def test_a_seed_is_made_only_where_nothing_is(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "engine-cache", "k", _fill_two_files, holder=HOLDER).entry
    dest = tmp_path / "engine-cache"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        cache.seed(e, dest, copy=_Roles(e).copy)


@pytest.mark.parametrize("report, says", [
    (lambda got: {**got, "shards/w0.bin": "c" * 64}, "changed shards/w0.bin"),
    (lambda got: {k: v for k, v in got.items() if k != ENTRY_FILE}, "its ENTRY.json is not the entry's"),
], ids=["wrong digest", "no manifest digest"])
def test_a_seed_is_kept_only_when_its_roles_report_matches_the_manifest(tmp_path, report, says):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "engine-cache", "k", _fill_two_files, holder=HOLDER).entry
    roles = _Roles(e)
    dest = tmp_path / "node" / "engine-cache"
    with pytest.raises(CacheCorrupt, match=says):
        cache.seed(e, dest, copy=lambda d: report(roles.copy(d)))
    assert list(dest.parent.iterdir()) == []


def test_a_node_copy_takes_no_more_disk_than_its_entry_states(tmp_path):
    """``copy_disk_bytes`` bounds what a node-local copy of an entry takes on disk, so a stage
    can declare the disk of its data's copies before it makes them: each file and the copy's
    ENTRY.json in whole blocks, its stamp in one, and a block for the name of every file and
    directory, which bounds what the directories take."""
    block = COPY_BLOCK_BYTES
    sizes = {"empty": 0, "one": 1, "almost": block - 1, "over": block + 1, "a/b/c/deep.bin": 3 * block,
             "a/big.bin": 2**20 + 7}

    def fill(d: Path) -> None:
        for rel, n in sizes.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_bytes(os.urandom(n))

    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = populate(cache, "weights", "k", fill, holder=HOLDER).entry
    assert e.manifest_size == (e.path / ENTRY_FILE).stat().st_size
    copy = _materialize(cache, e, tmp_path / "node", "job-1")
    assert disk_usage([copy]) <= copy_disk_bytes(e)
    data = sum(-(-n // block) * block for n in sizes.values()) + -(-e.manifest_size // block) * block + block
    names = block * (len(sizes) + 2 + len({".", "a", "a/b", "a/b/c"}))
    assert copy_disk_bytes(e) == data + names


# ---- file times -----------------------------------------------------------------------------

#: A JIT's build tree, each file with its mtime in ns: every output newer than its source, while
#: a copy in path order would write the source last.
BUILT_MTIMES = {"a/kernel.so": 1_700_000_300_123_456_789, "a/kernel.o": 1_700_000_200_123_456_789,
                "z/kernel.cu": 1_700_000_100_123_456_789}


def _built(root: Path) -> Path:
    for rel, mtime_ns in BUILT_MTIMES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(rel.encode())
        os.utime(root / rel, ns=(mtime_ns, mtime_ns))
    return root


def _mtimes(root: Path) -> dict[str, int]:
    return {p: os.lstat(root / p).st_mtime_ns for p in BUILT_MTIMES}


def _fill_role(cache: ContentCache, source: Path) -> CacheEntry:
    """The entry the fill role publishes from the tree *source*, as ``StageContext.publish`` fills one."""
    reported: dict[str, str] = {}
    return cache.populate("engine-cache", "k", lambda staging: reported.update(cache_role.fill_entry(source, staging)),
                          holder=HOLDER, digests=lambda: reported).entry


def test_a_fill_then_a_copy_keeps_each_files_mtime_so_outputs_stay_newer_than_their_sources(tmp_path):
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = _fill_role(cache, _built(tmp_path / "engine-cache"))
    assert {f.path: f.mtime_ns for f in e.manifest.files} == BUILT_MTIMES
    assert _mtimes(_materialize(cache, e, tmp_path / "node", "job-1")) == BUILT_MTIMES
    seeded = tmp_path / "node" / "role" / "engine-cache"
    cache.seed(e, seeded, copy=_Roles(e).copy)
    assert _mtimes(seeded) == BUILT_MTIMES


def _retimed(roles: _Roles, path: str):
    """A copy role that makes the copy as *roles* does and then moves the mtime of file *path*
    by one ns, its report unchanged: a role that passed a copy it should not have."""

    def copy(dest: Path) -> dict[str, str]:
        got = roles.copy(dest)
        st = os.lstat(dest / path)
        os.utime(dest / path, ns=(st.st_atime_ns, st.st_mtime_ns + 1))
        return got

    return copy


def test_a_copy_whose_mtime_is_not_the_record_is_not_kept(tmp_path):
    """The runner holds each file's mtime against the record, as it holds names and sizes."""
    cache = ContentCache(PosixStore(tmp_path / "store"))
    e = _fill_role(cache, _built(tmp_path / "engine-cache"))
    roles = _Roles(e)
    says = "a/kernel.o has mtime 1700000200123456790 ns, not the recorded 1700000200123456789"
    with pytest.raises(CacheCorrupt, match=says):
        cache.materialize(e, tmp_path / "node", "job-1", copy=_retimed(roles, "a/kernel.o"), verify=roles.verify)
    assert list((tmp_path / "node" / "job-1" / "engine-cache").iterdir()) == []
    seeded = tmp_path / "node" / "role" / "engine-cache"
    with pytest.raises(CacheCorrupt, match=says):
        cache.seed(e, seeded, copy=_retimed(roles, "a/kernel.o"))
    assert list(seeded.parent.iterdir()) == []


def _publish_without_mtimes(root: Path, kind: str, key: str, *, drop: str = "mtime_ns", named: str | None = None,
                            only: int | None = None) -> Path:
    """Publish an entry whose file records lack the field *drop*, by default their mtimes: every
    record, or the record at index *only*. Its manifest names the entry *named* (a key) when given, else its own."""
    d = root / "cache" / kind / key
    d.mkdir(parents=True)
    _fill_two_files(d)
    files = [{k: v for k, v in f.model_dump().items() if not (k == drop and only in (None, i))}
             for i, f in enumerate(scan_dir(d, digests=hash_tree(d)))]
    (d / ENTRY_FILE).write_text(json.dumps({"kind": kind, "key": named or key, "files": files,
                                            "total_size": 1008, "created_at": 1.0, "source": {}}))
    return d


def test_an_entry_whose_records_carry_no_mtimes_is_absent_and_filled_again(tmp_path):
    """No copy of such an entry could restore its files' times, so no stage may use it: the next
    fill takes its place under the entry's lock."""
    older = _publish_without_mtimes(tmp_path, "engine-cache", "k")
    cache = ContentCache(PosixStore(tmp_path))
    assert cache.lookup("engine-cache", "k") is None
    got = populate(cache, "engine-cache", "k", lambda d: (d / "new.bin").write_bytes(b"built again"),
                   holder=HOLDER)
    assert got.published and got.entry.path == older
    assert [f.path for f in got.entry.manifest.files] == ["new.bin"]
    assert sorted(p.name for p in older.iterdir()) == sorted([ENTRY_FILE, "new.bin"])
    assert entry_sha256(cache.lookup("engine-cache", "k")) == entry_sha256(got.entry)
    assert list(_staging_root(tmp_path, "engine-cache").iterdir()) == []


def test_an_entry_whose_records_lack_anything_else_is_still_corrupt(tmp_path):
    _publish_without_mtimes(tmp_path, "weights", "k", drop="sha256")
    cache = ContentCache(PosixStore(tmp_path))
    with pytest.raises(CacheCorrupt, match="not a valid entry manifest"):
        cache.lookup("weights", "k")
    with pytest.raises(CacheCorrupt, match="not a valid entry manifest"):
        populate(cache, "weights", "k", _no_fill, holder=HOLDER).entry


@pytest.mark.parametrize("make, says", [
    (lambda root: _publish_without_mtimes(root, "weights", "k", named="other"), "holds the entry weights/other"),
    (lambda root: _publish_without_mtimes(root, "weights", "k", only=0), "not a valid entry manifest"),
], ids=["another entry's", "some records timed"])
def test_a_manifest_without_some_mtimes_that_is_not_an_untimed_entry_is_still_corrupt(tmp_path, make, says):
    make(tmp_path)
    cache = ContentCache(PosixStore(tmp_path))
    with pytest.raises(CacheCorrupt, match=says):
        cache.lookup("weights", "k")
    with pytest.raises(CacheCorrupt, match=says):
        populate(cache, "weights", "k", _no_fill, holder=HOLDER).entry
    assert (tmp_path / "cache" / "weights" / "k" / "config.json").exists()


@pytest.mark.parametrize("mtime", [None, "x", 1.5], ids=["null", "string", "fraction"])
def test_a_record_whose_mtime_is_there_but_not_a_time_is_corrupt(tmp_path, mtime):
    d = _publish_without_mtimes(tmp_path, "weights", "k")
    manifest = json.loads((d / ENTRY_FILE).read_text())
    (d / ENTRY_FILE).write_text(json.dumps({**manifest, "files": [{**f, "mtime_ns": mtime} for f in manifest["files"]]}))
    with pytest.raises(CacheCorrupt, match="not a valid entry manifest"):
        ContentCache(PosixStore(tmp_path)).lookup("weights", "k")


def test_a_refill_reading_from_the_untimed_entry_it_replaces_still_finds_it(tmp_path):
    """A fill that reads the entry it replaces, an entry whose file records carry no times, still
    finds it: the entry stays in place until the new one is published."""
    older = _publish_without_mtimes(tmp_path, "images", "sha256-x")
    cache = ContentCache(PosixStore(tmp_path))
    got = populate(cache, "images", "sha256-x",
                   lambda d: (d / "config.json").write_bytes((older / "config.json").read_bytes()), holder=HOLDER)
    assert got.published and (got.entry.path / "config.json").read_text() == '{"a": 1}'
    assert [f.path for f in got.entry.manifest.files] == ["config.json"]
    assert list(_staging_root(tmp_path, "images", "sha256-x").iterdir()) == []


def test_a_failed_refill_leaves_the_untimed_entry_where_it_was(tmp_path):
    older = _publish_without_mtimes(tmp_path, "images", "sha256-x")
    cache = ContentCache(PosixStore(tmp_path))

    def failing(d: Path) -> None:
        raise RuntimeError("the import's source is unreadable")

    with pytest.raises(RuntimeError):
        populate(cache, "images", "sha256-x", failing, holder=HOLDER)
    assert (older / "config.json").read_text() == '{"a": 1}' and cache.lookup("images", "sha256-x") is None


def _taking_over(monkeypatch, store: PosixStore, act) -> None:
    """Make the look ``_retire_untimed`` takes at the place stale: before it returns what it read,
    *act* runs, as a successor that took the lock over meanwhile; the successor holds the lock."""
    real = ContentCache._find

    def stale(self, kind, key):
        found = real(self, kind, key)
        if sys._getframe(1).f_code.co_name == "_retire_untimed":
            act(kind, key)
            GenerationLock(store, f"cache-locks/{kind}/{key}", ttl_sec=120).try_acquire(
                Holder(host="h", pid=2, job_id="2", purpose="successor"), alive=lambda h: False)
        return found

    monkeypatch.setattr(ContentCache, "_find", stale)


def test_a_holder_that_lost_its_lock_never_removes_the_entry_its_successor_published(tmp_path, monkeypatch):
    """Holder A read the untimed entry and stalled past its lock's TTL; its successor retired it
    and published its own. A moves the place aside, finds a timed entry there, puts it back and
    reports the loss: the successor's entry stands."""
    store = PosixStore(tmp_path)
    older = _publish_without_mtimes(tmp_path, "weights", "k")

    def successor(kind: str, key: str) -> None:
        shutil.rmtree(older)
        _publish_by_hand(tmp_path, kind, key)

    _taking_over(monkeypatch, store, successor)
    cache = ContentCache(store)
    got = populate(cache, "weights", "k", lambda d: (d / "mine.bin").write_bytes(b"A's"), holder=HOLDER)
    assert not got.published and got.entry.manifest.created_at == 1.0
    assert sorted(p.name for p in older.iterdir()) == sorted([ENTRY_FILE, "config.json", "shards"])
    assert entry_sha256(cache.lookup("weights", "k")) == entry_sha256(got.entry)
    assert list(_staging_root(tmp_path).iterdir()) == []


def test_a_holder_whose_untimed_entry_another_retired_meanwhile_reports_the_loss(tmp_path, monkeypatch):
    """Two holders found it untimed; the other moved it first. The rename finds nothing, which is
    the entry retired already, and the holder that lost its lock publishes nothing."""
    store = PosixStore(tmp_path)
    older = _publish_without_mtimes(tmp_path, "weights", "k")
    _taking_over(monkeypatch, store, lambda kind, key: shutil.rmtree(older))
    cache = ContentCache(store)
    with pytest.raises(LockLost):
        populate(cache, "weights", "k", lambda d: (d / "mine.bin").write_bytes(b"A's"), holder=HOLDER)
    assert cache.lookup("weights", "k") is None and not older.exists()


def test_a_lost_holder_whose_moved_aside_successor_entry_was_swept_publishes_nothing(tmp_path, monkeypatch):
    """Holder A, its lock lost, moved its successor's entry aside, and a third holder that found
    the place empty took the lock and swept the staging root before A read it back. Only a holder
    of the lock sweeps there, so A has lost it: it reports the loss and publishes nothing."""
    store = PosixStore(tmp_path)
    older = _publish_without_mtimes(tmp_path, "weights", "k")

    def successor(kind: str, key: str) -> None:
        shutil.rmtree(older)
        _publish_by_hand(tmp_path, kind, key)

    _taking_over(monkeypatch, store, successor)
    real = ContentCache._read_place

    def swept_first(self, path, kind, key):
        if path.name.startswith(".retired-"):
            shutil.rmtree(path)
        return real(self, path, kind, key)

    monkeypatch.setattr(ContentCache, "_read_place", swept_first)
    cache = ContentCache(store)
    with pytest.raises(LockLost):
        populate(cache, "weights", "k", lambda d: (d / "mine.bin").write_bytes(b"A's"), holder=HOLDER)
    assert cache.lookup("weights", "k") is None


def test_a_holder_whose_moved_aside_entry_vanished_under_its_lock_publishes_nothing(tmp_path, monkeypatch):
    """Nothing legitimate removes what a holder moved aside while it holds the lock: the fill is
    refused, and the place stays empty."""
    older = _publish_without_mtimes(tmp_path, "weights", "k")
    real = ContentCache._read_place

    def swept_first(self, path, kind, key):
        if path.name.startswith(".retired-"):
            shutil.rmtree(path)
        return real(self, path, kind, key)

    monkeypatch.setattr(ContentCache, "_read_place", swept_first)
    cache = ContentCache(PosixStore(tmp_path))
    with pytest.raises(CacheCorrupt, match="vanished"):
        populate(cache, "weights", "k", lambda d: (d / "mine.bin").write_bytes(b"A's"), holder=HOLDER)
    assert not older.exists() and cache.lookup("weights", "k") is None
