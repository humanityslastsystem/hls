from __future__ import annotations

import errno
import os
import subprocess
from dataclasses import replace
import sys
import threading
import tracemalloc
from pathlib import Path

import contextlib
import shutil

import pytest

from hls.pipeline.stage import scored_end_fault, stage_fault
from hls.substrate import memory_watch
from hls.substrate.faults import RoleOutcome, classify_role, starvation_fault
from hls.substrate.memory_watch import DiskAtEnd, MemoryWatch, PidsHits, disk_usage, held_unlinked
from hls.substrate.nproc import CeilingRule, PoolAtEnd, PoolPeak, PoolSample
from tests.substrate.nproc_pool import UidPool

STEP = "1234110.1"
MIB = 2**20


@pytest.mark.parametrize("owner", ["harness", "contestant"])
def test_a_planned_stop_is_no_fault(owner):
    outcome = RoleOutcome("server", owner, -15, "planned_stop")
    assert classify_role(outcome) is None and stage_fault([outcome]) is None


@pytest.mark.parametrize("outcome, fault_class", [
    (RoleOutcome("server", "harness", -15, "planned_stop", oom_kills=1), "infra"),
    (RoleOutcome("server", "contestant", -15, "planned_stop", oom_kills=1, own_oom=True, job_oom=False), "contestant"),
    (RoleOutcome("server", "contestant", -15, "planned_stop", oom_kills=1), "infra"),
])
def test_an_oom_kill_before_a_planned_stop_still_counts(outcome, fault_class):
    fault = classify_role(outcome)
    assert fault is not None and fault.fault_class == fault_class


def test_writing_past_the_disk_cap_is_the_contestants_fault():
    fault = classify_role(RoleOutcome("artifact", "contestant", -9, "disk_limit", detail="5 bytes over 1 GiB"))
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason
    assert fault.reason.endswith(": 5 bytes over 1 GiB")  # the record says how the cap was passed
    assert (fault.evidence["ending"], fault.evidence["detail"]) == ("disk_limit", "5 bytes over 1 GiB")
    harness = classify_role(RoleOutcome("driver", "harness", -9, "disk_limit"))
    assert (harness.fault_class, harness.retryable) == ("infra", True)


def _du(path: Path) -> int:
    return int(subprocess.run(["du", "-s", "-B1", str(path)], capture_output=True, text=True,
                              check=True).stdout.split()[0])


def test_the_disk_used_is_what_du_counts(tmp_path):
    d = tmp_path / "wd"
    (d / "a" / "b").mkdir(parents=True)
    (d / "a" / "b" / "f").write_bytes(b"x" * 3 * MIB)
    (d / "small").write_bytes(b"y")
    os.link(d / "a" / "b" / "f", d / "again")  # one inode, counted once
    outside = tmp_path / "outside"
    outside.write_bytes(b"z" * 5 * MIB)
    (d / "link").symlink_to(outside)  # never followed
    used = disk_usage([d])
    assert used == _du(d)
    assert 3 * MIB <= used < 4 * MIB
    assert disk_usage([d, d / "a"]) == used  # a path inside another is not counted twice
    assert disk_usage([tmp_path / "gone"]) == 0


@pytest.mark.parametrize("mode", [0, 0o400, 0o100], ids=["nothing", "read-only", "search-only"])
def test_a_directory_its_owner_cannot_list_is_still_counted(tmp_path, mode):
    hidden = tmp_path / "wd" / "hidden"
    (hidden / "deeper").mkdir(parents=True)
    (hidden / "deeper" / "f").write_bytes(b"x" * 2 * MIB)
    (hidden / "deeper").chmod(mode)
    hidden.chmod(mode)
    try:
        assert disk_usage([tmp_path / "wd"]) >= 2 * MIB
        # The walk gave the owner back what it needs, and nothing more.
        assert (hidden.stat().st_mode & 0o777, (hidden / "deeper").stat().st_mode & 0o777) == (0o700, 0o700)
    finally:
        hidden.chmod(0o700)
        (hidden / "deeper").chmod(0o700)


def test_a_link_to_a_directory_it_cannot_list_is_never_followed_to_change_it(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o500)
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "link").symlink_to(outside)
    try:
        disk_usage([wd])
        assert outside.stat().st_mode & 0o777 == 0o500
    finally:
        outside.chmod(0o700)


def _deep(top: Path, levels: int, name: str) -> None:
    """Directories *levels* deep under *top*, each named *name*, with a MiB file at the bottom,
    made through descriptors, so no path is ever spelled out whole."""
    fd = os.open(top, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for _ in range(levels):
            os.mkdir(name, dir_fd=fd)
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
            os.close(fd)
            fd = child
        out = os.open("f", os.O_WRONLY | os.O_CREAT, 0o600, dir_fd=fd)
        os.write(out, b"x" * MIB)
        os.fsync(out)
        os.close(out)
    finally:
        os.close(fd)


def test_a_tree_as_deep_as_the_walk_goes_is_walked_whatever_its_paths_length(tmp_path):
    d = tmp_path / "wd"
    d.mkdir()
    _deep(d, memory_watch.DISK_WALK_MAX_DEPTH, "d")
    assert disk_usage([d]) >= MIB
    # Paths far longer than the kernel would take whole are walked too.
    long = tmp_path / "long"
    long.mkdir()
    _deep(long, 40, "n" * 250)
    assert disk_usage([long]) >= MIB


def test_a_tree_deeper_than_the_walk_goes_counts_as_past_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_DEPTH", 5)
    wd = tmp_path / "wd"
    wd.mkdir()
    _deep(wd, 6, "d")
    with pytest.raises(memory_watch.Unmeasured, match="levels deep"):
        disk_usage([wd])
    killed = []
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="artifact")
    watch.sample_disks_once()
    breach = watch.disk_breach(STEP)
    assert killed == [1] and breach is not None and breach.unmeasured == "depth" and breach.used < 1024 * MIB
    # One level less is measured as it is.
    shallow = tmp_path / "shallow"
    shallow.mkdir()
    _deep(shallow, 5, "d")
    assert disk_usage([shallow]) >= MIB


def test_the_walk_leaves_no_descriptor_open(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_DEPTH", 5)
    wd = tmp_path / "wd"
    wd.mkdir()
    _deep(wd, 4, "d")
    before = len(os.listdir("/proc/self/fd"))
    disk_usage([wd])
    _deep(wd / "d", 6, "e")
    with pytest.raises(memory_watch.Unmeasured, match="levels deep"):
        disk_usage([wd])
    assert len(os.listdir("/proc/self/fd")) == before


def test_a_role_past_its_disk_cap_is_killed_and_its_breach_kept(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    watch = MemoryWatch(enforce=False, clock=lambda: 7.0)
    seen = []
    watch.watch_disk(STEP, [wd], 2 * MIB, lambda: seen.append(watch.disk_breach(STEP)), role="artifact")
    watch.sample_disks_once()
    assert seen == [] and watch.disk_breach(STEP) is None
    (wd / "f").write_bytes(b"x" * 3 * MIB)
    watch.sample_disks_once()
    watch.sample_disks_once()
    # The breach is recorded before the kill, and a killed step is not killed again.
    [breach] = seen
    assert breach is not None and (breach.at, breach.used) == (7.0, disk_usage([wd]))
    assert watch.disk_breach(STEP) == breach and watch.disk_breach("1234110.2") is None


@pytest.mark.parametrize("fails", ["the uid's pool", "the step's own tasks", "the pids limit count"])
def test_no_failure_of_the_pool_or_pids_reading_skips_a_disk_cap(tmp_path, monkeypatch, caplog, fails):
    # The pool and the pids count are context; the cap is the contestant's limit. A count that
    # raises, whatever it raises, is logged, and every role is still walked and killed.
    def boom(*args, **kwargs):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    target = {"the uid's pool": "uid_tasks", "the step's own tasks": "step_tasks",
              "the pids limit count": "pids_limit_hits"}[fails]
    monkeypatch.setattr(memory_watch, target, boom)
    killed = []
    watch = MemoryWatch(enforce=False)
    for step in ("1.0", "1.1"):
        wd = tmp_path / step
        wd.mkdir()
        (wd / "f").write_bytes(b"x" * 2 * MIB)
        watch.watch_disk(step, [wd], MIB, lambda step=step: killed.append(step), role=f"r{step}",
                         cgroup_dir=tmp_path / f"cg-{step}", ceiling=131_072)
    with caplog.at_level("ERROR", logger="hls.substrate.memory_watch"):
        watch.sample_disks_once()
    assert sorted(killed) == ["1.0", "1.1"]
    assert watch.disk_breach("1.0") is not None and watch.disk_breach("1.1") is not None
    assert any("UnicodeDecodeError" in (r.exc_text or "") for r in caplog.records)


def test_a_process_named_in_bytes_that_are_not_utf8_leaves_the_disk_cap_in_force(tmp_path):
    # A process anywhere on the node names itself b"\xffcontestant"
    # (prctl PR_SET_NAME), while a role with a process ceiling holds 2 MiB against a 1 MiB cap.
    namer = ("import ctypes, time\n"
             "assert ctypes.CDLL(None).prctl(15, ctypes.c_char_p(b'\\xffcontestant'), 0, 0, 0) == 0\n"
             "print('named', flush=True)\n"
             "time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, "-c", namer], stdout=subprocess.PIPE)
    try:
        assert proc.stdout.readline() == b"named\n"
        wd = tmp_path / "wd"
        wd.mkdir()
        (wd / "fill").write_bytes(b"x" * 2 * MIB)
        killed = []
        watch = MemoryWatch(enforce=False)
        watch.watch_disk("1.0", [wd], MIB, lambda: killed.append(True), role="sandbox", ceiling=131_072)
        watch.sample_disks_once()
        assert killed == [True] and watch.disk_breach("1.0") is not None
        assert watch.pool("1.0") is not None, "the pool was counted too"
    finally:
        proc.kill()
        proc.wait()


def test_a_failed_kill_is_tried_again_at_the_next_walk(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    (wd / "f").write_bytes(b"x" * 3 * MIB)
    calls = []

    def kill():
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("scancel timed out")

    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [wd], MIB, kill, role="artifact")
    watch.sample_disks_once()
    assert calls == [1] and watch.disk_breach(STEP) is None
    watch.sample_disks_once()
    assert calls == [1, 1] and watch.disk_breach(STEP) is not None


def test_a_walk_that_fails_is_tried_again_at_the_next(tmp_path, monkeypatch):
    killed = []
    walks = iter([OSError("the directory went away"), 3 * MIB])

    def walk(paths):
        result = next(walks)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(memory_watch, "disk_usage", walk)
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [tmp_path], MIB, lambda: killed.append(1), role="artifact")
    watch.sample_disks_once()
    assert killed == [] and watch.disk_breach(STEP) is None
    watch.sample_disks_once()
    assert killed == [1] and watch.disk_breach(STEP) is not None


def test_a_reported_role_is_walked_no_more(tmp_path, monkeypatch):
    wd = tmp_path / "wd"
    wd.mkdir()
    killed, walked = [], []
    monkeypatch.setattr(memory_watch, "disk_usage", lambda paths: walked.append(paths) or 3 * MIB)
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [wd], MIB, lambda: killed.append(1), role="artifact")
    watch.unwatch_disk(STEP)
    watch.sample_disks_once()
    assert (walked, killed) == ([], []) and watch.disk_breach(STEP) is None
    watch.unwatch_disk("1234110.9")  # a step never watched is no error


def test_a_role_reported_during_its_walk_gets_no_breach(tmp_path, monkeypatch):
    # The stage reads the role's ending once it has stopped the walks; a walk that was running
    # then records nothing.
    killed = []
    watch = MemoryWatch(enforce=False)

    def walk(paths):
        watch.unwatch_disk(STEP)
        return 3 * MIB

    monkeypatch.setattr(memory_watch, "disk_usage", walk)
    watch.watch_disk(STEP, [tmp_path], MIB, lambda: killed.append(1), role="artifact")
    watch.sample_disks_once()
    assert killed == [] and watch.disk_breach(STEP) is None


def test_the_disk_is_walked_in_a_thread_of_its_own(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    killed, release = threading.Event(), threading.Event()
    # A memory pass that does not return does not keep the disk walk from running.
    watch = MemoryWatch(enforce=False, interval_sec=0.01, disk_interval_sec=0.01)
    watch.sample_once = lambda: release.wait(30)
    watch.watch_disk(STEP, [wd], MIB, killed.set, role="artifact")
    watch.start()
    try:
        (wd / "f").write_bytes(b"x" * 3 * MIB)
        assert killed.wait(5.0)
    finally:
        release.set()
        watch.stop()
    assert not {"hls-memory-watch", "hls-disk-watch"} & {t.name for t in threading.enumerate()}


def _chmod_loop(d: Path) -> subprocess.Popen:
    """A process of the role's uid that takes the owner's permissions off *d* again and again,
    as any of a contestant's processes may in its own tree."""
    code = "import os, sys\nwhile True:\n    try:\n        os.chmod(sys.argv[1], 0)\n    except OSError:\n        pass\n"
    return subprocess.Popen([sys.executable, "-c", code, str(d)])


def test_a_directory_whose_permissions_the_role_keeps_taking_away_is_past_its_cap(tmp_path):
    # The walk gives the owner back what it needs, but the role takes it away again before the
    # walk is done: the walk is refused permission in the role's own tree, which only the
    # role's processes can do, so the cap counts as passed rather than walked again forever.
    wd = tmp_path / "wd"
    d = wd / "d"
    d.mkdir(parents=True)
    for i in range(2000):
        (d / f"f{i}").write_bytes(b"")
    killed: list[int] = []
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="artifact")
    loop = _chmod_loop(d)
    try:
        for _ in range(200):
            watch.sample_disks_once()
            if watch.disk_breach(STEP) is not None:
                break
    finally:
        loop.kill()
        loop.wait()
        d.chmod(0o700)
    breach = watch.disk_breach(STEP)
    assert killed == [1] and breach is not None and breach.unmeasured == "permission"


@pytest.mark.parametrize("error", [errno.EACCES, errno.EPERM])
def test_a_walk_refused_permission_is_a_breach_and_any_other_failure_is_walked_again(tmp_path, monkeypatch, error):
    # Inside the role's tree only its own processes can refuse the walk; any other failure may be
    # the node's, and the next walk tries again.
    killed: list[int] = []
    (tmp_path / "wd").mkdir()
    listings = iter([OSError(errno.EIO, "I/O error"), PermissionError(error, os.strerror(error))])

    def scandir(fd):
        raise next(listings)

    monkeypatch.setattr(memory_watch.os, "scandir", scandir)
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [tmp_path / "wd"], MIB, lambda: killed.append(1), role="artifact")
    watch.sample_disks_once()
    assert killed == [] and watch.disk_breach(STEP) is None
    watch.sample_disks_once()
    monkeypatch.undo()
    breach = watch.disk_breach(STEP)
    assert killed == [1] and breach is not None and breach.unmeasured == "permission"
    assert "refused permission" in breach.detail(1)


def test_a_tree_of_more_entries_than_the_walk_counts_is_past_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_ENTRIES", 10)
    wd = tmp_path / "wd"
    (wd / "a").mkdir(parents=True)
    for i in range(9):
        (wd / "a" / f"f{i}").write_bytes(b"")
    assert disk_usage([wd]) == _du(wd)  # ten entries: the directory and its nine files
    (wd / "a" / "f9").write_bytes(b"")
    with pytest.raises(memory_watch.Unmeasured, match="entries"):
        disk_usage([wd])
    killed: list[int] = []
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="artifact")
    watch.sample_disks_once()
    breach = watch.disk_breach(STEP)
    assert killed == [1] and breach is not None and breach.unmeasured == "entries"


def test_files_with_one_link_are_not_remembered_and_still_counted_once(tmp_path):
    # Only a file with more than one link can be met twice, so the walk remembers no other: its
    # memory does not grow with a tree of empty files.
    wd = tmp_path / "wd"
    wd.mkdir()
    fd = os.open(wd, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for i in range(20000):
            os.close(os.open(f"f{i}", os.O_CREAT | os.O_WRONLY, 0o600, dir_fd=fd))
    finally:
        os.close(fd)
    (wd / "shared").write_bytes(b"y" * MIB)
    os.link(wd / "shared", wd / "again")
    tracemalloc.start()
    try:
        used = disk_usage([wd])
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert used == _du(wd) and peak < MIB


def test_a_link_swapped_in_for_a_directory_after_a_refused_open_is_never_followed(tmp_path, monkeypatch):
    outside = tmp_path / "outside"
    outside.mkdir()
    outside.chmod(0o500)
    wd = tmp_path / "wd"
    (wd / "d").mkdir(parents=True)
    real = os.open
    swapped: list[int] = []

    def racing(path, flags, *args, **kwargs):
        if path == "d" and not flags & os.O_PATH and not swapped:
            # The role takes the directory away and puts a link to one outside its tree there,
            # just as the walk is refused the directory.
            os.rmdir(wd / "d")
            (wd / "d").symlink_to(outside)
            swapped.append(1)
            raise PermissionError(errno.EACCES, "Permission denied")
        return real(path, flags, *args, **kwargs)

    monkeypatch.setattr(memory_watch.os, "open", racing)
    try:
        disk_usage([wd])
    finally:
        monkeypatch.undo()
        mode = outside.stat().st_mode & 0o777
        outside.chmod(0o700)
    assert swapped == [1] and mode == 0o500


def test_what_a_role_wrote_is_removed_whatever_it_did_to_its_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_DEPTH", 5)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep").write_text("not the role's")
    wd, cache = tmp_path / "wd", tmp_path / "engine-cache"
    (wd / "a" / "b").mkdir(parents=True)
    (wd / "a" / "b" / "f").write_bytes(b"x" * MIB)
    (wd / "link").symlink_to(outside)
    os.mkfifo(wd / "fifo")
    cache.mkdir()
    _deep(cache, 12, "d")
    (wd / "a" / "b").chmod(0)
    (wd / "a").chmod(0o100)
    wd.chmod(0o500)  # listable, but nothing in it can be removed
    memory_watch.remove_written([wd, cache, tmp_path / "gone"])
    assert not wd.exists() and not cache.exists()
    assert (outside / "keep").read_text() == "not the role's"


def test_the_step_cgroups_pids_limit_hits_are_read_where_the_step_keeps_them(tmp_path):
    from hls.substrate.cgroup import pids_limit_hits

    user = tmp_path / "step_3" / "user"
    user.mkdir(parents=True)
    assert pids_limit_hits(user) is None
    (user / "pids.events").write_text("max 0\n")
    assert pids_limit_hits(user) == 0
    (user.parent / "pids.events").write_text("max 4\n")
    assert pids_limit_hits(user) == 4
    (user / "pids.events").write_text("max 7\n")
    assert pids_limit_hits(user) == 7


def test_the_disk_walk_also_reads_a_contestant_steps_pids_limit_hits_and_when_they_rose(tmp_path):
    """Each walk reads the count; a rise is dated by the first walk that read it, so the
    evidence rules can bound its age as they bound an OOM's."""
    user = tmp_path / "step_3" / "user"
    user.mkdir(parents=True)
    (user / "pids.events").write_text("max 0\n")
    now = [100.0]
    watch = MemoryWatch(enforce=False, clock=lambda: now[0])
    watch.watch_disk(STEP, [tmp_path / "wd"], MIB, lambda: None, role="artifact", cgroup_dir=user)
    assert watch.pids_hits(STEP) is None
    watch.sample_disks_once()
    assert watch.pids_hits(STEP) == PidsHits(count=0, rose_at=None)
    (user / "pids.events").write_text("max 2\n")
    now[0] = 110.0
    watch.sample_disks_once()
    now[0] = 120.0
    watch.sample_disks_once()  # no rise: the date stays the first read's
    assert watch.pids_hits(STEP) == PidsHits(count=2, rose_at=110.0)
    (user / "pids.events").write_text("max 3\n")
    now[0] = 130.0
    watch.sample_disks_once()
    (user / "pids.events").unlink()  # the step is gone; what was read stays
    now[0] = 140.0
    watch.sample_disks_once()
    assert watch.pids_hits(STEP) == PidsHits(count=3, rose_at=130.0) and watch.pids_hits("1234110.7") is None


def test_the_disk_walk_samples_the_uids_pool_and_the_steps_own_tasks(tmp_path, monkeypatch):
    # Every task of the uid on the node draws on the one pool a contestant step's ceiling bounds;
    # the walk reads it, and the step's own count, every few seconds while the step runs.
    pool = UidPool(tasks=9_000)
    counted: list[int] = []
    monkeypatch.setattr(memory_watch, "uid_tasks", lambda uid: counted.append(uid) or pool.tasks)
    user = tmp_path / "step_3" / "user"
    user.mkdir(parents=True)
    (user / "pids.current").write_text("40\n")
    now = [100.0]
    watch = MemoryWatch(enforce=False, clock=lambda: now[0])
    watch.watch_disk(STEP, [tmp_path / "wd"], MIB, lambda: None, role="artifact", cgroup_dir=user, ceiling=10_000)
    watch.watch_disk("1234110.2", [tmp_path / "wd2"], MIB, lambda: None, role="tool", cgroup_dir=user)
    assert watch.pool(STEP) is None
    watch.sample_disks_once()
    assert watch.pool(STEP) == PoolSample(at=100.0, uid_tasks=9_000, step_tasks=40, ceiling=10_000)
    assert counted == [os.getuid()] and watch.pool("1234110.2") is None
    now[0], pool.tasks = 105.0, 9_600
    watch.sample_disks_once()
    assert watch.pool(STEP) == PoolSample(at=105.0, uid_tasks=9_600, step_tasks=40, ceiling=10_000)
    # From the step's stop on, the last sample before it is what stays.
    watch.unwatch_disk(STEP)
    now[0], pool.tasks = 110.0, 2_000
    watch.sample_disks_once()
    assert watch.pool(STEP).uid_tasks == 9_600 and len(counted) == 2
    assert watch.pool_peak(STEP) == PoolPeak(walks=2, uid_tasks=9_600, step_tasks=40, others_tasks=9_560)
    assert watch.pool_peak("1234110.2") is None


def test_the_walks_keep_the_peak_of_each_count_over_the_steps_life_beside_the_last(tmp_path, monkeypatch):
    # The measurements run.nproc_reserve and FLOOR_PER_EVALUATION are set from: a storm that came
    # and went is in the peak, though the last sample shows it gone.
    pool = UidPool(tasks=9_000)
    monkeypatch.setattr(memory_watch, "uid_tasks", lambda uid: pool.tasks)
    user = tmp_path / "step_3" / "user"
    user.mkdir(parents=True)
    (user / "pids.current").write_text("40\n")
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [tmp_path / "wd"], MIB, lambda: None, role="artifact", cgroup_dir=user, ceiling=10_000)
    assert watch.pool_peak(STEP) is None
    watch.sample_disks_once()
    pool.tasks = 5_000
    (user / "pids.current").write_text("900\n")
    watch.sample_disks_once()
    pool.tasks = 300
    (user / "pids.current").unlink()
    watch.sample_disks_once()
    assert (watch.pool(STEP).uid_tasks, watch.pool(STEP).step_tasks) == (300, None)
    assert watch.pool_peak(STEP) == PoolPeak(walks=3, uid_tasks=9_000, step_tasks=900, others_tasks=8_960)


def test_a_pool_that_cannot_be_counted_keeps_the_last_sample(tmp_path, monkeypatch):
    counts = iter([500, OSError(errno.EACCES, "no /proc")])

    def count(uid):
        value = next(counts)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(memory_watch, "uid_tasks", count)
    watch = MemoryWatch(enforce=False)
    watch.watch_disk(STEP, [tmp_path / "wd"], MIB, lambda: None, role="artifact", cgroup_dir=tmp_path / "cg",
                     ceiling=10_000)
    watch.sample_disks_once()
    watch.sample_disks_once()
    assert watch.pool(STEP).uid_tasks == 500 and watch.pool(STEP).step_tasks is None


@pytest.mark.parametrize("owner, fault_class", [("contestant", "contestant"), ("harness", "infra")])
def test_a_step_that_reached_its_pids_limit_is_its_contestants_end(owner, fault_class):
    outcome = RoleOutcome("artifact", owner, -9, "contestant_pids", detail="its step reached its pids limit 2 times")
    fault = classify_role(outcome)
    assert fault is not None and fault.fault_class == fault_class and fault.retryable == (fault_class == "infra")
    if owner == "contestant":
        assert "process limit" in fault.reason and fault.reason.endswith(": its step reached its pids limit 2 times")


@pytest.mark.parametrize("driver", [RoleOutcome("driver", "harness", 1, "exited"),
                                    RoleOutcome("driver", "harness", None, "scheduler_ended", detail="NODE_FAIL")],
                         ids=["exited", "node-fail"])
def test_a_pids_limit_the_contestant_reached_beside_another_roles_infra_end_leaves_the_stage_infra(driver):
    """A stage is the contestant's only when every failed role's fault is the contestant's: a server
    at its step's pids limit beside a driver that failed as infra, such as one whose node failed,
    leaves the stage infra, retried, and the record keeps both roles' faults."""
    pids = RoleOutcome("artifact", "contestant", -9, "contestant_pids", detail="its step reached its pids limit 1 times")
    fault = stage_fault([pids, driver])
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.evidence["roles"] == "artifact,driver"
    assert (fault.evidence["artifact.ending"], fault.evidence["driver.ending"]) == ("contestant_pids", driver.ending)
    assert "process limit" in fault.reason and "driver" in fault.reason
    # Alone, or beside other contestant faults, the pids limit is the contestant's.
    assert stage_fault([pids]).fault_class == "contestant"
    disk = RoleOutcome("driver", "contestant", -9, "disk_limit")
    assert stage_fault([pids, disk]).fault_class == "contestant"


#: What the walk read last while the server ran: the pool full, of the server's own zombies as much
#: as of anyone's. Context, which decides nothing.
WALK = PoolSample(at=1.0, uid_tasks=9_900, step_tasks=120, ceiling=10_000)
#: The uid's tasks once the server's step was gone: the others' alone, at the pool's full mark and
#: one under it.
BY_UID = CeilingRule(ceiling=10_000, accounting="uid", parent_limit=None)
FULL = PoolAtEnd(at=2.0, others_tasks=9_500, rule=BY_UID, node_tasks=12_000, pid_max=4_194_304)
ROOM = PoolAtEnd(at=2.0, others_tasks=9_499, rule=BY_UID, node_tasks=12_000, pid_max=4_194_304)
#: The uid's tasks could not be counted once the server's step was gone.
UNCOUNTED = PoolAtEnd(at=2.0, others_tasks=None, rule=BY_UID, node_tasks=12_000, pid_max=4_194_304)
#: The same others, against commands whose ceiling counts their own namespace's tasks, its parent
#: levels against a hard limit of eight million.
PER_NAMESPACE = PoolAtEnd(at=2.0, others_tasks=9_500, node_tasks=12_000, pid_max=4_194_304,
                          rule=CeilingRule(ceiling=10_000, accounting="user_namespace", parent_limit=8_000_000))


@pytest.mark.parametrize("code", [1, -11])
def test_a_contestant_server_that_ended_while_the_others_alone_filled_the_pool_was_starved(code):
    outcome = RoleOutcome("artifact", "contestant", code, "exited", pool=WALK, pool_at_end=FULL)
    fault = classify_role(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert "9500" in fault.reason and "10000" in fault.reason
    assert {k: fault.evidence[k] for k in ("others_tasks", "ceiling", "walk_uid_tasks", "walk_step_tasks")} == {
        "others_tasks": "9500", "ceiling": "10000", "walk_uid_tasks": "9900", "walk_step_tasks": "120"}
    assert starvation_fault(outcome) == fault


@pytest.mark.parametrize("at_end", [ROOM, None], ids=["room-left-by-the-others", "no-ceiling"])
def test_a_server_whose_own_tasks_filled_the_pool_ended_as_its_own_result(at_end):
    # The walk saw the pool full and the server's step small, as its unreaped zombies make it:
    # once its step is gone they are too, and the others left room. A role with no ceiling has
    # no pool to be starved of.
    outcome = RoleOutcome("artifact", "contestant", 1, "exited", pool=WALK, pool_at_end=at_end)
    assert classify_role(outcome) is None and starvation_fault(outcome) is None


#: A failing end of each kind a stage scores: one the role exited with, one cut at its time
#: limit, and a server not ready in time that the stage stopped as planned.
FAILING_ENDS = [(1, "exited"), (-11, "exited"), (-9, "time_limit"), (-15, "planned_stop")]


@pytest.mark.parametrize("code, ending", FAILING_ENDS, ids=["exit-1", "segv", "time-limit", "stopped-not-ready"])
def test_a_failing_end_whose_uids_tasks_could_not_be_counted_is_infra_not_the_contestants(code, ending):
    # The count that would rule starvation out failed: the end may be the others', so it is
    # retried rather than scored.
    outcome = RoleOutcome("artifact", "contestant", code, ending, pool=WALK, pool_at_end=UNCOUNTED)
    fault = starvation_fault(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == ("contestant role artifact ended, and the uid's tasks could not be counted once its step "
                            "was gone, so starvation cannot be ruled out")
    assert {k: fault.evidence[k] for k in ("others_tasks", "ceiling", "walk_uid_tasks")} == {
        "others_tasks": "uncounted", "ceiling": "10000", "walk_uid_tasks": "9900"}
    # Every stage asks it before it scores the end: of a role that exited through the roles' own
    # rule, of one the stage ended directly.
    scored = scored_end_fault(outcome)
    assert (scored.fault_class, scored.retryable, scored.reason) == ("infra", True, fault.reason)
    assert (classify_role(outcome) == fault) == (ending == "exited")
    # Counted, with room the others left, the same end is the contestant's to score.
    assert scored_end_fault(RoleOutcome("artifact", "contestant", code, ending, pool_at_end=ROOM)) is None


@pytest.mark.parametrize("at_end", [FULL, UNCOUNTED], ids=["starved", "uncounted"])
def test_an_exit_of_0_is_judged_for_starvation_only_where_a_stage_scores_it_as_a_failure(at_end):
    # A role that exited 0 succeeded, whatever the pool: a pip step is never retried for it. A
    # stage that scores such an end as a failure, a server that exited 0 before it was ready,
    # asks before it does.
    outcome = RoleOutcome("artifact", "contestant", 0, "exited", pool=WALK, pool_at_end=at_end)
    assert classify_role(outcome) is None and stage_fault([outcome]) is None
    fault = scored_end_fault(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault == starvation_fault(outcome)


def test_the_others_at_a_ceiling_counted_per_namespace_starve_nothing_and_the_evidence_says_how_it_counted():
    # Another build's commands at their whole ceiling: on a kernel that counts the commands' own
    # tasks against theirs, this one's own failure is its own, never a free rerun.
    outcome = RoleOutcome("sandbox", "contestant", 1, "exited", pool_at_end=PER_NAMESPACE)
    assert starvation_fault(outcome) is None and classify_role(outcome) is None
    disk = classify_role(replace(outcome, ending="disk_limit"))
    assert {k: disk.evidence[k] for k in ("accounting", "parent_limit", "node_tasks", "pid_max")} == {
        "accounting": "user_namespace", "parent_limit": "8000000", "node_tasks": "12000", "pid_max": "4194304"}
    starved = starvation_fault(replace(outcome, pool_at_end=replace(PER_NAMESPACE, others_tasks=8_000_000 - 500)))
    assert starved.fault_class == "infra" and "the runner's hard limit 8000000" in starved.reason
    by_uid = classify_role(replace(outcome, pool_at_end=FULL, ending="disk_limit"))
    assert by_uid.evidence["accounting"] == "uid" and "parent_limit" not in by_uid.evidence


def test_a_pool_whose_nodes_tasks_could_not_be_counted_leaves_starvation_not_ruled_out_per_namespace():
    # Per namespace the node's pid_max is a level the others reach the commands through.
    outcome = RoleOutcome("sandbox", "contestant", 1, "exited", pool_at_end=replace(PER_NAMESPACE, node_tasks=None))
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "the node's tasks could not be counted" in fault.reason and fault.evidence["node_tasks"] == "uncounted"
    unread = classify_role(replace(outcome, pool_at_end=replace(PER_NAMESPACE, pid_max=None)))
    assert unread.fault_class == "infra" and unread.evidence["pid_max"] == "unread"


def test_counted_by_uid_a_pool_whose_uid_was_counted_is_judged_by_it_when_the_nodes_count_failed():
    # Under uid the uid's count is the level the ceiling counts against: counted with room left,
    # the end is the contestant's whatever the node's count did; counted full, it was starved.
    for missing in ({"node_tasks": None}, {"pid_max": None}):
        outcome = RoleOutcome("artifact", "contestant", 1, "exited", pool_at_end=replace(ROOM, **missing))
        assert classify_role(outcome) is None and starvation_fault(outcome) is None
        starved = starvation_fault(replace(outcome, pool_at_end=replace(FULL, **missing)))
        assert (starved.fault_class, starved.retryable) == ("infra", True) and "9500" in starved.reason


def test_an_uncounted_pool_neither_changes_a_certain_fault_nor_charges_a_harness_role():
    disk = classify_role(RoleOutcome("artifact", "contestant", -9, "disk_limit", pool_at_end=UNCOUNTED))
    assert (disk.fault_class, disk.evidence["others_tasks"]) == ("contestant", "uncounted")
    pids = classify_role(RoleOutcome("artifact", "contestant", -9, "contestant_pids", pool_at_end=UNCOUNTED))
    assert pids.fault_class == "contestant"
    oom = classify_role(RoleOutcome("artifact", "contestant", -9, "memory_limit", pool_at_end=UNCOUNTED))
    assert oom.fault_class == "contestant"
    assert starvation_fault(RoleOutcome("driver", "harness", 1, "exited", pool_at_end=UNCOUNTED)) is None
    harness = classify_role(RoleOutcome("driver", "harness", 1, "exited", pool_at_end=UNCOUNTED))
    assert "could not be counted" not in harness.reason


def test_the_counts_are_in_the_evidence_of_any_fault_and_a_certain_fault_stands_under_a_full_pool():
    disk = classify_role(RoleOutcome("artifact", "contestant", -9, "disk_limit", detail="5 bytes", pool=WALK,
                                     pool_at_end=FULL))
    assert (disk.fault_class, disk.evidence["others_tasks"], disk.evidence["walk_uid_tasks"]) == (
        "contestant", "9500", "9900")
    pids = classify_role(RoleOutcome("artifact", "contestant", -9, "contestant_pids", pool_at_end=FULL))
    assert pids.fault_class == "contestant"
    driver = classify_role(RoleOutcome("driver", "harness", 1, "exited"))
    assert "others_tasks" not in driver.evidence and "walk_uid_tasks" not in driver.evidence
    # The harness's own end is infra already; it is never taken for a starved contestant's.
    assert starvation_fault(RoleOutcome("driver", "harness", 1, "exited", pool_at_end=FULL)) is None


def test_a_server_stopped_as_planned_under_a_full_pool_is_starved_for_a_stage_that_scores_its_stop():
    # A server not ready when its budget ran out is stopped by the stage, no fault of the role's;
    # a stage that would score it asks whether it was starved.
    stopped = RoleOutcome("artifact", "contestant", -15, "planned_stop", pool_at_end=FULL)
    assert classify_role(stopped) is None
    fault = starvation_fault(stopped)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert starvation_fault(RoleOutcome("artifact", "contestant", -15, "planned_stop", pool_at_end=ROOM)) is None


# --- files held open with no name ---------------------------------------------------------------

#: A process that writes *n* bytes into a file in directory *d* that has no name, an unlinked
#: one or an ``O_TMPFILE``, holds it open, and says so.
_HOLDER = """
import os, sys, time
d, kind, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
if kind == "tmpfile":
    fd = os.open(d, os.O_TMPFILE | os.O_WRONLY, 0o600)
else:
    path = os.path.join(d, "hidden.bin")
    fd = os.open(path, os.O_CREAT | os.O_WRONLY, 0o600)
    os.unlink(path)
for _ in range(n // 65536):
    os.write(fd, b"y" * 65536)
os.fsync(fd)
print("held", flush=True)
time.sleep(60)
"""


@contextlib.contextmanager
def _holding(d: Path, kind: str, n: int):
    """A process holding *n* bytes in a file with no name in *d*; its pid."""
    proc = subprocess.Popen([sys.executable, "-c", _HOLDER, str(d), kind, str(n)], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    try:
        if proc.stdout.readline() != b"held\n":
            err = proc.stderr.read().decode()
            if kind == "tmpfile" and "Operation not supported" in err:
                pytest.skip("this filesystem makes no O_TMPFILE")
            raise AssertionError(err)
        yield proc.pid
    finally:
        proc.kill()
        proc.wait()


def _cgroup(tmp_path: Path, *pids: int) -> Path:
    """A step's cgroup tree that lists *pids*, one of them in a cgroup below it."""
    cg = tmp_path / "cg"
    (cg / "task_0").mkdir(parents=True)
    (cg / "cgroup.procs").write_text("".join(f"{pid}\n" for pid in pids[:1]))
    (cg / "task_0" / "cgroup.procs").write_text("".join(f"{pid}\n" for pid in pids[1:]))
    return cg


@pytest.mark.parametrize("kind", ["unlinked", "tmpfile"])
def test_a_file_a_role_holds_open_with_no_name_counts_against_its_disk_cap(tmp_path, kind):
    """The walk sees named files only. A file the role's processes hold open after unlinking it,
    or made with ``O_TMPFILE``, holds the node's disk as well: the watch reads it through the
    descriptors of the processes in the role's cgroup, once per inode, and a role past its cap so
    is killed, the breach saying how much of it no name held."""
    wd = tmp_path / "wd"
    wd.mkdir()
    with _holding(wd, kind, 3 * MIB) as pid:
        cg = _cgroup(tmp_path, pid)
        assert disk_usage([wd]) < MIB and held_unlinked(cg, [wd]).bytes >= 3 * MIB
        killed = []
        watch = MemoryWatch(enforce=False, clock=lambda: 7.0)
        watch.watch_disk(STEP, [wd], 2 * MIB, lambda: killed.append(True), role="artifact", cgroup_dir=cg)
        watch.sample_disks_once()
    breach = watch.disk_breach(STEP)
    assert killed == [True] and breach is not None and breach.unlinked >= 3 * MIB
    assert breach.used >= breach.unlinked and "with no name" in breach.detail(0)


def test_a_file_held_with_no_name_is_counted_once_and_only_on_the_roles_own_disk(tmp_path):
    """Two processes of the role holding one unlinked file count it once; a file held unlinked on
    another filesystem (``/dev/shm``, memory the role's memory limit already counts) is not the
    role's disk."""
    wd = tmp_path / "wd"
    wd.mkdir()
    shm = Path("/dev/shm") / f"hls-test-{os.getpid()}"
    shm.mkdir()
    try:
        with _holding(wd, "unlinked", 2 * MIB) as here, _holding(shm, "unlinked", 4 * MIB) as there:
            # A second process holding the first one's file: its descriptor, opened through /proc.
            twin = subprocess.Popen([sys.executable, "-c", "import sys, time; f = open(sys.argv[1], 'rb'); "
                                     "print('held', flush=True); time.sleep(60)", f"/proc/{here}/fd/3"],
                                    stdout=subprocess.PIPE)
            try:
                assert twin.stdout.readline() == b"held\n"
                held = held_unlinked(_cgroup(tmp_path, here, twin.pid, there), [wd]).bytes
                # What the filesystem allocated to it, which may be more than was written (XFS
                # preallocates past the end of a file still open, GPFS allocates whole blocks).
                allocated = os.stat(f"/proc/{here}/fd/3").st_blocks * 512
            finally:
                twin.kill()
                twin.wait()
    finally:
        shutil.rmtree(shm)
    assert held == allocated and allocated >= 2 * MIB


def test_a_process_whose_descriptors_cannot_be_listed_is_passed_over(tmp_path):
    """A process of the step this user may not look into (here the node's init) counts nothing,
    and one gone since its cgroup listed it is passed over too."""
    wd = tmp_path / "wd"
    wd.mkdir()
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    assert held_unlinked(_cgroup(tmp_path, 1, gone.pid), [wd]) == memory_watch.Held(0, frozenset())


def test_more_descriptors_than_the_walk_counts_are_past_the_cap(tmp_path, monkeypatch):
    """The descriptors the watch reads are bounded as the walk's entries are: a step whose
    processes hold more is not measured, and counts as past its cap."""
    monkeypatch.setattr(memory_watch, "DISK_WALK_MAX_ENTRIES", 2)
    wd = tmp_path / "wd"
    wd.mkdir()
    with _holding(wd, "unlinked", 0) as pid:
        cg = _cgroup(tmp_path, pid)
        with pytest.raises(memory_watch.Unmeasured, match="descriptors"):
            held_unlinked(cg, [wd])
        killed: list[int] = []
        watch = MemoryWatch(enforce=False)
        watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="artifact", cgroup_dir=cg)
        watch.sample_disks_once()
    breach = watch.disk_breach(STEP)
    assert killed == [1] and breach is not None and breach.unmeasured == "entries"


# --- disk others took ---------------------------------------------------------------------------

GIB = 2**30


def _disk(free_gib: float | None, own_gib: float | None = 40, cap_gib: int = 100) -> DiskAtEnd:
    """A contestant role's disk once its step was gone: *cap_gib* declared, *own_gib* in its
    writable directories, *free_gib* free on their filesystem."""
    return DiskAtEnd(at=2.0, limit_bytes=cap_gib * GIB, own_bytes=None if own_gib is None else int(own_gib * GIB),
                     free_bytes=None if free_gib is None else int(free_gib * GIB))


#: Others hold the node's disk: 2 GiB free of the 60 GiB its cap still left the role.
DISK_FULL = _disk(2)


@pytest.mark.parametrize("code, ending", FAILING_ENDS, ids=["exit-1", "segv", "time-limit", "stopped-not-ready"])
def test_a_contestant_role_that_failed_while_others_held_the_disk_its_cap_left_it_was_starved(code, ending):
    """Writes past the node's free space fail with ENOSPC whatever the role's own cap allows. A
    contestant role that failed while, its step gone, the free space of its filesystem was at most
    what ``POOL_FULL`` of the room its cap still left it leaves, failed for what others wrote: its
    end is infra, retried, never scored, whichever stage scores it."""
    outcome = RoleOutcome("artifact", "contestant", code, ending, disk_at_end=DISK_FULL)
    fault = starvation_fault(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert "disk" in fault.reason and "others" in fault.reason
    assert {k: fault.evidence[k] for k in ("disk_free_bytes", "disk_own_bytes", "disk_cap_bytes")} == {
        "disk_free_bytes": str(2 * GIB), "disk_own_bytes": str(40 * GIB), "disk_cap_bytes": str(100 * GIB)}
    scored = scored_end_fault(outcome)
    assert (scored.fault_class, scored.retryable, scored.reason) == ("infra", True, fault.reason)
    assert (classify_role(outcome) == fault) == (ending == "exited")


@pytest.mark.parametrize("at_end", [_disk(10), _disk(0, own_gib=100), _disk(0, own_gib=None), None],
                         ids=["room-left-by-the-others", "its-own-files-fill-its-cap", "its-tree-unmeasured",
                              "no-disk-cap"])
def test_a_role_others_left_room_or_that_filled_its_own_cap_ended_as_its_own_result(at_end):
    """Free space past what ``POOL_FULL`` of its room leaves let the role write; a role whose own
    files reach its cap had nothing more to write whatever others held; a tree the end's walk
    could not measure is the role's own doing; a role with no disk cap has no room to be starved
    of. Each failure is the contestant's."""
    outcome = RoleOutcome("artifact", "contestant", 1, "exited", disk_at_end=at_end)
    assert starvation_fault(outcome) is None and classify_role(outcome) is None


def test_a_disk_that_could_not_be_read_at_the_end_leaves_starvation_not_ruled_out():
    outcome = RoleOutcome("artifact", "contestant", 1, "exited", disk_at_end=_disk(None))
    fault = classify_role(outcome)
    assert fault is not None and (fault.fault_class, fault.retryable) == ("infra", True)
    assert "could not be read" in fault.reason and fault.evidence["disk_free_bytes"] == "unread"


def test_a_full_disk_neither_changes_a_certain_fault_nor_charges_a_harness_role():
    disk = classify_role(RoleOutcome("artifact", "contestant", -9, "disk_limit", disk_at_end=DISK_FULL))
    assert disk.fault_class == "contestant" and disk.evidence["disk_free_bytes"] == str(2 * GIB)
    pids = classify_role(RoleOutcome("artifact", "contestant", -9, "contestant_pids", disk_at_end=DISK_FULL))
    assert pids.fault_class == "contestant"
    assert starvation_fault(RoleOutcome("driver", "harness", 1, "exited", disk_at_end=DISK_FULL)) is None
    assert classify_role(RoleOutcome("artifact", "contestant", 0, "exited", disk_at_end=DISK_FULL)) is None


# --- a process whose open files the watch cannot list -------------------------------------------


@pytest.fixture
def fake_proc(tmp_path):
    """Makes a /proc for the watch (``_proc``), whose unlistable directories are made removable
    again once the test ends."""
    made: list[Path] = []

    def make(own: list[int], processes: dict[int, list[int]], *, unreadable: set[int],
             states: dict[int, tuple[str, int]] | None = None,
             threads: dict[int, dict[int, tuple[str, bool]]] | None = None) -> Path:
        made.append(_proc(tmp_path, own, processes, unreadable=unreadable, states=states, threads=threads))
        return made[-1]

    yield make
    for proc in made:
        for fd in [*proc.glob("*/fd"), *proc.glob("*/task/*/fd")]:
            fd.chmod(0o700)


#: A process's ``flags`` bit while it exits, from <linux/sched.h>.
PF_EXITING = 0x4


def _stat(pid: int, state: str, flags: int, start: int) -> str:
    """A ``/proc/<pid>/stat`` line: *state* third, *flags* ninth, *start* (its start time) 22nd."""
    return f"{pid} (x y) {state} 1 1 1 0 -1 {flags} 0 0 0 0 0 0 0 0 20 0 1 0 {start} 0 0\n"


def _proc(tmp_path: Path, own: list[int], processes: dict[int, list[int]], *, unreadable: set[int],
          states: dict[int, tuple[str, int]] | None = None,
          threads: dict[int, dict[int, tuple[str, bool]]] | None = None) -> Path:
    """A /proc for the watch: this process's ``NSpid`` *own*, and for each of *processes* its
    ``NSpid``, its ``stat`` (state and flags from *states*, running by default, start time 1000
    plus its pid), an ``exe`` link and a ``fd`` directory, one this user cannot list for those in
    *unreadable*, as the kernel makes it for a non-dumpable process or one with no memory left. A
    process of *threads* also gets those tasks, each with its state and whether its ``fd`` can be
    listed."""
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True, exist_ok=True)
    (proc / "self" / "status").write_text("Name:\trunner\nNSpid:\t" + "\t".join(map(str, own)) + "\n")
    for pid, nspid in processes.items():
        (proc / str(pid) / "fd").mkdir(parents=True)
        (proc / str(pid) / "status").write_text("Name:\tx\nNSpid:\t" + "\t".join(map(str, nspid)) + "\n")
        state, flags = (states or {}).get(pid, ("S", 0))
        (proc / str(pid) / "stat").write_text(_stat(pid, state, flags, 1000 + pid))
        (proc / str(pid) / "exe").symlink_to("/workspace/hidden")
        (proc / str(pid) / "cmdline").write_bytes(b"/workspace/hidden\0--fill\0")
        for tid, (tstate, listable) in (threads or {}).get(pid, {}).items():
            task = proc / str(pid) / "task" / str(tid)
            (task / "fd").mkdir(parents=True)
            (task / "stat").write_text(_stat(tid, tstate, 0, 1000 + pid))
            if not listable:
                (task / "fd").chmod(0)
        if pid in unreadable:
            (proc / str(pid) / "fd").chmod(0)
    return proc


def test_a_contestant_process_whose_open_files_the_watch_cannot_list_is_past_the_cap(tmp_path, fake_proc):
    """A process of the container's own, in a PID namespace below the runner's, whose descriptors
    this user may not list (it made itself non-dumpable, as a process running a program from a
    file it cannot read is) may hold any amount of the node's disk: it counts as past the role's
    cap, a disk breach like any other, and the breach names the process."""
    wd = tmp_path / "wd"
    wd.mkdir()
    proc = fake_proc([100], {700: [700, 15, 3], 701: [701, 16, 4]}, unreadable={700})
    cg = _cgroup(tmp_path, 701, 700)
    # One check finds it unlistable; it is judged only when the same process still is at the next.
    first = held_unlinked(cg, [wd], proc=proc)
    assert first == memory_watch.Held(0, frozenset({(700, 1700)}))
    with pytest.raises(memory_watch.Unmeasured) as caught:
        held_unlinked(cg, [wd], proc=proc, suspects=first.unlistable)
    assert caught.value.why == "hidden"
    sample = memory_watch.role_disk([wd], cg, at=7.0, proc=proc, suspects=first.unlistable)
    assert sample.unmeasured == "hidden"
    detail = sample.detail(8)
    assert "process 700" in detail and "/workspace/hidden" in detail and "700 15 3" in detail
    killed = []
    watch = MemoryWatch(enforce=False, clock=lambda: 7.0, proc=proc)
    watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(True), role="sandbox", cgroup_dir=cg)
    watch.sample_disks_once()
    assert killed == [] and watch.disk_breach(STEP) is None
    watch.sample_disks_once()
    assert killed == [True] and watch.disk_breach(STEP).unmeasured == "hidden"
    fault = classify_role(RoleOutcome("sandbox", "contestant", -9, "disk_limit", detail=detail))
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "process 700" in fault.reason
    # Where the kernel refuses its program's link too, its command line names it.
    (proc / "700" / "exe").unlink()
    with pytest.raises(memory_watch.Unmeasured, match=r"process 700 \(/workspace/hidden by its command line, "
                                                      r"its program link unreadable \(ENOENT\)"):
        held_unlinked(cg, [wd], proc=proc, suspects=first.unlistable)


def test_an_unreadable_harness_process_in_the_runners_own_pid_namespace_changes_nothing(tmp_path):
    """pasta makes itself non-dumpable in the runner's own PID namespace, so its descriptors are
    not listable either: a process at the runner's own depth is the harness's, and is passed
    over."""
    wd = tmp_path / "wd"
    wd.mkdir()
    code = ("import ctypes, time; ctypes.CDLL(None).prctl(4, ctypes.c_ulong(0), ctypes.c_ulong(0), "
            "ctypes.c_ulong(0), ctypes.c_ulong(0)); print('hidden', flush=True); time.sleep(60)")
    pasta = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE)
    try:
        assert pasta.stdout.readline() == b"hidden\n"
        with pytest.raises(PermissionError):
            os.listdir(f"/proc/{pasta.pid}/fd")
        cg = _cgroup(tmp_path, pasta.pid)
        assert held_unlinked(cg, [wd]) == memory_watch.Held(0, frozenset())
        killed: list[int] = []
        watch = MemoryWatch(enforce=False)
        watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="sandbox", cgroup_dir=cg)
        watch.sample_disks_once()
        watch.sample_disks_once()
        assert killed == [] and watch.disk_breach(STEP) is None
    finally:
        pasta.kill()
        pasta.wait()


def test_the_harnesss_own_processes_in_the_container_are_never_judged_and_none_is_while_unknown(tmp_path,
                                                                                                fake_proc):
    """The build sandbox's reaper, PID 1 of the commands' namespace, makes itself non-dumpable: the
    stage names it (``harness``), and it is passed over; while the stage cannot tell it yet, no
    process is judged by its descriptors, and with it told, any other is."""
    wd = tmp_path / "wd"
    wd.mkdir()
    proc = fake_proc([100], {701: [701, 16, 1], 702: [702, 17, 2]}, unreadable={701, 702})
    cg = _cgroup(tmp_path, 701, 702)
    for told, judged in (({701, 702}, False), (None, False), ({701}, True)):
        killed: list[int] = []
        watch = MemoryWatch(enforce=False, proc=proc)
        watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="sandbox", cgroup_dir=cg,
                         harness_pids=lambda told=told: told)
        watch.sample_disks_once()
        watch.sample_disks_once()
        assert (killed == [1]) == judged, told
        breach = watch.disk_breach(STEP)
        assert (breach is not None and "process 702" in breach.detail(1)) == judged


@pytest.mark.parametrize("state, flags", [("Z", 0), ("X", 0), ("R", PF_EXITING)], ids=["zombie", "dead", "exiting"])
def test_a_process_that_has_exited_or_is_exiting_is_never_judged(tmp_path, fake_proc, state, flags):
    """A process that has exited but is not yet reaped has no memory left, so the kernel gives its
    descriptors' directory to root: an agent's CLI forks all the time, and a check that meets one
    of its children so must not take it for one that hid its files. A zombie holds no descriptors,
    and a process in its exit opens none: neither is judged, however many checks meet it."""
    wd = tmp_path / "wd"
    wd.mkdir()
    proc = fake_proc([100], {700: [700, 15, 3]}, unreadable={700}, states={700: (state, flags)})
    cg = _cgroup(tmp_path, 700)
    assert held_unlinked(cg, [wd], proc=proc) == memory_watch.Held(0, frozenset())
    killed: list[int] = []
    watch = MemoryWatch(enforce=False, proc=proc)
    watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="sandbox", cgroup_dir=cg)
    for _ in range(3):
        watch.sample_disks_once()
    assert killed == [] and watch.disk_breach(STEP) is None


@pytest.mark.parametrize("second", ["listable", "gone", "reused"])
def test_a_process_unlistable_on_one_check_only_is_not_hidden(tmp_path, fake_proc, second):
    """Only the same process (its pid and its start time) unlistable on two checks in a row is
    judged: one listable again at the next check, gone, or whose pid another process now holds,
    was a passing state, never a hider."""
    wd = tmp_path / "wd"
    wd.mkdir()
    proc = fake_proc([100], {700: [700, 15, 3]}, unreadable={700})
    cg = _cgroup(tmp_path, 700)
    killed: list[int] = []
    watch = MemoryWatch(enforce=False, proc=proc)
    watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="sandbox", cgroup_dir=cg)
    watch.sample_disks_once()
    if second == "listable":
        (proc / "700" / "fd").chmod(0o700)
    elif second == "gone":
        (cg / "cgroup.procs").write_text("")
    else:
        (proc / "700" / "stat").write_text(_stat(700, "S", 0, 99_999))
    watch.sample_disks_once()
    assert killed == [] and watch.disk_breach(STEP) is None
    if second == "reused":
        # The new process unlistable at the next check too is judged.
        watch.sample_disks_once()
        assert killed == [1] and "process 700" in watch.disk_breach(STEP).detail(1)


@pytest.mark.parametrize("thread_listable", [True, False], ids=["thread-listable", "thread-hidden"])
def test_a_leader_that_exited_while_its_threads_live_is_read_through_a_live_thread(tmp_path, fake_proc,
                                                                                   thread_listable):
    """A process whose first thread exited while others run shows as a zombie, its descriptors'
    directory root's: its descriptors, which its threads share, are read through a thread still
    running. Such a thread that hides them too is judged as any process is."""
    wd = tmp_path / "wd"
    wd.mkdir()
    proc = fake_proc([100], {700: [700, 15, 3]}, unreadable={700}, states={700: ("Z", 0)},
                     threads={700: {700: ("Z", False), 703: ("S", thread_listable)}})
    cg = _cgroup(tmp_path, 700)
    killed: list[int] = []
    watch = MemoryWatch(enforce=False, proc=proc)
    watch.watch_disk(STEP, [wd], 1024 * MIB, lambda: killed.append(1), role="sandbox", cgroup_dir=cg)
    watch.sample_disks_once()
    watch.sample_disks_once()
    assert (killed == [1]) == (not thread_listable)
    if not thread_listable:
        assert "process 700" in watch.disk_breach(STEP).detail(1)



def test_a_process_one_pid_namespace_below_the_runner_is_the_containers_own(tmp_path, fake_proc):
    """A served artifact's or a pip step's processes run one PID namespace below the runner: one
    of them that cannot be listed on two checks in a row is judged as the sandbox's commands two
    below are, and one at the runner's own depth never is."""
    wd = tmp_path / "wd"
    wd.mkdir()
    proc = fake_proc([100], {700: [700, 3], 701: [701]}, unreadable={700, 701})
    cg = _cgroup(tmp_path, 701, 700)
    first = held_unlinked(cg, [wd], proc=proc)
    assert first.unlistable == frozenset({(700, 1700)})
    with pytest.raises(memory_watch.Unmeasured, match="process 700 .*pids 700 3 from"):
        held_unlinked(cg, [wd], proc=proc, suspects=first.unlistable)
