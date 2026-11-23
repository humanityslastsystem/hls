"""The build sandbox's memory split by owner, the largest contestant process, and the evidence
the build rule reads."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hls.agent import memory
from hls.agent.memory import (PROCESS_KILLS_KEPT, ContestantKiller, ReadyPids, SplitSample, SplitSampler,
                              kill_largest_contestant, largest_contestant, read_process, split_evidence, split_sample)
from hls.substrate.faults import SplitEvidence
from hls.substrate.memory_watch import CannotEndOne

MIB = 2**20
GIB = 2**30

# The sandbox as the step's processes see it, at base depth 1:
# the step wrapper, the seal and apptainer's starter at the base;
# the exec server and the spawner, which serves and forks every command, in the container's namespace;
# the commands' namespace, the spawner's: its pid 1, a bare reaper, and the contestant's commands.
SANDBOX = {
    100: ([100], 30 * MIB),            # role_exec, then the seal
    101: ([101], 10 * MIB),            # pasta
    102: ([102], 60 * MIB),            # the egress interceptor
    103: ([103], 5 * MIB),             # apptainer's starter
    110: ([110, 1], 2 * MIB),          # the container's init shim
    111: ([111, 7], 40 * MIB),         # the exec server
    112: ([112, 8], 15 * MIB),         # the spawner
    120: ([120, 9, 1], 6 * MIB, 112),  # the reaper, the spawner's child
    121: ([121, 10, 2], 900 * MIB),    # claude
    122: ([122, 11, 3], 3 * GIB),      # the contestant's server
}
HARNESS = (30 + 10 + 60 + 5 + 2 + 40 + 15) * MIB
# The reaper is in the commands' namespace, so its own few MiB count with the contestant's.
CONTESTANT = 6 * MIB + 900 * MIB + 3 * GIB
# The pids the exec server's ready line names: the spawner's in its own namespace (pid 112's
# innermost) and the reaper's in the spawner's (pid 120's NSpid next-to-last). The split measures
# the contestant's depth from them.
SANDBOX_READY = ReadyPids(spawner_pid=8, reaper_pid=9)


def _proc(root: Path, procs: dict[int, tuple]) -> Path:
    """A fake /proc with a status file per process: its NSpid line, VmRSS, and its PPid when the
    entry names one (a third element)."""
    for pid, (nspid, rss, *parent) in procs.items():
        d = root / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        ppid = "".join(f"PPid:\t{x}\n" for x in parent)
        (d / "status").write_text(f"Name:\tx\nPid:\t{pid}\n{ppid}NSpid:\t" + "\t".join(map(str, nspid))
                                  + f"\nVmRSS:\t    {rss // 1024} kB\nThreads:\t1\n")
    return root


def _cgroup(d: Path, pids: list[int], *, current: int = 6 * GIB, shmem: int = 0, oom_kill: int = 0,
            task_pids: list[int] | None = None) -> Path:
    """A fake step cgroup; *task_pids* go into a ``task_0`` child, as Slurm places them."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.current").write_text(f"{current}\n")
    (d / "memory.swap.current").write_text("0\n")
    (d / "memory.stat").write_text(f"anon {current - shmem}\nfile {shmem}\nshmem {shmem}\nslab_reclaimable 0\n")
    (d / "memory.events").write_text(f"oom 0\noom_kill {oom_kill}\n")
    (d / "cgroup.procs").write_text("".join(f"{p}\n" for p in pids))
    if task_pids is not None:
        (d / "task_0").mkdir(exist_ok=True)
        (d / "task_0" / "cgroup.procs").write_text("".join(f"{p}\n" for p in task_pids))
    return d


def test_the_split_follows_each_processs_pid_namespace_depth(tmp_path):
    proc = _proc(tmp_path / "proc", SANDBOX)
    cg = _cgroup(tmp_path / "cg", list(SANDBOX), shmem=GIB, oom_kill=2)
    s = split_sample(cg, 7.0, ready=SANDBOX_READY, proc=proc)
    assert s == SplitSample(at=7.0, total=6 * GIB, tmpfs=GIB, harness=HARNESS, contestant=CONTESTANT, oom_kills=2)


def test_the_base_depth_is_the_steps_own(tmp_path):
    # A site that runs each job in a PID namespace of its own: every depth is one more.
    deeper = {pid: ([pid * 10, *ns], *rest) for pid, (ns, *rest) in SANDBOX.items()}
    proc = _proc(tmp_path / "proc", deeper)
    # Every depth is one more, but the reaper's parent-namespace pid (NSpid[-2]) is still 9.
    s = split_sample(_cgroup(tmp_path / "cg", list(deeper)), 0.0, ready=SANDBOX_READY, proc=proc)
    assert (s.harness, s.contestant) == (HARNESS, CONTESTANT)


def test_a_namespace_the_contestant_makes_stays_the_contestants(tmp_path):
    # pid 1 of a namespace below the spawner's is the contestant's, however it is named.
    procs = {**SANDBOX, 130: ([130, 11, 4, 1], 2 * GIB)}
    proc = _proc(tmp_path / "proc", procs)
    s = split_sample(_cgroup(tmp_path / "cg", list(procs)), 0.0, ready=SANDBOX_READY, proc=proc)
    assert (s.harness, s.contestant) == (HARNESS, CONTESTANT + 2 * GIB)


def test_processes_in_the_steps_child_cgroups_count_and_ended_ones_are_skipped(tmp_path):
    proc = _proc(tmp_path / "proc", SANDBOX)
    top = [100, 101, 102, 103]
    rest = [p for p in SANDBOX if p not in top] + [999]       # 999 ended before its status was read
    s = split_sample(_cgroup(tmp_path / "cg", top, task_pids=rest), 0.0, ready=SANDBOX_READY, proc=proc)
    assert (s.harness, s.contestant) == (HARNESS, CONTESTANT)


def test_a_process_that_cannot_be_placed_fails_the_split_rather_than_drop_out_of_it(tmp_path):
    # Leaving out a live process could understate the harness, which the rule would read as blame.
    proc = _proc(tmp_path / "proc", SANDBOX)
    (proc / "5").mkdir()
    (proc / "5" / "status").write_text("Name:\tx\nVmRSS:\t 4 kB\n")
    with pytest.raises(ValueError, match="no NSpid line"):
        read_process(proc, 5)
    assert read_process(proc, 6) is None
    cg = _cgroup(tmp_path / "cg", [*SANDBOX, 5])
    with pytest.raises(ValueError):
        split_sample(cg, 0.0, ready=SANDBOX_READY, proc=proc)
    assert SplitSampler(cg, proc=proc, ready_source=lambda: SANDBOX_READY).sample_once() is None
    (proc / "5" / "status").unlink()
    (proc / "5" / "status").mkdir()                            # there, but not readable as a file
    with pytest.raises(IsADirectoryError):
        read_process(proc, 5)


def test_a_gone_step_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        split_sample(tmp_path / "gone", 0.0, proc=tmp_path)


def _raw_status(proc: Path, pid: int, status: bytes) -> None:
    (proc / str(pid)).mkdir(parents=True, exist_ok=True)
    (proc / str(pid) / "status").write_bytes(status)


def test_a_process_name_that_is_not_utf8_is_read_like_any_other(tmp_path):
    # A process names itself (its comm), and the kernel passes bytes above 0x7f through raw.
    proc = _proc(tmp_path / "proc", {p: v for p, v in SANDBOX.items() if p != 122})
    _raw_status(proc, 122, b"Name:\thog\xff\xfe\nPid:\t122\nNSpid:\t122\t11\t3\nVmRSS:\t 3145728 kB\n")
    cg = _cgroup(tmp_path / "cg", list(SANDBOX))
    assert read_process(proc, 122) == memory.ProcessMemory(pid=122, depth=3, innermost=3, parent_innermost=11,
                                                           ppid=None, rss=3 * GIB)
    assert (split_sample(cg, 0.0, ready=SANDBOX_READY, proc=proc).contestant,
            largest_contestant(cg, ready=SANDBOX_READY, proc=proc).pid) == (CONTESTANT, 122)
    assert SplitSampler(cg, proc=proc, ready_source=lambda: SANDBOX_READY).sample_once() is not None


def test_a_carriage_return_in_a_name_starts_no_line(tmp_path):
    # A thread-group leader that exited before its threads has no VmRSS line of its own.
    proc = tmp_path / "proc"
    _raw_status(proc, 7, b"Name:\tx\rVmRSS:\t 4194304 kB\nPid:\t7\nNSpid:\t7\t3\t2\n")
    assert read_process(proc, 7) == memory.ProcessMemory(pid=7, depth=3, innermost=2, parent_innermost=3, ppid=None,
                                                         rss=0)


def test_a_process_outside_the_readers_pid_namespace_fails_the_split(tmp_path):
    # cgroup.procs lists such a process as pid 0: it is there, but its memory cannot be read.
    proc = _proc(tmp_path / "proc", SANDBOX)
    cg = _cgroup(tmp_path / "cg", [*SANDBOX, 0])
    with pytest.raises(ValueError, match="outside"):
        split_sample(cg, 0.0, ready=SANDBOX_READY, proc=proc)
    with pytest.raises(CannotEndOne, match="could not be told apart"):
        kill_largest_contestant(cg, ready=SANDBOX_READY, proc=proc)


def test_a_step_with_no_process_to_read_has_no_split(tmp_path):
    cg = _cgroup(tmp_path / "cg", [999])                        # 999 ended before its status was read
    with pytest.raises(ValueError, match="no process"):
        split_sample(cg, 0.0, proc=_proc(tmp_path / "proc", {}))


def test_the_largest_contestant_process_is_never_the_reaper_or_a_harness_process(tmp_path):
    procs = {**SANDBOX, 111: ([111, 7], 50 * GIB), 112: ([112, 8], 45 * GIB), 120: ([120, 9, 1], 40 * GIB, 112)}
    proc = _proc(tmp_path / "proc", procs)
    target = largest_contestant(_cgroup(tmp_path / "cg", list(procs)), ready=SANDBOX_READY, proc=proc)
    assert target is not None and target.pid == 122


def test_a_sandbox_whose_commands_are_gone_has_no_largest_though_its_reaper_counts_as_theirs(tmp_path):
    # Ending the reaper would only end the commands' namespace, which the exec server makes anew.
    no_commands = {p: v for p, v in SANDBOX.items() if p not in (121, 122)}
    proc = _proc(tmp_path / "proc", no_commands)
    cg = _cgroup(tmp_path / "cg", list(no_commands))
    assert largest_contestant(cg, ready=SANDBOX_READY, proc=proc) is None
    assert kill_largest_contestant(cg, ready=SANDBOX_READY, proc=proc) is None
    assert (split_sample(cg, 0.0, ready=SANDBOX_READY, proc=proc).harness,
            split_sample(cg, 0.0, ready=SANDBOX_READY, proc=proc).contestant) == (HARNESS, 6 * MIB)


def test_only_the_reaper_is_spared_in_the_commands_namespace(tmp_path):
    # A pid 1 below it is a namespace the contestant made, and that process is the contestant's.
    procs = {p: v for p, v in SANDBOX.items() if p not in (121, 122)} | {130: ([130, 12, 4, 1], GIB)}
    proc = _proc(tmp_path / "proc", procs)
    target = largest_contestant(_cgroup(tmp_path / "cg", list(procs)), ready=SANDBOX_READY, proc=proc)
    assert target is not None and target.pid == 130


def test_a_sandbox_whose_processes_cannot_be_split_has_none_ended(tmp_path):
    proc = _proc(tmp_path / "proc", SANDBOX)
    (proc / "5").mkdir()
    (proc / "5" / "status").write_text("Name:\tx\n")
    with pytest.raises(CannotEndOne, match="could not be told apart by owner"):
        kill_largest_contestant(_cgroup(tmp_path / "cg", [*SANDBOX, 5]), ready=SANDBOX_READY, proc=proc)


# Two live-shaped layouts, read with the reaper's pid the exec server's ready line names (the
# reaper's pid in the spawner's namespace, its NSpid next-to-last). The split measures the reaper's
# depth from it, so it holds whatever the seal's layers make the depth.
#
# Layout A, no pasta: the exec server and the spawner one namespace below the step's base, the
# reaper two. (spawner_pid 8, reaper_pid 9.)
LAYOUT_A = {
    200: ([200], 20 * MIB),            # the step wrapper and the seal at the base
    201: ([201], 50 * MIB),            # the egress interceptor at the base
    210: ([210, 1], 2 * MIB),          # the container's init shim, base + 1
    211: ([211, 7], 40 * MIB),         # the exec server, base + 1
    212: ([212, 8], 15 * MIB),         # the spawner, base + 1
    220: ([220, 9, 1], 6 * MIB, 212),  # the reaper, base + 2 (pid 9 to the spawner, its parent)
    221: ([221, 10, 2], GIB),          # a command
}
# Layout B, a seal in which pasta makes a PID namespace of its own, so the exec server and the
# spawner are two below the base and the reaper three. (spawner_pid 3, reaper_pid 4.)
LAYOUT_B = {
    300: ([300], 20 * MIB),            # the seal's bash at the base
    301: ([301], 10 * MIB),            # pasta at the base
    310: ([310, 5], 50 * MIB),         # the egress interceptor, base + 1 (pasta's namespace)
    311: ([311, 6], 5 * MIB),          # apptainer's starter, base + 1
    320: ([320, 7, 1], 2 * MIB),       # the container's init shim, base + 2
    321: ([321, 8, 2], 40 * MIB),      # the exec server, base + 2
    322: ([322, 9, 3], 15 * MIB),      # the spawner, base + 2
    330: ([330, 10, 4, 1], 6 * MIB, 322),  # the reaper, base + 3 (pid 4 to the spawner, its parent)
    331: ([331, 11, 5, 2], GIB),       # a command
}
# Each layout's harness total, its contestant total (the reaper counts with it), its ready line,
# and the pid of its one command.
LAYOUTS = {
    "no pasta (base+1/+2)": (LAYOUT_A, (20 + 50 + 2 + 40 + 15) * MIB, 6 * MIB + GIB,
                             ReadyPids(spawner_pid=8, reaper_pid=9), 221),
    "pasta (base+2/+3)": (LAYOUT_B, (20 + 10 + 50 + 5 + 2 + 40 + 15) * MIB, 6 * MIB + GIB,
                          ReadyPids(spawner_pid=3, reaper_pid=4), 331),
}


@pytest.mark.parametrize("layout", list(LAYOUTS), ids=list(LAYOUTS))
def test_the_measured_split_follows_the_reapers_depth_in_either_seal_layout(tmp_path, layout):
    procs, harness, contestant, ready, command = LAYOUTS[layout]
    proc = _proc(tmp_path / "proc", procs)
    cg = _cgroup(tmp_path / "cg", list(procs), shmem=GIB, oom_kill=1)
    s = split_sample(cg, 3.0, ready=ready, proc=proc)
    assert s == SplitSample(at=3.0, total=6 * GIB, tmpfs=GIB, harness=harness, contestant=contestant, oom_kills=1)
    # The exec server and the spawner are the harness however deep the seal puts them.
    target = largest_contestant(cg, ready=ready, proc=proc)
    assert target is not None and target.pid == command


@pytest.mark.parametrize("layout", list(LAYOUTS), ids=list(LAYOUTS))
def test_a_harness_processs_excess_is_never_the_contestants_in_either_layout(tmp_path, layout):
    # The exec server swells past the whole step; the split lays that on the harness, and the
    # watch's kill never reaches it, so the build rule reads the breach as infra, never as blame.
    procs, _, _, ready, command = LAYOUTS[layout]
    server = 321 if layout.startswith("pasta") else 211
    procs = {**procs, server: (procs[server][0], 50 * GIB)}  # a harness process, its parent unnamed
    proc = _proc(tmp_path / "proc", procs)
    cg = _cgroup(tmp_path / "cg", list(procs))
    s = split_sample(cg, 0.0, ready=ready, proc=proc)
    assert s.harness > 50 * GIB and s.contestant == 6 * MIB + GIB
    assert largest_contestant(cg, ready=ready, proc=proc).pid == command


@pytest.mark.parametrize("layout", list(LAYOUTS), ids=list(LAYOUTS))
def test_without_the_named_reaper_the_split_is_unattributed_not_a_guess(tmp_path, layout):
    # The ready line named a reaper that is not among the step's processes (not up yet, or gone):
    # the split must not fall back to a depth it did not measure, or it could blame the harness.
    procs, _, _, ready, _ = LAYOUTS[layout]
    gone = ReadyPids(spawner_pid=ready.spawner_pid, reaper_pid=999)
    proc = _proc(tmp_path / "proc", procs)
    cg = _cgroup(tmp_path / "cg", list(procs))
    with pytest.raises(memory.Unattributed, match="reaper"):
        split_sample(cg, 0.0, ready=gone, proc=proc)
    with pytest.raises(memory.Unattributed, match="cannot tell which process is the contestant's"):
        largest_contestant(cg, ready=gone, proc=proc)
    with pytest.raises(CannotEndOne, match="cannot tell which process is the contestant's"):
        kill_largest_contestant(cg, ready=gone, proc=proc)
    # A sampler told the reaper's pid keeps no sample while it cannot be found, and none at all
    # while the source has no pid yet (the reaper not up); it never guesses the split.
    assert SplitSampler(cg, proc=proc, ready_source=lambda: gone).sample_once() is None
    assert SplitSampler(cg, proc=proc, ready_source=lambda: None).sample_once() is None
    # The killer ends nothing while the reaper is not up, or when it cannot be found, and says it
    # cannot tell the contestant's processes apart: never that the contestant had none left.
    for source in (lambda: None, lambda: gone):
        with pytest.raises(CannotEndOne, match="cannot tell which process is the contestant's"):
            ContestantKiller(cg, proc=proc, ready_source=source).end_one()


# What a contestant can plant in each layout to look like the reaper. Pids repeat across
# namespaces, so a command can fork until one of its processes holds the ready line's reaper_pid
# in the commands' namespace, and make a PID namespace from it: that namespace's PID 1 then has
# the reaper's parent-namespace pid. A second command can fork until it holds the spawner's pid
# there too, and plant the PID 1 one namespace below itself. Every such process runs below the
# real reaper. Each entry: the planted PID 1 alone, the forged spawner with a PID 1 below it.
PLANTED = {
    "no pasta (base+1/+2)": ({241: ([241, 31, 9, 1], 5 * GIB, 221)},
                             {240: ([240, 30, 8], MIB, 221), 241: ([241, 31, 9, 1], 5 * GIB, 240)}),
    "pasta (base+2/+3)": ({341: ([341, 21, 15, 4, 1], 5 * GIB, 331)},
                          {340: ([340, 20, 14, 3], MIB, 331), 341: ([341, 21, 15, 4, 1], 5 * GIB, 340)}),
}


@pytest.mark.parametrize("forged_spawner", [False, True], ids=["planted-pid1", "forged-spawner-and-pid1"])
@pytest.mark.parametrize("layout", list(LAYOUTS), ids=list(LAYOUTS))
def test_a_reaper_the_contestant_planted_below_the_real_one_is_the_contestants(tmp_path, layout, forged_spawner):
    procs, harness, contestant, ready, _ = LAYOUTS[layout]
    planted = PLANTED[layout][forged_spawner]
    # The planted processes come first in the step's listing, so a first match would take them.
    procs = {**planted, **procs}
    proc = _proc(tmp_path / "proc", procs)
    cg = _cgroup(tmp_path / "cg", list(procs))
    s = split_sample(cg, 0.0, ready=ready, proc=proc)
    planted_rss = sum(v[1] for v in planted.values())
    # The real reaper's depth wins: the harness is unchanged and the planted processes are the
    # contestant's, the planted PID 1 too, which the watch can end.
    assert (s.harness, s.contestant) == (harness, contestant + planted_rss)
    assert largest_contestant(cg, ready=ready, proc=proc).pid in planted
    spawner, reaper = memory.find_reaper(memory.step_processes(cg, proc=proc), ready)
    assert (spawner.pid, reaper.pid) == ((212, 220) if layout.startswith("no pasta") else (322, 330))


def test_the_reaper_is_the_spawners_own_child_not_a_pid_1_at_its_depth(tmp_path):
    # A PID 1 of another namespace at the reaper's depth, whose pid one namespace up is the ready
    # line's reaper_pid, and listed first with a lower pid: its parent is no spawner, so it is
    # not the reaper, however its pids line up.
    procs, harness, contestant, ready, _ = LAYOUTS["pasta (base+2/+3)"]
    decoy = {324: ([324, 97, 5], MIB), 325: ([325, 98, 4, 1], MIB, 324)}
    procs = {**decoy, **procs}
    proc = _proc(tmp_path / "proc", procs)
    spawner, reaper = memory.find_reaper(memory.step_processes(_cgroup(tmp_path / "cg", list(procs)), proc=proc),
                                         ready)
    assert (spawner.pid, reaper.pid) == (322, 330)
    # A reaper whose parent is not named cannot be paired with its spawner, so none is found.
    orphan = {**LAYOUTS["pasta (base+2/+3)"][0], 330: ([330, 10, 4, 1], 6 * MIB)}
    proc = _proc(tmp_path / "proc2", orphan)
    assert memory.find_reaper(memory.step_processes(_cgroup(tmp_path / "cg2", list(orphan)), proc=proc),
                              ready) is None


def test_the_sampler_uses_the_ready_source_when_it_has_the_pids(tmp_path):
    # Once the reaper is up, the source returns its pid and the measured split and kill proceed.
    procs, harness, contestant, ready, command = LAYOUTS["pasta (base+2/+3)"]
    proc = _proc(tmp_path / "proc", procs)
    cg = _cgroup(tmp_path / "cg", list(procs))
    sample = SplitSampler(cg, proc=proc, ready_source=lambda: ready).sample_once()
    assert sample is not None and (sample.harness, sample.contestant) == (harness, contestant)


# The ready line of the kill tests' own fixtures: a spawner whose innermost pid is 3, and the
# reaper one namespace below it, pid 4 there.
OWN_READY = ReadyPids(spawner_pid=3, reaper_pid=4)


def _sleepers(n: int) -> list[subprocess.Popen]:
    return [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]) for _ in range(n)]


def test_the_kill_ends_the_largest_contestant_process(tmp_path, monkeypatch):
    small, big, harness = _sleepers(3)
    by_pid, real_kill = [], os.kill
    monkeypatch.setattr(os, "kill", lambda pid, sig: by_pid.append(pid) or real_kill(pid, sig))
    try:
        procs = {harness.pid: ([harness.pid], 8 * GIB), 49: ([49, 3], MIB),             # 49: the spawner
                 50: ([50, 4, 1], 8 * GIB, 49),                                         # 50: the reaper
                 small.pid: ([small.pid, 5, 2], GIB), big.pid: ([big.pid, 6, 3], 2 * GIB)}
        proc = _proc(tmp_path / "proc", procs)
        killed = kill_largest_contestant(_cgroup(tmp_path / "cg", list(procs)), ready=OWN_READY, proc=proc)
        assert killed is not None and killed.process.pid == big.pid and killed.pidfd is not None
        os.close(killed.pidfd)
        assert by_pid == []                     # the signal went through the pidfd, never by pid
        assert big.wait(10) == -9
        assert small.poll() is None and harness.poll() is None
    finally:
        for p in (small, big, harness):
            p.kill()
            p.wait()


@pytest.mark.parametrize("taken_by", ["a harness process", "a process of another step"])
def test_a_pid_another_process_took_before_the_kill_is_never_signalled(tmp_path, monkeypatch, taken_by):
    [sleeper] = _sleepers(1)
    try:
        procs = {1000: ([1000], MIB), 59: ([59, 3], MIB), 60: ([60, 4, 1], MIB, 59),   # the spawner, the reaper
                 sleeper.pid: ([sleeper.pid, 5, 2], GIB)}
        proc = _proc(tmp_path / "proc", procs)
        cg = _cgroup(tmp_path / "cg", list(procs))
        real = memory.largest_contestant

        def largest(cgroup_dir, *, ready=None, proc):
            found = real(cgroup_dir, ready=ready, proc=proc)
            if found is not None:
                # The target ends, and its pid goes to another process before the signal.
                if taken_by == "a harness process":
                    _proc(proc, {sleeper.pid: ([sleeper.pid], GIB)})
                else:
                    (cg / "cgroup.procs").write_text("1000\n59\n60\n")  # the reaper stays, the command is gone
            return found

        monkeypatch.setattr(memory, "largest_contestant", largest)
        assert kill_largest_contestant(cg, ready=OWN_READY, proc=proc) is None
        time.sleep(0.2)
        assert sleeper.poll() is None
    finally:
        sleeper.kill()
        sleeper.wait()


def test_a_target_that_ended_first_is_skipped_for_the_next_largest(tmp_path, monkeypatch):
    [alive] = _sleepers(1)
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    try:
        procs = {1000: ([1000], MIB), 69: ([69, 3], MIB), 70: ([70, 4, 1], MIB, 69),   # the spawner, the reaper
                 gone.pid: ([gone.pid, 5, 2], 4 * GIB), alive.pid: ([alive.pid, 6, 3], GIB)}
        proc = _proc(tmp_path / "proc", procs)
        cg = _cgroup(tmp_path / "cg", list(procs))
        calls = []
        real = memory.largest_contestant

        def largest(cgroup_dir, *, ready=None, proc):
            found = real(cgroup_dir, ready=ready, proc=proc)
            calls.append(found.pid)
            if found.pid == gone.pid:
                (cg / "cgroup.procs").write_text(f"1000\n69\n70\n{alive.pid}\n")   # the reaper stays up
            return found

        monkeypatch.setattr(memory, "largest_contestant", largest)
        killed = kill_largest_contestant(cg, ready=OWN_READY, proc=proc)
        assert killed is not None and killed.process.pid == alive.pid
        os.close(killed.pidfd)
        assert calls == [gone.pid, alive.pid] and alive.wait(10) == -9
    finally:
        alive.kill()
        alive.wait()


def test_the_killer_ends_no_new_process_while_its_last_victim_is_still_exiting(tmp_path, monkeypatch):
    small, big = _sleepers(2)
    try:
        procs = {1000: ([1000], MIB), 79: ([79, 3], MIB), 80: ([80, 4, 1], MIB, 79),   # the spawner, the reaper
                 small.pid: ([small.pid, 5, 2], GIB), big.pid: ([big.pid, 6, 3], 2 * GIB)}
        proc = _proc(tmp_path / "proc", procs)
        cg = _cgroup(tmp_path / "cg", list(procs))
        killer = ContestantKiller(cg, proc=proc, ready_source=lambda: OWN_READY)
        assert killer.exiting() is False
        assert killer.end_one() is True and big.wait(10) == -9
        # The victim is gone once its pidfd says it exited, even before it is reaped.
        assert killer.exiting() is False
        (cg / "cgroup.procs").write_text(f"1000\n79\n80\n{small.pid}\n")   # the reaper stays up
        monkeypatch.setattr(memory, "_exited", lambda pidfd: False)    # stuck in its exit, say on a GPFS read
        assert killer.end_one() is True and small.wait(10) == -9
        assert killer.exiting() is True
        monkeypatch.undo()
        assert killer.exiting() is False and killer.exiting() is False
        (cg / "cgroup.procs").write_text("1000\n79\n80\n")             # only the harness and the reaper are left
        assert killer.end_one() is False
        killer.close()
        assert killer.end_one() is False
    finally:
        for p in (small, big):
            p.kill()
            p.wait()


def test_the_killer_says_when_the_processes_cannot_be_told_apart(tmp_path):
    proc = _proc(tmp_path / "proc", SANDBOX)
    with pytest.raises(CannotEndOne, match="could not be told apart"):
        ContestantKiller(_cgroup(tmp_path / "cg", [*SANDBOX, 0]), proc=proc,
                         ready_source=lambda: SANDBOX_READY).end_one()


def test_where_the_kernel_has_no_pidfds_the_kill_and_the_exit_are_told_from_proc(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "pidfd_open", lambda pid: None)
    [victim] = _sleepers(1)
    try:
        procs = {1000: ([1000], MIB), 89: ([89, 3], MIB), 90: ([90, 4, 1], MIB, 89),   # the spawner, the reaper
                 victim.pid: ([victim.pid, 5, 2], GIB)}
        proc = _proc(tmp_path / "proc", procs)
        killer = ContestantKiller(_cgroup(tmp_path / "cg", list(procs)), proc=proc, ready_source=lambda: OWN_READY)
        assert killer.end_one() is True and victim.wait(10) == -9
        # The fake /proc still shows it resident: it reads as exiting until its memory is gone.
        assert killer.exiting() is True
        _raw_status(proc, victim.pid, f"Name:\tx\nState:\tZ\nNSpid:\t{victim.pid}\t5\t2\n".encode())
        assert killer.exiting() is False
    finally:
        victim.kill()
        victim.wait()


def _s(at: float, oom_kills: int, harness: int = MIB) -> SplitSample:
    return SplitSample(at=at, total=2 * GIB, tmpfs=GIB // 2, harness=harness, contestant=GIB, oom_kills=oom_kills)


def _evidence(samples, *, ended_at, final_oom_kills, runner_kill_at=None) -> SplitEvidence:
    return split_evidence(samples, ended_at=ended_at, final_oom_kills=final_oom_kills, runner_kill_at=runner_kill_at,
                          reserve_bytes=4 * GIB)


def _found(oom_at, oom_by, sample: SplitSample | None, ended_at) -> SplitEvidence:
    figures = {} if sample is None else {
        "sample_at": sample.at, "total_bytes": sample.total, "tmpfs_bytes": sample.tmpfs,
        "harness_bytes": sample.harness, "contestant_bytes": sample.contestant}
    return SplitEvidence(reserve_bytes=4 * GIB, ended_at=ended_at, oom_at=oom_at, oom_by=oom_by, **figures)


def test_no_oom_is_found_when_none_happened():
    assert _evidence([_s(0, 0), _s(5, 0)], ended_at=9.0, final_oom_kills=0) == _found(None, None, None, 9.0)


def test_the_kernels_oom_is_dated_by_the_sample_that_first_shows_it():
    samples = [_s(0, 0), _s(5, 0), _s(10, 1), _s(15, 1)]
    assert _evidence(samples, ended_at=16.0, final_oom_kills=1) == _found(10.0, "kernel", samples[1], 16.0)


def test_the_last_of_several_ooms_is_the_one_found():
    samples = [_s(0, 0), _s(5, 1), _s(10, 1), _s(15, 3)]
    assert _evidence(list(reversed(samples)), ended_at=16.0, final_oom_kills=3) == _found(15.0, "kernel", samples[2],
                                                                                          16.0)


def test_an_oom_after_the_last_sample_is_dated_by_the_end():
    samples = [_s(0, 0), _s(5, 0)]
    assert _evidence(samples, ended_at=8.0, final_oom_kills=1) == _found(8.0, "kernel", samples[1], 8.0)


def test_the_runners_own_kill_is_an_oom_and_the_later_one_is_found():
    samples = [_s(0, 0), _s(5, 0), _s(10, 0)]
    assert _evidence(samples, ended_at=12.0, final_oom_kills=0, runner_kill_at=7.0) == _found(7.0, "runner",
                                                                                              samples[1], 12.0)
    samples = [_s(0, 0), _s(5, 1), _s(10, 1)]
    assert _evidence(samples, ended_at=12.0, final_oom_kills=1, runner_kill_at=2.0) == _found(5.0, "kernel",
                                                                                              samples[0], 12.0)
    # A tie is the kernel's, which the rule reads more strictly.
    assert _evidence(samples, ended_at=12.0, final_oom_kills=1, runner_kill_at=5.0).oom_by == "kernel"


def test_the_evidence_is_a_sample_from_before_the_oom_never_the_one_read_after_it():
    samples = [_s(0, 0, harness=GIB), _s(7, 0, harness=MIB)]
    found = _evidence(samples, ended_at=12.0, final_oom_kills=0, runner_kill_at=7.0)
    assert (found.sample_at, found.harness_bytes) == (0.0, GIB)


def test_an_oom_before_the_first_sample_has_no_sample_before_it():
    assert _evidence([_s(3, 1), _s(8, 1)], ended_at=9.0, final_oom_kills=1) == _found(3.0, "kernel", None, 9.0)
    assert _evidence([], ended_at=9.0, final_oom_kills=1) == _found(9.0, "kernel", None, 9.0)


def test_the_evidence_records_the_runners_single_kills_and_why_it_killed_the_step():
    samples = [_s(0, 0), _s(5, 0), _s(10, 0)]
    found = split_evidence(samples, ended_at=12.0, final_oom_kills=0, runner_kill_at=11.0, reserve_bytes=4 * GIB,
                           process_kills=[1.0, 6.0], step_kill_reason="no process of it was left to end")
    assert (found.process_kills, found.process_kill_count, found.step_kill_reason) == (
        (1.0, 6.0), 2, "no process of it was left to end")
    assert (found.oom_at, found.oom_by, found.sample_at) == (11.0, "runner", 10.0)
    none = _evidence(samples, ended_at=12.0, final_oom_kills=0)
    assert (none.process_kills, none.process_kill_count, none.step_kill_reason) == ((), 0, None)


def test_the_evidence_keeps_the_last_single_kills_and_counts_them_all():
    kills = [float(t) for t in range(100)]
    found = split_evidence([], ended_at=200.0, final_oom_kills=0, runner_kill_at=None, reserve_bytes=GIB,
                           process_kills=kills)
    assert PROCESS_KILLS_KEPT == 64
    assert (found.process_kills, found.process_kill_count) == (tuple(kills[-64:]), 100)
    assert found.oom_at is None


def test_the_samplers_first_failure_is_logged_once(tmp_path, caplog):
    proc = _proc(tmp_path / "proc", SANDBOX)
    cg = _cgroup(tmp_path / "cg", [*SANDBOX, 0])
    sampler = SplitSampler(cg, proc=proc, ready_source=lambda: SANDBOX_READY)
    with caplog.at_level(logging.WARNING, logger="hls.agent.memory"):
        assert sampler.sample_once() is None and sampler.sample_once() is None
    [record] = [r for r in caplog.records if r.name == "hls.agent.memory"]
    assert str(cg) in record.getMessage() and "outside this PID namespace" in record.getMessage()


def test_the_sampler_keeps_samples_in_order_and_skips_a_gone_step(tmp_path):
    proc = _proc(tmp_path / "proc", SANDBOX)
    cg = _cgroup(tmp_path / "cg", list(SANDBOX))
    ticks = iter([1.0, 2.0, 3.0])
    sampler = SplitSampler(cg, proc=proc, clock=lambda: next(ticks), keep=2, ready_source=lambda: SANDBOX_READY)
    assert sampler.sample_once().at == 1.0
    sampler.sample_once()
    sampler.sample_once()
    assert [s.at for s in sampler.samples()] == [2.0, 3.0]
    gone = SplitSampler(tmp_path / "gone", proc=proc, ready_source=lambda: SANDBOX_READY)
    assert gone.sample_once() is None and gone.samples() == []


def test_the_sampler_thread_samples_at_once_and_then_every_interval(tmp_path):
    proc = _proc(tmp_path / "proc", SANDBOX)
    cg = _cgroup(tmp_path / "cg", list(SANDBOX))
    sampler = SplitSampler(cg, proc=proc, interval_sec=60.0, ready_source=lambda: SANDBOX_READY)
    sampler.start()
    try:
        deadline = time.time() + 10
        while not sampler.samples() and time.time() < deadline:
            time.sleep(0.02)
        assert len(sampler.samples()) == 1 and sampler.samples()[0].contestant == CONTESTANT
    finally:
        sampler.stop()
    often = SplitSampler(cg, proc=proc, interval_sec=0.05, ready_source=lambda: SANDBOX_READY)
    often.start()
    try:
        deadline = time.time() + 10
        while len(often.samples()) < 3 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        often.stop()
    taken = often.samples()
    assert len(taken) >= 3 and [s.at for s in taken] == sorted(s.at for s in taken)


def test_the_harness_processes_the_disk_watch_never_judges_are_the_serving_reaper(tmp_path):
    """The disk watch counts a process of the container's own whose descriptors it cannot list
    as past the sandbox's cap; the reaper makes itself non-dumpable, so the stage names it, found as
    the memory split finds it, the shallowest, never a PID 1 planted below it. None while there is
    no ready record or no reaper, so the watch judges no process then."""
    procs, _, _, ready, _ = LAYOUTS["pasta (base+2/+3)"]
    planted = PLANTED["pasta (base+2/+3)"][True]
    proc = _proc(tmp_path / "proc", {**planted, **procs})
    cg = _cgroup(tmp_path / "cg", [*planted, *procs])
    assert memory.harness_pids(cg, ready, proc=proc) == {330}
    assert memory.harness_pids(cg, None, proc=proc) is None
    orphan = {**procs}
    del orphan[322]
    proc = _proc(tmp_path / "proc2", orphan)
    assert memory.harness_pids(_cgroup(tmp_path / "cg2", list(orphan)), ready, proc=proc) is None
