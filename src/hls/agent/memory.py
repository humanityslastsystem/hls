"""The build sandbox's memory, split between the harness and the contestant.

One step holds the whole sandbox. The harness is the step wrapper, the seal (pasta, the
interceptor and the rest of its processes) and apptainer's own processes at the step's base PID
namespace, and the exec server and the spawner, which serves and forks every command, in the
container's (``hls.agent.spawner``). The contestant is every process of the commands' PID
namespace, which the spawner makes nested in the container's, and of anything nested deeper:
the agent's CLI and every command it runs. That namespace's PID 1 is a bare reaper, which only
reaps, so a contestant that ends it loses its own commands and the exec server makes the
namespace anew. Its own few MiB count with the contestant's, and no breach ends it: that would
end every command at once and free nothing a command's own end would not.

A process's ``NSpid`` line in ``/proc/<pid>/status`` lists its pid in each PID namespace from
the reader's inward, so its length says how deep the process runs. The contestant is the
reaper's PID namespace: the reaper and every process at its depth or deeper; the harness is
everything shallower (the step wrapper, the seal's bash and pasta, apptainer's processes, the
exec server and the spawner). How deep the reaper runs is not a constant: the seal's layers
decide it (pasta makes a PID namespace of its own, so under the live seal the exec server and
spawner are two below the step's base and the reaper three, not one and two). So the split
measures the reaper's depth, never assumes it (``find_reaper``): from the pids the exec server's
ready file names (``ReadyPids``; ``exec_protocol.READY_NAME``), the reaper is PID 1 of its namespace, one namespace below a
process whose innermost pid is the spawner's, with its pid at that level the ready line's
``reaper_pid``. Pids repeat across namespaces, so a contestant can fork until one of its
processes holds ``reaper_pid`` in the commands' namespace and make a PID namespace from it, and
even fork one that holds the spawner's pid above it; every such match runs below the real
reaper, since a contestant can make a namespace only inside its own. So the shallowest match is
the reaper, and a deeper one is the contestant's like any other of its processes. Without the
ready line's pids, or with no match, the split does not guess (``Unattributed``), and the OOM
rule treats the window as infra, never the contestant's. There is no fallback depth. Only the
build sandbox splits its step by owner; a served artifact (``hls.stages.deploy_drive``) is not
split, its whole step is watched as one. If a served split is ever added it measures the
reaper's depth the same way.

A ``SplitSample`` holds the step's footprint (``MemorySample.footprint``), its tmpfs pages
(``shmem``: ``/dev/shm`` and the session tmpfs, charged to the step but owned by no process),
the resident memory (``VmRSS``) of the harness's processes and of the contestant's, and the
step's ``oom_kill`` count. Resident memory counts a shared page once per process that maps it,
so the harness figure errs high, and the build rule turns an error that way into infra, never
into blame. A process that ends between the listing and the read is skipped. One that is still
there but cannot be read, or whose status has no ``NSpid`` line, fails the whole sample, since
leaving it out could understate the harness. The status is read as bytes, split at newlines only,
and only the kernel's own ``NSpid``, ``VmRSS`` and ``PPid`` values are decoded
(``hls.substrate.procfs``): its ``Name`` line is whatever the process named itself.

The stage context samples the split of a role started with a harness reserve
(``StageContext.start_container(..., harness_reserve_gib=)``) every five seconds with a
``SplitSampler``. When the role ends, ``split_evidence`` finds its last OOM and the sample taken
before it, which ``hls.substrate.faults.classify_build_end`` judges.

The kernel's OOM killer takes the process with the highest ``oom_score``. The step wrapper starts
every process of a contestant's step with ``oom_score_adj`` 1000, the sandbox's harness
processes too. The exec server lowers its own back to 0, and the spawner and the reaper inherit
that, so in the container a command goes first. The harness's processes at the step's base keep 1000, and
among the processes at 1000 the kernel takes the one with the most memory, which is a harness
process when every command is smaller. So a kernel OOM at the step's limit can end the sandbox
through its harness. The build rule does not rest on whom the kernel picked: it reads the split
before the OOM. Where swap is unconstrained the kernel never OOM-kills at the step's limit, and
the runner enforces it: a breach ends the contestant's largest process
(``kill_largest_contestant``), never a harness process or the reaper, and leaves the sandbox
running. It signals through a pidfd, after checking that the pid is still that process of the
step, so a pid another process took meanwhile is never hit, and ``ContestantKiller`` ends no
other while that one is still exiting. The runner kills the step only when the contestant has
no process left, its processes cannot be told apart, or such kills do not bring the step back
to its limit (``hls.substrate.memory_watch``). A single kill never ends the sandbox, so the
build rule reads only the kill of the step as the runner's OOM; single kills are recorded as
evidence (``process_kills``)."""

from __future__ import annotations

import collections
import logging
import os
import select
import signal
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from hls.substrate.cgroup import pids, read_keyed
from hls.substrate.faults import SplitEvidence
from hls.substrate.memory_watch import CannotEndOne, read_sample
from hls.substrate.pidfd import pidfd_open, pidfd_send_signal
from hls.substrate.procfs import status_fields

log = logging.getLogger(__name__)

#: Seconds between split samples.
SAMPLE_INTERVAL_SEC = 5.0
#: Split samples kept: an hour at the default interval.
SAMPLES_KEPT = 720
#: Times of the runner's single kills kept in the evidence, the latest; all are counted.
PROCESS_KILLS_KEPT = 64
# Tries at the contestant's largest process when each one ends or changes before its signal.
_KILL_TRIES = 3
# Bytes in the kB that /proc/<pid>/status counts in.
_KB = 1024


@dataclass(frozen=True)
class ProcessMemory:
    """One process of the step, as its status file describes it."""

    #: Its pid as the runner sees it.
    pid: int
    #: How many PID namespaces it is in, its ``NSpid`` line's length.
    depth: int
    #: Its pid in its innermost PID namespace.
    innermost: int
    #: Its pid in the namespace one above its innermost (``NSpid`` next-to-last); None at depth 1.
    #: For the reaper, its pid in the spawner's namespace, which the exec server's ready line names
    #: as ``reaper_pid`` (``find_reaper``).
    parent_innermost: int | None
    #: Its parent's pid as the runner sees it (the status file's ``PPid``); None without that line.
    #: The reaper's is the spawner's, which forked it (``find_reaper``).
    ppid: int | None
    #: Resident memory in bytes.
    rss: int


class SplitSample(BaseModel):
    """One reading of the sandbox step's memory, split by owner. Bytes throughout."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: When it was read (POSIX time).
    at: float
    #: The step's footprint, as the memory watch computes it.
    total: int
    #: Pages of tmpfs files charged to the step (``shmem`` in ``memory.stat``).
    tmpfs: int
    #: Resident memory of the harness's processes.
    harness: int
    #: Resident memory of the contestant's processes.
    contestant: int
    #: ``oom_kill`` events in the step's cgroup so far.
    oom_kills: int


def read_process(proc: Path, pid: int) -> ProcessMemory | None:
    """*pid*'s depth, innermost pid, parent and resident memory from ``<proc>/<pid>/status``; None when
    it has ended. Raise OSError when its status is there but cannot be read, and ValueError when
    it has no ``NSpid`` line or is pid 0, which ``cgroup.procs`` lists for a process outside the
    reader's PID namespace: a split without the process could understate the harness."""
    if pid <= 0:
        raise ValueError(f"process {pid} of the step is outside this PID namespace, so its memory cannot be read")
    fields = status_fields(proc, pid, (b"NSpid", b"VmRSS", b"PPid"))
    if fields is None:
        return None
    ns = fields.get(b"NSpid", b"").split()
    rss = int(fields[b"VmRSS"].split()[0]) * _KB if b"VmRSS" in fields else 0
    ppid = int(fields[b"PPid"].split()[0]) if b"PPid" in fields else None
    if not ns:
        raise ValueError(f"{proc / str(pid) / 'status'} has no NSpid line, so its PID namespace depth is unknown")
    return ProcessMemory(pid=pid, depth=len(ns), innermost=int(ns[-1]),
                         parent_innermost=int(ns[-2]) if len(ns) >= 2 else None, ppid=ppid, rss=rss)


def step_processes(cgroup_dir: Path, *, proc: Path = Path("/proc")) -> list[ProcessMemory]:
    """Every process of the step whose cgroup is *cgroup_dir* that has not ended. Raise as
    ``read_process`` does."""
    found = (read_process(proc, pid) for pid in pids(cgroup_dir))
    return [p for p in found if p is not None]


class Unattributed(ValueError):
    """The contestant's PID namespace could not be found, so the step's memory cannot be split by
    owner: no ready line's pids were given, or the reaper they name is not among the step's
    processes (yet, or at all). The split is never guessed from a constant."""


@dataclass(frozen=True)
class ReadyPids:
    """The pids the exec server's ready file names for the serving spawner and reaper."""

    #: The spawner's pid in its own namespace, the container's.
    spawner_pid: int
    #: The reaper's pid in the spawner's namespace.
    reaper_pid: int


def find_reaper(found: Sequence[ProcessMemory], ready: ReadyPids) -> tuple[ProcessMemory, ProcessMemory] | None:
    """The serving spawner and reaper among *found*, the step's processes, by the pids *ready*
    names; None when there is no such pair. The spawner is a process whose innermost pid is
    ``spawner_pid``; the reaper is its child (its ``PPid`` is the spawner's pid as the runner sees
    it), PID 1 of a namespace one below the spawner's, with its pid at the spawner's level (its
    ``NSpid`` next-to-last, ``parent_innermost``) ``reaper_pid``.

    Pids repeat across namespaces, so a contestant can plant a match: fork until one command holds
    ``spawner_pid`` in the commands' namespace and another below it holds ``reaper_pid``, then make
    a PID namespace from that one. A contestant can make a namespace only inside its own, so while
    the real reaper runs every planted match is deeper, and the shallowest match is the real one.

    That leaves the pids of a reaper that has ended, and ordering closes it. A reaper's end ends its
    namespace and every command in it and below it, so until a fresh spawner starts there is no
    contestant process to misplace. The exec server writes the fresh spawner's record into its
    ready file before its pool hands that spawner to any command (``exec_server.SpawnerPool``),
    in place, in blocks allocated at its start, so a disk the contestant filled cannot stop it,
    and ends as infra when the write fails otherwise. The file lives in the control directory,
    which no command can read or write, and the host reads it fresh at every sample and every
    kill (``probe.ready_pids``), keeping nothing between calls; a record caught mid-write fails
    its checksum and names no pair. So no process runs in a fresh namespace before the file names
    it, and every sample or kill taken once one does asks for the fresh pair, whose real reaper
    runs above anything planted: a pair planted from the old pids is never asked for."""
    spawners = {p.pid: p for p in found if p.innermost == ready.spawner_pid}
    matches = [p for p in found
               if p.innermost == 1 and p.parent_innermost == ready.reaper_pid and p.ppid in spawners
               and spawners[p.ppid].depth == p.depth - 1]
    if not matches:
        return None
    reaper = min(matches, key=lambda p: (p.depth, p.pid))
    return spawners[reaper.ppid], reaper


def contestant_depth(found: Sequence[ProcessMemory], ready: ReadyPids | None) -> int | None:
    """The NSpid depth at or below which a process of the step is the contestant's: the serving
    reaper's own depth (``find_reaper``). The seal's layers set that depth, so it is measured,
    never assumed. None when there is no *ready* (no pids to measure from) or no serving reaper
    among *found*; the split is then unattributed, never guessed from a constant."""
    if ready is None:
        return None
    pair = find_reaper(found, ready)
    return pair[1].depth if pair is not None else None


def harness_pids(cgroup_dir: Path, ready: ReadyPids | None, *, proc: Path = Path("/proc")) -> set[int] | None:
    """The host pids of the harness's processes in the build sandbox's commands' PID namespace,
    which the disk watch never judges by their descriptors (``hls.substrate.memory_watch``): the
    serving reaper, which makes itself non-dumpable, found as ``find_reaper`` finds it among the
    processes of the step whose cgroup is *cgroup_dir*. None when they cannot be told: no ready
    record yet, no reaper among the step's processes, or a process that cannot be read."""
    if ready is None:
        return None
    try:
        pair = find_reaper(step_processes(cgroup_dir, proc=proc), ready)
    except (OSError, ValueError):
        return None
    return None if pair is None else {pair[1].pid}


def _unfound(ready: ReadyPids | None) -> str:
    """Why the contestant's namespace was not found with *ready*."""
    if ready is None:
        return "the exec server's ready file names no reaper"
    return f"no reaper (pid {ready.reaper_pid}, a child of spawner {ready.spawner_pid}) is among the step's processes"


def is_reaper_at(p: ProcessMemory, boundary: int) -> bool:
    """Whether *p* is the reaper, PID 1 of the contestant's namespace at the *boundary* depth: a
    breach never ends it, since its death would take every command at once."""
    return p.depth == boundary and p.innermost == 1


def split_sample(cgroup_dir: Path, at: float, *, ready: ReadyPids | None = None,
                 proc: Path = Path("/proc")) -> SplitSample:
    """The step's memory at *at*, split by owner. The contestant is every process at the
    contestant's namespace depth or deeper (``contestant_depth``), the harness everything
    shallower. Raise FileNotFoundError once the step is gone, ValueError when no process of it
    could be read, ``Unattributed`` when there is no *ready* or the reaper it names is not among the
    step's processes (the split would otherwise guess), and as ``read_process`` does."""
    whole = read_sample(cgroup_dir, at)
    shmem = read_keyed(cgroup_dir / "memory.stat").get("shmem", 0)
    found = step_processes(cgroup_dir, proc=proc)
    if not found:
        raise ValueError(f"no process of {cgroup_dir} could be read")
    boundary = contestant_depth(found, ready)
    if boundary is None:
        raise Unattributed(f"{cgroup_dir}'s memory cannot be split by owner: {_unfound(ready)}")
    contestant = sum(p.rss for p in found if p.depth >= boundary)
    harness = sum(p.rss for p in found if p.depth < boundary)
    return SplitSample(at=at, total=whole.footprint, tmpfs=shmem, harness=harness, contestant=contestant,
                       oom_kills=whole.oom_kills)


def largest_contestant(cgroup_dir: Path, *, ready: ReadyPids | None = None,
                       proc: Path = Path("/proc")) -> ProcessMemory | None:
    """The contestant's process with the most resident memory, never the reaper; None when it has
    none. Raise ``Unattributed`` when the contestant's namespace cannot be found
    (``contestant_depth`` None), so the watch ends no process it cannot prove is the contestant's,
    and as ``step_processes`` does."""
    found = step_processes(cgroup_dir, proc=proc)
    boundary = contestant_depth(found, ready)
    if boundary is None:
        raise Unattributed(f"cannot tell which process is the contestant's: {_unfound(ready)}")
    theirs = [p for p in found if p.depth >= boundary and not is_reaper_at(p, boundary)]
    return max(theirs, key=lambda p: (p.rss, p.pid), default=None)


@dataclass(frozen=True)
class Killed:
    """A contestant process ``kill_largest_contestant`` ended."""

    #: The process as it was read just before the signal.
    process: ProcessMemory
    #: A pidfd of it, open, for the caller to close; None where the kernel has no pidfds.
    pidfd: int | None


def kill_largest_contestant(cgroup_dir: Path, *, ready: ReadyPids | None = None,
                            proc: Path = Path("/proc")) -> Killed | None:
    """SIGKILL the contestant's largest process in *cgroup_dir*; None when the contestant has no
    process left. The harness, the reaper and the seal's processes are never the ones ended: when
    the contestant's namespace cannot be found, nothing is (``largest_contestant``). Just before the signal it checks
    that the pid is still in the step and at the
    same place in its PID namespaces, and the signal goes through a pidfd opened before that
    check, so a pid another process took after the choice is never hit. Where the kernel has no
    pidfds, ``os.kill`` follows the same check, which leaves only the moment between the two. A
    process that ended before the signal is skipped for the next largest. Raise CannotEndOne
    when the step's processes cannot be told apart by owner, and RuntimeError when the largest
    ended or changed on every try. Raise CannotEndOne too when the contestant's namespace cannot be
    found."""
    for _ in range(_KILL_TRIES):
        try:
            target = largest_contestant(cgroup_dir, ready=ready, proc=proc)
        except (OSError, ValueError) as e:
            log.warning("the processes of %s cannot be split by owner (%s); no single one is ended", cgroup_dir, e)
            raise CannotEndOne(f"its processes could not be told apart by owner ({e})") from e
        if target is None:
            return None
        try:
            pidfd = pidfd_open(target.pid)
        except ProcessLookupError:
            continue
        try:
            # While the process a pidfd holds is alive its pid is its own, so a check that passes
            # before a signal that reaches it was a check of that process.
            if not _still(target, cgroup_dir, proc):
                _close(pidfd)
                continue
            if pidfd is None:
                os.kill(target.pid, signal.SIGKILL)
            else:
                pidfd_send_signal(pidfd, signal.SIGKILL)
        except ProcessLookupError:
            _close(pidfd)
            continue
        except BaseException:
            _close(pidfd)
            raise
        log.warning("killed contestant process %d (%d bytes resident) to keep the sandbox within its memory",
                    target.pid, target.rss)
        return Killed(process=target, pidfd=pidfd)
    raise RuntimeError(f"the contestant's largest process in {cgroup_dir} ended or changed before each of "
                       f"{_KILL_TRIES} signals")


def _still(target: ProcessMemory, cgroup_dir: Path, proc: Path) -> bool:
    """Whether *target*'s pid is still a process of the step at *target*'s place in its PID
    namespaces. Raise as ``read_process`` does."""
    now = read_process(proc, target.pid)
    return (now is not None and (now.depth, now.innermost) == (target.depth, target.innermost)
            and target.pid in pids(cgroup_dir))


def _close(pidfd: int | None) -> None:
    if pidfd is not None:
        os.close(pidfd)


def _exited(pidfd: int) -> bool:
    """Whether the process *pidfd* holds has exited: its pidfd polls readable then."""
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    return bool(poller.poll(0))


class ContestantKiller:
    """Ends the contestant's largest process of one step at a breach (the memory watch's
    ``end_one``), and says whether the last one it ended is still exiting (its ``exiting``), so
    the watch ends no other meanwhile: a process stuck in its exit, say on a GPFS read, keeps its
    memory charged until it is gone. A victim has exited once its pidfd says so or, where the
    kernel has no pidfds, once its pid no longer holds it with resident memory. Safe to call from
    the watch's thread while another closes it."""

    def __init__(self, cgroup_dir: Path, *, proc: Path = Path("/proc"),
                 ready_source: Callable[[], ReadyPids | None] | None = None) -> None:
        #: The step's cgroup directory.
        self.cgroup_dir = cgroup_dir
        #: Where the process status files are.
        self.proc = proc
        #: Returns the pids the exec server's last ready line names, so the killer ends only the
        #: contestant's processes, never the harness, the reaper or the seal
        #: (``kill_largest_contestant``). While it is None, or returns None (no ready record yet),
        #: the killer ends nothing and says it cannot tell the contestant's processes apart.
        self.ready_source = ready_source
        self._guard = threading.Lock()
        # The last process ended, until it is seen to have exited.
        self._victim: Killed | None = None
        self._closed = False

    def end_one(self) -> bool:
        """End the contestant's largest process; False when it has none, or once closed. Raise
        CannotEndOne when the ready file names no reaper, and as ``kill_largest_contestant`` does:
        the watch then records that the step's processes could not be told apart, never that the
        contestant had none left."""
        with self._guard:
            if self._closed:
                return False
            self._forget()
            ready = self.ready_source() if self.ready_source is not None else None
            if ready is None:
                raise CannotEndOne(f"cannot tell which process is the contestant's: {_unfound(None)}")
            self._victim = kill_largest_contestant(self.cgroup_dir, ready=ready, proc=self.proc)
            return self._victim is not None

    def exiting(self) -> bool:
        """Whether the last process ``end_one`` ended has not exited yet."""
        with self._guard:
            victim = self._victim
            if victim is None:
                return False
            if victim.pidfd is not None:
                gone = _exited(victim.pidfd)
            else:
                try:
                    now = read_process(self.proc, victim.process.pid)
                    gone = not (_still(victim.process, self.cgroup_dir, self.proc) and now is not None
                                and now.rss > 0)
                except (OSError, ValueError):
                    gone = True
            if gone:
                self._forget()
            return not gone

    def close(self) -> None:
        """Let go of the last process ended; ``end_one`` then ends none."""
        with self._guard:
            self._closed = True
            self._forget()

    def _forget(self) -> None:
        if self._victim is not None:
            _close(self._victim.pidfd)
            self._victim = None

def split_evidence(samples: Sequence[SplitSample], *, ended_at: float, final_oom_kills: int,
                   runner_kill_at: float | None, reserve_bytes: int, process_kills: Sequence[float] = (),
                   step_kill_reason: str | None = None) -> SplitEvidence:
    """What the build rule reads of a sandbox that ended at *ended_at* with a harness reserve of
    *reserve_bytes*: its last OOM and the last of *samples* taken before it, with the runner's
    kills of single contestant processes (*process_kills*, the latest ``PROCESS_KILLS_KEPT`` of
    them and their count) and why it killed the whole step, if it did (*step_kill_reason*).

    The kernel's OOM kills are counted in ``memory.events``. The last one is dated by the first
    sample that shows the count's last rise, or by *ended_at* when *final_oom_kills*, the count
    when the sandbox ended, is above every sample's. *runner_kill_at* is the time of the
    runner's kill of the step, where the runner enforces the limit. A single kill is no OOM of
    the sandbox's: it leaves the sandbox running, so it cannot be what ended it. The later of
    the two is the last OOM; a tie goes to the kernel's, which the rule reads more strictly. The
    evidence is the last sample taken strictly before it: the sample that first shows a rise was
    read after the kill freed its victim's memory, and the victim can be a harness process."""
    ordered = sorted(samples, key=lambda s: s.at)
    oom_at: float | None = None
    for prev, cur in zip([None, *ordered], ordered):
        if cur.oom_kills > (prev.oom_kills if prev is not None else 0):
            oom_at = cur.at
    if final_oom_kills > (ordered[-1].oom_kills if ordered else 0):
        oom_at = ended_at
    oom_by: Literal["kernel", "runner"] | None = "kernel" if oom_at is not None else None
    if runner_kill_at is not None and (oom_at is None or runner_kill_at > oom_at):
        oom_at, oom_by = runner_kill_at, "runner"
    kills = {"process_kills": tuple(process_kills[-PROCESS_KILLS_KEPT:]), "process_kill_count": len(process_kills),
             "step_kill_reason": step_kill_reason}
    if oom_at is None:
        return SplitEvidence(reserve_bytes=reserve_bytes, ended_at=ended_at, **kills)
    before = [s for s in ordered if s.at < oom_at]
    if not before:
        return SplitEvidence(reserve_bytes=reserve_bytes, ended_at=ended_at, oom_at=oom_at, oom_by=oom_by, **kills)
    s = before[-1]
    return SplitEvidence(reserve_bytes=reserve_bytes, ended_at=ended_at, oom_at=oom_at, oom_by=oom_by,
                         sample_at=s.at, total_bytes=s.total, tmpfs_bytes=s.tmpfs, harness_bytes=s.harness,
                         contestant_bytes=s.contestant, **kills)


class SplitSampler:
    """Samples one step's split every *interval_sec* in a background thread."""

    def __init__(self, cgroup_dir: Path, *, interval_sec: float = SAMPLE_INTERVAL_SEC,
                 clock: Callable[[], float] = time.time, proc: Path = Path("/proc"),
                 keep: int = SAMPLES_KEPT, ready_source: Callable[[], ReadyPids | None] | None = None) -> None:
        #: The step's cgroup directory.
        self.cgroup_dir = cgroup_dir
        #: Seconds between samples.
        self.interval_sec = interval_sec
        #: Time source for the samples.
        self.clock = clock
        #: Where the process status files are.
        self.proc = proc
        #: Returns the pids the exec server's last ready line names, so the split measures the
        #: contestant's depth from them. A sample while it is None, or while it returns None (no
        #: ready line yet), is unattributed and kept as none: the split never guesses a depth.
        self.ready_source = ready_source
        self._samples: collections.deque[SplitSample] = collections.deque(maxlen=keep)
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = False

    def sample_once(self) -> SplitSample | None:
        """Take one sample; None when the step cannot be read (it is gone), not split, or
        unattributed (its reaper is not up yet, so the split would guess). The first failure is
        logged, so a sandbox left without evidence says why."""
        ready = self.ready_source() if self.ready_source is not None else None
        if ready is None:  # no reaper to measure from: unattributed, keep no sample
            return None
        try:
            sample = split_sample(self.cgroup_dir, self.clock(), ready=ready, proc=self.proc)
        except (OSError, ValueError) as e:
            if not self._failed:
                self._failed = True
                log.warning("the memory of %s could not be split by owner (%s); no sample is kept while that lasts",
                            self.cgroup_dir, e)
            return None
        with self._guard:
            self._samples.append(sample)
        return sample

    def samples(self) -> list[SplitSample]:
        """Every sample kept, oldest first."""
        with self._guard:
            return list(self._samples)

    def start(self) -> None:
        """Sample now, then every interval until ``stop``."""
        def loop() -> None:
            while True:
                try:
                    self.sample_once()
                except Exception:
                    log.exception("split memory sample of %s failed", self.cgroup_dir)
                if self._stop.wait(self.interval_sec):
                    return

        self._thread = threading.Thread(target=loop, name="hls-split-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """End the thread; the samples stay readable."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(self.interval_sec + 1.0)
