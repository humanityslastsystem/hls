"""The runner's watch over its role steps' memory.

Every second the watch reads each watched step's memory accounting. Where the job's swap
is unlimited (Slurm's ConstrainSwapSpace=no), a step that outgrows its declared
memory swaps instead of being killed, so the watch enforces the limit itself: once a step's
footprint passes its declared limit, it kills every process in the step. The footprint is
what would count against the limit if swap were off: ``memory.current`` plus
``memory.swap.current``, less the memory the kernel reclaims before it kills anything: page
cache (``file`` minus ``shmem`` in ``memory.stat``) and reclaimable slab
(``slab_reclaimable``), less the swap cache (``swapcached``, pages in memory that also hold a
place in swap, so counted in both), and never below zero. A kernel whose ``memory.stat`` has
no ``swapcached`` subtracts nothing for it. Page cache alone never trips it. Where swap is off
the kernel enforces the limit and the watch only records samples.

A step that must outlive a breach, the build sandbox, is watched with ``end_one``: at a breach
the watch calls it to end one process of the step instead, records the kill (``kills``,
``on_single_kill``) and keeps sampling. After each such kill it waits ``repeat_grace_sec``, and
while the process it ended is still exiting (``exiting``, which reads as still exiting when it
raises) it ends no other, as the kernel's OOM killer picks no new victim while one is exiting,
but only for ``END_ONE_STILL_OVER_SEC`` after that kill, so a victim stuck in its exit cannot
leave the step over its limit with no action.
Single kills must bring the step back to its limit. Since the step was last sampled at or under
its limit, the watch kills it as it kills any other at the first breach after
``END_ONE_MAX_KILLS`` single kills, or once it is still over ``END_ONE_STILL_OVER_SEC`` after
the first of them; so a contestant that replaces each process as it is ended loses its step
after three graces, 15 s, unless a victim is slow to exit. It
also kills the step when ``end_one`` finds no process to end or cannot tell which to end
(``CannotEndOne``), and records why (``kill_reason``).

Steps are keyed by their step id, "<job>.<step>", which is unique for the life of the job;
the role name is kept for log lines only, since a later attempt may reuse it.

Each sample keeps the step's ``oom_kill`` and ``oom`` counts from its ``memory.events``. They
are evidence for the fault rules only. A step marked OUT_OF_MEMORY may still be running, so
neither the counts nor the step state say whether a role is alive.

A kill in a step does not say whose limit was reached. ``oom_kill`` counts a kill by any OOM
killer: at the step's own limit, at the job's, at a limit above the job, or the node's when
the node itself runs out of memory. The kernel picks its victim among every process under the
limit that was reached, and a contestant process, which runs with ``oom_score_adj`` 1000, is
picked first. Each kill leaves the step OUT_OF_MEMORY. What tells them apart is ``oom``,
counted on the cgroup whose limit was reached and on every cgroup above it: the step's own
``oom`` rises only for a limit of the step. ``memory.events.local`` counts a cgroup's own limit
only. So the watch also reads the job cgroup, which outlives every step: its own count, the
count of the limits reached below it (``memory.events`` less ``memory.events.local``), and each
step cgroup's count. A step cgroup is made for its step, so its counts start at 0.

The watch also caps the node-local disk a contestant role writes (``watch_disk``). A site need
not enforce a quota on the node's scratch, so every ``disk_interval_sec`` the watch
adds up what the role's writable directories hold, as ``du`` does: the blocks allocated to each
file, once per inode, without following a link. Once that passes the role's declared disk, it
kills the role's step, and ``disk_breach`` says so. The walk runs in a thread of its own, so a
tree of many files never delays a memory sample. It caps blocks, not inodes, and a writer can
pass the cap by at most what it writes in one interval and one walk. A tree deeper than
``DISK_WALK_MAX_DEPTH``, of more than ``DISK_WALK_MAX_ENTRIES`` entries, or where the walk is
refused a permission, which only the role's own processes can take away, is not measured: it
counts as past the cap. A walk sees named files only, so with the step's cgroup each walk also
counts the files the step's processes hold open with no name left on the filesystems of those
directories, ones they unlinked or made with ``O_TMPFILE`` (``held_unlinked``): each once per
inode, its blocks read through the descriptor, bounded as the walk is. The kernel hides the
descriptors of a process that made itself non-dumpable unless its namespace maps root to this
user. One of the container's own, in a PID namespace below the runner's, whose descriptors this
user may not list may hold any amount of the node's disk unseen: it counts as past the cap
(``hidden``), and the breach names it. The kernel gives root the descriptors' directory of a
process with no memory left too, one that has exited and is not yet reaped or is in its exit,
which an agent's forks leave all the time: a process whose state says it has exited or is exiting
(``stat``'s state Z or X, or ``PF_EXITING`` in its flags) holds no descriptors and opens none,
and is never judged; one whose first thread exited while others run is read through a running
thread, which shares its descriptors. And a process is judged only when the same one (its pid and
its start time) is unlistable on two checks in a row, so no passing state of a process is taken
for a hider. The build sandbox's commands cannot call
``prctl(PR_SET_DUMPABLE, 0)`` (``hls.agent.spawner``), but one that runs a program from a file
it cannot read is non-dumpable all the same. The harness's own are never judged so: those at the
runner's own depth (pasta makes itself non-dumpable) and those the stage names (``harness_pids``, the
sandbox's reaper); while the stage cannot tell its own yet, no process is. Each walk also reads
how often the step's pids limit was reached, dating each rise of the count by the walk that first
read it (``pids_hits``), so the fault rules can bound its age; and for a step with a process
ceiling, the tasks of the uid on the node and of the step itself (``pool``,
``hls.substrate.nproc``), the last sample and the most of each over the step's life
(``pool_peak``).

When a contestant role is reported, its step gone and the files it held with no name freed, the
watch reads what its writable directories hold and the free space of their filesystem
(``disk_at_end``): a role that failed while others held what its cap still left it was starved of
disk (``disk_starvation``), as the PID pool's rule judges a starved pool."""

from __future__ import annotations

import errno
import logging
import math
import os
import stat
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, NamedTuple

from hls.substrate.cgroup import pids, pids_limit_hits, read_int, read_keyed
from hls.substrate.nproc import POOL_FULL, PoolPeak, PoolSample, peak_with, step_tasks, uid_tasks

log = logging.getLogger(__name__)

#: Seconds a step watched with ``end_one`` may stay over its limit after the first kill of one of
#: its processes since it was last within its limit, before the watch kills the whole step.
END_ONE_STILL_OVER_SEC = 30.0
#: Kills of single processes since the step was last within its limit after which its next
#: breach kills the whole step.
END_ONE_MAX_KILLS = 3


class CannotEndOne(Exception):
    """Raised by an ``end_one`` that cannot tell which process of the step to end. The watch
    kills the step whole and records the message as its reason."""


@dataclass(frozen=True)
class MemorySample:
    """One reading of a step's memory accounting, in bytes."""

    #: When it was read.
    at: float
    #: ``memory.current``.
    current: int
    #: ``memory.swap.current``; 0 when the file is absent.
    swap_current: int
    #: Page cache the kernel can drop: ``memory.stat`` file minus shmem.
    reclaimable: int
    #: ``oom_kill`` from ``memory.events``: processes of the step killed by any OOM killer.
    oom_kills: int
    #: Slab the kernel can reclaim: ``slab_reclaimable`` from ``memory.stat``.
    slab_reclaimable: int = 0
    #: ``oom`` from ``memory.events``: times the step's own limit was reached; None when the
    #: file has no such count.
    ooms: int | None = None
    #: Swap cache, counted in both ``memory.current`` and ``memory.swap.current``: ``swapcached``
    #: from ``memory.stat``; 0 where the kernel does not report it.
    swap_cached: int = 0

    @property
    def footprint(self) -> int:
        """What would count against the limit with swap off, never below zero."""
        return max(0, self.current + self.swap_current - self.reclaimable - self.slab_reclaimable
                   - self.swap_cached)


def read_sample(cgroup_dir: Path, at: float) -> MemorySample:
    """Read *cgroup_dir*'s accounting; FileNotFoundError once the step is gone."""
    stat = read_keyed(cgroup_dir / "memory.stat")
    events = read_keyed(cgroup_dir / "memory.events")
    swap_file = cgroup_dir / "memory.swap.current"
    return MemorySample(
        at=at,
        current=read_int(cgroup_dir / "memory.current"),
        swap_current=read_int(swap_file) if swap_file.exists() else 0,
        reclaimable=max(0, stat.get("file", 0) - stat.get("shmem", 0)),
        oom_kills=events.get("oom_kill", 0),
        slab_reclaimable=stat.get("slab_reclaimable", 0),
        ooms=events.get("oom"),
        swap_cached=stat.get("swapcached", 0),
    )


def read_peak(cgroup_dir: Path) -> int | None:
    """*cgroup_dir*'s ``memory.peak``, the most memory it has held, page cache included, where the
    kernel keeps the file; None where it does not, or it cannot be read."""
    try:
        return read_int(cgroup_dir / "memory.peak")
    except (OSError, ValueError, IndexError):
        return None


@dataclass(frozen=True)
class JobOoms:
    """Whose memory limits the job cgroup records as reached, at one moment."""

    #: ``oom`` in the job's ``memory.events.local``: times the job's own limit was reached.
    own: int
    #: ``oom`` in the job's ``memory.events`` less ``own``: times the limit of a cgroup below
    #: the job, such as a step's own, was reached. It stays after the step is gone.
    below: int
    #: ``oom`` in the ``memory.events`` of each step cgroup there now (``step_batch``,
    #: ``step_extern``, ``step_<n>``), by directory name.
    steps: Mapping[str, int]


#: Directories below a watched path the disk walk descends into at most. A role's writable tree
#: that goes deeper is not measured: it counts as past the role's cap (``DiskSample.unmeasured``),
#: and the walk keeps as many directories open as this, however deep the tree.
DISK_WALK_MAX_DEPTH = 256
#: Entries of a role's writable directories the disk walk looks at at most, so its time and memory
#: stay bounded, and as many descriptors of the step's processes (``held_unlinked``). A tree of
#: more, or a step holding more, is not measured, and counts as past the cap,
#: which the briefs state for the sandbox and the graded server. It leaves room for the
#: thousands of files an engine cache holds many hundred times over.
DISK_WALK_MAX_ENTRIES = 2_000_000
#: Where the processes' descriptors and status are read.
PROC = Path("/proc")
#: Characters of a hidden process's command line its breach records, when its program's link
#: cannot be read (``hidden``).
HIDDEN_ARGV0_CHARS = 256
#: The flag of a task on its way out, from <linux/sched.h>, in ``stat``'s ninth field.
PF_EXITING = 0x4
#: The states ``stat`` gives a process that has exited: a zombie not yet reaped, or dead.
EXITED_STATES = frozenset("ZXx")
# How the walk opens a directory: never through a link, and not into a child process.
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# What the walk needs of a directory it lists: its owner's read and search.
_LIST = stat.S_IRUSR | stat.S_IXUSR
#: Why a walk did not measure a role's writable tree: a directory deeper than
#: ``DISK_WALK_MAX_DEPTH``, more entries or descriptors than ``DISK_WALK_MAX_ENTRIES``, a
#: permission it was refused there (EACCES or EPERM) after it gave the owner back its own, which
#: only the role's own processes can take away again, or a process of the container's own whose
#: descriptors it may not list (``hidden``).
Unmeasurable = Literal["depth", "entries", "permission", "hidden"]


@dataclass(frozen=True)
class DiskSample:
    """One reading of a role's writable directories."""

    #: When the walk that read it began.
    at: float
    #: Bytes allocated to everything under them (``disk_usage``) and to the files the step's
    #: processes hold there with no name (``unlinked``); for a walk that stopped short
    #: (``unmeasured``), those it counted until then.
    used: int
    #: Why the walk did not measure the whole tree, which counts as past the cap; None when it
    #: did.
    unmeasured: Unmeasurable | None = None
    #: Of ``used``, the bytes of the files the step's processes held open with no name left
    #: (``held_unlinked``).
    unlinked: int = 0
    #: A contestant process whose open files the watch cannot list counts as past the cap, as the
    #: files its processes hold with no name count in ``used``.
    #: For a sample past the cap because a process hid its descriptors (``hidden``), which: its pid,
    #: its program as far as it can be read and its pids in each PID namespace; None otherwise.
    hidden: str | None = None
    #: The processes of the container's own this check found unlistable, by pid and start time,
    #: which the next check judges if it finds them so again (``held_unlinked``).
    unlistable: frozenset[tuple[int, int]] = frozenset()

    def detail(self, limit_gib: int) -> str:
        """How the role passed its cap of *limit_gib* GiB, for its record: the bytes over it, or
        what kept the walk from measuring its tree, with the bound."""
        if self.unmeasured == "depth":
            return (f"a directory more than {DISK_WALK_MAX_DEPTH} levels deep in its writable directories, "
                    "deeper than the disk walk measures")
        if self.unmeasured == "entries":
            return (f"more than {DISK_WALK_MAX_ENTRIES} entries in its writable directories, or descriptors its "
                    "processes hold, more than the disk watch reads")
        if self.unmeasured == "permission":
            return ("the disk walk was refused permission in its writable directories, which only its own "
                    "processes can take away")
        if self.unmeasured == "hidden":
            return (f"{self.hidden} holds files the disk watch cannot list, which count as past its {limit_gib} "
                    "GiB")
        if self.unlinked:
            return (f"{self.used} bytes in its writable directories, {self.unlinked} of them in files its processes "
                    f"hold open with no name, over {limit_gib} GiB")
        return f"{self.used} bytes in its writable directories over {limit_gib} GiB"


class Unmeasured(Exception):
    """A watched tree the walk does not measure (``Unmeasurable``)."""

    def __init__(self, why: Unmeasurable, used: int, detail: str) -> None:
        super().__init__(detail)
        self.why: Unmeasurable = why
        #: Bytes counted before the walk stopped.
        self.used = used


def _nspid(proc: Path, pid: int | str) -> list[str] | None:
    """Process *pid*'s pid in each PID namespace from the reader's down (``NSpid`` of its status);
    None once it has ended."""
    try:
        text = (proc / str(pid) / "status").read_text(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return None
    return next((line.split()[1:] for line in text.splitlines() if line.startswith("NSpid:")), [])


class Held(NamedTuple):
    """What one check of a step's processes found (``held_unlinked``)."""

    #: Bytes of the files they hold open with no name left.
    bytes: int
    #: The processes of the container's own whose descriptors could not be listed, running and not
    #: exiting, each by its pid and its start time: a later check judges any of them it finds
    #: unlistable again (``suspects``).
    unlistable: frozenset[tuple[int, int]]


def _task_state(task: Path) -> tuple[str, int, int] | None:
    """The state, flags and start time of the task whose ``/proc`` directory is *task*, from its
    ``stat``; None once it has ended."""
    try:
        text = (task / "stat").read_text(errors="replace")
    except (FileNotFoundError, ProcessLookupError):
        return None
    # The fields after the command name, which may hold spaces and parentheses: the state is the
    # third field, the flags the ninth, the start time the 22nd.
    fields = text.rsplit(")", 1)[1].split()
    return fields[0], int(fields[6]), int(fields[19])


def _running(state: tuple[str, int, int] | None) -> bool:
    """Whether a task in *state* runs: neither ended, exited nor exiting."""
    return state is not None and state[0] not in EXITED_STATES and not state[1] & PF_EXITING


def _listing(proc: Path, pid: int) -> tuple[Iterator[os.DirEntry[str]] | None, int | None] | None:
    """The descriptors of process *pid* to read, or why none: an open ``os.scandir`` of them, through
    a running thread when its first thread has exited; ``(None, start)`` when the process runs and
    they cannot be listed, *start* its start time; None when it has ended, exited or is exiting."""
    base = proc / str(pid)
    try:
        return os.scandir(base / "fd"), None
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError:
        pass
    state = _task_state(base)
    if state is None:
        return None
    if _running(state):
        return None, state[2]
    # Its first thread has ended; a thread still running shares its descriptors.
    try:
        tasks = sorted(os.listdir(base / "task"))
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None
    for tid in tasks:
        if not _running(_task_state(base / "task" / tid)):
            continue
        try:
            return os.scandir(base / "task" / tid / "fd"), None
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            return None, state[2]
    return None


def _hidden(proc: Path, pid: int, nspid: list[str]) -> str:
    """Which process hid its descriptors, for its role's record: its pid, its program as far as it
    can be read, and its pids in each PID namespace, the runner's first. The kernel may refuse the
    program's link of such a process too; its command line, which anyone may read, then names it,
    its first ``HIDDEN_ARGV0_CHARS`` characters."""
    try:
        program = os.readlink(proc / str(pid) / "exe")
    except OSError as e:
        why = errno.errorcode.get(e.errno, str(e.errno))
        try:
            with open(proc / str(pid) / "cmdline", "rb") as f:
                argv0 = f.read(HIDDEN_ARGV0_CHARS * 4).split(b"\0", 1)[0].decode("utf-8", "replace")
        except OSError:
            argv0 = ""
        program = (f"{argv0[:HIDDEN_ARGV0_CHARS]} by its command line, its program link unreadable ({why})" if argv0
                   else f"its program unreadable ({why})")
    return f"process {pid} ({program}; pids {' '.join(nspid)} from the runner's PID namespace down)"


def _open_dir(name: str, dir_fd: int | None, *, need: int = _LIST) -> int | None:
    """A descriptor of directory *name*, relative to *dir_fd* when one is given, opened without
    following a link; None when it is gone, or is no directory any more. A directory whose owner
    lacks any of the permission bits *need*, which the role may have taken away, is given its
    owner's read, write and search first, through a descriptor that names it alone: it is in the
    role's own directory, and the runner removes it with the rest. Raise PermissionError when it
    is refused even so: the role took the permission away again."""
    try:
        fd = os.open(name, _DIR_FLAGS, dir_fd=dir_fd)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:  # replaced by a link since it was listed
            return None
        if e.errno != errno.EACCES:
            raise
    else:
        mode = os.fstat(fd).st_mode
        if mode & need != need:
            try:
                os.fchmod(fd, stat.S_IMODE(mode) | stat.S_IRWXU)
            except BaseException:
                os.close(fd)
                raise
        return fd
    try:
        path_fd = os.open(name, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as e:
        if e.errno == errno.ELOOP:
            return None
        raise
    try:
        own = f"/proc/self/fd/{path_fd}"
        os.chmod(own, stat.S_IMODE(os.fstat(path_fd).st_mode) | stat.S_IRWXU)
        return os.open(own, _DIR_FLAGS & ~os.O_NOFOLLOW)
    finally:
        os.close(path_fd)


def disk_usage(paths: Sequence[Path]) -> int:
    """Bytes allocated to *paths* and everything under them: each file's blocks, counted once
    per inode however many links it has, as ``du`` counts them. A link is never followed, and a
    path that is gone counts nothing; a path that lies in another is counted once. The walk opens
    each directory relative to its parent's descriptor and keeps one open for each level it is
    in, so no path it builds is longer than a name, however long the tree's paths are, and it
    never recurses. Of the entries it counts it remembers only the files with more than one link,
    which alone can be met twice, and the names of the directories it has yet to enter, so a
    tree of plain files costs it nothing, and its worst at ``DISK_WALK_MAX_ENTRIES`` entries is
    about 150 MiB (a million files linked twice; two million directories in one take 120 MiB).
    See ``_open_dir`` for a directory its owner cannot list.

    Raise Unmeasured for a directory more than ``DISK_WALK_MAX_DEPTH`` levels below a path, for
    more than ``DISK_WALK_MAX_ENTRIES`` entries, and for a permission the walk is refused
    (``Unmeasurable``)."""
    tops: list[tuple[str, os.stat_result]] = []
    top_keys: set[tuple[int, int]] = set()
    for path in paths:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            continue
        if (st.st_dev, st.st_ino) not in top_keys:
            top_keys.add((st.st_dev, st.st_ino))
            tops.append((os.fspath(path), st))
    linked: set[tuple[int, int]] = set()
    used = entries = 0
    for top, st in tops:
        used += st.st_blocks * 512
        if not stat.S_ISDIR(st.st_mode):
            continue
        fd: int | None = None
        # Each open directory of the current path, with the subdirectories it has left to walk.
        stack: list[tuple[int, list[str]]] = []
        try:
            fd = _open_dir(top, None)
            while fd is not None:
                subdirs = []
                with os.scandir(fd) as listing:
                    for entry in listing:
                        entries += 1
                        if entries > DISK_WALK_MAX_ENTRIES:
                            raise Unmeasured("entries", used, f"more than {DISK_WALK_MAX_ENTRIES} entries")
                        try:
                            info = entry.stat(follow_symlinks=False)
                        except FileNotFoundError:
                            continue
                        key = (info.st_dev, info.st_ino)
                        if stat.S_ISDIR(info.st_mode):
                            if key in top_keys:  # another path given, counted as its own
                                continue
                            subdirs.append(entry.name)
                        elif info.st_nlink > 1:
                            if key in linked:
                                continue
                            linked.add(key)
                        used += info.st_blocks * 512
                stack.append((fd, subdirs))
                fd = None
                while stack and fd is None:
                    parent, pending = stack[-1]
                    if not pending:
                        stack.pop()
                        os.close(parent)
                        continue
                    child = _open_dir(pending.pop(), parent)
                    # The directory's level below *top* is the number of directories open above it.
                    if child is not None and len(stack) > DISK_WALK_MAX_DEPTH:
                        os.close(child)
                        raise Unmeasured("depth", used, f"a directory more than {DISK_WALK_MAX_DEPTH} levels deep")
                    fd = child
        except PermissionError as e:
            raise Unmeasured("permission", used, f"the walk was refused permission: {e.strerror}") from e
        finally:
            if fd is not None:
                os.close(fd)
            for parent, _ in stack:
                os.close(parent)
    return used


def held_unlinked(cgroup_dir: Path, paths: Sequence[Path], *, proc: Path = PROC,
                  harness_pids: Collection[int] | None = frozenset(),
                  suspects: Collection[tuple[int, int]] = frozenset()) -> Held:
    """Bytes allocated to the files the processes of the cgroup tree *cgroup_dir* hold open with no
    name left, ones they unlinked or made with ``O_TMPFILE``, on the filesystems *paths* lie on:
    each file's blocks, read through a descriptor that holds it, counted once per inode however
    many descriptors hold it. A named file is the walk's (``disk_usage``), and one on another
    filesystem is not the role's disk (``/dev/shm`` is memory, which its memory limit counts). A
    process gone since its cgroup listed it is passed over, and so is a descriptor closed since it
    was listed. A process that has exited or is exiting holds none and is passed over, and one
    whose first thread exited is read through a running thread (``_listing``). A running process
    whose descriptors this user may not list is passed over when it is the harness's: at the
    runner's own PID namespace depth, or among *harness_pids*, the host pids of the harness's own in
    the container, or with *harness_pids* None, when those cannot be told yet. Any other, one of the
    container's own, is in the answer's ``unlistable``, and when *suspects*, those of the check
    before, hold it too (the same pid and start time), raise Unmeasured (``hidden``). Raise
    Unmeasured for more than ``DISK_WALK_MAX_ENTRIES`` descriptors, and OSError for any other
    failure to read them."""
    devices: set[int] = set()
    for path in paths:
        try:
            devices.add(os.lstat(path).st_dev)
        except FileNotFoundError:
            continue
    seen: set[tuple[int, int]] = set()
    unlistable: set[tuple[int, int]] = set()
    used = entries = 0
    own_depth = None
    for pid in pids(cgroup_dir) if devices else []:
        found = _listing(proc, pid)
        if found is None:
            continue
        listing, start = found
        if listing is None:
            if harness_pids is None or pid in harness_pids:
                continue
            nspid = _nspid(proc, pid)
            own_depth = len(_nspid(proc, "self") or []) if own_depth is None else own_depth
            if nspid is None or len(nspid) <= own_depth:
                continue
            if (pid, start) in suspects:
                raise Unmeasured("hidden", used, _hidden(proc, pid, nspid))
            unlistable.add((pid, start))
            continue
        with listing:
            for entry in listing:
                entries += 1
                if entries > DISK_WALK_MAX_ENTRIES:
                    raise Unmeasured("entries", used, f"more than {DISK_WALK_MAX_ENTRIES} descriptors")
                try:
                    info = entry.stat()
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                key = (info.st_dev, info.st_ino)
                if info.st_nlink or not stat.S_ISREG(info.st_mode) or info.st_dev not in devices or key in seen:
                    continue
                seen.add(key)
                used += info.st_blocks * 512
    return Held(used, frozenset(unlistable))


def role_disk(paths: Sequence[Path], cgroup_dir: Path | None, *, at: float, proc: Path = PROC,
              harness_pids: Collection[int] | None = frozenset(),
              suspects: Collection[tuple[int, int]] = frozenset()) -> DiskSample:
    """One reading at *at* of the disk a role holds: what its writable directories *paths* hold
    (``disk_usage``) and, with its step's cgroup *cgroup_dir*, the files its processes hold there
    with no name (``held_unlinked``, with *harness_pids* and *suspects*, the processes the check
    before found unlistable). A walk or a count that stops short (``Unmeasured``) is a sample past
    the cap, with what was counted until then. Raise OSError as they do."""
    try:
        named = disk_usage(paths)
    except Unmeasured as e:
        return DiskSample(at=at, used=e.used, unmeasured=e.why)
    if cgroup_dir is None:
        return DiskSample(at=at, used=named)
    try:
        held = held_unlinked(cgroup_dir, paths, proc=proc, harness_pids=harness_pids, suspects=suspects)
    except Unmeasured as e:
        return DiskSample(at=at, used=named + e.used, unmeasured=e.why, unlinked=e.used,
                          hidden=str(e) if e.why == "hidden" else None)
    return DiskSample(at=at, used=named + held.bytes, unlinked=held.bytes, unlistable=held.unlistable)


@dataclass(frozen=True)
class DiskAtEnd:
    """A contestant role's disk once it was reported, its step and the files it held with no name
    gone: what decides whether others starved it of disk (``disk_starvation``)."""

    #: When it was read.
    at: float
    #: The role's declared disk, its cap.
    limit_bytes: int
    #: What its writable directories held then (``disk_usage``); None when the walk did not
    #: measure them, which only the role's own processes can cause.
    own_bytes: int | None
    #: The space free to an unprivileged writer on their filesystems then (``free_bytes``), the
    #: least where they lie on several; None when it, or what the directories hold, could not be
    #: read.
    free_bytes: int | None


def free_bytes(paths: Sequence[Path]) -> int:
    """The space an unprivileged writer may still take on the filesystems *paths* lie on: the
    least of their ``statvfs`` available blocks. Raise OSError when none can be read."""
    free = []
    for path in paths:
        try:
            st = os.statvfs(path)
        except FileNotFoundError:
            continue
        free.append(st.f_bavail * st.f_frsize)
    if not free:
        raise FileNotFoundError(f"none of {[str(p) for p in paths]} is there to read its filesystem's free space")
    return min(free)


def disk_starvation(at_end: DiskAtEnd) -> str | None:
    """Why the contestant role whose disk was read at its end as *at_end* was starved of it, or
    None. It was when the free space left it at most what ``POOL_FULL`` of the room its cap still
    allowed it leaves, the room being its cap less what its directories held: its writes may have
    failed with ENOSPC for what others wrote, whatever its cap allowed. A role whose own files
    reach its cap had no room left to be starved of, and one whose tree was not measured made it
    so itself. Raise ValueError for free space that was not read, which nothing rules out."""
    if at_end.free_bytes is None:
        raise ValueError("the free space was not read, so starvation cannot be judged")
    if at_end.own_bytes is None:
        return None
    room = at_end.limit_bytes - at_end.own_bytes
    if room <= 0:
        return None
    left = room - math.ceil(POOL_FULL * room)
    if at_end.free_bytes > left:
        return None
    return (f"its node-local disk had {at_end.free_bytes} bytes free of the {room} its {at_end.limit_bytes} byte cap "
            f"still left it")


def remove_written(paths: Sequence[Path]) -> None:
    """Remove each of *paths*, the writable directories of a role whose processes are gone, with
    everything under them, whatever the role made of them. The owner is given back what removing
    needs (``_open_dir``), a link is removed and never followed, and a directory deeper than
    ``DISK_WALK_MAX_DEPTH`` below its path is first moved up under that path, so no more
    directories are open at once than the walk keeps. A path that is gone is passed over. Raise
    OSError for what could not be removed."""
    for path in paths:
        try:
            st = os.lstat(path)
        except FileNotFoundError:
            continue
        if not stat.S_ISDIR(st.st_mode):
            os.unlink(path)
            continue
        fd = _open_dir(os.fspath(path), None, need=stat.S_IRWXU)
        if fd is None:
            continue
        try:
            _empty(fd)
        finally:
            os.close(fd)
        os.rmdir(path)


def _empty(top: int) -> None:
    """Remove everything in the directory open as *top*, as ``remove_written`` says."""
    # Each open directory of the current path, with its name in its parent and the entries it
    # has left; the top's own descriptor is the caller's.
    stack: list[tuple[int, str | None, list[str]]] = [(top, None, os.listdir(top))]
    try:
        while stack:
            fd, name, names = stack[-1]
            if not names:
                stack.pop()
                if name is not None:
                    os.close(fd)
                    os.rmdir(name, dir_fd=stack[-1][0])
                continue
            entry = names.pop()
            try:
                info = os.stat(entry, dir_fd=fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISDIR(info.st_mode):
                os.unlink(entry, dir_fd=fd)
            elif len(stack) > DISK_WALK_MAX_DEPTH:
                moved = f".hls-removed-{uuid.uuid4().hex}"
                os.rename(entry, moved, src_dir_fd=fd, dst_dir_fd=top)
                stack[0][2].append(moved)
            elif (child := _open_dir(entry, fd, need=stat.S_IRWXU)) is not None:
                stack.append((child, entry, os.listdir(child)))
    finally:
        for fd, name, _ in stack:
            if name is not None:
                os.close(fd)


@dataclass(frozen=True)
class PidsHits:
    """How often a step's pids limit was reached, and since when the count stands."""

    #: The times the limit was reached: the most a read saw.
    count: int
    #: When the count was first read at its latest rise, by the reader's clock: a rise is dated
    #: by the first read that shows it, as an OOM kill is (``hls.agent.memory.split_evidence``).
    #: None while no read saw a count above 0.
    rose_at: float | None


def pids_read(last: PidsHits | None, count: int, at: float) -> PidsHits:
    """The pids limit hits known after a read that saw *count* at *at*, *last* those known
    before: a higher count, dated *at*, else *last* as it was."""
    if last is not None and count <= last.count:
        return last
    return PidsHits(count=count, rose_at=at if count else None)


@dataclass
class _DiskWatched:
    #: Role name, for log lines.
    role: str
    #: The role's writable directories.
    paths: tuple[Path, ...]
    #: The role's declared disk in bytes.
    limit_bytes: int
    #: Stops every process of the step; called when the directories pass the limit.
    kill: Callable[[], None]
    #: The step's cgroup directory, whose ``pids.events`` each walk also reads; None for none.
    cgroup_dir: Path | None = None
    #: The most times the step's pids limit was reached, as the walks read it, and when a walk
    #: first read its latest rise; None until one read it.
    pids_hits: PidsHits | None = None
    #: The step's process ceiling (``hls.substrate.nproc``), whose pool each walk also samples;
    #: None for none.
    ceiling: int | None = None
    #: The last sample of the uid's pool and the step's own tasks; None until one was read.
    pool: PoolSample | None = None
    #: The most of each the samples read; None until one was read.
    peak: PoolPeak | None = None
    #: The sample at which the watch decided to kill the step, set before the kill starts;
    #: None while it has not, and again if the kill failed.
    breach: DiskSample | None = None
    #: False once the stage reported the role: its directories are walked no more.
    active: bool = True
    #: The host pids of the harness's own processes in the container, which a walk never judges by
    #: their descriptors (``held_unlinked``); None while they cannot be told, when it judges none.
    harness_pids: Callable[[], Collection[int] | None] = field(default=lambda: ())
    #: The processes the last walk found unlistable (``DiskSample.unlistable``), which this walk
    #: judges if it finds them so again.
    unlistable: frozenset[tuple[int, int]] = frozenset()


@dataclass
class _Watched:
    #: Role name, for log lines.
    role: str
    #: The step's ``step_<n>/user`` directory.
    cgroup_dir: Path
    #: The role's declared memory in bytes.
    limit_bytes: int
    #: Stops every process of the step; called when the footprint passes the limit (for a
    #: step with ``end_one``, once single kills do not answer it).
    kill: Callable[[], None]
    #: The latest sample; None until the first one is read.
    last: MemorySample | None = None
    #: The sample at which the watch decided to kill the step, set before the kill starts;
    #: None while it has not, and again if the kill failed.
    breach: MemorySample | None = None
    #: Ends one process of the step at a breach, in place of ``kill``; returns False when it
    #: found none to end, and raises ``CannotEndOne`` when it cannot tell which to end. None for
    #: a step the watch kills whole.
    end_one: Callable[[], bool] | None = None
    #: Whether the process ``end_one`` last ended is still exiting; None when that is not known,
    #: which reads as not.
    exiting: Callable[[], bool] | None = None
    #: The samples at which ``end_one`` ended a process, oldest first.
    kills: list[MemorySample] = field(default_factory=list)
    #: The samples at which ``end_one`` ended a process since the step was last sampled at or
    #: under its limit, oldest first: the kills that have not brought it back.
    unrelieved: list[MemorySample] = field(default_factory=list)
    #: Why the watch killed a step with ``end_one`` whole, set with ``breach``; None while it has
    #: not.
    kill_reason: str | None = None
    #: The sample with the highest footprint so far; None until the first one is read.
    highest: MemorySample | None = None
    #: Whether a footprint past ``limit_bytes`` kills the step; a step only observed is never
    #: killed (``MemoryWatch.observe``).
    enforced: bool = True


def _never_killed() -> None:
    raise RuntimeError("an observed cgroup is never killed")


class MemoryWatch:
    """Samples every watched step; kills a step past its limit when *enforce* is set."""

    def __init__(self, *, enforce: bool, interval_sec: float = 1.0, clock: Callable[[], float] = time.time,
                 disk_interval_sec: float = 5.0,
                 on_breach: Callable[[str, MemorySample, int], None] | None = None,
                 job_dir: Path | None = None, repeat_grace_sec: float = 5.0,
                 on_single_kill: Callable[[str, MemorySample, int], None] | None = None,
                 proc: Path = PROC) -> None:
        #: True where the kernel does not enforce memory limits because swap is unlimited.
        self.enforce = enforce
        #: The job's ``job_<id>`` cgroup, where its own memory limit is; None where there is no
        #: job cgroup, which leaves the job's OOM count unknown.
        self.job_dir = job_dir
        #: Seconds between passes.
        self.interval_sec = interval_sec
        #: Time source for samples.
        self.clock = clock
        #: Called with the step id, the sample and the limit after a step's kill returned.
        self.on_breach = on_breach
        #: Called with the step id, the sample and the limit after ``end_one`` ended one process
        #: of a step.
        self.on_single_kill = on_single_kill
        #: Seconds after an ``end_one`` kill before the watch acts on a breach of that step again.
        self.repeat_grace_sec = repeat_grace_sec
        #: Where the walks read the steps' processes (``held_unlinked``).
        self.proc = proc
        self._steps: dict[str, _Watched] = {}
        self._guard = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        #: Seconds between walks of the watched roles' writable directories (``watch_disk``).
        self.disk_interval_sec = disk_interval_sec
        self._disks: dict[str, _DiskWatched] = {}
        self._disk_thread: threading.Thread | None = None

    def watch(self, step: str, cgroup_dir: Path, limit_bytes: int, kill: Callable[[], None], *, role: str,
              end_one: Callable[[], bool] | None = None, exiting: Callable[[], bool] | None = None) -> None:
        """Start sampling *step*, the "<job>.<step>" id from the role's report, whose cgroup
        is *cgroup_dir* and whose declared memory is *limit_bytes*. *role* is the role's name,
        for log lines. With *end_one*, a breach ends one process of the step while single kills
        answer it, and none while *exiting* says the last one ended is still exiting (see the
        module docstring)."""
        with self._guard:
            self._steps[step] = _Watched(role, cgroup_dir, limit_bytes, kill, end_one=end_one, exiting=exiting)

    def observe(self, name: str, cgroup_dir: Path, *, role: str) -> None:
        """Start sampling the cgroup *cgroup_dir* under *name*, as a watched step is sampled, without
        ever killing it: its samples are evidence only (``last``, ``highest``)."""
        with self._guard:
            self._steps[name] = _Watched(role, cgroup_dir, 0, _never_killed, enforced=False)

    def job_ooms(self) -> JobOoms | None:
        """The job cgroup's OOM counts now; None when there is no job cgroup, or its own count
        or its count of limits reached below it cannot be read. A step cgroup that cannot be
        read is left out, as it is gone."""
        if self.job_dir is None:
            return None
        try:
            own = read_keyed(self.job_dir / "memory.events.local").get("oom")
            every = read_keyed(self.job_dir / "memory.events").get("oom")
        except (OSError, ValueError):
            return None
        if own is None or every is None:
            return None
        steps = {}
        for step in sorted(self.job_dir.glob("step_*")):
            try:
                count = read_keyed(step / "memory.events").get("oom")
            except (OSError, ValueError):
                continue
            if count is not None:
                steps[step.name] = count
        return JobOoms(own=own, below=every - own, steps=steps)

    def last(self, step: str) -> MemorySample | None:
        """The latest sample of *step*; it stays after the step is gone."""
        with self._guard:
            w = self._steps.get(step)
            return None if w is None else w.last

    def highest(self, step: str) -> MemorySample | None:
        """The sample of *step* with the highest footprint so far; it stays after the step is
        gone. A shorter spike between two samples is not in it."""
        with self._guard:
            w = self._steps.get(step)
            return None if w is None else w.highest

    def kills(self, step: str) -> list[MemorySample]:
        """The samples at which the watch ended one process of *step* (``end_one``), oldest
        first."""
        with self._guard:
            w = self._steps.get(step)
            return [] if w is None else list(w.kills)

    def kill_reason(self, step: str) -> str | None:
        """Why the watch killed *step*, watched with ``end_one``, whole rather than ending one
        more of its processes; None when it did not, or *step* has no ``end_one``. It is set with
        ``breach``."""
        with self._guard:
            w = self._steps.get(step)
            return None if w is None else w.kill_reason

    def breach(self, step: str) -> MemorySample | None:
        """The sample at which the watch killed *step*, if it did. It is set before the kill
        starts, so it is visible by the time the killed step is seen to end."""
        with self._guard:
            w = self._steps.get(step)
            return None if w is None else w.breach

    def sample_once(self) -> None:
        """One pass over every watched step that has not been killed. A step that cannot be
        read is skipped. A step whose kill raised has its breach cleared and is tried again at
        the next pass, since its footprint is still over the limit while it has processes."""
        with self._guard:
            items = [(s, w) for s, w in self._steps.items() if w.breach is None]
        for step, w in items:
            try:
                sample = read_sample(w.cgroup_dir, self.clock())
            except (OSError, ValueError):
                continue
            with self._guard:
                w.last = sample
                if w.highest is None or sample.footprint > w.highest.footprint:
                    w.highest = sample
            if not (self.enforce and w.enforced and sample.footprint > w.limit_bytes):
                w.unrelieved.clear()
                continue
            reason = None
            if w.end_one is not None:
                reason = self._end_one(step, w, sample)
                if reason is None:
                    continue
            # The breach is recorded before the kill: the stage reads it as soon as the killed
            # role is seen to end, which can be before kill() returns.
            with self._guard:
                w.breach = sample
                w.kill_reason = reason
            log.warning("role %s (step %s) used %d bytes against a %d byte limit%s; killing its step", w.role, step,
                        sample.footprint, w.limit_bytes, "" if reason is None else f", {reason}")
            try:
                w.kill()
            except Exception:
                log.exception("killing step %s of role %s failed; trying again at the next sample", step, w.role)
                with self._guard:
                    w.breach = None
                    w.kill_reason = None
                continue
            if self.on_breach is not None:
                try:
                    self.on_breach(step, sample, w.limit_bytes)
                except Exception:
                    log.exception("on_breach for step %s of role %s failed", step, w.role)

    def _end_one(self, step: str, w: _Watched, sample: MemorySample) -> str | None:
        """Answer a breach of *step* at *sample* by ending one of its processes, or by waiting
        while its last such kill is within the grace or its victim is still exiting, for up to
        ``END_ONE_STILL_OVER_SEC`` after that kill; an ``exiting`` that raises reads as still
        exiting. None once it is answered so. A call that raised is tried again at the next pass.
        Otherwise say why the step is to be killed whole: the single kills since it was last
        within its limit have not brought it back (``END_ONE_STILL_OVER_SEC`` from the first of
        them, ``END_ONE_MAX_KILLS``), or ``end_one`` found no process to end or could not tell
        which to end."""
        assert w.end_one is not None
        unrelieved = w.unrelieved
        if unrelieved and sample.at - unrelieved[0].at >= END_ONE_STILL_OVER_SEC:
            return (f"still over its limit {sample.at - unrelieved[0].at:.0f} s after the first of its processes was "
                    "ended")
        if w.kills and sample.at - w.kills[-1].at < self.repeat_grace_sec:
            return None
        try:
            still_exiting = w.exiting is not None and w.exiting()
        except Exception:
            # Not knowing reads as still exiting: ending the next process on a guess can end one
            # that the victim's exit would have spared.
            log.exception("whether the last process ended in step %s of role %s exited cannot be told; taking it "
                          "as still exiting", step, w.role)
            still_exiting = True
        # The wait is bounded from the victim's own kill. The 30 s rule above runs from the first
        # unrelieved kill, and a step that went back within its limit has none, so a victim stuck in
        # its exit could otherwise hold the next kill off for good.
        if still_exiting and w.kills and sample.at - w.kills[-1].at < END_ONE_STILL_OVER_SEC:
            return None
        if len(unrelieved) >= END_ONE_MAX_KILLS:
            return (f"{len(unrelieved)} of its processes were ended since it was last within its limit, and it is "
                    "still over it")
        log.warning("role %s (step %s) used %d bytes against a %d byte limit; ending one of its processes", w.role,
                    step, sample.footprint, w.limit_bytes)
        try:
            ended = w.end_one()
        except CannotEndOne as e:
            return str(e)
        except Exception:
            log.exception("ending a process of step %s of role %s failed; trying again at the next sample", step,
                          w.role)
            return None
        if not ended:
            return "no process of it was left to end"
        with self._guard:
            w.kills.append(sample)
        unrelieved.append(sample)
        if self.on_single_kill is not None:
            try:
                self.on_single_kill(step, sample, w.limit_bytes)
            except Exception:
                log.exception("on_single_kill for step %s of role %s failed", step, w.role)
        return None

    def start(self) -> None:
        """Sample memory, and walk the watched directories, each in a background thread of its
        own until ``stop``, so a long walk never delays a memory sample."""
        def loop(sample: Callable[[], None], interval_sec: float, what: str) -> None:
            while not self._stop.wait(interval_sec):
                try:
                    sample()
                except Exception:
                    log.exception("%s pass failed", what)

        memory = (self.sample_once, self.interval_sec, "memory watch")
        self._thread = threading.Thread(target=loop, args=memory, name="hls-memory-watch", daemon=True)
        self._thread.start()
        disk = (self.sample_disks_once, self.disk_interval_sec, "disk watch")
        self._disk_thread = threading.Thread(target=loop, args=disk, name="hls-disk-watch", daemon=True)
        self._disk_thread.start()

    def stop(self) -> None:
        """End the background threads started by ``start`` and wait for them to finish. Samples
        and breaches already recorded stay readable."""
        self._stop.set()
        for thread in (self._thread, self._disk_thread):
            if thread is not None:
                thread.join(max(self.interval_sec, self.disk_interval_sec) + 1.0)

    def watch_disk(self, step: str, paths: Sequence[Path], limit_bytes: int, kill: Callable[[], None], *,
                   role: str, cgroup_dir: Path | None = None, ceiling: int | None = None,
                   harness_pids: Callable[[], Collection[int] | None] = lambda: ()) -> None:
        """Start walking *paths*, the writable directories of the role whose step is *step*,
        and kill the step once they hold more than *limit_bytes*. *role* is the role's name, for
        log lines. With *cgroup_dir*, the step's cgroup directory, each walk also reads how often
        the step's pids limit was reached (``pids_hits``), and the files its processes hold with no
        name, *harness_pids* naming the host pids of the harness's own among them before each walk
        (``held_unlinked``). With *ceiling*, the step's process ceiling, each walk also counts the
        uid's tasks on the node and the step's own (``pool``)."""
        with self._guard:
            self._disks[step] = _DiskWatched(role, tuple(paths), limit_bytes, kill, cgroup_dir=cgroup_dir,
                                             ceiling=ceiling, harness_pids=harness_pids)

    def unwatch_disk(self, step: str) -> None:
        """Stop walking the directories of *step*, whose role was reported or is being stopped.
        A breach already recorded stays readable, and none is recorded after this returns. A
        step never watched is no error."""
        with self._guard:
            w = self._disks.get(step)
            if w is not None:
                w.active = False

    def disk_breach(self, step: str) -> DiskSample | None:
        """The sample at which the watch killed *step* for its disk, if it did. It is set before
        the kill starts, so it is visible by the time the killed step is seen to end."""
        with self._guard:
            w = self._disks.get(step)
            return None if w is None else w.breach

    def pool(self, step: str) -> PoolSample | None:
        """The last sample of *step*'s pool a walk read before the step's walks stopped; None
        when none was read, or the step has no ceiling."""
        with self._guard:
            w = self._disks.get(step)
            return None if w is None else w.pool

    def pool_peak(self, step: str) -> PoolPeak | None:
        """The most tasks of *step*'s pool and of its own the walks read before its walks
        stopped; None when none was read, or the step has no ceiling."""
        with self._guard:
            w = self._disks.get(step)
            return None if w is None else w.peak

    def disk_at_end(self, step: str) -> DiskAtEnd | None:
        """*step*'s disk now, its role reported and its step gone (``DiskAtEnd``): what its
        writable directories hold and the free space of their filesystem; None for a step whose
        disk was never watched. A walk that does not measure them leaves its own bytes unknown,
        and free space that cannot be read is kept as unread, never a crashed report."""
        with self._guard:
            w = self._disks.get(step)
        if w is None:
            return None
        at = self.clock()
        try:
            own: int | None = disk_usage(w.paths)
            free: int | None = free_bytes(w.paths)
        except Unmeasured:
            own, free = None, None
            try:
                free = free_bytes(w.paths)
            except OSError:
                log.exception("the free space of role %s's disk (step %s) could not be read at its end", w.role, step)
        except OSError:
            log.exception("role %s's disk (step %s) could not be read at its end", w.role, step)
            own, free = None, None
        return DiskAtEnd(at=at, limit_bytes=w.limit_bytes, own_bytes=own, free_bytes=free)

    def pids_hits(self, step: str) -> PidsHits | None:
        """The most times *step*'s pids limit was reached that a walk read, dated by the first walk
        that read its latest rise; None when no walk read it. It stays after the step is gone, as
        its cgroup does not."""
        with self._guard:
            w = self._disks.get(step)
            return None if w is None else w.pids_hits

    def _read_context(self, w: _DiskWatched, held: tuple[float, int] | None) -> None:
        """Record watched role *w*'s pool, from *held* (when and how many tasks the uid held on
        the node) and its step's own tasks, when it has a ceiling, as its last sample and in its
        peak, and its step's pids limit count, when it has a cgroup, dated by this walk when it
        rose (``pids_read``). Raise as the readings do."""
        if w.ceiling is not None and held is not None:
            own = step_tasks(w.cgroup_dir) if w.cgroup_dir is not None else None
            with self._guard:
                if w.active:
                    w.pool = PoolSample(at=held[0], uid_tasks=held[1], step_tasks=own, ceiling=w.ceiling)
                    w.peak = peak_with(w.peak, w.pool)
        if w.cgroup_dir is not None:
            read_at, hits = self.clock(), pids_limit_hits(w.cgroup_dir)
            if hits is not None:
                with self._guard:
                    w.pids_hits = pids_read(w.pids_hits, hits, read_at)

    def sample_disks_once(self) -> None:
        """One walk of every watched role's writable directories, as ``sample_once`` passes over
        the steps. A role past its limit, or whose tree the walk does not measure (``Unmeasured``:
        too deep, too many entries, or a permission refused, which only its own processes can
        do), has its breach recorded, then its step killed, and a role whose kill raised is tried
        again at the next walk. A walk that fails otherwise is logged and the role walked again at
        the next. Each walk first reads the step's pids limit count, when it has a cgroup, and
        its pool, when it has a ceiling: the uid's tasks on the node, counted once for every walk
        of the pass, and the step's own. Those are context, and the cap is the contestant's
        limit: a reading that fails, whatever it raises, is logged and the role's disk is walked
        all the same, so no failure of theirs can leave a role uncapped."""
        with self._guard:
            items = [(s, w) for s, w in self._disks.items() if w.active and w.breach is None]
        held = None
        if any(w.ceiling is not None for _, w in items):
            try:
                held = (self.clock(), uid_tasks(os.getuid()))
            except Exception:  # noqa: BLE001 -- a failed count costs the pool's context, never a cap
                log.exception("the uid's tasks on the node could not be counted; the walk goes on without them")
        for step, w in items:
            try:
                self._read_context(w, held)
            except Exception:  # noqa: BLE001 -- as the count's: the role's disk is walked all the same
                log.exception("the pids limit count or the pool of role %s (step %s) could not be read; its disk "
                              "is walked all the same", w.role, step)
            try:
                harness_pids = w.harness_pids()
            except Exception:  # noqa: BLE001 -- unknown, the walk judges no process by its descriptors
                log.exception("the harness's processes of role %s (step %s) could not be told; this walk judges "
                              "none by its descriptors", w.role, step)
                harness_pids = None
            try:
                sample = role_disk(w.paths, w.cgroup_dir, at=self.clock(), proc=self.proc, harness_pids=harness_pids,
                                   suspects=w.unlistable)
            except OSError:
                log.exception("the writable directories of role %s (step %s) could not be walked", w.role,
                              step)
                continue
            w.unlistable = sample.unlistable
            if sample.used <= w.limit_bytes and sample.unmeasured is None:
                continue
            with self._guard:
                if not w.active:
                    continue
                w.breach = sample
            log.warning("role %s (step %s) passed its %d byte disk cap: %s; killing its step", w.role, step,
                        w.limit_bytes, sample.detail(w.limit_bytes // 2**30))
            try:
                w.kill()
            except Exception:
                log.exception("killing step %s of role %s failed; trying again at the next walk", step,
                              w.role)
                with self._guard:
                    w.breach = None
