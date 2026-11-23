"""The local scheduler's state directory: one record per job, the reservation file and the host's
inventory, all changed under one ``flock``.

Layout under the state directory, which is this user's alone (mode 0700):

    lock               flock: exclusive to change anything, shared to read
    next_id            the next job id
    host.json          the host's inventory (``HostFacts``)
    reservations.json  what each allocation that holds resources holds, by job id
    scripts/<id>.sh    the batch script as it was at submission, as sbatch keeps it
    active/<id>.json   the record of a job that has not ended or still holds its reservation
    ended/<id>.json    the record of a job that ended and released its reservation
    logs/<id>.log      its supervisor's log

Every file is written whole to a temporary name and renamed over the old one, so a reader sees one
version or the other. The records are the truth; the reservation file is rewritten from them
whenever they change, for the readers of one allocation's reservation (the launcher, the grant
check), and the scheduler never reads it back.

A process is named by its pid with its start time in clock ticks since boot and the host's boot
id, so a pid the kernel gave out again is never taken for the process recorded. Whether it is
alive is known only on the host (by its machine id) and in the PID namespace it was recorded in."""

from __future__ import annotations

import contextlib
import errno
import fcntl
import json
import os
import re
import stat
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from hls.spec.resources import ResourceShape
from hls.substrate.base import JobState
from hls.substrate.local.inventory import HostFacts
from hls.substrate.local.placement import Reservation
from hls.substrate.pidfd import pidfd_open, pidfd_send_signal

#: Seconds a reader or writer waits for the state directory's lock before it counts the
#: scheduler as not answering. A pass holds the lock for milliseconds, and a reclaim for at most
#: ``RECLAIM_WAIT_SEC`` and its stopper's call.
LOCK_TIMEOUT_SEC = 60.0
#: Seconds a reclaim waits for the processes it killed to be gone.
RECLAIM_WAIT_SEC = 10.0
#: Seconds after a failed reclaim before a pass tries the job again, so a process that will not die
#: does not keep the lock held through every pass.
RECLAIM_BACKOFF_SEC = 60.0


_JOB_ID = re.compile(r"[0-9]+")


def check_job_id(job_id: str) -> str:
    """Return *job_id* if it is a local job id, decimal digits, else raise ValueError."""
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError(f"not a job id: {job_id!r}")
    return job_id


class StateError(RuntimeError):
    """The state directory cannot be used: it is not this user's alone, or its lock was not had in
    time."""


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def machine_id() -> str:
    """The host's ``/etc/machine-id``, which a change of its name leaves as it is."""
    return Path("/etc/machine-id").read_text().strip()


def pid_ns() -> int:
    return os.stat("/proc/self/ns/pid").st_ino


def start_ticks(pid: int, *, zombie: bool = False) -> int | None:
    """When process *pid* started, in clock ticks since boot (``/proc/<pid>/stat`` field 22); None
    when there is no such process, or, unless *zombie*, when it has exited and waits to be
    reaped."""
    try:
        raw = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    fields = raw[raw.rindex(b")") + 2:].split()
    return None if fields[0] in (b"Z", b"X") and not zombie else int(fields[19])


class ProcessId(BaseModel):
    """One process, as it was when it was recorded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pid: int = Field(gt=0)
    #: Its start, in clock ticks since boot.
    start_ticks: int = Field(ge=0)
    #: The host's machine id.
    machine: str
    boot_id: str
    pid_ns: int

    @classmethod
    def of(cls, pid: int, *, zombie: bool = False) -> ProcessId | None:
        """Process *pid* as it is now; None when it is gone, or, unless *zombie*, waits to be
        reaped: a child of the caller's that may already have exited."""
        ticks = start_ticks(pid, zombie=zombie)
        if ticks is None:
            return None
        return cls(pid=pid, start_ticks=ticks, machine=machine_id(), boot_id=boot_id(), pid_ns=pid_ns())

    def alive(self) -> bool | None:
        """Whether the process is alive; None when that cannot be told from here: another host or
        another PID namespace. A process of an earlier boot is gone."""
        if self.machine != machine_id() or self.pid_ns != pid_ns():
            return None
        if self.boot_id != boot_id():
            return False
        return start_ticks(self.pid) == self.start_ticks

    def pidfd(self) -> int | None:
        """A pidfd of the process, which a signal sent through cannot reach any other process; None
        when the process is gone, of another boot, or cannot be told from here."""
        if self.alive() is not True:
            return None
        try:
            fd = pidfd_open(self.pid)
        except ProcessLookupError:
            return None
        if fd is None:
            return None
        if start_ticks(self.pid) != self.start_ticks:
            os.close(fd)
            return None
        return fd

    def signal(self, signum: int) -> bool:
        """Send *signum* to the process if it is the one recorded; whether it was sent."""
        fd = self.pidfd()
        if fd is None:
            return False
        try:
            pidfd_send_signal(fd, signum)
        except ProcessLookupError:
            return False
        finally:
            os.close(fd)
        return True


#: A job's stored state. A record is written PENDING, RUNNING once the job holds its reservation,
#: and one of the terminal states once its supervisor ended it.
StoredState = Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"]


class JobRecord(BaseModel):
    """One allocation's job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The job id, decimal digits.
    job_id: str = Field(pattern=r"^[0-9]+$")
    #: The submission's tag (``SubmitRequest.tag``).
    tag: str
    #: The job name, which ``singleton`` goes by.
    name: str
    #: The batch script as it was given; the copy that runs is ``scripts/<id>.sh``.
    script: Path
    #: The file the batch script's stdout and stderr go to.
    output: Path
    #: The requested shape: GPUs, GPU kind, CPUs, memory and the time limit.
    shape: ResourceShape
    #: The job that must complete before this one starts (``afterok``); None for none.
    after: str | None
    #: True to start only while no other job of this name runs.
    singleton: bool
    #: Which stopper the supervisor calls at the end for anything labelled with the job id
    #: (``hls.substrate.local.stoppers``).
    stopper: str
    #: The stopper's options, such as the Docker daemon's socket.
    stopper_options: dict[str, str]
    #: Seconds between the SIGTERM at the end time or at a cancel and the SIGKILL.
    kill_wait_sec: float = Field(gt=0)
    #: Seconds between the supervisor's looks at the queue while it waits.
    poll_sec: float = Field(gt=0)
    #: A random label, set in the batch script's environment only, by which the processes the job
    #: leaves behind are found.
    token: str = Field(pattern=r"^[0-9a-f]{32}$")
    submitted_at: float
    #: The job's supervisor.
    supervisor: ProcessId
    state: StoredState = "PENDING"
    #: Why a pending job waits, as the last pass found it.
    reason: str | None = None
    #: When ``cancel`` was asked for; None when it was not.
    cancel_requested_at: float | None = None
    #: When the job took its reservation; None before.
    started_at: float | None = None
    #: Its end time: ``started_at`` plus the shape's time; None before it started.
    end_time: float | None = None
    #: The batch script's process once it started; None before.
    runner: ProcessId | None = None
    #: When the job ended; None before.
    ended_at: float | None = None
    #: How the batch script ended, as sacct prints it: "<status>:<signal>"; None while it runs and
    #: when it never ran.
    exit_code: str | None = None
    #: What the job holds; None before it held anything.
    reservation: Reservation | None = None
    #: True once the reservation was given back, or when there never was one and the job ended.
    released: bool = False
    #: When a pass may try again to reclaim the reservation after a try failed; None before.
    reclaim_after: float | None = None
    #: What the supervisor or a pass said about how the job ended, for people.
    note: str | None = None

    @property
    def holds(self) -> bool:
        """True while the reservation is held."""
        return self.reservation is not None and not self.released


def reported_state(record: JobRecord) -> JobState:
    """The job's state as the scheduler reports it. A job that still holds its reservation is
    RUNNING, whatever its record says, as Slurm reports a COMPLETING job: something of it may still
    run until its reclaim gave the reservation back. Otherwise the stored state, but for a record
    that says PENDING or RUNNING whose supervisor is gone, which is CANCELLED when a cancel was asked
    for and NODE_FAIL otherwise, and UNKNOWN when whether the supervisor is alive cannot be told
    here."""
    if record.holds:
        return JobState.RUNNING
    state = JobState(record.state)
    if state.terminal:
        return state
    alive = record.supervisor.alive()
    if alive is None:
        return JobState.UNKNOWN
    if alive:
        return state
    return JobState.CANCELLED if record.cancel_requested_at is not None else JobState.NODE_FAIL


class StateDir:
    """The state directory at *root*, made mode 0700 when it is not there."""

    def __init__(self, root: Path) -> None:
        self.root = root

    # paths

    @property
    def lock_path(self) -> Path:
        return self.root / "lock"

    def script_path(self, job_id: str) -> Path:
        return self.root / "scripts" / f"{check_job_id(job_id)}.sh"

    def log_path(self, job_id: str) -> Path:
        return self.root / "logs" / f"{check_job_id(job_id)}.log"

    def _record_path(self, job_id: str, *, ended: bool) -> Path:
        return self.root / ("ended" if ended else "active") / f"{check_job_id(job_id)}.json"

    # the lock

    def check(self) -> None:
        """Make the directory and its subdirectories when they are not there, and raise StateError
        unless the directory is this user's and no one else may write in it."""
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        st = os.lstat(self.root)
        if not stat.S_ISDIR(st.st_mode) or st.st_uid != os.getuid() or st.st_mode & 0o077:
            raise StateError(f"{self.root} must be a directory of this user's with mode 0700")
        for sub in ("active", "ended", "scripts", "logs"):
            (self.root / sub).mkdir(mode=0o700, exist_ok=True)

    @contextlib.contextmanager
    def locked(self, *, exclusive: bool = True, timeout_sec: float = LOCK_TIMEOUT_SEC) -> Iterator[None]:
        """Hold the directory's lock. Each holding opens the lock file afresh, so that two threads
        of one process exclude each other as two processes do. Raise StateError when the lock is
        not had within *timeout_sec*."""
        self.check()
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o600)
        try:
            deadline = time.monotonic() + timeout_sec
            while True:
                try:
                    fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
                    break
                except OSError as e:
                    if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                        raise
                if time.monotonic() >= deadline:
                    raise StateError(f"{self.lock_path} was not had within {timeout_sec:.0f}s")
                time.sleep(0.01)
            yield
        finally:
            # Unlocked before the close, since a child forked meanwhile shares the descriptor.
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    # files

    def _write(self, path: Path, data: bytes, mode: int = 0o600) -> None:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_CLOEXEC, mode)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

    def take_id(self) -> str:
        """The next job id, under the exclusive lock. The first id of a new directory is the current
        POSIX second, so a directory made again after one was removed does not give out the ids
        the old one gave, whose jobs plans may still name, unless the old one gave out more ids
        than seconds have passed since."""
        path = self.root / "next_id"
        try:
            n = int(path.read_text())
        except FileNotFoundError:
            n = int(time.time())
        self._write(path, f"{n + 1}\n".encode())
        return str(n)

    def write_script(self, job_id: str, content: bytes) -> Path:
        path = self.script_path(job_id)
        self._write(path, content, 0o700)
        return path

    def read(self, job_id: str) -> JobRecord | None:
        """The record of job *job_id*; None when there is none. An ended record wins over an active
        one left behind by a writer that stopped between the two."""
        for ended in (True, False):
            try:
                raw = self._record_path(job_id, ended=ended).read_bytes()
            except FileNotFoundError:
                continue
            return JobRecord.model_validate_json(raw)
        return None

    def active(self) -> list[JobRecord]:
        """The records of jobs that have not ended or still hold their reservation, by id."""
        out = []
        for path in (self.root / "active").glob("[0-9]*.json"):
            if not self._record_path(path.name.removesuffix(".json"), ended=True).exists():
                out.append(JobRecord.model_validate_json(path.read_bytes()))
        return sorted(out, key=lambda r: int(r.job_id))

    def write(self, record: JobRecord) -> None:
        """Write *record*, under the exclusive lock: to ``ended/`` once it has ended and holds
        nothing, removing its active copy, else to ``active/``."""
        done = record.state not in ("PENDING", "RUNNING") and not record.holds
        self._write(self._record_path(record.job_id, ended=done), record.model_dump_json(indent=1).encode())
        if done:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(self._record_path(record.job_id, ended=False))

    def host(self) -> HostFacts | None:
        try:
            return HostFacts.model_validate_json((self.root / "host.json").read_bytes())
        except FileNotFoundError:
            return None

    def write_host(self, facts: HostFacts) -> None:
        self._write(self.root / "host.json", facts.model_dump_json(indent=1).encode())

    def write_reservations(self, records: list[JobRecord]) -> None:
        """Rewrite the reservation file from *records*, the active ones."""
        held = {r.job_id: r.reservation.model_dump(mode="json") for r in records if r.holds and r.reservation}
        self._write(self.root / "reservations.json", json.dumps(held, indent=1, sort_keys=True).encode())

    def reservations(self) -> dict[str, Reservation]:
        """The reservation file: what each job that holds resources holds."""
        try:
            raw = json.loads((self.root / "reservations.json").read_bytes())
        except FileNotFoundError:
            return {}
        return {job_id: Reservation.model_validate(value) for job_id, value in raw.items()}
