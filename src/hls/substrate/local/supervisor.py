"""The command line that submits a local job, and the supervisor that runs it.

    python -I -m hls.substrate.local.supervisor submit --state=DIR --script=PATH ...
    python -I -m hls.substrate.local.supervisor run DIR JOB_ID

``submit`` is what ``LocalScheduler.submit`` runs, as sbatch is on Slurm. Under the state
directory's lock it checks the request against the host, takes the next id, keeps a copy of the
batch script, forks the job's supervisor and writes the job's record naming it, and prints the
id. It exits ``EXIT_REFUSED`` with the reason on stderr when it refuses the request, and
``EXIT_DEPENDENCY`` when the job to wait for is not known. The supervisor starts in a session of
its own (``setsid``), with stdin from ``/dev/null`` and its output in ``logs/<id>.log``, and with
an environment of PATH and LANG only, so no variable of the submitting process reaches the job.

``run`` is the supervisor. While its job is PENDING it runs a pass (``hls.substrate.local.schedule``)
every ``poll_sec``. Once a pass gave the job its reservation, it starts the batch script copy with
the interpreter its ``#!`` line names, in a process group of its own, pinned to the reservation's
CPUs, with SIGKILL as its parent-death signal, its output appended to the job's output file, and
an environment of ``JOB_ID_VAR``, ``END_TIME_VAR``, ``STATE_VAR`` and ``TOKEN_VAR`` alone, as a
Slurm job starts with Slurm's variables alone. At the end time, or when it gets SIGTERM (a
cancel), it sends SIGTERM to the batch script's process group and SIGKILL ``kill_wait_sec``
later. The supervisor is a child subreaper, so every process the script leaves behind becomes its
child. Once the script ended it kills what is left of its process group, then every child it has
left until none is, then every process that carries the job's token, then calls the job's stopper,
and only then gives the reservation back and writes the final state:
- COMPLETED for exit status 0, FAILED for any other end, the exit code as "<status>:<signal>";
- TIMEOUT when it stopped the script at the end time, CANCELLED when a cancel did;
- CANCELLED, with no exit code, for a job cancelled before its script started, and FAILED, with
  none, for a script that could not be started."""

from __future__ import annotations

import argparse
import ctypes
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from hls.spec.resources import ResourceShape
from hls.substrate.local.placement import never_fits
from hls.substrate.local.records import RECLAIM_WAIT_SEC, JobRecord, ProcessId, StateDir, StateError
from hls.substrate.local.schedule import TOKEN_VAR, ReclaimFailed, reclaim, run_pass
from hls.substrate.local.stoppers import check_stopper, stopper
from hls.substrate.pidfd import pidfd_open

#: Exit status of ``submit`` that refuses the request.
EXIT_REFUSED = 2
#: Exit status of ``submit`` that refuses the request because the job it waits for is not known.
EXIT_DEPENDENCY = 3
#: The batch script's environment variables: its job id, its end time in integer POSIX seconds,
#: and the state directory.
JOB_ID_VAR, END_TIME_VAR, STATE_VAR = "HLS_JOB_ID", "HLS_JOB_END_TIME", "HLS_LOCAL_STATE"
#: The only environment of ``submit`` and the supervisor.
BASE_ENV = {"PATH": os.defpath, "LANG": "C.UTF-8"}
_PR_SET_PDEATHSIG, _PR_SET_CHILD_SUBREAPER = 1, 36
#: Seconds between a starting supervisor's tries to read its record, before it knows its job's
#: ``poll_sec``.
_START_PAUSE_SEC = 0.1
T = TypeVar("T")


def output_name(job_id: str) -> str:
    """The name of job *job_id*'s output file in the submission's directory."""
    return f"job-{job_id}.out"


def _log(job_id: str, line: str) -> None:
    sys.stderr.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} job {job_id}: {line}\n")
    sys.stderr.flush()


def interpreter(script: bytes) -> list[str]:
    """The interpreter and its one optional argument from the ``#!`` line of *script*, as the kernel
    reads it; raise ValueError when the first line is not ``#!`` and an absolute path."""
    first = script.split(b"\n", 1)[0]
    parts = first[2:].decode("utf-8", "replace").split(None, 1) if first.startswith(b"#!") else []
    if not parts or not parts[0].startswith("/"):
        raise ValueError("the batch script's first line must be #! and the absolute path of an interpreter")
    return [parts[0], *([parts[1].strip()] if len(parts) > 1 and parts[1].strip() else [])]


# submit


def _options(pairs: list[str]) -> dict[str, str]:
    """``--stopper-option`` values, each ``NAME=VALUE``, as a mapping; raise ValueError for one
    without ``=`` or a name given twice."""
    out: dict[str, str] = {}
    for pair in pairs:
        name, sep, value = pair.partition("=")
        if not sep or not name or name in out:
            raise ValueError(f"--stopper-option {pair!r} must be NAME=VALUE, each name once")
        out[name] = value
    return out


def _submit(args: argparse.Namespace) -> int:
    def refuse(code: int, why: str) -> int:
        sys.stderr.write(f"submit: {why}\n")
        return code

    state = StateDir(args.state)
    try:
        shape = ResourceShape(gpus=args.gpus, gpu_kind=args.gpu_kind or None, cpus=args.cpus,
                              memory_gib=args.memory_gib, disk_gib=args.disk_gib, time_sec=args.time_sec)
        check_stopper(args.stopper, _options(args.stopper_option))
        content = args.script.read_bytes()
        interpreter(content)
    except (ValueError, OSError) as e:
        return refuse(EXIT_REFUSED, str(e))
    with state.locked():
        host = state.host()
        if host is None:
            return refuse(EXIT_REFUSED, f"{state.root} holds no inventory of the host")
        problems = never_fits(shape, host)
        if problems:
            return refuse(EXIT_REFUSED, "the host cannot hold the request: " + "; ".join(problems))
        if args.after is not None and state.read(args.after) is None:
            return refuse(EXIT_DEPENDENCY, f"job dependency problem: job {args.after} is not known")
        job_id = state.take_id()
        state.write_script(job_id, content)
        pid = os.fork()
        if pid == 0:
            _become_supervisor(state, job_id)
        supervisor = ProcessId.of(pid, zombie=True)
        if supervisor is None:
            return refuse(1, f"the supervisor of job {job_id} was gone at once")
        state.write(JobRecord(
            job_id=job_id, tag=args.tag, name=args.job_name, script=args.script,
            output=args.output_dir / output_name(job_id), shape=shape, after=args.after,
            singleton=args.singleton, stopper=args.stopper, stopper_options=_options(args.stopper_option),
            kill_wait_sec=args.kill_wait_sec,
            poll_sec=args.poll_sec, token=uuid.uuid4().hex, submitted_at=time.time(), supervisor=supervisor))
    print(job_id)
    return 0


def _become_supervisor(state: StateDir, job_id: str) -> None:
    """In the forked child: start a session, point stdin at ``/dev/null`` and stdout and stderr at
    the job's log, and become ``run``. Never returns."""
    try:
        os.setsid()
        null = os.open(os.devnull, os.O_RDONLY)
        log = os.open(state.log_path(job_id), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(null, 0)
        os.dup2(log, 1)
        os.dup2(log, 2)
        os.execve(sys.executable, [sys.executable, "-I", "-m", "hls.substrate.local.supervisor", "run",
                                   str(state.root), job_id], BASE_ENV)
    finally:
        os._exit(127)


# run


class _Signals:
    """SIGTERM as a flag, with a pipe that wakes a ``select`` when it arrives."""

    def __init__(self) -> None:
        self.read_fd, write_fd = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
        signal.set_wakeup_fd(write_fd)
        self.terminated: float | None = None
        signal.signal(signal.SIGTERM, self._on_term)

    def _on_term(self, signum: int, frame: object) -> None:
        if self.terminated is None:
            self.terminated = time.time()

    def wait(self, timeout: float, *fds: int) -> None:
        """Wait up to *timeout* seconds, or until a signal comes or one of *fds* is readable."""
        ready, _, _ = select.select([self.read_fd, *fds], [], [], max(0.0, timeout))
        if self.read_fd in ready:
            while True:
                try:
                    if not os.read(self.read_fd, 512):
                        break
                except BlockingIOError:
                    break


def _commit(state: StateDir, record: JobRecord) -> None:
    state.write(record)
    state.write_reservations(state.active())


def _pass(state: StateDir, job_id: str) -> None:
    try:
        run_pass(state, now=time.time(), stoppers=stopper, log=lambda line: _log(job_id, f"pass: {line}"))
    except Exception as e:
        _log(job_id, f"pass failed: {type(e).__name__}: {e}")


def _until_done(job_id: str, what: str, step: Callable[[], T], pause_sec: float) -> T:
    """*step*, called again after *pause_sec* for as long as the state directory cannot be used: a
    supervisor that gave up would leave its job to read NODE_FAIL. Each error is logged once in a row."""
    last = None
    while True:
        try:
            return step()
        except (StateError, OSError) as e:
            said = f"{what}: {type(e).__name__}: {e}; trying again"
            if said != last:
                _log(job_id, said)
                last = said
            time.sleep(pause_sec)


def supervise(state: StateDir, job_id: str) -> int:
    """Run job *job_id* to its end. Return 0, or 1 when the job's record does not name this process
    as its supervisor or is gone."""
    try:
        return _supervise(state, job_id)
    except RecordGone as e:
        _log(job_id, str(e))
        return 1


def _supervise(state: StateDir, job_id: str) -> int:
    signals = _Signals()
    # Every process the batch script leaves behind becomes this process's child when its parent
    # ends, whatever session, process group or environment it took, so the end finds them all.
    if ctypes.CDLL(None, use_errno=True).prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
        _log(job_id, f"PR_SET_CHILD_SUBREAPER failed: {os.strerror(ctypes.get_errno())}")
        return 1
    me = ProcessId.of(os.getpid())

    def first() -> JobRecord | None:
        with state.locked(exclusive=False):
            return state.read(job_id)

    record = _until_done(job_id, "the record could not be read", first, _START_PAUSE_SEC)
    if record is None or record.supervisor != me:
        _log(job_id, "its record does not name this process as its supervisor")
        return 1
    _log(job_id, "supervisor started")
    pause = record.poll_sec

    def look() -> JobRecord | None:
        """One pass, then the job's record; None once the job has ended while it waited."""
        with state.locked():
            _pass(state, job_id)
            record = _read(state, job_id)
            if record.state == "PENDING" and signals.terminated is not None:
                record = record.model_copy(update={
                    "state": "CANCELLED", "ended_at": time.time(), "released": True,
                    "cancel_requested_at": record.cancel_requested_at or signals.terminated,
                    "note": "SIGTERM while the job was pending"})
                _commit(state, record)
            if record.state not in ("PENDING", "RUNNING"):
                _log(job_id, f"ended {record.state} while it waited")
                return None
            return record

    while True:
        current = _until_done(job_id, "the queue could not be read", look, pause)
        if current is None:
            return 0
        if current.state == "RUNNING":
            break
        signals.wait(pause)
    _log(job_id, f"holds {current.reservation}; ends at {current.end_time}")
    cause, exit_code, note = _run(state, current, signals)

    def reread() -> JobRecord:
        with state.locked(exclusive=False):
            return _read(state, job_id)

    current = _until_done(job_id, "the record could not be read", reread, pause)
    try:
        _kill_descendants()
        reclaim(current, stopper(current.stopper, current.stopper_options))
        released = True
    except Exception as e:
        released = False
        note = f"{note}; " if note else ""
        note += f"its reservation stays held: {e}"
    final = cause or ("COMPLETED" if exit_code == "0:0" else "FAILED")

    def finish() -> None:
        with state.locked():
            record = _read(state, job_id)
            _commit(state, record.model_copy(update={
                "state": final, "exit_code": exit_code, "ended_at": time.time(), "released": released,
                "note": note}))
            _pass(state, job_id)

    _until_done(job_id, "the final state could not be written", finish, pause)
    _log(job_id, f"ended {final}, exit code {exit_code}" + (f"; {note}" if note else ""))
    return 0


class RecordGone(Exception):
    """The job's record is not in the state directory."""


def _read(state: StateDir, job_id: str) -> JobRecord:
    record = state.read(job_id)
    if record is None:
        raise RecordGone(f"the record of job {job_id} is gone")
    return record


def _run(state: StateDir, record: JobRecord, signals: _Signals) -> tuple[str | None, str | None, str | None]:
    """Start the batch script and watch it to its end; the job's cause of end (CANCELLED, TIMEOUT,
    FAILED, or None to go by the exit code), its exit code and a note."""
    if signals.terminated is not None or record.cancel_requested_at is not None:
        return "CANCELLED", None, "cancelled before the batch script started"
    assert record.reservation is not None and record.end_time is not None
    try:
        proc = _start(state, record)
    except (OSError, ValueError, subprocess.SubprocessError) as e:
        return "FAILED", None, f"the batch script could not be started: {type(e).__name__}: {e}"
    runner = ProcessId.of(proc.pid, zombie=True)
    try:
        with state.locked():
            current = _read(state, record.job_id)
            state.write(current.model_copy(update={"runner": runner}))
    except (StateError, OSError) as e:
        # Only people read it: the job's processes are found by its token.
        _log(record.job_id, f"the batch script's pid could not be recorded: {e}")
    _log(record.job_id, f"batch script started as pid {proc.pid}")
    # Readable once the script exits, which ends a wait early; a kernel without pidfds gives
    # none, and each wait then runs to its timeout.
    pidfd = pidfd_open(proc.pid)
    woken = () if pidfd is None else (pidfd,)
    cause: str | None = None
    term_at: float | None = None
    killed = False
    next_look = time.monotonic() + record.poll_sec
    try:
        while os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
            now = time.time()
            if term_at is None:
                if time.monotonic() >= next_look:
                    next_look = time.monotonic() + record.poll_sec
                    if _cancel_asked(state, record.job_id):
                        signals.terminated = signals.terminated or now
                if signals.terminated is not None or now >= record.end_time:
                    cause = "CANCELLED" if signals.terminated is not None else "TIMEOUT"
                    term_at = now
                    _log(record.job_id, f"SIGTERM to the batch script ({cause})")
                    _signal_group(proc.pid, signal.SIGTERM)
                    continue
                timeout = min(record.end_time - now, next_look - time.monotonic())
            elif killed:
                timeout = record.poll_sec
            elif now >= term_at + record.kill_wait_sec:
                _log(record.job_id, "SIGKILL to the batch script")
                _signal_group(proc.pid, signal.SIGKILL)
                killed = True
                timeout = record.poll_sec
            else:
                timeout = term_at + record.kill_wait_sec - now
            signals.wait(timeout, *woken)
            _reap_orphans(proc.pid)
        info = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOWAIT)
        # The script has exited and is not reaped, so its pid and its process group's id are
        # still its own: what is left of its group is killed before the pid is given back.
        _signal_group(proc.pid, signal.SIGKILL)
        os.waitpid(proc.pid, 0)
    finally:
        if pidfd is not None:
            os.close(pidfd)
    assert info is not None
    exited = info.si_code == os.CLD_EXITED
    proc.returncode = info.si_status if exited else -info.si_status
    return cause, f"{info.si_status}:0" if exited else f"0:{info.si_status}", None


def _reap_orphans(runner: int) -> None:
    """Reap every child that has exited but the batch script: processes it left that became this
    process's children."""
    while True:
        try:
            info = os.waitid(os.P_ALL, 0, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError:
            return
        if info is None or info.si_pid == runner:
            return
        os.waitpid(info.si_pid, 0)


def _children(proc: Path = Path("/proc")) -> list[int]:
    """The pids of this process's children, from each process's ``stat``."""
    me = os.getpid()
    out = []
    for entry in os.scandir(proc):
        if not entry.name.isdecimal():
            continue
        try:
            raw = (proc / entry.name / "stat").read_bytes()
        except OSError:
            continue
        if int(raw[raw.rindex(b")") + 2:].split()[1]) == me:
            out.append(int(entry.name))
    return out


def _kill_descendants(wait_sec: float = RECLAIM_WAIT_SEC) -> None:
    """SIGKILL every child of this process and reap it, until none is left: a killed child's own
    children become this process's in turn. A child's pid stays its own until it is reaped, so no
    other process is signalled. Raise ReclaimFailed when children remain after *wait_sec*."""
    deadline = time.monotonic() + wait_sec
    while True:
        children = _children()
        if not children:
            return
        for pid in children:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for pid in children:
            try:
                os.waitpid(pid, os.WNOHANG)
            except ChildProcessError:
                pass
        if time.monotonic() >= deadline:
            raise ReclaimFailed(f"processes {children} the batch script left are still there {wait_sec:.0f}s "
                                "after SIGKILL")
        time.sleep(0.02)


def _cancel_asked(state: StateDir, job_id: str) -> bool:
    """Whether the job's record says a cancel was asked for, in case its SIGTERM was lost; False
    when the record cannot be read now, since the SIGTERM still stops the job."""
    try:
        with state.locked(exclusive=False):
            return _read(state, job_id).cancel_requested_at is not None
    except (StateError, OSError) as e:
        _log(job_id, f"the record could not be read: {e}")
        return False


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _start(state: StateDir, record: JobRecord) -> subprocess.Popen[bytes]:
    script = state.script_path(record.job_id)
    argv = [*interpreter(script.read_bytes()), str(script)]
    assert record.reservation is not None and record.end_time is not None
    env = {JOB_ID_VAR: record.job_id, END_TIME_VAR: str(int(record.end_time)), STATE_VAR: str(state.root),
           TOKEN_VAR: record.token}
    cpus = set(record.reservation.cpus)
    parent = os.getpid()
    libc = ctypes.CDLL(None, use_errno=True)

    def setup() -> None:
        # Runs in the child between fork and exec; the supervisor has no other thread.
        if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
            raise OSError("the parent-death signal could not be set")
        os.sched_setaffinity(0, cpus)

    out = os.open(record.output, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o666)
    try:
        return subprocess.Popen(argv, env=env, cwd="/", stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, process_group=0, preexec_fn=setup)
    finally:
        os.close(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls.substrate.local.supervisor")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--state", type=Path, required=True)
    s.add_argument("--script", type=Path, required=True)
    s.add_argument("--tag", required=True)
    s.add_argument("--job-name", required=True)
    s.add_argument("--output-dir", type=Path, required=True)
    s.add_argument("--gpus", type=int, required=True)
    s.add_argument("--gpu-kind", default="")
    s.add_argument("--cpus", type=int, required=True)
    s.add_argument("--memory-gib", type=int, required=True)
    s.add_argument("--disk-gib", type=int, required=True)
    s.add_argument("--time-sec", type=int, required=True)
    s.add_argument("--after")
    s.add_argument("--singleton", action="store_true")
    s.add_argument("--stopper", required=True)
    s.add_argument("--stopper-option", action="append", default=[], metavar="NAME=VALUE")
    s.add_argument("--kill-wait-sec", type=float, required=True)
    s.add_argument("--poll-sec", type=float, required=True)
    r = sub.add_parser("run")
    r.add_argument("state", type=Path)
    r.add_argument("job_id")
    args = p.parse_args(argv)
    if args.command == "submit":
        try:
            return _submit(args)
        except StateError as e:
            sys.stderr.write(f"submit: {e}\n")
            return 1
    return supervise(StateDir(args.state), args.job_id)


if __name__ == "__main__":
    sys.exit(main())
