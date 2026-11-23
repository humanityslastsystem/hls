"""One pass over the queue, run under the state directory's exclusive lock by every supervisor
while it waits and once its job ended.

A pass:
1. ends every job whose record says PENDING or RUNNING and whose supervisor is gone: CANCELLED
   when a cancel was asked for, NODE_FAIL otherwise;
2. reclaims what each ended job whose supervisor is gone still holds: it kills every process that
   carries the job's token in its environment, calls the job's stopper, and only then gives the
   reservation back. A reclaim that fails keeps the reservation held and is tried again
   ``RECLAIM_BACKOFF_SEC`` later;
3. gives reservations to pending jobs in the order they were submitted. A job may start once the
   job it waits for (``after``) has COMPLETED and, for a singleton, no other job of its name runs.
   A job's GPUs are cards no other job holds and no process computes on, as the host's nvidia-smi
   (``HostFacts.nvidia_smi``) lists the compute processes at this pass
   (``hls.substrate.grant.gpus_computing``): a card that no job
   holds and that holds a process holds one hls did not start, and is never given. A job that asks
   for more than are free waits, its reason saying how many are, and is looked at again at the
   next pass, with no time limit but its cancel, as a Slurm job waits in its queue. When the
   GPUs' processes cannot be read, no card is free.
   The first job that does not fit holds back every later one that would not end before the
   time it can start, from the end times of the jobs that hold resources (backfill), so a large
   job is never starved by small ones. A job that would not fit even once those jobs ended holds
   back nothing, and the next one that does not fit holds the others back in its place.

A job given its reservation is RUNNING from then on: its end time is the pass's time plus the
shape's time, and its supervisor starts the batch script when it next looks."""

from __future__ import annotations

import os
import signal
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from hls.substrate.base import JobState
from hls.substrate.grant import gpus_computing
from hls.substrate.local.inventory import HostFacts
from hls.substrate.local.placement import Reservation, free_gpus, place
from hls.substrate.local.records import (RECLAIM_BACKOFF_SEC, RECLAIM_WAIT_SEC, JobRecord, StateDir,
                                        reported_state)
from hls.substrate.local.stoppers import Stopper
from hls.substrate.pidfd import pidfd_open, pidfd_send_signal

#: The environment variable that labels every process of a job with the job's token.
TOKEN_VAR = "HLS_LOCAL_JOB_TOKEN"


class ReclaimFailed(RuntimeError):
    """Something of a job may still run, so its reservation stays held."""


def labelled(token: str, *, proc: Path = Path("/proc")) -> list[int]:
    """The pids of this user's processes whose environment holds ``TOKEN_VAR=<token>``, but this
    process's. A process whose environment cannot be read (another user's, or one that made itself
    not dumpable) is not among them."""
    needle = f"{TOKEN_VAR}={token}".encode()
    out = []
    for entry in os.scandir(proc):
        if not entry.name.isdecimal() or int(entry.name) == os.getpid():
            continue
        if _carries(proc / entry.name, needle):
            out.append(int(entry.name))
    return sorted(out)


def _carries(path: Path, needle: bytes) -> bool:
    try:
        return needle in (path / "environ").read_bytes().split(b"\0")
    except OSError:
        return False


def kill_labelled(token: str, *, wait_sec: float = RECLAIM_WAIT_SEC, proc: Path = Path("/proc")) -> None:
    """SIGKILL every process ``labelled`` with *token*, each through a pidfd taken before its label
    is read again, so no other process is signalled, and wait until none is left. Raise
    ReclaimFailed when one is still there after *wait_sec*."""
    needle = f"{TOKEN_VAR}={token}".encode()
    deadline = time.monotonic() + wait_sec
    while True:
        pids = labelled(token, proc=proc)
        if not pids:
            return
        for pid in pids:
            try:
                fd = pidfd_open(pid)
            except ProcessLookupError:
                continue
            if fd is None:
                raise ReclaimFailed(f"this kernel has no pidfds, so process {pid} of the job cannot be "
                                    "signalled without the risk of reaching another")
            try:
                if _carries(proc / str(pid), needle):
                    pidfd_send_signal(fd, signal.SIGKILL)
            except ProcessLookupError:
                pass
            finally:
                os.close(fd)
        if time.monotonic() >= deadline:
            raise ReclaimFailed(f"processes {pids} of the job are still there {wait_sec:.0f}s after SIGKILL")
        time.sleep(0.05)


def reclaim(record: JobRecord, stopper: Stopper, *, wait_sec: float = RECLAIM_WAIT_SEC) -> None:
    """Make sure nothing of job *record* runs: kill its labelled processes, then call its stopper.
    Raise ReclaimFailed when that cannot be confirmed."""
    kill_labelled(record.token, wait_sec=wait_sec)
    try:
        stopper.stop(record.job_id)
    except Exception as e:
        raise ReclaimFailed(f"stopper {record.stopper}: {type(e).__name__}: {e}") from e


def dependency_reason(record: JobRecord, parent: JobRecord | None) -> str | None:
    """Why *record* waits for its dependency, as Slurm names it: "Dependency" while the job it waits
    for has not ended, "DependencyNeverSatisfied" once that job ended other than COMPLETED or is not
    known; None when there is no dependency or it is satisfied."""
    if record.after is None:
        return None
    state = JobState.UNKNOWN if parent is None else reported_state(parent)
    if state is JobState.COMPLETED:
        return None
    if parent is None or state.terminal:
        return "DependencyNeverSatisfied"
    return "Dependency"


def run_pass(state: StateDir, *, now: float, stoppers: Callable[[str, Mapping[str, str]], Stopper],
             log: Callable[[str], None] = lambda line: None,
             in_use: Callable[[], set[str] | None] | None = None) -> None:
    """One pass, under *state*'s exclusive lock. *stoppers* makes the stopper of a stopper name and
    its options (``hls.substrate.local.stoppers.stopper``). *in_use* reads the GPUs of the host a
    process computes on (None: they cannot be read), asked only when a pending job wants GPUs."""
    before = state.active()
    records = {r.job_id: r for r in before}
    for job_id, r in list(records.items()):
        if r.state in ("PENDING", "RUNNING") and r.supervisor.alive() is False:
            ended = "CANCELLED" if r.cancel_requested_at is not None else "NODE_FAIL"
            records[job_id] = r = r.model_copy(update={
                "state": ended, "ended_at": now, "released": r.reservation is None,
                "note": f"its supervisor was gone while the job was {r.state}"})
            log(f"job {job_id}: {ended}, its supervisor is gone")
        if (r.holds and r.state not in ("PENDING", "RUNNING") and r.supervisor.alive() is False
                and (r.reclaim_after is None or now >= r.reclaim_after)):
            try:
                reclaim(r, stoppers(r.stopper, r.stopper_options))
            except Exception as e:
                # Whatever stopped the reclaim, a stopper that could not be made among it, something
                # of the job may still run, so it keeps its reservation.
                records[job_id] = r.model_copy(update={"reclaim_after": now + RECLAIM_BACKOFF_SEC})
                log(f"job {job_id}: its reservation stays held: {e}")
                continue
            records[job_id] = r.model_copy(update={"released": True, "reclaim_after": None})
            log(f"job {job_id}: reclaimed")
    host = state.host()
    if host is not None:
        _assign(state, records, now=now, host=host, in_use=in_use or (lambda: _gpus_computing(host)))
    old = {r.job_id: r for r in before}
    changed = [r for r in records.values() if r != old.get(r.job_id)]
    for r in changed:
        state.write(r)
    if changed:
        state.write_reservations(list(records.values()))


def _gpus_computing(host: HostFacts) -> set[str] | None:
    """The GPUs *host*'s nvidia-smi lists a compute process on; None when it has none or it cannot
    say."""
    return None if host.nvidia_smi is None else gpus_computing(host.nvidia_smi)


def _assign(state: StateDir, records: dict[str, JobRecord], *, now: float, host: HostFacts,
            in_use: Callable[[], set[str] | None]) -> None:
    busy: frozenset[str] = frozenset()
    unread = False
    if host.gpus and any(r.state == "PENDING" and r.shape.gpus for r in records.values()):
        found = in_use()
        unread = found is None
        busy = frozenset(g.uuid for g in host.gpus) if found is None else frozenset(found)
    held: list[Reservation] = []
    releases: list[tuple[float, Reservation]] = []
    running: set[str] = set()
    for r in records.values():
        if r.holds and r.reservation is not None:
            held.append(r.reservation)
            if r.state == "RUNNING" and r.end_time is not None:
                releases.append((r.end_time + r.kill_wait_sec, r.reservation))
        if r.state == "RUNNING":
            running.add(r.name)
    shadow: float | None = None
    for job_id in sorted(records, key=int):
        r = records[job_id]
        if r.state != "PENDING" or r.supervisor.alive() is not True:
            continue
        parent = None if r.after is None else (records.get(r.after) or state.read(r.after))
        reason = dependency_reason(r, parent)
        if reason is None and r.singleton and r.name in running:
            reason = "Dependency"
        if reason is None:
            reservation = place(r.shape, host, held, busy)
            ends = now + r.shape.time_sec + r.kill_wait_sec
            if reservation is not None and (shadow is None or ends <= shadow):
                records[job_id] = r.model_copy(update={
                    "state": "RUNNING", "reason": None, "started_at": now, "end_time": now + r.shape.time_sec,
                    "reservation": reservation})
                held.append(reservation)
                releases.append((ends, reservation))
                running.add(r.name)
                continue
            if shadow is None:
                shadow = _earliest_fit(r, host, held, releases, busy)
                reason = "Resources"
            else:
                reason = "Resources" if reservation is None else "Priority"
            if reservation is None:
                reason = _gpu_wait(r, host, held, busy, unread=unread) or reason
        if r.reason != reason:
            records[job_id] = r.model_copy(update={"reason": reason})


def _gpu_wait(record: JobRecord, host: HostFacts, held: list[Reservation], busy: frozenset[str], *,
              unread: bool) -> str | None:
    """Why *record* waits for GPUs, with how many of its kind are free, when fewer are free than it
    asks for; None when its GPUs are not what it waits for."""
    shape = record.shape
    if not shape.gpus:
        return None
    free = free_gpus(shape, host, held, busy)
    if len(free) >= shape.gpus:
        return None
    kind = [g.uuid for g in host.gpus if g.kind == shape.gpu_kind]
    if unread:
        return (f"waiting for GPUs: none of the host's {len(kind)} {shape.gpu_kind} GPUs is free, since their compute "
                f"processes cannot be read (nvidia-smi failed); it asks for {shape.gpus}")
    taken = {u for r in held for u in r.gpus}
    others = len([u for u in kind if u in busy and u not in taken])
    return (f"waiting for GPUs: {len(free)} of the host's {len(kind)} {shape.gpu_kind} GPUs are free and it asks for "
            f"{shape.gpus}" + (f"; {others} hold a process hls did not start" if others else ""))


def _earliest_fit(record: JobRecord, host: HostFacts, held: list[Reservation],
                  releases: list[tuple[float, Reservation]], busy: frozenset[str]) -> float | None:
    """When *record* fits at the latest, as the reservations of *releases* are given back at their
    times and with the GPUs other users compute on (*busy*) as busy as now; None when it does not
    fit even once they all are, in which case it holds back nothing."""
    remaining = list(held)
    for at, reservation in sorted(releases, key=lambda x: x[0]):
        remaining.remove(reservation)
        if place(record.shape, host, remaining, busy) is not None:
            return at
    return None
