"""Locks that work across nodes of a shared filesystem, with liveness and takeover.

A lock is a directory of generation files under its key. The highest generation is
the current one. Taking the lock means creating the next generation with a
create-only write, which only one racer can win. That is allowed only when the
current generation is released, its heartbeat is older than its TTL, or the caller's
liveness check says the holder is dead. An unknown liveness answer counts as alive.
The holder renews its own generation file. A holder whose generation was taken over
gets LockLost on its next renew and must stop.

Heartbeats are written with the holder's clock and judged with the waiter's, so a
takeover is safe only while clocks across nodes differ by less than a quarter of the
TTL. The holder counts itself lost 0.75 TTL after its last heartbeat by its own clock,
and a waiter takes over 1.0 TTL after it by the waiter's clock. A waiter whose clock
runs at least 0.25 TTL ahead (30 s for a 120 s TTL) could take over before the holder
stops."""

from __future__ import annotations

import math
import os
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager

from pydantic import BaseModel, ConfigDict, Field

from hls.store.base import AlreadyExists, Store, StoreError, check_key
from hls.store.layout import InstanceLayout

# The name of a generation file directly under the lock's key.
_GEN_FILE = re.compile(r"gen-(\d{8})\.json")


def _boot_id() -> str | None:
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            return f.read().strip() or None
    except OSError:
        return None


def _pid_ns() -> int | None:
    try:
        return os.stat("/proc/self/ns/pid").st_ino
    except OSError:
        return None


class Holder(BaseModel):
    """Who holds a lock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Hostname of the holding process.
    host: str
    pid: int
    #: Its Slurm job id, when it runs inside a job.
    job_id: str | None
    #: What the lock is held for, for people reading the store.
    purpose: str
    #: Kernel boot id of the host, so a pid from before a reboot is not trusted. None if unknown.
    boot_id: str | None = None
    #: Inode of the process's PID namespace. A pid means something only inside its own
    #: namespace, and a hostname does not identify one. None if unknown.
    pid_ns: int | None = None

    @classmethod
    def here(cls, purpose: str, job_id: str | None = None) -> Holder:
        """The holder record for the current process."""
        return cls(
            host=socket.gethostname(),
            pid=os.getpid(),
            job_id=job_id,
            purpose=purpose,
            boot_id=_boot_id(),
            pid_ns=_pid_ns(),
        )


class LockRecord(BaseModel):
    """One generation of a lock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Generation number, from 1.
    generation: int = Field(ge=1)
    holder: Holder
    #: When the generation was created.
    acquired_at: float
    #: The holder's last renew.
    heartbeat_at: float
    #: Seconds after the last heartbeat at which others may take over.
    ttl_sec: float = Field(gt=0)
    #: True once the holder let go.
    released: bool = False


Liveness = Callable[[Holder], bool | None]


class LockHeld(StoreError):
    """The lock is held by someone alive."""

    def __init__(self, record: LockRecord, key: str) -> None:
        h = record.holder
        super().__init__(
            f"{key}: generation {record.generation} is held by {h.purpose} on {h.host} pid {h.pid}"
        )
        #: The current generation.
        self.record = record
        #: Directory key of the lock.
        self.key = key


class LockLost(StoreError):
    """Another holder took the lock over, or the heartbeat lapsed so it may have."""


def same_host_liveness(holder: Holder) -> bool | None:
    """Whether a holder is alive, by pid. A pid is only meaningful on the same host, in the
    same boot and in the same PID namespace, so the answer is None unless the holder
    recorded all three and they match this process's. Without them a live holder in
    another namespace would look dead."""
    if holder.boot_id is None or holder.pid_ns is None:
        return None
    if (holder.host, holder.boot_id, holder.pid_ns) != (socket.gethostname(), _boot_id(), _pid_ns()):
        return None
    try:
        os.kill(holder.pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class GenerationLock:
    """A lock at one store key."""

    def __init__(
        self, store: Store, key: str, *, ttl_sec: float, clock: Callable[[], float] = time.time
    ) -> None:
        #: Store the generation files live in.
        self.store = store
        #: Directory key of the lock.
        self.key = check_key(key)
        #: TTL written into each generation.
        self.ttl_sec = ttl_sec
        #: Time source.
        self.clock = clock

    def _gen_key(self, generation: int) -> str:
        return f"{self.key}/gen-{generation:08d}.json"

    def current(self) -> LockRecord | None:
        """The highest generation, or None if the lock was never taken."""
        names = self.store.list_children(self.key)
        gens = [int(m.group(1)) for name in names if (m := _GEN_FILE.fullmatch(name))]
        if not gens:
            return None
        return LockRecord.model_validate_json(self.store.read_bytes(self._gen_key(max(gens))))

    def _free(self, rec: LockRecord, alive: Liveness | None, now: float) -> bool:
        if rec.released or now - rec.heartbeat_at > rec.ttl_sec:
            return True
        return alive is not None and alive(rec.holder) is False

    def try_acquire(self, holder: Holder, *, alive: Liveness | None = None) -> LockHandle:
        """Take the lock now, or raise LockHeld.

        Expiry is judged by the clock read before the record. A read that returns late may
        hold an old heartbeat, and a later clock would then take over a holder that has
        renewed since."""
        while True:
            now = self.clock()
            cur = self.current()
            if cur is not None and not self._free(cur, alive, now):
                raise LockHeld(cur, self.key)
            gen = 1 if cur is None else cur.generation + 1
            rec = LockRecord(
                generation=gen, holder=holder, acquired_at=now, heartbeat_at=now, ttl_sec=self.ttl_sec
            )
            try:
                self.store.create(self._gen_key(gen), rec.model_dump_json().encode())
            except AlreadyExists:
                continue
            return LockHandle(self, gen, holder, now)

    def acquire(
        self,
        holder: Holder,
        *,
        alive: Liveness | None = None,
        wait_sec: float,
        poll_sec: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> LockHandle:
        """Take the lock, waiting up to *wait_sec* for it to come free; LockHeld if it
        does not."""
        deadline = self.clock() + wait_sec
        while True:
            try:
                return self.try_acquire(holder, alive=alive)
            except LockHeld:
                if self.clock() >= deadline:
                    raise
                sleep(poll_sec)


# A keep-alive block waits this long past one interval for its thread at the end.
_JOIN_MARGIN_SEC = 0.1
# The fraction of the TTL after the last heartbeat at which a lease counts as lost, early
# enough that the holder can stop before anyone else may take over.
_LOST_AFTER_TTL = 0.75


class LockHandle:
    """A held generation."""

    def __init__(self, lock: GenerationLock, generation: int, holder: Holder, heartbeat_at: float) -> None:
        self.lock = lock
        #: The generation this handle holds.
        self.generation = generation
        self.holder = holder
        #: Serializes renew and release, so an in-flight renew cannot undo a release.
        self._guard = threading.Lock()
        #: Set by release; renew does nothing after it.
        self._released = False
        #: The heartbeat this handle last wrote, as the lock's clock read it.
        self._heartbeat_at = heartbeat_at
        #: The first loss this handle saw: a takeover, a heartbeat that landed late, or a last
        #: heartbeat that grew too old. Once set it stays, and every later check raises it.
        self._lost: LockLost | None = None
        #: The last error a keep-alive renew failed with, cleared by the next success.
        self._last_error: Exception | None = None

    def _latch(self, lost: LockLost) -> LockLost:
        if self._lost is None:
            self._lost = lost
        return self._lost

    @property
    def lost(self) -> LockLost | None:
        """Why this handle should stop, or None while the lock looks held.

        That is a loss already seen, or a last heartbeat older than three quarters of the
        TTL. The age is measured from the heartbeat itself, so a renew that fails slowly or
        hangs is caught too. A loss is final: it is kept, and a later heartbeat does not
        clear it. A released handle that never lost the lock is not lost."""
        if self._lost is not None:
            return self._lost
        if self._released:
            return None
        silent = self.lock.clock() - self._heartbeat_at
        if silent < _LOST_AFTER_TTL * self.lock.ttl_sec:
            return None
        lost = LockLost(f"{self.lock.key}: generation {self.generation} not renewed for {silent:.1f}s")
        lost.__cause__ = self._last_error
        return self._latch(lost)

    def check(self) -> None:
        """Raise ``lost`` if there is one. A long block can call this to stop early. It
        reads no files, so it cannot see a takeover that the heartbeat has not shown yet;
        use ``confirm`` before an action that needs the lock."""
        lost = self.lost
        if lost is not None:
            raise lost

    def confirm(self) -> None:
        """Raise ``lost`` if there is one, and otherwise check that this handle's generation
        is still the lock's current one, raising LockLost if not. Call it right before an
        action that needs the lock. It does not write, and costs one directory read plus
        one small file read.

        The heartbeat age is checked again after the read. A read that returns late may still
        show this generation as current after the lease expired and someone took over."""
        self.check()
        self._mine()
        self.check()

    def _mine(self) -> LockRecord:
        cur = self.lock.current()
        if cur is None or cur.generation != self.generation:
            raise self._latch(LockLost(f"{self.lock.key}: generation {self.generation} was taken over"))
        return cur

    def renew(self) -> None:
        """Move the heartbeat to now; LockLost if the lock was taken over or lost. Does
        nothing after release.

        A heartbeat that lands a long time after the last one is refused, because the lock
        may have been taken over in between and a late write cannot tell. The refusal is
        final, like any loss."""
        with self._guard:
            if self._released:
                return
            self.check()
            cur = self._mine()
            beat = self.lock.clock()
            rec = cur.model_copy(update={"heartbeat_at": beat})
            self.lock.store.write_bytes(self.lock._gen_key(self.generation), rec.model_dump_json().encode())
            age = self.lock.clock() - self._heartbeat_at
            if age >= _LOST_AFTER_TTL * self.lock.ttl_sec:
                raise self._latch(
                    LockLost(
                        f"{self.lock.key}: generation {self.generation} heartbeat landed {age:.1f}s "
                        "after the last one; another holder may have taken over"
                    )
                )
            self._heartbeat_at = beat

    def release(self) -> None:
        """Let go; a lock that was already taken over is left alone.

        A renew stuck in the store holds the guard. Release waits one TTL for it, which is
        longer than the lock can stay held without a heartbeat, and then gives up."""
        self._released = True
        if not self._guard.acquire(timeout=self.lock.ttl_sec):
            return
        try:
            try:
                cur = self._mine()
            except LockLost:
                return
            rec = cur.model_copy(update={"released": True})
            self.lock.store.write_bytes(self.lock._gen_key(self.generation), rec.model_dump_json().encode())
        finally:
            self._guard.release()

    def __enter__(self) -> LockHandle:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def keep_alive(self, interval_sec: float) -> AbstractContextManager[LockHandle]:
        """Renew every *interval_sec* in a background thread while the block runs.

        A failed renew other than LockLost is retried at the next interval. Whether the
        lock is lost is decided from the last heartbeat (see ``lost``), whatever the
        thread is doing, and ``check()`` lets the block poll it. When the block ends without
        an error of its own, it calls ``confirm()``, so a loss, or a takeover only visible
        in the store, is raised then. The block end waits only briefly for a renew stuck in
        the store; the daemon thread is then abandoned. ``confirm`` then lists the lock's
        directory and reads the current generation file, and the block end is not bounded
        if either hangs.

        The interval must be finite, positive and at most a third of the TTL, so several
        renews fit before the lock expires."""
        ttl = self.lock.ttl_sec
        if not math.isfinite(interval_sec) or interval_sec <= 0:
            raise ValueError(f"keep-alive interval must be finite and positive, got {interval_sec}s")
        if interval_sec * 3 > ttl and not math.isclose(interval_sec * 3, ttl):
            raise ValueError(f"keep-alive interval {interval_sec}s is over a third of the {ttl}s TTL")
        return self._keep_alive(interval_sec)

    @contextmanager
    def _keep_alive(self, interval_sec: float) -> Iterator[LockHandle]:
        stop = threading.Event()

        def loop() -> None:
            while not stop.wait(interval_sec):
                try:
                    self.renew()
                except LockLost:
                    return
                except Exception as e:
                    self._last_error = e
                else:
                    self._last_error = None

        t = threading.Thread(target=loop, name="hls-lock-keepalive", daemon=True)
        t.start()
        try:
            yield self
        finally:
            stop.set()
            t.join(interval_sec + _JOIN_MARGIN_SEC)
        self.confirm()


def instance_lease(
    store: Store, layout: InstanceLayout, *, ttl_sec: float, clock: Callable[[], float] = time.time
) -> GenerationLock:
    """The lock one orchestrator holds while it drives an instance."""
    return GenerationLock(store, layout.lease, ttl_sec=ttl_sec, clock=clock)
