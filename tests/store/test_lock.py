from __future__ import annotations

import math
import multiprocessing as mp
import os
import socket
import threading
import time
from pathlib import Path

import pytest

from hls.store.layout import InstanceLayout
from hls.store.lock import GenerationLock, Holder, LockHeld, LockLost, instance_lease, same_host_liveness
from hls.store.posix import PosixStore


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


class _FlakyStore(PosixStore):
    """A store whose writes fail with ESTALE a set number of times (inf for always)."""

    def __init__(self, root: Path, failures: float) -> None:
        super().__init__(root)
        self.failures_left = failures

    def write_bytes(self, key: str, data: bytes) -> None:
        if self.failures_left > 0:
            self.failures_left -= 1
            raise OSError(116, "Stale file handle")
        super().write_bytes(key, data)


class _SlowFailStore(PosixStore):
    """A store whose writes take a while and then fail with ESTALE."""

    def __init__(self, root: Path, delay_sec: float) -> None:
        super().__init__(root)
        self.delay_sec = delay_sec

    def write_bytes(self, key: str, data: bytes) -> None:
        time.sleep(self.delay_sec)
        raise OSError(116, "Stale file handle")


class _BlockedStore(PosixStore):
    """A store whose writes wait on a gate, like a hung filesystem call."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.gate = threading.Event()
        self.entered = threading.Event()

    def write_bytes(self, key: str, data: bytes) -> None:
        self.entered.set()
        self.gate.wait(30)
        super().write_bytes(key, data)


def _wait_for(cond, timeout: float = 1.5) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(0.01)
    return False


def _holder(name: str) -> Holder:
    return Holder(host="h", pid=1, job_id=None, purpose=name)


def test_first_acquire_takes_generation_one(tmp_path):
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=60, clock=Clock())
    h = lock.try_acquire(_holder("a"))
    assert h.generation == 1 and lock.current().holder.purpose == "a"


def test_live_holder_blocks(tmp_path):
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=60, clock=Clock())
    lock.try_acquire(_holder("a"))
    with pytest.raises(LockHeld) as e:
        lock.try_acquire(_holder("b"))
    assert e.value.record.holder.purpose == "a"


def test_expired_heartbeat_allows_takeover_and_old_holder_loses(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=60, clock=clock)
    old = lock.try_acquire(_holder("a"))
    clock.t += 61
    new = lock.try_acquire(_holder("b"))
    assert new.generation == 2
    with pytest.raises(LockLost):
        old.renew()


def test_dead_holder_allows_takeover_unknown_does_not(tmp_path):
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=60, clock=Clock())
    lock.try_acquire(_holder("a"))
    with pytest.raises(LockHeld):
        lock.try_acquire(_holder("b"), alive=lambda h: None)
    assert lock.try_acquire(_holder("b"), alive=lambda h: False).generation == 2


def test_release_then_reacquire(tmp_path):
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=60, clock=Clock())
    with lock.try_acquire(_holder("a")):
        pass
    assert lock.current().released
    assert lock.try_acquire(_holder("b")).generation == 2


def test_renew_moves_heartbeat(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    clock.t += 40
    h.renew()
    clock.t += 50
    with pytest.raises(LockHeld):
        lock.try_acquire(_holder("b"))


def test_acquire_waits_then_times_out(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=600, clock=clock)
    lock.try_acquire(_holder("a"))

    def fake_sleep(sec: float) -> None:
        clock.t += sec

    with pytest.raises(LockHeld):
        lock.acquire(_holder("b"), wait_sec=30, poll_sec=10, sleep=fake_sleep)
    assert clock.t >= 1030


def _only_the_original_is_dead(h: Holder) -> bool | None:
    # the racers themselves are alive; only the first holder is known dead
    return False if h.purpose == "dead" else None


def _takeover(root: str, q) -> None:
    lock = GenerationLock(PosixStore(Path(root)), "locks/race", ttl_sec=60)
    try:
        q.put(
            lock.try_acquire(
                Holder(host="h", pid=os.getpid(), job_id=None, purpose="r"), alive=_only_the_original_is_dead
            ).generation
        )
    except LockHeld:
        q.put(None)


def test_concurrent_takeover_has_one_winner(tmp_path):
    GenerationLock(PosixStore(tmp_path), "locks/race", ttl_sec=60).try_acquire(_holder("dead"))
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_takeover, args=(str(tmp_path), q)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
    got = [q.get(timeout=5) for _ in procs]
    assert sorted(g for g in got if g is not None) == [2]


def test_keep_alive_holds_the_lock_past_its_ttl(tmp_path):
    # the one test on real time: a fill longer than the TTL keeps the lock
    lock = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=0.5)
    with lock.try_acquire(_holder("a")) as h, h.keep_alive(interval_sec=0.1):
        time.sleep(1.2)
        with pytest.raises(LockHeld):
            lock.try_acquire(_holder("b"))


def test_keep_alive_reports_a_lost_lock(tmp_path):
    clock = Clock()
    store = PosixStore(tmp_path)
    lock = GenerationLock(store, "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    clock.t += 61
    lock.try_acquire(_holder("b"))
    with pytest.raises(LockLost):
        with h.keep_alive(interval_sec=0.05):
            time.sleep(0.3)


def test_keep_alive_rides_out_a_transient_write_error(tmp_path):
    store = _FlakyStore(tmp_path, failures=1)
    lock = GenerationLock(store, "locks/k", ttl_sec=0.5)
    h = lock.try_acquire(_holder("a"))
    first = lock.current().heartbeat_at
    with h.keep_alive(interval_sec=0.1):
        time.sleep(0.35)
        assert h.lost is None
    assert store.failures_left == 0 and h.lost is None
    assert lock.current().heartbeat_at > first
    with pytest.raises(LockHeld):
        lock.try_acquire(_holder("b"))


def test_keep_alive_declares_loss_when_renewals_keep_failing(tmp_path):
    lock = GenerationLock(_FlakyStore(tmp_path, failures=math.inf), "locks/k", ttl_sec=0.5)
    h = lock.try_acquire(_holder("a"))
    with pytest.raises(LockLost) as e:
        with h.keep_alive(interval_sec=0.1):
            assert _wait_for(lambda: h.lost is not None)
            with pytest.raises(LockLost):
                h.check()
    assert isinstance(e.value.__cause__, OSError)
    assert "locks/k" in str(e.value) and "generation 1" in str(e.value)


def test_a_takeover_is_visible_during_the_block(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    h.check()
    assert h.lost is None
    lock.try_acquire(_holder("b"), alive=lambda holder: False)
    assert h.lost is None
    with pytest.raises(LockLost):
        with h.keep_alive(interval_sec=0.05):
            assert _wait_for(lambda: h.lost is not None)
            with pytest.raises(LockLost):
                h.check()


def test_a_takeover_seen_only_on_disk_is_raised_at_the_end_of_the_block(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    with pytest.raises(LockLost, match="taken over"):
        with h.keep_alive(interval_sec=20):  # no renew runs during this block
            lock.try_acquire(_holder("b"), alive=lambda holder: False)
            h.check()  # the heartbeat is fresh, so only a read of the lock can see the takeover


def test_keep_alive_interval_must_be_finite_positive_and_at_most_a_third_of_the_ttl(tmp_path):
    h = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60).try_acquire(_holder("a"))
    for bad in (21, 0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            h.keep_alive(interval_sec=bad)
    with h.keep_alive(interval_sec=20):
        pass
    short = GenerationLock(PosixStore(tmp_path), "locks/s", ttl_sec=0.6).try_acquire(_holder("a"))
    with short.keep_alive(interval_sec=0.2):  # a third of 0.6 is accepted despite float rounding
        pass


def test_slow_failing_renewals_are_reported_lost_before_the_ttl_ends(tmp_path):
    lock = GenerationLock(_SlowFailStore(tmp_path, delay_sec=0.06), "locks/k", ttl_sec=0.6)
    h = lock.try_acquire(_holder("a"))
    start = time.monotonic()
    with pytest.raises(LockLost):
        with h.keep_alive(interval_sec=0.15):
            assert _wait_for(lambda: h.lost is not None)
            assert time.monotonic() - start < 0.6  # nobody can have taken over yet
            with pytest.raises(LockLost) as e:
                h.check()
            assert isinstance(e.value.__cause__, OSError)


def test_a_blocked_write_is_reported_lost_and_does_not_hang_the_block_or_release(tmp_path):
    store = _BlockedStore(tmp_path)
    lock = GenerationLock(store, "locks/k", ttl_sec=0.4)
    h = lock.try_acquire(_holder("a"))
    start = time.monotonic()
    try:
        with pytest.raises(LockLost) as e:
            with h.keep_alive(interval_sec=0.1):
                assert store.entered.wait(1) and h.lost is None
                assert _wait_for(lambda: h.lost is not None)
                with pytest.raises(LockLost):
                    h.check()
        assert "locks/k" in str(e.value) and "generation 1" in str(e.value)
        h.release()  # the stuck renew holds the guard; release gives up after a TTL
        assert time.monotonic() - start < 5  # the gate would hold a hung call for 30 s
    finally:
        store.gate.set()


def test_a_lease_is_lost_from_its_last_heartbeat_even_with_no_renew(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    clock.t += 44
    h.check()
    clock.t += 2
    with pytest.raises(LockLost) as e:
        h.check()
    assert "locks/k" in str(e.value) and "generation 1" in str(e.value) and "46.0s" in str(e.value)
    assert h.lost is not None
    with pytest.raises(LockLost):  # entering keep_alive late does not hide it
        with h.keep_alive(interval_sec=20):
            pass


def test_the_last_heartbeat_is_the_value_renew_wrote(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    clock.t += 30
    h.renew()
    clock.t += 44
    h.check()
    clock.t += 2
    with pytest.raises(LockLost):
        h.check()
    with pytest.raises(LockLost):  # a loss is final: a later renew is refused and check keeps raising
        h.renew()
    with pytest.raises(LockLost):
        h.check()
    assert h.lost is h.lost


class _StallStore(PosixStore):
    """A store whose writes take a long time on a fake clock, with a hook that runs meanwhile."""

    def __init__(self, root: Path, clock: Clock, stall_sec: float) -> None:
        super().__init__(root)
        self.clock = clock
        self.stall_sec = stall_sec
        self.meanwhile = None

    def write_bytes(self, key: str, data: bytes) -> None:
        self.clock.t += self.stall_sec
        if self.meanwhile is not None:
            self.meanwhile()
        super().write_bytes(key, data)


def test_a_heartbeat_that_lands_late_is_rejected_and_the_loss_is_final(tmp_path):
    clock = Clock()
    lock = GenerationLock(_StallStore(tmp_path, clock, stall_sec=50), "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    with pytest.raises(LockLost, match="landed 50.0s after"):
        h.renew()
    assert h._heartbeat_at == 1000.0
    with pytest.raises(LockLost):
        h.check()


def test_a_takeover_during_a_slow_write_is_not_hidden_by_the_late_heartbeat(tmp_path):
    clock = Clock()
    store = _StallStore(tmp_path, clock, stall_sec=50)
    lock = GenerationLock(store, "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    store.meanwhile = lambda: lock.try_acquire(_holder("b"), alive=lambda holder: False)
    with pytest.raises(LockLost):
        h.renew()
    assert lock.current().generation == 2
    with pytest.raises(LockLost):
        h.check()
    with pytest.raises(LockLost):
        h.confirm()


class _SlowReadStore(PosixStore):
    """A store whose next generation-file read returns what it read only after a long stall on a
    fake clock. The stall is (seconds before the hook, seconds after it); the hook runs in between."""

    def __init__(self, root: Path, clock: Clock) -> None:
        super().__init__(root)
        self.clock = clock
        self.stall: tuple[float, float] | None = None
        self.meanwhile = None

    def read_bytes(self, key: str) -> bytes:
        data = super().read_bytes(key)
        stall, self.stall = self.stall, None
        if stall is not None and "/gen-" in key:
            self.clock.t += stall[0]
            if self.meanwhile is not None:
                self.meanwhile()
            self.clock.t += stall[1]
        return data


def test_a_slow_read_of_the_lock_does_not_take_over_a_fresh_holder(tmp_path):
    clock = Clock()
    store = _SlowReadStore(tmp_path, clock)
    lock = GenerationLock(store, "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    store.stall = (31, 31)  # the holder renews 31 s into the read, which returns the old heartbeat
    store.meanwhile = h.renew
    with pytest.raises(LockHeld):
        lock.try_acquire(_holder("b"))
    assert lock.current().generation == 1 and lock.current().heartbeat_at == 1031.0
    h.confirm()


def _stall_the_next_read_while_b_takes_over_by_expiry(store: _SlowReadStore, lock: GenerationLock) -> None:
    # B takes over 61 s into a 62 s read on a 60 s TTL; the read then returns A's generation as current
    store.stall = (61, 1)
    store.meanwhile = lambda: lock.try_acquire(_holder("b"))


def test_confirm_after_a_slow_read_sees_that_the_lease_expired(tmp_path):
    clock = Clock()
    store = _SlowReadStore(tmp_path, clock)
    lock = GenerationLock(store, "locks/k", ttl_sec=60, clock=clock)
    a = lock.try_acquire(_holder("a"))
    _stall_the_next_read_while_b_takes_over_by_expiry(store, lock)
    with pytest.raises(LockLost, match="62.0s"):
        a.confirm()
    assert lock.current().generation == 2
    with pytest.raises(LockLost):  # and the loss is final
        a.check()


def test_a_keep_alive_block_end_after_a_slow_read_sees_that_the_lease_expired(tmp_path):
    clock = Clock()
    store = _SlowReadStore(tmp_path, clock)
    lock = GenerationLock(store, "locks/k", ttl_sec=60, clock=clock)
    a = lock.try_acquire(_holder("a"))
    with pytest.raises(LockLost, match="62.0s"):
        with a.keep_alive(interval_sec=20):  # no renew runs during this block
            _stall_the_next_read_while_b_takes_over_by_expiry(store, lock)
    assert lock.current().generation == 2


def test_confirm_sees_a_takeover_that_the_heartbeat_does_not(tmp_path):
    lock = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60, clock=Clock())
    h = lock.try_acquire(_holder("a"))
    h.confirm()
    lock.try_acquire(_holder("b"), alive=lambda holder: False)
    h.check()  # the heartbeat is fresh, so the age says nothing
    with pytest.raises(LockLost):
        h.confirm()
    with pytest.raises(LockLost):  # and the loss is final
        h.check()


def test_a_released_handle_is_not_lost(tmp_path):
    clock = Clock()
    h = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60, clock=clock).try_acquire(_holder("a"))
    h.release()
    clock.t += 100
    assert h.lost is None


def test_renew_after_release_does_nothing(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/k", ttl_sec=60, clock=clock)
    h = lock.try_acquire(_holder("a"))
    h.release()
    clock.t += 5
    h.renew()
    assert lock.current().released and lock.current().heartbeat_at == 1000.0


def test_nested_lock_keys_do_not_leak_into_the_parent(tmp_path):
    store = PosixStore(tmp_path)
    nested = GenerationLock(store, "locks/a/b", ttl_sec=60, clock=Clock())
    for _ in range(3):
        nested.try_acquire(_holder("n")).release()
    parent = GenerationLock(store, "locks/a", ttl_sec=60, clock=Clock())
    assert nested.current().generation == 3
    assert parent.current() is None
    assert parent.try_acquire(_holder("p")).generation == 1


def test_holder_here_describes_this_process():
    me = Holder.here("x", job_id="42")
    assert (me.host, me.pid, me.job_id, me.purpose) == (socket.gethostname(), os.getpid(), "42", "x")
    assert me.boot_id and me.pid_ns


def test_same_host_liveness():
    me = Holder.here("x")
    assert same_host_liveness(me) is True
    assert same_host_liveness(me.model_copy(update={"pid": 2**22 + 12345})) is False
    assert same_host_liveness(Holder(host="elsewhere", pid=1, job_id=None, purpose="x")) is None


def test_liveness_is_unknown_in_another_pid_namespace():
    gone = Holder.here("x").model_copy(update={"pid_ns": 1, "pid": 2**22 + 12345})
    assert same_host_liveness(gone) is None


def test_liveness_is_unknown_without_a_boot_id_or_namespace():
    me = Holder.here("x")
    assert same_host_liveness(me.model_copy(update={"boot_id": None})) is None
    assert same_host_liveness(me.model_copy(update={"pid_ns": None})) is None
    old_style = Holder(host=socket.gethostname(), pid=os.getpid(), job_id=None, purpose="x")
    assert same_host_liveness(old_style) is None


def test_instance_lease_lives_under_the_instance(tmp_path):
    store = PosixStore(tmp_path)
    lay = InstanceLayout.rollout("t", "r")
    instance_lease(store, lay, ttl_sec=120).try_acquire(_holder("orchestrator"))
    assert store.list_keys(lay.lease) == [f"{lay.lease}/gen-00000001.json"]


def test_lock_held_names_the_key_and_generation(tmp_path):
    clock = Clock()
    lock = GenerationLock(PosixStore(tmp_path), "locks/a", ttl_sec=60, clock=clock)
    lock.try_acquire(_holder("a"))
    clock.t += 61
    lock.try_acquire(_holder("b"))
    with pytest.raises(LockHeld, match=r"locks/a: generation 2 is held by b on h pid 1") as caught:
        lock.try_acquire(_holder("c"))
    assert caught.value.key == "locks/a"
