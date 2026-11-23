"""``python -m tests.stages_lib.engine_watch_prog SCENARIO``: a role in miniature over the engine host
(``hls.verify.inprocess``), for the host's liveness and release bounds (``test_engine_watch``).

Its engine stands in for a serving stack: its load starts a process of its own through
``multiprocessing``'s spawn context, as vLLM's engine core and SGLang's scheduler are started, and
prints that process's pid on stdout as ``child <pid>``. The scenario says what the process and the
engine then do. A scenario the host does not end prints ``finished`` and exits 0; the host ends
every other one itself, with the exit status and line of ``GIVE_UP``, or of the host's default where
the scenario says so. The bounds a scenario gives are seconds, so a test that waits for the end is
short."""

from __future__ import annotations

import asyncio
import gc
import multiprocessing
import os
import signal
import sys
import threading
import time
from typing import ClassVar

from hls.core.engine import Engine, ResourceRequest
from hls.verify.inprocess import EngineHost, GiveUp

#: How the scenarios that name a give-up end.
GIVE_UP = GiveUp(exit_code=5, prefix="test-engine-fault: ")
#: The line the fake engine's process writes before it exits on its own.
LAST_WORDS = "fake scheduler: CUDA error: an illegal memory access was encountered"
#: The exit status of a fake engine process that exits on its own.
CHILD_EXIT = 3
#: The bound a scenario gives a release step, or a load.
BOUND_SEC = 1.5


def _child(behaviour: str) -> None:
    """The fake engine's process: "exits" writes ``LAST_WORDS`` and exits with ``CHILD_EXIT``, and
    "killed" is killed by SIGKILL, each at once, before the host's first check; anything else runs
    until it is ended."""
    if behaviour == "exits":
        print(LAST_WORDS, file=sys.stderr, flush=True)
        sys.exit(CHILD_EXIT)
    if behaviour == "killed":
        os.kill(os.getpid(), signal.SIGKILL)
    time.sleep(3600)


def _term_forever() -> None:
    """What a ZMQ context's finalizer does when a socket of it holds a message for a peer that is
    gone: wait for ever."""
    threading.Event().wait()


class _Lingering:
    """An object in a reference cycle whose finalizer never returns, as the finalizer of SGLang's
    tokenizer manager's ZMQ context did while nothing closed its sockets with linger 0."""

    def __init__(self) -> None:
        self.cycle = self

    def __del__(self) -> None:
        _term_forever()


class FakeServing(Engine):
    """An engine whose load starts one process of its own (``_child``), as a serving stack does."""

    runs_processes: ClassVar[bool] = True

    def __init__(self, behaviour: str, *, load_blocks: bool = False, stop_kills: bool = True,
                 unhealthy_after: float | None = None) -> None:
        #: What its process does (``_child``).
        self.behaviour = behaviour
        #: Whether its load waits for ever once its process started, as a constructor waiting on a
        #: peer that died.
        self.load_blocks = load_blocks
        #: Whether its stop kills its process without reaping it, as SGLang's does; else it leaves it
        #: running.
        self.stop_kills = stop_kills
        #: Seconds after its load past which it reports itself dead; None never.
        self.unhealthy_after = unhealthy_after
        self.process: multiprocessing.process.BaseProcess | None = None
        self.loaded_at: float | None = None

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        self.process = multiprocessing.get_context("spawn").Process(target=_child, args=(self.behaviour,),
                                                                    name="fake-scheduler")
        self.process.start()
        print(f"child {self.process.pid}", flush=True)
        if self.load_blocks:
            threading.Event().wait()
        self.loaded_at = time.monotonic()

    def _unload(self) -> None:
        if self.stop_kills and self.process is not None:
            os.kill(self.process.pid, signal.SIGKILL)

    def _health(self) -> str | None:
        if self.unhealthy_after is not None and time.monotonic() - self.loaded_at > self.unhealthy_after:
            return "the fake engine reports its scheduler gone"
        return None


class FakeLingering(Engine):
    """An engine whose load leaves a ``_Lingering`` behind, so freeing what it held never returns."""

    def __init__(self, *, stop_hangs: bool = False) -> None:
        #: Whether its stop waits for ever.
        self.stop_hangs = stop_hangs

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        self.context = _Lingering()

    def _unload(self) -> None:
        if self.stop_hangs:
            _term_forever()


class FakeLoadHangs(Engine):
    """An engine whose load never returns, as a kernel build that deadlocks."""

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        _term_forever()


def _gone(pid: int) -> bool:
    """Whether *pid* is no longer a child of this process waiting to be reaped."""
    try:
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return True
    return False


async def _scenario(name: str) -> None:
    if name == "dies-loading-default":
        async with EngineHost(devices=0) as host:
            await host.acquire(FakeServing("exits", load_blocks=True), name="dies")
    elif name == "dies-loading":
        async with EngineHost(devices=0, give_up=GIVE_UP) as host:
            await host.acquire(FakeServing("exits", load_blocks=True), name="dies")
    elif name == "killed-held":
        async with EngineHost(devices=0, give_up=GIVE_UP) as host:
            await host.acquire(FakeServing("killed"), name="killed")
            await asyncio.sleep(60)
    elif name == "unhealthy-held":
        async with EngineHost(devices=0, give_up=GIVE_UP) as host:
            await host.acquire(FakeServing("runs", unhealthy_after=0.5), name="unhealthy")
            await asyncio.sleep(60)
    elif name in ("healthy", "stop-leaves-it"):
        engine = FakeServing("runs", stop_kills=name == "healthy")
        async with EngineHost(devices=0, give_up=GIVE_UP) as host:
            _, release = await host.acquire(engine, name="healthy")
            # Long enough for the host to check the engine a few times.
            await asyncio.sleep(3)
            assert engine.process.is_alive()
            # The release puts the engine's attributes back, its process among them.
            pid = engine.process.pid
            await release()
        print(f"reaped {pid}" if _gone(pid) else f"left {pid} unreaped", flush=True)
    elif name in ("free-hangs", "stop-hangs"):
        async with EngineHost(devices=0, release_sec=BOUND_SEC, give_up=GIVE_UP) as host:
            _, release = await host.acquire(FakeLingering(stop_hangs=name == "stop-hangs"), name="lingering")
            await release()
    elif name == "load-hangs":
        async with EngineHost(devices=0, load_sec=BOUND_SEC, give_up=GIVE_UP) as host:
            await host.acquire(FakeLoadHangs(), name="hangs")
    else:
        raise SystemExit(f"no scenario {name!r}")
    gc.collect()
    print("finished", flush=True)


if __name__ == "__main__":
    asyncio.run(_scenario(sys.argv[1]))
