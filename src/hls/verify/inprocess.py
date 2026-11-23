"""The engine host: reference engines loaded inside the calling process.

A role that runs reference engines (the verify role, the oracle fill, the calibration roles)
runs them in its own process, on its own GPUs: nothing is shipped to another process and no
call crosses a wire. The role creates an ``EngineHost`` and hands it to the code that loads
engines; ``acquire`` loads an engine and returns it with its release, so every
``engine[Capability]`` call is a plain method call.

``acquire`` runs the engine's ``_load`` in a worker thread, as a model load blocks. A release
undoes it in three steps, which together do what ending a process of its own would. It stops
the engine (``Engine._unload``): a serving stack runs its engine in a child process, which
dropping the engine does not end. It then puts the engine's own attributes back to what they
were before ``_load``, which drops the model, the tokenizer and every other object the load
created, so the engine can be loaded again. Last it collects garbage and empties the CUDA
cache when torch is loaded. So the GPU is free for the next engine (the verification core
evicts before it loads), and the same engine can be loaded again. A ``_load`` that raises is
undone the same way before its error reaches the caller.

The load, the stop and the freeing run in worker threads, as each can block for long. A
worker thread cannot be stopped, so a cancelled acquire or release waits for its thread to
finish before it undoes anything and lets the cancellation through: a load still running
would otherwise fill the engine again after the undo, leaving it loaded, holding its GPU and
released by nobody.

When a stop fails, the engine is still put back, freed and deregistered, and the error
reaches the caller (the verify role counts any load or release failure as the harness's). A
load whose undo fails too reports the load, with the undo's failure noted on it.
``teardown`` releases every engine, then raises the first failure; leaving an
``async with EngineHost(...)`` block tears down.

A step may also never return: a load that deadlocks, a stop that waits on a process that will
not end, a collection whose finalizer waits for ever (SGLang's ZMQ context can, for
messages to the processes its engine's stop had killed). So each step has a bound: a load the
host's ``load_sec`` when it has one, and each step of a release (the stop, putting the engine
back, freeing what it held) ``release_sec``. A step past its bound ends the process from a
timer thread, since the thread that runs the step cannot be stopped and an orderly exit would
wait for it: the host gives up (``_give_up``). Giving up writes every thread's stack when a
step hung, kills and reaps whatever engine processes the host watches, writes last on stderr
one line ``<prefix><kind>: <reason>``, and exits with the status its ``GiveUp`` names, which
the role sets so its stage can read the line. Every give-up is the harness's: no contestant
runs on an engine host.

An engine that runs processes of its own (``Engine.runs_processes``: a serving stack's engine
core, scheduler or detokenizer) is watched from before its load until its release. Its
processes are those ``multiprocessing`` starts while it loads, which is how vLLM and SGLang
start theirs; the host holds one such engine at a time, so no other engine starts one then.
Every ``ENGINE_POLL_SEC`` a thread of the host asks whether one of them ended, reading its
exit status through ``multiprocessing``, which reaps it, and, once the engine loaded, whether
the engine reports itself dead (``Engine._health``). The host checks once more when the load
ends and when the release begins. Either answer ends the role (``EngineDied``): a role cannot
go on with an engine that died, as its load may wait for ever on a peer that is gone and a
serving's requests would each wait out their own timeout. The line names each process that
ended and how, what the engine reported, and the processes of it the host killed. What the
processes wrote is in the role's own log just above it, since they write to the role's stdout
and stderr. The processes are read from the registry ``multiprocessing`` keeps of the processes
this one started (``_children``), not through ``active_children``, which reaps and forgets a
process that ended before it lists the live ones: a process that ends between two checks, such
as an SGLang detokenizer that dies while its engine loads, which SGLang's own load never
checks, would go unseen. One that its own engine reaps first is gone from the registry too; an
engine that reaps its process has seen it end. A release stops the watch before the engine's
stop, which ends the processes as it means to, then reaps them, giving each ``REAP_SEC`` to end,
and kills and reaps any the stop left running: a stop that kills its processes without reaping
them, as SGLang's does, would leave them as zombies of the role.

An engine that places itself (``ReferenceEngine.placeable``, the HF engine) is given, before
its load, the lowest GPUs of the role that no other placed engine holds, as many as its
resources ask for: the host sets its ``device_index`` to the first of them, and the engine
loads and computes there. They are held from before the load starts, so an acquire that runs
while another engine loads never takes the same GPU, and they are free again once the engine
is released, once its load failed or once the host is torn down. An engine asking for more
GPUs than are free is refused before anything of it loads. So a calibration's pinned
reference and its fp32 anchor, held at once, compute on GPUs 0 and 1 of a two-GPU role. The
role's GPU count is the one the caller gives, else torch's, read once, the first time an
engine that places itself is acquired; a role that sees no GPU places nothing. An engine that
does not place itself (a serving stack, which spreads over every GPU its process sees, and
every engine but the HF one) is neither placed nor counted, and the count is never read for
it, so nothing here touches CUDA before a serving stack starts its workers. That such an
engine shares its role with no other is the planner's check
(``hls.stages.verify.multi_engine_problem``)."""

from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import gc
import logging
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from types import TracebackType
from typing import Any, NoReturn, TypeVar

from hls.core.engine import Engine

log = logging.getLogger(__name__)

#: What ``EngineHost.acquire`` hands back beside the engine: the call that releases it.
Release = Callable[[], Awaitable[None]]

#: Seconds between two checks of an engine that runs processes of its own. A check is a
#: non-blocking wait for each of its processes and a read of the engine's own report, so a check
#: every second costs nothing a role can measure, and a role whose engine died ends about a second
#: later instead of at whatever bound it was waiting under.
ENGINE_POLL_SEC = 1.0

#: Seconds each step of a release may take: an engine's stop, putting its attributes back, and
#: freeing what its load held. Five times the longest wait a supported engine's stop allows itself
#: (SGLang waits up to 60 s for the processes it killed to be reaped; vLLM gives its engine core
#: 5 s before it kills it), so a stop that is only slow is never cut, while one that hangs ends its
#: role in minutes rather than at its stage's time.
RELEASE_STEP_SEC = 300.0

#: Seconds the host gives an engine's processes to end once signalled: after the engine's stop,
#: before it counts one still running as left by the stop and kills it, and after it killed them
#: itself when it gives up, before it exits without them. A killed process that holds a GPU can
#: take seconds to end while the driver frees its memory, and a stop may signal its processes and
#: return without waiting for them (vLLM's kills what outlived its grace that way).
REAP_SEC = 10.0

_MISSING = object()

#: Held by the one thread that gives up (``EngineHost._give_up``), so any other waits for the exit.
_ENDING = threading.Lock()

_T = TypeVar("_T")


@dataclass(frozen=True)
class GiveUp:
    """How a host ends its process when it gives up on an engine (``EngineHost._give_up``): last on
    stderr the line ``<prefix><kind>: <reason>``, then exit status ``exit_code``. A role names its
    own, so its stage finds the line and knows the status."""

    #: The process's exit status.
    exit_code: int
    #: How the line starts.
    prefix: str


#: How a host gives up when its caller names no other way: with the status of any failure.
ANY_FAILURE = GiveUp(exit_code=1, prefix="hls-engine-host: ")


class PlacementError(Exception):
    """An engine the host cannot load: not an engine, already loaded, asking for more GPUs
    than the role has free, or running processes of its own beside another that does."""


async def _to_end(fn: Callable[[], _T]) -> _T:
    """Run *fn* in a worker thread, wait for it to return, and return what it returned, even
    when cancelled: the thread cannot be stopped, so on cancellation this still waits for *fn*
    to end and only then lets the cancellation through. *fn*'s own error is raised as usual."""
    future = asyncio.ensure_future(asyncio.to_thread(fn))
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        while not future.done():
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait([future])
        if not future.cancelled():
            future.exception()  # retrieved, so asyncio does not log it: the cancellation goes on
        raise


def _put_back(engine: Engine, before: dict[str, Any]) -> None:
    """Put *engine*'s attributes back to *before*."""
    now = vars(engine)
    for name in [n for n in now if n not in before]:
        object.__delattr__(engine, name)
    for name, value in before.items():
        if now.get(name, _MISSING) is not value:
            object.__setattr__(engine, name, value)


def _free_memory() -> None:
    """Collect garbage, then empty the CUDA cache when torch is loaded. The order
    matters: tensors a reference cycle holds are freed only by the collection, and the
    cache returns only freed blocks."""
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _visible_gpus() -> int:
    """The GPUs this process sees: torch's count, which honours ``CUDA_VISIBLE_DEVICES`` and is
    taken from NVML where torch can, so a role step's own GPUs; 0 without torch."""
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return 0
    return int(torch.cuda.device_count())


def _gpus_text(gpus: tuple[int, ...]) -> str:
    """*gpus* for a message: "GPU 0", or "GPUs 0, 1"."""
    return f"GPU {gpus[0]}" if len(gpus) == 1 else f"GPUs {', '.join(map(str, gpus))}"


def _named(process: BaseProcess) -> str:
    """*process* for a message: its pid and its name."""
    return f"process {process.pid} ({process.name})"


def _signal_text(number: int) -> str:
    """Signal *number* for a message, by its name where it has one."""
    try:
        return f"signal {number} ({signal.Signals(number).name})"
    except ValueError:
        return f"signal {number}"


def _started() -> list[BaseProcess]:
    """Every process ``multiprocessing`` started from this one that nobody has yet seen end, those
    that ended since included: its own registry, which ``active_children`` lists only after it
    reaped and dropped the ones that ended."""
    return list(multiprocessing.process._children)  # type: ignore[attr-defined]


def _ended(process: BaseProcess) -> str | None:
    """How *process*, a child of this one that ``multiprocessing`` started, ended, or None while it
    runs. Its exit status is read through ``multiprocessing``, which reaps it and keeps the status
    for whoever started it. A process another waiter reaped first, as SGLang's shutdown reaps its
    own through psutil, has no status left to read."""
    try:
        code = process.exitcode
    except ValueError:
        # Its owner closed it, which multiprocessing refuses while it runs.
        return "ended; its owner closed it"
    if code is None:
        try:
            if os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
                return None
        except ChildProcessError:
            return "ended; another waiter collected its status"
        # It ended since: read once more, which reaps it.
        code = process.exitcode
        if code is None:
            return "ended; another waiter collected its status"
    return f"exited with code {code}" if code >= 0 else f"was killed by {_signal_text(-code)}"


class _Watch:
    """What the host keeps of an engine that runs processes of its own (``Engine.runs_processes``)
    from before its load until its release: the processes ``multiprocessing`` started since, and a
    thread that checks them every ``ENGINE_POLL_SEC``, with the engine's own report once it
    loaded."""

    def __init__(self, host: EngineHost, engine: Engine, name: str) -> None:
        self._host = host
        self.engine = engine
        #: Its name in the host's lines.
        self.name = name
        #: The engine's processes, in the order the watch saw them.
        self.processes: list[BaseProcess] = []
        #: Whether the engine's load returned.
        self.loaded = False
        # What multiprocessing started before the load, which is not the engine's.
        self._before = set(_started())
        # Held while the processes are listed, which the thread and the host both do.
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"hls-engine-watch {name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(ENGINE_POLL_SEC):
            self.check()

    def _seen(self) -> list[BaseProcess]:
        """The engine's processes: every one multiprocessing started since the watch began."""
        with self._lock:
            for process in _started():
                if process not in self._before and process not in self.processes:
                    self.processes.append(process)
            return list(self.processes)

    def check(self, load_error: BaseException | None = None) -> None:
        """Give up on the engine (``EngineDied``) when one of its processes ended or, once it
        loaded, it reports itself dead. A check that fails gives up too: nothing then says the
        engine lives. *load_error* is what its load raised, which the line then names."""
        failed = None
        try:
            ended = [(process, how) for process in self._seen() if (how := _ended(process)) is not None]
            report = self.engine._health() if self.loaded else None
        except Exception as e:  # noqa: BLE001 -- an engine that cannot be checked is not known to live
            ended, report, failed = [], None, e
        if not ended and report is None and failed is None:
            return
        parts = [f"{_named(process)} {how}" for process, how in ended]
        if report is not None:
            parts.append(f"it reports: {report}")
        if failed is not None:
            parts.append(f"the host could not check it: {type(failed).__name__}: {failed}")
        if load_error is not None:
            parts.append(f"its load raised {type(load_error).__name__}: {load_error}")
        when = "after it loaded" if self.loaded else "while loading"
        self._host._give_up("EngineDied", f"{self.name} ({type(self.engine).__name__}) died {when}: "
                                          + "; ".join(parts))

    def stop(self) -> None:
        """Stop the thread that checks the engine, which its stop is about to end as it means to."""
        self._stop.set()
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _join(self, deadline: float | None) -> None:
        """Reap each of the engine's processes once it ends, waiting until *deadline* (a monotonic
        time) in all, None for as long as it takes."""
        for process in self._seen():
            with contextlib.suppress(ValueError):  # closed by its owner, so ended and reaped
                process.join(None if deadline is None else max(0.0, deadline - time.monotonic()))

    def _kill_running(self) -> list[BaseProcess]:
        """Kill the engine's processes that still run; return them."""
        running = [process for process in self._seen() if _ended(process) is None]
        for process in running:
            process.kill()
        return running

    def end_after_stop(self) -> list[BaseProcess]:
        """Reap the engine's processes once its stop returned, giving them ``REAP_SEC`` to end; kill
        those still running then, which the stop should have ended, and reap them as long as that
        takes. Return the processes killed."""
        self._join(time.monotonic() + REAP_SEC)
        killed = self._kill_running()
        self._join(None)
        return killed

    def end_now(self) -> list[BaseProcess]:
        """Kill the engine's processes that still run and reap every one within ``REAP_SEC``, for a
        host that gives up. Return the processes killed."""
        killed = self._kill_running()
        self._join(time.monotonic() + REAP_SEC)
        return killed


class EngineHost:
    """Loads reference engines into this process; see the module docstring."""

    def __init__(self, devices: int | None = None, *, load_sec: float | None = None,
                 release_sec: float = RELEASE_STEP_SEC, give_up: GiveUp = ANY_FAILURE) -> None:
        """*devices* is how many GPUs the role sees; None reads torch's count the first time an
        engine that places itself is acquired (``_visible_gpus``). *load_sec* bounds each load,
        None none; *release_sec* bounds each step of a release; *give_up* says how the process
        ends when the host gives up on an engine."""
        if devices is not None and devices < 0:
            raise ValueError(f"a role sees 0 GPUs or more, not {devices}")
        if (load_sec is not None and load_sec <= 0) or release_sec <= 0:
            raise ValueError(f"a bound is a positive number of seconds, not load_sec={load_sec}, "
                             f"release_sec={release_sec}")
        #: Seconds each load may take, or None for no bound.
        self.load_sec = load_sec
        #: Seconds each step of a release may take.
        self.release_sec = release_sec
        #: How the process ends when the host gives up on an engine.
        self.give_up = give_up
        # id of each engine acquired and not yet released, to the engine, the attributes it had
        # before its load and its name in the host's lines.
        self._before: dict[int, tuple[Engine, dict[str, Any], str]] = {}
        # The GPUs the role sees, once known.
        self._devices = devices
        # id of each placed engine acquired and not yet released, to its class's name and the
        # GPUs it holds. An engine is in it from before its load starts.
        self._placed: dict[int, tuple[str, tuple[int, ...]]] = {}
        # id of each engine that runs processes of its own, from before its load until its
        # release, to its watch.
        self._watches: dict[int, _Watch] = {}

    async def __aenter__(self) -> EngineHost:
        return self

    async def __aexit__(self, kind: type[BaseException] | None, error: BaseException | None,
                        traceback: TracebackType | None) -> None:
        await self.teardown()

    @property
    def outstanding(self) -> int:
        """How many engines are acquired and not yet released."""
        return len(self._before)

    @property
    def devices(self) -> int:
        """How many GPUs the role sees: the count given, else torch's, read once."""
        if self._devices is None:
            self._devices = _visible_gpus()
        return self._devices

    def _place(self, engine: Engine) -> tuple[int, ...]:
        """The GPUs *engine* is placed on: none for an engine that does not place itself
        (``ReferenceEngine.placeable``), for which the role's count is never read, and none in
        a role that sees no GPU; else the lowest GPUs no placed engine holds, as many as its
        resources ask for. Raise PlacementError when fewer are free."""
        if not getattr(engine, "placeable", False) or self.devices == 0:
            return ()
        wanted = engine.resources.gpu_count
        taken = {gpu for _, gpus in self._placed.values() for gpu in gpus}
        free = [gpu for gpu in range(self.devices) if gpu not in taken]
        if len(free) < wanted:
            held = ", ".join(f"{name} ({_gpus_text(gpus)})" for name, gpus in self._placed.values())
            others = f", the others held by {held}" if held else ""
            raise PlacementError(f"{type(engine).__name__} asks for {wanted} GPUs; {len(free)} of the role's "
                                 f"{self.devices} are free{others}")
        return tuple(free[:wanted])

    @contextlib.contextmanager
    def _bound(self, seconds: float | None, kind: str, what: str) -> Iterator[None]:
        """Give the step the block runs *seconds*, None for no bound: past them a timer thread gives
        up (*kind*, "*what* within *seconds* s"), whatever the thread that runs the step is doing."""
        if seconds is None:
            yield
            return
        claimed = threading.Lock()

        def expire() -> None:
            if claimed.acquire(blocking=False):
                self._give_up(kind, f"{what} within {seconds:g} s", stacks=True)

        timer = threading.Timer(seconds, expire)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            if claimed.acquire(blocking=False):
                timer.cancel()
            else:
                # The timer is giving up, and the process ends there.
                _ENDING.acquire()

    def _give_up(self, kind: str, reason: str, *, stacks: bool = False) -> NoReturn:
        """End this process at once, as ``give_up`` says: with every thread's stack first when a
        step hung (*stacks*), the engine processes the host watches killed and reaped, then last on
        stderr the line ``<prefix><kind>: <reason>``. At once, because a thread the host cannot get
        back would hold an orderly exit for ever. The first caller ends the process; any other
        waits for it."""
        _ENDING.acquire()
        try:
            with contextlib.suppress(Exception):
                sys.stdout.flush()
            if stacks:
                try:
                    print("hls-engine-host: every thread's stack follows", file=sys.stderr, flush=True)
                    faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                except Exception as e:  # noqa: BLE001 -- the line and the exit come first
                    reason += f"; writing the stacks failed: {type(e).__name__}: {e}"
            try:
                killed = [process for watch in list(self._watches.values()) for process in watch.end_now()]
            except Exception as e:  # noqa: BLE001 -- the line and the exit come first
                reason += f"; ending its processes failed: {type(e).__name__}: {e}"
            else:
                if killed:
                    reason += "; the host killed its processes still running: " + ", ".join(map(_named, killed))
            print(" ".join(f"{self.give_up.prefix}{kind}: {reason}".split()), file=sys.stderr, flush=True)
        finally:
            os._exit(self.give_up.exit_code)

    async def acquire(self, engine: Engine, *, name: str | None = None) -> tuple[Engine, Release]:
        """Load *engine* here and return ``(engine, release)``, on the GPUs it is placed on when
        it places itself (``_place``). *name* names it in the host's lines, its class's name when
        None. Refuse (PlacementError) an engine that runs processes of its own while the host holds
        another that does: every process that starts while such an engine loads is taken as its."""
        if not isinstance(engine, Engine):
            raise PlacementError(f"only reference engines run in process, not {type(engine).__name__}")
        key = id(engine)
        if key in self._before or key in self._placed or key in self._watches:
            raise PlacementError(f"{type(engine).__name__} is already acquired")
        if engine.runs_processes and self._watches:
            other = next(iter(self._watches.values())).name
            raise PlacementError(f"{type(engine).__name__} runs processes of its own, as {other} does, which the "
                                 "host holds: it takes every process that starts while such an engine loads as "
                                 "that engine's own")
        label = type(engine).__name__ if name is None else name
        gpus = self._place(engine)
        # Recorded before the placement, so the release puts the engine's device back too.
        before = dict(vars(engine))
        if gpus:
            self._placed[key] = (type(engine).__name__, gpus)
            object.__setattr__(engine, "device_index", gpus[0])
        if engine.runs_processes:
            self._watches[key] = _Watch(self, engine, label)
        load = getattr(engine, "_load", None)
        if callable(load):
            try:
                with self._bound(self.load_sec, "EngineLoadTimeout", f"{label} did not load"):
                    await _to_end(load)
            except BaseException as failed:
                if key in self._watches:
                    # A process of the engine that ended says why the load failed.
                    self._watches[key].check(load_error=failed)
                try:
                    await self._undo(key, engine, before, label)
                except Exception as e:  # noqa: BLE001 -- the load's failure is the one to report
                    failed.add_note(f"undoing the load failed too: {type(e).__name__}: {e}")
                finally:
                    self._placed.pop(key, None)
                raise
        if key in self._watches:
            self._watches[key].loaded = True
            self._watches[key].check()
        self._before[key] = (engine, before, label)

        async def release() -> None:
            held = self._before.pop(key, None)
            if held is None:
                return
            try:
                await self._undo(key, *held)
            finally:
                self._placed.pop(key, None)

        return engine, release

    async def _undo(self, key: int, engine: Engine, before: dict[str, Any], label: str) -> None:
        """Stop *engine*, put its attributes back to *before* and free what its load held, each
        step within ``release_sec``. An engine that runs processes of its own is checked once more
        first, and its processes are reaped after the stop, those it left running killed
        (``_Watch.end_after_stop``). The stop goes first, while the engine still holds the handles
        it stops. The rest runs even when the stop raises, and the stop's error then goes on."""
        watch = self._watches.get(key)
        try:
            if watch is not None:
                watch.stop()
                watch.check()
            with self._bound(self.release_sec, "EngineReleaseTimeout", f"{label}'s stop did not return"):
                await _to_end(engine._unload)
        finally:
            try:
                if watch is not None:
                    try:
                        with self._bound(self.release_sec, "EngineReleaseTimeout",
                                         f"the processes {label}'s stop left did not end"):
                            killed = await _to_end(watch.end_after_stop)
                        for process in killed:
                            log.warning("the stop of %s left %s running; the host killed it", label,
                                        _named(process))
                    finally:
                        del self._watches[key]
            finally:
                with self._bound(self.release_sec, "EngineReleaseTimeout",
                                 f"putting {label}'s attributes back did not return"):
                    _put_back(engine, before)
                with self._bound(self.release_sec, "EngineReleaseTimeout",
                                 f"freeing what {label}'s load held did not return"):
                    await _to_end(_free_memory)

    async def teardown(self) -> None:
        """Release every engine still acquired, each even when another's release fails;
        then raise the first failure, with each later one noted on it."""
        first: Exception | None = None
        for key, (engine, before, label) in list(self._before.items()):
            del self._before[key]
            try:
                await self._undo(key, engine, before, label)
            except Exception as e:  # noqa: BLE001 -- raised below, once every engine is released
                if first is None:
                    first = e
                else:
                    first.add_note(f"releasing {type(engine).__name__} failed too: {type(e).__name__}: {e}")
            finally:
                self._placed.pop(key, None)
        if first is not None:
            raise first
