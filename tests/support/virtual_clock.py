"""A virtual clock for tests that drive traffic through ``hls.spec.drivers``.

A drive on the wall clock never repeats to the microsecond, and on a loaded machine its
realized waits stretch by however long the process sat descheduled. On a virtual clock the
event loop's time moves only when every coroutine waits, and then straight to the next timer,
so a send time or a think gap is exactly what the schedule made it, whatever the machine is
doing. The driver's own clock (``hls.spec.drivers.harness_now``) is pointed at the same time.
"""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Iterator
from contextlib import contextmanager
from unittest import mock

from hls.spec import drivers


class _VirtualSelector(selectors.DefaultSelector):
    """A selector whose waits take no real time: a wait for a timer moves the clock to it.

    Real readiness is still polled first, so the loop's own wake-up pipe works. A wait with no
    timer and nothing ready can never end on this clock, and raises instead of hanging."""

    def __init__(self) -> None:
        super().__init__()
        #: The virtual time in seconds, starting at 0 when the loop is made.
        self.now = 0.0

    def select(self, timeout: float | None = None):
        ready = super().select(0)
        if ready or timeout == 0:
            return ready
        if timeout is None:
            raise RuntimeError("the drive waits on nothing a virtual clock can wake")
        self.now += timeout
        return []


class VirtualClockLoop(asyncio.SelectorEventLoop):
    """An event loop on a virtual clock, with every executor call run inline, so nothing the
    drive does depends on how fast this machine runs it."""

    def __init__(self) -> None:
        self._selector_clock = _VirtualSelector()
        super().__init__(self._selector_clock)

    def time(self) -> float:
        return self._selector_clock.now

    def run_in_executor(self, executor, func, *args):
        future = self.create_future()
        try:
            future.set_result(func(*args))
        except Exception as exc:  # handed to the coroutine that awaits the call
            future.set_exception(exc)
        return future


@contextmanager
def virtual_clock() -> Iterator[asyncio.Runner]:
    """An ``asyncio.Runner`` on a ``VirtualClockLoop`` whose time is also the driver's clock,
    so every stamp a drive run on it records is virtual. Time starts at 0."""
    with asyncio.Runner(loop_factory=VirtualClockLoop) as runner:
        with mock.patch.object(drivers, "harness_now", runner.get_loop().time):
            yield runner
