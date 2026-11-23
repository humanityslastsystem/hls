"""The driver role's memory guard, the second layer of its reply gate.

The reply gate (``hls.drive.replies``) bounds the reply bytes the driver holds at once, received
and not yet written into the trace, by the task's trace cap; but what the client's decoder holds
per byte of a reply it is reading depends on how a server spells it: the densest spelling the
decoder accepts holds up to 19 bytes per byte (an omni reply; a TTS stream 17.5).
Under the caps the tasks declare, that can pass the driver's memory: a TTS drive may
hold 9.3 GiB of replies before its cap closes the gate, an omni drive 1 GiB. So the role also
watches its own memory. About once a second during the drive the guard samples it: the
footprint of the role's Slurm step, as the runner's memory watch reads it
(``hls.substrate.memory_watch``: ``memory.current`` and swap, less the page cache and slab
the kernel reclaims first), or the process's resident set where that step's cgroup cannot be
read. Once a sample passes ``GUARD_SHARE`` of the driver's declared memory, the guard closes
the drive's reply gate, and every further reply fails as the contestant's request, before the
runner's memory watch or the kernel ends the role at the limit itself, which would read as the
harness's fault. A guard that fired stays fired."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from hls.drive.replies import ReplyGate
from hls.substrate.cgroup import CGROUP_ROOT, own_cgroup
from hls.substrate.factory import step
from hls.substrate.memory_watch import read_sample

log = logging.getLogger(__name__)

#: The share of the driver's declared memory past which every further reply fails.
GUARD_SHARE = 0.85
#: Seconds between two samples of the driver's memory.
SAMPLE_SEC = 1.0


@dataclass(frozen=True)
class MemorySource:
    """Where the guard reads the driver's memory."""

    #: For the role's log: ``step cgroup`` or ``RSS``.
    name: str
    #: Reads the memory, in bytes.
    read: Callable[[], int]


def rss_bytes() -> int:
    """This process's resident set, in bytes."""
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


RSS = MemorySource("RSS", rss_bytes)


def memory_source(proc_cgroup: Path = Path("/proc/self/cgroup"), root: Path = CGROUP_ROOT) -> MemorySource:
    """The footprint of this process's Slurm step when its cgroup can be read, else ``RSS``. The step
    is the one of the substrate the step's environment names (``hls.substrate.factory.step``)."""
    here = step(os.environ)
    try:
        limits = here.step_cgroup(own_cgroup(proc_cgroup), root)
        read_sample(limits, 0.0)
    except (OSError, RuntimeError, ValueError):
        return RSS
    return MemorySource("step cgroup", lambda: read_sample(limits, 0.0).footprint)


class MemoryGuard:
    """Samples the driver's memory while the drive runs and closes *gate* once it passes
    ``GUARD_SHARE`` of *limit_bytes*. A context manager: the sampling thread runs inside it."""

    def __init__(self, gate: ReplyGate, limit_bytes: int, source: MemorySource, *,
                 interval: float | None = None) -> None:
        self._gate = gate
        #: The driver's declared memory, in bytes.
        self.limit_bytes = limit_bytes
        #: The sample past which the guard fires.
        self.threshold = int(limit_bytes * GUARD_SHARE)
        #: Where the memory is read; ``RSS`` once the first source failed.
        self.source = source
        #: Seconds between two samples.
        self.interval = SAMPLE_SEC if interval is None else interval
        #: The largest sample so far.
        self.peak = 0
        #: The sample at which the guard fired; None while it has not.
        self.fired_at: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> int:
        """One reading, kept in ``peak``. A source that fails is replaced by ``RSS``."""
        try:
            value = self.source.read()
        except OSError as e:
            log.warning("driver memory: %s unreadable (%s); reading the process's RSS from now on", self.source.name, e)
            self.source = RSS
            value = self.source.read()
        self.peak = max(self.peak, value)
        return value

    def check(self) -> None:
        """Take one sample, and close the gate if it passes the threshold."""
        value = self._sample()
        if value > self.threshold and self.fired_at is None:
            self.fired_at = value
            self._gate.close(f"the driver's memory passed {GUARD_SHARE:.0%} of its {self.limit_bytes:,} bytes")
            log.warning("driver memory %s bytes (%s) passed %.0f%% of its %s bytes: every further reply fails",
                        f"{value:,}", self.source.name, GUARD_SHARE * 100, f"{self.limit_bytes:,}")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self.check()

    def __enter__(self) -> MemoryGuard:
        self.check()
        self._thread = threading.Thread(target=self._run, name="drive-memory-guard", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        # The drive has ended: one more reading for the peak, which can no longer fail a reply.
        self._sample()

    def summary(self) -> str:
        """The guard's part of the role's last log line."""
        fired = "not fired" if self.fired_at is None else f"fired at {self.fired_at:,}"
        return f"memory peak {self.peak:,} of {self.limit_bytes:,} bytes ({self.source.name}), guard {fired}"
