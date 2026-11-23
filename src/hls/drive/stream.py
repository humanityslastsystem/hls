"""The drive role's trace, written as the drive goes: every event the driver records is in the
trace on the node's disk soon after its reply's last byte arrived, and the role holds no
reply past that.

The driver records each event on its event loop, which also sends every request and stamps
every time the drive measures, so nothing is serialized there and nothing waits there.
``TraceStream.put`` hands the event to an unbounded queue and returns: the reply gate already
bounds the reply bytes the queued events hold (``hls.drive.replies``), and an event without a
reply is small. A sender thread takes each event, pickles it, and sends it down a pipe to a
writer process, ``python -m hls.drive.stream``, which unpickles it and writes it with
``hls.trace.store.TraceWriter.in_room`` into the trace's room on the node's disk, which the
stage made before the server started (``hls.trace.store.make_room``). A thread alone would
serialize under the event loop's interpreter lock, and turning one reply into the trace's JSON
holds that lock for as long as the reply is large:
the event loop would stop for it, as it stops for the garbage
collector's full passes. Pickling holds it for at most a few milliseconds at a time
(``_pickled``). The trace the role publishes from the room is the one
``hls.trace.store.write_trace`` writes of the same events, byte for byte.

After each event it writes, the writer process sends back how many events it has written and
the bytes of their JSON, which a reader thread keeps (``TraceStream.written``) for the drive's
reply gate. The stream keeps how far the writer fell behind: the most events handed over and
not yet written at once (``backlog_peak``), the most seconds from an event's ``put`` to the
writer's report of it written (``lag_max_sec``), and the longest the writer waited for space on
the disk (``space_wait_max_sec``), which the role reports in its last line and the drive record.

The room holds every block a trace within the task's bounds writes, so a server that fills the
node's disk takes none of it. A trace past its room asks the disk for more, and on a full disk
the writer waits for space (``TraceStream.waiting_for_space``), so that a server the runner ends
at its own disk cap, and whose files it removes, takes nothing from the trace; past
``WAIT_FOR_SPACE_SEC`` the writer fails, naming the room, how far the trace has gone past it and
how long it waited.

All of it is the harness's. A writer process that ends before it was told to finish, a pipe
that breaks, a writer that has events waiting and has written none of them for ``STALL_SEC``
(beyond ``WAIT_FOR_SPACE_SEC`` while it waits for space), and a writer that has not finished the
trace and exited ``FINISH_SEC`` after its last event, all fail the stream: every later ``put``
and ``finish`` raises ``TraceStreamFailed``, and the role exits 1, which the drive stage records as
infra with the cause. ``close`` ends the process, by SIGTERM and then SIGKILL, and joins it and
the threads; ``TraceStream`` is a context manager that closes on every exit."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import pickle
import queue
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Sequence
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hls.core.trace import DroppedSession, TraceEvent
from hls.trace.store import CHUNK_TARGET_BYTES, DriverDescription, Ending, TraceWriter, TraceWritten

log = logging.getLogger(__name__)

#: Seconds the writer may have events waiting and write none of them, while not waiting for
#: space on the disk, before the stream fails. One event takes at most a few seconds to write.
STALL_SEC = 300.0
#: Seconds the writer waits for space on a full disk before it fails: long enough for the runner
#: to end a server past its own disk cap and remove what it wrote. The runner walks the server's
#: files every 5 s (``hls.substrate.memory_watch``), at most 2,000,000 entries
#: (``DISK_WALK_MAX_ENTRIES``), and the stage looks at the server every 0.5 s. 300 s is the stall
#: bound.
WAIT_FOR_SPACE_SEC = 300.0
#: Seconds the writer may take, after its last event is written and it was told to finish, to
#: close the trace's two gzip streams, sync the room files and exit. The kernel writes back pages
#: unsynced for 30 s (``vm.dirty_expire_centisecs`` 3000, its flusher every 5 s), so the sync
#: writes at most the writer's last 35 s of output. 300 s, the
#: stall bound, leaves room for a disk a server keeps busy.
FINISH_SEC = 300.0
#: Seconds between the writer's tries of a disk with no space.
_SPACE_POLL_SEC = 1.0
#: Seconds between two looks of the stream's watch.
_WATCH_SEC = 1.0
#: Seconds ``close`` gives the writer process after SIGTERM, and then after SIGKILL.
_TERM_SEC = 10.0
#: The tags that start each message to the writer process: an event, or the drive's ending.
_EVENT, _FINISH = b"E", b"F"
#: The tags that start each message from it: its progress, what it wrote, or why it failed.
_PROGRESS, _WRITTEN, _FAILED = b"P", b"W", b"X"
#: Its progress: the events written, their JSON's bytes, and the POSIX time since which it has
#: waited for space on the disk (0 while it does not).
_PROGRESS_FORMAT = "<qqd"


#: The most characters of why a stream failed its record keeps (``_cut``): enough for the end of
#: a traceback, where it names its exception, and the stage's evidence holds no more.
FAILURE_CHARS = 2000


class TraceStreamFailed(RuntimeError):
    """The trace's writer failed; the drive's trace cannot be finished."""


class TraceStreamRecord(BaseModel):
    """How a drive's trace writer kept up with the drive, as the drive role reports it to the
    stage (``hls.drive.record.DriverRecord``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The most events the driver had recorded and the writer had not yet reported written, at
    #: once.
    backlog_peak: int = Field(ge=0)
    #: The most seconds from the driver's recording of an event to the writer's report of it
    #: written.
    lag_max_sec: float = Field(ge=0)
    #: The longest the writer waited at once for space on the node's disk, in seconds; 0 when it
    #: never did.
    space_wait_max_sec: float = Field(ge=0)
    #: Why the stream failed, its last ``FAILURE_CHARS`` characters; None when it did not.
    failure: str | None = Field(max_length=FAILURE_CHARS)

    def summary(self) -> str:
        """The record in a phrase, for the role's last log line."""
        return (f"the trace writer was at most {self.backlog_peak:,} events behind and {self.lag_max_sec:.3f}s late, "
                f"and waited at most {self.space_wait_max_sec:.1f}s for disk space")


class NoSpaceForTrace(RuntimeError):
    """The trace has filled its room and the node's disk has had no space for more for
    ``WAIT_FOR_SPACE_SEC``."""


def _pickled(tag: bytes, value: Any) -> memoryview:
    """*tag* followed by *value* pickled without a memo (``Pickler.fast``). A memo of every
    object of a large reply grows as one table, whose resize holds the interpreter lock for tens
    of milliseconds; without it every hold is a few
    milliseconds. An event is acyclic, which is what the memo would guard against."""
    buf = io.BytesIO()
    buf.write(tag)
    pickler = pickle.Pickler(buf, protocol=pickle.HIGHEST_PROTOCOL)
    pickler.fast = True
    pickler.dump(value)
    return buf.getbuffer()


def _cut(why: str) -> str:
    """*why* in at most ``FAILURE_CHARS`` characters: whole when it fits, else its first line's
    start and its end, where a traceback names its exception."""
    if len(why) <= FAILURE_CHARS:
        return why
    head, gap = why.splitlines()[0][:FAILURE_CHARS // 10], "\n...\n"
    return head + gap + why[-(FAILURE_CHARS - len(head) - len(gap)):]


def _no_space(local_dir: Path, writer: TraceWriter | None, waited: float) -> str:
    """Why a writer gives up on a full disk: its room in *local_dir*, how far *writer*'s trace
    has gone past each room file, and the *waited* seconds."""
    use = {} if writer is None else writer.room_use()
    files = "; ".join(f"{name} holds {written:,} bytes, {max(0, written - made):,} past its {made:,}"
                      for name, (written, made) in use.items())
    return (f"the trace's room in {local_dir} is full ({files or 'its files not yet open'}) and the node's disk has "
            f"had no space for {waited:.0f}s, past the {WAIT_FOR_SPACE_SEC:.0f}s the trace's writer waits for it")


def _write_events(events: Connection, results: Connection, local_dir: Path, driver: DriverDescription,
                  chunk_bytes: int) -> None:
    """The writer process's work: write every event it is sent into the trace's room in
    *local_dir*, sending its progress after each, until it is sent the drive's ending and the
    sessions the driver dropped, as JSON; then finish the trace and send back what it wrote, or
    why it could not."""
    writer: TraceWriter | None = None
    # When the current wait for space began: as a POSIX time, which the role is told, and on the
    # monotonic clock, which bounds it. 0 while the writer does not wait.
    waiting_since, waiting_from = 0.0, 0.0

    def progress() -> None:
        # A full disk may meet the writer before it has made its first file.
        written = (0, 0) if writer is None else (writer.events, writer.event_bytes)
        results.send_bytes(_PROGRESS + struct.pack(_PROGRESS_FORMAT, *written, waiting_since))

    def wait_for_space(error: OSError) -> None:
        nonlocal waiting_since, waiting_from
        now = time.monotonic()
        if waiting_since == 0:
            waiting_since, waiting_from = time.time(), now
            progress()
        elif now - waiting_from >= WAIT_FOR_SPACE_SEC:
            raise NoSpaceForTrace(_no_space(local_dir, writer, now - waiting_from)) from error
        time.sleep(_SPACE_POLL_SEC)

    try:
        writer = TraceWriter.in_room(local_dir, driver=driver, chunk_bytes=chunk_bytes,
                                     wait_for_space=wait_for_space)
        while True:
            message = events.recv_bytes()
            body = memoryview(message)[1:]
            if message[:1] == _EVENT:
                writer.write(pickle.loads(body))
                waiting_since = 0.0
                progress()
            elif message[:1] == _FINISH:
                end = json.loads(bytes(body))
                written = writer.finish(ending=end["ending"],
                                        dropped=[DroppedSession.model_validate(d) for d in end["dropped"]])
                results.send_bytes(_WRITTEN + written.model_dump_json().encode())
                return
            else:
                raise ValueError(f"the trace writer was sent a message tagged {message[:1]!r}")
    except BaseException:
        # Why first, so nothing the abandoning meets can keep it from the role.
        with contextlib.suppress(OSError):
            results.send_bytes(_FAILED + traceback.format_exc().encode())
        if writer is not None:
            writer.abandon()
        raise


class TraceStream:
    """A drive's trace written into its room in the node-local directory *local_dir*
    (``hls.trace.store.make_room``) as the drive goes, by a writer process the role starts
    (``start``) and ends (``close``). *driver* is the driver's description ``END.json`` records,
    *chunk_bytes* each chunk's size (``hls.trace.store.TraceWriter``), and *stall_sec* and
    *finish_sec* the writer's two time limits (``STALL_SEC``, ``FINISH_SEC``)."""

    def __init__(self, local_dir: Path, *, driver: DriverDescription, chunk_bytes: int = CHUNK_TARGET_BYTES,
                 stall_sec: float = STALL_SEC, finish_sec: float = FINISH_SEC) -> None:
        #: The directory of the room the writer process writes the trace into.
        self.local_dir = local_dir
        self._driver = dict(driver)
        self._chunk_bytes = chunk_bytes
        self._stall_sec = stall_sec
        self._finish_sec = finish_sec
        # Events for the sender thread, which never makes ``put`` wait.
        self._queue: queue.SimpleQueue[tuple[bytes, Any]] = queue.SimpleQueue()
        # When each event the writer has not yet reported written was put, oldest first, on the
        # monotonic clock: its length is how far behind the writer is.
        self._put_times: deque[float] = deque()
        # Events whose put time the reader has taken off ``_put_times`` as written.
        self._reported = 0
        self._process: subprocess.Popen | None = None
        self._events: Connection | None = None
        self._results: Connection | None = None
        self._sender = threading.Thread(target=self._send, name="drive-trace-sender", daemon=True)
        self._reader = threading.Thread(target=self._read, name="drive-trace-reader", daemon=True)
        self._watch = threading.Thread(target=self._watch_writer, name="drive-trace-watch", daemon=True)
        self._stop = threading.Event()
        self._result: bytes | None = None
        # Set once the reader has the writer's last word, or the pipe gave none.
        self._result_ready = threading.Event()
        # Set once the writer process has exited after giving its result, or the stream failed.
        self._ended = threading.Event()
        # Whether ``finish`` has told the writer to finish.
        self._finishing = False
        #: Why the stream failed; None while it has not.
        self.error: str | None = None
        #: Events handed to ``put`` so far.
        self.put_events = 0
        #: The most events handed to ``put`` and not yet reported written by the writer, at once.
        self.backlog_peak = 0
        #: The most seconds from an event's ``put`` to the writer's report of it written.
        self.lag_max_sec = 0.0
        #: The events the writer has reported written, and the bytes of their JSON.
        self._written = (0, 0)
        #: The POSIX time since which the writer has waited for space; 0 while it does not.
        self._waiting_since = 0.0
        #: The longest wait for space that has ended, in seconds.
        self._space_wait_max = 0.0
        self._closed = False

    # life

    def start(self) -> TraceStream:
        """Start the writer process and the threads that feed, read and watch it."""
        events_read, events_write = os.pipe()
        results_read, results_write = os.pipe()
        try:
            self._process = subprocess.Popen(
                [sys.executable, "-m", "hls.drive.stream", str(events_read), str(results_write), str(self.local_dir),
                 str(self._chunk_bytes), json.dumps(self._driver)],
                pass_fds=(events_read, results_write), stdin=subprocess.DEVNULL)
        finally:
            # The process holds its own ends; the role keeps the other two.
            os.close(events_read)
            os.close(results_write)
        self._events = Connection(events_write, readable=False)
        self._results = Connection(results_read, writable=False)
        for thread in (self._sender, self._reader, self._watch):
            thread.start()
        return self

    def __enter__(self) -> TraceStream:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """End the writer process, SIGTERM and then SIGKILL, join it and the threads, and close
        the pipes. Safe to call more than once and after ``finish``."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(_TERM_SEC)
            if process.poll() is None:
                process.kill()
                process.wait(_TERM_SEC)
        for end in (self._events, self._results):
            if end is not None:
                with contextlib.suppress(OSError):
                    end.close()
        # The sender may wait on the queue: a last item ends it.
        self._queue.put((_FINISH, None))
        for thread in (self._sender, self._reader, self._watch):
            if thread.is_alive():
                thread.join(_TERM_SEC)

    def _fail(self, why: str) -> None:
        if self.error is None:
            self.error = why
            log.error("the drive's trace writer failed: %s", why)
        self._ended.set()

    # the event loop's side

    def put(self, ev: TraceEvent) -> int:
        """Hand *ev*, the next event the driver recorded, to the writer, and return its number
        in emission order, from 0. Never waits. Raise ``TraceStreamFailed`` once the stream has
        failed."""
        if self.error is not None:
            raise TraceStreamFailed(self.error)
        # Kept before the event is queued, so the writer never reports an event whose time is not.
        self._put_times.append(time.monotonic())
        self._queue.put((_EVENT, ev))
        number = self.put_events
        self.put_events += 1
        self.backlog_peak = max(self.backlog_peak, len(self._put_times))
        return number

    def written(self) -> tuple[int, int]:
        """The events the writer has written, and the bytes of their JSON."""
        return self._written

    @property
    def waiting_for_space(self) -> bool:
        """Whether the writer waits for room on the node's disk."""
        return self._waiting_since != 0

    @property
    def space_wait_max_sec(self) -> float:
        """The longest the writer has waited at once for space on the node's disk, in seconds,
        a wait still going on counted to now; 0 when it never waited."""
        since = self._waiting_since
        return max(self._space_wait_max, time.time() - since if since else 0.0)

    def record(self) -> TraceStreamRecord:
        """How the writer has kept up so far, and why the stream failed, if it did."""
        return TraceStreamRecord(backlog_peak=self.backlog_peak, lag_max_sec=self.lag_max_sec,
                                 space_wait_max_sec=self.space_wait_max_sec,
                                 failure=None if self.error is None else _cut(self.error))

    def finish(self, ending: Ending, dropped: Sequence[DroppedSession] = ()) -> TraceWritten:
        """Once every event was put: have the writer finish the trace with *ending* and the
        sessions the driver *dropped*, and return what it wrote, once its process has exited.
        The watch bounds the wait: the writer must write each waiting event within ``stall_sec``
        of the last, and finish the trace and exit within ``finish_sec`` of its last event. Raise
        ``TraceStreamFailed`` when it could not."""
        if self.error is not None:
            raise TraceStreamFailed(self.error)
        self._finishing = True
        self._queue.put((_FINISH, json.dumps({"ending": ending, "dropped": [d.model_dump() for d in dropped]})))
        self._ended.wait()
        if self.error is not None:
            raise TraceStreamFailed(self.error)
        assert self._result is not None
        return TraceWritten.model_validate_json(self._result)

    # the threads

    def _send(self) -> None:
        """The sender thread: pickle each queued event and send it to the writer process, and
        the drive's ending last."""
        assert self._events is not None
        try:
            while True:
                tag, value = self._queue.get()
                if self._stop.is_set():
                    return
                if tag == _FINISH:
                    self._events.send_bytes(_FINISH + str(value).encode())
                    return
                self._events.send_bytes(_pickled(_EVENT, value))
        except Exception as e:
            self._fail(f"sending an event to the trace writer failed: {type(e).__name__}: {e}")

    def _read(self) -> None:
        """The reader thread: keep the writer's progress and how far behind it is, and its last
        word."""
        assert self._results is not None
        try:
            while True:
                message = self._results.recv_bytes()
                tag = message[:1]
                if tag == _PROGRESS:
                    self._progress(*struct.unpack(_PROGRESS_FORMAT, message[1:]))
                    continue
                if tag == _WRITTEN:
                    self._result = message[1:]
                    self._result_ready.set()
                    self._await_exit()
                else:
                    self._fail(f"the trace writer failed:\n{bytes(message[1:]).decode(errors='replace')}")
                return
        except (EOFError, OSError) as e:
            if not self._stop.is_set():
                self._fail(f"the trace writer ended without a result: {type(e).__name__}: {e}")
        finally:
            self._result_ready.set()

    def _await_exit(self) -> None:
        """Once the writer has given its result: wait for its process to exit, which the watch
        bounds, and let ``finish`` return, failing the stream for an exit status other than 0."""
        assert self._process is not None
        while not self._stop.is_set():
            try:
                status = self._process.wait(_WATCH_SEC)
            except subprocess.TimeoutExpired:
                continue
            if status != 0:
                self._fail(f"the trace writer process finished the trace and then ended with status {status}")
            self._ended.set()
            return

    def _progress(self, events: int, event_bytes: int, waiting_since: float) -> None:
        """Take the writer's report that it has written *events* events of *event_bytes* bytes of
        JSON, and has waited for space since *waiting_since* (0 for not)."""
        now = time.monotonic()
        while self._reported < events and self._put_times:
            self.lag_max_sec = max(self.lag_max_sec, now - self._put_times.popleft())
            self._reported += 1
        if self._waiting_since and not waiting_since:
            self._space_wait_max = max(self._space_wait_max, time.time() - self._waiting_since)
        self._waiting_since = waiting_since
        self._written = (events, event_bytes)

    def _watch_writer(self) -> None:
        """Fail the stream when the writer process ends before it gave its result, has events
        waiting and has written none of them for ``stall_sec``, or has not finished the trace and
        exited ``finish_sec`` after its last event, either beyond ``WAIT_FOR_SPACE_SEC`` while it
        waits for space, so that its own account of a full disk comes first. A failure ends the
        process."""
        assert self._process is not None
        try:
            last, since = -1, time.monotonic()
            while not self._stop.wait(_WATCH_SEC):
                if self._process.poll() is not None:
                    self._writer_exited()
                    return
                written, now = self._written[0], time.monotonic()
                finishing = self._finishing and written >= self.put_events
                if written != last or not (finishing or written < self.put_events):
                    last, since = written, now
                    continue
                allowed = (self._finish_sec if finishing else self._stall_sec) + (
                    WAIT_FOR_SPACE_SEC if self.waiting_for_space else 0.0)
                if now - since < allowed:
                    continue
                if finishing:
                    self._fail(f"the trace writer did not finish the trace within {self._finish_sec:.0f}s of its "
                               "last event")
                else:
                    self._fail(f"the trace writer wrote none of the {self.put_events - written} events waiting for "
                               f"{self._stall_sec:.0f}s")
                with contextlib.suppress(OSError):
                    self._process.kill()
                return
        except Exception as e:
            # Recorded as the stream's failure, so ``finish`` never waits on a watch that is gone.
            self._fail(f"the trace writer's watch failed: {type(e).__name__}: {e}")

    def _writer_exited(self) -> None:
        """The writer process has exited: fail the stream unless it gave its result, which the
        reader then takes from there (``_await_exit``)."""
        assert self._process is not None
        # The reader takes the writer's last word off the pipe.
        self._result_ready.wait(_WATCH_SEC)
        if self._result is None:
            self._fail(f"the trace writer process ended with status {self._process.returncode} before it finished "
                       "the trace")


def main(argv: list[str]) -> int:
    """The writer process: ``python -m hls.drive.stream EVENTS RESULTS DIR CHUNK_BYTES DRIVER``, its
    two pipes' descriptors, the directory of the trace's room, each chunk's size and the driver's
    description as JSON (``_write_events``). Exits 0 once the trace is finished, 1 otherwise."""
    events_fd, results_fd, local_dir, chunk_bytes, driver = argv
    events = Connection(int(events_fd), writable=False)
    results = Connection(int(results_fd), readable=False)
    try:
        _write_events(events, results, Path(local_dir), json.loads(driver), int(chunk_bytes))
    except BaseException:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
