"""The drive role's trace, written as the drive goes: a writer process fed through a queue that
never makes the event loop wait and a pickling thread writes the very trace ``write_trace`` writes
of the same events, says how far it has got and how far behind it fell, and is the harness's to
fail and to end, within stated times."""
from __future__ import annotations

import errno
import os
import pickle
import signal
import struct
import threading
import time
from multiprocessing import Pipe
from pathlib import Path
from types import SimpleNamespace

import pytest

from hls.core.trace import Trace
from hls.drive import stream as stream_mod
from hls.drive.stream import TraceStream, TraceStreamFailed, _pickled, _write_events
from hls.trace import store
from hls.trace.store import END_FILE, TraceEnd, TraceWritten, read_trace, write_trace
from tests.trace.test_store import _speech_event
from tests.trace.traces import drive_trace

DRIVER = {"class": "Driver", "split": "test"}


def _events() -> list:
    return [*drive_trace().events, _speech_event()] * 3


def _room(local: Path, *, chunk_bytes: int = store.CHUNK_TARGET_BYTES, events: list | None = None) -> Path:
    """*local*, made with the room *events* take (``_events`` unless given) in chunks of
    *chunk_bytes*, as the stage makes it before the server starts."""
    local.mkdir()
    store.make_room(local, event_bytes=sum(len(store._event_line(ev)) for ev in events or _events()),
                    chunk_bytes=chunk_bytes)
    return local


def _written(stream: TraceStream, events: int) -> None:
    """Wait until the writer has reported *events* events written."""
    deadline = time.monotonic() + 60
    while stream.written()[0] < events and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stream.written()[0] == events


def _same_trace(a: Path, b: Path) -> None:
    """Every file of trace *a* is byte for byte *b*'s, END.json aside from its write time."""
    names = sorted(p.name for p in a.iterdir())
    assert names == sorted(p.name for p in b.iterdir())
    for name in names:
        if name != END_FILE:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ends = [TraceEnd.model_validate_json((d / END_FILE).read_bytes()) for d in (a, b)]
    assert ends[0].model_copy(update={"written_at": 0.0}) == ends[1].model_copy(update={"written_at": 0.0})


@pytest.mark.parametrize("chunk_bytes", [1, 3_000, store.CHUNK_TARGET_BYTES])
def test_the_streamed_trace_is_the_one_write_trace_writes(tmp_path, chunk_bytes):
    events = _events()
    whole = write_trace(Trace(events=events), tmp_path / "whole", ending="wall", driver=DRIVER,
                        chunk_bytes=chunk_bytes)
    local = _room(tmp_path / "local", chunk_bytes=chunk_bytes)
    with TraceStream(local, driver=DRIVER, chunk_bytes=chunk_bytes) as stream:
        assert [stream.put(ev) for ev in events] == list(range(len(events)))
        written = stream.finish("wall")
    store.publish_trace(local, tmp_path / "store", written)
    _same_trace(tmp_path / "store", tmp_path / "whole")
    assert written.end.model_copy(update={"written_at": 0.0}) == whole.end.model_copy(update={"written_at": 0.0})
    assert {n: r for n, r in written.files.items() if n != END_FILE} == \
           {n: r for n, r in whole.files.items() if n != END_FILE}


def test_the_sessions_a_drive_dropped_reach_the_published_end_json(tmp_path):
    """The drive role hands the writer the sessions the driver dropped on a dead sandbox, and the
    published ``END.json`` names each with its tenant and reason. A drive that dropped none
    writes the same ``END.json`` as before, with no such key."""
    from hls.core.trace import DroppedSession

    dropped = [DroppedSession(session_id="r1", tenant_id="t0", reason="sandbox gone: session='r1' restarts=5")]
    events = _events()
    local = _room(tmp_path / "local")
    with TraceStream(local, driver=DRIVER) as stream:
        for ev in events:
            stream.put(ev)
        written = stream.finish("drained", dropped)
    store.publish_trace(local, tmp_path / "store", written)
    end = TraceEnd.model_validate_json((tmp_path / "store" / END_FILE).read_bytes())
    assert end.dropped == dropped and written.end.dropped == dropped

    local = _room(tmp_path / "local2")
    with TraceStream(local, driver=DRIVER) as stream:
        for ev in events:
            stream.put(ev)
        written = stream.finish("drained")
    store.publish_trace(local, tmp_path / "store2", written)
    assert b"dropped" not in (tmp_path / "store2" / END_FILE).read_bytes()


def test_the_writer_says_how_many_events_it_wrote_and_their_bytes(tmp_path):
    events = _events()
    expected = store.TraceWriter(tmp_path / "ref", driver=DRIVER)
    for ev in events:
        expected.write(ev)
    with TraceStream(_room(tmp_path / "local"), driver=DRIVER) as stream:
        for ev in events:
            stream.put(ev)
        deadline = time.monotonic() + 30
        while stream.written()[0] < len(events) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert stream.written() == (len(events), expected.event_bytes)
        assert not stream.waiting_for_space
        stream.finish("drained")
    expected.abandon()


def test_an_event_crosses_to_the_writer_unchanged():
    for ev in _events():
        back = pickle.loads(_pickled(b"E", ev)[1:])
        assert store._event_line(back) == store._event_line(ev)


def test_a_writer_process_that_dies_fails_the_stream_and_every_later_put(tmp_path):
    with TraceStream(_room(tmp_path / "local"), driver=DRIVER) as stream:
        stream.put(_events()[0])
        os.kill(stream._process.pid, signal.SIGKILL)
        with pytest.raises(TraceStreamFailed, match="trace writer"):
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                stream.put(_events()[0])
                time.sleep(0.05)
        with pytest.raises(TraceStreamFailed):
            stream.finish("drained")
    assert stream._process.poll() is not None


def test_a_writer_that_writes_nothing_while_events_wait_fails_the_stream(tmp_path):
    with TraceStream(_room(tmp_path / "local"), driver=DRIVER, stall_sec=1.0) as stream:
        stream.put(_events()[0])
        deadline = time.monotonic() + 30
        while stream.written()[0] < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        os.kill(stream._process.pid, signal.SIGSTOP)
        stream.put(_events()[1])
        deadline = time.monotonic() + 30
        while stream.error is None and time.monotonic() < deadline:
            time.sleep(0.05)
        assert "wrote none of the 1 events waiting for 1s" in stream.error
        with pytest.raises(TraceStreamFailed):
            stream.put(_events()[2])
    assert stream._process.poll() is not None, "the stalled writer was killed and reaped"


def test_the_writer_is_ended_and_reaped_however_the_drive_ends(tmp_path):
    with pytest.raises(RuntimeError, match="the drive raised"), \
            TraceStream(_room(tmp_path / "local"), driver=DRIVER) as stream:
        stream.put(_events()[0])
        raise RuntimeError("the drive raised")
    assert stream._process.poll() is not None
    assert not any(t.is_alive() for t in (stream._sender, stream._reader, stream._watch))
    assert not (tmp_path / "local" / END_FILE).exists()


def test_a_room_never_made_fails_the_writer(tmp_path):
    (tmp_path / "local").mkdir()
    with TraceStream(tmp_path / "local", driver=DRIVER) as stream:
        with pytest.raises(TraceStreamFailed, match="FileNotFoundError"):
            stream.put(_events()[0])
            stream.finish("drained")
    assert list((tmp_path / "local").iterdir()) == []


def test_the_writer_waits_out_a_full_disk_and_says_it_waits(tmp_path, monkeypatch):
    """Run in this process with a disk that is full for its first few writes past the trace's
    room: the writer waits, says since when, and writes the trace once there is room, saying it
    stops waiting."""
    real, calls = os.write, []

    def full_at_first(fd, data):
        calls.append(len(data))
        if len(calls) <= 3:
            raise OSError(28, "No space left on device")
        return real(fd, data)

    monkeypatch.setattr(store.os, "write", full_at_first)
    monkeypatch.setattr(stream_mod.time, "sleep", lambda sec: None)
    local = tmp_path / "local"
    local.mkdir()
    store.make_room(local, event_bytes=10)
    events_in, events_out = Pipe(duplex=False)
    results_in, results_out = Pipe(duplex=False)
    worker = threading.Thread(target=_write_events,
                              args=(events_in, results_out, local, DRIVER, store.CHUNK_TARGET_BYTES))
    worker.start()
    events = drive_trace().events
    for ev in events:
        events_out.send_bytes(_pickled(b"E", ev))
    events_out.send_bytes(b'F{"ending": "drained", "dropped": []}')
    progress = []
    while (message := results_in.recv_bytes())[:1] == b"P":
        progress.append(struct.unpack("<qqd", message[1:]))
    worker.join(30)
    assert message[:1] == b"W", message
    assert progress[0][2] > 0, "the writer said since when it waits while the disk was full"
    assert progress[-1][0] == len(events) and progress[-1][2] == 0
    monkeypatch.undo()
    store.publish_trace(local, tmp_path / "store", TraceWritten.model_validate_json(message[1:]))
    assert len(read_trace(tmp_path / "store").events) == len(events)


def test_put_never_waits_on_a_writer_that_writes_nothing_and_the_stream_says_how_far_behind_it_fell(tmp_path):
    """The event loop hands each event over at once however far the writer has fallen behind:
    with the writer stopped, three times the events a bounded queue of 1,024 held go in with no
    wait. The stream keeps the most events handed over and not yet written at once, and the most
    seconds an event waited to be written."""
    events = drive_trace().events * 231                                  # 3,003 events
    with TraceStream(_room(tmp_path / "local", events=events), driver=DRIVER, stall_sec=600) as stream:
        os.kill(stream._process.pid, signal.SIGSTOP)
        try:
            slowest = 0.0
            for ev in events:
                started = time.perf_counter()
                stream.put(ev)
                slowest = max(slowest, time.perf_counter() - started)
            assert stream.written()[0] == 0
            time.sleep(1.0)
        finally:
            os.kill(stream._process.pid, signal.SIGCONT)
        _written(stream, len(events))
        stream.finish("drained")
    assert slowest < 0.25, f"put waited {slowest:.3f}s"
    assert stream.backlog_peak == len(events)
    assert 1.0 <= stream.lag_max_sec < 60
    assert stream.space_wait_max_sec == 0.0


def test_a_writer_that_stops_after_its_last_event_fails_the_finish_within_its_limit(tmp_path):
    """Every event is written, then the writer stops: the finish, the writer's own closing of
    the trace and its exit, fails once it has taken ``finish_sec``, as the harness's, and says
    so; the writer is ended."""
    events = _events()
    with TraceStream(_room(tmp_path / "local"), driver=DRIVER, finish_sec=1.0) as stream:
        for ev in events:
            stream.put(ev)
        _written(stream, len(events))
        os.kill(stream._process.pid, signal.SIGSTOP)
        started = time.monotonic()
        with pytest.raises(TraceStreamFailed, match="did not finish the trace within 1s of its last event"):
            stream.finish("drained")
        assert time.monotonic() - started < 10
    assert "did not finish the trace" in stream.error
    assert stream._process.poll() is not None


def test_a_writer_that_stalls_with_events_waiting_fails_the_finish_by_the_stall_rule(tmp_path):
    """The finish waits on events the writer has not written: a writer that writes none of them
    for ``stall_sec`` fails it, as it fails the drive."""
    events = _events()
    with TraceStream(_room(tmp_path / "local"), driver=DRIVER, stall_sec=1.0, finish_sec=600) as stream:
        stream.put(events[0])
        _written(stream, 1)
        os.kill(stream._process.pid, signal.SIGSTOP)
        for ev in events[1:]:
            stream.put(ev)
        started = time.monotonic()
        with pytest.raises(TraceStreamFailed, match="wrote none of the"):
            stream.finish("drained")
        assert time.monotonic() - started < 10
    assert stream._process.poll() is not None


def test_the_record_keeps_a_long_failures_first_line_and_its_end():
    """The stage's evidence holds the record's failure, at most ``FAILURE_CHARS`` characters: what
    failed, and the exception a traceback ends with."""
    why = "the trace writer failed:\n" + "  a frame\n" * 1000 + "NoSpaceForTrace: the room is full"
    cut = stream_mod._cut(why)
    assert len(cut) == stream_mod.FAILURE_CHARS
    assert cut.startswith("the trace writer failed:\n...\n") and cut.endswith("NoSpaceForTrace: the room is full")
    assert stream_mod._cut("short") == "short"


def test_a_writer_killed_after_it_was_told_to_finish_fails_the_finish_at_once(tmp_path):
    """The window only the reader of the writer's pipe and the watch of its process can see: the
    writer is told to finish, and dies before it answers."""
    events = _events()
    with TraceStream(_room(tmp_path / "local"), driver=DRIVER, finish_sec=600) as stream:
        for ev in events:
            stream.put(ev)
        _written(stream, len(events))
        os.kill(stream._process.pid, signal.SIGSTOP)
        failed: list[BaseException] = []

        def finish() -> None:
            try:
                stream.finish("drained")
            except BaseException as e:
                failed.append(e)

        finishing = threading.Thread(target=finish)
        finishing.start()
        stream._sender.join(30)
        assert not stream._sender.is_alive(), "the writer was sent the drive's ending"
        os.kill(stream._process.pid, signal.SIGKILL)
        finishing.join(30)
        assert not finishing.is_alive()
    [error] = failed
    assert isinstance(error, TraceStreamFailed) and "ended" in str(error)


def test_a_writer_that_waits_for_space_past_its_bound_fails_and_says_why(tmp_path, monkeypatch):
    """Run in this process with a disk that has 100 bytes of space past each room file and then
    none: the writer waits for space up to ``WAIT_FOR_SPACE_SEC`` and then fails, naming the room,
    how far the trace has gone past it, and how long it waited."""
    real, spare, made = os.write, 100, {}

    def full_past_the_room(fd, data):
        made.setdefault(fd, os.fstat(fd).st_size)
        room = made[fd] + spare - os.lseek(fd, 0, os.SEEK_CUR)
        if room <= 0:
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(fd, bytes(data[:room]))

    clock = [1000.0]

    def sleep(sec: float) -> None:
        clock[0] += sec

    monkeypatch.setattr(store.os, "write", full_past_the_room)
    monkeypatch.setattr(stream_mod, "time", SimpleNamespace(monotonic=lambda: clock[0], sleep=sleep,
                                                            time=lambda: clock[0] + 1e9))
    local = tmp_path / "local"
    local.mkdir()
    store.make_room(local, event_bytes=10)
    events_in, events_out = Pipe(duplex=False)
    results_in, results_out = Pipe(duplex=False)
    raised: list[BaseException] = []

    def write() -> None:
        try:
            _write_events(events_in, results_out, local, DRIVER, store.CHUNK_TARGET_BYTES)
        except BaseException as e:
            raised.append(e)

    worker = threading.Thread(target=write)
    worker.start()
    for ev in drive_trace().events:
        events_out.send_bytes(_pickled(b"E", ev))
    events_out.send_bytes(b'F{"ending": "drained", "dropped": []}')
    progress = []
    while (message := results_in.recv_bytes())[:1] == b"P":
        progress.append(struct.unpack("<qqd", message[1:]))
    worker.join(30)
    assert not worker.is_alive()
    assert [type(e) for e in raised] == [stream_mod.NoSpaceForTrace]
    assert message[:1] == b"X", message
    why = message.decode().strip().splitlines()[-1]
    assert f"the trace's room in {local} is full" in why
    room = store.room_bytes(10)[store.CHUNKS_ROOM]
    assert f"{store.CHUNKS_ROOM} holds {room + spare:,} bytes, {spare} past its {room:,}" in why
    assert f"no space for {stream_mod.WAIT_FOR_SPACE_SEC:.0f}s" in why
    assert progress and progress[-1][2] > 0, "the writer said since when it waited"
