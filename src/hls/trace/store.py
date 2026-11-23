"""The drive trace, written in chunks one event at a time and read back a line at a time.

A drive's trace directory holds:

    chunk-000000.jsonl.gz ...   one event per line, in the order the driver emitted them
    summary.jsonl.gz            one SummaryEvent per event, in the same order
    END.json                    written last: the event count, each chunk's name, event count
                                and sha256, the summary's sha256, and how the drive ended

A directory without ``END.json`` is not a trace: the writer died, and a reader must not take
the chunks it left for a shorter, cleaner drive. Every reader checks each chunk's sha256 and
event count against ``END.json`` before it yields anything from that chunk.

Events, and the summary's records, are written as
``json.dumps(event.model_dump(mode="python"), allow_nan=True)``, so NaN and infinite floats (a
reference-free logprob, an unbounded gap) come back as the driver held them.
``model_dump_json`` would write them as null, which a typed float then refuses. JSON still
forces two conversions: a tuple comes back as a list, and a non-string key inside an opaque
value (``extras``, ``ground_truth``, envelope metadata) comes back as a string, a None key as
"null". Every verifier and metric reads a list where it could read a tuple
(``isinstance(v, (list, tuple))``). A tuple key fails the writer with TypeError, leaving no
trace; no producer writes a None or a tuple key. A set and bytes are written as pydantic
writes them. A speech reply's
packed codec report (``CodecPosition``) is written as a ``TokenLogprob``'s dict and read back
packed.

The writer (``TraceWriter``) serializes one event at a time into a gzip stream, as the drive
hands it each event it recorded, and never builds the whole trace as one object; it leaves the
caller's events untouched. ``write_trace`` writes a whole trace through it, so a trace written
event by event during a drive and one written whole after it are the same bytes. Gzip runs at
level 1 with a zero timestamp: the bytes depend only on the events. The writer returns the
sha256 and size of every file it wrote, counted from the bytes as it wrote them
(``TraceWritten``), so the role that drove reports them for the stage's manifest without
reading the trace a second time.

A drive writes its trace on the node's disk, which the contestant's server shares, into a room
(``make_room``): two files the stage makes before the server starts, every block of them given
then, at the most the trace's chunks and summary can take in gzip. The writer
(``TraceWriter.in_room``) writes the chunks one after another into ``CHUNKS_ROOM`` and the
summary into ``SUMMARY_ROOM``, in place, and makes no file: a server that fills the node's disk
afterwards takes none of the trace's space. A trace past its room grows the room's files, which
takes new space; such a writer may be told to wait out a full disk (``wait_for_space``), which
the server's own disk cap ends. ``publish_trace`` copies each file's stretch of the room into
the trace's directory on the store, checking it against the writer's record, and writes
``END.json`` there last.

Every reader takes a file a piece at a time and holds at most one line of it; a chunk is read
twice, once to check it against ``END.json`` and once to yield its events, so nothing of a bad
chunk is yielded. A reader that takes the events more than once reads them through
``StoredTrace``, whose views read them from the store on each pass and hold none of them.

The summary holds what scoring reads of each event: its identity, its timings, whether it was
answered, and its failure's reason and abandonment time, with ``packet_timestamps`` whole and
the completion time a trailing frame sets (``t_complete``), from which the driver stamped
``t_recv``. Every scorer and SLO reads nothing else, and of a reply's packet timestamps only the
first, the last and how many there are, so ``read_summary`` gives a trace they score exactly as
the full one, holding those three of each reply's packets (``PacketTimes``)."""

from __future__ import annotations

import contextlib
import errno
import gzip
import hashlib
import io
import json
import operator
import os
import time
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, get_args

import pydantic_core
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from hls.core.payloads import Envelope
from hls.core.trace import (ArtifactResponse, DroppedSession, FailureReason, RequestAttempt, RequestFailure, Trace,
                            TraceEvent)
from hls.store.digest import Sha256Hex
from hls.store.manifest import ReportedOutput
from hls.util import validation_errors

#: Bytes of JSON per chunk before the writer starts the next one.
CHUNK_TARGET_BYTES = 64 * 2**20
#: The file written last.
END_FILE = "END.json"
#: The per-event summary file.
SUMMARY_FILE = "summary.jsonl.gz"
#: The room files a drive's trace is written into on the node's disk (``make_room``): the
#: chunks, one after another, and the summary.
CHUNKS_ROOM, SUMMARY_ROOM = "chunks.room", "summary.room"
#: How a drive ended, as the driver reports it: every scheduled turn answered or faulted,
#: cut at the drive wall, or stopped by the circuit breaker or the stall watchdog.
Ending = Literal["drained", "wall", "aborted"]
#: The driver that drove, for people: its class and settings, each as text.
DriverDescription = dict[str, str]
_DRIVER_DESCRIPTION: TypeAdapter[DriverDescription] = TypeAdapter(DriverDescription)
_COMPRESS_LEVEL = 1
#: The most bytes of trace JSON one byte of a value a reply carries can become when the trace
#: writes it again: a DEL (0x7f), which a reply's JSON may carry raw in one byte, is written as
#: the six-byte escape ``\u007f``, since the writer escapes every character outside printable
#: ASCII. Every other value grows less: a character above it takes two wire bytes or more for its
#: six (twelve for four bytes of an astral one), a number at most from ``9e15`` to
#: ``9000000000000000.0``, a field the decoder fills with its default (a logprob entry's ``token``
#: and ``bytes``) from ``{"logprob":0},`` to 78 bytes, under 5.6 times.
ESCAPE_GROWTH = 6
#: The most bytes of trace JSON one harness timestamp takes: the longest ``repr`` of a float, 24
#: characters (``-2.2250738585072014e-308``), and the separator after it.
TIMESTAMP_JSON_BYTES = 25
#: The most bytes of JSON a driver's description may take in ``END.json``; the writer refuses a
#: longer one, so ``END.json`` is bounded. The drive role's names the driver's class, its split and
#: three of its settings, under 200 bytes.
DRIVER_DESCRIPTION_MAX_BYTES = 4096
#: Bytes a reader or a copy takes from a file at once.
_READ_BYTES = 2**20
#: The most plain bytes one step of a reader's decompression yields, so a piece of a file that
#: expands far on decompression is taken a bounded piece at a time.
_PIECE_BYTES = 16 * 2**20


class TraceIncomplete(Exception):
    """The directory does not hold a whole, unaltered trace."""


class ChunkRef(BaseModel):
    """One chunk file as the writer left it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: File name inside the trace directory, ``chunk-<6 digits>.jsonl.gz``.
    name: str = Field(pattern=r"^chunk-\d{6}\.jsonl\.gz$")
    #: Events in it.
    events: int = Field(ge=1)
    #: Hex SHA-256 of the file's bytes.
    sha256: Sha256Hex


class TraceEnd(BaseModel):
    """``END.json``: what a complete trace holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Events in the trace.
    events: int = Field(ge=0)
    #: Every chunk, in order.
    chunks: list[ChunkRef]
    #: Hex SHA-256 of the summary file's bytes.
    summary_sha256: Sha256Hex
    #: How the drive ended.
    ending: Ending
    #: The driver that drove it.
    driver: DriverDescription
    #: When the writer finished, in POSIX seconds.
    written_at: float
    #: The sessions the driver dropped on a dead sandbox, in the order dropped, each with its
    #: reason (``hls.core.trace.DroppedSession``). Left out of the file when there are none, which
    #: is every drive but a trajectory drive whose sandbox died under a session.
    dropped: list[DroppedSession] = Field(default_factory=list, exclude_if=lambda dropped: not dropped)


class PacketTimeNotHeld(LookupError):
    """A reader of a summary event asked for a packet time other than the reply's first or last,
    which the summary's events do not hold (``PacketTimes``)."""


class PacketTimes(Sequence[float]):
    """A reply's packet timestamps as a summary event holds them: the first, the last and how many
    there are, which is all any scorer reads of them (the time to the first packet, the mean
    time per packet, the end of the drive). A reply may carry millions of packets, and a summary
    holding every one of every reply would need memory in proportion to the trace.

    It reads as the list it stands for at index 0 and at the last index, and for its length and
    truth. Every other index, a slice and iteration raise ``PacketTimeNotHeld`` rather than give
    a value it does not hold; an index past either end raises IndexError, as a list's does."""

    __slots__ = ("_count", "_first", "_last")

    def __init__(self, first: float, last: float, count: int) -> None:
        #: The first packet's time.
        self._first = first
        #: The last packet's time.
        self._last = last
        #: How many packets the reply carried.
        self._count = count

    @classmethod
    def of(cls, times: Sequence[float]) -> PacketTimes:
        """The first, the last and the count of *times*. An empty one holds no time."""
        if not times:
            return cls(0.0, 0.0, 0)
        return cls(times[0], times[-1], len(times))

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise PacketTimeNotHeld("a summary event holds a reply's first and last packet times only, not a slice")
        i = operator.index(index)
        if i < 0:
            i += self._count
        if not 0 <= i < self._count:
            raise IndexError("packet time index out of range")
        if i == 0:
            return self._first
        if i == self._count - 1:
            return self._last
        raise PacketTimeNotHeld(f"a summary event holds a reply's first and last packet times only, not "
                                f"packet {i} of {self._count}")

    def __iter__(self) -> Iterator[float]:
        raise PacketTimeNotHeld("a summary event holds a reply's first and last packet times only, so they "
                                "cannot be iterated")

    def __reversed__(self) -> Iterator[float]:
        raise PacketTimeNotHeld("a summary event holds a reply's first and last packet times only, so they "
                                "cannot be iterated")

    def __repr__(self) -> str:
        return f"PacketTimes(first={self._first!r}, last={self._last!r}, count={self._count})"


class SummaryFailure(BaseModel):
    """The part of a request's failure that scoring reads."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Why the harness stopped waiting.
    reason: FailureReason
    #: When it stopped waiting, on the driver's clock.
    t_abandoned: float


class SummaryEvent(BaseModel):
    """One event as scoring reads it: identity and timing, no content.

    Its line is never longer than its event's: each field it holds is one the event's line holds,
    written the same way, and the two of its own (``tenant_id`` and ``answered``) and its nulls
    for a reply's timings take no more than the event's input, ground truth and extras do."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The event's request id.
    request_id: str
    #: Its session; None for a single-turn request.
    session_id: str | None
    #: Its turn within the session.
    turn_index: int
    #: The tenant its input was stamped with; None when unstamped, which no tenant's slice
    #: then holds, as with the full event.
    tenant_id: str | None
    #: When the driver sent it.
    t_send: float
    #: When the response arrived; None on a failure.
    t_recv: float | None
    #: The schedule's intended send time, where the driver records one.
    t_planned_send: float | None
    #: The fault string; None when answered.
    error: str | None
    #: The failure's reason and abandonment time; None when answered.
    failure: SummaryFailure | None
    #: True when the event has a response.
    answered: bool
    #: The response's first packet, on a streamed response.
    t_first_packet: float | None
    #: Every packet's arrival time, on a streamed response.
    packet_timestamps: list[float] | None
    #: When the reply's product was complete, on a protocol whose stream ends with a frame
    #: that is not product; the driver stamped ``t_recv`` from it.
    t_complete: float | None

    @classmethod
    def of_event(cls, ev: TraceEvent) -> SummaryEvent:
        """The summary of *ev*."""
        resp = ev.response
        failure = None if ev.failure is None else SummaryFailure(reason=ev.failure.reason,
                                                                  t_abandoned=ev.failure.t_abandoned)
        return cls(request_id=ev.request_id, session_id=ev.session_id, turn_index=ev.turn_index,
                   tenant_id=ev.tenant_id, t_send=ev.t_send, t_recv=ev.t_recv,
                   t_planned_send=ev.t_planned_send, error=ev.error, failure=failure,
                   answered=resp is not None,
                   t_first_packet=None if resp is None else resp.t_first_packet,
                   packet_timestamps=None if resp is None else resp.packet_timestamps,
                   t_complete=None if resp is None else resp.t_complete)

    def to_event(self) -> TraceEvent:
        """A trace event with this summary's identity and timings and no content: what
        every scorer reads of the full event, and nothing it does not. Of the packet
        timestamps it keeps the first, the last and their count (``PacketTimes``)."""
        metadata = {} if self.tenant_id is None else {"tenant_id": self.tenant_id}
        response = None
        if self.answered:
            packets = None if self.packet_timestamps is None else PacketTimes.of(self.packet_timestamps)
            response = ArtifactResponse(output=Envelope(request_id=self.request_id),
                                        t_first_packet=self.t_first_packet,
                                        packet_timestamps=packets, t_complete=self.t_complete)
        failure = None
        if self.failure is not None:
            failure = RequestFailure(reason=self.failure.reason, t_abandoned=self.failure.t_abandoned,
                                     observed=RequestAttempt())
        return TraceEvent(request_id=self.request_id, session_id=self.session_id, turn_index=self.turn_index,
                          input=Envelope(request_id=self.request_id, metadata=metadata), response=response,
                          t_send=self.t_send, t_recv=self.t_recv, t_planned_send=self.t_planned_send,
                          error=self.error, failure=failure)


def _jsonable(value: Any) -> Any:
    """What ``json.dumps`` cannot write itself (a set, bytes): pydantic's own rendering."""
    return pydantic_core.to_jsonable_python(value)


def _json_line(data: Any) -> bytes:
    """*data* as one line of JSON, NaN and infinite floats kept."""
    return json.dumps(data, allow_nan=True, separators=(",", ":"), default=_jsonable).encode() + b"\n"


def _event_line(ev: TraceEvent) -> bytes:
    return _json_line(ev.model_dump(mode="python"))


def event_line_bytes(ev: TraceEvent) -> int:
    """The bytes of the line *ev* takes in a trace's chunks, its newline included."""
    return len(_event_line(ev))


def input_json_bytes(env: Envelope) -> int:
    """The bytes *env* takes in a trace's chunks as an event's input. The line is written without
    whitespace, so a value nested in it is written exactly as it would be alone."""
    return len(_json_line(env.model_dump(mode="python"))) - len(b"\n")


class _Hashing(io.RawIOBase):
    """A file, open as the descriptor *fd*, that hashes and counts every byte written to it.

    Every write is finished whole, however few bytes one call of the system takes. A write the
    disk has no room for (ENOSPC) is handed to *wait_for_space*, which returns once there may be
    room again or raises; the write is then tried again from the first byte not yet written.
    Without one, such a write raises."""

    def __init__(self, fd: int, wait_for_space: Callable[[OSError], None] | None) -> None:
        #: The descriptor the bytes go to.
        self.fd = fd
        #: What a write the disk has no room for waits on; None raises it.
        self.wait_for_space = wait_for_space
        #: The running hash of everything written.
        self.sha = hashlib.sha256()
        #: Bytes written.
        self.size = 0

    def writable(self) -> bool:
        return True

    def write(self, b) -> int:
        view = memoryview(b).cast("B")
        done = 0
        while done < len(view):
            try:
                n = os.write(self.fd, view[done:])
            except OSError as e:
                if e.errno != errno.ENOSPC or self.wait_for_space is None:
                    raise
                self.wait_for_space(e)
                continue
            self.sha.update(view[done:done + n])
            done += n
        self.size += done
        return done

    def record(self) -> ReportedOutput:
        """The sha256 and size of everything written."""
        return ReportedOutput(sha256=self.sha.hexdigest(), size=self.size)


def _open_new(path: Path) -> int:
    """A descriptor of the new file *path*, open for writing; FileExistsError if it is there."""
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)


class _NewFile:
    """A file of a trace directory, written under a dot-prefixed temp name and renamed into
    place when closed."""

    def __init__(self, path: Path) -> None:
        #: Where the file is published.
        self.path = path
        #: The temp name it is written under.
        self.tmp = path.with_name(f".{path.name}.tmp")
        #: The temp file's descriptor.
        self.fd = _open_new(self.tmp)

    def close(self) -> None:
        """Publish the file whole under its name."""
        os.fsync(self.fd)
        os.close(self.fd)
        os.replace(self.tmp, self.path)

    def abandon(self) -> None:
        """Close the file unpublished and remove it, raising nothing."""
        with contextlib.suppress(OSError):
            os.close(self.fd)
        with contextlib.suppress(OSError):
            self.tmp.unlink(missing_ok=True)


class _Stretch:
    """A file of a trace written into its room: the next stretch of the room file open as *fd*,
    which the room keeps open and syncs (``_Room``)."""

    def __init__(self, fd: int) -> None:
        #: The room file's descriptor, at the stretch's first byte.
        self.fd = fd

    def close(self) -> None:
        """Nothing: the stretch ends where the next one starts."""

    def abandon(self) -> None:
        """Nothing: the room is the stage's to remove."""


class _Directory:
    """Where a trace written into the new directory *out_dir* puts its files: each gzip file
    whole under its name (``_NewFile``), and ``END.json`` through a synced temp file."""

    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True)
        #: The trace's directory.
        self.out_dir = out_dir

    def new_file(self, name: str) -> _NewFile:
        return _NewFile(self.out_dir / name)

    def end(self, data: bytes) -> ReportedOutput:
        return _write_synced(self.out_dir / END_FILE, data)

    def close(self) -> None:
        """Nothing: every file was published as it closed."""

    def abandon(self) -> None:
        """Nothing: a directory without ``END.json`` is not a trace."""


class _Room:
    """Where a trace written into its room (``make_room``) in *local* puts its files: the chunks
    one after another into ``CHUNKS_ROOM`` and the summary into ``SUMMARY_ROOM``, in place, and
    ``END.json`` nowhere, since ``publish_trace`` writes it. Raise FileNotFoundError, opening
    nothing, when the room was not made."""

    def __init__(self, local: Path) -> None:
        # In place: neither emptied nor appended to, so no block is asked of the disk.
        chunks = os.open(local / CHUNKS_ROOM, os.O_WRONLY)
        try:
            summary = os.open(local / SUMMARY_ROOM, os.O_WRONLY)
        except BaseException:
            os.close(chunks)
            raise
        #: The room files' descriptors, each at the first byte not yet written.
        self.chunks, self.summary = chunks, summary
        #: Each room file's size as the stage made it, by its name.
        self.made = {CHUNKS_ROOM: os.fstat(chunks).st_size, SUMMARY_ROOM: os.fstat(summary).st_size}

    def new_file(self, name: str) -> _Stretch:
        return _Stretch(self.summary if name == SUMMARY_FILE else self.chunks)

    def use(self) -> dict[str, tuple[int, int]]:
        """Each room file's bytes written so far and its size as made, by its name."""
        return {CHUNKS_ROOM: (os.lseek(self.chunks, 0, os.SEEK_CUR), self.made[CHUNKS_ROOM]),
                SUMMARY_ROOM: (os.lseek(self.summary, 0, os.SEEK_CUR), self.made[SUMMARY_ROOM])}

    def end(self, data: bytes) -> ReportedOutput:
        return ReportedOutput(sha256=hashlib.sha256(data).hexdigest(), size=len(data))

    def close(self) -> None:
        """Sync the room files and close them."""
        for fd in (self.chunks, self.summary):
            os.fsync(fd)
            os.close(fd)

    def abandon(self) -> None:
        """Close the room files, raising nothing."""
        for fd in (self.chunks, self.summary):
            with contextlib.suppress(OSError):
                os.close(fd)


class _GzipFile:
    """One gzip file of a trace, named *name* in it, compressed into *file* (``_NewFile`` or
    ``_Stretch``)."""

    def __init__(self, name: str, file: _NewFile | _Stretch,
                 wait_for_space: Callable[[OSError], None] | None) -> None:
        #: The file's name in the trace.
        self.name = name
        #: Where its bytes go.
        self.file = file
        #: The hash of the compressed bytes.
        self.hashing = _Hashing(file.fd, wait_for_space)
        #: The compressing stream into it.
        self.gz = gzip.GzipFile(filename="", mode="wb", fileobj=self.hashing, compresslevel=_COMPRESS_LEVEL,
                                mtime=0)
        #: Lines written.
        self.events = 0
        #: Bytes written before compression.
        self.bytes = 0

    def write(self, line: bytes) -> None:
        self.gz.write(line)
        self.events += 1
        self.bytes += len(line)

    def close(self) -> ReportedOutput:
        """Finish the file, publish it, and return the sha256 and size of the bytes written to
        it."""
        self.gz.close()
        self.file.close()
        return self.hashing.record()

    def abandon(self) -> None:
        """Close the file unpublished, never waiting for space on the disk for bytes no one will
        read. The writer is already failing, so an error here is not raised over the one that
        matters."""
        self.hashing.wait_for_space = None
        with contextlib.suppress(OSError, ValueError):
            self.gz.close()
        self.file.abandon()


def _end_json(end: TraceEnd) -> bytes:
    """The bytes of ``END.json`` holding *end*."""
    return end.model_dump_json(indent=2).encode()


def _write_synced(path: Path, data: bytes) -> ReportedOutput:
    """Write *data* to *path* through a synced temp file, and return its sha256 and size."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return ReportedOutput(sha256=hashlib.sha256(data).hexdigest(), size=len(data))


class TraceWritten(BaseModel):
    """What a trace's writer wrote, by its own account of the bytes it wrote."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The end record, as written to ``END.json``.
    end: TraceEnd
    #: Each file of the directory, by name (every chunk, the summary and ``END.json``), to the
    #: sha256 and size of the bytes the writer wrote into it, so that a role reports them
    #: (``hls.store.manifest.report_written_outputs``) without reading the trace again.
    files: dict[str, ReportedOutput]


def _checked_ending(ending: str) -> Ending:
    """*ending* when ``END.json`` can hold it; ValueError otherwise."""
    if ending not in get_args(Ending):
        raise ValueError(f"ending {ending!r} is not one of {get_args(Ending)}")
    return ending  # type: ignore[return-value]


def _checked_driver(driver: Mapping[str, str]) -> DriverDescription:
    """*driver* when ``END.json`` can hold it, in at most ``DRIVER_DESCRIPTION_MAX_BYTES``;
    ValueError otherwise."""
    try:
        checked = _DRIVER_DESCRIPTION.validate_python(dict(driver))
    except ValidationError as e:
        raise ValueError(f"the driver description must map names to text: {validation_errors(e)}") from None
    size = _end_json_bytes(0, checked) - _end_json_bytes(0, {})
    if size > DRIVER_DESCRIPTION_MAX_BYTES:
        raise ValueError(f"the driver description takes {size} bytes, more than {DRIVER_DESCRIPTION_MAX_BYTES}")
    return checked


def most_chunks(event_bytes: int, chunk_bytes: int = CHUNK_TARGET_BYTES) -> int:
    """The most chunks a trace whose events hold at most *event_bytes* of JSON is cut into: a
    chunk closes once it holds *chunk_bytes*, so every one but the last holds at least that."""
    return event_bytes // chunk_bytes + 1


def gzip_disk_bytes(plain_bytes: int, *, files: int) -> int:
    """The most bytes *files* gzip files of the writer take on disk together, holding
    *plain_bytes* between them: zlib's own bound on a deflate stream at the default window and
    memory level the writer's compressor uses (``deflateBound``: the input, a 4,096th of it, a
    16,384th and a 33,554,432nd, and 7 bytes), and gzip's 10-byte header and 8-byte trailer, per
    file."""
    return plain_bytes + (plain_bytes >> 12) + (plain_bytes >> 14) + (plain_bytes >> 25) + files * (7 + 10 + 8)


def _end_json_bytes(chunks: int, driver: DriverDescription) -> int:
    """The bytes of an ``END.json`` naming *chunks* chunks and *driver*, every other field at its
    longest."""
    sha = "0" * 64
    ref = ChunkRef(name="chunk-999999.jsonl.gz", events=2**63 - 1, sha256=sha)
    end = TraceEnd(events=2**63 - 1, chunks=[ref] * chunks, summary_sha256=sha, ending="drained", driver=driver,
                   written_at=-2.2250738585072014e-308)
    return len(end.model_dump_json(indent=2))


def room_bytes(event_bytes: int, *, chunk_bytes: int = CHUNK_TARGET_BYTES) -> dict[str, int]:
    """The size of each room file (``make_room``) of a trace whose events hold at most
    *event_bytes* of JSON in chunks of *chunk_bytes*: the most its chunks take in gzip
    (``gzip_disk_bytes``), and the most its summary takes, whose lines are never longer than
    their events' (``SummaryEvent``)."""
    return {CHUNKS_ROOM: gzip_disk_bytes(event_bytes, files=most_chunks(event_bytes, chunk_bytes)),
            SUMMARY_ROOM: gzip_disk_bytes(event_bytes, files=1)}


def make_room(local: Path, *, event_bytes: int, chunk_bytes: int = CHUNK_TARGET_BYTES) -> None:
    """Make the room files (``room_bytes``) of a trace whose events hold at most *event_bytes*
    of JSON in chunks of *chunk_bytes* in the directory *local*, each at its full size with
    every block given to it now (``posix_fallocate``), so that writing the trace in place
    (``TraceWriter.in_room``) asks the disk for none. Raise FileExistsError if a room file is
    already there."""
    for name, size in room_bytes(event_bytes, chunk_bytes=chunk_bytes).items():
        fd = _open_new(local / name)
        try:
            os.posix_fallocate(fd, 0, size)
            os.fsync(fd)
        finally:
            os.close(fd)


class TraceWriter:
    """Writes a trace into the new directory *out_dir* one event at a time: chunks of about
    *chunk_bytes* bytes of JSON, the summary beside them, and ``END.json`` when it is finished.
    ``in_room`` writes one into a room instead.

    Raise FileExistsError if the directory exists, since a trace has one writer, and ValueError,
    making nothing, for a *driver* ``END.json`` cannot hold. Not thread-safe: one caller writes
    and finishes it."""

    def __init__(self, out_dir: Path, *, driver: Mapping[str, str], chunk_bytes: int = CHUNK_TARGET_BYTES) -> None:
        checked = _checked_driver(driver)
        self._start(_Directory(out_dir), checked, chunk_bytes, None)

    @classmethod
    def in_room(cls, local: Path, *, driver: Mapping[str, str], chunk_bytes: int = CHUNK_TARGET_BYTES,
                wait_for_space: Callable[[OSError], None] | None = None) -> TraceWriter:
        """A writer of a trace into the room (``make_room``) in *local*, whose ``finish`` writes
        no ``END.json``: ``publish_trace`` takes the trace from the room to its directory.
        *wait_for_space*, when given, is what a write the disk has no room for waits on
        (``_Hashing``), which only a trace past its room meets. Raise FileNotFoundError when the
        room was not made, and ValueError for a *driver* ``END.json`` cannot hold, opening
        nothing."""
        checked = _checked_driver(driver)
        writer = cls.__new__(cls)
        writer._start(_Room(local), checked, chunk_bytes, wait_for_space)
        return writer

    def _start(self, places: _Directory | _Room, driver: DriverDescription, chunk_bytes: int,
               wait_for_space: Callable[[OSError], None] | None) -> None:
        #: The driver that drove, as ``END.json`` records it.
        self.driver = driver
        #: Where the trace's files go.
        self._places = places
        #: Bytes of JSON per chunk before the next one starts.
        self.chunk_bytes = chunk_bytes
        #: What a write the disk has no room for waits on; None raises it (``_Hashing``).
        self._wait_for_space = wait_for_space
        self._chunks: list[ChunkRef] = []
        self._files: dict[str, ReportedOutput] = {}
        self._summary = _GzipFile(SUMMARY_FILE, places.new_file(SUMMARY_FILE), wait_for_space)
        self._current: _GzipFile | None = None
        #: Events written so far.
        self.events = 0
        #: Bytes of the events' JSON written so far into the chunks, before compression: what
        #: a drive's trace cap counts (``hls.drive.bounds``).
        self.event_bytes = 0
        #: Bytes of JSON written so far into the summary, before compression. A summary line is
        #: never longer than its event's line (``SummaryEvent``), so never more than
        #: ``event_bytes``.
        self.summary_bytes = 0

    def room_use(self) -> dict[str, tuple[int, int]]:
        """Of a writer in a room (``in_room``), each room file's bytes written so far and its
        size as the stage made it, by its name; past that size the trace asks the disk for more.
        Empty for a writer into a directory."""
        return self._places.use() if isinstance(self._places, _Room) else {}

    def _close_current(self) -> None:
        current = self._current
        assert current is not None
        record = current.close()
        self._files[current.name] = record
        self._chunks.append(ChunkRef(name=current.name, events=current.events, sha256=record.sha256))
        self._current = None

    def write(self, ev: TraceEvent) -> None:
        """Write *ev*, the next event in emission order, into the current chunk and the summary."""
        if self._current is None:
            name = f"chunk-{len(self._chunks):06d}.jsonl.gz"
            self._current = _GzipFile(name, self._places.new_file(name), self._wait_for_space)
        event = _event_line(ev)
        summary = _json_line(SummaryEvent.of_event(ev).model_dump(mode="python"))
        self._current.write(event)
        self._summary.write(summary)
        self.events += 1
        self.event_bytes += len(event)
        self.summary_bytes += len(summary)
        if self._current.bytes >= self.chunk_bytes:
            self._close_current()

    def finish(self, *, ending: Ending, dropped: Sequence[DroppedSession] = ()) -> TraceWritten:
        """Close the last chunk and the summary, write ``END.json`` saying the drive ended
        *ending* and dropped the sessions *dropped* (in a room, only its record), and return what
        was written. Raise ValueError, writing no ``END.json``, for an *ending* it cannot hold."""
        ending = _checked_ending(ending)
        if self._current is not None:
            self._close_current()
        self._files[SUMMARY_FILE] = self._summary.close()
        self._places.close()
        end = TraceEnd(events=self.events, chunks=self._chunks, summary_sha256=self._files[SUMMARY_FILE].sha256,
                       ending=ending, driver=self.driver, written_at=time.time(), dropped=list(dropped))
        self._files[END_FILE] = self._places.end(_end_json(end))
        return TraceWritten(end=end, files=dict(self._files))

    def abandon(self) -> None:
        """Close the files still being written, unpublished. The chunks already published stay,
        and with no ``END.json`` they never read as a trace."""
        for f in (self._summary, self._current):
            if f is not None:
                f.abandon()
        self._places.abandon()


def write_trace(trace: Trace, out_dir: Path, *, ending: Ending, driver: Mapping[str, str],
                chunk_bytes: int = CHUNK_TARGET_BYTES) -> TraceWritten:
    """Write *trace* into the new directory *out_dir* through a ``TraceWriter``: chunks of about
    *chunk_bytes* bytes of JSON, the summary, then ``END.json``. Return the end record and the
    sha256 and size of every file, taken from the bytes as they were written. Raise
    FileExistsError if *out_dir* exists, since a trace has one writer, and ValueError, writing
    nothing, for an *ending* or a *driver* ``END.json`` cannot hold."""
    ending = _checked_ending(ending)
    writer = TraceWriter(out_dir, driver=driver, chunk_bytes=chunk_bytes)
    try:
        for ev in trace.events:
            writer.write(ev)
        return writer.finish(ending=ending)
    except BaseException:
        writer.abandon()
        raise


def _copy_checked(src: io.BufferedReader, dst: Path, record: ReportedOutput, *, room: Path) -> None:
    """Copy the next ``record.size`` bytes of *src*, open on the room file *room*, to *dst*
    through a synced temp file, a piece at a time, and publish it only when they are the bytes
    *record* says were written. Raise TraceIncomplete, leaving no temp file, when they are not."""
    tmp = dst.with_name(f".{dst.name}.tmp")
    fd = _open_new(tmp)
    try:
        out = _Hashing(fd, None)
        left = record.size
        while left and (data := src.read(min(left, _READ_BYTES))):
            out.write(data)
            left -= len(data)
        copied = out.record()
        if copied != record:
            raise TraceIncomplete(f"{room}: its stretch for {dst.name} holds {copied.size} bytes with sha256 "
                                  f"{copied.sha256}, not the {record.size} bytes with sha256 {record.sha256} "
                                  "its writer recorded")
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
        raise
    os.close(fd)
    os.replace(tmp, dst)


def publish_trace(local: Path, out_dir: Path, written: TraceWritten) -> TraceWritten:
    """Copy the finished trace in the room in *local* (``TraceWriter.in_room``), whose writer
    recorded *written*, into the new directory *out_dir*: every chunk in order from its stretch
    of ``CHUNKS_ROOM``, the summary from ``SUMMARY_ROOM``, then ``END.json``, written from the
    end record, each checked against the writer's record before it is published, so *out_dir* is
    a trace only once every file in it is the one written. Return *written*. Raise
    FileExistsError if *out_dir* exists, and TraceIncomplete for a file that is not the one
    recorded."""
    out_dir.mkdir(parents=True)
    with open(local / CHUNKS_ROOM, "rb") as chunks:
        for chunk in written.end.chunks:
            _copy_checked(chunks, out_dir / chunk.name, written.files[chunk.name], room=local / CHUNKS_ROOM)
    with open(local / SUMMARY_ROOM, "rb") as summary:
        _copy_checked(summary, out_dir / SUMMARY_FILE, written.files[SUMMARY_FILE], room=local / SUMMARY_ROOM)
    end = _end_json(written.end)
    record = written.files[END_FILE]
    if ReportedOutput(sha256=hashlib.sha256(end).hexdigest(), size=len(end)) != record:
        raise TraceIncomplete(f"{END_FILE} of the end record is not the {record.size} bytes with sha256 "
                              f"{record.sha256} its writer recorded")
    _write_synced(out_dir / END_FILE, end)
    return written


def read_end(out_dir: Path) -> TraceEnd:
    """``END.json``, after checking that every file it names exists. Raise TraceIncomplete
    otherwise. It does not read the chunks; ``check_trace`` and the readers check those."""
    try:
        end = TraceEnd.model_validate_json((out_dir / END_FILE).read_bytes())
    except FileNotFoundError as e:
        raise TraceIncomplete(f"{out_dir} has no {END_FILE}: the trace was never finished") from e
    except ValidationError as e:
        raise TraceIncomplete(f"{out_dir / END_FILE} is not a trace's end record: "
                              f"{validation_errors(e)}") from None
    missing = [c.name for c in end.chunks if not (out_dir / c.name).is_file()]
    if not (out_dir / SUMMARY_FILE).is_file():
        missing.append(SUMMARY_FILE)
    if missing:
        raise TraceIncomplete(f"{out_dir}: {END_FILE} names files that are not there: {missing}")
    if sum(c.events for c in end.chunks) != end.events:
        raise TraceIncomplete(f"{out_dir}: the chunks hold {sum(c.events for c in end.chunks)} events, "
                              f"{END_FILE} says {end.events}")
    return end


def _gzip_lines(path: Path, sha, *, what: str) -> Iterator[bytes]:
    """The lines of the one-member gzip file *path*, a piece at a time, each compressed byte fed
    to the hash *sha* as it is read. Raise TraceIncomplete for a file that is not one whole gzip
    member: corrupt, cut short, or followed by more bytes."""
    inflate = zlib.decompressobj(wbits=31)
    parts: list[bytes] = []

    def not_whole(why: str) -> TraceIncomplete:
        return TraceIncomplete(f"{what} {path.name} is not a whole gzip file: {why}")

    with open(path, "rb") as f:
        while data := f.read(_READ_BYTES):
            sha.update(data)
            while data:
                if inflate.eof:
                    raise not_whole("bytes follow its gzip member")
                try:
                    piece = inflate.decompress(data, _PIECE_BYTES)
                except zlib.error as e:
                    raise not_whole(str(e)) from None
                data = inflate.unconsumed_tail or inflate.unused_data
                start = 0
                while (newline := piece.find(b"\n", start)) >= 0:
                    parts.append(piece[start:newline])
                    yield b"".join(parts)
                    parts = []
                    start = newline + 1
                if start < len(piece):
                    parts.append(piece[start:])
    if not inflate.eof:
        raise not_whole("it ends before its gzip member does")
    if parts:
        yield b"".join(parts)


def _check_file(path: Path, *, sha256: str, events: int, what: str) -> None:
    """Check the gzip file *path* against *sha256*, then its line count against *events*, holding
    one line of it at a time. Raise TraceIncomplete otherwise."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while data := f.read(_READ_BYTES):
            sha.update(data)
    _check_digest(path, sha, sha256=sha256, what=what)
    _check_count(path, sum(1 for _ in _gzip_lines(path, hashlib.sha256(), what=what)), events=events, what=what)


def _check_digest(path: Path, sha, *, sha256: str, what: str) -> None:
    """Raise TraceIncomplete unless the bytes read of *path*, hashed into *sha*, are the ones
    *sha256* names."""
    actual = sha.hexdigest()
    if actual != sha256:
        raise TraceIncomplete(f"{what} {path.name}: sha256 {actual} is not the {END_FILE} record {sha256}")


def _check_count(path: Path, lines: int, *, events: int, what: str) -> None:
    """Raise TraceIncomplete unless *path* held the *events* lines recorded; it held *lines*."""
    if lines != events:
        raise TraceIncomplete(f"{what} {path.name} holds {lines} events, {END_FILE} says {events}")


def iter_events(out_dir: Path) -> Iterator[TraceEvent]:
    """Every event, in emission order, holding one line at a time. Each chunk is checked whole
    before any of its events is yielded, and checked again as it is read for them."""
    end = read_end(out_dir)
    for chunk in end.chunks:
        path = out_dir / chunk.name
        _check_file(path, sha256=chunk.sha256, events=chunk.events, what="chunk")
        sha, lines = hashlib.sha256(), 0
        for line in _gzip_lines(path, sha, what="chunk"):
            lines += 1
            yield TraceEvent.model_validate(json.loads(line))
        _check_digest(path, sha, sha256=chunk.sha256, what="chunk")
        _check_count(path, lines, events=chunk.events, what="chunk")


def check_trace(out_dir: Path) -> TraceEnd:
    """``END.json`` after checking every chunk and the summary in full. Raise
    TraceIncomplete when anything differs."""
    end = read_end(out_dir)
    for chunk in end.chunks:
        _check_file(out_dir / chunk.name, sha256=chunk.sha256, events=chunk.events, what="chunk")
    _check_file(out_dir / SUMMARY_FILE, sha256=end.summary_sha256, events=end.events, what="summary")
    return end


def read_trace(out_dir: Path) -> Trace:
    """The whole trace, held in memory: for a trace known to be small. A stage reads a stored trace
    through ``StoredTrace`` or the summary instead."""
    return Trace(events=list(iter_events(out_dir)))


def read_summary(out_dir: Path) -> Trace:
    """The trace rebuilt from the summary: every event's identity and timings, no content, and of
    each reply's packets the first, the last and their count (``SummaryEvent.to_event``). The
    summary is read a line at a time and checked against ``END.json`` before any of it is parsed,
    and again as it is."""
    end = read_end(out_dir)
    path = out_dir / SUMMARY_FILE
    _check_file(path, sha256=end.summary_sha256, events=end.events, what="summary")
    sha = hashlib.sha256()
    events = [SummaryEvent.model_validate(json.loads(line)).to_event()
              for line in _gzip_lines(path, sha, what="summary")]
    _check_digest(path, sha, sha256=end.summary_sha256, what="summary")
    _check_count(path, len(events), events=end.events, what="summary")
    return Trace(events=events)


def _answered(ev: TraceEvent) -> bool:
    """Whether *ev*'s request was answered without error, which a summary event says as the full
    one does."""
    return ev.response is not None and not ev.error


#: The tenant filter that keeps every tenant's events.
_EVERY_TENANT = object()


class StoredEvents:
    """The events of the stored trace in *out_dir*, of *tenant* (every tenant's by default) and
    with *answered* only those answered without error, in emission order. Each pass reads them
    from the store (``iter_events``, which checks each chunk before it yields an event of it)
    and holds one at a time: nothing of one pass is kept for the next. The count comes from the
    trace's summary (``StoredTrace.summary``), read once, which holds the tenant and the answer of
    every event."""

    def __init__(self, trace: StoredTrace, *, tenant: object = _EVERY_TENANT, answered: bool = False) -> None:
        self._trace = trace
        self._tenant = tenant
        self._answered = answered

    def _keeps(self, ev: TraceEvent) -> bool:
        return ((self._tenant is _EVERY_TENANT or ev.tenant_id == self._tenant)
                and (not self._answered or _answered(ev)))

    def __iter__(self) -> Iterator[TraceEvent]:
        for ev in iter_events(self._trace.out_dir):
            if self._keeps(ev):
                yield ev

    def __len__(self) -> int:
        return sum(1 for ev in self._trace.summary() if self._keeps(ev))

    def __bool__(self) -> bool:
        return len(self) > 0


class StoredTrace:
    """The trace stored in *out_dir*, for a reader that takes its events more than once without
    holding them, as the verify role does: ``events``, ``events_for`` and
    ``answered`` are views (``StoredEvents``) that read the events from the store on each pass,
    and ``summary`` is the trace's summary (``read_summary``), read once and kept, which holds of
    every event what a reader counts and keys on: its identity, its tenant, whether it was
    answered, and its error."""

    def __init__(self, out_dir: Path) -> None:
        #: The trace's directory.
        self.out_dir = out_dir
        self._summary: list[TraceEvent] | None = None

    def summary(self) -> list[TraceEvent]:
        """Every event of the trace as its summary holds it, in emission order."""
        if self._summary is None:
            self._summary = read_summary(self.out_dir).events
        return self._summary

    @property
    def events(self) -> StoredEvents:
        """Every event."""
        return StoredEvents(self)

    def events_for(self, tenant_id: str) -> StoredEvents:
        """The events of the tenant *tenant_id*."""
        return StoredEvents(self, tenant=tenant_id)

    def answered(self, tenant_id: object = _EVERY_TENANT) -> StoredEvents:
        """The events answered without error, of the tenant *tenant_id* when one is named."""
        return StoredEvents(self, tenant=tenant_id, answered=True)
