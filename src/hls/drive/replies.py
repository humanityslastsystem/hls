"""The drive role's gate on the contestant's replies: what it reads of each, and when it reads no
more.

The role writes every event into the drive's trace as the drive goes (``hls.drive.stream``), so
what grows with the drive is the trace, which its task caps (``hls.drive.bounds``). This module
holds the drive to that cap without touching the driver or its client: for the length of the
drive, ``counting_replies`` wraps the ``HTTPArtifactClient``'s ``submit``, which hands it each
request's envelope as the driver sends it, and httpx's own transport, the one every request of the
client goes through, and hands back each reply from the artifact's origin as a response that
counts its body as the client reads it.

- Only requests to and replies from the artifact's origin (scheme, host and port) are touched.
- Each such request is read as every harness client of a served system reads one
  (``hls.drive.reading``): it asks for an uncompressed reply (``Accept-Encoding: identity``), a
  reply in any other content coding fails at its headers, before any of its body is read
  (``CompressedReply``), and a reply is counted in the bytes that came over the connection, as the
  client reads them. Status line and headers are not counted.
- A reply's bytes are held from the moment they are read until the trace's writer has written its
  event (``ReplyGate``): charged to the asyncio task that reads them, handed with the event when
  the driver records it in the same task (``ReplyGate.detach``, ``ReplyGate.queued``), and released
  once the writer reports that event written. What a task still holds when it ends is released
  then, so a reply whose event another task records is held until its own task ends.
- A request is held the same way from the moment the client is handed it, charged to the task
  that sends it, at the trace JSON its event takes beside its reply: its envelope, which the event
  records as its input, measured as the trace writes it, and the event's own fields at their
  longest (``hls.drive.bounds.request_trace_bytes``). It is sent whether or not the gate is open.
- The gate closes, and stays closed, once the trace written so far, the replies held, at the
  most trace one byte of them can become (``WireProtocol.trace_growth``), and the requests held
  reach the task's cap: so no reply carries the trace past it, with every request sent before
  it closed. It also closes when the driver's memory guard fires (``hls.drive.memory``). Once
  closed, a reply being read fails at its next piece and every later reply at its headers. The
  turns the driver sends after it closed are recorded with their requests, which the task's
  request bytes bound (``hls.drive.bounds``).
- A reply past ``REPLY_MAX_BYTES`` fails at the piece that passes it, since a reply is held whole
  while it is read and decoded.

Every refusal is ``RefusedReply``, an ``ArtifactProtocolError``, which the client passes through
and the driver records as the request's failure, the contestant's: a failed event with
``failure.reason`` protocol, its error ``protocol: <why>``, and for a streamed reply the frames that
arrived before it as the partial output. It is an ``httpx.TransportError`` too, so where the client
tolerates a transport error, as its warm-up probe does, it tolerates this one.

The gate lives on the drive's event loop: every method but ``close`` is called there, so its
counts need no lock."""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator
from dataclasses import dataclass

import httpx

from hls.artifact.artifact_client import HTTPArtifactClient
from hls.core.payloads import Envelope
from hls.core.trace import ArtifactResponse, RequestAttempt
from hls.drive.bounds import request_trace_bytes
from hls.drive.reading import IDENTITY, RefusedReply, refuse_coding

#: What every refusal of the gate records starts with.
WHY = "reply past the drive's bounds"


class ReplyOverBounds(RefusedReply):
    """A reply past ``REPLY_MAX_BYTES``, or one read once the gate closed."""


@dataclass(frozen=True)
class Held:
    """What one task holds until its event is written: the bytes of the replies it read, and
    the trace JSON the events of the requests it sent take beside their replies
    (``hls.drive.bounds.request_trace_bytes``)."""

    #: Reply bytes, counted at the protocol's growth (``ReplyGate.growth``).
    replies: int = 0
    #: Bytes of trace JSON the requests' events take beside their replies.
    requests: int = 0

    def __add__(self, other: Held) -> Held:
        return Held(self.replies + other.replies, self.requests + other.requests)


class ReplyGate:
    """The drive's count of the contestant's reply bytes against its two bounds: one reply's
    ``reply_max_bytes``, and the trace's ``trace_cap_bytes``, which the trace written so far
    (*written*, the writer's events and their JSON's bytes), the replies held at *growth* trace
    bytes per byte, and the requests held, each at the JSON its event takes beside its reply
    (``request_trace_bytes``), together may not reach."""

    def __init__(self, *, reply_max_bytes: int, trace_cap_bytes: int, growth: float,
                 written: Callable[[], tuple[int, int]]) -> None:
        #: The most bytes one reply may carry.
        self.reply_max_bytes = reply_max_bytes
        #: The most bytes of JSON the trace's events may hold.
        self.trace_cap_bytes = trace_cap_bytes
        #: The most trace bytes one held reply byte can become.
        self.growth = growth
        self._written = written
        #: Reply bytes read and not yet written into the trace, every task's together.
        self.held = 0
        #: Bytes of trace JSON the events of the requests sent and not yet written take beside
        #: their replies, every task's together.
        self.held_requests = 0
        #: The most reply bytes held at once over the drive.
        self.held_peak = 0
        #: The most bytes of trace JSON of requests held at once over the drive.
        self.held_requests_peak = 0
        #: Reply bytes read over the drive, every reply's counted.
        self.total = 0
        #: Why the gate is closed; None while it is open.
        self.closed: str | None = None
        self._by_task: dict[asyncio.Task, Held] = {}
        # (event number, what it held), in the order the events were recorded.
        self._queued: deque[tuple[int, Held]] = deque()

    def close(self, why: str) -> None:
        """Fail every reply from now on, with *why*. Another thread may call it."""
        if self.closed is None:
            self.closed = why

    def _hold(self, held: Held) -> None:
        """Add *held* to what the gate counts as held."""
        self.held += held.replies
        self.held_requests += held.requests
        self.held_peak = max(self.held_peak, self.held)
        self.held_requests_peak = max(self.held_requests_peak, self.held_requests)

    def _release(self, held: Held) -> None:
        """Take *held* off what the gate counts as held."""
        self.held -= held.replies
        self.held_requests -= held.requests

    def _release_written(self) -> int:
        """Release what the events the writer has written held, and return their JSON's bytes."""
        events, event_bytes = self._written()
        while self._queued and self._queued[0][0] < events:
            self._release(self._queued.popleft()[1])
        return event_bytes

    def _check(self) -> None:
        """Raise ``ReplyOverBounds`` when the gate is closed, closing it first when the trace
        written, the replies held and the requests held reach the cap."""
        if self.closed is None:
            event_bytes = self._release_written()
            if event_bytes + self.growth * self.held + self.held_requests >= self.trace_cap_bytes:
                self.close(f"the drive's trace reached its cap of {self.trace_cap_bytes:,} bytes "
                           f"({event_bytes:,} written, {self.held:,} reply bytes and {self.held_requests:,} bytes "
                           "of requests held)")
        if self.closed is not None:
            raise ReplyOverBounds(f"{WHY}: {self.closed}")

    def admit(self) -> None:
        """Refuse a reply whose headers arrive once the gate is closed."""
        self._check()

    def _charge(self, held: Held) -> None:
        """Hold *held* for the current task, or for none outside an event loop's task."""
        task = _current_task()
        if task is not None:
            if task not in self._by_task:
                self._by_task[task] = Held()
                task.add_done_callback(self._task_done)
            self._by_task[task] += held
        self._hold(held)

    def sent(self, request: Envelope) -> None:
        """Hold *request*, which the current task sends, at the trace JSON its event takes beside
        its reply (``request_trace_bytes``). It is sent whether or not the gate is open: its reply
        is refused at its headers."""
        self._charge(Held(requests=request_trace_bytes(request)))

    def count(self, reply_bytes: int, n: int) -> None:
        """Count *n* more bytes of a reply that has now carried *reply_bytes*, held by the
        current task, or by none when it is read outside an event loop's task."""
        self._charge(Held(replies=n))
        self.total += n
        if reply_bytes > self.reply_max_bytes:
            raise ReplyOverBounds(f"{WHY}: this reply passed {self.reply_max_bytes:,} bytes, the most one reply "
                                  "may carry")
        self._check()

    def detach(self) -> Held:
        """What the current task holds, now handed over with the event it is recording."""
        task = _current_task()
        if task is None or task not in self._by_task:
            return Held()
        held, self._by_task[task] = self._by_task[task], Held()
        return held

    def queued(self, number: int, held: Held) -> None:
        """Hold *held* until the writer has written event *number*."""
        if held != Held():
            self._queued.append((number, held))
        self._release_written()

    def _task_done(self, task: asyncio.Task) -> None:
        self._release(self._by_task.pop(task, Held()))


def _current_task() -> asyncio.Task | None:
    """The asyncio task running now; None outside a running event loop."""
    try:
        return asyncio.current_task()
    except RuntimeError:
        return None


class _CountedResponse(httpx.Response):
    """A response from the artifact that counts its body against the gate as it is read.

    Every read of the client takes the body off the connection through ``aiter_raw``: ``aread``
    for a unary reply and a streamed error body, ``aiter_lines`` for a stream. Once the body has
    been read, it is replayed from memory, which reads nothing more and is not counted again."""

    def __init__(self, response: httpx.Response, gate: ReplyGate) -> None:
        super().__init__(response.status_code, headers=response.headers, stream=response.stream,
                         extensions=response.extensions)
        self._gate = gate
        #: Bytes of this reply read so far.
        self._reply_bytes = 0

    async def aiter_raw(self, chunk_size: int | None = None) -> AsyncIterator[bytes]:
        async for chunk in super().aiter_raw(chunk_size):
            self._reply_bytes += len(chunk)
            self._gate.count(self._reply_bytes, len(chunk))
            yield chunk


def _origin(url: httpx.URL) -> tuple[str, str, int | None]:
    return url.scheme, url.host, url.port


@contextlib.contextmanager
def counting_replies(artifact_url: str, gate: ReplyGate | None) -> Iterator[None]:
    """For the length of the block, hold every request an ``HTTPArtifactClient`` of
    *artifact_url*'s origin sends against *gate*, ask that origin for uncompressed replies and
    count every reply from it against *gate*; None touches nothing. The client and the transport
    are restored when the block ends, however it ends."""
    if gate is None:
        yield
        return
    origin = _origin(httpx.URL(artifact_url))
    real_submit = HTTPArtifactClient.submit
    real = httpx.AsyncHTTPTransport.handle_async_request

    async def submit(self: HTTPArtifactClient, env: Envelope, *,
                     attempt: RequestAttempt | None = None) -> ArtifactResponse:
        if _origin(httpx.URL(self.base_url)) == origin:
            gate.sent(env)
        return await real_submit(self, env, attempt=attempt)

    async def handle_async_request(self: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        if _origin(request.url) != origin:
            return await real(self, request)
        request.headers.update(IDENTITY)
        response = await real(self, request)
        try:
            gate.admit()
            refuse_coding(response)
        except RefusedReply:
            await response.aclose()
            raise
        return _CountedResponse(response, gate)

    HTTPArtifactClient.submit = submit  # type: ignore[method-assign]
    httpx.AsyncHTTPTransport.handle_async_request = handle_async_request  # type: ignore[method-assign]
    try:
        yield
    finally:
        HTTPArtifactClient.submit = real_submit  # type: ignore[method-assign]
        httpx.AsyncHTTPTransport.handle_async_request = real  # type: ignore[method-assign]
