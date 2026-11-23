"""ArtifactClient + HTTPArtifactClient.

The harness's view of the artifact's HTTP surface. The client is thin transport: it
asks the contract's WireProtocol to encode/decode, branches on transport (unary POST
vs SSE), and returns an ArtifactResponse (content + harness-observed timing).

Design contract: the substrate's deploy path guarantees the artifact is fully ready
(model loaded, inference endpoint serving) before returning a client. The client
therefore trusts that any 5xx response after deploy reflects a real error, not a
transient load state, and raises immediately.
"""

from __future__ import annotations
import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any

import httpx

from hls.core.faults import (ArtifactFault, ArtifactProtocolError, ArtifactStallError,
                             ArtifactTransportError)
from hls.spec.contracts import ArtifactContract, RequestResponseContract
from hls.core.payloads import Envelope
from hls.core.trace import ArtifactResponse, RequestAttempt

if TYPE_CHECKING:
    from hls.spec.wire import WireProtocol

#: Fallback per-request cap for clients not governed by a drive deadline (the
#: driver calls ``clear_request_cap`` and enforces its own completion deadline).
#: A single attempt, no retries -- a retry would distort the timing this benchmark
#: measures. Exceeding it raises an ArtifactTransportError (a failed event with
#: timing), not a silent retry.
REQUEST_TIMEOUT_SEC = 600.0

#: How long an idle pooled connection is kept before the client retires it. Must stay
#: comfortably below the served stack's own idle timeout, which is 5 s for uvicorn (its
#: ``--timeout-keep-alive`` default) and therefore for every server in these tasks.
#: httpx's default is also 5.0, which puts both ends on the same deadline; see the note
#: where the client is built.
_KEEPALIVE_EXPIRY_SEC = 2.0

#: How much of an error response's body the fault message carries. The served backend
#: states why it rejected a request in the body ("This model's maximum context length is
#: 8192 tokens. However, you requested 9200..."); the status code alone only says that it
#: did. Bounded because a 5xx may return a whole HTML page or traceback, and this string is
#: recorded on every failed TraceEvent.
_MAX_FAULT_BODY_CHARS = 300


def _status_fault_message(response: "httpx.Response") -> str:
    """The message for an HTTP error status: the code plus the server's own explanation
    from the body when there is one.

    The body is the only place a rejection's cause appears; without it a context-length
    overflow, a bad sampling parameter, and an unknown model are all indistinguishable
    "artifact returned 400". Bounded to _MAX_FAULT_BODY_CHARS and flattened to one line
    (this lands in TraceEvent.error, jsonl rows, and log lines).
    """
    try:
        body = response.text
    except httpx.ResponseNotRead:
        # A streamed response whose body was never read (see _post_sse, which reads it
        # eagerly on error precisely so this stays reachable only in odd paths).
        body = ""
    body = " ".join(body.split())
    if not body:
        return f"artifact returned {response.status_code}"
    if len(body) > _MAX_FAULT_BODY_CHARS:
        body = body[:_MAX_FAULT_BODY_CHARS] + "..."
    return f"artifact returned {response.status_code}: {body}"


def _json_body(text: str | bytes) -> Any:
    """A response body or SSE frame as JSON, or ArtifactProtocolError when Python's JSON
    reader refuses it.

    It refuses malformed text (JSONDecodeError) and two kinds of valid JSON: an integer
    longer than the interpreter's 4,300-digit conversion limit (a plain ValueError) and
    nesting past the recursion limit (RecursionError). Each is the server's own output,
    so each fails its own request rather than escaping the fault taxonomy."""
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as e:
        raise ArtifactProtocolError(f"non-JSON body: {e}") from e


class ArtifactClient(ABC):
    """Transport-specific handle to the artifact's HTTP surface (L2).

    `submit(env) -> ArtifactResponse` sends one request and returns the decoded
    content plus any harness-observed timing; exceptions propagate rather than
    being swallowed. Subclass per transport; `make_artifact_client` selects the
    concrete implementation from the contract.

    A caller that wants to know how far a failed request got passes a
    ``RequestAttempt`` for the transport to write as bytes arrive. It is a parameter
    rather than part of the return value because a request the caller abandons never
    returns at all (see ``RequestAttempt``); ``None`` simply means nobody is recording.
    """

    @abstractmethod
    async def submit(self, env: Envelope, *,
                     attempt: RequestAttempt | None = None) -> ArtifactResponse: ...

    async def warmup(self) -> None:
        """Best-effort: establish the reusable transport connection before the
        driver anchors its schedule clock, so the first scheduled request reuses
        a warm keep-alive connection instead of paying a TCP+TLS handshake on the
        hot path. Default no-op; transports holding a reused client
        (HTTPArtifactClient) override it. Must never raise -- on failure, the first
        request just pays the handshake instead."""

    async def aclose(self) -> None:
        """Release any transport resources (connection pool). Default no-op;
        transports holding a reused client (HTTPArtifactClient) override it."""


#: What starts a stream line that carries a frame; ``HTTPArtifactClient`` skips every other line.
SSE_DATA_PREFIX = "data:"


def sse_line_bytes(chunk: Mapping[str, Any]) -> int:
    """The fewest reply bytes a stream frame carrying *chunk* takes as this client reads a stream,
    a line at a time: ``SSE_DATA_PREFIX``, the chunk as compact JSON and the newline that ends
    the line. A server may spend more (a space after the prefix, a blank line between frames),
    never less."""
    return len((SSE_DATA_PREFIX + json.dumps(chunk, separators=(",", ":")) + "\n").encode())


def packet_wire_bytes(contract: ArtifactContract) -> int | None:
    """The fewest reply bytes one packet timestamp costs on a reply under *contract*, as this
    client reads it: None for a whole reply (``http``), which times no packet; on a stream, what
    the contract's protocol says one costs (``WireProtocol.stream_packet_bytes``), each frame
    taking ``sse_line_bytes`` of its chunk. Raise ValueError for a contract this client does not
    serve, or a stream whose protocol does not say."""
    if not isinstance(contract, RequestResponseContract):
        raise ValueError(f"no packet cost for a {type(contract).__name__}: the artifact client serves none")
    if contract.transport == "http":
        return None
    cost = contract.protocol.stream_packet_bytes(sse_line_bytes)
    if cost is None:
        raise ValueError(f"no packet cost for a {type(contract.protocol).__name__} stream: the protocol does not say "
                         "what one packet timestamp of its stream costs (WireProtocol.stream_packet_bytes)")
    return cost


def trace_growth(contract: ArtifactContract) -> float:
    """The most bytes of trace JSON one byte of a reply under *contract* can become, as this client
    reads replies: its protocol's (``WireProtocol.trace_growth``), of a stream on ``http-sse``
    and of a whole reply on ``http``. Raise ValueError for a contract this client does not
    serve, or a stream whose protocol does not say."""
    if not isinstance(contract, RequestResponseContract):
        raise ValueError(f"no trace growth for a {type(contract).__name__}: the artifact client serves none")
    return contract.protocol.trace_growth(stream=contract.transport == "http-sse")


class HTTPArtifactClient(ArtifactClient):
    def __init__(self, contract: RequestResponseContract, base_url: str,
                 request_timeout_sec: float | None = None,
                 stall_timeout_sec: float | None = None) -> None:
        self.contract = contract
        self.base_url = base_url.rstrip("/")
        self.request_timeout_sec = (
            request_timeout_sec if request_timeout_sec is not None else REQUEST_TIMEOUT_SEC
        )
        #: Seconds a stream may go between two tokens before the request fails
        #: (``ArtifactStallError``), or ``None`` for no such bound
        #: (``AgentSpec.stall_timeout_sec``). Armed by the first token: the wait before
        #: one is prefill and queueing, which the latency ratio prices. The read timeout
        #: cannot stand in for it, since it also bounds that wait and measures bytes,
        #: not tokens.
        self.stall_timeout_sec = stall_timeout_sec
        # One reused client, built once rather than per request: a per-call client
        # opens a fresh connection pool and TCP/TLS handshake per request, inflating
        # the very latency/throughput this benchmark measures. The per-request
        # deadline is passed at call time (httpx request-level timeout), so the
        # shared client carries the deadline without retries. Must be closed
        # (aclose / async with).
        # trust_env=False: the artifact serves at an internal ``<host>:<port>``, which an
        # ambient HTTP proxy in the environment would not reach, so benchmark traffic
        # dials the artifact directly.
        # max_connections=None: this client is an open-loop load generator, so a
        # request must reach the server the moment its arrival fires -- the server's
        # queue is the thing under test. httpx's default pool (100) queues overflow
        # client-side instead, which silently converts the workload to closed-loop
        # above 100 in-flight and can livelock (httpcore rescans every queued request
        # against every pooled connection on each lifecycle event while holding the
        # GIL, so a few thousand queued SSE streams starve the event loop).
        # max_keepalive_connections=512: generous, but bounded.
        # Bounding it at 100 while max_connections is unbounded means every connection
        # past the hundredth is closed as soon as its request finishes and reopened for
        # the next one, so the pool churns handshakes in proportion to how far
        # concurrency exceeds 100. On the sealed-container path that traffic crosses
        # pasta, which runs its own TCP handshake for the forwarded port, and a SYN that
        # is not accepted retries on a fixed ladder -- 1, 2, 3, 4, then 6 and 10 seconds,
        # giving up 8 after the last. Requests killed that way appear as transport
        # ReadError with nothing in the server's log, because the
        # connection never finished its handshake and the application never saw it.
        # Under a no-failed-request rule each one zeroes a run. A pool large enough that
        # normal operation never churns removes
        # what feeds it: 512 is above the in-flight count a task's pinned load reaches
        # (its request rate times its latency at the pin), so no connection is retired
        # for want of a slot.
        #
        # Not unbounded. With max_connections already
        # unbounded, removing the keepalive bound too leaves nothing capping open sockets:
        # an open-loop driver meeting a server that slows keeps opening connections until
        # file descriptors and the accept queue give out.
        # keepalive_expiry is deliberately BELOW the servers' idle timeout rather than at
        # httpx's default. That default is 5.0s and uvicorn's --timeout-keep-alive is
        # also 5s, so both ends retire the same idle socket at the same instant and the
        # client can hand a request to a connection the server has just closed. It
        # surfaces as `transport: ReadError('')`, and under a no-failed-request rule one
        # of those zeroes an entire run. Retiring first makes the race unreachable: the
        # client opens a fresh connection, which is what it would have done anyway.
        #
        # It arises when an off-clock warm-up has left the pool idle before the scored
        # drive begins.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout_sec), trust_env=False,
            limits=httpx.Limits(max_connections=None, max_keepalive_connections=512,
                                keepalive_expiry=_KEEPALIVE_EXPIRY_SEC))

    async def warmup(self) -> None:
        """Open a pooled keep-alive connection on the shared client before the driver
        starts dispatching. A GET to the artifact origin's ``/health`` shares the
        same (scheme, host, port) as the inference POSTs, so it warms the same
        connection they reuse. Best-effort: any transport error is swallowed, since
        the host-side readiness gate already proved the server serves.

        Tolerant of a client built via ``__new__`` without ``__init__`` (test
        doubles that override ``_post`` to hit an in-process ASGI app and never
        assign ``_client``): with no pooled client there is nothing to warm, so
        this is a no-op, matching ``aclose``."""
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            await client.get(self.base_url + "/health",
                             timeout=httpx.Timeout(5.0))
        except httpx.HTTPError:
            pass

    async def aclose(self) -> None:
        """Close the underlying httpx client and its connection pool. Tolerant of a
        client built via ``__new__`` without ``__init__`` (test doubles that set
        their own ``_post``), where ``_client`` was never assigned."""
        client = getattr(self, "_client", None)
        if client is not None:
            await client.aclose()

    def clear_request_cap(self) -> None:
        """Drop the per-request transport cap: the driver enforces a single
        drive-completion deadline instead (see Driver._drive), under which a
        request queued early may legitimately wait far longer than any fixed
        cap. ``httpx.Timeout(None)`` then disables every transport timeout;
        the drive deadline bounds the whole request."""
        self.request_timeout_sec = None

    async def __aenter__(self) -> "HTTPArtifactClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def _post(self, json_body: dict[str, Any]) -> dict[str, Any]:
        """POST to the artifact inference endpoint and return the response JSON.

        Fails fast on any HTTP error: the drive stage's readiness gate
        (``hls.drive.readiness.wait_inference_ready``) already verified that the artifact can
        serve completions, so any 5xx here is a real server-side failure, not a
        transient model-load state. Raising immediately gives the caller an
        accurate error instead of silently retrying for minutes.
        """
        url = self.base_url + self.contract.path
        r = await self._client.post(
            url, json=json_body, timeout=httpx.Timeout(self.request_timeout_sec))
        r.raise_for_status()
        return _json_body(r.content)

    async def submit(self, env: Envelope, *,
                     attempt: RequestAttempt | None = None) -> ArtifactResponse:
        """Send one request; return the decoded ArtifactResponse.

        Every failure surfaces as an ArtifactFault: transport errors (5xx, connect,
        timeout) as ArtifactTransportError, a non-JSON body or a protocol shape
        violation as ArtifactProtocolError, a stream past its stall budget as
        ArtifactStallError. The driver catches exactly ArtifactFault.
        No retries.

        ``attempt`` (when given) is filled as the response arrives, so a request that
        fails leaves a record of how far it got. This client is the only witness to
        that: the frames it collects are unreachable once its frame unwinds, whether
        the unwinding is an exception or the caller cancelling it.
        """
        proto = self.contract.protocol
        streaming = self.contract.transport == "http-sse"
        body = proto.encode_request(env, stream=streaming)
        try:
            if not streaming:
                out = await self._post(body)
                return ArtifactResponse(output=proto.decode_response(out, env.request_id))
            timed_chunks: list[tuple[float, dict]] = []
            frames = self._post_sse(body)
            if self.stall_timeout_sec is not None:
                frames = self._until_stalled(frames, proto, self.stall_timeout_sec)
            try:
                async for chunk in frames:
                    t = time.monotonic()
                    timed_chunks.append((t, chunk))
                    if attempt is not None:
                        attempt.note_frame(t)
                return proto.decode_stream(timed_chunks, env.request_id)
            except BaseException:                       # noqa: BLE001 -- re-raised below
                # Includes the CancelledError of a caller that gave up waiting, which
                # is exactly the case with no other record: decode what arrived before
                # the frames go out of scope, then let the original failure through.
                if attempt is not None:
                    attempt.partial = proto.decode_partial(timed_chunks)
                raise
        except ArtifactFault:
            raise                                       # protocol shape checks: already typed
        except httpx.TimeoutException as e:
            raise ArtifactTransportError(f"request exceeded {self.request_timeout_sec}s") from e
        except httpx.HTTPStatusError as e:
            # A served rejection, distinct from a connection that never delivered one.
            if attempt is not None:
                attempt.status_code = e.response.status_code
            raise ArtifactTransportError(_status_fault_message(e.response)) from e
        except httpx.HTTPError as e:                    # connect/read/transport errors
            raise ArtifactTransportError(repr(e)) from e

    @staticmethod
    async def _until_stalled(frames: AsyncGenerator[dict[str, Any], None], proto: WireProtocol,
                             stall_sec: float) -> AsyncIterator[dict[str, Any]]:
        """*frames*, failing with ``ArtifactStallError`` once *stall_sec* pass after a token
        with no further one.

        The deadline runs from the last frame that carried a token
        (``WireProtocol.stream_token_count``), so a frame carrying none spends the budget
        rather than renewing it: a server sending empty frames through a stuck decode
        still fails. Before the first token there is no deadline at all."""
        loop = asyncio.get_running_loop()
        tokens = 0
        deadline: float | None = None
        try:
            while True:
                window = asyncio.timeout_at(deadline)
                try:
                    async with window:
                        chunk = await anext(frames)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    if not window.expired():
                        raise
                    raise ArtifactStallError(
                        f"no token for {stall_sec:g}s after the stream had delivered {tokens} "
                        "of them; the connection was still open") from None
                carried = proto.stream_token_count(chunk)
                if carried:
                    tokens += carried
                    deadline = loop.time() + stall_sec
                yield chunk
        finally:
            await frames.aclose()

    @staticmethod
    async def _frame_sse(lines: AsyncIterator[str]) -> AsyncIterator[dict[str, Any]]:
        """Parse an OpenAI SSE line stream: yield each `data:` chunk's JSON, stop at [DONE].

        Skips comment/event/blank lines; tolerates `data:` with or without a leading space.
        """
        async for line in lines:
            line = line.strip()
            if not line.startswith(SSE_DATA_PREFIX):
                continue
            payload = line[len(SSE_DATA_PREFIX):].strip()
            if payload == "[DONE]":
                break
            yield _json_body(payload)

    async def _post_sse(self, json_body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        url = self.base_url + self.contract.path
        async with self._client.stream(
                "POST", url, json=json_body,
                timeout=httpx.Timeout(self.request_timeout_sec)) as r:
            if r.is_error:
                # A streamed response's body is not read yet, and `raise_for_status`
                # does not read it. The rejection reason is still on the wire and the
                # stream closes when this block exits, so reading it later (from
                # `submit`'s `except httpx.HTTPStatusError` handler) would raise
                # ResponseNotRead and lose it for good. Read it here, on the error path
                # only -- a healthy stream is still consumed incrementally, which is
                # what makes the TTFT/TPOT timings meaningful.
                await r.aread()
            r.raise_for_status()
            async for chunk in self._frame_sse(r.aiter_lines()):
                yield chunk


def make_artifact_client(contract: ArtifactContract, base_url: str,
                         request_timeout_sec: float | None = None,
                         stall_timeout_sec: float | None = None) -> ArtifactClient:
    if isinstance(contract, RequestResponseContract):
        return HTTPArtifactClient(contract, base_url, request_timeout_sec=request_timeout_sec,
                                  stall_timeout_sec=stall_timeout_sec)
    raise TypeError(f"No ArtifactClient for contract {type(contract).__name__}")
