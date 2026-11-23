"""The index's upstream-error count, which a stage reads as evidence that the index, not the
contestant, failed a build. It holds only failures of the upstream itself: never one a
client of the index can cause by how it asks, such as a full connection pool or a stream it
held until the upstream gave up on it. A failure while a file streams to a client that kept
reading counts."""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import socket
import struct
import time
from collections.abc import Iterator

import httpx
import pytest

from hls.egress import proxy as proxy_mod
from hls.egress.proxy import EgressProxy

POLICY = {"domains": {"mode": "allow", "entries": []}, "packages": None}
WHEEL = "demo-1.0-py3-none-any.whl"


@pytest.fixture
def local_file_host(monkeypatch):
    """A server on the loopback stands in for the file host."""
    monkeypatch.setattr(proxy_mod, "_FILE_HOSTS", frozenset({*proxy_mod._FILE_HOSTS, "127.0.0.1"}))


class Upstream:
    """A local upstream. With *answer* "body" it sends 200 and a body longer than every
    buffer between it and a client that does not read, through a small send buffer; with a
    status ("503", "429", "404") that status; with "reset" or "short" it sends 200 and part of
    its body, then resets the connection or closes it before the length it declared. It counts
    the connections it accepted."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.accepted = 0
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        return self.server.sockets[0].getsockname()[1]

    def close(self) -> None:
        assert self.server is not None
        self.server.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.accepted += 1
        writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
        try:
            await reader.readuntil(b"\r\n\r\n")
            if self.answer.isdecimal():
                writer.write(b"HTTP/1.1 %s Status\r\nContent-Length: 4\r\nConnection: close\r\n\r\nbusy"
                             % self.answer.encode())
                await writer.drain()
                return
            chunk = b"x" * 65536
            if self.answer in ("reset", "short"):
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                             b"Content-Length: %d\r\n\r\n" % (len(chunk) * 64) + chunk * 4)
                await writer.drain()
                await asyncio.sleep(0.5)
                if self.answer == "reset":
                    # A close with nothing lingering sends a reset.
                    writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                                                               struct.pack("ii", 1, 0))
                return
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                         b"Content-Length: %d\r\n\r\n" % (len(chunk) * 1024))
            for _ in range(1024):
                writer.write(chunk)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


@contextlib.contextmanager
def _refused_port() -> Iterator[int]:
    """A port that refuses connections for as long as it is held: a socket is bound there and
    never listens. A port bound and released at once could be the next one the kernel hands
    out, the index's own among them, which would make the index its own upstream."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        yield s.getsockname()[1]


async def _index(tmp_path, **kwargs) -> tuple[EgressProxy, asyncio.AbstractServer, int]:
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0, status_file=str(tmp_path / "status.json"), **kwargs)
    index = await proxy.start()
    proxy._write_status()
    return proxy, index, index.sockets[0].getsockname()[1]


def _count(tmp_path) -> int:
    return json.loads((tmp_path / "status.json").read_text())["upstream_errors"]


def _mid_stream(tmp_path) -> int:
    return json.loads((tmp_path / "status.json").read_text())["mid_stream_errors"]


def _get(target: str) -> bytes:
    return f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode()


def _file(proxy: EgressProxy, upstream_port: int) -> str:
    return f"/files/{proxy._sign_token(f'http://127.0.0.1:{upstream_port}/packages/{WHEEL}')}/{WHEEL}"


async def _ask(port: int, request: bytes, *, small_buffer: bool = False):
    """Send *request* to the index; with *small_buffer* the client's receive buffer is 4 KiB,
    so a stream it does not read holds little memory."""
    sock = socket.socket()
    if small_buffer:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.setblocking(False)
    await asyncio.get_running_loop().sock_connect(sock, ("127.0.0.1", port))
    reader, writer = await asyncio.open_connection(sock=sock)
    writer.write(request)
    await writer.drain()
    return reader, writer


async def _answer(port: int, request: bytes) -> bytes:
    reader, writer = await _ask(port, request)
    try:
        return await asyncio.wait_for(reader.read(), 60)
    finally:
        writer.close()


async def test_a_client_holding_every_slot_but_one_still_gets_its_file_and_counts_nothing(tmp_path, monkeypatch,
                                                                                        local_file_host):
    # A connection cap above httpx's default pool of 100 connections, as the real cap of 256 is.
    monkeypatch.setattr(proxy_mod, "_MAX_CONNECTIONS", 101)
    upstream = Upstream("body")
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path)
    held = []
    try:
        request = _get(_file(proxy, upstream_port))
        for _ in range(proxy_mod._MAX_CONNECTIONS - 1):
            held.append(await _ask(port, request, small_buffer=True))
        for _ in range(500):
            if upstream.accepted == len(held):
                break
            await asyncio.sleep(0.02)
        assert upstream.accepted == len(held)
        started = time.monotonic()
        reader, writer = await _ask(port, request)
        held.append((reader, writer))
        head = await asyncio.wait_for(reader.readuntil(b"\r\n"), 60)
        assert head.startswith(b"HTTP/1.1 200"), head
        assert time.monotonic() - started < 10
        assert _count(tmp_path) == 0
    finally:
        for _, w in held:
            w.close()
        await asyncio.sleep(0.2)
        index.close()
        upstream.close()
        await proxy.aclose()


@pytest.mark.parametrize("path", ["simple", "files"])
@pytest.mark.parametrize("failure, counted", [("503", 1), ("429", 1), ("404", 0), ("refused", 1)])
async def test_a_real_upstream_error_or_refusal_counts_and_is_written(tmp_path, local_file_host, path, failure,
                                                                      counted):
    """A server error or a rate limit of the upstream is its failure, which pip retries no
    further; a missing page or file is an answer."""
    upstream = Upstream(failure)
    with contextlib.ExitStack() as held:
        upstream_port = await upstream.start() if failure != "refused" else held.enter_context(_refused_port())
        proxy, index, port = await _index(tmp_path, upstream=f"http://127.0.0.1:{upstream_port}")
        try:
            assert port != upstream_port
            target = "/simple/demo/" if path == "simple" else _file(proxy, upstream_port)
            answer = await _answer(port, _get(target))
            assert answer.startswith(b"HTTP/1.1 502" if failure == "refused" else f"HTTP/1.1 {failure}".encode()), \
                answer[:80]
            assert _count(tmp_path) == counted and _mid_stream(tmp_path) == 0
        finally:
            index.close()
            if failure != "refused":
                upstream.close()
            await proxy.aclose()


@pytest.mark.parametrize("failure", ["reset", "short"], ids=["reset", "closed-before-its-length"])
async def test_an_upstream_that_fails_a_file_a_prompt_client_is_reading_counts_mid_stream(tmp_path, local_file_host,
                                                                                         failure):
    """A real upstream sends part of a file, then resets the connection or closes it before the
    length it declared, while the client reads all it is sent: the upstream failed, so a pip
    step that fails on it is the harness's."""
    upstream = Upstream(failure)
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path)
    try:
        answer = await _answer(port, _get(_file(proxy, upstream_port)))
        assert answer.startswith(b"HTTP/1.1 200") and len(answer) < 64 * 65536
        assert _count(tmp_path) == 1 and _mid_stream(tmp_path) == 1
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


class _Breaks(httpx.AsyncByteStream):
    """A response body of *count* chunks of 64 KiB, then *error*."""

    def __init__(self, count: int, error: Exception) -> None:
        self.count = count
        self.error = error

    async def __aiter__(self):
        for _ in range(self.count):
            yield b"x" * 65536
        raise self.error


@pytest.mark.parametrize("stall, counted", [(0.0, 1), (3.0, 0)], ids=["reading", "stalled-a-while"])
async def test_a_failure_after_the_client_stalled_its_read_is_not_counted(tmp_path, monkeypatch, local_file_host,
                                                                           stall, counted):
    """A client that leaves a reply untaken longer than ``_PROMPT_SEC`` holds the upstream for
    as long, which can make the upstream give up on the stream; a failure after that is not
    certainly the upstream's. The same failure under a client that kept reading is."""
    monkeypatch.setattr(proxy_mod, "_PROMPT_SEC", 1.0)
    proxy, index, port = await _index(tmp_path)
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, stream=_Breaks(64, httpx.ReadError("connection reset")))))
    try:
        reader, writer = await _ask(port, _get(_file(proxy, 9)), small_buffer=True)
        await asyncio.sleep(stall)
        try:
            body = await asyncio.wait_for(reader.read(), 60)
        finally:
            writer.close()
        assert body.startswith(b"HTTP/1.1 200")
        assert _count(tmp_path) == counted and _mid_stream(tmp_path) == counted
    finally:
        index.close()
        await proxy.aclose()


async def _read_at(reader: asyncio.StreamReader, rate: float | None) -> bytes:
    """All of a reply, read at *rate* bytes a second at most, or as fast as it comes for None."""
    got = bytearray()
    while data := await asyncio.wait_for(reader.read(16384), 60):
        got += data
        if rate is not None:
            await asyncio.sleep(len(data) / rate)
    return bytes(got)


@pytest.mark.parametrize("rate, counted", [(None, 1), (256 * 1024, 0)], ids=["reading-promptly", "reading-slowly"])
async def test_a_failure_while_the_client_reads_below_the_least_rate_is_not_counted(tmp_path, monkeypatch,
                                                                                     local_file_host, rate, counted):
    """A client that takes some of a file within every few seconds, but slowly, holds the
    upstream for the whole file all the same: a failure counts only when the client took the
    file at ``MIN_READ_RATE`` or faster, over the time since the reply started past the first
    ``_PROMPT_SEC``. The same reset under a client that reads all it is sent counts. Here the
    slow client takes each chunk well within the prompt window, for about 3.5 s in all."""
    monkeypatch.setattr(proxy_mod, "MIN_READ_RATE", 4 * 2**20)
    monkeypatch.setattr(proxy_mod, "_PROMPT_SEC", 1.0)
    proxy, index, port = await _index(tmp_path)
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, stream=_Breaks(24, httpx.ReadError("connection reset")))))
    try:
        reader, writer = await _ask(port, _get(_file(proxy, 9)), small_buffer=True)
        try:
            body = await _read_at(reader, rate)
        finally:
            writer.close()
        assert body.startswith(b"HTTP/1.1 200")
        assert _count(tmp_path) == counted and _mid_stream(tmp_path) == counted
    finally:
        index.close()
        await proxy.aclose()


class _SlowBreaks(httpx.AsyncByteStream):
    """A response body of *count* chunks of 64 KiB, *gap* seconds apart, then *error*: an
    upstream slower than ``MIN_READ_RATE`` on its own account."""

    def __init__(self, count: int, gap: float, error: Exception) -> None:
        self.count = count
        self.gap = gap
        self.error = error

    async def __aiter__(self):
        for _ in range(self.count):
            await asyncio.sleep(self.gap)
            yield b"x" * 65536
        raise self.error


async def test_a_failure_of_a_slow_upstream_under_a_prompt_client_counts(tmp_path, monkeypatch, local_file_host):
    """An upstream that sends a file slower than ``MIN_READ_RATE`` and then resets it failed on
    its own account: the client took all it was sent at once. Its rate is judged over the time
    the interceptor waited on the client, never over the time the upstream took."""
    monkeypatch.setattr(proxy_mod, "MIN_READ_RATE", 4 * 2**20)
    monkeypatch.setattr(proxy_mod, "_PROMPT_SEC", 1.0)
    proxy, index, port = await _index(tmp_path)
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, stream=_SlowBreaks(8, 0.3, httpx.ReadError("connection reset")))))
    try:
        started = time.monotonic()
        reader, writer = await _ask(port, _get(_file(proxy, 9)))
        try:
            body = await _read_at(reader, None)
        finally:
            writer.close()
        # Half a MiB in over two seconds: far under the rate the wall clock would ask.
        assert body.startswith(b"HTTP/1.1 200") and time.monotonic() - started > 2.0
        assert _count(tmp_path) == 1 and _mid_stream(tmp_path) == 1
    finally:
        index.close()
        await proxy.aclose()


def test_a_client_keeps_up_by_its_rate_over_the_time_the_interceptor_waited_on_it():
    """A client the interceptor waited on for ``_PROMPT_SEC`` in all, or less, held the upstream
    no longer than one prompt drain may, so it kept up whatever it took; one waited on longer
    must have taken ``MIN_READ_RATE`` for every second it was waited on."""
    rate, window = proxy_mod.MIN_READ_RATE, proxy_mod._PROMPT_SEC
    assert rate == 256 * 1024 and window == 5.0
    assert proxy_mod._kept_up(0, window)
    assert not proxy_mod._kept_up(0, window + 0.1)
    assert proxy_mod._kept_up(rate * 10, 10.0)
    assert not proxy_mod._kept_up(rate * 10 - 1, 10.0)


def _status_of(tmp_path) -> dict:
    return json.loads((tmp_path / "status.json").read_text())


@pytest.mark.parametrize("path", ["simple", "files"])
async def test_a_flood_past_the_fetch_cap_is_answered_locally_and_counts_nothing(tmp_path, monkeypatch,
                                                                                 local_file_host, path):
    """A build hook can ask the index for page after page to draw a rate limit from the
    upstream. Each step's interceptor fetches from the upstream at most
    ``UPSTREAM_FETCHES_MAX`` times; past that it answers every request itself with a refusal
    that counts as no upstream failure, and records how many it refused."""
    monkeypatch.setattr(proxy_mod, "UPSTREAM_FETCHES_MAX", 5)
    upstream = Upstream("200")
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path, upstream=f"http://127.0.0.1:{upstream_port}")
    try:
        targets = [f"/simple/demo{i}/" if path == "simple" else _file(proxy, upstream_port) for i in range(8)]
        answers = [await _answer(port, _get(target)) for target in targets]
        assert [a.split(b" ", 2)[1] for a in answers] == [b"200"] * 5 + [b"403"] * 3
        assert upstream.accepted == 5
        status = _status_of(tmp_path)
        assert (status["upstream_errors"], status["fetches_refused"]) == (0, 3)
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


@pytest.mark.parametrize("answer, counts", [("429", [1, 2, 0, 0]), ("503", [1, 2, 3, 3])])
async def test_a_429_counts_only_while_the_step_is_under_its_fetch_cap(tmp_path, monkeypatch, answer, counts):
    """A rate limit is the upstream's failure for a step that asked as an install does, far
    below the cap; a step that made all its allowed fetches flooded the index, so none of its
    429s count. A server error counts whatever the step asked."""
    monkeypatch.setattr(proxy_mod, "UPSTREAM_FETCHES_MAX", 3)
    upstream = Upstream(answer)
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path, upstream=f"http://127.0.0.1:{upstream_port}")
    try:
        seen = []
        for i in range(4):
            await _answer(port, _get(f"/simple/demo{i}/"))
            seen.append(_count(tmp_path))
        assert seen == counts and upstream.accepted == 3 and _status_of(tmp_path)["fetches_refused"] == 1
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


async def test_the_interceptor_writes_a_heartbeat_every_few_seconds(tmp_path, monkeypatch):
    """The status file says when the interceptor last ran, so a stage can tell one that stopped
    answering from one that counted nothing."""
    monkeypatch.setattr(proxy_mod, "_HEARTBEAT_SEC", 0.2)
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0, status_file=str(tmp_path / "status.json"))
    serving = asyncio.create_task(proxy.serve_forever())
    try:
        await asyncio.sleep(0.1)
        first = _status_of(tmp_path)["heartbeat"]
        assert abs(first - time.time()) < 5
        await asyncio.sleep(0.6)
        assert _status_of(tmp_path)["heartbeat"] > first
    finally:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving


def _connect_error_from(code: int) -> httpx.ConnectError:
    """A connect failure as httpx raises it, with the operating system's error as its cause."""
    try:
        try:
            raise OSError(code, "connect failed")
        except OSError as cause:
            raise httpx.ConnectError("connect failed") from cause
    except httpx.ConnectError as e:
        return e


class _BreaksMidBody(httpx.AsyncByteStream):
    """A response body that sends one chunk, then fails with *error*."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    async def __aiter__(self):
        yield b"x" * 1024
        raise self.error


BEFORE_THE_HEAD = [
    ("pool full", httpx.PoolTimeout("no connection free"), 0),
    ("out of descriptors", _connect_error_from(errno.EMFILE), 0),
    ("refused", _connect_error_from(errno.ECONNREFUSED), 1),
    ("connect timeout", httpx.ConnectTimeout("connect timed out"), 1),
    ("site proxy refused the tunnel", httpx.ProxyError("403 from the proxy"), 1),
    ("read timeout", httpx.ReadTimeout("upstream silent"), 1),
    ("reset before the head", httpx.ReadError("connection reset"), 1),
    ("malformed encoding", httpx.DecodingError("not gzip"), 1),
]


@pytest.mark.parametrize("error, counted", [(e, c) for _, e, c in BEFORE_THE_HEAD],
                         ids=[name for name, _, _ in BEFORE_THE_HEAD])
async def test_a_page_fetch_counts_only_the_upstreams_failures(tmp_path, error, counted):
    def fail(request: httpx.Request) -> httpx.Response:
        raise error

    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0, status_file=str(tmp_path / "status.json"))
    proxy._write_status()
    status, _, _ = await proxy._fetch("https://pypi.org/simple/demo/", hold=proxy._pages.hold(),
                                    transport=httpx.MockTransport(fail))
    # A page is read whole before the client gets any of it, so no failure of it is mid-stream.
    assert status == 502 and _count(tmp_path) == counted and _mid_stream(tmp_path) == 0


MID_BODY = [
    ("upstream closed before its end while the client read", httpx.RemoteProtocolError("peer closed"), 1),
    ("reset while the client read", httpx.ReadError("connection reset"), 1),
    ("upstream silent while the proxy waited on it", httpx.ReadTimeout("upstream silent"), 1),
    ("malformed encoding mid-body", httpx.DecodingError("not gzip"), 1),
    ("pool full", httpx.PoolTimeout("no connection free"), 0),
]


@pytest.mark.parametrize("error, counted", [(e, c) for _, e, c in BEFORE_THE_HEAD + MID_BODY],
                         ids=[f"head-{name}" for name, _, _ in BEFORE_THE_HEAD] + [n for n, _, _ in MID_BODY])
async def test_a_file_stream_counts_only_the_upstreams_failures(tmp_path, local_file_host, error, counted):
    mid_body = any(error is e for _, e, _ in MID_BODY)

    def answer(request: httpx.Request) -> httpx.Response:
        if mid_body:
            return httpx.Response(200, stream=_BreaksMidBody(error))
        raise error

    proxy, index, port = await _index(tmp_path)
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(answer))
    try:
        reply = await _answer(port, _get(_file(proxy, 9)))
        assert reply.startswith(b"HTTP/1.1 200" if mid_body else b"HTTP/1.1 502"), reply[:80]
        # The client read all it was sent, so a failure once the file streams is the upstream's too.
        assert _count(tmp_path) == counted and _mid_stream(tmp_path) == (counted if mid_body else 0)
    finally:
        index.close()
        await proxy.aclose()
