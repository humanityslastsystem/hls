"""What the index holds for its clients, and for how long. The interceptor runs in the sealed
step beside the contestant, so its memory must stay bounded whatever its clients do: the simple
pages it holds at once across every connection are capped (a request waits for room before it
asks the upstream, and is refused with a 503 when none comes), a page is read only up to a
bound, and a client that takes none of its reply for the write timeout is dropped. None of it
counts as an upstream failure, so a client that does not read cannot make the seal read its
index as out."""

from __future__ import annotations

import asyncio
import fcntl
import json
import socket
import struct
import termios
import time

import httpx
import pytest

from hls.egress import proxy as proxy_mod
from hls.egress.proxy import EgressProxy

MIB = 2**20
POLICY = {"domains": {"mode": "allow", "entries": []}, "packages": None}
WHEEL = "demo-1.0-py3-none-any.whl"


def _page(size: int) -> bytes:
    """A simple page of at least *size* bytes, of wheel links as PyPI writes them."""
    lines = [b"<!DOCTYPE html><html><body>\n"]
    total, n = len(lines[0]), 0
    while total < size:
        name = b"demo-1.0.%d-py3-none-any.whl" % n
        line = (b'<a href="https://files.pythonhosted.org/packages/ab/cd/%040x/' % n + name + b"#sha256=" + b"0" * 64
                + b'" data-requires-python="&gt;=3.8">' + name + b"</a><br />\n")
        lines.append(line)
        total += len(line)
        n += 1
    return b"".join(lines) + b"</body></html>\n"


class Upstream:
    """A local upstream: every simple page is *page*; a file is 64 MiB, sent through a small send
    buffer so a client that does not read holds the stream. It counts what it was asked."""

    def __init__(self, page: bytes) -> None:
        self.page = page
        self.accepted = 0
        self.server: asyncio.AbstractServer | None = None

    async def start(self) -> int:
        self.server = await asyncio.start_server(self._serve, "127.0.0.1", 0, backlog=512)
        return self.server.sockets[0].getsockname()[1]

    def close(self) -> None:
        assert self.server is not None
        self.server.close()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.accepted += 1
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            if head.startswith(b"GET /simple/"):
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\n\r\n"
                             % len(self.page) + self.page)
                await writer.drain()
                return
            writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 16384)
            chunk = b"x" * 65536
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\n"
                         b"Content-Length: %d\r\n\r\n" % (len(chunk) * 1024))
            for _ in range(1024):
                writer.write(chunk)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            writer.close()


async def _index(tmp_path, upstream_port: int) -> tuple[EgressProxy, asyncio.AbstractServer, int]:
    proxy = EgressProxy(POLICY, upstream=f"http://127.0.0.1:{upstream_port}", host="127.0.0.1", port=0,
                        status_file=str(tmp_path / "status.json"))
    index = await proxy.start()
    proxy._write_status()
    return proxy, index, index.sockets[0].getsockname()[1]


def _status(tmp_path) -> dict:
    return json.loads((tmp_path / "status.json").read_text())


def _counts(tmp_path) -> dict:
    """The index's upstream-error counts, from its status file."""
    status = _status(tmp_path)
    return {key: status[key] for key in ("upstream_errors", "mid_stream_errors")}


def _get(target: str) -> bytes:
    return f"GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".encode()


def _file(proxy: EgressProxy, upstream_port: int) -> str:
    return f"/files/{proxy._sign_token(f'http://127.0.0.1:{upstream_port}/packages/{WHEEL}')}/{WHEEL}"


async def _ask(port: int, request: bytes, *, small_buffer: bool = False):
    """Send *request* to the index; with *small_buffer* the client's receive buffer is 4 KiB, so
    the kernel takes little of a reply it does not read."""
    sock = socket.socket()
    if small_buffer:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
    sock.setblocking(False)
    await asyncio.get_running_loop().sock_connect(sock, ("127.0.0.1", port))
    reader, writer = await asyncio.open_connection(sock=sock)
    writer.write(request)
    await writer.drain()
    return reader, writer


async def _whole_reply(port: int, request: bytes) -> tuple[int, bytes]:
    """The status and body of the index's reply to a client that reads all of it."""
    reader, writer = await _ask(port, request)
    try:
        raw = await asyncio.wait_for(reader.read(), 60)
    finally:
        writer.close()
    head, _, body = raw.partition(b"\r\n\r\n")
    length = next(int(line.split(b":")[1]) for line in head.split(b"\r\n") if line.lower().startswith(b"content-length:"))
    assert len(body) == length, (len(body), length)
    return int(head.split(b" ", 2)[1]), body


@pytest.fixture
def local_file_host(monkeypatch):
    """A server on the loopback stands in for the file host."""
    monkeypatch.setattr(proxy_mod, "_FILE_HOSTS", frozenset({*proxy_mod._FILE_HOSTS, "127.0.0.1"}))


def test_the_figures_let_every_page_be_read_and_pip_hear_its_refusal():
    # A page request takes its start before it asks, and a page of the largest size must fit
    # in the room, or every such request would be refused.
    assert proxy_mod._PAGE_START_BYTES <= proxy_mod.PAGE_MAX_BYTES <= proxy_mod._PAGES_MAX_BYTES
    # pip waits 15 s for an answer by default; a refusal must come before it gives up.
    assert proxy_mod._ROOM_WAIT_SEC < 15
    # A page at the cap fits in the room both as read and as rewritten, about a third larger.
    assert proxy_mod.PAGE_MAX_BYTES * 7 // 3 <= proxy_mod._PAGES_MAX_BYTES
    assert (proxy_mod._PAGE_START_BYTES, proxy_mod.PAGE_MAX_BYTES, proxy_mod._PAGES_MAX_BYTES,
            proxy_mod._WRITE_TIMEOUT_SEC, proxy_mod._SEND_BUFFER_BYTES) == (16 * MIB, 64 * MIB, 512 * MIB, 60.0,
                                                                            256 * 1024)


class _Peak:
    """The most bytes the index's pages held at any sample, sampled every millisecond."""

    def __init__(self, proxy: EgressProxy) -> None:
        self.proxy = proxy
        self.most = 0
        self.task = asyncio.create_task(self._sample())

    async def _sample(self) -> None:
        while True:
            self.most = max(self.most, self.proxy._pages.held)
            await asyncio.sleep(0.001)

    def stop(self) -> int:
        self.task.cancel()
        return self.most


async def test_a_flood_of_clients_that_never_read_holds_pages_only_up_to_the_cap_and_counts_nothing(
        tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_PAGES_MAX_BYTES", 32 * MIB)
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", 16 * MIB)
    monkeypatch.setattr(proxy_mod, "_PAGE_START_BYTES", 10 * MIB)
    monkeypatch.setattr(proxy_mod, "_ROOM_WAIT_SEC", 1.0)
    # Each rewritten page, 8.6 MiB, is larger than the most the kernel buffers for a connection
    # (4 MiB by default), so a client that does not read leaves the rest of it with the index.
    upstream = Upstream(_page(6 * MIB))
    proxy, index, port = await _index(tmp_path, await upstream.start())
    served = len(proxy._rewrite_simple(upstream.page, f"http://127.0.0.1:{upstream.server.sockets[0].getsockname()[1]}"
                                                       "/simple/demo/"))
    peak = _Peak(proxy)
    clients = []
    try:
        for _ in range(40):
            clients.append(await _ask(port, _get("/simple/demo/"), small_buffer=True))
        await asyncio.sleep(3)
        heads = [await asyncio.wait_for(reader.readexactly(12), 10) for reader, _ in clients]
        held = heads.count(b"HTTP/1.1 200")
        # Pages held for clients that do not read stay within the cap; the requests past it are
        # refused, as an upstream that is out of room would refuse them.
        assert heads.count(b"HTTP/1.1 503") + held == len(clients), set(heads)
        assert 1 <= held <= 32 * MIB // served, held
        assert peak.stop() <= 32 * MIB
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
    finally:
        peak.stop()
        for _, writer in clients:
            writer.close()
        await asyncio.sleep(0.2)
        index.close()
        upstream.close()
        await proxy.aclose()


@pytest.fixture
def accepted(monkeypatch) -> list[asyncio.StreamWriter]:
    """The index's writer of every connection it accepts, in order."""
    writers: list[asyncio.StreamWriter] = []
    serve = EgressProxy._on_client

    async def recording(self, reader, writer):
        writers.append(writer)
        await serve(self, reader, writer)

    monkeypatch.setattr(EgressProxy, "_on_client", recording)
    return writers


@pytest.mark.parametrize("path", ["simple", "files"])
async def test_a_client_that_takes_none_of_its_reply_is_dropped_after_the_write_timeout_and_counts_nothing(
        tmp_path, monkeypatch, local_file_host, accepted, path):
    monkeypatch.setattr(proxy_mod, "_WRITE_TIMEOUT_SEC", 1.0)
    upstream = Upstream(_page(8 * MIB))
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path, upstream_port)
    target = "/simple/demo/" if path == "simple" else _file(proxy, upstream_port)
    try:
        reader, writer = await _ask(port, _get(target), small_buffer=True)
        started = time.monotonic()
        for _ in range(500):
            if proxy._conn_count == 0:
                break
            await asyncio.sleep(0.02)
        dropped = time.monotonic() - started
        assert proxy._conn_count == 0, "the connection of a client that reads nothing is still open"
        assert dropped >= 1.0
        # The connection is gone with nothing of its reply left waiting to be sent, though the
        # client still has not read.
        [ours] = accepted
        assert ours.transport.is_closing() and ours.transport.get_write_buffer_size() == 0
        # The client finds its reply cut short, as when an upstream drops a connection.
        got = b""
        with pytest.raises((ConnectionResetError, asyncio.IncompleteReadError)):
            while True:
                chunk = await asyncio.wait_for(reader.read(65536), 10)
                if not chunk:
                    raise asyncio.IncompleteReadError(got, None)
                got += chunk
        assert got.startswith(b"HTTP/1.1 200")
        assert proxy._pages.held == 0
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
        writer.close()
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


async def test_a_client_that_reads_slowly_but_steadily_is_never_dropped(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_WRITE_TIMEOUT_SEC", 1.0)
    # A page larger than the kernel buffers for a connection, read at 64 KiB every 25 ms, about
    # 2.5 MiB/s: the client takes some of it in far less than the timeout, all of it in several.
    upstream = Upstream(_page(8 * MIB))
    proxy, index, port = await _index(tmp_path, await upstream.start())
    loop = asyncio.get_running_loop()
    sock = socket.socket()
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 65536)
        sock.setblocking(False)
        await loop.sock_connect(sock, ("127.0.0.1", port))
        await loop.sock_sendall(sock, _get("/simple/demo/"))
        got = bytearray()
        started = time.monotonic()
        while chunk := await asyncio.wait_for(loop.sock_recv(sock, 65536), 10):
            got += chunk
            await asyncio.sleep(0.025)
        assert time.monotonic() - started > 3.0
        head, _, body = bytes(got).partition(b"\r\n\r\n")
        assert head.startswith(b"HTTP/1.1 200") and body.endswith(b"</body></html>\n")
        assert f"content-length: {len(body)}".encode() in head.lower()
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
    finally:
        sock.close()
        index.close()
        upstream.close()
        await proxy.aclose()


@pytest.mark.parametrize("room", ["default", "four pages at once"])
async def test_concurrent_clients_that_read_get_whole_pages(tmp_path, monkeypatch, room):
    if room != "default":
        # The pages asked for at once far outgrow the room, so most requests wait for theirs;
        # none may be starved of it by pages that others have half read.
        monkeypatch.setattr(proxy_mod, "_PAGES_MAX_BYTES", 8 * MIB)
        monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", 2 * MIB)
        monkeypatch.setattr(proxy_mod, "_PAGE_START_BYTES", 2 * MIB)
    upstream = Upstream(_page(MIB))
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path, upstream_port)
    expected = proxy._rewrite_simple(upstream.page, f"http://127.0.0.1:{upstream_port}/simple/demo/")
    try:
        replies = await asyncio.gather(*(_whole_reply(port, _get("/simple/demo/")) for _ in range(64)))
        assert all(reply == (200, expected) for reply in replies)
        assert proxy._pages.held == 0
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


async def test_a_page_request_waits_for_room_before_it_asks_the_upstream(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_PAGES_MAX_BYTES", 4 * MIB)
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", 2 * MIB)
    monkeypatch.setattr(proxy_mod, "_PAGE_START_BYTES", 2 * MIB)
    monkeypatch.setattr(proxy_mod, "_ROOM_WAIT_SEC", 30.0)
    upstream = Upstream(_page(MIB))
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path, upstream_port)
    other = proxy._pages.hold()
    assert other.resize(3 * MIB)
    try:
        reply = asyncio.create_task(_whole_reply(port, _get("/simple/demo/")))
        await asyncio.sleep(0.5)
        # Less than a page of the largest size is free, so the request waits and asks nothing.
        assert not reply.done() and upstream.accepted == 0
        other.resize(0)
        status, body = await asyncio.wait_for(reply, 30)
        assert status == 200 and body.endswith(b"</body></html>\n")
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
    finally:
        other.resize(0)
        index.close()
        upstream.close()
        await proxy.aclose()


async def test_a_page_request_that_finds_no_room_in_time_is_refused_without_asking_the_upstream(tmp_path,
                                                                                               monkeypatch):
    monkeypatch.setattr(proxy_mod, "_PAGES_MAX_BYTES", 4 * MIB)
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", 2 * MIB)
    monkeypatch.setattr(proxy_mod, "_PAGE_START_BYTES", 2 * MIB)
    monkeypatch.setattr(proxy_mod, "_ROOM_WAIT_SEC", 0.5)
    upstream = Upstream(_page(MIB))
    proxy, index, port = await _index(tmp_path, await upstream.start())
    other = proxy._pages.hold()
    assert other.resize(3 * MIB)
    try:
        started = time.monotonic()
        status, body = await _whole_reply(port, _get("/simple/demo/"))
        assert status == 503 and body == b"the index is busy"
        assert time.monotonic() - started >= 0.5
        assert upstream.accepted == 0
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
    finally:
        other.resize(0)
        index.close()
        upstream.close()
        await proxy.aclose()


@pytest.mark.parametrize("status, body", [
    (404, b"no such project"),
    (200, b'<a href="https://files.pythonhosted.org/packages/ab/demo-1.0-py3-none-any.whl">demo-1.0</a>'),
], ids=["error", "page"])
async def test_a_page_request_holds_its_reply_alone_once_it_has_one(status, body):
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0)

    async def fetch(url, **kwargs):
        return status, body, "text/html"

    proxy._fetch = fetch
    with proxy._pages.hold() as hold:
        got, reply = await proxy.handle_simple("demo", hold)
        # The room taken before asking the upstream goes back once the reply is known.
        assert got == status and hold.bytes == proxy._pages.held == len(reply)
    assert proxy._pages.held == 0


async def test_a_page_whose_rewrite_finds_no_room_is_refused_and_counts_nothing(tmp_path, monkeypatch):
    # The page itself fits in the room, but the rewritten page, a third larger, does not.
    monkeypatch.setattr(proxy_mod, "_PAGES_MAX_BYTES", MIB)
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", MIB)
    monkeypatch.setattr(proxy_mod, "_PAGE_START_BYTES", MIB)
    upstream = Upstream(_page(900 * 1024))
    proxy, index, port = await _index(tmp_path, await upstream.start())
    try:
        status, body = await _whole_reply(port, _get("/simple/demo/"))
        assert status == 503 and body == b"the index is busy"
        assert proxy._pages.held == 0
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


class _Chunks(httpx.AsyncByteStream):
    """A response body of *count* chunks of 64 KiB; it records how many were taken."""

    def __init__(self, count: int) -> None:
        self.count = count
        self.taken = 0

    async def __aiter__(self):
        for _ in range(self.count):
            self.taken += 1
            yield b"x" * 65536


async def test_a_page_read_that_finds_no_room_stops_reading_is_refused_and_counts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "_PAGES_MAX_BYTES", MIB)
    body = _Chunks(16)
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0, status_file=str(tmp_path / "status.json"))
    proxy._write_status()
    other = proxy._pages.hold()
    assert other.resize(MIB // 2)
    hold = proxy._pages.hold()
    status, _, _ = await proxy._fetch("https://pypi.org/simple/demo/", hold=hold,
                                      transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=body)))
    # Half the room was free: eight chunks fit, the ninth did not, and nothing more was read.
    assert status == 503 and body.taken == 9
    assert hold.bytes == MIB // 2 and proxy._pages.held == MIB
    hold.resize(0)
    other.resize(0)
    assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}


async def test_a_page_past_the_bound_is_refused_unread_past_it_and_counts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", MIB)
    body = _Chunks(64)
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0, status_file=str(tmp_path / "status.json"))
    proxy._write_status()
    with proxy._pages.hold() as hold:
        status, reason, _ = await proxy._fetch("https://pypi.org/simple/demo/", hold=hold,
                                               transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=body)))
        # Sixteen chunks are the bound; the seventeenth is past it.
        assert status == 502 and reason == b"the page is larger than the index serves" and body.taken == 17
        assert hold.bytes <= MIB
    assert proxy._pages.held == 0
    assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}


async def test_a_page_of_exactly_the_bound_is_served(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", MIB)
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0)
    with proxy._pages.hold() as hold:
        status, page, _ = await proxy._fetch("https://pypi.org/simple/demo/", hold=hold,
                                             transport=httpx.MockTransport(
                                                 lambda r: httpx.Response(200, stream=_Chunks(16))))
        assert status == 200 and len(page) == MIB and hold.bytes == MIB


async def test_each_page_refused_for_its_size_is_recorded_once_with_what_was_read_and_the_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", MIB)
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0, status_file=str(tmp_path / "status.json"))
    proxy._write_status()
    assert _status(tmp_path)["oversized_pages"] == []
    names = [f"big{n}" for n in range(proxy_mod._OVERSIZED_KEPT + 2)]
    # pip asks again after a 502, so the first page is refused twice.
    for name in [names[0], *names]:
        with proxy._pages.hold() as hold:
            status, _, _ = await proxy._fetch(f"https://pypi.org/simple/{name}/", hold=hold,
                                              transport=httpx.MockTransport(
                                                  lambda r: httpx.Response(200, stream=_Chunks(64))))
            assert status == 502
    # The status file keeps the first pages refused, each once, as the stage reads them.
    assert _status(tmp_path)["oversized_pages"] == [
        {"page": f"https://pypi.org/simple/{name}/", "read": MIB + 65536, "cap": MIB}
        for name in names[:proxy_mod._OVERSIZED_KEPT]]
    assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}


async def test_a_refused_page_is_recorded_by_the_first_characters_of_its_url(tmp_path, monkeypatch):
    # The name in the URL is the client's, as long as its request head allows.
    monkeypatch.setattr(proxy_mod, "PAGE_MAX_BYTES", MIB)
    proxy = EgressProxy(POLICY, host="127.0.0.1", port=0, status_file=str(tmp_path / "status.json"))
    url = "https://pypi.org/simple/" + "x" * 4000 + "/"
    with proxy._pages.hold() as hold:
        await proxy._fetch(url, hold=hold,
                           transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=_Chunks(64))))
    assert [page["page"] for page in _status(tmp_path)["oversized_pages"]] == [url[:proxy_mod._PAGE_URL_CHARS]]


async def test_a_page_near_the_cap_is_served_whole_to_a_client_that_reads(tmp_path):
    # 60 MiB, about eight times pydantic-core's page, is read and rewritten within the room.
    upstream = Upstream(_page(60 * MIB))
    upstream_port = await upstream.start()
    proxy, index, port = await _index(tmp_path, upstream_port)
    try:
        status, body = await _whole_reply(port, _get("/simple/demo/"))
        assert status == 200 and body.endswith(b"</body></html>\n") and body.count(b'href="/files/') > 200_000
        assert proxy._pages.held == 0
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


async def test_the_kernel_holds_little_of_a_reply_a_client_does_not_read(tmp_path, accepted):
    upstream = Upstream(_page(8 * MIB))
    proxy, index, port = await _index(tmp_path, await upstream.start())
    try:
        _, writer = await _ask(port, _get("/simple/demo/"), small_buffer=True)
        await asyncio.sleep(1.5)
        [ours] = accepted
        sock = ours.get_extra_info("socket")
        # The send buffer is set on every index connection, which stops the kernel growing it as
        # the reply waits; Linux grants twice what is set, at most twice net.core.wmem_max.
        granted = sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
        assert granted <= 2 * proxy_mod._SEND_BUFFER_BYTES
        # What the kernel still has to send of the reply stays within that buffer.
        queued = struct.unpack("i", fcntl.ioctl(sock.fileno(), termios.TIOCOUTQ, b"\0" * 4))[0]
        assert 0 < queued <= granted
        writer.close()
    finally:
        index.close()
        upstream.close()
        await proxy.aclose()


class _FailsToClose(httpx.AsyncByteStream):
    """A file longer than a client that does not read takes, whose close fails as an upstream
    gone silent would."""

    async def __aiter__(self):
        for _ in range(1024):
            yield b"x" * 65536

    async def aclose(self) -> None:
        raise httpx.ReadTimeout("silent while closing")


async def test_a_dropped_client_counts_nothing_even_when_closing_the_upstream_fails(tmp_path, monkeypatch,
                                                                                     local_file_host):
    monkeypatch.setattr(proxy_mod, "_WRITE_TIMEOUT_SEC", 1.0)
    proxy, index, port = await _index(tmp_path, 9)
    proxy._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=_FailsToClose())))
    try:
        _, writer = await _ask(port, _get(_file(proxy, 9)), small_buffer=True)
        for _ in range(500):
            if proxy._conn_count == 0:
                break
            await asyncio.sleep(0.02)
        assert proxy._conn_count == 0
        # The failed close came of the client's stall, so it is no failure of the upstream.
        assert _counts(tmp_path) == {"upstream_errors": 0, "mid_stream_errors": 0}
        writer.close()
    finally:
        index.close()
        await proxy.aclose()
