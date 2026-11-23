"""The interceptor through a site's upstream proxy: the transparent relay tunnels each vetted
address with CONNECT, the index fetches through it, and nothing of the environment's proxies
is used once one is named."""

from __future__ import annotations

import asyncio

import pytest

from hls.egress import proxy as proxy_mod
from hls.egress.policy import AccessList, NetworkPolicy
from hls.egress.proxy import EgressProxy, check_upstream_proxy
from hls.settings import SettingsError, load_settings


class FakeUpstreamProxy:
    """An HTTP proxy on the loopback that records each request head. A CONNECT is answered
    with *connect_status* and, on 200, tunnelled to the address it names, or never answered
    when it is None; any other request is answered 200 with a body naming its request line.
    After a refused CONNECT the connection stays open for the next request, as an HTTP
    proxy's does, so a client that went on talking would be talking to the proxy itself."""

    def __init__(self, *, connect_status: int | None = 200) -> None:
        self.connect_status = connect_status
        self.heads: list[bytes] = []
        self.server: asyncio.AbstractServer | None = None
        self.writers: list[asyncio.StreamWriter] = []

    async def start(self) -> str:
        self.server = await asyncio.start_server(self._on_client, "127.0.0.1", 0)
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def stop(self) -> None:
        """Stop listening and hang up on every client, so no connection outlives the test."""
        assert self.server is not None
        self.server.close()
        for writer in self.writers:
            writer.close()
        await self.server.wait_closed()

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.writers.append(writer)
        try:
            while True:
                head = await reader.readuntil(b"\r\n\r\n")
                self.heads.append(head)
                line = head.split(b"\r\n", 1)[0]
                method, target = line.split()[:2]
                if method != b"CONNECT":
                    body = b"via proxy: " + line
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nContent-Type: text/plain\r\n\r\n%s"
                                 % (len(body), body))
                    await writer.drain()
                    continue
                if self.connect_status is None:
                    await reader.read()  # no reply; ends when the client hangs up
                    return
                if self.connect_status == 200:
                    break
                writer.write(b"HTTP/1.1 %d Refused\r\nContent-Length: 0\r\n\r\n" % self.connect_status)
                await writer.drain()
            host, _, port = target.decode().rpartition(":")
            up_reader, up_writer = await asyncio.open_connection(host, int(port))
            writer.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
            await writer.drain()

            async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
                while data := await src.read(65536):
                    dst.write(data)
                    await dst.drain()
                dst.close()

            await asyncio.gather(pipe(reader, up_writer), pipe(up_reader, writer))
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


async def _echo_server() -> tuple[asyncio.AbstractServer, int]:
    async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        writer.close()

    server = await asyncio.start_server(echo, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


async def _through_transparent(proxy: EgressProxy, payload: bytes, monkeypatch) -> bytes:
    """Send *payload* to the transparent listener, as a redirected connection would arrive."""

    async def loopback(_host, _port):
        return "127.0.0.1"

    # The relay's address vetting refuses the loopback; this exercises the dial, not the vetting.
    monkeypatch.setattr(proxy_mod, "_resolve_global_ipv4", loopback)
    server = await proxy.start_transparent()
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.sockets[0].getsockname()[1])
        writer.write(payload)
        await writer.drain()
        try:
            return await asyncio.wait_for(reader.readexactly(len(payload)), 5)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            return b""
        finally:
            writer.close()
    finally:
        server.close()
        await server.wait_closed()


def _transparent(origin_port: int, upstream_proxy: str | None) -> EgressProxy:
    policy = NetworkPolicy(domains=AccessList(mode="allow", entries=["localhost"]))
    return EgressProxy(policy.model_dump(), host="127.0.0.1", port=None, transparent_port=0,
                       original_dst=lambda _s: ("127.0.0.1", origin_port), upstream_proxy=upstream_proxy)


async def test_the_relay_tunnels_the_vetted_address_through_the_upstream_proxy(monkeypatch):
    origin, port = await _echo_server()
    upstream = FakeUpstreamProxy()
    url = await upstream.start()
    try:
        request = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        assert await _through_transparent(_transparent(port, url), request, monkeypatch) == request
        [head] = upstream.heads
        # The proxy is asked for the address the interceptor vetted, never the claimed name.
        assert head.startswith(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n".encode())
    finally:
        await upstream.stop()
        origin.close()
        await origin.wait_closed()


async def test_a_tunnel_the_upstream_proxy_refuses_drops_the_connection(monkeypatch):
    origin, port = await _echo_server()
    upstream = FakeUpstreamProxy(connect_status=403)
    url = await upstream.start()
    try:
        request = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        assert await _through_transparent(_transparent(port, url), request, monkeypatch) == b""
        # The client's bytes never reached the proxy as a request of their own.
        assert len(upstream.heads) == 1 and upstream.heads[0].startswith(b"CONNECT ")
    finally:
        await upstream.stop()
        origin.close()
        await origin.wait_closed()


async def test_a_tunnel_the_upstream_proxy_never_answers_drops_the_connection_in_time(monkeypatch):
    monkeypatch.setattr(proxy_mod, "_CONNECT_REPLY_SEC", 0.3)
    origin, port = await _echo_server()
    upstream = FakeUpstreamProxy(connect_status=None)
    url = await upstream.start()
    try:
        started = asyncio.get_running_loop().time()
        request = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        # Bounded, so a relay that waits on the reply forever fails here instead of hanging.
        reply = await asyncio.wait_for(_through_transparent(_transparent(port, url), request, monkeypatch), 10)
        assert reply == b""
        # Dropped once the reply was overdue, not when the client gave up after 5 s.
        assert asyncio.get_running_loop().time() - started < 3
    finally:
        await upstream.stop()
        origin.close()
        await origin.wait_closed()


async def test_without_an_upstream_proxy_the_relay_dials_directly(monkeypatch):
    origin, port = await _echo_server()
    try:
        request = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        assert await _through_transparent(_transparent(port, None), request, monkeypatch) == request
    finally:
        origin.close()
        await origin.wait_closed()


async def test_the_index_fetches_through_the_named_proxy_and_ignores_the_environments(monkeypatch):
    upstream = FakeUpstreamProxy()
    url = await upstream.start()
    # A proxy in the environment that nothing listens on: using it would fail the fetch.
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    proxy = EgressProxy({"domains": {"mode": "allow", "entries": []}, "packages": None}, host="127.0.0.1", port=0,
                        upstream_proxy=url)
    try:
        status, body, _ = await proxy._fetch("http://pypi.example/simple/x/", hold=proxy._pages.hold())
        assert status == 200 and body == b"via proxy: GET http://pypi.example/simple/x/ HTTP/1.1"
    finally:
        await proxy.aclose()
        await upstream.stop()


#: An upstream proxy pasted with its credentials.
WITH_CREDENTIALS = "http://user:secret@host:3128"


def _texts(error: BaseException | None) -> str:
    """The text of *error* and of every exception it was raised from."""
    texts = []
    while error is not None:
        texts.append(str(error))
        error = error.__cause__ or error.__context__
    return "\n".join(texts)


def test_a_refused_upstream_proxy_is_described_without_any_part_of_its_value(tmp_path):
    with pytest.raises(ValueError) as checked:
        check_upstream_proxy(WITH_CREDENTIALS)
    with pytest.raises(ValueError) as constructed:
        EgressProxy({"domains": {"mode": "allow", "entries": []}}, host="127.0.0.1", upstream_proxy=WITH_CREDENTIALS)
    config = tmp_path / "hls.toml"
    config.write_text('[site]\npartition = "p"\nstore_root = "/s"\nnode_scratch = "/n"\n'
                      'secrets_path = "/k"\n')
    with pytest.raises(SettingsError) as loaded:
        load_settings(config, {"site.egress_upstream_proxy": WITH_CREDENTIALS})
    assert str(loaded.value).startswith(f"{config}: site.egress_upstream_proxy: ")
    for refused in (checked, constructed, loaded):
        # The settings file's path, which the refusal names, is the test's own.
        text = _texts(refused.value).replace(str(config), "")
        assert "it holds credentials" in text
        # The placeholder of the rule the message states is no part of the value.
        for part in ("user", "secret", "host", "3128"):
            assert part not in text.replace("<host>", ""), (part, text)


@pytest.mark.parametrize("url", ["https://p:1", "http://p", "http://u:pw@p:1", "http://u@p:1", "http://p:1/x",
                                 "http://p:1?q", "http://p:1#f", "p:1", "http://p:0"])
def test_the_interceptor_refuses_an_upstream_proxy_that_is_not_host_and_port(url):
    with pytest.raises(ValueError, match="http://<host>:<port>"):
        check_upstream_proxy(url)
    with pytest.raises(ValueError, match="http://<host>:<port>"):
        EgressProxy({"domains": {"mode": "allow", "entries": []}}, host="127.0.0.1", upstream_proxy=url)
