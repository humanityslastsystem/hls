"""Unit + loopback-integration tests for the filtering egress proxy.

The proxy has two roles: a transparent interceptor gated by the domain rule, and a PyPI
pull-through index gated by the package rule. These tests exercise the gates with NO real
upstream (upstream points at a dead port so any accidental fetch fails loudly), plus loopback
tests that relay an allowed host through the interceptor and refuse a denied one.
"""
from __future__ import annotations

import asyncio
import base64
import json
import socket
import ssl

import httpx
import pydantic
import pytest

from hls.egress.policy import AccessList, NetworkPolicy, SealPolicy
from hls.egress.proxy import (
    EgressProxy,
    _dist_name_from_url,
    _file_host_ok,
    _install_upstream_mark,
    _seal_policy,
    _sniff_host,
)

DEAD_UPSTREAM = "http://127.0.0.1:9"   # discard port; any fetch here fails fast


def _make_proxy(policy: NetworkPolicy, **kwargs: object) -> EgressProxy:
    """Construct an EgressProxy from a NetworkPolicy object."""
    kwargs.setdefault("host", "127.0.0.1")
    kwargs.setdefault("upstream", DEAD_UPSTREAM)
    return EgressProxy(policy.model_dump(), **kwargs)  # type: ignore[arg-type]


def _policy_dump(
    domains_mode: str,
    domains: list[str],
    pkgs_mode: str,
    pkgs: list[str],
) -> str:
    return json.dumps({
        "domains": {"mode": domains_mode, "entries": domains},
        "packages": {"mode": pkgs_mode, "entries": pkgs},
    })


# ---------------------------------------------------------------------------
# Flat-format tests
# ---------------------------------------------------------------------------


def _from_dump(policy_json: str) -> EgressProxy:
    """A proxy built from a flat ``NetworkPolicy.model_dump()`` JSON, as ``from_env`` reads it."""
    return EgressProxy(json.loads(policy_json), host="127.0.0.1")


def test_proxy_package_denylist_blocks_denied_name():
    p = _from_dump(_policy_dump("deny", [], "deny", ["tensorflow"]))
    assert p.package_permitted("tensorflow") is False
    assert p.package_permitted("numpy") is True


def test_proxy_domain_denylist_blocks_denied_host():
    p = _from_dump(_policy_dump("deny", ["evil.com", "*.evil.com"], "deny", []))
    assert p.domain_permitted("evil.com") is False
    assert p.domain_permitted("a.evil.com") is False
    assert p.domain_permitted("pypi.org") is True


def test_proxy_packages_none_permits_all():
    p = _from_dump(json.dumps({"domains": {"mode": "deny", "entries": []}, "packages": None}))
    assert p.package_permitted("anything") is True


# ---------------------------------------------------------------------------
# Gates on the flat NetworkPolicy.model_dump() shape
# ---------------------------------------------------------------------------


async def test_simple_index_denied_package_403():
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["torch"]))
    proxy = _make_proxy(pol, port=8888)
    # Denied package: short-circuits before any upstream call.
    status, body = await proxy.handle_simple("torch", proxy._pages.hold())
    assert status == 403
    assert b"torch" in body
    # Gate honors PEP 503 normalization.
    assert proxy._pkg_allowed("numpy") is True
    assert proxy._pkg_allowed("Torch") is False
    assert proxy._pkg_allowed("to_rch") is True  # different name, allowed


async def test_pkg_allowed_allowlist_normalizes():
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="allow", entries=["My_Pkg.X"]))
    proxy = _make_proxy(pol)
    assert proxy._pkg_allowed("my-pkg-x") is True
    assert proxy._pkg_allowed("numpy") is False


async def test_host_allowed_wildcard():
    pol = NetworkPolicy(domains=AccessList(mode="deny", entries=["*.evil.com"]))
    proxy = _make_proxy(pol)
    assert proxy._host_allowed("api.evil.com") is False
    assert proxy._host_allowed("evil.com") is True          # bare suffix not matched
    assert proxy._host_allowed("api.anthropic.com") is True


async def test_simple_index_rewrites_hrefs_and_preserves_hash():
    # Allowed package: the gate passes and the rewrite turns every wheel/sdist
    # href into a /files/<token> URL while preserving the #sha256 fragment.
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="allow", entries=["numpy"]))
    proxy = _make_proxy(pol, port=8888)
    upstream_html = (
        '<!DOCTYPE html><html><body>'
        '<a href="https://files.pythonhosted.org/packages/ab/numpy-1.0.whl#sha256=deadbeef">numpy-1.0.whl</a>'
        '<a href="https://files.pythonhosted.org/packages/cd/numpy-1.0.tar.gz">numpy-1.0.tar.gz</a>'
        '</body></html>'
    )

    captured = {}

    async def fake_fetch(url, **kwargs):
        captured["url"] = url
        captured["headers"] = kwargs.get("headers")
        return 200, upstream_html.encode(), "text/html"

    proxy._fetch = fake_fetch  # type: ignore[assignment]
    status, body = await proxy.handle_simple("numpy", proxy._pages.hold())
    assert status == 200
    text = body.decode()
    # both hrefs rewritten to the proxy's root-relative /files/ endpoint
    assert text.count('href="/files/') == 2
    assert "files.pythonhosted.org" not in text
    # the sha256 fragment is preserved verbatim
    assert "#sha256=deadbeef" in text
    # the signed token round-trips back to the original wheel URL through the
    # verifier (only proxy-emitted tokens resolve); the real filename is preserved as
    # the trailing segment so pip can parse the version
    ref = text.split('href="/files/')[1].split('"')[0]  # <token>/<filename>#frag
    token = ref.split("/")[0]
    assert proxy._verify_token(token) == (
        "https://files.pythonhosted.org/packages/ab/numpy-1.0.whl")
    assert "/numpy-1.0.whl" in ref
    assert captured["url"].endswith("/simple/numpy/")
    # We ask the upstream for HTML so uv/pip parse reliably (no PEP 691
    # JSON ambiguity).
    assert captured["headers"] is not None
    assert "text/html" in captured["headers"].get("Accept", "")


async def test_simple_index_absolutizes_relative_href():
    # A mirror that emits a RELATIVE artifact href must still resolve --
    # the rewrite absolutizes it against the simple page URL before encoding.
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="allow", entries=["numpy"]))
    proxy = EgressProxy(
        pol.model_dump(), upstream="https://mirror.test", host="127.0.0.1", port=8888
    )
    upstream_html = (
        '<!DOCTYPE html><html><body>'
        '<a href="../../packages/ab/numpy-1.0.whl#sha256=cafe">numpy-1.0.whl</a>'
        '</body></html>'
    )

    async def fake_fetch(url, **kwargs):
        return 200, upstream_html.encode(), "text/html"

    proxy._fetch = fake_fetch  # type: ignore[assignment]
    status, body = await proxy.handle_simple("numpy", proxy._pages.hold())
    assert status == 200
    text = body.decode()
    ref = text.split('href="/files/')[1].split('"')[0]
    token = ref.split("/")[0]
    # relative href resolved against https://mirror.test/simple/numpy/, then signed
    assert proxy._verify_token(token) == "https://mirror.test/packages/ab/numpy-1.0.whl"
    assert "#sha256=cafe" in text


def test_resolve_file_url_rejects_spoofed_host():
    # exact host match (defense in depth, AFTER the HMAC verifies): even a validly-signed
    # token whose host is a look-alike that merely ends with "pythonhosted.org"
    # (evilpythonhosted.org) is rejected; a signed legit files host yields the url to
    # fetch. (_resolve_file_url gates BEFORE any streaming.) Tokens are signed with the
    # proxy's own secret so they pass the signature check and reach the host gate.
    proxy = _make_proxy(NetworkPolicy())
    spoof = proxy._sign_token("https://evilpythonhosted.org/x/pkg-1.0.whl")
    url, err = proxy._resolve_file_url(spoof)
    assert url is None and err == (403, b"file host not allowed")
    legit = proxy._sign_token("https://files.pythonhosted.org/packages/ab/pkg-1.0.whl")
    url, err = proxy._resolve_file_url(legit)
    assert err is None and url.startswith("https://files.pythonhosted.org/")


def test_resolve_file_url_rejects_non_http_scheme():
    # a validly-signed token whose url has a non-http(s) scheme on a legit-looking host
    # is rejected (scheme gate is defense in depth after the HMAC verifies).
    proxy = _make_proxy(NetworkPolicy())
    token = proxy._sign_token("gopher://files.pythonhosted.org/packages/x.whl")
    url, err = proxy._resolve_file_url(token)
    assert url is None and err == (403, b"file scheme not allowed")


def test_resolve_file_url_metadata_suffix():
    # PEP 658: "<token>/<wheel>.metadata" appends ".metadata" to the upstream url; a plain
    # "<token>/<wheel>" resolves to the wheel itself.
    proxy = _make_proxy(NetworkPolicy())
    token = proxy._sign_token(
        "https://files.pythonhosted.org/packages/ab/six-1.0-py3-none-any.whl")
    url, err = proxy._resolve_file_url(f"{token}/six-1.0-py3-none-any.whl")
    assert err is None and url.endswith("/six-1.0-py3-none-any.whl")
    url, err = proxy._resolve_file_url(f"{token}/six-1.0-py3-none-any.whl.metadata")
    assert err is None and url.endswith("/six-1.0-py3-none-any.whl.metadata")


def test_resolve_file_url_decodes_padded_token():
    """pip percent-encodes the base64 '=' padding (-> %3D); the index url-decodes the
    token before verifying it, else padded tokens 400. The signed token carries
    '=' padding on its base64 fields, so this must survive pip's percent-encoding."""
    proxy = _make_proxy(NetworkPolicy())
    real = ("https://files.pythonhosted.org/packages/e3/a6a/"
            "httptools-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl")
    token = proxy._sign_token(real)
    assert token.endswith("="), "pick a URL whose signed token carries padding for this regression"
    quoted = token.replace("=", "%3D")            # exactly what pip sends in the request
    name = "httptools-0.8.0-cp312-cp312-manylinux_2_28_x86_64.whl.metadata"
    url, err = proxy._resolve_file_url(f"{quoted}/{name}")
    assert err is None and url == real + ".metadata"


async def test_fetch_does_not_follow_redirect_to_denied_host():
    # SSRF: _fetch with a host gate must NOT follow a redirect whose
    # Location points at a host the gate rejects. We drive _fetch directly with a
    # mocked httpx transport that 302s files.pythonhosted.org -> 169.254.169.254.
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=["files.pythonhosted.org"]))
    proxy = _make_proxy(pol)

    hits = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        if request.url.host == "files.pythonhosted.org":
            return httpx.Response(
                302, headers={"location": "http://169.254.169.254/latest/meta-data/"}
            )
        return httpx.Response(200, content=b"SECRET")

    transport = httpx.MockTransport(handler)

    def gate(host: str) -> bool:
        return proxy._host_allowed(host)

    status, body, _ = await proxy._fetch(
        "https://files.pythonhosted.org/packages/x.whl",
        hold=proxy._pages.hold(),
        gate=gate,
        transport=transport,
    )
    # the redirect target is denied -> we must NOT have fetched it
    assert "http://169.254.169.254/latest/meta-data/" not in hits
    assert body != b"SECRET"
    assert status in (403, 502) or 300 <= status < 400


async def test_from_env_builds_policy(monkeypatch):
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["torch"]))
    monkeypatch.setenv("HLS_EGRESS_POLICY", SealPolicy(network=pol).model_dump_json())
    monkeypatch.setenv("HLS_PROXY_HOST", "10.0.0.5")
    monkeypatch.setenv("HLS_PROXY_PORT", "9999")
    # Not a variable the seal sets: the index's upstream is PyPI's.
    monkeypatch.setenv("HLS_PYPI_UPSTREAM", "https://example-mirror.test")
    proxy = EgressProxy.from_env()
    assert proxy.host == "10.0.0.5"
    assert proxy.port == 9999
    assert proxy.upstream == "https://pypi.org"
    assert proxy._pkg_allowed("torch") is False
    assert proxy._pkg_allowed("numpy") is True


#: Policies the seal could write, as ``SealPolicy`` builds them.
SEALS = (
    SealPolicy(network=NetworkPolicy(domains=AccessList(mode="allow", entries=["gw.internal", "10.1.2.3"])),
               trusted_internal_hosts=("gw.internal", "10.1.2.3")),
    SealPolicy(network=NetworkPolicy(domains=AccessList(mode="allow", entries=[]),
                                     packages=AccessList(mode="deny", entries=["torch"]))),
    SealPolicy(network=NetworkPolicy(domains=AccessList(mode="deny", entries=["*.evil.com"]))),
)
#: Shapes no ``SealPolicy`` dumps: a bare network policy, a network policy that carries a vouched
#: host, an unknown key at either level, a domain list that is no access list, an empty or
#: non-string vouched host.
NOT_SEALS = (
    {"domains": {"mode": "allow", "entries": []}, "packages": None},
    {"domains": {"mode": "allow", "entries": []}, "packages": None, "trusted_internal_hosts": ["10.9.9.9"]},
    {"network": {"domains": {"mode": "allow", "entries": []}, "packages": None}, "trusted_internal_hosts": [],
     "extra": 1},
    {"network": {"domains": {"mode": "allow", "entries": []}, "packages": None, "trusted_internal_hosts": []},
     "trusted_internal_hosts": []},
    {"network": {"domains": {"mode": "open", "entries": []}, "packages": None}, "trusted_internal_hosts": []},
    {"network": {"domains": {"mode": "allow", "entries": "x"}, "packages": None}, "trusted_internal_hosts": []},
    {"network": {"domains": {"mode": "allow", "entries": []}, "packages": None}, "trusted_internal_hosts": [""]},
    {"network": {"domains": {"mode": "allow", "entries": []}, "packages": None}, "trusted_internal_hosts": [7]},
    [],
)


def test_the_interceptor_reads_exactly_what_a_seal_policy_dumps():
    """The interceptor imports nothing from hls, so it checks the shape of its policy itself
    (``_seal_policy``): every ``SealPolicy``'s dump passes and gives back its network policy and
    vouched hosts, and every shape the model refuses, the reader refuses too."""
    for seal in SEALS:
        assert _seal_policy(seal.model_dump_json()) == (seal.network.model_dump(mode="json"),
                                                        seal.trusted_internal_hosts)
    for bad in NOT_SEALS:
        with pytest.raises(pydantic.ValidationError):
            SealPolicy.model_validate(bad)
        with pytest.raises(ValueError):
            _seal_policy(json.dumps(bad))


async def test_from_env_reads_the_seals_vouched_hosts_and_refuses_a_policy_that_is_not_a_seals(monkeypatch):
    """The interceptor takes the hosts the harness vouches for from the seal's policy, and a
    policy of any other shape is refused before anything listens."""
    monkeypatch.setenv("HLS_PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("HLS_PROXY_TRANSPARENT_PORT", "0")
    monkeypatch.setenv("HLS_EGRESS_POLICY", SEALS[0].model_dump_json())
    proxy = EgressProxy.from_env()
    assert proxy._egress_host_allowed("10.1.2.3") is True
    assert proxy._egress_host_allowed("10.9.9.9") is False
    for bad in NOT_SEALS:
        monkeypatch.setenv("HLS_EGRESS_POLICY", json.dumps(bad))
        with pytest.raises(ValueError):
            EgressProxy.from_env()


async def test_from_env_minimal_env_denylist_round_trips(monkeypatch):
    # from_env validates the seal's policy and evaluates it with the proxy's own evaluator, with
    # only HLS_EGRESS_POLICY + HLS_PROXY_HOST + HLS_PROXY_PORT set.
    pol = NetworkPolicy(
        domains=AccessList(mode="deny", entries=["*.evil.com"]),
        packages=AccessList(mode="deny", entries=["torch"]),
    )
    monkeypatch.setenv("HLS_EGRESS_POLICY", SealPolicy(network=pol).model_dump_json())
    monkeypatch.setenv("HLS_PROXY_HOST", "0.0.0.0")
    monkeypatch.setenv("HLS_PROXY_PORT", "8888")

    proxy = EgressProxy.from_env()
    assert proxy.host == "0.0.0.0"
    assert proxy.port == 8888                       # index listener bound
    assert proxy.upstream == "https://pypi.org"
    # Denied package + denied host (PEP 503 normalize + *.suffix wildcard) blocked;
    # everything else permitted under denylist semantics.
    assert proxy._pkg_allowed("torch") is False
    assert proxy._pkg_allowed("Torch") is False     # PEP 503 normalize
    assert proxy._pkg_allowed("numpy") is True
    assert proxy._host_allowed("api.evil.com") is False   # *.suffix wildcard
    assert proxy._host_allowed("evil.com") is True        # bare suffix not matched
    assert proxy._host_allowed("api.anthropic.com") is True


async def test_from_env_no_index_port_runs_transparent_only(monkeypatch):
    # Runtime gateway: a MISSING HLS_PROXY_PORT disables the pull-through index
    # (port=None) -- the served artifact runs no pip, so the index is pure attack
    # surface and must not bind. Only the transparent interceptor listens.
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=["example.com"]))
    monkeypatch.delenv("HLS_PROXY_PORT", raising=False)
    monkeypatch.setenv("HLS_EGRESS_POLICY", SealPolicy(network=pol).model_dump_json())
    monkeypatch.setenv("HLS_PROXY_HOST", "0.0.0.0")
    monkeypatch.setenv("HLS_PROXY_TRANSPARENT_PORT", "8888")

    proxy = EgressProxy.from_env()
    assert proxy.port is None                       # index disabled
    assert proxy.transparent_port == 8888

    # serve_forever must NOT start the index listener when port is None: stub both
    # start methods to record which ran (and stop the gather immediately).
    started: list[str] = []

    async def _fake_start():
        started.append("index")
        raise AssertionError("index listener must not start in transparent-only mode")

    async def _fake_start_transparent():
        started.append("transparent")
        raise asyncio.CancelledError      # unwind serve_forever once we've recorded

    proxy.start = _fake_start                        # type: ignore[assignment]
    proxy.start_transparent = _fake_start_transparent  # type: ignore[assignment]
    try:
        await proxy.serve_forever()
    except asyncio.CancelledError:
        pass
    assert started == ["transparent"]                # only the transparent listener


async def test_from_env_requires_at_least_one_port(monkeypatch):
    # With neither HLS_PROXY_PORT nor HLS_PROXY_TRANSPARENT_PORT set the proxy would
    # bind no listener -- from_env errors clearly instead of starting a no-op.
    import pytest
    pol = NetworkPolicy()
    monkeypatch.delenv("HLS_PROXY_PORT", raising=False)
    monkeypatch.delenv("HLS_PROXY_TRANSPARENT_PORT", raising=False)
    monkeypatch.setenv("HLS_EGRESS_POLICY", SealPolicy(network=pol).model_dump_json())
    monkeypatch.setenv("HLS_PROXY_HOST", "0.0.0.0")
    with pytest.raises(ValueError, match="at least one"):
        EgressProxy.from_env()


def test_proxy_has_no_toplevel_hls_import():
    # The proxy ships into a sidecar/sandbox that lacks the `hls` package, so it
    # must NOT import `hls` at module top level. Parse the AST and assert no
    # top-level `import hls...` / `from hls... import ...` statement exists.
    import ast
    from pathlib import Path

    import hls.egress.proxy as proxy_mod

    src = Path(proxy_mod.__file__).read_text()
    tree = ast.parse(src)
    for node in tree.body:                          # top-level statements only
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.split(".")[0] == "hls", (
                    f"top-level `import {alias.name}` -- proxy must be hls-free")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root != "hls", (
                f"top-level `from {node.module} import ...` -- proxy must be hls-free")


# ---------------------------------------------------------------------------
# Transparent-intercept role: SNI/Host sniff (no decrypt) + gated blind relay
# ---------------------------------------------------------------------------


def _client_hello(server_hostname: str) -> bytes:
    """A REAL TLS ClientHello carrying ``server_hostname`` as SNI, produced via a
    MemoryBIO handshake (no socket, no cert) -- the exact bytes a client sends first."""
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    inbio, outbio = ssl.MemoryBIO(), ssl.MemoryBIO()
    sslobj = ctx.wrap_bio(inbio, outbio, server_hostname=server_hostname)
    try:
        sslobj.do_handshake()
    except ssl.SSLWantReadError:
        pass
    return outbio.read()


async def _serve_raw_echo() -> "tuple[asyncio.AbstractServer, int]":
    """A loopback origin that echoes raw bytes back (to prove the interceptor relayed
    the exact ClientHello / request to the gated host)."""

    async def handle(reader, writer):
        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                writer.write(chunk)
                await writer.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    return server, server.sockets[0].getsockname()[1]


# --- pure sniff tests (no sockets) ---


def test_sniff_real_client_hello_extracts_sni():
    host, status = _sniff_host(_client_hello("example.test"))
    assert status == "ok"
    assert host == "example.test"


def test_sniff_http_host_strips_port_and_lowercases():
    host, status = _sniff_host(b"POST /v1/x HTTP/1.1\r\nHost: Example.COM:3456\r\n\r\n")
    assert status == "ok"
    assert host == "example.com"


def test_sniff_partial_tls_record_needs_more():
    assert _sniff_host(b"\x16\x03\x01")[1] == "need_more"          # < 5 byte record header
    full = _client_hello("x.test")
    assert _sniff_host(full[:-10])[1] == "need_more"               # record not yet complete


def test_sniff_tls_without_sni_is_bad():
    # A complete handshake record whose ClientHello carries no server_name -> bad.
    host, status = _sniff_host(b"\x16\x03\x01\x00\x04\x01\x00\x00\x00")
    assert host is None and status == "bad"


def test_sniff_non_http_non_tls_is_bad():
    host, status = _sniff_host(b"GARBAGE not a request\r\n\r\n")
    assert host is None and status == "bad"


def test_install_upstream_mark_stamps_new_sockets(monkeypatch):
    """The mark is set in the socket CONSTRUCTOR (before connect) so it lands on the
    SYN -- verified via a recording base so it runs unprivileged (real SO_MARK needs
    CAP_NET_ADMIN). This is what makes the gateway exempt its own upstream egress."""
    import socket as _socket

    recorded = []

    class _RecordingBase:
        def __init__(self, *a, **k):
            pass

        def setsockopt(self, level, optname, value):
            recorded.append((level, optname, value))

        def close(self):
            pass

    monkeypatch.setattr(_socket, "socket", _RecordingBase)
    _install_upstream_mark(0x4853)
    _socket.socket()  # constructed through the installed mark-stamping subclass
    assert (_socket.SOL_SOCKET, _socket.SO_MARK, 0x4853) in recorded


def test_install_upstream_mark_fails_loud_when_unset(monkeypatch):
    """If SO_MARK cannot be applied (e.g. CAP_NET_ADMIN missing in the seal netns), refuse to
    install the half-working override -- a silently-dropped mark loops the proxy's own upstream
    back into the REDIRECT and breaks the seal with no diagnostic."""
    import socket as _socket

    class _RejectingBase:
        def __init__(self, *a, **k):
            pass

        def setsockopt(self, level, optname, value):
            raise OSError("SO_MARK not permitted")

        def close(self):
            pass

    monkeypatch.setattr(_socket, "socket", _RejectingBase)
    with pytest.raises(RuntimeError, match="SO_MARK"):
        _install_upstream_mark(0x4853)


# --- loopback transparent-intercept integration ---


async def _run_transparent(policy: NetworkPolicy, payload: bytes, *, origin_port: int,
                           interceptor_dst: int):
    """Send ``payload`` through a transparent listener whose SO_ORIGINAL_DST is stubbed
    to ``(127.0.0.1, interceptor_dst)``; return whatever bytes come back (b"" = dropped)."""
    proxy = EgressProxy(policy.model_dump(), host="127.0.0.1", port=0, transparent_port=0,
                        original_dst=lambda _s: ("127.0.0.1", interceptor_dst))
    # These tests relay to a LOOPBACK echo server standing in for the gated upstream. The
    # production resolve-and-vet step (``_resolve_global_ipv4``) correctly REFUSES a name that
    # resolves to 127.0.0.1 (the DNS-name SSRF guard, covered by its own unit test), so stub it
    # to return the loopback upstream here -- this exercises the relay, not the SSRF gate.
    import hls.egress.proxy as _pm
    _orig_resolve = _pm._resolve_global_ipv4

    async def _loopback_resolve(_host, _port):
        return "127.0.0.1"

    _pm._resolve_global_ipv4 = _loopback_resolve
    server = await proxy.start_transparent()
    iport = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", iport)
        writer.write(payload)
        await writer.drain()
        try:
            return await asyncio.wait_for(reader.readexactly(len(payload)), 5)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            data = b""
            try:
                data = await asyncio.wait_for(reader.read(), 2)
            except asyncio.TimeoutError:
                pass
            return data
        finally:
            writer.close()
    finally:
        _pm._resolve_global_ipv4 = _orig_resolve
        server.close()
        await server.wait_closed()


async def test_transparent_http_allowed_relays_to_gated_name():
    origin, oport = await _serve_raw_echo()
    try:
        pol = NetworkPolicy(domains=AccessList(mode="allow", entries=["localhost"]))
        req = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        echoed = await _run_transparent(pol, req, origin_port=oport, interceptor_dst=oport)
        assert echoed == req            # relayed to the gated host and echoed back
    finally:
        origin.close()
        await origin.wait_closed()


async def test_transparent_http_denied_host_dropped():
    origin, oport = await _serve_raw_echo()
    try:
        pol = NetworkPolicy(domains=AccessList(mode="deny", entries=["localhost"]))
        req = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        out = await _run_transparent(pol, req, origin_port=oport, interceptor_dst=oport)
        assert out == b""               # denied -> connection dropped, never relayed
    finally:
        origin.close()
        await origin.wait_closed()


async def test_transparent_tls_sni_allowed_relays():
    origin, oport = await _serve_raw_echo()
    try:
        pol = NetworkPolicy(domains=AccessList(mode="allow", entries=["localhost"]))
        ch = _client_hello("localhost")
        echoed = await _run_transparent(pol, ch, origin_port=oport, interceptor_dst=oport)
        assert echoed == ch             # real ClientHello peeked, gated, relayed verbatim
    finally:
        origin.close()
        await origin.wait_closed()


async def test_transparent_tls_sni_denied_dropped():
    origin, oport = await _serve_raw_echo()
    try:
        pol = NetworkPolicy(domains=AccessList(mode="allow", entries=["allowed.test"]))
        ch = _client_hello("denied.test")          # SNI not in the allowlist
        out = await _run_transparent(pol, ch, origin_port=oport, interceptor_dst=oport)
        assert out == b""
    finally:
        origin.close()
        await origin.wait_closed()


async def test_transparent_unknown_protocol_dropped():
    origin, oport = await _serve_raw_echo()
    try:
        pol = NetworkPolicy(domains=AccessList(mode="allow", entries=["localhost"]))
        # Looks like TLS (0x16) but the handshake is not a ClientHello -> fail closed.
        out = await _run_transparent(pol, b"\x16\x03\x01\x00\x04\x09\x00\x00\x00",
                                     origin_port=oport, interceptor_dst=oport)
        assert out == b""
    finally:
        origin.close()
        await origin.wait_closed()


async def test_transparent_package_host_denied_when_package_policy_active():
    """With a package policy active, the index hosts are unreachable directly (forces
    packages through the local pull-through index), even if domains would allow them."""
    origin, oport = await _serve_raw_echo()
    try:
        pol = NetworkPolicy(
            domains=AccessList(mode="deny", entries=["evil.com"]),       # domains: permissive (only evil.com denied)
            packages=AccessList(mode="deny", entries=["tensorflow"]),    # package policy active
        )
        req = b"GET /simple/ HTTP/1.1\r\nHost: pypi.org\r\n\r\n"
        out = await _run_transparent(pol, req, origin_port=oport, interceptor_dst=oport)
        assert out == b""               # pypi.org not directly reachable by the client
    finally:
        origin.close()
        await origin.wait_closed()


# --- transparent-path egress gate: package file-host block ---
# (The domain seal is the Docker transparent gateway + Modal's outbound_domain_allowlist,
#  so the only consumer of this gate is _on_transparent_client. End-to-end coverage:
#  test_transparent_package_host_denied_when_package_policy_active.)


def test_egress_host_allowed_blocks_index_hosts_when_package_policy_active():
    # While a package policy is active, pypi/pythonhosted are refused as raw egress so
    # packages can't be fetched past the filtering index. On Docker the transparent
    # intercept applies this gate; on Modal the same is enforced natively by leaving the
    # index hosts out of outbound_domain_allowlist.
    pol = NetworkPolicy(
        domains=AccessList(mode="deny", entries=["evil.com"]),      # domains: permissive (only evil.com denied)
        packages=AccessList(mode="deny", entries=["tensorflow"]))   # package policy active
    proxy = _make_proxy(pol)
    assert proxy._egress_host_allowed("files.pythonhosted.org") is False
    assert proxy._egress_host_allowed("pypi.org") is False
    assert proxy._egress_host_allowed("example.com") is True        # normal host still allowed


def test_egress_host_allowed_permits_index_hosts_without_package_policy():
    # No package policy -> no index to force traffic through -> pypi/pythonhosted are
    # gated by the domain rule like any other host.
    pol = NetworkPolicy(domains=AccessList(mode="deny", entries=[]))  # open, packages=None
    proxy = _make_proxy(pol)
    assert proxy._egress_host_allowed("files.pythonhosted.org") is True
    assert proxy._egress_host_allowed("pypi.org") is True


def test_egress_host_allowed_exempts_vouched_internal_literal():
    """A substrate-vouched gateway given as an internal IP LITERAL
    (ANTHROPIC_BASE_URL=https://10.x) is exempt from the is-internal-literal SSRF reject, but a
    non-vouched internal literal is still refused. The vouched hosts are the seal's, beside the
    network policy (``SealPolicy``)."""
    policy = {"domains": {"mode": "allow", "entries": ["10.1.2.3"]}, "packages": None}
    proxy = EgressProxy(policy, host="127.0.0.1", port=0, transparent_port=0, trusted_internal_hosts=("10.1.2.3",))
    assert proxy._egress_host_allowed("10.1.2.3") is True       # vouched literal -> exempt
    assert proxy._egress_host_allowed("10.9.9.9") is False      # not vouched -> SSRF reject


async def _run_transparent_vouched(policy: dict, payload: bytes, *, interceptor_dst: int,
                                   resolves_to: str, vouched: tuple[str, ...]):
    """Drive the transparent handler for a VOUCHED host whose name resolves (stubbed) to
    ``resolves_to``. Returns the bytes echoed back (b"" = dropped). Stubs ``_resolve_ipv4``
    (the vouched branch), unlike ``_run_transparent`` which stubs ``_resolve_global_ipv4``."""
    proxy = EgressProxy(policy, host="127.0.0.1", port=0, transparent_port=0, trusted_internal_hosts=vouched,
                        original_dst=lambda _s: ("127.0.0.1", interceptor_dst))
    import hls.egress.proxy as _pm
    _orig = _pm._resolve_ipv4

    async def _fake_resolve(_host, _port, *, require_global):
        return resolves_to

    _pm._resolve_ipv4 = _fake_resolve
    server = await proxy.start_transparent()
    iport = server.sockets[0].getsockname()[1]
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", iport)
        writer.write(payload)
        await writer.drain()
        try:
            return await asyncio.wait_for(reader.readexactly(len(payload)), 5)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError):
            try:
                return await asyncio.wait_for(reader.read(), 2)
            except asyncio.TimeoutError:
                return b""
        finally:
            writer.close()
    finally:
        _pm._resolve_ipv4 = _orig
        server.close()
        await server.wait_closed()


async def test_transparent_vouched_host_pins_resolved_ip():
    """A vouched host that resolves to its substrate-vouched (pinned) IP relays, but the
    SAME vouched name rebound to a DIFFERENT internal address (e.g. cloud metadata) is dropped
    -- the proxy dials only a resolved IP that is itself in the vouched set."""
    origin, oport = await _serve_raw_echo()  # loopback echo stands in for the pinned upstream
    try:
        ch = _client_hello("gw.internal")
        policy = {"domains": {"mode": "allow", "entries": ["gw.internal"]}, "packages": None}
        vouched = ("gw.internal", "127.0.0.1")  # 127.0.0.1 = the pin
        # Pinned: name resolves to 127.0.0.1 (vouched) -> relayed and echoed.
        echoed = await _run_transparent_vouched(policy, ch, interceptor_dst=oport, resolves_to="127.0.0.1",
                                                vouched=vouched)
        assert echoed == ch
        # Rebind: same vouched name resolves to an UNPINNED internal addr -> dropped.
        out = await _run_transparent_vouched(policy, ch, interceptor_dst=oport, resolves_to="169.254.169.254",
                                             vouched=vouched)
        assert out == b""
    finally:
        origin.close()
        await origin.wait_closed()


def test_resolve_global_ipv4_refuses_internal_a_record():
    """The interceptor refuses a gated NAME whose A record is internal (the DNS-name SSRF guard),
    and one that does not resolve."""
    from hls.egress import proxy

    table = {
        "good.example": "93.184.216.34",
        "evil.internal": "10.0.2.2",
        "metadata": "169.254.169.254",
        "loopback": "127.0.0.1",
    }

    async def go(host):
        loop = asyncio.get_running_loop()

        async def fake_gai(h, p, **k):
            ip = table.get(h)
            if ip is None:
                raise socket.gaierror("nxdomain")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, p))]

        loop.getaddrinfo = fake_gai  # shadow on the running loop instance
        return await proxy._resolve_global_ipv4(host, 443)

    assert asyncio.run(go("good.example")) == "93.184.216.34"
    assert asyncio.run(go("evil.internal")) is None
    assert asyncio.run(go("metadata")) is None
    assert asyncio.run(go("loopback")) is None
    assert asyncio.run(go("nxdomain.example")) is None


def test_resolve_ipv4_permissive_allows_internal_for_vouched_host():
    """``require_global=False`` (used only for trusted_internal_hosts) resolves an internal
    address instead of refusing it: a model gateway on a private address stays reachable."""
    from hls.egress import proxy

    async def go(require_global):
        loop = asyncio.get_running_loop()

        async def fake_gai(h, p, **k):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.0.0.5", p))]

        loop.getaddrinfo = fake_gai
        return await proxy._resolve_ipv4("relay.example", 8080, require_global=require_global)

    assert asyncio.run(go(True)) is None  # strict: internal -> refused (SSRF guard)
    assert asyncio.run(go(False)) == "10.0.0.5"  # vouched: internal allowed


# --- /files/ enforces the package rule from the DECODED url ---


def test_resolve_file_url_enforces_package_rule_from_url():
    # /files/ re-checks the package name parsed from the DECODED url (what
    # the upstream serves), not the spoofable client filename. _resolve_file_url returns a
    # 403 error (no url) for a denied package, so it is never streamed.
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["tensorflow"]))
    proxy = _make_proxy(pol)

    def tok(u: str) -> str:
        return proxy._sign_token(u)

    denied = ("https://files.pythonhosted.org/packages/aa/bb/"
              "tensorflow-2.15.0-cp311-cp311-manylinux1_x86_64.whl")
    url, err = proxy._resolve_file_url(f"{tok(denied)}/tensorflow-2.15.0-cp311.whl")
    assert url is None and err == (403, b"package not allowed")

    # SPOOF: denied token, but the client filename claims an allowed package -> still 403
    # (the gate reads the url the upstream will serve, not the client's label).
    url, err = proxy._resolve_file_url(f"{tok(denied)}/six-1.16.0-py3-none-any.whl")
    assert url is None and err == (403, b"package not allowed")

    # denied sdist (.tar.gz name parsing): name is everything before the last '-'
    sdist = "https://files.pythonhosted.org/packages/ee/ff/tensorflow-2.15.0.tar.gz"
    url, err = proxy._resolve_file_url(f"{tok(sdist)}/tensorflow-2.15.0.tar.gz")
    assert url is None and err == (403, b"package not allowed")

    # an ALLOWED package resolves to its url (then streamed)
    allowed = ("https://files.pythonhosted.org/packages/cc/dd/"
               "six-1.16.0-py2.py3-none-any.whl")
    url, err = proxy._resolve_file_url(f"{tok(allowed)}/six-1.16.0-py2.py3-none-any.whl")
    assert err is None and url == allowed


def test_resolve_file_url_fails_closed_on_unparseable_name_with_package_policy():
    # Package policy active + a file-host url with no recognizable artifact filename ->
    # the dist name can't be recovered -> fail CLOSED (403 error, no url).
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="deny", entries=["tensorflow"]))
    proxy = _make_proxy(pol)
    no_file = proxy._sign_token("https://files.pythonhosted.org/packages/aa/bb/")
    url, err = proxy._resolve_file_url(f"{no_file}/x")
    assert url is None and err == (403, b"package not allowed")


# --- /files/ package-seal gates (host gate, token HMAC, fail-closed name) ---


def test_file_host_ok_rejects_writable_staging_mirrors():
    # The staging mirrors are world-writable, so admitting any *.pypi.org /
    # *.pythonhosted.org would let a sealed client publish + then fetch arbitrary code.
    # The host gate is EXACT-only -> the staging hosts are rejected, production stays ok.
    assert _file_host_ok("test.pypi.org") is False
    assert _file_host_ok("test-files.pythonhosted.org") is False
    # a generic look-alike suffix is still rejected (no dotted-suffix match)
    assert _file_host_ok("evilpythonhosted.org") is False
    assert _file_host_ok("files.pythonhosted.org.evil.com") is False
    # the production hosts stay reachable
    assert _file_host_ok("files.pythonhosted.org") is True
    assert _file_host_ok("pypi.org") is True
    assert _file_host_ok("pythonhosted.org") is True


def test_resolve_file_url_rejects_forged_and_wrong_secret_tokens():
    # The /files/ token is HMAC-signed with a per-PROCESS secret, so a client
    # cannot mint a token for an arbitrary pypi-family URL without calling /simple/.
    proxy = _make_proxy(NetworkPolicy())
    legit_url = "https://files.pythonhosted.org/packages/ab/numpy-1.0.whl"

    # (1) a plain base64(url) token with NO signature -> rejected (no '.' separator path)
    unsigned = base64.urlsafe_b64encode(legit_url.encode()).decode()
    url, err = proxy._resolve_file_url(unsigned)
    assert url is None and err == (400, b"bad token")

    # (2) a forged token: real url + attacker-chosen (garbage) signature -> rejected
    forged = base64.urlsafe_b64encode(legit_url.encode()).decode() + "." + \
        base64.urlsafe_b64encode(b"not-a-real-hmac-signature-32byte").decode()
    url, err = proxy._resolve_file_url(forged)
    assert url is None and err == (400, b"bad token")

    # (3) a token validly signed under a DIFFERENT process's secret -> rejected here
    other = _make_proxy(NetworkPolicy())
    foreign = other._sign_token(legit_url)
    url, err = proxy._resolve_file_url(foreign)
    assert url is None and err == (400, b"bad token")
    # ...but it resolves against the process that minted it (sanity check)
    url, err = other._resolve_file_url(foreign)
    assert err is None and url == legit_url


def test_rewrite_simple_token_round_trips_and_is_accepted():
    # A token emitted by the real _rewrite_simple verifies + resolves -- only
    # proxy-emitted (from /simple/) URLs are fetchable.
    pol = NetworkPolicy(domains=AccessList(mode="allow", entries=[]), packages=AccessList(mode="allow", entries=["numpy"]))
    proxy = _make_proxy(pol)
    real = "https://files.pythonhosted.org/packages/ab/numpy-1.0.whl"
    html = f'<a href="{real}#sha256=deadbeef">numpy-1.0.whl</a>'.encode()
    rewritten = proxy._rewrite_simple(html, "https://pypi.org/simple/numpy/").decode()
    ref = rewritten.split('href="/files/')[1].split('"')[0]   # <token>/<filename>#frag
    token = ref.split("#")[0]                                  # strip the sha256 fragment
    url, err = proxy._resolve_file_url(token)
    assert err is None and url == real


def test_dist_name_from_url_fails_closed_on_hyphenless_name():
    # A filename with no recognizable name-version structure -> None (fail closed),
    # so the /files/ package gate denies it instead of admitting a bare mis-attributed name.
    assert _dist_name_from_url("https://files.pythonhosted.org/p/weirdpkg.tar.gz") is None
    assert _dist_name_from_url("https://files.pythonhosted.org/p/noversion.whl") is None
    # an egg without a clean name-version-pyTag triple also fails closed
    assert _dist_name_from_url("https://files.pythonhosted.org/p/justname.egg") is None
    assert _dist_name_from_url("https://files.pythonhosted.org/p/name-1.0.egg") is None
    # well-formed artifacts still parse to their leading name
    assert _dist_name_from_url(
        "https://files.pythonhosted.org/p/numpy-1.0-py3-none-any.whl") == "numpy"
    assert _dist_name_from_url(
        "https://files.pythonhosted.org/p/numpy-1.0.tar.gz") == "numpy"
    assert _dist_name_from_url(
        "https://files.pythonhosted.org/p/foo-1.0-py3.10.egg") == "foo"


# --- a multi-record ClientHello is reassembled, not fail-closed-dropped ---


def _records(payload: bytes, *cuts: int) -> bytes:
    """Re-frame a handshake ``payload`` as consecutive TLS handshake records, split at
    the given byte offsets (each piece gets its own 5-byte record header)."""
    pieces, prev = [], 0
    for c in (*cuts, len(payload)):
        pieces.append(payload[prev:c])
        prev = c
    return b"".join(b"\x16\x03\x01" + len(p).to_bytes(2, "big") + p for p in pieces)


def test_sniff_host_reassembles_multi_record_client_hello():
    # A ClientHello legitimately split across multiple TLS records (large extensions
    # / PQ keyshares) must NOT fail closed -- the SNI may land in a later record.
    ch = _client_hello("files.pythonhosted.org")           # single-record ClientHello
    payload = ch[5:]
    assert _sniff_host(ch) == ("files.pythonhosted.org", "ok")
    # split across two records, at a point well before the SNI extension
    assert _sniff_host(_records(payload, 8)) == ("files.pythonhosted.org", "ok")
    # split into three records
    assert _sniff_host(_records(payload, 8, 20)) == ("files.pythonhosted.org", "ok")
    # only the first of several records buffered -> need_more (don't decide off it alone)
    assert _sniff_host(b"\x16\x03\x01" + len(payload[:8]).to_bytes(2, "big")
                       + payload[:8]) == (None, "need_more")


# ---------------------------------------------------------------------------
# The index serves /simple/ and /files/ only; a stage reads the status file.
# ---------------------------------------------------------------------------


async def _index_get(port: int, target: str) -> int:
    """GET ``target`` from a running index listener; return the HTTP status code.

    Sends a single HTTP/1.1 request over a fresh loopback connection and parses the status
    line."""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(f"GET {target} HTTP/1.1\r\nHost: x\r\n\r\n".encode())
    await writer.drain()
    try:
        data = await asyncio.wait_for(reader.read(4096), 5)
    finally:
        writer.close()
    # status line: b"HTTP/1.1 <code> <reason>\r\n..."
    return int(data.split(b" ", 2)[1])


async def test_the_index_answers_a_status_request_as_an_unknown_path():
    """No listener serves a status page; the stage reads the status file."""
    proxy = _make_proxy(NetworkPolicy(), port=0)
    server = await proxy.start()
    iport = server.sockets[0].getsockname()[1]
    try:
        status = await _index_get(iport, "/_status")
        assert status == await _index_get(iport, "/no-such-page") == 404
    finally:
        server.close()
        await server.wait_closed()


# --- slow-loris -- total header deadline + connection cap ---


class _DribbleReader:
    """A fake StreamReader that dribbles one byte per read FOREVER (never completing a
    request line / ClientHello), modelling a slow-loris client. Each read also sleeps a
    hair so wall-clock advances past the (shrunk) total deadline."""

    def __init__(self) -> None:
        self.reads = 0

    async def read(self, _n: int) -> bytes:
        self.reads += 1
        await asyncio.sleep(0.01)
        return b"\x16"          # a TLS-handshake first byte -> sniff stays "need_more"


async def test_peek_host_total_deadline_drops_slow_loris(monkeypatch):
    # A client that dribbles bytes to keep resetting the per-read timeout is dropped
    # once the TOTAL _CONN_HEADER_DEADLINE wall-clock budget is exhausted (host is None ->
    # caller fails closed, no egress). Shrink the deadline so the test runs fast.
    import hls.egress.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_CONN_HEADER_DEADLINE", 0.05)
    proxy = _make_proxy(NetworkPolicy())
    buf, host = await asyncio.wait_for(proxy._peek_host(_DribbleReader()), 5)
    assert host is None                       # deadline hit -> fail closed
    assert buf and set(buf) == {0x16}         # only dribbled bytes were buffered


async def test_recv_request_total_deadline_drops_slow_loris(monkeypatch):
    # The same total-deadline guard on the index listener's request-head read.
    import h11

    import hls.egress.proxy as proxy_mod
    monkeypatch.setattr(proxy_mod, "_CONN_HEADER_DEADLINE", 0.05)
    proxy = _make_proxy(NetworkPolicy())

    class _SlowHeaderReader:
        async def read(self, _n: int) -> bytes:
            await asyncio.sleep(0.01)
            return b"G"           # never completes the request line -> h11 NEED_DATA loop

    conn = h11.Connection(h11.SERVER)
    request = await asyncio.wait_for(proxy._recv_request(_SlowHeaderReader(), conn), 5)
    assert request is None                    # deadline hit -> dropped, no request served


def test_connection_slot_cap_drops_excess():
    # The connection cap takes slots without blocking and drops once full, so a
    # slow-loris flood can't queue/pin unbounded coroutines/fds.
    import hls.egress.proxy as proxy_mod
    proxy = _make_proxy(NetworkPolicy())
    proxy._conn_count = proxy_mod._MAX_CONNECTIONS - 1
    assert proxy._try_acquire_slot() is True          # last free slot
    assert proxy._conn_count == proxy_mod._MAX_CONNECTIONS
    assert proxy._try_acquire_slot() is False         # cap hit -> drop
    proxy._release_slot()
    assert proxy._conn_count == proxy_mod._MAX_CONNECTIONS - 1
    assert proxy._try_acquire_slot() is True          # a freed slot is reusable


# ---------------------------------------------------------------------------
# Shared-client reuse
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_ensure_client_returns_one_shared_instance():
    """The proxy must own a single long-lived httpx client, not one per request.

    ``_fetch`` and ``_stream_file`` share one client: repeated ``_ensure_client()``
    calls return the SAME object, and ``aclose()`` disposes it exactly once (idempotent).
    """
    pol = NetworkPolicy(
        domains=AccessList(mode="deny", entries=["example.com"]),
        packages=AccessList(mode="deny", entries=["tensorflow"]),
    )
    proxy = _make_proxy(pol)
    assert proxy._client is None                     # lazily created, not at construction
    c1 = proxy._ensure_client()
    c2 = proxy._ensure_client()
    assert c1 is c2                                   # reused, not re-created per call
    assert proxy._client is c1
    await proxy.aclose()
    assert proxy._client is None                      # disposed
    await proxy.aclose()                              # idempotent -- no raise
    assert c1.is_closed
