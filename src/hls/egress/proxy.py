"""The egress interceptor of the network seal: a filtering PyPI index and a transparent
intercept, in one asyncio process.

The Apptainer seal (``src/hls/substrate/apptainer/transparent_seal.sh``, set up by
``hls.substrate.apptainer.egress``) runs it with the plan's interpreter as ``python -m
hls.egress.proxy``, in the namespaces ``pasta`` makes for a sealed role but outside the
container, which can neither see nor signal it. The seal's iptables rules REDIRECT every TCP
connection the container makes to the transparent listener, except loopback and the proxy's own
upstream connections, which carry its mark (``HLS_PROXY_FWMARK``) and so leave the namespace
through pasta. The index listens on the namespace's loopback, where only the sealed container
reaches it, so it asks for no credentials. The proxy is NOT a forward proxy: it terminates no
CONNECT or absolute-form requests (those 405). It has at most two listeners, and at least one
(``from_env`` errors if neither port is set):

  * PyPI pull-through INDEX (when ``HLS_PROXY_PORT`` is set, for a step that installs
    packages): ``GET /simple/<name>/`` checks the package
    name against the package rule; on a hit it fetches the real simple page,
    REWRITES every wheel/sdist href to ``/files/<token>`` (token = base64url of the
    real, absolutized URL + a per-PROCESS HMAC of that URL) preserving any
    ``#sha256=...`` fragment so uv/pip can still hash-verify, and returns it as
    ``text/html``. ``GET /files/<token>`` VERIFIES the HMAC (only proxy-emitted URLs
    are fetchable -- a forged token 400s), RE-CHECKS the package name parsed from the
    decoded URL, and streams the file from a verified pythonhosted/pypi host. A
    disallowed name returns 403 with no upstream call. Any other path is not found (404).
    pip/uv reach this via ``*_INDEX_URL``.
  * Transparent INTERCEPT (when ``HLS_PROXY_TRANSPARENT_PORT`` is set):
    the seal REDIRECTs the sealed client's TCP here; we recover the original dst,
    read the host WITHOUT decrypting (TLS ClientHello SNI / HTTP Host), gate it by
    the domain rule (and refuse pypi/pythonhosted while a package policy is active,
    forcing packages through the index), then blind-relay. Fails closed.

Security model on the ``/files/`` fetch path: the destination host is gated,
redirects are NOT auto-followed (each ``Location`` re-runs the scheme + host gate),
and only http/https schemes are accepted -- keeping the seal intact against
redirect-based SSRF.

What the index holds is bounded whatever its clients do, since the interceptor runs in the
sealed step beside them and its memory counts against that step's (in the build sandbox, against
the harness's reserve). Simple pages are the only replies
it holds whole: each is read up to ``PAGE_MAX_BYTES``, and all it holds at once, read or
rewritten, are capped at ``_PAGES_MAX_BYTES`` (``_PageRoom``). A page request takes room for
a page before it asks the upstream, waiting for it, and is refused with a 503 when none comes,
as an upstream out of room would refuse it; a page read never waits, so no client can hold up
the upstream. Files
stream a chunk at a time. A client that takes none of a reply for ``_WRITE_TIMEOUT_SEC`` has its
connection dropped (``_drain``). No refusal or dropped connection of these counts as an
upstream error: they are the clients' doing.

Config comes from the environment (``from_env``). The seal script sets ``HLS_EGRESS_POLICY``
(JSON of ``hls.egress.policy.SealPolicy``), ``HLS_PROXY_HOST``, ``HLS_PROXY_TRANSPARENT_PORT`` and
``HLS_PROXY_FWMARK`` (the redirect's target and the mark it exempts), ``HLS_PROXY_PORT`` (href
rewriting + index listen socket) only for a seal with an index (a served container's seal has
none: its artifact runs no pip), and ``HLS_PROXY_OWN_DNS=1`` for a seal that refuses every other
DNS query (with the mark: resolve every name over sockets that carry it). The sealed step's own
environment (``hls.substrate.apptainer.egress.seal_env``) carries the other two:
``HLS_PROXY_STATUS_FILE`` (a host path the proxy keeps ``{"upstream_errors": N,
"mid_stream_errors": M, "oversized_pages": [...], "policy_refused": [...], "fetches_refused": F,
"heartbeat": T}`` in, rewritten whole at start, at every upstream failure, at every page newly
refused for its size, at every host the transparent listener newly refused by the policy and
every ``_HEARTBEAT_SEC``, M of the failures while a file was already streaming to the client, F
the requests refused past ``UPSTREAM_FETCHES_MAX``, T when it was written; the seal points it
outside every container, so a stage reads the count without trusting anything a sealed client
wrote), and, when the site names one, ``HLS_UPSTREAM_PROXY_URL`` (``http://<host>:<port>``: an
HTTP proxy every upstream connection goes through, the index's fetches and the transparent
relay's tunnels alike; the proxy then never reads a proxy from its environment). The index's
upstream is PyPI (``https://pypi.org``). The module imports nothing from ``hls`` at runtime: it
checks that its policy has exactly the shape a ``hls.egress.policy.SealPolicy`` dumps, refusing
anything else before it listens (``_seal_policy``), and evaluates the network policy on that dump
(``_dim_permits``).

Server-side HTTP/1.1 framing (request parsing + response framing on the index
listener) is delegated to ``h11`` -- a small sans-IO state machine that ships as
httpx's own HTTP/1.1 backend, so it adds no dependency of its own. The TLS
ClientHello SNI peek and the transparent blind-relay stay hand-rolled (no library
peeks SNI without terminating TLS).

asyncio + httpx + h11; ASCII-only.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import errno
import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable
from urllib.parse import unquote, urljoin, urlsplit

import h11  # server-side HTTP/1.1 framing (sans-IO state machine); ships with httpx
import httpx

if TYPE_CHECKING:
    # Type-only: used in docstrings / cross-references. The runtime path (from_env)
    # checks the policy's dump and evaluates it itself, so this file imports nothing from hls
    # at runtime (see below).
    from hls.egress.policy import NetworkPolicy, SealPolicy  # noqa: F401

# Hosts the /files/ endpoint is permitted to stream from (defense in depth: the
# token is HMAC-signed by this process, but we still verify the decoded host). Match
# is EXACT against the PRODUCTION pypi-family hosts ONLY -- a broad dotted-suffix
# match (e.g. ".pypi.org") would admit the OPENLY-WRITABLE staging mirrors
# (test.pypi.org / test-files.pythonhosted.org), through which a sealed client could
# publish + then fetch arbitrary code, so those are deliberately excluded.
_FILE_HOSTS = frozenset(
    {
        "files.pythonhosted.org",
        "pypi.org",
        "www.pypi.org",
        "pythonhosted.org",
    }
)

_HTTP_SCHEMES = ("http", "https")

# Match href="..." in a PyPI simple page (PEP 503 anchor list).
_HREF = re.compile(rb'href="([^"]+)"')
# Wheel/sdist artifact extensions worth rewriting (others, if any, pass through).
_ARTIFACT_EXTS = (".whl", ".tar.gz", ".zip", ".tar.bz2", ".egg")

# Request-header bounds are owned by h11 (``max_incomplete_event_size``); no manual cap.
_MAX_REDIRECTS = 5
_STREAM_CHUNK = 65536

# Transparent-intercept tunables. The seal REDIRECTs the sealed
# client's TCP to the transparent listener; the listener identifies the target host
# (TLS ClientHello SNI, or HTTP Host) WITHOUT decrypting, gates it, and blind-relays.
_TLS_HANDSHAKE = 0x16  # first byte of a TLS record carrying a ClientHello
_PEEK_MAX = 16 * 1024  # cap bytes buffered while sniffing (fail closed past this)
_PEEK_TIMEOUT = 10.0  # seconds to wait on a SINGLE read for more sniff bytes
# TOTAL wall-clock budget for the header/peek phase of one connection (the SNI/Host
# sniff or the index request line+headers). Bounds a slow-loris client that dribbles
# bytes just often enough to keep resetting the per-read timeout: the connection is
# dropped if a complete ClientHello / request head isn't received within this deadline
# regardless of per-read progress. Fail-CLOSED -- a dropped connection never egresses.
_CONN_HEADER_DEADLINE = 30.0
# Cap on concurrently-accepted connections per listener entry point. Acquired at the
# top of each accept handler and released in its finally; if the cap is hit the new
# connection is dropped (closed) rather than queued unboundedly, so a flood of
# half-open slow-loris connections can't exhaust coroutines / file descriptors.
_MAX_CONNECTIONS = 256
#: The largest simple page the index reads, in bytes: eight times the largest page on
#: PyPI, pydantic-core's 7.8 MB (10.3 MB rewritten), since a page grows with every release. A
#: larger page is refused with a 502 and recorded in the status file (``_note_oversized``): the
#: package that needed it fails as the submission's, with a reason naming the page.
PAGE_MAX_BYTES = 64 * 2**20
#: Pages refused for their size that the status file keeps, the first ones, each once.
_OVERSIZED_KEPT = 8
#: Characters of a refused page's URL the status file keeps: the name in it is the client's.
_PAGE_URL_CHARS = 512
#: Bytes of the index's pages a page request takes before it asks the upstream, so that the
#: pages being read can never starve each other of room: pydantic-core's page fits in it as
#: read and as rewritten. A larger page grows its share as it is read and rewritten, and is
#: refused with a 503 when the room is full.
_PAGE_START_BYTES = 16 * 2**20
#: The most bytes of simple pages the index holds at once across all its connections, as read
#: from the upstream and as rewritten for a client (``_PageRoom``): a page at the cap fits both
#: as read and as rewritten, about a third larger; 32 pages can be read at once, and about 50
#: rewritten pydantic-core pages wait for clients that do not read. Beyond it there is only the
#: copy of the page being rewritten, one at a time, since a rewrite does not yield.
_PAGES_MAX_BYTES = 512 * 2**20
#: Seconds a page request waits for its ``_PAGE_START_BYTES`` before it asks the upstream; one
#: that finds no room is refused with a 503. Under pip's default read timeout of 15 s, so pip
#: gets the refusal and retries it.
_ROOM_WAIT_SEC = 10.0
#: Seconds a client may take none of a reply before its connection is dropped with what the
#: reply held (``_drain``). A client that reads, however slowly, takes some in far less.
_WRITE_TIMEOUT_SEC = 60.0
#: Seconds within which a client that keeps up takes what was written to it (``_drain``). A
#: client slower than that at any point of a file holds the upstream as long, and a failure of
#: the upstream later in that file is not certainly the upstream's own: an upstream gives up on a
#: stream held for tens of seconds, never for a few.
_PROMPT_SEC = 5.0
#: The least rate, in bytes a second, at which a client the interceptor waited on for more than
#: ``_PROMPT_SEC`` in all must have taken a file, over the time it was waited on, for a failure
#: of the upstream while the file streams to count (``_kept_up``): a client that takes a little
#: within every few seconds still holds the upstream for as long as it reads, while the time an
#: upstream slow on its own account takes is no client's. 256 KiB/s is 20 times what a client taking
#: one ``_STREAM_CHUNK`` per ``_PROMPT_SEC`` reads.
MIN_READ_RATE = 256 * 1024
#: The most fetches from its upstream one interceptor makes, which serves one sealed step:
#: pages, files and each redirect followed. Past it the index answers every request itself with
#: a refusal that counts as no upstream failure (``_take_fetch``), and none of the step's rate
#: limits (429) counts (``_status``): a step that made them all was flooding the index. 30 times
#: the fetches pip makes resolving a set of about a hundred pins from scratch.
UPSTREAM_FETCHES_MAX = 10_000
#: Bytes of the status file, its status and the spaces after it (``write_status_file``). The
#: largest status this interceptor writes is about 41 KiB, which leaves a third of it free:
#: eight refused pages, each URL 512 characters escaped at up to six bytes (about 25 KiB), the
#: counts and the heartbeat at their longest, and the 64 hosts the policy refused, each name
#: kept only when JSON writes it in at most 253 bytes (about 16 KiB, ``_note_policy_refusal``).
STATUS_FILE_BYTES = 64 * 1024
#: Seconds between the times the interceptor writes its status file while it runs, each with
#: the time it wrote it, so a stage can tell an interceptor that stopped from one that counted
#: nothing.
_HEARTBEAT_SEC = 5.0
#: The send buffer set on every index connection, in bytes. Set, it stops the kernel growing the
#: buffer while a reply waits for a client that does not read (to 4 MiB by default); Linux grants
#: twice the figure, at most twice ``net.core.wmem_max``.
_SEND_BUFFER_BYTES = 256 * 2**10
#: The body of a page request refused for want of room.
_BUSY = b"the index is busy"
# Distinct hosts refused by the policy that the status file records, and the longest name kept,
# in the bytes the status file writes it in: the record is the seal probe's evidence, and a
# client's names must not grow the file past ``STATUS_FILE_BYTES``.
_POLICY_REFUSALS_KEPT = 64
_HOST_NAME_MAX = 253
#: The body of a request refused once the step made all its upstream fetches.
_FETCHES_SPENT = b"the index made all the fetches from its upstream one step may make"
# SO_ORIGINAL_DST: the netfilter getsockopt returning a REDIRECT'd connection's
# pre-DNAT destination. Numeric Linux constant for the IPv4 path (SOL_IP).
_SO_ORIGINAL_DST = 80

HostGate = Callable[[str], bool]


def _b64url(s: str) -> str:
    return base64.urlsafe_b64encode(s.encode()).decode()


def _unb64url(token: str) -> str:
    return base64.urlsafe_b64decode(token.encode()).decode()


def _file_host_ok(host: str) -> bool:
    """True iff ``host`` is an EXACT production pypi-family file host.

    Exact-only (no suffix wildcard): the staging mirrors test.pypi.org and
    test-files.pythonhosted.org are world-writable, so admitting any ``*.pypi.org``
    / ``*.pythonhosted.org`` would let a sealed client smuggle arbitrary code past
    the seal. Both are absent from ``_FILE_HOSTS``, so this returns False for them.
    """
    return host in _FILE_HOSTS


def _url_ok(url: str, gate: HostGate) -> bool:
    """A URL is fetchable iff it is http(s) AND its host passes the gate."""
    parts = urlsplit(url)
    if parts.scheme not in _HTTP_SCHEMES:
        return False
    return gate(parts.hostname or "")


def _dist_name_from_url(url: str) -> "str | None":
    """Best-effort distribution name from an artifact URL's FILENAME, or None.

    The name is parsed from the decoded URL's last path segment -- i.e. the file the
    upstream will actually serve -- NOT from any client-supplied label, so the package
    gate on ``/files/`` cannot be spoofed by mismatching the two. A PEP 658
    ``.metadata`` sidecar suffix is stripped first.

    Fails CLOSED (returns None) on any filename without a recognizable
    ``name-version`` structure, so an attacker-chosen filename can never resolve to a
    bare, mis-attributed name the gate would wave through:

      * wheels (PEP 427 ``name-version-pyX-abi-plat.whl``) -> the text before the
        FIRST ``-``; a hyphen-less ``.whl`` stem (no version field) -> None;
      * eggs (``name-version-pyX.Y.egg``) -> require the full ``name-version-pyTag``
        triple, returning the leading name; a stem that doesn't split into three
        ``-``-separated parts -> None;
      * other sdists/archives (``name-version.tar.gz`` etc.) -> everything before
        the LAST ``-`` (the version); a hyphen-less stem -> None.
    """
    name = urlsplit(url).path.rsplit("/", 1)[-1]
    if name.endswith(".metadata"):
        name = name[: -len(".metadata")]
    low = name.lower()
    if low.endswith(".whl"):
        stem = name[: -len(".whl")]
        return stem.split("-", 1)[0] if "-" in stem else None
    if low.endswith(".egg"):
        # PEP 427 egg: name-version-pyX.Y(.egg). Require a clean name-version-pyTag
        # triple; anything we can't split that way fails closed.
        stem = name[: -len(".egg")]
        parts = stem.split("-")
        return parts[0] if len(parts) >= 3 and all(parts[:3]) else None
    for ext in _ARTIFACT_EXTS:
        if ext not in (".whl", ".egg") and low.endswith(ext):
            stem = name[: -len(ext)]
            return stem.rsplit("-", 1)[0] if "-" in stem else None
    return None


# --- transparent-intercept host sniffing (no decryption) ------------------
#
# Used by the seal: the client's TCP is REDIRECT'd here, and we must
# learn the destination host to gate it WITHOUT terminating TLS. For TLS we read the
# SNI from the cleartext ClientHello; for plain HTTP we read the Host header. Every
# parse is read-only and fails closed (returns None) on any malformation.


def _default_original_dst(sock: "socket.socket") -> "tuple[str, int] | None":
    """Return the pre-REDIRECT ``(ip, port)`` of a transparently-intercepted IPv4
    connection via ``SO_ORIGINAL_DST``, or ``None`` if unavailable (caller fails
    closed). The single un-unit-testable syscall; injected so tests can stub it."""
    try:
        raw = sock.getsockopt(socket.SOL_IP, _SO_ORIGINAL_DST, 16)
    except OSError:
        return None
    port = int.from_bytes(raw[2:4], "big")  # sockaddr_in.sin_port (network order)
    ip = ".".join(str(b) for b in raw[4:8])  # sockaddr_in.sin_addr
    if not port or ip == "0.0.0.0":
        return None
    return ip, port


def _host_only(authority: str) -> "str | None":
    """Strip any port (and IPv6 brackets) from an HTTP authority -> bare host."""
    h = urlsplit("//" + authority.strip()).hostname
    return h.lower() if h else None


def _is_internal_ip_literal(host: str) -> bool:
    """True if ``host`` is an IP literal pointing at a non-global address.

    The proxy's upstream connections leave the seal's namespace through pasta onto the host's
    network, so a raw-tunnel target of ``127.0.0.1``, the cloud metadata endpoint
    ``169.254.169.254``, or any private/reserved
    address would reach an internal surface the sandbox's egress seal is meant to exclude.
    That is an SSRF the domain rule alone does not catch (in deny-mode an IP literal matches
    no denied NAME and slips through). Legitimate egress is always to a public name, so
    refuse any IP literal that is not globally routable. A non-IP host returns False --
    the domain rule governs names."""
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


async def _resolve_ipv4(host: str, port: int, *, require_global: bool) -> "str | None":
    """Resolve ``host`` to its first IPv4 address; ``None`` if it does not resolve. When
    ``require_global`` (the default for all task-author domains), refuse the whole name if ANY
    resolved address is internal (loopback / link-local / private / reserved) -- the DNS-name
    SSRF guard. When False (only for the seal's vouched hosts, ``SealPolicy.trusted_internal_hosts``,
    such as a model gateway), the routability check is skipped so a legitimately-internal host resolves.

    Either way the caller connects to the RETURNED IP (not the name) so the vetted/dialed
    address is identical (no TOCTOU re-resolution). IPv4-only, matching the transparent
    interceptor (the seal disables IPv6)."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
    except (socket.gaierror, OSError):
        return None
    addrs = [info[4][0] for info in infos if info[4]]
    if not addrs:
        return None
    if require_global:
        for a in addrs:
            try:
                if not ipaddress.ip_address(a).is_global:
                    return None  # any internal address -> refuse the whole name (fail closed)
            except ValueError:
                return None
    return addrs[0]


async def _resolve_global_ipv4(host: str, port: int) -> "str | None":
    """Resolve ``host`` to a globally-routable IPv4 (the DNS-name SSRF guard); ``None`` if it
    does not resolve or ANY address is internal. Closes the SSRF the IP-literal guard
    (``_is_internal_ip_literal``) cannot catch: a gated NAME whose A record points at
    ``10.0.2.2`` / ``127.0.0.1`` / ``169.254.169.254`` / an RFC1918 host would otherwise be
    dialed through the seal's host-loopback NAT, reaching exactly the internal surface the seal
    excludes. Thin wrapper over ``_resolve_ipv4(require_global=True)``."""
    return await _resolve_ipv4(host, port, require_global=True)


def _install_upstream_mark(mark: int) -> None:
    """Stamp ``SO_MARK`` on every socket this process subsequently opens, set in the
    socket constructor (BEFORE connect) so it lands on the SYN.

    The seal's REDIRECT rule exempts this mark, so the proxy's OWN upstream
    connections -- the transparent relay AND the index's httpx fetches -- egress
    directly instead of being intercepted by itself (a loop). A process-wide install is
    required because httpx/httpcore apply per-request ``socket_options`` only AFTER the
    connection is established -- too late for the SYN that the REDIRECT acts on. Setting
    ``SO_MARK`` needs ``CAP_NET_ADMIN``, which the sealed client lacks, so it cannot
    forge the exemption. Only a proxy given a mark (the seal's) calls this; tests that run
    the proxy without one never mutate the global.

    Fails LOUD if the mark cannot be applied: a silently-dropped ``SO_MARK`` (e.g.
    ``CAP_NET_ADMIN`` missing in the netns) means the proxy's own upstream is REDIRECTed back
    into itself and the seal breaks with no diagnostic, so probe it ONCE here and refuse to
    install the half-working override rather than swallowing the failure on every socket."""
    probe = socket.socket()
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_MARK, mark)
    except OSError as e:
        raise RuntimeError(
            f"SO_MARK={mark} could not be set ({e}); the egress proxy's own upstream would be "
            "REDIRECTed into itself (CAP_NET_ADMIN missing in the seal netns?). Refusing to "
            "start the interceptor rather than run a silently-broken seal."
        ) from e
    finally:
        probe.close()
    base = socket.socket

    class _MarkedSocket(base):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            # The install-time probe already proved SO_MARK works; a stray socket type that
            # rejects it must not crash socket creation, so swallow per-socket (defensive only).
            try:
                self.setsockopt(socket.SOL_SOCKET, socket.SO_MARK, mark)
            except OSError:
                pass

    socket.socket = _MarkedSocket


#: Port and timeout of the marked resolver's queries, and the rounds it asks its nameservers.
_DNS_PORT = 53
_DNS_TIMEOUT_SEC = 3.0
_DNS_ROUNDS = 2
#: Files the marked resolver reads, as libc does.
_RESOLV_CONF = "/etc/resolv.conf"
_HOSTS = "/etc/hosts"
#: libc's lookup, for what needs no query: an address, a port or service name.
_LIBC_GETADDRINFO = socket.getaddrinfo


def _resolver_config(path: str) -> "tuple[list[str], list[str]]":
    """The IPv4 nameservers and the search domains of the resolver configuration at *path*; the
    last ``search`` or ``domain`` line wins, as in libc. IPv6 nameservers are left out: the seal
    disables IPv6."""
    servers: "list[str]" = []
    search: "list[str]" = []
    try:
        with open(path) as f:
            for line in f:
                words = line.split()
                if len(words) >= 2 and words[0] == "nameserver":
                    with contextlib.suppress(ValueError):
                        servers.append(str(ipaddress.IPv4Address(words[1])))
                elif words and words[0] in ("search", "domain"):
                    search = words[1:]
    except OSError:
        pass
    return servers, search


def _hosts_addresses(name: str, path: str) -> "list[str]":
    """The IPv4 addresses the hosts file at *path* gives *name*, in file order."""
    found: "list[str]" = []
    try:
        with open(path) as f:
            for line in f:
                words = line.split("#", 1)[0].split()
                if len(words) >= 2 and name.lower() in (w.lower().rstrip(".") for w in words[1:]):
                    with contextlib.suppress(ValueError):
                        found.append(str(ipaddress.IPv4Address(words[0])))
    except OSError:
        pass
    return found


def _dns_query(name: str, qid: int) -> bytes:
    """A recursive query for the A records of *name*, with id *qid*."""
    labels = [label.encode("idna") for label in name.split(".")]
    if any(not 0 < len(label) < 64 for label in labels):
        raise ValueError(f"{name!r} is not a DNS name")
    question = b"".join(bytes([len(label)]) + label for label in labels) + b"\0"
    return qid.to_bytes(2, "big") + b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00" + question + b"\x00\x01\x00\x01"


def _skip_name(packet: bytes, at: int) -> int:
    """The offset just past the (possibly compressed) name at *at* in *packet*."""
    while True:
        length = packet[at]
        if length & 0xC0 == 0xC0:
            return at + 2
        if length == 0:
            return at + 1
        at += 1 + length


def _dns_answer(packet: bytes, qid: int) -> "tuple[int, bool, list[str]] | None":
    """The response code, the truncation bit and the IPv4 addresses of a response to query
    *qid*; None for a packet that is not one. Records of another type, such as the CNAMEs
    before the addresses, are skipped."""
    if len(packet) < 12 or int.from_bytes(packet[:2], "big") != qid or not packet[2] & 0x80:
        return None
    rcode, truncated = packet[3] & 0x0F, bool(packet[2] & 0x02)
    questions, answers = int.from_bytes(packet[4:6], "big"), int.from_bytes(packet[6:8], "big")
    try:
        at = 12
        for _ in range(questions):
            at = _skip_name(packet, at) + 4
        addresses = []
        for _ in range(answers):
            at = _skip_name(packet, at)
            rtype, rclass = int.from_bytes(packet[at:at + 2], "big"), int.from_bytes(packet[at + 2:at + 4], "big")
            length = int.from_bytes(packet[at + 8:at + 10], "big")
            data = packet[at + 10:at + 10 + length]
            if len(data) != length:
                return None
            if rtype == 1 and rclass == 1 and length == 4:
                addresses.append(socket.inet_ntoa(data))
            at += 10 + length
    except IndexError:
        return None
    return rcode, truncated, addresses


def _ask(server: str, query: bytes, qid: int) -> "tuple[int, list[str]] | None":
    """Send *query* to *server* over UDP, and over TCP when the answer is truncated; the
    response code and the addresses, or None when *server* did not answer in time. The sockets
    come from ``socket.socket``, which in the seal stamps the interceptor's mark on them."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.settimeout(_DNS_TIMEOUT_SEC)
        try:
            udp.connect((server, _DNS_PORT))
            udp.send(query)
            while (answer := _dns_answer(udp.recv(4096), qid)) is None:
                pass
        except OSError:
            return None
    rcode, truncated, addresses = answer
    if not truncated:
        return rcode, addresses
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
        tcp.settimeout(_DNS_TIMEOUT_SEC)
        try:
            tcp.connect((server, _DNS_PORT))
            tcp.sendall(len(query).to_bytes(2, "big") + query)
            size = int.from_bytes(_read_exactly(tcp, 2), "big")
            whole = _dns_answer(_read_exactly(tcp, size), qid)
        except (OSError, EOFError):
            return None
    return None if whole is None else (whole[0], whole[2])


def _read_exactly(sock: "socket.socket", size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError("the nameserver closed the connection")
        data += chunk
    return data


def _resolve_marked(name: str) -> "list[str]":
    """The IPv4 addresses of *name*, from the hosts file or the nameservers of the resolver
    configuration, tried in rounds; a name without a dot tries the search domains first, one
    with a dot itself first, and an absolute one (a trailing dot) only itself. Raise
    socket.gaierror when no nameserver answers (EAI_AGAIN) or the name has no address
    (EAI_NONAME)."""
    absolute, name = name.endswith("."), name.rstrip(".")
    found = _hosts_addresses(name, _HOSTS)
    if found:
        return found
    servers, search = _resolver_config(_RESOLV_CONF)
    if not servers:
        raise socket.gaierror(socket.EAI_AGAIN, f"no IPv4 nameserver in {_RESOLV_CONF}")
    suffixed = [f"{name}.{domain.rstrip('.')}" for domain in search]
    candidates = [name] if absolute else [name, *suffixed] if "." in name else [*suffixed, name]
    for candidate in candidates:
        qid = secrets.randbelow(65536)
        try:
            query = _dns_query(candidate, qid)
        except ValueError:
            continue
        # A server's answer, or its refusal to know the name (NXDOMAIN), settles the candidate.
        reply = next((r for _ in range(_DNS_ROUNDS) for server in servers
                      if (r := _ask(server, query, qid)) is not None and r[0] in (0, 3)), None)
        if reply is None:
            raise socket.gaierror(socket.EAI_AGAIN, f"no nameserver answered for {candidate}")
        if reply[1]:
            return reply[1]
    raise socket.gaierror(socket.EAI_NONAME, f"{name} has no IPv4 address")


def _marked_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """``socket.getaddrinfo`` for a process whose resolver traffic must carry its mark: a name
    is resolved by ``_resolve_marked``, and libc builds the entries from each address without a
    query of its own (``AI_NUMERICHOST``)."""
    name = host.decode() if isinstance(host, bytes) else host
    if name is None:
        return _LIBC_GETADDRINFO(host, port, family, type, proto, flags)
    try:
        ipaddress.ip_address(name)
    except ValueError:
        pass
    else:
        return _LIBC_GETADDRINFO(host, port, family, type, proto, flags | socket.AI_NUMERICHOST)
    if family not in (0, socket.AF_INET):
        raise socket.gaierror(socket.EAI_FAMILY, "the interceptor resolves IPv4 only")
    entries = []
    for address in _resolve_marked(name):
        entries += _LIBC_GETADDRINFO(address, port, socket.AF_INET, type, proto, flags | socket.AI_NUMERICHOST)
    return entries


def _install_marked_resolver() -> None:
    """Resolve every name this process looks up with ``_marked_getaddrinfo``. A seal that
    refuses its payload's DNS lets out only queries carrying the interceptor's mark, and libc's
    resolver opens its sockets where ``_install_upstream_mark`` cannot stamp them; asyncio and
    httpx look ``socket.getaddrinfo`` up at each call, so this reaches every lookup."""
    socket.getaddrinfo = _marked_getaddrinfo


def _parse_sni_extension(ext: bytes) -> "str | None":
    """Extract host_name from a TLS server_name extension body. None if absent."""
    if len(ext) < 5:
        return None
    list_len = int.from_bytes(ext[0:2], "big")
    data = ext[2 : 2 + list_len]
    q = 0
    while q + 3 <= len(data):
        name_type = data[q]
        name_len = int.from_bytes(data[q + 1 : q + 3], "big")
        q += 3
        if q + name_len > len(data):
            return None
        if name_type == 0x00:  # host_name
            try:
                return data[q : q + name_len].decode("ascii").lower()
            except ValueError:
                return None
        q += name_len
    return None


def _parse_client_hello_sni(body: bytes) -> "str | None":
    """Extract the SNI host from a TLS handshake record body (bytes after the 5-byte
    record header). Returns the lowercased host or None (absent/malformed/not a
    ClientHello). Pure read-only walk with bounds checks at every step."""
    try:
        n = len(body)
        if n < 4 or body[0] != 0x01:  # HandshakeType.client_hello
            return None
        p = 4 + 2 + 32  # hdr(4) + client_version(2) + random(32)
        if p + 1 > n:
            return None
        p += 1 + body[p]  # session_id
        if p + 2 > n:
            return None
        p += 2 + int.from_bytes(body[p : p + 2], "big")  # cipher_suites
        if p + 1 > n:
            return None
        p += 1 + body[p]  # compression_methods
        if p + 2 > n:
            return None
        ext_end = min(n, p + 2 + int.from_bytes(body[p : p + 2], "big"))
        p += 2
        while p + 4 <= ext_end:
            ext_type = int.from_bytes(body[p : p + 2], "big")
            ext_len = int.from_bytes(body[p + 2 : p + 4], "big")
            p += 4
            if p + ext_len > n:
                return None
            if ext_type == 0x0000:  # server_name
                return _parse_sni_extension(body[p : p + ext_len])
            p += ext_len
        return None
    except (IndexError, ValueError):
        return None


def _tls_handshake_bytes(buf: bytes) -> "tuple[bytes | None, str]":
    """Reassemble the TLS handshake message from one OR MORE consecutive handshake
    records in ``buf``. Returns ``(handshake_bytes, "ok")`` once the full handshake (per
    its own 3-byte length) is present, ``(None, "need_more")`` if more bytes are needed,
    or ``(None, "bad")`` on a malformed / non-handshake / over-budget stream.

    A ClientHello may legitimately SPAN multiple TLS records (large extensions /
    post-quantum keyshares), so we must not decide off the first record alone -- doing so
    fail-closed-drops an allowed host whose SNI lands in a later record. We concatenate
    record payloads until we have ``handshake_length + 4`` bytes, bounded by _PEEK_MAX."""
    data = b""  # concatenated handshake payload (across records)
    p = 0
    while True:
        if len(buf) - p < 5:
            return None, "need_more"  # next record header not yet buffered
        if buf[p] != _TLS_HANDSHAKE:
            return None, "bad"
        rec_len = int.from_bytes(buf[p + 3 : p + 5], "big")
        if rec_len == 0 or rec_len > _PEEK_MAX:
            return None, "bad"
        if len(buf) - p < 5 + rec_len:
            return None, "need_more"  # this record's body not fully buffered
        data += buf[p + 5 : p + 5 + rec_len]
        p += 5 + rec_len
        if len(data) >= 4:  # the 4-byte handshake header is present
            if data[0] != 0x01:  # HandshakeType.client_hello
                return None, "bad"
            hs_len = int.from_bytes(data[1:4], "big")
            if hs_len + 4 > _PEEK_MAX:
                return None, "bad"
            if len(data) >= hs_len + 4:
                return data[: hs_len + 4], "ok"
        if len(data) >= _PEEK_MAX:
            return None, "bad"


def _sniff_host(buf: bytes) -> "tuple[str | None, str]":
    """Identify the target host from buffered connection-opening bytes.

    Returns ``(host, "ok")`` once a host is found, ``(None, "need_more")`` if more
    bytes are needed to decide, or ``(None, "bad")`` for a protocol we don't gate
    (the caller fails closed on "bad"). TLS -> ClientHello SNI; else HTTP/1.x Host."""
    if not buf:
        return None, "need_more"
    if buf[0] == _TLS_HANDSHAKE:
        hs, status = _tls_handshake_bytes(buf)
        if status != "ok":
            return None, status
        host = _parse_client_hello_sni(hs)
        return (host, "ok") if host else (None, "bad")
    end = buf.find(b"\r\n\r\n")
    if end == -1:
        return (None, "bad") if len(buf) > _PEEK_MAX else (None, "need_more")
    lines = buf[:end].split(b"\r\n")
    if not lines or b" HTTP/" not in lines[0]:
        return None, "bad"
    for line in lines[1:]:
        if line[:5].lower() == b"host:":
            host = _host_only(line[5:].strip().decode("latin-1", "replace"))
            return (host, "ok") if host else (None, "bad")
    return None, "bad"


# --- self-contained policy evaluator (no hls import) ----------------------
#
# The proxy evaluates the policy on its dumped dict with this copy of hls.egress.policy's
# matching rules (PEP 503 normalize + ``*.suffix`` wildcard + allow/deny), and imports
# nothing from hls at runtime. Keep the two in step; the egress unit tests pin both.


def _pep503_normalize(name: str) -> str:
    """PEP 503 normalized distribution name (case-fold + collapse -_. to '-')."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _dict_host_match(pattern: str, host: str) -> bool:
    """Exact host match, or ``*.suffix`` subdomain match (NOT the bare suffix)."""
    if pattern == host:
        return True
    if pattern.startswith("*."):
        suffix = pattern[1:]  # ".evil.com"
        bare = suffix[1:]  # "evil.com"
        return host.endswith(suffix) and host != bare
    return False


def _dim_permits(
    dim: dict | None,
    item: str,
    *,
    wildcard: bool,
    normalize: bool,
) -> bool:
    """Evaluate one dumped AccessList dict (``{"mode","entries"}``); None permits all.

    ``dim`` is the value of a single policy dimension from ``NetworkPolicy.model_dump()``
    -- a flat ``{"mode": "allow"|"deny", "entries": [...]}`` dict, or ``None`` when the
    dimension is absent (e.g. ``packages=None`` on a runtime policy means "no package
    restriction -- permit everything").
    """
    if dim is None:
        return True
    entries = dim.get("entries", [])
    if normalize:
        target = _pep503_normalize(item)
        listed = any(_pep503_normalize(e) == target for e in entries)
    elif wildcard:
        listed = any(_dict_host_match(e, item) for e in entries)
    else:
        listed = item in entries
    return listed if dim.get("mode", "allow") == "allow" else not listed


def _access_list(value: object) -> bool:
    """Whether *value* is an ``AccessList``'s dump: its mode and its entries, nothing else."""
    return (isinstance(value, dict) and set(value) == {"mode", "entries"} and value["mode"] in ("allow", "deny")
            and isinstance(value["entries"], list) and all(isinstance(e, str) for e in value["entries"]))


def _seal_policy(text: str) -> tuple[dict, tuple[str, ...]]:
    """The network policy's dump and the vouched hosts of *text*, the JSON of the seal's
    ``SealPolicy``, which the seal builds through that model only. ValueError for any other
    shape: a key either model does not have, a domain list that is no access list, a vouched
    host that is no name, so the interceptor never enforces what no model checked."""
    data = json.loads(text)
    if not isinstance(data, dict) or set(data) != {"network", "trusted_internal_hosts"}:
        raise ValueError("the seal's policy is not a SealPolicy: it holds exactly network and "
                         "trusted_internal_hosts")
    network, hosts = data["network"], data["trusted_internal_hosts"]
    if (not isinstance(network, dict) or set(network) != {"domains", "packages"}
            or not _access_list(network["domains"])
            or not (network["packages"] is None or _access_list(network["packages"]))):
        raise ValueError("the seal's network policy is not a NetworkPolicy: domains and packages, each an "
                         "access list (packages may be null)")
    if not isinstance(hosts, list) or not all(isinstance(h, str) and h for h in hosts):
        raise ValueError("the seal's vouched hosts are not a list of names")
    return network, tuple(hosts)


class _PageRoom:
    """The bytes of simple pages the index holds at once across all its connections, at most
    ``limit``. Each page request holds its share through a ``_PageHold``, from the first byte it
    reads until its client has taken its reply or its connection is dropped."""

    def __init__(self, limit: int) -> None:
        #: The most bytes held at once.
        self.limit = limit
        #: Bytes held now.
        self.held = 0
        # Requests waiting for room. Every release wakes them all, and each looks again.
        self._waiters: set[asyncio.Future[None]] = set()

    def hold(self) -> "_PageHold":
        """A new share, holding nothing yet."""
        return _PageHold(self)

    async def wait(self, free: int, timeout: float) -> bool:
        """Wait up to *timeout* seconds until *free* bytes are not held; whether they are."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while self.held + free > self.limit:
            waiter = loop.create_future()
            self._waiters.add(waiter)
            try:
                await asyncio.wait_for(waiter, deadline - loop.time())
            except asyncio.TimeoutError:
                return False
            finally:
                self._waiters.discard(waiter)
        return True

    def change(self, by: int) -> bool:
        """Hold *by* more bytes, or fewer when it is negative; False, changing nothing, when the
        more do not fit."""
        if by > 0 and self.held + by > self.limit:
            return False
        self.held += by
        if by < 0:
            for waiter in self._waiters:
                if not waiter.done():
                    waiter.set_result(None)
        return True


class _PageHold:
    """One page request's share of a ``_PageRoom``; leaving it as a context manager gives back
    all it holds."""

    def __init__(self, room: _PageRoom) -> None:
        self._room = room
        #: Bytes this share holds.
        self.bytes = 0

    def resize(self, size: int) -> bool:
        """Hold *size* bytes in all; False, holding what it held, when the growth does not fit."""
        if not self._room.change(size - self.bytes):
            return False
        self.bytes = size
        return True

    def __enter__(self) -> "_PageHold":
        return self

    def __exit__(self, *exc: object) -> None:
        self.resize(0)


class EgressProxy:
    """The interceptor a sealed role's seal runs: the package index and the transparent
    listener over one policy, and the status a stage reads from its file. ``upstream`` is the
    index's upstream, PyPI; tests point it at a dead or mocked one."""

    def __init__(
        self,
        policy: dict,
        *,
        upstream: str = "https://pypi.org",
        host: str,
        port: "int | None" = 8888,
        transparent_port: "int | None" = None,
        upstream_fwmark: "int | None" = None,
        original_dst: "Callable[[socket.socket], tuple[str, int] | None] | None" = None,
        status_file: "str | None" = None,
        upstream_proxy: "str | None" = None,
        own_dns: bool = False,
        trusted_internal_hosts: Iterable[str] = (),
    ) -> None:
        # policy is the flat dict from NetworkPolicy.model_dump():
        #   {"domains": {"mode": ..., "entries": [...]},
        #    "packages": {"mode": ..., "entries": [...]} | None}
        self._policy = policy
        # The hosts the seal vouches for (``SealPolicy.trusted_internal_hosts``), exempt from the
        # transparent intercept's is-global SSRF rejection only (such as a model gateway that
        # resolves to a private IP). Lowercased for case-insensitive
        # comparison against the sniffed host. They must still pass the domain allow-list: the
        # exemption widens nothing but routability.
        self._trusted_internal = {h.lower() for h in trusted_internal_hosts}
        self.upstream = upstream.rstrip("/")
        self.host = host
        # The PyPI pull-through INDEX listen port. None -> do NOT start the index
        # listener at all (transparent-only mode). The index is ONLY needed where a
        # client runs pip/uv (the BUILD/package phase); a served RUNTIME artifact runs
        # no pip, so omitting it removes pure attack surface -- a sealed client sharing
        # the seal's namespace cannot pull arbitrary distributions off a loopback
        # index past the domain allowlist. At least one of (port, transparent_port)
        # must be set or the proxy would bind nothing (from_env / serve_forever guard).
        self.port = port
        # Transparent-intercept listener, the seal's REDIRECT target. None -> only the
        # index runs. ``original_dst`` is injected so tests
        # can stub the one syscall (SO_ORIGINAL_DST) that needs a real REDIRECT.
        self.transparent_port = transparent_port
        self._original_dst = original_dst or _default_original_dst
        # Firewall mark stamped on the proxy's OWN upstream sockets (relay + index
        # fetch). The seal's REDIRECT rule exempts this mark so the proxy's egress
        # is not re-intercepted (loop), while the sealed client -- which lacks
        # CAP_NET_ADMIN and so cannot set SO_MARK -- cannot forge the exemption. None
        # where no seal redirects (tests).
        self._fwmark = upstream_fwmark
        # Per-PROCESS random secret keying the HMAC over every /files/ token. Generated
        # once here, never persisted or transmitted: only THIS process can mint a token
        # that verifies, so a sealed client cannot forge a /files/ URL it did not first
        # obtain from /simple/ (which is the only place tokens are emitted). Both the
        # signer (_rewrite_simple) and the verifier (_resolve_file_url) run in this same
        # process, so a per-process secret is sufficient and rotates each restart.
        self._token_secret = secrets.token_bytes(32)
        # Live count of in-flight connections across BOTH listeners' accept handlers
        # (_on_client and _on_transparent_client). At most _MAX_CONNECTIONS are serviced
        # at once; excess connections are dropped (closed) rather than queued, bounding
        # the fds/coroutines a slow-loris flood can pin open. See _try_acquire_slot.
        self._conn_count = 0
        # The simple pages the index holds at once, across all its connections.
        self._pages = _PageRoom(_PAGES_MAX_BYTES)
        # Cumulative count of UPSTREAM (pythonhosted/pypi) fetch failures -- a connection
        # error, a 5xx or a 429 from the real index. Written to the status file so a stage
        # can tell an infra fault (upstream down) from a contestant fault (bad submission)
        # after a failed `pip install` WITHOUT parsing pip's output. A policy denial (403)
        # or a missing package is NOT an upstream error and does not count.
        self._upstream_errors = 0
        # Those of them that failed while a file was already streaming to the client, counted
        # only when the client had taken all it was sent promptly (``_stream_file``): a client
        # that stalls its read can leave the upstream waiting until it gives up.
        self._mid_stream_errors = 0
        # The upstream's rate limits (429), which count as failures only while the step is
        # under its fetch cap (``_status``).
        self._rate_limited = 0
        # Fetches made from the upstream, at most ``UPSTREAM_FETCHES_MAX``.
        self._fetches = 0
        # Requests answered with a refusal once the fetches were all made.
        self._fetches_refused = 0
        # Bytes the last status written to the status file took; 0 before the first, which
        # writes the whole file (``write_status_file``).
        self._status_bytes = 0
        # The simple pages refused for their size, ``{"page", "read", "cap"}`` each: the first
        # ``_OVERSIZED_KEPT``, each once (see _note_oversized).
        self._oversized: list[dict[str, object]] = []
        # The hosts the transparent listener refused by the policy, first refused first, up to
        # _POLICY_REFUSALS_KEPT, which go into the status file: the seal probe's evidence that
        # the seal refuses a host its policy does not name. A host the policy permits that does
        # not resolve, or cannot be reached, is closed the same way and is not one of them.
        self._policy_refused: list[str] = []
        # Single long-lived upstream client shared by _fetch (the /simple/ page) and
        # _stream_file (wheel/sdist bodies), created lazily on first use (see
        # _ensure_client) and closed exactly once in aclose(). A sealed `pip install`
        # makes hundreds of /simple/ + /files/ requests; one client lets httpx keep its
        # connections to pypi.org / files.pythonhosted.org alive across them.
        self._client: "httpx.AsyncClient | None" = None
        # Optional host path the status is written to (see _write_status): the stage's
        # evidence, in a directory no container mounts, so no sealed client can reach it.
        self._status_file = status_file
        # Optional http://host:port proxy for every upstream connection. Set where the
        # proxy's host has no direct route out; the sealed step's own variables carry it
        # under a name of its own (``seal_env``), since the step wrapper removes every
        # *_proxy variable from a container step.
        self._upstream_proxy = check_upstream_proxy(upstream_proxy)
        # Resolve names over marked sockets (``_install_marked_resolver``), for a seal that
        # refuses every DNS query without the mark, so it needs the mark.
        if own_dns and upstream_fwmark is None:
            raise ValueError("own_dns needs upstream_fwmark: only marked queries pass a seal that refuses DNS")
        self._own_dns = own_dns

    # -- public policy predicates ------------------------------------------

    def domain_permitted(self, host: str) -> bool:
        """Return True iff the domain policy permits ``host``."""
        return _dim_permits(self._policy.get("domains"), host, wildcard=True, normalize=False)

    def package_permitted(self, name: str) -> bool:
        """Return True iff the package policy permits ``name`` (PEP 503 normalized).

        When ``packages`` is ``None`` (absent from the policy dump), all packages
        are permitted.
        """
        return _dim_permits(self._policy.get("packages"), name, wildcard=False, normalize=True)

    # -- internal aliases used by server handlers --------------------------

    def _host_allowed(self, host: str) -> bool:
        return self.domain_permitted(host)

    def _pkg_allowed(self, name: str) -> bool:
        return self.package_permitted(name)

    # -- shared upstream client lifecycle -----------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        """Return the shared upstream ``httpx.AsyncClient``, creating it lazily on the
        first ``_fetch`` / ``_stream_file`` call (the index role -- the only consumer of
        either -- may never run, e.g. transparent-only mode, so nothing is created until
        actually needed). Constructing ``httpx.AsyncClient`` performs no I/O and needs no
        running event loop, so this also works from callers (e.g. tests) that exercise
        ``_fetch`` / ``_stream_file`` directly without going through ``start()`` /
        ``serve_forever()``. No lock is needed: asyncio coroutines are cooperatively
        scheduled and nothing here awaits, so two callers can never interleave between
        the check and the assignment.
        """
        if self._client is None:
            # Each connection the listeners admit has at most one upstream request open, so
            # a pool as large as the cap never makes a request wait for a connection: a
            # client cannot fill the pool to fail its own next request.
            limits = httpx.Limits(max_connections=_MAX_CONNECTIONS, max_keepalive_connections=20)
            if self._upstream_proxy is None:
                self._client = httpx.AsyncClient(follow_redirects=False, timeout=30.0, limits=limits)
            else:
                # The named proxy is the only one, and nothing else of the environment applies.
                self._client = httpx.AsyncClient(follow_redirects=False, timeout=30.0, limits=limits,
                                                 proxy=self._upstream_proxy, trust_env=False)
        return self._client

    async def aclose(self) -> None:
        """Close the shared upstream client, if one was ever created. Idempotent, so it
        is safe to call from the ``serve_forever()`` shutdown path even when no request
        was ever served (client stayed ``None``)."""
        if self._client is not None:
            client, self._client = self._client, None
            await client.aclose()

    # -- upstream-error evidence -------------------------------------------

    def _note_upstream_error(self, *, mid_stream: bool = False) -> None:
        """Count one failed upstream fetch, *mid_stream* when a file was already streaming to
        the client, and record the new counts."""
        self._upstream_errors += 1
        self._mid_stream_errors += mid_stream
        self._write_status()

    def _note_oversized(self, url: str, read: int) -> None:
        """Record that the simple page at *url* was refused for its size once *read* bytes of it
        had come, unless it is recorded already or ``_OVERSIZED_KEPT`` pages are, and record the
        status. It counts as no upstream error: the stage names the page in the reason of the
        package it fails."""
        page = url[:_PAGE_URL_CHARS]
        if len(self._oversized) < _OVERSIZED_KEPT and all(seen["page"] != page for seen in self._oversized):
            self._oversized.append({"page": page, "read": read, "cap": PAGE_MAX_BYTES})
            self._write_status()

    def _note_policy_refusal(self, host: str) -> None:
        """Record that the policy refused *host*, once, while fewer than _POLICY_REFUSALS_KEPT
        are recorded and the name, as the status file writes it, is no longer than a host name
        can be, and rewrite the status file. A status that cannot be written leaves the refusal
        unrecorded, which a reader takes as none."""
        if host in self._policy_refused or len(self._policy_refused) >= _POLICY_REFUSALS_KEPT:
            return
        # Measured as JSON writes it, quotes aside: a client's name can hold characters JSON
        # writes in six bytes each, and 64 such names would not fit the status file.
        if len(json.dumps(host)) - 2 > _HOST_NAME_MAX:
            return
        self._policy_refused.append(host)
        with contextlib.suppress(OSError):
            self._write_status()

    def _note_reply_status(self, status: int) -> None:
        """Count a reply of the upstream with *status* that is its failure: a server error, or a
        rate limit, which pip does not retry (``_status`` decides whether rate limits count)."""
        if status == 429:
            self._rate_limited += 1
            self._write_status()
        elif status >= 500:
            self._note_upstream_error()

    def _take_fetch(self) -> bool:
        """Whether the index may make one more fetch from its upstream, taking it if so: at most
        ``UPSTREAM_FETCHES_MAX`` for the step. A request past them is answered with a refusal,
        counted in the status."""
        if self._fetches >= UPSTREAM_FETCHES_MAX:
            self._fetches_refused += 1
            self._write_status()
            return False
        self._fetches += 1
        if self._fetches == UPSTREAM_FETCHES_MAX and self._rate_limited:
            # The step's rate limits stop counting.
            self._write_status()
        return True

    def _status(self) -> dict[str, object]:
        """The status a stage reads, each key always present (``index_status``): the
        upstream-error counts, the rate limits among them only while the step is under its fetch
        cap, the pages refused for their size, the hosts the policy refused, the requests refused
        past the cap, and when this was written."""
        limited = self._rate_limited if self._fetches < UPSTREAM_FETCHES_MAX else 0
        return {"upstream_errors": self._upstream_errors + limited, "mid_stream_errors": self._mid_stream_errors,
                "oversized_pages": self._oversized, "policy_refused": self._policy_refused,
                "fetches_refused": self._fetches_refused, "heartbeat": time.time()}

    async def _heartbeat(self) -> None:
        """Rewrite the status every ``_HEARTBEAT_SEC`` for as long as the index runs."""
        while True:
            await asyncio.sleep(_HEARTBEAT_SEC)
            self._write_status()

    def _write_status(self) -> None:
        """Rewrite the status file in place with the current status (``_status``,
        ``write_status_file``), over at least as many bytes as the last status took, so no
        trace of it is left. If it cannot be written, remove it, so a reader finds no count
        rather than a stale one that hides failures; then raise."""
        if self._status_file is None:
            return
        try:
            self._status_bytes = write_status_file(self._status_file, self._status(), self._status_bytes)
        except OSError:
            with contextlib.suppress(OSError):
                os.unlink(self._status_file)
            raise

    # -- upstream fetch (overridable in tests) -----------------------------

    async def _fetch(
        self,
        url: str,
        *,
        hold: _PageHold,
        headers: dict[str, str] | None = None,
        gate: HostGate | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> tuple[int, bytes, str]:
        """Fetch a PyPI simple page into *hold*, which holds this page alone: the room taken for
        it beforehand, grown as the page outgrows it. Wheels/sdists are streamed by
        ``_stream_file`` so they never sit whole in RAM.

        The page is read as fast as the upstream sends it, never waiting on a client or for room,
        so no client can hold up the upstream mid-page. A page past ``PAGE_MAX_BYTES`` is
        refused with a 502, and one that outgrows *hold* when the index's pages (``_PageRoom``)
        have no more room with a 503; either is read no further, and neither counts as an
        upstream error.

        Redirects are NOT auto-followed: each ``Location`` is re-checked against ``gate``
        (scheme + host) before being followed, so an allowed host cannot redirect us to a
        denied one (SSRF seal). An upstream failure (``_upstream_failure``), a 5xx or a 429
        increments the upstream-error counter (see ``_status``). The index's upstream sockets are
        mark-stamped process-wide (``_install_upstream_mark``) so the seal's REDIRECT
        exempts them.

        Reuses the proxy's single long-lived client (see ``_ensure_client``) so repeated
        calls -- a sealed ``pip install`` makes hundreds -- share keep-alive connections
        to pypi.org / files.pythonhosted.org instead of paying a fresh TCP+TLS handshake
        each time. ``transport`` is test-only (a ``httpx.MockTransport``): a transport
        can only be set at client construction, so it gets a private, short-lived
        client, closed at the end of this call.
        """
        owns_client = transport is not None
        client = (
            httpx.AsyncClient(follow_redirects=False, timeout=30.0, transport=transport)
            if owns_client
            else self._ensure_client()
        )
        try:
            current = url
            for _ in range(_MAX_REDIRECTS + 1):
                if gate is not None and not _url_ok(current, gate):
                    return 403, b"host not allowed", "text/plain"
                if not self._take_fetch():
                    return 403, _FETCHES_SPENT, "text/plain"
                try:
                    async with client.stream("GET", current, headers=headers) as resp:
                        loc = resp.headers.get("location") if _is_redirect(resp) else None
                        if loc:
                            current = urljoin(current, loc)
                            continue
                        chunks: list[bytes] = []
                        size = 0
                        refused: "tuple[int, bytes, str] | None" = None
                        async for chunk in resp.aiter_bytes():
                            size += len(chunk)
                            if size > PAGE_MAX_BYTES:
                                self._note_oversized(url, size)
                                refused = 502, b"the page is larger than the index serves", "text/plain"
                                break
                            if size > hold.bytes and not hold.resize(size):
                                refused = 503, _BUSY, "text/plain"
                                break
                            chunks.append(chunk)
                        self._note_reply_status(resp.status_code)
                        return refused or (resp.status_code, b"".join(chunks), _ctype(resp))
                except httpx.HTTPError as e:
                    # The page is read whole before the client gets any of it, so the client
                    # cannot hold it up.
                    if _upstream_failure(e):
                        self._note_upstream_error()
                    return 502, b"upstream fetch failed", "text/plain"
            return 502, b"too many redirects", "text/plain"
        finally:
            if owns_client:
                await client.aclose()

    # -- PyPI index role ---------------------------------------------------

    async def handle_simple(self, name: str, hold: _PageHold) -> tuple[int, bytes]:
        """Serve ``/simple/<name>/`` filtered by the package rule, holding the page in *hold*.

        Denied -> (403, message) with NO upstream call. Allowed -> take ``_PAGE_START_BYTES``
        into *hold*, waiting up to ``_ROOM_WAIT_SEC`` for room among the index's pages, refused
        with a 503 and no upstream call when none comes; then fetch the real simple page (asking
        for HTML so uv/pip parse reliably, avoiding PEP 691 JSON ambiguity) and rewrite artifact
        hrefs to the proxy's /files/ endpoint, absolutizing relative hrefs against the simple
        page URL. *hold* then holds the reply alone, a rewritten page refused with a 503 when it
        does not fit.
        """
        if not self._pkg_allowed(name):
            return 403, f"package not allowed: {name}".encode()
        # No await comes between finding the room and taking it.
        if not await self._pages.wait(_PAGE_START_BYTES, _ROOM_WAIT_SEC) or not hold.resize(_PAGE_START_BYTES):
            return 503, _BUSY
        page_url = f"{self.upstream}/simple/{name}/"
        # Re-gate the upstream fetch's redirects (SSRF seal): a Location may only point at
        # the configured upstream host or a pythonhosted/pypi file host -- not, say, a
        # 302 to 169.254.169.254.
        upstream_host = urlsplit(self.upstream).hostname or ""
        status, body, _ct = await self._fetch(
            page_url,
            hold=hold,
            headers={"Accept": "text/html"},
            gate=lambda h: h == upstream_host or _file_host_ok(h),
        )
        if status == 200:
            body = self._rewrite_simple(body, page_url)
        # The reply alone stays held; what was taken beyond it goes back.
        if not hold.resize(len(body)):
            return 503, _BUSY
        return status, body

    def _sign_token(self, url: str) -> str:
        """Mint the opaque ``/files/`` token segment for an absolutized artifact ``url``.

        The token is ``base64url(url) + "." + base64url(hmac_sha256(secret, url))`` --
        the URL plus a signature only THIS process can produce. Both halves use the
        urlsafe alphabet and carry no ``/``, so the token survives as a single path
        segment; ``=`` padding is tolerated on the read side (pip percent-encodes it and
        ``_resolve_file_url`` ``unquote``s before decoding). ``.`` separates the two
        base64 fields (it is not in the urlsafe alphabet, so it is unambiguous).
        """
        sig = hmac.new(self._token_secret, url.encode(), "sha256").digest()
        return _b64url(url) + "." + base64.urlsafe_b64encode(sig).decode()

    def _verify_token(self, tok: str) -> "str | None":
        """Decode + signature-check a ``/files/`` token segment; return the URL or None.

        Splits the ``<b64url(url)>.<b64url(hmac)>`` token, base64-decodes both halves,
        recomputes the HMAC over the decoded URL with the per-process secret, and
        constant-time-compares it (``hmac.compare_digest``). Any token whose signature
        doesn't verify -- forged, unsigned, or signed under a different secret -- yields
        None, so only proxy-emitted (from ``/simple/``) URLs are fetchable.
        """
        if "." not in tok:
            return None
        body_b64, sig_b64 = tok.rsplit(".", 1)
        try:
            url = _unb64url(body_b64)
            sig = base64.urlsafe_b64decode(sig_b64.encode())
        except Exception:
            return None
        expected = hmac.new(self._token_secret, url.encode(), "sha256").digest()
        if not hmac.compare_digest(sig, expected):
            return None
        return url

    def _rewrite_simple(self, body: bytes, page_url: str) -> bytes:
        # Root-relative /files/ href so the client resolves it against the index URL it
        # used (the seal's http://127.0.0.1:<port>/simple/), without the proxy needing to
        # know its own address.
        base = b"/files/"

        def repl(m: "re.Match[bytes]") -> bytes:
            url = m.group(1).decode()
            frag = ""
            core = url
            if "#" in url:
                core, frag = url.split("#", 1)
                frag = "#" + frag
            # absolutize relative hrefs against the simple page URL (mirrors emit
            # relative links) before classifying / encoding.
            core = urljoin(page_url, core)
            # only rewrite genuine artifact links; leave anything else intact
            path = urlsplit(core).path.lower()
            if not path.endswith(_ARTIFACT_EXTS):
                return m.group(0)
            token = self._sign_token(core)
            # Keep the real filename as the trailing path segment so pip/uv can
            # parse the distribution name+version from it -- an opaque token alone
            # yields "no matching distribution". The proxy reads only the token;
            # the filename is informational.
            filename = urlsplit(core).path.rsplit("/", 1)[-1] or "pkg"
            return b'href="' + base + token.encode() + b"/" + filename.encode() + frag.encode() + b'"'

        return _HREF.sub(repl, body)

    def _resolve_file_url(self, token: str) -> "tuple[str | None, tuple[int, bytes] | None]":
        """Decode + GATE a ``/files/`` token. Returns ``(url, None)`` to fetch, or
        ``(None, (status, message))`` to reply as an error.

        Gates: a valid per-process HMAC signature on the token (only proxy-emitted URLs
        verify -- a forged/unsigned token 400s); scheme http(s); host an EXACT
        production pythonhosted/pypi file host; and -- when a package policy is active --
        the distribution name parsed from the DECODED url (NOT the spoofable
        client-supplied filename segment), failing CLOSED if unrecognizable. A PEP 658
        ``.metadata`` suffix on the client filename appends ``.metadata`` to the upstream
        url so the dist-info sidecar's hash verifies.
        """
        # "<token>/<filename>": the signed token has no "/". pip percent-encodes the
        # base64 "=" padding (-> %3D), so url-decode the token before verifying, or
        # padded tokens 400.
        seg = token.split("/", 1)
        tok, filename = unquote(seg[0]), (seg[1] if len(seg) > 1 else "")
        url = self._verify_token(tok)
        if url is None:
            return None, (400, b"bad token")
        parts = urlsplit(url)
        if parts.scheme not in _HTTP_SCHEMES:
            return None, (403, b"file scheme not allowed")
        if not _file_host_ok(parts.hostname or ""):
            return None, (403, b"file host not allowed")
        if self._policy.get("packages") is not None:
            dist = _dist_name_from_url(url)
            if dist is None or not self._pkg_allowed(dist):
                return None, (403, b"package not allowed")
        if filename.endswith(".metadata"):
            url = url + ".metadata"
        return url, None

    async def _stream_file(self, url: str, writer: asyncio.StreamWriter, conn: "h11.Connection") -> None:
        """Stream an upstream wheel/sdist to the client CHUNK BY CHUNK via h11 chunked
        transfer-encoding -- only one ``_STREAM_CHUNK`` is ever resident (a multi-hundred-MB
        wheel never sits whole in RAM), and NO Content-Length is sent: httpx yields DECODED
        bytes (Content-Encoding stripped), so chunked framing lets pip de-chunk to the exact
        wheel and its ``#sha256`` verifies. Redirects re-gated to the pythonhosted/pypi
        file-host set (SSRF seal); an upstream failure (``_upstream_failure``), a 5xx or a 429
        bumps the upstream-error counter.
        Once the head is sent a mid-stream error can only truncate the body (the client sees
        a short read); before the head, errors reply normally. A failure once the head is sent
        counts, as mid-stream, only when the client took every write so far within
        ``_PROMPT_SEC`` (``_drain``) and kept up over the time it was waited on (``_kept_up``): a
        client that held the stream longer could have made the upstream give up on it. A client
        that takes none of the stream for
        ``_WRITE_TIMEOUT_SEC`` has its connection dropped (``_drain``), which closes the upstream
        response and counts nothing.

        Reuses the proxy's single long-lived client (see ``_ensure_client``), so downloads
        share keep-alive connections to files.pythonhosted.org."""
        head_sent = False
        # Whether every drain of the client took ``_PROMPT_SEC`` or less, how much of the file it
        # took, and how long the interceptor waited on it in all (``_kept_up``): only a client
        # that kept up makes a failure of the upstream once the head is sent not of its making.
        prompt, delivered, waited = True, 0, 0.0
        client = self._ensure_client()
        current = url
        for _ in range(_MAX_REDIRECTS + 1):
            if not _url_ok(current, _file_host_ok):
                await self._respond(writer, conn, 403, b"host not allowed")
                return
            if not self._take_fetch():
                await self._respond(writer, conn, 403, _FETCHES_SPENT)
                return
            try:
                async with client.stream("GET", current) as resp:
                    if _is_redirect(resp):
                        loc = resp.headers.get("location")
                        if not loc:
                            await self._respond(writer, conn, resp.status_code, b"")
                            return
                        current = urljoin(current, loc)
                        continue
                    self._note_reply_status(resp.status_code)
                    self._write(
                        writer,
                        conn,
                        h11.Response(
                            status_code=resp.status_code,
                            headers=[
                                ("content-type", _ctype(resp) or "application/octet-stream"),
                                ("transfer-encoding", "chunked"),
                                ("connection", "close"),
                            ],
                        ),
                    )
                    head_sent = True
                    waited = await _drain(writer)
                    prompt = waited <= _PROMPT_SEC
                    async for chunk in resp.aiter_bytes(_STREAM_CHUNK):
                        self._write(writer, conn, h11.Data(data=chunk))
                        took = await _drain(writer)
                        prompt, waited, delivered = prompt and took <= _PROMPT_SEC, waited + took, delivered + len(chunk)
                    self._write(writer, conn, h11.EndOfMessage())
                    await _drain(writer)
                    return
            except httpx.HTTPError as e:
                if _upstream_failure(e) and (not head_sent or prompt and _kept_up(delivered, waited)):
                    self._note_upstream_error(mid_stream=head_sent)
                if not head_sent:
                    await self._respond(writer, conn, 502, b"upstream fetch failed")
                return  # head already sent -> truncated body, just close
        if not head_sent:
            await self._respond(writer, conn, 502, b"too many redirects")

    # -- server / dispatch -------------------------------------------------

    async def start(self) -> asyncio.AbstractServer:
        """Listen for the index's clients: plain HTTP on the seal's loopback."""
        return await asyncio.start_server(self._on_client, self.host, self.port)

    async def start_transparent(self) -> asyncio.AbstractServer:
        """Listen for REDIRECT'd connections from the seal. NEVER TLS-wrapped
        -- it sniffs the cleartext ClientHello SNI / HTTP Host to gate the destination."""
        return await asyncio.start_server(
            self._on_transparent_client, self.host, self.transparent_port)

    async def serve_forever(self) -> None:
        """Serve whichever listeners are configured -- any of the three combos:

          * index-only (``transparent_port is None``): the PyPI pull-through index only,
            which no seal starts, since the seal redirects every connection to the
            transparent listener.
          * transparent-only (``port is None``): the transparent-intercept listener only,
            with NO pip index. A served container's seal: a served artifact runs no pip, so the
            index is pure attack surface (a sealed client could pull arbitrary distros off
            a loopback index past the domain allowlist) -- omit it entirely.
          * both: a build or package seal, where a sealed client runs pip/uv through the
            index AND its other egress is domain-filtered transparently.

        The fwmark install happens regardless (the transparent relay's own upstream
        sockets must be exempted from the REDIRECT). ``start()`` is called ONLY when an
        index port is set. The shared upstream client (see ``_ensure_client``) is closed
        exactly once via ``aclose()`` when this coroutine unwinds -- normally that is
        only on cancellation/shutdown, since the ``gather`` below otherwise runs
        forever."""
        if self._fwmark is not None:
            # Mark our own upstream sockets so the seal's REDIRECT exempts them; must
            # happen before any listener/upstream socket is created.
            _install_upstream_mark(self._fwmark)
        if self._own_dns:
            _install_marked_resolver()
        # A count of zero exists before any listener binds, so a reader that finds no
        # status file knows the proxy never started.
        self._write_status()
        servers = []
        if self.port is not None:
            servers.append(await self.start())
        if self.transparent_port is not None:
            servers.append(await self.start_transparent())
        # Guarded by from_env / the constructor invariant: at least one listener exists.
        async with contextlib.AsyncExitStack() as stack:
            # Registered before the servers so it unwinds LAST (AsyncExitStack is
            # LIFO): listeners stop accepting first, the shared upstream client closes
            # after.
            stack.push_async_callback(self.aclose)
            for server in servers:
                await stack.enter_async_context(server)
            beating = [self._heartbeat()] if self._status_file is not None else []
            await asyncio.gather(*(s.serve_forever() for s in servers), *beating)

    # -- transparent-intercept role ----------------------------------------

    async def _peek_host(self, reader: asyncio.StreamReader) -> "tuple[bytes, str | None]":
        """Buffer connection-opening bytes until the target host is identifiable.

        Returns ``(buffered_bytes, host)`` -- ``host`` is ``None`` on timeout, EOF,
        an over-budget read, or an ungateable protocol (caller fails closed). The
        buffered bytes are replayed to the upstream so nothing is lost.

        A TOTAL ``_CONN_HEADER_DEADLINE`` wall-clock budget bounds the whole sniff: a
        slow-loris client dribbling bytes to keep resetting the per-read timeout is
        dropped once the deadline passes, regardless of per-read progress."""
        buf = b""
        deadline = asyncio.get_event_loop().time() + _CONN_HEADER_DEADLINE
        try:
            while len(buf) < _PEEK_MAX:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return buf, None  # total header budget exhausted
                chunk = await asyncio.wait_for(reader.read(_STREAM_CHUNK), min(_PEEK_TIMEOUT, remaining))
                if not chunk:
                    return buf, None
                buf += chunk
                host, status = _sniff_host(buf)
                if status == "ok":
                    return buf, host
                if status == "bad":
                    return buf, None
        except (asyncio.TimeoutError, OSError):
            return buf, None
        return buf, None

    def _egress_host_allowed(self, host: str) -> bool:
        """Gate a host for the transparent raw-egress intercept. The domain rule decides
        reachability, AND -- when a package policy is active -- the package index hosts
        (pypi/pythonhosted) are refused as raw tunnels: packages must flow through the local
        pull-through index (where the name filter lives), so they cannot be fetched directly
        past it.

        The transparent intercept is the raw-tunnel path that reaches this guard (the index
        listener 405s CONNECT/absolute-form), so refusing the package hosts here is what forces
        ``pip --index-url pypi.org`` (or a direct hit on ``files.pythonhosted.org``) through the
        name-filtering index.

        An IP literal targeting a non-global address is refused outright (SSRF guard): the
        proxy's upstream leaves the seal's namespace through pasta onto the host's network, so
        tunnelling to ``127.0.0.1`` / ``169.254.169.254`` / a private
        range would reach an internal surface past the sandbox's egress seal (see
        ``_is_internal_ip_literal``). A substrate-vouched trusted-internal host, such as a model
        gateway on a private address, may itself be an internal IP literal: the vouch is
        consulted before this reject so the literal form is admitted like the name form (it
        must still pass the allow-list below)."""
        if _is_internal_ip_literal(host) and host.lower() not in self._trusted_internal:
            return False
        if self._policy.get("packages") is not None and _file_host_ok(host):
            return False
        return self._host_allowed(host)

    async def _on_transparent_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if not self._try_acquire_slot():
            # Connection cap hit: drop (close) rather than queue unboundedly. Fail
            # CLOSED -- a dropped intercept connection egresses nothing.
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
            return
        up_writer: "asyncio.StreamWriter | None" = None
        try:
            sock = writer.get_extra_info("socket")
            orig = self._original_dst(sock) if sock is not None else None
            buf, host = await self._peek_host(reader)
            if not host or orig is None:
                return  # fail closed: ungateable protocol, or no original dst
            if not self._egress_host_allowed(host):
                self._note_policy_refusal(host)
                return  # fail closed: denied
            _ip, port = orig
            # Resolve the GATED NAME (not the client's chosen IP) and refuse if it points at an
            # internal address -- a static internal A record is an SSRF the IP-literal guard
            # cannot catch. Connecting to this vetted IP (rather than re-resolving the name on
            # connect) makes the checked and dialed addresses identical, and still defeats a
            # spoofed SNI/Host (we resolve the host the client claimed, never its chosen IP).
            # The socket is mark-stamped (see _install_upstream_mark) so the seal's REDIRECT
            # exempts it rather than looping it back here.
            # A substrate-vouched trusted-internal host may resolve to a
            # private IP -- skip the is-global rejection for it (it already passed the
            # allow-list above), but PIN the result: the substrate also vouched the host's
            # resolved IP literal(s), so an internal destination is admitted ONLY if it is one
            # of those pins. A DNS rebind of the vouched name to a DIFFERENT internal address
            # (e.g. 169.254.169.254) resolves to an unpinned internal IP and is refused; a
            # globally-routable result needs no pin. Every non-vouched name must resolve
            # globally (SSRF guard).
            if host.lower() in self._trusted_internal:
                dest = await _resolve_ipv4(host, port, require_global=False)
                if dest is not None and _is_internal_ip_literal(dest) and dest not in self._trusted_internal:
                    return  # internal, but not a substrate-vouched pin -> fail closed (rebind)
            else:
                dest = await _resolve_global_ipv4(host, port)
            if dest is None:
                return  # unresolvable, or resolves to an internal address -> fail closed
            try:
                up_reader, up_writer = await self._open_upstream(dest, port)
            except OSError:
                return
            up_writer.write(buf)  # replay the peeked ClientHello / request head
            await up_writer.drain()
            await self._relay(reader, writer, up_reader, up_writer)
        except Exception:  # noqa: BLE001 -- a sealed client must never crash the listener
            pass
        finally:
            self._release_slot()
            if up_writer is not None:
                try:
                    up_writer.close()
                except Exception:  # noqa: BLE001
                    pass
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def _open_upstream(self, dest: str, port: int) -> "tuple[asyncio.StreamReader, asyncio.StreamWriter]":
        """A connection to the vetted address *dest*:*port*: direct, or through the upstream
        proxy as an HTTP CONNECT to that address, so the proxy dials what this process vetted
        and never resolves the name again. Raise OSError when the proxy refuses or does not
        answer within ``_CONNECT_REPLY_SEC``."""
        if self._upstream_proxy is None:
            return await asyncio.open_connection(dest, port)
        parts = urlsplit(self._upstream_proxy)
        reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
        target = f"{dest}:{port}"
        try:
            writer.write(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode("ascii"))
            await writer.drain()
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), _CONNECT_REPLY_SEC)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError) as e:
            writer.close()
            raise OSError(f"the upstream proxy gave no CONNECT reply for {target}") from e
        except BaseException:
            writer.close()
            raise
        status = head.split(b"\r\n", 1)[0].split()
        if len(status) < 2 or status[1] != b"200":
            writer.close()
            raise OSError(f"the upstream proxy refused CONNECT {target}")
        return reader, writer

    async def _recv_request(
        self, reader: asyncio.StreamReader, conn: "h11.Connection"
    ) -> "h11.Request | None":
        """Drive h11 until a full request is received; return the ``h11.Request`` (after
        its EndOfMessage) or None on EOF / timeout / protocol or oversize-header error
        (the caller closes). h11 owns request-line parsing, header bounds, and framing.

        A TOTAL ``_CONN_HEADER_DEADLINE`` wall-clock budget bounds the whole request
        head: a slow-loris client dribbling bytes to keep resetting the per-read timeout
        is dropped once the deadline passes, regardless of per-read progress."""
        request: "h11.Request | None" = None
        deadline = asyncio.get_event_loop().time() + _CONN_HEADER_DEADLINE
        while True:
            try:
                event = conn.next_event()
            except h11.RemoteProtocolError:
                return None
            if event is h11.NEED_DATA:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    return None  # total header budget exhausted
                try:
                    data = await asyncio.wait_for(reader.read(_STREAM_CHUNK), min(_PEEK_TIMEOUT, remaining))
                except (asyncio.TimeoutError, OSError):
                    return None
                conn.receive_data(data)  # b"" signals EOF to h11
                continue
            if isinstance(event, h11.Request):
                request = event
            elif isinstance(event, h11.EndOfMessage):
                return request
            elif isinstance(event, h11.Data):
                continue  # bodyless GET expected; ignore any body
            else:
                return None  # ConnectionClosed / PAUSED

    @staticmethod
    def _write(writer: asyncio.StreamWriter, conn: "h11.Connection", event: object) -> None:
        """Serialize one h11 event and write it (h11 returns None for no-output events)."""
        data = conn.send(event)
        if data:
            writer.write(data)

    def _try_acquire_slot(self) -> bool:
        """Take a connection slot WITHOUT blocking; False if the cap is already hit.

        A plain integer counter (not ``asyncio.Semaphore.acquire``, which queues
        unboundedly when exhausted): asyncio accept handlers run on a single thread with
        no await between the check and the increment, so the counter needs no lock. When
        the count reaches _MAX_CONNECTIONS the new connection is dropped (the caller
        closes it) rather than queued, bounding the fds/coroutines a slow-loris flood
        can pin open. Paired with ``_release_slot`` in each handler's finally."""
        if self._conn_count >= _MAX_CONNECTIONS:
            return False
        self._conn_count += 1
        return True

    def _release_slot(self) -> None:
        """Release a connection slot taken by ``_try_acquire_slot``."""
        if self._conn_count > 0:
            self._conn_count -= 1

    async def _on_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not self._try_acquire_slot():
            # Connection cap hit: drop (close) rather than queue unboundedly. Fail
            # CLOSED -- a dropped index connection serves nothing.
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass
            return
        conn = h11.Connection(h11.SERVER)
        try:
            writer.get_extra_info("socket").setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, _SEND_BUFFER_BYTES)
            request = await self._recv_request(reader, conn)
            if request is None:
                return
            if request.method != b"GET":
                # Index-only listener: a PyPI pull-through index, NOT a forward proxy. The
                # domain rule is the transparent listener's, so CONNECT/absolute-form never
                # land here.
                await self._respond(writer, conn, 405, b"method not allowed")
                return
            await self._do_get(request.target.decode("latin-1"), writer, conn)
        except Exception:  # noqa: BLE001 -- one client must never crash the listener
            try:
                if conn.our_state is h11.SEND_RESPONSE:  # nothing written yet -> can reply
                    await self._respond(writer, conn, 502, b"proxy error")
            except Exception:  # noqa: BLE001
                pass
        finally:
            self._release_slot()
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass

    async def _do_get(self, target: str, writer: asyncio.StreamWriter, conn: "h11.Connection") -> None:
        if target.startswith("/simple/"):
            name = target[len("/simple/") :].strip("/")
            with self._pages.hold() as hold:
                status, body = await self.handle_simple(name, hold)
                await self._respond(writer, conn, status, body, ctype="text/html")
        elif target.startswith("/files/"):
            url, err = self._resolve_file_url(target[len("/files/") :])
            if err is not None:
                await self._respond(writer, conn, err[0], err[1])
            else:
                await self._stream_file(url, writer, conn)  # streamed, not buffered
        else:
            await self._respond(writer, conn, 404, b"not found")

    async def _relay(
        self,
        c_reader: asyncio.StreamReader,
        c_writer: asyncio.StreamWriter,
        u_reader: asyncio.StreamReader,
        u_writer: asyncio.StreamWriter,
    ) -> None:
        async def pipe(src: asyncio.StreamReader, dst: asyncio.StreamWriter) -> None:
            try:
                while True:
                    chunk = await src.read(_STREAM_CHUNK)
                    if not chunk:
                        break
                    dst.write(chunk)
                    await dst.drain()
            except Exception:
                pass
            finally:
                try:
                    dst.close()
                except Exception:
                    pass

        await asyncio.gather(
            pipe(c_reader, u_writer),
            pipe(u_reader, c_writer),
        )

    async def _respond(
        self,
        writer: asyncio.StreamWriter,
        conn: "h11.Connection",
        status: int,
        body: bytes,
        *,
        ctype: str = "text/plain",
    ) -> None:
        """Send a buffered HTTP/1.1 response via h11 (it owns Content-Length framing), a
        ``_STREAM_CHUNK`` at a time, so a client that takes none of it for
        ``_WRITE_TIMEOUT_SEC`` is dropped (``_drain``). ``Connection: close`` -- the index
        serves one request per connection."""
        out = [
            ("content-type", ctype),
            ("content-length", str(len(body))),
            ("connection", "close"),
        ]
        self._write(writer, conn, h11.Response(status_code=status, headers=out))
        for at in range(0, len(body), _STREAM_CHUNK):
            self._write(writer, conn, h11.Data(data=body[at:at + _STREAM_CHUNK]))
            await _drain(writer)
        self._write(writer, conn, h11.EndOfMessage())
        await _drain(writer)

    # -- construction from env --------------------------------------------

    @classmethod
    def from_env(cls) -> "EgressProxy":
        """Construct from environment variables, as the seal starts the proxy
        (``python -m hls.egress.proxy``).

        ``HLS_EGRESS_POLICY`` must be the JSON of a ``SealPolicy``, as the seal writes it: the
        network policy and the hosts the seal vouches for. Anything of another shape raises
        ValueError before any listener binds (``_seal_policy``); the network policy is then
        evaluated on its dump by ``_dim_permits``.

        ``HLS_PROXY_PORT`` is OPTIONAL: a missing or empty value disables the PyPI
        pull-through index (transparent-only mode -- a served container's seal, whose
        artifact runs no pip). At least one of ``HLS_PROXY_PORT`` (index) or
        ``HLS_PROXY_TRANSPARENT_PORT`` (transparent intercept) MUST be set, else the proxy
        would bind nothing; that case errors clearly here.
        """
        network, vouched = _seal_policy(os.environ["HLS_EGRESS_POLICY"])
        iport = os.environ.get("HLS_PROXY_PORT")
        tport = os.environ.get("HLS_PROXY_TRANSPARENT_PORT")
        fwmark = os.environ.get("HLS_PROXY_FWMARK")
        if not iport and not tport:
            raise ValueError(
                "EgressProxy.from_env: at least one of HLS_PROXY_PORT (pull-through index) "
                "or HLS_PROXY_TRANSPARENT_PORT (transparent intercept) must be set; "
                "with neither, the proxy would bind no listener."
            )
        return cls(
            network,
            trusted_internal_hosts=vouched,
            status_file=os.environ.get("HLS_PROXY_STATUS_FILE") or None,
            upstream_proxy=os.environ.get("HLS_UPSTREAM_PROXY_URL") or None,
            own_dns=os.environ.get("HLS_PROXY_OWN_DNS") == "1",
            host=os.environ["HLS_PROXY_HOST"],
            port=int(iport) if iport else None,
            transparent_port=int(tport) if tport else None,
            upstream_fwmark=int(fwmark, 0) if fwmark else None,
        )


#: Seconds the upstream proxy has to answer a CONNECT.
_CONNECT_REPLY_SEC = 30.0
# Operating-system errors that say this process ran out of something a client can use up by
# opening connections, not that the upstream failed.
_OWN_LIMIT_ERRNOS = frozenset({errno.EMFILE, errno.ENFILE})


def _upstream_failure(error: httpx.HTTPError) -> bool:
    """Whether *error* is a failure of the upstream itself, which no client of the index can
    cause by how it asks: a transport failure while the proxy waited on the upstream or read
    from it, such as a refused or timed-out connection, a tunnel the upstream proxy refused, a
    reset, silence, or an end before the length the response declared, or a body whose encoding
    does not decode. Never a full connection pool, nor a connection this process could not open
    for lack of file descriptors: a client's own requests use those up. Nor an error closing the
    upstream response of a client dropped for taking none of its reply (``_ClientStalled``).
    Whether a client that held a streaming file could have made the upstream fail is the
    stream's to judge (``_stream_file``)."""
    if isinstance(error, httpx.PoolTimeout) or not isinstance(error, (httpx.TransportError, httpx.DecodingError)):
        return False
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, OSError) and cause.errno in _OWN_LIMIT_ERRNOS or isinstance(cause, _ClientStalled):
            return False
        cause = cause.__cause__ or cause.__context__
    return True


def write_status_file(path: str, status: dict[str, object], last: int) -> int:
    """Write *status* as JSON into the file at *path* in place, followed by spaces, which JSON
    allows: over the whole ``STATUS_FILE_BYTES`` when *last* is 0, the first write, which makes
    the file that size before the step's payload starts; else over the bytes the status takes
    or the *last* the previous one took, whichever is more, all within what the file holds. So
    no write after the first needs a new block of a disk the step may have filled, renames
    nothing, and leaves nothing of an earlier status. Return the bytes this status took. Raise
    ValueError for a status larger than the file, which ``STATUS_FILE_BYTES`` rules out, and
    OSError when the write fails or falls short."""
    data = json.dumps(status).encode()
    if len(data) > STATUS_FILE_BYTES:
        raise ValueError(f"the status takes {len(data)} bytes, more than the {STATUS_FILE_BYTES} of its file")
    span = STATUS_FILE_BYTES if last == 0 else max(len(data), last)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        written = os.pwrite(fd, data.ljust(span, b" "), 0)
    finally:
        os.close(fd)
    if written != span:
        raise OSError(errno.EIO, f"the status file took {written} of {span} bytes", path)
    return len(data)


def _kept_up(delivered: int, waited: float) -> bool:
    """Whether a client that took *delivered* bytes of a file while the interceptor waited on
    it for *waited* seconds in all (``_drain``) kept up: it was waited on no longer than one
    prompt drain may take, ``_PROMPT_SEC``, or it took ``MIN_READ_RATE`` for every second it was.
    Only that waiting holds the upstream, so an upstream slow on its own account counts for
    nothing here."""
    return waited <= _PROMPT_SEC or delivered >= MIN_READ_RATE * waited


class _ClientStalled(Exception):
    """A client of the index took none of its reply for ``_WRITE_TIMEOUT_SEC``."""


async def _drain(writer: asyncio.StreamWriter) -> float:
    """Wait until the index's client has taken enough of what was written to it to make room
    for more, and return the seconds this waited on it. A client that takes none of it for
    ``_WRITE_TIMEOUT_SEC`` has its connection aborted, which frees what the reply held at once
    where a close would keep it until sent, and ``_ClientStalled`` is raised. The client sees a
    reset, as when an upstream drops a connection; nothing is counted, since the client's own
    stall ended the reply."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        await asyncio.wait_for(writer.drain(), _WRITE_TIMEOUT_SEC)
    except asyncio.TimeoutError:
        writer.transport.abort()
        raise _ClientStalled from None
    return loop.time() - started


def check_upstream_proxy(url: "str | None") -> "str | None":
    """Return *url* if it is ``http://<host>:<port>`` with nothing else, or None for None.
    Raise ValueError otherwise, saying what is wrong without quoting *url*, which may hold
    credentials: credentials in it would be written where the seal's arguments are recorded,
    and a path or a query would be dropped silently."""
    if url is None:
        return None
    parts = urlsplit(url)
    try:
        port = parts.port
    except ValueError:
        port = None
    wrong = [what for what, found in (
        ("its scheme is not http", parts.scheme != "http"),
        ("it names no host", not parts.hostname),
        ("it names no port from 1 to 65535", not port),
        ("it holds credentials", parts.username is not None or parts.password is not None),
        ("it has a path", parts.path not in ("", "/")),
        ("it has a query", bool(parts.query)),
        ("it has a fragment", bool(parts.fragment)),
    ) if found]
    if wrong:
        raise ValueError("the upstream proxy must be http://<host>:<port> with nothing else; " + "; ".join(wrong))
    return url


def _is_redirect(resp: httpx.Response) -> bool:
    return 300 <= resp.status_code < 400 and resp.status_code != 304


def _ctype(resp: httpx.Response) -> str:
    return resp.headers.get("content-type", "")


if __name__ == "__main__":
    asyncio.run(EgressProxy.from_env().serve_forever())
