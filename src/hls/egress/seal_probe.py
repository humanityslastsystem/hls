"""The seal probe: a harness program that runs in a seal before any scored work and shows that the
seal holds. The drive's seal probe runs it in a container sealed as a served
artifact will be (``served``); the build's seal probe runs its ``icmp`` and ``hold`` commands in
the build's sandbox, through the exec server (``hls.agent.probe``).

    python -I seal_probe.py served --port PORT --inbound-tcp PORT --inbound-udp PORT --out FILE
        --ucounts PAIRS [--refused-host HOST --dial ADDR:PORT] [--check-sec S]
    python -I seal_probe.py icmp
    python -I seal_probe.py hold --inbound-tcp PORT --inbound-udp PORT --ready FILE --done FILE
        [--check-sec S]

``served`` first binds two more ports on 0.0.0.0, a TCP listener on ``--inbound-tcp`` and a UDP
socket on ``--inbound-udp``, which the seal must not forward, and notes whatever reaches them
until its last check ends. Then it makes six checks, each within ``--check-sec`` seconds, in this
order:
1. caps: it reads its own capability sets and passes when every one is empty (``caps_check``): it
   runs with the capabilities the artifact will have, which hold none;
2. ucounts: it reads each limit the seal was to write in its user namespace, ``--ucounts``, the
   seal's own ``HLS_SEAL_UCOUNTS`` (``NAME=LIMIT`` pairs), from ``/proc/sys/user`` inside the seal,
   then tries to raise the first by one, and passes when each reads as given and the raise was
   refused (``ucounts_check``), before any contestant code runs;
3. refusal: it connects to ``--dial``, an address only the seal's redirect to its interceptor
   takes, and asks for ``--refused-host``, a host the seal's policy refuses. The interceptor
   reads the host and closes the connection without a reply. The check passes when the
   connection was taken and then closed, or reset, with no byte answered; a connection nothing
   took (no redirect) or a reply (the host was let through) fails it. Without
   ``--refused-host`` the policy refuses no host, and the check is not made;
4. icmp: it tries to send an ICMP echo out of the seal, to its namespace's default gateway, over
   a ping socket and over a raw socket (``icmp_check``). It passes when no echo left;
5. ingress: once its inbound ports have been bound for ``INBOUND_SETTLE_SEC``, it listens on
   ``--port``, the port the artifact will listen on inside the seal, and answers the first
   request with its path, which the stage sends through the seal's host port and checks. It
   passes once that answer is sent;
6. inbound: it waits for a second request on ``--port``, which the stage sends once it has
   knocked on the inbound ports from every address of the node (``knock_inbound``); then it
   closes the inbound ports, and only then answers. It passes when nothing reached them. The
   inbound ports are held for every knock, and closed for anything later, whatever the timing.
It writes how each check went to FILE (``write_report``) and exits 0 when every check it made
passed, 1 otherwise.

``icmp`` prints the ICMP check as one JSON line. ``hold`` binds the inbound ports, writes the file
``--ready`` once they have been bound for ``INBOUND_SETTLE_SEC``, holds them until the stage
makes the file ``--done`` (after its knocks), closes them, and prints the inbound check and the
protocols that reached them as one JSON line (``held``).

The stage's half lives here too, run on the node outside the seal: ``inbound_ports`` picks the
inbound ports, ``node_addresses`` the addresses to knock from, ``knock_inbound`` knocks, and
``knock_failures`` says what the knocks that found a port open mean.

It runs on the image's own interpreter, in isolated mode, as a script from the plan's checkout
(the drive) or as ``-c`` code (the build's sandbox, which has no checkout), and uses the standard
library alone, so it imports nothing of the checkout's packages or of the image's."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import random
import re
import select
import socket
import struct
import sys
import time
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from pathlib import Path

#: Seconds each check of a seal probe may take: the drive's probe gives each of its checks this
#: (``--check-sec``, its default too) within the probe's own time
#: (``hls.stages.deploy_drive.SEAL_PROBE_SEC``), and the build's seal probe runs each of its
#: commands under it (``hls.stages.agent_build``), the inbound check's waiting this long for the
#: stage's knocks. 30 s is well above the slowest check of a working seal, the inbound one, which
#: waits out ``INBOUND_SETTLE_SEC``. A passing check ends as soon as its answer comes, so only a
#: failing one waits it out.
CHECK_SEC = 30
#: Bytes of a request's head the ingress check reads at most.
HEAD_BYTES = 8192
#: Seconds the inbound ports stay bound before the stage may knock on them. pasta rescans the
#: ports a namespace binds each second for a forward direction its command line leaves unnamed
#: (FWD_PORT_SCAN_INTERVAL in the passt release hls.substrate.pasta pins), so a seal that
#: forwards them has done so by then.
INBOUND_SETTLE_SEC = 3.0
#: Seconds between the ``hold`` command's looks for the stage's done file.
DONE_POLL_SEC = 0.05
#: Seconds the stage waits for each knock on an inbound port to be taken, refused or answered.
KNOCK_SEC = 2.0
#: Where the ICMP check sends its echo when the namespace has no default gateway: an address of
#: TEST-NET-1 (RFC 5737).
ECHO_FALLBACK = "192.0.2.1"
#: Seconds the ICMP check waits for an answer to an echo that left.
ECHO_WAIT_SEC = 2.0
#: The node's ephemeral port range, which pasta never forwards by default.
PORT_RANGE_SYSCTL = Path("/proc/sys/net/ipv4/ip_local_port_range")
#: The first port an unprivileged process (pasta) may bind.
UNPRIVILEGED_PORT_SYSCTL = Path("/proc/sys/net/ipv4/ip_unprivileged_port_start")
#: The first port an unprivileged process may bind in a network namespace that sets no
#: ``ip_unprivileged_port_start`` of its own: the kernel's default, which pasta's fresh namespace
#: keeps whatever the host set. The build's sandbox commands bind there unprivileged.
NAMESPACE_PORT_FLOOR = 1024
#: This namespace's IPv4 routing table.
ROUTE_TABLE = Path("/proc/net/route")
#: This user namespace's own limits on the uid's counts (``hls.substrate.ucounts``).
UCOUNTS_ROOT = Path("/proc/sys/user")
#: This process's status, whose lines name its capability sets.
PROC_STATUS = Path("/proc/self/status")
#: The capability sets of a process, as its status names them (proc(5)): inheritable, permitted,
#: effective, bounding and ambient. Root of a user namespace regains on exec what its bounding and
#: inheritable sets keep, so an empty effective set alone shows nothing.
CAP_SETS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
#: The form of a ``NAME=LIMIT`` pair of a seal's limits: a file of ``UCOUNTS_ROOT`` and digits, the
#: form the seal script itself writes alone.
UCOUNT_PAIR = re.compile(r"(max_[a-z_]+)=([0-9]+)")
#: The ioctls that read an interface's flags and its IPv4 address (linux/sockios.h), and the flag
#: of an interface that is up (linux/if.h).
_SIOCGIFFLAGS, _SIOCGIFADDR, _IFF_UP = 0x8913, 0x8915, 0x1


@dataclass(frozen=True)
class Check:
    """How one check went."""

    #: Whether it passed.
    passed: bool
    #: Seconds it took.
    sec: float
    #: What it saw, for people.
    detail: str


def refusal_check(dial: str, host: str, *, within_sec: float) -> Check:
    """Connect to *dial* (``ADDR:PORT``), ask for *host* over HTTP, and pass when the connection
    is closed, or reset, without a byte of reply within *within_sec* seconds."""
    started = time.monotonic()
    address, _, port = dial.rpartition(":")

    def took() -> float:
        return round(time.monotonic() - started, 3)

    try:
        conn = socket.create_connection((address, int(port)), timeout=within_sec)
    except OSError as e:
        return Check(passed=False, sec=took(),
                     detail=f"the connection to {dial} could not be made, so nothing redirected it: {e}")
    with conn:
        try:
            conn.sendall(f"GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode())
            conn.settimeout(max(0.001, within_sec - (time.monotonic() - started)))
            reply = conn.recv(1)
        except ConnectionResetError:
            reply = b""
        except TimeoutError:
            return Check(passed=False, sec=took(), detail=f"{host} was neither answered nor refused within "
                                                           f"{within_sec}s")
        except OSError as e:
            return Check(passed=False, sec=took(), detail=f"asking for {host} failed: {e}")
    if reply:
        return Check(passed=False, sec=took(), detail=f"{host}, which the policy refuses, was answered")
    return Check(passed=True, sec=took(), detail=f"{host} was refused: the connection closed without a reply")


@dataclass
class Inbound:
    """The probe's inbound ports, bound on 0.0.0.0 inside the seal, and what reached them."""

    #: The TCP listener.
    tcp: socket.socket
    udp: socket.socket
    #: The port the TCP listener is bound on.
    tcp_port: int
    #: The port the UDP socket is bound on.
    udp_port: int
    #: When they were bound, on the monotonic clock.
    bound_at: float
    #: What reached them, for people: one entry a connection or datagram.
    reached: list[str] = field(default_factory=list)
    #: The protocols that reached them, ``tcp`` and ``udp``.
    protocols: set[str] = field(default_factory=set)

    @classmethod
    def bind(cls, tcp_port: int, udp_port: int) -> Inbound:
        tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        tcp.bind(("0.0.0.0", tcp_port))
        tcp.listen()
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind(("0.0.0.0", udp_port))
        return cls(tcp=tcp, udp=udp, tcp_port=tcp.getsockname()[1], udp_port=udp.getsockname()[1],
                   bound_at=time.monotonic())

    def note(self, ready: list[socket.socket]) -> None:
        """Take whatever is waiting on the inbound sockets among *ready*, and note it."""
        if self.tcp in ready:
            conn, peer = self.tcp.accept()
            conn.close()
            self.reached.append(f"tcp port {self.tcp_port} from {peer[0]}")
            self.protocols.add("tcp")
        if self.udp in ready:
            _, peer = self.udp.recvfrom(2048)
            self.reached.append(f"udp port {self.udp_port} from {peer[0]}")
            self.protocols.add("udp")

    def watch(self, sec: float) -> None:
        """Note what reaches the inbound sockets for *sec* seconds; with none left, take only
        what is already waiting."""
        until = time.monotonic() + sec
        while True:
            ready, _, _ = select.select([self.tcp, self.udp], [], [], max(0.0, until - time.monotonic()))
            self.note(ready)
            if time.monotonic() >= until and not ready:
                return

    def settle(self) -> None:
        """Hold the ports, noting what reaches them, until they have been bound for
        ``INBOUND_SETTLE_SEC``."""
        self.watch(self.bound_at + INBOUND_SETTLE_SEC - time.monotonic())

    def close(self) -> None:
        """Take what is waiting, then close both sockets."""
        if self.tcp.fileno() >= 0 and self.udp.fileno() >= 0:
            self.watch(0.0)
        self.tcp.close()
        self.udp.close()

    def check(self, since: float) -> Check:
        """The inbound check, from what reached the ports, timed from *since*."""
        sec = round(time.monotonic() - since, 3)
        if self.reached:
            return Check(passed=False, sec=sec, detail="reached inside the seal: " + "; ".join(self.reached[:8]))
        return Check(passed=True, sec=sec, detail=f"nothing reached tcp port {self.tcp_port} or udp port "
                                                  f"{self.udp_port}")


def _answer(server: socket.socket, inbound: Inbound, *, until: float, closing: bool = False) -> str | None:
    """Wait for one request on *server* until *until*, noting what reaches *inbound* meanwhile,
    and answer it with its path, having closed *inbound* first when *closing*; None once
    answered, else what went wrong."""
    while (left := until - time.monotonic()) > 0:
        ready, _, _ = select.select([server, inbound.tcp, inbound.udp], [], [], left)
        inbound.note(ready)
        if server not in ready:
            continue
        conn, _ = server.accept()
        with conn:
            conn.settimeout(max(0.001, until - time.monotonic()))
            head = b""
            while b"\r\n\r\n" not in head and len(head) < HEAD_BYTES:
                chunk = conn.recv(HEAD_BYTES - len(head))
                if not chunk:
                    break
                head += chunk
            fields = head.split(b"\r\n", 1)[0].split()
            if len(fields) < 2:
                return f"the request on port {server.getsockname()[1]} named no path"
            if closing:
                inbound.close()
            body = fields[1].lstrip(b"/")
            conn.sendall(b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n%s"
                         % (len(body), body))
        return None
    return f"no request reached port {server.getsockname()[1]}"


def serve(port: int, inbound: Inbound, *, within_sec: float) -> tuple[Check, Check]:
    """The ingress and the inbound checks: once *inbound* has settled, listen on *port* and answer
    the stage's first request with its path, then take its second, close *inbound* and answer
    it, each within *within_sec* seconds; the inbound check passes when nothing reached
    *inbound* from its binding until it closed."""
    inbound.settle()
    started = time.monotonic()

    def took(since: float) -> float:
        return round(time.monotonic() - since, 3)

    never = Check(passed=False, sec=0.0, detail="the stage never asked for the inbound check")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind(("0.0.0.0", port))
            server.listen()
            failed = _answer(server, inbound, until=started + within_sec)
            if failed is not None:
                return Check(passed=False, sec=took(started), detail=failed), never
            ingress = Check(passed=True, sec=took(started), detail=f"a request on port {port} was answered")
            window = time.monotonic()
            failed = _answer(server, inbound, until=window + within_sec, closing=True)
    except OSError as e:
        return Check(passed=False, sec=took(started), detail=f"port {port} could not be served: {e}"), never
    if failed is not None:
        return ingress, Check(passed=False, sec=took(window), detail=f"no second request: {failed}")
    return ingress, inbound.check(window)


def hold(inbound: Inbound, ready: Path, done: Path, *, within_sec: float) -> Check:
    """The build's inbound check: once *inbound* has settled, make the file *ready*, hold
    *inbound* until the file *done* exists, at most *within_sec* seconds, then close it; pass
    when nothing reached it."""
    inbound.settle()
    ready.write_text("ready\n")
    started = time.monotonic()
    until = started + within_sec
    while not done.exists():
        if time.monotonic() >= until:
            inbound.close()
            return Check(passed=False, sec=round(time.monotonic() - started, 3),
                         detail="the stage never ended the inbound check")
        inbound.watch(DONE_POLL_SEC)
    inbound.close()
    return inbound.check(started)


def default_gateway(route: Path = ROUTE_TABLE) -> str | None:
    """The IPv4 default gateway in the routing table *route*; None without one."""
    try:
        lines = route.read_text().splitlines()[1:]
    except OSError:
        return None
    for line in lines:
        fields = line.split()
        # RTF_GATEWAY (0x2) on the default route.
        if len(fields) > 3 and fields[1] == "00000000" and int(fields[3], 16) & 0x2:
            return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    return None


def _echo_request(token: bytes) -> bytes:
    """An ICMP echo request carrying *token*, its checksum filled in."""
    def checksum(data: bytes) -> int:
        data += b"\0" * (len(data) % 2)
        total = sum(struct.unpack(f"!{len(data) // 2}H", data))
        total = (total >> 16) + (total & 0xFFFF)
        return ~(total + (total >> 16)) & 0xFFFF

    header = struct.pack("!BBHHH", 8, 0, 0, 0, 1)
    return struct.pack("!BBHHH", 8, 0, checksum(header + token), 0, 1) + token


def icmp_check(target: str) -> Check:
    """Try to send an ICMP echo to *target* over a ping socket and over a raw socket, and pass when
    neither left: the socket would not open, or the send was refused. An echo that left fails the
    check whether or not it was answered, since the seal lets none out: a gateway that ignores
    ping answers none; one answered within ``ECHO_WAIT_SEC`` says so."""
    started = time.monotonic()
    held, left = [], []
    for kind, sock_type in (("ping", socket.SOCK_DGRAM), ("raw", socket.SOCK_RAW)):
        try:
            s = socket.socket(socket.AF_INET, sock_type, socket.IPPROTO_ICMP)
        except OSError as e:
            held.append(f"no {kind} socket ({e.strerror})")
            continue
        with s:
            token = f"hls-seal-probe-{kind}-{random.getrandbits(64):016x}".encode()
            try:
                s.sendto(_echo_request(token), (target, 0))
            except OSError as e:
                held.append(f"the {kind} echo was refused ({e.strerror})")
                continue
            s.settimeout(ECHO_WAIT_SEC)
            try:
                answer = s.recv(4096)
                left.append(f"a {kind} echo to {target} was answered{' with its data' if token in answer else ''}")
            except OSError:
                left.append(f"a {kind} echo to {target} left the seal, unanswered")
    sec = round(time.monotonic() - started, 3)
    if left:
        return Check(passed=False, sec=sec, detail="; ".join(left))
    return Check(passed=True, sec=sec, detail="no echo left: " + "; ".join(held))


def ucounts_check(given: str, root: Path) -> Check:
    """Read each limit of *given*, a seal's ``NAME=LIMIT`` pairs, from *root*/NAME, then try to
    raise the first by one, and pass when every one reads as given and the raise was refused; with
    none given, there is nothing to read. The probe runs with the capabilities the artifact will
    have, so a raise that takes shows the artifact could lift its own limits."""
    started = time.monotonic()

    def done(passed: bool, detail: str) -> Check:
        return Check(passed=passed, sec=round(time.monotonic() - started, 3), detail=detail)

    pairs = [UCOUNT_PAIR.fullmatch(pair) for pair in given.split()]
    bad = [pair for pair, m in zip(given.split(), pairs) if m is None]
    if bad:
        return done(False, f"the seal was given {bad[0]}, not NAME=LIMIT for a file of {root}")
    if not pairs:
        return done(True, "the seal was given no limits")
    read, wrong = [], []
    for m in pairs:
        name, limit = m.group(1), int(m.group(2))
        try:
            value = int((root / name).read_text().split()[0])
        except (OSError, ValueError, IndexError) as e:
            wrong.append(f"{name} could not be read: {e}")
            continue
        if value != limit:
            wrong.append(f"{name} reads {value}, not {limit}")
        read.append(f"{name} {value}")
    if wrong:
        return done(False, "; ".join(wrong))
    name, limit = pairs[0].group(1), int(pairs[0].group(2))
    try:
        (root / name).write_text(f"{limit + 1}\n")
    except OSError as e:
        return done(True, f"read as given: {', '.join(read)}; raising {name} to {limit + 1} was refused "
                          f"({e.strerror})")
    return done(False, f"{name} was raised from {limit} to {limit + 1} inside the seal: the role could lift "
                       "its own limits")


def caps_check(status: Path) -> Check:
    """Read this process's capability sets (``CAP_SETS``) from *status*, its ``/proc`` status, and
    pass when every one is empty."""
    started = time.monotonic()

    def done(passed: bool, detail: str) -> Check:
        return Check(passed=passed, sec=round(time.monotonic() - started, 3), detail=detail)

    try:
        text = status.read_text()
    except OSError as e:
        return done(False, f"{status} could not be read: {e}")
    sets = dict(line.split(":\t", 1) for line in text.splitlines() if line.startswith("Cap") and ":\t" in line)
    missing = [name for name in CAP_SETS if name not in sets]
    if missing:
        return done(False, f"{status} shows no {', '.join(missing)}")
    held = [f"{name} {sets[name].strip()}" for name in CAP_SETS if int(sets[name], 16) != 0]
    if held:
        return done(False, "the probe holds capabilities: " + ", ".join(held))
    return done(True, f"no capability in any set: {', '.join(CAP_SETS[:-1])} and {CAP_SETS[-1]} are 0")


@dataclass(frozen=True)
class Report:
    """How the ``served`` probe's checks went."""

    #: The refusal check; None when it was not made.
    refusal: Check | None
    ingress: Check
    inbound: Check
    icmp: Check
    #: The check of the limits the seal wrote in its user namespace.
    ucounts: Check
    #: The check of the probe's own capabilities.
    caps: Check
    #: The protocols that reached the inbound ports, ``tcp`` and ``udp``, sorted.
    reached: tuple[str, ...]


def write_report(path: Path, report: Report) -> None:
    """Write *report* to *path*: ``{"refusal": <check> | null, "ingress": <check>,
    "inbound": <check>, "icmp": <check>, "ucounts": <check>, "caps": <check>,
    "reached": [<protocol>, ...]}``."""
    path.write_text(json.dumps(asdict(report)))


def _check(data: object) -> Check | None:
    if not isinstance(data, dict) or set(data) != {"passed", "sec", "detail"}:
        return None
    passed, sec, detail = data["passed"], data["sec"], data["detail"]
    if not isinstance(passed, bool) or isinstance(sec, bool) or not isinstance(sec, (int, float)):
        return None
    if not isinstance(detail, str):
        return None
    return Check(passed=passed, sec=float(sec), detail=detail)


def _protocols(data: object) -> tuple[str, ...] | None:
    if not isinstance(data, list) or not set(data) <= {"tcp", "udp"} or len(set(data)) != len(data):
        return None
    return tuple(sorted(data))


def read_report(path: Path) -> Report | None:
    """The report ``write_report`` wrote to *path*; None when there is no such report."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or set(data) != {"refusal", "ingress", "inbound", "icmp", "ucounts", "caps",
                                                   "reached"}:
        return None
    ingress, inbound, icmp = _check(data["ingress"]), _check(data["inbound"]), _check(data["icmp"])
    ucounts, caps = _check(data["ucounts"]), _check(data["caps"])
    refusal = None if data["refusal"] is None else _check(data["refusal"])
    reached = _protocols(data["reached"])
    if (ingress is None or inbound is None or icmp is None or ucounts is None or caps is None or reached is None
            or (data["refusal"] is not None and refusal is None)):
        return None
    return Report(refusal=refusal, ingress=ingress, inbound=inbound, icmp=icmp, ucounts=ucounts, caps=caps,
                  reached=reached)


def read_check(line: str, key: str) -> Check | None:
    """The check *key* of the JSON line an ``icmp`` or ``hold`` command printed; None when there is
    no such check."""
    try:
        data = json.loads(line)
    except ValueError:
        return None
    return _check(data.get(key)) if isinstance(data, dict) else None


def held(line: str) -> tuple[Check, tuple[str, ...]] | None:
    """The inbound check and the protocols that reached the ports, as a ``hold`` command printed
    them on *line*; None when it printed no such line."""
    check = read_check(line, "inbound")
    try:
        reached = _protocols(json.loads(line).get("reached"))
    except (ValueError, AttributeError):
        return None
    return None if check is None or reached is None else (check, reached)


def inbound_ports(*, avoid: Collection[int], floor: int = 0) -> tuple[int, int]:
    """The stage's choice of the probe's inbound ports, a TCP one and a UDP one: free on this
    node over both protocols, none of *avoid* (the ports the seal's own listeners and the probe's
    served port hold in its namespace), and among the ports pasta forwards when a direction is
    left unnamed: from the first port it may bind unprivileged (``ip_unprivileged_port_start``),
    or *floor* when that is higher (where the probe binds unprivileged in the seal,
    ``NAMESPACE_PORT_FLOOR``), to the start of the node's ephemeral range
    (``ip_local_port_range``). Raises OSError when the node leaves no two."""
    start = max(int(UNPRIVILEGED_PORT_SYSCTL.read_text().split()[0]), floor, 1)
    low = int(PORT_RANGE_SYSCTL.read_text().split()[0])
    candidates = [p for p in range(start, low) if p not in avoid]
    random.shuffle(candidates)
    free: list[int] = []
    for port in candidates:
        try:
            with socket.socket() as t, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:
                t.bind(("0.0.0.0", port))
                u.bind(("0.0.0.0", port))
        except OSError:
            continue
        free.append(port)
        if len(free) == 2:
            return free[0], free[1]
    raise OSError(f"no two free ports between {start} and the ephemeral range at {low} for the inbound check")


def node_addresses() -> tuple[str, ...]:
    """The IPv4 address of each of this node's interfaces that is up and has one, loopback among
    them. A down interface's address has no local route left, so a knock on it would leave the
    node."""
    found: list[str] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        for _, name in socket.if_nameindex():
            request = struct.pack("256s", name.encode()[:15])
            try:
                flags = struct.unpack("H", fcntl.ioctl(s.fileno(), _SIOCGIFFLAGS, request)[16:18])[0]
                if not flags & _IFF_UP:
                    continue
                raw = fcntl.ioctl(s.fileno(), _SIOCGIFADDR, request)
            except OSError:
                continue
            found.append(socket.inet_ntoa(raw[20:24]))
    return tuple(dict.fromkeys(found))


@dataclass(frozen=True)
class Knock:
    """The stage's try to reach one inbound port from one address of the node."""

    #: The address it knocked on.
    address: str
    #: ``tcp`` or ``udp``.
    proto: str
    port: int
    #: For TCP: ``refused`` (reset), ``connected`` or ``timed out``; for UDP: ``refused`` (port
    #: unreachable came back), ``unanswered`` or ``answered``; or ``error: ...``.
    outcome: str

    @property
    def refused(self) -> bool:
        """Whether it found the port closed on the node: nothing there took it, over either
        protocol. While the probe holds the port in its seal, that means the seal forwards it
        nowhere."""
        return self.outcome == "refused"


_WORD = b"hls-seal-probe inbound knock\n"


def knock_inbound(addresses: tuple[str, ...], tcp_port: int, udp_port: int) -> tuple[Knock, ...]:
    """Try to reach *tcp_port* over TCP and *udp_port* over UDP on each of *addresses* at once,
    sending a word on each, and say what happened within ``KNOCK_SEC``: a TCP knock first for
    every address, then a UDP one, in that order."""
    tcp = {address: socket.socket(socket.AF_INET, socket.SOCK_STREAM) for address in addresses}
    udp = {address: socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for address in addresses}
    outcomes: dict[tuple[str, str], str] = {}
    try:
        for address, s in tcp.items():
            s.setblocking(False)
            err = s.connect_ex((address, tcp_port))
            if err not in (0, errno.EINPROGRESS):
                outcomes["tcp", address] = "refused" if err == errno.ECONNREFUSED else f"error: {os.strerror(err)}"
        for address, s in udp.items():
            try:
                s.connect((address, udp_port))
                s.send(_WORD)
            except ConnectionRefusedError:
                outcomes["udp", address] = "refused"
            except OSError as e:
                outcomes["udp", address] = f"error: {e}"
        until = time.monotonic() + KNOCK_SEC
        while (left := until - time.monotonic()) > 0:
            writing = [s for a, s in tcp.items() if ("tcp", a) not in outcomes]
            reading = [s for a, s in udp.items() if ("udp", a) not in outcomes]
            if not writing and not reading:
                break
            readable, writable, _ = select.select(reading, writing, [], left)
            for address, s in tcp.items():
                if s in writable:
                    err = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
                    if err == 0:
                        outcomes["tcp", address] = "connected"
                        try:
                            s.send(_WORD)
                        except OSError:
                            pass
                    else:
                        outcomes["tcp", address] = ("refused" if err == errno.ECONNREFUSED
                                                    else f"error: {os.strerror(err)}")
            for address, s in udp.items():
                if s in readable:
                    try:
                        s.recv(64)
                        outcomes["udp", address] = "answered"
                    except ConnectionRefusedError:
                        outcomes["udp", address] = "refused"
                    except OSError as e:
                        outcomes["udp", address] = f"error: {e}"
    finally:
        for s in (*tcp.values(), *udp.values()):
            s.close()
    return tuple(Knock(address=address, proto=proto, port=tcp_port if proto == "tcp" else udp_port,
                       outcome=outcomes.get((proto, address), "timed out" if proto == "tcp" else "unanswered"))
                 for proto in ("tcp", "udp") for address in addresses)


def knock_failures(addresses: tuple[str, ...], knocks: tuple[Knock, ...], reached: Collection[str]) -> list[str]:
    """Why the *knocks* from *addresses* do not show the inbound ports closed: no address but
    loopback to knock from, or a knock that found its port open. That port the seal forwarded when
    the probe saw its protocol arrive (*reached*); otherwise something on the node holds it.
    Either way the probe failed, as infra."""
    failures = []
    if not any(not a.startswith("127.") for a in addresses):
        failures.append("the node has no address but loopback to knock on the inbound ports from")
    for k in knocks:
        if k.refused:
            continue
        if k.proto in reached:
            failures.append(f"the seal forwarded {k.proto} port {k.port} on {k.address} ({k.outcome}): the probe "
                            "saw it inside the seal")
        else:
            failures.append(f"{k.proto} port {k.port} on {k.address} is held on the node ({k.outcome}): the probe "
                            "saw nothing of it inside the seal")
    return failures


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="seal_probe")
    commands = p.add_subparsers(dest="command", required=True)
    served = commands.add_parser("served")
    served.add_argument("--port", type=int, required=True)
    served.add_argument("--out", type=Path, required=True)
    served.add_argument("--refused-host")
    served.add_argument("--dial")
    served.add_argument("--ucounts", required=True)
    commands.add_parser("icmp")
    hold_ = commands.add_parser("hold")
    hold_.add_argument("--ready", type=Path, required=True)
    hold_.add_argument("--done", type=Path, required=True)
    for command in (served, hold_):
        command.add_argument("--inbound-tcp", type=int, required=True)
        command.add_argument("--inbound-udp", type=int, required=True)
        command.add_argument("--check-sec", type=float, default=CHECK_SEC)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "icmp":
        print(json.dumps({"icmp": asdict(icmp_check(default_gateway() or ECHO_FALLBACK))}))
        return 0
    inbound = Inbound.bind(args.inbound_tcp, args.inbound_udp)
    try:
        if args.command == "hold":
            check = hold(inbound, args.ready, args.done, within_sec=args.check_sec)
            print(json.dumps({"inbound": asdict(check), "reached": sorted(inbound.protocols)}))
            return 0
        if (args.refused_host is None) != (args.dial is None):
            p.error("--refused-host and --dial go together")
        caps = caps_check(PROC_STATUS)
        ucounts = ucounts_check(args.ucounts, UCOUNTS_ROOT)
        refusal = None
        if args.refused_host is not None:
            refusal = refusal_check(args.dial, args.refused_host, within_sec=args.check_sec)
        icmp = icmp_check(default_gateway() or ECHO_FALLBACK)
        ingress, inbound_check = serve(args.port, inbound, within_sec=args.check_sec)
    finally:
        inbound.close()
    report = Report(refusal=refusal, ingress=ingress, inbound=inbound_check, icmp=icmp, ucounts=ucounts, caps=caps,
                    reached=tuple(sorted(inbound.protocols)))
    write_report(args.out, report)
    passed = (ingress.passed and inbound_check.passed and icmp.passed and ucounts.passed and caps.passed
              and (refusal is None or refusal.passed))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
