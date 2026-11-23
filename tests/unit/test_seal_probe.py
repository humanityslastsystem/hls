from __future__ import annotations

import os
import random
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from hls.egress import seal_probe
from hls.egress.policy import AccessList, NetworkPolicy, SealPolicy
from hls.egress.seal_probe import Check, Inbound, Report, read_report, refusal_check, serve, write_report
from hls.substrate.seal import EGRESS_TRANSPARENT_PORT, index_status, namespace_ports, policy_refusals
from hls.substrate.container import SealPlan
from tests.drive.interceptor import Interceptor


def _listener() -> socket.socket:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen()
    return s


def _serve_once(sock: socket.socket, reply: bytes | None, *, hold_sec: float = 0.0) -> None:
    """Accept one connection, read its request, then answer *reply*, or close it unanswered;
    with *hold_sec*, hold it open that long first."""
    def run() -> None:
        conn, _ = sock.accept()
        with conn:
            conn.recv(4096)
            time.sleep(hold_sec)
            if reply is not None:
                conn.sendall(reply)

    threading.Thread(target=run, daemon=True).start()


def _free_port(*, avoid: tuple[int, ...] = ()) -> int:
    """A port free over TCP and UDP below the kernel's range for outgoing connections, so that no
    client socket on the machine takes it before the probe listens on it."""
    while True:
        port = random.randrange(20000, 32000)
        if port in avoid:
            continue
        try:
            with socket.socket() as t, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:
                t.bind(("0.0.0.0", port))
                u.bind(("0.0.0.0", port))
        except OSError:
            continue
        return port


def _knock(port: int, word: str, *, within_sec: float = 10) -> httpx.Response:
    """The stage's request through the seal's host port, sent until something answers."""
    deadline = time.monotonic() + within_sec
    while True:
        try:
            return httpx.get(f"http://127.0.0.1:{port}/{word}", timeout=2, trust_env=False)
        except httpx.TransportError:
            assert time.monotonic() < deadline
            time.sleep(0.05)


def _serving(port: int, *, within_sec: float = 5) -> tuple[threading.Thread, list, Inbound]:
    """The probe's ingress and inbound checks on *port*, run in a thread, with its inbound
    sockets on two more free ports."""
    tcp = _free_port(avoid=(port,))
    inbound = Inbound.bind(tcp, _free_port(avoid=(port, tcp)))
    result: list[tuple[Check, Check]] = []
    thread = threading.Thread(target=lambda: result.append(serve(port, inbound, within_sec=within_sec)))
    thread.start()
    return thread, result, inbound


def test_a_connection_closed_without_a_reply_is_a_refusal():
    with _listener() as sock:
        _serve_once(sock, None)
        check = refusal_check(f"127.0.0.1:{sock.getsockname()[1]}", "probe.invalid", within_sec=5)
    assert check.passed and "refused" in check.detail and 0 <= check.sec < 5


def test_a_host_the_seal_answered_fails_the_refusal_check():
    with _listener() as sock:
        _serve_once(sock, b"HTTP/1.1 200 OK\r\n\r\n")
        check = refusal_check(f"127.0.0.1:{sock.getsockname()[1]}", "probe.invalid", within_sec=5)
    assert not check.passed and "answered" in check.detail


def test_a_connection_nothing_took_fails_the_refusal_check():
    # Without the seal's redirect the dialed address takes no connection; that is no refusal.
    check = refusal_check(f"127.0.0.1:{_free_port()}", "probe.invalid", within_sec=5)
    assert not check.passed and "could not be made" in check.detail


def test_a_connection_held_open_fails_the_refusal_check_within_its_time():
    with _listener() as sock:
        _serve_once(sock, None, hold_sec=5)
        started = time.monotonic()
        check = refusal_check(f"127.0.0.1:{sock.getsockname()[1]}", "probe.invalid", within_sec=0.5)
    assert not check.passed and "within" in check.detail and time.monotonic() - started < 3


def test_the_probe_answers_the_stages_two_requests_and_nothing_else_arrived():
    # The first request ends the ingress check; the second, once the stage tried the inbound
    # ports, ends the inbound check. Nothing reached those ports, so both pass.
    port = _free_port()
    thread, result, _ = _serving(port)
    first, second = _knock(port, "nonce-123"), _knock(port, "nonce-456")
    thread.join(5)
    assert (first.status_code, first.content, second.status_code, second.content) == (
        200, b"nonce-123", 200, b"nonce-456")
    [(ingress, inbound)] = result
    assert ingress.passed and 0 <= ingress.sec < 5
    assert inbound.passed and "nothing reached" in inbound.detail


def test_the_probe_holds_its_inbound_ports_bound_before_it_listens(monkeypatch):
    # pasta rescans a namespace's bound ports about once a second, so a seal that forwards them
    # has done so by the time the stage's first request is answered.
    monkeypatch.setattr(seal_probe, "INBOUND_SETTLE_SEC", 1.5)
    port = _free_port()
    thread, result, inbound = _serving(port)
    _knock(port, "a")
    answered = time.monotonic()
    _knock(port, "b")
    thread.join(5)
    assert answered - inbound.bound_at >= 1.5 and result[0][1].passed


def test_the_probe_closes_its_inbound_ports_before_it_answers_the_second_request():
    # So a knock the stage makes before its second request finds them held, and one made after
    # it finds them closed, whatever the timing: the order alone decides what a knock sees.
    port = _free_port()
    thread, result, inbound = _serving(port)
    _knock(port, "a")
    held = seal_probe.knock_inbound(("127.0.0.1",), inbound.tcp_port, inbound.udp_port)
    _knock(port, "b")
    after = seal_probe.knock_inbound(("127.0.0.1",), inbound.tcp_port, inbound.udp_port)
    thread.join(5)
    assert [k.outcome for k in held] == ["connected", "unanswered"]
    assert [k.outcome for k in after] == ["refused", "refused"]
    assert not result[0][1].passed and inbound.protocols == {"tcp", "udp"}


@pytest.mark.parametrize("proto", ["tcp", "udp"])
def test_anything_that_reaches_an_inbound_port_fails_the_inbound_check(proto):
    # A seal that forwards the container's other ports lets the stage reach them; the probe
    # notes whatever arrives, from its binding until the stage's second request.
    port = _free_port()
    thread, result, inbound = _serving(port)
    _knock(port, "a")
    if proto == "tcp":
        with socket.create_connection(("127.0.0.1", inbound.tcp_port), timeout=5) as c:
            c.sendall(b"word")
    else:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as u:
            u.sendto(b"word", ("127.0.0.1", inbound.udp_port))
    _knock(port, "b")
    thread.join(5)
    [(ingress, check)] = result
    assert ingress.passed and not check.passed and inbound.protocols == {proto}
    assert f"{proto} port {inbound.tcp_port if proto == 'tcp' else inbound.udp_port}" in check.detail


def test_an_ingress_check_no_request_reaches_fails_and_makes_no_inbound_check():
    port = _free_port()
    tcp = _free_port(avoid=(port,))
    ingress, inbound = serve(port, Inbound.bind(tcp, _free_port(avoid=(port, tcp))), within_sec=0.3)
    assert not ingress.passed and "no request" in ingress.detail
    assert not inbound.passed and "never" in inbound.detail


def test_a_stage_that_never_ends_the_inbound_window_fails_the_inbound_check():
    port = _free_port()
    thread, result, _ = _serving(port, within_sec=0.5)
    _knock(port, "a")
    thread.join(5)
    [(ingress, inbound)] = result
    assert ingress.passed and not inbound.passed and "second request" in inbound.detail


def _holding(tmp_path: Path, *, within_sec: float = 5) -> tuple[threading.Thread, list, Inbound]:
    tcp = _free_port()
    inbound = Inbound.bind(tcp, _free_port(avoid=(tcp,)))
    result: list[Check] = []
    thread = threading.Thread(target=lambda: result.append(
        seal_probe.hold(inbound, tmp_path / "ready", tmp_path / "done", within_sec=within_sec)))
    thread.start()
    return thread, result, inbound


def _ready(path: Path) -> None:
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def test_the_build_holds_its_inbound_ports_from_ready_until_done(tmp_path, monkeypatch):
    # The build's sandbox has no forwarded port to talk over: files say when the ports have
    # settled, and when the stage's knocks are over.
    monkeypatch.setattr(seal_probe, "INBOUND_SETTLE_SEC", 1.0)
    thread, result, inbound = _holding(tmp_path)
    _ready(tmp_path / "ready")
    assert time.monotonic() - inbound.bound_at >= 1.0
    held = seal_probe.knock_inbound(("127.0.0.1",), inbound.tcp_port, inbound.udp_port)
    (tmp_path / "done").touch()
    thread.join(5)
    after = seal_probe.knock_inbound(("127.0.0.1",), inbound.tcp_port, inbound.udp_port)
    assert [k.outcome for k in held] == ["connected", "unanswered"] and [k.outcome for k in after] == [
        "refused", "refused"]
    [check] = result
    assert not check.passed and inbound.protocols == {"tcp", "udp"}


def test_a_build_hold_nothing_reaches_passes_and_one_never_ended_fails(tmp_path):
    thread, result, _ = _holding(tmp_path)
    _ready(tmp_path / "ready")
    (tmp_path / "done").touch()
    thread.join(5)
    assert result[0].passed and "nothing reached" in result[0].detail
    (tmp_path / "late").mkdir()
    thread, result, _ = _holding(tmp_path / "late", within_sec=0.3)
    thread.join(10)
    assert not result[0].passed and "never ended" in result[0].detail


class _SealedSocket(socket.socket):
    """A socket as a sealed container gets one: an ICMP socket never opens, as in a namespace with
    no ping sockets and no CAP_NET_RAW."""

    def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
        if fileno is None and proto == socket.IPPROTO_ICMP:
            raise PermissionError(13, "Permission denied")
        super().__init__(family, type, proto, fileno)


class _OpenIcmpSocket(socket.socket):
    """An ICMP socket that opens whatever this host allows, ping or raw: a UDP socket stands in
    for it, so a check's outcome never rests on the host's ping range or CAP_NET_RAW."""

    def __init__(self, family=-1, type=-1, proto=-1, fileno=None):
        icmp = fileno is None and proto == socket.IPPROTO_ICMP
        super().__init__(family, socket.SOCK_DGRAM if icmp else type, 0 if icmp else proto, fileno)
        #: True when it stands in for an ICMP socket.
        self.icmp = icmp


class _DroppedSocket(_OpenIcmpSocket):
    """A socket behind a firewall that drops ICMP: it opens, and an echo it sends is refused."""

    def sendto(self, *args):
        if self.icmp:
            raise PermissionError(1, "Operation not permitted")
        return super().sendto(*args)


class _SilentSocket(_OpenIcmpSocket):
    """A socket whose echo leaves and is never answered, as through a seal that lets ICMP out to a
    gateway that ignores ping: it opens, sends, and hears nothing."""

    def sendto(self, data, address):
        return len(data) if self.icmp else super().sendto(data, address)

    def recv(self, n):
        if self.icmp:
            raise TimeoutError("timed out")
        return super().recv(n)


def test_the_probe_reports_its_checks_and_passes_only_when_all_do(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "socket", _SealedSocket)
    monkeypatch.setattr(seal_probe, "PROC_STATUS", _status(tmp_path))
    out = tmp_path / "report.json"
    port = _free_port()
    tcp = _free_port(avoid=(port,))
    udp = _free_port(avoid=(port, tcp))
    inbound_args = ["--inbound-tcp", str(tcp), "--inbound-udp", str(udp), "--ucounts", ""]
    with _listener() as sock:
        _serve_once(sock, None)
        codes: list[int] = []
        argv = ["served", "--port", str(port), "--out", str(out), "--refused-host", "probe.invalid",
                "--dial", f"127.0.0.1:{sock.getsockname()[1]}", "--check-sec", "5", *inbound_args]
        thread = threading.Thread(target=lambda: codes.append(seal_probe.main(argv)))
        thread.start()
        _knock(port, "n")
        _knock(port, "m")
        thread.join(10)
    assert codes == [0]
    report = read_report(out)
    assert report is not None and report.refusal is not None and report.refusal.passed
    assert report.ingress.passed and report.inbound.passed and report.icmp.passed and report.reached == ()
    assert report.ucounts.passed and report.caps.passed
    # With no host to ask for, the refusal check is not made; an ingress check that fails fails the probe.
    assert seal_probe.main(["served", "--port", str(_free_port()), "--out", str(out), "--check-sec", "0.2",
                            *inbound_args]) == 1
    report = read_report(out)
    assert report is not None and report.refusal is None and not report.ingress.passed and not report.inbound.passed


def test_the_probe_needs_its_inbound_ports(tmp_path):
    with pytest.raises(SystemExit):
        seal_probe.main(["served", "--port", "8000", "--out", str(tmp_path / "r.json"), "--ucounts", ""])
    with pytest.raises(SystemExit):  # nor without the limits it was given, empty or not
        seal_probe.main(["served", "--port", "8000", "--out", str(tmp_path / "r.json"), "--inbound-tcp", "20001",
                         "--inbound-udp", "20002"])
    with pytest.raises(SystemExit):
        seal_probe.main(["hold", "--ready", str(tmp_path / "r"), "--done", str(tmp_path / "d")])


@pytest.mark.parametrize("sealed, held", [
    (_SealedSocket, "no ping socket (Permission denied); no raw socket (Permission denied)"),
    (_DroppedSocket, "the ping echo was refused (Operation not permitted); the raw echo was refused (Operation not "
                     "permitted)")], ids=["no-ping-sockets", "icmp-dropped"])
def test_an_echo_that_cannot_leave_passes_the_icmp_check(monkeypatch, sealed, held):
    monkeypatch.setattr(socket, "socket", sealed)
    check = seal_probe.icmp_check("192.0.2.1")
    assert check.passed and check.detail == f"no echo left: {held}"


def test_an_echo_that_left_unanswered_fails_the_icmp_check(monkeypatch):
    # A gateway that ignores ping answers nothing; the echo still left the seal.
    monkeypatch.setattr(socket, "socket", _SilentSocket)
    check = seal_probe.icmp_check("192.0.2.1")
    assert not check.passed
    assert check.detail == ("a ping echo to 192.0.2.1 left the seal, unanswered; "
                            "a raw echo to 192.0.2.1 left the seal, unanswered")


def _ping_sockets_open() -> bool:
    try:
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP).close()
        return True
    except OSError:
        return False


@pytest.mark.skipif(not _ping_sockets_open(), reason="this host opens no ping socket to the test's group")
def test_an_echo_that_leaves_fails_the_icmp_check():
    # Unsealed, the echo to loopback leaves and comes back with its data.
    check = seal_probe.icmp_check("127.0.0.1")
    assert not check.passed and "ping echo to 127.0.0.1 was answered with its data" in check.detail


def test_the_echo_goes_to_the_namespaces_default_gateway(tmp_path):
    route = tmp_path / "route"
    route.write_text("Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\n"
                     "eth0\t0040400A\t00000000\t0001\t0\t0\t0\t00C0FFFF\n"
                     "eth0\t00000000\t1440400A\t0003\t0\t0\t100\t00000000\n")
    assert seal_probe.default_gateway(route) == "10.64.64.20"
    route.write_text("Iface\tDestination\tGateway \tFlags\n")
    assert seal_probe.default_gateway(route) is None
    assert seal_probe.default_gateway(tmp_path / "missing") is None


def test_the_sandbox_runs_the_probe_as_code(tmp_path, monkeypatch):
    # The build's sandbox has no checkout: the module's own text, given to ``python -I -c``, is
    # the program, and its icmp and hold commands print one JSON line each.
    code = Path(seal_probe.__file__).read_text()
    done = subprocess.run([sys.executable, "-I", "-c", code, "icmp"], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert seal_probe.read_check(done.stdout.strip(), "icmp") is not None
    tcp = _free_port()
    argv = [sys.executable, "-I", "-c", code, "hold", "--inbound-tcp", str(tcp), "--inbound-udp",
            str(_free_port(avoid=(tcp,))), "--ready", str(tmp_path / "ready"), "--done", str(tmp_path / "done")]
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, text=True)
    _ready(tmp_path / "ready")
    (tmp_path / "done").touch()
    out, _ = proc.communicate(timeout=30)
    answer = seal_probe.held(out.strip())
    assert proc.returncode == 0 and answer is not None and answer[0].passed and answer[1] == ()


def test_a_report_reads_back_as_written(tmp_path):
    refusal, ingress = Check(passed=True, sec=0.25, detail="refused"), Check(passed=False, sec=1.5, detail="none")
    inbound, icmp = Check(passed=True, sec=0.5, detail="nothing reached"), Check(passed=True, sec=0.01, detail="held")
    ucounts = Check(passed=True, sec=0.0, detail="max_user_namespaces 484")
    caps = Check(passed=False, sec=0.0, detail="CapEff 0000000001000000")
    report = Report(refusal=refusal, ingress=ingress, inbound=inbound, icmp=icmp, ucounts=ucounts, caps=caps,
                    reached=("tcp", "udp"))
    write_report(tmp_path / "r.json", report)
    assert read_report(tmp_path / "r.json") == report
    report = Report(refusal=None, ingress=ingress, inbound=inbound, icmp=icmp, ucounts=ucounts, caps=caps, reached=())
    write_report(tmp_path / "r.json", report)
    assert read_report(tmp_path / "r.json") == report


_CHECK = '{"passed": true, "sec": 1, "detail": ""}'
_CHECKS = ('"ingress": ' + _CHECK + ', "inbound": ' + _CHECK + ', "icmp": ' + _CHECK + ', "ucounts": ' + _CHECK
           + ', "caps": ' + _CHECK)


@pytest.mark.parametrize("text", ["", "{", "[]", '{"refusal": null}', '{"refusal": null, "ingress": {"passed": 1}}',
                                  '{"refusal": null, "ingress": {"passed": true, "sec": "x", "detail": ""}}',
                                  # A report without its inbound, ICMP, ucounts or caps check is not the probe's.
                                  '{"refusal": null, "ingress": ' + _CHECK + ', "icmp": ' + _CHECK
                                  + ', "ucounts": ' + _CHECK + ', "caps": ' + _CHECK + ', "reached": []}',
                                  '{"refusal": null, "ingress": ' + _CHECK + ', "inbound": null, "icmp": ' + _CHECK
                                  + ', "ucounts": ' + _CHECK + ', "caps": ' + _CHECK + ', "reached": []}',
                                  '{"refusal": null, "ingress": ' + _CHECK + ', "inbound": ' + _CHECK + ', "icmp": '
                                  + _CHECK + ', "caps": ' + _CHECK + ', "reached": []}',
                                  '{"refusal": null, "ingress": ' + _CHECK + ', "inbound": ' + _CHECK + ', "icmp": '
                                  + _CHECK + ', "ucounts": ' + _CHECK + ', "reached": []}',
                                  '{"refusal": null, ' + _CHECKS + ', "caps": null, "reached": []}',
                                  # Nor one without what reached the inbound ports, or with another protocol.
                                  '{"refusal": null, ' + _CHECKS + "}",
                                  '{"refusal": null, ' + _CHECKS + ', "reached": ["icmp"]}',
                                  '{"refusal": null, ' + _CHECKS + ', "reached": ["tcp", "tcp"]}'])
def test_a_report_that_is_not_one_reads_as_none(tmp_path, text):
    (tmp_path / "r.json").write_text(text)
    assert read_report(tmp_path / "r.json") is None
    assert read_report(tmp_path / "missing.json") is None


def _sysctls(tmp_path: Path, monkeypatch, *, ephemeral: str, unprivileged: str) -> None:
    (tmp_path / "range").write_text(ephemeral)
    (tmp_path / "start").write_text(unprivileged)
    monkeypatch.setattr(seal_probe, "PORT_RANGE_SYSCTL", tmp_path / "range")
    monkeypatch.setattr(seal_probe, "UNPRIVILEGED_PORT_SYSCTL", tmp_path / "start")


def test_the_inbound_ports_lie_between_the_unprivileged_start_and_the_ephemeral_range(tmp_path, monkeypatch):
    # pasta forwards by default every port it may bind outside the ephemeral range: an
    # unprivileged one below the node's ip_local_port_range. The probe's ports are among those,
    # and none the seal's namespace holds already.
    low = random.randrange(20000, 30000)
    _sysctls(tmp_path, monkeypatch, ephemeral=f"{low + 50}\t60999\n", unprivileged=f"{low}\n")
    avoid = {low + 1, low + 7}
    for _ in range(20):
        tcp, udp = seal_probe.inbound_ports(avoid=avoid)
        assert tcp != udp and low <= tcp < low + 50 and low <= udp < low + 50 and not avoid & {tcp, udp}


def test_the_inbound_ports_avoid_the_seals_own_listeners(tmp_path, monkeypatch):
    # The interceptor's transparent listener and the served port are bound in the namespace
    # already; a probe that took one would fail before it reported anything.
    seal = SealPlan(policy=SealPolicy(network=NetworkPolicy.model_validate({"domains": {"mode": "allow", "entries": []}})),
                    index=False,
                    ingress_port=EGRESS_TRANSPARENT_PORT + 2)
    held = namespace_ports(seal)
    assert held == {EGRESS_TRANSPARENT_PORT, EGRESS_TRANSPARENT_PORT + 2}
    _sysctls(tmp_path, monkeypatch, ephemeral=f"{EGRESS_TRANSPARENT_PORT + 4}\t60999\n",
             unprivileged=f"{EGRESS_TRANSPARENT_PORT}\n")
    assert set(seal_probe.inbound_ports(avoid=held)) == {EGRESS_TRANSPARENT_PORT + 1, EGRESS_TRANSPARENT_PORT + 3}
    _sysctls(tmp_path, monkeypatch, ephemeral=f"{EGRESS_TRANSPARENT_PORT + 3}\t60999\n",
             unprivileged=f"{EGRESS_TRANSPARENT_PORT}\n")
    with pytest.raises(OSError, match="no two free ports"):
        seal_probe.inbound_ports(avoid=held)


def test_a_node_that_leaves_no_such_port_refuses_the_check(tmp_path, monkeypatch):
    _sysctls(tmp_path, monkeypatch, ephemeral="1024\t60999\n", unprivileged="1024\n")
    with pytest.raises(OSError, match="no two free ports"):
        seal_probe.inbound_ports(avoid={8000})


def test_the_node_addresses_are_its_interfaces_ipv4_addresses():
    addresses = seal_probe.node_addresses()
    assert "127.0.0.1" in addresses and len(set(addresses)) == len(addresses)
    for address in addresses:
        socket.inet_aton(address)


def test_a_knock_finds_a_port_closed_only_when_it_is_refused():
    # Over UDP too: a port nothing holds sends port unreachable back at once, while one something
    # holds and does not answer leaves the datagram unanswered.
    closed_tcp = _free_port()
    closed_udp = _free_port(avoid=(closed_tcp,))
    [tcp, udp] = seal_probe.knock_inbound(("127.0.0.1",), closed_tcp, closed_udp)
    assert (tcp.proto, tcp.outcome, tcp.refused) == ("tcp", "refused", True)
    assert (udp.proto, udp.outcome, udp.refused) == ("udp", "refused", True)
    with _listener() as listener, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as echo, \
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as silent:
        echo.bind(("127.0.0.1", 0))
        silent.bind(("127.0.0.1", 0))

        def answer() -> None:
            data, peer = echo.recvfrom(64)
            echo.sendto(data, peer)

        threading.Thread(target=answer, daemon=True).start()
        [tcp, udp] = seal_probe.knock_inbound(("127.0.0.1",), listener.getsockname()[1], echo.getsockname()[1])
        [_, quiet] = seal_probe.knock_inbound(("127.0.0.1",), listener.getsockname()[1], silent.getsockname()[1])
    assert (tcp.outcome, tcp.refused) == ("connected", False)
    assert (udp.outcome, udp.refused) == ("answered", False)
    assert (quiet.outcome, quiet.refused) == ("unanswered", False)


def test_a_knock_that_found_a_port_open_names_the_seal_or_the_node():
    # The probe saw the protocol arrive: the seal forwarded it. It saw nothing: something on the
    # node holds the port, as when another process took it between the pick and the knock.
    knocks = (seal_probe.Knock("10.0.0.5", "tcp", 20001, "connected"),
              seal_probe.Knock("10.0.0.5", "udp", 20002, "unanswered"),
              seal_probe.Knock("127.0.0.1", "tcp", 20001, "refused"))
    addresses = ("127.0.0.1", "10.0.0.5")
    assert seal_probe.knock_failures(addresses, knocks, ("udp",)) == [
        "tcp port 20001 on 10.0.0.5 is held on the node (connected): the probe saw nothing of it inside the seal",
        "the seal forwarded udp port 20002 on 10.0.0.5 (unanswered): the probe saw it inside the seal"]
    assert seal_probe.knock_failures(addresses, knocks[2:], ()) == []
    assert seal_probe.knock_failures(("127.0.0.1",), knocks[2:], ()) == [
        "the node has no address but loopback to knock on the inbound ports from"]


def test_the_probe_imports_only_the_standard_library():
    # It runs on the harness image as a script, with no site-packages to lean on.
    done = subprocess.run([sys.executable, "-I", "-S", seal_probe.__file__, "--help"], capture_output=True,
                          text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert Path(seal_probe.__file__).name == "seal_probe.py"


@pytest.mark.parametrize("policy, recorded", [
    (NetworkPolicy(domains=AccessList(mode="allow", entries=[])), True),
    (NetworkPolicy(domains=AccessList(mode="allow", entries=["example.com"])), True),
    # A policy that permits every host closes the probe's connection too, since the host does not
    # resolve; only the interceptor's record tells that refusal from the policy's.
    (NetworkPolicy(domains=AccessList(mode="deny", entries=[])), False),
], ids=["air-gap", "allow-list", "open"])
def test_the_interceptor_records_a_host_its_policy_refuses_and_no_other(tmp_path, policy, recorded):
    status = tmp_path / "status.json"
    interceptor = Interceptor(status, policy)
    try:
        check = refusal_check(interceptor.address, "hls-seal-probe.invalid", within_sec=10)
    finally:
        interceptor.close()
    assert check.passed  # closed unanswered either way
    refused = policy_refusals(status)
    assert (refused == ("hls-seal-probe.invalid",)) if recorded else (refused in (None, ()))
    if recorded:
        read = index_status(status)
        assert read is not None and (read.total, read.mid_stream, read.oversized, read.policy_refused,
                                     read.fetches_refused) == (0, 0, (), ("hls-seal-probe.invalid",), 0)
        assert abs(read.heartbeat - time.time()) < 60


def test_the_record_of_policy_refusals_stays_small_whatever_a_container_asks_for(tmp_path):
    # A sealed server asks for what hosts it likes: each is recorded once, a name no host can have
    # is not, and the record stops at 64 names. A short name of characters JSON writes in six
    # bytes each is too long as the status writes it.
    status = tmp_path / "status.json"
    interceptor = Interceptor(status, NetworkPolicy(domains=AccessList(mode="allow", entries=[])))
    addr, _, port = interceptor.address.rpartition(":")
    long = ".".join(["x" * 63] * 4) + ".invalid"
    escaped = "\xe9" * 50 + ".invalid"
    try:
        for host in ["h0.invalid", "h0.invalid", long, escaped, *(f"h{i}.invalid" for i in range(1, 80))]:
            with socket.create_connection((addr, int(port)), timeout=5) as c:
                c.sendall(f"GET / HTTP/1.1\r\nHost: {host}\r\n\r\n".encode("latin-1"))
                assert c.recv(1) == b""
    finally:
        interceptor.close()
    assert policy_refusals(status) == tuple(f"h{i}.invalid" for i in range(64))


#: The rest of a valid status, after its refusals by the policy.
_REST = '"fetches_refused": 0, "heartbeat": 1000.5'


@pytest.mark.parametrize("text, refused", [
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": [], ' + _REST + "}",
     ()),
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": ["a.example", '
     '"b.example"], ' + _REST + "}", ("a.example", "b.example")),
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": "a.example", '
     + _REST + "}", None),
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": [3], ' + _REST + "}",
     None),
    # The interceptor writes every key from its start, so a status without the refusals is not its.
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], ' + _REST + "}", None),
    # Nor is one without its fetch count or its heartbeat.
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": ["a.example"]}',
     None),
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": ["a.example"], '
     '"heartbeat": 1000.5}', None),
    # A record of refusals is read only with the rest of a valid status.
    ('{"upstream_errors": -1, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": ["a.example"], '
     + _REST + "}", None),
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [3], "policy_refused": ["a.example"], '
     + _REST + "}", None),
    ('{"upstream_errors": 0, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": ["a.example"], '
     '"fetches_refused": 0, "heartbeat": "1000.5"}', None),
    ('{"policy_refused": ["a.example"]}', None),
    ("{", None),
])
def test_the_record_of_policy_refusals_reads_only_as_the_interceptor_writes_it(tmp_path, text, refused):
    (tmp_path / "status.json").write_text(text)
    assert policy_refusals(tmp_path / "status.json") == refused
    assert policy_refusals(tmp_path / "missing.json") is None


class _AnyBind(socket.socket):
    """A socket whose bind always takes, as on a host where every port is free to the stage."""

    def bind(self, address):
        pass


def test_a_build_pick_starts_at_the_namespaces_floor_on_a_host_whose_floor_is_lower(tmp_path, monkeypatch):
    # The drive's probe binds as root of pasta's namespace, so the host's floor is the one that
    # counts; the build's sandbox command binds unprivileged in pasta's fresh namespace, whose floor
    # is the kernel's default whatever the host set.
    _sysctls(tmp_path, monkeypatch, ephemeral="1040\t60999\n", unprivileged="80\n")
    monkeypatch.setattr(socket, "socket", _AnyBind)
    build = [seal_probe.inbound_ports(avoid=(), floor=seal_probe.NAMESPACE_PORT_FLOOR) for _ in range(30)]
    assert seal_probe.NAMESPACE_PORT_FLOOR == 1024
    assert all(1024 <= port < 1040 for pair in build for port in pair)
    drive = [seal_probe.inbound_ports(avoid=()) for _ in range(30)]
    assert all(80 <= port < 1040 for pair in drive for port in pair) and min(min(pair) for pair in drive) < 1024


def test_the_node_addresses_are_only_those_of_interfaces_that_are_up(monkeypatch):
    # An interface that is down has no local route left: a knock on its address would leave the
    # node and time out.
    interfaces = {"lo": (0x1 | 0x8, "127.0.0.1"), "eth0": (0x1, "10.0.0.5"), "eth1": (0x0, "10.9.9.9")}

    def ioctl(fd, request, arg):
        name = arg[:16].rstrip(b"\0").decode()
        flags, address = interfaces[name]
        if request == seal_probe._SIOCGIFFLAGS:
            return arg[:16] + struct.pack("H", flags) + arg[18:]
        assert request == seal_probe._SIOCGIFADDR
        return arg[:20] + socket.inet_aton(address) + arg[24:]

    monkeypatch.setattr(socket, "if_nameindex", lambda: [(1, "lo"), (2, "eth0"), (3, "eth1")])
    monkeypatch.setattr(seal_probe.fcntl, "ioctl", ioctl)
    assert seal_probe.node_addresses() == ("127.0.0.1", "10.0.0.5")


#: Tests that stand a read-only file in for a limit the payload may not raise: root writes it anyway.
NOT_ROOT = pytest.mark.skipif(os.geteuid() == 0, reason="root is not held to file modes")


def _limits_dir(tmp_path: Path, *, raisable: bool = False, **limits: int) -> Path:
    """A stand-in for /proc/sys/user holding *limits*: files the probe may read, and write only
    when *raisable*, as a payload holding CAP_SYS_RESOURCE in the seal's user namespace could."""
    root = tmp_path / "user"
    root.mkdir()
    for name, value in limits.items():
        (root / name).write_text(f"{value}\n")
        (root / name).chmod(0o644 if raisable else 0o444)
    return root


@NOT_ROOT
def test_the_ucounts_check_passes_when_every_limit_reads_as_given_and_none_can_be_raised(tmp_path):
    # The probe reads, inside its seal, the limits the seal wrote, and compares them with the ones
    # the stage gave the seal: a cheap read, before any contestant code runs. Then it tries to raise
    # the first, with the capabilities the artifact will have, and the write must be refused.
    root = _limits_dir(tmp_path, max_user_namespaces=484, max_inotify_instances=64)
    check = seal_probe.ucounts_check("max_user_namespaces=484 max_inotify_instances=64", root=root)
    assert check.passed and "max_user_namespaces 484" in check.detail and "max_inotify_instances 64" in check.detail
    assert "raising max_user_namespaces to 485 was refused" in check.detail
    assert (root / "max_user_namespaces").read_text() == "484\n"


def test_a_limit_the_payload_can_raise_fails_the_ucounts_check(tmp_path):
    # A seal that leaves its payload CAP_SYS_RESOURCE lets it lift the limits once the probe has read
    # them: the write took.
    root = _limits_dir(tmp_path, raisable=True, max_user_namespaces=484, max_inotify_instances=64)
    check = seal_probe.ucounts_check("max_user_namespaces=484 max_inotify_instances=64", root=root)
    assert not check.passed
    assert "max_user_namespaces was raised from 484 to 485 inside the seal" in check.detail


@pytest.mark.parametrize("given, why", [
    ("max_user_namespaces=484 max_inotify_instances=8", "max_inotify_instances reads 64, not 8"),
    ("max_inotify_watches=8192", "max_inotify_watches could not be read"),
    ("max_user_namespaces=484 ../x=1", "not NAME=LIMIT"),
    ("max_user_namespaces=4a", "not NAME=LIMIT"),
], ids=["another-value", "no-file", "a-path", "not-digits"])
def test_a_limit_that_does_not_read_as_given_fails_the_ucounts_check(tmp_path, given, why):
    root = _limits_dir(tmp_path, max_user_namespaces=484, max_inotify_instances=64)
    check = seal_probe.ucounts_check(given, root=root)
    assert not check.passed and why in check.detail
    assert "raising" not in check.detail  # nothing is written once a limit reads otherwise


def test_a_seal_given_no_limits_passes_the_ucounts_check_with_nothing_read(tmp_path):
    # A node whose kernel keeps none of the counts gives its roles no limits.
    check = seal_probe.ucounts_check("", root=tmp_path / "absent")
    assert check.passed and check.detail == "the seal was given no limits"


@NOT_ROOT
def test_the_served_probe_reports_its_ucounts_check_and_fails_on_a_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "socket", _SealedSocket)
    real = 64
    monkeypatch.setattr(seal_probe, "UCOUNTS_ROOT", _limits_dir(tmp_path, max_inotify_instances=real))
    monkeypatch.setattr(seal_probe, "PROC_STATUS", _status(tmp_path))
    out = tmp_path / "report.json"
    codes: list[int] = []
    for given in (f"max_inotify_instances={real}", f"max_inotify_instances={real + 1}"):
        port = _free_port()
        tcp = _free_port(avoid=(port,))
        argv = ["served", "--port", str(port), "--out", str(out), "--check-sec", "5", "--inbound-tcp", str(tcp),
                "--inbound-udp", str(_free_port(avoid=(port, tcp))), "--ucounts", given]
        thread = threading.Thread(target=lambda: codes.append(seal_probe.main(argv)))
        thread.start()
        _knock(port, "n")
        _knock(port, "m")
        thread.join(10)
        report = read_report(out)
        assert report is not None and report.ucounts.passed is (given == f"max_inotify_instances={real}")
    assert codes == [0, 1]


def _status(tmp_path: Path, *, missing: str = "", **sets: str) -> Path:
    """A stand-in for /proc/self/status with each capability set at its hex value, 0 unless *sets*
    names it, and without the set *missing*."""
    values = {name: "0000000000000000" for name in seal_probe.CAP_SETS} | sets
    path = tmp_path / "status"
    path.write_text("Name:\tpython3\n" + "".join(f"{name}:\t{value}\n" for name, value in values.items()
                                                 if name != missing) + "NoNewPrivs:\t0\n")
    return path


def test_the_caps_check_passes_when_the_probe_holds_no_capability_in_any_set(tmp_path):
    # The probe runs with the capabilities the artifact will have: none, so none it could take either.
    check = seal_probe.caps_check(_status(tmp_path))
    assert check.passed and check.detail == "no capability in any set: CapInh, CapPrm, CapEff, CapBnd and CapAmb are 0"


@pytest.mark.parametrize("held", ["CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"])
def test_a_capability_in_any_set_fails_the_caps_check(tmp_path, held):
    # Root of the seal's user namespace takes back on exec what its bounding and inheritable sets
    # keep, so an empty effective set alone proves nothing.
    check = seal_probe.caps_check(_status(tmp_path, **{held: "0000000001000000"}))
    assert not check.passed and check.detail == f"the probe holds capabilities: {held} 0000000001000000"


def test_a_status_that_does_not_show_every_set_fails_the_caps_check(tmp_path):
    check = seal_probe.caps_check(_status(tmp_path, missing="CapAmb"))
    assert not check.passed and "CapAmb" in check.detail
    check = seal_probe.caps_check(tmp_path / "absent")
    assert not check.passed and "could not be read" in check.detail


def test_a_probe_that_holds_a_capability_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(socket, "socket", _SealedSocket)
    monkeypatch.setattr(seal_probe, "PROC_STATUS", _status(tmp_path, CapBnd="000001ffffffffff"))
    out = tmp_path / "report.json"
    port = _free_port()
    tcp = _free_port(avoid=(port,))
    argv = ["served", "--port", str(port), "--out", str(out), "--check-sec", "5", "--inbound-tcp", str(tcp),
            "--inbound-udp", str(_free_port(avoid=(port, tcp))), "--ucounts", ""]
    codes: list[int] = []
    thread = threading.Thread(target=lambda: codes.append(seal_probe.main(argv)))
    thread.start()
    _knock(port, "n")
    _knock(port, "m")
    thread.join(10)
    report = read_report(out)
    assert codes == [1] and report is not None and not report.caps.passed
    assert report.ingress.passed and report.inbound.passed and report.icmp.passed and report.ucounts.passed


def test_the_seal_probes_check_time_has_one_owner_which_both_stages_take():
    """The seconds each check of a seal probe may take are the probe module's, which the drive's
    probe gets as ``--check-sec`` and the build's seal probe as each command's time; no stage
    keeps a figure of its own."""
    from hls.stages import agent_build, deploy_drive

    assert not hasattr(agent_build, "SEAL_CHECK_SEC") and not hasattr(deploy_drive, "SEAL_CHECK_SEC")
