"""A build seal that reaches no host but through its index refuses its container's DNS, so a
build hook cannot carry data out in the names it looks up; its interceptor then resolves the
index's upstream itself, over sockets that carry its mark, the only DNS the firewall lets out.

The firewall is read from the fake ``iptables`` the real seal script runs; the resolver is
run against a nameserver on the loopback."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from pathlib import Path

import pytest

from hls.egress import proxy as proxy_mod
from hls.egress.policy import AccessList, EgressPosture, NetworkPolicy, SealPolicy
from hls.egress.proxy import EgressProxy
from hls.substrate.apptainer.egress import CLOSED_DNS_VAR
from hls.substrate.seal import closes_dns
from hls.substrate.container import SealPlan
from tests.substrate.test_egress_seal import DENY, POLICY, SEAL, _seal

FILTERED = SealPolicy(network=SEAL.network.model_copy(
    update={"domains": AccessList(mode="allow", entries=["api.example.org"])}))


def _rules(tmp_path: Path, **step_env: str) -> tuple[dict, list[list[str]], str]:
    """Run the real seal with *step_env*; what happened, the iptables rules it installed and
    the HLS_PROXY_OWN_DNS its interceptor was started with."""
    log, seen = tmp_path / "iptables.log", tmp_path / "interceptor.env"
    out = _seal(tmp_path, ["run"], step_env={"FAKE_IPTABLES_LOG": str(log), "FAKE_INTERCEPTOR_ENV": str(seen),
                                             **step_env})
    rules = [json.loads(line) for line in log.read_text().splitlines()]
    return out, rules, seen.read_text()


def _dns53(rules: list[list[str]]) -> list[list[str]]:
    return [r for r in rules if "--dport" in r and r[r.index("--dport") + 1] == "53"]


@pytest.mark.parametrize("policy, index, closed", [
    (SEAL, True, True), (SEAL, False, False), (FILTERED, True, False),
], ids=["air-gapped-index", "air-gapped-no-index", "filtered-index"])
def test_only_an_index_on_an_air_gapped_policy_closes_dns(policy, index, closed):
    assert closes_dns(SealPlan(policy=policy, index=index, deny_list=DENY if index else ())) is closed


def test_no_seal_with_an_index_is_open():
    """A seal with an index carries a package rule, which an open domain policy cannot hold
    (``NetworkPolicy``): the open case of the DNS decision cannot be planned at all."""
    with pytest.raises(ValueError, match="package policy requires a restricted domain policy"):
        SEAL.network.model_copy(update={"domains": AccessList(mode="deny", entries=[])})
    open_network = NetworkPolicy.model_validate({"domains": {"mode": "deny", "entries": []}})
    assert not closes_dns(SealPlan(policy=SealPolicy(network=open_network), index=False))


@pytest.mark.parametrize("posture", list(EgressPosture))
def test_the_seal_closes_dns_exactly_when_the_policys_posture_rule_says_air_gap(monkeypatch, posture):
    """The seal asks the policy's posture rule (``EgressPosture.of``) rather than reading the
    domain list itself, so a change to the rule carries the seal's DNS decision with it: told
    that every list is *posture*, a seal with an index closes its DNS for an air-gapped, a
    filtered and an open policy alike, and only when *posture* is the air gap."""
    monkeypatch.setattr(EgressPosture, "of", classmethod(lambda cls, domains: posture))
    for policy in (SEAL, FILTERED):
        assert closes_dns(SealPlan(policy=policy, index=True, deny_list=DENY)) is (
            posture is EgressPosture.AIR_GAP), policy.network.domains


def test_a_closed_seal_drops_every_dns_query_but_the_interceptors_own(tmp_path):
    """No rule lets the payload's DNS out: before the loopback's allowance, every UDP query to
    port 53 without the interceptor's mark is dropped, and no resolver gets an allowance."""
    out, rules, own_dns = _rules(tmp_path, **{CLOSED_DNS_VAR: "closed"})
    assert out["rc"] == 0 and out["runs"] == 1, out["stderr"]
    assert own_dns == "1"
    mark, drop, loopback = (rules.index(r) for r in (
        ["-A", "OUTPUT", "-m", "mark", "--mark", rules[0][rules[0].index("--mark") + 1], "-j", "ACCEPT"],
        ["-A", "OUTPUT", "-p", "udp", "--dport", "53", "-j", "DROP"],
        ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"]))
    assert mark < drop < loopback
    assert _dns53(rules) == [["-A", "OUTPUT", "-p", "udp", "--dport", "53", "-j", "DROP"]]


def test_an_open_seal_lets_dns_out_to_its_resolvers(tmp_path):
    resolvers = [line.split()[1] for line in Path("/etc/resolv.conf").read_text().splitlines()
                 if line.startswith("nameserver") and ":" not in line.split()[1]]
    out, rules, own_dns = _rules(tmp_path)
    assert out["rc"] == 0 and own_dns == ""
    assert _dns53(rules) == [["-A", "OUTPUT", "-p", "udp", "-d", ns, "--dport", "53", "-j", "ACCEPT"]
                             for ns in resolvers]


@pytest.mark.parametrize("closed", [True, False], ids=["closed-dns", "open-dns"])
def test_the_seal_redirects_all_tcp_to_its_interceptor_and_drops_every_other_datagram(tmp_path, closed):
    """The nat table lets the interceptor's own marked connections and the loopback through and
    redirects every other TCP connection to the port the interceptor listens on; the filter
    table lets out the interceptor's own datagrams, the loopback's and the DNS the seal allows,
    and its last rule drops every other UDP datagram; IPv6 is dropped but for the interceptor's
    own and the loopback's."""
    v6_log, started_with = tmp_path / "ip6tables.log", tmp_path / "interceptor.json"
    step_env = {"FAKE_IP6TABLES_LOG": str(v6_log), "FAKE_INTERCEPTOR_VARIABLES": str(started_with)}
    out, rules, _ = _rules(tmp_path, **step_env, **({CLOSED_DNS_VAR: "closed"} if closed else {}))
    assert out["rc"] == 0 and out["runs"] == 1, out["stderr"]
    interceptor = json.loads(started_with.read_text())
    mark, port = interceptor["HLS_PROXY_FWMARK"], interceptor["HLS_PROXY_TRANSPARENT_PORT"]
    assert [r[2:] for r in rules if r[:2] == ["-t", "nat"]] == [
        ["-A", "OUTPUT", "-m", "mark", "--mark", mark, "-j", "RETURN"],
        ["-A", "OUTPUT", "-o", "lo", "-j", "RETURN"],
        ["-A", "OUTPUT", "-d", "127.0.0.0/8", "-j", "RETURN"],
        ["-A", "OUTPUT", "-p", "tcp", "-j", "REDIRECT", "--to-ports", port]]
    resolvers = [line.split()[1] for line in Path("/etc/resolv.conf").read_text().splitlines()
                 if line.startswith("nameserver") and ":" not in line.split()[1]]
    loopback = ["-A", "OUTPUT", "-o", "lo", "-j", "ACCEPT"]
    if closed:
        dns = [["-A", "OUTPUT", "-p", "udp", "--dport", "53", "-j", "DROP"], loopback]
    else:
        dns = [loopback, *(["-A", "OUTPUT", "-p", "udp", "-d", ns, "--dport", "53", "-j", "ACCEPT"]
                           for ns in resolvers)]
    assert [r for r in rules if r[:2] != ["-t", "nat"]] == [
        ["-A", "OUTPUT", "-m", "mark", "--mark", mark, "-j", "ACCEPT"], *dns,
        ["-A", "OUTPUT", "-p", "udp", "-j", "DROP"],
        ["-A", "OUTPUT", "-p", "icmp", "-j", "DROP"]]
    assert [json.loads(line) for line in v6_log.read_text().splitlines()] == [
        ["-A", "OUTPUT", "-m", "mark", "--mark", mark, "-j", "RETURN"],
        ["-A", "OUTPUT", "-o", "lo", "-j", "RETURN"],
        ["-A", "OUTPUT", "-j", "DROP"]]


def test_a_seal_dns_setting_it_does_not_know_never_starts_pasta(tmp_path):
    out = _seal(tmp_path, ["run"], step_env={CLOSED_DNS_VAR: "open"})
    assert out["rc"] == 2 and out["runs"] == 0 and out["calls"] == []
    assert CLOSED_DNS_VAR in out["stderr"]


# The interceptor's own resolver.


def _name(name: str) -> bytes:
    return b"".join(bytes([len(label)]) + label.encode() for label in name.split(".")) + b"\0"


def _response(query: bytes, rcode: int, records: list[tuple[int, bytes]], *, truncated: bool = False) -> bytes:
    """A response to *query* with *records* (type, data) in its answer section, each owned by a
    compressed pointer to the question's name."""
    flags = 0x8180 | rcode | (0x0200 if truncated else 0)
    head = query[:2] + flags.to_bytes(2, "big") + b"\x00\x01" + len(records).to_bytes(2, "big") + b"\0\0\0\0"
    body = b"".join(b"\xc0\x0c" + rtype.to_bytes(2, "big") + b"\x00\x01\x00\x00\x00\x3c"
                    + len(data).to_bytes(2, "big") + data for rtype, data in records)
    return head + query[12:] + body


class Nameserver:
    """A nameserver on the loopback, over UDP and TCP on one port, that answers from *zone*
    (name to addresses; a name with no entry is NXDOMAIN), puts a CNAME before every answer,
    truncates UDP answers for the names in *truncate*, and keeps every name it was asked."""

    def __init__(self, zone: dict[str, list[str]], *, truncate: tuple[str, ...] = (), silent: bool = False) -> None:
        self.zone, self.truncate, self.silent = zone, truncate, silent
        self.asked: list[tuple[str, str]] = []
        # A free UDP port may be taken for TCP by another process: try again until both bind.
        for _ in range(50):
            self.udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp.bind(("127.0.0.1", 0))
            self.port = self.udp.getsockname()[1]
            self.tcp = socket.socket()
            try:
                self.tcp.bind(("127.0.0.1", self.port))
                break
            except OSError:
                self.udp.close()
                self.tcp.close()
        else:
            raise RuntimeError("no loopback port is free for both UDP and TCP")
        self.tcp.listen()
        for loop in (self._udp, self._tcp):
            threading.Thread(target=loop, daemon=True).start()

    def _answer(self, query: bytes, transport: str) -> bytes:
        at, labels = 12, []
        while query[at]:
            labels.append(query[at + 1:at + 1 + query[at]].decode())
            at += 1 + query[at]
        name = ".".join(labels)
        self.asked.append((transport, name))
        if name not in self.zone:
            return _response(query, 3, [])
        records = [(5, _name("alias." + name))] + [(1, socket.inet_aton(a)) for a in self.zone[name]]
        if transport == "udp" and name in self.truncate:
            return _response(query, 0, [], truncated=True)
        return _response(query, 0, records)

    def _udp(self) -> None:
        while True:
            try:
                query, peer = self.udp.recvfrom(4096)
            except OSError:
                return
            if not self.silent:
                self.udp.sendto(self._answer(query, "udp"), peer)

    def _tcp(self) -> None:
        while True:
            try:
                conn, _ = self.tcp.accept()
            except OSError:
                return
            with conn:
                size = int.from_bytes(conn.recv(2), "big")
                answer = self._answer(conn.recv(size), "tcp")
                conn.sendall(len(answer).to_bytes(2, "big") + answer)

    def close(self) -> None:
        self.udp.close()
        self.tcp.close()


@pytest.fixture
def resolver(tmp_path, monkeypatch):
    """The marked resolver pointed at a loopback nameserver and a hosts file of its own;
    returns a function that starts the nameserver."""
    servers: list[Nameserver] = []

    def start(zone: dict[str, list[str]], *, search: str = "", hosts: str = "", **kwargs) -> Nameserver:
        server = Nameserver(zone, **kwargs)
        servers.append(server)
        (tmp_path / "resolv.conf").write_text(f"nameserver 127.0.0.1\n{search}")
        (tmp_path / "hosts").write_text(hosts)
        monkeypatch.setattr(proxy_mod, "_RESOLV_CONF", str(tmp_path / "resolv.conf"))
        monkeypatch.setattr(proxy_mod, "_HOSTS", str(tmp_path / "hosts"))
        monkeypatch.setattr(proxy_mod, "_DNS_PORT", server.port)
        monkeypatch.setattr(proxy_mod, "_DNS_TIMEOUT_SEC", 0.2)
        return server

    yield start
    for server in servers:
        server.close()


def test_a_name_resolves_to_its_addresses_past_the_cname(resolver):
    server = resolver({"pypi.org": ["151.101.0.223", "151.101.64.223"]})
    assert proxy_mod._resolve_marked("pypi.org") == ["151.101.0.223", "151.101.64.223"]
    assert server.asked == [("udp", "pypi.org")]


def test_a_truncated_answer_is_asked_again_over_tcp(resolver):
    server = resolver({"files.pythonhosted.org": ["151.101.1.63"]}, truncate=("files.pythonhosted.org",))
    assert proxy_mod._resolve_marked("files.pythonhosted.org") == ["151.101.1.63"]
    assert server.asked == [("udp", "files.pythonhosted.org"), ("tcp", "files.pythonhosted.org")]


def test_search_domains_come_first_for_a_short_name_and_last_for_a_dotted_one(resolver):
    server = resolver({"proxy.site.example": ["10.1.2.3"], "pypi.org.site.example": ["10.9.9.9"],
                       "pypi.org": ["151.101.0.223"]}, search="search site.example\n")
    assert proxy_mod._resolve_marked("proxy") == ["10.1.2.3"]
    assert proxy_mod._resolve_marked("pypi.org") == ["151.101.0.223"]
    assert proxy_mod._resolve_marked("pypi.org.") == ["151.101.0.223"]
    assert server.asked == [("udp", "proxy.site.example"), ("udp", "pypi.org"), ("udp", "pypi.org")]


def test_the_hosts_file_answers_without_a_query(resolver):
    server = resolver({}, hosts="10.0.0.7 mirror.site mirror # the site's index mirror\n")
    assert proxy_mod._resolve_marked("mirror.site") == ["10.0.0.7"] == proxy_mod._resolve_marked("MIRROR")
    assert server.asked == []


def test_a_name_with_no_address_and_a_silent_nameserver_fail_as_libc_would(resolver):
    resolver({})
    with pytest.raises(socket.gaierror) as caught:
        proxy_mod._resolve_marked("nowhere.example")
    assert caught.value.errno == socket.EAI_NONAME
    resolver({"pypi.org": ["151.101.0.223"]}, silent=True)
    with pytest.raises(socket.gaierror) as caught:
        proxy_mod._resolve_marked("pypi.org")
    assert caught.value.errno == socket.EAI_AGAIN


def test_getaddrinfo_answers_an_address_without_a_query_and_a_name_through_the_resolver(resolver):
    server = resolver({"pypi.org": ["151.101.0.223"]})
    assert proxy_mod._marked_getaddrinfo("127.0.0.1", 443, type=socket.SOCK_STREAM)[0][4] == ("127.0.0.1", 443)
    entries = proxy_mod._marked_getaddrinfo("pypi.org", "https", socket.AF_INET, socket.SOCK_STREAM)
    assert [e[4] for e in entries] == [("151.101.0.223", 443)] and server.asked == [("udp", "pypi.org")]
    with pytest.raises(socket.gaierror):
        proxy_mod._marked_getaddrinfo("pypi.org", 443, socket.AF_INET6)


def test_the_resolver_opens_its_sockets_where_the_interceptors_mark_is_stamped(resolver, monkeypatch):
    """``_install_upstream_mark`` replaces ``socket.socket``; every query must go through it."""
    resolver({"pypi.org": ["151.101.0.223"]}, truncate=("pypi.org",))
    made = []

    class Recording(socket.socket):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            # The nameserver's own thread accepts through the same class.
            if threading.current_thread() is threading.main_thread():
                made.append(self.type)

    monkeypatch.setattr(socket, "socket", Recording)
    assert proxy_mod._resolve_marked("pypi.org") == ["151.101.0.223"]
    assert made == [socket.SOCK_DGRAM, socket.SOCK_STREAM]


async def test_an_installed_resolver_serves_asyncios_lookups(resolver, monkeypatch):
    resolver({"index.test": ["127.0.0.1"]})
    monkeypatch.setattr(socket, "getaddrinfo", socket.getaddrinfo)
    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        proxy_mod._install_marked_resolver()
        assert socket.getaddrinfo is proxy_mod._marked_getaddrinfo
        _, writer = await asyncio.open_connection("index.test", port)
        writer.close()
    finally:
        server.close()


def test_the_interceptor_resolves_for_itself_only_with_its_mark(monkeypatch):
    policy = SEAL.network.model_dump(mode="json")
    with pytest.raises(ValueError, match="own_dns needs upstream_fwmark"):
        EgressProxy(policy, host="127.0.0.1", port=0, own_dns=True)
    monkeypatch.setenv("HLS_EGRESS_POLICY", POLICY)
    monkeypatch.setenv("HLS_PROXY_HOST", "127.0.0.1")
    monkeypatch.setenv("HLS_PROXY_PORT", "0")
    monkeypatch.setenv("HLS_PROXY_FWMARK", "0x4853")
    monkeypatch.setenv("HLS_PROXY_OWN_DNS", "1")
    assert EgressProxy.from_env()._own_dns
    monkeypatch.delenv("HLS_PROXY_OWN_DNS")
    assert not EgressProxy.from_env()._own_dns


async def test_serving_installs_the_resolver_after_the_mark(tmp_path, monkeypatch):
    order = []
    monkeypatch.setattr(proxy_mod, "_install_upstream_mark", lambda mark: order.append("mark"))
    monkeypatch.setattr(proxy_mod, "_install_marked_resolver", lambda: order.append("resolver"))
    proxy = EgressProxy(SEAL.network.model_dump(mode="json"), host="127.0.0.1", port=0, upstream_fwmark=0x4853,
                        own_dns=True,
                        status_file=str(tmp_path / "status.json"))
    task = asyncio.create_task(proxy.serve_forever())
    for _ in range(100):
        if (tmp_path / "status.json").exists():
            break
        await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert order == ["mark", "resolver"]
