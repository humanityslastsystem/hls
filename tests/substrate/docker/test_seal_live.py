"""The Docker seal against the Docker daemon of the machine the tests run on: a real gateway
(``hls.substrate.docker.seal``) and a small payload container in its namespace, case by case
through the egress matrix, and the seal probe (``hls.egress.seal_probe``) in a sealed payload.

Opt-in, as every ``integration`` test (``HLS_RUN_SLOW=1``), and skipped where no Docker Engine
serving ``API_VERSION`` answers on the default socket. The tests make one image, the pinned
Ubuntu base with iptables, Python, pip and curl from the Ubuntu archive, by running and
committing a container (no build cache is left), which needs the network; with
``HLS_DOCKER_STEP_ARCHIVE`` naming a step image archive (the ``image.oci.tar`` of a store's
``images/step-*`` entry, ``hls.substrate.docker.step_image``) the gateways run on that image
instead. The cases that reach PyPI or a public host, and the controls
that show a probe would see a broken seal, need the network too and are skipped without it.
Everything the tests create carries ``hls.test=docker-seal`` and a label of its own run, and is
removed at the end, also when a test fails; the base image only when the tests pulled it."""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest

import hls
from hls.egress.policy import PROBE_DIAL, NetworkPolicy, SealPolicy, refused_host
from hls.egress.seal_probe import CHECK_SEC, inbound_ports, knock_failures, knock_inbound, node_addresses, read_report
from hls.substrate.container import SealPlan
from hls.substrate.docker import seal as dseal
from hls.substrate.docker.api import BindMount, ContainerConfig, DockerClient, DockerError, NotFound
from hls.substrate.seal import (
    EGRESS_FWMARK,
    EGRESS_INDEX_PORT,
    SealFiles,
    client_env,
    closes_dns,
    index_status,
    ingress_port,
    namespace_ports,
    payload_ending,
    policy_refusals,
    seal_files,
)

pytestmark = pytest.mark.integration

LANE = "docker-seal"
RUN = uuid.uuid4().hex[:12]
LABELS = {"hls.test": LANE, "hls.test.run": RUN}
#: The pinned public base of the test image: the step image's own base.
BASE = "ubuntu:24.04@sha256:9c7f200c07e3d4a03e7249d2c16bf66a51120fb876aa2242cf31451412543289"
PACKAGES = ("iptables", "python3", "python3-pip", "curl", "ca-certificates")
TAG = f"hls-test/docker-seal:{RUN}"
GIB = 1 << 30
CHECKOUT = "/checkout"

DENIED_DOMAIN = "example.com"
DENIED_PACKAGE = "tensorflow"
ALLOWED_PACKAGE = "six"
UNLISTED_DOMAIN = "example.org"
DENY = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "deny", "entries": [DENIED_DOMAIN, f"*.{DENIED_DOMAIN}"]},
     "packages": {"mode": "deny", "entries": [DENIED_PACKAGE]}}))
#: The allow mode a seal can hold: of domains. A seal with an index refuses its producer's deny list
#: by a package-deny rule (``SealPlan``), so its package rule is never an allow list.
ALLOW = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "allow", "entries": [DENIED_DOMAIN, f"*.{DENIED_DOMAIN}"]},
     "packages": {"mode": "deny", "entries": [DENIED_PACKAGE]}}))
#: A build seal's air gap, which also refuses the payload's DNS (``closes_dns``).
AIR_GAP = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "deny", "entries": [DENIED_PACKAGE]}}))
SERVED = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "deny", "entries": [DENIED_DOMAIN, f"*.{DENIED_DOMAIN}"]}}))
SERVED_PORT = 8000


def _docker_cli(*args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def _online() -> bool:
    try:
        with socket.create_connection(("pypi.org", 443), timeout=5):
            return True
    except OSError:
        return False


ONLINE = _online()
needs_network = pytest.mark.skipif(not ONLINE, reason="needs the network: pypi.org is not reachable from here")


@pytest.fixture(scope="module")
def docker() -> Iterator[DockerClient]:
    if os.environ.get("HLS_RUN_SLOW") != "1":
        pytest.skip("live Docker tests are opt-in: set HLS_RUN_SLOW=1")
    client = DockerClient()
    try:
        client.check()
    except DockerError as e:
        client.close()
        pytest.skip(f"no Docker Engine to test against: {e}")
    try:
        yield client
    finally:
        leftover = client.list_containers(labels=LABELS)
        for row in leftover:
            client.remove_container(row.id, force=True)
        client.close()
        assert not leftover, f"containers left by a test: {[r.id for r in leftover]}"


def _has_image(ref: str) -> bool:
    return _docker_cli("image", "inspect", ref, timeout=60).returncode == 0


@pytest.fixture(scope="module")
def image(docker: DockerClient) -> Iterator[str]:
    """The test image's ID: the pinned base, its packages installed by a container that is then
    committed and removed."""
    pulled = not _has_image(BASE)
    builder = f"hls-test-docker-seal-image-{RUN}"
    try:
        if pulled and _docker_cli("pull", "-q", BASE).returncode != 0:
            pytest.skip(f"needs the network: {BASE} cannot be pulled")
        script = ("export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && "
                  f"apt-get install -y -qq --no-install-recommends {' '.join(PACKAGES)} > /dev/null && apt-get clean")
        labels = [arg for k, v in LABELS.items() for arg in ("--label", f"{k}={v}")]
        made = _docker_cli("run", "--name", builder, *labels, BASE, "sh", "-c", script)
        if made.returncode != 0:
            pytest.skip(f"needs the network: the test image's packages did not install: {made.stderr[-500:]}")
        changes = [arg for k, v in LABELS.items() for arg in ("--change", f"LABEL {k}={v}")]
        committed = _docker_cli("commit", *changes, builder, TAG)
        assert committed.returncode == 0, committed.stderr
        assert _docker_cli("rm", builder, timeout=120).returncode == 0
        yield docker.inspect_image(TAG).id
    finally:
        _docker_cli("rm", "-f", builder, timeout=120)
        if _has_image(TAG):
            docker.remove_image(TAG)
        if pulled and _has_image(BASE):
            _docker_cli("rmi", BASE, timeout=120)


@pytest.fixture(scope="module")
def gateway_image(docker: DockerClient, image: str) -> Iterator[str]:
    """The image the gateways run: the step image from ``HLS_DOCKER_STEP_ARCHIVE`` when it names
    one, else the test image, which holds what a gateway needs (bash, iptables, setpriv)."""
    archive = os.environ.get("HLS_DOCKER_STEP_ARCHIVE")
    if not archive:
        yield image
        return
    index = json.loads(subprocess.run(["tar", "-xOf", archive, "index.json"], capture_output=True, check=True,
                                      timeout=120).stdout)
    [manifest] = [m["digest"] for m in index["manifests"]]
    existed = True
    try:
        docker.inspect_image(manifest)
    except NotFound:
        existed = False
    if not existed:
        docker.load_image(Path(archive), timeout=300)
    try:
        yield docker.inspect_image(manifest).id
    finally:
        if not existed:
            docker.remove_image(manifest)


def code_mounts() -> tuple[BindMount, ...]:
    """The read-only binds that let this interpreter import ``hls`` in a container at the same
    paths: the checkout, the venv and the directory its ``pyvenv.cfg`` names as home's parent."""
    checkout = Path(hls.__file__).resolve().parents[2]
    dirs = [checkout, Path(sys.prefix)]
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "home":
                dirs.append(Path(value.strip()).parent)
    else:
        dirs.append(Path(sys.base_prefix))
    return tuple(BindMount(source=d, target=str(d), read_only=True) for d in dict.fromkeys(dirs))


def _user() -> str:
    return f"{os.getuid()}:{os.getgid()}"


def _cpus() -> tuple[int, ...]:
    return tuple(sorted(os.sched_getaffinity(0)))[:4]


@dataclass
class Sealed:
    gateway: dseal.Gateway
    files: SealFiles
    work: Path
    payload: str


@contextmanager
def sealed(docker: DockerClient, gateway_image: str, image: str, root: Path, seal: SealPlan, script: str, *,
           env: dict[str, str] | None = None) -> Iterator[Sealed]:
    """A gateway for *seal* and a payload running the shell *script* in its namespace, as the
    launcher pairs them: the payload has /work (``root``/work) and the checkout read-only at
    ``CHECKOUT``. On leaving, the payload is waited for, its ending recorded, and both removed."""
    state, work = root / "seal", root / "work"
    state.mkdir(parents=True)
    work.mkdir(parents=True)
    (work / "tmp").mkdir()
    files = seal_files(state)
    config = dseal.gateway_config(seal, files, image=gateway_image, python=sys.executable, user=_user(),
                                  code=code_mounts(), cpuset=_cpus(), memory_bytes=2 * GIB, labels=LABELS)
    gateway = dseal.open_gateway(docker, seal, files, config, name=f"hls-test-seal-gw-{uuid.uuid4().hex[:8]}")
    payload = None
    exit_code = None
    try:
        checkout = Path(hls.__file__).resolve().parents[2]
        payload_env = {**client_env(seal), "HOME": "/work", "TMPDIR": "/work/tmp", **(env or {})}
        cfg = ContainerConfig(image=image, argv=("/bin/sh", "-c", script), user=_user(), labels=LABELS,
                              network_mode="none", cpuset=_cpus(), memory_bytes=GIB, pids_limit=512,
                              env=payload_env,
                              mounts=(BindMount(source=work, target="/work", read_only=False),
                                      BindMount(source=checkout, target=CHECKOUT, read_only=True)))
        payload = docker.create_container(dseal.payload_config(cfg, gateway))
        dseal.mark_started(gateway)
        docker.start_container(payload)
        yield Sealed(gateway=gateway, files=files, work=work, payload=payload)
        ended = docker.wait_container(payload, timeout=300)
        assert ended is not None, "the payload did not end within 300 s"
        exit_code = ended.status_code
    finally:
        dseal.close_gateway(docker, gateway, exit_code=exit_code)
        if payload is not None:
            docker.remove_container(payload, force=True)


def interceptors(proc: str) -> list[int]:
    """The processes under *proc* that run the interceptor, ``<python> -m hls.egress.proxy``: the
    host's half of ``PROCS_PROBE``."""
    found = []
    for pid in os.listdir(proc):
        if not pid.isdigit():
            continue
        try:
            with open(f"{proc}/{pid}/cmdline", "rb") as f:
                args = f.read().split(b"\0")
        except OSError:
            continue
        if args[1:3] == [b"-m", b"hls.egress.proxy"]:
            found.append(int(pid))
    return found


def _ppid(pid: int) -> int:
    return int(Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[1])


def _results(work: Path) -> dict[str, str]:
    """The ``name=value`` lines the payload wrote to /work/results."""
    lines = (work / "results").read_text().splitlines() if (work / "results").exists() else []
    return dict(line.split("=", 1) for line in lines if "=" in line)


# Each probe writes one name=value line to /work/results; a value is a status or a word.
_PY = "python3 -I -c"
UDP_PROBE = r"""
import socket, sys
q = b'\xab\xcd\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01'
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.settimeout(5)
try:
    s.sendto(q, (sys.argv[1], int(sys.argv[2]))); s.recvfrom(2048); print('answered')
except OSError as e:
    print('blocked:' + type(e).__name__)
"""
MARK_PROBE = r"""
import socket, ssl, sys
host = sys.argv[1]
r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    r.setsockopt(socket.SOL_SOCKET, socket.SO_MARK, int(sys.argv[2]))
except OSError as e:
    print('mark-refused:' + e.strerror.replace(' ', '-')); sys.exit()
r.settimeout(15)
try:
    r.connect((host, 443))
    s = ssl.create_default_context().wrap_socket(r, server_hostname=host)
    s.sendall(b'GET / HTTP/1.1\r\nHost: ' + host.encode() + b'\r\nConnection: close\r\n\r\n')
    print('reached' if s.recv(16).startswith(b'HTTP/') else 'no-reply')
except Exception as e:
    print('blocked:' + type(e).__name__)
"""
ICMP_PROBE = r"""
import socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_ICMP)
s.settimeout(3)
try:
    s.sendto(struct.pack('!BBHHH', 8, 0, 0, 0, 1) + b'hls', ('8.8.8.8', 0)); s.recvfrom(64); print('answered')
except OSError as e:
    print('blocked:' + type(e).__name__)
"""
TCP_PROBE = r"""
import socket, sys
s = socket.socket(); s.settimeout(5)
try:
    s.connect((sys.argv[1], int(sys.argv[2]))); s.sendall(sys.argv[3].encode() + b'\r\n\r\n')
    data = s.recv(64)
    print('answered' if data else 'closed')
except OSError as e:
    print('blocked:' + type(e).__name__)
"""
PROCS_PROBE = r"""
import os
found = []
for pid in os.listdir('/proc'):
    if pid.isdigit():
        try:
            args = open(f'/proc/{pid}/cmdline', 'rb').read().split(b'\0')
        except OSError:
            continue
        if args[1:3] == [b'-m', b'hls.egress.proxy']:
            found.append(pid)
print('interceptor-visible' if found else 'own-only')
"""


def _py(name: str, code: str, *args: str) -> str:
    quoted = code.replace("'", "'\"'\"'")
    return f"echo {name}=$({_PY} '{quoted}' {' '.join(args)}) >> /work/results\n"


def _rc(name: str, command: str) -> str:
    return f"{command} > /work/{name}.log 2>&1; echo {name}=$? >> /work/results\n"


def _code(name: str, command: str) -> str:
    return f"echo {name}=$({command} 2> /work/{name}.log) >> /work/results\n"


FORGED = base64.urlsafe_b64encode(
    f"https://files.pythonhosted.org/packages/aa/bb/cc/{DENIED_PACKAGE}-9.9.9-py3-none-any.whl".encode()).decode()
PIP = "python3 -m pip download --no-deps --no-cache-dir --disable-pip-version-check --dest"


@dataclass(frozen=True)
class Matrix:
    results: dict[str, str]
    refused: tuple[str, ...]
    ending: object
    work: Path


@pytest.fixture(scope="module")
def host_listener() -> Iterator[tuple[str, int, list[bytes]]]:
    """A TCP listener on the host's address on Docker's bridge, which a sealed payload must not
    reach, and what reached it."""
    with _bridge_listener() as listener:
        yield listener


@contextmanager
def _bridge_listener() -> Iterator[tuple[str, int, list[bytes]]]:
    gateway = _bridge_gateway()
    server = socket.socket()
    server.bind((gateway, 0))
    server.listen()
    server.settimeout(0.2)
    seen: list[bytes] = []
    stop = threading.Event()

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except OSError:
                continue
            with conn:
                conn.settimeout(1)
                try:
                    seen.append(conn.recv(64) or b"<connected>")
                except OSError:
                    seen.append(b"<connected>")

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    try:
        yield gateway, server.getsockname()[1], seen
    finally:
        stop.set()
        thread.join()
        server.close()


def _bridge_gateway() -> str:
    out = _docker_cli("network", "inspect", "bridge", "--format", "{{(index .IPAM.Config 0).Gateway}}", timeout=60)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


@pytest.fixture(scope="module")
def deny_matrix(docker, gateway_image, image, tmp_path_factory, host_listener) -> Matrix:
    """One build seal under a deny-mode domain and package policy, and every probe of the matrix
    from its payload."""
    bridge, port, _ = host_listener
    seal = SealPlan(policy=DENY, index=True, deny_list=(DENIED_PACKAGE,))
    index = f"http://127.0.0.1:{EGRESS_INDEX_PORT}"
    script = "".join([
        _rc("denied_domain", f"curl -sS -m 20 -o /dev/null https://www.{DENIED_DOMAIN}/"),
        _rc("allowed_package", f"{PIP} /work/a {ALLOWED_PACKAGE}"),
        _rc("denied_package", f"{PIP} /work/b {DENIED_PACKAGE}"),
        _code("denied_page", f"curl -sS -m 20 -o /dev/null -w '%{{http_code}}' {index}/simple/{DENIED_PACKAGE}/"),
        _rc("direct_pypi", "curl -sS -m 20 -o /dev/null https://pypi.org/simple/"),
        _rc("direct_files", f"curl -sS -m 20 -o /dev/null https://files.pythonhosted.org/packages/{DENIED_PACKAGE}"),
        _rc("index_override", f"{PIP} /work/c --index-url https://pypi.org/simple/ {DENIED_PACKAGE}"),
        _code("forged_token", f"curl -sS -m 20 -o /dev/null -w '%{{http_code}}' "
                              f"{index}/files/{FORGED}/{DENIED_PACKAGE}-9.9.9-py3-none-any.whl"),
        _rc("ipv6", "curl -6 -g -sS -m 10 -o /dev/null https://[2606:4700:4700::1111]/"),
        _py("udp_public_dns", UDP_PROBE, "8.8.8.8", "53"),
        _py("udp_resolver_other_port", UDP_PROBE, "$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf)", "5353"),
        _rc("dns_resolves", f"{_PY} 'import socket; socket.getaddrinfo(\"pypi.org\", 443)'"),
        _py("mark", MARK_PROBE, f"www.{DENIED_DOMAIN}", str(EGRESS_FWMARK)),
        _py("icmp", ICMP_PROBE),
        _rc("ip_literal_tls", "curl -sS -m 20 -o /dev/null -k https://1.1.1.1/"),
        _rc("ip_literal_internal", "curl -sS -m 20 -o /dev/null http://169.254.169.254/"),
        _py("ip_literal_host", TCP_PROBE, bridge, str(port), "hls-seal-knock"),
        _py("procs", PROCS_PROBE),
        _rc("iptables", "/usr/sbin/iptables -S"),
        "grep -E '^Cap(Inh|Prm|Eff|Bnd|Amb)' /proc/self/status | tr '\\t' ' ' | tr '\\n' ';' "
        "| sed 's/^/caps=/' >> /work/results; echo >> /work/results\n",
    ])
    root = tmp_path_factory.mktemp("deny")
    with sealed(docker, gateway_image, image, root, seal, script) as run:
        pass
    refused = policy_refusals(run.files.status) or ()
    return Matrix(results=_results(run.work), refused=refused, ending=payload_ending(run.files), work=run.work)


def test_the_payload_ran_to_its_end_with_the_interceptor_alive(deny_matrix):
    assert deny_matrix.ending is not None and deny_matrix.ending.exit_code == 0
    assert deny_matrix.ending.interceptor_alive


def test_a_denied_domain_is_refused_by_the_policy(deny_matrix):
    assert deny_matrix.results["denied_domain"] != "0"
    assert f"www.{DENIED_DOMAIN}" in deny_matrix.refused


def test_a_denied_package_never_resolves_through_the_index(deny_matrix):
    assert deny_matrix.results["denied_package"] != "0"
    assert deny_matrix.results["denied_page"] == "403"


@needs_network
def test_an_allowed_package_installs_through_the_index(deny_matrix):
    assert deny_matrix.results["allowed_package"] == "0", (deny_matrix.work / "allowed_package.log").read_text()
    assert list((deny_matrix.work / "a").glob(f"{ALLOWED_PACKAGE}-*.whl"))


def test_pypi_is_refused_directly(deny_matrix):
    assert deny_matrix.results["direct_pypi"] != "0" and deny_matrix.results["direct_files"] != "0"
    assert "pypi.org" in deny_matrix.refused and "files.pythonhosted.org" in deny_matrix.refused


def test_an_index_override_is_refused(deny_matrix):
    assert deny_matrix.results["index_override"] != "0"


def test_a_forged_files_token_is_refused(deny_matrix):
    assert deny_matrix.results["forged_token"] == "400"


def test_ipv6_does_not_leave(deny_matrix):
    assert deny_matrix.results["ipv6"] != "0"


def test_udp_other_than_dns_to_the_resolver_does_not_leave(deny_matrix):
    assert deny_matrix.results["udp_public_dns"].startswith("blocked:")
    assert deny_matrix.results["udp_resolver_other_port"].startswith("blocked:")


@needs_network
def test_dns_to_the_configured_resolver_still_answers(deny_matrix):
    assert deny_matrix.results["dns_resolves"] == "0", (deny_matrix.work / "dns_resolves.log").read_text()


def test_the_firewall_mark_cannot_be_forged(deny_matrix):
    assert deny_matrix.results["mark"].startswith("mark-refused:")


def test_no_icmp_leaves(deny_matrix):
    assert deny_matrix.results["icmp"].startswith("blocked:")


def test_a_connection_to_an_ip_literal_reaches_only_the_interceptor(deny_matrix, host_listener):
    _, _, seen = host_listener
    assert deny_matrix.results["ip_literal_tls"] != "0"
    assert deny_matrix.results["ip_literal_internal"] != "0"
    # The redirect took the connection to the host's own bridge address; the interceptor read no
    # host from it and closed it, so nothing reached the host's listener.
    assert deny_matrix.results["ip_literal_host"] in ("closed", "blocked:ConnectionResetError",
                                                      "blocked:TimeoutError"), deny_matrix.results
    assert seen == []


def test_the_payload_holds_no_capability_and_cannot_see_the_interceptor(deny_matrix):
    caps = [field.split() for field in deny_matrix.results["caps"].split(";") if field]
    assert len(caps) == 5 and all(int(value, 16) == 0 for _, value in caps), caps
    assert deny_matrix.results["procs"] == "own-only"
    assert deny_matrix.results["iptables"] != "0"


@pytest.fixture(scope="module")
def allow_matrix(docker, gateway_image, image, tmp_path_factory) -> Matrix:
    seal = SealPlan(policy=ALLOW, index=True, deny_list=(DENIED_PACKAGE,))
    script = "".join([
        _rc("allow_domain", f"curl -sS -m 20 -o /dev/null https://www.{DENIED_DOMAIN}/"),
        _rc("unlisted_domain", f"curl -sS -m 20 -o /dev/null https://{UNLISTED_DOMAIN}/"),
        _rc("allow_package", f"{PIP} /work/a {ALLOWED_PACKAGE}"),
        _rc("denied_package", f"{PIP} /work/b {DENIED_PACKAGE}"),
    ])
    root = tmp_path_factory.mktemp("allow")
    with sealed(docker, gateway_image, image, root, seal, script) as run:
        pass
    return Matrix(results=_results(run.work), refused=policy_refusals(run.files.status) or (),
                  ending=payload_ending(run.files), work=run.work)


@needs_network
def test_allow_mode_reaches_the_listed_domain_and_package(allow_matrix):
    assert allow_matrix.results["allow_domain"] == "0", (allow_matrix.work / "allow_domain.log").read_text()
    assert allow_matrix.results["allow_package"] == "0", (allow_matrix.work / "allow_package.log").read_text()


def test_allow_mode_refuses_what_it_does_not_list(allow_matrix):
    assert allow_matrix.results["unlisted_domain"] != "0"
    assert UNLISTED_DOMAIN in allow_matrix.refused
    assert allow_matrix.results["denied_package"] != "0"


@pytest.fixture(scope="module")
def air_gap(docker, gateway_image, image, tmp_path_factory) -> Matrix:
    seal = SealPlan(policy=AIR_GAP, index=True, deny_list=(DENIED_PACKAGE,))
    script = "".join([
        _rc("dns", f"{_PY} 'import socket; socket.getaddrinfo(\"pypi.org\", 443)'"),
        _py("udp_resolver", UDP_PROBE, "$(awk '/^nameserver/{print $2; exit}' /etc/resolv.conf)", "53"),
        _rc("any_domain", "curl -sS -m 20 -o /dev/null https://www.wikipedia.org/"),
        _rc("allow_package", f"{PIP} /work/a {ALLOWED_PACKAGE}"),
        _rc("denied_package", f"{PIP} /work/b {DENIED_PACKAGE}"),
    ])
    root = tmp_path_factory.mktemp("airgap")
    with sealed(docker, gateway_image, image, root, seal, script) as run:
        pass
    return Matrix(results=_results(run.work), refused=policy_refusals(run.files.status) or (),
                  ending=payload_ending(run.files), work=run.work)


def test_an_air_gapped_build_seal_refuses_the_payloads_dns(air_gap):
    assert air_gap.results["dns"] != "0"
    assert air_gap.results["udp_resolver"].startswith("blocked:")
    assert air_gap.results["any_domain"] != "0"
    assert air_gap.results["denied_package"] != "0"


@needs_network
def test_an_air_gapped_build_seals_index_still_resolves_its_upstream(air_gap):
    assert air_gap.results["allow_package"] == "0", (air_gap.work / "allow_package.log").read_text()


def _gateway_ip(container: str) -> str:
    out = _docker_cli("inspect", "--format", '{{(index .NetworkSettings.Networks "bridge").IPAddress}}', container,
                      timeout=60)
    assert out.returncode == 0 and out.stdout.strip(), out.stderr
    return out.stdout.strip()


def _run_unsealed(docker: DockerClient, image: str, script: str, work: Path, *, network_mode: str = "bridge",
                  cap_add: tuple[str, ...] = (), user: str | None = None) -> int:
    """Run *script* in a container of the test image outside any seal, with /work; its status."""
    cfg = ContainerConfig(image=image, argv=("/bin/sh", "-c", script), user=user or _user(), labels=LABELS,
                          network_mode=network_mode, cpuset=_cpus(), memory_bytes=GIB, pids_limit=256,
                          env={"HOME": "/work", "TMPDIR": "/work/tmp"}, cap_add=cap_add,
                          mounts=(BindMount(source=work, target="/work", read_only=False),))
    container = docker.create_container(cfg)
    try:
        docker.start_container(container)
        ended = docker.wait_container(container, timeout=120)
        assert ended is not None
        return ended.status_code
    finally:
        docker.remove_container(container, force=True)


SERVER = r"""
import http.server, socket, sys, threading
udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); udp.bind(('0.0.0.0', int(sys.argv[3])))
other = socket.socket(); other.bind(('0.0.0.0', int(sys.argv[2]))); other.listen()
def note(what):
    with open('/work/reached', 'a') as f:
        f.write(what + '\n')
def udp_loop():
    while True:
        data, _ = udp.recvfrom(64); note('udp')
def other_loop():
    while True:
        conn, _ = other.accept(); note('tcp'); conn.close()
threading.Thread(target=udp_loop, daemon=True).start()
threading.Thread(target=other_loop, daemon=True).start()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = self.path.encode()
        self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers()
        self.wfile.write(body)
        if self.path == '/stop':
            threading.Thread(target=server.shutdown).start()
server = http.server.ThreadingHTTPServer(('0.0.0.0', int(sys.argv[1])), H)
server.serve_forever()
"""


def _get(port: int, path: str, *, within: float = 60) -> httpx.Response:
    give_up = time.monotonic() + within
    while True:
        try:
            return httpx.get(f"http://127.0.0.1:{port}{path}", timeout=5, trust_env=False)
        except httpx.HTTPError:
            if time.monotonic() > give_up:
                raise
            time.sleep(0.2)


def test_the_served_port_is_reachable_from_the_host_alone(docker, gateway_image, image, tmp_path):
    """A served seal's port answers the host, through Docker's publish, and nothing else of the
    namespace does; a neighbour on Docker's bridge reaches neither the served port, which the
    gateway takes from the bridge's gateway (the host) alone, nor any other port."""
    seal = SealPlan(policy=SERVED, index=False, ingress_port=SERVED_PORT)
    other_tcp, other_udp = 9000, 9001
    quoted = SERVER.replace("'", "'\"'\"'")
    script = f"exec python3 -I -c '{quoted}' {SERVED_PORT} {other_tcp} {other_udp}"
    with sealed(docker, gateway_image, image, tmp_path / "served", seal, script) as run:
        port = ingress_port(run.files)
        assert port is not None and port == run.gateway.host_port
        assert _get(port, "/from-the-host").text == "/from-the-host"
        # The interceptor runs, as the host user, where the host sees it and the payload does not.
        running = interceptors("/proc")
        assert running and all(Path(f"/proc/{pid}").stat().st_uid == os.getuid() for pid in running)
        # Every process of the gateway is the host user's and holds CAP_NET_ADMIN alone.
        first = docker.inspect_container(run.gateway.id).state.pid
        for pid in (first, *running):
            status = dict(line.split(":\t", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines()
                          if ":\t" in line)
            assert status["Uid"].split() == [str(os.getuid())] * 4 and status["Gid"].split() == [str(os.getgid())] * 4
            assert int(status["CapEff"], 16) == int(status["CapPrm"], 16) == 1 << 12, status
        assert running[0] != first and _ppid(running[0]) == first
        address = _gateway_ip(run.gateway.id)
        probes = "".join([
            _py("served", TCP_PROBE, address, str(SERVED_PORT), "GET / HTTP/1.0"),
            _py("other_tcp", TCP_PROBE, address, str(other_tcp), "knock"),
            _py("interceptor", TCP_PROBE, address, "8888", "knock"),
            _py("other_udp", UDP_PROBE, address, str(other_udp)),
        ])
        (tmp_path / "neighbour" / "tmp").mkdir(parents=True)
        assert _run_unsealed(docker, image, probes, tmp_path / "neighbour") == 0
        assert _get(port, "/stop").text == "/stop"
    seen = _results(tmp_path / "neighbour")
    assert all(seen[probe].startswith("blocked:") for probe in ("served", "other_tcp", "interceptor")), seen
    reached = (run.work / "reached").read_text().split() if (run.work / "reached").exists() else []
    assert reached == [], reached
    ending = payload_ending(run.files)
    assert ending is not None and ending.exit_code == 0 and ending.interceptor_alive


def test_the_seal_probe_passes_in_a_sealed_payload(docker, gateway_image, image, tmp_path):
    seal = SealPlan(policy=SERVED, index=False, ingress_port=SERVED_PORT)
    host = refused_host(SERVED.network)
    assert host is not None
    tcp_in, udp_in = inbound_ports(avoid=namespace_ports(seal))
    command = (f"exec python3 -I {CHECKOUT}/src/hls/egress/seal_probe.py served --port {SERVED_PORT} "
               f"--inbound-tcp {tcp_in} --inbound-udp {udp_in} --out /work/report.json --check-sec {CHECK_SEC} "
               f"--ucounts '' --refused-host {host} --dial {PROBE_DIAL}")
    with sealed(docker, gateway_image, image, tmp_path / "probe", seal, command) as run:
        port = run.gateway.host_port
        assert port is not None
        word = uuid.uuid4().hex
        assert _get(port, f"/{word}").text == word
        addresses = node_addresses()
        knocks = knock_inbound(addresses, tcp_in, udp_in)
        second = uuid.uuid4().hex
        assert _get(port, f"/{second}").text == second
    report = read_report(run.work / "report.json")
    assert report is not None
    checks = {"caps": report.caps, "ucounts": report.ucounts, "refusal": report.refusal, "icmp": report.icmp,
              "ingress": report.ingress, "inbound": report.inbound}
    assert all(c is not None and c.passed for c in checks.values()), checks
    assert host in (policy_refusals(run.files.status) or ())
    assert knock_failures(addresses, knocks, report.reached) == []
    ending = payload_ending(run.files)
    assert ending is not None and ending.exit_code == 0


def test_the_seal_files_are_the_host_users_and_the_status_is_the_interceptors(deny_matrix):
    state = deny_matrix.work.parent / "seal"
    for path in state.iterdir():
        assert path.stat().st_uid == os.getuid(), path
    status = index_status(state / "status.json")
    assert status is not None and status.total == 0


@needs_network
def test_a_neighbour_with_the_mark_and_the_capability_bypasses_the_redirect(docker, gateway_image, image, tmp_path):
    """Control: the mark probe sees a broken seal. A container in the gateway's namespace that
    holds CAP_NET_ADMIN, which no payload is given, sets the mark and reaches the denied host
    past the interceptor."""
    seal = SealPlan(policy=DENY, index=True, deny_list=(DENIED_PACKAGE,))
    with sealed(docker, gateway_image, image, tmp_path / "control", seal, "sleep 60") as run:
        work = tmp_path / "forger"
        (work / "tmp").mkdir(parents=True)
        # Docker gives an added capability to root alone, and root without CAP_DAC_OVERRIDE
        # writes only where any user may.
        work.chmod(0o777)
        script = _py("mark", MARK_PROBE, f"www.{DENIED_DOMAIN}", str(EGRESS_FWMARK))
        _run_unsealed(docker, image, script, work, network_mode=f"container:{run.gateway.id}", cap_add=("NET_ADMIN",),
                      user="0:0")
        docker.kill_container(run.payload, signal.SIGKILL)
    assert _results(work)["mark"] == "reached"


@needs_network
def test_outside_the_seal_the_udp_probe_is_answered(docker, image, tmp_path):
    """Control: the UDP probe sees an open path."""
    (tmp_path / "tmp").mkdir()
    assert _run_unsealed(docker, image, _py("udp", UDP_PROBE, "8.8.8.8", "53"), tmp_path) == 0
    assert _results(tmp_path)["udp"] == "answered"


def _bridge_host(route: Path) -> str:
    """The default route's gateway in a copy of ``/proc/net/route``: the host's address on the
    bridge, from which Docker's publish reaches the gateway's namespace."""
    for line in route.read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == "00000000" and fields[2] != "00000000":
            return ".".join(str(int(fields[2][i:i + 2], 16)) for i in (6, 4, 2, 0))
    raise AssertionError(f"no default route in {route}")


#: The INPUT chains the gateway adds to the seal's firewall, as the calls that make them; the served
#: port is taken from the host's bridge address *host* alone.
def gateway_input(served: int | None, host: str) -> list[list[str]]:
    v4 = [["iptables", "-A", "INPUT", "-i", "lo", "-j", "ACCEPT"],
          ["iptables", "-A", "INPUT", "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"]]
    if served is not None:
        v4.append(["iptables", "-A", "INPUT", "-s", host, "-p", "tcp", "--dport", str(served), "-j", "ACCEPT"])
    v4.append(["iptables", "-A", "INPUT", "-j", "DROP"])
    return v4 + [["ip6tables", "-A", "INPUT", "-i", "lo", "-j", "ACCEPT"],
                 ["ip6tables", "-A", "INPUT", "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"],
                 ["ip6tables", "-A", "INPUT", "-j", "DROP"]]


DUMP = ("for t in filter nat; do echo \"== iptables $t\"; iptables -t $t -S; "
        "echo \"== ip6tables $t\"; ip6tables -t $t -S; done > /work/rules 2>&1")


@pytest.mark.parametrize("case, seal", [
    ("build seal", SealPlan(policy=DENY, index=True, deny_list=(DENIED_PACKAGE,))),
    ("build seal, DNS closed", SealPlan(policy=AIR_GAP, index=True, deny_list=(DENIED_PACKAGE,))),
    ("served seal", SealPlan(policy=SERVED, index=False, ingress_port=SERVED_PORT)),
])
def test_the_gateway_installs_the_pasta_seals_recorded_firewall(docker, gateway_image, image, tmp_path, case, seal):
    """The gateway's namespace holds exactly the rules the pasta seal's recording names
    (``tests/substrate/seal_rules_plan.json``), with the resolvers of the gateway's resolv.conf,
    and the gateway's INPUT chains: the same rules, made by the same iptables in a fresh
    namespace, read back the same."""
    from tests.substrate.test_seal_rules import RECORDING, expected_calls

    inspected, reference = tmp_path / "inspected", tmp_path / "reference"
    for work in (inspected, reference):
        (work / "tmp").mkdir(parents=True)
        work.chmod(0o777)
    with sealed(docker, gateway_image, image, tmp_path / "gw", seal, "sleep 60") as run:
        script = f"cp /etc/resolv.conf /work/resolv.conf; cp /proc/net/route /work/route; {DUMP}"
        _run_unsealed(docker, gateway_image, script, inspected, network_mode=f"container:{run.gateway.id}",
                      cap_add=("NET_ADMIN",), user="0:0")
        docker.kill_container(run.payload, signal.SIGKILL)
    calls = expected_calls(json.loads(RECORDING.read_text())[case], 8888, inspected / "resolv.conf")
    calls += gateway_input(seal.ingress_port, _bridge_host(inspected / "route"))
    make = "set -e; " + "; ".join(shlex.join(call) for call in calls) + "; " + DUMP
    assert _run_unsealed(docker, gateway_image, make, reference, network_mode="none", cap_add=("NET_ADMIN",),
                         user="0:0") == 0, (reference / "rules").read_text()
    assert (inspected / "rules").read_text() == (reference / "rules").read_text()
    if not closes_dns(seal):
        assert any("--dport 53 -j ACCEPT" in line for line in (inspected / "rules").read_text().splitlines())


def test_outside_the_seal_the_hosts_bridge_listener_is_reached(docker, image, tmp_path):
    """Control: the IP-literal probe sees a connection that reaches the host."""
    (tmp_path / "tmp").mkdir()
    with _bridge_listener() as (address, port, seen):
        assert _run_unsealed(docker, image, _py("host", TCP_PROBE, address, str(port), "hls-seal-knock"), tmp_path) == 0
        give_up = time.monotonic() + 5
        while not seen and time.monotonic() < give_up:
            time.sleep(0.05)
    assert seen and seen[0].startswith(b"hls-seal-knock"), seen


def _failing_gateway(docker, gateway_image, tmp_path, *, argv=None, env=None) -> str:
    seal = SealPlan(policy=DENY, index=True, deny_list=(DENIED_PACKAGE,))
    (tmp_path / "seal").mkdir()
    files = seal_files(tmp_path / "seal")
    config = dseal.gateway_config(seal, files, image=gateway_image, python=sys.executable, user=_user(),
                                  code=code_mounts(), cpuset=_cpus(), memory_bytes=2 * GIB, labels=LABELS)
    config = dataclasses.replace(config, argv=argv(config.argv) if argv else config.argv,
                                 env={**config.env, **(env or {})})
    with pytest.raises(dseal.GatewayFailed) as failed:
        dseal.open_gateway(docker, seal, files, config, wait_sec=60)
    assert docker.list_containers(labels=LABELS) == []
    return str(failed.value)


def test_a_gateway_whose_interceptor_never_starts_fails_with_its_log(docker, gateway_image, tmp_path):
    # A policy the interceptor refuses before it listens.
    message = _failing_gateway(docker, gateway_image, tmp_path, argv=lambda a: (*a[:8], "{}", *a[9:]))
    assert "before its interceptor listened" in message
    assert "FATAL: egress interceptor did not become ready" in message and "proxy exited" in message


def test_a_gateway_whose_firewall_fails_reports_the_root_parts_message(docker, gateway_image, tmp_path):
    message = _failing_gateway(docker, gateway_image, tmp_path, env={"HLS_SEAL_DNS": "half"})
    assert "FATAL: HLS_SEAL_DNS is closed or unset, got half" in message
    assert "could not install its firewall (status 2)" in message


@pytest.mark.parametrize("uid, gid", [("0", "1000"), ("1000", "00"), ("", "1000")])
def test_a_gateway_told_to_stay_root_ends_before_its_firewall(docker, gateway_image, tmp_path, uid, gid):
    """The builder refuses root (``gateway_argv``); the gateway refuses it too, before it does
    anything, so no interceptor ever runs as root."""
    message = _failing_gateway(docker, gateway_image, tmp_path, argv=lambda a: (*a[:10], uid, gid))
    assert "ended with status 2 before its interceptor listened" in message
    assert not (tmp_path / "seal" / "gateway" / "listening").read_text()
