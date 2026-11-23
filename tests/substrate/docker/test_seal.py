"""The pairing of a sealed role's gateway and payload on Docker (``hls.substrate.docker.seal``),
against the fake daemon: what the gateway container is, how ``open_gateway`` waits for its
listeners and gives up, what the payload's container becomes, and the evidence files, which the
stage reads through ``hls.substrate.seal`` as it reads the pasta seal's."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

import pytest

from hls.egress.denied_packages import DENIED_PACKAGES
from hls.egress.policy import NetworkPolicy, SealPolicy
from hls.substrate.container import SealPlan
from hls.substrate.docker import seal as dseal
from hls.substrate.docker.api import BindMount, ContainerConfig, DockerClient
from hls.substrate.seal import (
    EGRESS_FWMARK,
    EGRESS_INDEX_PORT,
    EGRESS_TRANSPARENT_PORT,
    SEAL_RULES,
    UPSTREAM_PROXY_VAR,
    index_status,
    ingress_port,
    payload_ended_at,
    payload_ending,
    seal_files,
)
from tests.substrate.docker.fake_daemon import FakeDocker

AIR_GAP = SealPolicy(network=NetworkPolicy.model_validate(
    {"domains": {"mode": "allow", "entries": []}, "packages": {"mode": "deny", "entries": list(DENIED_PACKAGES)}}))
FILTERED = SealPolicy(network=NetworkPolicy.model_validate({"domains": {"mode": "allow", "entries": ["api.example.org"]}}))
BUILD = SealPlan(policy=AIR_GAP, index=True, deny_list=tuple(DENIED_PACKAGES))
SERVED = SealPlan(policy=FILTERED, index=False, ingress_port=8000)
CODE = (BindMount(source=Path("/"), target="/checkout", read_only=True),)
IMAGE = "hls/step:test"


@pytest.fixture
def fake(tmp_path):
    daemon = FakeDocker(tmp_path / "d.sock")
    daemon.add_image(IMAGE)
    with daemon.serving():
        yield daemon


@pytest.fixture
def client(fake):
    with DockerClient(fake.path, timeout=5.0) as c:
        yield c


def _files(tmp_path: Path):
    state = tmp_path / "seal"
    state.mkdir(exist_ok=True)
    return seal_files(state)


def _config(seal: SealPlan, files, **over) -> ContainerConfig:
    kwargs = dict(image=IMAGE, python="/venv/bin/python", user="1000:1000", code=CODE, cpuset=(2, 3),
                  memory_bytes=2 << 30, labels={"hls.job": "7", "hls.role": "srv"})
    kwargs.update(over)
    return dseal.gateway_config(seal, files, **kwargs)


def _listen_when_started(fake: FakeDocker, files, *, delay: float = 0.2) -> threading.Thread:
    """Stand in for the gateway: once its container runs, write the listening file."""
    listening = dseal.gateway_files(files).listening

    def run() -> None:
        until = time.monotonic() + 10
        while time.monotonic() < until:
            if any(c["State"]["Running"] for c in fake.containers.values()):
                time.sleep(delay)
                with open(listening, "r+") as f:
                    f.write(dseal.LISTENING)
                return
            time.sleep(0.02)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def test_the_gateway_is_root_with_three_capabilities_on_the_bridge(tmp_path):
    files = _files(tmp_path)
    cfg = _config(SERVED, files)
    assert cfg.user == "0:0" and cfg.cap_add == ("NET_ADMIN", "SETGID", "SETUID")
    assert cfg.network_mode == "bridge" and cfg.read_only_root and not cfg.init
    assert [(p.container_port, p.host_ip, p.host_port) for p in cfg.port_bindings] == [(8000, "0.0.0.0", None)]
    assert cfg.labels == {"hls.job": "7", "hls.role": "srv", dseal.GATEWAY_LABEL: "gateway"}
    assert cfg.pids_limit == dseal.GATEWAY_PIDS_LIMIT and cfg.cpuset == (2, 3) and cfg.memory_bytes == 2 << 30


def test_the_gateway_mounts_its_code_read_only_and_writes_only_its_own_directory(tmp_path):
    files = _files(tmp_path)
    own = dseal.gateway_files(files)
    cfg = _config(BUILD, files)
    assert [(m.source, m.target) for m in cfg.mounts if not m.read_only] == [(own.dir, "/hls-seal")]
    assert [m for m in cfg.mounts if m.read_only] == list(CODE)
    assert own.dir.parent == files.started.parent and own.dir not in (files.started, files.port, files.ended)
    with pytest.raises(ValueError, match="read-only"):
        _config(BUILD, files, code=(BindMount(source=Path("/"), target="/checkout", read_only=False),))


def test_the_gateway_publishes_no_port_for_a_seal_that_serves_nothing(tmp_path):
    assert _config(BUILD, _files(tmp_path)).port_bindings == ()


def test_the_gateway_runs_the_rule_file_then_its_script_with_the_pasta_seals_parameters(tmp_path):
    argv = _config(BUILD, _files(tmp_path)).argv
    assert argv[:2] == ("/bin/bash", "-c")
    assert argv[2] == SEAL_RULES.read_text() + dseal.GATEWAY_SCRIPT.read_text()
    assert argv[3:] == ("hls-gateway", str(EGRESS_TRANSPARENT_PORT), str(EGRESS_INDEX_PORT), str(EGRESS_FWMARK),
                        "/venv/bin/python", AIR_GAP.model_dump_json(), "", "1000", "1000")
    served = _config(SERVED, _files(tmp_path)).argv
    assert served[5] == "" and served[9] == "8000"


@pytest.mark.parametrize("user", ["0:1000", "1000:0", "1000", "a:b", ""])
def test_the_interceptor_never_runs_as_root(tmp_path, user):
    with pytest.raises(ValueError, match="never root"):
        _config(BUILD, _files(tmp_path), user=user)


def test_the_gateway_env_names_its_files_closed_dns_and_the_upstream_proxy(tmp_path):
    env = _config(BUILD, _files(tmp_path), upstream_proxy="http://proxy.site:3128").env
    assert env == {"HLS_PROXY_STATUS_FILE": "/hls-seal/status.json", "HLS_GATEWAY_LISTENING_FILE": "/hls-seal/listening",
                   "HLS_GATEWAY_LOG_FILE": "/hls-seal/gateway.log", "HLS_GATEWAY_LOG_KEEP": str(dseal.GATEWAY_LOG_KEEP),
                   "HLS_SEAL_DNS": "closed", UPSTREAM_PROXY_VAR: "http://proxy.site:3128"}
    served = _config(SERVED, _files(tmp_path)).env
    assert "HLS_SEAL_DNS" not in served and UPSTREAM_PROXY_VAR not in served


def test_the_gateway_script_parses_and_its_unprivileged_block_holds_no_apostrophe():
    subprocess.run(["bash", "-n", str(dseal.GATEWAY_SCRIPT)], check=True)
    subprocess.run(["bash", "-n", "-c", dseal.gateway_script()], check=True)
    text = dseal.GATEWAY_SCRIPT.read_text()
    block = text.split("-- bash -c '", 1)[1].rsplit("' hls-gateway", 1)[0]
    assert "'" not in block


def test_open_gateway_returns_once_it_listens_and_writes_the_host_port(tmp_path, fake, client):
    files = _files(tmp_path)
    cfg = _config(SERVED, files)
    _listen_when_started(fake, files)
    gateway = dseal.open_gateway(client, SERVED, files, cfg, name="gw")
    assert gateway.host_port == 49153
    assert files.port.read_text() == "49153\n"
    # The port is known to a reader only once the payload starts.
    assert ingress_port(files) is None
    dseal.mark_started(gateway)
    assert ingress_port(files) == 49153
    [body] = fake.create_bodies
    assert body["HostConfig"]["CapAdd"] == ["NET_ADMIN", "SETGID", "SETUID"] and body["User"] == "0:0"


def test_open_gateway_clears_what_an_earlier_gateway_left(tmp_path, fake, client):
    files = _files(tmp_path)
    for path in (files.started, files.port, files.ended):
        path.write_text('{"exit": 0, "interceptor": true}\n')
    files.status.write_text("stale")
    (files.started.parent / "gateway").mkdir()
    (files.started.parent / "gateway" / "listening").write_text(dseal.LISTENING)
    _listen_when_started(fake, files)
    dseal.open_gateway(client, BUILD, files, _config(BUILD, files))
    assert not files.started.exists() and not files.port.exists()
    assert payload_ending(files) is None
    assert files.ended.read_bytes() == b" " * dseal.ENDED_BYTES
    assert files.status.read_text() == ""


def test_the_seal_status_is_the_interceptors_and_gone_when_it_removes_it(tmp_path, fake, client):
    """The interceptor removes a status it could not write, so a reader finds none rather than a
    stale count; through the link the seal's status file reads as gone too."""
    _, files = _opened(tmp_path, fake, client)
    own = dseal.gateway_files(files)
    assert files.status.is_symlink() and files.status.resolve() == own.status.resolve()
    proxy_status = {"upstream_errors": 2, "mid_stream_errors": 0, "oversized_pages": [], "policy_refused": [],
                    "fetches_refused": 0, "heartbeat": time.time()}
    own.status.write_text(json.dumps(proxy_status))
    assert index_status(files.status).total == 2
    own.status.unlink()
    assert index_status(files.status) is None


def test_a_gateway_that_ends_before_it_listens_fails_and_is_removed(tmp_path, fake, client):
    files = _files(tmp_path)

    def end() -> None:
        while not any(c["State"]["Running"] for c in fake.containers.values()):
            time.sleep(0.02)
        dseal.gateway_files(files).log.write_text("FATAL: iptables: no chain by that name\n")
        fake.exit(next(iter(fake.containers)), 2)

    threading.Thread(target=end, daemon=True).start()
    with pytest.raises(dseal.GatewayFailed, match="status 2 before its interceptor listened: FATAL: iptables"):
        dseal.open_gateway(client, BUILD, files, _config(BUILD, files))
    assert fake.containers == {}


def test_a_gateway_that_never_listens_fails_in_time_and_is_removed(tmp_path, fake, client):
    files = _files(tmp_path)
    started = time.monotonic()
    with pytest.raises(dseal.GatewayFailed, match="did not listen within 1 s"):
        dseal.open_gateway(client, BUILD, files, _config(BUILD, files), wait_sec=1.0)
    assert time.monotonic() - started < 5
    assert fake.containers == {}


def test_a_seal_and_a_gateway_disagreeing_on_the_served_port_are_refused(tmp_path, client):
    files = _files(tmp_path)
    with pytest.raises(ValueError, match="served port"):
        dseal.open_gateway(client, SERVED, files, _config(BUILD, files))


def _payload(**over) -> ContainerConfig:
    base = dict(image=IMAGE, argv=("/bin/sh", "-c", "true"), user="1000:1000", labels={"hls.role": "srv"},
                network_mode="none", cpuset=(2, 3), memory_bytes=1 << 30, pids_limit=512)
    base.update(over)
    return ContainerConfig(**base)


def test_the_payload_joins_the_gateways_namespace_and_nothing_else_changes():
    gateway = dseal.Gateway(id="a" * 64, files=seal_files(Path("/s")), host_port=None)
    cfg = _payload()
    joined = dseal.payload_config(cfg, gateway)
    assert joined.network_mode == "container:" + "a" * 64
    assert dict(vars(joined), network_mode=None) == dict(vars(cfg), network_mode=None)


@pytest.mark.parametrize("over, match", [
    ({"cap_add": ("NET_RAW",)}, "no capability"),
    ({"network_mode": "bridge", "port_bindings": (dseal.PortBinding(8000, "0.0.0.0"),)}, "publishes no port"),
    ({"user": "0:0"}, "never runs as root"),
])
def test_a_payload_that_could_break_the_seal_is_refused(over, match):
    gateway = dseal.Gateway(id="a" * 64, files=seal_files(Path("/s")), host_port=None)
    with pytest.raises(ValueError, match=match):
        dseal.payload_config(_payload(**over), gateway)


def _opened(tmp_path, fake, client, seal=BUILD):
    files = _files(tmp_path)
    _listen_when_started(fake, files)
    return dseal.open_gateway(client, seal, files, _config(seal, files)), files


def test_the_ending_says_the_interceptor_ran_when_the_gateway_still_did(tmp_path, fake, client):
    gateway, files = _opened(tmp_path, fake, client)
    dseal.mark_started(gateway)
    before = time.time()
    dseal.close_gateway(client, gateway, exit_code=3)
    ending = payload_ending(files)
    assert ending is not None and (ending.exit_code, ending.interceptor_alive) == (3, True)
    ended_at = payload_ended_at(files)
    assert ended_at is not None and before - 1 <= ended_at <= time.time() + 1
    assert json.loads(files.ended.with_name("ended.pending").read_text()) == {"exit": 3, "interceptor": True}
    assert fake.containers == {}
    assert ("POST", f"/v1.44/containers/{gateway.id}/kill") in [(m, p) for m, p, _, _ in fake.requests]


def test_the_ending_says_the_interceptor_was_gone_when_the_gateway_had_ended(tmp_path, fake, client):
    gateway, files = _opened(tmp_path, fake, client)
    dseal.mark_started(gateway)
    fake.exit(gateway.id, 0)
    dseal.close_gateway(client, gateway, exit_code=0)
    ending = payload_ending(files)
    assert ending is not None and (ending.exit_code, ending.interceptor_alive) == (0, False)
    assert fake.containers == {}


def test_a_payload_whose_end_was_not_seen_leaves_no_ending(tmp_path, fake, client):
    gateway, files = _opened(tmp_path, fake, client)
    dseal.mark_started(gateway)
    dseal.close_gateway(client, gateway, exit_code=None)
    assert payload_ending(files) is None and payload_ended_at(files) is None
    assert fake.containers == {}


def test_a_payload_that_never_started_has_no_ending(tmp_path, fake, client):
    gateway, files = _opened(tmp_path, fake, client)
    with pytest.raises(ValueError, match="never started"):
        dseal.close_gateway(client, gateway, exit_code=0)
    assert payload_ending(files) is None
    # The gateway is removed even so.
    assert fake.containers == {}


def test_the_ending_is_written_over_the_bytes_made_before_the_payload(tmp_path, fake, client):
    """No write of the ending needs a new block: a payload that filled the disk cannot stop it."""
    gateway, files = _opened(tmp_path, fake, client)
    dseal.mark_started(gateway)
    pending = files.ended.with_name("ended.pending")
    inodes = (files.ended.stat().st_ino, pending.stat().st_ino)
    dseal.close_gateway(client, gateway, exit_code=137)
    assert (files.ended.stat().st_ino, pending.stat().st_ino) == inodes
    assert files.ended.stat().st_size == pending.stat().st_size == dseal.ENDED_BYTES


def test_the_seal_files_are_the_host_users_alone(tmp_path, fake, client):
    _, files = _opened(tmp_path, fake, client)
    own = dseal.gateway_files(files)
    assert own.dir.stat().st_mode & 0o777 == 0o700
    for path in (own.status, own.listening, own.log, files.ended):
        assert path.stat().st_mode & 0o777 == 0o600, path
