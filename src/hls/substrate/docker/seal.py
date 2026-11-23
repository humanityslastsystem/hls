"""The network seal of a container role on Docker: a gateway container that owns the role's network
namespace, and the payload in it.

The gateway runs the step image on Docker's default bridge as root with ``CAP_NET_ADMIN``,
``CAP_SETUID`` and ``CAP_SETGID`` alone. Its entrypoint, ``gateway.sh`` with the shared rule file
(``hls.substrate.seal.SEAL_RULES``) at its head, installs the firewall the pasta seal installs and
an INPUT chain that takes only loopback, replies and the served port; then it becomes the host
user, keeping ``CAP_NET_ADMIN`` alone for the interceptor's mark, and runs ``python -m
hls.egress.proxy`` with the plan's interpreter, which the launcher binds read-only with the
checkout (*code*). Once both of the interceptor's listeners take connections it writes the
listening file, and only then does ``open_gateway`` return. The payload is then created with
``payload_config``: network mode ``container:<gateway>``, no capability, no published port, and
a PID namespace of its own (Docker's default), so it can neither see nor signal the interceptor,
nor rewrite the firewall, nor set the mark. A gateway lives as long as its interceptor.

The evidence a stage reads is the pasta seal's, in the same files and formats
(``hls.substrate.seal``). The gateway writes only in a directory of its own in the seal's state
directory (``gateway_files``), the one it mounts: the interceptor's status file, the listening
file and its capped log. The seal's status file is a link to the interceptor's, so an
interceptor that removes a status it could not write leaves a reader none, as on the pasta
seal. ``open_gateway`` writes the port file with the host port Docker published the served port
on; the launcher calls ``mark_started`` right before it releases the payload's gate; and
``close_gateway``, called as soon as the payload ended, writes the payload's exit status and
whether the gateway, and so its interceptor, still ran, into ``<ended>.pending`` and then the
ended file, both made at full size before the payload starts. No container mounts the started,
port or ended file."""

from __future__ import annotations

import dataclasses
import os
import shutil
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from hls.substrate.container import SealPlan
from hls.substrate.docker.api import (
    BindMount,
    ContainerConfig,
    DockerClient,
    NotFound,
    PortBinding,
    RequestRefused,
)
from hls.substrate.seal import (
    EGRESS_FWMARK,
    SEAL_RULES,
    UPSTREAM_PROXY_VAR,
    SealFiles,
    closes_dns,
    interceptor_ports,
)

#: The gateway's entrypoint, which runs after the rule file's text.
GATEWAY_SCRIPT = Path(__file__).with_name("gateway.sh")
#: The capabilities the gateway starts with: the firewall's, and the two that let it become the
#: host user. It keeps ``NET_ADMIN`` alone once it has.
GATEWAY_CAPS = ("NET_ADMIN", "SETGID", "SETUID")
#: The gateway's own user until it has installed the firewall: root of the host, holding no
#: capability but ``GATEWAY_CAPS`` and unable to gain one (``no-new-privileges``). The gateway
#: has no init process: its entrypoint is its first process and becomes the host user, so no
#: process of it stays root.
GATEWAY_USER = "0:0"
#: Where the gateway's own directory is inside it.
GATEWAY_DIR = "/hls-seal"
#: Tasks the gateway may hold: its shell, the interceptor with its resolver threads (at most 32,
#: Python's default executor), and the log capper.
GATEWAY_PIDS_LIMIT = 128
#: Bytes the gateway's log keeps at each end (``hls.substrate.log_cap``).
GATEWAY_LOG_KEEP = 1 << 20
#: Seconds ``open_gateway`` waits for the listening file: more than the 60 s the gateway itself
#: waits for its interceptor, so a slow interceptor ends as the gateway's own failure.
LISTEN_WAIT_SEC = 90.0
#: What the gateway writes over its listening file once both listeners take connections.
LISTENING = "listening\n"
#: Bytes of each ending record, made before the payload starts: far more than its one line of JSON.
ENDED_BYTES = 512
#: The label that marks a seal's gateway container.
GATEWAY_LABEL = "hls.seal"


class GatewayFailed(Exception):
    """The gateway did not seal its namespace: it ended, or did not listen in time, before the
    payload was created. The payload never ran, so the role ends as infra."""


@dataclass(frozen=True)
class GatewayFiles:
    """The gateway's own host files, in a directory of the seal's state directory."""

    #: The directory, which the gateway mounts at ``GATEWAY_DIR``, and the only one it writes.
    dir: Path
    #: The interceptor's status file, to which the seal's status file links.
    status: Path
    #: Holds ``LISTENING`` once both of the interceptor's listeners take connections.
    listening: Path
    #: The gateway's messages and the interceptor's stderr, capped.
    log: Path


@dataclass(frozen=True)
class Gateway:
    """A running gateway that seals its namespace."""

    #: The gateway container.
    id: str
    files: SealFiles
    #: The host port of the served port; None for a seal that serves nothing.
    host_port: int | None


def gateway_files(files: SealFiles) -> GatewayFiles:
    """The gateway's own files, in the directory of the seal files *files*."""
    own = files.status.parent / "gateway"
    return GatewayFiles(dir=own, status=own / "status.json", listening=own / "listening", log=own / "gateway.log")


def gateway_script() -> str:
    """The gateway's program as ``bash -c`` runs it: the rule file, which defines
    ``seal_rules``, then ``gateway.sh``."""
    return SEAL_RULES.read_text() + GATEWAY_SCRIPT.read_text()


def _host_user(user: str) -> tuple[str, str]:
    uid, sep, gid = user.partition(":")
    if not sep or not uid.isdecimal() or not gid.isdecimal() or int(uid) == 0 or int(gid) == 0:
        raise ValueError(f"the gateway runs its interceptor as a host user uid:gid, never root, not {user!r}")
    return uid, gid


def gateway_argv(seal: SealPlan, *, python: str, user: str) -> tuple[str, ...]:
    """The gateway's command: ``gateway_script`` and its parameters in ``gateway.sh``'s order.
    *python* runs the interceptor and must import ``hls``; *user* is the host ``uid:gid`` it
    becomes."""
    uid, gid = _host_user(user)
    transparent, index = interceptor_ports(seal)
    return ("/bin/bash", "-c", gateway_script(), "hls-gateway", str(transparent),
            "" if index is None else str(index), str(EGRESS_FWMARK), python, seal.policy.model_dump_json(),
            "" if seal.ingress_port is None else str(seal.ingress_port), uid, gid)


def gateway_env(seal: SealPlan, *, upstream_proxy: str | None) -> dict[str, str]:
    """The gateway's declared variables: where its three files are inside it, the log's cap,
    ``HLS_SEAL_DNS=closed`` when the seal refuses the payload's DNS (``closes_dns``) and the
    site's upstream proxy when it has one. None of them is secret, and none reaches the
    payload."""
    env = {"HLS_PROXY_STATUS_FILE": f"{GATEWAY_DIR}/status.json",
           "HLS_GATEWAY_LISTENING_FILE": f"{GATEWAY_DIR}/listening",
           "HLS_GATEWAY_LOG_FILE": f"{GATEWAY_DIR}/gateway.log", "HLS_GATEWAY_LOG_KEEP": str(GATEWAY_LOG_KEEP)}
    if closes_dns(seal):
        env["HLS_SEAL_DNS"] = "closed"
    if upstream_proxy is not None:
        env[UPSTREAM_PROXY_VAR] = upstream_proxy
    return env


def gateway_config(seal: SealPlan, files: SealFiles, *, image: str, python: str, user: str,
                   code: tuple[BindMount, ...], cpuset: tuple[int, ...], memory_bytes: int,
                   labels: Mapping[str, str], upstream_proxy: str | None = None,
                   host_ip: str = "0.0.0.0") -> ContainerConfig:
    """The gateway container of *seal*, whose files are *files*: the step image *image* on
    Docker's default bridge, publishing a served container's port on *host_ip* at a host port
    Docker picks (on every address, as the pasta seal forwards it). *code* are the read-only
    binds that let *python* import ``hls``; *user* is the host ``uid:gid`` the interceptor
    runs as; *cpuset* and *memory_bytes* are the gateway's own limits. Every mount but the
    gateway's own directory is read-only."""
    for mount in code:
        if not mount.read_only:
            raise ValueError(f"the gateway's code is bound read-only, not {mount.source} -> {mount.target}")
    own = gateway_files(files)
    mounts = (*code, BindMount(source=own.dir, target=GATEWAY_DIR, read_only=False))
    ports = () if seal.ingress_port is None else (PortBinding(container_port=seal.ingress_port, host_ip=host_ip),)
    return ContainerConfig(
        image=image, argv=gateway_argv(seal, python=python, user=user), user=GATEWAY_USER,
        labels={**labels, GATEWAY_LABEL: "gateway"}, network_mode="bridge", cpuset=cpuset,
        memory_bytes=memory_bytes, pids_limit=GATEWAY_PIDS_LIMIT, env=gateway_env(seal, upstream_proxy=upstream_proxy),
        mounts=mounts, port_bindings=ports, cap_add=GATEWAY_CAPS, init=False, read_only_root=True)


def _make(path: Path, size: int = 0) -> None:
    """Make *path* afresh, the host user's alone, *size* spaces long."""
    path.unlink(missing_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        if size:
            os.write(fd, b" " * size)
    finally:
        os.close(fd)


def prepare(files: SealFiles) -> None:
    """Clear what an earlier gateway of *files* left and make what a new one writes: its own
    directory, the host user's alone, with the status file, the listening file and the log
    empty; the seal's status file as a link to the interceptor's; and the ending records at full
    size, all spaces, which read as no ending, so a payload that fills the disk cannot keep its
    ending from being written."""
    own = gateway_files(files)
    for path in (files.started, files.port, files.status):
        path.unlink(missing_ok=True)
    if own.dir.is_symlink():
        own.dir.unlink()
    elif own.dir.exists():
        shutil.rmtree(own.dir)
    own.dir.mkdir(mode=0o700)
    for path in (own.status, own.listening, own.log):
        _make(path)
    files.status.symlink_to(own.status.relative_to(files.status.parent))
    for path in (files.ended, files.ended.with_name(f"{files.ended.name}.pending")):
        _make(path, ENDED_BYTES)


def _log_tail(path: Path, chars: int = 2000) -> str:
    try:
        text = path.read_text(errors="replace").strip()
    except OSError:
        return ""
    return text[-chars:]


def _remove(client: DockerClient, container: str) -> None:
    try:
        client.remove_container(container, force=True)
    except NotFound:
        pass


def _write_whole(path: Path, text: str) -> None:
    partial = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    partial.write_text(text)
    partial.replace(path)


def open_gateway(client: DockerClient, seal: SealPlan, files: SealFiles, config: ContainerConfig, *,
                 name: str | None = None, wait_sec: float = LISTEN_WAIT_SEC, poll_sec: float = 0.1) -> Gateway:
    """Create and start the gateway *config* (``gateway_config``) for *seal*, and return it once
    both of its interceptor's listeners take connections, with the served port's host port in
    the port file. Raise ``GatewayFailed`` when it ends or does not listen within *wait_sec*;
    the gateway is then removed. A ``DockerError`` from the daemon also removes it."""
    if (seal.ingress_port is None) != (not config.port_bindings):
        raise ValueError("a gateway publishes exactly the served port of its seal")
    prepare(files)
    listening = gateway_files(files).listening
    container = client.create_container(config, name=name)
    try:
        client.start_container(container)
        deadline = time.monotonic() + wait_sec
        while True:
            heard = listening.read_text() == LISTENING
            # Read after the file, so a gateway that listened and then ended is seen ended.
            info = client.inspect_container(container)
            if not info.state.running:
                when = "after" if heard else "before"
                raise GatewayFailed(f"the seal gateway ended with status {info.state.exit_code} {when} its "
                                    f"interceptor listened: {_log_tail(gateway_files(files).log)}")
            if heard:
                break
            if time.monotonic() > deadline:
                raise GatewayFailed(f"the seal gateway's interceptor did not listen within {wait_sec:.0f} s: "
                                    f"{_log_tail(gateway_files(files).log)}")
            time.sleep(poll_sec)
        host_port = None
        if seal.ingress_port is not None:
            bound = info.ports.get(f"{seal.ingress_port}/tcp") or ()
            host_port = next((port for ip, port in bound if ip == config.port_bindings[0].host_ip), None)
            if host_port is None:
                raise GatewayFailed(f"Docker published the served port {seal.ingress_port} on no host port of "
                                    f"{config.port_bindings[0].host_ip}: {bound}")
            _write_whole(files.port, f"{host_port}\n")
    except BaseException:
        _remove(client, container)
        raise
    return Gateway(id=container, files=files, host_port=host_port)


def payload_config(config: ContainerConfig, gateway: Gateway) -> ContainerConfig:
    """*config*, the payload's container, in *gateway*'s network namespace. Refuse a payload that
    would hold a capability, publish a port or run as root: the seal holds only for a payload
    that can neither change the firewall nor set the interceptor's mark."""
    if config.cap_add:
        raise ValueError(f"a sealed payload holds no capability, not {list(config.cap_add)}")
    if config.port_bindings:
        raise ValueError("a sealed payload publishes no port: its gateway publishes the served one")
    if config.user.partition(":")[0] == "0":
        raise ValueError("a sealed payload never runs as root")
    return dataclasses.replace(config, network_mode=f"container:{gateway.id}")


def mark_started(gateway: Gateway) -> None:
    """Record that the payload starts: the launcher calls this right before it releases the
    payload's gate, so a started file says the payload's status is its own."""
    gateway.files.started.touch()


def close_gateway(client: DockerClient, gateway: Gateway, *, exit_code: int | None) -> None:
    """Stop and remove *gateway*. With *exit_code*, the payload's own exit status, which the
    launcher read as soon as the payload ended, first record the ending: the status and whether
    the gateway still ran, and so its interceptor, which ends it; into ``<ended>.pending``
    first, whose time is the payload's end, then the ended file, each over the bytes
    ``prepare`` made. Without one (the payload never started, or its end was not seen), record
    none. Raise ValueError for an ending of a payload that never started, or not 0 to 255; the
    gateway is removed all the same."""
    try:
        if exit_code is not None:
            if not gateway.files.started.exists():
                raise ValueError("a payload that never started has no ending")
            if not 0 <= exit_code <= 255:
                raise ValueError(f"an exit status is 0 to 255, not {exit_code}")
            try:
                alive = client.inspect_container(gateway.id).state.running
            except NotFound:
                alive = False
            _kill(client, gateway.id)
            record = f'{{"exit": {exit_code}, "interceptor": {"true" if alive else "false"}}}\n'.encode()
            pending = gateway.files.ended.with_name(f"{gateway.files.ended.name}.pending")
            for path in (pending, gateway.files.ended):
                fd = os.open(path, os.O_WRONLY)
                try:
                    os.pwrite(fd, record, 0)
                finally:
                    os.close(fd)
    finally:
        _remove(client, gateway.id)


def _kill(client: DockerClient, container: str) -> None:
    try:
        client.kill_container(container, signal.SIGKILL)
    except NotFound:
        pass
    except RequestRefused as e:
        if e.status != 409:  # 409: it is not running
            raise
