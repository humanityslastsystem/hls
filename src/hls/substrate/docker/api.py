"""The Docker Engine API, spoken over the daemon's Unix socket.

The Docker launcher and the image roles create, watch and remove containers, and load and tag
images, through ``DockerClient``. It speaks one API version, ``API_VERSION``: 1.44 is Docker
Engine 25.0, the first engine that loads an OCI archive and takes ``ReadOnlyForceRecursive`` bind
mounts. ``check`` refuses a daemon that is not a Docker Engine or does not serve that version.

Every call has a timeout. A failed call raises one of four errors: the daemon could not be reached
or did not answer in time (``DaemonUnreachable``), it is not a Docker Engine serving
``API_VERSION`` (``WrongDaemon``), it refused the request (``RequestRefused``), or the object the
call names does not exist (``NotFound``). Each is a ``DockerError``, which on its own means the
daemon's reply could not be read. No request body and no value of a container's environment is
put in an error; the module logs nothing.

``ContainerConfig`` holds what differs between hls's containers. What must hold for every one is
fixed in ``container_body`` and cannot be turned off: every capability dropped before ``cap_add``,
``no-new-privileges``, no privileged mode, a private cgroup namespace, no restart, no automatic
removal and no log driver, so the daemon stores no output."""

from __future__ import annotations

import json
import os
import queue
import re
import signal
import socket
import struct
import threading
import time
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

#: The Engine API version every request names.
API_VERSION = "1.44"
API_VERSION_TUPLE = (1, 44)
#: The daemon's socket when ``site.docker.socket`` names none.
DEFAULT_SOCKET = Path("/var/run/docker.sock")
#: Seconds any call may wait that neither waits for a container nor moves an image: creating a
#: container or starting one takes seconds, not minutes.
CALL_TIMEOUT = 60.0
#: The OCI runtime of every container: the daemon's own runc, never a runtime of the NVIDIA
#: Container Toolkit's, even where the daemon's default is one, so no hook changes a container.
RUNTIME = "runc"
#: The name a Docker daemon's process gives itself.
DAEMON_COMM = "dockerd"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
_REF = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/:@+-]*")
_REPO = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*")
_TAG = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}")
_USER = re.compile(r"[0-9]+:[0-9]+")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAP = re.compile(r"[A-Z][A-Z_]*")
#: A systemd slice: a unit name without "@", whose dashes name its parents.
_SLICE = re.compile(r"[A-Za-z0-9_.:]+(?:-[A-Za-z0-9_.:]+)*\.slice")
#: A cgroupfs path: components that are not "." or "..", and no slice name as its last one.
_CGROUP_PATH = re.compile(r"/?(?!\.\.?(?:/|$))[A-Za-z0-9_.-]+(?:/(?!\.\.?(?:/|$))[A-Za-z0-9_.-]+)*(?<!\.slice)")
_LOADED = re.compile(r"Loaded image(?: ID)?: (\S+)")
#: The longest piece of a daemon's message an error carries.
_MESSAGE_CHARS = 2000


class DockerError(Exception):
    """A reply from the Docker daemon that hls cannot read."""


class DaemonUnreachable(DockerError):
    """The daemon's socket could not be reached, the connection broke, or the daemon did not
    answer within the call's timeout."""


class WrongDaemon(DockerError):
    """The socket is served by something other than a Docker Engine serving ``API_VERSION``."""


class RequestRefused(DockerError):
    """The daemon answered and refused the request. ``status`` is its HTTP status (409 for a
    container in the wrong state, 400 for a bad request), ``message`` its reason."""

    def __init__(self, what: str, status: int, message: str) -> None:
        super().__init__(f"the Docker daemon refused {what} (HTTP {status}): {message}")
        self.status = status
        self.message = message


class NotFound(DockerError):
    """The container or image a call names does not exist."""

    def __init__(self, what: str, message: str) -> None:
        super().__init__(f"{what}: not found: {message}")
        self.message = message


@dataclass(frozen=True)
class BindMount:
    """A bind mount. The daemon refuses a *source* that does not exist rather than making it."""

    source: Path
    target: str
    read_only: bool

    def __post_init__(self) -> None:
        if not self.source.is_absolute() or not self.target.startswith("/"):
            raise ValueError(f"a bind mount needs absolute paths: {self.source} -> {self.target}")


@dataclass(frozen=True)
class Device:
    """A device node of the host's, *source*, at *target* in a container, which the container's
    device cgroup lets it read and write and not create (``CgroupPermissions`` ``rw``)."""

    source: Path
    target: str

    def __post_init__(self) -> None:
        if not self.source.is_absolute() or not self.target.startswith("/dev/"):
            raise ValueError(f"a device is a host path given at a path under /dev: {self.source} -> {self.target}")


@dataclass(frozen=True)
class Ulimit:
    name: str
    soft: int
    hard: int


@dataclass(frozen=True)
class PortBinding:
    """A container's TCP port published on *host_ip*; Docker picks a free host port when
    *host_port* is None, and ``ContainerInfo.ports`` says which."""

    container_port: int
    host_ip: str
    host_port: int | None = None


@dataclass(frozen=True)
class ContainerConfig:
    """One container as hls creates it.

    *argv* is the whole command: it is the container's entrypoint and the image's entrypoint and
    command are not used. *user* is ``uid:gid`` and never left to the image. *env* holds declared
    non-secret variables only; its values never appear in a repr or an error. *cpuset*,
    *memory_bytes* and *pids_limit* are required, so no container is made without limits; the
    swap limit is *memory_bytes* plus *swap_bytes*. *devices* are the only device nodes it gets
    beside the runtime's own, a GPU's among them (``hls.substrate.docker.nvidia``).
    *seccomp_profile* is a profile's JSON text and *apparmor_profile* a loaded profile's name;
    None keeps the daemon's default. *unconfine_apparmor* is ``apparmor=unconfined``, which no
    profile name stands for, and *unmask_system_paths* is ``systempaths=unconfined``.
    *network_mode* is ``host``, ``bridge``, ``none`` or ``container:<id>``. *cgroup_parent* is the
    cgroup the container's own is made under, None for the daemon's default: a slice name such as
    ``hls-7.slice`` under the systemd cgroup driver, a path under cgroupfs. *group_add* are
    supplementary gids for the container's process, which otherwise has none; gid 0 is refused."""

    image: str
    argv: tuple[str, ...]
    user: str
    labels: Mapping[str, str]
    network_mode: str
    cpuset: tuple[int, ...]
    memory_bytes: int
    pids_limit: int
    env: Mapping[str, str] = field(default_factory=dict, repr=False)
    working_dir: str | None = None
    swap_bytes: int = 0
    shm_size_bytes: int | None = None
    ulimits: tuple[Ulimit, ...] = ()
    mounts: tuple[BindMount, ...] = ()
    port_bindings: tuple[PortBinding, ...] = ()
    cap_add: tuple[str, ...] = ()
    seccomp_profile: str | None = field(default=None, repr=False)
    apparmor_profile: str | None = None
    unconfine_apparmor: bool = False
    unmask_system_paths: bool = False
    devices: tuple[Device, ...] = ()
    oom_score_adj: int | None = None
    cgroup_parent: str | None = None
    group_add: tuple[int, ...] = ()
    init: bool = True
    read_only_root: bool = True

    def __post_init__(self) -> None:
        _check_ref(self.image)
        for gid in self.group_add:
            if type(gid) is not int or gid <= 0:
                raise ValueError(f"group_add holds gids above 0, not {gid!r}")
        if self.cgroup_parent is not None and not (_SLICE.fullmatch(self.cgroup_parent)
                                                   or _CGROUP_PATH.fullmatch(self.cgroup_parent)):
            raise ValueError(f"a cgroup parent is a slice name or a cgroup path, not {self.cgroup_parent!r}")
        if not self.argv or not self.argv[0]:
            raise ValueError("a container needs a command")
        if not _USER.fullmatch(self.user):
            raise ValueError(f"a container's user is uid:gid, not {self.user!r}")
        for name, value in self.env.items():
            if not _ENV_NAME.fullmatch(name) or "\0" in value:
                raise ValueError(f"environment variable {name!r} has a name or value Docker cannot carry")
        if not (self.network_mode in ("host", "bridge", "none")
                or (self.network_mode.startswith("container:") and _NAME.fullmatch(self.network_mode[10:]))):
            raise ValueError(f"unknown network mode {self.network_mode!r}")
        if self.port_bindings and self.network_mode != "bridge":
            raise ValueError("ports are published only from a container on the bridge network")
        if not self.cpuset or min(self.cpuset) < 0:
            raise ValueError(f"a container needs a cpuset of CPU numbers, not {self.cpuset!r}")
        if self.memory_bytes <= 0 or self.swap_bytes < 0 or self.pids_limit <= 0:
            raise ValueError("a container needs a positive memory limit and pids limit")
        if self.shm_size_bytes is not None and self.shm_size_bytes <= 0:
            raise ValueError("a shm size is positive")
        for cap in self.cap_add:
            if not _CAP.fullmatch(cap) or cap == "ALL" or cap.startswith("CAP_"):
                raise ValueError(f"cap_add names one capability without CAP_, not {cap!r}")
        if self.seccomp_profile is not None:
            try:
                profile = json.loads(self.seccomp_profile)
            except ValueError as e:
                raise ValueError(f"the seccomp profile is not JSON: {e}") from None
            if not isinstance(profile, dict):
                raise ValueError("the seccomp profile is not a JSON object")
        if self.apparmor_profile is not None and (
                not _NAME.fullmatch(self.apparmor_profile) or self.apparmor_profile == "unconfined"):
            raise ValueError(f"an AppArmor profile is a loaded profile's name, not {self.apparmor_profile!r}")
        if self.apparmor_profile is not None and self.unconfine_apparmor:
            raise ValueError("a container runs under a loaded AppArmor profile or under none, not both")
        if self.working_dir is not None and not self.working_dir.startswith("/"):
            raise ValueError(f"a working directory is absolute, not {self.working_dir!r}")


def container_body(cfg: ContainerConfig) -> dict[str, Any]:
    """The ``POST /containers/create`` body for *cfg*."""
    security = ["no-new-privileges"]
    if cfg.seccomp_profile is not None:
        security.append(f"seccomp={cfg.seccomp_profile}")
    if cfg.apparmor_profile is not None:
        security.append(f"apparmor={cfg.apparmor_profile}")
    if cfg.unconfine_apparmor:
        security.append("apparmor=unconfined")
    mounts = []
    for m in cfg.mounts:
        options: dict[str, Any] = {"Propagation": "rprivate", "NonRecursive": False, "CreateMountpoint": False}
        if m.read_only:
            # Fail rather than leave a submount of a read-only bind writable.
            options["ReadOnlyForceRecursive"] = True
        mounts.append({"Type": "bind", "Source": str(m.source), "Target": m.target,
                       "ReadOnly": m.read_only, "BindOptions": options})
    ports = {f"{p.container_port}/tcp": [{"HostIp": p.host_ip,
                                          "HostPort": "" if p.host_port is None else str(p.host_port)}]
             for p in cfg.port_bindings}
    host: dict[str, Any] = {
        "CapDrop": ["ALL"],
        "CapAdd": list(cfg.cap_add),
        "SecurityOpt": security,
        "Privileged": False,
        "Runtime": RUNTIME,
        "CgroupnsMode": "private",
        "CpusetCpus": ",".join(str(c) for c in sorted(set(cfg.cpuset))),
        "Memory": cfg.memory_bytes,
        "MemorySwap": cfg.memory_bytes + cfg.swap_bytes,
        "PidsLimit": cfg.pids_limit,
        "Ulimits": [{"Name": u.name, "Soft": u.soft, "Hard": u.hard} for u in cfg.ulimits],
        "Mounts": mounts,
        "NetworkMode": cfg.network_mode,
        "PortBindings": ports,
        "PublishAllPorts": False,
        "Init": cfg.init,
        "ReadonlyRootfs": cfg.read_only_root,
        "LogConfig": {"Type": "none", "Config": {}},
        "RestartPolicy": {"Name": "no"},
        "AutoRemove": False,
    }
    if cfg.shm_size_bytes is not None:
        host["ShmSize"] = cfg.shm_size_bytes
    if cfg.oom_score_adj is not None:
        host["OomScoreAdj"] = cfg.oom_score_adj
    if cfg.cgroup_parent is not None:
        host["CgroupParent"] = cfg.cgroup_parent
    if cfg.group_add:
        host["GroupAdd"] = [str(gid) for gid in cfg.group_add]
    if cfg.devices:
        host["Devices"] = [{"PathOnHost": str(d.source), "PathInContainer": d.target, "CgroupPermissions": "rw"}
                           for d in cfg.devices]
    if cfg.unmask_system_paths:
        # What the CLI sends for systempaths=unconfined: empty lists, where null keeps the defaults.
        host["MaskedPaths"] = []
        host["ReadonlyPaths"] = []
    return {
        "Image": cfg.image,
        # A non-empty entrypoint keeps the daemon from appending the image's command.
        "Entrypoint": list(cfg.argv),
        "Cmd": [],
        "User": cfg.user,
        "Env": [f"{name}={value}" for name, value in cfg.env.items()],
        "Labels": dict(cfg.labels),
        "WorkingDir": cfg.working_dir or "",
        "ExposedPorts": {key: {} for key in ports},
        "AttachStdin": False,
        "AttachStdout": False,
        "AttachStderr": False,
        "OpenStdin": False,
        "StdinOnce": False,
        "Tty": False,
        "HostConfig": host,
    }


@dataclass(frozen=True)
class EngineVersion:
    version: str
    api_version: str
    min_api_version: str
    os: str
    arch: str
    kernel_version: str
    components: tuple[str, ...]


@dataclass(frozen=True)
class EngineInfo:
    """What ``GET /info`` says of the daemon. *id* names its data root and is the same after a
    restart; ``DaemonIdentity`` tells one daemon process from the next. *containerd_store* is
    whether images live in containerd's store, where an image's ID is its manifest's or index's
    digest; in the classic store it is its config's digest."""

    id: str
    name: str
    server_version: str
    operating_system: str
    kernel_version: str
    cgroup_driver: str
    cgroup_version: str
    live_restore: bool
    security_options: tuple[str, ...]
    ncpu: int
    mem_total: int
    docker_root_dir: str
    storage_driver: str
    containerd_store: bool
    runtimes: tuple[str, ...]
    default_runtime: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class DaemonIdentity:
    """One daemon process: the engine's ID, its pid and its start time in clock ticks after boot
    (field 22 of ``/proc/<pid>/stat``), which tells it from a later process given the same pid."""

    engine_id: str
    pid: int
    start_ticks: int


@dataclass(frozen=True)
class ContainerState:
    """*started_at* and *finished_at* are None until the container first started or ended."""

    status: str
    running: bool
    pid: int
    exit_code: int
    oom_killed: bool
    error: str
    started_at: datetime | None
    finished_at: datetime | None


@dataclass(frozen=True)
class ContainerInfo:
    """*ports* maps ``<port>/tcp`` to the ``(host_ip, host_port)`` pairs it is published on."""

    id: str
    name: str
    image: str
    labels: dict[str, str]
    state: ContainerState
    ports: dict[str, tuple[tuple[str, int], ...]]


@dataclass(frozen=True)
class ContainerSummary:
    id: str
    names: tuple[str, ...]
    image: str
    labels: dict[str, str]
    state: str


@dataclass(frozen=True)
class ImageSummary:
    """An image as the daemon lists it: its ID and every name it has."""

    id: str
    repo_tags: tuple[str, ...]


@dataclass(frozen=True)
class WaitResult:
    status_code: int
    error: str


@dataclass(frozen=True)
class Event:
    """One daemon event. *attributes* holds the actor's labels and the action's details: a
    ``kill`` names its ``signal``, a ``die`` its ``exitCode``."""

    type: str
    action: str
    actor_id: str
    attributes: dict[str, str]
    time_ns: int


@dataclass(frozen=True)
class ImageInfo:
    """*id* is the config's digest in the classic store and the manifest's or index's digest in
    containerd's store."""

    id: str
    repo_tags: tuple[str, ...]
    repo_digests: tuple[str, ...]
    labels: dict[str, str]
    architecture: str
    os: str
    size: int
    #: The image's configuration as the daemon reports it (``Config``: Env, Entrypoint, Cmd,
    #: WorkingDir, User, Labels and the rest), which ``commit_container`` can give an image.
    config: dict[str, Any] = field(default_factory=dict, compare=False)


def _check_ref(ref: str) -> None:
    if not _REF.fullmatch(ref):
        raise ValueError(f"not an image reference: {ref!r}")


def _check_name(name: str) -> None:
    if not _NAME.fullmatch(name):
        raise ValueError(f"not a container ID or name: {name!r}")


def _time(text: str) -> datetime | None:
    """A Docker timestamp, or None for Go's zero time, which marks an event that has not happened."""
    when = datetime.fromisoformat(text)
    return None if when.year == 1 else when


def _since(ns: int) -> str:
    return f"{ns // 1_000_000_000}.{ns % 1_000_000_000:09d}"


def _filters(**values: Collection[str]) -> str:
    return json.dumps({key: sorted(v) for key, v in values.items() if v})


def _labels(raw: Any) -> dict[str, str]:
    return {str(k): str(v) for k, v in (raw or {}).items()}


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split("."))


def _container_info(d: dict[str, Any]) -> ContainerInfo:
    s = d["State"]
    state = ContainerState(
        status=s["Status"], running=bool(s["Running"]), pid=int(s["Pid"]), exit_code=int(s["ExitCode"]),
        oom_killed=bool(s["OOMKilled"]), error=s.get("Error") or "",
        started_at=_time(s["StartedAt"]), finished_at=_time(s["FinishedAt"]))
    ports: dict[str, tuple[tuple[str, int], ...]] = {}
    for key, bound in ((d.get("NetworkSettings") or {}).get("Ports") or {}).items():
        ports[key] = tuple((b["HostIp"], int(b["HostPort"])) for b in bound or ())
    return ContainerInfo(id=d["Id"], name=d["Name"].lstrip("/"), image=d["Image"],
                         labels=_labels((d.get("Config") or {}).get("Labels")), state=state, ports=ports)


def _event(d: dict[str, Any]) -> Event:
    actor = d.get("Actor") or {}
    return Event(type=d["Type"], action=d["Action"], actor_id=actor.get("ID", ""),
                 attributes=_labels(actor.get("Attributes")), time_ns=int(d["timeNano"]))


def _image_info(d: dict[str, Any]) -> ImageInfo:
    return ImageInfo(id=d["Id"], repo_tags=tuple(d.get("RepoTags") or ()),
                     repo_digests=tuple(d.get("RepoDigests") or ()),
                     labels=_labels((d.get("Config") or {}).get("Labels")),
                     architecture=d.get("Architecture", ""), os=d.get("Os", ""), size=int(d.get("Size", 0)),
                     config=dict(d.get("Config") or {}))


def _engine_info(d: dict[str, Any]) -> EngineInfo:
    status = {str(k): str(v) for k, v in d.get("DriverStatus") or ()}
    return EngineInfo(
        id=d["ID"], name=d["Name"], server_version=d["ServerVersion"],
        operating_system=d.get("OperatingSystem", ""), kernel_version=d.get("KernelVersion", ""),
        cgroup_driver=d.get("CgroupDriver", ""), cgroup_version=d.get("CgroupVersion", ""),
        live_restore=bool(d.get("LiveRestoreEnabled")),
        security_options=tuple(d.get("SecurityOptions") or ()), ncpu=int(d["NCPU"]),
        mem_total=int(d["MemTotal"]), docker_root_dir=d.get("DockerRootDir", ""),
        storage_driver=d.get("Driver", ""),
        containerd_store=status.get("driver-type") == "io.containerd.snapshotter.v1",
        runtimes=tuple(sorted(d.get("Runtimes") or {})), default_runtime=d.get("DefaultRuntime", ""),
        warnings=tuple(d.get("Warnings") or ()))


def _engine_version(d: dict[str, Any]) -> EngineVersion:
    return EngineVersion(
        version=d["Version"], api_version=d["ApiVersion"], min_api_version=d["MinAPIVersion"],
        os=d.get("Os", ""), arch=d.get("Arch", ""), kernel_version=d.get("KernelVersion", ""),
        components=tuple(str(c.get("Name", "")) for c in d.get("Components") or ()))


def _read(parse, data: Any, what: str):
    try:
        return parse(data)
    except (KeyError, TypeError, ValueError, AttributeError) as e:
        raise DockerError(f"the Docker daemon's reply to {what} is not in the form expected: "
                          f"{type(e).__name__}: {e}") from None


def _stat_fields(proc: Path, pid: int) -> tuple[str, list[str]]:
    """The command name and the fields after it in ``<proc>/<pid>/stat``; the name is in
    parentheses and may itself hold spaces and parentheses, so the fields start after the last
    closing one."""
    raw = (proc / str(pid) / "stat").read_text(errors="replace")
    head, _, rest = raw.rpartition(")")
    return head.partition("(")[2], rest.split()


def daemon_pid(listener: int, *, proc: Path = Path("/proc"), comm: str = DAEMON_COMM) -> int:
    """The pid of the daemon that serves a socket whose listening end *listener* made.

    A daemon that opened its own socket is the listener. Under socket activation (``dockerd -H
    fd://``) the service manager made the socket and started the daemon as its child, so the
    daemon is the one child of the listener named *comm*. Raise ``DockerError`` when neither
    names exactly one process."""
    try:
        name, _ = _stat_fields(proc, listener)
    except OSError as e:
        raise DockerError(f"cannot read the process {listener} that listens on the Docker socket: {e}") from None
    if name == comm:
        return listener
    children = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            child, fields = _stat_fields(proc, int(entry.name))
        except (OSError, ValueError):
            continue  # the process ended, or its stat is not readable
        if child == comm and len(fields) > 1 and fields[1] == str(listener):
            children.append(int(entry.name))
    if len(children) != 1:
        raise DockerError(f"the Docker socket's listener is {name!r} (pid {listener}), which has "
                          f"{len(children)} children named {comm!r}; the daemon cannot be identified")
    return children[0]


def start_ticks(pid: int, *, proc: Path = Path("/proc")) -> int:
    """Field 22 of ``<proc>/<pid>/stat``: when the process started, in clock ticks after boot."""
    _, fields = _stat_fields(proc, pid)
    # The fields after the name start at field 3.
    return int(fields[22 - 3])


class EventStream:
    """Events as the daemon sends them, read on a thread of its own, so that ``get`` can wait
    with a timeout. The stream ends only when it is closed or the daemon goes away."""

    _END = object()

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._queue: queue.Queue[Any] = queue.Queue()
        self._closing = False
        self._thread = threading.Thread(target=self._read, name="docker-events", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        try:
            for line in self._response.iter_lines():
                if line.strip():
                    self._queue.put(_read(_event, json.loads(line), "GET /events"))
        except Exception as e:  # handed to the reader by get()
            self._queue.put(e)
        else:
            self._queue.put(self._END)

    def get(self, timeout: float) -> Event | None:
        """The next event, or None when none came within *timeout* seconds. Raise
        ``DaemonUnreachable`` once the daemon has ended the stream, and ``DockerError`` once the
        stream was closed."""
        if self._closing:
            raise DockerError("the Docker daemon's events stream was closed")
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._END or isinstance(item, (httpx.TransportError, OSError)):
            self._queue.put(item)
            raise DaemonUnreachable("the Docker daemon's events stream ended") from None
        if isinstance(item, Exception):
            self._queue.put(item)
            if isinstance(item, DockerError):
                raise item
            raise DockerError(f"the Docker daemon's events stream is not in the form expected: {item}")
        return item

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        stream = self._response.extensions.get("network_stream")
        sock = stream.get_extra_info("socket") if stream is not None else None
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)  # wakes the reader blocked in recv
            except OSError:
                pass
        self._response.close()
        self._thread.join(timeout=CALL_TIMEOUT)

    def __enter__(self) -> EventStream:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class DockerClient:
    """A client of the Docker daemon on *socket*. Call ``check`` once before anything else; it
    tells a dead or wrong daemon from a failed call. Safe to share between threads."""

    def __init__(self, socket_path: Path = DEFAULT_SOCKET, *, timeout: float = CALL_TIMEOUT) -> None:
        self.socket = socket_path
        self.timeout = timeout
        self._http = httpx.Client(
            transport=httpx.HTTPTransport(uds=str(socket_path), retries=0),
            base_url="http://docker", trust_env=False, timeout=timeout, follow_redirects=False)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> DockerClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- transport -------------------------------------------------------------------------

    def _unreachable(self, what: str, e: Exception, timeout: float | None) -> DaemonUnreachable:
        if isinstance(e, httpx.TimeoutException):
            return DaemonUnreachable(f"the Docker daemon at {self.socket} did not answer {what} "
                                     f"within {timeout} s")
        return DaemonUnreachable(f"the Docker daemon at {self.socket} cannot be reached for {what}: "
                                 f"{type(e).__name__}: {e}")

    def _send(self, method: str, path: str, *, versioned: bool = True, params: dict[str, str] | None = None,
              json_body: Any = None, content: Any = None, headers: dict[str, str] | None = None,
              timeout: httpx.Timeout | float | None = None, stream: bool = False) -> httpx.Response:
        url = f"/v{API_VERSION}{path}" if versioned else path
        timeout = self.timeout if timeout is None else timeout
        request = self._http.build_request(method, url, params=params, json=json_body, content=content,
                                           headers=headers, timeout=timeout)
        try:
            return self._http.send(request, stream=stream)
        except httpx.TransportError as e:
            limit = timeout.read if isinstance(timeout, httpx.Timeout) else timeout
            raise self._unreachable(f"{method} {path}", e, limit) from e

    def _refuse(self, what: str, response: httpx.Response, redact: Collection[str] = ()) -> DockerError:
        try:
            body = response.read()
        except httpx.TransportError as e:
            return self._unreachable(what, e, self.timeout)
        try:
            message = str(json.loads(body).get("message", ""))
        except (ValueError, AttributeError):
            message = body.decode(errors="replace")
        for value in redact:
            message = message.replace(value, "<redacted>")
        message = message.strip()[:_MESSAGE_CHARS]
        if response.status_code == 404:
            return NotFound(what, message)
        return RequestRefused(what, response.status_code, message)

    def _call(self, method: str, path: str, *, ok: Collection[int] = (200,), redact: Collection[str] = (),
              **kwargs: Any) -> httpx.Response:
        response = self._send(method, path, **kwargs)
        if response.status_code not in ok:
            raise self._refuse(f"{method} {path}", response, redact)
        return response

    def _json(self, response: httpx.Response, what: str) -> Any:
        try:
            return response.json()
        except ValueError as e:
            raise DockerError(f"the Docker daemon's reply to {what} is not JSON: {e}") from None

    # -- the daemon ------------------------------------------------------------------------

    def version(self) -> EngineVersion:
        """``GET /version``, asked without an API version, so that a daemon too old for
        ``API_VERSION`` still says which it serves."""
        response = self._call("GET", "/version", versioned=False)
        return _read(_engine_version, self._json(response, "GET /version"), "GET /version")

    def check(self) -> EngineVersion:
        """The daemon's version, once it is known to be a Linux Docker Engine that serves
        ``API_VERSION``. Raise ``DaemonUnreachable`` for a dead daemon and ``WrongDaemon`` for
        another program or version on the socket."""
        version = self.version()
        if "Engine" not in version.components:
            raise WrongDaemon(f"{self.socket} is not served by a Docker Engine: its components are "
                              f"{list(version.components)}")
        if version.os != "linux":
            raise WrongDaemon(f"the Docker Engine at {self.socket} runs {version.os!r} containers, not linux")
        try:
            low, high = _version_tuple(version.min_api_version), _version_tuple(version.api_version)
        except ValueError:
            raise WrongDaemon(f"the Docker Engine at {self.socket} names API versions hls cannot read: "
                              f"{version.min_api_version!r} to {version.api_version!r}") from None
        if not low <= API_VERSION_TUPLE <= high:
            raise WrongDaemon(f"the Docker Engine {version.version} at {self.socket} serves API "
                              f"{version.min_api_version} to {version.api_version}; hls needs {API_VERSION}")
        return version

    def info(self) -> EngineInfo:
        response = self._call("GET", "/info")
        return _read(_engine_info, self._json(response, "GET /info"), "GET /info")

    def identity(self, *, proc: Path = Path("/proc")) -> DaemonIdentity:
        """The daemon process serving the socket. Two readings are equal exactly when the same
        process served both, so a restart in between shows as a change. The listening end's
        credentials (``SO_PEERCRED``) name the process that made the socket, from which
        ``daemon_pid`` finds the daemon."""
        engine_id = self.info().id
        peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            peer.settimeout(self.timeout)
            peer.connect(str(self.socket))
            pid, _, _ = struct.unpack("3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED,
                                                            struct.calcsize("3i")))
        except OSError as e:
            raise DaemonUnreachable(f"the Docker daemon at {self.socket} cannot be reached: {e}") from None
        finally:
            peer.close()
        if pid <= 0:
            raise DockerError(f"the process listening on {self.socket} is outside this PID namespace")
        daemon = daemon_pid(pid, proc=proc)
        try:
            return DaemonIdentity(engine_id=engine_id, pid=daemon, start_ticks=start_ticks(daemon, proc=proc))
        except (OSError, ValueError, IndexError) as e:
            raise DockerError(f"cannot read the start time of the Docker daemon, pid {daemon}: {e}") from None

    # -- containers ------------------------------------------------------------------------

    def create_container(self, config: ContainerConfig, *, name: str | None = None) -> str:
        """Create a container and return its ID. A warning from the daemon means it did not apply
        a setting as asked, so the container is removed and the request refused. A cgroup parent
        must have the form the daemon's cgroup driver reads: a slice under systemd, a path under
        cgroupfs; another form raises ValueError before anything is created."""
        params = {}
        if name is not None:
            _check_name(name)
            params["name"] = name
        if config.cgroup_parent is not None:
            driver = self.info().cgroup_driver
            slice_name = bool(_SLICE.fullmatch(config.cgroup_parent))
            if (driver, slice_name) not in (("systemd", True), ("cgroupfs", False)):
                raise ValueError(f"the Docker daemon's cgroup driver is {driver!r}, which does not take "
                                 f"the cgroup parent {config.cgroup_parent!r}")
        redact = [v for v in config.env.values() if len(v) >= 4]
        response = self._call("POST", "/containers/create", ok=(201,), params=params,
                              json_body=container_body(config), redact=redact)
        body = self._json(response, "POST /containers/create")
        try:
            container_id = str(body["Id"])
            warnings = [str(w) for w in body.get("Warnings") or ()]
        except (KeyError, TypeError, AttributeError):
            raise DockerError("the Docker daemon's reply to POST /containers/create has no Id") from None
        if warnings:
            self.remove_container(container_id, force=True)
            message = "; ".join(warnings)
            for value in redact:
                message = message.replace(value, "<redacted>")
            raise RequestRefused("POST /containers/create", response.status_code,
                                 f"it warned, so the container was removed: {message[:_MESSAGE_CHARS]}")
        return container_id

    def start_container(self, container: str) -> None:
        """Start a created container. One already running is refused with status 304."""
        _check_name(container)
        self._call("POST", f"/containers/{container}/start", ok=(204,))

    def inspect_container(self, container: str) -> ContainerInfo:
        _check_name(container)
        what = f"GET /containers/{container}/json"
        response = self._call("GET", f"/containers/{container}/json")
        return _read(_container_info, self._json(response, what), what)

    def wait_container(self, container: str, *, timeout: float) -> WaitResult | None:
        """Wait up to *timeout* seconds for the container to stop running and return how it
        ended, or None when it still runs. A container that never started or has already
        ended returns at once."""
        _check_name(container)
        what = f"POST /containers/{container}/wait"
        limit = httpx.Timeout(self.timeout, read=timeout)
        response = self._send("POST", f"/containers/{container}/wait", params={"condition": "not-running"},
                              timeout=limit, stream=True)
        try:
            if response.status_code != 200:
                raise self._refuse(what, response)
            try:
                body = response.read()
            except httpx.TimeoutException:
                return None
            except httpx.TransportError as e:
                raise self._unreachable(what, e, timeout) from e
        finally:
            response.close()
        try:
            data = json.loads(body)
        except ValueError as e:
            raise DockerError(f"the Docker daemon's reply to {what} is not JSON: {e}") from None
        return _read(lambda d: WaitResult(status_code=int(d["StatusCode"]),
                                          error=str((d.get("Error") or {}).get("Message") or "")), data, what)

    def kill_container(self, container: str, sig: signal.Signals) -> None:
        """Send *sig* to the container's main process. One that is not running is refused
        with status 409."""
        _check_name(container)
        self._call("POST", f"/containers/{container}/kill", ok=(204,), params={"signal": sig.name})

    def stop_container(self, container: str, *, grace: int, sig: signal.Signals = signal.SIGTERM) -> bool:
        """Send *sig*, and SIGKILL after *grace* seconds if the container still runs. False when
        it had already stopped."""
        _check_name(container)
        response = self._call("POST", f"/containers/{container}/stop", ok=(204, 304),
                              params={"t": str(grace), "signal": sig.name},
                              timeout=httpx.Timeout(self.timeout, read=grace + self.timeout))
        return response.status_code == 204

    def remove_container(self, container: str, *, force: bool = False) -> None:
        """Remove a container with its anonymous volumes; *force* kills it first if it runs."""
        _check_name(container)
        self._call("DELETE", f"/containers/{container}", ok=(204,),
                   params={"force": "1" if force else "0", "v": "1"})

    def list_containers(self, *, labels: Mapping[str, str]) -> list[ContainerSummary]:
        """Every container, running or not, that carries all of *labels*."""
        if not labels:
            raise ValueError("list containers by at least one label")
        params = {"all": "1", "filters": _filters(label=[f"{k}={v}" for k, v in labels.items()])}
        response = self._call("GET", "/containers/json", params=params)
        return _read(lambda rows: [
            ContainerSummary(id=r["Id"], names=tuple(n.lstrip("/") for n in r.get("Names") or ()),
                             image=r.get("Image", ""), labels=_labels(r.get("Labels")), state=r.get("State", ""))
            for r in rows], self._json(response, "GET /containers/json"), "GET /containers/json")

    def list_containers_named(self, prefix: str) -> list[ContainerSummary]:
        """Every container, running or not, one of whose names starts with *prefix*."""
        _check_name(prefix)
        params = {"all": "1", "filters": _filters(name=[prefix])}
        response = self._call("GET", "/containers/json", params=params)
        rows = _read(lambda rows: [
            ContainerSummary(id=r["Id"], names=tuple(n.lstrip("/") for n in r.get("Names") or ()),
                             image=r.get("Image", ""), labels=_labels(r.get("Labels")), state=r.get("State", ""))
            for r in rows], self._json(response, "GET /containers/json"), "GET /containers/json")
        # The daemon's name filter matches anywhere in a name.
        return [row for row in rows if any(n.startswith(prefix) for n in row.names)]

    # -- volumes ---------------------------------------------------------------------------

    def list_volumes(self, prefix: str) -> list[str]:
        """The names of every volume whose name starts with *prefix*."""
        _check_name(prefix)
        what = "GET /volumes"
        response = self._call("GET", "/volumes", params={"filters": _filters(name=[prefix])})
        names = _read(lambda d: [str(v["Name"]) for v in d.get("Volumes") or ()], self._json(response, what), what)
        return [name for name in names if name.startswith(prefix)]

    def remove_volume(self, name: str) -> None:
        """Remove the volume *name*. The daemon refuses (409) while a container uses it."""
        _check_name(name)
        self._call("DELETE", f"/volumes/{name}", ok=(204,))

    # -- events ----------------------------------------------------------------------------

    def _event_params(self, since_ns: int, until_ns: int | None, containers: Collection[str],
                      types: Collection[str], actions: Collection[str]) -> dict[str, str]:
        for c in containers:
            _check_name(c)
        params = {"since": _since(since_ns),
                  "filters": _filters(container=containers, type=types, event=actions)}
        if until_ns is not None:
            params["until"] = _since(until_ns)
        return params

    def events(self, *, since_ns: int, until_ns: int, containers: Collection[str] = (),
               types: Collection[str] = (), actions: Collection[str] = ()) -> list[Event]:
        """The events from *since_ns* to *until_ns* (nanoseconds after the epoch) that match
        every filter given. The daemon keeps only its most recent events (moby holds 256), so
        a reader that must see every event of a container keeps a ``watch_events`` stream open
        from before the container is created. An *until_ns* in the future waits for it."""
        wait = max(0.0, until_ns / 1e9 - time.time())
        params = self._event_params(since_ns, until_ns, containers, types, actions)
        response = self._call("GET", "/events", params=params,
                              timeout=httpx.Timeout(self.timeout, read=self.timeout + wait))
        out = []
        for line in response.text.splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                except ValueError as e:
                    raise DockerError(f"the Docker daemon's reply to GET /events is not JSON lines: {e}") from None
                out.append(_read(_event, data, "GET /events"))
        return out

    def watch_events(self, *, since_ns: int, containers: Collection[str] = (), types: Collection[str] = (),
                     actions: Collection[str] = ()) -> EventStream:
        """A live stream of the events after *since_ns* that match every filter given. Close
        it (it is a context manager) when done."""
        params = self._event_params(since_ns, None, containers, types, actions)
        response = self._send("GET", "/events", params=params,
                              timeout=httpx.Timeout(self.timeout, read=None), stream=True)
        if response.status_code != 200:
            try:
                raise self._refuse("GET /events", response)
            finally:
                response.close()
        return EventStream(response)

    # -- images ----------------------------------------------------------------------------

    def load_image(self, archive: Path, *, timeout: float) -> tuple[str, ...]:
        """Load an image archive (OCI layout or ``docker save`` form) and return what the daemon
        says it loaded: names, or ``sha256:`` IDs for images the archive does not name. No
        single read or write of the transfer waits longer than *timeout* seconds."""
        what = "POST /images/load"
        loaded: list[str] = []
        with archive.open("rb") as f:
            response = self._send("POST", "/images/load", params={"quiet": "1"}, content=f,
                                  headers={"Content-Type": "application/x-tar"},
                                  timeout=httpx.Timeout(timeout), stream=True)
            try:
                if response.status_code != 200:
                    raise self._refuse(what, response)
                try:
                    for line in response.iter_lines():
                        if not line.strip():
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError as e:
                            raise DockerError(f"the Docker daemon's reply to {what} is not JSON lines: {e}") from None
                        if not isinstance(data, dict):
                            raise DockerError(f"the Docker daemon's reply to {what} holds a line that is not an object")
                        if "error" in data or "errorDetail" in data:
                            detail = (data.get("errorDetail") or {}).get("message") or data.get("error")
                            raise RequestRefused(what, response.status_code, str(detail)[:_MESSAGE_CHARS])
                        found = _LOADED.search(str(data.get("stream", "")))
                        if found:
                            loaded.append(found.group(1))
                except httpx.TransportError as e:
                    raise self._unreachable(what, e, timeout) from e
            finally:
                response.close()
        if not loaded:
            raise DockerError(f"the Docker daemon loaded {archive} and named no image")
        return tuple(loaded)

    def list_images(self, *, repository: str) -> list[ImageSummary]:
        """Every image that has a name in the repository *repository*, with its names there: the
        daemon lists only the names the filter matches, not an image's names elsewhere."""
        if not _REPO.fullmatch(repository):
            raise ValueError(f"not a repository name: {repository!r}")
        what = "GET /images/json"
        response = self._call("GET", "/images/json", params={"filters": _filters(reference=[repository])})
        images = _read(lambda rows: [ImageSummary(id=r["Id"], repo_tags=tuple(r.get("RepoTags") or ()))
                                     for r in rows], self._json(response, what), what)
        kept = [ImageSummary(id=image.id, repo_tags=tuple(tag for tag in image.repo_tags
                                                         if tag.rpartition(":")[0] == repository))
                for image in images]
        return [image for image in kept if image.repo_tags]

    def inspect_image(self, ref: str) -> ImageInfo:
        _check_ref(ref)
        what = f"GET /images/{ref}/json"
        response = self._call("GET", f"/images/{ref}/json")
        return _read(_image_info, self._json(response, what), what)

    def tag_image(self, ref: str, *, repo: str, tag: str) -> None:
        """Give the image *ref* the name ``repo:tag``, moving the name from any image that had it."""
        _check_ref(ref)
        if not _REPO.fullmatch(repo) or not _TAG.fullmatch(tag):
            raise ValueError(f"not an image name: {repo}:{tag}")
        self._call("POST", f"/images/{ref}/tag", ok=(201,), params={"repo": repo, "tag": tag})

    def commit_container(self, container: str, *, repo: str, tag: str, config: Mapping[str, Any]) -> str:
        """Make an image of the stopped container *container*'s filesystem, named ``repo:tag``, with
        *config* as its configuration in place of the container's own (so the container's command,
        labels and user do not enter the image); return the new image's ID."""
        _check_name(container)
        if not _REPO.fullmatch(repo) or not _TAG.fullmatch(tag):
            raise ValueError(f"not an image name: {repo}:{tag}")
        what = "POST /commit"
        response = self._call("POST", "/commit", ok=(201,), json_body=dict(config),
                              params={"container": container, "repo": repo, "tag": tag, "pause": "0"})
        return _read(lambda d: str(d["Id"]), self._json(response, what), what)

    def _stream_to(self, what: str, path: str, dest: Path, *, timeout: float, params: dict[str, str] | None = None) -> None:
        """Stream the body of GET *path* into the new file *dest* (refusing one that exists), through a
        temporary file beside it that is renamed once the body is whole."""
        partial = dest.with_name(dest.name + ".partial")
        with open(partial, "xb") as out:
            try:
                response = self._send("GET", path, params=params, timeout=httpx.Timeout(timeout), stream=True)
                try:
                    if response.status_code != 200:
                        raise self._refuse(what, response)
                    try:
                        for chunk in response.iter_raw():
                            out.write(chunk)
                    except httpx.TransportError as e:
                        raise self._unreachable(what, e, timeout) from e
                finally:
                    response.close()
                out.flush()
                os.fsync(out.fileno())
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        if dest.exists():
            partial.unlink(missing_ok=True)
            raise FileExistsError(f"{dest} exists")
        os.rename(partial, dest)

    def save_image(self, ref: str, dest: Path, *, timeout: float) -> None:
        """Write the image *ref* as the daemon exports it (``docker save``: an OCI layout with a
        ``manifest.json``) into the new file *dest*. No single read or write waits longer than
        *timeout* seconds."""
        _check_ref(ref)
        self._stream_to(f"GET /images/{ref}/get", f"/images/{ref}/get", dest, timeout=timeout)

    def container_archive(self, container: str, path: str, dest: Path, *, timeout: float) -> None:
        """Write what the container *container* holds at the absolute *path*, as a tar the daemon
        reads as root, into the new file *dest*. NotFound when it holds nothing there."""
        _check_name(container)
        if not path.startswith("/"):
            raise ValueError(f"a path in a container is absolute, not {path!r}")
        self._stream_to(f"GET /containers/{container}/archive", f"/containers/{container}/archive", dest,
                        timeout=timeout, params={"path": path})

    def remove_image(self, ref: str, *, force: bool = False) -> None:
        """Remove the name *ref*, and the image once no name or container holds it."""
        _check_ref(ref)
        self._call("DELETE", f"/images/{ref}", params={"force": "1" if force else "0", "noprune": "0"})
