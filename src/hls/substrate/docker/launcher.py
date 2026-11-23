"""Role steps on Docker: one role container per role, started through the Engine API.

Every role runs in a container of its own, held to the role's slice by Docker's limits and
started behind the gate (``gate.sh``), a shell script the launcher puts in the role's
directory with three FIFOs, bound read-only at ``/.hls-gate``. The gate sends the
container's stdout and stderr into two of them, which the log cappers read
(``hls.substrate.log_cap``), and waits on the third until the launcher writes the role's
variables into it. So nothing of the role runs before the launcher has read the container's
cgroup, and no variable of a role's env file is in the container's configuration or argv.

- A harness role (``RoleSpec.argv``) runs the step wrapper (``hls.substrate.role_exec``) in the
  step image, with hls's checkout and interpreter bound read-only at their own paths
  (``code_dirs``) and the directories its roles write bound read-write at theirs. The wrapper
  reads its step id and its limit directory as ``hls.substrate.docker.step`` says, writes the
  start marker and the grant report and then becomes the role's command, as on any substrate.
  Its report names the cgroup as the container sees it; the handle rewrites it to the
  container's cgroup on the host, which the runner's memory watch reads.
- A container role (``RoleSpec.container``) runs its own image. The launcher writes its start
  marker once the container started, measures its grant from the host while the gate waits
  (``hls.substrate.docker.grant``), checks it as the step wrapper checks a step
  (``role_exec.check_role``, ``split_refusal``) and writes the same report. A grant that differs
  stops the container before anything of the role ran, and the role's status is
  ``GRANT_MISMATCH_EXIT``. The role's env file is taken as the step wrapper takes one
  (``role_exec.take_env_file``) and its variables go through the FIFO.

Every container runs as the host user, never root, and knows it by name as an Apptainer container
does: the image's own ``/etc/passwd`` and ``/etc/group`` with the host user's entry and the host
group's appended (``user_files``), bound read-only, read through a helper container of the image
that is created with the job's labels, never started, and removed. An image whose files cannot be
read refuses the role's start. Every container has every capability dropped,
``no-new-privileges``, a read-only root, bind mounts only, the host's network (as an unsealed
Apptainer container shares the node's), an init process, its cpuset, a memory limit with no swap,
a pids limit, the runner's own ``RLIMIT_NPROC`` as a Slurm step inherits it (a role with a process
ceiling of its own has that instead), a shm of its memory and, for a contestant, ``oom_score_adj``
1000, which the step
wrapper or the gate raises from inside the container, as an unprivileged process may, so a process
of the role may lower its own again as on Slurm; Docker's own setting would be the floor of every
process in the container, the build sandbox's exec server among them. Every container runs
under runc, never a runtime of the NVIDIA Container Toolkit's, and a role's GPUs, taken by UUID
from the allocation's reservation, are its device nodes with the driver's files
(``hls.substrate.docker.nvidia``): the launcher reads each GPU's minor number from the host when
it is made, and the gate (``--nvidia``) puts the driver's libraries and programs on the role's
paths. ``CUDA_VISIBLE_DEVICES`` and ``NVIDIA_VISIBLE_DEVICES`` name the role's GPUs, the latter
``void`` for a container with none, as on a host with the toolkit. A role that shares the
CPUs of the others (``RoleSpec.overlap``) or the GPUs of another (``share_gpus_of``) takes the
allocation's first CPUs, which other roles may hold, as Slurm places an overlap step; any other
role holds CPUs and GPUs no other running role holds.

Every container keeps the daemon's seccomp profile, AppArmor profile and masked ``/proc`` paths
but one: a container role whose spec asks for the ``spawner`` confinement
(``ContainerSpec.confinement``), which only the agent build's sandbox does. Its spawner makes the
commands' user, mount and PID namespaces and mounts a ``/proc`` of the new PID namespace
(``hls.agent.spawner``), which each of the three refuses. That container gets, and is the only one
to get:

- the spawner's seccomp profile (``hls.substrate.docker.seccomp``): Docker's default with
  ``unshare`` of exactly those three namespaces and ``mount`` added;
- no AppArmor profile (``apparmor=unconfined``): Docker's own profile denies every mount, and a
  narrower one must be loaded into the kernel by root, which hls never is;
- no masked or read-only ``/proc`` paths (``systempaths=unconfined``): the kernel mounts a new
  ``/proc`` only where the container's own is fully visible.

It stays everything else above: the host user, no capability, ``no-new-privileges`` (a user
namespace needs no privilege an exec could gain), no daemon socket, and its seal as any sealed
container's. A host whose kernel still refuses those namespaces to an unprivileged user is refused
before the build starts (``hls.substrate.docker.substrate.spawner_problem``).

The end. A role's status is its container's exit code. It ended through the scheduler
(``ended_by_scheduler``) when the launcher did not stop it and the daemon's events show a ``kill``
or ``stop`` the launcher did not send, when the daemon serving the socket is another process than
at the launch, when the events stream opened before the container was created broke before its
end was read, when its state could not be read, or when it reached its time limit, which the
launcher enforces as Slurm enforces a step's. The daemon keeps few events, so one stream per
launcher, opened before the first container is created, records every event of the job's
containers. Each container is removed once its state was read.

Its end state is OUT_OF_MEMORY when Docker says it was OOM killed or the ``oom_kill`` count of its
parent cgroup rose between its creation and its end. Docker's own flag misses some kills at the
memory limit, and the container's cgroup is gone once it ended, so each container is created
under a parent cgroup that outlives it (``cgroup_parent``): a slot of the allocation's, named by
the allocation's first CPU, which no other running allocation holds, and a number no other
running role of the launcher holds. The daemon makes it on first use and nothing removes it, so
slots are reused and the parents on a host stay as many as roles that ran at once. A count that
cannot be read, or a container not under its slot, leaves the end unknown: ended through the
scheduler, with no end state, never "not out of memory".

A sealed container role (``ContainerSpec.seal``) runs behind a gateway (``hls.substrate.docker.seal``):
the launcher opens the gateway, which owns the network namespace the payload joins, before it creates
the payload; marks the payload started right before it releases the gate; and closes the gateway
with the payload's exit status as soon as it has read the payload's end, which records the ending
the stage reads (``hls.substrate.seal.payload_ending``). A gateway that does not seal its namespace
ends the role before its payload is created, as the runtime's failure. The gateway's memory
(``gateway_memory_gib``) counts against the allocation beside the role's, and the gateway carries the
role's labels, so the job's end stops it too. The payload makes no user namespace, so the seal's
per-namespace kernel-count limits (``ContainerRole.ucounts``) are not applied; the role's stderr
says so."""

from __future__ import annotations

import errno
import grp
import json
import logging
import os
import pwd
import resource
import shlex
import shutil
import signal
import stat
import subprocess
import tarfile
import threading
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from hls.store.digest import check_sha256_hex, sha256_file
from hls.substrate.base import JobState, Launcher, RoleFailed, RoleHandle, RoleReport, RoleSpec
from hls.substrate.cgroup import CGROUP_ROOT, kill_all, pids, read_keyed
from hls.substrate.container import ContainerRuntime, ContainerSpec
from hls.substrate.docker.api import (
    CALL_TIMEOUT,
    BindMount,
    ContainerConfig,
    DaemonIdentity,
    Device,
    DockerClient,
    DockerError,
    Event,
    EventStream,
    NotFound,
    Ulimit,
)
from hls.substrate.docker.grant import GpusUnknown, container_cgroup, gpus_by_minor, measure_container
from hls.substrate.docker.images import remove_job_leftovers
from hls.substrate.docker.nvidia import DRIVER_ROOT, DriverUnusable, NvidiaDriver, driver_devices, driver_version
from hls.substrate.docker.seal import (
    GATEWAY_LABEL,
    Gateway,
    GatewayFailed,
    close_gateway,
    gateway_config,
    mark_started,
    open_gateway,
    payload_config,
)
from hls.substrate.docker.seccomp import spawner_profile
from hls.substrate.docker.step import STEP_CGROUP, STEP_ID_VAR
from hls.substrate.exits import GRANT_MISMATCH_EXIT, PROCESS_CEILING_EXIT
from hls.substrate.factory import SUBSTRATE_VAR
from hls.substrate.local.stoppers import Stopper
from hls.substrate.log_cap import LOG_KEEP_BYTES, errors_file, lost
from hls.substrate.nproc import uid_tasks
from hls.substrate.role_exec import (
    CONTESTANT_OOM_SCORE_ADJ,
    NODE_ROOT,
    check_role,
    placement,
    split_refusal,
    started_path,
    take_env_file,
    write_report,
)
from hls.substrate.secrets import parse_env_text
from hls.util import remove_tree, validation_errors

log = logging.getLogger(__name__)

#: The gate every role container starts with.
GATE = Path(__file__).with_name("gate.sh")
#: Where a role container sees its gate's directory, and the gate in it. A submission image
#: keeps its runscript under ``/.hls``, which this must not hide.
GATE_DIR = "/.hls-gate"
GATE_PATH = f"{GATE_DIR}/gate"
#: The FIFOs in the gate's directory: the role's variables, its stdout and its stderr.
ENV_FIFO, OUT_FIFO, ERR_FIFO = "env", "out", "err"
#: The directories a container role's workdir holds for its /tmp, /var/tmp and $HOME, by name.
WORKDIR_DIRS = ("tmp", "var_tmp", "home")
#: The line the gate prints on stderr just before a container role's command starts.
START_MARKER = "hls-container-started"
#: The host address a sealed server's port is published on: the loopback, where the drivers on the
#: host reach it, so no other machine that reaches the host can connect to a contestant's server.
SERVED_HOST_IP = "127.0.0.1"
#: GiB a sealed role's gateway holds when the launcher is given no other figure, the default of
#: ``site.docker.gateway_memory_gib``: what the interceptor and its index need beside the payload.
GATEWAY_MEMORY_GIB = 2
#: Where a container role's $HOME is.
CONTAINER_HOME = "/home/hls"
#: PATH of a harness role whose launcher was given none.
DEFAULT_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
#: The status of a role whose container the daemon would not create or start: the status
#: ``docker run`` gives for an error of the daemon's.
NOT_STARTED_STATUS = 125
#: The status of a role whose container ended and whose exit code could not be read: what the
#: daemon gives a container it killed, 128 and SIGKILL.
LOST_STATUS = 128 + signal.SIGKILL
#: This substrate's name in a harness role container's ``SUBSTRATE_VAR``, which
#: ``hls.substrate.factory.step`` reads.
SUBSTRATE_NAME = "docker"
#: Labels every role container carries: the job, the role and the step.
JOB_LABEL, ROLE_LABEL, STEP_LABEL = "hls.job", "hls.role", "hls.step"
#: The label of the helper container a role's user files are read through (``user_files``).
USER_FILES_LABEL = "hls.user-files"
#: The files of a role container's user database, each bound read-only over the image's own.
USER_FILES = ("passwd", "group")
#: The command of the helper container ``user_files`` reads the image's files through, which is
#: never started.
_HELPER_ARGV = ("/nonexistent/hls-user-files-helper",)
#: The memory limit of that helper, which never runs.
_HELPER_MEMORY = 64 * 2**20
#: The first part of every parent cgroup the launcher names (``cgroup_parent``).
PARENT_ROOT = "hls"
#: Seconds the gate has, once the container started, to open its variables' FIFO.
RELEASE_TIMEOUT_SEC = 60.0
#: Seconds between looks at the gate's FIFO while the launcher waits for the gate to open it.
_RELEASE_POLL_SEC = 0.01
#: Seconds each wait for a container's end lasts at most, so its time limit is kept.
_WAIT_SLICE_SEC = 60.0
#: Seconds between attempts to reach a daemon that did not answer.
_DAEMON_RETRY_SEC = 1.0
#: Seconds the handle waits, once a container has stopped, for the daemon's ``die`` event of
#: it, after which every event of it before its end has been read: a stream's events come in order.
DIE_EVENT_WAIT_SEC = 10.0
#: Seconds ``kill`` waits for the end of the container it killed to be read.
_KILL_WAIT_SEC = 10.0
#: The most role slots one launcher looks through for a free one.
MAX_SLOTS = 1024
#: Seconds the stopper gives a job's containers to end once killed.
STOP_WAIT_SEC = 30.0
#: Names a role's declared environment may not set: the launcher sets them.
_LAUNCHER_VARS = frozenset({"PATH", "HOME", "LANG", "TMPDIR", "CUDA_VISIBLE_DEVICES", SUBSTRATE_VAR, STEP_ID_VAR})
#: Prefixes a role's declared environment may not use: the NVIDIA Container Toolkit's, which name
#: a container's GPUs, and which the launcher sets.
_RESERVED_PREFIXES = ("NVIDIA_",)
#: Paths a container role may not mount over: the gate's and the ones the launcher binds itself.
_LAUNCHER_TARGETS = (GATE_DIR, "/tmp", "/var/tmp", CONTAINER_HOME, DRIVER_ROOT,
                     *(f"/etc/{name}" for name in USER_FILES))
#: Paths the container runtime provides; a bind there would replace them.
_RUNTIME_ROOTS = ("/proc", "/sys", "/dev")
_GIB = 2**30


def reserved_variable(name: str) -> bool:
    """Whether the launcher sets or forbids variable *name*, so that no role declares it and no
    secret grant may set it."""
    return name in _LAUNCHER_VARS or name.startswith(_RESERVED_PREFIXES)


def check_container(spec: ContainerSpec) -> None:
    """Raise ValueError for a container the launcher cannot run exactly as *spec* says: one that
    names a device node (``ContainerSpec.devices``), which this launcher does not give, and a
    mount target that is not an absolute path in normal form, that lies on a path the launcher or
    the runtime binds itself, or that two mounts share."""
    if spec.devices:
        raise ValueError(f"the Docker launcher gives a container no device node, and this one names "
                         f"{', '.join(spec.devices)}")
    targets = set()
    for m in spec.mounts:
        target = os.path.normpath(m.target)
        if target != m.target or not target.startswith("/") or target == "/":
            raise ValueError(f"mount target {m.target!r} is not an absolute path in normal form")
        if any(target == t or target.startswith(t + "/") for t in (*_LAUNCHER_TARGETS, *_RUNTIME_ROOTS)):
            raise ValueError(f"{m.target} is bound by the launcher or the runtime itself")
        if target in targets:
            raise ValueError(f"two mounts on {m.target}")
        targets.add(target)


class DockerRuntime(ContainerRuntime):
    """What a stage reads of the Docker launcher's containers: the gate's start marker, the
    directories of a workdir the launcher binds at /tmp, /var/tmp and $HOME, and no seal attempts."""

    start_marker = START_MARKER
    workdir_dirs = WORKDIR_DIRS
    attempt_line = None

    def __init__(self, *, gateway_memory_gib: int) -> None:
        #: A sealed role's gateway container holds this beside the role (``site.docker.gateway_memory_gib``).
        self.seal_memory_gib = gateway_memory_gib

    def check(self, spec: ContainerSpec) -> None:
        check_container(spec)


def code_dirs(python: Path, checkout: Path) -> tuple[Path, ...]:
    """The directories a harness role container needs read-only to run hls with *python*, each
    bound at its own path: the *checkout* hls runs from, the environment *python* belongs to,
    and the directory of the interpreter that environment was made from (the ``home`` of its
    ``pyvenv.cfg``), as it is named there, so the environment's links to it resolve. Raise
    ValueError when *python* is not in a virtual environment or lies outside them."""
    if not python.is_absolute() or not checkout.is_absolute():
        raise ValueError(f"the interpreter and the checkout are absolute paths: {python}, {checkout}")
    venv = python.parent.parent
    try:
        config = (venv / "pyvenv.cfg").read_text()
    except OSError as e:
        raise ValueError(f"{python} is not the interpreter of a virtual environment: {e}") from None
    homes = [line.partition("=")[2].strip() for line in config.splitlines() if line.partition("=")[0].strip() == "home"]
    if len(homes) != 1 or not homes[0].startswith("/"):
        raise ValueError(f"{venv / 'pyvenv.cfg'} names no absolute home of its interpreter")
    dirs: list[Path] = []
    for d in (checkout, venv, Path(homes[0]).parent):
        if not any(d.is_relative_to(kept) for kept in dirs):
            dirs = [kept for kept in dirs if not kept.is_relative_to(d)] + [d]
    return tuple(dirs)


def _within(path: Path, dirs: Collection[Path]) -> bool:
    return any(path.is_relative_to(d) for d in dirs)


def _env_text(data: bytes) -> tuple[str | None, str | None]:
    """The variables of an env file's content as the gate sources them, each value quoted for the
    shell, or None and why the content cannot be passed. Never quotes a value in the reason."""
    try:
        env = parse_env_text(data.decode())
    except UnicodeDecodeError:
        return None, "is malformed: not UTF-8 text"
    except ValueError as e:
        return None, f"is malformed: {e}"
    for name, value in env.items():
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            return None, f"is malformed: the value of {name} holds a control character"
    return "".join(f"{name}={shlex.quote(value)}\n" for name, value in sorted(env.items())), None


def _read_pipe(fd: int) -> bytes:
    """All a pipe with no writer holds, then close it."""
    data = b""
    try:
        while chunk := os.read(fd, 65536):
            data += chunk
    finally:
        os.close(fd)
    return data


def _empty_dir(path: Path) -> None:
    """Make *path* an empty directory of the host user's, mode 0700."""
    remove_tree(path)
    path.mkdir(mode=0o700, parents=True)


def cgroup_parent(driver: str, first_cpu: int, slot: int | None, *,
                  cgroup_root: Path = CGROUP_ROOT) -> tuple[str, Path]:
    """The parent cgroup of the allocation whose first CPU is *first_cpu*, or with *slot* of one
    of its role slots: its name as the daemon's cgroup *driver* takes it, and its directory on
    the host. Under systemd a slice's dashes name its parents, so ``hls-c4-2.slice`` lies in
    ``hls-c4.slice`` in ``hls.slice``. Raise ValueError for another driver."""
    parts = [PARENT_ROOT, f"c{first_cpu}"] + ([] if slot is None else [str(slot)])
    if driver == "systemd":
        names = ["-".join(parts[:i]) + ".slice" for i in range(1, len(parts) + 1)]
        return names[-1], cgroup_root.joinpath(*names)
    if driver == "cgroupfs":
        return "/" + "/".join(parts), cgroup_root.joinpath(*parts)
    raise ValueError(f"the Docker daemon's cgroup driver {driver!r} is neither systemd nor cgroupfs")


def oom_kills(directory: Path, *, made: bool = True) -> int | None:
    """The ``oom_kill`` count of cgroup *directory*'s ``memory.events``, which counts the kills in
    every cgroup below it; None when it cannot be read. With *made* False a directory that does
    not exist yet counts 0, as a parent the daemon has not made holds no kill."""
    try:
        return read_keyed(directory / "memory.events").get("oom_kill", 0)
    except FileNotFoundError:
        if not made and not directory.exists():
            return 0
        return None
    except (OSError, ValueError):
        return None


class _Events:
    """The daemon's events of one job's containers, read from one stream that is open before any
    of them is created, kept by container ID. A stream that breaks is replaced before the next
    container is created; the containers created under it are marked as having lost events."""

    _ACTIONS = ("kill", "stop", "die", "oom")

    def __init__(self, client: DockerClient, job_id: str) -> None:
        self._client = client
        self._job_id = job_id
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._events: dict[str, list[Event]] = {}
        self._stream: EventStream | None = None
        self._generation = 0
        self._broken: set[int] = set()
        self._closed = False

    def open(self) -> int:
        """Make sure a stream is open and return its generation."""
        with self._lock:
            if self._stream is not None and self._generation not in self._broken:
                return self._generation
            if self._stream is not None:
                self._stream.close()
            stream = self._client.watch_events(since_ns=time.time_ns(), types=("container",), actions=self._ACTIONS)
            self._generation += 1
            self._stream = stream
            generation = self._generation
        threading.Thread(target=self._pump, args=(stream, generation), name="docker-launcher-events",
                         daemon=True).start()
        return generation

    def _pump(self, stream: EventStream, generation: int) -> None:
        while True:
            try:
                event = stream.get(timeout=1.0)
            except DockerError as e:
                with self._lock:
                    if not self._closed and self._stream is stream:
                        log.warning("the Docker daemon's events stream broke: %s", e)
                    self._broken.add(generation)
                    self._changed.notify_all()
                return
            with self._lock:
                if self._stream is not stream:
                    return
            if event is not None and event.attributes.get(JOB_LABEL) == self._job_id:
                with self._lock:
                    self._events.setdefault(event.actor_id, []).append(event)
                    self._changed.notify_all()

    def through_end(self, container_id: str, generation: int, timeout: float) -> list[Event] | None:
        """Every event of the container up to its ``die``, once that came; None when it did not
        come within *timeout* seconds or the stream of *generation* broke first."""
        give_up = time.monotonic() + timeout
        with self._changed:
            while True:
                events = self._events.get(container_id, [])
                if any(e.action == "die" for e in events):
                    return list(events)
                left = give_up - time.monotonic()
                if generation in self._broken or left <= 0:
                    return None
                self._changed.wait(min(left, 1.0))

    def forget(self, container_id: str) -> None:
        with self._lock:
            self._events.pop(container_id, None)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            stream, self._stream = self._stream, None
        if stream is not None:
            stream.close()


class _Fifo:
    """A FIFO a log capper reads, with a write end the launcher holds until the container ended,
    so the capper does not see end of file before the gate opened its own write end."""

    def __init__(self, path: Path, out: Path, *, keep: int, python: str, errors: Path,
                 popen: Callable[..., subprocess.Popen]) -> None:
        os.mkfifo(path, 0o600)
        with open(out, "ab"):
            pass
        read_end = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        try:
            self.holder: int | None = os.open(path, os.O_WRONLY | os.O_CLOEXEC)
            os.set_blocking(read_end, True)
            with open(errors, "ab") as err:
                self.capper = popen([python, "-m", "hls.substrate.log_cap", "--keep", str(keep), str(out)],
                                    stdin=read_end, stdout=subprocess.DEVNULL, stderr=err, start_new_session=True)
        except BaseException:
            if getattr(self, "holder", None) is not None:
                os.close(self.holder)
            raise
        finally:
            os.close(read_end)

    def say(self, line: str) -> None:
        """Write *line* into the log, as the launcher's own word on the role."""
        if self.holder is not None:
            try:
                os.write(self.holder, line.encode() + b"\n")
            except OSError as e:
                log.warning("could not write into a role's log: %s", e)

    def close(self) -> None:
        if self.holder is not None:
            os.close(self.holder)
            self.holder = None


class DockerRoleHandle(RoleHandle):
    """A role running in a Docker container. A thread of its own creates and starts the
    container, releases its gate, waits for its end, reads how it ended and removes it."""

    def __init__(self, launcher: DockerLauncher, spec: RoleSpec, *, step: str, config: ContainerConfig,
                 gate_dir: Path, fifos: tuple[_Fifo, _Fifo], report_path: Path, gpus: tuple[str, ...],
                 shared: list[str] | None, parent: Path) -> None:
        self.spec = spec
        #: The step id the role's report names.
        self.step = step
        #: The container's configuration.
        self.config = config
        #: Where the role's grant report is, and its start marker beside it.
        self.report_path = report_path
        self.started_path = started_path(report_path)
        #: The container's ID once created.
        self.container_id: str | None = None
        #: The container's cgroup on the host once it started.
        self.cgroup_dir: Path | None = None
        #: True once ``terminate`` or ``kill`` acted on a running role.
        self.stopped_by_runner = False
        #: The directory of the parent cgroup the container is created under, which outlives it.
        self.parent = parent
        self._launcher = launcher
        self._gate_dir = gate_dir
        self._fifos = fifos
        self._gpus = gpus
        self._shared = shared
        self._lock = threading.Lock()
        self._stop_asked = False
        self._done = threading.Event()
        self._reported = threading.Event()
        self._report: RoleReport | None = None
        self._failure: str | None = None
        self._code: int | None = None
        self._end: tuple[bool, JobState | None] = (False, None)
        self._timed_out = False
        self._refused = False
        #: The parent's OOM kills before the container was created; None when unread.
        self._ooms_before: int | None = None
        #: Whether the container's cgroup was seen under its parent.
        self._under_parent = False
        self._generation = 0
        self._daemon: DaemonIdentity | None = None
        self._report_lock = threading.Lock()
        #: The gateway of a sealed role, once it seals the payload's namespace.
        self._gateway: Gateway | None = None
        self._thread = threading.Thread(target=self._run, name=f"docker-role-{spec.name}", daemon=True)

    # -- the role's thread ---------------------------------------------------------------

    def _start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        try:
            self._code = self._lifecycle()
        except Exception as e:  # noqa: BLE001 -- the role's end is recorded, never lost with the thread
            log.exception("role %s: the launcher failed", self.spec.name)
            self._failure = self._failure or f"the launcher failed: {type(e).__name__}: {e}"
            self._end = (True, None)
            self._code = self._code if self._code is not None else LOST_STATUS
            self._remove()
        finally:
            self._close_gateway(None)
            for fifo in self._fifos:
                fifo.close()
            remove_tree(self._gate_dir, ignore_errors=True)
            self._launcher._release(self.step)
            self._done.set()

    def _lifecycle(self) -> int:
        """Run the container through its life and return its status."""
        launcher, spec = self._launcher, self.spec
        env_text, env_problem = "", None
        if spec.container is not None:
            fd, problem = take_env_file(spec.container.env_file)
            if fd is not None:
                env_text, problem = _env_text(_read_pipe(fd))
            env_problem = problem
        if spec.nproc is not None:
            refused = launcher._ceiling_refusal(spec.nproc)
            if refused is not None:
                self._fifos[1].say(f"docker launcher: refusing to start: {refused}")
                self._failure = (f"role {spec.name} refused to start: its uid already held as many tasks as its "
                                 f"process ceiling of {spec.nproc} on the node (status {PROCESS_CEILING_EXIT})")
                return PROCESS_CEILING_EXIT
        try:
            launcher.user_files(self.config.image, self.config.labels, self._gate_dir)
        except RuntimeError as e:
            self._fatal(str(e))
            self._end = (True, None)
            return NOT_STARTED_STATUS
        try:
            self._generation = launcher._events.open()
            self._daemon = launcher._identity()
            if spec.container is not None and spec.container.spec.seal is not None:
                with self._lock:
                    if self._stop_asked:
                        raise GatewayFailed("the role was stopped before its gateway started")
                self._open_gateway()
            self._ooms_before = oom_kills(self.parent, made=False)
            container_id = launcher.client.create_container(self.config)
        except GatewayFailed as e:
            with self._lock:
                stopped = self._stop_asked
            if stopped:
                # The stop killed the gateway while it started; the payload was never made.
                self._failure = f"role {spec.name} was stopped before its container started"
                return NOT_STARTED_STATUS
            self._fatal(f"the seal's gateway did not seal the container's network: {e}")
            self._end = (True, None)
            return NOT_STARTED_STATUS
        except DockerError as e:
            self._fatal(f"the Docker daemon did not create its container: {e}")
            self._end = (True, None)
            return NOT_STARTED_STATUS
        with self._lock:
            self.container_id = container_id
            stop = self._stop_asked
        if stop:
            self._failure = f"role {spec.name} was stopped before its container started"
            self._remove()
            return NOT_STARTED_STATUS
        try:
            launcher.client.start_container(container_id)
            pid = launcher.client.inspect_container(container_id).state.pid
        except DockerError as e:
            self._fatal(f"the Docker daemon did not start its container: {e}")
            self._end = (True, None)
            self._remove()
            return NOT_STARTED_STATUS
        if spec.container is None:
            try:
                self.cgroup_dir = container_cgroup(pid, proc=launcher.proc, cgroup_root=launcher.cgroup_root)
            except RuntimeError as e:
                return self._ended_early(f"its cgroup could not be read: {e}")
            self._under_parent = self.cgroup_dir.parent == self.parent
        else:
            self.started_path.write_text(f"{self.step}\n")
            try:
                grant = measure_container(pid, requested_gpus=self._gpus, pids_limit=self.config.pids_limit,
                                          scratch=launcher.scratch, proc=launcher.proc,
                                          cgroup_root=launcher.cgroup_root, run=launcher.run,
                                          affinity=launcher.affinity)
            except RuntimeError as e:
                return self._ended_early(f"its grant could not be measured: {e}")
            self.cgroup_dir = grant.cgroup_dir
            self._under_parent = grant.cgroup_dir.parent == self.parent
            mismatches = check_role(spec.shape, grant.measured, shared=self._shared)
            if env_problem is not None:
                mismatches.append(env_problem)
            where = placement(grant.cpus, node_root=launcher.numa_root)
            refusal = split_refusal(where) if spec.one_numa_node else None
            if refusal is not None:
                mismatches.append(refusal)
            report = RoleReport(host=launcher.node, step=self.step, cgroup_dir=str(grant.cgroup_dir),
                                gpu_uuids=grant.measured.gpu_uuids,
                                cuda_visible_devices=self.config.env.get("CUDA_VISIBLE_DEVICES"),
                                granted=grant.measured.granted, swap_max_bytes=grant.measured.swap_max_bytes,
                                mismatches=mismatches, cpus=where.cpus, numa_nodes=where.numa_nodes,
                                numa_split=where.split)
            write_report(self.report_path, report)
            self._report = report
            self._reported.set()
            if mismatches:
                self._fifos[1].say(f"docker launcher: refusing to start, the container's grant differs: "
                                   f"{'; '.join(mismatches)}")
                self._refused = True
                self._kill_container()
                self._wait_end()
                self._read_end()
                self._remove()
                return GRANT_MISMATCH_EXIT
        if self._gateway is not None:
            mark_started(self._gateway)
        problem = self._release(env_text)
        if problem is not None:
            return self._ended_early(problem)
        self._wait_end()
        if self._gateway is not None:
            # The gateway records the payload's end as soon as it is seen, before the end's events
            # are waited for: the time of that record is the payload's end the stage charges.
            self._close_gateway(self._exit_code())
        code = self._read_end()
        self._remove()
        return code

    def _exit_code(self) -> int | None:
        """The container's exit code as the daemon gives it now; None when it cannot be read."""
        assert self.container_id is not None
        try:
            return self._launcher.client.inspect_container(self.container_id).state.exit_code
        except DockerError as e:
            log.warning("role %s: its container's exit code could not be read: %s", self.spec.name, e)
            return None

    def _open_gateway(self) -> None:
        """Open the gateway of the sealed role and put the payload in its network namespace."""
        launcher, role = self._launcher, self.spec.container
        assert role is not None and role.spec.seal is not None and role.seal_files is not None
        if role.ucounts:
            self._fifos[1].say(f"docker launcher: the seal's kernel-count limits {sorted(role.ucounts)} are not "
                               "applied: the payload makes no user namespace of its own")
        config = gateway_config(role.spec.seal, role.seal_files, image=launcher.step_image, python=launcher.python,
                                user=f"{launcher.uid}:{launcher.gid}",
                                code=tuple(launcher._own_bind(d, str(d), read_only=True)
                                           for d in launcher.code_dirs),
                                cpuset=self.config.cpuset, memory_bytes=launcher.gateway_memory_gib * _GIB,
                                labels=self.config.labels, upstream_proxy=launcher.upstream_proxy,
                                host_ip=SERVED_HOST_IP)
        self._gateway = open_gateway(launcher.client, role.spec.seal, role.seal_files, config)
        self.config = payload_config(self.config, self._gateway)

    def _close_gateway(self, code: int | None) -> None:
        """Close the gateway, if there is one still open: with the payload's own exit status, when
        the payload started and its end was read, so the gateway records the ending; else with none."""
        gateway, self._gateway = self._gateway, None
        if gateway is None:
            return
        own = code if code is not None and 0 <= code <= 255 and gateway.files.started.exists() else None
        try:
            close_gateway(self._launcher.client, gateway, exit_code=own)
        except (DockerError, OSError, ValueError) as e:
            log.warning("role %s: its gateway did not close cleanly: %s", self.spec.name, e)

    def _fatal(self, why: str) -> None:
        """Record why the role's container could not run, and say it in the role's stderr as a
        runtime's failure, a line starting ``FATAL``."""
        self._failure = f"role {self.spec.name}: {why}"
        self._fifos[1].say(f"FATAL: docker launcher: {why}")

    def _ended_early(self, why: str) -> int:
        """End a container whose role cannot go on, before its gate was released."""
        self._fatal(why)
        self._kill_container()
        self._wait_end()
        code = self._read_end()
        with self._lock:
            self._end = (False, None) if self.stopped_by_runner else (True, None)
        self._remove()
        return code

    def _release(self, env_text: str) -> str | None:
        """Write *env_text* into the gate's FIFO once the gate opened it; why that failed, or
        None. The gate waits for the FIFO's end before it runs anything of the role."""
        fifo = self._gate_dir / ENV_FIFO
        give_up = time.monotonic() + RELEASE_TIMEOUT_SEC
        while True:
            with self._lock:
                if self._stop_asked:
                    return "it was stopped before its gate was released"
            try:
                fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK | os.O_CLOEXEC)
                break
            except OSError as e:
                if e.errno != errno.ENXIO:
                    return f"its gate's FIFO cannot be opened: {e}"
            if time.monotonic() >= give_up:
                return f"its gate did not open its FIFO within {RELEASE_TIMEOUT_SEC:.0f}s"
            time.sleep(_RELEASE_POLL_SEC)
        try:
            os.set_blocking(fd, True)
            data = env_text.encode()
            while data:
                data = data[os.write(fd, data):]
        except OSError as e:
            return f"its variables could not be handed to its gate: {e}"
        finally:
            os.close(fd)
        return None

    def _wait_end(self) -> None:
        """Wait until the container has stopped, and enforce its time limit. A daemon that does
        not answer is asked again."""
        launcher = self._launcher
        limit = time.monotonic() + self.spec.shape.time_sec
        assert self.container_id is not None
        killed = False
        while True:
            left = limit - time.monotonic()
            if left <= 0 and not killed:
                with self._lock:
                    stopped = self.stopped_by_runner or self._refused
                if not stopped:
                    self._timed_out = True
                    self._fifos[1].say(f"docker launcher: role {self.spec.name} reached its time limit of "
                                       f"{self.spec.shape.time_sec}s")
                self._kill_container()
                killed = True
            try:
                result = launcher.client.wait_container(self.container_id,
                                                        timeout=min(_WAIT_SLICE_SEC, left) if left > 0 else
                                                        _WAIT_SLICE_SEC)
            except NotFound:
                return
            except DockerError as e:
                log.warning("role %s: waiting for its container failed: %s", self.spec.name, e)
                time.sleep(_DAEMON_RETRY_SEC)
                continue
            if result is not None:
                return

    def _read_end(self) -> int:
        """The container's status, once it ended, and whether the scheduler ended it."""
        launcher = self._launcher
        assert self.container_id is not None
        try:
            state = launcher.client.inspect_container(self.container_id).state
        except DockerError as e:
            log.warning("role %s: its container's state could not be read: %s", self.spec.name, e)
            state = None
        seen = launcher._events.through_end(self.container_id, self._generation, DIE_EVENT_WAIT_SEC)
        events = seen or []
        with self._lock:
            ours = self.stopped_by_runner or self._refused
        try:
            same_daemon = launcher._identity() == self._daemon
        except DockerError:
            same_daemon = False
        foreign_stop = not ours and any(e.action in ("kill", "stop") for e in events) and not self._timed_out
        after = oom_kills(self.parent) if self._under_parent else None
        counted = self._ooms_before is not None and after is not None
        oom = ((state is not None and state.oom_killed) or any(e.action == "oom" for e in events)
               or (counted and after > self._ooms_before))
        if state is None:
            died = [e.attributes.get("exitCode", "") for e in events if e.action == "die"]
            code = int(died[-1]) if died and died[-1].isdecimal() else LOST_STATUS
        else:
            code = state.exit_code
        if ours or code == 0:
            self._end = (False, None)
        elif self._timed_out:
            self._end = (True, JobState.TIMEOUT)
        elif state is None or foreign_stop or not same_daemon or seen is None:
            self._end = (True, JobState.CANCELLED if foreign_stop else None)
        elif oom:
            self._end = (False, JobState.OUT_OF_MEMORY)
        elif not counted:
            # Whether the kernel killed a process of it at its memory limit is not known.
            self._end = (True, None)
        else:
            self._end = (False, JobState.FAILED)
        return code

    def _kill_container(self) -> None:
        """Kill the container through the daemon, and every process of its cgroup when the
        daemon does not answer."""
        if self.container_id is None:
            return
        try:
            self._launcher.client.kill_container(self.container_id, signal.SIGKILL)
            return
        except NotFound:
            return
        except DockerError as e:
            if getattr(e, "status", None) == 409:
                return  # it has stopped
            log.warning("role %s: the Docker daemon did not kill its container: %s", self.spec.name, e)
        if self.cgroup_dir is not None:
            kill_all(self.cgroup_dir)

    def _remove(self) -> None:
        if self.container_id is None:
            return
        try:
            self._launcher.client.remove_container(self.container_id, force=True)
        except NotFound:
            pass
        except DockerError as e:
            log.warning("role %s: its container %s could not be removed: %s", self.spec.name, self.container_id, e)
        self._launcher._events.forget(self.container_id)

    # -- the handle --------------------------------------------------------------------------

    def _load(self) -> RoleReport | None:
        """A harness role's report once its file exists, its cgroup rewritten to the host's."""
        try:
            data = self.report_path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            report = RoleReport.model_validate_json(data)
        except ValidationError as e:
            raise RoleFailed(f"role {self.spec.name} wrote an invalid grant report: {validation_errors(e)}") from None
        if report.step != self.step:
            raise RoleFailed(f"role {self.spec.name} wrote a grant report for step {report.step}, not {self.step}")
        if report.cgroup_dir != str(STEP_CGROUP):
            raise RoleFailed(f"role {self.spec.name} reported the cgroup {report.cgroup_dir}, not its own "
                             f"{STEP_CGROUP}")
        outside = sorted(set(report.gpu_uuids) - set(self._gpus))
        if outside:
            raise RoleFailed(f"role {self.spec.name} reported GPUs {outside} that its container was not given")
        if self.cgroup_dir is None:
            raise RoleFailed(f"role {self.spec.name} reported before its container's cgroup was read")
        report = report.model_copy(update={"cgroup_dir": str(self.cgroup_dir)})
        write_report(self.report_path, report)
        return report

    def _harness_report(self) -> RoleReport | None:
        """A harness role's report, loaded once (``_load``); None before its file exists."""
        with self._report_lock:
            if self._report is None and self.cgroup_dir is not None:
                self._report = self._load()
            return self._report

    def _known_report(self) -> RoleReport | None:
        if self.spec.container is None:
            try:
                return self._harness_report()
            except RoleFailed:
                return None
        return self._report

    def started(self) -> bool:
        return self._known_report() is not None or self.started_path.exists()

    def report(self, timeout_sec: float) -> RoleReport:
        if self._report is not None:
            return self._report
        deadline = time.monotonic() + timeout_sec
        while True:
            done = self._done.is_set()
            if self.spec.container is None:
                report = self._harness_report()
                if report is not None:
                    return report
            elif self._reported.is_set():
                assert self._report is not None
                return self._report
            if done:
                if self._code == PROCESS_CEILING_EXIT and self._failure is not None:
                    raise RoleFailed(self._failure)
                why = f": {self._failure}" if self._failure else ""
                raise RoleFailed(f"role {self.spec.name} ended with status {self._code} before reporting its "
                                 f"grant{why}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"role {self.spec.name} did not report its grant in {timeout_sec}s")
            self._done.wait(min(0.2, max(0.0, deadline - time.monotonic())))

    def poll(self) -> int | None:
        return self._code if self._done.is_set() else None

    def wait(self, timeout_sec: float | None = None) -> int | None:
        if not self._done.wait(timeout_sec):
            return None
        return self._code

    def _signal(self, sig: signal.Signals) -> None:
        """Mark a running role stopped by the runner and send *sig* to its container; one whose
        gate has not been released is killed outright, since nothing of the role runs yet."""
        with self._lock:
            if self._done.is_set():
                return
            self.stopped_by_runner = True
            self._stop_asked = True
            container_id = self.container_id
        if container_id is None:
            # A gateway that is starting is stopped, so its start ends at once and the payload is
            # never made; nothing else of the role exists yet.
            self._kill_starting_gateway()
            return
        if sig != signal.SIGKILL:
            try:
                self._launcher.client.kill_container(container_id, sig)
                return
            except NotFound:
                return
            except DockerError as e:
                if getattr(e, "status", None) == 409:
                    return
                log.warning("role %s: the Docker daemon did not signal its container: %s", self.spec.name, e)
                return
        self._kill_container()

    def _kill_starting_gateway(self) -> None:
        """Kill the gateway of this role's step while it starts, found by its labels."""
        role = self.spec.container
        if role is None or role.spec.seal is None:
            return
        client = self._launcher.client
        try:
            rows = client.list_containers(labels={JOB_LABEL: self._launcher.job_id, STEP_LABEL: self.step,
                                                  GATEWAY_LABEL: "gateway"})
        except DockerError as e:
            log.warning("role %s: its starting gateway could not be found: %s", self.spec.name, e)
            return
        for row in rows:
            try:
                client.kill_container(row.id, signal.SIGKILL)
            except NotFound:
                continue
            except DockerError as e:
                if getattr(e, "status", None) != 409:
                    log.warning("role %s: its starting gateway could not be killed: %s", self.spec.name, e)

    def kill(self) -> None:
        self._signal(signal.SIGKILL)
        self._done.wait(_KILL_WAIT_SEC)

    def ended_by_scheduler(self) -> bool:
        """Whether the scheduler ended the container, or its end could not be confirmed, after
        ``wait`` returned a nonzero status the runner did not cause (see the module docstring).
        A role the runner stopped, and a clean exit, are never the scheduler's."""
        if not self._done.is_set():
            raise RuntimeError(f"role {self.spec.name} has not ended")
        return self._end[0]

    def end_state(self) -> JobState | None:
        """OUT_OF_MEMORY or FAILED for a container that ended on its own, TIMEOUT for one that
        reached its time limit, CANCELLED for one a kill the launcher did not send ended; None
        when nothing could be read or the role ended cleanly or by the runner's stop."""
        if not self._done.is_set():
            raise RuntimeError(f"role {self.spec.name} has not ended")
        return self._end[1]

    def close_logs(self, timeout_sec: float) -> None:
        give_up = time.monotonic() + timeout_sec
        for fifo in self._fifos:
            try:
                fifo.capper.wait(max(0.0, give_up - time.monotonic()))
            except subprocess.TimeoutExpired:
                log.warning("the log of role %s was not complete within %.0fs", self.spec.name, timeout_sec)

    def logs_lost(self) -> str | None:
        return lost([fifo.capper for fifo in self._fifos])


class DockerLauncher(Launcher):
    """Starts roles as containers of the allocation this process runs in, through the Docker
    daemon *client* speaks to. The allocation is *job_id*'s reservation: the GPUs
    *allocation_gpu_uuids*, the CPUs *cpus* and *memory_gib*, which the memory of the roles that
    run at once may not pass.

    A harness role runs in *step_image* (an image ID or name the daemon holds), with *code_dirs*
    bound read-only (``code_dirs``) and *data_dirs* and the job's node-local *scratch* bound
    read-write, each at its own path, and *python* running the step wrapper. A container whose
    role has no process ceiling is held to *pids_limit* tasks and to *nproc_limit*, the soft and
    hard ``RLIMIT_NPROC`` a Slurm step would inherit from the runner, this process's own unless
    given (None: unlimited, as a container's is without one). *labels* go on every container
    beside the job's, the role's and the step's. *driver* is the host's NVIDIA driver as the
    scheduler recorded it, which an allocation with GPUs needs, and whose version must be the
    running driver's; the GPUs' device nodes are under *dev*."""

    def __init__(self, *, client: DockerClient, job_id: str, python: str, scratch: Path, step_image: str,
                 allocation_gpu_uuids: Sequence[str], cpus: Sequence[int], memory_gib: int,
                 code_dirs: Sequence[Path],
                 data_dirs: Sequence[Path], pids_limit: int, labels: Mapping[str, str] | None = None,
                 nproc_limit: tuple[int, int] | None | Literal["own"] = "own",
                 gateway_memory_gib: int = GATEWAY_MEMORY_GIB, upstream_proxy: str | None = None,
                 batch_env: Mapping[str, str] | None = None, uid: int | None = None, gid: int | None = None,
                 node: str | None = None, log_keep_bytes: int = LOG_KEEP_BYTES, proc: Path = Path("/proc"),
                 cgroup_root: Path = CGROUP_ROOT, numa_root: Path = NODE_ROOT, node_root: Path = Path("/"),
                 run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                 affinity: Callable[[int], set[int]] = os.sched_getaffinity,
                 identity: Callable[[], DaemonIdentity] | None = None,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen, driver: NvidiaDriver | None = None,
                 dev: Path = Path("/dev")) -> None:
        if not cpus:
            raise ValueError("an allocation holds at least one CPU")
        if pids_limit < 1:
            raise ValueError(f"a pids limit of {pids_limit} lets a container start nothing")
        self.client = client
        self.job_id = job_id
        self.python = python
        self.scratch = scratch
        self.step_image = step_image
        self.allocation_gpu_uuids = tuple(allocation_gpu_uuids)
        #: The host's NVIDIA driver, whose files every GPU role is given.
        self.driver = driver
        self.dev = dev
        #: The minor number of each of the allocation's GPUs' device node, by UUID.
        self.minors = self._minors(run, proc) if self.allocation_gpu_uuids else {}
        self.cpus = tuple(sorted(set(cpus)))
        self.memory_gib = memory_gib
        self.code_dirs = tuple(code_dirs)
        self.data_dirs = tuple(dict.fromkeys([scratch, *data_dirs]))
        if not _within(Path(python), self.code_dirs):
            raise ValueError(f"the interpreter {python} lies outside the code directories {list(self.code_dirs)}")
        self.pids_limit = pids_limit
        if nproc_limit == "own":
            soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
            nproc_limit = None if resource.RLIM_INFINITY in (soft, hard) else (soft, hard)
        #: The soft and hard ``RLIMIT_NPROC`` of a container whose role has no ceiling; None for none.
        self.nproc_limit = nproc_limit
        #: GiB each sealed role's gateway holds, beside the role's own memory.
        self.gateway_memory_gib = gateway_memory_gib
        #: The HTTP proxy a gateway's interceptor reaches the internet through; None dials directly.
        self.upstream_proxy = upstream_proxy
        self.labels = dict(labels or {})
        if {JOB_LABEL, ROLE_LABEL, STEP_LABEL} & set(self.labels):
            raise ValueError(f"the labels {JOB_LABEL}, {ROLE_LABEL} and {STEP_LABEL} are the launcher's")
        self.batch_env = dict(os.environ if batch_env is None else batch_env)
        self.uid = os.getuid() if uid is None else uid
        self.gid = os.getgid() if gid is None else gid
        if self.uid == 0:
            raise ValueError("role containers never run as root")
        self.node = node if node is not None else os.uname().nodename
        self.log_keep_bytes = log_keep_bytes
        self.proc = proc
        self.cgroup_root = cgroup_root
        self.numa_root = numa_root
        self.node_root = node_root
        self.run = run
        self.affinity = affinity
        self._identity = identity if identity is not None else client.identity
        self._popen = popen
        self._events = _Events(client, job_id)
        self._lock = threading.Lock()
        self._next_step = 0
        self._held_gpus: dict[str, str] = {}
        self._held_cpus: dict[int, str] = {}
        #: The memory each running role holds, in GiB, by its step.
        self._held_memory: dict[str, int] = {}
        #: The daemon's cgroup driver, which names the parent cgroups.
        self.cgroup_driver = client.info().cgroup_driver
        #: The allocation's parent cgroup, and its OOM kills when the launcher was made.
        _, self.allocation_parent = cgroup_parent(self.cgroup_driver, self.cpus[0], None, cgroup_root=cgroup_root)
        self._allocation_ooms = oom_kills(self.allocation_parent, made=False)
        #: The role slots held, each by the step that holds it.
        self._slots: dict[int, str] = {}
        #: The parent cgroups of the slots passed over because a process was in them.
        self.busy_slots: list[Path] = []

    def _minors(self, run: Callable[..., subprocess.CompletedProcess], proc: Path) -> dict[str, int]:
        """The device node minor number of each of the allocation's GPUs, checking that the host
        runs the driver the scheduler recorded and has every node a GPU role is given. Raise
        RuntimeError, saying why in one line, when it does not."""
        if self.driver is None:
            raise RuntimeError("the allocation holds GPUs and the scheduler recorded no NVIDIA driver of the host")
        try:
            running = driver_version(proc)
            by_uuid = {gpu.uuid: minor for minor, gpu in gpus_by_minor(run, proc=proc).items()}
        except (DriverUnusable, GpusUnknown) as e:
            raise RuntimeError(str(e)) from None
        if running != self.driver.version:
            raise RuntimeError(f"the host runs NVIDIA driver {running}, and the local scheduler recorded driver "
                               f"{self.driver.version}; restart the scheduler so it records the running one")
        missing = [u for u in self.allocation_gpu_uuids if u not in by_uuid]
        if missing:
            raise RuntimeError(f"nvidia-smi does not list the allocation's GPUs {missing}")
        minors = {u: by_uuid[u] for u in self.allocation_gpu_uuids}
        try:
            driver_devices(sorted(minors.values()), dev=self.dev)
        except DriverUnusable as e:
            raise RuntimeError(str(e)) from None
        return minors

    def close(self) -> None:
        """Close the events stream. Roles still running keep running."""
        self._events.close()

    def allocation_oom_kills(self) -> int | None:
        """The OOM kills in the allocation's containers since the launcher was made, counted in
        the allocation's parent cgroup; None when that count cannot be read."""
        now = oom_kills(self.allocation_parent)
        if now is None or self._allocation_ooms is None:
            return None
        return now - self._allocation_ooms

    def _slot(self, step: str) -> tuple[str, Path]:
        """Take for *step* the lowest role slot that no running role of this launcher holds and no
        process is in, and return its parent cgroup's name and directory. Call with the lock held.
        A slot with a process in it is passed over and recorded, never joined."""
        for slot in range(MAX_SLOTS):
            if slot in self._slots:
                continue
            name, directory = cgroup_parent(self.cgroup_driver, self.cpus[0], slot, cgroup_root=self.cgroup_root)
            if directory.exists() and pids(directory):
                log.warning("the parent cgroup %s holds processes no running role of this launcher started; it is "
                            "passed over", directory)
                self.busy_slots.append(directory)
                continue
            self._slots[slot] = step
            return name, directory
        raise ValueError(f"no free role slot among the first {MAX_SLOTS}")

    def user_files(self, image: str, labels: Mapping[str, str], directory: Path) -> None:
        """Write into *directory* the user database a container of *image* runs with
        (``USER_FILES``), as Apptainer gives a container its own: the image's ``/etc/passwd`` with the
        host user's entry appended (its name, uid, gid, home and shell, no password and no comment)
        and the image's ``/etc/group`` with the host group's (its name and gid, no member). The
        image's files are read through a helper container of it, created with *labels* and
        ``USER_FILES_LABEL``, never started, and removed whatever happens. Raise RuntimeError,
        saying why in one line, when a file cannot be read: the role's start is refused, never
        given a database without the image's users."""
        texts: dict[str, str] = {}
        helper = None
        try:
            helper = self.client.create_container(ContainerConfig(
                image=image, argv=_HELPER_ARGV, user=f"{self.uid}:{self.gid}",
                labels={**labels, USER_FILES_LABEL: "1"}, network_mode="none", cpuset=(self.cpus[0],),
                memory_bytes=_HELPER_MEMORY, pids_limit=1))
            for name in USER_FILES:
                archive = directory / f"{name}.tar"
                archive.unlink(missing_ok=True)
                self.client.container_archive(helper, f"/etc/{name}", archive, timeout=CALL_TIMEOUT)
                try:
                    texts[name] = _archived_file(archive, name)
                finally:
                    archive.unlink(missing_ok=True)
        except (DockerError, OSError, ValueError) as e:
            raise RuntimeError(f"the image's /etc/passwd and /etc/group cannot be read: {e}") from None
        finally:
            if helper is not None:
                try:
                    self.client.remove_container(helper, force=True)
                except DockerError as e:
                    log.warning("the helper container %s that read the image's user files was not removed: %s",
                                helper, e)
        user, group = pwd.getpwuid(self.uid), grp.getgrgid(self.gid)
        own = {"passwd": f"{user.pw_name}:x:{user.pw_uid}:{user.pw_gid}::{user.pw_dir}:{user.pw_shell}\n",
               "group": f"{group.gr_name}:x:{group.gr_gid}:\n"}
        for name in USER_FILES:
            text = texts[name]
            path = directory / name
            path.write_text(text + ("\n" if text and not text.endswith("\n") else "") + own[name])
            path.chmod(0o644)

    # -- what a launch takes -----------------------------------------------------------------

    def _ceiling_refusal(self, ceiling: int) -> str | None:
        """Why a role with process ceiling *ceiling* may not start: its uid already holds as
        many tasks on the host; None when it holds fewer or they cannot be counted."""
        try:
            held = uid_tasks(self.uid, proc=self.proc)
        except Exception as e:  # noqa: BLE001 -- the check only spares a start that could fork nothing
            log.warning("the uid's tasks could not be counted (%s: %s); starting the role without that check",
                        type(e).__name__, e)
            return None
        if held < ceiling:
            return None
        return (f"uid {self.uid} holds {held} tasks on this node, as many as the step's process ceiling of "
                f"{ceiling} or more, so the step could fork nothing")

    def _take(self, step: str, spec: RoleSpec, share_gpus_of: RoleHandle | None
              ) -> tuple[tuple[str, ...], tuple[int, ...], list[str] | None, tuple[str, Path]]:
        """The GPUs and CPUs the role's container gets, held for *step* when they are its own, and
        the parent cgroup of the slot it holds."""
        s = spec.shape
        shared = None
        if share_gpus_of is not None:
            if not isinstance(share_gpus_of, DockerRoleHandle) or share_gpus_of._launcher is not self:
                raise ValueError(f"role {spec.name} can share the GPUs only of a role this launcher started")
            if s.gpus == 0:
                raise ValueError(f"role {spec.name} declares no GPUs, so it cannot share those of "
                                 f"{share_gpus_of.spec.name}")
            target = share_gpus_of.report(0.0)
            if target.mismatches:
                raise ValueError(f"role {spec.name} cannot share the GPUs of {share_gpus_of.spec.name}: "
                                 f"its grant differs: {target.mismatches}")
            shared = list(target.gpu_uuids)
            if len(shared) != s.gpus:
                raise ValueError(f"role {spec.name} declares {s.gpus} GPUs but would share {len(shared)} with "
                                 f"{share_gpus_of.spec.name}")
        if s.cpus > len(self.cpus):
            raise ValueError(f"role {spec.name} asks for {s.cpus} CPUs; the allocation has {len(self.cpus)}")
        sealed = spec.container is not None and spec.container.spec.seal is not None
        memory = s.memory_gib + (self.gateway_memory_gib if sealed else 0)
        with self._lock:
            held = sum(self._held_memory.values())
            if held + memory > self.memory_gib:
                with_gateway = f" with its gateway's {self.gateway_memory_gib}" if sealed else ""
                raise ValueError(f"role {spec.name} asks for {s.memory_gib} GiB{with_gateway}; the allocation's "
                                 f"running roles hold {held} of its {self.memory_gib} GiB")
            if shared is not None:
                gpus = tuple(shared)
            else:
                free_gpus = [u for u in self.allocation_gpu_uuids if u not in self._held_gpus]
                if len(free_gpus) < s.gpus:
                    raise ValueError(f"role {spec.name} asks for {s.gpus} GPUs; {len(free_gpus)} of the allocation's "
                                     "are free")
                gpus = tuple(free_gpus[:s.gpus])
            if spec.overlap or shared is not None:
                cpus = self.cpus[:s.cpus]
            else:
                free_cpus = [c for c in self.cpus if c not in self._held_cpus]
                if len(free_cpus) < s.cpus:
                    raise ValueError(f"role {spec.name} asks for {s.cpus} CPUs; {len(free_cpus)} of the allocation's "
                                     "are free")
                cpus = tuple(free_cpus[:s.cpus])
                self._held_cpus.update(dict.fromkeys(cpus, step))
            if shared is None:
                self._held_gpus.update(dict.fromkeys(gpus, step))
            try:
                parent = self._slot(step)
            except ValueError:
                self._release_locked(step)
                raise
            self._held_memory[step] = memory
        return gpus, cpus, shared, parent

    def _release(self, step: str) -> None:
        with self._lock:
            self._release_locked(step)

    def _release_locked(self, step: str) -> None:
        for held in (self._held_gpus, self._held_cpus, self._slots):
            for key in [k for k, owner in held.items() if owner == step]:
                del held[key]
        self._held_memory.pop(step, None)

    def _common(self, spec: RoleSpec, *, step: str, gpus: tuple[str, ...], cpus: tuple[int, ...],
                parent: str) -> dict:
        """The settings every role container has."""
        memory = spec.shape.memory_gib * _GIB
        if spec.nproc is not None:
            ulimits: tuple[Ulimit, ...] = (Ulimit("nproc", spec.nproc, spec.nproc),)
        else:
            ulimits = (Ulimit("nproc", *self.nproc_limit),) if self.nproc_limit is not None else ()
        env = {"NVIDIA_VISIBLE_DEVICES": ",".join(gpus) if gpus else "void"}
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        return dict(user=f"{self.uid}:{self.gid}",
                    labels={**self.labels, JOB_LABEL: self.job_id, ROLE_LABEL: spec.name, STEP_LABEL: step},
                    network_mode="host", cpuset=cpus, memory_bytes=memory, swap_bytes=0,
                    pids_limit=spec.nproc if spec.nproc is not None else self.pids_limit, shm_size_bytes=memory,
                    ulimits=ulimits, devices=self._devices(gpus), cgroup_parent=parent, init=True,
                    read_only_root=True, env=env)

    def _devices(self, gpus: tuple[str, ...]) -> tuple[Device, ...]:
        """The device nodes of a container given *gpus*: none without GPUs."""
        if not gpus:
            return ()
        try:
            nodes = driver_devices([self.minors[u] for u in gpus], dev=self.dev)
        except DriverUnusable as e:
            raise ValueError(str(e)) from None
        return tuple(Device(source, target) for source, target in nodes)

    def _driver(self, gpus: tuple[str, ...]) -> tuple[list[BindMount], tuple[str, ...]]:
        """The driver's files bound into a container given *gpus*, and the gate's option that puts
        them on the role's paths; none without GPUs."""
        if not gpus:
            return [], ()
        assert self.driver is not None
        return ([BindMount(source, target, read_only=True) for source, target in self.driver.binds()],
                ("--nvidia", self.driver.version))

    def _harness_config(self, spec: RoleSpec, *, step: str, role_dir: Path, gate_dir: Path, report: Path,
                        gpus: tuple[str, ...], cpus: tuple[int, ...], shared: list[str] | None,
                        parent: str) -> ContainerConfig:
        bad = sorted(k for k in spec.env if reserved_variable(k))
        if bad:
            raise ValueError(f"role {spec.name} may not set {bad}")
        if not _within(spec.log_dir, self.data_dirs):
            raise ValueError(f"role {spec.name}: its log directory {spec.log_dir} is not under a directory bound into "
                             f"harness containers ({[str(d) for d in self.data_dirs]})")
        if spec.env_file is not None and not _within(spec.env_file, self.data_dirs):
            raise ValueError(f"role {spec.name}: its env file is not under a directory bound into harness containers")
        s = spec.shape
        wrapper = [self.python, "-m", "hls.substrate.role_exec", "--report", str(report), "--gpus", str(s.gpus),
                   "--cpus", str(s.cpus), "--memory-gib", str(s.memory_gib), "--disk-gib", str(s.disk_gib),
                   "--scratch", str(self.scratch)]
        if s.gpu_kind is not None:
            wrapper += ["--gpu-kind", s.gpu_kind]
        if shared is not None:
            wrapper += ["--share-gpus", ",".join(shared)]
        if spec.env_file is not None:
            wrapper += ["--env-file", str(spec.env_file)]
            if spec.env_fd:
                wrapper.append("--env-fd")
        if spec.owner == "contestant":
            wrapper += ["--oom-score-adj", str(CONTESTANT_OOM_SCORE_ADJ)]
        if spec.nproc is not None:
            wrapper += ["--nproc", str(spec.nproc)]
        if spec.one_numa_node:
            wrapper.append("--one-numa-node")
        dirs = {name: role_dir / name for name in ("tmp", "var_tmp", "home")}
        for d in dirs.values():
            _empty_dir(d)
        common = self._common(spec, step=step, gpus=gpus, cpus=cpus, parent=parent)
        driver_mounts, nvidia = self._driver(gpus)
        env = {"PATH": self.batch_env.get("PATH", DEFAULT_PATH), "HOME": str(dirs["home"]), "LANG": "C.UTF-8",
               "TMPDIR": str(dirs["tmp"]), SUBSTRATE_VAR: SUBSTRATE_NAME, STEP_ID_VAR: step, **common.pop("env"),
               **spec.env}
        mounts = [*(self._own_bind(d, str(d), read_only=True) for d in self.code_dirs),
                  *(self._own_bind(d, str(d), read_only=False) for d in self.data_dirs),
                  self._own_bind(gate_dir, GATE_DIR, read_only=True),
                  self._own_bind(dirs["tmp"], "/tmp", read_only=False),
                  self._own_bind(dirs["var_tmp"], "/var/tmp", read_only=False), *driver_mounts,
                  *self._user_binds(gate_dir)]
        groups: tuple[int, ...] = ()
        if spec.daemon:
            socket_mount, groups = self._daemon_socket(spec)
            mounts.append(socket_mount)
        return ContainerConfig(image=self.step_image,
                               argv=("/bin/sh", GATE_PATH, *nvidia, "--", *wrapper, "--", *spec.argv),
                               group_add=groups,
                               mounts=tuple(mounts), working_dir=str(dirs["home"]), env=env, **common)

    def _container_config(self, spec: RoleSpec, *, step: str, gate_dir: Path, gpus: tuple[str, ...],
                          cpus: tuple[int, ...], parent: str) -> ContainerConfig:
        role = spec.container
        assert role is not None
        c = role.spec
        if spec.start_marker is None:
            raise ValueError(f"role {spec.name}: a container role prints a start marker, and none was given")
        if spec.daemon:
            raise ValueError(f"role {spec.name}: only a harness role that runs a command is given the daemon's socket")
        if spec.shape.gpus and not c.gpus:
            raise ValueError(f"role {spec.name} holds {spec.shape.gpus} GPUs that its container would not see")
        try:
            check_container(c)
        except ValueError as e:
            raise ValueError(f"role {spec.name}: {e}") from None
        mounts = [self._own_bind(gate_dir, GATE_DIR, read_only=True), *self._user_binds(gate_dir)]
        for name, target in zip(WORKDIR_DIRS, ("/tmp", "/var/tmp", CONTAINER_HOME)):
            (c.workdir / name).mkdir(mode=0o700, parents=True, exist_ok=True)
            mounts.append(self._spec_bind(spec, c.workdir / name, target, read_only=False))
        mounts += [self._spec_bind(spec, m.source, m.target, read_only=m.read_only) for m in c.mounts]
        given = gpus if c.gpus else ()
        common = self._common(spec, step=step, gpus=given, cpus=cpus, parent=parent)
        driver_mounts, nvidia = self._driver(given)
        mounts += driver_mounts
        if c.seal is not None:
            # The payload joins its gateway's network namespace once the gateway is open.
            common["network_mode"] = "none"
        if c.confinement == "spawner":
            common.update(seccomp_profile=spawner_profile(), unconfine_apparmor=True, unmask_system_paths=True)
        env = {"HOME": CONTAINER_HOME, **common.pop("env")}
        score = ("--oom-score-adj", str(CONTESTANT_OOM_SCORE_ADJ)) if spec.owner == "contestant" else ()
        return ContainerConfig(image=c.image.ref, argv=("/bin/sh", GATE_PATH, *nvidia, *score, "--marker",
                                                        spec.start_marker, "--", *c.command),
                               mounts=tuple(mounts), working_dir=c.cwd or CONTAINER_HOME, env=env, **common)

    def _user_binds(self, gate_dir: Path) -> list[BindMount]:
        """The role's user files (``user_files``), which its handle writes into *gate_dir* before it
        creates the container, each bound read-only over the image's own."""
        return [self._own_bind(gate_dir / name, f"/etc/{name}", read_only=True) for name in USER_FILES]

    def _exposes_socket(self, resolved: Path) -> bool:
        """Whether a bind of the real path *resolved* would show the daemon's socket: the socket
        itself or a directory that holds it."""
        socket_path = Path(os.path.realpath(self.client.socket))
        return socket_path == resolved or socket_path.is_relative_to(resolved)

    def _own_bind(self, source: Path, target: str, *, read_only: bool) -> BindMount:
        """A bind of a directory the launcher chose, a code, data, scratch or gate directory: its
        real path, which the daemon resolves as root, at *target*, so a link on the way changes
        nothing; never one that shows the daemon's socket."""
        resolved = Path(os.path.realpath(source))
        if self._exposes_socket(resolved):
            raise ValueError(f"{source} would show the Docker daemon's socket to a role container")
        return BindMount(resolved, target, read_only=read_only)

    def _spec_bind(self, spec: RoleSpec, source: Path, target: str, *, read_only: bool) -> BindMount:
        """A bind a stage named: refused unless *source* is its own real path, since the daemon
        resolves a link in it as root, and unless it shows no daemon socket."""
        resolved = Path(os.path.realpath(source))
        if resolved != source:
            raise ValueError(f"role {spec.name}: bind source {source} is reached through a link or is not in "
                             f"normal form (it resolves to {resolved}); a bind names a real path, so a Docker host's "
                             "site.node_scratch and site.store_root are real paths")
        if self._exposes_socket(resolved):
            raise ValueError(f"role {spec.name}: bind source {source} would show the Docker daemon's socket")
        return BindMount(source, target, read_only=read_only)

    def _daemon_socket(self, spec: RoleSpec) -> tuple[BindMount, tuple[int, ...]]:
        """The daemon's socket bound read-write at its own path, for a harness role that speaks to
        the daemon (``RoleSpec.daemon``), and the supplementary groups the role needs to open it:
        the socket's group when only that group's bits let the role's uid and gid write it. Raise
        ValueError when nothing would: no write bit for the role, or a socket of the root group,
        which no container is given."""
        socket_path = Path(self.client.socket)
        try:
            st = socket_path.stat()
        except OSError as e:
            raise ValueError(f"role {spec.name}: the daemon's socket {socket_path} cannot be read: {e}") from None
        if not stat.S_ISSOCK(st.st_mode):
            raise ValueError(f"role {spec.name}: {socket_path} is not a socket")
        # The real path, which the daemon would resolve as root, at the path the role is told.
        mount = BindMount(Path(os.path.realpath(socket_path)), str(socket_path), read_only=False)
        if st.st_uid == self.uid:
            # The kernel judges the owner by the owner bits alone.
            if st.st_mode & stat.S_IWUSR:
                return mount, ()
        elif st.st_gid == self.gid and st.st_mode & stat.S_IWGRP:
            return mount, ()
        elif st.st_mode & stat.S_IWGRP and st.st_gid != 0:
            return mount, (st.st_gid,)
        elif st.st_mode & stat.S_IWOTH:
            return mount, ()
        raise ValueError(f"role {spec.name}: uid {self.uid} and gid {self.gid} cannot open the daemon's socket "
                         f"{socket_path} (owner {st.st_uid}, group {st.st_gid}, mode {stat.S_IMODE(st.st_mode):o}), "
                         "even with the socket's group, and a container is never given the root group")

    # -- the launcher ------------------------------------------------------------------------

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        with self._lock:
            step = f"{self.job_id}.{self._next_step}"
            self._next_step += 1
        gpus, cpus, shared, (parent, parent_dir) = self._take(step, spec, share_gpus_of)
        fifos: list[_Fifo] = []
        try:
            spec.log_dir.mkdir(parents=True, exist_ok=True)
            report = spec.log_dir / f"{spec.name}.report.json"
            report.unlink(missing_ok=True)
            started_path(report).unlink(missing_ok=True)
            role_dir = self.scratch / "roles" / spec.name
            role_dir.mkdir(parents=True, exist_ok=True)
            gate_dir = role_dir / "gate"
            _empty_dir(gate_dir)
            gate = gate_dir / "gate"
            gate.write_bytes(GATE.read_bytes())
            gate.chmod(stat.S_IRUSR)
            if spec.container is None:
                config = self._harness_config(spec, step=step, role_dir=role_dir, gate_dir=gate_dir, report=report,
                                              gpus=gpus, cpus=cpus, shared=shared, parent=parent)
            else:
                config = self._container_config(spec, step=step, gate_dir=gate_dir, gpus=gpus, cpus=cpus,
                                                parent=parent)
            os.mkfifo(gate_dir / ENV_FIFO, 0o600)
            for name, suffix in ((OUT_FIFO, ".out"), (ERR_FIFO, ".err")):
                fifos.append(_Fifo(gate_dir / name, spec.log_dir / f"{spec.name}{suffix}", keep=self.log_keep_bytes,
                                   python=self.python, errors=errors_file(role_dir, suffix), popen=self._popen))
        except BaseException:
            for fifo in fifos:
                fifo.close()
            self._release(step)
            raise
        given = gpus if spec.container is None or spec.container.spec.gpus else ()
        handle = DockerRoleHandle(self, spec, step=step, config=config, gate_dir=gate_dir,
                                  fifos=(fifos[0], fifos[1]), report_path=report, gpus=given,
                                  shared=shared, parent=parent_dir)
        handle._start()
        return handle

    def terminate(self, handles: Sequence[RoleHandle]) -> None:
        """Send SIGTERM to every one of *handles* that runs; a role whose gate has not been
        released is killed, since nothing of it runs yet, and one that ended is left alone."""
        foreign = [h.spec.name for h in handles if not isinstance(h, DockerRoleHandle)]
        if foreign:
            raise TypeError(f"roles {foreign} were not started by a Docker launcher")
        for h in handles:
            assert isinstance(h, DockerRoleHandle)
            h._signal(signal.SIGTERM if h.started() else signal.SIGKILL)


def _archived_file(archive: Path, name: str) -> str:
    """The text of the one regular file *name* in *archive*, the tar the daemon makes of a path
    in a container; raise ValueError when it holds no such file or not UTF-8 text."""
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name.strip("/").rsplit("/", 1)[-1] == name]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError(f"/etc/{name} is not a regular file in the image")
        f = tar.extractfile(members[0])
        assert f is not None
        return f.read().decode()


#: The repository the step image is tagged in, beside the 32 hex digits of its manifest digest.
STEP_IMAGE_REPO = "hls/step"
#: Seconds the daemon may take over any one read or write of loading the step image.
STEP_LOAD_TIMEOUT_SEC = 600.0


def archive_digests(archive: Path) -> tuple[str, str]:
    """The manifest and config digests (``sha256:<hex>``) of the one image an OCI image layout
    archive holds. Raise ValueError for an archive that holds none or several."""
    try:
        with tarfile.open(archive) as tar:
            index = json.load(tar.extractfile("index.json"))  # type: ignore[arg-type]
            (manifest,) = index["manifests"]
            digest = str(manifest["digest"])
            hexa = check_sha256_hex(digest.removeprefix("sha256:"))
            blob = json.load(tar.extractfile(f"blobs/sha256/{hexa}"))  # type: ignore[arg-type]
            config = str(blob["config"]["digest"])
            check_sha256_hex(config.removeprefix("sha256:"))
    except (OSError, KeyError, TypeError, ValueError, tarfile.TarError, AttributeError) as e:
        raise ValueError(f"{archive} is not the OCI layout of one image: {type(e).__name__}: {e}") from None
    return digest, config


def load_step_image(client: DockerClient, archive: Path, *, sha256: str, scratch: Path) -> str:
    """Put the step image on this host and return the daemon's ID of it: copy *archive* once into
    *scratch*, check that the copy's sha256 is *sha256*, load the copy, and check that the image the
    daemon reports loading has as its ID the manifest digest (containerd's image store) or the
    config digest (the classic store) the copy's index names; then tag it
    ``hls/step:<32 hex of the manifest digest>``. The copy is removed. Raise ValueError when a digest
    differs or the daemon reports not exactly one image, and DockerError when it refuses."""
    check_sha256_hex(sha256)
    scratch.mkdir(parents=True, exist_ok=True)
    copy = scratch / "step.oci.tar"
    copy.unlink(missing_ok=True)
    try:
        shutil.copyfile(archive, copy)
        got, _ = sha256_file(copy)
        if got != sha256:
            raise ValueError(f"the step image {archive} has sha256 {got}, not {sha256}")
        manifest, config = archive_digests(copy)
        loaded = client.load_image(copy, timeout=STEP_LOAD_TIMEOUT_SEC)
        if len(loaded) != 1:
            raise ValueError(f"the daemon reports loading {list(loaded)} from the step image, not one image")
        # The classic store knows no image by its manifest digest, so the image is the one the load
        # names: its ID, or the name the archive gave it.
        info = client.inspect_image(loaded[0])
        if info.id not in (manifest, config):
            raise ValueError(f"the daemon names the step image {info.id}, neither its manifest {manifest} nor its "
                             f"config {config}")
        client.tag_image(info.id, repo=STEP_IMAGE_REPO, tag=manifest.removeprefix("sha256:")[:32])
        return info.id
    finally:
        copy.unlink(missing_ok=True)


class DockerStopper(Stopper):
    """Stops every container labelled with a job's id, for the local scheduler at the job's end:
    each is killed, waited for and removed, and none may be left. Then it removes what the job's
    image roles left under the job's name, builders, their volumes and image names
    (``hls.substrate.docker.images.remove_job_leftovers``), which raises when any is left."""

    def __init__(self, *, socket_path: Path, wait_sec: float = STOP_WAIT_SEC) -> None:
        #: The socket of the daemon the job's containers run under, the launcher's.
        self.socket_path = socket_path
        self.wait_sec = wait_sec

    def stop(self, job_id: str) -> None:
        client = DockerClient(self.socket_path)
        try:
            for row in client.list_containers(labels={JOB_LABEL: job_id}):
                if row.state in ("running", "paused", "restarting"):
                    try:
                        client.kill_container(row.id, signal.SIGKILL)
                        client.wait_container(row.id, timeout=self.wait_sec)
                    except NotFound:
                        continue
                    except DockerError as e:
                        if getattr(e, "status", None) != 409:
                            raise
                try:
                    client.remove_container(row.id, force=True)
                except NotFound:
                    pass
            left = client.list_containers(labels={JOB_LABEL: job_id})
            if left:
                raise RuntimeError(f"containers of job {job_id} are left: {[r.id[:12] for r in left]}")
            remove_job_leftovers(client, job_id)
        finally:
            client.close()
