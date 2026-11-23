"""The ``apptainer exec`` command line of every container role.

Every container gets the same isolation, whoever owns it:
- ``--containall --cleanenv --pid``. ``--containall`` already implies ``--contain``, a PID
  and an IPC namespace and ``--cleanenv``; the flags are spelled out so the command line
  says what the container gets. With its own PID namespace a contestant's server cannot
  see or signal srun, the runner or the harness, so it cannot end its own step to turn a
  score into a retried infra fault.
- ``--no-mount hostfs,bind-paths,cwd``. A site's ``apptainer.conf`` can bind paths into
  every container; with this flag none of them, nor the host
  filesystem or the working directory, reaches a role. Binds named by ``APPTAINER_BIND`` and
  the like in the environment apptainer starts in are not dropped by it; the step wrapper
  removes every such variable first (``read_by_runtime``). Then only the declared mounts and
  the two host files below reach a container, so a contestant never sees its own log
  directory, the store, the munge socket or the Slurm configuration unless a stage mounts
  them, and stages never do.
- The image is a directory extracted on the node (``ContainerSpec.image``, a pinned image),
  which apptainer runs as the container's root. It is never given ``--writable``, so the root
  is read-only: the writable paths are /tmp, /var/tmp and $HOME, backed by the workdir, and
  the mounts declared with ``read_only=False``. No squashfuse daemon serves the image, so none
  can die under a running container and end it with SIGBUS, an exit that would read as the
  command's. ``hls.substrate.apptainer.images.ApptainerImages`` makes the directory with
  ``apptainer build --sandbox``, which reads the image file's own partition table and unpacks
  its squashfs root without privileges.
  The same step first copies the image file into the extraction role's own directory, its one
  read of the file, and extracts that copy only if it has the sha256 its producer recorded.
- An unsealed container shares the host's network namespace: every TCP port and abstract
  Unix socket on the node is reachable from it. A sealed one (``ContainerSpec.seal``) runs
  behind the seal prefix of ``hls.substrate.apptainer.egress``: apptainer starts only inside
  the network namespace pasta made, once the redirect and the interceptor are up, so nothing
  of the container's runs on the host's network. It also loses ``CAP_NET_ADMIN``,
  ``CAP_NET_RAW`` and ``CAP_SYS_RESOURCE`` (``--drop-caps``), so it cannot rewrite the seal's
  firewall, forge the interceptor's mark or raise the limits the seal wrote; a contestant's
  payload (``ContainerSpec.contestant_payload``) loses every capability. A seal that fails
  before apptainer starts leaves no start marker, like any other runtime failure.
- /etc/resolv.conf and /etc/hosts are bound read-only by the runtime itself, because
  ``--no-mount bind-paths`` also drops apptainer's own binds of them, and a container
  without them cannot resolve names.
- ``--workdir <node-local dir>`` backs /tmp, /var/tmp and $HOME with disk. Without it they
  are apptainer's session tmpfs, 64 MiB by default. HOME cannot be set
  through an env file, since apptainer refuses to override it, so the workdir is how a role
  gets a writable home.
- Variables reach the container through ``--env-file``, with the exceptions below. Apptainer reads
  and expands that file on the host before the container starts, so secrets live in a
  node-local file and never on the command line, which every user of the node can read.
  The exceptions are ``--cleanenv`` itself and apptainer's own variables. ``--cleanenv``
  still passes TERM and the proxy variables (http_proxy, https_proxy, no_proxy, all_proxy
  and ftp_proxy, in lower and upper case) from the environment the runtime starts in, sets
  LANG=C and replaces PATH; apptainer also copies every ``APPTAINERENV_X`` in as ``X``. The
  step wrapper removes all of them, and apptainer's settings, from a container step's
  environment before apptainer starts (``read_by_runtime``), and then sets
  ``APPTAINERENV_CUDA_VISIBLE_DEVICES``, so the one host variable that crosses is the step's
  GPU list.
- The command runs under ``/bin/sh -c``. The shell first closes descriptor 3, on which the
  step wrapper hands over the env file and which apptainer has already read, so the command
  and everything it starts do not inherit it (it would be readable at /proc/self/fd/3).
  Apptainer's own PID 1 in the container keeps a copy of the pipe, which apptainer has
  already drained. The shell then prints ``START_MARKER`` on stderr and, only if that write
  succeeded, becomes the command.
  Apptainer's own failures (a missing or corrupt image, a mount or namespace that cannot
  be set up) exit nonzero before the marker, so a nonzero exit without it is the runtime's
  failure and never the command's.

A spec the command line cannot express exactly is refused. Apptainer's bind parser splits a
value on ":", "," and "^", trims spaces and reads "\\" as an escape, so mount sources and
targets, the env file path, the image, the workdir and the cwd hold only
[A-Za-z0-9._@=+/-], in normal form and without a leading "//" (apptainer cleans a workdir
textually, where a check that follows symlinks would see another directory). A later bind on
the same target silently shadows an earlier one. No mount may use a path under /proc, /sys
or /dev, which apptainer provides itself, and none may use the host root "/" as its source,
because binds are recursive and it would carry the host's /proc, /sys and /dev in. A device
node the spec names (``ContainerSpec.devices``) is the one path under /dev that is bound: the
node's own, at the same path, writable, into the minimal /dev that ``--containall`` gives, after
the declared mounts. Each is a node under /dev, given once."""

from __future__ import annotations

import posixpath
import re

from hls.substrate.apptainer.egress import ATTEMPT_LINE, seal_prefix
from hls.substrate.container import ContainerRuntime, ContainerSpec

#: The line every container prints on stderr just before its command starts.
START_MARKER = "hls-container-started"
#: Prefixes of the variables apptainer reads as its own settings (``APPTAINER_BIND``,
#: ``APPTAINER_CACHEDIR``), in their older ``SINGULARITY_`` spelling too, and of the ones it
#: copies into a container past ``--cleanenv`` (``APPTAINERENV_X`` becomes ``X``).
RUNTIME_PREFIXES = ("APPTAINER_", "SINGULARITY_", "APPTAINERENV_", "SINGULARITYENV_")
#: Host files every container gets, read-only, at the same path.
HOST_FILES = ("/etc/resolv.conf", "/etc/hosts")
# What /bin/sh runs: close the env file's descriptor, print the marker ($0), then become the command.
_WRAPPER = 'exec 3<&-; printf "%s\\n" "$0" >&2 && exec "$@"'
# The only characters a path may hold: apptainer's bind parser reads every other one specially.
_SAFE_PATH = re.compile(r"/[A-Za-z0-9._@=+/-]*")
# Paths apptainer provides itself; a bind there fails or replaces them.
_RUNTIME_ROOTS = ("/proc", "/sys", "/dev")
#: Capabilities a sealed container loses: the ones that could change the seal's firewall or
#: mark a socket as the interceptor's own, and the one that could raise the limits the seal wrote
#: in pasta's user namespace (``hls.substrate.ucounts``).
SEALED_DROP_CAPS = "CAP_NET_ADMIN,CAP_NET_RAW,CAP_SYS_RESOURCE"
#: What a sealed contestant payload loses (``ContainerSpec.contestant_payload``): every capability,
#: from every set, so it holds none and no program it runs can take one.
PAYLOAD_DROP_CAPS = "ALL"


def passed_by_cleanenv(name: str) -> bool:
    """Whether ``--cleanenv`` passes the variable *name* from the environment apptainer starts
    in into the container: TERM, and the proxy variables. Apptainer passes the http, https,
    no, all and ftp ones in lower and upper case; every name ending in ``_proxy``, in any
    case, counts here."""
    return name == "TERM" or name.lower().endswith("_proxy")


def read_by_runtime(name: str) -> bool:
    """Whether apptainer reads the variable *name* from the environment it starts in: one
    ``--cleanenv`` passes into the container (``passed_by_cleanenv``), one of apptainer's own
    settings, or one it copies into the container (``RUNTIME_PREFIXES``). Of its settings,
    ``APPTAINER_BIND`` adds binds that ``--no-mount bind-paths`` does not drop, and the rest set
    what the command line leaves unset, such as ``APPTAINER_HOME`` or ``APPTAINER_NETWORK``."""
    return passed_by_cleanenv(name) or name.startswith(RUNTIME_PREFIXES)


def _is_normal(path: str) -> bool:
    """True when *path* is absolute, in normal form and not a POSIX "//" root."""
    return path.startswith("/") and not path.startswith("//") and posixpath.normpath(path) == path


def _check_safe(what: str, path: str) -> None:
    """Raise ValueError unless apptainer reads *path* back exactly as written."""
    if not _SAFE_PATH.fullmatch(path):
        raise ValueError(f"{what} {path!r} cannot be passed to apptainer exactly: "
                         f"the allowed characters are [A-Za-z0-9._@=+/-]")


def exact_path(what: str, path: str) -> str:
    """*path*, *what* in a message, once it is an absolute path in normal form that apptainer
    reads back exactly; raise ValueError otherwise."""
    if not _is_normal(path):
        raise ValueError(f"{what} must be an absolute path in normal form, got {path!r}")
    _check_safe(what, path)
    return path


def _check_not_runtime_owned(path: str) -> None:
    for root in _RUNTIME_ROOTS:
        if path == root or path.startswith(root + "/"):
            raise ValueError(f"mount path {path} is under {root}, which apptainer provides itself")


def _check(spec: ContainerSpec) -> None:
    """Raise ValueError when the command line could not express *spec* exactly."""
    for what, path in (("image", spec.image.ref), ("workdir", str(spec.workdir))):
        if not _is_normal(path):
            raise ValueError(f"{what} must be an absolute path in normal form, got {path!r}")
        _check_safe(what, path)
    if not spec.command:
        raise ValueError("the container has no command")
    if spec.cwd is not None:
        if not _is_normal(spec.cwd):
            raise ValueError(f"cwd must be an absolute container path in normal form, got {spec.cwd!r}")
        _check_safe("cwd", spec.cwd)
    targets: set[str] = set()
    for m in spec.mounts:
        source = str(m.source)
        if not _is_normal(source):
            raise ValueError(f"mount source must be an absolute path in normal form, got {source!r}")
        if source == "/":
            raise ValueError("cannot mount the host root: binds are recursive, so it would expose the host's "
                             "/proc, /sys and /dev")
        if not _is_normal(m.target):
            raise ValueError(f"mount target must be an absolute container path in normal form, got {m.target!r}")
        if m.target == "/":
            raise ValueError(f"cannot mount over / (source {m.source})")
        _check_safe("mount source", source)
        _check_safe("mount target", m.target)
        _check_not_runtime_owned(source)
        _check_not_runtime_owned(m.target)
        if m.target in HOST_FILES:
            raise ValueError(f"{m.target} is bound by the runtime itself")
        if m.target in targets:
            raise ValueError(f"two mounts on {m.target}")
        targets.add(m.target)
    for device in spec.devices:
        if not isinstance(device, str) or not _is_normal(device) or not device.startswith("/dev/"):
            raise ValueError(f"a device is a node under /dev in normal form, got {device!r}")
        _check_safe("device", device)
        if spec.devices.count(device) > 1:
            raise ValueError(f"device {device} is named twice")


class ApptainerRuntime(ContainerRuntime):
    """Containers as ``apptainer exec`` command lines."""

    start_marker = START_MARKER
    #: What ``--workdir`` makes in the workdir: ``tmp`` for /tmp, ``var_tmp`` for /var/tmp and
    #: ``home`` for $HOME under ``--containall``.
    workdir_dirs = ("tmp", "var_tmp", "home")
    attempt_line = ATTEMPT_LINE
    #: pasta and the interceptor run in the role's own step, under its limit.
    seal_memory_gib = 0

    def __init__(self, apptainer: str = "apptainer") -> None:
        #: The apptainer program: a name found on the role's PATH, or an absolute path.
        self.apptainer = apptainer

    def check(self, spec: ContainerSpec) -> None:
        _check(spec)

    def argv(self, spec: ContainerSpec, *, env_file: str, seal_python: str | None = None) -> list[str]:
        """The command line that runs *spec*. *env_file* is the path apptainer reads the
        container's variables from, such as ``/dev/fd/3``; every container has one, which may
        hold no variable. *seal_python* is the host interpreter that runs a sealed container's
        interceptor, and a sealed spec needs one. Raise ValueError for a spec or an *env_file*
        the command line cannot express exactly."""
        _check(spec)
        if not isinstance(env_file, str) or not _is_normal(env_file):
            raise ValueError(f"env_file must be an absolute path in normal form, got {env_file!r}")
        _check_safe("env_file", env_file)
        prefix: list[str] = []
        if spec.seal is not None:
            if not seal_python:
                raise ValueError("a sealed container needs the interpreter that runs its interceptor")
            prefix = seal_prefix(spec.seal, python=seal_python)
        argv = [self.apptainer, "exec", "--containall", "--cleanenv", "--pid", "--no-mount", "hostfs,bind-paths,cwd",
                "--workdir", str(spec.workdir)]
        if spec.seal is not None:
            argv += ["--drop-caps", PAYLOAD_DROP_CAPS if spec.contestant_payload else SEALED_DROP_CAPS]
        if spec.gpus:
            argv.append("--nv")
        if spec.cwd is not None:
            argv += ["--cwd", spec.cwd]
        for f in HOST_FILES:
            argv += ["--bind", f"{f}:{f}:ro"]
        for m in spec.mounts:
            argv += ["--bind", f"{m.source}:{m.target}" + (":ro" if m.read_only else "")]
        for device in spec.devices:
            argv += ["--bind", f"{device}:{device}"]
        argv += ["--env-file", env_file]
        return [*prefix, *argv, spec.image.ref, "/bin/sh", "-c", _WRAPPER, START_MARKER, *spec.command]
