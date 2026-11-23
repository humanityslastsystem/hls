"""A stand-in for ``apptainer build --sandbox`` and ``apptainer exec`` that also gives a
container its own paths, so the real artifact and driver programs run on machines without
apptainer.

An image file is a tar file. ``build --sandbox <dir> <image file>`` unpacks it into *dir*, which
must not exist yet, and fails as apptainer does (a ``FATAL:`` line, exit 255) for a file that is
not a tar file. ``exec`` accepts the command lines ``ApptainerRuntime.argv`` builds and refuses
one that lacks an isolation flag, asks for a writable image or names an image that is not a
directory (exit 2), as ``tests.substrate.fake_apptainer`` does. It lays the container's root out
in ``<workdir>/root`` without writing to the image: a link to each of the image's entries, real
directories only on the way to a bind's target, and a link at each target to the bind's source.
Every argument and variable that names a path under one of the root's top-level entries is
rewritten to that path under the root, so ``/engine-cache/vllm`` or ``/opt/hls-src/src`` resolve
as they would inside the container, and ``/bin/sh`` stays the host's. Python programs in the
container also load ``fake_site/sitecustomize.py``, which points the driver's dataset root at
``<root>/data``. The container's variables are those of the env file, ``HOME`` under the
workdir, and ``PATH``; nothing else of the fake's own environment.

A sealed container's seal forwards a host port to the container's ``--port``. With no network
namespace to forward into, the fake has the program listen on the host port itself: when
``HLS_INGRESS_PORT_FILE`` is set, the value after ``--port`` becomes the port in that file. The
seal forwards nothing else, so no other port the container binds is reachable from the node; the
fake has the seal probe bind its inbound ports (the values after ``--inbound-tcp`` and
``--inbound-udp``) on ports the system picks, which nothing knocks on. A ``python -I`` program in
a fake seal's namespace runs under a prelude that stands in for that namespace's kernel: in a seal
whose namespace drops its ICMP (``FAKE_NETNS_RULES`` of ``tests.substrate.fake_pasta``, the rules
its block installed) its ICMP sends are refused, its reads of ``/proc/sys/user`` answer the
limits the seal wrote there (``FAKE_NETNS_SYSCTLS``), a limit it never wrote reading as a fresh
user namespace's own, 2147483647, and its writes there are refused (EACCES) unless the container
keeps ``CAP_SYS_RESOURCE``, when they take. Its ``/proc/self/status`` shows, in every capability
set, what root of the seal's namespace holds once ``--drop-caps`` dropped the capabilities it
names (``ALL`` for every one).

Knobs:
- ``FAKE_SEAL_FORWARDS=1``: leave the seal probe's inbound ports as given, as a seal that forwards
  every port its container binds would, so the stage reaches them on the node;
- ``FAKE_CONTAINER_PATH``: the PATH inside the container (an image's own), default the fake's;
- ``FAKE_CONTAINER_RECORDS=<dir>``: write ``<dir>/<role>.json`` with what it was asked to run,
  as given: the image, the workdir, the binds, the GPU flag, the dropped capabilities, the env
  file's variables, the command, and whether the fake pasta of ``tests.substrate.fake_pasta``
  started it (``in_pasta``). *role* is the name of the workdir's parent, the role's own
  directory;
- ``FAKE_CONTAINER_HOLD=<role>:<path>``: the container of *role* waits until *path* exists
  before anything of it runs, so before its start marker, as a runtime slow to start a
  container would, and exits 1 at once when the process that started it is gone;
- ``FAKE_COMMAND_STARTED=<path>``: handed to the container's command, whose runscript in the
  drive tests' server image writes there the time it started.

``install(dir)`` writes an executable ``apptainer`` into *dir* that runs this file with the
test's interpreter; ``make_image(path, files)`` writes a tar image."""

from __future__ import annotations

import io
import json
import os
import shlex
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

SITE = Path(__file__).with_name("fake_site")
#: Runs the program named after it as in the fake seal's namespace: ICMP sends refused when
#: ``DROPS_ICMP``, as the kernel refuses them where the firewall drops them; reads of
#: ``/proc/sys/user/NAME`` answering ``USER_LIMITS`` (by NAME), or a fresh user namespace's own;
#: writes there taking only when ``CAPS`` holds CAP_SYS_RESOURCE, as the kernel lets only its
#: holder in the namespace write them; and ``/proc/self/status`` showing ``CAPS`` in every
#: capability set.
_NAMESPACE = """import errno, json, pathlib, re, runpy, socket, sys
DROPS_ICMP, USER_LIMITS, CAPS = {drops_icmp!r}, json.loads({user_limits!r}), {caps!r}
class _Socket(socket.socket):
    def sendto(self, *args):
        if DROPS_ICMP and self.proto == socket.IPPROTO_ICMP:
            raise PermissionError(1, "Operation not permitted")
        return super().sendto(*args)
socket.socket = _Socket
_read_text = pathlib.Path.read_text
def read_text(self, *args, **kwargs):
    if str(self).startswith("/proc/sys/user/"):
        return USER_LIMITS.get(self.name, "2147483647") + "\\n"
    if str(self) == "/proc/self/status":
        return re.sub(r"(?m)^(Cap(?:Inh|Prm|Eff|Bnd|Amb):\\t)[0-9a-f]+$", rf"\\g<1>{{CAPS:016x}}",
                      _read_text(self, *args, **kwargs))
    return _read_text(self, *args, **kwargs)
pathlib.Path.read_text = read_text
_write_text = pathlib.Path.write_text
def write_text(self, data, *args, **kwargs):
    if str(self).startswith("/proc/sys/user/"):
        if not CAPS >> 24 & 1:
            raise PermissionError(errno.EACCES, "Permission denied", str(self))
        USER_LIMITS[self.name] = data.strip()
        return len(data)
    return _write_text(self, data, *args, **kwargs)
pathlib.Path.write_text = write_text
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
"""
#: The rule of the seal's firewall that drops every ICMP packet its namespace sends.
_ICMP_DROP = ["-A", "OUTPUT", "-p", "icmp", "-j", "DROP"]
#: Every capability the kernel knows (CAP_LAST_CAP 40), each a bit, as root of a user namespace
#: holds them; and the bits of the ones a sealed container's ``--drop-caps`` may name.
_ALL_CAPS = (1 << 41) - 1
_CAP_BITS = {"CAP_NET_ADMIN": 12, "CAP_NET_RAW": 13, "CAP_SYS_RESOURCE": 24}


def _caps_kept(drop_caps: str | None) -> int:
    """The capabilities, as bits, root of the seal's namespace keeps once *drop_caps* dropped its
    own."""
    names = drop_caps.split(",") if drop_caps else []
    if "ALL" in names:
        return 0
    kept = _ALL_CAPS
    for name in names:
        kept &= ~(1 << _CAP_BITS[name])
    return kept


def _namespace_prelude(drop_caps: str | None) -> str | None:
    """The prelude that runs a program as in the fake seal's namespace this runs in, in a
    container that drops the capabilities *drop_caps* names; None outside one."""
    rules, sysctls = os.environ.get("FAKE_NETNS_RULES"), os.environ.get("FAKE_NETNS_SYSCTLS")
    if os.environ.get("FAKE_PASTA_NAMESPACE") != "1" or not rules or not sysctls:
        return None
    drops = Path(rules).exists() and _ICMP_DROP in [json.loads(line) for line in Path(rules).read_text().splitlines()]
    held = json.loads(Path(sysctls).read_text()) if Path(sysctls).exists() else {}
    limits = {key.removeprefix("user."): value for key, value in held.items() if key.startswith("user.")}
    return _NAMESPACE.format(drops_icmp=drops, user_limits=json.dumps(limits), caps=_caps_kept(drop_caps))


_FLAGS = {"--containall", "--cleanenv", "--pid", "--nv"}
_OPTIONS = {"--no-mount", "--workdir", "--cwd", "--bind", "--env-file", "--drop-caps"}
_REQUIRED_FLAGS = ("--containall", "--cleanenv", "--pid")
_NO_MOUNT = "hostfs,bind-paths,cwd"


def install(directory: Path) -> Path:
    """Write an executable ``apptainer`` into *directory* and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "apptainer"
    path.write_text(f"#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(__file__)} \"$@\"\n")
    path.chmod(0o755)
    return path


def make_image(path: Path, files: dict[str, tuple[bytes, int]]) -> Path:
    """Write a tar image at *path* holding each file name's bytes with its mode."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, (data, mode) in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    return path


def records(directory: Path) -> dict[str, dict]:
    """What each container recorded, by role."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))}


def _usage(message: str) -> int:
    print(f"fake container: {message}", file=sys.stderr)
    return 2


def _parent_of(pid: int) -> int | None:
    """The parent of process *pid*; None once it is gone."""
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, IndexError, ValueError):
        return None


def _held(role: str) -> bool:
    """Wait until the hold ``FAKE_CONTAINER_HOLD`` names for *role*, if any, is released;
    False once the process that started this one, or the one that started that, is gone
    first."""
    held, _, gate = os.environ.get("FAKE_CONTAINER_HOLD", "").partition(":")
    if held != role:
        return True
    parent = os.getppid()
    chain = (parent, _parent_of(parent))
    while not Path(gate).exists():
        if (os.getppid(), _parent_of(parent)) != chain:
            return False
        time.sleep(0.05)
    return True


def _read_env_file(path: str) -> dict[str, str]:
    env = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        [word] = shlex.split(line)
        name, _, value = word.partition("=")
        env[name] = value
    return env


def _inside(root: Path, value: str) -> str:
    """*value* as a path under *root* when it names a path under one of root's entries."""
    if not value.startswith("/"):
        return value
    first = value[1:].split("/", 1)[0]
    if first and os.path.lexists(root / first):
        return f"{root}{value}"
    return value


def extract(argv: list[str]) -> int:
    """``build --sandbox <dir> <image file>``, as the module docstring says."""
    if len(argv) != 4:
        return _usage(f"unexpected build arguments {argv}")
    dest, image = Path(argv[2]), Path(argv[3])
    if os.path.lexists(dest):
        return _usage(f"{dest} exists; apptainer would not overwrite it without --force")
    if not image.is_file() or not tarfile.is_tarfile(image):
        print(f"FATAL:   could not open image {image}: not a valid image", file=sys.stderr)
        return 255
    dest.mkdir()
    with tarfile.open(image) as tar:
        tar.extractall(dest, filter="data")
    return 0


def _lay_out(root: Path, image: Path, binds: list[tuple[str, str]]) -> None:
    """The container's root at *root*: the image's entries linked, never written, and each
    bind's source linked at its target."""
    targets = {PurePosixPath(target) for _, target in binds}
    on_the_way = {PurePosixPath("/")} | {parent for t in targets for parent in t.parents}
    for directory in sorted(on_the_way, key=lambda p: len(p.parts)):
        here = root / directory.relative_to("/")
        here.mkdir(exist_ok=True)
        source = image / directory.relative_to("/")
        if not source.is_dir() or source.is_symlink():
            continue
        for entry in sorted(os.listdir(source)):
            if directory / entry not in on_the_way and directory / entry not in targets:
                (here / entry).symlink_to(source / entry)
    for source, target in binds:
        link = root / target.lstrip("/")
        if os.path.lexists(link):
            link.unlink()
        link.symlink_to(source)


def main(argv: list[str]) -> int:
    if argv[:2] == ["build", "--sandbox"]:
        return extract(argv)
    if argv[:1] != ["exec"]:
        return _usage(f"only exec and build --sandbox are supported, got {argv[:1]}")
    flags: set[str] = set()
    options: dict[str, list[str]] = {}
    i = 1
    while i < len(argv) and argv[i].startswith("--"):
        if argv[i] in _FLAGS:
            flags.add(argv[i])
            i += 1
        elif argv[i] in _OPTIONS and i + 1 < len(argv):
            options.setdefault(argv[i], []).append(argv[i + 1])
            i += 2
        else:
            return _usage(f"unknown or incomplete option {argv[i]}")
    missing = [f for f in _REQUIRED_FLAGS if f not in flags]
    if missing or options.get("--no-mount") != [_NO_MOUNT] or len(options.get("--workdir", [])) != 1:
        return _usage(f"missing isolation: {missing or options.get('--no-mount') or '--workdir'}")
    image, inside = Path(argv[i]), argv[i + 1:]
    if not image.is_dir():
        return _usage(f"image {image} is not an extracted directory")
    workdir = Path(options["--workdir"][0])
    root = workdir / "root"
    root.mkdir(parents=True)
    binds = [tuple(b.split(":")[:2]) for b in options.get("--bind", [])]
    _lay_out(root, image, binds)
    given = {}
    for path in options.get("--env-file", []):
        given.update(_read_env_file(path))
    command = inside[4:]  # after /bin/sh -c <wrapper> <marker>
    record = os.environ.get("FAKE_CONTAINER_RECORDS")
    if record:
        Path(record).mkdir(parents=True, exist_ok=True)
        (Path(record) / f"{workdir.parent.name}.json").write_text(json.dumps({
            "image": str(image), "workdir": str(workdir), "nv": "--nv" in flags, "binds": options.get("--bind", []),
            "drop_caps": (options.get("--drop-caps") or [None])[0], "env": given, "command": command,
            "in_pasta": os.environ.get("FAKE_PASTA_NAMESPACE") == "1"}))
    home = workdir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env = {k: ":".join(_inside(root, part) for part in v.split(":")) for k, v in given.items()}
    env.update(PATH=os.environ.get("FAKE_CONTAINER_PATH", os.environ.get("PATH", os.defpath)), HOME=str(home),
               FAKE_CONTAINER_ROOT=str(root),
               **{k: os.environ[k] for k in ("FAKE_COMMAND_STARTED",) if k in os.environ},
               PYTHONPATH=os.pathsep.join(p for p in (str(SITE), env.get("PYTHONPATH")) if p))
    command = [_inside(root, a) for a in command]
    port_file = os.environ.get("HLS_INGRESS_PORT_FILE")
    if port_file and "--port" in command:
        command[command.index("--port") + 1] = Path(port_file).read_text().strip()
    if port_file and os.environ.get("FAKE_SEAL_FORWARDS") != "1":
        for option in ("--inbound-tcp", "--inbound-udp"):
            if option in command:
                command[command.index(option) + 1] = "0"
    drop_caps = (options.get("--drop-caps") or [None])[0]
    prelude = _namespace_prelude(drop_caps) if command[:2] == ["python", "-I"] else None
    if prelude is not None:
        command = [*command[:2], "-c", prelude, *command[2:]]
    if not _held(workdir.parent.name):
        return 1
    os.execve(inside[0], [*inside[:4], *command], env)
    return 0  # not reached


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
