"""The harbor environment of the build sandbox: harbor's agents drive the sandbox through it.

The agent_build stage starts the sandbox as a container role whose one harness process is the
exec server (``hls.agent.exec_server``). ``SandboxEnvironment`` is what harbor's installed agents
see in its place, a harbor ``BaseEnvironment``:
- ``exec`` posts harbor's exec request to the server's UNIX socket, which the container binds
  from the role's 0700 control directory, with the sandbox's token. The client is httpx with
  its own transport over that socket and ``trust_env=False``, so neither a proxy nor a netrc of
  the runner's environment applies to it. Connecting is bounded; reading is not, because the
  agent CLI runs as one exec for hours and the stage's hard limit cancels it. ``user`` is
  accepted and ignored, as harbor's singularity backend ignores it without fakeroot: every
  command runs as the one user the container has.
- File transfers copy on the host, between the caller's path and the host directory that a
  mapped container directory is bound from (the workspace, the agent's logs, the task's files,
  the container's tmp and home). Nothing passes through a directory the sandbox can write
  while the harness reads it, as harbor's writable ``/staging`` did. A container path
  outside every mapped directory is refused. Below the mapped directory every path component is
  opened without following symlinks, relative to its parent, so a link the contestant planted
  is refused wherever it points. An upload's target is opened without blocking and must be a
  regular file, so a pipe left there neither holds the harness nor receives the file. A refusal
  that what the sandbox holds at the path explains (a link, a pipe, a lease, a mode) is a
  ``ValueError``, never a raw ``OSError``, which stays the machine's own failure.
- A downloaded tree is walked without recursing, within bounds, and a link in
  it is neither followed nor made on the host: what the walk cannot use is left out and recorded in
  ``skipped``. The downloads are still for trees the harness wrote; a tree the agent wrote must
  not be read through them unbounded.
- No mapped container directory may hold or lie under one of the paths the harness binds
  itself (``HARNESS_PATHS``: the exec directory, the control directory, the budget channel and
  the MCP server), where a transfer would land beside the bind the sandbox sees. The socket's
  directory may not lie in a mapped host directory: the sandbox writes those, and could put its
  own server on the socket.

A command's output that the server cut to its last bytes starts where a secret the sandbox holds
may have been cut through, which a scrub of whole values cannot find: ``exec`` drops the piece of
such a form at the cut (``hls.substrate.secret_forms.back_off``) before harbor reads it.

A command's end is its result, whatever ended it. A command killed along with its namespace (one
that ended PID 1 of the commands' namespace, the reaper) comes back killed by SIGKILL, and the
server runs the next one in a fresh namespace; a command that failed because the sandbox ran out
of processes comes back with its own status and output. The stage judges such ends from the
step's own counters, never from this channel.

The channel is lost when a connection to the server is refused or reset, or closed before a
reply. That is all this side measures: the server, or the container around it, going away, or a
server that dropped a connection, which it does only when its own machinery fails (it answers
every request it could not serve). Which it was is the stage's to judge, from the role's own
ending and log. ``exec`` then raises ``SandboxLost`` and keeps it in ``lost``,
and every later ``exec`` or ``start`` raises it again at once, whatever answers on the socket
later. The loss holds what was lost and why, never the failed request, whose header holds the
token and whose body holds the route's credential. Harbor's adapters can hide it: Claude Code's
re-entry loop raises the agent's own nonzero exit when examining that exit fails, and codex
swallows a failed cleanup. So the stage reads ``lost`` after the agent returns or raises, whatever
it raised, to tell a sandbox death from an agent's exit. A reply that is not 200,
or not a result, is ``SandboxError``: a harness defect, never the contestant's exit. Both
subclass ``RuntimeError``, which harbor's ``setup`` re-raises as it is, and neither is harbor's
``NonZeroAgentExitCodeError``, so a sandbox failure is never read as a command's status.

``start`` waits for the server's authenticated ``/health``: the server binds its socket only
once its spawner is ready, so an answer means commands can run. Its wait counts from the
runtime's start of the server's command, the container's start marker, which the stage waits for
first: the runtime's and the seal's own start before it are the harness's, bounded by the stage's
deadline alone. It keeps to its deadline even on a socket that accepts and never answers.
``stop`` closes the client; stopping the container
is the stage's, through its role."""

from __future__ import annotations

import asyncio
import errno
import os
import stat
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

import httpx
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from hls.agent import exec_protocol as protocol
from hls.agent import transcript
from hls.budget import mcp
from hls.substrate.secret_forms import SecretForm, back_off
from hls.util import MAX_SOCKET_PATH_BYTES, remove_tree

#: Seconds ``start`` waits for the server's health check to answer, counted from the runtime's
#: start of the server's command.
START_TIMEOUT_SEC = 300.0
#: Seconds between health checks while ``start`` waits.
START_POLL_SEC = 0.5
#: Seconds a connection to the server may take; reading a reply has no limit.
CONNECT_TIMEOUT_SEC = 10.0
#: The container paths the harness binds into the sandbox, which no mapped directory may overlap.
HARNESS_PATHS = (protocol.EXEC_TARGET, protocol.CONTROL_TARGET, mcp.CHANNEL_TARGET, mcp.MCP_TARGET)
# ``download_dir`` copies the agent's logs within the transcript copy's bounds, read when it runs:
# ``hls.agent.transcript.MAX_DEPTH`` directory levels below the downloaded
# directory, ``MAX_PATH_BYTES`` of an entry's path below it, ``MAX_BYTES`` copied in all (checked
# before each file) and ``MAX_ENTRIES`` entries looked at.
# Opens a directory below a mapped one: never through a symlink.
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# Opens an upload's target: never through a symlink, and without waiting for a pipe's reader.
_WRITE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
# Opens a downloaded file: never through a symlink, and without waiting for a pipe's writer.
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
# What opening a path component through a symlink, or through a file, fails with.
_NOT_A_DIR = (errno.ELOOP, errno.ENOTDIR)
# What a transfer fails with because of what the sandbox holds at the path: a link, a file where
# a directory belongs or the reverse, a pipe or socket, a lease, a mode, an overlong name, or an
# entry gone or made since it was looked at. Any other failure (a full disk, no descriptors left)
# is the machine's.
_SANDBOX_ERRNOS = frozenset({errno.ELOOP, errno.ENOTDIR, errno.EISDIR, errno.ENXIO, errno.EAGAIN,
                             errno.EACCES, errno.EPERM, errno.ENAMETOOLONG, errno.ETXTBSY, errno.ENOENT,
                             errno.EEXIST})


class SandboxError(RuntimeError):
    """The sandbox's exec server refused or failed a request: a harness defect."""


class SandboxLost(SandboxError):
    """A connection to the exec server was refused or reset, or closed before a reply."""


class SandboxEnvironment(BaseEnvironment):
    """Harbor's environment over a running sandbox's exec socket. See the module docstring."""

    def __init__(self, *, socket: Path, token: str, paths: Mapping[str, Path], root: Path,
                 workdir: str = "/workspace", gpus: int = 0, alive: Callable[[], bool] | None = None,
                 start_timeout_sec: float = START_TIMEOUT_SEC, secret_forms: Sequence[SecretForm] = (),
                 logger=None) -> None:
        #: The exec server's socket, the host side of the control directory's bind.
        self.socket = Path(socket)
        #: The sandbox's token, sent with every request.
        self.token = token
        #: Container directory to the host directory it is bound from; transfers stay inside these.
        self.paths = {posix: Path(host) for posix, host in paths.items()}
        #: Reports whether the sandbox role still runs; ``start`` stops waiting once it does not.
        self.alive = alive
        #: Seconds ``start`` waits for the server.
        self.start_timeout_sec = start_timeout_sec
        #: The forms of the secrets the sandbox holds, none of which a cut output may start inside.
        self.secret_forms = tuple(secret_forms)
        #: The first loss of the exec channel; None while it holds.
        self.lost: SandboxLost | None = None
        #: Each entry ``download_dir`` left out, as its sandbox path and why, in the order met.
        self.skipped: list[tuple[str, str]] = []
        self._client: httpx.AsyncClient | None = None
        root = Path(root)
        # A prebuilt image the stage started: no task definition on disk. The root is the
        # caller's directory, for harbor's bookkeeping only.
        super().__init__(environment_dir=root, environment_name="hls-sandbox", session_id=root.name,
                         trial_paths=TrialPaths(trial_dir=root),
                         task_env_config=EnvironmentConfig(workdir=workdir, gpus=gpus or None), logger=logger)

    @staticmethod
    def type() -> str:
        return "hls-sandbox"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # GPUs and the seal are the container role's; the logs are host directories.
        return EnvironmentCapabilities(gpus=True, disable_internet=True, mounted=True)

    def _validate_definition(self) -> None:
        if len(os.fsencode(self.socket)) > MAX_SOCKET_PATH_BYTES:
            raise ValueError(f"the exec socket path {self.socket} is longer than {MAX_SOCKET_PATH_BYTES} bytes")
        # A header carries the token as it is; the message never shows it.
        if not self.token or not all(" " < c < "\x7f" for c in self.token):
            raise ValueError("the sandbox needs its exec token, printable ASCII without spaces")
        names = list(self.paths)
        for posix in names:
            p = PurePosixPath(posix)
            if not p.is_absolute() or str(p) != posix or p == PurePosixPath("/") or ".." in p.parts:
                raise ValueError(f"mapped container directory {posix!r} must be an absolute path in normal "
                                 "form, not /")
            inside = [o for o in names if o != posix and PurePosixPath(o).is_relative_to(p)]
            if inside:
                raise ValueError(f"mapped container directories may not nest: {inside[0]} is inside {posix}")
            harness = [h for h in HARNESS_PATHS if p.is_relative_to(h) or PurePosixPath(h).is_relative_to(p)]
            if harness:
                raise ValueError(f"mapped container directory {posix} overlaps the harness's {harness[0]}")
            if not self.paths[posix].is_dir():
                raise ValueError(f"the host directory {self.paths[posix]} of {posix} does not exist")
        control = self.socket.parent.resolve()
        for posix, host in self.paths.items():
            if control.is_relative_to(host.resolve()):
                raise ValueError(f"the exec socket's directory {self.socket.parent} lies in {host}, "
                                 f"the host directory of {posix}")

    # The channel

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            transport = httpx.AsyncHTTPTransport(uds=str(self.socket))
            self._client = httpx.AsyncClient(transport=transport, base_url="http://sandbox", trust_env=False,
                                             timeout=httpx.Timeout(None, connect=CONNECT_TIMEOUT_SEC),
                                             headers={protocol.TOKEN_HEADER: self.token})
        return self._client

    def _lose(self, what: str, why: str) -> SandboxLost:
        """The first loss, kept in ``lost``. The callers raise it outside any ``except`` block,
        so it chains to no exception, and so to no request."""
        if self.lost is None:
            self.lost = SandboxLost(f"the sandbox's exec channel at {self.socket} was lost {what}: {why}")
        return self.lost

    def _still_lost(self) -> None:
        if self.lost is not None:
            raise SandboxLost(str(self.lost)) from self.lost

    async def start(self, force_build: bool = False, *, elapsed_sec: float = 0.0) -> None:
        """Wait until the server answers its health check with the token, at most
        ``start_timeout_sec`` from the runtime's start of the server's command, *elapsed_sec*
        before this call. Raise SandboxLost when the sandbox ended first or its channel was lost
        before, and SandboxError when the server refuses the token or does not answer in time."""
        self._still_lost()
        deadline = time.monotonic() + self.start_timeout_sec - elapsed_sec
        last: Exception | None = None
        while True:
            if self.alive is not None and not self.alive():
                why = "the role ended" if last is None else _named(last)
                raise self._lose("before its exec server answered", why)
            # Each check ends by the deadline, also on a socket that accepts and never answers.
            left = max(deadline - time.monotonic(), START_POLL_SEC)
            bounded = httpx.Timeout(left, connect=min(left, CONNECT_TIMEOUT_SEC))
            try:
                resp = await self._http().get("/health", timeout=bounded)
            except httpx.TransportError as e:
                last = e
            else:
                if resp.status_code == 200:
                    return
                raise SandboxError(f"the exec server at {self.socket} answered its health check with "
                                   f"{resp.status_code}: {resp.text[:200]}")
            if time.monotonic() >= deadline:
                raise SandboxError(f"the exec server at {self.socket} did not answer within "
                                   f"{self.start_timeout_sec:.0f} s: {type(last).__name__}: {last}")
            await asyncio.sleep(START_POLL_SEC)

    async def stop(self, delete: bool = False) -> None:
        """Close the client. The container is the stage's to stop."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def exec(self, command: str, cwd: str | None = None, env: dict[str, str] | None = None,
                   timeout_sec: int | None = None, user: str | int | None = None) -> ExecResult:
        """Run *command* in the sandbox and return its combined output and status, however the
        command ended. *user* is ignored: the container has one user. Raise SandboxLost once the
        channel is gone, and SandboxError for a reply that is not a result."""
        self._still_lost()
        body = {"command": command, "cwd": cwd, "env": self._merge_env(env), "timeout_sec": timeout_sec}
        failed: str | None = None
        try:
            resp = await self._http().post("/exec", json=body)
        except httpx.TransportError as e:
            failed = _named(e)
        if failed is not None:
            raise self._lose("during an exec", failed)
        if resp.status_code != 200:
            raise SandboxError(f"the exec server refused an exec with {resp.status_code}: {resp.text[:200]}")
        try:
            out = resp.json()
            return ExecResult(stdout=self._uncut(out.get("stdout")), stderr=out.get("stderr"),
                              return_code=out["return_code"])
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise SandboxError(f"the exec server's reply is not a result: {resp.text[:200]}") from e

    def _uncut(self, stdout: str | None) -> str | None:
        """*stdout* with the piece of a secret's form dropped from the start of its tail, when the
        server cut it to its last bytes (``exec_protocol.dropped_line_length``)."""
        line = protocol.dropped_line_length(stdout) if isinstance(stdout, str) else 0
        if not line:
            return stdout
        return stdout[:line] + back_off(stdout[line:], self.secret_forms, side="start")

    # Files

    def _resolve(self, path: str) -> tuple[Path, list[str]]:
        """The host directory *path* lies under and the components of *path* below it. Raise
        ValueError for a path that is not absolute, has a ``..`` component or lies outside every
        mapped directory."""
        p = PurePosixPath(path)
        if not p.is_absolute() or ".." in p.parts:
            raise ValueError(f"{path!r}: a sandbox path must be absolute, without '..'")
        for posix in sorted(self.paths, key=len, reverse=True):
            if p.is_relative_to(posix):
                return self.paths[posix], list(p.relative_to(posix).parts)
        raise ValueError(f"{path}: not in a directory the sandbox shares with the host "
                         f"({', '.join(sorted(self.paths))})")

    @contextmanager
    def _open_dir(self, path: str, *, create: bool) -> Iterator[int]:
        """A descriptor of the directory *path*, each component below its mapped directory opened
        without following a symlink; with *create*, missing components are made. Raise
        FileNotFoundError for a missing component without *create*."""
        root, parts = self._resolve(path)
        try:
            fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        except OSError as e:
            raise _refused(e, path) from None
        try:
            for name in parts:
                try:
                    nxt = _open_subdir(fd, name, path)
                except FileNotFoundError:
                    if not create:
                        raise
                    _make_subdir(fd, name, path)
                    nxt = _open_made(fd, name, path)
                os.close(fd)
                fd = nxt
            yield fd
        finally:
            os.close(fd)

    def _parent(self, path: str) -> tuple[str, str]:
        p = PurePosixPath(path)
        if p.name in ("", ".", ".."):
            raise ValueError(f"{path!r} names no file")
        return str(p.parent), p.name

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        """Copy the host file *source_path* to *target_path*, making its parents. A target that is a
        directory gets the file inside it, under the source's name, as ``cp`` would. A refusal
        that what the sandbox holds at the target explains is a ValueError."""
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        target = str(PurePosixPath(target_path))
        if target in self.paths:
            with self._open_dir(target, create=False) as inner:
                _write_file(source, inner, source.name, target)
            return
        parent, name = self._parent(target)
        with self._open_dir(parent, create=True) as dfd:
            try:
                st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except FileNotFoundError:
                st = None
            except OSError as e:
                raise _refused(e, target) from None
            if st is not None and stat.S_ISDIR(st.st_mode):
                inner = _open_made(dfd, name, target)
                try:
                    _write_file(source, inner, source.name, target)
                finally:
                    os.close(inner)
                return
            _write_file(source, dfd, name, target)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        """Copy the contents of the host directory *source_dir* into *target_dir*, which is made
        if missing. Files already there are replaced; symlinks in the source are followed, as
        ``cp -r`` of a harness tree does."""
        source = Path(source_dir)
        if not source.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source}")
        with self._open_dir(target_dir, create=True) as dfd:
            _upload_tree(source, dfd, target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        """Copy *source_path* to the host file *target_path*, making its parents, as long as the
        file was when it was opened. A source that is a symlink or not a regular file is refused,
        as is one the sandbox's modes refuse, with a ValueError. It has no size bound: a file the
        agent wrote must not be read through it unbounded."""
        parent, name = self._parent(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self._open_dir(parent, create=False) as dfd:
                _read_file(dfd, name, target, source_path)
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to download file: {source_path} does not exist") from e

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        """Copy the regular files and directories under *source_dir* into the host directory
        *target_dir*, as harbor's backends do: each directory there replaced whole, each file
        replaced. The walk keeps its own stack and neither follows nor makes a link; each entry it
        leaves out is recorded in ``skipped`` with why (``_download_tree``: links, special files,
        what it cannot read, and what lies past its bounds). Only a failed write on the host
        raises. The bounds keep the walk from failing on what the agent left, not from filling
        the host's disk: a tree the agent wrote must still not be read through it unbounded."""
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        try:
            with self._open_dir(source_dir, create=False) as dfd:
                _download_tree(dfd, target, str(PurePosixPath(source_dir)), self.skipped)
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to download directory: {source_dir} does not exist") from e


def _named(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _refused(error: OSError, shown: str) -> Exception:
    """What a transfer raises for *error* at *shown*: a ValueError naming it when what the sandbox
    holds there explains it, else *error* itself, the machine's failure."""
    if error.errno in _SANDBOX_ERRNOS:
        return ValueError(f"{shown}: {errno.errorcode[error.errno]} ({os.strerror(error.errno)})")
    return error


def _open_subdir(dfd: int, name: str, shown: str) -> int:
    """Open the directory *name* in the directory *dfd* without following a symlink. Raise
    FileNotFoundError for nothing there, ValueError for a symlink, a file or anything else the
    sandbox holds there that refuses the open, and any other OSError as it is."""
    try:
        return os.open(name, _DIR_FLAGS, dir_fd=dfd)
    except FileNotFoundError:
        raise FileNotFoundError(errno.ENOENT, f"{shown}: no such directory in the sandbox") from None
    except OSError as e:
        if e.errno in _NOT_A_DIR:
            raise ValueError(f"{shown}: {name} is a symlink or not a directory") from None
        raise _refused(e, shown) from None


def _open_made(dfd: int, name: str, shown: str) -> int:
    """``_open_subdir`` for a directory just made or seen: gone since, it is a ValueError too."""
    try:
        return _open_subdir(dfd, name, shown)
    except FileNotFoundError as e:
        raise _refused(e, shown) from None


def _make_subdir(dfd: int, name: str, shown: str) -> None:
    """Make the directory *name* in the directory *dfd*, unless something is there already."""
    try:
        os.mkdir(name, 0o755, dir_fd=dfd)
    except FileExistsError:
        pass
    except OSError as e:
        raise _refused(e, shown) from None


def _write_file(source: Path, dfd: int, name: str, shown: str) -> None:
    """Write the host file *source* as *name* in the directory *dfd*, with its mode and
    modification time. Refuse a target that is a symlink or anything but a regular file."""
    st = source.stat()
    mode = stat.S_IMODE(st.st_mode)
    try:
        out = os.open(name, _WRITE_FLAGS, mode, dir_fd=dfd)
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ValueError(f"{shown}: the target is a symlink") from None
        if e.errno in (errno.ENXIO, errno.EISDIR):  # a pipe without a reader, a socket, a directory
            raise ValueError(f"{shown}: the target is not a regular file") from None
        raise _refused(e, shown) from None
    try:
        if not stat.S_ISREG(os.fstat(out).st_mode):
            raise ValueError(f"{shown}: the target is not a regular file")
        os.set_blocking(out, True)
        os.ftruncate(out, 0)
        with source.open("rb") as src:
            while chunk := src.read(1 << 20):
                view = memoryview(chunk)
                while view:
                    view = view[os.write(out, view):]
        os.fchmod(out, mode)
        os.utime(out, ns=(st.st_atime_ns, st.st_mtime_ns))
    finally:
        os.close(out)


def _upload_tree(source: Path, dfd: int, shown: str) -> None:
    for entry in sorted(source.iterdir()):
        where = f"{shown}/{entry.name}"
        if entry.is_dir():
            _make_subdir(dfd, entry.name, where)
            sub = _open_made(dfd, entry.name, where)
            try:
                _upload_tree(entry, sub, where)
            finally:
                os.close(sub)
        elif entry.is_file():
            _write_file(entry, dfd, entry.name, where)


class _ReadFailed(Exception):
    """Reading a file in the sandbox failed partway; what was copied of it is removed."""

    def __init__(self, error: OSError) -> None:
        super().__init__(str(error))
        #: The read's error.
        self.error = error


def _copy_out(src: int, st: os.stat_result, target: Path) -> None:
    """Copy the open regular file *src*, as long as *st* (its status when opened) says, to the
    host file *target*, with its mode and modification time, replacing a link there rather than
    writing through it. A failed read raises ``_ReadFailed``; a failed write, the host's, raises
    as it is."""
    os.set_blocking(src, True)
    if target.is_symlink():
        target.unlink()
    failed: OSError | None = None
    with target.open("wb") as out:
        left = st.st_size
        while left:
            try:
                chunk = os.read(src, min(left, 1 << 20))
            except OSError as e:
                failed = e
                break
            if not chunk:  # it shrank since it was opened
                break
            out.write(chunk)
            left -= len(chunk)
    if failed is not None:
        target.unlink(missing_ok=True)
        raise _ReadFailed(failed)
    os.chmod(target, stat.S_IMODE(st.st_mode))
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))


def _read_file(dfd: int, name: str, target: Path, shown: str) -> None:
    """Copy *name* in the directory *dfd* to the host file *target* (``_copy_out``). Refuse a
    symlink, anything but a regular file and what the sandbox's modes refuse with a ValueError;
    nothing there is a FileNotFoundError."""
    try:
        src = os.open(name, _READ_FLAGS, dir_fd=dfd)
    except FileNotFoundError:
        raise
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise ValueError(f"{shown}: a symlink, not a file") from None
        raise _refused(e, shown) from None
    try:
        st = os.fstat(src)
        if not stat.S_ISREG(st.st_mode):
            raise ValueError(f"{shown}: not a regular file")
        try:
            _copy_out(src, st, target)
        except _ReadFailed as e:
            raise e.error from None
    finally:
        os.close(src)


def _why(error: OSError) -> str:
    """Why a download left out an entry it could not open, list or read."""
    if error.errno == errno.ELOOP:
        return "symlink"
    return f"unreadable: {errno.errorcode.get(error.errno or 0, 'unknown error')}"


def _download_tree(top: int, target: Path, shown: str, skipped: list[tuple[str, str]]) -> None:
    """Copy the tree open at *top* (*shown* in the sandbox) into the host directory *target*,
    depth first in name order, keeping its own stack rather than recursing. Each directory
    directly in *top* replaces the host's whole; each file replaces the host's. Every entry it
    leaves out is appended to *skipped* with why: a link (neither followed nor made), a special
    file, one it cannot open, list or read, a directory more than ``transcript.MAX_DEPTH`` levels
    down, a path longer than ``transcript.MAX_PATH_BYTES``, a file that would take the copy past
    ``transcript.MAX_BYTES``, and everything after the first ``transcript.MAX_ENTRIES`` entries.
    Only a failed write on the host raises."""
    copied = seen = 0
    # The open directories from the top down to the one being copied, each with its path below
    # the top, its depth and the names it still holds, last name first. Every descriptor the walk
    # opened is closed on the way out; *top* is the caller's.
    frames = [(top, "", 0, _listing(top, shown, skipped))]
    try:
        while frames:
            fd, rel, depth, names = frames[-1]
            if not names:
                frames.pop()
                if fd != top:
                    os.close(fd)
                continue
            name = names.pop()
            path = f"{rel}/{name}" if rel else name
            where = f"{shown}/{path}"
            seen += 1
            if seen > transcript.MAX_ENTRIES:
                skipped.append((where, f"past the first {transcript.MAX_ENTRIES} entries: the rest is not "
                                       "read"))
                return
            if len(os.fsencode(path)) > transcript.MAX_PATH_BYTES:
                skipped.append((where, f"its path is longer than {transcript.MAX_PATH_BYTES} bytes"))
                continue
            opened = _open_entry(fd, name, depth, where, skipped)
            if opened is None:
                continue
            handle, st = opened
            dest = target / path
            if stat.S_ISDIR(st.st_mode):
                frames.append((handle, path, depth + 1, _listing(handle, where, skipped)))
                if depth == 0:  # what lies below is then made afresh
                    remove_tree(dest)
                dest.mkdir()
                continue
            try:
                if copied + st.st_size > transcript.MAX_BYTES:
                    skipped.append((where, f"its {st.st_size} bytes would take the download past "
                                           f"{transcript.MAX_BYTES} bytes"))
                    continue
                if dest.is_dir() and not dest.is_symlink():
                    remove_tree(dest)
                try:
                    _copy_out(handle, st, dest)
                except _ReadFailed as e:
                    skipped.append((where, _why(e.error)))
                    continue
                copied += st.st_size
            finally:
                os.close(handle)
    finally:
        for fd, *_ in frames:
            if fd != top:
                os.close(fd)


def _listing(fd: int, where: str, skipped: list[tuple[str, str]]) -> list[str]:
    """The names in the directory open at *fd*, last name first; none, with the directory
    recorded in *skipped*, when it cannot be listed."""
    try:
        with os.scandir(fd) as entries:
            return sorted((entry.name for entry in entries), reverse=True)
    except OSError as e:
        skipped.append((where, _why(e)))
        return []


def _open_entry(dfd: int, name: str, depth: int, where: str,
                skipped: list[tuple[str, str]]) -> tuple[int, os.stat_result] | None:
    """Open the entry *name* of the directory *dfd*, *depth* levels below the top, without
    following a link or blocking, and return its descriptor and status; None, with the entry
    recorded in *skipped*, for one the download leaves out. The type is read before the open,
    since opening a socket fails outright, and again after it, so an entry swapped since is left
    out too."""
    try:
        before = os.stat(name, dir_fd=dfd, follow_symlinks=False)
    except OSError as e:
        skipped.append((where, _why(e)))
        return None
    if stat.S_ISLNK(before.st_mode):
        skipped.append((where, "symlink"))
        return None
    is_dir = stat.S_ISDIR(before.st_mode)
    if not is_dir and not stat.S_ISREG(before.st_mode):
        skipped.append((where, "not a regular file"))
        return None
    if is_dir and depth + 1 > transcript.MAX_DEPTH:
        skipped.append((where, f"deeper than {transcript.MAX_DEPTH} levels"))
        return None
    try:
        handle = os.open(name, _DIR_FLAGS if is_dir else _READ_FLAGS, dir_fd=dfd)
    except OSError as e:
        skipped.append((where, _why(e)))
        return None
    try:
        st = os.fstat(handle)
    except BaseException:
        os.close(handle)
        raise
    if stat.S_ISDIR(st.st_mode) != is_dir or not (is_dir or stat.S_ISREG(st.st_mode)):
        os.close(handle)
        skipped.append((where, "not a regular file"))
        return None
    return handle, st
