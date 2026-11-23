"""The build sandbox's exec server: the one harness process in the container.

It runs as ``/usr/bin/python3 -I exec_server.py --socket PATH --spawner FILE --shm-bytes B --uid U
--gid G [--hide PATH]... [--command-nproc N] [--ucount NAME=LIMIT]...`` (``exec_protocol.server_command``)
in the container's
own PID namespace,
and serves harbor's exec API (``POST /exec``, ``GET /health``) over a UNIX socket in the control
directory, which the runner reaches through a bind and no contestant command can see. It runs
nothing itself: every command goes to the spawner, which serves from outside the commands' PID
namespace and forks each one into it. A contestant cannot see, signal or trace the server or the
spawner: its own PID namespace holds only PID 1, a bare reaper, and the sibling commands. Ending
the reaper kills only the contestant's own commands; the ``SpawnerPool`` then serves the next
command from a fresh namespace, so a contestant cannot turn a result into a retried fault.

At start it takes the token from ``HLS_EXEC_TOKEN`` and removes it from its environment; without
one it exits 2. A request without the token gets 403 and runs nothing. It lowers its own
``oom_score_adj`` to ``HARNESS_OOM_SCORE_ADJ`` (the step wrapper raised the whole step to 1000),
so the spawner it starts inherits that and only commands start at 1000 (a command may lower
its own to 0: whose fault an out-of-memory end is rests on the memory split, not the kernel's
pick, ``exec_protocol.OOM_SCORE_ADJ``). It then starts the spawner
with an empty environment and binds its socket (mode 0600) only once the spawner says it is
ready, so a health check that answers means commands can run. The socket appears at its path
only once it listens (``bind_server``). Each spawner it starts is named, before it serves its
first command, in the ready file beside the socket (``exec_protocol.READY_NAME``, ``ReadyFile``),
from which the host's memory split finds the contestant's PID namespace (``SpawnerPool``). With
``--command-nproc N`` the spawner gives every command the process ceiling N (``RLIMIT_NPROC``,
soft and hard), while the server and the spawner keep the limit they were started with: a
contestant's fork past the ceiling fails inside its command, and the harness can still fork.
Each ``--ucount NAME=LIMIT`` goes to every spawner it starts, whose reaper gives the commands'
user namespace that limit in ``/proc/sys/user/NAME`` (``hls.substrate.ucounts``).

A fixed set of threads (``HANDLER_THREADS``), started before the socket is bound, serves the
requests; none is started per request, so a sandbox that has run out of processes cannot keep the
server from answering. Each connection carries one request (every reply says ``Connection:
close``), so a thread is held only while it serves one; a request waits for a free thread. A
connection the server accepted is always answered, unless its own machinery has failed: a command
it could not hand to the spawner comes back as one that could not start (``NOT_STARTED_STATUS``,
126, with why as its output), as a command the spawner cannot fork does.

Each exec behaves as harbor's singularity server's does:
- the command's environment is the server's own (the container's env file, the token removed),
  with ``PATH`` prefixed by ``/usr/bin:/usr/local/bin:``, then updated with the request's env;
- with ``timeout_sec`` it runs as ``timeout --signal=TERM <n> bash -c <command>`` (status 124 on
  a timeout), else as ``bash -c <command>``;
- stdout and stderr are combined; the reply waits for every process holding the output open to
  close it (a command that leaves a background job writing to it waits for that job), and
  carries the output stripped, or null when there was none, with ``stderr`` null and
  ``return_code`` the shell's status, or minus the number of the signal that ended it.
- Unlike harbor's, it keeps only the last ``OUTPUT_TAIL_BYTES`` of the output, read in chunks,
  since its memory counts as the harness's in the build's OOM rule. When it dropped bytes, the
  output starts with one line saying how many (``exec_protocol.dropped_marker``); the tail is
  decoded with replacement, since the cut can split a character.
A command whose working directory cannot be entered fails with status 127 and the reason as its
output.

Every command leaves two lines on the server's stderr, which the step writes to the role's log
on the host, out of the contestant's reach: ``hls-exec: start id=<n> pid=<p> cli=<0 or 1> cwd=<cwd>
cmd=<first 120 characters>``, once the command has started, where ``cli`` says whether the whole
command launches the agent's CLI (``exec_protocol.is_cli_launch``; ``exec_protocol.parse_start_line``
reads it back), and ``hls-exec: end id=<n> status=<code> signal=<NAME or -> dropped=<bytes>
sec=<s>``, its status the return code the caller got (minus the signal number for a command a
signal ended) and its dropped bytes the output not kept. They are the evidence of which commands
ran and how each ended. A line the log cannot take (its disk full) is lost, never the request:
the next line written follows one saying how many were lost. If the spawner's machinery fails,
every command it ran has ended with it, and the server exits 70 before it answers any command
still in flight; a reaper a contestant ended is recovered from instead, and only then are those
commands answered as killed."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import queue
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler
from pathlib import Path

if __package__:
    from hls.agent import exec_protocol as protocol
else:
    # Run as a file in the sandbox, where hls is not installed: the protocol file sits beside
    # this one in the same read-only directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import exec_protocol as protocol

#: Seconds the spawner has to say it is ready.
SPAWNER_READY_SEC = 60.0
#: Seconds a request whose send to the spawner failed waits for the channel to show closed. A
#: spawner that ended closes it at once; a channel still open after this failed some other way.
CHANNEL_CLOSE_SEC = 5.0
#: Threads that serve requests, all started with the server, so none is started per request and a
#: sandbox out of processes cannot keep the server from answering. A request waits for a free one.
#: The runner sends its execs one at a time: the build's probes and preflight await each before
#: the next, and harbor's agent runs its setup, its CLI and each re-entry as one exec after
#: another, the CLI's own tool calls running inside it. An exec its client gave up on (a cancelled
#: run, a timeout) holds its thread until its command ends, so a few can be held at once; 16
#: leaves room for those and for a harness that sends some at once, for 16 of the sandbox's
#: tasks, a small part of any pids ceiling.
HANDLER_THREADS = 16
#: The status of a command the server could not hand to the spawner, as the spawner answers a
#: command it cannot fork.
NOT_STARTED_STATUS = 126


class _Log:
    """The server's stderr, the role's log on the host. A write that fails (its disk full) loses
    the line, never the request that logged it: lost lines are counted, and the next line written
    comes after one saying how many, on a line of its own even when a failed write cut one short."""

    def __init__(self, fd: int) -> None:
        #: The descriptor written.
        self.fd = fd
        #: Lines lost since the last one written.
        self.lost = 0
        #: Whether the log ends inside a line that a failed write cut short.
        self.cut = False
        self._lock = threading.Lock()

    def write(self, line: str) -> None:
        with self._lock:
            if self.lost:
                note = f"{protocol.LOG_PREFIX}{self.lost} earlier log lines could not be written\n".encode()
                if not self._put((b"\n" if self.cut else b"") + note):
                    self.lost += 1
                    return
                self.lost = 0
            if not self._put((protocol.LOG_PREFIX + line + "\n").encode(errors="replace")):
                self.lost += 1

    def _put(self, data: bytes) -> bool:
        """Write *data* whole; False once a write fails, leaving what was written."""
        view = memoryview(data)
        try:
            while view:
                written = os.write(self.fd, view)
                if written:
                    self.cut = view[written - 1] != ord("\n")
                view = view[written:]
        except OSError:
            return False
        return True


_LOG = _Log(2)


def log(line: str) -> None:
    """Write *line* to the role's log; a failed write is counted, never raised (``_Log``)."""
    _LOG.write(line)


def lower_own_score(path: str = protocol.OWN_OOM_SCORE) -> str:
    """Lower this process's ``oom_score_adj`` to ``HARNESS_OOM_SCORE_ADJ`` when the step wrapper
    raised it above that; never raise it, and leave a lower one as it is. Return the score now
    held as *path* reads, or ``unknown``. A write the kernel refuses is logged and the server goes
    on with the score it has: the build's OOM rule rests on the harness's measured memory, not on
    its score."""
    try:
        with open(path) as f:
            now = int(f.read().strip())
        if now > protocol.HARNESS_OOM_SCORE_ADJ:
            with open(path, "w") as f:
                f.write(str(protocol.HARNESS_OOM_SCORE_ADJ))
        with open(path) as f:
            return f.read().strip()
    except (OSError, ValueError) as e:
        log(f"cannot lower the oom score ({e})")
        try:
            with open(path) as f:
                return f.read().strip()
        except OSError:
            return "unknown"


def _signal_name(num: int | None) -> str:
    if num is None:
        return "-"
    try:
        return signal.Signals(num).name
    except ValueError:
        return f"SIG{num}"


class _Pending:
    """One command the spawner was asked to run."""

    def __init__(self) -> None:
        #: Set once the spawner reported the command started (or could not start it).
        self.started = threading.Event()
        #: Set once the spawner reported how the command ended.
        self.ended = threading.Event()
        #: The command's pid in the spawner's namespace.
        self.pid: int | None = None
        #: Its exit status; None when a signal ended it.
        self.status: int | None = None
        #: The signal that ended it.
        self.signal: int | None = None
        #: True when the spawner could not start it because its namespace had ended (its reaper
        #: died): it never ran, and goes to a fresh spawner.
        self.gone = False


class ChannelGone(ConnectionError):
    """A request reached a spawner that had already ended, so it was never sent and never ran."""


class SpawnerChannel:
    """The server's side of the socketpair: sends spawn requests and routes the replies."""

    def __init__(self, sock: socket.socket, on_gone) -> None:
        #: The socket.
        self.sock = sock
        #: Called once with a reason when the spawner has gone; it ends the server unless the
        #: spawner's reaper is what ended. Commands in flight are answered only after it returns.
        self.on_gone = on_gone
        #: Set once the spawner sent ``ready``.
        self.ready = threading.Event()
        #: Set once the channel has closed, before ``on_gone`` runs, so the pool starts a fresh
        #: spawner rather than hand back this one while its end is still settling.
        self.closed = threading.Event()
        #: The ``ready`` message.
        self.hello: dict = {}
        self._send = threading.Lock()
        self._pending: dict[int, _Pending] = {}
        self._guard = threading.Lock()
        self._next = 0

    def start(self) -> None:
        threading.Thread(target=self._read, name="hls-exec-spawner", daemon=True).start()

    def _read(self) -> None:
        reason = "it closed the channel"
        try:
            while (received := protocol.recv_message(self.sock)) is not None:
                message, _ = received
                if message.get("op") == "ready":
                    self.hello = message
                    self.ready.set()
                    continue
                with self._guard:
                    pending = self._pending.get(message.get("id"))
                if pending is None:
                    continue
                if message["op"] == "started":
                    pending.pid = message["pid"]
                    pending.started.set()
                elif message["op"] == "gone":
                    pending.gone = True
                    pending.started.set()
                    pending.ended.set()
                elif message["op"] == "exit":
                    pending.status, pending.signal = message["status"], message["signal"]
                    pending.started.set()
                    pending.ended.set()
        except (OSError, ConnectionError, ValueError) as e:
            reason = f"the channel failed: {e}"
        self.closed.set()
        # How the spawner ended decides first: a machinery failure ends the server here, so no
        # command in flight is ever answered as killed, which would read as the agent's end.
        self.on_gone(reason)
        self._fail_pending()

    def _fail_pending(self) -> None:
        """End every command still in flight as killed by SIGKILL, so a request never hangs on a
        channel that closed before its command's exit arrived. It runs only once ``on_gone`` has
        returned, that is after a reaper's end, which kills the namespace's commands (the spawner
        reports those it ran, but not a spawn it never read)."""
        with self._guard:
            pending = list(self._pending.values())
        for p in pending:
            if not p.ended.is_set():
                p.status, p.signal = None, int(signal.SIGKILL)
                p.started.set()
                p.ended.set()

    def run(self, argv: list[str], cwd: str | None, env: dict[str, str]) -> tuple[int, _Pending, int]:
        """Ask the spawner to run *argv*. Returns the request id, its pending record and the
        read end of its output pipe, which the caller reads to the end and closes, then releases
        the id with ``forget``. A request that cannot be sent leaves no record; one whose send
        failed because the spawner had ended raises ``ChannelGone``."""
        r, w = os.pipe()
        with self._guard:
            self._next += 1
            ident = self._next
            pending = self._pending[ident] = _Pending()
        try:
            with self._send:
                protocol.send_message(self.sock, {"op": "spawn", "id": ident, "argv": argv, "cwd": cwd, "env": env},
                                      fds=(w,))
        except BaseException as e:
            self.forget(ident)
            os.close(r)
            if isinstance(e, OSError) and self.closed.wait(CHANNEL_CLOSE_SEC):
                raise ChannelGone(f"the spawner had ended ({e})") from e
            raise
        finally:
            os.close(w)
        return ident, pending, r

    def forget(self, ident: int) -> None:
        with self._guard:
            self._pending.pop(ident, None)


def _write_at(fd: int, data: bytes) -> None:
    """Write *data* whole at the start of *fd*."""
    view, at = memoryview(data), 0
    while at < len(view):
        at += os.pwrite(fd, view[at:], at)


def _score_or_none(score: str) -> int | None:
    """The server's ``oom_score_adj`` as ``lower_own_score`` read it; None when it could not."""
    try:
        return int(score)
    except ValueError:
        return None


class ReadyFile:
    """The ready file, which names the spawner the server serves from. Its ``READY_BYTES`` are
    allocated when it is made, at the server's start, before any command runs: reserved, written
    and synced, so no block of it is left to allocate. Each record then overwrites them in place
    (``write``), so writing one needs no new block of the node's disk: a contestant who fills
    that disk and ends its reaper cannot make the next record fail. A reader of a record caught
    mid-write finds its checksum wrong and reads no pair (``exec_protocol.parse_ready_record``).
    Raise OSError when it cannot be made."""

    def __init__(self, path: str) -> None:
        #: Where it is.
        self.path = path
        self._fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.posix_fallocate(self._fd, 0, protocol.READY_BYTES)
            _write_at(self._fd, b"\0" * protocol.READY_BYTES)
            os.fsync(self._fd)
        except BaseException:
            os.close(self._fd)
            raise

    def write(self, *, spawner_pid: int, reaper_pid: int, score: int | None) -> None:
        """Overwrite the record with one naming this spawner and its reaper. Raise OSError when
        it cannot be written."""
        _write_at(self._fd, protocol.ready_record(spawner_pid=spawner_pid, reaper_pid=reaper_pid, score=score))


class SpawnerPool:
    """One live spawner at a time, restarted on demand. A command that ends the reaper (PID 1 of
    its namespace) kills only the contestant's own commands; the spawner then exits
    ``REAPER_GONE_EXIT``, and this starts a fresh spawner, with a fresh namespace, when the next
    command asks for one. A spawner that fails any other way is the harness's own machinery
    failing, which ends the server with ``SPAWNER_GONE_EXIT`` (infra), as a lost sandbox is.

    Each spawner it starts is named in *ready_file* (``ReadyFile.write``) before ``ready`` hands
    its channel to any command, and a record that cannot be written ends the server like a spawner
    that will not start: the host must never read an old pair while a fresh namespace runs
    commands (``hls.agent.memory.find_reaper``)."""

    def __init__(self, *, spawner_file: str, shm_bytes: int, uid: int, gid: int, hide: list[str],
                 no_namespaces: bool, score: str, ready_file: ReadyFile,
                 command_nproc: int | None = None, ucounts: list[str] | None = None) -> None:
        self._file = spawner_file
        self._ready_file = ready_file
        self._shm_bytes = shm_bytes
        self._uid, self._gid = uid, gid
        self._hide = hide
        self._nproc = [] if command_nproc is None else ["--command-nproc", str(command_nproc)]
        self._ucounts = [x for limit in ucounts or [] for x in ("--ucount", limit)]
        self._mode = ["--no-namespaces"] if no_namespaces else []
        self._score = score
        self._lock = threading.Lock()
        self._channel: SpawnerChannel | None = None
        self._proc: subprocess.Popen | None = None

    def ready(self) -> SpawnerChannel:
        """The live spawner's channel, started if there is none. Ends the server
        (``SPAWNER_GONE_EXIT``) if a fresh spawner will not come up: its machinery has failed."""
        with self._lock:
            if self._channel is not None and not self._channel.closed.is_set():
                return self._channel
            try:
                self._start()
            except (OSError, RuntimeError) as e:
                log(f"the spawner would not start ({e}); ending the server")
                os._exit(protocol.SPAWNER_GONE_EXIT)
            return self._channel

    def _start(self) -> None:
        ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        cmd = [sys.executable, "-I", self._file, "--fd", str(theirs.fileno()), "--shm-bytes", str(self._shm_bytes),
               "--uid", str(self._uid), "--gid", str(self._gid)]
        cmd += [x for path in self._hide for x in ("--hide", path)] + self._nproc + self._ucounts + self._mode
        proc = subprocess.Popen(cmd, pass_fds=(theirs.fileno(),), env={})
        theirs.close()
        channel = SpawnerChannel(ours, lambda reason: self._gone(proc, reason))
        channel.start()
        if not channel.ready.wait(SPAWNER_READY_SEC):
            proc.kill()
            raise RuntimeError(f"spawner not ready after {SPAWNER_READY_SEC:.0f} s")
        line = (f"ready ns={channel.hello.get('pid_ns')} spawner_pid={channel.hello.get('pid')} "
                f"reaper_pid={channel.hello.get('reaper_pid')} oom_score_adj={self._score}")
        log(line)
        # Named on disk before the pool hands the channel out (``ready`` holds its lock until then),
        # so no command runs in the fresh namespace while the host still reads the old pair.
        try:
            self._ready_file.write(spawner_pid=int(channel.hello["pid"]), reaper_pid=int(channel.hello["reaper_pid"]),
                                   score=_score_or_none(self._score))
        except OSError:
            proc.kill()
            raise
        except (KeyError, TypeError, ValueError) as e:
            proc.kill()
            raise RuntimeError(f"the spawner's hello does not name its pids ({e!r})") from e
        self._channel, self._proc = channel, proc

    def _gone(self, proc: subprocess.Popen, reason: str) -> None:
        try:
            status = proc.wait(5)
        except subprocess.TimeoutExpired:
            status = None
        if status == protocol.REAPER_GONE_EXIT:
            # The lock is taken only here, never on the machinery path below, so a spawner that
            # dies while a start still holds the lock ends the server at once rather than blocking.
            with self._lock:
                if self._proc is proc:
                    self._channel, self._proc = None, None
            log(f"the sandbox's reaper ended ({reason}); the next command runs in a fresh namespace")
            return
        log(f"spawner ended ({reason}) status={status}; every command it ran has ended with it")
        os._exit(protocol.SPAWNER_GONE_EXIT)

    def run(self, argv: list[str], cwd: str | None, env: dict[str, str]) -> tuple[SpawnerChannel, int, _Pending, int]:
        """Run *argv* on the live spawner. Returns the channel that runs it and what its ``run``
        returns; the caller releases the id on that channel, since ids restart on each spawner.
        Requests run in parallel, so one can reach a spawner that ended (its reaper died) before
        the pool noticed; that request never ran, and goes once to a fresh spawner. A fresh
        spawner that has ended too is the server's own failure."""
        channel = self.ready()
        try:
            return (channel, *channel.run(argv, cwd, env))
        except ChannelGone:
            log("a command reached a spawner that had ended; it goes to a fresh one")
        channel = self.ready()
        return (channel, *channel.run(argv, cwd, env))


class ExecServer(socketserver.UnixStreamServer):
    """The HTTP server on the control socket. The listening thread only accepts each connection
    and queues it; *threads* threads, started before the socket is bound, serve them in turn."""

    def __init__(self, path: str, *, token: str, spawner: SpawnerPool, cli_launches: tuple[str, ...],
                 threads: int = HANDLER_THREADS) -> None:
        #: The token every request must carry.
        self.token = token
        #: The text a command that launches the agent's CLI holds (``--cli-launch``), which its
        #: start line reports (``exec_protocol.is_cli_launch``).
        self.cli_launches = cli_launches
        #: The pool that keeps one spawner live, restarting it after a reaper dies.
        self.spawner = spawner
        #: Accepted connections, each with its address, waiting for a thread; None ends a thread.
        self.waiting: queue.SimpleQueue = queue.SimpleQueue()
        #: The threads that serve requests.
        self.threads = [
            threading.Thread(target=self._serve_requests, name=f"hls-exec-request-{i}", daemon=True)
            for i in range(threads)]
        for thread in self.threads:
            thread.start()
        super().__init__(path, Handler)

    def process_request(self, request, client_address) -> None:
        self.waiting.put((request, client_address))

    def _serve_requests(self) -> None:
        while (waiting := self.waiting.get()) is not None:
            request, address = waiting
            try:
                self.finish_request(request, address)
            except Exception as e:  # noqa: BLE001 -- a thread that ended would serve no later request
                log(f"a request ended without a reply: {type(e).__name__}: {e}")
            finally:
                self.shutdown_request(request)

    def handle_error(self, request, client_address) -> None:
        """The listening thread failed to queue a connection: logged, never printed, since a
        print the log refuses would end the listening loop."""
        log(f"a connection failed: {sys.exc_info()[1]!r}")

    def server_close(self) -> None:
        super().server_close()
        for _ in self.threads:
            self.waiting.put(None)


class Handler(BaseHTTPRequestHandler):
    """One connection to the control socket."""

    protocol_version = "HTTP/1.1"
    server: ExecServer

    def address_string(self) -> str:
        return "control"

    def log_message(self, format: str, *args) -> None:
        return

    def _reply(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        # One request per connection, so a thread is held only while it serves one, never by a
        # client's idle connection.
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        given = self.headers.get(protocol.TOKEN_HEADER, "")
        if hmac.compare_digest(given.encode(), self.server.token.encode()):
            return True
        # Read and drop the body, so the connection stays usable for the reply.
        self.rfile.read(int(self.headers.get("Content-Length") or 0))
        self._reply(403, {"detail": "forbidden"})
        return False

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path != "/health":
            self._reply(404, {"detail": "not found"})
            return
        # Answering means commands can run, so confirm a live spawner (starting one if a reaper
        # death left none).
        self.server.spawner.ready()
        self._reply(200, {"status": "ok"})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path != "/exec":
            self._reply(404, {"detail": "not found"})
            return
        try:
            command, cwd, env, timeout_sec = parse_request(body)
        except ValueError as e:
            self._reply(400, {"detail": f"bad request: {e}"})
            return
        self._reply(200, answer(self.server.spawner, command, cwd, env, timeout_sec, self.server.cli_launches))


def parse_request(body: bytes) -> tuple[str, str | None, dict[str, str] | None, int | None]:
    """The command, working directory, environment and timeout of an exec request, checked as
    harbor's ``CommandRequest`` checks them; ValueError names what is wrong."""
    req = json.loads(body)
    if not isinstance(req, dict) or not isinstance(req.get("command"), str):
        raise ValueError("the body must be an object with a string command")
    cwd, env, timeout_sec = req.get("cwd"), req.get("env"), req.get("timeout_sec")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("cwd must be a string")
    if env is not None and not (isinstance(env, dict) and all(isinstance(k, str) and isinstance(v, str)
                                                                for k, v in env.items())):
        raise ValueError("env must map strings to strings")
    if timeout_sec is not None and (isinstance(timeout_sec, bool) or not isinstance(timeout_sec, int)):
        raise ValueError("timeout_sec must be an integer")
    return req["command"], cwd, env, timeout_sec


def command_env(request_env: dict[str, str] | None) -> dict[str, str]:
    """A command's environment: the server's own, PATH prefixed, then the request's."""
    env = dict(os.environ)
    env["PATH"] = protocol.PATH_PREFIX + env.get("PATH", "/bin")
    env.update(request_env or {})
    return env


def read_tail(fd: int, *, cap: int = protocol.OUTPUT_TAIL_BYTES,
              chunk: int = protocol.OUTPUT_CHUNK_BYTES) -> tuple[bytes, int]:
    """Read *fd* to its end in chunks and return its last *cap* bytes and how many bytes came
    before them. At most *cap* plus one chunk is held at any time."""
    tail, dropped = bytearray(), 0
    while data := os.read(fd, chunk):
        tail += data
        if len(tail) > cap:
            excess = len(tail) - cap
            del tail[:excess]
            dropped += excess
    return bytes(tail), dropped


def answer(spawner: SpawnerPool, command: str, cwd: str | None, env: dict[str, str] | None,
           timeout_sec: int | None, cli_launches: tuple[str, ...]) -> dict:
    """The reply to an exec: ``execute``'s result, or, when the server could not hand the command
    to the spawner (out of open files, a request over the channel's size), a command that could
    not start, with why, never a dropped connection. A command that reached a fresh spawner that
    had ended too is the server's own machinery failing, and is not answered as a command's end."""
    try:
        return execute(spawner, command, cwd, env, timeout_sec, cli_launches)
    except ChannelGone:
        raise
    except Exception as e:  # noqa: BLE001 -- anything else that kept the command from the spawner
        reason = f"{type(e).__name__}: {e}"
        log(f"cannot serve a command ({reason}); answered {NOT_STARTED_STATUS}")
        return {"stdout": f"{protocol.LOG_PREFIX}cannot serve the command: {reason}", "stderr": None,
                "return_code": NOT_STARTED_STATUS}


def execute(spawner: SpawnerPool, command: str, cwd: str | None, env: dict[str, str] | None,
            timeout_sec: int | None, cli_launches: tuple[str, ...]) -> dict:
    """Run one command through the spawner and return harbor's result; its start line says
    whether it holds one of *cli_launches*."""
    script = command if timeout_sec is None else (
        f"timeout --signal=TERM {int(timeout_sec)} bash -c {_shell_quote(command)}")
    shown = command[:protocol.COMMAND_LOG_CHARS].replace("\n", "\\n")
    began = time.monotonic()
    channel, ident, pending, r = spawner.run(["/bin/bash", "-c", script], cwd, command_env(env))
    pending.started.wait()
    if pending.gone:
        # Its namespace's PID 1 had ended: once that spawner has gone, a fresh one runs it.
        os.close(r)
        channel.forget(ident)
        log("a command reached a namespace whose PID 1 had ended; it goes to a fresh one")
        channel.closed.wait(CHANNEL_CLOSE_SEC)
        channel, ident, pending, r = spawner.run(["/bin/bash", "-c", script], cwd, command_env(env))
        pending.started.wait()
        if pending.gone:
            os.close(r)
            channel.forget(ident)
            log(f"a command's fresh namespace ended before it started too; answered {NOT_STARTED_STATUS}")
            return {"stdout": f"{protocol.LOG_PREFIX}the command's namespace ended before it started", "stderr": None,
                    "return_code": NOT_STARTED_STATUS}
    try:
        pending.started.wait()
        cli = int(protocol.is_cli_launch(command, cli_launches))
        log(f"start id={ident} pid={pending.pid} cli={cli} cwd={cwd} cmd={shown}")
        data, dropped = read_tail(r)
        pending.ended.wait()
    finally:
        os.close(r)
        channel.forget(ident)
    code = -pending.signal if pending.signal is not None else pending.status
    log(f"end id={ident} status={code} signal={_signal_name(pending.signal)} dropped={dropped} "
        f"sec={time.monotonic() - began:.3f}")
    text = data.decode("utf-8", errors="replace")
    if dropped:
        text = protocol.dropped_marker(dropped) + "\n" + text
    return {"stdout": text.strip() if text else None, "stderr": None, "return_code": code}


def _shell_quote(s: str) -> str:
    """Quote *s* for bash as harbor's server does."""
    return "'" + s.replace("'", "'\"'\"'") + "'"


def bind_server(path: Path, *, token: str, spawner: SpawnerPool, cli_launches: tuple[str, ...],
                threads: int = HANDLER_THREADS) -> ExecServer:
    """The server, listening on a socket at *path* with mode 0600 and served by *threads* threads.
    It is bound under a temporary name beside *path* and renamed into place, so whoever waits for
    *path* to exist finds it listening and owner-only; a file an earlier server left at *path* is
    replaced."""
    temp = path.with_name(path.name + ".binding")
    temp.unlink(missing_ok=True)
    server = ExecServer(str(temp), token=token, spawner=spawner, cli_launches=cli_launches, threads=threads)
    temp.chmod(0o600)
    os.replace(temp, path)
    return server


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="exec_server.py")
    p.add_argument("--socket", required=True)
    p.add_argument("--spawner", required=True)
    p.add_argument("--shm-bytes", type=int, required=True)
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("--gid", type=int, required=True)
    p.add_argument("--hide", action="append", default=[])
    p.add_argument("--command-nproc", type=int)
    p.add_argument("--ucount", action="append", default=[])
    # Without one no start line could say the agent's CLI launched: the server does not start.
    p.add_argument("--cli-launch", action="append", required=True)
    p.add_argument("--no-namespaces", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    token = os.environ.pop(protocol.TOKEN_ENV, "")
    if not token:
        log(f"no {protocol.TOKEN_ENV} in the environment; refusing to start")
        return protocol.NO_TOKEN_EXIT
    score = lower_own_score()
    try:
        ready_file = ReadyFile(str(Path(args.socket).parent / protocol.READY_NAME))
    except OSError as e:
        log(f"the ready file could not be made ({e}); ending the server")
        return protocol.SPAWNER_GONE_EXIT
    pool = SpawnerPool(spawner_file=args.spawner, shm_bytes=args.shm_bytes, uid=args.uid, gid=args.gid,
                       hide=args.hide, no_namespaces=args.no_namespaces, score=score,
                       ready_file=ready_file, command_nproc=args.command_nproc, ucounts=args.ucount)
    pool.ready()
    server = bind_server(Path(args.socket), token=token, spawner=pool, cli_launches=tuple(args.cli_launch))
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
