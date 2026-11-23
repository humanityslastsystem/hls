"""What the sandbox's exec server, its spawner and the runner side agree on.

Three files run inside the build sandbox on the image's own interpreter, ``/usr/bin/python3
-I``, where hls is not installed: this one, ``exec_server.py`` and ``spawner.py``. The stage
copies all three into one node-local directory that the container binds read-only at
``EXEC_TARGET``, and the two programs load this file from beside themselves. On the runner side
it is ``hls.agent.exec_protocol``. So it imports the standard library only.

The sandbox is a contestant's container, which may mount only paths in its own role
directory (and, read-only, parts of cache entry copies the mount wall allows). So the stage copies
the three files into a directory of the sandbox role's own node-local directory and binds it
read-only at ``EXEC_TARGET``, and makes the control directory there too (mode 0700), bound
writable at ``CONTROL_TARGET``. Neither is a cache entry, so the mount wall has no say over them.

The control channel is HTTP over a UNIX socket, with harbor's exec API: ``POST /exec`` takes
``{"command", "cwd", "env", "timeout_sec"}`` and returns ``{"stdout", "stderr", "return_code"}``
(``stderr`` is always null: output is combined), and ``GET /health`` returns ``{"status": "ok"}``.
Every request carries the sandbox's token in ``TOKEN_HEADER``. The server and the spawner talk
over a socketpair in length-prefixed JSON messages, which may carry file descriptors."""

from __future__ import annotations

import json
import re
import socket
import struct
import zlib
from collections.abc import Collection, Mapping, Sequence

#: The request header that carries the sandbox's token.
TOKEN_HEADER = "X-HLS-Token"
#: The variable of the server's env file that holds the token; the server removes it from its
#: own environment before it runs anything.
TOKEN_ENV = "HLS_EXEC_TOKEN"
#: Where the control directory, which holds the server's socket, is bound in the container.
CONTROL_TARGET = "/run/hls-control"
#: The server's socket, inside the control directory.
SOCKET_NAME = "exec.sock"
#: The server's ready file, inside the control directory: one record (``ready_record``) naming the
#: spawner it serves from, rewritten in place before that spawner serves its first command. The
#: host reads the serving spawner's and reaper's pids from it (``hls.agent.probe.ready_pids``); no
#: command can see it.
READY_NAME = "ready"
#: Bytes of the ready file: one record, padded with NULs. The server allocates them at its start,
#: before any command runs, and every later record overwrites them, so writing one never needs a
#: new block of the node's disk, which a contestant may have filled.
READY_BYTES = 256
#: Where the three files are bound in the container.
EXEC_TARGET = "/opt/hls-exec"
#: The files of the in-sandbox programs, which the stage copies together.
SANDBOX_FILES = ("exec_protocol.py", "exec_server.py", "spawner.py")
#: The interpreter that runs them: the image's own, which the contestant cannot change.
SANDBOX_PYTHON = "/usr/bin/python3"
#: What the server puts in front of every command's PATH, as harbor's server does.
PATH_PREFIX = "/usr/bin:/usr/local/bin:"
#: Every command's ``oom_score_adj`` when it starts, so the kernel's OOM killer takes a command
#: before the harness. A command may lower its own back to 0 (``HARNESS_OOM_SCORE_ADJ``) and then
#: ties with the harness: whose fault a build's out-of-memory end is rests on the memory split,
#: which measures the harness's memory and the commands' apart (``hls.agent.memory``), never on
#: which process the kernel took.
OOM_SCORE_ADJ = 1000
#: The ``oom_score_adj`` the server, and so the spawner it starts, gives itself at start. The step
#: wrapper raised every process of a contestant step to ``OOM_SCORE_ADJ``; an unprivileged process
#: may lower its own back to the value it was raised from, 0, never below, so the harness in the
#: container does, and within the step the kernel's OOM killer takes a command that kept its
#: score before it.
HARNESS_OOM_SCORE_ADJ = 0
#: The file a process sets its own ``oom_score_adj`` in.
OWN_OOM_SCORE = "/proc/self/oom_score_adj"
#: The server's exit status once its spawner's machinery has failed: a sandbox end, infra.
SPAWNER_GONE_EXIT = 70
#: The spawner's exit status when its namespace's PID 1 (the reaper) died, which kills only
#: the contestant's own commands. The server recovers, serving the next command from a fresh
#: namespace, so this is never a sandbox end.
REAPER_GONE_EXIT = 75
#: The server's exit status when it is started without a token.
NO_TOKEN_EXIT = 2
#: The start of every line the server and the spawner write on stderr.
LOG_PREFIX = "hls-exec: "
#: Characters of a command the start line shows.
COMMAND_LOG_CHARS = 120
#: Bytes of a command's output the server keeps: the last ones. Its memory counts as the
#: harness's in the build's OOM rule, so a contestant printing gigabytes cannot grow it. The
#: largest exec output harbor parses, the re-entry check's read of Claude Code's stream tee, is
#: about 151 KB.
OUTPUT_TAIL_BYTES = 4 * 2**20
#: Bytes the server reads from a command's output at a time.
OUTPUT_CHUNK_BYTES = 64 * 2**10
#: Largest message the server and the spawner exchange.
MAX_MESSAGE_BYTES = 64 * 2**20
#: Most descriptors one message carries.
MAX_FDS = 1
_HEADER = struct.Struct(">I")
_READY_RECORD = re.compile(rb"(hls-ready spawner_pid=(\d+) reaper_pid=(\d+) oom_score_adj=(-?\d+|unknown)) "
                           rb"crc=([0-9a-f]{8})\n")
_DROPPED = re.compile(r"\[hls-exec: [0-9]+ bytes of earlier output dropped\]\n")
_START = re.compile(r"hls-exec: start id=(\d+) pid=(\d+) cli=([01]) cwd=(.*?) cmd=(.*)")


def server_command(*, shm_bytes: int, uid: int, gid: int, cli_launches: Sequence[str],
                   command_nproc: int | None = None, ucounts: Mapping[str, int] | None = None) -> tuple[str, ...]:
    """The container command that starts the exec server, its spawner capping /dev/shm at
    *shm_bytes* and running every command as *uid* and *gid* inside its namespace, its start lines
    saying whether a command holds one of *cli_launches* (``is_cli_launch``: the arm's adapter's
    ``cli_launch``), with *command_nproc* as each command's process ceiling (``RLIMIT_NPROC``, soft
    and hard; ``hls.substrate.nproc``) when it is given, and each of *ucounts*, a
    ``/proc/sys/user`` file's name to the limit the commands' namespace gets in it
    (``hls.substrate.ucounts``). Raise ValueError for no launch marker, or a blank one."""
    if not cli_launches or not all(launch.strip() for launch in cli_launches):
        raise ValueError("the exec server needs at least one launch marker, none blank, to tell the CLI's launch")
    command = (SANDBOX_PYTHON, "-I", f"{EXEC_TARGET}/exec_server.py", "--socket", f"{CONTROL_TARGET}/{SOCKET_NAME}",
               "--spawner", f"{EXEC_TARGET}/spawner.py", "--shm-bytes", str(shm_bytes), "--uid", str(uid),
               "--gid", str(gid), "--hide", CONTROL_TARGET,
               *(x for launch in cli_launches for x in ("--cli-launch", launch)))
    if command_nproc is not None:
        command = (*command, "--command-nproc", str(command_nproc))
    return (*command, *(x for name, limit in (ucounts or {}).items() for x in ("--ucount", f"{name}={limit}")))


def send_message(sock: socket.socket, message: dict, fds: tuple[int, ...] = ()) -> None:
    """Send *message* as JSON behind its length, with *fds* attached to its first byte."""
    body = json.dumps(message).encode()
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError(f"message of {len(body)} bytes is over {MAX_MESSAGE_BYTES}")
    data = _HEADER.pack(len(body)) + body
    sent = socket.send_fds(sock, [data], list(fds)) if fds else 0
    if sent < len(data):
        sock.sendall(data[sent:])


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks, got = [], 0
    while got < n:
        chunk = sock.recv(n - got)
        if not chunk:
            raise ConnectionError("the peer closed the channel inside a message")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def recv_message(sock: socket.socket) -> tuple[dict, list[int]] | None:
    """The next message and the descriptors that came with it; None once the peer has closed
    the channel between messages."""
    head, fds, flags, _ = socket.recv_fds(sock, _HEADER.size, MAX_FDS)
    if not head:
        return None
    if flags & socket.MSG_CTRUNC:
        for fd in fds:
            socket.close(fd)
        raise ConnectionError("a message carried more descriptors than the protocol allows")
    if len(head) < _HEADER.size:
        head += _recv_exact(sock, _HEADER.size - len(head))
    (size,) = _HEADER.unpack(head)
    if size > MAX_MESSAGE_BYTES:
        raise ConnectionError(f"message of {size} bytes is over {MAX_MESSAGE_BYTES}")
    return json.loads(_recv_exact(sock, size)), list(fds)


def ready_record(*, spawner_pid: int, reaper_pid: int, score: int | None) -> bytes:
    """The ready file's record for one spawner: the spawner's pid in its own namespace and the
    reaper's in the spawner's, which the memory split finds the contestant's processes by
    (``hls.agent.memory.find_reaper``), and the server's ``oom_score_adj`` (None when it could not
    read it), which the probe checks, with a CRC-32 of those fields, padded to ``READY_BYTES``."""
    body = (f"hls-ready spawner_pid={spawner_pid} reaper_pid={reaper_pid} "
            f"oom_score_adj={'unknown' if score is None else score}").encode()
    line = body + f" crc={zlib.crc32(body):08x}\n".encode()
    if len(line) > READY_BYTES:
        raise ValueError(f"a ready record of {len(line)} bytes is over {READY_BYTES}")
    return line.ljust(READY_BYTES, b"\0")


def parse_ready_record(data: bytes) -> dict[str, int | None] | None:
    """The fields of the ready file's record *data*; None for one not written yet, cut short, or
    torn, read while it was being rewritten: anything but one whole record whose checksum agrees
    and whose padding is all NULs. A torn record names no pair rather than a wrong one."""
    if len(data) != READY_BYTES:
        return None
    m = _READY_RECORD.match(data)
    if m is None or data[m.end():].strip(b"\0") or zlib.crc32(m.group(1)) != int(m.group(5), 16):
        return None
    spawner, reaper = int(m.group(2)), int(m.group(3))
    score = None if m.group(4) == b"unknown" else int(m.group(4))
    return {"spawner_pid": spawner, "reaper_pid": reaper, "score": score}


def dropped_marker(n: int) -> str:
    """The line in front of a command's output when its first *n* bytes were dropped."""
    return f"[hls-exec: {n} bytes of earlier output dropped]"


def dropped_line_length(output: str) -> int:
    """The length of the line ``dropped_marker`` put in front of *output*, its newline with it;
    0 when *output* starts with none."""
    m = _DROPPED.match(output)
    return 0 if m is None else m.end()


def is_cli_launch(command: str, launches: Collection[str]) -> bool:
    """Whether *command* launches the agent's CLI: it holds one of *launches*, the markers the
    exec server was given (the arm's adapter's ``cli_launch``). The server reads the whole command,
    so a launch past the characters its start line shows is still seen."""
    return any(launch in command for launch in launches)


def parse_start_line(line: str) -> dict | None:
    """The fields of one of the server's ``start`` lines: ``id``, ``pid``, ``cli`` (whether the
    command launches the agent's CLI, ``is_cli_launch``), ``cwd`` and ``cmd`` (the command's first
    ``COMMAND_LOG_CHARS`` characters, newlines escaped). None for any other line; anchored at the
    start."""
    m = _START.match(line)
    if m is None:
        return None
    return {"id": int(m[1]), "pid": int(m[2]), "cli": m[3] == "1", "cwd": m[4], "cmd": m[5]}

