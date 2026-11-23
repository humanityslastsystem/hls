"""Harbor's environment over the sandbox's exec socket: execs against a real exec server (its
spawner without namespaces), a lost channel told apart from a command's exit, and file transfers
that stay inside the directories the sandbox shares with the host."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import os
import resource
import signal
import socket
import stat
import sys
import threading
import time
import traceback
from pathlib import Path

import httpx
import pytest
from harbor.agents.installed.base import BaseInstalledAgent, NonZeroAgentExitCodeError

from hls.agent import exec_server, transcript
from hls.agent import exec_protocol as protocol
from hls.agent.environment import CONNECT_TIMEOUT_SEC, SandboxEnvironment, SandboxError, SandboxLost
from hls.substrate.secret_forms import forms_of
from hls.util import remove_tree
from tests.agent.test_exec_server import TOKEN, _Server, end_fields


class _Agent(BaseInstalledAgent):
    """The smallest installed agent: what harbor's adapters run their commands through."""

    @staticmethod
    def name() -> str:
        return "probe"

    async def install(self, environment) -> None:
        return None

    async def run(self, instruction, environment, context) -> None:
        return None

    def populate_context_post_run(self, context) -> None:
        return None


@pytest.fixture
def tmp_path(short_dir) -> Path:
    """Every test here puts an exec socket, or the check of one, under its directory, and the
    kernel takes at most 107 bytes of a socket's path: so it is a short directory, whatever
    ``--basetemp`` is."""
    return short_dir


@pytest.fixture
def start_server(short_dirs):
    started: list[_Server] = []

    def start(**kw) -> _Server:
        started.append(_Server(short_dirs(), **kw))
        return started[-1]

    yield start
    for s in started:
        s.stop()


@pytest.fixture
def server(start_server):
    return start_server()


@pytest.fixture
def host(tmp_path) -> dict[str, Path]:
    """The host directories of the mapped container directories, and one outside them all."""
    dirs = {"/workspace": tmp_path / "ws", "/logs": tmp_path / "logs", "/task": tmp_path / "task",
            "outside": tmp_path / "outside"}
    for d in dirs.values():
        d.mkdir()
    (dirs["outside"] / "secret.txt").write_text("not yours")
    return dirs


def _env(socket_path: Path, host: dict[str, Path], tmp_path: Path, **kw) -> SandboxEnvironment:
    kw.setdefault("token", TOKEN)
    kw.setdefault("start_timeout_sec", 20.0)
    root = tmp_path / "role"
    root.mkdir(exist_ok=True)
    return SandboxEnvironment(socket=socket_path, paths={k: v for k, v in host.items() if k.startswith("/")},
                              root=root, **kw)


def _run(coro):
    return asyncio.run(coro)


async def _started(env: SandboxEnvironment) -> SandboxEnvironment:
    await env.start(force_build=False)
    return env


# ---------------------------------------------------------------------------
# Execs
# ---------------------------------------------------------------------------


def test_an_exec_returns_harbors_result(server, host, tmp_path):
    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            return await env.exec("echo out; echo err >&2; exit 3")
        finally:
            await env.stop()

    result = _run(go())
    assert (result.stdout, result.stderr, result.return_code) == ("out\nerr", None, 3)


def test_an_exec_carries_its_directory_environment_and_timeout_and_ignores_the_user(server, host, tmp_path):
    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            here = await env.exec('pwd; echo "$ADDED"', cwd=str(tmp_path), env={"ADDED": "yes"}, user="root")
            late = await env.exec("sleep 5", timeout_sec=1)
            return here, late
        finally:
            await env.stop()

    here, late = _run(go())
    assert here.stdout == f"{tmp_path}\nyes" and here.return_code == 0
    assert late.return_code == 124


def test_execs_run_concurrently(server, host, tmp_path):
    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            began = time.monotonic()
            results = await asyncio.gather(*(env.exec(f"sleep 1; echo {i}") for i in range(4)))
            return results, time.monotonic() - began
        finally:
            await env.stop()

    results, took = _run(go())
    assert [r.stdout for r in results] == ["0", "1", "2", "3"] and took < 3.5


def test_harbors_installed_agents_run_their_commands_through_it(server, host, tmp_path):
    agent = _Agent(logs_dir=tmp_path / "agent-logs")

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            ok = await agent.exec_as_agent(env, "echo one | cat", env={"V": "2"})
            with pytest.raises(NonZeroAgentExitCodeError) as failed:
                await agent.exec_as_agent(env, "false | cat")
            return ok, failed.value
        finally:
            await env.stop()

    ok, failed = _run(go())
    assert ok.stdout == "one" and ok.return_code == 0
    # harbor prefixes `set -o pipefail;`, which the sandbox's bash honors
    assert "Command failed (exit 1)" in str(failed)


def test_the_client_ignores_the_runners_proxy_settings(server, host, tmp_path, monkeypatch):
    for var in ("ALL_PROXY", "all_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.setenv(var, "http://127.0.0.1:9")

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            return await env.exec("echo direct")
        finally:
            await env.stop()

    assert _run(go()).stdout == "direct"


def test_reading_a_reply_has_no_limit_and_connecting_has_one(host, tmp_path, monkeypatch):
    """The agent CLI runs as one exec for hours: a bound on reading would turn every exec longer
    than it into a lost sandbox."""
    env = _env(tmp_path / "s", host, tmp_path)
    seen: list[httpx.Timeout] = []
    real_send = httpx.AsyncClient.send

    async def send(self, request, **kw):
        seen.append(httpx.Timeout(**request.extensions["timeout"]))
        return await real_send(self, request, **kw)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    with pytest.raises(SandboxLost):
        _run(env.exec("true"))
    assert seen == [httpx.Timeout(None, connect=CONNECT_TIMEOUT_SEC)]


def test_one_environment_serves_execs_from_successive_event_loops(server, host, tmp_path):
    """The stage may drive the same environment from more than one ``asyncio.run``."""
    env = _env(server.socket, host, tmp_path)
    _run(env.start(force_build=False))
    assert _run(env.exec("echo first")).stdout == "first"
    assert _run(env.exec("echo second")).stdout == "second"
    _run(env.stop())


def test_a_command_that_kills_pid_1_comes_back_killed_and_the_next_exec_runs(server, host, tmp_path):
    """Ending PID 1 of the commands' namespace kills only the contestant's own commands: the
    command comes back as a result, killed by SIGKILL, never as a lost sandbox, and the next exec
    runs in a fresh namespace. Without namespaces a bare child stands in for the reaper, and the
    command ends it by its pid."""

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            killed = await env.exec(f"kill -KILL {server.reaper_pid()}; sleep 30")
            after = await env.exec("echo recovered")
            return env, killed, after
        finally:
            await env.stop()

    env, killed, after = _run(go())
    assert killed.return_code == -signal.SIGKILL and env.lost is None
    assert (after.stdout, after.return_code) == ("recovered", 0)
    assert server.proc.poll() is None


def test_a_command_that_fails_while_the_channel_lives_is_its_result_never_a_lost_sandbox(server, host,
                                                                                         tmp_path):
    """A command that cannot fork (a contestant exhausted the step's pids limit) is a result: the
    stage judges such an end from the step's own counters, never from the channel."""

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            failed = await env.exec("echo 'bash: fork: Resource temporarily unavailable' >&2; exit 254")
            after = await env.exec("echo still")
            return env, failed, after
        finally:
            await env.stop()

    env, failed, after = _run(go())
    assert (failed.stdout, failed.return_code) == ("bash: fork: Resource temporarily unavailable", 254)
    assert env.lost is None and after.stdout == "still"


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not held to RLIMIT_NPROC")
def test_execs_while_the_sandbox_is_out_of_processes_come_back_as_commands_that_could_not_start(server, host,
                                                                                                tmp_path):
    """A fork bomb that exhausts the processes the server's uid may start (here, run without
    namespaces, ``RLIMIT_NPROC`` counts every task of the uid; in a sandbox the commands run
    under their own ceiling in a user namespace of their own, ``hls.substrate.nproc``) keeps the
    server from starting a thread and the spawner from forking. Each exec still comes back, as a command that could not start (126,
    as the spawner answers a fork it cannot make), never as a lost sandbox; once processes free,
    the next exec runs. Two execs at once, so one needs a new connection."""

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        pids = (server.proc.pid, server.spawner_pid())
        held = [resource.prlimit(pid, resource.RLIMIT_NPROC) for pid in pids]
        for pid, (_, hard) in zip(pids, held, strict=True):
            resource.prlimit(pid, resource.RLIMIT_NPROC, (1, hard))
        try:
            starved = await asyncio.gather(env.exec("sleep 1; echo a"), env.exec("echo b"),
                                           return_exceptions=True)
        finally:
            for pid, limits in zip(pids, held, strict=True):
                resource.prlimit(pid, resource.RLIMIT_NPROC, limits)
        after = await env.exec("echo next")
        await env.stop()
        return env, starved, after

    env, starved, after = _run(go())
    assert [type(r).__name__ for r in starved] == ["ExecResult"] * 2, starved
    assert [(r.return_code, "cannot fork the command" in r.stdout) for r in starved] == [(126, True)] * 2
    assert env.lost is None and (after.stdout, after.return_code) == ("next", 0)


def test_an_exec_whose_log_lines_cannot_be_written_is_its_result_and_the_next_exec_runs(server, host,
                                                                                       tmp_path):
    """The server's stderr is the role's log on the host. When writing it fails (its disk full;
    here the server's file size limit stands at the log's size) the exec comes back as the
    command's own result, never a lost sandbox, and the log later says how many lines it lost."""

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        held = resource.prlimit(server.proc.pid, resource.RLIMIT_FSIZE)
        resource.prlimit(server.proc.pid, resource.RLIMIT_FSIZE, (server.err.stat().st_size, held[1]))
        try:
            full = await env.exec("echo hi")
        finally:
            resource.prlimit(server.proc.pid, resource.RLIMIT_FSIZE, held)
        after = await env.exec("echo next")
        await env.stop()
        return env, full, after

    env, full, after = _run(go())
    assert (full.stdout, full.return_code) == ("hi", 0) and env.lost is None
    assert (after.stdout, after.return_code) == ("next", 0)
    lines = server.log()
    assert "hls-exec: 2 earlier log lines could not be written" in lines
    assert [end_fields(x)["id"] for x in lines if x.startswith("hls-exec: end ")] == [2]


def test_more_execs_at_once_than_the_server_has_threads_all_come_back(server, host, tmp_path):
    """The server serves from a fixed set of threads started with it. More execs at once than it
    has wait for a free one, twice over, so no connection the client keeps open holds a thread."""
    count = exec_server.HANDLER_THREADS + 4

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        try:
            waves = []
            for command in ("sleep 0.5; echo {}", "echo {}"):
                waves.append(await asyncio.wait_for(
                    asyncio.gather(*(env.exec(command.format(i)) for i in range(count))), 30))
            return env, waves
        finally:
            await env.stop()

    env, waves = _run(go())
    assert [[r.stdout for r in wave] for wave in waves] == [[str(i) for i in range(count)]] * 2
    assert env.lost is None


# ---------------------------------------------------------------------------
# Starting, and a lost channel
# ---------------------------------------------------------------------------


def test_start_waits_until_the_server_binds_its_socket(start_server, host, tmp_path):
    server = start_server(wait=False)

    async def go():
        env = _env(server.socket, host, tmp_path, start_timeout_sec=30)
        await env.start(force_build=False)
        try:
            return (await env.exec("echo up")).stdout
        finally:
            await env.stop()

    assert _run(go()) == "up"


def test_start_gives_up_naming_the_socket_when_nothing_answers(host, tmp_path):
    missing = tmp_path / "exec.sock"
    env = _env(missing, host, tmp_path, start_timeout_sec=1.0)
    with pytest.raises(SandboxError, match=str(missing)) as raised:
        _run(env.start(force_build=False))
    assert not isinstance(raised.value, SandboxLost) and env.lost is None


def test_start_stops_waiting_once_the_sandbox_role_has_ended(host, tmp_path):
    env = _env(tmp_path / "exec.sock", host, tmp_path, alive=lambda: False)
    began = time.monotonic()
    with pytest.raises(SandboxLost, match="before its exec server answered"):
        _run(env.start(force_build=False))
    assert time.monotonic() - began < 2 and env.lost is not None


def test_a_wrong_token_is_a_harness_error_not_a_lost_sandbox(server, host, tmp_path):
    env = _env(server.socket, host, tmp_path, token="wrong-token-0000", start_timeout_sec=20)
    began = time.monotonic()
    with pytest.raises(SandboxError, match="403") as raised:
        _run(env.start(force_build=False))
    assert time.monotonic() - began < 5, "a refusal is not waited out"
    assert not isinstance(raised.value, SandboxLost)
    with pytest.raises(SandboxError, match="403") as raised:
        _run(env.exec("true"))
    assert not isinstance(raised.value, SandboxLost) and env.lost is None


def test_a_server_that_dies_during_an_exec_loses_the_sandbox_for_good(server, host, tmp_path):
    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        running = asyncio.ensure_future(env.exec("sleep 30"))
        await asyncio.sleep(0.5)
        server.proc.kill()
        with pytest.raises(SandboxLost) as first:
            await running
        began = time.monotonic()
        with pytest.raises(SandboxLost):
            await env.exec("true")
        again = time.monotonic() - began
        await env.stop()
        return env, first.value, again

    env, lost, again = _run(go())
    assert env.lost is lost and again < 0.5


def test_when_the_servers_own_machinery_fails_the_sandbox_is_lost(server, host, tmp_path):
    """The serving spawner's death, unlike the reaper's, is the harness's machinery failing: the
    server exits and its channel is gone."""

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        os.kill(server.spawner_pid(), signal.SIGKILL)
        status = await asyncio.to_thread(server.proc.wait, 10)
        with pytest.raises(SandboxLost):
            await env.exec("true")
        await env.stop()
        return env, status

    env, status = _run(go())
    assert status == protocol.SPAWNER_GONE_EXIT and env.lost is not None


def test_a_lost_sandbox_holds_neither_the_token_nor_the_requests_environment(server, host, tmp_path):
    """The loss is kept for the stage's lifetime and may be logged or dumped. It carries what was
    lost and why, never the failed request, whose header holds the token and whose body holds the
    route's credential."""
    credential = "sk-route-" + "0123456789abcdef"

    async def go():
        env = await _started(_env(server.socket, host, tmp_path))
        server.stop()
        with pytest.raises(SandboxLost):
            await env.exec("true", env={"ROUTE_KEY": credential})
        await env.stop()
        return env

    env = _run(go())
    text = "".join(traceback.format_exception(env.lost))
    assert "during an exec: ConnectError" in text
    assert TOKEN not in text and credential not in text
    assert env.lost.__cause__ is None and env.lost.__context__ is None, "it chains to no request"


def test_a_refused_connection_loses_the_sandbox(host, tmp_path):
    env = _env(tmp_path / "exec.sock", host, tmp_path)
    with pytest.raises(SandboxLost, match="during an exec"):
        _run(env.exec("true"))
    assert env.lost is not None


class _Rude:
    """A UNIX socket server that reads one request and then closes, or resets, the connection,
    answers it with *reply*, or, *silent*, holds it open and never answers."""

    def __init__(self, path: Path, *, reset: bool = False, reply: bytes | None = None,
                 silent: bool = False) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(str(path))
        self.sock.listen()
        self.reset = reset
        self.reply = reply
        self.silent = silent
        self.held: list[socket.socket] = []
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            conn.recv(65536)
            if self.silent:
                self.held.append(conn)
                continue
            if self.reply is not None:
                conn.sendall(self.reply)
            elif self.reset:
                conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
            conn.close()

    def close(self) -> None:
        self.sock.close()
        for conn in self.held:
            conn.close()


def _http(status: str, body: bytes) -> bytes:
    head = f"HTTP/1.0 {status}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n"
    return head.encode() + body


@pytest.mark.parametrize("reply", [_http("200 OK", b"not json"), _http("200 OK", b'{"stdout": "x"}'),
                                   _http("500 Internal Server Error", b'{"detail": "boom"}')],
                         ids=["not-json", "no-status", "500"])
def test_a_reply_that_is_not_a_result_is_a_harness_error_not_a_lost_sandbox(host, tmp_path, reply):
    canned = _Rude(tmp_path / "exec.sock", reply=reply)
    try:
        env = _env(tmp_path / "exec.sock", host, tmp_path)
        with pytest.raises(SandboxError) as raised:
            _run(env.exec("true"))
        assert not isinstance(raised.value, SandboxLost) and env.lost is None
    finally:
        canned.close()


@pytest.mark.parametrize("reset", [False, True], ids=["closed", "reset"])
def test_a_connection_closed_before_the_reply_loses_the_sandbox(host, tmp_path, reset):
    rude = _Rude(tmp_path / "exec.sock", reset=reset)
    try:
        env = _env(tmp_path / "exec.sock", host, tmp_path)
        with pytest.raises(SandboxLost):
            _run(env.exec("true"))
    finally:
        rude.close()


def test_start_gives_up_on_a_server_that_never_answers(host, tmp_path):
    """A socket that accepts and never replies is not a server that answered: ``start`` keeps
    to its deadline."""
    silent = _Rude(tmp_path / "exec.sock", silent=True)
    try:
        env = _env(tmp_path / "exec.sock", host, tmp_path, start_timeout_sec=1.0)
        began = time.monotonic()
        with pytest.raises(SandboxError, match=str(tmp_path / "exec.sock")) as raised:
            _run(env.start(force_build=False))
        assert time.monotonic() - began < 5
        assert not isinstance(raised.value, SandboxLost) and env.lost is None
    finally:
        silent.close()


def test_a_lost_sandbox_stays_lost_whatever_answers_on_its_socket_later(host, tmp_path):
    path = tmp_path / "exec.sock"
    env = _env(path, host, tmp_path)
    rude = _Rude(path, reset=True)
    try:
        with pytest.raises(SandboxLost):
            _run(env.exec("true"))
    finally:
        rude.close()
    path.unlink()
    impostor = _Rude(path, reply=_http("200 OK", b'{"stdout": "fine", "stderr": null, "return_code": 0}'))
    try:
        with pytest.raises(SandboxLost):
            _run(env.exec("true"))
        with pytest.raises(SandboxLost):
            _run(env.start(force_build=False))
    finally:
        impostor.close()


def test_a_lost_sandbox_is_neither_a_command_exit_nor_hidden_by_harbors_setup(host, tmp_path):
    assert issubclass(SandboxLost, RuntimeError) and not issubclass(SandboxLost, NonZeroAgentExitCodeError)
    env = _env(tmp_path / "exec.sock", host, tmp_path)
    agent = _Agent(logs_dir=tmp_path / "agent-logs")
    # harbor's setup wraps any error that is not a RuntimeError in "Agent install failed"
    with pytest.raises(SandboxLost):
        _run(agent.setup(env))


# ---------------------------------------------------------------------------
# The environment's own checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("paths, message", [
    ({"/workspace": "ws", "/workspace/sub": "task"}, "may not nest"),
    ({"workspace": "ws"}, "absolute path in normal form"),
    ({"/workspace/": "ws"}, "absolute path in normal form"),
    ({"/": "ws"}, "not /"),
    ({"/workspace": "missing"}, "does not exist"),
], ids=["nested", "relative", "trailing-slash", "root", "missing-host-dir"])
def test_bad_mappings_are_refused(host, tmp_path, paths, message):
    lookup = {"ws": host["/workspace"], "task": host["/task"], "missing": tmp_path / "nope"}
    with pytest.raises(ValueError, match=message):
        SandboxEnvironment(socket=tmp_path / "s", token=TOKEN, paths={k: lookup[v] for k, v in paths.items()},
                           root=tmp_path)


@pytest.mark.parametrize("posix", ["/opt/hls-exec", "/opt/hls-exec/sub", "/run/hls-control", "/run/hls",
                                   "/run", "/opt/hls-budget", "/opt"])
def test_a_mapping_that_overlaps_the_harnesss_own_binds_is_refused(host, tmp_path, posix):
    with pytest.raises(ValueError, match="overlaps the harness's"):
        SandboxEnvironment(socket=tmp_path / "s", token=TOKEN, paths={posix: host["/workspace"]},
                           root=tmp_path)


def test_a_mapping_beside_the_harnesss_binds_is_accepted(host, tmp_path):
    env = SandboxEnvironment(socket=tmp_path / "s", token=TOKEN, root=tmp_path,
                             paths={"/run/hls-other": host["/workspace"], "/opt/hls-execs": host["/task"]})
    assert set(env.paths) == {"/run/hls-other", "/opt/hls-execs"}


def test_a_socket_in_a_mapped_host_directory_is_refused_even_through_a_link(host, tmp_path):
    control = host["/workspace"] / "control"
    control.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(host["/workspace"])
    for socket_path in (host["/workspace"] / "exec.sock", control / "exec.sock",
                        alias / "control" / "exec.sock"):
        with pytest.raises(ValueError, match="exec socket's directory"):
            _env(socket_path, host, tmp_path)
    with pytest.raises(ValueError, match="exec socket's directory"):
        _env(control / "exec.sock", {**host, "/workspace": alias}, tmp_path)


def test_a_socket_path_the_kernel_cannot_bind_and_an_empty_token_are_refused(host, tmp_path):
    with pytest.raises(ValueError, match="longer than 107 bytes"):
        _env(tmp_path / ("s" * 120), host, tmp_path)
    with pytest.raises(ValueError, match="token"):
        _env(tmp_path / "s", host, tmp_path, token="")


@pytest.mark.parametrize("token", ["tok\nen-0123456789", "tok en-0123456789", "t\u00f6k-0123456789",
                                   "tok\x7f-0123456789"], ids=["newline", "space", "not-ascii", "delete"])
def test_a_token_no_header_can_carry_is_refused_without_showing_it(host, tmp_path, token):
    """Otherwise the client would fail on every request, naming the token in its error."""
    with pytest.raises(ValueError, match="token") as raised:
        _env(tmp_path / "s", host, tmp_path, token=token)
    assert "0123456789" not in str(raised.value)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def _file(path: Path, text: str, mode: int = 0o640) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(mode)
    os.utime(path, (1_000_000, 1_000_000))
    return path


def test_an_uploaded_file_lands_in_the_mapped_directory_with_its_parents_mode_and_time(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    src = _file(tmp_path / "src" / "brief.md", "brief", 0o664)
    umask = os.umask(0o077)  # the mode is the source's, whatever the runner's umask
    try:
        _run(env.upload_file(src, "/workspace/a/b/brief.md"))
    finally:
        os.umask(umask)
    out = host["/workspace"] / "a" / "b" / "brief.md"
    assert out.read_text() == "brief"
    assert stat.S_IMODE(out.stat().st_mode) == 0o664 and out.stat().st_mtime == 1_000_000
    _run(env.upload_file(src, "/workspace/a"))
    assert (host["/workspace"] / "a" / "brief.md").read_text() == "brief"
    _run(env.upload_file(src, "/workspace"))
    assert (host["/workspace"] / "brief.md").read_text() == "brief"
    (host["/workspace"] / "brief.md").unlink()
    _run(env.upload_file(src, "/workspace/"))
    assert (host["/workspace"] / "brief.md").read_text() == "brief"


def test_an_upload_replaces_a_file_in_place(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    _file(host["/workspace"] / "x.txt", "old long text")
    _run(env.upload_file(_file(tmp_path / "src" / "x.txt", "new"), "/workspace/x.txt"))
    assert (host["/workspace"] / "x.txt").read_text() == "new"


@pytest.mark.parametrize("target", ["/etc/passwd", "/workspace/../outside/x", "relative/x", "/workspaces/x"],
                         ids=["unmapped", "dotdot", "relative", "prefix-of-a-name"])
def test_an_upload_outside_the_shared_directories_is_refused(host, tmp_path, target):
    env = _env(tmp_path / "s", host, tmp_path)
    with pytest.raises(ValueError):
        _run(env.upload_file(_file(tmp_path / "src" / "x", "x"), target))


def test_an_upload_through_a_planted_symlink_is_refused_and_writes_nothing_outside(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    (host["/workspace"] / "link").symlink_to(host["outside"])
    (host["/workspace"] / "inner").mkdir()
    (host["/workspace"] / "inner-link").symlink_to(host["/workspace"] / "inner")
    (host["/workspace"] / "secret.txt").symlink_to(host["outside"] / "secret.txt")
    src = _file(tmp_path / "src" / "x", "planted")
    for target in ("/workspace/link/x", "/workspace/inner-link/x", "/workspace/secret.txt"):
        with pytest.raises(ValueError, match="symlink"):
            _run(env.upload_file(src, target))
    assert sorted(p.name for p in host["outside"].iterdir()) == ["secret.txt"]
    assert (host["outside"] / "secret.txt").read_text() == "not yours"
    assert list((host["/workspace"] / "inner").iterdir()) == []


def test_an_upload_onto_a_planted_fifo_is_refused_without_blocking_or_writing_into_it(host, tmp_path):
    """A pipe the contestant left where a file is uploaded would hold the harness's write open
    until someone reads it; with a reader on it, the upload would feed the reader."""
    env = _env(tmp_path / "s", host, tmp_path)
    pipe = host["/workspace"] / "pipe"
    os.mkfifo(pipe)
    src = _file(tmp_path / "src" / "x", "upload")
    errors: list[BaseException] = []

    def upload() -> None:
        try:
            _run(env.upload_file(src, "/workspace/pipe"))
        except BaseException as e:  # noqa: BLE001 -- the test reads what the upload raised
            errors.append(e)

    worker = threading.Thread(target=upload, daemon=True)
    worker.start()
    worker.join(5)
    if worker.is_alive():
        os.close(os.open(pipe, os.O_RDONLY | os.O_NONBLOCK))  # let the blocked write go
        pytest.fail("the upload blocked on a pipe")
    assert len(errors) == 1 and isinstance(errors[0], ValueError) and "not a regular file" in str(errors[0])
    reader = os.open(pipe, os.O_RDONLY | os.O_NONBLOCK)
    try:
        with pytest.raises(ValueError, match="not a regular file"):
            _run(env.upload_file(src, "/workspace/pipe"))
        assert os.read(reader, 100) == b"", "nothing was written into the pipe"
    finally:
        os.close(reader)


def test_an_upload_from_a_missing_source_is_refused(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    with pytest.raises(FileNotFoundError):
        _run(env.upload_file(tmp_path / "nope", "/workspace/x"))
    with pytest.raises(FileNotFoundError):
        _run(env.upload_dir(tmp_path / "nope", "/workspace/d"))


def test_an_uploaded_tree_merges_into_its_target_and_follows_the_sources_links(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    src = tmp_path / "refs"
    _file(src / "a.txt", "a")
    _file(src / "deep" / "b.txt", "b")
    _file(tmp_path / "assets" / "c.txt", "c")
    (src / "c.txt").symlink_to(tmp_path / "assets" / "c.txt")
    _file(host["/task"] / "files" / "kept.txt", "kept")
    _run(env.upload_dir(src, "/task/files"))
    files = host["/task"] / "files"
    assert sorted(str(p.relative_to(files)) for p in files.rglob("*") if p.is_file()) == [
        "a.txt", "c.txt", "deep/b.txt", "kept.txt"]
    assert not (files / "c.txt").is_symlink() and (files / "c.txt").read_text() == "c"
    _run(env.upload_dir(src, "/workspace/new/references"))
    assert (host["/workspace"] / "new" / "references" / "deep" / "b.txt").read_text() == "b"


def test_an_uploaded_tree_does_not_enter_a_planted_symlink(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    src = tmp_path / "refs"
    _file(src / "deep" / "b.txt", "b")
    (host["/workspace"] / "refs").mkdir()
    (host["/workspace"] / "refs" / "deep").symlink_to(host["outside"])
    with pytest.raises(ValueError, match="symlink"):
        _run(env.upload_dir(src, "/workspace/refs"))
    assert sorted(p.name for p in host["outside"].iterdir()) == ["secret.txt"]


@pytest.mark.parametrize("swap, refusal", [("link", "symlink"), ("remove", "ENOENT")])
@pytest.mark.parametrize("upload", ["file", "dir"])
def test_a_directory_the_upload_made_and_the_sandbox_swapped_is_refused(host, tmp_path, monkeypatch, upload,
                                                                       swap, refusal):
    """The sandbox replaces a directory the upload just made with a link, or removes it, before
    the upload opens it: a ValueError, and nothing written outside."""
    env = _env(tmp_path / "s", host, tmp_path)
    src = _file(tmp_path / "src" / "made" / "x", "planted")
    real_mkdir = os.mkdir

    def mkdir_then_swap(name, mode=0o777, *, dir_fd=None):
        real_mkdir(name, mode, dir_fd=dir_fd)
        if name == "made":
            os.rmdir(name, dir_fd=dir_fd)
            if swap == "link":
                os.symlink(host["outside"], name, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", mkdir_then_swap)
    with pytest.raises(ValueError, match=refusal):
        if upload == "file":
            _run(env.upload_file(src, "/workspace/made/x"))
        else:
            _run(env.upload_dir(tmp_path / "src", "/workspace"))
    assert sorted(p.name for p in host["outside"].iterdir()) == ["secret.txt"]


@pytest.mark.parametrize("swap, refusal", [("link", "symlink"), ("remove", "ENOENT")])
def test_a_directory_target_the_sandbox_swapped_after_the_check_is_refused(host, tmp_path, monkeypatch, swap,
                                                                          refusal):
    env = _env(tmp_path / "s", host, tmp_path)
    src = _file(tmp_path / "src" / "x", "planted")
    (host["/workspace"] / "target").mkdir()
    real_stat = os.stat

    def stat_then_swap(path, *, dir_fd=None, follow_symlinks=True):
        st = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == "target" and dir_fd is not None:
            os.rmdir(path, dir_fd=dir_fd)
            if swap == "link":
                os.symlink(host["outside"], path, dir_fd=dir_fd)
        return st

    monkeypatch.setattr(os, "stat", stat_then_swap)
    with pytest.raises(ValueError, match=refusal):
        _run(env.upload_file(src, "/workspace/target"))
    assert sorted(p.name for p in host["outside"].iterdir()) == ["secret.txt"]


def test_a_transfer_failure_the_sandbox_does_not_explain_is_raised_as_the_machines(host, tmp_path,
                                                                                    monkeypatch):
    """A full disk is the machine's failure: it is raised as the OSError it is, never as a
    refusal that reads as the contestant's."""
    env = _env(tmp_path / "s", host, tmp_path)
    src = _file(tmp_path / "src" / "x", "x")
    real_open = os.open

    def open_on_a_full_disk(path, flags, mode=0o777, *, dir_fd=None):
        if path == "x" and flags & os.O_CREAT:
            raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", open_on_a_full_disk)
    with pytest.raises(OSError) as raised:
        _run(env.upload_file(src, "/workspace/x"))
    assert raised.value.errno == errno.ENOSPC


def test_a_directory_made_at_an_uploads_target_after_the_check_is_refused(host, tmp_path, monkeypatch):
    env = _env(tmp_path / "s", host, tmp_path)
    src = _file(tmp_path / "src" / "x", "planted")
    real_stat = os.stat

    def stat_then_make(path, *, dir_fd=None, follow_symlinks=True):
        try:
            return real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        except FileNotFoundError:
            if path == "target" and dir_fd is not None:
                os.mkdir(path, dir_fd=dir_fd)
            raise

    monkeypatch.setattr(os, "stat", stat_then_make)
    with pytest.raises(ValueError, match="not a regular file"):
        _run(env.upload_file(src, "/workspace/target"))
    assert list((host["/workspace"] / "target").iterdir()) == []


def test_an_upload_onto_a_file_the_sandbox_holds_a_lease_on_is_refused_at_once(host, tmp_path):
    """A lease the contestant holds on its own file would hold a blocking open for the lease-break
    time; the upload's open does not block, and its refusal is a ValueError."""
    env = _env(tmp_path / "s", host, tmp_path)
    leased = _file(host["/workspace"] / "leased", "mine")
    src = _file(tmp_path / "src" / "x", "new")
    held = os.open(leased, os.O_RDONLY)
    # The kernel signals a lease's holder (here this process) when an open breaks it.
    old = signal.signal(signal.SIGIO, signal.SIG_IGN)
    try:
        try:
            fcntl.fcntl(held, fcntl.F_SETLEASE, fcntl.F_RDLCK)
        except OSError as e:
            pytest.skip(f"this file system grants no lease: {e}")
        began = time.monotonic()
        with pytest.raises(ValueError, match="EAGAIN"):
            _run(env.upload_file(src, "/workspace/leased"))
        assert time.monotonic() - began < 5
        fcntl.fcntl(held, fcntl.F_SETLEASE, fcntl.F_UNLCK)
    finally:
        os.close(held)
        signal.signal(signal.SIGIO, old)
    assert leased.read_text() == "mine"


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not held to file modes")
def test_a_transfer_the_sandboxs_file_modes_refuse_is_a_valueerror_never_a_raw_oserror(host, tmp_path):
    """A caller may read an OSError as the machine's failing: a refusal that what the sandbox holds
    at the path explains is a ValueError, so the contestant's doing is never taken for ours."""
    env = _env(tmp_path / "s", host, tmp_path)
    ws, logs, out = host["/workspace"], host["/logs"], tmp_path / "host"
    src = _file(tmp_path / "src" / "sub" / "x", "x")
    modes = ((ws / "closed", 0), (ws / "readonly", 0o500), (ws / "unsearchable", 0o400), (logs / "closed", 0))
    for d, mode in modes:
        d.mkdir()
        d.chmod(mode)
    _file(logs / "closed.txt", "no", 0)
    transfers = {"open a directory": lambda: env.upload_file(src, "/workspace/closed/x"),
                 "create a file": lambda: env.upload_file(src, "/workspace/readonly/x"),
                 "make a directory": lambda: env.upload_file(src, "/workspace/readonly/new/x"),
                 "look at the target": lambda: env.upload_file(src, "/workspace/unsearchable/x"),
                 "make a tree's directory": lambda: env.upload_dir(tmp_path / "src", "/workspace/readonly"),
                 "read a file": lambda: env.download_file("/logs/closed.txt", out / "f"),
                 "open a tree": lambda: env.download_dir("/logs/closed", out / "d")}
    try:
        for what, transfer in transfers.items():
            with pytest.raises(ValueError, match="EACCES"):
                _run(transfer())
                pytest.fail(f"{what}: no refusal")
        ws.chmod(0)
        with pytest.raises(ValueError, match="EACCES"):
            _run(env.upload_file(src, "/workspace/x"))
    finally:
        for d in (ws, *(d for d, _ in modes)):
            d.chmod(0o755)
    assert not (ws / "readonly" / "x").exists() and not (out / "f").exists()


def test_a_downloaded_file_keeps_its_mode_and_time(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    _file(host["/logs"] / "agent" / "claude-code.txt", "stream", 0o604)
    out = tmp_path / "host" / "deep" / "claude-code.txt"
    _run(env.download_file("/logs/agent/claude-code.txt", out))
    assert out.read_text() == "stream"
    assert stat.S_IMODE(out.stat().st_mode) == 0o604 and out.stat().st_mtime == 1_000_000


def test_a_download_of_a_symlink_a_special_file_or_a_missing_file_is_refused(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    (host["/logs"] / "leak").symlink_to(host["outside"] / "secret.txt")
    os.mkfifo(host["/logs"] / "pipe")
    out = tmp_path / "host" / "x"
    with pytest.raises(ValueError, match="symlink"):
        _run(env.download_file("/logs/leak", out))
    with pytest.raises(ValueError, match="not a regular file"):
        _run(env.download_file("/logs/pipe", out))
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(str(host["/logs"] / "sock"))
    try:
        with pytest.raises(ValueError, match="ENXIO"):
            _run(env.download_file("/logs/sock", out))
    finally:
        listening.close()
    with pytest.raises(RuntimeError, match="does not exist"):
        _run(env.download_file("/logs/none", out))
    (host["/logs"] / "dir-link").symlink_to(host["outside"])
    with pytest.raises(ValueError, match="symlink"):
        _run(env.download_file("/logs/dir-link/secret.txt", out))
    assert not out.exists()


def _tree(root: Path) -> list[str]:
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*"))


def test_a_downloaded_tree_skips_links_and_special_files_and_records_why(host, tmp_path):
    """The agent may have written the tree. A link is neither followed nor made on the host, where
    a reader of the copy would follow it into the host's own files; a pipe is never read."""
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    _file(agent / "sessions" / "projects" / "t.jsonl", "turns")
    _file(agent / "codex.txt", "codex")
    (agent / "leak").symlink_to(host["outside"] / "secret.txt")
    (agent / "sessions" / "out").symlink_to(host["outside"])
    os.mkfifo(agent / "pipe")
    listening = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listening.bind(str(agent / "sock"))
    target = tmp_path / "host" / "agent"
    try:
        _run(env.download_dir("/logs/agent", target))
    finally:
        listening.close()
    assert (target / "sessions" / "projects" / "t.jsonl").read_text() == "turns"
    assert (target / "codex.txt").read_text() == "codex"
    assert _tree(target) == ["codex.txt", "sessions", "sessions/projects", "sessions/projects/t.jsonl"]
    assert env.skipped == [("/logs/agent/leak", "symlink"), ("/logs/agent/pipe", "not a regular file"),
                           ("/logs/agent/sessions/out", "symlink"), ("/logs/agent/sock", "not a regular file")]


def test_a_downloaded_tree_is_walked_without_recursing_and_cut_at_its_depth_bound(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    _file(agent / "top.txt", "top")
    (agent / "nest").mkdir()
    fd = os.open(agent / "nest", os.O_RDONLY | os.O_DIRECTORY)
    for _ in range(sys.getrecursionlimit() + 100):
        os.mkdir("d", dir_fd=fd)
        deeper = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
        os.close(fd)
        fd = deeper
    os.close(fd)
    target = tmp_path / "host" / "agent"
    try:
        _run(env.download_dir("/logs/agent", target))
    finally:
        remove_tree(agent / "nest")
    bound = transcript.MAX_DEPTH
    assert env.skipped == [("/logs/agent/nest" + "/d" * bound, f"deeper than {bound} levels")]
    assert (target / ("nest" + "/d" * (bound - 1))).is_dir()
    assert list((target / ("nest" + "/d" * (bound - 1))).iterdir()) == []
    assert (target / "top.txt").read_text() == "top"


def test_a_downloaded_entry_whose_path_is_past_the_bound_is_skipped(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    name = "n" * 255
    deep = "/".join([name] * 4)  # 1,023 bytes
    _file(agent / deep / "f.txt", "far")
    _file(agent / "near.txt", "near")
    target = tmp_path / "host" / "agent"
    _run(env.download_dir("/logs/agent", target))
    bound = transcript.MAX_PATH_BYTES
    assert env.skipped == [(f"/logs/agent/{deep}/f.txt", f"its path is longer than {bound} bytes")]
    assert (target / deep).is_dir() and list((target / deep).iterdir()) == []
    assert (target / "near.txt").read_text() == "near"


def test_a_downloaded_file_that_would_take_the_copy_past_its_size_bound_is_skipped(host, tmp_path,
                                                                                  monkeypatch):
    monkeypatch.setattr(transcript, "MAX_BYTES", 10)
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    for name, text in (("a.txt", "12345"), ("b.txt", "123456"), ("c.txt", "1234")):
        _file(agent / name, text)
    target = tmp_path / "host" / "agent"
    _run(env.download_dir("/logs/agent", target))
    assert env.skipped == [("/logs/agent/b.txt", "its 6 bytes would take the download past 10 bytes")]
    assert _tree(target) == ["a.txt", "c.txt"]


def test_a_downloaded_tree_stops_after_its_entry_bound(host, tmp_path, monkeypatch):
    monkeypatch.setattr(transcript, "MAX_ENTRIES", 3)
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    for name in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"):
        _file(agent / name, name)
    target = tmp_path / "host" / "agent"
    _run(env.download_dir("/logs/agent", target))
    assert env.skipped == [("/logs/agent/d.txt", "past the first 3 entries: the rest is not read")]
    assert _tree(target) == ["a.txt", "b.txt", "c.txt"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root is not held to file modes")
def test_a_downloaded_tree_skips_what_it_cannot_read(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    _file(agent / "ok.txt", "ok")
    _file(agent / "closed.txt", "no", 0)
    _file(agent / "shut" / "in.txt", "in")
    (agent / "shut").chmod(0)
    _file(agent / "unsearchable" / "in.txt", "in")
    (agent / "unsearchable").chmod(0o400)  # listed, but nothing in it can be looked at
    target = tmp_path / "host" / "agent"
    try:
        _run(env.download_dir("/logs/agent", target))
    finally:
        for d in ("shut", "unsearchable"):
            (agent / d).chmod(0o755)
    assert env.skipped == [("/logs/agent/closed.txt", "unreadable: EACCES"),
                           ("/logs/agent/shut", "unreadable: EACCES"),
                           ("/logs/agent/unsearchable/in.txt", "unreadable: EACCES")]
    assert _tree(target) == ["ok.txt", "unsearchable"]


class _Listed:
    """What ``os.scandir`` returns, from entries already read."""

    def __init__(self, entries: list) -> None:
        self.entries = entries

    def __enter__(self):
        return iter(self.entries)

    def __exit__(self, *exc) -> bool:
        return False

    def __iter__(self):
        return iter(self.entries)


@pytest.mark.parametrize("when", ["listed", "looked-at"])
def test_an_entry_that_became_a_pipe_since_it_was_seen_is_skipped_not_fatal(host, tmp_path, monkeypatch,
                                                                            when):
    """A file the sandbox swaps for a pipe after the walk listed it, or after it read its type,
    is left out; it neither ends the download nor reaches the host."""
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    _file(agent / "a.txt", "a")
    _file(agent / "b.txt", "b")
    real_scandir, real_stat = os.scandir, os.stat

    def listed_then_swapped(path):
        with real_scandir(path) as it:
            entries = list(it)
        if any(e.name == "a.txt" for e in entries):
            (agent / "a.txt").unlink()
            os.mkfifo(agent / "a.txt")
        return _Listed(entries)

    def looked_at_then_swapped(path, *, dir_fd=None, follow_symlinks=True):
        st = real_stat(path, dir_fd=dir_fd, follow_symlinks=follow_symlinks)
        if path == "a.txt" and dir_fd is not None:
            os.unlink(path, dir_fd=dir_fd)
            os.mkfifo(path, dir_fd=dir_fd)
        return st

    if when == "listed":
        monkeypatch.setattr(os, "scandir", listed_then_swapped)
    else:
        monkeypatch.setattr(os, "stat", looked_at_then_swapped)
    target = tmp_path / "host" / "agent"
    _run(env.download_dir("/logs/agent", target))
    assert env.skipped == [("/logs/agent/a.txt", "not a regular file")]
    assert _tree(target) == ["b.txt"]


def test_a_directory_the_download_cannot_list_is_skipped(host, tmp_path, monkeypatch):
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    _file(agent / "ok.txt", "ok")
    _file(agent / "dark" / "in.txt", "in")
    dark = (agent / "dark").stat().st_ino
    real_scandir = os.scandir

    def scandir(path):
        if isinstance(path, int) and os.fstat(path).st_ino == dark:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", scandir)
    target = tmp_path / "host" / "agent"
    _run(env.download_dir("/logs/agent", target))
    assert env.skipped == [("/logs/agent/dark", "unreadable: EIO")]
    assert _tree(target) == ["dark", "ok.txt"]


def test_a_failed_read_leaves_a_file_out_of_a_tree_and_is_raised_for_one_file(host, tmp_path, monkeypatch):
    """A tree's download never fails on what the agent left; a file asked for by name that cannot
    be read fails as the machine's OSError. Neither leaves a partial copy."""
    env = _env(tmp_path / "s", host, tmp_path)
    agent = host["/logs"] / "agent"
    _file(agent / "a.txt", "aaa")
    _file(agent / "b.txt", "bbb")
    bad = (agent / "a.txt").stat().st_ino
    real_read = os.read

    def read(fd, n):
        if os.fstat(fd).st_ino == bad:
            raise OSError(errno.EIO, os.strerror(errno.EIO))
        return real_read(fd, n)

    monkeypatch.setattr(os, "read", read)
    target = tmp_path / "host" / "agent"
    _run(env.download_dir("/logs/agent", target))
    assert env.skipped == [("/logs/agent/a.txt", "unreadable: EIO")]
    assert _tree(target) == ["b.txt"]
    one = tmp_path / "host" / "one.txt"
    with pytest.raises(OSError) as raised:
        _run(env.download_file("/logs/agent/a.txt", one))
    assert raised.value.errno == errno.EIO and not one.exists()


@pytest.mark.parametrize("change, copied", [("grows", "start"), ("shrinks", "st")])
def test_a_downloaded_file_is_copied_at_most_as_long_as_it_was_when_opened(host, tmp_path, monkeypatch,
                                                                          change, copied):
    """A file the agent's leftover processes keep writing can neither hold a download nor take it
    past its size bound; one they cut short ends its copy early."""
    env = _env(tmp_path / "s", host, tmp_path)
    path = _file(host["/logs"] / "agent" / "f.txt", "start")
    real = os.set_blocking

    def set_blocking_then_change(fd, blocking):
        real(fd, blocking)
        if change == "grows":
            with path.open("a") as f:
                f.write("more" * 1000)
        else:
            os.truncate(path, 2)

    monkeypatch.setattr(os, "set_blocking", set_blocking_then_change)
    _run(env.download_dir("/logs/agent", tmp_path / "host" / "agent"))
    assert (tmp_path / "host" / "agent" / "f.txt").read_text() == copied


def test_a_download_replaces_each_top_level_directory_whole(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    _file(host["/logs"] / "agent" / "sessions" / "new.jsonl", "new")
    target = tmp_path / "host" / "agent"
    _file(target / "sessions" / "stale.jsonl", "stale")
    _file(target / "other.txt", "other")
    _run(env.download_dir("/logs/agent", target))
    assert sorted(p.name for p in (target / "sessions").iterdir()) == ["new.jsonl"]
    assert (target / "other.txt").read_text() == "other"


def test_a_download_replaces_a_target_tree_deeper_than_the_interpreters_recursion_limit(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    _file(host["/logs"] / "agent" / "sessions" / "new.jsonl", "new")
    target = tmp_path / "host" / "agent"
    (target / "sessions").mkdir(parents=True)
    fd = os.open(target / "sessions", os.O_RDONLY | os.O_DIRECTORY)
    for _ in range(sys.getrecursionlimit() + 100):
        os.mkdir("d", dir_fd=fd)
        deeper = os.open("d", os.O_RDONLY | os.O_DIRECTORY, dir_fd=fd)
        os.close(fd)
        fd = deeper
    os.close(fd)
    _run(env.download_dir("/logs/agent", target))
    assert sorted(p.name for p in (target / "sessions").iterdir()) == ["new.jsonl"]


def test_a_download_of_a_missing_or_linked_directory_is_refused(host, tmp_path):
    env = _env(tmp_path / "s", host, tmp_path)
    (host["/logs"] / "agent").symlink_to(host["outside"])
    with pytest.raises(ValueError, match="symlink"):
        _run(env.download_dir("/logs/agent", tmp_path / "host" / "a"))
    with pytest.raises(RuntimeError, match="does not exist"):
        _run(env.download_dir("/logs/none", tmp_path / "host" / "b"))
    with pytest.raises(ValueError, match="not in a directory the sandbox shares"):
        _run(env.download_dir("/etc", tmp_path / "host" / "c"))



def test_an_output_cut_to_its_tail_leaves_no_secret_cut_through_at_its_start(server, host, tmp_path):
    """The server keeps a command's last ``OUTPUT_TAIL_BYTES``. A secret the sandbox holds whose
    start fell before that cut leaves its end at the tail's start, which no whole-value scrub
    finds: the environment backs off past it."""
    key = "sk-test-0123456789abcdef"
    after = protocol.OUTPUT_TAIL_BYTES - 5

    async def go():
        env = await _started(_env(server.socket, host, tmp_path, secret_forms=forms_of("key", key)))
        try:
            return await env.exec(f"printf 'x%.0s' $(seq 1 100); printf '%s' {key}; head -c {after} /dev/zero | tr '\\0' y")
        finally:
            await env.stop()

    result = _run(go())
    marker, _, tail = result.stdout.partition("\n")
    assert marker == protocol.dropped_marker(100 + len(key) - 5)
    assert tail == "y" * after
