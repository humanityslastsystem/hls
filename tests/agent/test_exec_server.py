"""The exec server, run for real as the sandbox runs it (``python -I`` on the file, no hls on the
path), with its spawner in the mode without namespaces."""

from __future__ import annotations

import collections
import errno
import http.client
import json
import os
import re
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import hls.agent.exec_server as exec_server
from hls.arms import ADAPTERS
from hls.agent import exec_protocol as protocol

AGENT = Path(exec_server.__file__).parent
TOKEN = "tok-0123456789abcdef"
BASH_OWN = {"PWD", "SHLVL", "_"}
#: One of the server's end lines, matched from the start of a line, so a start line whose command
#: text embeds an end line is not read as one.
END_LINE = re.compile(r"hls-exec: end id=(\d+) status=(-?\d+) signal=(\S+) dropped=(\d+) sec=([0-9.]+)")


def end_fields(line: str) -> dict | None:
    """The fields of one of the server's end lines (``exec_server``): ``id``, ``status``,
    ``signal`` (None for ``-``), ``dropped`` and ``sec``; None for any other line."""
    m = END_LINE.match(line)
    if m is None:
        return None
    return {"id": int(m[1]), "status": int(m[2]), "signal": None if m[3] == "-" else m[3], "dropped": int(m[4]),
            "sec": float(m[5])}


class _Unix(http.client.HTTPConnection):
    def __init__(self, path: Path, timeout: float = 30.0) -> None:
        super().__init__("control", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._path))


# What the step wrapper does to a contestant step before apptainer starts: raise the score, then
# become the command.
RAISED = ("/bin/sh", "-c", 'echo 1000 > /proc/self/oom_score_adj && exec "$0" "$@"')


#: The launch markers of every harness hls runs (``HarnessAdapter.cli_launch``), which the
#: servers here are given.
LAUNCHES = tuple(adapter.cli_launch for adapter in ADAPTERS.values())


class _Server:
    """An exec server process on a socket in a short temporary directory; with *raised*, started
    the way the step wrapper starts a contestant step, at ``oom_score_adj`` 1000; with *through*,
    run by that script (and its arguments), which is given the server's path and arguments."""

    def __init__(self, root: Path, *, env: dict[str, str] | None = None, shm_bytes: int = 1 << 20,
                 wait: bool = True, raised: bool = False, command_nproc: int | None = None,
                 ucounts: tuple[str, ...] = (), launches: tuple[str, ...] = LAUNCHES,
                 through: tuple[str, ...] = ()) -> None:
        self.socket = root / protocol.SOCKET_NAME
        self.err = root / "server.err"
        base = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", protocol.TOKEN_ENV: TOKEN, "KEEP": "kept"}
        self.proc = subprocess.Popen(
            [*(RAISED if raised else ()), sys.executable, "-I", *through, str(AGENT / "exec_server.py"), "--socket",
             str(self.socket),
             "--spawner", str(AGENT / "spawner.py"), "--shm-bytes", str(shm_bytes), "--uid", "1234", "--gid", "5678",
             "--hide", str(root), "--no-namespaces",
             *(("--command-nproc", str(command_nproc)) if command_nproc is not None else ()),
             *(x for limit in ucounts for x in ("--ucount", limit)),
             *(x for launch in launches for x in ("--cli-launch", launch))],
            env=base if env is None else env, stdin=subprocess.PIPE, stderr=self.err.open("w"))
        if not wait:
            return
        deadline = time.monotonic() + 20
        while not self.socket.exists():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.stop()
                raise AssertionError(f"the server never bound its socket: {self.err.read_text()}")
            time.sleep(0.02)

    def call(self, method: str, path: str, body: object = None, *, token: str | None = TOKEN,
             raw: bytes | None = None, timeout: float = 30.0) -> tuple[int, dict]:
        conn = _Unix(self.socket, timeout)
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers[protocol.TOKEN_HEADER] = token
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        conn.request(method, path, body=data, headers=headers)
        resp = conn.getresponse()
        out = json.loads(resp.read())
        conn.close()
        return resp.status, out

    def run(self, command: str, **kw) -> dict:
        status, body = self.call("POST", "/exec", {"command": command, **kw})
        assert status == 200, body
        return body

    def log(self) -> list[str]:
        return self.err.read_text().splitlines()

    def spawner_pid(self) -> int:
        [child] = Path(f"/proc/{self.proc.pid}/task/{self.proc.pid}/children").read_text().split()
        return int(child)

    def reaper_pid(self) -> int:
        """The reaper of the spawner the server started last."""
        return int(re.search(r"reaper_pid=(\d+)", _ready_lines(self.log())[-1]).group(1))

    def wait_for_log(self, predicate, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate(self.log()):
            assert time.monotonic() < deadline, "the server never logged what the test waited for"
            time.sleep(0.005)

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture
def start_server(short_dirs):
    """Starts servers, and stops every one of them when the test ends, passed or not."""
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


@pytest.mark.parametrize("token", [None, "wrong-token-0000"], ids=["missing", "wrong"])
def test_a_request_without_the_token_is_refused_and_runs_nothing(server, tmp_path, token):
    marker = tmp_path / "ran"
    assert server.call("POST", "/exec", {"command": f"touch {marker}"}, token=token)[0] == 403
    assert server.call("GET", "/health", token=token)[0] == 403
    time.sleep(0.2)
    assert not marker.exists()
    assert not any(line.startswith("hls-exec: start") for line in server.log())


def test_health_answers_with_the_token(server):
    assert server.call("GET", "/health") == (200, {"status": "ok"})


def test_the_output_is_combined_and_stripped_and_the_status_returned(server):
    assert server.run("echo out; echo err >&2; exit 3") == {"stdout": "out\nerr", "stderr": None, "return_code": 3}


def test_no_output_is_null_and_blank_output_is_empty(server):
    assert server.run("true")["stdout"] is None
    assert server.run("printf '  \\n'")["stdout"] == ""


def test_the_command_sees_the_servers_environment_without_the_token_then_the_requests(server):
    out = server.run("exec env -0", env={"ADDED": "1", "KEEP": "over"})["stdout"]
    env = {k: v for k, v in (kv.split("=", 1) for kv in out.split("\0") if kv) if k not in BASH_OWN}
    assert env == {"PATH": "/usr/bin:/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "KEEP": "over", "ADDED": "1"}


def test_a_requests_environment_reaches_only_its_own_command(server):
    """The agent's credential travels in each exec request's env and never becomes the server's
    own, so a later command that was not given it does not see it."""
    assert server.run("echo $ROUTE_KEY", env={"ROUTE_KEY": "k-0123"})["stdout"] == "k-0123"
    assert server.run("echo ${ROUTE_KEY-unset}")["stdout"] == "unset"


def test_the_log_holds_no_environment(server):
    """The server's stderr is the role's log in the attempt: no variable of a request, nor the
    token, ever reaches it."""
    value = "k-0123456789-route"
    server.run("true", env={"ROUTE_KEY": value}, cwd="/")
    server.run("exit 3", env={"ROUTE_KEY": value})
    assert server.call("POST", "/exec", {"command": "true", "env": {"ROUTE_KEY": value}, "cwd": 5})[0] == 400
    text = server.err.read_text()
    assert value not in text and "ROUTE_KEY" not in text and TOKEN not in text
    assert "KEEP" not in text and "kept" not in text, "nor the server's own environment"


def test_a_request_may_replace_path(server):
    assert server.run("echo $PATH", env={"PATH": "/x:/usr/bin:/bin"})["stdout"] == "/x:/usr/bin:/bin"


def test_the_spawner_starts_with_an_empty_environment(server):
    """The container's variables, the agent's credentials among them, reach a command only
    in the environment its request is given, never through PID 1 of its namespace."""
    assert Path(f"/proc/{server.spawner_pid()}/environ").read_bytes() == b""


def _score(pid: int) -> int:
    return int(Path(f"/proc/{pid}/oom_score_adj").read_text())


def test_each_command_leads_its_own_session_with_oom_score_adj_1000(server):
    adj, pid, sid = server.run("cat /proc/self/oom_score_adj; echo $$; awk '{print $6}' /proc/$$/stat")[
        "stdout"].split()
    assert adj == "1000" and sid == pid
    harness = min(_score(os.getpid()), protocol.HARNESS_OOM_SCORE_ADJ)
    assert _score(server.proc.pid) == _score(server.spawner_pid()) == harness


def test_the_harness_lowers_the_score_the_step_wrapper_raised_and_commands_keep_1000(start_server):
    """The step wrapper raises a contestant step to 1000 before apptainer starts. The server
    lowers itself back, the spawner inherits that, and only commands run at 1000, so the kernel's
    OOM killer takes a command before the harness."""
    if _score(os.getpid()) < 0:
        pytest.skip("this test process runs below 0, which an unprivileged process cannot reach again")
    server = start_server(raised=True)
    assert _score(server.proc.pid) == _score(server.spawner_pid()) == protocol.HARNESS_OOM_SCORE_ADJ
    assert server.run("cat /proc/self/oom_score_adj")["stdout"] == "1000"
    assert any(line.endswith(f" oom_score_adj={protocol.HARNESS_OOM_SCORE_ADJ}") for line in server.log()
               if line.startswith("hls-exec: ready "))


@pytest.mark.parametrize("held, expected", [("1000", "0"), ("300", "0"), ("0", "0"), ("-500", "-500")])
def test_the_score_is_lowered_to_the_harness_value_and_never_raised(tmp_path, held, expected):
    path = tmp_path / "oom_score_adj"
    path.write_text(held)
    assert exec_server.lower_own_score(str(path)) == expected
    assert path.read_text() == expected


def test_a_score_the_kernel_will_not_lower_is_logged_and_kept(tmp_path, capfd):
    path = tmp_path / "oom_score_adj"
    path.write_text("1000")
    path.chmod(0o444)
    assert exec_server.lower_own_score(str(path)) == "1000"
    assert "hls-exec: cannot lower the oom score" in capfd.readouterr().err


def test_stdin_is_empty(server):
    """The server's own stdin is an open pipe here, so a command that inherited it would block."""
    started = time.monotonic()
    assert server.run("cat; echo done")["stdout"] == "done"
    assert time.monotonic() - started < 5


def test_a_working_directory_is_honored_and_one_that_cannot_be_entered_fails_with_127(server, tmp_path):
    assert server.run("pwd", cwd=str(tmp_path))["stdout"] == str(tmp_path)
    failed = server.run("pwd", cwd="/no/such/dir")
    assert failed["return_code"] == 127 and "cannot start the command" in failed["stdout"]


def test_a_timeout_ends_the_command_with_124(server):
    started = time.monotonic()
    assert server.run("sleep 30", timeout_sec=1)["return_code"] == 124
    assert time.monotonic() - started < 10


def test_a_command_a_signal_ends_returns_minus_the_signal_and_logs_it(server):
    assert server.run("kill -9 $$")["return_code"] == -signal.SIGKILL
    ends = [end_fields(line) for line in server.log()]
    assert [e for e in ends if e][-1]["status"] == -9
    assert [e for e in ends if e][-1]["signal"] == "SIGKILL"


def test_every_command_logs_a_start_and_an_end_line(server):
    command = "echo hi\n" + "#" * 200
    server.run(command, cwd="/")
    [start] = [line for line in server.log() if line.startswith("hls-exec: start ")]
    shown = command[:protocol.COMMAND_LOG_CHARS].replace("\n", "\\n")
    assert start.startswith("hls-exec: start id=1 pid=") and start.endswith(f" cli=0 cwd=/ cmd={shown}")
    assert protocol.parse_start_line(start) == {"id": 1, "pid": int(start.split("pid=")[1].split()[0]),
                                                "cli": False, "cwd": "/", "cmd": shown}
    [end] = [end_fields(line) for line in server.log() if line.startswith("hls-exec: end ")]
    assert end["id"] == 1 and end["status"] == 0 and end["signal"] is None and end["dropped"] == 0
    assert end["sec"] >= 0
    assert any(line.startswith("hls-exec: ready ns=") for line in server.log())


@pytest.mark.parametrize("launch", [
    "(echo $BASHPID > /logs/agent/claude.pid; exec claude --verbose --output-format=stream-json",
    "if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; codex exec --json -- 'go' 2>&1 </dev/null",
], ids=["claude-code", "codex"])
def test_a_start_line_says_whether_its_command_launches_the_agents_cli_wherever_the_launch_is(server, launch):
    """The start line shows only the command's first characters, but says, from the whole command,
    whether it launches the agent's CLI (``is_cli_launch``): the build reads it as the evidence that a
    contestant process ran."""
    command = f"export LONG={'x' * protocol.COMMAND_LOG_CHARS}; true || {launch}"
    server.run(command)
    [start] = [protocol.parse_start_line(line) for line in server.log() if line.startswith("hls-exec: start ")]
    assert start["cli"] is True and "claude" not in start["cmd"] and "codex" not in start["cmd"]


def test_commands_run_concurrently(server):
    results = []
    threads = [threading.Thread(target=lambda: results.append(server.run("sleep 1; echo x"))) for _ in range(3)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert [r["stdout"] for r in results] == ["x", "x", "x"]
    assert time.monotonic() - started < 2.5


def _threads(pid: int) -> int:
    return len(os.listdir(f"/proc/{pid}/task"))


def test_requests_are_served_by_threads_started_with_the_server(server):
    """None is started per request, so a sandbox that has run out of processes cannot keep the
    server from answering."""
    idle = _threads(server.proc.pid)
    results: list[dict] = []
    callers = [threading.Thread(target=lambda: results.append(server.run("sleep 1; echo x")))
               for _ in range(3)]
    for t in callers:
        t.start()
    server.wait_for_log(lambda lines: sum(x.startswith("hls-exec: start ") for x in lines) == 3)
    busy = _threads(server.proc.pid)
    for t in callers:
        t.join(10)
    assert busy == idle
    assert [r["stdout"] for r in results] == ["x"] * 3


def test_each_reply_closes_its_connection(server):
    """A connection serves one request, so a thread is held only while a request is served, never
    by a client's idle connection."""
    conn = _Unix(server.socket)
    conn.request("GET", "/health", headers={protocol.TOKEN_HEADER: TOKEN})
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("Connection") == "close"
    conn.close()


def test_log_lines_that_cannot_be_written_are_counted_never_raised(tmp_path):
    """The log is the role's on the host. A write there that fails (its disk full) loses the line,
    never the request. The next line written says how many were lost, after ending a line the
    failure cut short, so every line still starts a line of its own."""
    code = (
        "import resource, sys\n"
        f"sys.path.insert(0, {str(AGENT)!r})\n"
        "import exec_server\n"
        "soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (12, hard))\n"
        "exec_server.log('first')\n"
        "exec_server.log('second')\n"
        "resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))\n"
        "exec_server.log('third')\n"
        "exec_server.log('fourth')\n")
    log = tmp_path / "server.err"
    with log.open("w") as f:
        subprocess.run([sys.executable, "-I", "-c", code], env={"PATH": "/usr/bin:/bin"}, stderr=f,
                       check=True)
    assert log.read_text().splitlines() == [
        "hls-exec: fi", "hls-exec: 2 earlier log lines could not be written", "hls-exec: third",
        "hls-exec: fourth"]


def test_a_background_job_holding_the_output_delays_the_reply_as_harbors_server_does(server):
    """Harbor's server reads the output to its end, so a job left writing to it holds the
    reply until it ends; one sent elsewhere does not."""
    started = time.monotonic()
    assert server.run("sleep 1.5 & echo started")["stdout"] == "started"
    assert time.monotonic() - started >= 1.4
    started = time.monotonic()
    assert server.run("sleep 5 >/dev/null 2>&1 & echo started")["stdout"] == "started"
    assert time.monotonic() - started < 1.0


def test_only_the_tail_of_a_long_output_is_kept_after_a_line_saying_how_much_was_dropped(server):
    full = "".join(f"{i}\n" for i in range(1, 1_000_001)).encode()
    cap = protocol.OUTPUT_TAIL_BYTES
    assert len(full) > cap
    out = server.run("seq 1 1000000")["stdout"]
    dropped = len(full) - cap
    assert out == (protocol.dropped_marker(dropped) + "\n" + full[-cap:].decode()).strip()
    assert end_fields(server.log()[-1])["dropped"] == dropped


def test_output_at_the_cap_is_kept_whole(server):
    out = server.run(f"head -c {protocol.OUTPUT_TAIL_BYTES} /dev/zero | tr '\\0' a")["stdout"]
    assert out == "a" * protocol.OUTPUT_TAIL_BYTES
    assert end_fields(server.log()[-1])["dropped"] == 0


def test_a_cut_through_a_character_is_replaced_not_an_error(server):
    """Three-byte characters, one byte more than the cap: the cut falls inside the first kept one."""
    count = protocol.OUTPUT_TAIL_BYTES // 3 + 1
    out = server.run(f"python3 -c \"import sys; sys.stdout.write('\\u20ac' * {count})\"")["stdout"]
    marker, tail = out.split("\n", 1)
    assert marker == protocol.dropped_marker(3 * count - protocol.OUTPUT_TAIL_BYTES)
    assert tail.startswith("\ufffd") and tail.lstrip("\ufffd") == "\u20ac" * (count - 1)


def _peak_rss_kib(pid: int) -> int:
    line = next(x for x in Path(f"/proc/{pid}/status").read_text().splitlines() if x.startswith("VmHWM:"))
    return int(line.split()[1])


def test_the_servers_memory_stays_bounded_while_a_command_prints_64_mib(server):
    """Its memory counts as the harness's in the build's OOM rule."""
    server.run("echo warm")
    before = _peak_rss_kib(server.proc.pid)
    out = server.run("head -c 67108864 /dev/zero | tr '\\0' x")["stdout"]
    assert len(out) > protocol.OUTPUT_TAIL_BYTES
    grown_mib = (_peak_rss_kib(server.proc.pid) - before) / 1024
    assert grown_mib < 48, f"the server's peak memory grew {grown_mib:.0f} MiB"


def test_the_reader_keeps_the_last_bytes_and_counts_the_rest():
    r, w = os.pipe()
    data = bytes(range(256)) * 40
    writer = threading.Thread(target=lambda: (os.write(w, data), os.close(w)))
    writer.start()
    tail, dropped = exec_server.read_tail(r, cap=1000, chunk=64)
    writer.join()
    os.close(r)
    assert tail == data[-1000:] and dropped == len(data) - 1000


def test_an_unauthenticated_request_to_an_unknown_path_is_refused_before_the_path(server, tmp_path):
    """The token is checked before the path, so an unauthenticated request learns nothing from the
    status and runs nothing, whatever path it names."""
    marker = tmp_path / "ran"
    assert server.call("POST", "/nope", {"command": f"touch {marker}"}, token=None)[0] == 403
    assert server.call("POST", "/nope", {"command": "true"}, token="wrong-token-0000")[0] == 403
    assert server.call("GET", "/nope", token=None)[0] == 403
    time.sleep(0.2)
    assert not marker.exists()
    assert not any(line.startswith("hls-exec: start") for line in server.log())


def test_bad_requests_run_nothing(server):
    assert server.call("POST", "/nope", {"command": "true"})[0] == 404
    assert server.call("GET", "/nope")[0] == 404
    assert server.call("POST", "/exec", raw=b"{not json")[0] == 400
    assert server.call("POST", "/exec", {"cwd": "/"})[0] == 400
    assert server.call("POST", "/exec", [1, 2])[0] == 400
    assert server.call("POST", "/exec", {"command": 5})[0] == 400
    assert server.call("POST", "/exec", {"command": "true", "cwd": 5})[0] == 400
    assert server.call("POST", "/exec", {"command": "true", "env": {"A": 1}})[0] == 400
    assert server.call("POST", "/exec", {"command": "true", "env": ["A"]})[0] == 400
    assert server.call("POST", "/exec", {"command": "true", "timeout_sec": "5"})[0] == 400
    assert server.call("POST", "/exec", {"command": "true", "timeout_sec": True})[0] == 400
    assert not any(line.startswith("hls-exec: start") for line in server.log())


def test_the_socket_is_owner_only(server):
    assert stat.S_IMODE(server.socket.stat().st_mode) == 0o600


def test_the_socket_appears_at_its_path_only_listening_and_owner_only(short_dir, monkeypatch):
    """Whoever waits for the socket to exist finds it listening, with mode 0600: it is bound
    under another name and renamed into place, over whatever an earlier server left there."""
    path = short_dir / protocol.SOCKET_NAME
    path.write_text("left by an earlier server")
    seen = []
    real = os.replace

    def replace(src, dst):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.connect(str(src))
        probe.close()
        seen.append((Path(dst), stat.S_IMODE(os.stat(src).st_mode), path.is_socket()))
        real(src, dst)

    monkeypatch.setattr(exec_server.os, "replace", replace)
    server = exec_server.bind_server(path, token=TOKEN, spawner=None, cli_launches=LAUNCHES)
    try:
        assert seen == [(path, 0o600, False)]
        assert path.is_socket() and stat.S_IMODE(path.stat().st_mode) == 0o600
        assert list(short_dir.iterdir()) == [path]
    finally:
        server.server_close()


def test_when_the_spawners_serving_process_dies_the_server_exits_70(server):
    """Killing the serving process (not PID 1 of the commands' namespace) is the harness's own
    machinery failing, which is a lost sandbox: infra."""
    os.kill(server.spawner_pid(), signal.SIGKILL)
    assert server.proc.wait(10) == protocol.SPAWNER_GONE_EXIT
    assert any("spawner ended" in line for line in server.log())


#: Times the machinery-death race is run.
MACHINERY_RACE_TRIALS = 10


def test_a_command_in_flight_when_the_machinery_dies_gets_no_reply_and_the_server_exits_70(start_server):
    """The serving spawner dies and its command with it, as the reaper's death signal kills the
    namespace's commands in production. That is the harness's failure: the server exits 70 before
    it answers anything, so the client sees its channel lost, never a command killed by SIGKILL,
    which would read as the agent's end. Which death the server sees first varies, so it is run
    several times."""
    for _ in range(MACHINERY_RACE_TRIALS):
        server = start_server()
        outcome: list[object] = []

        def call(s=server, out=outcome) -> None:
            try:
                out.append(s.call("POST", "/exec", {"command": "sleep 30"}, timeout=20))
            except (OSError, http.client.HTTPException) as e:
                out.append(type(e).__name__)

        caller = threading.Thread(target=call)
        caller.start()
        server.wait_for_log(lambda lines: any(x.startswith("hls-exec: start ") for x in lines))
        [start] = [x for x in server.log() if x.startswith("hls-exec: start ")]
        command = int(re.search(r" pid=(\d+) ", start).group(1))
        os.kill(server.spawner_pid(), signal.SIGKILL)
        os.kill(command, signal.SIGKILL)
        caller.join(20)
        assert server.proc.wait(10) == protocol.SPAWNER_GONE_EXIT
        assert len(outcome) == 1 and isinstance(outcome[0], str), f"answered {outcome}"
        server.stop()


def test_commands_in_flight_are_answered_only_once_the_spawners_end_is_judged():
    """The pool learns how the spawner ended (``on_gone``) before anything in flight is answered
    as killed: on a machinery failure ``on_gone`` ends the server, so nothing is answered."""
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    record: dict[str, object] = {"answered_before": []}
    channel = exec_server.SpawnerChannel(
        ours, lambda reason: record["answered_before"].append(record["pending"].ended.is_set()))
    channel.start()
    _, pending, r = channel.run(["/bin/true"], None, {})
    record["pending"] = pending
    _, fds = protocol.recv_message(theirs)
    for fd in fds:
        os.close(fd)
    theirs.close()
    assert pending.ended.wait(5)
    assert record["answered_before"] == [False]
    os.close(r)


def test_when_the_reaper_dies_the_server_recovers_and_the_next_command_runs(server):
    """A command that ends PID 1 of its own namespace (here the reaper is killed directly, since
    without namespaces a command cannot reach it) kills only the contestant's own commands. The
    server does not end; it serves the next command from a fresh namespace."""
    results: list[dict] = []
    inflight = threading.Thread(target=lambda: results.append(server.run("sleep 30")))
    inflight.start()
    time.sleep(0.5)
    os.kill(server.reaper_pid(), signal.SIGKILL)
    inflight.join(15)
    assert results == [{"stdout": None, "stderr": None, "return_code": -int(signal.SIGKILL)}]
    assert server.proc.poll() is None, "the server kept running"
    assert server.run("echo recovered")["stdout"] == "recovered"
    assert sum(1 for line in server.log() if line.startswith("hls-exec: ready ")) == 2, "a fresh spawner served it"
    assert any("reaper ended" in line for line in server.log())


def test_the_ready_file_names_the_serving_spawner_before_its_first_command_runs(server):
    """Without namespaces the control directory is not hidden, so a command can read the ready
    file: the first command of each spawner already finds that spawner named there, at the start
    and after a reaper's end brings a fresh one."""
    ready = server.socket.parent / protocol.READY_NAME
    first = _named(server.run(f"tr -d '\\000' < {ready}")["stdout"])
    assert first == _named(_ready_lines(server.log())[-1])
    os.kill(server.reaper_pid(), signal.SIGKILL)
    server.wait_for_log(lambda lines: any("reaper ended" in line for line in lines))
    fresh = _named(server.run(f"tr -d '\\000' < {ready}")["stdout"])
    assert len(_ready_lines(server.log())) == 2 and fresh != first
    assert fresh == _named(_ready_lines(server.log())[-1])


def _named(text: str) -> tuple[int, int]:
    """The spawner's and the reaper's pids a ready record or a logged ready line names."""
    found = re.search(r"spawner_pid=(\d+) reaper_pid=(\d+)", text)
    assert found, text
    return int(found.group(1)), int(found.group(2))


def test_a_respawn_rewrites_the_ready_file_in_place(server):
    """The record is rewritten in the blocks allocated at the server's start: the same file, the
    same size, no block more."""
    ready = server.socket.parent / protocol.READY_NAME
    before = ready.stat()
    assert (before.st_size, before.st_blocks > 0) == (protocol.READY_BYTES, True)
    first = exec_protocol_record(ready)
    os.kill(server.reaper_pid(), signal.SIGKILL)
    server.wait_for_log(lambda lines: any("reaper ended" in line for line in lines))
    assert server.run("echo again")["stdout"] == "again"
    after = ready.stat()
    assert (after.st_ino, after.st_size, after.st_blocks) == (before.st_ino, before.st_size, before.st_blocks)
    record = exec_protocol_record(ready)
    # The fresh spawner and its reaper are named in place of the first pair.
    assert (record["spawner_pid"], record["reaper_pid"]) == _named(_ready_lines(server.log())[-1])
    assert (record["spawner_pid"], record["reaper_pid"]) != (first["spawner_pid"], first["reaper_pid"])


def exec_protocol_record(ready: Path) -> dict:
    record = protocol.parse_ready_record(ready.read_bytes())
    assert record is not None
    return record


#: Runs the exec server with every new file and every new allocation failing ENOSPC once the flag
#: file named by its first argument exists: a node disk the contestant filled after the start.
_FULL_DISK = """import errno, os, runpy, sys
flag, script = sys.argv[1], sys.argv[2]
sys.argv = [script, *sys.argv[3:]]
real_open, real_fallocate, real_ftruncate = os.open, os.posix_fallocate, os.ftruncate


def full(*_):
    if os.path.exists(flag):
        raise OSError(errno.ENOSPC, os.strerror(errno.ENOSPC))


def opener(path, flags, mode=0o777, *, dir_fd=None):
    if flags & os.O_CREAT and not os.path.exists(path):
        full()
    return real_open(path, flags, mode, dir_fd=dir_fd)


os.open = opener
os.posix_fallocate = lambda fd, offset, length: full() or real_fallocate(fd, offset, length)
os.ftruncate = lambda fd, length: full() or real_ftruncate(fd, length)
runpy.run_path(script, run_name="__main__")
"""


def test_a_disk_filled_after_the_start_cannot_stop_a_respawn(start_server, short_dirs):
    """A contestant fills the node's disk and ends its reaper: the next command still gets a
    result from a fresh namespace, since the fresh spawner's record needs no new block."""
    aside = short_dirs()
    wrapper, flag = aside / "full_disk.py", aside / "full"
    wrapper.write_text(_FULL_DISK)
    server = start_server(through=(str(wrapper), str(flag)))
    assert server.run("echo before")["stdout"] == "before"
    flag.touch()
    os.kill(server.reaper_pid(), signal.SIGKILL)
    server.wait_for_log(lambda lines: any("reaper ended" in line for line in lines))
    assert server.run("echo after")["stdout"] == "after"
    assert server.proc.poll() is None, "the server kept running"
    record = exec_protocol_record(server.socket.parent / protocol.READY_NAME)
    assert (record["spawner_pid"], record["reaper_pid"]) == _named(_ready_lines(server.log())[-1])


REAPER_GONE_LINE = "hls-exec: the sandbox's reaper ended"


def _reapers_gone(lines: list[str]) -> int:
    return sum(1 for line in lines if line.startswith(REAPER_GONE_LINE))


def _ready_lines(lines: list[str]) -> list[str]:
    """The line each spawner the server started logged once it was ready, in order."""
    return [line for line in lines if line.startswith("hls-exec: ready ")]


def test_parallel_commands_across_reaper_deaths_all_get_a_reply(server):
    """Agent CLIs run tool calls in parallel, so a request can reach the spawner after its reaper
    died and it exited, before the server has noticed. Each still gets a reply: the command runs
    on a fresh spawner, or comes back killed with its namespace. A dropped connection would read
    as a lost sandbox, which is infra. Each reaper death costs one fresh spawner, no more."""
    kills, clients = 40, 6
    replies: collections.Counter[str] = collections.Counter()
    lock, stop = threading.Lock(), threading.Event()

    def client() -> None:
        while not stop.is_set():
            try:
                status, body = server.call("POST", "/exec", {"command": "true"}, timeout=10)
                key = f"rc={body['return_code']}" if status == 200 else f"http {status}"
            except (OSError, http.client.HTTPException) as e:
                key = f"dropped: {type(e).__name__}"
            with lock:
                replies[key] += 1

    threads = [threading.Thread(target=client, daemon=True) for _ in range(clients)]
    for t in threads:
        t.start()
    try:
        for kill in range(kills):
            server.wait_for_log(lambda lines, n=kill: len(_ready_lines(lines)) > n)
            time.sleep(0.02)
            os.kill(server.reaper_pid(), signal.SIGKILL)
        server.wait_for_log(lambda lines: len(_ready_lines(lines)) > kills)
    finally:
        stop.set()
        for t in threads:
            t.join(15)
    assert server.proc.poll() is None, "the server kept running"
    assert set(replies) <= {"rc=0", f"rc={-int(signal.SIGKILL)}"}, dict(replies)
    assert sum(replies.values()) > kills
    lines = server.log()
    assert [x for x in lines if not x.startswith("hls-exec: ")] == [], "no request failed inside the server"
    assert len(_ready_lines(lines)) == _reapers_gone(lines) + 1 == kills + 1


def test_a_command_from_before_a_respawn_that_ends_after_it_leaves_the_fresh_spawners_commands_alone(server,
                                                                                                    tmp_path):
    """Request ids restart at 1 on each fresh spawner, so a command from before a reaper death
    that ends only after the respawn must release its id on the spawner that ran it. On the fresh
    spawner the same id belongs to another command, which would then never get its reply. Without
    namespaces a job in a session of its own outlives the reaper; it holds the old command's
    output open across the respawn, standing in for any delay before that command ends."""
    old: list[dict] = []
    marker = tmp_path / "in-own-session"
    first = threading.Thread(target=lambda: old.append(
        server.run(f"setsid sh -c 'touch {marker}; exec sleep 1.5' & sleep 30")))
    first.start()
    deadline = time.monotonic() + 10
    while not marker.exists():
        assert time.monotonic() < deadline, "the job never started"
        time.sleep(0.005)
    os.kill(server.reaper_pid(), signal.SIGKILL)
    server.wait_for_log(lambda lines: _reapers_gone(lines) == 1)
    # Id 1 on the fresh spawner, still running when the old command ends and releases its id 1.
    status, body = server.call("POST", "/exec", {"command": "sleep 3; echo done"}, timeout=15)
    first.join(15)
    assert (status, body) == (200, {"stdout": "done", "stderr": None, "return_code": 0})
    assert old == [{"stdout": None, "stderr": None, "return_code": -int(signal.SIGKILL)}]
    assert len(_ready_lines(server.log())) == 2


def test_a_health_check_after_a_reaper_death_brings_up_a_fresh_spawner(server):
    """A health check that answers means commands can run, so after a reaper death it starts the
    fresh spawner itself rather than leave that to the next command."""
    os.kill(server.reaper_pid(), signal.SIGKILL)
    server.wait_for_log(lambda lines: _reapers_gone(lines) == 1)
    assert len(_ready_lines(server.log())) == 1
    assert server.call("GET", "/health") == (200, {"status": "ok"})
    assert len(_ready_lines(server.log())) == 2


# The pool and its channel, in this process.


def _channel() -> tuple[exec_server.SpawnerChannel, socket.socket]:
    """A channel whose spawner side is a socket the test holds."""
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    channel = exec_server.SpawnerChannel(ours, lambda reason: None)
    channel.start()
    return channel, theirs


def test_a_command_the_spawner_took_but_never_answered_ends_killed_when_the_channel_closes():
    """A spawner whose reaper died can exit before it reads a spawn it was sent. That request
    must not wait for ever: the command comes back killed, as its namespace's commands are."""
    channel, theirs = _channel()
    _, pending, r = channel.run(["/bin/true"], None, {})
    _, fds = protocol.recv_message(theirs)
    for fd in fds:
        os.close(fd)
    theirs.close()
    assert pending.ended.wait(5)
    assert (pending.status, pending.signal) == (None, int(signal.SIGKILL))
    os.close(r)


def test_a_command_sent_to_a_spawner_that_had_ended_leaves_no_record_and_says_the_channel_is_gone():
    channel, theirs = _channel()
    theirs.close()
    with pytest.raises(exec_server.ChannelGone):
        channel.run(["/bin/true"], None, {})
    assert channel._pending == {}


def test_a_request_that_cannot_be_sent_leaves_no_record_and_the_channel_serving():
    channel, theirs = _channel()
    with pytest.raises(ValueError, match="over"):
        channel.run(["/bin/true"], None, {"BIG": "x" * protocol.MAX_MESSAGE_BYTES})
    assert channel._pending == {}
    ident, _, r = channel.run(["/bin/true"], None, {})
    message, fds = protocol.recv_message(theirs)
    assert message["id"] == ident and len(fds) == 1
    for fd in (r, *fds):
        os.close(fd)


def _pool(ready_file: Path) -> exec_server.SpawnerPool:
    return exec_server.SpawnerPool(spawner_file=str(AGENT / "spawner.py"), shm_bytes=1 << 20, uid=1234, gid=5678,
                                   hide=[], no_namespaces=True, score="0",
                                   ready_file=exec_server.ReadyFile(str(ready_file)))


def test_the_ready_file_is_allocated_whole_and_empty_before_any_spawner(tmp_path):
    ready = tmp_path / protocol.READY_NAME
    exec_server.ReadyFile(str(ready))
    assert ready.read_bytes() == b"\0" * protocol.READY_BYTES and ready.stat().st_blocks > 0
    assert protocol.parse_ready_record(ready.read_bytes()) is None


def _ends_quietly(pool: exec_server.SpawnerPool) -> list[str]:
    """Replace *pool*'s judgement of a spawner's end, which ends the server's process, so a test
    that ends a real spawner itself never ends the test process; returns the reasons seen."""
    seen: list[str] = []
    pool._gone = lambda proc, reason: seen.append(reason)
    return seen


def test_a_pool_names_its_spawner_in_the_ready_file_before_it_hands_the_spawner_out(tmp_path):
    ready = tmp_path / protocol.READY_NAME
    pool = _pool(ready)
    _ends_quietly(pool)
    channel = pool.ready()
    try:
        # Whole when the channel is handed out, and no file made beside it.
        assert exec_protocol_record(ready) == {"spawner_pid": channel.hello["pid"],
                                               "reaper_pid": channel.hello["reaper_pid"], "score": 0}
        assert sorted(p.name for p in tmp_path.iterdir()) == [protocol.READY_NAME]
    finally:
        pool._proc.kill()
        pool._proc.wait()


class _ServerExit(Exception):
    """``os._exit`` in the server, raised instead so the test can see its status."""


def test_a_ready_file_that_cannot_be_written_ends_the_server_before_any_command(tmp_path, monkeypatch):
    started = []
    real_popen = exec_server.subprocess.Popen
    monkeypatch.setattr(exec_server.subprocess, "Popen", lambda *a, **kw: started.append(real_popen(*a, **kw))
                        or started[-1])

    def failing(self, **_record):
        raise OSError(errno.EIO, "input/output error")      # not a full disk: still the server's end

    def exit_(status):
        raise _ServerExit(status)

    monkeypatch.setattr(exec_server.ReadyFile, "write", failing)
    monkeypatch.setattr(exec_server.os, "_exit", exit_)
    pool = _pool(tmp_path / protocol.READY_NAME)
    _ends_quietly(pool)
    with pytest.raises(_ServerExit) as ended:
        pool.ready()
    assert ended.value.args == (protocol.SPAWNER_GONE_EXIT,)
    assert pool._channel is None, "no command got the spawner the file never named"
    [spawner] = started
    assert spawner.wait(10) is not None, "the spawner it started was ended"


class _Ended:
    """A channel whose spawner had ended before the request reached it."""

    def run(self, argv, cwd, env):
        raise exec_server.ChannelGone("the spawner had ended")


class _Live:
    def run(self, argv, cwd, env):
        return 1, "record", 9


class _Answering:
    """A channel that answers each command at once: gone, as a spawner whose namespace ended under
    it does, or exited 0 with *output*."""

    def __init__(self, *, gone: bool, output: bytes = b"") -> None:
        self.gone = gone
        self.output = output
        self.closed = threading.Event()
        self.closed.set()
        self.forgotten: list[int] = []

    def run(self, argv, cwd, env):
        pending = exec_server._Pending()
        pending.gone = self.gone
        pending.status = None if self.gone else 0
        pending.started.set()
        pending.ended.set()
        r, w = os.pipe()
        os.write(w, self.output)
        os.close(w)
        return 1, pending, r

    def forget(self, ident):
        self.forgotten.append(ident)


def test_a_command_whose_namespace_ended_before_it_started_runs_once_in_a_fresh_one(tmp_path, monkeypatch):
    pool = _pool(tmp_path / protocol.READY_NAME)
    first, fresh = _Answering(gone=True), _Answering(gone=False, output=b"ran\n")
    handed = iter([first, fresh])
    monkeypatch.setattr(pool, "ready", lambda: next(handed))
    assert exec_server.execute(pool, "echo ran", None, None, None, LAUNCHES) == {"stdout": "ran", "stderr": None,
                                                                         "return_code": 0}
    assert first.forgotten == [1] and fresh.forgotten == [1]


def test_a_command_whose_fresh_namespace_ended_too_could_not_start(tmp_path, monkeypatch):
    pool = _pool(tmp_path / protocol.READY_NAME)
    monkeypatch.setattr(pool, "ready", lambda: _Answering(gone=True))
    out = exec_server.execute(pool, "echo ran", None, None, None, LAUNCHES)
    assert out["return_code"] == exec_server.NOT_STARTED_STATUS
    assert "ended before it started" in out["stdout"]


def test_a_command_that_reaches_an_ended_spawner_runs_once_on_a_fresh_one(tmp_path, monkeypatch):
    pool, live = _pool(tmp_path / protocol.READY_NAME), _Live()
    handed = iter([_Ended(), live])
    monkeypatch.setattr(pool, "ready", lambda: next(handed))
    assert pool.run(["/bin/true"], None, {}) == (live, 1, "record", 9)


def test_a_fresh_spawner_that_has_ended_too_is_the_servers_own_failure(tmp_path, monkeypatch):
    pool, handed = _pool(tmp_path / protocol.READY_NAME), []
    monkeypatch.setattr(pool, "ready", lambda: handed.append(_Ended()) or handed[-1])
    with pytest.raises(exec_server.ChannelGone):
        pool.run(["/bin/true"], None, {})
    assert len(handed) == 2, "tried once more, on a fresh spawner, and no more"


class _Refusing:
    """A pool whose every command fails with *error* before it is sent to a spawner."""

    def __init__(self, error: Exception) -> None:
        self.error = error

    def ready(self) -> None:
        return None

    def run(self, argv, cwd, env):
        raise self.error


@pytest.fixture
def serve(short_dir):
    """Serves a server in this process with a single request thread, so a request that ended its
    thread would leave the next one unserved; shut down when the test ends."""
    servers: list[exec_server.ExecServer] = []

    def start(pool) -> Path:
        path = short_dir / protocol.SOCKET_NAME
        servers.append(exec_server.bind_server(path, token=TOKEN, spawner=pool, cli_launches=LAUNCHES, threads=1))
        threading.Thread(target=servers[-1].serve_forever, daemon=True).start()
        return path

    yield start
    for s in servers:
        s.shutdown()
        s.server_close()


def _post(path: Path, body: dict) -> tuple[int, dict]:
    conn = _Unix(path, 10)
    try:
        conn.request("POST", "/exec", body=json.dumps(body).encode(), headers={protocol.TOKEN_HEADER: TOKEN})
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read())
    finally:
        conn.close()


def test_a_request_the_server_cannot_serve_is_answered_as_a_command_that_could_not_start(serve):
    """A command the server cannot hand to the spawner (here the system is out of open files)
    comes back as one that could not start: 126 with why, as the spawner answers a fork it cannot
    make. Never a dropped connection, which reads as a lost sandbox. The thread that served it
    serves the next request."""
    path = serve(_Refusing(OSError(errno.ENFILE, "Too many open files in system")))
    for _ in range(2):
        status, body = _post(path, {"command": "true"})
        assert (status, body["stderr"], body["return_code"]) == (200, None, 126)
        assert body["stdout"].startswith("hls-exec: cannot serve the command: ")
        assert "Too many open files in system" in body["stdout"]


def test_a_request_that_reached_two_ended_spawners_gets_no_reply(serve):
    """That is the server's own machinery failing, which is never answered as a command's end. The
    thread that took it serves the next request."""
    path = serve(_Refusing(exec_server.ChannelGone("the spawner had ended")))
    with pytest.raises((http.client.RemoteDisconnected, ConnectionResetError)):
        _post(path, {"command": "true"})
    conn = _Unix(path, 10)
    conn.request("GET", "/health", headers={protocol.TOKEN_HEADER: TOKEN})
    assert conn.getresponse().status == 200
    conn.close()


class _Exited:
    """A spawner process that has exited with *status*."""

    def __init__(self, status: int) -> None:
        self.status = status

    def wait(self, timeout=None) -> int:
        return self.status


def test_the_late_end_of_an_old_spawner_leaves_the_fresh_one_in_place(tmp_path):
    """A fresh spawner can start while the old one's end is still settling (a request saw the old
    channel closed and started it). Only the end of the pool's current spawner clears it."""
    pool = _pool(tmp_path / protocol.READY_NAME)
    old, fresh, channel = _Exited(protocol.REAPER_GONE_EXIT), _Exited(protocol.REAPER_GONE_EXIT), object()
    pool._channel, pool._proc = channel, fresh
    pool._gone(old, "it closed the channel")
    assert (pool._channel, pool._proc) == (channel, fresh)
    pool._gone(fresh, "it closed the channel")
    assert (pool._channel, pool._proc) == (None, None)


#: A command that reports its process limit and what a fork of its own gets.
_FORK = ("import errno, os, resource\n"
         "print('limit', *resource.getrlimit(resource.RLIMIT_NPROC))\n"
         "try:\n"
         "    pid = os.fork()\n"
         "except OSError as e:\n"
         "    print('fork', errno.errorcode[e.errno])\n"
         "else:\n"
         "    if pid == 0:\n"
         "        os._exit(0)\n"
         "    os.waitpid(pid, 0)\n"
         "    print('fork ok')\n")


def test_the_spawner_caps_each_commands_process_count_at_the_ceiling_it_was_given(start_server):
    """Every command's ``RLIMIT_NPROC``, soft and hard, is the ceiling the stage gave: past it a
    fork of the command's fails with EAGAIN, while the spawner and the server keep their own
    limit and run the next command. Run without namespaces, as here, ``RLIMIT_NPROC`` counts every
    task of the uid, which already runs more than these few, so the command's first fork is past
    it; in a sandbox, inside the commands' user namespace, which tasks it counts depends on the
    kernel (``hls.substrate.nproc``)."""
    import shlex

    server = start_server(command_nproc=3)
    out = server.run(f"{shlex.quote(sys.executable)} -I -c {shlex.quote(_FORK)}")
    assert (out["return_code"], out["stdout"]) == (0, "limit 3 3\nfork EAGAIN")
    again = server.run(f"{shlex.quote(sys.executable)} -I -c {shlex.quote(_FORK)}")
    assert again["stdout"] == "limit 3 3\nfork EAGAIN"
    assert server.run("echo next") == {"stdout": "next", "stderr": None, "return_code": 0}


def test_without_a_ceiling_a_command_keeps_the_servers_process_limit(start_server):
    import resource
    import shlex

    server = start_server()
    out = server.run(f"{shlex.quote(sys.executable)} -I -c {shlex.quote(_FORK)}")
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    assert out["stdout"] == f"limit {soft} {hard}\nfork ok"


def test_the_server_command_passes_the_ceiling_on():
    assert protocol.server_command(shm_bytes=1, uid=2, gid=3, cli_launches=("exec x ",))[-4:] == (
        "--hide", protocol.CONTROL_TARGET, "--cli-launch", "exec x ")
    command = protocol.server_command(shm_bytes=1, uid=2, gid=3, cli_launches=LAUNCHES, command_nproc=60_000)
    assert command[-2:] == ("--command-nproc", "60000")
    limited = protocol.server_command(shm_bytes=1, uid=2, gid=3, cli_launches=LAUNCHES, command_nproc=60_000,
                                      ucounts={"max_user_namespaces": 64, "max_inotify_instances": 8})
    assert limited[len(command):] == ("--ucount", "max_user_namespaces=64", "--ucount", "max_inotify_instances=8")


def test_the_server_gives_each_ucount_to_every_spawner_it_starts(start_server):
    server = start_server(ucounts=("max_user_namespaces=64", "max_inotify_instances=8"))
    argv = Path(f"/proc/{server.spawner_pid()}/cmdline").read_bytes().split(bytes(1))
    at = argv.index(b"--ucount")
    assert argv[at:at + 4] == [b"--ucount", b"max_user_namespaces=64", b"--ucount", b"max_inotify_instances=8"]


def test_a_server_without_a_token_refuses_to_start(start_server):
    s = start_server(env={"PATH": "/usr/bin:/bin"}, wait=False)
    assert s.proc.wait(10) == protocol.NO_TOKEN_EXIT
    assert not s.socket.exists()


def test_a_spawner_that_never_becomes_ready_leaves_no_socket(start_server):
    s = start_server(shm_bytes=0, wait=False)
    assert s.proc.wait(20) == protocol.SPAWNER_GONE_EXIT
    assert not s.socket.exists()


def test_the_sandbox_programs_import_only_the_standard_library():
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        f"sys.path.insert(0, {str(AGENT)!r})\n"
        "import exec_protocol, exec_server, spawner\n"
        "ours = {'exec_protocol', 'exec_server', 'spawner'}\n"
        "added = {m.split('.')[0] for m in set(sys.modules) - before} - ours\n"
        "print(sorted(m for m in added if m not in sys.stdlib_module_names))\n"
        "print(exec_server.protocol is exec_protocol, spawner.protocol is exec_protocol)\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], env={"PATH": "/usr/bin:/bin"}, capture_output=True,
                         text=True, check=True).stdout.splitlines()
    assert out == ["[]", "True True"]


def test_the_server_command_runs_the_bound_files_on_the_images_python():
    assert protocol.server_command(shm_bytes=8 << 30, uid=1234, gid=5678, cli_launches=LAUNCHES) == (
        "/usr/bin/python3", "-I", "/opt/hls-exec/exec_server.py", "--socket", "/run/hls-control/exec.sock",
        "--spawner", "/opt/hls-exec/spawner.py", "--shm-bytes", str(8 << 30), "--uid", "1234", "--gid", "5678",
        "--hide", "/run/hls-control", "--cli-launch", "exec claude ", "--cli-launch", "codex exec ")
    assert {Path(f).name for f in protocol.SANDBOX_FILES} == {"exec_protocol.py", "exec_server.py", "spawner.py"}
    assert all((AGENT / f).is_file() for f in protocol.SANDBOX_FILES)


def test_messages_carry_large_bodies_and_descriptors_and_end_cleanly():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    r, w = os.pipe()
    big = {"op": "spawn", "blob": "x" * (3 << 20)}
    sender = threading.Thread(target=protocol.send_message, args=(a, big), kwargs={"fds": (w,)})
    sender.start()
    message, fds = protocol.recv_message(b)
    sender.join()
    assert message == big and len(fds) == 1
    os.write(fds[0], b"through")
    assert os.read(r, 7) == b"through"
    for fd in (r, w, fds[0]):
        os.close(fd)
    a.close()
    assert protocol.recv_message(b) is None


def test_a_message_with_more_descriptors_than_the_protocol_allows_is_refused():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.settimeout(2)
    r1, w1 = os.pipe()
    r2, w2 = os.pipe()
    protocol.send_message(a, {"op": "spawn"}, fds=(w1, w2))
    with pytest.raises(ConnectionError, match="more descriptors"):
        protocol.recv_message(b)
    for fd in (r1, w1, r2, w2):
        os.close(fd)


def test_an_oversized_message_is_refused():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    b.settimeout(2)
    a.sendall((protocol.MAX_MESSAGE_BYTES + 1).to_bytes(4, "big"))
    with pytest.raises(ConnectionError, match="over"):
        protocol.recv_message(b)


def test_a_command_whose_text_embeds_an_end_line_logs_one_end(server):
    # The start line shows the command's text, which is the caller's; only the line the server
    # writes when the command ends starts with an end line.
    server.run(": hls-exec: end id=9 status=0 signal=- dropped=0 sec=1.0", cwd="/")
    [start] = [line for line in server.log() if line.startswith("hls-exec: start ")]
    assert "cmd=: hls-exec: end id=9" in start
    assert [e["id"] for e in map(end_fields, server.log()) if e is not None] == [1]


def test_parse_start_line_reads_only_start_lines():
    assert protocol.parse_start_line("hls-exec: start id=3 pid=4 cli=1 cwd=None cmd=a b cmd=c") == {
        "id": 3, "pid": 4, "cli": True, "cwd": "None", "cmd": "a b cmd=c"}
    assert protocol.parse_start_line("hls-exec: end id=3 status=0 signal=- dropped=0 sec=1.0") is None
    assert protocol.parse_start_line("x hls-exec: start id=3 pid=4 cli=1 cwd=/ cmd=x") is None




def test_the_server_counts_the_launches_it_was_given_and_no_other(start_server):
    """The stage gives the server the arm's launch marker (``HarnessAdapter.cli_launch``): a
    command that holds it is a launch of the CLI, whatever harness it is, and one that holds only
    another harness's is not."""
    server = start_server(launches=("exec fakecli ",))
    server.run("true || exec fakecli --go")
    server.run("true || exec claude --verbose")
    starts = [protocol.parse_start_line(line) for line in server.log() if line.startswith("hls-exec: start ")]
    assert [start["cli"] for start in starts] == [True, False]
    assert protocol.is_cli_launch("x; exec fakecli --go", ("exec fakecli ",))
    assert not protocol.is_cli_launch("x; exec fakecli --go", ())


def test_a_server_given_no_launch_marker_refuses_to_start(start_server):
    """Without a launch marker no start line could say a CLI launched, and every failure of the
    agent's part would be taken for the harness's: the server does not start."""
    server = start_server(launches=(), wait=False)
    assert server.proc.wait(20) == 2
    assert "--cli-launch" in server.err.read_text()
    with pytest.raises(ValueError, match="at least one launch marker"):
        protocol.server_command(shm_bytes=1, uid=2, gid=3, cli_launches=())
