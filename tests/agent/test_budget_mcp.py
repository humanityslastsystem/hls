from __future__ import annotations

import ast
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import hls.budget.mcp as mcp
from hls.agent.exec_protocol import SANDBOX_PYTHON
import hls.budget.service as service
from hls.budget.service import BudgetService, WallClock

MCP_FILE = Path(mcp.__file__)
BANNED = ("soft", "advisory", "not enforced", "unenforced")


class _Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def live(short_dir):
    channel = short_dir / "ch"
    channel.mkdir(mode=0o700)
    moved = _Clock()
    clock = WallClock(limit_sec=43200, hard_sec=45000, clock=moved)
    svc = BudgetService(clock, channel / mcp.SOCKET_NAME)
    svc.start()
    yield svc, clock, moved
    svc.stop()


class _Server:
    """The MCP server as the harness runs it: its own process on an isolated interpreter
    with no hls on the path, spoken to over stdio."""

    def __init__(self, program: list[str]) -> None:
        self.proc = subprocess.Popen(program, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     env={"PATH": "/usr/bin:/bin"}, cwd="/")

    def send(self, msg: dict) -> None:
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()

    def rpc(self, msg: dict) -> dict:
        self.send(msg)
        return json.loads(self.proc.stdout.readline())

    def close(self) -> str:
        self.proc.stdin.close()
        rest = self.proc.stdout.read()
        self.proc.wait(timeout=5)
        return rest


def _start(socket_path: Path | str) -> _Server:
    return _Server([sys.executable, "-I", str(MCP_FILE), "--socket", str(socket_path)])


def test_the_whole_handshake_and_a_call_against_the_live_service(live):
    svc, clock, moved = live
    clock.start()
    moved.now += 3725.6
    server = _start(svc.socket_path)
    init = server.rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2024-11-05", "capabilities": {}}})
    assert init["result"] == {"protocolVersion": "2024-11-05",
                              "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                              "serverInfo": {"name": "budget", "version": "1.0.0"}}
    server.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    tools = server.rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert tools["result"]["tools"] == [mcp.TOOL]
    assert mcp.TOOL["name"] == "get_budget" and mcp.TOOL["inputSchema"] == {"type": "object", "properties": {}}
    call = server.rpc({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                       "params": {"name": "get_budget", "arguments": {}}})
    [content] = call["result"]["content"]
    assert content["type"] == "text"
    assert json.loads(content["text"]) == clock.response()
    assert json.loads(content["text"])["dimensions"]["wallclock"]["spent"] == 3725
    assert server.close() == ""


def test_the_tool_is_get_budget_with_its_description():
    """The goal prompts and the evaluator read this tool by name; the agent reads its
    description."""
    assert mcp.TOOL == {"name": "get_budget",
                        "description": "Return your build budget: limit, spent, and remaining for cost, turns, "
                                       "and wall-clock time. Consult it to pace your work.",
                        "inputSchema": {"type": "object", "properties": {}}}


def test_listings_are_answered_empty(live):
    svc, _, _ = live
    server = _start(svc.socket_path)
    assert server.rpc({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})["result"] == {"resources": []}
    assert server.rpc({"jsonrpc": "2.0", "id": 2, "method": "resources/templates/list"})["result"] == {
        "resourceTemplates": []}
    assert server.rpc({"jsonrpc": "2.0", "id": 3, "method": "prompts/list"})["result"] == {"prompts": []}
    server.close()


def test_notifications_get_no_reply_whatever_their_method(live):
    svc, _, _ = live
    server = _start(svc.socket_path)
    for method in ("notifications/initialized", "notifications/cancelled", "notifications/roots/list_changed",
                   "tools/list", "made/up"):
        server.send({"jsonrpc": "2.0", "method": method})
    server.send({"jsonrpc": "2.0", "id": 9, "method": "tools/list"})
    assert json.loads(server.proc.stdout.readline())["id"] == 9
    assert server.close() == ""


def test_an_unknown_method_or_tool_is_method_not_found(live):
    svc, _, _ = live
    server = _start(svc.socket_path)
    for i, msg in enumerate(({"method": "tools/describe"},
                             {"method": "tools/call", "params": {"name": "set_budget", "arguments": {}}})):
        reply = server.rpc({"jsonrpc": "2.0", "id": i, **msg})
        assert reply["error"] == {"code": -32601, "message": "method not found"}
    server.close()


def test_lines_that_are_not_messages_are_skipped(live):
    svc, _, _ = live
    server = _start(svc.socket_path)
    server.proc.stdin.write("not json\n\n[1, 2]\n")
    reply = server.rpc({"jsonrpc": "2.0", "id": 4, "method": "tools/list"})
    assert reply["id"] == 4
    server.close()


def test_a_service_that_is_gone_is_an_error_result_not_a_failed_call(short_dir):
    server = _start(short_dir / "nothing.sock")
    call = server.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_budget"}})
    payload = json.loads(call["result"]["content"][0]["text"])
    assert payload["dimensions"] == {} and payload["note"] == mcp.NOTE
    assert payload["error"].startswith("budget service unavailable")
    assert len(payload["error"]) <= 200
    server.close()


def test_a_service_answering_garbage_is_an_error_result(short_dir):
    path = short_dir / "bad.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)

    def serve():
        conn, _ = listener.accept()
        with conn:
            conn.recv(100)
            conn.sendall(b"[]\n")

    t = threading.Thread(target=serve)
    t.start()
    try:
        server = _start(path)
        call = server.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_budget"}})
        assert json.loads(call["result"]["content"][0]["text"])["error"].startswith("budget service unavailable")
        server.close()
    finally:
        t.join(5)
        listener.close()


def test_the_shipped_socket_is_the_channel_target():
    assert mcp.SOCKET_PATH == "/run/hls/budget.sock"
    assert mcp.MCP_TARGET == "/opt/hls-budget/get-budget-mcp"


def test_the_note_is_the_services():
    assert mcp.NOTE == service.NOTE


def test_the_executable_copy_runs_under_its_shebang_with_the_stdlib_only(live, tmp_path):
    svc, clock, _ = live
    if not Path("/usr/bin/python3").exists():
        pytest.skip("no /usr/bin/python3 on this machine")
    copy = tmp_path / "get-budget-mcp"
    copy.write_text(mcp.executable_text(SANDBOX_PYTHON))
    copy.chmod(0o755)
    assert copy.read_text().splitlines()[0] == "#!/usr/bin/python3 -IS"
    server = _Server([str(copy), "--socket", str(svc.socket_path)])
    call = server.rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_budget"}})
    assert json.loads(call["result"]["content"][0]["text"]) == clock.response()
    server.close()


def _imported_modules(import_time_log: str) -> set[str]:
    """The module names in the stderr of ``python -X importtime``."""
    return {line.rsplit("|", 1)[1].strip() for line in import_time_log.splitlines() if "|" in line}


def test_the_shebang_keeps_the_systems_site_files_from_running(live):
    """-S: no .pth file or sitecustomize of the image's Python runs in the server, whose
    imports are all the standard library's. Run the shim on the interpreter its own first line
    names, and see that it answers and that ``site`` was never imported."""
    svc, clock, _ = live
    shebang = mcp.executable_text(SANDBOX_PYTHON).splitlines()[0]
    assert shebang == "#!/usr/bin/python3 -IS"
    program = shlex.split(shebang.removeprefix("#!"))
    if not Path(program[0]).exists():
        pytest.skip(f"no {program[0]} on this machine")
    proc = subprocess.run([*program, "-X", "importtime", str(MCP_FILE), "--socket", str(svc.socket_path)],
                          input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                            "params": {"name": "get_budget"}}) + "\n",
                          capture_output=True, text=True, timeout=10, env={"PATH": "/usr/bin:/bin"}, cwd="/")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(json.loads(proc.stdout)["result"]["content"][0]["text"]) == clock.response()
    modules = _imported_modules(proc.stderr)
    assert {"json", "socket"} <= modules
    assert not modules & {"site", "sitecustomize", "usercustomize"}


def test_a_service_that_never_answers_ends_the_call_with_the_timeout_error(short_dir, monkeypatch):
    """A socket that is bound and listening but never accepts takes the connection and the
    request and then says nothing, as a hung service would."""
    path = short_dir / "mute.sock"
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(1)
    monkeypatch.setattr(mcp, "CALL_TIMEOUT_SEC", 0.3)
    replies: list[dict] = []
    caller = threading.Thread(target=lambda: replies.append(mcp.ask(str(path))), daemon=True)
    started = time.monotonic()
    caller.start()
    caller.join(5)
    elapsed = time.monotonic() - started
    hung = caller.is_alive()
    listener.close()
    assert not hung, "ask is still waiting for a service that never answers"
    assert 0.2 <= elapsed < 3
    assert replies == [{"dimensions": {}, "note": mcp.NOTE, "error": "budget service unavailable: timed out"}]


def test_the_server_imports_nothing_outside_the_standard_library():
    tree = ast.parse(MCP_FILE.read_text())
    names = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
             for alias in node.names}
    names |= {node.module.split(".")[0] for node in ast.walk(tree)
              if isinstance(node, ast.ImportFrom) and node.module}
    assert names <= set(sys.stdlib_module_names) | {"__future__"}, names


@pytest.mark.parametrize("path", [MCP_FILE, Path(service.__file__)], ids=["mcp", "service"])
def test_no_file_frames_the_budget_as_optional(path):
    """The agent reads the MCP server's source in its sandbox, and the service writes every
    answer it gets."""
    text = path.read_text().lower()
    for banned in BANNED:
        assert banned not in text, f"{path.name} says {banned!r}"


def test_the_running_server_ignores_python_variables(live):
    """-I keeps PYTHONPATH and PYTHONSTARTUP from changing what runs."""
    svc, clock, _ = live
    planted = svc.socket_path.parent.parent / "planted"
    planted.mkdir()
    (planted / "json.py").write_text("raise SystemExit('planted json imported')\n")
    proc = subprocess.run([sys.executable, "-I", str(MCP_FILE), "--socket", str(svc.socket_path)],
                          input=json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}) + "\n",
                          capture_output=True, text=True, timeout=10,
                          env={**os.environ, "PYTHONPATH": str(planted)})
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["result"]["tools"] == [mcp.TOOL]
