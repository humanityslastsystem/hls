"""The ``budget`` MCP server inside the build sandbox: a stdio JSON-RPC server with one tool,
``get_budget``, that asks the runner's budget service and passes its answer on.

It runs on the image's ``/usr/bin/python3 -IS`` with the standard library only. The stage writes
an executable copy (``executable_text``) into node-local scratch and binds it read-only at
``MCP_TARGET``, so both harnesses register it as a bare command with no arguments: Claude Code
in ``.claude.json``, and codex in its ``config.toml``, whose ``command`` takes one program.

It keeps no state and computes nothing. Each call is one request on ``budget.sock`` in the
channel directory the sandbox sees read-only at ``CHANNEL_TARGET``. The tool's name, the
server's name and the fields the agent reads (``dimensions.wallclock.{limit, spent,
remaining, over_budget}``) are the ones the brief and the goal prompts name. Protocol details
the clients rely on:
- notifications get no reply, whatever their method;
- ``resources/list``, ``resources/templates/list`` and ``prompts/list`` are answered with
  empty lists and declared in the handshake, because a client that gets "method not found"
  for them reports the server unavailable;
- a service that cannot be reached is a tool result carrying an ``error``, never a failed
  call, so the session goes on."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

#: The MCP protocol version the server speaks.
PROTOCOL = "2024-11-05"
#: Where the executable copy is bound in the sandbox.
MCP_TARGET = "/opt/hls-budget/get-budget-mcp"
#: Where the channel directory is bound in the sandbox.
CHANNEL_TARGET = "/run/hls"
#: The budget service's socket in the channel directory.
SOCKET_NAME = "budget.sock"
#: The socket the server asks.
SOCKET_PATH = f"{CHANNEL_TARGET}/{SOCKET_NAME}"
#: The flags of the executable copy's first line, after the image's interpreter
#: (``executable_text``): isolated from the environment's Python variables and the
#: user's site directory (-I), and without the ``site`` module (-S), so no ``.pth`` file or
#: ``sitecustomize`` of the image's Python runs in it.
SHEBANG_FLAGS = "-IS"
#: The note of an error result, as the service's own responses carry it.
NOTE = "Your budget for this build. Pace your work to stay within it."
#: Seconds to wait for the service's answer.
CALL_TIMEOUT_SEC = 10.0
#: The one tool.
TOOL = {"name": "get_budget",
        "description": "Return your build budget: limit, spent, and remaining for cost, "
                       "turns, and wall-clock time. Consult it to pace your work.",
        "inputSchema": {"type": "object", "properties": {}}}
#: Listing methods answered with an empty collection, by the result's key.
EMPTY_LISTINGS = {"resources/list": "resources", "resources/templates/list": "resourceTemplates",
                  "prompts/list": "prompts"}


def executable_text(python: str) -> str:
    """The executable copy the stage writes: this file under a first line that runs it on
    *python*, the image's system interpreter (``hls.agent.exec_protocol.SANDBOX_PYTHON``), with
    ``SHEBANG_FLAGS``. The caller names the interpreter, since this file imports nothing of hls."""
    return f"#!{python} {SHEBANG_FLAGS}\n" + Path(__file__).read_text()


def ask(socket_path: str) -> dict:
    """The service's answer to ``get_budget``, or an error result."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(CALL_TIMEOUT_SEC)
            conn.connect(socket_path)
            conn.sendall(b'{"op": "get_budget"}\n')
            with conn.makefile("rb") as reader:
                line = reader.readline()
        reply = json.loads(line)
        if not isinstance(reply, dict):
            raise ValueError("the reply is not an object")
        return reply
    except (OSError, ValueError) as exc:
        return {"dimensions": {}, "note": NOTE, "error": f"budget service unavailable: {exc}"[:200]}


def handle(msg: dict, socket_path: str) -> dict | None:
    """The reply to one message, or None for a notification."""
    mid, method = msg.get("id"), msg.get("method")
    if "id" not in msg:
        return None
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": mid, "result": {
            "protocolVersion": PROTOCOL, "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
            "serverInfo": {"name": "budget", "version": "1.0.0"}}}
    if method in EMPTY_LISTINGS:
        return {"jsonrpc": "2.0", "id": mid, "result": {EMPTY_LISTINGS[method]: []}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": [TOOL]}}
    if method == "tools/call" and (msg.get("params") or {}).get("name") == TOOL["name"]:
        text = json.dumps(ask(socket_path), indent=2)
        return {"jsonrpc": "2.0", "id": mid, "result": {"content": [{"type": "text", "text": text}]}}
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": "method not found"}}


def main(argv: list[str]) -> int:
    """Serve stdin until it closes. ``--socket PATH`` replaces ``SOCKET_PATH`` (tests)."""
    socket_path = argv[argv.index("--socket") + 1] if "--socket" in argv else SOCKET_PATH
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        if not isinstance(msg, dict):
            continue
        reply = handle(msg, socket_path)
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
