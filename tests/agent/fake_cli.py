"""Fake ``claude`` and ``codex`` programs for the agent build's tests, driven by a script.

``install(bin_dir, control)`` writes both into *bin_dir*; *control* is a directory where the test
writes ``script.json`` and the programs write what they did. Each invocation takes the next step
of ``script.json``'s ``steps`` for its program (the last step repeats once they run out); the
preflight's invocation is told apart by its configuration directory and takes ``preflight``
instead. A step may:
- ``commit``: commit ``server.py`` with that text on ``main`` in the working directory, with
  git's automatic maintenance off;
- ``commit_files``: with ``commit``, more files to commit beside it, by name to text;
- ``advance``: add that many seconds to ``offset``, which the test's clock reads, so the
  runner's clock moves while the agent works;
- ``turns``: write that many assistant turns to the session transcript, of ``kind`` "work"
  (each different), "repeat" (one text again and again) or "invalid" (each tool call refused
  with an InputValidationError);
- ``stamp``: the timestamp every transcript line of the step carries;
- ``leak``: write that variable's value into a file of the agent's logs, as an agent printing
  its environment would;
- ``say``: print that variable's value on standard output, which harbor reads back as the
  command's output;
- ``link``: plant a symlink to that path in the agent's logs;
- ``write``: write that many bytes into ``blob.bin`` in the working directory;
- ``nested``: write about that many bytes of nested JSON lines (``{"type":"user","a":[[],...]}``,
  each under harbor's line cap) into a second session file of the agent's logs: the transcript
  harbor's reader holds at many times its size;
- ``fill``: write each host path of that object with that many bytes;
- ``mkdirs``: make each host directory of that list, with its parents;
- ``deep``: make that many directories, each in the last, under ``deep`` in the working
  directory;
- ``files``: make that many empty files in ``many`` in the working directory;
- ``chmodrace``: make that directory, relative to the working directory, and start a child that
  takes its owner's permissions away again and again until killed;
- ``touch``: write each host path of that object with its text, as the step runs (a test's fake
  cgroup files);
- ``server``: start a test server, a child that sleeps until killed, and write its pid into
  ``server.pid``;
- ``stderr``: print that line on standard error;
- ``sleep``: sleep that many seconds before ending;
- ``late``: then sleep ``after`` seconds and commit ``server.py`` with the text ``commit`` on
  ``main`` again, as an agent still working past the end of its part would;
- ``hang``: sleep until killed;
- ``kill_server``: SIGKILL the sandbox's exec server, its ancestor, and write the time into
  ``killed_at``;
- ``chmod``: set each host path of that object to its mode (a number), as the step runs;
- ``model``: report that model in place of the one it was given;
- ``oom_self``: end by SIGKILL, as an OOM kill of the CLI would, before its result event;
- ``exit``: exit with that status (0 by default).
Claude Code's stream gets an ``init`` event naming the model and the MCP servers registered in
``$CLAUDE_CONFIG_DIR/.claude.json``, and a ``result`` event whose ``modelUsage`` reports the
script's ``window`` for the model. Codex writes a session rollout under ``$CODEX_HOME/sessions``
with the model in its ``turn_context`` and ``window`` in its ``task_started`` event. Every
invocation appends one line to ``calls.jsonl``: the program, its arguments, its working
directory and what is in it, its standard input's start, named variables (the credential only as
whether it is set) and its process limit (``RLIMIT_NPROC``, soft and hard; -1 for unlimited)."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

_PROGRAM = r'''
import json, os, resource, signal, subprocess, sys, time, uuid
from pathlib import Path

CONTROL = Path(@CONTROL@)
PROGRAM = @PROGRAM@
if sys.argv[1:] == ["--version"]:
    print("2.1.251 (Claude Code)" if PROGRAM == "claude" else "codex-cli 0.146.0")
    sys.exit(0)
script = json.loads((CONTROL / "script.json").read_text())
args = sys.argv[1:]
stdin = sys.stdin.read() if not sys.stdin.isatty() else ""
config = os.environ.get("CLAUDE_CONFIG_DIR" if PROGRAM == "claude" else "CODEX_HOME", "")
preflight = "/logs/preflight/" in config
counter = CONTROL / f"{PROGRAM}.count"
n = 0 if preflight else int(counter.read_text()) if counter.exists() else 0
if not preflight:
    counter.write_text(str(n + 1))
steps = script.get(PROGRAM, [{}])
step = script.get("preflight", {}) if preflight else steps[min(n, len(steps) - 1)]
names = ("ANTHROPIC_MODEL", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CODE_AUTO_COMPACT_WINDOW", "ANTHROPIC_BASE_URL",
         "OPENAI_BASE_URL", "TORCH_CUDA_ARCH_LIST", "HF_HUB_OFFLINE")
record = {"program": PROGRAM, "preflight": preflight, "n": n, "args": args, "cwd": os.getcwd(),
          "listing": sorted(os.listdir(".")), "stdin": stdin[:200], "env": {k: os.environ.get(k) for k in names},
          "credential_set": bool(os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
                                 or os.environ.get("OPENAI_API_KEY")),
          "token_seen": "HLS_EXEC_TOKEN" in os.environ,
          "nproc": [-1 if v == resource.RLIM_INFINITY else v for v in resource.getrlimit(resource.RLIMIT_NPROC)]}
cfg = Path(config) / (".claude.json" if PROGRAM == "claude" else "config.toml")
record["config"] = cfg.read_text() if cfg.is_file() else None
with open(CONTROL / "calls.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")

if step.get("advance"):
    offset = CONTROL / "offset"
    now = float(offset.read_text()) if offset.exists() else 0.0
    offset.write_text(str(now + step["advance"]))
def commit(text, files, message):
    # No automatic maintenance: from git 2.54 a commit can start a detached repack, which could
    # still be writing the store when the commit point reads it.
    git_env = {**os.environ, "GIT_AUTHOR_NAME": "agent", "GIT_AUTHOR_EMAIL": "a@x", "GIT_COMMITTER_NAME": "agent",
               "GIT_COMMITTER_EMAIL": "a@x", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "maintenance.auto", "GIT_CONFIG_VALUE_0": "false",
               "GIT_CONFIG_KEY_1": "gc.auto", "GIT_CONFIG_VALUE_1": "0"}
    if not Path(".git").exists():
        subprocess.run(["git", "init", "-q", "-b", "main"], check=True, env=git_env)
    Path("server.py").write_text(text)
    for name, body in files.items():
        Path(name).write_text(body)
    subprocess.run(["git", "add", "server.py", *files], check=True, env=git_env)
    subprocess.run(["git", "commit", "-q", "-m", message], check=True, env=git_env)

if step.get("commit"):
    commit(step["commit"], step.get("commit_files", {}), f"step {n}")

def ancestors():
    pid = os.getpid()
    while pid > 1:
        stat = Path(f"/proc/{pid}/stat").read_text()
        pid = int(stat.rsplit(")", 1)[1].split()[1])
        yield pid

if step.get("kill_server"):
    for pid in ancestors():
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            continue
        if b"exec_server.py" in cmd:
            (CONTROL / "killed_at").write_text(str(time.time()))
            os.kill(pid, signal.SIGKILL)
            break

model = step.get("model") or (os.environ.get("ANTHROPIC_MODEL") if PROGRAM == "claude"
                              else args[args.index("--model") + 1])
session = "4f2a1c5e-0000-4000-8000-00000000000" + str(min(n, 9))
if "--resume" in args:
    session = args[args.index("--resume") + 1]

def turns(kind, count):
    lines = []
    for i in range(count):
        text = "I will keep going." if kind == "repeat" else f"Turn {n}.{i}: working."
        lines.append({"type": "assistant", "timestamp": step.get("stamp", f"2026-10-02T00:{n:02d}:{i:02d}Z"),
                      "message": {"id": f"m-{n}-{i}", "role": "assistant",
                                  "content": [{"type": "text", "text": text},
                                              {"type": "tool_use", "name": "Bash", "input": {"command": "true"}}]}})
        result = "<tool_use_error>InputValidationError: Bash was called with input that could not be parsed"
        lines.append({"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "content": result if kind == "invalid" else "ok", "is_error": kind == "invalid"}]}})
    return lines

if PROGRAM == "claude":
    servers = []
    if record["config"]:
        servers = [{"name": k, "status": "connected"} for k in json.loads(record["config"]).get("mcpServers", {})]
    print(json.dumps({"type": "system", "subtype": "init", "model": model, "session_id": session,
                      "mcp_servers": servers}), flush=True)
    projects = Path(config) / "projects" / "-workspace"
    projects.mkdir(parents=True, exist_ok=True)
    with open(projects / f"{session}.jsonl", "a") as f:
        for line in turns(step.get("kind", "work"), step.get("turns", 1)):
            f.write(json.dumps(line) + "\n")
else:
    tid = "0199aaaa-0000-7000-8000-000000000001"
    if "resume" not in args:
        print(json.dumps({"type": "thread.started", "thread_id": tid}), flush=True)
    day = Path(config) / "sessions" / "2026" / "10" / "02"
    day.mkdir(parents=True, exist_ok=True)
    with open(day / f"rollout-2026-10-02T00-00-00-{tid}.jsonl", "a") as f:
        f.write(json.dumps({"type": "turn_context", "payload": {"model": model}}) + "\n")
        f.write(json.dumps({"type": "event_msg", "payload": {"type": "task_started",
                                                            "model_context_window": script["window"]}}) + "\n")
        for i in range(step.get("turns", 1)):
            f.write(json.dumps({"type": "response_item", "timestamp": f"2026-10-02T00:{n:02d}:{i:02d}Z",
                                "payload": {"type": "message", "role": "assistant",
                                            "content": [{"type": "output_text", "text": f"turn {n}.{i}"}]}}) + "\n")
logs = Path(config).parent if PROGRAM == "claude" else None
if step.get("leak") and logs is not None:
    (logs / "printed-env.txt").write_text(f"KEY={os.environ.get(step['leak'], '')}\n")
if step.get("link") and logs is not None:
    os.symlink(step["link"], logs / "planted-link")
if step.get("server"):
    server = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    (CONTROL / "server.pid").write_text(str(server.pid))
for path, text in step.get("touch", {}).items():
    Path(path).write_text(text)
if step.get("say"):
    print(f"env {step['say']}={os.environ.get(step['say'], '')}", flush=True)
for path in step.get("mkdirs", []):
    Path(path).mkdir(parents=True, exist_ok=True)
for path, size in step.get("fill", {}).items():
    with open(path, "wb") as f:
        f.write(b"x" * size)
if step.get("deep"):
    Path("deep", *(["d"] * step["deep"])).mkdir(parents=True)
if step.get("files"):
    Path("many").mkdir()
    for i in range(step["files"]):
        (Path("many") / str(i)).touch()
if step.get("chmodrace"):
    target = Path(step["chmodrace"]).resolve()
    (target / "inner").mkdir(parents=True)
    racer = subprocess.Popen([sys.executable, "-c", "import os, sys\nwhile True:\n    os.chmod(sys.argv[1], 0)",
                              str(target)])
    (CONTROL / "racer.pid").write_text(str(racer.pid))
if step.get("nested") and PROGRAM == "claude":
    crafted = ('{"type":"user","a":[' + ",".join(["[]"] * 300_000) + ']}\n').encode()
    with open(Path(config) / "projects" / "-workspace" / "crafted.jsonl", "wb") as f:
        written = 0
        while written < step["nested"]:
            f.write(crafted)
            written += len(crafted)
if step.get("write"):
    with open("blob.bin", "wb") as f:
        f.write(b"x" * step["write"])
        f.flush()
        os.fsync(f.fileno())
for path, mode in step.get("chmod", {}).items():
    os.chmod(path, mode)
if step.get("stderr"):
    print(step["stderr"], file=sys.stderr, flush=True)
if step.get("sleep"):
    time.sleep(step["sleep"])
if step.get("late"):
    time.sleep(step["late"]["after"])
    commit(step["late"]["commit"], {}, f"step {n}, late")
if step.get("hang"):
    while True:
        time.sleep(60)
if step.get("oom_self"):
    os.kill(os.getpid(), signal.SIGKILL)
if PROGRAM == "claude":
    usage = {model: {"contextWindow": script["window"], "maxOutputTokens": 32000}}
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": session,
                      "total_cost_usd": 0.5, "num_turns": step.get("turns", 1), "modelUsage": usage}), flush=True)
sys.exit(step.get("exit", 0))
'''


def install(bin_dir: Path, control: Path) -> None:
    """Write the fake ``claude`` and ``codex`` into *bin_dir*, reading and writing *control*."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    for program in ("claude", "codex"):
        body = _PROGRAM.replace("@CONTROL@", repr(str(control))).replace("@PROGRAM@", repr(program))
        path = bin_dir / program
        path.write_text(f"#!{shlex.quote(sys.executable)}\n{body}")
        path.chmod(0o755)
