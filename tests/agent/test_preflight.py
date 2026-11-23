"""The preflight: the readback parsers on the CLIs' own event shapes, the judgment, and one request
through a real exec server (its spawner without namespaces) to fake ``claude`` and ``codex``
programs that record how they were run, so the whole command path is exercised. The commands are
checked against the ones harbor's adapters run in the build itself, so the two cannot drift.

The event lines are cut down from real Claude Code streams and a codex 0.146.0 session rollout,
keeping key names and numbers only."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys
import tomllib
from pathlib import Path

import pytest
from harbor.agents.installed.codex import Codex
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext

from hls.agent import preflight as pf
from hls.agent.environment import SandboxEnvironment, SandboxLost
from hls.agent.harnesses import HlsCodex, budget_server, build_agent, harness_env_leaks
from hls.arms import ArmError, ArmRecord, Route, adapter_for
from hls.budget.mcp import MCP_TARGET
from hls.budget.service import WallClock
from tests.agent.test_exec_server import TOKEN, _Server

OPUS = ArmRecord(name="arm-a", harness="claude-code", model="claude-opus-4-8", context_window=1_000_000,
                 output_limit=64_000, effort="max", route="anthropic", route_table=Route(credential="anthropic_key"))
K3 = ArmRecord(name="k3", harness="claude-code", model="k3", context_window=200_000, output_limit=64_000,
               effort="max", route="kimi",
               route_table=Route(credential="kimi_token", base_url="kimi_url", auth="bearer"))
GPT = ArmRecord(name="arm-b", harness="codex", model="gpt-5.5", context_window=272_000, output_limit=128_000,
                effort="xhigh", route="openai_relay",
                route_table=Route(credential="openai_key", base_url="openai_url"))
KEY = "sk-test-0123456789"
# The relays' base URLs, secrets like the credential, and the host they share.
HOST = "relay.example"
ANTHROPIC_URL = f"https://{HOST}/anthropic"
OPENAI_URL = f"https://{HOST}/v1"
# The value the stage holds for each variable a test arm's route grants.
VALUES = {"ANTHROPIC_API_KEY": KEY, "ANTHROPIC_AUTH_TOKEN": KEY, "ANTHROPIC_BASE_URL": ANTHROPIC_URL,
          "OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": OPENAI_URL}

# Cut from a Claude Code stream: the init event and the last result.
K3_INIT = {"type": "system", "subtype": "init", "model": "k3", "claude_code_version": "2.1.251",
           "mcp_servers": [{"name": "budget", "status": "connected"}]}
K3_RESULT = {"type": "result", "subtype": "success", "is_error": False, "num_turns": 688,
             "terminal_reason": "completed",
             "modelUsage": {"k3": {"inputTokens": 1213176, "outputTokens": 826187, "contextWindow": 200000,
                                   "maxOutputTokens": 32000, "canonicalModel": "k3",
                                   "provider": "firstParty"}}}
# Cut from a Claude Code stream: the model string with its window suffix keys the entry.
OPUS_RESULT = {"type": "result", "subtype": "success", "is_error": False,
               "modelUsage": {"claude-opus-4-8[1m]": {"contextWindow": 1000000, "maxOutputTokens": 64000,
                                                       "canonicalModel": "claude-opus-4-8"}}}
# Cut from a codex session rollout: each invocation's start, and a turn.
GPT_STARTED = {"timestamp": "2026-09-24T08:51:34.000Z", "type": "event_msg",
               "payload": {"type": "task_started", "turn_id": "01a0d29c-ed18-7ee0-94b3-bad9f1c0509e",
                           "started_at": 1790239894, "model_context_window": 258400,
                           "collaboration_mode_kind": "default"}}
GPT_TURN = {"timestamp": "2026-09-24T08:51:35.000Z", "type": "turn_context",
            "payload": {"turn_id": "01a0d29c-ed18-7ee0-94b3-bad9f1c0509e", "cwd": "/workspace",
                        "model": "gpt-5.5", "effort": "xhigh", "summary": "auto"}}
# Token counts carry the window too; the readback does not use them.
GPT_TOKENS = {"timestamp": "2026-09-24T09:02:00.000Z", "type": "event_msg",
              "payload": {"type": "token_count",
                          "info": {"last_token_usage": {"input_tokens": 203405, "output_tokens": 268},
                                   "model_context_window": 1}}}


@pytest.fixture(autouse=True)
def _no_runner_settings(monkeypatch):
    """The runner's environment holds none of the variables harbor would read, as the stage
    requires before a build."""
    for var in harness_env_leaks(os.environ):
        monkeypatch.delenv(var)


def _lines(*events) -> str:
    return "\n".join(e if isinstance(e, str) else json.dumps(e) for e in events) + "\n"


def _agent(arm: ArmRecord, tmp_path: Path, *, mcp: bool = True, goal: bool = True, values: dict | None = None):
    values = {**VALUES, **(values or {})}
    return build_agent(arm, logs_dir=tmp_path / "logs", clock=WallClock(limit_sec=3600, hard_sec=3900),
                       mcp=mcp, skills_dir=None, goal=goal, max_turns=None, emit=lambda kind, **f: None,
                       secrets={var: values[var] for var in adapter_for(arm).grants(arm.route_table)})


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def test_claude_code_reads_back_the_window_of_the_model_it_was_given():
    stream = _lines(K3_INIT, "a line that is not JSON", {"type": "assistant"}, K3_RESULT)
    assert pf.claude_window(stream, "k3") == 200_000
    assert pf.claude_window(_lines(OPUS_RESULT), "claude-opus-4-8[1m]") == 1_000_000
    # The key is the exact string: the bare id, or another arm's window, is not the arm's entry.
    assert pf.claude_window(_lines(OPUS_RESULT), "claude-opus-4-8") is None
    assert pf.claude_init(stream) == K3_INIT and pf.claude_budget_tool(K3_INIT) == "connected"


def test_the_last_result_counts_and_other_models_entries_are_ignored():
    side = {**K3_RESULT, "modelUsage": {"claude-haiku-4-5": {"contextWindow": 200000},
                                        "k3": {"contextWindow": 1000000}}}
    assert pf.claude_window(_lines(K3_RESULT, side), "k3") == 1_000_000
    assert pf.claude_window(_lines({**K3_RESULT, "modelUsage": {"claude-haiku-4-5": {"contextWindow": 1}}}),
                            "k3") is None
    assert pf.claude_window(_lines(K3_INIT), "k3") is None


@pytest.mark.parametrize("servers", [[], [{"name": "budget", "status": "failed"}], None],
                         ids=["none", "failed", "absent"])
def test_a_budget_server_that_is_not_connected_is_missing(servers):
    init = {**K3_INIT, "mcp_servers": servers}
    if servers is None:
        init = {"type": "system", "subtype": "init"}
    assert pf.claude_budget_tool(init) == "missing"
    assert pf.claude_budget_tool(None) == "missing"


def test_a_failed_request_says_why():
    assert pf.claude_failure(_lines(K3_INIT, "API Error: 401 the relay refused the credential")) == (
        "API Error: 401 the relay refused the credential")
    error = {"type": "result", "is_error": True, "result": "Credit balance is too low"}
    assert pf.claude_failure(_lines(K3_INIT, error)) == "Credit balance is too low"
    assert pf.claude_failure("") == "no output"


def test_codex_reads_back_the_last_invocations_window_and_the_last_turns_model():
    rollout = _lines({**GPT_STARTED, "payload": {**GPT_STARTED["payload"], "model_context_window": 1}},
                     GPT_TURN, GPT_TOKENS, GPT_STARTED, GPT_TOKENS)
    assert pf.codex_window(rollout) == 258_400 and pf.codex_model(rollout) == "gpt-5.5"
    assert pf.codex_window(_lines(GPT_TURN, GPT_TOKENS)) is None and pf.codex_model(_lines(GPT_STARTED)) is None
    resumed = {**GPT_TURN, "payload": {"model": "gpt-5.6-sol"}}
    assert pf.codex_model(_lines(GPT_TURN, resumed)) == "gpt-5.6-sol"


def test_the_output_limit_is_the_setting_the_cli_runs_with():
    assert pf.output_limit_value("64000\n") == 64_000
    assert pf.output_limit_value("\n") is None and pf.output_limit_value("64k") is None
    flags = "-c model_reasoning_effort=xhigh -c model_context_window=272000 -c model_max_output_tokens=128000"
    assert pf.codex_output_limit(f"codex exec {flags} -- hi") == 128_000
    assert pf.codex_output_limit("codex exec -c model_context_window=272000 -- hi") is None
    assert pf.codex_output_limit("codex exec -c xmodel_max_output_tokens=5 -- hi") is None


def test_codex_reports_95_percent_of_its_configured_window():
    assert adapter_for(GPT).reported_window(GPT) == 258_400
    assert adapter_for(OPUS).reported_window(OPUS) == 1_000_000
    assert adapter_for(K3).reported_window(K3) == 200_000


def test_each_cli_reports_the_model_as_it_names_it():
    """Claude Code's init event names the model string it was given, with the window suffix;
    codex's turn context names the bare model id."""
    assert adapter_for(OPUS).reported_model(OPUS) == "claude-opus-4-8[1m]"
    assert adapter_for(K3).reported_model(K3) == "k3"
    assert adapter_for(GPT).reported_model(GPT) == "gpt-5.5"


def test_a_route_value_in_what_the_cli_said_is_redacted(tmp_path):
    agent = _agent(GPT, tmp_path)
    said = f"error sending request for url ({OPENAI_URL}/responses) via {HOST}:443 with {KEY}"
    shown = pf.redact(said, agent, GPT)
    assert shown == ("error sending request for url (<redacted:OPENAI_BASE_URL>/responses) via "
                     "<redacted:OPENAI_BASE_URL.host>:443 with <redacted:OPENAI_API_KEY>")
    assert pf.redact(f"dialling {HOST.upper()}", agent, GPT) == "dialling <redacted:OPENAI_BASE_URL.host>"


def test_a_route_host_that_identifies_no_machine_is_neither_redacted_nor_refused(tmp_path):
    """A relay reached as ``https://relay:8443/v1`` has a host that names no particular machine
    (``hls.substrate.secret_forms.identifies``): the base URL is redacted whole, but "relay" in
    what the CLI said stays, and a command naming ``/opt/Relay-tools`` is no route value."""
    agent = _agent(GPT, tmp_path, values={"OPENAI_BASE_URL": "https://relay:8443/v1"})
    assert pf.redact("relay said no at https://relay:8443/v1/responses", agent, GPT) == (
        "relay said no at <redacted:OPENAI_BASE_URL>/responses")
    values = pf.agent_route_values(agent, adapter_for(GPT))
    assert [name for name, _ in values] == ["OPENAI_BASE_URL", "OPENAI_API_KEY"]
    pf.check_command("cd /opt/Relay-tools && ls", values)
    with pytest.raises(pf.RouteValueInCommand):
        pf.check_command("curl https://relay:8443/v1", values)


# ---------------------------------------------------------------------------
# The judgment
# ---------------------------------------------------------------------------


def _judge(arm=K3, **kw):
    base = {"model_key": "k3", "exit_code": 0, "model_seen": "k3", "window": 200_000, "output_limit": 64_000,
            "budget_tool": "connected", "failure": ""}
    return pf.judge(arm, **{**base, **kw})


def test_the_arms_model_and_window_pass():
    r = _judge()
    assert (r.ok, r.retryable, r.reason, r.window, r.expected_window, r.output_limit) == (
        True, False, "", 200_000, 200_000, 64_000)


@pytest.mark.parametrize("kw, words, retryable", [
    ({"exit_code": 124}, "timed out", True),
    ({"exit_code": 1, "failure": "API Error: 401"}, "failed with status 1: API Error: 401", True),
    ({"window": None}, "no context window", True),
    ({"model_seen": None}, "did not say which model", True),
    ({"budget_tool": "missing"}, "budget MCP server did not connect", True),
    ({"window": 1_000_000}, "context window of 1,000,000 tokens", False),
    ({"model_seen": "claude-sonnet-4-5"}, "ran 'claude-sonnet-4-5', not the arm's 'k3'", False),
    ({"output_limit": 32_000}, "an output limit of 32,000 tokens, not the arm's 64,000", False),
    ({"output_limit": None}, "no output limit, not the arm's 64,000", False),
], ids=["timeout", "failed", "no-window", "no-model", "no-budget-tool", "other-window", "other-model",
        "other-output-limit", "no-output-limit"])
def test_what_fails_and_whether_a_retry_can_help(kw, words, retryable):
    r = _judge(**kw)
    assert not r.ok and words in r.reason and r.retryable is retryable


@pytest.mark.parametrize("arm", [K3, GPT], ids=["from-claude-code", "from-codex"])
def test_a_harness_no_adapter_runs_has_no_preflight(arm):
    stray = arm.model_copy(update={"harness": "terminus-2"})
    with pytest.raises(ArmError, match="no adapter runs harness 'terminus-2'"):
        _judge(stray)
    with pytest.raises(ArmError, match="no adapter runs harness 'terminus-2'"):
        asyncio.run(pf.preflight(None, None, stray, config_dir="/preflight/config", cwd="/preflight/work"))


def test_a_codex_turn_on_another_model_is_refused_and_its_window_is_read_at_95_percent():
    ok = pf.judge(GPT, model_key="gpt-5.5", exit_code=0, model_seen="gpt-5.5", window=258_400,
                  output_limit=128_000, budget_tool="unchecked", failure="")
    assert ok.ok
    moved = pf.judge(GPT, model_key="gpt-5.5", exit_code=0, model_seen="gpt-5.6-sol", window=258_400,
                     output_limit=128_000, budget_tool="unchecked", failure="")
    assert not moved.ok and not moved.retryable
    full = pf.judge(GPT, model_key="gpt-5.5", exit_code=0, model_seen="gpt-5.5", window=272_000,
                    output_limit=128_000, budget_tool="unchecked", failure="")
    assert not full.ok and not full.retryable and "reads back as 258,400" in full.reason


# ---------------------------------------------------------------------------
# One request through the exec server to fake CLIs
# ---------------------------------------------------------------------------

FAKE_CLAUDE = """\
import json, os, sys, time
from pathlib import Path
fake = Path(os.environ["FAKE_CLI"])
control = json.loads((fake / "control.json").read_text())
cfg = Path(os.environ["CLAUDE_CONFIG_DIR"]) / ".claude.json"
registered = json.loads(cfg.read_text()) if cfg.exists() else None
names = ("ANTHROPIC_MODEL", "CLAUDE_CONFIG_DIR", "IS_SANDBOX", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")
(fake / "record.json").write_text(json.dumps({
    "argv": sys.argv[1:], "cwd": os.getcwd(), "stdin": sys.stdin.read(),
    "env": {k: os.environ.get(k) for k in names}, "claude_json": registered}))
time.sleep(control.get("sleep", 0))
model = control.get("model") or os.environ["ANTHROPIC_MODEL"]
connected = registered is not None and "budget" in registered["mcpServers"] and not control.get("mcp_failed")
servers = [{"name": "budget", "status": "connected" if connected else "failed"}] if registered else []
print(json.dumps({"type": "system", "subtype": "init", "model": model, "mcp_servers": servers}))
if control.get("fail"):
    print("API Error: 401 the relay refused the credential")
    sys.exit(1)
if not control.get("no_result"):
    usage = {model: {"contextWindow": control["window"], "maxOutputTokens": 32000},
             "claude-haiku-4-5": {"contextWindow": 200000}}
    print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "modelUsage": usage}))
"""

FAKE_CODEX = """\
import json, os, sys
from pathlib import Path
fake = Path(os.environ["FAKE_CLI"])
control = json.loads((fake / "control.json").read_text())
home = Path(os.environ["CODEX_HOME"])
auth = json.loads((home / "auth.json").read_text())
(fake / "record.json").write_text(json.dumps({
    "argv": sys.argv[1:], "cwd": os.getcwd(), "config": (home / "config.toml").read_text(),
    "auth_names": sorted(auth),
    "auth_matches": auth.get("OPENAI_API_KEY") == os.environ.get("OPENAI_API_KEY"),
    "env": {k: os.environ.get(k) for k in ("CODEX_HOME", "OPENAI_BASE_URL")}}))
if control.get("fail"):
    print("Error: stream disconnected before completion")
    sys.exit(1)
if control.get("fail_naming_route"):
    url, key = os.environ["OPENAI_BASE_URL"], os.environ["OPENAI_API_KEY"]
    print(f"Error: error sending request for url ({url}/responses) with {key}")
    sys.exit(1)
model = sys.argv[sys.argv.index("--model") + 1]
print(json.dumps({"type": "thread.started", "thread_id": "t-1"}))
if control.get("no_rollout"):
    sys.exit(0)
day = home / "sessions" / "2026" / "10" / "02"
day.mkdir(parents=True)
lines = [{"type": "session_meta", "payload": {"id": "t-1", "cli_version": "0.146.0"}},
         {"type": "event_msg", "payload": {"type": "task_started", "model_context_window": control["window"]}},
         {"type": "turn_context", "payload": {"model": control.get("model") or model}},
         {"type": "event_msg",
          "payload": {"type": "token_count", "info": {"model_context_window": control["window"] + 1}}}]
(day / "rollout-2026-10-02T00-00-00-t-1.jsonl").write_text("".join(json.dumps(x) + "\\n" for x in lines))
"""


@pytest.fixture
def sandbox(request, short_dirs, tmp_path):
    """A running exec server whose PATH finds the fake CLIs, and the harbor environment over it.
    The server's environment, the sandbox's env file, holds Claude Code's output limit, or what
    an indirect parameter gives; harbor's requests carry the agent's own over it. Yields
    (environment, fake directory, server)."""
    env_file = getattr(request, "param", {"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000"})
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    for name, body in (("claude", FAKE_CLAUDE), ("codex", FAKE_CODEX)):
        path = fake / "bin" / name
        path.write_text(f"#!{sys.executable}\n{body}")
        path.chmod(0o755)
    home = tmp_path / "home"
    home.mkdir()
    server = _Server(short_dirs(), env={
        "PATH": f"{fake / 'bin'}:/usr/bin:/bin", "HOME": str(home), "LANG": "C.UTF-8",
        "HLS_EXEC_TOKEN": TOKEN, "FAKE_CLI": str(fake), **env_file})
    ws = tmp_path / "ws"
    ws.mkdir()
    root = tmp_path / "role"
    root.mkdir()
    env = SandboxEnvironment(socket=server.socket, token=TOKEN, paths={"/workspace": ws}, root=root,
                             start_timeout_sec=20.0)
    yield env, fake, server
    server.stop()


def _control(fake: Path, **kw) -> None:
    (fake / "control.json").write_text(json.dumps(kw))


def _record(fake: Path) -> dict:
    return json.loads((fake / "record.json").read_text())


def _preflight(env, agent, arm, tmp_path, **kw):
    async def go():
        await env.start()
        try:
            return await pf.preflight(env, agent, arm, config_dir=str(tmp_path / "pre" / "config"),
                                      cwd=str(tmp_path / "pre" / "work"), **kw)
        finally:
            await env.stop()

    return asyncio.run(go())


def _no_route_value_logged(server) -> None:
    """The exec server logs the start of every command; none holds a route value."""
    for line in server.log():
        for value in (KEY, OPENAI_URL, ANTHROPIC_URL, HOST):
            assert value not in line


def test_claude_code_preflight_runs_the_builds_cli_and_reads_its_window(sandbox, tmp_path):
    env, fake, server = sandbox
    _control(fake, window=1_000_000)
    agent = _agent(OPUS, tmp_path)
    r = _preflight(env, agent, OPUS, tmp_path)
    assert (r.ok, r.window, r.model_key, r.model_seen, r.budget_tool, r.output_limit) == (
        True, 1_000_000, "claude-opus-4-8[1m]", "claude-opus-4-8[1m]", "connected", 64_000)
    rec = _record(fake)
    assert rec["argv"] == ["--verbose", "--output-format=stream-json", "--permission-mode=bypassPermissions",
                           *agent.build_cli_flags().split(), "--print"]
    assert rec["stdin"] == pf.PROMPT and rec["cwd"] == str(tmp_path / "pre" / "work")
    assert rec["env"] == {"ANTHROPIC_MODEL": "claude-opus-4-8[1m]", "IS_SANDBOX": "1",
                          "CLAUDE_CONFIG_DIR": str(tmp_path / "pre" / "config"),
                          "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    assert rec["claude_json"] == {
        "mcpServers": {"budget": {"type": "stdio", "command": MCP_TARGET, "args": []}}}
    # Nothing lands where the build's transcript is read from.
    assert not (tmp_path / "logs").exists() or not any((tmp_path / "logs").rglob("*"))
    _no_route_value_logged(server)


@pytest.mark.parametrize("control, words, retryable", [
    ({"window": 200_000}, "context window of 200,000", False),
    ({"window": 1_000_000, "model": "claude-sonnet-4-5"}, "ran 'claude-sonnet-4-5'", False),
    ({"window": 1_000_000, "mcp_failed": True}, "budget MCP server did not connect", True),
    ({"window": 1_000_000, "fail": True}, "status 1: API Error: 401", True),
    ({"window": 1_000_000, "no_result": True}, "no context window", True),
], ids=["window", "model", "budget-tool", "failed", "no-result"])
def test_claude_code_preflight_failures(sandbox, tmp_path, control, words, retryable):
    env, fake, _ = sandbox
    _control(fake, **control)
    r = _preflight(env, _agent(OPUS, tmp_path), OPUS, tmp_path)
    assert not r.ok and words in r.reason and r.retryable is retryable


@pytest.mark.parametrize("sandbox, limit, words", [
    ({}, "32000", "an output limit of 32,000 tokens, not the arm's 64,000"),
    ({}, None, "no output limit, not the arm's 64,000"),
    ({"CLAUDE_CODE_MAX_OUTPUT_TOKENS": "32000"}, "64000", None),
], ids=["other", "unset", "request-wins"], indirect=["sandbox"])
def test_claude_codes_output_limit_is_the_one_its_process_runs_with(sandbox, tmp_path, limit, words):
    """The CLI's process takes the limit from harbor's request, which carries the agent's
    ``extra_env`` over whatever the sandbox's own environment holds."""
    env, fake, _ = sandbox
    _control(fake, window=1_000_000)
    agent = _agent(OPUS, tmp_path)
    agent._extra_env.pop("CLAUDE_CODE_MAX_OUTPUT_TOKENS")
    if limit is not None:
        agent._extra_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = limit
    r = _preflight(env, agent, OPUS, tmp_path)
    if words is None:
        assert r.ok and r.output_limit == 64_000
    else:
        assert not r.ok and not r.retryable and words in r.reason


def test_a_preflight_that_does_not_answer_in_time_is_retried(sandbox, tmp_path):
    env, fake, _ = sandbox
    _control(fake, window=1_000_000, sleep=30)
    r = _preflight(env, _agent(OPUS, tmp_path), OPUS, tmp_path, timeout_sec=1)
    assert not r.ok and r.retryable and "timed out" in r.reason


def test_an_arm_without_the_budget_tool_is_not_checked_for_it(sandbox, tmp_path):
    env, fake, _ = sandbox
    _control(fake, window=200_000)
    r = _preflight(env, _agent(K3, tmp_path, mcp=False), K3, tmp_path)
    assert r.ok and r.budget_tool == "unchecked" and _record(fake)["claude_json"] is None


def test_codex_preflight_runs_the_builds_first_call_and_reads_its_rollout(sandbox, tmp_path):
    env, fake, server = sandbox
    _control(fake, window=258_400)
    agent = _agent(GPT, tmp_path)
    r = _preflight(env, agent, GPT, tmp_path)
    assert (r.ok, r.window, r.expected_window, r.model_seen, r.budget_tool, r.output_limit) == (
        True, 258_400, 258_400, "gpt-5.5", "unchecked", 128_000)
    rec = _record(fake)
    argv = rec["argv"]
    assert argv[:argv.index("--")] == [
        "exec", "--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--ignore-rules",
        "--model", "gpt-5.5", "--json", "--enable", "unified_exec", *agent.build_cli_flags().split(),
        "-c", 'web_search="disabled"']
    assert argv[argv.index("--") + 1:] == [pf.PROMPT]
    assert rec["env"] == {"CODEX_HOME": str(tmp_path / "pre" / "config"), "OPENAI_BASE_URL": OPENAI_URL}
    assert rec["auth_names"] == ["OPENAI_API_KEY"] and rec["auth_matches"]
    # The auth file held the credential only while codex ran.
    assert not (tmp_path / "pre" / "config" / "auth.json").exists()
    # The build's writes in the build's order: the web search setting, the provider block, the MCP block.
    assert rec["config"].startswith('web_search = "disabled"\n\nmodel_provider = "custom"\n')
    config = tomllib.loads(rec["config"])
    assert config["model_providers"]["custom"]["base_url"] == OPENAI_URL
    assert config["mcp_servers"] == {"budget": {"command": MCP_TARGET, "args": []}}
    _no_route_value_logged(server)


@pytest.mark.parametrize("control, words, retryable", [
    ({"window": 272_000}, "reads back as 258,400", False),
    ({"window": 258_400, "model": "gpt-5.6-sol"}, "ran 'gpt-5.6-sol', not the arm's 'gpt-5.5'", False),
    ({"window": 258_400, "fail": True}, "status 1: Error: stream disconnected", True),
    ({"window": 258_400, "no_rollout": True}, "no context window", True),
], ids=["window", "model", "failed", "no-rollout"])
def test_codex_preflight_failures(sandbox, tmp_path, control, words, retryable):
    env, fake, _ = sandbox
    _control(fake, **control)
    r = _preflight(env, _agent(GPT, tmp_path), GPT, tmp_path)
    assert not r.ok and words in r.reason and r.retryable is retryable
    assert not (tmp_path / "pre" / "config" / "auth.json").exists()


def test_a_failure_that_names_the_route_is_reported_without_its_values(sandbox, tmp_path):
    env, fake, _ = sandbox
    _control(fake, window=258_400, fail_naming_route=True)
    r = _preflight(env, _agent(GPT, tmp_path), GPT, tmp_path)
    assert not r.ok and r.retryable and "url (<redacted:OPENAI_BASE_URL>/responses)" in r.reason
    for value in (KEY, OPENAI_URL, HOST):
        assert value not in r.model_dump_json()


class _Sent:
    """An environment that runs nothing and keeps every command it is sent, answering each with
    the next of *answers*, then with exit 0 and no output."""

    default_user = None

    def __init__(self, *answers: ExecResult) -> None:
        self.sent: list[str] = []
        self.answers = list(answers)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.sent.append(command)
        return self.answers.pop(0) if self.answers else ExecResult(stdout="", stderr=None, return_code=0)


def _sent_preflight(env, agent, arm):
    return asyncio.run(pf.preflight(env, agent, arm, config_dir="/logs/preflight/config",
                                    cwd="/logs/preflight/work"))


def _literal_provider(self):
    """A provider command with the base URL written into the command."""
    base_url = self._get_env("OPENAI_BASE_URL")
    toml = ('\nmodel_provider = "custom"\n\n[model_providers.custom]\nname = "custom"\n'
            f'base_url = "{base_url}"\nwire_api = "responses"\nrequires_openai_auth = true\n')
    return f'printf %s {shlex.quote(toml)} >> "$CODEX_HOME/config.toml"'


@pytest.mark.parametrize("case, var", [("literal-provider", "OPENAI_BASE_URL"),
                                       ("only-the-last-command", "OPENAI_BASE_URL")])
def test_a_preflight_command_holding_a_route_value_is_refused_before_any_is_sent(tmp_path, monkeypatch, case,
                                                                                 var):
    """Refused at the preflight, before the agent's clock, never inside the build after it."""
    arm, agent = GPT, _agent(GPT, tmp_path)
    if case == "literal-provider":
        monkeypatch.setattr(Codex, "_build_provider_config_command", _literal_provider)
    else:
        monkeypatch.setattr(pf, "codex_readback_command", lambda: f"ls /{HOST}")
    env = _Sent()
    r = _sent_preflight(env, agent, arm)
    assert not r.ok and not r.retryable and f"the value of {var}" in r.reason
    assert env.sent == []
    for value in (KEY, OPENAI_URL, HOST, "https://hls"):
        assert value not in r.reason.lower()


@pytest.mark.parametrize("harness", ["codex", "claude-code"])
def test_a_single_label_route_host_the_commands_fixed_text_holds_refuses_nothing(tmp_path, harness):
    """A host of one label names no machine, so it is no form of the route
    (``hls.substrate.secret_forms.identifies``): the commands' fixed text, which says ``hls``, may
    hold it, and every command is sent."""
    if harness == "codex":
        arm, agent = GPT, _agent(GPT, tmp_path, values={"OPENAI_BASE_URL": "https://hls/v1"})
    else:
        # The route as the stage gives it to Claude Code in harbor's requests.
        arm, agent = OPUS, _agent(OPUS, tmp_path)
        agent._extra_env["ANTHROPIC_BASE_URL"] = "https://HLS/anthropic"
    env = _Sent()
    r = _sent_preflight(env, agent, arm)
    assert "the value of" not in (r.reason or "")
    assert env.sent and any("hls" in command.lower() for command in env.sent)


def test_a_setup_failure_that_names_the_route_is_reported_without_its_values(tmp_path):
    said = f"bash: line 4: {OPENAI_URL}: refused for {KEY} at {HOST.upper()}"
    env = _Sent(ExecResult(stdout=said, stderr=None, return_code=1))
    r = _sent_preflight(env, _agent(GPT, tmp_path), GPT)
    assert not r.ok and r.retryable and "failed with status 1: bash: line 4: <redacted:OPENAI_BASE_URL>" in r.reason
    for value in (KEY, OPENAI_URL, HOST):
        assert value not in r.model_dump_json().lower()
    # The auth file the setup may have written is removed all the same.
    assert env.sent[-1] == pf.codex_auth_removal_command()


def test_a_codex_agent_run_without_the_arms_output_limit_is_refused(sandbox, tmp_path):
    env, fake, _ = sandbox
    _control(fake, window=258_400)
    agent = HlsCodex(logs_dir=tmp_path / "logs", model_name="gpt-5.5", mcp_servers=[budget_server()],
                     reasoning_effort="xhigh", model_context_window=272_000,
                     extra_env={"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": OPENAI_URL})
    r = _preflight(env, agent, GPT, tmp_path)
    assert not r.ok and not r.retryable and "no output limit, not the arm's 128,000" in r.reason


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_a_lost_sandbox_raises(sandbox, tmp_path, arm):
    env, fake, server = sandbox
    _control(fake, window=1_000_000)
    agent = _agent(arm, tmp_path)

    async def go():
        await env.start()
        server.stop()
        try:
            await pf.preflight(env, agent, arm, config_dir=str(tmp_path / "pre" / "config"),
                               cwd=str(tmp_path / "pre" / "work"))
        finally:
            await env.stop()

    with pytest.raises(SandboxLost):
        asyncio.run(go())


# ---------------------------------------------------------------------------
# The same commands as harbor's own build
# ---------------------------------------------------------------------------


class _Recording:
    """An environment that runs nothing and records every exec harbor makes."""

    default_user = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.calls.append((command.removeprefix("set -o pipefail; "), dict(env or {})))
        return ExecResult(stdout="", stderr=None, return_code=0)


def _harbor_run(agent) -> list[tuple[str, dict]]:
    rec = _Recording()
    asyncio.run(agent.run(pf.PROMPT, rec, AgentContext()))
    return rec.calls


def test_claude_codes_preflight_is_harbors_launch_with_the_same_flags_and_environment(tmp_path):
    agent = _agent(OPUS, tmp_path, goal=False)
    # Variables given to the agent itself go with every exec harbor makes.
    agent._extra_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = "64000"
    [launch] = [(c, e) for c, e in _harbor_run(agent) if "exec claude " in c]
    command, env = launch
    flags = agent.build_cli_flags()
    assert f"exec {pf.CLAUDE_LAUNCH} {flags} --print < " in command
    assert f"{pf.CLAUDE_LAUNCH} {flags} --print 2>&1" in pf.claude_command(agent)
    assert env == {**pf.claude_env(agent, config_dir="x"), "CLAUDE_CONFIG_DIR": "/logs/agent/sessions"}


def test_codexs_preflight_is_harbors_first_call_and_config_writes(tmp_path):
    agent = _agent(GPT, tmp_path, goal=False)
    calls = _harbor_run(agent)
    [(first, env)] = [(c, e) for c, e in calls if "codex exec " in c]
    assert first == pf.codex_command(agent) + " | tee -a /logs/agent/codex.txt"
    assert env == {**pf.codex_env(agent, config_dir="x"), "CODEX_HOME": "/tmp/codex-home"}
    setup = pf.codex_setup_command(agent, config_dir="x", cwd="y")
    [web_search] = [c for c, _ in calls if c.startswith("printf '%s\\n' 'web_search")]
    [provider] = [c for c, _ in calls if "[model_providers.custom]" in c]
    [registers] = [c for c, _ in calls if "[mcp_servers.budget]" in c]
    assert web_search in setup and provider in setup and provider == agent._build_provider_config_command()
    register = agent.mcp_config_command()
    assert register in registers and register in setup
    # The build records its config after the MCP block; the preflight writes no record.
    assert registers.endswith("/logs/agent/codex-config.toml") and "codex-config.toml" not in setup
    # Neither the build's commands nor the preflight's hold a route value.
    for command in [c for c, _ in calls] + [setup, pf.codex_command(agent)]:
        for value in (KEY, OPENAI_URL, HOST):
            assert value not in command
