"""The agent's preflight: one trivial request through the pinned CLI, in the arm's exact
configuration, before the agent's clock starts.

The stage runs it in the sandbox once the exec server answers and the task files are in. It is
the arm's model-route probe, a real request through the seal to the arm's route, and it reads
back what the CLI says it ran on (as of Claude Code 2.1.251 and codex 0.146.0):
- Claude Code: the stream's ``init`` event names the model the CLI was given and the MCP
  servers it connected; the last ``result`` event's ``modelUsage`` is keyed by that exact
  model string, ``[1m]`` included, and the entry's ``contextWindow`` is the
  window the CLI used: 1,000,000 under ``[1m]``, its 200,000 default otherwise. The CLI
  computes it with the function its auto-compaction uses.
- Codex: its session rollout under ``$CODEX_HOME/sessions``
  (``YYYY/MM/DD/rollout-...-<thread id>.jsonl``) records each turn's model
  (``turn_context.payload.model``), and each invocation's ``task_started`` event carries the
  window (``payload.model_context_window``). ``codex.txt``, the ``exec --json`` stream, carries
  neither.

The output limit has no readback: the entry's ``maxOutputTokens`` is the CLI's model-table default
(32,000 even where every request carries ``max_tokens: 64000``), and codex reports none. So
the preflight checks the setting the CLI is run with: Claude Code's ``CLAUDE_CODE_MAX_OUTPUT_TOKENS``
as the CLI's process sees it, which the CLI applies to every request it builds, and codex's
``-c model_max_output_tokens`` on its command line.

The command is built from the very agent ``build_agent`` returned, its CLI flags and its run
environment included, so the preflight cannot drift from the build. It keeps out of the build's
records: Claude Code gets its own ``CLAUDE_CONFIG_DIR`` and codex its own ``CODEX_HOME``, both
*config_dir*, and the CLI runs in *cwd*, so no session file lands where the build's transcript is
read from. The budget MCP server is registered there by the agent's own registration command.
The route's values travel only in the requests' environment, as in the build: a command that
would hold one, in any case, is refused before the first command is sent (``check_command``,
which ``HlsCodex`` applies to the build's commands too), and what the CLI printed about a failure
keeps none of them (``redact``).

Outcomes, ``PreflightResult``:
- ok: the readback names the arm's model and its window, the CLI runs with the arm's output
  limit, and Claude Code connected the budget server when the arm has it;
- not ok and not retried: the CLI ran another model, another window or another
  output limit. The arm, or its harness, is not what the record says, and a retry would see the
  same;
- not ok and retried: the request failed, timed out, or left no readback, which a dead route, a
  seal that refused it or a provider outage would also do; a budget server that did not
  connect.

This module imports no harbor: the stage imports it where harbor cannot be imported. The agent is
used through the attributes harbor's adapters have.

Codex reports the window it uses, 95 percent of the configured one (gpt-5.5 and gpt-5.6-sol
without a setting report 258,400 of the bundled table's 272,000), so
the readback is compared with what the arm's adapter says the CLI reports
(``HarnessAdapter.reported_window``, ``reported_model``). Codex reports no MCP server status in a
form checked here, so its budget server is not checked."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterator
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from hls.arms import ArmRecord, HarnessAdapter, adapter_for, route_values

#: The request the preflight makes.
PROMPT = "Reply with the single word: ready."
#: Seconds the preflight's request may take before it counts as failed.
TIMEOUT_SEC = 300
#: The exit status of a command the exec server's timeout ended.
TIMEOUT_EXIT = 124
#: How Claude Code is launched, as harbor's ``ClaudeCode._launch_command`` launches it.
CLAUDE_LAUNCH = "claude --verbose --output-format=stream-json --permission-mode=bypassPermissions"
#: How codex's first call is launched, as harbor's ``Codex.run`` launches it, without its model,
#: settings and prompt.
CODEX_LAUNCH = ("codex exec --dangerously-bypass-approvals-and-sandbox --skip-git-repo-check --ignore-rules "
                "--model {model} --json --enable unified_exec {flags}{web_search}-- {prompt}")
#: Lines of the rollout the readback keeps: the turns' contexts and the invocations' starts.
ROLLOUT_LINES = 40
#: The variable Claude Code takes its output limit from.
CLAUDE_OUTPUT_LIMIT_VAR = "CLAUDE_CODE_MAX_OUTPUT_TOKENS"
#: Characters of a failure's text a reason keeps.
REASON_CHARS = 300
#: Seconds each of the preflight's own short commands (the readbacks, the auth file's removal)
#: may take.
SHORT_TIMEOUT_SEC = 60


class PreflightResult(BaseModel):
    """What the preflight found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: True when the CLI ran the arm's model with the arm's window and output limit.
    ok: bool
    #: The window the CLI reported; None when it reported none.
    window: int | None
    #: The window the readback must show for the arm.
    expected_window: int
    #: The output limit the CLI runs with (see the module docstring); None when none is set.
    output_limit: int | None
    #: The model string the readback is keyed by: what the CLI was given.
    model_key: str
    #: The model the CLI said it ran; None when it said nothing.
    model_seen: str | None
    #: Whether the budget MCP server connected: "connected", "missing", or "unchecked" where the
    #: arm has none or the harness reports none.
    budget_tool: Literal["connected", "missing", "unchecked"]
    #: Why it is not ok; empty when it is. It holds no value of the arm's route.
    reason: str
    #: True when another attempt could pass: only for a failure the arm's record does not decide.
    retryable: bool


def json_lines(text: str) -> Iterator[dict[str, Any]]:
    """The lines of *text* that are JSON objects, in order; every other line is skipped."""
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def claude_init(stream_text: str) -> dict[str, Any] | None:
    """The first ``system``/``init`` event of a Claude Code stream."""
    return next((e for e in json_lines(stream_text)
                 if e.get("type") == "system" and e.get("subtype") == "init"), None)


def claude_window(stream_text: str, model_string: str) -> int | None:
    """The ``contextWindow`` of *model_string* in the last ``result`` event's ``modelUsage``. Other
    models' entries (a subagent's, a side call's) are ignored."""
    results = [e for e in json_lines(stream_text) if e.get("type") == "result"]
    if not results:
        return None
    usage = results[-1].get("modelUsage")
    entry = usage.get(model_string) if isinstance(usage, dict) else None
    window = entry.get("contextWindow") if isinstance(entry, dict) else None
    return window if isinstance(window, int) and not isinstance(window, bool) else None


def claude_budget_tool(init: dict[str, Any] | None) -> Literal["connected", "missing"]:
    """Whether the init event shows the budget server connected."""
    servers = init.get("mcp_servers") if init is not None else None
    connected = isinstance(servers, list) and any(
        isinstance(s, dict) and s.get("name") == "budget" and s.get("status") == "connected" for s in servers)
    return "connected" if connected else "missing"


def claude_failure(stream_text: str) -> str:
    """What a failed Claude Code request said: its last result's error, else its last line that
    is not JSON."""
    results = [e for e in json_lines(stream_text) if e.get("type") == "result"]
    if results and results[-1].get("is_error") is True and isinstance(results[-1].get("result"), str):
        return results[-1]["result"]
    plain = [line.strip() for line in stream_text.splitlines()
             if line.strip() and not line.lstrip().startswith("{")]
    return plain[-1] if plain else "no output"


def codex_window(rollout_text: str) -> int | None:
    """The ``model_context_window`` of the last ``task_started`` event of a codex rollout: the
    window of its newest invocation."""
    window = None
    for e in json_lines(rollout_text):
        payload = e.get("payload")
        if e.get("type") != "event_msg" or not isinstance(payload, dict):
            continue
        if payload.get("type") != "task_started":
            continue
        value = payload.get("model_context_window")
        if isinstance(value, int) and not isinstance(value, bool):
            window = value
    return window


def codex_model(rollout_text: str) -> str | None:
    """The model of the last ``turn_context`` of a codex rollout."""
    model = None
    for e in json_lines(rollout_text):
        payload = e.get("payload")
        if e.get("type") != "turn_context" or not isinstance(payload, dict):
            continue
        if isinstance(payload.get("model"), str):
            model = payload["model"]
    return model


def codex_output_limit(command: str) -> int | None:
    """The ``model_max_output_tokens`` a codex *command* sets; None when it sets none."""
    found = re.findall(r"(?:^|\s)-c model_max_output_tokens=(\d+)(?=\s|$)", command)
    return int(found[-1]) if found else None


def output_limit_value(text: str) -> int | None:
    """The output limit ``claude_output_limit_command`` printed; None when it is unset or not a
    number."""
    value = text.strip()
    return int(value) if value.isdigit() else None


class RouteValueInCommand(RuntimeError):
    """A command would carry a value of the arm's route. The message names the variable, never
    the value."""


def agent_route_values(agent: Any, adapter: HarnessAdapter) -> list[tuple[str, str]]:
    """The values of *adapter*'s route variables that *agent* sends, each named, longest first
    (``hls.arms.route_values``): read as harbor reads them, from the agent's ``extra_env`` and
    then the runner's environment, which the stage keeps clear of them. The one source of
    the commands' check, of the codex thread id's and of ``redact``."""
    return route_values(adapter, {var: agent._get_env(var)
                                  for var in (*adapter.credential_vars, adapter.base_url_var)})


def held_route_value(text: str, values: list[tuple[str, str]]) -> str | None:
    """The name of the first of *values* (``agent_route_values``) that *text* holds, in any case;
    None when it holds none."""
    folded = text.lower()
    return next((name for name, value in values if value.lower() in folded), None)


def check_command(command: str, values: list[tuple[str, str]]) -> None:
    """Raise RouteValueInCommand when *command* holds one of *values*, in any case. A command's
    text is logged (by harbor, by the exec server, in a failed command's error) where its
    environment is not, so the route reaches the CLI only in the environment of its requests."""
    name = held_route_value(command, values)
    if name is not None:
        raise RouteValueInCommand(f"a command would carry the value of {name}; the route reaches the CLI only in "
                                  "the environment of its requests")


def redact(text: str, agent: Any, arm: ArmRecord) -> str:
    """*text* with each value of *arm*'s route that *agent* sends (its credential, its base URL
    and the URL's host when it identifies a machine), in any case, replaced by
    ``<redacted:NAME>``: what a CLI prints about a failed request can quote the URL it dialled."""
    for name, value in agent_route_values(agent, adapter_for(arm)):
        text = re.sub(re.escape(value), f"<redacted:{name}>", text, flags=re.IGNORECASE)
    return text


def claude_setup_command(agent: Any, *, config_dir: str, cwd: str) -> str:
    """Make the preflight's directories and register the agent's MCP servers in *config_dir*,
    with the agent's own registration command."""
    command = f"mkdir -p {shlex.quote(config_dir)} {shlex.quote(cwd)}"
    register = agent._build_register_mcp_servers_command()
    return f"{command} && {register}" if register else command


def claude_command(agent: Any, *, prompt: str = PROMPT) -> str:
    """The preflight's Claude Code invocation: harbor's launch, the agent's flags, *prompt* on
    stdin."""
    flags = agent.build_cli_flags()
    return (f'export PATH="$HOME/.local/bin:$PATH"; printf %s {shlex.quote(prompt)} | '
            f"{CLAUDE_LAUNCH} {flags + ' ' if flags else ''}--print 2>&1")


def claude_env(agent: Any, *, config_dir: str) -> dict[str, str]:
    """The environment harbor runs the agent's CLI with, its config directory moved: the run
    environment, then the agent's ``extra_env``, which harbor adds to every exec."""
    return {**agent._build_run_env(), "CLAUDE_CONFIG_DIR": config_dir, **agent._extra_env}


def claude_output_limit_command() -> str:
    """Print the output limit Claude Code's process would see, empty when it is unset."""
    return f'printf \'%s\\n\' "${{{CLAUDE_OUTPUT_LIMIT_VAR}-}}"'


def _web_search_line(agent: Any) -> str:
    """The web search setting as harbor's codex adapter writes it into ``config.toml``."""
    return f"web_search = {json.dumps(agent._WEB_SEARCH_MODE)}"


def _web_search_override(agent: Any) -> str:
    """The web search setting as harbor's codex adapter passes it to every call with ``-c``."""
    return f"web_search={json.dumps(agent._WEB_SEARCH_MODE)}"


def codex_env(agent: Any, *, config_dir: str) -> dict[str, str]:
    """The environment harbor runs codex with, its home moved: ``CODEX_HOME``, the credential and
    base URL it sets itself, then the agent's ``extra_env``, which harbor adds to every exec."""
    env = {"CODEX_HOME": config_dir, "OPENAI_API_KEY": agent._get_env("OPENAI_API_KEY") or ""}
    base_url = agent._get_env("OPENAI_BASE_URL")
    if base_url:
        env["OPENAI_BASE_URL"] = base_url
    return {**env, **agent._extra_env}


def codex_setup_command(agent: Any, *, config_dir: str, cwd: str) -> str:
    """Make the preflight's directories and write codex's auth and configuration into
    *config_dir* as the build writes them into its own home: the auth file and the provider
    block expanded from the environment, the web search setting, the MCP block, and no recorded
    copy of the file. *config_dir* reaches the command as ``$CODEX_HOME``."""
    web_search = _web_search_line(agent)
    parts = [f'mkdir -p "$CODEX_HOME" {shlex.quote(cwd)}',
             'cat >"$CODEX_HOME/auth.json" <<EOF\n{\n  "OPENAI_API_KEY": "${OPENAI_API_KEY}"\n}\nEOF',
             f"printf '%s\\n' {shlex.quote(web_search)} > \"$CODEX_HOME/config.toml\" "
             f'&& grep -qxF {shlex.quote(web_search)} "$CODEX_HOME/config.toml"']
    for command in (agent._build_provider_config_command(), agent.mcp_config_command()):
        if command:
            parts.append(command)
    return "set -e\n" + "\n".join(parts)


def codex_auth_removal_command() -> str:
    """Remove the auth file the setup wrote, which holds the credential, as harbor's adapter removes
    its own when its run ends."""
    return 'rm -f "$CODEX_HOME/auth.json"'


def codex_command(agent: Any, *, prompt: str = PROMPT) -> str:
    """The preflight's codex invocation: harbor's first call with the agent's model and flags."""
    flags = agent.build_cli_flags()
    launch = CODEX_LAUNCH.format(model=agent.model_name.split("/")[-1], flags=f"{flags} " if flags else "",
                                 web_search=f"-c {shlex.quote(_web_search_override(agent))} ", prompt=shlex.quote(prompt))
    return f"if [ -s ~/.nvm/nvm.sh ]; then . ~/.nvm/nvm.sh; fi; {launch} 2>&1 </dev/null"


def codex_readback_command() -> str:
    """Print the turn contexts and invocation starts of the newest rollout under
    ``$CODEX_HOME``."""
    return ('f=$(ls -t "$CODEX_HOME"/sessions/*/*/*/rollout-*.jsonl 2>/dev/null | head -n 1); '
            f'if [ -n "$f" ]; then grep -E \'"(task_started|turn_context)"\' "$f" '
            f"| tail -n {ROLLOUT_LINES}; fi")


def _short(text: str) -> str:
    return " ".join(text.split())[:REASON_CHARS]


def judge(arm: ArmRecord, *, model_key: str, exit_code: int, model_seen: str | None, window: int | None,
          output_limit: int | None, budget_tool: Literal["connected", "missing", "unchecked"],
          failure: str) -> PreflightResult:
    """The preflight's outcome from what the CLI returned and read back. *failure* is what the CLI
    said, already redacted."""
    adapter = adapter_for(arm)
    expected = adapter.reported_window(arm)
    want_model = adapter.reported_model(arm)

    def result(ok: bool, reason: str, retryable: bool) -> PreflightResult:
        return PreflightResult(ok=ok, window=window, expected_window=expected, output_limit=output_limit,
                               model_key=model_key, model_seen=model_seen, budget_tool=budget_tool,
                               reason=reason, retryable=retryable)

    if exit_code == TIMEOUT_EXIT:
        return result(False, "the preflight request timed out", True)
    if exit_code != 0:
        return result(False, f"the preflight request failed with status {exit_code}: {_short(failure)}", True)
    if model_seen is not None and model_seen != want_model:
        return result(False, f"the CLI ran {model_seen!r}, not the arm's {want_model!r}", False)
    if window is None:
        return result(False, "the CLI reported no context window for the arm's model", True)
    if window != expected:
        return result(False, f"the CLI ran with a context window of {window:,} tokens; the arm's "
                             f"{arm.context_window:,} reads back as {expected:,}", False)
    if output_limit != arm.output_limit:
        ran = "no output limit" if output_limit is None else f"an output limit of {output_limit:,} tokens"
        return result(False, f"the CLI runs with {ran}, not the arm's {arm.output_limit:,}", False)
    if model_seen is None:
        return result(False, "the CLI did not say which model it ran", True)
    if budget_tool == "missing":
        return result(False, "the budget MCP server did not connect", True)
    return result(True, "", False)


async def preflight(env: Any, agent: Any, arm: ArmRecord, *, config_dir: str, cwd: str,
                    timeout_sec: int = TIMEOUT_SEC) -> PreflightResult:
    """Run *arm*'s CLI once through the sandbox environment *env*, configured as *agent* (what
    ``build_agent`` returned for *arm*), with *config_dir* as its configuration directory and
    *cwd* as its working directory, and judge what it read back, by its harness's steps
    (``HarnessAdapter.preflight``: ``claude_code_preflight``, ``codex_preflight``). Every command
    is checked for a route value before the first is sent; one that holds one is refused, not
    retried, before the agent's clock (the build's commands get the same check later,
    ``HlsCodex``). A lost sandbox raises."""
    return await adapter_for(arm).preflight(env, agent, arm, config_dir=config_dir, cwd=cwd, timeout_sec=timeout_sec)


def _unsent(adapter: HarnessAdapter, agent: Any, arm: ArmRecord, commands: list[str]) -> PreflightResult | None:
    """The preflight's outcome when one of *commands* holds a value of *arm*'s route
    (``check_command``): sent nowhere and not retried. None when none holds one."""
    values = agent_route_values(agent, adapter)
    try:
        for command in commands:
            check_command(command, values)
    except RouteValueInCommand as e:
        return PreflightResult(ok=False, window=None, expected_window=adapter.reported_window(arm), output_limit=None,
                               model_key=adapter.model_string(arm), model_seen=None, budget_tool="unchecked",
                               reason=f"the preflight was not sent: {e}", retryable=False)
    return None


async def claude_code_preflight(adapter: HarnessAdapter, env: Any, agent: Any, arm: ArmRecord, *, config_dir: str,
                                cwd: str, timeout_sec: int) -> PreflightResult:
    """Claude Code's preflight (``preflight``): register the agent's MCP servers in
    *config_dir*, read the output limit the CLI's process sees, then send the request and read
    its stream."""
    model_key = adapter.model_string(arm)
    commands = [claude_setup_command(agent, config_dir=config_dir, cwd=cwd), claude_output_limit_command(),
                claude_command(agent)]
    unsent = _unsent(adapter, agent, arm, commands)
    if unsent is not None:
        return unsent
    setup_command, limit_command, launch = commands
    cenv = claude_env(agent, config_dir=config_dir)
    setup = await env.exec(setup_command, env=cenv, timeout_sec=SHORT_TIMEOUT_SEC)
    if setup.return_code != 0:
        return judge(arm, model_key=model_key, exit_code=setup.return_code, model_seen=None, window=None,
                     output_limit=None, budget_tool="unchecked",
                     failure=redact(setup.stdout or "the preflight setup failed", agent, arm))
    limit = await env.exec(limit_command, env=cenv, timeout_sec=SHORT_TIMEOUT_SEC)
    output_limit = output_limit_value(limit.stdout or "") if limit.return_code == 0 else None
    done = await env.exec(launch, cwd=cwd, env=cenv, timeout_sec=timeout_sec)
    stream = done.stdout or ""
    init = claude_init(stream)
    model_seen = init.get("model") if init is not None and isinstance(init.get("model"), str) else None
    return judge(arm, model_key=model_key, exit_code=done.return_code, model_seen=model_seen,
                 window=claude_window(stream, model_key), output_limit=output_limit,
                 budget_tool=claude_budget_tool(init) if agent.mcp_servers else "unchecked",
                 failure=redact(claude_failure(stream), agent, arm))


async def codex_preflight(adapter: HarnessAdapter, env: Any, agent: Any, arm: ArmRecord, *, config_dir: str,
                          cwd: str, timeout_sec: int) -> PreflightResult:
    """Codex's preflight (``preflight``): write its auth and configuration into *config_dir*,
    send the request, remove the auth file whatever happened, then read the newest rollout."""
    model_key = adapter.model_string(arm)
    commands = [codex_setup_command(agent, config_dir=config_dir, cwd=cwd), codex_command(agent),
                codex_auth_removal_command(), codex_readback_command()]
    unsent = _unsent(adapter, agent, arm, commands)
    if unsent is not None:
        return unsent
    setup_command, command, removal, readback = commands
    xenv = codex_env(agent, config_dir=config_dir)
    try:
        setup = await env.exec(setup_command, env=xenv, timeout_sec=SHORT_TIMEOUT_SEC)
        if setup.return_code != 0:
            return judge(arm, model_key=model_key, exit_code=setup.return_code, model_seen=None, window=None,
                         output_limit=None, budget_tool="unchecked",
                         failure=redact(setup.stdout or "the preflight setup failed", agent, arm))
        done = await env.exec(command, cwd=cwd, env=xenv, timeout_sec=timeout_sec)
    finally:
        await env.exec(removal, env=xenv, timeout_sec=SHORT_TIMEOUT_SEC)
    output = (done.stdout or "").strip()
    failure = redact(output.splitlines()[-1] if output else "no output", agent, arm)
    if done.return_code != 0:
        return judge(arm, model_key=model_key, exit_code=done.return_code, model_seen=None, window=None,
                     output_limit=codex_output_limit(command), budget_tool="unchecked", failure=failure)
    rollout = await env.exec(readback, env=xenv, timeout_sec=SHORT_TIMEOUT_SEC)
    text = rollout.stdout or ""
    return judge(arm, model_key=model_key, exit_code=0, model_seen=codex_model(text), window=codex_window(text),
                 output_limit=codex_output_limit(command), budget_tool="unchecked", failure=failure)
