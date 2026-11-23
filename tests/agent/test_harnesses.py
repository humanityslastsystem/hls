"""The harbor agent an arm runs as: every setting from the arm record, re-entry on the runner's
clock for both harnesses, codex's MCP servers written as codex reads them, no route value in any
command harbor sends, and the runner variables harbor would read instead.

The adapters are harbor's own, at the checkout's harbor pin, which holds ``SessionReentry`` and
the re-entry loop. Codex's re-entry is checked through harbor's own ``run``, against
an environment that runs each command with bash on this host (the container's ``/tmp``, ``/logs``
and ``/workspace`` moved into the test's directory) and a fake ``codex`` that records how it was
called and exits as told."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import ClaudeCode, SessionReentry
from harbor.agents.installed.codex import Codex
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig

from hls.agent import harnesses
from hls.agent.harnesses import (BUDGET_SERVER, CLAUDE_CODE_DENIED_TOOLS, CODEX_CONFIG_KEYS, HARNESS_ENV,
                                 HlsCodex, budget_server, build_agent, harness_env_leaks)
from hls.agent.workspace import SKILLS
from hls.arms import ArmError, ArmRecord, CodexAdapter, Route, adapter_for
from hls.budget.mcp import MCP_TARGET
from hls.budget.service import WallClock
from hls.prompt import codex_goal_prompt, ralph_goal_prompt, reentry_prompt

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
# A codex thread id, shaped as codex 0.146.0 writes them.
THREAD = "01a0d29c-ec2c-71f2-962e-a7aec60b3ef5"
# A relay credential shaped like a thread id.
UUID_KEY = "0f1e2d3c-4b5a-4978-8796-a5b4c3d2e1f0"


@pytest.fixture(autouse=True)
def _no_runner_settings(monkeypatch):
    """The runner's environment holds none of the variables harbor would read, as the stage
    requires before a build."""
    for var in harness_env_leaks(os.environ):
        monkeypatch.delenv(var)


def _grants(arm: ArmRecord) -> dict[str, str]:
    """The stage's granted values for *arm*'s route, by variable."""
    return {var: VALUES[var] for var in adapter_for(arm).grants(arm.route_table)}


class _Clock:
    """A clock the test moves."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Events:
    def __init__(self) -> None:
        self.seen: list[tuple[str, dict]] = []

    def __call__(self, kind: str, **fields) -> None:
        self.seen.append((kind, fields))


def _build(arm: ArmRecord, tmp_path: Path, *, clock: WallClock | None = None, events: _Events | None = None,
           goal: bool = True, max_turns: int | None = None, mcp: bool = True, secrets: dict | None = None):
    return build_agent(arm, logs_dir=tmp_path / "logs",
                       clock=clock or WallClock(limit_sec=3600, hard_sec=3900), mcp=mcp, skills_dir=None,
                       goal=goal, max_turns=max_turns, emit=events or _Events(),
                       secrets=_grants(arm) if secrets is None else secrets)


# ---------------------------------------------------------------------------
# Claude Code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm, model", [(OPUS, "claude-opus-4-8[1m]"), (K3, "k3")],
                         ids=["1m", "default-window"])
def test_a_claude_code_arm_runs_with_its_model_string_effort_and_denied_tools(tmp_path, arm, model):
    agent = _build(arm, tmp_path)
    assert type(agent) is ClaudeCode
    assert agent.model_name == model == adapter_for(arm).model_string(arm)
    flags = agent.build_cli_flags()
    assert "--effort max" in flags and "--disallowedTools WebSearch,WebFetch" in flags
    assert "--max-turns" not in flags
    env = agent._build_run_env()
    assert env["ANTHROPIC_MODEL"] == model
    # The route and the output limit ride every exec request (extra_env), never the run environment.
    assert not {"ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "CLAUDE_CODE_MAX_OUTPUT_TOKENS"} & set(env)
    assert agent._ralph_kickoff == ralph_goal_prompt()


def test_the_denied_set_names_websearch():
    # WebSearch is the provider-executed one, which the sandbox's seal never sees.
    assert "WebSearch" in CLAUDE_CODE_DENIED_TOOLS


@pytest.mark.parametrize("arm", [OPUS, K3], ids=["api-key", "relay"])
def test_claude_code_gets_its_route_and_output_limit_with_every_exec_request(tmp_path, arm):
    """Claude Code gets the route's credential, its base URL, the output limit and the window it
    compacts against as harbor's ``extra_env``, which harbor sends with every exec request, as
    codex gets its route: they reach the CLI's process over the sandbox's exec channel, never
    through the container's env file."""
    agent = _build(arm, tmp_path)
    assert agent._extra_env == {**_grants(arm), "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
                                "CLAUDE_CODE_AUTO_COMPACT_WINDOW": str(arm.context_window)}


@pytest.mark.parametrize("url, host, held", [(ANTHROPIC_URL, HOST, True), ("https://kimi/v1", "kimi", False),
                                             ("https://127.0.0.1:8443/v1", "127.0.0.1", False)],
                         ids=["identifying", "one-label", "loopback"])
def test_claude_code_reads_a_route_host_as_secret_by_the_scrubbers_rule(tmp_path, url, host, held):
    """Harbor does not resume a session id holding a form of a granted value, and the forms are
    the scrubber's (``hls.substrate.secret_forms.forms_of``): the value, and the base URL's host
    only when it identifies a machine. One rule decides what is redacted and what is never
    resumed."""
    agent = _build(K3, tmp_path, secrets={**_grants(K3), "ANTHROPIC_BASE_URL": url})
    assert agent._holds_secret(f"0a{host.upper()}b") is held
    assert agent._holds_secret(f"x{KEY}") and agent._holds_secret(url)


@pytest.mark.parametrize("arm", [OPUS, K3], ids=["api-key", "relay"])
def test_a_claude_code_arm_whose_grant_the_stage_lacks_is_refused_by_name(tmp_path, arm):
    grants = _grants(arm)
    for var in grants:
        held = {k: v for k, v in grants.items() if k != var}
        with pytest.raises(ValueError, match=f"the stage holds no value for \\['{var}'\\]") as e:
            _build(arm, tmp_path, secrets=held)
        assert KEY not in str(e.value) and HOST not in str(e.value)


def test_claude_code_takes_nothing_from_the_runners_environment(tmp_path, monkeypatch):
    for var, value in (("ANTHROPIC_BASE_URL", "https://proxy.example"), ("ANTHROPIC_API_KEY", KEY),
                       ("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "8000"), ("ANTHROPIC_MODEL", "claude-sonnet-4-5")):
        monkeypatch.setenv(var, value)
    env = _build(OPUS, tmp_path)._build_run_env()
    assert env["ANTHROPIC_MODEL"] == "claude-opus-4-8[1m]"
    assert not {"ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "CLAUDE_CODE_MAX_OUTPUT_TOKENS"} & set(env)


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_reentry_reads_the_runners_clock_not_the_transcript(tmp_path, arm):
    tick = _Clock()
    clock = WallClock(limit_sec=3600, hard_sec=3900, clock=tick)
    reentry = _build(arm, tmp_path, clock=clock)._reentry
    assert isinstance(reentry, SessionReentry) and reentry.remaining_sec == clock.remaining_sec
    # A transcript the agent could write, claiming the session began long ago, moves nothing.
    projects = tmp_path / "logs" / "sessions" / "projects" / "-workspace"
    projects.mkdir(parents=True)
    started = {"timestamp": "2020-01-01T00:00:00Z", "type": "user"}
    (projects / "s.jsonl").write_text(json.dumps(started) + "\n")
    clock.start()
    tick.now += 600
    assert reentry.remaining_sec() == pytest.approx(3000)
    tick.now += 3300
    assert reentry.remaining_sec() == pytest.approx(-300)
    clock.stop_reentry()
    assert reentry.remaining_sec() == 0.0


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_a_resumed_session_gets_the_fixed_sentence(tmp_path, arm):
    reentry = _build(arm, tmp_path)._reentry
    assert reentry.prompt(1234.0) == reentry.prompt(5.0) == "Your session ended, but its goal is not met."


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_reentries_and_session_ends_are_events_and_a_failing_record_does_not_end_the_build(tmp_path, arm):
    events = _Events()
    reentry = _build(arm, tmp_path, events=events)._reentry
    reentry.on_event("reentered", {"session_id": "s1", "remaining_sec": 99.0})
    reentry.on_event("session_ended", {"reason": "budget_spent"})
    assert events.seen == [("build.agent_reentered", {"session_id": "s1", "remaining_sec": 99.0}),
                           ("build.agent_session_ended", {"reason": "budget_spent"})]

    def broken(kind, **fields):
        raise ValueError("the event log is gone")

    _build(arm, tmp_path, events=broken)._reentry.on_event("reentered", {})


@pytest.mark.parametrize("goal, max_turns", [(False, None), (True, 40)], ids=["no-goal", "turn-cap"])
def test_no_claude_code_reentry_without_a_goal_or_under_a_turn_cap(tmp_path, goal, max_turns):
    agent = _build(OPUS, tmp_path, goal=goal, max_turns=max_turns)
    assert agent._reentry is None
    assert (agent._ralph_kickoff is not None) is goal
    if max_turns is not None:
        assert f"--max-turns {max_turns}" in agent.build_cli_flags()


def test_an_arm_without_an_effort_passes_none_to_claude_code(tmp_path):
    arm = OPUS.model_copy(update={"effort": None})
    assert "--effort" not in _build(arm, tmp_path).build_cli_flags()


def test_claude_code_takes_no_flag_from_the_runners_fallbacks(tmp_path, monkeypatch):
    for var, value in (("CLAUDE_CODE_EFFORT_LEVEL", "low"), ("CLAUDE_CODE_MAX_TURNS", "7"),
                       ("MAX_THINKING_TOKENS", "5")):
        monkeypatch.setenv(var, value)
    agent = _build(OPUS.model_copy(update={"effort": None}), tmp_path)
    flags = agent.build_cli_flags()
    assert not {"--effort", "--max-turns", "--max-thinking-tokens"} & set(flags.split())
    assert "MAX_THINKING_TOKENS" not in agent._build_run_env()


# ---------------------------------------------------------------------------
# Codex
# ---------------------------------------------------------------------------


def test_a_codex_arm_runs_with_its_window_output_limit_effort_and_route(tmp_path):
    agent = _build(GPT, tmp_path)
    assert type(agent) is HlsCodex and isinstance(agent, Codex)
    assert agent.model_name == "gpt-5.5"
    flags = agent.build_cli_flags()
    for part in ("-c model_reasoning_effort=xhigh", "-c model_context_window=272000",
                 "-c model_max_output_tokens=128000"):
        assert part in flags
    # The route's credential and base URL, both from the stage's granted values.
    assert agent._extra_env == {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": OPENAI_URL}
    assert agent._goal_kickoff == codex_goal_prompt()
    assert isinstance(agent._reentry, SessionReentry)


def test_a_codex_arm_on_the_providers_own_host_gets_its_credential_alone(tmp_path):
    arm = GPT.model_copy(update={"route": "openai", "route_table": Route(credential="openai_key")})
    agent = _build(arm, tmp_path, secrets={"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": OPENAI_URL})
    assert agent._extra_env == {"OPENAI_API_KEY": KEY}


def test_every_codex_setting_the_adapter_makes_has_a_flag():
    keys = {f.kwarg for f in HlsCodex.CLI_FLAGS}
    assert set(CodexAdapter().cli_config(GPT)) == set(CODEX_CONFIG_KEYS) <= keys


def test_a_codex_setting_without_a_flag_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(CodexAdapter, "cli_config", lambda self, arm: {"model_auto_compact_token_limit": 1})
    with pytest.raises(ValueError, match="no command-line setting for"):
        _build(GPT, tmp_path)


def test_a_codex_arm_without_an_effort_is_refused(tmp_path):
    with pytest.raises(ValueError, match=r"^arm arm-b: codex always runs a reasoning effort .*must declare one"):
        _build(GPT.model_copy(update={"effort": None}), tmp_path)


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["from-claude-code", "from-codex"])
def test_an_arm_of_a_harness_no_adapter_runs_builds_no_agent(tmp_path, arm):
    with pytest.raises(ArmError, match="no adapter runs harness 'terminus-2'"):
        _build(arm.model_copy(update={"harness": "terminus-2"}), tmp_path, secrets={})


@pytest.mark.parametrize("held, missing", [({"OPENAI_BASE_URL": OPENAI_URL}, "OPENAI_API_KEY"),
                                           ({"OPENAI_API_KEY": KEY}, "OPENAI_BASE_URL")],
                         ids=["credential", "base-url"])
def test_a_codex_arm_whose_grant_the_stage_lacks_is_refused_by_name(tmp_path, held, missing):
    with pytest.raises(ValueError, match=rf"no value for \['{missing}'\]") as e:
        _build(GPT, tmp_path, secrets=held)
    for value in (KEY, OPENAI_URL, HOST):
        assert value not in str(e.value)


def test_a_turn_cap_codex_cannot_keep_is_an_event(tmp_path):
    events = _Events()
    agent = _build(GPT, tmp_path, events=events, max_turns=40)
    assert events.seen == [("build.turn_cap_unsupported", {"harness": "codex", "max_turns": 40})]
    assert "max-turns" not in agent.build_cli_flags()


def test_without_a_goal_codex_gets_no_kickoff_and_no_reentry(tmp_path):
    agent = _build(GPT, tmp_path, goal=False)
    assert agent._goal_kickoff is None and agent._reentry is None


# ---------------------------------------------------------------------------
# Codex through harbor's own run: the fake codex, the shell environment
# ---------------------------------------------------------------------------

FAKE_CODEX = """\
import json, os, signal, subprocess, sys, time
from pathlib import Path
fake = Path(os.environ["FAKE_CLI"])
control = json.loads((fake / "control.json").read_text())
calls = fake / "calls.jsonl"
n = len(calls.read_text().splitlines()) if calls.exists() else 0
args = sys.argv[1:]
home = Path(os.environ["CODEX_HOME"])
with calls.open("a") as f:
    f.write(json.dumps({"argv": args, "config": (home / "config.toml").read_text()}) + "\\n")
with (fake / "pids").open("a") as f:
    f.write(f"{os.getpid()}\\n")
thread = args[2] if args[:2] == ["exec", "resume"] else control.get("thread_id", THREAD)
if control.get("thread", True):
    print(json.dumps({"type": "thread.started", "thread_id": thread}), flush=True)
rollout = home / "sessions" / "2026" / "10" / "03" / f"rollout-2026-10-03T00-00-00-{thread}.jsonl"
if control.get("rollout", True):
    rollout.parent.mkdir(parents=True, exist_ok=True)
    with rollout.open("a") as f:
        f.write(json.dumps({"type": "session_meta", "payload": {"id": thread}}) + "\\n")


def ended(how):
    with (fake / "ends.jsonl").open("a") as f:
        f.write(json.dumps({"call": n, "how": how}) + "\\n")


# This call's turn events, each after its delay, in compact JSON as codex writes them.
for kind, delay in control.get("turns", {}).get(str(n), []):
    time.sleep(delay)
    with rollout.open("a") as f:
        f.write(json.dumps({"timestamp": "2026-10-03T00:00:00.000Z", "type": "event_msg",
                            "payload": {"type": kind, "turn_id": f"turn-{n}"}}, separators=(",", ":")) + "\\n")
if n in control.get("hang", []):
    # As codex 0.146.0 once its goal stops continuing: no turn runs and it does not exit.
    # SIGTERM ends it with status 0, as it ends codex.
    def term(signum, frame):
        ended("term")
        sys.exit(0)

    def spawn(code, output=None):
        # A process that sleeps on, its command line naming the fake so the test can end it.
        proc = subprocess.Popen([sys.executable, "-c", code, str(fake)], stdout=output, stderr=output)
        with (fake / "pids").open("a") as f:
            f.write(f"{proc.pid}\\n")
        return proc

    signal.signal(signal.SIGTERM, signal.SIG_IGN if control.get("ignore_term") else term)
    if control.get("child"):
        # As codex's native binary under its node launcher: a child holding the stream's pipe.
        spawn("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)")
    if control.get("decoy_pid"):
        # A process that resumes no thread, named as the drive's codex, apart from the stream.
        decoy = spawn("import time; time.sleep(30)", subprocess.DEVNULL)
        (home / "drive.pid").write_text(f"{decoy.pid}\\n")
        (fake / "decoy").write_text(str(decoy.pid))
    time.sleep(control.get("hang_sec", 30))
    ended("hang_timeout")
exits = control.get("exits", [])
sys.exit(exits[n] if n < len(exits) else 0)
"""

# The container paths harbor's codex commands use, at the start of a path.
_CONTAINER_PATHS = re.compile(r"(?<![\w.~/-])/(tmp|logs|workspace)(?=/|\b)")


class _Shell:
    """A harbor environment that runs each command with bash on this host, the container's
    ``/tmp``, ``/logs`` and ``/workspace`` moved under *root*, the fake codex first on PATH. It
    keeps every command it was sent in *sent*. After each command it calls *after* with the
    number of codex calls so far. A command containing *lost_on* raises, as a lost sandbox
    does, and so does each look at a rollout (``HlsCodex._look``) whose number, counted from 1,
    *failing_looks* says fails."""

    default_user = None

    def __init__(self, root: Path, fake: Path, after, lost_on: str | None, sent: list[str],
                 failing_looks=lambda n: False) -> None:
        self.root = root
        self.fake = fake
        self.after = after
        self.lost_on = lost_on
        self.sent = sent
        self.failing_looks = failing_looks
        self.looks = 0

    def move(self, text: str) -> str:
        moved = _CONTAINER_PATHS.sub(lambda m: f"{self.root}/{m.group(1)}", text)
        assert not _CONTAINER_PATHS.search(moved.replace(str(self.root), "ROOT")), text
        return moved

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.sent.append(command)
        if self.lost_on is not None and self.lost_on in command:
            raise RuntimeError("the sandbox is gone")
        if "wc -c" in command:
            self.looks += 1
            if self.failing_looks(self.looks):
                raise RuntimeError("the look failed")
        run_env = {"PATH": f"{self.fake / 'bin'}:/usr/bin:/bin", "HOME": str(self.root / "home"),
                   "FAKE_CLI": str(self.fake), **{k: self.move(v) for k, v in (env or {}).items()}}
        proc = await asyncio.create_subprocess_exec("bash", "-c", self.move(command), env=run_env,
                                                    cwd=self.root, stdout=asyncio.subprocess.PIPE,
                                                    stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        self.after(len(_calls(self.fake)))
        return ExecResult(stdout=out.decode(), stderr=None, return_code=proc.returncode)


def _calls(fake: Path) -> list[dict]:
    path = fake / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _ends(fake: Path) -> list[tuple[int, str]]:
    """How each fake codex call that hung ended: ``term`` (SIGTERM) or ``hang_timeout``."""
    path = fake / "ends.jsonl"
    return [(e["call"], e["how"]) for e in map(json.loads, path.read_text().splitlines())] if path.exists() else []


@pytest.fixture
def codex_run(tmp_path, monkeypatch):
    """Run a codex arm's agent through harbor's ``run`` against the shell environment. Yields
    ``go(agent, budget_after=None, left=-6400.0, lost_on=None, timeout=None, failing_looks=...,
    **control)``, which runs it, failing after *timeout* seconds and failing the looks
    *failing_looks* names (``_Shell``), and returns the fake's calls, moving the clock to
    *left* seconds of budget once *budget_after* codex calls were made; the clock (its ``clock``
    is the time source); the events; the environment's root; and every command harbor sent. A fake
    codex still running at the end is killed."""
    root = tmp_path / "box"
    (root / "home").mkdir(parents=True)
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    codex = fake / "bin" / "codex"
    codex.write_text(f"#!{sys.executable}\n" + FAKE_CODEX.replace("THREAD", repr(THREAD), 1))
    codex.chmod(0o755)
    tick = _Clock()
    clock = WallClock(limit_sec=3600, hard_sec=3900, clock=tick)
    events = _Events()
    sent: list[str] = []

    def go(agent, *, budget_after: int | None = None, left: float = -6400.0, lost_on: str | None = None,
           timeout: float | None = None, failing_looks=lambda n: False, **control):
        (fake / "control.json").write_text(json.dumps(control))

        def after(calls: int) -> None:
            if budget_after is not None and calls >= budget_after:
                tick.now = 1000.0 + clock.limit_sec - left

        clock.start()
        run = agent.run("the brief", _Shell(root, fake, after, lost_on, sent, failing_looks), AgentContext())
        asyncio.run(asyncio.wait_for(run, timeout))
        return _calls(fake)

    yield go, clock, events, root, sent
    for pid in (fake / "pids").read_text().split() if (fake / "pids").exists() else []:
        try:
            if str(fake).encode() in Path(f"/proc/{pid}/cmdline").read_bytes():
                os.kill(int(pid), signal.SIGKILL)
        except OSError:
            pass


def _reentries(events: _Events) -> list[tuple[str, dict]]:
    return [(kind, fields) for kind, fields in events.seen if kind.startswith("build.agent_")]


def test_a_codex_session_that_ends_early_is_resumed_until_the_budget_is_spent(tmp_path, codex_run):
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=4)
    assert len(calls) == 4
    kickoff, drive, *resumes = (c["argv"] for c in calls)
    assert kickoff[:2] == ["exec", "--dangerously-bypass-approvals-and-sandbox"]
    assert drive[:3] == ["exec", "resume", THREAD] and drive[-1] == "Continue working toward the goal."
    for argv in resumes:
        # The drive's own call, the thread named, the fixed sentence for harbor's prompt.
        assert argv[:-1] == drive[:-1] and argv[-1] == reentry_prompt()
    for argv in (kickoff, drive, *resumes):
        # The arm's model on every call: a resumed thread does not keep the one it was recorded with.
        assert argv[argv.index("--model") + 1] == "gpt-5.5"
        assert {"model_context_window=272000", "model_max_output_tokens=128000"} <= set(argv)
    seen = _reentries(events)
    assert [kind for kind, _ in seen] == ["build.agent_reentered", "build.agent_reentered",
                                          "build.agent_session_ended"]
    for _, fields in seen:
        assert (fields["session_id"], fields["exit_code"], fields["paused_sec"]) == (THREAD, 0, 0.0)
    assert [(f["reentries"], f["fast_exits"], f["remaining_sec"]) for _, f in seen] == [
        (1, 0, 3600.0), (2, 1, 3600.0), (2, 2, -6400.0)]
    assert seen[-1][1]["reason"] == "budget_spent"


@pytest.mark.parametrize("left, resumed", [(300.0, False), (301.0, True)], ids=["floor", "above-floor"])
def test_a_session_is_resumed_only_with_more_than_the_floor_left(tmp_path, codex_run, left, resumed):
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=2, left=left)
    kind, fields = _reentries(events)[0]
    assert fields["remaining_sec"] == left
    assert (kind == "build.agent_reentered") is resumed and (len(calls) > 2) is resumed


def test_a_codex_session_that_keeps_ending_at_once_stops_after_three_fast_exits(tmp_path, codex_run):
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events))
    assert len(calls) == 5
    seen = _reentries(events)
    assert [kind for kind, _ in seen] == ["build.agent_reentered"] * 3 + ["build.agent_session_ended"]
    assert seen[-1][1]["reason"] == "fast_exits" and seen[-1][1]["fast_exits"] == 3


def test_a_resume_that_is_not_fast_resets_the_count(tmp_path, codex_run, monkeypatch):
    monkeypatch.setattr(harnesses, "REENTRY_FAST_EXIT_SEC", 0.0)
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=7)
    assert len(calls) == 7
    assert {f["fast_exits"] for _, f in _reentries(events)} == {0}
    assert _reentries(events)[-1][1]["reason"] == "budget_spent"


def test_a_nonzero_exit_is_resumed_after_a_pause_and_raised_when_it_is_not(tmp_path, codex_run, monkeypatch):
    monkeypatch.setattr(harnesses, "REENTRY_ERROR_PAUSE_SEC", 0.01)
    go, clock, events, _, _ = codex_run
    with pytest.raises(NonZeroAgentExitCodeError) as e:
        go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=4, exits=[0, 0, 1, 1])
    assert e.value.return_code == 1
    seen = _reentries(events)
    assert [(kind, f["exit_code"], f["paused_sec"]) for kind, f in seen] == [
        ("build.agent_reentered", 0, 0.0), ("build.agent_reentered", 1, 0.02),
        ("build.agent_session_ended", 1, 0.0)]
    assert seen[-1][1]["reason"] == "budget_spent"


def test_a_pause_that_spends_the_budget_ends_the_session(tmp_path, codex_run, monkeypatch):
    go, clock, events, _, _ = codex_run
    real_sleep = asyncio.sleep

    async def sleep(sec, *args, **kwargs):
        # The error pause spends the budget; the idle watcher's waits pass as they would.
        if sec == harnesses.REENTRY_ERROR_PAUSE_SEC:
            clock.clock.now += 10_000
            return await real_sleep(0)
        return await real_sleep(sec, *args, **kwargs)

    monkeypatch.setattr(harnesses.asyncio, "sleep", sleep)
    with pytest.raises(NonZeroAgentExitCodeError):
        go(_build(GPT, tmp_path, clock=clock, events=events), exits=[0, 1])
    [(kind, fields)] = _reentries(events)
    assert (kind, fields["reason"], fields["paused_sec"]) == ("build.agent_session_ended", "budget_spent", 30.0)
    assert fields["remaining_sec"] == -6400.0


def test_a_thread_id_that_is_not_an_id_is_never_put_in_a_command(tmp_path, codex_run):
    go, clock, events, root, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), thread_id=f"{THREAD} $(touch pwned)")
    assert len(calls) == 2 and not (root / "pwned").exists()
    [(kind, fields)] = _reentries(events)
    assert fields["reason"] == "no_session"


@pytest.mark.parametrize("thread_id, key", [
    (KEY, KEY), ("--last", KEY), (UUID_KEY, UUID_KEY), (UUID_KEY, UUID_KEY.upper()), (THREAD.upper(), KEY),
], ids=["the-credential", "a-flag", "a-uuid-credential", "a-uuid-credential-in-another-case", "upper-case"])
def test_a_thread_id_codex_would_not_write_ends_the_session_unshown(tmp_path, codex_run, caplog, thread_id, key):
    """The kickoff's stream and the rollouts are the contestant's to write. An id that is not
    codex's lowercase UUID, or that holds a route value in any case, is the contestant's end of
    its session: graded, never a command, an event or a log line."""
    caplog.set_level(logging.DEBUG)
    go, clock, events, _, sent = codex_run
    agent = _build(GPT, tmp_path, clock=clock, events=events,
                   secrets={"OPENAI_API_KEY": key, "OPENAI_BASE_URL": OPENAI_URL})
    calls = go(agent, budget_after=4, thread_id=thread_id)
    assert len(calls) == 2
    [(kind, fields)] = _reentries(events)
    assert (kind, fields["reason"], fields["session_id"]) == ("build.agent_session_ended", "no_session", None)
    for text in [*sent, json.dumps(events.seen), *(r.getMessage() for r in caplog.records)]:
        assert key.lower() not in text.lower()


@pytest.mark.parametrize("thread_id", [THREAD, "ffffffff-0000-4aaa-8bbb-0123456789ab"])
def test_a_thread_id_shaped_as_codex_writes_it_is_resumed(tmp_path, codex_run, thread_id):
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, thread_id=thread_id)
    assert [c["argv"][2] for c in calls[1:]] == [thread_id, thread_id]
    assert [(kind, f["session_id"]) for kind, f in _reentries(events)] == [
        ("build.agent_reentered", thread_id), ("build.agent_session_ended", thread_id)]


@pytest.mark.parametrize("exits, raised", [([0, 1], NonZeroAgentExitCodeError), ([0, 0], RuntimeError)],
                         ids=["nonzero", "clean"])
def test_a_session_that_cannot_be_examined_ends_with_the_exit_it_had(tmp_path, codex_run, exits, raised):
    go, clock, events, _, _ = codex_run
    with pytest.raises(raised) as e:
        go(_build(GPT, tmp_path, clock=clock, events=events), exits=exits, lost_on="sed 's/^/thread /'")
    if raised is NonZeroAgentExitCodeError:
        assert e.value.return_code == 1
    assert not _reentries(events)


@pytest.mark.parametrize("control", [{"thread": False}, {"rollout": False}], ids=["no-thread", "no-rollout"])
def test_a_session_the_drive_cannot_name_is_not_resumed(tmp_path, codex_run, control):
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), **control)
    assert len(calls) == 2
    [(kind, fields)] = _reentries(events)
    assert kind == "build.agent_session_ended" and fields["reason"] == "no_session"
    assert fields["session_id"] is None and fields["reentries"] == 0


def test_stopped_reentry_resumes_nothing(tmp_path, codex_run):
    go, clock, events, _, _ = codex_run
    agent = _build(GPT, tmp_path, clock=clock, events=events)
    clock.stop_reentry()
    assert len(go(agent)) == 2
    [(kind, fields)] = _reentries(events)
    assert fields["reason"] == "budget_spent" and fields["remaining_sec"] == 0.0


def test_without_a_goal_codex_runs_once(tmp_path, codex_run):
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events, goal=False))
    assert len(calls) == 1 and calls[0]["argv"][-1] == "the brief" and not _reentries(events)


def test_the_preflights_launch_lines_are_the_ones_harbor_runs(tmp_path, codex_run):
    """The preflight checks the arm with harbor's own launch of each CLI, which it copies since it
    may not import harbor: Claude Code's fixed flags as harbor's ``_launch_command`` writes them,
    and codex's first call from ``codex exec`` to the end of its flags, its model, the agent's
    flags and the web search override included, as harbor's ``run`` sends it."""
    from hls.agent import preflight as pf

    assert f"exec {pf.CLAUDE_LAUNCH} --print < /in" in ClaudeCode._launch_command("--print ", "/in", reentry=0)
    go, clock, events, _, sent = codex_run
    agent = _build(GPT, tmp_path, clock=clock, events=events, goal=False)
    go(agent)
    [harbors] = [c for c in sent if "codex exec " in c and " resume " not in c]

    def flags(command: str) -> str:
        """The command from ``codex exec`` to the ``--`` that ends its flags."""
        return command[command.index("codex exec "):command.index(" -- ")]

    assert flags(pf.codex_command(agent)) == flags(harbors)


def test_a_drive_of_another_shape_is_refused(tmp_path):
    agent = _build(GPT, tmp_path)
    with pytest.raises(RuntimeError, match="changed shape"):
        asyncio.run(agent.exec_as_agent(None, 'codex exec resume "${tid:---last}" -- other'))


# ---------------------------------------------------------------------------
# A drive whose codex runs no turn and does not exit
# ---------------------------------------------------------------------------

# A goal turn after which the goal stops continuing: its start, then its end 0.1 s later.
_LAST_TURN = [["task_started", 0.0], ["task_complete", 0.1]]


@pytest.fixture
def quick_idle(monkeypatch):
    """The idle watch scaled down: a call is idle after 0.5 s with no turn, a look every 0.05 s,
    1 s between SIGTERM and SIGKILL."""
    monkeypatch.setattr(harnesses, "CODEX_IDLE_SEC", 0.5)
    monkeypatch.setattr(harnesses, "CODEX_IDLE_POLL_SEC", 0.05)
    monkeypatch.setattr(harnesses, "CODEX_STOP_GRACE_SEC", 1)


def _stops(events: _Events) -> list[dict]:
    return [fields for kind, fields in events.seen if kind == "build.agent_session_stopped"]


def _turn_ends(root: Path) -> int:
    """How many turn ends the thread's rollout holds; it outlives ``$CODEX_HOME``, being in the logs."""
    [rollout] = (root / "logs" / "agent" / "sessions").glob(f"*/*/*/rollout-*-{THREAD}.jsonl")
    return rollout.read_text().count('"type":"task_complete"')


def test_a_drive_whose_goal_stops_continuing_is_stopped_and_resumed(tmp_path, codex_run, quick_idle):
    """Codex 0.146.0 resumed on an active goal does not exit once the goal stops continuing, and
    idles for hours after marking it blocked. The drive is stopped
    once it has run no turn for CODEX_IDLE_SEC, the stop is an event, and the session is resumed
    with the fixed sentence, with no pause: the stop is not a failure."""
    go, clock, events, root, sent = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, timeout=15,
               turns={"1": _LAST_TURN}, hang=[1])
    assert len(calls) == 3 and calls[2]["argv"][-1] == reentry_prompt()
    assert _ends(tmp_path / "fake") == [(1, "term")]
    seen = _reentries(events)
    assert [kind for kind, _ in seen] == ["build.agent_session_stopped", "build.agent_reentered",
                                          "build.agent_session_ended"]
    stopped, entered, ended = (fields for _, fields in seen)
    assert (stopped["session_id"], stopped["after"], stopped["stop"], stopped["exit_code"], stopped["reentries"]) == (
        THREAD, "task_complete", "terminated", 0, 0)
    assert stopped["idle_sec"] >= 0.5
    assert (entered["session_id"], entered["exit_code"], entered["paused_sec"], entered["reentries"]) == (
        THREAD, 0, 0.0, 1)
    assert ended["reason"] == "budget_spent"
    # The looks and the stop are no CLI launch: only codex's three calls hold the exec server's marker.
    assert len([c for c in sent if CodexAdapter.cli_launch in c]) == 3


def test_a_goal_that_stops_again_at_once_after_each_resume_is_a_fast_exit(tmp_path, codex_run, quick_idle,
                                                                          monkeypatch):
    """A stopped call's time runs to the first look that saw it idle, not to the stop, so a resume
    whose goal stops again within seconds is a fast exit, and re-entry ends after
    REENTRY_MAX_FAST_EXITS of them. Counting the idle wait would make none fast (it is longer than
    the fast-exit bound here) and resume until the budget is spent."""
    monkeypatch.setattr(harnesses, "CODEX_IDLE_SEC", 2.0)
    monkeypatch.setattr(harnesses, "REENTRY_FAST_EXIT_SEC", 1.5)
    go, clock, events, _, _ = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), timeout=60,
               turns={str(n): _LAST_TURN for n in range(1, 6)}, hang=list(range(1, 6)))
    assert len(calls) == 5
    seen = _reentries(events)
    assert [kind for kind, _ in seen] == ["build.agent_session_stopped", "build.agent_reentered"] * 3 + [
        "build.agent_session_stopped", "build.agent_session_ended"]
    for stopped in _stops(events):
        assert stopped["idle_sec"] >= 2.0 and stopped["invocation_sec"] < 1.5
    assert [f["fast_exits"] for kind, f in seen if kind != "build.agent_session_stopped"] == [0, 1, 2, 3]
    assert seen[-1][1]["reason"] == "fast_exits"
    assert _ends(tmp_path / "fake") == [(n, "term") for n in range(1, 5)]


def test_codex_between_the_turns_of_a_continuing_goal_is_not_stopped(tmp_path, codex_run, quick_idle, monkeypatch):
    """Each new turn end restarts the wait, also when no look saw the turn run: three turns 0.8 s
    apart, each shorter than the 0.25 s between looks, run whole under a 1.5 s bound, and the drive
    is stopped only after the last."""
    monkeypatch.setattr(harnesses, "CODEX_IDLE_SEC", 1.5)
    monkeypatch.setattr(harnesses, "CODEX_IDLE_POLL_SEC", 0.25)
    go, clock, events, root, _ = codex_run
    turns = [["task_started", 0.0], ["task_complete", 0.0]] + [["task_started", 0.8], ["task_complete", 0.0]] * 2
    go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, timeout=20, turns={"1": turns}, hang=[1])
    [stopped] = _stops(events)
    assert stopped["invocation_sec"] >= 1.6
    assert _turn_ends(root) == 3 and _ends(tmp_path / "fake") == [(1, "term")]


def test_a_turn_end_from_before_the_call_does_not_count(tmp_path, codex_run, quick_idle):
    """When the drive starts, the rollout ends with the kickoff's turn end; a drive whose first turn
    starts later than CODEX_IDLE_SEC is not stopped for it."""
    go, clock, events, root, _ = codex_run
    go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, timeout=20,
       turns={"0": _LAST_TURN, "1": [["task_started", 1.5], ["task_complete", 0.1]]}, hang=[1])
    [stopped] = _stops(events)
    assert stopped["invocation_sec"] >= 1.5
    assert _turn_ends(root) == 2 and _ends(tmp_path / "fake") == [(1, "term")]


def test_a_codex_that_outlives_sigterm_is_killed(tmp_path, codex_run, quick_idle):
    """A codex and its child (the native binary under node's launcher, holding the stream's pipe)
    that outlive SIGTERM are both killed; the child left alive would hold the call open."""
    go, clock, events, _, _ = codex_run
    go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, timeout=20, turns={"1": _LAST_TURN},
       hang=[1], ignore_term=True, child=True)
    [stopped] = _stops(events)
    assert (stopped["stop"], stopped["exit_code"]) == ("killed", 137)
    [entered] = [f for kind, f in _reentries(events) if kind == "build.agent_reentered"]
    # Killed by the stop, not failed: resumed at once, and nothing raised.
    assert (entered["exit_code"], entered["paused_sec"]) == (137, 0.0)
    assert _ends(tmp_path / "fake") == []


def test_looks_that_fail_now_and_then_do_not_end_the_watch(tmp_path, codex_run, quick_idle, caplog):
    """A failed look tells nothing and the watch goes on: here the look before the call fails, so
    the watch's first look reads where the call began (past the kickoff's turn end, which must
    not count), and two more fail in a row later."""
    caplog.set_level(logging.WARNING)
    go, clock, events, _, _ = codex_run
    go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, timeout=20,
       failing_looks=lambda n: n in (1, 3, 4),
       turns={"0": _LAST_TURN, "1": [["task_started", 0.8], ["task_complete", 0.1]]}, hang=[1])
    [stopped] = _stops(events)
    assert stopped["invocation_sec"] >= 0.9 and _ends(tmp_path / "fake") == [(1, "term")]
    assert "could not be read before the call" in caplog.text and "a look at codex's rollout failed" in caplog.text


def test_looks_that_keep_failing_end_the_watch_and_not_the_build(tmp_path, codex_run, quick_idle, caplog):
    """Watching never ends the build: after _LOOK_FAILURES failed looks in a row the call is no
    longer watched and is waited for as it was before, here until the fake stops hanging."""
    caplog.set_level(logging.WARNING)
    go, clock, events, _, _ = codex_run
    go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, timeout=20,
       failing_looks=lambda n: n >= 2, turns={"1": _LAST_TURN}, hang=[1], hang_sec=1.5)
    assert not _stops(events) and _ends(tmp_path / "fake") == [(1, "hang_timeout")]
    assert "this call is no longer watched" in caplog.text
    assert [kind for kind, _ in _reentries(events)] == ["build.agent_reentered", "build.agent_session_ended"]


def test_a_recorded_pid_that_resumes_no_thread_is_not_signalled(tmp_path, codex_run, quick_idle, caplog):
    """The PID file is the contestant's to write: a PID whose command line resumes no thread is
    left alone, and the call is waited for as it was before."""
    caplog.set_level(logging.WARNING)
    go, clock, events, _, _ = codex_run
    go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3, timeout=20, turns={"1": _LAST_TURN},
       hang=[1], hang_sec=2.0, decoy_pid=True)
    decoy = (tmp_path / "fake" / "decoy").read_text()
    assert Path(f"/proc/{decoy}").exists()
    assert not _stops(events) and _ends(tmp_path / "fake") == [(1, "hang_timeout")]
    assert "its recorded PID is no codex resuming a thread" in caplog.text


def test_a_look_reads_codex_0_146_0s_own_lines(tmp_path):
    """The look reads the turn events codex 0.146.0 writes (lines as it wrote them,
    cut short), and no turn type a message or a tool's result holds."""
    rollout = tmp_path / "home" / "sessions" / "2026" / "10" / "14" / f"rollout-2026-10-14T09-18-39-{THREAD}.jsonl"
    rollout.parent.mkdir(parents=True)
    lines = [
        '{"timestamp":"2026-10-14T09:19:00.686Z","type":"event_msg","payload":{"type":"task_started",'
        '"turn_id":"0f18c9b3-7f4d-491c-abe1-0456d6d1a7cd","started_at":1791969540}}',
        '{"timestamp":"2026-10-14T09:19:05.826Z","type":"event_msg","payload":{"type":"task_complete",'
        '"turn_id":"0f18c9b3-7f4d-491c-abe1-0456d6d1a7cd","last_agent_message":"blocked"}}',
        # What may follow a turn's end: a message quoting a turn event, and a result holding one.
        json.dumps({"timestamp": "2026-10-14T09:19:06.000Z", "type": "event_msg",
                    "payload": {"type": "agent_message", "message": '{"type":"task_started"}'}},
                   separators=(",", ":")),
        json.dumps({"timestamp": "2026-10-14T09:19:07.000Z", "type": "event_msg",
                    "payload": {"type": "mcp_tool_call_end", "result": {"Ok": {"structured_content": {
                        "type": "task_started"}}}}}, separators=(",", ":")),
    ]
    rollout.write_text("".join(line + "\n" for line in lines))
    out = subprocess.run(["bash", "-c", f"set -o pipefail; {harnesses._look_command(THREAD)}"],
                         env={"PATH": "/usr/bin:/bin", "CODEX_HOME": str(tmp_path / "home")},
                         capture_output=True, text=True, check=True).stdout
    size, last = out.splitlines()
    assert size == f"size {rollout.stat().st_size}"
    m = harnesses._LAST_EVENT.fullmatch(last)
    assert m is not None and m[3] == "task_complete"
    assert int(m[1]) + int(m[2]) == rollout.read_text().index('"type":"event_msg","payload":{"type":"task_complete"')


# ---------------------------------------------------------------------------
# The route's values reach codex only in the environment of harbor's requests
# ---------------------------------------------------------------------------


def test_no_command_harbor_sends_holds_a_route_value_and_the_provider_block_still_names_the_url(
        tmp_path, codex_run):
    go, clock, events, _, sent = codex_run
    calls = go(_build(GPT, tmp_path, clock=clock, events=events), budget_after=3)
    assert len(calls) == 3 and sent
    for command in sent:
        for value in (KEY, OPENAI_URL, HOST):
            assert value not in command
    # The shell wrote the base URL from the request's environment.
    for call in calls:
        assert tomllib.loads(call["config"])["model_providers"]["custom"]["base_url"] == OPENAI_URL


@pytest.mark.parametrize("method", ["exec_as_agent", "exec_as_root"])
@pytest.mark.parametrize("value, var", [(KEY, "OPENAI_API_KEY"), (OPENAI_URL, "OPENAI_BASE_URL"),
                                        (f"curl https://{HOST}:443", "OPENAI_BASE_URL"),
                                        (f"curl https://{HOST.upper()}", "OPENAI_BASE_URL")],
                         ids=["credential", "base-url", "host", "host-in-another-case"])
def test_a_command_holding_a_route_value_is_refused_before_it_is_sent(tmp_path, method, value, var):
    sent: list[str] = []

    class _Env:
        default_user = None

        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            sent.append(command)
            return ExecResult(stdout="", stderr=None, return_code=0)

    agent = _build(GPT, tmp_path)
    with pytest.raises(RuntimeError, match=rf"the value of {var}") as e:
        asyncio.run(getattr(agent, method)(_Env(), f"echo {value}"))
    assert not sent
    for shown in (KEY, OPENAI_URL, HOST):
        assert shown not in str(e.value)


# ---------------------------------------------------------------------------
# The MCP servers, as each harness's own writer registers them
# ---------------------------------------------------------------------------


def _shell(command: str, env: dict[str, str]) -> None:
    subprocess.run(["bash", "-c", command], env={**os.environ, **env}, check=True)


def test_the_budget_server_is_a_bare_command():
    server = budget_server()
    assert (server.name, server.transport, server.command, server.args) == (
        BUDGET_SERVER, "stdio", MCP_TARGET, [])


def test_claude_code_registers_the_budget_server_as_a_bare_command(tmp_path):
    command = _build(OPUS, tmp_path)._build_register_mcp_servers_command()
    _shell(command, {"CLAUDE_CONFIG_DIR": str(tmp_path)})
    assert json.loads((tmp_path / ".claude.json").read_text()) == {
        "mcpServers": {"budget": {"type": "stdio", "command": MCP_TARGET, "args": []}}}


PROBE = MCPServerConfig(name="probe", transport="stdio", command="/opt/probe/server",
                        args=["--socket", "/run/hls/probe.sock", "it's \"quoted\" \\ here", "tab\there \U0001f600",
                              "del\x7f"])


@pytest.mark.parametrize("server, entry", [
    (budget_server(), {"budget": {"command": MCP_TARGET, "args": []}}),
    (PROBE, {"probe": {"command": "/opt/probe/server", "args": PROBE.args}}),
    (MCPServerConfig(name="probe.v2", transport="streamable-http", url="http://127.0.0.1:9/mcp"),
     {"probe.v2": {"url": "http://127.0.0.1:9/mcp"}}),
], ids=["bare-command", "with-args", "url"])
def test_codex_writes_a_servers_command_and_arguments_as_separate_keys(tmp_path, server, entry):
    agent = HlsCodex(logs_dir=tmp_path / "logs", model_name="gpt-5.5", mcp_servers=[server])
    (tmp_path / "config.toml").write_text('web_search = "disabled"\n')
    _shell(agent.mcp_config_command(), {"CODEX_HOME": str(tmp_path)})
    assert tomllib.loads((tmp_path / "config.toml").read_text()) == {"web_search": "disabled",
                                                                    "mcp_servers": entry}


def test_harbors_codex_setup_writes_both_servers_and_records_the_file_after_them(tmp_path, codex_run):
    go, clock, events, root, _ = codex_run
    agent = _build(GPT, tmp_path, clock=clock, events=events, goal=False)
    agent.mcp_servers.append(PROBE)
    [call] = go(agent)
    config = tomllib.loads(call["config"])
    assert config["mcp_servers"] == {
        "budget": {"command": MCP_TARGET, "args": []},
        "probe": {"command": "/opt/probe/server", "args": PROBE.args}}
    assert config["web_search"] == "disabled" and config["model_provider"] == "custom"
    assert config["model_providers"]["custom"] == {"name": "custom", "base_url": OPENAI_URL, "wire_api": "responses",
                                                "requires_openai_auth": True}
    # The recorded copy is the file codex read, MCP block included.
    assert (root / "logs" / "agent" / "codex-config.toml").read_text() == call["config"]


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_without_mcp_no_server_is_registered(tmp_path, arm):
    agent = _build(arm, tmp_path, mcp=False)
    assert agent._build_register_mcp_servers_command() is None
    if arm.harness == "codex":
        assert agent.mcp_config_command() is None


# ---------------------------------------------------------------------------
# What harbor would read from the runner
# ---------------------------------------------------------------------------


def test_the_variables_harbor_reads_from_the_runner_are_the_contracts():
    assert set(HARNESS_ENV) == {"CLAUDE_CODE_RETRY_WATCHDOG", "CLAUDE_CODE_EFFORT_LEVEL",
                                "CLAUDE_CODE_MAX_TURNS", "MAX_THINKING_TOKENS", "OPENAI_API_KEY",
                                "OPENAI_BASE_URL", "CODEX_AUTH_JSON_PATH", "CODEX_FORCE_AUTH_JSON"}


def test_every_variable_harbor_falls_back_to_is_named():
    fallbacks = {d.env_fallback for cls in (ClaudeCode, HlsCodex) for d in [*cls.CLI_FLAGS, *cls.ENV_VARS]
                 if d.env_fallback}
    assert fallbacks <= set(HARNESS_ENV)


@pytest.mark.parametrize("var", [*HARNESS_ENV, "ANTHROPIC_BASE_URL", "ANTHROPIC_MODEL",
                                 "ANTHROPIC_SOMETHING_NEW"])
def test_a_variable_harbor_would_read_is_a_leak(var):
    assert harness_env_leaks({"PATH": "/usr/bin", "HOME": "/h", var: "x"}) == [var]


def test_no_leak_in_a_clean_environment_and_leaks_are_sorted():
    assert harness_env_leaks({"PATH": "/usr/bin", "OPENAI_ORG": "x", "CLAUDE_CODE_OTHER": "y"}) == []
    leaks = harness_env_leaks({"OPENAI_BASE_URL": "u", "ANTHROPIC_API_KEY": "k", "MAX_THINKING_TOKENS": "1"})
    assert leaks == ["ANTHROPIC_API_KEY", "MAX_THINKING_TOKENS", "OPENAI_BASE_URL"]


# ---------------------------------------------------------------------------
# Claude Code through harbor's own run: the session id it reads back
# ---------------------------------------------------------------------------

FAKE_CLAUDE = """\
import json, os, sys
from pathlib import Path
fake = Path(os.environ["FAKE_CLI"])
control = json.loads((fake / "control.json").read_text())
with (fake / "calls.jsonl").open("a") as f:
    f.write(json.dumps({"argv": sys.argv[1:]}) + "\\n")
sid = control["session"]
print(json.dumps({"type": "system", "subtype": "init", "model": "m", "session_id": sid}))
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "session_id": sid}))
projects = Path(os.environ["CLAUDE_CONFIG_DIR"]) / "projects" / "-workspace"
projects.mkdir(parents=True, exist_ok=True)
(projects / (sid.replace("/", "_") + ".jsonl")).write_text("{}\\n")
"""


@pytest.fixture
def claude_run(tmp_path):
    """Run a Claude Code arm's agent through harbor's ``run`` against the shell environment, a
    fake ``claude`` naming the session *session* in its stream. Yields ``go(agent, session)``,
    which runs it with the clock spent after the first call and returns the fake's calls; the
    events; and every command harbor sent."""
    root = tmp_path / "box"
    (root / "home").mkdir(parents=True)
    fake = tmp_path / "fake"
    (fake / "bin").mkdir(parents=True)
    claude = fake / "bin" / "claude"
    claude.write_text(f"#!{sys.executable}\n" + FAKE_CLAUDE)
    claude.chmod(0o755)
    tick = _Clock()
    clock = WallClock(limit_sec=3600, hard_sec=3900, clock=tick)
    events = _Events()
    sent: list[str] = []

    def go(agent, session: str):
        (fake / "control.json").write_text(json.dumps({"session": session}))

        def after(calls: int) -> None:
            if calls >= 2:
                tick.now = 1000.0 + clock.limit_sec + 1

        clock.start()
        asyncio.run(agent.run("the brief", _Shell(root, fake, after, None, sent), AgentContext()))
        return _calls(fake)

    return go, clock, events, sent


#: A relay credential that is itself shaped as Claude Code writes a session id.
UUID_CREDENTIAL = "4c1d2e3f-5a6b-4c7d-8e9f-0a1b2c3d4e5f"


@pytest.mark.parametrize("session, value", [
    (UUID_CREDENTIAL, UUID_CREDENTIAL), ("--continue", "--continue"), (HOST.upper(), HOST.upper())],
    ids=["route-credential", "flag", "route-host"])
def test_a_claude_session_id_that_is_not_a_uuid_or_holds_a_route_value_ends_the_session(
        tmp_path, claude_run, caplog, session, value):
    """Harbor reads the id a resume takes from the stream tee, which the contestant's commands
    can write. Only Claude Code's lowercase UUID that holds no value of the agent's route, in
    any case, is resumed; anything else ends the session as ``no_session``, the contestant's
    end, and the id never reaches a command, an event or a log line."""
    go, clock, events, sent = claude_run
    values = {"ANTHROPIC_AUTH_TOKEN": UUID_CREDENTIAL, "ANTHROPIC_BASE_URL": ANTHROPIC_URL}
    agent = build_agent(K3, logs_dir=tmp_path / "logs", clock=clock, mcp=False, skills_dir=None, goal=True,
                        max_turns=None, emit=events, secrets=values)
    caplog.set_level(logging.DEBUG)
    calls = go(agent, session)
    assert len(calls) == 1
    [(kind, fields)] = _reentries(events)
    assert (kind, fields["reason"], fields["session_id"]) == ("build.agent_session_ended", "no_session", None)
    folded = value.lower()
    assert not [c for c in sent if folded in c.lower()]
    assert folded not in json.dumps(events.seen).lower()
    assert not [r for r in caplog.records if folded in r.getMessage().lower()]


@pytest.mark.parametrize("session", [UUID_KEY, "0a64000f-1111-4222-8333-444444444444"],
                         ids=["uuid", "holds-the-output-limit"])
def test_a_claude_session_id_shaped_as_claude_code_writes_it_is_resumed(tmp_path, claude_run, session):
    """Only the route's secret values are compared with the id: one that holds a plain
    setting's value, here the output limit 64000, is resumed."""
    go, clock, events, _ = claude_run
    agent = build_agent(K3, logs_dir=tmp_path / "logs", clock=clock, mcp=False, skills_dir=None, goal=True,
                        max_turns=None, emit=events, secrets=_grants(K3))
    assert agent._extra_env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] == "64000"
    calls = go(agent, session)
    assert len(calls) == 2 and calls[1]["argv"][calls[1]["argv"].index("--resume") + 1] == session


def test_harbors_exec_debug_line_carries_no_env(tmp_path, caplog):
    """harbor logs each exec at DEBUG; the line keeps the command and the user and drops the
    environment, which holds the agent's route."""

    class _Ok:
        async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
            return ExecResult(stdout="ok", stderr=None, return_code=0)

    caplog.set_level(logging.DEBUG)
    for arm in (K3, GPT):
        agent = _build(arm, tmp_path)
        asyncio.run(agent.exec_as_agent(_Ok(), "true"))
    running = [r for r in caplog.records if r.getMessage().startswith("Running command")]
    assert len(running) == 2 and all(not hasattr(r, "env") and r.user for r in running)
    assert not [r for r in caplog.records if KEY in str(vars(r))]


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_the_tasks_skills_reach_the_harness_as_their_sandbox_directory(tmp_path, arm):
    """The skills agent_build copied into the sandbox reach the adapter as their directory
    there, which harbor copies into the CLI's own skills directory; a task with none gives the
    adapter none."""
    agent = build_agent(arm, logs_dir=tmp_path / "logs", clock=WallClock(limit_sec=3600, hard_sec=3900), mcp=True,
                        skills_dir=SKILLS, goal=True, max_turns=None, emit=_Events(), secrets=_grants(arm))
    assert agent.skills_dir == SKILLS
    assert f"cp -r {SKILLS}/* " in agent._build_register_skills_command()
    none = _build(arm, tmp_path)
    assert none.skills_dir is None and none._build_register_skills_command() is None


@pytest.mark.parametrize("arm", [OPUS, GPT], ids=["claude-code", "codex"])
def test_harbors_cut_of_a_failed_commands_output_leaves_no_granted_value_cut_through(tmp_path, arm):
    """Harbor quotes a failed command's first 1,000 characters of output in its error and its log:
    the build's agent backs that cut off so no granted value's form spans it, and the scrubber,
    which replaces whole values, leaves none of it."""
    agent = _build(arm, tmp_path)
    output = "x" * 990 + KEY + " after"
    cut = agent._truncate_output(output)
    assert cut == "x" * 990 + " ... [truncated]"
    whole = agent._truncate_output("x" * 900 + KEY + "y" * 200)
    assert whole == "x" * 900 + KEY + "y" * (1000 - 900 - len(KEY)) + " ... [truncated]"
