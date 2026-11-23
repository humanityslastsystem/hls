"""The harbor agent an arm runs as, and the runner variables harbor would read in place of the arm.

``build_agent`` is the one place an ``ArmRecord`` becomes a harbor agent, and every setting comes
from the record through its harness's adapter (``hls.arms``), which builds the agent with this
module's builder for its harness (``claude_code_agent``, ``codex_agent``):
- Claude Code is harbor's ``ClaudeCode`` with the adapter's model string (the model id, with
  ``[1m]`` for a 1M window), the arm's effort, the denied web tools and no orchestrator
  environment. The route's credential, its base URL, the output limit and the window it
  compacts against go to the adapter as ``extra_env``, which harbor sends with every exec
  request, so they reach the CLI's process over the sandbox's exec channel and never through
  the container's env file, which may hold no secret. With a goal it gets the ``/goal``
  kickoff, and, without a turn cap, harbor's session re-entry. Harbor resumes only a session id shaped as Claude Code writes it, a
  lowercase UUID holding, in any case, no form of a value of the route's grants (its
  ``secret_env``) by the scrubber's rule (``secret_forms``, ``forms_of``): the value, or the
  host of one that is a URL when the host identifies a machine. The id is read from the
  stream, which the contestant's commands can write, and anything else ends the session as
  ``no_session``. Plain settings,
  such as the output limit, are not compared.
- Codex is ``HlsCodex``: harbor's ``Codex`` with the window and the output limit as ``-c``
  settings (the adapter's ``cli_config``), each MCP server's command and arguments written as
  separate keys, and, with a goal, the ``create_goal`` kickoff and re-entry of its drive, whose
  codex is ended once it has run no turn for ``CODEX_IDLE_SEC``. The
  route's credential and base URL go to the adapter as ``extra_env``, which harbor sends with
  every exec request; the shell expands them from that environment into codex's auth file and
  provider block, so no command string holds either. Harbor always
  passes codex an effort and falls back to "high", so an arm without one is refused
  (``CodexAdapter.check``) rather than run on a level its record does not show.

Each harbor setting that falls back to a runner variable when it is not given (effort, turn cap,
thinking budget) is given explicitly, as the record's value or None.

Re-entry is the same for both harnesses: a session that ends while the agent's budget lasts is
resumed with ``reentry_prompt``'s fixed sentence, against the runner's clock
(``WallClock.remaining_sec``), which nothing in the sandbox can move, and under harbor's bounds
(``REENTRY_*``: no resume with 300 s or less left, none after 3 fast exits in a row). Each resume
and each end is an event, ``build.agent_reentered`` or ``build.agent_session_ended``; a codex
session the harness ended for running no turn is ``build.agent_session_stopped`` before them.

The agent runs in the runner's process, and harbor reads some settings from the process
environment when the agent is given none: its retry watchdog, and codex's credential, base URL
and auth file. ``harness_env_leaks`` names every such variable the runner's environment holds,
the flags' fallbacks among them, so the stage can refuse to start rather than run an agent the
arm record does not describe."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import CliFlag, NonZeroAgentExitCodeError
from harbor.agents.installed.claude_code import (REENTRY_ERROR_PAUSE_SEC, REENTRY_FAST_EXIT_SEC,
                                                 REENTRY_FLOOR_SEC, REENTRY_MAX_FAST_EXITS, ClaudeCode,
                                                 SessionReentry)
from harbor.agents.installed.codex import Codex
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.paths import EnvironmentPaths

from hls.agent.preflight import agent_route_values, check_command, held_route_value
from hls.arms import ArmRecord, CodexAdapter, HarnessAdapter, adapter_for
from hls.budget.mcp import MCP_TARGET
from hls.budget.service import WallClock
from hls.prompt import codex_goal_prompt, ralph_goal_prompt, reentry_prompt
from hls.substrate.secret_forms import back_off, forms_of, registry_forms

log = logging.getLogger(__name__)

#: The budget MCP server's name; its tool shows as ``mcp__budget__get_budget``.
BUDGET_SERVER = "budget"
#: The codex configuration keys ``HlsCodex`` sets on its command line.
CODEX_CONFIG_KEYS = ("model_context_window", "model_max_output_tokens")
#: The tools Claude Code is started without (``--disallowedTools``), its own web search and fetch.
#: WebSearch runs on the provider's side, where the sandbox's seal never sees it; the seal
#: refuses WebFetch, so offering it would only advertise a route the brief says does not exist.
#: Claude Code 2.1.251 drops both from every request's tool list, in subagents too.
CLAUDE_CODE_DENIED_TOOLS = ("WebSearch", "WebFetch")
#: Variables harbor reads from the runner's environment in place of a setting it was not given.
HARNESS_ENV = ("CLAUDE_CODE_RETRY_WATCHDOG", "CLAUDE_CODE_EFFORT_LEVEL", "CLAUDE_CODE_MAX_TURNS",
               "MAX_THINKING_TOKENS", "OPENAI_API_KEY", "OPENAI_BASE_URL", "CODEX_AUTH_JSON_PATH",
               "CODEX_FORCE_AUTH_JSON")
#: The prefix of Claude Code's own variables, none of which the runner may hold.
ANTHROPIC_PREFIX = "ANTHROPIC_"
#: The recorded copy of codex's ``config.toml``, in the agent's log directory.
CODEX_CONFIG_COPY = (EnvironmentPaths.agent_dir / "codex-config.toml").as_posix()
#: How harbor's codex drive call names its session; only that call contains it.
CODEX_DRIVE = 'codex exec resume "${tid:---last}" '
#: Seconds a drive's codex may run no turn before ``HlsCodex`` ends it. Codex 0.146.0's ``exec``
#: returns only when the turn it started completes, but resuming a thread whose goal is active
#: starts the goal's own turn first, which takes in exec's prompt, so exec's turn never runs. While
#: the goal continues, each turn starts within milliseconds of the last one's end; once it stops
#: (blocked, complete, or a limit reached) no turn starts and exec waits for good. A gap between
#: turns of a continuing goal is at most about 13 s, across a restart of codex.
CODEX_IDLE_SEC = 60.0
#: Seconds between two looks at the drive's rollout for a turn running.
CODEX_IDLE_POLL_SEC = 10.0
#: Seconds the stop gives codex to exit on SIGTERM (it exits 0 within 3 s) before SIGKILL.
CODEX_STOP_GRACE_SEC = 10
#: Where the drive's command records its codex's PID (the node launcher, which passes signals on).
CODEX_DRIVE_PID = '"$CODEX_HOME/drive.pid"'

#: Called with an event's kind and its fields, as ``EventLog.emit`` is.
Emit = Callable[..., None]

# Codex's adapter: the route variables it is granted.
_CODEX = CodexAdapter()
# A TOML key that needs no quotes.
_BARE_KEY = re.compile(r"[A-Za-z0-9_-]+")
# A codex thread id, a lowercase UUID as codex writes it. The stream and the rollouts it is read
# from are the contestant's to write, so nothing else is put in a command, an event or a log line.
_THREAD_ID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
# The longest stream line read back for the thread id.
_THREAD_LINE_CHARS = 2000
# What follows the prompt in harbor's drive command: its output, teed to the stream.
_DRIVE_TAIL = "2>&1 </dev/null | tee -a "
# The bytes at the end of a rollout a look searches for its last turn event. Codex writes nothing
# after a turn's end until the next turn starts, so the end is in them unless its own line is
# longer; a turn that has written more than this since it started reads as running.
_ROLLOUT_TAIL_BYTES = 1 << 20
# A turn event as codex 0.146.0 writes it into a rollout, an event message's envelope and its
# payload's type in compact JSON, as an extended regular expression. A string the model or a
# command wrote holds its quotes escaped, so it never matches.
_TURN_EVENT = r'"type":"event_msg","payload":\{"type":"(task_started|task_complete|turn_aborted)"'
# A look's report of the last turn event in the tail it searched.
_LAST_EVENT = re.compile(r"last (\d+) (\d+):" + _TURN_EVENT)
# Seconds a look may run in the sandbox, and a stop beyond its grace.
_LOOK_TIMEOUT_SEC = 60
# Looks that may fail in a row before a call is no longer watched.
_LOOK_FAILURES = 3


def _toml_string(value: str) -> str:
    """*value* as a TOML basic string: quotes, backslashes and control characters escaped."""
    out = []
    for c in value:
        if c in '"\\':
            out.append("\\" + c)
        elif ord(c) < 0x20 or ord(c) == 0x7F:
            out.append(f"\\u{ord(c):04x}")
        else:
            out.append(c)
    return '"' + "".join(out) + '"'


def _stop_reason(thread_id: str | None, remaining: float, fast_exits: int) -> str | None:
    """Why a codex session that ended with ``remaining`` seconds of budget left, after
    ``fast_exits`` fast re-entries in a row, is not resumed, or None when it is. The names and
    the order are harbor's for Claude Code."""
    if remaining <= REENTRY_FLOOR_SEC:
        return "budget_spent"
    if fast_exits >= REENTRY_MAX_FAST_EXITS:
        return "fast_exits"
    if thread_id is None:
        return "no_session"
    return None


def _look_command(thread_id: str) -> str:
    """A shell command that prints the size of *thread_id*'s rollout as ``size <bytes>`` and the
    last turn event in its last ``_ROLLOUT_TAIL_BYTES`` as ``last <start> <offset>:<event>``, the
    offset counted from byte *start*; nothing when the thread has no rollout. *thread_id* is
    codex's lowercase UUID."""
    rollout = f'"$CODEX_HOME"/sessions/*/*/*/rollout-*-{thread_id}.jsonl'
    return (f"f=$(ls -1 {rollout} 2>/dev/null | head -n 1); "
            '[ -f "$f" ] || exit 0; n=$(wc -c < "$f") || exit 0; size=$((n)); '
            f"from=$(( size > {_ROLLOUT_TAIL_BYTES} ? size - {_ROLLOUT_TAIL_BYTES} : 0 )); "
            'echo "size $size"; '
            'tail -c +"$(( from + 1 ))" -- "$f" | head -c "$(( size - from ))" | '
            f"grep -a -b -o -E '{_TURN_EVENT}' | tail -n 1 | sed \"s/^/last $from /\"; true")


def _stop_command(grace_sec: int) -> str:
    """A shell command that ends the drive's codex by the PID its command recorded
    (``CODEX_DRIVE_PID``), when that PID is still a process resuming a thread: SIGTERM, then,
    after *grace_sec*, SIGKILL to its children (the native binary, which would otherwise hold the
    stream's pipe open) and to it. Prints ``stop terminated``, ``stop killed`` or
    ``stop not_found``."""
    return (f"pid=$(cat {CODEX_DRIVE_PID} 2>/dev/null); "
            'case "$pid" in ""|*[!0-9]*) echo "stop not_found"; exit 0;; esac; '
            'case "$(tr "\\000" " " < "/proc/$pid/cmdline" 2>/dev/null)" in *" exec resume "*) ;; '
            '*) echo "stop not_found"; exit 0;; esac; '
            'kill -TERM "$pid" 2>/dev/null; i=0; '
            f'while kill -0 "$pid" 2>/dev/null && [ "$i" -lt {grace_sec} ]; do sleep 1; i=$((i + 1)); done; '
            'kill -0 "$pid" 2>/dev/null || { echo "stop terminated"; exit 0; }; '
            'for c in $(cat /proc/"$pid"/task/*/children 2>/dev/null); do kill -KILL "$c" 2>/dev/null; done; '
            'kill -KILL "$pid" 2>/dev/null; echo "stop killed"; true')


#: How a stop can end (``_stop_command``).
_STOP_ENDS = ("terminated", "killed", "not_found")


@dataclass(frozen=True)
class _Look:
    """One look at a codex thread's rollout (``HlsCodex._look``)."""

    #: The rollout's size in bytes; None when the thread has no rollout.
    size: int | None
    #: The last turn event in the rollout's last ``_ROLLOUT_TAIL_BYTES``, as its type and its byte
    #: offset in the rollout; None for none there.
    last: tuple[str, int] | None


@dataclass(frozen=True)
class _Idle:
    """A drive's codex seen running no turn for ``CODEX_IDLE_SEC`` (``HlsCodex._idle``)."""

    #: The turn event it was last seen after: ``task_complete`` or ``turn_aborted``.
    after: str
    #: ``time.monotonic()`` at the first look that saw it so.
    since: float
    #: Seconds from that look to the last one.
    idle_sec: float


@dataclass(frozen=True)
class _CallEnd:
    """How one codex call of the drive ended (``HlsCodex._call``)."""

    #: The call's result; None when it exited nonzero.
    result: Any
    #: Its nonzero exit, None for a clean one or one ``HlsCodex`` stopped.
    failure: NonZeroAgentExitCodeError | None
    #: Its exit status.
    exit_code: int
    #: Seconds it ran: to its exit, or, when it was stopped, to the first look that saw it idle.
    ran_sec: float
    #: Why it was stopped; None when it exited by itself.
    idle: _Idle | None
    #: How the stop ended, ``terminated`` or ``killed``; None when it was not stopped.
    stop: str | None


class HlsCodex(Codex):
    """Harbor's codex adapter with four changes:
    - the arm's context window and output limit as ``-c`` settings on every codex call;
    - each MCP server's command and arguments as separate keys of ``config.toml`` (harbor joins
      them into one ``command`` string, which codex starts as a single program name),
      and the recorded copy of the file taken after them (harbor's is taken before);
    - every command that would hold a route value (the credential, the base URL or its host, in
      any case) refused before it is sent: the exec server and harbor log the commands, and a failed one's
      error quotes it. Harbor writes the auth file and the provider block from the
      request's environment, so none of its own commands holds one;
    - with *reentry*, the drive's session resumed after it ends, and ended
      once its codex has run no turn for ``CODEX_IDLE_SEC``.

    Harbor's codex ``run`` makes two codex calls: the first creates the goal, the second, the
    drive, resumes that thread and runs until the goal is met or blocked. ``run`` deletes
    ``$CODEX_HOME`` once the drive returns, so the resume has to happen inside the drive's exec:
    ``exec_as_agent`` recognizes the drive by ``CODEX_DRIVE`` and re-enters there, with the
    drive's own command (``--model`` included), the thread id named and the fixed
    sentence for harbor's prompt. The thread id is read from files the contestant's commands can
    write, so only codex's lowercase UUID holding no route value is resumed; anything else ends
    the session as ``no_session``. Like Claude Code's, every exit is re-entered, clean or not; a
    nonzero exit waits ``REENTRY_ERROR_PAUSE_SEC`` (doubled per fast exit before it) first, and
    one that is not re-entered raises its ``NonZeroAgentExitCodeError``. A failure of the first
    call is harbor's: it ends the run before the drive. The events carry ``session_id`` (the
    thread id), ``exit_code``, ``invocation_sec``, ``paused_sec``, ``remaining_sec``,
    ``reentries`` and ``fast_exits``, and an end its ``reason``.

    A drive call whose thread's goal is active when it starts never exits by itself: the goal's
    own turn takes in its prompt, so the turn ``codex exec`` waits for never runs, and once the
    goal stops continuing (the model marks it blocked or complete, or a turn error or a limit
    stops it) codex waits with no turn running until the harness's cut. So while each drive call
    runs, ``_idle`` looks at its rollout every ``CODEX_IDLE_POLL_SEC``, and once the last turn
    event is the same end of a turn for ``CODEX_IDLE_SEC`` the call is stopped (``_stop_command``)
    and recorded as ``build.agent_session_stopped`` (``session_id``, ``after``, ``idle_sec``,
    ``stop``, ``exit_code``, ``invocation_sec``, ``reentries``). The stopped call is then re-entered
    or ended as any exit is, never paused or raised as a failure, and its ``invocation_sec``
    runs to the first look that saw it idle, so a goal blocked again within seconds of a resume is
    a fast exit. A resumed thread whose goal is no longer active runs the resume's prompt as its
    own turn and exits when it completes. Watching never ends the build: a look that fails is
    logged, and after ``_LOOK_FAILURES`` in a row, or a stop that fails or finds no codex, the
    call is waited for as it was before."""

    CLI_FLAGS = [*Codex.CLI_FLAGS, *(CliFlag(key, cli="-c", type="int", format=f"-c {key}={{value}}")
                                     for key in CODEX_CONFIG_KEYS)]

    def __init__(self, logs_dir: Path, *args: Any, reentry: SessionReentry | None = None, **kwargs: Any) -> None:
        #: How the drive's session is resumed after it ends: the budget left, the message, the
        #: events. None leaves a session that ends ended.
        self._reentry = reentry
        super().__init__(logs_dir, *args, **kwargs)

    def mcp_config_command(self) -> str | None:
        """A shell command that appends each MCP server to ``$CODEX_HOME/config.toml``, its
        command and its arguments as separate keys; None without servers."""
        if not self.mcp_servers:
            return None
        lines: list[str] = [""]
        for server in self.mcp_servers:
            key = server.name if _BARE_KEY.fullmatch(server.name) else _toml_string(server.name)
            lines.append(f"[mcp_servers.{key}]")
            if server.transport == "stdio":
                lines.append(f"command = {_toml_string(server.command or '')}")
                lines.append(f"args = [{', '.join(_toml_string(a) for a in server.args)}]")
            else:
                lines.append(f"url = {_toml_string(server.url or '')}")
            lines.append("")
        block = "\n".join(lines)
        return f'printf %s {shlex.quote(block)} >> "$CODEX_HOME/config.toml"'

    def _build_register_mcp_servers_command(self) -> str | None:
        """Harbor's hook: the MCP block, then the recorded copy of the whole file."""
        block = self.mcp_config_command()
        if block is None:
            return None
        agent_dir = shlex.quote(EnvironmentPaths.agent_dir.as_posix())
        return f'{block} && mkdir -p {agent_dir} && cp "$CODEX_HOME/config.toml" {shlex.quote(CODEX_CONFIG_COPY)}'

    async def _exec(self, environment: Any, command: str, user: str | int | None = None,
                    env: dict[str, str] | None = None, cwd: str | None = None,
                    timeout_sec: int | None = None) -> Any:
        """Harbor's exec, which every command of the agent's passes through, refusing a command
        that holds a route value (``check_command``, as the preflight's), naming the variable and
        never the value."""
        check_command(command, agent_route_values(self, _CODEX))
        return await super()._exec(environment, command, user=user, env=env, cwd=cwd, timeout_sec=timeout_sec)

    async def exec_as_agent(self, environment: Any, command: str, env: dict[str, str] | None = None,
                            cwd: str | None = None, timeout_sec: int | None = None) -> Any:
        """Harbor's exec, except that the drive call is re-entered while ``_reentry`` allows."""
        if self._reentry is None or CODEX_DRIVE not in command:
            return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)
        return await self._drive(environment, command, env or {}, self._reentry)

    async def _drive(self, environment: Any, command: str, env: dict[str, str], reentry: SessionReentry) -> Any:
        """Run harbor's drive *command*, then resume its thread after each exit until
        ``_stop_reason`` gives a reason, each call stopped once its codex runs no turn
        (``_call``). Returns the last call's result. A harness cut cancels this coroutine, so it
        cannot start another call, and each call is awaited to its end, so one codex runs at a
        time. Every call records its codex's PID (``CODEX_DRIVE_PID``) for the stop."""
        harbor_prompt = shlex.quote(self._RESUME_PROMPT)
        tail = f"{harbor_prompt} {_DRIVE_TAIL}"
        if command.count(CODEX_DRIVE) != 1 or command.count(harbor_prompt) != 1 or command.count(tail) != 1:
            raise RuntimeError("harbor's codex drive call has changed shape; HlsCodex cannot resume it")
        command = command.replace(CODEX_DRIVE, f"(echo $BASHPID > {CODEX_DRIVE_PID}; exec {CODEX_DRIVE}").replace(
            tail, f"{harbor_prompt}) {_DRIVE_TAIL}")
        current = command
        reentries = 0
        fast_exits = 0
        try:
            thread_id = await self._resumable_thread(environment, env)
        except Exception:
            self.logger.warning("the drive's thread could not be found; its codex is not watched", exc_info=True)
            thread_id = None
        while True:
            ended = await self._call(environment, current, env, thread_id)
            failure, ran_sec = ended.failure, ended.ran_sec
            if ended.idle is not None:
                stopped = {
                    "session_id": thread_id, "after": ended.idle.after, "idle_sec": round(ended.idle.idle_sec, 1),
                    "stop": ended.stop, "exit_code": ended.exit_code, "invocation_sec": round(ran_sec, 1),
                    "reentries": reentries}
                self.logger.info(f"codex ran no turn for {CODEX_IDLE_SEC:.0f} s; its session was stopped: {stopped}")
                reentry.on_event("session_stopped", stopped)
            if reentries > 0:
                fast_exits = fast_exits + 1 if ran_sec < REENTRY_FAST_EXIT_SEC else 0
            try:
                thread_id = await self._resumable_thread(environment, env)
                remaining = reentry.remaining_sec()
                reason = _stop_reason(thread_id, remaining, fast_exits)
                paused = 0.0
                if reason is None and failure is not None:
                    paused = REENTRY_ERROR_PAUSE_SEC * 2**fast_exits
                    await asyncio.sleep(paused)
                    remaining = reentry.remaining_sec()
                    reason = _stop_reason(thread_id, remaining, fast_exits)
            except Exception:
                # After a nonzero exit, a failure to look at it (the sandbox may be gone) must not
                # replace the exit the build collects after.
                if failure is None:
                    raise
                self.logger.warning("codex exited nonzero and its session could not be examined; not re-entered",
                                    exc_info=True)
                raise failure
            fields: dict[str, Any] = {
                "session_id": thread_id, "exit_code": ended.exit_code,
                "invocation_sec": round(ran_sec, 1), "paused_sec": paused, "remaining_sec": round(remaining, 1),
                "reentries": reentries, "fast_exits": fast_exits}
            if reason is not None:
                self.logger.info(f"codex session ended, not re-entered ({reason}): {fields}")
                reentry.on_event("session_ended", {"reason": reason, **fields})
                if failure is not None:
                    raise failure
                return ended.result
            reentries += 1
            fields["reentries"] = reentries
            self.logger.info(f"re-entering the codex session: {fields}")
            reentry.on_event("reentered", fields)
            current = command.replace(CODEX_DRIVE, f"codex exec resume {thread_id} ").replace(
                harbor_prompt, shlex.quote(reentry.prompt(remaining)))

    async def _call(self, environment: Any, command: str, env: dict[str, str], thread_id: str | None) -> _CallEnd:
        """Run one codex call of the drive, *command*, to its end, stopping it once its codex has
        run no turn for ``CODEX_IDLE_SEC`` (``_idle``, ``_stop``). Only turn events past the
        rollout's size before the call count; when that size could not be read, past the size the
        watch's first look reads. Without *thread_id* the call is not watched; once the watch gives
        up (``_LOOK_FAILURES`` failed looks in a row), or the stop fails or finds no codex to stop,
        it is waited for as it was before."""
        base = None
        if thread_id is not None:
            try:
                base = (await self._look(environment, env, thread_id)).size
            except Exception:
                self.logger.warning("codex's rollout could not be read before the call", exc_info=True)
        started = time.monotonic()
        call = asyncio.ensure_future(super().exec_as_agent(environment, command, env=env))
        watch = None if thread_id is None else asyncio.ensure_future(self._idle(environment, env, thread_id, base))
        idle: _Idle | None = None
        stop: str | None = None
        try:
            if watch is not None:
                await asyncio.wait((call, watch), return_when=asyncio.FIRST_COMPLETED)
                if watch.done() and watch.exception() is not None:
                    self.logger.warning("codex's rollout could not be read; this call is no longer watched",
                                        exc_info=watch.exception())
                elif not call.done():
                    try:
                        stop = await self._stop(environment, env)
                    except Exception:
                        self.logger.warning("codex ran no turn but could not be stopped; waiting for it to exit",
                                            exc_info=True)
                    else:
                        if stop == "not_found":
                            self.logger.warning("codex ran no turn but its recorded PID is no codex resuming a "
                                                "thread; waiting for it to exit")
                            stop = None
                        else:
                            idle = watch.result()
            failure: NonZeroAgentExitCodeError | None = None
            try:
                result = await call
                exit_code = result.return_code
            except NonZeroAgentExitCodeError as exc:
                # A call the harness stopped ended by the stop, not by a failure of its own.
                result, exit_code = None, exc.return_code
                failure = exc if idle is None else None
            ended_at = time.monotonic() if idle is None else idle.since
            return _CallEnd(result=result, failure=failure, exit_code=exit_code, ran_sec=ended_at - started,
                            idle=idle, stop=stop)
        finally:
            for task in (call, watch):
                if task is not None and not task.done():
                    task.cancel()

    async def _idle(self, environment: Any, env: dict[str, str], thread_id: str, base: int | None) -> _Idle:
        """Return once the drive's codex has run no turn for ``CODEX_IDLE_SEC``: at every look
        over that time, one each ``CODEX_IDLE_POLL_SEC``, the last turn event of *thread_id*'s
        rollout is the same end of a turn, past byte *base* (None: past the size the first look
        reads). A look that finds no turn event past *base* sees codex starting, and one whose last
        is ``task_started`` sees a turn running. A failed look tells nothing, and a turn that ran
        meanwhile shows at the next as a new turn event; the ``_LOOK_FAILURES``-th failure in a
        row is raised."""
        mark: tuple[int, float] | None = None
        failures = 0
        while True:
            await asyncio.sleep(CODEX_IDLE_POLL_SEC)
            try:
                look = await self._look(environment, env, thread_id)
            except Exception:
                failures += 1
                if failures >= _LOOK_FAILURES:
                    raise
                self.logger.warning("a look at codex's rollout failed", exc_info=True)
                continue
            failures = 0
            now = time.monotonic()
            if base is None:
                base = look.size
                continue
            last = look.last
            if last is None or last[0] == "task_started" or last[1] < base:
                mark = None
            elif mark is None or mark[0] != last[1]:
                mark = (last[1], now)
            elif now - mark[1] >= CODEX_IDLE_SEC:
                return _Idle(after=last[0], since=mark[1], idle_sec=now - mark[1])

    async def _look(self, environment: Any, env: dict[str, str], thread_id: str) -> _Look:
        """*thread_id*'s rollout's size and its last turn event (``_look_command``). The rollout
        is the contestant's to write, so a line the look does not read as its own is ignored."""
        found = await super().exec_as_agent(environment, _look_command(thread_id), env=env,
                                            timeout_sec=_LOOK_TIMEOUT_SEC)
        size: int | None = None
        last: tuple[str, int] | None = None
        for line in (found.stdout or "").splitlines():
            tag, _, rest = line.partition(" ")
            if tag == "size" and rest.isdigit():
                size = int(rest)
            elif (m := _LAST_EVENT.fullmatch(line)) is not None:
                last = (m[3], int(m[1]) + int(m[2]))
        return _Look(size=size, last=last)

    async def _stop(self, environment: Any, env: dict[str, str]) -> str:
        """End the drive's codex (``_stop_command``) and say how: ``terminated``, ``killed``, or
        ``not_found`` when its recorded PID is no codex resuming a thread, as when it exited by
        itself meanwhile."""
        done = await super().exec_as_agent(environment, _stop_command(CODEX_STOP_GRACE_SEC), env=env,
                                           timeout_sec=CODEX_STOP_GRACE_SEC + _LOOK_TIMEOUT_SEC)
        for line in (done.stdout or "").splitlines():
            tag, _, how = line.partition(" ")
            if tag == "stop" and how in _STOP_ENDS:
                return how
        return "not_found"

    async def _resumable_thread(self, environment: Any, env: dict[str, str]) -> str | None:
        """The thread the drive resumes, the first ``thread.started`` of the kickoff's stream, as
        harbor's drive finds it; None when the stream names none, names one that is not codex's
        lowercase UUID or that holds a route value in any case, or its rollout is not under
        ``$CODEX_HOME/sessions``, where a resume loads it from. None ends the session as
        ``no_session``, the contestant's end, since the contestant's commands can write both."""
        out = shlex.quote((EnvironmentPaths.agent_dir / self._OUTPUT_FILENAME).as_posix())
        found = await super().exec_as_agent(environment, (
            f"grep -m1 'thread\\.started' {out} 2>/dev/null | cut -c1-{_THREAD_LINE_CHARS} | sed 's/^/thread /'; "
            "ls -1 \"$CODEX_HOME\"/sessions/*/*/*/ 2>/dev/null | grep '^rollout-' | sed 's/^/rollout /'; true"),
            env=env)
        thread_id = None
        rollouts = []
        for line in (found.stdout or "").splitlines():
            tag, _, rest = line.partition(" ")
            if tag == "rollout":
                rollouts.append(rest.strip())
            elif tag == "thread" and thread_id is None:
                try:
                    event = json.loads(rest)
                except ValueError:
                    continue
                value = event.get("thread_id") if isinstance(event, dict) else None
                if (isinstance(value, str) and _THREAD_ID.fullmatch(value)
                        and held_route_value(value, agent_route_values(self, _CODEX)) is None):
                    thread_id = value
        if thread_id is None or not any(name.endswith(f"-{thread_id}.jsonl") for name in rollouts):
            return None
        return thread_id


def budget_server() -> MCPServerConfig:
    """The budget MCP server: the executable the sandbox binds at ``MCP_TARGET``, no arguments."""
    return MCPServerConfig(name=BUDGET_SERVER, transport="stdio", command=MCP_TARGET, args=[])


def harness_env_leaks(environ: Mapping[str, str]) -> list[str]:
    """The variables of *environ* harbor would read in place of an arm's settings, sorted."""
    return sorted(k for k in environ if k in HARNESS_ENV or k.startswith(ANTHROPIC_PREFIX))


def _emitter(emit: Emit) -> Callable[[str, dict[str, Any]], None]:
    """The re-entry callback, as events. It runs inside the session loop, where an exception
    would end the build, so a failure to record is logged instead."""
    def on_event(kind: str, fields: dict[str, Any]) -> None:
        try:
            emit(f"build.agent_{kind}", **fields)
        except Exception:  # noqa: BLE001 -- recording an event must not end the build
            log.exception("could not record the agent's %s event", kind)

    return on_event


def _reentry(clock: WallClock, emit: Emit) -> SessionReentry:
    """Re-entry on the runner's *clock*, with the fixed sentence, recorded through *emit*."""
    return SessionReentry(remaining_sec=clock.remaining_sec, prompt=lambda _sec: reentry_prompt(),
                          on_event=_emitter(emit))


def _form_texts(name: str, value: str) -> list[str]:
    """The texts the secret *name* holding *value* appears as, by the scrubber's rule
    (``forms_of``): the value, and the host of a URL that identifies a machine. Harbor's Claude
    Code resumes no session id holding one."""
    return [form.text for form in forms_of(name, value)]


def _output_cut(granted: Mapping[str, str]) -> Callable[[str, int], str]:
    """Harbor's cut of a failed command's output to its first characters for its error and its log
    (``cut_output``), backed off so no form of a *granted* value is left cut through
    (``back_off``): the scrubber that reads those records replaces whole forms only."""
    forms = registry_forms(granted)
    return lambda text, max_len: back_off(text[:max_len], forms, side="end")


@dataclass(frozen=True)
class AgentRun:
    """What ``build_agent`` gives a harness's builder beside the arm's record
    (``HarnessAdapter.build_agent``)."""

    #: The directory harbor logs the agent into.
    logs_dir: Path
    #: The MCP servers the agent registers; None for none.
    servers: list[MCPServerConfig] | None
    #: The sandbox directory of the task's skills; None for none.
    skills_dir: str | None
    #: What harbor sends with every exec request (``extra_env``), by variable: the values of the
    #: route's grants and the arm's plain settings.
    env: dict[str, str]
    #: The variables of ``env`` that hold the route's grants.
    secret_vars: tuple[str, ...]
    #: Whether the harness's goal mechanism is armed, and with it re-entry.
    goal: bool
    #: The turn cap; None for none.
    max_turns: int | None
    #: The runner's clock, which re-entry runs against.
    clock: WallClock
    #: Records the agent's events.
    emit: Emit

    @property
    def granted(self) -> dict[str, str]:
        """The values of the route's grants, by the variable of ``env`` that holds each."""
        return {var: self.env[var] for var in self.secret_vars}


def build_agent(arm: ArmRecord, *, logs_dir: Path, clock: WallClock, mcp: bool, skills_dir: str | None,
                goal: bool, max_turns: int | None, emit: Emit, secrets: Mapping[str, str]) -> Any:
    """The harbor agent that runs *arm*, logging into *logs_dir*, built by its harness's adapter
    (``claude_code_agent``, ``codex_agent``). *mcp* registers the budget server; *skills_dir* is
    the sandbox directory of the task's skills; *goal* arms the harness's goal mechanism, and
    with it re-entry on *clock* (for Claude Code only without *max_turns*); *emit* records the
    agent's events; *secrets* holds the values of the stage's grants by variable, which the
    adapter sends with every exec request beside the arm's plain settings (``extra_env``). Raise
    ValueError for an arm this cannot run exactly as recorded (the adapter's ``check``), and for
    a grant *secrets* holds no value for, naming the variable and never a value."""
    adapter = adapter_for(arm)
    try:
        adapter.check(arm)
    except ValueError as e:
        raise ValueError(f"arm {arm.name}: {e}") from None
    grants = adapter.grants(arm.route_table)
    missing = sorted(var for var in grants if var not in secrets)
    if missing:
        raise ValueError(f"arm {arm.name}: the stage holds no value for {missing}")
    run = AgentRun(logs_dir=logs_dir, servers=[budget_server()] if mcp else None, skills_dir=skills_dir,
                   env={**{var: secrets[var] for var in grants}, **adapter.plain_env(arm)},
                   secret_vars=tuple(grants), goal=goal, max_turns=max_turns, clock=clock, emit=emit)
    return adapter.build_agent(arm, run)


def claude_code_agent(adapter: HarnessAdapter, arm: ArmRecord, run: AgentRun) -> ClaudeCode:
    """Claude Code's agent for *arm* (module docstring), through its *adapter*."""
    return ClaudeCode(logs_dir=run.logs_dir, model_name=adapter.model_string(arm), max_turns=run.max_turns,
                      reasoning_effort=arm.effort, max_thinking_tokens=None, mcp_servers=run.servers,
                      skills_dir=run.skills_dir, extra_env=run.env, secret_env=run.secret_vars,
                      secret_forms=_form_texts, cut_output=_output_cut(run.granted),
                      # Harbor renders the value unquoted; the CLI also takes a comma.
                      disallowed_tools=",".join(CLAUDE_CODE_DENIED_TOOLS), inherit_process_env=False,
                      ralph_kickoff=ralph_goal_prompt() if run.goal else None,
                      # --max-turns counts per invocation, so each resume would grant a fresh cap.
                      reentry=_reentry(run.clock, run.emit) if run.goal and run.max_turns is None else None)


def codex_agent(adapter: HarnessAdapter, arm: ArmRecord, run: AgentRun) -> HlsCodex:
    """Codex's agent for *arm* (module docstring), through its *adapter*. Raise ValueError for a
    configuration value of the adapter's that ``HlsCodex`` sets no flag for. A turn cap, which
    codex has none of, is an event."""
    config = adapter.cli_config(arm)
    unknown = sorted(set(config) - set(CODEX_CONFIG_KEYS))
    if unknown:
        raise ValueError(f"arm {arm.name}: HlsCodex has no command-line setting for {unknown}")
    if run.max_turns is not None:
        run.emit("build.turn_cap_unsupported", harness=arm.harness, max_turns=run.max_turns)
    return HlsCodex(logs_dir=run.logs_dir, model_name=adapter.model_string(arm), mcp_servers=run.servers,
                    skills_dir=run.skills_dir, goal_kickoff=codex_goal_prompt() if run.goal else None,
                    reentry=_reentry(run.clock, run.emit) if run.goal else None,
                    extra_env=run.env, cut_output=_output_cut(run.granted),
                    reasoning_effort=arm.effort, **config)
