"""The harness's ``Model`` and ``Environment`` for the vendored mini-SWE-agent.

One trajectory is one Terminal-Bench task. The model reads a task statement and a
shell, and every turn it calls the ``bash`` tool once. The loop runs the command and
feeds the result back as a tool result. The trajectory ends when a command's output is
the submit sentinel, when the step limit is reached, when the context window would
overflow, when the model stops producing parseable calls, or when the server returns a
completion carrying no tokens.

The loop itself is not here. It is ``minisweagent.agents.default.DefaultAgent``,
forked from upstream at a named commit so it can be diffed; read that package's
``__init__`` for the provenance and the full divergence list. This module is the two
adapters upstream leaves to the integrator: a ``Model`` that reaches the artifact
through the harness's wire, and an ``Environment`` that reaches an Apptainer sandbox
through the harness's tool runner.

No ``hls`` imports, for the reason ``research_loop`` has none: the same file runs on
the harness's driver during a graded run and inside a build sandbox that has no
harness to import from. ``framework.py`` is the thin adapter that feeds it harness
envelopes and turns its per-call records into ``TraceEvent``s.

Fidelity to upstream mini-SWE-agent
-----------------------------------

Two upstream behaviours are kept because they shape the traffic, not just the outcome.
Each command runs in a fresh subshell, so ``cd`` and environment assignments do not
persist and the model has to re-establish its working directory every turn, which is
what makes the prompts grow the way they do. And submission is detected from a
command's *output* rather than from the text of the command, which is upstream's own
rule: the sentinel runs like any other command and the environment recognises it in
what came back.

What this loop does not do
--------------------------

It does not decide whether the task was solved. Under the report-KL gate the served
distribution is what is graded, so a trajectory that flails is as useful a measurement
as one that succeeds. Scoring a Terminal-Bench task means running its verifier in a
second container built from the task's own tests, which this harness never builds.

What stops a trajectory
-----------------------

The task scores makespan, so a trajectory that ends sooner scores better and every
number the stopping rules branch on has to be one the harness owns. Two of them are
about context: the window is declared by the task rather than read off the server's
``hls_context``, and the prompt's length is counted harness-side through the
checkpoint's own tokenizer rather than taken from the reply's ``usage.prompt_tokens``.
A server that reported a tiny window, or a long prompt, would otherwise end every
trajectory on its first turn and collect the speedup for it.
"""
from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel

from tasks.agentic_coding.minisweagent.agents.default import (
    AgentConfig, DefaultAgent, EmptyReply,
)
from tasks.agentic_coding.minisweagent.config.mini import (
    FORMAT_ERROR_TRUNCATED, FORMAT_ERROR_UNREADABLE, INSTANCE_TEMPLATE,
    SYSTEM_TEMPLATE, render_observation,
)
from tasks.agentic_coding.minisweagent.exceptions import (
    FormatError, Submitted,
)
from tasks.agentic_coding.minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL, format_toolcall_observation_messages, has_toolcall_delimiter,
    parse_toolcall_actions, parse_toolcall_text,
)

log = logging.getLogger(__name__)

#: The tool the loop calls. One command per turn, run in the instance's checkout.
BASH_TOOL_NAME = BASH_TOOL["function"]["name"]

#: What the model echoes to end the trajectory. Upstream mini-SWE-agent's sentinel,
#: kept verbatim so a published agent config reads the same here.
SUBMIT_SENTINEL = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"

#: Working directory commands run in when a trajectory names none. Most Terminal-Bench
#: images are built around ``/app``, and it is what the container backend itself falls
#: back to; a trajectory whose image uses another passes it in per session.
WORK_DIR = "/app"

#: Characters of command output fed back per COMMAND -- not per turn. Head and tail
#: are kept and the middle elided, because a failure's cause is usually at one end.
#:
#: Per command is upstream's shape: each output is rendered through the observation
#: template on its own, so one command's long output cannot starve another's. It
#: matters here because a reply usually carries two calls, so a single turn admits up
#: to twice this. See
#: ``minisweagent.config.mini.OBSERVATION_LONG`` for what that does to the context
#: arithmetic. Nothing is unbounded by it: the context guard counts the whole rendered
#: prompt before every send, so a turn that admitted two full observations is caught
#: there rather than here.
DEFAULT_OUTPUT_LIMIT_CHARS = 32_000

#: Replies in a row the loop can fail to read a call from before the trajectory ends.
#: Upstream's own name and default, and both halves of the name carry weight. It
#: counts *errors*, not repairs offered, so at 3 the third unreadable reply in a row
#: is the last turn rather than the third retry. And it is *consecutive*: it resets on
#: any clean step, so it bounds a model that cannot produce the format rather than
#: tallying every failure a long trajectory ever had.
DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS = 3

#: Wall-clock ceiling on one command, in seconds. Upstream's default. A test suite
#: that hangs would otherwise hold a trajectory open to the drive deadline.
DEFAULT_COMMAND_TIMEOUT_SEC = 300

#: Bytes allowed per message for the chat template's own delimiters, when a prompt has
#: to be bounded without a tokenizer. Far above any real template's per-message
#: wrapper on purpose: the bound is only sound while it cannot fall below the true
#: token count, and clamping an honest server would be worse than the slack it costs.
TEMPLATE_BYTES_PER_MESSAGE = 1024

#: Why a trajectory stopped. Recorded per session so the diagnostics can separate a
#: model that finished from one that ran out of room.
STOP_SUBMITTED = "submitted"
STOP_STEP_LIMIT = "step_limit"
STOP_CONTEXT_EXHAUSTED = "context_exhausted"
#: The repair budget ran out on replies the parser could not read. Named for the
#: parser because the parser is what a reader should go and look at.
STOP_FORMAT_REPAIR_EXHAUSTED = "format_repair_exhausted"
#: The same budget, spent by replies that carried no call for the parser to fail on.
#: A separate ending because it sends a reader somewhere else entirely: nothing is
#: wrong with the parser and the model is not producing calls.
STOP_NO_COMMAND_REPEATED = "no_command_repeated"

#: Why one turn produced no command, which is what the two endings above are told
#: apart by. Counted over the consecutive run that spent the budget, not the session:
#: a parser failure early in a trajectory says nothing about what ended it.
#:
#: The reply wrote a call delimiter and no readable action. The parser's case.
NO_COMMAND_UNPARSED = "unparsed_tool_call"
#: The reply ran the output cap out without writing a delimiter at all -- on this
#: checkpoint, a turn that spent its whole budget inside the reasoning channel.
NO_COMMAND_RUNAWAY = "runaway"
#: The reply finished on its own, inside the cap, and simply did not call anything.
NO_COMMAND_NO_CALL = "no_call"
#: Turns the resumed prefix had already counted against the budget. A prefix stores
#: how many there were and not what they were, so these cannot be classified, and a
#: run holding any of them keeps the parser's name for the same reason a run holding
#: a real one does: the reading that would hide a parser failure is the one to avoid.
NO_COMMAND_INHERITED = "inherited"
#: A completion that carried no tokens at all. The one stop that names the server
#: rather than the model: every other way a turn fails to yield a call is the model
#: answering badly, which is asked again, and calling those "empty" would report a
#: starved turn as a broken artifact.
STOP_EMPTY_REPLY = "empty_reply"
#: A trajectory the drive's scheduled stop ended between turns: the drive runs in a
#: bounded wall-clock window and this session wanted a turn after the window closed.
#: Neither the server nor the model, and not a failure of either -- every trajectory
#: still working when a bounded drive ends stops this way.
STOP_WALL_CLOCK = "wall_clock"
#: A trajectory the sandbox ended: enough commands in a row came back as sandbox
#: faults rather than as failed commands that the container is taken to be gone.
#:
#: Its own ending because every other reading of it is wrong in a way that hides the
#: cause. The tool runner reports a broken sandbox to the model as a failed command,
#: which is right for a blip but lets a dead exec server serve failed commands turn
#: after turn. Left to run, such a session ends on the repair budget and reports the
#: model failing to produce calls, which sends a reader to the parser instead of to
#: the container.
STOP_SANDBOX_FAULT = "sandbox_fault"

#: Opening of the message on a trajectory that ended this way. The harness raises the
#: same fault from its tool runner when a container cannot be replaced, and a reader
#: looking through run records for a dead sandbox should not have to know which layer
#: noticed. ``hls.core.faults.SANDBOX_GONE`` is the same string, spelled out here for
#: the reason at the top of this module -- nothing here may import ``hls`` -- and a
#: test holds the two together.
SANDBOX_GONE = "sandbox gone:"

#: Characters of a quoted error kept in that message. The whole message is recorded in
#: one field of the run record, and the text being quoted is a command's output.
ERROR_CHARS = 200

#: A trajectory the caller's own ``before_turn`` hook ended. The offline pre-roll is
#: the only caller that uses one: it bounds a pass in wall clock and in spend and cuts
#: by raising out of the hook, which is a scheduled stop and not a fault in the
#: trajectory. Its turns read as an error otherwise, while the pass's own summary said
#: it stopped on its deadline, and two records disagreeing about one trajectory gets
#: resolved in favour of whichever one the reader opened.
#:
#: One reason for all of the caller's bounds, because which bound it was belongs to
#: the caller and is already in the caller's result. The loop only has to stop
#: claiming the trajectory broke.
STOP_CALLER_STOPPED = "caller_stopped"
#: A trajectory that raised on its way out. Upstream stamps the exception's own class
#: name as the exit status, which is not one of the endings above; naming those
#: separately keeps a crashed sandbox from being counted as a trajectory that quietly
#: ran out of steps.
STOP_ERROR = "error"

#: Upstream names an ending by the ``exit_status`` it stamps on the final message; the
#: diagnostics name it by the constants above. ``LimitsExceeded`` covers upstream's
#: step and cost limits, and cost is disabled here, so the only limit left is steps.
#: Anything not named here is an exception's class name, which is ``STOP_ERROR``.
#:
#: ``DriveWallClockReached`` is the harness's, raised into the model call when the
#: drive stops issuing requests; ``hls.spec.agent`` defines it and a test holds the
#: two names together. It is matched as a string for the reason at the top of this
#: module: nothing here may import ``hls``.
EXIT_STATUS_TO_STOP_REASON = {
    "Submitted": STOP_SUBMITTED,
    "LimitsExceeded": STOP_STEP_LIMIT,
    "ContextExhausted": STOP_CONTEXT_EXHAUSTED,
    "RepeatedFormatError": STOP_FORMAT_REPAIR_EXHAUSTED,
    "EmptyReply": STOP_EMPTY_REPLY,
    "DriveWallClockReached": STOP_WALL_CLOCK,
    "SandboxFault": STOP_SANDBOX_FAULT,
    # The tool runner's, raised when a session's container could not be replaced
    # inside its restart budget. Matched as a string for the reason at the top of this
    # module -- nothing here may import ``hls`` -- and ``hls.core.faults`` defines it;
    # a test holds the two names together. Mapped at all so the ending names the
    # sandbox: without it the class name is unrecognised and the trajectory is
    # recorded as a generic error, which is a stop reason naming the wrong cause.
    "SandboxGone": STOP_SANDBOX_FAULT,
}

#: Sandbox faults in a row that end a trajectory, or 0 to never end one on them.
#:
#: Zero by default, so no path changes behaviour until a caller chooses it: the
#: runner's own contract is that a broken sandbox reaches the model as a failed
#: command, and that is load-bearing for a blip. What it is not sized for is a
#: container that has stopped executing anything, where every later turn is the
#: harness reporting a corpse.
#:
#: Setting it is also how a caller says which it would rather lose. Ending here drops
#: the session: the driver keeps the turns it served and writes the drop and its reason
#: into the trace (``hls.trace.store.TraceEnd.dropped``), and the drive is scored on the
#: shorter trajectory. Zero keeps the trajectory running to one of its own endings, with
#: every fault visible to the model as a failed command.
DEFAULT_MAX_CONSECUTIVE_SANDBOX_FAULTS = 0

#: Key each turn's prompt length is recorded under, in tokens and counted
#: harness-side. ``StreamingLatencySLO`` reads it to grow a request's TTFT budget with
#: its prefill; ``hls.spec.workload.PREFILL_TOKENS_EXTRA`` is the same string, and a
#: test holds the two together. Spelled out rather than imported for the reason at the
#: top of this module: nothing here may import ``hls``.
#:
#: A turn carries it twice, under this one key: in the event's extras, and in the
#: request's metadata for the driver to find when the drive's stop cuts the turn
#: before it can be recorded.
PREFILL_TOKENS_EXTRA = "prefill_tokens"

#: Set on the event of a phase that was cut at the thinking cap and handed back to the
#: model to finish. Both phases are served requests and each carries its own event, so
#: a reader needs to know which is which: the cut one produced no command by design,
#: and counting it as a turn that failed to act would end trajectories that are working.
THINK_CAP_EXTRA = "hit_think_cap"
#: Set beside it on the phase that was continued, so the pair can be found in a trace
#: without inferring it from adjacency.
CONTINUED_EXTRA = "continued"
#: Set on a cut phase that could NOT be handed back: the checkpoint's template could not
#: render the carried text as the resumed message (see ``MiniSweLoop._continuation``),
#: so no second request was sent and the turn ended as a reply that ran to its cap.
CONTINUATION_UNRENDERABLE_EXTRA = "continuation_unrenderable"

#: The reasoning channel's delimiters, as they appear in the reply text.
#:
#: Contestants serve raw -- no reasoning parser, no tool-call parser -- and
#: ``framework.check_text_matches_tokens`` refuses a reply that is not the exact text
#: of its own tokens, so a stripped reply fails the run rather than reaching here. The
#: tolerance below for an already-stripped reply is therefore belt-and-braces and NOT
#: a supported configuration: a raw reply passes through it unchanged, so it costs
#: nothing to keep, and it is a sensible margin against a reply that arrives shaped
#: some other way. A later reader should not read it as an endorsement of reasoning
#: parsers; the contract check is what decides that, and it rejects one.
#:
#: An opening delimiter after a reply has closed its thought is text, not a second
#: thought: this checkpoint's template emits the content of an assistant turn untouched
#: whenever the reasoning field is set, so the serving stack reads it as text too, and
#: an agent writing shell commands is the ordinary way to produce one. ``split_reply``
#: keeps it.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def split_reply(text: str, *, thinking_opened: bool = False) -> tuple[str, str]:
    """One reply as ``(visible, reasoning)``.

    ``thinking_opened`` says whether the prompt this reply answered ends inside an
    open thought, which is what a chat template with thinking on does: it emits the
    opening delimiter as part of the *prompt*, so the reply begins already inside the
    reasoning. GLM-5.3-Flash's replies look like ``...one sentence,
    short.</think><tool_call>bash...``. The harness renders that prompt, so this is a
    fact passed in rather than guessed from the reply -- and it has to be, because a
    reply that never left its thought and a reply an engine already stripped are the
    same string.

    Under an open thought there is one rule: the reply splits at its FIRST closing
    delimiter, everything before is the thought, everything after is the reply
    verbatim. A reply with no closing delimiter never left the thought, so all of it
    is the thought and there is no reply yet.

    Nothing is taken out of either half. Rendering this checkpoint's own template shows
    why: with the reasoning field set, the template emits the content untouched, and it
    splits content on a closing delimiter only when that field is absent. So an opening
    delimiter inside the reply is literal text to the serving stack -- a shell command
    that mentions one is the ordinary way to get it -- and a replay that stripped it
    would render a shorter sequence than the artifact was served. The rule that a call
    drafted while reasoning is not one the model asked for is carried entirely by the
    split: everything written before the close is on the reasoning side and never
    reaches the parser.

    With no thought open, the reply is one whose reasoning an engine has already taken
    into a field of its own. The paired-span handling below is the tolerance for that
    shape described at ``THINK_OPEN``; it is not a supported configuration, and the
    contract check refuses the server that produces it."""
    if thinking_opened:
        thought, closed, rest = text.partition(THINK_CLOSE)
        return (rest.strip(), thought.strip()) if closed else ("", text.strip())
    lead = None
    opened = text.find(THINK_OPEN)
    closed = text.find(THINK_CLOSE)
    if closed != -1 and (opened == -1 or closed < opened):
        lead = text[:closed]
        text = text[closed + len(THINK_CLOSE):]
    spans = [s[len(THINK_OPEN):-len(THINK_CLOSE)] for s in _THINK_RE.findall(text)]
    text = _THINK_RE.sub("", text)
    head, sep, tail = text.partition(THINK_OPEN)
    if sep:
        spans.append(tail)
        text = head
    return (text.strip(), (lead if lead is not None else "\n".join(spans)).strip())


def visible_text(text: str, *, thinking_opened: bool = False) -> str:
    """The reply with its reasoning removed, whichever way the channel was opened.

    The reply this is built for is the raw one every contestant serves. On an
    already-stripped reply it is a no-op, which is why it can run over every reply
    rather than only the raw ones -- and why tolerating both shapes costs nothing,
    though only the raw shape is allowed. See :func:`split_reply` for the cases."""
    return split_reply(text, thinking_opened=thinking_opened)[0]


def reasoning_text(text: str, *, thinking_opened: bool = False) -> str:
    """The reasoning a reply wrote, as the history has to echo it back.

    The complement of :func:`visible_text`. It exists because the chat template renders
    an assistant turn's ``reasoning_content`` into its own ``<think>`` span: on
    this checkpoint's own template, a history carrying the reasoning renders
    ``<think>thought</think>reply`` and one that dropped it renders
    ``<think></think>reply``. So dropping it does not merely shorten the prompt, it
    replaces the thought with an empty one, and the oracle teacher-forces that
    prompt."""
    return split_reply(text, thinking_opened=thinking_opened)[1]


def prompt_byte_bound(messages: list[dict]) -> int:
    """An upper bound on the tokens this conversation's prompt renders to.

    Every token covers at least one byte, so a prompt's UTF-8 length bounds its token
    count. The rendered prompt needs a tokenizer, which is the case this exists for, so
    the parts are bounded instead: each message's own JSON, and a per-message allowance
    for the delimiters a template would add around it."""
    total = TEMPLATE_BYTES_PER_MESSAGE
    for message in messages:
        total += len(json.dumps(message, default=str).encode())
        total += TEMPLATE_BYTES_PER_MESSAGE
    return total


def wire_messages(messages: list[dict]) -> list[dict]:
    """The conversation as it goes on the wire.

    Upstream's ``_prepare_messages_for_api``: every message minus the ``extra`` key,
    which is the agent's own bookkeeping and not something a chat template renders.
    Sending it would put the whole previous response object into the next prompt."""
    return [{k: v for k, v in m.items() if k != "extra"} for m in messages]


def no_command_cause(extras: dict) -> str | None:
    """Why this turn produced no command, or None if it produced one.

    Read off the extras the turn already records rather than passed down from the
    parse, so there is one definition of what a turn did and the recorder does not
    have to be told twice. A reply that wrote a delimiter the parser refused is the
    parser's; one that ran out its cap without writing one is a runaway; anything else
    is a model that finished and did not call."""
    if extras.get("parsed_command"):
        return None
    if extras.get(CONTINUED_EXTRA):
        # Cut at the thinking cap and handed back to the model to finish. It produced
        # no command because it was stopped on purpose, and the continuation is what
        # answers; counting it here would spend the repair budget on turns that are
        # working and end trajectories three caps in.
        return None
    if extras.get("unparsed_tool_call"):
        return NO_COMMAND_UNPARSED
    return NO_COMMAND_RUNAWAY if extras.get("ran_to_output_cap") else NO_COMMAND_NO_CALL


@dataclass
class TurnRecorder:
    """One trace event per LLM call, with the last one held back.

    ``emit`` is one event per LLM call, so the trajectory's verdict cannot be an event
    of its own: that would offer a per-call verifier a record with no served tokens
    behind it. It is stamped onto the last turn's extras instead, so a metric grouping
    events by session can read how the trajectory ended without tracking which turn was
    last. That needs the last event held back until the loop is over, which is what
    ``open`` and ``flush`` do below -- a trajectory that runs out of steps ends on an
    ordinary turn that nothing would otherwise mark.

    A turn is opened when the model answers and amended when the command it asked for
    has run, because the event carries both."""

    emit: Callable
    session_id: str
    #: The turn awaiting either an amendment or the next turn's arrival.
    pending: dict | None = None
    #: Turns that reached the server. Not the loop's step count: the context guard
    #: ends an iteration before it sends, and that iteration is not a turn.
    #:
    #: This session's, counted from zero even when it resumed from a frozen
    #: conversation. The driver numbers a session's own fault events from zero too
    #: (``TrajectorySessionDriver``), so a recorder that started at the resume depth
    #: would leave one session's events numbered two ways.
    turns: int = 0
    #: Commands the environment has run, including the pre-roll's where this session
    #: resumed from a prefix. Seeded rather than counted from zero so the trajectory
    #: shape reads as one trajectory.
    commands_run: int = 0
    #: Turns the prefix this session resumed from had already served, or 0 for a
    #: session that started cold. Stamped on every event, so the diagnostics can
    #: recover the conversation's true depth from a session-local turn index.
    prefix_turns: int = 0
    #: Causes of the unbroken run of turns that produced no command, oldest first,
    #: cleared by any turn that produced one. The repair budget counts exactly this
    #: run, so when it ends a trajectory this is what it was spent on -- and naming
    #: the ending needs it, because "three turns in a row produced no command" does
    #: not say whether the parser failed or the model never called.
    no_command_run: list[str] = field(default_factory=list)
    #: Turns the prefix this session resumed from had already counted against the
    #: repair budget. Seeded because the budget carries over, unclassifiable because
    #: a prefix records the count and not the causes.
    inherited_no_command: int = 0

    async def open(self, request: dict, completion: dict, t_send: float,
                   t_recv: float, extras: dict) -> None:
        """Emit the turn held so far and hold this one in its place."""
        if self.pending is not None:
            await self.emit(self.pending)
        cause = no_command_cause(extras)
        if cause is None:
            self.no_command_run.clear()
        else:
            self.no_command_run.append(cause)
        turn_idx = self.turns
        self.turns += 1
        self.pending = {"request_id": f"{self.session_id}.{turn_idx}",
                        "session_id": self.session_id, "turn_index": turn_idx,
                        "request": request, "completion": completion,
                        "t_send": t_send, "t_recv": t_recv,
                        "extras": {**extras, "commands_run": self.commands_run,
                                   "prefix_turns": self.prefix_turns}}

    def note_command(self, command: str, result: dict, output_limit: int) -> None:
        """Record what the held turn's command did."""
        if self.pending is None:
            return
        self.commands_run += 1
        output = str(result.get("output", ""))
        self.pending["extras"].update(
            parsed_command=command, exit_code=int(result.get("returncode", 0)),
            output_chars=len(output), output_truncated=len(output) > output_limit,
            commands_run=self.commands_run,
            # Containers this session has had replaced under it. On the event rather
            # than only in the runner, because a session whose sandbox died twice
            # produced its traffic under conditions nobody intended and the drive has
            # to be able to see that rather than average it in. Always present, so a
            # reader never has to tell a missing key from a zero.
            sandbox_restarts=int(result.get("sandbox_restarts") or 0),
            # And what it was replaced for, on the turn that replaced it. Empty on
            # every other turn. Without this a recovered fault leaves a count and no
            # cause: restarts make a fault survivable and therefore invisible.
            sandbox_restart_cause=str(result.get("sandbox_restart_cause") or ""),
            timed_out=bool(result.get("timed_out", False)))

    def repair_verdict(self, stop_reason: str) -> tuple[str, dict[str, int]]:
        """The ending, refined by what the repair budget was actually spent on, and
        the breakdown of that run.

        Only an ending the budget caused has a run to describe; every other
        trajectory reports an empty breakdown rather than a claim about itself.

        A run holding even one unreadable reply keeps the parser's name. The reading
        that would hide a real parser failure is the one this exists to remove, so a
        mixed run resolves toward the parser and the counts say how many of each."""
        if stop_reason != STOP_FORMAT_REPAIR_EXHAUSTED:
            return (stop_reason, {})
        causes = Counter(self.no_command_run)
        if self.inherited_no_command:
            causes[NO_COMMAND_INHERITED] = self.inherited_no_command
        if NO_COMMAND_UNPARSED in causes or NO_COMMAND_INHERITED in causes:
            return (STOP_FORMAT_REPAIR_EXHAUSTED, dict(causes))
        return (STOP_NO_COMMAND_REPEATED, dict(causes))

    async def flush(self, stop_reason: str, causes: dict[str, int]) -> None:
        """Emit the held turn, stamped with the trajectory's verdict."""
        if self.pending is None:
            return
        self.pending["extras"].update(stop_reason=stop_reason,
                                      no_command_causes=causes,
                                      session_turns=self.turns)
        await self.emit(self.pending)
        self.pending = None


class HarnessModelConfig(BaseModel):
    """The settings ``HarnessModel`` reports and renders with.

    Serializable only: the callables that reach the harness are constructor arguments,
    the way upstream's litellm client is a module-level import rather than config."""

    #: Characters of command output returned to the model per turn.
    output_limit_chars: int = DEFAULT_OUTPUT_LIMIT_CHARS


class HarnessModel:
    """Upstream's ``Model`` over the harness's wire.

    Upstream's litellm client sends ``tools=[BASH_TOOL]`` and reads the structured
    ``tool_calls`` off the reply. This one sends the same tool definitions -- they ride
    on the request's metadata, which is where the harness's wire protocol, the
    reference engines and the verifier all look for them, so the artifact and the
    oracle render one prompt -- and parses the call out of the reply text. The reason
    is in ``actions_toolcall``'s module docstring.
    """

    def __init__(self, *, session_id: str, base_metadata: dict, complete: Callable,
                 recorder: TurnRecorder, count_prompt_tokens: Callable | None = None,
                 max_output_tokens: int | None = None,
                 reply_budget: int | None = None, thinking_opened: bool = False,
                 config_class: type = HarnessModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.session_id = session_id
        self.base_metadata = dict(base_metadata)
        self.complete = complete
        self.recorder = recorder
        self._count_prompt_tokens = count_prompt_tokens
        self.max_output_tokens = max_output_tokens
        #: Tokens the continuation of a cut reply may spend, or None for no second
        #: phase at all. Separate from ``max_output_tokens`` because the two bound
        #: different things: that one bounds thinking, this one bounds the visible
        #: reply that follows it, and the visible half is far shorter.
        self.reply_budget = reply_budget
        #: Whether the prompt these replies answer ends inside an open thought, which
        #: is what tells a reply that never closed one from a reply that had none.
        #: Measured by the framework from the render, never guessed from a reply.
        self.thinking_opened = thinking_opened
        #: What the last reply said about its own prompt, and how many messages that
        #: prompt was. Only the no-tokenizer fallback in ``count_prompt_tokens`` reads
        #: them.
        self._reported: object = None
        self._reported_prefix = 0
        #: The last conversation the tokenizer counted, as ``(messages, tokens)``, and
        #: None before the first count. See ``counted_prompt_tokens``.
        self._counted: tuple[int, int | None] | None = None

    # --- upstream's Model protocol ---
    def format_message(self, **kwargs) -> dict:
        """Upstream's, minus the multimodal expansion: command output is text, and the
        task declares ``output_modalities={"text"}``."""
        return dict(kwargs)

    def get_template_vars(self, **kwargs) -> dict:
        return {**self.config.model_dump(), **kwargs}

    def format_observation_messages(self, message: dict, outputs: list[dict],
                                    template_vars: dict | None = None) -> list[dict]:
        """Upstream's, with the observation rendered in Python rather than jinja."""
        return format_toolcall_observation_messages(
            actions=message.get("extra", {}).get("actions", []),
            outputs=outputs,
            render_observation=lambda output: render_observation(
                output, output_limit=self.config.output_limit_chars))

    async def query(self, messages: list[dict], **kwargs) -> dict:
        """Send the conversation, parse the call out of the reply, record the turn.

        Upstream's invariant is preserved: the response is persisted on the
        ``FormatError`` message's ``extra`` before the error propagates, so a turn the
        parser could not read is still in the trace. Cost is not, because nothing
        bills -- see ``AgentConfig.cost_limit``."""
        wire = wire_messages(messages)
        sent = len(messages)
        # Before the clock starts. The context guard has almost always counted this
        # same conversation already, so this is free, but tokenizing a long transcript
        # is seconds of CPU and none of it is the server's latency.
        prefill = await self.counted_prompt_tokens(wire)
        request = {"request_id": f"{self.session_id}.{self.recorder.turns}",
                   "text": messages[-1].get("content") or "",
                   "metadata": {**self.base_metadata, "messages": wire,
                                "tools": [BASH_TOOL],
                                # The same count the event below carries, on the
                                # request too: a turn the drive's scheduled stop cuts
                                # never reaches the recorder, and the driver builds
                                # that turn's event out of the envelope alone.
                                PREFILL_TOKENS_EXTRA: prefill}}
        # Monotonic, the clock the harness measures on. The driver keeps the later of
        # this and its own send time, and the transport stamps packet arrivals the same
        # way, so a wall-clock stamp here would win that comparison and make every
        # latency negative.
        t_send = time.monotonic()
        completion = await self.complete(request)
        t_recv = time.monotonic()

        cap_extras: dict = {}
        if self.reply_budget and self.ran_to_output_cap(completion):
            continuation = await self._continuation(wire, completion.get("text") or "")
            if continuation is None:
                # The template cannot render the carried text as the resumed message, so
                # there is nothing faithful to hand back and no second request is sent.
                # The turn ends here as a reply that ran to its cap, and its one event
                # says why, so a reader does not take it for a continuation that failed.
                cap_extras = {THINK_CAP_EXTRA: True, CONTINUED_EXTRA: False,
                              CONTINUATION_UNRENDERABLE_EXTRA: True}
            else:
                # The cap fired. Record this phase as the served request it was -- its
                # own event, priced against its own prompt -- and then hand the reply
                # back for the model to finish. Both halves are traffic the server
                # produced and the gate grades; folding them into one event would price
                # the pair against the first prompt and hide the second entirely.
                await self.recorder.open(
                    request, completion, t_send, t_recv,
                    {"ran_to_output_cap": True, "had_visible_text": False,
                     "attempted_tool_call": False, PREFILL_TOKENS_EXTRA: prefill,
                     THINK_CAP_EXTRA: True, CONTINUED_EXTRA: True,
                     "parsed_command": None})
                request, completion, t_send, t_recv, prefill = await self._continue(
                    *continuation)

        self._reported = (completion.get("metadata") or {}).get("prompt_tokens")
        self._reported_prefix = sent

        text = completion.get("text") or ""
        visible, reasoning = split_reply(text, thinking_opened=self.thinking_opened)
        truncated = self.ran_to_output_cap(completion)
        # Whether the reply TRIED to call, which is a different question from whether
        # a call came out of it. See ``has_toolcall_delimiter``.
        attempted = has_toolcall_delimiter(visible)

        async def record(extras: dict) -> None:
            await self.recorder.open(request, completion, t_send, t_recv,
                                     {"ran_to_output_cap": truncated,
                                      "had_visible_text": bool(visible),
                                      "attempted_tool_call": attempted,
                                      PREFILL_TOKENS_EXTRA: prefill, **cap_extras,
                                      **extras})

        # A completion that carried no tokens at all is a server that did not answer
        # rather than a model that answered badly, and asking again would spend the
        # trajectory on a server that is not replying.
        if not self.carried_tokens(completion):
            await record({"parsed_command": None})
            raise EmptyReply({"role": "exit", "content": "EmptyReply",
                              "extra": {"exit_status": "EmptyReply", "submission": ""}})

        try:
            actions = parse_toolcall_actions(
                parse_toolcall_text(visible,
                                    call_id_prefix=f"call_{self.recorder.turns}"),
                format_error_template=self.repair_template(completion),
                template_kwargs={"submit_sentinel": SUBMIT_SENTINEL})
        except FormatError as e:
            # Upstream's rule, in upstream's place: the reply is persisted on the
            # error's own message, because the call was served before parsing failed
            # and a turn the artifact spent time on belongs in the trace either way.
            e.messages[0].setdefault("extra", {})["response"] = dict(completion)
            # A reply that wrote a delimiter and still produced no action is a parser
            # suspect, not merely a model that would not act -- but only if it
            # finished. One the output cap cut off mid-call has demonstrated nothing
            # about the parser: half a command is not an action, which is what the
            # parser documents. This field is the only alarm for our own parser, so a
            # runaway that happened to open a call must not set it off; it is believed
            # once and dismissed afterwards.
            await record({"parsed_command": None,
                          "unparsed_tool_call": attempted and not truncated})
            raise

        # The visible reply goes back verbatim, calls and all, carrying no structured
        # ``tool_calls`` field. See the note on the class.
        #
        # Upstream's timestamp, and wall clock on purpose: it dates the message for a
        # reader, and ``wire_messages`` drops ``extra`` before the next prompt, so
        # nothing measures a duration with it. The two stamps above are the ones the
        # harness reads, and those are monotonic.
        message = {"role": "assistant", "content": visible,
                   "extra": {"actions": actions, "response": dict(completion),
                             "timestamp": time.time()}}
        if reasoning:
            # Spelled the way this checkpoint's chat template reads it, so the next
            # prompt carries the reasoning the artifact actually served.
            message["reasoning_content"] = reasoning
        await record({"parsed_command": actions[0]["command"] if actions else None,
                      "n_tool_calls": len(actions)})
        return message

    # --- ours ---
    def repair_template(self, completion: dict) -> str:
        """Which of the two repair messages this turn has earned.

        Upstream branches in jinja on the server's ``finish_reason``. The branch is
        taken from the token count here instead, because the two messages give
        opposite advice -- reformat, or be shorter -- and which one a turn deserves
        must not be a string the server chose. ``ran_to_output_cap`` falls back to the
        label only when no count arrived at all."""
        return (FORMAT_ERROR_TRUNCATED if self.ran_to_output_cap(completion)
                else FORMAT_ERROR_UNREADABLE)

    async def _continuation(self, wire: list[dict], cut: str):
        """The resumed message a cut reply is handed back as, or None when the template
        cannot render it faithfully.

        The conversation plus one assistant message holding exactly what the server
        produced, with ``</think>`` appended when the cut landed inside the thought.
        Nothing is invented: the checkpoint's template renders an assistant message
        whose content begins inside a thought identically whether or not an opening
        tag is supplied, so the loop carries the served text and adds only the close.

        The prompt this produces is the first phase's prompt with that text after it,
        byte for byte and token for token, which is what lets the reference reproduce
        it (``tests/test_continuation_render.py``). That claim holds only while the
        template renders the carried text verbatim, and this checkpoint's does not when
        the thought itself contains a literal ``<think>`` or ``</think>``: it keeps the
        text after the last opening tag and before the first closing one, so the resumed
        message would silently be a different one from the one served. transformers
        detects that (``continue_final_message`` requires the final message to appear
        in the render) and raises ValueError from the prompt count; a server rendering
        the same template would refuse the request the same way. Uncaught, that error
        raises out of the driver and loses the whole drive. Returns None in that case,
        so the caller ends the turn as a reply that ran to its cap rather than sending
        a continuation nobody can render. Returns ``(cont_wire, carried, prefill)``
        otherwise."""
        carried = cut + ("" if THINK_CLOSE in cut else THINK_CLOSE)
        cont_wire = list(wire) + [{"role": "assistant", "content": carried}]
        try:
            prefill = await self.counted_prompt_tokens(cont_wire,
                                                       continue_final_message=True)
        except ValueError as exc:
            log.warning("agentic_coding: %s: the cut reply cannot be handed back, the template "
                        "does not render the carried text as the resumed message (%s); "
                        "the turn ends at the cap", self.session_id, one_line(str(exc))[:200])
            return None
        return cont_wire, carried, prefill

    async def _continue(self, cont_wire: list[dict], carried: str, prefill):
        """Send the continuation ``_continuation`` built and return that phase.
        ``continue_final_message`` is what tells the server to resume the message
        rather than open a new turn."""
        request = {"request_id": f"{self.session_id}.{self.recorder.turns}",
                   "text": carried,
                   "metadata": {**self.base_metadata, "messages": cont_wire,
                                "tools": [BASH_TOOL],
                                "continue_final_message": True,
                                "max_tokens": self.reply_budget,
                                PREFILL_TOKENS_EXTRA: prefill}}
        t_send = time.monotonic()
        completion = await self.complete(request)
        t_recv = time.monotonic()
        # The turn's reply is both halves: the model's own text, the tag the loop
        # added, and what it wrote once it had been given the tag. The history keeps
        # that whole string, so the next turn sees what was actually served.
        completion = {**completion,
                      "text": carried + (completion.get("text") or "")}
        return request, completion, t_send, t_recv, prefill

    def ran_to_output_cap(self, completion: dict) -> bool:
        """Whether a reply spent the whole output cap it was given.

        Decided on the tokens the reply carried rather than its ``finish_reason``: the
        label is a string the server chooses and the count is not. Recorded per turn so
        the diagnostics can tell a cap truncation from a model that stopped on its own,
        which are the two ways a turn arrives with nothing visible in it.

        ``finish_reason`` decides where no token count reached the loop, which is the
        build sandbox's self-test rather than a graded run: the contract makes
        ``token_ids`` mandatory, so a served reply always carries its own length."""
        cap = self.max_output_tokens
        served = completion.get("served_tokens")
        if isinstance(served, int) and isinstance(cap, int) and cap > 0:
            return served >= cap
        return str(completion.get("finish_reason") or "").casefold() == "length"

    def carried_tokens(self, completion: dict) -> bool:
        """Whether a reply carried any tokens at all, visible or not.

        This is what separates a starved turn from an empty one, and the cap does not
        come into it. A turn can reason for tens of thousands of tokens, reach no
        answer and stop on its own well short of the cap; that is a starved turn, not
        a server saying nothing.

        Repairing on any non-empty reply is safe because the repair budget bounds it.
        A server cannot buy turns by answering cheaply: it gets at most
        ``max_consecutive_format_errors`` of them before the trajectory ends either
        way."""
        served = completion.get("served_tokens")
        if isinstance(served, int):
            return served > 0
        if str(completion.get("finish_reason") or "").casefold() == "length":
            return True
        # No count and no label, so fall back to the reply itself. Raw text, not the
        # visible part: reasoning the server left inline is still tokens it served.
        return bool(str(completion.get("text") or "").strip())

    async def counted_prompt_tokens(self, wire: list[dict],
                                    continue_final_message: bool = False) -> int | None:
        """The tokenizer's count of this prompt, or None where nothing can count it.

        The harness's own number and only that, where ``count_prompt_tokens`` below
        falls back to bounding the reply's. Two readers want it for opposite reasons
        and both want it unadulterated: the context guard ends a trajectory on it, and
        the trace event carries it to a TTFT budget that grows with prefill. Either
        way a server that inflated its reported prompt would be buying something.

        Counted once per turn. The guard asks immediately before the send and the
        event's stamp asks immediately after, so the answer is remembered and the
        second ask is free. Keyed on the conversation's length, which identifies it
        within a trajectory: a transcript only ever grows, one message at a time."""
        if self._count_prompt_tokens is None:
            return None
        # Keyed on the continuation flag too: a continued prompt has one more message
        # and no generation prompt, so it is a different length from the turn it
        # resumes even though both are "this turn".
        key = (len(wire), continue_final_message)
        if self._counted is not None and self._counted[0] == key:
            return self._counted[1]
        counted = await self._count_prompt_tokens(wire, continue_final_message)
        counted = None if counted is None else int(counted)
        self._counted = (key, counted)
        return counted

    async def count_prompt_tokens(self, messages: list[dict]) -> int | None:
        """How long the prompt these messages render to, or None when nothing can say.

        Not the reply's ``usage.prompt_tokens``. The guard that reads this ends a
        trajectory, and the task scores how long a trajectory takes, so a server that
        inflates its reported prompt length ends every trajectory on its first turn and
        collects the speedup for it. The injected counter renders the conversation the
        harness sent through the checkpoint's own tokenizer, which is the same count a
        trusted calibration member reports for the same turn.

        Where nothing can count -- a unit test, a process with no checkpoint mounted --
        the last reply's figure is carried forward instead: what the server said about
        the first ``_reported_prefix`` messages it was sent, plus a byte bound on
        everything appended since, the whole capped at the byte bound of the
        conversation. A token covers at least one byte, so that cap cannot bite an
        honest server, which runs three to four bytes a token; it only narrows an
        inflated figure rather than removing it, which is why it is the fallback and
        not the rule."""
        wire = wire_messages(messages)
        counted = await self.counted_prompt_tokens(wire)
        if counted is not None:
            return counted
        reported = self._reported
        if not isinstance(reported, (int, float)) or isinstance(reported, bool):
            return None
        carried = float(reported) + prompt_byte_bound(wire[self._reported_prefix:])
        return int(min(carried, float(prompt_byte_bound(wire))))


class HarnessEnvironmentConfig(BaseModel):
    """The settings ``HarnessEnvironment`` reports."""

    #: Wall-clock ceiling on one command, passed to the tool runner.
    command_timeout_sec: int = DEFAULT_COMMAND_TIMEOUT_SEC
    #: Working directory this trajectory's commands run in, named in the opening
    #: prompt so the model knows where it landed. Per trajectory because each
    #: Terminal-Bench task brings its own image and each image its own directory.
    work_dir: str = WORK_DIR
    #: Sandbox faults in a row that end the trajectory. See the constant.
    max_consecutive_sandbox_faults: int = DEFAULT_MAX_CONSECUTIVE_SANDBOX_FAULTS


class HarnessEnvironment:
    """Upstream's ``Environment`` over the harness's tool runner.

    The Sec.9 trust invariant is what puts the command here rather than in the
    artifact: the artifact serves one reply per HTTP call and never runs anything, so
    the wall time of every command counts against the trajectory makespan and a
    contestant cannot buy latency by pretending a test suite ran instantly.
    """

    def __init__(self, *, run_tool: Callable, recorder: TurnRecorder,
                 output_limit_chars: int = DEFAULT_OUTPUT_LIMIT_CHARS,
                 config_class: type = HarnessEnvironmentConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.run_tool = run_tool
        self.recorder = recorder
        self.output_limit_chars = output_limit_chars
        #: Sandbox faults since the last command that ran. Consecutive rather than
        #: cumulative: a session failing once every twenty turns is limping, and
        #: ending it would throw away a usable trajectory.
        self.consecutive_sandbox_faults = 0

    def get_template_vars(self, **kwargs) -> dict:
        return {**self.config.model_dump(), **kwargs}

    async def execute(self, action: dict, cwd: str = "") -> dict:
        """Run one command in this trajectory's sandbox.

        The result is upstream's shape -- ``output``, ``returncode``,
        ``exception_info`` -- because ``OBSERVATION_TEMPLATE`` renders those names."""
        result = await self.run_tool(BASH_TOOL_NAME,
                                     {"command": action["command"],
                                      "timeout_sec": self.config.command_timeout_sec})
        output = {"output": str(result.get("output", "")),
                  "returncode": int(result.get("exit_code", 0)),
                  "exception_info": ""}
        # Carried under its own name: the runner calls it ``restarted`` on a command's
        # result and the recorder records a session property, so the rename happens
        # where the two meanings meet rather than at either end.
        #
        # Indexed rather than defaulted. A default here cannot tell a session with no
        # restarts from a writer that dropped the field, and would record zero
        # restarts for a session that had one.
        output["sandbox_restarts"] = int(result["restarted"])
        # Beside the count, because the count alone says a replacement happened and
        # not what it was replacing. A session that restarts and recovers is the
        # common case.
        output["sandbox_restart_cause"] = str(result["restart_cause"])
        if result.get("timed_out"):
            output["exception_info"] = (
                f"The command timed out after {self.config.command_timeout_sec} seconds.")
        self.recorder.note_command(action["command"],
                                   {**output, "timed_out": result.get("timed_out")},
                                   self.output_limit_chars)
        self._check_sandbox(result)
        self._check_finished(output)
        return output

    def _check_sandbox(self, result: dict):
        """Raises ``SandboxFault`` once enough commands in a row never reached a shell.

        The runner flags a fault it caught itself, as opposed to a command that ran and
        failed, so the count is over what the sandbox did rather than over how the
        model's commands fared.

        Ending here drops the session, so the limit is how a caller says it would
        rather keep a shorter trajectory than one whose later turns report a dead
        container. Leaving it at zero is how a caller says the opposite: faults go
        through to the model as failed commands and the trajectory runs to one of its
        own endings.

        The message is one structured line because the driver writes it as the drop's
        reason, in a single field of the trace's ``END.json``. It names what this layer
        counted; the session's turns and commands are already on the final event the
        flush emits."""
        limit = self.config.max_consecutive_sandbox_faults
        if not result.get("sandbox_error"):
            self.consecutive_sandbox_faults = 0
            return
        self.consecutive_sandbox_faults += 1
        if limit and self.consecutive_sandbox_faults >= limit:
            raise SandboxFault(
                f"{SANDBOX_GONE} session={self.recorder.session_id!r} "
                f"consecutive_faults={self.consecutive_sandbox_faults} "
                f"restarts={int(result['restarted'])} "
                f"last_error={one_line(result.get('output') or '')}")

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion.

        Upstream's rule, verbatim in substance: the sentinel is recognised in what the
        command *printed*, not in the text of the command. That is why the sentinel
        runs like any other command -- a trajectory ends because its shell said so."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == SUBMIT_SENTINEL and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted({"role": "exit", "content": submission,
                             "extra": {"exit_status": "Submitted",
                                       "submission": submission}})


def one_line(text: str) -> str:
    """``text`` with every run of whitespace collapsed to a single space, truncated.

    A fault message is recorded as one line, and what gets quoted into it here is a
    command's output, which is many."""
    return " ".join(str(text).split())[:ERROR_CHARS]


class SandboxFault(Exception):
    """Ends a trajectory whose container has stopped executing commands. The driver
    drops the session and records the drop; the run is scored on the turns it served.

    Not upstream's: upstream has no sandbox to lose. Raised by the environment
    adapter, which is the only place that sees the runner's own fault flag, and named
    so ``EXIT_STATUS_TO_STOP_REASON`` maps it to an ending that points at the
    container rather than at the parser.

    Deliberately not an ``InterruptAgentFlow``. Upstream's interrupts are trajectories
    that ended the way they were meant to, and the agent swallows one and returns a
    summary the run is then scored on. This ending is not that: a session cut short by
    a dead container served fewer requests than the drive asked for, which biases the
    scored population whatever the truncation's proximate cause. So it propagates, and
    ``framework.py`` renames it to the harness's own ``SandboxGone``, which is the
    same fault the tool runner raises when it cannot replace a container. A caller
    that wants faults absorbed says so by leaving ``max_consecutive_sandbox_faults``
    at zero, which never ends a trajectory here at all.

    Propagating still leaves a full record: upstream stamps the exception's class name
    as the exit status before re-raising, so the mapping still names the ending and
    ``MiniSweLoop.run`` still flushes the held turn with the session's counts on it."""


class ResumePointAgent(DefaultAgent):
    """The forked agent, offering its state at every point a prefix could be cut.

    A subclass rather than a sixth divergence in the fork, because nothing in a graded
    run uses it: only the offline pre-roll, which has to freeze a conversation partway
    and needs the same state a resumed trajectory will be handed back.

    The hook runs *before* a turn is sent, which is the only moment the state is the
    resume state. Upstream's own loop does its bookkeeping between steps -- it resets
    the repair counter after a clean one and increments it after a failed one -- so a
    hook at the end of a step would read a counter that is about to change, and freeze
    a trajectory with a repair budget no continuous one ever had."""

    #: ``async (agent) -> None``, called before each turn reaches the server.
    before_turn: Callable | None = None
    #: Whether the hook is what ended this pass. Read by ``MiniSweLoop.run`` to name
    #: the ending, since upstream stamps any exception's class name as the exit status
    #: and the loop would otherwise file the caller's own bound as a fault.
    stopped_by_caller: bool = False

    async def step(self) -> list[dict]:
        if self.before_turn is not None:
            try:
                await self.before_turn(self)
            except Exception:
                # Raised by the caller's hook, so it is the caller stopping the pass
                # rather than the trajectory failing. Flagged and re-raised unchanged:
                # the pre-roll tells a deadline from a spend from a band cut by
                # catching its own exception types, so converting this would take that
                # away.
                self.stopped_by_caller = True
                raise
        return await super().step()


@dataclass
class MiniSweLoop:
    """One trajectory of the forked agent, wired to the harness.

    Every field is a knob the traffic's shape depends on, so
    ``AgentFramework.traffic_config`` reports them all and a change to any of them
    marks a stored calibration stale."""

    #: Characters of command output returned to the model per turn.
    output_limit_chars: int = DEFAULT_OUTPUT_LIMIT_CHARS
    #: Consecutive unparseable replies tolerated before the trajectory ends.
    max_consecutive_format_errors: int = DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS
    #: Wall-clock ceiling on one command, passed to the tool runner.
    command_timeout_sec: int = DEFAULT_COMMAND_TIMEOUT_SEC
    #: Sandbox faults in a row that end a trajectory, 0 to never end one on them.
    #: A knob the traffic's shape depends on, so it is reported like the others.
    max_consecutive_sandbox_faults: int = DEFAULT_MAX_CONSECUTIVE_SANDBOX_FAULTS
    #: Working directory this trajectory's commands run in.
    work_dir: str = field(default=WORK_DIR)
    #: Context window, in tokens, a trajectory is budgeted against. The transcript
    #: grows every turn, so a long trajectory eventually asks for more context than the
    #: server has; the server would reject that request outright rather than answer it
    #: short, which ends the trajectory on an error instead of an outcome. Declared by
    #: the task, never read off the server. None budgets against no window.
    max_context_len: int | None = None
    #: ``async (messages) -> int | None``: how many tokens the prompt those messages
    #: render to, counted harness-side. Injected by ``framework.py``, which has the
    #: checkpoint's tokenizer; None in a process that has no checkpoint, and then the
    #: reply's own figure is bounded instead.
    count_prompt_tokens: Callable | None = None
    #: The per-request output cap the harness stamps on every turn, taken from the
    #: tenant's workload. A prompt that fits the window but leaves no room to answer is
    #: as fatal as one that does not fit, so the guard reserves it. None reserves
    #: nothing, which is what a loop driven with no declared cap has to mean.
    max_output_tokens: int | None = None
    #: Tokens a cut reply's continuation may spend, or None to never continue one.
    #:
    #: Its own budget because it bounds a different thing: ``max_output_tokens`` caps
    #: the thinking, and this caps the visible reply that follows once the thought has
    #: been closed for the model. The visible half is far shorter -- it is a tool call
    #: and a sentence -- so a budget sized for thinking would let a continuation that
    #: went wrong run as long as the runaway it was meant to end.
    reply_budget: int | None = None
    #: Whether the rendered prompt ends inside an open thought, which is what the
    #: checkpoint's chat template does when thinking is on. It decides how a reply
    #: carrying no delimiter at all is read: a thought the reply never closed, or an
    #: ordinary reply with no thought in it. The two are the same string, so nothing in
    #: the reply can tell them apart; ``framework.py`` renders the prompt and measures
    #: it. False is what a loop driven with no render behind it has to mean, and it is
    #: the reading that leaves an already-stripped reply alone.
    thinking_opened: bool = False

    def agent(self, *, session_id: str, base_metadata: dict, complete, run_tool,
              recorder: TurnRecorder, max_turns: int,
              before_turn: Callable | None = None) -> DefaultAgent:
        """The forked agent, with both adapters and this trajectory's limits."""
        model = HarnessModel(session_id=session_id, base_metadata=base_metadata,
                             complete=complete, recorder=recorder,
                             count_prompt_tokens=self.count_prompt_tokens,
                             max_output_tokens=self.max_output_tokens,
                             reply_budget=self.reply_budget,
                             thinking_opened=self.thinking_opened,
                             output_limit_chars=self.output_limit_chars)
        env = HarnessEnvironment(
            run_tool=run_tool, recorder=recorder,
            output_limit_chars=self.output_limit_chars,
            command_timeout_sec=self.command_timeout_sec,
            max_consecutive_sandbox_faults=self.max_consecutive_sandbox_faults,
            work_dir=self.work_dir)
        cls = DefaultAgent if before_turn is None else ResumePointAgent
        agent = cls(
            model, env, config_class=AgentConfig,
            system_template=SYSTEM_TEMPLATE, instance_template=INSTANCE_TEMPLATE,
            step_limit=max_turns,
            max_consecutive_format_errors=self.max_consecutive_format_errors,
            max_context_len=self.max_context_len,
            max_output_tokens=self.max_output_tokens,
            reply_budget=self.reply_budget)
        if before_turn is not None:
            agent.before_turn = before_turn
        return agent

    async def run(self, *, session_id: str, problem_statement: str,
                  base_metadata: dict, complete, run_tool, emit,
                  max_turns: int, prefix=None,
                  before_turn: Callable | None = None) -> dict:
        """Drive one trajectory to termination, emitting one event per LLM call and
        returning the trajectory's own outcome.

        ``complete`` sends a request and returns the reply; ``run_tool`` executes a
        bash command in the instance's sandbox; ``emit`` records a finished turn.
        Each is supplied by the caller, so this runs identically under the harness
        driver and under the build sandbox's self-test.

        ``run_tool``'s result must carry two fields. ``restarted`` is the containers
        this session has had replaced under it, cumulative and never decreasing.
        ``restart_cause`` is the exception that provoked a replacement during *this*
        command, or "" when this command provoked none -- per turn, where the tally is
        per session. Both are part of every result, not only the ones where a restart
        happened, so a caller with none to report returns 0 and "" explicitly. Read
        directly rather than defaulted here, so a caller that omits one fails loudly;
        a default would let a writer that forgot record a believable zero instead.

        ``emit`` arrives one turn late, and a caller has to know it. A turn is not
        finished when its reply comes back -- its command has still to run, and
        ``TurnRecorder.note_command`` amends the held turn afterwards -- so each turn
        is held and emitted while the *next* one is being opened, and the last is
        flushed at the end. Everything about the turn is on the dict, so reading it is
        always right. Reading anything else at that moment is not: whatever a caller
        keeps outside has already moved on to the following turn. Key per-turn state
        on ``turn_index``, which is on the event for exactly this reason.

        ``before_turn`` is the offline pre-roll's hook, called with the agent before
        each turn is sent so a prefix can be cut there; see ``ResumePointAgent``.

        ``prefix`` is a frozen conversation to resume from (``prefix.Prefix``), or
        None to start at turn 0. A resumed trajectory counts its own turns from zero,
        because ``max_turns`` is what this session serves rather than where it began;
        the counters that decide when it *stops* carry over instead, so it does not
        collect a fresh repair budget that a continuous trajectory would not have.
        The sandbox it resumes into has to hold what the pre-roll left there, which
        is not this function's doing.

        The summary is returned rather than emitted, for the reason ``TurnRecorder``
        gives."""
        recorder = TurnRecorder(
            emit=emit, session_id=session_id,
            commands_run=prefix.commands_run if prefix else 0,
            prefix_turns=prefix.turns if prefix else 0,
            inherited_no_command=prefix.consecutive_format_errors if prefix else 0)
        agent = self.agent(session_id=session_id, base_metadata=base_metadata,
                           complete=complete, run_tool=run_tool, recorder=recorder,
                           max_turns=max_turns, before_turn=before_turn)
        try:
            await agent.run(task=problem_statement, work_dir=self.work_dir,
                            submit_sentinel=SUBMIT_SENTINEL, prefix=prefix)
        finally:
            # Read off the agent's own last message rather than what ``run`` returned,
            # so the verdict is the same one whether the trajectory ended or raised:
            # upstream stamps an exit message either way. The held turn is emitted in
            # both cases too, because a turn the artifact was served and spent time on
            # is part of the measurement even when the run did not finish.
            last = (agent.messages[-1].get("extra") or {}) if agent.messages else {}
            # Upstream has one name for the budget running out. Which of the two
            # failures spent it is the recorder's to say, since it saw every turn.
            named = EXIT_STATUS_TO_STOP_REASON.get(
                str(last.get("exit_status", "")), STOP_ERROR)
            if named is STOP_ERROR and getattr(agent, "stopped_by_caller", False):
                named = STOP_CALLER_STOPPED
            stop_reason, causes = recorder.repair_verdict(named)
            await recorder.flush(stop_reason, causes)
        return {"session_id": session_id, "stop_reason": stop_reason,
                "no_command_causes": causes,
                "commands_run": recorder.commands_run, "turns": recorder.turns,
                "prefix_turns": recorder.prefix_turns}


__all__ = [
    "BASH_TOOL", "BASH_TOOL_NAME", "DEFAULT_COMMAND_TIMEOUT_SEC",
    "DEFAULT_MAX_CONSECUTIVE_SANDBOX_FAULTS", "ERROR_CHARS", "SANDBOX_GONE",
    "STOP_SANDBOX_FAULT", "SandboxFault", "one_line",
    "DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS", "DEFAULT_OUTPUT_LIMIT_CHARS",
    "EXIT_STATUS_TO_STOP_REASON", "FORMAT_ERROR_TRUNCATED", "FORMAT_ERROR_UNREADABLE",
    "HarnessEnvironment", "HarnessModel", "INSTANCE_TEMPLATE", "MiniSweLoop",
    "CONTINUATION_UNRENDERABLE_EXTRA", "CONTINUED_EXTRA",
    "PREFILL_TOKENS_EXTRA",
    "THINK_CAP_EXTRA",
    "NO_COMMAND_NO_CALL", "NO_COMMAND_RUNAWAY", "NO_COMMAND_UNPARSED",
    "STOP_CALLER_STOPPED", "STOP_CONTEXT_EXHAUSTED", "STOP_EMPTY_REPLY",
    "STOP_ERROR",
    "STOP_FORMAT_REPAIR_EXHAUSTED", "STOP_NO_COMMAND_REPEATED",
    "STOP_STEP_LIMIT", "STOP_SUBMITTED", "STOP_WALL_CLOCK", "SUBMIT_SENTINEL",
    "SYSTEM_TEMPLATE",
    "TEMPLATE_BYTES_PER_MESSAGE", "THINK_CLOSE", "THINK_OPEN", "TurnRecorder",
    "WORK_DIR", "has_toolcall_delimiter",
    "no_command_cause",
    "parse_toolcall_text", "prompt_byte_bound", "reasoning_text", "render_observation",
    "split_reply", "visible_text", "wire_messages",
]
