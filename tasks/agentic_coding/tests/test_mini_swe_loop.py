"""The forked loop: how a call is read, and its five ways of stopping."""
from __future__ import annotations

import time

import pytest

from tasks.agentic_coding.minisweagent.config.mini import (
    FORMAT_ERROR_TRUNCATED, FORMAT_ERROR_UNREADABLE,
)
from tasks.agentic_coding.minisweagent.exceptions import (
    FormatError, InterruptAgentFlow,
)
from tasks.agentic_coding.minisweagent.models.utils.actions_toolcall import (
    parse_toolcall_actions, parse_toolcall_text, strip_toolcalls,
)
from tasks.agentic_coding.mini_swe_loop import (
    THINK_CAP_EXTRA, THINK_CLOSE,
    INSTANCE_TEMPLATE, PREFILL_TOKENS_EXTRA, STOP_CONTEXT_EXHAUSTED, STOP_EMPTY_REPLY,
    NO_COMMAND_NO_CALL, NO_COMMAND_RUNAWAY, NO_COMMAND_UNPARSED,
    SANDBOX_GONE, STOP_CALLER_STOPPED, SandboxFault,
    STOP_FORMAT_REPAIR_EXHAUSTED, STOP_ERROR, STOP_NO_COMMAND_REPEATED,
    STOP_SANDBOX_FAULT, EXIT_STATUS_TO_STOP_REASON,
    STOP_STEP_LIMIT, STOP_SUBMITTED,
    SUBMIT_SENTINEL, SYSTEM_TEMPLATE, MiniSweLoop, TurnRecorder, prompt_byte_bound,
    reasoning_text, render_observation, visible_text, wire_messages,
)


class Recorder:
    """Stands in for the harness: replies from a script, records tool calls, collects
    emitted turns.

    ``reported_prompt_tokens`` is what the server claims its prompt was. Only the
    loop's no-counter fallback reads it, so it is None unless a test is exercising
    that path. ``finish_reason`` and ``served_tokens`` are what the framework lifts
    off the reply's payload, and only a turn with nothing visible in it reads them.

    ``run_tool`` echoes: ``echo X`` prints ``X``. That is what makes the submit
    sentinel work the way it does on a real shell, since the environment recognises a
    submission in what a command *printed* rather than in the command itself."""

    def __init__(self, replies: list[str], tool_result: dict | None = None,
                 reported_prompt_tokens: int | None = None,
                 finish_reason: str | None = None,
                 served_tokens: int | None = None) -> None:
        self.replies = list(replies)
        self.tool_result = tool_result
        self.reported_prompt_tokens = reported_prompt_tokens
        self.finish_reason = finish_reason
        self.served_tokens = served_tokens
        self.events: list[dict] = []
        self.commands: list[str] = []
        self.requests: list[dict] = []
        self.tool_args: list[dict] = []

    async def complete(self, request: dict) -> dict:
        self.requests.append(request)
        text = self.replies.pop(0) if self.replies else "no call here"
        metadata = ({} if self.reported_prompt_tokens is None
                    else {"prompt_tokens": self.reported_prompt_tokens})
        return {"text": text, "metadata": metadata,
                "finish_reason": self.finish_reason,
                "served_tokens": self.served_tokens}

    async def run_tool(self, name: str, args: dict) -> dict:
        """One command, shaped the way the real runner shapes its results.

        ``restarted`` is stamped once here rather than written into each return, the
        same way ``SandboxToolRunner.run`` stamps it, because the seam requires it on
        every result and a double that met the contract only sometimes would let the
        loop's reader pass these tests and fail on a real runner."""
        assert name == "bash"
        self.commands.append(args["command"])
        self.tool_args.append(args)
        return {"restarted": 0, "restart_cause": "",
                **self._result(args["command"])}

    def _result(self, command: str) -> dict:
        # Echo first, so a pinned ``tool_result`` can govern the ordinary commands of
        # a trajectory without also swallowing the sentinel the submit turn prints.
        if command.startswith("echo "):
            return {"output": command[len("echo "):].strip("'\"") + "\n",
                    "exit_code": 0}
        if self.tool_result is not None:
            return dict(self.tool_result)
        return {"output": "ok", "exit_code": 0}

    async def emit(self, event: dict) -> None:
        self.events.append(event)


def call(command: str, thought: str = "Looking around.", prose: str = "") -> str:
    """One reply as this checkpoint emits it: the reasoning it was already inside when
    generation began, then any prose, then the call in the template's own syntax."""
    return (f"{thought}</think>{prose}<tool_call>bash<arg_key>command</arg_key>"
            f"<arg_value>{command}</arg_value></tool_call>")


SUBMIT = call(f"echo {SUBMIT_SENTINEL}")


async def drive(replies: list[str], *, max_turns: int = 10,
                tool_result: dict | None = None,
                reported_prompt_tokens: int | None = None,
                finish_reason: str | None = None, served_tokens: int | None = None,
                prefix=None, **kwargs) -> tuple[Recorder, dict]:
    """One trajectory against a scripted server. ``kwargs`` configure the loop;
    ``prefix`` is a frozen conversation to resume from, which belongs to the run."""
    rec = Recorder(replies, tool_result, reported_prompt_tokens, finish_reason,
                   served_tokens)
    loop = MiniSweLoop(**kwargs)
    summary = await loop.run(session_id="s0", problem_statement="Fix the bug.",
                             base_metadata={"tenant_id": "t0"}, complete=rec.complete,
                             run_tool=rec.run_tool, emit=rec.emit, max_turns=max_turns,
                             prefix=prefix)
    return rec, summary


async def dead_sandbox(replies, **kwargs) -> tuple[Recorder, Exception]:
    """One trajectory whose every command comes back as a sandbox fault, run to
    whatever ending it reaches. Returns the recorder and the exception it raised.

    Its own helper because ``drive`` returns the loop's summary, and a trajectory that
    ends on a dead sandbox raises instead of returning one.

    The condition is fabricated, and it has to be. ``sandbox_error`` has one producer
    in the tree -- ``hls.spec.sandbox`` -- which sets it only when a session's restart
    budget is zero; with a budget the runner raises instead of returning the flag. So
    ``task.py``, whose budget is five, cannot reach the endings these tests cover. They
    describe the restarts-off configuration, which is real and supported and is not the
    one this task ships. Kept because that is the configuration the knob exists for and
    the one a calibration taken here would be reused under."""
    rec = Recorder(replies,
                   tool_result={"output": "Sandbox error: ReadError: ", "exit_code": 125,
                                "sandbox_error": True})
    loop = MiniSweLoop(**kwargs)
    with pytest.raises(SandboxFault) as caught:
        await loop.run(session_id="s0", problem_statement="Fix the bug.",
                       base_metadata={"tenant_id": "t0"}, complete=rec.complete,
                       run_tool=rec.run_tool, emit=rec.emit, max_turns=12)
    return rec, caught.value


async def test_a_run_of_sandbox_faults_ends_the_trajectory():
    # The runner reports a broken sandbox to the model as a failed command so that a
    # one-off blip does not end a trajectory. It also lets a dead exec server serve
    # failed commands turn after turn. So a run of them ends the session.
    rec, _ = await dead_sandbox([call("ls")] * 12, max_consecutive_sandbox_faults=3)
    assert rec.events[-1]["extras"]["stop_reason"] == STOP_SANDBOX_FAULT
    # Ended on the run rather than at the turn cap.
    assert rec.events[-1]["extras"]["session_turns"] == 3


async def test_a_trajectory_ended_by_the_sandbox_does_not_get_to_be_scored():
    # One outcome, one meaning. A session cut short by a dead container served fewer
    # requests than the drive asked for, and that biases the scored population however
    # the truncation came about -- so the ending propagates rather than returning a
    # summary the run would then be scored on. A caller that wants these absorbed
    # leaves the limit at zero and takes them through to the model as failed commands,
    # which is a trajectory that was never cut short.
    _, exc = await dead_sandbox([call("ls")] * 12, max_consecutive_sandbox_faults=3)
    assert isinstance(exc, SandboxFault)


def test_the_sandbox_ending_is_not_one_of_upstreams_clean_ones():
    # Upstream's InterruptAgentFlow means a trajectory that ended the way it was meant
    # to, and `DefaultAgent.run` swallows it and returns. This ending is not that, and
    # inheriting from it would put the run back on the scored path silently.
    assert not issubclass(SandboxFault, InterruptAgentFlow)


async def test_the_sandbox_ending_says_what_it_knows_on_one_line():
    # The harness fault this becomes is recorded as `repr()` of itself in a single
    # field of the run record, so the message has to be one line a person can read and
    # a script can split. Counts belong to the loop and are already on the final event
    # above; what is here is what names the session and the fault run.
    _, exc = await dead_sandbox([call("ls")] * 12, max_consecutive_sandbox_faults=3)
    message = str(exc)
    assert "\n" not in message
    assert message.startswith(SANDBOX_GONE)
    fields = dict(part.split("=", 1) for part in message.split(" ") if "=" in part)
    assert fields["session"] == "'s0'"
    assert fields["consecutive_faults"] == "3"
    assert fields["restarts"] == "0"


async def test_one_sandbox_fault_does_not_end_a_trajectory():
    # The blip case: a single fault, then clean turns.
    rec, summary = await drive([call("ls"), call("pwd"), SUBMIT],
                               max_consecutive_sandbox_faults=3)
    assert summary["stop_reason"] == STOP_SUBMITTED


async def test_the_sandbox_fault_run_resets_on_a_command_that_worked():
    # Consecutive, not cumulative. A session that fails once every twenty turns is
    # limping, not dead, and ending it would discard a usable trajectory.
    #
    # Restarts off, which is fabricated here and unreachable under the shipped task;
    # see `dead_sandbox` above for why.
    calls = 0

    async def flaky(name, args):
        nonlocal calls
        calls += 1
        if calls % 3 == 0:
            return {"output": "Sandbox error: ReadError: ", "exit_code": 125,
                    "sandbox_error": True, "restarted": 0,
                    "restart_cause": ""}
        return {"output": "ok", "exit_code": 0, "restarted": 0,
                "restart_cause": ""}

    rec = Recorder([call("ls")] * 9 + [SUBMIT])
    rec.run_tool = flaky
    loop = MiniSweLoop(max_consecutive_sandbox_faults=3)
    summary = await loop.run(session_id="s0", problem_statement="p",
                             base_metadata={}, complete=rec.complete,
                             run_tool=rec.run_tool, emit=rec.emit, max_turns=12)
    assert summary["stop_reason"] != STOP_SANDBOX_FAULT


async def test_a_sandbox_fault_is_not_reported_as_a_missing_command():
    # The ending has to name the sandbox. A dead sandbox returns a failed command,
    # which the loop would otherwise spend its repair budget on and then report as the
    # model failing to produce calls -- a stop reason lying about its cause.
    rec, _ = await dead_sandbox([call("ls")] * 12, max_consecutive_sandbox_faults=3)
    assert rec.events[-1]["extras"]["stop_reason"] not in (
        STOP_NO_COMMAND_REPEATED, STOP_FORMAT_REPAIR_EXHAUSTED, STOP_ERROR)


async def test_the_fault_run_is_off_by_default():
    # Off unless a caller asks for it, so no shipped path changes behaviour without
    # someone choosing it. This is also how a caller says it wants faults absorbed:
    # the trajectory is never cut short, so it stays a trajectory the run can be
    # scored on, and every fault reaches the model as a failed command. This is the
    # only configuration in which a fault reaches the loop at all; see `dead_sandbox`.
    _, summary = await drive(
        [call("ls")] * 6 + [SUBMIT],
        tool_result={"output": "Sandbox error: ReadError: ", "exit_code": 125,
                     "sandbox_error": True})
    assert summary["stop_reason"] != STOP_SANDBOX_FAULT


async def test_a_replaced_container_is_recorded_on_the_turn():
    # A session whose sandbox was replaced under it produced its traffic under
    # conditions nobody intended, so the count has to reach the drive rather than
    # stop at the runner. Without it, a heavily-restarted session averages into the
    # score indistinguishably from a clean one.
    rec, _ = await drive([call("ls"), SUBMIT],
                         tool_result={"output": "ok", "exit_code": 0, "restarted": 2})
    assert rec.events[0]["extras"]["sandbox_restarts"] == 2


async def test_the_turn_says_what_its_container_was_replaced_for():
    # The count says a replacement happened; this says what it was replacing. Without
    # it a session that restarts and recovers, which is the common case, records that
    # something went wrong and nothing about what. Restarts make a fault survivable
    # and so unobservable.
    rec, _ = await drive(
        [call("ls"), SUBMIT],
        tool_result={"output": "ok", "exit_code": 0, "restarted": 1,
                     "restart_cause": "ReadError: "})
    assert rec.events[0]["extras"]["sandbox_restart_cause"] == "ReadError: "


async def test_a_turn_that_replaced_nothing_reports_no_cause():
    # Present on every turn like the count, and empty rather than absent, so a reader
    # never tells a missing key from a turn that simply did not restart.
    rec, _ = await drive([call("ls"), SUBMIT])
    assert rec.events[0]["extras"]["sandbox_restart_cause"] == ""


async def test_a_session_that_never_lost_its_sandbox_says_zero():
    # Present on every turn rather than only on the ones that restarted: a reader
    # asking "was this session's sandbox replaced" must not have to tell a missing
    # key from a zero.
    rec, _ = await drive([call("ls"), SUBMIT])
    assert rec.events[0]["extras"]["sandbox_restarts"] == 0


def test_a_lost_sandbox_is_named_as_one_rather_than_as_an_error():
    # The loop maps an ending by the exit_status upstream stamps, which for an
    # uncaught exception is its class name. A bare HarnessFault would therefore land
    # as STOP_ERROR and the record would never name the sandbox. The runner raises a
    # named subclass so this mapping can exist.
    #
    # Matched as a string, like DriveWallClockReached, because nothing in this module
    # may import hls; the assertion below is what holds the two names together.
    from hls.core.faults import SandboxGone

    assert EXIT_STATUS_TO_STOP_REASON[SandboxGone.__name__] == STOP_SANDBOX_FAULT


def test_both_producers_of_the_ending_write_the_same_line():
    # Two layers see a session's container stop executing commands -- the tool runner
    # when it cannot replace one, this loop when enough commands in a row never reach
    # a shell -- and both end as the same harness fault. A reader grepping the run
    # records for one of them should not have to know which layer noticed.
    #
    # Spelled out in both places rather than imported, for the reason above: nothing
    # in this module may import hls. This assertion is what holds them together.
    from hls.core.faults import SANDBOX_GONE as harness_side

    assert SANDBOX_GONE == harness_side


async def test_a_lost_sandbox_still_delivers_the_turn_it_died_on():
    # The record has to survive a propagating fault. Upstream stamps an exit message
    # before re-raising, so the loop's own flush still emits the held turn and the
    # session's turn count reaches the trace even though the session is dropped and no
    # later turn will carry it.
    #
    # The restart count does not. The held turn is the one that died, and a turn whose
    # command never returned never reached `note_command`, which is what writes
    # `sandbox_restarts`. So the one event triage reads for a trajectory killed by its
    # sandbox carries no count of how many times that sandbox was replaced.
    class SandboxGone(Exception):
        """Named to match the real one; see the test above."""

    rec = Recorder([call("ls"), call("pwd")])
    # Counted here rather than off the recorder: replacing run_tool means the
    # recorder's own command list is never appended to, so a condition reading it
    # would never fire and the test would pass by not raising at all.
    ran = []

    async def dies(name, args):
        ran.append(args["command"])
        if len(ran) > 1:
            raise SandboxGone("sandbox s0 gone after 2 restarts")
        # Two, matching what this session's own fault message says it spent. The
        # trajectory's restarts are half of what the flush is being asserted to
        # carry, so a stub reporting none would leave that half untested.
        return {"output": "ok", "exit_code": 0, "restarted": 2,
                "restart_cause": "RuntimeError: ReadError: "}

    rec.run_tool = dies
    loop = MiniSweLoop()
    with pytest.raises(SandboxGone):
        await loop.run(session_id="s0", problem_statement="p", base_metadata={},
                       complete=rec.complete, run_tool=rec.run_tool, emit=rec.emit,
                       max_turns=10)
    last = rec.events[-1]["extras"]
    assert last["stop_reason"] == STOP_SANDBOX_FAULT
    assert last["session_turns"] >= 1
    # The gap, pinned so it is not rediscovered as a surprise: this event has no
    # restart count, though the session had spent two by the time it died. The turn
    # before it does carry one, so the information exists in the trace and is missing
    # only from the event a reader goes to first.
    assert "sandbox_restarts" not in last
    assert rec.events[-2]["extras"]["sandbox_restarts"] == 2


def commands_of(text: str) -> list[str]:
    """The commands a reply asks for, through the whole parse the loop uses."""
    actions = parse_toolcall_actions(parse_toolcall_text(visible_text(text)),
                                     format_error_template=FORMAT_ERROR_UNREADABLE,
                                     template_kwargs={"submit_sentinel": "S"})
    return [a["command"] for a in actions]


# --- a reply cut at the thinking cap ---------------------------------------


class Capped(Recorder):
    """Replies whose served lengths are scripted per call, so one turn can hit the cap
    and its continuation not.

    The base Recorder carries one ``served_tokens`` for every reply, which cannot
    express a two-phase turn: phase 1 spends its whole budget and phase 2 does not.
    """

    def __init__(self, replies, served, **kw):
        super().__init__(list(replies), **kw)
        self.served = list(served)

    async def complete(self, request):
        out = await super().complete(request)
        out["served_tokens"] = self.served.pop(0) if self.served else 0
        return out


async def drive_capped(replies, served, *, think_cap, reply_budget, max_turns=6):
    rec = Capped(replies, served)
    loop = MiniSweLoop(max_output_tokens=think_cap, reply_budget=reply_budget)
    summary = await loop.run(session_id="s0", problem_statement="p", base_metadata={},
                             complete=rec.complete, run_tool=rec.run_tool,
                             emit=rec.emit, max_turns=max_turns)
    return rec, summary


async def test_a_reply_that_closes_its_thought_is_one_request():
    # The ordinary turn, and the control for everything below: nothing is continued
    # unless the cap actually fired, so a trajectory of well-behaved turns sends
    # exactly as many requests as it has turns.
    rec, _ = await drive_capped([call("ls"), SUBMIT], [10, 10],
                                think_cap=4096, reply_budget=1024)
    assert len(rec.requests) == 2
    assert all("continue_final_message" not in r["metadata"] for r in rec.requests)


async def test_a_thought_cut_at_the_cap_is_continued_from_the_closing_tag():
    # The runaway shape: the budget ran out inside the thought, so nothing visible
    # ever arrived. The tag is appended and the model is asked to carry on from it,
    # which is how the turn produces a command instead of being thrown away.
    rec, _ = await drive_capped(["thinking and thinking", call("ls"), SUBMIT],
                                [4096, 10, 10], think_cap=4096, reply_budget=1024)

    first, second = rec.requests[0], rec.requests[1]
    assert "continue_final_message" not in first["metadata"]
    assert second["metadata"]["continue_final_message"] is True
    assert second["metadata"]["max_tokens"] == 1024

    carried = second["metadata"]["messages"][-1]
    assert carried["role"] == "assistant"
    assert carried["content"] == "thinking and thinking" + THINK_CLOSE


async def test_a_closed_thought_cut_mid_reply_is_continued_without_a_second_tag():
    # The other way a cap fires: the thought closed and the visible half was cut, so
    # there is nothing to close. Appending a tag here would put a second one in the
    # reply and the split would read the command as reasoning.
    rec, _ = await drive_capped(["done" + THINK_CLOSE + "cat <<EO",
                                 call("ls"), SUBMIT],
                                [4096, 10, 10], think_cap=4096, reply_budget=1024)

    carried = rec.requests[1]["metadata"]["messages"][-1]
    assert carried["content"] == "done" + THINK_CLOSE + "cat <<EO"
    assert carried["content"].count(THINK_CLOSE) == 1


async def test_the_continuation_is_not_continued_again():
    # Phase 2 is the last phase. A continuation that also hits its budget ends the
    # turn as a truncated reply rather than starting a third request, or a model that
    # will not stop would be served indefinitely at one turn.
    rec, _ = await drive_capped(["thinking", "still going", call("ls"), SUBMIT],
                                [4096, 1024, 10, 10],
                                think_cap=4096, reply_budget=1024)
    continued = [r for r in rec.requests
                 if r["metadata"].get("continue_final_message")]
    assert len(continued) == 1


async def test_a_cut_thought_the_template_cannot_render_ends_the_turn_at_the_cap():
    # The checkpoint's template keeps only the text after the last opening tag inside a
    # thought, so a cut thought that itself contains a literal <think> cannot be handed
    # back verbatim; transformers refuses the render with a ValueError from the prompt
    # count. Uncaught, that error raises out of the driver and loses the whole drive.
    # The loop must instead end that turn as a reply that ran to its cap,
    # send no continuation, mark the event, and let the trajectory go on.
    from tasks.agentic_coding.mini_swe_loop import (
        CONTINUATION_UNRENDERABLE_EXTRA, CONTINUED_EXTRA, THINK_OPEN)

    async def count(messages, continue_final_message=False):
        if continue_final_message and THINK_OPEN in messages[-1]["content"]:
            raise ValueError("continue_final_message is set but the final message does "
                             "not appear in the chat after applying the chat template!")
        return 7

    rec = Capped(["planning <think> and planning", call("ls"), SUBMIT], [4096, 10, 10])
    loop = MiniSweLoop(max_output_tokens=4096, reply_budget=1024, count_prompt_tokens=count)
    summary = await loop.run(session_id="s0", problem_statement="p", base_metadata={},
                             complete=rec.complete, run_tool=rec.run_tool,
                             emit=rec.emit, max_turns=6)

    assert all("continue_final_message" not in r["metadata"] for r in rec.requests)
    cut = [e for e in rec.events if e["extras"].get(THINK_CAP_EXTRA)]
    assert len(cut) == 1
    assert cut[0]["extras"][CONTINUATION_UNRENDERABLE_EXTRA] is True
    assert cut[0]["extras"][CONTINUED_EXTRA] is False
    assert cut[0]["extras"]["ran_to_output_cap"] is True
    # The trajectory went on to act and finish rather than dying with the drive.
    assert len(rec.requests) == 3
    assert summary["stop_reason"] == STOP_SUBMITTED


async def test_both_phases_are_their_own_request_on_the_trace():
    # Both are served requests, so both are scored against their own prompt sizes and
    # both are graded. One event carrying the pair would price the whole turn against
    # phase 1's prompt and hide phase 2's serving entirely.
    rec, _ = await drive_capped(["thinking", call("ls"), SUBMIT],
                                [4096, 10, 10], think_cap=4096, reply_budget=1024)

    cut = [e for e in rec.events if e["extras"].get(THINK_CAP_EXTRA)]
    assert len(cut) == 1, "the cut phase is its own event"
    assert cut[0]["extras"]["ran_to_output_cap"] is True
    # And it is not counted as a turn that failed to produce a command: it was cut on
    # purpose and its continuation is what answers.
    assert cut[0]["extras"].get("continued") is True


# --- parsing ---------------------------------------------------------------

def test_a_well_formed_call_yields_its_command():
    assert commands_of(call("ls -la")) == ["ls -la"]


def test_a_call_with_no_prose_around_it_still_parses():
    assert commands_of(call("pwd", prose="")) == ["pwd"]


def test_a_reply_with_no_call_raises_format_error():
    with pytest.raises(FormatError) as excinfo:
        commands_of("I think I should look at the file.")
    assert "MUST include at least one tool call" in excinfo.value.messages[0]["content"]


def test_an_unknown_tool_raises_format_error():
    # The model invoked something it was never offered. Upstream refuses rather than
    # guessing that it meant bash.
    text = ("<tool_call>python<arg_key>command</arg_key>"
            "<arg_value>print(1)</arg_value></tool_call>")
    with pytest.raises(FormatError) as excinfo:
        commands_of(text)
    assert "Unknown tool 'python'" in excinfo.value.messages[0]["content"]


def test_a_call_missing_its_command_argument_raises_format_error():
    text = "<tool_call>bash<arg_key>cmd</arg_key><arg_value>ls</arg_value></tool_call>"
    with pytest.raises(FormatError) as excinfo:
        commands_of(text)
    assert "Missing 'command' argument" in excinfo.value.messages[0]["content"]


def test_a_call_with_no_arguments_at_all_raises_format_error():
    with pytest.raises(FormatError):
        commands_of("<tool_call>bash</tool_call>")


def test_two_calls_in_one_reply_both_parse():
    # Upstream's format permits more than one action per turn and executes them in
    # order, so the parser returns both rather than calling the reply ambiguous.
    text = f"{call('ls')}{call('pwd', thought='')}"
    assert commands_of(text) == ["ls", "pwd"]


def test_an_unterminated_call_yields_nothing():
    # The reply hit its output cap partway through writing the call. Half a command is
    # not an action.
    assert parse_toolcall_text("<tool_call>bash<arg_key>command</arg_key>") == []


def test_a_command_keeps_its_whitespace_and_quoting():
    # The argument is carried raw, because the chat template renders a string value
    # raw: what the model emitted and what the history renders back must agree.
    cmd = "grep -rn 'a  b' . | head -5"
    assert commands_of(call(cmd)) == [cmd]


def test_stripping_a_call_leaves_the_prose_around_it():
    span = "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    assert strip_toolcalls(f"Listing it. {span} Done.") == "Listing it.  Done."


#: The same reply as the two engines hand it over. SGLang returns it raw, with the
#: reasoning still in the middle; vLLM run with ``--reasoning-parser glm45`` returns
#: the visible part alone. The chat template opens the reasoning channel in the
#: prompt, so the raw form carries a closing delimiter and no opening one.
_VISIBLE = ("Listing the directory.<tool_call>bash<arg_key>command</arg_key>"
            "<arg_value>ls -la</arg_value></tool_call>")
_RAW = "The user wants one command. I should look around first.</think>" + _VISIBLE


def test_a_reply_that_only_closes_the_thought_is_stripped():
    # A chat template with thinking on emits the opening delimiter in the PROMPT, so
    # the reply starts inside the reasoning and only closes it, as GLM-5.3-Flash's
    # does. Without this the model's private reasoning reaches the parser.
    assert visible_text(_RAW).startswith("Listing the directory.")
    assert "I should look around" not in visible_text(_RAW)
    assert commands_of(_RAW) == ["ls -la"]


def test_an_already_stripped_reply_passes_through_unchanged():
    # The engine that strips reasoning server-side leaves nothing to strip here, and
    # a normalizer that took anything off it would eat the answer.
    assert visible_text(_VISIBLE) == _VISIBLE
    assert commands_of(_VISIBLE) == ["ls -la"]


def test_both_engines_parse_the_same_reply_to_the_same_command():
    # The two forms are one reply served two ways. A trajectory that read them
    # differently would depend on which stack served it.
    assert visible_text(_RAW) == visible_text(_VISIBLE)
    assert commands_of(_RAW) == commands_of(_VISIBLE) == ["ls -la"]


def test_a_call_written_inside_the_thought_is_not_an_action():
    # A call the model drafted while reasoning is not one it asked for.
    text = ("<think>maybe <tool_call>bash<arg_key>command</arg_key>"
            "<arg_value>rm -rf /</arg_value></tool_call> no</think>" + _VISIBLE)
    assert commands_of(text) == ["ls -la"]


def test_a_thought_that_only_closes_hides_a_call_inside_it():
    text = ("maybe <tool_call>bash<arg_key>command</arg_key>"
            "<arg_value>rm -rf /</arg_value></tool_call> no</think>" + _VISIBLE)
    assert commands_of(text) == ["ls -la"]


def test_an_unterminated_thought_yields_no_command():
    # The reply hit the output cap mid-thought, so there is no action in it.
    assert parse_toolcall_text(visible_text("<think>still reasoning" + _VISIBLE)) == []
    assert visible_text("visible<think>hidden") == "visible"


def test_the_reasoning_is_recovered_for_the_history():
    # The chat template renders an assistant turn's reasoning into its own <think>
    # span, so the next prompt only matches the one the artifact was served if the
    # reasoning went back into the history.
    assert reasoning_text(_RAW) == "The user wants one command. I should look around first."
    assert reasoning_text(_VISIBLE) == ""


# --- a reply that closes nothing -------------------------------------------
#
# The prompt opens the thinking channel, so a reply that spends its whole output cap
# thinking carries no delimiter at all. An already-stripped reply is the same string
# shape, which is why the two are told apart by what the prompt rendered to rather
# than by anything in the reply.

_UNCLOSED = "I should check whether the parser handles this, because if it does"


def test_a_reply_that_closes_nothing_is_all_reasoning_when_the_prompt_opened_one():
    # A reply can spend its whole cap mid-thought. Recorded as visible text it puts the
    # raw thinking where the reply belongs, and the template then renders it back as
    # ordinary content behind an empty thought on every later prompt of that session,
    # each of which the oracle teacher-forces.
    assert visible_text(_UNCLOSED, thinking_opened=True) == ""
    assert reasoning_text(_UNCLOSED, thinking_opened=True) == _UNCLOSED


def test_the_same_reply_is_visible_when_the_prompt_opened_none():
    # A gateway that returns reasoning in a field of its own serves replies of exactly
    # this shape on every turn. Reading them as reasoning would leave the loop parsing
    # nothing, every turn, for as long as it ran.
    assert visible_text(_VISIBLE, thinking_opened=False) == _VISIBLE
    assert reasoning_text(_VISIBLE, thinking_opened=False) == ""


def test_an_opened_channel_does_not_change_a_reply_that_closed_it():
    # The flag decides one case and no other: a reply that did close its thought is
    # split on the delimiter it sent, whatever the prompt did.
    assert visible_text(_RAW, thinking_opened=True) == visible_text(_RAW)
    assert reasoning_text(_RAW, thinking_opened=True) == reasoning_text(_RAW)


def test_a_thought_the_reply_opened_and_never_closed_is_kept_too():
    # The other unterminated shape, and the same fidelity argument: the model wrote
    # the thought, so the history has to carry it. Dropping it renders a prompt
    # shorter than the one the artifact was served.
    assert visible_text("visible<think>hidden") == "visible"
    assert reasoning_text("visible<think>hidden") == "hidden"


def test_a_reply_that_closes_nothing_is_all_reasoning_even_with_a_stray_opener():
    # The rule is about the CLOSING delimiter. A reply that spent its whole cap inside
    # the thought the prompt opened is all thought, and an opening delimiter it happens
    # to have written on the way does not end it: the model cannot open what is already
    # open. Splitting here would put half the thought in the visible reply.
    text = "checking whether <think> appears in the sources, because if it"
    assert visible_text(text, thinking_opened=True) == ""
    assert reasoning_text(text, thinking_opened=True) == text


def test_text_after_the_thought_is_kept_verbatim_delimiters_and_all():
    # thought, text, thought. Rendered through the real template this round-trips
    # byte for byte, because the template leaves content alone when the reasoning
    # field is set -- it only splits content on a closing delimiter when that field is
    # absent. So an opening delimiter inside the reply is literal text to the serving
    # stack, and dropping it would replay a shorter sequence than the one served.
    #
    # Reachable without the model ever meaning a second thought: this is an agent
    # writing shell commands, and a command that mentions the delimiter produces it.
    text = "thought one</think>I will grep for <think> in the sources"
    assert visible_text(text, thinking_opened=True) == "I will grep for <think> in the sources"
    assert reasoning_text(text, thinking_opened=True) == "thought one"


def test_nothing_is_stripped_from_the_reply_after_its_thought():
    # Everything before the first close is the thought and never reaches the parser,
    # which is the whole of the "a call drafted while reasoning is not one it asked
    # for" rule. After it, the reply is what the model wrote.
    text = "weighing it</think>ran <think>one</think> and <think>two"
    assert visible_text(text, thinking_opened=True) == "ran <think>one</think> and <think>two"
    assert reasoning_text(text, thinking_opened=True) == "weighing it"


def test_the_stripped_path_drops_an_open_tail_after_a_leading_close():
    # The control for the stripped branch, on the exact input where the two rules
    # disagree most: a leading close, then a thought left open. With no thought open
    # in the prompt, the tail is dropped from the reply and only the lead is kept as
    # reasoning. The verbatim rule above must not leak into this path.
    text = "reasoning</think>visible<think>hidden"
    assert visible_text(text) == "visible"
    assert reasoning_text(text) == "reasoning"
    # And the two shapes that carry the contract: raw, and already stripped.
    assert visible_text(_RAW) == _VISIBLE
    assert visible_text(_VISIBLE) == _VISIBLE


# --- output elision, which is the observation template's job ---------------

def observation(output: str, returncode: int = 0, limit: int = 40) -> str:
    return render_observation(
        {"output": output, "returncode": returncode, "exception_info": ""},
        output_limit=limit)


def test_short_output_is_returned_whole():
    assert '"output": "hello"' in observation("hello")


def test_long_output_keeps_both_ends_and_says_what_went():
    rendered = observation("A" * 100 + "B" * 100)
    assert '"output_head": "' + "A" * 20 in rendered
    assert "B" * 20 + '"' in rendered
    assert '"elided_chars": 160' in rendered
    assert "Output too long." in rendered


def test_the_exit_code_rides_on_the_observation():
    assert '"returncode": 2' in observation("boom", returncode=2)


def test_the_observation_stays_parseable_json_whatever_the_command_printed():
    # Command output is arbitrary bytes and the observation looks like JSON, so it had
    # better be JSON: a quote or a newline in a build log must not turn the message
    # the model reads into something it has to guess at.
    import json
    nasty = 'he said "hi"\n\ttab \\ backslash \x01'
    assert json.loads(observation(nasty, limit=10_000))["output"] == nasty
    long = json.loads(observation("x" * 500, limit=100))
    assert long["elided_chars"] == 400


def test_every_template_renders():
    # A str.format template with a typo'd placeholder, or an unescaped literal brace,
    # raises here rather than at the first turn of a graded run.
    assert INSTANCE_TEMPLATE.format(task="t", work_dir="/app", submit_sentinel="S")
    assert FORMAT_ERROR_UNREADABLE.format(error="e", submit_sentinel="S")
    assert FORMAT_ERROR_TRUNCATED.format()
    assert SYSTEM_TEMPLATE


def test_the_rendered_templates_keep_their_literal_braces():
    # The example response and the arguments line both contain JSON, which str.format
    # would eat if the braces were not doubled.
    assert '{"command": "ls -la"}' in INSTANCE_TEMPLATE.format(
        task="t", work_dir="/app", submit_sentinel="S")
    assert '{"command": "your_command_here"}' in FORMAT_ERROR_UNREADABLE.format(
        error="e", submit_sentinel="S")


# --- termination -----------------------------------------------------------

async def test_the_sentinel_ends_the_trajectory():
    rec, summary = await drive([call("ls"), SUBMIT])
    assert summary["stop_reason"] == STOP_SUBMITTED
    assert rec.commands == ["ls", f"echo {SUBMIT_SENTINEL}"]


async def test_submission_is_read_from_what_the_command_printed():
    # Upstream's rule, and it is not the same as matching the command text: the
    # sentinel runs like anything else and the shell's output is what ends the run.
    # A command that merely mentions the sentinel does not submit.
    rec, summary = await drive([call(f"grep -r {SUBMIT_SENTINEL} ."), call("ls"),
                                SUBMIT],
                               tool_result=None)
    assert summary["stop_reason"] == STOP_SUBMITTED
    assert summary["commands_run"] == 3


async def test_a_sentinel_that_did_not_come_first_does_not_submit():
    # The prompt says not to combine it with another command, and the rule enforces
    # that: the sentinel has to be the first line the command printed.
    rec, summary = await drive([call("ls"), call("pwd")], max_turns=2,
                               tool_result={"output": f"other\n{SUBMIT_SENTINEL}\n",
                                            "exit_code": 0})
    assert summary["stop_reason"] == STOP_STEP_LIMIT


async def test_a_failed_sentinel_command_does_not_submit():
    # returncode 0 is part of upstream's rule. A shell that printed the sentinel while
    # failing did not complete the task.
    _, summary = await drive([call("./submit.sh"), call("ls")], max_turns=2,
                             tool_result={"output": SUBMIT_SENTINEL + "\n",
                                          "exit_code": 1})
    assert summary["stop_reason"] == STOP_STEP_LIMIT


async def test_the_step_limit_ends_the_trajectory():
    _, summary = await drive([call("ls")] * 3, max_turns=3)
    assert summary["stop_reason"] == STOP_STEP_LIMIT
    assert summary["turns"] == 3


async def test_an_empty_reply_ends_the_trajectory():
    _, summary = await drive(["   "])
    assert summary["stop_reason"] == STOP_EMPTY_REPLY


# --- a turn with nothing visible in it -------------------------------------

#: The per-turn output cap in these tests. Small enough to write a served count
#: against; the real one is the task's ``MAX_NEW_TOKENS``.
_CAP = 4_096

#: A turn that reasoned to the cap, as the two kinds of server hand it over. One
#: strips the reasoning into its own field and returns nothing; the other closes the
#: channel with no room left to answer, so the raw reply is long and its visible part
#: is still nothing.
#:
#: Where each is reachable, because the stripped one is narrower than it looks. These
#: tests drive the loop directly, and the framework above it refuses a reply whose text
#: is not what its tokens say (``check_text_matches_tokens``) -- empty text against a
#: full cap of tokens is exactly that, so a mounted checkpoint rejects it before the
#: loop sees it. It reaches the loop only where that check is unarmed, which is any
#: process with no checkpoint: the build sandbox's self-test, and these tests. The
#: closed-at-the-cap one has no such caveat and is what a conformant server sends.
#:
#: Both are worth keeping -- the loop's behaviour on them is real in the configuration
#: that produces them -- but a stand-in that can hand the loop something the shipped
#: path cannot is a test that passes forever and stops meaning anything the day the
#: real component changes.
_STRIPPED = ""
_CLOSED_AT_THE_CAP = "I should check whether the widget serialises its choices." * 20 + "</think>"


async def _starved(reply_text: str, **kwargs) -> dict:
    """One trajectory whose first reply has nothing visible in it."""
    _, summary = await drive([reply_text], max_output_tokens=_CAP, **kwargs)
    return summary


async def test_a_turn_that_reasoned_to_the_cap_is_asked_again():
    # A model that spent a whole cap thinking produced plenty of tokens and simply
    # never reached an answer. Ending the trajectory there would be this harness
    # inventing a terminal condition no production agent loop has.
    rec, _ = await drive([_STRIPPED, call("ls"), SUBMIT], max_output_tokens=_CAP,
                         finish_reason="length", served_tokens=_CAP)
    assert rec.commands[0] == "ls"


async def test_a_turn_that_thought_at_length_and_stopped_is_asked_again():
    # A turn can serve tens of thousands of tokens, carry no visible text, and stop on
    # its own well short of the cap. That is the same starved turn as one that hit the
    # cap -- the model spent real budget and never reached an answer -- so it is asked
    # again rather than ending the run.
    rec, summary = await drive([_STRIPPED, call("ls"), SUBMIT],
                               max_output_tokens=131_072,
                               finish_reason="stop", served_tokens=60_294)
    assert summary["stop_reason"] != STOP_EMPTY_REPLY
    assert rec.commands[0] == "ls"


async def test_a_server_that_said_nothing_is_still_an_empty_reply():
    summary = await _starved("   ", finish_reason="stop", served_tokens=0)
    assert summary["stop_reason"] == STOP_EMPTY_REPLY


async def test_a_server_that_said_nothing_is_not_asked_again():
    # There is nothing to ask again for: the reply carried no tokens at all, so the
    # repair path would spend turns on a server that is not answering.
    rec, summary = await drive(["   ", call("ls")], finish_reason="stop",
                               served_tokens=0)
    assert summary["turns"] == 1
    assert rec.commands == []


async def test_whitespace_that_cost_tokens_is_repaired_not_called_empty():
    # Zero is the boundary, not "not many". A reply of three whitespace tokens is the
    # model producing something unusable, which is the repair case; only a completion
    # that carried nothing at all is the server failing to answer. This keeps
    # STOP_EMPTY_REPLY meaning a broken server rather than a disappointing turn.
    rec, _ = await drive(["   ", call("ls"), SUBMIT], finish_reason="stop",
                         served_tokens=3)
    assert rec.commands[0] == "ls"


async def test_repeated_starved_turns_end_the_trajectory_on_their_own_verdict():
    # Asked again, but not forever, and the trajectory ends naming what happened
    # rather than reporting a format problem the model never had.
    rec, summary = await drive([_STRIPPED] * 6, max_output_tokens=_CAP,
                               finish_reason="length", served_tokens=_CAP,
                               max_consecutive_format_errors=2)
    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert rec.events[-1]["extras"]["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert rec.commands == []


async def test_both_kinds_of_starved_turn_are_read_the_same_way():
    # Whether the reasoning was stripped server-side is the server's choice, and the
    # same starved turn must not end one contestant's trajectory as a broken artifact
    # and another's as a turn worth repeating.
    for text in (_STRIPPED, _CLOSED_AT_THE_CAP):
        _, summary = await drive([text] * 6, max_output_tokens=_CAP,
                                 finish_reason="length", served_tokens=_CAP,
                                 max_consecutive_format_errors=2)
        assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED


async def test_a_short_reply_with_nothing_visible_is_still_asked_again():
    # The boundary is whether the reply carried tokens, not how many. A server that
    # claims "length" cheaply buys at most max_consecutive_format_errors more turns and
    # then the same ending, so the budget is the protection and the cap does not need
    # to be.
    rec, summary = await drive([_STRIPPED, call("ls"), SUBMIT],
                               max_output_tokens=_CAP,
                               finish_reason="length", served_tokens=12)
    assert summary["stop_reason"] != STOP_EMPTY_REPLY
    assert rec.commands[0] == "ls"


async def test_the_repair_budget_bounds_a_server_that_never_answers():
    # What stops a server spending the trajectory on cheap unanswerable turns. Without
    # this, repairing any non-empty reply would let one run to the step limit.
    rec, summary = await drive([_STRIPPED] * 20, max_output_tokens=_CAP,
                               finish_reason="stop", served_tokens=12,
                               max_consecutive_format_errors=2, max_turns=20)
    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    # Upstream's budget counts errors, not retries: at 2, the second unreadable reply
    # in a row is the last turn.
    assert summary["turns"] == 2
    assert rec.commands == []


async def test_the_stop_label_decides_where_no_token_count_arrived():
    # The build sandbox's self-test drives the loop with no served counts.
    _, ran_on = await drive([_STRIPPED] * 6, max_output_tokens=_CAP,
                            finish_reason="length", max_consecutive_format_errors=2)
    assert ran_on["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert (await _starved(_STRIPPED, finish_reason="stop"))["stop_reason"] \
        == STOP_EMPTY_REPLY


# --- what spent the repair budget --------------------------------------------
#
# The budget ends a trajectory after N consecutive turns that produced no command.
# Two quite different failures reach it. A reply the parser could not read is the
# parser under suspicion; a reply that spent its whole output cap reasoning and never
# wrote a call is the model.

#: A reply that wrote a delimiter and no readable action: the parser's case.
_UNPARSEABLE = ("Let me look.</think><tool_call>bash<arg_key>command</arg_key>"
                "<arg_value>ls")


async def test_a_trajectory_killed_by_runaways_does_not_report_a_parser_failure():
    # Every failure here ran the output cap out inside the reasoning channel and
    # never wrote a delimiter, so naming the ending for the parser sends a reader to
    # look at a parser that did nothing wrong.
    _, summary = await drive([_STRIPPED] * 6, max_output_tokens=_CAP,
                             finish_reason="length", served_tokens=_CAP,
                             max_consecutive_format_errors=2)

    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert summary["no_command_causes"] == {NO_COMMAND_RUNAWAY: 2}


async def test_a_trajectory_killed_by_unparsed_calls_still_reports_one():
    # The control: separating the two endings is worth nothing if the real parser
    # failure stops being reported as one.
    _, summary = await drive([_UNPARSEABLE] * 6, served_tokens=50,
                             finish_reason="stop", max_consecutive_format_errors=2)

    assert summary["stop_reason"] == STOP_FORMAT_REPAIR_EXHAUSTED
    assert summary["no_command_causes"] == {NO_COMMAND_UNPARSED: 2}


class _MixedEndings(Recorder):
    """A server whose replies end differently from one another.

    The helper applies one stop label to a whole drive, which cannot express a run
    holding both a runaway and a reply that finished unreadable. With one label every
    reply is marked as having hit the cap, so the unreadable one is a runaway too and
    the parser is never in the run."""

    async def complete(self, request: dict) -> dict:
        text = self.replies.pop(0) if self.replies else "no call here"
        self.requests.append(request)
        at_cap = text != _UNPARSEABLE
        return {"text": text, "metadata": {},
                "finish_reason": "length" if at_cap else "stop",
                "served_tokens": _CAP if at_cap else 50}


async def test_a_run_holding_one_parser_failure_is_reported_as_a_parser_failure():
    # Mixed runs resolve toward the parser. A run that contains a real unparsed call
    # is a run where the parser did fail, and the reading that hides that is the one
    # to avoid; the breakdown still says how many of each.
    rec = _MixedEndings([_STRIPPED, _UNPARSEABLE, _STRIPPED] + [_STRIPPED] * 3)
    loop = MiniSweLoop(max_output_tokens=_CAP, max_consecutive_format_errors=3)
    summary = await loop.run(
        session_id="s0", problem_statement="Fix the bug.",
        base_metadata={"tenant_id": "t0"}, complete=rec.complete,
        run_tool=rec.run_tool, emit=rec.emit, max_turns=10)

    assert summary["stop_reason"] == STOP_FORMAT_REPAIR_EXHAUSTED
    assert summary["no_command_causes"][NO_COMMAND_UNPARSED] == 1
    assert summary["no_command_causes"][NO_COMMAND_RUNAWAY] == 2


async def test_a_reply_that_simply_did_not_call_is_neither():
    # The third way a turn yields nothing: prose, inside the cap, no delimiter. Not a
    # parser failure and not a runaway, and reporting it as either would be the same
    # mistake in a smaller font.
    _, summary = await drive(["just prose"] * 6, served_tokens=50,
                             finish_reason="stop", max_consecutive_format_errors=2)

    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert summary["no_command_causes"] == {NO_COMMAND_NO_CALL: 2}


async def test_a_turn_that_ran_a_command_clears_what_came_before_it():
    # The budget counts a consecutive run, so the breakdown has to describe the run
    # that ended the trajectory rather than the whole session. A parser failure early
    # on must not make a later death by runaway read as the parser's.
    _, summary = await drive([_UNPARSEABLE, call("ls")] + [_STRIPPED] * 6,
                             max_output_tokens=_CAP, finish_reason="length",
                             served_tokens=_CAP, max_consecutive_format_errors=2)

    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert NO_COMMAND_UNPARSED not in summary["no_command_causes"]


async def test_the_breakdown_rides_on_the_last_event_too():
    # The diagnostics group by session and read the last event, so a breakdown only
    # on the returned summary would not reach them.
    rec, _ = await drive([_STRIPPED] * 6, max_output_tokens=_CAP,
                         finish_reason="length", served_tokens=_CAP,
                         max_consecutive_format_errors=2)

    assert rec.events[-1]["extras"]["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert rec.events[-1]["extras"]["no_command_causes"] == {NO_COMMAND_RUNAWAY: 2}


async def test_a_trajectory_that_ended_well_carries_no_breakdown():
    # Only an ending the budget caused has a run to describe. An empty mapping on
    # every other trajectory would read as a claim about them.
    _, summary = await drive([call("ls"), SUBMIT])

    assert summary["stop_reason"] == STOP_SUBMITTED
    assert summary["no_command_causes"] == {}


# --- the parser alarm, and what must not set it off ------------------------
#
# ``unparsed_tool_call`` exists to accuse this parser: a reply that opened a call and
# yielded no action is the parser under suspicion, and a non-zero count is meant to
# send someone to read it. That is only true of a reply that finished. One cut off at
# the output cap mid-call has demonstrated nothing about the parser, and on this task
# that is the common failure.

#: A reply that opened a call and then ran the whole cap out repeating itself.
_CALL_CUT_OFF_AT_THE_CAP = (
    "Let me look.</think><tool_call>bash<arg_key>command</arg_key><arg_value>ls -la ")


async def test_a_call_the_output_cap_cut_off_does_not_accuse_the_parser():
    # The parser does what it documents, half a command is not an action, so an alarm
    # that fires here sends the first person who sees it to audit working code.
    rec, _ = await drive([_CALL_CUT_OFF_AT_THE_CAP] * 6, max_output_tokens=_CAP,
                         finish_reason="length", served_tokens=_CAP,
                         max_consecutive_format_errors=2)

    assert rec.events[0]["extras"]["unparsed_tool_call"] is False
    assert rec.events[0]["extras"]["ran_to_output_cap"] is True


async def test_a_call_that_really_was_unreadable_still_accuses_the_parser():
    # The control. Narrowing the alarm is worth nothing if it stops firing on the case
    # it exists for: a reply that finished, inside its cap, having written a call this
    # parser could not read.
    rec, _ = await drive([_UNPARSEABLE] * 6, served_tokens=50, finish_reason="stop",
                         max_consecutive_format_errors=2)

    assert rec.events[0]["extras"]["unparsed_tool_call"] is True
    assert rec.events[0]["extras"]["ran_to_output_cap"] is False


async def test_a_cut_off_call_is_a_runaway_rather_than_the_parsers_fault():
    # The two records agree: the alarm stays quiet and the ending names the runaway.
    _, summary = await drive([_CALL_CUT_OFF_AT_THE_CAP] * 6, max_output_tokens=_CAP,
                             finish_reason="length", served_tokens=_CAP,
                             max_consecutive_format_errors=2)

    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert summary["no_command_causes"] == {NO_COMMAND_RUNAWAY: 2}


async def test_the_reply_still_records_that_it_tried_to_call():
    # The raw fact is kept beside the accusation built on it, so a reader can still
    # see a call was attempted.
    rec, _ = await drive([_CALL_CUT_OFF_AT_THE_CAP] * 6, max_output_tokens=_CAP,
                         finish_reason="length", served_tokens=_CAP,
                         max_consecutive_format_errors=2)

    assert rec.events[0]["extras"]["attempted_tool_call"] is True


# --- a caller that stops the pass is not a trajectory that failed ----------
#
# The pre-roll bounds a pass in wall clock and in spend and cuts by raising out of its
# own hook. That is a scheduled stop, the distinction the drive layer draws between a
# cut and a fault. Turns recorded as "error" would contradict the pass's own summary,
# which says it stopped on its deadline.


async def _run_with_hook(replies, before_turn, **kwargs):
    """Drive the loop directly, keeping the recorder when the hook raises.

    The hook's exception propagates to its caller by design -- the pre-roll catches
    its own -- so there is no summary to read and the verdict has to be read off the
    last turn, which is where the diagnostics read it from anyway."""
    rec = Recorder(list(replies))
    loop = MiniSweLoop(**kwargs)
    await loop.run(session_id="s0", problem_statement="Fix the bug.",
                   base_metadata={"tenant_id": "t0"}, complete=rec.complete,
                   run_tool=rec.run_tool, emit=rec.emit, max_turns=10,
                   before_turn=before_turn)
    return rec


class _OutOfTime(Exception):
    """Stands in for the pre-roll's own deadline, which is private to it."""


async def test_a_caller_that_stops_the_pass_is_not_recorded_as_an_error():
    async def cut(agent):
        if agent.model.recorder.turns >= 1:
            raise _OutOfTime

    rec = Recorder([call("ls")] * 4)
    loop = MiniSweLoop()
    with pytest.raises(_OutOfTime):
        await loop.run(session_id="s0", problem_statement="Fix the bug.",
                       base_metadata={"tenant_id": "t0"}, complete=rec.complete,
                       run_tool=rec.run_tool, emit=rec.emit, max_turns=10,
                       before_turn=cut)

    assert rec.events[-1]["extras"]["stop_reason"] == STOP_CALLER_STOPPED


async def test_the_callers_exception_still_reaches_the_caller():
    # Naming the stop must not swallow the signal the pre-roll branches on: it tells
    # a deadline from a spend from a band cut by catching its own exceptions.
    async def cut(agent):
        raise _OutOfTime

    with pytest.raises(_OutOfTime):
        await _run_with_hook([call("ls")] * 3, cut)


async def test_a_trajectory_that_really_raised_is_still_an_error():
    # THE control for this half. Narrowing "error" is worth nothing if a genuine
    # failure stops being one. This raises from the server call rather than the hook,
    # which is a fault in the trajectory and has to keep reading as one.
    class Broken(Recorder):
        async def complete(self, request):
            raise RuntimeError("the artifact died")

    rec = Broken([call("ls")])
    loop = MiniSweLoop()
    with pytest.raises(RuntimeError, match="the artifact died"):
        await loop.run(session_id="s0", problem_statement="Fix the bug.",
                       base_metadata={"tenant_id": "t0"}, complete=rec.complete,
                       run_tool=rec.run_tool, emit=rec.emit, max_turns=10,
                       before_turn=None)

    # No turn was recorded at all here, so the check that matters is the one below:
    # a hook that returns cleanly leaves every ending exactly as it was.
    assert rec.events == []


async def test_a_hook_that_does_not_raise_changes_no_ending():
    seen = []

    async def watch(agent):
        seen.append(agent.model.recorder.turns)

    rec = await _run_with_hook([call("ls"), SUBMIT], watch)

    assert seen == [0, 1]
    assert rec.events[-1]["extras"]["stop_reason"] == STOP_SUBMITTED


def _user_prompts(rec: Recorder) -> list[str]:
    """Every user message the loop appended, across all requests."""
    seen: list[str] = []
    for req in rec.requests:
        for m in req["metadata"]["messages"]:
            if m["role"] == "user" and m["content"] not in seen:
                seen.append(m["content"])
    return seen


async def test_a_truncated_turn_is_told_it_ran_out_of_room():
    # The two failures need opposite advice. A reply the loop could not parse should be
    # reformatted; one that never finished should be shorter. Telling a model that ran
    # out of room to "use the bash tool" does not say what went wrong.
    rec, _ = await drive([_STRIPPED, call("ls"), SUBMIT], max_output_tokens=_CAP,
                         finish_reason="length", served_tokens=_CAP)
    asked = _user_prompts(rec)
    assert any("reached the output token limit" in m for m in asked)
    assert not any("Tool call error" in m for m in asked)


async def test_an_unparseable_turn_is_told_to_reformat():
    # The other arm, so the branch is pinned in both directions rather than just one.
    rec, _ = await drive(["prose, no call", call("ls"), SUBMIT],
                         max_output_tokens=_CAP,
                         finish_reason="stop", served_tokens=40)
    asked = _user_prompts(rec)
    assert any("Tool call error" in m for m in asked)
    assert not any("reached the output token limit" in m for m in asked)


async def test_which_repair_a_turn_gets_is_not_the_servers_to_choose():
    # The template branches on finish_reason, and finish_reason is a string the server
    # writes. A server claiming "length" on a reply that spent 40 of a 4,096 cap gets
    # the reformat message, because the token count is what decides.
    rec, _ = await drive(["prose, no call", call("ls"), SUBMIT],
                         max_output_tokens=_CAP,
                         finish_reason="length", served_tokens=40)
    assert any("Tool call error" in m for m in _user_prompts(rec))


def test_the_system_prompt_does_not_ask_a_thinking_model_to_think_twice():
    # This checkpoint's chat template ends its generation prompt with <think>, so the
    # reasoning channel is already open. A prompt that also demands written reasoning
    # gets both, and the reply then closes the channel twice.
    assert "THOUGHT" not in SYSTEM_TEMPLATE
    assert "THOUGHT" not in INSTANCE_TEMPLATE
    assert "reasoning text" not in INSTANCE_TEMPLATE.casefold()


async def test_the_repair_budget_is_consecutive_not_lifetime():
    # The budget bounds a model that cannot produce the format, which is a run of
    # failures, not a tally of every one a long trajectory ever had. Upstream resets on
    # any clean step and so does this.
    good = call("ls")
    rec, summary = await drive(["no call", good, "no call", good, "no call", good,
                                SUBMIT],
                               max_consecutive_format_errors=2, max_turns=20)
    assert summary["stop_reason"] == STOP_SUBMITTED
    assert rec.commands == ["ls", "ls", "ls", f"echo {SUBMIT_SENTINEL}"]


async def test_a_run_of_failures_still_ends_the_trajectory():
    # The other half: reset on progress, but a model that never recovers still stops.
    _, summary = await drive(["no call"] * 6, max_consecutive_format_errors=2,
                             max_turns=20)
    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert summary["turns"] == 2


async def test_replies_carrying_no_call_are_repaired_then_give_up():
    # Named for what the fixture is. These replies write no delimiter, so the parser
    # is never asked to read one and the ending is not the parser's. An unreadable
    # delimiter is a different fixture and a different ending, pinned separately.
    rec, summary = await drive(["no call"] * 10, max_consecutive_format_errors=2)
    assert summary["stop_reason"] == STOP_NO_COMMAND_REPEATED
    # One repair is offered, and the second unreadable reply ends it: the budget is a
    # count of errors, which is upstream's reading of its own knob.
    assert summary["turns"] == 2
    assert rec.commands == []


async def test_a_repaired_reply_resumes_the_loop():
    rec, summary = await drive(["no call", call("ls"), SUBMIT])
    assert summary["stop_reason"] == STOP_SUBMITTED
    assert rec.commands == ["ls", f"echo {SUBMIT_SENTINEL}"]


async def test_an_unparseable_turn_is_not_echoed_back_into_the_conversation():
    # Upstream drops the assistant turn it could not read and appends only the repair
    # prompt, so a model that rambles does not pay for the ramble on every later turn.
    rec, _ = await drive(["rambling with no call", call("ls"), SUBMIT])
    roles = [m["role"] for m in rec.requests[1]["metadata"]["messages"]]
    assert roles == ["system", "user", "user"]


# --- what the conversation carries ----------------------------------------

async def test_the_first_request_carries_system_then_instance():
    rec, _ = await drive([SUBMIT])
    messages = rec.requests[0]["metadata"]["messages"]
    assert [m["role"] for m in messages] == ["system", "user"]
    assert "Fix the bug." in messages[1]["content"]


async def test_every_request_offers_the_bash_tool():
    # The definitions ride on the request's metadata, which is where the wire
    # protocol, the reference engines and the verifier all read them, so the artifact
    # and the oracle render one prompt.
    rec, _ = await drive([call("ls"), SUBMIT])
    for req in rec.requests:
        tools = req["metadata"]["tools"]
        assert [t["function"]["name"] for t in tools] == ["bash"]


async def test_the_observation_becomes_a_tool_message_answering_the_call():
    rec, _ = await drive([call("ls"), SUBMIT],
                         tool_result={"output": "file.py", "exit_code": 0})
    messages = rec.requests[1]["metadata"]["messages"]
    assert messages[-1]["role"] == "tool"
    assert "file.py" in messages[-1]["content"]
    assert '"returncode": 0' in messages[-1]["content"]
    assert messages[-2]["role"] == "assistant"
    # The result still names the call it answers, which is what the tool role is for.
    assert messages[-1]["tool_call_id"] == "call_0_0"


async def test_the_assistant_turn_goes_back_as_the_text_the_model_wrote():
    # The reply is replayed verbatim and carries no structured `tool_calls` field.
    # Three reasons. A dict `arguments` is refused by the OpenAI request
    # schema outright; a JSON-string `arguments` is what the schema wants but raises
    # UndefinedError under a direct apply_chat_template, because this template iterates
    # it as a mapping -- which is what the prompt counter and the oracle both do. And
    # replaying the text is what makes the round trip exact, since the bytes we send
    # are the bytes the model produced.
    rec, _ = await drive([call("ls -la", prose="Listing it."), SUBMIT])
    assistant = rec.requests[1]["metadata"]["messages"][-2]
    assert assistant["role"] == "assistant"
    assert "tool_calls" not in assistant
    assert "Listing it." in assistant["content"]
    assert assistant["content"].count("<tool_call>") == 1
    assert "<arg_value>ls -la</arg_value>" in assistant["content"]


async def test_the_assistant_turn_carries_the_reasoning_it_served():
    rec, _ = await drive([call("ls", thought="Check the directory first."), SUBMIT])
    assistant = rec.requests[1]["metadata"]["messages"][-2]
    assert assistant["reasoning_content"] == "Check the directory first."


async def test_the_agents_own_bookkeeping_never_reaches_the_wire():
    # Every message carries an `extra` the agent uses to hold the response object and
    # the parsed actions. Sending it would put the whole previous completion into the
    # next prompt.
    rec, _ = await drive([call("ls"), SUBMIT])
    for req in rec.requests:
        assert all("extra" not in m for m in req["metadata"]["messages"])


async def test_a_command_that_printed_nothing_says_so():
    rec, _ = await drive([call("true"), SUBMIT],
                         tool_result={"output": "", "exit_code": 0})
    assert '"output": ""' in rec.requests[1]["metadata"]["messages"][-1]["content"]


async def test_a_turn_that_wrote_a_delimiter_and_no_action_is_marked_as_such():
    # The distinction that keeps a parser bug from being filed as a model failure: a
    # reply with no delimiter at all is a model that did not act, and a reply that
    # wrote one and still produced nothing is this parser under suspicion.
    cut_off = "Let me look.</think><tool_call>bash<arg_key>command</arg_key><arg_value>ls"
    rec, _ = await drive([cut_off, "just prose", call("ls"), SUBMIT],
                         served_tokens=50, finish_reason="stop")
    assert rec.events[0]["extras"]["unparsed_tool_call"] is True
    assert rec.events[1]["extras"]["unparsed_tool_call"] is False
    assert rec.events[2]["extras"]["attempted_tool_call"] is True


async def test_the_output_cap_is_per_command_not_per_turn():
    # Deliberate, and worth pinning because it doubles the worst-case context growth
    # of a turn: a reply usually carries two calls, and each output is rendered and
    # elided on its own. Capping the pair jointly would let one command's long output
    # starve the other's, and rendering per output is upstream's shape.
    #
    # The total is not unbounded by it -- the context guard counts the whole rendered
    # prompt before every send -- but the arithmetic on
    # minisweagent.config.mini.OBSERVATION_LONG depends on this being per command.
    cap = 4_000
    two = (f"{call('cmd1')}"
           "<tool_call>bash<arg_key>command</arg_key>"
           "<arg_value>cmd2</arg_value></tool_call>")
    rec, _ = await drive([two, SUBMIT], output_limit_chars=cap,
                         tool_result={"output": "X" * 100_000, "exit_code": 0})
    observations = [m for m in rec.requests[1]["metadata"]["messages"]
                    if m["role"] == "tool"]
    assert len(observations) == 2
    for m in observations:
        assert cap < len(m["content"]) < cap * 1.1   # each elided to the cap alone
    assert sum(len(m["content"]) for m in observations) > cap * 1.9


async def test_both_calls_in_a_two_call_reply_are_run():
    # The common case on this checkpoint. Upstream executes every action in the
    # message, in order.
    two = (f"{call('ls')}"
           "<tool_call>bash<arg_key>command</arg_key>"
           "<arg_value>pwd</arg_value></tool_call>")
    rec, _ = await drive([two, SUBMIT])
    assert rec.commands[:2] == ["ls", "pwd"]
    assert rec.events[0]["extras"]["n_tool_calls"] == 2


async def test_the_timeout_rides_on_the_tool_call():
    rec, _ = await drive([call("sleep 1"), SUBMIT], command_timeout_sec=42)
    assert rec.tool_args[0]["timeout_sec"] == 42


async def test_a_timed_out_command_says_so_in_the_observation():
    rec, _ = await drive([call("sleep 1000"), SUBMIT], command_timeout_sec=42,
                         tool_result={"output": "", "exit_code": -1,
                                      "timed_out": True})
    assert "timed out after 42 seconds" in \
        rec.requests[1]["metadata"]["messages"][-1]["content"]


async def test_the_last_event_carries_the_trajectory_verdict():
    # Every way of stopping must leave the verdict on a turn, including the step
    # limit, which ends on an otherwise ordinary turn.
    for replies, kwargs, expected in [
        ([call("ls"), SUBMIT], {}, STOP_SUBMITTED),
        ([call("ls")] * 3, {"max_turns": 3}, STOP_STEP_LIMIT),
        (["   "], {}, STOP_EMPTY_REPLY),
        (["no call"] * 5, {"max_consecutive_format_errors": 1},
         STOP_NO_COMMAND_REPEATED),
        ([_UNPARSEABLE] * 5, {"max_consecutive_format_errors": 1,
                              "served_tokens": 50, "finish_reason": "stop"},
         STOP_FORMAT_REPAIR_EXHAUSTED),
    ]:
        rec, summary = await drive(replies, **kwargs)
        assert rec.events[-1]["extras"]["stop_reason"] == expected
        assert summary["stop_reason"] == expected
        assert rec.events[-1]["extras"]["session_turns"] == summary["turns"]


async def test_only_the_last_event_carries_a_stop_reason():
    rec, _ = await drive([call("ls"), call("pwd"), SUBMIT])
    assert [("stop_reason" in e["extras"]) for e in rec.events] == [False, False, True]


async def test_one_event_per_llm_call():
    rec, summary = await drive([call("ls"), call("pwd"), SUBMIT])
    assert len(rec.events) == len(rec.requests) == 3
    assert [e["turn_index"] for e in rec.events] == [0, 1, 2]
    assert all(e["request_id"] == f"s0.{e['turn_index']}" for e in rec.events)


async def test_a_turn_the_parser_could_not_read_is_still_in_the_trace():
    # Upstream's invariant: the response is persisted even when parsing failed. A turn
    # the artifact was served and billed for is part of the measurement.
    rec, _ = await drive(["no call", call("ls"), SUBMIT])
    assert len(rec.events) == 3
    assert rec.events[0]["extras"]["parsed_command"] is None
    assert rec.events[0]["completion"]["text"] == "no call"


async def test_a_trajectory_that_raised_still_emits_the_turn_it_had_served():
    class Boom(Recorder):
        async def run_tool(self, name, args):
            raise RuntimeError("sandbox died")

    rec = Boom([call("ls")])
    with pytest.raises(RuntimeError, match="sandbox died"):
        await MiniSweLoop().run(session_id="s0", problem_statement="p",
                                base_metadata={}, complete=rec.complete,
                                run_tool=rec.run_tool, emit=rec.emit, max_turns=5)
    assert len(rec.events) == 1
    # Named for what happened. A crashed sandbox counted as a step limit would read,
    # in the diagnostics, as a trajectory that quietly used up its turns.
    assert rec.events[-1]["extras"]["stop_reason"] == STOP_ERROR


# --- the context guard -----------------------------------------------------

#: A base64 blob, the densest thing a coding agent routinely puts in its transcript.
#: The checkpoint's tokenizer reads about 1.42 characters per token on text like this.
DENSE = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5"

#: Characters per token a guard would assume as a floor to convert a window into a
#: character budget. Tests assert against it to show what such a budget lets through.
OLD_CHARS_PER_TOKEN = 2.5


def counter(chars_per_token: int):
    """A stand-in tokenizer: one token per ``chars_per_token`` characters of
    transcript.

    Four is about what prose and source measure on the real checkpoint; one is what a
    base64 blob or a hex dump approaches. A stub rather than a mounted checkpoint keeps
    the suite on CPU and off the weights."""
    async def count(messages: list[dict],
                    continue_final_message: bool = False) -> int:
        # The flag is accepted and ignored: a stub that counts characters has no
        # generation prompt to leave off. Accepting it is what keeps the double
        # honouring the seam's signature, so a caller that passes it here behaves the
        # way it will against the real counter.
        return sum(len(m["content"]) for m in messages) // chars_per_token
    return count


def transcript(chars: int, filler: str = DENSE) -> list[dict]:
    """A two-message conversation carrying ``chars`` characters of ``filler``."""
    body = (filler * (chars // len(filler) + 1))[:chars]
    return [{"role": "system", "content": "s"}, {"role": "user", "content": body}]


def guard(*, messages: list[dict] | None = None, reported: object = None,
          reported_prefix: int = 0, **kwargs):
    """An agent holding ``messages``, so the guard can be asked directly."""
    rec = Recorder([])
    loop = MiniSweLoop(**kwargs)
    agent = loop.agent(session_id="s0", base_metadata={}, complete=rec.complete,
                       run_tool=rec.run_tool,
                       recorder=TurnRecorder(emit=rec.emit, session_id="s0"),
                       max_turns=10)
    agent.messages = list(messages or [])
    agent.model._reported = reported
    agent.model._reported_prefix = reported_prefix
    return agent


async def test_dense_text_over_the_window_is_stopped_though_its_characters_fit():
    # The case the token count exists for: a transcript whose character count sits
    # inside a character budget while its tokens are 60% past the window.
    messages = transcript(6_400)
    assert sum(len(m["content"]) for m in messages) < 4_000 * OLD_CHARS_PER_TOKEN
    agent = guard(messages=messages, max_context_len=4_000,
                  count_prompt_tokens=counter(1))
    assert await agent.over_context_budget()


async def test_ordinary_prose_under_the_window_is_not_stopped():
    # The guard must not end trajectories early: that would shrink the workload
    # without anything saying so.
    agent = guard(messages=transcript(6_400), max_context_len=4_000,
                  count_prompt_tokens=counter(4))
    assert not await agent.over_context_budget()


async def test_a_prompt_that_fits_but_leaves_no_room_to_answer_is_stopped():
    # 5,000 prompt tokens sit inside an 8,192-token window, and the 4,096 the next
    # reply may use do not.
    messages = transcript(20_000)
    fits = guard(messages=messages, max_context_len=8_192,
                 count_prompt_tokens=counter(4))
    assert not await fits.over_context_budget()
    no_room = guard(messages=messages, max_context_len=8_192,
                    count_prompt_tokens=counter(4), max_output_tokens=4_096)
    assert await no_room.over_context_budget()


async def test_an_undeclared_window_leaves_the_guard_unarmed():
    agent = guard(messages=transcript(10**6), count_prompt_tokens=counter(1))
    assert not await agent.over_context_budget()


async def test_the_counted_prompt_beats_what_the_server_claims():
    # A server inflating its reported prompt length would end every trajectory on its
    # first turn and collect the speedup. The count is what decides.
    agent = guard(messages=transcript(400), reported=10**9, reported_prefix=2,
                  max_context_len=4_000, count_prompt_tokens=counter(4))
    assert not await agent.over_context_budget()


# --- the context guard with no tokenizer to count with ---------------------

async def test_the_byte_bound_caps_an_inflated_reported_prompt():
    # No counter, so the reply's own figure is all there is -- capped at the bytes the
    # conversation occupies, since a token covers at least one byte.
    messages = transcript(400)
    bound = prompt_byte_bound(messages)
    agent = guard(messages=messages, reported=10**9, reported_prefix=len(messages),
                  max_context_len=bound * 2)
    assert await agent.model.count_prompt_tokens(messages) == bound
    assert not await agent.over_context_budget()


async def test_an_honest_reported_prompt_over_the_window_is_stopped():
    # 401 characters, which a character budget would call small, and 500 tokens
    # against a 200-token window.
    messages = transcript(400)
    assert sum(len(m["content"]) for m in messages) < 200 * OLD_CHARS_PER_TOKEN
    agent = guard(messages=messages, reported=500, reported_prefix=len(messages),
                  max_context_len=200)
    assert await agent.over_context_budget()


async def test_the_fallback_carries_the_reported_figure_past_what_it_measured():
    # The reported figure describes the prompt of the last send, and the transcript
    # has grown since. What was appended is bounded by its bytes rather than assumed
    # away, so the fallback cannot read short.
    sent = transcript(400)
    appended = [{"role": "assistant", "content": "x" * 2_000}]
    agent = guard(reported=500, reported_prefix=len(sent), max_context_len=10**9)
    carried = await agent.model.count_prompt_tokens(sent + appended)
    assert carried == 500 + prompt_byte_bound(appended)


async def test_nothing_to_count_and_nothing_reported_leaves_the_guard_unarmed():
    agent = guard(messages=transcript(400), max_context_len=1)
    assert await agent.model.count_prompt_tokens(transcript(400)) is None
    assert not await agent.over_context_budget()


# --- the context guard on a running trajectory -----------------------------

#: One trajectory's worth of arguments, so the two tests below differ only in how
#: densely the transcript tokenizes. Five turns of 2,400-character command output
#: reach about 14,500 characters, which is 14,500 tokens dense and 3,600 in prose
#: against an 8,000-token window. Every one of those prompts is inside the 20,000
#: characters a budget at ``OLD_CHARS_PER_TOKEN`` allows, so it would send them all.
_TRAJECTORY = dict(max_turns=5, tool_result={"output": DENSE * 50, "exit_code": 0},
                   max_context_len=8_000)


async def test_a_dense_trajectory_ends_on_its_own_terms():
    rec, summary = await drive([call("xxd fixture.bin")] * 5,
                               count_prompt_tokens=counter(1), **_TRAJECTORY)
    assert summary["stop_reason"] == STOP_CONTEXT_EXHAUSTED
    # Stopped before the send, so the over-length prompt was never a request.
    assert len(rec.requests) == summary["turns"] < 5
    assert rec.events[-1]["extras"]["stop_reason"] == STOP_CONTEXT_EXHAUSTED


async def test_the_same_trajectory_in_prose_runs_to_its_end():
    _, summary = await drive([call("cat README.md")] * 4 + [SUBMIT],
                             count_prompt_tokens=counter(4), **_TRAJECTORY)
    assert summary["stop_reason"] == STOP_SUBMITTED
    assert summary["commands_run"] == 5


async def test_a_running_trajectory_ignores_an_inflated_reported_prompt():
    _, summary = await drive([call("ls")] * 4 + [SUBMIT],
                             reported_prompt_tokens=10**9,
                             count_prompt_tokens=counter(4), **_TRAJECTORY)
    assert summary["stop_reason"] == STOP_SUBMITTED


# --- the prompt length each turn carries into its trace event --------------

def counting_counter(chars_per_token: int) -> tuple[list[int], object]:
    """``counter``, plus a list that grows by one per call, so a test can see how
    many times a turn tokenized its transcript."""
    calls: list[int] = []
    inner = counter(chars_per_token)

    async def count(messages: list[dict],
                    continue_final_message: bool = False) -> int:
        calls.append(len(messages))
        return await inner(messages, continue_final_message)

    return calls, count


async def test_every_turn_records_the_prompt_it_was_sent():
    # What the context-linear TTFT budget prices a turn against. Every turn carries
    # it, and it grows, because the transcript does.
    rec, _ = await drive([call("ls"), call("pwd"), SUBMIT],
                         count_prompt_tokens=counter(4))
    lengths = [ev["extras"][PREFILL_TOKENS_EXTRA] for ev in rec.events]
    assert len(lengths) == 3
    assert all(isinstance(n, int) and n > 0 for n in lengths)
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


async def test_the_recorded_length_is_the_count_not_what_the_server_claimed():
    # The reason the count is taken harness-side. Under a slope the recorded length
    # buys TTFT budget, so a server claiming a billion-token prompt would be claiming
    # a budget; what lands in the event is the harness's own figure.
    rec, _ = await drive([SUBMIT], reported_prompt_tokens=10**9,
                         count_prompt_tokens=counter(4))
    counted = await counter(4)(wire_messages(rec.requests[0]["metadata"]["messages"]))
    assert rec.events[0]["extras"][PREFILL_TOKENS_EXTRA] == counted < 10**6


async def test_a_turn_counts_its_transcript_once():
    # The guard counts before the send and the event wants the same number after it.
    # A transcript reaching hundreds of thousands of tokens is seconds of CPU to
    # render and tokenize, so the second ask must be free.
    calls, count = counting_counter(4)
    rec, _ = await drive([call("ls"), call("pwd"), SUBMIT], max_context_len=10**9,
                         count_prompt_tokens=count)
    assert len(rec.events) == 3
    assert len(calls) == 3


async def test_a_turn_with_nothing_to_count_records_no_length():
    # No tokenizer, so there is no harness-side count and the event says so rather
    # than carrying the reply-derived bound: that bound is built partly out of what
    # the server reported, which is the one number a budget may not read. Two turns,
    # so the second has a reported figure behind it and the bound is a real number
    # this could have recorded instead of None.
    rec, _ = await drive([call("ls"), SUBMIT], reported_prompt_tokens=500)
    assert len(rec.events) == 2
    assert [ev["extras"][PREFILL_TOKENS_EXTRA] for ev in rec.events] == [None, None]


def test_the_loop_and_the_slo_agree_on_the_key():
    # The loop may not import hls (it runs in a build sandbox that has no harness),
    # so the key it stamps and the key the SLO reads are two spellings of one string.
    from hls.spec.workload import PREFILL_TOKENS_EXTRA as SLO_KEY

    assert PREFILL_TOKENS_EXTRA == SLO_KEY


async def test_a_turn_hands_its_prompt_length_to_the_driver_too():
    # The count rides on the request as well as on the event, because a turn the
    # drive's scheduled stop cuts never reaches the recorder: its event is the
    # driver's own, built from the envelope, and that is the only place the driver can
    # read the length a context-linear budget needs.
    rec, _ = await drive([SUBMIT], count_prompt_tokens=counter(4))
    assert (rec.requests[0]["metadata"][PREFILL_TOKENS_EXTRA]
            == rec.events[0]["extras"][PREFILL_TOKENS_EXTRA])


async def test_a_turn_with_nothing_to_count_hands_the_driver_nothing():
    # Present and None rather than absent, so a cut turn's event says "counted
    # nothing" exactly the way an answered turn's does, and neither can be budgeted.
    rec, _ = await drive([SUBMIT])
    assert rec.requests[0]["metadata"][PREFILL_TOKENS_EXTRA] is None


# --- the clock the turn is stamped on --------------------------------------

async def test_every_turn_is_stamped_on_the_monotonic_clock():
    # The one clock the harness measures on. Packet arrivals come from
    # time.monotonic() in the transport and the driver re-stamps a turn's send time
    # with max(the framework's, its own), so a wall-clock stamp here is about 1.8e9,
    # wins that max, and makes every time to first token roughly negative fifty-seven
    # years. Under max(r_ttft, r_tpot) a negative TTFT on a short reply reads as 0.0,
    # which is the best score the metric can award.
    t0 = time.monotonic()
    rec, _ = await drive([call("ls"), SUBMIT])
    t1 = time.monotonic()

    assert len(rec.events) == 2
    for ev in rec.events:
        assert t0 <= ev["t_send"] <= ev["t_recv"] <= t1


def test_the_loop_and_the_driver_agree_on_the_clock():
    # The loop cannot call the harness's accessor: this module may not import hls,
    # because the same file runs in a build sandbox with no harness (see the note at
    # the top of it). So the two pick their clock separately, and this is what holds
    # them together -- the values agree, not merely the ordering, so an accessor that
    # moved to a different epoch fails here and names the loop.
    from hls.spec.drivers import harness_now

    assert abs(harness_now() - time.monotonic()) < 0.05
