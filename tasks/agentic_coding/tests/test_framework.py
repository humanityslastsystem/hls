"""The harness adapter: what it puts on a TraceEvent, and the sandbox lifetime it owns."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from hls.core.dataitem import DatasetItem
from hls.core.faults import ArtifactFault, ArtifactProtocolError, HarnessFault
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse

from tasks.agentic_coding import prefix as prefix_format
from tasks.agentic_coding.datasets import (
    PREFIX_DIGEST_FIELD, TASK_NAME_FIELD, TURN_BUDGET_FIELD,
)
from tasks.agentic_coding.mini_swe_loop import PREFILL_TOKENS_EXTRA
from tasks.agentic_coding.framework import (
    PROBLEM_STATEMENT_KEY, REFUSED_EXTRA, REFUSED_KEEP_CHARS, REFUSED_KEEP_TOKENS,
    REPLACED_EXTRA, WORK_DIR_KEY, MiniSweFramework, check_text_matches_tokens,
    split_characters,
)
from tasks.agentic_coding.tests.test_prefix import a_prefix
from tasks.agentic_coding.tests.test_toolcall_conformance import _checkpoint
from tasks.agentic_coding.mini_swe_loop import (
    NO_COMMAND_RUNAWAY, STOP_NO_COMMAND_REPEATED, STOP_SUBMITTED, SUBMIT_SENTINEL,
    WORK_DIR,
)


class Ctx:
    """The fields ``run_trajectory`` reads off a ``HarnessContext``."""

    tenant_id = "t0"
    #: The per-request output cap the driver stamps. The loop reserves it beside the
    #: prompt, so a context guard that read it from the reply instead would show up
    #: here as a stub that never had to carry it.
    max_output_tokens = 4096


def test_the_fake_context_carries_only_what_the_harness_context_has():
    # A field the framework reads that ``HarnessContext`` does not have would raise in every
    # real trajectory, before its first turn.
    from dataclasses import fields

    from hls.spec.drivers import HarnessContext

    faked = {name for name in vars(Ctx) if not name.startswith("_")}
    assert faked <= {f.name for f in fields(HarnessContext)}


class FakeRunner:
    """Records the session lifecycle and every command dispatched through it."""

    def __init__(self, fail_on_open: Exception | None = None):
        self.opened: list[tuple[str, dict]] = []
        self.closed: list[str] = []
        self.calls: list[tuple[str, dict]] = []
        self.fail_on_open = fail_on_open

    async def open_session(self, session_id, **context):
        if self.fail_on_open is not None:
            raise self.fail_on_open
        self.opened.append((session_id, context))

    async def close_session(self, session_id):
        self.closed.append(session_id)

    async def run(self, name, args):
        """One command, shaped the way the real runner shapes its results.

        ``restarted`` is stamped once here rather than written into each return, the
        same way ``SandboxToolRunner.run`` stamps it. Subclasses override ``_result``
        rather than this, so a double that scripts an unusual outcome cannot drop the
        field and leave the loop's reader passing here and failing on a real runner."""
        self.calls.append((name, args))
        return {"restarted": 0, "restart_cause": "",
                **self._result(args["command"])}

    def _result(self, command):
        # Echoes, so the submit sentinel ends a trajectory the way it does on a real
        # shell: the environment reads a submission out of what the command printed.
        if command.startswith("echo "):
            return {"output": command[len("echo "):].strip("'\"") + "\n",
                    "exit_code": 0}
        return {"output": "ok", "exit_code": 0}


def seed(task_name="wal-recovery-ordering", statement="Fix the recovery order.",
         work_dir="/app") -> DatasetItem:
    env = Envelope(request_id="s0", contents=[TextPayload(text=statement)],
                   metadata={TASK_NAME_FIELD: task_name,
                             PROBLEM_STATEMENT_KEY: statement,
                             WORK_DIR_KEY: work_dir})
    return DatasetItem(request_id="s0", turns=(env,), ground_truth_turns=())


def reply(text: str) -> ArtifactResponse:
    return ArtifactResponse(
        output=Envelope(request_id="r", contents=[TextPayload(text=text)], metadata={}))


class Scripted:
    def __init__(self, texts):
        self.texts = list(texts)
        self.seen: list[Envelope] = []

    async def __call__(self, env: Envelope) -> ArtifactResponse:
        self.seen.append(env)
        return reply(self.texts.pop(0) if self.texts else "no block")


async def drive(texts, *, item=None, runner=None, max_turns=10, framework=None):
    events: list = []
    runner = runner or FakeRunner()
    llm = Scripted(texts)

    async def on_event(ev):
        events.append(ev)

    await (framework or MiniSweFramework()).run_trajectory(
        seed_item=item or seed(), llm_call=llm, tool_runner=runner,
        on_event=on_event, ctx=Ctx(), max_turns=max_turns)
    return events, runner, llm


def block(cmd):
    """One reply as this checkpoint emits it: the reasoning it was already inside when
    generation began, then the call in the syntax its own chat template documents."""
    return (f"Thinking.</think><tool_call>bash<arg_key>command</arg_key>"
            f"<arg_value>{cmd}</arg_value></tool_call>")


# --- which turn a trace event describes -------------------------------------

async def test_each_event_carries_the_request_its_own_turn_sent():
    """Turn k's event carries turn k's request, not its neighbour's.

    `TurnRecorder.open` emits the turn it held and holds the new one, so an
    event reaches `emit` one turn after the turn it describes. That is deliberate --
    a turn is not finished until its command has run, and `note_command` amends the
    held turn afterwards -- but it means a consumer reading anything *other* than the
    event, at the moment `emit` fires, sees the following turn's state.

    This pins the graded path against that shift, because the stakes are higher here:
    the TTFT budget is context-linear in `PREFILL_TOKENS_EXTRA`, so a one-turn shift
    would price every request against its neighbour's prompt size and every ratio in
    the run would be wrong.

    Asserted by identity against the envelopes the model actually received, in the
    order it received them, rather than by anything recomputed from the event.
    """
    events, _, llm = await drive([block("ls"), block("pwd"),
                                  block(f"echo {SUBMIT_SENTINEL}")])

    assert len(events) == len(llm.seen), "one event per model call"
    for k, ev in enumerate(events):
        assert ev.turn_index == k
        assert ev.input is llm.seen[k], f"event {k} carries another turn's request"


async def test_each_event_is_priced_on_a_prompt_of_its_own_depth():
    """The same pairing, made distinguishable turn by turn.

    The test above compares object identity, which is exact but says nothing about
    whether the turns differ at all; two identical requests would satisfy it. The
    conversation grows by one exchange per turn, so depth is a per-turn quantity a
    shift by one cannot hide.

    Depth rather than the prefill stamp itself. `PREFILL_TOKENS_EXTRA` is counted by
    the harness-side tokenizer, which loads nothing when no checkpoint snapshot is
    mounted, so it is None throughout a unit test and nothing here can pin its value.
    The stamp is computed from this request, so pinning that the request is the right
    one is what is available -- and it is the same claim, one step earlier.
    """
    events, _, _ = await drive([block("ls"), block("pwd"),
                                block(f"echo {SUBMIT_SENTINEL}")])

    depths = [len(ev.input.metadata["messages"]) for ev in events]
    assert len(set(depths)) == len(depths), f"turns not distinguishable: {depths}"
    assert depths == sorted(depths), f"a growing conversation must grow: {depths}"
    # The first turn is the shallowest of the run: it carries the seed alone. Under a
    # shift by one this event would hold a later, deeper request.
    assert depths[0] == min(depths)


class WordTokenizer:
    """A checkpoint stand-in: a template that is one word per message, one per tool.

    The smallest thing that lets the real counting path run. `_count_prompt_ids` ->
    `chat_prompt_ids` -> `chat_prompt_text` are the shipped functions and are not
    replaced here; what is missing on a host with no snapshot is the checkpoint, and
    this is the checkpoint's two methods and nothing else.

    Whitespace tokens so a conversation's expected count is arithmetic rather than a
    number copied back out of the code under test.
    """

    def apply_chat_template(self, messages, tools=None, tokenize=False,
                            add_generation_prompt=False, **kwargs):
        assert tokenize is False, "chat_prompt_text renders to text"
        assert add_generation_prompt is True, "a prompt ends ready to generate"
        parts = [f"m{i}" for i, _ in enumerate(messages)]
        parts += [f"tool{i}" for i, _ in enumerate(tools or ())]
        if add_generation_prompt:
            parts.append("gen")
        return " ".join(parts)

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False, "the rendered template carries its own"
        return type("Encoding", (), {"input_ids": text.split()})()


@pytest.fixture
def word_tokenizer(monkeypatch):
    """Make the prefill counter able to count, which it cannot without a snapshot."""
    from tasks.agentic_coding import framework as fw

    monkeypatch.setattr(fw, "prompt_tokenizer", lambda: WordTokenizer())


async def test_the_prefill_stamp_is_this_turns_prompt_counted(word_tokenizer):
    """The number the score divides by, pinned to an arithmetic expectation.

    The TTFT budget is context-linear in this stamp, so it is the one field on the
    event whose *value* changes every ratio in the run. Everything else about the
    pairing is pinned above by identity; this pins the quantity.

    Expected count, from the stand-in template: one token per message in the
    conversation sent, one per tool definition, and one closing generation marker.
    Turn k has sent 2k+2 messages, so its stamp is 2k+2 + 1 tool + 1 = 2k+4.
    """
    events, _, llm = await drive([block("ls"), block("pwd"),
                                  block(f"echo {SUBMIT_SENTINEL}")])

    stamps = [ev.extras[PREFILL_TOKENS_EXTRA] for ev in events]
    depths = [len(ev.input.metadata["messages"]) for ev in events]
    assert stamps == [d + 2 for d in depths], (stamps, depths)
    # Not a constant, and rising with the conversation: the property the budget rests
    # on. A stamp that ignored the messages would satisfy neither.
    assert len(set(stamps)) == len(stamps)
    assert stamps == sorted(stamps)


async def test_the_tool_schema_is_counted_into_the_prefill(word_tokenizer):
    """Every turn is sent with the tool definitions, so the count includes them.

    On this checkpoint the schema is several hundred tokens of the system preamble, so a
    count taken without it reads short by a fixed amount on every turn -- which would
    make the budget generous by that much on every request, and the context guard let
    a trajectory past the window.
    """
    from tasks.agentic_coding.framework import _count_prompt_ids

    messages = [{"role": "user", "content": "hi"}]
    counted = _count_prompt_ids(WordTokenizer(), messages, {})
    # one message + one tool + the generation marker
    assert counted == 3
    assert counted > len(messages) + 1, "the tool schema is not in the count"


async def test_the_ttft_budget_scales_with_the_stamp(word_tokenizer):
    """The stamp reaches the budget, and linearly.

    Two real events off one trajectory rather than hand-built ones, so what is
    measured is the number the loop actually stamped.
    """
    from hls.spec.workload import ContextLinearLatencySLO

    events, _, _ = await drive([block("ls"), block("pwd"),
                                block(f"echo {SUBMIT_SENTINEL}")])
    slo = ContextLinearLatencySLO(ttft_ms=1000.0, tpot_ms=20.0,
                                  ttft_ms_per_prefill_token=2.5)

    first, last = events[0], events[-1]
    assert slo.ttft_budget_ms(first) == 1000.0 + 2.5 * first.extras[
        PREFILL_TOKENS_EXTRA]
    # And the deeper turn is given more, by exactly the slope times the difference.
    grew = last.extras[PREFILL_TOKENS_EXTRA] - first.extras[PREFILL_TOKENS_EXTRA]
    assert grew > 0
    assert (slo.ttft_budget_ms(last) - slo.ttft_budget_ms(first)) == 2.5 * grew


# --- session lifetime ------------------------------------------------------

async def test_the_sandbox_opens_before_the_trajectory_and_closes_after():
    _, runner, _ = await drive([block(f"echo {SUBMIT_SENTINEL}")])
    assert [s for s, _ in runner.opened] == ["s0"]
    assert runner.closed == ["s0"]


async def test_the_seeds_metadata_reaches_open_session():
    # The tool runner picks the task's own image and working directory out of this
    # context, so a framework that forwarded only part of it would route every
    # trajectory to the same container.
    _, runner, _ = await drive([block(f"echo {SUBMIT_SENTINEL}")],
                               item=seed(task_name="formal-crypto",
                                         work_dir="/workspace"))
    context = runner.opened[0][1]
    assert context[TASK_NAME_FIELD] == "formal-crypto"
    assert context[WORK_DIR_KEY] == "/workspace"


async def test_the_trajectory_is_told_the_directory_it_starts_in():
    # The opening prompt names the working directory, and it is the one the session's
    # own image was built around rather than a default shared by every task. The
    # directory here is deliberately unlike any word in the prompt: "/task" would have
    # matched the statement's own closing tag whatever the loop was told.
    events, _, _ = await drive([block(f"echo {SUBMIT_SENTINEL}")],
                               item=seed(work_dir="/srv/bench-root"))
    opening = events[0].input.first_of("text").text
    assert "/srv/bench-root" in opening
    assert WORK_DIR not in opening


async def test_the_sandbox_is_released_even_when_the_trajectory_raises():
    class Boom(Scripted):
        async def __call__(self, env):
            raise RuntimeError("artifact died")

    runner = FakeRunner()
    with pytest.raises(RuntimeError, match="artifact died"):
        await MiniSweFramework().run_trajectory(
            seed_item=seed(), llm_call=Boom([]), tool_runner=runner,
            on_event=lambda ev: None, ctx=Ctx(), max_turns=3)
    assert runner.closed == ["s0"]


async def test_a_failed_open_does_not_try_to_close_what_never_opened():
    runner = FakeRunner(fail_on_open=RuntimeError("sandbox setup failed"))
    with pytest.raises(RuntimeError, match="sandbox setup failed"):
        await drive([block("ls")], runner=runner)
    assert runner.closed == []


# --- what reaches the tool -------------------------------------------------

async def test_every_command_carries_its_session():
    _, runner, _ = await drive([block("ls"), block(f"echo {SUBMIT_SENTINEL}")])
    assert runner.calls[0][0] == "bash"
    assert runner.calls[0][1]["session_id"] == "s0"
    assert runner.calls[0][1]["command"] == "ls"


# --- what reaches the trace ------------------------------------------------

async def test_one_event_per_llm_call_carrying_the_served_reply():
    events, _, llm = await drive([block("ls"), block(f"echo {SUBMIT_SENTINEL}")])
    assert len(events) == 2 == len(llm.seen)
    assert [e.turn_index for e in events] == [0, 1]
    assert all(e.session_id == "s0" for e in events)
    assert all(e.response is not None for e in events)


async def test_the_event_input_is_the_envelope_that_was_sent():
    events, _, llm = await drive([block(f"echo {SUBMIT_SENTINEL}")])
    # The gate teacher-forces this envelope, so it must be the one the artifact saw.
    assert events[0].input is llm.seen[0]
    assert events[0].input.metadata["messages"][0]["role"] == "system"


async def test_the_task_statement_seeds_the_conversation():
    _, _, llm = await drive([block(f"echo {SUBMIT_SENTINEL}")],
                            item=seed(statement="Parser drops a trailing comma."))
    assert "Parser drops a trailing comma." in llm.seen[0].metadata["messages"][1]["content"]


async def test_the_command_and_exit_code_land_in_extras():
    events, _, _ = await drive([block("pytest"), block(f"echo {SUBMIT_SENTINEL}")])
    assert events[0].extras["parsed_command"] == "pytest"
    assert events[0].extras["exit_code"] == 0


async def test_the_last_event_carries_the_stop_reason():
    events, _, _ = await drive([block("ls"), block(f"echo {SUBMIT_SENTINEL}")])
    assert events[-1].extras["stop_reason"] == STOP_SUBMITTED


async def test_the_harness_context_is_stamped_on_the_request():
    _, _, llm = await drive([block(f"echo {SUBMIT_SENTINEL}")])
    assert llm.seen[0].metadata["tenant_id"] == "t0"
    assert llm.seen[0].metadata["session_id"] == "s0"


# --- model acceptance ------------------------------------------------------

def test_any_checkpoint_is_accepted():
    from hls.core.model import ModelSpec
    # Nothing is rejected: the task pins its model, so the loop and the checkpoint's
    # tool-call syntax are paired before a run starts.
    for fmt in ("chatml", "gemma", "glm"):
        MiniSweFramework().check_model(ModelSpec(base_checkpoint="x", arch="TX",
                                                 chat_format=fmt))


def test_the_traffic_config_reports_every_knob():
    cfg = MiniSweFramework(output_limit_chars=123, command_timeout_sec=45,
                           max_context_len=8192).traffic_config()
    assert cfg["output_limit_chars"] == 123
    assert cfg["command_timeout_sec"] == 45
    assert cfg["max_context_len"] == 8192
    assert "max_consecutive_format_errors" in cfg


def test_the_traffic_config_reports_no_machinery():
    # The counter's tokenizer pin and its cached callable are state, not settings:
    # fingerprinting them would stale a calibration for a process detail.
    cfg = MiniSweFramework().traffic_config()
    assert not any(k.startswith("_") for k in cfg)


def test_the_continuation_budget_is_reported_as_an_extra_output_cap():
    # The loop stamps a capped turn's continuation at reply_budget, not the output cap,
    # so calibration must know that value is one this workload legitimately produces.
    assert MiniSweFramework(reply_budget=16_384).extra_output_caps() == {16_384}


def test_a_loop_that_never_continues_a_reply_declares_no_extra_cap():
    # Without a reply_budget the loop stamps only the workload cap, so there is no
    # second cap for calibration to accept.
    assert MiniSweFramework(reply_budget=None).extra_output_caps() == set()


# --- the context guard's wiring --------------------------------------------

def test_the_declared_window_reaches_the_loop():
    assert MiniSweFramework(max_context_len=8192).loop().max_context_len == 8192


def test_the_loop_is_handed_a_prompt_counter():
    # Counting is the rule and bounding the reply's figure is the fallback, so the
    # loop must always get a counter, even where it will report that it cannot count.
    assert MiniSweFramework().loop().count_prompt_tokens is not None


async def test_the_counter_says_it_cannot_count_with_no_checkpoint_mounted(monkeypatch):
    monkeypatch.delenv("HLS_WEIGHTS_DIR", raising=False)
    count = MiniSweFramework().prompt_counter()
    assert await count([{"role": "user", "content": "hi"}]) is None


def test_the_output_cap_comes_from_the_harness_not_the_framework():
    # The cap belongs to the tenant's workload, so the driver hands it over per
    # trajectory off the HarnessContext rather than the framework declaring one.
    assert MiniSweFramework().loop().max_output_tokens is None
    assert MiniSweFramework().loop(4096).max_output_tokens == 4096


async def test_the_trajectory_budgets_against_the_caps_the_context_carries():
    seen: list[int | None] = []
    framework = MiniSweFramework(max_context_len=8192)
    original = framework.loop

    def record(max_output_tokens=None, work_dir=WORK_DIR):
        seen.append(max_output_tokens)
        return original(max_output_tokens, work_dir)

    framework.loop = record
    llm = Scripted([block(f"echo {SUBMIT_SENTINEL}")])
    await framework.run_trajectory(seed_item=seed(), llm_call=llm,
                                   tool_runner=FakeRunner(),
                                   on_event=lambda ev: asyncio.sleep(0), ctx=Ctx(),
                                   max_turns=5)
    assert seen == [Ctx.max_output_tokens]


async def test_a_turn_that_spent_its_cap_reasoning_is_not_read_as_a_dead_server():
    # The reply's length and stop label live on the payload, not the envelope
    # metadata, so a framework that copied only the metadata would hand the loop a
    # turn it could only call empty.
    events: list = []

    async def llm(env):
        return ArtifactResponse(output=Envelope(
            request_id="r",
            contents=[TextPayload(text="", tokens=list(range(Ctx.max_output_tokens)),
                                  finish_reason="length")],
            metadata={}))

    async def on_event(ev):
        events.append(ev)

    await MiniSweFramework().run_trajectory(
        seed_item=seed(), llm_call=llm, tool_runner=FakeRunner(),
        on_event=on_event, ctx=Ctx(), max_turns=5)
    # Named for what it was. Every turn here ran the whole output cap out inside the
    # reasoning channel and wrote no call, so the parser was never asked to read one
    # and the ending must not send a reader to look at it.
    assert events[-1].extras["stop_reason"] == STOP_NO_COMMAND_REPEATED
    assert events[-1].extras["no_command_causes"] == {NO_COMMAND_RUNAWAY: 3}


def test_the_chat_template_pin_is_taken_from_the_model():
    # The count is only the one a trusted pool member reports if it renders with the
    # same pin the reference engines render with.
    from hls.core.model import ModelSpec
    framework = MiniSweFramework()
    framework.check_model(ModelSpec(base_checkpoint="x", arch="TX", chat_format="glm",
                                    chat_template_kwargs={"enable_thinking": True}))
    assert framework._chat_template_kwargs == {"enable_thinking": True}


# --- the text must be what the tokens say ----------------------------------

class Detokenizer:
    """A tokenizer stand-in: one token per character, the token being its ordinal.

    A real round trip rather than a canned answer, so a test can hand it the tokens of
    one string and the text of another and get the disagreement the check is looking
    for. Which vocabulary produced the tokens does not matter to the check; that the
    decode is exact does.

    ``apply_chat_template`` is here because a trajectory counts its prompt every turn,
    not only when the context guard is armed: a whole-trajectory test reaches the
    prompt counter as well as the text check. It concatenates, which is enough to
    stand for a template that a per-character tokenizer then measures."""

    @staticmethod
    def encode(text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, tokens, skip_special_tokens=True):
        return "".join(chr(t) for t in tokens)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(str(m.get("content") or "") for m in messages)

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=self.encode(text))


#: One reply as a contestant must serve it: reasoning, prose and the call, all in
#: `content`, with no parser having moved any of it elsewhere.
RAW_REPLY = ("Let me look around.</think>I will list the directory."
             "<tool_call>bash<arg_key>command</arg_key>"
             "<arg_value>ls -la</arg_value></tool_call>")


def test_a_raw_reply_passes():
    # The contract stated positively: no parsers, so content IS the tokens. This is
    # the direction that matters most, because a check that fired here would reject
    # every correct server.
    check_text_matches_tokens(RAW_REPLY, Detokenizer.encode(RAW_REPLY), Detokenizer())


def test_a_reply_whose_tool_call_was_parsed_out_is_refused():
    # The fatal configuration: --tool-call-parser lifts the call into a structured
    # field, leaving content without the only thing the loop reads. Shown failing on
    # purpose, because a check that has never disagreed proves nothing.
    stripped = "Let me look around.</think>I will list the directory."
    with pytest.raises(ArtifactProtocolError, match="--tool-call-parser"):
        check_text_matches_tokens(stripped, Detokenizer.encode(RAW_REPLY),
                                  Detokenizer())


def test_a_reply_whose_reasoning_was_parsed_out_is_also_refused():
    # --reasoning-parser is not fatal to the parse -- the call survives it -- but it
    # is still a server that did not serve what its tokens say, and the contract is
    # one requirement rather than two. Refusing it keeps there being a single shape
    # the artifact and the reference both produce.
    visible_only = ("I will list the directory."
                    "<tool_call>bash<arg_key>command</arg_key>"
                    "<arg_value>ls -la</arg_value></tool_call>")
    with pytest.raises(ArtifactProtocolError, match="--reasoning-parser"):
        check_text_matches_tokens(visible_only, Detokenizer.encode(RAW_REPLY),
                                  Detokenizer())


def test_even_one_missing_character_is_refused():
    # Exact, not approximate. A slack window is just enough room to hide a short call.
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens(RAW_REPLY[:-1], Detokenizer.encode(RAW_REPLY),
                                  Detokenizer())


class _SpecialTokenDetokenizer(Detokenizer):
    """The per-character vocabulary plus one special token, the turn-boundary id
    ``EOT``: it decodes to its own text when special tokens are kept and to nothing
    when they are dropped, the way a real tokenizer treats ``<|observation|>``."""

    EOT = 0x110000  # past chr()'s range, so no character can collide with it

    def decode(self, tokens, skip_special_tokens=True):
        out = []
        for t in tokens:
            if t == self.EOT:
                if not skip_special_tokens:
                    out.append("<|observation|>")
            else:
                out.append(chr(t))
        return "".join(out)


def test_a_reply_ending_on_a_turn_boundary_token_passes_kept_or_dropped():
    # Two honest servers: this task's baseline keeps special tokens in content
    # (skip_special_tokens=False), a stock server drops them. Each is the exact text of
    # the tokens under one detokenization, so each passes.
    tok = _SpecialTokenDetokenizer()
    ids = Detokenizer.encode(RAW_REPLY) + [_SpecialTokenDetokenizer.EOT]
    check_text_matches_tokens(RAW_REPLY + "<|observation|>", ids, tok)
    check_text_matches_tokens(RAW_REPLY, ids, tok)
    # Part of the token's text is neither convention.
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens(RAW_REPLY + "<|obs", ids, tok)
    # A parsed-out call matches neither either.
    with pytest.raises(ArtifactProtocolError, match="--tool-call-parser"):
        check_text_matches_tokens("Let me look around.</think>I will list the directory.",
                                  ids, tok)


class _ByteLevelDetokenizer(Detokenizer):
    """The per-character vocabulary plus raw byte pieces, the way a byte-level
    tokenizer works: a character too rare to have a piece of its own is spelled as
    several pieces, one per byte, and decoding joins the bytes and reads them as UTF-8.
    Bytes that do not finish a character read as the replacement character, which is
    the whole subject of the tests below.

    ``SPLIT_PIECE`` is one piece carrying a space and then the first two bytes of a
    four-byte character -- the shape a merge produces, and the one where the text a
    server withholds is more than replacement characters."""

    SPLIT_PIECE = 0x110001

    def decode(self, tokens, skip_special_tokens=True):
        raw = b"".join(b" \xf0\x9f" if t == self.SPLIT_PIECE
                       else bytes([t]) if t < 0x100 else chr(t).encode("utf-8")
                       for t in tokens)
        return raw.decode("utf-8", "replace")


def test_a_reply_the_cap_cut_mid_character_passes_without_the_bytes_it_never_finished():
    # GLM-5.3-Flash can run to its 32,768-token output cap in the middle of a
    # character, so the reply's last bytes are a character that was never finished.
    # Decoding the tokens at once writes them as two replacement characters; a server
    # that detokenizes as it streams sends neither, and its content is two characters
    # short.
    tok = _ByteLevelDetokenizer()
    ids = Detokenizer.encode(RAW_REPLY) + [0xF0, 0x9F, 0xF0, 0x9F]
    assert tok.decode(ids) == RAW_REPLY + "\ufffd\ufffd"
    check_text_matches_tokens(RAW_REPLY, ids, tok)


def test_a_withheld_tail_is_still_refused_when_a_tool_call_is_missing_from_it():
    # The difference the check accepts is bounded by what a streaming server cannot
    # have sent, not by a number of characters: a reply that ends mid-character and is
    # ALSO missing its call matches neither decode nor the emittable prefix of either.
    tok = _ByteLevelDetokenizer()
    ids = Detokenizer.encode(RAW_REPLY) + [0xF0, 0x9F]
    with pytest.raises(ArtifactProtocolError, match="--tool-call-parser"):
        check_text_matches_tokens("Let me look around.</think>I will list the directory.",
                                  ids, tok)
    # And the hole the rule would have if it only trimmed replacement characters: a
    # server cannot hide a whole reply behind bytes it says it could not finish.
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens("", ids, tok)


def test_the_withheld_tail_is_the_whole_piece_the_server_could_not_finish():
    # A piece can carry text and then the start of a character. A server holds back the
    # piece, not the part of it that failed to decode, so the space goes with the bytes.
    tok = _ByteLevelDetokenizer()
    ids = Detokenizer.encode(RAW_REPLY) + [_ByteLevelDetokenizer.SPLIT_PIECE]
    assert tok.decode(ids) == RAW_REPLY + " \ufffd"
    check_text_matches_tokens(RAW_REPLY, ids, tok)
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens(RAW_REPLY + " ", ids, tok)


def test_undecodable_bytes_in_the_middle_of_a_reply_are_not_a_licence_to_drop_text():
    # Only a reply's tail can be withheld: bytes in the middle are finished by the
    # tokens after them, so a server that streamed past them has already sent them.
    tok = _ByteLevelDetokenizer()
    ids = (Detokenizer.encode("Let me look around.") + [0xF0, 0x9F]
           + Detokenizer.encode("</think>I will list the directory."))
    decoded = tok.decode(ids)
    assert decoded.count("\ufffd") == 1 and not decoded.endswith("\ufffd")
    check_text_matches_tokens(decoded, ids, tok)
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens(decoded.replace("\ufffd", ""), ids, tok)


def test_a_refusal_carries_the_reply_it_refused_for_whoever_reads_the_trace():
    # The refused reply is dropped: its event has no response and nothing else keeps
    # it. Without this the only account of a refusal is the error string, whose 200
    # characters per side can all be identical when the divergence is at the end.
    tok = _ByteLevelDetokenizer()
    ids = Detokenizer.encode(RAW_REPLY) + [0xF0, 0x9F]
    served = "Let me look around.</think>I will list the directory."
    with pytest.raises(ArtifactProtocolError) as exc:
        check_text_matches_tokens(served, ids, tok)
    ev = exc.value.extras[REFUSED_EXTRA]
    assert ev["chars"] == len(served) and ev["tokens"] == len(ids)
    assert ev["decoded_chars"]["stripped"] == len(tok.decode(ids))
    # Where the served text parts from its own tokens, which is the whole diagnosis:
    # here the call begins and the served text has ended.
    assert ev["first_diff"]["stripped"] == len(served)
    assert ev["text_tail"].endswith("directory.")
    # And what the tokens said instead, ending in the character the cap cut in half.
    assert ev["decoded_tail"]["stripped"].endswith("</tool_call>\ufffd")
    assert ev["token_ids_tail"][-2:] == [0xF0, 0x9F]


def test_the_evidence_a_refusal_leaves_is_bounded():
    # A server that fails this check fails it on every reply, and a reply can be
    # 115,000 characters of 32,768 ids. Whole replies on every fault event would be a
    # trace of hundreds of megabytes for a run already scored 0.
    tok = Detokenizer()
    ids = Detokenizer.encode("x" * (REFUSED_KEEP_TOKENS * 20))
    with pytest.raises(ArtifactProtocolError) as exc:
        check_text_matches_tokens("y" * (REFUSED_KEEP_CHARS * 20), ids, tok)
    ev = exc.value.extras[REFUSED_EXTRA]
    assert len(ev["text_tail"]) == REFUSED_KEEP_CHARS
    assert len(ev["decoded_tail"]["stripped"]) == REFUSED_KEEP_CHARS
    assert len(ev["token_ids_head"]) == len(ev["token_ids_tail"]) == REFUSED_KEEP_TOKENS
    # The counts are the full ones, so a reader knows what was left out.
    assert ev["chars"] == REFUSED_KEEP_CHARS * 20 and ev["tokens"] == len(ids)
    assert ev["first_diff"]["stripped"] == 0


def test_an_accepted_reply_leaves_no_evidence_because_there_is_no_fault():
    # The evidence rides on the fault. A reply the cap cut mid-character raises
    # nothing, so nothing is recorded and no trace grows for an honest server.
    tok = _ByteLevelDetokenizer()
    check_text_matches_tokens(RAW_REPLY, Detokenizer.encode(RAW_REPLY) + [0xF0, 0x9F],
                              tok)


def test_the_check_is_unarmed_with_nothing_to_detokenize():
    # Every process without a mounted checkpoint, including this suite by default.
    check_text_matches_tokens("anything", [1, 2, 3], None)
    check_text_matches_tokens("anything", None, Detokenizer())


def test_a_detokenizer_that_raises_does_not_fail_the_run():
    # A degraded check is better than a run that dies because a decode threw.
    class Broken:
        def decode(self, tokens, skip_special_tokens=True):
            raise RuntimeError("no vocab")

    check_text_matches_tokens("x", [1, 2, 3], Broken())


# --- characters whose bytes span tokens --------------------------------------

class _SplitDetokenizer(Detokenizer):
    """A byte-level vocabulary whose pieces cut characters, plus a special token.

    An id below 0x100 is that one byte, ``SPACE_APPROX`` is a space and the first two
    bytes of "\u2248" (the cut GLM-5.3-Flash's own vocabulary makes, ids 21056 and 230), and
    any other id is the character with that code point, whole. ``EOT`` is the
    turn-boundary token, kept or dropped as ``_SpecialTokenDetokenizer``'s is. Decoding
    joins the bytes and reads them as UTF-8, writing a replacement character for bytes
    that do not read, which is what a byte-level tokenizer does."""

    EOT = 0x110000
    SPACE_APPROX = 0x110001

    def piece(self, t, skip_special_tokens=True) -> bytes:
        if t == self.EOT:
            return b"" if skip_special_tokens else b"<|observation|>"
        if t == self.SPACE_APPROX:
            return b" \xe2\x89"
        return bytes([t]) if t < 0x100 else chr(t).encode("utf-8")

    def decode(self, tokens, skip_special_tokens=True):
        return b"".join(self.piece(t, skip_special_tokens)
                        for t in tokens).decode("utf-8", "replace")


#: A reply with two characters its tokens split, "U+2248" (a space and two bytes, then the
#: last byte) and "U+1F600" (one byte a token), and one they do not, "U+4E2D", a token of its
#: own. It ends on the turn-boundary token.
SPLIT_REPLY_IDS = (
    Detokenizer.encode("Let me look around. About") + [_SplitDetokenizer.SPACE_APPROX, 0x88]
    + Detokenizer.encode(" 3 ") + [0xF0, 0x9F, 0x98, 0x80] + [ord("\u4e2d")]
    + Detokenizer.encode("</think>I will list the directory.<tool_call>bash"
                         "<arg_key>command</arg_key><arg_value>ls -la</arg_value>"
                         "</tool_call>")
    + [_SplitDetokenizer.EOT])


def decoded_by_chunks(tok, ids, size, skip_special_tokens=False) -> str:
    """What a server sends that decodes each chunk of ``size`` ids on its own."""
    return "".join(tok.decode(ids[i:i + size], skip_special_tokens=skip_special_tokens)
                   for i in range(0, len(ids), size))


def redecoded(tok, ids, skip_special_tokens=False) -> str:
    """What a server sends that re-decodes every id so far and sends what is new."""
    sent = ""
    for k in range(1, len(ids) + 1):
        text = tok.decode(ids[:k], skip_special_tokens=skip_special_tokens)
        sent += text[len(tok.decode(ids[:k - 1],
                                    skip_special_tokens=skip_special_tokens)):]
    return sent


def test_the_split_characters_are_the_ones_the_tokens_cut():
    # What the rule rests on, pinned directly: the two characters the tokens cut, each
    # with its length in bytes, and not the one a token carries whole.
    tok = _SplitDetokenizer()
    decoded = tok.decode(SPLIT_REPLY_IDS, skip_special_tokens=False)
    split = split_characters(SPLIT_REPLY_IDS, tok, decoded, False)
    assert {decoded[i]: n for i, n in split.items()} == {"\u2248": 3, "\U0001f600": 4}


def test_a_reply_decoded_chunk_by_chunk_passes():
    # Each chunk's own ids decoded, so a character cut between chunks arrives as a
    # replacement character for each part. Every chunk size moves where the cuts land,
    # and all of them are honest.
    tok = _SplitDetokenizer()
    for skip in (True, False):
        decoded = tok.decode(SPLIT_REPLY_IDS, skip_special_tokens=skip)
        for size in (1, 2, 3, 4):
            served = decoded_by_chunks(tok, SPLIT_REPLY_IDS, size, skip)
            assert served != decoded, size
            added = check_text_matches_tokens(served, SPLIT_REPLY_IDS, tok)
            assert added == served.count("\ufffd") - decoded.count("\ufffd") > 0


def test_a_reply_re_decoded_as_it_grew_passes():
    # Every id so far re-decoded and only the new text sent, which sends
    # one replacement character where a cut character begins and never the character.
    tok = _SplitDetokenizer()
    for skip in (True, False):
        decoded = tok.decode(SPLIT_REPLY_IDS, skip_special_tokens=skip)
        served = redecoded(tok, SPLIT_REPLY_IDS, skip)
        assert served == decoded.replace("\u2248", "\ufffd").replace("\U0001f600", "\ufffd")
        assert check_text_matches_tokens(served, SPLIT_REPLY_IDS, tok) == 2


def test_the_exact_decode_passes_and_adds_nothing():
    tok = _SplitDetokenizer()
    for skip in (True, False):
        decoded = tok.decode(SPLIT_REPLY_IDS, skip_special_tokens=skip)
        assert check_text_matches_tokens(decoded, SPLIT_REPLY_IDS, tok) == 0


def test_a_split_character_takes_at_most_its_bytes():
    # Four bytes can decode to four replacement characters and no more.
    tok = _SplitDetokenizer()
    decoded = tok.decode(SPLIT_REPLY_IDS, skip_special_tokens=False)
    assert check_text_matches_tokens(decoded.replace("\u2248", "\ufffd" * 3).replace("\U0001f600", "\ufffd" * 4),
                                     SPLIT_REPLY_IDS, tok) == 7
    for garbled in (decoded.replace("\u2248", "\ufffd" * 4), decoded.replace("\U0001f600", "\ufffd" * 5)):
        with pytest.raises(ArtifactProtocolError):
            check_text_matches_tokens(garbled, SPLIT_REPLY_IDS, tok)


def test_split_characters_are_no_licence_for_anything_else():
    # Each of these differs from an accepted rendering by one thing the rule does not
    # cover, and each is refused.
    tok = _SplitDetokenizer()
    served = decoded_by_chunks(tok, SPLIT_REPLY_IDS, 1)
    call = served.index("<tool_call>")
    refused = {
        # What the check exists to catch.
        "call removed": served[:call] + "<|observation|>",
        "one character added": served + "x",
        "one character missing": served.replace("Let", "Lt"),
        # A character a token carries whole, or a plain ASCII one.
        "unsplit character replaced": served.replace("\u4e2d", "\ufffd"),
        "ASCII character replaced": served.replace("Let", "\ufffdet"),
        # Markup is ASCII, so it is never split.
        "call markup replaced": served.replace("<tool_call>", "\ufffdtool_call>"),
    }
    accepted = []
    for what, text in refused.items():
        try:
            check_text_matches_tokens(text, SPLIT_REPLY_IDS, tok)
        except ArtifactProtocolError:
            continue
        accepted.append(what)
    assert accepted == []
    with pytest.raises(ArtifactProtocolError, match="--tool-call-parser"):
        check_text_matches_tokens(refused["call removed"], SPLIT_REPLY_IDS, tok)


def test_special_tokens_are_kept_or_dropped_the_same_way_throughout():
    # Two turn-boundary tokens, so a reply can keep one and drop the other. Neither
    # treatment reads that way, and replacing split characters does not make it one.
    tok = _SplitDetokenizer()
    ids = SPLIT_REPLY_IDS + [_SplitDetokenizer.EOT]
    kept = decoded_by_chunks(tok, ids, 1, skip_special_tokens=False)
    assert kept.endswith("<|observation|><|observation|>")
    check_text_matches_tokens(kept, ids, tok)
    check_text_matches_tokens(decoded_by_chunks(tok, ids, 1, skip_special_tokens=True),
                              ids, tok)
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens(kept[:-len("<|observation|>")], ids, tok)


def test_a_reply_cut_mid_character_combines_with_split_characters():
    # The cap cut the reply two bytes into a character, and those two bytes are two
    # tokens. A server decoding token by token writes each as a replacement character;
    # one that holds unfinished bytes back sends neither. Both are accepted with the
    # split characters before them replaced, and one more replacement character than
    # the cut bytes can make is not.
    tok = _SplitDetokenizer()
    ids = SPLIT_REPLY_IDS[:-1] + [0xF0, 0x9F]
    decoded = tok.decode(ids)
    assert decoded.endswith("</tool_call>\ufffd")
    by_token = decoded_by_chunks(tok, ids, 1)
    assert by_token.endswith("</tool_call>\ufffd\ufffd")
    assert check_text_matches_tokens(by_token, ids, tok) == by_token.count("\ufffd") - 1
    held_back = redecoded(tok, ids)[:-1]
    assert held_back.endswith("</tool_call>")
    assert check_text_matches_tokens(held_back, ids, tok) == 2
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens(by_token + "\ufffd", ids, tok)


async def test_the_count_of_replacement_characters_reaches_the_event(monkeypatch):
    # So the report can say how many replies were accepted by this rule and no other.
    import tasks.agentic_coding.framework as fw

    monkeypatch.setattr(fw, "prompt_tokenizer", _SplitDetokenizer)
    tok = _SplitDetokenizer()
    first = (Detokenizer.encode("Thinking about") + [_SplitDetokenizer.SPACE_APPROX, 0x88]
             + Detokenizer.encode(" it.</think><tool_call>bash<arg_key>command</arg_key>"
                                  "<arg_value>ls</arg_value></tool_call>"))
    second = Detokenizer.encode(block(f"echo {SUBMIT_SENTINEL}"))

    class ByToken(Scripted):
        async def __call__(self, env):
            ids = self.texts.pop(0)
            return ArtifactResponse(output=Envelope(
                request_id="r",
                contents=[TextPayload(text=decoded_by_chunks(tok, ids, 1), tokens=ids,
                                      finish_reason="stop")],
                metadata={}))

    events: list = []

    async def on_event(ev):
        events.append(ev)

    await MiniSweFramework().run_trajectory(
        seed_item=seed(), llm_call=ByToken([first, second]), tool_runner=FakeRunner(),
        on_event=on_event, ctx=Ctx(), max_turns=5)
    assert [ev.extras[REPLACED_EXTRA] for ev in events] == [2, 0]


async def test_the_count_is_absent_where_the_check_is_unarmed():
    # A missing key says nothing was checked; a zero says the text was exact.
    events, _, _ = await drive([block(f"echo {SUBMIT_SENTINEL}")])
    assert REPLACED_EXTRA not in events[0].extras


# --- the same rule on GLM-5.3-Flash's own tokenizer --------------------------

needs_checkpoint = pytest.mark.skipif(
    _checkpoint() is None, reason="needs the GLM-5.3-Flash snapshot for its tokenizer")


@pytest.fixture(scope="module")
def glm_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_checkpoint(), local_files_only=True)


#: A reply in the checkpoint's own format with characters its vocabulary cuts: "U+2248" is
#: ids 21056 and 230, a space and two bytes then one, and "U+2717" is cut the same way.
GLM_SPLIT_REPLY = ("The tempo \u2248 79.9 BPM and the check \u2717 failed.</think>I will look."
                   "<tool_call>bash<arg_key>command</arg_key><arg_value>ls -la"
                   "</arg_value></tool_call>")


@needs_checkpoint
def test_the_real_tokenizer_cuts_the_characters_this_rule_is_about(glm_tokenizer):
    # If a vocabulary change stopped cutting them, the tests below would pass for
    # nothing, so the cut is asserted first.
    tok = glm_tokenizer
    ids = tok.encode(GLM_SPLIT_REPLY, add_special_tokens=False)
    assert tok.decode(ids, skip_special_tokens=False) == GLM_SPLIT_REPLY
    split = split_characters(ids, tok, GLM_SPLIT_REPLY, False)
    assert {GLM_SPLIT_REPLY[i]: n for i, n in split.items()} == {"\u2248": 3, "\u2717": 3}
    assert [21056, 230] == ids[ids.index(21056):ids.index(21056) + 2]


@needs_checkpoint
def test_the_real_tokenizer_accepts_both_servers_renderings(glm_tokenizer):
    tok = glm_tokenizer
    ids = tok.encode(GLM_SPLIT_REPLY, add_special_tokens=False)
    for size in (1, 2, 3):
        served = decoded_by_chunks(tok, ids, size)
        if served != GLM_SPLIT_REPLY:
            assert check_text_matches_tokens(served, ids, tok) > 0
    by_token = decoded_by_chunks(tok, ids, 1)
    assert by_token == GLM_SPLIT_REPLY.replace("\u2248", "\ufffd\ufffd").replace("\u2717", "\ufffd\ufffd")
    served = redecoded(tok, ids)
    assert served == GLM_SPLIT_REPLY.replace("\u2248", "\ufffd").replace("\u2717", "\ufffd")
    assert check_text_matches_tokens(served, ids, tok) == 2
    assert check_text_matches_tokens(GLM_SPLIT_REPLY, ids, tok) == 0


@needs_checkpoint
def test_the_real_tokenizer_refuses_everything_else(glm_tokenizer):
    tok = glm_tokenizer
    ids = tok.encode(GLM_SPLIT_REPLY, add_special_tokens=False)
    served = decoded_by_chunks(tok, ids, 1)
    for text in (served[:served.index("<tool_call>")], served + ".",
                 served.replace("tempo", "temp\ufffd"), served.replace("\ufffd\ufffd", "\ufffd\ufffd\ufffd\ufffd", 1),
                 served.replace("<arg_key>", "<arg_key\ufffd")):
        with pytest.raises(ArtifactProtocolError):
            check_text_matches_tokens(text, ids, tok)


#: The first 64 ids of a real reply, and the text its server sent for them. The server
#: re-decoded every id so far and sent what was new, so "U+2248" (ids 21056 and 230)
#: arrived as one replacement character. The text is the first 200 characters served.
REFUSED_HEAD_IDS = [
    3862, 525, 3040, 1760, 3419, 27655, 518, 220, 15, 13, 99916, 22, 11, 220, 16, 13, 15,
    99241, 11, 220, 16, 13, 22, 101411, 11, 220, 17, 13, 20, 98729, 1959, 25862, 3968, 15,
    13, 110448, 21, 274, 11392, 9378, 284, 220, 15, 13, 110448, 21, 274, 11392, 23167,
    21056, 230, 220, 102626, 13, 24, 87488, 13, 10712, 8470, 518, 220, 18, 13, 99951]
REFUSED_HEAD_TEXT = (
    "There are four count-in clicks at 0.267, 1.022, 1.765, 2.519 \u2014 spacing ~0.7506 s "
    "\u2192 beat = 0.7506 s \u2192 tempo \ufffd 79.9 BPM. Music starts at 3.274 = click 5 (i.e., the "
    "downbeat of measure 1).\n\nSo beat = 0.2")


@needs_checkpoint
def test_a_real_reply_with_a_split_character_passes(glm_tokenizer):
    # The served text differs from the decode by this one character, written as a
    # replacement character, so it is accepted; the same text with the character
    # removed instead of replaced is refused.
    tok = glm_tokenizer
    decoded = tok.decode(REFUSED_HEAD_IDS, skip_special_tokens=False)
    served = REFUSED_HEAD_TEXT[:len(decoded)]
    assert served == redecoded(tok, REFUSED_HEAD_IDS)
    assert [(i, c) for i, c in enumerate(decoded) if c != served[i]] == [(107, "\u2248")]
    assert check_text_matches_tokens(served, REFUSED_HEAD_IDS, tok) == 1
    with pytest.raises(ArtifactProtocolError):
        check_text_matches_tokens(served.replace("\ufffd", ""), REFUSED_HEAD_IDS, tok)


# --- what the calibration fingerprint records --------------------------------

def test_the_sandbox_knobs_are_not_in_the_fingerprint():
    """The reference does not fault on Terminal-Bench, so neither constant can change
    which sessions exist or how many turns each serves, and an entry that cannot change
    the traffic invalidates measurements for nothing."""
    framework = MiniSweFramework(max_sandbox_restarts=5,
                                 max_consecutive_sandbox_faults=3)

    assert "max_sandbox_restarts" not in framework.traffic_config()
    assert "max_consecutive_sandbox_faults" not in framework.traffic_config()


def test_the_prefix_directory_is_not_in_the_fingerprint():
    """A host staging path, and paths are not traffic.

    A prefix is loaded by task name and refused unless it hashes to the digest its row
    declares, so a directory either holds the same conversations -- identical traffic --
    or fails the run loudly. It cannot quietly serve different traffic, which is the
    only thing a fingerprint entry is for.

    In, it would mean a calibration taken on one cluster could not be reused on another
    for a reason with nothing to do with what was served."""
    framework = MiniSweFramework(prefix_dir="/data/one/place")

    assert "prefix_dir" not in framework.traffic_config()
    # And two hosts staging the same conversations agree, which is the point.
    assert (framework.traffic_config()
            == MiniSweFramework(prefix_dir="/scratch/another/place").traffic_config())


def test_the_checkpoint_stays_in_the_fingerprint():
    """The control for the rule, and the reason prefix_dir is not just "paths are out".

    A conversation is the writing of the weights that produced it, so a prefix cut
    against another model is off-distribution rather than merely stale -- and nothing
    downstream catches it, because the digest pins the file and not the model. Changing
    it changes the traffic without failing, so it is in."""
    assert "checkpoint" in MiniSweFramework(checkpoint="some/model").traffic_config()


def test_the_knobs_are_still_live():
    """The control. Leaving the fingerprint is not being turned off: the runner still
    gets its restart budget and the loop still gets its fault run."""
    framework = MiniSweFramework(max_sandbox_restarts=5,
                                 max_consecutive_sandbox_faults=3)

    assert framework.max_sandbox_restarts == 5
    assert framework.max_consecutive_sandbox_faults == 3
    assert framework.loop().max_consecutive_sandbox_faults == 3


def test_the_settings_that_do_shape_traffic_are_still_recorded():
    """The other control, so the exemption stays short rather than becoming a
    habit. These change the prompts or the turn counts and must still stale a stored
    calibration when they move."""
    config = MiniSweFramework(max_context_len=1234).traffic_config()

    assert config["max_context_len"] == 1234
    for key in ("output_limit_chars", "max_consecutive_format_errors",
                "command_timeout_sec"):
        assert key in config, key


# --- whether the prompt leaves a thought open -------------------------------

class OpensAThought(Detokenizer):
    """A template that turns thinking on: the generation prompt ends inside a thought,
    which is what this checkpoint's own template does."""

    def apply_chat_template(self, messages, **kwargs):
        return super().apply_chat_template(messages, **kwargs) + "<think>"


def test_the_framework_measures_the_open_thought_from_the_render(monkeypatch):
    # Never inferred from a reply: a reply that never closed a thought and one an
    # engine already stripped are the same string. The prompt is the harness's own
    # render, so what it ends with is a fact to read.
    import tasks.agentic_coding.framework as fw

    monkeypatch.setattr(fw, "prompt_tokenizer", OpensAThought)
    framework = MiniSweFramework()
    asyncio.run(framework.warmup())
    assert framework.loop().thinking_opened is True

    monkeypatch.setattr(fw, "prompt_tokenizer", Detokenizer)
    framework = MiniSweFramework()
    asyncio.run(framework.warmup())
    assert framework.loop().thinking_opened is False


def test_a_framework_with_no_checkpoint_measures_no_open_thought(monkeypatch):
    # Nothing to render with, so nothing is claimed. False is the reading that leaves
    # an already-stripped reply alone, which is the safe way to be wrong.
    import tasks.agentic_coding.framework as fw

    monkeypatch.setattr(fw, "prompt_tokenizer", lambda: None)
    framework = MiniSweFramework()
    asyncio.run(framework.warmup())
    assert framework.loop().thinking_opened is False


def test_a_mounted_checkpoint_whose_tokenizer_does_not_load_ends_the_drive_as_the_harnesss_fault(
        monkeypatch, tmp_path):
    # The driver warms the framework up before it anchors its clock. Driven on with no
    # tokenizer, no request carries the prompt count its latency budget is computed
    # from, so the run scores zero as the artifact's, and a reply's text is not checked
    # against its tokens.
    import tasks.agentic_coding.framework as fw

    monkeypatch.setenv("HLS_WEIGHTS_DIR", str(tmp_path))
    monkeypatch.setattr(fw, "_TOKENIZERS", {})
    monkeypatch.setattr(fw, "_LOAD_ERRORS", {})
    framework = MiniSweFramework()
    with pytest.raises(HarnessFault) as raised:
        asyncio.run(framework.warmup())
    assert not isinstance(raised.value, ArtifactFault)
    assert f"no tokenizer from the mounted checkpoint {tmp_path}: " in str(raised.value)
    # The loader's own error is in the message, not a bare None.
    assert fw._LOAD_ERRORS[str(tmp_path)] in str(raised.value)

    # A tokenizer that loads from the same mount starts the drive.
    monkeypatch.setattr(fw, "_TOKENIZERS", {str(tmp_path): Detokenizer()})
    asyncio.run(framework.warmup())


def test_the_measurement_uses_the_chat_template_pin(monkeypatch):
    # The pin is what turns thinking on or off for a checkpoint that takes the
    # argument, so a measurement taken without it would describe a different prompt
    # from the one the artifact is served.
    import tasks.agentic_coding.framework as fw

    seen: list[dict] = []

    class Recording(Detokenizer):
        def apply_chat_template(self, messages, **kwargs):
            seen.append(kwargs)
            return super().apply_chat_template(messages, **kwargs)

    monkeypatch.setattr(fw, "prompt_tokenizer", Recording)
    framework = MiniSweFramework()
    framework.check_model(SimpleNamespace(chat_template_kwargs={"enable_thinking": True}))
    asyncio.run(framework.warmup())

    assert any(kw.get("enable_thinking") is True for kw in seen)


# --- the same check on a reply the framework never gets to read -------------

def test_a_cut_reply_gets_the_same_check(monkeypatch):
    import tasks.agentic_coding.framework as fw

    # The driver asks on the framework's behalf for a turn the scheduled stop cut,
    # because that call never returns and the reply would otherwise be the one reply
    # of a drive nobody checks. Padding the token array is what it buys: every id
    # carries an arrival time, and more of them lowers the mean inter-token latency
    # the score is computed from.
    monkeypatch.setattr(fw, "prompt_tokenizer", Detokenizer)
    framework = MiniSweFramework()
    padded = Detokenizer.encode(RAW_REPLY + "xxxxxxxxxx")

    assert framework.check_reply(RAW_REPLY, Detokenizer.encode(RAW_REPLY)) is None
    refused = framework.check_reply(RAW_REPLY, padded)
    assert refused is not None and "token_ids" in refused


def test_a_cut_reply_check_reports_rather_than_raises(monkeypatch):
    import tasks.agentic_coding.framework as fw

    # Its caller is the driver's cancellation handler, which has a cut to finish
    # recording. A raise there would unwind the recording instead of failing the
    # request.
    monkeypatch.setattr(fw, "prompt_tokenizer", Detokenizer)

    assert isinstance(MiniSweFramework().check_reply("x", Detokenizer.encode("yy")),
                      str)


def test_a_cut_reply_with_nothing_to_check_is_not_refused(monkeypatch):
    import tasks.agentic_coding.framework as fw

    # Unarmed exactly where the answered path is unarmed: no mounted checkpoint, or a
    # reply that carried no ids at all.
    monkeypatch.setattr(fw, "prompt_tokenizer", lambda: None)
    assert MiniSweFramework().check_reply("anything", [1, 2, 3]) is None
    monkeypatch.setattr(fw, "prompt_tokenizer", Detokenizer)
    assert MiniSweFramework().check_reply("", None) is None


async def test_a_stripped_reply_fails_the_trajectory_rather_than_starving_it(monkeypatch):
    # End to end, and this is the point of the check. Without it a server running a
    # tool-call parser serves replies with the call lifted out of the text; the loop
    # finds nothing to run, repairs until the budget is gone, and the record reports a
    # model that would not act. With it the run fails naming the cause.
    import tasks.agentic_coding.framework as fw

    served = block("ls -la")

    class Stripping(Scripted):
        async def __call__(self, env):
            # Tokens for the whole reply, text with the call removed.
            return ArtifactResponse(output=Envelope(
                request_id="r",
                contents=[TextPayload(text="Thinking.",
                                      tokens=Detokenizer.encode(served),
                                      finish_reason="stop")],
                metadata={}))

    monkeypatch.setattr(fw, "prompt_tokenizer", Detokenizer)
    with pytest.raises(ArtifactProtocolError, match="--tool-call-parser"):
        await MiniSweFramework().run_trajectory(
            seed_item=seed(), llm_call=Stripping([]), tool_runner=FakeRunner(),
            on_event=lambda ev: asyncio.sleep(0), ctx=Ctx(), max_turns=5)


async def test_a_raw_reply_runs_normally_through_the_same_path(monkeypatch):
    # The other direction end to end, so the check is pinned as not firing on the
    # server we want. Every turn of a whole trajectory goes through it.
    import tasks.agentic_coding.framework as fw

    monkeypatch.setattr(fw, "prompt_tokenizer", Detokenizer)

    class Faithful(Scripted):
        async def __call__(self, env):
            text = self.texts.pop(0) if self.texts else "no call"
            return ArtifactResponse(output=Envelope(
                request_id="r",
                contents=[TextPayload(text=text, tokens=Detokenizer.encode(text),
                                      finish_reason="stop")],
                metadata={}))

    runner = FakeRunner()
    await MiniSweFramework().run_trajectory(
        seed_item=seed(),
        llm_call=Faithful([block("ls -la"), block(f"echo {SUBMIT_SENTINEL}")]),
        tool_runner=runner, on_event=lambda ev: asyncio.sleep(0), ctx=Ctx(),
        max_turns=5)
    assert [args["command"] for _, args in runner.calls] == [
        "ls -la", f"echo {SUBMIT_SENTINEL}"]


# --- warm starts -----------------------------------------------------------

def warm_seed(task_name="wal-recovery-ordering", *, digest="", turn_budget=0):
    """A seed whose row declares a warm start."""
    item = seed(task_name=task_name)
    env = item.turns[0]
    return DatasetItem(
        request_id=item.request_id,
        turns=(env.model_copy(update={"metadata": {**env.metadata,
                                                   PREFIX_DIGEST_FIELD: digest,
                                                   TURN_BUDGET_FIELD: turn_budget}}),),
        ground_truth_turns=())


async def test_a_row_that_declares_no_prefix_starts_cold():
    # The seed carries an empty digest, so nothing is loaded and no prefix directory
    # is needed.
    events, _, llm = await drive([block(f"echo {SUBMIT_SENTINEL}")],
                                 item=warm_seed())
    assert len(llm.seen[0].metadata["messages"]) == 2


async def test_a_declared_prefix_with_nowhere_to_load_it_from_fails_the_trajectory():
    # The framework is configured without a prefix directory but the corpus says this
    # task resumes. Running it cold would serve traffic the calibration never saw, so
    # it fails instead.
    with pytest.raises(prefix_format.PrefixError, match="no prefix_dir"):
        await drive([block("ls")], item=warm_seed(digest="a" * 64))


async def test_a_declared_prefix_is_loaded_and_resumed(tmp_path):
    frozen = a_prefix(task_name="wal-recovery-ordering",
                      statement="Fix the recovery order.")
    prefix_format.write(tmp_path, frozen)
    framework = MiniSweFramework(prefix_dir=str(tmp_path))
    _, _, llm = await drive([block(f"echo {SUBMIT_SENTINEL}")],
                            item=warm_seed(digest=frozen.digest),
                            framework=framework)
    assert llm.seen[0].metadata["messages"][:len(frozen.messages)] == [
        dict(m) for m in frozen.messages]


async def test_a_staged_prefix_that_is_not_the_declared_one_fails_the_trajectory(
        tmp_path):
    # The whole point of putting the digest on the row: a re-cut prefix must not run
    # under a corpus pin that did not move.
    prefix_format.write(tmp_path, a_prefix(task_name="wal-recovery-ordering",
                                           statement="Fix the recovery order."))
    framework = MiniSweFramework(prefix_dir=str(tmp_path))
    with pytest.raises(prefix_format.PrefixError, match="not the one this task"):
        await drive([block("ls")], item=warm_seed(digest="f" * 64),
                    framework=framework)


async def test_a_prefix_is_loaded_before_any_container_is_started(tmp_path):
    # A trajectory that cannot load its prefix must not have provisioned a sandbox
    # first; at 30 GiB a session, a run failing this way would churn through disk.
    runner = FakeRunner()
    with pytest.raises(prefix_format.PrefixError):
        await drive([block("ls")], item=warm_seed(digest="a" * 64), runner=runner)
    assert runner.opened == []


async def test_a_rows_turn_budget_bounds_its_session():
    # The seam a per-depth budget uses.
    events, _, _ = await drive([block("ls")] * 8, item=warm_seed(turn_budget=3),
                               max_turns=10)
    assert len(events) == 3


async def test_a_turn_budget_cannot_lift_a_session_past_the_task_ceiling():
    # The lower of the two, so a mis-set row cannot buy turns the task did not declare.
    events, _, _ = await drive([block("ls")] * 8, item=warm_seed(turn_budget=9),
                               max_turns=2)
    assert len(events) == 2


# --- the sandbox knobs ------------------------------------------------------

def test_the_sandbox_knobs_stay_out_of_the_fingerprint_when_both_are_set():
    from tasks.agentic_coding.framework import MiniSweFramework

    config = MiniSweFramework(max_sandbox_restarts=2,
                              max_consecutive_sandbox_faults=3).traffic_config()
    assert "max_sandbox_restarts" not in config
    assert "max_consecutive_sandbox_faults" not in config


async def test_a_trajectory_the_sandbox_ended_raises_sandbox_gone():
    # The loop ends a trajectory whose container stopped executing commands, and
    # cannot say what that means for the run: it may not import `hls`. The framework
    # says it, in the harness's own vocabulary, raising SandboxGone so the trajectory
    # does not score a truncated session against a server that never touched the exec
    # channel. Same fault the tool runner raises when a container could not be
    # replaced, because the two are one outcome.
    #
    # The stand-in runner is what makes this reachable. The real one sets
    # `sandbox_error` only when its restart budget is zero, and `task.py` runs at
    # five, so the path under test is the restarts-off configuration rather than the
    # shipped one.
    from hls.core.faults import SandboxGone

    class DeadSandbox(FakeRunner):
        def _result(self, command):
            return {"output": "Sandbox error: ReadError: ", "exit_code": 125,
                    "sandbox_error": True}

    runner = DeadSandbox()
    with pytest.raises(SandboxGone) as caught:
        await drive([block("ls")] * 6, runner=runner,
                    framework=MiniSweFramework(max_consecutive_sandbox_faults=2))
    # The loop's own account of the ending survives the translation: the framework
    # renames the fault, it does not write a new one.
    assert "consecutive_faults=2" in str(caught.value)
    # And the container is still released.
    assert runner.closed == ["s0"]


def test_the_framework_hands_the_fault_run_to_the_loop():
    # The framework declares it; the loop enforces it. A value that reached the
    # framework and not the loop would record a setting the run did not have.
    from tasks.agentic_coding.framework import MiniSweFramework

    loop = MiniSweFramework(max_consecutive_sandbox_faults=4).loop()
    assert loop.max_consecutive_sandbox_faults == 4


def test_the_task_turns_both_on_and_agrees_with_its_runner():
    # One constant, two consumers: the framework holds it and the runner enforces it.
    # Two sources of truth would let one describe a restart budget the sandbox did not
    # have.
    from tasks.agentic_coding import task as t

    assert t.MAX_SANDBOX_RESTARTS > 0
    assert t.MAX_CONSECUTIVE_SANDBOX_FAULTS > 0
    assert t.TOOL_RUNNER.max_sandbox_restarts == t.MAX_SANDBOX_RESTARTS
    assert t.AGENT.framework.max_sandbox_restarts == t.MAX_SANDBOX_RESTARTS
    assert (t.AGENT.framework.max_consecutive_sandbox_faults
            == t.MAX_CONSECUTIVE_SANDBOX_FAULTS)
