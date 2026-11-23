"""Task-specific ``AgentFramework`` for agentic_coding (mini-SWE-agent on Terminal-Bench).

The loop itself lives in ``mini_swe_loop`` so the build-sandbox self-test can run it
without the harness. This adapter feeds it the harness's envelopes and turns its
per-call records into ``TraceEvent``s: one event per LLM call, the command and its
exit code in ``extras``, and the trajectory's stop reason stamped on the last event
for the report-only diagnostics to group on.

The Sec.9 trust invariant is what puts this loop on the harness's side. The artifact
serves one reply per HTTP call and never runs a command; the bash tool is dispatched
through the harness's own ``ToolRunner``, so the wall time of every command counts
against the trajectory makespan and a contestant cannot buy latency by pretending a
test suite ran instantly.
"""
from __future__ import annotations

import asyncio
import logging
import os
from bisect import bisect_left
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import ClassVar

from hls.core.faults import ArtifactProtocolError, HarnessFault, SandboxGone
from hls.core.model import ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent
from hls.spec.agent import AgentFramework
from hls.spec.tools import ToolRunner

from tasks.agentic_coding.mini_swe_loop import (  # noqa: F401  re-exported for task.py and tests
    BASH_TOOL, BASH_TOOL_NAME, DEFAULT_COMMAND_TIMEOUT_SEC,
    DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS, DEFAULT_MAX_CONSECUTIVE_SANDBOX_FAULTS,
    DEFAULT_OUTPUT_LIMIT_CHARS, SUBMIT_SENTINEL,
    THINK_OPEN, WORK_DIR, MiniSweLoop, SandboxFault,
)
from tasks.agentic_coding import prefix as prefix_format
from tasks.agentic_coding.datasets import (
    INSTRUCTION_FIELD, PREFIX_DIGEST_FIELD, TASK_NAME_FIELD, TURN_BUDGET_FIELD,
    WORKDIR_FIELD,
)


log = logging.getLogger(__name__)

#: Metadata key the dataset hangs the task statement on. The seed envelope's text
#: payload carries the same string; the key is read first so a row that renders its
#: prompt differently still reaches the loop with the raw statement.
PROBLEM_STATEMENT_KEY = INSTRUCTION_FIELD

#: Metadata key naming the directory this trajectory's container starts in. Read off
#: the seed because it is a property of that task's image.
WORK_DIR_KEY = WORKDIR_FIELD

#: The checkpoint's tokenizer, kept per weights directory, and ``None`` for a directory
#: whose tokenizer would not load. The driver runs every trajectory of a drive in one
#: process and loading a tokenizer costs about a second, so it is loaded once -- and a
#: failure is remembered too, so a process does not retry it on every call.
#: Keyed rather than a single slot only so a process that somehow saw two
#: snapshots could not count with the wrong one.
_TOKENIZERS: dict[str, object | None] = {}

#: Why a directory's tokenizer would not load, for the fault a drive ends on.
_LOAD_ERRORS: dict[str, str] = {}


def _mounted_weights() -> str | None:
    """The mounted checkpoint snapshot, or ``None`` when this process has none.

    Mount-only, deliberately. The driver role mounts the task's weights at
    ``HLS_WEIGHTS_DIR`` (``hls.provision.mounts``), so a graded drive and every
    calibration pool member have one. Falling back to the checkpoint name the way the
    reference engines do would put a Hub reach inside a sealed container or a unit
    test, and the prompt counter has its own fallback for having no tokenizer."""
    wdir = os.environ.get("HLS_WEIGHTS_DIR")
    return wdir if wdir and Path(wdir).is_dir() else None


def _load_tokenizer(wdir: str):
    """Load the mounted checkpoint's tokenizer, or report that it would not load.

    A load that raises is answered as ``None`` and its error is kept. A drive does not
    start on it (``MiniSweFramework.warmup``): the SLO's budget is a function of the
    prompt count, so a request without one has no ratio, and the check of a reply's
    text against its tokens needs the tokenizer too."""
    try:
        from transformers import AutoTokenizer  # type: ignore[import-untyped]
        return AutoTokenizer.from_pretrained(wdir, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 -- any load failure is the same fault
        _LOAD_ERRORS[wdir] = repr(exc)
        log.error("agentic_coding: no tokenizer from the mounted checkpoint %s (%r)",
                  wdir, exc)
        return None


def prompt_tokenizer():
    """The checkpoint's tokenizer, or ``None`` where none is mounted or none loads.

    Loaded exactly as the reference engines load theirs (``HFEngine._load``), because
    the count is only worth having if it is the count they would get."""
    wdir = _mounted_weights()
    if wdir is None:
        return None
    if wdir not in _TOKENIZERS:
        _TOKENIZERS[wdir] = _load_tokenizer(wdir)
    return _TOKENIZERS[wdir]


def _count_prompt_ids(tokenizer, messages: list[dict], template_kwargs: dict,
                      continue_final_message: bool = False) -> int:
    """How many tokens this conversation's prompt renders to.

    Goes through ``hls.reference.engines``'s own renderer rather than repeating it:
    that is the single definition of how messages become the prompt a serving stack is
    fed, and a count taken any other way would drift from the one a trusted pool member
    reports for the same turn. Imported here rather than at module import so the
    framework still loads on a host without the reference's dependencies.

    Rendered with the tool definitions, because every turn is sent with them. The chat
    template folds them into the system preamble -- several hundred tokens of schema on
    this checkpoint -- so a count taken without them would read short by a fixed amount
    on every turn, and the guard it feeds would let a trajectory past the window."""
    from hls.reference.engines import chat_prompt_ids
    return len(chat_prompt_ids(tokenizer, messages, [BASH_TOOL], template_kwargs,
                               continue_final_message))


#: What a decoder writes where bytes do not form a character. A streaming server never
#: emits one it might still be reading: see ``emittable_text``.
REPLACEMENT_CHAR = "\ufffd"

#: Where a refused reply's evidence lands on its failed trace event. A refused reply is
#: dropped -- the event has no response and the reply is nowhere else -- so this is the
#: only record of what the server actually sent, and the only way the next refusal can
#: be read off the trace instead of guessed at from a log line.
REFUSED_EXTRA = "agentic_coding.refused_reply"

#: Where a checked reply records how many more replacement characters its text has than
#: its tokens decode to. Zero unless the reply was accepted by writing split characters
#: as replacement characters (``split_characters``), so it counts the replies that
#: relied on that and nothing else. Absent where the check was unarmed.
REPLACED_EXTRA = "agentic_coding.replacement_chars_added"

#: How much of the reply that evidence carries, per string and per id list. A runaway
#: is 115,000 characters of 32,768 ids, and a server that fails this check fails it on
#: every reply it serves, so keeping whole replies would write hundreds of megabytes of
#: trace for a run already scored 0. What is kept locates the divergence instead of
#: reproducing the reply: the lengths and the first differing index say where it is,
#: and the ends say what it looks like there.
REFUSED_KEEP_CHARS = 400
REFUSED_KEEP_TOKENS = 64


def _first_difference(a: str, b: str) -> int | None:
    """The index where two strings first differ, or where the shorter one ends; None
    if they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def refused_reply_evidence(text: str, tokens: list[int], stripped: str,
                           kept: str) -> dict:
    """What a refused reply leaves behind for whoever reads the trace.

    Bounded by ``REFUSED_KEEP_CHARS`` and ``REFUSED_KEEP_TOKENS``. A reader gets the
    three lengths, where the served text first parts from each decoding, the last of
    each string, and the ends of the id list -- enough to tell a reply the cap cut
    mid-character from one a parser edited, which is the distinction the refusal turns
    on. It is not enough to re-decode the middle of a long reply by hand; that would
    be the whole id list, and this runs on every refused reply."""
    return {
        "chars": len(text), "tokens": len(tokens),
        "decoded_chars": {"stripped": len(stripped), "kept": len(kept)},
        "first_diff": {"stripped": _first_difference(text, stripped),
                       "kept": _first_difference(text, kept)},
        "text_tail": text[-REFUSED_KEEP_CHARS:],
        "decoded_tail": {"stripped": stripped[-REFUSED_KEEP_CHARS:],
                         "kept": kept[-REFUSED_KEEP_CHARS:]},
        "token_ids_head": list(tokens[:REFUSED_KEEP_TOKENS]),
        "token_ids_tail": list(tokens[-REFUSED_KEEP_TOKENS:]),
    }


def emittable_text(tokens: list[int], tokenizer, decoded: str,
                   skip_special_tokens: bool) -> str:
    """The most of ``decoded`` a server streaming these tokens can have sent.

    A reply cut by the output cap can end in the middle of a character: the last few
    tokens are the first bytes of one, and the rest were never generated. Decoding the
    whole reply at once writes those bytes as replacement characters, because that is
    what a decoder does with bytes it cannot read. A server cannot: it detokenizes as
    it generates, and text it has not finished reading is text it must hold back rather
    than send a character it may have to take back. At the end of a stream whatever is
    still held back is never sent at all, and its ``content`` is short by exactly that.

    So the text a streaming server can have emitted is this decode with its trailing
    tokens removed while the decode still ends in a replacement character. The stopping
    point is the last token after which a decoder had a whole character in hand, which
    is the last point it could have emitted anything, and no text beyond it is
    reachable: a reply that ends in ordinary characters loses nothing here, and one
    that ends mid-character loses only the bytes it never finished."""
    n = len(tokens)
    while n > 0 and decoded.endswith(REPLACEMENT_CHAR):
        n -= 1
        decoded = tokenizer.decode(tokens[:n], skip_special_tokens=skip_special_tokens)
    return decoded


def split_characters(tokens: list[int], tokenizer, decoded: str,
                     skip_special_tokens: bool) -> dict[int, int] | None:
    """The characters of ``decoded`` whose bytes come from more than one token, by
    index, each with the most replacement characters a server may write in its place.

    A byte-level vocabulary spells a character it has no piece for as several pieces,
    and a server that turns its stream into text as it goes can get that character
    wrong and nothing else. Decoding each chunk's ids on its own writes a replacement
    character for each part of it. Re-decoding every id so far and sending only what
    is new sends the replacement character the first part decoded to, and never the
    character. The most is the character's length in bytes, because no split of it
    decodes to more replacement characters than it has bytes.

    Found by decoding, which is all a tokenizer is asked for here. A decode that does
    not end in a replacement character ends between characters, so the reply is read
    in stretches that end there. Inside a stretch, the decode up to a token boundary
    ends in one replacement character for the character that boundary cuts, and where
    the stretch's own decode has a real character at that place, that character is
    split. A replacement character the decode writes for bytes that form no character
    can be split too, but no decode says how many bytes it stands for: its first part
    reads as one replacement character however long it is. So its most is one more
    than the replacement characters its remaining bytes decode to, which is what a
    server decoding token by token writes and never more than its bytes. This bound
    only comes into play where such bytes are split across a server's chunks: bytes
    the model wrote that form no character, or the start of one the output cap cut.

    None where the stretches do not add up to ``decoded``: a tokenizer whose decode is
    not its tokens' bytes read in order, on which nothing is accepted this way."""
    def decode(ids: list[int]) -> str:
        return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    split: dict[int, int] = {}
    start = base = 0  # the stretch's first token, and its first character in `decoded`
    heads: list[str] = []  # the stretch's decode up to each boundary inside it
    for end in range(1, len(tokens) + 1):
        piece = decode(tokens[start:end])
        if piece.endswith(REPLACEMENT_CHAR) and end < len(tokens):
            heads.append(piece)
            continue
        if decoded[base:base + len(piece)] != piece:
            return None
        for cut, head in enumerate(heads, start + 1):
            at = base + len(head) - 1
            if at >= base + len(piece) or head[:-1] != decoded[base:at]:
                return None
            if decoded[at] != REPLACEMENT_CHAR:
                split[at] = len(decoded[at].encode("utf-8"))
                continue
            tail = decode(tokens[cut:end])
            if head + tail == piece:
                continue  # the boundary falls between characters after all
            extra = len(head) + len(tail) - len(piece)
            if (extra < 1 or tail[:extra] != REPLACEMENT_CHAR * extra
                    or head + tail[extra:] != piece):
                return None
            split[at] = max(split.get(at, 0), 1 + extra)
        start, base, heads = end, base + len(piece), []
    return split if base == len(decoded) else None


def replacements_added(text: str, decoded: str, split: dict[int, int]) -> int | None:
    """How many more replacement characters ``text`` has than ``decoded``, if it is
    ``decoded`` with some split characters written as replacement characters; None if
    it differs in any other way.

    Each character of ``text`` that is not a replacement character must be the next
    character of the decode. Each run of replacement characters must stand for a
    stretch of the decode that can be written that way: the decode's own replacement
    characters take one each, and a split character between one and its most. A run
    can often be read more than one way, so every reading is carried forward until the
    text rules it out, and none is chosen. The count needs no choice either, because
    it is the difference in replacement characters, which every reading shares."""
    most = [split.get(i, 1 if c == REPLACEMENT_CHAR else 0)
            for i, c in enumerate(decoded)]
    total = list(accumulate(most, initial=0))
    # Where the stretch of characters that can be written as replacement characters,
    # starting at each index, ends.
    stretch_end = list(range(len(decoded) + 1))
    for i in range(len(decoded) - 1, -1, -1):
        if most[i]:
            stretch_end[i] = stretch_end[i + 1]
    reach = {0}  # the indices of the decode the text read so far can have ended at
    j = 0
    while j < len(text) and reach:
        if text[j] != REPLACEMENT_CHAR:
            reach = {i + 1 for i in reach if i < len(decoded) and decoded[i] == text[j]}
            j += 1
            continue
        run = j
        while run < len(text) and text[run] == REPLACEMENT_CHAR:
            run += 1
        n = run - j
        # A stretch from i to e stands for n replacement characters if it has at most
        # n characters and their mosts add up to at least n.
        reach = {e for i in reach
                 for e in range(max(i + 1, bisect_left(total, total[i] + n)),
                                min(i + n, stretch_end[i]) + 1)}
        j = run
    if len(decoded) not in reach:
        return None
    return text.count(REPLACEMENT_CHAR) - decoded.count(REPLACEMENT_CHAR)


def check_text_matches_tokens(text: str, tokens: list[int] | None,
                              tokenizer) -> int | None:
    """Refuse a reply whose ``content`` is not exactly what its ``token_ids`` say.

    The loop reads the model's tool call out of the reply text, so the text has to be
    the tokens the gate scores and nothing else. Two server-side parsers break that by
    moving text out of ``content`` into a field of its own -- a reasoning parser takes
    the thinking, a tool-call parser takes the call -- and the second is fatal, because
    what it removes is the only thing the loop is looking for. The trajectory would die
    on repeated format errors having done nothing wrong, and the record would blame the
    model.

    So contestants serve raw, with neither parser, and this is what holds them to it.
    The comparison is exact rather than approximate because under that contract the two
    sides are the same string by construction, save for the shapes documented below,
    each of which is exact in its own right. ``decode(encode(reply))`` reproduces every
    completion in ``tests/fixtures`` character for character, and the markup the loop
    reads -- ``<think>``, ``<tool_call>``, ``<arg_key>`` -- survives
    ``skip_special_tokens``. A slack window is a hole: it is exactly enough to hide a
    short call.

    Special tokens are the one place two honest servers differ. A stock server drops
    them from the text (``skip_special_tokens=True``), so a reply that ends on the
    ``<|observation|>`` turn-boundary token reads without it; this task's own baseline
    keeps them, so the same reply ends in the token's text. Both are
    the exact text of the tokens under one detokenization, so the check accepts either
    and nothing in between: a parser that removes a tool call matches neither.

    A reply cut mid-character is the third honest shape, and the only one where the
    two sides are not the same string. A server streams, so it cannot send bytes it has
    not finished reading; a decode of the whole reply at once writes them as
    replacement characters. The reply is short by exactly what the server could not
    finish, which ``emittable_text`` computes, and the check accepts that too. It is
    not a slack window: the accepted difference is a run of characters the reply's own
    tokens say are unreadable, nothing within it can be text a parser moved, and the
    same reply with a tool call removed is refused whether or not it ends mid-character.
    This is what ``tokenizers.decoders.DecodeStream`` emits, stepped token by token the
    way ``vllm.v1.engine.detokenizer`` steps it: a piece that carries a space and then
    the first bytes of a character is withheld whole, so the space goes with them.

    A character whose bytes more than one token carries is the fourth shape, and the
    one that can fall anywhere in a reply. A streaming server has to hold such a
    character back until its last piece arrives, and one that does not can write it
    wrong: decoding each chunk's ids on its own writes a replacement character for each
    part, and re-decoding every id so far and sending only what is new sends a
    replacement character and never the character. So the check also accepts any of
    the shapes above with some split characters written as between one and as many
    replacement characters as the character has bytes (``split_characters``), and the
    reply's event records how many replacement characters that added. This is not a
    slack window either. A split character is never ASCII, so none of the markup the
    loop reads can be replaced; a run of replacement characters must stand for split
    characters and nothing else; text missing or added is refused as before; and the
    special tokens are kept or dropped the same way throughout.

    Returns the replacement characters added, zero for a reply accepted without them,
    and None when unarmed, which is when nothing can detokenize: every process without
    a mounted checkpoint."""
    if tokenizer is None or not tokens:
        return None
    try:
        stripped = tokenizer.decode(tokens, skip_special_tokens=True)
        kept = tokenizer.decode(tokens, skip_special_tokens=False)
    except Exception as exc:  # noqa: BLE001 -- a decode that fails cannot judge
        log.error("agentic_coding: could not detokenize a reply to check it against its own "
                  "text (%r); the check is unarmed for this turn", exc)
        return None
    if text == stripped or text == kept:
        return 0
    # Only for a reply whose decode ends in characters it cannot have sent; for every
    # other reply the helper returns the decode it was handed and nothing is accepted
    # here that was not already accepted above.
    for decoded, skip in ((stripped, True), (kept, False)):
        if text == emittable_text(tokens, tokenizer, decoded, skip):
            log.warning("agentic_coding: a reply ends %d characters before its own tokens do; "
                        "they are the bytes of a character the output cap cut in half, "
                        "which a streaming server cannot send. Accepted.",
                        len(decoded) - len(text))
            return 0
    for decoded, skip in ((stripped, True), (kept, False)):
        # Writing a character as replacement characters adds at least one, so a text
        # with no more of them than a shape is not that shape written this way, and
        # the tokens need not be walked for it. The walk is about 75 ms for a
        # 32,768-token reply, and only a reply refused above pays it.
        shapes = [shape for shape in dict.fromkeys(
                      (decoded, emittable_text(tokens, tokenizer, decoded, skip)))
                  if text.count(REPLACEMENT_CHAR) > shape.count(REPLACEMENT_CHAR)]
        if not shapes:
            continue
        split = split_characters(tokens, tokenizer, decoded, skip)
        if split is None:
            continue
        # The shorter shape is a prefix of the decode, so the decode's split characters
        # are its split characters as far as it goes.
        for shape in shapes:
            added = replacements_added(text, shape, split)
            if added is not None:
                log.warning("agentic_coding: a reply writes %d more replacement characters than "
                            "its tokens decode to, in place of characters whose bytes "
                            "span tokens, the way a server that decodes piece by piece "
                            "does. Accepted.", added)
                return added
    raise ArtifactProtocolError(
        f"the reply's `content` is not what its own `token_ids` detokenize to "
        f"({len(text)} characters against {len(stripped)} with special tokens dropped, "
        f"{len(kept)} with them kept). This harness reads the model's tool call out of "
        f"`content`, so `content` must be the exact text of the tokens, with nothing "
        f"removed. The usual cause is a server-side parser moving text into a field of "
        f"its own: run with no --reasoning-parser and no --tool-call-parser. Served: "
        f"{text[:200]!r}. Tokens decode to: {stripped[:200]!r} (special tokens "
        f"dropped) / {kept[:200]!r} (kept). "
        # Both ends, because a reply that diverges does it at one of them.
        f"Served ends: {text[-120:]!r}. Tokens end: {stripped[-120:]!r} (dropped) / "
        f"{kept[-120:]!r} (kept).",
        # The trace keeps the error, not the reply, so the rest of what a reader needs
        # goes on the failed event itself.
        extras={REFUSED_EXTRA: refused_reply_evidence(text, tokens, stripped, kept)})


def _problem_statement(seed_env: Envelope) -> str:
    """The task statement a trajectory is seeded with."""
    stated = seed_env.metadata.get(PROBLEM_STATEMENT_KEY)
    if isinstance(stated, str) and stated.strip():
        return stated
    payload = seed_env.first_of("text")
    return payload.text if payload is not None else ""


@dataclass
class MiniSweFramework(AgentFramework):
    """The mini-SWE-agent loop on the harness's driver.

    The fields are the loop's knobs, and ``traffic_config`` reports them all, so the
    calibration fingerprint sees a change to any of them: each one moves the
    population of prompts the gate is calibrated over.
    """

    #: Characters of command output returned to the model per turn.
    output_limit_chars: int = DEFAULT_OUTPUT_LIMIT_CHARS
    #: Consecutive unparseable replies tolerated before the trajectory ends.
    max_consecutive_format_errors: int = DEFAULT_MAX_CONSECUTIVE_FORMAT_ERRORS
    #: Wall-clock ceiling on one command, in seconds.
    command_timeout_sec: int = DEFAULT_COMMAND_TIMEOUT_SEC
    #: Tokens the continuation of a reply cut at the output cap may spend, or None to
    #: never continue one.
    #:
    #: Declared here so it reaches ``traffic_config`` and the calibration fingerprint.
    #: It belongs there by the usual test: moving it changes what the server is asked
    #: to produce on every capped turn, and nothing fails when it moves.
    #:
    #: Its own budget rather than the output cap reused. The cap bounds thinking; this
    #: bounds the visible reply that follows once the thought has been closed for the
    #: model, which is a tool call and a sentence.
    reply_budget: int | None = None

    #: Context window, in tokens, a trajectory is budgeted against. Set it to the same
    #: value as the task's ``MAX_CONTEXT_LEN``: that is the window the servers and the
    #: reference are built at. None leaves the loop unbounded.
    max_context_len: int | None = None
    #: Where the staged prefixes live, or None where no task declares one. A directory
    #: on the node the drive runs on, like the sandbox images, because a frozen
    #: conversation at depth is megabytes and there is one per task.
    #:
    #: Not in the calibration fingerprint; see ``_FINGERPRINT_EXEMPT`` below.
    prefix_dir: str | None = None
    #: The checkpoint a prefix must have been cut against. A conversation is the
    #: writing of the weights that produced it, so a prefix from another model is
    #: off-distribution rather than merely stale. None does not check.
    checkpoint: str | None = None
    #: Replacement containers a session may have before its sandbox faults are handed
    #: to the model instead, and faults in a row that end the trajectory.
    #:
    #: Not in the calibration fingerprint; see ``_FINGERPRINT_EXEMPT`` below.
    #:
    #: The runner enforces the restart budget and the loop enforces the fault run;
    #: the task passes one constant to each so these fields cannot describe a
    #: budget the sandbox did not have.
    max_sandbox_restarts: int = 0
    max_consecutive_sandbox_faults: int = DEFAULT_MAX_CONSECUTIVE_SANDBOX_FAULTS

    #: Fields ``traffic_config`` withholds from the calibration fingerprint.
    #:
    #: The test each one is judged against: a setting belongs in the fingerprint when
    #: changing it changes the traffic without failing. Both halves matter. Timing that
    #: alters which sessions exist is in; a value that is verified downstream, so a
    #: wrong one stops the run rather than serving different traffic, is out; a knob
    #: guarding something the reference does not do is out.
    #:
    #: Named rather than simply left out, because an absence reads as an oversight and
    #: the next person to look has no way to tell which it was.
    #:
    #: ``prefix_dir`` is a host staging path. A prefix is loaded by task name and
    #: refused unless it hashes to the digest its row declares, so a different
    #: directory either holds the same conversations -- identical traffic -- or fails
    #: the run. It cannot quietly serve something else. In, it would mean a calibration
    #: taken on one cluster could not be reused on another for a reason with nothing to
    #: do with what was served.
    #:
    #: ``checkpoint`` is the counter-example that keeps this from being "paths are
    #: out". A prefix cut against another model is off-distribution, the digest pins
    #: the file rather than the model, and nothing downstream catches it. Changing it
    #: changes the traffic without failing, so it stays in.
    #:
    #: The two sandbox knobs guard sandbox faults, which the reference does not produce
    #: on Terminal-Bench, so moving them does not change the traffic. The knobs stay
    #: on; moving them does not stale a calibration.
    _FINGERPRINT_EXEMPT: ClassVar[frozenset[str]] = frozenset(
        {"max_sandbox_restarts", "max_consecutive_sandbox_faults", "prefix_dir"})

    def traffic_config(self) -> dict:
        return {k: v for k, v in super().traffic_config().items()
                if k not in self._FINGERPRINT_EXEMPT}

    def extra_output_caps(self) -> set[int]:
        """The continuation budget the loop stamps on the second half of a capped turn.

        A turn whose reply reached the output cap is continued under ``reply_budget``
        rather than the cap, so a stored capture holds those continuations at that
        smaller value; reported here so calibration keys them as this workload's."""
        return {self.reply_budget} if self.reply_budget else set()

    def __post_init__(self) -> None:
        """Take the chat-template pin the prompt counter renders with, and the render's
        own answer to whether it leaves a thought open.

        Machinery, not settings, so ``check_model`` and ``warmup`` fill them in and
        ``traffic_config`` drops them with every other underscored name."""
        self._chat_template_kwargs: dict = {}
        self._thinking_opened = False

    async def warmup(self) -> None:
        """Load the prompt counter's tokenizer, before the drive anchors its clock, and
        read the one fact about the render that the loop cannot work out for itself.

        About a second of CPU. Loading it lazily would bill that second to whichever
        trajectory finished its first turn first. A process with no snapshot mounted
        loads nothing and says nothing: the counter reports that it cannot count, and
        the loop bounds the reply's own figure instead.

        A mounted snapshot whose tokenizer does not load is the harness's fault and
        ends the drive here, before a request is sent. Driven on, every request would
        lack the prompt count its latency budget is computed from, and the run would
        score zero as the artifact's."""
        tokenizer = await asyncio.to_thread(prompt_tokenizer)
        wdir = _mounted_weights()
        if tokenizer is None and wdir is not None:
            raise HarnessFault(f"no tokenizer from the mounted checkpoint {wdir}: "
                               f"{_LOAD_ERRORS.get(wdir)}")
        self._thinking_opened = await asyncio.to_thread(self._renders_an_open_thought)

    def _renders_an_open_thought(self) -> bool:
        """Whether a generation prompt rendered with this task's pin ends inside an
        open thought.

        The fact that decides how a reply carrying no reasoning delimiter is read. A
        reply that spent its whole output cap inside a thought and a reply an engine
        already stripped are the same string, so it cannot come from the reply; it
        comes from the prompt, which the harness renders itself. Rendered through the
        same function the prompt counter uses, with the same pin, so it describes the
        prompt the artifact is actually served.

        False where nothing can render, which is every process without a mounted
        checkpoint. That is the reading that leaves an already-stripped reply alone."""
        tokenizer = prompt_tokenizer()
        if tokenizer is None:
            return False
        from hls.reference.engines import chat_prompt_text
        probe = [{"role": "user", "content": "x"}]
        rendered = chat_prompt_text(tokenizer, probe, [BASH_TOOL],
                                    dict(self._chat_template_kwargs))
        opened = rendered.endswith(THINK_OPEN)
        log.info("agentic_coding: the generation prompt %s an open thought, so a reply with no "
                 "reasoning delimiter is read as %s",
                 "leaves" if opened else "does not leave",
                 "an unfinished thought" if opened else "visible text")
        return opened

    def check_reply(self, text: str, tokens: list[int] | None) -> str | None:
        """``check_text_matches_tokens`` for a reply this framework never reads.

        Every reply it does read goes through that check inside ``complete``. A turn
        the drive's scheduled stop cut never returns from there, so the driver runs
        this on its behalf; without it the only unchecked reply of a drive is the one
        arriving in a window the server can predict, and padding the token array there
        buys a lower mean inter-token latency with no text to justify it.

        A reason rather than a raise, because the caller is finishing a record inside a
        cancellation. Unarmed wherever the answered path is: no mounted checkpoint, or
        a reply carrying no ids."""
        try:
            check_text_matches_tokens(text, tokens, prompt_tokenizer())
        except ArtifactProtocolError as e:
            return str(e)
        return None

    def check_model(self, model: ModelSpec) -> None:
        """Accept any checkpoint, and take the chat-template pin the counter renders
        with.

        Any checkpoint is accepted, but only one is read correctly: the loop parses
        calls in the ``<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value>``
        syntax this task's checkpoint documents in its own chat template, and a
        checkpoint whose template asks for a JSON call body would parse to nothing.
        Nothing is rejected here because the task pins its model, so the pairing is
        settled before a run starts; ``actions_toolcall.parse_toolcall_text`` is the
        one place that would have to grow to read a second dialect. An inline reasoning
        span is stripped before parsing, which covers both a server that separates
        reasoning into its own field and one that does not.

        The pin is read here because this is the one place the framework is handed its
        model. ``TaskSpec`` calls this at construction and on every revalidated copy,
        so an overridden pin is picked up; a framework that never met a TaskSpec keeps
        the empty pin ``__post_init__`` gave it and renders without one, which is what
        a model that declares none renders with anyway."""
        self._chat_template_kwargs = dict(model.chat_template_kwargs)

    def prompt_counter(self):
        """The loop's prompt-token counter.

        It counts the prompt the harness rendered, through the checkpoint's own
        tokenizer and the model's chat-template pin -- the same render a trusted
        calibration member is served and reports its ``prompt_tokens`` from, so a
        graded trajectory and a calibration trajectory are budgeted on the same number.

        It answers ``None`` where no checkpoint is mounted, which is every process but
        the driver and the pool members; the loop bounds the reply's own figure
        there instead. Whether this process has one is decided per call rather than
        here, so a framework built before the mount still counts."""
        template_kwargs = dict(self._chat_template_kwargs)

        async def count(messages: list[dict],
                        continue_final_message: bool = False) -> int | None:
            tokenizer = prompt_tokenizer()
            if tokenizer is None:
                return None
            # In a thread: a long conversation is hundreds of kilobytes to render and
            # tokenize, and one event loop drives every trajectory of the drive.
            return await asyncio.to_thread(_count_prompt_ids, tokenizer, messages,
                                           template_kwargs, continue_final_message)

        return count

    def loop(self, max_output_tokens: int | None = None,
             work_dir: str = WORK_DIR) -> MiniSweLoop:
        """The loop for one trajectory.

        ``max_output_tokens`` is the per-request output cap the harness stamps, handed
        over per trajectory because it belongs to the tenant's workload rather than to
        this framework: ``run_trajectory`` reads it off the ``HarnessContext``.
        ``work_dir`` likewise belongs to the trajectory's own task image. The context
        window above is declared instead, because the task declares it and no workload
        carries it."""
        return MiniSweLoop(output_limit_chars=self.output_limit_chars,
                           max_consecutive_format_errors=self.max_consecutive_format_errors,
                           max_consecutive_sandbox_faults=self.max_consecutive_sandbox_faults,
                           command_timeout_sec=self.command_timeout_sec,
                           max_context_len=self.max_context_len,
                           count_prompt_tokens=self.prompt_counter(),
                           max_output_tokens=max_output_tokens,
                           reply_budget=self.reply_budget,
                           thinking_opened=self._thinking_opened,
                           work_dir=work_dir)

    def prefix_for(self, seed_env: Envelope):
        """The frozen conversation this session resumes from, or None for a cold one.

        Read from the row's declared digest rather than from whatever is staged: a
        prefix is part of the workload, so a session must resume from the one its
        corpus was calibrated over or fail saying it could not."""
        digest = str(seed_env.metadata.get(PREFIX_DIGEST_FIELD) or "")
        if not digest:
            return None
        if not self.prefix_dir:
            raise prefix_format.PrefixError(
                f"the corpus declares a prefix for "
                f"{seed_env.metadata.get(TASK_NAME_FIELD)!r} but this framework has no "
                f"prefix_dir to load it from")
        return prefix_format.load(self.prefix_dir,
                                  str(seed_env.metadata[TASK_NAME_FIELD]),
                                  expect_digest=digest,
                                  expect_checkpoint=self.checkpoint)

    def turns_for(self, seed_env: Envelope, max_turns: int) -> int:
        """Turns this session serves: its own budget where the corpus sets one, and
        the framework's otherwise.

        The lower of the two, so a per-task budget can shorten a deep session but
        never lift one past the trajectory ceiling the task declares."""
        budget = int(seed_env.metadata.get(TURN_BUDGET_FIELD) or 0)
        return min(max_turns, budget) if budget > 0 else max_turns

    async def run_trajectory(self, seed_item, llm_call, tool_runner: ToolRunner,
                             on_event, ctx, max_turns: int) -> None:
        seed_env = seed_item.turns[0]
        session_id = seed_item.request_id
        ground_truth = (seed_item.ground_truth_turns[0]
                        if seed_item.ground_truth_turns else None)
        base_metadata = {**seed_env.metadata, "tenant_id": ctx.tenant_id,
                         "session_id": session_id}

        async def complete(request: dict) -> dict:
            env = Envelope(request_id=request["request_id"],
                           contents=[TextPayload(text=request["text"])],
                           metadata=dict(request["metadata"]))
            response = await llm_call(env)
            output = response.output if response is not None else None
            payload = output.first_of("text") if output is not None else None
            tokens = payload.tokens if payload is not None else None
            text = payload.text if payload is not None else ""
            # Before anything reads the text: it is where the tool call is, so a
            # server that stripped the call out of it must be refused rather than
            # quietly driving a trajectory that can never act.
            replaced = check_text_matches_tokens(text, tokens, prompt_tokenizer())
            # The reply's length and stop label ride on the payload rather than the
            # envelope metadata, and the loop needs both to tell a turn that spent its
            # whole output cap from one that said nothing.
            return {"text": text,
                    "served_tokens": len(tokens) if tokens is not None else None,
                    "finish_reason": (payload.finish_reason
                                      if payload is not None else None),
                    "metadata": dict(output.metadata) if output is not None else {},
                    "envelope": env, "response": response, REPLACED_EXTRA: replaced}

        async def run_tool(name: str, args: dict) -> dict:
            # The session id routes the command to this trajectory's own checkout
            # rather than a shared one.
            return await tool_runner.run(name, {**args, "session_id": session_id})

        async def emit(turn: dict) -> None:
            completion = turn["completion"]
            extras = turn["extras"]
            if completion.get(REPLACED_EXTRA) is not None:
                extras = {**extras, REPLACED_EXTRA: completion[REPLACED_EXTRA]}
            await on_event(TraceEvent(
                request_id=turn["request_id"], session_id=session_id,
                turn_index=turn["turn_index"], input=completion["envelope"],
                response=completion["response"], ground_truth=ground_truth,
                t_send=turn["t_send"], t_recv=turn["t_recv"], extras=extras))

        # The trajectory owns the sandbox's lifetime because it is the only scope that
        # knows when the trajectory began and ended. The close is in a finally so a
        # trajectory that raised still releases its container; leaking one would hold
        # a node's memory until the job ended.
        # The seed's metadata is the session's context: the tool runner reads the task
        # name off it to pick that task's image, and the working directory to start the
        # container somewhere the image has.
        # Loaded before the container starts, so a prefix that is missing or not the
        # declared one fails without having provisioned anything.
        prefix = self.prefix_for(seed_env)
        await tool_runner.open_session(session_id, **dict(seed_env.metadata))
        work_dir = seed_env.metadata.get(WORK_DIR_KEY) or WORK_DIR
        try:
            await self.loop(ctx.max_output_tokens, work_dir).run(
                session_id=session_id,
                problem_statement=_problem_statement(seed_env),
                base_metadata=base_metadata, complete=complete, run_tool=run_tool,
                emit=emit, max_turns=self.turns_for(seed_env, max_turns),
                prefix=prefix)
        except SandboxFault as exc:
            # The loop ends a trajectory whose container has stopped executing
            # commands, and cannot say what that means for the run: it may not import
            # `hls`. Said here, where the two vocabularies meet, and said as the same
            # fault the tool runner raises when it cannot replace a container, because
            # the two are one outcome: the driver drops the session, keeps the turns it
            # served, and writes the drop and this message into the trace, rather than
            # score a failed request against a server that never touched the exec
            # channel. The loop's own account of the ending is the message; renaming
            # the fault is all that happens here.
            raise SandboxGone(str(exc)) from exc
        finally:
            await tool_runner.close_session(session_id)


__all__ = ["BASH_TOOL_NAME", "MiniSweFramework", "MiniSweLoop",
           "PROBLEM_STATEMENT_KEY", "SUBMIT_SENTINEL", "WORK_DIR", "WORK_DIR_KEY",
           "prompt_tokenizer"]
