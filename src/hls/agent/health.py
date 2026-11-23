"""The build's health: what the agent's recent turns say, read from its session transcript.

A session that has derailed keeps writing: a model that loops on its own special tokens can
grow its transcript for hours after it stops making progress, so a check on growth alone
passes it. ``check`` samples content instead. It reads the end of the agent's newest session
transcript (Claude Code's ``sessions/projects/<project>/<session>.jsonl``, codex's
``sessions/<y>/<m>/<d>/rollout-*.jsonl``), takes the last ``WINDOW`` assistant turns and the
last ``WINDOW`` tool results, and calls the session derailed when one of these holds:
- ``repeat``: one turn with prose, its text and its tool calls alike, is ``REPEAT_MIN`` or more
  of them. A turn that is only a tool call does not count: an agent waiting on a long job
  issues the same ``sleep`` or poll call for hours, by design;
- ``invalid_calls``: ``INVALID_MIN`` or more tool results are Claude Code's
  ``InputValidationError``, a tool call the model could not form;
- ``special_tokens``: ``SPECIAL_MIN`` or more turns hold a chat template's control token
  (``SPECIAL_TOKENS``) in the model's own prose.
Nothing is judged before ``WINDOW`` turns exist. The thresholds are high on purpose: a false
alarm stops re-entry for a session that was working.

The stage calls ``check`` at each heartbeat, in the runner, while the agent writes. The files are
the agent's, so what is read here only decides re-entry; the budget comes from the runner's
clock, never from them. Reading them never follows a symlink, so a link the agent planted does
not lead the runner out of its logs, and opens only regular files, without blocking, so a fifo
in place of a transcript cannot stall the runner's loop. Nothing the agent wrote may make the
check raise: a line longer than ``LINE_MAX_CHARS``, or one the decoder refuses whatever the
reason (a nesting past its depth among them), is unreadable, no evidence, and a warning names
the file and how many such lines it had; the turns are judged on the other lines.
``Health`` also carries what the heartbeat event reports: the last assistant turn's time, kept
only as an ISO 8601 timestamp of at most ``TIMESTAMP_MAX_CHARS`` characters, and the size of
the agent's logs."""

from __future__ import annotations

import json
import logging
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from hls.agent import transcript
from hls.agent.listing import first_entries

if TYPE_CHECKING:
    from hls.arms import HarnessAdapter

#: Turns, and tool results, the check looks at.
WINDOW = 20
#: Identical turns of the window that make a repeat.
REPEAT_MIN = 12
#: InputValidationError results of the window that make a derailment.
INVALID_MIN = 8
#: Turns of the window with a control token in their prose that make a derailment.
SPECIAL_MIN = 8
#: Chat-template control tokens a model never writes in its own prose when it is working.
SPECIAL_TOKENS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "[gMASK]", "<sop>", "<|user|>", "<|assistant|>",
                  "<|observation|>", "<|system|>", "<|begin_of_text|>", "<|eot_id|>", "<|start_header_id|>",
                  "<|end_header_id|>")
#: Claude Code's marker of a tool call it could not parse.
INVALID_CALL = "InputValidationError"
#: Bytes read from the end of a transcript: room for the lines the check
#: judges (``WINDOW`` turns and as many tool results) at about 200 KB each. A tail of longer
#: lines holds fewer, and with fewer than ``WINDOW`` turns the check judges nothing.
TAIL_BYTES = 8 << 20
#: Characters of a transcript line the check parses; a longer one is unreadable.
LINE_MAX_CHARS = 1 << 20
#: Characters of the last turn's time the heartbeat reports, as an ISO 8601 timestamp.
TIMESTAMP_MAX_CHARS = 64
# The size reading and the newest-transcript search each list at most the transcript copy's
# bound on entries in all (``hls.agent.transcript.MAX_ENTRIES``, read when they run).
# Opens a transcript: never through a symlink, and without blocking on a fifo.
_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC

log = logging.getLogger(__name__)
# A timestamp as Claude Code and codex write one: a date, a time, fractions and a zone.
_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d{1,9})?(Z|[+-]\d{2}:?\d{2})?")


class TranscriptLayout(NamedTuple):
    """Where a harness's CLI writes a session's transcript under the agent's logs
    (``HarnessAdapter.transcript_layout``)."""

    #: The directories between ``sessions`` and the transcript, each by its name, or None for
    #: any name.
    dirs: tuple[str | None, ...]
    #: The start of the transcript's file name.
    prefix: str
    #: The end of the transcript's file name.
    suffix: str


@dataclass(frozen=True)
class Turn:
    """One assistant turn."""

    #: Its prose.
    text: str
    #: Its tool calls, each as name and input.
    calls: str
    #: Its timestamp as the transcript writes it; None when it has none.
    at: str | None


@dataclass(frozen=True)
class Health:
    """What one check found."""

    #: True when the session has derailed.
    derailed: bool
    #: Which signal: "repeat", "invalid_calls" or "special_tokens"; empty when not derailed.
    signal: str
    #: The signal's count in the window, and the window's size, for the event.
    count: int
    #: Assistant turns read.
    turns: int
    #: The last assistant turn's timestamp; None when there is none.
    last_turn_at: str | None
    #: Bytes of regular files under the agent's logs.
    log_bytes: int


def _listed(path: Path, budget: list[int]) -> list[os.DirEntry]:
    """The entries of *path* among the first the read may still look at: *budget*'s one count
    of what is left of the transcript copy's ``MAX_ENTRIES``, which this spends
    (``first_entries``); none when *path* cannot be listed."""
    try:
        entries, _ = first_entries(path, budget[0])
    except OSError:
        return []
    budget[0] -= len(entries)
    return entries


def _subdirs(path: Path, name: str | None, budget: list[int]) -> list[Path]:
    """The directories in *path*, or the one called *name*, that are not symlinks, among the
    entries *budget* still allows."""
    return [Path(entry.path) for entry in _listed(path, budget)
            if entry.is_dir(follow_symlinks=False) and (name is None or entry.name == name)]


def newest_transcript(agent_logs: Path, adapter: HarnessAdapter) -> Path | None:
    """The agent's newest session transcript under *agent_logs*, where *adapter*'s harness
    writes one, a regular file reached through no symlink, among the first
    ``transcript.MAX_ENTRIES`` entries the search lists in all; None before there is one.
    Subagents' transcripts, a level deeper, are not the session's."""
    levels, prefix, suffix = adapter.transcript_layout
    budget = [transcript.MAX_ENTRIES]
    dirs = [] if agent_logs.is_symlink() else _subdirs(agent_logs, "sessions", budget)
    for name in levels:
        dirs = [sub for d in dirs for sub in _subdirs(d, name, budget)]
    found: list[tuple[float, str, Path]] = []
    for d in dirs:
        for entry in _listed(d, budget):
            try:
                if (entry.name.startswith(prefix) and entry.name.endswith(suffix)
                        and entry.is_file(follow_symlinks=False)):
                    found.append((entry.stat(follow_symlinks=False).st_mtime, entry.name, Path(entry.path)))
            except OSError:
                continue
    return max(found)[2] if found else None


def tail_lines(path: Path, max_bytes: int = TAIL_BYTES) -> list[str]:
    """The whole lines in the last *max_bytes* of the regular file *path*. Raise OSError for a
    path that is a symlink or not a regular file."""
    fd = os.open(path, _OPEN_FLAGS)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(f"{path} is not a regular file")
        size = os.lseek(fd, 0, os.SEEK_END)
        start = max(0, size - max_bytes)
        data = os.pread(fd, size - start, start)
    finally:
        os.close(fd)
    lines = data.decode("utf-8", errors="replace").splitlines()
    return lines[1:] if size > max_bytes and lines else lines


def _objects(lines: list[str]) -> tuple[list[dict[str, Any]], int]:
    """The JSON objects of *lines*, and how many lines were unreadable: longer than
    ``LINE_MAX_CHARS``, or refused by the decoder for any reason."""
    out = []
    unreadable = 0
    for line in lines:
        if len(line) > LINE_MAX_CHARS:
            unreadable += 1
            continue
        try:
            obj = json.loads(line)
        except Exception:  # noqa: BLE001 -- the agent wrote the line: any refusal is no evidence
            unreadable += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
    return out, unreadable


def _timestamp(value: Any) -> str | None:
    """*value* when it is an ISO 8601 timestamp of at most ``TIMESTAMP_MAX_CHARS`` characters;
    None otherwise."""
    if isinstance(value, str) and len(value) <= TIMESTAMP_MAX_CHARS and _TIMESTAMP.fullmatch(value):
        return value
    return None


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict) and isinstance(b.get("text"), str))
    return ""


def claude_turns(objs: list[dict[str, Any]]) -> tuple[list[Turn], list[bool]]:
    """Claude Code's assistant turns, a message's blocks merged by its id, and whether each tool
    result is an InputValidationError."""
    turns: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    results: list[bool] = []
    for i, o in enumerate(objs):
        msg = o.get("message")
        if not isinstance(msg, dict):
            continue
        if o.get("type") == "assistant" and isinstance(msg.get("content"), list):
            key = msg.get("id") if isinstance(msg.get("id"), str) else f"line-{i}"
            if key not in turns:
                turns[key] = {"text": [], "calls": [], "at": _timestamp(o.get("timestamp"))}
                order.append(key)
            for block in msg["content"]:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    turns[key]["text"].append(block["text"])
                elif block.get("type") == "tool_use":
                    turns[key]["calls"].append(f"{block.get('name')} {json.dumps(block.get('input'), sort_keys=True)}")
        elif o.get("type") == "user" and isinstance(msg.get("content"), list):
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.append(INVALID_CALL in _text_of(block.get("content")))
    out = [Turn(text="\n".join(t["text"]), calls="\n".join(t["calls"]), at=t["at"])
           for t in (turns[k] for k in order)]
    return out, results


def codex_turns(objs: list[dict[str, Any]]) -> tuple[list[Turn], list[bool]]:
    """Codex's assistant turns: each assistant message and each tool call. Codex reports no
    InputValidationError, so its results are not counted."""
    out = []
    for o in objs:
        p = o.get("payload")
        if o.get("type") != "response_item" or not isinstance(p, dict):
            continue
        at = _timestamp(o.get("timestamp"))
        if p.get("type") == "message" and p.get("role") == "assistant":
            out.append(Turn(text=_text_of(p.get("content")), calls="", at=at))
        elif p.get("type") in ("function_call", "custom_tool_call"):
            args = p.get("arguments") if p.get("type") == "function_call" else p.get("input")
            out.append(Turn(text="", calls=f"{p.get('name')} {args}", at=at))
    return out, []


def judge(turns: list[Turn], results: list[bool]) -> tuple[bool, str, int]:
    """Whether the last ``WINDOW`` turns and results show a derailed session: (derailed, signal,
    count)."""
    if len(turns) < WINDOW:
        return False, "", 0
    window = turns[-WINDOW:]
    # Only turns with prose count: an agent polling a long job repeats one tool call by design.
    shapes = Counter((" ".join(t.text.split()), t.calls) for t in window if t.text.strip())
    top = shapes.most_common(1)[0][1] if shapes else 0
    if top >= REPEAT_MIN:
        return True, "repeat", top
    invalid = sum(results[-WINDOW:])
    if invalid >= INVALID_MIN:
        return True, "invalid_calls", invalid
    special = sum(any(tok in t.text for tok in SPECIAL_TOKENS) for t in window)
    if special >= SPECIAL_MIN:
        return True, "special_tokens", special
    return False, "", 0


def log_bytes(agent_logs: Path) -> int:
    """Bytes of the regular files under *agent_logs*, not following links, among the first
    ``transcript.MAX_ENTRIES`` entries its listings hand out in all; 0 when *agent_logs* is itself
    a link. The walk keeps its own stack."""
    if agent_logs.is_symlink():
        return 0
    total = 0
    budget = [transcript.MAX_ENTRIES]
    pending = [Path(agent_logs)]
    while pending and budget[0] > 0:
        for entry in _listed(pending.pop(), budget):
            try:
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    total += entry.stat(follow_symlinks=False).st_size
            except OSError:
                continue
    return total


def check(agent_logs: Path, adapter: HarnessAdapter) -> Health:
    """Read the agent's newest transcript, as *adapter*'s harness writes it, and judge it. A
    transcript that cannot be read is no evidence of anything: the check reports a healthy
    session with no turns."""
    path = newest_transcript(agent_logs, adapter)
    turns: list[Turn] = []
    results: list[bool] = []
    if path is not None:
        unreadable = 0
        try:
            objs, unreadable = _objects(tail_lines(path))
            turns, results = adapter.transcript_turns(objs)
        except Exception as e:  # noqa: BLE001 -- the agent wrote the transcript: no evidence
            log.warning("the transcript %s could not be read as turns, so it is no evidence: %r", path, e)
            turns, results = [], []
        if unreadable:
            log.warning("the transcript %s had %d lines that could not be read; they are no evidence", path,
                        unreadable)
    derailed, signal, count = judge(turns, results)
    return Health(derailed=derailed, signal=signal, count=count, turns=len(turns),
                  last_turn_at=turns[-1].at if turns else None,
                  log_bytes=log_bytes(agent_logs))
