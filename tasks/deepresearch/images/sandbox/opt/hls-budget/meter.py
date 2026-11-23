"""Live budget metering from Claude Code usage. Pure stdlib: baked into the sealed
agent image and imported by both the in-sandbox MCP server and the host watchdog."""
from __future__ import annotations
import glob
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Prices:
    """Per-token USD prices for the build model (resolved once at launch)."""
    input: float
    output: float
    cache_read: float
    cache_write: float


@dataclass(frozen=True)
class TurnUsage:
    """Normalized per-assistant-turn token counts."""
    input: int
    output: int
    cache_read: int
    cache_write: int


@dataclass(frozen=True)
class Spent:
    """Current spend across the three budget dimensions. cost_usd is None when prices
    are unavailable."""
    cost_usd: float | None
    wallclock_sec: float
    turns: int


def _usage_from_message(msg: dict) -> TurnUsage | None:
    u = (msg or {}).get("usage")
    if not isinstance(u, dict):
        return None
    return TurnUsage(
        input=int(u.get("input_tokens", 0) or 0),
        output=int(u.get("output_tokens", 0) or 0),
        cache_read=int(u.get("cache_read_input_tokens", 0) or 0),
        cache_write=int(u.get("cache_creation_input_tokens", 0) or 0),
    )


def _parse_usages(path: str) -> list[tuple[str | None, TurnUsage]]:
    """Return ``(message_id, TurnUsage)`` for every assistant node in a JSONL transcript.

    Claude Code writes its session transcript as a tree with one node per content block:
    a single billed assistant response (one ``message.id``) whose ``thinking`` / ``text`` /
    ``tool_use`` blocks each become a separate node, and the response-level ``usage`` is
    stamped on every one. The same ``message.id`` recurs (e.g. 62 nodes for 31 ids), so
    callers must dedupe by ``message.id`` -- summing every node counts each billed response
    once per block (~2x). The id is Claude Code's ``message.id`` (``msg_...``) or None.
    """
    out: list[tuple[str | None, TurnUsage]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "assistant":
                    msg = ev.get("message", {}) or {}
                    tu = _usage_from_message(msg)
                    if tu is not None:
                        out.append((msg.get("id"), tu))
    except OSError:
        pass
    return out


def _first_timestamp(path: str) -> float | None:
    """Epoch seconds of the first event carrying a parseable ISO-8601 ``timestamp`` in a
    JSONL session file, or None. Session transcripts are append-only, so the first
    timestamped line is the earliest event in that file -- reading further would only cost
    I/O. datetime.fromisoformat parses the trailing ``Z`` (Python 3.11+). Any read/parse
    failure (missing field, bad JSON, malformed date) yields None for this file, never an
    exception: a broken transcript degrades the budget rather than crashing it."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = ev.get("timestamp")
                if not isinstance(ts, str):
                    continue
                try:
                    return datetime.fromisoformat(ts).timestamp()
                except ValueError:
                    return None
    except OSError:
        pass
    return None


class TranscriptSource:
    """Reads Claude Code's session transcript (sandbox-local).

    Claude Code writes the transcript to ``<config>/projects/<slug>/<id>.jsonl`` where
    ``<slug>`` derives from the agent's working directory (e.g. ``/workspace`` becomes
    ``-workspace``). We search both ``<dir>/*.jsonl`` and one level of subdirs
    ``<dir>/*/*.jsonl`` so the transcript is found whether ``directory`` points at the
    slug dir itself or its stable parent -- independent of the derived slug name.
    """
    def __init__(self, directory: str) -> None:
        self.directory = directory

    def _session_files(self) -> list[str]:
        # All of this build's session files, not just the newest: a long build can span
        # multiple <id>.jsonl (context compaction / crash-and-resume). The 2-level glob
        # (projects/<slug>/<id>.jsonl) excludes subagent transcripts at
        # projects/<slug>/subagents/<id>.jsonl (one level deeper), matching harbor's own
        # session discovery, so subagent spend is attributed to the subagent, not this build.
        return sorted(glob.glob(os.path.join(self.directory, "*.jsonl")) +
                      glob.glob(os.path.join(self.directory, "*", "*.jsonl")))

    def read(self) -> list[tuple[str | None, TurnUsage]]:
        # Reporting only the newest file would reset the agent's spend; BudgetMeter dedupes
        # by message.id, so concatenating every session file is safe.
        out: list[tuple[str | None, TurnUsage]] = []
        for f in self._session_files():
            out.extend(_parse_usages(f))
        return out

    def earliest_timestamp(self) -> float | None:
        # Epoch seconds of the earliest event across this build's session files, or None
        # when no transcript or no parseable timestamp exists. Anchors the wall-clock budget
        # to the agent's first action (agent time) rather than harness setup time -- see
        # BudgetMeter.spent. Reads only the first timestamped line of each file (append-only),
        # so it stays cheap even for a long build spanning many files.
        stamps = [t for t in (_first_timestamp(f) for f in self._session_files()) if t is not None]
        return min(stamps) if stamps else None


class NoTranscript:
    """A usage source with nothing to read, for a harness that writes no Claude Code
    transcript.

    Reports no turns, no spend and no anchor, so a ``BudgetMeter`` built on it measures
    wall-clock from the ``start_epoch`` it was given no matter what is on disk. That is
    the difference from pointing a ``TranscriptSource`` at a directory expected to stay
    empty: the agent image bakes the Claude Code CLI, so a session an agent starts in its
    own shell would otherwise re-anchor the clock to that session.
    """
    def read(self) -> list[tuple[str | None, TurnUsage]]:
        return []

    def earliest_timestamp(self) -> float | None:
        return None


class BudgetMeter:
    """Computes current spend from a usage source. Cheap to call repeatedly."""
    def __init__(self, start_epoch: float, prices: Prices | None, source) -> None:
        self.start_epoch = start_epoch
        self.prices = prices
        self.source = source

    def spent(self) -> Spent:
        records = self.source.read()   # list[(message_id, TurnUsage)]
        # Dedupe by message.id: the same billed response can appear as multiple
        # transcript nodes (see _parse_usages), so summing every node would double- or
        # triple-count turns and cost. A None id (older formats) is treated as unique.
        dedup: dict = {}
        for i, (mid, tu) in enumerate(records):
            dedup[mid if mid is not None else f"__anon_{i}"] = tu
        usages = list(dedup.values())

        cost: float | None
        if self.prices is None:
            cost = None
        else:
            p = self.prices
            cost = sum(u.input * p.input + u.output * p.output
                       + u.cache_read * p.cache_read + u.cache_write * p.cache_write
                       for u in usages)

        # Wall-clock is anchored to the agent's first transcript event (agent time), not
        # harness setup time. The orchestrator stamps start_epoch before sandbox setup
        # (image resolve, container create, weight pull: tens of minutes for a large model)
        # because HLS_BUDGET_B64 must be baked at container creation; measuring from it
        # would spend that much of the budget before the agent ran a single turn. A missing
        # or broken transcript falls back to start_epoch instead of crashing (server.py
        # guards get_budget too).
        anchor = self.source.earliest_timestamp()
        if anchor is None:
            anchor = self.start_epoch
        return Spent(cost_usd=cost, wallclock_sec=time.time() - anchor, turns=len(usages))
