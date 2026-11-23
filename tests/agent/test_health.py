"""The health check: the turns it reads from each harness's transcript, and the three signals of a
derailed session, at their thresholds."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from hls.agent import health, transcript
from hls.arms import ClaudeCodeAdapter, CodexAdapter

CLAUDE_CODE = ClaudeCodeAdapter()
CODEX = CodexAdapter()


def _claude_line(i: int, text: str, *, tool: str = "true", msg_id: str | None = None) -> list[dict]:
    mid = msg_id or f"m{i}"
    # Claude Code writes one line per content block of a message.
    return [{"type": "assistant", "timestamp": f"2026-10-02T00:00:{i:02d}Z",
             "message": {"id": mid, "content": [{"type": "text", "text": text}]}},
            {"type": "assistant", "message": {"id": mid, "content": [{"type": "tool_use", "name": "Bash",
                                                                       "input": {"command": tool}}]}}]


def _result(error: bool) -> dict:
    content = "<tool_use_error>InputValidationError: could not parse" if error else [{"type": "text", "text": "ok"}]
    return {"type": "user", "message": {"content": [{"type": "tool_result", "content": content, "is_error": error}]}}


def _write_claude(logs: Path, lines: list[dict], name: str = "s1") -> Path:
    path = logs / "sessions" / "projects" / "-workspace" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


def _turns(n: int, text=lambda i: f"step {i}", tool=lambda i: f"cmd {i}", error=lambda i: False) -> list[dict]:
    out = []
    for i in range(n):
        out += [*_claude_line(i, text(i), tool=tool(i)), _result(error(i))]
    return out


def test_a_working_session_is_healthy_and_reports_its_last_turn(tmp_path):
    _write_claude(tmp_path, _turns(30))
    h = health.check(tmp_path, CLAUDE_CODE)
    assert (h.derailed, h.turns, h.last_turn_at) == (False, 30, "2026-10-02T00:00:29Z")
    assert h.log_bytes == sum(p.stat().st_size for p in tmp_path.rglob("*") if p.is_file())


def test_blocks_of_one_message_are_one_turn():
    objs = _claude_line(0, "hello", tool="ls")
    [turn], results = health.claude_turns(objs)
    assert turn.text == "hello" and turn.calls == 'Bash {"command": "ls"}' and results == []


@pytest.mark.parametrize("count, derailed", [(health.REPEAT_MIN - 1, False), (health.REPEAT_MIN, True)])
def test_a_repeated_turn_with_prose_derails_at_the_threshold(tmp_path, count, derailed):
    lines = _turns(20, text=lambda i: "I will keep going." if i < count else f"step {i}",
                   tool=lambda i: "true" if i < count else f"cmd {i}")
    _write_claude(tmp_path, lines)
    h = health.check(tmp_path, CLAUDE_CODE)
    assert h.derailed is derailed and (h.signal == "repeat") is derailed


def test_a_repeated_tool_call_without_prose_is_polling_not_a_derailment(tmp_path):
    lines = []
    for i in range(20):
        lines += [{"type": "assistant", "message": {"id": f"m{i}", "content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": "sleep 595; echo done"}}]}}, _result(False)]
    _write_claude(tmp_path, lines)
    assert not health.check(tmp_path, CLAUDE_CODE).derailed


@pytest.mark.parametrize("count, derailed", [(health.INVALID_MIN - 1, False), (health.INVALID_MIN, True)])
def test_invalid_tool_calls_derail_at_the_threshold(tmp_path, count, derailed):
    _write_claude(tmp_path, _turns(20, error=lambda i: i < count))
    h = health.check(tmp_path, CLAUDE_CODE)
    assert h.derailed is derailed and (h.signal == "invalid_calls") is derailed


@pytest.mark.parametrize("count, derailed", [(health.SPECIAL_MIN - 1, False), (health.SPECIAL_MIN, True)])
def test_control_tokens_in_the_models_own_prose_derail_at_the_threshold(tmp_path, count, derailed):
    _write_claude(tmp_path, _turns(20, text=lambda i: f"step {i} [gMASK]<sop>" if i < count else f"step {i}"))
    h = health.check(tmp_path, CLAUDE_CODE)
    assert h.derailed is derailed and (h.signal == "special_tokens") is derailed


def test_control_tokens_in_tool_calls_do_not_count(tmp_path):
    _write_claude(tmp_path, _turns(20, tool=lambda i: f"grep '<|im_end|>' tokenizer_config.json {i}"))
    assert not health.check(tmp_path, CLAUDE_CODE).derailed


def test_nothing_is_judged_before_a_full_window(tmp_path):
    _write_claude(tmp_path, _turns(health.WINDOW - 1, error=lambda i: True))
    h = health.check(tmp_path, CLAUDE_CODE)
    assert not h.derailed and h.turns == health.WINDOW - 1


@pytest.mark.parametrize("old", ["invalid", "repeat", "special"])
def test_only_the_last_window_counts(tmp_path, old):
    """A session that went wrong and recovered: whatever happened before the last window does not
    count."""
    early = {"invalid": {"error": lambda i: i < 20},
             "repeat": {"text": lambda i: "Again." if i < 20 else f"step {i}",
                        "tool": lambda i: "true" if i < 20 else f"cmd {i}"},
             "special": {"text": lambda i: f"step {i} <|im_end|>" if i < 20 else f"step {i}"}}[old]
    _write_claude(tmp_path, _turns(40, **early))
    assert not health.check(tmp_path, CLAUDE_CODE).derailed


def test_the_newest_session_is_read_and_subagents_are_not(tmp_path):
    old = _write_claude(tmp_path, _turns(20, error=lambda i: True), name="old")
    os.utime(old, (time.time() - 100, time.time() - 100))
    _write_claude(tmp_path, _turns(20), name="new")
    sub = tmp_path / "sessions" / "projects" / "-workspace" / "new" / "subagents" / "a.jsonl"
    sub.parent.mkdir(parents=True)
    sub.write_text("".join(json.dumps(x) + "\n" for x in _turns(20, error=lambda i: True)))
    assert health.newest_transcript(tmp_path, CLAUDE_CODE).name == "new.jsonl"
    assert not health.check(tmp_path, CLAUDE_CODE).derailed


def test_codex_turns_are_its_messages_and_tool_calls(tmp_path):
    day = tmp_path / "sessions" / "2026" / "10" / "02"
    day.mkdir(parents=True)
    lines = []
    for i in range(20):
        lines.append({"type": "response_item", "timestamp": f"2026-10-02T00:00:{i:02d}Z", "payload": {
            "type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Same again."}]}})
    (day / "rollout-x.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    h = health.check(tmp_path, CODEX)
    assert h.derailed and h.signal == "repeat" and h.last_turn_at == "2026-10-02T00:00:19Z"
    calls, _ = health.codex_turns([{"type": "response_item", "payload": {"type": "function_call", "name": "shell",
                                                                         "arguments": "{\"cmd\": \"ls\"}"}}])
    assert calls[0].calls == 'shell {"cmd": "ls"}' and calls[0].text == ""


def test_a_missing_or_garbled_transcript_is_no_evidence(tmp_path):
    assert health.check(tmp_path, CLAUDE_CODE) == health.Health(
        derailed=False, signal="", count=0, turns=0, last_turn_at=None, log_bytes=0)
    path = _write_claude(tmp_path, [])
    path.write_text("not json\n{\n")
    assert not health.check(tmp_path, CLAUDE_CODE).derailed


def test_the_tail_drops_a_partial_first_line(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text("first line\nsecond\nthird\n")
    assert health.tail_lines(path, max_bytes=15) == ["second", "third"]  # "e\nsecond\nthird\n"
    assert health.tail_lines(path) == ["first line", "second", "third"]


def test_log_bytes_count_regular_files_and_not_links(tmp_path):
    (tmp_path / "a.txt").write_text("12345")
    os.symlink("/etc/hostname", tmp_path / "link")
    assert health.log_bytes(tmp_path) == 5


def test_a_fifo_in_place_of_the_newest_transcript_is_skipped_without_blocking(tmp_path):
    """The agent writes these files while the runner reads them: a fifo named like a newer
    transcript must neither be chosen nor block the read."""
    _write_claude(tmp_path, _turns(30))
    fifo = tmp_path / "sessions" / "projects" / "-workspace" / "s9.jsonl"
    os.mkfifo(fifo)
    os.utime(fifo, (time.time() + 60, time.time() + 60), follow_symlinks=False)
    h = health.check(tmp_path, CLAUDE_CODE)
    assert h.turns == 30
    with pytest.raises(OSError, match="not a regular file"):
        health.tail_lines(fifo)


def test_links_the_agent_planted_are_never_followed(tmp_path):
    """A transcript, a session directory or the logs directory itself that is a link reads as
    nothing: the runner does not leave the agent's logs."""
    outside = tmp_path / "outside"
    _write_claude(outside, _turns(30))
    logs = tmp_path / "logs"
    (logs / "sessions" / "projects").mkdir(parents=True)
    os.symlink(outside / "sessions" / "projects" / "-workspace", logs / "sessions" / "projects" / "-workspace")
    assert health.newest_transcript(logs, CLAUDE_CODE) is None
    project = logs / "sessions" / "projects" / "-real"
    project.mkdir()
    os.symlink(outside / "sessions" / "projects" / "-workspace" / "s1.jsonl", project / "s2.jsonl")
    assert health.newest_transcript(logs, CLAUDE_CODE) is None
    with pytest.raises(OSError):
        health.tail_lines(project / "s2.jsonl")
    os.symlink(outside, tmp_path / "logs-link")
    assert health.check(tmp_path / "logs-link", CLAUDE_CODE).turns == 0
    assert health.log_bytes(tmp_path / "logs-link") == 0


def test_log_bytes_look_at_a_bounded_number_of_entries(tmp_path, monkeypatch):
    """The heartbeat reads the logs' size in the loop that enforces the hard limit: an agent that
    fills its logs with files cannot make the read long."""
    monkeypatch.setattr(transcript, "MAX_ENTRIES", 3)
    (tmp_path / "d").mkdir()
    for i in range(6):
        (tmp_path / "d" / f"{i}.txt").write_text("x")
    assert health.log_bytes(tmp_path) == 2


@pytest.mark.parametrize("where", ["logs", "sessions"])
def test_the_heartbeats_reads_list_no_more_than_their_bound_of_an_agent_written_directory(tmp_path, monkeypatch,
                                                                                           where):
    """The size reading and the newest-transcript search run in the runner's loop at every
    heartbeat, over directories the agent writes, which may hold millions of names. Neither hands
    out more names than the transcript copy's ``MAX_ENTRIES`` (with one more read to tell the rest is there)."""
    from hls.agent import listing
    from tests.agent.test_transcript import _CountedScandir

    monkeypatch.setattr(transcript, "MAX_ENTRIES", 5)
    counted = _CountedScandir(listing.os.scandir)
    monkeypatch.setattr(listing.os, "scandir", counted)
    many = tmp_path / ("many" if where == "logs" else "sessions/projects/-workspace")
    many.mkdir(parents=True)
    for i in range(40):
        (many / f"{i:02d}.jsonl").write_text("{}\n")
    if where == "logs":
        health.log_bytes(tmp_path)
    else:
        health.newest_transcript(tmp_path, CLAUDE_CODE)
    assert 0 < counted.read <= 5 + 3


def test_a_line_the_parser_cannot_take_is_unreadable_and_the_rest_judged(tmp_path, monkeypatch, caplog):
    """The agent writes its transcripts: a line nested past the decoder's depth, or longer than
    ``LINE_MAX_CHARS``, is no evidence, never an error. It is named in a warning, and the other
    lines are read."""
    monkeypatch.setattr(health, "LINE_MAX_CHARS", 60_000)
    path = _write_claude(tmp_path, _turns(30))
    with open(path, "a") as f:
        f.write("[" * 50_000 + "\n")
        f.write(json.dumps({"type": "assistant", "message": {"id": "long", "content": [
            {"type": "text", "text": "x" * 70_000}]}}) + "\n")
    with caplog.at_level("WARNING", logger="hls.agent.health"):
        h = health.check(tmp_path, CLAUDE_CODE)
    assert h.turns == 30 and not h.derailed
    assert any("2 lines" in r.getMessage() and str(path) in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("stamp, kept", [
    ("2026-10-02T00:00:01.123Z", True), ("2026-10-02T00:00:01+00:00", True), ("\u00e9" * 100_000, False),
    ("not a time", False), ("2026-10-02T00:00:01Z" + "x" * 100, False)],
    ids=["iso-z", "iso-offset", "huge", "words", "long"])
def test_the_last_turns_time_is_a_bounded_timestamp_or_none(tmp_path, stamp, kept):
    """The last turn's time reaches every heartbeat event: one the agent wrote that is no short
    ISO 8601 timestamp is dropped."""
    lines = _turns(3)
    lines.append({"type": "assistant", "timestamp": stamp, "message": {"id": "last", "content": [
        {"type": "text", "text": "done"}]}})
    _write_claude(tmp_path, lines)
    assert health.check(tmp_path, CLAUDE_CODE).last_turn_at == (stamp if kept else None)


def test_a_transcript_the_turn_reading_cannot_take_is_no_evidence(tmp_path, monkeypatch, caplog):
    """Whatever reading the turns raises, the check says nothing is derailed and names the file."""
    path = _write_claude(tmp_path, _turns(30))

    def refuse(self, objs):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(ClaudeCodeAdapter, "transcript_turns", refuse)
    with caplog.at_level("WARNING", logger="hls.agent.health"):
        h = health.check(tmp_path, CLAUDE_CODE)
    assert (h.derailed, h.turns, h.last_turn_at) == (False, 0, None)
    assert any(str(path) in r.getMessage() for r in caplog.records)
