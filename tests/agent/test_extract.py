"""The extraction role's program (``hls.agent.extract``): harbor's reading of the agent's
trajectory and spend from the transcript's copy, run as the stage runs it, in a process of its
own."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hls.agent.extract import TRAJECTORY_FILE, ExtractionRecord, Usage
from hls.arms import ArmError
from hls.store.digest import sha256_file
from hls.store.manifest import read_outputs_report

REPO = Path(__file__).resolve().parents[2]
SESSION = "4f2a1c5e-0000-4000-8000-000000000000"


def _claude_copy(root: Path) -> Path:
    """A transcript copy of a Claude Code session as the CLI writes it: its session file, three
    turns each of 10 input tokens, 4 of them read from the provider's cache besides, and 5 output
    tokens, and its stream with the result event that carries the run's cost."""
    logs = root / "logs"
    projects = logs / "sessions" / "projects" / "-workspace"
    projects.mkdir(parents=True)
    lines = []
    for i in range(3):
        lines.append({"type": "assistant", "sessionId": SESSION, "timestamp": f"2026-10-02T00:00:0{i}Z",
                      "message": {"id": f"m-{i}", "role": "assistant", "model": "claude-opus-4-8",
                                  "content": [{"type": "text", "text": f"turn {i}"}],
                                  "usage": {"input_tokens": 10, "cache_read_input_tokens": 4,
                                            "output_tokens": 5}}})
    (projects / f"{SESSION}.jsonl").write_text("".join(json.dumps(line) + "\n" for line in lines))
    (logs / "claude-code.txt").write_text(json.dumps({"type": "result", "subtype": "success", "session_id": SESSION,
                                                      "total_cost_usd": 0.5}) + "\n")
    return logs


def _codex_copy(root: Path) -> Path:
    """A transcript copy of a codex session: its rollout."""
    logs = root / "logs"
    day = logs / "sessions" / "2026" / "10" / "02"
    day.mkdir(parents=True)
    rows = [{"type": "turn_context", "payload": {"model": "gpt-5.5"}},
            {"type": "event_msg", "payload": {"type": "task_started", "model_context_window": 258400}},
            {"type": "response_item", "timestamp": "2026-10-02T00:00:00Z",
             "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "hi"}]}}]
    (day / "rollout-2026-10-02T00-00-00-0199aaaa-0000-7000-8000-000000000001.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows))
    return logs


def _run(root: Path, logs: Path, harness: str = "claude-code", model: str = "claude-opus-4-8"):
    out = root / "outputs" / "transcript"
    out.mkdir(parents=True)
    record, report = root / "record.json", root / "outputs.json"
    done = subprocess.run([sys.executable, "-m", "hls.agent.extract", "--harness", harness, "--model", model,
                           "--logs", str(logs), "--out", str(out), "--record", str(record), "--outputs-out",
                           str(report)], capture_output=True, text=True, cwd=root,
                          env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO}"})
    return done, out, record, report


@pytest.mark.parametrize("harness, copy", [("claude-code", _claude_copy), ("codex", _codex_copy)])
def test_harbor_reads_a_trajectory_from_the_copy_and_the_role_reports_it_by_its_digest(tmp_path, harness, copy):
    done, out, record, report = _run(tmp_path, copy(tmp_path), harness)
    assert done.returncode == 0, done.stderr
    trajectory = out / TRAJECTORY_FILE
    read = ExtractionRecord.model_validate_json(record.read_text())
    assert read.trajectory and trajectory.is_file() and json.loads(trajectory.read_text())["steps"]
    [(name, reported)] = read_outputs_report(report).items()
    assert (name, reported.sha256, reported.size) == ("transcript/trajectory.json", sha256_file(trajectory)[0],
                                                      trajectory.stat().st_size)
    if harness == "claude-code":
        # Harbor counts a turn's cache reads among its input tokens (14 a turn) and apart (4 a turn).
        assert read.usage == Usage(n_input_tokens=42, n_cache_tokens=12, n_output_tokens=15, cost_usd=0.5)


def test_a_trajectory_past_its_bound_is_withheld_and_the_spend_still_read(tmp_path, monkeypatch):
    """Harbor writes the trajectory whole; one past ``TRAJECTORY_MAX_BYTES`` is removed, reported
    as no output, and the record says why, beside the spend harbor read."""
    import hls.agent.extract as extract_mod

    monkeypatch.setattr(extract_mod, "TRAJECTORY_MAX_BYTES", 64)
    out = tmp_path / "outputs" / "transcript"
    out.mkdir(parents=True)
    read = extract_mod.extract("claude-code", "claude-opus-4-8", _claude_copy(tmp_path), out)
    assert not read.trajectory and not (out / TRAJECTORY_FILE).exists()
    assert read.trajectory_withheld.startswith("harbor's trajectory took ")
    assert read.trajectory_withheld.endswith(" bytes, past the 64 the build keeps")
    assert read.usage is not None and read.usage.cost_usd == 0.5


def test_the_trajectory_bound_is_a_margin_over_the_largest_measured():
    """A trajectory of 17,004,142 bytes, which harbor writes from 62 MB of codex logs, fits the
    bound with a margin of more than 15 times."""
    from hls.agent.extract import TRAJECTORY_MAX_BYTES

    assert TRAJECTORY_MAX_BYTES == 256 * 2**20 > 15 * 17_004_142


def test_a_copy_with_no_session_reads_no_trajectory_and_reports_nothing(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "claude-code.txt").write_text("not a stream\n")
    done, out, record, report = _run(tmp_path, logs)
    assert done.returncode == 0, done.stderr
    assert ExtractionRecord.model_validate_json(record.read_text()) == ExtractionRecord(
        trajectory=False, trajectory_withheld=None, usage=None)
    assert read_outputs_report(report) == {} and not (out / TRAJECTORY_FILE).exists()


def test_a_failure_of_harbors_reading_ends_the_role_with_its_reason_and_writes_no_record(tmp_path):
    """Harbor's own readers skip what they cannot parse; what else fails, here an output directory
    the trajectory cannot be reported from, ends the role non-zero with the reason on stderr."""
    logs = _claude_copy(tmp_path)
    out = tmp_path / "elsewhere" / "transcript"
    out.mkdir(parents=True)
    done = subprocess.run([sys.executable, "-m", "hls.agent.extract", "--harness", "claude-code", "--model", "m",
                           "--logs", str(logs), "--out", str(out), "--record", str(tmp_path / "record.json"),
                           "--outputs-out", str(tmp_path / "missing" / "outputs.json")],
                          capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": f"{REPO / 'src'}{os.pathsep}{REPO}"})
    assert done.returncode == 1 and done.stderr.strip().splitlines()[-1].startswith("hls-extract: ")
    assert not (tmp_path / "record.json").exists()


def test_a_harness_no_adapter_runs_reads_nothing(tmp_path):
    """The role reads a harness's logs only through its adapter; another name is refused, never
    read as a known harness's."""
    import hls.agent.extract as extract_mod

    out = tmp_path / "outputs" / "transcript"
    out.mkdir(parents=True)
    with pytest.raises(ArmError, match="no adapter runs harness 'terminus-2'"):
        extract_mod.extract("terminus-2", "m", _codex_copy(tmp_path), out)
    assert not (out / TRAJECTORY_FILE).exists()
    done, _, record, _ = _run(tmp_path / "role", _codex_copy(tmp_path / "role"), "terminus-2")
    assert done.returncode != 0 and "invalid choice: 'terminus-2'" in done.stderr and not record.exists()
