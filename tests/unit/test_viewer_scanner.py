"""Unit tests for hls.viewer.scanner.RolloutScanner -- TDD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from hls.viewer.scanner import RolloutScanner

# ---------------------------------------------------------------------------
# Verifier-kind detection -- token_level
# ---------------------------------------------------------------------------

TASK_RELATIVE_TASK = "sample_task"
TASK_RELATIVE_ROLLOUT = "20260620T050810Z-bffba8"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RUNS_DIR = Path(__file__).parent.parent.parent / "runs"
SAMPLE_TASK = "sample-token-level"
SAMPLE_ROLLOUT = "20260604T064710Z-b4dae3"
SAMPLE_EVAL = "terminal"
#: The committed rollout fixture these tests read. runs/ is gitignored, so it is ABSENT
#: in a fresh checkout -- and can also hold other content WITHOUT this fixture. Guard on the fixture itself, not merely runs/ existence, so the tests skip
#: cleanly when the fixture is absent regardless of other runs/ content.
_FIXTURE_DIR = RUNS_DIR / SAMPLE_TASK / SAMPLE_ROLLOUT

# New fixture with per_verifier_results diagnostics and 10 events (all token_match=True)
NEW_ROLLOUT = "20260619T154447Z-6cfcfe"


def _write_grading(rollout_dir: Path, eval_id: str, episode: dict[str, Any]) -> Path:
    """Seal one grading under *rollout_dir*, where the scanner reads episodes from."""
    d = rollout_dir / "evals" / eval_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "grading.json"
    p.write_text(json.dumps({"episode": episode, "source_rollout": rollout_dir.name}))
    return p


# ---------------------------------------------------------------------------
# list_rollouts -- sealed records
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_rollouts_includes_sealed() -> None:
    """list_rollouts must include the sealed sample-token-level b4dae3 fixture."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    rows = scanner.list_rollouts()
    ids = {r["rollout_id"] for r in rows}
    assert SAMPLE_ROLLOUT in ids


@pytest.mark.unit
def test_list_rollouts_has_task_id() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    rows = scanner.list_rollouts()
    target = next(r for r in rows if r["rollout_id"] == SAMPLE_ROLLOUT)
    assert target["task_id"] == SAMPLE_TASK


@pytest.mark.unit
def test_list_rollouts_sorted_newest_first() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    rows = scanner.list_rollouts()
    created_ats = [r.get("created_at") or 0 for r in rows]
    assert created_ats == sorted(created_ats, reverse=True)


# ---------------------------------------------------------------------------
# get_rollout -- sealed
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_rollout_sealed() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    rec = scanner.get_rollout(SAMPLE_TASK, SAMPLE_ROLLOUT)
    assert rec is not None
    assert rec["status"] == "ok"
    assert rec["rollout_id"] == SAMPLE_ROLLOUT


@pytest.mark.unit
def test_get_rollout_sealed_carries_its_gradings(tmp_path: Path) -> None:
    """rollout.json stores no episodes, so the read surface composes them from the
    rollout's gradings and the UI sees one record shape."""
    task_id, rollout_id = "sealed-task", "20260101T000000Z-sealed"
    rdir = tmp_path / task_id / rollout_id
    rdir.mkdir(parents=True)
    (rdir / "rollout.json").write_text(
        json.dumps(
            {
                "rollout_id": rollout_id,
                "task_id": task_id,
                "created_at": 1780000000.0,
                "status": "ok",
                "config": {"task_id": task_id},
                "build": None,
                "error": None,
            }
        )
    )
    _write_grading(
        rdir,
        "terminal",
        {
            "eval_id": "terminal",
            "checkpoint": "docker:img",
            "finished_at": 1780000100.0,
            "score": {"value": 0.7},
        },
    )

    rec = RolloutScanner(tmp_path).get_rollout(task_id, rollout_id)
    assert rec is not None
    assert "evals" not in json.loads((rdir / "rollout.json").read_text())  # not on disk
    assert [e["eval_id"] for e in rec["evals"]] == ["terminal"]
    assert rec["evals"][0]["score"]["value"] == 0.7


@pytest.mark.unit
@pytest.mark.parametrize("doc", ['{"episode": null}', "[]", '"a string"'])
def test_a_malformed_grading_does_not_take_down_the_listing(tmp_path: Path, doc: str) -> None:
    """One unreadable file must cost its own rollout at most. The scan walks every
    task directory with no try around it, so a raise here loses every listing under
    runs/ -- including the SSE stream the live list is built on."""
    from hls.viewer.scanner import episodes_of, gradings_of

    task_id = "malformed-task"
    good = tmp_path / task_id / "20260101T000000Z-good"
    bad = tmp_path / task_id / "20260101T000001Z-bad"
    for d in (good, bad):
        d.mkdir(parents=True)
        (d / "rollout.json").write_text(json.dumps({
            "rollout_id": d.name, "task_id": task_id, "created_at": 1780000000.0,
            "status": "ok", "config": {"task_id": task_id}, "build": None, "error": None,
        }))
    _write_grading(good, "terminal", {"eval_id": "terminal", "checkpoint": "docker:img",
                                      "finished_at": 1780000100.0, "score": {"value": 0.7}})
    (bad / "evals" / "g-1").mkdir(parents=True)
    (bad / "evals" / "g-1" / "grading.json").write_text(doc)

    gradings_of(bad)                    # the sort key is where it raised
    assert episodes_of(bad) == []       # no episode, whichever shape the doc took
    rows = RolloutScanner(tmp_path).list_rollouts()
    assert {r["rollout_id"] for r in rows} == {good.name, bad.name}
    assert next(r for r in rows if r["rollout_id"] == good.name)["score"] == 0.7


@pytest.mark.unit
def test_get_rollout_missing_returns_none() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    assert scanner.get_rollout("no-such-task", "no-such-id") is None


# ---------------------------------------------------------------------------
# trace_summary -- real fixture
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_summary_b4dae3() -> None:
    """Verify the exact rollup numbers for the b4dae3 fixture."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    ts = scanner.trace_summary(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL)
    assert ts is not None
    # total = 5 events
    assert ts.total == 5
    # token_match: all 5 passed per verdict.json
    assert ts.token_match == 5
    assert ts.token_match_total == 5
    # correct: leading-answer match -- only the last event ("A. 0" starts with gt "A") is correct
    assert ts.correct == 1
    assert ts.graded == 5
    # p99_ms from score.json breakdown.slo.p99_ms
    assert ts.p99_ms == pytest.approx(2265.7981689844746, abs=0.01)


@pytest.mark.unit
def test_trace_summary_missing_eval_returns_none() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    ts = scanner.trace_summary(SAMPLE_TASK, SAMPLE_ROLLOUT, "no-such-eval")
    assert ts is None


# ---------------------------------------------------------------------------
# trace_page -- real fixture
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_page_basic(tmp_path: Path) -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, offset=0, limit=2)
    assert page is not None
    assert page.total == 5
    assert page.offset == 0
    assert page.limit == 2
    assert len(page.rows) == 2


@pytest.mark.unit
def test_trace_page_row_fields() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, offset=0, limit=1)
    row = page.rows[0]
    # Required row fields
    assert "turn_index" in row
    assert "request_id" in row
    assert "latency_ms" in row
    assert row["latency_ms"] > 0


@pytest.mark.unit
def test_trace_page_only_incorrect() -> None:
    """only='incorrect' filters to rows where correct is False."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, offset=0, limit=10, only="incorrect")
    # 4 of 5 events have wrong leading-answer (the "A. 0" event is correct vs gt "A")
    assert page.total == 4
    for row in page.rows:
        assert row.get("correct") is False


@pytest.mark.unit
def test_trace_page_only_diverged() -> None:
    """only='diverged' -> token_match is False. But all matched here -> 0 rows."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, offset=0, limit=10, only="diverged")
    assert page.total == 0
    assert page.rows == []


@pytest.mark.unit
def test_trace_page_leading_answer_correct() -> None:
    """The 'A. 0' artifact row (gt='A') must be correct=True; earlier rows must be False."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, offset=0, limit=10)
    # Last row: artifact "A. 0" starts with gt "A" -> correct
    last_row = page.rows[-1]
    assert last_row["artifact_text"] == "A. 0"
    assert last_row["correct"] is True
    # First row: artifact "A" does not start with gt "B" -> incorrect
    first_row = page.rows[0]
    assert first_row["correct"] is False


@pytest.mark.unit
def test_trace_page_offset_slices() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page1 = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, offset=0, limit=3)
    page2 = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, offset=3, limit=3)
    assert len(page1.rows) == 3
    assert len(page2.rows) == 2
    # No overlap
    ids1 = {r["request_id"] for r in page1.rows}
    ids2 = {r["request_id"] for r in page2.rows}
    assert not ids1 & ids2


# ---------------------------------------------------------------------------
# list_files / read_file
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_files_b4dae3() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    files = scanner.list_files(SAMPLE_TASK, SAMPLE_ROLLOUT)
    paths = {f["path"] for f in files}
    assert "rollout.json" in paths
    assert "evals/terminal/trace.json" in paths


@pytest.mark.unit
def test_read_file_rollout_json() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    data, media = scanner.read_file(SAMPLE_TASK, SAMPLE_ROLLOUT, "rollout.json")
    assert media == "application/json"
    parsed = json.loads(data)
    assert parsed["rollout_id"] == SAMPLE_ROLLOUT


@pytest.mark.unit
def test_read_file_traversal_blocked() -> None:
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    with pytest.raises(ValueError, match="traversal"):
        scanner.read_file(SAMPLE_TASK, SAMPLE_ROLLOUT, "../../etc/passwd")


# ---------------------------------------------------------------------------
# only="" / only=None -- no-filter equivalence
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_page_only_empty_string_returns_all() -> None:
    """only='' (the frontend's 'All' value after URL decode) must return ALL rows, not filter to session ''."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page_none = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, only=None)
    page_empty = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, only="")
    page_ws = scanner.trace_page(SAMPLE_TASK, SAMPLE_ROLLOUT, SAMPLE_EVAL, only="   ")
    assert page_none.total == 5
    assert page_empty.total == 5, "only='' must behave as no-filter (all rows)"
    assert page_ws.total == 5, "only='   ' must behave as no-filter (all rows)"


# ---------------------------------------------------------------------------
# per_verifier_results diagnostics -- new fixture
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_page_new_fixture_total() -> None:
    """New fixture has 10 events; only=None/'' must both return total=10."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, NEW_ROLLOUT, SAMPLE_EVAL, offset=0, limit=20)
    assert page.total == 10
    assert len(page.rows) == 10


@pytest.mark.unit
def test_trace_page_new_fixture_only_empty_all() -> None:
    """only='' on the new fixture must return all 10 rows."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, NEW_ROLLOUT, SAMPLE_EVAL, offset=0, limit=20, only="")
    assert page.total == 10


@pytest.mark.unit
def test_trace_page_new_fixture_n_tokens_populated() -> None:
    """New fixture uses per_verifier_results; each row must have n_tokens populated."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(SAMPLE_TASK, NEW_ROLLOUT, SAMPLE_EVAL, offset=0, limit=20)
    for row in page.rows:
        assert row.get("n_tokens") is not None, f"n_tokens missing for row {row.get('request_id')}"
        assert row["n_tokens"] > 0, f"n_tokens must be positive, got {row['n_tokens']}"
        # All pass in this fixture, so worst_gap must be 0.0
        assert row.get("worst_gap") == 0.0, (
            f"worst_gap must be 0.0 for passing row, got {row.get('worst_gap')}"
        )
        assert row.get("token_match") is True


# ---------------------------------------------------------------------------
# Verifier-kind detection -- token_level
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_summary_token_level_kind() -> None:
    """token-level fixture must detect verifier_kind='token_level' and keep token_match populated."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    # Use the 10-event fixture, whose eval is the token-level TokenFaithfulnessVerifier
    ts = scanner.trace_summary(SAMPLE_TASK, NEW_ROLLOUT, SAMPLE_EVAL)
    assert ts is not None
    assert ts.verifier_kind == "token_level"
    assert ts.token_match is not None
    assert ts.token_match_total is not None
    assert ts.gate_passed is not None


# ---------------------------------------------------------------------------
# Verifier-kind detection -- task_relative
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_summary_task_relative_kind() -> None:
    """ReferenceRelativeVerifier verdict must yield verifier_kind='task_relative', artifact+reference values, token_match=None."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    ts = scanner.trace_summary(TASK_RELATIVE_TASK, TASK_RELATIVE_ROLLOUT, SAMPLE_EVAL)
    assert ts is not None
    assert ts.verifier_kind == "task_relative"
    assert ts.artifact_value is not None
    assert ts.reference_value is not None
    assert ts.tol is not None
    # token_match MUST be None for non-token_level verifiers
    assert ts.token_match is None
    assert ts.token_match_total is None
    assert ts.gate_passed is True  # aggregate_passed=true in the fixture


@pytest.mark.unit
def test_trace_page_task_relative_rows() -> None:
    """task_relative rows must carry predicted, oracle_predicted, oracle_correct."""
    if not _FIXTURE_DIR.exists():
        pytest.skip("runs/ fixture not present")
    scanner = RolloutScanner(RUNS_DIR)
    page = scanner.trace_page(TASK_RELATIVE_TASK, TASK_RELATIVE_ROLLOUT, SAMPLE_EVAL, offset=0, limit=10)
    assert page.total > 0
    for row in page.rows:
        # predicted is the artifact's answer letter
        assert "predicted" in row
        assert "oracle_predicted" in row
        assert "oracle_correct" in row
        # predicted should be a non-empty string (e.g. "A", "D")
        if row["predicted"] is not None:
            assert isinstance(row["predicted"], str)


# ---------------------------------------------------------------------------
# _p99_from_latencies -- index matches SLO gate (workload.py)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_p99_index_matches_slo_gate_n100() -> None:
    """n=100: SLO gate uses max(0, round(0.99*(100-1))) = round(98.01) = 98 (0-indexed).
    The 98th element of [0..99] sorted is 98. Scanner must return the same value."""
    from hls.viewer.scanner import _p99_from_latencies

    latencies = list(range(100))  # [0, 1, 2, ..., 99]
    result = _p99_from_latencies(latencies)
    # SLO gate index: max(0, int(round(0.99 * (100 - 1)))) = int(round(98.01)) = 98
    assert result == 98, f"expected index-98 value (98), got {result}"


@pytest.mark.unit
def test_p99_index_matches_slo_gate_n1000() -> None:
    """n=1000: SLO gate uses max(0, round(0.99*(1000-1))) = round(989.01) = 989."""
    from hls.viewer.scanner import _p99_from_latencies

    latencies = list(range(1000))  # [0, 1, ..., 999]
    result = _p99_from_latencies(latencies)
    assert result == 989, f"expected index-989 value (989), got {result}"


@pytest.mark.unit
def test_p99_index_n1() -> None:
    """n=1: max(0, round(0)) = 0; only element returned."""
    from hls.viewer.scanner import _p99_from_latencies

    assert _p99_from_latencies([42.0]) == 42.0


@pytest.mark.unit
def test_p99_empty_returns_none() -> None:
    from hls.viewer.scanner import _p99_from_latencies

    assert _p99_from_latencies([]) is None


# ---------------------------------------------------------------------------
# Verifier-kind detection -- task_absolute (synthetic)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_summary_task_absolute_kind(tmp_path: Path) -> None:
    """Synthetic MetricVerifier verdict must yield verifier_kind='task_absolute', artifact_value+min_value, token_match=None."""
    # Build a minimal eval dir with trace.json + verdict.json (MetricVerifier)
    task_id = "abs-task"
    rollout_id = "20260101T000000Z-aabbcc"
    eval_dir = tmp_path / task_id / rollout_id / "evals" / "terminal"
    eval_dir.mkdir(parents=True)

    # trace.json with 2 events
    trace = {
        "events": [
            {
                "request_id": "r1",
                "turn_index": 0,
                "session_id": None,
                "input": {"contents": [{"text": "q1"}]},
                "response": {"output": {"contents": [{"text": "predicted_val_1"}]}},
                "ground_truth": "expected_1",
                "t_send": 1.0,
                "t_recv": 1.5,
            },
            {
                "request_id": "r2",
                "turn_index": 1,
                "session_id": None,
                "input": {"contents": [{"text": "q2"}]},
                "response": {"output": {"contents": [{"text": "predicted_val_2"}]}},
                "ground_truth": "expected_2",
                "t_send": 2.0,
                "t_recv": 2.5,
            },
        ]
    }
    (eval_dir / "trace.json").write_text(json.dumps(trace))

    # verdict.json with MetricVerifier shape
    verdict = {
        "aggregate_passed": False,
        "diagnostics": {
            "0:MetricVerifier": {
                "value": 0.45,
                "min_value": 0.60,
            }
        },
    }
    (eval_dir / "verdict.json").write_text(json.dumps(verdict))

    scanner = RolloutScanner(tmp_path)
    ts = scanner.trace_summary(task_id, rollout_id, "terminal")
    assert ts is not None
    assert ts.verifier_kind == "task_absolute"
    assert ts.artifact_value == pytest.approx(0.45)
    assert ts.min_value == pytest.approx(0.60)
    assert ts.token_match is None
    assert ts.token_match_total is None
    assert ts.gate_passed is False


# ---------------------------------------------------------------------------
# Multi-verifier: VerifierSummary + verifier_cells
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_summary_multi_verifier(tmp_path: Path) -> None:
    """Synthetic 2-verifier verdict: TokenFaithfulnessVerifier PASS, ReferenceRelative FAIL.
    summary.verifiers must have 2 entries with correct name/kind/gate_passed."""

    task_id = "multi-v-task"
    rollout_id = "20260101T000000Z-multiv"
    eval_dir = tmp_path / task_id / rollout_id / "evals" / "terminal"
    eval_dir.mkdir(parents=True)

    trace = {
        "events": [
            {
                "request_id": "r1", "turn_index": 0, "session_id": None,
                "input": {"contents": [{"text": "q1"}]},
                "response": {"output": {"contents": [{"text": "A"}]}},
                "ground_truth": "A", "t_send": 1.0, "t_recv": 1.5, "error": None,
            },
            {
                "request_id": "r2", "turn_index": 1, "session_id": None,
                "input": {"contents": [{"text": "q2"}]},
                "response": {"output": {"contents": [{"text": "B"}]}},
                "ground_truth": "B", "t_send": 2.0, "t_recv": 2.5, "error": None,
            },
            {
                "request_id": "r3", "turn_index": 2, "session_id": None,
                "input": {"contents": [{"text": "q3"}]},
                "response": {"output": {"contents": [{"text": "C"}]}},
                "ground_truth": "C", "t_send": 3.0, "t_recv": 3.5, "error": None,
            },
        ]
    }
    (eval_dir / "trace.json").write_text(json.dumps(trace))

    # Gate diagnostics: rate_thresh = rate_multiplier * q_max plus the flip-count
    # curve; gate_passed = not (rate_reject or curve_reject or insufficient_trace).
    verdict = {
        "aggregate_passed": False,
        "diagnostics": {
            "0:TokenFaithfulnessVerifier": {
                "n_tokens": 500, "n_flips": 2, "flip_rate": 0.004,
                "max_gap": 0.05, "q_max": 0.01,
                "rate_multiplier": 2.0,
                "rate_ratio": 0.004 / (2.0 * 0.01),
                "min_tokens": 10, "insufficient_trace": False,
                "rate_reject": False,
                "flip_curve": [{"c": 0.5, "k": 4, "count": 0, "fired": False}],
                "curve_reject": False, "curve_fired_levels": [],
            },
            "1:ReferenceRelativeVerifier": {"artifact_value": 0.33, "reference_value": 0.66, "tol": 0.0},
        },
        "per_verifier_results": [
            # verifier 0: TokenFaithfulnessVerifier -- all 3 rows mirror the trace-level pass
            [
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 2, "n_flips": 0, "max_gap": 0.0}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 3, "n_flips": 0, "max_gap": 0.0}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 2, "n_flips": 0, "max_gap": 0.0}},
            ],
            # verifier 1: ReferenceRelativeVerifier -- all fail (predicted != expected)
            [
                {"passed": False, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "A", "expected": "D"}},
                {"passed": False, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "B", "expected": "C"}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "C", "expected": "C"}},
            ],
        ],
    }
    (eval_dir / "verdict.json").write_text(json.dumps(verdict))

    scanner = RolloutScanner(tmp_path)
    ts = scanner.trace_summary(task_id, rollout_id, "terminal")
    assert ts is not None

    # Must have exactly 2 verifiers
    assert len(ts.verifiers) == 2

    # Verifier 0: TokenFaithfulnessVerifier PASS (neither threshold rejected)
    v0 = ts.verifiers[0]
    assert v0.name == "TokenFaithfulnessVerifier"
    assert v0.kind == "token_level"
    assert v0.gate_passed is True
    assert v0.flip_rate == pytest.approx(0.004)
    assert v0.rate_ratio == pytest.approx(0.2)
    assert v0.curve_fired_levels == []
    assert v0.n_tokens == 500

    # Verifier 1: ReferenceRelativeVerifier FAIL (0.33 + 0.0 < 0.66)
    v1 = ts.verifiers[1]
    assert v1.name == "ReferenceRelativeVerifier"
    assert v1.kind == "task_relative"
    assert v1.gate_passed is False
    assert v1.artifact_value == pytest.approx(0.33)
    assert v1.reference_value == pytest.approx(0.66)


@pytest.mark.unit
def test_token_level_summary_reads_the_verdict_off_the_matrix() -> None:
    """The token-level gate is trace-level, so its diagnostics publish measurements and
    ceilings and no verdict of their own. The summary reads pass/fail off the
    per-verifier matrix, and reports which levels fired by comparing each row's count
    against the most permissive engine's enforced ceiling."""
    from hls.viewer.scanner import _build_verifier_summaries

    verdict = {
        "diagnostics": {
            "0:TokenFaithfulnessVerifier": {
                "n_tokens": 500, "flip_rate": 0.004, "rate_ratio": 0.2,
                "flip_curve": [
                    {"c": 0.5, "count": 2, "ceiling_per_engine": {"vllm": 1, "sglang": 4}},
                    {"c": 1.0, "count": 9, "ceiling_per_engine": {"vllm": 3, "sglang": 2}},
                ],
            },
        },
        "per_verifier_results": [[{"passed": False}, {"passed": False}]],
    }
    (v0,) = _build_verifier_summaries(verdict)
    assert v0.gate_passed is False
    # 0.5 is under the more permissive engine's ceiling of 4; 1.0 is over its 3.
    assert v0.curve_fired_levels == [1.0]


def test_token_level_summary_leaves_the_curve_unknown_without_rows() -> None:
    """No curve rows means unknown, not clean."""
    from hls.viewer.scanner import _build_verifier_summaries

    (v0,) = _build_verifier_summaries({
        "diagnostics": {"0:TokenFaithfulnessVerifier": {"n_tokens": 500}},
        "per_verifier_results": [[{"passed": True}]],
    })
    assert v0.gate_passed is True and v0.curve_fired_levels is None


@pytest.mark.unit
def test_trace_page_multi_verifier_cells(tmp_path: Path) -> None:
    """trace_page rows must carry verifier_cells with correct passed/detail per verifier."""
    task_id = "multi-v-task"
    rollout_id = "20260101T000000Z-multiv"
    eval_dir = tmp_path / task_id / rollout_id / "evals" / "terminal"
    eval_dir.mkdir(parents=True)

    trace = {
        "events": [
            {
                "request_id": "r1", "turn_index": 0, "session_id": None,
                "input": {"contents": [{"text": "q1"}]},
                "response": {"output": {"contents": [{"text": "A"}]}},
                "ground_truth": "A", "t_send": 1.0, "t_recv": 1.5, "error": None,
            },
            {
                "request_id": "r2", "turn_index": 1, "session_id": None,
                "input": {"contents": [{"text": "q2"}]},
                "response": {"output": {"contents": [{"text": "B"}]}},
                "ground_truth": "B", "t_send": 2.0, "t_recv": 2.5, "error": None,
            },
            {
                "request_id": "r3", "turn_index": 2, "session_id": None,
                "input": {"contents": [{"text": "q3"}]},
                "response": {"output": {"contents": [{"text": "C"}]}},
                "ground_truth": "C", "t_send": 3.0, "t_recv": 3.5, "error": None,
            },
        ]
    }
    (eval_dir / "trace.json").write_text(json.dumps(trace))

    verdict = {
        "aggregate_passed": False,
        "diagnostics": {
            "0:TokenFaithfulnessVerifier": {
                "n_tokens": 500, "n_flips": 2, "flip_rate": 0.004,
                "max_gap": 0.05, "q_max": 0.01,
                "rate_multiplier": 2.0,
                "rate_ratio": 0.004 / (2.0 * 0.01),
                "min_tokens": 10, "insufficient_trace": False,
                "rate_reject": False,
                "flip_curve": [{"c": 0.5, "k": 4, "count": 0, "fired": False}],
                "curve_reject": False, "curve_fired_levels": [],
            },
            "1:ReferenceRelativeVerifier": {"artifact_value": 0.33, "reference_value": 0.66, "tol": 0.0},
        },
        "per_verifier_results": [
            [
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 2, "n_flips": 0, "max_gap": 0.0}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 3, "n_flips": 0, "max_gap": 0.0}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 2, "n_flips": 0, "max_gap": 0.0}},
            ],
            [
                {"passed": False, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "A", "expected": "D"}},
                {"passed": False, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "B", "expected": "C"}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "C", "expected": "C"}},
            ],
        ],
    }
    (eval_dir / "verdict.json").write_text(json.dumps(verdict))

    scanner = RolloutScanner(tmp_path)
    page = scanner.trace_page(task_id, rollout_id, "terminal", offset=0, limit=10)
    assert page.total == 3

    for row in page.rows:
        cells = row.get("verifier_cells")
        assert cells is not None
        assert len(cells) == 2

    # Row 0: TokenFaithfulnessVerifier pass (match), ReferenceRelative fail (A!=D)
    row0_cells = page.rows[0]["verifier_cells"]
    assert row0_cells[0]["name"] == "TokenFaithfulnessVerifier"
    assert row0_cells[0]["passed"] is True
    assert row0_cells[0]["detail"] == "match"
    assert row0_cells[1]["name"] == "ReferenceRelativeVerifier"
    assert row0_cells[1]["passed"] is False
    assert "A" in row0_cells[1]["detail"]
    assert "D" in row0_cells[1]["detail"]

    # Row 2: both pass
    row2_cells = page.rows[2]["verifier_cells"]
    assert row2_cells[0]["passed"] is True
    assert row2_cells[1]["passed"] is True
    assert row2_cells[1]["detail"] == "C=C"


@pytest.mark.unit
def test_trace_page_only_failing_filter(tmp_path: Path) -> None:
    """only='failing' must return only rows where at least one verifier_cell failed."""
    task_id = "multi-v-task"
    rollout_id = "20260101T000000Z-multiv2"
    eval_dir = tmp_path / task_id / rollout_id / "evals" / "terminal"
    eval_dir.mkdir(parents=True)

    trace = {
        "events": [
            {"request_id": "r1", "turn_index": 0, "session_id": None,
             "input": {"contents": [{"text": "q1"}]},
             "response": {"output": {"contents": [{"text": "A"}]}},
             "ground_truth": "A", "t_send": 1.0, "t_recv": 1.5, "error": None},
            {"request_id": "r2", "turn_index": 1, "session_id": None,
             "input": {"contents": [{"text": "q2"}]},
             "response": {"output": {"contents": [{"text": "B"}]}},
             "ground_truth": "B", "t_send": 2.0, "t_recv": 2.5, "error": None},
        ]
    }
    (eval_dir / "trace.json").write_text(json.dumps(trace))

    verdict = {
        "aggregate_passed": False,
        "diagnostics": {
            "0:TokenFaithfulnessVerifier": {
                "n_tokens": 300, "n_flips": 1, "flip_rate": 0.0033,
                "max_gap": 0.05, "q_max": 0.01,
                "rate_multiplier": 2.0,
                "rate_ratio": 0.0033 / (2.0 * 0.01),
                "min_tokens": 10, "insufficient_trace": False,
                "rate_reject": False,
                "flip_curve": [{"c": 0.5, "k": 4, "count": 0, "fired": False}],
                "curve_reject": False, "curve_fired_levels": [],
            },
            "1:ReferenceRelativeVerifier": {"artifact_value": 0.33, "reference_value": 0.66, "tol": 0.0},
        },
        "per_verifier_results": [
            [
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 2, "n_flips": 0, "max_gap": 0.0}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"n_tokens": 3, "n_flips": 0, "max_gap": 0.0}},
            ],
            [
                {"passed": False, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "A", "expected": "D"}},
                {"passed": True, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {"predicted": "B", "expected": "B"}},
            ],
        ],
    }
    (eval_dir / "verdict.json").write_text(json.dumps(verdict))

    scanner = RolloutScanner(tmp_path)
    # only="failing" must return only r1 (ReferenceRelative failed)
    page = scanner.trace_page(task_id, rollout_id, "terminal", only="failing")
    assert page.total == 1
    assert page.rows[0]["request_id"] == "r1"


# ---------------------------------------------------------------------------
# Enrichment columns: index fast-path vs. rollout.json fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_list_rollouts_uses_enrichment_from_index_without_rollout_json(tmp_path: Path) -> None:
    """An index.jsonl row that already carries the four enrichment columns must be
    surfaced by list_rollouts WITHOUT reading rollout.json (which is absent here).
    """
    task_id = "enrich-task"
    rollout_id = "20260101T000000Z-enriched"
    task_dir = tmp_path / task_id
    task_dir.mkdir(parents=True)
    # The rollout subdir exists but has NO rollout.json
    (task_dir / rollout_id).mkdir()

    index_row = {
        "rollout_id": rollout_id,
        "created_at": 1780000000.0,
        "finished_at": 1780000100.0,
        "status": "ok",
        "score": 0.95,
        "gate_passed": True,
        "cost_usd": 0.01,
        "image_digests": {},
        "backend": "modal",
        "model_fingerprint": "fp-abc",
        "accelerator": "A100",
        "agent_implementer_model": "claude-opus-4",
    }
    (task_dir / "index.jsonl").write_text(json.dumps(index_row) + "\n")

    scanner = RolloutScanner(tmp_path)
    rows = scanner.list_rollouts()
    target = next((r for r in rows if r["rollout_id"] == rollout_id), None)
    assert target is not None, "row must be present"
    assert target["backend"] == "modal"
    assert target["model_fingerprint"] == "fp-abc"
    assert target["accelerator"] == "A100"
    assert target["agent_implementer_model"] == "claude-opus-4"


@pytest.mark.unit
def test_list_rollouts_old_index_row_falls_back_to_rollout_json(tmp_path: Path) -> None:
    """An index.jsonl row lacking enrichment columns must fall back to reading
    rollout.json to get the values via _enrich_summary."""
    task_id = "fallback-task"
    rollout_id = "20260101T000000Z-fallback"
    rdir = tmp_path / task_id / rollout_id
    rdir.mkdir(parents=True)

    # An index row with no enrichment columns
    old_row = {
        "rollout_id": rollout_id,
        "created_at": 1780000000.0,
        "finished_at": 1780000100.0,
        "status": "ok",
        "score": 0.8,
        "gate_passed": True,
        "cost_usd": 0.02,
        "image_digests": {},
    }
    (tmp_path / task_id / "index.jsonl").write_text(json.dumps(old_row) + "\n")

    # rollout.json with enrichment data
    rollout_rec = {
        "rollout_id": rollout_id,
        "task_id": task_id,
        "created_at": 1780000000.0,
        "finished_at": 1780000100.0,
        "status": "ok",
        "config": {
            "task_id": task_id,
            "spec": {
                "model": {"fingerprint": "legacy-model"},
                "hardware": {"accelerator": {"kind": "T4"}},
            },
            "image_digests": {},
            "rollout_overrides": {},
            "substrate": {},
            "container_env_public": {},
        },
        "build": None,
        "error": None,
    }
    (rdir / "rollout.json").write_text(json.dumps(rollout_rec))
    # The backend column is derived from the terminal grading's checkpoint.
    _write_grading(
        rdir,
        "terminal",
        {
            "eval_id": "terminal",
            "triggered_by": "harness",
            "checkpoint": "docker:sha256:abc123",
            "anchor": None,
            "started_at": 1780000050.0,
            "finished_at": 1780000100.0,
            "trace_path": "evals/terminal/trace.json",
            "verdict_summary": {"gate_passed": True},
            "score": {"value": 0.8},
            "container_logs": {},
            "error": None,
        },
    )

    scanner = RolloutScanner(tmp_path)
    rows = scanner.list_rollouts()
    target = next((r for r in rows if r["rollout_id"] == rollout_id), None)
    assert target is not None, "row must be present"
    # Enrichment must come from rollout.json via _enrich_summary
    assert target["backend"] == "docker"
    assert target["model_fingerprint"] == "legacy-model"
    assert target["accelerator"] == "T4"


@pytest.mark.unit
def test_list_rollouts_does_not_list_a_grading_as_a_rollout(tmp_path: Path) -> None:
    """A grading appends its own index row under its parent's rollout_id. Listing it
    would show two entries that look like one rollout with two different scores, both
    linking to the same detail page -- where the grading is already an episode."""
    task_id = "graded-task"
    rollout_id = "20260101T000000Z-graded"
    rdir = tmp_path / task_id / rollout_id
    rdir.mkdir(parents=True)
    (rdir / "rollout.json").write_text(json.dumps({
        "rollout_id": rollout_id, "task_id": task_id, "created_at": 1780000000.0,
        "finished_at": 1780000100.0, "status": "ok",
        "config": {"task_id": task_id}, "build": None, "error": None,
    }))
    base = {"rollout_id": rollout_id, "created_at": 1780000000.0,
            "finished_at": 1780000100.0, "status": "ok", "gate_passed": True,
            "cost_usd": 0.02, "image_digests": {}, "backend": "docker",
            "model_fingerprint": "fp", "accelerator": "T4",
            "agent_implementer_model": "m"}
    (tmp_path / task_id / "index.jsonl").write_text(
        json.dumps({**base, "score": 0.8, "grading_id": None, "mode": "original"}) + "\n"
        + json.dumps({**base, "score": 0.9, "grading_id": "g-20260102T000000Z-abc123",
                      "mode": "adopt"}) + "\n")

    rows = [r for r in RolloutScanner(tmp_path).list_rollouts()
            if r["rollout_id"] == rollout_id]
    assert len(rows) == 1
    assert rows[0]["score"] == 0.8


@pytest.mark.unit
def test_list_rollouts_still_shows_a_rollout_known_only_from_a_grading(
        tmp_path: Path) -> None:
    """A rollout killed before finalize has no index row of its own, so its only row is
    the one a later grading appended. Dropping that row must not drop the rollout: the
    directory walk still has to reach it."""
    task_id = "killed-task"
    rollout_id = "20260101T000000Z-killed"
    rdir = tmp_path / task_id / rollout_id
    rdir.mkdir(parents=True)
    (rdir / "config.json").write_text(json.dumps({"task_id": task_id}))
    (rdir / "events.jsonl").write_text(
        json.dumps({"type": "rollout.started", "ts": 1780000000.0,
                    "data": {"rollout_id": rollout_id}}) + "\n")
    (tmp_path / task_id / "index.jsonl").write_text(json.dumps({
        "rollout_id": rollout_id, "grading_id": "g-20260102T000000Z-abc123",
        "mode": "adopt", "created_at": 1780000200.0, "finished_at": 1780000300.0,
        "status": "ok", "score": 0.9, "gate_passed": True, "cost_usd": None,
        "image_digests": {}, "backend": "docker", "model_fingerprint": "fp",
        "accelerator": "T4", "agent_implementer_model": "m",
    }) + "\n")

    rows = [r for r in RolloutScanner(tmp_path).list_rollouts()
            if r["rollout_id"] == rollout_id]
    assert len(rows) == 1
    assert rows[0]["status"] == "unsealed"


# ---------------------------------------------------------------------------
# _build_trace_index -- cache-hit and invalidation
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_build_trace_index_cache_hit(tmp_path: Path) -> None:
    """Calling trace_page multiple times for the same (eval, file-version) must only
    parse the files ONCE. Verified by the index cache's own count (one miss, three hits) and
    by counting the calls of the reader that parses a published trace, via monkeypatching."""
    import hls.viewer.scanner as scanner_mod
    from hls.viewer.published import published_events

    task_id = "cache-task"
    rollout_id = "20260101T000000Z-cachehit"
    eval_dir = tmp_path / task_id / rollout_id / "evals" / "terminal"
    eval_dir.mkdir(parents=True)

    trace = {
        "events": [
            {
                "request_id": f"r{k}", "turn_index": k, "session_id": "s1",
                "input": {"contents": [{"text": f"q{k}"}]},
                "response": {"output": {"contents": [{"text": "A"}]}},
                "ground_truth": "B", "t_send": float(k), "t_recv": float(k) + 0.5, "error": None,
            }
            for k in range(5)
        ]
    }
    (eval_dir / "trace.json").write_text(json.dumps(trace))

    import unittest.mock as mock

    scanner = RolloutScanner(tmp_path)
    with mock.patch.object(scanner_mod, "published_events", side_effect=published_events) as reader:
        scanner_mod._build_trace_index.cache_clear()
        # Different offsets and only= values -- same file, so cache should be hit
        scanner.trace_page(task_id, rollout_id, "terminal", offset=0, limit=2)
        scanner.trace_page(task_id, rollout_id, "terminal", offset=2, limit=2)
        scanner.trace_page(task_id, rollout_id, "terminal", only="incorrect")
        scanner.trace_page(task_id, rollout_id, "terminal", only="s1")
        info = scanner_mod._build_trace_index.cache_info()

    # trace.json read once, verdict.json absent so key=(0,0) and skipped.
    # All 4 calls share the same cache entry -> one build, three hits, one read of the trace.
    assert (info.misses, info.hits) == (1, 3), f"Expected one parse and three cache hits, got {info}"
    assert reader.call_count == 1, f"Expected the trace read once, got {reader.call_count}"


@pytest.mark.unit
def test_build_trace_index_cache_invalidated_on_file_change(tmp_path: Path) -> None:
    """When trace.json is rewritten, _build_trace_index must re-parse and return the new rows."""
    import hls.viewer.scanner as scanner_mod

    task_id = "cache-inval-task"
    rollout_id = "20260101T000000Z-inval"
    eval_dir = tmp_path / task_id / rollout_id / "evals" / "terminal"
    eval_dir.mkdir(parents=True)

    def _make_trace(answer: str) -> dict:
        return {
            "events": [
                {
                    "request_id": "r0", "turn_index": 0, "session_id": None,
                    "input": {"contents": [{"text": "q"}]},
                    "response": {"output": {"contents": [{"text": answer}]}},
                    "ground_truth": "A", "t_send": 1.0, "t_recv": 1.5, "error": None,
                }
            ]
        }

    trace_path = eval_dir / "trace.json"
    trace_path.write_text(json.dumps(_make_trace("B")))

    scanner = RolloutScanner(tmp_path)
    scanner_mod._build_trace_index.cache_clear()

    page1 = scanner.trace_page(task_id, rollout_id, "terminal", only="incorrect")
    assert page1.total == 1, "First write: 'B' != 'A' -> incorrect"
    assert page1.rows[0]["artifact_text"] == "B"

    # Rewrite trace.json with a different answer; ensure mtime_ns or size changes.
    # Use a larger payload so st_size changes even if mtime resolution is coarse.
    new_content = json.dumps(_make_trace("A"))  # now correct
    trace_path.write_text(new_content + " ")  # extra space -> different size

    page2 = scanner.trace_page(task_id, rollout_id, "terminal", only="incorrect")
    assert page2.total == 0, "After rewrite: 'A' == 'A' -> no incorrect rows"
    assert page2.rows == []


# ---------------------------------------------------------------------------
# timeline() -- synthetic 2-tenant fixture
# ---------------------------------------------------------------------------


def _make_timeline_eval(tmp_path: Path) -> tuple[Path, str, str, str]:
    """A rollout dir with a 2-tenant trace.json + a config.json carrying SLO targets."""
    task_id, rollout_id, eval_id = "tl-task", "20260101T000000Z-tl", "terminal"
    edir = tmp_path / task_id / rollout_id / "evals" / eval_id
    edir.mkdir(parents=True)
    base = 1000.0
    events = [
        {  # t00 turn 0
            "request_id": "t00:a.0", "session_id": "a", "turn_index": 0,
            "input": {"contents": [{"text": "q"}], "metadata": {"tenant_id": "t00"}},
            "response": {"output": {"contents": [{"text": "ans", "tokens": [1, 2, 3]}]},
                          "t_first_packet": base + 0.30},
            "t_planned_send": base, "t_send": base + 0.10, "t_recv": base + 1.00,
            "error": None,
        },
        {  # t01 turn 0, errored (no t_recv)
            "request_id": "t01:b.0", "session_id": "b", "turn_index": 0,
            "input": {"contents": [{"text": "q2"}], "metadata": {"tenant_id": "t01"}},
            "response": None, "t_planned_send": base + 0.5, "t_send": base + 0.60,
            "t_recv": None, "error": "boom",
        },
    ]
    (edir / "trace.json").write_text(json.dumps({"events": events}))
    (edir / "verdict.json").write_text(json.dumps({}))
    config = {"spec": {"tenants": [
        {"id": "t00", "data_source": {"name": "coqa"},
         "workload": {"slo": {"ttft_ms": 6000.0, "tpot_ms": 100.0}}},
        {"id": "t01", "data_source": {"name": "hotpot"},
         "workload": {"slo": {"ttft_ms": 8000.0, "tpot_ms": 100.0}}},
    ]}}
    (tmp_path / task_id / rollout_id / "config.json").write_text(json.dumps(config))
    return tmp_path, task_id, rollout_id, eval_id


@pytest.mark.unit
def test_timeline_payload(tmp_path: Path) -> None:
    base, task_id, rollout_id, eval_id = _make_timeline_eval(tmp_path)
    payload = RolloutScanner(base).timeline(task_id, rollout_id, eval_id)
    assert payload is not None
    # t0 anchors to the earliest send (1000.10); times are relative
    assert payload.t0 == pytest.approx(1000.10)
    assert payload.duration == pytest.approx(0.90)  # recv 1001.00 - t0
    # two tenants discovered in first-seen order, enriched from config
    assert [t.id for t in payload.tenants] == ["t00", "t01"]
    assert payload.tenants[0].name == "coqa"
    assert payload.tenants[0].slo is not None
    assert payload.tenants[0].slo.kind == "streaming"
    assert payload.tenants[0].slo.ttft_ms == 6000.0
    assert payload.tenants[0].req_n == 1
    # requests: relative times, ti index, ntok, err, null recv on error
    r0 = payload.requests[0]
    assert r0 == {"ti": 0, "sid": "a", "turn": 0, "send": pytest.approx(0.0),
                  "recv": pytest.approx(0.90), "plan": pytest.approx(-0.10),
                  "ttft": pytest.approx(0.20), "tlast": None, "ntok": 3, "err": False}
    r1 = payload.requests[1]
    assert r1["ti"] == 1 and r1["recv"] is None and r1["err"] is True


@pytest.mark.unit
def test_timeline_missing_trace_returns_none(tmp_path: Path) -> None:
    assert RolloutScanner(tmp_path).timeline("nope", "nope", "terminal") is None


def test_detect_verifier_kind_keeps_legacy_class_names():
    """verdict.json persists verifier CLASS NAMES: a verdict naming GreedyMatchVerifier
    or GreedyMatch must keep resolving to token_level, not silently demote to generic
    (which would drop their turn table + rollup card in the viewer)."""
    from hls.viewer.scanner import _detect_verifier_kind

    for legacy in ("GreedyMatchVerifier", "GreedyMatch"):
        verdict = {"diagnostics": {f"0:{legacy}": {"epsilon": 1e-4}}}
        assert _detect_verifier_kind(verdict) == "token_level"


def test_answer_correct_grades_both_reply_shapes():
    """Short-answer replies lead with the option; free-form reasoning replies end
    with an "Answer: <option>" line. Both must grade against a bare-letter ground
    truth, by leading TOKEN (gt "A" must never match "Applying ...")."""
    from hls.viewer.scanner import _answer_correct

    assert _answer_correct("A. 0", "A") is True
    assert _answer_correct("B. 4", "A") is False
    assert _answer_correct("The kernel is normal, order 2.\nAnswer: A", "A") is True
    assert _answer_correct("Applying the definition gives it.\nAnswer: C", "A") is False
    assert _answer_correct("Applying the definition gives it.", "A") is False
    assert _answer_correct("prose only, no verdict line", "A") is False
    assert _answer_correct("anything", None) is None
