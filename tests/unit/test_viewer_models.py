"""Unit tests for hls.viewer.models -- TDD (write first, run RED, implement GREEN)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hls.viewer.models import (
    FileInfo,
    PaginatedResponse,
    RolloutSummary,
    TracePage,
    TraceSummary,
)


# ---------------------------------------------------------------------------
# PaginatedResponse
# ---------------------------------------------------------------------------


def test_paginated_response_generic() -> None:
    page: PaginatedResponse[str] = PaginatedResponse(
        items=["a", "b"],
        total=10,
        page=1,
        page_size=2,
        total_pages=5,
    )
    assert page.items == ["a", "b"]
    assert page.total_pages == 5


def test_paginated_response_with_dicts() -> None:
    item = {"id": "x", "score": 0.9}
    page: PaginatedResponse[dict] = PaginatedResponse(
        items=[item],
        total=1,
        page=0,
        page_size=50,
        total_pages=1,
    )
    assert page.items[0]["score"] == 0.9


# ---------------------------------------------------------------------------
# FileInfo
# ---------------------------------------------------------------------------


def test_file_info_file() -> None:
    fi = FileInfo(path="evals/terminal/trace.json", name="trace.json", is_dir=False, size=1024)
    assert fi.size == 1024
    assert not fi.is_dir


def test_file_info_dir() -> None:
    fi = FileInfo(path="evals", name="evals", is_dir=True)
    assert fi.size is None
    assert fi.is_dir


# ---------------------------------------------------------------------------
# RolloutSummary
# ---------------------------------------------------------------------------


def test_rollout_summary_minimal() -> None:
    rs = RolloutSummary(task_id="example-task", rollout_id="20260604T064710Z-b4dae3")
    assert rs.task_id == "example-task"
    assert rs.status is None
    assert rs.phase is None


def test_rollout_summary_full() -> None:
    rs = RolloutSummary(
        task_id="example-task",
        rollout_id="20260604T064710Z-b4dae3",
        status="ok",
        score=2.2,
        gate_passed=True,
        cost_usd=0.05,
        phase="eval 1/1",
        split="dev",
        model_fingerprint="fp-abc",
        accelerator="H200",
        backend="modal",
        agent_implementer_model="claude-opus-4",
    )
    assert rs.score == pytest.approx(2.2)
    assert rs.phase == "eval 1/1"


def test_rollout_summary_extra_fields_allowed() -> None:
    # extra="allow" -- arbitrary extra keys must not raise
    rs = RolloutSummary(
        task_id="T",
        rollout_id="r",
        arbitrary_extra_field="hello",
        another=42,
    )
    assert rs.model_extra is not None
    assert rs.model_extra.get("arbitrary_extra_field") == "hello"


def test_rollout_summary_missing_required_raises() -> None:
    with pytest.raises(ValidationError):
        RolloutSummary(rollout_id="r")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# TraceSummary
# ---------------------------------------------------------------------------


def test_trace_summary_all_none() -> None:
    ts = TraceSummary(total=0)
    assert ts.token_match is None
    assert ts.p99_ms is None


def test_trace_summary_full() -> None:
    ts = TraceSummary(
        total=5,
        token_match=5,
        token_match_total=5,
        correct=1,
        graded=5,
        p99_ms=2265.8,
    )
    assert ts.total == 5
    assert ts.correct == 1
    assert ts.p99_ms == pytest.approx(2265.8, abs=0.1)


# ---------------------------------------------------------------------------
# TracePage
# ---------------------------------------------------------------------------


def test_trace_page_basic() -> None:
    rows = [{"turn_index": 0, "request_id": "req-0"}, {"turn_index": 1, "request_id": "req-1"}]
    page = TracePage(offset=0, limit=2, total=5, rows=rows)
    assert len(page.rows) == 2
    assert page.total == 5


def test_trace_page_empty() -> None:
    page = TracePage(offset=10, limit=2, total=10, rows=[])
    assert page.rows == []
