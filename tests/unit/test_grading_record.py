"""A grading is a file in its own directory under the rollout that produced the
artifact, so one artifact can be graded many times."""
from __future__ import annotations

import re

from hls.measured import MeasuredFiles
from hls.obs.conditions import HarnessConditions
from hls.obs.record import EvalEpisode, GradingRecord, RolloutProvenance
from hls.store.layout import new_grading_id


def _grading() -> GradingRecord:
    """An adopted regrade, under the conditions of an instance that ran no stage."""
    return GradingRecord(
        episode=EvalEpisode(
            eval_id="g-x", triggered_by="regrade", checkpoint="apptainer:/x/y.sif",
            started_at=1.0, trace_path="evals/g-x/trace.json",
            score={"value": 2.41, "gate_passed": True},
        ),
        provenance=RolloutProvenance(task_id="demo", hls_commit="abc1234"),
        source_rollout="runs/demo/r1",
        mode="adopt",
        conditions=HarnessConditions(attempts={}, allocations=[], hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty=False, hardware=None, baseline_hardware=None, hardware_warning=None,
                                     credit=None,
                                     calibration=None, measured=MeasuredFiles(), measured_note=None, arm=None, budget=None, grade_after_sandbox_oom=None, images={},
                                     placement=None),
    )


def test_grading_ids_sort_and_are_distinct():
    a, b = new_grading_id(), new_grading_id()
    assert a.startswith("g-") and b.startswith("g-")
    assert a != b
    assert re.fullmatch(r"g-\d{8}T\d{6}Z-[0-9a-f]{6}", a)


def test_grading_record_round_trips():
    rec = _grading()
    back = GradingRecord.model_validate_json(rec.model_dump_json())
    assert back == rec
    assert back.episode.score["value"] == 2.41
    assert back.mode == "adopt"
    assert back.provenance.hls_commit == "abc1234"
    assert back.episode.triggered_by == "regrade"


def test_the_rollout_record_no_longer_carries_gradings():
    """A rollout is the trajectory and the artifacts it produced. Its gradings are
    separate records, so a projection cannot drift from them."""
    from hls.obs.record import RolloutRecord

    assert "evals" not in RolloutRecord.model_fields


def test_gradings_are_read_from_disk(tmp_path):
    from hls.viewer.scanner import gradings_of

    rollout = tmp_path / "demo" / "r1"
    (rollout / "evals" / "g-x").mkdir(parents=True)
    (rollout / "evals" / "g-x" / "grading.json").write_text(_grading().model_dump_json())
    found = gradings_of(rollout)
    assert [g["episode"]["eval_id"] for g in found] == ["g-x"]
    assert found[0]["episode"]["score"]["value"] == 2.41


def test_a_rollout_with_no_gradings_reads_as_empty(tmp_path):
    from hls.viewer.scanner import gradings_of

    d = tmp_path / "demo" / "r2"
    d.mkdir(parents=True)
    assert gradings_of(d) == []
