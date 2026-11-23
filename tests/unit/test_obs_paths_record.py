from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.obs.conditions import HarnessConditions
from hls.measured import MeasuredFiles
from hls.obs.record import EvalEpisode, GradingRecord, RolloutBuild, RolloutProvenance, RolloutRecord
from hls.store.layout import new_rollout_id


def _conditions() -> HarnessConditions:
    """The harness conditions of an instance that ran no stage."""
    return HarnessConditions(attempts={}, allocations=[], hls_commit="a" * 40, harbor_commit="b" * 40, dirty=False,
                             hardware=None, baseline_hardware=None, hardware_warning=None,
                             credit=None, calibration=None, measured=MeasuredFiles(), measured_note=None,
                             arm=None, budget=None, grade_after_sandbox_oom=None, images={}, placement=None)


def test_rollout_id_is_sortable_and_unique():
    a, b = new_rollout_id(), new_rollout_id()
    assert a != b
    assert len(a) >= 8
    assert re.fullmatch(r"\d{8}T\d{6}Z-[0-9a-f]{6}", a)


def test_record_round_trips(tmp_path: Path):
    rec = RolloutRecord(
        rollout_id="r1",
        task_id="example-task",
        created_at=1.0,
        config=RolloutProvenance(
            task_id="example-task",
            image_digests={},
            hls_commit=None,
            harbor_commit=None,
        ),
        build=RolloutBuild(build={"arm": {"name": "example"}}, outputs={"build": "runs/example-task/r1/agent_build/1/build.json"},
                           package=None, started_at=1.0, ended_at=2.0, usage=None, upload_sha256=None),
        conditions=_conditions(),
    )
    blob = rec.model_dump_json()
    again = RolloutRecord.model_validate_json(blob)
    assert (again.build, again.conditions) == (rec.build, rec.conditions)
    assert again.config.task_id == "example-task"


def test_a_record_names_its_harness_conditions_and_holds_no_old_build():
    """Only the stage pipeline writes these records: each names how its instance ran, and a
    rollout's build is the pipeline's."""
    config = RolloutProvenance(task_id="example-task")
    with pytest.raises(ValidationError, match="conditions"):
        RolloutRecord(rollout_id="r1", task_id="example-task", created_at=1.0, config=config)
    episode = EvalEpisode(eval_id="terminal", checkpoint="sha256:a", started_at=1.0,
                          trace_path="evals/terminal/trace.json")
    with pytest.raises(ValidationError, match="conditions"):
        GradingRecord(episode=episode, provenance=config, source_rollout="r1")
    with pytest.raises(ValidationError, match="build"):
        RolloutRecord(rollout_id="r1", task_id="example-task", created_at=1.0, config=config, conditions=_conditions(),
                      build={"started_at": 1.0, "artifact_image": "sha256:a"})
