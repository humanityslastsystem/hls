"""The statuses a rollout record can carry, which of them mean the rollout ended, and the shapes the
stage pipeline's records hold."""

from __future__ import annotations

from typing import get_args

import pytest
from pydantic import ValidationError

from hls.obs.record import (TERMINAL_STATUSES, EvalEpisode, ExceptionInfo, GradingRecord, RolloutBuild,
                            RolloutProvenance, RolloutRecord, RolloutStatus)


def test_terminal_statuses_are_every_status_but_running():
    assert TERMINAL_STATUSES == frozenset(get_args(RolloutStatus)) - {"running"}
    assert "preview" not in get_args(RolloutStatus)


def test_a_rollouts_build_is_the_stage_pipelines_record_or_none():
    annotation = RolloutRecord.model_fields["build"].annotation
    assert {a.__name__ for a in get_args(annotation) if a is not type(None)} == {"RolloutBuild"}


def test_provenance_holds_only_what_the_pipeline_sets():
    """Finalize is its one writer and nothing reads a record back through it, so a key it does not
    hold is refused, a published record's among them."""
    gone = {"substrate", "container_env_public", "image_recipes", "sandbox_ids", "sandbox_exits"}
    assert not gone & set(RolloutProvenance.model_fields)
    published = {"task_id": "t", "substrate": {"backend": "modal"}, "sandbox_ids": {"artifact": "sb-1"}}
    with pytest.raises(ValidationError, match="substrate"):
        RolloutProvenance.model_validate(published)
    assert RolloutProvenance.model_validate({"task_id": "t"}).task_id == "t"


def test_an_episode_and_an_error_hold_no_field_that_nothing_writes_or_shows():
    """Finalize sets neither an episode's anchor nor its error, and nothing writes an error's return
    code; the records refuse them, as they refuse any key they do not hold."""
    assert not {"anchor", "error"} & set(EvalEpisode.model_fields)
    assert "returncode" not in ExceptionInfo.model_fields
    with pytest.raises(ValidationError, match="anchor"):
        EvalEpisode.model_validate({"eval_id": "terminal", "checkpoint": "docker:img", "started_at": 1.0,
                                    "trace_path": "evals/terminal/trace.json", "anchor": {"turn_index": 3}})
    with pytest.raises(ValidationError, match="returncode"):
        ExceptionInfo.model_validate({"type": "SubmissionBuildError", "message": "pip failed", "traceback": "",
                                      "returncode": 1})


@pytest.mark.parametrize("model", [ExceptionInfo, RolloutProvenance, RolloutBuild, EvalEpisode, GradingRecord,
                                   RolloutRecord], ids=lambda m: m.__name__)
def test_every_record_model_refuses_a_key_it_does_not_hold(model):
    """Only finalize writes these records and nothing reads one back through them (the viewer reads
    the stored JSON), so a key a model does not hold is a writer's mistake, never data to keep."""
    assert model.model_config.get("extra") == "forbid"
