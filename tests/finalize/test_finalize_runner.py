"""The score and finalize stages run by the real runner, as an allocation runs them, on a
fixture instance whose drive and verify stages are published: what the runner requires of a
stage, its outputs in its attempt's own directory among it, holds for both."""

from __future__ import annotations

import json

from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.runner import EXIT_FAILED, EXIT_PUBLISHED
from tests.finalize.published import ROLLOUT, TASK, publish_instance
from tests.finalize.test_failed_records import SECRET, STAGES, _plan
from tests.pipeline.test_runner import _grant, _runner, _status, _store, _sub
from tests.trace.traces import drive_trace, passing_verdict


def test_the_runner_publishes_score_and_finalize_with_the_row_outside_their_outputs(tmp_path, checkout):
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    plan = _store(plan, _sub(plan))
    assert [a.stages for a in plan.allocations] == [STAGES]
    request = plan.allocation(0).request
    runner = _runner(plan, code=plan.code,
                     measure=_grant(gpus=request.gpus, cpus=request.cpus, memory_gib=request.memory_gib))
    assert runner.run() == EXIT_PUBLISHED, _status(plan).fault
    layout = plan.layout
    for stage in ("score", "finalize"):
        manifest, _ = read_manifest(store, layout, stage)
        prefix = f"{layout.attempt(stage, manifest.attempt)}/outputs/"
        assert all(ref.key.startswith(prefix) for ref in manifest.outputs.values())
    finalized, _ = read_manifest(store, layout, "finalize")
    assert sorted(finalized.outputs) == ["grading", "rollout"]
    score = json.loads(store.read_bytes(read_manifest(store, layout, "score")[0].outputs["score"].key))
    row = json.loads(store.read_bytes(f"runs/{TASK}/index/{ROLLOUT}.json"))
    assert (row["rollout_id"], row["score"], row["failed_stage"]) == (ROLLOUT, score["value"], None)
    for ref in finalized.outputs.values():
        assert SECRET.encode() not in store.read_bytes(ref.key)


def test_a_plan_revision_that_holds_a_value_refuses_the_instance(tmp_path, checkout):
    # A plan made without the planner's check of its text (the secrets file was not given to it)
    # holds the value verbatim; finalize's scan reads the plan's revisions too.
    plan = _plan(tmp_path, checkout, checkpoint=f"hub/{SECRET}", checked=False)
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, trace=trace, verdict=passing_verdict(trace))
    plan = _store(plan, _sub(plan))
    request = plan.allocation(0).request
    runner = _runner(plan, code=plan.code,
                     measure=_grant(gpus=request.gpus, cpus=request.cpus, memory_gib=request.memory_gib))
    assert runner.run() == EXIT_FAILED
    fault = _status(plan).fault
    revision = plan.layout.plan_revision(plan.revision)
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == f"the harness leaked secret values into published files: {revision} (hf_token)"
    assert not store.exists(f"runs/{TASK}/index/{ROLLOUT}.json")
