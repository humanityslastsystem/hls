"""Published rollouts as the viewer meets them: a sealed run in the legacy ``runs/`` layout with its
index row; an unsealed run that a later ``hls grade`` graded; and a sealed cell named by its arm,
``<task>/<repeat>/<arm>/``, whose record carries the run's own id and task id. Each sealed record
holds its build and its producer in the legacy shape, as every published record does, and each
trace is written in one of the published forms."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from hls.core.trace import Trace
from tests.viewer.store_fixture import CREATED, GRADING, ROLLOUT, TASK, grading_record

#: The unsealed run, and the arm-named cell with the ids its record carries.
UNSEALED_ID = "20260901T000000Z-killed"
ARM_CELL, ARM_RECORD_ID, ARM_RECORD_TASK = "arm-a", "20260910T000000Z-0f0f0f", "T1.2"
#: A build in the legacy shape, as every sealed published record holds it.
OLD_BUILD = {"started_at": 300.0, "finished_at": 400.0, "artifact_image": "hls-artifact:x",
             "n_input_tokens": 1, "n_output_tokens": 2, "cost_usd": 45.2, "total_steps": 4,
             "has_trajectory": True, "logs": {}, "source": None}
#: The legacy producer, which named the arm that built the artifact, in a published
#: record's provenance (``config.artifact.producer``).
OLD_PRODUCER = {"model_name": "old-model", "harness": "claude-code", "artifact_mode": "git_repo",
                "network": {"domains": None}}
#: Every published form of a trace. ``PARTS`` stands for a gzipped trace split into
#: numbered parts, ``trace.json.gz.part00`` on.
PARTS = "trace.json.gz.part"
TRACE_FORMS = ("trace.json", "trace.json.gz", PARTS, "trace.elided.json.gz", "trace.audio-elided.json.gz",
               "trace.media-elided.json.gz")


def trace_bytes(trace: Trace) -> bytes:
    """*trace* as a published ``trace.json``."""
    return json.dumps({"events": [json.loads(ev.model_dump_json()) for ev in trace.events]}).encode()


def write_trace_file(eval_dir: Path, trace: Trace, form: str, *, parts: int = 3) -> None:
    """Write *trace* into *eval_dir* in *form*; a split trace in *parts* parts."""
    eval_dir.mkdir(parents=True, exist_ok=True)
    data = trace_bytes(trace)
    if form == "trace.json":
        (eval_dir / form).write_bytes(data)
        return
    packed = gzip.compress(data)
    if form != PARTS:
        (eval_dir / form).write_bytes(packed)
        return
    size = -(-len(packed) // parts)
    for n in range(parts):
        (eval_dir / f"{PARTS}{n:02d}").write_bytes(packed[n * size:(n + 1) * size])


def published_runs(root: Path, trace: Trace, *, arm_form: str = "trace.json.gz") -> None:
    """The three published rollouts under ``root/TASK/``. Each graded eval drove *trace*; the
    arm-named cell's is written in *arm_form*."""
    rdir = root / TASK / ROLLOUT
    terminal = rdir / "evals" / "terminal"
    terminal.mkdir(parents=True)
    (rdir / "rollout.json").write_text(json.dumps({"rollout_id": ROLLOUT, "task_id": TASK, "status": "ok",
                                                   "created_at": CREATED, "finished_at": CREATED + 900,
                                                   "config": {"artifact": {"producer": OLD_PRODUCER}},
                                                   "build": OLD_BUILD}))
    (rdir / "events.jsonl").write_text('{"type": "rollout.finished", "ts": 2.0, "data": {}}\n')
    (terminal / "grading.json").write_text(json.dumps(grading_record("terminal", 1.12)))
    write_trace_file(terminal, trace, "trace.json")
    (terminal / "verdict.json").write_text(json.dumps({"aggregate_passed": True, "per_event": {},
                                                       "diagnostics": {}}))
    (rdir.parent / "index.jsonl").write_text(json.dumps({
        "rollout_id": ROLLOUT, "created_at": CREATED, "status": "ok", "score": 1.12, "backend": "apptainer",
        "model_fingerprint": "fp", "accelerator": "H200", "agent_implementer_model": "claude-opus-5"}) + "\n")

    unsealed = root / TASK / UNSEALED_ID
    unsealed.mkdir()
    (unsealed / "config.json").write_text(json.dumps({"task_id": "deepresearch", "image_digests": {}}))
    (unsealed / "events.jsonl").write_text('{"type": "rollout.started", "ts": 1.0, "data": {}}\n')
    (unsealed / "build").mkdir()
    (unsealed / "build" / "trajectory.atif.json").write_text(json.dumps({"steps": [{"step_id": 1}]}))
    graded = unsealed / "evals" / GRADING
    graded.mkdir(parents=True)
    (graded / "grading.json").write_text(json.dumps(grading_record(GRADING, 0.42)))

    cell = root / TASK / ARM_CELL
    cell_terminal = cell / "evals" / "terminal"
    cell_terminal.mkdir(parents=True)
    (cell / "rollout.json").write_text(json.dumps({
        "rollout_id": ARM_RECORD_ID, "task_id": ARM_RECORD_TASK, "status": "ok", "created_at": CREATED + 50,
        "finished_at": CREATED + 950, "config": {"artifact": {"producer": OLD_PRODUCER}},
        "build": OLD_BUILD}))
    (cell_terminal / "grading.json").write_text(json.dumps(grading_record("terminal", 0.7)))
    write_trace_file(cell_terminal, trace, arm_form)
