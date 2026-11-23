"""The viewer on published rollouts: rows and records addressed by their directories, traces in
every published form, a run never sealed shown from its config and its gradings, and a record's
build and producer, in the legacy shape, never read."""

from __future__ import annotations

import gzip
import json

import pytest

from hls.core.trace import Trace
from hls.viewer.scanner import RolloutScanner
from tests.trace.traces import drive_trace
from tests.viewer.published_fixture import (ARM_CELL, ARM_RECORD_ID, ARM_RECORD_TASK, OLD_BUILD, OLD_PRODUCER,
                                            PARTS, TRACE_FORMS, UNSEALED_ID, published_runs, write_trace_file)
from tests.viewer.store_fixture import GRADING, ROLLOUT, TASK, viewer_client

TRACE = drive_trace()


def test_the_viewer_still_reads_the_published_layout(tmp_path):
    published_runs(tmp_path, TRACE)
    scanner = RolloutScanner(tmp_path)
    row = next(r for r in scanner.list_rollouts() if r["rollout_id"] == ROLLOUT)
    assert (row.get("layout"), row["backend"]) == (None, "apptainer")
    assert [e["eval_id"] for e in scanner.get_rollout(TASK, ROLLOUT)["evals"]] == ["terminal"]
    assert scanner.trace_summary(TASK, ROLLOUT, "terminal").total == len(TRACE.events)


def test_a_published_cell_is_listed_and_opened_by_its_directory(tmp_path):
    """A published cell's directory is named by its arm while its record keeps the run's own ids;
    the row and the record are addressed by the directories, so the cell opens, and every view
    the page builds from the record's ids reaches it."""
    published_runs(tmp_path, TRACE)
    scanner = RolloutScanner(tmp_path)
    [row] = [r for r in scanner.list_rollouts() if r.get("record_rollout_id") == ARM_RECORD_ID]
    assert (row["task_id"], row["rollout_id"], row["score"]) == (TASK, ARM_CELL, 0.7)
    rec = scanner.get_rollout(row["task_id"], row["rollout_id"])
    assert (rec["task_id"], rec["rollout_id"]) == (TASK, ARM_CELL)
    assert (rec["record_task_id"], rec["record_rollout_id"]) == (ARM_RECORD_TASK, ARM_RECORD_ID)
    assert scanner.trace_summary(rec["task_id"], rec["rollout_id"], "terminal").total == len(TRACE.events)
    client = viewer_client(tmp_path)
    assert client.get(f"/api/rollouts/{TASK}/{ARM_CELL}").json()["rollout_id"] == ARM_CELL
    # A record whose ids are its directory's carries no second copy of them.
    assert "record_rollout_id" not in scanner.get_rollout(TASK, ROLLOUT)


@pytest.mark.parametrize("form", TRACE_FORMS)
def test_a_published_trace_is_read_in_every_published_form(tmp_path, form):
    published_runs(tmp_path, TRACE, arm_form=form)
    scanner = RolloutScanner(tmp_path)
    summary = scanner.trace_summary(TASK, ARM_CELL, "terminal")
    assert summary.total == len(TRACE.events)
    page = scanner.trace_page(TASK, ARM_CELL, "terminal", limit=100)
    assert [r["request_id"] for r in page.rows] == [ev.request_id for ev in TRACE.events]
    assert len(scanner.timeline(TASK, ARM_CELL, "terminal").requests) == len(TRACE.events)


def test_a_split_trace_missing_a_part_reads_as_no_trace(tmp_path):
    """Parts cut at gzip members and between events: without its middle part the rest still
    decompresses to a trace, events short, which is not the drive."""
    published_runs(tmp_path, TRACE)
    terminal = tmp_path / TASK / ARM_CELL / "evals" / "terminal"
    (terminal / "trace.json.gz").unlink()
    events = [json.dumps(json.loads(ev.model_dump_json())) for ev in TRACE.events]
    pieces = ['{"events": [' + events[0] + ",", ",".join(events[1:-1]) + ",", events[-1] + "]}"]
    for n, piece in enumerate(pieces):
        (terminal / f"{PARTS}{n:02d}").write_bytes(gzip.compress(piece.encode()))
    assert RolloutScanner(tmp_path).trace_summary(TASK, ARM_CELL, "terminal").total == len(TRACE.events)
    (terminal / f"{PARTS}01").unlink()
    assert RolloutScanner(tmp_path).trace_summary(TASK, ARM_CELL, "terminal") is None


def test_a_split_trace_is_read_in_its_parts_numeric_order(tmp_path):
    """A hundred and one parts: part100 sorts before part11 by name and after it by number."""
    published_runs(tmp_path, TRACE)
    terminal = tmp_path / TASK / ARM_CELL / "evals" / "terminal"
    (terminal / "trace.json.gz").unlink()
    write_trace_file(terminal, TRACE, PARTS, parts=101)
    assert sorted(terminal.glob(f"{PARTS}*"))[11].name == f"{PARTS}100"
    assert RolloutScanner(tmp_path).trace_summary(TASK, ARM_CELL, "terminal").total == len(TRACE.events)


def test_the_first_trace_form_present_is_read(tmp_path):
    """``trace.json`` before its gzipped forms: a cell keeps one, but the order is fixed."""
    published_runs(tmp_path, TRACE)
    terminal = tmp_path / TASK / ARM_CELL / "evals" / "terminal"
    write_trace_file(terminal, Trace(events=TRACE.events[:3]), "trace.json")
    assert RolloutScanner(tmp_path).trace_summary(TASK, ARM_CELL, "terminal").total == 3


def test_an_unsealed_published_rollout_is_shown_from_its_config_and_gradings(tmp_path):
    published_runs(tmp_path, TRACE)
    scanner = RolloutScanner(tmp_path)
    row = next(r for r in scanner.list_rollouts() if r["rollout_id"] == UNSEALED_ID)
    assert (row["task_id"], row["status"], row["score"], row["created_at"]) == (TASK, "unsealed", 0.42, 1.0)
    rec = scanner.get_rollout(TASK, UNSEALED_ID)
    assert (rec["task_id"], rec["status"]) == (TASK, "unsealed")
    assert [e["eval_id"] for e in rec["evals"]] == [GRADING]
    # A directory with neither record is not listed.
    (tmp_path / TASK / "20260902T000000Z-empty").mkdir()
    assert "20260902T000000Z-empty" not in {r["rollout_id"] for r in scanner.list_rollouts()}


def test_an_unsealed_published_rollout_has_no_build_record_and_its_kept_trajectory_is_served(tmp_path):
    """No build record is made up for it: the page asks the trajectory endpoint whether there is a
    trajectory, which serves ``build/trajectory.atif.json`` while it is there; nothing is folded
    from its events."""
    published_runs(tmp_path, TRACE)
    scanner = RolloutScanner(tmp_path)
    assert scanner.get_rollout(TASK, UNSEALED_ID)["build"] is None
    assert scanner.get_trajectory(TASK, UNSEALED_ID) == {"steps": [{"step_id": 1}]}
    (tmp_path / TASK / UNSEALED_ID / "build" / "trajectory.atif.json").unlink()
    assert scanner.get_rollout(TASK, UNSEALED_ID)["build"] is None
    assert scanner.get_trajectory(TASK, UNSEALED_ID) is None


def test_a_published_records_build_and_producer_are_not_read(tmp_path):
    """A published record holds its build, and the producer that names the arm that built it, in
    the legacy shape, which the viewer does not read: the record is served with no build,
    and its row, listed from the record where no index row names it, with no cost and no
    implementer, though the legacy build holds a cost and the producer a model. The page says why
    (``apps/viewer/app/lib/build.ts``, ``ABSENT.publishedBuild`` and ``ABSENT.publishedProducer``)."""
    published_runs(tmp_path, TRACE)
    record = json.loads((tmp_path / TASK / ARM_CELL / "rollout.json").read_bytes())
    assert (record["build"], record["config"]["artifact"]["producer"]) == (OLD_BUILD, OLD_PRODUCER)
    scanner = RolloutScanner(tmp_path)
    assert scanner.get_rollout(TASK, ARM_CELL)["build"] is None
    assert viewer_client(tmp_path).get(f"/api/rollouts/{TASK}/{ARM_CELL}").json()["build"] is None
    # Not read is not invalid: where the old build fails to validate as a store record's build is
    # never reported for it.
    assert "build_invalid" not in scanner.get_rollout(TASK, ARM_CELL)
    [row] = [r for r in scanner.list_rollouts() if r["rollout_id"] == ARM_CELL]
    assert (row["cost_usd"], row.get("agent_implementer_model")) == (None, None)


@pytest.mark.parametrize(("checkpoint", "backend"), [
    ("apptainer:/scratch/hls/artifact.sif", "apptainer"), ("docker:sha256:" + "a" * 64, "docker"),
    ("modal:modal.Image.from_id('im-0')", "modal"), ("sha256:" + "a" * 64, None), ("e" * 64, None)])
def test_a_published_rows_backend_is_the_prefix_of_its_terminal_checkpoint(tmp_path, checkpoint, backend):
    """A published row no index row names takes its backend from the checkpoint its terminal
    grading served, the backend named before the first colon; arm-named cells
    served ``apptainer:<sif>``. A checkpoint with no backend's prefix names none."""
    published_runs(tmp_path, TRACE)
    grading = tmp_path / TASK / ARM_CELL / "evals" / "terminal" / "grading.json"
    doc = json.loads(grading.read_bytes())
    grading.write_text(json.dumps({**doc, "episode": {**doc["episode"], "checkpoint": checkpoint}}))
    [row] = [r for r in RolloutScanner(tmp_path).list_rollouts() if r["rollout_id"] == ARM_CELL]
    assert row.get("backend") == backend
