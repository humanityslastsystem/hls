"""A task missing what a plan reads of it (the drive bounds it declares, the packet cost its
contract gives the score stage, the image catalog's measured roles) is refused in one line when a
command plans it, never a traceback."""

from __future__ import annotations

import pytest
import typer

from hls import cli
from hls.drive.bounds import drive_bounds
from hls.recipes import catalog
from hls.stages.score import score_memory_gib
from hls.taskref import TaskRef, load_task
from tests.evaluate.fixture import unbounded_task


def _unsized_stream():
    """omni's contract streamed: a stream without token ids, whose packet cost no protocol says."""
    contract = load_task(TaskRef(module="tasks.multimodal_omni.task")).contract
    return contract.model_copy(update={"transport": "http-sse"})


LOOKUPS = {
    "drive bounds": (lambda: drive_bounds(unbounded_task()), "DriveBounds"),
    "packet cost": (lambda: score_memory_gib(_unsized_stream()), "stream_packet_bytes"),
}


@pytest.mark.parametrize("table", sorted(LOOKUPS))
def test_a_task_missing_from_a_table_is_refused_in_one_line(table, capsys):
    lookup, names = LOOKUPS[table]
    with pytest.raises(typer.Exit) as ended:
        cli._plan_and_start("run", lookup, {}, start=lambda *a, **k: 0)
    assert ended.value.exit_code == 1
    out, err = capsys.readouterr()
    assert out == "" and len(err.splitlines()) == 1 and err.startswith("hls run: no "), err
    assert any(name in err for name in names.split("|")), err


def test_a_task_module_that_declares_no_images_is_refused_naming_the_declaration(monkeypatch, capsys):
    import tasks.sample_task.task as module

    monkeypatch.delattr(module, "IMAGES")
    with pytest.raises(typer.Exit) as ended:
        cli._plan_and_start("run", lambda: catalog.task_images("sample_task"), {}, start=lambda *a, **k: 0)
    assert ended.value.exit_code == 1
    out, err = capsys.readouterr()
    assert out == "" and len(err.splitlines()) == 1 and "declares no IMAGES = ImageSet(...)" in err, err
