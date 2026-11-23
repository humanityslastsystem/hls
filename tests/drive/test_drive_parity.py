"""A drive's requests are the ones it sent while the drive role still configured the driver by
attribute: ``drive_parity.json`` holds what it sent, recorded before the driver took its contract,
seed and tenants as arguments. ``parity_tasks.py`` holds the two tasks and says how both sides
drive them. And a drive's trace is the one a writer wrote while the drive role held every
event until the drive ended: the same drive, its events written as they are recorded, writes the
bytes ``drive_parity_trace.json`` holds."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from hls.drive.stream import TraceStream
from hls.trace import store
from hls.trace.store import END_FILE
from tests.drive.parity_tasks import record_drive, scripted_task, trajectory_task

GOLDEN = json.loads(Path(__file__).with_name("drive_parity.json").read_text())
TASKS = [("scripted", scripted_task), ("trajectory", trajectory_task)]


@pytest.mark.parametrize("name, build", TASKS)
def test_a_drive_sends_what_it_sent_before(name, build):
    assert record_drive(build()) == GOLDEN[name]


@pytest.mark.parametrize("name, build", TASKS)
def test_the_golden_tells_another_seed_apart(name, build):
    """The comparison above is only worth something if a drive that drew differently fails it."""
    task = build()
    other = record_drive(task.model_copy(update={"seed": task.seed + 1}))
    assert other != GOLDEN[name]
    assert [row[4] for row in other] != [row[4] for row in GOLDEN[name]], "the send times moved with the seed"


#: The driver's description both traces record.
DRIVER = {"class": "Driver", "split": "test"}
#: The trace of each parity drive written with every event held until the end, by task and chunk
#: size: the sha256 of every file but END.json, and END.json's fields but its write time, as that
#: writer wrote them. Each file's plain bytes recompress to
#: the same bytes under zlib 1.2.13 (Python 3.12.3) and 1.3.1 (Python 3.13).
MAIN_TRACES = json.loads(Path(__file__).with_name("drive_parity_trace.json").read_text())


@pytest.mark.parametrize("chunk_bytes", [1, 2_000])
@pytest.mark.parametrize("name, build", TASKS)
def test_a_drive_written_as_it_goes_writes_the_trace_mains_writer_wrote(tmp_path, name, build, chunk_bytes):
    """The same drive on the virtual clock, each event handed to the role's trace stream as it is
    recorded: every file is the one main's writer wrote of it when the role held every event,
    and END.json differs only in its write time."""
    held: list = []
    record_drive(build(), held.append)
    local = tmp_path / "local"
    local.mkdir()
    store.make_room(local, event_bytes=sum(len(store._event_line(ev)) for ev in held), chunk_bytes=chunk_bytes)
    with TraceStream(local, driver=DRIVER, chunk_bytes=chunk_bytes) as stream:
        record_drive(build(), stream.put)
        written = stream.finish("drained")
    streamed = tmp_path / "streamed"
    store.publish_trace(local, streamed, written)
    main = MAIN_TRACES[f"{name}-{chunk_bytes}"]
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in streamed.iterdir() if p.name != END_FILE}
    assert files == main["files"] and len(files) >= 2
    end = json.loads((streamed / END_FILE).read_bytes())
    del end["written_at"]
    assert end == main["end"] and end["events"] == len(held) > 0
