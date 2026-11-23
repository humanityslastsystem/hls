"""A plan's pins of the measured files its roles load (``hls.measured``), the line a record
carries for a task's default, and the dev reference the loader adds to a
data source that declares one."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from hls import measured
from hls.measured import (DEV_REFERENCE_FILE, MeasuredFile, MeasuredFiles, PinMismatch, defaults_path, load_pinned,
                          pinned_key, read_pinned, store_key, write_measured, write_pinned)
from hls.obs.conditions import measured_note
from hls.taskref import MeasuredPaths, TaskRef, blob_id, read_task

ASR = TaskRef(module="tasks.multimodal_asr.task")

STORE = MeasuredFile(source="store", path=pinned_key("t", "baseline.json", "a" * 40), blob="a" * 40)
DEFAULT = MeasuredFile(source="defaults", path="tasks/t/defaults/calibration.json", blob="b" * 40)


@pytest.mark.parametrize("source, path", [
    ("store", "measured/../baseline.json"), ("store", "measured/t/other.json"), ("store", "elsewhere/t/baseline.json"),
    # The current file a measurement replaces is never pinned, nor the key of other bytes.
    ("store", "measured/t/baseline.json"), ("store", f"measured/t/{'b' * 40}/baseline.json"),
    ("defaults", "/abs/defaults/baseline.json"), ("defaults", "tasks/../defaults/baseline.json"),
    ("defaults", "tasks/t/baseline.json"), ("defaults", "tasks/t/defaults/removed_requests.json"),
])
def test_a_pin_names_a_key_of_the_measured_area_or_a_tasks_default_and_nothing_else(source, path):
    with pytest.raises(ValidationError):
        MeasuredFile(source=source, path=path, blob="a" * 40)


def test_a_record_notes_a_default_and_never_a_store_measurement():
    assert measured_note(MeasuredFiles()) is None
    assert measured_note(MeasuredFiles(credit=STORE)) is None
    note = measured_note(MeasuredFiles(credit=STORE, calibration=DEFAULT))
    assert note == "used the task's default tasks/t/defaults/calibration.json"
    both = measured_note(MeasuredFiles(credit=DEFAULT.model_copy(update={"path": "tasks/t/defaults/baseline.json"}),
                                       calibration=DEFAULT))
    assert both == ("used the task's defaults tasks/t/defaults/baseline.json and "
                    "tasks/t/defaults/calibration.json")


def test_a_role_gets_a_checked_copy_and_a_changed_file_is_refused(tmp_path):
    store, checkout = tmp_path / "store", tmp_path / "checkout"
    pinned = pinned_key("t", "baseline.json", blob_id(b"credit"))
    (store / pinned).parent.mkdir(parents=True)
    (checkout / "tasks" / "t" / "defaults").mkdir(parents=True)
    (store / pinned).write_bytes(b"credit")
    (checkout / "tasks" / "t" / "defaults" / "calibration.json").write_bytes(b"calibration")
    files = MeasuredFiles(credit=MeasuredFile(source="store", path=pinned, blob=blob_id(b"credit")),
                          calibration=DEFAULT.model_copy(update={"blob": blob_id(b"calibration")}))
    copies = write_pinned(files, tmp_path / "role", checkout=checkout, store_root=store)
    assert (copies.credit.read_bytes(), copies.calibration.read_bytes()) == (b"credit", b"calibration")
    (store / pinned).write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="changed since the plan was made"):
        read_pinned(files.credit, checkout=checkout, store_root=store)
    (store / pinned).unlink()
    with pytest.raises(ValueError, match="cannot be read"):
        write_pinned(files, tmp_path / "role", checkout=checkout, store_root=store)


def _slo(tmp_path: Path):
    from hls.store.posix import PosixStore

    ref = TaskRef(module="tasks.multitenant_slo.task")
    return ref, read_task(ref).task, PosixStore(tmp_path / "store"), defaults_path(ref, "baseline.json").read_bytes()


def test_a_later_measurement_leaves_the_bytes_a_plan_pinned_in_place(tmp_path):
    """A measurement writes its bytes' key beside the current file, and ``find`` pins that key, so
    the next measurement, which replaces the current file, leaves every plan that pinned the
    earlier one able to read it."""
    ref, task, store, default = _slo(tmp_path)
    assert write_measured(store, "multitenant_slo", "baseline.json", default) == store_key("multitenant_slo",
                                                                                          "baseline.json")
    found = measured.find("baseline.json", task_ref=ref, task=task, task_id="multitenant_slo",
                          checkout=Path(__file__).resolve().parents[2], store=store)
    assert found.pin == MeasuredFile(source="store", path=pinned_key("multitenant_slo", "baseline.json",
                                                                     blob_id(default)), blob=blob_id(default))
    assert found.pin.path == f"measured/multitenant_slo/{blob_id(default)}/baseline.json"
    write_measured(store, "multitenant_slo", "baseline.json", b"a later measurement")
    assert store.read_bytes(store_key("multitenant_slo", "baseline.json")) == b"a later measurement"
    assert read_pinned(found.pin, checkout=tmp_path, store_root=store.root) == default


def test_a_current_file_whose_bytes_key_is_not_there_is_not_used(tmp_path):
    """Only a measurement writes the current file, and it writes its bytes' key first; a current
    file without it is said and passed over for the task's default."""
    ref, task, store, default = _slo(tmp_path)
    store.write_bytes(store_key("multitenant_slo", "baseline.json"), default)
    found = measured.find("baseline.json", task_ref=ref, task=task, task_id="multitenant_slo",
                          checkout=Path(__file__).resolve().parents[2], store=store)
    assert found.pin is not None and found.pin.source == "defaults"
    assert found.refused == [f"store measured/multitenant_slo/baseline.json: "
                             f"{pinned_key('multitenant_slo', 'baseline.json', blob_id(default))}, which the "
                             "measurement that wrote it writes first, is not there"]


def test_a_stage_loads_the_task_from_the_bytes_the_plan_pinned_and_refuses_others(tmp_path):
    """``load_pinned`` reads each pinned file once, and the bytes it holds to the pin are the bytes
    the task is loaded from."""
    ref, task, store, default = _slo(tmp_path)
    write_measured(store, "multitenant_slo", "baseline.json", default)
    checkout = Path(__file__).resolve().parents[2]
    pin = measured.find("baseline.json", task_ref=ref, task=task, task_id="multitenant_slo", checkout=checkout,
                        store=store).pin
    files = MeasuredFiles(credit=pin)
    loaded = load_pinned(ref, files, calibration=False, credit=True, checkout=checkout, store_root=store.root)
    assert loaded == read_task(ref, credit=True, files=MeasuredPaths(credit=defaults_path(ref, "baseline.json"))).task
    other = default.replace(b'"drives"', b'"drives" ', 1)
    (store.root / pin.path).write_bytes(other)
    with pytest.raises(PinMismatch, match="changed since the plan was made"):
        load_pinned(ref, files, calibration=False, credit=True, checkout=checkout, store_root=store.root)
    (store.root / pin.path).unlink()
    with pytest.raises(PinMismatch, match="is not there"):
        load_pinned(ref, files, calibration=False, credit=True, checkout=checkout, store_root=store.root)


def test_a_task_that_declares_a_dev_reference_loads_it_as_a_measured_file_and_reads_none_itself():
    """multimodal_asr's task module names the file its dev reference is published under and reads
    no file: the dev reference is a measured file the plan finds, not part of the task or its rows
    entry."""
    loaded = read_task(ASR).task.data_source
    assert (loaded.measured_dev_reference, loaded.dev_reference) == ("oracle_dev.json", {})
    assert measured.needed_files(read_task(ASR).task) == ["baseline.json", DEV_REFERENCE_FILE]
    assert defaults_path(ASR, DEV_REFERENCE_FILE).is_file()
