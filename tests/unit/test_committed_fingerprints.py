"""Each graded task module hashes to the workload fingerprint its default baseline credit and
calibration record, and those name the module.

A task's default measurements are data in its ``defaults/`` directory, never among its source
files (``hls.measured``). The fingerprint hashes the dumped spec of the task's model, contract,
seed, tenants, references and agent seam. The records are read at the head under test. A change
that moves a fingerprint comes with a restamp commit beside it, merged with it, whose message holds
the old and new values and the proof that the task did not change; a change that moves one without
that fails here."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from hls.measured import BASELINE_CREDIT_FILE, CALIBRATION_FILE, DEFAULTS_DIR, DEV_REFERENCE_FILE
from hls.recipes import catalog
from hls.verify.calibration import workload_fingerprint

REPO = Path(__file__).resolve().parents[2]
#: The files that hold a measurement's output: the baseline credit, the calibration and the dev
#: split's reference answers.
MEASURED_FILES = (BASELINE_CREDIT_FILE, CALIBRATION_FILE, DEV_REFERENCE_FILE)

#: Each graded module, and the records its task ships as defaults.
GRADED = {
    "tasks.multitenant_slo.task": (BASELINE_CREDIT_FILE, CALIBRATION_FILE),
    "tasks.tab_completion.task": (BASELINE_CREDIT_FILE, CALIBRATION_FILE),
    "tasks.multimodal_tts.task": (BASELINE_CREDIT_FILE, CALIBRATION_FILE),
    "tasks.multimodal_asr.task": (BASELINE_CREDIT_FILE,),
    "tasks.multimodal_omni.task": (BASELINE_CREDIT_FILE,),
    "tasks.deepresearch.task": (BASELINE_CREDIT_FILE, CALIBRATION_FILE),
    "tasks.agentic_coding.task": (BASELINE_CREDIT_FILE, CALIBRATION_FILE),
}


def _defaults(module: str) -> Path:
    return REPO / "tasks" / module.split(".")[1] / DEFAULTS_DIR


def _recorded(module: str, name: str) -> tuple[str, str]:
    """The module and the workload fingerprint the default record *name* names."""
    data = json.loads((_defaults(module) / name).read_text())
    holder = data["campaign"] if name == BASELINE_CREDIT_FILE else data
    return holder["task"], holder["workload_fingerprint"]


@pytest.mark.parametrize("module", sorted(GRADED))
def test_each_graded_module_hashes_to_the_fingerprint_its_default_records_hold(module):
    fingerprint = workload_fingerprint(importlib.import_module(module).TASK)
    for name in GRADED[module]:
        assert _recorded(module, name) == (module, fingerprint), name


def test_the_tasks_that_ship_a_record_are_the_graded_ones():
    # A record in any other task would name a fingerprint nothing ties.
    holding = {d.name for d in (REPO / "tasks").iterdir()
               if any((d / DEFAULTS_DIR / name).is_file() for name in (BASELINE_CREDIT_FILE, CALIBRATION_FILE))}
    assert holding == {module.split(".")[1] for module in GRADED}


@pytest.mark.parametrize("module", sorted(GRADED))
def test_each_graded_task_ships_exactly_its_records(module):
    for name in (BASELINE_CREDIT_FILE, CALIBRATION_FILE):
        assert (_defaults(module) / name).is_file() == (name in GRADED[module]), name


def test_only_a_defaults_directory_holds_a_measurement():
    """A measurement is data beside the task's source, in its ``defaults/`` directory, never among
    the source files themselves."""
    held = sorted(str(path.relative_to(REPO)) for name in MEASURED_FILES for path in (REPO / "tasks").rglob(name)
                  if path.parent.name != DEFAULTS_DIR)
    assert held == []


@pytest.mark.parametrize("module", sorted(GRADED))
def test_each_graded_module_is_the_catalogs_module_for_its_directory(module):
    directory = module.split(".")[1]
    assert catalog.task_module(directory) == module
    assert catalog.catalog_task(module) == directory
