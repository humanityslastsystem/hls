"""A checkout for the verify stage's tests: ``src`` and ``tests`` linked from this repository,
and one task package per fixture task, each with the calibration it needs among its defaults
(``<package>/defaults/calibration.json``, ``hls.measured``)."""

from __future__ import annotations

import importlib
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from hls.measured import CALIBRATION_FILE, DEFAULTS_DIR, MeasuredFile, MeasuredFiles
from hls.taskref import blob_id
from hls.verify.calibration import StackInfo
from tests.unit.calibration_fixtures import make_calibration

REPO = Path(__file__).resolve().parents[2]

#: Task package to the fixture expression its ``TASK`` is, and the calibration overrides of its
#: default calibration (None: no calibration.json).
PACKAGES: dict[str, tuple[str, dict | None]] = {
    "vstage_gate": ("gate_task()", {}),
    "vstage_drift": ("gate_task()", {"stack": StackInfo(torch="2.5.0", runtime={"cuda": "12.4", "cudnn": "90100"},
                                                        device_name="B200")}),
    "vstage_silicon": ("gate_task(fixture.ForeignSiliconReference)", {}),
    "vstage_nocal": ("gate_task()", None),
    "vstage_split": ("split_gate_task()", {}),
    "vstage_pair": ("pair_task()", None),
    "vstage_pair_placed": ("pair_task(fixture.PlacedGeneratorReference, fixture.PlacedForcerReference)",
                           None),
    "vstage_pairstack": ("pair_task(forcer=fixture.BrokenStackForcerReference)", None),
    "vstage_badload": ("pair_task(generator=fixture.LoadRefusingGeneratorReference)", None),
    "vstage_badstop": ("pair_task(forcer=fixture.StopFailingForcerReference)", None),
    "vstage_badstops": ("pair_task(fixture.StopFailingGeneratorReference,"
                        " fixture.StopFailingForcerReference)", None),
    "vstage_wide": ("gate_task(fixture.WideReference)", None),
    "vstage_metric": ("metric_task()", None),
    "vstage_cached": ("cached_task()", None),
    "vstage_cachedweights": ("cached_task(reference=fixture.WeightsGeneratorReference)", None),
    "vstage_cachedchat": ("cached_task(turns=2.5)", None),
    "vstage_cachedagent": ("cached_task(agent=True)", None),
    "vstage_fill": ("fill_task()", None),
    "vstage_flaky": ("fill_task(fixture.FlakyGeneratorReference)", None),
    "vstage_multi": ("fill_task(multi_turn=True)", None),
    "vstage_fill_mixed": ("mixed_fill_task()", None),
    "vstage_devref": ("dev_reference_task()", None),
    "vstage_uncached": ("fill_task(cache=False)", None),
    "vstage_importing": ("importing_task()", None),
}


@pytest.fixture(scope="session")
def checkout(tmp_path_factory) -> Iterator[Path]:
    """The checkout, also on this process's import path (last, so only its task packages
    resolve there)."""
    root = tmp_path_factory.mktemp("checkout")
    (root / "src").symlink_to(REPO / "src")
    (root / "tests").symlink_to(REPO / "tests")
    for package, (expression, _) in PACKAGES.items():
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("")
        (root / package / "task.py").write_text(f"from tests.verify_stage import fixture\n\nTASK = fixture.{expression}\n")
    sys.path.append(str(root))
    for package, (_, overrides) in PACKAGES.items():
        if overrides is not None:
            cal = make_calibration(importlib.import_module(f"{package}.task").TASK, **overrides)
            defaults = root / package / DEFAULTS_DIR
            defaults.mkdir(parents=True)
            (defaults / CALIBRATION_FILE).write_text(json.dumps(cal.model_dump(mode="json")))
    yield root
    sys.path.remove(str(root))


def default_calibration(checkout: Path, module: str) -> Path | None:
    """The default calibration of the task module *module* in *checkout*; None when it has none."""
    path = checkout / module.split(".")[0] / DEFAULTS_DIR / CALIBRATION_FILE
    return path if path.is_file() else None


def measured_pins(checkout: Path, module: str) -> MeasuredFiles:
    """The pins a plan of *module* makes of its defaults in *checkout*, as ``hls run`` and
    ``hls grade`` pin a default file."""
    path = default_calibration(checkout, module)
    if path is None:
        return MeasuredFiles()
    return MeasuredFiles(calibration=MeasuredFile(source="defaults", path=path.relative_to(checkout).as_posix(),
                                                  blob=blob_id(path.read_bytes())))
