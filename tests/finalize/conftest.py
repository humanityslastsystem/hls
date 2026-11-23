"""A checkout for the score and finalize stages' tests: ``src`` and ``tests`` linked from this
repository, and one task package per fixture task, each with the baseline credit it needs among
its defaults (``<package>/defaults/baseline.json``, ``hls.measured``)."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.support.credit import write_credit
from tests.finalize import fixture

REPO = Path(__file__).resolve().parents[2]

#: Task package to the fixture expression its ``TASK`` is, and its default baseline credit:
#: "measured" for one measured for the task, "other_gpu" for one measured on another GPU
#: kind, None for no ``baseline.json``.
PACKAGES: dict[str, tuple[str, str | None]] = {
    "fin_goodput": ("goodput_task()", None),
    "fin_session": ("goodput_task(TrajectorySessionUnit())", None),
    "fin_latency": ("latency_task()", "measured"),
    "fin_t0": ("t0_task()", "measured"),
    "fin_single": ("t0_single_task()", "measured"),
    "fin_stale": ("latency_task()", "other_gpu"),
    "fin_nocredit": ("latency_task()", None),
    "fin_vacuous": ("vacuous_task()", None),
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
        (root / package / "task.py").write_text(
            "from hls.scoring import TrajectorySessionUnit  # noqa: F401\n"
            f"from tests.finalize import fixture\n\nTASK = fixture.{expression}\n")
    sys.path.append(str(root))
    for package, (_, credit) in PACKAGES.items():
        if credit is not None:
            module = f"{package}.task"
            task = importlib.import_module(module).TASK
            gpu_kind = "B200" if credit == "other_gpu" else None
            defaults = root / package / "defaults"
            defaults.mkdir(parents=True)
            write_credit(defaults, fixture.credit(task, module, gpu_kind=gpu_kind))
    yield root
    sys.path.remove(str(root))

