"""Fixtures for the calibration tests: a checkout of the tests' own (``src`` and ``tests``
linked from this repository, and one task package per fixture task), the login node planning
from it, the fixture task in the image catalog with the measured images a calibration runs on,
the test split's rows as a role's container sees them, and the fake engines built in this
process."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

import hls.calibrate as calibrate
from hls.obs.logsetup import OWNED_LOGGERS
from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from hls.recipes import catalog
from tests.calibration import engines, fixture
from tests.pipeline.plans import CODE
from tests.recipes.declared import declare

REPO = Path(__file__).resolve().parents[2]
#: Task package to the fixture expression its ``TASK`` is.
PACKAGES = {"cal_flips": "flips_task()", "cal_report": "report_task()", "cal_capped": "capped_task()",
            "cal_agentic": "agentic_task()", "cal_audio": "audio_task()", "cal_plain": "plain_task()",
            "cal_reports": "two_reports_task()", "cal_chunks": "two_chunks_task()"}


@pytest.fixture(autouse=True)
def _restore_logging():
    """A role and a command configure logging as a program does (``configure_logging``): their
    own handler on the owned trees, which stop propagating, and the root at WARNING. Tests run
    them in this process, so put every one of those back after each test."""
    root = logging.getLogger()
    owned = [logging.getLogger(name) for name in OWNED_LOGGERS]
    saved = [(list(lg.handlers), lg.level, lg.propagate) for lg in owned]
    saved_root_level = root.level
    yield
    for lg, (handlers, level, propagate) in zip(owned, saved, strict=True):
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate
    root.setLevel(saved_root_level)


def _git(repo: Path, *args: str) -> None:
    """Run git in *repo* as the tests' own user, whatever the machine's git configuration holds."""
    config = ("-c", "user.name=calibration", "-c", "user.email=calibration@example.com", "-c", "commit.gpgsign=false",
              "-c", "maintenance.auto=false", "-c", "gc.auto=0")
    subprocess.run(["git", *config, "-C", str(repo), *args], capture_output=True, check=True)


@pytest.fixture(scope="session")
def checkout(tmp_path_factory) -> Path:
    """The checkout, committed, its HEAD pinning harbor at the commit ``CODE`` records, so a
    scored command's committed-tree check holds for it."""
    root = tmp_path_factory.mktemp("checkout")
    (root / "src").symlink_to(REPO / "src")
    (root / "tests").symlink_to(REPO / "tests")
    for package, expression in PACKAGES.items():
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("")
        recorder = "record_removed_rows = fixture.record_removed_rows\n" if package == "cal_capped" else ""
        (root / package / "task.py").write_text(f"from tests.calibration import fixture\n\n"
                                                f"TASK = fixture.{expression}\n{recorder}")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{CODE.harbor_commit},harbor")
    _git(root, "commit", "-q", "-m", "the calibration tests' checkout")
    return root


@pytest.fixture
def login(monkeypatch, tmp_path, checkout, warmed_servers):
    """The login node as the commands see it, planning from *checkout*: a fixed, committed
    checkout state there, this interpreter, and the checkout on the import path, as the CLI puts
    it, on a cluster whose images warmers built every server image (``warmed_servers``)."""
    code = CODE.model_copy(update={"checkout": checkout})
    monkeypatch.setattr(commands, "checkout_root", lambda: checkout)
    monkeypatch.setattr(commands, "code_state", lambda path: code)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    monkeypatch.syspath_prepend(str(checkout))
    return code


Record = Callable[..., None]


@pytest.fixture
def in_catalog(monkeypatch) -> Record:
    """The fixture tasks' modules in the image catalog as the evaluate tests' task, on
    sample_task's layer, with a server image of its reference and of both pool members. Returns
    ``record(roles=fixture.ROLES)``, which declares a server image for each of *roles*."""
    catalog_task = catalog.catalog_task
    modules = {f"{package}.task" for package in PACKAGES}
    monkeypatch.setattr(catalog, "catalog_task", lambda m: fixture.TASK_ID if m in modules else catalog_task(m))
    sets = declare(monkeypatch, {})

    def record(roles: Sequence[str] = fixture.ROLES) -> None:
        sets[fixture.TASK_ID] = fixture.image_set(roles)

    record()
    return record


@pytest.fixture
def rows(tmp_path) -> Path:
    """The test split's rows of the fixture tasks' one tenant, laid out as a role's container
    sees them under ``/data``."""
    root = tmp_path / "data"
    split = root / "tenants" / "t0" / "test"
    split.mkdir(parents=True)
    (split / "rows.json").write_text(json.dumps(fixture.TEST_ROWS))
    return root


@pytest.fixture
def fake_engines(monkeypatch) -> list[tuple[str, int | None]]:
    """``hls.calibrate`` builds the engines of ``tests.calibration.engines`` in this process;
    returns the reference loads they record."""
    engines.LOADS.clear()
    engines.MEMBER_LOADS.clear()
    engines.MEMBER_UNLOADS.clear()
    monkeypatch.setattr(calibrate, "HFEngine", engines.FakeReference)
    monkeypatch.setattr(calibrate, "member_engines", engines.members)
    return engines.LOADS
