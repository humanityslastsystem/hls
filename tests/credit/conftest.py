"""Fixtures for the baseline credit tests: a committed checkout of the tests' own (``src`` and
``tests`` linked from this repository, one task package per fixture task, and harbor pinned at
the commit the login node's checkout state names), on this process's import path; the login
node planning from it; and the credit task in the image catalog under the evaluate tests' task
id, its measured baseline recorded."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from hls.recipes import catalog
from tests.evaluate import fixture as evaluate
from tests.pipeline.plans import CODE
from tests.recipes.declared import declare

REPO = Path(__file__).resolve().parents[2]
#: The digest the catalog records for the credit task's measured baseline unless a test records another.
BASELINE_SHA = "1" * 64
#: Task package to the fixture module and the expression its ``TASK`` is.
PACKAGES: dict[str, tuple[str, str]] = {
    "cr_task": ("tests.credit.fixture", "credit_task()"),
    "cr_goodput": ("tests.credit.fixture", "credit_task(latency=False)"),
    "cr_nobase": ("tests.credit.fixture", "credit_task(baseline=None)"),
    "cr_latency": ("tests.finalize.fixture", "latency_task()"),
    "cr_single": ("tests.finalize.fixture", "t0_single_task()"),
    "cr_crash": ("tests.credit.fixture", "credit_task(baseline=fixture.CRASHING)"),
    "cr_stream": ("tests.credit.fixture", "credit_task(stream=True)"),
}


def _git(repo: Path, *args: str) -> None:
    """Run git in *repo* as a test's own user, whatever the machine's git configuration holds."""
    config = ("-c", "user.name=credit", "-c", "user.email=credit@example.com", "-c", "commit.gpgsign=false",
              "-c", "maintenance.auto=false", "-c", "gc.auto=0")
    subprocess.run(["git", *config, "-C", str(repo), *args], capture_output=True, check=True)


@pytest.fixture(scope="session")
def checkout(tmp_path_factory) -> Iterator[Path]:
    """The checkout, committed, with harbor pinned at ``CODE.harbor_commit``; also on this
    process's import path (last, so only its task packages resolve there)."""
    root = tmp_path_factory.mktemp("checkout")
    (root / "src").symlink_to(REPO / "src")
    (root / "tests").symlink_to(REPO / "tests")
    for package, (module, expression) in PACKAGES.items():
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("")
        (root / package / "task.py").write_text(f"import {module} as fixture\n\nTASK = fixture.{expression}\n")
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "src", "tests", *PACKAGES)
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{CODE.harbor_commit},harbor")
    _git(root, "commit", "-q", "-m", "a checkout")
    sys.path.append(str(root))
    yield root
    sys.path.remove(str(root))


@pytest.fixture
def login(monkeypatch, tmp_path, checkout, warmed_servers):
    """The login node as the commands see it, planning from *checkout*: a fixed, committed
    checkout state there and this interpreter, on a cluster whose images warmers built every
    measured image (``warmed_servers``)."""
    code = CODE.model_copy(update={"checkout": checkout})
    monkeypatch.setattr(commands, "checkout_root", lambda: checkout)
    monkeypatch.setattr(commands, "code_state", lambda path: code)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    return code


Record = Callable[[Mapping[str, str]], None]


@pytest.fixture
def in_catalog(monkeypatch) -> None:
    """The credit task's modules in the image catalog as the evaluate tests' task, with that
    task's images (``evaluate.image_set``): on sample_task's layer, a server image of its baseline that
    starts as the evaluate tests' baseline does."""
    catalog_task = catalog.catalog_task
    modules = {f"{package}.task" for package in PACKAGES}
    monkeypatch.setattr(catalog, "catalog_task", lambda m: evaluate.TASK_ID if m in modules else catalog_task(m))
    declare(monkeypatch, {evaluate.TASK_ID: evaluate.image_set(("baseline",))})
