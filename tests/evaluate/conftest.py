"""Fixtures for the ``hls grade`` tests: a checkout of the tests' own (``src`` and ``tests``
linked from this repository, and one task package per fixture task), the login node planning
from it, and the fixture task in the image catalog."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from hls.recipes import catalog
from tests.evaluate import fixture
from tests.pipeline.plans import CODE, pin_harbor
from tests.recipes.declared import declare

REPO = Path(__file__).resolve().parents[2]

#: Task package to the fixture expression its ``TASK`` is.
PACKAGES = {"ev_task": "evaluate_task()", "ev_agent": "agent_task()", "ev_pair": "pair_task()", "ev_net": "net_task()",
            "ev_wide": "wide_task()", "ev_art": "evaluate_task()", "ev_alloc": "evaluate_task()",
            "ev_alloc_own": "evaluate_task()", "ev_gpu_alloc": "agent_task()", "ev_cached": "cached_task()",
            "ev_alloc_verify": "evaluate_task()", "ev_alloc_memory": "evaluate_task()",
            "ev_unbounded": "unbounded_task()"}
#: Task package to the fixture expression its ``ARTIFACT`` is, for those that declare one.
ARTIFACTS = {"ev_art": "served_artifact()", "ev_alloc": "allocated_artifact()",
             "ev_alloc_own": "allocated_artifact(driver_cpus=2)", "ev_gpu_alloc": "allocated_artifact()",
             "ev_alloc_verify": "allocated_artifact(verify_time_sec=fixture.VERIFY_TIME_SEC)",
             "ev_alloc_memory": "allocated_artifact(verify_memory_gib=fixture.VERIFY_MEMORY_GIB)"}


@pytest.fixture(scope="session")
def checkout(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("checkout")
    (root / "src").symlink_to(REPO / "src")
    (root / "tests").symlink_to(REPO / "tests")
    for package, expression in PACKAGES.items():
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("")
        artifact = f"ARTIFACT = fixture.{ARTIFACTS[package]}\n" if package in ARTIFACTS else ""
        (root / package / "task.py").write_text(f"from tests.evaluate import fixture\n\nTASK = fixture.{expression}\n"
                                                f"{artifact}")
    return root


@pytest.fixture
def login(monkeypatch, tmp_path, checkout, warmed_servers):
    """The login node as the commands see it, planning from *checkout*: a fixed checkout state
    there, this interpreter, and the checkout on the import path, as the CLI puts it, on a cluster
    whose images warmers built every server image (``warmed_servers``)."""
    code = CODE.model_copy(update={"checkout": checkout})
    monkeypatch.setattr(commands, "checkout_root", lambda: checkout)
    monkeypatch.setattr(commands, "code_state", lambda path: code)
    pin_harbor(monkeypatch, code)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    monkeypatch.syspath_prepend(str(checkout))
    return code


Measure = Callable[..., None]


@pytest.fixture
def in_catalog(monkeypatch) -> Measure:
    """The fixture task's images (``fixture.image_set``) as the catalog reads them: on
    sample_task's layer, a server image of its baseline that starts with ``fixture.BASELINE_ENTRYPOINT``.
    Its modules live outside ``tasks/``, so this names them in the catalog as the fixture task,
    which ``catalog.catalog_task`` would refuse; every other module goes to it. Returns
    ``measure(roles)``, which declares a server image for each of *roles*."""
    catalog_task = catalog.catalog_task
    modules = {f"{package}.task" for package in PACKAGES}
    monkeypatch.setattr(catalog, "catalog_task", lambda m: fixture.TASK_ID if m in modules else catalog_task(m))
    sets = declare(monkeypatch, {})

    def measure(roles: Sequence[str] = ("baseline",)) -> None:
        sets[fixture.TASK_ID] = fixture.image_set(roles)

    measure()
    return measure
