"""Fixtures for the ``hls run`` tests: a checkout of the tests' own (``src`` and ``tests`` linked
from this repository, and one task package per fixture variant), the login node planning from
it, and the fixture task in the image catalog; the probe's sealed checks answered as a working
seal answers them; and, for every test here, the CLI's logging setup stubbed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from hls.recipes import catalog
from hls.egress.policy import PROBE_HOST
from hls.agent import probe
from tests.agent.build_fixture import IMAGE_SET
from tests.agent.probe_answers import SEAL_ANSWERS, answering, held_inbound
from tests.evaluate.fixture import TASK_ID
from tests.pipeline.plans import CODE, pin_harbor
from tests.recipes.declared import declare

REPO = Path(__file__).resolve().parents[2]

#: Task package to the fixture expression its ``ARTIFACT`` is; None for a package without one.
BUILD_PACKAGES = {"bd_task": "artifact()", "bd_effort": "effort_artifact()", "bd_turns": "soft_turns_artifact()",
                  "bd_alloc": "allocation_artifact()", "bd_none": None, "bd_credit": None}
#: Task package to the module and the expression its ``TASK`` is, for those whose task is not the
#: build task (``build_fixture.build_task()``): the credit tests' task, which scores against its
#: measured baseline.
BUILD_TASKS = {"bd_credit": ("tests.credit.fixture", "credit_task()")}


@pytest.fixture(autouse=True)
def cli_logging_untouched(monkeypatch):
    """The CLI's commands configure hls logging for a terminal, which stops the ``hls`` logger
    propagating, so every later test's ``caplog`` sees nothing. A test that invokes the CLI
    leaves logging as pytest set it."""
    from hls import cli

    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)


@pytest.fixture(scope="session")
def build_checkout(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("checkout")
    (root / "src").symlink_to(REPO / "src")
    (root / "tests").symlink_to(REPO / "tests")
    for package, expression in BUILD_PACKAGES.items():
        (root / package).mkdir()
        (root / package / "__init__.py").write_text("")
        artifact = f"ARTIFACT = build_fixture.{expression}\n" if expression is not None else ""
        if package in BUILD_TASKS:
            module, task = BUILD_TASKS[package]
            (root / package / "task.py").write_text(
                f"from tests.agent import build_fixture\nimport {module} as fixture\n\n"
                f"TASK = fixture.{task}\n{artifact}")
        else:
            (root / package / "task.py").write_text(
                f"from tests.agent import build_fixture\n\nTASK = build_fixture.build_task()\n{artifact}")
    return root


@pytest.fixture
def build_login(monkeypatch, tmp_path, build_checkout):
    """The login node as the commands see it, planning from *build_checkout*: a fixed checkout
    state there, this interpreter, and the checkout on the import path, as the CLI puts it."""
    code = CODE.model_copy(update={"checkout": build_checkout})
    monkeypatch.setattr(commands, "checkout_root", lambda: build_checkout)
    monkeypatch.setattr(commands, "code_state", lambda path: code)
    pin_harbor(monkeypatch, code)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
    monkeypatch.syspath_prepend(str(build_checkout))
    return code


@pytest.fixture
def build_catalog(monkeypatch):
    """The fixture task's images (``build_fixture.IMAGE_SET``) as the catalog reads them: on
    sample_task's layer, declaring none of its own. Its modules live outside ``tasks/``, so this
    names them in the catalog as the fixture task, which ``catalog.catalog_task`` would refuse;
    every other module goes to it."""
    catalog_task = catalog.catalog_task
    modules = {f"{package}.task" for package in BUILD_PACKAGES}
    monkeypatch.setattr(catalog, "catalog_task", lambda m: TASK_ID if m in modules else catalog_task(m))
    declare(monkeypatch, {TASK_ID: IMAGE_SET})



@pytest.fixture
def fake_seal(monkeypatch):
    """The probe's sealed checks answered as a working seal answers them (``SEAL_ANSWERS``, and
    ``held_inbound`` for the inbound check), and the fake interceptor recording the refused host by
    its policy. A test plants a failure with ``answering``, by setting ``FAKE_PROXY_POLICY_REFUSED``
    again, or by putting the real ``probe.inbound_check`` back."""
    answering(SEAL_ANSWERS, monkeypatch)
    monkeypatch.setattr(probe, "inbound_check", held_inbound)
    monkeypatch.setenv("FAKE_PROXY_POLICY_REFUSED", json.dumps([PROBE_HOST]))
