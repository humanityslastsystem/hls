from __future__ import annotations

import sys
from pathlib import Path

import pytest

from hls.pipeline import commands
from hls.pipeline.plan import Toolchain
from tests.pipeline.plans import CODE


@pytest.fixture
def login(monkeypatch, tmp_path):
    """The login node as the commands see it: a fixed checkout state and this interpreter, since
    the scratch tree a test runs in is no git checkout."""
    monkeypatch.setattr(commands, "code_state", lambda path: CODE)
    monkeypatch.setattr(commands, "toolchain_here",
                        lambda: Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path))
