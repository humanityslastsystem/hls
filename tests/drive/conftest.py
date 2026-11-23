"""Shared fixtures for the drive tests."""
from __future__ import annotations

import logging

import pytest

from hls.obs.logsetup import OWNED_LOGGERS


@pytest.fixture(autouse=True)
def _restore_logging():
    """The driver role configures logging as a program does (``configure_logging``): its own
    handler on the owned trees, which stop propagating, and the root at WARNING. Tests run the
    role in this process, so put every one of those back after each test."""
    root = logging.getLogger()
    owned = [logging.getLogger(name) for name in OWNED_LOGGERS]
    saved = [(list(lg.handlers), lg.level, lg.propagate) for lg in owned]
    saved_root_level = root.level
    yield
    for lg, (handlers, level, propagate) in zip(owned, saved):
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate
    root.setLevel(saved_root_level)
