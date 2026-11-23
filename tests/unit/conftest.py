"""Shared fixtures for the unit suite."""
from __future__ import annotations

import logging

import pytest

from hls.obs.logsetup import OWNED_LOGGERS


@pytest.fixture(autouse=True)
def _restore_hls_logger():
    """Snapshot + restore every logger tree configure_logging owns (``OWNED_LOGGERS``) so that
    configure_logging in one test does not leak handler/level/propagate state into others: harbor's
    tree left without propagation hides its records from a later test's caplog. Also restore the
    root level (configure_logging pins root at WARNING) so it does not leak into later tests."""
    owned = [logging.getLogger(name) for name in OWNED_LOGGERS]
    root = logging.getLogger()
    saved = [(list(lg.handlers), lg.level, lg.propagate) for lg in owned]
    saved_root_level = root.level
    yield
    for lg, (handlers, level, propagate) in zip(owned, saved):
        lg.handlers[:] = handlers
        lg.setLevel(level)
        lg.propagate = propagate
    root.setLevel(saved_root_level)
