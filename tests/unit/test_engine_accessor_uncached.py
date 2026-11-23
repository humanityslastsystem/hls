"""``EngineAccessor.uncached``: the view for a question about the contestant.

The stored reference answers (``hls.verify.answers``) are an oracle's own answers to
corpus rows, which every arm asks the same. A question whose answer depends on
the contestant's reply, such as teacher-forcing it, could never be served from them, since
no other run asks it, and a verifier says so by asking through ``uncached``. These tests
pin the two views and the accessor's refusals.
"""

from __future__ import annotations

import pytest

from hls.core.capabilities import Generate
from hls.verify.base import ArtifactRollout, EngineAccessor, WiringError


def _accessor(**kw):
    return EngineAccessor(artifact_events=[], **kw)


def test_uncached_returns_the_unwrapped_provider():
    cached, raw = object(), object()
    engine = _accessor(cap_to_provider={Generate: cached},
                       cap_to_uncached={Generate: raw})

    assert engine[Generate] is cached
    assert engine.uncached(Generate) is raw


def test_without_an_uncached_map_both_views_are_the_same():
    """An accessor built with no uncached map answers ``uncached`` from its providers."""
    provider = object()
    engine = _accessor(cap_to_provider={Generate: provider})

    assert engine[Generate] is provider
    assert engine.uncached(Generate) is provider


def test_an_undeclared_capability_still_fails_loud_on_the_uncached_view():
    engine = _accessor(cap_to_provider={}, cap_to_uncached={})

    with pytest.raises(WiringError, match="no declared engine provides it"):
        engine.uncached(Generate)


def test_the_trace_is_not_an_engine():
    """``ArtifactRollout`` resolves to the events, so asking for it uncached is a
    category error rather than a silently empty answer."""
    engine = _accessor(cap_to_provider={Generate: object()})

    with pytest.raises(WiringError, match="not an engine"):
        engine.uncached(ArtifactRollout)
