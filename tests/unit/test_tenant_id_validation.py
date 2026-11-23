"""Tenant.id charset validation.

Tenant.id names the directory its rows lie in, tenants/<id>, in the provision cache
entry and under every role's /data, so it must stay one plain path component: "/" or
"." / ".." would escape the per-tenant directory. Tenant.id must match
[A-Za-z0-9._-]+ and reject "." / "..".
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify import LetterMatchMetric, MetricVerifier


def _wl() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(), slo=NoSLO())


def _tenant(tid: str) -> Tenant:
    return Tenant(id=tid, workload=_wl(),
                  verifiers=[MetricVerifier(metric=LetterMatchMetric())])


@pytest.mark.parametrize("tid", ["t0", "t00", "corp-a", "corp_a", "corp.a", "A1", "x"])
def test_valid_ids_accepted(tid):
    assert _tenant(tid).id == tid


@pytest.mark.parametrize("tid", [
    "corp a",      # space: not a plain path component
    "corp/a",      # slash: escapes the tenants/<id> directory
    "caf\u00e9",   # non-ascii (cafe + U+00E9); \u keeps source ASCII
    "a@b",         # "@": outside the charset
    "",            # empty: no path component at all
    ".",           # current-dir traversal
    "..",          # parent-dir traversal
])
def test_unsafe_ids_rejected(tid):
    with pytest.raises(ValidationError):
        _tenant(tid)
