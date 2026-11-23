"""A pooled gate grades the whole trace ONCE, not each tenant separately.

``q_max`` is a pooled worst-member statistic over the calibration's whole traffic mix,
so the only measurement it licenses is a pooled one. Comparing it against a single
tenant's slice is incoherent, because intrinsic flip rates differ by STRATUM rather than
by implementation quality. A single mean applied per stratum fails the hottest stratum
whatever the server does, and leaves the coldest headroom to be broken in. These tests
pin the arithmetic of grading the pool once.
"""
from __future__ import annotations

import asyncio

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent
from hls.verify.base import Verifier
from hls.verify.verifiers import TokenFaithfulnessVerifier

from tests.unit.calibration_fixtures import make_calibration
from tests.unit.test_verifiers_port import _acc, _ref, _resp


#: A deliberately permissive curve side (a very high clean event rate at the lowest gap
#: level) so the curve rule cannot fire on these traces. These tests are about the RATE
#: rule's POOLED-vs-per-tenant behaviour, and a curve rejection would make a pass/fail
#: assertion here mean something else.
_PERMISSIVE_CURVE = [10_000, 0, 0, 0, 0, 0, 0]


def _cal(q_max=0.004, **overrides):
    """A calibration whose worst engine sits exactly at ``q_max``."""
    return make_calibration(
        engines={"eng": {"rates": [q_max] * 3, "events": _PERMISSIVE_CURVE}},
        **overrides)


def _event(tenant, req, n_tokens):
    """One event stamped with a tenant, so event_key is tenant-namespaced."""
    out = Envelope(request_id=req, contents=[TextPayload(text="a", tokens=[1] * n_tokens)])
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")],
                   metadata={"messages": [{"role": "user", "content": "Q?"}],
                             "tenant_id": tenant, "max_tokens": 1_000_000})
    return TraceEvent(request_id=req, input=inp, response=_resp(out),
                      t_send=0.0, t_recv=1.0)


def test_token_faithfulness_is_declared_pooled():
    # The marker is what makes the runner grade the whole trace as one population.
    assert TokenFaithfulnessVerifier.pooled is True


def test_per_tenant_verification_is_not_the_default():
    # A content verifier's ground truth IS its tenant's dataset, so pooling must be
    # opt-in rather than something every verifier inherits.
    assert Verifier.pooled is False


def test_hot_stratum_alone_does_not_reject_when_the_pool_passes():
    """One hot stratum over the gate, the pooled rate under it.

    Rates are scaled to the same relationship: gate = 2*q_max = 0.8%; the hot tenant
    sits at 1.0% (over), the pool at 0.4% (under). Graded as one population it passes.
    """
    cal = _cal(q_max=0.004)   # gate = 0.8%
    v = TokenFaithfulnessVerifier(reference=_ref(), calibration=cal)

    hot = [_event("long_session", "r0", 1000)]          # 1000 tokens, 10 flips = 1.0%
    cold = [_event("chat", f"c{i}", 1000) for i in range(4)]   # 4000 tokens, 6 flips
    gaps_hot = [[0.5] * 10 + [0.0] * 990]
    gaps_cold = [[0.5] * 2 + [0.0] * 998, [0.5] * 2 + [0.0] * 998,
                 [0.5] * 1 + [0.0] * 999, [0.5] * 1 + [0.0] * 999]

    hot_only = asyncio.run(v.verify(_acc(hot, gaps_hot)))
    assert hot_only.diagnostics["flip_rate"] == 0.010
    assert hot_only.passed is False, "the hot stratum alone trips a pooled-derived gate"

    pooled = asyncio.run(v.verify(_acc(hot + cold, gaps_hot + gaps_cold)))
    assert pooled.diagnostics["n_tokens"] == 5000
    assert pooled.diagnostics["n_flips"] == 16
    assert pooled.diagnostics["flip_rate"] == 0.0032
    assert pooled.passed is True, "graded as one population the same trace passes"


def test_pooled_event_keys_are_tenant_namespaced():
    # A pooled map spans tenants, and request_id renumbers from 0 within each -- two
    # tenants over the same dataset path would silently clobber each other. The gate is
    # trace-level, so the map that has to carry the namespaced keys is the per-event
    # detail rather than a per-event verdict.
    cal = _cal()
    v = TokenFaithfulnessVerifier(reference=_ref(), calibration=cal)
    events = [_event("chat", "r0", 1000), _event("api_short", "r0", 1000)]
    verdict = asyncio.run(v.verify(_acc(events, [[0.0] * 1000, [0.0] * 1000])))
    assert verdict.per_event == {}
    assert set(verdict.event_diagnostics) == {"chat:r0", "api_short:r0"}
