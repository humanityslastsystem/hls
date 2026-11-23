"""A reply served past the output cap its request was stamped with fails the token-level gate on
its over-cap rule whatever its tokens are, so the gate never teacher-forces it: pricing it could
change no verdict, and a server that ignores ``max_tokens`` would otherwise choose how much the
reference must hold (a 2,000,000-token reply is 24 GiB of fp32 keys and values at 131k tokens).

The verdicts of a gate that teacher-forces every reply and checks the cap afterwards are
recorded in ``data/token_gate_over_cap.json``, made by ``today_json`` below over the two traces
``published_like`` builds. They are never regenerated from the skipping verifier: they are what
the skip must leave unchanged."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path

from hls.core.capabilities import StackReport, TeacherForce
from hls.core.decode import Greedy
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.reference.references import HFReference
from hls.verify import EngineAccessor, TokenFaithfulnessVerifier
from hls.verify.verifiers import SHOWN_EVENTS
from tests.unit.calibration_fixtures import make_calibration

#: The recorded verdicts over the two traces.
TODAY = Path(__file__).parent / "data" / "token_gate_over_cap.json"
#: The reported stack the calibration fixture was taken on.
STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"}, "device_name": "B200"}
#: A multitenant_slo-shaped mix: each tenant's stamped output cap.
CAPS = {"chat": 1024, "api_short": 256, "reasoning": 4096, "long_session": 2048}
#: Requests per tenant.
PER_TENANT = 12
#: The replies served past their cap, by tenant and request, with how many tokens each served.
OVER_CAP = {("chat", "r3"): 1025, ("reasoning", "r7"): 8192, ("api_short", "r11"): 300}
#: The fields of the token gate's diagnostics that the reference's answers decide: the pooled
#: measurement and what is read against it. The flip curve's ceilings are recomputed for the
#: pooled length, so the whole curve is among them.
PRICING = {"n_tokens", "n_flips", "flip_rate", "max_gap", "rate_ratio", "flip_curve", "n_nan_gaps", "n_inf_gaps"}


def _event(tenant: str, rid: str, n: int, cap: int) -> TraceEvent:
    inp = Envelope(request_id=rid, contents=[TextPayload(text=f"{tenant} {rid}?")],
                   metadata={"messages": [{"role": "user", "content": f"{tenant} {rid}?"}],
                             "tenant_id": tenant, "max_tokens": cap})
    out = Envelope(request_id=rid, contents=[TextPayload(text="x" * n, tokens=[7] * n)])
    return TraceEvent(request_id=rid, input=inp, response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0)


def published_like(*, over_cap: bool) -> list[TraceEvent]:
    """Twelve requests for each of four tenants, most under their cap and some at it, and with
    *over_cap* the three replies of ``OVER_CAP`` past theirs, each in place of its request."""
    rng = random.Random(20261005)
    events = []
    for tenant, cap in CAPS.items():
        for i in range(PER_TENANT):
            rid = f"r{i}"
            n = cap if rng.random() < 0.2 else rng.randint(1, cap - 1)
            if (tenant, rid) in OVER_CAP:
                if not over_cap:
                    continue
                n = OVER_CAP[tenant, rid]
            events.append(_event(tenant, rid, n, cap))
    return events


class Reference:
    """A reference that teacher-forces as the engine shapes it, each item's gaps drawn from a
    generator seeded by its request and length so that an item prices the same in any batch,
    and that records every item it was asked. A position is a flip with probability *p_flip*,
    under the calibration's lowest gap level, so the rate rule reads it and the curve does not."""

    def __init__(self, p_flip: float = 0.002) -> None:
        #: How often a position is a flip.
        self.p_flip = p_flip
        #: Each item asked, as its request id and served length.
        self.asked: list[tuple[str, int]] = []

    async def teacher_force(self, items: list[Envelope]) -> list[list[float]]:
        out = []
        for item in items:
            n, cap = len(item.first_of("text").tokens), item.metadata["max_tokens"]
            self.asked.append((item.request_id, n))
            rng = random.Random(f"{item.request_id}/{n}/{cap}")
            # One row per served token, and the stop row of a reply short of its cap.
            rows = n + (1 if n < cap else 0)
            out.append([rng.choice((0.1, 0.2, 0.3)) if rng.random() < self.p_flip else 0.0 for _ in range(rows)])
        return out

    async def stack_report(self) -> dict:
        return STACK


def verify(events: list[TraceEvent], reference: Reference | None = None):
    """The token gate's verdict on *events*, with *reference* answering."""
    cal = make_calibration(engines={"eng": {"rates": [0.004, 0.004, 0.004], "events": [2, 0, 0, 0, 0, 0, 0],
                                            "exposure": 1_000_000}})
    v = TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()), calibration=cal,
                                  allow_underpowered_trace=True)
    ref = reference or Reference()
    acc = EngineAccessor(artifact_events=events,
                         cap_to_provider={TeacherForce: ref.teacher_force, StackReport: ref.stack_report})
    return asyncio.run(v.verify(acc))


def today_json() -> dict:
    """The record ``TODAY`` holds: the verdict over each trace, as JSON."""
    return {name: json.loads(verify(published_like(over_cap=flag)).model_dump_json())
            for name, flag in (("with_over_cap", True), ("without_over_cap", False))}


def _keys() -> list[str]:
    """The event keys of the replies past their cap, in the trace's order."""
    return [ev.event_key for ev in published_like(over_cap=True) if (ev.tenant_id, ev.request_id) in OVER_CAP]


def test_the_verdict_on_an_over_cap_trace_is_todays_but_for_the_pricing_of_the_replies_it_skips():
    today = json.loads(TODAY.read_text())
    before, unpriced = today["with_over_cap"], today["without_over_cap"]
    now = json.loads(verify(published_like(over_cap=True)).model_dump_json())
    assert now["passed"] is before["passed"] is False
    assert now["per_event"] == before["per_event"] and now["failure_mode"] == before["failure_mode"]
    # Every rule and count is today's, the over-cap count and the rejections among them.
    kept = {k: v for k, v in now["diagnostics"].items() if k not in PRICING | {"over_cap_events"}}
    assert kept == {k: v for k, v in before["diagnostics"].items() if k not in PRICING}
    assert kept["n_over_cap_events"] == len(OVER_CAP) and kept["rejections"] == ["over_cap_replies"]
    # The pooled measurement is the one over the replies the reference saw, which today's
    # verifier gives the trace without them.
    assert {k: now["diagnostics"][k] for k in PRICING} == {k: unpriced["diagnostics"][k] for k in PRICING}
    assert {k: before["diagnostics"][k] for k in PRICING} != {k: unpriced["diagnostics"][k] for k in PRICING}
    # The skipped replies are named, and only their records are missing.
    assert now["diagnostics"]["over_cap_events"] == _keys()
    assert now["event_diagnostics"] == {k: v for k, v in before["event_diagnostics"].items() if k not in _keys()}


def test_a_two_million_token_reply_never_reaches_the_reference_and_the_trace_fails_as_today():
    ref = Reference(p_flip=0.0)
    events = [_event("t0", "r0", 300, 16_384), _event("t0", "r1", 2_000_000, 16_384)]
    verdict = verify(events, ref)
    assert ref.asked == [("r0", 300)]
    assert verdict.passed is False
    assert verdict.diagnostics["rejections"] == ["over_cap_replies"]
    assert verdict.diagnostics["n_over_cap_events"] == 1
    assert verdict.diagnostics["over_cap_events"] == ["t0:r1"]


def test_a_reply_at_its_cap_is_still_priced():
    ref = Reference(p_flip=0.0)
    verdict = verify([_event("t0", "r0", 256, 256), _event("t0", "r1", 255, 256)], ref)
    assert ref.asked == [("r0", 256), ("r1", 255)]
    d = verdict.diagnostics
    assert (d["n_over_cap_events"], d["over_cap_events"], d["n_capped_events"]) == (0, [], 1)
    assert d["n_tokens"] == 256 + 256 and set(verdict.event_diagnostics) == {"t0:r0", "t0:r1"}


def test_a_trace_whose_every_reply_ran_past_its_cap_prices_nothing_and_fails():
    ref = Reference()
    verdict = verify([_event("t0", "r0", 257, 256)], ref)
    assert ref.asked == []
    assert verdict.passed is False
    assert verdict.diagnostics["rejections"] == ["no_teacher_forced_tokens", "over_cap_replies"]


def test_the_verdict_counts_every_reply_past_its_cap_and_names_the_first_few():
    """A server that ignores every cap must not make the verdict as long as its trace: like the
    untokenized replies, the over-cap ones are counted whole and the first ``SHOWN_EVENTS`` named."""
    ref = Reference(p_flip=0.0)
    events = [_event("t0", "r0", 300, 16_384)] + [_event("t0", f"r{i}", 257, 256) for i in range(1, 8)]
    d = verify(events, ref).diagnostics
    assert ref.asked == [("r0", 300)]
    assert d["n_over_cap_events"] == 7
    assert d["over_cap_events"] == [f"t0:r{i}" for i in range(1, 1 + SHOWN_EVENTS)]
    assert SHOWN_EVENTS < 7
