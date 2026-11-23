"""Unit test for the driver's adapter-prerendered `metadata['messages']`
passthrough (Driver._session): when a turn Envelope carries a non-empty
`metadata['messages']` list (MRCR's scripted history, BFCL's system
template), the request uses it verbatim instead of the driver's own
stitched cumulative history. Reuses the fake-artifact + drive-one-session
pattern from tests/unit/test_multiturn.py."""
from __future__ import annotations

import asyncio

from hls.core.dataitem import DatasetItem
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse
from hls.spec.drivers import Driver
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC


def _workload() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=ShortOutputCap(cap_tokens=8),
                    slo=NoSLO())


def _drive_one_session(item, client, on_event):
    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    wl = _workload()
    return eng._session(wl, 0, 0.0, item, "t0", 8, client, on_event)


class _EchoClient:
    """Fake artifact: records each request it receives and replies with a
    deterministic assistant message derived from the turn index."""
    def __init__(self):
        self.seen: list[Envelope] = []

    async def submit(self, env: Envelope, *, attempt=None) -> ArtifactResponse:
        self.seen.append(env)
        turn = env.metadata.get("turn_index")
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text=f"reply-{turn}")]))


def test_prerendered_messages_used_verbatim_not_driver_stitched():
    pre = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    item = DatasetItem(
        request_id="conv0",
        turns=(Envelope(request_id="conv0.0", contents=[TextPayload(text="U")],
                        metadata={"messages": pre}),),
        ground_truth_turns=("g0",))

    client = _EchoClient()
    events = []

    def on_event(ev):
        events.append(ev)

    asyncio.run(_drive_one_session(item, client, on_event))

    assert len(events) == 1
    # The request carries the adapter's pre-rendered messages verbatim, NOT the
    # driver's own stitched [{"role": "user", "content": <envelope text>}].
    assert events[0].input.metadata["messages"] == pre
    assert client.seen[0].metadata["messages"] == pre


def test_multi_turn_prerendered_messages_beat_the_live_reply_stitching():
    """A SESSION whose every turn pre-renders its own history (MRCR's cumulative
    scripted replay) must be driven on that scripted context, turn after turn: the
    driver still accumulates its own stitched history, but each request carries the
    turn's own list, so the artifact's live replies never enter the context."""
    scripted = [{"role": "user", "content": "u0"},
                {"role": "assistant", "content": "scripted-0"},
                {"role": "user", "content": "u1"}]
    turns = tuple(
        Envelope(request_id=f"conv0.{k}", contents=[TextPayload(text=f"u{k}")],
                 metadata={"messages": scripted[:2 * k + 1]})
        for k in range(2))
    item = DatasetItem(request_id="conv0", turns=turns, ground_truth_turns=("g0", "g1"))

    client = _EchoClient()
    events = []
    asyncio.run(_drive_one_session(item, client, lambda ev: events.append(ev)))

    assert [e.metadata["messages"] for e in client.seen] == [scripted[:1], scripted[:3]]
    assert all("reply-0" not in str(e.metadata["messages"]) for e in client.seen)
    assert [ev.turn_index for ev in events] == [0, 1]


def test_empty_prerendered_messages_falls_back_to_driver_stitching():
    item = DatasetItem(
        request_id="conv0",
        turns=(Envelope(request_id="conv0.0", contents=[TextPayload(text="u0")],
                        metadata={"messages": []}),),
        ground_truth_turns=("g0",))

    client = _EchoClient()
    events = []
    asyncio.run(_drive_one_session(item, client, lambda ev: events.append(ev)))

    assert events[0].input.metadata["messages"] == [{"role": "user", "content": "u0"}]
