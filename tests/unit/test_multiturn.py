"""Unit tests for the multi-turn pieces: multi-turn dispatch (history stitching,
now in Driver._session, derived from each item's turn count), the
ContainsMatch metric, and InlineMultiTurnDataset. No Docker/GPU."""
from __future__ import annotations

import asyncio
import json

import pytest
from hls.core.faults import ArtifactTransportError
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.core.dataitem import DatasetItem
from hls.spec.datasets import HFDataset, Split
from tests.support.inline_dataset import InlineMultiTurnDataset
from hls.spec.drivers import Driver
from hls.verify import ContainsMatchMetric, EngineAccessor, MetricVerifier
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload)
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC


def _workload() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=ShortOutputCap(cap_tokens=8),
                    slo=NoSLO())


def _drive_one_session(item, client, on_event):
    """Drive a single multi-turn item through the Driver's two-level loop,
    exercising the same history-stitching path the production run uses.
    tenant_id 't0', ShortOutputCap -> cap=8, t0=0.0."""
    eng = Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC)
    wl = _workload()
    return eng._session(wl, 0, 0.0, item, "t0", 8, client, on_event)


# --- InlineMultiTurnDataset ----------------------------------------------

def test_inline_dataset_yields_multiturn_items():
    ds = InlineMultiTurnDataset(name="mt", conversations=[
        ("conv0", [("remember 7, say OK", "OK"), ("times 6?", "42")]),
        ("conv1", [("name is Ada, say OK", "OK"), ("my name?", "Ada")]),
    ])
    items = list(ds.items_at(None, Split.DEV))
    assert len(items) == 2
    it = items[0]
    assert it.request_id == "conv0"
    assert len(it.turns) == 2
    assert it.turns[0].first_of("text").text == "remember 7, say OK"
    assert it.ground_truth_turns == ("OK", "42")


# --- HFDataset multi-turn mode (CoQA-shaped, no network) -----------------


def test_hf_multiturn_unmounted_split_fails_loud(tmp_path):
    # Split policy is not dataset config (no split_map): items_at() reads the
    # mounted partition under mount_dir/<split>/rows.json. A split the provisioner
    # did not attach for this role (e.g. TEST withheld from the agent) fails loud.
    ds = HFDataset(
        path="stanfordnlp/coqa", source_split="validation", context_col="story",
        questions_col="questions", answers_col="answers")
    with pytest.raises(FileNotFoundError, match="(?i)test split not mounted"):
        list(ds.items_at(tmp_path, Split.TEST))

def test_hf_multiturn_builds_turns_from_coqa_shape(tmp_path):
    # Two fake CoQA rows: a context story, a list of questions, answers dict.
    fake_rows = [
        {"source_index": 0, "story": "Cotton was a white kitten in a barn.",
         "questions": ["What color was Cotton?", "Where did she live?",
                       "Did she live alone?", "extra-Q-beyond-cap"],
         "answers": {"input_text": ["white", "in a barn", "no", "ignored"]}},
        {"source_index": 1, "story": "Ada built an engine.",
         "questions": ["Who built it?", "What did she build?"],
         "answers": {"input_text": ["Ada", "an engine"]}},
    ]
    # Mount the seeded partition (the two rows) as the dev split; items_at() reads
    # and renders these raw rows -- no _load_hf_split, no runtime download.
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "rows.json").write_text(json.dumps(fake_rows))

    ds = HFDataset(
        path="stanfordnlp/coqa", source_split="validation", context_col="story",
        questions_col="questions", answers_col="answers",
        answers_text_key="input_text", max_turns=3)
    items = list(ds.items_at(tmp_path, Split.DEV))
    assert len(items) == 2

    a = items[0]
    assert len(a.turns) == 3                      # capped at max_turns=3
    # turn 0 prepends the passage; later turns are just the question text
    assert "Cotton was a white kitten" in a.turns[0].first_of("text").text
    assert "What color was Cotton?" in a.turns[0].first_of("text").text
    assert a.turns[1].first_of("text").text == "Where did she live?"
    assert a.ground_truth_turns == ("white", "in a barn", "no")

    b = items[1]
    assert len(b.turns) == 2                       # fewer questions than the cap
    assert b.ground_truth_turns == ("Ada", "an engine")


# --- Multi-turn dispatch (history stitching, per-item turn count) ---------

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


def test_multiturn_dispatch_stitches_history_and_records_per_turn():
    item = DatasetItem(
        request_id="conv0",
        turns=(
            Envelope(request_id="conv0.0", contents=[TextPayload(text="u0")]),
            Envelope(request_id="conv0.1", contents=[TextPayload(text="u1")]),
            Envelope(request_id="conv0.2", contents=[TextPayload(text="u2")]),
        ),
        ground_truth_turns=("g0", "g1", "g2"))

    client = _EchoClient()
    events = []

    def on_event(ev):       # the Driver invokes the callback synchronously
        events.append(ev)

    # The Driver's two-level loop runs the per-turn history-stitching path
    # (derived from the item's own turn count, not a separate dispatch policy).
    asyncio.run(_drive_one_session(item, client, on_event))

    # One event per turn, correctly keyed.
    assert len(events) == 3
    for i, ev in enumerate(events):
        assert ev.session_id == "conv0"
        assert ev.turn_index == i
        assert ev.request_id == f"conv0.{i}"
        assert ev.ground_truth == ("g0", "g1", "g2")[i]

    # History stitching: the request for turn i carries the full conversation so
    # far in metadata['messages'] (prior user+assistant turns + current user).
    msgs_turn2 = client.seen[2].metadata["messages"]
    assert [m["role"] for m in msgs_turn2] == [
        "user", "assistant", "user", "assistant", "user"]
    assert msgs_turn2[0]["content"] == "u0"
    assert msgs_turn2[1]["content"] == "reply-0"   # prior assistant reply stitched in
    assert msgs_turn2[2]["content"] == "u1"
    assert msgs_turn2[3]["content"] == "reply-1"
    assert msgs_turn2[4]["content"] == "u2"
    # turn 0 sees only its own user message
    assert [m["role"] for m in client.seen[0].metadata["messages"]] == ["user"]


def test_openai_protocol_encodes_stitched_messages():
    # The wire protocol carries the stitched conversation so multi-turn works over
    # OpenAI chat/completions; it falls back to the single user turn otherwise.
    from hls.spec.wire import OpenAIChatProtocol
    proto = OpenAIChatProtocol()
    env = Envelope(request_id="c.1", contents=[TextPayload(text="latest")],
                   metadata={"messages": [
                       {"role": "user", "content": "u0"},
                       {"role": "assistant", "content": "a0"},
                       {"role": "user", "content": "u1"}]})
    body = proto.encode_request(env)
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][-1]["content"] == "u1"

    env2 = Envelope(request_id="x", contents=[TextPayload(text="hi")], metadata={})
    assert proto.encode_request(env2)["messages"] == [{"role": "user", "content": "hi"}]


# --- ContainsMatchMetric via MetricVerifier (case-insensitive substring) ------

def _ev(rid, text, gt):
    """A TraceEvent whose artifact output is *text* and ground truth is *gt*.
    text=None models an errored event (no response)."""
    inp = Envelope(request_id=rid, contents=[TextPayload(text="Q?")])
    resp = (ArtifactResponse(output=Envelope(request_id=rid,
                                             contents=[TextPayload(text=text)]))
            if text is not None else None)
    return TraceEvent(request_id=rid, input=inp, response=resp, ground_truth=gt,
                      t_send=0.0, t_recv=1.0 if text is not None else None)


def test_contains_match_case_insensitive_substring():
    events = [
        _ev("present", "The answer is 42.", "42"),    # present (case-insensitive)
        _ev("caseins", "Your name is ADA", "Ada"),    # present, mixed case
        _ev("absent", "I don't know", "42"),          # absent
        _ev("empty", None, "42"),                     # no response -> fail
    ]
    acc = EngineAccessor(artifact_events=events, cap_to_provider={})
    verdict = asyncio.run(MetricVerifier(metric=ContainsMatchMetric(), min_value=1.0).verify(acc))
    assert verdict.per_event["present"] is True
    assert verdict.per_event["caseins"] is True
    assert verdict.per_event["absent"] is False
    assert verdict.per_event["empty"] is False


def test_contains_match_passrate_gate():
    # One hit, one miss -> value 0.5; min_value=0.5 -> the run passes.
    events = [_ev("r0", "contains foo here", "foo"), _ev("r1", "no match", "foo")]
    acc = EngineAccessor(artifact_events=events, cap_to_provider={})
    verdict = asyncio.run(MetricVerifier(metric=ContainsMatchMetric(), min_value=0.5).verify(acc))
    assert verdict.passed is True            # 0.5 >= 0.5
    assert verdict.diagnostics["value"] == 0.5


class _FailTurn1Client:
    """Replies normally except on turn_index==1, which raises an ArtifactFault."""
    async def submit(self, env: Envelope, *, attempt=None) -> ArtifactResponse:
        turn = env.metadata.get("turn_index")
        if turn == 1:
            raise ArtifactTransportError("artifact returned 500")
        return ArtifactResponse(output=Envelope(
            request_id=env.request_id, contents=[TextPayload(text=f"reply-{turn}")]))


def test_multiturn_failed_turn_records_failed_event_and_continues():
    item = DatasetItem(
        request_id="conv0",
        turns=(
            Envelope(request_id="conv0.0", contents=[TextPayload(text="u0")]),
            Envelope(request_id="conv0.1", contents=[TextPayload(text="u1")]),
            Envelope(request_id="conv0.2", contents=[TextPayload(text="u2")]),
        ),
        ground_truth_turns=("g0", "g1", "g2"))
    events = []
    def on_event(ev): events.append(ev)
    asyncio.run(_drive_one_session(item, _FailTurn1Client(), on_event))

    assert len(events) == 3                       # all three turns recorded
    assert events[1].response is None             # turn 1 failed
    assert events[1].error is not None and "transport" in events[1].error
    assert events[1].t_recv is None
    # The session CONTINUED: turn 2's stitched history carries an empty assistant
    # reply for the failed turn 1, not a crash.
    msgs_turn2 = [m["content"] for m in events[2].input.metadata["messages"]]
    assert msgs_turn2 == ["u0", "reply-0", "u1", "", "u2"]
