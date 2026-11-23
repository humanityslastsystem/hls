"""Unit tests for MRCRRatio (openai/mrcr's official grader) and MRCRDataset
(the openai/mrcr adapter): the prefix-gated SequenceMatcher score, and the
corpus-row -> DatasetItem rendering (pre-rendered `messages` passthrough for
the Driver, last-user-turn envelope text, {"answer", "prepend"} ground
truth). No network/tokenizer access -- item rendering is exercised via
`items_at` on fixture rows, mirroring tests/unit/test_chatrag_dataset.py and
tests/unit/test_multiif_dataset.py."""
from __future__ import annotations

import json

import pytest
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics.mrcr import MRCRRatio


def _event(response_text: str, gt: dict) -> TraceEvent:
    inp = Envelope(request_id="r0", contents=[TextPayload(text="prompt")])
    out = Envelope(request_id="r0", contents=[TextPayload(text=response_text)])
    return TraceEvent(request_id="r0", input=inp, response=ArtifactResponse(output=out),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


# --- MRCRRatio ---------------------------------------------------------------

def test_missing_prepend_scores_zero():
    ev = _event("wrong start poem", {"answer": "RS: poem", "prepend": "RS: "})
    assert MRCRRatio().score_one(ev).score == 0.0


def test_exact_match_scores_one():
    ev = _event("RS: the poem body", {"answer": "RS: the poem body", "prepend": "RS: "})
    assert MRCRRatio().score_one(ev).score == 1.0


def test_partial_match_is_ratio_between():
    ev = _event("RS: the poem", {"answer": "RS: the poem body", "prepend": "RS: "})
    score = MRCRRatio().score_one(ev).score
    assert 0.0 < score < 1.0


# --- MRCRDataset ---------------------------------------------------------------
# token filter + rendering, with the token counter never invoked in unit tests
# (items_at reads already-filtered rows.json fixture rows; no tokenizer download).

def test_mrcr_row_filter_and_render(tmp_path):
    from hls.spec.datasets import MRCRDataset, Split
    rows = [{"source_index": 0, "messages": [{"role": "user", "content": "write a poem"},
                          {"role": "assistant", "content": "poem 1"},
                          {"role": "user", "content": "give me the 1st poem"}],
             "answer": "RS: poem 1", "prepend": "RS: "}]
    d = tmp_path / "test"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows))
    ds = MRCRDataset(path="openai/mrcr", revision="f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
    items = list(ds.items_at(tmp_path, Split.TEST))
    assert len(items) == 1 and len(items[0].turns) == 1
    env = items[0].turns[0]
    assert env.metadata["messages"] == rows[0]["messages"]      # passthrough payload
    assert env.first_of("text").text == "give me the 1st poem"  # last user turn as text
    assert items[0].ground_truth_turns[0] == {"answer": "RS: poem 1", "prepend": "RS: "}


# --- MRCRDataset cumulative scripted replay ------------------------------------

_REPLAY_ROW = {
    "source_index": 7,
    "messages": [{"role": "user", "content": "write poem 1"},
                 {"role": "assistant", "content": "poem 1"},
                 {"role": "user", "content": "write poem 2"},
                 {"role": "assistant", "content": "poem 2"},
                 {"role": "user", "content": "give me the 1st poem"}],
    "answer": "RS: poem 1", "prepend": "RS: "}


def _replay_items(tmp_path, rows=None):
    from hls.spec.datasets import MRCRDataset, Split
    d = tmp_path / "test"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows if rows is not None else [_REPLAY_ROW]))
    ds = MRCRDataset(path="openai/mrcr", revision="f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d",
                     replay=True)
    return list(ds.items_at(tmp_path, Split.TEST))


def test_replay_makes_one_turn_per_scripted_user_message(tmp_path):
    items = _replay_items(tmp_path)
    assert len(items) == 1
    item = items[0]
    assert len(item.turns) == 3 == item.metadata["n_turns"]
    assert [e.first_of("text").text for e in item.turns] == [
        "write poem 1", "write poem 2", "give me the 1st poem"]


def test_replay_turns_carry_cumulative_scripted_history(tmp_path):
    """Turn k's pre-rendered messages are the corpus's own list THROUGH user turn k
    -- scripted, never the artifact's live replies, so the needle structure the
    benchmark builds survives and every turn re-sends a longer prefix."""
    msgs = _REPLAY_ROW["messages"]
    turns = _replay_items(tmp_path)[0].turns
    assert [e.metadata["messages"] for e in turns] == [msgs[:1], msgs[:3], msgs[:5]]
    assert all(m["role"] == "user" for m in
               (e.metadata["messages"][-1] for e in turns))


def test_replay_ground_truth_is_scripted_then_the_gold_answer(tmp_path):
    """Intermediate turns are graded against the reply the corpus scripts for them;
    the final turn -- the one with no scripted reply -- carries MRCR's own
    answer/prepend pair."""
    gts = _replay_items(tmp_path)[0].ground_truth_turns
    assert gts[:2] == ({"answer": "poem 1", "prepend": ""},
                       {"answer": "poem 2", "prepend": ""})
    assert gts[-1] == {"answer": "RS: poem 1", "prepend": "RS: "}


def test_replay_is_off_by_default(tmp_path):
    """Without the flag the adapter stays single-turn (one graded request per
    conversation) -- the replay lever must be opt-in per tenant."""
    from hls.spec.datasets import MRCRDataset, Split
    d = tmp_path / "test"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps([_REPLAY_ROW]))
    ds = MRCRDataset(path="openai/mrcr", revision="f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
    items = list(ds.items_at(tmp_path, Split.TEST))
    assert len(items[0].turns) == 1
    assert items[0].turns[0].metadata["messages"] == _REPLAY_ROW["messages"]


def test_replay_leaves_corpus_identity_alone():
    """`replay` changes RENDERING only, so it must not enter corpus_id() -- a flip
    would otherwise re-download and re-partition an identical corpus."""
    from hls.spec.datasets import MRCRDataset
    rev = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"
    plain = MRCRDataset(path="openai/mrcr", revision=rev)
    assert MRCRDataset(path="openai/mrcr", revision=rev, replay=True).corpus_id() == \
        plain.corpus_id()


def test_corpus_id_includes_cap_and_tokenizer():
    from hls.spec.datasets import MRCRDataset
    ds = MRCRDataset(path="openai/mrcr", revision="f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
    cid = ds.corpus_id()
    assert "cap=8176" in cid and "tok=Qwen/Qwen2.5-32B" in cid


def test_unmounted_split_fails_loud(tmp_path):
    from hls.spec.datasets import MRCRDataset, Split
    ds = MRCRDataset(path="openai/mrcr", revision="f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
    with pytest.raises(FileNotFoundError, match="(?i)test split not mounted"):
        list(ds.items_at(tmp_path, Split.TEST))


# --- MRCRDataset.max_turns (cap replay session depth) -------------------------

_REV = "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d"

_FOUR_TURN_ROW = {
    "source_index": 11,
    "messages": [
        {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
        {"role": "user", "content": "u4"},
    ],
    "answer": "RS: x", "prepend": "RS: ",
}


def _replay_items_with_max_turns(tmp_path, max_turns=None, rows=None):
    from hls.spec.datasets import MRCRDataset, Split
    d = tmp_path / "test"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows if rows is not None else [_FOUR_TURN_ROW]))
    ds = MRCRDataset(path="openai/mrcr", revision=_REV, replay=True, max_turns=max_turns)
    return list(ds.items_at(tmp_path, Split.TEST))


def test_mrcr_replay_caps_turns_when_max_turns_set(tmp_path):
    uncapped = _replay_items_with_max_turns(tmp_path)
    assert len(uncapped[0].turns) == 4  # all four user turns

    tmp_path2 = tmp_path / "capped"
    tmp_path2.mkdir()
    capped = _replay_items_with_max_turns(tmp_path2, max_turns=2)
    assert len(capped[0].turns) == 2                         # truncated to first 2
    assert len(capped[0].ground_truth_turns) == 2            # gts truncated in lockstep
    assert capped[0].metadata["n_turns"] == 2
    # First two turns are byte-identical to the uncapped render (prefix preserved).
    assert [t.metadata["messages"] for t in capped[0].turns] == \
           [t.metadata["messages"] for t in uncapped[0].turns[:2]]


def test_mrcr_replay_max_turns_none_is_unbounded(tmp_path):
    from hls.spec.datasets import MRCRDataset
    ds = MRCRDataset(path="openai/mrcr", revision=_REV, replay=True)
    assert ds.max_turns is None
    items = _replay_items_with_max_turns(tmp_path)
    assert len(items[0].turns) == 4  # all four user turns returned


def test_mrcr_replay_max_turns_not_in_corpus_id():
    """`max_turns` is rendering-only and must NOT enter corpus_id() -- same as `replay`."""
    from hls.spec.datasets import MRCRDataset
    plain = MRCRDataset(path="openai/mrcr", revision=_REV)
    capped = MRCRDataset(path="openai/mrcr", revision=_REV, replay=True, max_turns=2)
    assert capped.corpus_id() == plain.corpus_id()
