"""Unit tests for ChatRAGDataset -- the nvidia/ChatRAG-Bench (TopiOCQA) adapter.

ChatRAG-Bench ships one row PER TURN, each carrying the FULL cumulative gold
``messages`` history plus that turn's retrieved ``ctxs``. ``corpus()`` groups
these turn-rows into one row per CONVERSATION so SplitSpec partitions
conversation-granularly; ``items_at`` then renders each turn's FRESH content
only (the driver stitches history with the model's own replies at serving
time, same contract as CoQA). These tests pin: one DatasetItem per
conversation, turn 0 carrying the ChatQA instruction + first context, later
turns carrying ONLY their own fresh context + question (no instruction, no
earlier contexts), ground truth as the raw answer strings, and the raw-row
grouping boundary detection used by ``corpus()``."""
from __future__ import annotations

import json

import pytest

from hls.core.faults import DatasetError
from hls.spec.datasets import ChatRAGDataset, Split


def _conv_rows():
    # two conversations, 2 and 1 turns, in the adapter's CORPUS shape
    return [
        {"source_index": 0, "questions": ["q1", "q2"], "ctxs": ["ctx-a", "ctx-b"],
         "answers": ["a1", "a2"]},
        {"source_index": 1, "questions": ["solo"], "ctxs": ["ctx-c"], "answers": ["a3"]},
    ]


def _write(tmp_path, rows):
    d = tmp_path / "dev"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows))
    return tmp_path


def _ds():
    return ChatRAGDataset(path="nvidia/ChatRAG-Bench", subset="topiocqa",
                          source_split="dev", revision="22ece8bb870ddcf3f7aacfd5b6b0446d112a1e92")


def test_one_item_per_conversation(tmp_path):
    items = list(_ds().items_at(_write(tmp_path, _conv_rows()), Split.DEV))
    assert [len(i.turns) for i in items] == [2, 1]


def test_turn0_carries_instruction_and_ctx(tmp_path):
    it = next(iter(_ds().items_at(_write(tmp_path, _conv_rows()), Split.DEV)))
    t0 = it.turns[0].first_of("text").text
    assert "This is a chat between" in t0 and "ctx-a" in t0 and "q1" in t0


def test_later_turns_fresh_content_only(tmp_path):
    it = next(iter(_ds().items_at(_write(tmp_path, _conv_rows()), Split.DEV)))
    t1 = it.turns[1].first_of("text").text
    assert "ctx-b" in t1 and "q2" in t1
    assert "ctx-a" not in t1 and "This is a chat between" not in t1  # driver stitches history


def test_ground_truth_is_answer_strings(tmp_path):
    it = next(iter(_ds().items_at(_write(tmp_path, _conv_rows()), Split.DEV)))
    assert it.ground_truth_turns == ("a1", "a2")


def test_group_rows_detects_boundaries():
    # RAW ChatRAG turn-rows -> conversations: new conversation when a row's
    # messages contain exactly one user message
    raw = [
        {"messages": [{"role": "user", "content": "q1"}],
         "ctxs": [{"title": "", "text": "c1"}], "answers": ["a1"]},
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                      {"role": "user", "content": "q2"}],
         "ctxs": [{"title": "", "text": "c2"}], "answers": ["a2"]},
        {"messages": [{"role": "user", "content": "z1"}],
         "ctxs": [{"title": "", "text": "c3"}], "answers": ["z-a"]},
    ]
    convs = ChatRAGDataset._group_rows(raw, n_ctx=5)
    assert len(convs) == 2
    assert convs[0]["questions"] == ["q1", "q2"] and convs[0]["ctxs"] == ["c1", "c2"]
    assert convs[1]["answers"] == ["z-a"]


def _turn(n: int) -> dict:
    """A raw ChatRAG row for the ``n``-th turn (1-based) of a conversation: the full
    gold history up to and including that turn's question."""
    msgs: list[dict] = []
    for k in range(1, n + 1):
        if k > 1:
            msgs.append({"role": "assistant", "content": f"a{k - 1}"})
        msgs.append({"role": "user", "content": f"q{k}"})
    return {"messages": msgs, "ctxs": [{"title": "", "text": f"c{n}"}],
            "answers": [f"a{n}"]}


def test_group_rows_refuses_rows_out_of_turn_order():
    """Boundaries are inferred from each row's turn number, which only holds if rows
    arrive grouped by conversation in turn order. A reordered or partially filtered
    corpus would otherwise stitch a turn onto the wrong conversation and grade it
    against another conversation's gold answers, so it is refused instead."""
    assert len(ChatRAGDataset._group_rows([_turn(1), _turn(2), _turn(3)], n_ctx=5)) == 1

    # A turn dropped out of the middle.
    with pytest.raises(DatasetError, match="grouped by conversation in turn order"):
        ChatRAGDataset._group_rows([_turn(1), _turn(3)], n_ctx=5)

    # A trailing turn of one conversation arriving after the next has opened: the
    # silent-mis-grade case, since turn 3 would otherwise land on conversation two.
    with pytest.raises(DatasetError, match="grouped by conversation in turn order"):
        ChatRAGDataset._group_rows([_turn(1), _turn(2), _turn(1), _turn(3)], n_ctx=5)
