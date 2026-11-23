"""Unit tests for WildChatDataset (the allenai/WildChat-1M adapter): conversation
-> user-turn extraction (the corpus()-time reshaping), and the corpus-row ->
DatasetItem rendering. No network access -- rendering is exercised via `items_at`
on fixture rows, mirroring tests/unit/test_multiif_dataset.py and
tests/unit/test_mrcr.py."""
from __future__ import annotations

import json

import pytest

from hls.spec.datasets import Split, WildChatDataset

_REV = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"


def _ds(**kw) -> WildChatDataset:
    return WildChatDataset(revision=_REV, **kw)


def _conv(*pairs):
    """A raw WildChat `conversation` cell from (user, assistant) text pairs; an
    assistant of None models a conversation that ends on the user's message."""
    msgs = []
    for user, assistant in pairs:
        msgs.append({"role": "user", "content": user})
        if assistant is not None:
            msgs.append({"role": "assistant", "content": assistant})
    return msgs


def _mount(tmp_path, rows, split="test"):
    d = tmp_path / split
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows))
    return tmp_path


# --- conversation -> turns (corpus()-time reshaping) -------------------------

def test_turns_pair_user_messages_with_their_replies():
    users, replies = _ds()._turns_of(_conv(("u1", "a1"), ("u2", "a2")))
    assert users == ["u1", "u2"]
    assert replies == ["a1", "a2"]


def test_turns_are_capped_at_max_turns():
    convo = _conv(*[(f"u{i}", f"a{i}") for i in range(10)])
    users, replies = _ds(max_turns=3)._turns_of(convo)
    assert users == ["u0", "u1", "u2"]
    assert replies == ["a0", "a1", "a2"]


def test_trailing_user_message_keeps_its_turn_with_an_empty_reply():
    """A conversation that ends on the user still yields that turn -- the artifact
    generates the reply, and the empty scripted one is never graded."""
    users, replies = _ds()._turns_of(_conv(("u1", "a1"), ("u2", None)))
    assert users == ["u1", "u2"]
    assert replies == ["a1", ""]


def test_blank_user_messages_are_dropped():
    users, _ = _ds()._turns_of(_conv(("  ", "a1"), ("u2", "a2")))
    assert users == ["u2"]


# --- rendering ---------------------------------------------------------------

def test_items_render_one_envelope_per_user_turn(tmp_path):
    rows = [{"conversation_hash": "c1", "user_turns": ["u1", "u2", "u3"],
             "assistant_turns": ["a1", "a2", "a3"]}]
    items = list(_ds().items_at(_mount(tmp_path, rows), Split.TEST))

    assert len(items) == 1
    item = items[0]
    assert [e.first_of("text").text for e in item.turns] == ["u1", "u2", "u3"]
    assert item.ground_truth_turns == ("a1", "a2", "a3")
    assert item.metadata["n_turns"] == 3


def test_turns_carry_no_prerendered_messages(tmp_path):
    """The artifact's OWN replies are the history here: leaving
    metadata['messages'] unset is what makes the driver stitch them in (contrast
    MRCRDataset's scripted replay, which pre-renders every turn)."""
    rows = [{"conversation_hash": "c2", "user_turns": ["u1", "u2"], "assistant_turns": ["a1", "a2"]}]
    items = list(_ds().items_at(_mount(tmp_path, rows), Split.TEST))
    assert all("messages" not in e.metadata for e in items[0].turns)


def test_rendering_is_deterministic(tmp_path):
    rows = [{"conversation_hash": "c3", "user_turns": ["u1", "u2"], "assistant_turns": ["a1", "a2"]},
            {"conversation_hash": "c4", "user_turns": ["q"], "assistant_turns": [""]}]
    mount = _mount(tmp_path, rows)
    first = [(i.request_id, [e.request_id for e in i.turns]) for i in _ds().items_at(mount, Split.TEST)]
    second = [(i.request_id, [e.request_id for e in i.turns]) for i in _ds().items_at(mount, Split.TEST)]
    assert first == second
    assert first[0][1] == ["c3.0", "c3.1"]


def test_every_turn_has_a_ground_truth(tmp_path):
    """The Driver refuses an item with fewer ground truths than turns, so a corpus
    row whose reply list is short must not produce one."""
    rows = [{"conversation_hash": "c5", "user_turns": ["u1", "u2"], "assistant_turns": ["a1", ""]}]
    items = list(_ds().items_at(_mount(tmp_path, rows), Split.TEST))
    assert len(items[0].ground_truth_turns) == len(items[0].turns)


# --- corpus() pool shaping ---------------------------------------------------
# The stream is faked and the tokenizer stubbed through the module's own cache, so
# the filter chain (language / toxic / dedupe / token cap / subsample stop) is
# exercised without a 3.4 GB download.

class _WordTokenizer:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


@pytest.fixture
def stub_stream(monkeypatch):
    import datasets

    from hls.spec.datasets import _TOKENIZER_CACHE
    monkeypatch.setitem(_TOKENIZER_CACHE, ("Qwen/Qwen2.5-32B", "main"), _WordTokenizer())

    def _install(rows):
        monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: rows)
    return _install


def _raw(user, *, language="English", toxic=False, turns=1):
    convo = _conv(*[(user if i == 0 else f"{user} more {i}", f"a{i}") for i in range(turns)])
    return {"conversation_hash": f"hash:{user}", "language": language, "toxic": toxic,
            "conversation": convo}


def test_corpus_keeps_only_the_configured_language(stub_stream):
    stub_stream([_raw("keep me"), _raw("skip me", language="Spanish")])
    rows = _ds(subsample_size=10).corpus().to_list()
    assert [r["user_turns"][0] for r in rows] == ["keep me"]


def test_corpus_drops_toxic_conversations(stub_stream):
    stub_stream([_raw("toxic one", toxic=True), _raw("clean one")])
    rows = _ds(subsample_size=10).corpus().to_list()
    assert [r["user_turns"][0] for r in rows] == ["clean one"]


def test_corpus_dedupes_on_the_opening_user_message(stub_stream):
    """WildChat logs the same opening prompt many times; each repeat would be a
    byte-identical turn-0 request, i.e. a free response-cache hit."""
    stub_stream([_raw("same opening"), _raw("same opening"), _raw("other")])
    rows = _ds(subsample_size=10).corpus().to_list()
    assert [r["user_turns"][0] for r in rows] == ["same opening", "other"]


def test_corpus_drops_conversations_over_the_input_budget(stub_stream):
    stub_stream([_raw("a b c d e"), _raw("short one")])
    rows = _ds(subsample_size=10, max_input_tokens=3).corpus().to_list()
    assert [r["user_turns"][0] for r in rows] == ["short one"]


def test_corpus_stops_at_the_subsample_size(stub_stream):
    stub_stream([_raw(f"prompt {i}") for i in range(50)])
    assert len(_ds(subsample_size=7).corpus()) == 7


def test_corpus_id_carries_the_filters_that_shape_the_pool():
    cid = _ds(subsample_size=1188, max_input_tokens=3072).corpus_id()
    for part in ("lang=English", "turns=6", "n=1188", "cap=3072", "toxic=drop"):
        assert part in cid


def test_unmounted_split_fails_loud(tmp_path):
    with pytest.raises(FileNotFoundError, match="(?i)test split not mounted"):
        list(_ds().items_at(tmp_path, Split.TEST))
