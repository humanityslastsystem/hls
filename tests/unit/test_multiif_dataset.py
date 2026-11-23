"""Unit tests for MultiIFDataset -- the facebook/Multi-IF adapter.

Multi-IF is a <=3-turn instruction-following benchmark whose per-turn ground truth is
an IFEval spec ({instruction_id_list, kwargs}), not a text answer. These tests pin the
row->DatasetItem rendering (per-turn prompts + cumulative instruction specs), the
short-conversation truncation, the nested-JSON kwargs parsing, and that the emitted
ground-truth dict is exactly what IFEvalConstraint consumes."""
from __future__ import annotations

import json
from pathlib import Path

from hls.spec.datasets import MultiIFDataset, Split
from hls.verify.metrics.ifeval import IFEvalConstraint


def _prompt(content: str) -> str:
    # Multi-IF stores each turn prompt as a JSON string {"role","content"}.
    return json.dumps({"role": "user", "content": content})


def _row(turns: list[tuple[str, list[str], list[dict]]]) -> dict:
    """Build one Multi-IF row from (content, instruction_ids, kwargs) per turn.

    Mirrors the on-disk column shapes: prompt/id-list/kwargs are all JSON strings,
    and each kwargs entry is itself a JSON string (a list-of-JSON-strings)."""
    row = {"language": "English", "key": "k:0:en", "turn_index": 0}
    for i in range(3):
        n = i + 1
        if i < len(turns):
            content, ids, kwargs = turns[i]
            row[f"turn_{n}_prompt"] = _prompt(content)
            row[f"turn_{n}_instruction_id_list"] = json.dumps(ids)
            row[f"turn_{n}_kwargs"] = json.dumps([json.dumps(k) for k in kwargs])
        else:
            row[f"turn_{n}_prompt"] = None
            row[f"turn_{n}_instruction_id_list"] = None
            row[f"turn_{n}_kwargs"] = None
    return row


def _mount(tmp_path: Path, rows: list[dict], split: str = "test") -> Path:
    d = tmp_path / split
    d.mkdir(parents=True)
    (d / "rows.json").write_text(json.dumps(rows))
    return tmp_path


THREE_TURN = _row([
    ("Ask a question about the sentence.", ["change_case:english_lowercase"], [{}]),
    ("End with the exact phrase: the end.",
     ["change_case:english_lowercase", "startend:end_checker"],
     [{}, {"end_phrase": "the end"}]),
    ("Now write at least 100 words.",
     ["change_case:english_lowercase", "startend:end_checker", "length_constraints:number_words"],
     [{}, {"end_phrase": "the end"}, {"relation": "at least", "num_words": 100}]),
])


def test_construction_does_not_require_prompt_or_conversational_fields():
    # Multi-IF is a third mode; HFDataset's prompt/target-vs-conversational validator
    # must not fire for it.
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    assert ds.max_turns == 3
    assert ds.language == "English"


def test_three_turn_row_renders_all_turns(tmp_path):
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    mount = _mount(tmp_path, [THREE_TURN])
    items = list(ds.items_at(mount, Split.TEST))
    assert len(items) == 1
    it = items[0]
    assert len(it.turns) == 3
    assert len(it.ground_truth_turns) == 3
    # per-turn prompt text is the message content, not the raw JSON envelope
    assert it.turns[0].contents[0].text == "Ask a question about the sentence."
    assert it.turns[2].contents[0].text == "Now write at least 100 words."


def test_ground_truth_is_ifeval_spec_with_parsed_kwargs(tmp_path):
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    mount = _mount(tmp_path, [THREE_TURN])
    gt = list(ds.items_at(mount, Split.TEST))[0].ground_truth_turns
    # turn 3 carries the cumulative instruction list...
    assert gt[2]["instruction_id_list"] == [
        "change_case:english_lowercase", "startend:end_checker",
        "length_constraints:number_words"]
    # ...and kwargs are decoded from the nested JSON-string form into real dicts
    assert gt[2]["kwargs"][1] == {"end_phrase": "the end"}
    assert gt[2]["kwargs"][2] == {"relation": "at least", "num_words": 100}


def test_short_conversation_truncates_at_first_missing_turn(tmp_path):
    two = _row([
        ("Turn one.", ["change_case:english_lowercase"], [{}]),
        ("Turn two, end with: bye.", ["startend:end_checker"], [{"end_phrase": "bye"}]),
    ])
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    mount = _mount(tmp_path, [two])
    it = list(ds.items_at(mount, Split.TEST))[0]
    assert len(it.turns) == 2
    assert it.metadata["n_turns"] == 2


def test_max_turns_caps_rendered_turns(tmp_path):
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train", max_turns=2)
    mount = _mount(tmp_path, [THREE_TURN])
    it = list(ds.items_at(mount, Split.TEST))[0]
    assert len(it.turns) == 2


def test_stable_request_ids(tmp_path):
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    mount = _mount(tmp_path, [THREE_TURN])
    it = list(ds.items_at(mount, Split.TEST))[0]
    assert it.request_id == THREE_TURN["key"]
    assert [e.request_id for e in it.turns] == [
        f"{THREE_TURN['key']}.0", f"{THREE_TURN['key']}.1", f"{THREE_TURN['key']}.2"]


def test_size_counts_rows(tmp_path):
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    mount = _mount(tmp_path, [THREE_TURN, THREE_TURN])
    assert len(list(ds.items_at(mount, Split.TEST))) == 2


def test_corpus_id_includes_language():
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    assert "lang=English" in ds.corpus_id()


def test_emitted_ground_truth_scores_through_ifeval_metric(tmp_path):
    # The rendered ground-truth dict must be consumable verbatim by IFEvalConstraint.
    from hls.core.payloads import Envelope, TextPayload
    from hls.core.trace import ArtifactResponse, TraceEvent
    ds = MultiIFDataset(path="facebook/Multi-IF", source_split="train")
    mount = _mount(tmp_path, [THREE_TURN])
    it = list(ds.items_at(mount, Split.TEST))[0]
    spec = it.ground_truth_turns[1]  # turn 2: lowercase + end with "the end"
    good = Envelope(request_id="e", contents=[TextPayload(text="all good the end")])
    bad = Envelope(request_id="e", contents=[TextPayload(text="WRONG ending here")])
    m = IFEvalConstraint()

    def score(env):
        ev = TraceEvent(request_id="e", input=env,
                        response=ArtifactResponse(output=env), t_send=0.0, t_recv=1.0,
                        ground_truth=spec)
        return m.score([ev]).per_event["e"].passed

    assert score(good) is True
    assert score(bad) is False
