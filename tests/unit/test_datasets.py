import json

import pytest
from pydantic import ValidationError

from hls.prompt import JinjaPromptTemplate
from hls.spec.datasets import HFDataset, Split


def test_prompt_renders_over_columns():
    p = JinjaPromptTemplate(template="Q: {{ question }} / {{ 'ABCD'[answer] }}")
    assert p.render(question="x", answer=1) == "Q: x / B"


def test_hfdataset_items_render_request_and_target(tmp_path):
    rows = [
        {"source_index": 0, "question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": 1},
        {"source_index": 1, "question": "cap?", "choices": ["a", "b", "c", "d"], "answer": 0},
    ]
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "rows.json").write_text(json.dumps(rows))

    ds = HFDataset(
        path="cais/mmlu", subset="abstract_algebra", source_split="test",
        prompt=JinjaPromptTemplate(template="Question: {{ question }}\nA.{{choices[0]}} B.{{choices[1]}} "
                                   "C.{{choices[2]}} D.{{choices[3]}}\nAnswer:"),
        target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))

    items = list(ds.items_at(tmp_path, Split.DEV))
    assert len(items) == 2
    assert "2+2?" in items[0].request.first_of("text").text
    assert items[0].ground_truth == "B"
    assert items[1].ground_truth == "A"


# --- HFDataset mode-validator tests --------------------------------------

def test_hfdataset_both_modes_raises():
    """Setting both prompt+target AND context_col is rejected at construction."""
    with pytest.raises(ValidationError, match="(?i)both"):
        HFDataset(
            path="x", source_split="test",
            prompt=JinjaPromptTemplate(template="{{ q }}"),
            target=JinjaPromptTemplate(template="{{ a }}"),
            context_col="c")


def test_hfdataset_neither_mode_raises():
    """Setting neither mode (no prompt/target, no context_col) is rejected."""
    with pytest.raises(ValidationError):
        HFDataset(path="x", source_split="test")


def test_items_at_reads_arbitrary_mount_dir(tmp_path):
    import json
    from hls.prompt import JinjaPromptTemplate
    from hls.spec.datasets import HFDataset, Split
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "rows.json").write_text(json.dumps([{"source_index": 0, "question": "q1", "answer": "a1"}]))
    ds = HFDataset(path="openai/gsm8k", subset="main", source_split="test",
                   prompt=JinjaPromptTemplate(template="{{ question }}"),
                   target=JinjaPromptTemplate(template="{{ answer }}"))
    items = list(ds.items_at(tmp_path, Split.TEST))
    assert len(items) == 1
    assert items[0].turns[0].first_of("text").text == "q1"
    assert items[0].ground_truth_turns[0] == "a1"
