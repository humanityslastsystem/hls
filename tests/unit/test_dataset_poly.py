import json

from hls.core.poly import dump_spec
from hls.prompt import JinjaPromptTemplate
from hls.spec.datasets import HFDataset, Split


def _ds():
    return HFDataset(path="cais/mmlu", subset="anatomy", revision="main", source_split="test",
                     prompt=JinjaPromptTemplate(template="Q: {{question}}"),
                     target=JinjaPromptTemplate(template="{{answer}}"))


def test_prompt_template_renders_and_dumps():
    t = JinjaPromptTemplate(template="Q: {{question}}")
    assert t.render(question="x") == "Q: x"
    assert dump_spec(t)["kind"] == "JinjaPromptTemplate"


def test_hfdataset_dumps_with_kind():
    rec = dump_spec(_ds())
    assert rec["kind"] == "HFDataset" and rec["path"] == "cais/mmlu"


def test_items_render_mounted_rows(tmp_path):
    # items_at() reads the mounted partition and renders; no split_map / load_dataset.
    rows = [{"source_index": 0, "question": "q0", "answer": "A"},
            {"source_index": 1, "question": "q1", "answer": "B"}]
    (tmp_path / "dev").mkdir()
    (tmp_path / "dev" / "rows.json").write_text(json.dumps(rows))
    items = list(_ds().items_at(tmp_path, Split.DEV))
    assert len(items) == 2
    assert items[0].turns[0].first_of("text").text == "Q: q0"
    assert items[0].ground_truth_turns == ("A",)
