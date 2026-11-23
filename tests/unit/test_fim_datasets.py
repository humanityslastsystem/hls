"""FIM rendering: Real-FIM-Eval rows as typing sessions, SAFIM rows split at their
placeholder. No network: rows are fixtures and corpora are faked through
``datasets.load_dataset``."""
import json

import pytest

from hls.core.faults import DatasetError
from hls.core.poly import dump_spec
from hls.prompt import JinjaPromptTemplate
from hls.spec.datasets import (FIM_PSM_PROMPT, FimSessionDataset, HFDataset, SafimDataset,
                               Split, _fim_cuts)

PRE, SUF, MID = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"
#: Neither class defaults its template, so every construction below states one. The
#: Qwen2.5-Coder spelling is as good as any here: rendering never reads a vocabulary.
_TMPL = JinjaPromptTemplate(template=FIM_PSM_PROMPT)


def _mount(tmp_path, rows):
    d = tmp_path / "test"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows))
    return tmp_path


# --- cuts ---------------------------------------------------------------------

def test_cuts_are_evenly_spaced_line_boundaries_starting_at_zero():
    assert _fim_cuts(16, 8) == [0, 2, 4, 6, 8, 10, 12, 14]
    assert _fim_cuts(8, 8) == list(range(8))


def test_a_short_middle_yields_fewer_steps_never_an_empty_final_step():
    assert _fim_cuts(3, 8) == [0, 1, 2]
    assert _fim_cuts(1, 8) == [0]
    assert _fim_cuts(0, 8) == []
    assert _fim_cuts(5, 1) == [0]


# --- FimSessionDataset ---------------------------------------------------------

def _session_row(n_lines=4):
    return {"source_index": 3, "prompt": "def f():\n", "suffix": "\nprint(f())\n",
            "canonical_solution": "".join(f"    x{i} = {i}\n" for i in range(n_lines))}


def test_a_source_row_becomes_a_session_whose_step_zero_is_the_original_prompt(tmp_path):
    ds = FimSessionDataset(subset="add", revision="r", prompt=_TMPL, max_steps=2)
    row = _session_row()
    items = list(ds.items_at(_mount(tmp_path, [row]), Split.TEST))
    assert len(items) == 1
    item = items[0]
    assert item.request_id == "3" and len(item.turns) == 2
    assert item.turns[0].request_id == "3.0"
    text0 = item.turns[0].first_of("text").text
    assert text0 == f"{PRE}def f():\n{SUF}\nprint(f())\n{MID}"
    assert item.turns[0].metadata["messages"] == [{"role": "user", "content": text0}]
    assert item.ground_truth_turns[0] == row["canonical_solution"]
    assert item.metadata["n_turns"] == 2


def test_later_steps_grow_the_prefix_by_typed_lines_and_complete_the_rest(tmp_path):
    ds = FimSessionDataset(subset="add", revision="r", prompt=_TMPL, max_steps=4)
    row = _session_row(4)
    item = next(ds.items_at(_mount(tmp_path, [row]), Split.TEST))
    lines = row["canonical_solution"].splitlines(keepends=True)
    for j, cut in enumerate([0, 1, 2, 3]):
        typed, rest = "".join(lines[:cut]), "".join(lines[cut:])
        assert item.turns[j].first_of("text").text == f"{PRE}def f():\n{typed}{SUF}\nprint(f())\n{MID}"
        assert item.ground_truth_turns[j] == rest
        assert item.turns[j].request_id == f"3.{j}"


def test_a_row_with_fewer_lines_than_max_steps_yields_fewer_steps(tmp_path):
    ds = FimSessionDataset(subset="add", revision="r", prompt=_TMPL, max_steps=8)
    item = next(ds.items_at(_mount(tmp_path, [_session_row(3)]), Split.TEST))
    assert len(item.turns) == 3


def test_a_row_with_an_empty_middle_yields_no_item(tmp_path):
    row = _session_row(0)
    items = list(FimSessionDataset(subset="add", revision="r", prompt=_TMPL).items_at(
        _mount(tmp_path, [row]), Split.TEST))
    assert items == []


def test_code_renders_untouched(tmp_path):
    """Jinja must not escape or trim code: braces, angle brackets, trailing newlines."""
    row = {"source_index": 0, "prompt": "if a < b && c > d {\n", "suffix": "}\n\n",
           "canonical_solution": "  x = {{1}};\n"}
    item = next(FimSessionDataset(subset="add", revision="r", prompt=_TMPL).items_at(
        _mount(tmp_path, [row]), Split.TEST))
    assert item.turns[0].first_of("text").text == f"{PRE}if a < b && c > d {{\n{SUF}}}\n\n{MID}"
    assert item.ground_truth_turns[0] == "  x = {{1}};\n"


def test_session_knobs_are_in_the_provenance_and_out_of_the_corpus_id():
    a = FimSessionDataset(subset="add", revision="r", prompt=_TMPL, max_steps=8)
    b = FimSessionDataset(subset="add", revision="r", prompt=_TMPL, max_steps=4)
    assert dump_spec(a)["max_steps"] == 8 and dump_spec(a)["prompt"]["template"] == FIM_PSM_PROMPT
    assert dump_spec(a) != dump_spec(b)
    assert a.corpus_id() == b.corpus_id()          # same provisioned rows
    assert a.corpus_id() == ("gonglinyuan/real_fim_eval:add@r|src=test|id=source_index"
                             "|fim-cols=prompt,suffix,canonical_solution")


def test_the_pruned_columns_are_in_the_corpus_id():
    # corpus() drops every column but these three (plus the row position), so two
    # configs naming different ones must not share a provisioning cache entry.
    base = dict(subset="add", revision="r", prompt=_TMPL)
    a = FimSessionDataset(**base)
    for field in ("prefix_col", "suffix_col", "middle_col"):
        other = FimSessionDataset(**base, **{field: "elsewhere"})
        assert a.corpus_id() != other.corpus_id(), field
    # A permutation of the same three names renders one column and grades another.
    swapped = FimSessionDataset(**base, prefix_col="suffix", suffix_col="prompt")
    assert a.corpus_id() != swapped.corpus_id()


def test_corpus_keeps_only_the_code_columns_and_the_row_position(stub_corpus):
    # Real-FIM-Eval's provenance columns include a commit timestamp, which the
    # provisioner's json.dumps of the rows refuses.
    from datetime import datetime
    stub_corpus([{"repo": "o/r", "ref": "main", "path": "a.py", "lang": "python",
                  "prompt": "def f():\n", "suffix": "\n", "canonical_solution": "    pass\n",
                  "timestamp": datetime(2024, 1, 2, 3, 4, 5)}])
    rows = FimSessionDataset(subset="add", revision="r", prompt=_TMPL).corpus().to_list()
    assert set(rows[0]) == {"prompt", "suffix", "canonical_solution", "source_index"}
    json.dumps(rows)


def test_session_dataset_refuses_zero_steps():
    with pytest.raises(ValueError, match="max_steps"):
        FimSessionDataset(subset="add", revision="r", prompt=_TMPL, max_steps=0)


def test_session_dataset_still_runs_the_base_single_vs_multi_turn_validator():
    # _validate_max_steps must not shadow HFDataset._validate_mode: a FimSessionDataset
    # missing half its single-turn rendering should still be refused by it.
    with pytest.raises(ValueError, match="requires both prompt and target"):
        FimSessionDataset(subset="add", revision="r", prompt=_TMPL, target=None)


def test_neither_class_defaults_the_sentinel_template():
    # The spelling belongs to a checkpoint's vocabulary, so there is no template to
    # inherit; omitting it is a construction error, not a silent Qwen-shaped prompt.
    for cls in (FimSessionDataset, SafimDataset):
        with pytest.raises(ValueError, match="prompt"):
            cls(subset="add", revision="r")


# --- SafimDataset --------------------------------------------------------------

def _safim_row(eval_prompt="a = 1\n{{completion}}\nprint(a)\n"):
    return {"task_id": "block_completion_000007", "lang": "python", "prompt": "Complete...",
            "eval_prompt": eval_prompt, "ground_truth": "a += 1", "unit_tests": "[huge]"}


@pytest.fixture
def stub_corpus(monkeypatch):
    import datasets

    def _install(rows):
        monkeypatch.setattr(datasets, "load_dataset",
                            lambda *a, **k: datasets.Dataset.from_list(rows))
    return _install


def test_corpus_adds_prefix_and_suffix_and_drops_the_execution_columns(stub_corpus):
    stub_corpus([_safim_row()])
    rows = SafimDataset(subset="block", revision="r", prompt=_TMPL).corpus().to_list()
    assert rows[0]["prefix"] == "a = 1\n" and rows[0]["suffix"] == "\nprint(a)\n"
    assert "unit_tests" not in rows[0] and "eval_prompt" not in rows[0]
    assert rows[0]["task_id"] == "block_completion_000007"


def test_corpus_refuses_a_row_whose_placeholder_is_not_exactly_once(stub_corpus):
    for bad in ("no placeholder here", "{{completion}} twice {{completion}}"):
        stub_corpus([_safim_row(bad)])
        with pytest.raises(DatasetError, match="(?i)placeholder"):
            SafimDataset(subset="block", revision="r", prompt=_TMPL).corpus()


def test_items_render_the_fim_prompt_from_the_split_columns(tmp_path):
    row = {"task_id": "control_completion_000001", "prefix": "for i in ",
           "suffix": ":\n    pass\n", "ground_truth": "range(3)"}
    item = next(SafimDataset(subset="control", revision="r", prompt=_TMPL).items_at(
        _mount(tmp_path, [row]), Split.TEST))
    assert item.request_id == "control_completion_000001" and len(item.turns) == 1
    assert item.turns[0].first_of("text").text == f"{PRE}for i in {SUF}:\n    pass\n{MID}"
    assert item.ground_truth == "range(3)"


def test_safim_corpus_id_differs_from_a_plain_hfdataset_on_the_same_source():
    # Same id_column on both, so what is left is the SAFIM-specific suffix rather than
    # the identity component every HFDataset carries.
    plain = HFDataset(path="gonglinyuan/safim", subset="block", revision="r",
                      source_split="test", prompt=_TMPL,
                      id_column=SafimDataset.model_fields["id_column"].default,
                      target=SafimDataset.model_fields["target"].default_factory())
    safim = SafimDataset(subset="block", revision="r", prompt=_TMPL)
    assert safim.corpus_id() != plain.corpus_id()
    assert safim.corpus_id().startswith(plain.corpus_id())


def test_the_split_column_and_the_placeholder_are_both_in_the_corpus_id():
    # corpus() splits eval_prompt_col at placeholder and then drops it, so configs
    # naming either differently write different prefix/suffix into rows.json.
    base = dict(subset="block", revision="r", prompt=_TMPL)
    a = SafimDataset(**base)
    assert a.corpus_id() != SafimDataset(**base, eval_prompt_col="prompt").corpus_id()
    assert a.corpus_id() != SafimDataset(**base, placeholder="<FILL>").corpus_id()


def test_both_classes_render_the_template_they_are_given():
    assert FimSessionDataset(subset="add", revision="r", prompt=_TMPL).prompt is _TMPL
    assert SafimDataset(subset="block", revision="r", prompt=_TMPL).prompt is _TMPL
