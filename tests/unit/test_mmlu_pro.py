"""Unit tests for MMLUProMatch (the TIGER-Lab/MMLU-Pro letter-extraction exact
match metric) and MMLUProDataset (the adapter): the answer-letter extraction
rules (answer-is regex, last match wins, case-insensitivity, bare-letter
fallback, fail-closed on unparseable text), the corpus-row -> DatasetItem
rendering (pre-rendered prompt/target columns riding HFDataset's single-turn
mode), the instruction-first render format, and the seeded subsample helper.
No network access -- item rendering is exercised via `items_at` on fixture
rows, mirroring tests/unit/test_mrcr.py and tests/unit/test_bfcl.py."""
from __future__ import annotations

import json

import pytest
from hls.core.faults import HarnessFault
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics.mmlu_pro import MMLUProMatch, extract_letter


def _event(response_text: str, gt, rid: str = "r0") -> TraceEvent:
    inp = Envelope(request_id=rid, contents=[TextPayload(text="prompt")])
    out = Envelope(request_id=rid, contents=[TextPayload(text=response_text)])
    return TraceEvent(request_id=rid, input=inp, response=ArtifactResponse(output=out),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


# --- extract_letter -----------------------------------------------------------

def test_extract_from_full_cot_string():
    text = "Let's reason step by step... Therefore, the answer is (D)."
    assert extract_letter(text) == "D"


def test_bare_letter_matches():
    assert extract_letter("D") == "D"


def test_case_insensitive():
    assert extract_letter("the answer is d") == "D"


def test_unparseable_returns_none():
    assert extract_letter("I think it's option four") is None


def test_last_match_wins_when_several_letters_mentioned():
    text = ("Option A looks tempting, and B could also work. On reflection the "
            "answer is (C). Actually, wait -- the answer is (D).")
    assert extract_letter(text) == "D"


def test_empty_text_returns_none():
    assert extract_letter("") is None
    assert extract_letter(None) is None


# --- MMLUProMatch ---------------------------------------------------------------

def test_correct_letter_passes():
    ev = _event("... Therefore, the answer is (D).", "D")
    v = MMLUProMatch().score_one(ev)
    assert v.passed is True
    assert v.score == 1.0


def test_wrong_letter_fails():
    ev = _event("... Therefore, the answer is (D).", "C")
    v = MMLUProMatch().score_one(ev)
    assert v.passed is False
    assert v.score == 0.0


def test_unparseable_answer_fails_closed():
    ev = _event("I think it's option four", "D")
    v = MMLUProMatch().score_one(ev)
    assert v.passed is False
    assert v.score == 0.0


def test_score_reduces_over_batch():
    ok = _event("the answer is (A)", "A", rid="r0")
    bad = _event("the answer is (B)", "A", rid="r1")
    res = MMLUProMatch().score([ok, bad])
    assert res.value == 0.5


def test_missing_ground_truth_raises_harness_fault():
    ev = _event("the answer is (D)", None)
    with pytest.raises(HarnessFault):
        MMLUProMatch().score([ev])


# --- MMLUProDataset ---------------------------------------------------------------

def _write(tmp_path, rows):
    d = tmp_path / "test"
    d.mkdir()
    (d / "rows.json").write_text(json.dumps(rows))
    return tmp_path


def test_mmlu_pro_item_renders_prerendered_prompt_and_target(tmp_path):
    from hls.spec.datasets import MMLUProDataset, Split
    rows = [{"question_id": 1,
             "prompt": ("Answer the following multiple choice question...\n\n"
                        "Question: What is 2+2?\nOptions:\nA. 3\nB. 4\nAnswer:"),
             "target": "B"}]
    ds = MMLUProDataset()
    items = list(ds.items_at(_write(tmp_path, rows), Split.TEST))
    assert len(items) == 1
    assert len(items[0].turns) == 1
    assert items[0].request.first_of("text").text == rows[0]["prompt"]
    assert items[0].ground_truth == "B"


def test_multiple_rows_render_independently(tmp_path):
    from hls.spec.datasets import MMLUProDataset, Split
    rows = [{"question_id": 1, "prompt": "Question: Q1\nAnswer:", "target": "A"},
            {"question_id": 2, "prompt": "Question: Q2\nAnswer:", "target": "J"}]
    ds = MMLUProDataset()
    items = list(ds.items_at(_write(tmp_path, rows), Split.TEST))
    assert [it.ground_truth for it in items] == ["A", "J"]
    assert [it.request.first_of("text").text for it in items] == ["Question: Q1\nAnswer:",
                                                                   "Question: Q2\nAnswer:"]


def test_unmounted_split_fails_loud(tmp_path):
    from hls.spec.datasets import MMLUProDataset, Split
    ds = MMLUProDataset()
    with pytest.raises(FileNotFoundError, match="(?i)test split not mounted"):
        list(ds.items_at(tmp_path, Split.TEST))


def test_corpus_id_includes_subsample_size_and_seed():
    from hls.spec.datasets import MMLUProDataset
    ds = MMLUProDataset()
    cid = ds.corpus_id()
    assert "TIGER-Lab/MMLU-Pro" in cid
    assert "n=2500" in cid
    assert "seed=20240915" in cid


def test_corpus_id_reflects_overridden_size_and_seed():
    from hls.spec.datasets import MMLUProDataset
    ds = MMLUProDataset(subsample_size=100, subsample_seed=7)
    cid = ds.corpus_id()
    assert "n=100" in cid
    assert "seed=7" in cid


def test_default_revision_is_pinned():
    from hls.spec.datasets import MMLUProDataset
    ds = MMLUProDataset()
    assert ds.revision == "b189ec765aa7ed75c8acfea42df31fdae71f97be"
    assert ds.source_split == "test"


# --- render_prompt (instruction-first rendering) --------------------------------

def test_render_prompt_is_instruction_first_with_lettered_options():
    from hls.spec.datasets import MMLU_PRO_INSTRUCTION, MMLUProDataset
    text = MMLUProDataset.render_prompt("What is 2+2?", ["3", "4", "5"])
    expected = (f"{MMLU_PRO_INSTRUCTION}\n\n"
                "Question: What is 2+2?\nOptions:\nA. 3\nB. 4\nC. 5\nAnswer:")
    assert text == expected


def test_render_prompt_handles_up_to_ten_options():
    from hls.spec.datasets import MMLUProDataset
    options = [f"opt{i}" for i in range(10)]
    text = MMLUProDataset.render_prompt("Q?", options)
    assert "J. opt9" in text
    assert "A. opt0" in text


# --- seeded subsample helper -----------------------------------------------------

def test_sample_indices_returns_exact_count_sorted_and_in_range():
    from hls.spec.datasets import MMLUProDataset
    idx = MMLUProDataset.sample_indices(1000, 50, seed=123)
    assert len(idx) == 50
    assert len(set(idx)) == 50
    assert idx == sorted(idx)
    assert all(0 <= i < 1000 for i in idx)


def test_sample_indices_deterministic_across_calls():
    from hls.spec.datasets import MMLUProDataset
    a = MMLUProDataset.sample_indices(12032, 2500, seed=20240915)
    b = MMLUProDataset.sample_indices(12032, 2500, seed=20240915)
    assert a == b


def test_sample_indices_different_seed_gives_different_sample():
    from hls.spec.datasets import MMLUProDataset
    a = MMLUProDataset.sample_indices(1000, 50, seed=1)
    b = MMLUProDataset.sample_indices(1000, 50, seed=2)
    assert a != b
