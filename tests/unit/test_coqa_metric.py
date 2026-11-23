"""Unit tests for the official CoQA scoring: the pure F1 math (dataset_verifiers/
coqa.py) ported from CoQA's evaluate-v1.0.py, and the MetricVerifier+CoqaF1Metric
path (per-turn F1 whose run-level verdict is the mean F1 across turns -- the
headline CoQA number)."""
from __future__ import annotations

import asyncio

import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics import CoqaF1Metric, coqa_f1, normalize_answer
from hls.verify import EngineAccessor, MetricVerifier


# --- normalize_answer (official: lower -> strip punct -> drop articles -> ws) ---

@pytest.mark.parametrize("raw,expected", [
    ("The Cat.", "cat"),
    ("a barn", "barn"),
    ("In a Barn!", "in barn"),          # "in" is not an article; kept
    ("  white   ", "white"),
    ("No", "no"),
    ("", ""),
])
def test_normalize_answer(raw, expected):
    assert normalize_answer(raw) == expected


# --- coqa_f1 single reference --------------------------------------------

def test_f1_exact_match_is_one():
    assert coqa_f1("in a barn", ["in a barn"]) == 1.0


def test_f1_normalization_makes_match_exact():
    # articles/punct/case differ but normalize to the same tokens
    assert coqa_f1("In A Barn.", ["in a barn"]) == 1.0


def test_f1_partial_overlap():
    # pred -> [above, barn, near, farmhouse] (4), ref -> [in, barn] (2)
    # common = {barn}: num_same=1, p=1/4, r=1/2 -> f1 = 2*.25*.5/.75 = 1/3
    f1 = coqa_f1("Above the barn near a farmhouse", ["in a barn"])
    assert f1 == pytest.approx(1.0 / 3.0)


def test_f1_no_overlap_is_zero():
    assert coqa_f1("completely different", ["in a barn"]) == 0.0


def test_f1_empty_prediction_is_zero():
    assert coqa_f1("", ["in a barn"]) == 0.0


def test_f1_both_empty_is_one():
    # official: when both token lists are empty, score is int(equal) == 1
    assert coqa_f1("", [""]) == 1.0


def test_f1_yes_no_tokens():
    assert coqa_f1("no", ["no"]) == 1.0
    assert coqa_f1("yes", ["no"]) == 0.0


# --- coqa_f1 multiple references (leave-one-out averaging) ----------------

def test_f1_multi_ref_takes_best_per_fold():
    # 3 refs; prediction exactly equals one of them. Leave-one-out: each fold
    # excludes one ref and takes the max over the rest. The prediction matches
    # "in a barn" exactly, so any fold that still contains it scores 1.0; the
    # fold that excludes it falls back to the next-best ref.
    refs = ["in a barn", "in the barn", "a small barn"]
    f1 = coqa_f1("in a barn", refs)
    # fold0 excludes "in a barn": best of {"in the barn"->in,barn vs in,barn=1.0}
    # fold1 excludes "in the barn": contains "in a barn" -> 1.0
    # fold2 excludes "a small barn": contains "in a barn" -> 1.0
    assert f1 == pytest.approx(1.0)


def test_f1_multi_ref_averages_folds():
    # prediction matches only ref0 exactly; refs 1/2 share just "barn".
    # pred tokens: [big, barn] (2)
    refs = ["big barn", "the barn", "barn door"]
    f1 = coqa_f1("big barn", refs)
    # _f1(pred,"big barn")=1.0; _f1(pred,"barn")=2/3; _f1(pred,"barn door"):
    #   common={barn}, p=1/2,r=1/2 -> 0.5
    # fold0 (drop "big barn"): max(2/3, 0.5)=2/3
    # fold1 (drop "the barn"): max(1.0, 0.5)=1.0
    # fold2 (drop "barn door"): max(1.0, 2/3)=1.0
    # mean = (2/3 + 1 + 1)/3
    assert f1 == pytest.approx((2.0 / 3.0 + 1.0 + 1.0) / 3.0)


def test_f1_no_references_is_zero():
    assert coqa_f1("anything", []) == 0.0


# --- MetricVerifier + CoqaF1Metric (mean per-turn F1 as the run-level verdict) ---

def _event(req: str, text: str | None, gt=None) -> TraceEvent:
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")])
    resp = None
    if text is not None:
        out = Envelope(request_id=req, contents=[TextPayload(text=text)])
        resp = ArtifactResponse(output=out)
    return TraceEvent(request_id=req, input=inp, response=resp, ground_truth=gt,
                      t_send=0.0, t_recv=1.0 if text is not None else None)


def _verify(events):
    v = MetricVerifier(metric=CoqaF1Metric(), min_value=0.5)
    acc = EngineAccessor(artifact_events=events, cap_to_provider={})
    return asyncio.run(v.verify(acc))


def test_coqa_verifier_requires_nothing():
    assert MetricVerifier(metric=CoqaF1Metric()).requires == frozenset()
    assert MetricVerifier(metric=CoqaF1Metric(), min_value=0.5).min_value == 0.5


def test_coqa_verifier_exact_match_mean_one_passes():
    v = _verify([_event("r", "in a barn", gt="in a barn")])
    assert v.diagnostics["value"] == 1.0
    assert v.passed is True
    assert v.per_event == {"r": True}                # f1 > 0


def test_coqa_verifier_partial_overlap_scores_fraction():
    v = _verify([_event("r", "Above the barn near a farmhouse", gt="in a barn")])
    assert v.diagnostics["value"] == pytest.approx(1.0 / 3.0)
    assert v.passed is False                         # 1/3 < 0.5
    assert v.per_event == {"r": True}                # any token overlap


def test_coqa_verifier_handles_multi_ref_tuple():
    v = _verify([_event("r", "in a barn", gt=("in a barn", "the barn"))])
    # leave-one-out: fold dropping "in a barn" scores f1(pred,"the barn")=2/3;
    # fold dropping "the barn" scores f1(pred,"in a barn")=1.0; mean = 5/6.
    assert v.diagnostics["value"] == pytest.approx(5.0 / 6.0)


def test_coqa_verifier_none_response_is_zero():
    v = _verify([_event("r", None, gt="in a barn")])
    assert v.diagnostics["value"] == 0.0
    assert v.passed is False
    assert v.per_event == {"r": False}               # no overlap


def test_coqa_verifier_mean_over_turns_passes_min():
    # mean of {1.0, 0.0, 0.8} = 0.6 >= 0.5 (the headline mean-F1 criterion).
    events = [_event("r0", "in a barn", gt="in a barn"),
              _event("r1", "completely different", gt="in a barn"),
              _event("r2", "big barn here", gt="big barn here")]
    v = _verify(events)
    assert v.diagnostics["value"] == pytest.approx((1.0 + 0.0 + 1.0) / 3.0)
    assert v.passed is True


def test_coqa_verifier_mean_below_min_fails():
    events = [_event("r0", "completely different", gt="in a barn"),
              _event("r1", "nothing matching", gt="big barn")]
    v = _verify(events)
    assert v.diagnostics["value"] == pytest.approx(0.0)
    assert v.passed is False
