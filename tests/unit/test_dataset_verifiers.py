"""Unit tests for the dataset-based metrics via MetricVerifier (no oracle/engine):
LetterMatchMetric, ContainsMatchMetric, CoqaF1Metric. Each compares the artifact
output to the event's ``ground_truth``; ``requires`` is empty. These pin each
metric's semantics and gate, each driven through MetricVerifier."""
from __future__ import annotations

import asyncio

import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics import CoqaF1Metric
from hls.verify import (ArtifactRollout, ContainsMatchMetric, EngineAccessor,
                        LetterMatchMetric, MetricVerifier)


def _event(req: str, text: str, gt=None) -> TraceEvent:
    out = Envelope(request_id=req, contents=[TextPayload(text=text)])
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")])
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


def _run(metric, events, min_value=1.0):
    v = MetricVerifier(metric=metric, min_value=min_value)
    return asyncio.run(v.verify(EngineAccessor(artifact_events=events, cap_to_provider={})))


# --- LetterMatchMetric -------------------------------------------------

def test_letter_match_all_match_passes():
    events = [_event("r0", "The answer is B", gt="B"), _event("r1", "A", gt="A")]
    verdict = _run(LetterMatchMetric(), events)
    assert verdict.passed is True
    assert verdict.per_event == {"r0": True, "r1": True}
    assert verdict.diagnostics["value"] == pytest.approx(1.0)


def test_letter_match_one_miss_fails_at_full_rate():
    events = [_event("r0", "B", gt="B"), _event("r1", "A", gt="C")]
    verdict = _run(LetterMatchMetric(), events)
    assert verdict.passed is False
    assert verdict.per_event == {"r0": True, "r1": False}
    assert verdict.diagnostics["value"] == pytest.approx(0.5)


def test_letter_match_one_miss_passes_at_half_rate():
    events = [_event("r0", "B", gt="B"), _event("r1", "A", gt="C")]
    verdict = _run(LetterMatchMetric(), events, min_value=0.5)
    assert verdict.passed is True
    assert verdict.per_event == {"r0": True, "r1": False}
    assert verdict.diagnostics["value"] == pytest.approx(0.5)


def test_letter_match_no_letter_in_output_misses():
    events = [_event("r0", "no letters here", gt="B")]
    verdict = _run(LetterMatchMetric(), events)
    assert verdict.per_event == {"r0": False}


def test_letter_match_failed_event_misses():
    # A failed event (response is None) scores False, dropping the pass-rate.
    inp = Envelope(request_id="r0", contents=[TextPayload(text="Q?")])
    ev = TraceEvent(request_id="r0", input=inp, response=None, ground_truth="B",
                    t_send=0.0, t_recv=None)
    verdict = _run(LetterMatchMetric(), [ev])
    assert verdict.passed is False
    assert verdict.per_event == {"r0": False}


# --- ContainsMatchMetric -----------------------------------------------

def test_contains_match_substring_hit():
    events = [_event("r0", "The capital is Paris, France.", gt="paris")]
    verdict = _run(ContainsMatchMetric(), events)
    assert verdict.passed is True
    assert verdict.per_event == {"r0": True}


def test_contains_match_substring_miss():
    events = [_event("r0", "The capital is Berlin.", gt="paris")]
    verdict = _run(ContainsMatchMetric(), events)
    assert verdict.passed is False
    assert verdict.per_event == {"r0": False}


def test_contains_match_one_miss_passes_at_half_rate():
    events = [_event("r0", "It is Paris.", gt="paris"),
              _event("r1", "It is Berlin.", gt="madrid")]
    verdict = _run(ContainsMatchMetric(), events, min_value=0.5)
    assert verdict.passed is True
    assert verdict.per_event == {"r0": True, "r1": False}


def test_contains_match_empty_needle_misses():
    events = [_event("r0", "anything", gt="")]
    verdict = _run(ContainsMatchMetric(), events)
    assert verdict.per_event == {"r0": False}


# --- CoqaF1Metric ------------------------------------------------------

def test_coqa_f1_exact_match_mean_is_one():
    events = [_event("r0", "in a barn", gt="in a barn"),
              _event("r1", "yes", gt="yes")]
    verdict = _run(CoqaF1Metric(), events, min_value=0.5)
    assert verdict.passed is True
    assert verdict.per_event == {"r0": True, "r1": True}
    assert verdict.diagnostics["value"] == pytest.approx(1.0)
    assert verdict.diagnostics["min_value"] == 0.5


def test_coqa_f1_partial_overlap_fraction():
    # r0 exact (1.0); r1 partial overlap = 1/3 (see coqa_f1 math tests).
    events = [_event("r0", "in a barn", gt="in a barn"),
              _event("r1", "Above the barn near a farmhouse", gt="in a barn")]
    verdict = _run(CoqaF1Metric(), events, min_value=0.5)
    # mean = (1.0 + 1/3) / 2 = 2/3 >= 0.5 -> pass; both have overlap.
    assert verdict.diagnostics["value"] == pytest.approx((1.0 + 1.0 / 3.0) / 2.0)
    assert verdict.passed is True
    assert verdict.per_event == {"r0": True, "r1": True}


def test_coqa_f1_below_min_value_fails():
    # both well below 0.5: r0 zero overlap, r1 = 1/3.
    events = [_event("r0", "completely different", gt="in a barn"),
              _event("r1", "Above the barn near a farmhouse", gt="in a barn")]
    verdict = _run(CoqaF1Metric(), events, min_value=0.5)
    assert verdict.diagnostics["value"] == pytest.approx((0.0 + 1.0 / 3.0) / 2.0)
    assert verdict.passed is False
    assert verdict.per_event == {"r0": False, "r1": True}


def test_coqa_f1_multi_ref_tuple():
    # leave-one-out over a 2-ref tuple: 5/6 (see coqa_f1 math tests).
    events = [_event("r0", "in a barn", gt=("in a barn", "the barn"))]
    verdict = _run(CoqaF1Metric(), events, min_value=0.5)
    assert verdict.diagnostics["value"] == pytest.approx(5.0 / 6.0)
    assert verdict.per_event == {"r0": True}


def test_coqa_f1_failed_event_is_zero():
    inp = Envelope(request_id="r0", contents=[TextPayload(text="Q?")])
    ev = TraceEvent(request_id="r0", input=inp, response=None, ground_truth="in a barn",
                    t_send=0.0, t_recv=None)
    verdict = _run(CoqaF1Metric(), [ev], min_value=0.5)
    assert verdict.diagnostics["value"] == pytest.approx(0.0)
    assert verdict.passed is False
    assert verdict.per_event == {"r0": False}


# --- requires are empty (dataset-based; no oracle) -----------------------

def test_dataset_verifiers_require_nothing():
    assert MetricVerifier(metric=LetterMatchMetric()).requires == frozenset()
    assert MetricVerifier(metric=ContainsMatchMetric()).requires == frozenset()
    assert MetricVerifier(metric=CoqaF1Metric()).requires == frozenset()
    assert ArtifactRollout is ArtifactRollout  # sentinel importable
