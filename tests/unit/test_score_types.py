import pytest
from pydantic import ValidationError

from hls.core.verdict import MetricResult, PerEventVerdict


def test_per_event_verdict_defaults():
    pv = PerEventVerdict(passed=True)
    assert pv.score is None and pv.rationale is None and pv.diagnostics == {}


def test_metric_result_holds_value_and_per_event():
    r = MetricResult(value=0.75,
                     per_event={"r0": PerEventVerdict(passed=True, score=1.0),
                                "r1": PerEventVerdict(passed=False, score=0.5)},
                     diagnostics={"n": 2})
    assert r.value == 0.75
    assert r.per_event["r1"].score == 0.5
    assert r.diagnostics["n"] == 2


def test_both_are_frozen():
    pv = PerEventVerdict(passed=True)
    with pytest.raises(ValidationError):
        pv.passed = False
