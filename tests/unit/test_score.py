import pytest

from hls.core.verdict import Verdict
from hls.core.score import SLOResult, Score


def test_verdict_round_trip():
    v = Verdict(passed=False, per_event={"r1": True, "r2": False}, diagnostics={"pass_rate": 0.5})
    assert v.passed is False
    assert v.per_event["r2"] is False


def test_slo_result_credit():
    r = SLOResult(aggregate_passed=True, per_event_credit={"r1": 1.0, "r2": 0.0}, diagnostics={})
    assert r.per_event_credit["r1"] == 1.0


def test_score_zero_constructor():
    s = Score.zero(reasons=("correctness",), diagnostics={"k": "v"})
    assert s.value == 0.0
    assert s.gate_passed is False
    assert "correctness" in s.zero_reasons


def test_score_basic_fields():
    s = Score(value=0.42, achieved=4.2, sol=10.0, unit="request", gate_passed=True)
    assert s.value == 0.42
    assert s.zero_reasons == ()


def test_verdict_carries_diagnostics():
    """A failing Verdict carries the verifier's diagnostics: the verifier's Verdict is
    the run-level result that carries them."""
    v = Verdict(passed=False, per_event={"r0": False}, diagnostics={"pass_rate": 0.4})
    assert v.passed is False
    assert v.diagnostics["pass_rate"] == 0.4


def test_score_zero_nests_diagnostics_and_keeps_top_level_entries():
    """A gated zero carries two things: the failing section's diagnostics, nested under
    the section the reason belongs to, and whatever top-level entries the scorer still
    reports on a zeroed run (its upper bound). Both must survive."""
    s = Score.zero(reasons=("slo",), diagnostics={"n_missing": 1},
                   breakdown={"upper_bound": 0.45})
    assert s.breakdown["slo"] == {"n_missing": 1}
    assert s.breakdown["upper_bound"] == 0.45
    assert s.value == 0.0 and s.gate_passed is False


def test_score_zero_without_diagnostics_still_carries_its_breakdown():
    s = Score.zero(reasons=("correctness",), breakdown={"upper_bound": 2.0})
    assert s.breakdown == {"upper_bound": 2.0}


def test_score_zero_refuses_to_guess_a_section_for_several_reasons():
    """`diagnostics` names one section by its reason, so a zero with several reasons
    cannot use it -- silently filing them under the first reason would mislabel them."""
    with pytest.raises(ValueError, match="pass the sections in `breakdown`"):
        Score.zero(reasons=("correctness", "slo"), diagnostics={"k": "v"})


def test_score_zero_refuses_a_bare_string_of_reasons():
    """A string is a collection of its own characters, so `reasons="slo"` is accepted by
    the annotation and files three one-character reasons into score.json and every index
    row read off it."""
    with pytest.raises(TypeError, match="collection of its characters"):
        Score.zero(reasons="slo", breakdown={"upper_bound": 0.45})
    # A single-character reason is the corner that passes the arity guard as well, so it
    # is corrupted with no error at all unless the type itself is refused.
    with pytest.raises(TypeError, match="collection of its characters"):
        Score.zero(reasons="a", diagnostics={"k": "v"})


def test_the_arity_refusal_reports_the_reasons_in_the_order_they_were_passed():
    """Sorting them made the message disagree with `zero_reasons`, which keeps the order
    the scorer checked the layers in."""
    with pytest.raises(ValueError, match=r"\('slo', 'correctness'\)"):
        Score.zero(reasons=("slo", "correctness"), diagnostics={"k": "v"})


def test_score_reasons_survive_the_json_round_trip_a_record_is_written_through():
    """Scores are persisted as JSON, where the tuple is a list; reading one back must
    give the tuple again, so a record's reasons read the same as a fresh score's."""
    s = Score.zero(reasons=("correctness", "slo"), breakdown={"upper_bound": 0.45})
    back = Score.model_validate_json(s.model_dump_json())
    assert back.zero_reasons == ("correctness", "slo")
    assert back.gate_passed is False and back.value == 0.0


def test_slo_result_defaults_to_no_ratios_at_all():
    """An SLO that defines no latency normalizer leaves the ratio map empty, so a
    scorer built on it can tell "no ratio here" from "ratio of zero"."""
    r = SLOResult(aggregate_passed=True, per_event_credit={"r1": 1.0})
    assert r.per_event_ratio == {}
