from hls.core.poly import dump_spec
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit, SloLatencyScorer


def test_scorer_dumps_poly_fields():
    s = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=2.0))
    rec = dump_spec(s)
    assert rec["unit"]["kind"] == "RequestUnit"
    assert rec["sol"]["kind"] == "ConstantSoL" and rec["sol"]["rps"] == 2.0


def test_unit_name_is_classvar_not_a_field():
    rec = dump_spec(RequestUnit())
    assert rec == {"kind": "RequestUnit"}                   # name is a ClassVar, not serialized


def test_scorer_kind_names_the_objective_a_run_was_graded_on():
    # The two members are field-incompatible but both land under `scorer` in provenance,
    # so the discriminator is the only record of which objective produced the score.
    assert dump_spec(GoodputScorer(unit=RequestUnit(), sol=ConstantSoL()))["kind"] \
        == "GoodputScorer"
    assert dump_spec(SloLatencyScorer())["kind"] == "SloLatencyScorer"


def test_slo_latency_scorer_records_the_baseline_it_was_scored_against():
    # The score is exp(L - B), so the record is unreadable without B: it must serialize.
    # Whether a scorer grades against a baseline at all is a method, not a field, so it
    # stays out of the record: it tells the loader what to inject, and is not a property
    # of the run.
    rec = dump_spec(SloLatencyScorer(baseline_log_credit=0.75))
    assert rec == {"kind": "SloLatencyScorer", "baseline_log_credit": 0.75}
    assert dump_spec(SloLatencyScorer())["baseline_log_credit"] is None
