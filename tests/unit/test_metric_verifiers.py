import asyncio

from hls.core.capabilities import Generate
from hls.core.decode import Greedy
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.reference.references import HFReference
from hls.verify import (EngineAccessor, LetterMatchMetric, MetricVerifier,
                        ReferenceRelativeVerifier)


def _event(req, text, gt=None):
    out = Envelope(request_id=req, contents=[TextPayload(text=text)])
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")])
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


def _acc(events, cap=None):
    return EngineAccessor(artifact_events=events, cap_to_provider=cap or {})


def _ref():
    return HFReference(decode=Greedy())


class _GenProxy:
    def __init__(self, outs):
        self._outs = outs

    async def generate(self, prompts):
        return self._outs


def test_metric_verifier_passes_when_mean_meets_min_value():
    v = MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", gt="B"), _event("r1", "A", gt="A")])))
    assert verdict.passed is True and verdict.per_event == {"r0": True, "r1": True}


def test_metric_verifier_fails_below_min_value():
    v = MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)
    verdict = asyncio.run(v.verify(_acc([_event("r0", "B", gt="B"), _event("r1", "A", gt="B")])))
    assert verdict.passed is False                          # accuracy 0.5 < 1.0
    assert verdict.diagnostics["value"] == 0.5


def test_reference_relative_compares_values():
    v = ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=HFReference(decode=Greedy()), tol=0.0)
    events = [_event("r0", "B", gt="B"), _event("r1", "A", gt="B")]      # artifact: 1/2
    ref_outs = [Envelope(request_id="r0", contents=[TextPayload(text="B")]),
                Envelope(request_id="r1", contents=[TextPayload(text="B")])]   # reference: 2/2
    verdict = asyncio.run(v.verify(_acc(events, {Generate: _GenProxy(ref_outs).generate})))
    assert verdict.passed is False                          # art 0.5 + 0 < ref 1.0
    assert verdict.diagnostics["artifact_value"] == 0.5 and verdict.diagnostics["reference_value"] == 1.0
    assert verdict.per_event == {"r0": True, "r1": False}   # per-event is the ARTIFACT side


def test_reference_relative_surfaces_oracle_per_event_detail():
    # Path B: the verdict re-exposes the reference's per-event detail (nested under
    # `reference`, with the oracle's raw text) so the viewer can render the per-turn
    # artifact-vs-oracle comparison; the gate still compares only the reduced values.
    v = ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_ref(), tol=0.0)
    events = [_event("r0", "B", gt="B"), _event("r1", "A", gt="B")]
    ref_outs = [Envelope(request_id="r0", contents=[TextPayload(text="C")]),   # oracle != artifact
                Envelope(request_id="r1", contents=[TextPayload(text="B")])]
    verdict = asyncio.run(v.verify(_acc(events, {Generate: _GenProxy(ref_outs).generate})))
    d0 = verdict.event_diagnostics["r0"]
    assert d0["predicted"] == "B" and d0["expected"] == "B"          # artifact side, unchanged
    assert d0["reference"]["predicted"] == "C"                       # the ORACLE's letter, not "B"
    assert d0["reference"]["text"] == "C"                            # the oracle's raw answer
    d1 = verdict.event_diagnostics["r1"]
    assert d1["predicted"] == "A"
    assert d1["reference"]["predicted"] == "B" and d1["reference"]["text"] == "B"


def test_metric_field_rejects_non_metric_at_construction():
    # Nominal isinstance safety: the metric field is typed RolloutMetric, so a non-metric
    # cannot be wired into a verifier -- it fails when the TaskSpec/verifier is built.
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        MetricVerifier(metric=object())


def test_reference_relative_works_with_any_metric():
    # Swapping the metric needs no verifier change: CoqaF1Metric
    # drops straight into ReferenceRelativeVerifier.
    from hls.verify.metrics import CoqaF1Metric
    v = ReferenceRelativeVerifier(metric=CoqaF1Metric(), reference=_ref(), tol=0.0)
    events = [_event("r0", "in a barn", gt="in a barn")]                 # artifact f1 = 1.0
    ref_outs = [Envelope(request_id="r0", contents=[TextPayload(text="totally wrong")])]  # ref f1 = 0.0
    verdict = asyncio.run(v.verify(_acc(events, {Generate: _GenProxy(ref_outs).generate})))
    assert verdict.passed is True                                        # art 1.0 >= ref 0.0


def test_reference_relative_min_ratio_gate():
    # Multiplicative bar: artifact >= reference * min_ratio. art 0.5 vs ref 1.0:
    # fails at 0.95 (0.5 < 0.95) but passes at 0.5 (0.5 >= 0.5).
    events = [_event("r0", "B", gt="B"), _event("r1", "A", gt="B")]      # artifact: 1/2
    ref_outs = [Envelope(request_id="r0", contents=[TextPayload(text="B")]),
                Envelope(request_id="r1", contents=[TextPayload(text="B")])]   # reference: 2/2
    strict = ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_ref(), min_ratio=0.95)
    verdict = asyncio.run(strict.verify(_acc(events, {Generate: _GenProxy(ref_outs).generate})))
    assert verdict.passed is False
    assert verdict.diagnostics["min_ratio"] == 0.95 and "tol" not in verdict.diagnostics
    loose = ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_ref(), min_ratio=0.5)
    verdict = asyncio.run(loose.verify(_acc(events, {Generate: _GenProxy(ref_outs).generate})))
    assert verdict.passed is True


def test_reference_relative_tol_and_min_ratio_are_exclusive():
    import pytest
    with pytest.raises(ValueError, match="not both"):
        ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_ref(),
                                  tol=0.1, min_ratio=0.95)


def test_reference_relative_refuses_a_min_ratio_of_zero():
    """A 0 bar is ``artifact >= reference * 0``, which every metric value in [0, 1]
    clears -- including an artifact that answered nothing. Both multimodal arenas gate
    correctness on this verifier, so a mistyped or defaulted zero would turn the gate
    off for a whole arena without a symptom."""
    import pytest
    with pytest.raises(ValueError, match="min_ratio must be > 0"):
        ReferenceRelativeVerifier(metric=LetterMatchMetric(), reference=_ref(), min_ratio=0.0)
