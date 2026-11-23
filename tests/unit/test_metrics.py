import pytest

from hls.core.faults import HarnessFault
from hls.core.payloads import Envelope, TextPayload
from hls.core.verdict import MetricResult
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics import CoqaF1Metric
from hls.verify import ContainsMatchMetric, LetterMatchMetric


def _event(req, text, gt=None):
    out = Envelope(request_id=req, contents=[TextPayload(text=text)])
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")])
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out),
                      ground_truth=gt, t_send=0.0, t_recv=1.0)


def test_letter_match_value_is_accuracy():
    m = LetterMatchMetric()
    r = m.score([_event("r0", "The answer is B", gt="B"), _event("r1", "A", gt="B")])
    assert isinstance(r, MetricResult)
    assert r.value == 0.5                                  # 1 of 2 correct
    assert r.per_event["r0"].passed is True and r.per_event["r1"].passed is False


def test_contains_match_value_is_accuracy():
    m = ContainsMatchMetric()
    r = m.score([_event("r0", "the answer is forty-two", gt="forty-two"),
                 _event("r1", "nope", gt="forty-two")])
    assert r.value == 0.5


def test_label_metric_raises_when_no_labels():
    m = LetterMatchMetric()
    with pytest.raises(HarnessFault, match="ground-truth labels"):
        m.score([_event("r0", "B", gt=None), _event("r1", "A", gt=None)])


def test_label_metric_ok_with_partial_labels():
    # one missing label is just a wrong event, not a misconfigured dataset
    m = LetterMatchMetric()
    r = m.score([_event("r0", "B", gt="B"), _event("r1", "A", gt=None)])
    assert r.value == 0.5


def test_coqa_f1_metric_value_is_mean_f1():
    m = CoqaF1Metric()
    # turn 0 exact (f1=1.0), turn 1 no overlap (f1=0.0) -> mean 0.5
    r = m.score([_event("r0", "in a barn", gt="in a barn"),
                 _event("r1", "totally wrong", gt="in a barn")])
    assert r.value == pytest.approx(0.5)
    assert r.per_event["r0"].score == pytest.approx(1.0)


def test_coqa_f1_metric_raises_when_no_labels():
    with pytest.raises(HarnessFault, match="ground-truth labels"):
        CoqaF1Metric().score([_event("r0", "in a barn", gt=None)])


def test_wer_does_not_charge_a_transcript_for_formatting_numbers():
    """A recognizer writing "$250,000" and a corpus writing it out mean the same thing.

    Scoring them as plain tokens charges the correct transcript six edits, and hands a
    server free accuracy for spelling numbers back out -- a formatting choice, not a
    transcription one. The row is from the ASR dev split.
    """
    from hls.verify.metrics.asr import word_error_rate
    said = "Find some cress wells there. Big plantations, rated at $250,000."
    truth = ("find some cresswells there big plantations rated at "
             "two hundred and fifty thousand dollars")
    # "cress wells" against "cresswells" is a real split and still costs; the number is
    # what must not cost.
    assert 1.0 - word_error_rate(said, truth) > 0.7

    # Spelling the number out must not pay better than writing it.
    spelled = ("find some cresswells there big plantations rated at "
               "two hundred and fifty thousand dollars")
    formatted = "Find some Cresswells there. Big plantations, rated at $250,000."
    assert word_error_rate(spelled, truth) == word_error_rate(formatted, truth) == 0.0


def test_wer_empty_reference_conventions():
    from hls.verify.metrics.asr import word_error_rate
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("something", "") == 1.0
    assert word_error_rate("", "a b") == 1.0
