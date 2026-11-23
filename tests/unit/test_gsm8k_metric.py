from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics.gsm8k import Gsm8kExactMatch, extract_final_number


def _ev(rid, pred_text, gt):
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")])
    out = Envelope(request_id=rid, contents=[TextPayload(text=pred_text)])
    return TraceEvent(request_id=rid, input=inp,
                      response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0,
                      ground_truth=gt)


def test_extract_final_number():
    assert extract_final_number("The answer is #### 42") == "42"
    assert extract_final_number("...so 1,024 total.") == "1024"
    assert extract_final_number("no digits here") is None
    assert extract_final_number("worth $3.50") == "3.50"


def test_metric_passes_on_match():
    ev = _ev("r0", "Reasoning... #### 18", "Long solution #### 18")
    res = Gsm8kExactMatch().score([ev])
    assert res.value == 1.0
    assert res.per_event["r0"].passed is True


def test_metric_fails_on_mismatch():
    ev = _ev("r0", "#### 7", "#### 8")
    res = Gsm8kExactMatch().score([ev])
    assert res.value == 0.0
