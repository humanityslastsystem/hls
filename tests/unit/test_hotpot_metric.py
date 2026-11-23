from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.verify.metrics.hotpot import HotpotEmF1


def _ev(rid, pred, gt):
    inp = Envelope(request_id=rid, contents=[TextPayload(text="q")])
    out = Envelope(request_id=rid, contents=[TextPayload(text=pred)])
    return TraceEvent(request_id=rid, input=inp,
                      response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0,
                      ground_truth=gt)


def test_exact_match_scores_one():
    res = HotpotEmF1().score([_ev("r0", "The Eiffel Tower", "the eiffel tower")])
    assert res.per_event["r0"].score == 1.0
    assert res.per_event["r0"].diagnostics["em"] == 1.0


def test_partial_overlap_is_partial_f1():
    res = HotpotEmF1().score([_ev("r0", "Paris France", "Paris")])
    f1 = res.per_event["r0"].score
    assert 0.0 < f1 < 1.0
    assert res.per_event["r0"].diagnostics["em"] == 0.0
