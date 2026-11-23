from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent


def _ev(rid, t_send, t_first, ts):
    out = Envelope(request_id=rid, contents=[TextPayload(text="x")])
    return TraceEvent(
        request_id=rid, input=Envelope(request_id=rid, contents=[TextPayload(text="q")]),
        response=ArtifactResponse(output=out, t_first_packet=t_first, packet_timestamps=ts),
        t_send=t_send, t_recv=(ts[-1] if ts else None))


def test_ttft_only_counts_streamed():
    tr = Trace(events=[
        _ev("a", 1.0, 1.2, [1.2, 1.4, 1.6]),
        _ev("b", 2.0, None, None),
    ])
    result = tr.ttft_sec()
    assert len(result) == 1
    assert abs(result[0] - 0.2) < 1e-9


def test_tpot_needs_two_tokens():
    tr = Trace(events=[
        _ev("a", 0.0, 1.0, [1.0, 1.5, 2.0]),
        _ev("b", 0.0, 3.0, [3.0]),
    ])
    result = tr.tpot_sec()
    assert len(result) == 1
    assert abs(result[0] - 0.5) < 1e-9
