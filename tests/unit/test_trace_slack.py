from hls.core.trace import Trace, TraceEvent
from hls.core.payloads import Envelope, TextPayload


def _ev(req, planned, send):
    inp = Envelope(request_id=req, contents=[TextPayload(text="q")])
    return TraceEvent(request_id=req, input=inp, t_send=send, t_planned_send=planned)


def test_scheduler_slack_is_send_minus_planned():
    tr = Trace(events=[_ev("r0", 1.0, 1.2), _ev("r1", 2.0, 2.05)])
    slack = tr.scheduler_slack_sec()
    assert abs(slack[0] - 0.2) < 1e-9 and abs(slack[1] - 0.05) < 1e-9


def test_planned_send_optional():
    inp = Envelope(request_id="r", contents=[TextPayload(text="q")])
    ev = TraceEvent(request_id="r", input=inp, t_send=1.0)
    assert ev.t_planned_send is None
