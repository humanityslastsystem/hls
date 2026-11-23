from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent, Trace, ArtifactResponse


def _ev(rid: str, sid: str, t_send: float, t_recv: float) -> TraceEvent:
    env = Envelope(request_id=rid, contents=[TextPayload(text="x")])
    return TraceEvent(
        request_id=rid, session_id=sid, turn_index=0,
        input=env, response=ArtifactResponse(output=env), ground_truth=None,
        t_send=t_send, t_recv=t_recv,
    )


def test_trace_partitions():
    trace = Trace(events=[
        _ev("a.0", "a", 0.0, 0.1),
        _ev("a.1", "a", 0.1, 0.2),
        _ev("b.0", "b", 0.0, 0.05),
    ])
    by_session = trace.by_session()
    assert set(by_session.keys()) == {"a", "b"}
    assert len(by_session["a"]) == 2
    assert len(by_session["b"]) == 1


def test_trace_latencies():
    trace = Trace(events=[
        _ev("a.0", "a", 0.0, 0.10),
        _ev("a.1", "a", 0.10, 0.25),
    ])
    assert trace.latencies_sec() == [0.10, 0.15]


def test_artifact_response_defaults_to_no_timing():
    env = Envelope(request_id="r", contents=[TextPayload(text="x")])
    resp = ArtifactResponse(output=env)
    assert resp.output is env
    assert resp.t_first_packet is None
    assert resp.packet_timestamps is None


def test_artifact_response_carries_timing():
    env = Envelope(request_id="r", contents=[TextPayload(text="x")])
    resp = ArtifactResponse(output=env, t_first_packet=1.5, packet_timestamps=[1.5, 1.7])
    assert resp.t_first_packet == 1.5
    assert resp.packet_timestamps == [1.5, 1.7]


def _ev_tenant(rid: str, tenant_id: str | None) -> TraceEvent:
    """Event with optional tenant_id stamp in metadata."""
    meta = {"tenant_id": tenant_id} if tenant_id is not None else {}
    env = Envelope(request_id=rid, contents=[TextPayload(text="x")], metadata=meta)
    return TraceEvent(
        request_id=rid, session_id=rid, turn_index=0,
        input=env, response=ArtifactResponse(output=env), ground_truth=None,
        t_send=0.0, t_recv=0.1,
    )


def test_events_for_multi_tenant_returns_correct_slice():
    """events_for(tid) returns only the events stamped with that tenant_id."""
    trace = Trace(events=[
        _ev_tenant("a", "t00"),
        _ev_tenant("b", "t01"),
        _ev_tenant("c", "t00"),
    ])
    assert [e.request_id for e in trace.events_for("t00")] == ["a", "c"]
    assert [e.request_id for e in trace.events_for("t01")] == ["b"]
    assert trace.events_for("missing") == []


def test_events_for_unstamped_events_not_returned_for_t0():
    """No None fallback: events_for('t0') does NOT pick up unstamped (None-keyed)
    events. The driver always stamps tenant_id (single tenant -> 't0'), so
    production never relies on a fallback; unstamped test events must stamp 't0'."""
    trace = Trace(events=[
        _ev_tenant("a", None),   # unstamped -> filed under the None key
        _ev_tenant("b", None),
    ])
    assert trace.events_for("t0") == []


def test_events_for_real_t0_stamp():
    """events_for('t0') returns the t0-stamped slice when events carry tenant_id='t0'."""
    trace = Trace(events=[
        _ev_tenant("a", "t0"),
        _ev_tenant("b", "t0"),
    ])
    result = trace.events_for("t0")
    assert [e.request_id for e in result] == ["a", "b"]


def test_t_end_is_the_last_timestamp_in_the_drive():
    # The run ends at the last thing that happened: here the second event's t_recv.
    trace = Trace(events=[_ev("a", "a", 0.0, 1.5), _ev("b", "b", 0.5, 2.25)])
    assert trace.t_end() == 2.25
    assert Trace(events=[]).t_end() == 0.0


def test_t_end_covers_a_request_the_harness_gave_up_on():
    # A failed request has no t_recv, so its abandonment time is the only record of how
    # long it ran. It must still bound the window: a request abandoned at 9.0s means the
    # drive was still going at 9.0s, however early every other event finished.
    from hls.core.trace import FailureReason, RequestAttempt, RequestFailure

    env = Envelope(request_id="dead", contents=[TextPayload(text="x")])
    dead = TraceEvent(request_id="dead", input=env, response=None, error="timeout",
                      failure=RequestFailure(reason=FailureReason.DEADLINE, t_abandoned=9.0,
                                             observed=RequestAttempt()),
                      t_send=0.0, t_recv=None)
    assert Trace(events=[_ev("ok", "s", 0.0, 1.0), dead]).t_end() == 9.0


def test_t_end_covers_a_stream_still_arriving_after_t_recv():
    # Packet timestamps count too, so a trace whose events carry streaming timing is
    # bounded by the last packet even if it is stamped after the response was recorded.
    env = Envelope(request_id="r0", contents=[TextPayload(text="x")])
    ev = TraceEvent(request_id="r0", input=env,
                    response=ArtifactResponse(output=env, t_first_packet=0.1,
                                              packet_timestamps=[0.1, 3.0]),
                    t_send=0.0, t_recv=2.0)
    assert Trace(events=[ev]).t_end() == 3.0
