"""Realistic drive traces for the trace-store tests and the stages that read traces: three
tenants on three SLO kinds, multi-turn sessions emitted interleaved, streamed and unary
replies, logprobs with non-finite values, a request the drive deadline abandoned after a
partial stream, and one the circuit breaker never sent."""

from __future__ import annotations

from hls.core.model import AcceleratorSpec, Hardware
from hls.core.payloads import Envelope, TextPayload, TokenLogprob, TopLogprob
from hls.core.trace import (ArtifactResponse, FailureReason, PartialOutput, RequestAttempt, RequestFailure,
                            Trace, TraceEvent)
from hls.core.verdict import Verdict
from hls.spec.tenant import Tenant
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, StreamingLatencySLO, TrajectoryMakespanSLO,
                               Uncapped, Workload)

NAN = float("nan")
INF = float("inf")


def _workload(slo) -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(), slo=slo)


#: The tenants of ``drive_trace``, one each of three SLO kinds.
TENANTS = [
    Tenant(id="chat", workload=_workload(StreamingLatencySLO(ttft_ms=500.0, tpot_ms=50.0)), verifiers=[]),
    Tenant(id="agent", workload=_workload(TrajectoryMakespanSLO(target_sec=30.0)), verifiers=[]),
    Tenant(id="plain", workload=_workload(NoSLO()), verifiers=[]),
]
#: The tenants a latency scorer can price: every one defines a ratio.
LATENCY_TENANTS = TENANTS[:2]
HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1))


def _input(tenant: str, rid: str, session: str, turn: int) -> Envelope:
    return Envelope(request_id=rid, contents=[TextPayload(text=f"ask {rid}")],
                    metadata={"tenant_id": tenant, "session_id": session, "turn_index": turn,
                              "max_tokens": 64, "messages": [{"role": "user", "content": f"ask {rid}"}],
                              "shape": (1, 2), "by_id": {7: "seven"}})


def _reply(rid: str, n: int, *, nan_at: int | None = None) -> Envelope:
    logprobs = []
    for i in range(n):
        lp = NAN if i == nan_at else -0.1 * (i + 1)
        logprobs.append(TokenLogprob(token=f"t{i}", token_id=100 + i, logprob=lp,
                                     top_logprobs=[TopLogprob(token_id=100 + i, logprob=lp),
                                                   TopLogprob(token_id=9, logprob=-INF)]))
    return Envelope(request_id=rid, contents=[TextPayload(text="x" * n, tokens=list(range(100, 100 + n)),
                                                          logprobs=logprobs, finish_reason="stop")])


def _answered(tenant: str, session: str, turn: int, t_send: float, *, streamed: bool, n: int = 4,
              planned: float | None = None, nan_at: int | None = None) -> TraceEvent:
    rid = f"{session}.{turn}"
    t_first = t_send + 0.2
    packets = [t_first + 0.03 * i for i in range(n)] if streamed else None
    t_recv = (packets[-1] if packets else t_send + 1.5) + 0.01
    return TraceEvent(request_id=rid, session_id=session, turn_index=turn,
                      input=_input(tenant, rid, session, turn),
                      response=ArtifactResponse(output=_reply(rid, n, nan_at=nan_at),
                                                t_first_packet=t_first if streamed else None,
                                                packet_timestamps=packets),
                      ground_truth=("gold", turn), t_send=t_send, t_recv=t_recv, t_planned_send=planned,
                      extras={"prefill_tokens": 12 + turn, "gap": INF, "pair": (turn, n)})


def _abandoned(tenant: str, session: str, turn: int, t_send: float) -> TraceEvent:
    rid = f"{session}.{turn}"
    attempt = RequestAttempt(n_frames=3, t_first_frame=t_send + 0.4, status_code=None,
                             partial=PartialOutput(token_ids=[1, 2, 3], text="par", finish_reason=None))
    return TraceEvent(request_id=rid, session_id=session, turn_index=turn,
                      input=_input(tenant, rid, session, turn),
                      t_send=t_send, t_planned_send=t_send - 0.05 if turn == 0 else None,
                      error="deadline: abandoned after 9.0s",
                      failure=RequestFailure(reason=FailureReason.DEADLINE, t_abandoned=t_send + 9.0,
                                             observed=attempt))


def _unsent(tenant: str, session: str, turn: int, t_send: float) -> TraceEvent:
    rid = f"{session}.{turn}"
    return TraceEvent(request_id=rid, session_id=session, turn_index=turn,
                      input=_input(tenant, rid, session, turn),
                      t_send=t_send, error="circuit breaker tripped; not sent",
                      failure=RequestFailure(reason=FailureReason.CIRCUIT_BREAKER, t_abandoned=t_send,
                                             observed=RequestAttempt()))


def drive_trace(*, failures: bool = True, tenants: list[Tenant] | None = None) -> Trace:
    """Events in completion order, interleaved across tenants and sessions. With
    *failures*, one chat request is abandoned at the deadline after a partial stream and one
    agent turn is never sent. With *tenants*, only those tenants' events."""
    events = [
        _answered("chat", "s0", 0, 10.0, streamed=True, planned=9.98, nan_at=1),
        _answered("agent", "a0", 0, 10.1, streamed=False, planned=10.0),
        _answered("chat", "s1", 0, 10.2, streamed=True, planned=10.2),
        _answered("plain", "p0", 0, 10.25, streamed=False),
        _answered("agent", "a1", 0, 10.3, streamed=False, planned=10.3),
        _answered("chat", "s0", 1, 11.0, streamed=True, n=7),
        _answered("agent", "a0", 1, 11.8, streamed=False),
        _answered("chat", "s1", 1, 12.0, streamed=True, n=1),
        _answered("agent", "a1", 1, 12.4, streamed=False),
        _answered("plain", "p1", 0, 12.5, streamed=False),
        _answered("agent", "a0", 2, 13.9, streamed=False),
    ]
    if failures:
        events.insert(6, _abandoned("chat", "s2", 0, 10.6))
        events.append(_unsent("agent", "a1", 2, 14.0))
    else:
        events.append(_answered("chat", "s2", 0, 10.6, streamed=True, planned=10.6))
        events.append(_answered("agent", "a1", 2, 14.0, streamed=False))
    if tenants is not None:
        keep = {t.id for t in tenants}
        events = [ev for ev in events if ev.tenant_id in keep]
    return Trace(events=events)


def passing_verdict(trace: Trace) -> Verdict:
    """A run-level verdict that passes every event, keyed as the runner keys it."""
    return Verdict(passed=True, per_event={ev.event_key: True for ev in trace.events})
