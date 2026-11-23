"""L0 -- TraceEvent + Trace.

A Trace is the record of what happened during a workload run. Owned by the
Driver (only thing with the wall clock). Read by Verifier and Scorer.

All timings are seconds-since-some-epoch from the harness's monotonic clock.

A request that succeeded is recorded by ``ArtifactResponse``; one that failed by
``RequestFailure``, built from the ``RequestAttempt`` the transport wrote as the
request was in flight. The two are disjoint: exactly one is set on an event.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any
from pydantic import BaseModel, ConfigDict, Field

from hls.core.payloads import Envelope, TokenLogprob


@dataclass(frozen=True)
class ArtifactResponse:
    """What the artifact returned (content) + what the harness observed (timing).

    `output` is the decoded Envelope. The timing fields are harness-observed and
    modality-agnostic -- a "packet" is one streamed output unit (a token for text, a
    frame for audio). Both are None on the unary path."""
    output: "Envelope"
    t_first_packet: float | None = None
    packet_timestamps: list[float] | None = None
    #: Harness time at which the reply's product was complete, when the stream went on
    #: afterwards carrying something that is not product. A protocol sets this only
    #: when its contract puts such a frame at the end: multimodal_tts sends the codec
    #: codes its correctness gate teacher-forces after the last audio frame, and the
    #: contract states that those codes are not latency. The driver stamps
    #: ``TraceEvent.t_recv`` from it, so a trailing frame cannot inflate a request's
    #: measured job completion time. None on every protocol whose stream ends with its
    #: last product frame, and the driver then stamps the moment the stream closed.
    t_complete: float | None = None


class FailureReason(str, Enum):
    """Why the harness stopped waiting on a request.

    Assigned from what the harness observed, never parsed out of the fault text: the
    driver knows its own two cases, and the transport's three are told apart by
    whether an HTTP status came back and by the fault's own type."""

    #: The artifact answered with an HTTP error status.
    SERVER_ERROR = "server_error"
    #: No usable HTTP exchange: connection refused, reset, or dropped mid-stream.
    TRANSPORT = "transport"
    #: Bytes arrived but did not form a conformant completion -- a stream that ended
    #: without a finish_reason, a missing token_ids array, a non-JSON body.
    PROTOCOL = "protocol"
    #: The stream delivered tokens and then none for longer than the task allows
    #: between two (``AgentSpec.stall_timeout_sec``). Not TRANSPORT: the connection
    #: is still open and the server has stopped decoding.
    STALLED = "stalled"
    #: Still in flight when the drive deadline passed, so the harness abandoned it.
    DEADLINE = "deadline"
    #: The consecutive-fault circuit-breaker ended this request: either it had already
    #: tripped when the request came up, so nothing was sent, or it tripped while the
    #: request was outstanding and the harness stopped waiting. The fault string says
    #: which; both mean the drive gave up rather than the request failing on its own.
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass(frozen=True)
class PartialOutput:
    """The output a failed request had produced before the harness gave up on it.

    Decoded by the same wire protocol that decodes a complete response
    (``WireProtocol.decode_partial``), so the ids and text here mean what they mean on
    a successful event. Unvalidated by construction: this is an incomplete response,
    so a stream that broke contributes whatever it managed to send. The token count is
    ``len(token_ids)``."""
    #: Ids of the tokens whose frames arrived, in emission order. Empty when the
    #: stream carried none.
    token_ids: list[int]
    #: The text those frames carried, concatenated.
    text: str
    #: The last frame's finish_reason, if any frame reported one. A stream can report
    #: one and still fail -- a server that finishes a reply and then breaks the
    #: connection before ``[DONE]`` is a different fault from one that never finished.
    finish_reason: str | None
    #: The per-token report those frames carried, or ``None`` where none arrived.
    #: Read only by a drive whose scheduled stop cut the request, which records the
    #: reply on its event's response; never written as part of a failure.
    logprobs: Annotated[list[TokenLogprob] | None, Field(exclude=True)] = None
    #: Arrival time of each token above, one per id, read and kept out of a failure
    #: as ``logprobs`` is.
    packet_timestamps: Annotated[list[float], Field(exclude=True)] = field(default_factory=list)


@dataclass
class RequestAttempt:
    """The harness's running record of one in-flight request.

    Mutable, and passed into ``ArtifactClient.submit`` rather than returned from it,
    because it is the only thing that outlives a request the harness abandons: when
    the drive deadline cancels a submit, everything the transport had collected dies
    with the coroutine's frame. The driver creates one per request, the transport
    writes to it as frames land, and it is read only to build a ``RequestFailure``. A
    successful request's is discarded."""

    #: Frames received so far. Zero means the server sent nothing back at all.
    n_frames: int = 0
    #: Harness time at the first frame's arrival, on the monotonic clock that stamps
    #: ``TraceEvent.t_send``. ``None`` while nothing has arrived. Unlike
    #: ``ArtifactResponse.t_first_packet`` this counts any frame, token-bearing or
    #: not: on a failed request the question is whether the server responded at all.
    t_first_frame: float | None = None
    #: The HTTP status when the artifact answered with an error one, else ``None``. A
    #: served rejection is exactly the case where this is set, which is how
    #: ``FailureReason.SERVER_ERROR`` is told from ``TRANSPORT``.
    status_code: int | None = None
    #: What the incomplete response decoded to, filled by the transport on its way out
    #: of a failed submit. ``None`` when nothing decodable arrived.
    partial: PartialOutput | None = None

    def note_frame(self, t: float) -> None:
        """Record one frame's arrival at harness time ``t``."""
        if self.t_first_frame is None:
            self.t_first_frame = t
        self.n_frames += 1


@dataclass(frozen=True)
class RequestFailure:
    """Everything the harness knows about a failed request, beyond the fault string
    already in ``TraceEvent.error``.

    One failed request zeroes the run's score and skips the correctness gate, so the
    trace is the only surviving evidence for why a run scored nothing. This record is
    what separates a server that refused instantly from one that streamed fifteen
    thousand tokens and then stalled -- cases whose fault strings can be identical.

    ``observed`` is final once this exists: the driver builds the failure only after
    it has stopped waiting, so nothing writes to the attempt afterwards."""
    #: Which class of failure this was.
    reason: FailureReason
    #: Harness time when the harness stopped waiting, comparable to
    #: ``TraceEvent.t_send``. ``t_recv`` stays ``None`` on a failure so the event is
    #: not in the latency sample; the time it took to fail goes here instead.
    t_abandoned: float
    #: The in-flight observations, as the transport left them.
    observed: RequestAttempt


#: The most characters of a dropped session's reason the trace keeps.
DROPPED_REASON_MAX_CHARS = 1000


class DroppedSession(BaseModel):
    """A session a trajectory drive dropped because its sandbox died after it had started
    (``hls.core.faults.SandboxGone``). The turns it served are in the trace as answered
    events; nothing is recorded for the turns it never reached, so this is the trace's only
    account of why the session is short."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The session's id, as its events carry it.
    session_id: str
    #: The tenant the session belongs to.
    tenant_id: str
    #: The fault's one-line message, cut to ``DROPPED_REASON_MAX_CHARS``.
    reason: str = Field(max_length=DROPPED_REASON_MAX_CHARS)


def event_key(tenant_id: str | None, request_id: str) -> str:
    """The trace-global identity of one served request: ``"<tenant>:<request_id>"``.

    ``request_id`` is unique only within a tenant (two tenants over the same dataset
    renumber their rows from 0), so every map that spans tenants keys on this form.
    Defined once so a trace event and a calibration item built from the same request
    produce the same key; ``Tenant.id`` excludes ``':'`` (``Tenant._safe_id``), so the
    split point is unambiguous."""
    return f"{tenant_id}:{request_id}"


class TraceEvent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    session_id: str | None = None
    turn_index: int = 0

    input: Envelope
    response: ArtifactResponse | None = None   # None if artifact errored
    ground_truth: Any | None = None     # set by Driver from DatasetItem; opaque to artifact

    t_send: float                       # harness time when Driver sent the request
    t_recv: float | None = None         # harness time when the response was received
    t_planned_send: float | None = None  # the open-loop schedule's intended send time

    error: str | None = None
    #: How this request failed, and how far it got before it did. Set exactly when
    #: ``error`` is (``response`` is then ``None``). ``error`` alone says a request
    #: failed; this says whether the server answered at all, what partial output
    #: arrived, and when the harness gave up -- the run's only record of that once the
    #: containers are gone.
    failure: RequestFailure | None = None
    extras: dict[str, Any] = Field(default_factory=dict)

    @property
    def tenant_id(self) -> str | None:
        """The tenant that owns this event, stamped into the request's input
        metadata by the Driver (single-tenant tasks stamp the synthetic ``"t0"``)."""
        return self.input.metadata.get("tenant_id")

    @property
    def event_key(self) -> str:
        """Trace-global event identity for cross-tenant aggregation: the module-level
        ``event_key`` of this event's stamped tenant and its ``request_id``. The
        run-level ``Verdict.per_event``, the Scorer's credit map and the pooled
        verifier's maps all key on it."""
        return event_key(self.tenant_id, self.request_id)


class Trace(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    events: list[TraceEvent] = Field(default_factory=list)

    def by_session(self) -> dict[str, list[TraceEvent]]:
        out: dict[str, list[TraceEvent]] = {}
        for ev in self.events:
            sid = ev.session_id or ev.request_id
            out.setdefault(sid, []).append(ev)
        return out

    def by_tenant(self) -> dict[str | None, list[TraceEvent]]:
        out: dict[str | None, list[TraceEvent]] = {}
        for ev in self.events:
            out.setdefault(ev.tenant_id, []).append(ev)
        return out

    def events_for(self, tenant_id: str) -> list[TraceEvent]:
        """Return the event slice for ``tenant_id``.

        A plain total lookup: driver output always stamps ``tenant_id`` (the single
        tenant gets ``"t0"`` -- ``_session`` stamps ``HarnessContext(tenant_id=
        tenant.id)`` via ``_stamped``), so ``by_tenant()`` keys are exactly the
        stamped tenant ids and an unknown id yields an empty slice. (Test doubles
        must likewise stamp ``tenant_id``; there is no None fallback.)
        """
        return self.by_tenant().get(tenant_id, [])

    def latencies_sec(self) -> list[float]:
        return [
            ev.t_recv - ev.t_send
            for ev in self.events
            if ev.t_recv is not None
        ]

    def scheduler_slack_sec(self) -> list[float]:
        """Per-event t_send - t_planned_send: did the open-loop schedule actually deliver?"""
        return [
            e.t_send - e.t_planned_send
            for e in self.events
            if e.t_planned_send is not None
        ]

    def ttft_sec(self) -> list[float]:
        """Per-event time-to-first-token = t_first_packet - t_send (seconds).
        Only streamed responses (those carrying t_first_packet) are included."""
        return [
            ev.response.t_first_packet - ev.t_send
            for ev in self.events
            if ev.response is not None and ev.response.t_first_packet is not None
        ]

    def tpot_sec(self) -> list[float]:
        """Per-event time-per-output-token = (last_ts - first_ts)/(n-1), from
        packet_timestamps. Events with fewer than 2 tokens are excluded."""
        out: list[float] = []
        for ev in self.events:
            resp = ev.response
            if resp is None or not resp.packet_timestamps:
                continue
            ts = resp.packet_timestamps
            if len(ts) < 2:
                continue
            out.append((ts[-1] - ts[0]) / (len(ts) - 1))
        return out

    def duration_sec(self) -> float:
        if not self.events:
            return 0.0
        t0 = min(ev.t_send for ev in self.events)
        t1 = max((ev.t_recv or ev.t_send) for ev in self.events)
        return max(t1 - t0, 1e-9)

    def t_end(self) -> float:
        """Harness time of the last thing that happened in the drive.

        The max over every timestamp any event carries: when it was sent, when its
        response came back, when its last packet arrived, and when the harness gave up
        on it. A request that never answered has no ``t_recv``, so its abandonment time
        is the only record of how long it ran; taking the max over all four keeps a
        stalled request inside the window rather than off the end of it. Callers that
        need a censoring time for a request with no measured latency use this.
        0.0 on an empty trace."""
        stamps: list[float] = []
        for ev in self.events:
            stamps.append(ev.t_send)
            if ev.t_recv is not None:
                stamps.append(ev.t_recv)
            if ev.response is not None and ev.response.packet_timestamps:
                stamps.append(ev.response.packet_timestamps[-1])
            if ev.failure is not None:
                stamps.append(ev.failure.t_abandoned)
        return max(stamps) if stamps else 0.0
