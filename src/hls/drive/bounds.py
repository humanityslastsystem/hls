"""What one drive of a task may make the harness hold: its trace on disk, and each reply as it is
read.

The drive role writes each reply into the drive's trace as soon as it has been received
(``hls.drive.stream``), so the harness holds no reply past its own request; what grows with the
drive is the trace on the node's disk, and every reader of it after. Each task declares the cap
on that trace (``DriveBounds.trace_cap_bytes``), and the drive role fails, as the contestant's
request, every reply that would carry the trace past it (``hls.drive.replies``):

- the cap counts the JSON of the trace's events as written into its chunks, before
  compression (the summary beside them holds no more), every reply received but not yet
  written at the most trace its protocol lets one byte of it become
  (``WireProtocol.trace_growth``), and every request sent and not yet written at the JSON its
  event takes beside its reply: its envelope's, measured as the trace writes it, and the
  event's own fields at their longest (``request_trace_bytes``), so the trace never passes it on
  those replies and requests;
- a reply is still held whole while it is read and decoded, so no reply may carry more than
  ``REPLY_MAX_BYTES``.

A task derives its cap from its published drives by one rule (``trace_cap``), citing the
figures it is derived from, unless verifying a trace that large would not fit the verify role's
memory: each task declares what its verify role holds (``VerifyPeak``), and its cap is
the largest whole GiB at which that fits, when that is less (``verify_cap``, ``bounds_for``).
The verify role reads the trace from the store a pass at a time, but the task's verifiers keep
what they grade, so what it holds grows with the trace. It also declares the most bytes of requests its trace holds beside
the replies (``DriveBounds.trace_request_bytes``): once the cap is reached the drive goes on,
recording each further turn's request and its failure, which those bytes bound, with every
event's identifiers, ground truth and extras, its task's own, which no request's charge counts
(``REQUEST_EVENT_BYTES``). The trace's room on the node's disk (``trace_disk_bytes``) and its
readers are sized from the two together.

No task override may set either (``hls.taskref.FIXED_TASK_FIELDS``). A task that declares none
cannot be evaluated: the planner refuses to plan its evaluation. No
drive-wide byte budget holds the replies."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hls.artifact.artifact_client import trace_growth
from hls.core.payloads import Envelope
from hls.core.trace import ArtifactResponse, FailureReason, PartialOutput, RequestAttempt, RequestFailure, TraceEvent
from hls.trace.store import event_line_bytes, input_json_bytes, room_bytes

if TYPE_CHECKING:
    from hls.spec.contracts import ArtifactContract
    from hls.spec.taskspec import TaskSpec

_GiB = 2**30

#: The most bytes one reply may carry: a reply is held whole while the client reads and decodes
#: it. A reply at a 32,768-token cap with logprobs is about 55 MB.
REPLY_MAX_BYTES = 256 * 2**20
#: How many times its largest published trace a task's cap is at least, unless its
#: verification would not fit (``verify_cap``).
TRACE_MARGIN = 4
#: The memory the verify role has: the reference job's 180 GiB less the runner's
#: default reserve of 4 GiB, as ``hls.pipeline.pipelines.reference_shape`` gives it
#: (``REFERENCE_SHAPE``, ``hls.settings.RunSettings.runner_memory_gib``).
VERIFY_MEMORY_BYTES = (180 - 4) * _GiB
#: The share of ``VERIFY_MEMORY_BYTES`` a task's verify peak may reach at its cap. A tenth is
#: kept as a margin over the declared peaks.
VERIFY_SHARE = 0.9


class VerifyPeak(BaseModel):
    """What a task's verify role holds at once while it grades a drive's trace, as the task
    declares it: *base_bytes* whatever the trace, and *bytes_per_byte* besides for each byte of
    the trace's event JSON."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Bytes held whatever the trace's size: the interpreter, the reference engines' host memory
    #: and what the task's workload bounds of its verifiers' state.
    base_bytes: int = Field(ge=0)
    #: Bytes held besides for each byte of the trace's event JSON: what the task's verifiers and
    #: the verification core keep of a trace whose replies take the worst measured shape the
    #: task's wire protocol accepts.
    bytes_per_byte: float = Field(gt=0)


def verify_peak_bytes(peak: VerifyPeak, event_bytes: int) -> int:
    """The verify role's peak, by *peak*, over a trace whose events hold *event_bytes* of JSON."""
    return peak.base_bytes + math.ceil(peak.bytes_per_byte * event_bytes)


def _verify_budget_bytes(verify_memory_bytes: int = VERIFY_MEMORY_BYTES) -> float:
    return VERIFY_SHARE * verify_memory_bytes


def verify_cap(peak: VerifyPeak, *, trace_request_bytes: int,
               verify_memory_bytes: int = VERIFY_MEMORY_BYTES) -> int:
    """The largest trace cap, in whole GiB, whose verification fits: the verify role's peak by
    *peak* over a trace at that cap and its *trace_request_bytes* of requests
    (``trace_event_bytes``) stays within ``VERIFY_SHARE`` of the role's *verify_memory_bytes*.
    Raise ValueError when not even 1 GiB fits."""
    budget = _verify_budget_bytes(verify_memory_bytes)
    events = (budget - peak.base_bytes) / peak.bytes_per_byte
    cap = math.floor((events - trace_request_bytes) / _GiB) * _GiB
    while cap >= _GiB and verify_peak_bytes(peak, cap + trace_request_bytes) > budget:
        cap -= _GiB
    if cap < _GiB:
        raise ValueError(f"the verify role cannot hold even a 1 GiB trace: {peak.base_bytes:,} bytes and "
                         f"{peak.bytes_per_byte} per trace byte against {budget:,.0f}")
    return cap


class DriveBounds(BaseModel):
    """What one drive of a task may make the harness hold, as the task declares it
    (``bounds_for``). Raise ValueError for a cap the verify role cannot hold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The most bytes of JSON the drive's trace may hold in its chunks before compression
    #: (``bounds_for``); past it every further reply fails as the contestant's request.
    trace_cap_bytes: int = Field(gt=0)
    #: The most bytes of requests the drive's trace holds beside the replies: the most a
    #: published drive held, with room. It bounds what the trace
    #: grows by once the cap is reached, when each further turn is still recorded with its
    #: request, and the task's own identifiers, ground truth and extras of the turns the gate held
    #: requests for (``REQUEST_EVENT_BYTES``).
    trace_request_bytes: int = Field(gt=0)
    #: What the task's verify role holds while it grades a trace (``VerifyPeak``), from which
    #: the role's memory is sized (``hls.pipeline.pipelines.reference_shape``).
    verify_peak: VerifyPeak
    #: Whether the cap is the most the verify role can hold (``verify_cap``) rather than the one
    #: the task's published drives give (``trace_cap``), which the brief then says.
    verify_limited: bool
    #: The memory the task's verify role has: ``VERIFY_MEMORY_BYTES`` unless the task's
    #: reference runs on more than the one-GPU reference job, as a reference sharded over a
    #: whole node does. Left out of the dump at the default, so the bounds of every task that
    #: declares none read as they did.
    verify_memory_bytes: int = Field(default=VERIFY_MEMORY_BYTES, gt=0,
                                     exclude_if=lambda n: n == VERIFY_MEMORY_BYTES)

    @model_validator(mode="after")
    def _verify_fits(self) -> DriveBounds:
        events = trace_event_bytes(self.trace_cap_bytes, self.trace_request_bytes)
        budget = _verify_budget_bytes(self.verify_memory_bytes)
        if verify_peak_bytes(self.verify_peak, events) > budget:
            raise ValueError(f"a {self.trace_cap_bytes:,}-byte trace cap is more than the verify role can hold: "
                             f"its peak would be {verify_peak_bytes(self.verify_peak, events):,} bytes against "
                             f"{budget:,.0f}")
        return self


def bounds_for(contract: ArtifactContract, *, trace_bytes: int, held_reply_bytes: int, trace_request_bytes: int,
               verify_peak: VerifyPeak, verify_memory_bytes: int = VERIFY_MEMORY_BYTES) -> DriveBounds:
    """The drive bounds of a task whose replies follow *contract*, whose largest published trace
    held *trace_bytes* of JSON and whose published drives held at most *held_reply_bytes* of
    replies at once (``trace_cap``), whose trace holds at most *trace_request_bytes* of requests,
    and whose verify role holds *verify_peak* of its *verify_memory_bytes*: the published rule's
    cap, or the largest the verify role can hold when that is less (``verify_cap``)."""
    ruled = trace_cap(contract, trace_bytes=trace_bytes, held_reply_bytes=held_reply_bytes)
    fits = verify_cap(verify_peak, trace_request_bytes=trace_request_bytes, verify_memory_bytes=verify_memory_bytes)
    return DriveBounds(trace_cap_bytes=min(ruled, fits), trace_request_bytes=trace_request_bytes,
                       verify_peak=verify_peak, verify_limited=fits < ruled, verify_memory_bytes=verify_memory_bytes)


def trace_cap(contract: ArtifactContract, *, trace_bytes: int, held_reply_bytes: int) -> int:
    """The trace cap of a task whose replies follow *contract*, whose largest published trace held
    *trace_bytes* of JSON, and whose published drives held at most *held_reply_bytes* of replies
    received but not yet written at once: the larger of ``TRACE_MARGIN`` times that trace, and that
    trace beside those reply bytes at the most trace the contract's protocol lets one of them
    become (``trace_growth``), so a drive like the published ones never reaches it however long its
    replies wait to be written. In whole GiB, and at least one."""
    held = trace_bytes + trace_growth(contract) * held_reply_bytes
    return max(1, math.ceil(max(TRACE_MARGIN * trace_bytes, held) / _GiB)) * _GiB


def _event_shell_bytes() -> int:
    """The bytes of the longest line an event takes in the trace's chunks beyond its input, with
    every value its reply and its task's dataset carry left empty and every number the harness
    writes at its longest, of an answered event and of a failed one with its failure's record,
    whichever is longer."""
    number, time = -2**63, -2.2250738585072014e-308
    request = Envelope(request_id="")
    answered = TraceEvent(request_id="", session_id="", turn_index=number, input=request,
                          response=ArtifactResponse(output=Envelope(request_id=""), t_first_packet=time,
                                                    t_complete=time),
                          t_send=time, t_recv=time, t_planned_send=time)
    attempt = RequestAttempt(n_frames=number, t_first_frame=time, status_code=number,
                             partial=PartialOutput(token_ids=[], text="", finish_reason=None))
    failed = TraceEvent(request_id="", session_id="", turn_index=number, input=request, error="",
                        failure=RequestFailure(reason=max(FailureReason, key=lambda r: len(r.value)),
                                               t_abandoned=time, observed=attempt),
                        t_send=time, t_planned_send=time)
    return max(event_line_bytes(answered), event_line_bytes(failed)) - input_json_bytes(request)


#: The bytes of trace JSON a request's event takes beyond its input and what its reply and its
#: task's dataset carry (``_event_shell_bytes``): the event's own fields, every number at its
#: longest (an int64's 20 characters, a float's 24), their names and their nulls. Its identifiers,
#: its ground truth and extras, and its fault's text are not in it. The identifiers and ground
#: truth are the task's dataset's, the same in every drive of it. The extras are whatever the
#: task's driver records beside the turn: deepresearch's depend on the drive's replies (its tool
#: results, the tool response text, a final reply's text). All three are bounded by the room the
#: task's request bytes leave beyond its requests (``DriveBounds.trace_request_bytes``). A
#: fault's text is the harness's own words or what the reply sent, held with the reply.
REQUEST_EVENT_BYTES = _event_shell_bytes()


def request_trace_bytes(request: Envelope) -> int:
    """The most bytes of trace JSON the event of *request* can hold beside its reply: the request
    as the event's input, its JSON measured as the trace writes it (``input_json_bytes``), and
    ``REQUEST_EVENT_BYTES``. The drive role holds a request at this from its send until its event
    is written (``hls.drive.replies``)."""
    return input_json_bytes(request) + REQUEST_EVENT_BYTES


def trace_event_bytes(trace_cap_bytes: int, trace_request_bytes: int) -> int:
    """The most bytes of JSON a drive's trace holds in its chunks, under a cap of
    *trace_cap_bytes* and *trace_request_bytes* of requests: the cap, which the reply gate keeps
    every reply, and every request sent before it closed, from passing, and the turns sent after
    it, each with its request and its failure, which the request bytes bound."""
    return trace_cap_bytes + trace_request_bytes


def trace_disk_bytes(trace_cap_bytes: int, trace_request_bytes: int) -> int:
    """The bytes a drive's trace takes on the node's disk: its room, which the stage makes before
    the server starts, at the most its chunks and summary take in gzip with its events' JSON at
    ``trace_event_bytes`` (``hls.trace.store.room_bytes``)."""
    return sum(room_bytes(trace_event_bytes(trace_cap_bytes, trace_request_bytes)).values())


class UnknownTask(ValueError):
    """A task declares no drive bounds (``TaskSpec.drive_bounds``). A ValueError, so a command
    planning for that task refuses in one line."""


def drive_bounds(task: TaskSpec) -> DriveBounds:
    """The drive bounds *task* declares. Raises ``UnknownTask`` for a task that declares none;
    there is no default."""
    if task.drive_bounds is None:
        raise UnknownTask(f"no drive bounds for task {task.id!r}: its TaskSpec declares no drive_bounds "
                          "(hls.drive.bounds.DriveBounds)")
    return task.drive_bounds
