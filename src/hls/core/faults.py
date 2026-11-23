"""L0 -- the harness failure taxonomy.

One axis: whose fault is it?

  ArtifactFault  -- the artifact's fault. Caught at the ArtifactClient->Driver
                   boundary and converted to a FAILED TraceEvent (response=None,
                   error set). The run stays VALID; a failed event is legitimate
                   signal about the contestant.
  HarnessFault   -- our fault (or the task's). Propagates out of the role that raised
                   it, so its stage ends as an infra fault the record names -- never
                   a silent 0 charged to the contestant.

Native exceptions (genuine bugs) also end the stage as infra; we type only the
recognized harness-input fault (DatasetError) so it stops surfacing as an opaque
IndexError.
"""
from __future__ import annotations


class ArtifactFault(Exception):
    """The artifact's fault: caught at the boundary -> failed TraceEvent.

    ``category`` is a stable short tag recorded in TraceEvent.error / diagnostics.

    ``extras`` is evidence about the reply that caused the fault, copied onto the
    failed event's own ``extras``. A failed event carries no response, so without it
    the only surviving record of what the server sent is the error string -- which is
    where a reader has to reconstruct the refusal from, and a string is a poor place
    to put a token id. Keys are the raiser's to choose and should be namespaced, since
    the driver stamps its own onto the same dict. A raiser that has nothing to add
    leaves it empty, which is every fault in the harness today."""
    category: str = "artifact"

    def __init__(self, *args: object, extras: "dict[str, object] | None" = None) -> None:
        super().__init__(*args)
        self.extras: dict[str, object] = dict(extras or {})


class ArtifactTransportError(ArtifactFault):
    """Transport-level: the artifact returned 5xx, refused/dropped the connection,
    or the request exceeded REQUEST_TIMEOUT_SEC. Recorded WITH timing (t_send kept)."""
    category = "transport"


class ArtifactStallError(ArtifactFault):
    """A stream that delivered tokens and then stopped delivering them, for longer
    than the task allows between tokens.

    Its own category because the connection is still open and the server is still
    answering -- a hang, not a dropped connection, and the two need different fixes.
    Only ever raised after a first token has arrived: the silence before one is
    prefill plus queueing, which the latency ratio already prices."""
    category = "stall"


class ArtifactProtocolError(ArtifactFault):
    """Parsed at the transport layer but not a conformant completion: missing
    choices[0]/message, missing finish_reason, a non-JSON body, or a stream that
    broke / truncated / was empty (strict streaming)."""
    category = "protocol"


class HarnessFault(Exception):
    """Our fault (or the task's): propagates -> the run is INVALID (status='error'),
    never silently scored 0.

    A name for the category, not the mechanism that produces it. ``Recorder.finalize``
    derives the terminal status as ``getattr(type(error), "rollout_status", "error")``,
    so 'error' is what *every* exception gets unless it declares itself a contestant
    fault, as ``SubmissionError`` does. Nothing catches this class and nothing tests
    for it.

    Worth saying because the sentence above reads as though subclassing this is what
    makes a run invalid. It is not: raising a bare ``RuntimeError`` from the same place
    invalidates identically. What subclassing buys is that a reader of the traceback,
    and of the code that raises, can see which kind of failure was meant -- and for
    ``SandboxGone`` specifically, a message prefix that run records can be searched on.
    Choose the type for the reader; do not convert a bare raise into one expecting the
    outcome to change."""


class DatasetError(HarnessFault):
    """A dataset item the harness cannot dispatch: empty turns, or
    turns/ground_truth_turns shorter than the dialog requires."""


#: Opening of every ``SandboxGone`` message. Two layers raise this fault -- the tool
#: runner when it cannot replace a container, a task's agent loop when enough commands
#: in a row never reach a shell -- and a reader looking through run records for a dead
#: sandbox should not have to know which of them noticed. A loop that may not import
#: ``hls`` spells the same string out and a test holds the two together.
SANDBOX_GONE = "sandbox gone:"


class SandboxGone(HarnessFault):
    """A session's container stopped executing commands after it had started, and the
    trajectory ended.

    Either because a replacement could not be launched inside the session's restart
    budget, or because a caller set a limit on consecutive faults and the session hit
    it. One fault for both, because they are one outcome. The trajectory driver drops
    the session and drives the rest (``hls.spec.drivers.TrajectorySessionDriver``):
    the turns it served stay in the trace, nothing is recorded for the turns it never
    reached, and the drop and this message are written into the trace's ``END.json``
    (``hls.trace.store.TraceEnd.dropped``). Raised anywhere else, it ends the role as a
    harness fault. A container that never started is not this fault but
    ``SandboxNotStarted``.

    Ours rather than the artifact's: the exec server runs on our side of the trust
    boundary and the artifact never touches it, so a dead sandbox is never one of the
    server's failed requests.

    A named subclass rather than a bare ``HarnessFault`` because a framework that
    names its endings stamps the exception's class name as the trajectory's exit
    status. A bare one would be recorded as a generic error and the record would never
    say the sandbox died, which is the same defect as a stop reason naming the wrong
    cause.

    The message is the record: the driver writes it as the drop's reason, and a role it
    escapes reports it as its failure. So it is one line of ``key=value``, opening with
    ``SANDBOX_GONE``, carrying what the raising layer knows and nothing it would have
    to be taught. Turn counts reach the trace through the session's own final event."""


class SandboxNotStarted(HarnessFault):
    """A session's container did not start within its bound, so the trajectory never had a
    sandbox.

    The harness's failure, a slow node or store, and never a dropped session: the driver
    does not catch it, so the drive fails and its stage records infrastructure. A drive
    that went on without the session would be scored on fewer sessions than it was asked
    for, for a reason that has nothing to do with the server or the model."""
