"""L1 -- Driver: turns a list of Tenants into a trace's events.

For each tenant the driver pulls items from the tenant's data_source, assigns
sessions to that tenant, and dispatches each item to the artifact, recording a
TraceEvent per turn. The drive role (``hls.drive.role``) builds the task's driver
whole with ``driver_for`` and drives it in its own process: ``Driver.run`` takes
the artifact's URL and the drive's split, tenants, contract and seed as
arguments, runs the two-level open-loop loop against the artifact and hands each
event, as it is recorded, to the caller's sink. A driver has no task-authored
config knobs.

Two timing levels (the open-loop model):
  - Session arrival is open-loop: arrival.schedule(n, seed) gives absolute
    session-start offsets, waited via _sleep_until regardless of completions.
  - inter_turn_delay is the one reactive point, waited after the prior turn's
    response, before sending a follow-up turn.

Single-turn items send one request; multi-turn items are dispatched with
cumulative history stitching: each turn appends the user message to a running
OpenAI-style conversation carried in metadata['messages'], sends the cumulative
conversation, records one TraceEvent (keyed by session_id + turn_index), then
appends the artifact's reply so the next turn carries full context. The
contestant server stays stateless: every request contains everything it needs.

Multi-tenant model
------------------
``run`` takes the drive's tenants (a list[Tenant]) as an argument. ``_drive``
captures one shared t0 anchor, then concurrently dispatches all tenants' sessions
against it.  Per-tenant:
  - items are loaded via ``tenant.data_source.dataset.items_at(mount_dir, split)``
    where mount_dir = ``/data/tenants/<id>`` for every tenant.
  - the arrival schedule is seeded with ``tenant_seed(seed, tenant.id)`` (the
    shared helper in ``hls.core.seeds``), ``seed`` being the drive's.
  - each event is stamped with ``tenant_id = tenant.id``.

A single-tenant task synthesizes ``Tenant(id="t0", ...)`` via
``TaskSpec.effective_tenants()`` and then travels the same path as any other: one
unit, mounted at ``/data/tenants/t0``, seeded by ``tenant_seed(seed, "t0")``.
"""

from __future__ import annotations
import asyncio
import gc
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol

from hls.core.engine import ResourceRequest
from hls.core.faults import (ArtifactFault, ArtifactProtocolError, ArtifactStallError, DatasetError,
                             SandboxGone)
from hls.core.model import TemplateArg
from hls.core.payloads import Envelope, TextPayload
from hls.core.seeds import tenant_seed
from hls.core.trace import (DROPPED_REASON_MAX_CHARS, ArtifactResponse, DroppedSession, FailureReason,
                            RequestAttempt, RequestFailure, TraceEvent)
from hls.spec import datasets
from hls.spec.agent import DriveWallClockReached
from hls.spec.workload import CUT_EXTRA, MAX_TOKEN_GAP_EXTRA, PREFILL_TOKENS_EXTRA

if TYPE_CHECKING:
    from hls.spec.agent import AgentFramework, ScheduledStop
    from hls.spec.contracts import ArtifactContract
    from hls.spec.tenant import Tenant
    from hls.spec.tools import ToolRunner
    from hls.spec.workload import InterTurnDelay

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harness wire types
# ---------------------------------------------------------------------------

def harness_now() -> float:
    return time.monotonic()


@dataclass(frozen=True)
class HarnessContext:
    tenant_id: str
    t_send: float
    #: The tenant's per-request output cap (``Workload.output_length_cap``), for an
    #: agentic framework that budgets a trajectory against it. The driver stamps this
    #: on every turn it sends, so a framework reading it here decides on the number
    #: the request actually carried rather than on the one the reply claimed about
    #: itself. ``None`` on the paths that stamp no cap.
    max_output_tokens: int | None = None
    #: The task's context window (``TaskSpec.max_context_len``), for an agentic framework
    #: that budgets a trajectory against it. Stamped by the trajectory driver from the task
    #: it was built for, so the window a trajectory is cut at is the one the workload
    #: fingerprint names. ``None`` when the task declares none, and on the paths that
    #: stamp none.
    max_context_len: int | None = None
    #: The model's chat-template pin (``ModelSpec.chat_template_kwargs``), for a framework
    #: that renders a prompt to count it. Stamped beside the window, from the same task.
    chat_template_kwargs: Mapping[str, TemplateArg] = field(default_factory=dict)


class ArtifactClient(Protocol):
    """The harness-side handle to a deployed artifact.

    A structural interface: anything with an async `submit(envelope) -> ArtifactResponse`
    can be driven. The concrete client comes from a deployment (deployment.py);
    the driver only depends on this method so it never sees the transport.

    ``attempt`` is the driver's per-request ledger; a transport that can observe a
    response arriving in pieces writes to it so a failed request leaves a diagnosable
    record. One that cannot (an in-process engine) ignores it.
    """

    async def submit(self, env: Envelope, *,
                     attempt: RequestAttempt | None = None) -> ArtifactResponse: ...


def _stamped(env: Envelope, ctx: HarnessContext, session_id: str, turn_idx: int,
             max_tokens: int | None = None) -> Envelope:
    """Return a copy of `env` with harness metadata stamped on it.

    ``max_tokens`` (when provided) is the workload's per-request output cap; the
    wire protocol forwards it into the request body so artifact and oracle run
    under identical decoding budgets (see OpenAIChatProtocol.encode_request)."""
    meta: dict[str, Any] = dict(env.metadata)
    meta.update({
        "tenant_id": ctx.tenant_id,
        "session_id": session_id,
        "turn_index": turn_idx,
    })
    if max_tokens is not None:
        meta["max_tokens"] = max_tokens
    return env.model_copy(update={"metadata": meta})


def request_messages(turn_env: Envelope, messages: list) -> list:
    """The conversation one request actually carries on the wire.

    An adapter may pre-render the full OpenAI conversation on the turn envelope's own
    ``metadata['messages']`` (MRCR's scripted history, BFCL's system template); that
    rendering wins over the drive loop's stitched ``messages``. Otherwise the stitched
    history is what is sent.

    The single definition of that rule, so the request's size and the request's bytes
    are always decided from the same list: ``stamp_request`` shapes the envelope with
    it, and ``Driver._session`` measures it against the tenant's context budget.
    Returns a snapshot (``list(...)``), so a later ``append`` by the drive loop cannot
    mutate an already-shaped request."""
    pre = turn_env.metadata.get("messages")
    return list(pre) if pre else list(messages)


def message_content(message: dict) -> str:
    """The text one chat message carries.

    A missing or null ``content`` is the empty string; any other value is stringified.
    The rule matters because a pre-rendered conversation comes straight from a dataset's
    own JSON, which does not promise every ``content`` is a string, and two readings of
    the same message would put one prompt on the wire and teacher-force another: under
    raw conditioning the reference reproduces the exact bytes the server saw, so
    ``hls.spec.wire`` and the reference engines derive them here rather than each their
    own way."""
    content = message.get("content")
    return "" if content is None else str(content)


#: The fields beyond role and content that a chat template reads off a message: the
#: model's own tool calls, the reasoning it wrote before them, and the id and tool
#: name a tool result answers with. A turn of an agentic conversation carries them,
#: and a request that dropped them would render a shorter prompt than the history it
#: was built from. Everything else a task hangs on a message stays out.
#:
#: The reasoning is spelled twice on purpose. A chat template reads whichever of the
#: two it was written against, and vLLM's chat parsing rebuilds every message from
#: ``reasoning`` alone, so a history that carried only ``reasoning_content`` would
#: reach a stock vLLM server without its reasoning.
#: The self-test the deepresearch build sandbox carries
#: (``/app/tasks/deepresearch/selftest.py``) repeats this list, because the sandbox has
#: no harness to import it from; a test holds this one to that copy.
CHAT_MESSAGE_FIELDS: tuple[str, ...] = ("reasoning", "reasoning_content", "tool_calls",
                                        "tool_call_id", "name")


def chat_message(m: dict) -> dict:
    """One history message as a chat template consumes it: role and content, then
    whichever of ``CHAT_MESSAGE_FIELDS`` the message carries.

    Content passes through as None when it is None, which is how the OpenAI schema
    spells an assistant turn that only calls a tool; any other non-string content is
    stringified. Shared with the reference engines, so the artifact and the oracle
    render the same conversation."""
    content = m["content"]
    out: dict = {"role": str(m["role"]),
                 "content": content if content is None or isinstance(content, str)
                 else str(content)}
    out.update({k: m[k] for k in CHAT_MESSAGE_FIELDS if k in m})
    return out


def messages_of(env: Envelope) -> list[dict]:
    """The chat messages one request carries: the driver-stitched history from
    ``env.metadata['messages']`` (multi-turn) when present, else the single latest
    user turn from the envelope's text payload. Distinct from ``request_messages``
    above, which picks between a turn's pre-rendered conversation and the drive
    loop's stitched one before either is sent; this reads whichever list that already
    settled on.

    A stitched entry missing a ``role`` or ``content`` key is dropped rather than
    crashing on it: a pre-rendered conversation comes straight from a dataset's own
    JSON, which does not promise every entry is well-formed. The wire protocols and
    the reference engines both build a request's messages through this one function,
    so a malformed entry is dropped identically on the wire and in the oracle that
    reproduces the same conversation, and neither ever scores or sends a message the
    other dropped.

    A surviving entry keeps the ``CHAT_MESSAGE_FIELDS`` it carries (``chat_message``).
    An agentic turn's tool calls and reasoning are part of the prompt the chat template
    renders, so a reading that kept only role and content would send one conversation
    and teacher-force a shorter one.

    A null ``content`` survives as null rather than flattening to the empty string,
    because that is how the OpenAI schema spells an assistant turn that only calls a
    tool, and the two render differently. Both the wire and the oracle read the message
    here, so they still agree on every position whatever a template makes of it. Raw
    conditioning is unaffected: it reads the prompt through ``message_content``, where a
    null is still the empty prompt."""
    stitched = env.metadata.get("messages")
    if isinstance(stitched, list) and stitched:
        return [chat_message(m) for m in stitched
                if isinstance(m, dict) and "role" in m and "content" in m]
    tp = env.first_of("text")
    if not isinstance(tp, TextPayload):
        raise TypeError(f"request carries no text payload to read a message from, got {type(tp)!r}")
    return [{"role": "user", "content": tp.text or ""}]


def history_chars(messages: list) -> int:
    """Total content length of a conversation, in characters -- the size a context
    budget (``Workload.history_char_budget``) is measured against.

    Characters, not tokens, because the driver never tokenizes; a task derives its
    budget from the serving envelope at a conservative chars-per-token floor. Reads
    content through ``message_content``, the same rule the wire protocols and the
    reference engines use, so a falsy-but-present value like ``0`` counts as one
    character rather than none."""
    return sum(len(message_content(m)) for m in messages)


def _over_context_budget(messages: list, char_budget: "int | None") -> bool:
    """Whether the conversation a request would carry exceeds the tenant's context
    budget. ``char_budget is None`` means the tenant declares no budget, so nothing
    ever exceeds it."""
    return char_budget is not None and history_chars(messages) > char_budget


def user_message(turn_env: Envelope) -> dict[str, str]:
    """The user chat message dict for one rendered turn Envelope: the turn's text
    payload text (empty string when the turn carries no text payload).

    The single definition of how a rendered turn becomes an OpenAI-style chat
    message, shared by the drive loop (``Driver._session``) and by the calibration
    preflight's probe shaping (``hls.calibrate``) so a probe is rendered the way real
    traffic is."""
    tp = turn_env.first_of("text")
    return {"role": "user", "content": tp.text if tp is not None else ""}


def stamp_request(turn_env: Envelope, messages: list[dict[str, str]], *,
                  tenant_id: str, session_id: str, turn_idx: int,
                  max_tokens: int | None) -> Envelope:
    """Shape one request Envelope exactly as the drive loop sends it on the wire.

    Stamps the harness metadata (tenant/adapter/session/turn + the workload's
    ``max_tokens``) onto a copy of the rendered turn and attaches the cumulative
    conversation as ``metadata['messages']``. Pure (no I/O) and the one
    request-shaping seam: the drive loop (``Driver._session``) and the calibration
    preflight's probe shaping (``hls.calibrate``) both go through it, so the same
    (turn, history, cap) always produces the same request bytes.

    ``messages`` is the full stitched history up to and including this turn's user
    message. What actually rides on the wire is ``request_messages(turn_env,
    messages)`` -- a pre-rendered conversation on the turn envelope wins over the
    stitched one -- resolved here rather than at the call sites so the live drive and
    the calibration renderer cannot diverge on it."""
    ctx = HarnessContext(tenant_id=tenant_id, t_send=0.0)
    stamped = _stamped(turn_env, ctx, session_id=session_id, turn_idx=turn_idx,
                       max_tokens=max_tokens)
    meta = dict(stamped.metadata)
    meta["messages"] = request_messages(turn_env, messages)
    return stamped.model_copy(update={"metadata": meta})


# ---------------------------------------------------------------------------
# Drive-time control: the fail-fast circuit-breaker
# ---------------------------------------------------------------------------

#: Default number of consecutive artifact faults that trips the circuit-breaker. A
#: healthy server never strings this many faults; a dead one (crashed engine)
#: reaches it and aborts the drive fast instead of firing thousands more requests
#: that would otherwise hang until the drive deadline.
DEFAULT_FAULT_ABORT_THRESHOLD = 50

#: How long the drive may go with requests outstanding and NOTHING coming back before the
#: breaker trips on the stall instead of on a fault run.
#:
#: The fault counter above only sees COMPLETED submits, so it is blind to the server that
#: accepts a connection and then never answers: nothing completes, the consecutive run
#: stays at zero, and the drive keeps firing until each request reaches the deadline grace
#: hours later.
#:
#: Fifteen minutes separates that from a merely slow server by a wide margin: a server
#: far under its pin still completes a request every few seconds.
DEFAULT_STALL_ABORT_SEC = 900.0

#: How many graces past its last scheduled arrival a drive may run before its hard wall
#: (see ``_Deadline``). Four is chosen to clear any healthy drive by a wide margin -- one
#: grace for the deadline anchor, one for a session's think time, one for the slowest turn
#: in flight, one spare -- while still bounding a pathological multi-turn drive well inside
#: the wall clock of the job running it.
_WALL_GRACES = 4

@dataclass(frozen=True)
class _Deadline:
    """The drive's completion deadline with a per-request floor: a request sent at
    ``t_send`` gets ``max(t - t_send, grace)`` to finish (monotonic clock, the same
    one stamping t_send/t_recv). A request sent early in an overloaded drive may
    wait far longer than any fixed cap; a follow-up turn sent late (think time,
    slow prior turns) still gets the full ``grace``. ``error`` is the fault string
    stamped on a miss: one constant string per drive, so the failure census groups
    all misses together.

    ``wall`` is the drive's own cut-off (``None`` = unbounded). It clamps the allowance
    from above, because the floor above re-arms the full grace for every turn a
    session sends and a session's turns are serial: without the clamp a slow
    trajectory workload has no bound on total drive time. The drive's own
    ``_DriveWall``, not a copy of its instant, so one drive has one wall.

    The clamp is what keeps a pathological multi-turn drive inside the wall clock of
    the job running it. A run killed by that outer clock yields no verdict at all,
    rather than the bounded zero the gated scorer already knows how to produce. On a
    single-turn workload the schedule's own wall never binds, since every arrival is
    scheduled at or before ``t - grace``; a ``drive_timeout_sec`` shorter than the last
    arrival plus ``grace`` does."""
    t: float
    grace: float
    error: str
    wall: "_DriveWall | None" = None

    def allowance(self, t_send: float) -> "tuple[float, str]":
        """How long a request sent at ``t_send`` may run, with the fault string for the
        bound that would cut it: the per-request floor, or the drive wall when the wall
        comes first. Never negative -- at or past the wall the request is already over,
        and 0.0 makes ``wait_for`` say so immediately.

        Measured from ``t_send`` rather than from ``_DriveWall.remaining()``, which
        counts from now: a request already running gets what is left of the wall from
        when it went out, never a fresh span that would outlast it.

        The error travels with the number so a request cut by the wall is not filed
        under the grace it never reached. Otherwise the failure census names a bound
        that had thousands of seconds left, and the next operator raises it."""
        allowed = max(self.t - t_send, self.grace)
        if self.wall is None:
            return allowed, self.error
        to_wall = max(self.wall.t - t_send, 0.0)
        return ((to_wall, WALL_ERROR) if to_wall < allowed
                else (allowed, self.error))


#: Error string stamped on a request that the circuit-breaker skipped (the fault
#: threshold was reached, so the request is not sent -- it is recorded as a fault so
#: the run scores low). Any non-empty ``error`` makes the event fail in verify. The
#: string states the measurement (a run of consecutive faults), never a diagnosis
#: like "server dead": an alive-but-overloaded server trips the breaker exactly like
#: a crashed one.
ABORTED_ERROR = "aborted: circuit-breaker tripped (consecutive request faults)"

#: Error string stamped on a request that was already in flight when the breaker
#: tripped. Distinct from ``ABORTED_ERROR`` because this one was sent: the failure
#: census should not merge "we never asked" with "we asked and stopped waiting", and
#: only the latter says anything about the server. Cancelling rather than waiting out
#: the allowance is what keeps a drive that has given up from running for another
#: ``grace`` seconds (``hls.pipeline.pipelines.REQUEST_TIMEOUT_SEC`` on a graded drive) before it can return.
ABANDONED_ERROR = "abandoned: circuit-breaker tripped while the request was outstanding"

#: Error string stamped on a request the drive wall cut off: the drive reached its
#: total time limit, so this turn and every remaining one are recorded as faults
#: without being sent. Distinct from the breaker's string so the failure census
#: separates a server that stopped answering from one that never kept up.
WALL_ERROR = "aborted: drive wall clock reached (total drive time limit)"


class _WallExpired(Exception):
    """Raised inside a drive loop when the drive wall cut the work in flight.

    Caught by whichever loop started that work, which records the unit as a fault.
    Never escapes a drive: an expired wall is a measured outcome, not a harness bug."""


class _TrajectoryAborted(Exception):
    """Raised to unwind a trajectory whose artifact fault has already been recorded.

    A framework hands the driver no way to end a trajectory mid-turn, so the fault is
    written to the trace and this is thrown to stop the remaining turns. Caught by the
    loop that started the trajectory; like ``_WallExpired`` it never escapes a drive."""


@dataclass(frozen=True)
class _DriveWall:
    """The drive's own clock bound, shared by both drive loops.

    ``request_timeout_sec`` bounds one request and re-arms on every turn a session
    sends; a session's turns are serial, so on a session or trajectory workload it
    bounds no total. This bounds the drive itself: past ``t`` no new work starts and
    nothing already in flight continues.

    Both ``Driver`` and ``TrajectorySessionDriver`` run their own loop -- scripted
    turns against a schedule, versus a framework choosing a trajectory's turns -- so
    the loops cannot merge. The bound is the part they must agree on, and it lives
    here rather than in either of them, because a wall implemented in one loop is
    silently absent from every task driven by the other."""

    t: float

    def remaining(self) -> float:
        """Seconds left before the wall. Never negative, so a caller past the wall
        gets 0.0 and ``wait_for`` reports it immediately."""
        return max(self.t - harness_now(), 0.0)

    @property
    def expired(self) -> bool:
        return harness_now() >= self.t

    async def cut(self, awaitable):
        """Await ``awaitable``, raising ``_WallExpired`` if the wall passes first."""
        try:
            return await asyncio.wait_for(awaitable, timeout=self.remaining())
        except asyncio.TimeoutError:
            raise _WallExpired from None


class _CutReached(Exception):
    """Raised inside the trajectory loop when the scheduled stop's cut cancelled a
    trajectory still running.

    Not ``_WallExpired``: the wall records the remaining work as faults, the cut records
    none. Caught by the loop that started the trajectory; never escapes a drive."""


@dataclass(frozen=True)
class _StopWindow:
    """A task's ``ScheduledStop`` anchored on the drive's own start.

    Past ``t_stop`` no request is issued and each trajectory ends cleanly; at ``t_cut``
    whatever still runs is cancelled, which cuts its open stream and unwinds the
    framework's cleanup, tearing down the session's sandbox. Neither records a fault.
    The drive's wall stays around both (``_DriveWall``)."""

    #: Harness time after which no request is issued.
    t_stop: float
    #: Harness time every open stream is cut.
    t_cut: float

    @property
    def past_stop(self) -> bool:
        return harness_now() >= self.t_stop

    @property
    def cutting(self) -> bool:
        """Whether the cut instant has passed.

        Read from the clock rather than from a flag the cut sets: the cancelled frame is
        what records itself, and no point between ``wait_for`` deciding to cancel and the
        cancellation landing belongs to either side."""
        return harness_now() >= self.t_cut

    async def run_until_cut(self, awaitable):
        """Await ``awaitable``, raising ``_CutReached`` if the cut passes first."""
        try:
            return await asyncio.wait_for(awaitable, timeout=max(self.t_cut - harness_now(), 0.0))
        except asyncio.TimeoutError:
            raise _CutReached from None


#: A request id of the form ``<session>.<turn>``, the form the drive loop and the trajectory
#: frameworks give their requests.
_TURN_SUFFIX = re.compile(r"^.*\.(\d+)$")


def _turn_index(env: Envelope) -> int | None:
    """The turn a request names: its metadata's ``turn_index``, else the turn its id ends in
    (``<session>.<turn>``), else None. A framework need not stamp ``turn_index``, but every
    one in the tree numbers its request ids this way."""
    turn = env.metadata.get("turn_index")
    if isinstance(turn, int) and not isinstance(turn, bool):
        return turn
    m = _TURN_SUFFIX.match(env.request_id)
    return int(m.group(1)) if m else None


def max_token_gap_sec(packet_timestamps: "list[float] | None") -> float:
    """The longest a reply went between two of its tokens, in seconds, from their arrival
    times (``ArtifactResponse.packet_timestamps``): 0.0 under two tokens. Tokens of one frame
    share its time, so this is the longest gap between frames that carried tokens."""
    ts = packet_timestamps or []
    return max((b - a for a, b in zip(ts, ts[1:])), default=0.0)


@dataclass(frozen=True)
class _Sent:
    """What one send produced, as the trace event will record it.

    Either ``response`` is set (and ``error``/``failure`` are not) or the reverse; a
    named record rather than a tuple because the two halves have to stay consistent and
    the caller copies all five fields onto the event."""
    #: The decoded response, or ``None`` when the request failed.
    response: ArtifactResponse | None
    #: The verbatim fault string, or ``None`` on success. What the failure census in
    #: verify groups on.
    error: str | None
    #: How the request failed and how far it got, or ``None`` on success.
    failure: RequestFailure | None
    #: Harness time the request was sent (for a never-sent request, when it would have
    #: been).
    t_send: float
    #: Harness time the response arrived; ``None`` on failure, which is what keeps a
    #: failed request out of the latency sample.
    t_recv: float | None


def _fault_reason(fault: ArtifactFault, attempt: RequestAttempt) -> FailureReason:
    """Which failure class an ArtifactFault belongs to.

    Decided from what the harness observed and from the fault's own type, never by
    reading the fault's text. A stall is asked about first, since it is the one fault
    on a connection still open and answering. A served rejection is exactly the case
    where a status code came back; a protocol fault means bytes arrived but did not
    form a completion; anything else transport-level is a connection that never
    delivered one."""
    if isinstance(fault, ArtifactStallError):
        return FailureReason.STALLED
    if attempt.status_code is not None:
        return FailureReason.SERVER_ERROR
    if isinstance(fault, ArtifactProtocolError):
        return FailureReason.PROTOCOL
    return FailureReason.TRANSPORT


def _failed(error: str, reason: FailureReason, attempt: RequestAttempt,
            t_send: float) -> _Sent:
    """The outcome of a request the harness gave up on, stamped with the moment it did."""
    return _Sent(response=None, error=error,
                 failure=RequestFailure(reason=reason, t_abandoned=harness_now(),
                                        observed=attempt),
                 t_send=t_send, t_recv=None)


def _collapse(window: "asyncio.Timeout") -> bool:
    """Bring *window*'s deadline forward to now, so whatever it guards gives up at the
    next opportunity. True when that actually shortened it.

    False means the window was already at or past its own deadline, so the request it
    guards is expiring on the allowance it was given and the trip took nothing from
    it. The caller uses that to decide how the request is recorded: a natural deadline
    miss that merely happened to be outstanding when the breaker tripped is not an
    abandonment, and calling it one would report a request as cut short when it ran
    its full course.

    Never raises: the window may have been exited between the trip deciding to
    collapse it and getting here, and a drive already abandoning itself must not fail
    on the way out."""
    try:
        now = asyncio.get_running_loop().time()
        when = window.when()
        if when is not None and when <= now:
            return False
        window.reschedule(now)
        return True
    except RuntimeError as exc:                  # wrong-state window, or no loop
        # Logged rather than swallowed: a collapse that silently no-ops puts the drive
        # back to waiting out the whole allowance, and driver.log is the only place
        # that failure is visible.
        log.debug("driver: could not collapse an in-flight window: %s", exc)
        return False


#: Log every Nth fault (plus the first, plus the trip), so driver.log says when faults
#: started and what they were in one line per hundred faults rather than one per fault.
_FAULT_LOG_EVERY = 100


def _deadline_anchor(sorted_starts: "list[float]") -> float:
    """The scheduled arrival the drive deadline anchors on: the whole schedule's last
    arrival, since the drive waits on every one of them. A follow-up turn scheduled past it
    still gets the per-request grace floor, so no request is ever granted less than
    ``grace``. ``sorted_starts`` is the merged, ascending arrival offsets across all tenants
    and must be non-empty."""
    return sorted_starts[-1]


def rampdown_turn_budget(scheduled: list, seed: int) -> "dict[tuple[str, int], int]":
    """How many leading turns of each session exist, for tenants that opt into
    ramp-down trimming (``Workload.trim_rampdown``) -- decided at draw time against
    the workload's own nominal timeline, never against the wall clock.

    The nominal timeline assumes zero service time: session ``s`` of a trimming
    tenant nominally dispatches follow-up turn ``k`` at ``starts[s] +
    sum(think_1..think_k)``, and the turn exists iff that instant falls before the
    merged schedules' last session arrival. Turn 0 always exists.

    Deciding ramp-down at serve time instead (drop a follow-up once the wall clock
    passes the cutoff) would make the request set a function of the contestant's
    latency: a slower stack would shed its deepest, longest-context turns, shifting
    the graded token mixture with the very performance being scored. So the wall
    clock decides only when a turn dispatches, never whether it exists; the drive
    deadline remains the operational bound on a run that cannot keep up.

    A tenant's context budget (``Workload.history_char_budget``) can end a session
    earlier still, but at serve time, on the length of the replies generated rather
    than on latency -- so it does not reintroduce this coupling.

    ``scheduled`` is ``[(tenant, items, starts)]`` as ``_drive`` builds it, with
    ``starts`` schedule-relative (pre-t0). The cutoff is the last arrival. Returns
    ``(tenant_id, session_index) -> kept turn count``, with entries only for
    trimming tenants (absent = serve every turn)."""
    starts_all = [t for _, _, starts in scheduled for t in starts]
    if not starts_all:
        return {}
    t_stop = max(starts_all)
    budget: dict[tuple[str, int], int] = {}
    for tenant, items, starts in scheduled:
        if not tenant.workload.trim_rampdown:
            continue
        delay = tenant.workload.inter_turn_delay
        t_seed = tenant_seed(seed, tenant.id)
        for s, item in enumerate(items):
            kept, at = len(item.turns), starts[s]
            for k in range(1, len(item.turns)):
                at += delay.delay(k, seed=(t_seed, s))
                if at >= t_stop:
                    kept = k
                    break
            budget[(tenant.id, s)] = kept
    return budget


def load_mounted_sessions(tenants: list, split) -> list[tuple]:
    """Materialize every tenant's sessions from the dataset mount, as
    ``[(tenant, items)]``.

    Read before the drive anchors its clock, so a one-time dataset load is never
    charged against the open-loop arrival schedule. A tenant with no data source, or
    whose split is empty, contributes no sessions (an empty split is not an
    error here -- the verify role refuses an empty trace, a ``HarnessFault``).

    Every tenant reads ``/data/tenants/<id>/<split>``, which is where provisioning
    writes it -- one convention, so a one-tenant task and a five-tenant one differ in
    nothing but the number of units.

    Module-level rather than a Driver method because it is the half of the drive that
    is not the clock: ``hls.calibrate`` materializes its sessions on the host (from the
    provisioned host dirs, since a reference container has no dataset mount) and ships
    them to the engine, which then runs the clock half alone."""
    loaded: list[tuple] = []
    for tenant in tenants:
        if tenant.data_source is None:
            continue
        dataset = tenant.data_source.dataset
        items = list(dataset.items_at(datasets.MOUNT_ROOT / "tenants" / tenant.id, split))
        if items:
            loaded.append((tenant, items))
    return loaded


def remove_listed_rows(loaded: list[tuple]) -> list[tuple]:
    """Drop each tenant's ``DataSource.removed_rows`` from ``[(tenant, items)]``.

    A listed row takes its whole session; the rest keep their order. This is the one
    place the removal list acts, and it runs before the arrival schedule is drawn
    (``plan_sessions``), so the workload a drive offers is the list of sessions that
    remain and nothing about a removed row reaches the schedule. A tenant left with no
    sessions drops out, as one whose split was empty does at load time."""
    kept: list[tuple] = []
    for tenant, items in loaded:
        removed = set(tenant.data_source.removed_rows) if tenant.data_source else set()
        remaining = [item for item in items if item.request_id not in removed]
        if removed:
            log.info("driver: tenant %s -- %d of %d sessions remain after the removal "
                     "list", tenant.id, len(remaining), len(items))
        if remaining:
            kept.append((tenant, remaining))
    return kept


def plan_sessions(loaded: list[tuple], drive_seed: int) -> tuple[list[tuple], dict]:
    """Draw the drive's schedule over ``[(tenant, items)]``, the sessions the drive
    serves (``remove_listed_rows`` has already run).

    Returns ``([(tenant, items, starts)], turn_budget)``: per tenant its sessions and
    every session's schedule-relative arrival, plus the ramp-down turn budget of
    ``rampdown_turn_budget``. A session's arrival and think-time stream are keyed by
    its index in this list. Both a graded run and a calibration serving plan through
    here (``Driver._drive_sessions``)."""
    scheduled = [
        (tenant, items,
         tenant.workload.arrival.schedule(len(items), tenant_seed(drive_seed, tenant.id)))
        for tenant, items in loaded]
    return scheduled, rampdown_turn_budget(scheduled, drive_seed)


def _subsample_sessions(loaded: list[tuple], max_sessions: int | None) -> list[tuple]:
    """Trim ``[(tenant, items)]`` to about ``max_sessions`` sessions in total, keeping
    each tenant's share of the traffic.

    Every tenant keeps the same fraction of its sessions (at least one, so no tenant
    drops out of the mix) and keeps them as a leading prefix. Applied before the
    arrival schedules are drawn, so each tenant's process is generated for its own
    trimmed count: the offered rate per tenant, and hence the traffic mix and the
    total offered load, are exactly what the untrimmed drive would have offered --
    only the span is shorter. The span cannot go below the workload's own session
    structure (a session still runs all its turns, each after its think-time).

    ``None``, or a bound at least as large as the population, returns ``loaded``
    unchanged. Rounding is to the nearest session per tenant, so the realized total
    may differ from the bound by a session per tenant; the bound is a size knob, not
    a contract."""
    if max_sessions is None:
        return loaded
    total = sum(len(items) for _, items in loaded)
    if max_sessions >= total:
        return loaded
    fraction = max_sessions / total
    trimmed = [(tenant, items[:max(1, round(len(items) * fraction))])
               for tenant, items in loaded]
    log.info("driver: session subsample -- %d of %d sessions (%s)",
             sum(len(items) for _, items in trimmed), total,
             ", ".join(f"{t.id}={len(i)}" for t, i in trimmed))
    return trimmed


class _DriveState:
    """Shared per-drive control block: the consecutive-fault circuit-breaker.

    One instance is created per ``_drive`` call and shared (read + mutated) by every
    concurrent session. Every mutation happens in a synchronous method with no
    ``await`` between a read and its dependent write, so the cooperative event loop
    never interleaves two sessions inside one -- the counters need no lock.

    Circuit-breaker -- ``record_result`` counts consecutive faults across completed
    requests (any success resets the run to 0). When the run reaches
    ``fault_threshold`` (> 0) it sets ``stop`` and marks ``aborted``. A dead server
    trips it in a handful of requests -- and so does an alive-but-overloaded one
    once its queue pushes waits past the drive deadline, so the abort log states
    the measured fault run, not a cause. A threshold ``<= 0`` disables the breaker.

    Stall watchdog -- the fault counter above is blind to the one server that never
    completes anything, because a request that does not come back is never counted.
    ``trip_if_stalled`` therefore also trips when ``fault_threshold`` requests are
    outstanding and nothing has completed for ``stall_abort_sec``. It runs at send time
    (``gate_send``) and on the drive's timer (``Driver._stall_watchdog``), since a
    silent server makes ``record_result`` never fire.

    Either trip abandons the requests in flight (``_abandon_outstanding``): once the
    breaker has given up, each one would otherwise hold the drive open for its whole
    allowance.

    ``stop`` (an ``asyncio.Event``) is the wake signal of every way a drive is stopped:
    the breaker, the stall watchdog and the drive wall (``expire``). Each is an abort:
    a session records this and every remaining turn as a fault, so the run scores
    low."""

    def __init__(self, *, fault_threshold: int, total_requests: int,
                 stall_abort_sec: float = DEFAULT_STALL_ABORT_SEC) -> None:
        self.fault_threshold = fault_threshold
        self.total_requests = total_requests
        self.stall_abort_sec = stall_abort_sec
        self.stop = asyncio.Event()
        #: The fault string an aborted drive stamps on the turns it never sent. The
        #: breaker's by default; ``expire`` swaps in the wall's.
        self.abort_error = ABORTED_ERROR
        #: The failure class those turns carry, which travels with the string above so
        #: the two cannot name different causes. A wall cut is a deadline wherever it
        #: lands: the request it caught in flight is filed under DEADLINE by
        #: ``_submit_or_fault``, and the turns behind it are the same event.
        self.abort_reason = FailureReason.CIRCUIT_BREAKER
        self._consecutive = 0
        self._sent = 0
        self._completed = 0                      # real submits folded back, ok or not
        self._faults_total = 0                   # cumulative faults (not just the run)
        self._last_success_t: float | None = None
        # Any completion, fault or not -- distinct from _last_success_t, because the
        # stall watchdog asks whether the server is ANSWERING, not whether it is right.
        self._last_completion_t: float | None = None
        #: Timeout windows of the submits currently in flight (see ``track``).
        self._outstanding: set[asyncio.Timeout] = set()
        #: Windows this breaker actually collapsed. A request that expired on its own
        #: allowance and one the trip cut short both surface as ``TimeoutError``, and
        #: only this tells them apart -- ``aborted`` alone would relabel every natural
        #: deadline miss after the trip as an abandonment.
        self._collapsed: set[asyncio.Timeout] = set()

    @property
    def aborted(self) -> bool:
        """Whether the drive was stopped: the breaker or the stall watchdog tripped, or the
        wall was reached. An abort means waiting out an outstanding request's allowance
        measures nothing."""
        return self.stop.is_set()

    def collapsed(self, window: "asyncio.Timeout") -> bool:
        """Whether the trip is what ended *window*, rather than its own allowance."""
        return window in self._collapsed

    def track(self, window: "asyncio.Timeout") -> None:
        """Register an in-flight submit's timeout window so a trip can collapse it.

        Gating the next send alone would leave a request already outstanding its own
        allowance. Past the drive deadline that allowance is the full ``grace`` floor,
        so a drive that had given up could not return for another full ``grace``
        (``hls.pipeline.pipelines.REQUEST_TIMEOUT_SEC`` on a graded drive). The
        wasted GPU is the visible cost; the one that loses a verdict
        is a run whose remaining wall clock is shorter than that tail, killed mid-wait
        and scoring nothing.

        Registering after the trip collapses on the spot. Unreachable while every
        mutation here is synchronous -- there is no suspension point between
        ``gate_send`` returning "send" and this call, and ``Timeout.__aenter__``
        contains no ``await`` -- so this is belt and braces against that stopping
        being true, not a race the class admits today."""
        if self.aborted:
            if _collapse(window):
                self._collapsed.add(window)
            return
        self._outstanding.add(window)

    def untrack(self, window: "asyncio.Timeout") -> None:
        """Stop considering a submit outstanding, however it ended.

        Deliberately leaves ``_collapsed``: the caller reads it after unwinding, to
        decide what ended the request. It is bounded by the concurrency at the single
        trip, since every send after that is gated away rather than tracked."""
        self._outstanding.discard(window)

    def gate_send(self) -> str:
        """Decide (synchronously) what to do with the request about to be sent:

        - ``"abort"``: the drive was stopped -- record this and every remaining turn
          as a fault (no network), so a dead server scores low without waiting out
          the drive deadline on thousands of doomed requests.
        - ``"send"``: proceed with the real submit.

        This is one of the two places the stall check runs (the other is the drive's
        ``_stall_watchdog`` timer): ``record_result`` fires on completion, which is
        exactly what a hung server never produces."""
        if self.aborted:
            return "abort"
        if self._last_completion_t is None:
            self._last_completion_t = harness_now()   # the drive's own start
        elif self.trip_if_stalled():
            return "abort"
        self._sent += 1
        return "send"

    def trip_if_stalled(self) -> bool:
        """Abort the drive if the server is taking requests and answering none.

        Called from two places, and it needs both. ``gate_send`` catches it early but
        only ticks while arrivals are still going out, which on these workloads is a
        window much shorter than the silence threshold. A drive that goes quiet after its
        last arrival would never be looked at again. So the drive also runs
        ``_stall_watchdog`` alongside it, which ticks on a timer and is the path that
        actually fires on these tasks.
        """
        now = harness_now()
        if not self._stalled(now):
            return False
        log.warning(
            "driver: aborting drive at t=%.1fs: nothing has completed for %.0fs with "
            "%d request(s) outstanding (%d sent, %d completed). The server is "
            "accepting requests and not answering them; %d/%d requests driven",
            now, now - self._last_completion_t, self._sent - self._completed,
            self._sent, self._completed, self._sent, self.total_requests)
        self.stop.set()
        self._abandon_outstanding()
        return True

    def _stalled(self, now: float) -> bool:
        """Is the server taking requests and answering none of them?

        Two conditions, and both are needed. The elapsed silence alone would fire on a
        drive whose arrivals have not started; the outstanding count alone would fire on
        any server slower than its pin, which is a thing to SCORE rather than abort --
        the breaker exists for a server that is not answering, not one answering late.
        """
        if self.stall_abort_sec <= 0 or self.fault_threshold <= 0 or self.stop.is_set():
            return False
        outstanding = self._sent - self._completed
        return (outstanding >= self.fault_threshold
                and self._last_completion_t is not None
                and now - self._last_completion_t >= self.stall_abort_sec)

    def expire(self) -> None:
        """Stop the drive because it reached its wall clock. Takes the breaker's abort
        path so every turn not yet sent is recorded as a fault: the trace stays
        complete, the SLO gate zeroes the run, and the scorer still prices the
        unfinished units for ``upper_bound``. Sticky, and a no-op once stopped -- a
        drive already aborted keeps the reason that stopped it first."""
        if self.stop.is_set():
            return
        self.abort_error = WALL_ERROR
        self.abort_reason = FailureReason.DEADLINE
        log.warning("driver: drive wall clock reached at t=%.1fs; %d/%d requests driven, "
                    "recording the remainder as faults",
                    harness_now(), self._sent, self.total_requests)
        self.stop.set()

    def record_result(self, *, faulted: bool, error: str | None = None) -> None:
        """Fold one completed real submit into the breaker. Trips on the
        ``fault_threshold``-th consecutive fault; any success resets the run. The
        trip is sticky (``stop`` stays set) -- a late success after the abort cannot
        un-abort the drive.

        ``error`` is the event's verbatim fault string; it is logged (the first
        fault, every ``_FAULT_LOG_EVERY``-th after it, and the trip) with a ``t=``
        stamp on the same monotonic clock that stamps the trace's ``t_send``/
        ``t_recv``, so driver.log alone answers when faults started and what they
        were, and its lines correlate 1:1 with trace.json."""
        now = harness_now()
        self._completed += 1
        # Any answer at all, right or wrong, clears the stall watchdog: it asks whether
        # the server is responding, and a fault that came BACK is a response.
        self._last_completion_t = now
        if not faulted:
            self._consecutive = 0
            self._last_success_t = now
            return
        self._consecutive += 1
        self._faults_total += 1
        if self._faults_total == 1 or self._faults_total % _FAULT_LOG_EVERY == 0:
            log.warning(
                "driver: request fault #%d at t=%.1fs (consecutive: %d): %s",
                self._faults_total, now, self._consecutive, error)
        if (self.fault_threshold > 0 and self._consecutive >= self.fault_threshold
                and not self.stop.is_set()):
            last_ok = ("last success %.1fs ago" % (now - self._last_success_t)
                       if self._last_success_t is not None
                       else "no successful response ever received")
            log.warning(
                "driver: aborting drive at t=%.1fs: %d consecutive request faults "
                "(%d/%d completed submits faulted; %s; last fault: %s); "
                "%d/%d requests driven",
                now, self._consecutive, self._faults_total, self._completed,
                last_ok, error, self._sent, self.total_requests)
            self.stop.set()
            self._abandon_outstanding()

    def _abandon_outstanding(self) -> None:
        """Collapse the timeout window of every submit in flight. Nothing outstanding
        is worth waiting for once the breaker has given up on the drive, and each one
        otherwise holds the drive open for its whole allowance."""
        for window in self._outstanding:
            if _collapse(window):
                self._collapsed.add(window)
        self._outstanding.clear()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

class Driver:
    """The driver: runs the two-level open-loop loop against the artifact (L1).

    The drive role builds it with ``driver_for`` from the task and drives it in its own
    process with ``run``, passing the drive's split, tenants, contract and seed. What it
    is built with is fixed at construction: the drive deadline's grace
    (``request_timeout_sec``), the drive wall (``drive_timeout_sec``) and the breaker's
    threshold. The seed (``TaskSpec.seed``) draws every arrival schedule, think time and
    ramp-down budget of the drive.

    Owns the wall clock: it stamps each request's send/receive times and the
    open-loop schedule's intended send time. ``_drive`` captures one shared t0
    anchor for all tenants, then concurrently dispatches each tenant's sessions
    from that anchor.
    """

    #: Seconds a reply's stream may go between two tokens before its request fails, handed to
    #: the artifact client ``run`` builds; None for no such bound. Only a trajectory driver
    #: whose task declares one (``AgentSpec.stall_timeout_sec``) sets it.
    stall_timeout_sec: float | None = None
    #: The sessions the drive dropped on a dead sandbox (``SandboxGone``), in the order dropped,
    #: which the drive role writes into the trace's ``END.json``. Only a trajectory driver drops
    #: one; it sets this when its drive has run.
    dropped_sessions: tuple[DroppedSession, ...] = ()

    def __init__(self, *, request_timeout_sec: float,
                 drive_timeout_sec: float | None = None,
                 fault_abort_threshold: int = DEFAULT_FAULT_ABORT_THRESHOLD) -> None:
        #: Deadline grace: every request must complete within this many seconds after the
        #: last scheduled arrival. A graded drive's is ``hls.pipeline.pipelines.REQUEST_TIMEOUT_SEC``,
        #: a calibration serving's its campaign's. Not a per-request cap; see ``_Deadline``.
        self.request_timeout_sec = request_timeout_sec
        #: Total wall-clock bound on the drive itself (None = unbounded). The grace
        #: above is a per-request floor that re-arms on every turn, so on a session
        #: workload it bounds no total; this does. Reaching it stops the drive and
        #: records the undriven remainder as faults, which zeroes the run on the SLO
        #: gate while still producing a scored record with an upper bound.
        self.drive_timeout_sec = drive_timeout_sec
        #: Consecutive-fault circuit-breaker threshold (see _DriveState); <= 0
        #: disables the breaker.
        self.fault_abort_threshold = fault_abort_threshold

    def _open_wall(self, t0: float) -> "_DriveWall | None":
        """The drive's wall for a drive anchored at ``t0``, or None when unbounded.

        The single place either loop turns ``drive_timeout_sec`` into a bound, so the
        two cannot disagree about what the knob means."""
        if self.drive_timeout_sec is None:
            return None
        wall = _DriveWall(t=t0 + self.drive_timeout_sec)
        log.info("driver: drive wall t=%.1fs (%gs total); past it the drive stops and "
                 "records what it has not driven as faults",
                 wall.t, self.drive_timeout_sec)
        return wall

    @property
    def resources(self) -> ResourceRequest:
        """What the driver asks of the drive role beyond the role's own shape: no GPU and no
        memory (``hls.pipeline.pipelines.driver_shape`` sizes the role from it)."""
        return ResourceRequest(gpu_count=0)

    async def run(self, artifact_url: str, *, split: str, tenants: list[Tenant],
                  contract: ArtifactContract, seed: int, on_event: Callable[[TraceEvent], None]) -> None:
        """Drive the *split* sessions of *tenants* against the artifact at *artifact_url*,
        in this process, with *seed* drawing the schedule, handing *on_event* each event as
        it is recorded, in the order recorded.

        *contract* builds the artifact client every request goes through. *on_event* runs on
        the drive's event loop, between its sends, so it serializes nothing: the drive role's
        hands each event to its trace's writer (``hls.drive.stream``)."""
        # Imported here: the client's module imports hls.spec.contracts, which imports this
        # module through hls.spec.wire.
        from hls.artifact.artifact_client import make_artifact_client

        artifact_client = make_artifact_client(contract, artifact_url,
                                               request_timeout_sec=self.request_timeout_sec,
                                               stall_timeout_sec=self.stall_timeout_sec)
        # The cyclic garbage collector is paused for the drive. A full collection walks
        # every tracked object, and a drive's replies in flight and its workload's sessions
        # are many: one width-65 TTS reply is about 157,000 tracked objects, and a full pass
        # takes time linear in the replies held.
        # A pass stalls the event loop, which drops connections, which the no-failed-request
        # rule counts against the server, and adds itself to every in-flight request's
        # measured latency. A reply is acyclic, so reference counting frees whatever the
        # drive discards without the collector.
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            await self._drive(artifact_client, on_event, datasets.Split(split),
                              tenants=tenants, seed=seed)
        finally:
            if gc_was_enabled:
                gc.enable()
            # Close the reused HTTP client's connection pool; a non-HTTP
            # client's aclose is a no-op.
            await artifact_client.aclose()

    async def _drive(
        self,
        artifact: ArtifactClient,
        on_event: Callable[[TraceEvent], None],
        split,
        *,
        tenants: list[Tenant],
        seed: int,
    ) -> None:
        """Run the two-level open-loop loop over all *tenants*' sessions, *seed* drawing
        the schedule.

        Underscored because it takes an ``on_event`` closure and an in-process
        ``ArtifactClient``; ``run`` builds both and is the driver's entry point.

        One shared t0 is captured before any tenant dispatch, so all sessions from
        all tenants schedule against the same wall-clock anchor (arrival offsets are
        absolute from that anchor). A tenant with ``data_source=None`` is silently
        skipped (produces no sessions).

        Every tenant mounts at ``/data/tenants/<id>/<split>``, is seeded with
        ``tenant_seed(seed, tenant.id)``, and stamps its own ``tenant_id`` on each
        event.
        """
        loaded = load_mounted_sessions(tenants, split)
        if not loaded:
            # No tenants with data: nothing to drive. The verify phase will raise
            # HarnessFault("empty trace") as normal.
            return
        await self._drive_sessions(artifact, on_event, loaded, drive_seed=seed)

    async def _drive_sessions(
        self,
        artifact: ArtifactClient,
        on_event: Callable[[TraceEvent], None],
        loaded: list[tuple],
        *,
        drive_seed: int,
        max_sessions: int | None = None,
    ) -> None:
        """The clock: drive already-materialized ``[(tenant, items)]`` sessions.

        Split out of ``_drive`` so the two callers that need the same timing share
        the code rather than reimplementing it. ``_drive`` reads its sessions off the
        dataset mount and calls this; a calibration serving
        (``ReferenceEngine.serve_traffic``) is handed sessions materialized on the host
        (a reference has no dataset mount) and calls this in-process against the
        engine's own async serving API. Arrivals, think-time draws, ramp-down
        membership, history stitching, the drive deadline and the circuit-breaker are
        therefore one implementation, so a calibration serving and a graded run differ
        only in what is on the other end of ``submit``.

        ``drive_seed`` draws the schedule: the drive's seed from ``_drive``, the
        serving's own from a calibration. Underscored for the same reason ``_drive``
        is: it takes an ``on_event`` closure and an in-process ``ArtifactClient``.
        """
        # The removal list first, so the workload is the sessions that remain; then
        # the session bound, applied before the schedules are drawn so each tenant's
        # arrival process is generated for its own trimmed count -- the offered rate
        # is untouched and only the span shortens.
        loaded = _subsample_sessions(remove_listed_rows(loaded), max_sessions)

        # Warm the shared connection pool off the measured clock, so the first
        # scheduled request reuses an established keep-alive connection instead of
        # paying TCP+TLS handshake on the hot path (which would otherwise delay
        # co-scheduled tenants' first sends, since the handshake occupies the single
        # event loop). Test doubles without warmup() are skipped.
        warmup = getattr(artifact, "warmup", None)
        if warmup is not None:
            await warmup()

        # Schedules and ramp-down membership are decided before the clock anchor,
        # from the workload alone (see plan_sessions): no runtime quantity can reach
        # the request set.
        scheduled, turn_budget = plan_sessions(loaded, drive_seed)

        # The drive deadline must cover every arrival the drive will actually wait
        # on: the last one (see _deadline_anchor).
        last_arrival = _deadline_anchor(
            sorted(t for _, _, starts in scheduled for t in starts))
        grace = self.request_timeout_sec

        # One shared control block for the whole drive: the consecutive-fault
        # circuit-breaker. total_requests (used only for the abort log's "M/total
        # driven") is an upper bound for a tenant with a context budget, whose sessions
        # can end before their last drawn turn. On a healthy drive this is inert:
        # gate_send always returns "send" and stop never fires.
        total_requests = sum(
            turn_budget.get((tenant.id, s), len(item.turns))
            for tenant, items, _ in scheduled for s, item in enumerate(items))
        state = _DriveState(fault_threshold=self.fault_abort_threshold,
                            total_requests=total_requests)

        # One shared t0 anchor for all tenants (no per-tenant clock reset).
        t0 = harness_now()

        # Two bounds can cut this drive, and the earlier one holds. The schedule-derived
        # instant is always present, sized by `_WALL_GRACES` so a healthy drive clears it
        # by a wide margin; the driver's own `drive_timeout_sec`, the stage's drive wall,
        # is the other when it is set. Taking the earlier means neither can loosen the
        # other.
        wall = _DriveWall(t=t0 + last_arrival + _WALL_GRACES * grace)
        own_wall = self._open_wall(t0)
        if own_wall is not None and own_wall.t < wall.t:
            wall = own_wall
        deadline = _Deadline(
            t=t0 + last_arrival + grace,
            grace=grace,
            error=("transport: request did not complete by the drive deadline "
                   f"(last scheduled arrival + {grace:g}s, min {grace:g}s per request)"),
            wall=wall)
        log.info("driver: drive deadline t=%.1fs, wall t=%.1fs (last scheduled arrival "
                 "t=%.1fs + %gs grace; every request gets at least %gs and none runs "
                 "past the wall)",
                 deadline.t, wall.t, t0 + last_arrival, grace, grace)
        # The deadline is the only time cap; drop any client-side per-request cap so
        # a request queued early in the drive can wait out the server's queue.
        clear_cap = getattr(artifact, "clear_request_cap", None)
        if clear_cap is not None:
            clear_cap()

        async def _run_tenant(tenant, items, starts) -> None:
            workload = tenant.workload
            seed = tenant_seed(drive_seed, tenant.id)
            cap = workload.output_length_cap.max_tokens()
            # A session's index keys its think-time stream and its turn budget.
            # return_exceptions=True so a fault in one session lets siblings finish;
            # then surface the first real exception.
            results = await asyncio.gather(*[
                self._session(workload, s, t0 + starts[s], item, tenant.id,
                              cap, artifact, on_event, seed=seed, state=state,
                              deadline=deadline,
                              n_turns=turn_budget.get((tenant.id, s)))
                for s, item in enumerate(items)], return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException):
                    raise r

        async def _wall_watchdog() -> None:
            """Expire the drive at its wall. Sleeping on the remaining span rather than
            polling keeps the healthy path free of wakeups; the task is cancelled the
            moment the tenants finish, so a drive that keeps up never fires it."""
            await asyncio.sleep(wall.remaining())
            state.expire()

        # Dispatch all tenants concurrently under the shared t0, with both watchdogs
        # ticking beside them. They watch different things: the wall bounds the drive's
        # total time, while the stall watchdog catches a server that has gone silent
        # long before the wall. The stall check has to be its own task because the
        # send-time check only runs while arrivals are going out, and on these workloads
        # that window is shorter than the silence it is watching for, so a drive that
        # goes quiet after its last arrival would never be
        # re-examined.
        watchdogs = [asyncio.create_task(self._stall_watchdog(state)),
                     asyncio.create_task(_wall_watchdog())]
        try:
            tenant_results = await asyncio.gather(
                *[_run_tenant(t, items, starts) for t, items, starts in scheduled],
                return_exceptions=True)
        finally:
            for w in watchdogs:
                w.cancel()
        for r in tenant_results:
            if isinstance(r, BaseException):
                raise r

    async def _stall_watchdog(self, state: "_DriveState | None") -> None:
        """Ask ``state`` whether the server has gone silent, on a timer.

        Cancelled when the drive returns. Polls at a tenth of the threshold, which only
        decides how promptly the abort lands, not whether it is right.

        It makes the breaker see a hung server, which the fault counter cannot: that
        counter only moves on a completed submit, and a server accepting requests and
        answering none produces none. A trip stops future sends and abandons the
        requests in flight, as a fault-run trip does, so a drive that goes quiet after
        its last arrival ends at the stall threshold instead of waiting each request's
        allowance out, which can be hours.
        """
        if state is None or state.stall_abort_sec <= 0 or state.fault_threshold <= 0:
            return
        poll = max(1.0, min(30.0, state.stall_abort_sec / 10.0))
        while not state.stop.is_set():
            await asyncio.sleep(poll)
            if state.trip_if_stalled():
                return

    async def _session(self, workload, s, t0, item, tenant_id: str, cap,
                       artifact, on_event, *, seed: int = 0, state=None,
                       deadline: _Deadline | None = None,
                       n_turns: int | None = None) -> None:
        """Drive one session: open-loop arrival, then turn-by-turn with the one
        reactive inter-turn delay and cumulative history stitching.

        Each turn appends the user message, sends the cumulative conversation,
        records one event, then appends the assistant reply.

        ``tenant_id`` comes from the Tenant.id.

        ``state`` (the shared ``_DriveState``, or ``None`` for direct callers/tests)
        adds the fail-fast circuit-breaker without touching the healthy path: each
        send is gated (``gate_send``) and each real result folded back
        (``record_result``). When the drive has been stopped the session records this
        and every remaining turn as a fault (no network) so a dead server scores low
        fast. The arrival + inter-turn waits become abort-aware so a tripped drive does
        not stall on the full arrival span. With ``state=None`` (or a healthy
        ``state``) the loop is byte-for-byte the ungated one.

        ``n_turns`` is the session's draw-time turn budget from
        ``rampdown_turn_budget`` (None = serve every turn): the leading ``n_turns``
        of ``item.turns`` are driven and the rest do not exist for this run. The
        slice is decided before the drive's clock starts, so it cannot depend on
        anything the artifact does.

        A tenant's context budget (``Workload.history_char_budget``) can end a session
        earlier than that: the first turn whose conversation would exceed the budget is
        not sent, and neither is any turn after it. Unlike the draw-time budget above,
        this cut is decided at serve time from the replies the artifact actually
        generated."""
        if not item.turns:
            raise DatasetError(f"item {item.request_id!r} has no turns")
        if len(item.ground_truth_turns) < len(item.turns):
            raise DatasetError(
                f"item {item.request_id!r}: {len(item.ground_truth_turns)} ground-truth "
                f"turn(s) for {len(item.turns)} input turn(s)")

        stop = state.stop if state is not None else None
        await self._sleep_until(t0, stop)                        # open-loop session arrival (un-gated)
        # The draw-time turn budget: turns past it do not exist for this run (see
        # rampdown_turn_budget). Validation above still covers the full item -- the
        # dataset's integrity is not conditional on how much of it this drive uses.
        turns = item.turns if n_turns is None else item.turns[:n_turns]
        messages: list[dict[str, str]] = []
        for k, turn_env in enumerate(turns):                 # rendered Envelopes
            # Explicitly stamp metadata['messages'] rather than relying on the
            # artifact's and oracle's fallbacks to independently derive the same list.
            # Shaping goes through the shared helpers (user_message/request_messages/
            # stamp_request), the single request-shaping seam. Message dicts are never
            # mutated after append, so sharing them across turns avoids an O(N^2)
            # per-element copy; stamp_request snapshots the list per request.
            messages.append(user_message(turn_env))
            sent = request_messages(turn_env, messages)
            # The tenant's context budget ends the session (see
            # Workload.history_char_budget): a conversation that does not fit is
            # over, and its remaining turns go unserved. Decided before the
            # think-time wait, since this turn will never be sent, and before any
            # bookkeeping, so an unserved turn is not a fault, not an SLO miss and
            # not an event.
            if _over_context_budget(sent, workload.history_char_budget):
                if k == 0:
                    raise DatasetError(
                        f"item {item.request_id!r}: its FIRST turn is already "
                        f"{history_chars(sent):,} characters against tenant "
                        f"{tenant_id!r}'s {workload.history_char_budget:,}-character "
                        "context budget, so the session can serve no turn at all. "
                        "An opening turn is corpus text alone -- no generated reply "
                        "is in it -- so this is a corpus that does not fit the "
                        "tenant, not a runtime condition: fix the dataset's "
                        "input-length filter or the tenant's budget.")
                return
            if k > 0 and (stop is None or not stop.is_set()):
                # The one reactive timing point: think-time after the prior response.
                # Use the per-tenant seed so two tenants get independent inter-turn
                # RNG streams (aligns with the per-tenant arrival seed above). Waited
                # through _sleep_until so it wakes on the stop event -- a breaker
                # that trips while this session is mid think-time must not leave it
                # sitting out a think-time it will never use (the pre-turn guard only
                # catches a stop already set before the wait).
                think = workload.inter_turn_delay.delay(k, seed=(seed, s))
                await self._sleep_until(harness_now() + think, stop)
            req = stamp_request(turn_env, sent, tenant_id=tenant_id,
                                session_id=item.request_id, turn_idx=k, max_tokens=cap)
            # Circuit-breaker gate, checked just before the send. "send" is the only
            # path that touches the network. "abort" records this + every remaining
            # turn as a fault without sending, so a dead server scores low without
            # waiting out the drive deadline thousands of times.
            decision = state.gate_send() if state is not None else "send"
            if decision == "abort":
                # Never sent, so nothing was observed: t_send is when the send
                # would have happened, and that is also when we gave up on it.
                sent = _failed(state.abort_error, state.abort_reason,
                               RequestAttempt(), harness_now())
            else:
                sent = await self._submit_or_fault(
                    artifact, req, deadline=deadline, state=state)
                # A request the breaker itself abandoned is not evidence about the
                # server, so it stays out of the breaker's census -- as the gate's
                # own "abort" path already does. Folding it in would push counts past
                # the abort line that already reported them, and could log fresh
                # "request fault #N" lines for requests the harness cancelled.
                if state is not None and sent.error != ABANDONED_ERROR:
                    state.record_result(faulted=sent.error is not None, error=sent.error)
            gt = item.ground_truth_turns[k] if k < len(item.ground_truth_turns) else None
            on_event(TraceEvent(
                request_id=f"{item.request_id}.{k}",
                session_id=item.request_id,
                turn_index=k,
                input=req,
                response=sent.response,
                error=sent.error,
                failure=sent.failure,
                ground_truth=gt,
                t_send=sent.t_send,
                t_recv=sent.t_recv,
                t_planned_send=t0 if k == 0 else None,
            ))
            rtp = sent.response.output.first_of("text") if sent.response is not None else None
            messages.append(
                {"role": "assistant", "content": rtp.text if rtp is not None else ""})

    async def _sleep_until(self, t0: float, stop: "asyncio.Event | None" = None) -> None:
        """Open-loop wait until the absolute monotonic target ``t0`` (the run-start
        anchor plus the schedule offset). A non-positive remaining wait returns
        immediately (the target is at/behind now).

        When ``stop`` is given (the drive's stop event) the wait wakes early if it
        fires, so a tripped drive finalizes fast instead of
        stalling until this session's scheduled arrival (which may be minutes into
        the open-loop span). Absent a ``stop`` -- or on a healthy drive where it
        never fires -- this waits the full ``remaining``, i.e. the arrival timing is
        identical to the plain-sleep path."""
        remaining = t0 - harness_now()
        if remaining <= 0:
            return
        if stop is None:
            await asyncio.sleep(remaining)
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            return

    async def _submit_or_fault(
        self, artifact: ArtifactClient, env: Envelope,
        deadline: _Deadline | None = None,
        state: "_DriveState | None" = None,
    ) -> _Sent:
        """Submit one request, converting an ArtifactFault or a deadline miss into a
        failed ``_Sent`` carrying the diagnostic record.

        On success the response and ``t_recv`` are set; on a failure the fault string
        and a ``RequestFailure`` are, and ``t_recv`` stays ``None`` -- the event is out
        of the latency sample while it still counts against pass-rate. With a
        ``deadline``, allowance = ``deadline.allowance(t_send)`` -- the per-request floor cut
        at the drive wall. A miss gets the deadline's one fault string. Only the
        harness bug propagates (invalid run).

        The ``RequestAttempt`` is created here, before the send, because a deadline miss
        cancels the submit: by the time ``asyncio.TimeoutError`` surfaces, the transport's
        own view of the request is gone, and this is the only reference left to what it
        saw."""
        attempt = RequestAttempt()
        t_send = harness_now()
        try:
            if deadline is None:
                resp = await artifact.submit(env, attempt=attempt)
            else:
                # The allowance carries its own fault string: the per-request floor, or
                # the drive wall's when the wall comes first, so a request the wall cut
                # is not filed under a grace it never reached.
                allowance, miss_error = deadline.allowance(t_send)
                # ``asyncio.timeout`` rather than ``wait_for`` so the breaker can reach
                # a request already in flight: it collapses the deadline of every
                # outstanding submit, which unwinds them where waiting would hold the
                # drive open for the rest of their allowance. Both await the coroutine
                # inline, without a Task, so a submit that faults without suspending
                # still folds back before the next session is scheduled -- which is
                # what makes the breaker trip after its threshold rather than after
                # every session has sent.
                try:
                    async with asyncio.timeout(allowance) as window:
                        if state is not None:
                            state.track(window)
                        try:
                            resp = await artifact.submit(env, attempt=attempt)
                        finally:
                            if state is not None:
                                state.untrack(window)
                except TimeoutError:
                    # Which of the two ended it is per window, not per drive: once the
                    # breaker has tripped, a request that goes on to exhaust its own
                    # allowance is still a deadline miss, and calling it an
                    # abandonment would report it as having been cut short when it
                    # was not.
                    if state is not None and state.collapsed(window):
                        # Sent, unlike the request the gate skips, so it gets its own
                        # string: the census must not read "we asked and stopped
                        # waiting" as "we never asked", and only the former observed
                        # anything.
                        return _failed(ABANDONED_ERROR, FailureReason.CIRCUIT_BREAKER,
                                       attempt, t_send)
                    return _failed(miss_error, FailureReason.DEADLINE, attempt, t_send)
            # A protocol whose stream ends with a frame that is not product reports
            # when the product itself was complete; that is the request's job
            # completion time, and the trailing frame is not charged to the server.
            t_recv = resp.t_complete if resp.t_complete is not None else harness_now()
            return _Sent(response=resp, error=None, failure=None,
                         t_send=t_send, t_recv=t_recv)
        except ArtifactFault as f:
            return _failed(f"{f.category}: {f}", _fault_reason(f, attempt), attempt, t_send)


class TrajectorySessionDriver(Driver):
    """Drives one agentic trajectory per dataset item.

    For an agentic task a scored session is a whole trajectory: one seed question,
    then as many LLM turns and tool calls as the framework decides to take, bounded
    by ``max_turns``. Each LLM call emits its own TraceEvent, so a token-level gate
    scores every call the artifact served, and the tool calls run through the harness
    ``ToolRunner`` so their wall time is real and counts toward the session makespan.

    A subclass of ``Driver``: it reuses the base ``run`` and ``_drive``, the drive
    deadline and the breaker's threshold. Only the loop differs, so it overrides
    ``_drive_sessions``. The base driver stitches a scripted user/assistant
    alternation; here the history's shape is chosen by the model as it decides what to
    search for and read, which no amount of stitching configuration expresses.

    ``driver_for`` builds it whole from the task's agent seam (``AgentSpec``): the
    framework, the tool runner, the turn and LLM-admission bounds, the task's window
    and chat-template pin, and the tool assets' memory and GPU asks, which size the
    driver role.

    Open-loop scheduling: ``workload.arrival.schedule`` gives absolute per-session
    start offsets, and ``workload.inter_turn_delay`` is waited before every follow-up
    LLM call. When ``max_concurrent_llm`` is set, a semaphore gates only the artifact
    submit -- a trajectory waiting on the browser or on think time never holds an LLM
    slot, so sessions overlap freely while the contestant's concurrent LLM batch stays
    bounded. Each event's ``t_send`` is stamped once the think time is over, so a
    turn's measured latency excludes it; time spent queueing for an LLM slot is still
    inside the span, since that is the contestant's own throughput.

    What a task's agent seam opts into: a scheduled stop (``AgentSpec.scheduled_stop``),
    past which no request is issued and at whose cut every open stream is recorded on what
    it streamed (``CUT_EXTRA``), neither as a fault; and a stall budget, which the artifact
    client enforces between tokens (``ArtifactStallError``). Whatever the task, an
    artifact fault the framework raises about an accepted reply is recorded on its turn,
    and a session whose sandbox died (``SandboxGone``) is dropped while the rest drive on.
    """

    def __init__(self, *, framework: AgentFramework, tool_runner: ToolRunner, max_turns: int,
                 max_concurrent_llm: int | None, max_context_len: int | None = None,
                 chat_template_kwargs: Mapping[str, TemplateArg] | None = None,
                 assets_memory_mb: int | None = None, assets_gpu: bool = False,
                 gpu_kind: str | None = None, request_timeout_sec: float,
                 drive_timeout_sec: float | None = None,
                 fault_abort_threshold: int = DEFAULT_FAULT_ABORT_THRESHOLD,
                 scheduled_stop: ScheduledStop | None = None,
                 stall_timeout_sec: float | None = None) -> None:
        super().__init__(request_timeout_sec=request_timeout_sec,
                         drive_timeout_sec=drive_timeout_sec,
                         fault_abort_threshold=fault_abort_threshold)
        #: The task's bounded window (``AgentSpec.scheduled_stop``), or None for a drive that
        #: runs until its trajectories end or its wall.
        self.scheduled_stop = scheduled_stop
        #: The task's stall budget (``AgentSpec.stall_timeout_sec``), handed to the artifact
        #: client; None for none.
        self.stall_timeout_sec = stall_timeout_sec
        #: The framework that chooses each trajectory's turns (``AgentSpec.framework``).
        self.framework = framework
        #: The harness-side runner every tool call goes through (``AgentSpec.tool_runner``).
        self.tool_runner = tool_runner
        #: The most LLM turns one trajectory may take (``AgentSpec.max_turns``).
        self.max_turns = max_turns
        #: How many artifact submits may be in flight at once across trajectories; None
        #: admits every one (``AgentSpec.max_concurrent_llm``).
        self.max_concurrent_llm = max_concurrent_llm
        #: The task's context window (``TaskSpec.max_context_len``), stamped on every
        #: trajectory's ``HarnessContext``; None when the task declares none.
        self.max_context_len = max_context_len
        #: The model's chat-template pin (``ModelSpec.chat_template_kwargs``), stamped
        #: beside the window.
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        #: Host memory the tool assets' loaded bundle needs, in MB
        #: (``ToolAssets.memory_request``); None asks for none.
        self.assets_memory_mb = assets_memory_mb
        #: Whether the tool assets encode on an accelerator, which gives the driver one GPU.
        self.assets_gpu = assets_gpu
        #: The kind of that GPU, the task's accelerator kind; None without one.
        self.gpu_kind = gpu_kind

    @property
    def resources(self) -> ResourceRequest:
        """What the task's tool assets ask of the drive role: host memory for the loaded
        bundle and, when the tool encodes on an accelerator, one GPU of the task's kind;
        nothing, like the base driver, when the tools load nothing."""
        return ResourceRequest(gpu_count=1 if self.assets_gpu else 0,
                               gpu_kind=self.gpu_kind if self.assets_gpu else None,
                               memory_mb=self.assets_memory_mb)

    def harness_context(self, tenant_id: str, *, max_output_tokens: int | None) -> HarnessContext:
        """The context a trajectory of tenant *tenant_id* is handed, sent now: the tenant's
        output cap, and the task's context window and chat-template pin."""
        return HarnessContext(tenant_id=tenant_id, t_send=harness_now(),
                              max_output_tokens=max_output_tokens, max_context_len=self.max_context_len,
                              chat_template_kwargs=self.chat_template_kwargs)

    def _open_scheduled_stop(self, t0: float) -> "_StopWindow | None":
        """The task's scheduled stop anchored on a drive starting at ``t0``, or None where the
        task declared none."""
        stop = self.scheduled_stop
        if stop is None:
            return None
        window = _StopWindow(t_stop=stop.stop_at(t0), t_cut=stop.cut_at(t0))
        log.info("driver: scheduled stop -- no request issued after t=%.1fs (%gs), open streams "
                 "cut at t=%.1fs (%gs later); neither records a fault",
                 window.t_stop, stop.wall_clock_sec, window.t_cut, stop.cut_grace_sec)
        return window

    async def _drive_sessions(
        self,
        artifact: ArtifactClient,
        on_event: Callable[[TraceEvent], None],
        loaded: list[tuple],
        *,
        drive_seed: int,
        max_sessions: int | None = None,
    ) -> None:
        """Run one trajectory per already-materialized session in ``[(tenant, items)]``.

        Same signature as the base driver's, so a calibration serving hands either
        driver its sessions the same way. ``max_sessions`` is the knob that bounds a
        trajectory serving: a trajectory's request count is the model's own decision.
        """
        # The same two steps the base driver takes before anchoring its clock: drop the
        # rows the task removed from the workload (DataSource.removed_rows), then trim
        # to the session bound. Sharing them keeps one definition of what a tenant's
        # traffic is, so a removal list means the same thing to both drivers.
        loaded = _subsample_sessions(remove_listed_rows(loaded), max_sessions)
        if not loaded:
            return

        warmup = getattr(artifact, "warmup", None)
        if warmup is not None:
            await warmup()
        # Tool and framework state load before the clock too, so a corpus open or a
        # tokenizer load is not billed to the first trajectory that happens to need it.
        await self.tool_runner.warmup()
        await self.framework.warmup()

        t_start = harness_now()

        llm_sem = (asyncio.Semaphore(self.max_concurrent_llm)
                   if self.max_concurrent_llm else None)
        wall = self._open_wall(t_start)
        stop = self._open_scheduled_stop(t_start)
        stamped_inputs: dict[str, Envelope] = {}
        submitted: dict[str, float] = {}
        #: Turns already recorded per session, so a trajectory the wall cuts can stamp
        #: its fault on the turn index that would have come next.
        turns_seen: dict[str, int] = {}
        #: The request each session sent last, by session id: the one whose reply a
        #: framework refuses when it raises about a reply (``_record_framework_fault``).
        last_sent: dict[str, str] = {}
        #: Sessions dropped because their sandbox died under them. A served agent chooses
        #: the commands its container runs, so it can exhaust or wedge it, and one that
        #: does must not end the drive for the sessions it never touched.
        dropped: list[DroppedSession] = []
        self.dropped_sessions = ()

        async def _aev(ev: TraceEvent) -> None:
            # The base run hands us a sync on_event (appends to a list); the
            # framework awaits on_event, so adapt sync -> async here. The event is
            # re-stamped with the envelope actually sent, so the trace records the
            # request the artifact saw -- which is what the verifier teacher-forces.
            # Popped, not read: this is the only reader, and a framework emits one
            # event per LLM call, so an entry left behind is one no event will ever
            # claim. Getting instead would grow the map for the whole drive, one entry per
            # turn of every session -- and for a call whose event never fires, the
            # entry is the only thing still holding that turn's envelope, which on a
            # trajectory task is the accumulated conversation. The fault paths below
            # pop for the same reason.
            stamped = stamped_inputs.pop(ev.input.request_id if ev.input else "", None)
            if stamped is not None:
                ev = ev.model_copy(update={"input": stamped})
            # A framework takes its own send time before it hands the call over, which
            # is ahead of the driver's think time. Re-stamp with the driver's, taken
            # once the thinking is done, so waiting between turns is not billed as
            # latency. Keyed on the envelope submitted, like the input re-stamp above,
            # so a framework that numbers its events some other way still gets it.
            key = ev.input.request_id if ev.input is not None else ev.request_id
            if key in submitted:
                ev = ev.model_copy(
                    update={"t_send": max(ev.t_send, submitted.pop(key))})
            # The longest the reply went between tokens, which a stall budget is chosen
            # against; recorded only by a drive that has one, on a reply that timed its tokens.
            if self.stall_timeout_sec is not None and ev.response is not None and ev.response.packet_timestamps:
                gap = max_token_gap_sec(ev.response.packet_timestamps)
                ev = ev.model_copy(update={"extras": {**ev.extras, MAX_TOKEN_GAP_EXTRA: gap}})
            sid = ev.session_id or ev.request_id
            turns_seen[sid] = max(turns_seen.get(sid, 0), (ev.turn_index or 0) + 1)
            on_event(ev)

        async def _llm_call(env: Envelope, output_cap: int | None,
                            inter_turn_delay: "InterTurnDelay") -> ArtifactResponse:
            # Think time before every follow-up turn. It waits before the semaphore, so
            # a thinking trajectory never holds an LLM admission slot, and before the
            # submit, so it is a gap between turns rather than work asked of the
            # artifact. Turn 0 is placed by the arrival schedule instead, and a
            # framework that does not number its turns gets no delay.
            turn = env.metadata.get("turn_index", 0)
            session = env.metadata.get("session_id") or env.request_id.rsplit(".", 1)[0]
            if isinstance(turn, int) and turn > 0:
                await asyncio.sleep(
                    inter_turn_delay.delay(turn, seed=(drive_seed, session)))
            # The scheduled stop, checked after the think time so it bounds the send rather
            # than the framework's decision to ask. The trajectory ends here with no event
            # and no fault: nothing was asked of the server.
            if stop is not None and stop.past_stop:
                raise DriveWallClockReached(
                    f"the drive issues no request after t={stop.t_stop:.1f}s, and turn {turn} of "
                    f"session {env.metadata.get('session_id') or env.request_id} came up after it")
            # Stamp the workload cap on every LLM turn (mirrors the base driver's
            # per-turn stamp) so the wire protocol forwards max_tokens and the
            # reference decodes under the same budget as the artifact. A framework may
            # request a smaller cap for a constrained repair or final-answer turn;
            # that explicit cap is preserved, so artifact and verifier replay the same
            # narrower contract.
            meta = dict(env.metadata)
            requested_cap = meta.get("max_tokens")
            meta["max_tokens"] = (
                requested_cap if isinstance(requested_cap, int) and requested_cap > 0
                else output_cap
            )
            req = env.model_copy(update={"metadata": meta})
            stamped_inputs[env.request_id] = req
            last_sent[session] = env.request_id
            # The send time the event carries. Recorded after the think time, which is
            # a gap between turns and not work asked of the artifact, and before
            # admission control, because queueing for an LLM slot is the contestant's
            # own throughput and belongs in the turn's latency.
            submitted[env.request_id] = harness_now()
            # The transport's running record of the request, kept where the scheduled stop
            # may cut it: the frames a cancelled submit collected outlive it only here.
            attempt = RequestAttempt() if stop is not None else None
            # Whether the turn reached the server. A turn the cut takes while it queues for
            # an admission slot never did.
            posted = False

            async def _submit() -> ArtifactResponse:
                nonlocal posted
                posted = True
                try:
                    return await artifact.submit(req, attempt=attempt)
                except ArtifactFault as f:
                    # The artifact's fault is a measurement, not a harness error:
                    # record the turn it broke and unwind this trajectory, exactly as
                    # the base driver's _submit_or_fault does. Letting it escape would
                    # end the whole rollout with no trace, so a server that crashed or
                    # stopped answering would score nothing at all rather than zero
                    # with the bound the scorer exists to report.
                    #
                    # The envelope actually sent, so the fault event carries the same
                    # tenant/session stamping the answered turns do, and the send time
                    # taken before the submit, which is the only record of how long the
                    # turn waited -- a failed event has no t_recv.
                    # The fault is classified from its own type, not from the attempt,
                    # and the failed event's ``observed`` stays empty. A served 5xx
                    # therefore reads "transport" rather than a server error -- the
                    # census loses that one distinction, not the fault itself.
                    _record_fault(req, f"{f.category}: {f}",
                                  _fault_reason(f, RequestAttempt()),
                                  t_send=submitted.get(env.request_id), evidence=f.extras)
                    raise _TrajectoryAborted from f

            try:
                # Admission control wraps only the artifact submit, never tool_runner.run:
                # tool waits do not hold an LLM slot, so trajectories blocked on the browser
                # overlap while the contestant's concurrent LLM batch stays bounded.
                if llm_sem is not None:
                    async with llm_sem:
                        return await _submit()
                return await _submit()
            except BaseException as exc:
                # A cancellation at or past the cut instant is the scheduled stop taking
                # this turn, recorded on what it streamed before the maps are dropped. Only
                # a posted turn: one the cut takes in the admission queue never reached the
                # server, so there is no latency of the server's to price and it leaves no
                # event. That queue is still charged to every turn that went out, whose
                # send time is taken ahead of it.
                if (isinstance(exc, asyncio.CancelledError) and posted
                        and stop is not None and stop.cutting):
                    _record_cut(req, attempt, submitted.get(env.request_id))
                # The turn ended without an event to carry either entry; drop both
                # rather than let them re-stamp a later turn that reuses the request
                # id. Out here rather than around the submit, because a turn can be
                # cancelled while it queues for an LLM slot -- before the submit runs
                # at all -- and the entry left behind would then hold that turn's
                # accumulated conversation for the rest of the drive.
                submitted.pop(env.request_id, None)
                stamped_inputs.pop(env.request_id, None)
                raise

        def _record_fault(env: Envelope, error: str, reason: FailureReason,
                          observed: "RequestAttempt | None" = None,
                          *, due: float | None = None,
                          t_send: float | None = None,
                          evidence: "Mapping[str, Any] | None" = None) -> None:
            """Record a turn that produced no response, so the trace stays complete.

            This is what makes a broken or truncated trajectory measurable. Every turn
            it did finish answered, so without an unanswered one the SLO hands the
            session a makespan and credits it for stopping early. The unanswered turn
            is what leaves the unit with no ratio, which fails the SLO gate and sends
            it to ``censored_ratio`` for the run's ``upper_bound``.

            ``t_send`` is when the turn went out, for a turn that was sent; the fault
            instant only for one that never was. A failed event carries no ``t_recv``,
            so the send time is the only record of how long the turn waited, and
            stamping it at the fault leaves the turn a zero duration -- which
            ``censored_ratio`` prices at its most lenient floor.

            ``due`` is the session's scheduled arrival when known, so a censored price
            runs from when the trajectory was owed rather than from when the harness
            gave up on it.

            ``env`` is the request the fault is about: the envelope the framework sent, or
            for a turn that never was, the dataset row ``_record_session_fault`` stamps.
            The event takes the turn that request names, not the count of turns emitted:
            a framework may hold a finished turn's event until the next turn answers, and
            the fault must not take the held turn's place. It carries the request's
            prompt length where the framework counted one (``PREFILL_TOKENS_EXTRA``),
            since ``censored_ratio`` prices an unanswered turn against the TTFT budget
            that length buys and against the flat term without it. ``evidence`` is what
            the fault said about the reply (``ArtifactFault.extras``), which a failed event
            has no response to hold."""
            sid = env.metadata.get("session_id") or env.request_id.rsplit(".", 1)[0]
            k = _turn_index(env)
            if k is None:
                k = turns_seen.get(sid, 0)
            now = harness_now()
            extras = dict(evidence or {})
            # The harness's count wins a collision: the SLO reads it, and evidence is for a
            # person.
            if PREFILL_TOKENS_EXTRA in env.metadata:
                extras[PREFILL_TOKENS_EXTRA] = env.metadata[PREFILL_TOKENS_EXTRA]
            # Anything recorded for the session afterwards is numbered past this turn.
            turns_seen[sid] = max(turns_seen.get(sid, 0), k + 1)
            on_event(TraceEvent(
                request_id=f"{sid}.{k}", session_id=sid, turn_index=k,
                input=env, response=None, error=error,
                failure=RequestFailure(reason=reason, t_abandoned=now,
                                       observed=observed or RequestAttempt()),
                ground_truth=None, t_send=now if t_send is None else t_send,
                t_recv=None, t_planned_send=due if k == 0 else None, extras=extras))

        def _record_cut(env: Envelope, attempt: "RequestAttempt | None",
                        t_send: float | None) -> None:
            """Record a request the scheduled stop cut mid-flight, on what it streamed.

            Not a failure: the window closing says nothing about the server. The event
            carries the tokens that arrived, their arrival times and their report, so the
            latency SLO prices the rate the reply streamed at and the token-level gate
            grades the positions it served, and ``t_recv`` is the cut instant, which for a
            request that reached no first token makes its whole wait its time to first
            token. ``CUT_EXTRA`` marks the reply as ending where the harness stopped
            listening rather than where the server stopped.

            The framework never sees this reply, so the driver asks its check on the
            framework's behalf (``AgentFramework.check_reply``); a refused reply is a failed
            request, as the same refusal is on an answered turn.

            The turn is the one the cut request names, not the count of turns emitted: a
            framework may hold a finished turn's event to stamp a later tool result on it
            and emit it during this same cancellation, just after this event."""
            partial = attempt.partial if attempt is not None else None
            sid = env.metadata.get("session_id") or env.request_id.rsplit(".", 1)[0]
            k = _turn_index(env)
            if k is None:
                k = turns_seen.get(sid, 0)
            now = harness_now()
            payload = TextPayload(
                text=partial.text if partial else "",
                tokens=list(partial.token_ids) if partial and partial.token_ids else None,
                logprobs=list(partial.logprobs) if partial and partial.logprobs else None,
                finish_reason=None)
            broken = self.framework.check_reply(payload.text, payload.tokens)
            if broken is not None:
                _record_fault(env, f"{ArtifactProtocolError.category}: {broken}",
                              FailureReason.PROTOCOL, observed=attempt, t_send=t_send)
                return
            stamps = list(partial.packet_timestamps) if partial else []
            extras: dict[str, Any] = {CUT_EXTRA: True}
            if self.stall_timeout_sec is not None and stamps:
                extras[MAX_TOKEN_GAP_EXTRA] = max_token_gap_sec(stamps)
            # An answered turn's event gets the prompt length from the framework's own
            # recorder, which a cut turn never reaches; the envelope is the only place left.
            if PREFILL_TOKENS_EXTRA in env.metadata:
                extras[PREFILL_TOKENS_EXTRA] = env.metadata[PREFILL_TOKENS_EXTRA]
            # The cut occupies this turn, so anything recorded for the session afterwards,
            # the wall's fault if its cleanup wedges, is numbered past it.
            turns_seen[sid] = max(turns_seen.get(sid, 0), k + 1)
            on_event(TraceEvent(
                request_id=env.request_id, session_id=sid, turn_index=k, input=env,
                response=ArtifactResponse(
                    output=Envelope(request_id=env.request_id, contents=[payload]),
                    t_first_packet=stamps[0] if stamps else None,
                    packet_timestamps=stamps or None),
                ground_truth=None, t_send=now if t_send is None else t_send,
                t_recv=now, extras=extras))

        def _record_session_fault(item, tenant_id: str, due: float, error: str,
                                  reason: FailureReason, *, t_send: float | None = None,
                                  evidence: "Mapping[str, Any] | None" = None) -> None:
            """``_record_fault`` for a fault known at the trajectory's boundary rather than
            inside the turn that hit it.

            The envelope here is the raw dataset one, which carries no harness
            metadata: the framework stamps that on the copy it sends, and a turn that
            never emitted has no sent copy here. Stamping it is what puts the fault in the
            tenant's slice -- ``Trace.events_for`` is a total lookup on
            ``metadata["tenant_id"]``, so an unstamped fault belongs to no tenant, the
            cut trajectory's answered turns group alone, and it is priced as a fast
            finish rather than sent to ``censored_ratio``.

            Its request id is the session's, not the turn's, for the same reason, so
            it is renumbered here to the turn the event is recorded under. Answered
            turns carry the envelope the framework built, whose id already names the
            turn; the driver's own maps key off ``input.request_id`` and the verifiers
            read ``ev.input``, so the two must not disagree."""
            k = turns_seen.get(item.request_id, 0)
            env = item.turns[k] if k < len(item.turns) else item.turns[0]
            env = env.model_copy(update={"request_id": f"{item.request_id}.{k}"})
            env = _stamped(env, HarnessContext(tenant_id=tenant_id, t_send=due), item.request_id, k)
            _record_fault(env, error, reason, due=due, t_send=t_send, evidence=evidence)

        def _record_wall_fault(item, tenant_id: str, due: float) -> None:
            """The wall's case: the trajectory was cut, or never started, because the drive
            reached its total time limit."""
            _record_session_fault(item, tenant_id, due, WALL_ERROR, FailureReason.DEADLINE)

        def _record_framework_fault(item, tenant_id: str, due: float, fault: ArtifactFault) -> None:
            """An artifact fault the framework raised about a reply the submit path had
            accepted, such as a reply whose text is not its own tokens. Nothing recorded it,
            and left to escape it would end the drive with no trace rather than zero the run
            with the evidence.

            It lands on the request whose reply was refused: the one the session sent
            last, whose event has not been emitted, so its sent envelope and send time are
            still in the driver's maps. A fault raised after that request's event was
            emitted is known only at the trajectory's boundary and lands on the session's
            next unrecorded turn."""
            error, reason = f"{fault.category}: {fault}", _fault_reason(fault, RequestAttempt())
            rid = last_sent.get(item.request_id)
            sent = stamped_inputs.pop(rid, None) if rid is not None else None
            if sent is None:
                _record_session_fault(item, tenant_id, due, error, reason, evidence=fault.extras)
                return
            _record_fault(sent, error, reason, due=due, t_send=submitted.pop(rid, None), evidence=fault.extras)

        def _drop_session(item, tenant_id: str, exc: SandboxGone) -> None:
            """Drop a session whose sandbox died after it started, and keep driving the rest.

            The framework's own ``finally`` has already torn the container down, so
            nothing leaks. The turns the session served stay in the trace as the server's
            work, and nothing is recorded for the turns it never reached, so the drop
            neither zeroes the SLO gate nor makes a calibration serving refuse itself: it
            measures the shorter trajectory the sandbox allowed. No number of drops fails
            the drive, since each served agent drives its own container differently. Each
            drop and its reason go into ``dropped_sessions``, which the drive role writes
            into the trace's ``END.json``. A sandbox that never started is not dropped:
            ``SandboxNotStarted`` fails the drive."""
            reason = str(exc)
            if len(reason) > DROPPED_REASON_MAX_CHARS:
                reason = reason[:DROPPED_REASON_MAX_CHARS - 3] + "..."
            dropped.append(DroppedSession(session_id=item.request_id, tenant_id=tenant_id, reason=reason))
            log.warning("driver: session %s dropped on a dead sandbox, the rest keep driving: %s",
                        item.request_id, exc)

        async def _run_tenant(tenant, items) -> None:
            workload = tenant.workload
            seed = tenant_seed(drive_seed, tenant.id)
            starts = workload.arrival.schedule(len(items), seed)
            output_cap = workload.output_length_cap.max_tokens()
            inter_turn_delay = workload.inter_turn_delay

            async def _one(i: int) -> None:
                item = items[i]
                due = t_start + starts[i]
                # Open-loop session arrival, cut short at the wall or the scheduled stop,
                # whichever comes first. A session whose arrival is past one would otherwise
                # sleep out its own schedule before recording anything, and the drive lasts
                # until the last arrival rather than until the bound -- the base loop wakes
                # its arrivals on the drive's stop event for the same reason.
                bounds = [due, *(() if wall is None else (wall.t,)),
                          *(() if stop is None else (stop.t_stop,))]
                await self._sleep_until(min(bounds))
                # Past the scheduled stop the session has no request to issue, so it is never
                # opened, and nothing is recorded: nothing was asked of the server.
                if stop is not None and stop.past_stop:
                    log.info("driver: session %s arrives past the scheduled stop; not opened",
                             item.request_id)
                    return
                # Past the wall nothing new starts: a trajectory that has not begun is
                # recorded as a fault rather than opened and cut a moment later.
                if wall is not None and wall.expired:
                    _record_wall_fault(item, tenant.id, due)
                    return
                ctx = self.harness_context(tenant.id, max_output_tokens=output_cap)

                async def _tenant_llm_call(env: Envelope) -> ArtifactResponse:
                    return await _llm_call(env, output_cap, inter_turn_delay)

                run = self.framework.run_trajectory(
                    seed_item=item, llm_call=_tenant_llm_call,
                    tool_runner=self.tool_runner, on_event=_aev,
                    ctx=ctx, max_turns=self.max_turns)
                # Both bounds hold the whole trajectory, not just its artifact calls: a
                # framework blocked in a tool call has to be cut by the same clock. The wall
                # stays around the cut, because cancelling a trajectory waits for its cleanup,
                # a container teardown on a coding task, and one that wedges must not hold
                # the drive open with no trace.
                bounded = run if stop is None else stop.run_until_cut(run)
                try:
                    await (bounded if wall is None else wall.cut(bounded))
                except (_TrajectoryAborted, _CutReached, DriveWallClockReached):
                    # An artifact fault already recorded on the turn that hit it, the cut
                    # (which records its own turn), or the stop reached between turns.
                    pass
                except _WallExpired:
                    _record_wall_fault(item, tenant.id, due)
                except ArtifactFault as f:
                    _record_framework_fault(item, tenant.id, due, f)
                except SandboxGone as exc:
                    _drop_session(item, tenant.id, exc)

            results = await asyncio.gather(*[_one(i) for i in range(len(items))],
                                           return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException):
                    raise r

        # return_exceptions=True so a fault in one trajectory lets its siblings finish
        # (no orphaned coroutines); then surface the first real fault.
        results = await asyncio.gather(*[_run_tenant(t, items) for t, items in loaded],
                                       return_exceptions=True)
        for r in results:
            if isinstance(r, BaseException):
                raise r
        self.dropped_sessions = tuple(dropped)
        if dropped:
            log.warning("driver: %d of %d session(s) dropped on a dead sandbox: %s", len(dropped),
                        sum(len(items) for _, items in loaded), ", ".join(d.session_id for d in dropped))


def driver_for(agent, *, request_timeout_sec: float, max_context_len: int | None = None,
               chat_template_kwargs: Mapping[str, TemplateArg] | None = None,
               drive_timeout_sec: float | None = None,
               gpu_kind: str | None = None) -> Driver:
    """The driver a task's traffic is driven with, built whole from the task.

    A task that declares an agent seam (``TaskSpec.agent``) is driven one trajectory
    per item, with the framework choosing the turns; any other task is driven by the
    scripted-turn base driver. Both a graded drive (the drive role, ``hls.drive.role``)
    and a calibration serving (``ReferenceEngine.serve_traffic``) build their driver
    here, so a calibration measures the traffic a graded run serves by construction.

    ``gpu_kind`` is the task's accelerator kind, used only when the agent's tool
    assets ask for a GPU on the driver.

    ``max_context_len`` and ``chat_template_kwargs`` are the task's context window and its
    model's chat-template pin, which the trajectory driver stamps on every
    ``HarnessContext``; the framework object, shared by every copy of a TaskSpec, holds
    neither.

    ``drive_timeout_sec`` is the stage's drive wall (the planner's) and wins when set;
    otherwise the agent's own (``AgentSpec.drive_wall_sec``) applies. A task with no
    agent seam has no wall unless the stage names one. The trajectory driver carries the
    agent's scheduled stop and stall budget, and refuses a wall before the stop's own
    (ValueError): it would fire during the drive's planned ending and record every
    trajectory the stop meant to cut cleanly as a fault.
    """
    if agent is None:
        return Driver(request_timeout_sec=request_timeout_sec,
                      drive_timeout_sec=drive_timeout_sec)
    stop = agent.scheduled_stop()
    if drive_timeout_sec is None:
        drive_timeout_sec = agent.drive_wall_sec()
    elif stop is not None and drive_timeout_sec < agent.drive_wall_sec():
        raise ValueError(f"a drive wall of {drive_timeout_sec:g}s is before the {agent.drive_wall_sec():g}s "
                         f"the task's scheduled stop runs behind (stop {stop.wall_clock_sec:g}s, cut "
                         f"{stop.cut_grace_sec:g}s later)")
    assets = agent.assets
    return TrajectorySessionDriver(
        framework=agent.framework, tool_runner=agent.tool_runner, max_turns=agent.max_turns,
        max_concurrent_llm=agent.max_concurrent_llm, max_context_len=max_context_len,
        chat_template_kwargs=chat_template_kwargs,
        assets_memory_mb=assets.memory_request() if assets is not None else None,
        assets_gpu=assets is not None and bool(assets.gpu),
        gpu_kind=gpu_kind if assets is not None else None,
        request_timeout_sec=request_timeout_sec, drive_timeout_sec=drive_timeout_sec,
        scheduled_stop=stop, stall_timeout_sec=agent.stall_timeout_sec)
