"""L1 -- AgentFramework ABC + AgentSpec.

The harness-side trajectory loop interface (design Sec.9). A concrete
``AgentFramework`` runs ONE trajectory end-to-end: it builds the prompt, calls
the artifact (LLM), parses tool calls from the response, routes each through
the harness ``ToolRunner``, appends the result to the conversation history,
and re-prompts the LLM until the model emits a final answer or ``max_turns``
is hit.

This ABC is harness-side because the Sec.9 trust invariant says the artifact
cannot execute tool calls itself -- if the artifact owned the loop it could
fake zero-latency tool calls and escape the timing budget.

Each concrete framework is task-specific (knows the model's tool-call format)
and lives in the task directory: e.g. ``tasks/deepresearch/framework.py`` hosts
``DeepResearchFramework``. Adding a new agentic task = adding a new file under
``tasks/``, not under ``src/hls/``.

Implementers emit one ``TraceEvent`` per LLM call with harness-recorded
send/recv timestamps. Use ``TraceEvent.extras`` to carry per-turn metadata
(``tool_calls``, ``tool_results``, ``final_answer``, ``stop_reason``) so the
metric and SLO can reconstruct the trajectory shape without re-parsing model
text.

``AgentSpec`` is the optional agentic seam on the TaskSpec. It is a plain frozen
dataclass (NOT a pydantic Spec) because it carries the live framework + tool
runner objects, which are task-module code, not serializable config -- it rides
on ``TaskSpec.agent`` as an excluded (rederivable) field. The drive reads it to select
the ``TrajectorySessionDriver`` instead of the plain ``Driver``
(``hls.spec.drivers.driver_for``).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from hls.core.model import ModelSpec
from hls.core.payloads import Envelope
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.spec.tools import ToolRunner


class AgentFramework(ABC):
    """Drives one agentic trajectory (L1, Sec.9).

    Loops LLM calls + tool calls until the model emits a final answer or
    ``max_turns`` is hit. Harness-side glue: subclasses own the parsing of the
    model's tool-call format, the conversation-history shape, and the
    invariant that tool calls go through the ``ToolRunner``, not the artifact.
    """

    @abstractmethod
    async def run_trajectory(
        self,
        seed_item,
        llm_call: Callable[[Envelope], Awaitable[ArtifactResponse]],   # artifact.submit
        tool_runner: ToolRunner,                                        # harness-side
        on_event: Callable[[TraceEvent], Awaitable[None]],
        ctx,                                                             # HarnessContext
        max_turns: int,
    ) -> None: ...

    async def warmup(self) -> None:
        """Load whatever the framework needs before traffic starts.

        The trajectory driver awaits this before it anchors its clock, beside the tool
        runner's own warmup, so a framework's load time is never a trajectory's
        latency. A framework that counts its own prompts loads a tokenizer here, which
        costs about a second and would otherwise land inside the first trajectory that
        finished a turn. Default: nothing to load."""
        return None

    def check_reply(self, text: str, tokens: "list[int] | None") -> "str | None":
        """Why this reply breaks the task's wire contract, or None if it does not.

        A framework that validates replies does it inside the call the driver hands it,
        which covers every reply it reads. It does not cover the one the scheduled stop
        cuts: that call never returns, so the driver records the event itself and asks
        here on the framework's behalf. Otherwise the only unchecked reply of a drive
        would be the one arriving in the window a server can predict.

        Returns a reason rather than raising, because the caller is a cancellation
        handler with a cut to finish recording; the driver turns the reason into a
        failed request. Default: nothing to check, which leaves the cut a cut."""
        return None

    def check_model(self, model: ModelSpec) -> None:
        """Refuse a model this framework cannot drive.

        ``TaskSpec`` calls this at construction with the task's ``ModelSpec``, so a
        framework that reads replies in one family's syntax can reject a checkpoint
        that writes another's instead of failing silently at run time. Raise
        ``ValueError`` naming the disagreement; the default accepts any model."""

    def traffic_config(self) -> dict:
        """The settings that shape the prompts and turn counts this framework produces,
        as plain JSON.

        The calibration fingerprint records them, so changing one marks a stored
        calibration stale. The default is every public scalar attribute, which is what a
        framework's knobs normally are; override when a setting is not a scalar, or when
        an attribute is state rather than a setting."""
        return {k: v for k, v in sorted(vars(self).items())
                if not k.startswith("_")
                and isinstance(v, (type(None), bool, int, float, str))}

    def extra_output_caps(self) -> set[int]:
        """Output-token caps this framework's loop stamps on a request besides the
        tenant's ``Workload.output_length_cap``.

        A loop that continues a reply cut at the output cap issues that continuation
        under its own, smaller budget, so a captured drive holds requests at more than
        one cap. Calibration keys a stored capture by its stamped cap to refuse one
        from a different workload; a framework reports its other legitimate caps here so
        that check accepts them. The values ride the fingerprint already, through
        ``traffic_config``; this only names which of those knobs are per-request caps a
        stored record may carry. Default: none, for a loop that stamps only the
        workload cap."""
        return set()


class AssetsRefused(Exception):
    """A tool bundle cannot be built as its task declares it: building it again from the same
    pinned sources and row ids fails the same way, such as a dev bundle that would reveal a
    test row. ``ToolAssets.build`` raises it, or a subclass, for such an error and for no
    other; anything else it raises may not happen again."""


class ToolAssets(ABC):
    """Data a task's tools load at run time, provisioned like weights and mounted
    per role at ``/data/tools/<name>``.

    Two variants exist: ``dev`` is what the build sandbox may see (built from the dev
    rows only), ``full`` is what the driver and the calibration members load. The
    builder runs on the provisioning host, where the Hub is reachable, and writes one
    self-contained directory per variant. ``identity`` names the pinned sources, so a
    changed source re-provisions and re-fingerprints the workload.

    ``memory_mb`` sizes the driver process for the loaded bundle; ``gpu`` asks for a
    GPU on the driver slot so the tool can encode there instead of on CPU.
    """

    name: str
    memory_mb: int = 0
    gpu: bool = False

    def memory_request(self) -> int | None:
        """``memory_mb`` as a slot request: zero means "no ask", i.e. the substrate
        default, which a request spells as ``None``."""
        return self.memory_mb or None

    @abstractmethod
    def identity(self) -> str:
        """Stable digest of every pinned source the bundle is built from."""

    @abstractmethod
    def hub_revisions(self) -> list[tuple[str, str]]:
        """Every hub repo ``build()`` reads, for either variant, with the revision it reads it
        at, as ``(repo, revision)``: the sources ``identity()`` names. Both bundles' keys stand
        for fixed bytes only when each revision is a commit, which the provision role checks
        (``hls.spec.hub``). Empty only for a bundle built from nothing on the hub."""

    @abstractmethod
    def build(self, variant: str, *, dev_row_ids: list[str], test_row_ids: list[str],
              out_dir: Path, hf_token: str | None = None) -> None:
        """Write the ``variant`` bundle into ``out_dir``. ``dev_row_ids`` and
        ``test_row_ids`` are the provisioned partitions (``Dataset.row_id`` values);
        a ``dev`` bundle must contain nothing that reveals a test row. Raise AssetsRefused
        for a bundle that cannot be built as declared.

        The ``full`` variant is cached by ``identity()`` alone, so its contents must
        depend only on the pinned sources; the row ids are there for the ``dev``
        variant, whose directory is keyed by the partition as well."""


class DriveWallClockReached(Exception):
    """Raised into a framework's ``llm_call`` when the drive's scheduled stop has
    passed, so the trajectory ends without issuing the turn.

    A clean ending, not a fault: the drive is bounded in wall-clock time and reaching
    that bound says nothing about the server. A framework that names its endings
    should name this one too -- upstream mini-swe-agent and its forks stamp an
    exception's class name as the trajectory's exit status, so ``DriveWallClockReached``
    is the string such a loop maps. One that does not name it simply stops, and the
    driver records no fault either way."""


#: Seconds the drive wall sits past a scheduled stop's own cut. It is the backstop for
#: the stop failing to fire, so it must be late enough that a drive stopping on
#: schedule never reaches it.
#:
#: What it is room for, since a margin whose purpose is unrecorded is one nobody can
#: safely change. The cut instant is when the last stream is cancelled, not when the
#: drive is done: each cut request is then priced, which for the latency SLO means
#: adding the phantom token that completes at the cut, each trajectory unwinds through
#: its framework's cleanup, which for a coding task tears down a container per session,
#: and the trace is flushed. All of that happens after the clock the stop is measured
#: against, so a wall set exactly at the cut would fire during the drive's own orderly
#: shutdown and file it as a failure.
BACKSTOP_MARGIN_SEC = 120.0


@dataclass(frozen=True)
class ScheduledStop:
    """The bounded wall-clock window a trajectory drive runs in.

    Two instants, both measured from the drive's own start. At ``wall_clock_sec`` no
    further request is issued and each session ends cleanly. ``cut_grace_sec`` later
    every stream still open is cancelled, its session's sandbox dies with it, and each
    cut request is scored on what it streamed. The drive never exceeds the sum, for
    any server.

    Distinct from ``AgentSpec.drive_timeout_sec``, which records the remaining work as
    faults and so zeroes the run. This records none: a drive that runs out of window
    served everything it was asked for up to the window."""

    #: Seconds from the drive's start after which no request is issued.
    wall_clock_sec: float
    #: Seconds after that before open streams are cut.
    cut_grace_sec: float

    def stop_at(self, t0: float) -> float:
        """Harness time after which no request is issued, for a drive anchored at
        ``t0``."""
        return t0 + self.wall_clock_sec

    def cut_at(self, t0: float) -> float:
        """Harness time every open stream is cut, for a drive anchored at ``t0``."""
        return t0 + self.wall_clock_sec + self.cut_grace_sec

    def backstop_timeout_sec(self) -> float:
        """The drive wall this stop runs behind (``BACKSTOP_MARGIN_SEC``)."""
        return self.wall_clock_sec + self.cut_grace_sec + BACKSTOP_MARGIN_SEC


@dataclass(frozen=True)
class AgentSpec:
    """The agentic seam declared on a TaskSpec (``TaskSpec.agent``).

    Carries the harness-side framework + tool runner plus the per-trajectory
    budget. ``max_concurrent_llm`` caps in-flight artifact (LLM) submits while
    leaving tool waits unbounded, so many trajectories can overlap on a slow tool
    (e.g. a corpus search that encodes the query) without exceeding the contestant's
    LLM batch headroom. ``None`` means no admission control (rely on the ArrivalProcess
    to pace).

    A frozen dataclass rather than a pydantic ``Spec``: it holds the live
    framework + tool_runner objects (task-module code, not serializable
    config), so it rides on the TaskSpec as a provenance-excluded field -- the
    framework/tool wiring is rederivable from the task module.

    ``assets`` declares data the tools load at run time (a corpus, an index);
    provisioning builds and mounts it per role, see ``ToolAssets``.

    ``drive_timeout_sec`` is the total wall-clock bound on a drive of this task's
    traffic, or ``None`` when the task sets none (the drive stage then gives the drive
    ``hls.pipeline.pipelines.OPEN_DRIVE_SEC``). ``max_turns`` bounds one trajectory and
    the drive's per-request grace (``hls.pipeline.pipelines.REQUEST_TIMEOUT_SEC``) bounds
    one turn, but a trajectory's turns are serial and a tool wait is neither, so neither
    bounds the drive: an artifact slow enough runs until the job running the drive is
    killed, writing no trace and no score. Sized off the task's measured baseline, so it
    lives with the task rather than with the site every task shares. Reaching it stops
    the drive and records the undriven remainder as faults, so the run zeroes on the SLO
    gate and still produces the bound it was reaching for. Slowness inside the wall keeps
    costing continuously through the score; this stops only a drive that cannot finish.

    ``wall_clock_sec`` and ``cut_grace_sec`` are the opt-in scheduled stop
    (``ScheduledStop``), declared together or not at all. Where ``drive_timeout_sec``
    is the failure the harness resorts to, these are the drive's intended shape: a
    fixed window every server gets, whose end is not a fault. A task that declares
    them and no ``drive_timeout_sec`` gets the stop's backstop as its wall
    (``drive_wall_sec``).
    """

    framework: AgentFramework
    tool_runner: ToolRunner
    max_turns: int = 1
    max_concurrent_llm: int | None = None
    assets: ToolAssets | None = None
    drive_timeout_sec: float | None = None
    #: Seconds from the drive's start after which no request is issued, or ``None``
    #: for a drive with no scheduled stop. See ``ScheduledStop``.
    wall_clock_sec: float | None = None
    #: Seconds after ``wall_clock_sec`` before open streams are cut.
    cut_grace_sec: float | None = None
    #: Seconds a reply may go between two tokens before the request fails, or ``None``
    #: for no such bound.
    #:
    #: A stall is a failure, where a cut is not: the drive's window closing says
    #: nothing about a server, but a stream that delivered tokens and then stopped is
    #: one that stopped working. Armed only once a first token has arrived, since the
    #: silence before one is prefill plus queueing and the latency ratio prices it.
    #: Size it well above the longest gap a healthy server shows under load -- a
    #: preemption recomputing a long context is tens of seconds -- because this ends
    #: the run, not one request's credit.
    stall_timeout_sec: float | None = None

    def __post_init__(self) -> None:
        """Refuse half a scheduled stop, and times that bound nothing.

        The two are one declaration: a stop with no grace would cut the streams it
        just stopped issuing behind, and a grace with no stop bounds nothing. Checked
        here so a task that gets it wrong fails at import rather than an hour into a
        drive."""
        wall, grace = self.wall_clock_sec, self.cut_grace_sec
        if (wall is None) != (grace is None):
            raise ValueError(
                "a scheduled stop needs both wall_clock_sec and cut_grace_sec; got "
                f"wall_clock_sec={wall!r}, cut_grace_sec={grace!r}")
        if wall is not None and wall <= 0:
            raise ValueError(f"wall_clock_sec must be positive, got {wall!r}")
        if grace is not None and grace < 0:
            raise ValueError(f"cut_grace_sec cannot be negative, got {grace!r}")
        if self.stall_timeout_sec is not None and self.stall_timeout_sec <= 0:
            raise ValueError(
                f"stall_timeout_sec must be positive, got {self.stall_timeout_sec!r}")
        # The backstop must never fire before the ending the drive plans. One that did
        # would cancel the trajectories the stop was about to cut cleanly and record
        # every one as a fault, so a drive that ended exactly as designed would zero
        # and read as the contestant failing.
        #
        # Refused rather than quietly lifted. ``drive_wall_sec`` does lift a wall that sits
        # between the plan and the margin, which is applying ``BACKSTOP_MARGIN_SEC``;
        # replacing a wall that contradicts the stop would be something else -- a file
        # stating one bound while the run used another.
        if wall is not None and self.drive_timeout_sec is not None:
            planned_end = wall + grace
            if self.drive_timeout_sec <= planned_end:
                raise ValueError(
                    f"drive_timeout_sec={self.drive_timeout_sec!r} is at or before "
                    f"the ending this stop plans ({wall!r} + {grace!r} = "
                    f"{planned_end!r}s), so the backstop would fire first and record "
                    f"every cleanly cut trajectory as a fault. Leave it unset to take "
                    f"the stop's own backstop, or set it past {planned_end!r}.")

    def scheduled_stop(self) -> ScheduledStop | None:
        """This seam's scheduled stop, or ``None`` where the task declared none."""
        if self.wall_clock_sec is None:
            return None
        return ScheduledStop(wall_clock_sec=self.wall_clock_sec,
                             cut_grace_sec=self.cut_grace_sec)

    def drive_wall_sec(self) -> float | None:
        """The wall a drive of this seam runs under, the one the stage plans and the driver
        keeps: with a scheduled stop, its backstop, or ``drive_timeout_sec`` when that is
        later; without one, ``drive_timeout_sec``, or None (the stage's open wall)."""
        stop = self.scheduled_stop()
        if stop is None:
            return self.drive_timeout_sec
        return max(stop.backstop_timeout_sec(), self.drive_timeout_sec or 0.0)
