"""Cutting the prefixes a graded run resumes from, offline against the baseline.

One pass per corpus task. The loop runs from turn 0 exactly as a graded trajectory
does, and whenever the prompt crosses the next depth target the conversation is
frozen and the sandbox's state is captured beside it. One pass therefore yields every
band that task can reach, and the per-turn growth curve that says where the bands
should have been.

Not a graded path. Nothing here runs during a run: the prefixes it writes are staged
like the container images, and the corpus declares their digests.

What this module does not do
----------------------------

It does not capture the filesystem itself. Fifty commands in, the model's next command
refers to files it created and builds it ran, and a conversation alone resumes into a
sandbox that has none of them. Capturing that is ``capture_state``, which the caller
supplies, because where a container's writes live is the substrate's business and not
this module's. Called once per band with the session's own identifiers, it returns the
directory holding them.

It does not stop the container either, and that is why a pass yields ``Cut`` objects
rather than prefixes. The directory keeps being written until the caller tears the
container down, so a prefix -- which carries a manifest that has to match what a
resumed session finds -- can only be made afterwards. ``Cut.seal`` makes one.

Depth targets are token counts
------------------------------

Not turn counts. Per-turn growth varies several-fold across tasks and across a single
trajectory, so a turn count names a different depth on each. The prompt is counted
harness-side through the checkpoint's own tokenizer, the same count the graded loop
budgets its context against.

A target is a floor, not an exact depth: a turn's observation can be tens of thousands
of tokens, so the prompt jumps past a target rather than landing on it. The prefix
records both the target it was cut for and the depth it actually reached.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from tasks.agentic_coding import prefix as prefix_format
from tasks.agentic_coding.mini_swe_loop import MiniSweLoop, wire_messages

log = logging.getLogger(__name__)

#: The depths, in prompt tokens, a pass cuts prefixes at. One per row that resumes, so
#: the graded drive carries a session at each.
#:
#: Not a range the corpus cannot reach: a Terminal-Bench trajectory gets deep on its
#: own given turns.
#:
#: Turns are what it does not get. A drive issues requests for 45 minutes, so a
#: session starting at turn 0 spends the whole window climbing and the deep bands are
#: out of reach inside it whatever the tasks are. What a pre-roll buys is the head
#: start, not depth the corpus lacks -- which is why these are bands to resume from
#: rather than a target to climb to.
#:
#: The ladder goes to 600k because depth is nearly free per unit. A turn at depth
#: costs barely more than a shallow one, so total cost grows close to linearly and
#: each extra 100k is the cheapest thing this pass buys.
#:
#: The top is bounded by what a session can still do after resuming: 600k is the
#: deepest band that leaves the 1,048,576-token window room for the ten turns a deep
#: session serves (``DEEP_TURN_BUDGET``).
#:
#: A task that stops short takes the deepest band it did reach, and the pin records
#: the depth actually cut.
DEPTH_TARGETS = (200_000, 400_000, 600_000)


#: Why a pass stopped when the caller asked for a fixed number of bands rather than
#: for the whole trajectory. Its own reason so a short pass is never read as a task
#: that ran out of work.
STOP_BANDS_CUT = "bands_cut"


#: Why a pass stopped on its wall-clock bound. Its own reason, because a pass that ran
#: out of time and one that reached its target produce the same artifact and must not
#: read the same in the record.
STOP_DEADLINE = "deadline"


#: Why a pass stopped on its spend. Its own reason for the same purpose as
#: ``STOP_DEADLINE``, and distinct from it because slow and expensive are different
#: things to have been: a pass on a metered server can spend its budget in an hour
#: while another runs six hours on cached prompts for a few dollars, and the fix for
#: one is not the fix for the other.
STOP_COST = "cost"


class _BandsCut(Exception):
    """Ends a pass that has cut every band its caller asked for."""


class _Deadline(Exception):
    """Ends a pass that has reached its wall-clock bound."""


class _Spent(Exception):
    """Ends a pass that has reached its cost bound."""


@dataclass
class TurnDepth:
    """How deep the conversation was before one turn was sent.

    The growth curve is the sequence of these over a trajectory. It is the measurement
    the depth targets are chosen from, and it is why the pre-roll runs even for a task
    that reaches no target."""

    #: Turns served before this one.
    turn: int
    #: Prompt tokens, counted harness-side.
    prefill_tokens: int


@dataclass(frozen=True)
class Cut:
    """A conversation frozen at a depth, and where the state that goes with it is.

    Not a ``Prefix``, and the missing piece is the state manifest. A prefix carries
    one and ``prefix.load`` refuses a prefix whose manifest disagrees with the
    directory; the directory is an overlay upper the pass's own container is still
    writing, and it keeps writing until it is stopped -- the exec server flushes,
    the runtime tears its mounts down. A manifest taken here would describe a
    directory that has moved on by the time a graded run reads it, and the refusal
    would land at the first resumed trajectory rather than at the pass.

    So the two halves are taken at the two moments each one is right at. The depth,
    the conversation and the counters freeze here, because the instant is the only
    thing that cannot be recovered afterwards. The manifest is taken in ``seal``,
    once the caller has stopped the container and the directory has stopped moving."""

    task_name: str
    depth_tokens: int
    prefill_tokens: int
    turns: int
    commands_run: int
    consecutive_format_errors: int
    checkpoint: str
    image: str
    #: The directory holding what the container wrote, as ``capture_state`` named it.
    #: Read at seal time and not before.
    state_dir: str
    messages: tuple[dict, ...]

    def seal(self) -> prefix_format.Prefix:
        """This cut as a prefix, over the state directory as it now stands.

        Call it after the container has stopped. Called while it is running, it
        records a manifest the directory will not match by the time a run reads it,
        which is the whole reason a cut is not a prefix already."""
        return prefix_format.build(
            task_name=self.task_name, depth_tokens=self.depth_tokens,
            prefill_tokens=self.prefill_tokens, turns=self.turns,
            commands_run=self.commands_run,
            consecutive_format_errors=self.consecutive_format_errors,
            checkpoint=self.checkpoint, image=self.image,
            state_manifest=prefix_format.manifest_of(self.state_dir),
            messages=list(self.messages))


@dataclass
class PrerollResult:
    """What one task's pass produced."""

    task_name: str
    #: One cut per depth target the trajectory reached, in the order it reached them,
    #: each awaiting the seal that makes it a prefix. A task whose trajectory ends
    #: early supplies fewer than were asked for, which is a fact about the task rather
    #: than a failure: a Terminal-Bench trajectory ends when its task is done.
    cuts: list[Cut] = field(default_factory=list)
    #: Depth before every turn, so the growth rate can be read off afterwards.
    depths: list[TurnDepth] = field(default_factory=list)
    #: Why the trajectory stopped, as the loop names it.
    stop_reason: str = ""

    def reached(self) -> list[int]:
        """The depth targets this task can supply."""
        return [c.depth_tokens for c in self.cuts]

    def growth_per_turn(self) -> float:
        """Mean prompt tokens added per turn, or 0.0 with fewer than two turns.

        The number the depth targets are computed from. A mean over the trajectory
        rather than a fit: the growth is dominated by how much output each command
        returned, which is not smooth, and a slope through it would read as more
        precision than the quantity has."""
        if len(self.depths) < 2:
            return 0.0
        first, last = self.depths[0], self.depths[-1]
        return (last.prefill_tokens - first.prefill_tokens) / (last.turn - first.turn)


def declared_prefixes(prefixes: list[prefix_format.Prefix],
                      depths: dict[str, int]) -> dict[str, str]:
    """The ``prefixes`` pin for a set of passes, ready to paste into ``datasets.py``.

    Takes sealed prefixes rather than the passes that cut them, because a cut has no
    digest until it is sealed and sealing reads the state directory: a function that
    sealed on the caller's behalf would decide, silently, at what moment the
    filesystem was described.

    ``depths`` names the band each task runs at, chosen by hand from what its pass
    reached (``PrerollResult.reached()``). By hand and not by rule: there are three or
    four of these tasks, they are the only source of deep prompts in the corpus, and
    how the deep end of the workload is composed is a decision about the benchmark
    rather than something to derive from a hash. A task left out of ``depths`` is left
    out of the pin, which is a cold start.

    One depth per task, because the corpus cannot carry a task twice: ``row_id`` is the
    task name and the split partition hashes it, so ``<task>@<depth>`` rows could land
    on opposite sides of a partition whose dev half is published to the contestant.

    Refuses a depth no pass produced. That is the one error this cannot let through
    quietly: the pin would name a prefix nothing ever wrote, and the failure would
    surface as a missing file at the first resumed trajectory of a graded run."""
    by_task: dict[str, list[prefix_format.Prefix]] = {}
    for p in prefixes:
        by_task.setdefault(p.task_name, []).append(p)
    pin: dict[str, str] = {}
    for task_name, depth in sorted(depths.items()):
        cut_for_task = by_task.get(task_name)
        if not cut_for_task:
            raise ValueError(
                f"no prefix for {task_name!r} at any depth; its prefix was never cut, "
                f"either because no pass ran for it or because the trajectory ended "
                f"before the first band")
        chosen = [p for p in cut_for_task if p.depth_tokens == depth]
        if not chosen:
            raise ValueError(
                f"{task_name!r} was asked for the {depth}-token band but its pass "
                f"reached {sorted(p.depth_tokens for p in cut_for_task)}; the "
                f"trajectory ended before it got there")
        pin[task_name] = chosen[0].digest
    return pin


def next_target(targets: list[int], cut: list[int]) -> int | None:
    """The shallowest target not yet cut, or None when every one has been.

    Targets are taken in order, so a prompt that jumps past two at once cuts the
    deeper of them and leaves the shallower unfilled rather than cutting one prefix
    twice. The band that went unfilled is visible in ``PrerollResult.reached``."""
    remaining = [t for t in sorted(targets) if t not in cut]
    return remaining[0] if remaining else None


class Preroll:
    """One task's pass: drive the baseline from turn 0 and cut prefixes on the way.

    ``capture_state`` is the sandbox half, supplied by the caller:
    ``(task_name, depth_tokens, session_id) -> str``, returning the directory holding
    what the container wrote. It runs while the container is still alive, so what it
    names is the state that produced the conversation frozen alongside it, and it
    must leave that directory on the filesystem the run will read it from -- see
    ``prefix``'s module docstring for why it cannot be copied there afterwards.

    What a pass returns is ``Cut`` objects, which become prefixes once the caller has
    stopped the container and called ``seal``. The split is the container's doing: the
    conversation can only be frozen while it runs and the directory can only be
    described once it has stopped.

    ``image`` is the base image the pass ran, recorded on every cut it takes: a
    resuming session starts from that image with the captured directory stacked under
    its own writable layer, so nothing is baked into a derivative.
    """

    def __init__(self, *, loop: MiniSweLoop, checkpoint: str, depth_targets: list[int],
                 capture_state: Callable[[str, int, str], str],
                 count_prompt_tokens: Callable, image: str = ""):
        self.loop = loop
        self.checkpoint = checkpoint
        self.image = image
        self.depth_targets = sorted(depth_targets)
        self.capture_state = capture_state
        self.count_prompt_tokens = count_prompt_tokens

    async def run(self, *, task_name: str, problem_statement: str, session_id: str,
                  base_metadata: dict, complete, run_tool,
                  emit: Callable | None = None,
                  stop_after_band: int | None = None,
                  deadline_sec: float | None = None,
                  cost_budget_usd: float | None = None,
                  spent_usd: Callable[[], float] | None = None) -> PrerollResult:
        """Drive one task to termination, cutting a prefix at each depth it reaches.

        ``emit`` receives every finished turn, as the graded driver's does. The pass
        scores nothing, so it needs none of it; what a caller wants it for is the
        telemetry a pass is the only cheap chance to collect, since it is the one
        thing that drives this loop to depth on the baseline server.

        ``stop_after_band`` ends the pass once that many bands are cut, for taking one
        prefix and checking it resumes before spending the rest of the pass on a
        mechanism that might not work.

        ``deadline_sec`` bounds the pass in wall clock and ``cost_budget_usd`` bounds
        what it spends, read through ``spent_usd``. Two bounds rather than one because
        a pass can be slow without being expensive and expensive without being slow: on
        a metered server a trajectory with long replies spends its budget in an hour,
        while one on cached prompts runs all day for a few dollars.

        Either bound firing cuts a prefix at whatever depth the conversation has
        reached and stops, rather than returning nothing: a pass that ran out still
        produces a usable band, shallower than the one asked for, and the prefix
        records the depth actually reached. A bound whose firing destroys the work is a
        bound nobody can afford to set.

        A cost budget with no ``spent_usd`` is refused rather than ignored, because a
        budget nothing reports against never fires and the pass would run to the wall
        looking bounded."""
        if cost_budget_usd is not None and spent_usd is None:
            raise ValueError(
                "a cost budget was set but the pass has no way to read what it has "
                "spent; pass spent_usd alongside cost_budget_usd")
        result = PrerollResult(task_name=task_name)
        started = time.monotonic()

        async def before_turn(agent) -> None:
            wire = wire_messages(agent.messages)
            counted = await self.count_prompt_tokens(wire)
            if counted is None:
                raise RuntimeError(
                    "the pre-roll cannot count the prompt, so it cannot tell which "
                    "depth band a turn is in; run it where the checkpoint is mounted")
            turn = agent.model.recorder.turns
            result.depths.append(TurnDepth(turn=turn, prefill_tokens=int(counted)))
            # Cut where we are rather than returning nothing. The conversation and the
            # sandbox are captured at the same instant either way, which is the
            # invariant a resumed session depends on.
            out_of_time = (deadline_sec is not None
                           and time.monotonic() - started >= deadline_sec)
            out_of_money = (cost_budget_usd is not None
                            and spent_usd() >= cost_budget_usd)
            if out_of_time or out_of_money:
                result.cuts.append(self._cut(agent, wire, task_name, int(counted),
                                             int(counted), session_id))
                raise _Deadline if out_of_time else _Spent
            target = next_target(self.depth_targets, result.reached())
            if target is None or counted < target:
                return
            result.cuts.append(self._cut(agent, wire, task_name, target,
                                         int(counted), session_id))
            if stop_after_band is not None and len(result.cuts) >= stop_after_band:
                raise _BandsCut

        try:
            summary = await self.loop.run(
                session_id=session_id, problem_statement=problem_statement,
                base_metadata=base_metadata, complete=complete, run_tool=run_tool,
                emit=emit or _ignore_event, max_turns=_UNBOUNDED_TURNS,
                before_turn=before_turn)
            result.stop_reason = str(summary["stop_reason"])
        except _BandsCut:
            result.stop_reason = STOP_BANDS_CUT
        except _Deadline:
            result.stop_reason = STOP_DEADLINE
        except _Spent:
            result.stop_reason = STOP_COST
        return result

    def _cut(self, agent, wire: list[dict], task_name: str, target: int,
             counted: int, session_id: str) -> Cut:
        """Freeze the conversation and name the sandbox directory that produced it.

        The directory is named, not read. See ``Cut`` for why the manifest waits."""
        state_dir = self.capture_state(task_name, target, session_id)
        log.info("preroll %s: cut at %d tokens (target %d) after %d turns, state in "
                 "%s", task_name, counted, target, agent.model.recorder.turns,
                 state_dir)
        return Cut(
            task_name=task_name, depth_tokens=target, prefill_tokens=counted,
            turns=agent.model.recorder.turns,
            commands_run=agent.model.recorder.commands_run,
            consecutive_format_errors=agent.n_consecutive_format_errors,
            checkpoint=self.checkpoint, image=self.image,
            state_dir=str(state_dir), messages=tuple(dict(m) for m in wire))


#: Turn ceiling for a pre-roll. Far above the task's own, because the pre-roll is not
#: measuring a server and has to reach the deepest band it can; what stops it is the
#: trajectory ending, the context window, or a target list it has exhausted.
_UNBOUNDED_TURNS = 100_000


async def _ignore_event(turn: dict) -> None:
    """The pre-roll records no trace. Nothing scores it, and the conversation it is
    building is captured at the resume points instead."""


__all__ = ["DEPTH_TARGETS", "Cut", "Preroll", "PrerollResult", "STOP_BANDS_CUT",
           "STOP_COST",
           "STOP_DEADLINE", "TurnDepth", "declared_prefixes", "next_target"]
