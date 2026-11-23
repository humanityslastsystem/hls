"""``hls follow``: an instance's events as their writers append them, read from the store.

Every writer of an instance appends to a log of its own (``hls.pipeline.events``): the
orchestrator of each lease generation, ``orchestrator/gen-<n>/events.jsonl``; the runner of each
allocation submission, ``allocations/<k>/submissions/<n>/events.jsonl``; and each stage attempt,
``stages/<stage>/attempts/<n>/events.jsonl``. ``EventTail`` reads each log on from where it last
stopped and returns only whole lines. A line whose newline is not written yet is the torn end of
an append in progress: it is kept back until its newline arrives, and never parsed. ``follow``
prints each poll's new events in time order, and stops once the instance has ended (its latest
plan has an outcome) and no orchestrator holds its lease. That is checked before the poll's read,
so the events an orchestrator writes before it lets go are printed too.

It only reads: it takes no lease and writes nothing, so it can follow an instance that an
orchestrator on another login node drives."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from hls.pipeline.plan import Plan, PlanStore
from hls.store.base import StoreError
from hls.store.layout import InstanceLayout
from hls.store.lock import LockRecord, instance_lease
from hls.store.posix import PosixStore

#: An instance's logs, as globs under its directory.
LOG_GLOBS = ("orchestrator/*/events.jsonl", "allocations/*/submissions/*/events.jsonl",
             "stages/*/attempts/*/events.jsonl")
#: How an event's time is printed when it has none.
_NO_TIME = "????-??-?? ??:??:??"


@dataclass(frozen=True)
class TailEvent:
    """One whole line of one of an instance's logs."""

    #: The log it came from, for people: "orchestrator gen 1", "allocation 0 submission 1",
    #: "stage verify attempt 2".
    source: str
    #: The event as its writer wrote it: ``at``, ``kind`` and its fields.
    event: dict[str, Any]
    #: Where its line begins in its log, in bytes. A log is only appended to, so with ``source`` it
    #: names the event on every read of the logs.
    offset: int


def log_source(parts: tuple[str, ...]) -> str | None:
    """The name of the log whose path under the instance's directory is *parts*; None for a
    file that is not one of the instance's logs."""
    match parts:
        case ("orchestrator", gen, "events.jsonl") if gen.startswith("gen-") and gen[4:].isdigit():
            return f"orchestrator gen {int(gen[4:])}"
        case ("allocations", k, "submissions", n, "events.jsonl") if k.isdigit() and n.isdigit():
            return f"allocation {k} submission {n}"
        case ("stages", stage, "attempts", n, "events.jsonl") if n.isdigit():
            return f"stage {stage} attempt {n}"
    return None


def _time(event: dict[str, Any]) -> float | None:
    """The event's ``at``, when it is a number."""
    at = event.get("at")
    return float(at) if isinstance(at, (int, float)) and not isinstance(at, bool) else None


def _event(line: bytes) -> dict[str, Any] | None:
    """The event a whole line holds; None for a blank line or one that is not a JSON object."""
    if not line.strip():
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


class EventTail:
    """Reads an instance's logs on from where each last stopped."""

    def __init__(self, root: Path) -> None:
        #: The instance's directory.
        self.root = root
        self._offsets: dict[Path, int] = {}
        self._pending: dict[Path, bytes] = {}

    def _logs(self) -> list[tuple[Path, str]]:
        found = []
        for pattern in LOG_GLOBS:
            for path in self.root.glob(pattern):
                source = log_source(path.relative_to(self.root).parts)
                if source is not None and path.is_file():
                    found.append((path, source))
        return sorted(found)

    def poll(self, skip: Callable[[Path], bool] = lambda path: False) -> list[TailEvent]:
        """Every whole line written to any of the logs since the last poll, in time order: its
        ``at``, then its log's name, then its place in the log. A line that is not a JSON object
        is left out, as ``hls.pipeline.events.read_events`` leaves it: the logs are diagnostic.
        A log whose path *skip* refuses is not read."""
        found: list[tuple[float, str, int, TailEvent]] = []
        for path, source in self._logs():
            if skip(path):
                continue
            start = self._offsets.get(path, 0)
            try:
                with path.open("rb") as f:
                    f.seek(start)
                    data = f.read()
            except OSError:
                continue
            if not data:
                continue
            self._offsets[path] = start + len(data)
            held = self._pending.pop(path, b"")
            whole, newline, rest = (held + data).rpartition(b"\n")
            if rest:
                self._pending[path] = rest
            if not newline:
                continue
            offset = start - len(held)
            for line in whole.split(b"\n"):
                event = _event(line)
                if event is not None:
                    when = _time(event)
                    found.append((0.0 if when is None else when, source, offset,
                                  TailEvent(source, event, offset)))
                offset += len(line) + 1
        found.sort(key=lambda f: f[:3])
        return [f[3] for f in found]

    def unfinished(self) -> list[str]:
        """The names of the logs whose last line has no newline yet."""
        return sorted(source for path, source in self._logs() if self._pending.get(path))


def lease_held(record: LockRecord | None, *, now: float) -> bool:
    """Whether the lease generation *record* still holds its instance at *now*: it is not
    released and was renewed within its TTL. A lease never taken (None) holds nothing."""
    return record is not None and not record.released and now - record.heartbeat_at <= record.ttl_sec


def _held(store: PosixStore, plan: Plan, *, now: float) -> bool:
    """Whether an orchestrator holds *plan*'s instance at *now*. A lease that cannot be read
    counts as held, so the instance is followed on rather than left early."""
    try:
        record = instance_lease(store, plan.layout, ttl_sec=plan.settings.run.lease_ttl_sec).current()
    except (StoreError, OSError, ValidationError):
        return True
    return lease_held(record, now=now)


def printable(text: str) -> str:
    """*text* as it is when every character of it prints, else as JSON: an event can carry what
    a contestant's program wrote (an agent's output in its build's events), and a newline in it
    would print a line of its own, an escape sequence would drive the terminal."""
    return text if text.isprintable() else json.dumps(text)


def event_line(found: TailEvent) -> str:
    """How ``follow`` prints an event: its time, its log, its kind and its other fields, a
    string as it is (``printable``) and anything else as JSON."""
    event = found.event
    at = _time(event)
    when = _NO_TIME
    if at is not None:
        try:
            when = datetime.fromtimestamp(at).strftime("%Y-%m-%d %H:%M:%S")
        except (OverflowError, OSError, ValueError):
            pass
    kind = event.get("kind")
    fields = "".join(f" {printable(k)}={printable(v) if isinstance(v, str) else json.dumps(v, sort_keys=True)}"
                     for k, v in sorted(event.items()) if k not in ("at", "kind"))
    shown = printable(kind) if isinstance(kind, str) else json.dumps(kind)
    return f"{when} {printable(found.source)}: {shown}{fields}"


def follow(instance: str, *, store: PosixStore, echo: Callable[[str], None] = print,
           sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.time,
           poll_sec: float = 2.0) -> int:
    """Print the events of *instance* as they are written, until it has ended and no
    orchestrator holds it; then say how it ended. Return 0, or 2, saying so, when the store holds
    no such instance."""
    try:
        layout = InstanceLayout(instance)
    except ValueError:
        echo(f"{instance!r} is not an instance's store key")
        return 2
    plans = PlanStore(store, layout)
    plan = plans.latest()
    if plan is None:
        echo(f"no such instance under {store.root}")
        return 2
    tail = EventTail(store.path(layout.root))
    said_unheld = False
    while True:
        plan = plans.latest() or plan
        held = _held(store, plan, now=clock())
        outcome = plan.outcome
        for found in tail.poll():
            echo(event_line(found))
        if outcome is not None and not held:
            for source in tail.unfinished():
                echo(f"{printable(source)}: its last line was never finished and is not shown")
            echo(f"{plan.instance}: {outcome.status}: {printable(outcome.reason)}")
            return 0
        if outcome is None and not held:
            if not said_unheld:
                echo(f"{plan.instance}: no orchestrator holds it; hls resume {plan.instance} takes it over")
                said_unheld = True
        else:
            said_unheld = False
        sleep(poll_sec)
