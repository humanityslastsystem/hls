"""Append-only event logs, one file per writer: a stage attempt, an allocation submission's
runner, or the orchestrator holding one lease generation. No file is appended to from two
nodes; readers merge the files of an instance by time.

Events are diagnostic. The plan and the stage manifests are what the orchestrator decides
from, so a damaged log line costs a reader one event and nothing else."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class EventLog:
    """Appends one JSON line per event to a file."""

    def __init__(self, path: Path, *, clock: Callable[[], float] = time.time,
                 scrub: Callable[[str], str] | None = None) -> None:
        #: The log file.
        self.path = path
        #: Time source for each event's ``at``.
        self.clock = clock
        #: Applied to each event's whole JSON line before it is written, such as a scrubber that
        #: replaces every value of the secrets file (records are scrubbed when written); None
        #: writes the line as it is.
        self.scrub = scrub
        self._guard = threading.Lock()

    def emit(self, kind: str, /, **fields: Any) -> None:
        """Append an event of *kind*. Values that are not JSON types are written as text.
        The fields ``at`` and ``kind`` belong to the log, and passing either raises
        ValueError."""
        reserved = sorted({"at", "kind"} & fields.keys())
        if reserved:
            raise ValueError(f"event fields cannot be named {', '.join(map(repr, reserved))}")
        line = json.dumps({"at": self.clock(), "kind": kind, **fields}, sort_keys=True, allow_nan=False,
                          default=str)
        if self.scrub is not None:
            line = self.scrub(line)
        with self._guard:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")


def read_events(path: Path) -> list[dict[str, Any]]:
    """Every event in *path*, oldest first; [] when the file does not exist. A line that is
    not a JSON object, such as the torn end of an append, is skipped with a warning that
    names the file and the line."""
    try:
        lines = path.read_bytes().splitlines()
    except FileNotFoundError:
        return []
    events = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as e:
            log.warning("%s: line %d is not valid JSON, skipped: %s", path, number, e)
            continue
        if not isinstance(event, dict):
            log.warning("%s: line %d is not a JSON object, skipped", path, number)
            continue
        events.append(event)
    return events
