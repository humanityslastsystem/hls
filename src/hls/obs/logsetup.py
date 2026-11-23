"""Console logging setup for the HLS CLI.

One stderr handler on each tree this CLI owns (``OWNED_LOGGERS``), each line in one
console layout (``format_console_line``). Color is TTY-gated. Only the live console
rendering uses this layout; the records an instance writes are unaffected.

Verbosity is enforced on the HANDLER, not only on the logger. An owned tree may pin
its own level for its own reasons -- harbor holds its loggers at DEBUG so its
standalone file handler can capture everything and its console handler filters -- and
a level we set on an ancestor is simply not consulted when a descendant has one. The
handler is the one point every record must pass through regardless.

While a process holds a scrubber for what it records, as an orchestrator does, every line the
console formatter renders is scrubbed with it too (``scrub_console``), so the log says no more
than the records, as the runner's own log handler scrubs its lines."""
from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Callable
from typing import TextIO

#: Width the label column is left-justified to, so INFO / WARNING / ERROR align.
#: 8 == len("CRITICAL"), the longest standard level name, so every label pads.
_LABEL_W = 8

#: Each applied in turn to every line the console formatter renders (``scrub_console``).
_console_scrubs: list[Callable[[str], str]] = []


def scrub_console(scrub: Callable[[str], str]) -> Callable[[], None]:
    """Scrub every line the console formatter renders from now on with *scrub* too, on the console
    and in a rollout's ``run.log`` alike, until the returned function is called."""
    _console_scrubs.append(scrub)

    def stop() -> None:
        if scrub in _console_scrubs:
            _console_scrubs.remove(scrub)

    return stop


#: SGR sequences per level label, each coloring the label's foreground.
_RESET = "\033[0m"
_LABEL_SGR = {
    "DEBUG": "\033[2m",        # dim
    "INFO": "\033[32m",        # green
    "WARNING": "\033[33m",     # yellow
    "ERROR": "\033[31m",       # red
    "CRITICAL": "\033[1;31m",  # bold red
}


def supports_color(stream: TextIO) -> bool:
    """True when ANSI color is appropriate for ``stream``: a real TTY whose
    environment does not opt out. Honors NO_COLOR (https://no-color.org) and
    TERM=dumb. Reading these env vars is terminal-capability detection, not config
    (configuration comes only from hls.toml, and secrets from the file at ``site.secrets_path``)."""
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def format_console_line(ts: float, label: str, context: str, message: str,
                        *, color: bool = False) -> str:
    """The single console line layout: ``HH:MM:SS LABEL   context: message``.

    The log formatter passes the level name as the label and the logger name as the
    context. ``: message`` is omitted when ``message`` is empty. When ``color`` is set,
    only the (padded) label token is wrapped in its SGR."""
    when = time.strftime("%H:%M:%S", time.localtime(ts))
    lab = f"{label:<{_LABEL_W}}"
    if color and label in _LABEL_SGR:
        lab = f"{_LABEL_SGR[label]}{lab}{_RESET}"
    body = f"{context}: {message}" if message else context
    return f"{when} {lab} {body}"


class _ConsoleFormatter(logging.Formatter):
    """Render a LogRecord via ``format_console_line``: time from ``record.created``,
    label from the level name, context from the logger name, message from getMessage().

    An ``hls.`` prefix is stripped because it is on every record from this package and
    carries nothing. Harbor's names are left whole: they read
    ``harbor.utils.logger.harbor.environments...`` because harbor builds each logger as
    a child of its own root, and cutting that apart here would guess at a shape that is
    harbor's to define."""

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        context = name[len("hls."):] if name.startswith("hls.") else name
        line = format_console_line(record.created, record.levelname, context, record.getMessage(), color=self._color)
        for scrub in list(_console_scrubs):
            line = scrub(line)
        return line


#: Marker attribute on our handler so configure_logging can find + replace its own
#: handler (idempotent) rather than stacking across repeated CLI/test invocations.
_HANDLER_TAG = "_hls_console_handler"

#: The logger trees this CLI owns and renders. ``harbor`` is the bundled fork of Harbor, not a
#: third-party library, and inside an agent sandbox it is the only witness there is --
#: the container's exec server logs its own stdout through it. Without this its records
#: fall to the root logger's WARNING and are dropped.
OWNED_LOGGERS = ("hls", "harbor")


def attach_owned(handler: logging.Handler) -> None:
    """Add *handler* to every owned tree, replacing any prior handler of ours.

    Callers hand over a handler and stay out of the tree layout, so adding a tree to
    ``OWNED_LOGGERS`` reaches every attach point at once."""
    for name in OWNED_LOGGERS:
        logger = logging.getLogger(name)
        for h in list(logger.handlers):
            if getattr(h, _HANDLER_TAG, False) and getattr(handler, _HANDLER_TAG, False):
                logger.removeHandler(h)
        logger.addHandler(handler)


def configure_logging(verbose: bool = False, *, stream: TextIO | None = None) -> None:
    """Install one console handler on every owned logger tree (stderr by default).

    Shows INFO and above, or DEBUG and above under ``verbose``. That threshold lives on
    the HANDLER: an owned tree may pin its own level (harbor holds its loggers at DEBUG
    so its standalone file handler sees everything), and a level set on an ancestor is
    not consulted once a descendant has one. Filtering where the record is emitted is
    the only point that governs every tree.

    Idempotent: replaces our own prior handler instead of stacking. ``propagate`` is
    disabled on each owned tree so records use only this handler (no double-emit via
    root); the root logger is left at WARNING so noisy third-party loggers (httpx,
    huggingface_hub, ...) stay quiet without per-library tuning."""
    stream = stream if stream is not None else sys.stderr
    handler = logging.StreamHandler(stream)
    setattr(handler, _HANDLER_TAG, True)
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    handler.setFormatter(_ConsoleFormatter(color=supports_color(stream)))
    attach_owned(handler)
    for name in OWNED_LOGGERS:
        logger = logging.getLogger(name)
        # Open the tree to DEBUG and let the handler decide. Raising it here would
        # drop records before any handler sees them, including the rollout's run.log.
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
    logging.getLogger().setLevel(logging.WARNING)
