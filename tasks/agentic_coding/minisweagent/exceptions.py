"""Vendored verbatim from mini-SWE-agent ``src/minisweagent/exceptions.py``.

Upstream: https://github.com/SWE-agent/mini-swe-agent at
``04d809ceab9df28f9adaed044884180159172930``. MIT licensed, Copyright (c) 2025
Kilian A. Lieret and Carlos E. Jimenez; see ``LICENSE`` beside this file.

No divergences. ``Submitted`` and ``UserInterruption`` are unused by this fork's
agent -- ``Submitted`` is raised by the environment adapter, ``UserInterruption`` by
upstream's interactive agent, which we do not vendor -- but both are kept so the
hierarchy a reader diffs against is upstream's whole one.
"""


class InterruptAgentFlow(Exception):
    """Raised to interrupt the agent flow and add messages."""

    def __init__(self, *messages: dict):
        self.messages = messages
        super().__init__()


class Submitted(InterruptAgentFlow):
    """Raised when the agent has completed its task."""


class LimitsExceeded(InterruptAgentFlow):
    """Raised when the agent has exceeded its cost or step limit."""


class TimeExceeded(LimitsExceeded):
    """Raised when the agent has exceeded its wall-clock time limit."""


class UserInterruption(InterruptAgentFlow):
    """Raised when the user interrupts the agent."""


class FormatError(InterruptAgentFlow):
    """Raised when the LM's output is not in the expected format."""
