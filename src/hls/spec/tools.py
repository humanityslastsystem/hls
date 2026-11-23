"""L1 -- ToolSpec + ToolRunner.

The harness side of the agentic trust boundary (design Sec.9). An agentic artifact
emits text containing tool calls; the harness-side ``AgentFramework`` parses
them and routes each call to a ``ToolRunner`` it owns. The artifact is NEVER
given direct access to the tool environment -- otherwise it could fake zero-
latency tool calls and escape the trajectory makespan budget. The wall time of
every ``ToolRunner.run`` counts toward the trajectory.

``LocalCallableToolRunner`` is the in-process runner: each tool is a Python
callable held in the ToolRunner. Used by tasks whose tools are cheap enough to
run in-band (e.g. an HTTP retriever query). ``hls.spec.sandbox.SandboxToolRunner``
gives each trajectory a container of its own, for a coding agent's shell.

The runner is built with the task module in the process that drives the
trajectories (the drive role, a calibration serving), and the framework calls
``.run(...)`` there.
"""

from __future__ import annotations
import asyncio
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal


@dataclass(frozen=True)
class ToolSpec:
    """Describes one tool the harness exposes to the agent (L1, Sec.9).

    ``json_schema`` is the canonical args schema; the agent prompt renders it as
    JSON Schema (OpenAI's ``function`` envelope and Anthropic's ``input_schema``
    envelope are surface-level rewraps handled by the prompt layer, not here).
    """

    name: str
    description: str
    json_schema: dict[str, Any] = field(default_factory=dict)


#: ``HostPath.store`` of a device node, bound from the same path on the node.
DEVICE = "device"


@dataclass(frozen=True)
class HostPath:
    """A path a tool runner needs inside the driver's container.

    A sandbox launched from inside the container sees the container's binds too, so a
    bind left writable is writable by whatever runs in the sandbox. A store the role
    only reads is therefore bound read-only: a graded model that can write the prefix
    store changes what every later drive of the task resumes from."""

    #: Absolute path inside the driver's container, where the runner opens it.
    path: str
    #: Whether the bind is read-only. False only for what the role writes, and for a
    #: device, which cannot be opened for a mount when bound read-only.
    read_only: bool
    #: Where the path comes from on the node: the name of one of the site's sandbox
    #: stores, or ``DEVICE`` for a device node bound from ``path`` itself.
    store: str

    def __post_init__(self) -> None:
        if not self.path.startswith("/"):
            raise ValueError(f"a host path is absolute inside the container, got {self.path!r}")
        if not self.store:
            raise ValueError(f"host path {self.path!r} names no store")
        if self.store == DEVICE and not self.path.startswith("/dev/"):
            raise ValueError(f"a device is a node under /dev, got {self.path!r}")


@dataclass(frozen=True)
class StoreEntry:
    """A file or directory a tool runner needs in one of the site's sandbox stores before a
    drive of its task starts (``ToolRunner.store_entries``)."""

    #: The store's name, as a ``HostPath`` names it.
    store: str
    #: Its path under the store's directory; empty for the directory itself.
    path: str
    kind: Literal["file", "dir"]
    #: What it is, as a refusal names it.
    what: str


class StoreRefused(ValueError):
    """What a task declares for its sandbox stores is not what it pins: provisioning it again
    changes nothing (``ToolRunner.provision_stores``)."""


class ToolRunner(ABC):
    """Executes tool calls on behalf of the harness (L1, Sec.9).

    The agent framework calls ``run(name, args)``; the runner returns a result
    dict. Wall time of ``run`` is real and counts toward the trajectory makespan
    SLO. ``describe_tools`` is read by the implementer prompt so the agent
    knows what calls are available.
    """

    @abstractmethod
    async def run(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def describe_tools(self) -> list[ToolSpec]: ...

    @property
    def host_paths(self) -> tuple[HostPath, ...]:
        """Paths this runner needs inside the driver's container.

        The driver runs in a container of its own, so a path that exists on the node is
        not a path the runner can open. Anything named here is bound into that container
        when the driver is deployed, from the site store it names, read-only where the
        entry says so. A runner whose tools are pure Python needs nothing.
        """
        return ()

    def store_entries(self, rows: Sequence[Mapping[str, Any]]) -> tuple[StoreEntry, ...]:
        """What this runner needs in the site's sandbox stores to drive the corpus *rows*, which
        a plan checks before anything runs. A runner that names no store needs nothing."""
        return ()

    def provision_stores(self, stores: Mapping[str, Path], rows: Sequence[Any] = ()) -> list[str]:
        """Make what this runner needs in the site's sandbox stores, *stores* by name to their
        directories, for a drive of the corpus *rows*, as the task's provisioning does; one line
        per step taken. Raise StoreRefused for something the task declares that cannot be
        placed as declared. A runner that names no store makes nothing."""
        return []

    async def warmup(self) -> None:
        """Load whatever the tools need before traffic starts. The trajectory driver
        awaits this before it anchors its clock, so a tool's load time is never a
        trajectory's latency. Default: nothing to load."""
        return None

    async def open_session(self, session_id: str, **context: Any) -> None:
        """Prepare whatever this trajectory's tools need of their own.

        A runner whose tools are stateless shares one loaded corpus across every
        trajectory and has nothing to do here. A runner that gives each trajectory a
        container does: two trajectories editing one checkout would write over each
        other, so the sandbox is per-session and is built here.

        The framework calls this once before a trajectory's first turn and pairs it
        with ``close_session`` in a ``finally``, because the trajectory is the only
        place that knows when the session began and ended. ``context`` carries what
        the runner needs to build the session -- for a coding task, which repository
        snapshot to check out.

        Its wall time is inside the trajectory's makespan, which is deliberate: a
        contestant does not pay for it, but neither is it free to the measurement, and
        billing it to the first turn's latency instead would misattribute it to the
        server. Default: nothing to prepare."""
        return None

    async def close_session(self, session_id: str) -> None:
        """Release what ``open_session`` built. Called in a ``finally``, so it runs
        even when the trajectory raised, and must tolerate a session that was never
        opened. Default: nothing to release."""
        return None


class LocalCallableToolRunner(ToolRunner):
    """In-process tool runner: each tool is a Python callable (sync or async).

    Used when the tool is cheap enough to run in-band. Sync callables are
    wrapped with ``asyncio.to_thread`` so they don't block the event loop.
    ``specs`` and ``callables`` must cover the same set of names; this is
    checked at construction.

    ``warmup`` is the optional load step the driver awaits before its clock, for
    tools that open a corpus or an index the first call would otherwise pay for.
    """

    def __init__(
        self,
        specs: list[ToolSpec],
        callables: dict[str, Callable[..., Any] | Callable[..., Awaitable[Any]]],
        warmup: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        spec_names = {s.name for s in specs}
        call_names = set(callables.keys())
        if spec_names != call_names:
            raise ValueError(
                f"specs and callables must cover the same names: "
                f"specs={sorted(spec_names)}, callables={sorted(call_names)}"
            )
        self._specs = list(specs)
        self._callables = dict(callables)
        self._warmup = warmup

    async def run(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        fn = self._callables.get(tool_name)
        if fn is None:
            raise KeyError(f"unknown tool {tool_name!r}; have {sorted(self._callables)}")
        if inspect.iscoroutinefunction(fn):
            result = await fn(**args)
        else:
            result = await asyncio.to_thread(fn, **args)
        if not isinstance(result, dict):
            raise TypeError(
                f"tool {tool_name!r} returned {type(result).__name__}; expected dict"
            )
        return result

    def describe_tools(self) -> list[ToolSpec]:
        return list(self._specs)

    async def warmup(self) -> None:
        if self._warmup is not None:
            await self._warmup()
