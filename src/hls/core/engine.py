"""Engine: the runtime face of a stateful backend (the engine plane).

An engine gains capabilities by subclassing capability ABCs, and ``supports(cls)`` reads the
capability set off its MRO. The matching config (e.g. ``HFReference``) builds the engine with
``build_engine()``; only the config is serialized into a record. An engine names the GPUs it
asks for (``resources``); the role that runs it loads it through an ``EngineHost``
(``hls.verify.inprocess``), which the role creates and hands down."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from hls.core.capabilities import Capability, cap_method


@dataclass(frozen=True)
class ResourceRequest:
    """What one engine asks of the role that loads it."""

    #: GPUs the engine computes on; 0 for none.
    gpu_count: int
    #: The GPU kind it needs ("H200", ...), or None for any.
    gpu_kind: str | None = None
    #: Host memory it needs, in MiB, or None when it declares none.
    memory_mb: int | None = None


class Engine(ABC):
    """Base for a runtime engine. Subclass capability ABCs to gain them."""

    #: Whether the engine runs in processes its load starts beside the role's own, as a serving
    #: stack runs its engine core, scheduler or detokenizer. The engine host watches those while
    #: it holds the engine and ends the role when one of them ends (``hls.verify.inprocess``).
    runs_processes: ClassVar[bool] = False

    @property
    @abstractmethod
    def resources(self) -> ResourceRequest:
        """What this engine asks of the role that loads it."""

    def _unload(self) -> None:
        """Stop what ``_load`` started that dropping this engine's references does not
        stop, such as a serving stack's engine process. The engine host calls this before it
        drops what the load created. Blocking, so callers run it in a worker thread. This base
        has nothing to stop."""

    def _health(self) -> str | None:
        """Why the engine can serve no more, as its own API reports it, or None while it reports
        itself healthy or has no such report. The engine host asks an engine that runs processes
        of its own (``runs_processes``) once it has loaded, every few seconds, from a thread of
        its own, so an answer reads state and waits on nothing. This base has no report."""
        return None


def supports(engine_cls: type) -> frozenset[type[Capability]]:
    """The capabilities an engine class provides, derived from its MRO.

    A capability is a direct subclass of Capability; the engine class only subclasses
    Capability transitively (via its capability bases), so a direct-base check excludes it.
    Excludes a capability whose abstract method the class has not implemented (a bare check
    would over-report for an abstract class)."""
    abstract = getattr(engine_cls, "__abstractmethods__", frozenset())
    return frozenset(
        b for b in engine_cls.__mro__
        if isinstance(b, type) and Capability in getattr(b, "__bases__", ())
        and cap_method(b) not in abstract
    )
