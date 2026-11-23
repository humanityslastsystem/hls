"""Where an allocation's GPUs, CPUs and memory come from on the host.

An allocation holds its GPUs by UUID, whole cores, and an amount of memory. Its CPUs are the
first of those cores' CPUs, exactly as many as it asked for, so that the grant check counts what
was asked; the rest of a core it holds stays idle, so no two allocations ever share a core. The
cores lie on as few NUMA nodes as hold them: one node when one node has enough free cores, the
fullest such node first, so that larger holes stay whole; else the nodes with the most free CPUs.
GPUs of the asked kind are taken in the host's order, passing over any that hold a compute process
(``busy``): on a host shared with others, a card is free only when no job of the scheduler holds it
and no process hls did not start computes on it."""

from __future__ import annotations

from collections.abc import Collection, Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hls.spec.resources import ResourceShape
from hls.substrate.local.inventory import CoreFact, HostFacts


class Reservation(BaseModel):
    """What one allocation holds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Its GPUs' UUIDs, in the host's order.
    gpus: tuple[str, ...]
    #: The CPUs it may run on, ascending: its cpuset.
    cpus: tuple[int, ...] = Field(min_length=1)
    #: Every CPU of the cores it holds, ascending: its CPUs and any idle siblings.
    held_cpus: tuple[int, ...]
    #: Its memory in GiB.
    memory_gib: int = Field(ge=1)

    @model_validator(mode="after")
    def _cpus_held(self) -> Reservation:
        if not set(self.cpus) <= set(self.held_cpus):
            raise ValueError("a reservation's CPUs lie outside the cores it holds")
        return self


def never_fits(shape: ResourceShape, host: HostFacts) -> list[str]:
    """Every reason *shape* cannot run on *host* even when nothing else runs; empty when it can."""
    out = []
    kind = [g for g in host.gpus if g.kind == shape.gpu_kind] if shape.gpus else []
    if len(kind) < shape.gpus:
        out.append(f"it asks for {shape.gpus} {shape.gpu_kind} GPUs and the host has {len(kind)}")
    online = sum(len(c.cpus) for c in host.cores)
    if shape.cpus > online:
        out.append(f"it asks for {shape.cpus} CPUs and the host has {online}")
    if shape.memory_gib > host.allocatable_memory_gib:
        out.append(f"it asks for {shape.memory_gib} GiB and the host gives allocations "
                   f"{host.allocatable_memory_gib:.1f} GiB")
    return out


def free_gpus(shape: ResourceShape, host: HostFacts, held: Iterable[Reservation],
              busy: Collection[str] = ()) -> list[str]:
    """The UUIDs of *host*'s GPUs of *shape*'s kind, in the host's order, that no reservation of
    *held* holds and no process computes on (*busy*)."""
    taken = {u for r in held for u in r.gpus}
    return [g.uuid for g in host.gpus if g.kind == shape.gpu_kind and g.uuid not in taken and g.uuid not in busy]


def place(shape: ResourceShape, host: HostFacts, held: Iterable[Reservation],
          busy: Collection[str] = ()) -> Reservation | None:
    """Where *shape* runs on *host* beside the reservations *held*, on GPUs no process computes on
    (*busy*); None when it does not fit now."""
    held = list(held)
    if shape.memory_gib > host.allocatable_memory_gib - sum(r.memory_gib for r in held):
        return None
    gpus = free_gpus(shape, host, held, busy) if shape.gpus else []
    if len(gpus) < shape.gpus:
        return None
    taken_cpus = {c for r in held for c in r.held_cpus}
    cores = _cores(shape.cpus, [core for core in host.cores if not taken_cpus.intersection(core.cpus)])
    if cores is None:
        return None
    cpus = sorted(c for core in cores for c in core.cpus)
    given = [c for core in cores for c in core.cpus][:shape.cpus]
    return Reservation(gpus=tuple(gpus[:shape.gpus]), cpus=tuple(sorted(given)), held_cpus=tuple(cpus),
                       memory_gib=shape.memory_gib)


def _cores(need: int, free: list[CoreFact]) -> list[CoreFact] | None:
    """Free cores that hold *need* CPUs on as few NUMA nodes as hold them; None when the free
    cores hold fewer."""
    by_node: dict[int, list[CoreFact]] = {}
    for core in free:
        by_node.setdefault(core.numa, []).append(core)

    def size(node: int) -> int:
        return sum(len(core.cpus) for core in by_node[node])

    one = sorted((size(n), n) for n in by_node if size(n) >= need)
    nodes = [one[0][1]] if one else [n for _, n in sorted(((-size(n), n) for n in by_node))]
    out: list[CoreFact] = []
    have = 0
    for node in nodes:
        for core in sorted(by_node[node], key=lambda c: c.cpus[0]):
            if have >= need:
                return out
            out.append(core)
            have += len(core.cpus)
    return out if have >= need else None
