"""Where an allocation's GPUs, cores and memory come from."""

from __future__ import annotations

import pytest

from hls.substrate.local.inventory import GpuFact
from hls.substrate.local.placement import Reservation, never_fits, place
from tests.substrate.local.helpers import FakeInventory, shape


def _host(**kw):
    return FakeInventory(**kw).read(kept_memory_gib=4.0)


def test_cpus_come_from_one_numa_node_when_one_holds_them():
    host = _host(numa=2, cores_per_node=4)
    held = [Reservation(gpus=(), cpus=(0,), held_cpus=(0,), memory_gib=1)]
    # Node 0 has three cores left and node 1 four: three CPUs go to node 0, the fuller one.
    assert place(shape(cpus=3), host, held).cpus == (1, 2, 3)
    assert place(shape(cpus=4), host, held).cpus == (4, 5, 6, 7)


def test_cpus_spread_over_as_few_nodes_as_hold_them():
    host = _host(numa=3, cores_per_node=4)
    held = [Reservation(gpus=(), cpus=(0, 1), held_cpus=(0, 1), memory_gib=1)]
    got = place(shape(cpus=7), host, held)
    # Nodes 1 and 2 have four free cores each and node 0 two: two nodes hold seven.
    assert got.cpus == (4, 5, 6, 7, 8, 9, 10)
    assert place(shape(cpus=11), host, held) is None


def test_cpus_are_whole_cores_and_an_idle_sibling_is_held():
    host = _host(numa=1, cores_per_node=4, threads=2)
    first = place(shape(cpus=3), host, [])
    assert first.cpus == (0, 1, 2) and first.held_cpus == (0, 1, 2, 3)
    second = place(shape(cpus=4), host, [first])
    assert second.cpus == second.held_cpus == (4, 5, 6, 7)
    assert place(shape(cpus=1), host, [first, second]) is None


def test_a_core_with_any_cpu_held_is_not_free():
    host = _host(numa=1, cores_per_node=2, threads=2)
    held = [Reservation(gpus=(), cpus=(1,), held_cpus=(1,), memory_gib=1)]
    assert place(shape(cpus=2), host, held).cpus == (2, 3)
    assert place(shape(cpus=3), host, held) is None


def test_gpus_are_taken_by_uuid_of_the_asked_kind_and_never_twice():
    inventory = FakeInventory(gpus=3)
    inventory.facts["gpus"] = (*inventory.facts["gpus"], GpuFact(uuid="GPU-other", name="Some GPU", kind=None))
    host = inventory.read(kept_memory_gib=0)
    first = place(shape(gpus=2), host, [])
    assert first.gpus == ("GPU-00000000-fake", "GPU-00000001-fake")
    second = place(shape(gpus=1), host, [first])
    assert second.gpus == ("GPU-00000002-fake",)
    assert place(shape(gpus=1), host, [first, second]) is None
    assert place(shape(), host, [first, second]).gpus == ()


def test_memory_leaves_what_the_host_keeps():
    host = _host(memory_gib=64)
    first = place(shape(memory_gib=50), host, [])
    assert first is not None and first.memory_gib == 50
    assert place(shape(memory_gib=11), host, [first]) is None
    assert place(shape(memory_gib=10), host, [first]) is not None


def test_never_fits_names_every_shortfall_of_the_empty_host():
    host = _host(gpus=2, numa=1, cores_per_node=4, memory_gib=16)
    assert never_fits(shape(gpus=2, cpus=4, memory_gib=12), host) == []
    problems = never_fits(shape(gpus=3, cpus=5, memory_gib=13), host)
    assert len(problems) == 3 and "3 H200 GPUs" in problems[0] and "5 CPUs" in problems[1]


def test_a_reservation_holds_its_cpus():
    with pytest.raises(ValueError):
        Reservation(gpus=(), cpus=(0, 1), held_cpus=(0,), memory_gib=1)
