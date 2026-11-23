"""Tests of the engine host's concurrency rules: the GPUs are held
from before the load starts, an engine still loading cannot be acquired again, and a GPU is
free only once its engine's release has ended."""

from __future__ import annotations

import asyncio
import threading

import pytest

from hls.verify.inprocess import EngineHost, PlacementError
from tests.stages_lib.test_placement import _LOADS, _Placed, _run


@pytest.fixture(autouse=True)
def _clear():
    _LOADS.clear()


class _Gate:
    """Two events a worker thread and the test meet on."""

    def __init__(self) -> None:
        self.reached = threading.Event()
        self.go_on = threading.Event()

    def hold(self) -> None:
        self.reached.set()
        assert self.go_on.wait(10)

    async def reached_in_thread(self) -> None:
        assert await asyncio.to_thread(self.reached.wait, 10)


class _SlowLoad(_Placed):
    """A placed engine whose load waits at *gate* until the test lets it go on."""

    def __init__(self, tag: str, gate: _Gate) -> None:
        super().__init__(tag)
        self.gate = gate

    def _load(self) -> None:
        self.gate.hold()
        super()._load()


def test_an_acquire_while_another_engine_loads_takes_another_gpu():
    gate = _Gate()

    async def go():
        host = EngineHost(devices=2)
        loading = asyncio.ensure_future(host.acquire(_SlowLoad("slow", gate)))
        await gate.reached_in_thread()
        await host.acquire(_Placed("fast"))
        gate.go_on.set()
        await loading
        await host.teardown()

    _run(go())
    assert sorted(_LOADS) == [("fast", 1), ("slow", 0)]


def test_an_engine_still_loading_is_not_acquired_again():
    gate = _Gate()
    slow = _SlowLoad("slow", gate)

    async def go():
        host = EngineHost(devices=2)
        loading = asyncio.ensure_future(host.acquire(slow))
        await gate.reached_in_thread()
        try:
            with pytest.raises(PlacementError, match="already acquired"):
                await asyncio.wait_for(host.acquire(slow), 2)
        finally:
            gate.go_on.set()
            await loading
            await host.teardown()

    _run(go())


def test_a_gpu_is_free_only_once_its_engines_release_has_ended():
    gate = _Gate()

    class _SlowUnload(_Placed):
        def _unload(self) -> None:
            gate.hold()

    async def go():
        host = EngineHost(devices=1)
        _, release = await host.acquire(_SlowUnload("old"))
        releasing = asyncio.ensure_future(release())
        await gate.reached_in_thread()
        with pytest.raises(PlacementError, match="0 of the role's 1 are free"):
            await host.acquire(_Placed("early"))
        gate.go_on.set()
        await releasing
        await host.acquire(_Placed("new"))
        await host.teardown()

    _run(go())
    assert _LOADS == [("old", 0), ("new", 0)]


@pytest.mark.parametrize("where", ["release", "teardown", "load"])
def test_the_gpus_are_freed_even_when_the_undo_fails(where):
    class _BadUnload(_Placed):
        def _unload(self) -> None:
            raise RuntimeError("stop failed")

    async def go():
        host = EngineHost(devices=1)
        if where == "load":
            with pytest.raises(RuntimeError, match="cannot load"):
                await host.acquire(_BadUnload("bad", fail_load=True))
        else:
            _, release = await host.acquire(_BadUnload("bad"))
            with pytest.raises(RuntimeError, match="stop failed"):
                await (release() if where == "release" else host.teardown())
        await host.acquire(_Placed("next"))
        await host.teardown()

    _run(go())
    assert _LOADS == [("bad", 0), ("next", 0)]
