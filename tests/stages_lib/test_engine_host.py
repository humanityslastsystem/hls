"""The engine host: a role loads its reference engines through the host it creates and hands
down, nothing binds a host to a context, and the host takes nothing from hls but the engine."""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from typing import ClassVar

import pytest

import hls.verify.inprocess as inprocess
from hls.core.engine import Engine, ResourceRequest
from hls.verify.inprocess import EngineHost, PlacementError


class _Toy(Engine):
    """An engine whose load and unload are recorded."""

    placeable: ClassVar[bool] = False

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        self.calls.append("load")
        self.model = object()

    def _unload(self) -> None:
        self.calls.append("unload")


def test_an_engine_has_no_lifecycle_of_its_own():
    assert not hasattr(Engine, "acquire") and not hasattr(Engine, "release")
    assert list(ResourceRequest.__dataclass_fields__) == ["gpu_count", "gpu_kind", "memory_mb"]


def test_an_engine_must_say_what_it_asks_for():
    class _Silent(Engine):
        pass

    with pytest.raises(TypeError, match="resources"):
        _Silent()


def test_the_host_loads_an_engine_and_hands_back_its_release():
    async def go():
        engine = _Toy()
        async with EngineHost(devices=0) as host:
            proxy, release = await host.acquire(engine)
            assert proxy is engine and engine.calls == ["load"] and host.outstanding == 1
            await release()
            assert engine.calls == ["load", "unload"] and host.outstanding == 0
            assert not hasattr(engine, "model")
    asyncio.run(go())


def test_leaving_the_host_releases_every_engine_even_after_an_error():
    engines = [_Toy(), _Toy()]

    async def go():
        async with EngineHost(devices=0) as host:
            for engine in engines:
                await host.acquire(engine)
            raise RuntimeError("the verifier failed")
    with pytest.raises(RuntimeError, match="the verifier failed"):
        asyncio.run(go())
    assert [e.calls for e in engines] == [["load", "unload"], ["load", "unload"]]


@pytest.mark.parametrize("item, message", [
    pytest.param(object(), "only reference engines run in process, not object", id="not-an-engine"),
])
def test_the_host_refuses_what_is_not_an_engine(item, message):
    async def go():
        async with EngineHost(devices=0) as host:
            await host.acquire(item)
    with pytest.raises(PlacementError, match=message):
        asyncio.run(go())


def test_one_engine_is_loaded_once():
    async def go():
        engine = _Toy()
        async with EngineHost(devices=0) as host:
            await host.acquire(engine)
            with pytest.raises(PlacementError, match="_Toy is already acquired"):
                await host.acquire(engine)
    asyncio.run(go())


def test_nothing_binds_a_host_to_a_context():
    assert [name for name in ("bind_substrate", "current_substrate", "NoSubstrateBoundError", "bound_in_process",
                              "InProcessSubstrate") if hasattr(inprocess, name)] == []


def test_the_host_takes_nothing_from_hls_but_the_engine():
    tree = ast.parse(Path(inprocess.__file__).read_text())
    named = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    named |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert sorted(m for m in named if m.split(".")[0] == "hls") == ["hls.core.engine"]
