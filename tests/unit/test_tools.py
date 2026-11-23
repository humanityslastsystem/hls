"""Unit tests for ToolSpec + LocalCallableToolRunner."""

from __future__ import annotations
import asyncio

import pytest

from hls.spec.tools import LocalCallableToolRunner, ToolSpec


def _spec(name: str = "search") -> ToolSpec:
    return ToolSpec(name=name, description="d",
                    json_schema={"type": "object",
                                 "properties": {"q": {"type": "string"}}})


def test_sync_callable_runs_in_thread():
    def search(q: str) -> dict:
        return {"q": q, "n": 3}

    runner = LocalCallableToolRunner(specs=[_spec()], callables={"search": search})
    result = asyncio.run(runner.run("search", {"q": "hi"}))
    assert result == {"q": "hi", "n": 3}


def test_async_callable_awaited_directly():
    async def search(q: str) -> dict:
        await asyncio.sleep(0)
        return {"q": q, "async": True}

    runner = LocalCallableToolRunner(specs=[_spec()], callables={"search": search})
    result = asyncio.run(runner.run("search", {"q": "hi"}))
    assert result == {"q": "hi", "async": True}


def test_describe_tools_returns_specs_in_order():
    s1, s2 = _spec("a"), _spec("b")
    runner = LocalCallableToolRunner(specs=[s1, s2],
                                     callables={"a": lambda: {}, "b": lambda: {}})
    assert runner.describe_tools() == [s1, s2]


def test_specs_and_callables_must_agree_on_names():
    with pytest.raises(ValueError, match="same names"):
        LocalCallableToolRunner(specs=[_spec("a")], callables={"b": lambda: {}})


def test_unknown_tool_raises_keyerror():
    runner = LocalCallableToolRunner(specs=[_spec()], callables={"search": lambda q: {}})
    with pytest.raises(KeyError, match="unknown tool"):
        asyncio.run(runner.run("unknown", {}))


def test_non_dict_return_value_is_rejected():
    runner = LocalCallableToolRunner(specs=[_spec()], callables={"search": lambda q: "not a dict"})
    with pytest.raises(TypeError, match="expected dict"):
        asyncio.run(runner.run("search", {"q": "x"}))
