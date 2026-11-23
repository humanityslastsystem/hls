"""Per-engine device placement on the engine host (``hls.verify.inprocess``): an
engine that places itself gets the lowest GPUs of the role that no other placed engine holds,
before its load, and gives them back when it is released, its load fails or the host is
torn down; an engine that does not place itself is neither placed nor counted. The role's GPU
count is the one given, or torch's, read once and only for an engine that places itself."""

from __future__ import annotations

import asyncio
import sys
import types
from typing import ClassVar

import pytest

from hls.core.decode import Greedy
from hls.core.engine import Engine, ResourceRequest
from hls.reference.asr import WhisperASREngine
from hls.reference.engines import HFEngine, ReferenceEngine, SGLangEngine, VLLMEngine
from hls.reference.omni import Qwen3OmniEngine
from hls.reference.tts import Qwen3TTSEngine
from hls.verify.calibration import workload_fingerprint
from hls.verify.inprocess import EngineHost, PlacementError
from tests.verify_stage.fixture import FakeReference, gate_task

#: (tag, device_index) of each load, in order.
_LOADS: list[tuple[str, int | None]] = []


class _Placed(Engine):
    """An engine that places itself, as the HF engine does, recording where it loaded."""

    placeable: ClassVar[bool] = True

    def __init__(self, tag: str, gpus: int = 1, fail_load: bool = False) -> None:
        self.tag = tag
        self.gpus = gpus
        self.fail_load = fail_load
        self.device_index: int | None = None
        self._model = None

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=self.gpus, gpu_kind="H200")

    def _load(self) -> None:
        _LOADS.append((self.tag, self.device_index))
        if self.fail_load:
            raise RuntimeError(f"{self.tag}: cannot load")
        self._model = {"weights": self.tag}


class _Unplaced(_Placed):
    """An engine that does not place itself, as a serving stack, which takes every GPU its
    process sees."""

    placeable: ClassVar[bool] = False


@pytest.fixture(autouse=True)
def _clear():
    _LOADS.clear()


def _run(coro):
    return asyncio.run(coro)


def _torch(device_count) -> types.SimpleNamespace:
    """A stand-in for torch whose ``cuda.device_count`` is *device_count* and which has no GPU
    to empty a cache on."""
    cuda = types.SimpleNamespace(device_count=device_count, is_available=lambda: False)
    return types.SimpleNamespace(cuda=cuda)


@pytest.mark.parametrize("first_released", ["pinned", "anchor"])
def test_two_hf_engines_load_on_two_gpus(first_released):
    # A calibration's instrument check holds the pinned reference and its fp32 anchor at once
    # on a two-GPU role (``hls.calibrate._run_preflight``), and lets them go in either order.
    pinned, anchor = _Placed("pinned"), _Placed("anchor")

    async def go():
        host = EngineHost(devices=2)
        releases = {"pinned": (await host.acquire(pinned))[1], "anchor": (await host.acquire(anchor))[1]}
        held = host.outstanding
        await releases[first_released]()
        await host.acquire(_Placed("third"))
        await host.teardown()
        return held, host.outstanding

    assert _run(go()) == (2, 0)
    freed = 0 if first_released == "pinned" else 1
    assert _LOADS == [("pinned", 0), ("anchor", 1), ("third", freed)]
    # Released, each engine is put back as it was before its load, its device included.
    assert (pinned.device_index, anchor.device_index) == (None, None)
    assert (pinned._model, anchor._model) == (None, None)


def test_a_released_engines_gpu_is_given_to_the_next():
    a, b, c = _Placed("a"), _Placed("b"), _Placed("c")

    async def go():
        host = EngineHost(devices=3)
        await host.acquire(a)
        _, release_b = await host.acquire(b)
        await host.acquire(c)
        await release_b()
        await host.acquire(_Placed("d"))
        await host.teardown()

    _run(go())
    # The lowest free GPU, the one b gave back, not the next unused one.
    assert _LOADS == [("a", 0), ("b", 1), ("c", 2), ("d", 1)]


def test_an_engine_asking_for_more_gpus_than_free_is_refused():
    one, wide = _Placed("one"), _Placed("wide", gpus=2)

    async def go():
        host = EngineHost(devices=2)
        _, release_one = await host.acquire(one)
        with pytest.raises(PlacementError, match=r"^_Placed asks for 2 GPUs; 1 of the role's 2 are free, "
                                                 r"the others held by _Placed \(GPU 0\)$"):
            await host.acquire(wide)
        refused = (host.outstanding, wide._model, wide.device_index)
        await release_one()
        await host.acquire(wide)
        placed = wide.device_index
        await host.teardown()
        return refused, placed

    # Refused before anything of it loaded; once both GPUs are free it gets them, named by the first.
    assert _run(go()) == ((1, None, None), 0)
    assert _LOADS == [("one", 0), ("wide", 0)]


def test_a_role_with_fewer_gpus_than_an_engine_asks_for_refuses_it():
    with pytest.raises(PlacementError, match=r"^_Placed asks for 2 GPUs; 1 of the role's 1 are free$"):
        _run(EngineHost(devices=1).acquire(_Placed("wide", gpus=2)))
    assert _LOADS == []


def test_a_load_that_fails_gives_its_gpus_back():
    async def go():
        host = EngineHost(devices=1)
        with pytest.raises(RuntimeError, match="bad: cannot load"):
            await host.acquire(_Placed("bad", fail_load=True))
        await host.acquire(_Placed("good"))
        await host.teardown()

    _run(go())
    assert _LOADS == [("bad", 0), ("good", 0)]


def test_teardown_gives_every_gpu_back():
    async def go():
        host = EngineHost(devices=2)
        for tag in ("a", "b"):
            await host.acquire(_Placed(tag))
        await host.teardown()
        for tag in ("c", "d"):
            await host.acquire(_Placed(tag))
        await host.teardown()

    _run(go())
    assert _LOADS == [("a", 0), ("b", 1), ("c", 0), ("d", 1)]


def test_an_engine_that_does_not_place_itself_is_neither_placed_nor_counted():
    # A serving stack spreads over every GPU its process sees. That it shares its role with no
    # other engine is the planner's check (``hls.stages.verify.multi_engine_problem``).
    stack, hf = _Unplaced("stack"), _Placed("hf")

    async def go():
        host = EngineHost(devices=1)
        await host.acquire(stack)
        await host.acquire(hf)
        await host.teardown()

    _run(go())
    assert _LOADS == [("stack", None), ("hf", 0)]


def test_the_gpu_count_is_torchs_read_once_and_only_for_an_engine_that_places_itself(monkeypatch):
    def never() -> int:
        raise AssertionError("the GPU count was read for an engine that does not place itself")

    # Reading the count could start CUDA before a serving stack starts its workers.
    monkeypatch.setitem(sys.modules, "torch", _torch(never))

    async def serving_stack():
        host = EngineHost()
        await host.acquire(_Unplaced("stack"))
        await host.teardown()

    _run(serving_stack())
    reads: list[int] = []

    def two() -> int:
        reads.append(2)
        return 2

    monkeypatch.setitem(sys.modules, "torch", _torch(two))

    async def instrument():
        host = EngineHost()
        for tag in ("pinned", "anchor"):
            await host.acquire(_Placed(tag))
        await host.teardown()

    _run(instrument())
    assert _LOADS == [("stack", None), ("pinned", 0), ("anchor", 1)]
    assert reads == [2]


@pytest.mark.parametrize("torch", [None, _torch(lambda: 0)], ids=["no-torch", "no-gpu"])
def test_a_role_that_sees_no_gpu_places_nothing(monkeypatch, torch):
    # Every engine loads as it did before placement, on the process's current device. The HF
    # engine refuses to load without CUDA, so only tests and a misplanned role get here.
    monkeypatch.setitem(sys.modules, "torch", torch)

    async def go():
        host = EngineHost()
        for tag in ("a", "b"):
            await host.acquire(_Placed(tag))
        await host.teardown()
        return host.devices

    assert _run(go()) == 0
    assert _LOADS == [("a", None), ("b", None)]


def test_a_negative_gpu_count_is_refused():
    with pytest.raises(ValueError, match="a role sees 0 GPUs or more, not -1"):
        EngineHost(devices=-1)


def test_only_the_hf_engine_places_itself():
    assert HFEngine.placeable is True
    unplaced = (ReferenceEngine, VLLMEngine, SGLangEngine, Qwen3TTSEngine, Qwen3OmniEngine, WhisperASREngine)
    assert [cls.__name__ for cls in unplaced if cls.placeable] == []


def test_device_index_is_not_part_of_an_engines_identity():
    task = gate_task()
    reference = FakeReference(decode=Greedy())
    identity = (reference.fingerprint(task), workload_fingerprint(task))
    # The fake teacher is an HF engine, so it places itself.
    engine = reference.build_engine(task)
    before = dict(vars(engine))

    async def go():
        host = EngineHost(devices=2)
        await host.acquire(_Placed("first"))
        _, release = await host.acquire(engine)
        placed = engine.device_index
        await release()
        await host.teardown()
        return placed

    assert _run(go()) == 1
    assert vars(engine) == before and engine.device_index is None
    assert (reference.fingerprint(task), workload_fingerprint(task)) == identity
