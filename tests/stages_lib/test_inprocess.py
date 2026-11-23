from __future__ import annotations

import asyncio
import gc
import sys
import threading
import types
import weakref
from collections.abc import Callable
from typing import ClassVar

import pytest

from hls.core.capabilities import StackReport
from hls.core.decode import Greedy
from hls.core.engine import Engine, ResourceRequest
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.core.verdict import Verdict
from hls.reference.references import Reference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.verify.base import ArtifactRollout, Verifier
from hls.verify.inprocess import EngineHost, PlacementError
from hls.verify.run import verify_trace

#: ("load" | "load-failed" | "report" | "unload", tag) in call order, kept outside the
#: engines because release puts an engine's own attributes back as they were.
_LOG: list[tuple[str, str]] = []
#: Whether each load ran on the main thread.
_ON_MAIN: list[bool] = []
#: What each ``_unload`` saw: the model the engine still held, and whether it ran on the
#: main thread.
_UNLOADS: list[tuple[object, bool]] = []


class _ToyEngine(Engine, StackReport):
    """An engine whose ``_load`` adds state the way a model load does, and whose
    ``_unload`` stops it the way a serving stack's shutdown does."""

    def __init__(self, tag: str = "toy", fail_load: bool = False, fail_unload: bool = False) -> None:
        self.tag = tag
        self.fail_load = fail_load
        self.fail_unload = fail_unload
        self._model = None

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1, gpu_kind="H200")

    def _load(self) -> None:
        if self._model is not None:
            return
        _ON_MAIN.append(threading.current_thread() is threading.main_thread())
        self._model = {"weights": self.tag}
        self._tokenizer = object()
        if self.fail_load:
            _LOG.append(("load-failed", self.tag))
            raise RuntimeError(f"{self.tag}: cannot load")
        _LOG.append(("load", self.tag))

    def _unload(self) -> None:
        _LOG.append(("unload", self.tag))
        _UNLOADS.append((self._model, threading.current_thread() is threading.main_thread()))
        if self.fail_unload:
            raise RuntimeError(f"{self.tag}: cannot stop")

    async def stack_report(self) -> dict:
        _LOG.append(("report", self.tag))
        return {"device_name": "fake", "loaded": str(self._model is not None)}


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _clear_log():
    _LOG.clear()
    _ON_MAIN.clear()
    _UNLOADS.clear()


def _fake_torch(on_empty: Callable[[], None]) -> types.SimpleNamespace:
    """A stand-in for a loaded torch with a GPU, calling *on_empty* on ``empty_cache``."""
    return types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=on_empty))


def test_acquire_loads_the_engine_in_process_and_it_is_its_own_proxy():
    engine = _ToyEngine()

    async def go():
        async with EngineHost() as host:
            proxy, release = await host.acquire(engine)
            report = await proxy.stack_report()
            await release()
            return proxy, report

    proxy, report = _run(go())
    assert proxy is engine
    assert report == {"device_name": "fake", "loaded": "True"}
    assert _LOG == [("load", "toy"), ("report", "toy"), ("unload", "toy")]
    # The load blocks, so it runs in a worker thread and leaves the event loop free.
    assert _ON_MAIN == [False]


def test_an_engine_cannot_be_acquired_twice_before_it_is_released():
    engine = _ToyEngine()

    async def go():
        host = EngineHost()
        _, release = await host.acquire(engine)
        with pytest.raises(PlacementError, match="already acquired"):
            await host.acquire(engine)
        await release()
        return engine._model

    assert _run(go()) is None


def test_release_drops_what_load_added_and_a_second_acquire_loads_again():
    engine = _ToyEngine()

    async def go():
        async with EngineHost() as host:
            _, release = await host.acquire(engine)
            await release()
            after_release = (engine._model, "_tokenizer" in vars(engine))
            _, release = await host.acquire(engine)
            report = await engine.stack_report()
            await release()
            return after_release, report

    after_release, report = _run(go())
    assert after_release == (None, False)
    assert report["loaded"] == "True"
    assert [k for k, _ in _LOG].count("load") == 2


def test_release_is_idempotent():
    engine = _ToyEngine()

    async def go():
        host = EngineHost()
        _, release = await host.acquire(engine)
        await release()
        await release()
        return host

    host = _run(go())
    assert engine._model is None and host.outstanding == 0


def test_a_load_that_raises_frees_what_it_set_and_the_engine_can_load_later():
    engine = _ToyEngine(fail_load=True)

    async def go():
        async with EngineHost() as host:
            with pytest.raises(RuntimeError, match="cannot load"):
                await host.acquire(engine)
            state = (engine._model, "_tokenizer" in vars(engine), host.outstanding)
            engine.fail_load = False
            _, release = await host.acquire(engine)
            report = await engine.stack_report()
            await release()
            return state, report

    state, report = _run(go())
    assert state == (None, False, 0)
    assert report["loaded"] == "True"


def test_releasing_empties_the_cuda_cache_only_where_torch_is_loaded(monkeypatch):
    # Each entry says whether the cache was emptied on the main thread.
    emptied: list[bool] = []

    async def go():
        host = EngineHost()
        _, release = await host.acquire(_ToyEngine())
        await release()

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    _run(go())
    assert emptied == [] and "torch" not in sys.modules
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(
        lambda: emptied.append(threading.current_thread() is threading.main_thread())))
    _run(go())
    # Emptied once, in a worker thread: freeing a model's memory can take long enough to
    # stall the event loop.
    assert emptied == [False]


class _Cycle:
    """Half of a reference cycle, as a model with parent pointers holds."""


#: A weak reference to the cycle each ``_CyclicEngine`` load made.
_CYCLES: list[weakref.ref] = []


class _CyclicEngine(Engine):
    """An engine whose load leaves a reference cycle, which only the cycle collector frees."""

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        a, b = _Cycle(), _Cycle()
        a.other, b.other = b, a
        self._model = a
        _CYCLES.append(weakref.ref(a))


def test_release_collects_cycles_before_it_empties_the_cache(monkeypatch):
    # Memory a cycle holds is not returned until the cycle is collected, so emptying the
    # CUDA cache before the collection would leave the old model's blocks allocated.
    _CYCLES.clear()
    dead_when_emptied: list[bool] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(lambda: dead_when_emptied.append(_CYCLES[0]() is None)))

    async def go():
        host = EngineHost()
        _, release = await host.acquire(_CyclicEngine())
        await release()

    gc.disable()
    try:
        _run(go())
    finally:
        gc.enable()
    assert dead_when_emptied == [True]


class _NoLoadEngine(Engine):
    """An engine with nothing to load."""

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)


def test_an_engine_with_no_load_is_acquired_as_it_is():
    engine = _NoLoadEngine()

    async def go():
        host = EngineHost()
        proxy, release = await host.acquire(engine)
        held = (proxy is engine, host.outstanding)
        await release()
        return held, host.outstanding

    assert _run(go()) == ((True, 1), 0)


@pytest.mark.parametrize("path", ["release", "teardown", "failed load"])
def test_an_engine_is_stopped_once_off_the_loop_before_its_attributes_go_back(path):
    # A serving stack's engine runs in a process of its own, which dropping the engine's
    # attributes does not end, so the host stops it first, while the engine still holds
    # what its load built.
    engine = _ToyEngine(fail_load=path == "failed load")

    async def go():
        host = EngineHost()
        if path == "failed load":
            with pytest.raises(RuntimeError, match="cannot load"):
                await host.acquire(engine)
            return host.outstanding
        _, release = await host.acquire(engine)
        await (release() if path == "release" else host.teardown())
        # Nothing is stopped twice.
        await release()
        await host.teardown()
        return host.outstanding

    assert _run(go()) == 0
    assert _UNLOADS == [({"weights": "toy"}, False)]
    assert (engine._model, "_tokenizer" in vars(engine)) == (None, False)


def test_a_stop_that_fails_reaches_the_caller_after_the_engine_is_put_back(monkeypatch):
    emptied: list[bool] = []
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(lambda: emptied.append(True)))
    engine = _ToyEngine(fail_unload=True)

    async def go():
        host = EngineHost()
        _, release = await host.acquire(engine)
        with pytest.raises(RuntimeError, match="toy: cannot stop"):
            await release()
        return host.outstanding

    assert _run(go()) == 0
    # Put back, freed and deregistered: the error is all that is left.
    assert (engine._model, "_tokenizer" in vars(engine), emptied) == (None, False, [True])


def test_a_failed_load_whose_stop_also_fails_reports_the_load():
    engine = _ToyEngine(fail_load=True, fail_unload=True)

    async def go():
        host = EngineHost()
        with pytest.raises(RuntimeError, match="toy: cannot load") as raised:
            await host.acquire(engine)
        return host.outstanding, raised.value

    outstanding, error = _run(go())
    assert outstanding == 0 and (engine._model, "_tokenizer" in vars(engine)) == (None, False)
    assert any("toy: cannot stop" in note for note in getattr(error, "__notes__", []))


def test_teardown_stops_every_engine_and_then_raises_the_first_failure():
    a, b, c = _ToyEngine("a", fail_unload=True), _ToyEngine("b", fail_unload=True), _ToyEngine("c")

    async def go():
        host = EngineHost()
        for engine in (a, b, c):
            await host.acquire(engine)
        with pytest.raises(RuntimeError, match="a: cannot stop") as raised:
            await host.teardown()
        return host.outstanding, raised.value

    outstanding, error = _run(go())
    assert [t for k, t in _LOG if k == "unload"] == ["a", "b", "c"]
    assert (a._model, b._model, c._model, outstanding) == (None, None, None, 0)
    assert any("b: cannot stop" in note for note in getattr(error, "__notes__", []))


class _GatedEngine(_ToyEngine):
    """An engine whose load and stop each wait at a gate, as a slow model load and a
    slow engine shutdown do, and record when they end."""

    def __init__(self, gate: str) -> None:
        super().__init__()
        #: Which of "load" and "unload" waits.
        self.gate = gate
        #: Set once the waiting call is running.
        self.started = threading.Event()
        #: Set by the test to let the waiting call finish.
        self.go_on = threading.Event()
        #: Set once the waiting call has returned.
        self.ended = threading.Event()

    def _wait(self, name: str) -> None:
        if self.gate == name:
            self.started.set()
            self.go_on.wait(5)

    def _load(self) -> None:
        self._wait("load")
        super()._load()
        if self.gate == "load":
            self.ended.set()

    def _unload(self) -> None:
        self._wait("unload")
        super()._unload()
        if self.gate == "unload":
            self.ended.set()


async def _cancel_while_waiting(engine: _GatedEngine, call: asyncio.Task) -> bool:
    """Cancel *call* once *engine*'s gated call runs, then let that call finish; return
    whether *call* was still waiting for it after the cancellation."""
    assert await asyncio.to_thread(engine.started.wait, 5), f"the engine's {engine.gate} never ran"
    call.cancel()
    await asyncio.sleep(0.05)
    waiting = not call.done()
    engine.go_on.set()
    with pytest.raises(asyncio.CancelledError):
        await call
    # The worker thread ends on its own whatever the host does; wait for it before reading
    # the engine.
    await asyncio.to_thread(engine.ended.wait, 5)
    return waiting


def test_a_cancelled_acquire_waits_for_its_load_then_stops_and_restores_the_engine():
    # A worker thread cannot be stopped, so an acquire that gave up at once would put the
    # engine back while the load still runs, and the load would then fill it again:
    # loaded, holding its GPU, and released by nobody.
    engine = _GatedEngine("load")

    async def go():
        host = EngineHost()
        waiting = await _cancel_while_waiting(engine, asyncio.ensure_future(host.acquire(engine)))
        return waiting, host.outstanding

    assert _run(go()) == (True, 0)
    assert (engine._model, "_tokenizer" in vars(engine)) == (None, False)
    assert _LOG == [("load", "toy"), ("unload", "toy")]
    assert _UNLOADS == [({"weights": "toy"}, False)]


def test_a_cancelled_release_waits_for_the_stop_then_restores_the_engine():
    engine = _GatedEngine("unload")

    async def go():
        host = EngineHost()
        _, release = await host.acquire(engine)
        waiting = await _cancel_while_waiting(engine, asyncio.ensure_future(release()))
        return waiting, host.outstanding

    assert _run(go()) == (True, 0)
    assert (engine._model, "_tokenizer" in vars(engine)) == (None, False)
    assert _UNLOADS == [({"weights": "toy"}, False)]


def test_teardown_releases_whatever_is_still_acquired():
    a, b = _ToyEngine("a"), _ToyEngine("b")

    async def go():
        host = EngineHost()
        await host.acquire(a)
        await host.acquire(b)
        await host.teardown()
        return host

    host = _run(go())
    assert (a._model, b._model, host.outstanding) == (None, None, 0)


def test_an_engine_released_by_teardown_can_be_acquired_again():
    engine = _ToyEngine()

    async def go():
        async with EngineHost() as host:
            await host.acquire(engine)
            await host.teardown()
            _, release = await host.acquire(engine)
            report = await engine.stack_report()
            await release()
            return report

    assert _run(go())["loaded"] == "True"
    assert [k for k, _ in _LOG].count("load") == 2


# --- the verification core on an engine host ---------------------------------------

_BUILT: list[_ToyEngine] = []


class _ToyRef(Reference):
    engine_cls: ClassVar[type] = _ToyEngine
    tag: str = "ora"

    def build_engine(self, task):
        engine = _ToyEngine(self.tag)
        _BUILT.append(engine)
        return engine


class _ReportVerifier(Verifier):
    """Asks its reference for a stack report and passes every event once it answers loaded."""

    reference: _ToyRef

    async def verify(self, engine) -> Verdict:
        report = await engine[StackReport]()
        events = engine[ArtifactRollout]
        ok = report["loaded"] == "True"
        return Verdict(passed=ok, per_event={ev.request_id: ok for ev in events})


def _event(req: str, tenant: str) -> TraceEvent:
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")], metadata={"tenant_id": tenant})
    out = Envelope(request_id=req, contents=[TextPayload(text="A")])
    return TraceEvent(request_id=req, input=inp, response=ArtifactResponse(output=out), t_send=0.0, t_recv=1.0)


def _task(tenants: list[Tenant]) -> TaskSpec:
    contract = RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                       output_modalities=frozenset({"text"}))
    return TaskSpec(id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
                    hardware=Hardware(accelerator=AcceleratorSpec(kind="H200", count=1)), contract=contract,
                    tenants=tenants, scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def _wl() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(), slo=NoSLO())


def test_the_verification_core_runs_on_an_engine_host():
    """Two tenants' references load on the engine host in turn, each stopped before the next
    loads, as the verify role runs them."""
    _BUILT.clear()
    ref_a, ref_b = _ToyRef(decode=Greedy(), tag="A"), _ToyRef(decode=Greedy(), tag="B")
    tenants = [Tenant(id="a", workload=_wl(), verifiers=[_ReportVerifier(reference=ref_a)]),
               Tenant(id="b", workload=_wl(), verifiers=[_ReportVerifier(reference=ref_b)])]
    host = EngineHost()

    async def go():
        return await verify_trace(_task(tenants), Trace(events=[_event("a0", "a"), _event("b0", "b")]),
                                  eval_dir=None, host=host, answers=None)

    verdict, results = _run(go())
    assert verdict.passed is True
    assert verdict.per_event == {"a:a0": True, "b:b0": True}
    # The engines loaded in turn, each answered while loaded, and each was stopped before
    # the next one loaded, so its GPU was free for it.
    assert _LOG == [("load", "A"), ("report", "A"), ("unload", "A"), ("load", "B"), ("report", "B"),
                    ("unload", "B")]
    assert [e._model for e in _BUILT] == [None, None] and host.outstanding == 0
    assert len(results) == 2
