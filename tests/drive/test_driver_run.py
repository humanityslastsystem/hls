"""``Driver.run``: the drive in the drive role's own process, with the drive's split, tenants,
contract and seed as arguments.

The artifact is an in-process ASGI app behind the harness's own HTTP client (no socket), so the
whole loop runs: the contract builds the client, every request goes through its wire protocol,
and the caller's sink gets one event per turn as it is recorded."""

from __future__ import annotations

import asyncio
import gc
import json
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

import hls.artifact.artifact_client as artifact_client_mod
import hls.spec.datasets as datasets_mod
from hls.artifact.artifact_client import HTTPArtifactClient
from hls.core.dataitem import DatasetItem
from hls.core.faults import DatasetError
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import Trace, TraceEvent
from hls.prompt import JinjaPromptTemplate
from hls.spec.agent import AgentFramework
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import Dataset, HFDataset
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import Driver, TrajectorySessionDriver
from hls.spec.tenant import Tenant
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, Uncapped, Workload
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC

CONTRACT = RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                   output_modalities=frozenset({"text"}))
URL = "http://test-artifact"


def _workload() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                    slo=NoSLO())


def _app(reply: str) -> FastAPI:
    """A minimal OpenAI-compatible server answering ``reply`` to every request."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        return {"id": "c", "choices": [{"message": {"role": "assistant", "content": reply},
                                        "finish_reason": "stop", "token_ids": [1]}],
                "usage": {"completion_tokens": 1}}

    return app


class _AsgiClient(HTTPArtifactClient):
    """The harness's HTTP client posting to an in-process app. Built without ``__init__``, so
    it holds no pool of its own; ``closed`` says whether the drive closed it."""

    #: The in-process server.
    app: FastAPI
    #: Whether ``aclose`` was called.
    closed: bool

    async def _post(self, json_body):
        transport = httpx.ASGITransport(app=self.app)
        async with httpx.AsyncClient(transport=transport, base_url=self.base_url) as cli:
            r = await cli.post(self.contract.path, json=json_body)
            r.raise_for_status()
            return r.json()

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def clients(monkeypatch):
    """Every client a drive builds, with the arguments it was built from, answering "B"."""
    built: list[tuple[_AsgiClient, object, str, float | None]] = []

    def make(contract, base_url, request_timeout_sec=None, stall_timeout_sec=None):
        client = HTTPArtifactClient.__new__(_AsgiClient)
        client.contract, client.base_url, client.request_timeout_sec = contract, base_url, request_timeout_sec
        client.app, client.closed = _app("B"), False
        built.append((client, contract, base_url, request_timeout_sec))
        return client

    monkeypatch.setattr(artifact_client_mod, "make_artifact_client", make)
    return built


@pytest.fixture
def tenants(tmp_path, monkeypatch):
    """Two tenants' rows where the driver reads them: one question for "a", two for "b"."""
    for tid, n in (("a", 1), ("b", 2)):
        d = tmp_path / "data" / "tenants" / tid / "test"
        d.mkdir(parents=True)
        (d / "rows.json").write_text(json.dumps(
            [{"question": f"{tid}{i}", "choices": ["a", "b", "c", "d"], "answer": 1, "source_index": i}
             for i in range(n)]))
    monkeypatch.setattr(datasets_mod, "MOUNT_ROOT", tmp_path / "data")
    dataset = HFDataset(path="cais/mmlu", subset="x", revision="main", source_split="test",
                        prompt=JinjaPromptTemplate(template="{{question}}"),
                        target=JinjaPromptTemplate(template="{{'ABCD'[answer]}}"))
    return [Tenant(id=tid, data_source=DataSource(dataset=dataset, split=SplitSpec(dev=Count(value=1),
                                                                                   test=Count(value=n))),
                   workload=_workload(), verifiers=[])
            for tid, n in (("a", 1), ("b", 2))]


def _run(driver: Driver, tenants, *, split: str = "test", seed: int = 0) -> Trace:
    """*driver*'s drive of *tenants*, its events collected as its sink got them."""
    events: list[TraceEvent] = []
    assert asyncio.run(driver.run(URL, split=split, tenants=tenants, contract=CONTRACT, seed=seed,
                                  on_event=events.append)) is None
    return Trace(events=events)


def test_run_drives_the_tenants_it_is_given_through_the_contracts_client(clients, tenants):
    driver = Driver(request_timeout_sec=12.0)
    trace = _run(driver, tenants, seed=3)

    assert len(trace.events) == 3
    assert all(ev.response.output.first_of("text").text == "B" for ev in trace.events)
    assert sorted(ev.input.metadata["tenant_id"] for ev in trace.events) == ["a", "b", "b"]
    [(client, contract, url, timeout)] = clients
    assert (contract, url, timeout) == (CONTRACT, URL, 12.0)
    assert client.closed, "the drive left its client's connection pool open"


def test_the_artifact_client_is_closed_after_a_drive_that_raises(clients, tenants, monkeypatch):
    """A drive that raises still releases its client: the close is owned by a ``finally``."""
    def broken(self, mount_dir, split):
        yield DatasetItem(request_id="bad", turns=(), ground_truth_turns=())

    monkeypatch.setattr(HFDataset, "items_at", broken)
    with pytest.raises(DatasetError):
        _run(Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC), tenants)
    assert [client.closed for client, *_ in clients] == [True]


def test_a_drive_mirrors_no_trace_from_the_environment(clients, tenants, monkeypatch, tmp_path, capsys):
    """A drive writes no partial trace and logs no event, whatever the environment names; it hands
    each event to its caller's sink, and the drive role writes the trace."""
    monkeypatch.setenv("HLS_PARTIAL_TRACE_PATH", str(tmp_path / "partial.jsonl"))
    monkeypatch.setenv("HLS_TRACE_EVENT_LOG", "true")
    trace = _run(Driver(request_timeout_sec=REQUEST_TIMEOUT_SEC), tenants)
    assert len(trace.events) == 3
    assert not (tmp_path / "partial.jsonl").exists()
    assert "HLS_TRACE_EVENT" not in capsys.readouterr().out


class _OneQuestion(Dataset):
    """One single-turn question, r0."""

    def items_at(self, mount_dir, split):
        env = Envelope(request_id="r0", contents=[TextPayload(text="Q?")])
        yield DatasetItem(request_id="r0", turns=(env,), ground_truth_turns=("A",))


class _Framework(AgentFramework):
    """One artifact call per turn, each emitting its own event."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        for k in range(max_turns):
            t_send = time.monotonic()
            env = Envelope(request_id=f"{seed_item.request_id}.{k}", contents=[TextPayload(text="Q?")],
                           metadata={"tenant_id": ctx.tenant_id, "turn_index": k,
                                     "session_id": seed_item.request_id})
            resp = await llm_call(env)
            await on_event(TraceEvent(request_id=env.request_id, session_id=seed_item.request_id,
                                      turn_index=k, input=env, response=resp, ground_truth=None,
                                      t_send=t_send, t_recv=time.monotonic()))


def test_the_drive_runs_with_the_cyclic_collector_paused(clients, monkeypatch):
    """A full collection walks every reply the drive holds and stalls the event loop in
    proportion, which drops connections and inflates measured latency on a long drive. The
    collector is off while requests are in flight and back on afterwards."""
    seen: list[bool] = []
    real_submit = _AsgiClient.submit

    async def watching(self, env, *, attempt=None):
        seen.append(gc.isenabled())
        return await real_submit(self, env, attempt=attempt)

    monkeypatch.setattr(_AsgiClient, "submit", watching)
    driver = TrajectorySessionDriver(framework=_Framework(), request_timeout_sec=REQUEST_TIMEOUT_SEC,
                                     max_turns=3, max_concurrent_llm=None,
                                     tool_runner=LocalCallableToolRunner(
                                         specs=[ToolSpec(name="search", description="d")],
                                         callables={"search": lambda: {}}))
    tenants = [SimpleNamespace(id="t0", data_source=SimpleNamespace(dataset=_OneQuestion(), removed_rows=()),
                               workload=_workload())]
    assert gc.isenabled()
    trace = _run(driver, tenants, split="dev", seed=42)
    assert len(trace.events) == 3
    assert seen == [False, False, False]
    assert gc.isenabled()
