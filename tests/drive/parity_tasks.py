"""Two tasks and a recording artifact for the drive parity test (``test_drive_parity.py``).

Each task is driven the way the drive role drives one (``hls.drive.role.drive``), and every
request that reaches the artifact is recorded: its tenant, request id, session, turn, send time
and the sha256 of the body the wire protocol encoded. ``drive_parity.json`` holds what the drive
sent for both tasks, recorded by ``record_drive``.

A drive on the wall clock never repeats to the microsecond, so both tasks run on a virtual clock
(``tests.support.virtual_clock``): an event loop whose time moves only when every coroutine waits,
straight to the next timer, and whose thread pool runs each call inline. The driver's own clock
(``hls.spec.drivers.harness_now``) reads the same time. The artifact answers each request after a
service time drawn from its body,
so requests overlap, a follow-up turn waits for its answer and then its think time, and a
trajectory queues for its LLM slot. A send time is then exactly what the schedule made it.

The tasks are built so that every draw shapes what is sent: two tenants interleave on one clock,
with Poisson and Gamma arrivals seeded per tenant, Gamma and constant think times, a ramp-down that
trims late sessions' turns, a removal list applied before the schedule, a context budget that ends
a session mid-way, and trajectories that take as many turns as their item asks, up to ``max_turns``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Callable
from typing import Any
from unittest import mock

from hls.artifact.artifact_client import HTTPArtifactClient
from hls.core.dataitem import DatasetItem
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent
from hls.drive import role
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec import drivers
from hls.spec.agent import AgentFramework, AgentSpec
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import Dataset, Split
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.tools import LocalCallableToolRunner, ToolSpec
from hls.spec.workload import (ConstantInterTurn, Gamma, GammaInterTurn, NoSLO, Poisson, ShortOutputCap,
                               Uncapped, Workload)
from hls.verify import LetterMatchMetric, MetricVerifier
from tests.support.inline_dataset import InlineMultiTurnDataset
from tests.support.virtual_clock import virtual_clock

#: The URL the drive is pointed at. Nothing listens there: the recording client never dials it.
ARTIFACT_URL = "http://127.0.0.1:1"
#: The drive's per-request grace, as the stage passes ``--request-timeout-sec``.
REQUEST_TIMEOUT_SEC = 30.0

_CONTRACT = RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                    output_modalities=frozenset({"text"}), transport="http")
_MODEL = ModelSpec(base_checkpoint="fake/model", arch="TX", chat_format="llama3-instruct")
_HARDWARE = Hardware(accelerator=AcceleratorSpec(kind="H200", count=1, memory_gb_per_device=141))
_SCORER = GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))
_VERIFIERS = [MetricVerifier(metric=LetterMatchMetric(), min_value=0.5)]


def _sha256(body: Any) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def _service_sec(body_sha: str) -> float:
    """How long the artifact takes to answer a body: 0.25 s to 1.125 s in eighths, from its hash."""
    return 0.25 + (int(body_sha[:4], 16) % 8) * 0.125


class _RecordingClient(HTTPArtifactClient):
    """The harness's HTTP client with its POST replaced: it records what would have been posted
    and answers after a service time, with a reply that names the body, so each follow-up turn
    carries a history the earlier answers shaped.

    Built without ``__init__``, so it holds no connection pool and its warmup and close do
    nothing, as for any client double that replaces ``_post``."""

    #: The rows recorded, one per request in the order the artifact received them.
    sent: list[list]
    #: The loop clock the send times are read from.
    clock: Any
    #: The envelope of the request being posted, set by ``submit`` for ``_post`` to record.
    _env: Envelope | None

    async def submit(self, env: Envelope, *, attempt=None):
        self._env = env
        return await super().submit(env, attempt=attempt)

    async def _post(self, json_body: dict[str, Any]) -> dict[str, Any]:
        env, sha = self._env, _sha256(json_body)
        meta = env.metadata
        self.sent.append([meta["tenant_id"], env.request_id, meta["session_id"], meta["turn_index"],
                          round(self.clock(), 6), sha])
        await asyncio.sleep(_service_sec(sha))
        return {"id": "c", "choices": [{"message": {"role": "assistant", "content": f"reply {sha[:12]}"},
                                        "finish_reason": "stop", "token_ids": [33]}],
                "usage": {"completion_tokens": 1}}


def recording_client(sent: list[list], clock):
    """A stand-in for ``make_artifact_client`` that builds a ``_RecordingClient`` appending to
    *sent*, its send times read from *clock*."""

    def make(contract, base_url, request_timeout_sec=None, stall_timeout_sec=None):
        client = HTTPArtifactClient.__new__(_RecordingClient)
        client.contract, client.base_url = contract, base_url.rstrip("/")
        client.request_timeout_sec = request_timeout_sec
        client.sent, client.clock, client._env = sent, clock, None
        return client

    return make


def record_drive(task: TaskSpec, on_event: Callable[[TraceEvent], None] = lambda ev: None) -> list[list]:
    """Drive *task* as the drive role does, on a virtual clock, handing *on_event* each event as it
    is recorded, and return every request the artifact received: ``[tenant_id, request_id,
    session_id, turn_index, send time from the drive's start rounded to the microsecond, sha256 of
    the posted body]``.

    The drive anchors its clock at virtual time 0, since nothing before the anchor waits."""
    sent: list[list] = []
    with virtual_clock() as runner:
        clock = runner.get_loop().time
        with mock.patch("hls.artifact.artifact_client.make_artifact_client",
                        recording_client(sent, clock)):
            driver = role.build_driver(task, request_timeout_sec=REQUEST_TIMEOUT_SEC, drive_timeout_sec=None)
            runner.run(role.drive(task, driver, artifact_url=ARTIFACT_URL, split="test", on_event=on_event,
                                  gate=None))
    return sent


def _conversations(tenant: str, turns: list[int], *,
                   long: frozenset[int] = frozenset()) -> InlineMultiTurnDataset:
    """One conversation per entry of *turns*, with that many turns; a conversation whose index is
    in *long* has 300-character user turns, enough to run into a 700-character context budget."""
    def user_text(c: int, k: int) -> str:
        return f"{tenant} c{c} turn {k} " + ("x" * 280 if c in long else "")

    return InlineMultiTurnDataset(conversations=[
        (f"{tenant}-c{c:02d}", [(user_text(c, k), f"g{k}") for k in range(n)]) for c, n in enumerate(turns)])


def scripted_task() -> TaskSpec:
    """Two tenants of scripted multi-turn conversations, seed 7.

    ``chat`` arrives Poisson, thinks a Gamma time, trims its ramp-down and has one row removed;
    ``code`` arrives in Gamma bursts, thinks a constant time and has a context budget that ends
    its long conversation after two turns."""
    split = SplitSpec(dev=Count(value=1), test=Count(value=6))
    chat = Tenant(
        id="chat",
        data_source=DataSource(dataset=_conversations("chat", [3, 4, 2, 4, 3, 4]), split=split,
                               removed_rows=("chat-c02",)),
        workload=Workload(arrival=Poisson(rate_per_sec=0.5),
                          inter_turn_delay=GammaInterTurn(shape=2.0, scale=1.5, max_gap=6.0),
                          output_length_cap=ShortOutputCap(cap_tokens=64), slo=NoSLO(),
                          mean_turns_per_session=3.4, trim_rampdown=True),
        verifiers=_VERIFIERS)
    code = Tenant(
        id="code",
        data_source=DataSource(dataset=_conversations("code", [4, 2, 4, 3, 4], long=frozenset({2})),
                               split=split),
        workload=Workload(arrival=Gamma(shape=0.5, scale=2.0), inter_turn_delay=ConstantInterTurn(gap=1.25),
                          output_length_cap=Uncapped(), slo=NoSLO(), mean_turns_per_session=3.4,
                          history_char_budget=700),
        verifiers=_VERIFIERS)
    return TaskSpec(id="parity-scripted", model=_MODEL, hardware=_HARDWARE, contract=_CONTRACT,
                    scorer=_SCORER, tenants=[chat, code], seed=7)


class _SeedQuestions(Dataset):
    """Single-turn seed questions, each asking for a number of trajectory turns."""

    #: ``(row id, question, turns the trajectory takes)`` per question.
    questions: list[tuple[str, str, int]]

    def items_at(self, mount_dir, split: Split):
        for rid, question, turns in self.questions:
            yield DatasetItem(request_id=rid, turns=(Envelope(request_id=f"{rid}.0",
                                                              contents=[TextPayload(text=question)]),),
                              ground_truth_turns=("A",), metadata={"turns": turns})


#: The one tool the trajectory framework offers on every turn.
_LOOKUP_TOOL = {"type": "function", "function": {"name": "lookup"}}


class _LookupFramework(AgentFramework):
    """Asks the model, looks its reply up with the task's tool and asks again with what it found,
    for as many turns as the seed item asks or ``max_turns``, whichever is fewer. Like the
    deep-research loop it stamps its context's tenant, names its session and numbers its turns on
    every request, so the driver waits its think time before each follow-up turn."""

    async def run_trajectory(self, seed_item, llm_call, tool_runner, on_event, ctx, max_turns):
        sid = seed_item.request_id
        messages = [{"role": "user", "content": seed_item.turns[0].first_of("text").text}]
        for turn in range(min(seed_item.metadata["turns"], max_turns)):
            env = Envelope(request_id=f"{sid}.{turn}", contents=[TextPayload(text=messages[-1]["content"])],
                           metadata={"messages": list(messages), "tenant_id": ctx.tenant_id,
                                     "session_id": sid, "turn_index": turn, "tools": [_LOOKUP_TOOL]})
            t_send = drivers.harness_now()
            resp = await llm_call(env)
            reply = resp.output.first_of("text").text
            await on_event(TraceEvent(request_id=env.request_id, session_id=sid, turn_index=turn, input=env,
                                      response=resp, t_send=t_send, t_recv=drivers.harness_now()))
            found = await tool_runner.run("lookup", {"q": reply})
            messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": found["text"]}]


def _lookup(q: str) -> dict:
    return {"text": f"found {q}"}


def trajectory_task() -> TaskSpec:
    """Two tenants of agentic trajectories, seed 11, ``max_turns=3``: Poisson arrivals seeded per
    tenant, a Gamma think time before every follow-up turn, two LLM slots shared by both tenants,
    and a drive wall. The tool is a plain function, so it runs through the executor."""
    runner = LocalCallableToolRunner(specs=[ToolSpec(name="lookup", description="d", json_schema={})],
                                     callables={"lookup": _lookup})
    agent = AgentSpec(framework=_LookupFramework(), tool_runner=runner, max_turns=3, max_concurrent_llm=2,
                      drive_timeout_sec=3600.0)
    split = SplitSpec(dev=Count(value=1), test=Count(value=4))

    def tenant(tid: str, turns: list[int], rate: float) -> Tenant:
        questions = [(f"{tid}-q{i}", f"{tid} question {i}", n) for i, n in enumerate(turns)]
        return Tenant(id=tid,
                      data_source=DataSource(dataset=_SeedQuestions(questions=questions), split=split),
                      workload=Workload(arrival=Poisson(rate_per_sec=rate),
                                        inter_turn_delay=GammaInterTurn(shape=2.0, scale=0.75, max_gap=4.0),
                                        output_length_cap=ShortOutputCap(cap_tokens=32), slo=NoSLO()),
                      verifiers=_VERIFIERS)

    tenants = [tenant("web", [3, 1, 5, 2], 1.5), tenant("docs", [2, 3, 3, 1], 0.8)]
    return TaskSpec(id="parity-trajectory", model=_MODEL, hardware=_HARDWARE, contract=_CONTRACT,
                    scorer=_SCORER, tenants=tenants, seed=11, agent=agent)
