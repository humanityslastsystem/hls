"""Tests for hls.calibrate -- the offline calibration library.

Everything here runs on the torch-free config host: no GPU, no docker. Deployment is
injected through the calibration roles' ``deploy`` seam -- the pinned reference, the fp32
preflight anchor and every pool member come from a fake -- and the roles are composed as
the pipeline composes them (``tests.unit.calibration_campaign``), so the whole procedure
(the exposure-counter stop, the single (P) look, the recurrence census, the refusal paths)
is exercised without touching a substrate. The engine side of a serving (the clocked drive
against an async API) is covered in test_clocked_serving.py.

The exposure target is a constant of ~12.3M tokens at the default tolerance, far more
than a test wants to synthesize, so the tests set ``mu_per_mtok`` to move it (see
``_mu_for_target``). That is the same knob a real campaign turns, not a test-only
escape hatch.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
from pathlib import Path
from typing import ClassVar

import pytest

from hls.core.dataitem import DatasetItem
from hls.core.decode import Greedy
from hls.core.model import AcceleratorSpec, Hardware, ModelSpec
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.pipeline.pipelines import REQUEST_TIMEOUT_SEC
from hls.prompt import JinjaPromptTemplate
from hls.reference.references import HFReference
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit
from hls.spec.contracts import RequestResponseContract
from hls.spec.datasets import Dataset, HFDataset, Split
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import (AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Uncapped,
                               Workload)
from hls.verify import LetterMatchMetric, MetricVerifier, TokenFaithfulnessVerifier
from hls.verify.calibration import CalibrationError, exposure_target, read_calibration, workload_fingerprint

from hls.verify.calibration_roles import ServingFailed
from tests.unit.calibration_campaign import in_process_campaign
from tests.unit.calibration_fixtures import LEVELS_NATS
from hls.calibrate import (
    DEFAULT_EPS,
    DEFAULT_MU_PER_MTOK,
    MEMBER_MAX_TRUSTED_FLIP_RATE,
    MIN_SERVINGS,
    POOL_ENGINE_NAMES,
    PREFLIGHT_MAX_FLIP_RATE,
    PREFLIGHT_MAX_GAP,
    PREFLIGHT_PROBES,
    CappedRows,
    CaptureStore,
    RowIndex,
    capped_rows,
    capture_items,
    item_event_key,
    load_sessions,
    refuse_past_cap,
    score_draw,
    serving_seed_of,
    validate_pool,
    write_calibration,
)


# ---- TaskSpec construction helpers ------------------------------------------

def _model(checkpoint: str = "m") -> ModelSpec:
    return ModelSpec(base_checkpoint=checkpoint, arch="TX")


def _hardware(kind: str = "B200") -> Hardware:
    return Hardware(accelerator=AcceleratorSpec(kind=kind, count=1))


def _contract(path: str = "/v1/chat/completions") -> RequestResponseContract:
    return RequestResponseContract(
        path=path,
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        protocol=OpenAIChatProtocol())


def _mmlu_dataset(subset: str = "abstract_algebra") -> HFDataset:
    return HFDataset(path="cais/mmlu", subset=subset, revision="main", source_split="test",
                     prompt=JinjaPromptTemplate(template="{{ question }}"),
                     target=JinjaPromptTemplate(template="{{ 'ABCD'[answer] }}"))


def _data_source(subset: str = "abstract_algebra") -> DataSource:
    return DataSource(dataset=_mmlu_dataset(subset),
                      split=SplitSpec(dev=Count(value=5), test=Count(value=10)))


def _workload(cap=None) -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=cap or Uncapped(), slo=NoSLO())


def _scorer() -> GoodputScorer:
    return GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0))


def _token_task(**overrides) -> TaskSpec:
    """A needs_calibration single-tenant TaskSpec (TokenFaithfulnessVerifier over MMLU).

    It declares no baseline, deliberately: a pool member is a reference engine on its
    own serving-stack image, so nothing about calibrating a task depends on that task
    also declaring a performance baseline."""
    kwargs = dict(
        id="T", model=_model(), hardware=_hardware(), contract=_contract(),
        workload=_workload(), data_source=_data_source(),
        verifiers=[TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))],
        scorer=_scorer())
    kwargs.update(overrides)
    return TaskSpec(**kwargs)


def _agentic_task(**overrides) -> TaskSpec:
    """The same task with an agent seam declared: a framework that takes two LLM calls
    per item, and one tool."""
    from hls.spec.agent import AgentSpec
    from hls.spec.tools import LocalCallableToolRunner, ToolSpec
    from tests.unit.test_trajectory_driver import _TwoCallFramework

    runner = LocalCallableToolRunner(
        specs=[ToolSpec(name="lookup", description="d", json_schema={})],
        callables={"lookup": lambda q: {"text": q}})
    agent = AgentSpec(framework=_TwoCallFramework(), tool_runner=runner, max_turns=3)
    return _token_task(agent=agent, **overrides)


_MMLU_ROWS = [
    {"source_index": 0, "question": "What is 2+2?", "answer": 0},
    {"source_index": 1, "question": "Capital of France?", "answer": 1},
    {"source_index": 2, "question": "Color of the sky?", "answer": 2},
    {"source_index": 3, "question": "Largest planet?", "answer": 3},
]

#: The request ids the provisioned rows serve under (each row's source position, one
#: turn each), so a fake member's trace names real rows and the capture store can key them.
_ROW_IDS = [f"{i}.0" for i in range(len(_MMLU_ROWS))]


def _provisioned(tmp_path, rows: list[dict] | None = None) -> Path:
    """The rows of the default tenant ``t0`` in a real tmp directory, laid out as the provision
    stage writes them and a calibration role mounts them under ``/data``
    (``tenants/<id>/<split>/rows.json``); the directory is returned."""
    rows = _MMLU_ROWS if rows is None else rows
    base = tmp_path / "prov"
    sub = base / "tenants" / "t0"
    for split, split_rows in (("dev", []), ("test", rows)):
        (sub / split).mkdir(parents=True, exist_ok=True)
        (sub / split / "rows.json").write_text(json.dumps(split_rows))
    return base


@pytest.fixture
def prov(tmp_path) -> Path:
    """The provisioned rows a campaign's sessions are materialized from."""
    return _provisioned(tmp_path)


# ---- doubles ----------------------------------------------------------------

_STACK = {"torch": "2.4.0", "runtime": {"cuda": "12.4", "cudnn": "90100"},
          "device_name": "B200"}


class _FakeAnchor:
    """Fake fp32 preflight anchor: echoes one reply-with-tokens per probe."""

    def __init__(self, tokens=(11, 22, 33)):
        self._tokens = list(tokens)

    async def generate(self, prompts):
        return [Envelope(request_id=p.request_id,
                         contents=[TextPayload(text="x", tokens=list(self._tokens))],
                         metadata={"source": "anchor"})
                for p in prompts]


class _FakeAnchorNoTokens(_FakeAnchor):
    async def generate(self, prompts):
        return [Envelope(request_id=p.request_id,
                         contents=[TextPayload(text="x", tokens=None)],
                         metadata={"source": "anchor"})
                for p in prompts]


class _FakePinned:
    """Fake pinned reference: scores the preflight once, then returns a per-engine gap
    pattern for every serving, however many the exposure counter decides to take.

    The serving count is an OUTPUT of the stop rule, so a fixture scripting a fixed
    number of ``teacher_force`` calls would be asserting the answer it is supposed to
    measure. ``flips_for`` decides one serving's flip count per reply, so a test can
    make an engine as reproducible (or as variable) as the (P) case it is exercising
    needs."""

    def __init__(self, gap: float, *, flips_for=None, stack=None,
                 site_request_id: str | None = None):
        #: The log-probability gap every flip carries, in nats.
        self.gap = gap
        #: serving index -> flips per reply (default: a fixed 20).
        self.flips_for = flips_for or (lambda draw: 20)
        #: When set, ONLY this request id flips -- the fixture for a recurring
        #: severe-event site.
        self.site_request_id = site_request_id
        self._stack = stack or dict(_STACK)
        self._preflight_done = False
        self._draw = 0
        self.tf_calls = 0

    async def teacher_force(self, items):
        self.tf_calls += 1
        if not self._preflight_done:
            self._preflight_done = True
            return [[0.0] * 3 for _ in items]
        flips = self.flips_for(self._draw) if self.gap else 0
        self._draw += 1
        out = []
        for item in items:
            n = len(item.first_of("text").tokens)
            k = flips
            if self.site_request_id is not None and item.request_id != self.site_request_id:
                k = 0
            # Clamped: a real reference returns exactly one gap per served token, so a
            # reply shorter than the scripted flip count must not grow gaps.
            k = min(k, n)
            out.append([self.gap] * k + [0.0] * (n - k))
        return out

    async def stack_report(self):
        return dict(self._stack)


async def _noop_release() -> None:
    return None


#: What a pool member reports as its serving package version in these tests. The
#: campaign records it verbatim; nothing computes on it.
_MEMBER_VERSION = "0.0.0"


class _FakeMember:
    """Pool-member proxy double: records every ``serve_traffic`` call and synthesizes
    that serving's Trace.

    Every serving answers the first ``replies`` sessions it is handed, one event of
    ``tokens`` ids each (never more than the tenant's cap, as a real engine cannot),
    so an engine's exposure per serving is a known constant and a test can reason
    about exactly when the counter stop fires. The events are stamped the way the
    drive loop stamps them -- tenant and cap on the request -- and name real rows, so
    the capture store can key them against the task. ``cap_replies`` names the
    replies that run to the cap instead: their length is the cap itself. The real
    member runs the harness's own Driver against its async API to produce this trace;
    that half is covered in test_clocked_serving.py."""

    def __init__(self, *, replies: int = 4, tokens: int = 2500,
                 cap_replies: frozenset[int] = frozenset(),
                 fault_on: "int | str | None" = None):
        self.replies = replies
        self.tokens = tokens
        self.cap_replies = cap_replies
        #: serving index whose first event comes back faulted; "all" faults every
        #: serving (a member that persistently cannot serve its load); None = never.
        self.fault_on = fault_on
        #: one (sessions, seed, max_sessions, agent) per serving driven.
        self.drives: list[tuple] = []

    async def serving_version(self) -> str:
        return _MEMBER_VERSION

    async def serve_traffic(self, sessions, *, seed, request_timeout_sec, max_sessions=None,
                            drive_timeout_sec=None,
                            agent=None, max_context_len=None):
        index = len(self.drives)
        self.drives.append((sessions, seed, max_sessions, agent))
        tenant, items = sessions[0]
        cap = tenant.workload.output_length_cap.max_tokens()
        events = []
        for i in range(self.replies):
            faulted = i == 0 and (self.fault_on == "all" or self.fault_on == index)
            rid = f"{items[i].request_id}.0"
            n = cap if i in self.cap_replies else min(self.tokens, cap)
            env = Envelope(request_id=rid,
                           metadata={"messages": [{"role": "user", "content": f"q{i}"}],
                                     "tenant_id": tenant.id, "max_tokens": cap})
            events.append(TraceEvent(
                request_id=rid, session_id=items[i].request_id, turn_index=0, input=env,
                response=None if faulted else ArtifactResponse(output=Envelope(
                    request_id=rid,
                    contents=[TextPayload(text="x", tokens=list(range(n)),
                                          finish_reason="length" if n >= cap else "stop")])),
                error="transport: boom" if faulted else None,
                t_send=0.0, t_recv=None if faulted else 1.0))
        return Trace(events=events)


class _NoTokenMember(_FakeMember):
    """A member whose replies carry no token ids -- unscoreable by construction."""

    async def serve_traffic(self, sessions, **kw):
        trace = await super().serve_traffic(sessions, **kw)
        for ev in trace.events:
            ev.response = ArtifactResponse(output=Envelope(
                request_id=ev.request_id,
                contents=[TextPayload(text="x", tokens=None, finish_reason="stop")]))
        return trace


def _make_deploy(pinned, member=None, anchor=None, members: dict | None = None):
    """The ONE deploy seam: the pinned reference, the fp32 anchor, and every pool
    member come through it, exactly as they do in a role. ``members`` hands each
    engine its own double, keyed by engine name; ``member`` serves every engine.

    Deploy names carry their role, as the roles' do -- ``pinned`` and ``fp32_anchor``
    for the instrument check, ``score-<share>`` for a share's reference,
    ``<engine>-<serving>`` for a member -- so a test can read the order and what was held
    at once off ``deploy.deployed``, ``deploy.live`` and ``deploy.peak``."""
    anchor = anchor or _FakeAnchor()
    member = member if member is not None else _FakeMember()
    deployed: list[str] = []
    live: list[str] = []
    peak = [0]

    async def deploy(name, engine):
        deployed.append(name)
        live.append(name)
        peak[0] = max(peak[0], len(live))

        async def release():
            live.remove(name)
        if name == "pinned" or name.startswith("score-"):
            return pinned, release
        if name == "fp32_anchor":
            return anchor, release
        if members is not None:
            return members[name.split("-")[0]], release
        return member, release
    deploy.deployed = deployed          # every deploy, in the order it came up
    deploy.live = live                  # the ones still held, right now
    deploy.peak = peak                  # the most held at once, over the whole run
    deploy.member = member
    return deploy


def _mu_for_target(tokens: float) -> float:
    """The per-token tail tolerance whose exposure target is ``tokens``. Inverts
    ``exposure_target`` so a test can state the campaign size it wants to exercise."""
    return math.log(1 / 0.025) * 1e6 / tokens


def _run(task, prov, *, engines=("vllm",), deploy, captures,
         module="tasks.demo.task", **kw):
    kw.setdefault("mu_per_mtok", _mu_for_target(40_000))
    return asyncio.run(in_process_campaign(
        task, module, engines=list(engines), deploy=deploy, rows=prov,
        captures=captures, **kw))


# ---- the exposure target ----------------------------------------------------

def test_exposure_target_is_a_task_independent_constant():
    """T = ln(1/alpha)/mu with mu per TOKEN, so the target is one number for every task
    and every trace length -- about 12.3M tokens at the default tolerance."""
    assert exposure_target(DEFAULT_MU_PER_MTOK) == pytest.approx(12_296_265, rel=1e-6)
    # ln(1/0.025) is the Garwood 97.5% limit for zero observed events.
    assert exposure_target(1.0) == pytest.approx(math.log(40) * 1e6, rel=1e-12)
    # Halving the tolerance doubles the exposure; nothing about the task enters.
    assert exposure_target(0.15) == pytest.approx(2 * exposure_target(0.3), rel=1e-12)
    with pytest.raises(CalibrationError):
        exposure_target(0.0)

# ---- the pool ---------------------------------------------------------------

def test_pool_is_the_serving_stacks():
    assert POOL_ENGINE_NAMES == ("vllm", "sglang")


def test_a_serving_of_an_agentic_task_hands_the_member_the_agent(prov, tmp_path):
    """The agent seam has to reach the member: it is what makes the member drive the
    task's own trajectories instead of one seed request per session, so the prompts
    measured are the prompts a graded run serves."""
    task = _agentic_task()
    member = _FakeMember()
    _run(task, prov, engines=["vllm"],
         deploy=_make_deploy(_FakePinned(0.10), member=member),
         captures=tmp_path / "caps")
    assert member.drives
    assert all(drive[3] is task.agent for drive in member.drives)


def test_a_pool_member_is_unsealed_only_for_an_agentic_task_without_provisioned_assets():
    """A member runs the task's tools itself. Tools that call the network need the
    member unsealed; tools that load a provisioned bundle do not, and the bundle's
    memory is requested for the member. Every other member is sealed, as a graded artifact
    is served."""
    from dataclasses import replace

    from hls.calibrate import member_engines, member_internet
    from tests.unit.test_tool_assets import FakeAssets

    provisioned = _token_task(agent=replace(_agentic_task().agent, assets=FakeAssets()))
    assert member_internet(_token_task()) is False
    assert member_internet(_agentic_task()) is True
    assert member_internet(provisioned) is False
    for name in POOL_ENGINE_NAMES:
        assert member_engines(_token_task())[name]().resources.memory_mb is None
        assert member_engines(_agentic_task())[name]().resources.memory_mb is None
        assert member_engines(provisioned)[name]().resources.memory_mb == 1024


def test_batch_engines_are_refused_with_the_reason(prov, tmp_path):
    """A serving submits clocked requests to an engine's async API one at a time, so a
    batch-oriented HF engine cannot be a pool member. The refusal has to SAY that --
    "unknown engine" would send a user hunting for a typo -- and it must fire before
    anything is deployed."""
    for name in ("hf_pinned", "hf_default", "hf_flash_attn"):
        with pytest.raises(CalibrationError, match="clocked|asynchronous"):
            validate_pool([name])
    with pytest.raises(CalibrationError, match="unknown"):
        validate_pool(["not_a_real_engine"])

    deploy = _make_deploy(_FakePinned(0.1))
    with pytest.raises(CalibrationError, match="clocked|asynchronous"):
        _run(_token_task(), prov, engines=["hf_pinned"], deploy=deploy,
             captures=tmp_path / "caps")
    assert deploy.deployed == []


def test_pool_members_are_the_async_serving_engines():
    """The pool is built from the reference engines that implement ClockedServing --
    the capability IS the membership criterion, so a member that cannot take clocked
    traffic is unrepresentable rather than merely discouraged."""
    from hls.calibrate import member_engines
    from hls.reference.engines import ClockedServing

    built = {name: build() for name, build in member_engines(_token_task()).items()}
    assert set(built) == set(POOL_ENGINE_NAMES)
    assert all(isinstance(e, ClockedServing) for e in built.values())
    assert all(e.model_spec == _token_task().model for e in built.values())


def _penalized_task() -> TaskSpec:
    """The same calibration task, but its verifier's reference declares a frequency
    penalty -- so every engine of the run has to serve and score under it."""
    return _token_task(verifiers=[TokenFaithfulnessVerifier(
        reference=HFReference(decode=Greedy(frequency_penalty=0.1)))])


def test_every_engine_of_a_run_carries_the_tasks_decode():
    """A penalized task's pool members, its pinned reference and its fp32 preflight
    anchor all decode under the penalty the verifier declares. A member serving plain
    greedy would emit tokens the penalized reference scores as flips, so its measured
    noise floor would be the penalty rather than the stack's numerics."""
    from hls.calibrate import _task_decode, member_engines, pinned_reference

    task = _penalized_task()
    assert _task_decode(task) == Greedy(frequency_penalty=0.1)
    members = {name: build() for name, build in member_engines(task).items()}
    assert all(e.decode.frequency_penalty == 0.1 for e in members.values())
    assert pinned_reference(task).decode.frequency_penalty == 0.1


def test_calibrated_verifiers_that_disagree_on_the_decode_are_refused():
    """Each decode has its own flip rates, so a task carrying two of them has no
    single noise floor to measure."""
    from hls.calibrate import _task_decode

    task = _token_task(verifiers=[
        TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy())),
        TokenFaithfulnessVerifier(
            reference=HFReference(decode=Greedy(frequency_penalty=0.1)))])
    with pytest.raises(CalibrationError, match="(?i)decode"):
        _task_decode(task)


def _chunked_task(query_chunk: int) -> TaskSpec:
    """The token task with its oracle's teacher-forcing chunk set."""
    return _token_task(verifiers=[TokenFaithfulnessVerifier(
        reference=HFReference(decode=Greedy(), query_chunk=query_chunk))])


def test_every_hf_engine_of_a_run_is_tiled_at_the_tasks_chunk():
    """The pinned reference and the fp32 preflight anchor tile their forward at the
    chunk the task declares. A run measured at one width and graded at another is two
    instruments, which is what the ceilings cannot survive."""
    import asyncio

    import hls.calibrate as calibrate

    task = _chunked_task(4096)
    assert calibrate._task_query_chunk(task) == 4096
    assert calibrate.pinned_reference(task).query_chunk == 4096

    # The anchor is built inside _run_preflight, so it is read off the constructor: a
    # deploy that refuses stops the coroutine the line after.
    built: dict = {}

    class _Recorder:
        def __init__(self, **kwargs):
            built.update(kwargs)

    class _NoDeploy(Exception):
        pass

    async def _deploy(name, engine):
        raise _NoDeploy

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(calibrate, "HFEngine", _Recorder)
        with pytest.raises(_NoDeploy):
            asyncio.run(calibrate._run_preflight(task, _deploy, None))
    assert built["query_chunk"] == 4096


def test_calibrated_verifiers_that_disagree_on_the_chunk_are_refused():
    """The math kernel tiles differently at each width, so a task carrying two of them
    has no single noise floor to measure."""
    from hls.calibrate import _task_query_chunk

    task = _token_task(verifiers=[
        TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy(), query_chunk=512)),
        TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy(), query_chunk=4096))])
    with pytest.raises(CalibrationError, match="(?i)query_chunk"):
        _task_query_chunk(task)


# ---- the sessions a serving drives ------------------------------------------

def test_sessions_are_read_from_the_rows_directory_handed_in(tmp_path):
    """A serving's sessions are read from the provisioned rows the caller names, laid out
    ``tenants/<id>/<split>`` as a role mounts them under ``/data``, and nothing else."""
    loaded = load_sessions(_token_task(), _provisioned(tmp_path), Split.TEST)
    assert [t.id for t, _ in loaded] == ["t0"]
    (tenant, items), = loaded
    assert len(items) == len(_MMLU_ROWS)
    assert items[0].turns[0].first_of("text").text == "What is 2+2?"
    # The workload rides along (it is what the member clocks the traffic with) but the
    # verifiers do not: a member only generates traffic, and shipping a verifier would
    # drag its reference engine into a container built for a serving stack.
    assert tenant.workload == _token_task().effective_tenants()[0].workload
    assert tenant.verifiers == []


def test_sessions_are_one_split(prov, tmp_path):
    """ONE split, not the whole dataset: the arrival schedule is drawn for the session
    count of the split being served, so mixing splits would serve a workload no run
    ever offers."""
    prov = _provisioned(tmp_path)
    dev_rows = [dict(r, question=f"[dev] {r['question']}") for r in _MMLU_ROWS[:2]]
    (prov / "tenants" / "t0" / "dev" / "rows.json").write_text(json.dumps(dev_rows))
    task = _token_task()
    (_t, test_items), = load_sessions(task, prov, Split.TEST)
    (_d, dev_items), = load_sessions(task, prov, Split.DEV)
    assert len(test_items) == 4 and len(dev_items) == 2
    assert all("[dev]" in i.turns[0].first_of("text").text for i in dev_items)


def test_sessions_refuse_a_task_with_no_traffic(tmp_path):
    """A task whose tenants have no provisioned rows calibrates nothing; saying so
    beats emitting a ceiling measured on an empty pool."""
    task = TaskSpec(id="T", model=_model(), hardware=_hardware(), contract=_contract(),
                    workload=_workload(), scorer=_scorer(),
                    verifiers=[MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)])
    with pytest.raises(CalibrationError, match="no .*traffic"):
        load_sessions(task, _provisioned(tmp_path), Split.TEST)


def test_missing_split_fails_loud_naming_the_fix(prov, tmp_path):
    prov = _provisioned(tmp_path)
    (prov / "tenants" / "t0" / "test" / "rows.json").unlink()
    with pytest.raises(CalibrationError, match="(?s)'test' split.*provision stage"):
        load_sessions(_token_task(), prov, Split.TEST)


def test_each_tenant_reads_its_own_rows_and_a_tenant_without_data_drives_none(tmp_path):
    """Every tenant with a data source reads its split from its own ``tenants/<id>/<split>``, and
    one without contributes no sessions. A tenant whose rows are missing is refused by name."""
    def tenant(tid: str, source: DataSource | None) -> Tenant:
        return Tenant(id=tid, data_source=source, workload=_workload(),
                      verifiers=[MetricVerifier(metric=LetterMatchMetric(), min_value=1.0)])

    task = TaskSpec(id="T", model=_model(), hardware=_hardware(), contract=_contract(), scorer=_scorer(),
                    tenants=[tenant("t0", _data_source()), tenant("t1", _data_source()), tenant("t2", None)])
    prov = tmp_path / "prov"
    for tid in ("t0", "t1"):
        (prov / "tenants" / tid / "test").mkdir(parents=True)
        rows = [dict(r, question=f"[{tid}] {r['question']}") for r in _MMLU_ROWS]
        (prov / "tenants" / tid / "test" / "rows.json").write_text(json.dumps(rows))
    loaded = load_sessions(task, prov, Split.TEST)
    assert [t.id for t, _ in loaded] == ["t0", "t1"]
    for t, items in loaded:
        assert [i.turns[0].first_of("text").text for i in items] == [f"[{t.id}] {r['question']}"
                                                                      for r in _MMLU_ROWS]
    shutil.rmtree(prov / "tenants" / "t1")
    with pytest.raises(CalibrationError, match="(?s)the 'test' split for tenant 't1'.*provision stage"):
        load_sessions(task, prov, Split.TEST)


# ---- score_draw and the shared tally ----------------------------------------

def _tf_item(tokens=3, request_id="r.0", *, cap=1_000_000) -> Envelope:
    """One teacher-force item as ``capture_items`` produces it: tenant ``t0``, the
    request's cap, and ``tokens`` served ids."""
    return Envelope(request_id=request_id,
                    metadata={"messages": [], "tenant_id": "t0", "max_tokens": cap},
                    contents=[TextPayload(text="x", tokens=[1] * tokens)])


def _rows(**caps) -> RowIndex:
    """A row index for ``_tf_item`` rows: every row ``r`` belongs to ``t0`` under a
    1M cap unless ``caps`` says otherwise."""
    return RowIndex(rows_of_tenant={"t0": {"r", "hot", "cold"}},
                    cap_of_tenant={"t0": caps.get("t0", 1_000_000)})


def test_score_draw_counts_flips_strictly_positive():
    """A flip is gap > 0 exactly: an exact bf16 tie -- a different token id at an
    identical log-probability -- has gap 0.0 and is NOT a flip."""
    stats = score_draw("vllm", 0, 7, _MEMBER_VERSION, [_tf_item(3), _tf_item(1)],
                       [[0.0, 0.3, 0.0], [1.5]])
    assert stats.n_tokens == 4
    assert stats.n_flips == 2 and stats.flip_rate == 0.5
    assert stats.max_gap == 1.5
    assert stats.seed == 7


def test_score_draw_counts_events_at_every_level_the_gap_clears():
    """level_events(grid)[i] counts flips with gap >= grid[i], so a single large gap
    contributes to every rung below it -- the nesting the curve rule is built on. The
    grid is a parameter: the records keep the severe gaps, not counts on one grid."""
    stats = score_draw("vllm", 0, 0, _MEMBER_VERSION, [_tf_item(4)], [[0.6, 2.2, 0.0, 0.1]])
    counts = stats.level_events(LEVELS_NATS)
    by_level = dict(zip(LEVELS_NATS, counts))
    assert by_level[0.5] == 2      # 0.6 and 2.2
    assert by_level[1.0] == 1 and by_level[2.0] == 1
    assert by_level[2.5] == 0
    assert counts == sorted(counts, reverse=True)
    assert stats.level_events([0.5, 2.2, 6.0]) == [2, 1, 0]


def test_score_draw_records_the_prompt_behind_every_severe_event():
    """The recurrence census needs to know WHICH prompt produced a severe flip, at
    every level it reached -- levels are nested, so a 2.2-nat flip makes its prompt a
    site at 0.5, 1.0, 1.5 and 2.0 but not above."""
    items = [_tf_item(3, request_id="hot.0"), _tf_item(3, request_id="cold.0")]
    stats = score_draw("vllm", 0, 0, _MEMBER_VERSION, items, [[2.2, 0.0, 0.0], [0.1, 0.0, 0.0]])
    by_level = dict(zip(LEVELS_NATS, stats.severe_sites(LEVELS_NATS)))
    # Sites are event keys -- the tenant-qualified form the verifier uses -- so a
    # census over a multi-tenant campaign never conflates two tenants' rows.
    assert by_level[0.5] == by_level[2.0] == ("t0:hot.0",)
    assert by_level[2.5] == ()


def test_score_draw_counts_replies_that_reach_the_cap_and_marks_them():
    """A reply whose served length reached the request's cap is counted like any other
    and marked ``at_cap`` on its record -- by length against the stamped cap, never by
    the engine's own label. The mark is what ``capped_rows`` reads to derive a task's
    removal list; the tally itself leaves nothing out."""
    items = [_tf_item(2), _tf_item(3, request_id="hot.0", cap=3)]
    scored = score_draw("vllm", 0, 0, _MEMBER_VERSION, items, [[0.0, 0.9], [0.0, 0.6, 0.0]])
    stats = scored
    assert stats.n_tokens == 5 and stats.n_flips == 2
    assert stats.capped_events == 1 and stats.capped_tokens == 3
    assert stats.flip_rate == 0.4
    assert [ev.at_cap for ev in scored.events] == [False, True]


def test_score_draw_records_the_stop_row():
    """The reference returns one gap per served token plus one for the stop when it
    scored it; the per-event record keeps that gap apart from the served positions,
    and the reply's token count includes the extra position."""
    scored = score_draw("vllm", 0, 0, _MEMBER_VERSION, [_tf_item(2), _tf_item(2, request_id="hot.0")],
                        [[0.0, 0.0, 4.0], [0.0, 0.0]])
    stopped, ended = scored.events
    assert stopped.stop_gap == 4.0 and stopped.n_tokens == 3 and stopped.n_flips == 1
    assert ended.stop_gap is None and ended.n_tokens == 2
    assert scored.n_tokens == 5 and scored.max_gap == 4.0


def test_score_draw_refuses_gaps_that_do_not_line_up_with_the_reply():
    """Any gap count other than the served length or one more is a malformed reference
    reply, refused rather than counted position-blind."""
    with pytest.raises(CalibrationError, match="malformed gap list"):
        score_draw("vllm", 0, 0, _MEMBER_VERSION, [_tf_item(2)], [[0.0, 0.0, 0.0, 0.0]])


def test_score_draw_empty():
    stats = score_draw("vllm", 0, 0, _MEMBER_VERSION, [], [])
    assert (stats.n_tokens, stats.n_flips, stats.max_gap) == (0, 0, 0.0)
    assert stats.flip_rate == 0.0


def test_score_draw_refuses_non_finite_gaps():
    """Every gap here comes from teacher-forcing a TRUSTED engine's own reply, so NaN
    means the instrument's arithmetic is broken and +inf means an out-of-vocabulary id.
    NaN especially must not pass silently: it loses every > comparison, so a
    NaN-producing engine would otherwise score a perfect 0.0 rate and sail under the
    trust bounds written for exactly that defect class."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(CalibrationError, match="non-finite"):
            score_draw("vllm", 0, 0, _MEMBER_VERSION, [_tf_item(2)], [[0.0, bad]])


def test_calibration_and_the_verifier_count_identically():
    """The measurement is ONE piece of code. The same gaps counted by
    calibration and by the deployed verifier must agree on the token count, the flip
    count, the worst gap and every level count -- otherwise the two sides of the gate
    would be comparing numbers that do not mean the same thing."""
    from hls.verify.flips import PooledFlips, score_reply

    gaps = [[0.0, 0.3, 2.2], [0.6, 0.0], [1.0, 4.5, 0.0, 0.0]]
    items = [_tf_item(len(g), request_id=f"r{i}.0") for i, g in enumerate(gaps)]
    stats = score_draw("vllm", 0, 0, _MEMBER_VERSION, items, gaps)

    flips = PooledFlips(LEVELS_NATS)
    for i, g in enumerate(gaps):
        flips.add(score_reply(g, event_key=f"r{i}.0", n_served=len(g), at_cap=False))
    assert (flips.n_tokens, flips.n_flips, flips.max_gap) == (
        stats.n_tokens, stats.n_flips, stats.max_gap)
    assert flips.level_counts() == stats.level_events(LEVELS_NATS)


def test_a_scored_serving_round_trips_through_the_capture_store(tmp_path):
    """The one phase-2 file holds the serving's identity as a header line and one
    record per reply in the shared schema; the tally is derived from the records on
    read and comes back equal to the one that was scored."""
    store = CaptureStore(tmp_path, "fingerprint")
    scored = score_draw("vllm", 7, 1234, _MEMBER_VERSION, [_tf_item(3), _tf_item(2, request_id="hot.0", cap=2)],
                        [[0.0, 0.6, 2.2], [0.0, 0.0]])
    store.write_scored(scored)
    assert store.read_scored("vllm", 7) == scored
    import gzip
    with gzip.open(store.replies_path("vllm", 7), "rt") as fh:
        header, *records = [json.loads(line) for line in fh]
    assert header == {"engine": "vllm", "draw": 7, "seed": 1234,
                      "serving_version": _MEMBER_VERSION}
    assert records == [
        {"event_key": "t0:r.0", "n_tokens": 3, "n_flips": 2, "max_gap": 2.2,
         "severe_gaps": [0.6, 2.2], "n_nan": 0, "n_inf": 0, "stop_gap": None,
         "at_cap": False},
        {"event_key": "t0:hot.0", "n_tokens": 2, "n_flips": 0, "max_gap": 0.0,
         "severe_gaps": [], "n_nan": 0, "n_inf": 0, "stop_gap": None, "at_cap": True}]


def _rewrite_replies(store, engine, draw, edit):
    """Rewrite one stored replies file through ``edit(header, records)``."""
    import gzip
    path = store.replies_path(engine, draw)
    with gzip.open(path, "rt") as fh:
        header, *records = [json.loads(line) for line in fh]
    edit(header, records)
    with gzip.open(path, "wt") as fh:
        for rec in [header, *records]:
            fh.write(json.dumps(rec) + "\n")


def test_stored_replies_count_on_any_grid_that_starts_at_the_floor(tmp_path):
    """The records keep the severe gaps themselves, so the same stored serving yields
    its counts on whatever grid the pool ends up with -- a re-grid is a file read."""
    from hls.verify.flips import candidate_rungs

    store = CaptureStore(tmp_path, "fingerprint")
    store.write_scored(score_draw("vllm", 0, 0, _MEMBER_VERSION, [_tf_item(3)], [[0.0, 0.6, 5.2]]))
    stats = store.read_scored("vllm", 0)
    assert stats.level_events([0.5, 1.0, 1.5]) == [2, 1, 1]
    assert candidate_rungs(5.2)[-2:] == [5.5, 6.0]
    assert stats.level_events(candidate_rungs(5.2))[-3:] == [1, 0, 0]


def test_pool_grid_keeps_a_rung_only_where_the_binding_clean_count_drops():
    """One 6-nat clean event makes every rung from 1.5 to 6.0 carry the same count and
    therefore the same ceiling, so the grid keeps 1.5 and then jumps to the first rung
    no clean engine reached. The binding count is the max over engines."""
    from hls.calibrate import pool_grid

    def serving(engine, draw, gaps):
        return score_draw(engine, draw, draw, _MEMBER_VERSION, [_tf_item(len(gaps))], [gaps])

    sglang = [serving("sglang", 0, [0.7] * 84 + [1.2] * 11 + [6.0] + [0.0] * 4)]
    vllm = [serving("vllm", 0, [0.7] * 78 + [1.2] * 6 + [0.0] * 16)]
    assert pool_grid({"sglang": sglang, "vllm": vllm}) == [0.5, 1.0, 1.5, 6.5]
    # A pool whose clean flips thin out gradually keeps a rung at every step.
    spread = [serving("x", 0, [0.6, 1.1, 1.6, 2.1, 2.6, 3.1, 0.0])]
    assert pool_grid({"x": spread}) == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    # No clean flip above the floor: the floor and the first empty rung.
    quiet = [serving("q", 0, [0.1, 0.2, 0.0])]
    assert pool_grid({"q": quiet}) == [0.5]


def test_scored_replies_moved_to_another_servings_path_are_refused(tmp_path):
    """The header carries the serving's identity so a file that is not where its path
    says cannot be read as that serving."""
    store = CaptureStore(tmp_path, "fingerprint")
    store.write_scored(score_draw("vllm", 0, 0, _MEMBER_VERSION, [_tf_item(2)], [[0.0, 0.6]]))
    store.replies_path("vllm", 0).rename(store.replies_path("vllm", 3))
    with pytest.raises(CalibrationError, match="moved") as exc:
        store.read_scored("vllm", 3)
    assert str(store.replies_path("vllm", 3)) in str(exc.value)


def test_capture_store_round_trips_the_serving_seed(tmp_path):
    """The seed a serving was clocked with rides with its capture, so a stored serving
    can be re-driven exactly."""
    store = CaptureStore(tmp_path, "fingerprint")
    store.write_capture("vllm", 3, 4242, _MEMBER_VERSION, [_tf_item(2)])
    items, seed, version = store.read_capture("vllm", 3, rows=_rows())
    assert seed == 4242
    assert items[0].first_of("text").tokens == [1, 1]
    assert items[0].metadata == {"messages": [], "tenant_id": "t0", "max_tokens": 1_000_000}


# ---- the join key: a capture item and a trace event name one request alike -----

def _driven_event(task, prov, *, tokens: int, cap: int | None = None) -> TraceEvent:
    """One event exactly as the drive loop records it: the request shaped by
    ``stamp_request`` (tenant, session, turn and the workload's cap), the reply with
    its emitted ids."""
    from hls.spec.drivers import stamp_request, user_message

    (tenant, items), = load_sessions(task, prov, Split.TEST)
    item = items[0]
    turn = item.turns[0]
    cap = tenant.workload.output_length_cap.max_tokens() if cap is None else cap
    req = stamp_request(turn, [user_message(turn)], tenant_id=tenant.id,
                        session_id=item.request_id, turn_idx=0, max_tokens=cap)
    reply = Envelope(request_id=f"{item.request_id}.0",
                     contents=[TextPayload(text="x", tokens=list(range(tokens)))])
    return TraceEvent(request_id=f"{item.request_id}.0", session_id=item.request_id,
                      turn_index=0, input=req, response=ArtifactResponse(output=reply),
                      t_send=0.0, t_recv=1.0)


def test_a_capture_item_and_a_trace_event_share_one_event_key(prov):
    """Per-event records are keyed the same way on both sides, so the key a calibration
    item carries has to be the key the verifier reads off a trace event for the same
    request -- tenant and all, built from the same stamps."""
    task = _token_task()
    ev = _driven_event(task, prov, tokens=5)
    item, = capture_items(Trace(events=[ev]), engine="vllm", draw=0)
    assert item_event_key(item) == ev.event_key == "t0:0.0"
    assert item.metadata["max_tokens"] == ev.input.metadata["max_tokens"]


def test_a_serving_the_scheduled_stop_cut_is_refused(prov):
    """A ceiling is measured on replies a graded run can have. A cut reply ends where
    the harness stopped listening, and the requests still running when the window
    closed are the longest ones -- so capturing them measures truncated replies and
    dropping them measures an easier workload."""
    from hls.spec.workload import CUT_EXTRA

    task = _token_task()
    whole = _driven_event(task, prov, tokens=5)
    cut = whole.model_copy(update={"request_id": "0.1",
                                   "extras": {CUT_EXTRA: True}})
    with pytest.raises(CalibrationError, match="scheduled stop cut 1 of 2"):
        capture_items(Trace(events=[whole, cut]), engine="vllm", draw=0)


# ---- tool definitions ride along on an agentic request -----------------------

def test_a_capture_item_keeps_the_tools_the_request_carried():
    """An agentic task's request carries the tool set the pool member rendered into
    its prompt (a different one on different turns); the teacher-force item must keep
    it, or the pinned reference renders a context the member was never sent. A request
    with no tools (or an empty list) gets no ``tools`` key at all -- the same rendering
    a chat template produces with no tools argument."""
    from hls.calibrate import _teacher_force_item

    tools = [{"type": "function", "function": {"name": "lookup"}}]
    reply = Envelope(request_id="r.0", contents=[TextPayload(text="x", tokens=[1, 2])])

    with_tools = Envelope(request_id="r.0",
                          metadata={"messages": [], "tenant_id": "t", "max_tokens": 8,
                                    "tools": tools})
    item = _teacher_force_item(with_tools, reply, source="test")
    assert item.metadata["tools"] == tools

    for empty in ({"messages": [], "tenant_id": "t", "max_tokens": 8},
                  {"messages": [], "tenant_id": "t", "max_tokens": 8, "tools": []}):
        item = _teacher_force_item(Envelope(request_id="r.0", metadata=empty), reply,
                                   source="test")
        assert "tools" not in item.metadata


def test_the_capture_store_round_trips_tools(tmp_path):
    """A stored capture keeps the tools an agentic item carried and restores them on
    read; an item with none stays without the key, so a non-agentic capture's record
    shape is unchanged."""
    store = CaptureStore(tmp_path, "fingerprint")
    tools = [{"type": "function", "function": {"name": "lookup"}}]
    with_tools = Envelope(request_id="hot.0",
                          metadata={"messages": [], "tenant_id": "t0", "max_tokens": 1_000_000,
                                    "tools": tools},
                          contents=[TextPayload(text="x", tokens=[1, 1])])
    store.write_capture("vllm", 0, 1, _MEMBER_VERSION, [with_tools, _tf_item(2)])
    items, _seed, _version = store.read_capture("vllm", 0, rows=_rows())
    assert items[0].metadata["tools"] == tools
    assert "tools" not in items[1].metadata


def test_a_capture_with_no_replies_is_refused_naming_the_file(tmp_path):
    """A serving that answered nothing is refused at read, naming the file, rather than
    scored as an empty serving several phases later."""
    store = CaptureStore(tmp_path, "fp")
    store.write_capture("vllm", 0, 1, _MEMBER_VERSION, [])
    with pytest.raises(CalibrationError, match="holds no replies") as exc:
        store.read_capture("vllm", 0, rows=_rows())
    assert str(store.capture_path("vllm", 0)) in str(exc.value)


def test_a_capture_records_the_member_version_that_produced_it(tmp_path):
    """The serving package lives in the member's container, so the capture is where the
    version has to be recorded: a campaign that only scores stored captures still names
    the build that drove them."""
    store = CaptureStore(tmp_path, "fp")
    store.write_capture("vllm", 0, 1, "1.2.3", [_tf_item(2)])
    _items, seed, version = store.read_capture("vllm", 0, rows=_rows())
    assert (seed, version) == (1, "1.2.3")


def test_the_row_index_keys_rows_within_their_tenant(prov):
    """Native row ids are per corpus, so two tenants over one corpus each serve the
    same row id; a record is keyed by its own tenant and never by the row alone."""
    (tenant, items), = load_sessions(_token_task(), prov, Split.TEST)
    twin = tenant.model_copy(update={"id": "t1"})
    rows = RowIndex.from_sessions([(tenant, items), (twin, items)])
    rows.check("t0", "2.0")
    rows.check("t1", "2.0")
    with pytest.raises(CalibrationError, match="does not serve"):
        rows.check("t2", "2.0")


def test_the_row_index_refuses_a_row_outside_the_split(prov):
    rows = RowIndex.from_sessions(load_sessions(_token_task(), prov, Split.TEST))
    rows.check("t0", "2.0")
    with pytest.raises(CalibrationError, match="does not serve"):
        rows.check("t0", "elsewhere-test-9.0")


def test_a_stamped_capture_that_disagrees_with_the_task_is_refused(tmp_path):
    """A record that carries a tenant or cap the task does not serve that row under
    was captured over a different workload; scoring it here would pool two
    workloads."""
    store = CaptureStore(tmp_path, "fp")
    store.write_capture("vllm", 0, 1, _MEMBER_VERSION, [_tf_item(2, cap=16)])
    with pytest.raises(CalibrationError, match="different workload"):
        store.read_capture("vllm", 0, rows=_rows(t0=1_000_000))


# ---- serving seeds ----------------------------------------------------------

def test_serving_seeds_differ_per_serving_and_are_reproducible():
    """Each serving gets its own clock (different arrivals, think-times and ramp-down
    membership), and every one of them is reproducible from the task alone -- which is
    what lets a rerun, or a second machine, extend the same campaign."""
    task = _token_task()
    seeds = [serving_seed_of(task.seed, i) for i in range(6)]
    assert len(set(seeds)) == 6
    assert all(0 <= s < 2 ** 31 for s in seeds)
    assert [serving_seed_of(task.seed, i) for i in range(6)] == seeds
    other = _token_task(seed=task.seed + 1)
    assert serving_seed_of(other.seed, 0) != seeds[0]


# ---- the counter stop ---------------------------------------------------------

def test_a_campaign_serves_until_the_exposure_target_is_reached(prov, tmp_path):
    """The stop rule is a token counter against a constant: keep serving until pooled
    gradable exposure reaches the target. Nothing about the measured flip rate enters,
    which is what makes the (P) look below a single look at a pre-committed size."""
    task = _token_task()
    member = _FakeMember(replies=4, tokens=2500)     # 10,000 tokens per serving
    deploy = _make_deploy(_FakePinned(0.10), member=member)
    cal = _run(task, prov, deploy=deploy,
               captures=tmp_path / "caps", mu_per_mtok=_mu_for_target(85_000))

    exposure = cal.curve_side.engines["vllm"].exposure
    assert exposure >= 85_000
    # ...and it stopped as soon as it got there: one serving fewer would not have.
    assert exposure - 10_000 < 85_000
    assert cal.rate_side["vllm"].draws == 9
    assert cal.tail["vllm"].rhs == pytest.approx(85_000)
    assert cal.sizing.stopped == "exposure target reached"
    assert len(member.drives) == 9
    # The instrument check runs first and releases its engines; then one member deploy
    # per serving. The peak of 2 is the preflight's anchor and reference pair, which the
    # instrument role holds on two GPUs.
    assert deploy.deployed[:2] == ["pinned", "fp32_anchor"]
    assert sum(1 for n in deploy.deployed if n.startswith("vllm-")) == 9
    assert deploy.peak[0] == 2
    assert deploy.live == []


def test_the_serving_floor_holds_even_when_one_serving_would_suffice(prov, tmp_path):
    """Exposure alone would stop a big-serving task after one drive, and a standard
    deviation over one sample is not an estimate. The floor is an engineering default
    under (P)'s spread estimate, not a statistical claim of its own."""
    task = _token_task()
    drive = _FakeMember(replies=4, tokens=25_000)    # 100,000 tokens per serving
    cal = _run(task, prov, deploy=_make_deploy(_FakePinned(0.10), member=drive),
               captures=tmp_path / "caps", mu_per_mtok=_mu_for_target(50_000))
    assert cal.rate_side["vllm"].draws == MIN_SERVINGS


def test_a_campaign_produces_both_rules(prov, tmp_path):
    """The end-to-end shape: both rules are populated from the same servings, q_max is
    the worst engine's mean, and the artifact records what produced it."""
    task = _token_task()
    cal = _run(task, prov, deploy=_make_deploy(_FakePinned(0.10)), captures=tmp_path / "caps")

    assert set(cal.rate_side) == set(cal.curve_side.engines) == {"vllm"}
    assert cal.q_max == max(r.mean for r in cal.rate_side.values())
    assert cal.rate_side["vllm"].max_gap == 0.10
    assert cal.tolerances.eps == DEFAULT_EPS
    assert cal.gpu_kind == "B200" and cal.stack.device_name == "B200"
    assert cal.preflight.passed is True
    assert cal.task == "tasks.demo.task"
    assert cal.engines["vllm"].config.startswith("VLLMEngine(")
    assert "clocked serving" in cal.engines["vllm"].config
    assert cal.engines["vllm"].version == _MEMBER_VERSION
    # One seed per serving, keyed by engine, in serving order -- so a named serving
    # can be re-driven exactly.
    assert cal.draw.serving_seeds == {
        "vllm": tuple(serving_seed_of(task.seed, i)
                      for i in range(cal.rate_side["vllm"].draws))}


# ---- top-ups ------------------------------------------------------------------

def test_a_top_up_round_closes_an_exposure_undershoot(prov, tmp_path):
    """The first round rests on the serving floor, and the next on an ESTIMATED
    per-serving yield, so a round can land short of the exposure target. That is not a
    failure: the grower asks again with the yield just measured, and the stop it converges
    on is the same deterministic token counter."""
    task = _token_task()
    # 10,000 tokens per serving against an 85,000-token target: the serving floor of 8
    # leaves the campaign 5,000 short, so exactly one top-up serving is owed.
    deploy = _make_deploy(_FakePinned(0.10), member=_FakeMember(replies=4, tokens=2500))
    cal = _run(task, prov, deploy=deploy, captures=tmp_path / "caps", mu_per_mtok=_mu_for_target(85_000))
    assert cal.rate_side["vllm"].draws == MIN_SERVINGS + 1
    assert cal.curve_side.engines["vllm"].exposure >= 85_000
    # Two rounds, each scored as one share: the floor, then the one serving the counter owed.
    assert [n for n in deploy.deployed if n.startswith("vllm-")][-1] == "vllm-8"
    assert [n for n in deploy.deployed if n.startswith("score-")] == ["score-1", "score-2"]


class _GrowingYieldMember:
    """A member whose serving is a function of its seed: each of the four sessions gets one
    reply of 2,500 ids in the first ``MIN_SERVINGS`` servings and of 25,000 in every later
    one, starting at an id read off the seed, which ``_SeededPinned`` reads its flips off."""

    def __init__(self, task: TaskSpec):
        #: The serving index each seed clocks.
        self.draw_of = {serving_seed_of(task.seed, d): d for d in range(200)}
        #: The seed of each serving driven, in order.
        self.seeds: list[int] = []

    async def serving_version(self) -> str:
        return _MEMBER_VERSION

    async def serve_traffic(self, sessions, *, seed, request_timeout_sec, max_sessions=None,
                            drive_timeout_sec=None, agent=None, max_context_len=None):
        self.seeds.append(seed)
        n = 2500 if self.draw_of[seed] < MIN_SERVINGS else 25_000
        tenant, items = sessions[0]
        cap = tenant.workload.output_length_cap.max_tokens()
        events = []
        for i, item in enumerate(items):
            rid, first = f"{item.request_id}.0", (seed + 31 * i) % 1009
            env = Envelope(request_id=rid, metadata={"messages": [{"role": "user", "content": f"q{i}"}],
                                                     "tenant_id": tenant.id, "max_tokens": cap})
            events.append(TraceEvent(
                request_id=rid, session_id=item.request_id, turn_index=0, input=env,
                response=ArtifactResponse(output=Envelope(request_id=rid, contents=[
                    TextPayload(text="x", tokens=[first + j for j in range(min(n, cap))], finish_reason="stop")])),
                t_send=0.0, t_recv=1.0))
        return Trace(events=events)


class _SeededPinned:
    """A pinned reference whose gaps are a function of each reply's ids: the preflight's probes
    score no gap, and a reply flips at its first 5 to 21 positions, the count and the gaps read
    off its first id."""

    async def teacher_force(self, items):
        out = []
        for item in items:
            tokens = item.first_of("text").tokens
            if item.request_id.startswith("preflight-"):
                out.append([0.0] * len(tokens))
                continue
            k = min(len(tokens), 5 + tokens[0] % 17)
            out.append([0.05 + ((tokens[0] + j) % 23) * 0.3 for j in range(k)] + [0.0] * (len(tokens) - k))
        return out

    async def stack_report(self):
        return dict(_STACK)


@pytest.mark.parametrize(("fanout", "servings", "written"), [(1, 9, True), (4, 12, False)])
def test_a_top_up_round_takes_at_most_fanout_servings(prov, tmp_path, fanout, servings, written):
    """A top-up rests on the yield its engine's servings measured so far, so a round takes at
    most --fanout of what the counter owes and the counter is asked again once they are
    scored, one wave of a top-up at a time; these are its serving
    counts and verdicts. The floor here yields a tenth of every later serving, so the first
    estimate owes seven servings where one closes the target: at a fanout of 1 one more serving
    is taken and the calibration is written; at 4, four, whose spread fails (P)."""
    task = _token_task()
    member = _GrowingYieldMember(task)
    deploy = _make_deploy(_SeededPinned(), member=member)

    def run():
        return _run(task, prov, deploy=deploy, captures=tmp_path / "caps", mu_per_mtok=_mu_for_target(150_000),
                    eps=1.0, fanout=fanout)

    if written:
        assert run().rate_side["vllm"].draws == servings
    else:
        with pytest.raises(CalibrationError, match="precision target"):
            run()
    assert member.seeds == [serving_seed_of(task.seed, d) for d in range(servings)]


def test_serving_seeds_are_recorded_per_engine_in_serving_order(prov, tmp_path):
    """``serving_seed_of`` is engine-INDEPENDENT: vllm's serving 3 and sglang's serving 3
    are clocked from the same seed. So the record is per engine, in draw-index order,
    which is what lets any one serving be named by its index and re-driven. Pooling the
    two engines into one flat set would halve the recorded count and destroy that
    alignment."""
    cal = _run(_token_task(), prov, engines=("vllm", "sglang"), deploy=_make_deploy(_FakePinned(0.10)),
               captures=tmp_path / "caps")
    seeds = cal.draw.serving_seeds
    assert set(seeds) == {"vllm", "sglang"}
    for engine, engine_seeds in seeds.items():
        n = cal.rate_side[engine].draws
        assert list(engine_seeds) == [
            serving_seed_of(_token_task().seed, i) for i in range(n)], engine
    # Both engines drew the same seed sequence, which is exactly why a flat set cannot
    # represent the pool: it would collapse to half the servings.
    assert seeds["vllm"] == seeds["sglang"]
    total = sum(len(s) for s in seeds.values())
    assert total == sum(cal.rate_side[e].draws for e in seeds)
    assert len({s for row in seeds.values() for s in row}) == total // 2


# ---- (P): the one look ------------------------------------------------------

def test_precision_is_a_single_look_when_it_passes(prov, tmp_path):
    """A reproducible engine is evaluated once, at the size the counter stop already
    fixed -- no sequential testing, so the chi-square bound holds at face value. The
    verdicts are not stored: both sides are recomputed from the servings and the
    tolerances the artifact carries."""
    cal = _run(_token_task(), prov, deploy=_make_deploy(_FakePinned(0.10)), captures=tmp_path / "caps")
    assert all(c.passed for c in cal.precision.values())
    assert cal.sizing.c_m > 1.0
    check = cal.precision["vllm"]
    rate = cal.rate_side["vllm"]
    assert check.lhs == pytest.approx(cal.sizing.c_m * rate.sd, rel=1e-12)
    assert check.rhs == pytest.approx(cal.tolerances.eps * rate.mean, rel=1e-12)


def test_precision_failure_refuses_naming_the_engine_and_both_sides(prov, tmp_path):
    """One look and no more: a second would spend the spread bound's error budget for a
    number that has already been measured, and more servings shrink the confidence
    factor on the spread rather than the spread itself. The refusal names the engine,
    quotes the measured spread against the tolerance, and names both remedies."""
    def flips(draw):
        return 20 + 60 * (draw % 2)

    with pytest.raises(CalibrationError) as exc:
        _run(_token_task(), prov,
             deploy=_make_deploy(_FakePinned(0.10, flips_for=flips)),
             captures=tmp_path / "caps", eps=0.01)
    message = str(exc.value)
    assert "precision target" in message and "eps=0.01" in message
    assert "vllm" in message and "spread" in message and "--eps" in message
    assert "at both looks" not in message


def test_a_marginal_spread_is_not_topped_up(prov, tmp_path):
    """The campaign never adds servings to rescue (P): the sample size is what the
    exposure counter fixed, so a marginal engine is refused at that size rather than
    served until it passes."""
    def flips(draw):
        return 26 if draw == 7 else 20

    deploy = _make_deploy(_FakePinned(0.10, flips_for=flips))
    with pytest.raises(CalibrationError, match="precision target"):
        _run(_token_task(), prov, deploy=deploy, captures=tmp_path / "caps", eps=0.16)
    assert sum(1 for n in deploy.deployed if n.startswith("vllm-")) == MIN_SERVINGS


# ---- one serving's drive ------------------------------------------------------

def _drive_one(task, prov, tmp_path, *, member, deploy=None):
    """Drive serving 0 through ``drive_serving`` itself, returning its result and the
    store it wrote to."""
    from hls.calibrate import Serving, drive_serving, flip_measure, member_engines

    store = CaptureStore(tmp_path / "caps", workload_fingerprint(task))
    deploy = deploy or _make_deploy(_FakePinned(0.10), member=member)
    result = asyncio.run(drive_serving(
        Serving("vllm", 0, serving_seed_of(task.seed, 0)), task=task,
        sessions=load_sessions(task, prov, Split.TEST), sessions_per_draw=None,
        request_timeout_sec=REQUEST_TIMEOUT_SEC, store=store, measure=flip_measure(), deploy=deploy,
        builders=member_engines(task)))
    return result, store


def test_a_failed_serving_leaves_no_capture(prov, tmp_path):
    """A serving with a faulted request is refused whole and banks no capture, which is
    what leaves it to be driven again under its own index: dropping the faulted request
    would measure a smaller and easier workload than the one offered."""
    result, store = _drive_one(_token_task(), prov, tmp_path,
                               member=_FakeMember(fault_on="all"))
    assert isinstance(result.error, CalibrationError) and not result.captured and result.failed
    assert str(result.error).startswith(
        f"pool member 'vllm' faulted 1 of 4 requests on serving 0 (first: {_ROW_IDS[0]} -- transport: boom)")
    assert not store.capture_path("vllm", 0).exists()


def _flaky_release(deploy, *, on=lambda name: True):
    """Wrap a deploy seam so releasing an engine raises: a teardown that fails on the way
    out, after the engine has done everything it was asked for."""
    async def deploying(name, engine):
        proxy, release = await deploy(name, engine)

        async def bad_release():
            await release()
            if on(name):
                raise RuntimeError(f"teardown of {name} failed")
        return proxy, bad_release
    deploying.deployed = deploy.deployed
    deploying.live = deploy.live
    return deploying


def test_a_capture_that_outlived_a_failed_release_is_not_a_failed_serving(prov, tmp_path):
    """The capture is written before the engine is released, so a release that fails
    afterwards leaves a serving that is in the store and ready to score: it is not driven
    again."""
    result, store = _drive_one(
        _token_task(), prov, tmp_path, member=_FakeMember(),
        deploy=_flaky_release(_make_deploy(_FakePinned(0.10), member=_FakeMember())))

    assert store.capture_path("vllm", 0).exists()
    assert result.captured and result.error is not None
    assert not result.failed


def test_a_campaign_whose_member_fails_to_release_fails_that_serving(prov, tmp_path):
    """The capture is in the store before its member is released, but a capture role's
    outputs stand only once its member let go of its GPU: the release that fails fails the
    serving, which another attempt of its stage drives again, and the campaign goes no
    further."""
    member = _FakeMember()
    deploy = _flaky_release(_make_deploy(_FakePinned(0.10), member=member),
                            on=lambda name: name.startswith("vllm-"))
    with pytest.raises(ServingFailed, match="serving 0 of pool member 'vllm' captured, but its engine did "
                                            "not release: teardown of vllm-0 failed"):
        _run(_token_task(), prov, deploy=deploy, captures=tmp_path / "caps")
    assert len(member.drives) == 1


def test_p_is_per_engine_at_one_pool_wide_confidence_factor():
    """(P) is evaluated per engine, but at the SAME confidence factor for all of them:
    one factor, taken at the pool's smallest serving count, reproduces every engine's
    check from the artifact, and an engine that served more is judged slightly
    conservatively rather than more leniently."""
    from hls.calibrate import Campaign, _c_m, _rate_side, _sizing
    from hls.verify.calibration import StopReason, precision_check

    campaign = Campaign()
    # "steady" reproduces to well under eps; "jumpy" does not.
    for i in range(8):
        for engine, flips in (("steady", 20), ("jumpy", 20 + 12 * (i % 2))):
            campaign.add(score_draw(engine, i, i, _MEMBER_VERSION, [_tf_item(10_000)],
                                    [[0.1] * flips + [0.0] * (10_000 - flips)]))

    rate_side = _rate_side(campaign.stats)
    sizing = _sizing(rate_side, StopReason.TARGET_REACHED)
    assert sizing.c_m == _c_m(8)
    checks = {e: precision_check(r, sizing.c_m, 0.05) for e, r in rate_side.items()}
    assert checks["steady"].passed and not checks["jumpy"].passed


# ---- the recurrence census --------------------------------------------------

def test_recurrence_census_is_clean_when_severe_events_move_around(prov, tmp_path, caplog):
    """What licenses pooling servings as one Poisson process: severe flips follow the
    serving numerics, not the prompt. A clean census warns of nothing."""
    import logging

    with caplog.at_level(logging.WARNING, logger="hls.calibrate"):
        _run(_token_task(), prov, deploy=_make_deploy(_FakePinned(0.10)), captures=tmp_path / "caps")
    assert not [r for r in caplog.records if "recur across servings" in r.message]


def test_recurrence_census_flags_a_site_that_fires_every_serving(prov, tmp_path, caplog):
    """The same prompt producing a severe flip serving after serving is flagged by the
    recurrence census. Advisory, never a refusal: the ceilings stand as measured."""
    import logging

    task = _token_task()
    pinned = _FakePinned(2.5, site_request_id=_ROW_IDS[0])   # only the first row flips
    with caplog.at_level(logging.WARNING, logger="hls.calibrate"):
        cal = _run(task, prov, deploy=_make_deploy(pinned),
                   captures=tmp_path / "caps")
    [reason] = [r.message for r in caplog.records if "recur across servings" in r.message]
    assert reason.startswith("pool engine 'vllm': ")
    # Named by event key, at the most severe rung of the pool's grid it reached. The
    # grid keeps a rung only where the clean count drops, and this pool's only severe
    # flips are this one site's, so its 2.5-nat gap creates no rung above the floor.
    assert f"'t0:{_ROW_IDS[0]}'" in reason and "0.5 nats" in reason

    # It is reported when the calibration is measured; the file carries none of it.
    written = json.loads(write_calibration(cal, tmp_path).read_text())
    assert "recurrence_warnings" not in written
    assert read_calibration(tmp_path / "calibration.json", task)[0] == cal


# ---- the refusal paths ----------------------------------------------------------

def test_a_campaign_refuses_wrong_silicon(prov, tmp_path):
    """The measured device must BE the task's declared accelerator: recording the
    declared kind unverified would leave read_calibration comparing a declaration
    against itself. Refusal fires before the preflight or any serving burns GPU time."""
    pinned = _FakePinned(0.10, stack={**_STACK, "device_name": "NVIDIA H100"})
    deploy = _make_deploy(pinned)
    with pytest.raises(CalibrationError, match="declared accelerator"):
        _run(_token_task(), prov, deploy=deploy, captures=tmp_path / "caps")
    assert deploy.deployed == ["pinned"]     # refused before the preflight, let alone
                                             # before any member came up


def test_a_campaign_refuses_failed_preflight(prov, tmp_path):
    """A failed preflight (fp32 anchor disagrees with the pinned bf16 reference) is
    refused BEFORE any serving, so a broken instrument never costs a serving's GPU
    time."""
    class _BadPreflight(_FakePinned):
        async def teacher_force(self, items):
            self.tf_calls += 1
            return [[PREFLIGHT_MAX_GAP + 0.5] for _ in items]

    deploy = _make_deploy(_BadPreflight(0.10))
    with pytest.raises(CalibrationError, match="preflight"):
        _run(_token_task(), prov, deploy=deploy, captures=tmp_path / "caps")
    assert deploy.deployed == ["pinned", "fp32_anchor"]   # no member ever came up


def test_a_campaign_refuses_degenerate_ceiling(prov, tmp_path):
    """Every engine agrees perfectly (q_max == 0): the degenerate ceiling would fail
    any trace with a single mismatch, so it is refused rather than shipped."""
    with pytest.raises(CalibrationError, match="degenerate"):
        _run(_token_task(), prov, deploy=_make_deploy(_FakePinned(0.0)), captures=tmp_path / "caps")


def test_member_reply_without_tokens_is_a_hard_error(prov, tmp_path):
    """Calibration cannot score a continuation it cannot read, and a silent skip would
    quietly shrink the measurement: the serving captures nothing."""
    with pytest.raises(ServingFailed, match="token ids"):
        _run(_token_task(), prov, captures=tmp_path / "caps",
             deploy=_make_deploy(_FakePinned(0.10), member=_NoTokenMember()))


def test_preflight_anchor_reply_without_tokens_is_a_hard_error(prov, tmp_path):
    with pytest.raises(CalibrationError, match="fp32 preflight anchor"):
        _run(_token_task(), prov,
             deploy=_make_deploy(_FakePinned(0.10), anchor=_FakeAnchorNoTokens()),
             captures=tmp_path / "caps")


def test_a_campaign_write_read_round_trip(prov, tmp_path):
    task = _token_task()
    cal = _run(task, prov, deploy=_make_deploy(_FakePinned(0.10)), captures=tmp_path / "caps")
    path = write_calibration(cal, tmp_path)
    assert path == tmp_path / "calibration.json"
    loaded = read_calibration(tmp_path / "calibration.json", task)[0]   # re-validates fingerprint/gpu_kind/preflight
    assert loaded == cal
    assert loaded.workload_fingerprint == workload_fingerprint(task)


def test_sessions_per_draw_rides_to_the_member_without_moving_the_fingerprint(prov, tmp_path):
    """How much traffic one serving drove is a property of the MEASUREMENT, not of the
    workload, so it is recorded and forwarded to the member but must never invalidate
    the calibration of a task that never changed."""
    task = _token_task()
    member = _FakeMember()
    cal = _run(task, prov, deploy=_make_deploy(_FakePinned(0.10), member=member),
               captures=tmp_path / "caps", sessions_per_draw=2)
    assert cal.draw.max_sessions == 2
    assert all(max_sessions == 2 for _, _, max_sessions, _ in member.drives)
    assert cal.workload_fingerprint == workload_fingerprint(task)


def test_every_serving_gets_the_same_sessions_on_its_own_clock(prov, tmp_path):
    """The servings differ in their CLOCK, not in their workload: the same materialized
    sessions go to the member every time, and only the seed -- arrivals, think-times,
    ramp-down membership -- changes between them."""
    task = _token_task()
    member = _FakeMember()
    _run(task, prov, deploy=_make_deploy(_FakePinned(0.10), member=member),
         captures=tmp_path / "caps")
    expected = load_sessions(task, prov, Split.TEST)
    assert all(sessions == expected for sessions, _, _, _ in member.drives)
    seeds = [seed for _, seed, _, _ in member.drives]
    assert seeds == [serving_seed_of(task.seed, i) for i in range(len(seeds))]
    assert len(set(seeds)) == len(seeds)


def test_output_cap_task_still_calibrates(prov, tmp_path):
    """A capped workload is nothing special to the procedure -- the cap only decides
    how long the replies it teacher-forces may be."""
    task = _token_task(workload=_workload(cap=ShortOutputCap(cap_tokens=7)))
    cal = _run(task, prov, deploy=_make_deploy(_FakePinned(0.10), member=_FakeMember(tokens=5)),
               captures=tmp_path / "caps", mu_per_mtok=_mu_for_target(100))
    assert cal.q_max > 0


# ---- replies at the output cap --------------------------------------------

def test_replies_at_the_cap_are_measured_into_both_rules_and_counted(prov, tmp_path):
    """A reply that reached the cap is measured like any other -- its tokens are in
    every serving's length and its flips in the rate and the curve -- and the artifact
    records how much of the pool's text did so. Nothing is subtracted anywhere."""
    member = _FakeMember(replies=4, tokens=2500, cap_replies=frozenset({0}))
    task = _token_task(workload=_workload(cap=ShortOutputCap(cap_tokens=3000)))
    # 20 flips of 0.6 nats per reply: level-0.5 events, and a rate the (P) check can
    # reproduce serving after serving.
    cal = _run(task, prov, deploy=_make_deploy(_FakePinned(0.6), member=member),
               captures=tmp_path / "caps")
    draws = cal.rate_side["vllm"].draws
    assert cal.rate_side["vllm"].tokens == [3 * 2500 + 3000] * draws
    assert cal.rate_side["vllm"].mean == pytest.approx(80 / 10_500)
    curve = cal.curve_side.engines["vllm"]
    assert curve.exposure == 10_500 * draws
    assert curve.levels[0].events == 80 * draws
    assert cal.capped_text.events == draws and cal.capped_text.tokens == 3000 * draws


#: What the campaign of ``_long_reply_campaign`` measures with its long reply exactly at the cap
#: (``measured_json``), recorded from a campaign that made no test of a reply's length against
#: its cap. Never regenerated: a campaign whose replies all keep their cap must go on
#: assembling these numbers.
AT_CAP = Path(__file__).parent / "data" / "calibrate_reply_at_cap.json"
#: The campaign's output cap, and the length of every reply but the long one.
_CAP, _SHORT = 2600, 2500
#: The serving and the reply of it that runs long.
_LONG_SERVING, _LONG_REPLY = 1, 2
#: The fields of a calibration that its servings' measurement decides, not its task's identity.
_MEASURED = ("q_max", "rate_side", "curve_side", "sizing", "capped_text", "draw", "tolerances")


class _OneLongReplyMember(_FakeMember):
    """A member whose replies are ``_SHORT`` tokens, except reply ``_LONG_REPLY`` of serving
    ``_LONG_SERVING``, which runs to ``length`` tokens whatever cap its request carried."""

    def __init__(self, length: int):
        super().__init__(replies=4, tokens=_SHORT)
        self.length = length

    async def serve_traffic(self, sessions, **kw):
        serving = len(self.drives)
        trace = await super().serve_traffic(sessions, **kw)
        if serving == _LONG_SERVING:
            ev = trace.events[_LONG_REPLY]
            ev.response = ArtifactResponse(output=Envelope(request_id=ev.request_id, contents=[
                TextPayload(text="x", tokens=list(range(self.length)), finish_reason="length")]))
        return trace


class _RecordingPinned(_FakePinned):
    """The fake pinned reference, which records each call's items as their request ids and
    served lengths: the preflight's probes first, then one call a serving."""

    def __init__(self, gap: float):
        super().__init__(gap)
        self.asked: list[list[tuple[str, int]]] = []

    async def teacher_force(self, items):
        self.asked.append([(item.request_id, len(item.first_of("text").tokens)) for item in items])
        return await super().teacher_force(items)


def _long_reply_campaign(prov, tmp_path, *, length: int, pinned: _FakePinned | None = None):
    """The calibration of a campaign under a ``_CAP``-token cap whose one long reply serves
    *length* tokens, each reply carrying 20 flips of 0.6 nats; its captures are kept under
    ``tmp_path / "caps"``."""
    task = _token_task(workload=_workload(cap=ShortOutputCap(cap_tokens=_CAP)))
    deploy = _make_deploy(pinned or _FakePinned(0.6), member=_OneLongReplyMember(length))
    return _run(task, prov, deploy=deploy, captures=tmp_path / "caps")


def measured_json(cal) -> dict:
    """The record ``AT_CAP`` holds: the measured fields of *cal*, as JSON."""
    dumped = json.loads(cal.model_dump_json())
    return {name: dumped[name] for name in _MEASURED}


def test_a_campaign_with_a_reply_at_its_cap_assembles_the_recorded_numbers(prov, tmp_path):
    """A reply exactly at its cap is a reply like any other: it is teacher-forced, its tokens
    and flips are in the rate and the curve, and it is tallied among the capped text."""
    pinned = _RecordingPinned(0.6)
    cal = _long_reply_campaign(prov, tmp_path, length=_CAP, pinned=pinned)
    assert measured_json(cal) == json.loads(AT_CAP.read_text())
    # The long reply was teacher-forced whole, and is the campaign's one capped reply.
    assert pinned.asked[1 + _LONG_SERVING][_LONG_REPLY] == (_ROW_IDS[_LONG_REPLY], _CAP)
    assert (cal.capped_text.events, cal.capped_text.tokens) == (1, _CAP)


#: The refusal of the long-reply campaign whose long reply is one token past its cap.
_PAST_CAP = (f"pool engine 'vllm' served {_CAP + 1} tokens for request {_ROW_IDS[_LONG_REPLY]!r} of tenant 't0' on "
             f"serving {_LONG_SERVING}, past the {_CAP}-token cap the harness stamped on it; the member did not "
             "honor the cap, so the serving holds text no graded trace is measured on. Fix the stack and re-run.")


def test_a_campaign_with_a_reply_past_its_cap_is_refused_and_writes_no_candidate(prov, tmp_path):
    """One reply one token past its cap refuses the campaign, naming the reply: the verifier
    fails a trace on such a reply, so ceilings that counted it would be measured on text no
    graded trace is read against. No calibration is assembled, so none is written."""
    task = _token_task(workload=_workload(cap=ShortOutputCap(cap_tokens=_CAP)))
    with pytest.raises(CalibrationError) as refused:
        write_calibration(_long_reply_campaign(prov, tmp_path, length=_CAP + 1), tmp_path)
    assert str(refused.value) == _PAST_CAP
    assert not (tmp_path / "calibration.json").exists()
    # The serving that held the reply left no scored record; the one before it was scored.
    store = CaptureStore(tmp_path / "caps", workload_fingerprint(task))
    assert store.replies_path("vllm", 0).exists()
    assert store.capture_path("vllm", _LONG_SERVING).exists()
    assert not store.replies_path("vllm", _LONG_SERVING).exists()


def test_a_serving_with_a_reply_past_its_cap_never_reaches_the_reference(prov, tmp_path):
    """The serving is refused before the reference is asked to teacher-force it: a member
    that ignores the cap does not choose how much the reference must hold."""
    pinned = _RecordingPinned(0.6)
    with pytest.raises(CalibrationError, match="past the 2600-token cap"):
        _long_reply_campaign(prov, tmp_path, length=_CAP + 1, pinned=pinned)
    # The preflight's probes and the serving before it, and nothing of the refused serving.
    probes, first = pinned.asked
    assert all(rid.startswith("preflight-") for rid, _ in probes)
    assert first == [(rid, _SHORT) for rid in _ROW_IDS]
    assert not any(n > _CAP for call in pinned.asked for _, n in call)


def test_score_draw_refuses_a_reply_past_its_cap():
    """The measurement itself refuses, whoever hands it the gaps, by the verifier's own test
    of a served length against a cap (``hls.verify.flips.past_cap``); a reply exactly at its
    cap passes it."""
    items = [_tf_item(2), _tf_item(4, request_id="hot.0", cap=3)]
    with pytest.raises(CalibrationError) as refused:
        score_draw("sglang", 5, 0, _MEMBER_VERSION, items, [[0.0, 0.9], [0.0, 0.6, 0.0, 0.0]])
    assert str(refused.value) == (
        "pool engine 'sglang' served 4 tokens for request 'hot.0' of tenant 't0' on serving 5, past the 3-token cap "
        "the harness stamped on it; the member did not honor the cap, so the serving holds text no graded trace is "
        "measured on. Fix the stack and re-run.")
    assert refuse_past_cap("sglang", 5, [_tf_item(2), _tf_item(3, request_id="hot.0", cap=3)]) is None


def test_the_campaign_and_the_verifier_refuse_a_reply_by_one_test(monkeypatch):
    """The campaign asks ``hls.verify.flips.past_cap``, the function the verifier asks, and
    refuses exactly when it says so: no second comparison of a length with a cap decides."""
    import hls.calibrate as calibrate
    import hls.verify.verifiers as verifiers
    from hls.verify import flips

    assert calibrate.past_cap is flips.past_cap is verifiers.past_cap
    asked = []
    monkeypatch.setattr(calibrate, "past_cap", lambda n, cap: asked.append((n, cap)) or n == 2)
    with pytest.raises(CalibrationError, match="served 2 tokens for request 'r.0'"):
        refuse_past_cap("vllm", 0, [_tf_item(2, cap=3), _tf_item(4, request_id="hot.0", cap=3)])
    assert asked == [(2, 3)]


# ---- capped_rows: the removal list a task derives from a campaign ---------

def test_capped_rows_names_whole_sessions_the_engine_ran_to_the_cap(tmp_path):
    """Per tenant, every session with a turn at the cap in ANY stored serving, as sorted
    dataset item ids, with the counts that price the removal: a capped second turn takes
    its whole conversation, a session capped in one serving of many is still listed, and
    a reply that merely reports "length" without reaching the cap is not."""
    store = CaptureStore(tmp_path, "fp")
    rows = RowIndex(rows_of_tenant={"chat": {"a", "b"}, "api": {"c"}},
                    cap_of_tenant={"chat": 8, "api": 8})

    def item(rid, n, *, tenant, finish="stop"):
        return Envelope(request_id=rid,
                        metadata={"messages": [], "tenant_id": tenant, "max_tokens": 8},
                        contents=[TextPayload(text="x", tokens=[1] * n, finish_reason=finish)])

    store.write_capture("vllm", 0, 1, _MEMBER_VERSION, [
        item("a.0", 3, tenant="chat"), item("a.1", 8, tenant="chat"),      # a capped at turn 1
        item("b.0", 4, tenant="chat"), item("c.0", 5, tenant="api", finish="length")])
    store.write_capture("vllm", 1, 2, _MEMBER_VERSION, [
        item("a.0", 3, tenant="chat"), item("b.0", 4, tenant="chat"),
        item("b.1", 8, tenant="chat"), item("c.0", 8, tenant="api")])     # b and c cap here
    store.write_capture("sglang", 0, 1, _MEMBER_VERSION, [item("b.0", 8, tenant="chat")])   # another engine

    out = capped_rows(store, "vllm", rows)
    assert out["chat"] == CappedRows(rows=("a", "b"), capped_turns=2, served_rows=2,
                                     served_turns=4)
    assert out["api"] == CappedRows(rows=("c",), capped_turns=1, served_rows=1,
                                    served_turns=1)
    # Another engine's captures play no part; a pool member that never capped lists
    # nothing but is still reported, with its served counts.
    assert capped_rows(store, "sglang", rows) == {
        "chat": CappedRows(rows=("b",), capped_turns=1, served_rows=1, served_turns=1)}


def test_capped_rows_refuses_a_reply_over_its_cap(tmp_path):
    """A reply longer than the cap the harness stamped means the member did not honor
    the cap, so its captures do not describe the workload; refuse rather than list."""
    store = CaptureStore(tmp_path, "fp")
    rows = RowIndex(rows_of_tenant={"t0": {"a"}}, cap_of_tenant={"t0": 8})
    store.write_capture("vllm", 0, 1, _MEMBER_VERSION, [Envelope(
        request_id="a.0", metadata={"messages": [], "tenant_id": "t0", "max_tokens": 8},
        contents=[TextPayload(text="x", tokens=[1] * 9)])])
    with pytest.raises(CalibrationError, match="did not honor the cap"):
        capped_rows(store, "vllm", rows)


def test_capped_rows_keys_a_driven_capture_like_the_tasks_rows(prov, tmp_path):
    """A capture written from the drive loop's own events comes back keyed by the
    task's dataset row ids, with the cap read off the stamp the loop put on it."""
    task = _token_task(workload=_workload(cap=ShortOutputCap(cap_tokens=40)))
    ev = _driven_event(task, prov, tokens=40)
    store = CaptureStore(tmp_path, "fp")
    store.write_capture("vllm", 0, 5, _MEMBER_VERSION,
                        capture_items(Trace(events=[ev]), engine="vllm", draw=0))
    rows = RowIndex.from_sessions(load_sessions(task, prov, Split.TEST))
    out = capped_rows(store, "vllm", rows)
    assert out == {"t0": CappedRows(rows=("0",), capped_turns=1,
                                    served_rows=1, served_turns=1)}


# ---- the removal list -------------------------------------------------------

def test_sessions_are_the_whole_split_and_the_drive_plan_drops_the_listed_rows(prov):
    """A serving loads the whole split, as a graded run does, and the shared drive plan
    drops the listed rows before it draws the schedule, so a serving offers exactly the
    sessions a graded run offers."""
    from hls.spec.drivers import plan_sessions, remove_listed_rows

    task = _token_task(data_source=_data_source().model_copy(
        update={"removed_rows": ("1", "3")}))
    (_t, items), = load_sessions(task, prov, Split.TEST)
    assert [i.request_id for i in items] == [f"{i}" for i in range(4)]
    (tenant, kept, starts), = plan_sessions(
        remove_listed_rows(load_sessions(task, prov, Split.TEST)), task.seed)[0]
    assert [i.request_id for i in kept] == ["0", "2"] and len(starts) == 2


# ---- a serving that measured nothing ------------------------------------------

def test_a_serving_with_no_tokens_refuses_instead_of_looping(prov, tmp_path):
    """A member whose replies carry no tokens at all yields nothing the exposure
    counter can advance on, so the campaign has to refuse -- a counter stop against a
    target it can never reach is an infinite loop."""
    with pytest.raises(CalibrationError, match="no tokens at all"):
        _run(_token_task(), prov, captures=tmp_path / "caps",
             deploy=_make_deploy(_FakePinned(0.10),
                                 member=_FakeMember(tokens=0)))


# ---- preflight probes -------------------------------------------------------

def test_preflight_probes_committed():
    assert len(PREFLIGHT_PROBES) == 8
    assert all(isinstance(p, str) and p.strip() for p in PREFLIGHT_PROBES)
    # the long-context probe is present (a ~2k-token passage)
    assert max(len(p) for p in PREFLIGHT_PROBES) > 4000
    assert PREFLIGHT_MAX_FLIP_RATE == 0.02 and PREFLIGHT_MAX_GAP == 1.0


# ---- advisory member warnings ----------------------------------------------

def test_untrusted_engine_is_warned_about_but_kept(prov, tmp_path, caplog):
    """An engine whose flip rate leaves the benign band must NOT refuse the run: the
    calibration is emitted, the engine stays in the pool and in the ceilings, and the
    breach is warned about when the calibration is measured, naming the engine for
    review. The file does not carry the warning."""
    import logging

    # 60 flips in each 2500-token reply is a 2.4% rate, outside the 2% band.
    task = _token_task()
    with caplog.at_level(logging.WARNING, logger="hls.calibrate"):
        cal = _run(task, prov,
                   deploy=_make_deploy(_FakePinned(0.10, flips_for=lambda draw: 60)),
                   captures=tmp_path / "caps")

    assert cal.rate_side["vllm"].mean > MEMBER_MAX_TRUSTED_FLIP_RATE
    warnings = [r.message for r in caplog.records if "trusted band" in r.message]
    assert len(warnings) == 1 and warnings[0].startswith("pool engine 'vllm': ")

    written = json.loads(write_calibration(cal, tmp_path).read_text())
    assert "trust_warnings" not in written
    assert read_calibration(tmp_path / "calibration.json", task)[0] == cal


def _trust_lines(caplog) -> list[str]:
    """The stack-review census's warnings among the records *caplog* captured."""
    return [r.message for r in caplog.records
            if "trusted band" in r.message or "sets q_max alone" in r.message]


def test_a_large_single_gap_is_not_a_trust_warning(prov, tmp_path, caplog):
    """There is no bound on gap size, because a clean pool produces rare multi-nat
    events of its own: what the gate bounds is how many large-gap flips a trace
    carries. A member whose worst gap is nearly 2 nats at a healthy flip rate is
    recorded and not flagged."""
    import logging

    with caplog.at_level(logging.WARNING, logger="hls.calibrate"):
        cal = _run(_token_task(), prov, deploy=_make_deploy(_FakePinned(1.875)),
                   captures=tmp_path / "caps")
    assert cal.rate_side["vllm"].max_gap == 1.875
    assert _trust_lines(caplog) == []


def test_clean_pool_is_warned_of_nothing(prov, tmp_path, caplog):
    """A pool inside the bound is checked and warned of nothing, and its file carries no
    advisory text."""
    import logging

    task = _token_task()
    with caplog.at_level(logging.WARNING, logger="hls.calibrate"):
        cal = _run(task, prov, deploy=_make_deploy(_FakePinned(0.10)), captures=tmp_path / "caps")
    assert _trust_lines(caplog) == []
    written = json.loads(write_calibration(cal, tmp_path).read_text())
    assert not {"trust_warnings", "recurrence_warnings"} & set(written)
    assert read_calibration(tmp_path / "calibration.json", task)[0] == cal


def test_ceiling_outlier_is_recorded_beside_the_band_breach(caplog):
    """A single engine that sets q_max alone is flagged even while it sits inside the
    absolute band. Both findings name a member for review, so both land in the same
    map. Warning only, never a refusal."""
    import logging

    from hls.calibrate import _trust_warnings

    with caplog.at_level(logging.WARNING, logger="hls.calibrate"):
        flagged = _trust_warnings({"a": 0.001, "b": 0.010})
    assert list(flagged) == ["b"] and "sets q_max alone" in flagged["b"]
    assert any("sets q_max alone" in r.message and "'b'" in r.message
               for r in caplog.records)

    assert _trust_warnings({"a": 0.009, "b": 0.010}) == {}   # inside the ratio
    assert _trust_warnings({"a": 0.010}) == {}               # nothing to compare against
    # Both findings at once: one entry naming both.
    both = _trust_warnings({"a": 0.001, "b": 0.030})["b"]
    assert both == ("flip_rate=0.03 is outside the trusted band (> 0.02); it sets q_max alone: its "
                    "mean flip rate 0.03 is 30.0x the next-highest engine's 0.001. Review its stack; "
                    "to exclude it, drop it (--engines) and recompute over the remaining engines.")


# ---- the session subsample --------------------------------------------------

class _NItemDataset(Dataset):
    """``n`` single-turn items -- the subsample fixture."""
    name: ClassVar[str] = "many"
    version: ClassVar[str] = "v0"
    n: int = 10
    tag: str = "x"

    def items_at(self, mount_dir, split):
        for i in range(self.n):
            rid = f"{self.tag}-{split.value}-{i}"
            yield DatasetItem(
                request_id=rid,
                turns=(Envelope(request_id=rid, contents=[TextPayload(text=f"q{i}")]),),
                ground_truth_turns=("A",))


def test_session_subsample_keeps_the_mix_and_trims_the_span():
    """A serving of a task whose full drive would overshoot the exposure target several
    times over subsamples WHOLE sessions at the same arrival rate: every tenant keeps
    its share (so the mix is the task's), and the shorter session count is what
    shortens the span."""
    from hls.spec.drivers import _subsample_sessions

    loaded = [("big", list(range(200))), ("small", list(range(20)))]
    trimmed = _subsample_sessions(
        [(type("T", (), {"id": t})(), items) for t, items in loaded], 44)
    counts = {t.id: len(items) for t, items in trimmed}
    assert counts == {"big": 40, "small": 4}          # both scaled by 44/220
    # Prefixes, so each tenant's arrival schedule is simply generated for fewer
    # sessions at the same rate.
    assert [t for t, items in trimmed if items != list(range(len(items)))] == []

    # No bound, or one the population cannot fill, drives everything.
    every = [(type("T", (), {"id": t})(), items) for t, items in loaded]
    assert _subsample_sessions(every, None) is every
    assert _subsample_sessions(every, 10_000) is every


def test_session_subsample_never_drops_a_tenant():
    """A bound small enough to round a small tenant to zero would silently change the
    traffic MIX, which is the one thing the subsample must preserve."""
    from hls.spec.drivers import _subsample_sessions

    loaded = [(type("T", (), {"id": "big"})(), list(range(1000))),
              (type("T", (), {"id": "tiny"})(), list(range(3)))]
    trimmed = _subsample_sessions(loaded, 10)
    assert {t.id: len(items) for t, items in trimmed} == {"big": 10, "tiny": 1}


def test_members_serve_at_the_context_window_the_task_declares():
    """Both members are built at the task's own context window, and report it on every
    reply: an agentic framework budgets its prompts against the window the reply names,
    so a member left at its checkpoint's context would finalize trajectories later than
    a graded run does -- or not at all. A task that declares none leaves both stacks
    where their checkpoint puts them."""
    from hls.calibrate import member_engines
    built = {name: build()
             for name, build in member_engines(_token_task(max_context_len=131072)).items()}
    assert {name: eng.max_model_len for name, eng in built.items()} == {
        "vllm": 131072, "sglang": 131072}
    undeclared = {name: build().max_model_len
                  for name, build in member_engines(_token_task()).items()}
    assert set(undeclared.values()) == {None}


def test_sglang_member_serves_with_the_larger_kv_pool():
    """The SGLang pool member runs at mem_fraction_static 0.92: at 0.85 its KV pool
    caps a request's prompt near 85k tokens on a dense 31B, which an agentic drive
    exceeds late in long trajectories."""
    from hls.calibrate import member_engines
    assert member_engines(_token_task())["sglang"]().mem_fraction_static == 0.92
