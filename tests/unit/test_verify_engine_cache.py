"""Engine cache across the verify phase (cross-tenant dedup).

verify_trace keeps a single engine cache live across ALL tenants/verifiers, keyed by
id(cfg). Before each verifier it reconciles the cache to exactly that verifier's
engine set: evict (release) what it does not need FIRST -- the slot pool is
fail-loud, so the device must be freed before the next acquire -- then deploy what
is missing. Asserted here:

  - A shared oracle Reference object reused by every tenant is deployed ONCE and
    its engine reused (not redeployed per verifier).
  - Distinct oracle objects are NEVER live concurrently: the previous engine is
    released before the next is acquired (evict-before-acquire), so at most one
    distinct engine is up at a time.
  - A verify() exception still releases every live engine (outer finally).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import ClassVar

import pytest

from hls.core.capabilities import Generate, StackReport, TeacherForce
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
from hls.store.posix import PosixStore
from hls.verify.answers import AnswerEntry, AnswerFile, StoreAnswers, answer_key, prompt_digest
from hls.verify.base import ArtifactRollout, Verifier, WiringError, check_wiring
from hls.verify.inprocess import EngineHost
from hls.verify.run import verify_trace


# ---------------------------------------------------------------------------
# Fakes: an engine that logs its lifecycle, a Reference that builds it, and a
# Verifier that owns the Reference. _LOG records ("build"|"acquire"|"release", tag)
# in call order so a test can reconstruct concurrency and ordering: the host acquires
# an engine by loading it and releases it by stopping it.
# ---------------------------------------------------------------------------

_LOG: list[tuple[str, str]] = []


class _CountingEngine(Engine):
    """Carries the one capability ``supports`` is patched to report, which the core
    resolves and the fake verifiers call: the core loads an engine on first use, so a
    verifier that declared one and never reached for it would never cause a load to
    observe."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    def _load(self) -> None:
        _LOG.append(("acquire", self.tag))

    def _unload(self) -> None:
        _LOG.append(("release", self.tag))

    async def teacher_force(self, items):
        return []


class _FakeRef(Reference):
    engine_cls: ClassVar[type] = _CountingEngine
    tag: str = "ora"

    def build_engine(self, task):
        _LOG.append(("build", self.tag))
        return _CountingEngine(self.tag)


class _EngineVerifier(Verifier):
    """Owns one reference engine; passes every event it is handed."""

    reference: _FakeRef
    raises: bool = False

    async def verify(self, engine) -> Verdict:
        await engine[TeacherForce]([])          # reach for the declared engine, as a real
        if self.raises:                         # verifier does, so it is deployed
            raise RuntimeError("boom in verify")
        events = engine[ArtifactRollout]
        return Verdict(passed=True, per_event={ev.request_id: True for ev in events})


class _PooledEngineVerifier(Verifier):
    """A POOLED gate that owns a reference engine.

    Grades every tenant's events as one population and keys all three of its per-event
    maps by ``event_key``, exactly as the shipped pooled gate
    (``TokenFaithfulnessVerifier``) does. Logs a ``("verify", tag)`` entry so a test can
    count how many times it actually ran."""

    pooled: ClassVar[bool] = True
    reference: _FakeRef
    tag: str = "pooled"

    async def verify(self, engine) -> Verdict:
        await engine[TeacherForce]([])
        _LOG.append(("verify", self.tag))
        events = engine[ArtifactRollout]
        return Verdict(
            passed=True,
            per_event={ev.event_key: True for ev in events},
            event_rationale={ev.event_key: f"graded {ev.event_key}" for ev in events},
            event_diagnostics={ev.event_key: {"n_tokens": 7} for ev in events})


class _RolloutVerifier(Verifier):
    """Asks its reference for its own answer to every event's prompt, the one question
    the answer cache serves."""

    reference: _FakeRef
    requires = frozenset({Generate})

    async def verify(self, engine) -> Verdict:
        events = engine[ArtifactRollout]
        await engine[Generate]([ev.input for ev in events])
        return Verdict(passed=True, per_event={ev.request_id: True for ev in events})


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def _wl() -> Workload:
    return Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                    output_length_cap=Uncapped(), slo=NoSLO())


def _contract() -> RequestResponseContract:
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}))


def _ev(req: str, tenant_id: str) -> TraceEvent:
    inp = Envelope(request_id=req, contents=[TextPayload(text="Q?")],
                   metadata={"tenant_id": tenant_id})
    out = Envelope(request_id=req, contents=[TextPayload(text="A")])
    return TraceEvent(request_id=req, input=inp,
                      response=ArtifactResponse(output=out), ground_truth="A",
                      t_send=0.0, t_recv=1.0)


def _task(tenants: list[Tenant]) -> TaskSpec:
    return TaskSpec(
        id="T", model=ModelSpec(base_checkpoint="m", arch="TX"),
        hardware=Hardware(accelerator=AcceleratorSpec(kind="B200", count=1)),
        contract=_contract(), tenants=tenants,
        scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=1.0)))


def _run_verify(task: TaskSpec, trace: Trace):
    async def _go():
        async with EngineHost() as host:
            return await verify_trace(task, trace, eval_dir=None, host=host, answers=None)
    return asyncio.run(_go())


def _answers(tmp_path, writer: str) -> StoreAnswers:
    """A verify's writer on the task's answers area under *tmp_path*."""
    area = tmp_path / "answers"
    area.mkdir(exist_ok=True)
    return StoreAnswers(PosixStore(area), gpu_kind="H200", image="a" * 64, writer=writer)


def _verify_with_answers(task: TaskSpec, trace: Trace, answers: StoreAnswers):
    """The verification core serving the corpus answers of caching references from *answers*,
    as the verify stage's role runs it."""
    async def _go():
        async with EngineHost() as host:
            return await verify_trace(task, trace, eval_dir=None, host=host, answers=answers)
    return asyncio.run(_go())


@pytest.fixture(autouse=True)
def _engine_caps(monkeypatch):
    """capability_providers reads capabilities via hls.verify.base.supports; map the fake
    engine to one real capability so the verifier wires to an engine."""
    _LOG.clear()
    monkeypatch.setattr("hls.verify.base.supports",
                        lambda cls: frozenset({TeacherForce}))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_shared_oracle_deployed_once_across_tenants():
    """One Reference object reused by every tenant deploys ONE engine, reused."""
    shared = _FakeRef(decode=Greedy(), tag="ora")
    tenants = [
        Tenant(id="a", workload=_wl(), verifiers=[_EngineVerifier(reference=shared)]),
        Tenant(id="b", workload=_wl(), verifiers=[_EngineVerifier(reference=shared)]),
    ]
    verdict, _ = _run_verify(_task(tenants), Trace(events=[_ev("a0", "a"), _ev("b0", "b")]))

    builds = [t for k, t in _LOG if k == "build"]
    acquires = [t for k, t in _LOG if k == "acquire"]
    releases = [t for k, t in _LOG if k == "release"]
    assert builds == ["ora"], "shared oracle built exactly once across both tenants"
    assert acquires == ["ora"], "shared oracle acquired exactly once"
    assert releases == ["ora"], "shared oracle released exactly once, at the end"
    assert _LOG[-1] == ("release", "ora"), "release is the final action (held until end)"
    assert verdict.passed is True


def test_shared_oracle_reused_across_verifiers_within_a_tenant():
    """Two verifiers in one tenant sharing a reference reuse the one engine."""
    shared = _FakeRef(decode=Greedy(), tag="ora")
    tenant = Tenant(id="a", workload=_wl(), verifiers=[
        _EngineVerifier(reference=shared), _EngineVerifier(reference=shared)])
    _run_verify(_task([tenant]), Trace(events=[_ev("a0", "a")]))

    assert [k for k, _ in _LOG].count("acquire") == 1
    assert [k for k, _ in _LOG].count("release") == 1


def test_distinct_oracles_never_live_concurrently():
    """Distinct reference objects are released-before-acquired: only one is ever up."""
    ref_a = _FakeRef(decode=Greedy(), tag="A")
    ref_b = _FakeRef(decode=Greedy(), tag="B")
    tenants = [
        Tenant(id="a", workload=_wl(), verifiers=[_EngineVerifier(reference=ref_a)]),
        Tenant(id="b", workload=_wl(), verifiers=[_EngineVerifier(reference=ref_b)]),
    ]
    _run_verify(_task(tenants), Trace(events=[_ev("a0", "a"), _ev("b0", "b")]))

    # Concurrency never exceeds 1: walking the log, live = acquires - releases.
    live = 0
    peak = 0
    for kind, _tag in _LOG:
        if kind == "acquire":
            live += 1
        elif kind == "release":
            live -= 1
        peak = max(peak, live)
    assert peak == 1, "two distinct oracles must never be live at the same time"

    # Evict-before-acquire: A is released before B is acquired (frees the device first).
    rel_a = _LOG.index(("release", "A"))
    acq_b = _LOG.index(("acquire", "B"))
    assert rel_a < acq_b
    assert [t for k, t in _LOG if k == "acquire"] == ["A", "B"]
    assert [t for k, t in _LOG if k == "release"] == ["A", "B"]


def test_pooled_verifier_does_no_engine_work_on_a_cache_hit():
    """A pooled gate declared on every tenant is graded once, on the first tenant, and
    later tenants take the cached verdict WITHOUT touching the engine cache.

    The interleaved per-tenant verifier is what makes that visible: it evicts the
    pooled reference to bring its own up, so a later tenant that reconciled the cache
    before consulting the pooled verdict would evict Q, redeploy P, then throw the
    result away -- a redeploy per tenant."""
    gate = _PooledEngineVerifier(reference=_FakeRef(decode=Greedy(), tag="P"))
    plain_ref = _FakeRef(decode=Greedy(), tag="Q")
    tenants = [
        Tenant(id="a", workload=_wl(),
               verifiers=[gate, _EngineVerifier(reference=plain_ref)]),
        Tenant(id="b", workload=_wl(),
               verifiers=[gate, _EngineVerifier(reference=plain_ref)]),
    ]
    verdict, _pvr = _run_verify(_task(tenants), Trace(events=[_ev("a0", "a"), _ev("b0", "b")]))

    assert verdict.passed is True
    # Graded once over the whole trace, not once per tenant.
    assert [t for k, t in _LOG if k == "verify"] == ["pooled"]
    # Two deploys total: P for the pooled gate, Q for the per-tenant one. Tenant b
    # reuses Q (still live) because its pooled verifier never reconciled the cache.
    assert [t for k, t in _LOG if k == "build"] == ["P", "Q"]
    assert [t for k, t in _LOG if k == "acquire"] == ["P", "Q"]


def test_two_differently_configured_pooled_instances_are_refused():
    """A pooled gate returns ONE verdict for the whole run, so two instances of the
    same pooled class carrying DIFFERENT settings cannot both be honoured. Refused up
    front -- before any engine is built -- rather than resolved by iteration order."""
    ref = _FakeRef(decode=Greedy(), tag="P")
    tenants = [
        Tenant(id="a", workload=_wl(),
               verifiers=[_PooledEngineVerifier(reference=ref, tag="strict")]),
        Tenant(id="b", workload=_wl(),
               verifiers=[_PooledEngineVerifier(reference=ref, tag="loose")]),
    ]
    with pytest.raises(WiringError, match="POOLED"):
        _run_verify(_task(tenants), Trace(events=[_ev("a0", "a"), _ev("b0", "b")]))
    assert _LOG == [], "refused before any engine was built or graded"


def test_equal_pooled_copies_are_one_gate_not_a_wiring_error():
    """Equal copies of one pooled verifier are the SUPPORTED wiring, not a conflict.

    A task declares a single shared instance across its tenants, but calibration
    injection rebuilds each tenant's verifier (a model_copy carrying the loaded
    calibration), so by the time the runner sees them they are distinct objects with
    identical settings. Comparing by object identity would refuse the shipped
    multitenant task outright; comparing by configuration accepts it and still grades
    the pool once."""
    ref = _FakeRef(decode=Greedy(), tag="P")
    tenants = [
        Tenant(id="a", workload=_wl(),
               verifiers=[_PooledEngineVerifier(reference=ref, tag="same")]),
        Tenant(id="b", workload=_wl(),
               verifiers=[_PooledEngineVerifier(reference=ref, tag="same")]),
    ]
    _run_verify(_task(tenants), Trace(events=[_ev("a0", "a"), _ev("b0", "b")]))
    assert [e for e in _LOG if e[0] == "verify"] == [("verify", "same")], (
        "the pool must be graded exactly once across both tenants")


def test_pooled_cells_are_read_from_its_event_key_keyed_maps():
    """A pooled verdict's per-event maps key on event_key, so the per-verifier matrix
    must read them by event_key too. Reading them by request_id misses every entry and
    paints a passing run's cells red with empty diagnostics.

    Both tenants use request_id "r0" here (two tenants over one dataset path), which is
    exactly the case a request_id lookup cannot resolve."""
    gate = _PooledEngineVerifier(reference=_FakeRef(decode=Greedy(), tag="P"))
    tenants = [Tenant(id="a", workload=_wl(), verifiers=[gate]),
               Tenant(id="b", workload=_wl(), verifiers=[gate])]
    verdict, pvr = _run_verify(
        _task(tenants), Trace(events=[_ev("r0", "a"), _ev("r0", "b")]))

    assert verdict.passed is True
    assert dict(verdict.per_event) == {"a:r0": True, "b:r0": True}
    # One row per (tenant, verifier) pair, in tenant order; columns follow trace.events.
    a_row, b_row = pvr
    assert a_row[0]["passed"] is True
    assert a_row[0]["rationale"] == "graded a:r0"
    assert a_row[0]["diagnostics"] == {"n_tokens": 7}
    assert b_row[1]["passed"] is True
    assert b_row[1]["rationale"] == "graded b:r0"
    # Each row still reports only its own tenant's column; together they cover both.
    assert a_row[1]["passed"] is None and b_row[0]["passed"] is None


def test_engine_released_when_verify_raises():
    """A verify() exception still tears the live engine down (outer finally)."""
    ref = _FakeRef(decode=Greedy(), tag="ora")
    tenant = Tenant(id="a", workload=_wl(),
                    verifiers=[_EngineVerifier(reference=ref, raises=True)])
    with pytest.raises(RuntimeError, match="boom in verify"):
        _run_verify(_task([tenant]), Trace(events=[_ev("a0", "a")]))

    assert ("acquire", "ora") in _LOG
    assert ("release", "ora") in _LOG, "engine released despite the verify() exception"


def test_a_fully_cached_reference_is_never_deployed(tmp_path, monkeypatch):
    """The point of deploying on first use.

    An engine whose every answer is already on disk has nothing to compute, so building and
    acquiring it would hold a device for a call it never makes, and on a
    verifier with two references would start the first one's scheduler clock while the
    second queued for a device. With the answers area serving every prompt, the wrapper never
    calls through and nothing is deployed at all."""
    monkeypatch.setattr("hls.verify.base.supports", lambda cls: frozenset({Generate}))
    ref = _FakeRef(decode=Greedy(), tag="ora", cache=True)
    task = _task([Tenant(id="a", workload=_wl(),
                         verifiers=[_RolloutVerifier(reference=ref)])])
    prompt = _ev("a0", "a").input
    answer = Envelope(request_id="a0", contents=[TextPayload(text="a")])
    key = tmp_path / "answers" / answer_key(ref.fingerprint(task), "H200", "a" * 64)
    key.mkdir(parents=True)
    (key / "g0+verify+1.000001.json").write_text(AnswerFile(
        fingerprint=ref.fingerprint(task), gpu_kind="H200", image="a" * 64, writer="g0+verify+1", written_at=0.0,
        answers=[AnswerEntry(request_id="a0", digest=prompt_digest(prompt),
                             answer=answer.model_dump(mode="json"))]).model_dump_json())

    _LOG.clear()
    _verify_with_answers(task, Trace(events=[_ev("a0", "a")]), _answers(tmp_path, "g1+verify+1"))

    assert _LOG == [], f"a fully cached reference was deployed anyway: {_LOG}"


# ---------------------------------------------------------------------------
# The answer cache serves a reference's own answers to corpus rows, nothing else
# ---------------------------------------------------------------------------

#: What the last ``_AsksEverything`` verify was handed back, per capability.
_SEEN: dict = {}


class _AnsweringEngine(Engine):
    """An engine that answers a row, teacher-forces a reply and reports its stack,
    logging every call that reaches it."""

    @property
    def resources(self) -> ResourceRequest:
        return ResourceRequest(gpu_count=1)

    async def generate(self, prompts):
        _LOG.append(("call", "generate"))
        return [Envelope(request_id=p.request_id, contents=[TextPayload(text="gen")])
                for p in prompts]

    async def teacher_force(self, items):
        _LOG.append(("call", "teacher_force"))
        return [[0.0, 0.1] for _ in items]

    async def stack_report(self):
        _LOG.append(("call", "stack_report"))
        return {"torch": "x"}


class _AnsweringRef(Reference):
    engine_cls: ClassVar[type] = _AnsweringEngine

    def build_engine(self, task):
        return _AnsweringEngine()


class _AsksEverything(Verifier):
    """Declares only the rollout, and also teacher-forces each reply and reads the
    stack, the way the text report gates do."""

    reference: _AnsweringRef
    requires = frozenset({Generate})

    async def verify(self, engine) -> Verdict:
        events = engine[ArtifactRollout]
        answers = await engine[Generate]([ev.input for ev in events])
        items = [Envelope(request_id=ev.request_id, contents=ev.response.output.contents)
                 for ev in events]
        _SEEN.update(answers=[a.first_of("text").text for a in answers],
                     gaps=await engine[TeacherForce](items),
                     stack=await engine[StackReport]())
        return Verdict(passed=True, per_event={ev.request_id: True for ev in events})


class _PricesReplies(Verifier):
    """Asks its reference only questions about the contestant's reply and its stack."""

    reference: _AnsweringRef
    requires = frozenset({TeacherForce, StackReport})

    async def verify(self, engine) -> Verdict:
        raise AssertionError("only wired, never run")


@pytest.fixture
def _three_caps(monkeypatch):
    """The fake engine provides a corpus capability and two that are not."""
    monkeypatch.setattr("hls.verify.base.supports",
                        lambda cls: frozenset({Generate, TeacherForce, StackReport}))


def _token_contract():
    """The two contract facts check_wiring reads for a teacher-forcing text gate."""
    return SimpleNamespace(output_modalities=frozenset({"text"}),
                           protocol=SimpleNamespace(emits_token_ids=True))


def test_a_caching_reference_caches_its_rollout_and_nothing_else(tmp_path, _three_caps):
    """The cache stores answer envelopes to prompt envelopes, the shape of a corpus
    capability. A teacher-force call takes reply items and returns gap vectors, and a
    stack report takes nothing, so wrapping either would take the verify phase down, and
    neither answer is one another run could be served. The rollout of a row is cached;
    every other call reaches the engine, on every run."""
    ref = _AnsweringRef(decode=Greedy(), cache=True)
    task = _task([Tenant(id="a", workload=_wl(),
                         verifiers=[_AsksEverything(reference=ref)])])
    trace = Trace(events=[_ev("a0", "a")])

    def run(writer: str) -> tuple[list[str], StoreAnswers]:
        _LOG.clear()
        _SEEN.clear()
        answers = _answers(tmp_path, writer)
        _verify_with_answers(task, trace, answers)
        return [tag for kind, tag in _LOG if kind == "call"], answers

    calls, first = run("g1+verify+1")
    assert calls == ["generate", "teacher_force", "stack_report"]
    assert _SEEN == {"answers": ["gen"], "gaps": [[0.0, 0.1]], "stack": {"torch": "x"}}
    [path] = first.written
    stored = AnswerFile.model_validate_json((tmp_path / "answers" / path).read_bytes())
    assert (stored.fingerprint, [e.request_id for e in stored.answers]) == (ref.fingerprint(task), ["a0"])

    # A later run is served the rollout from the area and asks the engine the rest again.
    assert run("g2+verify+1")[0] == ["teacher_force", "stack_report"]
    assert _SEEN == {"answers": ["gen"], "gaps": [[0.0, 0.1]], "stack": {"torch": "x"}}


def test_check_wiring_refuses_a_caching_reference_required_for_another_capability(
        _three_caps):
    """A reference with caching on that a verifier requires for teacher-forcing a reply
    or reporting its stack is configured to cache answers the cache cannot hold.
    Refused at authoring, naming the capabilities, rather than run under a flag that
    says otherwise."""
    v = _PricesReplies(reference=_AnsweringRef(decode=Greedy(), cache=True))

    with pytest.raises(WiringError, match=r"cache=True.*StackReport, TeacherForce"):
        check_wiring([v], _token_contract())


def test_check_wiring_accepts_a_caching_reference_required_only_for_its_rollout(
        _three_caps):
    """The refusal reads what a verifier requires of a caching reference, not what its
    engine can do: this engine also teacher-forces, and a verifier that requires only
    the rollout may still cache it. A reference with caching off may be required for
    anything."""
    check_wiring([_RolloutVerifier(reference=_FakeRef(decode=Greedy(), cache=True))],
                 _token_contract())
    check_wiring([_PricesReplies(reference=_AnsweringRef(decode=Greedy(), cache=False))],
                 _token_contract())


class _RolloutAndPrices(Verifier):
    """Requires the rollout of a caching oracle, and teacher forcing and the stack of a
    second reference that does not cache."""

    oracle: _FakeRef
    pricer: _AnsweringRef
    requires = frozenset({Generate, TeacherForce, StackReport})

    async def verify(self, engine) -> Verdict:
        raise AssertionError("only wired, never run")


def test_check_wiring_reads_each_reference_for_what_it_provides(monkeypatch):
    """The refusal is per reference. The caching oracle here is required only for its
    rollout. The teacher forcing and the stack report come from another reference that
    does not cache, so nothing is asked of the cache that it cannot hold."""
    monkeypatch.setattr("hls.verify.base.supports", lambda cls: {
        _CountingEngine: frozenset({Generate}),
        _AnsweringEngine: frozenset({TeacherForce, StackReport})}[cls])
    v = _RolloutAndPrices(oracle=_FakeRef(decode=Greedy(), cache=True),
                          pricer=_AnsweringRef(decode=Greedy(), cache=False))

    check_wiring([v], _token_contract())
