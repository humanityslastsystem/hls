"""The verification core: run a task's verifiers over a trace and combine their verdicts.

This is the verification core the verify role, the oracle fill's checks and every rescoring
run. The rules it keeps, which the viewer, the record and every rescoring rely on: the
matrix's columns follow the trace's event order; a pooled verdict keys on ``event_key`` and a
per-tenant one on ``request_id``; diagnostics indices count the (tenant, verifier) pairs that
ran, and a pooled verdict is written once and aliased after; any failed request skips every
verifier and every reference load; a reference engine loads on its first use, on the engine
host the caller hands in, and at most one verifier's engine set is live at a time; a reference
with caching on answers corpus capabilities from the task's answers area when the caller gives
one. ``may_deploy`` says ahead of a verify, from the same area, whether any engine may load, so
the verify stage copies the model's weights only then. A stored trace (``StoredTrace``) is never
held: the core keys and counts from its summary, and each verifier takes its events from a view
that reads them from the store on each pass."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

from hls.core.capabilities import cap_method
from hls.core.faults import HarnessFault
from hls.core.payloads import Envelope
from hls.core.trace import Trace, TraceEvent
from hls.trace.store import StoredEvents, StoredTrace
from hls.core.verdict import Verdict
from hls.spec.taskspec import TaskSpec
from hls.util import gzip_text_writer, json_safe
from hls.verify.answers import CORPUS_CAPABILITIES, StoreAnswers
from hls.verify.base import EngineAccessor, WiringError, capability_providers
from hls.verify.inprocess import EngineHost, Release

log = logging.getLogger(__name__)


def _verifier_config_key(v) -> str:
    """A stable identity for a verifier's CONFIGURATION.

    Two verifiers with the same settings share a key even when they are different
    objects, which is what a pooled gate needs: calibration injection rebuilds each
    tenant's verifier, so a task that declares one shared instance still arrives here
    as several equal copies. Falls back to object identity for anything that will not
    serialize, which degrades to one pool per verifier object rather than crashing."""
    from hls.core.poly import dump_spec

    try:
        return f"{type(v).__name__}:{json.dumps(dump_spec(v), sort_keys=True)}"
    except Exception:  # noqa: BLE001 -- identity is a safe fallback, never a wrong answer
        return f"{type(v).__name__}:id{id(v)}"


def _verdict_key(ev, pooled: bool) -> str:
    """The key under which a verdict's per-event maps (``per_event``,
    ``event_rationale``, ``event_diagnostics``) hold ``ev``.

    A pooled verifier grades every tenant's events as one population, so its maps span
    tenants and key on ``event_key`` (``"<tenant_id>:<request_id>"``); request_id alone
    renumbers from 0 within each tenant and two tenants over the same dataset would
    collide. A per-tenant verifier only ever sees its own tenant's slice, within which
    request_id is unique, and keys on that."""
    return ev.event_key if pooled else ev.request_id


def _event_records_name(index: int, verifier: str) -> str:
    """File name of the per-event sidecar for the verdict keyed ``<index>:<Verifier>``
    in ``verdict.json``'s diagnostics."""
    return f"replies.{index}.{verifier}.jsonl.gz"


def _write_event_records(eval_dir: Path, index: int, verifier: str, vd: Verdict) -> None:
    """Persist one verifier's per-event diagnostics beside ``verdict.json``, as
    ``replies.<index>.<Verifier>.jsonl.gz`` with one JSON record per graded event;
    ``<index>:<Verifier>`` is the verdict's key in ``verdict.json``'s diagnostics.

    ``verdict.json`` keeps the run-level summary; the per-event detail lives here so
    it can be re-read without inflating that file. For the token gate each record is
    an ``hls.verify.flips.ReplyFlips``, the same schema a calibration serving stores per
    event, which is what makes a later re-count of a graded trace a file read."""
    path = eval_dir / _event_records_name(index, verifier)
    with gzip_text_writer(path) as fh:
        for key, diag in vd.event_diagnostics.items():
            fh.write(json.dumps(json_safe({"event_key": key, **diag})) + "\n")


def _event_passed(vd: Verdict, key: str) -> bool:
    """What verdict ``vd`` says about the event it holds under ``key``.

    A verifier that grades event by event publishes ``per_event``. One that grades the
    whole trace at once has no per-event opinion and publishes none, and every event it
    covers then carries the trace verdict."""
    if not vd.per_event:
        return vd.passed
    return bool(vd.per_event.get(key, False))


def _matrix_cell(vd: Verdict, key: str) -> dict:
    """One ``per_verifier_results`` cell: what verdict ``vd`` recorded for the event it
    holds under ``key``. ``score``/``scores`` are always empty here -- the matrix
    carries pass/fail plus the verifier's own per-event detail, not metric scores."""
    return {"passed": _event_passed(vd, key),
            "score": None, "scores": {},
            "rationale": vd.event_rationale.get(key),
            "diagnostics": dict(vd.event_diagnostics.get(key, {}))}


def write_verdict(eval_dir: Path, verdict: Verdict, per_verifier_results: list) -> None:
    """Write ``verdict.json`` in the viewer's RolloutVerdict shape: the combined gate plus
    the per-verifier, per-event results the eval tab renders. Non-finite numbers become
    null (``json_safe``), since strict RFC 8259 readers reject Infinity and NaN."""
    verdict_doc = {
        "aggregate_passed": verdict.passed,
        "failure_mode": verdict.failure_mode,
        "per_event": dict(verdict.per_event),
        "diagnostics": verdict.diagnostics,
        "per_verifier_results": per_verifier_results,
    }
    (eval_dir / "verdict.json").write_text(json.dumps(json_safe(verdict_doc), indent=2))


async def _deploy_reference(taskspec, host: EngineHost, tenant, v, cfg) -> tuple[object, Release]:
    """Build one reference engine and load it on *host*, returning the ``(proxy, release)``
    pair the verify phase's live pool holds.

    Called on a capability's first use rather than before verify, so an engine whose
    answers are all cached is never built. ``check_wiring`` validates the declared
    ``engine_cls`` from config; this checks what ``build_engine`` actually returned,
    which only the built object can show."""
    log.info("tenant %s, verifier %s: deploying %s",
             tenant.id, type(v).__name__, type(cfg).__name__)
    eng = cfg.build_engine(taskspec)
    if not isinstance(eng, cfg.engine_cls):
        raise WiringError(
            f"{type(v).__name__}: {type(cfg).__name__}.build_engine returned "
            f"{type(eng).__name__}, not the declared engine_cls "
            f"{cfg.engine_cls.__name__}")
    return await host.acquire(eng)


def may_deploy(taskspec: TaskSpec, prompts: Iterable[Envelope], answers: StoreAnswers | None) -> bool:
    """Whether ``verify_trace`` may deploy a reference engine for a trace whose events sent
    *prompts*, decided before it runs from what *answers* holds, so the verify stage copies the
    task model's weights only when an engine may load. False when no verifier of *taskspec* has
    an engine, or when every engine is a reference with caching on (``Reference.cache``), its
    verifiers require only corpus capabilities of it (``CORPUS_CAPABILITIES``,
    ``Verifier.requires``), and *answers* holds an answer to every prompt it may be asked
    (``StoreAnswers.unanswered``). Those are the prompts of each tenant whose verifiers hold it,
    and every prompt for a pooled verifier. *prompts* is read only as far as the first prompt
    that misses, and not at all when an engine may deploy whatever the area holds. The answer
    errs towards True, where ``verify_trace`` asks less: a failed request skips every deploy,
    and a verifier asks only about the events that were answered."""
    # Each reference that may be served whole: its fingerprint and the tenants whose prompts it
    # is asked, None for every prompt.
    served: dict[int, tuple[str, set[str] | None]] = {}
    for tenant in taskspec.effective_tenants():
        for v in tenant.verifiers:
            provided = capability_providers(v)
            for cfg in {id(c): c for c in v.engines()}.values():
                asked = {cap for cap in v.requires if provided.get(cap) is cfg}
                # check_wiring already refuses a caching reference required for anything but a
                # corpus capability; the last test keeps a task built without it on the safe side.
                if answers is None or not cfg.cache or not asked or asked - set(CORPUS_CAPABILITIES):
                    return True
                fingerprint, scope = served.get(id(cfg), (cfg.fingerprint(taskspec), set()))
                if getattr(v, "pooled", False) or scope is None:
                    scope = None
                else:
                    scope.add(tenant.id)
                served[id(cfg)] = (fingerprint, scope)
    if not served:
        return False
    assert answers is not None
    for prompt in prompts:
        tenant_id = prompt.metadata.get("tenant_id")
        for fingerprint, scope in served.values():
            if (scope is None or tenant_id in scope) and answers.unanswered(fingerprint, prompt):
                return True
    return False


def ungraded_verdict(taskspec: TaskSpec, events: Sequence[TraceEvent]) -> tuple[Verdict, list] | None:
    """What ``verify_trace`` returns for a trace of *events* without grading any of them: the
    failed verdict and its ``per_verifier_results`` matrix when any request failed (no
    response, or an error), None when every one was answered and the trace must be graded.
    It refuses first, as ``verify_trace`` does: a trace of no events (HarnessFault), and pooled
    verifiers configured differently across tenants (WiringError).

    Of each event it reads only whether it has a response, its ``error`` and its ``event_key``,
    which a summary event holds as the full one does (``hls.trace.store.read_summary``), so
    the verify role decides from a trace's summary, and writes a failed verdict from it,
    without reading the trace's chunks."""
    if not events:
        raise HarnessFault("verify: the test split selected zero items; nothing was evaluated")
    tenants = taskspec.effective_tenants()
    all_verifiers = [v for t in tenants for v in t.verifiers]

    # A pooled verifier grades the whole trace once and every tenant shares that
    # single verdict, so a pooled gate is a property of the run, not of a tenant.
    # Two distinct instances of one pooled class across tenants would therefore be
    # two configurations of the same run-level gate (different thresholds,
    # multipliers, references) with no way to honour both: whichever tenant ran
    # first would silently decide the answer for all of them. Refuse instead.
    # Declaring one shared instance on several tenants is the supported way to say
    # "this gate applies to the whole run".
    # Keyed on CONFIGURATION, not object identity: calibration injection rebuilds
    # each tenant's verifier (a model_copy carrying the loaded calibration), so one
    # shared instance legitimately arrives here as several equal objects. Identity
    # would reject that wiring; what cannot be honoured is two pooled verifiers
    # whose settings actually differ.
    pooled_instances: dict[type, dict[str, object]] = {}
    for v in all_verifiers:
        if getattr(v, "pooled", False):
            pooled_instances.setdefault(type(v), {})[_verifier_config_key(v)] = v
    for cls, instances in pooled_instances.items():
        if len(instances) > 1:
            raise WiringError(
                f"{len(instances)} differently-configured {cls.__name__} instances "
                f"are declared across tenants, but {cls.__name__} is POOLED: it "
                f"grades every tenant's events as one population and returns one "
                f"verdict for the whole run, so it cannot honour a per-tenant "
                f"configuration. Build ONE instance and declare that same object on "
                f"each tenant that the gate should cover.")

    # Census of the failed events keyed by their verbatim error string (a
    # deadline miss reads "transport: request did not complete by the drive
    # deadline ...", a breaker skip "aborted: circuit-breaker tripped ...", and a
    # request the breaker cut short "abandoned: circuit-breaker tripped while the
    # request was outstanding"), so the verdict --
    # and the Score.zero diagnostics copied from it -- record which failure
    # mode zeroed the run, not only that one did.
    errs = Counter((ev.error or "no response")
                   for ev in events if ev.response is None or ev.error)
    if not errs:
        return None
    # Any failed request already zeroes the run (hard gate), so grading the
    # survivors cannot change the score -- skip verification and its
    # reference-engine deploys entirely; the census names the failure.
    # per_event keys on event_key (tenant-namespaced); request_id alone is
    # not unique across tenants sharing a dataset path.
    failed_requests = {"count": sum(errs.values()), "total": len(events),
                       "errors": dict(errs.most_common())}
    per_event = {ev.event_key: False for ev in events}
    results = [[{"passed": False, "score": None, "scores": {}, "rationale": None,
                 "diagnostics": {}} for _ in events] for _ in all_verifiers]
    reason = ("verification skipped: {count} of {total} requests failed"
              .format(**failed_requests))
    return (
        Verdict(passed=False, per_event=per_event, failure_mode="failed_requests",
                diagnostics={"reason": reason,
                             "failed_requests": failed_requests}),
        results,
    )


def _answered(trace: Trace | StoredTrace, tenant_id: str | None = None) -> list[TraceEvent] | StoredEvents:
    """The events of *trace* answered without error, of the tenant *tenant_id* when one is
    named, in emission order: a list for a trace held in memory, a view that reads them from the
    store on each pass for a stored one."""
    if isinstance(trace, StoredTrace):
        return trace.answered() if tenant_id is None else trace.answered(tenant_id)
    events = trace.events if tenant_id is None else trace.events_for(tenant_id)
    return [ev for ev in events if ev.response is not None and not ev.error]


async def verify_trace(
    taskspec: TaskSpec, trace: Trace | StoredTrace, *, eval_dir: Path | None, host: EngineHost,
    answers: StoreAnswers | None,
) -> tuple[Verdict, list]:
    """Build and load each verifier's declared engines, run verify(), combine.

    Shared structural layer (metric-agnostic), before/around the verifiers
    (``ungraded_verdict``): empty trace -> INVALID; any request failed -> failed
    verdict, skip deploy and every verifier, since one failed request hard-fails
    the run. (A label metric raising on an unlabeled batch -> INVALID is the
    metric's own concern, surfaced through verify().)

    Multi-tenant: groups events by tenant_id (via Trace.by_tenant()), then for each
    tenant runs its own verifiers against its own slice.  Single-tenant tasks
    (one synthetic Tenant("t0"), whose events the driver stamps tenant_id="t0")
    produce the same verdict as one verifier list over the whole trace would.

    Each reference engine a verifier needs is built and loaded on its first use on *host*,
    the engine host the caller created, and released through it once no verifier needs it or
    the verify ends. A reference with caching on answers its corpus capabilities from
    ``answers``, loading only for a miss; None asks every question of the engine. Per-event
    sidecars are written into ``eval_dir``, and none when it is None. Returns the verdict and
    the ``per_verifier_results`` matrix.

    A *trace* held in memory is read as it is. A stored one is never held: what the core keys and
    counts on comes from its summary (``StoredTrace.summary``), which holds every event's
    identity, tenant, answer and error, and each verifier is handed a view of its events that
    reads them from the store on each pass (``StoredTrace.answered``)."""
    # Every event's identity, tenant, answer and error, in emission order.
    events = trace.summary() if isinstance(trace, StoredTrace) else trace.events
    ungraded = ungraded_verdict(taskspec, events)
    if ungraded is not None:
        return ungraded
    tenants = taskspec.effective_tenants()

    # Per-tenant verify: each tenant's verifiers see only that tenant's slice.
    # tenant_verdicts[i] holds the per-verifier Verdicts for tenants[i] (same
    # order), so the aggregation/diagnostics loops zip the two directly.
    tenant_verdicts: list[list[Verdict]] = []
    # (own_keys, pooled, verdict) per verifier, flat across tenants. own_keys is the
    # set of event_keys belonging to that verifier's tenant; events from other
    # tenants are not-applicable cells in per_verifier_results, not failures -- a
    # False default would conflate "this verifier failed the event" with "this
    # verifier never graded it". own_keys holds event_keys rather than request_ids
    # so a same-request_id sibling in another tenant is not mistaken for an own
    # event. `pooled` says which key that verdict's per-event maps use (see
    # `_verdict_key`).
    per_verifier_meta: list[tuple[set[str], bool, Verdict]] = []

    # Engine cache, live across the whole verify phase (all tenants/verifiers).
    # Keyed by id(cfg): a config object reused by multiple verifiers -- e.g. a
    # shared oracle Reference used by every tenant -- is deployed once and its
    # proxy reused, instead of redeploying per verifier. Before each verifier we
    # reconcile the cache to exactly the engines it needs: evict (release) any
    # live engine it does not need first -- the slot pool is fail-loud (acquire
    # raises when no device is free, it does not queue), so the device must be
    # freed before the next acquire -- then deploy any missing one. So at most
    # the current verifier's own engine set is ever live; two distinct oracles
    # are never up at once. Limits (deferred): structurally-identical but
    # distinct cfg objects are not deduped (would need a config __eq__/__hash__),
    # and a distinct engine reached again after an intervening one is redeployed.
    live: dict[int, tuple[object, Release]] = {}   # id(cfg) -> (proxy, release)
    # One verdict per pooled verifier instance, computed on first encounter over
    # the whole trace and shared by every tenant thereafter (see below). Keyed by
    # id(v) rather than by class so two configurations of one pooled class could
    # never collapse into each other; declaring two is refused outright above, and
    # this keying is what makes that refusal the only way they can differ.
    pooled_verdicts: dict[str, Verdict] = {}
    all_good = _answered(trace)
    try:
        for tenant in tenants:
            good_tenant = _answered(trace, tenant.id)
            per_v_for_tenant: list[Verdict] = []
            if not good_tenant:
                # No events for this tenant (errored events cannot reach here --
                # ungraded_verdict returned their verdict). Skip its engine deploy and
                # verify: grading [] would vacuously trip each metric's "no ground
                # truth" guard (all([]) == True -> HarnessFault) or ZeroDivisionError.
                tenant_verdicts.append(per_v_for_tenant)  # empty: no verifiers ran
                continue
            for v in tenant.verifiers:
                pooled = getattr(v, "pooled", False)
                # A pooled verifier grades the whole trace once and every tenant
                # shares that one verdict; a per-tenant verifier sees only its own
                # slice. The cache is consulted before any engine work: a pooled
                # gate is a property of the run, so declaring it on each of five
                # tenants must not run it -- or deploy its reference -- five times,
                # and must not produce five different answers.
                if pooled and _verifier_config_key(v) in pooled_verdicts:
                    per_v_for_tenant.append(pooled_verdicts[_verifier_config_key(v)])
                    continue
                provided = capability_providers(v)            # cap -> cfg
                required = {id(cfg): cfg for cfg in provided.values()}
                # Evict engines this verifier does not need, freeing the device(s)
                # before deploying new ones (fail-loud pool, see above).
                for key in [k for k in live if k not in required]:
                    _, release = live.pop(key)
                    await release()
                # Bind one provider per capability, deploying on FIRST USE rather
                # than here. The cache wrapper below only calls through on a miss,
                # so an engine whose every answer is cached is never deployed at all
                # and holds no GPU for a call it never makes. It also starts a
                # deployed engine's own scheduler clock only when it is needed: a
                # verifier holding two references that acquired both up front would
                # burn the first's wall-clock limit idling while the second queued
                # for a device.
                cap_to_provider = {}
                # The same engines without their cache, for a question whose answer
                # depends on the contestant (see EngineAccessor.uncached). Same deploy
                # either way -- only the cache wrapper differs -- so this costs no
                # extra device.
                cap_to_uncached = {}
                for cap, cfg in provided.items():
                    # *args: capabilities do not share a signature (StackReport takes
                    # none), so this stands in for the bound method as it is.
                    async def call(*args, cap=cap, cfg=cfg, key=id(cfg),
                                   tenant=tenant, v=v, **kwargs):
                        if key not in live:
                            live[key] = await _deploy_reference(taskspec, host, tenant, v, cfg)
                        return await getattr(live[key][0],
                                             cap_method(cap))(*args, **kwargs)
                    cap_to_uncached[cap] = call
                    # Only a corpus capability: the wrapper stores answer envelopes to
                    # prompt envelopes, and every other capability takes or returns
                    # something else (CORPUS_CAPABILITIES).
                    if (answers is not None and cfg.cache
                            and cap in CORPUS_CAPABILITIES):
                        call = answers.wrap(cfg.fingerprint(taskspec), call)
                    cap_to_provider[cap] = call
                if pooled:
                    # First encounter (the cache miss that fell through above):
                    # grade every tenant's events as one population.
                    accessor = EngineAccessor(artifact_events=all_good,
                                              cap_to_provider=cap_to_provider,
                                              cap_to_uncached=cap_to_uncached)
                    cached = await v.verify(accessor)
                    pooled_verdicts[_verifier_config_key(v)] = cached
                    log.info("pooled verifier %s: graded %d events across %d "
                             "tenants as ONE population",
                             type(v).__name__, len(all_good), len(tenants))
                    per_v_for_tenant.append(cached)
                else:
                    # Verifiers see only non-errored events from their tenant's slice.
                    accessor = EngineAccessor(artifact_events=good_tenant,
                                              cap_to_provider=cap_to_provider,
                                              cap_to_uncached=cap_to_uncached)
                    per_v_for_tenant.append(await v.verify(accessor))
            tenant_verdicts.append(per_v_for_tenant)
            own_keys = {ev.event_key for ev in events if ev.tenant_id == tenant.id}
            per_verifier_meta.extend(
                (own_keys, getattr(v, "pooled", False), vd)
                for v, vd in zip(tenant.verifiers, per_v_for_tenant))
    finally:
        for _proxy, release in live.values():
            await release()

    # Aggregate per_event across all tenants, over each tenant's answered events. The
    # run-level map keys on event_key (tenant-namespaced), because request_id alone
    # is not unique across tenants driving the same dataset path; the key each
    # verdict's own maps use depends on whether it is pooled (see `_verdict_key`).
    per_event: dict[str, bool] = {}
    for tenant, t_verdicts in zip(tenants, tenant_verdicts):
        for ev in (ev for ev in events if ev.tenant_id == tenant.id and ev.response is not None and not ev.error):
            per_event[ev.event_key] = all(
                _event_passed(vd, _verdict_key(ev, getattr(v, "pooled", False)))
                for v, vd in zip(tenant.verifiers, t_verdicts))

    # Errored events never reach this point (ungraded_verdict returned), so
    # the only failure left to label is a verifier gate: "correctness".
    aggregate_passed = all(
        all(vd.passed for vd in t_verdicts) for t_verdicts in tenant_verdicts)
    diagnostics: dict = {}
    # A pooled verdict is shared by every tenant that declares the gate. Its
    # diagnostics and per-event records are written once, under the first key it
    # appears at; each later key holds a pointer to that entry (``pooled_alias_of``
    # and, when per-event records exist, their ``events_file``), so verdict.json
    # shows one gate covering several tenants rather than several copies of it.
    first_index: dict[int, int] = {}
    failed_gates: list[dict] = []
    idx = 0
    for tenant, t_verdicts in zip(tenants, tenant_verdicts):
        for v, vd in zip(tenant.verifiers, t_verdicts):
            name = type(v).__name__
            if not vd.passed:
                gate = {"verifier": name, "tenant": tenant.id, "key": f"{idx}:{name}"}
                # Verifiers that name their own rejected rules say which fired;
                # the rest are identified by the gate alone.
                rules = vd.diagnostics.get("rejections")
                if rules:
                    gate["rejections"] = list(rules)
                failed_gates.append(gate)
            first = first_index.get(id(vd))
            if first is not None:
                alias: dict = {"pooled_alias_of": first}
                if vd.event_diagnostics:
                    alias["events_file"] = _event_records_name(first, name)
                diagnostics[f"{idx}:{name}"] = alias
            else:
                first_index[id(vd)] = idx
                diagnostics[f"{idx}:{name}"] = vd.diagnostics
                if eval_dir is not None and vd.event_diagnostics:
                    _write_event_records(eval_dir, idx, name, vd)
            idx += 1
    failure_mode = None if aggregate_passed else "correctness"
    if failed_gates:
        # "correctness" covers every verifier gate, so the verdict names which ones
        # rejected and why. Read this before the per-verifier diagnostics below.
        diagnostics["failed_gates"] = failed_gates
    verdict = Verdict(passed=aggregate_passed, per_event=per_event,
                      diagnostics=diagnostics, failure_mode=failure_mode)
    # Matrix is [verifier][event] over all events (rectangular, column-aligned to
    # `events` -- the viewer relies on that). Each row belongs to one (tenant,
    # verifier) pair and reports only that tenant's events; a foreign-tenant event
    # is not-applicable (passed=None), distinct from an event this row's verifier
    # graded and failed (passed=False). A pooled verifier declared on every tenant
    # contributes one row per tenant, and those rows together cover every event.
    per_verifier_results = [
        [
            _matrix_cell(vd, _verdict_key(ev, pooled))
            if ev.event_key in own_keys else
            {"passed": None, "score": None, "scores": {},
             "rationale": None, "diagnostics": {}}
            for ev in events
        ]
        for own_keys, pooled, vd in per_verifier_meta
    ]
    return verdict, per_verifier_results
