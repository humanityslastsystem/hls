"""Wiring test for tasks.multitenant_slo.task.

Cross-checks every load-bearing constant against the derivation block in task.py:
the 4 tenant ids, per-tenant SESSION rates AND Gamma scales (both recomputed from
the CHOSEN-portion formula, never hardcoded), the counts that realize those
portions at the request level, output caps, StreamingLatencySLO targets, the
serving-envelope budget the multi-turn tenants must fit, that EVERY tenant gates at
the token level (its own TokenFaithfulnessVerifier over the shared HF oracle), and
the reasoning tenant's train->dev split mapping. TASK is built at import, so a
construction-time model-validate failure (e.g. a verifier/contract wiring error)
fails collection.
"""
from __future__ import annotations

import pytest

from hls.core.capabilities import StackReport, TeacherForce
from hls.reference.references import HFReference
from hls.spec.datasource import Count
from hls.spec.workload import Gamma, StreamingLatencySLO
from hls.verify.verifiers import TokenFaithfulnessVerifier
from tests.unit.brief_task import render_brief

# The re-derived mix (task.py's derivation block): portion of the offered REQUEST rate,
# mean turns/session (measured on the pinned pools for the multi-turn tenants), arrival CV.
_PORTIONS = {"chat": 0.18, "api_short": 0.60, "reasoning": 0.12, "long_session": 0.10}
_TURNS = {"chat": 2.3914, "api_short": 1.0, "reasoning": 1.0, "long_session": 7.6266}
_CVS = {"chat": 1.0, "api_short": 1.5, "reasoning": 1.0, "long_session": 1.0}
# Per-tenant test-split size (sessions for the multi-turn tenants) and the request
# budget it realizes.
_TEST_COUNTS = {"chat": 126, "api_short": 999, "reasoning": 200, "long_session": 22}
_DEV_COUNTS = {"chat": 32, "api_short": 250, "reasoning": 50, "long_session": 6}
# The KEPT requests those test-split sizes actually realize, read back off the
# workload at TaskSpec.seed (task.py's counts table): the hash partition fixes which
# sessions exist and their turn counts, the removal list drops the listed sessions,
# then draw-time ramp-down membership (drivers.rampdown_turn_budget) excludes the
# follow-up turns of the two think-time tenants whose nominal dispatch falls past the
# arrival span. Every layer is a pure function of the rows and the seed, so these are
# constants of the task, not measurements of a run. For chat (the one tenant with a
# history_char_budget) the count is an UPPER BOUND on what a run serves: a session ends
# at the first turn whose conversation would exceed the budget, which depends on
# generated reply lengths. This is the workload a graded run and a calibration serving
# both drive.
_KEPT_TEST_REQUESTS = {"chat": 210, "api_short": 839, "reasoning": 199, "long_session": 78}
# The task's draw pools, kept at the calibration's sizes (a pool is part of
# the distribution definition, not sized to a split).
_POOLS = {"chat": 1188}

# Per-tenant table: cap, (ttft_ms, tpot_ms). Shape/scale are NOT hardcoded here --
# they are recomputed from the formulas above per-test.
_MIX = {
    "chat": (16384, (500.0, 50.0)),
    "api_short": (16384, (300.0, 50.0)),
    "reasoning": (16384, (300.0, 50.0)),
    "long_session": (16384, (1000.0, 50.0)),
}


def _total_rate():
    """The AGGREGATE rate is the one calibrated knob (re-pinned by a baseline ramp
    search), so it is read from the task rather than pinned here -- pinning it would
    make this test fail on a re-calibration that changed nothing about the mix. What
    IS fixed is the MIX, asserted by test_portions_are_the_authored_mix below: every
    per-tenant rate is a fixed portion of whatever TOTAL_RATE is."""
    from tasks.multitenant_slo.task import TOTAL_RATE
    return TOTAL_RATE


def _expected_session_rate(name):
    return _PORTIONS[name] * _total_rate() / _TURNS[name]


def _expected_shape(name):
    return 1.0 / _CVS[name] ** 2


def _expected_scale(name):
    return 1.0 / (_expected_shape(name) * _expected_session_rate(name))


def _tenants():
    from tasks.multitenant_slo.task import TASK
    return {t.id: t for t in TASK.effective_tenants()}


def test_five_tenants_with_expected_ids():
    assert list(_tenants()) == [
        "chat", "api_short", "reasoning", "long_session"]


def test_task_validates_and_serializes():
    """TASK is built at import, so its model_validator (which runs check_wiring per
    tenant) already passed. Confirm it also dumps to a stable provenance record."""
    from hls.core.poly import dump_spec
    from tasks.multitenant_slo.task import TASK

    rec = dump_spec(TASK)
    assert rec["id"] == "multitenant_slo"
    assert len(rec["tenants"]) == 4
    assert dump_spec(TASK) == rec  # deterministic


@pytest.mark.parametrize("name", list(_MIX))
def test_session_rate_matches_the_mix_formula(name):
    """SESSION rate = portion*TOTAL_RATE/turns; the Gamma's implied rate
    (1/(shape*scale)) must equal it to 1e-3. The /turns conversion is what keeps a
    multi-turn tenant's offered REQUEST load at its portion instead of turns-times it."""
    a = _tenants()[name].workload.arrival
    assert isinstance(a, Gamma)
    implied = 1.0 / (a.shape * a.scale)
    assert implied == pytest.approx(_expected_session_rate(name), abs=1e-3)


@pytest.mark.parametrize("name", list(_MIX))
def test_shape_scale_cap_slo_match_the_task(name):
    """Shape comes from the tenant's CV (shape = 1/CV^2) and scale from the rate
    formula -- neither is hardcoded in task.py; cap and SLO are cross-checked here."""
    cap, (ttft, tpot) = _MIX[name]
    w = _tenants()[name].workload
    assert w.arrival.shape == pytest.approx(_expected_shape(name))
    assert w.arrival.scale == pytest.approx(_expected_scale(name), abs=1e-6)
    assert w.output_length_cap.cap_tokens == cap
    assert isinstance(w.slo, StreamingLatencySLO)
    assert (w.slo.ttft_ms, w.slo.tpot_ms) == (ttft, tpot)


@pytest.mark.parametrize("name", list(_MIX))
def test_portions_are_the_authored_mix(name):
    """The invariant that survives any re-calibration of TOTAL_RATE: each tenant's
    share of the offered REQUEST rate is the authored portion, independent of the
    aggregate. Recovered from the wired Gamma: request_rate = turns/(shape*scale)."""
    a = _tenants()[name].workload.arrival
    request_rate = _TURNS[name] / (a.shape * a.scale)
    assert request_rate / _total_rate() == pytest.approx(_PORTIONS[name], abs=1e-6)


@pytest.mark.parametrize("name", list(_MIX))
def test_declared_turns_are_the_ones_the_session_rate_was_derived_from(name):
    """The brief multiplies each tenant's session rate by ``mean_turns_per_session``
    to state its request rate, so the declared factor must be the one ``_arrival``
    divided by; otherwise the brief states a load the driver does not offer."""
    w = _tenants()[name].workload
    assert w.mean_turns_per_session == _TURNS[name]
    assert w.prompt_fields()["request_rate"] == pytest.approx(
        _PORTIONS[name] * _total_rate(), abs=1e-6)


def test_brief_states_the_workload_in_sessions_and_requests(tmp_path):
    """What a contestant sizes to. The chat tenant's arrival process is a session rate
    2.39x below its request rate, so a brief that printed the process alone would state
    the smaller number."""
    from tasks.multitenant_slo.task import ARTIFACT
    injected = _inject(tmp_path, _calibration(q_max=0.0045))
    text, _ = render_brief(injected, ARTIFACT.brief)
    workload = text.split("## Workload", 1)[1].split("## Network Contract", 1)[0]
    assert "Offered load over all\ntenants: 1.11 requests/s" in workload
    chat = workload.split("### Tenant chat", 1)[1].split("### Tenant", 1)[0]
    assert "- Sessions: gamma arrivals (shape=1, scale=11.969), 0.08355 sessions/s" in chat
    assert "- Requests: 2.391 turns/session, 0.1998 requests/s" in chat
    assert "- Between turns: gamma think-time" in chat
    api = workload.split("### Tenant api_short", 1)[1].split("### Tenant", 1)[0]
    assert "- Requests: 1 turn/session, 0.666 requests/s" in api
    assert "Between turns" not in api
    # The rest of the brief never calls a session a request.
    assert "Requests arrive" not in text
    assert "Sessions arrive open-loop" in text


def test_portions_sum_to_the_total_rate():
    """The per-tenant request rates must sum to EXACTLY TOTAL_RATE -- a mix that does
    not partition the offered load would silently drive at another rate than the knob
    says."""
    total = sum(_TURNS[n] / (t.workload.arrival.shape * t.workload.arrival.scale)
                for n, t in _tenants().items())
    assert total == pytest.approx(_total_rate(), abs=1e-6)


@pytest.mark.parametrize("name", list(_MIX))
def test_counts_realize_the_portions_at_the_request_level(name):
    """Portions are CHOSEN; the test-split Counts are how they are REALIZED. A
    tenant's request budget is its session count x mean turns, and its share of the
    mix's ~1.7k requests must match its portion to within 1% (the slack is integer
    rounding on the session counts)."""
    ds = _tenants()[name].data_source
    assert ds.split.test == Count(value=_TEST_COUNTS[name])
    requests = {n: _TEST_COUNTS[n] * _TURNS[n] for n in _TEST_COUNTS}
    share = requests[name] / sum(requests.values())
    assert share == pytest.approx(_PORTIONS[name], abs=0.01)


@pytest.mark.parametrize("name", list(_MIX))
def test_all_tenants_deplete_together(name):
    """Pool-bounded drive: with counts chosen rate-proportional, every tenant's
    schedule spans the same wall clock (count/session_rate), so no tenant runs dry
    while others still drive. Checked against the mean within 2%."""
    tenants = _tenants()
    spans = {n: _TEST_COUNTS[n] * t.workload.arrival.shape * t.workload.arrival.scale
             for n, t in tenants.items()}
    mean_span = sum(spans.values()) / len(spans)
    # The achievable tolerance is set by session granularity: a tenant's span moves in
    # steps of one session's expected inter-arrival gap (shape*scale), so it can only be
    # placed within half a step of the target. For the fine-grained tenants that bound
    # is far below 2% and the 2% floor governs; for long_session (portion 0.05 x 7.63
    # turns) one session is ~10% of the span and the granularity bound governs.
    t = tenants[name]
    step = t.workload.arrival.shape * t.workload.arrival.scale
    tol = max(0.02, 0.5 * step / mean_span)
    assert spans[name] == pytest.approx(mean_span, rel=tol)


@pytest.mark.parametrize("name", list(_MIX))
def test_dev_is_a_quarter_of_the_request_budget(name):
    """DEV is 25% of each tenant's REQUEST budget (not of its pool), so the agent's
    dev rehearsal is the same mix at the same rates -- including the reasoning
    tenant, whose dev corpus is independent (openai/gsm8k train) and is SAMPLED to
    that share rather than drained."""
    ds = _tenants()[name].data_source
    assert ds.split.dev == Count(value=_DEV_COUNTS[name])
    # Integer session counts can only realize the quarter to within half a session of
    # the test split; for long_session (10 test sessions) that bound is 0.05 and
    # governs, for every other tenant the 0.02 floor does.
    tol = max(0.02, 0.5 / _TEST_COUNTS[name])
    assert _DEV_COUNTS[name] / _TEST_COUNTS[name] == pytest.approx(0.25, abs=tol)


#: Generated tokens per request, per tenant, measured on the calibration's
#: servings of its workload (natural end-of-sequence under the 16,384-token
#: output guard) and quoted in task.py's expected-tokens table. A reply's length is the
#: model's own stopping behaviour, so these are properties of the checkpoint on each
#: tenant's corpus.
_GENERATED_TOKENS_PER_REQUEST = {
    "chat": 3_352.2, "api_short": 586.8, "reasoning": 1_830.2, "long_session": 2_361.4}


def _graded_length(cal) -> int:
    """The graded trace length the deployed per-level ceilings are quoted at, in
    teacher-forced tokens: the upper end of the calibration's own realized spread, mean
    plus three standard deviations of the wider engine's per-serving lengths
    (``rate_side[*].tokens``). The artifact carries no expected length of its own;
    every graded trace's ceilings are recomputed at its actual token count, so the
    length a k table is quoted at has to be stated wherever it is quoted."""
    import statistics

    return round(max(statistics.mean(r.tokens) + 3 * statistics.stdev(r.tokens)
                     for r in cal.rate_side.values()))


def test_graded_trace_is_sized_to_the_calibrated_length():
    """The corpus sizing's whole point: the graded test half must realize the trace
    length the calibration was measured at, and comfortably clear the calibrated
    fail-closed floor.

    A calibration serving drives the whole graded workload through the task's own
    clocked driver, so its realized length is the graded length measured directly
    rather than a second estimate of it. What the committed numbers cross-check is the
    request side (task.py's kept-request table against what the servings served) and
    the token side (the expected-tokens table against the length the servings
    realized)."""
    from hls.verify.poisson import rate_rule_min_tokens
    from tasks.multitenant_slo.task import TASK

    cal = _committed_calibration()
    # What task.py's expected-tokens table claims the mix generates (~1,745,000
    # pooled).
    generated = sum(_KEPT_TEST_REQUESTS[t] * _GENERATED_TOKENS_PER_REQUEST[t]
                    for t in _KEPT_TEST_REQUESTS)
    assert generated == pytest.approx(1_745_000, rel=0.03)

    # Every serving drives the whole workload, with no session bound.
    assert cal.draw.max_sessions is None
    assert sum(_KEPT_TEST_REQUESTS.values()) == 1_326

    # The calibrated length: a serving's teacher-forced length, read off the
    # per-serving lengths the artifact stores.
    calibrated_length = max(sum(r.tokens) / len(r.tokens) for r in cal.rate_side.values())
    assert calibrated_length == pytest.approx(1_575_180, abs=1)
    # The yields include the guard-hitting replies of the rows the list has since
    # removed, so the table bounds the graded length from above.
    assert calibrated_length < generated

    # The verifier's floor sits far below the trace, and deliberately: it is the point
    # below which the rate rule loses its false-alarm guarantee, not an
    # anti-truncation rule. Shortening a trace enough to matter means dropping
    # requests (which fails the run outright) or ending replies early, and an early
    # end-of-sequence is a large-gap flip on every truncated reply, which the curve
    # rule counts at ceilings a handful of such events already exceed. The floor is
    # derived on the verifier from q_max and the family budget at the deployed
    # headroom, never stored in the artifact.
    v = TASK.effective_tenants()[0].verifiers[0].model_copy(update={"calibration": cal})
    assert v.min_tokens == rate_rule_min_tokens(
        cal.q_max, cal.delta_per_rung, v.rate_multiplier)
    assert v.min_tokens == 8_712
    assert v.min_tokens < calibrated_length * 0.05


def test_the_serving_envelope_is_the_checkpoints_own_limit():
    """Neither the baseline nor the vLLM oracle may pin --max-model-len.

    vLLM rejects a request whose prompt + max_tokens exceeds max_model_len, and every
    request carries the 16,384 output guard as max_tokens -- so a hand-set envelope has to
    stay above the longest prompt the corpus produces, forever. A request it rejects is an
    HTTP 400 and, under the no-failed-request gate, a zero.

    Omitting the flag makes vLLM derive the envelope from the checkpoint's own
    max_position_embeddings, which cannot drift away from the model.
    """
    from tasks.multitenant_slo.task import OUTPUT_GUARD, TASK

    assert "--max-model-len" not in TASK.baseline.serve_command, (
        "the baseline must not pin an envelope; vLLM should size it from the checkpoint")

    # The output guard is still what every request reserves on top of its prompt; that
    # arithmetic is the reason the envelope matters at all.
    for t_ in _tenants().values():
        assert t_.workload.output_length_cap.cap_tokens == OUTPUT_GUARD


@pytest.mark.parametrize("name", list(_MIX))
def test_every_tenant_gates_at_the_token_level(name):
    """Every tenant declares a TokenFaithfulnessVerifier carrying this task's ONE
    authored constant: 2x headroom over the measured rate ceiling. Everything else the gate runs on -- both
    rules' parameters and the fail-closed trace floor -- is measured, and comes from
    calibration.json.

    All four tenants share ONE verifier instance. The gate is pooled, so the runner
    grades it once for the whole run whatever the tenants declare; stating the
    constant once is what keeps it from drifting apart per tenant."""
    verifiers = _tenants()[name].verifiers
    assert len(verifiers) == 1
    v = verifiers[0]
    assert isinstance(v, TokenFaithfulnessVerifier)
    assert v.rate_multiplier == 2.0
    # The emitted-token consistency check stays off: this contract's
    # OpenAIChatProtocol() requests no logprobs, and check_wiring would refuse the
    # flag against it (flipping the protocol would also stale both calibrations).
    assert v.check_emit_consistency is False
    # The measured ceilings are never authored: q_max comes from calibration.json,
    # injected by the CLI loader, and the task refuses to run until then.
    assert v.calibration is None
    assert v.needs_calibration is True


def test_every_tenant_shares_one_gate_instance():
    """One instance across the mix, not four equal copies: the pooled runner grades
    a single verdict per verifier TYPE, so four copies would be four chances for the
    authored constants to diverge with no behavioural difference to show for it."""
    gates = {id(t.verifiers[0]) for t in _tenants().values()}
    assert len(gates) == 1


def test_token_verifiers_share_one_oracle_and_its_capabilities():
    """All four verifiers hold the SAME HFReference instance -- one deployed oracle
    teacher-forces every tenant -- and it is an HF reference because HFEngine is the
    only engine providing TeacherForce + StackReport (a VLLMReference here would fail
    check_wiring at TaskSpec construction)."""
    references = {id(t.verifiers[0].reference) for t in _tenants().values()}
    assert len(references) == 1
    oracle = next(iter(_tenants().values())).verifiers[0].reference
    assert isinstance(oracle, HFReference)
    assert {TeacherForce, StackReport} <= set(TokenFaithfulnessVerifier.requires)


def _task_dir():
    """The task's default measurements (``hls.measured``)."""
    from pathlib import Path

    return Path(__file__).resolve().parents[2] / "tasks" / "multitenant_slo" / "defaults"


def _committed_calibration():
    """The real calibration.json the task ships as its default, read as its schema defines it.

    Not through ``read_calibration``, which additionally applies the staleness gate:
    everything asserted below is about what the calibration measured, which a change of
    the fingerprint's definition does not alter. Reads the committed file rather than a fixture on
    purpose: a fixture cannot record real measured numbers."""
    from hls.verify.calibration import Calibration

    return Calibration.model_validate_json(
        (_task_dir() / "calibration.json").read_text())


def test_the_committed_calibration_is_fresh_for_the_shipped_task():
    """The shipped pair loads end to end: the artifact's persisted workload copy
    carries the same fingerprint the task computes today, history budget and
    ramp-down trimming included, so the staleness lock passes. A task edit that
    changes the teacher-forced population without a recalibration flips this test
    before it can mis-grade anything."""
    from hls.verify.calibration import read_calibration
    from tasks.multitenant_slo.task import TASK

    cal = read_calibration(_task_dir() / "calibration.json", TASK)[0]
    assert cal.q_max == pytest.approx(0.0019515060460739662, abs=5e-15)


def test_committed_calibration_passes_the_gates_that_do_not_depend_on_freshness():
    """The other gates read_calibration applies still hold on the shipped pair: the
    GPU-kind gate, and the calibration's own fp32 preflight."""
    from tasks.multitenant_slo.task import TASK

    cal = _committed_calibration()
    assert cal.gpu_kind == TASK.hardware.accelerator.kind == "H200"
    assert cal.preflight.passed is True
    assert cal.q_max > 0


def test_committed_calibration_carries_the_measured_pool():
    """The shipped ceilings are the calibration's two engines over
    their 8 servings each, and the validators lock them to exactly that: q_max the
    worst engine's mean rate (vllm binds, the two engines having converged to
    within two percent under the task's own clocked traffic), each per-level
    upper limit the Garwood bound for its own event count. Pinned here because
    these numbers are the deployed gate; a rebuild that quietly moved one would
    otherwise reach grading unnoticed."""
    cal = _committed_calibration()
    assert set(cal.rate_side) == set(cal.curve_side.engines) == {"vllm", "sglang"}
    assert [r.draws for r in cal.rate_side.values()] == [8, 8]
    assert cal.rate_side["vllm"].max_gap == 5.125
    assert cal.rate_side["sglang"].max_gap == 1.625
    assert cal.curve_side.engines["vllm"].exposure == 12_373_693
    assert cal.curve_side.engines["sglang"].exposure == 12_601_439
    assert cal.q_max == pytest.approx(0.0019515060460739662, abs=5e-15)
    assert cal.q_max == cal.rate_side["vllm"].mean
    # The calibration's pooled per-level clean event counts, ascending in severity, on
    # the grid the pool chose: the floor, every half-nat rung where the binding count
    # strictly drops, and the first empty rung. vllm's single event above 2.0 nats
    # (5.125, one isolated position in 12.4M tokens) holds the 2.0 rung; sglang saw
    # nothing past 1.5 nats.
    assert cal.curve_side.levels_nats == [0.5, 1.0, 1.5, 2.0, 5.5]
    assert [lvl.events for lvl in cal.curve_side.engines["vllm"].levels] == [73, 3, 1, 1, 0]
    assert [lvl.events for lvl in cal.curve_side.engines["sglang"].levels] == [89, 5, 2, 0, 0]
    # How much of the pool's text reached the output guard; inside the ceilings, not
    # excluded from them.
    assert (cal.capped_text.events, cal.capped_text.tokens) == (67, 1_097_728)
    # The pool members the ceilings describe: the clocked-serving members, each driving
    # the task's traffic through its engine's async API, each recording the serving
    # package version its own process reported.
    assert {n: e.config for n, e in cal.engines.items()} == {
        "vllm": "VLLMEngine(dtype='auto'), clocked serving",
        "sglang": "SGLangEngine(), clocked serving"}
    assert all(e.version for e in cal.engines.values())


def test_committed_calibration_sizing_holds():
    """The artifact may not ship unless both sizing inequalities held on the campaign
    that produced it -- the file records the verdicts so a reader never has to take
    that on trust."""
    from hls.calibrate import MIN_SERVINGS, _c_m
    from hls.verify.calibration import exposure_target

    cal = _committed_calibration()
    assert all(c.passed for c in cal.precision.values())
    assert all(c.passed for c in cal.tail.values())
    # The counter stop fired on its own terms and (P) passed on the single
    # pre-committed look.
    assert cal.sizing.stopped == "exposure target reached"
    assert (cal.tolerances.eps, cal.tolerances.mu_per_mtok, cal.tolerances.delta) == (
        0.05, 0.3, 1e-3)
    # (P) as recorded: c_m at the pool's smallest serving count, applied to every
    # engine, against the tolerance times that engine's own mean.
    assert cal.sizing.c_m == _c_m(min(r.draws for r in cal.rate_side.values()))
    assert cal.sizing.c_m == pytest.approx(1.7971507316515405, abs=1e-12)
    for engine, check in cal.precision.items():
        rate = cal.rate_side[engine]
        assert check.lhs == pytest.approx(cal.sizing.c_m * rate.sd, rel=1e-12)
        assert check.rhs == pytest.approx(cal.tolerances.eps * rate.mean, rel=1e-12)
    # (T) is a token counter against a task-independent constant, and the counter
    # stopped the calibration in the first round past it.
    target = exposure_target(cal.tolerances.mu_per_mtok)
    assert all(c.rhs == target for c in cal.tail.values())
    assert target == pytest.approx(12_296_265, rel=1e-6)
    # An engine's exposure is the sum of its serving lengths. Seven servings left both
    # engines short of the target and the eighth carried both past it, so the calibration
    # overshoots by one round: 2.5% (sglang) and 0.6% (vllm).
    for engine, check in cal.tail.items():
        tokens = cal.rate_side[engine].tokens
        assert check.lhs == cal.curve_side.engines[engine].exposure == sum(tokens)
        assert sum(tokens[:7]) < target <= sum(tokens)
    assert cal.tail["sglang"].lhs / target == pytest.approx(1.0248, abs=5e-5)
    assert cal.tail["vllm"].lhs / target == pytest.approx(1.0063, abs=5e-5)
    # ...above the serving floor, and each serving's drive clock is recorded so any
    # one of them can be re-driven.
    assert all(r.draws >= MIN_SERVINGS for r in cal.rate_side.values())
    assert {e: len(v) for e, v in cal.draw.serving_seeds.items()} == {
        "vllm": 8, "sglang": 8}


def test_committed_calibration_carries_no_advisory_text():
    """What the censuses flag is reported when a calibration is measured; the shipped
    file holds the measurement only."""
    import json

    doc = json.loads((_task_dir() / "calibration.json").read_text())
    assert not {"trust_warnings", "recurrence_warnings"} & set(doc)


def test_committed_calibration_yields_the_intended_thresholds():
    """The arithmetic the gate actually runs, end to end from the shipped file and the
    one authored constant -- the check that spans both halves of the wiring."""
    from tasks.multitenant_slo.task import TASK

    cal = _committed_calibration()
    v = TASK.effective_tenants()[0].verifiers[0]
    assert v.rate_multiplier * cal.q_max == 2 * cal.q_max
    assert v.rate_multiplier * cal.q_max == pytest.approx(0.0039030120921479325, abs=5e-15)
    # The curve ceilings at the length the graded half realizes. The artifact
    # records no expected length of its own (the ceilings are recomputed at each
    # graded trace's actual token count), so the length is stated here, at the upper
    # end of the calibration's own realized spread (_graded_length), which is where the
    # k table below is read off.
    n = _graded_length(cal)
    assert n == 1_698_214
    rows = cal.thresholds(n)
    assert [r["k_per_engine"]["vllm"] for r in rows] == [27, 7, 5, 5, 5]
    assert [r["k_per_engine"]["sglang"] for r in rows] == [30, 8, 6, 5, 5]
    # sglang binds the shallow rungs (89, 5 and 2 events at >=0.5, 1.0 and 1.5 nats
    # against vllm's 73, 3 and 1); vllm's lone 5.125-nat event binds the 2.0 rung, and
    # the empty 5.5 rung ties at the zero-count ceiling.
    assert [r["engine"] for r in rows] == ["sglang", "sglang", "sglang", "vllm", "vllm"]
    assert [r["k"] for r in rows] == [30, 8, 6, 5, 5]


def test_run_refuses_without_a_calibration_json(tmp_path):
    """The calibration gate, called exactly as a role calls it: pointed at a path with no
    calibration.json (what a task looks like before `hls calibrate` has run), it must
    refuse with the fix named -- never proceed on unauthored thresholds."""
    from hls.verify.calibration import CalibrationError, calibrated
    from tasks.multitenant_slo.task import TASK

    with pytest.raises(CalibrationError) as exc:
        calibrated(TASK, tmp_path / "calibration.json")[0]
    assert "hls calibrate" in str(exc.value)


def _calibration(q_max: float = 0.0045, **overrides):
    """A calibration.json-shaped record valid for THIS task (the workload fingerprint
    and gpu_kind read_calibration gates on)."""
    from tasks.multitenant_slo.task import TASK
    from tests.unit.calibration_fixtures import make_calibration

    return make_calibration(
        TASK, task="tasks.multitenant_slo.task",
        gpu_kind=TASK.hardware.accelerator.kind,
        engines={"hf_pinned": {"rates": [q_max] * 3}}, **overrides)


def _inject(tmp_path, cal):
    """Inject *cal* exactly as a role does: write it to the file the role is handed and
    run the loader over it."""
    import json

    from hls.verify.calibration import calibrated
    from tasks.multitenant_slo.task import TASK

    (tmp_path / "calibration.json").write_text(json.dumps(cal.model_dump(mode="json")))
    return calibrated(TASK, tmp_path / "calibration.json")[0]


def test_calibration_injects_into_every_tenants_verifier(tmp_path):
    """With a matching calibration.json given, EVERY tenant's verifier
    carries the loaded ceilings -- one artifact, one pooled gate."""
    from tasks.multitenant_slo.task import TASK

    cal = _calibration()
    injected = _inject(tmp_path, cal)
    for tenant in injected.effective_tenants():
        assert tenant.verifiers[0].calibration == cal, tenant.id
    # The original frozen spec is never mutated.
    assert all(t.verifiers[0].calibration is None for t in TASK.effective_tenants())


def test_brief_states_the_calibrated_thresholds_as_numbers(tmp_path):
    """The brief the BUILD agent reads states the token gate in NUMBERS, rendered from
    the injected calibration -- the real path, since the agent_build stage loads the task
    with its calibration injected (``load_task(..., calibration=True)``) before it renders
    the brief."""
    from hls.prompt import verification_fields
    from tasks.multitenant_slo.task import ARTIFACT

    cal = _calibration(q_max=0.0045)
    injected = _inject(tmp_path, cal)

    text, prov = render_brief(injected, ARTIFACT.brief)
    goal = {s.block: s for s in prov.sections}["goal"]
    assert goal.source == "task"     # the numbers ride the task's own block override

    v = verification_fields(injected)
    # The numbers the agent is told are the ones the gate ENFORCES, built from the same
    # authored constants and the same injected calibration/envelope the verifier reads.
    assert v["flip_rate_threshold"] == 2.0 * cal.q_max
    assert v["flip_curve_levels"] == [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    # Prose assertions run whitespace-normalized: the template hard-wraps its
    # sentences, and WHERE it wraps is not part of the contract being pinned.
    prose = " ".join(goal.rendered.split())
    # The rate threshold, stated at the enforced precision, never floored to two
    # figures: the fixture's 2.0 x 0.0045 renders as 0.9000 -- four significant figures with
    # the trailing zeros kept, the same width the live 0.3779% gate needs.
    assert v["flip_rate_pct"] == "0.9000"
    assert "more than 0.9000% of your served tokens" in prose
    # The flip criterion must describe the rule the verifier ENFORCES (gap > 0), not
    # argmax inequality. They differ on exact bf16 ties -- a different token id with
    # an identical reference log-probability.
    assert ("have a lower reference log-probability than the reference's most likely "
            "token") in prose
    assert ("Exact ties do not count: a token equally probable under the reference is "
            "not a deviation.") in prose
    # The removal list is not disclosed: it screens the dev rows on disk as well as
    # the graded draw (provisioning applies it to dev, the driver to test), so the
    # contestant meets one workload shape and there is no rule to state. Nothing about
    # a measurement exclusion either, since none exists.
    assert "removed from the workload" not in prose
    assert "excluded from the measurement" not in prose
    assert "disagree with that reference's greedy argmax" not in prose
    # The curve rule is stated qualitatively -- severity levels with calibrated
    # ceilings -- because the per-level ceilings depend on the graded trace's
    # length; the level grid itself is deliberately not enumerated.
    assert ("or if the number of such tokens exceeds a calibrated ceiling at any "
            "of several severity levels, where severity is that log-probability "
            "gap") in prose
    assert "ceilings scale with the run's total token count" in prose
    assert "0.5, 1.0" not in prose  # no level-grid enumeration
    # The reference path is stated completely: the oracle teacher-forces in 4,096-token
    # chunks against a KV cache, and a whole-sequence forward is a different numerical path.
    assert "scored in 4,096-token chunks against a KV cache" in prose
    # What fails is deviation, not aggressive-but-faithful engineering. The brief
    # states no figure for how much of the rate ceiling a trusted stack uses.
    assert ("The thresholds are calibrated on fast production-grade serving stacks. "
            "What fails is genuine deviation: quantized weights, altered algorithms, "
            "modified sampling") in prose
    assert "of the rate ceiling and less than" not in prose
    assert "passes comfortably" not in prose
    # The anti-conservatism incentive: the gate awards nothing for margin.
    assert ("The gate is pass/fail with no credit for margin -- a run that passes at "
            "90% of a ceiling scores identically to one at 10% -- so latency spent "
            "buying faithfulness headroom below the ceiling is pure loss.") in prose
    # The brief does not state the drive deadline (last scheduled arrival + 600 s,
    # drivers.py _Deadline), which the driver enforces.
    assert "the eval ends 10 minutes after the last" not in prose
    # The reference stack is fully pinned INSIDE the gate sentence -- every setting
    # that determines the reference's numerics, stated once, with no separate
    # "how to reproduce" recipe (the check must be unambiguous, not tutorialized).
    assert ("teacher-forced through a bf16 reference implementation of this model -- "
            "HuggingFace transformers, batch size 1, MATH SDPA backend, cuDNN "
            "attention disabled, the model class resolved from the checkpoint's "
            "declared architectures, and the linear-attention layers on the torch "
            "fallback path") in prose
    assert "To reproduce the check" not in prose
    # The serving-semantics passage: raw-logit greedy (this checkpoint's declared
    # sampling defaults named as the footgun, without quoting their values) and EOS
    # as finish_reason.
    assert ("Serve greedy: the argmax of the raw logits, with none of the checkpoint's "
            "generation_config sampling defaults applied") in prose
    assert ("(a stack that silently inherits those defaults fails outright)") in prose
    assert "Report end-of-sequence as finish_reason, never as an emitted token." in prose
    # The graded trace is the test half (~1.76M generated tokens at this model's
    # yields), and the dev-split reproduction guidance states the variance consequence
    # in the agent's own terms (a mis-proportioned or small sample misreads by ~50%).
    # No total token volume in the brief. Arrival rates and the output cap are stated and the
    # dev split is representative, so rate x measured mean reply length gets there; a stated
    # total only lets a contestant skip the measurement.
    assert "2,221,000" not in prose
    assert "1.9 million" not in prose
    assert ("so a sample drawn in different proportions -- or a small one -- can "
            "easily read 50% higher or lower than the number you are graded on") in prose
    assert "0.26" not in prose  # no expected-rate figure in the brief
    # No all-caps words for emphasis anywhere in the block (real
    # names and acronyms -- TTFT, TPOT, SLO, MATH, HuggingFace -- stay).
    assert "one pooled measurement over every served token" in prose
    assert "draw your sample in the per-tenant request proportions" in prose
    for caps in ("HIGHER IS BETTER", "SUM of credit", "ONE pooled", "SAME tenant",
                 "NUMBER of"):
        assert caps not in goal.rendered, caps
    # The reference stack is stated EXACTLY ONCE (the reproduction recipe); the
    # gate paragraph carries no parenthetical restatement of it.
    assert prose.count("MATH SDPA backend") == 1
    # Deliberately absent (harness statistics, not implementation guidance): the
    # threshold provenance, the fail-closed token floor, and anti-gaming hints.
    assert "Garwood" not in prose and "Poisson" not in prose
    assert "1,670,000" not in prose
    assert "truncated or empty" not in prose
    assert "pinned" not in prose
    # Provenance carries the same rendered block that went to the agent.
    assert goal.rendered in text


# ---- the baseline credit: the scoring artifact -----------------------------
#
# The task's score is a speedup over its own reference stack, so beside the measured
# correctness ceilings in calibration.json it carries a measured scoring constant in
# baseline.json: B, the reference stack's own mean per-request log credit. These tests
# pin the same three things the calibration ones do -- that the task refuses to run
# without it, that the shipped artifact is fresh for the shipped task, and that it
# reaches the object that grades with it.


def test_the_scorer_needs_a_measured_baseline_credit():
    """The authored task carries no B. Like the token gate's ceilings, it is measured
    and injected by the CLI loader, never written into task.py -- so a scoring constant
    cannot be edited into the benchmark by hand."""
    from tasks.multitenant_slo.task import TASK

    assert TASK.scorer.scores_against_baseline() is True
    assert TASK.scorer.baseline_log_credit is None


def test_run_refuses_without_a_baseline_json(tmp_path):
    """The baseline-credit gate, called exactly as a role calls it: pointed at a
    path with no baseline.json (what a task looks like before `hls baseline-credit`
    has run), it must refuse with the fix named. A run that proceeded would score every
    contestant against the SLO itself rather than against the reference stack, which is
    a different and much larger number, silently."""
    from hls.baseline_credit import BaselineCreditError, credited
    from tasks.multitenant_slo.task import TASK

    with pytest.raises(BaselineCreditError) as exc:
        credited(TASK, tmp_path / "baseline.json")[0]
    assert "hls baseline-credit" in str(exc.value)


def _baseline_credit(credits=(0.4, 0.6)):
    """A baseline.json-shaped record valid for THIS task: the campaign identity the
    loader gates on (workload fingerprint, load record, gpu kind, task module,
    reference server) with stand-in drives."""
    from hls.baseline_credit import (BaselineDrive, BaselineRecord, CampaignIdentity,
                                     credit_from_drives, load_record)
    from hls.verify.calibration import workload_fingerprint
    from tasks.multitenant_slo.task import TASK

    campaign = CampaignIdentity(
        task="tasks.multitenant_slo.task",
        workload_fingerprint=workload_fingerprint(TASK),
        load=load_record(TASK), gpu_kind=TASK.hardware.accelerator.kind,
        baseline=BaselineRecord(image=TASK.baseline.image,
                                serve_command=list(TASK.baseline.serve_command)))
    drives = [BaselineDrive(started_at=f"2026-09-05T00:{i:02d}:00+00:00",
                            duration_sec=2400.0, n_requests=1416, n_units=1416,
                            mean_log_credit=c)
              for i, c in enumerate(credits)]
    return credit_from_drives(campaign, drives)


def test_baseline_credit_injects_into_the_scorer(tmp_path):
    """With a matching baseline.json given, the scorer that grades the run
    carries the measured B -- and the authored task is left untouched, so nothing
    downstream can read a half-injected tree."""
    import json

    from hls.baseline_credit import credited
    from hls.core.poly import dump_spec
    from tasks.multitenant_slo.task import TASK

    bc = _baseline_credit()
    (tmp_path / "baseline.json").write_text(json.dumps(dump_spec(bc)))
    injected = credited(TASK, tmp_path / "baseline.json")[0]

    assert injected.scorer.baseline_log_credit == pytest.approx(0.5)
    assert injected.scorer.baseline_log_credit == bc.log_credit_mean
    assert TASK.scorer.baseline_log_credit is None


def test_baseline_credit_refuses_a_relabelled_slo(tmp_path):
    """The SLO budgets are the yardstick every latency is divided by, so re-pinning one
    invalidates B. They are deliberately absent from the workload fingerprint the
    calibration is keyed by, so this refusal is the baseline credit's own: without it a
    budget change would silently rescale every score."""
    import json

    from hls.baseline_credit import BaselineCreditError, credited
    from hls.core.poly import dump_spec
    from tasks.multitenant_slo.task import TASK

    bc = _baseline_credit()
    (tmp_path / "baseline.json").write_text(json.dumps(dump_spec(bc)))
    tenants = [t.model_copy(update={"workload": t.workload.model_copy(
                   update={"slo": StreamingLatencySLO(ttft_ms=t.workload.slo.ttft_ms * 2,
                                                      tpot_ms=t.workload.slo.tpot_ms)})})
               for t in TASK.effective_tenants()]
    relaxed = TASK.model_copy(update={"tenants": tenants})

    with pytest.raises(BaselineCreditError) as exc:
        credited(relaxed, tmp_path / "baseline.json")[0]
    message = str(exc.value)
    assert "different load" in message
    assert "chat.slo" in message


def _committed_baseline_credit():
    """The real baseline.json shipped as this task's default, read through the loader the CLI
    uses, freshness gate included. Reads the committed file rather than a fixture: a
    fixture cannot record a real campaign's numbers."""
    from hls.baseline_credit import read_baseline_credit
    from tasks.multitenant_slo.task import TASK

    return read_baseline_credit(_task_dir() / "baseline.json", TASK)[0]


def test_the_committed_baseline_credit_is_fresh_for_the_shipped_task():
    """The shipped pair loads end to end: the artifact's campaign identity still matches
    the task's workload fingerprint, load record and GPU kind, so B reaches the scorer.
    An edit to a prompt pool, an arrival process or an SLO budget flips this test before
    it can rescale a leaderboard."""
    bc = _committed_baseline_credit()

    assert bc.log_credit_mean == pytest.approx(0.5711810188013315, abs=5e-15)


def test_committed_baseline_credit_records_the_campaign_that_measured_it():
    """What B is a measurement of. The image is the name the task authors, not the
    compiled artifact one cluster happened to build, so a ledger measured anywhere
    pools with these drives; the serve command is the reference's operating point, and
    a flag added or dropped there is a different baseline under the same name."""
    from tasks.multitenant_slo.task import TASK

    bc = _committed_baseline_credit()

    assert bc.campaign.task == "tasks.multitenant_slo.task"
    assert bc.campaign.gpu_kind == "H200"
    assert bc.campaign.workload_fingerprint == (
        "8f94167ab6541b2e5e6cd00b5b718cb765d8ab3a3980cf3487cc547b70774f54")
    assert bc.campaign.baseline.image == "hls-baseline-multitenant-slo"
    assert bc.campaign.baseline.image == TASK.baseline.image
    assert bc.campaign.baseline.serve_command == list(TASK.baseline.serve_command)
    assert bc.campaign.baseline.serve_command == [
        "--served-model-name", "served-model",
        "--gpu-memory-utilization", "0.92",
        "--max-num-seqs", "512",
        "--dtype", "bfloat16",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--override-generation-config",
        '{"temperature":0.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0,'
        '"presence_penalty":0.0,"frequency_penalty":0.0}']


def test_committed_baseline_credit_carries_the_drives_it_averages():
    """The four drives of the baseline campaign, each a fresh deploy of the
    reference stack and one clocked serving of the graded workload. Pinned because B is
    a global rescale of every contestant's score: a rebuild that quietly moved one drive
    would otherwise reach grading unnoticed. The request counts differ by two across the
    drives, and only chat moves: it is the one tenant with a history budget, so a session
    there ends at the first turn whose conversation would exceed it, which depends on how
    long the server's own replies ran rather than on the seed. That is why a campaign
    pools drives on their identity rather than on their size."""
    bc = _committed_baseline_credit()

    assert [d.mean_log_credit for d in bc.drives] == [
        pytest.approx(0.5904062441397466, abs=5e-15),
        pytest.approx(0.5604503054446777, abs=5e-15),
        pytest.approx(0.567781560698718, abs=5e-15),
        pytest.approx(0.566085964922184, abs=5e-15)]
    assert [d.n_requests for d in bc.drives] == [1317, 1319, 1317, 1319]
    assert [round(d.duration_sec) for d in bc.drives] == [3357, 3148, 3082, 3451]
    assert [d.started_at[:10] for d in bc.drives] == ["2026-09-05"] * 4


def test_committed_baseline_credit_spread_is_locked_to_its_drives():
    """The two summaries a reader judges B by, recomputed from the drives stored beside
    them."""
    import math

    from hls.baseline_credit import credit_stats

    bc = _committed_baseline_credit()
    mean, sd, se = credit_stats([d.mean_log_credit for d in bc.drives])

    assert (bc.log_credit_mean, bc.log_credit_sd, bc.log_credit_se) == (mean, sd, se)
    assert bc.log_credit_sd == pytest.approx(0.01319435975304535, abs=5e-15)
    assert bc.log_credit_se == pytest.approx(0.006597179876522675, abs=5e-15)
    # The reference stack serves a typical request at 1.77x its SLO budget, and B is
    # pinned to within a factor of 1.0066 on every score it grades.
    assert math.exp(bc.log_credit_mean) == pytest.approx(1.7704, abs=5e-5)
    assert math.exp(bc.log_credit_se) == pytest.approx(1.0066, abs=5e-5)


def test_the_committed_baseline_credit_reaches_the_scorer():
    """The end-to-end wiring, exactly as `hls run` performs it: the shipped artifact is
    injected into the frozen task's scorer, and the authored task still carries no B."""
    from hls.baseline_credit import credited
    from tasks.multitenant_slo.task import TASK

    injected = credited(TASK, _task_dir() / "baseline.json")[0]

    assert injected.scorer.baseline_log_credit == _committed_baseline_credit().log_credit_mean
    assert TASK.scorer.baseline_log_credit is None


def test_slos_are_distinct():
    slos = {(t.workload.slo.ttft_ms, t.workload.slo.tpot_ms) for t in _tenants().values()}
    assert len(slos) > 1


def test_reasoning_has_train_to_dev_split_mapping():
    """reasoning is the only tenant with a real train/test pair: DEV samples
    openai/gsm8k train, TEST samples GSM8K-Platinum, wired via dev_dataset so the two
    corpora are sampled independently (disjoint by corpus identity)."""
    ds = _tenants()["reasoning"].data_source
    assert ds.dataset.path == "madrylab/gsm8k-platinum"
    assert ds.dev_dataset is not None and ds.dev_dataset.path == "openai/gsm8k"
    assert ds.dev_dataset.source_split == "train"


def test_reasoning_8shot_prefix_in_prompt_template():
    """The reasoning prompt carries the 8-shot CoT prefix (README prefix layer), and
    the SAME template is used for the platinum test corpus and the openai/gsm8k dev
    corpus (identical prefix for oracle + artifact + dev/test)."""
    from tasks.multitenant_slo.prompts import GSM8K_8SHOT

    ds = _tenants()["reasoning"].data_source
    assert GSM8K_8SHOT in ds.dataset.prompt.template
    assert ds.dataset.prompt.template == ds.dev_dataset.prompt.template


def test_chat_pool_is_the_calibrations_draw_pool():
    """The two authored subsamples are the pools the calibration's fresh draws were taken
    from, and the split is a seeded subsample of the pool rather than all of it. That is
    what makes the graded corpus a draw from the calibrated distribution -- shrinking a
    pool to dev+test would make the graded split a stream PREFIX instead, which is a
    different sample of the same corpus. dev+test must still fit inside the pool or
    provisioning's partition guard fails at provision time, after the 3.4 GB download."""
    from hls.spec.datasets import WildChatDataset

    ds = _tenants()["chat"].data_source.dataset
    assert isinstance(ds, WildChatDataset)
    assert ds.subsample_size == _POOLS["chat"]
    assert _DEV_COUNTS["chat"] + _TEST_COUNTS["chat"] <= ds.subsample_size
    assert (ds.language, ds.max_turns) == ("English", 6)


def test_the_removal_lists_are_the_committed_ones():
    """Each tenant's ``removed_rows`` is the list removed_requests.json carries for it,
    named by the tenant's own row ids (WildChat conversation hashes, BFCL test ids,
    MRCR and GSM8K-Platinum source positions), sorted, in the counts the file records;
    and the lists are part of the workload fingerprint."""
    import re

    from hls.verify.calibration import workload_fingerprint
    from tasks.multitenant_slo.removed_requests import removed_rows
    from tasks.multitenant_slo.task import TASK

    tenants = _tenants()
    expected = {"chat": (r"[0-9a-f]{32}", 36),
                "api_short": (r"(live_)?(simple|multiple|parallel|parallel_multiple)_\d+(-\d+-\d+)?", 333),
                "long_session": (r"\d+", 6),
                "reasoning": (r"\d+", 1)}
    for name, (pattern, count) in expected.items():
        removed = tenants[name].data_source.removed_rows
        assert removed == removed_rows(name) and len(removed) == count, name
        assert list(removed) == sorted(set(removed)), name
        assert all(re.fullmatch(pattern, r) for r in removed), name
    # Dropping the lists is a different workload, so a different calibration.
    bare = TASK.model_copy(update={"tenants": [
        t.model_copy(update={"data_source": t.data_source.model_copy(
            update={"removed_rows": ()})}) for t in TASK.tenants]})
    assert workload_fingerprint(bare) != workload_fingerprint(TASK)


def test_the_workload_fingerprint_is_the_calibrated_one():
    """The workload the shipped calibration was computed for: the fingerprint is
    exactly the committed value."""
    from hls.verify.calibration import workload_fingerprint
    from tasks.multitenant_slo.task import TASK

    assert workload_fingerprint(TASK) == (
        "8f94167ab6541b2e5e6cd00b5b718cb765d8ab3a3980cf3487cc547b70774f54")


def test_long_session_replays_scripted_history():
    """The long_session tenant is MRCR in cumulative-replay mode (one request per
    scripted user turn); without the flag it would collapse back to one graded
    request per conversation and lose ~91% of its request budget."""
    from hls.spec.datasets import MRCRDataset

    ds = _tenants()["long_session"].data_source.dataset
    assert isinstance(ds, MRCRDataset) and ds.replay is True


def test_multi_turn_tenants_use_gamma_think_time_and_trim_rampdown():
    """Both multi-turn tenants (chat, long_session) wire GammaInterTurn with
    Bailian-grounded params and trim_rampdown=True; single-turn tenants keep NoDelay."""
    from hls.spec.workload import GammaInterTurn, NoDelay

    tenants = _tenants()
    for name in ("chat", "long_session"):
        d = tenants[name].workload.inter_turn_delay
        assert isinstance(d, GammaInterTurn), f"{name}: expected GammaInterTurn, got {type(d)}"
        assert d.shape > 0 and d.scale > 0 and d.max_gap > 0
        assert tenants[name].workload.trim_rampdown is True
    # single-turn tenants keep NoDelay and do not need trimming
    for name in ("api_short", "reasoning"):
        assert isinstance(tenants[name].workload.inter_turn_delay, NoDelay)


def test_long_session_walk_time_within_ramp_budget():
    """The pinned long_session truncation keeps the deepest walk D_max =
    (turns-1)*mean_gap within one arrival span (<= 0.7*W), so an early-arriving
    session completes its full walk. trim_rampdown guards the tail directly, so
    this is a depth/CV bound, not a tail-length guard."""
    import statistics

    from tasks.multitenant_slo.task import TOTAL_RATE, _MRCR_TURNS, _TEST_COUNTS

    d = _tenants()["long_session"].workload.inter_turn_delay
    xs = [d.delay(1, seed=(s, 0)) for s in range(50_000)]
    mean_gap = statistics.mean(xs)
    d_max = (_MRCR_TURNS - 1) * mean_gap
    # W ~= total test requests / TOTAL_RATE (chat contributes count*turns).
    total_requests = (_TEST_COUNTS["chat"] * 2.3914
                      + _TEST_COUNTS["api_short"]
                      + _TEST_COUNTS["reasoning"]
                      + _TEST_COUNTS["long_session"] * _MRCR_TURNS)
    W = total_requests / TOTAL_RATE
    assert d_max <= 0.7 * W   # D_max ~= 970.4 s vs 0.7*W ~= 1,051 s at this mix


def test_baseline_decodes_greedily():
    """The declared baseline must pin greedy decoding explicitly.

    This checkpoint's generation_config.json ships do_sample=true / temperature=1.0 /
    top_k=20 / top_p=0.95, and `vllm serve` applies those for any sampling field a request
    omits -- which is all of them, since the harness sends only messages and max_tokens.
    Without an override the baseline samples, and it flips far more positions
    than the calibrated ceiling allows. The brief
    demands greedy decoding of contestants, so a sampling baseline fails the task's own
    rule and is not a comparator.

    The pinned values must equal hls.reference.engines._GREEDY_SAMPLING, so the baseline
    decodes the same way the pool members that produced the ceiling did.
    """
    import json

    from hls.reference.engines import _GREEDY_SAMPLING
    from tasks.multitenant_slo.task import TASK

    cmd = TASK.baseline.serve_command
    assert "--override-generation-config" in cmd, (
        "baseline must pin greedy decoding; without it vllm serve inherits the "
        "checkpoint's sampling defaults")
    pinned = json.loads(cmd[cmd.index("--override-generation-config") + 1])
    assert pinned == dict(_GREEDY_SAMPLING), (
        f"baseline sampling pin {pinned} differs from the calibrated pool's "
        f"{dict(_GREEDY_SAMPLING)}; the two must decode identically")

def test_the_reasoning_prefix_is_published_and_is_not_traffic():
    """The one prompt element a contestant cannot infer from its rows is handed over.

    reasoning's dev rows are a bare question and answer, so nothing in them shows that a
    served prompt carries 715 tokens of few-shot exemplars first -- a contestant sizing
    prefill from a row would measure a tenth of the real prompt. Publishing it must not
    disturb the calibration: the prefix is already inside the prompt template, which the
    workload fingerprint hashes, so the published copy is disclosure and nothing else.
    """
    import json
    from pathlib import Path

    from hls.verify.calibration import workload_fingerprint
    from tasks.multitenant_slo.prompts import GSM8K_8SHOT
    from tasks.multitenant_slo.task import TASK

    reasoning = next(t for t in TASK.effective_tenants() if t.id == "reasoning")
    # Declared once on the source, beside the dev split it is published to, not once per
    # corpus the source draws from.
    assert reasoning.data_source.dev_reference == {"prompt_prefix.txt": GSM8K_8SHOT}

    # The other tenants' prompts are the corpus text, so they publish nothing.
    for tenant in TASK.effective_tenants():
        if tenant.id == "reasoning":
            continue
        assert not tenant.data_source.dev_reference

    stored = json.loads(
        (Path(__file__).parents[2] / "tasks/multitenant_slo/defaults/calibration.json").read_text()
    )["workload_fingerprint"]
    assert workload_fingerprint(TASK) == stored

