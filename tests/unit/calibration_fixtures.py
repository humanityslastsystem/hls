"""Schema-valid calibration artifacts for tests, built from compact per-engine inputs.

Every test that needs a calibration -- the loader's staleness gates, the verifier's two
rules, the CLI's injection path, the task-wiring checks -- needs the same handful of
internally consistent sections, and the model's lock validators refuse anything else
(``q_max`` to the worst engine's mean, ``lambda_hat`` to events over exposure,
``lambda_up`` to the exact Garwood limit). Building them here once means a test states
only what it is actually testing.
"""

from __future__ import annotations

import hashlib
import json

from hls.calibrate import DEFAULT_DELTA, DEFAULT_EPS, DEFAULT_MU_PER_MTOK
from hls.verify.calibration import (
    DEFAULT_EOT_EXPECTED_MAX,
    UPPER_Z,
    Calibration,
    CappedText,
    CurveSide,
    DrawSpec,
    EndOfTurnCheck,
    EndOfTurnEngine,
    EngineCurve,
    EngineRate,
    EngineRecord,
    EngineReport,
    LevelStats,
    PreflightResult,
    ReportCalibration,
    Sizing,
    StackInfo,
    StopReason,
    Tolerances,
    workload_fingerprint,
    workload_payload,
)
from hls.verify.poisson import garwood_upper

#: The severity grid the fixtures are built on: what a pool whose worst clean gap
#: is 4.0 nats gets (hls.verify.flips.severity_grid), minus the 3.5 rung, which
#: the fixtures' hand-written event counts leave out.
LEVELS_NATS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)

#: Default pool: one engine, three draws at a 0.4% flip rate over 100k tokens each, with
#: a handful of events on the lowest gap levels. Small enough to read.
_DEFAULT_RATES = [0.0039, 0.0040, 0.0041]
_DEFAULT_TOKENS = [100_000, 100_000, 100_000]
_DEFAULT_EVENTS = [6, 2, 1, 0, 0, 0, 0]


def engine_rate(rates: list[float] | None = None, tokens: list[int] | None = None,
                max_gap: float = 0.5) -> EngineRate:
    """One engine's rate side from its per-draw rates and gradable lengths."""
    rates = _DEFAULT_RATES if rates is None else rates
    tokens = (_DEFAULT_TOKENS if tokens is None else tokens)[:len(rates)]
    mean = sum(rates) / len(rates)
    sd = (sum((r - mean) ** 2 for r in rates) / (len(rates) - 1)) ** 0.5 if len(rates) > 1 else 0.0
    return EngineRate(mean=mean, sd=sd, draws=len(rates), rates=rates, tokens=tokens,
                      max_gap=max_gap)


def engine_curve(events: list[int] | None = None, exposure: int = 300_000) -> EngineCurve:
    """One engine's curve side from its per-level event counts over a pooled exposure."""
    events = _DEFAULT_EVENTS if events is None else events
    return EngineCurve(exposure=exposure, levels=[
        LevelStats(level_nats=c, events=e, lambda_hat=e / exposure,
                   lambda_up=garwood_upper(e, exposure))
        for c, e in zip(LEVELS_NATS, events, strict=True)])


#: The workload a fixture calibration describes when no TaskSpec is supplied. The model
#: validates that the persisted copy hashes to the recorded fingerprint, so the two must
#: be produced together -- which is why a hand-rolled stub cannot be constructed.
_FIXTURE_WORKLOAD = {"probe": "fixture-workload"}
_FIXTURE_FINGERPRINT = hashlib.sha256(
    json.dumps(_FIXTURE_WORKLOAD, sort_keys=True).encode()).hexdigest()

#: The end-of-turn id a fixture checkpoint declares. It matches the one the sampling
#: gate's own tests serve, so a fixture calibration describes the traces they build.
_FIXTURE_EOT_ID = 9


def fixture_workload(**fields) -> dict:
    """The fixture workload payload and its fingerprint, as override keyword arguments.

    For a test that needs one declared field of the payload -- ``max_context_len``, say,
    which the sampling gate reads a reply's ending against -- without standing up a whole
    TaskSpec. The two are produced together because the model refuses a copy that does
    not hash to its recorded fingerprint. A field set to ``None`` is dropped, which is
    how the payload spells a task that declares nothing there."""
    payload = {**_FIXTURE_WORKLOAD,
               **{k: v for k, v in fields.items() if v is not None}}
    return {"workload": payload,
            "workload_fingerprint": hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def make_calibration(taskspec=None, *, engines: dict[str, dict] | None = None,
                     **overrides) -> Calibration:
    """A valid ``Calibration`` for the ``TaskSpec`` ``taskspec``, or for a fixture
    workload when it is ``None`` (which is what tests exercising the verifier rather
    than the loader want). The parameter is not called ``task`` because the artifact
    has a ``task`` FIELD -- the dotted module path -- that callers override.

    ``engines`` maps engine name -> keyword arguments for ``engine_rate`` and
    ``engine_curve`` (``rates``, ``tokens``, ``max_gap``, ``events``, ``exposure``);
    the default is a single ``hf_pinned`` engine. Any other field can be overridden
    directly, which is how the lock validators are exercised.

    The trace floor is NOT here: it is a function of the deployed headroom, so it lives
    on the verifier (``TokenFaithfulnessVerifier.min_tokens``)."""
    engines = engines or {"hf_pinned": {}}
    rate_side = {
        name: engine_rate(**{k: v for k, v in spec.items()
                             if k in ("rates", "tokens", "max_gap")})
        for name, spec in engines.items()}
    curve_side = CurveSide(
        levels_nats=list(LEVELS_NATS),
        engines={name: engine_curve(**{k: v for k, v in spec.items()
                                       if k in ("events", "exposure")})
                 for name, spec in engines.items()})
    tol = Tolerances(eps=DEFAULT_EPS, mu_per_mtok=DEFAULT_MU_PER_MTOK,
                     delta=DEFAULT_DELTA)
    q_max = max(r.mean for r in rate_side.values())
    kwargs = dict(
        task="tests.support.token_level_task",
        workload_fingerprint=(workload_fingerprint(taskspec) if taskspec is not None
                              else _FIXTURE_FINGERPRINT),
        workload=(workload_payload(taskspec) if taskspec is not None
                  else _FIXTURE_WORKLOAD),
        gpu_kind="B200",
        stack=StackInfo(torch="2.4.0", runtime={"cuda": "12.4", "cudnn": "90100"},
                        device_name="B200"),
        preflight=PreflightResult(passed=True, flip_rate_vs_fp32=0.001,
                                  max_gap_vs_fp32=0.05, n_tokens=1000),
        engines={name: EngineRecord(
                     version="0.0.0",
                     config="live drive: --served-model-name served-model")
                 for name in rate_side},
        draw=DrawSpec(max_sessions=None,
                      # One seed per serving, per engine -- the model locks the two
                      # counts together, so this follows whatever `engines` asked for.
                      serving_seeds={name: [11, 22, 33][:r.draws]
                                     for name, r in rate_side.items()}),
        tolerances=tol,
        rate_side=rate_side,
        curve_side=curve_side,
        sizing=Sizing(c_m=1.5, stopped=StopReason.TARGET_REACHED),
        capped_text=CappedText(events=0, tokens=0),
        q_max=q_max,
        created_at="2026-07-24T00:00:00+00:00")
    kwargs.update(overrides)
    return Calibration(**kwargs)


def engine_report(mean_kl: float = 0.02, se_kl: float = 0.0005, n_servings: int = 1,
                  n_replies: int = 400, n_tokens: int = 300_000,
                  outside_rate: float = 0.1, se_outside: float = 0.002) -> EngineReport:
    """One pool engine's measured report divergence and served-outside rate, with both
    upper limits derived the way the model locks them.

    The default rate is the tail mass of the sampling gate's own fixture report (0.1),
    so a trace drawn from that report sits at the calibrated rate rather than above
    it."""
    return EngineReport(n_servings=n_servings, n_replies=n_replies, n_tokens=n_tokens,
                        mean_kl=mean_kl, se_kl=se_kl,
                        upper=mean_kl + UPPER_Z * se_kl,
                        precision_passed=se_kl <= DEFAULT_EPS * mean_kl,
                        outside_rate=outside_rate, se_outside=se_outside,
                        outside_upper=outside_rate + UPPER_Z * se_outside)


def end_of_turn_check(names, *, eps: float = 0.1) -> EndOfTurnCheck:
    """The end-of-turn section for the engines ``names``, each sitting comfortably
    inside its null: about the expected count, so the p-value is of order one half and
    no fixture is near the check's own rejection."""
    return EndOfTurnCheck(
        ids=[_FIXTURE_EOT_ID], eps=eps, expected_max=DEFAULT_EOT_EXPECTED_MAX,
        engines={name: EndOfTurnEngine(priced_positions=300_000, expected=4.0,
                                       observed=4, p_value=0.37)
                 for name in names})


def make_report_calibration(taskspec=None, *, engines: dict[str, dict] | None = None,
                            **overrides) -> ReportCalibration:
    """A valid ``ReportCalibration`` for ``taskspec``, or for the fixture workload when
    it is ``None``.

    ``engines`` maps engine name -> keyword arguments for ``engine_report``; the default
    is a single ``vllm`` engine whose ceiling works out to a readable number, so a test
    can state only what it is testing. Any field can be overridden directly, which is
    how the lock validators are exercised."""
    engines = engines or {"vllm": {}}
    measured = {name: engine_report(**spec) for name, spec in engines.items()}
    kwargs = dict(
        task="tests.support.token_level_task",
        workload_fingerprint=(workload_fingerprint(taskspec) if taskspec is not None
                              else _FIXTURE_FINGERPRINT),
        workload=(workload_payload(taskspec) if taskspec is not None
                  else _FIXTURE_WORKLOAD),
        gpu_kind="B200",
        stack=StackInfo(torch="2.4.0", runtime={"cuda": "12.4", "cudnn": "90100"},
                        device_name="B200"),
        engines=measured,
        kappa=max(e.upper for e in measured.values()),
        outside_ceiling=max(e.outside_upper for e in measured.values()),
        eps=DEFAULT_EPS,
        end_of_turn=end_of_turn_check(measured),
        created_at="2026-09-12T00:00:00+00:00")
    kwargs.update(overrides)
    return ReportCalibration(**kwargs)
