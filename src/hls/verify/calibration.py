"""Calibration: the data models, workload fingerprint, and load/inject layer for
calibration-requiring verifiers.

A calibration-requiring verifier's measured ceilings are not authored by hand --
they come from an offline calibration run (``uv run hls calibrate <module>``)
that profiles several independent implementations of the same reference weights
against the pinned reference, and writes one ``calibration.json`` into the store
(``hls.measured``).

There are two such gates and so two artifacts, distinguished by the file's ``kind``
tag. ``Calibration`` (``"flips"``) carries both of the token-level gate's rules: the
rate ceiling (``q_max``, from ``rate_side``) and the per-gap-level flip-count ceilings
(``curve_side``), plus the tolerances they were sized against, the sizing verdicts, and
the trace-length floor the deployed gate fails closed below. ``ReportCalibration``
(``"report"``) carries the sampling gate's single measured number, ``kappa``: the
divergence a trusted stack's own reported distributions already show against the
reference. It also carries the one threshold that gate's end-of-turn count needs, which
is chosen rather than measured (``EndOfTurnCheck``). A
verifier names which it reads (``Verifier.calibration_kind``) and the loader refuses a
file of the other kind.

This module is the fail-loud bridge between that file and the TaskSpec the
verifier runs inside: ``workload_fingerprint`` identifies which traffic
distribution a calibration was computed for, ``read_calibration`` refuses a
stale, GPU-mismatched, or preflight-failed calibration outright, and
``calibrated`` threads the loaded ``Calibration`` into the (frozen)
TaskSpec's calibration-requiring verifiers without mutating the original tree.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, ValidationError, model_validator

from hls.core.poly import Spec, dump_spec
from hls.measured import CALIBRATION_FILE
from hls.spec.resources import HardwareRecord
from hls.verify.poisson import GARWOOD_ALPHA, flip_count_threshold, garwood_upper
from hls.verify.report import EOT_EPS_GRID

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec

class CalibrationError(Exception):
    """Raised whenever a calibration-requiring task cannot safely proceed: no
    calibration file, a stale (fingerprint-mismatched) one, one computed on a
    different GPU kind, or one whose own offline preflight failed. Every raise
    site names the fix (``uv run hls calibrate <module>``) so a user hitting
    this never has to guess what to run."""


class StackInfo(Spec):
    """The software stack a calibration run was computed on -- recorded for audit
    and compared verbatim by the verifier's stack-drift refusal (a reference whose
    stack differs from the calibration's cannot reuse its noise floor).

    Vendor-neutral by construction: ``runtime`` maps accelerator-stack component
    names to versions, so the components an NVIDIA stack reports (``{"cuda": ...,
    "cudnn": ...}``) and the ones another vendor would (e.g. ``{"hip": ...,
    "miopen": ...}``) live under the same schema -- supporting a new vendor
    changes what the engine's ``stack_report`` emits, never this model.
    ``device_name`` is descriptive only; ``Calibration.gpu_kind`` is the field
    ``read_calibration`` actually gates on."""

    torch: str
    runtime: dict[str, str]
    device_name: str


class PreflightResult(Spec):
    """The calibration run's instrument check, run before any pool member is
    profiled (hence "preflight"): an independent fp32 anchor of the same
    checkpoint generates probe replies and the pinned bf16 reference
    teacher-forces them. Large disagreement means the reference instrument
    itself is numerically broken, so the member measurements that would follow
    are meaningless -- the calibration's instrument stage refuses before any
    serving runs (``hls.calibrate.check_instrument``), and ``read_calibration``
    refuses a persisted calibration whose ``passed=False``."""

    passed: bool
    flip_rate_vs_fp32: float
    max_gap_vs_fp32: float
    n_tokens: int


class DrawSpec(Spec):
    """How much of the task's traffic one calibration draw served.

    A draw is one serving: one live drive of the task's own traffic generator --
    clocked arrivals, think-times, ramp-down trimming, live self-accumulated chat
    histories -- against a pool engine running as a continuous-batching server,
    exactly as a graded run is driven. Each serving is clocked with its own seed,
    so arrivals, think-times, and turn membership differ between servings while
    the workload identity (``Calibration.workload_fingerprint``) does not.

    ``serving_seeds`` is keyed per engine rather than pooled because a seed is a
    function of the serving index alone -- two engines' serving 3 share a seed -- while
    the engines need not have served the same number of them."""

    #: Sessions one serving was asked to drive; None means the split's whole drawn
    #: workload.
    max_sessions: int | None
    #: Engine -> the clock seed of each of its servings, in serving order, so any named
    #: serving can be re-driven exactly.
    serving_seeds: dict[str, tuple[int, ...]]


class Tolerances(Spec):
    """The statistical demands the campaign was sized against, recorded so a verdict
    always says which tolerances produced it.

    ``mu_per_mtok`` is per token, not per run: a longer graded trace receives a
    proportionally larger allowance, so the exposure the campaign must reach
    (``hls.calibrate.exposure_target``) is a constant independent of the task and of
    the trace length.

    The rate headroom is absent by design: it multiplies ``q_max`` into a deployed
    threshold after calibration finishes and plays no part in sizing, so it belongs to
    the verifier (``TokenFaithfulnessVerifier.rate_multiplier``), not here. Keeping it
    out is also what lets several verifiers grade against one calibration at different
    headrooms."""

    #: (P): the largest relative spread one graded run's flip rate may have
    #: (dimensionless fraction of the mean rate).
    eps: float
    #: (T): expected flips per million graded tokens tolerated at gap levels the
    #: calibration never observed.
    mu_per_mtok: float
    #: Family-wise false-alarm probability per graded trace, shared across the rungs.
    delta: float


class EngineRate(Spec):
    """One engine's flip rate across the campaign's servings.

    Every serving's rate and length is kept, so ``mean``, ``sd`` and the (P) check are
    recomputable from the file rather than trusted, and a reader can re-derive the
    length the deployed ceilings were quoted at. ``max_gap`` is audit only: the
    deployed gate bounds how many large-gap flips a trace carries (``CurveSide``), not
    how large one may be, because the clean pool itself produces rare multi-nat
    events."""

    #: Mean of ``rates``: the engine's flip rate, flips per teacher-forced token.
    mean: float
    #: Sample standard deviation of ``rates``, same unit; 0.0 for a single serving.
    sd: float
    #: Servings measured; equals ``len(rates)``.
    draws: int
    #: Each serving's flip rate, flips per teacher-forced token, in serving order.
    rates: list[float]
    #: Each serving's teacher-forced token count, in serving order.
    tokens: list[int]
    #: Worst single-position log-probability gap anywhere in the campaign, in nats.
    max_gap: float

    @model_validator(mode="after")
    def _check_derived(self) -> "EngineRate":
        if not (self.draws == len(self.rates) == len(self.tokens)):
            raise CalibrationError(
                f"EngineRate says {self.draws} servings but carries {len(self.rates)} "
                f"rates and {len(self.tokens)} serving lengths")
        if not self.rates:
            raise CalibrationError("EngineRate carries no servings")
        want_mean = sum(self.rates) / len(self.rates)
        if not math.isclose(self.mean, want_mean, rel_tol=1e-12):
            raise CalibrationError(
                f"EngineRate mean={self.mean} contradicts its own servings ({want_mean})")
        want_sd = statistics.stdev(self.rates) if len(self.rates) > 1 else 0.0
        if not math.isclose(self.sd, want_sd, rel_tol=1e-12, abs_tol=1e-15):
            raise CalibrationError(
                f"EngineRate sd={self.sd} contradicts its own servings ({want_sd})")
        return self


class LevelStats(Spec):
    """One engine's clean-event statistics at one gap level.

    The deployed ceiling is built from ``lambda_up``, an upper bound, so a level the
    campaign happened to see zero events at still gets a finite ceiling."""

    #: The gap level, in nats: this row counts flips whose gap is at least this.
    level_nats: float
    #: Flips with gap >= ``level_nats`` over the engine's whole exposure.
    events: int
    #: ``events / exposure``: the point estimate of the per-token event rate.
    lambda_hat: float
    #: Garwood 97.5% upper limit on the per-token event rate.
    lambda_up: float


class EngineCurve(Spec):
    """One engine's clean flip-count envelope: its exposure and per-level event
    rates."""

    #: Pooled teacher-forced tokens over every reply of the engine's servings.
    exposure: int
    #: One row per level of ``CurveSide.levels_nats``, ascending in ``level_nats`` and
    #: non-increasing in ``events`` (a gap over 2.0 nats is also over 1.0).
    levels: list[LevelStats]


class SizingCheck(Spec):
    """One sizing inequality as evaluated: both sides in their own units, and whether
    it held. Stored per engine, because each engine's spread and exposure differ and
    the binding one is what certifies the campaign."""

    #: Left side of the inequality: a flip rate for (P), a token count for (T).
    lhs: float
    #: Right side, in the same unit as ``lhs``.
    rhs: float
    #: Whether the inequality held: ``lhs <= rhs`` for (P), ``lhs >= rhs`` for (T).
    passed: bool


class StopReason(StrEnum):
    """Why the campaign stopped taking servings. A partial campaign must never read as
    a complete one, so the reason is a closed vocabulary rather than free text."""

    #: Every engine reached the exposure target and the serving floor.
    TARGET_REACHED = "exposure target reached"
    #: The GPU-hour budget ran out first, so the exposure is whatever it bought.
    BUDGET_EXHAUSTED = "budget exhausted"


def exposure_target(mu_per_mtok: float) -> float:
    """The teacher-forced exposure, in tokens, one engine must reach: ``ln(1/alpha)``
    over the per-token tail tolerance (T).

    ``ln(1/alpha)`` is the Garwood upper limit on a rate that produced zero events,
    expressed per unit exposure (the same bound the curve rule uses at ``e = 0``). So
    the identity reads: even the worst per-token rate consistent with having observed
    nothing at a gap level puts at most ``mu_per_mtok`` such flips in a million graded
    tokens. Because both sides are per token, the target is a constant -- independent of
    the task and of how long a graded trace runs -- about 12.3M tokens per engine at the
    default tolerance."""
    if not mu_per_mtok > 0:
        raise CalibrationError(f"mu_per_mtok must be > 0, got {mu_per_mtok}")
    return math.log(1.0 / GARWOOD_ALPHA) / (mu_per_mtok / 1e6)


def precision_check(rate: "EngineRate", c_m: float, eps: float) -> SizingCheck:
    """(P) for one engine: ``c_m * sd <= eps * mean``, both sides flip rates."""
    lhs = c_m * rate.sd
    rhs = eps * rate.mean
    return SizingCheck(lhs=lhs, rhs=rhs, passed=lhs <= rhs)


def tail_check(exposure: int, mu_per_mtok: float) -> SizingCheck:
    """(T) for one engine: pooled exposure against the target, both in tokens."""
    rhs = exposure_target(mu_per_mtok)
    return SizingCheck(lhs=float(exposure), rhs=rhs, passed=exposure >= rhs)


class Sizing(Spec):
    """Whether this campaign's exposure and serving count can carry the ceilings.

    Only the two facts a reader cannot recompute are stored. Both verdicts are pure
    functions of numbers the artifact already carries, and ``Calibration`` exposes them
    as ``precision`` and ``tail``.

    (T) is what sizes the campaign: pooled teacher-forced exposure against the constant
    target ``ln(1/alpha) / mu_per_mtok`` tokens per engine (``exposure_target``). Enough
    exposure that a flip at a gap level the campaign never observed is expected at most
    ``mu_per_mtok`` times per million graded tokens. It is a token counter against a
    constant -- no statistics, no looks -- so the realized serving count is independent
    of the flip-rate noise.

    (P) is ``c_m * sd <= eps * mean`` per engine: one graded run's rate is reproducible
    to ``eps``. Because the stop above pre-commits the sample size, this is a single
    look at the realized servings rather than a sequential test."""

    #: Chi-square upper-confidence factor on the spread estimate,
    #: ``sqrt((m-1) / chi2_{m-1, alpha})`` at the pool's smallest serving count ``m``,
    #: which is why it is stored rather than derived per engine; dimensionless,
    #: multiplies each engine's ``sd`` in (P). It approaches 1 as servings grow, which
    #: is the only way more servings help -- the spread itself does not shrink with them.
    c_m: float
    #: Why the campaign ended.
    stopped: StopReason


class CurveSide(Spec):
    """The flip-count curve rule's parameters: the gap-level grid, the false-alarm
    budget split across its rungs, and each engine's measured clean envelope.

    The model: at each gap level ``c``, flips with gap >= c arrive as one Poisson
    process per teacher-forced token, measured per reference-class engine. Pooling
    servings into one process treats severe flips as not recurring across servings
    of the same prompt -- they are induced by serving numerics (batch composition,
    reduction order), not by prompt identity. Calibration takes a recurrence census
    and reports any (prompt, gap level) site that fires in half or more of an
    engine's servings (``hls.calibrate.pool_warnings``).

    A trace of ``N`` teacher-forced tokens is rejected at level ``c`` when its count of
    flips with gap >= c exceeds the verifier's tolerance multiple of ``k(c, N)`` -- a
    nonparametric stochastic-dominance test of the trace's flip-gap exceedance curve
    against the clean envelope. Grading is against
    the most permissive engine at each level: the ceiling is what some clean stack
    produces, not what all of them do, so a contestant is never failed for behaving
    like one of the pool's own members."""

    #: The severity grid, in nats, strictly ascending: half-nat lattice rungs from 0.5,
    #: kept where the pool's binding clean count strictly drops plus the first rung no
    #: clean engine reached (``hls.calibrate.pool_grid``); a measurement of the pool,
    #: not an authored list.
    levels_nats: list[float]
    #: Engine -> its measured clean envelope.
    engines: dict[str, EngineCurve]

    @property
    def n_rungs(self) -> int:
        """Rungs the family false-alarm budget is split over: one per gap level plus
        one for the rate rule. Every rule that can reject a clean trace draws from the
        same budget."""
        return len(self.levels_nats) + 1

    def thresholds(self, n_tokens: int, delta_per_rung: float) -> list[dict]:
        """The per-level rejection thresholds for a trace of ``n_tokens`` pooled
        teacher-forced tokens, one row per level of ``levels_nats``.

        Each row carries the operative ``k`` (the max over engines, per the grading
        convention) plus the binding engine's parameters for diagnostics:
        ``{"c", "k", "k_per_engine", "engine", "e", "lambda_up", "mu"}``, where
        ``mu = lambda_up * n_tokens`` is that engine's expected clean count in the
        trace. Ties on k break toward the larger ``mu`` (the more permissive process),
        then engine name, so the reported parameters are deterministic.
        ``delta_per_rung`` is the Bonferroni share the counts are sized at; it is the
        same for every row, so it is an argument rather than a column.

        These are the calibrated ceilings. A verifier grading at a tolerance multiplier
        over them applies it to every entry of ``k_per_engine``, and floor is monotone,
        so the max-over-engines relation survives; the rows returned here are freshly
        built each call and belong to the caller.
        """
        delta = delta_per_rung
        rows: list[dict] = []
        for i, c in enumerate(self.levels_nats):
            per_engine = {
                name: flip_count_threshold(eng.levels[i].lambda_up, n_tokens, delta)
                for name, eng in sorted(self.engines.items())
            }
            engine = max(per_engine,
                         key=lambda n: (per_engine[n], self.engines[n].levels[i].lambda_up))
            lvl = self.engines[engine].levels[i]
            rows.append({
                "c": c,
                "k": per_engine[engine],
                "k_per_engine": per_engine,
                "engine": engine,
                "e": lvl.events,
                "lambda_up": lvl.lambda_up,
                "mu": lvl.lambda_up * n_tokens,
            })
        return rows

    @model_validator(mode="after")
    def _check_grid(self) -> "CurveSide":
        if not self.engines:
            raise CalibrationError("curve_side carries no engines; nothing to grade against")
        if (not self.levels_nats or any(c <= 0 for c in self.levels_nats)
                or sorted(set(self.levels_nats)) != self.levels_nats):
            raise CalibrationError(
                f"curve_side levels_nats must be a strictly ascending positive grid, "
                f"got {self.levels_nats}")
        return self

    @model_validator(mode="after")
    def _check_engines_locked(self) -> "CurveSide":
        # Every stored number that decides pass/fail is a pure function of the counts
        # stored beside it, so lock them: lambda_hat to e/T, lambda_up to the exact
        # Garwood bound. A hand-edit that loosens a ceiling is then unrepresentable.
        for name, eng in self.engines.items():
            if eng.exposure <= 0:
                raise CalibrationError(f"curve_side engine {name!r} has exposure <= 0")
            grid = [lvl.level_nats for lvl in eng.levels]
            if grid != self.levels_nats:
                raise CalibrationError(
                    f"curve_side engine {name!r} levels {grid} do not match the declared "
                    f"grid {self.levels_nats}")
            for lvl in eng.levels:
                if lvl.events < 0:
                    raise CalibrationError(
                        f"curve_side engine {name!r} level {lvl.level_nats} nats has "
                        f"events < 0")
                want_hat = lvl.events / eng.exposure
                if lvl.lambda_hat != want_hat:
                    raise CalibrationError(
                        f"curve_side engine {name!r} level {lvl.level_nats} nats: "
                        f"lambda_hat={lvl.lambda_hat} contradicts events/exposure = "
                        f"{want_hat}; re-run hls calibrate")
                want_up = garwood_upper(lvl.events, eng.exposure)
                if not math.isclose(lvl.lambda_up, want_up, rel_tol=1e-9):
                    raise CalibrationError(
                        f"curve_side engine {name!r} level {lvl.level_nats} nats: "
                        f"lambda_up={lvl.lambda_up} is not the Garwood 97.5% upper limit "
                        f"for events={lvl.events}, exposure={eng.exposure} (that is "
                        f"{want_up}); re-run hls calibrate rather than hand-editing the "
                        f"file")
            for hi, lo in zip(eng.levels, eng.levels[1:]):
                if lo.events > hi.events:
                    raise CalibrationError(
                        f"curve_side engine {name!r}: event counts must be non-increasing "
                        f"in the level (gap >= level is nested), but {lo.level_nats} nats "
                        f"has {lo.events} events vs {hi.events} at {hi.level_nats}")
        return self


class EngineRecord(Spec):
    """One pool member as it was profiled: which build served, and how.

    The ceilings are claims about this exact member, so both halves matter -- a
    different serving-package version and a different serve flag are each a different
    noise source. Audit only; nothing gates on it."""

    #: Version of the serving package (``vllm``/``sglang``) as the member's own serving
    #: process reported it, so the record names the build that produced the captures
    #: rather than whatever the host happens to have installed.
    version: str
    #: The engine class and the non-default settings that make it a distinct noise
    #: source, as one auditable line.
    config: str


class CappedText(Spec):
    """How much of what the campaign measured came from replies whose served length
    reached the request's output cap, pooled over every serving of every engine.

    Descriptive, not an exclusion: these replies are inside ``rate_side`` and
    ``curve_side`` like any other. A task that wants them out of the measurement
    removes their requests from the workload itself (``DataSource.removed_rows``,
    derived from a campaign's captures by ``hls.calibrate.capped_rows``), so a
    campaign on that workload measures none. Recorded so a reader can compare a graded
    trace's own capped share (the verifier's ``n_capped_tokens`` over its graded tokens)
    against the campaign's (``tokens`` over the pooled exposure of ``curve_side``), and
    see whether the two sides saw traffic of the same shape."""

    #: Replies whose served length reached the request's output cap.
    events: int
    #: Teacher-forced tokens those replies contributed.
    tokens: int


class Calibration(Spec):
    """The full record of one calibration run for one task-folder's workload.

    Fields:
        kind:                  Which gate this file calibrates, and the tag
                               ``read_calibration`` dispatches on. ``"flips"`` is
                               the token-level gate's artifact, this model; a
                               sampled task's file says ``"report"`` and is a
                               ``ReportCalibration``. Every file carries it:
                               ``read_calibration`` refuses one without it.
        task:                  Dotted module path the calibration was computed
                               for (e.g. ``"tasks.sample_task.task"``), recorded
                               for audit -- not what ``read_calibration`` gates
                               on (that is ``workload_fingerprint``).
        workload_fingerprint:  The ``workload_fingerprint(task)`` hash this
                               calibration is valid for; ``read_calibration``
                               refuses a mismatch (a stale calibration for a
                               task whose workload changed since).
        workload:              Verbatim copy of the fingerprint payload -- the
                               model/contract/seed/tenant specs that define the
                               reference-visible traffic (``workload_payload``),
                               persisted so the file records exactly what workload
                               the noise floor was profiled on. A validator checks
                               its canonical-JSON hash equals ``workload_fingerprint``,
                               so the copy can't drift from what the gate checks.
        gpu_kind:              The accelerator kind the calibration ran on
                               (e.g. ``"B200"``); ``read_calibration`` refuses a
                               task whose ``hardware.accelerator.kind`` differs,
                               since bf16 noise floors are GPU-kind specific. Has
                               no default: a noise floor with no recorded GPU kind
                               cannot be gated.
        stack:                 The software/hardware stack (audit only).
        preflight:             The fp32-agreement sanity check; a failed
                               preflight makes the whole calibration
                               untrustworthy.
        engines:               How each pool engine was served while it was
                               profiled, keyed like ``rate_side`` (``EngineRecord``).
                               Audit only.
        draw:                  How many sessions one serving was asked to drive, and
                               the seed each serving was clocked with.
        tolerances:            The statistical demands the campaign was sized
                               against.
        rate_side:             Per-engine flip rate across the servings; ``q_max``
                               is the worst engine's mean.
        curve_side:            The flip-count curve rule's calibrated parameters.
        sizing:                Whether the campaign's exposure and serving count
                               can carry those ceilings, and why it stopped.
        capped_text:           How much of the measured text ran to the output cap
                               rather than ending on end-of-sequence, over every
                               engine. Descriptive; it is inside the ceilings.
        q_max:                 The calibrated ceiling on token-level disagreement
                               rate; validator-locked to the worst engine's mean
                               rate, so a hand-edit that loosens the deployed gate
                               is unrepresentable. Must be > 0: a 0 ceiling would
                               fail any trace with a single mismatch, defeating
                               the point of calibrating a noise floor.
        created_at:            ISO-8601 timestamp the calibration run completed.
        hardware:              The distinct hardware records of the allocations its servings
                               ran on, as their grant checks measured them, sorted
                               (``calibration_hardware``); absent from a file that records
                               none. It gates nothing.
    """

    kind: Literal["flips"] = "flips"
    task: str
    workload_fingerprint: str
    workload: dict
    gpu_kind: str
    stack: StackInfo
    preflight: PreflightResult
    engines: dict[str, EngineRecord]
    draw: DrawSpec
    tolerances: Tolerances
    rate_side: dict[str, EngineRate]
    curve_side: CurveSide
    sizing: Sizing
    capped_text: CappedText
    q_max: float
    created_at: str
    hardware: list[HardwareRecord] | None = Field(default=None, exclude_if=lambda v: v is None)

    @property
    def delta_per_rung(self) -> float:
        """The per-rung Bonferroni share of the family false-alarm budget: the family
        probability the campaign was sized against, split over the gap rungs plus the
        rate rule."""
        return self.tolerances.delta / self.curve_side.n_rungs

    def thresholds(self, n_tokens: int) -> list[dict]:
        """The curve rule's per-level rejection thresholds for a trace of ``n_tokens``
        pooled teacher-forced tokens (see ``CurveSide.thresholds``)."""
        return self.curve_side.thresholds(n_tokens, self.delta_per_rung)

    @property
    def precision(self) -> dict[str, SizingCheck]:
        """(P) per engine, recomputed from the servings and tolerances stored here."""
        return {name: precision_check(rate, self.sizing.c_m, self.tolerances.eps)
                for name, rate in self.rate_side.items()}

    @property
    def tail(self) -> dict[str, SizingCheck]:
        """(T) per engine, recomputed from the exposures and tolerances stored here."""
        return {name: tail_check(eng.exposure, self.tolerances.mu_per_mtok)
                for name, eng in self.curve_side.engines.items()}

    @model_validator(mode="after")
    def _check_ceiling(self) -> "Calibration":
        if not self.rate_side:
            raise CalibrationError("Calibration carries no engines")
        want = max(e.mean for e in self.rate_side.values())
        if self.q_max != want:
            raise CalibrationError(
                f"Calibration's q_max={self.q_max} contradicts its own servings (worst "
                f"engine mean {want}). Re-run `uv run hls calibrate <module>` rather "
                f"than hand-editing calibration.json.")
        if self.q_max <= 0:
            raise CalibrationError(
                f"Calibration has a degenerate ceiling (q_max={self.q_max}); it must be "
                f"> 0 -- a 0 ceiling means any single token mismatch fails the gate, "
                f"defeating the point of calibrating a bf16 noise floor.")
        return self

    @model_validator(mode="after")
    def _check_engine_keyed_maps_agree(self) -> "Calibration":
        # Every engine-keyed map describes the same pool. A curve for an engine with no
        # measured rate (or the reverse) is a half-measured engine the gate would still
        # grade against; a config naming an engine the file carries no stats for is
        # unreviewable.
        rate_engines = set(self.rate_side)
        if set(self.curve_side.engines) != rate_engines:
            raise CalibrationError(
                f"Calibration's curve_side covers engines {sorted(self.curve_side.engines)} "
                f"but its rate_side covers {sorted(rate_engines)}; both rules must be "
                f"measured on the same pool.")
        unknown = sorted(set(self.engines) - rate_engines)
        if unknown:
            raise CalibrationError(
                f"Calibration's engines name engine(s) {unknown} that are absent from "
                f"its pool (known: {sorted(rate_engines)}).")
        # The seed record is engine-keyed too, and it must describe the same servings
        # the rates were measured over: one seed per serving, per engine. A record that
        # is short is a record that cannot re-drive what it claims to.
        seeds = self.draw.serving_seeds
        if set(seeds) != rate_engines:
            raise CalibrationError(
                f"Calibration's draw.serving_seeds cover engines {sorted(seeds)} but its "
                f"rate_side covers {sorted(rate_engines)}; every engine's servings must "
                f"be recorded.")
        for engine, engine_seeds in seeds.items():
            if len(engine_seeds) != self.rate_side[engine].draws:
                raise CalibrationError(
                    f"Calibration records {len(engine_seeds)} serving seeds for engine "
                    f"{engine!r} but {self.rate_side[engine].draws} servings; each "
                    f"serving is clocked with exactly one seed. Re-run `uv run hls "
                    f"calibrate <module>` rather than hand-editing calibration.json.")
        return self

    @model_validator(mode="after")
    def _check_workload_copy(self) -> "Calibration":
        digest = hashlib.sha256(
            json.dumps(self.workload, sort_keys=True).encode()).hexdigest()
        if digest != self.workload_fingerprint:
            raise CalibrationError(
                f"Calibration's persisted workload copy does not hash to its "
                f"workload_fingerprint ({digest} != {self.workload_fingerprint}); the audit "
                f"copy has been edited independently of the gate. Re-run "
                f"`uv run hls calibrate <module>` rather than hand-editing calibration.json."
            )
        return self


#: One-sided 95% normal quantile: the factor between an engine's measured mean
#: divergence and the upper limit the sampling ceiling is taken from.
UPPER_Z: float = 1.645

#: Default declared expected end-of-turn count for one graded trace (E_0). The
#: end-of-turn count's threshold is chosen to hold the expectation under this, which is
#: what fixes the check's resolution: the null's spread is about its square root, so a
#: trace at 5 separates about 15 excess endings at three standard deviations. Declaring
#: the expectation rather than the threshold keeps the check's power the same across
#: tasks whose traffic prices endings differently.
DEFAULT_EOT_EXPECTED_MAX: float = 5.0


class EngineReport(Spec):
    """One pool engine's measured report divergence, as it was profiled.

    Fields:
        n_servings:        Servings of the task's traffic this engine drove.
        n_replies:         Replies measured over those servings.
        n_tokens:          Positions those replies contributed, which is every position
                           they served: a reply is measured at all of them or at none.
                           The mean is per position, so this is what its standard error
                           is over.
        mean_kl:           Mean coarsened divergence of this engine's own reports
                           from the pinned reference (``hls.verify.report``).
        se_kl:             That mean's standard error, clustered by reply -- a reply
                           served from a slightly wrong distribution is wrong all the
                           way down, so its positions are one piece of evidence.
        upper:             ``mean_kl + UPPER_Z * se_kl``, the one-sided 95% upper limit;
                           validator-locked to that arithmetic.
        precision_passed:  Whether ``se_kl <= eps * mean_kl`` held for this engine, i.e.
                           whether the campaign measured it closely enough.
        outside_rate:      The fraction of this engine's served positions whose token
                           was not one of its reported atoms
                           (``hls.verify.report.PooledReport.outside_rate``): the
                           model's own tail mass on this traffic, as a faithful
                           report shows it.
        se_outside:        That rate's standard error, clustered by reply like
                           ``se_kl``.
        outside_upper:     ``outside_rate + UPPER_Z * se_outside``, the same one-sided
                           95% upper limit; validator-locked to that arithmetic.
    """

    n_servings: int
    n_replies: int
    n_tokens: int
    mean_kl: float
    se_kl: float
    upper: float
    precision_passed: bool
    outside_rate: float
    se_outside: float
    outside_upper: float


class EndOfTurnEngine(Spec):
    """One pool engine's end-of-turn count at the calibrated threshold.

    The check's null says a faithful engine ends its turn at the rate the reference
    prices, so these four numbers are the evidence that it holds on stacks known to be
    faithful. They gate nothing at run time; the deployed verifier recomputes the same
    numbers on the graded trace.

    All four are pooled over every serving this engine drove, so at more than one
    serving each ``expected`` runs above ``EndOfTurnCheck.expected_max``, which bounds
    one serving. That is the same pooling the divergence ceiling uses.

    Fields:
        priced_positions: Positions the reference priced at or below the threshold.
        expected:         End-of-turn tokens the reference expects among them.
        observed:         End-of-turn tokens this engine served among them.
        p_value:          The Poisson-binomial upper tail of ``observed``.
    """

    priced_positions: int
    expected: float
    observed: int
    p_value: float


class EndOfTurnCheck(Spec):
    """The end-of-turn count's one calibrated input, and what the trusted pool measured
    at it.

    The check itself needs no measured constant: its null mean is computed from the
    graded trace. What has to be chosen once per task is which positions are cheap
    enough to be evidence, and that is ``eps`` -- picked as the largest value on
    ``hls.verify.report.EOT_EPS_GRID`` at which the expected count over one serving of
    the task's own traffic stays at or below ``expected_max``.

    Fields:
        ids:          The checkpoint's end-of-turn ids, as the pinned reference
                      reported them. The deployed verifier refuses a reference that
                      names a different set, since the threshold below was chosen for
                      this one.
        eps:          The chosen threshold on the reference's end-of-turn probability.
        expected_max: The declared expectation the threshold was chosen to hold under.
        engines:      Per pool engine, its count at ``eps`` (``EndOfTurnEngine``).
    """

    ids: list[int]
    eps: float
    expected_max: float
    engines: dict[str, EndOfTurnEngine]

    @model_validator(mode="after")
    def _check_threshold(self) -> "EndOfTurnCheck":
        if not self.ids:
            raise CalibrationError(
                "EndOfTurnCheck carries no end-of-turn ids, so the count it describes "
                "was taken over nothing; re-run `uv run hls calibrate <module>`.")
        if self.eps not in EOT_EPS_GRID:
            raise CalibrationError(
                f"EndOfTurnCheck's eps={self.eps} is not one of the published "
                f"thresholds {list(EOT_EPS_GRID)}; the value is stated in the brief, so "
                f"it is chosen from that grid rather than solved for. Re-run `uv run "
                f"hls calibrate <module>` rather than hand-editing calibration.json.")
        if self.expected_max <= 0:
            raise CalibrationError(
                f"EndOfTurnCheck's expected_max={self.expected_max} must be > 0: the "
                f"threshold is chosen to hold the expected count under it, and no "
                f"threshold holds it under zero.")
        if not self.engines:
            raise CalibrationError("EndOfTurnCheck carries no engines")
        return self


class ReportCalibration(Spec):
    """The calibrated ceiling for the sampling gate, for one task-folder's workload.

    The sampling gate has two measured numbers, not a rule family: ``kappa``, the
    divergence a trusted implementation of the same checkpoint already shows against
    the pinned reference, and ``outside_ceiling``, how often a faithful report's served
    token falls outside its own atoms. The gate's other tests read exact nulls and need
    nothing measured, which is why none of the token gate's machinery -- severity
    rungs, Poisson counts, exposure targets -- appears here. The end-of-turn count
    needs one chosen threshold rather than a measured ceiling, and that is
    ``end_of_turn``.

    Fields:
        kind:                  ``"report"``; the tag ``read_calibration`` dispatches on.
        task:                  Dotted module path the calibration was computed for,
                               recorded for audit.
        workload_fingerprint:  The ``workload_fingerprint(task)`` this file is valid
                               for; ``read_calibration`` refuses a mismatch.
        workload:              Verbatim copy of the fingerprint payload, validator-locked
                               to hash to ``workload_fingerprint`` so the audit copy
                               cannot drift from what the gate checks.
        gpu_kind:              The accelerator kind the calibration ran on; the verifier
                               and the loader both refuse other silicon, since the
                               numerical noise this ceiling covers is kind-specific.
        stack:                 The software stack it was measured on. The verifier
                               refuses a reference whose stack drifted from it.
        engines:               Per pool engine, its measured divergence
                               (``EngineReport``).
        kappa:                 The ceiling: the highest upper limit over the pool, so it
                               covers every trusted stack rather than the average of
                               them. Validator-locked to that maximum, and must be > 0.
        outside_ceiling:       The ceiling on the served-outside rate, the highest
                               ``outside_upper`` over the pool for the same reason.
                               Validator-locked to that maximum, and must be > 0: a
                               sampler at the contract's temperature lands outside any
                               19 atoms at their tail mass, so a pool measuring 0
                               did not measure a sampler.
        eps:                   The precision target each engine's ``precision_passed``
                               was decided against. Recorded, never defaulted: the
                               campaign is told what to measure to (``--eps``, which
                               resolves ``hls.calibrate.DEFAULT_EPS`` for both gates),
                               and a file that named none would be a measurement whose
                               precision claim came from whatever the reader assumed.
        end_of_turn:           The end-of-turn count's threshold and what the pool
                               measured at it (``EndOfTurnCheck``).
        created_at:            ISO-8601 timestamp the calibration run completed.
        hardware:              The distinct hardware records of the allocations its servings
                               ran on, as their grant checks measured them, sorted
                               (``calibration_hardware``); absent from a file that records
                               none. It gates nothing.
    """

    kind: Literal["report"] = "report"
    task: str
    workload_fingerprint: str
    workload: dict
    gpu_kind: str
    stack: StackInfo
    engines: dict[str, EngineReport]
    kappa: float
    outside_ceiling: float
    eps: float
    end_of_turn: EndOfTurnCheck
    created_at: str
    hardware: list[HardwareRecord] | None = Field(default=None, exclude_if=lambda v: v is None)

    @model_validator(mode="after")
    def _check_ceiling(self) -> "ReportCalibration":
        if not self.engines:
            raise CalibrationError("ReportCalibration carries no engines")
        for name, eng in self.engines.items():
            want = eng.mean_kl + UPPER_Z * eng.se_kl
            if not math.isclose(eng.upper, want, rel_tol=1e-9, abs_tol=1e-12):
                raise CalibrationError(
                    f"ReportCalibration engine {name!r} records upper={eng.upper} but its "
                    f"own mean and standard error give {want}. Re-run `uv run hls "
                    f"calibrate <module>` rather than hand-editing calibration.json.")
            want_outside = eng.outside_rate + UPPER_Z * eng.se_outside
            if not math.isclose(eng.outside_upper, want_outside, rel_tol=1e-9,
                                abs_tol=1e-12):
                raise CalibrationError(
                    f"ReportCalibration engine {name!r} records "
                    f"outside_upper={eng.outside_upper} but its own rate and standard "
                    f"error give {want_outside}. Re-run `uv run hls calibrate <module>` "
                    f"rather than hand-editing calibration.json.")
        want_kappa = max(eng.upper for eng in self.engines.values())
        if not math.isclose(self.kappa, want_kappa, rel_tol=1e-9, abs_tol=1e-12):
            raise CalibrationError(
                f"ReportCalibration's kappa={self.kappa} contradicts its own servings "
                f"(highest engine upper limit {want_kappa}). Re-run `uv run hls calibrate "
                f"<module>` rather than hand-editing calibration.json.")
        if self.kappa <= 0:
            raise CalibrationError(
                f"ReportCalibration has a degenerate ceiling (kappa={self.kappa}); it "
                f"must be > 0 -- a 0 ceiling fails any trace whose reports differ from "
                f"the reference at all, which two correct implementations of one "
                f"checkpoint always do.")
        want_outside_ceiling = max(eng.outside_upper for eng in self.engines.values())
        if not math.isclose(self.outside_ceiling, want_outside_ceiling, rel_tol=1e-9,
                            abs_tol=1e-12):
            raise CalibrationError(
                f"ReportCalibration's outside_ceiling={self.outside_ceiling} contradicts "
                f"its own servings (highest engine outside_upper "
                f"{want_outside_ceiling}). Re-run `uv run hls calibrate <module>` rather "
                f"than hand-editing calibration.json.")
        if self.outside_ceiling <= 0:
            raise CalibrationError(
                f"ReportCalibration has a degenerate served-outside ceiling "
                f"(outside_ceiling={self.outside_ceiling}); it must be > 0 -- a sampler "
                f"lands outside any 19 atoms at their tail mass, so a 0 ceiling fails "
                f"any faithful trace on its first such position.")
        return self

    @model_validator(mode="after")
    def _check_workload_copy(self) -> "ReportCalibration":
        digest = hashlib.sha256(
            json.dumps(self.workload, sort_keys=True).encode()).hexdigest()
        if digest != self.workload_fingerprint:
            raise CalibrationError(
                f"ReportCalibration's persisted workload copy does not hash to its "
                f"workload_fingerprint ({digest} != {self.workload_fingerprint}); the "
                f"audit copy has been edited independently of the gate. Re-run `uv run "
                f"hls calibrate <module>` rather than hand-editing calibration.json.")
        return self


def calibration_hardware(records: list[HardwareRecord]) -> list[HardwareRecord]:
    """*records*, the hardware of the allocations a calibration's servings ran on, without repeats
    and sorted by their JSON, as a calibration file records them."""
    return sorted(set(records), key=lambda h: h.model_dump_json())


#: Every calibration artifact, by the ``kind`` tag its file carries. A verifier names
#: the one it can read (``Verifier.calibration_kind``), and ``read_calibration``
#: refuses a file of any other kind rather than validating it into the wrong model.
CALIBRATION_KINDS: dict[str, type[Spec]] = {"flips": Calibration, "report": ReportCalibration}


def device_matches_kind(device_name: str, kind: str) -> bool:
    """True when a measured accelerator name (``torch.cuda.get_device_name``) is a
    device of accelerator kind ``kind`` -- the kind token as a whole word, so
    "NVIDIA H200" and "NVIDIA H200 NVL" both match kind "H200" while "NVIDIA
    B200" does not.

    Both ends of the GPU-identity gate use this. A calibration refuses to measure on
    silicon that is not the task's declared accelerator (``hls.calibrate._stack_of``
    and ``assert_declared_device``), which is what makes ``read_calibration``'s
    declared-vs-recorded ``gpu_kind`` comparison meaningful -- otherwise both sides of
    it come from the same declaration. The verifier refuses to score when the serving
    reference's device is not the calibrated kind (substrates do not all enforce the
    declared accelerator, so the load-time check alone cannot catch a wrong-silicon
    run)."""
    return re.search(rf"\b{re.escape(kind)}\b", device_name) is not None


def _traffic_spec(data_source) -> dict | None:
    """A data source projected to what it makes the reference see.

    ``dev_reference`` and the name of a measured one are dropped: they are text handed to the
    contestant beside the dev rows, published so a prompt element it could not otherwise infer is visible. It
    changes no served byte -- whatever it discloses is already in the prompt template,
    which is hashed -- so including it would let a disclosure invalidate a calibration.
    """
    if data_source is None:
        return None
    dumped = dump_spec(data_source)
    dumped.pop("dev_reference", None)
    dumped.pop("measured_dev_reference", None)
    return dumped


def agent_payload(agent) -> dict:
    """The agent seam's traffic-shaping identity: which framework, configured how,
    offering which tools, for how many turns.

    An agentic task's prompts are whatever its framework builds out of the tool results
    it gets, so all four decide the population the reference teacher-forces.
    ``max_concurrent_llm`` is concurrency and stays out, like the arrival process.
    ``assets`` names the provisioned tool data by
    identity: a different corpus or index changes every tool result, so it changes the
    population.

    A declared scheduled stop is in for the reason ``history_char_budget`` is: it ends
    a session at a fixed point, so it decides how many turns of each trajectory exist
    to be served. Present only where a task declared one, so a task without a
    scheduled stop keeps the identity it had."""
    fw = agent.framework
    payload = {
        "framework": f"{type(fw).__module__}.{type(fw).__qualname__}",
        "framework_config": fw.traffic_config(),
        "tools": [{"name": s.name, "description": s.description,
                   "json_schema": s.json_schema}
                  for s in agent.tool_runner.describe_tools()],
        "max_turns": agent.max_turns,
    }
    if getattr(agent, "assets", None) is not None:
        payload["assets"] = agent.assets.identity()
    if getattr(agent, "wall_clock_sec", None) is not None:
        payload["wall_clock_sec"] = agent.wall_clock_sec
        payload["cut_grace_sec"] = agent.cut_grace_sec
    return payload


def workload_payload(task: "TaskSpec") -> dict:
    """The fields that determine the reference-visible traffic distribution, as a
    plain-JSON dict. ``workload_fingerprint`` is the hash of this payload, and
    ``Calibration.workload`` persists it verbatim, so a calibration file records
    exactly what workload profiled it.

    The criterion: anything that changes the population of prompts and continuations
    the reference teacher-forces belongs here and invalidates a calibration; anything
    that only changes the timing with which that population is served does not. The
    fields are:

      - ``model``:                the weights whose bf16 noise floor is being measured.
      - ``contract``:             the request shape/protocol every prompt is rendered through.
      - ``seed``:                 draws the arrival schedule that, through
                                  ``trim_rampdown``, decides which follow-up turns exist.
      - ``tenants[].id``:         keys the tenant's provisioned rows and its per-tenant seed.
      - ``tenants[].data_source``: the rows themselves -- dataset, the commit of every hub
                                  repo it reads (a corpus-time tokenizer's included),
                                  prompt/target templates, split sizes: the prompts the
                                  reference sees.
                                  Minus ``dev_reference`` (see ``_traffic_spec``), which
                                  is disclosure to the contestant, not traffic.
      - ``tenants[].output_length_cap``: bounds the continuations the reference scores.
      - ``tenants[].history_char_budget``: the budget a live session's stitched history
                                  may reach. A session ends at the first turn that would
                                  exceed it, so the budget decides how many turns of each
                                  conversation exist to be served -- it sets the
                                  population directly, not just the size of each prompt.
      - ``tenants[].trim_rampdown``: whether follow-up turns past the last scheduled
                                  session arrival are dropped at draw time, which
                                  decides which turns exist to be served at all.
      - ``tenants[].references``: the reference config the noise floor is measured
                                  against -- the oracle's weights and decode, and the
                                  numeric knobs of the instrument itself, down to the
                                  width its teacher-forcing forward is tiled at
                                  (``HFReference.query_chunk``). A ceiling measured on
                                  one instrument cannot grade another.
      - ``agent``:                the framework, its traffic-shaping settings, the
                                  tool definitions rendered into every prompt, and the
                                  turn budget -- an agentic task's prompts are built by
                                  its framework, not read off its rows. None for a task
                                  with no agent seam.
      - ``max_context_len``:      the context window the task's server serves in. A
                                  framework finalizes a trajectory against it, so it
                                  decides how many turns of each conversation exist to
                                  be served. None for a task that declares none.

    Excluded: the tenant's ``arrival`` process, its ``slo``, and its
    ``inter_turn_delay``; plus the task id (so sibling task modules in one folder --
    good/bad fixtures -- share one calibration file), scorer, images, and verifier
    thresholds.

    Calibration serves each engine through the task's own clocked drive, so arrival
    rate and think-times shape the batch composition a flip rate is measured under.
    They stay out of the identity because a re-pinned load (a baseline search moving
    the task's total rate) would otherwise stale an otherwise-valid noise floor, and
    because the gate's headroom is set well above how far load moves a noise floor.
    A change of load leaves the fingerprint as it is; recalibrating after one is the
    task author's call.
    """
    tenants = []
    for t in task.effective_tenants():
        refs = [dump_spec(v.reference) for v in t.verifiers
                if getattr(v, "needs_calibration", False)]
        tenants.append({
            "id": t.id,
            "data_source": _traffic_spec(t.data_source),
            "output_length_cap": dump_spec(t.workload.output_length_cap),
            "history_char_budget": t.workload.history_char_budget,
            "trim_rampdown": t.workload.trim_rampdown,
            "references": refs,
        })
    return {"model": dump_spec(task.model), "contract": dump_spec(task.contract),
            "seed": task.seed, "tenants": tenants,
            "agent": agent_payload(task.agent) if task.agent is not None else None,
            "max_context_len": task.max_context_len}


def workload_fingerprint(task: "TaskSpec") -> str:
    """Canonical-JSON hash of ``workload_payload(task)`` -- the identity
    ``read_calibration`` gates on (a stored calibration whose fingerprint differs
    from the task's current one is stale and refused)."""
    return hashlib.sha256(
        json.dumps(workload_payload(task), sort_keys=True).encode()).hexdigest()


#: How a refusal of an unusable calibration tells the operator to replace it: ``hls calibrate``
#: measures one into the store.
_REMEASURE = "measure it again with `hls calibrate <module>`, which writes it into the store"


def read_calibration(path: Path, task: "TaskSpec",
                     kind: str = "flips") -> tuple[Calibration | ReportCalibration, bytes]:
    """Read and validate the calibration file at *path* against ``task``, once: the calibration
    and the bytes it was parsed from, which a record names by their blob id.

    ``kind`` is the artifact the task's calibrated verifiers can read
    (``Verifier.calibration_kind``); a file tagged anything else is refused rather
    than coerced, because the two artifacts calibrate different gates and neither
    verifier can grade on the other's numbers.

    Fails loud (``CalibrationError``, always naming the fix) rather than ever
    running with a wrong, stale, or untrustworthy calibration:
      - the file is missing;
      - it carries no ``kind`` tag, or not the one the verifier needs;
      - its ``workload_fingerprint`` does not match this task's current one
        (the workload changed since it was calibrated);
      - its ``gpu_kind`` does not match ``task.hardware.accelerator.kind``
        (the numerical noise floors it records are GPU-kind specific);
      - its own ``preflight.passed`` is False (the calibration run itself was
        untrustworthy). Only the token-level artifact carries a preflight.
    """
    path = Path(path)
    if not path.exists():
        raise CalibrationError(
            f"no calibration at {path}; measure it with `hls calibrate <module>`, which writes it into the store")
    model = CALIBRATION_KINDS.get(kind)
    if model is None:
        raise CalibrationError(
            f"unknown calibration kind {kind!r} (known: {sorted(CALIBRATION_KINDS)})")
    # The tag is read from the raw file before any model validates it: a file of the
    # other kind would otherwise fail as a heap of missing fields instead of saying
    # which gate it calibrates.
    data = path.read_bytes()
    try:
        raw = json.loads(data)
    except ValueError as exc:
        raise CalibrationError(f"{path} is not valid JSON ({exc}); {_REMEASURE}") from exc
    if not isinstance(raw, dict) or "kind" not in raw:
        raise CalibrationError(
            f"{path} carries no kind tag, and every calibration `hls calibrate <module>` writes has one; "
            f"{_REMEASURE}")
    found = raw["kind"]
    if found != kind:
        owner = raw.get("task")
        measured_for = f" (measured for {owner!r})" if owner else ""
        raise CalibrationError(
            f"{path} calibrates the {found!r} gate{measured_for} but this task's "
            f"verifiers need the {kind!r} one; {_REMEASURE}")
    try:
        cal = model.model_validate(raw)
    except ValidationError as exc:
        # JSON that does not fit the model surfaces here as a
        # pydantic.ValidationError; the refusal names the file and the fix like every
        # other path here.
        raise CalibrationError(f"{path} is not a valid calibration file ({exc}); {_REMEASURE}") from exc
    fingerprint = workload_fingerprint(task)
    if cal.workload_fingerprint != fingerprint:
        raise CalibrationError(
            f"stale calibration {path}: it was computed for workload_fingerprint="
            f"{cal.workload_fingerprint!r} but this task's current fingerprint is "
            f"{fingerprint!r}; {_REMEASURE}"
        )
    gpu_kind = task.hardware.accelerator.kind
    if cal.gpu_kind != gpu_kind:
        raise CalibrationError(
            f"calibration {path} was computed on gpu_kind={cal.gpu_kind!r} but this "
            f"task's hardware.accelerator.kind={gpu_kind!r}; bf16 noise floors are GPU-kind "
            f"specific -- measure it again on the target GPU with `hls calibrate <module>`"
        )
    if not isinstance(cal, Calibration):
        # The remaining gates below read the token-level artifact's own sections. The
        # report artifact has no preflight: its ceiling is one
        # measured mean per engine, each with its own precision check, and a ceiling
        # built on an imprecise mean is as untrustworthy as a failed preflight.
        imprecise = [name for name, eng in cal.engines.items() if not eng.precision_passed]
        if imprecise:
            raise CalibrationError(
                f"calibration {path} failed its precision check on "
                f"{', '.join(sorted(imprecise))}, so its ceiling is not trustworthy; measure it "
                f"again with `hls calibrate <module>` and a larger `--servings-per-engine`"
            )
        return cal, data
    if not cal.preflight.passed:
        raise CalibrationError(
            f"calibration {path} failed its own fp32-agreement preflight, so it is not "
            f"trustworthy; {_REMEASURE}"
        )
    return cal, data


def calibration_kind(task: "TaskSpec") -> str | None:
    """Which calibration artifact ``task``'s verifiers need, or ``None`` when none do.

    A task has one calibration.json, so its calibrated verifiers must all
    read the same artifact. Two that disagree is an authoring error with no right
    resolution -- whichever file were written, one of them would grade on numbers that
    are not about its gate -- so it is refused here, before any campaign runs."""
    kinds = {getattr(v, "calibration_kind", "flips")
             for t in task.effective_tenants() for v in t.verifiers
             if getattr(v, "needs_calibration", False)}
    if not kinds:
        return None
    if len(kinds) > 1:
        raise CalibrationError(
            f"task {task.id!r} declares calibrated verifiers needing different "
            f"calibration artifacts ({sorted(kinds)}), and a task has one "
            f"calibration.json; declare verifiers that share one gate")
    return next(iter(kinds))


def calibrated(task: "TaskSpec", path: Path | None
               ) -> tuple["TaskSpec", tuple[Calibration | ReportCalibration, bytes] | None]:
    """Return ``task`` with a loaded calibration threaded into every
    calibration-requiring verifier, never mutating the input, and the calibration with the
    bytes of the file it was parsed from (``read_calibration``); None for those when no verifier
    needs one.

    Returns ``task`` unchanged (same object) when no verifier declares
    ``needs_calibration``. Otherwise requires the file's *path* (``None`` raises
    ``CalibrationError``), reads the artifact its verifiers name
    (``calibration_kind``) once, and rebuilds the
    frozen tree with ``model_copy`` -- verifier, then its owning verifiers list
    (top-level ``TaskSpec.verifiers`` for the single-tenant authoring form, or
    each ``Tenant.verifiers`` for the multi-tenant form), then the TaskSpec
    itself. ``Spec.model_copy`` validates every copy, so the rebuilt task passes
    ``TaskSpec._check``/``check_wiring`` again; injecting a calibration changes no
    field either one reads.
    """
    kind = calibration_kind(task)
    if kind is None:
        return task, None
    if path is None:
        raise CalibrationError(
            f"task has a calibration-requiring verifier but no {CALIBRATION_FILE} was given; {_REMEASURE}")
    cal, data = read_calibration(path, task, kind)

    def _inject(v):
        if getattr(v, "needs_calibration", False):
            return v.model_copy(update={"calibration": cal})
        return v

    if task.tenants is None:
        # Single-tenant authoring form: the top-level `verifiers` list owns them.
        return task.model_copy(update={"verifiers": [_inject(v) for v in task.verifiers]}), (cal, data)
    # Multi-tenant authoring form: each Tenant owns its own `verifiers` list.
    new_tenants = [t.model_copy(update={"verifiers": [_inject(v) for v in t.verifiers]})
                   for t in task.tenants]
    return task.model_copy(update={"tenants": new_tenants}), (cal, data)
