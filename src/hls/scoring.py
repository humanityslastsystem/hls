"""L1 -- performance scoring: Scorer + Unit + SoLCeiling.

Two Scorer members, one per objective:

  - ``GoodputScorer`` -- goodput: SLO-credited throughput (unit/sec) against a
    roofline. The Unit subclass carries the per-shape counting logic
    (Request unit counts 1 per event); the SoLCeiling subclass carries the
    roofline. Request unit and ConstantSoL are implemented.
  - ``SloLatencyScorer`` -- speedup over the baseline serving stack in
    SLO-normalized latency: the geometric mean of 1/r over the SLO's own
    units, divided by the baseline's. The unit and r are the SLO's
    (a request at max(TTFT/S_ttft, TPOT/S_tpot), a trajectory at
    makespan/target). The workload's arrival is open-loop and fixed, so
    throughput isn't the server's to choose; how fast it answers relative to
    each tenant's budget is.

``Score.value`` is higher-is-better on both, and positive for every valid run;
0.0 with a non-empty ``zero_reasons`` means the run was gated. Both members check
every gate layer before returning, so a run that failed several is recorded as
failing all of them, and its breakdown carries every section it computed.

This is the performance domain (throughput / SLO / speed-of-light). The
correctness gates live in ``hls.verify``.
"""

from __future__ import annotations
import math
from abc import abstractmethod
from collections.abc import Collection
from typing import TYPE_CHECKING, ClassVar

from hls.core.model import Hardware
from hls.core.poly import Poly
from hls.core.score import Score
from hls.core.trace import Trace, TraceEvent, event_key
from hls.core.verdict import Verdict

if TYPE_CHECKING:
    from hls.spec.tenant import Tenant


class Unit(Poly):
    """The throughput unit the score is denominated in (L1).

    `count_for_event` says how many units one TraceEvent is worth (1 per request
    for RequestUnit; token counts for TokenUnit); `duration_sec` defines the time
    base. The Scorer divides total counted units by duration to get achieved
    throughput. Subclass per unit shape.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def count_for_event(self, ev: TraceEvent) -> float: ...

    def duration_sec(self, trace: Trace) -> float:
        return trace.duration_sec()


class RequestUnit(Unit):
    name = "request"

    def count_for_event(self, ev: TraceEvent) -> float:
        return 1.0


class TrajectorySessionUnit(Unit):
    """One unit per *session*, not per event. An agentic trajectory is N
    LLM calls that share a ``session_id``; the throughput grain is the
    trajectory. The first event in each session (``turn_index == 0``)
    contributes 1.0; the rest contribute 0.

    Composes with ``TrajectoryMakespanSLO``: a trajectory that misses the makespan
    target gets per-event credit 0, so the Scorer counts only completed-on-time
    trajectories in achieved throughput."""

    name = "trajectory"

    def count_for_event(self, ev: TraceEvent) -> float:
        # First-event detection without a Trace pass: any sane agentic framework
        # numbers turns from 0, so turn_index == 0 is the trajectory's anchor.
        return 1.0 if ev.turn_index == 0 else 0.0


class TokenUnit(Unit):  # pragma: no cover - stub
    name = "token"

    def count_for_event(self, ev: TraceEvent) -> float:
        raise NotImplementedError("TokenUnit is not implemented: an ArtifactResponse carries no token count")


class SoLCeiling(Poly):
    """The speed-of-light roofline the achieved throughput is normalized against (L1).

    `ceiling(trace, hardware)` returns the theoretical peak in unit/sec for this
    workload on this hardware; the Scorer reports value = achieved / ceiling.
    Subclass to model a hardware/workload-specific roofline (ConstantSoL is the
    trivial 1.0 case).
    """

    @abstractmethod
    def ceiling(self, trace: Trace, hardware: Hardware) -> float: ...


class ConstantSoL(SoLCeiling):
    """Constant theoretical throughput in unit/sec, regardless of trace/hardware.
    At the default, 1.0, Score.value is the raw achieved throughput in rps."""

    rps: float = 1.0

    def ceiling(self, trace: Trace, hardware: Hardware) -> float:
        return self.rps


class Scorer(Poly):
    """The run's objective: aggregates a Trace + Verdict into one Score (L1).

    Two members, one per objective: ``GoodputScorer`` scores SLO-credited throughput
    against a roofline, ``SloLatencyScorer`` scores speedup over the baseline serving
    stack in SLO-normalized latency. A Poly family so the objective a run was graded on
    is identifiable in provenance (``kind``).

    The four methods below are the whole baseline-credit protocol
    (``hls.baseline_credit``): whether this scorer grades against a measured baseline,
    how one is installed, and how the credit and the count of units behind it are read
    back off a drive of the baseline itself. A scorer graded against nothing but its own
    trace inherits all four and is never asked again.
    """

    def scores_against_baseline(self) -> bool:
        """True on a member scored relative to the task's baseline serving stack. The
        CLI loader asks before it requires the task's baseline.json to exist."""
        return False

    def with_baseline_credit(self, b: float) -> "Scorer":
        """A copy of this scorer grading against the measured baseline credit ``b``;
        a scorer that grades against no baseline has nowhere to put one and returns
        itself unchanged."""
        return self

    def measured_credit(self, score: Score) -> float | None:
        """The baseline credit ``score`` measured, this scorer having graded a drive
        of the baseline itself; None when it measures no such thing. The number a
        baseline campaign averages over its drives."""
        return None

    def measured_units(self, score: Score) -> int | None:
        """How many units the credit in ``score`` is a mean over; None when this
        scorer measures no credit. A baseline campaign records it per drive and
        refuses to pool drives that credited different numbers of units."""
        return None

    @abstractmethod
    async def score(
        self,
        tenants: "list[Tenant]",
        trace: Trace,
        verdict: Verdict,
        hardware: Hardware,
    ) -> Score: ...


#: Diagnostics on the gated zero a trace with no events earns. A drive that served
#: nothing has no latency to normalize and no throughput to report, so both scorers
#: refuse it rather than reporting the vacuous mean of an empty sample -- which reads
#: as a run that sat exactly on its SLO, and passes the gate.
_SERVED_NOTHING = {
    "n_requests": 0,
    "detail": "the drive recorded no requests at all, so there is nothing to score",
}


class GoodputScorer(Scorer):
    """SLO-credited throughput against a speed-of-light roofline (L1).

    ``value = achieved / sol``, where achieved counts each event that passed both its
    tenant's SLO credit and the verdict, in ``unit``s per second of drive."""

    unit: Unit
    sol: SoLCeiling

    async def score(
        self,
        tenants: "list[Tenant]",
        trace: Trace,
        verdict: Verdict,
        hardware: Hardware,
    ) -> Score:
        """Score a trace per-tenant: each tenant is gated by its own SLO and its
        own slice of the verdict; credited events sum into total goodput.

        Two gate layers: the aggregate verdict (correctness/failed_requests;
        tenant-agnostic) and, for each tenant, its workload.slo applied to that
        tenant's event slice. Only events that pass both the per-event verdict and
        the tenant's SLO credit are counted. Both layers are checked before the
        method returns, so a run that failed both says so. Single-tenant (one
        Tenant("t0"), whose events the driver stamps tenant_id="t0") is the same
        path with one slice.

        The SoL ceiling and unit are artifact-level and shared across tenants."""
        # Before the verdict, which passes vacuously on an empty trace: a drive that
        # recorded nothing served nothing, whatever any gate says about it. It carries
        # the verdict section like every other gated return, so a reader never has to
        # ask whether a missing section means "clean" or "never computed".
        if not trace.events:
            return Score.zero(reasons=("slo",),
                              breakdown={"slo": dict(_SERVED_NOTHING),
                                         "verdict": dict(verdict.diagnostics)})

        reasons: list[str] = []
        if not verdict.passed:
            # The verdict names why it failed: "failed_requests" (every verifier gate
            # passed but requests errored/timed out; diagnostics["failed_requests"]
            # holds the verbatim driver fault strings) vs "correctness" (a verifier
            # gate failed). Verdicts without a failure_mode (per-verifier verdicts,
            # hand-built test verdicts) fall back to the "correctness" label.
            reasons.append(verdict.failure_mode or "correctness")

        # Per-tenant SLO gate: for each tenant apply its slo to its slice.
        # Trace.events_for() returns the tenant's driver-stamped slice.
        # Per-tenant: failing the SLO zeros that tenant's credit; other tenants
        # are unaffected (the run-level gate stays passed on partial credit).
        # Credits are keyed by ev.event_key (tenant-namespaced), not request_id:
        # two tenants over the same dataset path share request_ids, so a flat
        # request_id map would let one tenant's credit clobber the other's. The
        # SLO's own per_event_credit is request_id-keyed within the tenant's slice
        # (tenant-agnostic), so the inner lookup stays request_id.
        tenant_credits: dict[str, float] = {}   # event_key -> credit
        slo_diagnostics: dict = {}
        any_slo_failure = False                  # did any tenant fail its SLO gate?
        for tenant in tenants:
            t_events = trace.events_for(tenant.id)
            if not t_events:
                continue
            t_trace = Trace(events=t_events)
            slo_res = tenant.workload.slo.apply(t_trace, tenant.workload)
            slo_diagnostics[tenant.id] = dict(slo_res.diagnostics)
            if not slo_res.aggregate_passed:
                any_slo_failure = True
                # This tenant failed its SLO: its events get 0 credit. We continue
                # (other tenants may still pass); we do not zero the whole score
                # here -- only that tenant's contribution.
                for ev in t_events:
                    tenant_credits[ev.event_key] = 0.0
            else:
                for ev in t_events:
                    tenant_credits[ev.event_key] = slo_res.per_event_credit.get(
                        ev.request_id, 1.0
                    )

        # breakdown['slo'] is always nested per tenant ({tenant_id: {...}}), single-
        # and multi-tenant alike, so the viewer reads every run the same way (a
        # single-tenant run is just {"t0": {...}}). The viewer indexes by tenant id.
        slo_section = slo_diagnostics

        # Per-request SLO attainment: per-event credit combined with the per-event
        # verdict, averaged over the trace: a saturation signal a reader of score.json
        # takes without re-applying every tenant's SLO. The verdict intersection
        # matters: an overloaded system that sheds load by fast-failing must not read
        # as fully in-SLO. A gated-zero Score omits this key.
        slo_attainment = sum(
            tenant_credits.get(ev.event_key, 0.0)
            for ev in trace.events
            if verdict.per_event.get(ev.event_key, False)
        ) / len(trace.events)

        # Count credited events: must pass the per-event verdict and have a credit > 0.
        # Both maps are keyed by event_key (tenant-namespaced) so an event is matched
        # to its own tenant's verdict + credit, never a same-request_id sibling's.
        counted = 0.0
        for ev in trace.events:
            if verdict.per_event.get(ev.event_key, False):
                credit = tenant_credits.get(ev.event_key, 0.0)
                counted += self.unit.count_for_event(ev) * credit

        # Report an SLO-gated zero only when an SLO gate actually failed (tracked
        # explicitly above) and that zeroed all credit: a passing event exists but
        # nothing counted. Gating on `any_slo_failure` -- not inferring "slo" from
        # `counted == 0.0` alone -- avoids mislabeling a genuine zero a future
        # unit/SLO could produce without an SLO failure (e.g. a unit whose
        # count_for_event returns 0, or per_event_credit=0 under a passing SLO);
        # those fall through to a real value=0 score, gate still passed.
        if any_slo_failure and counted == 0.0 and any(
            verdict.per_event.get(ev.event_key, False) for ev in trace.events
        ):
            reasons.append("slo")

        if reasons:
            return Score.zero(reasons=reasons,
                              breakdown={"slo": slo_section,
                                         "verdict": dict(verdict.diagnostics)})

        duration = self.unit.duration_sec(trace)
        achieved = counted / duration if duration > 0 else 0.0
        sol = self.sol.ceiling(trace, hardware)
        value = achieved / sol if sol > 0 else 0.0
        return Score(
            value=value,
            achieved=achieved,
            sol=sol,
            unit=self.unit.name,
            gate_passed=True,
            breakdown={
                "slo": slo_section,
                "slo_attainment": slo_attainment,
                "verdict": dict(verdict.diagnostics),
            },
        )


#: Floor applied to a unit's SLO-normalized latency r before ``log(1/r)``. r is a
#: measured latency ratio and is > 0 for any real unit; it can only be exactly 0 if the
#: clock reported a zero TTFT (sub-resolution) on a request short enough to make TPOT
#: vacuous, and log(1/0) is +inf -- which would poison the sum and is not JSON-serializable
#: for provenance. The floor caps one unit's credit at log(1/1e-9) ~= 20.7 nats (a unit a
#: billion times faster than its budget). Only a measured ratio can reach it: the
#: censored path floors its own r at 1, so an unanswered request never lands here.
_R_FLOOR = 1e-9


def _log_credit(r: float) -> float:
    """One unit's credit: ``log(1/r)`` (natural log). r = 1 (exactly at the tenant's
    SLO) -> 0.0 exactly; faster -> positive; slower -> negative. See ``_R_FLOOR``."""
    return math.log(1.0 / max(r, _R_FLOOR))


def _mean_credit(credits: Collection[float]) -> float:
    """Mean of the per-unit log credits, unit-weighted. 0.0 when nothing was scored,
    the credit of a run sitting exactly on its SLO."""
    return sum(credits) / len(credits) if credits else 0.0


class SloLatencyScorer(Scorer):
    """Speedup over the baseline serving stack in SLO-normalized latency::

        r  = the SLO's own normalized latency, per unit, against that tenant's budgets
        L  = mean over units of log(1 / r)     log of the geometric mean of 1/r
        S  = exp(L - baseline_log_credit)      Score.value; 1.0 = the baseline stack

    The unit is whatever the tenant's SLO normalizes: a request for
    ``StreamingLatencySLO`` (r = max(TTFT/ttft_ms, TPOT/tpot_ms)), a trajectory for
    ``TrajectoryMakespanSLO`` (r = makespan/target_sec). ``SLO.units`` groups a trace at
    that grain and ``SLO.grain`` names it, so this scorer never branches on it.

    r comes from the tenant's own SLO (``SLOResult.per_event_ratio``), so tenants with
    different budgets are commensurable on one scale. ``baseline_log_credit`` is L measured
    for the task's baseline stack serving the same workload at the same arrival rate, the
    mean over repeated drives, injected by the CLI loader from the task folder's
    baseline.json; task authors never set it. None reports S = exp(L), the speedup against
    the SLO itself: the baseline measurement scores its own drives that way. S is the
    geometric mean of the per-unit speedups r_baseline / r: S = 2 halves a typical unit's
    SLO-normalized latency,
    halving every latency doubles S, and there is no ceiling.

    Why the geometric mean (Fleming & Wallace, CACM 29(3), 1986). It is the one mean of
    normalized ratios that agrees under inversion, so "latency 2x lower" and "2x faster"
    are the same statement and the ranking does not depend on which system is the
    reference. An arithmetic mean of r is set by the slowest few requests and an
    arithmetic mean of 1/r by the fastest few: a server that parks 90% of requests at
    their SLO and makes 10% blazing (r = 0.05) beats a uniformly 2x-better server on mean
    1/r (2.9 vs 2.0) and loses under the log (1.35 vs 2.0). A given relative improvement
    is worth the same on every request regardless of its tenant's budget.

    The SLO's units are weighted equally: requests for a per-request SLO, trajectories for
    a per-trajectory one. So the score follows the workload's production request mix, and a
    per-trajectory SLO does not weight a trajectory by how many turns it happened to take.
    The arrival process is open-loop and fixed, so throughput is not the server's to
    choose, and the metric has no plateau: one that saturated once every SLO is met would
    cap every compliant server at a tie.

    Hard gate: any failed request fails the run. Three layers, none a per-request penalty
    (which would let a server shed its hardest requests to lift the rest):
      1. the verdict -- ``verdict.passed`` False zeroes the run, contributing the verdict's
         ``failure_mode`` ("correctness" or "failed_requests");
      2. the SLO's own gate -- ``aggregate_passed=False`` when any request has no
         computable r, covering a baseline credit's assume-correct drives and a response that
         never streamed;
      3. this scorer -- any event whose unit has no ratio (including one belonging to no
         declared tenant) fails the run.
    Layers 2 and 3 are one reason, "slo". Every layer is checked before the score is
    returned, so a run that was both wrong and slow names both, and a gated run's
    breakdown carries the ``slo`` and ``verdict`` sections alike -- a correctness zero
    says what its latency was, and an SLO zero says what its verdict was.
    A gated run has ``Score.value == 0.0`` and a non-empty ``zero_reasons``; a valid run
    always has ``value > 0``. ``breakdown["upper_bound"]`` is the score the run would have earned with
    every unit lacking a computable r charged for the time it was observably
    outstanding (``SLO.censored_ratio``), and for its whole budget at the least: an upper
    bound on the artifact's score, for diagnosis, never ranked. It is 0.0 for a run that
    priced nothing at all, and equal to the score on a valid one.
    """

    #: L for the baseline serving stack on this workload, or None until the loader
    #: injects it. See the class docstring; task authors leave it None.
    baseline_log_credit: float | None = None

    def scores_against_baseline(self) -> bool:
        return True

    def with_baseline_credit(self, b: float) -> "Scorer":
        return self.model_copy(update={"baseline_log_credit": b})

    def measured_credit(self, score: Score) -> float | None:
        return score.breakdown.get("mean_log_credit")

    def measured_units(self, score: Score) -> int | None:
        # The units that earned a credit, which on a run this scorer let through the
        # gate is every unit the run served.
        return score.breakdown.get("n_scored")

    def _speedup(self, mean_log_credit: float) -> float:
        """S = exp(L - B), the geometric mean of the per-unit speedups over the
        baseline -- or exp(L), against the SLO itself, with no baseline measured."""
        b = self.baseline_log_credit
        return math.exp(mean_log_credit if b is None else mean_log_credit - b)

    def _bound(self, log_credits: dict[str, float],
               censored: dict[str, float]) -> float:
        """The score a gated run was reaching for: what it would have earned had every
        unit that produced no measurable latency answered at the moment the harness
        stopped watching it (``SLO.censored_ratio``), pooled with the credits it did
        measure. Built only on the gated branches; a valid run's bound is its score.

        0.0 when neither map holds anything, which is a run that scored nothing at all:
        every event belonged to a tenant the task does not declare, or no declared
        tenant's SLO defines a ratio to censor. A mean over no units is vacuous, and
        exp(-B) would report a passing bound for a run that measured nothing."""
        priced = log_credits | censored
        return self._speedup(_mean_credit(priced.values())) if priced else 0.0

    async def score(
        self,
        tenants: "list[Tenant]",
        trace: Trace,
        verdict: Verdict,
        hardware: Hardware,
    ) -> Score:
        # Per-tenant: apply each tenant's SLO to its own driver-stamped slice, exactly as
        # the goodput member does, then mean over that SLO's own units. The maps below are
        # keyed by tenant and unit id (``event_key``, which for a request unit is exactly
        # ev.event_key) -- two tenants over the same dataset path share request_ids and
        # session_ids, so a flat map would let one tenant clobber the other's. The SLO's
        # own maps are keyed within the tenant's slice, so the inner lookups stay bare.
        #
        # The empty trace goes first, before the verdict, which passes vacuously on it:
        # with no requests the mean log credit is 0.0 and the run would score exp(-B),
        # a pass, for serving nothing at all. Its upper bound is 0.0 for the same
        # reason -- there is no evidence it was reaching for anything.
        if not trace.events:
            return Score.zero(reasons=("slo",),
                              breakdown={"slo": dict(_SERVED_NOTHING),
                                         "verdict": dict(verdict.diagnostics),
                                         "upper_bound": 0.0})

        t_end = trace.t_end()                # the run's end, the SLO's censoring time
        log_credits: dict[str, float] = {}   # unit key -> log(1/r), the metric's terms
        censored: dict[str, float] = {}      # the same for units that produced no r
        met: dict[str, float] = {}           # unit key -> binary met (attainment only)
        units: dict[str, list[TraceEvent]] = {}   # unit key -> that unit's events
        priced: set[str] = set()             # event keys inside a unit that has an r
        slo_section: dict = {}
        gate_failed = False
        for tenant in tenants:
            t_events = trace.events_for(tenant.id)
            if not t_events:
                continue
            slo = tenant.workload.slo
            t_trace = Trace(events=t_events)
            slo_res = slo.apply(t_trace, tenant.workload)
            slo_section[tenant.id] = dict(slo_res.diagnostics)
            gate_failed = gate_failed or not slo_res.aggregate_passed
            t_credits: list[float] = []
            for unit_id, unit_events in slo.units(t_trace).items():
                key = event_key(tenant.id, unit_id)
                units[key] = unit_events
                r = slo_res.per_event_ratio.get(unit_id)
                if r is None:
                    gate_failed = True    # a unit with no r == a failed request
                    # Price it for the bound only: what it would have been worth had
                    # it answered as the harness stopped watching it. None when the SLO
                    # defines no ratio at all, and then the unit is in no bound either.
                    r_censored = slo.censored_ratio(unit_events, t_end)
                    if r_censored is not None:
                        censored[key] = _log_credit(r_censored)
                else:
                    log_credits[key] = _log_credit(r)
                    t_credits.append(log_credits[key])
                    priced.update(ev.event_key for ev in unit_events)
                # The SLO's credit is per request, so a unit is met when every request
                # in it is: one request at request grain, a trajectory's turns at
                # session grain.
                met[key] = min(slo_res.per_event_credit.get(ev.request_id, 0.0)
                               for ev in unit_events)
            # The tenant's own contribution to the score. Derived here, not by the SLO:
            # the SLO owns the normalization (r), this scorer owns the objective's shape
            # (the log), so there is one definition of the credit and no chance of the
            # two drifting. It is the per-tenant read the arithmetic mean of r in the
            # SLO's own diagnostics cannot give, the score being a geometric mean.
            slo_section[tenant.id]["mean_log_credit"] = _mean_credit(t_credits)

        reasons: list[str] = []
        if not verdict.passed:
            # Same reason plumbing as the goodput member: the verdict's failure_mode
            # ("correctness" vs "failed_requests") is the reason, with "correctness" as
            # the fallback for verdicts that never carried one.
            reasons.append(verdict.failure_mode or "correctness")

        # The hard gate. Any failed/timed-out/errored/never-streamed request -- or any
        # event no tenant's SLO scored -- fails the whole run, rather than being dropped
        # from the mean (which would reward shedding the hard requests). An event is
        # priced when its own UNIT produced a measured ratio, so this covers both a unit
        # with no r and an event belonging to no declared tenant. Membership is tested
        # against ``priced`` rather than ``log_credits``: at session grain the credit map
        # is keyed by unit, so an event key would never be found in it and every
        # trajectory run would gate.
        if gate_failed or any(ev.event_key not in priced for ev in trace.events):
            reasons.append("slo")

        if reasons:
            # Both sections on every gated return: the latency this run measured and what
            # the verifiers said about it, whichever layer zeroed it.
            return Score.zero(reasons=reasons,
                              breakdown={"slo": slo_section,
                                         "verdict": dict(verdict.diagnostics),
                                         "upper_bound": self._bound(log_credits, censored)})

        # Same contract as the goodput member's: mean per-unit SLO attainment (here the
        # binary r < 1 credit) combined with the per-event verdict, the same saturation
        # signal. A unit counts only when every
        # request in it passed the verdict. Diagnostic only -- it is not the score.
        slo_attainment = sum(
            met[key]
            for key, unit_events in units.items()
            if all(verdict.per_event.get(ev.event_key, False) for ev in unit_events)
        ) / len(units)

        mean_log_credit = _mean_credit(log_credits.values())
        b = self.baseline_log_credit
        value = self._speedup(mean_log_credit)
        return Score(
            value=value,
            # Split the same way for both scorers: what this artifact measured over what
            # it is normalized against. Here, its own geometric mean of 1/r over the
            # baseline stack's -- so achieved is the speedup against the SLO itself, and
            # sol is 1.0 while no baseline has been measured.
            achieved=math.exp(mean_log_credit),
            sol=1.0 if b is None else math.exp(b),
            unit="slo_speedup",
            gate_passed=True,
            breakdown={
                "slo": slo_section,
                "slo_attainment": slo_attainment,
                # L and B, the two terms of the score, kept raw: L is what another drive
                # of the same stack is averaged with to measure a baseline.
                "mean_log_credit": mean_log_credit,
                "baseline_log_credit": b,
                "n_scored": len(log_credits),
                # Every unit was measured to get here, so there is nothing censored
                # to price and the bound is the score itself.
                "upper_bound": value,
                "verdict": dict(verdict.diagnostics),
            },
        )
