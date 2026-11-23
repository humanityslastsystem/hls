"""Tasks and baseline credits for the score and finalize stages' tests.

The tasks score the drives of ``tests/trace/traces.py`` and the single-tenant events of
``tests/unit/test_scorers.py`` with the harness's scorers. Each tenant also gets a letter
match, which a task must declare and no scorer reads. ``conftest.py`` writes each task into
a checkout of its own, beside the baseline credit it scores against."""

from __future__ import annotations

import math

from hls.baseline_credit import BaselineCredit, BaselineDrive, BaselineRecord, CampaignIdentity, load_record
from hls.core.model import ModelSpec
from hls.core.trace import Trace
from hls.scoring import ConstantSoL, GoodputScorer, RequestUnit, SloLatencyScorer
from hls.spec.baseline import Baseline
from hls.spec.contracts import RequestResponseContract
from hls.spec.taskspec import TaskSpec
from hls.spec.tenant import Tenant
from hls.spec.wire import OpenAIChatProtocol
from hls.verify import LetterMatchMetric, MetricVerifier
from hls.verify.calibration import workload_fingerprint
from tests.trace.traces import HARDWARE, LATENCY_TENANTS, TENANTS
from tests.unit.test_scorers import _at_r, _dead, _hw, _slo_wl, _unsent

MODEL = ModelSpec(base_checkpoint="m", arch="TX")
CONTRACT = RequestResponseContract(path="/v1/chat/completions", transport="http-sse",
                                   input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
                                   protocol=OpenAIChatProtocol())
BASELINE = Baseline(image="hls-baseline", serve_command=["--flag", "1"])


def _checked(tenants: list[Tenant]) -> list[Tenant]:
    return [t.model_copy(update={"verifiers": [MetricVerifier(metric=LetterMatchMetric())]}) for t in tenants]


def goodput_task(unit=None) -> TaskSpec:
    """The three tenants of ``drive_trace`` under a goodput scorer counting *unit*."""
    scorer = GoodputScorer(unit=unit or RequestUnit(), sol=ConstantSoL(rps=2.0))
    return TaskSpec(id="fin-goodput", model=MODEL, hardware=HARDWARE, contract=CONTRACT, scorer=scorer,
                    tenants=_checked(TENANTS))


def latency_task() -> TaskSpec:
    """The two latency tenants of ``drive_trace``, scored as a speedup over the baseline."""
    return TaskSpec(id="fin-latency", model=MODEL, hardware=HARDWARE, contract=CONTRACT, scorer=SloLatencyScorer(),
                    tenants=_checked(LATENCY_TENANTS), baseline=BASELINE)


def t0_task() -> TaskSpec:
    """``test_scorers``'s tenant t0 on a 100 ms TTFT budget, scored as a speedup."""
    tenant = Tenant(id="t0", workload=_slo_wl(), verifiers=[MetricVerifier(metric=LetterMatchMetric())])
    return TaskSpec(id="fin-t0", model=MODEL, hardware=_hw(), contract=CONTRACT, scorer=SloLatencyScorer(),
                    tenants=[tenant], baseline=BASELINE)


def t0_single_task() -> TaskSpec:
    """``t0_task`` authored as a single tenant, at the top level, as most published tasks are."""
    return TaskSpec(id="fin-single", model=MODEL, hardware=_hw(), contract=CONTRACT,
                    scorer=SloLatencyScorer(), workload=_slo_wl(),
                    verifiers=[MetricVerifier(metric=LetterMatchMetric())], baseline=BASELINE)


def vacuous_task() -> TaskSpec:
    """Tenant t0 with no TPOT budget (an infinite one), under a goodput scorer: a task whose
    spec holds a number strict JSON cannot."""
    workload = _slo_wl(tpot_ms=math.inf)
    tenant = Tenant(id="t0", workload=workload, verifiers=[MetricVerifier(metric=LetterMatchMetric())])
    return TaskSpec(id="fin-vacuous", model=MODEL, hardware=_hw(), contract=CONTRACT,
                    scorer=GoodputScorer(unit=RequestUnit(), sol=ConstantSoL(rps=2.0)), tenants=[tenant])


def t0_trace(*, failures: bool) -> Trace:
    """Requests of t0 at known latency ratios; with *failures*, one errored and one the
    breaker never sent, which gate the run and leave only its upper bound."""
    events = [_at_r("ok0", 0.5), _at_r("ok1", 2.0), _at_r("ok2", 0.8)]
    if failures:
        events += [_dead("bad"), _unsent("skip", due=0.5, skipped_at=0.9)]
    return Trace(events=events)


def credit(task: TaskSpec, module: str, credits: tuple[float, ...] = (0.9, 1.1), *,
           gpu_kind: str | None = None) -> BaselineCredit:
    """A baseline credit measured for *task*, or on *gpu_kind* instead of its GPU."""
    server = BaselineRecord(image=BASELINE.image, serve_command=list(BASELINE.serve_command))
    campaign = CampaignIdentity(task=module, workload_fingerprint=workload_fingerprint(task),
                                load=load_record(task), gpu_kind=gpu_kind or task.hardware.accelerator.kind,
                                baseline=server)
    drives = [BaselineDrive(started_at=f"2026-09-05T00:{i + 1:02d}:00+00:00", duration_sec=60.0, n_requests=10,
                            n_units=10, mean_log_credit=c) for i, c in enumerate(credits)]
    return BaselineCredit(campaign=campaign, drives=drives, created_at="2026-09-05T00:10:00+00:00")
