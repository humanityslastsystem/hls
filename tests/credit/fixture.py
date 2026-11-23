"""Tasks for the baseline credit tests.

``credit_task()`` is the ``hls evaluate`` tests' task (``tests.evaluate.fixture``) scored as a
speedup over its baseline: its one tenant held to a per-request latency SLO, which the fake
server meets by far, under a ``SloLatencyScorer``, and its baseline the fake server in its
serving mode. It keeps that task's id, model, corpus, contract, hardware and drive bounds, so
the fake hub and the image catalog serve it as they serve that task."""

from __future__ import annotations

from hls.scoring import SloLatencyScorer
from hls.spec.baseline import Baseline
from hls.spec.taskspec import TaskSpec
from hls.spec.workload import AllAtOnce, NoDelay, RequestLatencySLO, Uncapped, Workload
from tests.evaluate import fixture as evaluate

#: The serve command the credit task's baseline declares: the fake server serving.
BASELINE_SERVE = ["--mode", "serve"]
#: The latency every request is held to, which the fake server meets by far.
TARGET_MS = 60_000.0
#: The baseline the credit task declares.
BASELINE = Baseline(image="hls-baseline-evaluate-task", serve_command=BASELINE_SERVE)
#: The same baseline told to exit before it is ready, as a server that does not fit does.
CRASHING = Baseline(image="hls-baseline-evaluate-task", serve_command=["--mode", "crash"])


def credit_task(*, baseline: Baseline | None = BASELINE, latency: bool = True, stream: bool = False) -> TaskSpec:
    """The evaluate tests' task, scored as a speedup over *baseline*; with *latency* False it
    keeps that task's goodput scorer, which scores against no baseline; with *stream* its replies
    stream, each token id a packet timestamp, as multitenant_slo's do."""
    base = evaluate.evaluate_task()
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(), output_length_cap=Uncapped(),
                        slo=RequestLatencySLO(target_ms=TARGET_MS))
    contract = base.contract.model_copy(update={"transport": "http-sse"}) if stream else base.contract
    return base.model_copy(update={"workload": workload, "baseline": baseline, "contract": contract,
                                   "scorer": SloLatencyScorer() if latency else base.scorer})
