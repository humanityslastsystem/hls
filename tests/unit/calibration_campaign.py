"""A calibration's stages composed in one process, for the unit tests of ``hls.calibrate``: the
role functions of ``hls.verify.calibration_roles`` over one capture store and an injected deploy
seam, in the order the pipeline runs them, the grower's round rule
(``hls.pipeline.calibration_grower.round_servings``) deciding each round of servings. It keeps
nothing the roles do not do: every campaign starts from an empty store."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from hls.calibrate import (DEFAULT_DELTA, DEFAULT_EPS, DEFAULT_MU_PER_MTOK, CaptureStore, RowIndex,
                           check_campaign, engine_cache_config, load_sessions, member_engines,
                           member_internet, pinned_reference)
from hls.pipeline.calibration import SCORE_PER_SERVING_SEC
from hls.pipeline.calibration_grower import round_servings
from hls.pipeline.pipelines import READINESS_TIMEOUT_SEC, REQUEST_TIMEOUT_SEC
from hls.spec.datasets import Split
from hls.spec.resources import ResourceShape
from hls.spec.taskspec import TaskSpec
from hls.verify.calibration import DEFAULT_EOT_EXPECTED_MAX, Calibration, ReportCalibration, workload_fingerprint
from hls.verify.calibration_roles import CampaignSettings, aggregate, capture, instrument, score

#: The bound on servings these campaigns are given, which no test reaches: in a plan the grower
#: holds an engine to ``max_servings_per_engine``.
MAX_SERVINGS = 1000


async def in_process_campaign(task: TaskSpec, module: str, *, engines: Sequence[str], deploy: Any,
                              rows: Path, captures: Path, servings_per_engine: int | None = None,
                              sessions_per_draw: int | None = None, eps: float = DEFAULT_EPS,
                              mu_per_mtok: float = DEFAULT_MU_PER_MTOK, delta: float = DEFAULT_DELTA,
                              eot_expected_max: float = DEFAULT_EOT_EXPECTED_MAX,
                              fanout: int = 1) -> Calibration | ReportCalibration:
    """*task*'s calibration, the ``TASK`` of *module*, by the pool *engines* deployed through
    *deploy*, its sessions read from the provisioned *rows* (``load_sessions``) and its captures kept in
    the store at *captures*:
    ``check_campaign``, the instrument check, then the rounds of servings ``round_servings``
    takes at *fanout*, each round's servings captured one after another and then scored as one
    share, then the aggregate, whose warnings are in the log only. Images and shapes are
    placeholders: nothing here loads an image."""
    kind, servings = check_campaign(task, engines=engines, servings_per_engine=servings_per_engine)
    shape = ResourceShape(gpus=1, gpu_kind=task.hardware.accelerator.kind, cpus=8, memory_gib=176, time_sec=3600)
    campaign = CampaignSettings(
        kind=kind, workload_fingerprint=workload_fingerprint(task), task_seed=task.seed, engines=list(engines),
        member_internet=member_internet(task),
        eps=eps, mu_per_mtok=mu_per_mtok, delta=delta, eot_expected_max=eot_expected_max,
        sessions_per_draw=sessions_per_draw, servings_per_engine=servings, max_servings_per_engine=MAX_SERVINGS,
        fanout=fanout, member_images={e: f"member-{e}" for e in engines}, reference_image="reference",
        member_engine_configs={e: engine_cache_config(member_engines(task)[e]()) for e in engines},
        reference_engine_config=engine_cache_config(pinned_reference(task)),
        member_shape=shape, reference_shape=shape, request_timeout_sec=REQUEST_TIMEOUT_SEC,
        load_timeout_sec=READINESS_TIMEOUT_SEC, score_per_serving_sec=SCORE_PER_SERVING_SEC)
    store = CaptureStore(captures, campaign.workload_fingerprint)
    sessions = load_sessions(task, rows, Split.TEST)
    index = RowIndex.from_sessions(sessions)
    scored: dict[str, dict[int, int]] = {e: {} for e in engines}
    record = await instrument(task, kind, deploy)
    share = 0
    while True:
        tokens = {e: [n for _, n in sorted(scored[e].items())] for e in engines}
        taken = round_servings(campaign, tokens, {e: max(scored[e], default=-1) + 1 for e in engines})
        if not taken:
            return aggregate(task, module, campaign, record, store, scored)[0]
        for serving in taken:
            await capture(task, serving, kind=kind, sessions=sessions, sessions_per_draw=sessions_per_draw,
                          request_timeout_sec=campaign.request_timeout_sec,
                          drive_timeout_sec=campaign.drive_timeout_sec, store=store, deploy=deploy)
        share += 1
        for stats in await score(task, taken, kind=kind, share=share, store=store, rows=index, deploy=deploy):
            scored[stats.engine][stats.draw] = stats.n_tokens
