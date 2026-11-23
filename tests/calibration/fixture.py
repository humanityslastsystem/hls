"""Tasks for the calibration tests, and a campaign's settings for the stage and role tests.

Each task is the grading tests' task (``tests.evaluate.fixture``) under one calibrated
gate, keeping its id, model, corpus and hardware, so the fake hub, the image catalog and the
in-process Slurm serve it as they serve that task:
- ``flips_task()``: the token-level gate, a ``TokenFaithfulnessVerifier`` on a greedy HF
  reference;
- ``report_task()``: the sampling gate, a ``ReportFaithfulnessVerifier`` on a sampling HF
  reference, under a contract whose replies report their log-probabilities;
- ``capped_task()``: ``flips_task()`` with a ``CAP_TOKENS`` output cap, which every reply of the
  fake member reaches;
- ``agentic_task()``: ``flips_task()`` with an agent seam whose tool loads no provisioned
  assets, so its pool members run with the network open, and whose drive wall is
  ``DRIVE_WALL_SEC``;
- ``two_reports_task()``: ``report_task()`` with a second report verifier;
- ``two_chunks_task()``: ``flips_task()`` with a second token-level verifier, whose reference
  teacher-forces at another query chunk;
- ``audio_task()``: the TTS tests' speaking task, under this id;
- ``plain_task()``: the evaluate task itself, whose verifier is not calibrated.

``cal_capped``'s module also defines ``record_removed_rows`` (``record_removed_rows`` here),
which notes what it is handed."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hls.calibrate import (DEFAULT_DELTA, DEFAULT_EPS, DEFAULT_MU_PER_MTOK, engine_cache_config,
                           member_engines, member_internet, pinned_reference)
from hls.core.decode import Greedy, Sample
from hls.pipeline.calibration import SCORE_PER_SERVING_SEC
from hls.pipeline.pipelines import READINESS_TIMEOUT_SEC, REQUEST_TIMEOUT_SEC
from hls.recipes.catalog import ImageSet
from hls.reference.references import HFReference
from hls.spec.agent import AgentSpec
from hls.spec.contracts import RequestResponseContract
from hls.spec.resources import ResourceShape
from hls.spec.taskspec import TaskSpec
from hls.spec.wire import OpenAIChatProtocol
from hls.spec.workload import AllAtOnce, NoDelay, NoSLO, ShortOutputCap, Workload
from hls.verify import ReportFaithfulnessVerifier, TokenFaithfulnessVerifier
from hls.verify.calibration import DEFAULT_EOT_EXPECTED_MAX, calibration_kind, workload_fingerprint
from hls.verify.calibration_roles import CampaignSettings
from hls.verify.report import TOP_LOGPROBS_REQUESTED
from tests.evaluate import fixture as evaluate
from tests.unit import test_calibrate_tts as tts
from tests.unit.test_trajectory_driver import _agent

#: The fixture tasks' id: the evaluate tests' task's.
TASK_ID = evaluate.TASK_ID
#: The output cap of ``capped_task()``, in tokens.
CAP_TOKENS = 8
#: The drive wall of ``agentic_task()``'s agent seam, in seconds.
DRIVE_WALL_SEC = 5400.0
#: The test split's rows, as a role reads them under ``/data``: four sessions of one tenant.
TEST_ROWS = [{"q": str(i)} for i in range(1, 5)]
#: The measured roles the catalog lists for the fixture task unless a test lists others.
ROLES = ("reference", "member-vllm", "member-sglang")


def image_set(roles: Sequence[str] = ROLES) -> ImageSet:
    """The fixture tasks' images: on sample_task's layer, a server image for each of *roles*."""
    from tests.recipes.declared import image_set

    return image_set(TASK_ID, roles)
#: A role's shape in the stage and role tests: one B200, four CPUs and 16 GiB.
ROLE_SHAPE = ResourceShape(gpus=1, gpu_kind="B200", cpus=4, memory_gib=16, time_sec=600)
#: What ``record_removed_rows`` was handed, one mapping of tenant to rows per call.
RECORDED: list[dict[str, tuple[str, ...]]] = []


def _with(verifiers: list, *, contract: RequestResponseContract | None = None,
          workload: Workload | None = None, agent: AgentSpec | None = None) -> TaskSpec:
    """The evaluate task with *verifiers*, and *contract*, *workload* and *agent* when given."""
    base = evaluate.evaluate_task()
    return TaskSpec(id=base.id, model=base.model, hardware=base.hardware, contract=contract or base.contract,
                    data_source=base.data_source, seed=base.seed, workload=workload or base.workload,
                    verifiers=verifiers, scorer=base.scorer, agent=agent, drive_bounds=base.drive_bounds)


def flips_task() -> TaskSpec:
    return _with([TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))])


#: The contract of ``report_task()``: its replies report their log-probabilities.
REPORT_CONTRACT = RequestResponseContract(
    path="/v1/chat/completions", input_modalities=frozenset({"text"}), output_modalities=frozenset({"text"}),
    protocol=OpenAIChatProtocol(return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED))


def report_task() -> TaskSpec:
    return _with([ReportFaithfulnessVerifier(reference=HFReference(decode=Sample(temperature=1.0)))],
                 contract=REPORT_CONTRACT)


def two_reports_task() -> TaskSpec:
    return _with([ReportFaithfulnessVerifier(reference=HFReference(decode=Sample(temperature=1.0)))] * 2,
                 contract=REPORT_CONTRACT)


def two_chunks_task() -> TaskSpec:
    reference = HFReference(decode=Greedy())
    wider = reference.model_copy(update={"query_chunk": 2 * reference.query_chunk})
    return _with([TokenFaithfulnessVerifier(reference=reference), TokenFaithfulnessVerifier(reference=wider)])


def capped_task() -> TaskSpec:
    workload = Workload(arrival=AllAtOnce(), inter_turn_delay=NoDelay(),
                        output_length_cap=ShortOutputCap(cap_tokens=CAP_TOKENS), slo=NoSLO())
    return _with([TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))], workload=workload)


def agentic_task() -> TaskSpec:
    return _with([TokenFaithfulnessVerifier(reference=HFReference(decode=Greedy()))],
                 agent=dataclasses.replace(_agent(), drive_timeout_sec=DRIVE_WALL_SEC))


def audio_task() -> TaskSpec:
    return tts._task(audio=True).model_copy(update={"id": TASK_ID})


def plain_task() -> TaskSpec:
    return evaluate.evaluate_task()


def record_removed_rows(rows_by_tenant: Mapping[str, tuple[str, ...]]) -> tuple[Path, dict[str, int]]:
    """A task's recorder of its removal list: notes *rows_by_tenant* and says every row was
    added."""
    RECORDED.append(dict(rows_by_tenant))
    return Path("removed_requests.json"), {tenant: len(rows) for tenant, rows in rows_by_tenant.items()}


def campaign(task: TaskSpec | None = None, **overrides: Any) -> CampaignSettings:
    """The settings ``hls calibrate`` gives *task*'s campaign (``flips_task()``'s by default),
    with *overrides*: the gate the task measures, its fingerprint and seed, both pool members on
    the catalog's image names and on the network the task's tools need, the default
    tolerances, a serving per engine for the sampling gate, at most 24 servings, a fanout of 2,
    the engines' own engine-cache configs, the plan's load bounds and ``ROLE_SHAPE`` for every
    role."""
    task = task or flips_task()
    kind = calibration_kind(task)
    fields: dict[str, Any] = dict(
        kind=kind, workload_fingerprint=workload_fingerprint(task), task_seed=task.seed, engines=["vllm", "sglang"],
        member_internet=member_internet(task),
        eps=DEFAULT_EPS, mu_per_mtok=DEFAULT_MU_PER_MTOK, delta=DEFAULT_DELTA,
        eot_expected_max=DEFAULT_EOT_EXPECTED_MAX, servings_per_engine=1 if kind == "report" else None,
        max_servings_per_engine=24, fanout=2, reference_image="evaluate-task-reference", member_shape=ROLE_SHAPE,
        reference_shape=ROLE_SHAPE, request_timeout_sec=REQUEST_TIMEOUT_SEC, load_timeout_sec=READINESS_TIMEOUT_SEC,
        score_per_serving_sec=SCORE_PER_SERVING_SEC)
    fields.update(overrides)
    fields.setdefault("member_images", {e: f"evaluate-task-member-{e}" for e in fields["engines"]})
    fields.setdefault("member_engine_configs", {e: engine_cache_config(member_engines(task)[e]())
                                                for e in fields["engines"]})
    fields.setdefault("reference_engine_config", engine_cache_config(pinned_reference(task)))
    return CampaignSettings(**fields)
