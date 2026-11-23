"""The harness-conditions table (``hls.obs.conditions``): what a record says about how its
instance ran, read from the plan and the store."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

from hls.obs.conditions import (AllocationNote, AttemptNote, BudgetNote, StageAttempts, hardware_check,
                                harness_conditions)
from hls.obs.record import GradingRecord, RolloutRecord
from hls.pipeline import commands
from hls.pipeline.plan import Allocation, AllocationStatus, AttemptStatus, Failure, Plan, Submission
from hls.pipeline.stage import StageFault
from hls.spec.resources import ArtifactPlacement, BaselineHardware, GrantedResources, HardwareRecord
from hls.stages.agent_build import AgentBuildParams, budget_params
from hls.stages.finalize import FinalizeStage, budget_note
from hls.stages.score import ScoreStage
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord
from hls.substrate.versions import ToolVersions
from hls.taskref import blob_sha
from hls.verify.role import CalibrationBlob, VerifyRecord
from tests.finalize.published import (IMAGES, context, make_plan, publish, publish_instance, run_and_publish,
                                      write)
from tests.credit.test_credit_stages import HARDWARE, ran_on
from tests.finalize import fixture
from tests.finalize.test_agent_build_scan import PARAMS
from tests.pipeline.plans import VERSIONS
from tests.support.credit import write_credit
from tests.trace.traces import LATENCY_TENANTS, drive_trace, passing_verdict

NODE_LOST = FaultRecord(fault_class="infra", reason="the node lost its scratch", retryable=True)


def _status(stage: str, n: int, allocation: int, job: str, state: str, fault: FaultRecord | None) -> bytes:
    return AttemptStatus(stage=stage, attempt=n, allocation=allocation, submission=1, job_id=job, state=state,
                         started_at=10.0 * n, ended_at=10.0 * n + 5, fault=fault, no_retry=None,
                         manifest_sha256=None).model_dump_json().encode()


def _allocation_status(plan: Plan, k: int, job: str, versions: ToolVersions | None,
                       grant: GrantedResources | None = None) -> bytes:
    """The status a runner of job *job*, submission 1 of allocation *k*, wrote once it ended,
    with the tool *versions* and the *grant* its grant check measured (None before it)."""
    return AllocationStatus(instance=plan.instance, allocation=k, submission=1, job_id=job, node="n047",
                            state="succeeded", started_at=1.0, heartbeat_at=2.0, end_time=None, stage=None,
                            attempt=None, grant=grant, tool_versions=versions, swap_max_bytes=None,
                            memory_enforced_by=None, fault=None, ended_at=3.0).model_dump_json().encode()


def test_the_conditions_name_every_attempt_and_retry(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_goodput.task", names_published=False)
    store = PosixStore(plan.store_root)
    first = plan.allocations[0]
    sub = Submission(n=1, tag="t0", intended_at=1.0, after_job=None, status="submitted", job_id="100")
    failed = first.model_copy(update={"submissions": [sub],
                                      "failure": Failure(at=2.0, stage="deploy_drive", fault=NODE_LOST)})
    retry = Allocation(index=1, stages=["deploy_drive", "score", "finalize"], request=first.request,
                       placement=first.placement.model_copy(update={"exclude": ["n047"]}), depends_on=None,
                       retry_of=0, submissions=[sub.model_copy(update={"tag": "t1", "job_id": "101"})])
    plan = plan.with_allocation(failed).with_allocation(retry)
    layout = plan.layout
    store.write_bytes(layout.attempt_file("deploy_drive", 1, "status.json"),
                      _status("deploy_drive", 1, 0, "100", "failed", NODE_LOST))
    store.write_bytes(layout.attempt_file("deploy_drive", 2, "status.json"),
                      _status("deploy_drive", 2, 1, "101", "succeeded", None))
    store.write_bytes(layout.attempt_file("deploy_drive", 3, "status.json"), b"{not a status")
    store.write_bytes(layout.attempt_file("score", 1, "CLAIM"), b"")
    # The first job's runner recorded its node's tools with its grant; the retry's wrote no status.
    grant = GrantedResources(gpus=1, gpu_kind="H200", gpu_model="NVIDIA H200", cpus=8, cpu_model="Example CPU 9000",
                             memory_gib=180.0, disk_free_gib=100.0)
    store.write_bytes(layout.allocation_status(0, 1), _allocation_status(plan, 0, "100", VERSIONS, grant))
    drive = write(store, plan, "deploy_drive", "drive.json", b"{}", attempt=2)
    manifest = publish(store, plan, "deploy_drive", {"drive": drive}, attempt=2)
    conditions = harness_conditions(store, plan, published=[manifest], credit=None,
                                    calibration=None, arm=None, budget=None, placement=None, drive_stage=None, grade_after_sandbox_oom=None)
    # The attempt that counted is the published manifest's, not the last one claimed.
    assert conditions.attempts["deploy_drive"] == StageAttempts(counted=2, attempts=[
        AttemptNote(attempt=1, state="failed", fault_class="infra", reason=NODE_LOST.reason, retryable=True),
        AttemptNote(attempt=2, state="succeeded"), AttemptNote(attempt=3, state="unreadable")])
    # An attempt claimed with no status yet is named as such; a stage with none has none.
    assert conditions.attempts["score"] == StageAttempts(counted=None, attempts=[AttemptNote(attempt=1,
                                                                                             state="claimed")])
    assert conditions.attempts["finalize"] == StageAttempts(counted=None, attempts=[])
    assert list(conditions.attempts) == ["deploy_drive", "score", "finalize"]
    assert conditions.allocations == [
        AllocationNote(index=0, retry_of=None, job_ids=["100"], exclude=[], failure=f"deploy_drive: {NODE_LOST.reason}",
                       tool_versions={"100": VERSIONS}, hardware={"100": HARDWARE}),
        AllocationNote(index=1, retry_of=0, job_ids=["101"], exclude=["n047"], failure=None,
                       tool_versions={"101": None}, hardware={"101": None})]
    assert (conditions.hls_commit, conditions.harbor_commit, conditions.dirty) == ("a" * 40, "b" * 40, False)
    # No drive named and no credit, so no hardware compared.
    assert (conditions.hardware, conditions.baseline_hardware, conditions.hardware_warning) == (None, None, None)
    assert conditions.images == IMAGES["deploy_drive"]


@pytest.mark.parametrize("kind", ["unreadable", "another-job", "before-the-grant-check"])
def test_a_job_whose_status_records_no_versions_of_its_own_has_none(tmp_path, checkout, kind):
    """M-17: the versions are a job's own runner's, read from its submission's status: a status that
    does not parse, one another job wrote, or one written before the grant check names none."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task", names_published=False)
    sub = Submission(n=1, tag="t0", intended_at=1.0, after_job=None, status="submitted", job_id="100")
    plan = plan.with_allocation(plan.allocations[0].model_copy(update={"submissions": [sub]}))
    status = {"unreadable": b"{not a status", "another-job": _allocation_status(plan, 0, "999", VERSIONS),
              "before-the-grant-check": _allocation_status(plan, 0, "100", None)}[kind]
    store = PosixStore(plan.store_root)
    store.write_bytes(plan.layout.allocation_status(0, 1), status)
    conditions = harness_conditions(store, plan, published=[], credit=None,
                                    calibration=None, arm=None, budget=None, placement=None, drive_stage=None, grade_after_sandbox_oom=None)
    assert conditions.allocations[0].tool_versions == {"100": None}


def test_a_job_the_plan_does_not_know_yet_is_named_by_its_runners_status(tmp_path, checkout):
    """A runner may start before the orchestrator records its job's id, so finalize's copy of the
    plan can lack its own job: its runner's status names the job, and its versions are recorded
    under it. A submission with no job and no status has nothing to record."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task", names_published=False)
    sub = Submission(n=1, tag="t0", intended_at=1.0, after_job=None, status="intended")
    store = PosixStore(plan.store_root)
    plan = plan.with_allocation(plan.allocations[0].model_copy(update={"submissions": [sub]}))
    note = harness_conditions(store, plan, published=[], credit=None,
                              calibration=None, arm=None, budget=None, placement=None, drive_stage=None, grade_after_sandbox_oom=None).allocations[0]
    assert (note.job_ids, note.tool_versions) == ([], {})
    store.write_bytes(plan.layout.allocation_status(0, 1), _allocation_status(plan, 0, "100", VERSIONS))
    note = harness_conditions(store, plan, published=[], credit=None,
                              calibration=None, arm=None, budget=None, placement=None, drive_stage=None, grade_after_sandbox_oom=None).allocations[0]
    assert (note.job_ids, note.tool_versions) == ([], {"100": VERSIONS})


def test_a_dirty_checkout_is_named(tmp_path, checkout):
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    plan = plan.model_copy(update={"code": plan.code.model_copy(update={"dirty_diff_sha256": "d" * 64})})
    conditions = harness_conditions(PosixStore(plan.store_root), plan, published=[],
                                    credit=None, calibration=None, arm=None, budget=None, placement=None,
                                    drive_stage=None, grade_after_sandbox_oom=None)
    assert conditions.dirty is True


#: The calibration file a fixture verify role recorded it gave the verifiers.
BLOB = CalibrationBlob(blob_sha="c" * 40, kind="flips", workload_fingerprint="f" * 64)


def test_the_conditions_name_the_credit_and_calibration_blobs(tmp_path, checkout):
    """The credit is the one the score role recorded, and the calibration the one the verify role
    recorded it gave the verifiers, each read from its stage's published records, never from the
    checkout again."""
    trace = drive_trace(failures=False, tenants=LATENCY_TENANTS)
    plan = make_plan(tmp_path, checkout, "fin_latency.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace),
                     verify_record=VerifyRecord(eval_id="terminal", engines=[], libraries={}, calibration=BLOB))
    run_and_publish(store, plan, "score", ScoreStage)
    outputs = FinalizeStage().run(context(store, plan, "finalize", secrets={}))
    conditions = RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"])).conditions
    provenance = json.loads(store.read_bytes(plan.layout.attempt_file("score", 1, "outputs/score-provenance.json")))
    assert conditions.credit == provenance["baseline"]
    assert conditions.credit["blob_sha"] == blob_sha(checkout / "fin_latency" / "defaults" / "baseline.json")
    assert conditions.calibration == BLOB.model_dump(mode="json")


#: Another CPU model than the fixture drives' (``HARDWARE``).
OTHER_CPU = HARDWARE.model_copy(update={"cpu_model": "Another CPU 1"})
#: What a baseline credit measured on the fixture drives' hardware records, every field.
MEASURED = BaselineHardware.of(HARDWARE)
#: A baseline credit's record of the GPU model alone, as the shipped tasks' credits record it.
GPU_ONLY = BaselineHardware(gpu_model="NVIDIA H200")


def test_the_hardware_check_warns_of_each_recorded_field_that_differs():
    """The same hardware is said nothing of; another GPU model, GPU count or
    CPU model the baseline records is a warning naming both sides; a field the baseline leaves
    blank is not compared and says nothing; a baseline that ran on several records is compared
    with each; hardware recorded on neither side says nothing."""
    assert hardware_check(HARDWARE, [MEASURED]) is None
    assert hardware_check(OTHER_CPU, [MEASURED]) == (
        "warning: this drive ran on other hardware than its baseline credit's drives, so its score compares the two: "
        "cpu_model Another CPU 1 (baseline Example CPU 9000)")
    other = HARDWARE.model_copy(update={"gpu_model": "NVIDIA H100 80GB HBM3", "gpus": 2})
    assert "gpu_model NVIDIA H100 80GB HBM3 (baseline NVIDIA H200), gpus 2 (baseline 1)" in hardware_check(
        other, [MEASURED])
    # The shipped credits record the GPU model alone: only it is compared.
    assert hardware_check(OTHER_CPU, [GPU_ONLY]) is None
    assert hardware_check(other.model_copy(update={"gpus": 1}), [GPU_ONLY]) == (
        "warning: this drive ran on other hardware than its baseline credit's drives, so its score compares the two: "
        "gpu_model NVIDIA H100 80GB HBM3 (baseline NVIDIA H200)")
    assert "the baseline's drives ran on 2 hardware records; against 1 x NVIDIA H200, CPU Another CPU 1: cpu_model " \
           "Example CPU 9000 (baseline Another CPU 1)" in hardware_check(HARDWARE, [MEASURED,
                                                                          BaselineHardware.of(OTHER_CPU)])
    assert hardware_check(HARDWARE, None) is None
    assert hardware_check(None, [MEASURED]) is None


def _graded(tmp_path: Path, checkout: Path, drive: HardwareRecord, baseline: list[BaselineHardware] | None):
    """Finalize's records of a grading of ``fin_latency`` whose drive ran on *drive*, scored
    against the fixture credit recording *baseline*, the plan whose finalize published them, and
    its score. The credit is written before the plan pins it, and the checkout's is restored after."""
    path = checkout / "fin_latency" / "defaults" / "baseline.json"
    committed = path.read_bytes()
    trace = drive_trace(failures=False, tenants=LATENCY_TENANTS)
    try:
        task = importlib.import_module("fin_latency.task").TASK
        write_credit(path.parent, fixture.credit(task, "fin_latency.task").model_copy(update={"hardware": baseline}))
        plan = make_plan(tmp_path, checkout, "fin_latency.task")
        store = PosixStore(plan.store_root)
        publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace))
        ran_on(store, plan, "deploy_drive", drive)
        run_and_publish(store, plan, "score", ScoreStage)
        outputs = FinalizeStage().run(context(store, plan, "finalize", secrets={}))
    finally:
        path.write_bytes(committed)
    publish(store, plan, "finalize", outputs)
    records = (RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"])),
               GradingRecord.model_validate_json(store.read_bytes(outputs["grading"])))
    return records, plan, json.loads(store.read_bytes(plan.layout.attempt_file("score", 1, "outputs/score.json")))


def test_a_grading_on_the_baselines_hardware_records_it_and_warns_of_nothing(tmp_path, checkout):
    """The record states the hardware the drive ran on, every job's, and
    the hardware the credit's drives ran on, with no warning when they match."""
    records, plan, _ = _graded(tmp_path, checkout, HARDWARE, [MEASURED])
    for record in records:
        conditions = record.conditions
        assert (conditions.hardware, conditions.baseline_hardware) == (HARDWARE, [MEASURED])
        assert conditions.hardware_warning is None
        assert conditions.credit["hardware"] == [MEASURED.model_dump(mode="json")]
    said: list[str] = []
    commands._say_hardware(plan, said.append)
    # The plan pins the task's default credit, so the command says which default it used and warns of nothing.
    note = records[1].conditions.measured_note
    assert note is not None and said == [f"{plan.instance}: {note}"]


def test_a_grading_on_another_cpu_is_warned_of_and_scored_as_on_the_same(tmp_path, checkout):
    """A drive on another CPU model than the credit's drives is graded, its
    score exactly as on the same hardware; its record warns, naming both sides, and the command
    says the warning when the instance ends."""
    _, _, same = _graded(tmp_path / "same", checkout, HARDWARE, [MEASURED])
    records, plan, score = _graded(tmp_path / "other", checkout, OTHER_CPU, [MEASURED])
    assert score == same and score["value"] > 0
    warning = ("warning: this drive ran on other hardware than its baseline credit's drives, so its score compares "
               "the two: cpu_model Another CPU 1 (baseline Example CPU 9000)")
    for record in records:
        assert record.conditions.hardware_warning == warning
    assert records[1].episode.score == same
    said: list[str] = []
    commands._say_hardware(plan, said.append)
    assert said == [f"{plan.instance}: {warning}", f"{plan.instance}: {records[1].conditions.measured_note}"]


@pytest.mark.parametrize("baseline", [[GPU_ONLY], None], ids=["gpu-model-only", "none"])
def test_a_credit_that_leaves_the_differing_field_blank_says_nothing(tmp_path, checkout, baseline):
    """A credit that records the GPU model alone, as the shipped ones do, or
    no hardware, is not compared on the CPU model: a drive on another CPU gets no warning, and
    the record carries the credit's hardware as it is."""
    records, plan, _ = _graded(tmp_path, checkout, OTHER_CPU, baseline)
    for record in records:
        conditions = record.conditions
        assert (conditions.hardware, conditions.baseline_hardware, conditions.hardware_warning) == (
            OTHER_CPU, baseline, None)
    said: list[str] = []
    commands._say_hardware(plan, said.append)
    # The plan pins the task's default credit, so the command says which default it used and warns of nothing.
    assert said == [f"{plan.instance}: {records[1].conditions.measured_note}"]


def test_a_verify_record_changed_since_it_published_is_refused(tmp_path, checkout):
    """The calibration a record names is the one the verify stage published."""
    trace = drive_trace()
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace))
    run_and_publish(store, plan, "score", ScoreStage)
    key = plan.layout.attempt_file("verify", 1, "outputs/verify.json")
    store.write_bytes(key, VerifyRecord(eval_id="terminal", engines=[], libraries={},
                                        calibration=BLOB).model_dump_json().encode())
    with pytest.raises(StageFault) as e:
        FinalizeStage().run(context(store, plan, "finalize", secrets={}))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert f"{key} does not match its manifest" in e.value.fault.reason


def test_a_verify_that_recorded_no_calibration_names_none(tmp_path, checkout):
    """A task none of whose verifiers needs a calibration, or a drive that served nothing to verify,
    has a verify record with no calibration, and the conditions name none."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, outcome="crashed")
    run_and_publish(store, plan, "score", ScoreStage)
    outputs = FinalizeStage().run(context(store, plan, "finalize", secrets={}))
    assert GradingRecord.model_validate_json(store.read_bytes(outputs["grading"])).conditions.calibration is None


def test_the_conditions_name_where_the_scored_server_ran(tmp_path, checkout):
    """Where the site cannot ask for the server's CPUs on one socket, the record
    says where they lay, a split among them, as the drive recorded it."""
    split = ArtifactPlacement(cpus=[0, 1, 2, 3, 64, 65, 66, 67], numa_nodes=[0, 1], numa_split=True)
    trace = drive_trace()
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace), placement=split)
    run_and_publish(store, plan, "score", ScoreStage)
    outputs = FinalizeStage().run(context(store, plan, "finalize", secrets={}))
    conditions = GradingRecord.model_validate_json(store.read_bytes(outputs["grading"])).conditions
    assert conditions.placement == split


@pytest.mark.parametrize("graded", [True, False, None], ids=["graded", "zero", "no-build"])
def test_the_conditions_state_the_sandbox_oom_rule_the_build_ran_under(tmp_path, checkout, graded):
    """The table states the rule the instance's agent build ran under for a
    sandbox that ends on its memory, None for an instance that ran no build."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    conditions = harness_conditions(PosixStore(plan.store_root), plan, published=[], credit=None, calibration=None,
                                    arm=None, budget=None, placement=None, drive_stage=None,
                                    grade_after_sandbox_oom=graded)
    assert conditions.grade_after_sandbox_oom is graded


def test_budgets_other_than_the_tasks_own_are_named_overridden():
    """``hls run --soft-sec/--hard-sec`` replace the task's budgets for one run: the record says the
    build ran under budgets that are not the task's."""
    module = "tasks.multitenant_slo.task"
    own = budget_params(importlib.import_module(module).ARTIFACT.producer)
    params = AgentBuildParams.model_validate({**PARAMS, **own})
    assert budget_note(params, module) == BudgetNote(**own, overridden=False)
    shorter = params.model_copy(update={"hard_sec": own["hard_sec"] - 60})
    assert budget_note(shorter, module) == BudgetNote(**{**own, "hard_sec": own["hard_sec"] - 60}, overridden=True)
