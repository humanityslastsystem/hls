"""The credit stages over a published credit instance: ``credit_<k>`` runs the credit role on its
drive and records what the role reported; ``fold`` pools the drives into the credit
``credit_from_drives`` makes, written by the writer of every committed ``baseline.json``."""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.baseline_credit import (BaselineCredit, BaselineDrive, BaselineRecord, CampaignIdentity,
                                 baseline_credit_bytes, credit_from_drives, load_record, read_baseline_credit)
from hls.pipeline.checkout import CodeState
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES, STAGE_NAME_PATTERNS
from hls.pipeline.plan import AllocationStatus, AttemptStatus, Plan, Toolchain
from hls.pipeline.planner import StageDecl, build_plan
from hls.pipeline.stage import StageFault
from hls.score.credit_role import CREDIT_DRIVE_FILE, CreditDrive, measure
from hls.spec.resources import BaselineHardware, GrantedResources, HardwareRecord
from hls.stages.baseline_warm import WARM_BASELINE
from hls.stages.credit import (CREDIT_SUMMARY_FILE, CreditDriveStage, CreditFoldParams, CreditFoldStage,
                               CreditSummary)
from hls.store.layout import InstanceLayout
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, apply_task_overrides, load_task
from hls.trace.store import write_trace
from hls.verify.calibration import workload_fingerprint
from tests.finalize.fixture import t0_single_task
from tests.finalize.published import (CPU, CPU_ROLE, DRIVE_PARAMS, RUNNER_GIB, SIF_SHA, context, drive_record,
                                      drive_shape, publish, write)
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import settings
from tests.trace.traces import LATENCY_TENANTS, drive_trace

MODULE = "cr_latency.task"
TASK = {"module": MODULE, "overrides": {}}
#: The measured baseline the fixture drives served, as the planner names it.
IMAGE = {"kind": "image", "image": "evaluate-task-baseline", "entrypoint": ["python", "/workspace/server.py"]}
#: A load override of the single-tenant task: its TTFT budget doubled.
LOOSER = {"task.workload.slo.ttft_ms": 200.0}
#: The hardware the fixture drives' allocations ran on.
HARDWARE = HardwareRecord(gpu_model="NVIDIA H200", gpus=1, cpu_model="Example CPU 9000", cpus=8, memory_gib=180.0)


def ran_on(store: PosixStore, plan: Plan, stage: str, hardware: HardwareRecord, *, attempt: int = 1, allocation: int = 0,
           job: str = "100") -> None:
    """Write the statuses that say attempt *attempt* of *stage* ran in job *job*, submission 1 of
    allocation *allocation*, whose runner's grant check measured *hardware*."""
    store.write_bytes(plan.layout.attempt_file(stage, attempt, "status.json"), AttemptStatus(
        stage=stage, attempt=attempt, allocation=allocation, submission=1, job_id=job, state="succeeded",
        started_at=1.0, ended_at=2.0, fault=None, manifest_sha256=None, no_retry=None).model_dump_json().encode())
    grant = GrantedResources(gpus=hardware.gpus, gpu_kind="H200" if hardware.gpus else None,
                             gpu_model=hardware.gpu_model, cpus=hardware.cpus, cpu_model=hardware.cpu_model,
                             memory_gib=hardware.memory_gib, disk_free_gib=100.0)
    store.write_bytes(plan.layout.allocation_status(allocation, 1), AllocationStatus(
        instance=plan.instance, allocation=allocation, submission=1, job_id=job, node="n001", state="succeeded",
        started_at=1.0, heartbeat_at=2.0, end_time=None, stage=None, attempt=None, grant=grant, tool_versions=None,
        swap_max_bytes=None, memory_enforced_by=None, fault=None, ended_at=3.0).model_dump_json().encode())


def _plan(tmp_path: Path, checkout: Path, *, drives: int = 2, task: dict | None = None) -> Plan:
    """A credit instance's drives, their credits and the fold, as the credit pipeline declares
    them, without its provision and images stages, which these stages do not read; every stage
    loads *task*, the latency task as its module declares it unless given."""
    task = task or TASK
    drive = {**DRIVE_PARAMS, "task": task, "artifact": IMAGE, "measured_baseline": True,
             "engine_cache_owner": WARM_BASELINE}
    decls = []
    for k in range(1, drives + 1):
        decls += [StageDecl(f"drive_{k}", "deploy_drive", drive_shape(IMAGE), drive),
                  StageDecl(f"credit_{k}", "credit_drive", CPU, {"task": task, "drive_stage": f"drive_{k}"},
                            after=(f"drive_{k}",))]
    credits = [f"credit_{k}" for k in range(1, drives + 1)]
    decls.append(StageDecl("fold", "credit_fold", CPU, {"task": task, "credit_stages": credits}, after=tuple(credits)))
    return build_plan(decls=decls, task="fin-latency", pipeline="credit",
                      layout=InstanceLayout.credit("fin-latency", "c-1"),
                      settings=settings(tmp_path),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-credit", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0,
                      known_patterns=STAGE_NAME_PATTERNS)


def _publish_drive(store: PosixStore, plan: Plan, k: int, *, failures: bool = False, outcome: str = "served",
                   edit: Callable[[dict], None] | None = None) -> Path:
    """Publish ``drive_<k>`` as a drive of the baseline ending *outcome*, its record's JSON
    changed by *edit* when given; return its outputs directory."""
    stage = f"drive_{k}"
    outputs, end = {}, None
    if outcome == "served":
        trace = drive_trace(failures=failures, tenants=LATENCY_TENANTS)
        out = store.path(plan.layout.attempt_file(stage, 1, "outputs/trace"))
        end = write_trace(trace, out, ending="drained", driver={"class": "Driver"}).end
        outputs["trace"] = plan.layout.attempt_file(stage, 1, "outputs/trace/END.json")
    record = drive_record(outcome, end)
    if edit is not None:
        data = json.loads(record)
        edit(data)
        record = json.dumps(data).encode()
    outputs["drive"] = write(store, plan, stage, "drive.json", record)
    publish(store, plan, stage, outputs)
    return store.path(plan.layout.attempt_file(stage, 1, "outputs"))


def test_the_credit_stage_runs_the_role_on_its_drive(tmp_path, checkout):
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    drive_dir = _publish_drive(store, plan, 1)
    ctx = context(store, plan, "credit_1")
    outputs = CreditDriveStage().run(ctx)
    assert sorted(outputs) == ["credit_drive"]
    assert outputs["credit_drive"] == ctx.output_key(CREDIT_DRIVE_FILE)
    got = CreditDrive.model_validate_json(store.read_bytes(outputs["credit_drive"]))
    assert got == measure(load_task(TaskRef(module=MODULE)), drive_dir)
    assert (got.started_at, got.duration_sec) == (datetime.fromtimestamp(99.5, timezone.utc).isoformat(), 100.0)
    [spec] = ctx.launcher.launched
    assert (spec.name, spec.owner, spec.shape) == ("credit", "harness", CPU_ROLE)
    assert list(spec.argv[1:3]) == ["-m", "hls.score.credit_role"]


def test_the_credit_stage_hands_the_role_its_task_with_the_overrides(tmp_path, checkout):
    """The role scores the task the plan's reference names, ``--task.*`` overrides and all."""
    task = {"module": MODULE, "overrides": {"task.seed": 7}}
    plan = _plan(tmp_path, checkout, task=task)
    store = PosixStore(plan.store_root)
    _publish_drive(store, plan, 1)
    ctx = context(store, plan, "credit_1")
    CreditDriveStage().run(ctx)
    [spec] = ctx.launcher.launched
    handed = Path(spec.argv[list(spec.argv).index("--task-ref") + 1]).read_text()
    assert TaskRef.model_validate_json(handed) == TaskRef.model_validate(task)


def _tamper(store: PosixStore, key: str) -> None:
    """Change the published file at *key* after its manifest recorded its digest."""
    store.path(key).write_bytes(store.read_bytes(key) + b"\n")


def test_a_credit_stage_refuses_a_trace_its_drives_manifest_did_not_record_without_a_role(tmp_path, checkout):
    """The trace's ``END.json`` pins the summary the role scores, so a changed one is refused
    before any role starts, and not retried: a retry reads the same file."""
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_drive(store, plan, 1)
    _tamper(store, plan.layout.attempt_file("drive_1", 1, "outputs/trace/END.json"))
    ctx = context(store, plan, "credit_1")
    with pytest.raises(StageFault) as e:
        CreditDriveStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "END.json does not match its manifest" in e.value.fault.reason and ctx.launcher.launched == []


def test_a_served_drive_without_its_servers_readiness_time_is_refused_without_a_role(tmp_path, checkout):
    """A served drive's record names when its server opened its readiness gates, which the
    credit's times are measured from; a record without it is no drive record, refused before
    any role starts and not retried."""
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_drive(store, plan, 1, edit=lambda record: record["artifact"].update(ready_after_command_sec=None))
    ctx = context(store, plan, "credit_1")
    with pytest.raises(StageFault) as e:
        CreditDriveStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "drive.json is not a drive record" in e.value.fault.reason and ctx.launcher.launched == []


def test_a_credit_stage_refuses_a_drive_that_did_not_serve_without_a_role(tmp_path, checkout):
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_drive(store, plan, 1, outcome="crashed")
    ctx = context(store, plan, "credit_1")
    with pytest.raises(StageFault) as e:
        CreditDriveStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "did not serve" in e.value.fault.reason and ctx.launcher.launched == []


def test_the_credit_roles_refusal_is_not_retried(tmp_path, checkout):
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_drive(store, plan, 1, failures=True)
    with pytest.raises(StageFault) as e:
        CreditDriveStage().run(context(store, plan, "credit_1"))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason.startswith("credit refused: ") and "scored zero" in e.value.fault.reason


def test_a_credit_role_that_reports_none_of_its_outputs_is_retried(tmp_path, checkout):
    stub = tmp_path / "stub.py"
    stub.write_text("import sys\nfrom pathlib import Path\n"
                    "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
                    "(out / 'credit-drive.json').write_text('{}')\n")
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_drive(store, plan, 1)
    ctx = context(store, plan, "credit_1",
                  launcher=LocalLauncher(gpus=0, replace_modules={"hls.score.credit_role": stub}))
    with pytest.raises(StageFault) as e:
        CreditDriveStage().run(ctx)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", True)
    assert "reported no digest of output credit-drive.json" in e.value.fault.reason


def _credit(i: int, credit: float, *, units: int = 10, image: str = SIF_SHA) -> CreditDrive:
    return CreditDrive(started_at=f"2026-10-05T00:0{i}:00+00:00", duration_sec=60.0, n_requests=units,
                       n_units=units, mean_log_credit=credit, image_sha256=image, score={})


def _publish_credits(store: PosixStore, plan: Plan, drives: list[CreditDrive],
                     hardware: list[HardwareRecord | None] | None = None) -> None:
    """Publish ``credit_<k>`` over each of *drives*, and ``drive_<k>`` as run on an allocation of
    its own whose grant check measured *hardware*'s k-th record (``HARDWARE`` for every drive by
    default; None for a drive whose allocation recorded no status)."""
    for k, drive in enumerate(drives, start=1):
        record = (hardware or [HARDWARE] * len(drives))[k - 1]
        publish(store, plan, f"drive_{k}", {})
        if record is not None:
            ran_on(store, plan, f"drive_{k}", record, allocation=k - 1, job=str(100 + k))
        key = write(store, plan, f"credit_{k}", CREDIT_DRIVE_FILE, drive.model_dump_json(indent=2).encode())
        publish(store, plan, f"credit_{k}", {"credit_drive": key})


def _campaign() -> CampaignIdentity:
    """The identity a credit of the fixture task records."""
    task = load_task(TaskRef(module=MODULE))
    return CampaignIdentity(task=MODULE, workload_fingerprint=workload_fingerprint(task), load=load_record(task),
                            gpu_kind=task.hardware.accelerator.kind,
                            baseline=BaselineRecord(image=task.baseline.image,
                                                    serve_command=list(task.baseline.serve_command)))


def _baseline_drive(d: CreditDrive) -> BaselineDrive:
    return BaselineDrive(started_at=d.started_at, duration_sec=d.duration_sec, n_requests=d.n_requests,
                         n_units=d.n_units, mean_log_credit=d.mean_log_credit)


def test_the_fold_writes_the_credit_from_drives_makes(tmp_path, checkout):
    plan = _plan(tmp_path, checkout, drives=3)
    store = PosixStore(plan.store_root)
    drives = [_credit(3, 0.6), _credit(1, 0.4), _credit(2, 0.5)]
    _publish_credits(store, plan, drives)
    outputs = CreditFoldStage().run(context(store, plan, "fold"))
    assert sorted(outputs) == ["baseline", "credit"]
    data = store.read_bytes(outputs["baseline"])
    candidate = BaselineCredit.model_validate_json(data)
    want = credit_from_drives(_campaign(), [_baseline_drive(d) for d in drives], [HARDWARE] * 3)
    assert (candidate.campaign, candidate.drives, candidate.hardware) == (want.campaign, want.drives,
                                                                          [BaselineHardware.of(HARDWARE)])
    # Ordered by start time.
    assert [d.mean_log_credit for d in candidate.drives] == [0.4, 0.5, 0.6]
    assert candidate.log_credit_mean == pytest.approx(0.5)
    # Byte for byte what a committed baseline.json holds (baseline_credit_bytes), which holds no
    # computed field: B and its spread are in credit.json, for people.
    assert data == baseline_credit_bytes(candidate)
    assert not {"log_credit_mean", "log_credit_sd", "log_credit_se"} & json.loads(data).keys()
    summary = CreditSummary.model_validate_json(store.read_bytes(outputs["credit"]))
    assert outputs["credit"].endswith(f"/outputs/{CREDIT_SUMMARY_FILE}")
    assert (summary.drives, summary.image_sha256, summary.hls_commit) == (3, SIF_SHA, "a" * 40)
    assert summary.baseline_sha256 == hashlib.sha256(data).hexdigest()
    assert (summary.log_credit_mean, summary.log_credit_sd, summary.log_credit_se) == (
        candidate.log_credit_mean, candidate.log_credit_sd, candidate.log_credit_se)


def test_the_fold_records_the_hardware_its_drives_ran_on(tmp_path, checkout):
    """The candidate and its summary record the distinct hardware records of
    the allocations the drives ran on, every one when they differ, and none when a drive's
    allocation recorded none; B is the same either way."""
    other = HARDWARE.model_copy(update={"cpu_model": "Another CPU 1"})
    measured = [BaselineHardware.of(HARDWARE), BaselineHardware.of(other)]
    for name, hardware, want in (("one", [HARDWARE, HARDWARE], measured[:1]),
                                 ("two", [HARDWARE, other], sorted(measured, key=lambda h: h.model_dump_json())),
                                 ("unrecorded", [HARDWARE, None], None)):
        plan = _plan(tmp_path / name, checkout)
        store = PosixStore(plan.store_root)
        _publish_credits(store, plan, [_credit(1, 0.4), _credit(2, 0.6)], hardware)
        outputs = CreditFoldStage().run(context(store, plan, "fold"))
        data = store.read_bytes(outputs["baseline"])
        candidate = BaselineCredit.model_validate_json(data)
        assert candidate.hardware == want, name
        assert CreditSummary.model_validate_json(store.read_bytes(outputs["credit"])).hardware == want, name
        assert candidate.log_credit_mean == pytest.approx(0.5)
        # A candidate that records no hardware is written as a file that records none.
        assert ("hardware" in json.loads(data)) == (want is not None), name


def test_the_fold_pools_only_the_credit_stages_it_names(tmp_path, checkout):
    """A credit record of this instance the fold's parameters do not name is not pooled."""
    plan = _plan(tmp_path, checkout, drives=2)
    store = PosixStore(plan.store_root)
    _publish_credits(store, plan, [_credit(1, 0.4), _credit(2, 0.6)])
    stray = write(store, plan, "credit_3", CREDIT_DRIVE_FILE, _credit(3, 9.0).model_dump_json().encode())
    publish(store, plan, "credit_3", {"credit_drive": stray})
    outputs = CreditFoldStage().run(context(store, plan, "fold"))
    candidate = BaselineCredit.model_validate_json(store.read_bytes(outputs["baseline"]))
    assert [d.mean_log_credit for d in candidate.drives] == [0.4, 0.6]


def test_the_credit_records_todays_fingerprint(tmp_path, checkout):
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_credits(store, plan, [_credit(1, 0.4), _credit(2, 0.6)])
    outputs = CreditFoldStage().run(context(store, plan, "fold"))
    task = load_task(TaskRef(module=MODULE))
    handed = tmp_path / "handed"
    handed.mkdir()
    (handed / "baseline.json").write_bytes(store.read_bytes(outputs["baseline"]))
    # The credit loads as a file a role is handed, its fingerprint today's.
    loaded = read_baseline_credit(handed / "baseline.json", task)[0]
    assert loaded.campaign.workload_fingerprint == workload_fingerprint(task)


def test_the_fold_records_the_load_its_task_overrides_make(tmp_path, checkout):
    """A candidate measured under ``--task.*`` overrides records the load they set, which the
    graded task is then held to: the fold loads the task its reference names, overrides and all."""
    plan = _plan(tmp_path, checkout, task={"module": "cr_single.task", "overrides": LOOSER})
    store = PosixStore(plan.store_root)
    _publish_credits(store, plan, [_credit(1, 0.4), _credit(2, 0.6)])
    outputs = CreditFoldStage().run(context(store, plan, "fold"))
    campaign = BaselineCredit.model_validate_json(store.read_bytes(outputs["baseline"])).campaign
    task = apply_task_overrides(t0_single_task(), LOOSER)
    assert (campaign.task, campaign.load, campaign.workload_fingerprint) == (
        "cr_single.task", load_record(task), workload_fingerprint(task))
    assert campaign.load != load_record(t0_single_task())


def test_the_fold_refuses_a_credit_record_its_manifest_did_not_record(tmp_path, checkout):
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_credits(store, plan, [_credit(1, 0.4), _credit(2, 0.6)])
    _tamper(store, plan.layout.attempt_file("credit_2", 1, f"outputs/{CREDIT_DRIVE_FILE}"))
    with pytest.raises(StageFault) as e:
        CreditFoldStage().run(context(store, plan, "fold"))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert f"{CREDIT_DRIVE_FILE} does not match its manifest" in e.value.fault.reason


def test_the_fold_refuses_a_task_that_declares_no_baseline_server(tmp_path, checkout):
    """The candidate names the baseline server it measured, so a task that declares none is
    refused, and not retried: a retry loads the same task."""
    plan = _plan(tmp_path, checkout, task={"module": "cr_nobase.task", "overrides": {}})
    store = PosixStore(plan.store_root)
    _publish_credits(store, plan, [_credit(1, 0.4), _credit(2, 0.6)])
    with pytest.raises(StageFault) as e:
        CreditFoldStage().run(context(store, plan, "fold"))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "task cr_nobase.task declares no baseline server" in e.value.fault.reason


def test_a_fold_that_names_a_credit_stage_twice_is_refused():
    """A drive named twice would be pooled twice, weighing B toward it."""
    with pytest.raises(ValidationError, match="a fold names each credit stage once"):
        CreditFoldParams.model_validate({"task": TASK, "credit_stages": ["credit_1", "credit_2", "credit_1"]})


@pytest.mark.parametrize("stage, shape, match", [
    (CreditDriveStage, CPU.model_copy(update={"gpus": 1, "gpu_kind": "H200"}), "uses no GPU"),
    (CreditFoldStage, CPU.model_copy(update={"gpus": 1, "gpu_kind": "H200"}), "uses no GPU"),
    (CreditFoldStage, CPU.model_copy(update={"memory_gib": RUNNER_GIB - 1}),
     f"must hold the runner's {RUNNER_GIB}"),
], ids=["credit-gpu", "fold-gpu", "fold-below-runner"])
def test_a_credit_stage_shape_that_cannot_run_is_refused_when_planned(stage, shape, match):
    """Both stages read small files, and the fold runs in the runner, so its memory holds the
    runner's."""
    params = {"task": TASK, "drive_stage": "drive_1"} if stage is CreditDriveStage else {
        "task": TASK, "credit_stages": ["credit_1", "credit_2"]}
    with pytest.raises(ValueError, match=match):
        stage.check(shape, stage.params_model.model_validate(params), runner_memory_gib=RUNNER_GIB)


@pytest.mark.parametrize("drives, match", [
    ([_credit(1, 0.4, image="1" * 64), _credit(2, 0.6, image="2" * 64)], "different image files"),
    ([_credit(1, 0.4, units=100), _credit(2, 0.6, units=90)], "cannot be pooled"),
], ids=["two-images", "units"])
def test_the_fold_refuses_drives_served_by_different_images(tmp_path, checkout, drives, match):
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    _publish_credits(store, plan, drives)
    with pytest.raises(StageFault) as e:
        CreditFoldStage().run(context(store, plan, "fold"))
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert match in e.value.fault.reason


def test_a_fold_of_no_drives_is_refused_when_planned(tmp_path, checkout):
    with pytest.raises(ValueError, match="at least 1"):
        _plan(tmp_path, checkout, drives=0)
