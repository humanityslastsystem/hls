"""The credit stages: one drive of a task's measured baseline
scored with no baseline credit (``credit_drive``), and the drives pooled into a
``baseline.json`` (``credit_fold``).

``credit_drive`` stays light, as the score stage does. It reads ``drive.json`` from the drive
stage of this instance its parameters name, checked against the digest its manifest recorded,
and refuses a drive that did not serve before starting anything; for a served drive it checks
the trace's ``END.json`` too, whose digests pin the summary the role scores. Then it starts one
role, ``hls.score.credit_role``, as a harness host role with the plan's checkout on
``PYTHONPATH``, in the stage's shape less the runner's memory. The role writes
``credit-drive.json`` into the attempt's outputs directory and reports its digest, which the
manifest records (``StageContext.reported_output``).

``credit_fold`` runs in the runner and starts no role. It reads each ``credit-drive.json`` its
parameters name, checked against its manifest; rebuilds the task with no baseline credit; builds
the campaign identity a ``baseline.json`` records (the task module, the workload
fingerprint, the load, the GPU kind and the baseline server the task declares); pools the drives
with ``credit_from_drives``, with the hardware records of the allocations its drives ran on;
and writes the credit with ``baseline_credit_bytes``, byte for byte as a
task's default ``baseline.json``, as its output and as the store's measured file of the task
(``hls.measured.store_key``), which runs of the task then use, and ``credit.json``, its summary.

Neither stage decides a contestant fault: a credit instance has no contestant. Every failure here
is infra:
- a drive that did not serve, the role's deterministic refusal (exit 3: a drive no baseline
  credit may pool), and the fold's refusals (drives served by different image files, drives
  ``credit_from_drives`` will not pool, a task that does not load or declares no baseline
  server) are not retried, since a retry reads the same files. The fault carries the reason;
- any other failure of the role, and a role that exits 0 without its file or without reporting
  it, is retried;
- inputs that contradict the plan (a manifest missing from the stage's inputs, a file that is not
  the one its manifest recorded, a record that does not parse) are not retried.

At plan time ``check`` refuses a shape with GPUs (both stages read small files), a credit stage
whose memory leaves its role none beside the runner's, and a fold whose memory does not hold the
runner's."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

from hls.baseline_credit import (BASELINE_CREDIT_FILE, MIN_DRIVES, BaselineCreditError, BaselineDrive, BaselineRecord,
                                 CampaignIdentity, baseline_credit_bytes, credit_from_drives, load_record)
from hls.drive.record import DRIVE_FILE, TRACE_END
from hls.measured import store_key, write_measured
from hls.pipeline.plan import attempt_hardware
from hls.pipeline.stage import Stage, StageContext, StageFault, register
from hls.score.credit_role import CREDIT_DRIVE_FILE, REFUSED_EXIT, REFUSED_PREFIX, CreditDrive
from hls.settings import StageName
from hls.spec.resources import BaselineHardware, HardwareRecord, ResourceShape
from hls.stages.provision import role_shape
from hls.stages.inputs import input_manifest, output, read_output
from hls.store.manifest import read_manifest
from hls.stages.score import drive_record, host_role_env, judge_host_role
from hls.store.digest import Sha256Hex, sha256_bytes
from hls.taskref import TaskRef, load_task
from hls.verify.calibration import workload_fingerprint

ROLE = "credit"
#: The fold's summary of the credit it wrote.
CREDIT_SUMMARY_FILE = "credit.json"


class CreditDriveParams(BaseModel):
    """What a credit stage scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task whose scorer runs, as the role rebuilds it with no baseline credit.
    task: TaskRef
    #: The drive stage of this instance whose ``drive.json`` and trace are scored.
    drive_stage: StageName


class _CreditDrive:
    """One credit stage attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None:
            raise RuntimeError("the credit_drive stage needs node-local scratch")
        self.ctx = ctx
        self.params: CreditDriveParams = ctx.params
        #: The attempt's outputs directory, which the role writes.
        self.out = ctx.output_path(CREDIT_DRIVE_FILE).parent

    def run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        drive = input_manifest(ctx, p.drive_stage)
        drive_ref = output(drive, DRIVE_FILE)
        record = drive_record(ctx, drive_ref)
        if record.outcome != "served":
            raise StageFault.infra(f"{p.drive_stage} did not serve ({record.outcome}: {record.reason}), so the "
                                   "baseline's drive measures nothing", retryable=False)
        read_output(ctx, output(drive, TRACE_END))
        task_ref = ctx.role_dir(ROLE) / "task-ref.json"
        task_ref.write_text(p.task.model_dump_json())
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.score.credit_role", "--task-ref", str(task_ref),
                "--drive", str(ctx.store.path(drive_ref.key).parent), "--out", str(self.out),
                "--outputs-out", str(ctx.outputs_report(ROLE))]
        shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
        ctx.start_role(ROLE, argv, shape, env=host_role_env(ctx), owner="harness")
        [outcome] = ctx.wait_roles()
        judge_host_role(ctx, ROLE, outcome, refused_exit=REFUSED_EXIT, refused_prefix=REFUSED_PREFIX)
        if not (self.out / CREDIT_DRIVE_FILE).is_file():
            raise StageFault.infra(f"the credit role exited 0 without writing {CREDIT_DRIVE_FILE}", retryable=True)
        return {"credit_drive": ctx.reported_output(CREDIT_DRIVE_FILE, ROLE)}


@register
class CreditDriveStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "credit_drive"
    params_model: ClassVar[type[BaseModel]] = CreditDriveParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the credit role."""
        return 1

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, CreditDriveParams)
        if shape.gpus:
            raise ValueError(f"the credit_drive stage reads a summary and uses no GPU; its shape asks for {shape.gpus}")
        role_shape(shape, runner_memory_gib)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _CreditDrive(ctx).run()


def _distinct(names: list[str]) -> list[str]:
    if len(names) < MIN_DRIVES:
        raise ValueError(f"a fold pools at least {MIN_DRIVES} drive, got {len(names)}")
    if len(set(names)) != len(names):
        raise ValueError(f"a fold names each credit stage once, got {names}")
    return names


class CreditFoldParams(BaseModel):
    """What the fold pools."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task whose campaign the drives measured, rebuilt with no baseline credit.
    task: TaskRef
    #: The credit stages of this instance whose ``credit-drive.json`` it pools, at least one.
    credit_stages: Annotated[list[StageName], AfterValidator(_distinct)]


class CreditSummary(BaseModel):
    """``credit.json``: the credit the fold wrote, in brief, for people."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: B: the mean of the drives' credits.
    log_credit_mean: float
    #: The drives' sample standard deviation: the reference's own drive-to-drive spread. None for
    #: one drive, which has no spread between drives.
    log_credit_sd: float | None
    #: The standard error of B; None for one drive.
    log_credit_se: float | None
    #: Drives pooled.
    drives: int
    #: The sha256 of the image file every drive served.
    image_sha256: Sha256Hex
    #: The hls commit the instance ran, the credit's provenance.
    hls_commit: str
    #: The sha256 of the ``baseline.json``.
    baseline_sha256: Sha256Hex
    #: The store key of the task's measured file the fold wrote it to (``hls.measured.store_key``).
    measured_key: str
    #: The distinct hardware records of the allocations the drives ran on, as the credit
    #: records them; None when a drive's allocation status records none.
    hardware: list[BaselineHardware] | None


class _Fold:
    """One fold attempt."""

    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx
        self.params: CreditFoldParams = ctx.params

    def drive(self, stage: str) -> CreditDrive:
        """The ``credit-drive.json`` of credit stage *stage*, checked against its manifest."""
        ref = output(input_manifest(self.ctx, stage), CREDIT_DRIVE_FILE)
        try:
            return CreditDrive.model_validate_json(read_output(self.ctx, ref))
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a credit drive record: {e.error_count()} errors",
                                   retryable=False) from e

    def hardware(self) -> list[HardwareRecord] | None:
        """The hardware record of the allocation each drive stage's published attempt ran on
        (``attempt_hardware``); None when one of them records none, so the credit claims no
        hardware it was not measured on."""
        records = []
        for stage in self.params.credit_stages:
            drive = CreditDriveParams.model_validate(self.ctx.plan.stage(stage).params).drive_stage
            found = read_manifest(self.ctx.store, self.ctx.plan.layout, drive)
            record = attempt_hardware(self.ctx.store, self.ctx.plan.layout, drive, found[0].attempt) if found else None
            if record is None:
                return None
            records.append(record)
        return records

    def campaign(self) -> CampaignIdentity:
        """What the drives measured, as a ``baseline.json`` records it."""
        p = self.params
        try:
            task = load_task(p.task)
        except Exception as e:
            raise StageFault.infra(f"task {p.task.module} does not load: {type(e).__name__}: {e}",
                                   retryable=False) from e
        if task.baseline is None:
            raise StageFault.infra(f"task {p.task.module} declares no baseline server", retryable=False)
        return CampaignIdentity(task=p.task.module, workload_fingerprint=workload_fingerprint(task),
                                load=load_record(task), gpu_kind=task.hardware.accelerator.kind,
                                baseline=BaselineRecord(image=task.baseline.image,
                                                        serve_command=list(task.baseline.serve_command)))

    def run(self) -> dict[str, str]:
        ctx = self.ctx
        drives = [self.drive(stage) for stage in self.params.credit_stages]
        images = sorted({d.image_sha256 for d in drives})
        if len(images) != 1:
            raise StageFault.infra(f"the drives were served by different image files ({', '.join(images)}); a baseline "
                                   "credit is a measurement of one server", retryable=False)
        pooled = [BaselineDrive(started_at=d.started_at, duration_sec=d.duration_sec, n_requests=d.n_requests,
                                n_units=d.n_units, mean_log_credit=d.mean_log_credit) for d in drives]
        try:
            credit = credit_from_drives(self.campaign(), pooled, self.hardware())
        except BaselineCreditError as e:
            raise StageFault.infra(f"the drives cannot be pooled: {e}", retryable=False) from e
        data = baseline_credit_bytes(credit)
        key = store_key(ctx.plan.task, BASELINE_CREDIT_FILE)  # where write_measured writes it, below
        summary = CreditSummary(log_credit_mean=credit.log_credit_mean, log_credit_sd=credit.log_credit_sd,
                                log_credit_se=credit.log_credit_se, drives=len(credit.drives), image_sha256=images[0],
                                hls_commit=ctx.plan.code.hls_commit, baseline_sha256=sha256_bytes(data),
                                measured_key=key, hardware=credit.hardware)
        outputs = {"baseline": ctx.write_output(BASELINE_CREDIT_FILE, data),
                   "credit": ctx.write_output(CREDIT_SUMMARY_FILE, summary.model_dump_json(indent=2).encode())}
        # The task's measured file is the last thing written: what runs of the task use from now on.
        write_measured(ctx.store, ctx.plan.task, BASELINE_CREDIT_FILE, data)
        ctx.events.emit("credit_folded", drives=summary.drives, log_credit_mean=summary.log_credit_mean,
                        log_credit_se=summary.log_credit_se, measured_key=key)
        return outputs


@register
class CreditFoldStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "credit_fold"
    params_model: ClassVar[type[BaseModel]] = CreditFoldParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: none, the fold reads its records itself."""
        return 0

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, CreditFoldParams)
        if shape.gpus:
            raise ValueError(f"the credit_fold stage reads small records and uses no GPU; its shape asks for "
                             f"{shape.gpus}")
        if shape.memory_gib < runner_memory_gib:
            raise ValueError(f"the credit_fold stage runs in the runner, so its {shape.memory_gib} GiB must hold the "
                             f"runner's {runner_memory_gib}")

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Fold(ctx).run()
