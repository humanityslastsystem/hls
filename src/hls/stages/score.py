"""The score stage: a drive scored against its verdict, and the baseline credit it was scored
against.

The stage runs in the runner and stays light: it checks the files it hands over against their
manifests and starts one role.
1. It reads ``drive.json`` from this instance's drive stage and checks ``verdict.json`` of its
   verify stage, each against the digest its manifest recorded. For a served drive it also
   checks the trace's ``END.json``, whose digests pin the summary the role scores from.
2. For an invalid artifact it finds the ``package.json`` that said so: the drive stage's
   parameters name the package manifest it read, the drive's manifest pins that manifest's
   digest, and the package manifest pins the record's. The role takes the diagnostics from it.
3. It copies the baseline credit the plan pins (``hls.measured``), checked against the blob id
   it was pinned by, into the role's directory, and starts the score role, ``hls.score.role``,
   on that copy, as a harness host role with the plan's checkout on ``PYTHONPATH``, in the
   stage's shape less the runner's memory. The role writes
   ``score.json`` and ``score-provenance.json`` into the attempt's outputs directory and
   reports their digests, which the manifest records (``StageContext.reported_output``).

A drive that did not serve is not a fault here either: the role scores it zero, and the stage
publishes that. A stage before it that failed with a contestant fault never lets this stage
run; the orchestrator records that instance's zero (``hls.stages.finalize.failed_records``).
Every failure here is infra, the roles' ``stage_fault``:
- the role's deterministic refusal (exit 3: a baseline credit that is missing, stale or from
  other silicon) is not retried, since a retry repeats it. The fault carries the role's reason;
- any other failure of the role, and a role that exits 0 without its files or without
  reporting them, is retried;
- inputs that contradict the plan (a manifest missing from the stage's inputs, a file that is
  not the one its manifest recorded, a measured file that is not the one the plan pinned, an
  invalid artifact whose package record cannot be traced) are not retried: another attempt
  reads the same files.

At plan time ``check`` refuses a shape with GPUs, scoring reads a summary and holds none, and
one whose memory leaves the role none beside the runner's.

The role reads the summary a line at a time and holds each event's identity and timings, of a
reply's packet timestamps the first, the last and their count (``hls.trace.store.read_summary``).
What it holds whole for a moment is one reply's line, every packet timestamp of it, and a
contestant's server chooses how many there are up to ``REPLY_MAX_BYTES`` of reply. So the role's
memory follows from that cap (``score_memory_gib``): the most timestamps one reply buys on the
task's stream (``packet_wire_bytes``), at what the role measurably holds for each
(``SUMMARY_BYTES_PER_PACKET``), with a margin, beside 8 GiB for the summary, the verdict and the
scorer (``SCORE_BASE_GIB``)."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from hls.artifact.artifact_client import packet_wire_bytes
from hls.drive.bounds import REPLY_MAX_BYTES
from hls.drive.record import DRIVE_FILE, TRACE_END, DriveRecord
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.score.role import PROVENANCE_FILE, REFUSED_EXIT, REFUSED_PREFIX, SCORE_FILE
from hls.settings import StageName
from hls.spec.contracts import ArtifactContract
from hls.spec.resources import ResourceShape
from hls.stages.deploy_drive import DeployDriveParams, PackagedArtifact
from hls.stages.inputs import input_manifest, measured_copies, output, read_output
from hls.stages.provision import role_shape
from hls.store.base import NotFound
from hls.store.digest import sha256_bytes
from hls.store.manifest import Manifest, OutputRef
from hls.substrate.faults import RoleOutcome
from hls.taskref import TaskRef
from hls.util import validation_errors

ROLE = "score"
#: The verify stage's verdict file.
VERDICT_FILE = "verdict.json"
#: The name of the package stage's output that holds its ``PackageResult``.
PACKAGE_OUTPUT = "package"
#: Bytes from the end of the role's stderr searched for its refusal.
REASON_SEARCH_BYTES = 64 * 1024
#: The role's memory before one reply's packets: 8 GiB for a drive's summary, its verdict and the
#: scorer (a summary of a few replies holds 0.08 GiB). The events the
#: role holds are the drive's turns, which its workload fixes, whatever the server sends.
SCORE_BASE_GIB = 8
#: Bytes the role holds per packet timestamp of one reply while it reads that reply's summary
#: line, at most: the role holds 8.83 GiB for one reply of 120,000,000 timestamps, which the
#: reader parses whole with the rest of its file beside it.
#: Read a line at a time, holding the line, its parsed list and the validated one,
#: a reply's timestamp costs less.
SUMMARY_BYTES_PER_PACKET = 80
#: How much more than ``SUMMARY_BYTES_PER_PACKET`` a reply's packets are given.
SCORE_MARGIN = 1.25


def score_memory_gib(contract: ArtifactContract) -> int:
    """The score role's memory for a task whose replies follow *contract*: ``SCORE_BASE_GIB``, and
    the most packet timestamps one reply of ``REPLY_MAX_BYTES`` buys on the contract's stream
    (``packet_wire_bytes``) at ``SUMMARY_BYTES_PER_PACKET`` each, with ``SCORE_MARGIN``. Raise
    ValueError for a contract whose packet cost the client cannot say."""
    per_packet = packet_wire_bytes(contract)
    if per_packet is None:
        return SCORE_BASE_GIB
    held = SCORE_MARGIN * SUMMARY_BYTES_PER_PACKET * (REPLY_MAX_BYTES // per_packet)
    return SCORE_BASE_GIB + math.ceil(held / 2**30)


class ScoreParams(BaseModel):
    """What the score stage scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task whose scorer runs, as the role rebuilds it with its baseline credit.
    task: TaskRef
    #: The stage of this instance whose ``drive.json`` and trace are scored.
    drive_stage: StageName = "deploy_drive"
    #: The stage of this instance whose ``verdict.json`` gates the score.
    verify_stage: StageName = "verify"


def drive_record(ctx: StageContext, ref: OutputRef) -> DriveRecord:
    """The ``DriveRecord`` that output *ref* holds, checked against its manifest."""
    try:
        return DriveRecord.model_validate_json(read_output(ctx, ref))
    except ValidationError as e:
        raise StageFault.infra(f"{DRIVE_FILE} is not a drive record: {validation_errors(e)}", retryable=False) from None


def host_role_env(ctx: StageContext) -> dict[str, str]:
    """The environment of a host role that runs a program of the plan's checkout."""
    checkout = ctx.plan.code.checkout
    return {"PYTHONPATH": f"{checkout}/src{os.pathsep}{checkout}"}


def judge_host_role(ctx: StageContext, role: str, outcome: RoleOutcome, *, refused_exit: int,
                    refused_prefix: str) -> None:
    """Raise the fault the ending *outcome* of the host role *role* is, if any: its
    ``stage_fault``, made not retryable when the role refused, since a retry reads the same
    files. A role refused when it exited *refused_exit* on its own, never killed for memory; the
    fault then gives its reason, the last line of its stderr starting with *refused_prefix*
    (``refusal``), scrubbed, after ``"<role> refused: "``."""
    fault = stage_fault([outcome])
    if fault is None:
        return
    if outcome.ending == "exited" and outcome.oom_kills == 0 and outcome.exit_code == refused_exit:
        reason = refusal(ctx.attempt_dir / "logs" / f"{role}.err", refused_prefix)
        fault = fault.model_copy(update={"reason": f"{role} refused: {ctx.scrubber.scrub(reason)}",
                                         "retryable": False})
    raise StageFault(fault)


def refusal(err: Path, prefix: str) -> str:
    """The last line starting with *prefix* in the end of the stderr file *err*, without
    the prefix."""
    try:
        with open(err, "rb") as f:
            f.seek(max(0, os.path.getsize(err) - REASON_SEARCH_BYTES))
            text = f.read().decode(errors="replace")
    except FileNotFoundError:
        text = ""
    lines = [line for line in text.splitlines() if line.startswith(prefix)]
    return lines[-1][len(prefix):] if lines else "the role gave no reason"


class _Score:
    """One score attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None:
            raise RuntimeError("the score stage needs node-local scratch")
        self.ctx = ctx
        self.params: ScoreParams = ctx.params
        #: The attempt's outputs directory, which the role writes.
        self.out = ctx.output_path(SCORE_FILE).parent

    def package_record(self, drive: Manifest) -> Path:
        """The ``package.json`` the drive read, traced through the digests the drive's
        manifest and the package manifest recorded."""
        ctx = self.ctx
        params = DeployDriveParams.model_validate(ctx.plan.stage(self.params.drive_stage).params)
        if not isinstance(params.artifact, PackagedArtifact):
            raise StageFault.infra("the drive says its artifact is invalid, but it served a prebuilt image",
                                   retryable=False)
        key = params.artifact.manifest or ctx.plan.layout.manifest(params.artifact.stage)
        pinned = drive.inputs.get(key)
        if pinned is None:
            raise StageFault.infra(f"the {drive.stage} manifest does not pin the package manifest {key}",
                                   retryable=False)
        try:
            data = ctx.store.read_bytes(key)
        except NotFound as e:
            raise StageFault.infra(f"the package manifest {key} is gone", retryable=False) from e
        if sha256_bytes(data) != pinned:
            raise StageFault.infra(f"{key} is not the package manifest the drive read", retryable=False)
        ref = Manifest.model_validate_json(data).outputs.get(PACKAGE_OUTPUT)
        if ref is None:
            raise StageFault.infra(f"the package manifest {key} has no output {PACKAGE_OUTPUT}", retryable=False)
        read_output(ctx, ref)
        return ctx.store.path(ref.key)

    # results

    def outputs(self) -> dict[str, str]:
        """The two files the role wrote, each recorded by the digest the role reported."""
        names = {"score": SCORE_FILE, "score_provenance": PROVENANCE_FILE}
        missing = [f for f in names.values() if not (self.out / f).is_file()]
        if missing:
            raise StageFault.infra(f"the score role exited 0 without writing {missing}", retryable=True)
        return {name: self.ctx.reported_output(file, ROLE) for name, file in names.items()}

    # the stage

    def run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        drive = input_manifest(ctx, p.drive_stage)
        drive_ref = output(drive, DRIVE_FILE)
        record = drive_record(ctx, drive_ref)
        verdict = output(input_manifest(ctx, p.verify_stage), VERDICT_FILE)
        read_output(ctx, verdict)
        if record.outcome == "served":
            read_output(ctx, output(drive, TRACE_END))
        task_ref = ctx.role_dir(ROLE) / "task-ref.json"
        task_ref.write_text(p.task.model_dump_json())
        measured = measured_copies(ctx, ctx.role_dir(ROLE) / "measured")
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.score.role", "--task-ref", str(task_ref),
                "--drive", str(ctx.store.path(drive_ref.key).parent),
                "--verdict", str(ctx.store.path(verdict.key)), "--out", str(self.out),
                "--outputs-out", str(ctx.outputs_report(ROLE))]
        if measured.credit is not None:
            argv += ["--credit", str(measured.credit)]
        if record.outcome == "invalid_artifact":
            argv += ["--package", str(self.package_record(drive))]
        shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
        ctx.start_role(ROLE, argv, shape, env=host_role_env(ctx), owner="harness")
        [outcome] = ctx.wait_roles()
        judge_host_role(ctx, ROLE, outcome, refused_exit=REFUSED_EXIT, refused_prefix=REFUSED_PREFIX)
        return self.outputs()


@register
class ScoreStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "score"
    params_model: ClassVar[type[BaseModel]] = ScoreParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    #: The score carries the verdict's diagnostics, which quote the server's replies, and an
    #: invalid artifact's build output (``Stage.contestant_outputs``).
    contestant_outputs: ClassVar[tuple[str, ...]] = (SCORE_FILE,)

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the score role."""
        return 1

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, ScoreParams)
        if shape.gpus:
            raise ValueError(f"the score stage reads a summary and uses no GPU; its shape asks for {shape.gpus}")
        role_shape(shape, runner_memory_gib)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Score(ctx).run()
