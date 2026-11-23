"""The oracle_fill stage: a task's reference answers to its test split, computed into the store
ahead of its gradings (``hls oracle fill``).

The stage runs in the runner and starts one role, ``hls.verify.fill``, as a harness container
on the task's reference image: the image its gradings verify on, which this instance's images
stage names in ``images.json``. The stage pins it as the verify stage does (``image_on_node``:
extracted on the node against the entry digest recorded there), so the answers are keyed by the
same image file digest and GPU kind a grading's verify looks them up under. The tree is
read-only to the container, and the stage's disk declares it (``KEPT_DISK_GIB``). The image is
pinned and the role's data copied before the role starts, since a copy or an extraction runs
alone. The role gets what the verify role gets: the plan's checkout read-only at
``/opt/hls-src`` on ``PYTHONPATH``, the task reference, the reference's data (the task model's
weights and the full tool bundles), the stage's GPUs and ``TORCH_CUDA_ARCH_LIST`` for their kind,
the attempt's outputs directory and the file it reports its outputs in, writable, and the task's
answers area, ``oracle/<task>/``, writable, under a writer name unique to the attempt
(``answers_writer``). It also gets the test rows at ``/data/tenants/<id>/test``, which no verify
role mounts, and the dev rows beside them, which the dev reference answers (``hls.provision.mounts``
role ``oracle``), and the key of the rows entry they were
copied from, which names the question set the role records last once every answer is banked
(``hls.verify.fill``). It runs for the stage's own time. Its output is the role's ``fill.json``,
recorded by the role's own report of it (``reported_output``), and, when the stage produces the
task's dev reference (``OracleFillParams.dev_reference``), its ``dev_reference.json``, which the
stage also writes into the store's measured area (``hls.measured``). The answers and the question
sets it adds are no output: the area lies outside every instance, as for the verify stage, and
only harness containers mount it.

A fill banks every chunk it finishes, so a retry after a time limit or a lost node resumes
where the last attempt stopped. Every failure is infra: the one role is the
harness's, so no contestant fault is ever decided here, and every role fault is infra
(``stage_fault``), an OOM included.
- the role's deterministic refusal (exit 3 on its own with no OOM kill: a multi-turn corpus, no
  caching reference, an engine that is not its reference's declared class, a question set under
  its name that holds other questions, which the reason names for removal) is not retried,
  since a retry repeats it. The fill reads no contestant's bytes, only the test rows and the
  references' own answers, so the exit alone decides, and the fault carries the role's last
  ``REFUSED_PREFIX`` line, scrubbed;
- any other failure of the role, an answer it could not write to the area among them (a full
  quota, a key directory it may not write), and a role that exits 0 without ``fill.json`` or
  without reporting it, is retried, and the retry asks only for the rows still missing;
- inputs that contradict the plan are not retried, as another attempt reads the same files: no
  provision or images manifest of this instance among the stage's inputs, a file that is not
  the one its manifest recorded, a ``provisioned.json`` that is not a provision record or names no
  rows entry, an
  ``images.json`` that does not name the reference image's key and entry digest or names one
  the cache entry does not have, and data that cannot be mounted.

At plan time ``check`` refuses a task with no reference that caches its corpus answers, a task
that does not load, and, as the verify stage does, a caching reference on a task that declares
an agent or on a workload of more than one turn per session (``multi_turn_caching``), whose
prompts carry the server's replies; the role's refusal of a row with more than one turn is the
second guard. It applies the verify stage's rules to the stage's shape (the role's, with the
runner's memory and the extracted image's disk, in all but time) and to the engines the role
loads."""

from __future__ import annotations

from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from hls.package.wheels import COMPUTE_CAPABILITY
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.measured import DEV_REFERENCE_FILE, write_measured
from hls.provision.model import Provisioned
from hls.provision.mounts import DataCopies, data_mounts
from hls.settings import StageName
from hls.spec.resources import ResourceShape
from hls.stages.deploy_drive import CHECKOUT, OUT, OUTPUTS_REPORT, TASK_REF
from hls.stages.inputs import input_manifest, output, read_output
from hls.stages.score import refusal
from hls.stages.verify import (ANSWERS, IMAGES_FILE, NO_GPU, PROVISIONED_FILE, answers_writer, check_engines_fit,
                               check_role_shape, image_on_node)
from hls.store.base import check_segment
from hls.substrate.container import ContainerSpec, Mount
from hls.substrate.faults import RoleOutcome
from hls.taskref import TaskRef, load_task
from hls.util import validation_errors
from hls.verify.answers import answers_area, multi_turn_caching
from hls.verify.fill import DEFAULT_CHUNK, FILL_FILE, caching_references
from hls.verify.role import REFUSED_EXIT, REFUSED_PREFIX

#: The role's name, under which its image is recorded.
ROLE = "oracle-fill"
#: Where every tenant's rows mount in the role's container (``hls.provision.rows.tenant_mount_path``).
ROWS_ROOT = "/data"


class OracleFillParams(DataCopies):
    """What the fill fills, and on what."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task whose caching references answer its test split, as the role rebuilds it.
    task: TaskRef
    #: The name in this instance's ``images.json`` of the image the role runs on: the image the
    #: task's gradings verify on.
    reference_image: Annotated[str, AfterValidator(check_segment)]
    #: The stage of this instance whose ``images.json`` names the reference image.
    images_stage: StageName = "images"
    #: The stage of this instance whose ``provisioned.json`` names the weights, the rows and the
    #: tool bundles.
    provision_stage: StageName = "provision"
    #: The role's shape: the verify role's. Its time is only the stage's default: the role runs
    #: for the stage's.
    reference_shape: ResourceShape
    #: Rows asked per chunk; each chunk's answers are banked as it finishes.
    chunk: int = Field(default=DEFAULT_CHUNK, gt=0)
    #: Whether the role also produces the task's dev reference (``hls.verify.dev_reference``),
    #: which the stage then writes into the store's measured area as the task's measured file.
    dev_reference: bool = False


class _Fill:
    """One fill attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None or ctx.runtime is None or ctx.cache is None:
            raise RuntimeError("the oracle_fill stage needs node-local scratch, a container runtime and the cache")
        self.ctx = ctx
        self.params: OracleFillParams = ctx.params

    def provisioned(self) -> Provisioned:
        """This instance's provision record, checked against its manifest."""
        ref = output(input_manifest(self.ctx, self.params.provision_stage), PROVISIONED_FILE)
        try:
            return Provisioned.model_validate_json(read_output(self.ctx, ref))
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a provision record: {validation_errors(e)}",
                                   retryable=False) from None

    def raise_fault(self, outcome: RoleOutcome) -> None:
        """Raise the fault the role's ending *outcome* is, if any: its ``stage_fault``, given the
        role's reason and made not retryable when the role refused (exit 3 on its own with no
        OOM kill)."""
        fault = stage_fault([outcome])
        if fault is None:
            return
        if outcome.ending == "exited" and outcome.oom_kills == 0 and outcome.exit_code == REFUSED_EXIT:
            reason = refusal(self.ctx.attempt_dir / "logs" / f"{ROLE}.err", REFUSED_PREFIX)
            fault = fault.model_copy(update={"reason": f"oracle fill refused: {self.ctx.scrubber.scrub(reason)}",
                                             "retryable": False})
        raise StageFault(fault)

    def run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        provisioned = self.provisioned()
        if provisioned.rows is None:
            raise StageFault.infra(f"{PROVISIONED_FILE} names no rows entry, so there are no test rows to fill",
                                   retryable=False)
        images = output(input_manifest(ctx, p.images_stage), IMAGES_FILE)
        # Every extraction and copy runs alone, so all of them come before the role.
        image = image_on_node(ctx, read_output(ctx, images), p.reference_image, images_key=images.key)
        try:
            data, env = data_mounts("oracle", provisioned, ctx.materialize, declared=p.data_copies)
        except ValueError as e:
            raise StageFault.infra(f"the fill's data cannot be mounted: {e}", retryable=False) from e
        role_dir = ctx.role_dir(ROLE)
        workdir = role_dir / "wd"
        workdir.mkdir()
        task_ref = role_dir / "task-ref.json"
        task_ref.write_text(p.task.model_dump_json())
        # The role writes its report in place, so the file it is bound to is there first.
        report = ctx.outputs_report(ROLE)
        report.touch()
        out = ctx.outputs_dir
        out.mkdir(parents=True, exist_ok=True)
        # The role runs for the stage's own time, which a site's [resources] table may change.
        shape = p.reference_shape.model_copy(update={"time_sec": ctx.stage.shape.time_sec})
        env["PYTHONPATH"] = f"{CHECKOUT}/src:{CHECKOUT}"
        if shape.gpu_kind is not None:
            env["TORCH_CUDA_ARCH_LIST"] = COMPUTE_CAPABILITY[shape.gpu_kind]
        area = ctx.store.path(answers_area(ctx.plan.task))
        area.mkdir(parents=True, exist_ok=True)
        mounts = (*data, Mount(ctx.plan.code.checkout, CHECKOUT), Mount(task_ref, TASK_REF),
                  Mount(out, OUT, read_only=False), Mount(report, OUTPUTS_REPORT, read_only=False),
                  Mount(area, ANSWERS, read_only=False))
        command = ("python", "-m", "hls.verify.fill", "--task-ref", TASK_REF, "--rows", ROWS_ROOT,
                   "--rows-key", provisioned.rows.key, "--answers", ANSWERS, "--writer", answers_writer(ctx),
                   "--gpu-kind", shape.gpu_kind or NO_GPU,
                   "--reference-image", image.file_sha256, "--out", f"{OUT}/{FILL_FILE}",
                   "--outputs-out", OUTPUTS_REPORT, "--chunk", str(p.chunk))
        if p.dev_reference:
            command += ("--dev-out", f"{OUT}/{DEV_REFERENCE_FILE}")
        spec = ContainerSpec(image=image, command=command, workdir=workdir, mounts=mounts, gpus=shape.gpus > 0)
        ctx.start_container(ROLE, spec, shape, owner="harness", env=env)
        [outcome] = ctx.wait_roles(ROLE)
        self.raise_fault(outcome)
        made = [FILL_FILE, DEV_REFERENCE_FILE] if p.dev_reference else [FILL_FILE]
        missing = [name for name in made if not (out / name).is_file()]
        if missing:
            raise StageFault.infra(f"the fill role exited 0 without writing {', '.join(missing)}", retryable=True)
        outputs = {"fill": ctx.reported_output(FILL_FILE, ROLE)}
        if p.dev_reference:
            outputs["dev_reference"] = ctx.reported_output(DEV_REFERENCE_FILE, ROLE)
            # The task's measured file is the last thing written: what runs of the task use from now on.
            key = write_measured(ctx.store, ctx.plan.task, DEV_REFERENCE_FILE,
                                 (out / DEV_REFERENCE_FILE).read_bytes())
            ctx.events.emit("dev_reference_measured", measured_key=key)
        return outputs


@register
class OracleFillStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "oracle_fill"
    params_model: ClassVar[type[BaseModel]] = OracleFillParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the reference image's extraction, a copy of each
        cache entry the fill mounts (the weights, the rows and the full tool bundle;
        ``hls.provision.mounts`` role ``oracle``), and the fill role."""
        return 5

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, OracleFillParams)
        role, module = params.reference_shape, params.task.module
        check_role_shape(cls.kind, shape, role, runner_memory_gib=runner_memory_gib, data_gib=params.data_gib)
        try:
            task = load_task(params.task)
            engines = [ref.reference.build_engine(task) for ref in caching_references(task)]
        except Exception as e:
            raise ValueError(f"task {module} does not load: {type(e).__name__}: {e}") from e
        if not engines:
            raise ValueError(f"{module}: no reference caches its answers to a corpus question, so there is nothing "
                             "to fill")
        multi_turn = multi_turn_caching(task)
        if multi_turn:
            raise ValueError(f"{module}: {'; '.join(multi_turn)}. The answers it would keep in the store's answers "
                             "area, which no scan reads, would depend on a contestant's text; set cache=False")
        check_engines_fit(module, role, [engine.resources for engine in engines])

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Fill(ctx).run()
