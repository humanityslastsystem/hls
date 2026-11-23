"""The verify stage: a task's verifiers over a drive's trace, and the verdict.

The stage runs in the runner and stays light: it reads four small files, the trace's prompts when
a caching reference may be served whole (step 3), and starts one role.
1. It reads ``drive.json`` from this instance's drive stage (``DriveRecord``). A drive that did
   not serve (the package was an invalid artifact, the server crashed before it was ready, or
   it never became ready) is a scored result, not a fault. The stage writes a failing verdict
   whose ``failure_mode`` is the drive's outcome, with no per-event map, and a ``verify.json``
   with no engines, and starts no role, so no GPU is held for a trace that does not exist.
2. For a served drive it checks the trace's ``END.json`` against the digest the drive's
   manifest recorded. ``END.json`` holds each chunk's digest and the role checks every chunk
   against it as it reads, so the role verifies the trace that was published and no other.
   The stage reads no event itself.
3. It pins the reference image: this instance's images stage names it in ``images.json`` with
   its cache key and entry digest, and the stage extracts it on the node with ``node_image``
   against that digest; the container's start records the image file's digest under the
   role's name. The tree is read-only to the container, and the stage's disk declares it
   (``KEPT_DISK_GIB``). Then it copies the reference's data to the node (``data_mounts``
   through ``materialize``). The task model's weights are copied only when a reference engine
   may load (``hls.verify.run.may_deploy``): not when the verifiers have no engine, nor when
   every engine is a caching reference whose answer to every prompt of the trace is in the
   task's answers area, under the key the role reads (step 4). The area only grows, so the role
   finds every answer the stage found; a storage fault that hides one leaves the role's engine
   without weights, which fails to load, infra that is retried. A copy or an extraction runs
   alone, so both come before the role.
4. It starts the verify role, ``hls.verify.role``, as a harness container on that image. The
   role gets the plan's checkout read-only at ``/opt/hls-src`` on ``PYTHONPATH``, the task
   reference, a copy of the calibration the plan pins (``hls.measured``), checked against the
   blob id it was pinned by, read-only, the trace read-only from the store, the attempt's outputs
   directory writable,
   the file it reports its outputs in, the reference role's data mounts (the task model's
   weights and the full tool bundle; ``hls.provision.mounts``), the stage's GPUs, and
   ``TORCH_CUDA_ARCH_LIST`` for their kind. The reference engines run inside the role, on
   those GPUs. It runs for the stage's own time. The role also gets the task's reference answers
   area of the store, ``oracle/<task>/``, writable (``hls.verify.answers``): a caching
   reference's corpus answers are served from there and every new one is added there, under a
   writer name unique to this attempt (``_Verify.writer``), keyed by the GPU kind (``cpu`` for
   a role without GPUs) and the digest of the image file the role runs. These are the
   reference's answers to the withheld test split: the area lies outside every instance, so it
   is neither an output of the stage nor among the files finalize scans and publishes, and
   only harness containers mount it (this one, and the oracle fill's, which fills it ahead of
   the gradings); no contestant container may mount anything in the store.
5. Its outputs are the role's files, each recorded by the role's own report of it
   (``reported_output``): ``verdict`` (``verdict.json``), ``verify`` (``verify.json``) and
   ``replies.<index>.<Verifier>`` for each ``replies.<index>.<Verifier>.jsonl.gz``.

Every failure is infra; they differ only in whether a retry can help. The one
role is the harness's, so no contestant fault is ever decided here: a verifier that fails the
contestant writes a verdict, and every role fault is infra (``stage_fault``), its OOM kills
and its memory limit included.
- the role's deterministic refusal (exit 3: an empty trace, a pooled gate wired two ways, a
  missing or stale calibration, a reference stack that drifted from it or runs on other
  silicon) is not retried, since a retry repeats it. The fault carries the role's reason,
  scrubbed. It is a refusal only from a role that exited on its own with no OOM kill, and
  only with the role's own ``REFUSED_PREFIX`` line on its stderr: the role logs what it reads,
  replies a contestant's server wrote among it, so an exit 3 without that line is any other
  failure, and a line of another form, such as provision's ``refused:``, decides nothing;
- any other failure of the role (an engine that failed to load, answer or stop, a chunk that
  does not match, the runtime, the scheduler, the deadline, memory), and a role that exits 0
  without its files or without reporting them, is retried;
- inputs that contradict the plan are not retried, as another attempt reads the same files:
  no drive, images or provision manifest of this instance, one without the output the stage
  reads or with it at another key than its attempt's ``outputs/<file>``, a file that is not
  the one its manifest recorded, a ``drive.json`` that is not a ``DriveRecord``, a
  ``provisioned.json`` that is not a provision record, an ``images.json`` that does not name
  the reference image's key and entry digest or names one the cache entry does not have, and
  reference data that cannot be mounted.

At plan time ``check`` refuses what the role cannot run. The stage's resources are its one
role's, ``reference_shape``, with the runner's memory and the extracted image's disk beside
it, in all but time: a site may give the stage another time, which the role then runs for.
The task must load. A caching reference on a task that declares an agent or on a workload of
more than one turn per session is refused (``hls.verify.answers.multi_turn_caching``): its
prompts carry the server's replies, so the answers it kept in the store would depend on a
contestant's text. A verifier that loads two or more engines at once is held only when every
one of them places itself on GPUs of its own (``ReferenceEngine.placeable``: the role's engine
host gives it the role's lowest free GPUs) and the role has a GPU for each (``verifier_gpus``);
an engine that does not place itself takes every GPU the role sees, so it shares the role with
no other (``multi_engine_problem``). One engine over several GPUs is one engine and is allowed (vLLM
spreads a tensor-parallel engine over the visible devices), and the role's GPUs must cover the
most any engine asks for, of the kind the engines ask for.

The oracle fill stage (``hls.stages.oracle_fill``) pins its image and names its answers writer
as this stage does (``image_on_node``, ``answers_writer``), and its role is held to the same
rules (``check_role_shape``, ``check_engines_fit``)."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError

from hls.core.engine import Engine, ResourceRequest
from hls.core.verdict import Verdict
from hls.drive.record import DRIVE_FILE, TRACE_END, DriveRecord
from hls.package.wheels import COMPUTE_CAPABILITY
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.provision.model import Provisioned
from hls.provision.mounts import DataCopies, data_mounts
from hls.settings import StageName
from hls.spec.resources import ResourceShape
from hls.spec.taskspec import TaskSpec
from hls.stages.deploy_drive import CHECKOUT, OUT, OUTPUTS_REPORT, TASK_REF
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.inputs import measured_copies
from hls.stages import score
from hls.stages.package import IMAGES_OUTPUT
from hls.store.base import check_segment
from hls.store.digest import check_sha256_hex, sha256_bytes
from hls.store.manifest import Manifest, OutputRef
from hls.store.posix import PosixStore
from hls.substrate.container import ContainerSpec, Mount, PinnedImage
from hls.substrate.faults import RoleOutcome
from hls.taskref import TaskRef, load_task
from hls.trace.store import TraceIncomplete, iter_events
from hls.util import validation_errors
from hls.verify.answers import StoreAnswers, answers_area, multi_turn_caching
from hls.verify.role import REFUSED_EXIT, REFUSED_PREFIX, VERIFY_FILE, VerifyRecord
from hls.verify.run import may_deploy, write_verdict

#: The role's name, under which its image is recorded.
ROLE = "verify"
#: Where the trace mounts in the role's container. The checkout, the task reference, the out
#: directory and the outputs report mount where the drive stage's driver has them.
TRACE = "/hls/trace"
#: Where the task's reference answers area mounts in the role's container.
ANSWERS = "/hls/answers"
#: Where the copies of the measured files the plan pins mount in the role's container.
MEASURED = "/hls/measured"
#: The GPU kind an answer is keyed by when the role has no GPU.
NO_GPU = "cpu"
#: The provision stage's record.
PROVISIONED_FILE = "provisioned.json"
#: The images stage's record of each image's key and digests.
IMAGES_FILE = "images.json"
#: The verdict file, the verification core's.
VERDICT_FILE = "verdict.json"
# The drive stage's outputs this stage reads, by name (``hls.stages.deploy_drive``), and the
# provision stage's.
_DRIVE, _TRACE, _PROVISIONED = "drive", "trace", "provisioned"
# A per-event replies file of the verification core.
_REPLIES = re.compile(r"replies\.[0-9]+\.[A-Za-z_][A-Za-z0-9_]*\.jsonl\.gz")


class VerifyParams(DataCopies):
    """What the verify stage verifies, and on what."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task whose verifiers run, as the role rebuilds it.
    task: TaskRef
    #: The name in this instance's ``images.json`` of the image the role runs on: the task's
    #: measured reference image.
    reference_image: Annotated[str, AfterValidator(check_segment)]
    #: The stage of this instance whose ``drive.json`` and trace are verified.
    drive_stage: StageName = "deploy_drive"
    #: The stage of this instance whose ``images.json`` names the reference image.
    images_stage: StageName = "images"
    #: The stage of this instance whose ``provisioned.json`` names what the reference mounts.
    provision_stage: StageName = "provision"
    #: The role's shape. Its time is only the stage's default: the role runs for the stage's.
    reference_shape: ResourceShape
    #: The eval the verdict belongs to, recorded in ``verify.json``.
    eval_id: Annotated[str, AfterValidator(check_segment)] = "terminal"


def _verifier_engines(task: TaskSpec) -> list[tuple[str, list[Engine]]]:
    """Each of *task*'s verifiers, by class name, with each distinct engine it loads together.
    An engine counts once per verifier however many of its fields name it. Engines are built,
    never loaded."""
    out = []
    for tenant in task.effective_tenants():
        for verifier in tenant.verifiers:
            configs = {id(cfg): cfg for cfg in verifier.engines()}.values()
            out.append((type(verifier).__name__, [cfg.build_engine(task) for cfg in configs]))
    return out


def reference_engines(task: TaskSpec) -> list[tuple[str, list[ResourceRequest]]]:
    """Each of *task*'s verifiers, by class name, with the resource request of each distinct
    engine it loads together."""
    return [(name, [engine.resources for engine in engines]) for name, engines in _verifier_engines(task)]


def verifier_gpus(task: TaskSpec) -> int:
    """The most GPUs one of *task*'s verifiers holds at once: the GPUs its distinct engines ask
    for, summed, since they are live together while it runs, and the largest such sum over its
    verifiers, since the verification core releases one verifier's engines before it loads the
    next's; 0 for a task whose verifiers load none."""
    return max((sum(r.gpu_count for r in requests) for _, requests in reference_engines(task)), default=0)


def multi_engine_problem(task: TaskSpec, role_gpus: int) -> str | None:
    """Why a verify role of *role_gpus* GPUs cannot hold the engines one of *task*'s verifiers
    loads together, or None. A verifier that loads several is held when every one places
    itself (``ReferenceEngine.placeable``: the role's engine host gives it the role's lowest
    free GPUs) and the role has a GPU for each. An engine that does not place itself takes every
    GPU the role sees, so it shares the role with no other, however many GPUs it has."""
    for name, engines in _verifier_engines(task):
        if len(engines) < 2:
            continue
        unplaced = sorted({type(engine).__name__ for engine in engines if not getattr(engine, "placeable", False)})
        if unplaced:
            return (f"{name} loads {len(engines)} reference engines at once, and these take every GPU the role "
                    f"sees, so they cannot share it: {', '.join(unplaced)}")
        wanted = sum(engine.resources.gpu_count for engine in engines)
        if role_gpus < wanted:
            return (f"{name} loads {len(engines)} reference engines at once, on {wanted} GPUs in all; "
                    f"reference_shape has {role_gpus}")
    return None


def image_entry(images: bytes, name: str, *, images_key: str) -> tuple[str, str]:
    """The cache key and the entry digest that *images*, the bytes of the images stage's
    ``images.json`` at store key *images_key*, records for image *name*. One that does not name
    them is an infra fault that is not retried: another attempt reads the same file."""
    try:
        entry = json.loads(images)[name]
        return check_segment(entry["key"]), check_sha256_hex(entry["entry_sha256"])
    except (ValueError, KeyError, TypeError) as e:
        raise StageFault.infra(f"{images_key} does not name image {name}'s key and entry digest",
                               retryable=False) from e


def image_on_node(ctx: StageContext, images: bytes, name: str, *, images_key: str) -> PinnedImage:
    """Image *name*, extracted on the node and pinned to the entry digest that *images*, the
    bytes of the images stage's ``images.json`` at store key *images_key*, records for it
    (``image_entry``). A cache entry of another digest is an infra fault that is not retried
    (``StageContext.node_image``): another attempt reads the same files."""
    return ctx.node_image(*image_entry(images, name, images_key=images_key))


def answers_writer(ctx: StageContext) -> str:
    """The attempt's name among the writers of the task's answers area, one store segment:
    ``<instance with "/" as "+">+<stage>+<attempt>``. No other attempt gets it while the
    instance's ids hold no "+" (a rollout id never does), and a name used anyway loses its
    write rather than replace a file (``StoreAnswers.bank``)."""
    return f"{ctx.plan.instance.replace('/', '+')}+{ctx.stage.name}+{ctx.attempt}"


def writer_instance(writer: str) -> str | None:
    """The instance whose attempt the answers writer *writer* is (``answers_writer``); None for a
    name no attempt is given, one without a stage and an attempt number after the instance."""
    parts = writer.rsplit("+", 2)
    if len(parts) != 3 or not parts[2].isdigit():
        return None
    return parts[0].replace("+", "/")


def check_role_shape(kind: str, shape: ResourceShape, role: ResourceShape, *, runner_memory_gib: int,
                     data_gib: int) -> None:
    """Refuse a stage of kind *kind* that runs one role of shape *role* when the stage's *shape*
    is not *role* with *runner_memory_gib* more memory for the runner, ``KEPT_DISK_GIB`` more
    disk for the image it extracts and *data_gib* for the copies of its data, in all but time: a
    site may give the stage another time, which the role then runs for."""
    want = role.model_copy(update={"memory_gib": role.memory_gib + runner_memory_gib,
                                   "disk_gib": role.disk_gib + KEPT_DISK_GIB + data_gib, "time_sec": shape.time_sec})
    if shape != want:
        raise ValueError(f"the {kind} stage runs one role, so its resources must be reference_shape's with "
                         f"{runner_memory_gib} GiB more memory for the runner, {KEPT_DISK_GIB} GiB more disk for "
                         f"the image it extracts and {data_gib} GiB for the copies of its data, {want}, in all but "
                         f"time; they are {shape}")


def check_engines_fit(module: str, role: ResourceShape, requests: list[ResourceRequest]) -> None:
    """Refuse a role of shape *role* that cannot hold the reference engines of task *module*
    whose resource *requests* are given: fewer GPUs than the widest asks for, or another kind."""
    gpus = max((r.gpu_count for r in requests), default=0)
    kinds = {r.gpu_kind for r in requests if r.gpu_count and r.gpu_kind is not None}
    if role.gpus < gpus:
        raise ValueError(f"a reference engine of {module} asks for {gpus} GPUs; reference_shape has {role.gpus}")
    if role.gpus and kinds - {role.gpu_kind}:
        raise ValueError(f"the reference engines of {module} ask for {sorted(kinds)} GPUs; reference_shape has "
                         f"{role.gpu_kind}")


def _reason(err: Path) -> str | None:
    """The last refusal line in the end of the role's stderr *err*, as far back as the score
    stage searches its own (``hls.stages.score.REASON_SEARCH_BYTES``), without its prefix; None
    when there is none or the log cannot be read."""
    try:
        with open(err, "rb") as f:
            f.seek(max(0, os.fstat(f.fileno()).st_size - score.REASON_SEARCH_BYTES))
            text = f.read().decode(errors="replace")
    except OSError:
        return None
    lines = [line for line in text.splitlines() if line.startswith(REFUSED_PREFIX)]
    return lines[-1].removeprefix(REFUSED_PREFIX) if lines else None


class _Verify:
    """One verify attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None or ctx.runtime is None or ctx.cache is None:
            raise RuntimeError("the verify stage needs node-local scratch, a container runtime and the cache")
        self.ctx = ctx
        self.params: VerifyParams = ctx.params
        #: The attempt's outputs directory, which the role writes.
        self.out = ctx.outputs_dir

    # inputs

    def manifest(self, stage: str) -> Manifest:
        """This instance's manifest of *stage*, among the stage's inputs."""
        ctx = self.ctx
        key = ctx.plan.layout.manifest(stage)
        manifest = ctx.inputs.get(key)
        if manifest is None:
            raise StageFault.infra(f"the stage was planned without input manifest {key}", retryable=False)
        if (manifest.instance, manifest.stage) != (ctx.plan.instance, stage):
            raise StageFault.infra(f"input manifest {key} is the {manifest.stage} manifest of {manifest.instance}",
                                   retryable=False)
        return manifest

    def output(self, manifest: Manifest, name: str, file: str) -> OutputRef:
        """Output *name* of *manifest*, which must be its attempt's ``outputs/<file>``."""
        ref = manifest.outputs.get(name)
        if ref is None:
            raise StageFault.infra(f"{manifest.instance} stage {manifest.stage} published no {name}", retryable=False)
        want = self.ctx.plan.layout.attempt_file(manifest.stage, manifest.attempt, f"outputs/{file}")
        if ref.key != want:
            raise StageFault.infra(f"output {name} of the {manifest.stage} manifest is {ref.key}, which is not its "
                                   f"attempt's outputs/{file}", retryable=False)
        return ref

    def read(self, ref: OutputRef) -> bytes:
        """The bytes of output *ref*, checked against the digest its manifest recorded."""
        data = self.ctx.store.read_bytes(ref.key)
        digest = sha256_bytes(data)
        if digest != ref.sha256:
            raise StageFault.infra(f"{ref.key} does not match its manifest: {digest} is not {ref.sha256}",
                                   retryable=False)
        return data

    def provisioned(self) -> Provisioned:
        ref = self.output(self.manifest(self.params.provision_stage), _PROVISIONED, PROVISIONED_FILE)
        try:
            return Provisioned.model_validate_json(self.read(ref))
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a provision record: {validation_errors(e)}",
                                   retryable=False) from None

    def image(self) -> PinnedImage:
        """The reference image, extracted on the node and pinned to the entry digest this
        instance's ``images.json`` records."""
        ref = self.output(self.manifest(self.params.images_stage), IMAGES_OUTPUT, IMAGES_FILE)
        return image_on_node(self.ctx, self.read(ref), self.params.reference_image, images_key=ref.key)

    def writer(self) -> str:
        """This attempt's name among the writers of the answers area."""
        return answers_writer(self.ctx)

    def may_deploy(self, trace: Path, area: Path, gpu_kind: str, image: str) -> bool:
        """Whether the role may deploy a reference engine (``hls.verify.run.may_deploy``), and so
        needs the task model's weights: from the answers the task's *area* holds for the role's
        *gpu_kind* and reference *image*, which is what the role reads, and the prompts of the
        events in *trace*, read one chunk at a time and only when a reference may be served
        whole. A trace that does not read whole may deploy: the role reads it again and reports
        it."""
        def prompts():
            for ev in iter_events(trace):
                yield ev.input

        task = load_task(self.params.task)
        answers = StoreAnswers(PosixStore(area), gpu_kind=gpu_kind, image=image, writer=self.writer())
        try:
            deploys = may_deploy(task, prompts(), answers)
        except TraceIncomplete:
            deploys = True
        self.ctx.events.emit("reference_may_deploy", deploys=deploys, answers_read=len(answers.read))
        return deploys

    # results

    def outputs(self) -> dict[str, str]:
        """The role's files, each recorded by the role's own report of it."""
        names = {"verdict": VERDICT_FILE, "verify": VERIFY_FILE}
        missing = [f for f in names.values() if not (self.out / f).is_file()]
        if missing:
            raise StageFault.infra(f"the verify role exited 0 without writing {missing}", retryable=True)
        replies = sorted(p.name for p in self.out.iterdir() if _REPLIES.fullmatch(p.name))
        names |= {name.removesuffix(".jsonl.gz"): name for name in replies}
        return {name: self.ctx.reported_output(file, ROLE) for name, file in names.items()}

    def not_served(self, outcome: str, reason: str) -> dict[str, str]:
        """The failing verdict of a drive that did not serve, written by the stage itself, and a
        ``verify.json`` with no engines."""
        ctx = self.ctx
        assert ctx.scratch is not None
        verdict = Verdict(passed=False, per_event={}, failure_mode=outcome, diagnostics={"reason": reason})
        ctx.scratch.mkdir(parents=True, exist_ok=True)
        # The verification core's writer, so the file is the one it writes for this verdict.
        with tempfile.TemporaryDirectory(dir=ctx.scratch) as tmp:
            write_verdict(Path(tmp), verdict, [])
            data = (Path(tmp) / VERDICT_FILE).read_bytes()
        record = VerifyRecord(eval_id=self.params.eval_id, engines=[], libraries={})
        ctx.events.emit("verify_skipped", outcome=outcome)
        return {"verdict": ctx.write_output(VERDICT_FILE, data),
                "verify": ctx.write_output(VERIFY_FILE, record.model_dump_json(indent=2).encode())}

    def raise_fault(self, outcome: RoleOutcome) -> None:
        """Raise the fault the role's ending *outcome* is, if any: a refusal, exit 3 on its own
        with no OOM kill and the role's own refusal line, is not retried."""
        if (outcome.owner, outcome.ending, outcome.oom_kills, outcome.exit_code) == ("harness", "exited", 0,
                                                                                      REFUSED_EXIT):
            reason = _reason(self.ctx.attempt_dir / "logs" / f"{ROLE}.err")
            if reason is not None:
                raise StageFault.infra(f"verify refused: {self.ctx.scrubber.scrub(reason)}", retryable=False)
        fault = stage_fault([outcome])
        if fault is not None:
            raise StageFault(fault)

    # the stage

    def run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        drive = self.manifest(p.drive_stage)
        try:
            record = DriveRecord.model_validate_json(self.read(self.output(drive, _DRIVE, DRIVE_FILE)))
        except ValidationError as e:
            raise StageFault.infra(f"{DRIVE_FILE} is not a drive record: {validation_errors(e)}",
                                   retryable=False) from None
        if record.outcome != "served":
            assert record.reason is not None
            return self.not_served(record.outcome, record.reason)
        end = self.output(drive, _TRACE, TRACE_END)
        self.read(end)
        trace = ctx.store.path(end.key).parent
        provisioned = self.provisioned()
        # The role runs for the stage's own time, which a site's [resources] table may change.
        shape = p.reference_shape.model_copy(update={"time_sec": ctx.stage.shape.time_sec})
        area = ctx.store.path(answers_area(ctx.plan.task))
        area.mkdir(parents=True, exist_ok=True)
        # Every extraction and copy runs alone, so all of them come before the role.
        image = self.image()
        deploys = self.may_deploy(trace, area, shape.gpu_kind or NO_GPU, image.file_sha256)
        try:
            data, env = data_mounts("reference", provisioned, ctx.materialize, declared=p.data_copies, weights=deploys)
        except ValueError as e:
            raise StageFault.infra(f"the reference's data cannot be mounted: {e}", retryable=False) from e
        role_dir = ctx.role_dir(ROLE)
        workdir = role_dir / "wd"
        workdir.mkdir()
        task_ref = role_dir / "task-ref.json"
        task_ref.write_text(p.task.model_dump_json())
        measured = measured_copies(ctx, role_dir / "measured")
        # The role writes its report in place, so the file it is bound to is there first.
        report = ctx.outputs_report(ROLE)
        report.touch()
        self.out.mkdir(parents=True, exist_ok=True)
        env["PYTHONPATH"] = f"{CHECKOUT}/src:{CHECKOUT}"
        if shape.gpu_kind is not None:
            env["TORCH_CUDA_ARCH_LIST"] = COMPUTE_CAPABILITY[shape.gpu_kind]
        mounts = (*data, Mount(ctx.plan.code.checkout, CHECKOUT), Mount(task_ref, TASK_REF), Mount(trace, TRACE),
                  Mount(self.out, OUT, read_only=False), Mount(report, OUTPUTS_REPORT, read_only=False),
                  Mount(area, ANSWERS, read_only=False), Mount(role_dir / "measured", MEASURED))
        command = ("python", "-m", "hls.verify.role", "--task-ref", TASK_REF, "--trace", TRACE, "--out", OUT,
                   "--outputs-out", OUTPUTS_REPORT, "--eval-id", p.eval_id, "--answers", ANSWERS,
                   "--writer", self.writer(), "--gpu-kind", shape.gpu_kind or NO_GPU,
                   "--reference-image", image.file_sha256)
        if measured.calibration is not None:
            command += ("--calibration", f"{MEASURED}/{measured.calibration.name}")
        spec = ContainerSpec(image=image, command=command, workdir=workdir, mounts=mounts, gpus=shape.gpus > 0)
        ctx.start_container(ROLE, spec, shape, owner="harness", env=env)
        [outcome] = ctx.wait_roles(ROLE)
        self.raise_fault(outcome)
        return self.outputs()


@register
class VerifyStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "verify"
    params_model: ClassVar[type[BaseModel]] = VerifyParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    #: The verdict and the replies files record what the contestant's server said
    #: (``Stage.contestant_outputs``).
    contestant_outputs: ClassVar[tuple[str, ...]] = (VERDICT_FILE, "replies.")
    #: The verify role logs what it reads, the server's replies among it
    #: (``Stage.contestant_log_roles``).
    contestant_log_roles: ClassVar[tuple[str, ...]] = (ROLE,)

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the reference image's extraction, a copy of each
        cache entry the reference mounts (the weights and the full tool bundle;
        ``hls.provision.mounts``), and the verify role."""
        return 4

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, VerifyParams)
        role, module = params.reference_shape, params.task.module
        check_role_shape(cls.kind, shape, role, runner_memory_gib=runner_memory_gib, data_gib=params.data_gib)
        try:
            task = load_task(params.task)
            verifiers = reference_engines(task)
        except Exception as e:
            raise ValueError(f"task {module} does not load: {type(e).__name__}: {e}") from e
        multi_turn = multi_turn_caching(task)
        if multi_turn:
            raise ValueError(f"{module}: {'; '.join(multi_turn)}. The answers it keeps in the store's answers area, "
                             "which no scan reads, would depend on a contestant's text; set cache=False")
        problem = multi_engine_problem(task, role.gpus)
        if problem is not None:
            raise ValueError(f"{module}: {problem}")
        check_engines_fit(module, role, [r for _, requests in verifiers for r in requests])

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Verify(ctx).run()
