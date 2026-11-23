"""The sandbox_images stage: a task's sandbox images into their site store, each left alone when
the store holds it.

A task whose sandboxes each start from their own image declares the set
(``hls.provision.sandbox_images.SandboxImageSet``). ``hls provision`` plans this stage after
``provision`` when the site's store lacks one of them (``missing_images``), and leaves it out
when the store holds them all.

The stage runs in the runner and stays light. One host-side harness role, the role program
``hls.provision.sandbox_images``, does the work on the node, with the stage's shape less the
memory the stage leaves the runner and no disk of its own (``role_shape``): it pulls each image
the store lacks from its published manifest, by digest, converts it with ``apptainer pull`` on
node-local disk, several at a time, and puts the file and its record into the store. Nothing is
built: an image is the published one. The role reaches the registry over the node's network, and
the store as the directory the site names for it (``Settings.sandbox_stores``), which the plan
refuses to leave unnamed.

Each image's outcome is an event (``sandbox_image_found``, ``sandbox_image_pulled``,
``sandbox_image_failed``). When every image is in the store the output ``sandbox-images.json``
names each with the source it was pulled from, the digest of its published manifest in it, and
its file's sha256 and size (``SandboxImagesRecord``).

An image that cannot be pulled leaves nothing in the store under its name, and the stage fails
with one line that names each such image and why. The images the role did pull stay, with their
records, so the retry and any later provision pull only the rest. The fault is an infra fault
and is retried once: a registry that did not answer may answer again, and a pull that fails the
same way twice is reported as it failed.

A registry login is the images stage's: ``site.registry_username`` and the token granted to this
stage as ``REGISTRY_TOKEN_GRANT``, both or neither, which the planner checks. The user name goes
in the role's declared environment and the token through its env file."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from hls.pipeline.stage import Stage, StageContext, StageFault, budget_problem, register, stage_fault
from hls.provision.sandbox_images import (FAILED_EXIT, ImageRecord, SandboxImageSet, failure_line, read_report)
from hls.spec.resources import ResourceShape
from hls.stages.provision import role_shape
from hls.substrate.apptainer.images import REGISTRY_TOKEN_GRANT, REGISTRY_USER_VAR

#: The role program, run as ``<plan python> -m ROLE_MODULE``.
ROLE_MODULE = "hls.provision.sandbox_images"
#: The stage's one role.
ROLE = "sandbox-images"
#: The stage's output: every image with its source and its file's digest.
RECORD_FILE = "sandbox-images.json"

__all__ = ["RECORD_FILE", "ROLE", "ROLE_MODULE", "SandboxImagesParams", "SandboxImagesRecord",
           "SandboxImagesStage"]


class SandboxImagesParams(BaseModel):
    """What the sandbox_images stage leaves in the site's store."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task's declared set: the store, the repository, each image with its digest, and how
    #: many are pulled at a time.
    images: SandboxImageSet


class SandboxImagesRecord(BaseModel):
    """The stage's output ``sandbox-images.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The site store the images are in.
    store: str
    #: Each image's record in the store, in the set's order.
    images: list[ImageRecord]


def role_argv(python: Path, *, images: Path, store: Path, work: Path, report: Path) -> list[str]:
    """The role's command: the role program under the interpreter *python*, with the file that
    holds the set, the store's directory, the node-local directory the pulls work in and the file
    the outcomes go to."""
    return [str(python), "-m", ROLE_MODULE, "--images", str(images), "--store", str(store), "--work", str(work),
            "--report", str(report)]


@register
class SandboxImagesStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "sandbox_images"
    params_model: ClassVar[type[BaseModel]] = SandboxImagesParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    # One retry: a registry that did not answer may answer the second time, and the images the
    # first attempt pulled are held.
    default_retries: ClassVar[int] = 1

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """One: the role that pulls."""
        return 1

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        if shape.gpus:
            raise ValueError("sandbox_images runs on the host side and asks for no GPU")
        problem = budget_problem(shape, [(role_shape(shape, runner_memory_gib), False)],
                                 runner_memory_gib=runner_memory_gib)
        if problem is not None:
            raise ValueError(problem)

    @classmethod
    def sandbox_stores(cls, params: SandboxImagesParams) -> tuple[str, ...]:
        """The store the images go to, which the site must name."""
        return (params.images.store,)

    def run(self, ctx: StageContext) -> dict[str, str]:
        params: SandboxImagesParams = ctx.params
        images = params.images
        store = ctx.plan.settings.sandbox_stores[images.store]
        own = ctx.role_dir(ROLE)
        declared, report_path = own / "images.json", own / "report.json"
        declared.write_text(images.model_dump_json())
        # The plan holds both halves of the login or neither (``registry_login_problem``).
        user = ctx.plan.settings.site.registry_username
        login = user is not None and REGISTRY_TOKEN_GRANT in ctx.secrets
        env = {"PYTHONPATH": str(ctx.plan.code.checkout), **({REGISTRY_USER_VAR: user} if login else {})}
        argv = role_argv(ctx.plan.toolchain.python, images=declared, store=store, work=own / "pulls",
                         report=report_path)
        ctx.start_role(ROLE, argv, role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib), env=env,
                       secret_env=(REGISTRY_TOKEN_GRANT,) if login else (), owner="harness")
        outcomes = ctx.wait_roles()
        fault = stage_fault(outcomes)
        report = read_report(report_path)
        for outcome in report.outcomes if report is not None else ():
            record = outcome.record
            if record is None:
                ctx.events.emit("sandbox_image_failed", name=outcome.name, why=ctx.scrubber.scrub(outcome.why or ""))
            else:
                ctx.events.emit(f"sandbox_image_{outcome.status}", name=record.name, source=record.source,
                                sha256=record.sha256, size=record.size, seconds=outcome.seconds)
        failed = report.failed() if report is not None else []
        if failed and any(o.name == ROLE and o.exit_code == FAILED_EXIT for o in outcomes):
            raise StageFault.infra(f"sandbox images failed: {ctx.scrubber.scrub(failure_line(failed))}",
                                   retryable=True, images=",".join(o.name for o in failed), store=images.store)
        if fault is not None:
            raise StageFault(fault)
        if report is None:
            raise StageFault.infra(f"role {ROLE} reported no outcome of the sandbox images", retryable=True, role=ROLE)
        records = {o.name: o.record for o in report.outcomes if o.record is not None}
        unreported = [i.name for i in images.images if i.name not in records]
        if unreported:
            raise StageFault.infra(f"role {ROLE} reported no record of sandbox images {', '.join(unreported)}",
                                   retryable=True, role=ROLE)
        record = SandboxImagesRecord(store=images.store, images=[records[i.name] for i in images.images])
        return {"sandbox_images": ctx.write_output(RECORD_FILE, record.model_dump_json(indent=2).encode())}
