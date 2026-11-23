"""The baseline_warm stage: the engine cache entry every drive of the task's measured baseline
starts its server from, built once when the store lacks it.

A measured baseline's server builds what its engines need at its first load: tab_completion's vLLM
baseline JIT-builds FlashInfer's ``fused_moe_90`` for over an hour on an H200
(``hls.stages.engine_cache.COLD_BUILD_SEC``), past a drive's readiness budget. So every drive of the
baseline (``hls.stages.deploy_drive``, ``DeployDriveParams.engine_cache_owner``) starts from a
checked copy of one stored entry for exactly that server, which this stage owns, as a calibration's
warm stage owns a pool member's (``hls.stages.engine_cache``):

1. it reads this instance's ``provisioned.json`` and ``images.json`` and keys the entry by the
   baseline's image, its start (the catalog's entrypoint and variables and the serve command), its
   GPUs, its weights, the build environment and the seal it is served behind
   (``hls.stages.deploy_drive.served_cache_key``), as each drive of it does. When the entry is
   published already, by this stage of an earlier instance, the stage records it and starts no
   role, copies nothing and extracts nothing;
2. otherwise it copies the artifact role's data and pins the baseline's image, and serves the
   baseline once with an empty engine cache, as the harness's role ``server``
   (``ServedAttempt.start_server``), with a drive's image, command, variables, data, GPUs, CPUs and
   seal. It sets no build variable of its own, so what it builds is what a drive's server would
   build, and a drive's server finds it built. It is the harness's role, so it runs without a
   contestant's process ceiling and disk watch, and with no seal probe before it; none of these
   changes what is built. When its runtime's start, its load budget and its margin do not fit
   before the allocation's end, it is not started (infra, retried);
3. it waits for the server's command to start, then for its health and inference gates
   (``ServedAttempt.wait_ready``), together within the load budget from the command's start: the
   stage's time less its staging, its runtime's start and its margin (``load_sec``), so a stage that
   a ``[resources]`` table gives more time gives the load as much more. The ``baseline_warm.ready``
   event records the seconds from the command's start to ready and the server's CPUs;
4. it stops the server as planned, publishes its engine cache as the entry, the engine's temporary
   files removed first, unless another stage published it first, whose entry it then leaves, and
   writes ``engine-cache.json`` (output ``engine_cache``), whose entry each drive of the baseline is
   seeded from.

The baseline is the harness's own server, so every failure is infra: a server that ends before it
is ready, one not ready within its load budget or the stage's time, a stop that faults, a node that
cannot hold a graded server's per-uid limits and a publish that fails are retried; a published
entry that is corrupt, a tree the cache refuses, a task module that does not import and inputs that
contradict the plan are not."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, NoReturn, Self

from pydantic import BaseModel, Field, model_validator

from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.spec.resources import ResourceShape
from hls.stages import engine_cache
from hls.stages.deploy_drive import ARTIFACT, ImageArtifact, ServedAttempt, ServedParams, served_cache_key
from hls.stages.images import KEPT_DISK_GIB
from hls.substrate.container import Mount, PinnedImage
from hls.taskref import load_task

#: The stage's name in every plan that serves the measured baseline.
WARM_BASELINE = "warm_baseline"
#: The role name of the baseline's server in this stage.
SERVER = "server"

log = logging.getLogger(__name__)


class BaselineWarmParams(ServedParams):
    """The measured baseline whose engine cache the stage builds, served as its drives serve it."""

    #: The measured baseline: an image of this instance's ``images.json``.
    artifact: ImageArtifact
    #: Seconds of the stage's time kept after the load, in which it stops the server and publishes
    #: its engine cache: the planner's ``hls.pipeline.pipelines.DRIVE_MARGIN_SEC``.
    margin_sec: int = Field(ge=0)

    @model_validator(mode="after")
    def _on_gpus(self) -> Self:
        if self.artifact_shape.gpu_kind is None:
            raise ValueError("a measured baseline's engine cache is keyed by its GPU kind, and this one runs on none")
        return self


def load_sec(params: BaselineWarmParams, time_sec: int) -> int:
    """The seconds a stage of *params* given *time_sec* gives the baseline's load from its command's
    start: its time less its staging, its runtime's start and its margin."""
    return time_sec - params.staging_sec - params.runtime_start_sec - params.margin_sec


class _Warm(ServedAttempt):
    """One baseline_warm attempt."""

    def __init__(self, ctx: StageContext) -> None:
        super().__init__(ctx)
        self.params: BaselineWarmParams = ctx.params

    def run(self) -> dict[str, str]:
        """The stage's work (the module docstring), every role it started stopped however it
        ends."""
        try:
            outputs = self._run()
        except Exception:
            try:
                self.stop_all()
            except Exception:
                log.exception("stopping the stage's roles failed")
            raise
        self.stop_all()
        return outputs

    def _run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        provisioned = self.provisioned()
        entries = self.image_entries()
        key = served_cache_key(p, self.image_sha256(entries, p.artifact.image), provisioned.weights)
        found = engine_cache.published(ctx, key)
        if found is not None:
            # Built by an earlier instance's warm stage: nothing to build, copy or extract.
            return {"engine_cache": engine_cache.found_record(ctx, key, found)}
        try:
            contract = load_task(p.task).contract
        except Exception as e:
            raise StageFault.infra(f"task {p.task.module} does not load: {type(e).__name__}: {e}",
                                   retryable=False) from e
        # A node that cannot hold a graded server's per-uid limits is infra, retried.
        self.held = ctx.grading_ucounts(p.artifact_shape, None)
        # Every copy and extraction runs alone, so all of them come before the server.
        data = self.data(ARTIFACT, provisioned)
        image = self.catalog_image(entries, p.artifact.image)
        cache, seeded = engine_cache.prepare(ctx, SERVER, key, pinned=None, owner=True)
        if seeded is None:
            self.build(image, data, cache, contract)
        return {"engine_cache": engine_cache.owned_record(ctx, cache, key, seeded, must_publish=True)}

    def build(self, image: PinnedImage, data: tuple[list[Mount], dict[str, str]], cache: Path,
              contract: Any) -> None:
        """Serve the baseline from *image* with *data* and the empty engine cache *cache* until both
        readiness gates open within its load budget (``load_sec``), then stop it as planned. Raise
        infra, retried, for a load that does not fit before the allocation's end, and for any end
        but ready."""
        ctx, p = self.ctx, self.params
        budget = load_sec(p, ctx.stage.shape.time_sec)
        need = p.runtime_start_sec + budget + p.margin_sec
        if ctx.deadline_cap is not None and ctx.clock() + need > ctx.deadline_cap:
            raise self.no_room(f"the {p.runtime_start_sec}s allowed for the server's runtime to start its command, its "
                               f"{budget}s load and the {p.margin_sec}s margin after it do not fit before the "
                               "allocation's end, so the server was not started")
        handle = self.start_server(SERVER, image, p.artifact.entrypoint, p.artifact.runtime_env, data, cache,
                                   owner="harness")
        port = self.host_port(SERVER, handle)
        if port is None:
            self.ended()
        state = self.wait_ready(SERVER, handle, f"http://127.0.0.1:{port}", contract, budget_sec=budget, shared=True,
                                margin_sec=p.margin_sec)
        if state == "ended":
            self.ended()
        if state == "not_ready":
            raise StageFault.infra(f"the measured baseline was not ready within its {budget}s load", retryable=True)
        began = ctx.command_started_at(SERVER)
        assert began is not None
        ctx.events.emit("baseline_warm.ready", after_command_sec=round(ctx.clock() - began, 3),
                        cpus=p.artifact_shape.cpus)
        fault = stage_fault([self.stop(SERVER)])
        if fault is not None:
            raise StageFault(fault)

    def ended(self) -> NoReturn:
        """Raise the fault of a server that ended, or that the stage was told to stop, before it
        was ready: its role's own fault, or infra, retried."""
        outcome = self.wait(SERVER)
        fault = stage_fault([outcome])
        if fault is not None:
            raise StageFault(fault)
        raise StageFault.infra(f"the measured baseline's server exited with status {outcome.exit_code} before it "
                               "was ready", retryable=True)


@register
class BaselineWarmStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "baseline_warm"
    params_model: ClassVar[type[BaseModel]] = BaselineWarmParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the baseline image's extraction, a copy of each cache
        entry the artifact role mounts (the weights and the rows; ``hls.provision.mounts``), the
        server, and either the copy that seeds its engine cache or the copy that publishes the cache
        it built."""
        return 5

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        """The stage's resources are the artifact role's, with *runner_memory_gib* more memory for
        the runner and the disk of the image it extracts (``KEPT_DISK_GIB``) and of the copies of
        the role's data beside the role's own, in all but time; and its time leaves the load some."""
        assert isinstance(params, BaselineWarmParams)
        role = params.artifact_shape
        want = role.model_copy(update={"memory_gib": role.memory_gib + runner_memory_gib,
                                       "disk_gib": role.disk_gib + KEPT_DISK_GIB + params.data_gib,
                                       "time_sec": shape.time_sec})
        if shape != want:
            raise ValueError(f"the {cls.kind} stage runs the artifact role, so its resources must be the role's with "
                             f"{runner_memory_gib} GiB more memory for the runner and {KEPT_DISK_GIB} GiB of disk for "
                             f"the image it extracts beside {params.data_gib} GiB for the copies of its data, {want}, "
                             f"in all but time; they are {shape}")
        if load_sec(params, shape.time_sec) <= 0:
            raise ValueError(f"the {cls.kind} stage's {shape.time_sec}s leave the baseline's load no time after its "
                             f"{params.staging_sec}s of staging, {params.runtime_start_sec}s for its runtime's start "
                             f"and {params.margin_sec}s margin")

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Warm(ctx).run()
