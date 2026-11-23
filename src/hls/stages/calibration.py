"""The stages of a calibration (``hls calibrate``), each of which starts one role
of ``hls.verify.calibration_roles`` and records what it reported:

- ``calibrate_instrument``: the pinned reference's instrument check, as a harness container on
  the task's measured reference image, with the reference's data (``hls.provision.mounts``
  role ``reference``), on the reference role's GPUs, or twice them for the token-level gate,
  whose fp32 anchor is up beside the pinned reference. Output ``instrument``.
- ``calibrate_warm``: one pool member's engine cache (``warm_stage``). When the member's entry is
  published already the stage starts no role, copies nothing and extracts nothing. Otherwise a
  harness container on the member's measured image, with the reference role's data (the
  weights and the full tool bundles), sealed as the member's captures are (``member_seal``),
  loads the member once on the stage's GPUs, within the stage's time less its copies and its
  margin (``warm_load_sec``), which builds its kernels into the cache the stage then publishes.
  Output ``engine_cache`` alone.
- ``calibrate_capture``: one serving of one pool member, as a harness container on that
  member's measured image, with the test rows the serving drives and the full tool bundles
  (role ``calibration``) and a view directory on the node's disk. The container is sealed under
  the air gap (``AIR_GAP``), as a graded artifact is served, unless the task's tools call the
  network themselves (``member_seal``). Output ``capture``, the capture
  under ``capture/<fingerprint>/<engine>/<draw>.capture.jsonl.gz``.
- ``calibrate_score``: one share of servings, each captured by its own stage of this instance,
  teacher-forced through one load of the pinned reference, as a harness container on the
  reference image with the test rows (role ``calibration``), each capture mounted read-only and
  checked by the role against the sha256 its manifest recorded. Outputs ``scored``
  (``scored.json``) and ``replies_<engine>_<draw>`` for each serving.
- ``calibrate_aggregate``: the calibration, assembled by a host role on the plan's interpreter
  from the instrument's record and every score stage's replies, each named by the digest its
  manifest recorded, with each serving's engine cache as its capture stage recorded it and the
  hardware each serving's allocation ran on (``attempt_hardware``). Outputs ``calibration``, the
  ``calibration.json``, which the stage then writes into the store's measured area as the task's
  measured file (``hls.measured.store_key``), and ``aggregate`` (``aggregate.json``).

Each of the four container roles runs its engines with an engine cache on the node's disk at
``/engine-cache``, where they keep their compile products, and its stage records which one in
``engine-cache.json`` (``EngineCacheRecord``, output ``engine_cache``), by the mechanism
``hls.stages.engine_cache`` describes: an engine's cache is a cache entry keyed by the engine's
image, GPU kind and config, and by the build environment its roles run with and their network
(``engine_cache_key``), a member's for its captures, the pinned reference's for the instrument and
the score stages, holding whatever the engine keeps there.

The plan makes the first stage to run each engine its owner, in the calibration's first
allocation, which every share allocation runs after: the instrument owns the reference's entry,
and each member's warm stage owns the member's. An owner finds the entry published, or builds it
and publishes it. A warm stage whose publish fails fails, since the entry is all it is for; an
instrument's is recorded and fails nothing. Every other stage of the engine seeds its role's cache
with a checked writable copy of the entry its owner left, and a stage whose owner left no entry
starts its role's cache empty and publishes nothing. So every role of an engine after its owner
replays the compile products its owner built.

A container role gets what the verify role gets: the plan's checkout read-only at
``/opt/hls-src`` on ``PYTHONPATH``, the task reference, the campaign's settings, its data, the
stage's GPUs and ``TORCH_CUDA_ARCH_LIST`` for their kind, and the attempt's outputs directory
and the file it reports its outputs in, writable. Its image is pinned and its data copied
before it starts, since a copy or an extraction runs alone. The image is one the task declares,
named by the campaign and recorded by ``images.json`` with the digest of the file the images stage
built, which the stage pins as the verify stage does (``image_on_node``). A
role runs for the stage's own time.

Every failure is infra: a calibration has no contestant, so no contestant
fault is ever decided here, and every role fault is infra (``stage_fault``), an OOM included.
- A role's deterministic refusal (exit 3 on its own with no OOM kill) is not retried, since
  another attempt reads the same files: the reason is the role's last ``REFUSED_PREFIX`` line,
  scrubbed. A failed instrument check, a serving that measured no token and a calibration that
  fails its own checks (the precision target among them) end the instance this way, as does a
  published engine cache entry that is corrupt, which no role is seeded from.
- Any other failure of a role, a serving whose drive captured nothing, an engine the role's engine
  host gave up on (``ENGINE_FAULT_EXIT``: it did not load or release within its bound, or it died;
  its reason the role's last ``ENGINE_FAULT_PREFIX`` line) and a role that exits 0 without writing
  or reporting an output among them, is retried.
- Inputs that contradict the plan are not retried: no manifest of a stage the stage reads among
  its inputs, a file that is not the one its manifest recorded, a ``provisioned.json`` that is
  not a provision record, an ``images.json`` that does not name a measured image's key and
  entry digest, data that cannot be mounted, a copy of a role's data the plan did not declare
  or that takes more of the node's disk than declared (``data_copies``: an entry published again
  since the plan), checked before it is copied, a score stage's record of other servings than it
  was planned to score, an aggregate whose score stages are not every score stage of its plan,
  a capture's engine cache record that is not one, and an engine cache entry its owner left under
  another key than the stage's.

At plan time each kind's ``check`` holds the stage's shape to its role's: the role's shape with
``run.runner_memory_gib`` more memory for the runner and the disk of what the stage keeps on the
node, the copies of its role's data among it (``data_gib`` of the copies the planner declares
from the provision record the cache holds, ``hls.provision.mounts.copied_entries``), in all but
time; the aggregate's stage asks for no GPU."""

from __future__ import annotations

import math
import re
import string
from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar, Self

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError, model_validator

from hls.calibrate import POOL_ENGINE_NAMES, serving_seed_of
from hls.egress.policy import SealPolicy
from hls.measured import write_measured
from hls.pipeline.grown import declare_grower
from hls.pipeline.plan import attempt_hardware
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.provision.model import Provisioned
from hls.provision.mounts import DataCopies, data_mounts
from hls.settings import StageName
from hls.spec.resources import HardwareRecord, ResourceShape
from hls.stages import engine_cache
from hls.stages.deploy_drive import AIR_GAP, CHECKOUT, OUT, OUTPUTS_REPORT, TASK_REF
from hls.stages.engine_cache import ENGINE_CACHE, ENGINE_CACHE_GIB, engine_build_env, engine_cache_key
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.provision import role_shape
from hls.stages.inputs import input_manifest, output, read_output
from hls.stages.score import host_role_env, refusal
from hls.stages.verify import IMAGES_FILE, PROVISIONED_FILE, image_entry, image_on_node
from hls.store.engine_cache import ENGINE_CACHE_FILE, EngineCacheRecord
from hls.store.manifest import Manifest, read_manifest
from hls.substrate.container import ContainerSpec, Mount, PinnedImage, SealPlan
from hls.substrate.faults import RoleOutcome
from hls.taskref import TaskRef
from hls.verify.calibration import CALIBRATION_FILE
from hls.util import validation_errors
from hls.verify.calibration_roles import (AGGREGATE_FILE, ENGINE_FAULT_EXIT, ENGINE_FAULT_PREFIX, INSTRUMENT_FILE,
                                          REFUSED_EXIT, REFUSED_PREFIX, SCORED_FILE, CampaignSettings, ScoredServings,
                                          capture_file, replies_file)

#: The role programs, run with ``python -m``.
ROLE_MODULE = "hls.verify.calibration_roles"
#: The grower a calibration's plan names (``hls.pipeline.calibration_grower``), which adds the
#: stages this module names after the first plan's (``GROWN_PATTERNS``, ``AGGREGATE``).
GROWER = "calibration"
#: The stage that checks the reference first, in the calibration's first allocation.
INSTRUMENT = "instrument"
#: The stage that assembles the calibration, which the grower adds once no engine owes a serving.
AGGREGATE = "aggregate"
# How a capture stage and a score stage are named (``capture_stage``, ``score_stage``).
_CAPTURE, _SCORE = "capture_{engine}_{draw}", "score_{share}"
#: Where every tenant's rows mount in a role's container (``hls.provision.rows.tenant_mount_path``).
ROWS_ROOT = "/data"
#: Where a container role reads the campaign's settings.
CAMPAIGN = "/hls/campaign.json"
#: A role's node-local view directory, where it lays out the capture store it works on.
VIEW = "/hls/view"
#: Where the score role reads its share's captures, one file each.
CAPTURES = "/hls/captures"
#: Node-local disk, in GiB, beside the engine cache, that a capture stage keeps for its capture and
#: a score stage for the scored replies before the role copies them out: twice the largest a
#: serving's replies come to. The sampling gate's captures keep every position's report: a
#: deepresearch drive carries up to 1.66 GB of logprob replies, which its capture holds repacked and
#: gzipped. The token-level gate's replies are far smaller; its capture role's directory is almost
#: all the engine cache's build files (``ENGINE_CACHE_GIB``).
SCRATCH_GIB = math.ceil(2 * 1.66e9 / 2**30)


def member_seal(campaign: CampaignSettings) -> SealPlan | None:
    """How *campaign*'s pool members are sealed, in their warm stages and their captures alike, so
    nothing one fetches lands in the engine cache its captures replay: under the air gap
    (``AIR_GAP``), as a graded artifact is served, unless the task's tools call the network
    themselves (``CampaignSettings.member_internet``), when None runs it on the step's own network."""
    return None if campaign.member_internet else SealPlan(policy=SealPolicy(network=AIR_GAP), index=False)


def warm_stage(engine: str) -> str:
    """The name of the stage that builds and publishes pool member *engine*'s engine cache."""
    return f"warm_{engine}"


def capture_stage(engine: str, draw: int) -> str:
    """The name of the stage that captures serving *draw* of *engine*."""
    return _CAPTURE.format(engine=engine, draw=draw)


def score_stage(share: int) -> str:
    """The name of the score stage of share *share*, from 1."""
    return _SCORE.format(share=share)


def _pattern(template: str, **fields: str) -> str:
    """The regular expression that matches, whole, every name *template* makes, each of its
    fields matching the expression *fields* gives it."""
    return "".join(re.escape(text) + (f"({fields[field]})" if field is not None else "")
                   for text, field, _, _ in string.Formatter().parse(template))


#: The numbered names the grower adds: each top-up serving's capture by a pool member, draws
#: counted from 0, and the score of its share, shares counted from 1.
GROWN_PATTERNS = (
    _pattern(_CAPTURE, engine="|".join(map(re.escape, POOL_ENGINE_NAMES)), draw="0|[1-9][0-9]*"),
    _pattern(_SCORE, share="[1-9][0-9]*"),
)
declare_grower(GROWER, names=(AGGREGATE,), patterns=GROWN_PATTERNS)


def stage_shape(role: ResourceShape, *, runner_memory_gib: int, disk_gib: int, time_sec: int) -> ResourceShape:
    """The shape of a stage that runs one role of shape *role*: the role's, with
    *runner_memory_gib* more memory for the runner, *disk_gib* of node-local disk and *time_sec*."""
    return role.model_copy(update={"memory_gib": role.memory_gib + runner_memory_gib, "disk_gib": disk_gib,
                                   "time_sec": time_sec})


def _check_shape(kind: str, shape: ResourceShape, role: ResourceShape, *, runner_memory_gib: int,
                 disk_gib: int, data_gib: int) -> None:
    """Refuse a stage of kind *kind* whose *shape* is not ``stage_shape`` of its *role*, its disk
    *disk_gib* and *data_gib* for the copies of its role's data, in all but time: a site may give
    the stage another time, which the role then runs for."""
    want = stage_shape(role, runner_memory_gib=runner_memory_gib, disk_gib=disk_gib + data_gib,
                       time_sec=shape.time_sec)
    if shape != want:
        raise ValueError(f"the {kind} stage runs one role, so its resources must be the role's with "
                         f"{runner_memory_gib} GiB more memory for the runner and {disk_gib} GiB of disk beside "
                         f"{data_gib} GiB for the copies of its data, {want}, in all but time; they are {shape}")


def _gpu_kind(shape: ResourceShape) -> str:
    """The GPU kind of *shape*, which every role that runs an engine has."""
    if shape.gpu_kind is None:
        raise RuntimeError("a calibration role runs its engine on no GPU kind")
    return shape.gpu_kind


def _key(image_sha256: str, shape: ResourceShape, config: str, seal: SealPlan | None) -> str:
    """The engine cache key of a role of *shape* running an engine of *config* on the image whose
    file has the sha256 *image_sha256*, on the network *seal* gives it (``engine_cache_key``)."""
    return engine_cache_key(image_sha256, _gpu_kind(shape), config, seal=seal)


class InstrumentParams(DataCopies):
    """The instrument check of a calibration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, as the role rebuilds it.
    task: TaskRef
    campaign: CampaignSettings
    #: The stage of this instance whose ``images.json`` names the reference image.
    images_stage: StageName = "images"
    #: The stage of this instance whose ``provisioned.json`` names the weights and tool bundles.
    provision_stage: StageName = "provision"


def warm_load_sec(params: WarmParams, time_sec: int) -> int:
    """The seconds a warm stage of *params* given *time_sec* gives its member's load: its time less
    its copies' and its margin, so a stage given more time gives its load as much more."""
    return time_sec - params.staging_sec - params.margin_sec


class WarmParams(DataCopies):
    """One pool member's engine cache."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, as the role rebuilds it.
    task: TaskRef
    campaign: CampaignSettings
    #: The pool member whose cache it builds.
    engine: str
    #: Seconds of the stage's time allowed for its image's extraction and its data's copies before
    #: the role starts, a fixed allowance (``hls.pipeline.pipelines.STAGING_SEC``).
    staging_sec: int = Field(ge=0)
    #: Seconds of the stage's time kept after the role, in which the stage publishes the cache.
    margin_sec: int = Field(ge=0)
    #: The stage of this instance whose ``images.json`` names the member's image.
    images_stage: StageName = "images"
    #: The stage of this instance whose ``provisioned.json`` names the weights and tool bundles.
    provision_stage: StageName = "provision"

    @model_validator(mode="after")
    def _member(self) -> Self:
        if self.engine not in self.campaign.engines:
            raise ValueError(f"engine {self.engine} is not one of the campaign's pool members {self.campaign.engines}")
        return self


class CaptureParams(DataCopies):
    """One serving of one pool member."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, as the role rebuilds it.
    task: TaskRef
    campaign: CampaignSettings
    #: The pool member that serves.
    engine: str
    #: The serving's index.
    draw: int = Field(ge=0)
    #: The seed its clock is drawn from: ``serving_seed_of(campaign.task_seed, draw)``.
    seed: int
    #: The member's warm stage of this instance, the owner of its engine cache.
    warm_stage: StageName
    #: The stage of this instance whose ``images.json`` names the member's image.
    images_stage: StageName = "images"
    #: The stage of this instance whose ``provisioned.json`` names the weights, rows and tool bundles.
    provision_stage: StageName = "provision"

    @model_validator(mode="after")
    def _serving(self) -> Self:
        if self.engine not in self.campaign.engines:
            raise ValueError(f"engine {self.engine} is not one of the campaign's pool members {self.campaign.engines}")
        want = serving_seed_of(self.campaign.task_seed, self.draw)
        if self.seed != want:
            raise ValueError(f"serving {self.draw} is clocked with seed {want}, not {self.seed}")
        return self


class ScoreShareParams(DataCopies):
    """One share of servings, scored on one load of the pinned reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, as the role rebuilds it.
    task: TaskRef
    campaign: CampaignSettings
    #: The share's number, from 1, which names the stage (``score_stage``).
    share: int = Field(ge=1)
    #: The servings it scores, as (engine, draw), each captured by ``capture_stage(engine, draw)``.
    servings: list[tuple[str, int]] = Field(min_length=1)
    #: The instrument stage of this instance, the owner of the reference's engine cache.
    instrument_stage: StageName = "instrument"
    #: The stage of this instance whose ``images.json`` names the reference image.
    images_stage: StageName = "images"
    #: The stage of this instance whose ``provisioned.json`` names the weights, rows and tool bundles.
    provision_stage: StageName = "provision"

    @model_validator(mode="after")
    def _servings(self) -> Self:
        if len(set(self.servings)) != len(self.servings):
            raise ValueError(f"servings {self.servings} name a serving twice")
        strangers = sorted({e for e, _ in self.servings} - set(self.campaign.engines))
        if strangers:
            raise ValueError(f"servings of {strangers}, which are not the campaign's pool members")
        if any(draw < 0 for _, draw in self.servings):
            raise ValueError(f"servings {self.servings} hold a negative index")
        return self


class AggregateParams(BaseModel):
    """The calibration's assembly."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, as the role rebuilds it.
    task: TaskRef
    campaign: CampaignSettings
    #: The instrument stage of this instance.
    instrument_stage: StageName = "instrument"
    #: Every score stage of this instance, whose servings the calibration is assembled from.
    score_stages: list[StageName] = Field(min_length=1)


class _Attempt:
    """One attempt of a calibration stage: what every kind's run shares."""

    #: The role the stage starts, which names its directory, its logs and its image record.
    role: ClassVar[str]

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None:
            raise RuntimeError(f"the calibration {self.role} stage needs node-local scratch")
        self.ctx = ctx

    def settings_files(self, task: TaskRef, campaign: CampaignSettings) -> tuple[Path, Path]:
        """The task reference and the campaign's settings, written in the role's directory."""
        d = self.ctx.role_dir(self.role)
        task_ref, settings = d / "task-ref.json", d / "campaign.json"
        task_ref.write_text(task.model_dump_json())
        settings.write_text(campaign.model_dump_json())
        return task_ref, settings

    def raise_fault(self, outcome: RoleOutcome) -> None:
        """Raise the fault the role's ending *outcome* is, if any: its ``stage_fault``, given the
        role's reason and made not retryable when the role refused (exit 3 on its own with no OOM
        kill), and given its reason and retried when its engine host gave up on an engine (exit 4
        on its own with no OOM kill)."""
        fault = stage_fault([outcome])
        if fault is None:
            return
        if outcome.ending == "exited" and outcome.oom_kills == 0 and outcome.exit_code == REFUSED_EXIT:
            reason = refusal(self.ctx.attempt_dir / "logs" / f"{self.role}.err", REFUSED_PREFIX)
            fault = fault.model_copy(update={
                "reason": f"calibration {self.role} refused: {self.ctx.scrubber.scrub(reason)}", "retryable": False})
        elif outcome.ending == "exited" and outcome.oom_kills == 0 and outcome.exit_code == ENGINE_FAULT_EXIT:
            reason = refusal(self.ctx.attempt_dir / "logs" / f"{self.role}.err", ENGINE_FAULT_PREFIX)
            fault = fault.model_copy(update={
                "reason": f"calibration {self.role}: {self.ctx.scrubber.scrub(reason)}", "retryable": True})
        raise StageFault(fault)

    def reported(self, name: str) -> str:
        """The key of output *name*, a path under the attempt's outputs directory, recorded by the
        role's report of it. A role that exited 0 without writing it is retried."""
        if not (self.ctx.outputs_dir / name).is_file():
            raise StageFault.infra(f"the {self.role} role exited 0 without writing {name}", retryable=True)
        return self.ctx.reported_output(name, self.role)


class _ContainerAttempt(_Attempt):
    """An attempt whose role runs in a harness container on a measured image."""

    def __init__(self, ctx: StageContext) -> None:
        super().__init__(ctx)
        if ctx.runtime is None or ctx.cache is None:
            raise RuntimeError(f"the calibration {self.role} stage needs a container runtime and the cache")

    def provisioned(self, stage: str) -> Provisioned:
        """This instance's provision record, from *stage*, checked against its manifest."""
        ref = output(input_manifest(self.ctx, stage), PROVISIONED_FILE)
        try:
            return Provisioned.model_validate_json(read_output(self.ctx, ref))
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a provision record: {validation_errors(e)}",
                                   retryable=False) from e

    def image(self, stage: str, name: str) -> PinnedImage:
        """Image *name*, which *stage*'s ``images.json`` records, extracted and pinned on the node."""
        images = output(input_manifest(self.ctx, stage), IMAGES_FILE)
        return image_on_node(self.ctx, read_output(self.ctx, images), name, images_key=images.key)

    def image_sha256(self, stage: str, name: str) -> str:
        """The sha256 of image *name*'s file, which *stage*'s ``images.json`` records the entry of,
        as its entry records it: read off the entry's manifest, with nothing extracted."""
        images = output(input_manifest(self.ctx, stage), IMAGES_FILE)
        key, expect = image_entry(read_output(self.ctx, images), name, images_key=images.key)
        assert self.ctx.images is not None
        return self.ctx.entry_file("images", key, expect, self.ctx.images.image_file)[1]

    def data(self, unit_role: str, provisioned: Provisioned) -> tuple[list[Mount], dict[str, str]]:
        """The mounts and variables of the mount wall's role *unit_role*, its entries copied to
        the node once each is known to be among the copies the plan declared (``data_copies``;
        ``data_mounts``)."""
        try:
            return data_mounts(unit_role, provisioned, self.ctx.materialize, declared=self.ctx.params.data_copies)
        except ValueError as e:
            raise StageFault.infra(f"the {self.role} role's data cannot be mounted: {e}", retryable=False) from e

    def start(self, image: PinnedImage, args: Sequence[str], shape: ResourceShape, *, task: TaskRef,
              campaign: CampaignSettings, data: tuple[list[Mount], dict[str, str]], mounts: Sequence[Mount] = (),
              seal: SealPlan | None = None) -> None:
        """Run the role on *image* with *args* after its common ones, in *shape* for the stage's
        own time, with *data* and *mounts* beside what every container role gets, the build
        environment its engine cache key digests among it (``engine_build_env``), sealed by *seal*
        (None: on the step's own network); wait for it and raise the fault it ended with, if any."""
        ctx = self.ctx
        task_ref, settings = self.settings_files(task, campaign)
        workdir = ctx.role_dir(self.role) / "wd"
        workdir.mkdir()
        # The role writes its report in place, so the file it is bound to is there first.
        report = ctx.outputs_report(self.role)
        report.touch()
        out = ctx.outputs_dir
        out.mkdir(parents=True, exist_ok=True)
        wall, wall_env = data
        role_env = {**wall_env, "PYTHONPATH": f"{CHECKOUT}/src:{CHECKOUT}", **engine_build_env(_gpu_kind(shape))}
        all_mounts = (*wall, Mount(ctx.plan.code.checkout, CHECKOUT), Mount(task_ref, TASK_REF),
                      Mount(settings, CAMPAIGN), Mount(out, OUT, read_only=False),
                      Mount(report, OUTPUTS_REPORT, read_only=False), *mounts)
        command = ("python", "-m", ROLE_MODULE, self.role, "--task-ref", TASK_REF, "--campaign", CAMPAIGN, *args,
                   "--out", OUT, "--outputs-out", OUTPUTS_REPORT)
        spec = ContainerSpec(image=image, command=command, workdir=workdir, mounts=all_mounts, gpus=shape.gpus > 0,
                             seal=seal)
        ctx.start_container(self.role, spec, shape.model_copy(update={"time_sec": ctx.stage.shape.time_sec}),
                            owner="harness", env=role_env)
        [outcome] = ctx.wait_roles(self.role)
        self.raise_fault(outcome)


class _Instrument(_ContainerAttempt):
    role = "instrument"

    def run(self) -> dict[str, str]:
        p: InstrumentParams = self.ctx.params
        c = p.campaign
        # Every extraction and copy runs alone, so all of them come before the role.
        image = self.image(p.images_stage, c.reference_image)
        data = self.data("reference", self.provisioned(p.provision_stage))
        shape = c.instrument_shape()
        key = _key(image.file_sha256, shape, c.reference_engine_config, None)
        cache, seeded = engine_cache.prepare(self.ctx, self.role, key, pinned=None, owner=True)
        self.start(image, ("--gpus", str(shape.gpus)), shape, task=p.task, campaign=c, data=data,
                   mounts=(Mount(cache, ENGINE_CACHE, read_only=False),))
        outputs = {"instrument": self.reported(INSTRUMENT_FILE)}
        return {**outputs, "engine_cache": engine_cache.owned_record(self.ctx, cache, key, seeded,
                                                                     must_publish=False)}


class _Warm(_ContainerAttempt):
    role = "warm"

    def run(self) -> dict[str, str]:
        p: WarmParams = self.ctx.params
        c = p.campaign
        name = c.member_images[p.engine]
        key = _key(self.image_sha256(p.images_stage, name), c.member_shape, c.member_engine_configs[p.engine],
                   member_seal(c))
        found = engine_cache.published(self.ctx, key)
        if found is not None:
            # Built by an earlier calibration of the engine: nothing to build, copy or extract.
            return {"engine_cache": engine_cache.found_record(self.ctx, key, found)}
        image = self.image(p.images_stage, name)
        data = self.data("reference", self.provisioned(p.provision_stage))
        cache, seeded = engine_cache.prepare(self.ctx, self.role, key, pinned=None, owner=True)
        if seeded is None:
            args = ("--engine", p.engine, "--load-sec", str(warm_load_sec(p, self.ctx.stage.shape.time_sec)))
            self.start(image, args, c.member_shape, task=p.task, campaign=c, data=data,
                       mounts=(Mount(cache, ENGINE_CACHE, read_only=False),), seal=member_seal(c))
        return {"engine_cache": engine_cache.owned_record(self.ctx, cache, key, seeded, must_publish=True)}


class _Capture(_ContainerAttempt):
    role = "capture"

    def run(self) -> dict[str, str]:
        p: CaptureParams = self.ctx.params
        c = p.campaign
        image = self.image(p.images_stage, c.member_images[p.engine])
        data = self.data("calibration", self.provisioned(p.provision_stage))
        key = _key(image.file_sha256, c.member_shape, c.member_engine_configs[p.engine], member_seal(c))
        cache, seeded = engine_cache.prepare(self.ctx, self.role, key,
                                             pinned=engine_cache.owner_entry(self.ctx, p.warm_stage, key))
        # Made afresh for each attempt.
        view = self.ctx.role_dir(self.role) / "view"
        view.mkdir()
        args = ("--rows", ROWS_ROOT, "--engine", p.engine, "--draw", str(p.draw), "--seed", str(p.seed), "--view", VIEW)
        self.start(image, args, c.member_shape, task=p.task, campaign=c, data=data,
                   mounts=(Mount(view, VIEW, read_only=False), Mount(cache, ENGINE_CACHE, read_only=False)),
                   seal=member_seal(c))
        outputs = {"capture": self.reported(capture_file(c.workload_fingerprint, p.engine, p.draw))}
        record = EngineCacheRecord(key=key, seeded_from=seeded)
        return {**outputs, "engine_cache": engine_cache.write_record(self.ctx, record)}


class _Score(_ContainerAttempt):
    role = "score"

    def run(self) -> dict[str, str]:
        p: ScoreShareParams = self.ctx.params
        c = p.campaign
        fingerprint = c.workload_fingerprint
        captures = [(engine, draw, output(input_manifest(self.ctx, capture_stage(engine, draw)),
                                          capture_file(fingerprint, engine, draw)))
                    for engine, draw in p.servings]
        image = self.image(p.images_stage, c.reference_image)
        data = self.data("calibration", self.provisioned(p.provision_stage))
        key = _key(image.file_sha256, c.reference_shape, c.reference_engine_config, None)
        cache, seeded = engine_cache.prepare(self.ctx, self.role, key,
                                             pinned=engine_cache.owner_entry(self.ctx, p.instrument_stage, key))
        view = self.ctx.role_dir(self.role) / "view"
        view.mkdir()
        mounts = [Mount(view, VIEW, read_only=False), Mount(cache, ENGINE_CACHE, read_only=False)]
        args = ["--rows", ROWS_ROOT, "--gpus", str(c.reference_shape.gpus), "--share", str(p.share)]
        for engine, draw, ref in captures:
            target = f"{CAPTURES}/{engine}-{draw}.capture.jsonl.gz"
            mounts.append(Mount(self.ctx.store.path(ref.key), target))
            args += ["--capture", engine, str(draw), ref.sha256, target]
        self.start(image, [*args, "--view", VIEW], c.reference_shape, task=p.task, campaign=c, data=data,
                   mounts=mounts)
        outputs = {"scored": self.reported(SCORED_FILE)}
        for engine, draw in p.servings:
            outputs[f"replies_{engine}_{draw}"] = self.reported(replies_file(fingerprint, engine, draw))
        record = EngineCacheRecord(key=key, seeded_from=seeded)
        return {**outputs, "engine_cache": engine_cache.write_record(self.ctx, record)}


#: The aggregate role's ``--member-caches`` file: each serving's engine cache record, by engine and
#: serving index.
_MEMBER_CACHES = TypeAdapter(dict[str, dict[int, EngineCacheRecord]])
#: The aggregate role's ``--hardware`` file: the hardware record of each serving's allocation.
_HARDWARE = TypeAdapter(list[HardwareRecord])


class _Aggregate(_Attempt):
    role = "aggregate"

    def scored(self, stage: str) -> tuple[Manifest, ScoredServings]:
        """Score stage *stage*'s manifest and its record, checked against the manifest and
        against the servings the stage was planned to score."""
        manifest = input_manifest(self.ctx, stage)
        ref = output(manifest, SCORED_FILE)
        try:
            record = ScoredServings.model_validate_json(read_output(self.ctx, ref))
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a scored record: {validation_errors(e)}", retryable=False) from e
        planned = sorted(tuple(s) for s in ScoreShareParams.model_validate(self.ctx.plan.stage(stage).params).servings)
        got = sorted((s.engine, s.draw) for s in record.servings)
        if got != planned:
            raise StageFault.infra(f"{stage} scored {got}; its plan says {planned}", retryable=False)
        return manifest, record

    def hardware(self, servings: list[tuple[str, int]]) -> list[HardwareRecord] | None:
        """The hardware record of the allocation each serving's capture stage published from
        (``attempt_hardware``); None when one of them records none, so the calibration claims no
        hardware it was not measured on."""
        records = []
        for engine, draw in servings:
            stage = capture_stage(engine, draw)
            found = read_manifest(self.ctx.store, self.ctx.plan.layout, stage)
            record = attempt_hardware(self.ctx.store, self.ctx.plan.layout, stage, found[0].attempt) if found else None
            if record is None:
                return None
            records.append(record)
        return records

    def member_cache(self, engine: str, draw: int) -> EngineCacheRecord:
        """The engine cache record of serving *draw* of *engine*'s capture stage, checked against its
        manifest."""
        ref = output(input_manifest(self.ctx, capture_stage(engine, draw)), ENGINE_CACHE_FILE)
        try:
            return EngineCacheRecord.model_validate_json(read_output(self.ctx, ref))
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not an engine cache record: {validation_errors(e)}",
                                   retryable=False) from e

    def run(self) -> dict[str, str]:
        ctx = self.ctx
        p: AggregateParams = ctx.params
        fingerprint = p.campaign.workload_fingerprint
        planned = sorted(s.name for s in ctx.plan.stages if s.kind == ScoreShareStage.kind)
        if sorted(p.score_stages) != planned:
            raise StageFault.infra(f"the aggregate reads score stages {sorted(p.score_stages)}, not every score stage "
                                   f"of its plan, {planned}", retryable=False)
        instrument = output(input_manifest(ctx, p.instrument_stage), INSTRUMENT_FILE)
        args = ["--instrument", instrument.sha256, str(ctx.store.path(instrument.key))]
        caches: dict[str, dict[int, EngineCacheRecord]] = {}
        for stage in p.score_stages:
            manifest, record = self.scored(stage)
            for s in record.servings:
                ref = output(manifest, replies_file(fingerprint, s.engine, s.draw))
                args += ["--serving", s.engine, str(s.draw), str(s.n_tokens), ref.sha256, str(ctx.store.path(ref.key))]
                caches.setdefault(s.engine, {})[s.draw] = self.member_cache(s.engine, s.draw)
        task_ref, settings = self.settings_files(p.task, p.campaign)
        member_caches = ctx.role_dir(self.role) / "member-caches.json"
        member_caches.write_text(_MEMBER_CACHES.dump_json(caches, indent=2).decode())
        args += ["--member-caches", str(member_caches)]
        hardware = self.hardware(sorted((e, d) for e, by_draw in caches.items() for d in by_draw))
        if hardware is not None:
            recorded = ctx.role_dir(self.role) / "hardware.json"
            recorded.write_bytes(_HARDWARE.dump_json(hardware, indent=2))
            args += ["--hardware", str(recorded)]
        view = ctx.role_dir(self.role) / "view"
        view.mkdir()
        out = ctx.outputs_dir
        out.mkdir(parents=True, exist_ok=True)
        argv = [str(ctx.plan.toolchain.python), "-m", ROLE_MODULE, self.role, "--task-ref", str(task_ref),
                "--campaign", str(settings), *args, "--view", str(view), "--out", str(out),
                "--outputs-out", str(ctx.outputs_report(self.role))]
        shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
        ctx.start_role(self.role, argv, shape, env=host_role_env(ctx), owner="harness")
        [outcome] = ctx.wait_roles(self.role)
        self.raise_fault(outcome)
        outputs = {"calibration": self.reported(CALIBRATION_FILE), "aggregate": self.reported(AGGREGATE_FILE)}
        # The task's measured file is the last thing written: what runs of the task use from now on.
        key = write_measured(ctx.store, ctx.plan.task, CALIBRATION_FILE, (out / CALIBRATION_FILE).read_bytes())
        ctx.events.emit("calibration_measured", measured_key=key)
        return outputs


@register
class InstrumentStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "calibrate_instrument"
    params_model: ClassVar[type[BaseModel]] = InstrumentParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the reference image's extraction, a copy of each
        cache entry the reference mounts (the weights and the full tool bundle;
        ``hls.provision.mounts`` role ``reference``), the instrument role, and either the copy
        that seeds its engine cache or the copy that publishes the cache it started cold with."""
        return 5

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, InstrumentParams)
        _check_shape(cls.kind, shape, params.campaign.instrument_shape(), runner_memory_gib=runner_memory_gib,
                     disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB, data_gib=params.data_gib)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Instrument(ctx).run()


@register
class WarmStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "calibrate_warm"
    params_model: ClassVar[type[BaseModel]] = WarmParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the member image's extraction, a copy of each cache
        entry the role mounts (the weights and the full tool bundle; ``hls.provision.mounts`` role
        ``reference``), the warm role and the copy that publishes its cache."""
        return 5

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, WarmParams)
        _check_shape(cls.kind, shape, params.campaign.member_shape, runner_memory_gib=runner_memory_gib,
                     disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB, data_gib=params.data_gib)
        if warm_load_sec(params, shape.time_sec) <= 0:
            raise ValueError(f"the {cls.kind} stage's {shape.time_sec}s leave its member's load no time after its "
                             f"{params.staging_sec}s of copies and its {params.margin_sec}s margin")

    @classmethod
    def seals_containers(cls, params: BaseModel | None) -> bool:
        # A member is sealed unless its campaign gives it the internet (``member_seal``).
        assert isinstance(params, WarmParams)
        return not params.campaign.member_internet

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Warm(ctx).run()


@register
class CaptureStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "calibrate_capture"
    params_model: ClassVar[type[BaseModel]] = CaptureParams
    # Its capture is compressed, but it is a pool member's serving, which no contestant chose: no
    # file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the member image's extraction, a copy of each
        cache entry the role mounts (the weights, the test rows and the full tool bundle;
        ``hls.provision.mounts`` role ``calibration``), the capture role, and either the copy
        that seeds its engine cache or the copy that publishes the cache it started cold with."""
        return 6

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, CaptureParams)
        _check_shape(cls.kind, shape, params.campaign.member_shape, runner_memory_gib=runner_memory_gib,
                     disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB, data_gib=params.data_gib)

    @classmethod
    def seals_containers(cls, params: BaseModel | None) -> bool:
        # A member is sealed unless its campaign gives it the internet (``member_seal``).
        assert isinstance(params, CaptureParams)
        return not params.campaign.member_internet

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Capture(ctx).run()


@register
class ScoreShareStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "calibrate_score"
    params_model: ClassVar[type[BaseModel]] = ScoreShareParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the reference image's extraction, a copy of each
        cache entry the role mounts (the weights, the test rows and the full tool bundle;
        ``hls.provision.mounts`` role ``calibration``), the score role, and either the copy that
        seeds its engine cache or the copy that publishes the cache it started cold with. The
        captures are mounted where their stages published them, never copied."""
        return 6

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, ScoreShareParams)
        _check_shape(cls.kind, shape, params.campaign.reference_shape, runner_memory_gib=runner_memory_gib,
                     disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB, data_gib=params.data_gib)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Score(ctx).run()


@register
class AggregateStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "calibrate_aggregate"
    params_model: ClassVar[type[BaseModel]] = AggregateParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the aggregate role."""
        return 1

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, AggregateParams)
        if shape.gpus:
            raise ValueError(f"the {cls.kind} stage reads records and loads no engine; its shape asks for "
                             f"{shape.gpus} GPUs")
        role_shape(shape, runner_memory_gib)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Aggregate(ctx).run()
