"""The role programs of a calibration (``hls calibrate``), each started by its
stage (``hls.stages.calibration``):

    python -m hls.verify.calibration_roles instrument --task-ref FILE --campaign FILE --gpus N
        --out DIR --outputs-out REPORT
    python -m hls.verify.calibration_roles warm --task-ref FILE --campaign FILE --engine NAME
        --load-sec SECONDS --out DIR --outputs-out REPORT
    python -m hls.verify.calibration_roles capture --task-ref FILE --campaign FILE --rows DIR
        --engine NAME --draw N --seed S --view DIR --out DIR --outputs-out REPORT
    python -m hls.verify.calibration_roles score --task-ref FILE --campaign FILE --rows DIR --gpus N
        --share J --capture ENGINE DRAW SHA256 FILE [--capture ...] --view DIR --out DIR
        --outputs-out REPORT
    python -m hls.verify.calibration_roles aggregate --task-ref FILE --campaign FILE
        --instrument SHA256 FILE --serving ENGINE DRAW TOKENS SHA256 FILE [--serving ...]
        --member-caches FILE --view DIR --out DIR --outputs-out REPORT

Each role is one phase of a calibration, done by the measurement functions of ``hls.calibrate``:
``instrument`` brings the pinned reference up and checks it (``check_instrument``, with the
fp32 anchor beside it, for the token-level gate; ``check_stack`` for the sampling gate) and
writes ``instrument.json``; ``warm`` loads one pool member once and releases it, so its load
builds the kernels every capture of it replays into the engine cache its stage mounts, and
writes nothing else; ``capture`` drives one serving of one pool member
(``drive_serving``) and writes its capture; ``score`` teacher-forces a share of captures
through one load of the pinned reference (``score_share``) and writes each serving's scored
replies and ``scored.json``; ``aggregate`` assembles the ``calibration.json`` from the
instrument's record and every scored serving (``assemble`` or ``assemble_report``), and records
in ``aggregate.json`` the engine cache each serving's member ran with, as its capture stage
recorded it (``--member-caches``, a JSON object of ``EngineCacheRecord`` by engine and serving). The
engines load in the role's own process, on one engine host the role creates
(``hls.verify.inprocess.EngineHost``): a member on its measured image, which is the capture role's
image, over every GPU the role sees, and the pinned reference and its anchor on the reference's,
each on a GPU of its own. Each engine's load is given a bound: in the warm role,
which builds its engine's cache cold, ``--load-sec``, which its stage derives from its own time;
elsewhere, where the cache is seeded or the engine builds no kernels, the campaign's
``load_timeout_sec``. The host bounds each step of a release too, and watches a pool member's
processes. A load or a release step past its bound, and a member whose process ended or that
reports itself dead, end the role at once (``GIVE_UP``), since the thread a load or a release
runs in cannot be stopped.

A role's outputs stand only once every engine it loaded was released: a member whose release
fails ends the warm or capture role as a failure, and the host ends a role whose release does not
return, so a stage publishes only what a role built whose engines all let go of their GPUs.

Those functions read and write a capture store (``CaptureStore``), laid out under one root by
workload fingerprint, engine and serving. A role keeps one in its node-local view directory:
each capture or scored serving it reads is linked there from the file its stage names, after
the file is checked against the sha256 the stage recorded for it, and each file it writes there
is copied into its outputs directory under the same layout, below ``capture/`` or ``replies/``.
So the outputs hold only what the role wrote, whole, and the view never reaches the store.

The role writes its files under ``DIR`` and reports them, keyed by their paths under it, in
``REPORT`` for the stage's manifest (``hls.store.manifest.report_outputs``). The exit status
says whether another attempt can help:
- 0: every file written and reported;
- 3 (``REFUSED_EXIT``): a deterministic refusal, one stderr line starting with
  ``REFUSED_PREFIX``: ``hls.calibrate``'s own (``CalibrationError``: a reference on silicon other
  than the task's, a failed preflight, a share the measurement refused, a serving that
  measured no token, a campaign that fails its sizing or precision checks), or inputs that
  contradict the plan (``InputMismatch``: a file that is not the one its stage recorded, a task
  whose workload fingerprint is not the plan's, a serving the campaign does not hold, member
  caches of other servings than the aggregate is handed);
- 4 (``ENGINE_FAULT_EXIT``): an engine the engine host gave up on, one stderr line starting with
  ``ENGINE_FAULT_PREFIX``, after every thread's stack when a step hung: ``EngineLoadTimeout`` (it
  did not load within its bound), ``EngineReleaseTimeout`` (a step of its release did not return
  within the host's bound) or ``EngineDied`` (a process of it ended, or it reports itself dead),
  which another attempt may pass, as a load slowed by what else shares the node, or a process lost
  to the node;
- 1: anything else, with the traceback on stderr, among it a serving whose drive captured
  nothing and a share whose scoring failed otherwise (``ServingFailed``), which another
  attempt may complete."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import shutil
import sys
import traceback
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, ValidationInfo,
                      field_validator, model_validator)

import hls.calibrate as calibrate
from hls.calibrate import MIN_SERVINGS, CaptureStore, PoolEngine, RowIndex, Serving
from hls.core.engine import Engine
from hls.obs.logsetup import configure_logging
from hls.spec.datasets import Split
from hls.spec.resources import HardwareRecord, ResourceShape
from hls.spec.taskspec import TaskSpec
from hls.store.base import check_segment
from hls.store.digest import Sha256Hex, sha256_file
from hls.store.engine_cache import EngineCacheRecord
from hls.store.manifest import report_outputs
from hls.taskref import TaskRef, load_task
from hls.verify.calibration import (Calibration, CalibrationError, PreflightResult, ReportCalibration, StackInfo,
                                    StopReason, Tolerances, calibration_hardware, workload_fingerprint)
from hls.verify.inprocess import EngineHost, GiveUp, Release

# Named, not ``__name__``: run as a program the module is ``__main__``, which no hls logging
# handler covers.
log = logging.getLogger("hls.verify.calibration_roles")

#: The exit status of a deterministic refusal, which another attempt would repeat.
REFUSED_EXIT = 3
#: How a refusal's stderr line starts; the stage quotes the rest.
REFUSED_PREFIX = "hls-calibrate-refused: "
#: The exit status of a role whose engine host gave up on an engine.
ENGINE_FAULT_EXIT = 4
#: How the stderr line of a role whose engine host gave up on an engine starts; the stage quotes the
#: rest.
ENGINE_FAULT_PREFIX = "hls-calibrate-engine-fault: "
#: How every role's engine host gives up.
GIVE_UP = GiveUp(exit_code=ENGINE_FAULT_EXIT, prefix=ENGINE_FAULT_PREFIX)
#: The instrument role's record.
INSTRUMENT_FILE = "instrument.json"
#: The score role's record of the servings it scored.
SCORED_FILE = "scored.json"
#: The aggregate's record of what the calibration was assembled from.
AGGREGATE_FILE = "aggregate.json"

def capture_file(fingerprint: str, engine: str, draw: int) -> str:
    """Where serving *draw* of *engine*'s capture lies in a capture role's outputs: the capture
    store's own layout (``CaptureStore.capture_path``) below ``capture/``."""
    return CaptureStore(Path("capture"), fingerprint).capture_path(engine, draw).as_posix()


def replies_file(fingerprint: str, engine: str, draw: int) -> str:
    """Where serving *draw* of *engine*'s scored replies lie in a score role's outputs: the
    capture store's own layout (``CaptureStore.replies_path``) below ``replies/``."""
    return CaptureStore(Path("replies"), fingerprint).replies_path(engine, draw).as_posix()


class CampaignSettings(BaseModel):
    """What a calibration measures and how it is planned. Every stage of the instance carries
    it, and the grower plans more servings from it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The gate measured: the token-level gate's flips or the sampling gate's report divergence
    #: (``hls.verify.calibration.calibration_kind``).
    kind: Literal["flips", "report"]
    #: The task's workload fingerprint when the plan was made: it keys every capture, and each
    #: role refuses a task that has another.
    workload_fingerprint: Sha256Hex
    #: The task's seed, from which each serving's seed is derived (``serving_seed_of``).
    task_seed: int
    #: The pool members, each a serving stack with a measured image.
    engines: list[PoolEngine] = Field(min_length=1)
    #: Whether a capture role runs its member with the network open (``hls.calibrate.member_internet``):
    #: only for a task whose agent's tools call the network; otherwise it is sealed with no
    #: egress, as a graded artifact is served.
    member_internet: bool
    #: (P)'s relative precision target, read by both gates.
    eps: float = Field(gt=0)
    #: (T)'s per-token tail tolerance, which sets the token-level gate's exposure target.
    mu_per_mtok: float = Field(gt=0)
    #: The family-wise false-alarm budget of one graded trace, for the token-level gate.
    delta: float = Field(gt=0, lt=1)
    #: The expected end-of-turn count one graded trace may carry, for the sampling gate.
    eot_expected_max: float = Field(gt=0)
    #: Sessions one serving drives; None for the whole test split.
    sessions_per_draw: int | None = Field(default=None, ge=1)
    #: Servings each engine takes, for the sampling gate; None for the token-level gate, whose
    #: count follows from its exposure target.
    servings_per_engine: int | None = Field(default=None, ge=1)
    #: The most servings an engine may take; a round that would take it past them is refused.
    max_servings_per_engine: int = Field(ge=1)
    #: Allocations that take servings side by side.
    fanout: int = Field(ge=1)
    #: Each engine's measured image, by its name in ``images.json``.
    member_images: dict[str, Annotated[str, AfterValidator(check_segment)]]
    #: The measured reference image's name in ``images.json``.
    reference_image: Annotated[str, AfterValidator(check_segment)]
    #: What each pool member's compile and kernel caches depend on beside its image and GPU kind
    #: (``hls.calibrate.engine_cache_config``), by engine: with those, the key of the engine cache
    #: its capture roles share (``hls.stages.engine_cache.engine_cache_key``).
    member_engine_configs: dict[str, str]
    #: The same of the pinned reference, whose engine cache the instrument and score roles share.
    reference_engine_config: str
    #: A capture role's shape; its time is only the stage's default.
    member_shape: ResourceShape
    #: A score role's shape, the verify role's; the instrument role's follows from it
    #: (``instrument_shape``).
    reference_shape: ResourceShape
    #: Each request's timeout in a serving's drive, in seconds.
    request_timeout_sec: float = Field(gt=0)
    #: Seconds a role gives each engine to load where its engine cache is seeded, or where the
    #: engine builds no kernels (the reference); past them the role's engine host gives up.
    load_timeout_sec: float = Field(gt=0)
    #: Seconds a score stage gives each serving it scores (``hls.pipeline.calibration_grower.score_sec``):
    #: the verify time the task's evaluation allocation declares, or the planner's own figure.
    score_per_serving_sec: int = Field(gt=0)
    #: The wall one serving's drive may run to, in seconds; None for none.
    drive_timeout_sec: float | None = Field(default=None, gt=0)

    @field_validator("max_servings_per_engine")
    @classmethod
    def _holds_the_first_servings(cls, value: int, info: ValidationInfo) -> int:
        # A field's own check, so a plan-time refusal names the field (``hls.util.validation_errors``).
        first = info.data.get("servings_per_engine") or MIN_SERVINGS
        if value < first:
            raise ValueError(f"max_servings_per_engine {value} is under the {first} servings each engine takes first")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if len(set(self.engines)) != len(self.engines):
            raise ValueError(f"engines {self.engines} name a member twice")
        if sorted(self.member_images) != sorted(self.engines):
            raise ValueError(f"member_images names {sorted(self.member_images)}, not the engines {sorted(self.engines)}")
        if sorted(self.member_engine_configs) != sorted(self.engines):
            raise ValueError(f"member_engine_configs names {sorted(self.member_engine_configs)}, not the engines "
                             f"{sorted(self.engines)}")
        if (self.kind == "report") != (self.servings_per_engine is not None):
            raise ValueError("servings_per_engine is set for the sampling gate and only for it")
        if not self.member_shape.gpus or not self.reference_shape.gpus:
            raise ValueError("a pool member and the reference each run on a GPU")
        return self

    def instrument_shape(self) -> ResourceShape:
        """The instrument role's shape: the reference's, and for the token-level gate twice its
        GPUs, CPUs and memory, since the fp32 anchor is up beside the pinned reference."""
        r = self.reference_shape
        if self.kind == "report":
            return r
        return r.model_copy(update={"gpus": 2 * r.gpus, "cpus": 2 * r.cpus, "memory_gib": 2 * r.memory_gib})


class InstrumentRecord(BaseModel):
    """``instrument.json``: the pinned reference's stack and, for the token-level gate, its
    agreement with the fp32 anchor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The stack the reference reported, on the task's declared accelerator.
    stack: StackInfo
    #: The preflight against the fp32 anchor; None for the sampling gate, which runs none.
    preflight: PreflightResult | None


class ScoredServing(BaseModel):
    """One serving a score role scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The pool member that served it.
    engine: PoolEngine
    #: The serving's index.
    draw: int = Field(ge=0)
    #: Teacher-forced tokens over its replies: the exposure it adds.
    n_tokens: int = Field(ge=1)
    #: Replies it measured.
    n_replies: int = Field(ge=1)
    #: The serving package version the member reported.
    serving_version: str


class ScoredServings(BaseModel):
    """``scored.json``: what one score role scored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Each serving, in the order scored.
    servings: list[ScoredServing] = Field(min_length=1)


class AggregateRecord(BaseModel):
    """``aggregate.json``: what the calibration was assembled from."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The gate measured.
    kind: Literal["flips", "report"]
    #: The workload fingerprint the calibration is keyed by.
    workload_fingerprint: Sha256Hex
    #: Why the campaign stopped taking servings.
    stopped: StopReason
    #: Each engine's servings, by index, ascending.
    servings: dict[str, list[int]]
    #: Each engine's teacher-forced tokens over those servings.
    tokens: dict[str, int]
    #: The ceiling the calibration sets: ``q_max`` for the token-level gate, ``kappa`` for the
    #: sampling gate.
    ceiling: float
    #: The engine cache each serving's member ran with, by engine and serving index, as its
    #: capture stage recorded it: the compile products and the tuned kernel configurations it
    #: replayed, from the entry it was seeded from, or its own, built cold.
    member_caches: dict[str, dict[int, EngineCacheRecord]]
    #: What the instance that measured the calibration says when it ends: one line for each pool engine
    #: a census of the token-level gate flags (``hls.calibrate.pool_warnings``), as the role
    #: logged it. Empty for the sampling gate, which takes no census.
    warnings: list[str]


class InputMismatch(ValueError):
    """An input that is not the one its stage recorded, or not what the plan describes."""


class ServingFailed(RuntimeError):
    """A serving whose drive captured nothing or whose member did not release, or a share whose
    scoring failed: another attempt may succeed."""


def checked(path: Path, sha256: str) -> Path:
    """*path*, refused (InputMismatch) unless its bytes have *sha256*."""
    found, _size = sha256_file(path)
    if found != sha256:
        raise InputMismatch(f"{path} is not the file its stage recorded: its sha256 is {found}, not {sha256}")
    return path


def _link(where: Path, source: Path) -> None:
    """Link *source* at *where* in a role's view."""
    where.parent.mkdir(parents=True, exist_ok=True)
    where.symlink_to(source)


def _copy_out(source: Path, dest: Path) -> Path:
    """Copy *source*, a file the role wrote in its view, to *dest* in its outputs."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    return dest


def measure_for(task: TaskSpec, kind: str) -> calibrate.Measure:
    """The measurement ``hls.calibrate`` takes for the gate *kind* of *task*: the report measure for
    the sampling gate, the flip measure for the token-level gate."""
    if kind == "report":
        return calibrate.report_measure(calibrate.report_verifier(task), task.max_context_len)
    return calibrate.flip_measure()


@contextlib.asynccontextmanager
async def in_process(devices: int | None = None, *, load_sec: float) -> AsyncIterator[calibrate.Deploy]:
    """The deploy seam ``hls.calibrate`` loads engines through (``hls.calibrate.Deploy``), over one
    engine host of this process, seeing *devices* GPUs (None: torch's count): each engine loads
    within *load_sec*, an HF engine on the lowest GPU no other engine holds and a serving stack
    over every GPU the role sees, the host gives up as ``GIVE_UP`` says (``hls.verify.inprocess``),
    and every engine still loaded is released when the block ends."""
    async with EngineHost(devices=devices, load_sec=load_sec, give_up=GIVE_UP) as host:
        async def deploy(name: str, engine: Engine) -> tuple[Engine, Release]:
            log.info("loading %s as %s", type(engine).__name__, name)
            return await host.acquire(engine, name=name)

        yield deploy


async def instrument(task: TaskSpec, kind: str, deploy: calibrate.Deploy) -> InstrumentRecord:
    """The instrument check of *task*'s reference: for the token-level gate its
    stack and the fp32 anchor's preflight (``check_instrument``), for the sampling gate its
    stack alone (``check_stack``). Raise CalibrationError for a reference on silicon other
    than the task's and a failed preflight."""
    if kind == "report":
        return InstrumentRecord(stack=await calibrate.check_stack(task, deploy), preflight=None)
    stack, preflight = await calibrate.check_instrument(task, deploy)
    return InstrumentRecord(stack=stack, preflight=preflight)


async def warm(task: TaskSpec, engine: str, deploy: calibrate.Deploy) -> None:
    """Load pool member *engine* of *task* once, on its own device, and release it: the load
    builds the kernels its captures replay, into the engine cache the role mounts. Raise
    CalibrationError for a member on silicon other than the task's, whose cache would be
    published under the task's GPU kind, with a release that failed too noted on it, and the
    release's own error when only the release failed: the stage publishes the cache only from a
    role whose member let go of its GPU."""
    proxy, release = await deploy(f"{engine}-warm", calibrate.member_engines(task)[engine]())
    try:
        await calibrate.assert_declared_device(proxy, kind=task.hardware.accelerator.kind, role=f"{engine} warm")
    except BaseException as refused:
        try:
            await release()
        except Exception as e:  # noqa: BLE001 -- the refusal is the one to report
            refused.add_note(f"releasing the member failed too: {type(e).__name__}: {e}")
        raise
    await release()


async def capture(task: TaskSpec, serving: Serving, *, kind: str, sessions: list[tuple],
                  sessions_per_draw: int | None, request_timeout_sec: float,
                  drive_timeout_sec: float | None, store: CaptureStore, deploy: calibrate.Deploy) -> Path:
    """Drive *serving* of *task*'s traffic once (``drive_serving``) and return its capture in
    *store*. Raise ServingFailed, from the drive's own error, when the serving captured nothing,
    and from the release's when its member did not let go of its GPU: another attempt may
    capture it, on a member that ends cleanly."""
    result = await calibrate.drive_serving(
        serving, task=task, sessions=sessions, sessions_per_draw=sessions_per_draw,
        request_timeout_sec=request_timeout_sec, store=store, measure=measure_for(task, kind),
        drive_timeout_sec=drive_timeout_sec, deploy=deploy, builders=calibrate.member_engines(task))
    if result.failed:
        raise ServingFailed(f"serving {serving.draw} of pool member {serving.engine!r} captured nothing: "
                            f"{result.error}") from result.error
    if result.error is not None:
        raise ServingFailed(f"serving {serving.draw} of pool member {serving.engine!r} captured, but its engine did not "
                            f"release: {result.error}") from result.error
    return store.capture_path(serving.engine, serving.draw)


async def score(task: TaskSpec, servings: Sequence[Serving], *, kind: str, share: int, store: CaptureStore,
                rows: RowIndex, deploy: calibrate.Deploy) -> list[Any]:
    """Teacher-force the captures in *store* of *servings* through one load of *task*'s pinned
    reference (``score_share``) and return their tallies, each persisted in *store*. Raise
    CalibrationError for a serving that measured no token (``check_measured``) and for a
    share the measurement itself refused, which another attempt over the same captures would
    repeat; and ServingFailed,
    from the share's own error, for a share that failed otherwise."""
    measure = measure_for(task, kind)
    result = await calibrate.score_share(list(servings), share, task=task, deploy=deploy, store=store, rows=rows,
                                         measure=measure)
    if isinstance(result.error, CalibrationError):
        raise result.error
    if result.error is not None:
        raise ServingFailed(f"share {share} scored {len(result.stats)} of its {len(servings)} servings: "
                            f"{result.error}") from result.error
    calibrate.check_measured(result.stats)
    for stats in result.stats:
        log.info("serving %s/%d: %s", stats.engine, stats.draw, measure.log_line(stats))
    return list(result.stats)


def engine_config(engine: str, task: TaskSpec) -> str:
    """How a pool member was served, as one auditable line: its engine class plus the
    settings that make it a distinct noise source. Versions are not here: they belong
    to ``Calibration.stack``, which records the one software stack the whole run
    used."""
    built = calibrate.member_engines(task)[engine]()
    settings = {k: getattr(built, k) for k in ("max_model_len", "dtype")
                if getattr(built, k, None) is not None}
    # The frequency penalty moves which token the member emits, so it belongs in the
    # record of how it served -- but only when it is in force, so a plain-greedy
    # member's line names none.
    if getattr(built.decode, "penalized", False):
        settings["frequency_penalty"] = built.decode.frequency_penalty
    detail = ", ".join(f"{k}={v!r}" for k, v in sorted(settings.items()))
    return f"{type(built).__name__}({detail}), clocked serving"


def aggregate(task: TaskSpec, module: str, campaign: CampaignSettings, record: InstrumentRecord,
              store: CaptureStore, servings: Mapping[str, Mapping[int, int]]
              ) -> tuple[Calibration | ReportCalibration, list[str]]:
    """The calibration of *task*, the ``TASK`` of *module*, assembled from
    *record* and *servings*: each engine's scored servings in *store*, by index, to the tokens
    their score role recorded. ``assemble`` for the token-level gate, with what its censuses
    flag (``pool_warnings``), ``assemble_report`` for the sampling gate, with nothing flagged,
    the campaign stopped at its target. Raise CalibrationError as they do, and
    InputMismatch for a serving that measured other tokens than its score role recorded and for
    a token-level campaign whose instrument record holds no preflight."""
    measure = measure_for(task, campaign.kind)
    state = calibrate.Campaign()
    for engine in campaign.engines:
        for draw, recorded in sorted(servings.get(engine, {}).items()):
            stats = measure.read(store, engine, draw)
            if stats.n_tokens != recorded:
                raise InputMismatch(f"serving {draw} of {engine} measured {stats.n_tokens} tokens; its score role "
                                    f"recorded {recorded}")
            state.add(stats)
    if campaign.kind == "report":
        return calibrate.assemble_report(
            state, task=task, module=module, engines=list(campaign.engines), stack=record.stack, eps=campaign.eps,
            eot_expected_max=campaign.eot_expected_max, alpha=calibrate.report_verifier(task).alpha,
            n_servings=campaign.servings_per_engine, stopped=StopReason.TARGET_REACHED), []
    if record.preflight is None:
        raise InputMismatch("a token-level calibration is assembled with the instrument's preflight, and the "
                            "instrument record holds none")
    cal = calibrate.assemble(
        state, task=task, module=module, engines=list(campaign.engines),
        tol=Tolerances(eps=campaign.eps, mu_per_mtok=campaign.mu_per_mtok, delta=campaign.delta),
        stack=record.stack, preflight=record.preflight,
        engine_configs={e: engine_config(e, task) for e in campaign.engines},
        sessions_per_draw=campaign.sessions_per_draw, stopped=StopReason.TARGET_REACHED)
    return cal, calibrate.pool_warnings(state, cal)


def _serving(campaign: CampaignSettings, engine: str, draw: int) -> Serving:
    """Serving *draw* of *engine*, refused (InputMismatch) when *campaign* has no such member."""
    if engine not in campaign.engines:
        raise InputMismatch(f"{engine} is not one of the campaign's pool members {campaign.engines}")
    return Serving(engine, draw, calibrate.serving_seed_of(campaign.task_seed, draw))


def _run_instrument(args: argparse.Namespace, ref: TaskRef, task: TaskSpec,
                    campaign: CampaignSettings) -> list[Path]:
    async def check() -> InstrumentRecord:
        async with in_process(args.gpus, load_sec=campaign.load_timeout_sec) as deploy:
            return await instrument(task, campaign.kind, deploy)

    out = args.out / INSTRUMENT_FILE
    out.write_text(asyncio.run(check()).model_dump_json(indent=2))
    return [out]


def _run_warm(args: argparse.Namespace, ref: TaskRef, task: TaskSpec, campaign: CampaignSettings) -> list[Path]:
    if args.engine not in campaign.engines:
        raise InputMismatch(f"{args.engine} is not one of the campaign's pool members {campaign.engines}")

    async def build() -> None:
        async with in_process(load_sec=args.load_sec) as deploy:
            await warm(task, args.engine, deploy)

    asyncio.run(build())
    return []


def _run_capture(args: argparse.Namespace, ref: TaskRef, task: TaskSpec, campaign: CampaignSettings) -> list[Path]:
    serving = _serving(campaign, args.engine, args.draw)
    if args.seed != serving.seed:
        raise InputMismatch(f"serving {args.draw} is clocked with seed {serving.seed}, not {args.seed}")
    sessions = calibrate.load_sessions(task, args.rows, Split.TEST)
    store = CaptureStore(args.view, campaign.workload_fingerprint)

    async def drive() -> Path:
        async with in_process(load_sec=campaign.load_timeout_sec) as deploy:
            return await capture(task, serving, kind=campaign.kind, sessions=sessions,
                                 sessions_per_draw=campaign.sessions_per_draw,
                                 request_timeout_sec=campaign.request_timeout_sec,
                                 drive_timeout_sec=campaign.drive_timeout_sec, store=store, deploy=deploy)

    made = asyncio.run(drive())
    return [_copy_out(made, args.out / capture_file(campaign.workload_fingerprint, args.engine, args.draw))]


def _run_score(args: argparse.Namespace, ref: TaskRef, task: TaskSpec, campaign: CampaignSettings) -> list[Path]:
    fingerprint = campaign.workload_fingerprint
    store = CaptureStore(args.view, fingerprint)
    servings = []
    for engine, draw, sha256, path in args.capture:
        serving = _serving(campaign, engine, int(draw))
        if serving in servings:
            raise InputMismatch(f"serving {serving.draw} of {engine} is handed twice")
        _link(store.capture_path(engine, serving.draw), checked(Path(path), sha256))
        servings.append(serving)
    rows = RowIndex.from_sessions(calibrate.load_sessions(task, args.rows, Split.TEST))

    async def teacher_force() -> list[Any]:
        async with in_process(args.gpus, load_sec=campaign.load_timeout_sec) as deploy:
            return await score(task, servings, kind=campaign.kind, share=args.share, store=store, rows=rows,
                               deploy=deploy)

    scored = asyncio.run(teacher_force())
    written = [_copy_out(store.replies_path(s.engine, s.draw), args.out / replies_file(fingerprint, s.engine, s.draw))
               for s in scored]
    record = ScoredServings(servings=[ScoredServing(engine=s.engine, draw=s.draw, n_tokens=s.n_tokens,
                                                    n_replies=s.n_items, serving_version=s.serving_version)
                                      for s in scored])
    out = args.out / SCORED_FILE
    out.write_text(record.model_dump_json(indent=2))
    return [*written, out]


def _run_aggregate(args: argparse.Namespace, ref: TaskRef, task: TaskSpec,
                   campaign: CampaignSettings) -> list[Path]:
    sha256, path = args.instrument
    record = InstrumentRecord.model_validate_json(checked(Path(path), sha256).read_bytes())
    store = CaptureStore(args.view, campaign.workload_fingerprint)
    servings: dict[str, dict[int, int]] = {}
    for engine, draw, tokens, digest, file in args.serving:
        serving = _serving(campaign, engine, int(draw))
        if serving.draw in servings.get(engine, {}):
            raise InputMismatch(f"serving {serving.draw} of {engine} is handed twice")
        _link(store.replies_path(engine, serving.draw), checked(Path(file), digest))
        servings.setdefault(engine, {})[serving.draw] = int(tokens)
    caches = _MEMBER_CACHES.validate_json(args.member_caches.read_bytes())
    named = sorted((e, d) for e, by_draw in caches.items() for d in by_draw)
    handed = sorted((e, d) for e, by_draw in servings.items() for d in by_draw)
    if named != handed:
        raise InputMismatch(f"the member caches name servings {named}, not the ones handed, {handed}")
    cal, warnings = aggregate(task, ref.module, campaign, record, store, servings)
    if args.hardware is not None:
        hardware = _HARDWARE.validate_json(args.hardware.read_bytes())
        cal = cal.model_copy(update={"hardware": calibration_hardware(hardware)})
    written = calibrate.write_calibration(cal, args.out)
    summary = AggregateRecord(
        kind=campaign.kind, workload_fingerprint=cal.workload_fingerprint, stopped=StopReason.TARGET_REACHED,
        servings={e: sorted(by_draw) for e, by_draw in servings.items()},
        tokens={e: sum(by_draw.values()) for e, by_draw in servings.items()},
        ceiling=cal.kappa if isinstance(cal, ReportCalibration) else cal.q_max,
        member_caches={e: dict(sorted(by_draw.items())) for e, by_draw in sorted(caches.items())},
        warnings=warnings)
    out = args.out / AGGREGATE_FILE
    out.write_text(summary.model_dump_json(indent=2))
    return [written, out]


#: The aggregate's ``--member-caches`` file: each serving's ``EngineCacheRecord``, by engine and index.
_MEMBER_CACHES = TypeAdapter(dict[str, dict[int, EngineCacheRecord]])
#: The aggregate's ``--hardware`` file: the hardware record of each serving's allocation.
_HARDWARE = TypeAdapter(list[HardwareRecord])

_ROLES: dict[str, Callable[[argparse.Namespace, TaskRef, TaskSpec, CampaignSettings], list[Path]]] = {
    "instrument": _run_instrument, "warm": _run_warm, "capture": _run_capture, "score": _run_score,
    "aggregate": _run_aggregate}


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--task-ref", type=Path, required=True)
    common.add_argument("--campaign", type=Path, required=True)
    common.add_argument("--out", type=Path, required=True)
    common.add_argument("--outputs-out", type=Path, required=True)
    p = argparse.ArgumentParser(prog="hls.verify.calibration_roles")
    roles = p.add_subparsers(dest="role", required=True)
    instrument_role = roles.add_parser("instrument", parents=[common])
    instrument_role.add_argument("--gpus", type=int, required=True)
    warm_role = roles.add_parser("warm", parents=[common])
    warm_role.add_argument("--engine", required=True)
    warm_role.add_argument("--load-sec", type=float, required=True)
    capture_role = roles.add_parser("capture", parents=[common])
    capture_role.add_argument("--rows", type=Path, required=True)
    capture_role.add_argument("--engine", required=True)
    capture_role.add_argument("--draw", type=int, required=True)
    capture_role.add_argument("--seed", type=int, required=True)
    capture_role.add_argument("--view", type=Path, required=True)
    score_role = roles.add_parser("score", parents=[common])
    score_role.add_argument("--rows", type=Path, required=True)
    score_role.add_argument("--gpus", type=int, required=True)
    score_role.add_argument("--share", type=int, required=True)
    score_role.add_argument("--capture", nargs=4, action="append", required=True,
                            metavar=("ENGINE", "DRAW", "SHA256", "FILE"))
    score_role.add_argument("--view", type=Path, required=True)
    aggregate_role = roles.add_parser("aggregate", parents=[common])
    aggregate_role.add_argument("--instrument", nargs=2, required=True, metavar=("SHA256", "FILE"))
    aggregate_role.add_argument("--serving", nargs=5, action="append", required=True,
                                metavar=("ENGINE", "DRAW", "TOKENS", "SHA256", "FILE"))
    aggregate_role.add_argument("--member-caches", type=Path, required=True)
    aggregate_role.add_argument("--hardware", type=Path)
    aggregate_role.add_argument("--view", type=Path, required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    configure_logging()
    try:
        ref = TaskRef.model_validate_json(args.task_ref.read_bytes())
        task = load_task(ref)
        campaign = CampaignSettings.model_validate_json(args.campaign.read_bytes())
        found = workload_fingerprint(task)
        if found != campaign.workload_fingerprint:
            raise InputMismatch(f"{ref.module} has workload fingerprint {found}; the plan keys its captures by "
                                f"{campaign.workload_fingerprint}")
        args.out.mkdir(parents=True, exist_ok=True)
        files = _ROLES[args.role](args, ref, task, campaign)
        report_outputs(args.outputs_out, args.out, *files)
    except (CalibrationError, InputMismatch) as e:
        reason = " ".join(str(e).split())
        print(f"{REFUSED_PREFIX}{type(e).__name__}: {reason}", file=sys.stderr)
        return REFUSED_EXIT
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print(f"calibration {args.role} role failed; it reported no output", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
