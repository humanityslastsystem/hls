"""The calibration grower (``hls calibrate``): what a calibration's plan adds once
every planned stage has published, and the stages it lays a round of servings out as, which the
first plan's servings are laid out as too (``hls.pipeline.calibration``).

A round's servings are dealt round robin over ``min(fanout, servings)`` allocations, each serving
a capture stage (``capture_<engine>_<draw>``) and each allocation's captures scored by one score
stage (``score_<j>``) after them (``serving_stages``). Once every planned stage has published, the
orchestrator asks the grower (``grow``), which reads the score stages' ``scored.json`` and the
provision stage's ``provisioned.json``, each checked against its manifest, and nothing else:
while an engine's exposure falls short of the token-level gate's target it adds the first
``fanout`` of the servings the counter owes (``hls.calibrate.servings_owed``, ``round_servings``)
in new allocations dealt the same way, and asks the counter again once they are scored; when no
engine owes any it adds ``aggregate``, alone in a last allocation sized for the scored servings it
reads (``aggregate_shape``), which assembles the calibration; once that is in the plan it adds
nothing, and the instance ends. A round that would take an engine past
``max_servings_per_engine`` servings is refused, and the instance ends void, as do a calibration
that fails its own checks and a record that is not what the plan describes. A round's stages
size their copies of the calibration role's data by the provision record the instance's
provision stage published, which they copy from.

The stages it adds are named, and declared to the planner, by ``hls.stages.calibration``. This
module imports no command, so ``hls.pipeline.growers`` registers it wherever an orchestrator
runs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from pydantic import ValidationError

from hls.calibrate import Serving, plan_servings, servings_owed
from hls.pipeline.growth import GrowthDecl, GrowthRefused, register_grower
from hls.pipeline.pipelines import DRIVE_MARGIN_SEC, OPEN_DRIVE_SEC, STAGING_SEC
from hls.pipeline.plan import Plan
from hls.pipeline.planner import StageDecl
from hls.provision.model import Provisioned
from hls.provision.mounts import copied_entries, copies_gib
from hls.spec.resources import ResourceShape
from hls.stages.calibration import (AGGREGATE, GROWER, INSTRUMENT, SCRATCH_GIB, CaptureParams, InstrumentParams,
                                    ScoreShareParams, capture_stage, score_stage, stage_shape, warm_stage)
from hls.stages.engine_cache import ENGINE_CACHE_GIB
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.verify import PROVISIONED_FILE
from hls.store.base import Store
from hls.store.digest import sha256_bytes
from hls.store.manifest import read_manifest
from hls.substrate.factory import seal_memory_gib
from hls.taskref import TaskRef
from hls.util import validation_errors
from hls.verify.calibration import exposure_target
from hls.verify.calibration_roles import SCORED_FILE, CampaignSettings, ScoredServing, ScoredServings

#: The aggregate stage's CPUs: its role's one Python thread, and the runner's beside it.
AGGREGATE_CPUS = 2
#: Bytes the aggregate role holds before it reads a serving: the interpreter, the harness and the
#: task. This and the figures below are the role's own on an Intel Xeon Platinum 8570.
AGGREGATE_BASE_BYTES = 92 * 2**20
#: Bytes the aggregate role holds for each scored reply of the token-level gate: its record of
#: counts and its few severe gaps, whatever the reply's length.
AGGREGATE_FLIPS_REPLY_BYTES = 298
#: Bytes the aggregate role holds for each teacher-forced token of the sampling gate, whose
#: records keep two values a position, each reply's own record included.
AGGREGATE_REPORT_TOKEN_BYTES = 153
#: Seconds the aggregate role takes for each scored reply of the token-level gate.
AGGREGATE_FLIPS_REPLY_SEC = 5.7e-6
#: Seconds the aggregate role takes for each teacher-forced token of the sampling gate.
AGGREGATE_REPORT_TOKEN_SEC = 0.62e-6
#: The margin over the memory the figures above give for what the aggregate role reads.
AGGREGATE_MEMORY_MARGIN = 2
#: The margin over the reading time the figures above give, for a cluster node slower than that
#: core and a loaded shared filesystem.
AGGREGATE_TIME_MARGIN = 10
#: Seconds the aggregate stage is given beside its role's reads: ten minutes for the stage's own
#: reads and its role's start, on a node whose shared file system may answer slowly.
AGGREGATE_FLOOR_SEC = 600


def capture_sec(campaign: CampaignSettings) -> int:
    """A capture stage's time: its staging, its copies' and its extraction's (``STAGING_SEC``), the
    bound of the member's load (``CampaignSettings.load_timeout_sec``), the drive's wall
    (``OPEN_DRIVE_SEC`` when the task declares none) and the margin after it."""
    wall = campaign.drive_timeout_sec
    drive = math.ceil(wall) if wall is not None else OPEN_DRIVE_SEC
    return STAGING_SEC + math.ceil(campaign.load_timeout_sec) + drive + DRIVE_MARGIN_SEC


def score_sec(campaign: CampaignSettings, servings: int) -> int:
    """A score stage's time for *servings* captures: its staging, its copies' and its extraction's
    (``STAGING_SEC``), the bound of the reference's load (``CampaignSettings.load_timeout_sec``),
    and ``CampaignSettings.score_per_serving_sec`` for each capture."""
    return STAGING_SEC + math.ceil(campaign.load_timeout_sec) + servings * campaign.score_per_serving_sec


def demand(campaign: CampaignSettings, tokens: Mapping[str, Sequence[int]]) -> dict[str, int]:
    """How many more servings each engine of *campaign* owes, given *tokens*, each engine's
    scored servings' teacher-forced tokens: for the token-level gate the counter
    (``hls.calibrate.servings_owed``) against ``exposure_target(mu_per_mtok)``, for the sampling gate the rest
    of ``servings_per_engine``."""
    owed: dict[str, int] = {}
    for engine in campaign.engines:
        have = list(tokens.get(engine, ()))
        # The sampling gate's count is set, and only that gate's (``CampaignSettings``).
        if campaign.servings_per_engine is not None:
            owed[engine] = max(0, campaign.servings_per_engine - len(have))
        else:
            owed[engine] = servings_owed(len(have), sum(have), exposure_target(campaign.mu_per_mtok))
    return owed


def round_servings(campaign: CampaignSettings, tokens: Mapping[str, Sequence[int]],
                   next_draw: Mapping[str, int]) -> list[Serving]:
    """The servings the next round of *campaign* takes, given *tokens*, each engine's scored
    servings' teacher-forced tokens, and *next_draw*, each engine's next serving index: every
    serving ``demand`` owes while no engine owing one has scored any, since that demand is the
    serving floor or the sampling gate's count, which no measurement would shorten; otherwise
    the first ``fanout`` of them, since a top-up rests on the yield measured so far and the
    counter is asked again once they are scored."""
    servings = plan_servings(campaign.task_seed, next_draw, demand(campaign, tokens))
    if all(not tokens.get(s.engine) for s in servings):
        return servings
    return servings[:campaign.fanout]


def member_reserve_gib(campaign: CampaignSettings, runner_memory_gib: int, seal_memory_gib: int) -> int:
    """What a stage running one of *campaign*'s members leaves beside it: the runner's
    *runner_memory_gib*, and *seal_memory_gib* for the member's seal unless the campaign gives
    members the internet (``hls.stages.calibration.member_seal``)."""
    return runner_memory_gib + (0 if campaign.member_internet else seal_memory_gib)


def serving_stages(*, task_ref: TaskRef, campaign: CampaignSettings, servings: Sequence[Serving], first_share: int,
                   runner_memory_gib: int, seal_memory_gib: int,
                   data_copies: dict[str, int]) -> tuple[list[StageDecl], list[tuple[str, ...]]]:
    """The stages that take *servings* in ``min(fanout, len(servings))`` allocations, the servings
    dealt round robin, each allocation's captures followed by its share's score stage (shares
    numbered from *first_share*), and each allocation's stage names, in order. Each stage declares
    *data_copies*, its copies of the calibration role's data, and its disk holds them; its time holds
    its staging (``STAGING_SEC``)."""
    data_gib = copies_gib(data_copies)
    n = min(campaign.fanout, len(servings))
    ref, settings = task_ref.model_dump(mode="json"), campaign.model_dump(mode="json")
    capture_shape = stage_shape(campaign.member_shape,
                                runner_memory_gib=member_reserve_gib(campaign, runner_memory_gib, seal_memory_gib),
                                disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB + data_gib,
                                time_sec=capture_sec(campaign))
    decls: list[StageDecl] = []
    groups: list[tuple[str, ...]] = []
    for i in range(n):
        share, j = servings[i::n], first_share + i
        names = [capture_stage(s.engine, s.draw) for s in share]
        decls += [StageDecl(name=name, kind="calibrate_capture", shape=capture_shape,
                            after=("provision", "images", warm_stage(s.engine)),
                            params={"task": ref, "campaign": settings, "engine": s.engine, "draw": s.draw,
                                    "seed": s.seed, "warm_stage": warm_stage(s.engine), "data_copies": data_copies})
                  for name, s in zip(names, share, strict=True)]
        score_shape = stage_shape(campaign.reference_shape, runner_memory_gib=runner_memory_gib,
                                  disk_gib=KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB + data_gib,
                                  time_sec=score_sec(campaign, len(share)))
        decls.append(StageDecl(name=score_stage(j), kind="calibrate_score", shape=score_shape,
                               after=("provision", "images", INSTRUMENT, *names),
                               params={"task": ref, "campaign": settings, "share": j,
                                       "servings": [[s.engine, s.draw] for s in share], "data_copies": data_copies}))
        groups.append((*names, score_stage(j)))
    return decls, groups


def aggregate_shape(campaign: CampaignSettings, servings: Sequence[ScoredServing], *,
                    runner_memory_gib: int) -> ResourceShape:
    """The aggregate stage's shape for *campaign*'s scored *servings*, from what its role reads of
    them: each reply's record for the token-level gate, each token's values for the sampling gate.
    ``AGGREGATE_CPUS``; the memory the role holds for them, ``AGGREGATE_BASE_BYTES`` and
    ``AGGREGATE_FLIPS_REPLY_BYTES`` a reply or ``AGGREGATE_REPORT_TOKEN_BYTES`` a token, times
    ``AGGREGATE_MEMORY_MARGIN`` and rounded up to the GiB, with *runner_memory_gib* beside it; and
    ``AGGREGATE_FLOOR_SEC`` with the time the role reads them in, ``AGGREGATE_FLIPS_REPLY_SEC`` a
    reply or ``AGGREGATE_REPORT_TOKEN_SEC`` a token, times ``AGGREGATE_TIME_MARGIN``."""
    if campaign.kind == "flips":
        units = sum(s.n_replies for s in servings)
        held, sec = AGGREGATE_FLIPS_REPLY_BYTES, AGGREGATE_FLIPS_REPLY_SEC
    else:
        units = sum(s.n_tokens for s in servings)
        held, sec = AGGREGATE_REPORT_TOKEN_BYTES, AGGREGATE_REPORT_TOKEN_SEC
    memory = math.ceil(AGGREGATE_MEMORY_MARGIN * (AGGREGATE_BASE_BYTES + held * units) / 2**30)
    return ResourceShape(cpus=AGGREGATE_CPUS, memory_gib=memory + runner_memory_gib,
                         time_sec=AGGREGATE_FLOOR_SEC + math.ceil(AGGREGATE_TIME_MARGIN * sec * units))


def aggregate_stage(*, task_ref: TaskRef, campaign: CampaignSettings, servings: Sequence[ScoredServing],
                    score_stages: Sequence[str], capture_stages: Sequence[str], runner_memory_gib: int) -> StageDecl:
    """The stage that assembles the calibration from the instrument's record and the scored
    *servings* of *score_stages*, and reads the engine cache each serving ran with off
    *capture_stages*, sized for those servings (``aggregate_shape``)."""
    shape = aggregate_shape(campaign, servings, runner_memory_gib=runner_memory_gib)
    return StageDecl(name=AGGREGATE, kind="calibrate_aggregate", shape=shape,
                     after=(INSTRUMENT, *score_stages, *capture_stages),
                     params={"task": task_ref.model_dump(mode="json"), "campaign": campaign.model_dump(mode="json"),
                             "score_stages": list(score_stages)})


def scored_servings(plan: Plan, store: Store) -> dict[str, dict[int, ScoredServing]]:
    """Every serving the score stages of *plan* scored, by engine and index, from each one's
    published ``scored.json``, checked against the digest its manifest recorded and against the
    servings the stage was planned to score. Raise ValueError for a score stage that has not
    published, a record that is not the one its manifest recorded or not a scored record, one
    that names other servings than planned, and a serving scored twice."""
    found: dict[str, dict[int, ScoredServing]] = {}
    for stage in plan.stages:
        if stage.kind != "calibrate_score":
            continue
        published = read_manifest(store, plan.layout, stage.name)
        if published is None:
            raise ValueError(f"{stage.name} has not published")
        ref = published[0].outputs.get("scored")
        if ref is None or not ref.key.endswith(f"/outputs/{SCORED_FILE}"):
            raise ValueError(f"the {stage.name} manifest records no {SCORED_FILE}")
        data = store.read_bytes(ref.key)
        if sha256_bytes(data) != ref.sha256:
            raise ValueError(f"{ref.key} is not the {SCORED_FILE} its manifest recorded")
        try:
            record = ScoredServings.model_validate_json(data)
        except ValidationError as e:
            raise ValueError(f"{ref.key} is not a scored record: {validation_errors(e)}") from None
        planned = sorted(tuple(s) for s in ScoreShareParams.model_validate(stage.params).servings)
        got = sorted((s.engine, s.draw) for s in record.servings)
        if got != planned:
            raise ValueError(f"{stage.name} scored {got}, not the servings it was planned to score, {planned}")
        for s in record.servings:
            if s.draw in found.setdefault(s.engine, {}):
                raise ValueError(f"serving {s.draw} of {s.engine} is scored twice")
            found[s.engine][s.draw] = s
    return found


@register_grower(GROWER)
def grow(plan: Plan, store: Store) -> GrowthDecl | GrowthRefused | None:
    """What a calibration adds once every planned stage has published (``hls.pipeline.growth``):
    nothing once its aggregate is in the plan; while an engine owes servings (``demand``), the
    round of them ``round_servings`` takes, numbered after the plan's and dealt into new
    allocations as the first were; once none does, the aggregate, alone in a new allocation.
    A round's stages hold their copies of the calibration role's data in their disk, by the
    provision record the instance's provision stage published (``provision_record``), which they
    copy from. Refused for scored records that cannot be read as planned (``scored_servings``), for
    a provision record that is not the one its manifest recorded, and for a round that would take
    an engine past the campaign's ``max_servings_per_engine``. Reads only the score stages'
    ``scored.json`` and the provision stage's ``provisioned.json``."""
    if any(s.name == AGGREGATE for s in plan.stages):
        return None
    params = InstrumentParams.model_validate(plan.stage(INSTRUMENT).params)
    campaign = params.campaign
    try:
        scored = scored_servings(plan, store)
    except ValueError as e:
        return GrowthRefused(str(e))
    tokens = {e: [s.n_tokens for _, s in sorted(scored.get(e, {}).items())] for e in campaign.engines}
    owed = demand(campaign, tokens)
    target = exposure_target(campaign.mu_per_mtok)
    shares = [ScoreShareParams.model_validate(s.params).share for s in plan.stages if s.kind == "calibrate_score"]
    if any(owed.values()):
        captured = [CaptureParams.model_validate(s.params) for s in plan.stages if s.kind == "calibrate_capture"]
        next_draw = {e: max((c.draw for c in captured if c.engine == e), default=-1) + 1 for e in campaign.engines}
        servings = round_servings(campaign, tokens, next_draw)
        taken = Counter(s.engine for s in servings)
        # Held a round at a time: the counter's estimate rests on the yield measured so far.
        for engine, more in owed.items():
            total = len(tokens[engine]) + taken[engine]
            if total > campaign.max_servings_per_engine:
                return GrowthRefused(
                    f"{engine} would take {total} servings with this round, over the campaign's bound of "
                    f"{campaign.max_servings_per_engine}: its {len(tokens[engine])} servings measured "
                    f"{sum(tokens[engine])} of the {target:.0f} tokens its exposure target asks for, and the "
                    f"counter owes {more} more; raise --max-servings-per-engine, or --mu-per-mtok to lower the "
                    "target")
        try:
            data = copied_entries(("calibration",), provision_record(store, plan))
        except ValueError as e:
            return GrowthRefused(str(e))
        decls, groups = serving_stages(task_ref=params.task, campaign=campaign, servings=servings,
                                       first_share=max(shares) + 1,
                                       runner_memory_gib=plan.settings.run.runner_memory_gib,
                                       seal_memory_gib=seal_memory_gib(plan.settings.site), data_copies=data)
        short = "; ".join(f"{e}: {len(tokens[e])} servings, {sum(tokens[e])} tokens of {target:.0f}, "
                          f"{n} more owed, {taken[e]} of them now" for e, n in owed.items() if n)
        return GrowthDecl(decls=tuple(decls), groups=tuple(groups), reason=f"exposure short of its target: {short}")
    stages = [score_stage(j) for j in sorted(shares)]
    captures = [s.name for s in plan.stages if s.kind == "calibrate_capture"]
    every = [s for by_draw in scored.values() for s in by_draw.values()]
    return GrowthDecl(decls=(aggregate_stage(task_ref=params.task, campaign=campaign, servings=every,
                                             score_stages=stages, capture_stages=captures,
                                             runner_memory_gib=plan.settings.run.runner_memory_gib),),
                      groups=((AGGREGATE,),),
                      reason="every engine has the servings it owes; the aggregate assembles the calibration")


def _published(store: Store, plan: Plan, stage: str, name: str) -> tuple[str, str]:
    """The store key and recorded sha256 of output *name* of *plan*'s published *stage*. Raise
    ValueError when the stage has not published or records no such output."""
    found = read_manifest(store, plan.layout, stage)
    if found is None:
        raise ValueError(f"{plan.instance}'s {stage} stage has not published")
    ref = found[0].outputs.get(name)
    if ref is None:
        raise ValueError(f"{plan.instance}'s {stage} manifest records no output {name}")
    return ref.key, ref.sha256


def provision_record(store: Store, plan: Plan) -> Provisioned:
    """The provision record *plan*'s provision stage published, checked against the digest its
    manifest recorded. Raise ValueError when the stage has not published, for a file that is not
    the one its manifest recorded, and for one that is not a provision record."""
    key, sha256 = _published(store, plan, "provision", "provisioned")
    data = store.read_bytes(key)
    if sha256_bytes(data) != sha256 or not key.endswith(f"/{PROVISIONED_FILE}"):
        raise ValueError(f"{key} is not the {PROVISIONED_FILE} its manifest recorded")
    try:
        return Provisioned.model_validate_json(data)
    except ValidationError as e:
        raise ValueError(f"{key} is not a provision record: {validation_errors(e)}") from None
