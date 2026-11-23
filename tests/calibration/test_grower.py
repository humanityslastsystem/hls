"""The calibration grower (``hls.pipeline.calibration_grower.grow``), asked as the orchestrator asks it once
every planned stage has published (``hls.pipeline.growth.grow``): more servings while an engine's
exposure falls short of its target, dealt over --fanout new allocations; the aggregate once no
engine owes a serving; nothing once the aggregate is in the plan. It is refused past the
campaign's servings bound, for scored records that are not what was planned and for a provision
record that is not its manifest's, and it reads only the score stages' published ``scored.json``
and the provision stage's ``provisioned.json``, by which it sizes a round's copies of the data."""

from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from hls.measured import store_key
from hls.calibrate import DEFAULT_DELTA, DEFAULT_EPS, POOL_ENGINE_NAMES, serving_seed_of
from hls.pipeline import calibration, calibration_grower, commands
from hls.pipeline.grown import grown_stages
from hls.pipeline.growth import GrowthComplete, GrowthDecl, GrowthRefused, PlanGrowth, grow
from hls.pipeline.plan import Outcome, Plan
from hls.spec.resources import ResourceShape
from hls.stages.calibration import (AGGREGATE, GROWER, SCRATCH_GIB, AggregateParams, CaptureParams,
                                    InstrumentParams, ScoreShareParams, capture_stage, score_stage)
from hls.stages.engine_cache import ENGINE_CACHE_GIB
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.verify import PROVISIONED_FILE
from hls.provision.model import Provisioned
from hls.provision.mounts import copied_entries, copies_gib
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.verify.calibration import CALIBRATION_FILE, DEFAULT_EOT_EXPECTED_MAX, StopReason
from hls.verify.calibration_roles import (AGGREGATE_FILE, SCORED_FILE, AggregateRecord, ScoredServing,
                                          ScoredServings)
from tests.calibration import fixture
from tests.calibration.published import create, publish
from tests.calibration.test_hls_calibrate import limited_config
from tests.provision.cached import provision_task
from tests.provision.warmer_config import write_config

REPO = Path(__file__).resolve().parents[2]
#: Teacher-forced tokens per serving in these tests.
PER_SERVING = 5000
#: The per-token tail tolerance whose exposure target is 49,000 tokens: eight servings of 5,000
#: leave an engine 9,000 short, which two more servings close.
MU = math.log(40) * 1e6 / 49_000


def _plan(tmp_path: Path, *, module: str = "cal_flips.task", servings_per_engine: int | None = None,
          max_servings: int = 24, overrides: dict[str, str] | None = None,
          config: Path | None = None, published: Provisioned | None = None) -> tuple[Plan, PosixStore]:
    """A calibration of vllm alone at a fanout of 2, written to the store, the task's provisioning in
    the cache first; its provision stage published that record, or *published*."""
    provisioned = provision_task(tmp_path / "store", module, overrides)
    plan = calibration.plan_calibrate(
        config=config or write_config(tmp_path), overrides=overrides or {}, task_module=module, engines=["vllm"],
        fanout=2,
        servings_per_engine=servings_per_engine, sessions_per_draw=None, eps=DEFAULT_EPS, mu_per_mtok=MU,
        delta=DEFAULT_DELTA, eot_expected_max=DEFAULT_EOT_EXPECTED_MAX, max_servings_per_engine=max_servings,
        job_name="hls-calibrate", echo=[].append)
    store = PosixStore(plan.store_root)
    plan = create(store, plan)
    record = published if published is not None else provisioned
    publish(store, plan, "provision", {"provisioned": (PROVISIONED_FILE, record.model_dump_json().encode())})
    return plan, store


def _score(store: PosixStore, plan: Plan, stage: str, *, tokens: int = PER_SERVING,
           servings: list[tuple[str, int]] | None = None) -> None:
    """Publish *stage* as having scored its planned servings, or *servings*, *tokens* each."""
    planned = [tuple(s) for s in ScoreShareParams.model_validate(plan.stage(stage).params).servings]
    record = ScoredServings(servings=[ScoredServing(engine=e, draw=d, n_tokens=tokens, n_replies=4,
                                                    serving_version="0.0.0") for e, d in servings or planned])
    publish(store, plan, stage, {"scored": (SCORED_FILE, record.model_dump_json().encode())})


def _grown(plan: Plan, growth: PlanGrowth) -> Plan:
    """*plan* with *growth* appended, as the orchestrator's next revision holds it."""
    return plan.model_copy(update={"stages": [*plan.stages, *growth.stages],
                                   "allocations": [*plan.allocations, *growth.allocations]})


def test_the_grower_tops_up_while_exposure_is_short(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    asked = calibration_grower.grow(plan, store)
    assert isinstance(asked, GrowthDecl)
    # 8 servings of 5,000 tokens are 9,000 short of 49,000: two more, numbered after the plan's,
    # dealt over the fanout of two, each scored in its own allocation.
    assert [d.name for d in asked.decls] == ["capture_vllm_8", "score_3", "capture_vllm_9", "score_4"]
    assert asked.groups == (("capture_vllm_8", "score_3"), ("capture_vllm_9", "score_4"))
    task = fixture.flips_task()
    seeds = [CaptureParams.model_validate(d.params).seed for d in asked.decls if d.kind == "calibrate_capture"]
    assert seeds == [serving_seed_of(task.seed, 8), serving_seed_of(task.seed, 9)]
    assert asked.reason == ("exposure short of its target: vllm: 8 servings, 40000 tokens of 49000, 2 more owed, "
                            "2 of them now")
    growth = grow(plan, store)
    assert isinstance(growth, PlanGrowth)
    # New allocations after the plan's three, each waiting for nothing: everything they read has
    # published.
    assert [(a.index, a.stages, a.depends_on, a.waits_for) for a in growth.allocations] == [
        (3, ["capture_vllm_8", "score_3"], None, []), (4, ["capture_vllm_9", "score_4"], None, [])]
    score = next(s for s in growth.stages if s.name == "score_3")
    assert score.after == ["provision", "images", "instrument", "capture_vllm_8"]
    # Each serving is seeded from the engine cache its member's warm stage left.
    capture = growth.stages[0]
    assert capture.after == ["provision", "images", "warm_vllm"]
    assert CaptureParams.model_validate(capture.params).warm_stage == "warm_vllm"
    campaign = InstrumentParams.model_validate(plan.stage("instrument").params).campaign
    assert score.shape.time_sec == calibration_grower.score_sec(campaign, 1)


def test_a_rounds_captures_are_timed_as_the_first_rounds(tmp_path, login, in_catalog):
    """A round's capture stages get the campaign's capture time (``capture_sec``), as the first
    round's do."""
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    asked = calibration_grower.grow(plan, store)
    assert isinstance(asked, GrowthDecl)
    campaign = InstrumentParams.model_validate(plan.stage("instrument").params).campaign
    first = plan.stage("capture_vllm_0").shape.time_sec
    assert first == calibration_grower.capture_sec(campaign)
    assert {d.name: d.shape.time_sec for d in asked.decls if d.kind == "calibrate_capture"} == {
        "capture_vllm_8": first, "capture_vllm_9": first}


def test_a_rounds_stages_hold_the_copies_of_the_data_their_instance_provisioned(tmp_path, login, in_catalog):
    """A round's capture and score stages copy the calibration role's data as the first round's do,
    and the grower sizes those copies by the provision record the instance's provision stage
    published, which they copy from: here one whose rows take 7 GiB more than the cache's record the
    first round was planned with."""
    provisioned = provision_task(tmp_path / "store", "cal_flips.task")
    rows = provisioned.rows.model_copy(update={"disk_bytes": provisioned.rows.disk_bytes + 7 * 2**30})
    published = provisioned.model_copy(update={"rows": rows})
    plan, store = _plan(tmp_path, published=published)
    first = copied_entries(("calibration",), provisioned)
    assert CaptureParams.model_validate(plan.stage("capture_vllm_0").params).data_copies == first
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    asked = calibration_grower.grow(plan, store)
    data = copied_entries(("calibration",), published)
    assert copies_gib(data) == copies_gib(first) + 7
    for decl in asked.decls:
        assert decl.params["data_copies"] == data, decl.name
        assert decl.shape.disk_gib == KEPT_DISK_GIB + ENGINE_CACHE_GIB + SCRATCH_GIB + copies_gib(data), decl.name


def test_a_provision_record_that_is_not_its_manifests_refuses_the_growth(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    key = read_manifest(store, plan.layout, "provision")[0].outputs["provisioned"].key
    store.write_bytes(key, store.read_bytes(key) + b" ")
    asked = calibration_grower.grow(plan, store)
    assert isinstance(asked, GrowthRefused)
    assert asked.reason == f"{key} is not the {PROVISIONED_FILE} its manifest recorded"


def test_a_top_up_round_takes_at_most_the_fanout(tmp_path, login, in_catalog):
    # Eight servings of 2,000 tokens leave the engine 33,000 short of 49,000, so the counter's
    # estimate owes seventeen more; the round takes two, the fanout, and the counter is asked
    # again once they are scored.
    plan, store = _plan(tmp_path, max_servings=40)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage, tokens=2000)
    asked = calibration_grower.grow(plan, store)
    assert [d.name for d in asked.decls] == ["capture_vllm_8", "score_3", "capture_vllm_9", "score_4"]
    assert asked.reason == ("exposure short of its target: vllm: 8 servings, 16000 tokens of 49000, 17 more owed, "
                            "2 of them now")


def test_the_grower_appends_the_aggregate_once_demand_is_met(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    plan = _grown(plan, grow(plan, store))
    for stage in ("score_3", "score_4"):
        _score(store, plan, stage)
    growth = grow(plan, store)
    assert isinstance(growth, PlanGrowth)
    [stage] = growth.stages
    shares = ["score_1", "score_2", "score_3", "score_4"]
    # It reads each serving's engine cache off its capture stage.
    captures = [f"capture_vllm_{d}" for d in (0, 2, 4, 6, 1, 3, 5, 7, 8, 9)]
    assert (stage.name, stage.kind, stage.after) == ("aggregate", "calibrate_aggregate",
                                                     ["instrument", *shares, *captures])
    # Sized for the forty replies the ten servings measured: a GiB beside the runner's, and the
    # floor's ten minutes with a second for the reads.
    assert stage.shape == ResourceShape(cpus=2, memory_gib=1 + plan.settings.run.runner_memory_gib, time_sec=601)
    assert AggregateParams.model_validate(stage.params).score_stages == shares
    [alloc] = growth.allocations
    assert (alloc.index, alloc.stages) == (5, ["aggregate"])


def _servings(n: int, *, replies: int, tokens: int) -> list[ScoredServing]:
    """*n* of vllm's servings as score roles record them, each of *replies* replies and *tokens* tokens."""
    return [ScoredServing(engine="vllm", draw=d, n_tokens=tokens, n_replies=replies, serving_version="0.0.0")
            for d in range(n)]


def test_the_aggregate_is_sized_for_the_replies_or_tokens_it_reads():
    flips, report = fixture.campaign(), fixture.campaign(fixture.report_task())
    # tab_completion's eight servings, of about 22,000 replies and 1.71M tokens each: their records
    # hold well under a GiB, read in ten seconds at the margin, beside the floor's ten minutes.
    tab = calibration_grower.aggregate_shape(flips, _servings(8, replies=22_000, tokens=1_710_000),
                                             runner_memory_gib=4)
    assert tab == ResourceShape(cpus=2, memory_gib=1 + 4, time_sec=611)
    # A token-level record holds no value a token: twice the tokens leave the shape as it was, and
    # twice the replies take twice the reading.
    assert calibration_grower.aggregate_shape(flips, _servings(8, replies=22_000, tokens=3_420_000),
                                              runner_memory_gib=4) == tab
    assert calibration_grower.aggregate_shape(flips, _servings(8, replies=44_000, tokens=1_710_000),
                                              runner_memory_gib=4).time_sec == 621
    # A sampling-gate record keeps two values a token: eight servings of TTS's 3.47M tokens hold
    # 4.3 GB, twice that rounded up to 9 GiB beside the runner's, read in 173 s at the margin;
    # their replies' count does not change it.
    tts = calibration_grower.aggregate_shape(report, _servings(8, replies=2400, tokens=3_470_000),
                                             runner_memory_gib=4)
    assert tts == ResourceShape(cpus=2, memory_gib=9 + 4, time_sec=773)
    assert calibration_grower.aggregate_shape(report, _servings(8, replies=4800, tokens=3_470_000),
                                              runner_memory_gib=4) == tts


def test_enough_exposure_in_the_first_servings_goes_straight_to_the_aggregate(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage, tokens=7000)
    asked = calibration_grower.grow(plan, store)
    assert [d.name for d in asked.decls] == ["aggregate"] and asked.groups == (("aggregate",),)


def test_the_grower_refuses_past_the_servings_bound(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path, max_servings=9)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    refused = grow(plan, store)
    assert isinstance(refused, GrowthRefused)
    assert refused.reason == ("vllm would take 10 servings with this round, over the campaign's bound of 9: its 8 "
                              "servings measured 40000 of the 49000 tokens its exposure target asks for, and the "
                              "counter owes 2 more; raise --max-servings-per-engine, or --mu-per-mtok to lower the "
                              "target")


@pytest.mark.parametrize(("max_servings", "grows"), [(24, True), (9, False)])
def test_the_bound_holds_what_a_round_takes_not_the_counters_projection(tmp_path, login, in_catalog, max_servings,
                                                                        grows):
    # Eight servings of 2,000 tokens: the counter's estimate owes seventeen more, 25 in all,
    # past a bound of 24, but the round takes two, 10 in all, which 24 holds and 9 does not. The
    # estimate rests on the yield measured so far, so the bound is held a round at a time.
    plan, store = _plan(tmp_path, max_servings=max_servings)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage, tokens=2000)
    asked = calibration_grower.grow(plan, store)
    if grows:
        assert isinstance(asked, GrowthDecl)
        assert [d.name for d in asked.decls] == ["capture_vllm_8", "score_3", "capture_vllm_9", "score_4"]
    else:
        assert isinstance(asked, GrowthRefused)
        assert asked.reason == ("vllm would take 10 servings with this round, over the campaign's bound of 9: its 8 "
                                "servings measured 16000 of the 49000 tokens its exposure target asks for, and the "
                                "counter owes 17 more; raise --max-servings-per-engine, or --mu-per-mtok to lower "
                                "the target")


def test_the_grower_is_done_once_the_aggregate_published(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage, tokens=7000)
    plan = _grown(plan, grow(plan, store))
    publish(store, plan, "aggregate", {"aggregate": (AGGREGATE_FILE, b"{}")})
    assert calibration_grower.grow(plan, store) is None
    assert isinstance(grow(plan, store), GrowthComplete)


def test_a_succeeded_calibration_says_each_warning_its_aggregate_recorded(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage, tokens=7000)
    plan = _grown(plan, grow(plan, store))
    warning = "pool engine 'vllm': severe flips recur across servings"
    record = AggregateRecord(kind="flips", workload_fingerprint="0" * 64, stopped=StopReason.TARGET_REACHED,
                             servings={"vllm": list(range(8))}, tokens={"vllm": 56_000}, ceiling=0.002,
                             member_caches={}, warnings=[warning])
    keys = publish(store, plan, "aggregate", {"calibration": (CALIBRATION_FILE, b"{}"),
                                              "aggregate": (AGGREGATE_FILE, record.model_dump_json().encode())})
    said: list[str] = []
    ended = plan.model_copy(update={"outcome": Outcome(status="succeeded", reason="every stage published", at=1.0)})
    commands._say_calibration(ended, said.append)
    measured = store_key(plan.task, CALIBRATION_FILE)
    assert said == [f"calibration (flips): ceiling 0.002 over vllm 8 servings, in {keys['calibration']}; the store's "
                    f"{measured} holds it, and runs of the task use it", f"warning: {warning}"]


def test_the_grower_reads_only_records_their_manifests_recorded(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    key = read_manifest(store, plan.layout, "score_2")[0].outputs["scored"].key
    store.path(key).write_text(store.path(key).read_text().replace('"n_tokens":5000', '"n_tokens":50000'))
    refused = grow(plan, store)
    assert isinstance(refused, GrowthRefused)
    assert refused.reason == f"{key} is not the {SCORED_FILE} its manifest recorded"


def test_a_share_that_scored_other_servings_is_refused(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path)
    _score(store, plan, "score_1")
    _score(store, plan, "score_2", servings=[("vllm", 1), ("vllm", 3), ("vllm", 5)])
    refused = grow(plan, store)
    assert isinstance(refused, GrowthRefused)
    assert refused.reason == ("score_2 scored [('vllm', 1), ('vllm', 3), ('vllm', 5)], not the servings it was "
                              "planned to score, [('vllm', 1), ('vllm', 3), ('vllm', 5), ('vllm', 7)]")


def test_a_serving_two_shares_scored_is_refused(tmp_path, login, in_catalog):
    # A plan whose second share was dealt a serving of the first: both scored it as planned.
    plan, store = _plan(tmp_path)
    second = plan.stage("score_2")
    params = ScoreShareParams.model_validate(second.params)
    dealt = second.model_copy(update={"params": params.model_copy(
        update={"servings": [("vllm", 0), *params.servings]}).model_dump(mode="json")})
    plan = plan.model_copy(update={"stages": [dealt if s.name == "score_2" else s for s in plan.stages]})
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    refused = calibration_grower.grow(plan, store)
    assert isinstance(refused, GrowthRefused) and refused.reason == "serving 0 of vllm is scored twice"


def test_a_resources_table_reaches_a_grown_stage_through_the_declaration(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path, overrides={f"resources.{fixture.TASK_ID}.score_3.time_sec": "9000"})
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    growth = grow(plan, store)
    assert isinstance(growth, PlanGrowth)
    times = {s.name: s.shape.time_sec for s in growth.stages if s.kind == "calibrate_score"}
    campaign = InstrumentParams.model_validate(plan.stage("instrument").params).campaign
    assert times == {"score_3": 9000, "score_4": calibration_grower.score_sec(campaign, 1)}


def test_a_grown_allocation_no_qos_holds_refuses_the_growth(tmp_path, login, in_catalog):
    # A top-up's allocations are sized under the site's limits as the first plan's are: one that
    # no QOS holds, here through a [resources] table on its score stage, voids the instance.
    plan, store = _plan(tmp_path, config=limited_config(tmp_path, "{normal = 24, long = 168}"),
                        overrides={f"resources.{fixture.TASK_ID}.score_3.time_sec": str(200 * 3600)})
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage)
    refused = grow(plan, store)
    assert isinstance(refused, GrowthRefused)
    assert re.fullmatch(r"its stages cannot be planned: allocation 3: it asks for [0-9.]+ h, more than any QOS of "
                        r"the site lets a job on partition gpu-a ask \(site.qos_by_hours\): the longest, long, "
                        r"allows 168 h; the pipeline lays this allocation out itself, so the planner does not split "
                        r"it", refused.reason), refused.reason


def test_a_stage_the_declaration_does_not_hold_is_refused_by_grow(tmp_path, login, in_catalog, monkeypatch):
    plan, store = _plan(tmp_path)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage, tokens=7000)
    monkeypatch.setattr(calibration_grower, "AGGREGATE", "assemble")
    refused = grow(plan, store)
    assert isinstance(refused, GrowthRefused)
    assert refused.reason == "its stages cannot be planned: the grower added stages it does not declare: ['assemble']"


def test_the_grower_declares_exactly_the_names_its_stages_are_given():
    # The stage module that names the grown stages declares them, so its declaration holds every
    # capture of a pool member's serving, draws from 0, and every share's score stage, shares from
    # 1, and no other name.
    declared = grown_stages(GROWER)
    assert declared.names == {AGGREGATE}
    made = [capture_stage(e, d) for e in POOL_ENGINE_NAMES for d in (0, 1, 9, 10, 123)]
    made += [score_stage(s) for s in (1, 2, 9, 10, 123)]
    assert [n for n in made if not declared.holds(n)] == []
    near = ["capture_vllm_01", "capture_vllm_-1", "capture_vllm_1.0", "capture_vllm_", "capture_hf_pinned_0",
            "capture_vllmx_0", "capture_vllm_0_0", "capture_vllm|sglang_0", "score_0", "score_01", "score_",
            "score_1x", "warm_vllm", "instrument", "aggregate_1"]
    assert [n for n in near if declared.holds(n)] == []


def test_a_report_calibration_grows_only_its_aggregate(tmp_path, login, in_catalog):
    plan, store = _plan(tmp_path, module="cal_report.task", servings_per_engine=2)
    for stage in ("score_1", "score_2"):
        _score(store, plan, stage, tokens=12)
    asked = calibration_grower.grow(plan, store)
    assert [d.name for d in asked.decls] == ["aggregate"]


@pytest.mark.parametrize(("tokens", "owed"), [
    pytest.param([], 8, id="nothing-scored"),
    pytest.param([5000] * 8, 2, id="short"),
    pytest.param([7000] * 8, 0, id="met"),
    pytest.param([5000] * 3, 7, id="short-of-both"),
    pytest.param([20000] * 3, 5, id="under-the-floor"),
])
def test_demand_is_the_old_exposure_counter(tokens, owed):
    campaign = fixture.campaign(engines=["vllm"], mu_per_mtok=MU)
    assert calibration_grower.demand(campaign, {"vllm": tokens}) == {"vllm": owed}


def test_the_sampling_gates_demand_is_the_rest_of_its_servings():
    campaign = fixture.campaign(fixture.report_task(), engines=["vllm"], servings_per_engine=3)
    assert calibration_grower.demand(campaign, {"vllm": [10]}) == {"vllm": 2}
    assert calibration_grower.demand(campaign, {"vllm": [10] * 3}) == {"vllm": 0}


FRESH = """
import sys
from pathlib import Path
import hls.pipeline.commands
from hls.pipeline.growth import grow
from hls.pipeline.plan import Plan
from hls.store.posix import PosixStore
plan = Plan.model_validate_json(Path(sys.argv[1]).read_text())
print("hls.pipeline.calibration" in sys.modules, type(grow(plan, PosixStore(plan.store_root))).__name__)
"""


def test_an_orchestrator_that_never_imported_the_calibration_finds_its_grower(tmp_path, login, in_catalog):
    # hls resume and hls cancel import the commands, which import no calibration command: the
    # orchestrator finds the grower its plan names through hls.pipeline.growers. Nothing has
    # scored, so the grower refuses.
    plan, _ = _plan(tmp_path)
    path = tmp_path / "plan.json"
    path.write_text(plan.model_dump_json())
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO)])}
    done = subprocess.run([sys.executable, "-c", FRESH, str(path)], capture_output=True, text=True, cwd=REPO,
                          env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "False GrowthRefused"
