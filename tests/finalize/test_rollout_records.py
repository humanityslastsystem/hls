"""The records a rollout and a grading write: a rollout that built its submission
records the build from its agent_build and package manifests; a build-only rollout, which grades
nothing, records ``built`` or ``invalid_artifact`` and no grading; the index row names the arm
that built the graded submission and has no ``backend``; a grading of an agent's package counts the
route secrets of the arm that built it contestant-facing (``FinalizeParams.built_arm``, which must
be the arm the build recorded), so a server that echoes the relay key has the files holding it
quarantined and is scored, while the same key in a grading that names no arm is the harness's
leak; and the viewer's fixture rollout and grading are the records finalize writes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.arms import ArmRecord, Route
from hls.obs.record import GradingRecord, RolloutBuild, RolloutRecord
from hls.pipeline import commands
from hls.pipeline.checkout import CodeState
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
from hls.pipeline.plan import Outcome, Plan, Toolchain
from hls.pipeline.planner import StageDecl, build_plan
from hls.pipeline.stage import StageFault
from hls.stages.agent_build import AgentBuildStage
from hls.stages.finalize import FinalizeParams, FinalizeStage, contestant_facing
from hls.stages.score import ScoreStage
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.manifest import output_ref, publish_manifest, read_manifest
from hls.store.posix import PosixStore
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE
from tests.finalize import stand_in_stage  # noqa: F401  (registers the stand-ins)
from tests.finalize.published import (CPU, GRADING, INSTALL_SCAN, ROLLOUT, SIF, TASK, TREE_SCAN, context,
                                      echoed_trace, make_plan, package_result, publish, publish_instance,
                                      publish_rollout_package, run_and_publish, with_inputs, with_params,
                                      write)
from tests.finalize.test_agent_build_scan import ARM, PARAMS, REGISTRY, RELAY, SECRET
from tests.pipeline.plans import settings
from tests.trace.traces import drive_trace, passing_verdict

#: What the fixture agent_build wrote as ``build.json``: read back as written, never parsed into
#: a model.
BUILD = {"arm": ARM.model_dump(mode="json"), "arm_change": None, "sandbox_image": "c" * 64,
         "upload_sha256": "d" * 64, "soft_sec": None,
         "hard_sec": 1100, "max_turns": None, "started_at": 300.0, "ended_at": 400.0, "agent_sec": 100.0,
         "ending": "agent_ended", "reentries": 0, "sessions_ended": 1,
         "extraction": {"usage": {"n_input_tokens": 1000, "n_output_tokens": 50, "n_cache_tokens": None,
                                  "cost_usd": 12.5},
                        "trajectory_sha256": None, "unavailable": None, "trajectory_unavailable": None}}
#: The fixture agent_build's outputs beside its source bundle, by name: each one's file and bytes.
BUILD_FILES = {"build": ("build.json", json.dumps(BUILD, indent=2).encode()),
               "brief": ("brief.md", b"# The brief\n"),
               "brief_provenance": ("brief_provenance.json", b"{}"),
               "agent_env": ("agent_env.json", b'{"container": {}, "exec": {"ANTHROPIC_AUTH_TOKEN": "grant:relay"}}'),
               "transcript": ("transcript/INDEX.json", b'{"files": [], "skipped": []}')}
#: The fixture build's source bundle.
BUNDLE = b"# v2 git bundle\n"
ROW = f"runs/{TASK}/index/{ROLLOUT}.json"
GRADING_ROW = f"runs/{TASK}/index/{ROLLOUT}+{GRADING}.json"
#: The columns of an index row, in order: an ``index.jsonl`` line's but
#: ``backend``; the failed stage and its fault; the scan's three; the arm, and the columns read
#: from the record.
ROW_COLUMNS = ["rollout_id", "grading_id", "mode", "created_at", "finished_at", "status", "score", "gate_passed",
               "cost_usd", "image_digests", "failed_stage", "fault", "scanned", "quarantined", "quarantined_count",
               "arm", "model_fingerprint", "accelerator", "agent_implementer_model"]
#: The rollout.json finalize wrote for ``_graded_rollout``, on which the viewer's build views are
#: tested (``apps/viewer/app/components/rollout/build-views.test.tsx``).
VIEWER_FIXTURE = Path(__file__).resolve().parents[2] / "apps/viewer/app/lib/fixtures/stage-rollout.json"
#: The grading.json finalize wrote for the same rollout, on which the viewer's deploy tab is tested
#: (``apps/viewer/app/components/rollout/build-files.test.tsx``).
VIEWER_GRADING_FIXTURE = Path(__file__).resolve().parents[2] / "apps/viewer/app/lib/fixtures/stage-grading.json"


def _as_agent_build(plan: Plan, arm: ArmRecord = ARM, arm_change: str | None = None) -> Plan:
    """*plan* with its stage ``agent_build`` an agent build of *arm*, with *arm_change* declared,
    granted the arm's route, as the planner makes it; the stage it replaces was planned as a
    stand-in with no parameters."""
    params = {**PARAMS, "arm": arm.model_dump(), "arm_change": arm_change}
    stages = [s.model_copy(update={"kind": AgentBuildStage.kind, "params": params,
                                   "grants": {"ANTHROPIC_AUTH_TOKEN": "relay"}}) if s.name == "agent_build" else s
              for s in plan.stages]
    return plan.model_copy(update={"stages": stages})


def _pin(store: PosixStore, plan: Plan, stage: str) -> dict[str, str]:
    """*stage*'s manifest's key and digest, as a stage that read it pins it."""
    key = plan.layout.manifest(stage)
    return {key: hashlib.sha256(store.read_bytes(key)).hexdigest()}


def _graded_rollout(tmp_path, checkout, arm: ArmRecord = ARM,
                    arm_change: str | None = None) -> tuple[Plan, PosixStore, dict[str, str]]:
    """The fixture rollout of *arm* with an agent_build, its source_scan and its package before its
    drive, served, verified and scored, its finalize naming the build and the package stages; with
    the key of each of the build's outputs by name."""
    drive = make_plan(tmp_path, checkout, "fin_goodput.task")
    build = drive.stage("score").model_copy(update={"name": "agent_build", "after": [], "inputs": {}})
    plan = _as_agent_build(drive.model_copy(update={"stages": [build, *drive.stages]}), arm, arm_change)
    plan = with_params(plan, "finalize", build_stage="agent_build", package_stage="package")
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    published = publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace),
                                 bundle=BUNDLE, build_extra=BUILD_FILES)
    run_and_publish(store, plan, "score", ScoreStage)
    return plan, store, {name: ref.key for name, ref in published["agent_build"].outputs.items()}


def _build_only(tmp_path, checkout, *, invalid: bool = False) -> tuple[Plan, PosixStore]:
    """A rollout of an agent_build, its source_scan, its package and a finalize that grades nothing,
    as ``hls run --build-only`` plans it, with the build, the scan and the package published, each
    scan clean; the package did not build when *invalid*."""
    task = {"module": "fin_goodput.task", "overrides": {}}
    finalize = {"task": task, "kind": "rollout", "score_stage": None, "build_stage": "agent_build",
                "package_stage": "package"}
    decls = [StageDecl("agent_build", "stand_in", CPU),
             StageDecl("source_scan", "stand_in", CPU, after=("agent_build",)),
             StageDecl("package", "stand_in", CPU, after=("source_scan",)),
             StageDecl("finalize", "finalize", CPU, finalize, after=("agent_build", "package"))]
    plan = build_plan(decls=decls, task=TASK, pipeline="rollout", layout=InstanceLayout.rollout(TASK, ROLLOUT),
                      settings=settings(tmp_path),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-fin", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0)
    plan = _as_agent_build(plan)
    store = PosixStore(plan.store_root)
    built = {name: write(store, plan, "agent_build", file, data) for name, (file, data) in BUILD_FILES.items()}
    built["source_bundle"] = write(store, plan, "agent_build", "source.bundle", BUNDLE)
    publish(store, plan, "agent_build", built)
    publish(store, plan, "source_scan", {
        "package_scan": write(store, plan, "source_scan", PACKAGE_SCAN_FILE, TREE_SCAN),
        "source": write(store, plan, "source_scan", "source.json", b"{}")}, inputs=_pin(store, plan, "agent_build"))
    outputs = {"package": write(store, plan, "package", "package.json", package_result(plan, invalid=invalid))}
    if not invalid:
        outputs["submission"] = write(store, plan, "package", "submission.sif", SIF)
        outputs["package_scan"] = write(store, plan, "package", PACKAGE_SCAN_FILE, INSTALL_SCAN)
    publish(store, plan, "package", outputs,
            inputs=_pin(store, plan, "agent_build") | _pin(store, plan, "source_scan"))
    return plan, store


def _echoing_grading(tmp_path, checkout, value: str, *, built: bool = True, **finalize) -> tuple[Plan, PosixStore]:
    """A grading of the fixture rollout's package whose server echoes *value* in its first reply,
    its finalize given *finalize*; the package pins the agent build of ``ARM`` that it packaged,
    or with *built* False none, as a package warmer's."""
    pinned = InstanceLayout.rollout(TASK, ROLLOUT).manifest("package")
    plan = make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading",
                     artifact={"kind": "package", "manifest": pinned})
    store = PosixStore(plan.store_root)
    pins = publish_rollout_package(store, plan, build=BUILD_FILES["build"][1] if built else None)
    plan = with_inputs(plan, "deploy_drive", pins)
    trace = echoed_trace(value)
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace), drive_inputs=pins)
    run_and_publish(store, plan, "score", ScoreStage)
    return with_params(plan, "finalize", **finalize), store


def _finalize(plan: Plan, store: PosixStore) -> dict[str, str]:
    return FinalizeStage().run(context(store, plan, "finalize", secrets=REGISTRY))


def test_a_rollouts_record_holds_its_build_and_package(tmp_path, checkout):
    plan, store, built = _graded_rollout(tmp_path, checkout)
    outputs = _finalize(plan, store)
    assert sorted(outputs) == ["grading", "rollout"]
    rollout = RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"]))
    assert rollout.status == "ok"
    build = rollout.build
    assert isinstance(build, RolloutBuild)
    # build.json and package.json as their stages wrote them, and every build output by its key.
    assert build.build == BUILD and build.outputs == built
    assert sorted(built) == ["agent_env", "brief", "brief_provenance", "build", "source_bundle", "transcript"]
    assert build.package == json.loads(package_result(plan, invalid=False))
    assert (build.started_at, build.ended_at, build.usage) == (300.0, 400.0, BUILD["extraction"]["usage"])
    # The tree digest of the folder the task uploaded into the agent's workspace.
    assert build.upload_sha256 == "d" * 64
    # The arm's effort is the provenance's.
    assert rollout.config.reasoning_effort == "max"
    conditions = rollout.conditions
    assert conditions.arm.record == ARM and conditions.arm.arm_change is None
    assert (conditions.budget.soft_sec, conditions.budget.hard_sec, conditions.budget.max_turns) == (None, 1100, None)
    # The fixture module declares no ARTIFACT, so whether these are its own budgets is not known.
    assert conditions.budget.overridden is None
    # The build ran under the plan's rule for a sandbox's end on its memory, the setting's default.
    assert conditions.grade_after_sandbox_oom is True
    grading = GradingRecord.model_validate_json(store.read_bytes(outputs["grading"]))
    assert grading.conditions == conditions
    row = json.loads(store.read_bytes(ROW))
    assert (row["arm"], row["agent_implementer_model"], row["cost_usd"]) == ("k3", "k3", 12.5)


@pytest.mark.parametrize("graded", [True, False], ids=["graded", "zero"])
def test_a_rollouts_conditions_state_the_sandbox_oom_rule_its_plan_records(tmp_path, checkout, graded):
    """Both records state the rule of the plan's
    ``run.grade_after_sandbox_oom`` that the rollout's build ran under."""
    plan, store, _ = _graded_rollout(tmp_path, checkout)
    run = plan.settings.run.model_copy(update={"grade_after_sandbox_oom": graded})
    plan = plan.model_copy(update={"settings": plan.settings.model_copy(update={"run": run})})
    outputs = _finalize(plan, store)
    rollout = RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"]))
    grading = GradingRecord.model_validate_json(store.read_bytes(outputs["grading"]))
    assert rollout.conditions.grade_after_sandbox_oom is grading.conditions.grade_after_sandbox_oom is graded


def test_the_viewers_fixture_rollout_is_the_one_finalize_records(tmp_path, checkout):
    """The viewer's build views are tested on ``VIEWER_FIXTURE``: the rollout record finalize
    writes for this fixture rollout, whole, but for when it was finished, which is the clock's. A
    change to what finalize records fails here until the fixture is that record again."""
    plan, store, _ = _graded_rollout(tmp_path, checkout)
    written = json.loads(store.read_bytes(_finalize(plan, store)["rollout"]))
    fixture = json.loads(VIEWER_FIXTURE.read_bytes())
    assert isinstance(written["finished_at"], float) and isinstance(fixture["finished_at"], float)
    assert fixture == {**written, "finished_at": fixture["finished_at"]}
    assert isinstance(RolloutRecord.model_validate(fixture).build, RolloutBuild)


def test_the_viewers_fixture_grading_is_the_one_finalize_records(tmp_path, checkout):
    """The viewer's deploy tab is tested on ``VIEWER_GRADING_FIXTURE``: the grading of this fixture
    rollout as finalize writes it, whose episode names each role log by its file name and its
    store key. A change to what finalize records fails here until the fixture is that record
    again."""
    plan, store, _ = _graded_rollout(tmp_path, checkout)
    written = json.loads(store.read_bytes(_finalize(plan, store)["grading"]))
    assert json.loads(VIEWER_GRADING_FIXTURE.read_bytes()) == written
    assert sorted(written["episode"]["container_logs"]) == [
        "artifact.err", "artifact.out", "driver.err", "driver.out", "score.err", "score.out", "verify.err"]


def test_a_rollouts_conditions_name_its_arms_declared_change(tmp_path, checkout):
    """A rerun whose arm differs from the task's earlier builds of it declares why, and the record
    says so."""
    plan, store, _ = _graded_rollout(tmp_path, checkout, arm_change="the route moved to the new relay")
    outputs = _finalize(plan, store)
    conditions = RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"])).conditions
    assert (conditions.arm.record, conditions.arm.arm_change) == (ARM, "the route moved to the new relay")


def test_a_rebuilt_packages_record_is_in_the_grading(tmp_path, checkout):
    """A grading that packaged the rollout's source again (``--artifact.rebuild``) records the package
    it built, read from its manifest; a grading of a package it did not build records none."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading")
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace))
    run_and_publish(store, plan, "score", ScoreStage)
    rebuilt = with_params(plan, "finalize", package_stage="package")
    grading = GradingRecord.model_validate_json(store.read_bytes(_finalize(rebuilt, store)["grading"]))
    assert grading.package == json.loads(package_result(plan, invalid=False))
    store.delete(GRADING_ROW)
    adopted = FinalizeStage().run(context(store, plan, "finalize", attempt=2, secrets=REGISTRY))
    assert GradingRecord.model_validate_json(store.read_bytes(adopted["grading"])).package is None


def test_a_build_record_is_read_against_its_manifest(tmp_path, checkout):
    """The record holds what the stage published: a ``build.json`` changed since is not it."""
    plan, store, built = _graded_rollout(tmp_path, checkout)
    store.write_bytes(built["build"], json.dumps({**BUILD, "hard_sec": 1}).encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "does not match its manifest" in e.value.fault.reason


def _scored(tmp_path, checkout, provenance: bytes | None) -> tuple[Plan, PosixStore]:
    """The fixture instance served, verified and scored, its score stage publishing *provenance* as
    its ``score-provenance.json``, or no such output for None."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task")
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace))
    outputs = ScoreStage().run(context(store, plan, "score"))
    if provenance is None:
        del outputs["score_provenance"]
    else:
        store.write_bytes(outputs["score_provenance"], provenance)
    publish(store, plan, "score", outputs)
    return plan, store


@pytest.mark.parametrize("provenance, reason", [
    (b"{not json", "is not a score provenance"),
    (b'{"baseline": {"log_credit": "high"}}', "is not a score provenance"),
    (None, "has no output score-provenance.json"),
], ids=["unparseable", "not-a-provenance", "missing"])
def test_a_score_whose_provenance_cannot_be_read_is_refused(tmp_path, checkout, provenance, reason):
    """The credit a record names is the one the score role recorded: a score stage that published
    none, or one that does not parse, gives the record nothing to name, and finalize refuses."""
    plan, store = _scored(tmp_path, checkout, provenance)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert reason in e.value.fault.reason and not store.exists(ROW)


def test_a_build_only_status_is_read_against_its_manifest(tmp_path, checkout):
    """A build-only rollout's status is its package's, as published: a ``package.json`` changed since
    is not it, whether or not the rollout records a build."""
    plan, store = _build_only(tmp_path, checkout, invalid=True)
    plan = with_params(plan, "finalize", build_stage=None)
    record = plan.layout.attempt_file("package", 1, "outputs/package.json")
    store.write_bytes(record, package_result(plan, invalid=False))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "does not match its manifest" in e.value.fault.reason and not store.exists(ROW)


def test_a_build_only_rollout_is_recorded_built(tmp_path, checkout):
    plan, store = _build_only(tmp_path, checkout)
    ctx = context(store, plan, "finalize", secrets=REGISTRY)
    outputs = FinalizeStage().run(ctx)
    # It grades nothing, so it writes no grading record; a grading of it writes its own.
    assert outputs == {"rollout": ctx.output_key("rollout.json")}
    rollout = RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"]))
    assert (rollout.status, rollout.build.build, rollout.build.package["status"]) == ("built", BUILD, "built")
    assert (rollout.config.artifact, rollout.config.split) == ({}, "test")
    assert rollout.conditions.attempts["package"].counted == 1
    assert rollout.conditions.credit is None
    row = json.loads(store.read_bytes(ROW))
    assert list(row) == ROW_COLUMNS
    assert (row["status"], row["score"], row["gate_passed"], row["mode"]) == ("built", None, None, "original")
    assert (row["scanned"], row["arm"], row["cost_usd"], row["failed_stage"]) == (True, "k3", 12.5, None)


def test_a_build_only_rollout_whose_package_is_invalid_is_recorded_invalid_artifact(tmp_path, checkout):
    plan, store = _build_only(tmp_path, checkout, invalid=True)
    outputs = _finalize(plan, store)
    assert RolloutRecord.model_validate_json(store.read_bytes(outputs["rollout"])).status == "invalid_artifact"
    row = json.loads(store.read_bytes(ROW))
    assert (row["status"], row["score"]) == ("invalid_artifact", None)


def test_a_build_only_rollouts_bundle_its_source_scan_read_is_not_read_again(tmp_path, checkout):
    """Finalize does not read the build's bundle (``AgentBuildStage.unread_files``): its source
    scan's record of the bundle's objects covers it, and that record is clean, so a key in the
    bundle's bytes is the object scan's to find, not finalize's, and nothing is quarantined; the
    rollout is recorded."""
    plan, store = _build_only(tmp_path, checkout)
    write(store, plan, "agent_build", "source.bundle", BUNDLE + RELAY.encode())
    outputs = _finalize(plan, store)
    assert "quarantine" not in outputs
    assert json.loads(store.read_bytes(ROW))["quarantined"] == []


def test_the_index_row_names_the_arm_and_its_model_and_has_no_backend(tmp_path, checkout):
    arm = ARM.model_copy(update={"model": "k3-0905"})
    plan, store, _ = _graded_rollout(tmp_path, checkout, arm)
    _finalize(plan, store)
    row = json.loads(store.read_bytes(ROW))
    assert list(row) == ROW_COLUMNS and "backend" not in row
    # The arm's name and its own model, which the record's artifact does not name.
    assert (row["arm"], row["agent_implementer_model"], row["accelerator"]) == ("k3", "k3-0905", "H200")


def test_a_grading_counts_the_building_arms_route_secrets_contestant_facing():
    plan = make_plan(Path("/x"), Path("/c"), "fin_goodput.task", kind="grading")
    assert contestant_facing(plan, REGISTRY) == frozenset()
    # The arm's route secret is the contestant's; the harness's own hf_token stays ours.
    assert contestant_facing(plan, REGISTRY, ARM) == {"relay"}
    # A route secret the site's registry does not hold is still the contestant's: a record
    # written while the registry held it names it.
    elsewhere = ARM.model_copy(update={"route_table": Route(credential="kimi_key", auth="bearer")})
    assert contestant_facing(plan, REGISTRY, elsewhere) == {"kimi_key"}


def test_a_building_arms_route_secret_is_the_contestants_under_every_name_that_holds_its_value():
    plan = make_plan(Path("/x"), Path("/c"), "fin_goodput.task", kind="grading")
    # A scan finds the bytes, and cannot tell which of two names with one value put them there.
    shared = {**REGISTRY, "relay_alias": RELAY}
    assert contestant_facing(plan, shared, ARM) == {"relay", "relay_alias"}
    # Another value stays the harness's, whatever names hold it.
    assert contestant_facing(plan, {**shared, "hub_alias": SECRET}, ARM) == {"relay", "relay_alias"}
    assert contestant_facing(plan, shared) == frozenset()


def test_a_facing_secret_echoed_by_the_server_is_quarantined_and_the_grading_scored(tmp_path, checkout):
    """The agent held its route's key and may have baked it into its server, so in a
    grading of its package the key is the contestant's: the drive's file holding it is kept out of
    any export, and the grading is scored."""
    plan, store = _echoing_grading(tmp_path, checkout, RELAY, built_arm=ARM.model_dump(mode="json"))
    outputs = _finalize(plan, store)
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert sorted(outputs) == ["grading", "quarantine"]
    assert json.loads(store.read_bytes(outputs["quarantine"])) == {"files": {chunk: ["relay"]}}
    row = json.loads(store.read_bytes(GRADING_ROW))
    score = json.loads(store.read_bytes(plan.layout.attempt_file("score", 1, "outputs/score.json")))
    assert (row["status"], row["score"], row["quarantined"]) == ("ok", score["value"], [chunk])
    assert score["value"] > 0
    # The grading names the arm that built what it graded; its cost is the rollout's, not its own.
    assert (row["arm"], row["agent_implementer_model"], row["cost_usd"]) == ("k3", "k3", None)
    grading = GradingRecord.model_validate_json(store.read_bytes(outputs["grading"]))
    assert grading.conditions.arm.record == ARM and grading.conditions.budget is None
    # It ran no build, so it states no rule for one.
    assert grading.conditions.grade_after_sandbox_oom is None
    assert grading.provenance.reasoning_effort == "max"


def test_a_route_key_in_a_grading_that_names_no_arm_is_our_leak(tmp_path, checkout):
    """A grading of a package no agent built names no arm, and nothing in its own plan hands the key
    to a contestant, so the key in its files is the harness's leak: the instance is refused and
    nothing is published."""
    plan, store = _echoing_grading(tmp_path, checkout, RELAY, built=False)
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert e.value.fault.reason == f"the harness leaked secret values into published files: {chunk} (relay)"
    assert not store.exists(GRADING_ROW)


def test_a_secret_no_contestant_could_hold_still_refuses(tmp_path, checkout):
    plan, store = _echoing_grading(tmp_path, checkout, SECRET, built_arm=ARM.model_dump(mode="json"))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert "(hf_token)" in e.value.fault.reason and not store.exists(GRADING_ROW)


def test_a_grading_whose_arm_is_not_the_builds_is_refused(tmp_path, checkout):
    """The arm a grading names decides which secrets are the contestant's, so it must be the arm the
    build recorded: another arm's would count the wrong secrets facing."""
    other = ARM.model_copy(update={"name": "arm-a"})
    plan, store = _echoing_grading(tmp_path, checkout, RELAY, built_arm=other.model_dump(mode="json"))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "arm-a" in e.value.fault.reason and "k3" in e.value.fault.reason
    assert not store.exists(GRADING_ROW)


def test_a_grading_that_names_an_arm_its_package_has_no_build_of_is_refused(tmp_path, checkout):
    """A package whose inputs reach no agent build (a package warmer's) was built by no arm: a grading
    of it that names one would count that arm's route secrets the contestant's."""
    plan, store = _echoing_grading(tmp_path, checkout, RELAY, built=False, built_arm=ARM.model_dump(mode="json"))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "no agent build" in e.value.fault.reason and not store.exists(GRADING_ROW)


def test_a_grading_of_an_image_may_name_the_arm_that_built_it(tmp_path, checkout):
    """An image graded by its digest carries no build to check: the arm the operator
    names (``--built-arm``) is recorded, and its route secrets are the contestant's."""
    image = {"kind": "image", "image": "published-0123", "entrypoint": ["serve"]}
    plan = make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading", artifact=image)
    store = PosixStore(plan.store_root)
    trace = echoed_trace(RELAY)
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace))
    run_and_publish(store, plan, "score", ScoreStage)
    plan = with_params(plan, "finalize", built_arm=ARM.model_dump(mode="json"))
    outputs = _finalize(plan, store)
    chunk = plan.layout.attempt_file("deploy_drive", 1, "outputs/trace/chunk-000000.jsonl.gz")
    assert json.loads(store.read_bytes(outputs["quarantine"])) == {"files": {chunk: ["relay"]}}
    assert json.loads(store.read_bytes(GRADING_ROW))["arm"] == "k3"


def test_a_same_named_arm_on_another_route_is_refused(tmp_path, checkout):
    """Arms are compared whole: one of the build's name on another route holds other secrets."""
    rerouted = ARM.model_copy(update={"route_table": Route(credential="hf_token", auth="bearer")})
    plan, store = _echoing_grading(tmp_path, checkout, RELAY, built_arm=rerouted.model_dump(mode="json"))
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "is not the arm" in e.value.fault.reason and not store.exists(GRADING_ROW)


def test_a_rebuild_whose_arm_is_not_its_builds_is_refused(tmp_path, checkout):
    """A grading that packaged a build's bundle again (its own package stage) is held to the arm of
    the build its package pinned, as a grading of a pinned package is."""
    plan = make_plan(tmp_path, checkout, "fin_goodput.task", kind="grading")
    store = PosixStore(plan.store_root)
    trace = drive_trace()
    publish_instance(store, plan, outcome="served", trace=trace, verdict=passing_verdict(trace), bundle=BUNDLE,
                     build_extra={"build": BUILD_FILES["build"]})
    run_and_publish(store, plan, "score", ScoreStage)
    other = ARM.model_copy(update={"name": "arm-a"})
    with pytest.raises(StageFault) as e:
        _finalize(with_params(plan, "finalize", built_arm=other.model_dump(mode="json")), store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "built_arm arm-a is not the arm k3" in e.value.fault.reason


def test_a_grading_whose_build_record_changed_is_refused(tmp_path, checkout):
    """The arm a grading is checked against is the one the build published: a ``build.json`` changed
    since is not it."""
    plan, store = _echoing_grading(tmp_path, checkout, RELAY, built_arm=ARM.model_dump(mode="json"))
    other = {**BUILD, "arm": ARM.model_copy(update={"name": "arm-a"}).model_dump(mode="json")}
    build = InstanceLayout.rollout(TASK, ROLLOUT).attempt_file("agent_build", 1, "outputs/build.json")
    store.write_bytes(build, json.dumps(other).encode())
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert f"{build} does not match its manifest" in e.value.fault.reason


def test_a_grading_of_an_agents_package_must_name_its_arm(tmp_path, checkout):
    """A package whose inputs reach an agent build was built by that build's arm; a grading of it
    that names no arm would judge the arm's own route key as the harness's leak."""
    plan, store = _echoing_grading(tmp_path, checkout, RELAY)
    with pytest.raises(StageFault) as e:
        _finalize(plan, store)
    assert (e.value.fault.fault_class, e.value.fault.retryable) == ("infra", False)
    assert "built_arm" in e.value.fault.reason and not store.exists(GRADING_ROW)


@pytest.mark.parametrize("params, match", [
    ({"kind": "grading", "eval_id": GRADING, "score_stage": None}, "a grading grades"),
    ({"kind": "grading", "eval_id": GRADING, "build_stage": "agent_build"}, "only a rollout records a build"),
    ({"kind": "rollout", "score_stage": None}, "names its package stage"),
    ({"kind": "rollout", "built_arm": ARM.model_dump(mode="json")}, "built_arm is a grading's"),
], ids=["grading-without-score", "grading-with-build", "nothing-recorded", "rollout-with-built-arm"])
def test_finalize_params_that_cannot_be_recorded_are_refused(params, match):
    with pytest.raises(ValidationError, match=match):
        FinalizeParams.model_validate({"task": {"module": "fin_goodput.task"}, **params})


# What the command says when the instance ends (``commands._finish``), read from the records the
# stages' manifests pin.


def _replace_output(store: PosixStore, plan: Plan, stage: str, name: str, data: bytes, *, republish: bool) -> str:
    """Overwrite output *name* of *plan*'s published *stage* with *data* and return its key; with
    *republish*, publish the stage's manifest again with the new bytes' digest, so the output is
    the one its manifest records."""
    manifest, _ = read_manifest(store, plan.layout, stage)
    ref = manifest.outputs[name]
    store.write_bytes(ref.key, data)
    if republish:
        store.path(plan.layout.manifest(stage)).unlink()
        outputs = {**manifest.outputs, name: output_ref(store, ref.key, sha256_bytes(data))}
        publish_manifest(store, plan.layout, manifest.model_copy(update={"outputs": outputs}))
    return ref.key


def _succeeded(plan: Plan) -> Plan:
    return plan.model_copy(update={"outcome": Outcome(status="succeeded", reason="every stage published", at=1.0)})


def test_the_end_summary_says_the_score_its_manifest_recorded(tmp_path, checkout):
    plan, store, _ = _graded_rollout(tmp_path, checkout)
    ref = read_manifest(store, plan.layout, "score")[0].outputs["score"]
    value = json.loads(store.read_bytes(ref.key))["value"]
    lines: list[str] = []
    assert commands._finish(_succeeded(plan), lines.append) == 0
    assert lines == [f"{plan.instance}: succeeded: every stage published", f"{plan.instance}: score {value} ({ref.key})"]


def test_the_end_summary_refuses_a_score_rewritten_after_it_was_published(tmp_path, checkout):
    plan, store, _ = _graded_rollout(tmp_path, checkout)
    score = json.loads(store.read_bytes(read_manifest(store, plan.layout, "score")[0].outputs["score"].key))
    key = _replace_output(store, plan, "score", "score", json.dumps({**score, "value": 99.0}).encode(),
                          republish=False)
    lines: list[str] = []
    with pytest.raises(ValueError, match=f"{key} does not match its manifest"):
        commands._finish(_succeeded(plan), lines.append)
    assert lines == [f"{plan.instance}: succeeded: every stage published"]


@pytest.mark.parametrize("data, why", [(b'{"score": 1.0}', "missing value"), (b"not json", "json_invalid")],
                         ids=["no-value", "not-json"])
def test_the_end_summary_refuses_a_score_of_another_shape_saying_where(tmp_path, checkout, data, why):
    plan, store, _ = _graded_rollout(tmp_path, checkout)
    key = _replace_output(store, plan, "score", "score", data, republish=True)
    with pytest.raises(ValueError, match=f"{key} is not a Score: {why}"):
        commands._finish(_succeeded(plan), [].append)


@pytest.mark.parametrize("republish, data, match", [
    (False, None, "does not match its manifest"),
    (True, b'{"rollout_id": "r1"}', "is not a RolloutRecord: missing task_id"),
], ids=["rewritten", "another-shape"])
def test_the_end_summary_of_a_build_only_rollout_reads_the_record_its_manifest_recorded(tmp_path, checkout,
                                                                                       republish, data, match):
    plan, store = _build_only(tmp_path, checkout)
    run_and_publish(store, plan, "finalize", FinalizeStage, secrets=REGISTRY)
    ref = read_manifest(store, plan.layout, "finalize")[0].outputs["rollout"]
    lines: list[str] = []
    assert commands._finish(_succeeded(plan), lines.append) == 0
    assert lines[-1] == f"{plan.instance}: rollout built ({ref.key})"
    record = json.loads(store.read_bytes(ref.key))
    data = data if data is not None else json.dumps({**record, "status": "ok"}).encode()
    key = _replace_output(store, plan, "finalize", "rollout", data, republish=republish)
    with pytest.raises(ValueError, match=f"{key} {match}"):
        commands._finish(_succeeded(plan), [].append)
