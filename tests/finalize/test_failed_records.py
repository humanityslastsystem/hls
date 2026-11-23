"""A failed instance, one that a stage's contestant fault ended, ends with its records: a zero
score naming the stage and the fault, written by the orchestrator before the revision that ends
the instance, and nothing else. The instance runs on one machine as in
``tests.pipeline.test_end_to_end_local``: the real commands, orchestrator and runner, with the
failed stage raising the ``stage_fault`` of its roles' endings. agent_build and verify
are stand-ins (``stand_in_stage``), and the drive stage's ``run`` is replaced, so no container
starts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hls.core.poly import dump_spec
from hls.obs.conditions import BudgetNote
from hls.obs.record import GradingRecord, RolloutBuild, RolloutRecord
from hls.pipeline import commands
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import read_events
from hls.pipeline.export import check_instance
from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.pipelines import KNOWN_STAGE_NAMES
from hls.pipeline.plan import AllocationStatus, Plan, PlanStore, RecordsRefused, Toolchain
from hls.pipeline.planner import PlanError, StageDecl, build_plan
from hls.pipeline.stage import StageFault, stage_fault
from hls.stages import finalize
from hls.stages.deploy_drive import DeployDriveStage
from hls.stages.agent_build import AgentBuildStage, sandbox_oom_zero
from hls.stages.finalize import (FAILED_REASON, failed_key, failed_records, failed_task, finalize_params,
                                 record_failed)
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord, JobState
from hls.substrate.faults import RoleOutcome
from hls.substrate.secrets import Scrubber, load_secrets
from hls.taskref import TaskRef, load_task
from hls.util import json_safe
from tests.finalize import stand_in_stage
from tests.finalize.published import CPU, DRIVE_PARAMS, DRIVE_SHAPE, GRADING, ROLLOUT, TASK, naming_published
from tests.finalize.test_agent_build_scan import PARAMS
from tests.pipeline.plans import settings
from tests.pipeline.test_end_to_end_local import LocalSlurm, _drive
from tests.pipeline.test_orchestrator import _events

SECRET = "hf-token-0123456789abcdef"
#: What the runner and the records make of a text that holds the secret.
SCRUB = Scrubber({"hf_token": SECRET})
#: The stages of the instance, in order.
STAGES = ["agent_build", "deploy_drive", "verify", "score", "finalize"]
#: The ending of the roles of each failed stage, a contestant fault each. The agent's detail
#: holds a secret's value, which no record may hold.
ENDINGS = {
    "agent_build": RoleOutcome("agent", "contestant", 137, "disk_limit", detail=f"wrote {SECRET} past 10 GiB"),
    "deploy_drive": RoleOutcome("artifact", "contestant", 137, "memory_limit", 1, "footprint 9 GiB past 8 GiB"),
    "verify": RoleOutcome("artifact", "contestant", 137, "memory_limit", 1, "footprint 9 GiB past 8 GiB"),
}


class TickingSlurm(LocalSlurm):
    """Runs every job whose turn it is each time the orchestrator asks about jobs, so a command
    that drives an instance to its end needs no one to tick."""

    def states(self, job_ids):
        self.tick()
        return super().states(job_ids)


def _secrets_file(tmp_path: Path) -> Path:
    path = tmp_path / "site" / "secrets.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f'[secrets]\nhf_token = "{SECRET}"\n')
    path.chmod(0o600)
    return path


def _plan(tmp_path: Path, checkout: Path, *, kind: str = "rollout", checkpoint: str = "m2",
          checked: bool = True, build_stage: str | None = None, finalized: bool = True) -> Plan:
    """A rollout, or a grading of the fixture rollout, that builds, drives, verifies, scores and
    finalizes, with *checkpoint* in the task's overrides, its finalize recording the build of
    *build_stage*. *checked* plans with the
    secrets file, as the commands do; without it the planner cannot see what the plan's text holds.
    It names the other stages the fixture publishes too (``naming_published``). Not *finalized*,
    the plan is its build stage alone, as a probe or preflight-only build is, which records no
    score and has no finalize to read them."""
    path = _secrets_file(tmp_path)
    layout = InstanceLayout.rollout(TASK, ROLLOUT)
    if kind == "grading":
        layout = layout.grading(GRADING)
    task = {"module": "fin_goodput.task", "overrides": {"task.model.base_checkpoint": checkpoint}}
    drive = {"task": task, "artifact": {"kind": "package"}, **DRIVE_PARAMS}
    finalize = {"task": task, "kind": kind, "eval_id": GRADING if kind == "grading" else "terminal",
                "build_stage": build_stage}
    decls = [StageDecl("agent_build", "stand_in", CPU),
             StageDecl("deploy_drive", "deploy_drive", DRIVE_SHAPE, drive, after=("agent_build",)),
             StageDecl("verify", "stand_in", CPU, after=("deploy_drive",)),
             StageDecl("score", "score", CPU, {"task": task}, after=("deploy_drive", "verify")),
             StageDecl("finalize", "finalize", CPU, finalize, after=("deploy_drive", "verify", "score"))]
    if not finalized:
        decls = decls[:1]
    plan = build_plan(decls=decls, task=TASK, pipeline="evaluate", layout=layout,
                      settings=settings(tmp_path, site={"secrets_path": str(path)}),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-fin", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0,
                      secrets=load_secrets(path) if checked else None)
    return naming_published(plan) if finalized else plan


def _plant(stage: str, monkeypatch, *, before: list[FaultRecord] = ()) -> FaultRecord:
    """Make *stage* fail with the contestant fault of its roles' ending, after an attempt for
    each of *before* failed with it; return the contestant fault as the runner records it,
    scrubbed. The drive stage publishes nothing when it does not fail."""
    fault = stage_fault([ENDINGS[stage]])
    assert fault is not None and fault.fault_class == "contestant"
    monkeypatch.setitem(stand_in_stage.PLANTED, stage, [*before, fault])

    def run(self, ctx):
        if ctx.stage.name == stage:
            raise StageFault(fault)
        return {}

    monkeypatch.setattr(DeployDriveStage, "run", run)
    return fault.model_copy(update={"reason": SCRUB.scrub(fault.reason),
                                    "evidence": {k: SCRUB.scrub(v) for k, v in fault.evidence.items()}})


def _create(plan: Plan, slurm: LocalSlurm, on_failed=failed_records) -> Orchestrator:
    return Orchestrator.create(plan, slurm, holder=Holder(host="login", pid=1, job_id=None, purpose="orchestrator"),
                               sleep=lambda s: None, on_failed=on_failed)


def _left(layout: InstanceLayout) -> str:
    """What the commands say of an instance whose records are left."""
    return f"{layout.root}: its records are left for hls resume; its failed_records events say why"


def _strict(data: bytes):
    return json.loads(data, parse_constant=lambda c: pytest.fail(f"a record holds {c}"))


def _files(store: PosixStore, prefix: str) -> list[str]:
    root = store.path(prefix)
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()) if root.is_dir() else []


@pytest.mark.parametrize("stage", ["agent_build", "deploy_drive", "verify"])
def test_a_failed_stage_ends_with_a_scored_zero_naming_it_and_nothing_else(tmp_path, checkout, monkeypatch, stage):
    fault = _plant(stage, monkeypatch)
    plan = _plan(tmp_path, checkout)
    slurm = TickingSlurm()
    lines: list[str] = []
    # A failed instance is a scored result: the command says so and exits 0.
    assert commands.start(plan, {}, scheduler=slurm, echo=lines.append) == 0
    store, layout = PosixStore(plan.store_root), plan.layout
    assert lines[-1] == f"{plan.instance}: score 0.0, zero for {FAILED_REASON} ({failed_key(layout, 'score.json')})"
    plan = PlanStore(store, layout).latest()
    # The runner's record of the fault: the stage's, scrubbed, with the job's evidence beside it.
    ended = plan.outcome.fault
    assert (plan.outcome.status, ended.fault_class, ended.reason) == ("failed", "contestant", fault.reason)
    # Scored, so never retried or relaunched: one job, which failed.
    assert len(slurm.submits()) == 1 and slurm.jobs["100"].state is JobState.FAILED

    score = _strict(store.read_bytes(failed_key(layout, "score.json")))
    assert (score["value"], score["gate_passed"], score["zero_reasons"]) == (0.0, False, [FAILED_REASON])
    assert score["breakdown"] == {FAILED_REASON: {"stage": stage, "reason": fault.reason, "evidence": ended.evidence}}

    grading = GradingRecord.model_validate_json(store.read_bytes(failed_key(layout, "grading.json")))
    ep = grading.episode
    assert (ep.eval_id, ep.triggered_by, ep.checkpoint, ep.trace_path) == ("terminal", "harness", "", "")
    assert ep.score == score and ep.verdict_summary["gate_passed"] is False
    attempt = json.loads(store.read_bytes(layout.attempt_file(stage, 1, "status.json")))
    assert (ep.started_at, ep.finished_at) == (attempt["started_at"], attempt["ended_at"])
    assert (grading.source_rollout, grading.mode) == (ROLLOUT, "original")
    assert grading.provenance.rollout_overrides == {"task.model.base_checkpoint": "m2"}
    rollout = RolloutRecord.model_validate_json(store.read_bytes(failed_key(layout, "rollout.json")))
    assert (rollout.rollout_id, rollout.status, rollout.config) == (ROLLOUT, "ok", grading.provenance)

    row = _strict(store.read_bytes(f"runs/{TASK}/index/{ROLLOUT}.json"))
    assert (row["status"], row["score"], row["gate_passed"]) == ("ok", 0.0, False)
    assert (row["failed_stage"], row["fault"]) == (stage, fault.reason)
    # No scan read a failed instance's files: the export scans it before anything leaves the store.
    assert (row["scanned"], row["quarantined"], row["quarantined_count"]) == (False, [], 0)

    # Nothing else: no stage from the failed one on published, no later stage claimed an
    # attempt, and the records are the only files outside the stages.
    published = [s for s in STAGES if read_manifest(store, layout, s) is not None]
    assert published == STAGES[:STAGES.index(stage)]
    for later in STAGES[STAGES.index(stage) + 1:]:
        assert layout.attempts(store, later) == []
    assert _files(store, f"{layout.root}/failed") == ["grading.json", "rollout.json", "score.json"]
    assert _files(store, f"runs/{TASK}/index") == [f"{ROLLOUT}.json"]
    for key in [*(failed_key(layout, f) for f in ("grading.json", "rollout.json", "score.json")),
                f"runs/{TASK}/index/{ROLLOUT}.json"]:
        assert SECRET.encode() not in store.read_bytes(key)


def test_a_void_instance_gets_no_score_and_no_row(tmp_path, checkout, monkeypatch):
    # An infra fault that is not retried voids the instance: not the contestant's result, so it
    # is relaunched, and nothing may read as its score.
    infra = FaultRecord(fault_class="infra", reason="the image's digest is not the recorded one", retryable=False)
    monkeypatch.setitem(stand_in_stage.PLANTED, "agent_build", [infra])
    slurm = LocalSlurm()
    orch = _create(_plan(tmp_path, checkout), slurm)
    plan = _drive(orch, slurm)
    assert (plan.outcome.status, plan.outcome.fault.reason) == ("void", infra.reason)
    store, layout = PosixStore(plan.store_root), plan.layout
    assert _files(store, f"{layout.root}/failed") == [] and _files(store, f"runs/{TASK}/index") == []
    assert not [e for e in _events(orch) if e["kind"].startswith("failed_records")]


def test_the_records_are_written_before_the_revision_that_ends_the_instance(tmp_path, checkout, monkeypatch):
    fault = _plant("deploy_drive", monkeypatch)
    plan = _plan(tmp_path, checkout, kind="grading")
    seen = []

    def writer(store, failed):
        seen.append(PlanStore(store, failed.layout).latest().outcome)
        return failed_records(store, failed)

    slurm = LocalSlurm()
    orch = _create(plan, slurm, on_failed=writer)
    plan = _drive(orch, slurm)
    assert seen == [None] and plan.outcome.fault.reason == fault.reason
    store, layout = PosixStore(plan.store_root), plan.layout
    assert _files(store, f"{layout.root}/failed") == ["grading.json", "score.json"]
    grading = GradingRecord.model_validate_json(store.read_bytes(failed_key(layout, "grading.json")))
    assert (grading.episode.eval_id, grading.episode.triggered_by) == (GRADING, "regrade")
    assert (grading.source_rollout, grading.mode) == (ROLLOUT, "rebuild")
    row = _strict(store.read_bytes(f"runs/{TASK}/index/{ROLLOUT}+{GRADING}.json"))
    assert (row["grading_id"], row["failed_stage"], row["score"]) == (GRADING, "deploy_drive", 0.0)
    written = [e for e in _events(orch) if e["kind"] == "failed_records_written"]
    assert [sorted(e["records"]) for e in written] == [["grading", "index", "score"]]


def test_records_that_cannot_be_scrubbed_are_left_and_the_instance_ends_without_them(tmp_path, checkout,
                                                                                       monkeypatch):
    """The secrets file turns unreadable before the decision that ends the instance failed: the
    writer refuses, and the orchestrator ends the instance all the same, its outcome marked with
    the records left, and asks no more; resumed with the file usable, the records are written."""
    fault = _plant("agent_build", monkeypatch)
    plan = _plan(tmp_path, checkout)
    slurm = LocalSlurm()
    orch = _create(plan, slurm)
    orch.step()
    slurm.tick()
    secrets = plan.settings.site.secrets_path
    secrets.chmod(0o644)
    ended = _drive(orch, slurm)
    orch.close()
    store, layout = PosixStore(plan.store_root), plan.layout
    assert (ended.outcome.status, ended.outcome.records_left) == ("failed", True)
    assert PlanStore(store, layout).latest().outcome == ended.outcome
    assert _files(store, f"{layout.root}/failed") == [] and _files(store, f"runs/{TASK}/index") == []
    [left] = [e for e in _events(orch) if e["kind"].startswith("failed_records")]
    assert left["kind"] == "failed_records_left" and "group and others must have no access" in left["error"]
    assert SECRET not in left["error"]

    secrets.chmod(0o600)
    lines: list[str] = []
    code = commands.resume(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=lines.append, sleep=lambda s: None)
    assert code == 0 and lines[-2] == f"{layout.root}: failed: agent_build: {fault.reason}"
    assert _files(store, f"{layout.root}/failed") == ["grading.json", "rollout.json", "score.json"]
    assert _strict(store.read_bytes(f"runs/{TASK}/index/{ROLLOUT}.json"))["failed_stage"] == "agent_build"
    assert not PlanStore(store, layout).latest().outcome.records_left
    # Resumed once it has ended, the instance is reported with its zero, and scored.
    again: list[str] = []
    code = commands.resume(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=again.append, sleep=lambda s: None)
    assert code == 0 and again == [f"{layout.root} already ended: failed: agent_build: {fault.reason}",
                                   f"{layout.root}: score 0.0, zero for {FAILED_REASON} "
                                   f"({failed_key(layout, 'score.json')})"]


def test_a_cancel_that_cannot_read_the_secrets_file_leaves_the_records_and_resume_writes_them(tmp_path, checkout,
                                                                                             monkeypatch):
    """The cancel ends the failed instance with its records left, which its outcome and ``hls
    status`` say; once the file is usable again, resuming the instance writes them, clears the
    mark and reports its zero."""
    fault = _plant("agent_build", monkeypatch)
    plan = _plan(tmp_path, checkout)
    slurm = LocalSlurm()
    orch = _create(plan, slurm)
    orch.step()
    slurm.tick()
    orch.close()
    secrets = plan.settings.site.secrets_path
    secrets.chmod(0o644)
    store, layout = PosixStore(plan.store_root), plan.layout
    lines: list[str] = []
    code = commands.cancel(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=lines.append, sleep=lambda s: None)
    outcome = PlanStore(store, layout).latest().outcome
    assert code == 1 and (outcome.status, outcome.records_left) == ("failed", True)
    assert lines[-2:] == [f"{layout.root}: failed: agent_build: {fault.reason}", _left(layout)]
    assert _files(store, f"{layout.root}/failed") == []
    status = commands.status_text(layout.root, config=tmp_path / "unused.toml", store=plan.store_root,
                                  scheduler=slurm)
    assert "records: left for hls resume, which writes them; the failed_records events say why" in status

    secrets.chmod(0o600)
    again: list[str] = []
    code = commands.resume(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=again.append, sleep=lambda s: None)
    assert code == 0 and _files(store, f"{layout.root}/failed") == ["grading.json", "rollout.json", "score.json"]
    assert again[-2:] == [f"{layout.root}: failed: agent_build: {fault.reason}",
                          f"{layout.root}: score 0.0, zero for {FAILED_REASON} ({failed_key(layout, 'score.json')})"]
    assert not PlanStore(store, layout).latest().outcome.records_left


def test_resume_of_a_marked_instance_whose_writer_refuses_says_its_records_are_left(tmp_path, checkout,
                                                                                      monkeypatch):
    """Resumed with its records left and a writer that refuses them, the instance is reported as
    the cancel reported it, with no score and exit status 1, and the mark stays for the next."""
    fault = _plant("agent_build", monkeypatch)
    plan = _plan(tmp_path, checkout)
    slurm = LocalSlurm()
    orch = _create(plan, slurm)
    orch.step()
    slurm.tick()
    orch.close()
    secrets = plan.settings.site.secrets_path
    secrets.chmod(0o644)
    commands.cancel(plan.layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                    echo=lambda s: None, sleep=lambda s: None)
    store, layout = PosixStore(plan.store_root), plan.layout
    assert PlanStore(store, layout).latest().outcome.records_left
    secrets.chmod(0o600)
    asked: list[int] = []

    def refuses(store, plan):
        asked.append(plan.revision)
        raise RecordsRefused("the build changed since it published")

    monkeypatch.setattr(commands, "failed_records", refuses)
    lines: list[str] = []
    code = commands.resume(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=lines.append, sleep=lambda s: None)
    assert code == 1 and len(asked) == 1 and lines[0].startswith(f"attached to {layout.root}")
    assert lines[1:] == [f"{layout.root}: failed: agent_build: {fault.reason}", _left(layout)]
    assert PlanStore(store, layout).latest().outcome.records_left
    assert _files(store, f"{layout.root}/failed") == []


def test_a_run_whose_writer_refuses_ends_failed_and_says_its_records_are_left(tmp_path, checkout, monkeypatch):
    """The command that drives the instance from its start reports the records left as the cancel
    and resume do, and exits 1: the instance's zero is not recorded."""
    fault = _plant("agent_build", monkeypatch)
    plan = _plan(tmp_path, checkout)

    def refuses(store, failed):
        raise RecordsRefused("the build changed since it published")

    monkeypatch.setattr(commands, "failed_records", refuses)
    lines: list[str] = []
    assert commands.start(plan, {}, scheduler=TickingSlurm(), echo=lines.append) == 1
    layout = plan.layout
    assert lines[-2:] == [f"{layout.root}: failed: agent_build: {fault.reason}", _left(layout)]
    assert PlanStore(PosixStore(plan.store_root), layout).latest().outcome.records_left


def test_a_failed_build_without_a_finalize_stage_has_no_records_to_leave(tmp_path, checkout, monkeypatch):
    """A probe or preflight-only build records no score: a cancel that cannot read the secrets
    file ends it failed with nothing left, and it is then settled, with nothing for a resume."""
    _plant("agent_build", monkeypatch)
    plan = _plan(tmp_path, checkout, finalized=False)
    slurm = LocalSlurm()
    orch = _create(plan, slurm)
    orch.step()
    slurm.tick()
    orch.close()
    plan.settings.site.secrets_path.chmod(0o644)
    store, layout = PosixStore(plan.store_root), plan.layout
    lines: list[str] = []
    code = commands.cancel(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=lines.append, sleep=lambda s: None)
    outcome = PlanStore(store, layout).latest().outcome
    assert code == 1 and (outcome.status, outcome.records_left) == ("failed", False)
    assert outcome.reason.startswith("agent_build: ") and lines[-1] == f"{layout.root}: failed: {outcome.reason}"
    status = commands.status_text(layout.root, config=tmp_path / "unused.toml", store=plan.store_root,
                                  scheduler=slurm)
    assert "records:" not in status
    events = [e["kind"] for path in store.path(layout.root).rglob("events.jsonl") for e in read_events(path)]
    assert not [kind for kind in events if kind.startswith("failed_records")]
    again: list[str] = []
    code = commands.resume(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=again.append, sleep=lambda s: None)
    assert code == 1 and again == [f"{layout.root} already ended: failed: {outcome.reason}"]


def test_an_export_refuses_an_instance_whose_records_are_left(tmp_path, checkout, monkeypatch):
    """Its zero and its records are not in the store yet, so an export would publish the instance
    without them; once a resume has written them, the export takes it."""
    _plant("agent_build", monkeypatch)
    plan = _plan(tmp_path, checkout)
    slurm = LocalSlurm()
    orch = _create(plan, slurm)
    orch.step()
    slurm.tick()
    orch.close()
    secrets = plan.settings.site.secrets_path
    secrets.chmod(0o644)
    commands.cancel(plan.layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                    echo=lambda s: None, sleep=lambda s: None)
    store = PosixStore(plan.store_root)
    with pytest.raises(ValueError, match="its records are left; hls resume writes them first"):
        check_instance(store, plan.layout.root)
    secrets.chmod(0o600)
    assert commands.resume(plan.layout.root, config=tmp_path / "unused.toml", store=plan.store_root,
                           scheduler=slurm, echo=lambda s: None, sleep=lambda s: None) == 0
    assert not check_instance(store, plan.layout.root).outcome.records_left


def test_resume_of_a_failed_instance_with_its_records_needs_no_secrets_file(tmp_path, checkout, monkeypatch):
    """Its records are there and its outcome leaves none, so resume reports it and its zero and
    reads no secret, whatever became of the file since."""
    fault = _plant("agent_build", monkeypatch)
    plan = _plan(tmp_path, checkout)
    slurm = TickingSlurm()
    assert commands.start(plan, {}, scheduler=slurm, echo=lambda s: None) == 0
    layout = plan.layout
    plan.settings.site.secrets_path.chmod(0o644)
    lines: list[str] = []
    code = commands.resume(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=lines.append, sleep=lambda s: None)
    assert code == 0 and lines == [f"{layout.root} already ended: failed: agent_build: {fault.reason}",
                                   f"{layout.root}: score 0.0, zero for {FAILED_REASON} "
                                   f"({failed_key(layout, 'score.json')})"]


def test_resume_of_a_void_grading_and_the_take_over_of_a_cancelled_one_write_no_records(tmp_path, checkout,
                                                                                        monkeypatch):
    """Only a failed instance has records to write: a void grading is reported, and a cancel that
    takes a grading over ends it cancelled, neither with a record or a writer's refusal."""
    (tmp_path / "void").mkdir()
    (tmp_path / "cancelled").mkdir()
    plan = _plan(tmp_path / "void", checkout, kind="grading")
    refusing = LocalSlurm()
    refusing.submit_fault = "refuse"
    orch = _create(plan, refusing)
    orch.step()
    orch.close()
    store, layout = PosixStore(plan.store_root), plan.layout
    assert PlanStore(store, layout).latest().outcome.status == "void"
    lines: list[str] = []
    code = commands.resume(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=refusing,
                           echo=lines.append, sleep=lambda s: None)
    assert code == 1 and lines[0].startswith(f"{layout.root} already ended: void: sbatch refused allocation 0")
    assert _files(store, f"{layout.root}/failed") == []

    plan = _plan(tmp_path / "cancelled", checkout, kind="grading")
    slurm = LocalSlurm()
    orch = _create(plan, slurm)
    orch.step()
    orch.close()
    store, layout = PosixStore(plan.store_root), plan.layout
    code = commands.cancel(layout.root, config=tmp_path / "unused.toml", store=plan.store_root, scheduler=slurm,
                           echo=lines.append, sleep=lambda s: None)
    assert code == 0 and PlanStore(store, layout).latest().outcome.status == "cancelled"
    events = [e["kind"] for path in store.path(layout.root).rglob("events.jsonl") for e in read_events(path)]
    assert not [kind for kind in events if kind.startswith("failed_records")]
    assert _files(store, f"{layout.root}/failed") == []


def test_the_records_name_the_contestant_fault_and_its_attempt_after_an_infra_retry(tmp_path, checkout,
                                                                                     monkeypatch):
    infra = FaultRecord(fault_class="infra", reason="the node lost its scratch", retryable=True)
    fault = _plant("agent_build", monkeypatch, before=[infra])
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout), slurm), slurm)
    assert [a.failure.fault.reason for a in plan.allocations if a.failure] == [infra.reason, fault.reason]
    store, layout = PosixStore(plan.store_root), plan.layout
    named = _strict(store.read_bytes(failed_key(layout, "score.json")))["breakdown"][FAILED_REASON]
    assert (named["stage"], named["reason"]) == ("agent_build", fault.reason)
    second = json.loads(store.read_bytes(layout.attempt_file("agent_build", 2, "status.json")))
    episode = GradingRecord.model_validate_json(store.read_bytes(failed_key(layout, "grading.json"))).episode
    assert (episode.started_at, episode.finished_at) == (second["started_at"], second["ended_at"])


def test_every_record_is_scrubbed_whatever_the_plan_holds(tmp_path, checkout, monkeypatch):
    _plant("deploy_drive", monkeypatch)
    slurm = LocalSlurm()
    # A plan made without the planner's check of its text holds the value in its overrides.
    plan = _plan(tmp_path, checkout, checkpoint=SECRET, checked=False)
    plan = _drive(_create(plan, slurm, on_failed=None), slurm)
    # A fault reason the runner did not scrub, as one recorded by other code could be.
    leaked = plan.outcome.fault.model_copy(update={"reason": f"the server printed {SECRET}"})
    allocations = [a.model_copy(update={"failure": a.failure.model_copy(update={"fault": leaked})})
                   if a.failure is not None else a for a in plan.allocations]
    plan = plan.model_copy(update={"allocations": allocations,
                                   "outcome": plan.outcome.model_copy(update={"fault": leaked})})
    store = PosixStore(plan.store_root)
    keys = record_failed(store, plan, finalize_params(plan), scrubber=Scrubber({"hf_token": SECRET}))
    assert sorted(keys) == ["grading", "index", "rollout", "score"]
    for key in keys.values():
        data = store.read_bytes(key)
        assert SECRET.encode() not in data and b"<redacted:hf_token>" in data


def test_records_are_create_only_and_a_plan_without_finalize_gets_none(tmp_path, checkout, monkeypatch):
    _plant("verify", monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout), slurm, on_failed=None), slurm)
    store, layout = PosixStore(plan.store_root), plan.layout
    assert _files(store, f"{layout.root}/failed") == []
    earlier = failed_key(layout, "score.json")
    store.write_bytes(earlier, b"an earlier orchestrator's record\n")
    keys = record_failed(store, plan, finalize_params(plan), scrubber=Scrubber({}))
    assert keys["score"] == earlier and store.read_bytes(earlier) == b"an earlier orchestrator's record\n"
    again = {key: store.read_bytes(key) for key in keys.values()}
    assert failed_records(store, plan) == keys
    assert {key: store.read_bytes(key) for key in keys.values()} == again

    stages = [s for s in plan.stages if s.kind != "finalize"]
    assert failed_records(store, plan.model_copy(update={"stages": stages})) == {}


def test_the_planner_refuses_a_plan_that_would_hold_a_secrets_value(tmp_path, checkout):
    with pytest.raises(PlanError) as e:
        _plan(tmp_path, checkout, checkpoint=f"hub/{SECRET}")
    message = str(e.value)
    assert SECRET not in message
    assert "hf_token" in message and "stages[1].params.task.overrides.task.model.base_checkpoint" in message


def test_a_refusal_is_recorded_scrubbed(tmp_path, checkout, monkeypatch):
    _plant("agent_build", monkeypatch)

    def broken(store, plan, params, *, scrubber):
        raise KeyError(f"no field named {SECRET}")

    monkeypatch.setattr(finalize, "record_failed", broken)
    slurm = LocalSlurm()
    orch = _create(_plan(tmp_path, checkout), slurm)
    plan = _drive(orch, slurm)
    orch.close()
    assert (plan.outcome.status, plan.outcome.records_left) == ("failed", True)
    [left] = [e for e in _events(orch) if e["kind"] == "failed_records_left"]
    assert left["error"] == "RecordsRefused: KeyError: 'no field named <redacted:hf_token>'"


def test_an_unexpected_writer_error_is_recorded_by_its_type_alone(tmp_path, checkout, monkeypatch):
    _plant("agent_build", monkeypatch)

    def writer(store, plan):
        raise ValueError(f"input_value={SECRET}")

    slurm = LocalSlurm()
    orch = _create(_plan(tmp_path, checkout), slurm, on_failed=writer)
    plan = _drive(orch, slurm)
    orch.close()
    assert (plan.outcome.status, plan.outcome.records_left) == ("failed", True)
    [left] = [e for e in _events(orch) if e["kind"] == "failed_records_left"]
    assert left["error"] == "ValueError"


def test_a_cancel_after_a_failed_pass_keeps_the_contestant_failure(tmp_path, checkout, monkeypatch):
    _plant("deploy_drive", monkeypatch)
    plan = _plan(tmp_path, checkout)
    store = PosixStore(plan.store_root)
    calls = []

    def writer(s, failed):
        calls.append(failed.revision)
        return failed_records(s, failed)

    slurm = LocalSlurm()
    orch = _create(plan, slurm, on_failed=writer)
    write = orch.plans.write

    def stalls_once(revision, handle):
        if revision.outcome is not None and calls == [revision.revision]:
            # The shared filesystem stalls after the records: the pass fails before its
            # revision, and the operator cancels the instance before the next pass.
            store.write_bytes(revision.layout.cancel_request, b"{}")
            raise OSError("stale file handle")
        return write(revision, handle)

    monkeypatch.setattr(orch.plans, "write", stalls_once)
    for _ in range(60):
        try:
            plan, done, acted = orch.step()
        except OSError:
            continue
        if done:
            break
        if not acted:
            slurm.tick()
    assert plan.outcome.status == "failed" and plan.outcome.fault.fault_class == "contestant"
    assert not plan.outcome.records_left and len(calls) == 2
    assert _files(store, f"{plan.layout.root}/failed") == ["grading.json", "rollout.json", "score.json"]


#: A secret made of a name's characters, which an override key can hold, and one of digits.
WORDY, DIGITS = "hfwordysecretvalue0123", "4242424242424242"


def _plan_with(tmp_path: Path, checkout: Path, *, overrides: dict, drive: dict | None = None) -> Plan:
    path = tmp_path / "site" / "secrets.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(f'[secrets]\nwordy = "{WORDY}"\ndigits = "{DIGITS}"\n')
    path.chmod(0o600)
    task = {"module": "fin_goodput.task", "overrides": overrides}
    params = {"task": task, "artifact": {"kind": "package"}, **DRIVE_PARAMS, **(drive or {})}
    return build_plan(decls=[StageDecl("deploy_drive", "deploy_drive", DRIVE_SHAPE, params)], task=TASK,
                      pipeline="evaluate", layout=InstanceLayout.rollout(TASK, ROLLOUT),
                      settings=settings(tmp_path, site={"secrets_path": str(path)}),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-fin", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0,
                      secrets=load_secrets(path))


def test_the_planner_refuses_a_value_in_a_key_and_names_the_key_scrubbed(tmp_path, checkout):
    with pytest.raises(PlanError) as e:
        _plan_with(tmp_path, checkout, overrides={f"task.model.{WORDY}": "x"})
    message = str(e.value)
    assert WORDY not in message and "wordy" in message
    assert "stages[0].params.task.overrides.task.model.<redacted:wordy> holds wordy" in message


def test_the_planner_refuses_a_value_in_a_number(tmp_path, checkout):
    with pytest.raises(PlanError) as e:
        _plan_with(tmp_path, checkout, overrides={}, drive={"request_timeout_sec": float(DIGITS)})
    assert "stages[0].params.request_timeout_sec holds digits" in str(e.value) and DIGITS not in str(e.value)


def test_a_failed_rollouts_records_carry_the_task_spec(tmp_path, checkout, monkeypatch):
    _plant("deploy_drive", monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout), slurm), slurm)
    store, layout = PosixStore(plan.store_root), plan.layout
    rollout = RolloutRecord.model_validate_json(store.read_bytes(failed_key(layout, "rollout.json")))
    task = load_task(TaskRef(module="fin_goodput.task", overrides={"task.model.base_checkpoint": "m2"}),
                     calibration=True, credit=True)
    assert rollout.config.spec == json.loads(json.dumps(json_safe(dump_spec(task))))
    score = _strict(store.read_bytes(failed_key(layout, "score.json")))
    assert "spec_error" not in score["breakdown"][FAILED_REASON]
    row = _strict(store.read_bytes(f"runs/{TASK}/index/{ROLLOUT}.json"))
    assert (row["model_fingerprint"], row["accelerator"]) == (task.model.fingerprint, "H200")
    # The failed stage's attempt is in the conditions; no manifest of it counted.
    conditions = rollout.conditions
    assert conditions.attempts["deploy_drive"].counted is None
    assert [a.state for a in conditions.attempts["deploy_drive"].attempts] == ["failed"]
    assert conditions.attempts["agent_build"].counted == 1
    grading = GradingRecord.model_validate_json(store.read_bytes(failed_key(layout, "grading.json")))
    assert grading.conditions == conditions
    # Its drive never published and nothing scored, so no hardware is compared.
    assert (conditions.hardware, conditions.baseline_hardware, conditions.hardware_warning) == (None, None, None)


def test_a_failed_instances_records_state_the_hardware_each_job_ran_on(tmp_path, checkout, monkeypatch):
    """A failed instance's records state the hardware its jobs ran on, as
    each job's runner's grant check recorded it in its status, as finalize's records would."""
    _plant("deploy_drive", monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout), slurm), slurm)
    store, layout = PosixStore(plan.store_root), plan.layout
    want = {}
    for a in plan.allocations:
        for sub in a.submissions:
            status = AllocationStatus.model_validate_json(store.read_bytes(layout.allocation_status(a.index, sub.n)))
            want[status.job_id] = status.grant.hardware()
    assert want
    for name, model in (("rollout.json", RolloutRecord), ("grading.json", GradingRecord)):
        conditions = model.model_validate_json(store.read_bytes(failed_key(layout, name))).conditions
        assert {job: hw for note in conditions.allocations for job, hw in note.hardware.items()} == want


def test_a_failed_record_whose_task_does_not_load_names_the_error(tmp_path, checkout, monkeypatch):
    _plant("deploy_drive", monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout), slurm, on_failed=None), slurm)

    def broken(ref, files, **kwargs):
        raise RuntimeError(f"the module reads {SECRET} and fails")

    monkeypatch.setattr(finalize, "load_pinned", broken)
    store = PosixStore(plan.store_root)
    keys = record_failed(store, plan, finalize_params(plan), scrubber=SCRUB)
    # The zero stands; it says why the spec is missing by the error's type alone, never its message.
    breakdown = _strict(store.read_bytes(keys["score"]))["breakdown"][FAILED_REASON]
    assert (breakdown["spec_error"], breakdown["stage"]) == ("RuntimeError", "deploy_drive")
    assert RolloutRecord.model_validate_json(store.read_bytes(keys["rollout"])).config.spec == {}


#: What the fixture's agent build wrote as ``build.json`` before the drive failed: read back as
#: written.
BUILT = {"started_at": 3.0, "ended_at": 4.0, "extraction": {"usage": {"n_input_tokens": 9, "cost_usd": 2.5}}}


def _publishing_build(monkeypatch) -> None:
    """Make the stand-in agent build publish ``BUILT`` as its ``build.json``."""
    stand_in = stand_in_stage.StandInStage.run

    def run(self, ctx):
        if ctx.stage.name == "agent_build":
            return {"build": ctx.write_output("build.json", json.dumps(BUILT).encode())}
        return stand_in(self, ctx)

    monkeypatch.setattr(stand_in_stage.StandInStage, "run", run)


def test_a_failed_rollouts_records_hold_the_build_its_agent_published(tmp_path, checkout, monkeypatch):
    _plant("deploy_drive", monkeypatch)
    _publishing_build(monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout, build_stage="agent_build"), slurm), slurm)
    store, layout = PosixStore(plan.store_root), plan.layout
    build = RolloutRecord.model_validate_json(store.read_bytes(failed_key(layout, "rollout.json"))).build
    key = layout.attempt_file("agent_build", 1, "outputs/build.json")
    # A build.json that names no uploaded folder records none.
    assert build == RolloutBuild(build=BUILT, outputs={"build": key}, package=None, started_at=3.0, ended_at=4.0,
                                 usage=BUILT["extraction"]["usage"], upload_sha256=None)
    assert _strict(store.read_bytes(f"runs/{TASK}/index/{ROLLOUT}.json"))["cost_usd"] == 2.5


def test_a_failed_rollouts_build_changed_since_it_published_is_refused(tmp_path, checkout, monkeypatch):
    """The failed records hold what the build published: a ``build.json`` changed since is not it,
    and the records are refused, scrubbed, leaving them to a resume."""
    _plant("deploy_drive", monkeypatch)
    _publishing_build(monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout, build_stage="agent_build"), slurm, on_failed=None), slurm)
    store = PosixStore(plan.store_root)
    key = plan.layout.attempt_file("agent_build", 1, "outputs/build.json")
    store.write_bytes(key, json.dumps({**BUILT, "note": SECRET}).encode())
    with pytest.raises(RecordsRefused) as e:
        failed_records(store, plan)
    assert f"{key} does not match its manifest" in str(e.value) and SECRET not in str(e.value)
    assert _files(store, f"{plan.layout.root}/failed") == []


def test_a_failed_build_whose_task_does_not_load_records_its_budgets_as_unknown(tmp_path, checkout, monkeypatch):
    """Whether a build's budgets are its task's own is read from the task's module; when the task did
    not load, the module is not imported again, and the records say the answer is unknown."""
    _plant("deploy_drive", monkeypatch)
    _publishing_build(monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout, build_stage="agent_build"), slurm, on_failed=None), slurm)
    plan = plan.model_copy(update={"stages": [s.model_copy(update={"kind": AgentBuildStage.kind, "params": PARAMS})
                                              if s.name == "agent_build" else s for s in plan.stages]})

    def broken(ref, files, **kwargs):
        raise RuntimeError("the module does not load")

    def reimported(params, task_module):
        raise SyntaxError("the module was imported again")

    monkeypatch.setattr(finalize, "load_pinned", broken)
    monkeypatch.setattr(finalize, "budget_note", reimported)
    store = PosixStore(plan.store_root)
    keys = record_failed(store, plan, finalize_params(plan), scrubber=SCRUB)
    budget = RolloutRecord.model_validate_json(store.read_bytes(keys["rollout"])).conditions.budget
    assert budget == BudgetNote(soft_sec=None, hard_sec=1100, max_turns=None, overridden=None)


def test_a_build_the_sandbox_oom_rule_zeroed_has_failed_records_that_name_the_rule(tmp_path, checkout,
                                                                                   monkeypatch):
    """A build whose sandbox ended on its memory under a plan whose
    ``run.grade_after_sandbox_oom`` is false fails with the fault the stage declares
    (``sandbox_oom_zero``), and the instance's failed records name the rule: the zero's breakdown
    carries the fault's reason and evidence, and the harness-conditions table states the plan's
    rule."""
    oom = FaultRecord(fault_class="contestant", retryable=False, evidence={"roles": "sandbox"},
                      reason="build sandbox sandbox ran out of memory while the harness stayed within its reserve")
    zero = sandbox_oom_zero(oom)
    monkeypatch.setitem(stand_in_stage.PLANTED, "agent_build", [zero])
    slurm = LocalSlurm()
    plan = _drive(_create(_plan(tmp_path, checkout, build_stage="agent_build"), slurm, on_failed=None), slurm)
    run = plan.settings.run.model_copy(update={"grade_after_sandbox_oom": False})
    plan = plan.model_copy(update={
        "settings": plan.settings.model_copy(update={"run": run}),
        "stages": [s.model_copy(update={"kind": AgentBuildStage.kind, "params": PARAMS}) if s.name == "agent_build"
                   else s for s in plan.stages]})
    store = PosixStore(plan.store_root)
    keys = record_failed(store, plan, finalize_params(plan), scrubber=SCRUB)
    named = _strict(store.read_bytes(keys["score"]))["breakdown"][FAILED_REASON]
    assert (named["stage"], named["reason"]) == ("agent_build", zero.reason)
    assert named["evidence"]["grade_after_sandbox_oom"] == "false"
    assert "run.grade_after_sandbox_oom is false" in named["reason"]
    rollout = RolloutRecord.model_validate_json(store.read_bytes(keys["rollout"]))
    assert rollout.conditions.grade_after_sandbox_oom is False and rollout.build is None


def test_a_failed_task_loads_from_the_plans_checkout_off_the_import_path(tmp_path, checkout, monkeypatch):
    """An orchestrator that took the instance over may run where the plan's checkout is not on the
    import path: the task still loads from that checkout, which leaves the path as it found it."""
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "fin_elsewhere").mkdir(parents=True)
    (elsewhere / "fin_elsewhere" / "__init__.py").write_text("")
    (elsewhere / "fin_elsewhere" / "task.py").write_text("from tests.finalize import fixture\n\n"
                                                          "TASK = fixture.goodput_task()\n")
    monkeypatch.delitem(sys.modules, "fin_elsewhere", raising=False)
    monkeypatch.delitem(sys.modules, "fin_elsewhere.task", raising=False)
    plan = _plan(tmp_path, checkout)
    plan = plan.model_copy(update={"code": plan.code.model_copy(update={"checkout": elsewhere})})
    params = finalize_params(plan).model_copy(update={"task": TaskRef(module="fin_elsewhere.task")})
    before = list(sys.path)
    task, error = failed_task(plan, params)
    assert (error, task is not None) == (None, True) and sys.path == before
    sys.modules.pop("fin_elsewhere.task", None)
    sys.modules.pop("fin_elsewhere", None)


def _build_only_plan(tmp_path: Path, checkout: Path) -> Plan:
    """A build-only rollout: an agent build, its source scan, its package and a finalize that grades
    nothing."""
    path = _secrets_file(tmp_path)
    task = {"module": "fin_goodput.task", "overrides": {}}
    finalize = {"task": task, "kind": "rollout", "score_stage": None, "build_stage": "agent_build",
                "package_stage": "package"}
    decls = [StageDecl("agent_build", "stand_in", CPU),
             StageDecl("source_scan", "stand_in", CPU, after=("agent_build",)),
             StageDecl("package", "stand_in", CPU, after=("source_scan",)),
             StageDecl("finalize", "finalize", CPU, finalize, after=("agent_build", "package"))]
    return build_plan(decls=decls, task=TASK, pipeline="rollout", layout=InstanceLayout.rollout(TASK, ROLLOUT),
                      settings=settings(tmp_path, site={"secrets_path": str(path)}),
                      code=CodeState(checkout=checkout, hls_commit="a" * 40, harbor_commit="b" * 40,
                                     dirty_diff_sha256=None),
                      toolchain=Toolchain(python=Path(sys.executable), path="/usr/bin:/bin", home=tmp_path),
                      job_name="hls-fin", known_stage_names=KNOWN_STAGE_NAMES, known_tasks=(), now=1000.0,
                      secrets=load_secrets(path))


def test_a_build_only_failure_writes_no_grading_record(tmp_path, checkout, monkeypatch):
    fault = _plant("agent_build", monkeypatch)
    slurm = LocalSlurm()
    plan = _drive(_create(_build_only_plan(tmp_path, checkout), slurm), slurm)
    assert (plan.outcome.status, plan.outcome.fault.reason) == ("failed", fault.reason)
    store, layout = PosixStore(plan.store_root), plan.layout
    # The rollout's zero and its record, and no grading: a build-only rollout grades nothing.
    assert _files(store, f"{layout.root}/failed") == ["rollout.json", "score.json"]
    row = _strict(store.read_bytes(f"runs/{TASK}/index/{ROLLOUT}.json"))
    assert (row["status"], row["score"], row["failed_stage"], row["arm"]) == ("ok", 0.0, "agent_build", None)
    rollout = RolloutRecord.model_validate_json(store.read_bytes(failed_key(layout, "rollout.json")))
    # The agent build published nothing, so there is no build to record.
    assert rollout.build is None and rollout.config.artifact == {}
