from __future__ import annotations

import logging
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import hls.substrate.runner as runner_mod
import hls.substrate.slurm.substrate as slurm_substrate
from hls.pipeline.checkout import CodeState
from hls.pipeline.events import EventLog, read_events
from hls.pipeline.pipelines import noop_one, noop_two
from hls.pipeline.plan import AllocationStatus, AttemptStatus, Outcome, Plan, PlanStore, Submission, submission_tag
from hls.pipeline.planner import StageDecl
from hls.pipeline.reconcile import Observation, decide
from hls.pipeline.stage import STAGES, StageFault, stage_fault
from hls.spec.resources import GrantedResources, ResourceShape
from hls.store.base import NotFound
from hls.store.digest import sha256_bytes
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder, instance_lease
from hls.store.manifest import read_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import FaultRecord, JobInfo, JobState, Launcher, NodeSize, RoleFailed, RoleHandle
from hls.substrate.cgroup import CGROUP_ROOT
from hls.substrate.factory import substrate
from hls.substrate.grant import Measured
from hls.substrate.memory_watch import MemorySample
from hls.substrate.nproc import DEFAULT_NPROC_MAX, FLOOR_PER_EVALUATION, node_evaluations, nproc_max_problem
from hls.substrate.runner import (EXIT_FAILED, EXIT_PUBLISHED, EXIT_REFUSED, EXIT_TERMINATED, JobEnd, Runner,
                                  read_latest_plan)
from hls.substrate.secrets import Scrubber
from hls.substrate.slurm.commands import DEFAULT_TIMEOUT_SEC, SlurmCommands
from hls.substrate.slurm.scheduler import read_job_end
from hls.substrate.versions import VERSION_TIMEOUT_SEC, ToolVersions, measure_tools, tool_version
from tests.pipeline.fakes import NODE, LocalLauncher
from tests.pipeline.plans import CODE, VERSIONS, noop_plan
from tests.substrate.slurm_fakes import FakeRun, done


def _store(plan: Plan, *subs: Submission, k: int = 0, outcome: Outcome | None = None) -> Plan:
    """Write *plan* with *subs* on allocation *k* as the instance's next revision."""
    plan = plan.with_allocation(plan.allocation(k).model_copy(update={"submissions": list(subs)}))
    plan = plan.model_copy(update={"outcome": outcome})
    store = PosixStore(plan.store_root)
    handle = instance_lease(store, plan.layout, ttl_sec=30).try_acquire(Holder.here("test"))
    plans = PlanStore(store, plan.layout)
    latest = plans.latest()
    written = plans.create(plan, handle) if latest is None else plans.write(
        plan.model_copy(update={"revision": latest.revision}), handle)
    handle.release()
    return written


def _sub(plan: Plan, n: int = 1, *, k: int = 0, job_id: str | None = "100", status: str = "submitted") -> Submission:
    return Submission(n=n, tag=submission_tag(plan.instance, k, n), intended_at=999.0, after_job=None, status=status,
                      job_id=job_id, submitted_at=1000.0 if job_id else None)


def _grant(gpus: int = 2, cpus: int = 8, memory_gib: float = 12.0, swap: int | None = 0,
           problems: list[str] | None = None):
    granted = GrantedResources(gpus=gpus, gpu_kind="H200" if gpus else None, cpus=cpus, memory_gib=memory_gib,
                               disk_free_gib=1000.0)
    return lambda: Measured(granted=granted, gpu_uuids=[f"GPU-{i}" for i in range(gpus)], swap_max_bytes=swap,
                            problems=problems or [])


def _runner(plan: Plan, *, k: int = 0, n: int = 1, job_id: str = "100", job_end: JobEnd | None = None,
            measure=None, code: CodeState = CODE, launcher: Launcher | None = None, **overrides) -> Runner:
    """A runner whose job ends a day from now by the controller's record unless *job_end* says
    otherwise, on a node with ``VERSIONS``; *overrides* replace any constructor argument."""
    launcher = launcher or LocalLauncher(gpus=2)
    job_end = JobEnd(controller=time.time() + 86400, env=None) if job_end is None else job_end
    args = {"job_id": job_id, "node": "g1", "read_job_end": lambda: job_end,
            "measure_grant": measure or _grant(), "measure_tools": lambda: VERSIONS,
            "make_launcher": lambda gpus, uuids, scratch: launcher, "checkout_state": lambda path: code,
            "node_name": substrate(plan.settings.site).node_name, "runtime": substrate(plan.settings.site).runtime(),
            "images": substrate(plan.settings.site).images, **overrides}
    return Runner(plan, k, n, **args)


def _status(plan: Plan, k: int = 0, n: int = 1) -> AllocationStatus:
    store = PosixStore(plan.store_root)
    return AllocationStatus.model_validate_json(store.read_bytes(plan.layout.allocation_status(k, n)))


def _attempt(plan: Plan, stage: str, n: int = 1) -> AttemptStatus:
    store = PosixStore(plan.store_root)
    return AttemptStatus.model_validate_json(store.read_bytes(plan.layout.attempt_file(stage, n, "status.json")))


def _events(plan: Plan, k: int = 0, n: int = 1) -> list[dict]:
    """The events in the submission's log."""
    store = PosixStore(plan.store_root)
    return read_events(store.path(plan.layout.submission_file(k, n, "events.jsonl")))


def _kinds(plan: Plan, k: int = 0, n: int = 1) -> list[str]:
    """Kinds of the events in the submission's log."""
    return [e["kind"] for e in _events(plan, k, n)]


def _no_attempts(plan: Plan, stage: str) -> bool:
    return plan.layout.attempts(PosixStore(plan.store_root), stage) == []


def test_runs_every_stage_and_publishes_their_manifests(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert read_latest_plan(plan.plan_dir) == plan
    assert _runner(plan).run() == EXIT_PUBLISHED
    store = PosixStore(plan.store_root)
    a, a_digest = read_manifest(store, plan.layout, "noop_a")
    b, _ = read_manifest(store, plan.layout, "noop_b")
    assert b.inputs == {plan.layout.manifest("noop_a"): a_digest}
    assert set(b.outputs) == {"gpu0", "gpu1", "cpu", "shared", "summary"}
    assert b.resources.declared.gpus == 2 and b.resources.granted.gpus == 2
    assert b.times.submitted_at == 1000.0 and b.provenance.hls_commit == "a" * 40
    status = _status(plan)
    assert (status.state, status.memory_enforced_by, status.fault) == ("succeeded", "kernel", None)
    attempt = AttemptStatus.model_validate_json(store.read_bytes(plan.layout.attempt_file("noop_a", 1, "status.json")))
    assert attempt.state == "succeeded" and attempt.manifest_sha256 == a_digest


@pytest.mark.parametrize("change, why", [
    ({"status": "absent"}, "is absent"),
    ({"status": "cancelled"}, "is cancelled"),
])
def test_a_runner_that_is_not_the_current_submission_does_nothing(tmp_path, change, why):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan).model_copy(update=change))
    assert _runner(plan).run() == EXIT_REFUSED
    assert why in _status(plan).fault.reason
    assert PosixStore(plan.store_root).list_children(f"{plan.layout.stage('noop_a')}/attempts") == []


def test_a_runner_of_an_ended_instance_does_nothing(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan), outcome=Outcome(status="cancelled", reason="hls cancel", at=1.0))
    assert _runner(plan).run() == EXIT_REFUSED


def test_a_grant_that_differs_from_the_request_runs_nothing(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert _runner(plan, measure=_grant(gpus=1)).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.reason == "grant mismatch: gpus: requested 2, granted 1" and fault.retryable
    assert PosixStore(plan.store_root).list_children(plan.layout.stage("noop_a")) == []


def test_a_grant_fault_names_the_check_and_its_node(tmp_path, monkeypatch):
    # Without slurmd's name for the node, its host name up to the first dot, so a retry can
    # exclude it.
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert _runner(plan, measure=_grant(gpus=1), node="node01.example.com").run() == EXIT_FAILED
    assert _status(plan).fault.evidence == {"check": "grant", "node": "node01"}


def test_a_grant_fault_names_the_node_as_slurmd_does(tmp_path, monkeypatch):
    # slurmd sets the name Slurm knows the node by, which a host name need not start with.
    monkeypatch.setenv("SLURMD_NODENAME", "node02")
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert _runner(plan, measure=_grant(gpus=1), node="host-17.cluster.local").run() == EXIT_FAILED
    assert _status(plan).fault.evidence == {"check": "grant", "node": "node02"}
    assert _status(plan).node == "host-17.cluster.local"


def test_a_changed_checkout_is_not_retried(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    moved = CODE.model_copy(update={"hls_commit": "c" * 40})
    assert _runner(plan, code=moved).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert "checkout changed" in fault.reason and not fault.retryable


#: A CPU partition's node: no GPU and 192 CPUs.
CPU_NODE = NodeSize(gpus=0, cpus=192)


def _cpu_plan(tmp_path: Path, nproc_max: int | None) -> Plan:
    """The one-stage no-op plan with no GPU, its kind one that runs a contestant's code, under the
    site's process cap *nproc_max* (None for a site that sets none), with its first submission."""
    plan = noop_plan(tmp_path, decls=noop_one(gpus=0, gpu_kind="H200", sleep_sec=0, stage_time_sec=600),
                     site={"nproc_max": nproc_max})
    return _store(plan, _sub(plan))


@pytest.mark.parametrize("short", [1, 0], ids=["under", "at"])
def test_an_allocation_refuses_a_process_cap_under_its_nodes_floor_before_any_stage(tmp_path, short):
    # A plan checks the cap against one evaluation alone; a CPU node holds many of a small stage's
    # size, and a cap under their floor costs the allocation's start, never a stage's hours.
    evaluations = node_evaluations(CPU_NODE, gpus=0, cpus=2)
    nproc_max = evaluations * FLOOR_PER_EVALUATION - short
    plan = _cpu_plan(tmp_path, nproc_max)
    launcher = LocalLauncher(gpus=0, node=CPU_NODE)
    code = _runner(plan, measure=_grant(gpus=0, cpus=2, memory_gib=6.0), launcher=launcher).run()
    if not short:
        assert code == EXIT_PUBLISHED
        return
    assert code == EXIT_FAILED
    fault = _status(plan).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == (f"no contestant step of stage noop can run on this node: "
                            f"{nproc_max_problem(nproc_max, evaluations)}")
    assert _no_attempts(plan, "noop") and launcher.launched == []


@pytest.mark.parametrize("cpus, evaluations", [(64, 32), (264, 132)])
def test_an_allocation_of_a_site_that_sets_no_cap_starts_on_a_node_of_any_size(tmp_path, cpus, evaluations):
    # A 64-CPU node holds 32 evaluations of the two-CPU stage and a 264-CPU node 132. Where the
    # default's 32 floors are under the node's own floor, the cap is that floor, so neither node is
    # refused; a site that sets the default's value itself is refused the larger one.
    node = NodeSize(gpus=0, cpus=cpus)
    assert node_evaluations(node, gpus=0, cpus=2) == evaluations
    plan = _cpu_plan(tmp_path / "unset", None)
    assert plan.settings.site.nproc_max is None
    launcher = LocalLauncher(gpus=0, node=node)
    assert _runner(plan, measure=_grant(gpus=0, cpus=2, memory_gib=6.0), launcher=launcher).run() == EXIT_PUBLISHED
    [checked] = [e for e in _events(plan) if e["kind"] == "process_cap_checked"]
    assert (checked["node_cpus"], checked["nproc_max"], checked["problem"]) == (cpus, None, None)
    strict = _cpu_plan(tmp_path / "set", DEFAULT_NPROC_MAX)
    code = _runner(strict, measure=_grant(gpus=0, cpus=2, memory_gib=6.0),
                   launcher=LocalLauncher(gpus=0, node=node)).run()
    assert code == (EXIT_PUBLISHED if evaluations <= 32 else EXIT_FAILED)


def test_an_allocation_on_a_node_whose_size_cannot_be_told_runs_no_contestant_stage(tmp_path):
    plan = _cpu_plan(tmp_path, FLOOR_PER_EVALUATION)
    launcher = LocalLauncher(gpus=0, node=NodeSize(gpus=0, cpus=None))
    assert _runner(plan, measure=_grant(gpus=0, cpus=2, memory_gib=6.0), launcher=launcher).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert (fault.fault_class, fault.retryable) == ("infra", False)
    assert fault.reason == ("no contestant step of stage noop can run on this node: the node's CPU count is unknown, "
                            "so how many evaluations of this size it holds is unknown too")
    assert _no_attempts(plan, "noop") and launcher.launched == []


def test_a_stage_that_runs_no_contestant_code_is_not_held_to_the_process_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(STAGES["noop"], "runs_contestant_code", False)
    plan = _cpu_plan(tmp_path, FLOOR_PER_EVALUATION)
    launcher = LocalLauncher(gpus=0, node=NodeSize(gpus=0, cpus=None))
    assert _runner(plan, measure=_grant(gpus=0, cpus=2, memory_gib=6.0), launcher=launcher).run() == EXIT_PUBLISHED
    assert "process_cap_checked" not in _kinds(plan)


def test_an_allocation_checks_the_cap_for_the_stages_it_has_left_to_run(tmp_path):
    # noop_a, of one GPU, published on a node of two GPUs; on a node of eight the cap would not
    # hold its eight evaluations, but only noop_b, of two GPUs, is left, and the cap holds its four.
    decls = noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600)
    decls[1] = StageDecl(**{**decls[1].__dict__, "params": {**decls[1].params, "fail_attempts": 1}})
    k_b = node_evaluations(NODE, gpus=decls[1].shape.gpus, cpus=decls[1].shape.cpus)
    assert nproc_max_problem(k_b * FLOOR_PER_EVALUATION,
                             node_evaluations(NODE, gpus=decls[0].shape.gpus, cpus=decls[0].shape.cpus)) is not None
    plan = noop_plan(tmp_path, decls=decls, site={"nproc_max": k_b * FLOOR_PER_EVALUATION})
    plan = _store(plan, _sub(plan))
    assert _runner(plan, launcher=LocalLauncher(gpus=2, node=NodeSize(gpus=2, cpus=64))).run() == EXIT_FAILED
    assert "planted fault" in _status(plan).fault.reason
    plan = _store(plan, _sub(plan).model_copy(update={"status": "absent"}), _sub(plan, 2, job_id="101"))
    assert _runner(plan, n=2, job_id="101", launcher=LocalLauncher(gpus=2, node=NODE)).run() == EXIT_PUBLISHED


def test_a_rerun_skips_published_stages_and_counts_attempts(tmp_path):
    decls = noop_two(gpu_kind="H200", sleep_sec=0, stage_time_sec=600)
    decls[1] = StageDecl(**{**decls[1].__dict__, "params": {**decls[1].params, "fail_attempts": 1}})
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    assert _runner(plan).run() == EXIT_FAILED
    assert "planted fault" in _status(plan).fault.reason
    plan = _store(plan, _sub(plan).model_copy(update={"status": "absent"}), _sub(plan, 2, job_id="101"))
    assert _runner(plan, n=2, job_id="101").run() == EXIT_PUBLISHED
    store = PosixStore(plan.store_root)
    assert plan.layout.attempts(store, "noop_a") == [1]
    assert plan.layout.attempts(store, "noop_b") == [1, 2]
    assert read_manifest(store, plan.layout, "noop_b")[0].attempt == 2


def test_a_stage_that_cannot_finish_before_the_end_time_is_not_started(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan, job_end=JobEnd(controller=time.time() + 100, env=None))
    assert runner.run() == EXIT_FAILED
    assert f"noop_a needs {600 + int(runner_mod.END_MARGIN_SEC)}s (its time" in _status(plan).fault.reason
    assert PosixStore(plan.store_root).list_children(plan.layout.stage("noop_a")) == []


@pytest.mark.parametrize("swap, enforced_by", [(None, "runner"), (2**30, "runner"), (0, "kernel")])
def test_the_runner_enforces_memory_where_swap_is_not_off(tmp_path, swap, enforced_by):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan, measure=_grant(swap=swap))
    assert runner.run() == EXIT_PUBLISHED
    assert _status(plan).memory_enforced_by == enforced_by
    assert runner.watch.enforce is (enforced_by == "runner")


def test_the_runner_records_a_kill_of_one_process_apart_from_a_kill_of_a_step(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan, measure=_grant(swap=None))
    assert runner.run() == EXIT_PUBLISHED
    sample = MemorySample(at=5.0, current=3 * 2**30, swap_current=0, reclaimable=0, oom_kills=0)
    runner.watch.on_single_kill("100.4", sample, 2 * 2**30)
    runner.watch.on_breach("100.4", sample, 2 * 2**30)
    single, whole = [e for e in _events(plan) if e["kind"] in ("memory_process_killed", "memory_enforced")]
    assert (single["kind"], single["step"], single["footprint"], single["limit"]) == (
        "memory_process_killed", "100.4", 3 * 2**30, 2 * 2**30)
    assert (whole["kind"], whole["step"]) == ("memory_enforced", "100.4")


def test_sigterm_stops_the_roles_and_records_it(tmp_path):
    decls = noop_two(gpu_kind="H200", sleep_sec=30, stage_time_sec=600)
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    threading.Timer(1.0, runner.terminate).start()
    assert runner.run() == EXIT_TERMINATED
    store = PosixStore(plan.store_root)
    status = _status(plan)
    assert status.state == "terminated" and "terminated" in status.fault.reason
    attempt = AttemptStatus.model_validate_json(store.read_bytes(plan.layout.attempt_file("noop_a", 1, "status.json")))
    assert attempt.state == "failed" and read_manifest(store, plan.layout, "noop_a") is None


def test_each_beat_moves_the_heartbeat(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    now = [5000.0]
    runner = _runner(plan, launcher=LocalLauncher(), clock=lambda: now[0])
    runner.beat()
    now[0] += 30
    runner.beat()
    assert _status(plan).heartbeat_at == 5030.0


def test_read_latest_plan_checks_its_directory(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    other = tmp_path / "elsewhere" / "plan"
    other.mkdir(parents=True)
    (other / "rev-00000001.json").write_bytes((plan.plan_dir / "rev-00000001.json").read_bytes())
    with pytest.raises(ValueError, match="is the plan of"):
        read_latest_plan(other)
    with pytest.raises(FileNotFoundError):
        read_latest_plan(Path(tmp_path / "nothing"))
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        read_latest_plan(empty)


def test_read_latest_plan_reads_the_highest_revision_and_ignores_other_names(tmp_path):
    plan = noop_plan(tmp_path)
    first = _store(plan, _sub(plan))
    second = _store(first, _sub(first, job_id="101"))
    assert second.revision == 2
    first_bytes = (plan.plan_dir / "rev-00000001.json").read_bytes()
    (plan.plan_dir / "rev-00000009-copy.json").write_bytes(first_bytes)
    (plan.plan_dir / "notes.txt").write_text("not a revision")
    assert read_latest_plan(plan.plan_dir) == second


def test_read_latest_plan_refuses_a_file_that_holds_another_revision(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    (plan.plan_dir / "rev-00000003.json").write_bytes((plan.plan_dir / "rev-00000001.json").read_bytes())
    with pytest.raises(ValueError, match="holds revision 1, not 3"):
        read_latest_plan(plan.plan_dir)


def test_a_sigterm_before_the_stage_context_exists_still_stops_the_stage(tmp_path):
    decls = noop_two(gpu_kind="H200", sleep_sec=30, stage_time_sec=600)
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    launcher = LocalLauncher(gpus=2)
    box: list[Runner] = []
    checks = []

    def checkout(path):
        # The first call is the startup check; the second is the first stage's, made just
        # before its context is built.
        checks.append(path)
        if len(checks) == 2:
            box[0].terminate()
        return CODE

    runner = _runner(plan, launcher=launcher, checkout_state=checkout)
    box.append(runner)
    started = time.monotonic()
    assert runner.run() == EXIT_TERMINATED
    assert time.monotonic() - started < 10
    assert launcher.launched == [] and _status(plan).state == "terminated"


def test_the_runner_hands_the_launcher_the_allocations_gpus_and_uuids(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    seen = []

    def make_launcher(gpus, uuids, scratch):
        seen.append((gpus, uuids, scratch))
        return LocalLauncher(gpus=2)

    runner = _runner(plan, make_launcher=make_launcher)
    assert runner.run() == EXIT_PUBLISHED
    assert seen == [(2, ["GPU-0", "GPU-1"], plan.settings.site.node_scratch / "hls-100")]


# What a runner does when something goes wrong in its own code.

def _extra_output(monkeypatch, key_of) -> None:
    """Make the no-op stage name one more output, whose key is *key_of(ctx)*."""
    real_run = STAGES["noop"].run

    def run(self, ctx):
        return {**real_run(self, ctx), "extra": key_of(ctx)}

    monkeypatch.setattr(STAGES["noop"], "run", run)


def test_an_output_the_stage_never_wrote_fails_the_attempt_with_a_fault(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    _extra_output(monkeypatch, lambda ctx: ctx.output_key("ghost"))
    assert _runner(plan).run() == EXIT_FAILED
    status = _status(plan)
    assert status.state == "failed" and status.fault.fault_class == "infra" and status.fault.retryable
    assert status.fault.evidence["exception"] == "NotFound"
    attempt = _attempt(plan, "noop_a")
    assert attempt.state == "failed" and attempt.fault == status.fault and attempt.manifest_sha256 is None
    assert read_manifest(PosixStore(plan.store_root), plan.layout, "noop_a") is None


@pytest.mark.parametrize("where", ["another attempt", "outside outputs", "path back out"])
def test_an_output_outside_the_attempts_outputs_is_refused(tmp_path, monkeypatch, where):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    store = PosixStore(plan.store_root)
    other = plan.layout.attempt_file("noop_a", 7, "outputs/planted")
    store.write_bytes(other, b"another attempt's output")
    named = []

    def key_of(ctx):
        layout = ctx.plan.layout
        key = {"another attempt": other,
               "outside outputs": layout.attempt_file("noop_a", ctx.attempt, "status.json"),
               # Starts with the outputs prefix and resolves to the file of attempt 7.
               "path back out": layout.attempt("noop_a", ctx.attempt) + "/outputs/../../7/outputs/planted"}[where]
        named.append(key)
        return key

    _extra_output(monkeypatch, key_of)
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.fault_class == "infra" and not fault.retryable and "/outputs/" in fault.reason
    assert named[0] in fault.reason
    assert read_manifest(store, plan.layout, "noop_a") is None
    assert _attempt(plan, "noop_a", 8).state == "failed"


def test_the_runner_records_the_versions_of_its_nodes_tools_with_its_grant(tmp_path):
    """M-17: each submission's status and its grant check name the versions of Slurm, apptainer,
    pasta and the kernel its node has."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert _runner(plan).run() == EXIT_PUBLISHED
    assert _status(plan).tool_versions == VERSIONS
    [checked] = [e for e in _events(plan) if e["kind"] == "grant_checked"]
    assert checked["tool_versions"] == VERSIONS.model_dump()


def test_tool_versions_are_recorded_even_when_the_grant_check_fails(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert _runner(plan, measure=_grant(gpus=1)).run() == EXIT_FAILED
    assert _status(plan).tool_versions == VERSIONS


def test_a_tool_version_is_the_first_line_of_its_version_and_none_when_it_cannot_be_read():
    seen = []

    def answering(stdout: str = "", code: int = 0, raises: BaseException | None = None):
        def run(argv, **kw):
            seen.append((argv, kw))
            if raises is not None:
                raise raises
            return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")
        return run

    banner = "pasta 2024_09_06.6b38f07\nCopyright Red Hat\nGNU General Public License, version 2 or later\n"
    assert tool_version(["pasta", "--version"], answering(banner)) == "pasta 2024_09_06.6b38f07"
    assert seen[-1] == (["pasta", "--version"], {"capture_output": True, "text": True,
                                                 "timeout": VERSION_TIMEOUT_SEC})
    assert tool_version(["srun", "--version"], answering("\n  slurm 23.02.7  \n")) == "slurm 23.02.7"
    # A tool that is missing, hangs, fails or prints nothing has no version on record.
    assert tool_version(["srun", "--version"], answering(raises=FileNotFoundError(2, "srun"))) is None
    assert tool_version(["srun", "--version"], answering(raises=PermissionError(13, "srun"))) is None
    assert tool_version(["srun", "--version"], answering(raises=subprocess.TimeoutExpired("srun", 10))) is None
    assert tool_version(["srun", "--version"], answering("slurm 23.02.7\n", code=1)) is None
    assert tool_version(["srun", "--version"], answering("\n\n")) is None


def test_the_tools_are_the_ones_the_node_runs_and_pasta_the_seals(tmp_path, monkeypatch):
    """Slurm from srun and apptainer as the allocation's PATH finds them, pasta from the site's
    pasta_path when it names one, as the seal runs it, else from PATH, and the kernel's release; a
    tool the node lacks is recorded as None, never refused."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, line in (("srun", "slurm 23.02.7"), ("pasta", "pasta 2024_09_06.6b38f07")):
        (bin_dir / name).write_text(f"#!/bin/sh\necho '{line}'\n")
        (bin_dir / name).chmod(0o755)
    built = tmp_path / "pasta-build" / "pasta"
    built.parent.mkdir()
    built.write_text("#!/bin/sh\necho 'pasta 2025_01_21.4f2c8e7'\necho 'Copyright Red Hat'\n")
    built.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    kernel = os.uname().release
    assert measure_tools(None) == ToolVersions(slurm="slurm 23.02.7", apptainer=None,
                                               pasta="pasta 2024_09_06.6b38f07", kernel=kernel)
    assert measure_tools(built).pasta == "pasta 2025_01_21.4f2c8e7"
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert measure_tools(None) == ToolVersions(slurm=None, apptainer=None, pasta=None, kernel=kernel)


def test_a_grant_measurement_that_raises_is_an_infra_fault(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))

    def measure():
        raise RuntimeError("no cgroup")

    assert _runner(plan, measure=measure).run() == EXIT_FAILED
    status = _status(plan)
    assert status.state == "failed" and status.ended_at is not None
    assert status.fault.fault_class == "infra" and status.fault.retryable
    assert "RuntimeError: no cgroup" in status.fault.reason and status.fault.evidence["exception"] == "RuntimeError"
    assert _no_attempts(plan, "noop_a")


def test_a_corrupt_published_manifest_fails_the_run_and_is_not_retried(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    PosixStore(plan.store_root).create(plan.layout.manifest("noop_a"), b"{not json")
    assert _runner(plan).run() == EXIT_FAILED
    status = _status(plan)
    assert status.state == "failed" and status.fault.fault_class == "infra" and not status.fault.retryable
    assert "noop_a" in status.fault.reason and _no_attempts(plan, "noop_a")


def test_a_final_status_that_cannot_be_written_exits_1(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    real = Runner._update

    def update(self, **fields):
        if fields.get("state") == "succeeded":
            raise OSError("disk full")
        real(self, **fields)

    monkeypatch.setattr(Runner, "_update", update)
    assert _runner(plan).run() == EXIT_FAILED
    assert _status(plan).state != "succeeded"


# Refusals.

def test_a_runner_of_another_job_writes_nothing_under_the_submission(tmp_path, caplog):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan, job_id="999"))
    store = PosixStore(plan.store_root)
    live = AllocationStatus(instance=plan.instance, allocation=0, submission=1, job_id="999", node="g2",
                            state="running", started_at=1.0, heartbeat_at=2.0, end_time=None, stage="noop_a",
                            attempt=1, grant=None, tool_versions=None, swap_max_bytes=None, memory_enforced_by=None,
                            fault=None, ended_at=None)
    store.write_bytes(plan.layout.allocation_status(0, 1), live.model_dump_json(indent=2).encode())
    store.write_bytes(plan.layout.submission_file(0, 1, "events.jsonl"), b'{"at": 1.0, "kind": "stage_started"}\n')

    def snapshot():
        return (store.read_bytes(plan.layout.allocation_status(0, 1)),
                store.read_bytes(plan.layout.submission_file(0, 1, "events.jsonl")),
                store.list_children(plan.layout.submission(0, 1)))

    before = snapshot()
    with caplog.at_level(logging.ERROR):
        assert _runner(plan).run() == EXIT_REFUSED
    assert "is job 999, not 100" in caplog.text
    assert snapshot() == before and _status(plan) == live
    assert _no_attempts(plan, "noop_a")


def test_a_plan_that_records_no_job_id_yet_is_not_a_refusal(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan, job_id=None, status="intended"))
    assert _runner(plan).run() == EXIT_PUBLISHED
    store = PosixStore(plan.store_root)
    assert read_manifest(store, plan.layout, "noop_b")[0].times.submitted_at == 999.0


# The grant, the end time and the inputs.

def test_a_problem_the_measurement_reports_fails_the_grant_check(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    problem = "nvidia-smi lists 1 GPUs but 2 device nodes open"
    assert _runner(plan, measure=_grant(problems=[problem])).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.reason == f"grant mismatch: {problem}" and fault.retryable
    assert PosixStore(plan.store_root).list_children(plan.layout.stage("noop_a")) == []


def test_an_unknown_end_time_fails_before_any_stage(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan, job_end=JobEnd(controller=None, env=None))
    assert runner.run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault.reason.startswith("EndTime unknown") and fault.retryable and fault.fault_class == "infra"
    assert PosixStore(plan.store_root).list_children(plan.layout.stage("noop_a")) == []


def test_a_stage_whose_time_and_margins_exactly_fit_the_rest_of_the_job_starts(tmp_path):
    # Each stage needs its own time, the runner's margin before the job's end and the launch
    # allowance of each role it may start (noop_b's 4); the clock stands still.
    plan = noop_plan(tmp_path, site={"step_launch_allowance_sec": 10})
    plan = _store(plan, _sub(plan))
    end = 5000.0 + 600 + runner_mod.END_MARGIN_SEC + 4 * 10
    assert _runner(plan, job_end=JobEnd(controller=end, env=None), clock=lambda: 5000.0).run() == EXIT_PUBLISHED


def test_a_stage_a_second_short_of_its_margins_is_not_started(tmp_path):
    plan = noop_plan(tmp_path, site={"step_launch_allowance_sec": 10})
    plan = _store(plan, _sub(plan))
    end = 5000.0 + 600 + runner_mod.END_MARGIN_SEC + 1 * 10 - 1
    assert _runner(plan, job_end=JobEnd(controller=end, env=None), clock=lambda: 5000.0).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert fault is not None and fault.fault_class == "infra"
    assert f"stage noop_a needs {600 + int(runner_mod.END_MARGIN_SEC) + 10}s" in fault.reason


def test_a_stages_deadline_may_grow_up_to_the_jobs_end_less_the_runners_margin(tmp_path, monkeypatch):
    contexts = []

    class _Recorded(runner_mod.StageContext):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            contexts.append(self)

    monkeypatch.setattr(runner_mod, "StageContext", _Recorded)
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    assert _runner(plan, job_end=JobEnd(controller=9000.0, env=None), clock=lambda: 5000.0).run() == EXIT_PUBLISHED
    assert [c.deadline_cap for c in contexts] == [9000.0 - runner_mod.END_MARGIN_SEC] * 2


# A preemptible job: SLURM_JOB_END_TIME is its start + 60 s, while the
# controller's EndTime is its start + its 10-minute limit, and the job runs past the variable's.
PREEMPT_START = 1790975243
PREEMPT_ENV = PREEMPT_START + 60
PREEMPT_CONTROLLER = PREEMPT_START + 600
SCONTROL_TIMEOUT = subprocess.TimeoutExpired(["scontrol"], 60)


def _show_job(end_time: int | str) -> subprocess.CompletedProcess:
    """What ``scontrol show job 100`` prints, times as epoch seconds, for a job ending at *end_time*."""
    return done(f"JobId=100 JobName=x\n   StartTime={PREEMPT_START} EndTime={end_time} Deadline=N/A\n")


def _differ_events(plan: Plan) -> list[dict]:
    return [e for e in _events(plan) if e["kind"] == "end_time_sources_differ"]


def test_the_controllers_end_time_is_read_even_when_the_variable_is_set():
    run = FakeRun().on("scontrol", _show_job(PREEMPT_CONTROLLER))
    job_end = read_job_end("100", SlurmCommands(run=run), {"SLURM_JOB_END_TIME": str(PREEMPT_ENV)})
    assert job_end == JobEnd(controller=PREEMPT_CONTROLLER, env=PREEMPT_ENV)
    assert job_end.end_time == PREEMPT_CONTROLLER
    assert run.calls == [["scontrol", "show", "job", "100"]]


@pytest.mark.parametrize("reply", [
    SCONTROL_TIMEOUT,
    done(stderr="scontrol: error: Unable to contact slurm controller (connect failure)", rc=1),
    done(stderr="scontrol: error: Invalid job id specified", rc=1),
    done(rc=-9),
    _show_job("Unknown"),
    FileNotFoundError(2, "No such file or directory", "scontrol"),
], ids=["timeout", "unreachable", "refused", "killed", "unknown", "missing"])
def test_the_variable_stands_in_when_the_controller_does_not_answer(reply):
    job_end = read_job_end("100", SlurmCommands(run=FakeRun().on("scontrol", reply)),
                           {"SLURM_JOB_END_TIME": str(PREEMPT_ENV)})
    assert job_end == JobEnd(controller=None, env=PREEMPT_ENV)
    assert job_end.end_time == PREEMPT_ENV


@pytest.mark.parametrize("environ", [{}, {"SLURM_JOB_END_TIME": ""}, {"SLURM_JOB_END_TIME": "Unknown"}])
def test_with_neither_source_the_end_time_is_unknown(environ):
    job_end = read_job_end("100", SlurmCommands(run=FakeRun().on("scontrol", SCONTROL_TIMEOUT)), environ)
    assert job_end == JobEnd(controller=None, env=None)
    assert job_end.end_time is None


def test_on_a_preemptible_job_the_runner_keeps_to_the_controllers_end_time_and_records_both(tmp_path):
    # Both stages take 600 s; by the variable only 55 s are left.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = PREEMPT_START + 4 * 3600
    runner = _runner(plan, job_end=JobEnd(controller=controller, env=PREEMPT_ENV), clock=lambda: PREEMPT_START + 5.0)
    assert runner.run() == EXIT_PUBLISHED, _status(plan).fault
    assert _status(plan).end_time == controller
    [event] = _differ_events(plan)
    assert (event["end_time"], event["scontrol_end_time"], event["slurm_job_end_time"]) == (
        controller, controller, PREEMPT_ENV)


@pytest.mark.parametrize("env_minus_controller", [-61, 61])
def test_sources_more_than_a_minute_apart_either_way_are_recorded(tmp_path, env_minus_controller):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = PREEMPT_START + 4 * 3600
    env = controller + env_minus_controller
    runner = _runner(plan, job_end=JobEnd(controller=controller, env=env), clock=lambda: PREEMPT_START + 5.0)
    assert runner.run() == EXIT_PUBLISHED
    assert _status(plan).end_time == controller
    [event] = _differ_events(plan)
    assert (event["end_time"], event["scontrol_end_time"], event["slurm_job_end_time"]) == (
        controller, controller, env)


# The two are usually equal; a few seconds apart is the same EndTime.
@pytest.mark.parametrize("env_minus_controller", [0, 4, -60, 60])
def test_sources_that_agree_within_a_minute_record_nothing(tmp_path, env_minus_controller):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    controller = PREEMPT_START + 4 * 3600
    job_end = JobEnd(controller=controller, env=controller + env_minus_controller)
    assert _runner(plan, job_end=job_end, clock=lambda: PREEMPT_START + 5.0).run() == EXIT_PUBLISHED
    assert _status(plan).end_time == controller
    assert _differ_events(plan) == []


def test_without_the_controller_the_runner_keeps_to_the_variable(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan, job_end=JobEnd(controller=None, env=PREEMPT_ENV), clock=lambda: PREEMPT_START + 5.0)
    assert runner.run() == EXIT_FAILED
    assert _status(plan).end_time == PREEMPT_ENV
    assert _status(plan).fault.reason == (f"stage noop_a needs {600 + int(runner_mod.END_MARGIN_SEC)}s (its time, the "
                                          "runner's margin and its roles' launch allowance) and the allocation ends in 55s")
    assert _differ_events(plan) == []


OTHER = InstanceLayout.rollout("noop", "other").manifest("source")


def _pin(plan: Plan, pinned: dict[str, str]) -> Plan:
    """*plan* with *pinned* as the other-instance inputs of its second stage."""
    stages = [s.model_copy(update={"inputs": pinned}) if s.name == "noop_b" else s for s in plan.stages]
    return plan.model_copy(update={"stages": stages})


def test_every_pinned_input_is_checked_before_the_first_stage_starts(tmp_path):
    plan = _pin(noop_plan(tmp_path), {OTHER: "0" * 64})
    plan = _store(plan, _sub(plan))
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert OTHER in fault.reason and "not published" in fault.reason and fault.retryable
    assert _no_attempts(plan, "noop_a") and _no_attempts(plan, "noop_b")


@pytest.mark.parametrize("raw, digest, why", [
    (b"{}", "0" * 64, "is not the planned"),
    (b"{junk", sha256_bytes(b"{junk"), "not a valid manifest"),
], ids=["digest differs", "unparseable"])
def test_a_pinned_input_that_cannot_be_trusted_is_not_retried(tmp_path, raw, digest, why):
    plan = _pin(noop_plan(tmp_path), {OTHER: digest})
    plan = _store(plan, _sub(plan))
    PosixStore(plan.store_root).create(OTHER, raw)
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert why in fault.reason and fault.fault_class == "infra" and not fault.retryable
    assert _no_attempts(plan, "noop_a")


def test_a_stage_after_an_unpublished_stage_is_an_infra_fault(tmp_path):
    plan = noop_plan(tmp_path)
    plan = plan.with_allocation(plan.allocation(0).model_copy(update={"stages": ["noop_b"]}))
    plan = _store(plan, _sub(plan))
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert "stages ['noop_a'] are not published" in fault.reason and fault.retryable
    assert _no_attempts(plan, "noop_b")


# What the runner records while it runs and when it ends.

def test_the_status_says_starting_while_the_grant_is_measured(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    seen = []

    def measure():
        seen.append(_status(plan).state)
        return _grant()()

    assert _runner(plan, measure=measure).run() == EXIT_PUBLISHED
    assert seen == ["starting"]


def test_the_heartbeat_moves_while_a_stage_runs_and_stops_when_run_returns(tmp_path):
    decls = noop_one(gpus=1, gpu_kind="H200", sleep_sec=3, stage_time_sec=600)
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    seen: set[float] = set()
    done = threading.Event()

    def sample():
        while not done.is_set():
            try:
                status = _status(plan)
            except NotFound:
                continue
            if status.state == "running":
                seen.add(status.heartbeat_at)
            time.sleep(0.1)

    runner = _runner(plan, measure=_grant(gpus=1, cpus=2, memory_gib=6.0), launcher=LocalLauncher(gpus=1))
    sampler = threading.Thread(target=sample)
    sampler.start()
    try:
        assert runner.run() == EXIT_PUBLISHED
    finally:
        done.set()
        sampler.join()
    assert len(seen) >= 2
    last = _status(plan).heartbeat_at
    time.sleep(2.2)
    assert _status(plan).heartbeat_at == last


def test_no_background_thread_outlives_run_and_the_scratch_goes_after_the_final_status(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    scratch = plan.settings.site.node_scratch / "hls-100"
    scratch_at_final = []
    real = Runner._update

    def update(self, **fields):
        if fields.get("state") == "succeeded":
            scratch_at_final.append(scratch.exists())
        real(self, **fields)

    def make_launcher(gpus, uuids, where):
        (where / "roles").mkdir(parents=True)
        return LocalLauncher(gpus=2)

    monkeypatch.setattr(Runner, "_update", update)
    assert _runner(plan, make_launcher=make_launcher).run() == EXIT_PUBLISHED
    assert scratch_at_final == [True] and not scratch.exists()
    names = {t.name for t in threading.enumerate()}
    assert not names & {"hls-memory-watch", "hls-runner-heartbeat"}


@pytest.mark.parametrize("hooked, code, state, left", [
    ("noop_a", EXIT_TERMINATED, "terminated", ["noop_b"]),
    ("noop_b", EXIT_PUBLISHED, "succeeded", []),
], ids=["between stages", "after the last stage"])
def test_a_sigterm_after_a_publish(tmp_path, monkeypatch, hooked, code, state, left):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    real = runner_mod.publish_manifest
    during: list[str] = []

    def publish(store, layout, manifest):
        digest = real(store, layout, manifest)
        if manifest.stage == hooked:
            runner.terminate()
            # The SIGTERM's own records leave an attempt that is publishing to the stage.
            runner._shutdown.join(5)
            during.append(_attempt(plan, hooked).state)
        return digest

    monkeypatch.setattr(runner_mod, "publish_manifest", publish)
    assert runner.run() == code
    assert during == ["running"] and _attempt(plan, hooked).state == "succeeded"
    assert _status(plan).state == state
    store = PosixStore(plan.store_root)
    for name in left:
        assert plan.layout.attempts(store, name) == [] and read_manifest(store, plan.layout, name) is None
    assert read_manifest(store, plan.layout, "noop_a") is not None


def test_a_manifest_another_attempt_published_first_counts_and_records_no_digest(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    real = runner_mod.publish_manifest

    def publish(store, layout, manifest):
        real(store, layout, manifest)
        return real(store, layout, manifest)

    monkeypatch.setattr(runner_mod, "publish_manifest", publish)
    assert _runner(plan).run() == EXIT_PUBLISHED
    attempt = _attempt(plan, "noop_a")
    assert attempt.state == "succeeded" and attempt.manifest_sha256 is None
    assert "stage_published_elsewhere" in _kinds(plan)


def test_a_stage_that_runs_past_its_time_ends_time_limit_and_publishes_nothing(tmp_path):
    decls = noop_one(gpus=1, gpu_kind="H200", sleep_sec=30, stage_time_sec=2)
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    started = time.monotonic()
    runner = _runner(plan, measure=_grant(gpus=1, cpus=2, memory_gib=6.0), launcher=LocalLauncher(gpus=1))
    assert runner.run() == EXIT_FAILED
    assert time.monotonic() - started < 2 + 10 + 3
    attempt = _attempt(plan, "noop")
    assert attempt.state == "failed" and attempt.manifest_sha256 is None
    assert attempt.fault.evidence["gpu0.ending"] == "time_limit"
    assert read_manifest(PosixStore(plan.store_root), plan.layout, "noop") is None


# A role whose step never reports its grant.

class _SilentHandle(RoleHandle):
    """A role whose step never starts, so it never reports. Waiting for its report lasts the
    timeout it is given, up to 20 s, and ends early when the role is killed."""

    def __init__(self, spec) -> None:
        self.spec = spec
        self.timeouts: list[float] = []
        self.killed = threading.Event()

    def started(self):
        return False

    def report(self, timeout_sec):
        self.timeouts.append(timeout_sec)
        if self.killed.wait(min(timeout_sec, 20.0)):
            raise RoleFailed("killed before it reported")
        raise TimeoutError("no report")

    def poll(self):
        return -9 if self.killed.is_set() else None

    def wait(self, timeout_sec=None):
        self.killed.wait(timeout_sec)
        return self.poll()

    def kill(self):
        self.killed.set()

    def ended_by_scheduler(self):
        return False

    def end_state(self):
        return None


class _SilentLauncher(Launcher):
    def __init__(self) -> None:
        self.handles: list[_SilentHandle] = []

    def node_size(self) -> NodeSize:
        return NODE

    def launch(self, spec, *, share_gpus_of=None):
        self.handles.append(_SilentHandle(spec))
        return self.handles[-1]

    def terminate(self, handles):
        for handle in handles:
            handle.killed.set()


def _silent_runner(tmp_path, stage_time_sec: int,
                   job_end: JobEnd | None = None) -> tuple[Plan, Runner, _SilentLauncher]:
    decls = noop_one(gpus=1, gpu_kind="H200", sleep_sec=0, stage_time_sec=stage_time_sec)
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    launcher = _SilentLauncher()
    runner = _runner(plan, measure=_grant(gpus=1, cpus=2, memory_gib=6.0), launcher=launcher, job_end=job_end)
    return plan, runner, launcher


def test_a_sigterm_while_a_role_has_not_reported_stops_the_run_within_a_second(tmp_path):
    plan, runner, launcher = _silent_runner(tmp_path, stage_time_sec=600)
    threading.Timer(1.0, runner.terminate).start()
    started = time.monotonic()
    assert runner.run() == EXIT_TERMINATED
    assert time.monotonic() - started < 5
    assert launcher.handles[0].killed.is_set() and max(launcher.handles[0].timeouts) <= 1.0


def test_the_wait_for_a_step_to_start_is_sliced_and_ends_at_the_jobs_end_less_the_margin(tmp_path):
    # A step that never starts is waited for past the stage's time, up to what the job leaves.
    end = JobEnd(controller=time.time() + runner_mod.END_MARGIN_SEC + 2, env=None)
    plan, runner, launcher = _silent_runner(tmp_path, stage_time_sec=1, job_end=end)
    started = time.monotonic()
    assert runner.run() == EXIT_FAILED
    assert time.monotonic() - started < 6
    timeouts = launcher.handles[0].timeouts
    assert len(timeouts) >= 2 and max(timeouts) <= 1.0 and sum(timeouts) <= 2.5
    assert "role gpu0: launch failed" in _status(plan).fault.reason


def test_stopping_the_background_waits_for_a_heartbeat_in_flight(tmp_path, monkeypatch):
    # run() writes its final status after this wait, so no heartbeat can follow it.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    beating = threading.Event()
    real = Runner.beat

    def slow_beat(self):
        beating.set()
        time.sleep(0.5)
        real(self)

    monkeypatch.setattr(Runner, "beat", slow_beat)
    runner._beat = threading.Thread(target=runner._heartbeat_loop, daemon=True)
    runner._beat.start()
    try:
        assert beating.wait(5)
        runner._stop_background()
        assert not runner._beat.is_alive()
    finally:
        runner._stop.set()


def test_an_error_after_the_attempt_is_claimed_still_writes_its_failed_status(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    real = Runner._update

    def update(self, **fields):
        if fields.get("state") == "running":
            raise OSError("store went away")
        real(self, **fields)

    monkeypatch.setattr(Runner, "_update", update)
    assert _runner(plan).run() == EXIT_FAILED
    attempt = _attempt(plan, "noop_a")
    assert attempt.state == "failed" and attempt.ended_at is not None
    assert attempt.fault.fault_class == "infra" and attempt.fault.retryable
    assert attempt.fault.evidence["exception"] == "OSError"
    assert _status(plan).fault == attempt.fault


def test_a_sigterm_after_the_roles_ended_but_before_the_publish_publishes_nothing(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    real_run = STAGES["noop"].run

    def run(self, ctx):
        outputs = real_run(self, ctx)
        runner.terminate()
        return outputs

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_TERMINATED
    attempt = _attempt(plan, "noop_a")
    assert attempt.state == "failed" and "terminated" in attempt.fault.reason
    assert read_manifest(PosixStore(plan.store_root), plan.layout, "noop_a") is None


@pytest.mark.parametrize("failing", ["status", "event"])
def test_a_failed_final_record_of_an_attempt_keeps_the_stages_own_fault(tmp_path, monkeypatch, failing):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    contestant = FaultRecord(fault_class="contestant", reason="contestant role x ran out of memory", retryable=False)

    def run(self, ctx):
        raise StageFault(contestant)

    monkeypatch.setattr(STAGES["noop"], "run", run)
    real_write, real_emit = PosixStore.write_bytes, EventLog.emit

    def write_bytes(self, key, data):
        if failing == "status" and key.endswith("attempts/1/status.json") and b'"failed"' in data:
            raise OSError("store went away")
        real_write(self, key, data)

    def emit(self, kind, /, **fields):
        if failing == "event" and kind == "stage_ended":
            raise OSError("store went away")
        real_emit(self, kind, **fields)

    monkeypatch.setattr(PosixStore, "write_bytes", write_bytes)
    monkeypatch.setattr(EventLog, "emit", emit)
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert (fault.fault_class, fault.retryable, fault.reason) == ("contestant", False, contestant.reason)
    assert fault.evidence["status_write_exception"] == "OSError"
    assert fault.evidence["status_write_message"] == "store went away"


def test_the_runners_memory_watch_reads_the_jobs_own_oom_count(tmp_path):
    # Whose limit an OOM kill hit is told by the job cgroup's own oom count.
    job = tmp_path / "cgroup" / "job_100"
    job.mkdir(parents=True)
    (job / "memory.events.local").write_text("low 0\nhigh 0\nmax 0\noom 2\noom_kill 0\noom_group_kill 0\n")
    (job / "memory.events").write_text("low 0\nhigh 0\nmax 0\noom 3\noom_kill 0\noom_group_kill 0\n")
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan, job_cgroup=job)
    assert runner.run() == EXIT_PUBLISHED
    assert runner.watch.job_dir == job and (runner.watch.job_ooms().own, runner.watch.job_ooms().below) == (2, 1)


class _RecordsAtStop(LocalLauncher):
    """A launcher that reads the attempt's and the allocation's status from the store when the
    stage asks it to stop the roles."""

    def __init__(self, plan: Plan) -> None:
        super().__init__(gpus=2)
        self.plan = plan
        self.seen: list[tuple[AttemptStatus, AllocationStatus]] = []

    def terminate(self, handles) -> None:
        self.seen.append((_attempt(self.plan, "noop_a"), _status(self.plan)))
        super().terminate(handles)


def test_a_sigterm_writes_the_final_records_before_it_stops_the_roles(tmp_path):
    # With a KillWait of 10 s, roles that take longer to stop than what is left of it would
    # leave both records saying "running" if they were written only after the roles stopped.
    decls = noop_two(gpu_kind="H200", sleep_sec=30, stage_time_sec=600)
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    launcher = _RecordsAtStop(plan)
    runner = _runner(plan, launcher=launcher)
    threading.Timer(1.0, runner.terminate).start()
    assert runner.run() == EXIT_TERMINATED
    [(attempt, status)] = launcher.seen
    assert attempt.state == "failed" and attempt.ended_at is not None and "terminated" in attempt.fault.reason
    assert status.state == "terminated" and status.ended_at is not None and "terminated" in status.fault.reason
    assert _attempt(plan, "noop_a").state == "failed" and _status(plan).state == "terminated"


def test_a_final_status_is_never_replaced_by_one_that_is_not(tmp_path, monkeypatch):
    # The SIGTERM's records land between the attempt's start and the status that says the stage
    # runs; that status must not undo them.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    real_begin = Runner._begin_attempt

    def begin(self, key, running):
        began = real_begin(self, key, running)
        self.terminate()
        self._shutdown.join(5)
        return began

    monkeypatch.setattr(Runner, "_begin_attempt", begin)
    states: list[str] = []
    real_write = PosixStore.write_bytes

    def write_bytes(store, key, data):
        if key == plan.layout.allocation_status(0, 1):
            states.append(AllocationStatus.model_validate_json(data).state)
        real_write(store, key, data)

    monkeypatch.setattr(PosixStore, "write_bytes", write_bytes)
    assert runner.run() == EXIT_TERMINATED
    assert set(states[states.index("terminated"):]) == {"terminated"}
    assert _attempt(plan, "noop_a").state == "failed"


def test_a_sigterm_before_the_run_writes_nothing_for_a_submission_of_another_job(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan, job_id="999"))
    runner = _runner(plan)
    runner.terminate()
    runner._shutdown.join(5)
    assert runner.run() == EXIT_REFUSED
    assert not PosixStore(plan.store_root).exists(plan.layout.allocation_status(0, 1))


def test_an_attempt_claimed_after_the_sigterms_records_is_never_recorded_running(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    box: list[Runner] = []
    checks = []

    def checkout(path):
        # The second call is the first stage's check, made just before its attempt is claimed.
        checks.append(path)
        if len(checks) == 2:
            box[0].terminate()
            box[0]._shutdown.join(5)
        return CODE

    runner = _runner(plan, checkout_state=checkout)
    box.append(runner)
    states: list[str] = []
    real_write = PosixStore.write_bytes

    def write_bytes(store, key, data):
        if key == plan.layout.attempt_file("noop_a", 1, "status.json"):
            states.append(AttemptStatus.model_validate_json(data).state)
        real_write(store, key, data)

    monkeypatch.setattr(PosixStore, "write_bytes", write_bytes)
    assert runner.run() == EXIT_TERMINATED
    assert states == ["failed"] and "stage_started" not in _kinds(plan)


def test_a_job_scratch_deeper_than_the_interpreter_recurses_is_removed_when_the_runner_ends(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    real = STAGES["noop"].run

    def run(self, ctx):
        p = ctx.scratch / "left"
        for _ in range(sys.getrecursionlimit() + 100):
            p = p / "a"
            p.mkdir(parents=True, exist_ok=True)
        return real(self, ctx)

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_PUBLISHED
    assert not runner.scratch.exists()


def test_a_job_scratch_holding_read_only_directories_is_removed_when_the_runner_ends(tmp_path, monkeypatch):
    # An extraction cut short leaves its directories as the image has them.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    real = STAGES["noop"].run

    def run(self, ctx):
        left = ctx.scratch / "sandboxes" / ".extract-left" / "usr"
        if not left.exists():
            left.mkdir(parents=True)
            (left / "f").write_text("x")
            left.chmod(0o555)
            left.parent.chmod(0o500)
        return real(self, ctx)

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_PUBLISHED, _status(plan).fault
    assert not runner.scratch.exists()


def test_a_checkout_that_changes_after_the_first_stage_refuses_the_second(tmp_path):
    # The checkout is checked again before every stage, not only at the start.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    checks: list[Path] = []
    moved = CODE.model_copy(update={"hls_commit": "c" * 40})

    def checkout(path):
        # The startup check and noop_a's see the planned checkout; noop_b's sees another.
        checks.append(path)
        return CODE if len(checks) <= 2 else moved

    assert _runner(plan, checkout_state=checkout).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert "checkout changed" in fault.reason and not fault.retryable
    store = PosixStore(plan.store_root)
    assert read_manifest(store, plan.layout, "noop_a") is not None and _no_attempts(plan, "noop_b")
    assert len(checks) == 3


def _fake_job_cgroup(root: Path, *, below: int) -> Path:
    """A job cgroup whose own limit was never reached and *below* limits under it were."""
    job = root / "job_100"
    job.mkdir(parents=True, exist_ok=True)
    (job / "memory.events.local").write_text("low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\n")
    (job / "memory.events").write_text(f"low 0\nhigh 0\nmax 0\noom {below}\noom_kill 0\n")
    return job


def test_a_contestant_fault_the_stage_decided_before_a_sigterm_is_kept_in_both_records(tmp_path, monkeypatch):
    # A contestant fault is never relaunched. The contestant's server
    # reached its own limit and the stage reported it before the SIGTERM (a time limit, a
    # preemption) came.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    job = _fake_job_cgroup(tmp_path / "cg", below=0)
    runner = _runner(plan, launcher=LocalLauncher(gpus=2, steps_under=job, out_of_memory=("server",)),
                     job_cgroup=job)
    early: list[tuple[AttemptStatus, AllocationStatus]] = []

    def run(self, ctx):
        _fake_job_cgroup(tmp_path / "cg", below=1)
        ctx.start_role("server", [sys.executable, "-c", "pass"], ResourceShape(cpus=1, memory_gib=1, time_sec=60),
                       owner="contestant")
        outcomes = ctx.wait_roles()
        runner.terminate()
        runner._shutdown.join(5)
        early.append((_attempt(plan, "noop_a"), _status(plan)))
        raise StageFault(stage_fault(outcomes))

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_TERMINATED
    [(attempt, status)] = early
    assert (attempt.state, attempt.fault.fault_class) == ("failed", "contestant")
    assert (status.state, status.fault.fault_class) == ("terminated", "contestant")
    final = _status(plan)
    assert final.fault.fault_class == "contestant" and _attempt(plan, "noop_a").fault.fault_class == "contestant"
    d = decide(plan, Observation(now=final.ended_at + 1, published=frozenset(),
                                 jobs={"100": JobInfo("100", JobState.CANCELLED)}, statuses={(0, 1): final}))
    assert d.plan.outcome.status == "failed" and len(d.plan.allocations) == 1


def test_a_contestant_fault_the_stage_raises_after_a_sigterm_is_kept(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    contestant = FaultRecord(fault_class="contestant", reason="contestant role x ran out of memory", retryable=False)

    def run(self, ctx):
        runner.terminate()
        runner._shutdown.join(5)
        raise StageFault(contestant)

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_TERMINATED
    assert _status(plan).fault == contestant and _attempt(plan, "noop_a").fault == contestant


def test_a_sigterm_after_the_stage_passed_its_point_of_no_retry_keeps_that_in_both_records(tmp_path, monkeypatch):
    """Once a stage has done work a retry would do again, the agent build's commit point, it
    declares the fault an end from then on is: infra, never retried. A SIGTERM after that
    writes it into both records in place of the termination's retryable fault, and the
    stage's own failure that follows, a role the shutdown stopped, keeps it too."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    committed = FaultRecord(fault_class="infra", reason="the build failed after its commit point", retryable=False,
                            evidence={"commit_point": "bundle"})
    early: list[tuple[AttemptStatus, AllocationStatus]] = []

    def run(self, ctx):
        ctx.declare_no_retry(committed)
        runner.terminate()
        runner._shutdown.join(5)
        early.append((_attempt(plan, "noop_a"), _status(plan)))
        raise StageFault(FaultRecord(fault_class="infra", reason="role transcript: stopped", retryable=True))

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_TERMINATED
    [(attempt, status)] = early
    assert attempt.fault == committed and (status.state, status.fault) == ("terminated", committed)
    assert _status(plan).fault == committed and _attempt(plan, "noop_a").fault == committed


#: The fault the agent build declares once its commit point wrote the bundle.
COMMITTED = FaultRecord(fault_class="infra", reason="the build failed after its commit point", retryable=False,
                        evidence={"commit_point": "bundle"})


def test_a_declaration_of_no_retry_is_in_the_attempts_status_the_moment_it_is_made(tmp_path, monkeypatch):
    """The runner writes the declaration into the attempt's running status at once, and its
    own status names the attempt, so the orchestrator reads it after a runner that died with no
    record of its own (``reconcile.job_fault``). The attempt's final record keeps it, and a
    retryable failure that follows, the runner's own included, ends the attempt with it."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    seen: list[tuple[AttemptStatus, AllocationStatus]] = []

    def run(self, ctx):
        seen.append((_attempt(plan, "noop_a"), _status(plan)))
        ctx.declare_no_retry(COMMITTED)
        seen.append((_attempt(plan, "noop_a"), _status(plan)))
        raise StageFault(FaultRecord(fault_class="infra", reason="role transcript: failed", retryable=True))

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_FAILED
    [(before, running), (after, _)] = seen
    assert (before.state, before.no_retry) == ("running", None)
    assert (running.stage, running.attempt) == ("noop_a", 1)
    assert (after.state, after.no_retry) == ("running", COMMITTED)
    final = _attempt(plan, "noop_a")
    assert (final.state, final.no_retry, final.fault.reason, final.fault.retryable) == (
        "failed", COMMITTED, COMMITTED.reason, False)
    assert final.fault.evidence["fault"] == "role transcript: failed"
    status = _status(plan)
    assert (status.fault.retryable, status.stage, status.attempt) == (False, None, None)


def test_a_declaration_that_cannot_be_written_ends_the_stage_never_retried(tmp_path, monkeypatch):
    """Fail closed: a stage may not go on past its point of no retry when the declaration is not
    on the store, since a runner that died later would have its session run again."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    attempt_key = plan.layout.attempt_file("noop_a", 1, "status.json")
    went_on: list[bool] = []
    real_write = PosixStore.write_bytes

    def write(self, key, data):
        if key == attempt_key and b'"no_retry": {' in data:
            raise OSError(5, "EIO: a test")
        return real_write(self, key, data)

    def run(self, ctx):
        monkeypatch.setattr(PosixStore, "write_bytes", write)
        try:
            ctx.declare_no_retry(COMMITTED)
            went_on.append(True)
        finally:
            monkeypatch.setattr(PosixStore, "write_bytes", real_write)
        return {}

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_FAILED
    assert went_on == []
    final = _attempt(plan, "noop_a")
    assert (final.state, final.fault.retryable, final.fault.reason) == ("failed", False, COMMITTED.reason)


def test_a_declaration_after_the_sigterms_records_replaces_their_retried_fault(tmp_path, monkeypatch):
    """A SIGTERM during the commit point's last try writes the final records with the
    termination's fault, which is retried; the declaration that follows rewrites both, so a kill
    at the end of the job's grace leaves the declared fault."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    seen: list[tuple[AttemptStatus, AllocationStatus]] = []

    def run(self, ctx):
        runner.terminate()
        runner._shutdown.join(5)
        assert _status(plan).fault.retryable is True
        ctx.declare_no_retry(COMMITTED)
        seen.append((_attempt(plan, "noop_a"), _status(plan)))
        raise StageFault(FaultRecord(fault_class="infra", reason="role transcript: stopped", retryable=True))

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_TERMINATED
    [(attempt, status)] = seen
    assert (attempt.state, attempt.fault, attempt.no_retry) == ("failed", COMMITTED, COMMITTED)
    assert (status.state, status.fault) == ("terminated", COMMITTED)


def test_a_sigterm_before_the_stage_decided_anything_is_infra_whoever_owns_the_roles(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)

    def run(self, ctx):
        ctx.start_role("server", [sys.executable, "-c", "import time; time.sleep(30)"],
                       ResourceShape(cpus=1, memory_gib=1, time_sec=60), owner="contestant")
        runner.terminate()
        fault = stage_fault(ctx.wait_roles())
        raise StageFault(fault)

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_TERMINATED
    status, attempt = _status(plan), _attempt(plan, "noop_a")
    assert (status.fault.fault_class, status.fault.retryable) == ("infra", True)
    assert "terminated" in status.fault.reason and attempt.fault == status.fault


def test_a_sigterm_between_the_running_record_and_the_context_still_stops_the_roles(tmp_path, monkeypatch):
    # The shutdown finds no context to stop yet; the stage's thread must stop the roles itself
    # once the context exists, or they run to the stage's time limit.
    plan = noop_plan(tmp_path, decls=noop_two(gpu_kind="H200", sleep_sec=30, stage_time_sec=600))
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    real_ctx = runner_mod.StageContext

    def make_ctx(**kw):
        runner.terminate()
        runner._shutdown.join(5)
        return real_ctx(**kw)

    monkeypatch.setattr(runner_mod, "StageContext", make_ctx)
    started = time.monotonic()
    assert runner.run() == EXIT_TERMINATED
    assert time.monotonic() - started < 20


def test_a_sigterm_after_the_final_status_leaves_it(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    runner = _runner(plan)
    contestant = FaultRecord(fault_class="contestant", reason="contestant role x ran out of memory", retryable=False)

    def run(self, ctx):
        raise StageFault(contestant)

    monkeypatch.setattr(STAGES["noop"], "run", run)
    assert runner.run() == EXIT_FAILED
    runner.terminate()
    runner._shutdown.join(5)
    status = _status(plan)
    assert (status.state, status.fault) == ("failed", contestant)


def _main(plan: Plan, monkeypatch, own_cgroup, *, env_end: int | None = None,
          scontrol: FakeRun | None = None, handlers: dict | None = None) -> int:
    """Run the runner's entry point on *plan*'s allocation 0, submission 1, as job 100, with
    *own_cgroup* standing in for the lookup of this process's cgroup. ``SLURM_JOB_END_TIME``
    is *env_end*, and the Slurm tools run through *scontrol*; by default both say the job ends
    a day from now. Each signal handler the entry point installs goes into *handlers*, by
    signal, instead of the process. The log handler and the capture of warnings the entry point
    installs are taken away again, so no later test logs through them."""
    env_end = int(time.time()) + 86400 if env_end is None else env_end
    scontrol = FakeRun().on("scontrol", _show_job(env_end)) if scontrol is None else scontrol
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("SLURM_JOB_END_TIME", str(env_end))
    monkeypatch.setattr(slurm_substrate, "SlurmCommands", lambda: SlurmCommands(run=scontrol))
    monkeypatch.setattr(slurm_substrate, "own_cgroup", own_cgroup)
    installed = {} if handlers is None else handlers
    monkeypatch.setattr(runner_mod.signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler))
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        return runner_mod.main([str(plan.plan_dir), "--allocation", "0", "--submission", "1"])
    finally:
        for handler in [h for h in root.handlers if h not in handlers]:
            root.removeHandler(handler)
        root.setLevel(level)
        logging.captureWarnings(False)


def _record_runners(monkeypatch) -> list[dict]:
    """Replace the Runner with one that only records the keyword arguments it was made with,
    in the list returned, and succeeds."""
    made: list[dict] = []

    class Recorded:
        def __init__(self, *args, **kwargs) -> None:
            made.append(kwargs)
            self.status = AllocationStatus.model_construct(state="succeeded")
            self.scrubber = Scrubber({})

        def run(self) -> int:
            return EXIT_PUBLISHED

    monkeypatch.setattr(runner_mod, "Runner", Recorded)
    return made


def test_the_runners_entry_point_hands_it_the_jobs_cgroup(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    made = _record_runners(monkeypatch)
    assert _main(plan, monkeypatch, lambda: "/system.slice/slurmstepd.scope/job_100/step_batch/user/task_0") == 0
    [kwargs] = made
    assert kwargs["job_cgroup"] == CGROUP_ROOT / "system.slice/slurmstepd.scope/job_100"


def test_the_runners_entry_point_raises_its_soft_process_limit_before_the_runner_starts(tmp_path, monkeypatch):
    # Every harness process keeps the raised limit, so the harness keeps its reserve whatever the
    # contestant's steps hold; the site's soft limit (RHEL sets 4,096) would leave it far less.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    order: list[str] = []
    monkeypatch.setattr(runner_mod, "raise_soft_limit", lambda: order.append("raised") or (4096, 65536))
    made = _record_runners(monkeypatch)
    real_init = runner_mod.Runner.__init__

    def init(self, *args, **kwargs):
        order.append("runner")
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(runner_mod.Runner, "__init__", init)
    assert _main(plan, monkeypatch, lambda: "/system.slice/slurmstepd.scope/job_100/step_batch/user/task_0") == 0
    assert order == ["raised", "runner"] and len(made) == 1


def test_the_runners_entry_point_asks_the_controller_once_within_a_timeout(tmp_path, monkeypatch):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    made = _record_runners(monkeypatch)
    scontrol = FakeRun().on("scontrol", _show_job(PREEMPT_CONTROLLER))
    assert _main(plan, monkeypatch, lambda: "/job_100/step_batch", env_end=PREEMPT_ENV, scontrol=scontrol) == 0
    [kwargs] = made
    assert kwargs["read_job_end"]() == JobEnd(controller=PREEMPT_CONTROLLER, env=PREEMPT_ENV)
    assert scontrol.calls == [["scontrol", "show", "job", "100"]]
    assert scontrol.kwargs[0]["timeout"] == DEFAULT_TIMEOUT_SEC["scontrol"]


def test_a_cancel_while_the_controller_is_asked_for_the_end_time_leaves_the_final_records(tmp_path, monkeypatch):
    # scontrol may take its whole 60 s timeout. A SIGTERM meanwhile must find the status written
    # and its handler installed, so the allocation ends "terminated" and not as a silent death.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    handlers: dict = {}
    seen = {}
    end = int(time.time()) + 86400

    def cancelled_meanwhile(argv):
        seen["status_written"] = PosixStore(plan.store_root).exists(plan.layout.allocation_status(0, 1))
        seen["handler"] = signal.SIGTERM in handlers
        if seen["handler"]:
            handlers[signal.SIGTERM](signal.SIGTERM, None)
        return _show_job(end)

    code = _main(plan, monkeypatch, lambda: "/system.slice/slurmstepd.scope/job_100/step_batch/user/task_0",
                 scontrol=FakeRun().on("scontrol", cancelled_meanwhile), handlers=handlers)
    assert seen == {"status_written": True, "handler": True}
    status = _status(plan)
    assert (code, status.state, status.fault.reason) == (EXIT_TERMINATED, "terminated", "the allocation was terminated")
    assert status.end_time == end and _no_attempts(plan, "noop_a")


def test_a_runner_whose_job_cgroup_cannot_be_found_records_why(tmp_path, monkeypatch):
    # The lookup runs before the runner exists; its failure must still end in a recorded fault,
    # not a traceback with no status.
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))

    def no_cgroup() -> str:
        raise RuntimeError("/proc/self/cgroup has no cgroup v2 entry")

    assert _main(plan, monkeypatch, no_cgroup) == EXIT_FAILED
    status = _status(plan)
    assert (status.state, status.fault.fault_class, status.fault.retryable) == ("failed", "infra", True)
    assert "no cgroup v2 entry" in status.fault.reason
    assert _no_attempts(plan, "noop_a")


# --- the job's node-local scratch -------------------------------------------------------------


def test_the_jobs_scratch_is_made_0700_before_any_stage_uses_it(tmp_path):
    """The scratch holds every role's directory and every node-local copy (weights, rows,
    extracted images): it is made 0700 before the first stage, so no other user of the node can
    read what is under it, whatever modes those have."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    modes = []

    def make_launcher(gpus, uuids, where):
        modes.append(stat.S_IMODE(os.lstat(where).st_mode))
        return LocalLauncher(gpus=2)

    assert _runner(plan, make_launcher=make_launcher).run() == EXIT_PUBLISHED
    assert modes == [0o700]


@pytest.mark.parametrize("planted", ["link", "file"])
def test_a_scratch_path_another_planted_is_refused_and_left_alone(tmp_path, planted):
    """A world-writable node scratch lets anyone make ``hls-<job>`` first: a link there, or
    anything that is not this user's own directory, is refused, infra, and nothing is written
    through it."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    scratch = plan.settings.site.node_scratch / "hls-100"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    if planted == "link":
        scratch.symlink_to(elsewhere)
    else:
        scratch.write_text("x")
    assert _runner(plan).run() == EXIT_FAILED
    fault = _status(plan).fault
    assert (fault.fault_class, fault.retryable) == ("infra", True) and "scratch" in fault.reason
    assert list(elsewhere.iterdir()) == [] and _no_attempts(plan, "noop_a")
    assert scratch.is_symlink() if planted == "link" else scratch.read_text() == "x"


def test_the_scratch_this_jobs_earlier_run_left_is_made_0700_and_used(tmp_path):
    """A requeued job runs again under its own id, on the same node perhaps: the directory its
    earlier run left is this user's, and is used, made 0700 again."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    scratch = plan.settings.site.node_scratch / "hls-100"
    scratch.mkdir(parents=True)
    scratch.chmod(0o755)
    modes = []

    def make_launcher(gpus, uuids, where):
        modes.append(stat.S_IMODE(os.lstat(where).st_mode))
        return LocalLauncher(gpus=2)

    assert _runner(plan, make_launcher=make_launcher).run() == EXIT_PUBLISHED
    assert modes == [0o700] and not scratch.exists()


def test_the_scratch_of_jobs_the_controller_reports_ended_is_swept_by_exact_id_at_start(tmp_path):
    """A job SIGKILLed before its runner removed its scratch leaves ``hls-<job>`` on the node. At
    its start a runner asks the controller about each such directory of this user by its exact job
    id and removes those whose job has ended, an event each; a running or unknown job's, its own,
    another user's, a link, and a name that holds no job id stay."""
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    root = plan.settings.site.node_scratch
    root.mkdir(parents=True)
    for name in ("hls-1234571", "hls-1234572", "hls-1234573", "hls-1234571x", "hls-abc", "other"):
        (root / name / "roles" / "sandbox").mkdir(parents=True)
        (root / name / "roles" / "sandbox" / "f").write_text("left")
    # An extraction cut short leaves a directory without its owner's write bit.
    (root / "hls-1234571" / "roles" / "sandbox").chmod(0o500)
    (root / "hls-9").symlink_to(root / "other")
    asked: list[list[str]] = []

    def ended_jobs(ids):
        asked.append(sorted(ids))
        return {"1234571", "1234573", "9"} & set(ids)

    mine = os.getuid()
    real_lstat = os.lstat

    def lstat(path, *args, **kwargs):
        st = real_lstat(path, *args, **kwargs)
        if Path(path).name == "hls-1234573":
            # Another user's directory: never this user's to sweep.
            return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, mine + 1, *tuple(st)[5:]))
        return st

    import unittest.mock as mock
    with mock.patch.object(runner_mod.os, "lstat", lstat):
        assert _runner(plan, ended_jobs=ended_jobs).run() == EXIT_PUBLISHED
    assert asked == [["1234571", "1234572"]]
    assert sorted(p.name for p in root.iterdir()) == ["hls-1234571x", "hls-1234572", "hls-1234573", "hls-9",
                                                      "hls-abc", "other"]
    [swept] = [e for e in _events(plan) if e["kind"] == "scratch_swept"]
    assert (swept["job_id"], swept["path"]) == ("1234571", str(root / "hls-1234571"))


def test_a_controller_that_cannot_be_asked_sweeps_nothing(tmp_path):
    plan = noop_plan(tmp_path)
    plan = _store(plan, _sub(plan))
    root = plan.settings.site.node_scratch
    (root / "hls-1234571").mkdir(parents=True)

    def ended_jobs(ids):
        raise OSError("squeue: not found")

    assert _runner(plan, ended_jobs=ended_jobs).run() == EXIT_PUBLISHED
    assert (root / "hls-1234571").is_dir()
    assert [e["kind"] for e in _events(plan) if e["kind"].startswith("scratch_")] == ["scratch_sweep_skipped"]


def test_a_sigterm_removes_the_scratch_once_the_roles_have_stopped_without_waiting_for_the_run_to_end(
        tmp_path, monkeypatch):
    """Slurm sends SIGKILL a KillWait after the SIGTERM (10 s here), which a role's own grace
    can take whole, so the run's end may never come: the shutdown removes the scratch as soon as
    the stage's roles have stopped."""
    decls = noop_two(gpu_kind="H200", sleep_sec=30, stage_time_sec=600)
    plan = noop_plan(tmp_path, decls=decls)
    plan = _store(plan, _sub(plan))
    launcher = LocalLauncher(gpus=2)
    runner = _runner(plan, launcher=launcher)
    released = threading.Event()
    real_conclude = Runner._conclude
    removed_while: list[tuple[bool, bool]] = []
    real_remove = runner_mod.remove_extracted

    def conclude(self, alloc, fault):
        # The run's end, the finally's removal after it, waits as a SIGKILL would cut it.
        released.wait(30)
        return real_conclude(self, alloc, fault)

    def remove(path):
        if path == runner.scratch:
            removed_while.append((threading.current_thread().name == "hls-runner-shutdown",
                                  all(h.poll() is not None for h in launcher.handles)))
        real_remove(path)

    monkeypatch.setattr(Runner, "_conclude", conclude)
    monkeypatch.setattr(runner_mod, "remove_extracted", remove)
    threading.Timer(1.0, runner.terminate).start()
    done: list[int] = []
    main = threading.Thread(target=lambda: done.append(runner.run()))
    main.start()
    deadline = time.monotonic() + 30
    while runner.scratch.exists() or not removed_while:
        assert time.monotonic() < deadline, "the shutdown never removed the scratch"
        time.sleep(0.05)
    assert removed_while[0] == (True, True) and not done
    released.set()
    main.join(30)
    assert done == [EXIT_TERMINATED] and not runner.scratch.exists()
