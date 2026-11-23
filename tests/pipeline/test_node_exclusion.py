"""A retry after a node's own grant check failed goes to another node: the runner's grant fault
names its node, and the retry's submissions exclude it with every node the failed allocation
already excluded. A node whose own state fails the check, such as a disk another user filled,
fails every allocation placed on it, so a retry there would fail the same way."""

from __future__ import annotations

import pytest

from hls.pipeline.events import read_events
from hls.pipeline.orchestrator import Orchestrator
from hls.pipeline.plan import Plan, PlanStore
from hls.substrate.base import FaultRecord, JobState
from hls.substrate.slurm.scheduler import submit_argv
from tests.pipeline.fake_scheduler import FakeScheduler
from tests.pipeline.test_orchestrator import Clock, _advance, _create, _heartbeat


def _grant_fault(node: str) -> FaultRecord:
    """The fault the runner's grant check writes on *node*."""
    return FaultRecord(fault_class="infra", reason="grant mismatch: disk_free_gib: requested 216, granted 121.0",
                       retryable=True, evidence={"check": "grant", "node": node})


def _fail(orch: Orchestrator, sched: FakeScheduler, clock: Clock, k: int, job_id: str,
          fault: FaultRecord | None) -> Plan:
    """Job *job_id* of allocation *k* ends FAILED, its runner's status holding *fault* (none when
    None); the next pass retries it. The plan after that pass."""
    if fault is not None:
        plan = PlanStore(orch.store, orch.layout).latest()
        _heartbeat(plan, k, 1, clock.t, job_id=job_id, fault=fault)
    sched.finish(job_id, JobState.FAILED)
    plan, _, _ = orch.step()
    return plan


def _excludes(sched: FakeScheduler, job_id: str) -> list[str]:
    """The ``--exclude`` arguments job *job_id* was submitted with."""
    return [a for a in submit_argv(sched.jobs[job_id].request) if a.startswith("--exclude=")]


def test_a_grant_check_failure_excludes_its_node_from_the_retry(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    assert _excludes(sched, "100") == []
    plan = _fail(orch, sched, clock, 0, "100", _grant_fault("n1"))
    retry = plan.allocation(1)
    assert (retry.retry_of, retry.placement.exclude, retry.current().job_id) == (0, ["n1"], "101")
    assert _excludes(sched, "101") == ["--exclude=n1"]
    # The revision on the store holds the list, so an orchestrator that takes over submits the same.
    assert PlanStore(orch.store, orch.layout).latest().allocation(1).placement.exclude == ["n1"]
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    [excluded] = [e for e in events if e["kind"] == "node_excluded"]
    assert (excluded["allocation"], excluded["nodes"]) == (1, ["n1"])


def test_a_retrys_recorded_sbatch_line_carries_its_exclude(tmp_path):
    """The submitted event and the plan's submission record the line sbatch was given, so the
    exclusion is on record without the scheduler's accounting, which ages out."""
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    plan = _fail(orch, sched, clock, 0, "100", _grant_fault("n1"))
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    lines = {e["allocation"]: e["argv"] for e in events if e["kind"] == "submitted"}
    assert lines == {k: list(sched.submit_argv(sched.jobs[job].request)) for k, job in ((0, "100"), (1, "101"))}
    assert [a for a in lines[1] if a.startswith("--exclude")] == ["--exclude=n1"]
    assert not [a for a in lines[0] if a.startswith("--exclude")]
    assert [plan.allocation(k).current().argv for k in (0, 1)] == [lines[0], lines[1]]


def test_a_second_grant_check_failure_excludes_both_nodes(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    _fail(orch, sched, clock, 0, "100", _grant_fault("n1"))
    plan = _fail(orch, sched, clock, 1, "101", _grant_fault("n2"))
    assert (plan.allocation(2).retry_of, plan.allocation(2).placement.exclude) == (1, ["n1", "n2"])
    assert _excludes(sched, "102") == ["--exclude=n1,n2"]


def test_a_node_failing_twice_is_excluded_once(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    _fail(orch, sched, clock, 0, "100", _grant_fault("n1"))
    plan = _fail(orch, sched, clock, 1, "101", _grant_fault("n1"))
    assert plan.allocation(2).placement.exclude == ["n1"] and _excludes(sched, "102") == ["--exclude=n1"]


@pytest.mark.parametrize("fault", [
    FaultRecord(fault_class="infra", reason="the role gpu0 was killed by the kernel's OOM killer", retryable=True,
                evidence={"ending": "oom_killed", "oom_kills": "1"}),
    FaultRecord(fault_class="infra", reason="grant mismatch: gpus: requested 2, granted 1", retryable=True,
                evidence={"check": "grant", "node": "g1,g2"}),
    FaultRecord(fault_class="infra", reason="the checkout changed", retryable=True, evidence={"node": "n1"}),
    None,
], ids=["oom", "unusable_node_name", "node_without_the_grant_check", "no_runner_status"])
def test_a_fault_of_another_kind_excludes_nothing(tmp_path, fault):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    plan = _fail(orch, sched, clock, 0, "100", fault)
    assert plan.allocation(1).retry_of == 0 and plan.allocation(1).placement.exclude == []
    assert _excludes(sched, "101") == []


def test_a_lost_heartbeat_excludes_nothing(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    _heartbeat(plan, 0, 1, clock.t, job_id="100")
    _advance(orch, clock, 30)
    plan, _, _ = orch.step()
    assert plan.allocation(0).failure.fault.reason.endswith("while it ran")
    assert plan.allocation(1).placement.exclude == [] and _excludes(sched, plan.allocation(1).current().job_id) == []


def test_a_retry_keeps_the_nodes_the_site_excludes(tmp_path):
    # sbatch keeps only the last --exclude it is given, so the site's and the retry's are one.
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, site={"sbatch_extra": ["--exclude=node[11,15]"]})
    orch.step()
    assert _excludes(sched, "100") == ["--exclude=node[11,15]"]
    _fail(orch, sched, clock, 0, "100", _grant_fault("node01"))
    assert _excludes(sched, "101") == ["--exclude=node[11,15],node01"]
