from __future__ import annotations

from dataclasses import replace

import pytest

from hls.pipeline.plan import AllocationStatus, Outcome, Plan, Submission, submission_tag
from hls.pipeline.planner import StageDecl
from hls.pipeline.reconcile import (Cancel, Cancelled, Observation, Rewire, Rewired, Submit, Submitted, decide,
                                    finished, record)
from hls.spec.resources import ResourceShape
from hls.substrate.base import FaultRecord, JobInfo, JobState
from tests.pipeline.plans import noop_plan

P, R, F = JobState.PENDING, JobState.RUNNING, JobState.FAILED


def _obs(now: float = 2000.0, published=(), jobs=None, tags=None, statuses=None, cancel=False,
         declared=None) -> Observation:
    infos = {j: s if isinstance(s, JobInfo) else JobInfo(j, s) for j, s in (jobs or {}).items()}
    return Observation(now=now, published=frozenset(published), jobs=infos, tags=tags or {}, statuses=statuses or {},
                       cancel_requested=cancel, declared=declared or {})


def _sub(plan: Plan, k: int, n: int = 1, *, job_id: str | None = None, status: str = "submitted",
         after: str | None = None, at: float = 1000.0, error: str | None = None) -> Plan:
    sub = Submission(n=n, tag=submission_tag(plan.instance, k, n), intended_at=at, after_job=after, status=status,
                     job_id=job_id, submitted_at=at if job_id else None, error=error)
    alloc = plan.allocation(k)
    return plan.with_allocation(alloc.model_copy(update={"submissions": [*alloc.submissions, sub]}))


def _void(plan: Plan) -> Plan:
    return plan.model_copy(update={"outcome": Outcome(status="void", reason="test", at=1500.0)})


def _status(plan: Plan, k: int, n: int, *, heartbeat: float, fault: FaultRecord | None = None,
            state: str = "running", job_id: str = "x") -> AllocationStatus:
    return AllocationStatus(instance=plan.instance, allocation=k, submission=n, job_id=job_id, node="g", state=state,
                            started_at=heartbeat, heartbeat_at=heartbeat, end_time=None, stage=None, attempt=None,
                            grant=None, tool_versions=None, swap_max_bytes=None, memory_enforced_by=None, fault=fault,
                            ended_at=None)


def test_the_first_pass_submits_the_first_allocation_and_records_the_intent(tmp_path):
    plan = noop_plan(tmp_path, gpu_policy="split")
    d = decide(plan, _obs())
    assert d.effects == [Submit(0, 1)]
    sub = d.plan.allocation(0).submissions[0]
    assert (sub.status, sub.tag, sub.after_job, sub.job_id) == ("intended", f"hls:{plan.instance}/0/1", None, None)
    assert d.plan.allocation(1).submissions == []


def test_a_dependent_waits_on_a_live_predecessor_job(tmp_path):
    plan = _sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100")
    d = decide(plan, _obs(jobs={"100": P}))
    assert d.effects == [Submit(1, 1)] and d.plan.allocation(1).submissions[0].after_job == "100"


@pytest.mark.parametrize("state", [JobState.UNKNOWN, JobState.COMPLETED])
def test_a_dependent_never_waits_on_a_job_that_may_be_gone(tmp_path, state):
    # Slurm refuses a dependency on a finished job once it is purged.
    # COMPLETED without noop_a published is a failure of allocation 0, retried in its own
    # allocation; allocation 1 is never submitted against job 100.
    plan = _sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100")
    d = decide(plan, _obs(jobs={"100": state}))
    assert d.plan.allocation(1).submissions == []
    assert all(sub.after_job != "100" for a in d.plan.allocations for sub in a.submissions)


def test_a_predecessor_that_published_everything_means_no_dependency(tmp_path):
    plan = _sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100")
    d = decide(plan, _obs(published={"noop_a"}, jobs={"100": JobState.COMPLETED}))
    assert d.effects == [Submit(1, 1)] and d.plan.allocation(1).submissions[0].after_job is None


#: The fault the agent build declares once its commit point wrote the bundle.
COMMITTED = FaultRecord(fault_class="infra", reason="the build failed after its commit point", retryable=False,
                        evidence={"commit_point": "bundle"})


def test_a_job_that_died_after_its_attempt_declared_no_retry_ends_with_that_fault_and_is_not_retried(tmp_path):
    """A runner killed outright after its stage declared that a retry would redo its work
    (the agent build's commit point: an OOM kill of the runner, a node failure) wrote no fault of
    its own, but its attempt's status holds the declaration: the allocation fails with it, never
    retried, and nothing is submitted again. The same end without it is retried."""
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    killed = JobInfo("100", F, exit_code="0:9")
    running = {(0, 1): _status(plan, 0, 1, heartbeat=1990.0, job_id="100")}
    d = decide(plan, _obs(jobs={"100": killed}, statuses=running, declared={(0, 1): COMMITTED}))
    failure = d.plan.allocation(0).failure
    assert (failure.stage, failure.fault.reason, failure.fault.retryable) == ("noop_a", COMMITTED.reason, False)
    assert failure.fault.evidence == {"commit_point": "bundle", "job_state": "FAILED", "exit_code": "0:9"}
    assert d.plan.outcome.status == "void" and len(d.plan.allocations) == 1
    assert not any(isinstance(e, Submit) for e in d.effects)
    retried = decide(plan, _obs(now=5000.0, jobs={"100": killed}, statuses=running))
    assert retried.plan.allocation(0).failure.fault.retryable is True
    assert retried.plan.outcome is None and retried.plan.allocations[-1].retry_of == 0


def test_the_runners_own_fault_comes_before_its_attempts_declaration(tmp_path):
    """A runner that wrote its final status said how the job ended, whatever its attempt declared
    before: a contestant fault the stage decided stands."""
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    contestant = FaultRecord(fault_class="contestant", reason="nothing gradable", retryable=False)
    final = _status(plan, 0, 1, heartbeat=1990.0, job_id="100", fault=contestant, state="failed")
    d = decide(plan, _obs(jobs={"100": F}, statuses={(0, 1): final}, declared={(0, 1): COMMITTED}))
    assert d.plan.allocation(0).failure.fault.reason == "nothing gradable"
    assert d.plan.outcome.status == "failed"


def test_out_of_memory_is_evidence_not_an_ending(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    d = decide(plan, _obs(published={"noop_a", "noop_b"}, jobs={"100": JobState.OUT_OF_MEMORY}))
    assert d.plan.outcome.status == "succeeded" and d.effects == []
    running = decide(plan, _obs(published={"noop_a"}, jobs={"100": R},
                                statuses={(0, 1): _status(plan, 0, 1, heartbeat=1990.0)}))
    assert running.plan == plan and running.effects == []


def test_unknown_counts_as_alive(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    d = decide(plan, _obs(jobs={"100": JobState.UNKNOWN}))
    assert d.plan == plan and d.effects == []


def test_a_failed_allocation_is_retried_from_its_failed_stage(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    d = decide(plan, _obs(published={"noop_a"}, jobs={"100": F}))
    failure = d.plan.allocation(0).failure
    assert failure.stage == "noop_b" and failure.fault.reason == "job 100: the job ended FAILED"
    retry = d.plan.allocation(1)
    assert (retry.stages, retry.retry_of, retry.depends_on, retry.request.gpus) == (["noop_b"], 0, None, 2)
    assert retry.request.time_sec == 600 + 360
    assert d.effects == [Submit(1, 1)]
    assert ("retry", {"stage": "noop_b", "failed_allocation": 0, "allocation": 1, "retry": 1, "of": 2,
                      "reason": failure.fault.reason}) in d.events


def test_a_retry_is_planned_under_the_qos_its_own_time_needs(tmp_path):
    # Both stages ask 30 h, past the site's own QOS; a retry of the second asks its 10 h alone.
    first = StageDecl("first", "noop", ResourceShape(cpus=2, memory_gib=6, time_sec=20 * 3600),
                      {"gpu_roles": 0, "cpu_role": True})
    second = replace(first, name="second", shape=first.shape.model_copy(update={"time_sec": 10 * 3600}))
    plan = noop_plan(tmp_path, decls=[first, second], site={"qos": "normal", "qos_by_hours": {"normal": 24,
                                                                                            "long": 168}})
    assert plan.allocation(0).placement.qos == "long"
    d = decide(_sub(plan, 0, job_id="100"), _obs(published={"first"}, jobs={"100": F}))
    retry = d.plan.allocation(1)
    assert (retry.stages, retry.retry_of, retry.placement.qos) == (["second"], 0, "normal")


@pytest.mark.parametrize("exit_code, reason", [
    ("0:9", "job 100: the runner was killed by signal 9 (SIGKILL)"),
    ("0:6", "job 100: the runner was killed by signal 6 (SIGABRT)"),
    ("0:99", "job 100: the runner was killed by signal 99"),
    ("1:0", "job 100: the runner exited with status 1"),
    ("0:0", "job 100: the job ended FAILED with exit code 0:0"),
    (None, "job 100: the job ended FAILED"),
])
def test_a_failed_jobs_reason_says_how_its_runner_ended(tmp_path, exit_code, reason):
    # scancel --signal=KILL --batch leaves the job FAILED 0:9 and no final status from the runner.
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    d = decide(plan, _obs(published={"noop_a"}, jobs={"100": JobInfo("100", F, exit_code=exit_code)}))
    fault = d.plan.allocation(0).failure.fault
    assert fault.reason == reason and fault.evidence["exit_code"] == (exit_code or "")


def test_the_runners_own_fault_is_the_reason(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    grant = FaultRecord(fault_class="infra", reason="grant mismatch: gpus: requested 2, granted 1", retryable=True)
    d = decide(plan, _obs(jobs={"100": F}, statuses={(0, 1): _status(plan, 0, 1, heartbeat=1.0, fault=grant)}))
    fault = d.plan.allocation(0).failure.fault
    assert fault.reason == grant.reason and fault.evidence["job_state"] == "FAILED"


def test_a_retry_repoints_the_pending_dependent(tmp_path):
    plan = noop_plan(tmp_path, gpu_policy="split")
    plan = _sub(_sub(plan, 0, job_id="100"), 1, job_id="101", after="100", status="submitted")
    d = decide(plan, _obs(jobs={"100": F, "101": JobInfo("101", P, reason="DependencyNeverSatisfied")}))
    assert d.plan.allocation(2).retry_of == 0 and d.plan.allocation(1).depends_on == 2
    assert d.effects == [Submit(2, 1)]
    plan = record(d.plan, [Submitted(2, 1, "102")], 2001.0)
    d = decide(plan, _obs(jobs={"100": F, "101": P, "102": P}))
    assert d.effects == [Rewire(1, 1, "102")] and d.plan.allocation(1).submissions[0].rewire_to == "102"
    plan = record(d.plan, [Rewired(1, 1, "102", ok=True)], 2002.0)
    sub = plan.allocation(1).submissions[0]
    assert (sub.after_job, sub.rewire_to) == ("102", None)
    assert decide(plan, _obs(jobs={"100": F, "101": P, "102": R})).effects == []


def test_retries_run_out_and_the_void_cancels_the_pending_dependent(tmp_path):
    plan = noop_plan(tmp_path, gpu_policy="split", retry={"noop_a": {"max": 1}})
    plan = _sub(_sub(plan, 0, job_id="100"), 1, job_id="101", after="100")
    plan = decide(plan, _obs(jobs={"100": F, "101": P})).plan
    plan = record(plan, [Submitted(2, 1, "102")], 2001.0)
    plan = decide(plan, _obs(jobs={"100": F, "101": P, "102": P})).plan
    plan = record(plan, [Rewired(1, 1, "102", ok=True)], 2002.0)
    d = decide(plan, _obs(jobs={"100": F, "101": JobInfo("101", P, reason="DependencyNeverSatisfied"), "102": F}))
    assert d.plan.outcome.status == "void"
    assert d.plan.outcome.reason == "noop_a: job 102: the job ended FAILED; retries exhausted (1 of 1)"
    assert d.effects == [Cancel(1, 1, "101")]
    assert d.plan.allocation(1).submissions[0].status == "cancel_intended"
    assert not finished(d.plan, _obs(jobs={"100": F, "101": P, "102": F}))
    plan = record(d.plan, [Cancelled(1, 1, ok=True)], 2003.0)
    assert finished(plan, _obs(jobs={"100": F, "101": JobState.CANCELLED, "102": F}))


def test_a_fault_that_is_not_retryable_voids_at_once(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    moved = FaultRecord(fault_class="infra", reason="the checkout changed since planning", retryable=False)
    d = decide(plan, _obs(jobs={"100": F}, statuses={(0, 1): _status(plan, 0, 1, heartbeat=1.0, fault=moved)}))
    assert d.plan.outcome.status == "void" and d.plan.outcome.reason.endswith("; not retryable")
    assert len(d.plan.allocations) == 1


CONTESTANT_OOM = FaultRecord(fault_class="contestant", reason="contestant role server ran out of memory",
                             retryable=False, evidence={"roles": "server"})


def test_a_contestant_fault_ends_the_instance_failed_with_its_fault_and_is_never_retried(tmp_path):
    # A void instance is relaunched, so a contestant fault recorded as void would run again.
    plan = noop_plan(tmp_path, gpu_policy="split", retry={"noop_a": {"max": 2}})
    plan = _sub(_sub(plan, 0, job_id="100"), 1, job_id="101", after="100")
    status = _status(plan, 0, 1, heartbeat=1900.0, fault=CONTESTANT_OOM, state="failed", job_id="100")
    d = decide(plan, _obs(jobs={"100": JobInfo("100", F, exit_code="1:0"), "101": P}, statuses={(0, 1): status}))
    outcome = d.plan.outcome
    assert (outcome.status, outcome.reason) == ("failed", "noop_a: contestant role server ran out of memory")
    assert outcome.fault.fault_class == "contestant" and not outcome.fault.retryable
    assert outcome.fault.evidence == {"roles": "server", "job_state": "FAILED", "exit_code": "1:0"}
    assert len(d.plan.allocations) == 2 and d.effects == [Cancel(1, 1, "101")]


def test_a_void_after_a_stage_failure_carries_the_infra_fault(tmp_path):
    plan = _sub(noop_plan(tmp_path, retry={"noop_a": {"enabled": False}}), 0, job_id="100")
    d = decide(plan, _obs(jobs={"100": F}))
    outcome = d.plan.outcome
    assert outcome.status == "void" and outcome.fault == d.plan.allocation(0).failure.fault
    assert (outcome.fault.fault_class, outcome.fault.reason) == ("infra", "job 100: the job ended FAILED")


def test_disabled_retries_void(tmp_path):
    plan = _sub(noop_plan(tmp_path, retry={"noop_a": {"enabled": False}}), 0, job_id="100")
    assert decide(plan, _obs(jobs={"100": F})).plan.outcome.reason.endswith("; retries are disabled")


def test_the_backoff_delays_the_retry(tmp_path):
    plan = _sub(noop_plan(tmp_path, retry={"noop_a": {"backoff_sec": 300}}), 0, job_id="100")
    plan = decide(plan, _obs(now=2000.0, jobs={"100": F})).plan
    assert len(plan.allocations) == 1 and plan.allocation(0).failure is not None
    assert decide(plan, _obs(now=2299.0, jobs={"100": F})).effects == []
    assert decide(plan, _obs(now=2300.0, jobs={"100": F})).effects == [Submit(1, 1)]


def test_a_running_job_without_a_heartbeat_is_cancelled_and_retried(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    fresh = {(0, 1): _status(plan, 0, 1, heartbeat=1995.0)}
    assert decide(plan, _obs(jobs={"100": R}, statuses=fresh)).effects == []
    stale = {(0, 1): _status(plan, 0, 1, heartbeat=1989.0)}
    d = decide(plan, _obs(jobs={"100": R}, statuses=stale))
    assert d.effects == [Cancel(0, 1, "100"), Submit(1, 1)]
    assert d.plan.allocation(0).failure.fault.reason == "job 100: no heartbeat for 11s while it ran"
    never_wrote = decide(plan, _obs(jobs={"100": JobInfo("100", R, started_at=1980.0)}))
    assert Cancel(0, 1, "100") in never_wrote.effects


def test_a_lost_sbatch_answer_is_found_by_its_tag(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id=None, status="intended")
    tag = plan.allocation(0).submissions[0].tag
    d = decide(plan, _obs(tags={tag: ["100"]}))
    sub = d.plan.allocation(0).submissions[0]
    assert (sub.status, sub.job_id) == ("submitted", "100") and d.effects == []


def test_a_submission_is_absent_only_after_the_window(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id=None, status="intended", at=1000.0)
    tag = plan.allocation(0).submissions[0].tag
    # No answer says nothing, however old the intent is.
    silent = decide(plan, _obs(now=1500.0, tags={tag: None}))
    assert silent.plan == plan and silent.effects == []
    # The first empty answer starts the window; the intent's age does not count.
    first = decide(plan, _obs(now=1500.0, tags={tag: []}))
    sub = first.plan.allocation(0).submissions[0]
    assert (sub.status, sub.missing_since) == ("intended", 1500.0) and first.effects == []
    assert decide(first.plan, _obs(now=1619.0, tags={tag: []})).plan == first.plan
    # Not asking says nothing. An answer that did not come clears the window: it starts again
    # with the next empty answer, so an outage cannot use it up.
    assert decide(first.plan, _obs(now=1700.0)).plan == first.plan
    cleared = decide(first.plan, _obs(now=1700.0, tags={tag: None})).plan
    assert cleared.allocation(0).submissions[0].missing_since is None
    again = decide(cleared, _obs(now=1720.0, tags={tag: []}))
    assert again.plan.allocation(0).submissions[0].missing_since == 1720.0 and again.effects == []
    assert decide(again.plan, _obs(now=1839.0, tags={tag: []})).plan == again.plan
    d = decide(first.plan, _obs(now=1620.0, tags={tag: []}))
    assert d.plan.allocation(0).submissions[0].status == "absent" and d.effects == [Submit(0, 2)]
    assert d.plan.allocation(0).submissions[1].tag == f"hls:{plan.instance}/0/2"


def test_hls_cancel_stops_every_live_job_by_id(tmp_path):
    plan = noop_plan(tmp_path, gpu_policy="split")
    plan = _sub(_sub(plan, 0, job_id="100"), 1, job_id="101", after="100")
    d = decide(plan, _obs(jobs={"100": R, "101": P}, cancel=True))
    assert d.plan.outcome.status == "cancelled"
    assert d.effects == [Cancel(0, 1, "100"), Cancel(1, 1, "101")]


def test_record_folds_in_every_kind_of_answer(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, status="intended"), 1, status="intended")
    got = record(plan, [Submitted(0, 1, None, error="sbatch timed out"),
                        Submitted(1, 1, None, refused=True, dependency=True, error="Job dependency problem")],
                 5.0)
    s0, s1 = got.allocation(0).submissions[0], got.allocation(1).submissions[0]
    assert (s0.status, s0.error) == ("intended", "sbatch timed out")
    assert s1.status == "absent" and got.outcome is None
    voided = record(plan, [Submitted(0, 1, None, refused=True, error="Invalid account")], 5.0)
    assert voided.outcome.reason == "sbatch refused allocation 0: Invalid account"
    # Only the substrate's type says a refusal is about the dependency, never the message text.
    worded = record(plan, [Submitted(0, 1, None, refused=True, error="dependency on a foreign job denied")], 5.0)
    assert worded.outcome.status == "void"
    plan = _sub(noop_plan(tmp_path), 0, job_id="100")
    refused = record(plan, [Rewired(0, 1, "102", ok=False, refused=True, dependency=True,
                                    error="Job dependency problem")], 5.0)
    assert refused.allocation(0).submissions[0].status == "cancel_intended"
    # Any other refusal keeps the intent, counts, and sets the error; the next pass asks again.
    asked = plan.with_allocation(plan.allocation(0).model_copy(update={"submissions": [
        plan.allocation(0).submissions[0].model_copy(update={"rewire_to": "102"})]}))
    other = record(asked, [Rewired(0, 1, "102", ok=False, refused=True, error="Access denied")], 5.0)
    sub = other.allocation(0).submissions[0]
    assert (sub.status, sub.rewire_to, sub.rewire_refusals, sub.error) == ("submitted", "102", 1, "Access denied")
    lost = record(plan, [Cancelled(0, 1, ok=False, error="Socket timed out")], 5.0)
    assert lost.allocation(0).submissions[0].error == "Socket timed out"


# A dependent that waits for a job that ended badly never starts, and Slurm keeps it forever.


@pytest.mark.parametrize("after_state, reason, cancelled", [
    (JobState.FAILED, None, True),
    (JobState.TIMEOUT, "Dependency", True),
    (JobState.OUT_OF_MEMORY, "DependencyNeverSatisfied", True),
    (JobState.UNKNOWN, "DependencyNeverSatisfied", True),
    (JobState.COMPLETED, "Dependency", False),
    (JobState.RUNNING, None, False),
    (JobState.UNKNOWN, "Dependency", False),
])
def test_a_dependent_stuck_behind_a_job_that_ended_badly_is_cancelled(tmp_path, after_state, reason, cancelled):
    # noop_a published, so allocation 1 needs nothing from job 100 any more.
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    jobs = {"100": after_state, "101": JobInfo("101", P, reason=reason)}
    d = decide(plan, _obs(published={"noop_a"}, jobs=jobs))
    assert d.effects == ([Cancel(1, 1, "101")] if cancelled else [])
    assert d.plan.allocation(1).submissions[0].status == ("cancel_intended" if cancelled else "submitted")


def test_the_cancelled_dependent_is_submitted_again_without_a_dependency_and_without_a_retry(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    jobs = {"100": JobState.OUT_OF_MEMORY, "101": JobInfo("101", P, reason="DependencyNeverSatisfied")}
    plan = record(decide(plan, _obs(published={"noop_a"}, jobs=jobs)).plan, [Cancelled(1, 1, ok=True)], 2001.0)
    d = decide(plan, _obs(published={"noop_a"}, jobs={**jobs, "101": JobState.CANCELLED}))
    assert d.effects == [Submit(1, 2)] and len(d.plan.allocations) == 2
    assert (d.plan.allocation(1).submissions[1].n, d.plan.allocation(1).submissions[1].after_job) == (2, None)


def test_a_dependent_whose_predecessor_published_is_not_rewired_to_it(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    plan = record(decide(plan, _obs(jobs={"100": F, "101": P})).plan, [Submitted(2, 1, "102")], 2001.0)
    # The retry published and its job is still ending; the dependent waits for the failed job 100.
    stuck = JobInfo("101", P, reason="DependencyNeverSatisfied")
    d = decide(plan, _obs(published={"noop_a"}, jobs={"100": F, "101": stuck, "102": R}))
    assert d.effects == [Cancel(1, 1, "101")]


def test_a_retry_that_already_ended_is_never_a_rewire_target(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    plan = record(decide(plan, _obs(jobs={"100": F, "101": P})).plan, [Submitted(2, 1, "102")], 2001.0)
    stuck = JobInfo("101", P, reason="DependencyNeverSatisfied")
    d = decide(plan, _obs(jobs={"100": F, "101": stuck, "102": F}))
    assert not any(isinstance(e, Rewire) for e in d.effects)
    assert d.effects == [Submit(3, 1)] and d.plan.allocation(1).depends_on == 3


# Each stage has its own retries, and a failed predecessor blocks its dependents.


def test_retries_are_counted_per_stage(tmp_path):
    plan = _sub(noop_plan(tmp_path, retry={"noop_b": {"max": 1}}), 0, job_id="100")
    d = decide(plan, _obs(jobs={"100": F}))
    assert d.plan.allocation(1).stages == ["noop_a", "noop_b"] and d.plan.allocation(1).retry_of == 0
    plan = record(d.plan, [Submitted(1, 1, "101")], 2001.0)
    # The retry gets through noop_a and fails at noop_b, which has had no retry yet.
    d = decide(plan, _obs(published={"noop_a"}, jobs={"100": F, "101": F}))
    assert d.plan.outcome is None and d.plan.allocation(2).retry_of == 1 and d.plan.allocation(2).stages == ["noop_b"]


def test_a_failed_predecessor_waiting_out_its_backoff_blocks_the_dependent(tmp_path):
    plan = noop_plan(tmp_path, gpu_policy="split", retry={"noop_a": {"backoff_sec": 300}})
    plan = decide(_sub(plan, 0, job_id="100"), _obs(now=2000.0, jobs={"100": F})).plan
    # The scheduler's answer lags and still lists the job as running.
    d = decide(plan, _obs(now=2100.0, jobs={"100": R}))
    assert d.effects == [] and d.plan.allocation(1).submissions == []


def test_a_retry_of_a_dependent_allocation_keeps_its_dependency(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    heartbeat = {(0, 1): _status(plan, 0, 1, heartbeat=1995.0, job_id="100")}
    # Job 101 was cancelled outside hls while job 100 still runs.
    d = decide(plan, _obs(jobs={"100": R, "101": JobState.CANCELLED}, statuses=heartbeat))
    retry = d.plan.allocation(2)
    assert (retry.stages, retry.retry_of, retry.depends_on) == (["noop_b"], 1, 0)
    assert d.effects == [Submit(2, 1)] and retry.submissions[0].after_job == "100"


# Lookups by tag.


def test_a_cancel_intended_submission_found_by_tag_is_cancelled_by_id(tmp_path):
    plan = _void(_sub(noop_plan(tmp_path), 0, status="cancel_intended"))
    d = decide(plan, _obs(tags={plan.allocation(0).submissions[0].tag: ["100"]}))
    sub = d.plan.allocation(0).submissions[0]
    assert (sub.status, sub.job_id) == ("cancel_intended", "100") and d.effects == [Cancel(0, 1, "100")]


def test_a_cancel_intended_submission_no_job_carries_ends_cancelled(tmp_path):
    plan = _void(_sub(noop_plan(tmp_path), 0, status="cancel_intended", at=1000.0))
    tag = plan.allocation(0).submissions[0].tag
    first = decide(plan, _obs(now=1100.0, tags={tag: []}))
    assert first.plan.allocation(0).submissions[0].status == "cancel_intended"
    second = decide(first.plan, _obs(now=1219.0, tags={tag: []}))
    assert second.plan.allocation(0).submissions[0].status == "cancel_intended"
    third = decide(first.plan, _obs(now=1220.0, tags={tag: []}))
    assert third.plan.allocation(0).submissions[0].status == "cancelled" and third.effects == []
    # A job could still turn up under the tag, so the instance is done one more window later.
    assert not finished(third.plan, _obs(now=1339.0)) and finished(third.plan, _obs(now=1340.0))


@pytest.mark.parametrize("status", ["absent", "cancelled"])
def test_a_job_that_turns_up_under_an_absent_tag_is_cancelled_by_id(tmp_path, status):
    plan = _sub(_sub(noop_plan(tmp_path), 0, status=status), 0, n=2, job_id="101")
    tag = plan.allocation(0).submissions[0].tag
    alive = {(0, 2): _status(plan, 0, 2, heartbeat=1999.0, job_id="101")}
    d = decide(plan, _obs(jobs={"101": R}, tags={tag: ["100"]}, statuses=alive))
    first, second = d.plan.allocation(0).submissions
    assert (first.status, first.job_id) == ("cancel_intended", "100") and d.effects == [Cancel(0, 1, "100")]
    assert (second.status, second.job_id) == ("submitted", "101") and d.plan.outcome is None
    ended = decide(d.plan, _obs(jobs={"100": JobState.CANCELLED, "101": R}, statuses=alive))
    assert ended.plan.allocation(0).submissions[0].status == "cancelled"


def test_an_absent_submission_nothing_carries_costs_nothing(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path), 0, status="absent"), 0, n=2, job_id="101")
    tag = plan.allocation(0).submissions[0].tag
    alive = {(0, 2): _status(plan, 0, 2, heartbeat=1999.0, job_id="101")}
    d = decide(plan, _obs(jobs={"101": R}, tags={tag: []}, statuses=alive))
    assert d.plan == plan and d.effects == []


def test_a_tag_that_several_jobs_carry_keeps_the_first_and_cancels_the_rest(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, status="intended")
    tag = plan.allocation(0).submissions[0].tag
    d = decide(plan, _obs(tags={tag: ["100", "103", "104"]}))
    sub = d.plan.allocation(0).submissions[0]
    assert (sub.status, sub.job_id, sub.strays) == ("submitted", "100", ["103", "104"])
    assert d.effects == [Cancel(0, 1, "103"), Cancel(0, 1, "104")]
    assert ("stray_job", {"allocation": 0, "submission": 1, "job_id": "103"}) in d.events
    # The answer for a stray says nothing about the submission's own job.
    got = record(d.plan, [Cancelled(0, 1, ok=True, job_id="103")], 5.0)
    assert got.allocation(0).submissions[0].status == "submitted"
    own = record(d.plan, [Cancelled(0, 1, ok=True, job_id="100")], 5.0)
    assert own.allocation(0).submissions[0].status == "cancelled"


def test_the_runners_status_names_the_job_when_the_scheduler_cannot(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, status="intended")
    tag = plan.allocation(0).submissions[0].tag
    status = _status(plan, 0, 1, heartbeat=1999.0, job_id="100")
    for tags in ({tag: None}, {tag: []}, {}):
        d = decide(plan, _obs(tags=tags, statuses={(0, 1): status}))
        sub = d.plan.allocation(0).submissions[0]
        assert (sub.status, sub.job_id, sub.missing_since) == ("submitted", "100", None) and d.effects == []


# Three lost or refused submissions in a row stop the instance.


def test_three_absent_submissions_in_a_row_void_the_instance(tmp_path):
    plan = noop_plan(tmp_path)
    for n in (1, 2):
        plan = _sub(plan, 0, n=n, status="absent", error="Job dependency problem")
    d = decide(plan, _obs())
    assert d.effects == [Submit(0, 3)] and d.plan.outcome is None
    plan = _sub(plan, 0, n=3, status="absent", error="Job dependency problem")
    d = decide(plan, _obs())
    assert d.effects == [] and d.plan.outcome.status == "void"
    assert d.plan.outcome.reason == ("allocation 0: submissions 1, 2, 3 were refused or lost; "
                                     "last error: Job dependency problem")


def test_absent_submissions_that_are_not_in_a_row_do_not_void(tmp_path):
    plan = noop_plan(tmp_path)
    for n, (status, job_id) in enumerate([("absent", None), ("absent", None), ("cancelled", "100"), ("absent", None)],
                                         start=1):
        plan = _sub(plan, 0, n=n, status=status, job_id=job_id)
    d = decide(plan, _obs(jobs={"100": JobState.CANCELLED}))
    assert d.effects == [Submit(0, 5)] and d.plan.outcome is None


# What happens once the instance ended.


def test_success_cancels_a_pending_job_and_leaves_a_running_one(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    d = decide(plan, _obs(published={"noop_a", "noop_b"}, jobs={"100": R, "101": P}))
    assert d.plan.outcome.status == "succeeded" and d.effects == [Cancel(1, 1, "101")]
    assert [a.submissions[0].status for a in d.plan.allocations] == ["submitted", "cancel_intended"]
    assert not finished(d.plan, _obs(jobs={"100": R, "101": P}))
    done = record(d.plan, [Cancelled(1, 1, ok=True)], 2001.0)
    assert finished(done, _obs(jobs={"100": R, "101": JobState.CANCELLED}))


def test_success_waits_for_an_outstanding_cancel(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, job_id="100", status="cancel_intended")
    plan = plan.model_copy(update={"outcome": Outcome(status="succeeded", reason="x", at=1500.0)})
    assert not finished(plan, _obs(jobs={"100": R}))
    d = decide(plan, _obs(published={"noop_a", "noop_b"}, jobs={"100": R}))
    assert d.effects == [Cancel(0, 1, "100")]
    ended = _obs(jobs={"100": JobState.CANCELLED})
    assert finished(decide(plan, ended).plan, ended)


def test_a_void_instance_is_not_finished_while_a_submission_has_no_job_id(tmp_path):
    plan = _void(_sub(noop_plan(tmp_path), 0, status="intended"))
    assert not finished(plan, _obs())


def test_a_cancel_intended_whose_job_ended_is_cancelled_without_another_scancel(tmp_path):
    plan = _void(_sub(noop_plan(tmp_path), 0, job_id="100", status="cancel_intended"))
    d = decide(plan, _obs(jobs={"100": JobState.CANCELLED}))
    assert d.effects == [] and d.plan.allocation(0).submissions[0].status == "cancelled"
    assert decide(plan, _obs(jobs={"100": R})).effects == [Cancel(0, 1, "100")]


def test_no_sbatch_is_issued_for_an_instance_that_ends_in_the_same_pass(tmp_path):
    # Allocation 0 would be submitted again; allocation 1 fails for good in the same pass.
    plan = noop_plan(tmp_path, gpu_policy="split")
    plan = plan.with_allocation(plan.allocation(1).model_copy(update={"depends_on": None}))
    plan = _sub(_sub(plan, 0, status="absent"), 1, job_id="101")
    moved = FaultRecord(fault_class="infra", reason="the checkout changed", retryable=False)
    d = decide(plan, _obs(jobs={"101": F}, statuses={(1, 1): _status(plan, 1, 1, heartbeat=1.0, fault=moved)}))
    assert d.plan.outcome.status == "void"
    assert not any(isinstance(e, Submit) for e in d.effects)
    assert [s.status for s in d.plan.allocation(0).submissions] == ["absent", "cancelled"]
    assert finished(d.plan, _obs(jobs={"101": F}))


# Rewires that stay refused, stuck dependents, strays, and when an instance is done.


def _retry_submitted(tmp_path):
    """Allocation 1's pending job 101 waits for job 100, which failed; allocation 0 was retried as job 102."""
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    return record(decide(plan, _obs(jobs={"100": F, "101": P})).plan, [Submitted(2, 1, "102")], 2001.0)


STUCK = JobInfo("101", P, reason="DependencyNeverSatisfied")


def test_a_rewire_refused_for_another_reason_is_asked_again_with_no_plan_change_and_then_given_up(tmp_path):
    jobs = {"100": F, "101": STUCK, "102": P}
    d = decide(_retry_submitted(tmp_path), _obs(jobs=jobs))
    assert d.effects == [Rewire(1, 1, "102")]
    plan = d.plan
    refused = Rewired(1, 1, "102", ok=False, refused=True, error="Access denied")
    for count in (1, 2):
        plan = record(plan, [refused], 2002.0 + count)
        sub = plan.allocation(1).submissions[0]
        assert (sub.rewire_to, sub.rewire_refusals, sub.error) == ("102", count, "Access denied")
        again = decide(plan, _obs(jobs=jobs))
        assert again.plan == plan and again.effects == [Rewire(1, 1, "102")]
    plan = record(plan, [refused], 2005.0)
    d = decide(plan, _obs(jobs=jobs))
    assert d.effects == [Cancel(1, 1, "101")] and d.plan.allocation(1).submissions[0].status == "cancel_intended"
    plan = record(d.plan, [Cancelled(1, 1, ok=True)], 2006.0)
    d = decide(plan, _obs(jobs={**jobs, "101": JobState.CANCELLED}))
    assert d.effects == [Submit(1, 2)] and d.plan.allocation(1).submissions[1].after_job == "102"


def test_a_rewire_nobody_answered_is_not_counted_as_refused(tmp_path):
    plan = decide(_retry_submitted(tmp_path), _obs(jobs={"100": F, "101": STUCK, "102": P})).plan
    for _ in range(5):
        plan = record(plan, [Rewired(1, 1, "102", ok=False, error="Socket timed out")], 2002.0)
    sub = plan.allocation(1).submissions[0]
    assert (sub.rewire_to, sub.rewire_refusals) == ("102", 0)


def test_a_new_rewire_target_starts_the_count_again(tmp_path):
    jobs = {"100": F, "101": STUCK, "102": P}
    plan = decide(_retry_submitted(tmp_path), _obs(jobs=jobs)).plan
    refused = Rewired(1, 1, "102", ok=False, refused=True, error="Access denied")
    plan = record(record(plan, [refused], 2002.0), [refused], 2003.0)
    assert plan.allocation(1).submissions[0].rewire_refusals == 2
    # The retry failed too and was retried as job 103.
    d = decide(plan, _obs(jobs={**jobs, "102": F}))
    plan = record(d.plan, [Submitted(3, 1, "103")], 2004.0)
    d = decide(plan, _obs(jobs={**jobs, "102": F, "103": P}))
    sub = d.plan.allocation(1).submissions[0]
    assert d.effects == [Rewire(1, 1, "103")] and (sub.rewire_to, sub.rewire_refusals) == ("103", 0)
    ok = record(d.plan, [Rewired(1, 1, "103", ok=True)], 2005.0)
    assert ok.allocation(1).submissions[0].rewire_refusals == 0


@pytest.mark.parametrize("reason, cancelled", [("DependencyNeverSatisfied", True), ("Dependency", False)])
def test_while_a_rewire_is_intended_only_the_reason_says_the_dependent_is_stuck(tmp_path, reason, cancelled):
    # Slurm may have applied the rewire to job 102 though its answer was lost, so after_job (100) is stale.
    plan = decide(_retry_submitted(tmp_path), _obs(jobs={"100": F, "101": STUCK, "102": P})).plan
    assert plan.allocation(1).submissions[0].rewire_to == "102"
    jobs = {"100": F, "101": JobInfo("101", P, reason=reason), "102": JobState.COMPLETED}
    d = decide(plan, _obs(published={"noop_a"}, jobs=jobs))
    assert d.effects == ([Cancel(1, 1, "101")] if cancelled else [])


def test_a_dependent_is_judged_by_the_job_it_waits_for_not_by_its_predecessors_retry(tmp_path):
    # A site whose pending reason says nothing: the state of job 100 does.
    quiet = JobInfo("101", P, reason="Dependency")
    jobs = {"100": F, "101": quiet, "102": JobState.COMPLETED}
    d = decide(_retry_submitted(tmp_path), _obs(published={"noop_a"}, jobs=jobs))
    assert d.effects == [Cancel(1, 1, "101")]


def test_the_absence_note_keeps_the_last_error(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, status="intended", at=1000.0, error="ValueError: --gpus=h200:99 is too many")
    tag = plan.allocation(0).submissions[0].tag
    first = decide(plan, _obs(now=1100.0, tags={tag: []}))
    d = decide(first.plan, _obs(now=1220.0, tags={tag: []}))
    error = d.plan.allocation(0).submissions[0].error
    assert "--gpus=h200:99 is too many" in error and f"no job carries tag {tag}" in error


def test_strays_are_recorded_before_they_are_cancelled_and_cancelled_until_they_ended(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, status="intended")
    tag = plan.allocation(0).submissions[0].tag
    d = decide(plan, _obs(tags={tag: ["100", "103", "104"]}))
    assert d.plan.allocation(0).submissions[0].strays == ["103", "104"]
    # An orchestrator that died after writing this revision cancels them when it resumes.
    resumed = decide(d.plan, _obs(jobs={"100": R}))
    assert resumed.effects == [Cancel(0, 1, "103"), Cancel(0, 1, "104")]
    partly = decide(d.plan, _obs(jobs={"100": R, "103": JobState.CANCELLED, "104": R}))
    assert partly.plan.allocation(0).submissions[0].strays == ["104"] and partly.effects == [Cancel(0, 1, "104")]
    ended = decide(partly.plan, _obs(jobs={"100": R, "104": JobState.CANCELLED}))
    assert ended.plan.allocation(0).submissions[0].strays == [] and ended.effects == []
    # A void instance is not finished while a stray lives.
    over = _obs(jobs={"100": JobState.CANCELLED})
    assert not finished(_void(partly.plan), over) and finished(_void(ended.plan), over)


@pytest.mark.parametrize("status", ["absent", "cancelled"])
def test_a_finished_instance_waits_two_windows_for_a_job_that_may_still_turn_up(tmp_path, status):
    plan = _void(_sub(noop_plan(tmp_path), 0, status=status))
    lost = plan.allocation(0).submissions[0].model_copy(update={"missing_since": 1100.0})
    plan = plan.with_allocation(plan.allocation(0).model_copy(update={"submissions": [lost]}))
    assert not finished(plan, _obs(now=1339.0)) and finished(plan, _obs(now=1340.0))
    # Nothing to wait for when sbatch refused: no job can turn up.
    refused = _void(_sub(noop_plan(tmp_path), 0, status=status, error="Invalid account"))
    assert finished(refused, _obs(now=1101.0))


def test_success_waits_for_a_queued_job_whose_state_is_unknown(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, job_id="101", after="100")
    published = {"noop_a", "noop_b"}
    unknown = _obs(published=published, jobs={"100": R, "101": JobState.UNKNOWN})
    d = decide(plan, unknown)
    assert d.plan.outcome.status == "succeeded" and d.effects == [] and not finished(d.plan, unknown)
    assert decide(plan, _obs(published=published, jobs={"100": R, "101": P})).effects == [Cancel(1, 1, "101")]
    assert finished(d.plan, _obs(published=published, jobs={"100": R, "101": R}))


def test_success_waits_for_an_sbatch_answer_nobody_resolved(tmp_path):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, status="intended")
    obs = _obs(published={"noop_a", "noop_b"}, jobs={"100": R})
    d = decide(plan, obs)
    assert d.plan.outcome.status == "succeeded" and not finished(d.plan, obs)


def test_the_runners_status_names_the_job_first_and_the_tag_adds_only_other_jobs(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, status="intended")
    tag = plan.allocation(0).submissions[0].tag
    own = _status(plan, 0, 1, heartbeat=1999.0, job_id="100")
    # The status and the tag agree on job 100; job 103 is the stray, and job 100 is not.
    d = decide(plan, _obs(tags={tag: ["100", "103"]}, statuses={(0, 1): own}))
    sub = d.plan.allocation(0).submissions[0]
    assert (sub.job_id, sub.strays) == ("100", ["103"]) and d.effects == [Cancel(0, 1, "103")]
    # The status names job 101; the tag found job 100 as well, which is then the stray.
    other = _status(plan, 0, 1, heartbeat=1999.0, job_id="101")
    d = decide(plan, _obs(tags={tag: ["100"]}, statuses={(0, 1): other}))
    sub = d.plan.allocation(0).submissions[0]
    assert (sub.job_id, sub.strays) == ("101", ["100"]) and d.effects == [Cancel(0, 1, "100")]


def test_a_job_found_clears_the_window_that_was_running(tmp_path):
    plan = _sub(noop_plan(tmp_path), 0, status="intended", at=1000.0)
    tag = plan.allocation(0).submissions[0].tag
    waiting = decide(plan, _obs(now=1100.0, tags={tag: []})).plan
    assert waiting.allocation(0).submissions[0].missing_since == 1100.0
    found = decide(waiting, _obs(now=1150.0, tags={tag: ["100"]})).plan
    sub = found.allocation(0).submissions[0]
    assert (sub.job_id, sub.missing_since) == ("100", None)


def test_a_rewire_that_finally_succeeds_clears_the_count(tmp_path):
    plan = decide(_retry_submitted(tmp_path), _obs(jobs={"100": F, "101": STUCK, "102": P})).plan
    refused = Rewired(1, 1, "102", ok=False, refused=True, error="Access denied")
    plan = record(plan, [refused], 2002.0)
    assert plan.allocation(1).submissions[0].rewire_refusals == 1
    sub = record(plan, [Rewired(1, 1, "102", ok=True)], 2003.0).allocation(1).submissions[0]
    assert (sub.after_job, sub.rewire_to, sub.rewire_refusals, sub.error) == ("102", None, 0, None)


@pytest.mark.parametrize("outcome", ["succeeded", "void", "cancelled"])
def test_a_live_submission_without_a_job_id_holds_an_ended_instance_whatever_the_outcome(tmp_path, outcome):
    plan = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, status="intended")
    plan = plan.model_copy(update={"outcome": Outcome(status=outcome, reason="test", at=1500.0)})
    obs = _obs(published={"noop_a", "noop_b"}, jobs={"100": JobState.COMPLETED})
    assert not finished(plan, obs)
    gone = _sub(_sub(noop_plan(tmp_path, gpu_policy="split"), 0, job_id="100"), 1, status="absent")
    gone = gone.model_copy(update={"outcome": Outcome(status=outcome, reason="test", at=1500.0)})
    assert finished(gone, obs)


def test_a_failed_instance_cancels_its_running_jobs(tmp_path):
    # Only a succeeded instance lets its running jobs end; a failed one stops them by id.
    plan = noop_plan(tmp_path, gpu_policy="split", retry={"noop_a": {"max": 2}})
    plan = _sub(_sub(plan, 0, job_id="100"), 1, job_id="101")
    status = _status(plan, 0, 1, heartbeat=1900.0, fault=CONTESTANT_OOM, state="failed", job_id="100")
    d = decide(plan, _obs(jobs={"100": JobInfo("100", F, exit_code="1:0"), "101": R}, statuses={(0, 1): status}))
    assert d.plan.outcome.status == "failed" and d.effects == [Cancel(1, 1, "101")]
