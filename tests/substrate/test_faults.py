from __future__ import annotations

from dataclasses import replace

import pytest
from pydantic import ValidationError

from hls.substrate.base import FaultRecord
from hls.substrate.faults import RoleOutcome, SplitEvidence, classify_build_end, classify_role
from hls.substrate.nproc import CeilingRule, PoolAtEnd, PoolSample


def test_a_clean_exit_is_no_fault():
    assert classify_role(RoleOutcome("driver", "harness", 0, "exited")) is None


def test_evidence_explains_an_end_only_within_its_age_before_it():
    """The one age rule for the contestant's evidence: a kill, an OOM or a pids limit
    reached explains an end at most ``OOM_EVIDENCE_MAX_AGE_SEC`` before it, inclusive; evidence
    first read after the end, as a count is dated by the read that shows it, is the end's own;
    undated evidence explains nothing."""
    from hls.substrate.faults import OOM_EVIDENCE_MAX_AGE_SEC, within_evidence_age

    assert within_evidence_age(100.0, 100.0 + OOM_EVIDENCE_MAX_AGE_SEC)
    assert not within_evidence_age(100.0, 100.0 + OOM_EVIDENCE_MAX_AGE_SEC + 0.001)
    assert within_evidence_age(105.0, 100.0)
    assert not within_evidence_age(None, 100.0)


@pytest.mark.parametrize("outcome, fault_class, retryable", [
    (RoleOutcome("driver", "harness", 1, "exited"), "infra", True),
    (RoleOutcome("driver", "harness", 137, "memory_limit"), "infra", True),
    # Slurm marks a step OUT_OF_MEMORY after any process in it was OOM-killed, even when the
    # main process survived; for a harness role that is still infra.
    (RoleOutcome("driver", "harness", 0, "exited", oom_kills=1), "infra", True),
    (RoleOutcome("reference", "harness", None, "time_limit"), "infra", True),
    (RoleOutcome("artifact", "contestant", 137, "exited", oom_kills=1, own_oom=True, job_oom=False), "contestant",
     False),
    (RoleOutcome("artifact", "contestant", 137, "memory_limit"), "contestant", False),
    (RoleOutcome("artifact", "contestant", 97, "grant_mismatch"), "infra", True),
    (RoleOutcome("artifact", "contestant", None, "launch_failed"), "infra", True),
    (RoleOutcome("artifact", "contestant", None, "stopped"), "infra", True),
    # The scheduler ended the step, or its end could not be confirmed: never the contestant's.
    (RoleOutcome("artifact", "contestant", 137, "scheduler_ended"), "infra", True),
    (RoleOutcome("driver", "harness", 137, "scheduler_ended"), "infra", True),
    # The container runtime failed before the command started: never the contestant's exit.
    (RoleOutcome("artifact", "contestant", 255, "runtime_failed"), "infra", True),
    (RoleOutcome("driver", "harness", 1, "runtime_failed"), "infra", True),
])
def test_fault_classes(outcome, fault_class, retryable):
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == (fault_class, retryable)
    assert fault.evidence["ending"] == outcome.ending


@pytest.mark.parametrize("outcome", [
    RoleOutcome("artifact", "contestant", 1, "exited"),
    RoleOutcome("artifact", "contestant", None, "time_limit"),
])
def test_other_contestant_failures_are_results(outcome):
    assert classify_role(outcome) is None


def test_a_contestant_fault_is_never_retryable():
    # Contestant faults are scored results and are never retried.
    with pytest.raises(ValidationError, match="a contestant fault is never retryable"):
        FaultRecord(fault_class="contestant", reason="x", retryable=True)
    assert not FaultRecord(fault_class="contestant", reason="x", retryable=False).retryable


@pytest.mark.parametrize("own_oom, job_oom, fault_class, why", [
    (True, False, "contestant", None),
    (True, True, "infra", "the job's own memory limit was reached"),
    (True, None, "infra", "it is unknown whether the job's own memory limit was reached"),
    (False, False, "infra", "its own memory limit was not reached"),
    (False, True, "infra", "the job's own memory limit was reached"),
    (None, False, "infra", "it is unknown whether its own memory limit was reached"),
    (None, None, "infra", "it is unknown whether its own memory limit was reached"),
])
def test_a_kernel_oom_kill_of_a_contestant_role_is_its_own_only_when_its_own_limit_alone_was_reached(
        own_oom, job_oom, fault_class, why):
    # The kernel counts a kill by any OOM killer in the victim's step, and picks a contestant
    # process (oom_score_adj 1000) first: on the node, under a limit above the job, or under
    # the job's own limit. Only the step's own limit, with the job's not reached, is certain.
    fault = classify_role(RoleOutcome("artifact", "contestant", 137, "exited", oom_kills=1, own_oom=own_oom,
                                      job_oom=job_oom))
    assert (fault.fault_class, fault.retryable) == (fault_class, fault_class == "infra")
    assert (fault.evidence["own_oom"], fault.evidence["job_oom"]) == (str(own_oom), str(job_oom))
    if why is not None:
        assert fault.reason == f"contestant role artifact was OOM-killed and {why}"


@pytest.mark.parametrize("own_oom, job_oom", [(None, None), (False, True), (True, False)])
def test_the_runners_own_kill_of_a_contestant_role_past_its_limit_is_the_contestants(own_oom, job_oom):
    # The memory watch measured the role's own footprint past the role's own limit.
    fault = classify_role(RoleOutcome("artifact", "contestant", 137, "memory_limit", own_oom=own_oom,
                                      job_oom=job_oom))
    assert (fault.fault_class, fault.retryable) == ("contestant", False)


# The build sandbox: any end but the planned stop is a fault, and the contestant's only when
# attribution is certain.

GIB = 2**30
RESERVE = 4 * GIB


def _sandbox(ending="exited", *, code=137, oom_kills=1, own_oom=True, job_oom=False, **split) -> RoleOutcome:
    """The sandbox's end: by default a kernel OOM at its own limit 2 s before it ended, with a
    split sampled 5 s before the OOM that has the harness at 1 GiB of its 4 GiB reserve."""
    fields = {"reserve_bytes": RESERVE, "ended_at": 102.0, "oom_at": 100.0, "oom_by": "kernel", "sample_at": 95.0,
              "total_bytes": 16 * GIB, "tmpfs_bytes": GIB, "harness_bytes": GIB, "contestant_bytes": 14 * GIB}
    return RoleOutcome("sandbox", "contestant", code, ending, oom_kills=oom_kills, own_oom=own_oom, job_oom=job_oom,
                       split=SplitEvidence(**{**fields, **split}))


NO_OOM = {"oom_at": None, "oom_by": None, "sample_at": None, "total_bytes": None, "tmpfs_bytes": None,
          "harness_bytes": None, "contestant_bytes": None}


def test_a_kernel_oom_at_the_sandboxs_own_limit_with_the_harness_in_its_reserve_is_the_contestants():
    fault = classify_role(_sandbox())
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert (fault.evidence["harness_bytes"], fault.evidence["contestant_bytes"], fault.evidence["oom_at"],
            fault.evidence["oom_by"], fault.evidence["own_oom"]) == (str(GIB), str(14 * GIB), "100.0", "kernel", "True")
    assert classify_build_end(_sandbox()) == fault


def test_the_runners_own_memory_kill_is_certain_whatever_the_kernels_counts():
    outcome = replace(_sandbox("memory_limit", code=-9, oom_kills=0, own_oom=None, job_oom=None, oom_by="runner"),
                      detail="footprint over")
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.evidence["detail"]) == ("contestant", "footprint over")


def test_the_boundaries_are_inclusive():
    # Exactly the reserve, a sample exactly 30 s before the OOM, an end exactly 30 s after it.
    fault = classify_role(_sandbox(sample_at=70.0, harness_bytes=RESERVE, ended_at=130.0))
    assert fault.fault_class == "contestant"


@pytest.mark.parametrize("outcome, why", [
    (_sandbox(code=1, oom_kills=0, own_oom=False, **NO_OOM), "no OOM recorded"),
    (_sandbox(own_oom=False), "its own memory limit was not reached"),
    (_sandbox(own_oom=None), "unknown whether its own memory limit"),
    (_sandbox(job_oom=True), "the job's own memory limit was reached"),
    (_sandbox(job_oom=None), "unknown whether the job's own memory limit"),
    (_sandbox(ended_at=130.5), "before it ended"),
    (_sandbox(sample_at=None, total_bytes=None, tmpfs_bytes=None, harness_bytes=None, contestant_bytes=None),
     "no memory split"),
    (_sandbox(sample_at=69.5), "older than it"),
    (_sandbox(harness_bytes=RESERVE + 1), "byte reserve"),
], ids=["no-oom", "own-limit-not-reached", "own-limit-unknown", "job-limit-reached", "job-limit-unknown",
        "oom-long-before-the-end", "no-sample", "stale-sample", "harness-over-reserve"])
def test_each_doubt_makes_the_early_end_infra(outcome, why):
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason.startswith("build sandbox sandbox ended early: ") and why in fault.reason


@pytest.mark.parametrize("ending", ["grant_mismatch", "launch_failed", "stopped", "scheduler_ended",
                                    "runtime_failed", "time_limit", "log_lost"])
def test_endings_the_harness_or_the_scheduler_caused_are_infra_whatever_the_memory(ending):
    fault = classify_role(_sandbox(ending))
    assert (fault.fault_class, fault.retryable) == ("infra", True)


@pytest.mark.parametrize("code", [0, 1, 137])
def test_a_sandbox_that_ended_by_itself_without_an_oom_is_infra(code):
    fault = classify_role(_sandbox(code=code, oom_kills=0, own_oom=False, **NO_OOM))
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    # The same end of a contestant role without a split is a result, no fault.
    assert classify_role(RoleOutcome("artifact", "contestant", code, "exited")) is None


def test_the_planned_stop_is_no_fault_whatever_the_sandbox_survived_before_it():
    outcome = _sandbox("planned_stop", code=0)
    assert classify_role(outcome) is None
    # Without the split the same stop after an OOM at its own limit is the contestant's fault.
    assert classify_role(replace(outcome, split=None)).fault_class == "contestant"


def test_writing_past_its_disk_cap_is_the_contestants():
    fault = classify_role(_sandbox("disk_limit", code=-9, oom_kills=0, own_oom=False, **NO_OOM))
    assert (fault.fault_class, fault.retryable) == ("contestant", False) and "disk cap" in fault.reason


def test_reaching_its_steps_pids_limit_is_the_contestants_as_for_any_contestant_role():
    pool = PoolSample(at=90.0, uid_tasks=5_000, step_tasks=4_096, ceiling=60_000)
    outcome = replace(_sandbox("contestant_pids", code=1, oom_kills=0, own_oom=False, **NO_OOM),
                      detail="its step reached its pids limit 3 times", pool=pool)
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert fault.reason == ("build sandbox sandbox reached its step's process limit: its step reached its pids "
                            "limit 3 times")
    # The pool is context in the evidence, as for any contestant role, next to the split.
    assert (fault.evidence["walk_step_tasks"], fault.evidence["reserve_bytes"]) == ("4096", str(RESERVE))


def test_the_build_rule_reads_only_an_outcome_with_a_split():
    with pytest.raises(ValueError, match="no memory split"):
        classify_build_end(RoleOutcome("sandbox", "contestant", 137, "exited"))


@pytest.mark.parametrize("detail", ["5368709120 bytes in its writable directories over 4 GiB",
                                    "a directory more than 256 levels deep in its writable directories, deeper than "
                                    "the disk walk measures"], ids=["bytes", "depth"])
def test_the_sandboxs_disk_limit_reason_names_how_the_cap_was_passed(detail):
    outcome = replace(_sandbox("disk_limit", code=-9, oom_kills=0, own_oom=False, **NO_OOM), detail=detail)
    fault = classify_role(outcome)
    assert fault.reason == f"build sandbox sandbox wrote past its disk cap: {detail}"
    assert fault.evidence["detail"] == detail


def _pool_at_end(others: int | None) -> PoolAtEnd:
    return PoolAtEnd(at=103.0, others_tasks=others, rule=CeilingRule(ceiling=60_000, accounting="uid", parent_limit=None),
                     node_tasks=3_000, pid_max=4_194_304)


def test_a_sandbox_that_ends_while_the_uids_other_tasks_fill_its_pool_was_starved():
    """Counted when it was reported, its step gone, the uid's other tasks alone fill 95% of the
    sandbox's process ceiling: its end may be theirs, and is infra, retried, saying so."""
    outcome = replace(_sandbox("exited", code=70, oom_kills=0, own_oom=False, **NO_OOM),
                      pool_at_end=_pool_at_end(57_000))
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert "starved" in fault.reason and fault.evidence["others_tasks"] == "57000"
    assert fault.evidence["reserve_bytes"] == str(RESERVE)
    # Below the mark the end is judged as before: with no OOM, infra all the same.
    calm = classify_role(replace(outcome, pool_at_end=_pool_at_end(56_999)))
    assert calm.fault_class == "infra" and "starved" not in calm.reason


def test_a_sandbox_that_ends_while_the_uids_tasks_cannot_be_counted_is_infra_saying_so():
    """Uncounted once its step was gone, the uid's tasks may have filled the sandbox's pool: its
    end is infra, retried, naming the failed count rather than an end with no OOM."""
    outcome = replace(_sandbox("exited", code=70, oom_kills=0, own_oom=False, **NO_OOM),
                      pool_at_end=_pool_at_end(None))
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("infra", True)
    assert fault.reason == ("build sandbox sandbox ended early: contestant role sandbox ended, and the uid's tasks "
                            "could not be counted once its step was gone, so starvation cannot be ruled out")
    assert fault.evidence["others_tasks"] == "uncounted" and fault.evidence["reserve_bytes"] == str(RESERVE)


@pytest.mark.parametrize("others", [60_000, None], ids=["starved", "uncounted"])
def test_a_starved_pool_does_not_change_what_the_sandboxs_own_limits_decide(others):
    """Its memory, its disk and its step's pids limit are its own whatever the uid's pool holds,
    or when it could not be counted."""
    full = _pool_at_end(others)
    assert classify_role(replace(_sandbox(), pool_at_end=full)).fault_class == "contestant"
    disk = replace(_sandbox("disk_limit", code=-9, oom_kills=0, own_oom=False, **NO_OOM), pool_at_end=full)
    assert classify_role(disk).fault_class == "contestant"
    pids = replace(_sandbox("contestant_pids", code=1, oom_kills=0, own_oom=False, **NO_OOM), pool_at_end=full)
    assert classify_role(pids).fault_class == "contestant"
    assert classify_role(replace(_sandbox("planned_stop", code=0), pool_at_end=full)) is None
