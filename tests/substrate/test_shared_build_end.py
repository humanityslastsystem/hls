"""The build rule beside the in-build reference service (``classify_shared_build_end``): only a
kernel kill at the job's own limit is reopened, and it is the contestant's only when the service
was within its own limit and the runner within its reserve, read, never assumed."""

from __future__ import annotations

import pytest

from hls.substrate.faults import SharedMemoryEvidence, StepMemory, classify_build_end, classify_shared_build_end
from tests.substrate.test_faults import GIB, _sandbox

#: Within its own limit, the runner within its reserve: the session's excess.
WITHIN = SharedMemoryEvidence(
    service_running=True, service_limit_bytes=400 * GIB, service_ooms=0, service_breach=False,
    service=StepMemory(last_bytes=330 * GIB, highest_bytes=340 * GIB, peak_bytes=350 * GIB),
    runner_reserve_bytes=4 * GIB, runner=StepMemory(last_bytes=GIB, highest_bytes=2 * GIB, peak_bytes=3 * GIB),
    sandbox=StepMemory(last_bytes=1500 * GIB, highest_bytes=1560 * GIB, peak_bytes=1570 * GIB),
    job_ooms_at_start=0, job_ooms_at_end=1)
#: The sandbox killed by the kernel at the job's limit, its own not reached.
AT_JOB = {"own_oom": False, "job_oom": True}


def _with(**changes) -> SharedMemoryEvidence:
    return SharedMemoryEvidence(**{**WITHIN.__dict__, **changes})


def test_a_kill_at_the_jobs_limit_with_the_service_and_runner_within_theirs_is_the_contestants():
    assert classify_build_end(_sandbox(**AT_JOB)).fault_class == "infra"
    fault = classify_shared_build_end(_sandbox(**AT_JOB), WITHIN)
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert fault.evidence["shared.service.highest_bytes"] == str(340 * GIB)
    assert fault.evidence["shared.runner.peak_bytes"] == str(3 * GIB)
    assert fault.evidence["shared.job_ooms_at_end"] == "1"


@pytest.mark.parametrize("changes, needle", [
    ({"service_ooms": 1}, "reached its own limit"),
    ({"service_breach": True}, "reached its own limit"),
    ({"service_ooms": None}, "could not be read"),
    ({"runner": None}, "the runner's memory could not be read"),
    ({"runner": StepMemory(last_bytes=GIB, highest_bytes=5 * GIB, peak_bytes=None)}, "against its"),
])
def test_anything_the_rule_cannot_rule_out_is_the_harnesss(changes, needle):
    fault = classify_shared_build_end(_sandbox(**AT_JOB), _with(**changes))
    assert (fault.fault_class, fault.retryable) == ("infra", True) and needle in fault.reason


def test_with_no_service_running_the_build_rules_verdict_stands():
    decided = classify_build_end(_sandbox(**AT_JOB))
    fault = classify_shared_build_end(_sandbox(**AT_JOB), _with(service_running=False))
    assert (fault.fault_class, fault.reason) == ("infra", decided.reason)
    assert fault.evidence["shared.service_running"] == "False"


@pytest.mark.parametrize("outcome", [
    _sandbox("runtime_failed", **AT_JOB),
    _sandbox(own_oom=False, job_oom=None),
    _sandbox(own_oom=False, job_oom=False),
    _sandbox(**AT_JOB, oom_by="runner"),
    _sandbox(**AT_JOB, harness_bytes=5 * GIB),
    _sandbox(**AT_JOB, oom_at=60.0),
], ids=["runtime-failed", "job-unread", "job-not-reached", "runner-kill", "harness-over", "oom-stale"])
def test_every_other_end_is_judged_as_the_build_rule_judges_it_or_stays_infra(outcome):
    """An end the build rule does not leave to the job's limit keeps its verdict (a runner's kill
    is at the sandbox's own limit); one it does, whose own conditions fail, stays infra."""
    fault = classify_shared_build_end(outcome, WITHIN)
    decided = classify_build_end(outcome)
    if outcome.ending == "runtime_failed" or outcome.job_oom is not True or outcome.split.oom_by == "runner":
        assert fault == decided
    else:
        assert fault.fault_class == "infra"


def test_an_own_limit_kill_stays_the_contestants_as_the_build_rule_has_it():
    assert classify_shared_build_end(_sandbox(), WITHIN) == classify_build_end(_sandbox())
    assert classify_shared_build_end(_sandbox(), WITHIN).fault_class == "contestant"
