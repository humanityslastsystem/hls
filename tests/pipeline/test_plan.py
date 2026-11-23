from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hls.pipeline.checkout import CodeState
from hls.pipeline.plan import (Allocation, InstanceExists, Outcome, Plan, PlanConflict, PlanMismatch, PlanStore,
                               PlanUnreadable, StagePlan, Submission, Toolchain, revision_numbers, submission_tag)
from hls.settings import RetryPolicy, Settings, SiteSettings
from hls.spec.resources import ResourceShape
from hls.store.base import NotFound
from hls.store.layout import InstanceLayout
from hls.store.lock import Holder, LockLost, instance_lease
from hls.store.posix import PosixStore
from typer.testing import CliRunner
from hls.substrate.base import FaultRecord
from hls.substrate.slurm.resources import SlurmPlacement

LAY = InstanceLayout.rollout("noop", "20261002T000000Z-abc123")
SHAPE = ResourceShape(gpus=1, gpu_kind="H200", cpus=2, memory_gib=4, time_sec=600)
#: Where each allocation of these plans goes: the site's GPU partition, under its QOS.
PLACED = SlurmPlacement(partition=None, qos=None, socket_tasks=None, exclude=[])


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _plan(tmp_path: Path) -> Plan:
    site = SiteSettings(partition="gpu", store_root=tmp_path, node_scratch=Path("/tmp"),
                        secrets_path=Path("/x"))
    stage = StagePlan(name="noop", kind="noop", shape=SHAPE, shape_overridden=False, params={}, after=[], inputs={},
                      retry=RetryPolicy(max=2))
    return Plan(
        instance=LAY.root, store_root=tmp_path, task="noop", pipeline="noop", job_name="hls-noop", created_at=1.0,
        revision=1, lease_generation=1,
        code=CodeState(checkout=Path("/c"), hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None),
        toolchain=Toolchain(python=Path("/c/.venv/bin/python"), path="/usr/bin:/bin", home=Path("/h")),
        settings=Settings(site=site), stages=[stage],
        allocations=[Allocation(index=0, stages=["noop"], request=SHAPE, placement=PLACED, depends_on=None)],
        warnings=[])


def _holder(name: str) -> Holder:
    return Holder(host="h", pid=1, job_id=None, purpose=name)


def test_revisions_are_whole_plans_with_the_writers_generation(tmp_path):
    store = PosixStore(tmp_path)
    lease = instance_lease(store, LAY, ttl_sec=120, clock=Clock())
    handle = lease.try_acquire(_holder("orchestrator"))
    plans = PlanStore(store, LAY)
    first = plans.create(_plan(tmp_path), handle)
    assert (first.revision, first.lease_generation) == (1, 1)
    second = plans.write(first.model_copy(update={"warnings": ["w"]}), handle)
    assert second.revision == 2 and plans.latest() == second
    assert plans.revisions() == [1, 2]
    assert store.list_children(LAY.plan) == ["rev-00000001.json", "rev-00000002.json"]
    with pytest.raises(InstanceExists):
        plans.create(_plan(tmp_path), handle)


def test_a_stale_orchestrator_cannot_write_after_a_takeover(tmp_path):
    store = PosixStore(tmp_path)
    clock = Clock()
    lease = instance_lease(store, LAY, ttl_sec=120, clock=clock)
    old = lease.try_acquire(_holder("old"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), old)
    clock.t += 121
    new = lease.try_acquire(_holder("new"))
    takeover = plans.write(plans.latest(), new)
    assert takeover.lease_generation == 2
    # The displaced orchestrator wakes up and records an intent from what it read before.
    with pytest.raises(LockLost):
        plans.write(base.model_copy(update={"warnings": ["stale"]}), old)
    assert plans.latest() == takeover


def test_a_stale_writer_that_passes_its_lease_check_still_cannot_overwrite(tmp_path, monkeypatch):
    store = PosixStore(tmp_path)
    clock = Clock()
    lease = instance_lease(store, LAY, ttl_sec=120, clock=clock)
    old = lease.try_acquire(_holder("old"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), old)
    clock.t += 121
    new = lease.try_acquire(_holder("new"))
    # The old holder's lease check races the takeover and passes; the generation in the
    # newest revision still stops it.
    monkeypatch.setattr(old, "confirm", lambda: None)
    plans.write(base, new)
    with pytest.raises(LockLost, match="generation 2 wrote revision 2"):
        plans.write(base, old)
    assert plans.latest().lease_generation == 2


def test_losing_the_create_race_is_a_conflict(tmp_path, monkeypatch):
    store = PosixStore(tmp_path)
    lease = instance_lease(store, LAY, ttl_sec=120, clock=Clock())
    handle = lease.try_acquire(_holder("o"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), handle)
    store.create(LAY.plan_revision(2), b"{}")
    monkeypatch.setattr(plans, "latest", lambda: base)
    with pytest.raises(PlanConflict, match="revision 2 was written first"):
        plans.write(base, handle)


def test_a_plan_derived_from_an_old_revision_is_a_conflict(tmp_path):
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), handle)
    plans.write(base, handle)
    with pytest.raises(PlanConflict, match="derived from revision 1, latest is 2"):
        plans.write(base, handle)


def test_current_submission_and_allocation_replacement(tmp_path):
    plan = _plan(tmp_path)
    subs = [Submission(n=1, tag="t1", intended_at=1.0, after_job=None, status="absent"),
            Submission(n=2, tag="t2", intended_at=2.0, after_job=None, status="submitted", job_id="7")]
    alloc = plan.allocation(0).model_copy(update={"submissions": subs})
    plan = plan.with_allocation(alloc)
    assert plan.allocation(0).current().n == 2
    retry = Allocation(index=1, stages=["noop"], request=SHAPE, placement=PLACED, depends_on=None, retry_of=0)
    assert [a.index for a in plan.with_allocation(retry).allocations] == [0, 1]
    with pytest.raises(ValueError):
        plan.with_allocation(retry.model_copy(update={"index": 5}))
    assert plan.plan_dir == tmp_path / LAY.plan
    assert submission_tag(LAY.root, 1, 2) == f"hls:{LAY.root}/1/2"


def test_a_takeover_that_has_not_written_yet_still_stops_the_old_holder(tmp_path):
    store = PosixStore(tmp_path)
    clock = Clock()
    lease = instance_lease(store, LAY, ttl_sec=120, clock=clock)
    old = lease.try_acquire(_holder("old"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), old)
    clock.t += 121
    lease.try_acquire(_holder("new"))
    # The new holder has taken the lease and written nothing; only the lease check stops the old one.
    with pytest.raises(LockLost):
        plans.write(base, old)
    assert plans.revisions() == [1]


def _confirm_that_fails_the_second_time(handle, monkeypatch):
    """Make *handle*'s lease check pass once and then report a takeover, as a lease can be
    taken over between the check and the create."""
    calls = []

    def confirm() -> None:
        calls.append(1)
        if len(calls) > 1:
            raise LockLost("taken over")

    monkeypatch.setattr(handle, "confirm", confirm)
    return calls


def test_losing_the_create_race_to_a_new_holder_is_a_lock_loss(tmp_path, monkeypatch):
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), handle)
    store.create(LAY.plan_revision(2), b"{}")
    monkeypatch.setattr(plans, "latest", lambda: base)
    calls = _confirm_that_fails_the_second_time(handle, monkeypatch)
    with pytest.raises(LockLost, match="taken over"):
        plans.write(base, handle)
    assert len(calls) == 2


def test_a_newer_generation_is_reported_after_a_second_lease_check(tmp_path, monkeypatch):
    store = PosixStore(tmp_path)
    clock = Clock()
    lease = instance_lease(store, LAY, ttl_sec=120, clock=clock)
    old = lease.try_acquire(_holder("old"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), old)
    clock.t += 121
    plans.write(base, lease.try_acquire(_holder("new")))
    calls = _confirm_that_fails_the_second_time(old, monkeypatch)
    with pytest.raises(LockLost, match="taken over"):
        plans.write(base, old)
    assert len(calls) == 2


def test_writing_when_there_is_no_plan_is_not_found(tmp_path):
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    with pytest.raises(NotFound):
        PlanStore(store, LAY).write(_plan(tmp_path), handle)


def test_a_revision_file_must_hold_its_own_revision_and_instance(tmp_path):
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    plans = PlanStore(store, LAY)
    plans.create(_plan(tmp_path), handle)
    store.create(LAY.plan_revision(2), store.read_bytes(LAY.plan_revision(1)))
    with pytest.raises(PlanMismatch, match="holds revision 1"):
        plans.read(2)
    other = InstanceLayout.rollout("noop", "other-id")
    store.create(other.plan_revision(1), store.read_bytes(LAY.plan_revision(1)))
    with pytest.raises(PlanMismatch, match=f"belongs to {LAY.root}"):
        PlanStore(store, other).read(1)


def _written_by_another_version(tmp_path: Path) -> PosixStore:
    """A store holding the fixture plan as a version of hls with a setting this one lacks wrote
    it: its site table names the site."""
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    PlanStore(store, LAY).create(_plan(tmp_path), handle)
    path = store.path(LAY.plan_revision(1))
    data = json.loads(path.read_bytes())
    data["settings"]["site"]["name"] = "alpha"
    path.write_text(json.dumps(data))
    return store


#: What reading the plan ``_written_by_another_version`` wrote says.
UNREADABLE = (f"{LAY.plan_revision(1)} was written by another version of hls (hls commit {'a' * 40}) and does not "
              "load under this one: settings.site.name: Extra inputs are not permitted. Resume, follow or inspect "
              "it with a checkout of that version")


def test_a_plan_another_version_of_hls_wrote_is_refused_naming_the_plan_and_its_version(tmp_path):
    """No compatibility: a stored plan whose settings this version does not have fails to load, and
    says which plan, that another version wrote it and which, and the field it refuses."""
    plans = PlanStore(_written_by_another_version(tmp_path), LAY)
    with pytest.raises(PlanUnreadable) as refused:
        plans.latest()
    assert str(refused.value) == UNREADABLE
    store = PosixStore(tmp_path / "other")
    store.write_bytes(LAY.plan_revision(1), b"{not json")
    with pytest.raises(PlanUnreadable, match=rf"^{LAY.plan_revision(1)} is not a plan: it is not a JSON object"):
        PlanStore(store, LAY).latest()


@pytest.mark.parametrize("command", ["status", "follow", "resume", "cancel"])
def test_the_commands_that_read_a_stored_plan_refuse_another_versions_in_one_line(tmp_path, command, monkeypatch):
    from hls import cli

    monkeypatch.setattr(cli, "configure_logging", lambda **kw: None)
    _written_by_another_version(tmp_path)
    result = CliRunner().invoke(cli.app, [command, LAY.root, "--store", str(tmp_path)])
    assert (result.exit_code, result.stderr) == (1, f"hls {command}: {UNREADABLE}\n")


def test_a_plan_of_another_instance_is_refused(tmp_path):
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    other = InstanceLayout.rollout("noop", "other-id")
    with pytest.raises(PlanMismatch):
        PlanStore(store, other).create(_plan(tmp_path), handle)
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), handle)
    with pytest.raises(PlanMismatch):
        plans.write(base.model_copy(update={"instance": other.root}), handle)
    assert plans.revisions() == [1]


def test_create_stamps_the_generation_of_its_handle(tmp_path):
    store = PosixStore(tmp_path)
    clock = Clock()
    lease = instance_lease(store, LAY, ttl_sec=120, clock=clock)
    lease.try_acquire(_holder("first"))
    clock.t += 121
    second = lease.try_acquire(_holder("second"))
    assert second.generation == 2
    assert PlanStore(store, LAY).create(_plan(tmp_path), second).lease_generation == 2


def test_a_holder_that_lost_the_lease_cannot_create_the_plan(tmp_path):
    store = PosixStore(tmp_path)
    clock = Clock()
    lease = instance_lease(store, LAY, ttl_sec=120, clock=clock)
    old = lease.try_acquire(_holder("old"))
    clock.t += 121
    lease.try_acquire(_holder("new"))
    plans = PlanStore(store, LAY)
    with pytest.raises(LockLost):
        plans.create(_plan(tmp_path), old)
    assert plans.revisions() == []


def test_files_that_are_not_revisions_are_ignored(tmp_path):
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), handle)
    store.create(f"{LAY.plan}/rev-00000001.json.bak", b"junk")
    store.create(f"{LAY.plan}/rev-1.json", b"junk")
    assert plans.revisions() == [1]
    assert plans.write(base, handle).revision == 2
    assert revision_numbers(["rev-00000003.json", "notes", "rev-00000001.json", "rev-3.json"]) == [1, 3]


def test_writing_an_invalid_plan_raises(tmp_path):
    store = PosixStore(tmp_path)
    handle = instance_lease(store, LAY, ttl_sec=120, clock=Clock()).try_acquire(_holder("o"))
    plans = PlanStore(store, LAY)
    base = plans.create(_plan(tmp_path), handle)
    with pytest.raises(ValidationError):
        plans.write(base.model_copy(update={"task": None}), handle)
    assert plans.revisions() == [1]


def _two_stage_plan(tmp_path: Path) -> Plan:
    plan = _plan(tmp_path)
    second = plan.stages[0].model_copy(update={"name": "other", "after": ["noop"]})
    return plan.model_copy(update={
        "stages": [*plan.stages, second],
        "allocations": [plan.allocations[0],
                        Allocation(index=1, stages=["other"], request=SHAPE, placement=PLACED, depends_on=0)]})


def _sub(n: int, status: str = "absent") -> Submission:
    return Submission(n=n, tag=f"t{n}", intended_at=float(n), after_job=None, status=status)


def _revalidated(plan: Plan, **update) -> Plan:
    return Plan.model_validate(plan.model_copy(update=update).model_dump())


@pytest.mark.parametrize("make,message", [
    (lambda p: {"allocations": [p.allocations[0], p.allocations[1].model_copy(update={"index": 2})]},
     "allocation 1 has index 2"),
    (lambda p: {"stages": [p.stages[0], p.stages[0]]}, "stage names are not unique"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(update={"stages": ["ghost"]}), p.allocations[1]]},
     "allocation 0 runs ghost"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(update={"depends_on": 0}), p.allocations[1]]},
     "allocation 0 depends_on 0, which is not another allocation"),
    (lambda p: {"allocations": [p.allocations[0], p.allocations[1].model_copy(update={"depends_on": 5})]},
     "allocation 1 depends_on 5, which is not another allocation"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(update={"depends_on": 1}), p.allocations[1]]},
     "allocation 0 waits on itself through allocation 1"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(update={"waits_for": [1]}), p.allocations[1]]},
     "allocation 0 waits on itself through allocation 1"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(update={"waits_for": [0]}), p.allocations[1]]},
     "allocation 0 waits_for 0, which is not another allocation"),
    (lambda p: {"allocations": [p.allocations[0], p.allocations[1].model_copy(update={"waits_for": [5]})]},
     "allocation 1 waits_for 5, which is not another allocation"),
    (lambda p: {"allocations": [p.allocations[0], p.allocations[1].model_copy(
        update={"depends_on": None, "waits_for": [0, 0]})]},
     r"allocation 1 waits_for \[0, 0\], which names an allocation twice"),
    (lambda p: {"allocations": [p.allocations[0], p.allocations[1].model_copy(update={"singleton": True})]},
     "allocation 1 runs one at a time with the site's other warmers .singleton. and depends_on 0"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(update={"placement": p.allocations[0].placement.model_copy(
        update={"exclude": ["g1,g2"]})}), p.allocations[1]]},
     "'g1,g2' is not a node name"),
    (lambda p: {"allocations": [p.allocations[0], p.allocations[1].model_copy(update={"retry_of": 1})]},
     "allocation 1 retry_of 1"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(update={"submissions": [_sub(2)]}), p.allocations[1]]},
     "allocation 0 submission 1 is numbered 2"),
    (lambda p: {"allocations": [p.allocations[0].model_copy(
        update={"submissions": [_sub(1, "intended"), _sub(2, "submitted")]}), p.allocations[1]]},
     "allocation 0 has 2 live submissions"),
], ids=["index", "duplicate_stage", "unknown_stage", "depends_on_self", "depends_on_unknown", "depends_on_cycle",
        "waits_for_cycle", "waits_for_self", "waits_for_unknown", "waits_for_twice", "singleton_depends",
        "exclude_name", "retry_of_not_below", "submission_number", "two_live"])
def test_a_plan_must_be_consistent(tmp_path, make, message):
    plan = _two_stage_plan(tmp_path)
    assert _revalidated(plan) == plan
    with pytest.raises(ValidationError, match=message):
        _revalidated(plan, **make(plan))


def test_an_allocation_may_wait_for_two_that_share_a_predecessor(tmp_path):
    plan = _two_stage_plan(tmp_path)
    third = plan.stages[0].model_copy(update={"name": "third", "after": ["noop", "other"]})
    diamond = Allocation(index=2, stages=["third"], request=SHAPE, placement=PLACED, depends_on=None,
                         waits_for=[0, 1])
    grown = _revalidated(plan, stages=[*plan.stages, third], allocations=[*plan.allocations, diamond])
    assert grown.allocation(2).waits_for == [0, 1] and grown.allocation(1).depends_on == 0
    assert _revalidated(plan, grower="calibration").grower == "calibration"


def test_a_consistent_plan_with_retries_and_cancelled_submissions_is_valid(tmp_path):
    plan = _two_stage_plan(tmp_path)
    first = plan.allocations[0].model_copy(update={"submissions": [_sub(1, "cancelled"), _sub(2, "submitted")]})
    retry = Allocation(index=2, stages=["noop"], request=SHAPE, placement=PLACED, depends_on=None, retry_of=0)
    # The allocation that waited for the failed one is rewired to the retry added after it.
    rewired = plan.allocations[1].model_copy(update={"depends_on": 2})
    assert _revalidated(plan, allocations=[first, rewired, retry]).allocation(2).retry_of == 0


def test_allocation_lookup_does_not_wrap_or_run_past_the_end(tmp_path):
    plan = _plan(tmp_path)
    assert plan.allocation(0).index == 0
    for k in (-1, 1):
        with pytest.raises(IndexError):
            plan.allocation(k)


_CONTESTANT = FaultRecord(fault_class="contestant", reason="contestant role server ran out of memory", retryable=False)
_INFRA = FaultRecord(fault_class="infra", reason="job 100: the node failed", retryable=True)


def test_a_failed_outcome_carries_the_contestant_fault_and_only_a_void_one_an_infra_fault():
    assert Outcome(status="failed", reason="x", at=1.0, fault=_CONTESTANT).fault == _CONTESTANT
    assert Outcome(status="void", reason="x", at=1.0, fault=_INFRA).fault == _INFRA
    assert Outcome(status="void", reason="x", at=1.0).fault is None
    for status, fault, message in [("failed", None, "a failed instance carries the contestant fault"),
                                   ("failed", _INFRA, "a failed instance carries the contestant fault"),
                                   ("void", _CONTESTANT, "a contestant fault ends the instance failed"),
                                   ("succeeded", _INFRA, "only a failed or void instance carries a fault"),
                                   ("cancelled", _INFRA, "only a failed or void instance carries a fault")]:
        with pytest.raises(ValidationError, match=message):
            Outcome(status=status, reason="x", at=1.0, fault=fault)


def test_only_a_failed_outcome_leaves_records():
    assert Outcome(status="failed", reason="x", at=1.0, fault=_CONTESTANT, records_left=True).records_left
    for status, fault in [("succeeded", None), ("void", None), ("void", _INFRA), ("cancelled", None)]:
        with pytest.raises(ValidationError, match="only a failed instance has records to leave"):
            Outcome(status=status, reason="x", at=1.0, fault=fault, records_left=True)
