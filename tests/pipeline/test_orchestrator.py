from __future__ import annotations

import io
import logging
import time

import pytest

import hls.pipeline.orchestrator as orchestrator_module
from hls.obs.logsetup import OWNED_LOGGERS, configure_logging
from hls.pipeline.events import read_events
from hls.pipeline.orchestrator import WITHHELD, Orchestrator, login_memory, request_cancel
from hls.pipeline.plan import (AllocationStatus, AttemptStatus, InstanceExists, Plan, PlanConflict, PlanStore,
                               RecordsRefused)
from hls.pipeline.stage import StageFault
from hls.pipeline.reconcile import ABSENT_AFTER_SEC, Rewire, Rewired, Submit
from hls.spec.resources import GrantedResources
from hls.store.lock import Holder, LockHeld, LockLost, instance_lease
from hls.store.manifest import Manifest, Provenance, Resources, Times, publish_manifest
from hls.store.posix import PosixStore
from hls.substrate.base import (DependencyRefused, FaultRecord, JobInfo, JobState, Refused,
                                SchedulerUnavailable, SubmitUnknown)
from hls.substrate.secrets import SecretsError, load_secrets
from hls.substrate.slurm.scheduler import submit_argv
from tests.pipeline.fake_scheduler import FakeJob, FakeScheduler
from tests.pipeline.plans import noop_plan

HOLDER = Holder(host="login", pid=1, job_id=None, purpose="orchestrator")


@pytest.fixture
def console():
    """The CLI's console handler on the owned loggers, writing to a buffer; the loggers as they
    were afterwards, so later tests' log capture still works."""
    root = logging.getLogger()
    saved = {name: (list(logging.getLogger(name).handlers), logging.getLogger(name).propagate,
                    logging.getLogger(name).level) for name in OWNED_LOGGERS}
    level = root.level
    stream = io.StringIO()
    configure_logging(stream=stream)
    yield stream
    for name, (handlers, propagate, was) in saved.items():
        logger = logging.getLogger(name)
        logger.handlers[:] = handlers
        logger.propagate = propagate
        logger.setLevel(was)
    root.setLevel(level)


class Clock:
    def __init__(self) -> None:
        self.t = 10_000.0

    def __call__(self) -> float:
        return self.t


def _publish(plan: Plan, stage: str) -> None:
    store = PosixStore(plan.store_root)
    s = plan.stage(stage)
    granted = GrantedResources(gpus=2, gpu_kind="H200", cpus=4, memory_gib=8.0, disk_free_gib=1.0)
    publish_manifest(store, plan.layout, Manifest(
        instance=plan.instance, stage=stage, attempt=1, outputs={}, inputs={},
        provenance=Provenance(hls_commit="a" * 40, harbor_commit="b" * 40, dirty_diff_sha256=None, image_digests={},
                              config_sha256="c" * 64),
        resources=Resources(declared=s.shape, granted=granted),
        times=Times(submitted_at=None, allocation_started_at=None, started_at=1.0, ended_at=2.0)))


def _heartbeat(plan: Plan, k: int, n: int, at: float, job_id: str = "100", fault: FaultRecord | None = None,
               filed_as: tuple[int, int] | None = None) -> None:
    """Write the status of submission *n* of allocation *k*, in the place of another submission when
    *filed_as* names it."""
    store = PosixStore(plan.store_root)
    status = AllocationStatus(instance=plan.instance, allocation=k, submission=n, job_id=job_id, node="g",
                              state="running", started_at=at, heartbeat_at=at, end_time=None, stage=None, attempt=None,
                              grant=None, tool_versions=None, swap_max_bytes=None, memory_enforced_by=None,
                              fault=fault, ended_at=None)
    store.write_bytes(plan.layout.allocation_status(*(filed_as or (k, n))), status.model_dump_json().encode())


def _advance(orch: Orchestrator, clock: Clock, sec: float) -> None:
    """Move the clock, renewing the lease on the way as its keep-alive thread would."""
    while sec > 0:
        clock.t += min(sec, 5.0)
        sec -= 5.0
        orch.handle.renew()


def _create(tmp_path, sched: FakeScheduler, clock: Clock, **run) -> Orchestrator:
    return Orchestrator.create(noop_plan(tmp_path, **run), sched, holder=HOLDER, clock=clock, sleep=lambda s: None)


def _attach(orch: Orchestrator, sched: FakeScheduler, clock: Clock) -> Orchestrator:
    """A new orchestrator taking over after *orch* died."""
    return Orchestrator.attach(orch.store, orch.layout, sched, holder=HOLDER.model_copy(update={"pid": 2}),
                               alive=lambda h: False, clock=clock, sleep=lambda s: None)


def _secrets_file(tmp_path, text: str) -> str:
    path = tmp_path / "secrets.toml"
    path.write_text(text)
    path.chmod(0o600)
    return str(path)


def _lines(orch: Orchestrator) -> list[list[str]]:
    """The sbatch line each ``submitted`` event of *orch* records."""
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    return [e["argv"] for e in events if e["kind"] == "submitted"]


def test_the_recorded_sbatch_line_has_every_secret_value_scrubbed(tmp_path):
    """No secret reaches sbatch's arguments, which come from the plan the planner checked for
    every value. The recorded line is scrubbed all the same, here of a value the secrets file came
    to hold after the plan took it as its job name; sbatch itself is given the plan's line."""
    plan = noop_plan(tmp_path, site={"secrets_path": str(tmp_path / "secrets.toml")})
    _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')
    sched, clock = FakeScheduler(), Clock()
    orch = Orchestrator.create(plan, sched, holder=HOLDER, clock=clock, sleep=lambda s: None)
    plan, _, _ = orch.step()
    [line] = _lines(orch)
    assert "--job-name=<redacted:named>" in line and not any("hls-noop" in a for a in line)
    assert plan.allocation(0).current().argv == line
    assert "--job-name=hls-noop" in submit_argv(sched.jobs["100"].request)


def test_a_taking_over_orchestrator_records_its_sbatch_lines_scrubbed(tmp_path):
    plan = noop_plan(tmp_path, site={"secrets_path": str(tmp_path / "secrets.toml")})
    sched, clock = FakeScheduler(), Clock()
    first = Orchestrator.create(plan, sched, holder=HOLDER, clock=clock, sleep=lambda s: None)
    _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')
    first.close()
    orch = _attach(first, sched, clock)
    plan, _, _ = orch.step()
    [line] = _lines(orch)
    assert "--job-name=<redacted:named>" in line and plan.allocation(0).current().argv == line


class Answering(FakeScheduler):
    """A scheduler whose every submit raises *answer*."""

    def __init__(self, answer: Exception) -> None:
        super().__init__()
        self.answer = answer

    def submit(self, request):
        self.calls.append(("submit", request.tag, request.after))
        raise self.answer


@pytest.mark.parametrize("answer, kind", [
    (Refused("sbatch: error: Batch job submission failed: Invalid job name hls-noop"), "submit_refused"),
    (DependencyRefused("sbatch: error: Job dependency problem for hls-noop"), "submit_refused"),
    (SubmitUnknown("sbatch: error: Socket timed out submitting hls-noop"), "submit_unknown"),
], ids=["refused", "dependency", "lost"])
def test_an_unanswered_submission_records_its_line_and_its_error_scrubbed(tmp_path, answer, kind):
    """Whatever sbatch answers is recorded, in the event and in the plan's submission, with every
    value of the secrets file replaced; so is the void outcome a plain refusal ends the instance
    with."""
    plan = noop_plan(tmp_path, site={"secrets_path": str(tmp_path / "secrets.toml")})
    _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')
    orch = Orchestrator.create(plan, Answering(answer), holder=HOLDER, clock=Clock(), sleep=lambda s: None)
    plan, _, _ = orch.step()
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    [event] = [e for e in events if e["kind"] == kind]
    sub = plan.allocation(0).submissions[0]
    assert "--job-name=<redacted:named>" in event["argv"] and sub.argv == event["argv"]
    assert "<redacted:named>" in event["error"] and "<redacted:named>" in sub.error
    shown = [*event["argv"], event["error"], sub.error, *(plan.outcome.reason for _ in [0] if plan.outcome)]
    assert not any("hls-noop" in text for text in shown)
    if type(answer) is Refused:
        assert plan.outcome.status == "void" and "<redacted:named>" in plan.outcome.reason


def test_an_effect_that_breaks_and_a_refused_preflight_are_recorded_scrubbed(tmp_path):
    plan = noop_plan(tmp_path, site={"secrets_path": str(tmp_path / "secrets.toml")})
    _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')
    orch = Orchestrator.create(plan, Answering(RuntimeError("the adapter broke on hls-noop")), holder=HOLDER,
                               clock=Clock(), sleep=lambda s: None)
    plan, _, _ = orch.step()
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    [failed] = [e for e in events if e["kind"] == "effect_failed"]
    error = plan.allocation(0).submissions[0].error
    assert "<redacted:named>" in failed["error"] and "<redacted:named>" in error and "hls-noop" not in error
    refusing = FakeScheduler(test_only_refusal="sbatch: error: Invalid job name hls-noop")
    second = noop_plan(tmp_path, instance="r2", site={"secrets_path": str(tmp_path / "secrets.toml")})
    void = Orchestrator.create(second, refusing, holder=HOLDER, clock=Clock(), sleep=lambda s: None)
    outcome = void.plans.latest().outcome
    assert outcome.status == "void" and "<redacted:named>" in outcome.reason and "hls-noop" not in outcome.reason


def test_an_orchestrator_that_could_not_read_the_secrets_file_only_cancels(tmp_path):
    plan = noop_plan(tmp_path, site={"secrets_path": _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')})
    sched = FakeScheduler()
    first = Orchestrator.create(plan, sched, holder=HOLDER, clock=Clock(), sleep=lambda s: None)
    plan, _, _ = first.step()
    first.close()
    (tmp_path / "secrets.toml").chmod(0o644)
    orch = Orchestrator.attach(first.store, first.layout, sched, holder=HOLDER.model_copy(update={"pid": 2}),
                               alive=lambda h: False, clock=Clock(), sleep=lambda s: None, cancel_only=True)
    submits = len(sched.submits())
    with pytest.raises(RuntimeError, match="could not read the secrets file and only cancels"):
        orch.perform(orch.plans.latest(), Submit(0, 1))
    assert len(sched.submits()) == submits


def _failed_with_a_dependent(tmp_path, sched: FakeScheduler, clock: Clock, *,
                             cancelled: bool = True) -> Orchestrator:
    """An instance whose first allocation failed with a contestant fault, its dependent still
    queued, left by an orchestrator that died before it decided; *cancelled*, its secrets file
    then made unreadable, and its cancel asked for."""
    plan = noop_plan(tmp_path, gpu_policy="split",
                     site={"secrets_path": _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')})
    first = Orchestrator.create(plan, sched, holder=HOLDER, clock=clock, sleep=lambda s: None)
    first.step()
    first.step()
    assert set(sched.jobs) == {"100", "101"}
    contestant = FaultRecord(fault_class="contestant", reason="contestant role server ran out of memory",
                             retryable=False)
    _heartbeat(first.plans.latest(), 0, 1, clock.t, job_id="100", fault=contestant)
    sched.finish("100", JobState.FAILED)
    first.close()
    if cancelled:
        (tmp_path / "secrets.toml").chmod(0o644)
        request_cancel(first.store, first.layout, at=clock.t)
    return first


def _needs_secrets(store, ended):
    """A writer that, as the real one, cannot write without the secrets file."""
    load_secrets(ended.settings.site.secrets_path)
    return {"score": f"{ended.layout.root}/failed/score.json"}


def _kinds_of(orch: Orchestrator, prefix: str) -> list[str]:
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    return [e["kind"] for e in events if e["kind"].startswith(prefix)]


def _take_over(first: Orchestrator, sched: FakeScheduler, clock: Clock, pid: int, **kw) -> Orchestrator:
    return Orchestrator.attach(first.store, first.layout, sched, holder=HOLDER.model_copy(update={"pid": pid}),
                               alive=lambda h: False, clock=clock, sleep=lambda s: None, **kw)


def test_a_cancel_of_a_failed_instance_cancels_and_leaves_its_records_to_a_full_take_over(tmp_path):
    """The cancel, which cannot read the secrets file, ends the instance failed and cancels the
    dependent; the records are left, which the outcome says, and the next orchestrator that can
    read the file writes them in its pass and clears the mark."""
    sched, clock = FakeScheduler(), Clock()
    first = _failed_with_a_dependent(tmp_path, sched, clock)
    written: list[str] = []

    def records(store, ended):
        load_secrets(ended.settings.site.secrets_path)
        written.append(ended.outcome.status)
        return {"score": f"{ended.layout.root}/failed/score.json"}

    orch = _take_over(first, sched, clock, 2, on_failed=records, cancel_only=True)
    final = orch.run()
    assert (final.outcome.status, final.outcome.records_left) == ("failed", True)
    assert sched.jobs["101"].state is JobState.CANCELLED and written == []
    assert _kinds_of(orch, "failed_records") == ["failed_records_left"]
    (tmp_path / "secrets.toml").chmod(0o600)
    full = _take_over(first, sched, clock, 3, on_failed=records)
    assert written == []
    final = full.run()
    assert written == ["failed"] and not final.outcome.records_left
    assert not PlanStore(first.store, first.layout).latest().outcome.records_left
    assert _kinds_of(full, "failed_records") == ["failed_records_written"]


@pytest.mark.parametrize("refusal", [RecordsRefused("the task module of the failed instance does not load"),
                                     OSError("stale file handle")], ids=["refused", "store"])
def test_a_take_over_whose_writer_fails_on_the_left_records_still_cancels(tmp_path, refusal):
    """The records were left and a cancel is still outstanding when an orchestrator that can read
    the file takes over: its pass cancels first. A refusal leaves the mark for the next without
    stopping the take-over; a store that keeps failing fails its passes, after the cancel."""
    sched, clock = FakeScheduler(), Clock()
    first = _failed_with_a_dependent(tmp_path, sched, clock)
    cancel, lost = sched.cancel, []

    def unanswered_once(job_id):
        if not lost:
            lost.append(job_id)
            raise SchedulerUnavailable("Socket timed out")
        cancel(job_id)

    sched.cancel = unanswered_once
    left = _take_over(first, sched, clock, 2, on_failed=_needs_secrets, cancel_only=True)
    plan, _, _ = left.step()
    assert plan.outcome.records_left and sched.jobs["101"].state is JobState.PENDING
    left.close()
    (tmp_path / "secrets.toml").chmod(0o600)

    asked = []

    def refusing(store, ended):
        asked.append(ended.revision)
        raise refusal

    full = _take_over(first, sched, clock, 3, on_failed=refusing)
    if isinstance(refusal, OSError):
        with pytest.raises(OSError, match="stale file handle"):
            full.run()
        assert sched.jobs["101"].state is JobState.CANCELLED
        assert PlanStore(first.store, first.layout).latest().outcome.records_left
        return
    for _ in range(4):
        final, done, _ = full.step()
    full.close()
    assert done and sched.jobs["101"].state is JobState.CANCELLED and len(asked) == 1
    assert final.outcome.records_left and _kinds_of(full, "failed_records") == ["failed_records_refused"]


def test_a_lease_lost_while_the_left_records_are_written_stops_the_pass(tmp_path):
    """A take-over whose writer finds the lease lost stops at once: no refusal is recorded and the
    mark stays, for whichever orchestrator holds the lease now."""
    sched, clock = FakeScheduler(), Clock()
    first = _failed_with_a_dependent(tmp_path, sched, clock)
    left = _take_over(first, sched, clock, 2, on_failed=_needs_secrets, cancel_only=True)
    assert left.run().outcome.records_left and sched.jobs["101"].state is JobState.CANCELLED
    (tmp_path / "secrets.toml").chmod(0o600)

    def deposed(store, ended):
        raise LockLost("generation 4 took the lease over")

    full = _take_over(first, sched, clock, 3, on_failed=deposed)
    with pytest.raises(LockLost, match="generation 4"):
        full.step()
    full.close()
    assert PlanStore(first.store, first.layout).latest().outcome.records_left
    assert _kinds_of(full, "failed_records") == []


def test_a_writer_that_refuses_when_the_instance_ends_holds_back_no_cancel_and_is_asked_once(tmp_path):
    """The orchestrator that first ends the instance failed, with the secrets file readable, has
    a writer that refuses: it writes the revision that ends the instance with the records left,
    cancels the dependent, and does not ask again in its later passes."""
    sched, clock = FakeScheduler(), Clock()
    first = _failed_with_a_dependent(tmp_path, sched, clock, cancelled=False)
    asked = []

    def refusing(store, ended):
        asked.append(sched.jobs["101"].state)
        raise RecordsRefused("the task module of the failed instance does not load")

    orch = _take_over(first, sched, clock, 2, on_failed=refusing)
    for _ in range(4):
        plan, done, _ = orch.step()
    orch.close()
    assert (plan.outcome.status, plan.outcome.records_left) == ("failed", True) and done
    assert sched.jobs["101"].state is JobState.CANCELLED and asked == [JobState.PENDING]
    assert _kinds_of(orch, "failed_records") == ["failed_records_left"]


def test_a_lease_lost_while_the_records_are_written_ends_the_pass_with_nothing_recorded(tmp_path):
    """A lost lease is not a writer's failure: the pass stops at once, and the orchestrator
    neither marks the instance nor cancels."""
    sched, clock = FakeScheduler(), Clock()
    first = _failed_with_a_dependent(tmp_path, sched, clock, cancelled=False)

    def deposed(store, ended):
        raise LockLost("generation 3 took the lease over")

    orch = _take_over(first, sched, clock, 2, on_failed=deposed)
    with pytest.raises(LockLost, match="generation 3"):
        orch.step()
    orch.close()
    assert PlanStore(first.store, first.layout).latest().outcome is None
    assert _kinds_of(orch, "failed_records") == [] and sched.jobs["101"].state is JobState.PENDING


def test_a_writer_that_fails_on_the_store_when_the_instance_ends_is_asked_again_after_the_cancel(tmp_path):
    """A store error is not a refusal: the revision that ends the instance is marked and the
    dependent cancelled all the same, and the same pass asks the writer again, which writes the
    records and clears the mark."""
    sched, clock = FakeScheduler(), Clock()
    first = _failed_with_a_dependent(tmp_path, sched, clock, cancelled=False)
    asked = []

    def stalls_once(store, ended):
        asked.append(sched.jobs["101"].state)
        if len(asked) == 1:
            raise OSError("stale file handle")
        return {"score": f"{ended.layout.root}/failed/score.json"}

    orch = _take_over(first, sched, clock, 2, on_failed=stalls_once)
    plan, _, _ = orch.step()
    orch.close()
    assert plan.outcome.status == "failed" and not plan.outcome.records_left
    assert asked == [JobState.PENDING, JobState.CANCELLED]
    assert _kinds_of(orch, "failed_records") == ["failed_records_left", "failed_records_written"]


def test_a_cancelling_orchestrator_without_a_scrubber_withholds_what_failed_in_its_record_and_log(tmp_path,
                                                                                                 console):
    """A pass that failed on the store and a cancel that broke in the adapter are recorded and
    logged as withheld, since this orchestrator could not scrub them."""
    plan = noop_plan(tmp_path, site={"secrets_path": _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')})
    sched, clock = FakeScheduler(), Clock()
    first = Orchestrator.create(plan, sched, holder=HOLDER, clock=clock, sleep=lambda s: None)
    first.step()
    first.close()
    (tmp_path / "secrets.toml").chmod(0o644)
    request_cancel(first.store, first.layout, at=clock.t)
    (states, cancel), stalled, broke = (sched.states, sched.cancel), [], []

    def stalls_once(job_ids):
        if not stalled:
            stalled.append(1)
            raise OSError("the store stalled reading hls-noop")
        return states(job_ids)

    def breaks_once(job_id):
        if not broke:
            broke.append(1)
            raise RuntimeError("the adapter broke cancelling hls-noop")
        cancel(job_id)

    sched.states, sched.cancel = stalls_once, breaks_once
    orch = Orchestrator.attach(first.store, first.layout, sched, holder=HOLDER.model_copy(update={"pid": 2}),
                               alive=lambda h: False, clock=clock, sleep=lambda s: None, cancel_only=True)
    assert orch.run().outcome.status == "cancelled"
    events = read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))
    assert [e["error"] for e in events if e["kind"] in ("pass_failed", "effect_failed")] == [WITHHELD, WITHHELD]
    shown = console.getvalue()
    assert shown.count(WITHHELD) >= 2 and "the store stalled" not in shown and "the adapter broke" not in shown


def test_the_orchestrators_log_lines_are_scrubbed_while_it_runs(tmp_path, console):
    """Its console lines say no more than its records: every line of the process is scrubbed with
    its scrubber from its start to its end, a line of another module's included."""
    value = "console-value-0123"
    plan = noop_plan(tmp_path, site={"secrets_path": _secrets_file(tmp_path, f'[secrets]\nshown = "{value}"\n')})
    orch = Orchestrator.create(plan, Answering(RuntimeError(f"the adapter broke on {value}")), holder=HOLDER,
                               clock=Clock(), sleep=lambda s: None)
    orch.step()
    logging.getLogger("hls.store").warning("a store line naming %s", value)
    shown = console.getvalue()
    assert "Submit of allocation 0 submission 1 failed: Traceback" in shown
    assert shown.count("<redacted:shown>") >= 2 and value not in shown
    orch.close()
    logging.getLogger("hls.store").warning("%s after the orchestrator", value)
    assert console.getvalue().endswith(f"{value} after the orchestrator\n")


def test_a_taking_over_orchestrator_whose_secrets_file_cannot_be_used_takes_no_lease(tmp_path):
    plan = noop_plan(tmp_path, site={"secrets_path": _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')})
    first = Orchestrator.create(plan, FakeScheduler(), holder=HOLDER, clock=Clock(), sleep=lambda s: None)
    first.close()
    (tmp_path / "secrets.toml").chmod(0o644)
    with pytest.raises(SecretsError, match="secrets.toml"):
        _attach(first, FakeScheduler(), Clock())
    lease = instance_lease(first.store, first.layout, ttl_sec=30).current()
    assert lease.generation == first.handle.generation and lease.released


def test_an_orchestrator_whose_secrets_file_cannot_be_used_takes_no_lease(tmp_path):
    """It could not scrub what it records, so it refuses before the instance is written."""
    plan = noop_plan(tmp_path, site={"secrets_path": _secrets_file(tmp_path, '[secrets]\nnamed = "hls-noop"\n')})
    (tmp_path / "secrets.toml").chmod(0o644)
    with pytest.raises(SecretsError, match="secrets.toml"):
        Orchestrator.create(plan, FakeScheduler(), holder=HOLDER, clock=Clock(), sleep=lambda s: None)
    assert PlanStore(PosixStore(plan.store_root), plan.layout).latest() is None
    assert instance_lease(PosixStore(plan.store_root), plan.layout, ttl_sec=30).current() is None


def test_create_checks_each_allocation_then_submits_the_chain(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    assert [c[0] for c in sched.calls] == ["test_only", "test_only"]
    plan, done, acted = orch.step()
    assert acted and not done and sched.submits() == [("submit", f"hls:{plan.instance}/0/1", None)]
    plan, _, _ = orch.step()
    assert sched.submits()[1] == ("submit", f"hls:{plan.instance}/1/1", "100")
    assert [a.current().job_id for a in plan.allocations] == ["100", "101"]
    job = sched.jobs["101"].request
    assert job.script == orch.store.path(plan.layout.submission_file(1, 1, "batch.sh"))
    line = submit_argv(job)
    assert "--no-requeue" in line and "--export=NIL" in line and "--gpus=h200:2" in line
    assert "--allocation 1 --submission 1" in job.script.read_text()


def test_dying_between_sbatch_and_recording_the_id_never_submits_twice(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)

    def die(job):
        raise SystemExit("orchestrator killed after sbatch, before the job id was recorded")

    sched.after_create = die
    with pytest.raises(SystemExit):
        orch.step()
    sched.after_create = None
    plan = PlanStore(orch.store, orch.layout).latest()
    assert plan.allocation(0).submissions[0].status == "intended"
    clock.t += 5
    again = _attach(orch, sched, clock)
    plan, _, _ = again.step()
    sub = plan.allocation(0).current()
    assert (sub.n, sub.status, sub.job_id) == (1, "submitted", "100")
    assert len(sched.submits()) == 1
    assert plan.lease_generation == 2


def test_a_lost_sbatch_answer_is_looked_up_by_its_tag(tmp_path):
    sched, clock = FakeScheduler(submit_fault="lose_answer"), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sub = plan.allocation(0).submissions[0]
    assert (sub.status, sub.job_id, sub.error) == ("intended", None, "sbatch timed out")
    plan, _, _ = orch.step()
    assert plan.allocation(0).current().job_id == "100"
    assert len(sched.submits()) == 1
    assert ("find_by_tags", (sub.tag,)) in sched.calls


def test_a_tag_that_never_turns_up_is_submitted_again_after_the_window(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    sched.after_create = lambda job: sched.jobs.pop(job.job_id)
    sched.submit_fault = "lose_answer"
    orch.step()
    sched.after_create = None
    orch.step()
    assert len(sched.submits()) == 1
    _advance(orch, clock, 121)
    plan, _, _ = orch.step()
    assert [s.status for s in plan.allocation(0).submissions] == ["absent", "submitted"]
    assert len(sched.submits()) == 2


def test_reattaching_never_relaunches_a_running_allocation(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    _heartbeat(plan, 0, 1, clock.t)
    again = _attach(orch, sched, clock)
    for _ in range(3):
        clock.t += 1
        _heartbeat(plan, 0, 1, clock.t)
        plan, done, _ = again.step()
    assert not done and len(sched.submits()) == 1
    _publish(plan, "noop_a")
    _publish(plan, "noop_b")
    plan, done, _ = again.step()
    assert done and plan.outcome.status == "succeeded"
    assert len(sched.submits()) == 1 and sched.cancels() == []


def test_attach_waits_for_a_live_orchestrator(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    with pytest.raises(LockHeld):
        Orchestrator.attach(orch.store, orch.layout, sched, holder=HOLDER, alive=lambda h: None, clock=clock)


def test_a_stale_orchestrator_stops_without_acting(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    _attach(orch, sched, clock)
    with pytest.raises(LockLost):
        orch.step()
    assert sched.submits() == []


def test_a_preflight_refusal_voids_before_anything_is_submitted(tmp_path):
    sched, clock = FakeScheduler(test_only_refusal="Invalid account or account/partition combination"), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, done, _ = orch.step()
    assert done and plan.outcome.status == "void" and "preflight" in plan.outcome.reason
    assert sched.submits() == []


def test_hls_cancel_reaches_the_orchestrator_holding_the_lease(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    request_cancel(orch.store, orch.layout, at=clock.t)
    request_cancel(orch.store, orch.layout, at=clock.t)
    plan, done, _ = orch.step()
    assert plan.outcome.status == "cancelled" and sorted(sched.cancels()) == ["100", "101"]
    assert all(s.status == "cancelled" for a in plan.allocations for s in a.submissions)
    assert done


def test_a_failed_allocation_is_retried_and_its_pending_dependent_rewired(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    assert sched.jobs["101"].reason == "DependencyNeverSatisfied"
    plan, _, _ = orch.step()
    assert plan.allocation(2).retry_of == 0 and plan.allocation(2).current().job_id == "102"
    plan, _, _ = orch.step()
    assert ("rewire", "101", "102") in sched.calls and sched.jobs["101"].after == "102"
    assert plan.allocation(1).current().after_job == "102"


def test_run_passes_until_the_instance_ends(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan = PlanStore(orch.store, orch.layout).latest()
    sleeps = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        clock.t += sec
        if len(sleeps) == 2:
            _publish(plan, "noop_a")
            _publish(plan, "noop_b")

    orch.sleep = sleep
    sched.after_create = lambda job: sched.start(job.job_id, at=clock.t)
    final = orch.run()
    assert final.outcome.status == "succeeded" and sleeps == [1, 1]
    assert orch.handle.lock.current().released


def test_a_store_error_in_a_pass_is_tried_again_at_the_next_poll(tmp_path):
    # A transient filesystem error or a lost plan race must not leave the instance driven by nobody.
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan = PlanStore(orch.store, orch.layout).latest()
    real_exists = orch.store.exists
    failures: list[Exception] = [OSError("stale file handle"), PlanConflict("raced")]

    def exists(key):
        if failures:
            raise failures.pop(0)
        return real_exists(key)

    orch.store.exists = exists
    sleeps = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        clock.t += sec
        if len(sleeps) == 4:
            _publish(plan, "noop_a")
            _publish(plan, "noop_b")

    orch.sleep = sleep
    sched.after_create = lambda job: sched.start(job.job_id, at=clock.t)
    final = orch.run()
    assert final.outcome.status == "succeeded" and sleeps[:2] == [1, 1]
    failed = [(e["error"], e["in_a_row"]) for e in _events(orch) if e["kind"] == "pass_failed"]
    assert failed == [("OSError: stale file handle", 1), ("PlanConflict: raced", 2)]


def test_a_pass_that_keeps_failing_ends_the_orchestrator_after_a_bounded_number_of_tries(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)

    def exists(key):
        raise OSError("gone")

    orch.store.exists = exists
    sleeps = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        clock.t += sec

    orch.sleep = sleep
    with pytest.raises(OSError, match="gone"):
        orch.run()
    assert len(sleeps) == orchestrator_module.MAX_FAILED_PASSES - 1
    assert sum(e["kind"] == "pass_failed" for e in _events(orch)) == orchestrator_module.MAX_FAILED_PASSES


def test_a_lost_lease_ends_the_orchestrator_at_once(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)

    def exists(key):
        raise LockLost("taken over")

    orch.store.exists = exists
    sleeps = []
    orch.sleep = sleeps.append
    with pytest.raises(LockLost, match="taken over"):
        orch.run()
    assert sleeps == [] and "pass_failed" not in _kinds(orch)


# The two tests below run on real time, since the keep-alive renews on real time. A 3 s lease
# is renewed every 0.75 s and counts as lost 2.25 s after its last heartbeat.


class SlowPreflight(FakeScheduler):
    """A controller that takes *sec* seconds to answer ``sbatch --test-only``."""

    def __init__(self, sec: float) -> None:
        super().__init__()
        self.sec = sec

    def test_only(self, request):
        time.sleep(self.sec)
        return super().test_only(request)


def test_a_create_slower_than_the_lease_keeps_the_lease_into_the_run(tmp_path):
    # A slow sbatch --test-only must not lose the lease before the first job id is recorded.
    # Here the preflight takes 0.87 of a 3 s lease.
    sched = SlowPreflight(2.6)
    orch = Orchestrator.create(noop_plan(tmp_path, lease_ttl_sec=3), sched, holder=HOLDER, sleep=lambda s: None)
    plan, _, _ = orch.step()
    assert plan.allocation(0).current().job_id == "100"
    lease = orch.handle.lock.current()
    assert lease.generation == 1 and lease.heartbeat_at > lease.acquired_at
    sched.start("100", at=time.time())
    _publish(plan, "noop_a")
    _publish(plan, "noop_b")
    assert orch.run().outcome.status == "succeeded"
    assert orch.handle.lock.current().released


def test_attach_keeps_the_lease_alive_before_the_run(tmp_path):
    first = Orchestrator.create(noop_plan(tmp_path, lease_ttl_sec=3), FakeScheduler(), holder=HOLDER,
                                sleep=lambda s: None)
    again = Orchestrator.attach(first.store, first.layout, FakeScheduler(), holder=HOLDER.model_copy(update={"pid": 2}),
                                alive=lambda h: False, sleep=lambda s: None)
    time.sleep(1.0)
    lease = again.handle.lock.current()
    assert lease.generation == 2 and lease.heartbeat_at > lease.acquired_at


# Helpers for the tests below.


def _events(orch: Orchestrator) -> list[dict]:
    return read_events(orch.store.path(orch.layout.orchestrator_file(orch.handle.generation, "events.jsonl")))


def _kinds(orch: Orchestrator) -> list[str]:
    return [e["kind"] for e in _events(orch)]


def _revisions(orch: Orchestrator) -> int:
    return len(PlanStore(orch.store, orch.layout).revisions())


class NoCancel(FakeScheduler):
    """scancel never reaches the controller."""

    def cancel(self, job_id):
        self.calls.append(("cancel", job_id))
        raise SchedulerUnavailable("Socket timed out")


class RewireFails(FakeScheduler):
    """scontrol update fails with ``error``."""

    error: Exception = SchedulerUnavailable("Unable to contact slurm controller")

    def rewire(self, job_id, after):
        self.calls.append(("rewire", job_id, after))
        raise self.error


# A dependent whose predecessor published everything must not wait for a job that ended badly.


def _finish_run(orch, sched, clock, plan):
    for _ in range(6):
        _advance(orch, clock, 5)
        plan, done, _ = orch.step()
    sub = plan.allocation(1).current()
    assert sub is not None and sub.n == 2 and sub.after_job is None and sub.job_id is not None
    assert plan.allocation(1).submissions[0].status == "cancelled"
    sched.start(sub.job_id, at=clock.t)
    _publish(plan, "noop_b")
    plan, done, _ = orch.step()
    assert done and plan.outcome.status == "succeeded"


@pytest.mark.parametrize("ending", [JobState.OUT_OF_MEMORY, JobState.FAILED, JobState.TIMEOUT])
def test_a_dependent_of_a_done_predecessor_whose_job_failed_runs_without_the_dependency(tmp_path, ending):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    plan = PlanStore(orch.store, orch.layout).latest()
    _publish(plan, "noop_a")
    sched.finish("100", ending)
    _finish_run(orch, sched, clock, plan)
    assert "101" in sched.cancels()


def test_a_retry_that_finished_before_the_rewire_leaves_no_dependent_stuck(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    plan, _, _ = orch.step()
    sched.start("102", at=clock.t)
    _publish(plan, "noop_a")
    sched.finish("102", JobState.COMPLETED)
    _advance(orch, clock, 40)
    again = _attach(orch, sched, clock)
    _finish_run(again, sched, clock, plan)
    assert "101" in sched.cancels()


def test_a_dependent_of_a_completed_predecessor_is_left_to_slurm(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    plan = PlanStore(orch.store, orch.layout).latest()
    _publish(plan, "noop_a")
    sched.finish("100", JobState.COMPLETED)
    for _ in range(3):
        _advance(orch, clock, 5)
        orch.step()
    assert sched.cancels() == [] and len(sched.submits()) == 2


# Passes that change nothing sleep, and a controller that does not answer costs no revisions.


def test_cancels_the_controller_does_not_answer_are_not_a_revision_each(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    request_cancel(orch.store, orch.layout, at=clock.t)
    sched.down = True
    before = _revisions(orch)
    results = [orch.step() for _ in range(50)]
    assert _revisions(orch) - before == 2
    assert [acted for _, _, acted in results] == [True] + [False] * 49
    assert not any(done for _, done, _ in results)
    sched.down = False
    plan, done, _ = orch.step()
    assert done and all(s.status == "cancelled" for a in plan.allocations for s in a.submissions)
    assert sched.jobs["100"].state is JobState.CANCELLED and sched.jobs["101"].state is JobState.CANCELLED


def test_an_unanswered_rewire_does_not_spin(tmp_path):
    sched, clock = RewireFails(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    orch.step()
    results = [orch.step() for _ in range(20)]
    assert not any(acted for _, _, acted in results[1:])
    sub = results[-1][0].allocation(1).submissions[0]
    # Unknown whether scontrol took effect: the intent stays, and the pending job is left alone.
    assert (sub.status, sub.rewire_to, "Unable to contact" in sub.error) == ("submitted", "102", True)
    assert sched.cancels() == []


@pytest.mark.parametrize("error, cancelled", [
    (Refused("Access/permission denied"), False),
    (DependencyRefused("Job dependency problem"), True),
])
def test_only_a_dependency_refusal_of_a_rewire_cancels_the_dependent(tmp_path, error, cancelled):
    sched, clock = RewireFails(), Clock()
    sched.error = error
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    orch.step()
    plan, _, _ = orch.step()
    sub = plan.allocation(1).submissions[0]
    assert sub.status == ("cancel_intended" if cancelled else "submitted")
    # A refusal for another reason keeps the intent, and the next pass asks again.
    assert sub.rewire_to == (None if cancelled else "102")


def test_perform_says_what_scontrol_and_scancel_answered(tmp_path):
    sched, clock = RewireFails(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    plan, _, _ = orch.step()
    cases = [(SchedulerUnavailable("down"), Rewired(1, 1, "100", ok=False, error="down")),
             (Refused("denied"), Rewired(1, 1, "100", ok=False, refused=True, error="denied")),
             (DependencyRefused("dep"), Rewired(1, 1, "100", ok=False, refused=True, dependency=True, error="dep"))]
    for error, expected in cases:
        sched.error = error
        assert orch.perform(plan, Rewire(1, 1, "100")) == expected


def test_run_sleeps_only_after_a_pass_that_changed_nothing(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan = PlanStore(orch.store, orch.layout).latest()
    acted, slept_after = [], []
    real = orch.step

    def step():
        assert len(acted) < 20, "runaway"
        out = real()
        acted.append(out[2])
        return out

    def sleep(sec: float) -> None:
        slept_after.append(len(acted))
        clock.t += sec
        if len(slept_after) == 2:
            _publish(plan, "noop_a")
            _publish(plan, "noop_b")

    orch.step, orch.sleep = step, sleep
    sched.after_create = lambda job: sched.start(job.job_id, at=clock.t)
    orch.run()
    assert acted == [True, False, False, True] and slept_after == [2, 3]


# The lease is confirmed before anything is done in its name.


@pytest.mark.parametrize("stall_sec", [0.0, 200.0])
def test_a_holder_taken_over_between_its_write_and_its_effects_never_submits(tmp_path, stall_sec):
    # With a stall the process and its keep-alive thread stopped, so its heartbeat is old; with
    # none the takeover shows only in the store.
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    real_write, taken = orch.plans.write, []

    def write_then_stall(plan, handle):
        out = real_write(plan, handle)
        if not taken:
            clock.t += stall_sec
            taken.append(_attach(orch, sched, clock))
        return out

    orch.plans.write = write_then_stall
    with pytest.raises(LockLost):
        orch.step()
    assert sched.submits() == []


def test_each_effect_checks_the_lease_before_it_runs(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    request_cancel(orch.store, orch.layout, at=clock.t)
    real_perform = orch.perform

    def perform_then_stall(plan, effect):
        out = real_perform(plan, effect)
        clock.t += 200
        return out

    orch.perform = perform_then_stall
    with pytest.raises(LockLost):
        orch.step()
    assert sched.cancels() == ["100"]


def test_a_holder_whose_lease_lapsed_stops_before_it_looks_at_anything(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    clock.t += 100
    sched.calls.clear()
    with pytest.raises(LockLost):
        orch.step()
    assert sched.calls == []


def test_the_takeover_revision_is_written_before_anything_else(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    before = PlanStore(orch.store, orch.layout).latest()
    again = _attach(orch, sched, clock)
    taken = PlanStore(again.store, again.layout).latest()
    assert (taken.revision, taken.lease_generation) == (before.revision + 1, 2)
    assert taken.allocations == before.allocations and "attached" in _kinds(again)


# What one pass reads: job states before manifests, and only the statuses that are the job's.


def test_a_job_that_publishes_and_ends_between_the_two_reads_does_not_void_the_instance(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, retry={"noop_b": {"enabled": False}})
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    _publish(plan, "noop_a")
    real_states = sched.states

    def states(ids):
        if "100" in ids and sched.jobs["100"].state is JobState.RUNNING:
            _publish(plan, "noop_b")
            sched.finish("100", JobState.COMPLETED)
        return real_states(ids)

    sched.states = states
    plan, done, _ = orch.step()
    assert done and plan.outcome.status == "succeeded"


def test_a_runner_that_keeps_reporting_is_never_declared_dead(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    for _ in range(6):  # 30 s of heartbeats against a 10 s limit
        _advance(orch, clock, 5)
        _heartbeat(plan, 0, 1, clock.t)
        plan, _, _ = orch.step()
    assert sched.cancels() == [] and len(sched.submits()) == 1 and plan.allocation(0).failure is None


def test_the_runners_own_fault_is_the_failure_reason(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    grant = FaultRecord(fault_class="infra", reason="grant mismatch: gpus: requested 2, granted 1", retryable=True)
    _heartbeat(plan, 0, 1, clock.t, fault=grant)
    sched.finish("100", JobState.FAILED)
    plan, _, _ = orch.step()
    assert plan.allocation(0).failure.fault.reason == grant.reason


def test_a_status_that_is_unreadable_or_another_jobs_is_ignored_with_an_event(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    orch.store.write_bytes(plan.layout.allocation_status(0, 1), b'{"instance": "not a status"')
    plan, _, _ = orch.step()
    assert plan.allocation(0).failure is None and "status_ignored" in _kinds(orch)
    # A status of another job is no heartbeat of this one: 20 s on, the job counts as silent.
    for _ in range(4):
        _advance(orch, clock, 5)
        _heartbeat(plan, 0, 1, clock.t, job_id="999")
    plan, _, _ = orch.step()
    assert plan.allocation(0).failure.fault.reason.startswith("job 100: no heartbeat for")
    ignored = [e for e in _events(orch) if e["kind"] == "status_ignored"]
    assert any("999" in e["reason"] for e in ignored)


def test_the_runners_status_names_a_job_whose_sbatch_answer_was_lost(tmp_path):
    sched, clock = FakeScheduler(submit_fault="lose_answer"), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.tags_down = True
    sched.start("100", at=clock.t)
    _heartbeat(plan, 0, 1, clock.t)
    plan, _, _ = orch.step()
    sub = plan.allocation(0).current()
    assert (sub.status, sub.job_id) == ("submitted", "100") and len(sched.submits()) == 1


def test_one_tag_lookup_a_pass_and_no_state_query_for_a_job_that_ended(tmp_path):
    sched, clock = FakeScheduler(submit_fault="lose_answer"), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    assert [c for c in sched.calls if c[0] == "find_by_tags"]
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    orch.step()
    sched.calls.clear()
    orch.step()
    states = [c for c in sched.calls if c[0] == "states"]
    assert len(states) == 1 and "100" not in states[0][1]


# A lost sbatch answer.


def test_a_slow_sbatch_is_not_judged_absent_by_its_age(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    hidden = {}

    def slow_and_late(job):
        # sbatch blocks for its 300 s timeout, and the controller shows the job only after that.
        hidden[job.job_id] = sched.jobs.pop(job.job_id)
        _advance(orch, clock, 300)

    sched.after_create = slow_and_late
    sched.submit_fault = "lose_answer"
    orch.step()
    sched.after_create = None
    plan, _, _ = orch.step()
    assert [s.status for s in plan.allocation(0).submissions] == ["intended"]
    sched.jobs.update(hidden)
    plan, _, _ = orch.step()
    assert plan.allocation(0).current().job_id == "100" and len(sched.submits()) == 1


def test_a_job_that_appears_after_its_submission_was_declared_absent_is_cancelled(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    hidden = {}
    sched.after_create = lambda job: hidden.update({job.job_id: sched.jobs.pop(job.job_id)})
    sched.submit_fault = "lose_answer"
    orch.step()
    sched.after_create = None
    orch.step()
    _advance(orch, clock, 121)
    plan, _, _ = orch.step()
    assert [s.status for s in plan.allocation(0).submissions] == ["absent", "submitted"]
    sched.jobs.update(hidden)
    plan, _, _ = orch.step()
    first, second = plan.allocation(0).submissions
    assert (first.status, first.job_id) == ("cancelled", "100") and second.status == "submitted"
    assert sched.cancels() == ["100"] and sched.jobs["100"].state is JobState.CANCELLED
    assert plan.outcome is None


def test_a_cancel_reaches_a_job_whose_sbatch_answer_was_lost_and_unfound(tmp_path):
    sched, clock = FakeScheduler(submit_fault="lose_answer"), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    sched.tags_down = True
    request_cancel(orch.store, orch.layout, at=clock.t)
    plan, done, _ = orch.step()
    assert plan.allocation(0).submissions[0].status == "cancel_intended" and not done
    sched.tags_down = False
    plan, done, _ = orch.step()
    assert done and sched.cancels() == ["100"] and plan.allocation(0).submissions[0].status == "cancelled"


def test_three_dependency_refusals_in_a_row_void_the_instance(tmp_path):
    class RefuseDeps(FakeScheduler):
        def submit(self, request):
            if request.after is None:
                return super().submit(request)
            self.calls.append(("submit", request.tag, request.after))
            raise DependencyRefused("Batch job submission failed: Job dependency problem")

    sched, clock = RefuseDeps(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    for _ in range(10):
        plan, done, _ = orch.step()
        if done:
            break
    assert done and plan.outcome.status == "void"
    assert plan.outcome.reason.startswith("allocation 1: submissions 1, 2, 3 were refused or lost")
    assert len(plan.allocation(1).submissions) == 3 and sched.cancels() == ["100"]


def test_a_dependency_refusal_is_retried_at_the_next_pass_not_voided(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    sched.purge("100")
    plan, _, _ = orch.step()
    assert plan.outcome is None and plan.allocation(1).submissions[0].status == "absent"


# The instance ends.


def test_an_instance_that_succeeds_waits_for_the_cancel_of_a_hung_job(tmp_path):
    sched, clock = NoCancel(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    _heartbeat(plan, 0, 1, clock.t)
    _advance(orch, clock, 20)
    plan, _, _ = orch.step()  # job 100 is cancelled (the controller does not hear), and retried as 101
    assert plan.allocation(1).current().job_id == "101"
    sched.start("101", at=clock.t)
    _publish(plan, "noop_a")
    _publish(plan, "noop_b")
    plan, done, _ = orch.step()
    assert plan.outcome.status == "succeeded" and not done
    assert plan.allocation(0).submissions[0].status == "cancel_intended"
    sched.finish("100", JobState.CANCELLED)
    plan, done, _ = orch.step()
    assert done and plan.allocation(0).submissions[0].status == "cancelled"


def test_a_cancel_that_failed_is_not_repeated_once_the_job_ended(tmp_path):
    sched, clock = NoCancel(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    request_cancel(orch.store, orch.layout, at=clock.t)
    plan, done, _ = orch.step()
    sub = plan.allocation(0).submissions[0]
    assert sub.status == "cancel_intended" and "Socket timed out" in sub.error and not done
    assert sched.cancels() == ["100"]
    sched.jobs["100"].state = JobState.FAILED
    plan, done, _ = orch.step()
    assert done and plan.allocation(0).submissions[0].status == "cancelled" and sched.cancels() == ["100"]


def test_the_orchestrator_reports_its_memory_every_so_many_passes(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator_module, "MEMORY_REPORT_PASSES", 2)
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    for _ in range(4):
        orch.step()
    reports = [e for e in _events(orch) if e["kind"] == "orchestrator_memory"]
    assert len(reports) == 2 and isinstance(reports[0]["rss"], int) and reports[0]["rss"] > 0
    assert _events(orch)[0]["kind"] == "instance_created" and _events(orch)[0]["memory"]["rss"] > 0


def test_login_memory_reads_this_process(tmp_path):
    got = login_memory()
    assert got["rss"] > 1_000_000 and (got["limit"] is None or got["limit"] > 0)


def test_login_memory_takes_the_tightest_limit_of_the_cgroups_above_this_process(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "status").write_text("Name:\tpython\nVmRSS:\t    2048 kB\n")
    (proc / "cgroup").write_text("0::/user.slice/user-1.slice/session-2.scope\n")
    root = tmp_path / "cg"
    for d, limit in [("", None), ("user.slice", "max"), ("user.slice/user-1.slice", "17179869184"),
                     ("user.slice/user-1.slice/session-2.scope", "34359738368")]:
        (root / d).mkdir(parents=True, exist_ok=True)
        if limit is not None:
            (root / d / "memory.max").write_text(f"{limit}\n")
    assert login_memory(proc=proc, cgroup_root=root) == {"rss": 2048 * 1024, "limit": 17179869184}
    (proc / "cgroup").write_text("1:name=systemd:/\n")
    assert login_memory(proc=proc, cgroup_root=root) == {"rss": 2048 * 1024, "limit": None}


# Dying, failing and conflicting.


def test_a_write_that_conflicts_does_nothing_and_logs_nothing(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    real_write, raised = orch.plans.write, []

    def conflict_once(plan, handle):
        if not raised:
            raised.append(1)
            raise PlanConflict("another revision came first")
        return real_write(plan, handle)

    orch.plans.write = conflict_once
    plan, done, acted = orch.step()
    assert (done, acted) == (False, True) and sched.submits() == [] and "submit_intended" not in _kinds(orch)
    orch.step()
    assert len(sched.submits()) == 1 and "submit_intended" in _kinds(orch)


def test_an_effect_that_raises_is_recorded_and_the_others_still_are(tmp_path):
    class FlakyCancel(FakeScheduler):
        def cancel(self, job_id):
            if job_id == "100":
                self.calls.append(("cancel", job_id))
                raise OSError("Stale file handle")
            super().cancel(job_id)

    sched, clock = FlakyCancel(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    request_cancel(orch.store, orch.layout, at=clock.t)
    plan, done, _ = orch.step()
    first, second = (a.submissions[0] for a in plan.allocations)
    assert (first.status, second.status) == ("cancel_intended", "cancelled") and not done
    assert "Stale file handle" in first.error and "effect_failed" in _kinds(orch)


def test_a_submission_that_raises_before_sbatch_is_found_absent_and_tried_again(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    real_submit = sched.submit

    def broken(request):
        raise OSError("No space left on device")

    sched.submit = broken
    plan, _, _ = orch.step()
    sub = plan.allocation(0).submissions[0]
    assert sub.status == "intended" and "No space left" in sub.error
    sched.submit = real_submit
    orch.step()
    _advance(orch, clock, 121)
    plan, _, _ = orch.step()
    assert [s.status for s in plan.allocation(0).submissions] == ["absent", "submitted"]


def test_create_releases_the_lease_when_the_instance_exists(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    first = _create(tmp_path, sched, clock)
    first.handle.release()
    with pytest.raises(InstanceExists):
        _create(tmp_path, sched, clock)
    record = first.handle.lock.current()
    assert record.generation == 2 and record.released


def test_create_releases_the_lease_when_the_preflight_breaks(tmp_path):
    class Broken(FakeScheduler):
        def test_only(self, request):
            raise RuntimeError("scheduler adapter bug")

    plan = noop_plan(tmp_path)
    with pytest.raises(RuntimeError):
        Orchestrator.create(plan, Broken(), holder=HOLDER, clock=Clock(), sleep=lambda s: None)
    lease = instance_lease(PosixStore(plan.store_root), plan.layout, ttl_sec=30)
    assert lease.current().released


def test_attach_releases_the_lease_when_the_takeover_cannot_be_written(tmp_path, monkeypatch):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)

    def broken(self, plan, handle):
        raise RuntimeError("store is read-only")

    monkeypatch.setattr(PlanStore, "write", broken)
    with pytest.raises(RuntimeError):
        _attach(orch, sched, clock)
    record = orch.handle.lock.current()
    assert record.generation == 2 and record.released


# Rewires, lost answers, strays and void instances.


class StrayCancel(FakeScheduler):
    """scancel fails for the ids in ``cancel_fails``, and the next call dies when ``die_once`` is set."""

    cancel_fails: frozenset = frozenset()
    die_once: bool = False

    def cancel(self, job_id):
        if self.die_once:
            self.die_once = False
            self.calls.append(("cancel", job_id))
            raise SystemExit("orchestrator killed inside scancel")
        if job_id in self.cancel_fails:
            self.calls.append(("cancel", job_id))
            raise SchedulerUnavailable("Socket timed out")
        super().cancel(job_id)


def _two_jobs_one_tag(tmp_path):
    sched, clock = StrayCancel(submit_fault="lose_answer"), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()  # job 100 exists, and sbatch's answer was lost
    sched.submit(sched.jobs["100"].request)  # the same tag once more: job 101
    return orch, sched, clock


def test_a_rewire_that_stays_refused_is_asked_a_few_times_and_then_the_job_is_submitted_again(tmp_path):
    sched, clock = RewireFails(), Clock()
    sched.error = Refused("scontrol: error: Access/permission denied")
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    orch.step()
    before = _revisions(orch)
    results = [orch.step() for _ in range(20)]
    assert sum(1 for c in sched.calls if c[0] == "rewire") == 3
    assert _revisions(orch) - before == 8
    # Recording one more refusal is not progress, so the loop sleeps between the three asks.
    assert [progress for _, _, progress in results[:6]] == [True, False, False, True, True, False]
    first, second = results[-1][0].allocation(1).submissions
    assert (first.status, second.status, second.after_job) == ("cancelled", "submitted", "102")


def test_a_dependent_slurm_already_released_is_not_cancelled_after_a_lost_rewire_answer(tmp_path):
    class LostRewire(FakeScheduler):
        def rewire(self, job_id, after):
            super().rewire(job_id, after)
            raise SchedulerUnavailable("Socket timed out on send/recv operation")

    sched, clock = LostRewire(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    orch.step()  # allocation 2 retries allocation 0 as job 102
    plan, _, _ = orch.step()  # job 101 is rewired to 102, and the answer is lost
    assert sched.jobs["101"].after == "102"
    sub = plan.allocation(1).current()
    assert (sub.after_job, sub.rewire_to) == ("100", "102")
    sched.start("102", at=clock.t)
    _publish(plan, "noop_a")
    sched.finish("102", JobState.COMPLETED)
    for _ in range(3):
        _advance(orch, clock, 5)
        plan, _, _ = orch.step()
    assert sched.cancels() == [] and len(sched.submits()) == 3
    sched.start("101", at=clock.t)
    _publish(plan, "noop_b")
    plan, done, _ = orch.step()
    assert done and plan.outcome.status == "succeeded"


def test_a_dependent_of_a_job_hls_cancelled_is_released_when_the_site_gives_no_reason(tmp_path):
    # Slurm's reason says "Dependency" for every pending job here; only the state of the job
    # the dependent waits for (100, cancelled by hls and not live any more) says it is stuck.
    sched, clock = FakeScheduler(pending_reason="Dependency"), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    orch.step()
    sched.start("100", at=clock.t)
    plan = PlanStore(orch.store, orch.layout).latest()
    _heartbeat(plan, 0, 1, clock.t)
    _advance(orch, clock, 20)
    orch.step()  # the silent job 100 is cancelled and its allocation retried as job 102
    sched.start("102", at=clock.t)
    _publish(plan, "noop_a")
    sched.finish("102", JobState.COMPLETED)
    again = _attach(orch, sched, clock)
    _finish_run(again, sched, clock, plan)
    assert "101" in sched.cancels()


def test_three_lost_answers_void_the_instance_but_it_waits_a_window_longer_for_a_late_job(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    sched.after_create = lambda job: sched.jobs.pop(job.job_id)
    done, passes = False, 0
    while not done and passes < 300:
        sched.submit_fault = "lose_answer"
        plan, done, _ = orch.step()
        _advance(orch, clock, 5)
        passes += 1
    subs = plan.allocation(0).submissions
    assert done and plan.outcome.status == "void" and [s.status for s in subs] == ["absent"] * 3
    assert clock.t - subs[-1].missing_since >= 2 * ABSENT_AFTER_SEC


def test_a_job_that_turns_up_after_the_void_is_cancelled_before_the_instance_is_done(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    hidden = {}
    sched.after_create = lambda job: hidden.update({job.job_id: sched.jobs.pop(job.job_id)})
    done, passes, revealed = False, 0, False
    while not done and passes < 300:
        sched.submit_fault = "lose_answer"
        plan, done, _ = orch.step()
        if plan.outcome is not None and not revealed:
            assert not done
            sched.jobs.update(hidden)
            revealed = True
        _advance(orch, clock, 5)
        passes += 1
    assert done and sorted(sched.cancels()) == ["100", "101", "102"]
    assert all(j.state is JobState.CANCELLED for j in sched.jobs.values())


def test_a_job_a_holder_submitted_after_its_takeover_is_cancelled_by_the_new_holder(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    real_perform, state = orch.perform, {}

    def stalled_perform(plan, effect):
        if "g2" not in state:
            clock.t += 200  # the process stops after its lease check
            state["g2"] = g2 = _attach(orch, sched, clock)
            for _ in range(30):
                clock.t += 5
                g2.handle.renew()
                g2.step()
        return real_perform(plan, effect)

    orch.perform = stalled_perform
    with pytest.raises(LockLost):
        orch.step()
    g2 = state["g2"]
    assert len([j for j in sched.jobs.values() if not j.state.terminal]) == 2
    clock.t += 5
    g2.handle.renew()
    g2.step()
    assert len([j for j in sched.jobs.values() if not j.state.terminal]) == 1


def test_a_submission_that_always_fails_is_named_in_the_void_reason(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)

    def broken(plan, k, n, **request):
        raise ValueError("--gpus=h200:99 exceeds the partition")

    orch._request = broken
    done, passes = False, 0
    while not done and passes < 400:
        plan, done, _ = orch.step()
        _advance(orch, clock, 5)
        passes += 1
    assert done and plan.outcome.status == "void"
    assert plan.outcome.reason.startswith("allocation 0: submissions 1, 2, 3 were refused or lost")
    assert "exceeds the partition" in plan.outcome.reason


def test_a_stray_whose_cancel_failed_is_cancelled_again_until_it_ended(tmp_path):
    orch, sched, clock = _two_jobs_one_tag(tmp_path)
    sched.cancel_fails = frozenset({"101"})
    plan, _, _ = orch.step()
    sub = plan.allocation(0).submissions[0]
    assert (sub.job_id, sub.strays) == ("100", ["101"])
    orch.step()
    assert sched.cancels() == ["101", "101"]
    sched.cancel_fails = frozenset()
    orch.step()
    assert sched.jobs["101"].state is JobState.CANCELLED and sched.jobs["100"].state is JobState.PENDING
    plan, _, _ = orch.step()
    assert plan.allocation(0).submissions[0].strays == [] and sched.cancels() == ["101"] * 3


def test_a_stray_is_not_leaked_by_a_crash_between_finding_it_and_cancelling_it(tmp_path):
    orch, sched, clock = _two_jobs_one_tag(tmp_path)
    sched.die_once = True
    with pytest.raises(SystemExit):
        orch.step()
    assert PlanStore(orch.store, orch.layout).latest().allocation(0).submissions[0].strays == ["101"]
    _attach(orch, sched, clock).step()
    assert sched.jobs["101"].state is JobState.CANCELLED and sched.jobs["100"].state is JobState.PENDING


def test_a_status_that_names_something_other_than_a_job_id_is_ignored(tmp_path):
    sched, clock = FakeScheduler(submit_fault="lose_answer"), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.tags_down = True
    _heartbeat(plan, 0, 1, clock.t, job_id="12; scancel --me")
    plan, _, _ = orch.step()
    assert plan.allocation(0).current().job_id is None
    ignored = [e for e in _events(orch) if e["kind"] == "status_ignored"]
    assert any("not a Slurm job id" in e["reason"] for e in ignored)


def test_a_void_instance_stays_finished_on_the_passes_after_its_jobs_ended(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, retry={"noop_a": {"enabled": False}})
    orch.step()
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    plan, done, _ = orch.step()
    assert done and plan.outcome.status == "void"
    asked = len([c for c in sched.calls if c[0] == "states"])
    for _ in range(3):
        plan, done, _ = orch.step()
        assert done
    assert len([c for c in sched.calls if c[0] == "states"]) == asked


def test_a_status_filed_under_another_submission_is_no_heartbeat(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock, gpu_policy="split")
    orch.step()
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    sched.start("101", at=clock.t)
    for _ in range(5):  # 25 s against a 10 s limit
        _advance(orch, clock, 5)
        _heartbeat(plan, 0, 1, clock.t, job_id="100")
        # Job 101's file holds a status of allocation 0 that names job 101.
        _heartbeat(plan, 0, 1, clock.t, job_id="101", filed_as=(1, 1))
    plan, _, _ = orch.step()
    assert plan.allocation(1).failure.fault.reason.startswith("job 101: no heartbeat for")
    assert any("allocation 0" in e["reason"] for e in _events(orch) if e["kind"] == "status_ignored")


def test_a_lost_lease_is_never_an_effect_failure(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)

    def lost(plan, effect):
        raise LockLost("generation 1 was taken over")

    orch.perform = lost
    with pytest.raises(LockLost):
        orch.step()
    assert "effect_failed" not in _kinds(orch)


def test_a_late_job_that_has_already_started_is_found_through_its_runners_status(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    hidden = {}
    sched.after_create = lambda job: hidden.update({job.job_id: sched.jobs.pop(job.job_id)})
    sched.submit_fault = "lose_answer"
    orch.step()
    sched.after_create = None
    orch.step()
    _advance(orch, clock, 121)
    plan, _, _ = orch.step()
    assert [s.status for s in plan.allocation(0).submissions] == ["absent", "submitted"]
    # The job turns up and starts while squeue cannot be asked; its runner writes its status.
    sched.jobs.update(hidden)
    sched.start("100", at=clock.t)
    sched.tags_down = True
    _heartbeat(plan, 0, 1, clock.t, job_id="100")
    plan, _, _ = orch.step()
    first = plan.allocation(0).submissions[0]
    assert (first.status, first.job_id) == ("cancelled", "100") and sched.cancels() == ["100"]


def _with_stray(sched):
    """The next submit creates the job and a second job under the same tag, then loses the answer."""
    def duplicate(job):
        sched.jobs["999"] = FakeJob("999", job.request, job.after)

    sched.after_create = duplicate
    sched.submit_fault = "lose_answer"


def test_a_stray_that_is_slow_to_end_is_cancelled_every_pass_without_spinning(tmp_path):
    class SlowEnd(FakeScheduler):
        """scancel is accepted for the stray, which then stays in the queue as COMPLETING does."""

        def cancel(self, job_id):
            self.calls.append(("cancel", job_id))
            if job_id != "999":
                job = self.jobs.get(job_id)
                if job is not None and not job.state.terminal:
                    job.state = JobState.CANCELLED

    sched, clock = SlowEnd(), Clock()
    orch = _create(tmp_path, sched, clock)
    _with_stray(sched)
    orch.step()
    sched.after_create = None
    sched.jobs["999"].state = JobState.RUNNING
    results = [orch.step() for _ in range(20)]
    assert results[0][2] and not any(progress for _, _, progress in results[1:])
    assert sched.cancels().count("999") == 20


def test_a_stray_the_scheduler_cannot_report_on_is_cancelled_every_pass_without_spinning(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    _with_stray(sched)
    orch.step()
    sched.after_create = None
    real_states = sched.states

    def states(ids):
        out = real_states(ids)
        if "999" in out:
            out["999"] = JobInfo("999", JobState.UNKNOWN)
        return out

    sched.states = states
    results = [orch.step() for _ in range(20)]
    assert results[0][2] and not any(progress for _, _, progress in results[1:])
    assert sched.cancels().count("999") == 20


def test_an_outage_of_the_tag_lookup_does_not_spend_the_absence_window(tmp_path):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    sched.after_create = lambda job: sched.jobs.pop(job.job_id)
    sched.submit_fault = "lose_answer"
    orch.step()
    sched.after_create = None
    orch.step()  # the first empty answer starts the window
    sched.tags_down = True
    _advance(orch, clock, 300)
    for _ in range(3):
        orch.step()
    sched.tags_down = False
    request_cancel(orch.store, orch.layout, at=clock.t)
    started = clock.t
    plan, done, _ = orch.step()
    sub = plan.allocation(0).submissions[0]
    assert (sub.status, sub.missing_since, done) == ("cancel_intended", started, False)
    _advance(orch, clock, 119)
    plan, done, _ = orch.step()
    assert plan.allocation(0).submissions[0].status == "cancel_intended" and not done
    _advance(orch, clock, 2)
    plan, done, _ = orch.step()
    assert plan.allocation(0).submissions[0].status == "cancelled" and not done
    _advance(orch, clock, 2 * 120)
    plan, done, _ = orch.step()
    assert done


# The order of observe's reads: the scheduler first, then the store. A runner may write its last
# status, publish its manifests and end between any two reads.


#: The fault the agent build declares once its commit point wrote the bundle.
COMMITTED = FaultRecord(fault_class="infra", reason="the build failed after its commit point", retryable=False,
                        evidence={"commit_point": "bundle"})


def _killed_after(orch: Orchestrator, sched: FakeScheduler, clock: Clock, monkeypatch, *, declare: bool) -> None:
    """Run job 100's runner in-process through stage noop_a, which declares no retry when
    *declare* says so and then fails, and leave the store as a runner SIGKILLed right after the
    declaration leaves it: what the runner wrote later is put back as it was then. The job then
    ends FAILED by signal 9."""
    from hls.pipeline.stage import STAGES
    from tests.pipeline.test_runner import _runner

    plan = PlanStore(orch.store, orch.layout).latest()
    sched.start("100", at=clock.t)
    store = PosixStore(plan.store_root)
    keys = [plan.layout.allocation_status(0, 1), plan.layout.attempt_file("noop_a", 1, "status.json")]
    at_kill: dict[str, bytes] = {}

    def run(self, ctx):
        if declare:
            ctx.declare_no_retry(COMMITTED)
        at_kill.update({key: store.read_bytes(key) for key in keys})
        raise StageFault(FaultRecord(fault_class="infra", reason="role transcript: killed", retryable=True))

    monkeypatch.setattr(STAGES["noop"], "run", run)
    _runner(plan).run()
    for key, data in at_kill.items():
        store.write_bytes(key, data)
    sched.finish("100", JobState.FAILED)
    sched.jobs["100"].exit_code = "0:9"


def test_a_runner_killed_after_its_stage_declared_no_retry_is_not_relaunched(tmp_path, monkeypatch):
    """The agent build's commit point declares that a retry would run the whole session again.
    The declaration is in the attempt's status the moment it is made and the runner's status names
    the attempt, so a runner killed outright after it (the job's OOM kill, a node failure), which
    writes no fault of its own, still ends the allocation with that fault: never retried."""
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    _killed_after(orch, sched, clock, monkeypatch, declare=True)
    plan, done, _ = orch.step()
    failure = plan.allocation(0).failure
    assert (failure.stage, failure.fault.reason, failure.fault.retryable) == ("noop_a", COMMITTED.reason, False)
    assert done and plan.outcome.status == "void" and len(plan.allocations) == 1
    assert sched.submits() == [("submit", f"hls:{plan.instance}/0/1", None)]


def test_a_runner_killed_before_any_declaration_is_relaunched(tmp_path, monkeypatch):
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    orch.step()
    _killed_after(orch, sched, clock, monkeypatch, declare=False)
    plan, _, _ = orch.step()
    assert plan.allocation(0).failure.fault.retryable is True
    _advance(orch, clock, 3600)
    plan, _, _ = orch.step()
    assert plan.outcome is None and plan.allocations[-1].retry_of == 0


def test_a_declaration_is_read_only_for_the_attempt_of_a_stage_that_has_not_published(tmp_path):
    """The allocation's status may still name an attempt that published, when its runner died
    before the next stage began: that attempt's declaration says nothing of the next stage. Nor is
    another job's attempt read as this runner's."""
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    store = PosixStore(plan.store_root)
    store.write_bytes(plan.layout.attempt_file("noop_a", 1, "status.json"), AttemptStatus(
        stage="noop_a", attempt=1, allocation=0, submission=1, job_id="100", state="succeeded", started_at=1.0,
        ended_at=2.0, fault=None, manifest_sha256=None, no_retry=COMMITTED).model_dump_json().encode())
    status = AllocationStatus(instance=plan.instance, allocation=0, submission=1, job_id="100", node="g",
                              state="running", started_at=1.0, heartbeat_at=1.0, end_time=None, stage="noop_a",
                              attempt=1, grant=None, tool_versions=None, swap_max_bytes=None,
                              memory_enforced_by=None, fault=None,
                              ended_at=None)
    store.write_bytes(plan.layout.allocation_status(0, 1), status.model_dump_json().encode())
    sched.start("100", at=clock.t)
    sched.finish("100", JobState.FAILED)
    assert orch.observe(plan).declared == {(0, 1): COMMITTED}
    store.write_bytes(plan.layout.allocation_status(0, 1),
                      status.model_copy(update={"stage": "noop_b"}).model_dump_json().encode())
    store.write_bytes(plan.layout.attempt_file("noop_b", 1, "status.json"), AttemptStatus(
        stage="noop_b", attempt=1, allocation=0, submission=1, job_id="999", state="running", started_at=1.0,
        ended_at=None, fault=None, manifest_sha256=None, no_retry=COMMITTED).model_dump_json().encode())
    assert orch.observe(plan).declared == {}
    store.write_bytes(plan.layout.allocation_status(0, 1), status.model_dump_json().encode())
    _publish(plan, "noop_a")
    assert orch.observe(plan).declared == {}


def test_runner_statuses_are_read_after_job_states(tmp_path):
    # A runner that writes its final status and ends between the reads must have its own fault
    # recorded. Read before the job's state, the status would still say running, and the job's
    # exit code would stand in for a fault that is not retryable.
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    _heartbeat(plan, 0, 1, at=clock.t)
    moved = FaultRecord(fault_class="infra", reason="the checkout changed since planning", retryable=False)
    real_states = sched.states

    def runner_ends_during_the_query(job_ids):
        store = PosixStore(plan.store_root)
        final = AllocationStatus(instance=plan.instance, allocation=0, submission=1, job_id="100", node="g",
                                 state="failed", started_at=clock.t, heartbeat_at=clock.t, end_time=None, stage=None,
                                 attempt=None, grant=None, tool_versions=None, swap_max_bytes=None,
                                 memory_enforced_by=None, fault=moved,
                                 ended_at=clock.t)
        store.write_bytes(plan.layout.allocation_status(0, 1), final.model_dump_json().encode())
        sched.finish("100", JobState.FAILED)
        return real_states(job_ids)

    sched.states = runner_ends_during_the_query
    plan, _, _ = orch.step()
    assert plan.allocation(0).failure.fault.reason == moved.reason
    assert plan.outcome.status == "void" and plan.outcome.reason.endswith("; not retryable")
    assert len(plan.allocations) == 1


def test_job_states_are_read_before_manifests(tmp_path):
    # A job seen ended has published every manifest it ever will. Read before the job's state,
    # the manifests could miss the last ones, and a job that succeeded would look failed.
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan, _, _ = orch.step()
    sched.start("100", at=clock.t)
    _heartbeat(plan, 0, 1, at=clock.t)
    real_states = sched.states

    def runner_publishes_and_ends_during_the_query(job_ids):
        if sched.jobs["100"].state is JobState.RUNNING:
            _publish(plan, "noop_a")
            _publish(plan, "noop_b")
            sched.finish("100")
        return real_states(job_ids)

    sched.states = runner_publishes_and_ends_during_the_query
    plan, done, _ = orch.step()
    assert plan.allocation(0).failure is None and len(plan.allocations) == 1
    assert done and plan.outcome.status == "succeeded"


def test_failed_passes_are_counted_in_a_row_only(tmp_path):
    # Nine failed passes, one good one, nine more: never ten in a row, so the orchestrator
    # carries on to the end.
    sched, clock = FakeScheduler(), Clock()
    orch = _create(tmp_path, sched, clock)
    plan = PlanStore(orch.store, orch.layout).latest()
    real_step = orch.step
    calls: list[int] = []

    def step():
        calls.append(1)
        n = len(calls)
        if n <= 9 or 11 <= n <= 19:
            raise OSError("blip")
        if n >= 21:
            _publish(plan, "noop_a")
            _publish(plan, "noop_b")
        return real_step()

    def sleep(sec: float) -> None:
        clock.t += sec

    orch.step, orch.sleep = step, sleep
    sched.after_create = lambda job: sched.start(job.job_id, at=clock.t)
    assert orch.run().outcome.status == "succeeded"
    assert sum(e["kind"] == "pass_failed" for e in _events(orch)) == 18
