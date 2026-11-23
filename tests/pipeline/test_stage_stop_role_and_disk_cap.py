from __future__ import annotations

import resource
import sys
import threading
import time
from pathlib import Path

import pytest

import hls.pipeline.stage as stage_mod
from hls.pipeline.events import read_events
from hls.pipeline.stage import StageFault, stage_fault
from hls.spec.resources import ResourceShape
from hls.substrate.base import JobState, RoleSpec
from hls.substrate.container import ContainerSpec, Mount
from hls.substrate.faults import OOM_EVIDENCE_MAX_AGE_SEC, classify_role
from hls.substrate import memory_watch
from hls.substrate.memory_watch import DISK_WALK_MAX_DEPTH, DiskSample, MemorySample, MemoryWatch, PidsHits
from hls.substrate.nproc import FLOOR_PER_EVALUATION, ProcessCeiling
from hls.substrate.slurm.launcher import srun_argv
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.test_stage_context import SHAPE, _ctx, _entry, _pinned, _python
from tests.substrate import fake_apptainer
from tests.substrate.nproc_pool import UidPool
from tests.substrate.sandboxes import IMAGE_SHA256, image_file

SLEEP = _python("import time; time.sleep(30)")
#: A small role beside one that holds every CPU of the stage.
ONE_CPU = ResourceShape(cpus=1, memory_gib=2, time_sec=60)


@pytest.fixture
def apptainer(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    fake_apptainer.install(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    monkeypatch.delenv("FAKE_APPTAINER_FAIL", raising=False)


def test_a_role_the_stage_stops_as_planned_ends_with_no_fault(tmp_path):
    launcher = LocalLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    ctx.start_role("srv", SLEEP, SHAPE, owner="harness")
    outcome = ctx.stop_role("srv")
    assert (outcome.ending, outcome.exit_code) == ("planned_stop", -15)
    assert classify_role(outcome) is None and stage_fault([outcome]) is None
    assert launcher.terminations == [["srv"]] and ctx.running_roles() == []
    assert ctx.close() == []


def test_a_role_that_outlives_its_grace_is_killed_and_still_stopped_as_planned(tmp_path, monkeypatch):
    monkeypatch.setattr(stage_mod, "STOP_GRACE_SEC", 0.5)
    launcher = LocalLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    ctx.start_role("srv", _python("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                                  "time.sleep(30)"), SHAPE, owner="harness")
    time.sleep(0.5)
    outcome = ctx.stop_role("srv")
    assert (outcome.ending, outcome.exit_code) == ("planned_stop", -9)
    assert ("kill", "srv") in launcher.calls


def test_a_role_that_ended_before_the_stop_keeps_its_own_ending(tmp_path):
    _, ctx = _ctx(tmp_path, LocalLauncher())
    handle = ctx.start_role("srv", _python("import sys; sys.exit(3)"), SHAPE, owner="harness")
    handle.wait(10)
    outcome = ctx.stop_role("srv")
    assert (outcome.ending, outcome.exit_code) == ("exited", 3)
    assert classify_role(outcome).fault_class == "infra"


def test_a_stop_during_the_allocations_shutdown_is_the_shutdowns(tmp_path):
    _, ctx = _ctx(tmp_path, LocalLauncher())
    ctx.start_role("srv", SLEEP, SHAPE, owner="harness")
    ctx.stop_roles()
    assert ctx.stop_role("srv").ending == "stopped"


def test_only_a_started_role_not_yet_reported_can_be_stopped(tmp_path):
    _, ctx = _ctx(tmp_path, LocalLauncher())
    with pytest.raises(ValueError, match="no started role left to stop named srv"):
        ctx.stop_role("srv")
    ctx.start_role("srv", _python("pass"), SHAPE, owner="harness")
    ctx.wait_roles()
    with pytest.raises(ValueError, match="no started role left to stop named srv"):
        ctx.stop_role("srv")


def test_whether_a_roles_command_started_is_read_from_its_start_marker(tmp_path, apptainer, monkeypatch):
    _, ctx = _ctx(tmp_path, LocalLauncher())
    with pytest.raises(ValueError, match="no started role named box"):
        ctx.command_started("box")
    image = _pinned(ctx)
    ctx.start_role("host", _python("pass"), SHAPE, owner="harness")
    ctx.wait_roles()
    ctx.start_container("box", ContainerSpec(image=image, command=("true",), workdir=ctx.role_dir("box") / "wd"),
                        SHAPE, owner="harness")
    ctx.wait_roles()
    # A runtime that fails before the container starts prints no marker.
    monkeypatch.setenv("FAKE_APPTAINER_FAIL", "1")
    ctx.start_container("stuck", ContainerSpec(image=image, command=("true",), workdir=ctx.role_dir("stuck") / "wd"),
                        SHAPE, owner="harness")
    ctx.wait_roles()
    # A role on the host has no marker to print, so its command started when the role did.
    assert [ctx.command_started(name) for name in ("host", "box", "stuck")] == [True, True, False]


class _OomLauncher(LocalLauncher):
    """Each step reads as OUT_OF_MEMORY when it ends, while its command runs as given."""

    def launch(self, spec, *, share_gpus_of=None):
        handle = super().launch(spec, share_gpus_of=share_gpus_of)
        handle.step_state = JobState.OUT_OF_MEMORY
        return handle


def test_an_oom_kill_before_a_planned_stop_is_kept(tmp_path):
    cgroups = tmp_path / "cg"
    _, ctx = _ctx(tmp_path, _OomLauncher(cgroup_root=cgroups))
    ctx.start_role("srv", SLEEP, SHAPE, owner="harness")
    (cgroups / "cg-srv" / "memory.events").write_text("oom 1\noom_kill 1\n")
    outcome = ctx.stop_role("srv")
    assert (outcome.ending, outcome.oom_kills) == ("planned_stop", 1) and "before its stop" in outcome.detail
    assert classify_role(outcome).reason == "harness role srv ran out of memory"


@pytest.mark.parametrize("owner", ["harness", "contestant"])
def test_an_oom_kill_after_the_stops_sigterm_is_part_of_the_stop(tmp_path, apptainer, owner):
    # Once asked to stop, whatever ends the role is the stop: a kernel OOM kill in its grace
    # period, which accounting then records for the step, counts for nothing.
    cgroups = tmp_path / "cg"
    _, ctx = _ctx(tmp_path, _OomLauncher(cgroup_root=cgroups))
    if owner == "harness":
        ctx.start_role("srv", SLEEP, SHAPE, owner="harness")
    else:
        spec = ContainerSpec(image=_pinned(ctx), workdir=ctx.role_dir("srv") / "wd",
                             command=(sys.executable, "-c", "import time; time.sleep(30)"))
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
        deadline = time.monotonic() + 10
        while not ctx.command_started("srv") and time.monotonic() < deadline:
            time.sleep(0.05)
    real = ctx.launcher.terminate

    def terminate(handles):
        (cgroups / "cg-srv" / "memory.events").write_text("oom 1\noom_kill 2\n")
        real(handles)

    ctx.launcher.terminate = terminate
    outcome = ctx.stop_role("srv")
    assert (outcome.ending, outcome.oom_kills) == ("planned_stop", 0) and classify_role(outcome) is None


class _DiskWatch:
    """A memory watch that records what it is asked to walk, and says the steps of the roles in
    *breached* passed their caps."""

    #: No job cgroup is watched.
    job_dir = None

    def __init__(self, breached: tuple[str, ...] = ()) -> None:
        self.breached = breached
        self.disks: list[tuple[str, list[Path], int, str]] = []
        self.unwatched: list[str] = []
        self.roles: dict[str, str] = {}

    def watch(self, step, cgroup_dir, limit_bytes, kill, *, role) -> None:
        self.roles[step] = role

    def last(self, step) -> None:
        return None

    def breach(self, step) -> None:
        return None

    def job_ooms(self) -> None:
        return None

    def watch_disk(self, step, paths, limit_bytes, kill, *, role, cgroup_dir=None, ceiling=None,
                   harness_pids=None) -> None:
        self.disks.append((step, list(paths), limit_bytes, role))
        self.harness_pids = harness_pids

    def unwatch_disk(self, step) -> None:
        self.unwatched.append(step)

    def disk_breach(self, step) -> DiskSample | None:
        return DiskSample(at=1.0, used=5 * 2**30) if self.roles.get(step) in self.breached else None

    def pids_hits(self, step) -> None:
        return None

    def pool(self, step) -> None:
        return None

    def pool_peak(self, step) -> None:
        return None

    def disk_at_end(self, step) -> None:
        return None


class _TooDeepWatch(_DiskWatch):
    """A watch whose walk of the breached roles' directories stopped at a tree too deep."""

    def disk_breach(self, step) -> DiskSample | None:
        return DiskSample(at=1.0, used=2**20, unmeasured="depth") if self.roles.get(step) in self.breached else None


def test_a_contestant_containers_writable_paths_are_walked_against_its_declared_disk(tmp_path, apptainer):
    watch = _DiskWatch()
    _, ctx = _ctx(tmp_path, LocalLauncher(), watch=watch)
    weights = _entry(ctx)
    own = ctx.role_dir("srv")
    (own / "out").mkdir()
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=own / "wd",
                         mounts=(Mount(own / "out", "/out", read_only=False), Mount(weights, "/data/weights")))
    ctx.start_container("srv", spec, SHAPE.model_copy(update={"disk_gib": 3}), owner="contestant")
    ctx.wait_roles()
    ctx.start_container("tool", ContainerSpec(image=_pinned(ctx), command=("true",),
                                              workdir=ctx.role_dir("tool") / "wd"), SHAPE, owner="harness")
    ctx.wait_roles()
    # Only the contestant's container is walked, and its walk ends once it is reported.
    [(step, paths, limit, role)] = watch.disks
    assert (paths, limit, role) == ([own / "wd", own / "out"], 3 * 2**30, "srv")
    assert step == ctx.report("srv").step
    assert watch.unwatched == [ctx.report("srv").step]


def test_a_contestant_container_that_declares_no_disk_is_refused(tmp_path, apptainer):
    launcher = LocalLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("srv") / "wd")
    launched = len(launcher.launched)
    with pytest.raises(ValueError, match="declares no disk"):
        ctx.start_container("srv", spec, SHAPE.model_copy(update={"disk_gib": 0}), owner="contestant")
    assert len(launcher.launched) == launched and not (ctx.role_dir("srv") / "container.env").exists()


def test_a_contestant_container_killed_at_its_disk_cap_is_the_contestants_fault(tmp_path, apptainer):
    _, ctx = _ctx(tmp_path, LocalLauncher(), watch=_DiskWatch(breached=("srv",)))
    # The watch's kill ends the command after it started; here it ends on its own.
    spec = ContainerSpec(image=_pinned(ctx), command=("false",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "disk_limit"
    assert outcome.detail == f"{5 * 2**30} bytes in its writable directories over 1 GiB"
    fault = stage_fault([outcome])
    assert (fault.fault_class, fault.retryable) == ("contestant", False)


def test_a_tree_too_deep_to_walk_is_past_the_cap_and_the_fault_names_the_depth(tmp_path, apptainer):
    # A writable tree deeper than the walk's bound counts as past the disk cap,
    # and the fault's reason, which the result records, says so with the bound.
    _, ctx = _ctx(tmp_path, LocalLauncher(), watch=_TooDeepWatch(breached=("srv",)))
    spec = ContainerSpec(image=_pinned(ctx), command=("false",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "disk_limit" and f"{DISK_WALK_MAX_DEPTH} levels deep" in outcome.detail
    fault = stage_fault([outcome])
    assert (fault.fault_class, fault.retryable) == ("contestant", False)
    assert "disk cap" in fault.reason and f"{DISK_WALK_MAX_DEPTH} levels deep" in fault.reason


def test_a_disk_breach_before_a_planned_stop_still_counts(tmp_path, apptainer):
    _, ctx = _ctx(tmp_path, LocalLauncher(), watch=_DiskWatch(breached=("srv",)))
    spec = ContainerSpec(image=_pinned(ctx), command=("sleep", "30"), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    time.sleep(0.5)
    assert ctx.stop_role("srv").ending == "disk_limit"


def test_an_overlap_step_takes_no_cpus_of_its_own_and_its_memory_counts(tmp_path):
    """With a role holding every CPU of the stage, roles that overlap it start, one that would
    take a CPU of its own does not, and an overlap role's memory still counts."""
    _, ctx = _ctx(tmp_path, LocalLauncher())
    stage = ResourceShape(cpus=2, memory_gib=12, time_sec=60)
    ctx.stage = ctx.stage.model_copy(update={"shape": stage})
    ctx.start_role("all-cpus", SLEEP, ResourceShape(cpus=2, memory_gib=2, time_sec=60), owner="harness")
    ctx.start_role("overlap-1", SLEEP, ONE_CPU, owner="harness", overlap=True)
    ctx.start_role("overlap-2", SLEEP, ONE_CPU, owner="harness", overlap=True)
    with pytest.raises(StageFault, match=r"the roles need 3 CPUs; the stage declares 2$"):
        ctx.start_role("own-cpu", SLEEP, ONE_CPU, owner="harness")
    big = ONE_CPU.model_copy(update={"memory_gib": stage.memory_gib - 4 - 6 + 1})
    with pytest.raises(StageFault, match=r"^role too-much-memory: the roles need 9 GiB and the runner"):
        ctx.start_role("too-much-memory", SLEEP, big, owner="harness", overlap=True)
    assert [s.overlap for s in ctx.launcher.launched] == [False, True, True]
    ctx.close()


def test_an_overlap_container_overlaps_and_an_overlap_with_gpus_is_refused(tmp_path, apptainer):
    launcher = LocalLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    ctx.start_container("box", ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("box") / "wd"),
                        SHAPE, owner="harness", overlap=True)
    ctx.wait_roles()
    assert launcher.launched[-1].overlap
    # srun's --overlap shares GPUs as it shares CPUs, so an overlap role's GPUs would be another's.
    gpu = ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=1, time_sec=60)
    launched = len(launcher.launched)
    with pytest.raises(ValueError, match="overlap"):
        ctx.start_role("gpu", SLEEP, gpu, owner="harness", overlap=True)
    with pytest.raises(ValueError, match="overlap"):
        ctx.start_container("gpu2", ContainerSpec(image=_pinned(ctx), command=("true",),
                                                  workdir=ctx.role_dir("gpu2") / "wd"), gpu, owner="harness",
                            overlap=True)
    assert len(launcher.launched) == launched
    with pytest.raises(ValueError, match="overlap"):
        RoleSpec(name="gpu", argv=("true",), shape=gpu, owner="harness", log_dir=tmp_path, overlap=True)


def test_an_overlap_steps_srun_line_overlaps_and_takes_no_gpu(tmp_path):
    spec = RoleSpec(name="snapshot-1", argv=("true",), shape=ONE_CPU, owner="harness", log_dir=tmp_path, overlap=True)
    argv = srun_argv(spec, python="py", report=tmp_path / "r.json", scratch=tmp_path, shared=None, allocation_gpus=1)
    assert "--overlap" in argv and "--exact" not in argv and "--gres=none" in argv and "--cpus-per-task=1" in argv
    own = srun_argv(RoleSpec(name="sandbox", argv=("true",), shape=SHAPE, owner="contestant", log_dir=tmp_path),
                    python="py", report=tmp_path / "r.json", scratch=tmp_path, shared=None, allocation_gpus=1)
    assert "--exact" in own and "--overlap" not in own


def test_a_harness_container_on_the_submissions_image_is_refused(tmp_path, apptainer):
    # An image holding a contestant's code runs only the contestant's own containers.
    launcher = LocalLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    submission = ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256, contestant=True)
    assert submission.contestant
    launched = len(launcher.launched)
    tool = ContainerSpec(image=submission, command=("true",), workdir=ctx.role_dir("tool") / "wd")
    with pytest.raises(ValueError, match="holds a contestant's code"):
        ctx.start_container("tool", tool, SHAPE, owner="harness")
    assert len(launcher.launched) == launched
    srv = ContainerSpec(image=submission, command=("true",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", srv, SHAPE, owner="contestant")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("exited", 0)


def test_an_image_pinned_as_the_contestants_stays_the_contestants(tmp_path, apptainer):
    # The same bytes pinned for the harness first, then as a contestant's, are a contestant's
    # image from then on; the harness's earlier pin does not make it the harness's.
    _, ctx = _ctx(tmp_path, LocalLauncher())
    harness = ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256, contestant=False)
    assert not harness.contestant
    assert ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256, contestant=True).contestant
    assert ctx.pin_file(image_file(ctx.scratch), sha256=IMAGE_SHA256, contestant=False).contestant
    with pytest.raises(ValueError, match="not one this stage pinned"):
        ctx.start_container("tool", ContainerSpec(image=harness, command=("true",), workdir=ctx.role_dir("tool") / "wd"),
                            SHAPE, owner="harness")


def test_a_breach_after_the_planned_stops_sigterm_does_not_count(tmp_path, apptainer, monkeypatch):
    # What the role does once the stage has asked it to stop is not what it did while it served:
    # its disk is walked no more from the stop's SIGTERM on.
    monkeypatch.setattr(stage_mod, "_GIB", 2**20)  # a declared GiB counted as a MiB
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.05)
    watch.start()
    try:
        _, ctx = _ctx(tmp_path, LocalLauncher(), watch=watch)
        code = ("import os, signal, time\n"
                "def fill(signum, frame):\n"
                "    with open(os.path.join(os.environ['HOME'], 'fill'), 'wb') as f:\n"
                "        f.write(b'x' * 4 * 2**20)\n"
                "        f.flush()\n"
                "        os.fsync(f.fileno())\n"
                "    time.sleep(1)\n"
                "    os._exit(0)\n"
                "signal.signal(signal.SIGTERM, fill)\n"
                "time.sleep(30)\n")
        spec = ContainerSpec(image=_pinned(ctx), command=(sys.executable, "-c", code),
                             workdir=ctx.role_dir("srv") / "wd")
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
        time.sleep(0.5)
        assert ctx.stop_role("srv").ending == "planned_stop"
    finally:
        watch.stop()


def _host_ceiling(wanted: int) -> int:
    """*wanted*, or this host's hard process limit when that is lower: a ceiling a child may set."""
    hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
    return wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)


def _stage_ceiling(monkeypatch, ceiling: int) -> None:
    """Give the stage's contestant launches the process ceiling *ceiling*
    (``StageContext.process_ceiling``), for a test whose pool is counted against a ceiling below
    the floor of the evaluations its node holds."""
    made = ProcessCeiling(hard=None, reserve=0, evaluations=1, evaluations_unknown=None, uncapped=None,
                          nproc_max=ceiling, ceiling=ceiling, floor=FLOOR_PER_EVALUATION)
    monkeypatch.setattr(stage_mod, "ceiling_for", lambda *args, **kwargs: made)


def test_a_contestant_container_runs_under_the_stages_process_ceiling(tmp_path, apptainer, monkeypatch):
    ceiling = _host_ceiling(400_000)
    _stage_ceiling(monkeypatch, ceiling)
    _, ctx = _ctx(tmp_path, LocalLauncher(apply_nproc=True))
    code = "import resource, sys; open(sys.argv[1], 'w').write(repr(resource.getrlimit(resource.RLIMIT_NPROC)))"
    own = ctx.role_dir("srv")
    (own / "out").mkdir()
    spec = ContainerSpec(image=_pinned(ctx), command=(sys.executable, "-c", code, str(own / "out" / "limits")),
                         workdir=own / "wd", mounts=(Mount(own / "out", "/out", read_only=False),))
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    ctx.wait_roles()
    assert (own / "out" / "limits").read_text() == f"({ceiling}, {ceiling})"
    assert ctx.launcher.launched[-1].nproc == ceiling


def _pids_hit(cgroups: Path, role: str) -> None:
    """Record in role *role*'s fake step cgroup that its pids limit was reached twice."""
    (cgroups / f"cg-{role}" / "pids.events").write_text("max 2\n")


@pytest.mark.parametrize("knobs", [{}, {"scheduler_ended": ("srv",)}], ids=["exited", "scheduler-ended"])
def test_a_contestant_container_that_reached_its_pids_limit_is_its_contestants_end(tmp_path, apptainer, knobs):
    # The count is read before any reading that would make the end infra.
    cgroups = tmp_path / "cg"
    _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=cgroups, **knobs), watch=_DiskWatch())
    spec = ContainerSpec(image=_pinned(ctx), command=(sys.executable, "-c", "import time; time.sleep(1)"),
                         workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    _pids_hit(cgroups, "srv")
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "contestant_pids" and "2 times" in outcome.detail
    fault = stage_fault([outcome])
    assert (fault.fault_class, fault.retryable) == ("contestant", False)


def test_a_pids_count_a_walk_read_counts_after_the_steps_cgroup_is_gone(tmp_path, apptainer):
    # A step's cgroup goes with the step, so its count is read while the step runs, by the walks.
    cgroups = tmp_path / "cg"
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.05)
    watch.start()
    try:
        _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=cgroups), watch=watch)
        spec = ContainerSpec(image=_pinned(ctx), command=(sys.executable, "-c", "import time; time.sleep(2)"),
                             workdir=ctx.role_dir("srv") / "wd")
        handle = ctx.start_container("srv", spec, SHAPE, owner="contestant")
        _pids_hit(cgroups, "srv")
        step = ctx.launcher.handles[-1].report(1.0).step
        deadline = time.monotonic() + 10
        while (watch.pids_hits(step) or PidsHits(count=0, rose_at=None)).count < 2 and time.monotonic() < deadline:
            time.sleep(0.05)
        (cgroups / "cg-srv" / "pids.events").unlink()
        handle.wait(10)
        [outcome] = ctx.wait_roles()
    finally:
        watch.stop()
    assert outcome.ending == "contestant_pids" and "2 times" in outcome.detail


class _PidsWatch(_DiskWatch):
    """A watch whose walks read every step's pids limit count as 2, first at ``rose_at``, and
    which takes a step watched with ``end_one`` (the build sandbox) as the real one does."""

    def __init__(self) -> None:
        super().__init__()
        #: When the walks first read the count, by the watch's clock.
        self.rose_at = 0.0

    def watch(self, step, cgroup_dir, limit_bytes, kill, *, role, end_one=None, exiting=None) -> None:
        self.roles[step] = role

    def kills(self, step) -> list:
        return []

    def kill_reason(self, step) -> None:
        return None

    def pids_hits(self, step) -> PidsHits:
        return PidsHits(count=2, rose_at=self.rose_at)


@pytest.mark.parametrize("age, ending", [(0.0, "contestant_pids"), (OOM_EVIDENCE_MAX_AGE_SEC - 1, "contestant_pids"),
                                         (OOM_EVIDENCE_MAX_AGE_SEC + 5, "planned_stop")],
                         ids=["at-the-stop", "within-the-age", "stale"])
def test_the_build_sandboxs_pids_limit_ends_it_only_when_reached_within_the_evidence_age(tmp_path, apptainer, age,
                                                                                         ending):
    """The build rule reads a pids limit reached as it reads an OOM, within
    ``OOM_EVIDENCE_MAX_AGE_SEC`` of the end it would explain: a sandbox the stage stops as planned
    long after a fork bomb it survived ended cleanly."""
    watch = _PidsWatch()
    _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
    spec = ContainerSpec(image=_pinned(ctx), command=(sys.executable, "-c", "import time; time.sleep(30)"),
                         workdir=ctx.role_dir("sandbox") / "wd")
    ctx.start_container("sandbox", spec, SHAPE, owner="contestant", harness_reserve_gib=1)
    _started(ctx, "sandbox")
    watch.rose_at = ctx.clock() - age
    outcome = ctx.stop_role("sandbox")
    assert outcome.ending == ending
    assert (stage_fault([outcome]) is None) is (ending == "planned_stop")


def _started(ctx, name: str) -> None:
    """Wait until role *name*'s command has started, so a stop finds it running."""
    deadline = time.monotonic() + 10
    while not ctx.command_started(name) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ctx.command_started(name)


@pytest.mark.parametrize("age, ending", [(OOM_EVIDENCE_MAX_AGE_SEC - 1, "contestant_pids"),
                                         (10 * OOM_EVIDENCE_MAX_AGE_SEC, "planned_stop")], ids=["within-the-age", "stale"])
def test_any_other_contestant_containers_pids_limit_ends_it_only_within_the_evidence_age(tmp_path, apptainer, age,
                                                                                       ending):
    """Evidence has one age bound everywhere: a contestant container without a harness reserve,
    such as the drive's graded server, ends ``contestant_pids`` only for a pids limit reached
    within ``OOM_EVIDENCE_MAX_AGE_SEC`` of its end or its stop, as the build sandbox does."""
    watch = _PidsWatch()
    _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
    spec = ContainerSpec(image=_pinned(ctx), command=(sys.executable, "-c", "import time; time.sleep(30)"),
                         workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    _started(ctx, "srv")
    watch.rose_at = ctx.clock() - age
    assert ctx.stop_role("srv").ending == ending


def test_the_stage_reads_a_containers_pids_count_with_the_time_it_rose(tmp_path, apptainer):
    """``StageContext.pids_hits``: the walks' count with the time the watch dated its rise, or the
    step's cgroup's count now, dated now, when that is higher."""
    watch = _PidsWatch()
    cgroups = tmp_path / "cg"
    _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=cgroups), watch=watch)
    spec = ContainerSpec(image=_pinned(ctx), command=(sys.executable, "-c", "import time; time.sleep(30)"),
                         workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    watch.rose_at = 5.0
    assert ctx.pids_hits("srv") == PidsHits(count=2, rose_at=5.0)
    (cgroups / "cg-srv" / "pids.events").write_text("max 3\n")
    before = ctx.clock()
    hits = ctx.pids_hits("srv")
    assert hits.count == 3 and before <= hits.rose_at <= ctx.clock()
    ctx.stop_role("srv")
    ctx.start_container("tool", ContainerSpec(image=_pinned(ctx), command=("sleep", "30"),
                                              workdir=ctx.role_dir("tool") / "wd"), SHAPE, owner="harness")
    assert ctx.pids_hits("tool") is None
    with pytest.raises(ValueError, match="no started role named nobody"):
        ctx.pids_hits("nobody")
    ctx.stop_role("tool")


def test_a_harness_roles_pids_count_is_not_read(tmp_path, apptainer):
    cgroups = tmp_path / "cg"
    _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=cgroups), watch=_DiskWatch())
    ctx.start_container("tool", ContainerSpec(image=_pinned(ctx), command=("sleep", "1"),
                                              workdir=ctx.role_dir("tool") / "wd"), SHAPE, owner="harness")
    _pids_hit(cgroups, "tool")
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("exited", 0)


def test_a_containers_env_file_can_be_written_before_the_contestant_starts(tmp_path, apptainer):
    # A harness container started once a contestant runs, which may fill the node's disk, needs no
    # new file there: its env file was written before.
    launcher = LocalLauncher()
    _, ctx = _ctx(tmp_path, launcher)
    env = {"PLAIN": "1"}
    ctx.prepare_container_env("tool", env)
    written = ctx.role_dir("tool") / "container.env"
    assert written.read_text() == "PLAIN=1\n" and written.stat().st_mode & 0o777 == 0o600
    inode = written.stat().st_ino
    seen: list[int] = []
    real = launcher.launch

    def launch(spec, **kw):
        seen.append(spec.container.env_file.stat().st_ino)
        return real(spec, **kw)

    launcher.launch = launch
    tool = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("tool") / "wd")
    ctx.start_container("tool", tool, SHAPE, owner="harness", env=env)
    ctx.wait_roles()
    assert seen == [inode]
    # The container gets what was prepared or nothing: another env is refused before any step.
    ctx.prepare_container_env("other", env)
    other = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("other") / "wd")
    with pytest.raises(ValueError, match="prepared"):
        ctx.start_container("other", other, SHAPE, owner="harness", env={"PLAIN": "2"})
    assert seen == [inode]


def test_a_prepared_env_file_holds_no_secret(tmp_path):
    _, ctx = _ctx(tmp_path, LocalLauncher(), secrets={"HF_TOKEN": "hf-token-0123456789"})
    with pytest.raises(ValueError, match="holds a secret"):
        ctx.prepare_container_env("tool", {"X": "hf-token-0123456789"})
    assert not (ctx.role_dir("tool") / "container.env").exists()


def test_a_role_directory_holds_the_tmp_directory_its_step_uses(tmp_path):
    _, ctx = _ctx(tmp_path, LocalLauncher())
    assert (ctx.role_dir("srv") / "tmp").is_dir()


def _pool_watch(monkeypatch, tasks: int) -> tuple[MemoryWatch, UidPool]:
    """A started watch that walks often and counts the uid's tasks in a pool of *tasks*."""
    pool = UidPool(tasks=tasks)
    monkeypatch.setattr(memory_watch, "uid_tasks", lambda uid: pool.tasks)
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.05)
    watch.start()
    return watch, pool


class _Node:
    """The uid's tasks on a node, as the kernel counts them: the other jobs' (*others*), and while
    the server's step runs, its own (*own*), unreaped zombies among them, which no cgroup's thread
    list shows. Its walks and the stage's count at the report read the same pool."""

    def __init__(self, monkeypatch, *, others: int, own: int) -> None:
        #: The other jobs' tasks.
        self.others = others
        #: The server's tasks while its step runs.
        self.own = own
        #: The server's step, once started.
        self.server = None
        #: Each count the walks read.
        self.walked: list[int] = []
        monkeypatch.setattr(memory_watch, "uid_tasks", self.count)
        monkeypatch.setattr(stage_mod, "uid_tasks", self.count)
        _stage_ceiling(monkeypatch, 10_000)

    def count(self, uid: int) -> int:
        alive = self.server is not None and self.server.poll() is None
        tasks = self.others + (self.own if alive else 0)
        if threading.current_thread().name == "hls-disk-watch":
            self.walked.append(tasks)
        return tasks


def _server(ctx, node: _Node, code: str) -> None:
    spec = ContainerSpec(image=_pinned(ctx), workdir=ctx.role_dir("srv") / "wd",
                         command=(sys.executable, "-c", code))
    node.server = ctx.start_container("srv", spec, SHAPE, owner="contestant")


def test_a_server_whose_zombies_fill_its_ceiling_ends_as_its_own_result(tmp_path, apptainer, monkeypatch):
    # A thousand unreaped children leave the pool full at every walk, while its step's thread list
    # shows none of them. Once its step is gone they are gone too, and the others left room.
    node = _Node(monkeypatch, others=9_000, own=1_000)
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.05)
    watch.start()
    try:
        _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
        _server(ctx, node, "import sys, time; time.sleep(1); sys.exit(3)")
        [outcome] = ctx.wait_roles()
    finally:
        watch.stop()
    # The walks saw the pool full while it ran, and its step's thread list empty.
    assert 10_000 in node.walked and outcome.pool.step_tasks is None
    assert (outcome.pool_at_end.others_tasks, outcome.pool_at_end.rule.ceiling) == (9_000, 10_000)
    assert classify_role(outcome) is None and stage_fault([outcome]) is None
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert (ended["others_tasks"], ended["ceiling"]) == (9_000, 10_000) and "walk_uid_tasks" in ended


def test_a_server_starved_by_a_storm_between_walks_ends_as_infra(tmp_path, apptainer, monkeypatch):
    # Another job fills the pool after the last walk and the server's next fork fails: no walk saw
    # it, and the count when the server is reported does.
    node = _Node(monkeypatch, others=2_000, own=40)
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=60.0)
    watch.start()
    try:
        _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
        _server(ctx, node, "import sys, time; time.sleep(1); sys.exit(3)")
        time.sleep(0.3)
        node.others = 9_700
        [outcome] = ctx.wait_roles()
    finally:
        watch.stop()
    assert outcome.pool is None and outcome.pool_at_end.others_tasks == 9_700
    fault = stage_fault([outcome])
    assert (fault.fault_class, fault.retryable) == ("infra", True) and fault.evidence["srv.others_tasks"] == "9700"


@pytest.mark.parametrize("free_gib, starved", [(0.01, True), (64.0, False)], ids=["disk-full", "room-left"])
def test_a_server_that_failed_while_others_held_the_nodes_disk_ends_as_infra(tmp_path, apptainer, monkeypatch,
                                                                              free_gib, starved):
    """A contestant server writes a MiB and fails. When it is reported, its step gone, the watch reads
    what its directories hold and its filesystem's free space: with others holding nearly all its
    cap still left it, its failure may be an ENOSPC they caused, and is infra, retried; with room
    left it is its own result. The role's end records both."""
    monkeypatch.setattr(memory_watch, "free_bytes", lambda paths: int(free_gib * 2**30))
    watch = MemoryWatch(enforce=False, interval_sec=0.05, disk_interval_sec=0.05)
    watch.start()
    try:
        _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
        spec = ContainerSpec(image=_pinned(ctx), workdir=ctx.role_dir("srv") / "wd",
                             command=(sys.executable, "-c", "import os, sys; os.makedirs(os.environ['HOME'], exist_ok=True); "
                                                            "open(os.environ['HOME'] + '/f', 'wb').write(b'x' * 2**20); "
                                                            "sys.exit(28)"))
        ctx.start_container("srv", spec, SHAPE.model_copy(update={"disk_gib": 2}), owner="contestant")
        [outcome] = ctx.wait_roles()
    finally:
        watch.stop()
    disk = outcome.disk_at_end
    assert disk is not None and disk.limit_bytes == 2 * 2**30 and disk.own_bytes >= 2**20
    assert disk.free_bytes == int(free_gib * 2**30)
    fault = stage_fault([outcome])
    if starved:
        assert (fault.fault_class, fault.retryable) == ("infra", True) and "held its disk" in fault.reason
    else:
        assert fault is None
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert ended["disk_at_end"]["free_bytes"] == int(free_gib * 2**30)


def test_a_contestant_steps_role_ended_records_the_walks_peak_beside_their_last_sample(tmp_path, apptainer,
                                                                                         monkeypatch):
    # A storm that came and went while the server ran: the last walk shows it gone, the peak
    # keeps it, for the reserve and the floor to be measured from.
    watch, pool = _pool_watch(monkeypatch, 300)
    _stage_ceiling(monkeypatch, 10_000)
    try:
        _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=tmp_path / "cg"), watch=watch)
        spec = ContainerSpec(image=_pinned(ctx), workdir=ctx.role_dir("srv") / "wd",
                             command=(sys.executable, "-c", "import time; time.sleep(2)"))
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
        time.sleep(0.4)
        pool.tasks = 6_000
        time.sleep(0.4)
        pool.tasks = 400
        [outcome] = ctx.wait_roles()
    finally:
        watch.stop()
    [ended] = [e for e in read_events(ctx.attempt_dir / "events.jsonl") if e["kind"] == "role_ended"]
    assert ended["walk_uid_tasks"] == outcome.pool.uid_tasks == 400
    peak = ended["pool_peak"]
    assert peak["uid_tasks"] == 6_000 and peak["walks"] >= 3
    assert set(peak) == {"walks", "uid_tasks", "step_tasks", "others_tasks"}


def test_the_pool_is_sampled_no_more_once_the_planned_stop_begins(tmp_path, apptainer, monkeypatch):
    cgroups = tmp_path / "cg"
    watch, pool = _pool_watch(monkeypatch, 300)
    _stage_ceiling(monkeypatch, 10_000)
    try:
        _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=cgroups), watch=watch)
        spec = ContainerSpec(image=_pinned(ctx), workdir=ctx.role_dir("srv") / "wd",
                             command=(sys.executable, "-c", "import time; time.sleep(30)"))
        ctx.start_container("srv", spec, SHAPE, owner="contestant")
        time.sleep(0.5)
        real = ctx.launcher.terminate

        def terminate(handles):
            pool.tasks = 9_900  # what a shutdown that forks to the ceiling would show
            time.sleep(0.3)
            real(handles)

        ctx.launcher.terminate = terminate
        outcome = ctx.stop_role("srv")
    finally:
        watch.stop()
    assert outcome.ending == "planned_stop" and outcome.pool.uid_tasks == 300


class _Late(_DiskWatch):
    """A watch whose memory breach and pids count of the server appear only once it is asked to
    stop."""

    def __init__(self) -> None:
        super().__init__()
        #: Whether the stop's SIGTERM was sent.
        self.stopped = False

    def breach(self, step) -> MemorySample | None:
        if not self.stopped:
            return None
        return MemorySample(at=0.0, current=5 * 2**30, swap_current=0, reclaimable=0, oom_kills=0)

    def pids_hits(self, step) -> PidsHits | None:
        return PidsHits(count=3, rose_at=0.0) if self.stopped else None


@pytest.mark.parametrize("late", ["memory", "pids"])
def test_what_a_server_does_once_asked_to_stop_does_not_count(tmp_path, apptainer, late):
    # Its memory breach and its pids count are read as they were before the stop's SIGTERM, as its
    # disk is.
    cgroups = tmp_path / "cg"
    watch = _Late()
    _, ctx = _ctx(tmp_path, LocalLauncher(cgroup_root=cgroups), watch=watch)
    spec = ContainerSpec(image=_pinned(ctx), workdir=ctx.role_dir("srv") / "wd",
                         command=(sys.executable, "-c", "import time; time.sleep(30)"))
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    deadline = time.monotonic() + 10
    while not ctx.command_started("srv") and time.monotonic() < deadline:
        time.sleep(0.05)
    real = ctx.launcher.terminate

    def terminate(handles):
        if late == "memory":
            watch.stopped = True
        else:
            (cgroups / "cg-srv" / "pids.events").write_text("max 3\n")
        real(handles)

    ctx.launcher.terminate = terminate
    outcome = ctx.stop_role("srv")
    assert outcome.ending == "planned_stop" and classify_role(outcome) is None


def test_a_container_with_a_ready_source_names_its_reaper_to_the_disk_watch(tmp_path, apptainer):
    """The build sandbox's reaper, a harness process in the commands' PID namespace, makes itself
    non-dumpable: the stage gives the watch a source of the harness's pids there, the reaper found
    by the ready source's pids, None while those are unknown; any other contestant container has
    none."""
    watch = _DiskWatch()
    _, ctx = _ctx(tmp_path, LocalLauncher(), watch=watch)
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("srv") / "wd")
    ctx.start_container("srv", spec, SHAPE, owner="contestant")
    ctx.wait_roles()
    assert tuple(watch.harness_pids()) == ()
    spec = ContainerSpec(image=_pinned(ctx), command=("true",), workdir=ctx.role_dir("box") / "wd")
    ctx.start_container("box", spec, SHAPE, owner="contestant", ready_source=lambda: None)
    ctx.wait_roles()
    assert watch.harness_pids() is None
