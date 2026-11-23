"""Stand-ins for Slurm role steps, so stages and the runner run on one machine without srun."""

from __future__ import annotations

import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from hls.pipeline.plan import Plan
from hls.spec.resources import GrantedResources
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.base import JobState, Launcher, NodeSize, RoleFailed, RoleHandle, RoleReport, RoleSpec
from hls.substrate.grant import Measured
from hls.substrate.log_cap import lost, spawn
from hls.substrate.role_exec import Placement, check_role, dropped, set_process_limit, split_refusal
from hls.substrate.slurm.launcher import srun_argv, step_env, step_spec

FAKE_ROLE = Path(__file__).with_name("fake_noop_role.py")
FAKE_STEP = Path(__file__).with_name("fake_step.py")
#: The node a ``LocalLauncher``'s roles run on unless a test gives another: eight GPUs and 64
#: CPUs, as the controller's record of a GPU node gives them. A test never reads the size of
#: the machine it runs on.
NODE = NodeSize(gpus=8, cpus=64)


def fake_uuid(i: int) -> str:
    """The UUID of fake GPU number *i*, in the form nvidia-smi prints."""
    return f"GPU-00000000-0000-0000-0000-{i:012d}"


def _apparent_gib(root: Path) -> float:
    """What the files under *root* hold in GiB, by their apparent size, links not followed."""
    total = 0
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            try:
                total += os.lstat(os.path.join(dirpath, f)).st_size
            except FileNotFoundError:
                pass
    return total / 2**30


#: What a late step runs first (``LocalLauncher.late_sec``): it sleeps, puts its grant report in
#: place, then becomes the role's command.
_LATE_STEP = ("import os, sys, time; time.sleep(float(sys.argv[1])); os.replace(sys.argv[2], sys.argv[3]); "
              "os.execvp(sys.argv[4], sys.argv[4:])")


def _exit_with(code: int) -> list[str]:
    return [sys.executable, "-c", f"import sys; sys.exit({code})"]


class FakeClock:
    """A clock that moves only when a fake step is waited on, for a stage context and the
    launcher it uses: a wait of minutes then takes none."""

    def __init__(self, now: float = 1_000_000.0) -> None:
        #: The time it reads.
        self.now = now

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        """Move the clock *seconds* on."""
        self.now += max(0.0, seconds)


class LocalRoleHandle(RoleHandle):
    """A role running as a local process."""

    def __init__(self, spec: RoleSpec, proc: subprocess.Popen, report_path: Path, uuids: list[str], *,
                 scheduler_ended: bool = False, step_state: JobState | None = None, no_report: bool = False,
                 query_sec: float = 0.0, calls: list[tuple[str, str]] | None = None,
                 log_cappers: tuple[subprocess.Popen, ...] = (), clock: FakeClock | None = None,
                 start_at: float = -math.inf, run_sec: float = 0.0, unreported: bool = False,
                 launch_evidence: dict[str, str] | None = None, late: bool = False) -> None:
        self.spec = spec
        #: The processes that write the role's capped stdout and stderr; none when uncapped.
        self.log_cappers = log_cappers
        #: The role's process.
        self.proc = proc
        #: The grant report file the launcher wrote.
        self.report_path = report_path
        #: The fake GPUs the role was given.
        self.uuids = uuids
        #: What ``ended_by_scheduler`` answers.
        self.scheduler_ended = scheduler_ended
        #: What ``end_state`` answers.
        self.step_state = step_state
        #: True when ``report`` fails the way it does for a step that ended before reporting.
        self.no_report = no_report
        #: Seconds ``ended_by_scheduler`` takes to answer, as a Slurm accounting query can.
        self.query_sec = query_sec
        #: How many times ``ended_by_scheduler`` was asked.
        self.asked = 0
        #: Each ``wait``, ``terminate`` (the launcher's SIGTERM to this role) and ``kill`` as
        #: (what, role name), so a test can check their order; the launcher passes the list all
        #: its handles share.
        self.calls = [] if calls is None else calls
        #: The timeout of each ``report`` call.
        self.report_timeouts: list[float] = []
        #: The clock a wait for the step's start or its report passes on; None when the step
        #: starts and reports at once.
        self.clock = clock
        #: When the step starts, on ``clock``; its report is there from then on.
        self.start_at = start_at
        #: When the step ends at the earliest, on ``clock``: until then ``poll`` reads it running and
        #: ``wait`` passes the clock, as a step that works that long would. A stop ends it there.
        self.end_at = start_at + run_sec
        #: True when the step starts but never reports.
        self.unreported = unreported
        #: What ``launch_evidence`` answers.
        self.evidence = dict(launch_evidence or {})
        #: How many times ``launch_evidence`` was asked.
        self.evidence_asked = 0
        #: True when the step starts late in real time: its report is there only once it started.
        self.late = late

    def started(self) -> bool:
        if self.late and not self.report_path.exists():
            return False
        return self.clock is None or self.clock() >= self.start_at

    def launch_evidence(self) -> dict[str, str]:
        self.evidence_asked += 1
        return dict(self.evidence)

    def _working(self) -> bool:
        """True while the step still works on ``clock`` (``end_at``)."""
        return self.clock is not None and self.clock() < self.end_at

    def _stopped(self) -> None:
        """End the step's work on ``clock`` now: it was stopped."""
        if self.clock is not None:
            self.end_at = min(self.end_at, self.clock())

    def close_logs(self, timeout_sec: float) -> None:
        for capper in self.log_cappers:
            capper.wait(timeout_sec)

    def logs_lost(self) -> str | None:
        return lost(self.log_cappers)

    def report(self, timeout_sec: float) -> RoleReport:
        self.report_timeouts.append(timeout_sec)
        if self.no_report:
            raise RoleFailed("no report")
        if self.late:
            give_up = time.monotonic() + timeout_sec
            while not self.report_path.exists():
                if self.proc.poll() is not None:
                    raise RoleFailed(f"role {self.spec.name} ended before reporting its grant")
                if time.monotonic() >= give_up:
                    raise TimeoutError(f"role {self.spec.name} did not report its grant in {timeout_sec}s")
                time.sleep(0.05)
        if self.clock is not None:
            # The report is there once the step has started, unless the step never reports.
            due = math.inf if self.unreported else self.start_at
            if self.clock() < due:
                self.clock.sleep(min(timeout_sec, due - self.clock()))
            if self.clock() < due:
                raise TimeoutError(f"role {self.spec.name} did not report its grant in {timeout_sec}s")
        return RoleReport.model_validate_json(self.report_path.read_bytes())

    def poll(self) -> int | None:
        return None if self._working() else self.proc.poll()

    def wait(self, timeout_sec: float | None = None) -> int | None:
        self.calls.append(("wait", self.spec.name))
        if self._working() and self.clock is not None:
            left = self.end_at - self.clock()
            self.clock.sleep(left if timeout_sec is None else min(timeout_sec, left))
            if self._working():
                return None
        try:
            return self.proc.wait(timeout_sec)
        except subprocess.TimeoutExpired:
            return None

    def send_term(self) -> None:
        """SIGTERM to the role's process, at once."""
        self.calls.append(("terminate", self.spec.name))
        self._stopped()
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        self.calls.append(("kill", self.spec.name))
        self._stopped()
        if self.proc.poll() is None:
            self.proc.kill()

    def ended_by_scheduler(self) -> bool:
        self.asked += 1
        time.sleep(self.query_sec)
        return self.scheduler_ended

    def end_state(self) -> JobState | None:
        return self.step_state


def _process_limit(ceiling: int) -> Callable[[], None]:
    """What a role's child runs before its command: the step wrapper's process ceiling."""

    def apply() -> None:
        problem = set_process_limit(ceiling)
        if problem is not None:
            raise OSError(problem)

    return apply


class LocalLauncher(Launcher):
    """Runs each role as a local process. GPUs are fake UUIDs handed out from a pool the way
    Slurm hands out a job's GPUs. A no-op role runs ``fake_noop_role.py``, which reports
    the UUIDs it was given. *cgroup_root* gives each role a directory of fake memory files. A
    role with an env file runs under ``fake_step.py``, which takes the file the way the step
    wrapper does. A container role becomes its step as the Slurm launcher makes it
    (``step_spec``), with ``runtime`` and the seal settings ``use_plan`` takes from a plan, and
    ``launched`` records that step. Every spec then goes through what the Slurm launcher checks
    before srun (``srun_argv`` and ``step_env``), so a role that real Slurm would refuse fails
    here too.

    Each of these is a tuple of role names:
    - *mismatch*: the role gets a report with a grant mismatch and exits 97, as the wrapper would;
    - *exit_97*: the role exits 97 with a report that has no mismatches;
    - *scheduler_ended*: the role exits 1 and its handle says the scheduler ended it;
    - *out_of_memory*: the role exits 1 and its handle says its step ended OUT_OF_MEMORY, as when
      the kernel OOM-killed it after the memory watch's last sample;
    - *no_report*: the handle's ``report`` raises RoleFailed, and the role sleeps 60 s until killed;
    - *no_view*: a no-op role reports that it could not read its GPUs from the driver;
    - *unreported*: the role's step starts but never reports, and the role sleeps 60 s until
      killed: its handle's ``report`` waits out its timeout on *clock* and raises TimeoutError.

    *start_after* maps a role to the seconds on *clock* from its launch until its step starts, as
    a busy controller makes a step wait; ``math.inf`` for a step that never starts, whose role
    then sleeps 60 s until killed. Until then the handle's ``started`` is False and its ``report``
    waits out its timeout on *clock* and raises TimeoutError; from then on the report is there.
    *run_sec* maps a role to the seconds on *clock* its step works from its start: until then
    its handle's ``poll`` reads it running and its ``wait`` passes *clock*, and a stop ends it.
    All three need *clock*; a role none names starts, reports and works at once. *late_sec* maps a
    role to real seconds its step starts after its launch, as a busy controller grants a step late:
    its grant report and its command both come that much later.
    *launch_evidence* is what every handle's ``launch_evidence`` says.

    *steps_under* lays the roles' fake cgroups out the way Slurm lays out a job's steps: role
    number <n> in launch order gets ``<steps_under>/step_<n>/user`` and step id ``1.<n>``;
    ``step_ids`` counts the step ids handed out, and a test can raise it to leave a gap, as a step
    nobody here started would. *wrong_share* hands a shared role a GPU it does not share;
    *scheduler_query_sec* is how long each handle takes to say whether the scheduler ended its
    step; *signal_sec* is how long each ``terminate`` call takes, as one scancel can while the
    controller is busy. A role whose command is ``<python> -m <module> ...`` with *module* in
    *replace_modules* runs the mapped script instead, with the same arguments. A role's declared
    PYTHONPATH comes first, before the test process's own path. *task_prolog* is added to every
    step's environment after that, as a site's TaskProlog exports variables into each step.

    *node_disk_gib* gives the node that much local disk, which the job's node-local *scratch*
    uses: each role's free disk is *node_disk_gib* less the apparent size of the files under
    *scratch* when the role starts, and its report's mismatches are what the step wrapper's own
    check (``role_exec.check_role``) finds in that grant, so a role that declares more disk than
    is left exits 97 as it would on a node. Without it every role has 100 GiB free.

    A role's process ceiling (``RoleSpec.nproc``) is recorded with its spec; with *apply_nproc*
    its child also sets it, as the step wrapper does, which only a ceiling at or under the
    host's own hard limit allows.

    *placement* maps a role name to the CPUs, NUMA nodes and NUMA split its report gives, as
    the step wrapper measures them (``role_exec.placement``); other roles report none. A role
    held to one node (``RoleSpec.one_numa_node``) whose placement is split gets the wrapper's
    refusal (``role_exec.split_refusal``) as a mismatch and exits 97.

    *node* is the size of the node the roles run on (``node_size``); None, a node not given, is
    ``NODE``."""

    def __init__(self, gpus: int = 2, *, cgroup_root: Path | None = None, mismatch: tuple[str, ...] = (),
                 exit_97: tuple[str, ...] = (), scheduler_ended: tuple[str, ...] = (),
                 out_of_memory: tuple[str, ...] = (), no_report: tuple[str, ...] = (), no_view: tuple[str, ...] = (),
                 wrong_share: bool = False, scheduler_query_sec: float = 0.0, signal_sec: float = 0.0,
                 steps_under: Path | None = None, replace_modules: dict[str, Path] | None = None,
                 task_prolog: dict[str, str] | None = None, node_disk_gib: float | None = None,
                 scratch: Path | None = None, apply_nproc: bool = False,
                 log_keep_bytes: int | None = None, clock: FakeClock | None = None,
                 start_after: dict[str, float] | None = None, unreported: tuple[str, ...] = (),
                 run_sec: dict[str, float] | None = None, launch_evidence: dict[str, str] | None = None,
                 late_sec: dict[str, float] | None = None,
                 placement: dict[str, tuple[list[int], list[int], bool]] | None = None,
                 node: NodeSize | None = None) -> None:
        if (node_disk_gib is None) != (scratch is None):
            raise ValueError("node_disk_gib and scratch go together")
        if (start_after or unreported or run_sec) and clock is None:
            raise ValueError("start_after, unreported and run_sec need a clock")
        #: UUIDs of the fake GPUs the launcher hands out.
        self.pool = [fake_uuid(i) for i in range(gpus)]
        #: Directory of the fake memory files of each role's step; None puts them in the role's log directory.
        self.cgroup_root = cgroup_root
        #: A fake job cgroup to lay the roles' steps out under, as ``step_<n>/user``; None for none.
        self.steps_under = steps_under
        #: Step ids handed out so far; the next role's step is ``1.<step_ids + 1>``.
        self.step_ids = 0
        #: Roles that get a grant mismatch.
        self.mismatch = mismatch
        #: Roles that exit 97 with a clean report.
        self.exit_97 = exit_97
        #: Roles the scheduler ends.
        self.scheduler_ended = scheduler_ended
        #: Roles whose steps end OUT_OF_MEMORY.
        self.out_of_memory = out_of_memory
        #: Roles that never report.
        self.no_report = no_report
        #: No-op roles that cannot read their GPUs.
        self.no_view = no_view
        #: True when a shared role is given a GPU it does not share.
        self.wrong_share = wrong_share
        #: Seconds each handle's ``ended_by_scheduler`` takes to answer.
        self.scheduler_query_sec = scheduler_query_sec
        #: Seconds each ``terminate`` call takes.
        self.signal_sec = signal_sec
        #: The roles of each ``terminate`` call of the launcher, in order.
        self.terminations: list[list[str]] = []
        #: Module name to the script run in its place.
        self.replace_modules = dict(replace_modules or {})
        #: Variables every step's environment gets last, as a site's TaskProlog exports them.
        self.task_prolog = dict(task_prolog or {})
        #: The node's local disk in GiB; None gives every role 100 GiB free.
        self.node_disk_gib = node_disk_gib
        #: The job's node-local scratch, whose files use the node's disk.
        self.scratch = scratch
        #: Role name to the CPUs, NUMA nodes and NUMA split its report gives.
        self.placement = dict(placement or {})
        #: Whether a role's child sets its process ceiling.
        self.apply_nproc = apply_nproc
        #: Every role started, in order.
        self.handles: list[LocalRoleHandle] = []
        #: The spec of every role started, in order.
        self.launched: list[RoleSpec] = []
        #: Every ``wait``, ``terminate`` and ``kill`` of every handle, in order.
        self.calls: list[tuple[str, str]] = []
        #: Bytes every role's stdout and stderr each keep at their start and end, through the
        #: runner's log cappers (``hls.substrate.log_cap``); None writes them uncapped.
        self.log_keep_bytes = log_keep_bytes
        #: The clock the steps of ``start_after`` and ``unreported`` wait on.
        self.clock = clock
        #: Seconds from each named role's launch until its step starts, on ``clock``.
        self.start_after = dict(start_after or {})
        #: Roles whose steps start but never report.
        self.unreported = unreported
        #: Seconds each named role's step works from its start, on ``clock``.
        self.run_sec = dict(run_sec or {})
        #: Real seconds each named role's step starts after its launch.
        self.late_sec = dict(late_sec or {})
        #: What every handle's ``launch_evidence`` says.
        self.launch_evidence = dict(launch_evidence or {})
        #: The size of the node the roles run on, as the test gave it; None when it gave none.
        self.node = node
        #: Builds the command line of each container role, as the Slurm launcher's does.
        self.runtime = ApptainerRuntime()
        #: The interpreter a seal's interceptor runs with, the pasta program and the upstream
        #: proxy a seal is given, as a plan sets them (``use_plan``).
        self.seal_python: str = sys.executable
        self.pasta: Path | None = None
        self.upstream_proxy: str | None = None

    def use_plan(self, plan: Plan) -> LocalLauncher:
        """Take the seal settings the runner gives the Slurm launcher from *plan*: its toolchain's
        interpreter, the site's pasta and its upstream proxy. Return this launcher."""
        self.seal_python = str(plan.toolchain.python)
        self.pasta = plan.settings.site.pasta_path
        self.upstream_proxy = plan.settings.site.egress_upstream_proxy
        return self

    def node_size(self) -> NodeSize:
        return NODE if self.node is None else self.node

    def _free(self) -> list[str]:
        held = {u for h in self.handles if h.poll() is None and h.spec.name != "shared" for u in h.uuids}
        return [u for u in self.pool if u not in held]

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        spec = step_spec(spec, runtime=self.runtime, python=self.seal_python, pasta=self.pasta,
                         upstream_proxy=self.upstream_proxy)
        srun_argv(spec, python=sys.executable, report=spec.log_dir / f"{spec.name}.report.json",
                  scratch=spec.log_dir, shared=None, allocation_gpus=len(self.pool))
        step_env({}, spec.env, tmpdir=spec.log_dir)
        self.launched.append(spec)
        if share_gpus_of is not None:
            uuids = list(share_gpus_of.report(0.0).gpu_uuids)
            if self.wrong_share:
                uuids = [u for u in self.pool if u not in uuids][: len(uuids)]
        else:
            free = self._free()
            if len(free) < spec.shape.gpus:
                raise RuntimeError(f"only {len(free)} free GPUs for {spec.name}")
            uuids = free[: spec.shape.gpus]
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        self.step_ids += 1
        if self.steps_under is not None:
            cgroup = self.steps_under / f"step_{self.step_ids}" / "user"
        else:
            cgroup = (self.cgroup_root or spec.log_dir) / f"cg-{spec.name}"
        cgroup.mkdir(parents=True, exist_ok=True)
        mismatches = ["gpus: declared 1, granted 2"] if spec.name in self.mismatch else []
        free = 100.0
        if self.node_disk_gib is not None and self.scratch is not None:
            free = max(self.node_disk_gib - _apparent_gib(self.scratch), 0.0)
        granted = GrantedResources(gpus=len(uuids), gpu_kind=spec.shape.gpu_kind if uuids else None,
                                   cpus=spec.shape.cpus, memory_gib=float(spec.shape.memory_gib),
                                   disk_free_gib=free)
        if self.node_disk_gib is not None:
            measured = Measured(granted=granted, gpu_uuids=uuids, swap_max_bytes=None)
            mismatches += check_role(spec.shape, measured, shared=None)
        local_path = os.pathsep.join(p for p in sys.path if p)
        env = {**os.environ, **spec.env,
               "PYTHONPATH": os.pathsep.join(p for p in (spec.env.get("PYTHONPATH"), local_path) if p),
               **self.task_prolog}
        # What the step wrapper removes from a role's step: fake_step removes them from a role
        # with an env file, and the launcher itself from one without, which runs no fake_step.
        stripped = sorted(k for k in env if dropped(k, container=spec.env_fd))
        if spec.env_file is None:
            for name in stripped:
                del env[name]
        report = spec.log_dir / f"{spec.name}.report.json"
        cpus, numa_nodes, split = self.placement.get(spec.name, ([], [], False))
        refusal = split_refusal(Placement(cpus=cpus, numa_nodes=numa_nodes, split=split))
        if spec.one_numa_node and refusal is not None:
            mismatches = [*mismatches, refusal]
        late = spec.name in self.late_sec
        # A late step's report is put in place when the step starts.
        pending = report.with_name(f"{report.name}.pending") if late else report
        if late:
            report.unlink(missing_ok=True)
        pending.write_text(RoleReport(host="local", step=f"1.{self.step_ids}", cgroup_dir=str(cgroup),
                                     gpu_uuids=uuids, cuda_visible_devices=",".join(uuids) or None, granted=granted,
                                     swap_max_bytes=None, mismatches=mismatches, stripped_env=stripped,
                                     cpus=cpus, numa_nodes=numa_nodes, numa_split=split).model_dump_json())
        argv = list(spec.argv)
        if mismatches or spec.name in self.exit_97:
            argv = _exit_with(97)
        elif spec.name in self.scheduler_ended or spec.name in self.out_of_memory:
            argv = _exit_with(1)
        elif (spec.name in self.no_report or spec.name in self.unreported
              or self.start_after.get(spec.name) == math.inf):
            argv = [sys.executable, "-c", "import time; time.sleep(60)"]
        elif "hls.substrate.noop_role" in argv:
            rest = argv[argv.index("hls.substrate.noop_role") + 1:]
            view = ["--no-view"] if spec.name in self.no_view else []
            argv = [sys.executable, str(FAKE_ROLE), "--uuids", ",".join(uuids), *view, *rest]
        elif len(argv) > 2 and argv[1] == "-m" and argv[2] in self.replace_modules:
            argv = [sys.executable, str(self.replace_modules[argv[2]]), *argv[3:]]
        if spec.env_file is not None:
            argv = [sys.executable, str(FAKE_STEP), "--env-file", str(spec.env_file),
                    *(["--env-fd"] if spec.env_fd else []), "--", *argv]
        if late:
            argv = [sys.executable, "-c", _LATE_STEP, str(self.late_sec[spec.name]), str(pending), str(report), *argv]
        logs = (spec.log_dir / f"{spec.name}.out", spec.log_dir / f"{spec.name}.err")
        limit = _process_limit(spec.nproc) if self.apply_nproc and spec.nproc else None
        cappers: tuple[subprocess.Popen, ...] = ()
        if self.log_keep_bytes is not None:
            spawned = [spawn(path, keep=self.log_keep_bytes, python=sys.executable,
                             errors=path.with_name(f"{path.name}.cap.err")) for path in logs]
            cappers = tuple(capper for _, capper in spawned)
            try:
                proc = subprocess.Popen(argv, env=env, stdout=spawned[0][0], stderr=spawned[1][0],
                                        stdin=subprocess.DEVNULL, preexec_fn=limit)
            finally:
                for writer, _ in spawned:
                    os.close(writer)
        else:
            with open(logs[0], "ab") as out, open(logs[1], "ab") as err:
                proc = subprocess.Popen(argv, env=env, stdout=out, stderr=err, stdin=subprocess.DEVNULL,
                                        preexec_fn=limit)
        start_at = -math.inf
        if self.clock is not None:
            start_at = self.clock() + self.start_after.get(spec.name, 0.0)
        handle = LocalRoleHandle(spec, proc, report, uuids, scheduler_ended=spec.name in self.scheduler_ended,
                                 step_state=JobState.OUT_OF_MEMORY if spec.name in self.out_of_memory else None,
                                 no_report=spec.name in self.no_report, query_sec=self.scheduler_query_sec,
                                 calls=self.calls, log_cappers=cappers, clock=self.clock, start_at=start_at,
                                 run_sec=self.run_sec.get(spec.name, 0.0), unreported=spec.name in self.unreported,
                                 launch_evidence=self.launch_evidence, late=late)
        self.handles.append(handle)
        return handle

    def terminate(self, handles) -> None:
        self.terminations.append([h.spec.name for h in handles])
        time.sleep(self.signal_sec)
        for h in handles:
            h.send_term()
