"""Role steps on Slurm: one ``srun`` step per role, from a clean environment.

Slurm behavior the launcher relies on:
- The batch step's environment carries the job's own request (SLURM_GPUS,
  SLURM_TRES_PER_TASK, SLURM_CPUS_PER_TASK, SLURM_MEM_PER_NODE and others), and srun reads
  it. A step asking ``--gres=gpu:1`` with SLURM_GPUS inherited claims every GPU of the job
  and silently serializes every later GPU step. So each step's environment drops the
  job-level resource variables, and each step passes ``--exact --gpus --cpus-per-task
  --mem`` itself.
- A step with no GPUs passes ``--gres=none``, since by default a step gets all of the
  job's GRES.
- The one shared step, the reference service on the build's GPUs, runs with
  ``--overlap`` and every GPU of the allocation, and its wrapper sets
  CUDA_VISIBLE_DEVICES to the shared role's UUIDs. An overlap step that asks for fewer
  GPUs lands on the allocation's lowest-index GPUs, whichever step holds them. Its device
  cgroup then admits every GPU of the allocation. Its CPU count is its own
  ``--cpus-per-task``, placed on the allocation's first cores, which an exclusive step may
  also hold. It runs harness code, so both are accepted.
- A role that shares the CPUs of the others (``RoleSpec.overlap``), such as the driver beside
  the server it drives, also runs with ``--overlap`` and
  its own ``--cpus-per-task`` and ``--mem``, and with no GPU.
- Every step runs ``hls.substrate.role_exec`` first, which reports and checks the step's
  grant before the role's command starts. For a contestant's role it also sets
  ``oom_score_adj`` 1000 first, which the command and a container's processes inherit.
- A container role (``RoleSpec.container``) runs as ``apptainer exec`` in its step, behind its
  seal when it has one (``container_step``).
- A busy controller can take minutes to grant a step, while srun
  prints nothing and Slurm lists no step until it is granted (its
  accounting gives the step's submit time as its start). So the start of a step is seen in the
  start marker its wrapper writes first (``role_exec.started_path``), not asked of the
  controller; the controller is asked only about a step that did not start or report in time
  (``SlurmRoleHandle.launch_evidence``).
- Stopping. SIGTERM sent to srun makes srun forward SIGKILL, so the role never sees it. A
  graceful stop is ``scancel --signal=TERM <job>.<step>``, which delivers SIGTERM to the
  role and leaves the step's state as the role's own exit. Stopping several roles names all
  their steps in one scancel, which can take its whole timeout while the controller is busy.
  A hard stop signals every process in the step's cgroup, then srun.
- A step's state in accounting says who ended it: the role's own exit leaves FAILED,
  COMPLETED or OUT_OF_MEMORY, while a step ended through srun or scancel is CANCELLED, and one
  that reached its time limit TIMEOUT. That limit is the harness's (``--time``), so a TIMEOUT is
  never the role's own ending. OUT_OF_MEMORY is also the one lasting record of an OOM kill: the
  step's cgroup, where the memory watch reads its oom_kill count, is gone once the step ends."""

from __future__ import annotations

import logging
import math
import os
import re
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from pydantic import ValidationError

from hls.substrate.apptainer.argv import RUNTIME_PREFIXES, ApptainerRuntime
from hls.substrate.apptainer.egress import SEAL_TOOL_PATH_VAR, seal_env
from hls.substrate.base import (JobState, Launcher, NodeSize, RoleFailed, RoleHandle, RoleReport, RoleSpec,
                                own_node_size)
from hls.substrate.cgroup import CGROUP_ROOT, kill_all
from hls.substrate.container import ContainerRole
from hls.substrate.log_cap import LOG_KEEP_BYTES, errors_file, lost, spawn
from hls.substrate.exits import PROCESS_CEILING_EXIT
from hls.substrate.role_exec import APPTAINER_KEPT, CONTESTANT_OOM_SCORE_ADJ, ENV_FD_PATH, started_path
from hls.substrate.ucounts import ucounts_env
from hls.substrate.slurm.commands import CommandResult, SlurmCommands
from hls.substrate.slurm.scheduler import normalize_state
from hls.util import remove_tree, validation_errors

log = logging.getLogger(__name__)

# Job-level variables srun reads as a resource or placement request.
_JOB_REQUEST_VARS = re.compile(
    r"SLURM_(GPUS(_PER_\w+)?|CPUS_PER_(GPU|TASK)|MEM_PER_\w+|TRES_\w+|GPU_BIND|GPU_FREQ|GRES|STEP_GRES"
    r"|NTASKS(_PER_\w+)?|NPROCS|EXACT|EXCLUSIVE|OVERLAP|OVERCOMMIT|EXPORT_ENV|DISTRIBUTION|CPU_BIND\w*"
    r"|MEM_BIND\w*|HINT|THREADS_PER_CORE)"
)
# Names a role's declared environment may not set: the launcher owns them.
_LAUNCHER_VARS = frozenset({"PATH", "HOME", "LANG", "TMPDIR", "CUDA_VISIBLE_DEVICES"})
# Prefixes a role's declared environment may not use: Slurm's own variables, the ones srun
# reads as options, the ones apptainer reads as its own settings (bind paths, env file, mount
# switches), and the ones it copies into a container, which reach it only through the role's
# env file. A role may still declare the two of ``APPTAINER_KEPT``, which only say where
# apptainer keeps its cache and its build files: a build or an extraction puts them on
# node-local disk.
_RESERVED_PREFIXES = ("SLURM_", "SRUN_", *RUNTIME_PREFIXES)
#: Seconds between looks at a role's report file while waiting for it.
_REPORT_POLL_SEC = 0.2
#: Seconds ``kill`` waits for srun to end the step after SIGTERM before it signals SIGKILL.
_KILL_GRACE_SEC = 5.0
#: Seconds between looks at srun while ``kill`` waits for it.
_KILL_POLL_SEC = 0.1
#: Seconds between sacct queries for a step that has ended, and how long they go on.
_STATE_POLL_SEC = 2.0
_STATE_WAIT_SEC = 30.0
# Seconds each tool a handle runs may take. A role that ended costs at most the wait above
# plus one call of each, so a stuck controller cannot hold the stage for long.
_STEP_TOOL_TIMEOUT_SEC = {"sacct": 10.0, "scancel": 10.0, "squeue": 10.0}
# squeue's fields for each step of a job: its id, its name and its state.
_STEP_FORMAT = "--Format=StepId:|,StepName:|,StepState:|"
# Bytes at the end of a role's stderr searched for srun's lines, the last few kept, each cut.
_SRUN_TAIL_BYTES, _SRUN_LINES, _SRUN_LINE_CHARS = 65536, 3, 300
# A step id as a role's report names it: "<job>.<step>".
_STEP_ID = re.compile(r"([0-9]+)\.([0-9]+)")
# Terminal step states that come from the role's own exit, running out of memory included. Any
# other terminal state was decided by the scheduler, a TIMEOUT at the limit the harness set among
# them.
_ROLE_STATES = frozenset({JobState.COMPLETED, JobState.FAILED, JobState.OUT_OF_MEMORY})


def reserved_variable(name: str) -> bool:
    """Whether the launcher sets or forbids variable *name*, so that no secret grant may set it,
    and a role's declared environment only if it is one of ``APPTAINER_KEPT``."""
    return name in _LAUNCHER_VARS or name.startswith(_RESERVED_PREFIXES)


def step_id(environ: Mapping[str, str]) -> str:
    """The id of the step whose environment is *environ*, as its report names it: "<job>.<step>"
    from ``SLURM_JOB_ID`` and ``SLURM_STEP_ID``, which slurmd sets in every step, with "?" for
    either that is unset."""
    return f"{environ.get('SLURM_JOB_ID', '?')}.{environ.get('SLURM_STEP_ID', '?')}"


def step_env(batch_env: Mapping[str, str], role_env: Mapping[str, str], *, tmpdir: Path) -> dict[str, str]:
    """A role step's environment: the batch step's SLURM_* variables without the job-level
    request, the PATH and HOME the batch script set, a node-local TMPDIR, and the role's own
    declared variables. Nothing else from the batch step reaches the role. The role may declare
    no variable the launcher owns (``reserved_variable``) but the two of ``APPTAINER_KEPT``."""
    bad = sorted(k for k in role_env if reserved_variable(k) and k not in APPTAINER_KEPT)
    if bad:
        raise ValueError(f"a role may not set {bad}")
    env = {k: v for k, v in batch_env.items() if k.startswith("SLURM_") and not _JOB_REQUEST_VARS.fullmatch(k)}
    env.update(PATH=batch_env.get("PATH", "/usr/bin:/bin"), HOME=batch_env.get("HOME", "/"), LANG="C.UTF-8",
               TMPDIR=str(tmpdir))
    env.update(role_env)
    return env


def _signal_steps(commands: Callable[..., CommandResult], steps: Sequence[str]) -> None:
    """Send SIGTERM to every one of *steps*, "<job>.<step>" ids, with one scancel."""
    if not steps:
        return
    result = commands("scancel", "--signal=TERM", *steps)
    if not result.ok:
        log.warning("scancel --signal=TERM %s did not succeed: %s", " ".join(steps), result.stderr.strip())


def node_size(commands: Callable[..., CommandResult], node: str) -> NodeSize:
    """The GPUs and CPUs of *node* in the controller's record of it: ``gres/gpu`` and ``cpu`` of
    its ``CfgTRES``, a node with no GPU in it having none. Unknown (None) when the command fails
    or its answer holds no such record."""
    result = commands("scontrol", "show", "node", "--oneliner", node)
    if not result.ok:
        return NodeSize(gpus=None, cpus=None)
    tres = next((f.split("=", 1)[1] for f in result.stdout.split() if f.startswith("CfgTRES=")), None)
    if tres is None:
        return NodeSize(gpus=None, cpus=None)
    counts = dict(item.partition("=")[::2] for item in tres.split(","))
    cpus = counts.get("cpu", "")
    gpus = counts.get("gres/gpu", "0")
    return NodeSize(gpus=int(gpus) if gpus.isdecimal() else None, cpus=int(cpus) if cpus.isdecimal() else None)


def _empty_tmpdir(tmpdir: Path) -> None:
    """Make *tmpdir* an empty directory. One that is there is emptied in place, so a launch needs
    no new space on the node's disk, which a contestant may have filled; anything else at the
    path is removed and the directory made."""
    if tmpdir.is_dir() and not tmpdir.is_symlink():
        with os.scandir(tmpdir) as entries:
            names = [entry.path for entry in entries]
        for name in names:
            remove_tree(Path(name))
        return
    remove_tree(tmpdir)
    tmpdir.mkdir(parents=True)


def container_step(role: ContainerRole, *, runtime: ApptainerRuntime, python: str, pasta: Path | None,
                   upstream_proxy: str | None) -> tuple[list[str], dict[str, str]]:
    """The command and declared environment of the step of container role *role*: ``apptainer
    exec``, which reads the container's variables from the env file the step wrapper hands over on
    descriptor 3, behind the seal of a sealed container, whose interceptor *python* runs. A sealed
    step's variables are the seal's host-side ones (``seal_env``): its files, *pasta*, the
    site's *upstream_proxy* and the kernel-count limits it writes (``ucounts_env``); none reaches
    the container. Raise ValueError for a container the command line cannot express."""
    argv = runtime.argv(role.spec, env_file=ENV_FD_PATH, seal_python=python)
    if role.spec.seal is None:
        return argv, {}
    assert role.seal_files is not None
    # The firewall tools' directories come from the runner's own environment only when it names
    # them (a test's fakes); production leaves them to the seal's /usr/sbin:/sbin.
    env = {**seal_env(role.seal_files, role.spec.seal, pasta=pasta, upstream_proxy=upstream_proxy,
                      tool_path=os.environ.get(SEAL_TOOL_PATH_VAR)),
           **ucounts_env(role.ucounts)}
    return argv, env


def step_spec(spec: RoleSpec, *, runtime: ApptainerRuntime, python: str, pasta: Path | None,
              upstream_proxy: str | None) -> RoleSpec:
    """*spec* as its step runs it: a container role (``RoleSpec.container``) with the command
    and variables of ``container_step`` and its env file handed over on descriptor 3; any other
    role as it is."""
    if spec.container is None:
        return spec
    argv, env = container_step(spec.container, runtime=runtime, python=python, pasta=pasta,
                               upstream_proxy=upstream_proxy)
    return replace(spec, argv=tuple(argv), env=env, env_file=spec.container.env_file, env_fd=True, container=None)


def srun_argv(spec: RoleSpec, *, python: str, report: Path, scratch: Path, shared: list[str] | None,
              allocation_gpus: int) -> list[str]:
    """The srun command for *spec*: its own step with exactly its slice, or with *shared*,
    an overlap step on every GPU of the allocation that the wrapper narrows to *shared*. A spec
    with ``overlap`` is an overlap step too, on CPUs other steps hold, with no GPU."""
    s = spec.shape
    overlap = shared is not None or spec.overlap
    argv = ["srun", "--overlap" if overlap else "--exact", "--nodes=1", "--ntasks=1",
            f"--cpus-per-task={s.cpus}", f"--mem={s.memory_gib}G"]
    if shared is not None:
        argv.append(f"--gpus={allocation_gpus}")
    elif s.gpus > 0:
        argv.append(f"--gpus={s.gpus}")
    else:
        argv.append("--gres=none")
    argv += [f"--time={math.ceil(s.time_sec / 60)}", f"--job-name={spec.name}", "--export=ALL"]
    wrapper = [python, "-m", "hls.substrate.role_exec", "--report", str(report), "--gpus", str(s.gpus),
               "--cpus", str(s.cpus), "--memory-gib", str(s.memory_gib), "--disk-gib", str(s.disk_gib),
               "--scratch", str(scratch)]
    if s.gpu_kind is not None:
        wrapper += ["--gpu-kind", s.gpu_kind]
    if shared is not None:
        wrapper += ["--share-gpus", ",".join(shared)]
    if spec.env_file is not None:
        wrapper += ["--env-file", str(spec.env_file)]
        if spec.env_fd:
            wrapper.append("--env-fd")
    if spec.owner == "contestant":
        wrapper += ["--oom-score-adj", str(CONTESTANT_OOM_SCORE_ADJ)]
    if spec.nproc is not None:
        wrapper += ["--nproc", str(spec.nproc)]
    if spec.one_numa_node:
        wrapper.append("--one-numa-node")
    return [*argv, *wrapper, "--", *spec.argv]


class SlurmRoleHandle(RoleHandle):
    """A role running under srun."""

    def __init__(self, spec: RoleSpec, proc: subprocess.Popen, report_path: Path, *,
                 job_id: str | None = None, cgroup_root: Path = CGROUP_ROOT,
                 allocation_gpu_uuids: Collection[str] | None = None,
                 commands: Callable[..., CommandResult] | None = None,
                 clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 log_cappers: Sequence[subprocess.Popen] = ()) -> None:
        self.spec = spec
        #: The processes that write the role's capped stdout and stderr.
        self.log_cappers = tuple(log_cappers)
        #: The srun process.
        self.proc = proc
        #: Where the wrapper writes its report.
        self.report_path = report_path
        #: Where the wrapper says the step started, beside the report (``role_exec.started_path``).
        self.started_path = started_path(report_path)
        #: True once the launcher's ``terminate`` or ``kill`` acted on a running step. The step's
        #: end is then the runner's doing, whatever Slurm records for it.
        self.stopped_by_runner = False
        #: The job the step belongs to; None skips the check that a report is for this job.
        self.job_id = job_id
        #: Where cgroup v2 is mounted; a report's cgroup must lie under it.
        self.cgroup_root = cgroup_root
        #: UUIDs of the allocation's GPUs; None skips the check that a report lists only
        #: GPUs of the allocation.
        self.allocation_gpu_uuids = None if allocation_gpu_uuids is None else frozenset(allocation_gpu_uuids)
        #: Runs scancel and sacct; by default each is given 10 seconds.
        self.commands = commands if commands is not None else SlurmCommands(timeout_sec=_STEP_TOOL_TIMEOUT_SEC)
        self._report: RoleReport | None = None
        #: Whether the scheduler ended the step, and the terminal state accounting gave for it;
        #: None until asked.
        self._end: tuple[bool, JobState | None] | None = None
        self._clock = clock
        self._sleep = sleep

    def _foreign(self, report: RoleReport) -> str | None:
        """Why *report* is not one this job's own step could have written, or None."""
        found = _STEP_ID.fullmatch(report.step)
        if found is None:
            return f"step {report.step!r} is not <job>.<step>"
        job, step = found.groups()
        if self.job_id is not None and job != self.job_id:
            return f"step {report.step} belongs to job {job}, not {self.job_id}"
        cgroup = Path(report.cgroup_dir)
        if (cgroup.parts[-3:] != (f"job_{job}", f"step_{step}", "user") or ".." in cgroup.parts
                or not cgroup.is_relative_to(self.cgroup_root)):
            return f"cgroup {cgroup} is not step_{step}/user of job {job} under {self.cgroup_root}"
        if self.allocation_gpu_uuids is not None:
            outside = sorted(set(report.gpu_uuids) - self.allocation_gpu_uuids)
            if outside:
                return f"GPUs {outside} are not in the allocation"
        return None

    def _load(self) -> RoleReport | None:
        """The report once its file exists; None before. The wrapper writes the file in one
        step, so a file that exists is complete: raise RoleFailed when it is not a valid report
        of this job's own step."""
        try:
            data = self.report_path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            report = RoleReport.model_validate_json(data)
        except ValidationError as e:
            raise RoleFailed(f"role {self.spec.name} wrote an invalid grant report: {validation_errors(e)}") from None
        problem = self._foreign(report)
        if problem is not None:
            raise RoleFailed(f"role {self.spec.name} wrote a grant report for another step: {problem}")
        return report

    def _known_report(self) -> RoleReport | None:
        """The report if there is a valid one now, without waiting and without raising."""
        if self._report is None:
            try:
                self._report = self._load()
            except RoleFailed:
                return None
        return self._report

    def started(self) -> bool:
        """True once the step's wrapper wrote its start marker, or the role reported."""
        return self._known_report() is not None or self.started_path.exists()

    def launch_evidence(self) -> dict[str, str]:
        """``slurm_steps``: each step of the job the controller lists under the role's name, as
        ``<job>.<step> <state>``; "none" when it lists none, as for a step it has not granted
        yet; or how squeue failed. ``srun_said``: srun's last lines on the role's stderr, such as
        why it waits for its step, or "nothing". Before the role's command starts, only srun and
        the step wrapper write there."""
        return {"slurm_steps": self._listed_steps(), "srun_said": self._srun_said()}

    def _listed_steps(self) -> str:
        if self.job_id is None:
            return "unknown: no job id"
        result = self.commands("squeue", "--steps", f"--jobs={self.job_id}", "--noheader", _STEP_FORMAT)
        if result.returncode is None:
            return "squeue timed out"
        if not result.ok:
            return f"squeue failed: {result.stderr.strip()[-_SRUN_LINE_CHARS:]}"
        steps = [fields for line in result.stdout.splitlines() if len(fields := line.split("|")) >= 3]
        named = [f"{step} {state}" for step, name, state, *_ in steps if name == self.spec.name]
        return ", ".join(named) or "none"

    def _srun_said(self) -> str:
        err = self.spec.log_dir / f"{self.spec.name}.err"
        try:
            with open(err, "rb") as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - _SRUN_TAIL_BYTES))
                tail = f.read().decode(errors="replace")
        except OSError:
            return "nothing"
        said = [line[:_SRUN_LINE_CHARS] for line in tail.splitlines() if line.startswith("srun: ")]
        return " | ".join(said[-_SRUN_LINES:]) or "nothing"

    def report(self, timeout_sec: float) -> RoleReport:
        if self._report is not None:
            return self._report
        deadline = self._clock() + timeout_sec
        while True:
            report = self._load()
            if report is None and self.proc.poll() is not None:
                # The step may have written its report between the read above and its end.
                report = self._load()
                if report is None and self.proc.returncode == PROCESS_CEILING_EXIT:
                    raise RoleFailed(f"role {self.spec.name} refused to start: its uid already held as many tasks "
                                     f"as its process ceiling of {self.spec.nproc} on the node (status "
                                     f"{PROCESS_CEILING_EXIT})")
                if report is None:
                    raise RoleFailed(f"role {self.spec.name} ended with status {self.proc.returncode} "
                                     "before reporting its grant")
            if report is not None:
                self._report = report
                return report
            if self._clock() >= deadline:
                raise TimeoutError(f"role {self.spec.name} did not report its grant in {timeout_sec}s")
            self._sleep(_REPORT_POLL_SEC)

    def poll(self) -> int | None:
        return self.proc.poll()

    def wait(self, timeout_sec: float | None = None) -> int | None:
        try:
            return self.proc.wait(timeout_sec)
        except subprocess.TimeoutExpired:
            return None

    def _stop_step(self) -> str | None:
        """Begin a graceful stop, for ``SlurmLauncher.terminate``: mark a running role as
        stopped by the runner and return the step that scancel must signal. None when there is
        nothing to signal: the step has ended, or the role has not reported, in which case
        srun gets SIGTERM now."""
        if self.proc.poll() is not None:
            return None
        self.stopped_by_runner = True
        report = self._known_report()
        if report is None:
            # The wrapper has not started the role, and srun turns SIGTERM into a step-wide
            # SIGKILL, which is all there is to stop.
            self.proc.send_signal(signal.SIGTERM)
            return None
        return report.step

    def kill(self) -> None:
        if self.proc.poll() is None:
            self.stopped_by_runner = True
        report = self._known_report()
        if report is not None:
            kill_all(Path(report.cgroup_dir))
        if self.proc.poll() is not None:
            return
        # SIGTERM to srun is a step-wide SIGKILL; SIGKILL to srun is the last resort.
        self.proc.send_signal(signal.SIGTERM)
        deadline = self._clock() + _KILL_GRACE_SEC
        while self.proc.poll() is None and self._clock() < deadline:
            self._sleep(_KILL_POLL_SEC)
        if self.proc.poll() is None:
            self.proc.kill()

    def _step_state(self, step: str) -> JobState | None:
        """The step's state in accounting; None when sacct gave no answer for it yet."""
        result = self.commands("sacct", "-j", step, "--noheader", "--parsable2", "--format=State")
        if not result.ok:
            return None
        for line in result.stdout.splitlines():
            if line.strip():
                return normalize_state(line)
        return None

    def _ask_scheduler(self) -> tuple[bool, JobState | None]:
        """Whether the scheduler ended the step, and the step's terminal state when it was read."""
        code = self.proc.poll()
        if code is None:
            raise RuntimeError(f"role {self.spec.name} has not ended")
        if self.stopped_by_runner or code == 0:
            return False, None
        if code < 0:
            return True, None
        report = self._known_report()
        if report is None:
            return True, None
        deadline = self._clock() + _STATE_WAIT_SEC
        while True:
            state = self._step_state(report.step)
            if state is not None and state.terminal:
                return state not in _ROLE_STATES, state
            if self._clock() >= deadline:
                # srun ended but accounting does not say the step is over: make sure the role
                # cannot keep running.
                result = self.commands("scancel", report.step)
                if not result.ok:
                    log.warning("scancel %s did not succeed: %s", report.step, result.stderr.strip())
                return True, None
            self._sleep(_STATE_POLL_SEC)

    def _ended(self) -> tuple[bool, JobState | None]:
        if self._end is None:
            self._end = self._ask_scheduler()
        return self._end

    def ended_by_scheduler(self) -> bool:
        """Whether Slurm ended this step, or its end could not be confirmed, after srun
        returned a nonzero code the runner did not cause.

        A negative return code means srun itself was killed, and a step that never wrote a
        report has no step id to ask about: both count. Otherwise the step's sacct state
        decides, asked every 2 seconds for up to 30 until it is terminal; a state that never
        becomes terminal counts, and the step is then cancelled by its recorded id. The answer
        is kept. A role the runner stopped, and a clean exit, are never the scheduler's."""
        return self._ended()[0]

    def end_state(self) -> JobState | None:
        """The terminal sacct state that ``ended_by_scheduler`` read for the step; None when it
        asked nothing (a clean exit, a role the runner stopped, srun killed, no report) or the
        state never became terminal."""
        return self._ended()[1]

    def close_logs(self, timeout_sec: float) -> None:
        give_up = self._clock() + timeout_sec
        for capper in self.log_cappers:
            try:
                capper.wait(max(0.0, give_up - self._clock()))
            except subprocess.TimeoutExpired:
                log.warning("the log of role %s was not complete within %.0fs", self.spec.name, timeout_sec)

    def logs_lost(self) -> str | None:
        return lost(self.log_cappers)


class SlurmLauncher(Launcher):
    """Starts roles as steps of the allocation this process runs in."""

    def __init__(self, *, python: str, scratch: Path, allocation_gpus: int,
                 batch_env: Mapping[str, str] | None = None,
                 popen: Callable[..., subprocess.Popen] = subprocess.Popen,
                 allocation_gpu_uuids: Collection[str] | None = None, cgroup_root: Path = CGROUP_ROOT,
                 commands: Callable[..., CommandResult] | None = None, node: str | None = None,
                 node_root: Path = Path("/"), log_keep_bytes: int = LOG_KEEP_BYTES,
                 runtime: ApptainerRuntime | None = None, pasta: Path | None = None,
                 upstream_proxy: str | None = None) -> None:
        #: Interpreter that runs the step wrapper, the log cappers and a seal's interceptor.
        self.python = python
        #: Builds the command line of each container role.
        self.runtime = runtime if runtime is not None else ApptainerRuntime()
        #: The pasta program a seal runs, the site's; None for the one on the step's PATH.
        self.pasta = pasta
        #: The proxy the site's nodes reach the internet through, which a seal's interceptor
        #: fetches through; None for none.
        self.upstream_proxy = upstream_proxy
        #: Bytes every role's stdout and stderr each keep at their start and their end.
        self.log_keep_bytes = log_keep_bytes
        #: The job's node-local directory; each role gets a TMPDIR under it.
        self.scratch = scratch
        #: GPUs in the allocation, all of which a shared step is given.
        self.allocation_gpus = allocation_gpus
        #: The batch step's environment, which the step environments are built from.
        self.batch_env = dict(os.environ if batch_env is None else batch_env)
        #: UUIDs of the allocation's GPUs. When set, a role's report may list only these; when
        #: None, that check is skipped.
        self.allocation_gpu_uuids = allocation_gpu_uuids
        #: Where cgroup v2 is mounted.
        self.cgroup_root = cgroup_root
        #: Runs scancel and sacct for the roles' handles; by default each is given 10 seconds.
        self.commands = commands if commands is not None else SlurmCommands(timeout_sec=_STEP_TOOL_TIMEOUT_SEC)
        #: The node the roles run on, this host by default.
        self.node = node if node is not None else socket.gethostname()
        #: The root of the node's file system, under which the node shows its own size
        #: (``own_node_size``): the size of a node whose controller's record lacks a count.
        self.node_root = node_root
        self._popen = popen

    def node_size(self) -> NodeSize:
        """The node's GPUs and CPUs, as ``scontrol show node`` gives them (``node_size``), and each
        count the controller does not give as the node itself shows it (``own_node_size``)."""
        given = node_size(self.commands, self.node)
        if given.gpus is not None and given.cpus is not None:
            return given
        own = own_node_size(self.node_root)
        return NodeSize(gpus=own.gpus if given.gpus is None else given.gpus,
                        cpus=own.cpus if given.cpus is None else given.cpus)

    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        spec = step_spec(spec, runtime=self.runtime, python=self.python, pasta=self.pasta,
                         upstream_proxy=self.upstream_proxy)
        shared = None
        if share_gpus_of is not None:
            if spec.shape.gpus == 0:
                raise ValueError(f"role {spec.name} declares no GPUs, so it cannot share those of "
                                 f"{share_gpus_of.spec.name}")
            # The stage launches a role only once its target reported, so the report is there.
            target = share_gpus_of.report(0.0)
            if target.mismatches:
                raise ValueError(f"role {spec.name} cannot share the GPUs of {share_gpus_of.spec.name}: "
                                 f"its grant differs: {target.mismatches}")
            shared = target.gpu_uuids
            if len(shared) != spec.shape.gpus:
                raise ValueError(f"role {spec.name} declares {spec.shape.gpus} GPUs but would share "
                                 f"{len(shared)} with {share_gpus_of.spec.name}")
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        tmpdir = self.scratch / "roles" / spec.name / "tmp"
        _empty_tmpdir(tmpdir)
        report = spec.log_dir / f"{spec.name}.report.json"
        report.unlink(missing_ok=True)
        started_path(report).unlink(missing_ok=True)
        argv = srun_argv(spec, python=self.python, report=report, scratch=self.scratch, shared=shared,
                         allocation_gpus=self.allocation_gpus)
        env = step_env(self.batch_env, spec.env, tmpdir=tmpdir)
        logs = (spec.log_dir / f"{spec.name}.out", spec.log_dir / f"{spec.name}.err")
        # Every role's logs keep both ends of what it wrote: a contestant writes as much as it
        # likes, and a harness role such as the driver can log what the contestant's replies said.
        cappers: list[subprocess.Popen] = []
        writers: list[int] = []
        try:
            for path in logs:
                # The capper's own errors go to the role's node-local directory.
                errors = errors_file(self.scratch / "roles" / spec.name, path.suffix)
                writer, capper = spawn(path, keep=self.log_keep_bytes, python=self.python, errors=errors,
                                       popen=self._popen)
                writers.append(writer)
                cappers.append(capper)
            proc = self._popen(argv, env=env, stdout=writers[0], stderr=writers[1], stdin=subprocess.DEVNULL,
                               start_new_session=True)
        finally:
            for writer in writers:
                os.close(writer)
        return SlurmRoleHandle(spec, proc, report, job_id=self.batch_env.get("SLURM_JOB_ID"),
                               cgroup_root=self.cgroup_root, allocation_gpu_uuids=self.allocation_gpu_uuids,
                               commands=self.commands, log_cappers=cappers)

    def terminate(self, handles: Sequence[RoleHandle]) -> None:
        """Stop every one of *handles* with one ``scancel --signal=TERM`` naming all of their
        reported steps, so that a busy controller costs the stop one call's timeout rather
        than one per role. A role that has not reported has srun signalled instead, which
        Slurm turns into a step-wide SIGKILL; a role whose step has ended is left alone."""
        foreign = [h.spec.name for h in handles if not isinstance(h, SlurmRoleHandle)]
        if foreign:
            raise TypeError(f"roles {foreign} were not started by a Slurm launcher")
        steps = [h._stop_step() for h in handles if isinstance(h, SlurmRoleHandle)]
        _signal_steps(self.commands, [s for s in steps if s is not None])
