"""The interfaces between the pipeline and the cluster it runs on.

A ``Scheduler`` submits allocations and reports on them; the orchestrator uses it from a
login node. A ``Launcher`` starts a stage's roles as steps of the allocation the runner is
in. Slurm implements both (``hls.substrate.slurm``). Another substrate implements the same
two, and nothing above this package changes. A container role reaches the launcher as a
``ContainerRole`` (``hls.substrate.container``), which the launcher starts its own way."""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hls.spec.resources import GrantedResources, ResourceShape
from hls.substrate.container import ContainerRole

if TYPE_CHECKING:
    from hls.settings import SiteSettings
    from hls.substrate.factory import Placement


#: A node name the scheduler takes in ``--exclude``: no comma, bracket or space, which would make
#: it a list or a range.
_NODE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def check_node_name(name: str) -> str:
    """Return *name* if it is a node's name the scheduler takes in an exclusion list, else raise
    ValueError."""
    if not _NODE_NAME.fullmatch(name):
        raise ValueError(f"{name!r} is not a node name")
    return name


class JobState(StrEnum):
    """A job's state, normalized from what squeue and sacct print."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    NODE_FAIL = "NODE_FAIL"
    PREEMPTED = "PREEMPTED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    BOOT_FAIL = "BOOT_FAIL"
    DEADLINE = "DEADLINE"
    UNKNOWN = "UNKNOWN"

    @property
    def terminal(self) -> bool:
        """True once the job has ended. PENDING, RUNNING and UNKNOWN are not terminal: a
        job the scheduler could not report on counts as alive."""
        return self not in (JobState.PENDING, JobState.RUNNING, JobState.UNKNOWN)


@dataclass(frozen=True)
class JobInfo:
    """What the scheduler said about one job."""

    job_id: str
    #: Normalized state.
    state: JobState
    #: Why a pending job waits, such as "DependencyNeverSatisfied"; None otherwise.
    reason: str | None = None
    #: Exit code as sacct prints it ("1:0"); None while the job is queued or when unknown.
    exit_code: str | None = None
    #: When the job started, in POSIX seconds; None before it starts or when unknown.
    started_at: float | None = None


#: Seconds the two sources of the job's EndTime may differ by before the runner records both.
END_TIME_SOURCES_TOLERANCE_SEC = 60.0


@dataclass(frozen=True)
class JobEnd:
    """The job's EndTime as each of its two sources gave it when the runner started. The
    runner keeps to the controller's record of the job: ``SLURM_JOB_END_TIME`` can differ
    from it (a site's preemptible partition may set it from a shorter limit), and the job runs
    until the controller's EndTime. The variable stands in only when the controller does not
    answer."""

    #: EndTime from the controller's record of the job (``scontrol show job``); None when it
    #: did not answer or knows no end.
    controller: float | None
    #: ``SLURM_JOB_END_TIME``; None when it is unset or not integer seconds.
    env: float | None

    @property
    def end_time(self) -> float | None:
        """The EndTime the runner keeps to: the controller's, else the variable's; None when
        neither is known."""
        return self.controller if self.controller is not None else self.env

    @property
    def sources_differ(self) -> bool:
        """True when both are known and more than END_TIME_SOURCES_TOLERANCE_SEC apart."""
        return (self.controller is not None and self.env is not None
                and abs(self.controller - self.env) > END_TIME_SOURCES_TOLERANCE_SEC)


@dataclass(frozen=True)
class SubmitRequest:
    """One allocation to submit."""

    #: The batch script.
    script: Path
    #: The allocation's request, after the site's rules (``Allocation.request``).
    shape: ResourceShape
    #: The name the job runs under.
    job_name: str
    #: The directory the job's output goes to; the scheduler names the file in it.
    output_dir: Path
    #: Where the scheduler places the allocation (``Allocation.placement``).
    placement: Placement
    #: The settings of the site the allocation was planned for, whose rules the scheduler's
    #: arguments follow.
    site: SiteSettings
    #: Tag the job carries, unique to this submission, so the job can be found when the
    #: scheduler's answer to the submission is lost.
    tag: str
    #: Job id that must succeed before this one starts; None to start when scheduled.
    after: str | None = None
    #: Start only when no other job of the user with the same job name runs (Slurm's
    #: ``singleton`` dependency), so such jobs run one at a time.
    singleton: bool = False


class SchedulerError(Exception):
    """Base class for scheduler failures."""


class SchedulerUnavailable(SchedulerError):
    """The scheduler did not answer, so nothing can be concluded about any job."""


class SubmitUnknown(SchedulerError):
    """A submission's answer was lost: the job may or may not exist. Look it up by its tag
    before submitting again."""


class Refused(SchedulerError):
    """The scheduler answered and refused the request."""


class DependencyRefused(Refused):
    """sbatch or scontrol refused because the dependency is invalid, for example a purged
    predecessor id. The caller resubmits or rewires on its next pass."""


class NoQOSHolds(ValueError):
    """An allocation that asks for more time than any QOS of the site lets a job ask."""


class Scheduler(ABC):
    """Submits allocations and reports on them. Every job is named by the id recorded when
    it was submitted; nothing selects jobs by name or user."""

    @abstractmethod
    def submit(self, request: SubmitRequest) -> str:
        """Submit and return the job id, running exactly the command line ``submit_argv``
        names. Raise DependencyRefused when the dependency is invalid, Refused for any other
        refusal, and SubmitUnknown when the answer was lost."""

    @abstractmethod
    def submit_argv(self, request: SubmitRequest) -> tuple[str, ...]:
        """The command line ``submit`` runs for *request*, for the record of what the scheduler
        was given."""

    @abstractmethod
    def test_only(self, request: SubmitRequest) -> str:
        """Ask whether the request could run, without submitting it, and return the
        scheduler's answer. Raise Refused or SchedulerUnavailable."""

    @abstractmethod
    def find_by_tags(self, tags: Sequence[str]) -> dict[str, list[str] | None]:
        """For every tag in *tags*, keyed by tag: the ids of the queued, running or completing
        jobs (squeue's default states) with that tag, sorted numerically; an empty list when
        the controller answered with none; None for every tag when it did not answer. A job
        the controller has purged is dead and its attempt is in the store, so it reads as
        absent. One lookup answers them all."""

    @abstractmethod
    def states(self, job_ids: Sequence[str]) -> dict[str, JobInfo]:
        """An entry for every id in *job_ids*, UNKNOWN for any the scheduler could not
        report on."""

    @abstractmethod
    def rewire(self, job_id: str, after: str) -> None:
        """Make the pending job *job_id* wait for *after* instead of what it waits for.
        Raise DependencyRefused when *after* is no longer known, Refused for any other
        refusal, for example because the job no longer pends, and SchedulerUnavailable
        when the scheduler did not answer."""

    @abstractmethod
    def cancel(self, job_id: str) -> None:
        """Cancel the job *job_id*. A job that already ended or is gone counts as
        cancelled. Raise Refused when the scheduler refused, and SchedulerUnavailable when
        it did not answer."""


class FaultRecord(BaseModel):
    """Why a stage or an allocation failed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Who the failure is charged to. Infra faults are retried; a contestant fault is a
    #: scored result, never retried.
    fault_class: Literal["infra", "contestant"]
    #: One line for people.
    reason: str
    #: False when a retry would fail the same way, such as after the checkout changed. Always
    #: False for a contestant fault.
    retryable: bool
    #: Supporting facts, such as Slurm states, exit codes and memory samples.
    evidence: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _contestant_faults_are_final(self) -> Self:
        if self.fault_class == "contestant" and self.retryable:
            raise ValueError("a contestant fault is never retryable")
        return self


@dataclass(frozen=True)
class RoleSpec:
    """One role to start as its own step of the allocation."""

    #: Role name, unique within a stage attempt.
    name: str
    #: The command run inside the step, with absolute paths.
    argv: tuple[str, ...]
    #: The role's slice of the allocation; ``time_sec`` bounds the step.
    shape: ResourceShape
    #: Whose code the role runs, which decides who a failure is charged to.
    owner: Literal["harness", "contestant"]
    #: Directory for the role's stdout, stderr and grant report.
    log_dir: Path
    #: Declared non-secret environment, set on top of the launcher's clean base.
    env: Mapping[str, str] = field(default_factory=dict)
    #: Node-local env file (mode 0600) of the role's secret variables; None for none. The step
    #: wrapper unlinks it as soon as it has started (``hls.substrate.role_exec``).
    env_file: Path | None = None
    #: True to hand the env file's content to the command on descriptor 3 (``/dev/fd/3``), a pipe
    #: that reads end of file once the content has been read: how ``apptainer --env-file`` reads
    #: it. False to set the file's variables in the command's environment.
    env_fd: bool = False
    #: A line the command prints on stderr once it has really started, such as a container's
    #: start marker; None when it prints none. A nonzero exit with no such line in the role's
    #: stderr is the runtime's failure, not the command's.
    start_marker: str | None = None
    #: True runs the step with srun's ``--overlap``: it shares the CPUs of the stage's other
    #: steps instead of taking CPUs of its own, as a driver beside the server it drives or a small
    #: harness role beside a role that holds every CPU of the allocation. Its memory is still its
    #: own. Such a step takes no GPU, since ``--overlap`` shares the GPUs another step holds too.
    overlap: bool = False
    #: The ceiling on the processes and threads of the role's uid its step may run past, as its
    #: soft and hard ``RLIMIT_NPROC``, which everything the step starts inherits; None leaves the
    #: limit the step was given (``hls.substrate.nproc``).
    nproc: int | None = None
    #: True refuses the step when its CPUs lie on more than one NUMA node although one node
    #: holds as many (``hls.substrate.role_exec.placement``): a scored server on a site with
    #: ``site.evaluation_socket_tasks``.
    one_numa_node: bool = False
    #: The container the role runs, which the launcher turns into the step's command, its
    #: declared environment and its env file; None for a role that runs *argv*. A container
    #: role's spec has no *argv*, *env* or *env_file* of its own.
    container: ContainerRole | None = None
    #: True for a harness role that speaks to the node's container daemon: the one that pins an
    #: image and the one that builds one (``ImageRuntime.pin_role``, ``build_role``). A launcher
    #: whose runtime has a daemon gives such a role its socket; one without ignores it.
    daemon: bool = False

    def __post_init__(self) -> None:
        check_role_name(self.name)
        if self.daemon and self.owner != "harness":
            raise ValueError(f"role {self.name}: only a harness role speaks to the node's container daemon")
        if self.container is not None and (self.argv or self.env or self.env_file is not None or self.env_fd):
            raise ValueError(f"role {self.name}: a container role's command, environment and env file are the "
                             "launcher's")
        if self.env_fd and self.env_file is None:
            raise ValueError(f"role {self.name}: env_fd needs an env_file")
        if self.overlap and self.shape.gpus:
            raise ValueError(f"role {self.name}: an overlap step takes no GPU, since it would share another "
                             "step's GPUs")
        if self.nproc is not None and self.nproc < 1:
            raise ValueError(f"role {self.name}: a process ceiling of {self.nproc} lets the step start nothing")


#: A role name. It names the role's node-local directory, its log files and its step, so it
#: is one path segment that does not start with a dot or a dash.
ROLE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def check_role_name(name: str) -> str:
    """Return *name* if it is a role name, else raise ValueError."""
    if not ROLE_NAME.fullmatch(name):
        raise ValueError(f"role name {name!r} must match {ROLE_NAME.pattern}")
    return name


class RoleReport(BaseModel):
    """What a role's step was granted. The step wrapper writes it before the role starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Node the step runs on.
    host: str
    #: Slurm job and step id, "<job>.<step>".
    step: str
    #: Absolute path of the step's cgroup directory that carries its memory limit.
    cgroup_dir: str
    #: UUIDs of the GPUs the step's device cgroup admits, as nvidia-smi lists them.
    gpu_uuids: list[str]
    #: CUDA_VISIBLE_DEVICES the role starts with; None when unset.
    cuda_visible_devices: str | None
    #: The measured slice.
    granted: GrantedResources
    #: The step cgroup's swap limit in bytes; None when unlimited.
    swap_max_bytes: int | None
    #: Every reason the step must not start the role: each difference from the declared
    #: shape, and a command that is not an executable file. The role starts only if this is
    #: empty.
    mismatches: list[str]
    #: Names of the variables the wrapper removed from the role's step environment before the
    #: role started (``hls.substrate.role_exec.dropped``): from every role, what a site's prolog
    #: adds that would change what it runs, BASH_ENV and apptainer's settings and the variables
    #: it copies in (``APPTAINER_*``, ``SINGULARITY_*``, ``APPTAINERENV_*``, ``SINGULARITYENV_*``)
    #: but APPTAINER_CACHEDIR and APPTAINER_TMPDIR; from a container role, those two as well, and
    #: TERM and the proxy variables, which the container runtime passes into the container.
    stripped_env: list[str] = Field(default_factory=list)
    #: The CPUs the step may use, ascending.
    cpus: list[int] = Field(default_factory=list)
    #: The NUMA nodes those CPUs lie on, ascending.
    numa_nodes: list[int] = Field(default_factory=list)
    #: True when those CPUs lie on more than one NUMA node although one of them holds as many
    #: (``hls.substrate.role_exec.placement``).
    numa_split: bool = False


class RoleFailed(Exception):
    """A role's step ended before it reported its grant."""


class RoleHandle(ABC):
    """A started role."""

    spec: RoleSpec

    @abstractmethod
    def started(self) -> bool:
        """Whether the role's step has started, as far as this handle can tell now, without
        waiting: the scheduler granted the step and its first harness code ran, or the role
        reported its grant. A stage waits for a step to start before it waits for its report
        (``hls.pipeline.stage.STEP_START_TIMEOUT_SEC``), since a busy scheduler can take minutes
        to grant a step."""

    def launch_evidence(self) -> dict[str, str]:
        """What the scheduler says of the role's step, each fact under its name, for a role whose
        step did not start or did not report: asked once, before the role is killed. A launcher
        with no scheduler to ask has nothing to say."""
        return {}

    @abstractmethod
    def report(self, timeout_sec: float) -> RoleReport:
        """Wait for the step's grant report. Raise TimeoutError if none arrives in time,
        and RoleFailed if the step ended without one."""

    @abstractmethod
    def poll(self) -> int | None:
        """The step's exit status, or None while it runs."""

    @abstractmethod
    def wait(self, timeout_sec: float | None = None) -> int | None:
        """Wait for the step to end and return its exit status; None on timeout."""

    @abstractmethod
    def kill(self) -> None:
        """Stop the step at once: every process in its cgroup, then the launch process."""

    @abstractmethod
    def ended_by_scheduler(self) -> bool:
        """After ``wait()`` returned a nonzero code for a role the caller did not stop: True
        when the scheduler ended the step or its end could not be confirmed. A step that
        ended through its own exit, including by running out of memory or time, is False."""

    @abstractmethod
    def end_state(self) -> JobState | None:
        """After ``wait()`` returned a nonzero code for a role the caller did not stop: the
        step's terminal state in the scheduler's accounting, such as OUT_OF_MEMORY, from the
        same query as ``ended_by_scheduler``; None when there is no such answer. The step's
        cgroup is gone once the step ends, so this is the only record left of an OOM kill that
        came after the last memory sample."""

    def close_logs(self, timeout_sec: float) -> None:
        """Wait up to *timeout_sec* for the role's logs to be complete once its step ended: a
        capped log (``hls.substrate.log_cap``) gets its last bytes when its pipe ends. A role
        whose step writes its logs itself has nothing to wait for."""

    def logs_lost(self) -> str | None:
        """Why the role's logs were lost, once ``close_logs`` returned: a capper that ended
        nonzero or by a signal (``hls.substrate.log_cap.lost``); None otherwise."""
        return None


@dataclass(frozen=True)
class NodeSize:
    """The GPUs and CPUs of the node a job runs on, as the controller has them or the node itself
    shows them (``Launcher.node_size``); None for a count neither gave."""

    #: The node's GPUs.
    gpus: int | None
    #: The node's CPUs.
    cpus: int | None


#: Where the kernel lists the node's online CPUs, under the root of the node's file system:
#: every CPU of the node, whatever a job's affinity holds it to, as ``os.cpu_count`` counts them.
ONLINE_CPUS = Path("sys/devices/system/cpu/online")
#: Where the NVIDIA driver lists the node's GPUs, one entry each, under the root of the node's
#: file system: every GPU the driver sees, whatever a job's cgroup lets it open.
NVIDIA_GPUS = Path("proc/driver/nvidia/gpus")


def parse_cpulist(text: str) -> set[int]:
    """The CPUs of a kernel CPU list such as ``0-3,8,10-11``."""
    out: set[int] = set()
    for part in text.strip().split(","):
        if part:
            low, _, high = part.partition("-")
            out.update(range(int(low), int(high or low) + 1))
    return out


def own_node_size(root: Path) -> NodeSize:
    """The size of the node whose file system's root is *root*, as the node itself shows it: the
    CPUs its kernel has online (``ONLINE_CPUS``) and the GPUs its NVIDIA driver lists
    (``NVIDIA_GPUS``), none on a node without the driver. A count that cannot be read is None."""
    try:
        cpus: int | None = len(parse_cpulist((root / ONLINE_CPUS).read_text())) or None
    except (OSError, ValueError):
        cpus = None
    try:
        gpus: int | None = len(os.listdir(root / NVIDIA_GPUS))
    except FileNotFoundError:
        gpus = 0
    except OSError:
        gpus = None
    return NodeSize(gpus=gpus, cpus=cpus)


class Launcher(ABC):
    """Starts roles as steps of the current allocation."""

    #: The root of the file system of the node this launcher's roles run on, under which the node
    #: shows its own size (``own_node_size``).
    node_root: Path = Path("/")

    def node_size(self) -> NodeSize:
        """The size of the node this launcher's roles run on, as the node itself shows it
        (``own_node_size``); a launcher that can ask its controller asks it first."""
        return own_node_size(self.node_root)

    @abstractmethod
    def launch(self, spec: RoleSpec, *, share_gpus_of: RoleHandle | None = None) -> RoleHandle:
        """Start *spec* in a step of its own. With *share_gpus_of*, the step runs on that
        role's GPUs instead of its own; that is the one deliberate overlap, the reference
        service on the build's GPUs."""

    @abstractmethod
    def terminate(self, handles: Sequence[RoleHandle]) -> None:
        """Ask every one of *handles*, roles this launcher started, to stop with SIGTERM, so
        that each can shut down cleanly; a role whose step has ended is left alone. This is the
        one graceful stop: the roles share one grace period, which starts before this call, so
        it signals them all in one go, with one call of a slow tool at most.
        ``RoleHandle.kill`` stops one role at once."""
