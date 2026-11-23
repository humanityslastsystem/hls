"""The Stage base class, the registry of stage implementations, and the context a stage runs in.

A stage runs inside the runner, on the host side of its allocation, and stays light: it
starts roles through its context, waits for them, and names its outputs. Each output is a
file the stage or one of its roles wrote under the attempt's ``outputs/`` directory, with the
sha256 its writer gave: the stage the digest of the bytes it wrote (``write_output``), a role
its own report (``reported_output``). The runner records those digests and each output's size
in the manifest, reading none of the bytes, and publishes it; a stage never publishes itself. A
stage that cannot finish raises StageFault, whose record says whose fault it is.

No role outlives its stage. The runner calls ``run_stage``, which closes the context
however the stage ended: roles still running are stopped and reported, and an exception that
is not a StageFault becomes an infra fault. A stage that returns without waiting for roles
that already ended cannot publish over their faults: closing reports those roles, and a fault
among them is the stage's. A stage stops a role it is done with, such as a server once its
drive is over, through ``stop_role``: that planned stop is no fault (``planned_stop``).

A contestant's code runs only in a container, and each contestant container declares its
disk, which caps what its workdir and writable mounts hold: the memory watch walks them and
kills the role past the cap (``disk_limit``, the contestant's fault). An image that holds a
contestant's code (``PinnedImage.contestant``) runs only the contestant's own containers.

The context also carries what a stage may hand its roles: the values of the secrets granted
to it, which reach a host role only through a node-local 0600 env file the step wrapper
unlinks before the role starts and never reach a container; the container
runtime and the image runtime its launcher starts containers with; and the cache, whose
entries it copies to the node, each pinned to the digest its producer recorded, and records
for the manifest. A harness role of the cache's program makes and checks each copy
(``materialize``), so this process reads no byte of an entry, as no byte of an image either.
Image entries are never copied to the node. A
container runs only from an image this context pinned: an image file, read once by a harness
role of the image runtime into its own directory, where the role checks the copy's sha256 and
pins only the copy, keyed by the file's digest, and removed when the stage's context closes
(``node_image``, ``pin_file``, ``close``), or before, when the stage releases it
(``release``). No stage reads another's image: each pins the images it runs itself. An image a
stage could not remove is reused by the next only after the image runtime checks it.
That check assumes no contestant code runs as a host process, which could rewrite a tree and
its manifest together: a contestant's code runs only in a container (``start_role`` refuses a
contestant host role but in the no-op stage, which pins no image).

A contestant container whose step also holds harness processes, the build sandbox, is started
with a harness reserve. The context then samples its step's memory split by owner, a memory
breach where the runner enforces memory ends the contestant's largest process rather than the
step, and the role's outcome carries the evidence the build rule reads (``classify_build_end``),
so ``stage_fault`` and ``contestant_fault`` read its end by that rule."""

from __future__ import annotations

import inspect
import logging
import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from hls.agent.memory import ContestantKiller, ReadyPids, SplitSample, SplitSampler, harness_pids, split_evidence
from hls.pipeline.events import EventLog
from hls.pipeline.plan import Plan, StagePlan
from hls.provision.mounts import contestant_mount_problem
from hls.settings import RetryPolicy, RunSettings, Settings, SiteSettings
from hls.spec.resources import ResourceShape
from hls.store.cache import (CacheCorrupt, CacheEntry, ContentCache, Populated, WaitEnded, copies_root, entry_sha256,
                             read_digests)
from hls.store.digest import DIGEST_MISMATCH_EXIT, check_sha256_hex, sha256_bytes
from hls.store.lock import Holder
from hls.store.manifest import Manifest, OutputRef, output_ref, read_outputs_report
from hls.store.posix import PosixStore
from hls.substrate import log_cap, memory_watch
from hls.substrate.base import (FaultRecord, JobState, Launcher, RoleFailed, RoleHandle, RoleReport, RoleSpec,
                                check_role_name)
from hls.substrate.cgroup import pids_limit_hits, read_keyed
from hls.substrate.container import (ContainerRole, ContainerRuntime, ContainerSpec, ImageRuntime, Mount, PinnedImage,
                                     SealPlan)
from hls.substrate.faults import (RoleOutcome, SplitEvidence, classify_role, infra_fault, starvation_fault,
                                  within_evidence_age)
from hls.substrate.memory_watch import JobOoms, MemorySample, MemoryWatch, PidsHits, pids_read
from hls.substrate.nproc import (AccountingChoice, CeilingRule, PoolAtEnd, PoolPeak, PoolSample, ProcessCeiling,
                                 choose_accounting, hard_limit, kernel_release, node_pid_max, node_tasks, uid_tasks)
from hls.substrate.nproc import node_evaluations
from hls.substrate.nproc import process_ceiling as ceiling_for
from hls.substrate.ucounts import ServedUcounts, UcountLimits, node_limits, node_shortfall, served_limits, ucount_limits
from hls.substrate.seal import IndexStatus, SealFiles, client_env, index_status, payload_ending, seal_files
from hls.substrate.secrets import Scrubber, write_env_file
from hls.util import remove_tree

log = logging.getLogger(__name__)

_GIB = 2**30
#: The cache's role program, run as ``<plan python> -m CACHE_ROLE_MODULE copy|verify|fill ...``.
CACHE_ROLE_MODULE = "hls.store.cache_role"
# The file in a role's directory where it reports the digests of the outputs it writes.
_OUTPUTS_REPORT = "outputs.json"
#: Bytes at the start of a role's stderr searched for its start marker. The marker comes
#: before anything the command prints, after at most the runtime's own warnings.
MARKER_SEARCH_BYTES = 2**20
# Names of the env files a role's directory can hold: one for a host role, one for a container.
_ROLE_ENV, _CONTAINER_ENV = "role.env", "container.env"
#: Longest wait for a role's step to start, from the request to launch it until its step wrapper
#: says it started or the role reports its grant (``RoleHandle.started``): the scheduler's time
#: to grant the step, which the stage's deadline grows by (``StageContext.deadline``) and only the
#: allocation's end cuts short (``StageContext.deadline_cap``). It is a bound on cost, not a fit
#: to data: half an hour is what a step that cannot start may cost an allocation, per attempt.
#: The ``launch_wait_sec`` that each ``role_started`` event records is what checks it.
STEP_START_TIMEOUT_SEC = 1800.0
#: Seconds before the job's EndTime that no stage's deadline passes (the runner gives each
#: stage the EndTime less this as ``StageContext.deadline_cap``), however long its roles' steps
#: took to start, and that a stage must leave before it starts.
#: A stage stopped at its deadline then stops its roles one after another, each up to 5 s for
#: the kill, 10 s for its step to end and ``LOG_CLOSE_SEC`` (30 s) for its logs, 90 s for the
#: drive's artifact and driver; the context removes the roles' directories and extracted images,
#: a contestant's filled workdir and engine cache among them; and the runner writes its records.
#: Past this time Slurm's SIGTERM at EndTime makes the runner record the allocation terminated,
#: which is infra, so nothing is charged to the contestant.
END_MARGIN_SEC = 300.0
#: Longest wait for a role step's grant report, from the step's start, cut short by the stage's
#: deadline. The step wrapper reports about a second after it starts.
REPORT_TIMEOUT_SEC = 300.0
# Files a role's step writes into its log directory, beside the role's name: its stdout and
# stderr, its start marker (``hls.substrate.role_exec.started_path``) and its grant report.
_LOG_FILES = (".out", ".err", ".started", ".report.json")
#: Seconds the roles of a stage get between SIGTERM and SIGKILL when the stage stops them.
#: All the roles share one grace period.
STOP_GRACE_SEC = 10.0
#: Seconds a reported role's capped logs are waited for to be complete.
LOG_CLOSE_SEC = 30.0
# Longest pause between looks at the roles while ``wait_roles`` waits.
_POLL_SEC = 0.2
# Bytes at the end of a role's stderr searched for its refusal line.
_REFUSAL_TAIL_BYTES = 65536
# Longest single wait for a role's grant report, so that a stop, and the start of the role's step, are
# noticed within this long.
_REPORT_SLICE_SEC = 1.0
# The directory of a numbered step under a Slurm job's cgroup.
_STEP_DIR = re.compile(r"step_[0-9]+")


def container_strings(spec: ContainerSpec) -> list[str]:
    """Every string of *spec* a runtime may pass where any user of the node can read it: the
    command, the image, the paths and the seal's policy."""
    strings = [*spec.command, spec.image.ref, str(spec.workdir), *([] if spec.cwd is None else [spec.cwd])]
    for m in spec.mounts:
        strings += [str(m.source), m.target]
    if spec.seal is not None:
        strings.append(spec.seal.policy.model_dump_json())
    return strings


@dataclass(frozen=True)
class Refusal:
    """What a harness role refused, as it told it: its exit status and the reason on its last
    ``refused:`` line on stderr (``StageContext.refusal``)."""

    #: The role's exit status, which says what kind of refusal it is.
    exit_code: int
    #: The reason it gave, scrubbed.
    reason: str


class StageFault(Exception):
    """A stage failed. Its record says whose fault: infra for the harness or the cluster,
    contestant for the contestant's code (a contestant role that ran out of memory is a
    StageFault too)."""

    def __init__(self, fault: FaultRecord) -> None:
        super().__init__(fault.reason)
        #: The fault, as the runner records it.
        self.fault = fault

    @classmethod
    def infra(cls, reason: str, *, retryable: bool, **evidence: str) -> StageFault:
        """An infra fault with *reason* for people and *evidence* as its supporting facts,
        retried when *retryable*. Every caller says whether a retry could help, since a failure
        that would repeat is only delayed by one."""
        return cls(infra_fault(reason, retryable=retryable, **evidence))


class Stage(ABC):
    """A stage implementation: the class every stage kind inherits and ``register`` adds.

    A stage writes diagnostics through ``logging``, never ``print``: the runner's log handler
    replaces every value of the secrets file in each record, and output that bypasses it reaches
    the job's log in the store as written.

    Each class attribute below is a trait of the kind, declared here once with its type and, for
    an optional one, its default. ``register`` checks every trait's value and refuses a class that
    leaves out a required one or sets a public attribute this class does not declare, so a
    misspelled trait fails when its module is imported instead of taking the default.

    Several traits tell finalize's secret scan (``hls.stages.finalize``) which of a stage's files
    hold bytes a contestant chose. A secret's value found anywhere else, in what the harness
    alone wrote, is the harness's own leak and refuses the instance. An output is named by a
    prefix of its "/"-separated path under its attempt's ``outputs/``; one that ends with "/"
    names a directory. Every role's logs are read as stored."""

    #: Registry name, the plan's ``StagePlan.kind``.
    kind: ClassVar[str]
    #: Model the stage's parameters are checked against when the plan is made.
    params_model: ClassVar[type[BaseModel]]
    #: Files of an attempt, by their path under the attempt's directory (a directory's ending with
    #: "/"), whose bytes a contestant chose and a compression hides from a scan as stored, such as
    #: a source bundle or a submission image: finalize keeps each out of any export unless a
    #: published package scan read what it holds. () for none. Required, with no default: a kind
    #: that left it out would have its containers exported unscanned.
    compressed_files: ClassVar[tuple[str, ...]]
    #: Directories of an attempt, by their path under the attempt's directory ending with "/",
    #: where every file the stage's published manifest does not hold is such a container, as what
    #: an agent build's killed commit point leaves in its outputs. () for none. Required.
    unpublished_containers: ClassVar[tuple[str, ...]]
    #: The directories under ``unpublished_containers`` a published manifest holds whole, through
    #: the one output it names in each, such as the transcript through its index. () for none.
    #: Required.
    published_dirs: ClassVar[tuple[str, ...]]
    #: Files of an attempt, by their path as ``compressed_files`` names them and each among them,
    #: that finalize's scan does not read: a container as large as a build's writable disk cap,
    #: which a package scan reads inside or finalize quarantines whole
    #: (``hls.stages.finalize.unscanned_containers``), so no byte of it goes out unread. The scan
    #: still searches their paths. () for none.
    unread_files: ClassVar[tuple[str, ...]] = ()
    #: Whether the stage must see every value of the site's secrets file, granted or not, as a
    #: stage that scrubs records or scans for secrets must. The runner then loads the file for
    #: the stage even when no stage of the allocation has grants, and puts every value in
    #: ``StageContext.secrets`` under its secret's name. Otherwise a stage sees only its grants.
    needs_secret_registry: ClassVar[bool] = False
    #: Whether the stage hands its grants to the contestant's code it runs, as the agent build
    #: hands its route to the agent. A stage's grants otherwise reach its harness roles only and
    #: are refused in a container. Only a secret granted to such a stage is one a contestant
    #: could hold, and only such a secret is ever quarantined; any other is the harness's leak
    #: wherever it is found.
    grants_reach_contestant: ClassVar[bool] = False
    #: Whether the stage starts a role whose code is the contestant's. Every file of such a
    #: stage's attempts holds a contestant's bytes, its harness's logs and records too, which echo
    #: the names and bytes the contestant chose: a secret's value there is quarantined with the
    #: instance still scored.
    runs_contestant_code: ClassVar[bool] = False
    #: The outputs whose bytes a contestant or an agent wrote, such as a submission image or an
    #: agent's transcript: read as stored, never decompressed, and quarantined as a contestant
    #: role's files are.
    raw_outputs: ClassVar[tuple[str, ...]] = ()
    #: For a stage that runs no contestant code, such as verify, the outputs the harness wrote
    #: around a contestant's bytes, such as a verdict quoting the server's replies: read
    #: decompressed when the harness compressed them, and quarantined in the same way.
    contestant_outputs: ClassVar[tuple[str, ...]] = ()
    #: For a stage that runs no contestant code, its harness roles whose logs echo a contestant's
    #: bytes: their logs count as a contestant role's.
    contestant_log_roles: ClassVar[tuple[str, ...]] = ()
    #: Whether a contestant's role may run as a host process rather than in a container
    #: (``StageContext.start_role``). Only the no-op stage, which pins no image, sets it.
    contestant_host_roles: ClassVar[bool] = False
    #: Whether a contestant role's fault leaves the stage to decide its end, as in the agent
    #: build, which collects the commit the agent left after its sandbox ended on the contestant's
    #: fault. Its roles then say nothing of the stage's decision: the stage declares the
    #: contestant fault it ends with (``StageContext.declare_contestant_fault``) just before it
    #: raises it, and ``StageContext.contestant_fault`` reads only that.
    decides_contestant_faults: ClassVar[bool] = False
    #: Retries after an infra fault when ``[run.retry.<stage>]`` does not set ``max``
    #: (``retry_policy``).
    default_retries: ClassVar[int] = 2
    #: How the stage's grants are given: None when the site's secrets file grants them, else a
    #: classmethod ``declared_grants(params, settings) -> dict[str, str]`` naming the secret each
    #: variable holds, from the stage's parameters and the settings, which the planner checks
    #: against the secrets file (``hls.pipeline.planner``); it raises ValueError for parameters it
    #: cannot grant.
    declared_grants: ClassVar[Callable[[Any, Settings], dict[str, str]] | None] = None

    @classmethod
    @abstractmethod
    def check(cls, shape: ResourceShape, params: Any, *, runner_memory_gib: int) -> None:
        """Raise ValueError if *params* cannot run in *shape*, whose roles must leave
        *runner_memory_gib* to the runner (``budget_problem``). Called when the plan is made."""

    @classmethod
    def check_site(cls, shape: ResourceShape, params: Any, *, site: SiteSettings) -> None:
        """Raise ValueError if *params* cannot run in *shape* on *site*, for what its substrate
        decides (``hls.substrate.factory.substrate``). Called when the plan is made, after
        ``check``; a stage whose parameters no substrate decides checks nothing here."""

    @classmethod
    @abstractmethod
    def max_roles(cls, params: Any) -> int:
        """The most roles one attempt with *params* starts, each leaving its two capped logs. The
        planner sizes by it the headroom an allocation keeps for its roles' steps to start."""

    @classmethod
    def seals_containers(cls, params: Any) -> bool:
        """Whether a stage with *params* starts a container under the network seal
        (``ContainerSpec.seal``), which needs pasta, the firewall tools and unprivileged user
        namespaces on its node. The runner checks them before any stage of an allocation with
        such a stage runs (``hls.substrate.site_checks.seal_problems``)."""
        return False

    @classmethod
    def scored_server_cpus(cls, params: Any) -> int | None:
        """The CPUs of the scored server a stage with *params* serves, which a site with
        ``site.evaluation_socket_tasks`` asks for as one of two tasks with one per socket
        (``hls.pipeline.planner.socket_tasks``); None for a stage that serves none."""
        return None

    @classmethod
    def sandbox_stores(cls, params: Any) -> tuple[str, ...]:
        """The site's sandbox stores (``Settings.sandbox_stores``) a stage with *params* binds
        into its roles. A plan is refused when its site does not name each of them
        (``hls.pipeline.planner``). Empty for a stage that binds none."""
        return ()

    @abstractmethod
    def run(self, ctx: StageContext) -> dict[str, str]:
        """Do the stage's work and return each output's name and store key. Once a role has
        ended with a contestant fault, the stage raises the ``stage_fault`` of its roles and
        never an infra fault of its own: the runner's records of a SIGTERM read the stage's
        decision from its roles (``StageContext.contestant_fault``). A stage that decides its
        contestant faults itself declares the one it raises instead."""


#: Every stage implementation, by kind.
STAGES: dict[str, type[Stage]] = {}


def _paths(cls: type, value: object) -> bool:
    """Whether *value* is a tuple of strings; a bare string, read as its characters, is not."""
    return isinstance(value, tuple) and all(isinstance(p, str) for p in value)


def _compressed(cls: type, value: object) -> bool:
    """Whether *value* is a tuple of paths each among *cls*'s own ``compressed_files``: finalize
    quarantines only those whole when no scan read inside them."""
    declared = getattr(cls, "compressed_files", None)
    return _paths(cls, value) and _paths(cls, declared) and set(value) <= set(declared)


def _flag(cls: type, value: object) -> bool:
    return isinstance(value, bool)


#: Each trait of ``Stage`` held as data, with what ``register`` requires its value to be: a check
#: of the value, given the class too, so that a trait whose entries must each be another trait's
#: can be checked against it.
_DATA_TRAITS: dict[str, tuple[str, Callable[[type, object], bool]]] = {
    "kind": ("a non-empty name", lambda cls, v: isinstance(v, str) and bool(v)),
    "params_model": ("a pydantic model class", lambda cls, v: isinstance(v, type) and issubclass(v, BaseModel)),
    "compressed_files": ("a tuple of paths, () for none", _paths),
    "unpublished_containers": ("a tuple of paths, () for none", _paths),
    "published_dirs": ("a tuple of paths, () for none", _paths),
    "unread_files": ("a tuple of compressed_files entries, () for none", _compressed),
    "needs_secret_registry": ("a bool", _flag),
    "grants_reach_contestant": ("a bool", _flag),
    "runs_contestant_code": ("a bool", _flag),
    "raw_outputs": ("a tuple of output prefixes, () for none", _paths),
    "contestant_outputs": ("a tuple of output prefixes, () for none", _paths),
    "contestant_log_roles": ("a tuple of role names, () for none", _paths),
    "contestant_host_roles": ("a bool", _flag),
    "decides_contestant_faults": ("a bool", _flag),
    "default_retries": ("a count of retries", lambda cls, v: type(v) is int and v >= 0),
}
#: The traits of ``Stage`` given as classmethods.
_CLASS_METHODS = ("check", "check_site", "max_roles", "scored_server_cpus", "sandbox_stores", "seals_containers")
#: Every public name a stage class may set.
_TRAITS = frozenset((*_DATA_TRAITS, *_CLASS_METHODS, "declared_grants", "run"))


def _trait_problems(cls: type) -> list[str]:
    """What is wrong with stage class *cls*'s traits (``Stage``), each problem as text; none when
    every trait is declared as ``Stage`` says."""
    if not issubclass(cls, Stage):
        return ["it does not inherit hls.pipeline.stage.Stage"]
    problems = [f"{name} must be {what}" for name, (what, valid) in _DATA_TRAITS.items()
                if not valid(cls, getattr(cls, name, None))]
    missing = sorted(cls.__abstractmethods__)
    if missing:
        problems.append(f"it does not define {', '.join(missing)}")
    problems += [f"{name} must be a classmethod" for name in _CLASS_METHODS
                 if not isinstance(inspect.getattr_static(cls, name), classmethod)]
    grants = inspect.getattr_static(cls, "declared_grants")
    if grants is not None and not isinstance(grants, classmethod):
        problems.append("declared_grants must be None or a classmethod")
    unknown = sorted({name for klass in cls.__mro__ if klass is not Stage and issubclass(klass, Stage)
                      for name in vars(klass) if not name.startswith("_") and name not in _TRAITS})
    if unknown:
        problems.append(f"no stage trait is named {' or '.join(unknown)}; the traits are {', '.join(sorted(_TRAITS))}")
    return problems


def register(cls: type[Stage]) -> type[Stage]:
    """Class decorator that adds a stage implementation to ``STAGES``. TypeError for a class
    whose traits are not as ``Stage`` declares them: one that does not inherit it, leaves out a
    required trait or method, gives a trait a value of another type, or sets a public attribute
    that is no trait, such as a misspelled one, which would otherwise leave the trait at its
    default."""
    problems = _trait_problems(cls)
    if problems:
        raise TypeError(f"stage class {cls.__name__} (kind {getattr(cls, 'kind', None)!r}): {'; '.join(problems)}")
    if cls.kind in STAGES and STAGES[cls.kind] is not cls:
        raise ValueError(f"two stage implementations named {cls.kind}")
    STAGES[cls.kind] = cls
    return cls


def stage_class(kind: str) -> type[Stage]:
    """The implementation registered as *kind*; ValueError if there is none."""
    if kind not in STAGES:
        raise ValueError(f"unknown stage kind {kind!r}; known kinds are {sorted(STAGES)}")
    return STAGES[kind]


def most_roles(kind: str, params: Mapping[str, Any]) -> int:
    """The most roles one attempt of a stage of *kind* with *params* starts (``Stage.max_roles``).
    Finalize sizes its scan by it, and the planner each allocation's headroom for its roles' steps
    to start. ValueError for a kind no imported module registers."""
    cls = stage_class(kind)
    return cls.max_roles(cls.params_model.model_validate(dict(params)))


def seals_containers(kind: str, params: Mapping[str, Any]) -> bool:
    """Whether a stage of *kind* with *params* starts a sealed container
    (``Stage.seals_containers``). ValueError for a kind no imported module registers, or for
    *params* its model refuses."""
    cls = stage_class(kind)
    return cls.seals_containers(cls.params_model.model_validate(dict(params)))


def retry_policy(run: RunSettings, name: str, kind: str) -> RetryPolicy:
    """How stage *name* of *kind* is retried: its ``[run.retry.<name>]`` table, with the kind's
    ``Stage.default_retries`` where the table does not set ``max``. ValueError for a kind no
    imported module registers."""
    return run.retry_for(name, default_max=stage_class(kind).default_retries)


def _with_seal(shape: ResourceShape, seal_memory_gib: int) -> ResourceShape:
    """*shape* as the stage's budget counts it: with the *seal_memory_gib* its seal holds beside it."""
    if not seal_memory_gib:
        return shape
    return shape.model_copy(update={"memory_gib": shape.memory_gib + seal_memory_gib})


def budget_problem(stage: ResourceShape, roles: Iterable[tuple[ResourceShape, bool]], *,
                   runner_memory_gib: int, oversubscribed: Iterable[ResourceShape] = ()) -> str | None:
    """Why *roles*, running together, do not fit in *stage*; None when they fit. Each role is
    its shape and whether it shares the GPUs of another role. Every role's GPUs must be of the
    stage's kind. GPUs and CPUs are counted for exclusive roles only, since a shared role runs
    on GPUs and cores an exclusive role holds; memory is counted for every role, and the roles
    leave *runner_memory_gib* of the stage's memory to the runner (``run.runner_memory_gib``).
    The *oversubscribed* roles share the memory of another as well (``StageContext.start_container``'s
    *share_memory_of*): nothing of theirs is counted, and each must fit the stage's memory by
    itself beside the runner, since its step's limit is its own memory and the job's limit bounds
    it with every other role. The plan checks a stage's roles with this at plan time, and the
    context at each role start."""
    roles = list(roles)
    oversubscribed = list(oversubscribed)
    for shape in [s for s, _ in roles] + oversubscribed:
        if shape.gpus and shape.gpu_kind != stage.gpu_kind:
            return f"a role asks for gpu kind {shape.gpu_kind}; the stage's is {stage.gpu_kind}"
    exclusive = [shape for shape, shared in roles if not shared]
    problems = [f"the roles need {want} {unit}; the stage declares {have}"
                for unit, want, have in (("GPUs", sum(s.gpus for s in exclusive), stage.gpus),
                                         ("CPUs", sum(s.cpus for s in exclusive), stage.cpus))
                if want > have]
    memory = sum(s.memory_gib for s, _ in roles)
    if memory + runner_memory_gib > stage.memory_gib:
        problems.append(f"the roles need {memory} GiB and the runner {runner_memory_gib} GiB; the stage declares "
                        f"{stage.memory_gib}")
    problems += [f"a role that shares another's memory asks for {s.memory_gib} GiB, and with the runner's "
                 f"{runner_memory_gib} GiB that is more than the stage's {stage.memory_gib}"
                 for s in oversubscribed if s.memory_gib + runner_memory_gib > stage.memory_gib]
    return "; ".join(problems) or None


def stage_fault(outcomes: Sequence[RoleOutcome]) -> FaultRecord | None:
    """The fault a stage ends with, decided from how its roles ended; None when no role
    faulted. It is infra if any role's fault is infra, retryable only if every infra fault
    is, and the contestant's only when every fault is: a contestant role at its step's pids
    limit (``contestant_pids``) beside a role that failed as infra leaves the stage infra. A role
    the stage stopped has no fault of its own: its stop counts only when no other role faulted.
    The evidence names each faulty role and carries that role's evidence under
    ``<role>.<key>``."""
    faults = [(o, f) for o in outcomes if (f := classify_role(o)) is not None]
    own = [(o, f) for o, f in faults if o.ending != "stopped"]
    decisive = own or faults
    if not decisive:
        return None
    infra = [f for _, f in decisive if f.fault_class == "infra"]
    evidence = {"roles": ",".join(o.name for o, _ in decisive)}
    for o, f in decisive:
        evidence.update({f"{o.name}.{key}": value for key, value in f.evidence.items()})
    return FaultRecord(fault_class="infra" if infra else "contestant", reason="; ".join(f.reason for _, f in decisive),
                       retryable=all(f.retryable for f in infra or [f for _, f in decisive]), evidence=evidence)


def _counted(at_end: PoolAtEnd | None, field: Literal["others_tasks", "node_tasks"]) -> int | str | None:
    """A count of the pool at a role's end for its ``role_ended`` event: None for a role with no
    pool, ``uncounted`` for a count that failed, which leaves starvation not ruled out."""
    if at_end is None:
        return None
    tasks = getattr(at_end, field)
    return "uncounted" if tasks is None else tasks


def scored_end_fault(outcome: RoleOutcome) -> FaultRecord | None:
    """The fault of a contestant role's failing end that the stage is about to score against the
    contestant: the roles' own (``stage_fault``), else that the uid's other tasks or others' files
    on the node's disk starved it, or could not be counted once its step was gone
    (``starvation_fault``), which ``classify_role``
    asks only of a role that exited with a status other than 0, never of one that exited 0, that
    the stage stopped or that it cut at a time limit. Every stage asks this before it scores such
    an end as a failure, so an end starvation may have caused is infra, retried, and never the
    contestant's."""
    return stage_fault([outcome]) or starvation_fault(outcome)


@dataclass(frozen=True)
class _AtStop:
    """What was known of a role the stage stopped as planned, read before the stop's SIGTERM:
    what it does once asked to stop does not count."""

    #: When the stop was asked for, on the stage's clock: the end its pids limit is read against.
    at: float
    #: The memory watch's breach of its step; None for none.
    breach: MemorySample | None
    #: How often its step reached its pids limit and since when (``StageContext._pids_hits``);
    #: None when no one read it.
    pids_hits: PidsHits | None
    #: The OOM kills in its step (``StageContext._oom_kills``).
    oom_kills: int


@dataclass
class _Launch:
    """The times of one role's launch on the stage's clock, which its two waits count from
    (``StageContext._await_report``)."""

    #: When the stage asked the launcher to start the role.
    asked: float
    #: When the stage first saw the role's step started; None until then.
    started: float | None = None

    def waits(self, now: float) -> dict[str, float | None]:
        """The role's two waits in seconds, at *now*, as its events record them:
        ``launch_wait_sec``, from the launch request to the step's start, or to *now* for a
        step not seen started; ``report_wait_sec``, from the step's start to *now*, or None for
        a step not seen started."""
        if self.started is None:
            return {"launch_wait_sec": round(now - self.asked, 3), "report_wait_sec": None}
        return {"launch_wait_sec": round(self.started - self.asked, 3),
                "report_wait_sec": round(now - self.started, 3)}

    def too_late(self, name: str, gave_up: float, *, cut: str | None) -> str:
        """Why role *name* was given up on at *gave_up*, the end of the wait it was in: its step
        did not start, or did not report once started, naming what *cut* the wait short, if
        anything did."""
        cut = "" if cut is None else f" ({cut} came first)"
        if self.started is None:
            return f"role {name}'s step did not start within {gave_up - self.asked:.0f} s of its launch{cut}"
        return (f"role {name}'s step started {self.started - self.asked:.0f} s after its launch but did not report "
                f"its grant within {gave_up - self.started:.0f} s of its start{cut}")


class StageContext:
    """What one stage attempt can use: its plan entry, its attempt directory, its roles.

    A stage's methods run on the stage's thread, and the runner's thread reads only what says
    another thread may read it (``contestant_fault``, ``no_retry_fault``). One call may run off the
    stage's thread: ``start_container`` of a harness role, at most one at a time, which
    agent_build's drive runs in a worker thread for each snapshot, so the scheduler's grant of
    its step never stalls the agent's I/O or the hard-limit check
    (``hls.stages.agent_build._Build._launch_snapshot``). While it runs, the stage's thread may
    read ``stopping``, ``deadline`` and ``clock``, a role's ``report``, ``seal_status`` and
    ``sample_split``, wait for or stop other roles by name (``wait_roles`` with names,
    ``stop_role``), emit events and scrub: the role table may grow under these, and every walk of
    it reads a copy. It must start no other role, pin no image, copy no cache entry, hold no
    deadline (``hold_until``) and close nothing until it has the launch's result: two admissions
    could each miss the other's resources, and the launch's own deadline growth (``_grow``) would
    race a hold. A role still launching has no grant report, so a role reported meanwhile cannot
    rule out its step in the job's count of OOM kills (``_own_oom`` reads None, the harness's
    side)."""

    def __init__(self, *, store: PosixStore, plan: Plan, stage: StagePlan, attempt: int, launcher: Launcher,
                 watch: MemoryWatch | None, events: EventLog, inputs: dict[str, Manifest], deadline: float,
                 clock: Callable[[], float] = time.time, secrets: Mapping[str, str] | None = None,
                 scrubber: Scrubber | None = None, cache: ContentCache | None = None,
                 scratch: Path | None = None, job_id: str = "local",
                 runtime: ContainerRuntime | None = None, images: ImageRuntime | None = None,
                 secret_registry: bool = False,
                 deadline_cap: float | None = None,
                 on_no_retry: Callable[[FaultRecord], None] | None = None) -> None:
        self.store = store
        #: The plan revision the runner read.
        self.plan = plan
        #: This stage's entry in it.
        self.stage = stage
        self.attempt = attempt
        #: Starts role steps.
        self.launcher = launcher
        #: The runner's memory watch; None in tests that do not sample.
        self.watch = watch
        #: The attempt's event log.
        self.events = events
        #: Manifests the stage reads, by manifest key.
        self.inputs = inputs
        #: POSIX time by which the stage must end; roles still running then are killed. It
        #: grows by each role's wait for its step to start (``launch_waits``), up to
        #: ``deadline_cap`` and never down, so the scheduler's time comes out of no role's own:
        #: the planned start of the stage, plus its time, plus its roles' launch waits.
        self.deadline = deadline
        #: POSIX time the deadline never grows past, and at which a wait for a step to start
        #: ends: the allocation's end, less the time the runner keeps to stop the stage and write
        #: its records (``END_MARGIN_SEC``). None when there is no end to keep to.
        self.deadline_cap = deadline_cap
        self.clock = clock
        #: Set by the runner when the allocation is shutting down.
        self.stopping = False
        #: Values of the secrets granted to this stage, by the variable each is delivered as;
        #: for a stage that needs the registry, also every value of the file by its name.
        self.secrets: Mapping[str, str] = dict(secrets or {})
        #: Replaces every value of the secrets file in text the stage writes to a record, and
        #: in the logs of a role given secrets; by default, the values of this stage's own.
        self.scrubber = scrubber if scrubber is not None else Scrubber(self.secrets)
        #: The store's content-addressed cache; None where a stage needs none.
        self.cache = cache
        #: The runner's node-local directory for this job, holding each role's directory and
        #: the node-local copies of cache entries; None where a stage needs none.
        self.scratch = scratch
        #: The Slurm job id, which keys the node-local copies of cache entries.
        self.job_id = job_id
        #: The runtime the launcher starts containers with; None where a stage starts none.
        self.runtime = runtime
        #: Pins images on the node for containers to run from; None where a stage pins none.
        self.images = images
        #: True when the runner loaded the secrets file for this stage and put every value of it
        #: in ``secrets`` (``Stage.needs_secret_registry``); a file with no secret leaves that empty.
        self.secret_registry = secret_registry
        #: Writes a fault ``declare_no_retry`` records into the attempt's status on the store, as
        #: the runner does; None where no runner keeps the attempt's status.
        self._on_no_retry = on_no_retry
        #: Role name to the digest of its image, from ``record_image``.
        self._image_digests: dict[str, str] = {}
        #: Each output's key to the sha256 and size its writer gave (``write_output``,
        #: ``reported_output``).
        self._outputs: dict[str, tuple[str, int]] = {}
        #: ``<kind>/<key>`` to the ENTRY.json digest of each entry ``materialize`` copied.
        self._cache_entries: dict[str, str] = {}
        #: The copy ``materialize`` returned, by kind, key and pinned digest, for the next role.
        self._materialized: dict[tuple[str, str, str], Path] = {}
        #: The env file written for each role whose report has not removed it yet.
        self._env_files: dict[str, Path] = {}
        #: What was known of each role the stage stopped as planned, read before the stop's
        #: SIGTERM (``stop_role``).
        self._at_stop: dict[str, _AtStop] = {}
        #: The env file prepared for a container role not started yet, with the variables it
        #: holds (``prepare_container_env``).
        self._prepared_env: dict[str, tuple[Path, dict[str, str]]] = {}
        #: The env file prepared for a host role not started yet, with the names of the secrets
        #: it holds (``prepare_role_env``).
        self._prepared_secrets: dict[str, tuple[Path, frozenset[str]]] = {}
        #: The role directories this context made, which ``close`` removes.
        self._role_dirs: list[Path] = []
        #: Roles given secrets and not reported yet. Each logs into its node-local directory, and
        #: its log files reach the attempt's ``logs/`` only scrubbed, when it is reported.
        self._private_logs: set[str] = set()
        #: Container roles whose logs in the attempt's ``logs/`` are scrubbed in place when they
        #: are reported (``start_container``'s *scrub_logs*).
        self._scrubbed_logs: set[str] = set()
        #: The process ceiling of each contestant role that has one, its step's or its commands',
        #: and how the kernel counts it, by role name: what the walks sample its pool against and
        #: its end is judged by (``starvation_fault``).
        self._ceilings: dict[str, CeilingRule] = {}
        #: The process ceiling of the stage's contestant steps once read (``process_ceiling``);
        #: None before.
        self._contestant_ceiling: ProcessCeiling | None = None
        #: How the build sandbox's commands are counted, once chosen (``nproc_accounting``); None before.
        self._accounting: AccountingChoice | None = None
        #: The limits of the per-uid kernel counts once read (``ucount_limits``); None before.
        self._ucount_limits: tuple[UcountLimits, ...] | None = None
        #: The node-local copy of each cache entry ``materialize`` made, parts of which a
        #: contestant's container may mount read-only.
        self._copies: list[_Copy] = []
        #: The seal files of each sealed container role.
        self._seals: dict[str, SealFiles] = {}
        #: GiB the seal of each sealed container role holds beside it, where that is not 0
        #: (``ContainerRuntime.seal_memory_gib``), which the stage's budget counts while it runs.
        self._seal_memory: dict[str, int] = {}
        #: Each image this context pinned, by its file's sha256; ``close`` removes them.
        self._pinned: dict[str, PinnedImage] = {}
        #: The image ``node_image`` pinned, by the entry's key and pinned digest.
        self._node_images: dict[tuple[str, str], PinnedImage] = {}
        #: True once every running role was asked to stop.
        self._terminated = False
        #: POSIX time at which roles that were asked to stop and still run are killed.
        self._grace_until = 0.0
        #: Roles that were asked to stop.
        self._signalled: set[str] = set()
        #: Roles the stage stopped as it planned (``stop_role``).
        self._planned: set[str] = set()
        #: Set when a role ended with a fault of its own.
        self._faulted = False
        #: The contestant fault a stage that decides its contestant faults declared it raises
        #: (``declare_contestant_fault``); None until it does.
        self._declared: FaultRecord | None = None
        #: The fault, infra and never retried, an end of the stage is once it has done work a retry
        #: would do again (``declare_no_retry``); None before.
        self._no_retry: FaultRecord | None = None
        self._roles: dict[str, RoleHandle] = {}
        self._shared: set[str] = set()
        #: The roles started with *share_memory_of*, whose memory the stage's budget does not count.
        self._memory_shared: set[str] = set()
        self._reports: dict[str, RoleReport] = {}
        #: When the stage saw each reported role's step start, on its clock (``role_started_at``).
        self._started_at: dict[str, float] = {}
        #: When the stage first found each container role's start marker, on its clock
        #: (``command_started_at``).
        self._marker_found_at: dict[str, float] = {}
        #: Seconds each role's step took to start once launched, for every role seen started.
        self._launch_waits: dict[str, float] = {}
        self._launch_errors: dict[str, str] = {}
        #: How each reported role ended; a role is reported once.
        self._outcomes: dict[str, RoleOutcome] = {}
        #: The memory split of each role started with a harness reserve, by role name.
        self._splits: dict[str, _Split] = {}
        #: A contestant container's source of its exec server's ready-line pids, by role, so its
        #: memory split measures the contestant's PID namespace depth rather than assume it
        #: (``start_container``'s *ready_source*).
        self._ready_sources: dict[str, Callable[[], ReadyPids | None]] = {}
        #: The job's OOM counts when the stage started; None when unknown.
        self._job_ooms_at_start = watch.job_ooms() if watch is not None else None
        #: The name of the role whose step is being started off the stage's thread
        #: (``start_container_off_thread``), and the worker thread's id, while that start is in
        #: flight; None otherwise. While it is set, the stage's thread may start no other role, pin
        #: no image, copy no cache entry, hold no deadline and close nothing
        #: (``_refuse_while_off_thread``); the worker thread finishes its own start.
        self._off_thread: str | None = None
        self._off_thread_id: int | None = None

    def _refuse_while_off_thread(self, action: str) -> None:
        """Raise RuntimeError when the stage's thread does *action* while a role's step is being
        started off it (``start_container_off_thread``): two admissions could each miss the other's
        resources, and the launch's own deadline growth would race a hold. The worker thread that
        runs the off-thread start is let through: it alone finishes that start."""
        off_thread = self._off_thread
        if off_thread is not None and threading.get_ident() != self._off_thread_id:
            raise RuntimeError(f"{action} while role {off_thread}'s step is being started off the stage's thread; "
                               "collect that launch first")

    @property
    def params(self) -> Any:
        """The stage's parameters as its parameter model."""
        return stage_class(self.stage.kind).params_model.model_validate(self.stage.params)

    @property
    def attempt_dir(self) -> Path:
        """Absolute path of this attempt's directory in the store."""
        return self.store.path(self.plan.layout.attempt(self.stage.name, self.attempt))

    def output_key(self, name: str) -> str:
        """Store key of output *name* of this attempt."""
        return self.plan.layout.attempt_file(self.stage.name, self.attempt, f"outputs/{name}")

    def output_path(self, name: str) -> Path:
        """Filesystem path of output *name*, with its directory created, for a role to write. The
        role reports the output's sha256 (``outputs_report``, ``reported_output``)."""
        path = self.store.path(self.output_key(name))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_output(self, name: str, data: bytes) -> str:
        """Write output *name* and return its key. Its sha256 and size are those of *data*, the
        bytes this process wrote, which the manifest records (``output_ref``)."""
        key = self.output_key(name)
        self.store.write_bytes(key, data)
        self._outputs[key] = (sha256_bytes(data), len(data))
        return key

    @property
    def outputs_dir(self) -> Path:
        """The attempt's ``outputs/`` directory, which a role reports its outputs' paths under
        (``report_outputs``): output *name* is ``outputs_dir / name``."""
        return self.attempt_dir / "outputs"

    def outputs_report(self, role: str) -> Path:
        """Where role *role* reports the sha256 and size of each output it writes, by the
        output's path under ``outputs_dir``: a file in its own node-local directory
        (``hls.store.manifest.report_outputs``)."""
        return self.role_dir(role) / _OUTPUTS_REPORT

    def reported_output(self, name: str, role: str) -> str:
        """The key of output *name*, which role *role* of this stage wrote, recorded for the
        manifest with the sha256 and size the role reported for it in its ``outputs_report``,
        keyed by *name*, its path under ``outputs_dir``: the runner reads none of the output's
        bytes. Only the report of a role whose code is the harness's counts: a
        harness role's, or a host role's, since a contestant's code runs only in containers
        (``start_role``). And only once the stage has waited for the role, which could otherwise
        write the output again after it reported it. Raise ValueError for a role the stage never
        started or a contestant's container, RuntimeError for a role not yet waited for, and
        StageFault (infra, retryable) when the role reported no record of the output."""
        handle = self._roles.get(role)
        if handle is None:
            raise ValueError(f"stage {self.stage.name} started no role {role}")
        if handle.spec.owner == "contestant" and handle.spec.start_marker is not None:
            raise ValueError(f"contestant container {role}: a contestant's code chooses what it writes, its "
                             "reports among them")
        if role not in self._outcomes:
            raise RuntimeError(f"role {role} has not been waited for, so it may still write its output "
                               f"{name}; wait for it first")
        reported = (read_outputs_report(self.outputs_report(role)) or {}).get(name)
        if reported is None:
            raise StageFault.infra(f"role {role} reported no digest of output {name}", retryable=True,
                                   role=role, output=name)
        key = self.output_key(name)
        self._outputs[key] = (reported.sha256, reported.size)
        return key

    def output_ref(self, key: str) -> OutputRef:
        """The manifest's record of the output at *key*: the sha256 its writer gave, of the bytes
        this process wrote (``write_output``) or by a role's report (``reported_output``), and its
        size from the store's metadata, which must be the size its writer gave. Nothing of it is
        read, so a rewrite that keeps the size goes unseen; only harness code can write an output.
        Raise as ``Store.size`` does when nothing is at *key*; StageFault (infra, not retryable)
        when no writer gave its sha256, which no retry of the same code would; and StageFault
        (infra, retryable) when it has another size now than its writer gave."""
        size = self.store.size(key)
        recorded = self._outputs.get(key)
        if recorded is None:
            raise StageFault.infra(f"no writer reported the sha256 of output {key}: a role reports it "
                                   "(reported_output) or the stage writes it (write_output)",
                                   retryable=False, output=key)
        sha256, written = recorded
        if size != written:
            raise StageFault.infra(f"output {key} has {size} bytes, not the {written} its writer gave",
                                   retryable=True, output=key)
        return output_ref(self.store, key, sha256)

    def report(self, name: str) -> RoleReport | None:
        """The grant report of role *name*; None if it never reported."""
        return self._reports.get(name)

    def launch_waits(self) -> dict[str, float]:
        """Seconds each role seen started waited for its step to start, from its launch request
        until the stage saw the start, by role name: what the deadline grew by."""
        return dict(self._launch_waits)

    def role_started_at(self, name: str) -> float | None:
        """When the stage saw role *name*'s step start, on ``clock``: the time a role's own work
        counts from, however long the scheduler took to start its step. None for a role that
        never reported."""
        return self._started_at.get(name)

    def running_roles(self) -> list[str]:
        """Names of the started roles whose steps are still running."""
        return [name for name, handle in list(self._roles.items()) if handle.poll() is None]

    @property
    def image_digests(self) -> dict[str, str]:
        """Role name to the digest of the image it ran, as recorded with ``record_image``."""
        return dict(self._image_digests)

    @property
    def cache_entries(self) -> dict[str, str]:
        """``<kind>/<key>`` to the ENTRY.json digest of each entry ``materialize`` copied,
        ``seed`` copied writable or ``entry_file`` read a file of."""
        return dict(self._cache_entries)

    def role_dir(self, name: str) -> Path:
        """Role *name*'s node-local directory, ``<scratch>/roles/<name>``, with mode 0700. The
        context owns it: on the first call it removes whatever is at that path, which an
        earlier stage's role of the same name may have left (without following a link), and
        makes the directory afresh, with the ``tmp`` directory the role's step gets as its
        TMPDIR in it and the files its log cappers append their own errors to
        (``log_cap.errors_file``); ``close`` removes it. The launcher empties that ``tmp``
        directory in place at each launch and opens those files as they are, so a role made
        ready before a contestant starts needs no new space on the node's disk to launch."""
        check_role_name(name)
        if self.scratch is None:
            raise RuntimeError("this stage context has no node-local scratch")
        path = self.scratch / "roles" / name
        if path not in self._role_dirs:
            _remove(path)
            path.mkdir(parents=True)
            path.chmod(0o700)
            (path / "tmp").mkdir(mode=0o700)
            for suffix in (".out", ".err"):
                log_cap.errors_file(path, suffix).touch(mode=0o600)
            self._role_dirs.append(path)
        return path

    def start_role(self, name: str, argv: Sequence[str], shape: ResourceShape, *,
                   owner: Literal["harness", "contestant"], env: Mapping[str, str] | None = None,
                   secret_env: Collection[str] = (), share_gpus_of: RoleHandle | None = None,
                   overlap: bool = False, scrub_logs: bool = False, daemon: bool = False,
                   share_memory_of: RoleHandle | None = None) -> RoleHandle:
        """Start role *name* in its own step and wait for its grant report, then put its step
        under the memory watch. The wait is two: for the scheduler to start the step, then for
        the step to report (``_await_report``). A role whose step does not start or report in
        time, or that is still waited for when the stage is told to stop, is killed and ends
        ``launch_failed``, an infra fault that is retried. *owner* says whose
        code the role runs, which decides what it may see and whose fault its failure is; it
        has no default, so no caller lands on either side of the wall by leaving it out. A
        contestant's code runs only in a container (``start_container``), so a contestant host
        role is refused unless the stage's class sets ``contestant_host_roles``: only the no-op
        stage does, to exercise the OOM rules on a host step. The reuse of an extracted image a
        stage left (``_pin``) rests on this.

        *argv* and *env* hold no secret. *secret_env* names secrets granted to the stage that a
        harness role gets in its environment: their values go into a fresh 0600 env file in
        the role's directory, which the step wrapper takes and unlinks before the role starts.
        Such a role writes its stdout, stderr and grant report into ``logs/`` in its node-local
        directory, never into the store, and the attempt's ``logs/`` gets them scrubbed when
        the role is reported; a runner killed before that leaves no value of them in the store.
        With *scrub_logs* a role given secrets logs into the attempt's ``logs/`` as every other
        role does, its report among them, and they are scrubbed in place when it is reported, as
        ``start_container``'s *scrub_logs* scrubs a container's: a role that starts after a
        contestant may have filled the node's disk then needs no new space there. A runner
        killed while it runs leaves them unscrubbed in the store, where finalize's secret scan
        reads them.
        A contestant role gets none: the secrets granted to a stage are host-only.

        With *overlap* the role's step shares the CPUs of the stage's other steps instead of
        taking its own (``RoleSpec.overlap``), as a driver step shares its server's: its CPUs
        are not counted against the stage's, as a shared role's are not, and its memory is. It
        asks for no GPU. With *daemon* the role speaks to the node's container daemon
        (``RoleSpec.daemon``): an image runtime's pin or build role.

        With *share_memory_of* the role's memory is oversubscribed on that role's, as a shared
        role's GPUs are on its target's: an overlap step (*share_gpus_of* or *overlap*), whose
        ``--mem`` is its own limit, that the stage's budget does not count, the job's memory limit
        bounding the two together (``budget_problem``'s *oversubscribed*). Only a role that sets
        it is counted so.

        Raise StageFault, starting nothing, when the stage is stopping or the share target did
        not start cleanly (infra, retryable), or when the role does not fit in the stage's
        resources next to the roles still running (infra, not retryable). Raise ValueError,
        writing nothing, for a name that is not a role name or is already started, a contestant
        role of a stage that runs none on the host, an overlap role that asks for a GPU, an
        argument or a value in *env* that holds a secret, a contestant role with *secret_env*,
        or a name in *secret_env* the stage was not granted."""
        check_role_name(name)
        self._refuse_while_off_thread(f"role {name} starts")
        if owner == "contestant" and not stage_class(self.stage.kind).contestant_host_roles:
            raise ValueError(f"contestant role {name}: a contestant's code runs only in a container "
                             "(start_container)")
        self._admit(name, shape, share_gpus_of, overlap=overlap, share_memory_of=share_memory_of)
        if secret_env and owner == "contestant":
            raise ValueError(f"contestant role {name}: the secrets granted to a stage are host-only and never reach "
                             "a contestant's code")
        env = dict(env or {})
        held = self._holding_secrets(env, argv, "argv")
        if held:
            raise ValueError(f"role {name}: {', '.join(held)} holds a secret; a role gets a secret only through "
                             "secret_env")
        missing = sorted(set(secret_env) - set(self.secrets))
        if missing:
            raise ValueError(f"stage {self.stage.name} was not granted {missing}")
        prepared = self._prepared_secrets.get(name)
        if prepared is not None and prepared[1] != frozenset(secret_env):
            raise ValueError(f"role {name}: its secrets are not the ones its env file was prepared with")
        if prepared is not None and not scrub_logs:
            raise ValueError(f"role {name}: a role prepared before a contestant starts logs into the attempt "
                             "(scrub_logs), never onto the node's disk")
        if self.scratch is not None:
            # Made before the launch, so that the role's step never runs in a directory the
            # context would later empty.
            self.role_dir(name)
        env_file = None
        log_dir = self.attempt_dir / "logs"
        if secret_env:
            env_file = (self._prepared_secrets.pop(name)[0] if prepared is not None
                        else self._write_env_file(name, _ROLE_ENV, {k: self.secrets[k] for k in secret_env}))
            if scrub_logs:
                self._scrubbed_logs.add(name)
            else:
                log_dir = self.role_dir(name) / "logs"
                self._private_logs.add(name)
        nproc = self.process_ceiling().ceiling if owner == "contestant" else None
        spec = RoleSpec(name=name, argv=tuple(argv), shape=shape, owner=owner, log_dir=log_dir, env=env,
                        env_file=env_file, overlap=overlap, nproc=nproc, daemon=daemon)
        if nproc is not None:
            self._ceilings[name] = CeilingRule(ceiling=nproc, accounting="uid", parent_limit=None)
        if share_memory_of is not None:
            self._memory_shared.add(name)
        return self._launch(spec, share_gpus_of)

    def process_ceiling(self) -> ProcessCeiling:
        """The process ceiling of every contestant step this stage starts (``hls.substrate.nproc``):
        the runner's hard limit less ``run.nproc_reserve`` for the harness of each evaluation of
        the stage's size the node holds, capped at ``site.nproc_max``. Read once and kept; the
        ``process_ceiling`` event records it. ``start_role`` and ``start_container`` give it to
        every contestant launch, so no stage can start one without it. Raise StageFault, infra
        and not retried, for a ceiling or a cap under its floor, or a node whose evaluations of
        the stage's size are unknown: no contestant step of the stage can run on this node."""
        if self._contestant_ceiling is None:
            shape = self.stage.shape
            ceiling = ceiling_for(hard_limit(), reserve=self.plan.settings.run.nproc_reserve,
                                  nproc_max=self.plan.settings.site.nproc_max, node=self.launcher.node_size(),
                                  gpus=shape.gpus, cpus=shape.cpus)
            self.events.emit("process_ceiling", **ceiling.model_dump())
            if ceiling.problem is not None:
                raise StageFault.infra(f"no contestant step of stage {self.stage.name} can run on this node: "
                                       f"{ceiling.problem}", retryable=False)
            self._contestant_ceiling = ceiling
        return self._contestant_ceiling

    def nproc_accounting(self) -> AccountingChoice:
        """How this node's kernel counts the build sandbox's commands against their process
        ceiling (``hls.substrate.nproc``): ``site.nproc_accounting`` when the site sets it, else
        what this node's kernel release says. Chosen once and kept; the ``nproc_accounting`` event
        records it, where it came from and the release. Raise StageFault, infra and not retried,
        for a release that says nothing when the site sets none."""
        if self._accounting is None:
            release = kernel_release()
            try:
                chosen = choose_accounting(self.plan.settings.site.nproc_accounting, release)
            except ValueError as e:
                raise StageFault.infra(str(e), retryable=False) from None
            self.events.emit("nproc_accounting", accounting=chosen.accounting,
                             source="site.nproc_accounting" if chosen.from_site else "kernel release", kernel=release)
            self._accounting = chosen
        return self._accounting

    def ucount_limits(self) -> tuple[UcountLimits, ...]:
        """The limits of each per-uid kernel count the node limits (``hls.substrate.ucounts``):
        (b) for each sealed contestant role but a graded server, written in its seal's user
        namespace (``hls.substrate.ucounts.ucounts_env``), and (a) for the build sandbox's commands
        within one, written by its spawner's reaper (``hls.substrate.ucounts.commands_limits``),
        from the node's own limits, the site's caps and the evaluations of the stage's size the
        node holds (``process_ceiling``): such a role runs on the node it is measured on. A graded
        server's are ``served_ucounts``. Read once and kept; the ``ucount_limits`` event records
        them. Raise StageFault, infra and not retried, for a limit under its floor, or a node limit
        that cannot be read: no sandbox of the stage can run on this node."""
        if self._ucount_limits is None:
            site = self.plan.settings.site
            made = ucount_limits(self._node_limits(retryable=False),
                                 {"sandbox_namespaces": site.sandbox_namespaces,
                                  "sandbox_inotify_instances": site.sandbox_inotify_instances,
                                  "sandbox_inotify_watches": site.sandbox_inotify_watches},
                                 self.process_ceiling().evaluations)
            self.events.emit("ucount_limits", limits=[c.model_dump() for c in made])
            problems = [c.problem for c in made if c.problem is not None]
            if problems:
                raise StageFault.infra(f"no sandbox of stage {self.stage.name} can run on this node: "
                                       + "; ".join(problems), retryable=False)
            self._ucount_limits = made
        return self._ucount_limits

    def served_ucounts(self, artifact: ResourceShape) -> ServedUcounts:
        """The limit a graded server of the served artifact's shape *artifact*
        (``hls.pipeline.pipelines.artifact_shape``) is held to on each per-uid kernel count: its
        share of a grading node as the site declares the node's own limits
        (``site.grading_node_*``), among the evaluations of *artifact*'s size this node holds, k
        from the controller's node size and *artifact*'s GPUs and CPUs alone
        (``hls.substrate.nproc.node_evaluations``) (``hls.substrate.ucounts.served_limits``). It
        is the one computation of them: a build computes them when it renders its brief, which
        states them, and records them with its artifact, whose grading seal writes them
        (``grading_ucounts``); only an artifact no build stated them for has them computed when it
        is graded. The ``served_ucounts`` event records them. Raise StageFault, infra and not
        retried, for a limit under its floor: the site's declaration leaves no graded server of
        this size room, and for a node whose evaluations of *artifact*'s size are unknown
        (``node_evaluations``)."""
        evaluations = self._evaluations_of(artifact)
        site = self.plan.settings.site
        made = served_limits({"grading_node_namespaces": site.grading_node_namespaces,
                              "grading_node_inotify_instances": site.grading_node_inotify_instances,
                              "grading_node_inotify_watches": site.grading_node_inotify_watches}, evaluations)
        self.events.emit("served_ucounts", limits=[c.model_dump() for c in made])
        problems = [c.problem for c in made if c.problem is not None]
        if problems:
            raise StageFault.infra(f"no graded server of stage {self.stage.name}'s artifact can run under the "
                                   "site's declared grading node limits: " + "; ".join(problems), retryable=False)
        return ServedUcounts(limits=made)

    def grading_ucounts(self, artifact: ResourceShape, recorded: ServedUcounts | None) -> dict[str, int]:
        """What the seal of a graded server of *artifact*'s shape, and of the probe sealed as it
        will be, writes: the limits its build *recorded* when it stated them in its brief, exactly;
        for an artifact no build stated them for (None: a measured image, an operator's named
        bundle), the limits ``served_ucounts`` gives here. Each is given by its ``/proc/sys/user``
        file's name, for each count this node's kernel has (one it lacks cannot be made here). The
        ``grading_ucounts`` event records where they came from, this node's count of servers of the
        size and what is written. Raise StageFault, infra and retried, when this node's own limits
        cannot hold that many servers' shares beside their harnesses
        (``hls.substrate.ucounts.node_shortfall``) or cannot be read: another node may; and infra
        not retried for a node whose evaluations of *artifact*'s size are unknown."""
        held = recorded if recorded is not None else self.served_ucounts(artifact)
        evaluations = self._evaluations_of(artifact)
        node = self._node_limits(retryable=True)
        written = {name: limit for name, limit in held.figures().items() if name in node}
        self.events.emit("grading_ucounts", source="declared" if recorded is None else "recorded",
                         evaluations=evaluations, held=written)
        short = node_shortfall(held.limits, node, evaluations)
        if short is not None:
            raise StageFault.infra(f"this node cannot give its graded servers the limits they are held to: {short}",
                                   retryable=True)
        return written

    def _evaluations_of(self, artifact: ResourceShape) -> int:
        """How many evaluations of *artifact*'s size a node like this one holds
        (``hls.substrate.nproc.node_evaluations``). Raise StageFault, infra and not retried, for a
        node whose count of them is unknown: any figure for it would be a guess, and the runner
        refuses such a node for a contestant stage when its allocation starts."""
        try:
            return node_evaluations(self.launcher.node_size(), gpus=artifact.gpus, cpus=artifact.cpus)
        except ValueError as e:
            raise StageFault.infra(f"the limits a graded server is held to on a node like this one cannot be "
                                   f"told: {e}", retryable=False) from e

    def _node_limits(self, *, retryable: bool) -> dict[str, int]:
        """This node's own per-uid limits (``hls.substrate.ucounts.node_limits``). Raise StageFault,
        infra, *retryable* or not, when they cannot be read."""
        try:
            return node_limits()
        except (OSError, ValueError) as e:
            raise StageFault.infra(f"the node's per-uid limits could not be read: {e}",
                                   retryable=retryable) from e

    def start_container(self, name: str, spec: ContainerSpec, shape: ResourceShape, *,
                        owner: Literal["harness", "contestant"], env: Mapping[str, str] | None = None,
                        share_gpus_of: RoleHandle | None = None, overlap: bool = False,
                        command_nproc: int | None = None,
                        harness_reserve_gib: float | None = None, scrub_logs: bool = False,
                        one_numa_node: bool = False, seal_ucounts: Mapping[str, int] | None = None,
                        ready_source: Callable[[], ReadyPids | None] | None = None,
                        share_memory_of: RoleHandle | None = None) -> RoleHandle:
        """Start role *name* as the container *spec* describes, as ``start_role`` starts a role,
        with the same required *owner*, *overlap* and *share_memory_of*. A contestant's container runs under the
        stage's process ceiling (``process_ceiling``), its step's ``RoleSpec.nproc``. One whose own
        harness gives its commands the ceiling, the build sandbox's spawner, passes that ceiling
        as *command_nproc*: its step keeps its limit, so the harness in it can still fork once the
        commands reach the ceiling. The spawner sets it inside a user namespace it made with that
        limit, so the kernel counts it as ``site.nproc_accounting`` says, against every task of
        the uid on the node or against the commands' own (``CeilingRule``); a step's own ceiling
        counts every task of the uid on any kernel. The context checks the uid's tasks before the
        launch against what they count against, as the step wrapper does for a step's: a uid
        already at it refuses the start (infra, retryable; ``CeilingRule.refusal``), and a count
        that fails lets it start, logged, the check only sparing a start whose commands could
        fork nothing. A *command_nproc* that is not the
        stage's ceiling, or given a harness container, is refused with ValueError.
        *one_numa_node* refuses the step when its CPUs are split over NUMA nodes
        (``RoleSpec.one_numa_node``): it ends ``grant_mismatch``, infra, with its CPUs and nodes in
        the detail. A contestant container's step also has its pids limit count read, and a count
        above 0 ends the role ``contestant_pids``, before any reading that would make its end infra.
        With either ceiling, the walks also sample its uid's process pool and its own tasks, and
        the role's outcome carries the last sample before it ended or was stopped (``pool``), as
        context; what decides whether it was starved is the uid's tasks counted when it is
        reported, its step gone (``pool_at_end``), and a count that fails there leaves that not
        ruled out, so its end is infra, retried (``starvation_fault``). With *scrub_logs* the
        role's logs, which may quote what a secret's host is, are scrubbed in place in the
        attempt's ``logs/`` when it is reported.
        An image that holds a contestant's code
        (``PinnedImage.contestant``) runs the contestant's own containers only: a harness
        container on one is refused. A contestant's container declares its disk
        (``shape.disk_gib``), which caps what its workdir and its writable mounts hold: the
        memory watch walks them and kills the role once they hold more
        (``MemoryWatch.watch_disk``), and the role then ends ``disk_limit``, a contestant fault.
        Without a watch nothing is walked. One that declares no disk is refused.
        *env* holds the container's only variables. They go into a fresh 0600 env file in the
        role's directory, which the launcher hands the runtime, so none appears on a command
        line. No secret reaches a container: the secrets granted to a stage are host-only, so a
        value in *env* or a string of the spec that holds a secret is refused
        (``container_strings``). ``_container_problems`` says which paths a container may mount.

        A sealed container (``spec.seal``) also gets the seal's client variables, and the
        launcher gives its seal its files, which live in the role's 0700 ``seal`` directory,
        cleared before every launch, that no mount may reach, and the role's limits on the uid's
        per-namespace counts (``ucount_limits``, ``ContainerRole.ucounts``), which the seal writes
        in its user namespace before the payload, the same for a contestant's container and a harness one
        sealed as a contestant's will be. A seal that serves (``SealPlan.ingress_port``), a graded
        server's or its probe's, writes *seal_ucounts* instead, the limits the server is held to
        (``grading_ucounts``), which it must be given; any other seal writes the stage's own share
        and refuses them. A stage reads those files through ``seal_status`` and
        ``seal_files``. A sealed role's nonzero exit after its start marker is its payload's
        only when the seal recorded that ending with the interceptor still running
        (``_seal_failure``). The digest of the image the container runs is recorded under
        *name* (``record_image``).

        A contestant container whose step also holds harness processes, the build sandbox,
        gives *harness_reserve_gib*: the memory of its step its harness may hold. Its step's
        memory is then sampled split by owner (``hls.agent.memory.SplitSampler``), and a breach
        where the runner enforces memory ends the contestant's largest process
        (``hls.agent.memory.ContestantKiller``) and leaves the role running; the watch kills the
        step only once no contestant process is left or such kills do not bring it back to its
        limit, a runner's kill over its limit like any other. When the role ends, its outcome
        carries the split's evidence (``RoleOutcome.split``), and ``classify_role`` reads it by
        the build rule. The split and the kill find the contestant's PID namespace by the pids
        *ready_source* returns (``hls.agent.memory.ReadyPids``, the exec server's ready line's);
        without them, or while it returns None, the split keeps no sample and the kill ends
        nothing. A harness reserve given a harness container, or not above 0 and below the role's
        memory, is refused with ValueError, writing nothing.

        Raise as ``start_role`` does, and ValueError, writing nothing, for a path the container
        may not see, a value that holds a secret, or a spec the runtime cannot express."""
        if self.runtime is None:
            raise RuntimeError("this stage context has no container runtime")
        runtime = self.runtime
        check_role_name(name)
        self._refuse_while_off_thread(f"container {name} starts")
        if owner == "harness" and command_nproc is not None:
            raise ValueError(f"harness container {name}: a process ceiling is for a contestant's container")
        if owner == "contestant" and shape.disk_gib < 1:
            raise ValueError(f"contestant container {name} declares no disk; its shape's disk_gib caps what it "
                             "writes")
        seal_gib = runtime.seal_memory_gib if spec.seal is not None else 0
        self._admit(name, shape, share_gpus_of, overlap=overlap, share_memory_of=share_memory_of,
                    seal_memory_gib=seal_gib)
        if harness_reserve_gib is not None and (owner != "contestant"
                                                or not 0 < harness_reserve_gib < shape.memory_gib):
            raise ValueError(f"{owner} container {name}: a harness reserve ({harness_reserve_gib} GiB) is for a "
                             f"contestant container, and must be above 0 and below its {shape.memory_gib} GiB")
        problems = self._container_problems(name, spec, owner)
        if problems:
            raise ValueError(f"{owner} container {name}: " + "; ".join(problems))
        serves = spec.seal is not None and spec.seal.ingress_port is not None
        if serves and seal_ucounts is None:
            raise ValueError(f"{owner} container {name}: a seal that serves writes the limits a graded server is "
                             "held to, which it must be given (grading_ucounts)")
        if not serves and seal_ucounts is not None:
            raise ValueError(f"{owner} container {name}: only a seal that serves is given a graded server's limits")
        env = dict(env or {}) if spec.seal is None else {**(env or {}), **client_env(spec.seal)}
        prepared = self._prepared_env.get(name)
        if prepared is not None and prepared[1] != env:
            raise ValueError(f"{owner} container {name}: its variables are not the ones its env file was prepared "
                             "with")
        runtime.check(spec)
        held = self._holding_secrets(env, container_strings(spec), "spec")
        if held:
            raise ValueError(f"{owner} container {name}: {', '.join(held)} holds a secret; the secrets granted to a "
                             "stage are host-only and never reach a container")
        ceiling = self.process_ceiling().ceiling if owner == "contestant" else None
        if command_nproc is not None and command_nproc != ceiling:
            raise ValueError(f"contestant container {name}: its commands' process ceiling {command_nproc} is not the "
                             f"stage's, {ceiling}")
        nproc = ceiling if command_nproc is None else None
        rule = None
        if ceiling is not None:
            rule = (CeilingRule(ceiling=ceiling, accounting="uid", parent_limit=None) if command_nproc is None
                    else CeilingRule(ceiling=ceiling, accounting=self.nproc_accounting().accounting,
                                     parent_limit=self.process_ceiling().hard))
        files, written = None, {}
        if spec.seal is not None:
            files = self._seal_state(name, spec)
            written = (dict(seal_ucounts) if seal_ucounts is not None
                       else {c.name: c.role for c in self.ucount_limits()})
            self._seals[name] = files
        if rule is not None and command_nproc is not None:
            try:
                held: int | None = uid_tasks(os.getuid())
            except Exception:  # noqa: BLE001 -- the check only spares a start whose commands could fork nothing
                # The role's end is judged by the pool counted when it is reported, so a count that
                # fails here starts the role; it never fails the launch.
                log.exception("the uid's tasks could not be counted before container %s; starting it without that "
                              "check", name)
                held = None
            refused = None if held is None else rule.refusal(held)
            if refused is not None:
                raise StageFault.infra(f"container {name}: {refused}", retryable=True, role=name,
                                       uid_tasks=str(held), ceiling=str(command_nproc),
                                       accounting=rule.accounting)
        env_file = (self._prepared_env.pop(name)[0] if prepared is not None
                    else self._write_env_file(name, _CONTAINER_ENV, env))
        container = ContainerRole(spec=spec, env_file=env_file, seal_files=files, ucounts=written)
        role = RoleSpec(name=name, argv=(), shape=shape, owner=owner, log_dir=self.attempt_dir / "logs",
                        start_marker=runtime.start_marker, overlap=overlap, nproc=nproc, one_numa_node=one_numa_node,
                        container=container)
        reserve = None if harness_reserve_gib is None else int(harness_reserve_gib * _GIB)
        if rule is not None:
            self._ceilings[name] = rule
        if ready_source is not None:
            self._ready_sources[name] = ready_source
        if scrub_logs:
            self._scrubbed_logs.add(name)
        if share_memory_of is not None:
            self._memory_shared.add(name)
        handle = self._launch(role, share_gpus_of, reserve_bytes=reserve)
        # Every image a container runs is in the stage's manifest, under the role that ran it.
        self.record_image(name, spec.image.file_sha256)
        report = self._reports.get(name)
        if owner == "contestant" and self.watch is not None and report is not None and not report.mismatches:
            written = [spec.workdir, *(m.source for m in spec.mounts if not m.read_only)]
            cgroup = Path(report.cgroup_dir)
            # The harness's own processes in the container, which the walks never judge by their
            # descriptors: the build sandbox's reaper, found by its ready source; none in any other.
            harness = ((lambda: harness_pids(cgroup, ready_source())) if ready_source is not None
                       else (lambda: ()))
            self.watch.watch_disk(report.step, written, shape.disk_gib * _GIB, handle.kill, role=name,
                                  cgroup_dir=cgroup, ceiling=ceiling, harness_pids=harness)
        return handle

    def start_off_thread(self, name: str, start: Callable[[], RoleHandle]) -> Future[RoleHandle]:
        """Run *start*, which starts role *name*'s step (``start_container``), in a worker thread,
        and return the future of its handle: a stage whose own thread must not wait for the
        scheduler to grant the step, such as agent_build's drive for each snapshot, starts it this
        way so its I/O and its own deadline keep going. At most one off-thread start is in flight:
        while it is, every start, pin, hold or close on the stage's thread is refused
        (``_refuse_while_off_thread``), so the two threads never admit against the same resources or
        grow the deadline at once; the stage collects the future (its handle, or the start's
        failure) before it does any of those. The worker thread alone finishes the start, so
        ``start``'s own ``start_container`` is let through. The step runs on after, like any role's;
        the stage tracks and stops it as it tracks the handle (``wait_roles``, ``stop_role``). Raise
        RuntimeError when one is already in flight."""
        self._refuse_while_off_thread(f"role {name} starts off the thread")
        launched: Future[RoleHandle] = Future()
        self._off_thread = name

        def run() -> None:
            self._off_thread_id = threading.get_ident()
            if not launched.set_running_or_notify_cancel():
                self._off_thread = self._off_thread_id = None
                return
            try:
                handle = start()
            except BaseException as e:  # noqa: BLE001 -- handed to the stage's thread with the future
                self._off_thread = self._off_thread_id = None
                launched.set_exception(e)
            else:
                self._off_thread = self._off_thread_id = None
                launched.set_result(handle)

        threading.Thread(target=run, name=f"start-{name}", daemon=True).start()
        return launched

    def prepare_container_env(self, name: str, env: Mapping[str, str], *, seal: SealPlan | None = None) -> None:
        """Write now the env file of container role *name*, which the stage starts later with
        *env* (and *seal*, whose client variables it gets too), as ``start_container`` would
        write it then. A stage whose contestant may fill the node's disk does this before the
        contestant starts, so starting a harness container afterwards needs no new space there:
        ``start_container`` takes this file, and refuses other variables. Raise ValueError,
        writing nothing, for a name that is not a role name, is started or is prepared already,
        and for variables that hold a secret."""
        check_role_name(name)
        if name in self._roles or name in self._prepared_env:
            raise ValueError(f"container {name} is already started or prepared")
        full = dict(env) if seal is None else {**env, **client_env(seal)}
        held = self._holding_secrets(full, (), "argv")
        if held:
            raise ValueError(f"container {name}: {', '.join(held)} holds a secret; the secrets granted to a stage "
                             "are host-only and never reach a container")
        self._prepared_env[name] = (self._write_env_file(name, _CONTAINER_ENV, full), full)

    def prepare_role_env(self, name: str, secret_env: Collection[str]) -> None:
        """Write now the env file of host role *name*, which the stage starts later with the
        secrets *secret_env*, as ``start_role`` would write it then. A stage whose contestant may
        fill the node's disk does this before the contestant starts, so starting the role
        afterwards needs no new space there: ``start_role`` takes this file, and refuses other
        secrets and logs that would go to that disk, so the role must be started with
        *scrub_logs*. Raise ValueError, writing nothing, for a name that is not a role name, is
        started or is prepared already, for no secrets, and for a secret the stage was not
        granted."""
        check_role_name(name)
        if name in self._roles or name in self._prepared_secrets:
            raise ValueError(f"role {name} is already started or prepared")
        if not secret_env:
            raise ValueError(f"role {name}: a role without secrets has no env file to prepare")
        missing = sorted(set(secret_env) - set(self.secrets))
        if missing:
            raise ValueError(f"stage {self.stage.name} was not granted {missing}")
        path = self._write_env_file(name, _ROLE_ENV, {k: self.secrets[k] for k in secret_env})
        self._prepared_secrets[name] = (path, frozenset(secret_env))

    def _seal_state(self, name: str, spec: ContainerSpec) -> SealFiles:
        """Fresh seal files for role *name*, in a 0700 directory of its own that no path of
        *spec* reaches, so the container can neither read nor forge the seal's evidence. Files
        an earlier launch of the role left there are removed. Raise ValueError, writing
        nothing, when the workdir or a mount source, judged by where its links lead, holds the
        directory or lies inside it."""
        state = self.role_dir(name) / "seal"
        sealed = state.resolve()
        exposed = [str(p) for p in (spec.workdir, *(m.source for m in spec.mounts))
                   if sealed.is_relative_to(Path(p).resolve()) or Path(p).resolve().is_relative_to(sealed)]
        if exposed:
            raise ValueError(f"sealed container {name} would see its seal's files through {exposed}")
        state.mkdir(mode=0o700, exist_ok=True)
        for path in state.iterdir():
            _remove(path)
        return seal_files(state)

    def seal_status(self, name: str) -> IndexStatus | None:
        """What the interceptor of sealed role *name* recorded: the upstream failures it counted,
        the pages it refused for their size and the hosts its policy refused; None when the role
        is not sealed or its interceptor never wrote a status (it never started)."""
        files = self._seals.get(name)
        return None if files is None else index_status(files.status)

    def seal_files(self, name: str) -> SealFiles | None:
        """The seal files of sealed role *name*, such as the port file of a served container;
        None when the role is not sealed."""
        return self._seals.get(name)

    def materialize(self, kind: str, key: str, expect: str) -> Path:
        """A verified node-local copy of cache entry *kind*/*key*, recorded for the manifest.
        *expect* is the bare hex digest of the entry's ENTRY.json that the stage which
        published or found the entry recorded, and the copy is of that entry or of none. A
        container may mount it only read-only, and a contestant's only the parts
        ``_container_problems`` allows. Raise ValueError for an *expect* that is not a bare hex
        digest. A second call for the same entry and digest returns the same copy at once, so
        every role of the stage that mounts it costs nothing after the first.

        This process reads none of the entry's files. A harness role of the
        cache's program (``hls.store.cache_role``), started and waited for here, copies the
        entry into the job's node-local copies, reading the store once, and checks every file
        it wrote against the manifest; a copy a stage of the job verified before is reused
        without a role while its stamp holds, and checked by such a role when it does not
        (``ContentCache.materialize``). The role runs alone, as an extraction does (``_alone``),
        so entries are copied before the stage starts its long roles, and the stage's deadline
        and a stop end it as they end any role. Each other failure is an infra fault:
        - an entry that is not published, retryable: the stage that publishes it did not run,
          or its output was removed;
        - a published entry that is corrupt, its manifest or its files, as the copy role refuses
          it (``refusal``), not retryable: a retry would read the same entry;
        - an entry published under the key with another digest, not retryable: it was
          removed and filled again since its producer recorded it, so that producer has to
          run again;
        - a copy that differs from a sound entry, retryable: only the node-local copy is bad,
          such as one that went wrong on the way, or one an earlier stage of the job verified
          and something wrote into since, which the verifying role refuses;
        - a role that fails otherwise or reports no digests, retryable.

        An image entry is never copied: its file is read where it is, by the role that extracts
        it (``node_image``, ``entry_file``), and asking for a copy of one raises ValueError."""
        self._refuse_while_off_thread(f"cache entry {kind}/{key} is copied")
        if kind == "images":
            raise ValueError("image entries are never copied to the node: pin one with node_image, or read "
                             "its file with entry_file")
        if self.cache is None or self.scratch is None:
            raise RuntimeError("this stage context has no cache or no node-local scratch")
        check_sha256_hex(expect)
        made = self._materialized.get((kind, key, expect))
        if made is not None:
            return made
        what = f"{kind}/{key}"
        entry = self._published(kind, key, expect)
        found = entry_sha256(entry)
        try:
            path = self.cache.materialize(entry, self.scratch, self.job_id,
                                          copy=lambda dest: self._cache_role(entry, "copy", dest),
                                          verify=lambda copy: self._cache_role(entry, "verify", copy))
        except CacheCorrupt as e:
            raise StageFault.infra(f"the node-local copy of cache entry {what} differs from the entry: {e}",
                                   retryable=True, entry=what) from e
        self._cache_entries[what] = found
        self._copies.append(_Copy(kind, key, path.resolve()))
        self._materialized[(kind, key, expect)] = path
        return path

    def seed(self, kind: str, key: str, dest: Path, *, expect: str | None = None) -> str | None:
        """A writable node-local copy at *dest* of cache entry *kind*/*key*, for a role to write
        into as its own, and the digest of the ENTRY.json it was copied from, recorded for the
        manifest. *expect*, when given, is the bare hex digest of the entry's ENTRY.json that the
        stage which published or found the entry recorded, and the copy is of that entry or of none,
        as for ``materialize``. Without it the copy is of the entry as it is published now, and
        None is returned, with nothing made, when none is. Nothing at *dest* may be there before.
        The copy is made as ``materialize`` makes one, by a harness role of the cache's program
        that runs alone, checks every file it wrote against the digests the entry's manifest
        records and gives it the modification time recorded with them, so an engine's JIT in it
        finds its outputs as new as the fill left them, and is kept only when its report and those
        times hold (``ContentCache.seed``); it is never stamped or reused, since its role writes
        into it. Raise StageFault, infra: as ``materialize`` does for an entry pinned by *expect*;
        for a published entry that is corrupt, its manifest or its files as the copy role refuses
        them (not retryable: a retry would read the same entry); and for a copy that differs from a
        sound entry, its bytes or its times, and a role that fails otherwise (retryable)."""
        self._refuse_while_off_thread(f"cache entry {kind}/{key} is seeded")
        if kind == "images":
            raise ValueError("image entries are never copied to the node")
        if self.cache is None or self.scratch is None:
            raise RuntimeError("this stage context has no cache or no node-local scratch")
        what = f"{kind}/{key}"
        if expect is not None:
            entry: CacheEntry | None = self._published(kind, key, check_sha256_hex(expect))
        else:
            try:
                entry = self.cache.lookup(kind, key)
            except CacheCorrupt as e:
                raise StageFault.infra(f"cache entry {what} is corrupt: {e}", retryable=False,
                                       entry=what) from e
        if entry is None:
            return None
        try:
            self.cache.seed(entry, dest, copy=lambda tmp: self._cache_role(entry, "copy", tmp))
        except CacheCorrupt as e:
            raise StageFault.infra(f"the node-local copy of cache entry {what} differs from the entry: {e}",
                                   retryable=True, entry=what) from e
        found = entry_sha256(entry)
        self._cache_entries[what] = found
        return found

    def publish(self, kind: str, key: str, source: Path) -> Populated:
        """Cache entry *kind*/*key*, published from the node-local tree *source* a role of the stage
        wrote unless it is published already, and whether this call published it. A harness role
        of the cache's program, run alone, copies the tree into the entry's staging directory and
        reports each copy's sha256 (``hls.store.cache_role`` ``fill``), which the cache records,
        with the stage and the instance as the entry's source (``ContentCache.populate``); this
        process reads none of the tree's files. One filler holds the entry's lock and the entry
        appears with one rename, so stages that publish it at once publish it once, and a stage
        that looks for it meanwhile finds all of it or nothing. While another filler holds the
        lock this waits, no longer than the stage's deadline and not once it is stopping. Raise
        StageFault, infra: for a wait cut short and for a role that fails otherwise or reports no
        digests (retryable), and for a tree the role or the cache refuses and for a published
        entry that is corrupt (not retryable)."""
        self._refuse_while_off_thread(f"cache entry {kind}/{key} is published")
        if self.cache is None or self.scratch is None:
            raise RuntimeError("this stage context has no cache or no node-local scratch")
        what = f"{kind}/{key}"
        reported: dict[str, str] = {}

        def fill(staging: Path) -> None:
            reported.update(self._fill_role(kind, key, source, staging))

        holder = Holder.here(f"{self.stage.name} {self.plan.instance}", job_id=self.job_id)
        try:
            return self.cache.populate(kind, key, fill, holder=holder, digests=lambda: reported,
                                       source={"stage": self.stage.name, "instance": self.plan.instance},
                                       deadline=self.deadline, stopping=lambda: self.stopping)
        except WaitEnded as e:
            raise StageFault.infra(str(e), retryable=True, entry=what) from e
        except CacheCorrupt as e:
            try:
                self.cache.lookup(kind, key)
            except CacheCorrupt as published:
                raise StageFault.infra(f"cache entry {what} is corrupt: {published}", retryable=False,
                                       entry=what) from e
            raise StageFault.infra(f"the fill of cache entry {what} was refused: {e}", retryable=False,
                                   entry=what) from e

    def _fill_role(self, kind: str, key: str, source: Path, staging: Path) -> dict[str, str]:
        """Run the cache's role program as a harness role, alone, to copy the tree *source* into
        *staging*, the staging directory of entry *kind*/*key*; return the sha256 it reported for
        each copy. Raise StageFault for a refusal of the tree (infra, not retryable), for any other
        failure of the role, its own fault, and for a role that reported no digests (infra,
        retryable)."""
        what = f"{kind}/{key}"
        name = self._fresh_name(f"fill-{kind}")
        shape = self._alone(name)
        report = self.role_dir(name) / "digests.json"
        argv = [str(self.plan.toolchain.python), "-m", CACHE_ROLE_MODULE, "fill", "--source", str(source),
                "--dest", str(staging), "--digests-out", str(report)]
        self.start_role(name, argv, shape, owner="harness")
        outcomes = self.wait_roles()
        fault = stage_fault(outcomes)
        if fault is not None:
            refused = self.refusal(name, outcomes, (DIGEST_MISMATCH_EXIT,))
            if refused is not None:
                raise StageFault.infra(f"the tree for cache entry {what} was refused: {refused.reason}",
                                       retryable=False, entry=what, role=name)
            raise StageFault(fault.model_copy(update={"reason": f"cache entry {what} could not be filled: "
                                                                f"{fault.reason}",
                                                      "evidence": {**fault.evidence, "entry": what}}))
        got = read_digests(report)
        if got is None:
            raise StageFault.infra(f"role {name} reported no digests of the files it copied into cache entry "
                                   f"{what}", retryable=True, entry=what, role=name)
        return got

    def _cache_role(self, entry: CacheEntry, mode: Literal["copy", "verify"], path: Path) -> dict[str, str]:
        """Run the cache's role program as a harness role, alone, to *mode* the node-local copy
        of *entry* at *path*: ``copy`` makes it there from the store, ``verify`` checks the one
        there. Return the sha256 the role reported for each file it read back. Raise StageFault
        for a refusal: of the entry in the store by ``copy`` (infra, not retryable), of the copy
        by ``verify`` (infra, retryable); for any other failure of the role, its own fault, and
        for a role that reported no digests (infra, retryable). Raise as ``_alone`` does when the
        role may not start."""
        what = f"{entry.kind}/{entry.key}"
        name = self._fresh_name(f"{mode}-{entry.kind}-{entry.manifest_sha256[:12]}")
        shape = self._alone(name)
        report = self.role_dir(name) / "digests.json"
        if mode == "copy":
            where = ["--entry", str(entry.path), "--expect", entry.manifest_sha256, "--dest", str(path)]
        else:
            where = ["--copy", str(path), "--expect", entry.manifest_sha256]
        argv = [str(self.plan.toolchain.python), "-m", CACHE_ROLE_MODULE, mode, *where,
                "--digests-out", str(report)]
        self.start_role(name, argv, shape, owner="harness")
        outcomes = self.wait_roles()
        fault = stage_fault(outcomes)
        if fault is not None:
            refused = self.refusal(name, outcomes, (DIGEST_MISMATCH_EXIT,))
            if refused is not None and mode == "copy":
                raise StageFault.infra(f"cache entry {what} is corrupt: {refused.reason}",
                                       retryable=False, entry=what, role=name)
            if refused is not None:
                raise StageFault.infra(f"the node-local copy of cache entry {what} changed since it "
                                       f"was verified: {refused.reason}", retryable=True, entry=what,
                                       role=name)
            doing = "could not be copied to the node" if mode == "copy" else "could not be checked"
            raise StageFault(fault.model_copy(update={"reason": f"cache entry {what} {doing}: {fault.reason}",
                                                      "evidence": {**fault.evidence, "entry": what}}))
        reported = read_digests(report)
        if reported is None:
            raise StageFault.infra(f"role {name} reported no digests of the copy of cache entry {what}",
                                   retryable=True, entry=what, role=name)
        return reported

    def _published(self, kind: str, key: str, expect: str) -> CacheEntry:
        """Cache entry *kind*/*key* as published at the ENTRY.json digest *expect*. Raise as
        ``materialize`` says for an entry that is not published, is corrupt, or has another
        digest."""
        assert self.cache is not None
        what = f"{kind}/{key}"
        try:
            entry = self.cache.lookup(kind, key)
        except CacheCorrupt as e:
            raise StageFault.infra(f"cache entry {what} is corrupt: {e}", retryable=False,
                                   entry=what) from e
        if entry is None:
            raise StageFault.infra(f"cache entry {what} is not published", retryable=True, entry=what)
        found = entry_sha256(entry)
        if found != expect:
            raise StageFault.infra(f"cache entry {what} has ENTRY.json digest {found}, not the "
                                   f"{expect} its producer recorded", retryable=False, entry=what,
                                   expected=expect, found=found)
        return entry

    def entry_file(self, kind: str, key: str, expect: str, path: str) -> tuple[Path, str]:
        """File *path* of cache entry *kind*/*key*, pinned to its ENTRY.json digest *expect*
        and recorded for the manifest, where the store holds it, with the sha256 the entry
        records for it. Nothing is read but the entry's manifest: a role that reads the file
        checks it against that digest itself (``node_image``, an image built on a parent). Raise
        as ``materialize`` does for the entry, and StageFault (infra, not retryable) for one
        that holds no such file."""
        if self.cache is None:
            raise RuntimeError("this stage context has no cache")
        check_sha256_hex(expect)
        entry = self._published(kind, key, expect)
        sha256 = next((f.sha256 for f in entry.manifest.files if f.path == path), None)
        if sha256 is None:
            raise StageFault.infra(f"cache entry {kind}/{key} holds no {path}", retryable=False,
                                   entry=f"{kind}/{key}")
        self._cache_entries[f"{kind}/{key}"] = expect
        return entry.path / path, sha256

    def record_image(self, role: str, sha256: str) -> None:
        """Record that role *role* ran the image whose file has the bare hex digest *sha256*."""
        self._image_digests[role] = check_sha256_hex(sha256)

    def node_image(self, key: str, expect: str) -> PinnedImage:
        """Images cache entry *key*, pinned to the digest *expect* of its ENTRY.json, and
        extracted on this node for the stage's containers to run from (``_pin``). The entry is
        never copied to the node: the extraction role reads its image file once, into the
        role's own directory, and refuses that copy unless its sha256 is the one the entry
        records, which a retry would find again (infra, not retryable). A second call for the
        same entry and digest returns the same image at once, before the store is asked
        anything. Raise StageFault (infra, retryable) once the stage is stopping or closed,
        before anything is read; otherwise raise as ``entry_file`` and ``_pin`` do."""
        self._refuse_while_off_thread(f"image entry images/{key} is pinned")
        if self.images is None:
            raise RuntimeError("this stage context has no image runtime")
        made = self._node_images.get((key, expect))
        if made is not None:
            return made
        if self.stopping or self._terminated:
            raise StageFault.infra(f"the stage is stopping, so image entry images/{key} was not pinned",
                                   retryable=True, entry=f"images/{key}")
        image_file, sif = self.entry_file("images", key, expect, self.images.image_file)
        pinned = self._pin(image_file, sif, stored=True)
        self._node_images[(key, expect)] = pinned
        return pinned

    def pin_file(self, image_file: Path, *, sha256: str, contestant: bool = False,
                 stored: bool = False) -> PinnedImage:
        """The image file *image_file*, which no cache entry holds (the images stage's own build
        or import, a package stage's submission image, or a published one), extracted for the
        stage's containers to run from. *sha256* is the digest the role that wrote the file
        reported, or the one the file must have: the extraction role copies the file and refuses
        the copy unless it has that digest, so the directory is keyed by the bytes that were
        extracted and this process reads none of them. A file with another digest was written
        by a role of this attempt, or changed since it was published, and its refusal is
        retried. *contestant* marks an image that holds a contestant's code, a submission's,
        which only the contestant's own containers run; a tree once pinned so stays so for the
        stage. *stored* marks a file a retry would read unchanged, a published file the plan
        names, whose refusal is then not retried. Raise as ``_pin`` does."""
        self._refuse_while_off_thread("an image file is pinned")
        return self._pin(image_file, sha256, stored=stored, contestant=contestant)

    def _pin(self, image_file: Path, sif_sha256: str, *, stored: bool,
             contestant: bool = False) -> PinnedImage:
        """The image of *image_file*, whose sha256 must be *sif_sha256*, pinned on the node: the
        one this context pinned before, from its memo; else one a context that could not remove
        it left, if the image runtime finds it unchanged (``ImageRuntime.left``); else a fresh
        pin by a harness role, ``extract-<12 hex>``, through the image runtime's ``pin_role``,
        which reads the file once, into the role's own directory, refuses the copy unless its
        sha256 is *sif_sha256*, and only then pins it, from the copy. *stored* says the file is a
        store entry's, whose refusal a retry would meet again. *contestant* marks the image as
        one holding a contestant's code, and a mark stays with the image for the stage: a later
        pin of the same bytes returns it marked whatever it asks. The role gets the stage's CPUs
        and its memory less the runner's, no GPU and no disk: the stage's declared disk, checked
        when the job starts, is the one budget for its images and its work. An image lives for
        the stage that pinned it, a left one included: ``close`` removes it, so a later stage of
        the job pins the image again, and each stage's declared disk holds its own images only.
        The check of a left image catches every change an unprivileged process makes to it, but
        not one that rewrites its record too, which only a host process of the job's user could:
        no contestant code runs as one (``start_role``).

        Raise RuntimeError when the context has no image runtime or no scratch, or when a pin
        is needed while roles of the stage still run (the role is waited for with the stage's
        others, so images are pinned before any long role starts). Raise ValueError for a
        *sif_sha256* that is not bare hex. Raise StageFault when the image file has another
        sha256 (infra, retryable unless *stored*), when the pin role fails otherwise (its roles'
        fault), when it pinned nothing (infra, retryable), or when the stage's memory leaves the
        role none (infra, not retryable); and as ``start_role`` does, before anything is made on
        the node, when the role may not start, such as once the stage is stopping or closed."""
        if self.images is None or self.scratch is None:
            raise RuntimeError("this stage context has no image runtime or no node-local scratch")
        images = self.images
        check_sha256_hex(sif_sha256)
        pinned = self._pinned.get(sif_sha256)
        if pinned is not None:
            if contestant and not pinned.contestant:
                pinned = replace(pinned, contestant=True)
                self._pinned[sif_sha256] = pinned
            return pinned
        left = images.left(sif_sha256, contestant=contestant)
        if isinstance(left, PinnedImage):
            self._pinned[sif_sha256] = left
            self.events.emit("image_reused", sif_sha256=sif_sha256, manifest_sha256=left.check_sha256)
            return left
        if left is not None:
            self.events.emit("image_reextracted", sif_sha256=sif_sha256, problem=left)
            images.remove_left(sif_sha256)
        name = self._fresh_name(f"extract-{sif_sha256[:12]}")
        role_shape = self._alone(name)
        own = self.role_dir(name)
        argv, env = images.pin_role(image_file, sha256=sif_sha256, own=own)
        started = self.clock()
        self.start_role(name, argv, role_shape, owner="harness", env=env, daemon=True)
        outcomes = self.wait_roles()
        fault = stage_fault(outcomes)
        if fault is not None:
            images.discard(own)
            refused = self.refusal(name, outcomes, (DIGEST_MISMATCH_EXIT,))
            if refused is not None:
                raise StageFault.infra(refused.reason, retryable=not stored, role=name,
                                       expected=sif_sha256)
            raise StageFault(fault)
        try:
            pinned = images.pinned(own, sha256=sif_sha256, contestant=contestant)
        except ValueError as e:
            raise StageFault.infra(f"role {name} {e}", retryable=True, role=name) from e
        self._pinned[sif_sha256] = pinned
        self.events.emit("image_extracted", sif_sha256=sif_sha256, manifest_sha256=pinned.check_sha256,
                         sec=round(self.clock() - started, 3))
        return pinned

    def _fresh_name(self, base: str) -> str:
        """*base*, or ``<base>.<n>`` for the first *n* from 2 that no role of the stage has, for a
        role the context starts itself, which a stage that caught its fault may need again."""
        name, n = base, 1
        while name in self._roles:
            n += 1
            name = f"{base}.{n}"
        return name

    def _alone(self, name: str) -> ResourceShape:
        """The shape of harness role *name*, which the context starts itself and waits for at once:
        an extraction (``_pin``) or a copy of a cache entry (``_cache_role``). It gets the stage's
        CPUs and its memory less the runner's, no GPU and no disk: the stage's declared disk,
        checked when the job starts, is the one budget for what the stage's roles write.

        Raise RuntimeError while a role of the stage has not been waited for, running or ended:
        the wait reports every such role, so it would wait for a running one and take an ended
        one's outcome, its fault or its success, from the stage. Images are pinned and entries
        copied before the stage starts its roles, or after it has waited for them. Raise
        StageFault (infra, not retryable) when the stage's memory leaves the role none, and as
        ``start_role`` does when the role may not start, such as once the stage is stopping or
        closed. Each refusal comes before the role's own directory is made; an extraction makes
        nothing on the node before it, while a copy's place has already been cleared and its
        parent made (``ContentCache.materialize``), which the next copy replaces anyway."""
        pending = [n for n in list(self._roles) if n not in self._outcomes]
        if pending:
            raise RuntimeError(f"role {name} runs alone, and roles {pending} have not been waited for; pin "
                               "images and copy cache entries before starting roles, or wait for them first")
        shape = self.stage.shape
        memory = shape.memory_gib - self.plan.settings.run.runner_memory_gib
        if memory < 1:
            raise StageFault.infra(f"the stage declares {shape.memory_gib} GiB, which leaves role "
                                   f"{name} none beside the runner's", retryable=False, role=name)
        role_shape = shape.model_copy(update={"gpus": 0, "gpu_kind": None, "memory_gib": memory,
                                              "disk_gib": 0})
        self._admit(name, role_shape, None)
        return role_shape

    def refusal(self, name: str, outcomes: Iterable[RoleOutcome], codes: Collection[int]) -> Refusal | None:
        """What harness role *name* refused, when by *outcomes* it exited with one of the
        statuses *codes*: that status and the reason on the last ``refused:`` line near the end
        of its stderr in the attempt's logs, scrubbed. None for a role that is not the
        harness's, whose code chooses its own status and stderr; and None when the role ended
        otherwise, gave no such line, or its log cannot be read. Every stage reads its roles'
        refusals here, and each decides what a status means."""
        code = next((o.exit_code for o in outcomes if o.name == name and o.owner == "harness"
                     and o.ending == "exited" and o.exit_code in codes), None)
        if code is None:
            return None
        try:
            with open(self.attempt_dir / "logs" / f"{name}.err", "rb") as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - _REFUSAL_TAIL_BYTES))
                tail = f.read().decode(errors="replace")
        except OSError:
            return None
        lines = [line for line in tail.splitlines() if line.startswith("refused: ")]
        return Refusal(exit_code=code, reason=self.scrubber.scrub(lines[-1].removeprefix("refused: "))) \
            if lines else None

    def _admit(self, name: str, shape: ResourceShape, share_gpus_of: RoleHandle | None, *,
               overlap: bool = False, share_memory_of: RoleHandle | None = None,
               seal_memory_gib: int = 0) -> None:
        """Raise unless role *name* may start now, as ``start_role`` says. An overlap role is
        counted as a shared one: its memory, and no CPU or GPU of its own. A role that shares
        another's memory (*share_memory_of*) is counted for nothing, and must be an overlap step
        whose target started cleanly; ValueError otherwise, writing nothing. A sealed container's
        role counts its seal's *seal_memory_gib* (``ContainerRuntime.seal_memory_gib``) with its
        own memory, as long as it runs."""
        if self.stopping or self._terminated:
            raise StageFault.infra(f"the stage is stopping, so role {name} was not started", retryable=True,
                                   role=name)
        if name in self._roles:
            raise ValueError(f"role {name} already started")
        if overlap and shape.gpus:
            raise ValueError(f"role {name}: an overlap role takes no GPU, since its step would share another "
                             "step's GPUs")
        if share_gpus_of is not None:
            self._check_share_target(name, share_gpus_of)
        if share_memory_of is not None:
            if share_gpus_of is None and not overlap:
                raise ValueError(f"role {name}: only an overlap step shares another role's memory")
            self._check_share_target(name, share_memory_of)
        live = [(n, h) for n, h in list(self._roles.items()) if n not in self._launch_errors and h.poll() is None]
        running = [(_with_seal(h.spec.shape, self._seal_memory.get(n, 0)), n in self._shared)
                   for n, h in live if n not in self._memory_shared]
        oversubscribed = [h.spec.shape for n, h in live if n in self._memory_shared]
        new = ([] if share_memory_of is not None
               else [(_with_seal(shape, seal_memory_gib), share_gpus_of is not None or overlap)])
        problem = budget_problem(self.stage.shape, [*running, *new],
                                 runner_memory_gib=self.plan.settings.run.runner_memory_gib,
                                 oversubscribed=[*oversubscribed, *([shape] if share_memory_of is not None else [])])
        if problem is not None:
            gateway = f" (its seal holds {seal_memory_gib} GiB beside it)" if seal_memory_gib else ""
            raise StageFault.infra(f"role {name}{gateway}: {problem}", retryable=False, role=name)
        if seal_memory_gib:
            self._seal_memory[name] = seal_memory_gib

    def _holding_secrets(self, env: Mapping[str, str], args: Sequence[str], what: str) -> list[str]:
        """Where *env* and the arguments *args* (called *what*) hold a secret, one granted to
        this stage or any value the scrubber knows: ``env NAME`` for a variable, ``what[i]``
        for an argument by its position. Never the values. A URL secret's host alone is no
        secret here (``Scrubber.holds``): it gives no access, and a seal's policy names it."""

        def holds(text: str) -> bool:
            return self.scrubber.holds(text) or any(s in text for s in self.secrets.values())

        return ([f"env {k}" for k in sorted(env) if holds(env[k])]
                + [f"{what}[{i}]" for i, arg in enumerate(args) if holds(arg)])

    def _write_env_file(self, name: str, file: str, env: Mapping[str, str]) -> Path:
        """Write *env* to a fresh 0600 file named *file* in role *name*'s directory and return
        its path. It is removed when the role is reported, or at once if its launch fails."""
        path = self.role_dir(name) / file
        path.unlink(missing_ok=True)
        write_env_file(path, env)
        self._env_files[name] = path
        return path

    def _container_problems(self, name: str, spec: ContainerSpec, owner: str) -> list[str]:
        """Why container *spec* may not run as role *name*. Its image must be one this context
        pinned (``node_image``, ``pin_file``), whoever owns the container, and a harness
        container's one that holds no contestant code (``PinnedImage.contestant``): a harness
        role there would run the contestant's interpreter and libraries, which could forge what
        it reports. Every path is resolved first, so a link is judged by where it leads. No
        container mounts a directory that holds the store or the checkout, such as ``/`` or the
        store's parent: the runtime checks only a path's form. No container writes a cache entry
        copy: its workdir and its writable mounts neither lie in nor hold the job's copies
        (``copies_root``), since a later stage of the job reuses a copy. Only an unsealed harness
        container is given a device node (``ContainerSpec.devices``). A contestant's workdir
        must lie in its own role directory, and each of its mounts there or, read-only, in a
        cache entry copy ``materialize`` made in this context, in a part of it the mount wall
        lets a contestant see (``contestant_mount_problem``: never a test split, the row ids or
        a full tool bundle); none in, or holding, the store, the checkout, this attempt's
        directory or its log directory. That keeps every run's records, the checkout, the logs,
        the other roles' directories with their env files, the job's other copies and the test
        data out of a contestant's view, also where the site put the scratch inside the store
        or the checkout."""
        contestant = owner == "contestant"
        problems = []
        if self._pinned.get(spec.image.file_sha256) != spec.image:
            problems.append(f"image {spec.image.ref} is not one this stage pinned; a container runs only "
                            "from an image extracted on this node (node_image, pin_file)")
        elif spec.image.contestant and not contestant:
            problems.append(f"image {spec.image.ref} holds a contestant's code, which only the contestant's own "
                            "containers run")
        if spec.devices and (contestant or spec.seal is not None or spec.contestant_payload):
            problems.append(f"only an unsealed harness container is given a device node; {name} names "
                            f"{', '.join(spec.devices)}")
        kept = {"the store": self.store.root, "the checkout": self.plan.code.checkout}
        if contestant:
            kept |= {"the attempt directory": self.attempt_dir, "the log directory": self.attempt_dir / "logs"}
        roots = {label: Path(root).resolve() for label, root in kept.items()}
        # Each path the container sees, with the mount it comes from; the workdir has none.
        paths: list[tuple[str, Path, Mount | None]] = [(f"mount {m.target} source", m.source, m)
                                                       for m in spec.mounts]
        if contestant:
            paths.insert(0, ("workdir", spec.workdir, None))
        own = self.role_dir(name).resolve() if contestant else None
        if self.scratch is not None:
            copies = copies_root(self.scratch, self.job_id).resolve()
            written = [("workdir", spec.workdir)] + [(f"mount {m.target} source", m.source) for m in spec.mounts
                                                       if not m.read_only]
            for what, path in written:
                resolved = Path(path).resolve()
                if resolved.is_relative_to(copies):
                    problems.append(f"{what} {path} is in a cache entry copy, which no container may write")
                elif copies.is_relative_to(resolved):
                    problems.append(f"{what} {path} holds the cache entry copies, which no container may write")
        for what, path, mount in paths:
            resolved = Path(path).resolve()
            for label, root in roots.items():
                if contestant and resolved.is_relative_to(root):
                    problems.append(f"{what} {path} is in {label}")
                elif resolved != root and root.is_relative_to(resolved):
                    problems.append(f"{what} {path} holds {label}")
            if own is None or resolved.is_relative_to(own):
                continue
            copy = next((c for c in self._copies if resolved.is_relative_to(c.path)), None)
            if mount is None:
                problems.append(f"{what} {path} is not in the role's own directory {own}")
            elif copy is None:
                problems.append(f"{what} {path} is not in the role's own directory {own} or in a cache entry "
                                "copy this stage made")
            elif (hidden := contestant_mount_problem(copy.kind, copy.key,
                                                     PurePosixPath(resolved.relative_to(copy.path)))) is not None:
                problems.append(f"{what} {path} is in {hidden}")
        return problems

    def _launch(self, spec: RoleSpec, share_gpus_of: RoleHandle | None, *,
                reserve_bytes: int | None = None) -> RoleHandle:
        """Launch *spec*, wait for its grant report (``_await_report``) and put its step under the
        memory watch. With *reserve_bytes*, its harness reserve, also sample the step's memory
        split, and let a breach end the contestant's largest process before the step. Its
        ``role_started`` event, or its ``role_launch_failed`` when it does not report, records how
        long each of the two waits took (``_Launch.waits``), so that live runs measure how long
        the scheduler takes to start a step. A launch that failed, but for one the stage's stop
        cut short, also carries what the scheduler says of the step
        (``RoleHandle.launch_evidence``): in its event as ``evidence``, and in its error, which
        is the role's ``launch_failed`` detail.

        With an allocation's end to keep to, the step's own time limit is the time left until
        ``deadline_cap``, whatever *spec* declares, so no limit the harness sets ends a role before
        the stage's deadline, which only grows; the stage's deadline stops it first."""
        if self.deadline_cap is not None:
            left = max(1, math.ceil(self.deadline_cap - self.clock()))
            spec = replace(spec, shape=spec.shape.model_copy(update={"time_sec": left}))
        name, shape = spec.name, spec.shape
        launch = _Launch(asked=self.clock())
        try:
            handle = self.launcher.launch(spec, share_gpus_of=share_gpus_of)
        except BaseException:
            # Nothing took the env file, and no report of the role will remove it.
            env_file = spec.env_file if spec.container is None else spec.container.env_file
            if env_file is not None:
                env_file.unlink(missing_ok=True)
                self._env_files.pop(name, None)
            raise
        self._roles[name] = handle
        if share_gpus_of is not None or spec.overlap:
            self._shared.add(name)
        if reserve_bytes is not None:
            self._splits[name] = _Split(reserve_bytes)
        try:
            report = self._await_report(name, handle, launch)
        except (TimeoutError, RoleFailed) as e:
            # Asked before the kill, while srun may still wait for the step.
            evidence = {} if self.stopping else handle.launch_evidence()
            error = "; ".join([str(e), *(f"{key}: {value}" for key, value in evidence.items())])
            self._launch_errors[name] = error
            self.events.emit("role_launch_failed", role=name, owner=spec.owner, error=error, evidence=evidence,
                             **launch.waits(self.clock()))
            handle.kill()
            return handle
        self._reports[name] = report
        assert launch.started is not None
        self._started_at[name] = launch.started
        self.events.emit("role_started", role=name, owner=spec.owner, step=report.step, host=report.host,
                         gpus=report.gpu_uuids,
                         cuda_visible_devices=report.cuda_visible_devices, mismatches=report.mismatches,
                         stripped_env=report.stripped_env, cpus=report.cpus, numa_nodes=report.numa_nodes,
                         numa_split=report.numa_split, deadline=self.deadline, **launch.waits(self.clock()))
        split = self._splits.get(name)
        if split is not None and not report.mismatches:
            cgroup = Path(report.cgroup_dir)
            source = self._ready_sources.get(name)
            split.sampler = SplitSampler(cgroup, ready_source=source)
            split.sampler.start()
            if self.watch is not None:
                split.killer = ContestantKiller(cgroup, ready_source=source)
                self.watch.watch(report.step, cgroup, shape.memory_gib * _GIB, handle.kill, role=name,
                                 end_one=split.killer.end_one, exiting=split.killer.exiting)
        elif self.watch is not None and not report.mismatches:
            self.watch.watch(report.step, Path(report.cgroup_dir), shape.memory_gib * _GIB, handle.kill, role=name)
        return handle

    def _await_report(self, name: str, handle: RoleHandle, launch: _Launch) -> RoleReport:
        """The grant report of role *name*, waited for in slices of at most a second, in two
        waits: for its step to start (``RoleHandle.started``), at most ``STEP_START_TIMEOUT_SEC``
        from the launch request, which is the scheduler's time; then for its report, at most
        ``REPORT_TIMEOUT_SEC`` from the start, which is the step wrapper's. A report ends both,
        and a step first seen started by its report started then. Once the start is seen the
        stage's deadline grows by the first wait (``_grow``). The first wait goes no further than
        ``deadline_cap``, the second no further than the deadline: one that reaches its bound ends
        there, after one last look. *launch* gets the time the start was seen. Raise TimeoutError
        when a wait ends or the stage is told to stop meanwhile, and RoleFailed when the step ended
        without reporting."""
        start_wait = STEP_START_TIMEOUT_SEC
        if self.deadline_cap is not None:
            start_wait = max(0.0, min(start_wait, self.deadline_cap - launch.asked))
        start_by = launch.asked + start_wait
        while True:
            if self.stopping:
                raise TimeoutError(f"the stage is stopping, so role {name} was not waited for")
            now = self.clock()
            if launch.started is None and handle.started():
                self._grow(name, launch, now)
            if launch.started is None:
                give_up = start_by
                cut = "the allocation's end" if start_wait < STEP_START_TIMEOUT_SEC else None
            else:
                report_wait = max(0.0, min(REPORT_TIMEOUT_SEC, self.deadline - launch.started))
                give_up = launch.started + report_wait
                cut = "the stage's deadline" if report_wait < REPORT_TIMEOUT_SEC else None
            try:
                report = handle.report(max(0.0, min(_REPORT_SLICE_SEC, give_up - now)))
            except TimeoutError:
                if self.clock() >= give_up:
                    raise TimeoutError(launch.too_late(name, give_up, cut=cut)) from None
                continue
            if launch.started is None:
                self._grow(name, launch, self.clock())
            return report

    def hold_until(self, until: float) -> bool:
        """Make the stage's deadline reach *until*, a time of ``clock``, growing it when it is
        earlier, unless *until* lies past ``deadline_cap``: a stage that owes a role all of a
        budget, such as a drive's whole wall, holds its deadline for it this way. Return whether
        the deadline now reaches *until*; a deadline that cannot stays where it was."""
        if until <= self.deadline:
            return True
        self._refuse_while_off_thread("the deadline is held")
        if self.deadline_cap is not None and until > self.deadline_cap:
            return False
        self.events.emit("deadline_held", until=until, was=self.deadline)
        self.deadline = until
        return True

    def _grow(self, name: str, launch: _Launch, started: float) -> None:
        """Record that role *name*'s step was seen started at *started*, and grow the stage's
        deadline by the time it took to start, up to ``deadline_cap``; a deadline already past the
        cap stays where it is."""
        launch.started = started
        wait = started - launch.asked
        self._launch_waits[name] = wait
        grown = self.deadline + wait
        if self.deadline_cap is not None:
            grown = min(grown, self.deadline_cap)
        self.deadline = max(self.deadline, grown)

    def _check_share_target(self, name: str, target: RoleHandle) -> None:
        """Raise StageFault unless the role *target* started with the grant it declared."""
        target_name = target.spec.name
        if target_name in self._launch_errors:
            why = self._launch_errors[target_name]
        elif target_name in self._reports and self._reports[target_name].mismatches:
            why = "; ".join(self._reports[target_name].mismatches)
        else:
            return
        raise StageFault.infra(f"role {name} cannot share the GPUs of {target_name}: {why}", retryable=True,
                               role=name, target=target_name)

    def wait_roles(self, *names: str, command_sec: float | None = None) -> list[RoleOutcome]:
        """Wait for every role not yet reported to end, and return how each ended, in start
        order. Roles still running at the deadline are killed (``time_limit``), and with
        *command_sec* so is each role whose command has run that many seconds since the stage
        saw it start (``command_started_at``), when that comes first. The time a runtime takes
        to start a container's command is the harness's and counts against no command: a role
        whose command has not started is held to the deadline alone, so a stop for its command's
        time never comes before its start marker, which would read as the runtime's failure
        (``runtime_failed``). When the allocation is stopping, or a role ends with a fault of its
        own, every other running role is asked to stop, and killed when the one grace period
        ends.

        With *names*, wait only for those roles, in that order, and leave the others running:
        a stage waits for its driver while the server it drives keeps serving. A name that was
        never started, whose ending was already reported, or that is named twice raises
        ValueError."""
        deadline = self.deadline
        if names:
            unknown = [n for n in names if n not in self._roles or n in self._outcomes]
            if unknown:
                raise ValueError(f"no started role left to wait for named {unknown}")
            if len(set(names)) < len(names):
                raise ValueError(f"roles named more than once: {list(names)}")
            waiting = list(names)
        else:
            waiting = [name for name in list(self._roles) if name not in self._outcomes]
        ended: set[str] = set()
        while len(ended) < len(waiting):
            # Signal before reporting: reporting a role can wait on the scheduler.
            if self.stopping:
                self._terminate_all()
            pending = [name for name in waiting if name not in ended]
            for name in pending:
                if self._roles[name].poll() is not None:
                    self._report_role(name, timed_out=False)
                    ended.add(name)
            pending = [name for name in waiting if name not in ended]
            if not pending:
                break
            # A role past its time ends for it before another's fault asks the rest to stop, so
            # its ending reads ``time_limit`` whatever else ended first.
            ends = {name: self._end_of(name, deadline, command_sec) for name in pending}
            now = self.clock()
            due = [name for name in pending if now >= ends[name]]
            if due:
                self._kill_and_report(due, timed_out=True)
                ended.update(due)
                continue
            if self.stopping or self._faulted:
                self._terminate_all()
            if self._terminated and now >= self._grace_until:
                self._kill_and_report(pending, timed_out=False)
                break
            grace = [self._grace_until - now] if self._terminated else []
            limit = min(_POLL_SEC, min(ends.values()) - now, *grace)
            self._roles[pending[0]].wait(max(0.0, limit))
        return [self._outcomes[name] for name in waiting if name in self._outcomes]

    def _end_of(self, name: str, deadline: float, command_sec: float | None) -> float:
        """When ``wait_roles`` kills role *name*, seen running: at *deadline*, or *command_sec*
        after its command was seen to start when that comes first. A container's stderr is
        looked at for its start marker until it is found (``command_started_at``)."""
        if command_sec is None:
            return deadline
        began = self.command_started_at(name)
        return deadline if began is None else min(deadline, began + command_sec)

    def _report_role(self, name: str, *, timed_out: bool) -> None:
        """Read the job's OOM counts and stop sampling the role's memory split, remove the role's
        env file if it is still there, wait for its capped logs to be whole, put the scrubbed log
        files of a role given secrets into the attempt's ``logs/``, then record and log how role
        *name* ended. The counts come first, while the role's step cgroup may still be there
        (``_own_oom``); scrubbing a long log can outlast it."""
        ended_at = self.clock()
        job_ooms = self.watch.job_ooms() if self.watch is not None else None
        split = self._splits.get(name)
        if split is not None and split.sampler is not None:
            split.sampler.stop()
        if split is not None and split.killer is not None:
            split.killer.close()
        self._roles[name].close_logs(LOG_CLOSE_SEC)
        # The step wrapper unlinks a role's env file first; this covers a step that died before
        # its wrapper ran.
        env_file = self._env_files.pop(name, None)
        if env_file is not None:
            env_file.unlink(missing_ok=True)
        handle = self._roles[name]
        if name in self._private_logs:
            self._private_logs.discard(name)
            self._publish_logs(name, handle.spec.log_dir)
        if name in self._scrubbed_logs:
            self._scrubbed_logs.discard(name)
            for suffix in _LOG_FILES:
                path = handle.spec.log_dir / f"{name}{suffix}"
                if path.exists():
                    self.scrubber.scrub_file(path, cut=True)
        sample = self._sample(name)
        start = self._job_ooms_at_start
        job_oom = None if start is None or job_ooms is None else job_ooms.own > start.own
        own_oom = self._own_oom(name, sample, job_ooms)
        ending = self._ending(name, handle, handle.poll(), timed_out, sample,
                              own_limit=own_oom is True and job_oom is False, ended_at=ended_at)
        outcome = replace(ending, job_oom=job_oom, own_oom=own_oom, pool=self._pool(handle),
                          pool_at_end=self._pool_at_end(handle), disk_at_end=self._disk_at_end(handle))
        peak = self._pool_peak(handle)
        rule = self._ceilings.get(name)
        if split is not None:
            outcome = replace(outcome, split=self._split_evidence(name, split, outcome, ended_at))
        self._outcomes[name] = outcome
        if outcome.ending != "stopped" and classify_role(outcome) is not None:
            self._faulted = True
        self.events.emit("role_ended", role=name, exit_code=outcome.exit_code, ending=outcome.ending,
                         oom_kills=outcome.oom_kills, detail=outcome.detail, own_oom=outcome.own_oom,
                         job_oom=outcome.job_oom, step_ooms=None if sample is None else sample.ooms,
                         others_tasks=_counted(outcome.pool_at_end, "others_tasks"),
                         node_tasks=_counted(outcome.pool_at_end, "node_tasks"),
                         walk_uid_tasks=None if outcome.pool is None else outcome.pool.uid_tasks,
                         walk_step_tasks=None if outcome.pool is None else outcome.pool.step_tasks,
                         pool_peak=None if peak is None else asdict(peak),
                         ceiling=None if rule is None else rule.ceiling,
                         accounting=None if rule is None else rule.accounting,
                         job_ooms_at_start=_counts(start), job_ooms=_counts(job_ooms),
                         split=None if outcome.split is None else asdict(outcome.split),
                         disk_at_end=None if outcome.disk_at_end is None else asdict(outcome.disk_at_end))

    def _publish_logs(self, name: str, local: Path) -> None:
        """Put role *name*'s log files from its node-local directory *local* into the attempt's
        ``logs/``, each with every value the scrubber knows replaced."""
        logs = self.attempt_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        for suffix in _LOG_FILES:
            src = local / f"{name}{suffix}"
            if src.exists():
                self.scrubber.scrub_file(src, to=logs / src.name, cut=True)

    def _kill_and_report(self, names: Sequence[str], *, timed_out: bool) -> list[RoleOutcome]:
        """Kill every one of *names* still running, then wait for them, report those not yet
        reported, and return how those ended."""
        survivors = {name for name in names if self._roles[name].poll() is None}
        for name in survivors:
            self._roles[name].kill()
        for name in survivors:
            self._roles[name].wait(STOP_GRACE_SEC)
        reported = []
        for name in names:
            if name not in self._outcomes:
                self._report_role(name, timed_out=timed_out and name in survivors)
                reported.append(self._outcomes[name])
        return reported

    def _terminate_all(self) -> None:
        """Ask every running role to stop, once and with one launcher call, so that a slow
        signal costs one call and their grace periods overlap and end together."""
        if self._terminated:
            return
        self._terminated = True
        self._grace_until = self.clock() + STOP_GRACE_SEC
        running = {name: handle for name, handle in list(self._roles.items()) if handle.poll() is None}
        self._signalled.update(running)
        self.launcher.terminate(list(running.values()))

    def contestant_fault(self) -> FaultRecord | None:
        """The contestant fault the stage has decided so far, or None. It is the
        ``stage_fault`` of the roles reported so far, when that is the contestant's and every
        role not reported yet is still running and was asked to stop. Such a role can only read
        "stopped", which adds nothing to another role's fault; a role not asked to stop could
        still end with a fault of its own and make the stage's fault infra. This reads the
        stage's decision from its roles, so it holds only for a stage that, once a contestant
        role fault is decided, raises the ``stage_fault`` of its roles and no infra fault of its
        own (``Stage.run``). For a stage that decides its contestant faults itself it is the one
        the stage declared, and None before it does. Another thread may call it."""
        if stage_class(self.stage.kind).decides_contestant_faults:
            return self._declared
        roles, outcomes, signalled = list(self._roles.items()), dict(self._outcomes), set(self._signalled)
        if any(name not in outcomes and (name not in signalled or handle.poll() is not None)
               for name, handle in roles):
            return None
        fault = stage_fault(list(outcomes.values()))
        return fault if fault is not None and fault.fault_class == "contestant" else None

    def declare_contestant_fault(self, fault: FaultRecord) -> FaultRecord:
        """Record *fault*, a contestant fault, as the one the stage is about to raise, and return
        it: ``contestant_fault`` reads it for a stage whose class sets
        ``decides_contestant_faults``. Raise ValueError for a fault that is not the contestant's,
        or for a stage whose class does not set it."""
        if not stage_class(self.stage.kind).decides_contestant_faults:
            raise ValueError(f"stage {self.stage.name} does not decide its contestant faults itself")
        if fault.fault_class != "contestant":
            raise ValueError(f"a {fault.fault_class} fault is not a contestant fault to declare")
        self._declared = fault
        return fault

    def declare_no_retry(self, fault: FaultRecord) -> None:
        """Record *fault*, infra and never retried, as the fault of any end of the stage from now
        on: the stage has done work a retry would do again, as the agent build once its commit
        point wrote the bundle. The runner reads it through ``no_retry_fault``: a SIGTERM that ends
        the attempt before the stage returns, and any failure of the attempt that would be retried,
        end it with this fault instead. Another thread may read it. It is also written into the
        attempt's status on the store at once (*on_no_retry*), so a runner killed outright, which
        writes no record of its own, still leaves it for the orchestrator (``reconcile.job_fault``);
        whatever that write raises propagates, since the stage must not go on past this point
        without the declaration on the store, and the attempt then ends with *fault*. Raise
        ValueError for a contestant fault or one that would be retried."""
        if fault.fault_class != "infra" or fault.retryable:
            raise ValueError(f"only an infra fault that is never retried can be declared; this one is "
                             f"{fault.fault_class} and {'retried' if fault.retryable else 'never retried'}")
        self._no_retry = fault
        if self._on_no_retry is not None:
            self._on_no_retry(fault)

    def no_retry_fault(self) -> FaultRecord | None:
        """The fault ``declare_no_retry`` recorded, or None before."""
        return self._no_retry

    def sample_split(self, name: str) -> list[SplitSample]:
        """Sample the memory split of role *name*, started with a harness reserve, once more now,
        and return every sample kept, oldest first: [] for another role, or for one whose grant
        report never came or was refused."""
        split = self._splits.get(name)
        if split is None or split.sampler is None:
            return []
        split.sampler.sample_once()
        return split.sampler.samples()

    def stop_role(self, name: str) -> RoleOutcome:
        """Stop role *name* as the stage planned, such as a server once its drive is over, and
        return how it ended: SIGTERM through the launcher, then SIGKILL once ``STOP_GRACE_SEC``
        has passed. A role stopped this way ends ``planned_stop``, which is no fault of its own
        (``classify_role`` reads it as a clean exit); what happened before the stop still counts,
        such as an OOM kill, a breach of its memory or its disk cap, its pids limit, or a runtime
        that never started its command, all read before the SIGTERM (``_AtStop``). Whatever ends
        it once it was asked, a kernel OOM kill in its grace period among it, is the stop. A role
        that ended before it was asked keeps its own ending, and so
        does one the allocation's shutdown stopped. Raise ValueError for a name that was never
        started or is already reported."""
        if name not in self._roles or name in self._outcomes:
            raise ValueError(f"no started role left to stop named {name}")
        handle = self._roles[name]
        if handle.poll() is None:
            self._planned.add(name)
            report = self._reports.get(name)
            step = report.step if report is not None else None
            # What a role does once asked to stop is not what it did while it ran: its memory
            # breach, its pids count, its disk and its pool are read as they were before.
            breach = self.watch.breach(step) if self.watch is not None and step is not None else None
            self._at_stop[name] = _AtStop(at=self.clock(), breach=breach,
                                          pids_hits=self._pids_hits(handle, step, report),
                                          oom_kills=self._oom_kills(name, report))
            watched = handle.spec.owner == "contestant" and handle.spec.start_marker is not None
            if self.watch is not None and step is not None and watched:
                self.watch.unwatch_disk(step)
            self.launcher.terminate([handle])
            if handle.wait(STOP_GRACE_SEC) is None:
                handle.kill()
                handle.wait(STOP_GRACE_SEC)
        self._report_role(name, timed_out=False)
        return self._outcomes[name]

    def command_started(self, name: str) -> bool:
        """Whether the command of role *name* started: its start marker is a whole line of the
        first ``MARKER_SEARCH_BYTES`` of its stderr, or it runs with no marker (a role on the
        host). A seal's started file says only that the runtime was launched, so a stage asks
        this before it blames a contestant's server for how its start went. Raise ValueError for
        a name that was never started."""
        handle = self._roles.get(name)
        if handle is None:
            raise ValueError(f"no started role named {name}")
        marker = handle.spec.start_marker
        return marker is None or _marker_seen(handle.spec.log_dir / f"{name}.err", marker)[0]

    def command_started_at(self, name: str) -> float | None:
        """When the stage saw the command of role *name* start, on ``clock``. For a container it
        is the first look that found its start marker (``command_started``) while the role was
        not yet reported: ``wait_roles`` looks on each pass when it waits with a *command_sec*,
        ``wait_command_started`` on each of its passes, and a call looks then. That is never
        earlier than the command's start, so a limit counted from it leaves the command all of its
        time, and a command that ended before a pass of those waits found it ran for less than
        one pass. For a role with no marker it is when its step was seen started
        (``role_started_at``), a little before its step wrapper starts the command. None while no
        look found the marker, and for a role that never reported. Raise ValueError for a name
        that was never started."""
        handle = self._roles.get(name)
        if handle is None:
            raise ValueError(f"no started role named {name}")
        if handle.spec.start_marker is None:
            return self._started_at.get(name)
        if name not in self._marker_found_at and name not in self._outcomes and self.command_started(name):
            self._marker_found_at[name] = self.clock()
        return self._marker_found_at.get(name)

    def wait_command_started(self, name: str) -> float | None:
        """Wait until the command of role *name* is seen started, looking every ``_POLL_SEC``,
        and return when (``command_started_at``); None once the role has ended, the stage is
        told to stop, or the deadline passes first. Only the deadline bounds the wait: the time a
        runtime takes to start a container's command is the harness's, and a budget of the
        command's own counts from what this returns. Raise ValueError for a name that was never
        started."""
        handle = self._roles.get(name)
        if handle is None:
            raise ValueError(f"no started role named {name}")
        while (began := self.command_started_at(name)) is None:
            now = self.clock()
            if handle.poll() is not None or self.stopping or now >= self.deadline:
                return None
            handle.wait(min(_POLL_SEC, self.deadline - now))
        return began

    def stop_roles(self) -> None:
        """Make ``wait_roles`` stop every role; the runner calls it on shutdown."""
        self.stopping = True

    def release(self, image: PinnedImage) -> None:
        """Remove *image*, which this context pinned, from the node now (``ImageRuntime.remove``),
        not when the context closes: for a stage done with an image before it ends, so that its
        declared disk holds it only for as long as the stage uses it. No other stage reads it (each
        pins its own), and a later pin of the same bytes in this stage pins them again.

        Raise RuntimeError while a role of the stage has not been waited for, since it may run from
        the tree; ValueError for an image this context does not hold pinned; and StageFault (infra,
        retryable) when the tree cannot be removed, which the stage's disk would then have to hold:
        it stays pinned, and ``close`` tries again."""
        self._refuse_while_off_thread(f"image {image.file_sha256} is released")
        pending = [n for n in self._roles if n not in self._outcomes]
        if pending:
            raise RuntimeError(f"image {image.file_sha256} is released while roles {pending} have not been waited "
                               "for, and one may run from its tree; wait for them first")
        held = self._pinned.get(image.file_sha256)
        if held is None or held.ref != image.ref:
            raise ValueError(f"image {image.file_sha256} is not pinned in this stage")
        assert self.images is not None
        try:
            self.images.remove(held)
        except OSError as e:
            raise StageFault.infra(f"the tree of image {image.file_sha256} could not be removed: {e}",
                                   retryable=True, sif_sha256=image.file_sha256) from e
        del self._pinned[image.file_sha256]
        for memo, pinned in list(self._node_images.items()):
            if pinned.file_sha256 == image.file_sha256:
                del self._node_images[memo]
        self.events.emit("image_removed", sif_sha256=image.file_sha256)

    def close(self) -> list[RoleOutcome]:
        """Stop every role still running, within one grace period, report every role not yet
        reported, and return how those ended, in start order: the roles the stage never waited
        for. Then remove the role directories this context made, and every image it pinned
        (``node_image``, ``pin_file``) with its manifest: an extracted tree lives for the stage
        that pinned it, so a stage's declared disk holds its own trees only, and a later stage of
        the job extracts the image again. The contexts of a job run one at a time. Safe to call
        again. A directory that cannot be removed is logged and left: the next context to use it
        empties it first, and one that pins a tree left so reuses it after checking it against
        its manifest, then removes it when it closes in turn."""
        self._refuse_while_off_thread("the stage is closed")
        if self.running_roles():
            self._terminate_all()
            for handle in list(self._roles.values()):
                if handle.poll() is None:
                    handle.wait(max(0.0, self._grace_until - self.clock()))
        self._terminated = True
        reaped = self._kill_and_report(list(self._roles), timed_out=False)
        while self._role_dirs:
            path = self._role_dirs.pop()
            try:
                _remove(path)
            except OSError as e:
                log.warning("role directory %s could not be removed: %s", path, e)
        while self._pinned:
            _, pinned = self._pinned.popitem()
            assert self.images is not None
            try:
                self.images.remove(pinned)
            except OSError as e:
                log.warning("extracted image %s could not be removed: %s", pinned.ref, e)
                continue
            self.events.emit("image_removed", sif_sha256=pinned.file_sha256)
        self._node_images.clear()
        return reaped

    def _sample(self, name: str) -> MemorySample | None:
        """The memory watch's last sample of role *name*'s step; None when there is none."""
        report = self._reports.get(name)
        if self.watch is None or report is None:
            return None
        return self.watch.last(report.step)

    def _step_of(self, name: str) -> str | None:
        """The directory of role *name*'s step under the job cgroup, ``step_<n>``; None when the
        role never reported or its cgroup is not a step of the watched job."""
        report = self._reports.get(name)
        job = self.watch.job_dir if self.watch is not None else None
        if report is None or job is None:
            return None
        cgroup = Path(report.cgroup_dir)
        if not cgroup.is_relative_to(job):
            return None
        step = cgroup.relative_to(job).parts[0]
        return step if _STEP_DIR.fullmatch(step) else None

    def _own_oom(self, name: str, sample: MemorySample | None, now: JobOoms | None) -> bool | None:
        """Whether role *name*'s own memory limit was reached (``RoleOutcome.own_oom``), from
        the first of these that is left when the role is reported. A step cgroup's counts
        start at 0.
        1. The role's step cgroup, while it is still there: its own count.
        2. The memory sample that saw a kill: the step's own count in it.
        3. Where the substrate has a job cgroup (``MemoryWatch.job_dir``), the job's count of
           limits reached below it, which outlives the steps: a rise names
           this role's step only when every other step there at the stage's start (the batch
           and extern steps, an earlier stage's) and every other role step of the stage is
           still there with its count unchanged, and no step came and went unseen: every role
           of the stage has a known step, and their step ids leave no gap.
        Anything else is None."""
        step = self._step_of(name)
        if now is not None and step is not None and step in now.steps:
            return now.steps[step] > 0
        if sample is not None and sample.oom_kills > 0:
            return None if sample.ooms is None else sample.ooms > 0
        start = self._job_ooms_at_start
        if start is None or now is None:
            return None
        if now.below <= start.below:
            return False
        others = {other: self._step_of(other) for other in list(self._roles) if other != name}
        if step is None or None in others.values():
            return None
        steady = {**start.steps, **dict.fromkeys(others.values(), 0)}
        steady.pop(step, None)
        if {s: now.steps.get(s) for s in steady} != steady or set(now.steps) - set(steady) - {step}:
            return None
        ids = sorted(int(s[len("step_"):]) for s in [step, *others.values()])
        if len(ids) != ids[-1] - ids[0] + 1:
            return None
        return True

    def _split_evidence(self, name: str, split: _Split, outcome: RoleOutcome, ended_at: float) -> SplitEvidence:
        """The build rule's evidence for role *name*, seen to end at *ended_at*: its step's split
        samples, its OOM kills at the end, and the runner's kill of the step (``breach``, with
        ``kill_reason``), its one memory kill that can end the role. The runner's kills of single
        processes (``MemoryWatch.kills``) leave the role running and go in as a record only."""
        report = self._reports.get(name)
        singles: list[MemorySample] = []
        breach, reason = None, None
        if self.watch is not None and report is not None:
            singles, breach, reason = (self.watch.kills(report.step), self.watch.breach(report.step),
                                       self.watch.kill_reason(report.step))
        return split_evidence(split.sampler.samples() if split.sampler is not None else [], ended_at=ended_at,
                              final_oom_kills=outcome.oom_kills,
                              runner_kill_at=None if breach is None else breach.at,
                              reserve_bytes=split.reserve_bytes, process_kills=[k.at for k in singles],
                              step_kill_reason=reason)

    def _ending(self, name: str, handle: RoleHandle, code: int | None, timed_out: bool,
                sample: MemorySample | None, *, own_limit: bool, ended_at: float) -> RoleOutcome:
        """How role *name*, seen to end at *ended_at*, ended. *own_limit* says its step reached
        its own memory limit and not the job's: the own-limit rule then decides a sealed role
        before its seal's record does. A contestant container ended ``contestant_pids`` only
        when its pids limit count rose within ``OOM_EVIDENCE_MAX_AGE_SEC`` of its end, or of its
        stop for one the stage stopped (``within_evidence_age``), the build sandbox and a graded
        server alike: a fork bomb it survived says nothing of a later end."""
        report = self._reports.get(name)
        step = report.step if report is not None else None
        at_stop = self._at_stop.get(name)
        if at_stop is not None:
            breach, hits, end_at = at_stop.breach, at_stop.pids_hits, at_stop.at
        else:
            breach = self.watch.breach(step) if self.watch is not None and step is not None else None
            hits, end_at = self._pids_hits(handle, step, report), ended_at
        if hits is not None and not within_evidence_age(hits.rose_at, end_at):
            hits = None
        disk = self._disk_breach(handle, step)
        oom_kills = sample.oom_kills if sample is not None else 0
        owner = handle.spec.owner
        if name in self._signalled:
            return RoleOutcome(name, owner, code, "stopped", oom_kills)
        if name in self._launch_errors:
            return RoleOutcome(name, owner, code, "launch_failed", oom_kills, self._launch_errors[name])
        if hits is not None and hits.count:
            # Its processes may have left the runtime, its seal or the step itself unable to fork,
            # so every reading that would make its end infra comes after this one, a lost log's
            # among them.
            return RoleOutcome(name, owner, code, "contestant_pids", oom_kills,
                               f"its step reached its pids limit {hits.count} times")
        lost = handle.logs_lost()
        if lost is not None:
            # Its step wrote into a broken pipe, whatever its own code did.
            return RoleOutcome(name, owner, code, "log_lost", oom_kills, lost)
        unstarted = self._runtime_failure(name, handle, code, report)
        if unstarted is not None:
            return RoleOutcome(name, owner, code, "runtime_failed", oom_kills, unstarted)
        if timed_out:
            return RoleOutcome(name, owner, code, "time_limit", oom_kills)
        if breach is not None:
            detail = f"footprint {breach.footprint} bytes over {handle.spec.shape.memory_gib} GiB"
            return RoleOutcome(name, owner, code, "memory_limit", oom_kills, detail)
        if disk is not None:
            return RoleOutcome(name, owner, code, "disk_limit", oom_kills, disk.detail(handle.spec.shape.disk_gib))
        if report is not None and report.mismatches:
            return RoleOutcome(name, owner, code, "grant_mismatch", oom_kills, "; ".join(report.mismatches))
        if self.stopping:
            # The scheduler signals every step of a cancelled job, so a role that ended first is
            # not worth a query; the shutdown is the cause. A role that exited 0 may have been
            # cut short, so it does not count as having finished either.
            return RoleOutcome(name, owner, code, "stopped", oom_kills)
        if name in self._planned:
            # The stage's own stop, which the scheduler may record as its own cancel, and which
            # ends a sealed role before its seal can record the payload's ending. Whatever ends
            # the role once it was asked to stop is the stop, a kernel OOM kill in its grace
            # period among it; only the kills seen before then count.
            kills = at_stop.oom_kills if at_stop is not None else 0
            if kills:
                return RoleOutcome(name, owner, code, "planned_stop", kills,
                                   f"step {step} had {kills} OOM kills before its stop")
            return RoleOutcome(name, owner, code, "planned_stop", 0)
        if code and handle.ended_by_scheduler():
            return RoleOutcome(name, owner, code, "scheduler_ended", oom_kills)
        # A step that reached its own memory limit, and not the job's, is decided by the
        # own-limit rule whatever the kernel killed, the seal's interceptor among them.
        unsealed = None if own_limit else self._seal_failure(name, code)
        if unsealed is not None:
            return RoleOutcome(name, owner, code, "runtime_failed", oom_kills, unsealed)
        if code and handle.end_state() is JobState.OUT_OF_MEMORY:
            # The kernel OOM-killed a process of the step. A kill after the watch's last sample
            # is recorded only in accounting.
            return RoleOutcome(name, owner, code, "exited", max(oom_kills, 1), f"step {step} ended OUT_OF_MEMORY")
        return RoleOutcome(name, owner, code, "exited", oom_kills)

    def _oom_kills(self, name: str, report: RoleReport | None) -> int:
        """The OOM kills in role *name*'s step so far: its cgroup's ``oom_kill`` count now, or
        the watch's last sample of it, whichever is higher; 0 when neither can be read."""
        sample = self._sample(name)
        seen = sample.oom_kills if sample is not None else 0
        if report is not None:
            try:
                seen = max(seen, read_keyed(Path(report.cgroup_dir) / "memory.events").get("oom_kill", 0))
            except (OSError, ValueError):
                pass
        return seen

    def _pool_at_end(self, handle: RoleHandle) -> PoolAtEnd | None:
        """For contestant container *handle* with a process ceiling, its step's or its commands',
        whose step has ended, the uid's tasks on the node now and the node's: the others' alone,
        its own and their zombies gone with the step, which decide whether it was starved
        (``starvation_fault``), with how its ceiling is counted. When they cannot be counted, a
        pool with no count, which leaves starvation not ruled out, never a crashed report. None
        for another role."""
        rule = self._ceilings.get(handle.spec.name)
        if handle.spec.owner != "contestant" or handle.spec.start_marker is None or rule is None:
            return None
        try:
            others: int | None = uid_tasks(os.getuid())
        except Exception:  # noqa: BLE001 -- uncounted is an answer of its own, never a crashed report
            log.exception("the uid's tasks on the node could not be counted at the end of role %s",
                          handle.spec.name)
            others = None
        return PoolAtEnd(at=self.clock(), others_tasks=others, rule=rule, node_tasks=node_tasks(),
                         pid_max=node_pid_max())

    def _disk_at_end(self, handle: RoleHandle) -> memory_watch.DiskAtEnd | None:
        """For contestant container *handle*, whose disk the watch caps and whose step has ended,
        what its writable directories hold and the free space of their filesystem now
        (``MemoryWatch.disk_at_end``), which decide whether others starved it of disk
        (``starvation_fault``); None for another role."""
        report = self._reports.get(handle.spec.name)
        if (self.watch is None or report is None or handle.spec.owner != "contestant"
                or handle.spec.start_marker is None):
            return None
        return self.watch.disk_at_end(report.step)

    def _pool(self, handle: RoleHandle) -> PoolSample | None:
        """The last sample of the process pool of contestant container *handle* before it ended
        or was stopped; None for another role, or when none was read."""
        report = self._reports.get(handle.spec.name)
        if self.watch is None or report is None or handle.spec.owner != "contestant":
            return None
        return self.watch.pool(report.step) if handle.spec.start_marker is not None else None

    def _pool_peak(self, handle: RoleHandle) -> PoolPeak | None:
        """The most tasks of the process pool of contestant container *handle*, and of its own,
        the walks read over its life (``MemoryWatch.pool_peak``), for its ``role_ended`` event;
        None for another role, or when none was read."""
        report = self._reports.get(handle.spec.name)
        if self.watch is None or report is None or handle.spec.owner != "contestant":
            return None
        return self.watch.pool_peak(report.step) if handle.spec.start_marker is not None else None

    def pids_hits(self, name: str) -> PidsHits | None:
        """How often the step of contestant container *name* reached its pids limit so far, and
        when the count was first read at its latest rise (``_pids_hits``); None for a role that
        is not a contestant container or has not reported, or whose count no one could read.
        Raise ValueError for a name that was never started."""
        if name not in self._roles:
            raise ValueError(f"no started role named {name}")
        report = self._reports.get(name)
        return self._pids_hits(self._roles[name], report.step if report is not None else None, report)

    def _pids_hits(self, handle: RoleHandle, step: str | None, report: RoleReport | None) -> PidsHits | None:
        """How often the step of contestant container *handle* reached its pids limit: the most
        the disk walks read while it ran, dated by the walk that read its latest rise, or its
        cgroup's count now, dated now, when that is higher (``pids_read``); None for a role that
        is not a contestant container, or whose count no one could read."""
        if handle.spec.owner != "contestant" or handle.spec.start_marker is None or report is None:
            return None
        walked = self.watch.pids_hits(step) if self.watch is not None and step is not None else None
        read_at, now = self.clock(), pids_limit_hits(Path(report.cgroup_dir))
        return walked if now is None else pids_read(walked, now, read_at)

    def _disk_breach(self, handle: RoleHandle, step: str | None) -> memory_watch.DiskSample | None:
        """The breach of the disk cap of the role *handle* started, as step *step*, once the
        walks of its directories are stopped, so none is recorded after its ending is read; None
        when it has none, or is not a contestant container, whose disk alone is walked."""
        container = handle.spec.start_marker is not None
        if self.watch is None or step is None or handle.spec.owner != "contestant" or not container:
            return None
        self.watch.unwatch_disk(step)
        return self.watch.disk_breach(step)

    def _runtime_failure(self, name: str, handle: RoleHandle, code: int | None,
                         report: RoleReport | None) -> str | None:
        """Why the runtime of role *name* failed before its command started, or None. That is
        a role with a start marker that ended nonzero, past its grant check and outside a
        shutdown, with no marker line in the first ``MARKER_SEARCH_BYTES`` of its stderr. Then
        the runtime's first FATAL line, scrubbed, is the reason, whatever the memory and time
        readings say: a runtime that ran out of memory or hung before the command started is
        never the command's failure."""
        marker = handle.spec.start_marker
        if marker is None or not code or self.stopping or (report is not None and report.mismatches):
            return None
        # A seal's earlier attempts, which it retried, failed for reasons of their own.
        attempt = self.runtime.attempt_line if self.runtime is not None and name in self._seals else None
        started, fatal = _marker_seen(handle.spec.log_dir / f"{name}.err", marker, attempt=attempt)
        if started:
            return None
        return self.scrubber.scrub(fatal) if fatal is not None else "no start marker on stderr"

    def _seal_failure(self, name: str, code: int | None) -> str | None:
        """Why the nonzero exit *code* of sealed role *name*, whose container started, is not
        its payload's own, or None when it is or the role is not sealed. It is the payload's
        only when the seal recorded the payload's ending once pasta returned, with the same
        status and the interceptor still running: a pasta or seal that ended first, or an
        interceptor that died while the payload ran (pip then finds no index), is the
        harness's failure, never the contestant's."""
        files = self._seals.get(name)
        if files is None or not code:
            return None
        ending = payload_ending(files)
        if ending is None:
            return "the seal recorded no ending of its payload: pasta or the seal ended before it"
        if not ending.interceptor_alive:
            return "the egress interceptor ended before the payload"
        if ending.exit_code != code:
            return f"the seal exited {code}, its payload {ending.exit_code}"
        return None


@dataclass(frozen=True)
class _Copy:
    """A node-local copy of a cache entry that ``materialize`` made."""

    kind: str
    key: str
    #: The copy's resolved path.
    path: Path


@dataclass
class _Split:
    """The memory split of a role started with a harness reserve."""

    #: The memory of the role's step its harness may hold, in bytes.
    reserve_bytes: int
    #: Samples the step's split from the role's grant report until the role is reported; None
    #: while it has no report, or when its grant was refused.
    sampler: SplitSampler | None = None
    #: Ends the contestant's largest process at a breach, for the memory watch, until the role is
    #: reported; None without a watch, a report, or a grant.
    killer: ContestantKiller | None = None


def _remove(path: Path) -> None:
    """Remove whatever is at *path*: a directory with everything in it, of any depth, or a file
    or link itself. A link is never followed, at *path* or inside the directory."""
    remove_tree(path)


def _marker_seen(err: Path, marker: str, *, attempt: str | None = None) -> tuple[bool, str | None]:
    """Whether *marker* is a whole line of the first ``MARKER_SEARCH_BYTES`` of the stderr
    file *err*, and the first line there that starts with ``FATAL``, if any. With *attempt*,
    the FATAL line is looked for only after the last line that starts with it: the lines of
    the last attempt."""
    try:
        with open(err, "rb") as f:
            head = f.read(MARKER_SEARCH_BYTES)
    except FileNotFoundError:
        return False, None
    lines = head.splitlines()
    last = lines
    if attempt is not None:
        starts = [i for i, line in enumerate(lines) if line.startswith(attempt.encode())]
        last = lines[starts[-1] + 1:] if starts else lines
    fatal = next((line.decode(errors="replace") for line in last if line.startswith(b"FATAL")), None)
    return marker.encode() in lines, fatal


def _counts(ooms: JobOoms | None) -> dict | None:
    """The job's own count, the count below it and each step's, for an event."""
    return None if ooms is None else {"own": ooms.own, "below": ooms.below, "steps": dict(ooms.steps)}


def _run(stage: Stage, ctx: StageContext) -> dict[str, str]:
    """Run *stage*; raise StageFault for anything that stops it from returning its outputs."""
    try:
        outputs = stage.run(ctx)
        running = ctx.running_roles()
        if running:
            raise StageFault.infra(f"stage {ctx.stage.name} returned with roles still running: "
                                   f"{', '.join(running)}", retryable=False, roles=",".join(running))
        return outputs
    except StageFault:
        raise
    except Exception as e:
        # Retried: an error the stage did not foresee may be transient, and a retry that repeats it
        # costs an allocation, never a score.
        raise StageFault.infra(f"stage {ctx.stage.name} raised {type(e).__name__}: {e}", retryable=True,
                               exception=type(e).__name__, message=str(e)) from e


def _close(ctx: StageContext) -> tuple[list[RoleOutcome], Exception | None]:
    """Close *ctx*: how the roles it reported ended, and the error if closing raised."""
    try:
        return ctx.close(), None
    except Exception as e:
        return [], e


def run_stage(stage: Stage, ctx: StageContext) -> dict[str, str]:
    """Run *stage* in *ctx* and return its outputs. However it ends, no role outlives it:
    the context is closed. Only StageFault is raised: the stage's own; the ``stage_fault`` of
    the roles that closing had to report, when the stage returned without waiting for them;
    an infra fault that is retryable for any other exception or a failed close, or one that is
    not retryable when the stage returned with roles still running. A close that fails while a
    StageFault is propagating is noted in that fault's evidence as ``close_exception`` and
    ``close_message``."""
    try:
        outputs = _run(stage, ctx)
    except BaseException as error:
        _, close_error = _close(ctx)
        if close_error is not None:
            log.warning("closing stage %s failed: %s", ctx.stage.name, close_error)
            if isinstance(error, StageFault):
                noted = {"close_exception": type(close_error).__name__, "close_message": str(close_error)}
                error.fault = error.fault.model_copy(update={"evidence": {**error.fault.evidence, **noted}})
        raise
    reaped, close_error = _close(ctx)
    if close_error is not None:
        raise StageFault.infra(f"closing stage {ctx.stage.name} raised {type(close_error).__name__}: "
                               f"{close_error}", retryable=True, exception=type(close_error).__name__,
                               message=str(close_error)) from close_error
    fault = stage_fault(reaped)
    if fault is not None:
        raise StageFault(fault)
    return outputs
