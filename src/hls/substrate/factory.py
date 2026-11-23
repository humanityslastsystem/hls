"""What the pipeline takes from the substrate it runs on.

A substrate is a scheduler that places allocations, a launcher that starts each role as a step of
its allocation, and a container runtime. ``Substrate`` holds everything the orchestrator, the
planner, the runner, the stage context and the step wrapper take from one, so no module outside a
substrate's own packages and this one imports them (``tests/unit/test_substrate_imports.py``).
Every member is required: a substrate that lacks one fails when it is built, not when a caller
first reaches the member. ``substrate`` gives the substrate a site's settings name
(``site.substrate``), built from them, so every caller asks with the settings it holds: the plan's,
or the ones the planner plans with.

A role's step takes what it needs, its id, its limit directory and the variables the container
runtime reads, from ``Step``, and never builds a ``Substrate``, since a step holds no settings. It runs under the
substrate its launcher names in the step's environment (``SUBSTRATE_VAR``); the Slurm launcher
names none, so a step whose environment holds no name is Slurm's."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Protocol

from pydantic import Field

from hls.settings import SiteSettings
from hls.spec.resources import ResourceShape
from hls.recipes.model import ImageDecl, RegistryImage, UvPin
from hls.substrate.base import JobEnd, Launcher, Scheduler
from hls.substrate.container import ContainerRuntime, ImageRuntime
from hls.substrate.grant import Measured
from hls.substrate.local.allocation import LocalPlacement
from hls.substrate.site_checks import SiteProblem
from hls.substrate.slurm.resources import SlurmPlacement
from hls.substrate.versions import ToolVersions


#: Where a substrate places an allocation beside its request shape, a value of its own kind
#: (``kind``), which the plan's allocation holds and the scheduler reads. Every kind answers
#: ``partition``, ``qos``, ``socket_tasks``, ``exclude`` and ``request_problem``, which the planner,
#: its retries and the run's records read.
Placement = Annotated[SlurmPlacement | LocalPlacement, Field(discriminator="kind")]


class BatchScript(Protocol):
    """Writes the batch script of one allocation submission, which starts the runner."""

    def __call__(self, *, python: Path, plan_dir: Path, allocation: int, submission: int, checkout: Path,
                 path: str, home: Path) -> str: ...


class MakePlacement(Protocol):
    """The substrate's placement of an allocation: the partition it goes to (None: the site's GPU
    partition), the QOS it runs under, the CPUs of each of its two socket tasks (None: one task)
    and the nodes it must not land on, each as far as the substrate places by it."""

    def __call__(self, *, partition: str | None, qos: str | None, socket_tasks: int | None,
                 exclude: Sequence[str]) -> Placement: ...


class CheckPlacement(Protocol):
    """Raise ValueError when the scheduler could not be given an allocation of *request* placed by
    *placement* on *site*, under the job name *job_name*."""

    def __call__(self, request: ResourceShape, site: SiteSettings, placement: Placement, *,
                 job_name: str) -> None: ...


class AllocationQOS(Protocol):
    """The QOS an allocation of *time_sec* seconds runs under on *site*. Raise
    ``hls.substrate.base.NoQOSHolds`` when none holds it."""

    def __call__(self, time_sec: int, site: SiteSettings, *, partition: str | None = None) -> str | None: ...


class SiteRequest(Protocol):
    """*shape* as *site* lets an allocation ask for it. Raise ValueError when it cannot be asked."""

    def __call__(self, shape: ResourceShape, site: SiteSettings, *, gpu_kind: str | None) -> ResourceShape: ...


class RecipeKey(Protocol):
    """The images cache key of *decl* as a substrate's image runtime builds it
    (``ImageRuntime.recipe_key``), given its parent's key (None for a base or a registry image)
    and the registry image its chain starts at, if it starts at one."""

    def __call__(self, decl: ImageDecl, *, parent_key: str | None, locks_dir: Path, uv: UvPin,
                 tasks_dir: Path, registry: RegistryImage | None = None) -> str: ...


class MakeLauncher(Protocol):
    """The launcher of the allocation this process runs in."""

    def __call__(self, *, python: str, scratch: Path, allocation_gpus: int,
                 allocation_gpu_uuids: list[str], pasta: Path | None,
                 upstream_proxy: str | None) -> Launcher: ...


@dataclass(frozen=True, kw_only=True)
class Substrate:
    """Everything the pipeline takes from one substrate, each member a value or a callable."""

    #: The substrate's name.
    name: str

    # The login node: the orchestrator and the planner.

    #: Makes the scheduler the orchestrator submits through and asks about jobs.
    scheduler: Callable[[], Scheduler]
    #: The batch script of one allocation submission.
    batch_script: BatchScript
    #: The placement of an allocation, which its plan holds.
    placement: MakePlacement
    #: Checks that the scheduler could be given an allocation, as the planner checks each.
    check_placement: CheckPlacement
    #: The QOS an allocation runs under, by its time.
    allocation_qos: AllocationQOS
    #: A request shape as the site lets it be asked for.
    site_request: SiteRequest
    #: Whether a file in a submission's directory is the runner's job log, as the scheduler names it.
    is_job_log: Callable[[str], bool]
    #: Returns a job id the scheduler gave, or raises ValueError for one it never gives.
    check_job_id: Callable[[str], str]
    #: Returns a node name the scheduler takes in an exclusion, or raises ValueError.
    check_node_name: Callable[[str], str]
    #: Whether the launcher sets or forbids a variable, so that no grant may set it.
    reserved_variable: Callable[[str], bool]
    #: Why the build sandbox's spawner could not make the commands' namespaces where the site runs
    #: its allocations, as the planner can tell, which plans no sandbox there
    #: (``AgentBuildStage.check_site``); None when it could, or when only the node an allocation
    #: lands on can tell, which the runner's seal check judges before the sandbox's stage runs
    #: (``hls.substrate.site_checks.userns_problem``).
    spawner_problem: Callable[[], SiteProblem | None]
    #: The per-uid kernel counts (``hls.substrate.ucounts.UCOUNTS``, by their ``/proc/sys/user``
    #: names) whose limit a graded server's seal holds it to on this substrate: the build's brief
    #: states a graded server's limit of these alone (``AgentBuildStage``).
    served_ucounts: frozenset[str]

    # The allocation: the runner.

    #: The id of the job a runner runs in, from the runner's environment.
    job_id: Callable[[Mapping[str, str]], str]
    #: The name the scheduler knows a node by, from its host name and the job's environment.
    node_name: Callable[[str, Mapping[str, str]], str]
    #: Both sources of a job's end time, by its id and the job's environment.
    read_job_end: Callable[[str, Mapping[str, str]], JobEnd]
    #: Why a job may not start a step yet; None once it may. Raises
    #: ``hls.substrate.base.SchedulerUnavailable`` or OSError when the scheduler does not answer.
    job_not_ready: Callable[[str], str | None]
    #: Which of the given jobs the scheduler reports ended, each asked for by its id.
    ended_jobs: Callable[[Collection[str]], set[str]]
    #: The cgroup of the job this process runs in, which holds the job's limits. Raises OSError
    #: or RuntimeError when it cannot be found.
    job_cgroup: Callable[[], Path]
    #: Why this node's cgroups are not laid out as the job's and its steps' limits are read
    #: (``job_cgroup``, ``step_cgroup``); None when they are.
    cgroup_problem: Callable[[], SiteProblem | None]
    #: Every reason this node could not run the network seal of a stage that seals a container,
    #: given the runner's environment; empty when it can. The runner asks before such a stage runs.
    seal_problems: Callable[[Mapping[str, str]], list[SiteProblem]]
    #: The launcher of the allocation the runner runs in.
    launcher: MakeLauncher
    #: Makes the container runtime that turns a container spec into a role's command line.
    runtime: Callable[[], ContainerRuntime]
    #: Makes the image runtime that pins images in a job's node-local scratch.
    images: Callable[[Path], ImageRuntime]
    #: The images cache key of a declaration as that image runtime builds it, which a plan names
    #: the entries it finds by (``ImageRuntime.recipe_key``).
    recipe_key: RecipeKey
    #: The file an images cache entry holds the image in for that image runtime
    #: (``ImageRuntime.image_file``).
    image_file: str
    #: The allocation's grant as the runner measures it, from the batch step's environment and the
    #: job's node-local scratch, whose free space it reads.
    measure_grant: Callable[[Mapping[str, str], Path], Measured]
    #: The versions of the node's tools, given the pasta the network seal runs (None: on PATH).
    measure_tools: Callable[[Path | None], ToolVersions]


def _slurm(site: SiteSettings) -> Substrate:
    """Slurm, with Apptainer as its container runtime (``hls.substrate.slurm.substrate``),
    imported when it is first asked for: its launcher imports the step wrapper, which asks for
    the substrate."""
    from hls.substrate.slurm.substrate import slurm

    return slurm(site)


def _docker(site: SiteSettings) -> Substrate:
    """One host's Docker (``hls.substrate.docker.substrate``), imported when it is first asked for."""
    from hls.substrate.docker.substrate import docker

    return docker(site)


#: Every substrate hls can run on, by the name ``site.substrate`` gives it, each built from the
#: site's settings.
SUBSTRATES: dict[str, Callable[[SiteSettings], Substrate]] = {"slurm": _slurm, "docker": _docker}


@dataclass(frozen=True, kw_only=True)
class Step:
    """What a role's step takes from the substrate it runs under."""

    #: The id the step's report names it by, from the step's environment.
    step_id: Callable[[Mapping[str, str]], str]
    #: The directory that holds the step's limits, from the cgroup path of a process in it
    #: (``hls.substrate.cgroup.own_cgroup``) and where cgroup v2 is mounted. Raises RuntimeError
    #: for a cgroup that is not a step's.
    step_cgroup: Callable[[str, Path], Path]
    #: Whether the step's container runtime reads a variable from the environment it starts in.
    runtime_reads: Callable[[str], bool]
    #: Whether a variable is one of the step's container runtime's own settings or one it copies
    #: into a container.
    runtime_setting: Callable[[str], bool]


#: The variable a launcher sets in a step's environment to name the substrate the step runs
#: under (a key of ``STEPS``); a step without it runs under Slurm, whose launcher sets none.
SUBSTRATE_VAR = "HLS_SUBSTRATE"


def _slurm_step() -> Step:
    """A Slurm step: its id from slurmd's variables, its limits on ``job_<id>/step_<n>/user``,
    Apptainer as its container runtime."""
    from hls.substrate.apptainer.argv import RUNTIME_PREFIXES, read_by_runtime
    from hls.substrate.cgroup import step_dir
    from hls.substrate.slurm.launcher import step_id

    return Step(step_id=step_id, step_cgroup=step_dir, runtime_reads=read_by_runtime,
                runtime_setting=lambda name: name.startswith(RUNTIME_PREFIXES))


def _docker_step() -> Step:
    """A role container's step (``hls.substrate.docker.step``), imported when a step asks for it."""
    from hls.substrate.docker.step import runtime_reads, runtime_setting, step_cgroup, step_id

    return Step(step_id=step_id, step_cgroup=step_cgroup, runtime_reads=runtime_reads,
                runtime_setting=runtime_setting)


#: The step of each substrate a launcher may name in ``SUBSTRATE_VAR``, built when it is asked for.
STEPS: dict[str, Callable[[], Step]] = {"slurm": _slurm_step, "docker": _docker_step}


def step(environ: Mapping[str, str]) -> Step:
    """The step of the substrate *environ*, a step's environment, names in ``SUBSTRATE_VAR``;
    Slurm's when it names none, since the Slurm launcher sets none. Raise RuntimeError for a name
    ``STEPS`` does not hold."""
    name = environ.get(SUBSTRATE_VAR, "slurm")
    if name not in STEPS:
        raise RuntimeError(f"{SUBSTRATE_VAR}={name!r} names no substrate hls runs a step under; it runs under "
                           f"{sorted(STEPS)}")
    return STEPS[name]()


def image_files() -> dict[str, str]:
    """The file an images cache entry holds an image in on each substrate
    (``ImageRuntime.image_file``), by the substrate's name in ``SUBSTRATES``: what a declaration
    that holds for an instance of any substrate, such as a stage kind's, names an image file by.
    Each image runtime is imported when this is first called."""
    from hls.substrate.apptainer.images import ApptainerImages
    from hls.substrate.docker.images import DockerImages

    return {"slurm": ApptainerImages.image_file, "docker": DockerImages.image_file}


def substrate(site: SiteSettings) -> Substrate:
    """The substrate *site* runs allocations on (``site.substrate``), built from its settings."""
    return SUBSTRATES[site.substrate](site)


def seal_memory_gib(site: SiteSettings) -> int:
    """GiB a sealed role's seal holds beside the role on *site*'s substrate
    (``ContainerRuntime.seal_memory_gib``), which a stage that seals keeps room for beside its roles,
    as it keeps the runner's."""
    return substrate(site).runtime().seal_memory_gib
