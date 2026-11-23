"""Docker on one host as a substrate (``hls.substrate.factory``): the local scheduler
(``hls.substrate.local``) places each allocation on the host, and the Docker launcher
(``hls.substrate.docker.launcher``) starts each of its roles as a container held to the
allocation's reservation.

A member that the Slurm shape gives no meaning on one host is not imitated:
- ``job_cgroup`` raises, as there is no job cgroup on a host; the runner then has no job-level OOM
  count from it (the launcher counts the allocation's in its parent cgroup).
- ``allocation_qos`` is None, since a host has no QOS, and ``site_request`` is the shape as
  declared, since a host has no GPU floor.
The runner on the host sees every GPU and has no memory cgroup, so the allocation's grant is
measured from its reservation (``measure_grant``)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import hls
from hls.settings import SiteSettings
from hls.spec.resources import GrantedResources, ResourceShape, gpu_kind_of
from hls.store.cache import CacheCorrupt, ContentCache
from hls.store.posix import PosixStore
from hls.substrate.base import JobEnd, Launcher, Refused, SubmitRequest
from hls.substrate.cgroup import CGROUP_ROOT
from hls.substrate.docker.api import DockerClient, DockerError
from hls.substrate.docker.images import DockerImages
from hls.substrate.docker.launcher import DockerLauncher, DockerRuntime, code_dirs, load_step_image, reserved_variable
from hls.substrate.docker.nvidia import DriverUnusable, driver_version
from hls.substrate.docker.step_image import docker_cli_problem, ensure_step_image, step_image_file, step_key
from hls.substrate.factory import Substrate
from hls.substrate.grant import Measured, list_gpus
from hls.substrate.local.allocation import LocalPlacement, check_node_name, check_placement, is_job_log, placement
from hls.substrate.local.batch import batch_script
from hls.substrate.local.inventory import SysInventory
from hls.substrate.local.records import StateDir, StateError, check_job_id
from hls.substrate.local.scheduler import (RESERVED_VARIABLES, LocalScheduler, ended_jobs, env_end_time, job_cgroup,
                                           job_end_time, job_id, job_not_ready, node_name, reservation, state_dir)
from hls.substrate.nproc import node_pid_max
from hls.substrate.site_checks import SiteProblem
from hls.substrate.versions import ToolVersions

_GIB = 2**30
#: The name the local scheduler's stopper registry gives the Docker stopper.
STOPPER = "docker"


def _no_qos(time_sec: int, site: SiteSettings, *, partition: str | None = None) -> str | None:
    return None


def _as_declared(shape: ResourceShape, site: SiteSettings, *, gpu_kind: str | None) -> ResourceShape:
    return shape


def _check_placement(site: SiteSettings, request: ResourceShape, placement: LocalPlacement, *,
                     job_name: str) -> None:
    """Raise ValueError when this host could never hold *request*, as its inventory reads now, or
    could not build the step image its roles run in (``step_image.docker_cli_problem``)."""
    problem = docker_cli_problem()
    if problem is not None:
        raise ValueError(f"{problem}; install the Docker CLI with buildx")
    check_placement(request, placement, SysInventory().read(kept_memory_gib=site.local.kept_memory_gib))


class _StepImageScheduler(LocalScheduler):
    """The local scheduler of a Docker host, which makes sure the store holds the step image before
    it submits an allocation, so no allocation spends its time building it
    (``step_image.ensure_step_image``). A step image that cannot be built refuses the submission, as
    no role of the allocation could run."""

    def __init__(self, site: SiteSettings, **kwargs) -> None:
        super().__init__(**kwargs)
        self._site = site

    def submit(self, request: SubmitRequest) -> str:
        site = self._site
        try:
            ensure_step_image(ContentCache(PosixStore(site.store_root)), socket=site.docker.socket,
                              buildkit_image=site.docker.buildkit_image, work_root=site.node_scratch)
        except (RuntimeError, OSError, CacheCorrupt) as e:
            raise Refused(f"the step image the allocation's roles run in is not in the store: {e}") from e
        return super().submit(request)


def _step_image_problem(site: SiteSettings) -> SiteProblem | None:
    """Why this allocation cannot load the step image: the store holds no entry for its recipe."""
    if step_image_file(ContentCache(PosixStore(site.store_root))) is None:
        return SiteProblem(reason=f"the store holds no step image images/{step_key()}, which the submission of "
                                  "every allocation builds first", retryable=False)
    return None


def measure_grant(environ: Mapping[str, str], scratch: Path, *,
                  run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                  affinity: Callable[[int], set[int]] = os.sched_getaffinity) -> Measured:
    """The grant of the allocation this process runs in, from the local scheduler's reservation
    of it: its GPUs, each of which ``nvidia-smi`` must list, all of one kind; its CPUs, which must
    be this process's affinity, as the supervisor pins the batch script to them; its memory; and
    the free space of *scratch*. Every role container is created with no swap, so the swap limit
    is 0. Raise RuntimeError when the reservation cannot be read."""
    try:
        held = reservation(StateDir(state_dir(environ)), job_id(environ))
    except (KeyError, OSError, ValueError, StateError) as e:
        raise RuntimeError(f"the allocation's reservation cannot be read: {type(e).__name__}: {e}") from None
    problems = []
    listed = list_gpus(run) if held.gpus else []
    if listed is None:
        problems.append("nvidia-smi failed, so the reserved GPUs cannot be identified")
        listed = []
    by_uuid = {g.uuid: g for g in listed}
    missing = [u for u in held.gpus if u not in by_uuid]
    if missing:
        problems.append(f"nvidia-smi does not list the reserved GPUs {missing}")
    kinds = {gpu_kind_of(by_uuid[u].name) for u in held.gpus if u in by_uuid}
    own = sorted(affinity(0))
    if own != sorted(held.cpus):
        problems.append(f"this process may run on CPUs {own}, not the reserved {sorted(held.cpus)}")
    granted = GrantedResources(gpus=len(held.gpus), gpu_kind=kinds.pop() if len(kinds) == 1 else None,
                               cpus=len(held.cpus), memory_gib=held.memory_gib,
                               disk_free_gib=shutil.disk_usage(scratch).free / _GIB)
    return Measured(granted=granted, gpu_uuids=list(held.gpus), swap_max_bytes=0, problems=problems)


def measure_tools(socket: Path, step_image_key: str, *, proc: Path = Path("/proc")) -> ToolVersions:
    """The versions of this host's tools: the Docker Engine's, as the daemon on *socket* reports
    it, the step image's cache key (``step_image.step_key``), the NVIDIA driver's, whose files
    every GPU role is given, and the kernel's; a version that cannot be read is None. Slurm,
    apptainer and pasta are none of a Docker host's."""
    try:
        with DockerClient(socket) as client:
            engine: str | None = client.version().version
    except DockerError:
        engine = None
    try:
        driver: str | None = driver_version(proc)
    except DriverUnusable:
        driver = None
    return ToolVersions(slurm=None, apptainer=None, pasta=None, kernel=os.uname().release, docker=engine,
                        step_image=step_image_key, nvidia_driver=driver)


#: The kernel settings that refuse an unprivileged user the namespaces the build sandbox's spawner
#: makes, or the mounts it makes in them, each with the value that refuses them and what to set.
SPAWNER_SWITCHES = {
    "kernel/apparmor_restrict_unprivileged_userns": ("1", "0"),
    "kernel/unprivileged_userns_clone": ("0", "1"),
    "user/max_user_namespaces": ("0", "a number above 0"),
    "user/max_mnt_namespaces": ("0", "a number above 0"),
    "user/max_pid_namespaces": ("0", "a number above 0"),
}


def spawner_problem(*, proc_sys: Path = Path("/proc/sys")) -> SiteProblem | None:
    """Why this host would refuse the build sandbox's spawner the user, mount and PID namespaces it
    makes and the mounts it makes in them, or None: a kernel without user namespaces, or a kernel
    setting of ``SPAWNER_SWITCHES`` at the value that refuses them. The sandbox's container asks
    Docker for everything else the spawner needs (its ``spawner`` confinement,
    ``hls.substrate.docker.launcher``); these only the host's administrator can change, and the
    reason names each with the value to set."""
    if not (proc_sys / "user" / "max_user_namespaces").is_file():
        return SiteProblem("this host's kernel has no user namespaces (no /proc/sys/user/max_user_namespaces), which "
                           "the build sandbox's spawner makes: run a kernel built with CONFIG_USER_NS", retryable=False)
    found = []
    for name, (refuses, allows) in SPAWNER_SWITCHES.items():
        try:
            value = (proc_sys / name).read_text().strip()
        except OSError:
            continue
        if value == refuses:
            found.append(f"{name.replace('/', '.')} is {value}; set it to {allows}")
    if not found:
        return None
    return SiteProblem("this host refuses an unprivileged user the user, mount and PID namespaces the build sandbox's "
                       f"spawner makes, or its mounts in them: {'; '.join(found)}", retryable=False)


def cgroup_problem(socket: Path, *, root: Path = CGROUP_ROOT) -> SiteProblem | None:
    """Why this host cannot hold role containers to their limits and count their OOM kills, or
    None: no cgroup v2 mount at *root*, no memory or pids controller in it, or a daemon on *socket*
    whose cgroup driver names no parent cgroup the launcher can make."""
    try:
        controllers = (root / "cgroup.controllers").read_text().split()
    except OSError:
        return SiteProblem(f"{root} is not a cgroup v2 mount: it has no cgroup.controllers", retryable=False)
    lacking = [c for c in ("memory", "pids", "cpuset") if c not in controllers]
    if lacking:
        return SiteProblem(f"the cgroup v2 mount {root} has no {', '.join(lacking)} controller", retryable=False)
    try:
        with DockerClient(socket) as client:
            client.check()
            driver = client.info().cgroup_driver
    except DockerError as e:
        return SiteProblem(f"the Docker daemon on {socket} cannot be used: {e}")
    if driver not in ("systemd", "cgroupfs"):
        return SiteProblem(f"the Docker daemon's cgroup driver is {driver!r}, neither systemd nor cgroupfs",
                           retryable=False)
    return None


def _launcher(site: SiteSettings, *, python: str, scratch: Path, allocation_gpus: int,
              allocation_gpu_uuids: list[str], pasta: Path | None, upstream_proxy: str | None) -> Launcher:
    """The launcher of the allocation this process runs in, held to its reservation. The step
    image is loaded once, here, from its checked archive in the store's cache, which the
    allocation's submission built when the cache held none (``_StepImageScheduler``). A sealed role's
    gateway reaches the
    internet through *upstream_proxy* (``site.egress_upstream_proxy``); *pasta* is the pasta seal's,
    which no Docker container runs."""
    environ = os.environ
    job = job_id(environ)
    state = StateDir(state_dir(environ))
    held = reservation(state, job)
    if list(allocation_gpu_uuids) != list(held.gpus) or allocation_gpus != len(held.gpus):
        raise RuntimeError(f"the runner was given GPUs {allocation_gpu_uuids}, not the reserved {list(held.gpus)}")
    pid_max = node_pid_max()
    if pid_max is None:
        raise RuntimeError("the host's pid_max cannot be read, so no container gets a pids limit")
    docker = site.docker
    client = DockerClient(docker.socket)
    client.check()
    found = step_image_file(ContentCache(PosixStore(site.store_root)))
    if found is None:
        raise RuntimeError(f"the store holds no step image images/{step_key()}, which the submission builds first")
    step = load_step_image(client, found[0], sha256=found[1], scratch=scratch)
    labels = dict([docker.owner_label.split("=", 1)]) if docker.owner_label else {}
    checkout = Path(hls.__file__).resolve().parents[2]
    host = state.host()
    return DockerLauncher(client=client, job_id=job, python=python, scratch=scratch, step_image=step,
                          allocation_gpu_uuids=held.gpus, cpus=held.cpus, memory_gib=held.memory_gib,
                          code_dirs=code_dirs(Path(python), checkout), data_dirs=(site.store_root,),
                          pids_limit=pid_max, labels=labels, upstream_proxy=upstream_proxy,
                          gateway_memory_gib=docker.gateway_memory_gib, driver=host.driver if host else None)


def docker(site: SiteSettings) -> Substrate:
    """Docker on this host, as *site* describes it: its daemon's socket and the BuildKit image its
    builds run (``site.docker``), the local scheduler's state directory and the memory it keeps back
    for the host (``site.local``), and the store (``site.store_root``), whose cache holds the step
    image (``hls.substrate.docker.step_image``)."""
    local = site.local
    return Substrate(
        name="docker",
        scheduler=lambda: _StepImageScheduler(site, state_dir=local.state_dir, inventory=SysInventory(),
                                         kept_memory_gib=local.kept_memory_gib, python=Path(sys.executable),
                                         stopper=STOPPER, stopper_options={"socket": str(site.docker.socket)}),
        batch_script=batch_script,
        placement=placement,
        check_placement=lambda request, site_, placement_, *, job_name: _check_placement(
            site, request, placement_, job_name=job_name),
        allocation_qos=_no_qos,
        site_request=_as_declared,
        is_job_log=is_job_log,
        check_job_id=check_job_id,
        check_node_name=check_node_name,
        reserved_variable=lambda name: reserved_variable(name) or name in RESERVED_VARIABLES,
        job_id=job_id,
        node_name=node_name,
        read_job_end=lambda job, environ: JobEnd(controller=job_end_time(StateDir(state_dir(environ)), job),
                                                 env=env_end_time(environ)),
        job_not_ready=job_not_ready,
        ended_jobs=lambda job_ids: ended_jobs(StateDir(state_dir(os.environ)), job_ids),
        job_cgroup=job_cgroup,
        cgroup_problem=lambda: cgroup_problem(site.docker.socket) or _step_image_problem(site),
        # A sealed role's gateway is a container of the step image, which holds the firewall tools,
        # on the daemon the cgroup check found: the host runs nothing of the seal itself.
        seal_problems=lambda environ: [],
        spawner_problem=spawner_problem,
        served_ucounts=frozenset(),
        launcher=lambda **kwargs: _launcher(site, **kwargs),
        runtime=lambda: DockerRuntime(gateway_memory_gib=site.docker.gateway_memory_gib),
        images=lambda scratch: DockerImages(scratch, job_id=job_id(os.environ), python=sys.executable,
                                            buildkit_image=site.docker.buildkit_image, socket=site.docker.socket),
        recipe_key=DockerImages.recipe_key,
        image_file=DockerImages.image_file,
        measure_grant=measure_grant,
        measure_tools=lambda pasta: measure_tools(site.docker.socket, step_key()),
    )
