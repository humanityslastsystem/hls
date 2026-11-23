"""An allocation's request shape as sbatch arguments, under the site's rules.

Every allocation asks for exactly its request shape on one node as one task, so the runner
can check the grant against it, but for one with socket tasks (``site.evaluation_socket_tasks``,
``Allocation.socket_tasks``): it asks for the request's CPUs as two tasks of half of them with
at most one task per socket, so a step of one task's CPUs lands on one socket, and the runner
checks the same total. Disk is not requested from Slurm; the runner's grant check
covers it. ``--export=NIL`` keeps the login node's environment out of the job,
``--no-requeue`` leaves preemption to the orchestrator's retries, and a zero GPU count is
written out as ``--gpus=0``, because a GPU partition can treat a missing count differently
from zero. An allocation goes to ``site.partition`` unless it names another one: the site's CPU
partition, for a cache warmer. It runs under the QOS the planner picked for its time against the
site's limits (``allocation_qos``).

``site.sbatch_extra`` is limited to the options in ``SITE_EXTRA_OPTIONS``, each written
``--<name>=<value>``. sbatch accepts unambiguous option prefixes (``--part``) and options
that move the job elsewhere (``--clusters``), so a list of forbidden options would miss some.
The allowed options are listed instead. They describe the GPU partition (its reservations,
constraints and nodes), so an allocation on another partition does not get them; they are
checked for every allocation all the same. The nodes an allocation excludes (``Allocation.exclude``:
a retry's, after a node's grant check failed) join the site's own ``--exclude`` in one, since
sbatch keeps only the last it is given."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from hls.settings import SBATCH_ARG, SiteSettings
from hls.spec.resources import ResourceShape
from hls.substrate.base import NoQOSHolds, check_node_name

#: The only sbatch options ``site.sbatch_extra`` may set. Every other option is one the
#: orchestrator sets, or one that changes what the job is or where it runs.
SITE_EXTRA_OPTIONS = frozenset({
    "--exclude", "--constraint", "--reservation", "--nodelist", "--nice", "--mail-type", "--mail-user",
})


#: A node's name as the scheduler knows it, checked so that it can go into ``--exclude``.
NodeName = Annotated[str, AfterValidator(check_node_name)]


class SlurmPlacement(BaseModel):
    """Where Slurm places an allocation beside the shape it asks for (``sbatch_args``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The substrate the placement is for.
    kind: Literal["slurm"] = "slurm"
    #: The CPUs of each of two tasks, at most one per socket, that sbatch asks the request's
    #: CPUs as, for an allocation that runs a scored server on a site with
    #: ``site.evaluation_socket_tasks`` (``hls.pipeline.planner.socket_tasks``); None for one task.
    socket_tasks: int | None = Field(ge=1)
    #: The Slurm partition it goes to when that is not ``site.partition``: the site's CPU
    #: partition, for a cache warmer. None for the GPU partition, the only one a run uses.
    partition: str | None = Field(min_length=1)
    #: The Slurm QOS it runs under, picked by its time against the site's limits
    #: (``allocation_qos``); None for the account's default.
    qos: str | None = Field(min_length=1)
    #: Nodes this allocation's submissions must not land on, from earlier attempts' grant checks:
    #: a retry after a node was found short of the request goes elsewhere.
    exclude: list[NodeName]

    def request_problem(self, request: ResourceShape) -> str | None:
        """Why an allocation of *request* cannot go where this places it, or None: one on another
        partition asks for no GPUs, since only a run uses GPUs and a run uses the GPU partition."""
        if self.partition is not None and request.gpus > 0:
            return (f"goes to partition {self.partition} and asks for {request.gpus} GPUs; only a run's allocation "
                    "on the GPU partition does")
        return None


def _allowed_extra(arg: str) -> bool:
    m = SBATCH_ARG.fullmatch(arg)
    return m is not None and m.group(1) in SITE_EXTRA_OPTIONS


def site_request(shape: ResourceShape, site: SiteSettings, *, gpu_kind: str | None) -> ResourceShape:
    """*shape* as this site lets it be requested: with at least ``site.min_gpus_per_job``
    GPUs. A shape raised from no GPUs takes *gpu_kind*, the kind the run's other stages
    use; with none to take, raise ValueError."""
    if shape.gpus >= site.min_gpus_per_job:
        return shape
    kind = shape.gpu_kind or gpu_kind
    if kind is None:
        raise ValueError(f"the site needs at least {site.min_gpus_per_job} GPUs per job, and no stage of "
                         "the run names a gpu kind to ask for")
    return ResourceShape(**{**shape.model_dump(), "gpus": site.min_gpus_per_job, "gpu_kind": kind})


def allocation_qos(time_sec: int, site: SiteSettings, *, partition: str | None = None) -> str | None:
    """The QOS an allocation that asks for *time_sec* seconds on *partition* (None for
    ``site.partition``) runs under on *site*: ``site.qos`` when the site sets no limits
    (``site.qos_by_hours`` is empty); else ``site.qos`` while its limit holds the time sbatch asks
    for, *time_sec* rounded up to whole minutes, and otherwise the QOS with the least limit that
    holds it. Raise NoQOSHolds, naming the partition, the hours asked and the longest limit, when
    none holds it."""
    limits = site.qos_by_hours
    if not limits:
        return site.qos
    asked = math.ceil(time_sec / 60) * 60

    def holds(name: str) -> bool:
        return asked <= limits[name] * 3600

    if site.qos is not None and holds(site.qos):
        return site.qos
    holding = sorted((limits[name], name) for name in limits if holds(name))
    if not holding:
        longest, name = max((limit, name) for name, limit in limits.items())
        raise NoQOSHolds(f"it asks for {asked / 3600:.2f} h, more than any QOS of the site lets a job on "
                         f"partition {partition or site.partition} ask (site.qos_by_hours): the longest, {name}, "
                         f"allows {longest:g} h")
    return holding[0][1]


def sbatch_args(request: ResourceShape, site: SiteSettings, *, job_name: str, output: Path,
                partition: str | None = None, qos: str | None = None,
                socket_tasks: int | None = None, exclude: Sequence[str] = ()) -> tuple[str, ...]:
    """The sbatch arguments that ask for *request* on *site* under the QOS *qos*, the one the
    allocation was planned under (``Allocation.qos``, ``allocation_qos``; None for the
    account's default), followed by the site's extra arguments. *partition* replaces
    ``site.partition`` for an allocation that runs elsewhere, and such an allocation gets no
    extra arguments. *socket_tasks* asks for the request's CPUs as two tasks of that many CPUs
    with at most one per socket. *exclude* names nodes the job must not land on (earlier
    attempts' grant checks found them short); on the site's partition they follow the value of
    the site's own ``--exclude`` in one ``--exclude``, since sbatch keeps only the last one it
    is given. Raise ValueError if the site has no gres name for the request's GPU kind, if two
    tasks of *socket_tasks* CPUs are not exactly the request's CPUs, if an extra argument is not
    ``--<name>=<value>`` with a name in ``SITE_EXTRA_OPTIONS`` and a value of printable ASCII
    without spaces that does not start with "=", or if a node of *exclude* is not a node name
    (``check_node_name``)."""
    if socket_tasks is not None and (socket_tasks < 1 or 2 * socket_tasks != request.cpus):
        raise ValueError(f"two tasks of {socket_tasks} CPUs are not the request's {request.cpus} CPUs")
    if request.gpus > 0:
        gres = site.gpu_gres.get(request.gpu_kind or "")
        if gres is None:
            raise ValueError(f"site.gpu_gres has no gres name for gpu kind {request.gpu_kind}")
        gpus = f"--gpus={gres}:{request.gpus}"
    else:
        gpus = "--gpus=0"
    refused = [a for a in site.sbatch_extra if not _allowed_extra(a)]
    if refused:
        raise ValueError(f"site.sbatch_extra may only hold --<name>=<value> with <name> one of "
                         f"{sorted(SITE_EXTRA_OPTIONS)}; refused: {refused}")
    args = [f"--job-name={job_name}", f"--partition={site.partition if partition is None else partition}"]
    if site.account is not None:
        args.append(f"--account={site.account}")
    if qos is not None:
        args.append(f"--qos={qos}")
    tasks = (["--ntasks=1", f"--cpus-per-task={request.cpus}"] if socket_tasks is None
             else ["--ntasks=2", "--ntasks-per-socket=1", f"--cpus-per-task={socket_tasks}"])
    args += [
        "--nodes=1",
        *tasks,
        f"--mem={request.memory_gib}G",
        gpus,
        f"--time={math.ceil(request.time_sec / 60)}",
        "--no-requeue",
        "--export=NIL",
        f"--output={output}",
    ]
    extra = list(site.sbatch_extra) if partition is None else []
    if exclude:
        nodes = [check_node_name(node) for node in exclude]
        # The site's value, given once at most (``SiteSettings``), may be a hostlist with commas
        # inside brackets, so it is kept whole.
        site_value = [a.split("=", 1)[1] for a in extra if a.startswith("--exclude=")]
        extra = [a for a in extra if not a.startswith("--exclude=")]
        args.append(f"--exclude={','.join([*site_value, *nodes])}")
    return (*args, *extra)


def placement_args(request: ResourceShape, site: SiteSettings, placement: SlurmPlacement, *, job_name: str,
                   output: Path) -> tuple[str, ...]:
    """The sbatch arguments of an allocation of *request* placed by *placement* (``sbatch_args``)."""
    return sbatch_args(request, site, job_name=job_name, output=output, partition=placement.partition,
                       qos=placement.qos, socket_tasks=placement.socket_tasks, exclude=placement.exclude)


def check_placement(request: ResourceShape, site: SiteSettings, placement: SlurmPlacement, *, job_name: str) -> None:
    """Raise ValueError when sbatch could not be given an allocation of *request* placed by
    *placement* on *site* (``sbatch_args``), as the planner checks each allocation."""
    placement_args(request, site, placement, job_name=job_name, output=Path("/dev/null"))
