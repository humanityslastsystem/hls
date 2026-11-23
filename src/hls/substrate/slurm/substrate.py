"""Slurm as a substrate (``hls.substrate.factory``), with Apptainer as its container runtime.

Each member that asks the controller runs the Slurm tools anew through ``SlurmCommands``, and
the job's cgroup is looked up from this process's own (``own_cgroup``), both when the member is
called."""

from __future__ import annotations

from hls.settings import SiteSettings
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.base import check_node_name
from hls.substrate.cgroup import job_dir, own_cgroup
from hls.substrate.factory import Substrate
from hls.substrate.grant import measure
from hls.substrate.site_checks import cgroup_problem, seal_problems
from hls.substrate.slurm.batch import batch_script
from hls.substrate.slurm.commands import SlurmCommands
from hls.substrate.slurm.launcher import SlurmLauncher, reserved_variable
from hls.substrate.slurm.resources import SlurmPlacement, allocation_qos, check_placement, site_request
from hls.substrate.slurm.scheduler import (SlurmScheduler, check_job_id, ended_jobs, is_job_log, job_not_ready,
                                           node_name, own_job_id, read_job_end)
from hls.substrate.ucounts import UCOUNTS
from hls.substrate.versions import measure_tools


def slurm(site: SiteSettings) -> Substrate:
    """Slurm, with Apptainer as its container runtime. *site* is not read here: the members that
    need the site's settings take them as arguments."""
    return Substrate(
        name="slurm",
        scheduler=lambda: SlurmScheduler(SlurmCommands()),
        batch_script=batch_script,
        placement=lambda *, partition, qos, socket_tasks, exclude: SlurmPlacement(
            partition=partition, qos=qos, socket_tasks=socket_tasks, exclude=list(exclude)),
        check_placement=check_placement,
        allocation_qos=allocation_qos,
        site_request=site_request,
        is_job_log=is_job_log,
        check_job_id=check_job_id,
        check_node_name=check_node_name,
        reserved_variable=reserved_variable,
        job_id=own_job_id,
        node_name=node_name,
        read_job_end=lambda job_id, environ: read_job_end(job_id, SlurmCommands(), environ),
        job_not_ready=lambda job_id: job_not_ready(SlurmCommands(), job_id),
        ended_jobs=lambda job_ids: ended_jobs(SlurmScheduler(SlurmCommands()), list(job_ids)),
        job_cgroup=lambda: job_dir(own_cgroup()),
        cgroup_problem=lambda: cgroup_problem(own_cgroup),
        # The pasta seal runs pasta and the firewall tools on the node, in a user namespace.
        seal_problems=lambda environ: seal_problems(site.pasta_path, environ),
        # The login node is not the compute node, which the runner's seal check judges.
        spawner_problem=lambda: None,
        # The pasta seal writes each in its user namespace before the payload starts.
        served_ucounts=frozenset(u.name for u in UCOUNTS),
        launcher=SlurmLauncher,
        runtime=ApptainerRuntime,
        images=ApptainerImages,
        recipe_key=ApptainerImages.recipe_key,
        image_file=ApptainerImages.image_file,
        measure_grant=lambda environ, scratch: measure(limit_dir=job_dir(own_cgroup()), scratch=scratch),
        measure_tools=measure_tools,
    )
