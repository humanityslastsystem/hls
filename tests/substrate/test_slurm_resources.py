from __future__ import annotations

from pathlib import Path

import pytest

from hls.settings import SiteSettings
from hls.spec.resources import ResourceShape
from hls.substrate.slurm.batch import batch_script
from hls.substrate.slurm.resources import SITE_EXTRA_OPTIONS, sbatch_args, site_request

SITE = SiteSettings(partition="gpu", account="acct", store_root=Path("/s"),
                    node_scratch=Path("/tmp"), secrets_path=Path("/x"))
TWO = ResourceShape(gpus=2, gpu_kind="H200", cpus=4, memory_gib=16, time_sec=1800)
CPU = ResourceShape(cpus=2, memory_gib=8, time_sec=61)


def test_sbatch_args_ask_for_exactly_the_shape():
    assert sbatch_args(TWO, SITE, job_name="hls-max-alpha", output=Path("/s/o/slurm-%j.out")) == (
        "--job-name=hls-max-alpha", "--partition=gpu", "--account=acct", "--nodes=1", "--ntasks=1",
        "--cpus-per-task=4", "--mem=16G", "--gpus=h200:2", "--time=30", "--no-requeue", "--export=NIL",
        "--output=/s/o/slurm-%j.out")


EVAL = ResourceShape(gpus=1, gpu_kind="H200", cpus=16, memory_gib=196, time_sec=3600)


def test_an_allocation_with_socket_tasks_asks_for_two_tasks_one_per_socket():
    # A scored server of 8 CPUs on a site with site.evaluation_socket_tasks: two tasks of 8 with
    # at most one per socket, 16 CPUs in all, so the server's 8-CPU step sits on one socket.
    args = sbatch_args(EVAL, SITE, job_name="j", output=Path("/o"), socket_tasks=8)
    assert args[3:8] == ("--nodes=1", "--ntasks=2", "--ntasks-per-socket=1", "--cpus-per-task=8", "--mem=196G")
    assert "--ntasks=1" not in args and "--cpus-per-task=16" not in args
    assert sbatch_args(EVAL, SITE, job_name="j", output=Path("/o"))[3:6] == (
        "--nodes=1", "--ntasks=1", "--cpus-per-task=16")


@pytest.mark.parametrize("tasks", [4, 10, 0])
def test_socket_tasks_must_hold_the_requests_cpus_exactly(tasks):
    with pytest.raises(ValueError, match="two tasks"):
        sbatch_args(EVAL, SITE, job_name="j", output=Path("/o"), socket_tasks=tasks)


def test_an_allocation_on_another_partition_names_it():
    args = sbatch_args(CPU, SITE, job_name="j", output=Path("/o"), partition="cpu-a")
    assert "--partition=cpu-a" in args and "--partition=gpu" not in args
    assert "--partition=gpu" in sbatch_args(CPU, SITE, job_name="j", output=Path("/o"))


def test_zero_gpus_are_written_out_and_time_rounds_up():
    args = sbatch_args(CPU, SITE, job_name="j", output=Path("/o"), qos="normal")
    assert "--gpus=0" in args and "--time=2" in args and "--qos=normal" in args


def test_the_qos_is_the_one_the_allocation_was_planned_under():
    # The planner picks it by the allocation's time (allocation_qos); the site's own is not read here.
    site = SITE.model_copy(update={"qos": "normal", "qos_by_hours": {"normal": 24, "long": 168}})
    assert "--qos=long" in sbatch_args(TWO, site, job_name="j", output=Path("/o"), qos="long")
    assert not [a for a in sbatch_args(TWO, site, job_name="j", output=Path("/o"), qos=None) if a.startswith("--qos")]


def _with_extra(*extra: str) -> SiteSettings:
    return SiteSettings(**{**SITE.model_dump(), "sbatch_extra": list(extra)})


def test_excluded_nodes_join_the_sites_own_exclusion():
    # sbatch keeps only the last --exclude it is given, so one --exclude holds both lists.
    site = _with_extra("--exclude=node[11,15]", "--constraint=h200")
    args = sbatch_args(TWO, site, job_name="j", output=Path("/o"), exclude=["node01", "node02"])
    assert [a for a in args if a.startswith("--exclude=")] == ["--exclude=node[11,15],node01,node02"]
    assert "--constraint=h200" in args
    # Another partition gets none of the site's extra arguments, and still the excluded nodes.
    other = sbatch_args(CPU, site, job_name="j", output=Path("/o"), partition="cpu-a", exclude=["n1"])
    assert [a for a in other if a.startswith("--exclude=")] == ["--exclude=n1"]
    assert sbatch_args(TWO, site, job_name="j", output=Path("/o")) == sbatch_args(TWO, site, job_name="j",
                                                                                   output=Path("/o"), exclude=())
    with pytest.raises(ValueError, match="'g1,g2' is not a node name"):
        sbatch_args(TWO, SITE, job_name="j", output=Path("/o"), exclude=["g1,g2"])



def test_unknown_gres_is_refused():
    with pytest.raises(ValueError, match="gpu_gres"):
        sbatch_args(TWO, SITE.model_copy(update={"gpu_gres": {"B200": "b200"}}), job_name="j", output=Path("/o"))


@pytest.mark.parametrize("bad", [
    "--mem=999G", "--part=gpu-b", "--clusters=other", "-Mother", "--time-min=1", "--export-file=/x",
    "--exclude", "--exclude node03", "--exclude=", "-x", "exclude=node03", "--exclude==x", "--exclude=a b",
    "--exclude=x\x00", "--exclude=x\r", "--exclude=x\n", "--exclude=caf\u00e9",
])
def test_site_extras_outside_the_allowlist_are_refused_and_named(bad):
    with pytest.raises(ValueError, match="sbatch_extra") as caught:
        sbatch_args(TWO, _with_extra("--exclude=node03", bad), job_name="j", output=Path("/o"))
    assert repr(bad) in str(caught.value) and repr("--exclude=node03") not in str(caught.value)


def test_site_extras_on_the_allowlist_are_appended_in_order():
    site = _with_extra("--exclude=node03", "--constraint=h200")
    assert sbatch_args(TWO, site, job_name="j", output=Path("/o"))[-2:] == ("--exclude=node03", "--constraint=h200")
    odd = _with_extra("--constraint=[h200*1]", "--mail-user=a@b.edu")
    assert sbatch_args(TWO, odd, job_name="j", output=Path("/o"))[-2:] == ("--constraint=[h200*1]",
                                                                           "--mail-user=a@b.edu")
    every = [f"{name}=1" for name in sorted(SITE_EXTRA_OPTIONS)]
    assert sbatch_args(TWO, _with_extra(*every), job_name="j", output=Path("/o"))[-len(every):] == tuple(every)


def test_site_extras_go_to_the_gpu_partition_only_and_are_checked_wherever_the_allocation_goes():
    site = _with_extra("--exclude=node03", "--reservation=hold")
    assert sbatch_args(CPU, site, job_name="j", output=Path("/o"))[-2:] == ("--exclude=node03", "--reservation=hold")
    on_cpu = sbatch_args(CPU, site, job_name="j", output=Path("/o"), partition="cpu-a")
    assert not set(site.sbatch_extra) & set(on_cpu) and on_cpu[-1] == "--output=/o"
    with pytest.raises(ValueError, match="sbatch_extra"):
        sbatch_args(CPU, _with_extra("--mem=999G"), job_name="j", output=Path("/o"), partition="cpu-a")


def test_site_request_raises_gpus_to_the_site_floor():
    floor = SiteSettings(**{**SITE.model_dump(), "min_gpus_per_job": 1})
    assert site_request(TWO, floor, gpu_kind=None) == TWO
    raised = site_request(CPU, floor, gpu_kind="H200")
    assert (raised.gpus, raised.gpu_kind, raised.cpus) == (1, "H200", 2)
    with pytest.raises(ValueError, match="gpu kind"):
        site_request(CPU, floor, gpu_kind=None)
    assert site_request(CPU, SITE, gpu_kind=None) == CPU


def test_batch_script_starts_only_the_runner():
    text = batch_script(python=Path("/c/hls/.venv/bin/python"), plan_dir=Path("/s/runs/noop/r 1/plan"),
                        allocation=1, submission=2, checkout=Path("/c/hls"), path="/opt/slurm/bin:/usr/bin:/bin",
                        home=Path("/home/u"))
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "export PATH=/opt/slurm/bin:/usr/bin:/bin" in lines
    assert "export SLURM_EXPORT_ENV=ALL" in lines
    assert lines[-1] == ("exec /c/hls/.venv/bin/python -m hls.substrate.runner '/s/runs/noop/r 1/plan' "
                         "--allocation 1 --submission 2")
