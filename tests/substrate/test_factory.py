"""The substrate factory: one registered substrate, every member required, and the Slurm
substrate's members running the same Slurm tools with the same arguments as the code they stand
for."""

from __future__ import annotations

import dataclasses
import sys
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

import hls.substrate.factory as factory
import hls.substrate.slurm.substrate as slurm_substrate
from hls.settings import SLURM_SITE_FIELDS, SiteSettings
from hls.spec.resources import ResourceShape
from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.base import JobEnd, NoQOSHolds
from hls.substrate.factory import SUBSTRATE_VAR, Substrate, step, substrate
from hls.substrate.slurm.commands import SlurmCommands
from hls.substrate.slurm.launcher import SlurmLauncher
from hls.substrate.slurm.resources import SlurmPlacement
from hls.substrate.slurm.scheduler import SlurmScheduler
from tests.substrate.slurm_fakes import FakeRun, done
from tests.substrate.test_slurm_resources import SITE

STEP = "/system.slice/slurmstepd.scope/job_12345/step_1/user/task_0"


def _slurm_tools(monkeypatch, run: FakeRun) -> None:
    """Make every Slurm tool the substrate's members run answer from *run*."""
    monkeypatch.setattr(slurm_substrate, "SlurmCommands", lambda: SlurmCommands(run=run))


def test_a_site_runs_on_slurm_unless_it_names_another_substrate():
    assert list(factory.SUBSTRATES) == ["slurm", "docker"]
    assert SITE.substrate == "slurm" and substrate(SITE).name == "slurm"
    with pytest.raises(ValidationError, match="Input should be 'slurm' or 'docker'"):
        SiteSettings.model_validate({**SITE.model_dump(), "substrate": "other"})


def test_a_slurm_site_writes_no_substrate_so_its_settings_and_their_digest_stay_as_they_were():
    assert "substrate" not in SITE.model_dump(mode="json")
    # A Docker site sets none of the Slurm settings and gives its own tables.
    docker_site = SiteSettings.model_validate(
        {**{k: v for k, v in SITE.model_dump(mode="json").items() if k not in SLURM_SITE_FIELDS},
         "substrate": "docker", "docker": {"buildkit_image": "moby/buildkit@sha256:" + "b" * 64},
         "local": {"state_dir": "/var/hls/state", "kept_memory_gib": 8}})
    assert docker_site.model_dump(mode="json")["substrate"] == "docker"
    assert SiteSettings.model_validate(docker_site.model_dump(mode="json")) == docker_site
    assert SiteSettings.model_validate(SITE.model_dump(mode="json")).substrate == "slurm"


def test_a_docker_site_runs_on_the_docker_substrate_module(monkeypatch):
    # A stand-in for the Docker launcher's module: the factory imports it only when a site names it.
    module = types.ModuleType("hls.substrate.docker.substrate")
    built = []
    module.docker = lambda site: built.append(site) or dataclasses.replace(substrate(SITE), name="docker")
    monkeypatch.setitem(sys.modules, "hls.substrate.docker.substrate", module)
    docker_site = SITE.model_copy(update={"substrate": "docker"})
    assert substrate(docker_site).name == "docker" and built == [docker_site]


@pytest.mark.parametrize("member", [f.name for f in dataclasses.fields(Substrate)])
def test_every_member_is_required(member):
    given = {f.name: getattr(substrate(SITE), f.name) for f in dataclasses.fields(Substrate) if f.name != member}
    with pytest.raises(TypeError, match=member):
        Substrate(**given)


def test_the_slurm_scheduler_launcher_and_runtime_are_slurms_and_apptainers(tmp_path):
    on = substrate(SITE)
    assert isinstance(on.scheduler(), SlurmScheduler)
    assert isinstance(on.runtime(), ApptainerRuntime)
    launcher = on.launcher(python="/py", scratch=tmp_path, allocation_gpus=2, allocation_gpu_uuids=["GPU-0", "GPU-1"])
    assert isinstance(launcher, SlurmLauncher)
    assert (launcher.python, launcher.scratch, launcher.allocation_gpus, launcher.allocation_gpu_uuids) == (
        "/py", tmp_path, 2, ["GPU-0", "GPU-1"])


def test_the_scheduler_asks_slurm(monkeypatch):
    run = FakeRun().on("scancel", done())
    _slurm_tools(monkeypatch, run)
    substrate(SITE).scheduler().cancel("12345")
    assert run.calls == [["scancel", "12345"]]


def test_the_runners_readers_ask_the_controller_as_slurm_does(monkeypatch):
    show = done("JobId=7 JobName=hls\n   JobState=RUNNING Reason=None Dependency=(null)\n"
                "   StartTime=1000 EndTime=4600 Deadline=N/A\n")
    run = FakeRun().on("scontrol", show, show).on("squeue", done("")).on(
        "sacct", done("7|COMPLETED|0:0|1000\n8|RUNNING|0:0|1000\n"))
    _slurm_tools(monkeypatch, run)
    on = substrate(SITE)
    assert on.read_job_end("7", {"SLURM_JOB_END_TIME": "4660"}) == JobEnd(controller=4600.0, env=4660.0)
    assert on.job_not_ready("7") is None
    assert on.ended_jobs(["7", "8"]) == {"7"}
    assert run.calls == [
        ["scontrol", "show", "job", "7"], ["scontrol", "show", "job", "7"],
        ["squeue", "--me", "--all", "--noheader", "--format=%i|%T|%r|%S|%k"],
        ["sacct", "--noheader", "--parsable2", "--allocations", "--jobs=7,8", "--format=JobID,State,ExitCode,Start"]]


def test_the_job_and_its_node_are_named_as_slurm_names_them():
    on = substrate(SITE)
    assert on.job_id({"SLURM_JOB_ID": "12345"}) == "12345"
    with pytest.raises(KeyError):
        on.job_id({})
    assert on.check_job_id("12345") == "12345"
    with pytest.raises(ValueError):
        on.check_job_id("--all")
    assert on.node_name("n17.cluster.local", {}) == "n17"
    assert on.node_name("host-17.cluster.local", {"SLURMD_NODENAME": "c17"}) == "c17"
    assert on.check_node_name("n17") == "n17"
    with pytest.raises(ValueError):
        on.check_node_name("g[1-2]")


def test_the_job_cgroup_follows_slurms_layout(monkeypatch):
    on = substrate(SITE)
    monkeypatch.setattr(slurm_substrate, "own_cgroup", lambda: STEP)
    assert on.job_cgroup() == Path("/sys/fs/cgroup/system.slice/slurmstepd.scope/job_12345")
    monkeypatch.setattr(slurm_substrate, "own_cgroup", lambda: "/user.slice/user-1.slice/session-3.scope")
    with pytest.raises(RuntimeError):
        on.job_cgroup()


def test_a_slurm_nodes_seal_check_is_the_pasta_seals(monkeypatch):
    """The pasta seal runs pasta and the firewall tools on the node: a node whose PATH holds no
    pasta, with no site.pasta_path, cannot run it."""
    on = substrate(SITE.model_copy(update={"pasta_path": None}))
    problems = on.seal_problems({"PATH": "/nonexistent"})
    assert any("holds no pasta" in p.reason for p in problems)


@pytest.mark.parametrize("environ", [{}, {SUBSTRATE_VAR: "slurm"}])
def test_a_step_that_names_no_substrate_or_slurm_is_a_slurm_step(tmp_path, environ):
    # The Slurm launcher sets no name, so a Slurm step's environment is as it was.
    here = step(environ)
    assert here.step_id({"SLURM_JOB_ID": "5", "SLURM_STEP_ID": "2"}) == "5.2"
    assert here.step_id({}) == "?.?"
    assert here.step_cgroup(STEP, tmp_path) == tmp_path / "system.slice/slurmstepd.scope/job_12345/step_1/user"
    with pytest.raises(RuntimeError, match="not inside a Slurm step"):
        here.step_cgroup("/user.slice/user-1.slice/session-3.scope", tmp_path)


def test_a_step_of_an_unknown_substrate_is_refused():
    with pytest.raises(RuntimeError, match=r"HLS_SUBSTRATE='other' names no substrate hls runs a step under; "
                                           r"it runs under \['docker', 'slurm'\]"):
        step({SUBSTRATE_VAR: "other"})


def test_a_docker_step_takes_its_id_and_limits_from_the_docker_step_module(tmp_path, monkeypatch):
    # A stand-in for the Docker launcher's module: the factory imports it only when a step asks.
    module = types.ModuleType("hls.substrate.docker.step")
    module.step_id = lambda environ: environ["HLS_STEP_ID"]
    module.step_cgroup = lambda cgroup, root: root.joinpath(*[p for p in cgroup.split("/") if p])
    module.runtime_reads = module.runtime_setting = lambda name: False
    monkeypatch.setitem(sys.modules, "hls.substrate.docker.step", module)
    here = step({SUBSTRATE_VAR: "docker"})
    assert here.step_id({"HLS_STEP_ID": "41.3"}) == "41.3"
    assert here.step_cgroup("/", tmp_path) == tmp_path
    assert not here.runtime_reads("TERM") and not here.runtime_setting("APPTAINER_BIND")


def test_the_variables_the_launcher_and_the_runtime_own():
    on = substrate(SITE)
    assert on.reserved_variable("SLURM_JOB_ID") and on.reserved_variable("APPTAINER_BIND")
    assert on.reserved_variable("PATH") and not on.reserved_variable("HF_TOKEN")
    here = step({})
    assert here.runtime_setting("APPTAINERENV_X") and here.runtime_setting("SINGULARITY_BIND")
    assert not here.runtime_setting("TERM") and here.runtime_reads("TERM") and here.runtime_reads("https_proxy")
    assert not here.runtime_reads("HF_TOKEN")


def test_the_planning_calls_are_slurms(tmp_path):
    on = substrate(SITE)
    shape = ResourceShape(gpus=0, cpus=4, memory_gib=8, disk_gib=0, time_sec=3600)
    assert on.site_request(shape, SITE, gpu_kind=None) == shape
    assert on.allocation_qos(3600, SITE) == SITE.qos
    limited = SITE.model_copy(update={"qos": None, "qos_by_hours": {"short": 1.0}})
    with pytest.raises(NoQOSHolds):
        on.allocation_qos(7200, limited)
    placed = on.placement(partition=None, qos="normal", socket_tasks=None, exclude=["n17"])
    assert placed == SlurmPlacement(partition=None, qos="normal", socket_tasks=None, exclude=["n17"])
    on.check_placement(shape, SITE, placed, job_name="j")
    with pytest.raises(ValueError, match="site.gpu_gres has no gres name for gpu kind H200"):
        on.check_placement(shape.model_copy(update={"gpus": 1, "gpu_kind": "H200"}),
                           SITE.model_copy(update={"gpu_gres": {"B200": "b200"}}), placed, job_name="j")
    assert on.is_job_log("slurm-12345.out") and not on.is_job_log("slurm-12345.err")
    assert not on.is_job_log("job-12345.out")
    text = on.batch_script(python=Path("/c/.venv/bin/python"), plan_dir=Path("/s/plan"), allocation=0, submission=1,
                           checkout=Path("/c"), path="/usr/bin", home=Path("/h"))
    assert "export SLURM_EXPORT_ENV=ALL" in text.splitlines()


def test_every_substrate_names_the_file_its_image_runtime_holds_an_image_in():
    from hls.substrate.apptainer.images import ApptainerImages
    from hls.substrate.docker.images import DockerImages

    assert factory.image_files() == {"slurm": ApptainerImages.image_file, "docker": DockerImages.image_file}
    assert set(factory.image_files()) == set(factory.SUBSTRATES)
