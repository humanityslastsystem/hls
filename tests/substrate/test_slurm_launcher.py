from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from hls.egress.policy import NetworkPolicy, SealPolicy
from hls.spec.resources import GrantedResources, ResourceShape
from hls.substrate import role_exec
from hls.substrate.apptainer.argv import START_MARKER, ApptainerRuntime
from hls.substrate.apptainer.egress import SEAL_TOOL_PATH_VAR, seal_env
from hls.substrate.container import ContainerRole, ContainerSpec, PinnedImage, SealPlan
from hls.substrate.seal import seal_files
from hls.substrate.base import JobState, Launcher, NodeSize, RoleFailed, RoleHandle, RoleReport, RoleSpec
from hls.substrate import log_cap
from hls.substrate.nproc import node_evaluations
from hls.substrate.slurm.commands import SlurmCommands
from hls.substrate.slurm.launcher import (SlurmLauncher, SlurmRoleHandle, reserved_variable, srun_argv,
                                         step_env)
from tests.substrate.slurm_fakes import FakeRun, done
from tests.substrate.test_egress_seal import refusing_new_space
from tests.substrate.test_nproc import node_root

JOB = "1234110"
BATCH_ENV = {
    "SLURM_JOB_ID": JOB, "SLURM_JOBID": JOB, "SLURM_JOB_NODELIST": "node05",
    "SLURM_GPUS": "2", "SLURM_TRES_PER_TASK": "cpu=8", "SLURM_CPUS_PER_TASK": "8", "SLURM_MEM_PER_NODE": "32768",
    "SLURM_GPUS_ON_NODE": "2", "SLURM_JOB_GPUS": "3,6", "SLURM_EXPORT_ENV": "NIL", "SLURM_NTASKS": "1",
    "PATH": "/opt/slurm/bin:/usr/bin:/bin", "HOME": "/home/u", "HTTPS_PROXY": "http://proxy:3128",
    "HF_HOME": "/shared/hf", "CUDA_VISIBLE_DEVICES": "0,1",
}
# Every job-level variable the step environment must drop, one name per alternative of the rule.
REQUEST_VARS = [
    "SLURM_GPUS", "SLURM_GPUS_PER_NODE", "SLURM_GPUS_PER_TASK", "SLURM_GPUS_PER_SOCKET", "SLURM_CPUS_PER_GPU",
    "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU", "SLURM_TRES_PER_TASK",
    "SLURM_TRES_PER_NODE", "SLURM_GPU_BIND", "SLURM_GPU_FREQ", "SLURM_GRES", "SLURM_STEP_GRES", "SLURM_NTASKS",
    "SLURM_NTASKS_PER_NODE", "SLURM_NTASKS_PER_CORE", "SLURM_NPROCS", "SLURM_EXACT", "SLURM_EXCLUSIVE",
    "SLURM_OVERLAP", "SLURM_OVERCOMMIT", "SLURM_EXPORT_ENV", "SLURM_DISTRIBUTION", "SLURM_CPU_BIND",
    "SLURM_CPU_BIND_TYPE", "SLURM_CPU_BIND_LIST", "SLURM_CPU_BIND_VERBOSE", "SLURM_MEM_BIND", "SLURM_MEM_BIND_TYPE",
    "SLURM_HINT", "SLURM_THREADS_PER_CORE",
]
# Slurm variables that describe the job and are not read by srun as a request; the step keeps them.
KEPT_VARS = [
    "SLURM_JOB_ID", "SLURM_JOBID", "SLURM_JOB_NODELIST", "SLURM_NODELIST", "SLURM_JOB_NAME", "SLURM_GPUS_ON_NODE",
    "SLURM_JOB_GPUS", "SLURM_JOB_CPUS_PER_NODE", "SLURM_CPUS_ON_NODE", "SLURM_JOB_NUM_NODES", "SLURM_NNODES",
    "SLURM_SUBMIT_DIR", "SLURM_CONF", "SLURM_CLUSTER_NAME", "SLURM_JOB_PARTITION", "SLURM_JOB_ACCOUNT",
]
GPU1 = ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=2, time_sec=600)
U1, U2 = "GPU-d7fdcb95-ed58-0b76-ad0f-88ff81569074", "GPU-6fe08ff0-96d2-f6db-285c-0abad16b8d5a"


def _spec(tmp_path: Path, name: str, shape: ResourceShape = GPU1, env: dict | None = None) -> RoleSpec:
    return RoleSpec(name=name, argv=("/venv/bin/python", "-m", "hls.substrate.noop_role"), shape=shape, owner="harness",
                    log_dir=tmp_path / "logs", env=env or {})


def test_step_env_drops_the_job_request_and_the_host():
    env = step_env(BATCH_ENV, {"HLS_ROLE": "gpu0"}, tmpdir=Path("/tmp/hls-1234110/roles/gpu0/tmp"))
    for name in ("SLURM_GPUS", "SLURM_TRES_PER_TASK", "SLURM_CPUS_PER_TASK", "SLURM_MEM_PER_NODE", "SLURM_EXPORT_ENV",
                 "SLURM_NTASKS", "HTTPS_PROXY", "HF_HOME", "CUDA_VISIBLE_DEVICES"):
        assert name not in env
    assert env["SLURM_JOB_ID"] == "1234110" and env["SLURM_JOB_NODELIST"] == "node05"
    assert env["PATH"] == "/opt/slurm/bin:/usr/bin:/bin" and env["HLS_ROLE"] == "gpu0"
    assert env["HOME"] == "/home/u" and env["LANG"] == "C.UTF-8"
    assert env["TMPDIR"] == "/tmp/hls-1234110/roles/gpu0/tmp"


def test_step_env_keeps_exactly_the_slurm_variables_that_are_not_a_request():
    batch = {name: "1" for name in [*REQUEST_VARS, *KEPT_VARS]}
    env = step_env(batch, {}, tmpdir=Path("/t"))
    assert {k for k in env if k.startswith("SLURM_")} == set(KEPT_VARS)


@pytest.mark.parametrize("bad", [
    "SLURM_JOB_ID", "SLURM_GPUS", "SRUN_CPUS_PER_TASK", "SRUN_EXPORT_ENV", "PATH", "HOME", "LANG", "TMPDIR",
    "CUDA_VISIBLE_DEVICES", "APPTAINERENV_CUDA_VISIBLE_DEVICES", "APPTAINERENV_HF_TOKEN", "SINGULARITYENV_X",
    "APPTAINER_BIND", "SINGULARITY_BIND", "APPTAINER_ENV_FILE", "APPTAINER_NO_MOUNT",
])
def test_a_role_cannot_set_what_the_launcher_owns(bad):
    with pytest.raises(ValueError, match="may not set"):
        step_env(BATCH_ENV, {bad: "x"}, tmpdir=Path("/t"))


def test_a_role_may_say_where_apptainer_keeps_its_cache_and_build_files():
    kept = {"APPTAINER_CACHEDIR": "/tmp/hls-5/apptainer-cache",
            "APPTAINER_TMPDIR": "/tmp/hls-5/roles/b/tmp-build"}
    env = step_env(BATCH_ENV, kept, tmpdir=Path("/t"))
    assert {k: env[k] for k in kept} == kept
    with pytest.raises(ValueError) as caught:
        step_env(BATCH_ENV, {**kept, "APPTAINER_BIND": "/shared"}, tmpdir=Path("/t"))
    assert str(caught.value) == "a role may not set ['APPTAINER_BIND']"


@pytest.mark.parametrize("name", ["APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"])
def test_a_secret_grant_still_may_not_set_where_apptainer_keeps_its_files(name):
    assert reserved_variable(name)


def test_an_exclusive_gpu_step(tmp_path):
    argv = srun_argv(_spec(tmp_path, "gpu0"), python="/venv/bin/python", report=Path("/l/gpu0.report.json"),
                     scratch=Path("/tmp/hls-5"), shared=None, allocation_gpus=2)
    assert argv[:9] == ["srun", "--exact", "--nodes=1", "--ntasks=1", "--cpus-per-task=1", "--mem=2G", "--gpus=1",
                        "--time=10", "--job-name=gpu0"]
    assert "--export=ALL" in argv
    assert "--gres=gpu:1" not in argv and "--overlap" not in argv
    i = argv.index("--")
    assert argv[i + 1:] == ["/venv/bin/python", "-m", "hls.substrate.noop_role"]
    assert argv[argv.index("--gpu-kind") + 1] == "H200"


def test_a_cpu_step_gets_no_gres(tmp_path):
    cpu = ResourceShape(cpus=1, memory_gib=1, time_sec=60)
    argv = srun_argv(_spec(tmp_path, "cpu", cpu), python="p", report=Path("/r"), scratch=Path("/s"), shared=None,
                     allocation_gpus=2)
    assert "--gres=none" in argv and not any(a.startswith("--gpus=") for a in argv)
    assert "--gpu-kind" not in argv


def test_an_env_file_is_handed_to_the_wrapper_by_path_only(tmp_path):
    spec = RoleSpec(name="gpu0", argv=("/venv/bin/python", "-m", "x"), shape=GPU1, owner="harness",
                    log_dir=tmp_path / "logs", env_file=Path("/tmp/hls-5/roles/gpu0/secrets.env"), env_fd=True)
    argv = srun_argv(spec, python="p", report=Path("/r"), scratch=Path("/s"), shared=None, allocation_gpus=2)
    i = argv.index("--")
    assert argv[argv.index("--env-file") + 1] == "/tmp/hls-5/roles/gpu0/secrets.env"
    assert argv.index("--env-file") < i and argv.index("--env-fd") < i
    plain = srun_argv(_spec(tmp_path, "gpu0"), python="p", report=Path("/r"), scratch=Path("/s"), shared=None,
                      allocation_gpus=2)
    assert "--env-file" not in plain and "--env-fd" not in plain


def test_only_a_contestant_role_has_its_wrapper_set_oom_score_adj_1000(tmp_path):
    def wrapper(owner: str) -> list[str]:
        spec = RoleSpec(name="srv", argv=("/usr/bin/apptainer", "exec", "x.sif"), shape=GPU1, owner=owner,
                        log_dir=tmp_path / "logs", env_file=Path("/tmp/hls-5/roles/srv/container.env"), env_fd=True)
        argv = srun_argv(spec, python="p", report=Path("/r"), scratch=Path("/s"), shared=None, allocation_gpus=2)
        assert argv[argv.index("--") + 1:] == list(spec.argv)
        return argv[:argv.index("--")]

    contestant = wrapper("contestant")
    assert contestant[contestant.index("--oom-score-adj") + 1] == "1000"
    assert "--oom-score-adj" not in wrapper("harness")


def test_a_role_held_to_one_numa_node_has_its_wrapper_check_its_cpus(tmp_path):
    spec = RoleSpec(name="srv", argv=("/usr/bin/apptainer", "exec", "x"), shape=GPU1, owner="contestant",
                    log_dir=tmp_path / "logs", one_numa_node=True)
    argv = srun_argv(spec, python="p", report=Path("/r"), scratch=Path("/s"), shared=None, allocation_gpus=2)
    assert "--one-numa-node" in argv[:argv.index("--")]
    plain = srun_argv(_spec(tmp_path, "gpu0"), python="p", report=Path("/r"), scratch=Path("/s"), shared=None,
                      allocation_gpus=2)
    assert "--one-numa-node" not in plain


def test_a_role_with_a_process_ceiling_has_its_wrapper_set_it(tmp_path):
    spec = RoleSpec(name="srv", argv=("/usr/bin/apptainer", "exec", "x"), shape=GPU1, owner="contestant",
                    log_dir=tmp_path / "logs", nproc=50_000)
    argv = srun_argv(spec, python="p", report=Path("/r"), scratch=Path("/s"), shared=None, allocation_gpus=2)
    wrapper = argv[:argv.index("--")]
    assert wrapper[wrapper.index("--nproc") + 1] == "50000"
    plain = srun_argv(_spec(tmp_path, "gpu0"), python="p", report=Path("/r"), scratch=Path("/s"), shared=None,
                      allocation_gpus=2)
    assert "--nproc" not in plain
    with pytest.raises(ValueError, match="process ceiling"):
        RoleSpec(name="srv", argv=("x",), shape=GPU1, owner="contestant", log_dir=tmp_path, nproc=0)


def test_a_tmpdir_that_is_there_is_emptied_in_place_so_a_launch_needs_no_new_space(tmp_path):
    # A node whose disk a contestant filled has no block for a new directory; the one the
    # stage made before the contestant started is reused. A descriptor held on it keeps its
    # inode from being handed to a directory made anew.
    tmpdir = tmp_path / "node" / "roles" / "gpu0" / "tmp"
    (tmpdir / "sub").mkdir(parents=True)
    (tmpdir / "sub" / "f").write_text("y")
    held = os.open(tmpdir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        _launcher(tmp_path).launch(_spec(tmp_path, "gpu0"))
        assert os.path.samestat(os.fstat(held), tmpdir.stat()) and list(tmpdir.iterdir()) == []
    finally:
        os.close(held)


def test_the_launcher_reads_its_nodes_size_from_the_controller(tmp_path):
    run = FakeRun().on("scontrol", done("NodeName=node04 CPUTot=64 CfgTRES=cpu=64,mem=1T,gres/gpu=8\n"))
    launcher = _launcher(tmp_path, commands=SlurmCommands(run=run), node="node04")
    assert launcher.node_size() == NodeSize(gpus=8, cpus=64)
    assert run.calls[0][1:] == ["show", "node", "--oneliner", "node04"]


def test_a_figure_the_controller_does_not_give_is_read_from_the_node_itself(tmp_path):
    # A controller that does not answer leaves the node's own size: a CPU node with no GPU driver.
    failed = FakeRun().on("scontrol", done(stderr="slurm_load_node error", rc=1))
    launcher = _launcher(tmp_path, commands=SlurmCommands(run=failed), node="c3101",
                         node_root=node_root(tmp_path / "c3101", online="0-191\n", gpus=None))
    assert launcher.node_size() == NodeSize(gpus=0, cpus=192)
    # A record that lacks one figure keeps the other: the controller's CPUs, the node's GPUs.
    partial = FakeRun().on("scontrol", done("NodeName=node04 CfgTRES=cpu=64,mem=1T,gres/gpu=x\n"))
    launcher = _launcher(tmp_path, commands=SlurmCommands(run=partial), node="node04",
                         node_root=node_root(tmp_path / "node04", online="0-111\n", gpus=8))
    assert launcher.node_size() == NodeSize(gpus=8, cpus=64)


def test_a_steps_evaluations_on_a_node_the_controller_does_not_describe_are_the_nodes_own(tmp_path):
    # A package's pip step of eight CPUs on a 192-CPU node holds 24 of its size, not the eight a
    # GPU node holds; a one-GPU step on a node whose driver lists eight GPUs, eight.
    root = node_root(tmp_path / "node04", online="0-191\n", gpus=8)
    failed = FakeRun().on("scontrol", *[done(stderr="slurm_load_node error", rc=1)] * 2)
    launcher = _launcher(tmp_path, commands=SlurmCommands(run=failed), node="node04", node_root=root)
    assert node_evaluations(launcher.node_size(), gpus=0, cpus=8) == 192 // 8
    assert node_evaluations(launcher.node_size(), gpus=1, cpus=8) == 8


def test_env_fd_needs_an_env_file(tmp_path):
    with pytest.raises(ValueError, match="env_fd"):
        RoleSpec(name="gpu0", argv=("x",), shape=GPU1, owner="harness", log_dir=tmp_path, env_fd=True)


@pytest.mark.parametrize("name", ["../x", "a/b", "..", ".hidden", "", "-x", "a b", "gpu0\n"])
def test_a_role_name_is_one_plain_path_segment(tmp_path, name):
    with pytest.raises(ValueError, match="role name"):
        RoleSpec(name=name, argv=("x",), shape=GPU1, owner="harness", log_dir=tmp_path)


@pytest.mark.parametrize("name", ["gpu0", "noop_a", "artifact.2", "a-b", "0"])
def test_plain_role_names_are_accepted(tmp_path, name):
    assert RoleSpec(name=name, argv=("x",), shape=GPU1, owner="harness", log_dir=tmp_path).name == name


def test_a_shared_step_takes_every_gpu_and_narrows_by_uuid(tmp_path):
    argv = srun_argv(_spec(tmp_path, "overlap"), python="p", report=Path("/r"), scratch=Path("/s"), shared=[U2],
                     allocation_gpus=2)
    assert argv[1] == "--overlap" and "--exact" not in argv and "--gpus=2" in argv
    assert "--export=ALL" in argv
    assert argv[argv.index("--share-gpus") + 1] == U2


class FakeProc:
    def __init__(self, rc: int | None = None, *, dies_on_term: bool = False) -> None:
        self.returncode = rc
        self.signals: list[str | int] = []
        self.dies_on_term = dies_on_term

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.signals.append("KILL")
        self.returncode = -9

    def send_signal(self, sig):
        self.signals.append(int(sig))
        if self.dies_on_term:
            self.returncode = 137


class WritesReportWhenPolled(FakeProc):
    """The step writes its report and ends between the handle's first read and its first poll."""

    def __init__(self, write, rc: int) -> None:
        super().__init__()
        self._write = write
        self._rc = rc

    def poll(self):
        if self.returncode is None:
            self._write()
            self.returncode = self._rc
        return self.returncode


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _step_cgroup(tmp_path: Path, n: int = 0, job: str = JOB) -> Path:
    return tmp_path / "cg" / "system.slice" / "slurmstepd.scope" / f"job_{job}" / f"step_{n}" / "user"


def _write_report(tmp_path: Path, name: str, uuids: list[str], *, n: int = 0, job: str = JOB,
                  mismatches: tuple[str, ...] = (), cgroup_dir: Path | None = None, step: str | None = None) -> Path:
    """Write the report a step of this job would, and return its path."""
    granted = GrantedResources(gpus=len(uuids), gpu_kind="H200", cpus=1, memory_gib=2.0, disk_free_gib=9.0)
    cgroup = cgroup_dir if cgroup_dir is not None else _step_cgroup(tmp_path, n, job)
    cgroup.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "logs" / f"{name}.report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(RoleReport(host="g", step=step or f"{job}.{n}", cgroup_dir=str(cgroup), gpu_uuids=uuids,
                               cuda_visible_devices="0", granted=granted, swap_max_bytes=None,
                               mismatches=list(mismatches)).model_dump_json())
    return path


def _handle(tmp_path: Path, proc: FakeProc | None = None, *, run: FakeRun | None = None, name: str = "gpu0",
            clock: Clock | None = None, allocation: list[str] | None = None) -> SlurmRoleHandle:
    clock = clock or Clock()
    return SlurmRoleHandle(_spec(tmp_path, name), proc or FakeProc(), tmp_path / "logs" / f"{name}.report.json",
                           job_id=JOB, cgroup_root=tmp_path / "cg", allocation_gpu_uuids=allocation,
                           commands=SlurmCommands(run=run or FakeRun()), clock=clock, sleep=clock.sleep)


def _launcher(tmp_path: Path, popen=None, *, cappers: bool = False, **kw) -> SlurmLauncher:
    """A launcher whose processes *popen* starts: srun alone, the log cappers too with *cappers*,
    and otherwise a capper that has ended."""
    srun = popen or (lambda argv, **kw: FakeProc())

    def start(argv, **kwargs):
        if not cappers and "hls.substrate.log_cap" in argv:
            return FakeProc(0)
        return srun(argv, **kwargs)

    return SlurmLauncher(python="/venv/bin/python", scratch=tmp_path / "node", allocation_gpus=2,
                         batch_env=BATCH_ENV, cgroup_root=tmp_path / "cg", popen=start, **kw)


def _stop(tmp_path: Path, handle: SlurmRoleHandle, run: FakeRun) -> None:
    """Stop *handle* gracefully the one way there is, through a launcher that runs scancel with *run*."""
    _launcher(tmp_path, commands=SlurmCommands(run=run)).terminate([handle])


def test_a_shared_role_runs_on_the_target_roles_gpus_by_uuid(tmp_path):
    launched = []

    def popen(argv, **kw):
        launched.append((argv, kw["env"]))
        return FakeProc()

    launcher = _launcher(tmp_path, popen)
    target = launcher.launch(_spec(tmp_path, "gpu1"))
    _write_report(tmp_path, "gpu1", [U2])
    launcher.launch(_spec(tmp_path, "overlap"), share_gpus_of=target)
    argv, env = launched[1]
    assert argv[1] == "--overlap" and "--gpus=2" in argv
    assert argv[argv.index("--share-gpus") + 1] == U2
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_a_shared_role_waits_for_no_report_of_its_target(tmp_path):
    # The stage starts a role only once its target reported, so its report is there or never comes.
    clock = Clock()
    target = _handle(tmp_path, name="gpu1", clock=clock)
    with pytest.raises(TimeoutError):
        _launcher(tmp_path).launch(_spec(tmp_path, "overlap"), share_gpus_of=target)
    assert clock.now == 0.0


def test_sharing_more_gpus_than_declared_is_refused(tmp_path):
    launcher = _launcher(tmp_path)
    target = launcher.launch(_spec(tmp_path, "gpu1", GPU1.model_copy(update={"gpus": 2})))
    _write_report(tmp_path, "gpu1", [U1, U2])
    with pytest.raises(ValueError, match="declares 1 GPUs but would share 2"):
        launcher.launch(_spec(tmp_path, "overlap"), share_gpus_of=target)


def test_sharing_with_a_role_whose_grant_differs_is_refused(tmp_path):
    launcher = _launcher(tmp_path)
    target = launcher.launch(_spec(tmp_path, "gpu1"))
    _write_report(tmp_path, "gpu1", [U1], mismatches=("gpus: declared 1, granted 2",))
    with pytest.raises(ValueError, match="grant differs"):
        launcher.launch(_spec(tmp_path, "overlap"), share_gpus_of=target)


def test_a_shared_role_with_no_gpus_is_refused(tmp_path):
    launcher = _launcher(tmp_path)
    target = launcher.launch(_spec(tmp_path, "gpu1"))
    _write_report(tmp_path, "gpu1", [U1])
    cpu = ResourceShape(cpus=1, memory_gib=1, time_sec=60)
    with pytest.raises(ValueError, match="declares no GPUs"):
        launcher.launch(_spec(tmp_path, "overlap", cpu), share_gpus_of=target)


def test_a_role_made_ready_launches_with_no_new_space_on_the_nodes_disk(tmp_path, monkeypatch):
    """A role whose directory was made before a contestant filled the node's disk, its step's
    ``tmp`` and its log cappers' error files in it, logging off that disk: the launch makes
    nothing new there."""
    node = tmp_path / "node"
    role = node / "roles" / "gpu0"
    (role / "tmp").mkdir(parents=True)
    for suffix in (".out", ".err"):
        log_cap.errors_file(role, suffix).touch()
    refused = refusing_new_space(monkeypatch, node)
    launched: list = []
    launcher = _launcher(tmp_path, lambda argv, **kw: launched.append(argv) or FakeProc(0), cappers=True,
                         log_keep_bytes=4096)
    launcher.launch(_spec(tmp_path, "gpu0"))
    assert refused == [] and len(launched) == 3


def test_a_tmpdir_left_deeper_than_the_interpreter_recurses_is_emptied_at_the_next_launch(tmp_path):
    tmpdir = tmp_path / "node" / "roles" / "gpu0" / "tmp"
    p = tmpdir
    for _ in range(sys.getrecursionlimit() + 100):
        p = p / "a"
        p.mkdir(parents=True)
    _launcher(tmp_path).launch(_spec(tmp_path, "gpu0"))
    assert tmpdir.is_dir() and list(tmpdir.iterdir()) == []


def test_each_launch_starts_with_an_empty_tmpdir_and_no_old_report(tmp_path):
    launcher = _launcher(tmp_path)
    spec = _spec(tmp_path, "gpu0")
    launcher.launch(spec)
    tmpdir = tmp_path / "node" / "roles" / "gpu0" / "tmp"
    (tmpdir / "old").write_text("x")
    (tmpdir / "sub").mkdir()
    (tmpdir / "sub" / "f").write_text("y")
    report = _write_report(tmp_path, "gpu0", [U1])
    marker = tmp_path / "logs" / "gpu0.started"
    marker.write_text(f"{JOB}.0\n")
    handle = launcher.launch(spec)
    assert tmpdir.is_dir() and list(tmpdir.iterdir()) == []
    assert not report.exists() and not marker.exists() and not handle.started()


def test_a_step_has_started_once_its_wrapper_says_so_or_it_reported(tmp_path):
    handle = _handle(tmp_path)
    assert not handle.started()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "gpu0.started").write_text(f"{JOB}.3\n")
    assert handle.started()
    reported = _handle(tmp_path, name="gpu1")
    assert not reported.started()
    _write_report(tmp_path, "gpu1", [U1])
    assert reported.started()


def test_the_start_marker_lies_beside_the_grant_report():
    assert role_exec.started_path(Path("/l/extract-ab.2.report.json")) == Path("/l/extract-ab.2.started")


#: What ``squeue --steps`` answers for the job while two other roles run.
STEPS = f"{JOB}.batch|batch|RUNNING|\n{JOB}.3|server|RUNNING|\n{JOB}.4|driver|RUNNING|\n"
SRUN_WAITS = f"srun: Job {JOB} step creation temporarily disabled, retrying (Requested nodes are busy)\n"


def _launch_evidence(tmp_path: Path, squeue, err: str | None, *, name: str = "gpu0") -> tuple[dict, FakeRun]:
    run = FakeRun().on("squeue", squeue)
    if err is not None:
        (tmp_path / "logs").mkdir(exist_ok=True)
        (tmp_path / "logs" / f"{name}.err").write_text(err)
    return _handle(tmp_path, run=run, name=name).launch_evidence(), run


def test_launch_evidence_is_the_steps_slurm_lists_under_the_roles_name_and_what_srun_said(tmp_path):
    evidence, run = _launch_evidence(tmp_path, done(STEPS + f"{JOB}.5|gpu0|RUNNING|\n"), SRUN_WAITS)
    assert evidence == {"slurm_steps": f"{JOB}.5 RUNNING", "srun_said": SRUN_WAITS.strip()}
    assert run.argv_of("squeue") == [["squeue", "--steps", f"--jobs={JOB}", "--noheader",
                                      "--Format=StepId:|,StepName:|,StepState:|"]]


def test_a_step_slurm_has_not_created_is_listed_as_none_and_a_quiet_srun_said_nothing(tmp_path):
    evidence, _ = _launch_evidence(tmp_path, done(STEPS), None)
    assert evidence == {"slurm_steps": "none", "srun_said": "nothing"}


def test_launch_evidence_keeps_only_sruns_last_lines(tmp_path):
    err = "".join(f"srun: line {i}\n" for i in range(10)) + "role_exec: a wrapper line\n" + "srun: " + "x" * 900
    evidence, _ = _launch_evidence(tmp_path, done(""), err)
    said = evidence["srun_said"].split(" | ")
    assert said[:2] == ["srun: line 8", "srun: line 9"] and len(said) == 3 and len(said[2]) == 300


def test_a_controller_that_does_not_answer_is_named_in_the_evidence(tmp_path):
    evidence, _ = _launch_evidence(tmp_path, done(rc=1, stderr="squeue: error: Socket timed out on send/recv"), None)
    assert evidence["slurm_steps"] == "squeue failed: squeue: error: Socket timed out on send/recv"
    timed_out, _ = _launch_evidence(tmp_path, subprocess.TimeoutExpired(["squeue"], 10.0), None)
    assert timed_out["slurm_steps"] == "squeue timed out"


def test_the_default_commands_give_squeue_ten_seconds(tmp_path):
    handle = SlurmRoleHandle(_spec(tmp_path, "gpu0"), FakeProc(), tmp_path / "r.json")
    assert handle.commands.timeout_sec["squeue"] == 10.0


def test_the_launchers_handles_check_reports_against_this_job_and_allocation(tmp_path):
    launcher = _launcher(tmp_path, allocation_gpu_uuids=[U1, U2])
    handle = launcher.launch(_spec(tmp_path, "gpu0"))
    _write_report(tmp_path, "gpu0", [U1], job="999")
    with pytest.raises(RoleFailed, match="job 999"):
        handle.report(1.0)
    handle = launcher.launch(_spec(tmp_path, "gpu1"))
    _write_report(tmp_path, "gpu1", ["GPU-elsewhere"])
    with pytest.raises(RoleFailed, match="not in the allocation"):
        handle.report(1.0)


def test_report_waits_and_fails_when_the_step_ends_without_one(tmp_path):
    spec = _spec(tmp_path, "gpu0")
    now = [0.0]
    handle = SlurmRoleHandle(spec, FakeProc(), tmp_path / "missing.json", clock=lambda: now[0],
                             sleep=lambda s: now.__setitem__(0, now[0] + s))
    with pytest.raises(TimeoutError):
        handle.report(1.0)
    ended = SlurmRoleHandle(spec, FakeProc(rc=1), tmp_path / "missing.json")
    with pytest.raises(RoleFailed, match="status 1"):
        ended.report(1.0)
    # The wrapper's refusal of a ceiling its uid already holds says so.
    ceiling = SlurmRoleHandle(dataclasses.replace(spec, nproc=4096), FakeProc(rc=role_exec.PROCESS_CEILING_EXIT),
                              tmp_path / "missing.json")
    with pytest.raises(RoleFailed, match="already held as many tasks as its process ceiling of 4096"):
        ceiling.report(1.0)


def test_report_reads_once_more_after_the_step_has_ended(tmp_path):
    # The wrapper writes its report and the role ends right after the handle's first look.
    path = tmp_path / "logs" / "gpu0.report.json"
    proc = WritesReportWhenPolled(lambda: _write_report(tmp_path, "gpu0", [U1]), rc=0)
    handle = _handle(tmp_path, proc)
    assert handle.report(5.0).gpu_uuids == [U1] and path.exists()


@pytest.mark.parametrize("text", ['{"host": "g"}', "not json", ""])
def test_a_report_that_does_not_validate_fails_at_once(tmp_path, text):
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "gpu0.report.json").write_text(text)
    handle = _handle(tmp_path)
    with pytest.raises(RoleFailed, match="invalid grant report"):
        handle.report(60.0)


@pytest.mark.parametrize("make, message", [
    (lambda tp: {"job": "999"}, "job 999"),
    (lambda tp: {"step": "oops"}, "not <job>.<step>"),
    (lambda tp: {"n": 2, "step": f"{JOB}.3"}, "step_3/user"),
    (lambda tp: {"cgroup_dir": tp / "elsewhere" / f"job_{JOB}" / "step_0" / "user"}, "is not step_0/user"),
])
def test_a_report_for_another_step_is_refused(tmp_path, make, message):
    _write_report(tmp_path, "gpu0", [U1], **make(tmp_path))
    with pytest.raises(RoleFailed, match=message):
        _handle(tmp_path).report(1.0)


def test_a_report_that_climbs_out_of_the_cgroup_root_is_refused(tmp_path):
    sneaky = tmp_path / "cg" / ".." / "elsewhere" / f"job_{JOB}" / "step_0" / "user"
    _write_report(tmp_path, "gpu0", [U1], cgroup_dir=sneaky)
    with pytest.raises(RoleFailed, match="is not step_0/user"):
        _handle(tmp_path).report(1.0)


def test_a_report_with_gpus_outside_the_allocation_is_refused(tmp_path):
    _write_report(tmp_path, "gpu0", [U1, "GPU-elsewhere"])
    with pytest.raises(RoleFailed, match="not in the allocation"):
        _handle(tmp_path, allocation=[U1, U2]).report(1.0)
    _write_report(tmp_path, "gpu0", [U1])
    assert _handle(tmp_path, allocation=[U1, U2]).report(1.0).gpu_uuids == [U1]


def test_a_graceful_stop_asks_the_reported_step_to_stop_with_scancel(tmp_path):
    run = FakeRun().on("scancel", done())
    proc = FakeProc()
    _write_report(tmp_path, "gpu0", [U1], n=3)
    _stop(tmp_path, _handle(tmp_path, proc, run=run), run)
    assert run.argv_of("scancel") == [["scancel", "--signal=TERM", f"{JOB}.3"]]
    assert proc.signals == []


def test_terminating_several_roles_is_one_scancel_for_every_reported_step(tmp_path):
    run = FakeRun().on("scancel", done())
    procs = [FakeProc(), FakeProc(), FakeProc(), FakeProc(rc=0)]
    launcher = _launcher(tmp_path, lambda argv, **kw: procs.pop(0), commands=SlurmCommands(run=run))
    handles = [launcher.launch(_spec(tmp_path, f"gpu{i}")) for i in range(4)]
    for i in (0, 1, 3):
        _write_report(tmp_path, f"gpu{i}", [U1], n=i)
    launcher.terminate(handles)
    assert run.argv_of("scancel") == [["scancel", "--signal=TERM", f"{JOB}.0", f"{JOB}.1"]]
    # gpu2 has not reported, so srun gets SIGTERM; gpu3 has ended.
    assert [h.proc.signals for h in handles] == [[], [], [15], []]
    assert [h.stopped_by_runner for h in handles] == [True, True, True, False]


def test_terminating_a_role_another_launcher_started_is_refused_before_any_signal(tmp_path):
    run = FakeRun()
    launcher = _launcher(tmp_path, commands=SlurmCommands(run=run))
    ours = launcher.launch(_spec(tmp_path, "gpu0"))

    class Foreign:
        spec = _spec(tmp_path, "other")

    with pytest.raises(TypeError, match="other"):
        launcher.terminate([ours, Foreign()])
    assert run.calls == [] and ours.proc.signals == [] and not ours.stopped_by_runner


def test_a_graceful_stop_before_the_report_signals_srun(tmp_path):
    # The wrapper has not started the role yet, and srun turns SIGTERM into a step-wide SIGKILL.
    run = FakeRun()
    proc = FakeProc()
    _stop(tmp_path, _handle(tmp_path, proc, run=run), run)
    assert proc.signals == [15] and run.calls == []


def test_a_graceful_stop_does_nothing_once_the_step_has_ended(tmp_path):
    run = FakeRun()
    proc = FakeProc(rc=3)
    _write_report(tmp_path, "gpu0", [U1])
    _stop(tmp_path, _handle(tmp_path, proc, run=run), run)
    assert proc.signals == [] and run.calls == []


def test_kill_signals_srun_and_escalates_after_five_seconds(tmp_path):
    clock = Clock()
    proc = FakeProc()
    _write_report(tmp_path, "gpu0", [U1])
    _handle(tmp_path, proc, clock=clock).kill()
    assert proc.signals == [15, "KILL"]
    assert 5.0 <= clock.now < 6.0


def test_kill_does_not_escalate_when_srun_ends(tmp_path):
    proc = FakeProc(dies_on_term=True)
    _handle(tmp_path, proc).kill()
    assert proc.signals == [15]


def test_kill_kills_every_process_in_the_reported_step(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        _write_report(tmp_path, "gpu0", [U1])
        (_step_cgroup(tmp_path) / "cgroup.procs").write_text(f"{child.pid}\n")
        _handle(tmp_path, FakeProc(dies_on_term=True)).kill()
        assert child.wait(10) == -9
    finally:
        if child.poll() is None:
            child.kill()


def test_kill_does_not_trust_a_report_for_another_step(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        (elsewhere / "cgroup.procs").write_text(f"{child.pid}\n")
        _write_report(tmp_path, "gpu0", [U1], cgroup_dir=elsewhere)
        _handle(tmp_path, FakeProc(dies_on_term=True)).kill()
        with pytest.raises(subprocess.TimeoutExpired):
            child.wait(0.5)
    finally:
        child.kill()
        child.wait(10)


@pytest.mark.parametrize("state, by_scheduler", [
    ("FAILED", False), ("COMPLETED", False), ("OUT_OF_MEMORY", False),
    # A step's time limit is one the harness set, so a step that reached it was ended for it.
    ("TIMEOUT", True),
    ("CANCELLED", True), ("CANCELLED by 1000", True), ("NODE_FAIL", True), ("PREEMPTED", True),
    ("BOOT_FAIL", True),
])
def test_a_nonzero_exit_is_judged_by_the_steps_sacct_state(tmp_path, state, by_scheduler):
    run = FakeRun().on("sacct", done(f"{state}\n"))
    _write_report(tmp_path, "gpu0", [U1], n=2)
    handle = _handle(tmp_path, FakeProc(rc=3), run=run)
    assert handle.ended_by_scheduler() is by_scheduler
    assert run.argv_of("sacct") == [["sacct", "-j", f"{JOB}.2", "--noheader", "--parsable2", "--format=State"]]


def test_the_sacct_answer_is_kept(tmp_path):
    run = FakeRun().on("sacct", done("CANCELLED by 1000\n"))
    _write_report(tmp_path, "gpu0", [U1])
    handle = _handle(tmp_path, FakeProc(rc=3), run=run)
    assert handle.ended_by_scheduler() and handle.ended_by_scheduler()
    assert len(run.argv_of("sacct")) == 1


@pytest.mark.parametrize("state, expected", [
    ("OUT_OF_MEMORY", JobState.OUT_OF_MEMORY), ("FAILED", JobState.FAILED), ("CANCELLED by 1000", JobState.CANCELLED),
])
def test_the_end_state_is_the_steps_sacct_state_from_the_same_query(tmp_path, state, expected):
    run = FakeRun().on("sacct", done(f"{state}\n"))
    _write_report(tmp_path, "gpu0", [U1], n=2)
    handle = _handle(tmp_path, FakeProc(rc=1), run=run)
    assert handle.end_state() is expected
    handle.ended_by_scheduler()
    assert handle.end_state() is expected and len(run.argv_of("sacct")) == 1


@pytest.mark.parametrize("rc, report, answers", [
    (0, True, []), (-9, True, []), (1, False, []), (3, True, [done("RUNNING\n")] * 40),
], ids=["clean exit", "srun killed", "no report", "never terminal"])
def test_no_end_state_without_a_terminal_sacct_answer(tmp_path, rc, report, answers):
    run = FakeRun().on("sacct", *answers).on("scancel", done())
    if report:
        _write_report(tmp_path, "gpu0", [U1])
    assert _handle(tmp_path, FakeProc(rc=rc), run=run).end_state() is None


def test_a_stopped_role_has_no_end_state(tmp_path):
    run = FakeRun().on("scancel", done())
    proc = FakeProc(dies_on_term=True)
    _write_report(tmp_path, "gpu0", [U1])
    handle = _handle(tmp_path, proc, run=run)
    _stop(tmp_path, handle, run)
    proc.returncode = 143
    assert handle.end_state() is None and run.argv_of("sacct") == []


def test_a_step_still_accounted_as_running_is_asked_again_every_two_seconds(tmp_path):
    run = FakeRun().on("sacct", done("RUNNING\n"), done("COMPLETING\n"), done(""), done("FAILED\n"))
    _write_report(tmp_path, "gpu0", [U1])
    clock = Clock()
    assert _handle(tmp_path, FakeProc(rc=3), run=run, clock=clock).ended_by_scheduler() is False
    assert len(run.argv_of("sacct")) == 4 and clock.now == 6.0


def test_an_unreachable_controller_is_asked_again(tmp_path):
    down = done(stderr="sacct: error: slurm_persist_conn_open: Unable to contact slurm controller", rc=1)
    run = FakeRun().on("sacct", down, down, done("CANCELLED by 1000\n"))
    _write_report(tmp_path, "gpu0", [U1])
    assert _handle(tmp_path, FakeProc(rc=3), run=run).ended_by_scheduler() is True
    assert len(run.argv_of("sacct")) == 3


def test_no_terminal_answer_in_thirty_seconds_counts_as_the_schedulers(tmp_path):
    run = FakeRun().on("sacct", *[done("RUNNING\n")] * 40).on("scancel", done())
    _write_report(tmp_path, "gpu0", [U1])
    clock = Clock()
    assert _handle(tmp_path, FakeProc(rc=3), run=run, clock=clock).ended_by_scheduler() is True
    assert clock.now == 30.0 and len(run.argv_of("sacct")) == 16


def test_a_step_with_no_terminal_state_is_cancelled_by_its_recorded_id(tmp_path):
    # srun ended but accounting never says the step is over: make sure the role cannot keep running.
    run = FakeRun().on("sacct", *[done("RUNNING\n")] * 40).on("scancel", done())
    _write_report(tmp_path, "gpu0", [U1], n=4)
    assert _handle(tmp_path, FakeProc(rc=3), run=run).ended_by_scheduler() is True
    assert run.argv_of("scancel") == [["scancel", f"{JOB}.4"]]
    assert run.calls[-1][0] == "scancel"


@pytest.mark.parametrize("answers", [
    [done("FAILED\n")],
    [done("CANCELLED by 1000\n")],
    [done(stderr="Unable to contact slurm controller", rc=1), done("NODE_FAIL\n")],
])
def test_a_step_with_a_terminal_state_is_not_cancelled_again(tmp_path, answers):
    run = FakeRun().on("sacct", *answers)
    _write_report(tmp_path, "gpu0", [U1])
    _handle(tmp_path, FakeProc(rc=3), run=run).ended_by_scheduler()
    assert run.argv_of("scancel") == []


def test_a_failing_scancel_does_not_change_the_answer(tmp_path):
    run = FakeRun().on("sacct", *[done("RUNNING\n")] * 40).on("scancel", done(stderr="Invalid job id specified", rc=1))
    _write_report(tmp_path, "gpu0", [U1])
    assert _handle(tmp_path, FakeProc(rc=3), run=run).ended_by_scheduler() is True


def test_the_default_commands_give_sacct_and_scancel_ten_seconds(tmp_path):
    spec = _spec(tmp_path, "gpu0")
    handle = SlurmRoleHandle(spec, FakeProc(), tmp_path / "r.json")
    assert handle.commands.timeout_sec["sacct"] == 10.0 and handle.commands.timeout_sec["scancel"] == 10.0
    launcher = SlurmLauncher(python="p", scratch=tmp_path, allocation_gpus=1, batch_env=BATCH_ENV,
                             popen=lambda argv, **kw: FakeProc())
    assert launcher.commands.timeout_sec["sacct"] == 10.0 and launcher.commands.timeout_sec["scancel"] == 10.0
    assert launcher.launch(spec).commands is launcher.commands


def test_a_step_killed_by_a_signal_the_runner_did_not_send_is_the_schedulers(tmp_path):
    run = FakeRun()
    _write_report(tmp_path, "gpu0", [U1])
    assert _handle(tmp_path, FakeProc(rc=-9), run=run).ended_by_scheduler() is True
    assert run.calls == []


def test_a_clean_exit_is_not_asked_about(tmp_path):
    run = FakeRun()
    assert _handle(tmp_path, FakeProc(rc=0), run=run).ended_by_scheduler() is False
    assert run.calls == []


def test_a_step_that_never_reported_cannot_be_confirmed(tmp_path):
    run = FakeRun()
    assert _handle(tmp_path, FakeProc(rc=1), run=run).ended_by_scheduler() is True
    assert run.calls == []


@pytest.mark.parametrize("stop", ["terminate", "kill"])
def test_a_role_the_runner_stopped_is_not_the_schedulers(tmp_path, stop):
    run = FakeRun().on("scancel", done())
    proc = FakeProc(dies_on_term=True)
    _write_report(tmp_path, "gpu0", [U1])
    handle = _handle(tmp_path, proc, run=run)
    if stop == "terminate":
        _stop(tmp_path, handle, run)
    else:
        handle.kill()
    proc.returncode = -9
    assert handle.ended_by_scheduler() is False
    assert run.argv_of("sacct") == []


def test_ended_by_scheduler_needs_an_ended_step(tmp_path):
    with pytest.raises(RuntimeError, match="has not ended"):
        _handle(tmp_path, FakeProc()).ended_by_scheduler()


def test_a_graceful_stop_is_the_launchers_alone():
    # One API per operation: Launcher.terminate signals every role in one go; a handle can only
    # kill its own role.
    class Handle(RoleHandle):
        def started(self): ...
        def report(self, timeout_sec): ...
        def poll(self): ...
        def wait(self, timeout_sec=None): ...
        def kill(self): ...
        def ended_by_scheduler(self): ...
        def end_state(self): ...

    Handle()
    assert not hasattr(RoleHandle, "terminate") and not hasattr(SlurmRoleHandle, "terminate")

    class NoStop(Launcher):
        def launch(self, spec, *, share_gpus_of=None): ...

    with pytest.raises(TypeError, match="terminate"):
        NoStop()


@pytest.mark.parametrize("owner", ["contestant", "harness"])
def test_every_roles_stdout_and_stderr_go_through_log_cappers(tmp_path, owner):
    launched = []

    def popen(argv, **kw):
        launched.append((list(argv), kw))
        return FakeProc(0)

    launcher = _launcher(tmp_path, popen, cappers=True, log_keep_bytes=4096)
    handle = launcher.launch(replace(_spec(tmp_path, "art"), owner=owner))
    cappers = [(argv, kw) for argv, kw in launched if "hls.substrate.log_cap" in argv]
    logs = tmp_path / "logs"
    assert [argv for argv, _ in cappers] == [
        ["/venv/bin/python", "-m", "hls.substrate.log_cap", "--keep", "4096", str(logs / f"art.{s}")]
        for s in ("out", "err")]
    [(srun, kw)] = [(argv, kw) for argv, kw in launched if "hls.substrate.log_cap" not in argv]
    # srun writes into the cappers' pipes, which this process no longer holds open.
    assert isinstance(kw["stdout"], int) and isinstance(kw["stderr"], int)
    for fd in (kw["stdout"], kw["stderr"], *(c["stdin"] for _, c in cappers)):
        with pytest.raises(OSError):
            os.fstat(fd)
    handle.close_logs(1.0)


def test_a_container_role_runs_as_apptainer_behind_its_seal_with_its_env_file_on_descriptor_3(tmp_path, monkeypatch):
    """The launcher, not the stage, turns a container role into its step: the command line of
    ``ApptainerRuntime.argv`` reading the env file on descriptor 3, behind the seal for a sealed
    one, whose host-side variables (its files, the site's pasta and upstream proxy, the limits it
    writes) are the step's declared ones."""
    monkeypatch.delenv(SEAL_TOOL_PATH_VAR, raising=False)
    started: list = []
    launcher = _launcher(tmp_path, lambda argv, **kw: started.append((argv, kw["env"])) or FakeProc(0),
                         pasta=Path("/opt/passt/pasta"), upstream_proxy="http://proxy.site:3128")
    image = PinnedImage(ref="/node/sandboxes/x", file_sha256="0" * 64, check_sha256="1" * 64)
    seal = SealPlan(policy=SealPolicy(network=NetworkPolicy()), index=False)
    env_file = tmp_path / "node" / "roles" / "srv" / "container.env"
    for sealed in (None, seal):
        spec = ContainerSpec(image=image, command=("true",), workdir=Path("/node/roles/srv/wd"), seal=sealed)
        files = None if sealed is None else seal_files(Path("/node/roles/srv/seal"))
        limits = {} if sealed is None else {"max_user_namespaces": 64}
        role = RoleSpec(name="srv", argv=(), shape=GPU1, owner="contestant", log_dir=tmp_path / "logs",
                        start_marker=START_MARKER,
                        container=ContainerRole(spec=spec, env_file=env_file, seal_files=files, ucounts=limits))
        handle = launcher.launch(role)
        argv, env = started.pop()
        assert argv[argv.index("--") + 1:] == ApptainerRuntime().argv(spec, env_file=role_exec.ENV_FD_PATH,
                                                                      seal_python="/venv/bin/python")
        assert argv[argv.index("--env-file") + 1] == str(env_file) and "--env-fd" in argv[:argv.index("--")]
        declared = {} if files is None else {
            **seal_env(files, seal, pasta=Path("/opt/passt/pasta"), upstream_proxy="http://proxy.site:3128"),
            "HLS_SEAL_UCOUNTS": "max_user_namespaces=64"}
        assert {k: env[k] for k in env if k.startswith("HLS_")} == declared
        assert (handle.spec.container, handle.spec.env_file, handle.spec.env_fd) == (None, env_file, True)
