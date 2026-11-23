"""Docker on one host as a substrate: what each member means there, the allocation's grant from
its reservation, and the host's cgroup check."""

from __future__ import annotations

import dataclasses
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hls.spec.resources import ResourceShape
from hls.substrate.docker import substrate as module
from hls.substrate.docker.launcher import DockerRuntime
from hls.substrate.docker.substrate import cgroup_problem, docker, measure_grant, spawner_problem
from hls.substrate.factory import Substrate
from hls.substrate.local.allocation import LocalPlacement
from hls.substrate.local.placement import Reservation
from tests.substrate.docker.fake_daemon import FakeDocker

GPU = "GPU-11111111-1111-1111-1111-111111111111"


def site(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(store_root=tmp_path / "store", node_scratch=tmp_path / "scratch",
                           docker=SimpleNamespace(socket=tmp_path / "d.sock", owner_label="owner=hls",
                                                  buildkit_image="moby/buildkit@sha256:" + "b" * 64,
                                                  gateway_memory_gib=3),
                           local=SimpleNamespace(state_dir=tmp_path / "state", kept_memory_gib=4.0))


def test_every_member_is_given_and_the_slurm_shaped_ones_say_what_they_mean(tmp_path, monkeypatch):
    on = docker(site(tmp_path))
    assert isinstance(on, Substrate) and on.name == "docker"
    assert all(getattr(on, f.name) is not None for f in dataclasses.fields(Substrate))
    shape = ResourceShape(cpus=2, memory_gib=2, time_sec=60)
    local = on.placement(partition=None, qos=None, socket_tasks=None, exclude=())
    assert isinstance(local, LocalPlacement)
    with pytest.raises(ValueError, match="a local allocation has no partition"):
        on.placement(partition="gpu", qos=None, socket_tasks=None, exclude=())
    assert on.is_job_log("job-17.out") and not on.is_job_log("slurm-17.out")
    assert on.allocation_qos(3600, site(tmp_path)) is None
    assert on.site_request(shape, site(tmp_path), gpu_kind="H200") is shape
    with pytest.raises(RuntimeError):
        on.job_cgroup()
    assert isinstance(on.runtime(), DockerRuntime)
    monkeypatch.setenv("HLS_JOB_ID", "1700000000")
    assert on.images(tmp_path / "scratch").image_file == "image.oci.tar"
    assert on.measure_grant is measure_grant
    for name in ("HLS_JOB_ID", "HLS_STEP_ID", "NVIDIA_VISIBLE_DEVICES", "PATH"):
        assert on.reserved_variable(name)
    assert not on.reserved_variable("HF_HOME")
    assert on.job_id({"HLS_JOB_ID": "1700000000"}) == "1700000000"
    assert on.spawner_problem is spawner_problem
    # A sealed role's gateway is a container of the step image: the host runs no pasta and no
    # firewall tool of its own, so nothing of the pasta seal is asked of it.
    assert on.seal_problems({"PATH": "/nonexistent"}) == []


def _proc_sys(tmp_path: Path, **values: str) -> Path:
    """A /proc/sys of user namespaces at their defaults, and *values* (``kernel__name``) beside them."""
    root = tmp_path / "sys"
    for name in ("user/max_user_namespaces", "user/max_mnt_namespaces", "user/max_pid_namespaces"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("63000\n")
    for key, value in values.items():
        path = root / key.replace("__", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{value}\n")
    return root


def test_a_host_that_gives_the_spawner_its_namespaces_has_no_spawner_problem(tmp_path):
    assert spawner_problem(proc_sys=_proc_sys(tmp_path)) is None
    assert spawner_problem(proc_sys=_proc_sys(tmp_path, kernel__apparmor_restrict_unprivileged_userns="0",
                                              kernel__unprivileged_userns_clone="1")) is None


@pytest.mark.parametrize("values, says", [
    ({"kernel__apparmor_restrict_unprivileged_userns": "1"},
     "kernel.apparmor_restrict_unprivileged_userns is 1; set it to 0"),
    ({"kernel__unprivileged_userns_clone": "0"}, "kernel.unprivileged_userns_clone is 0; set it to 1"),
    ({"user__max_user_namespaces": "0"}, "user.max_user_namespaces is 0; set it to a number above 0"),
    ({"user__max_pid_namespaces": "0", "kernel__apparmor_restrict_unprivileged_userns": "1"},
     "kernel.apparmor_restrict_unprivileged_userns is 1; set it to 0; user.max_pid_namespaces is 0; set it to a "
     "number above 0"),
], ids=["apparmor", "debian-switch", "no-user-namespaces", "two-at-once"])
def test_a_host_that_refuses_the_spawner_its_namespaces_is_refused_naming_what_to_set(tmp_path, values, says):
    """Each kernel setting that keeps an unprivileged user from the spawner's namespaces or their
    mounts is named, on one line, with the value to set; nothing a retry elsewhere could mend."""
    problem = spawner_problem(proc_sys=_proc_sys(tmp_path, **values))
    assert problem is not None and not problem.retryable and "\n" not in problem.reason
    assert problem.reason.endswith(says) and "the build sandbox's spawner makes" in problem.reason


def test_a_kernel_without_user_namespaces_is_refused(tmp_path):
    (tmp_path / "sys").mkdir()
    problem = spawner_problem(proc_sys=tmp_path / "sys")
    assert problem is not None and "CONFIG_USER_NS" in problem.reason


def held(**over) -> Reservation:
    base = dict(gpus=(GPU,), cpus=(2, 3), held_cpus=(2, 3), memory_gib=8)
    base.update(over)
    return Reservation(**base)


def smi(lines: str, code: int = 0):
    return lambda argv, **kwargs: subprocess.CompletedProcess(argv, code, lines, "")


def test_the_allocations_grant_is_its_reservation(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "reservation", lambda state, job: held())
    measured = measure_grant({"HLS_JOB_ID": "17", "HLS_LOCAL_STATE": str(tmp_path)}, tmp_path,
                             run=smi(f"{GPU}, NVIDIA H200\nGPU-2, NVIDIA H200\n"), affinity=lambda pid: {2, 3})
    assert measured.problems == [] and measured.gpu_uuids == [GPU] and measured.swap_max_bytes == 0
    g = measured.granted
    assert (g.gpus, g.gpu_kind, g.cpus, g.memory_gib) == (1, "H200", 2, 8)


def test_a_reservation_the_host_does_not_show_is_a_problem(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "reservation", lambda state, job: held())
    measured = measure_grant({"HLS_JOB_ID": "17", "HLS_LOCAL_STATE": str(tmp_path)}, tmp_path,
                             run=smi("GPU-2, NVIDIA H200\n"), affinity=lambda pid: {2, 3, 4})
    assert any("does not list the reserved GPUs" in p for p in measured.problems)
    assert any("not the reserved [2, 3]" in p for p in measured.problems)


def test_an_unreadable_reservation_refuses_the_grant(tmp_path):
    with pytest.raises(RuntimeError, match="reservation cannot be read"):
        measure_grant({}, tmp_path)


def test_a_docker_hosts_tools_are_recorded_and_a_slurm_nodes_record_is_unchanged(tmp_path):
    from hls.substrate.versions import ToolVersions

    daemon = FakeDocker(tmp_path / "d.sock")
    proc = tmp_path / "proc"
    (proc / "driver/nvidia").mkdir(parents=True)
    (proc / "driver/nvidia/version").write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  570.211.01  Tue Nov 25 16:36:36 UTC 2025\n")
    with daemon.serving():
        tools = module.measure_tools(daemon.path, "c" * 64, proc=proc)
    assert (tools.docker, tools.step_image, tools.nvidia_driver) == ("29.5.2", "c" * 64, "570.211.01")
    assert tools.slurm is None and tools.apptainer is None and tools.pasta is None
    assert ToolVersions.model_validate_json(tools.model_dump_json()) == tools
    slurm = ToolVersions(slurm="slurm 23.02.7", apptainer=None, pasta=None, kernel="k")
    assert slurm.model_dump_json() == '{"slurm":"slurm 23.02.7","apptainer":null,"pasta":null,"kernel":"k"}'
    unreachable = module.measure_tools(tmp_path / "none.sock", "c" * 64, proc=tmp_path / "none")
    assert unreachable.docker is None and unreachable.nvidia_driver is None


def test_the_hosts_cgroups_and_daemon_are_checked(tmp_path):
    root = tmp_path / "cgroup"
    root.mkdir()
    assert "not a cgroup v2 mount" in cgroup_problem(tmp_path / "d.sock", root=root).reason
    (root / "cgroup.controllers").write_text("cpu io memory\n")
    assert "no pids, cpuset controller" in cgroup_problem(tmp_path / "d.sock", root=root).reason
    (root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    assert "cannot be used" in cgroup_problem(tmp_path / "nothing.sock", root=root).reason
    daemon = FakeDocker(tmp_path / "d.sock")
    with daemon.serving():
        assert cgroup_problem(daemon.path, root=root) is None
        daemon.info["CgroupDriver"] = "none"
        assert "neither systemd nor cgroupfs" in cgroup_problem(daemon.path, root=root).reason


def test_the_launcher_is_held_to_the_reservation_and_dials_out_through_the_sites_proxy(tmp_path, monkeypatch):
    made = {}

    class Client:
        def __init__(self, socket):
            made["socket"] = socket

        def check(self):
            return None

    monkeypatch.setenv("HLS_JOB_ID", "17")
    monkeypatch.setenv("HLS_LOCAL_STATE", str(tmp_path / "state"))
    monkeypatch.setattr(module, "reservation", lambda state, job: held(gpus=()))
    monkeypatch.setattr(module, "node_pid_max", lambda: 4096)
    monkeypatch.setattr(module, "DockerClient", Client)
    stepped = {}

    def step_image_file(cache):
        stepped["store"] = cache.store.root
        return tmp_path / "store" / "step.oci.tar", "e" * 64

    def load_step_image(client, archive, *, sha256, scratch):
        stepped["loaded"] = (archive, sha256)
        return "sha256:" + "f" * 64

    monkeypatch.setattr(module, "step_image_file", step_image_file)
    monkeypatch.setattr(module, "load_step_image", load_step_image)
    monkeypatch.setattr(module, "code_dirs", lambda python, checkout: (checkout,))
    monkeypatch.setattr(module, "DockerLauncher", lambda **kwargs: made.update(kwargs) or "launcher")
    on = docker(site(tmp_path))
    assert on.launcher(python=f"{tmp_path}/checkout/.venv/bin/python", scratch=tmp_path / "scratch",
                       allocation_gpus=0, allocation_gpu_uuids=[], pasta=None,
                       upstream_proxy="http://proxy.example:3128") == "launcher"
    assert made["upstream_proxy"] == "http://proxy.example:3128" and made["gateway_memory_gib"] == 3
    assert (made["cpus"], made["memory_gib"], made["allocation_gpu_uuids"]) == ((2, 3), 8, ())
    assert made["labels"] == {"owner": "hls"} and made["socket"] == tmp_path / "d.sock"
    # The step image comes from the store's cache, checked against the sha256 its entry records.
    assert stepped["store"] == tmp_path / "store"
    assert stepped["loaded"] == (tmp_path / "store" / "step.oci.tar", "e" * 64)
    assert made["step_image"] == "sha256:" + "f" * 64


def test_an_allocation_is_submitted_once_the_store_holds_the_step_image(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "ensure_step_image",
                        lambda cache, **kwargs: calls.append(("ensure", cache.store.root, kwargs)))
    monkeypatch.setattr(module.LocalScheduler, "submit", lambda self, request: calls.append(("submit", request)) or "1")
    monkeypatch.setattr(module.LocalScheduler, "__init__", lambda self, **kwargs: None)
    scheduler = docker(site(tmp_path)).scheduler()
    assert scheduler.submit("request") == "1"
    [(ensured, store, kwargs), submitted] = calls
    assert (ensured, store, submitted) == ("ensure", tmp_path / "store", ("submit", "request"))
    assert kwargs == {"socket": tmp_path / "d.sock", "buildkit_image": "moby/buildkit@sha256:" + "b" * 64,
                      "work_root": tmp_path / "scratch"}


def test_a_step_image_that_cannot_be_built_refuses_the_submission(tmp_path, monkeypatch):
    from hls.substrate.base import Refused

    def ensure(cache, **kwargs):
        raise RuntimeError("the step image could not be built: the build exited 1")

    monkeypatch.setattr(module, "ensure_step_image", ensure)
    monkeypatch.setattr(module.LocalScheduler, "submit", lambda self, request: pytest.fail("submitted"))
    monkeypatch.setattr(module.LocalScheduler, "__init__", lambda self, **kwargs: None)
    with pytest.raises(Refused, match="not in the store: the step image could not be built"):
        docker(site(tmp_path)).scheduler().submit("request")


def test_a_host_without_the_docker_cli_is_refused_before_any_allocation(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "docker_cli_problem", lambda: "this host has no working the Docker CLI (docker)")
    with pytest.raises(ValueError, match="no working the Docker CLI .*; install the Docker CLI with buildx"):
        docker(site(tmp_path)).check_placement(ResourceShape(cpus=1, memory_gib=1, time_sec=60), site(tmp_path),
                                               None, job_name="j")


def test_an_allocation_whose_store_holds_no_step_image_is_refused_before_its_roles(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "cgroup_problem", lambda socket: None)
    problem = docker(site(tmp_path)).cgroup_problem()
    assert problem is not None and not problem.retryable and "holds no step image images/step-" in problem.reason
