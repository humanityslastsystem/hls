"""The Docker launcher against the fake daemon (``fake_daemon``) and a fake host
(``launcher_fakes``): each container's settings, the gate's FIFOs, the host-side grant check,
the end rule, the parent-cgroup OOM count, stopping, removal and the stopper."""

from __future__ import annotations

import grp
import json
import os
import pwd
import resource
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from hls.egress.policy import NetworkPolicy, SealPolicy
from hls.spec.resources import ResourceShape
from hls.substrate.base import JobState, RoleFailed, RoleSpec
from hls.substrate.container import ContainerRole, ContainerSpec, Mount, PinnedImage, SealPlan
from hls.substrate.docker import launcher as launcher_module
from hls.substrate.docker import step
from hls.substrate.docker.api import DaemonIdentity
from hls.substrate.docker.launcher import (
    CONTAINER_HOME,
    GATE,
    LOST_STATUS,
    NOT_STARTED_STATUS,
    DockerRoleHandle,
    DockerStopper,
    cgroup_parent,
    code_dirs,
    oom_kills,
    reserved_variable,
    USER_FILES_LABEL,
)
from hls.substrate.docker.seal import GATEWAY_LABEL, LISTENING
from hls.substrate.docker.seccomp import spawner_profile
from hls.substrate.exits import GRANT_MISMATCH_EXIT, PROCESS_CEILING_EXIT
from hls.substrate.seal import PayloadEnding, payload_ending
from hls.substrate.secrets import write_env_file
from tests.substrate.docker.launcher_fakes import (
    DRIVER_VERSION,
    GPUS,
    IMAGE_GROUP,
    IMAGE_PASSWD,
    JOB,
    ROLE_IMAGE,
    STEP_IMAGE,
    fake_world,
    launcher,
    parent_dir,
    play_gate,
    wait_for,
)

SECRET = "s3cret-value-never-shown"
MARKER = "hls-container-started"
SHA = "a" * 64


@pytest.fixture
def world(tmp_path):
    with fake_world(tmp_path) as w:
        yield w


def shape(**over) -> ResourceShape:
    base = dict(gpus=0, cpus=2, memory_gib=2, disk_gib=0, time_sec=600)
    base.update(over)
    return ResourceShape(**base)


def container_spec(tmp_path: Path, name: str = "serve", *, env: dict[str, str] | None = None, gpus: bool = False,
                   owner: str = "contestant", mounts=(), seal=None, confinement: str = "default",
                   **over) -> RoleSpec:
    role_dir = tmp_path / "scratch" / "roles" / name
    role_dir.mkdir(parents=True, exist_ok=True)
    env_file = role_dir / "container.env"
    write_env_file(env_file, env if env is not None else {"API_TOKEN": SECRET, "MODE": "it's fine"})
    workdir = role_dir / "work"
    workdir.mkdir(exist_ok=True)
    c = ContainerSpec(image=PinnedImage(ref=ROLE_IMAGE, file_sha256=SHA, check_sha256=SHA),
                      command=("/bin/server", "--port", "8000"), workdir=workdir, mounts=tuple(mounts), gpus=gpus,
                      seal=seal, confinement=confinement)
    files = None
    if seal is not None:
        from hls.substrate.seal import seal_files
        (role_dir / "seal").mkdir(mode=0o700, exist_ok=True)
        files = seal_files(role_dir / "seal")
    base = dict(name=name, argv=(), shape=shape(), owner=owner, log_dir=tmp_path / "store" / "logs",
                start_marker=MARKER, container=ContainerRole(spec=c, env_file=env_file, seal_files=files))
    base.update(over)
    return RoleSpec(**base)


def driver_targets(host) -> dict[str, Path]:
    """The driver's files a GPU role's container holds, by their paths in it."""
    return {target: source for source, target in host.driver.binds()}


def harness_spec(tmp_path: Path, name: str = "noop", **over) -> RoleSpec:
    base = dict(name=name, argv=(sys.executable, "-m", "hls.substrate.noop_role", "--out", "x"), shape=shape(),
                owner="harness", log_dir=tmp_path / "store" / "logs", env={"HLS_NOOP": "1"})
    base.update(over)
    return RoleSpec(**base)


def body(daemon, handle: DockerRoleHandle) -> dict:
    wait_for(lambda: handle.container_id is not None or handle.poll() is not None, what="the container")
    return next(b for b in daemon.create_bodies if b["Labels"]["hls.step"] == handle.step
                and GATEWAY_LABEL not in b["Labels"] and USER_FILES_LABEL not in b["Labels"])


# -- every container's settings --------------------------------------------------------------


def test_a_container_role_is_created_with_every_setting_the_design_lists(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    spec = container_spec(tmp_path, mounts=(Mount(tmp_path / "store", "/data", read_only=True),), gpus=True,
                          shape=shape(gpus=1, gpu_kind="H200"), nproc=512)
    handle = launch.launch(spec)
    gate = play_gate(daemon, host, handle, marker=MARKER)
    assert handle.wait(10) == 0
    gate.join()
    b = body(daemon, handle)
    h = b["HostConfig"]
    assert b["User"] == f"{os.getuid()}:{os.getgid()}" and not b["User"].startswith("0:")
    assert h["CapDrop"] == ["ALL"] and h["CapAdd"] == [] and h["Privileged"] is False
    assert h["SecurityOpt"] == ["no-new-privileges"]
    assert h["ReadonlyRootfs"] is True and h["Init"] is True and h["NetworkMode"] == "host"
    assert h["CgroupnsMode"] == "private" and h["LogConfig"]["Type"] == "none"
    assert all(m["Type"] == "bind" for m in h["Mounts"])
    targets = {m["Target"]: m for m in h["Mounts"]}
    assert targets["/.hls-gate"]["ReadOnly"] is True
    assert set(targets) == {"/.hls-gate", "/tmp", "/var/tmp", CONTAINER_HOME, "/data", "/etc/passwd", "/etc/group",
                            *driver_targets(host)}
    # The user database: the image's own with the host user's and group's entries, bound read-only.
    for name in ("passwd", "group"):
        assert targets[f"/etc/{name}"]["Source"] == os.path.realpath(handle._gate_dir / name)
        assert targets[f"/etc/{name}"]["ReadOnly"] is True
    assert targets["/data"]["ReadOnly"] is True and targets["/tmp"]["ReadOnly"] is False
    assert h["Memory"] == h["MemorySwap"] == h["ShmSize"] == 2 * 2**30
    assert h["PidsLimit"] == 512 and h["Ulimits"] == [{"Name": "nproc", "Soft": 512, "Hard": 512}]
    # The gate raises a contestant's score from inside, so a process of the role may lower its own
    # again; a score Docker set would be the floor of every process in the container.
    assert "OomScoreAdj" not in h
    assert h["CpusetCpus"] == "2,3"
    # Its GPU is its device node and the driver's, under runc, never the NVIDIA runtime's hook.
    assert h["Runtime"] == "runc" and "DeviceRequests" not in h
    assert h["Devices"] == [{"PathOnHost": str(host.dev / name), "PathInContainer": f"/dev/{name}",
                             "CgroupPermissions": "rw"}
                            for name in ("nvidia0", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset")]
    assert {t: (m["Source"], m["ReadOnly"]) for t, m in targets.items() if t.startswith("/usr/local/nvidia/")} == {
        t: (str(source), True) for t, source in driver_targets(host).items()}
    assert h["CgroupParent"] == "hls-c2-0.slice"
    assert b["Labels"] == {"hls.test": "docker-launcher", "hls.job": JOB, "hls.role": "serve",
                           "hls.step": f"{JOB}.0"}
    assert b["Image"] == ROLE_IMAGE
    assert b["Entrypoint"] == ["/bin/sh", "/.hls-gate/gate", "--nvidia", DRIVER_VERSION, "--oom-score-adj", "1000",
                               "--marker", MARKER, "--", "/bin/server", "--port", "8000"]
    assert sorted(b["Env"]) == sorted([f"HOME={CONTAINER_HOME}", f"NVIDIA_VISIBLE_DEVICES={GPUS[0]}",
                                       f"CUDA_VISIBLE_DEVICES={GPUS[0]}"])
    assert b["WorkingDir"] == CONTAINER_HOME


def test_a_role_knows_its_user_by_the_images_database_with_the_host_users_entry_appended(world, tmp_path):
    """As Apptainer gives a container: every line of the image's /etc/passwd and /etc/group, then
    the host user's entry and the host group's, read through a helper container of the image
    that carries the job's labels, is never started, and is removed."""
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    gate = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    gate.join()
    user, group = pwd.getpwuid(os.getuid()), grp.getgrgid(os.getgid())
    passwd = (handle._gate_dir / "passwd").read_text()
    assert passwd == (IMAGE_PASSWD.decode()
                      + f"{user.pw_name}:x:{user.pw_uid}:{user.pw_gid}::{user.pw_dir}:{user.pw_shell}\n")
    assert (handle._gate_dir / "group").read_text() == IMAGE_GROUP.decode() + f"{group.gr_name}:x:{group.gr_gid}:\n"
    [helper] = [b for b in daemon.create_bodies if USER_FILES_LABEL in b["Labels"]]
    assert helper["Labels"] == {"hls.test": "docker-launcher", "hls.job": JOB, "hls.role": "serve",
                                "hls.step": f"{JOB}.0", USER_FILES_LABEL: "1"}
    assert helper["Image"] == ROLE_IMAGE and helper["HostConfig"]["NetworkMode"] == "none"
    # Removed, and never started: the only container left is the role's.
    assert list(daemon.containers) == [handle.container_id]
    assert not any(m == "POST" and p.endswith("/start") and handle.container_id not in p
                   for m, p, _, _ in daemon.requests)
    handle.kill()


def test_an_image_whose_user_files_cannot_be_read_refuses_the_role_and_its_helper_is_removed(world, tmp_path):
    daemon, client, host = world
    image = next(i for i, img in daemon.images.items() if ROLE_IMAGE in img["RepoTags"])
    del daemon.trees[image]["/etc/group"]
    spec = container_spec(tmp_path)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    assert handle.wait(10) == NOT_STARTED_STATUS
    handle.close_logs(10)
    err = (spec.log_dir / "serve.err").read_text()
    assert "FATAL: docker launcher: the image's /etc/passwd and /etc/group cannot be read: " in err
    assert len(err.strip().splitlines()) == 1
    # The helper was made and removed; the role's container never was.
    assert [USER_FILES_LABEL in b["Labels"] for b in daemon.create_bodies] == [True]
    assert daemon.containers == {}


def test_a_two_gpu_role_gets_the_nodes_of_its_two_cards_and_no_other(world, tmp_path):
    """The create body of a role given two of the host's GPUs, whose device nodes are not numbered
    in nvidia-smi's order: the two granted cards' nodes, found by their minor numbers, and the
    driver's, and no device request; the harness role's wrapper runs behind the same gate."""
    daemon, client, host = world
    host.minors = {GPUS[0]: 6, GPUS[1]: 3, "GPU-33333333-3333-3333-3333-333333333333": 0}
    launch = launcher(client, host, tmp_path / "scratch")
    handle = launch.launch(container_spec(tmp_path, gpus=True, shape=shape(gpus=2, gpu_kind="H200")))
    play_gate(daemon, host, handle, marker=MARKER, stay=True).join()
    h = body(daemon, handle)["HostConfig"]
    assert h["Runtime"] == "runc" and "DeviceRequests" not in h
    assert [(d["PathOnHost"], d["PathInContainer"], d["CgroupPermissions"]) for d in h["Devices"]] == [
        (str(host.dev / name), f"/dev/{name}", "rw")
        for name in ("nvidia6", "nvidia3", "nvidiactl", "nvidia-uvm", "nvidia-uvm-tools", "nvidia-modeset")]
    assert body(daemon, handle)["Entrypoint"][2:4] == ["--nvidia", DRIVER_VERSION]
    report = handle.report(5)
    assert report.mismatches == [] and report.gpu_uuids == [GPUS[1], GPUS[0]]
    handle.kill()
    harness = launch.launch(harness_spec(tmp_path, "h", shape=shape(gpus=2, gpu_kind="H200")))
    wait_for(lambda: harness.container_id is not None, what="the harness container")
    hb = body(daemon, harness)
    assert [d["PathInContainer"] for d in hb["HostConfig"]["Devices"]][:2] == ["/dev/nvidia6", "/dev/nvidia3"]
    assert hb["Entrypoint"][:5] == ["/bin/sh", "/.hls-gate/gate", "--nvidia", DRIVER_VERSION, "--"]
    assert driver_targets(host).keys() <= {m["Target"] for m in hb["HostConfig"]["Mounts"]}
    harness.kill()


def test_a_role_without_gpus_gets_no_device_and_none_of_the_drivers_files(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    play_gate(daemon, host, handle, marker=MARKER, stay=True).join()
    b = body(daemon, handle)
    assert "Devices" not in b["HostConfig"] and b["HostConfig"]["Runtime"] == "runc"
    assert not any(m["Target"].startswith("/usr/local/nvidia") for m in b["HostConfig"]["Mounts"])
    assert "--nvidia" not in b["Entrypoint"]
    handle.kill()


@pytest.mark.parametrize("change, why", [
    (lambda host: (host.proc / "driver/nvidia/version").write_text(
        "NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.95.05  Thu Sep 25 2025\n"),
     "runs NVIDIA driver 580.95.05, and the local scheduler recorded driver 570.211.01"),
    (lambda host: (host.dev / "nvidia-uvm").unlink(), "nvidia-uvm does not exist"),
    (lambda host: (host.dev / "nvidia1").unlink(), "nvidia1 does not exist"),
    (lambda host: host.minors.pop(GPUS[1]), "does not list the allocation's GPUs"),
])
def test_a_host_whose_driver_a_gpu_role_cannot_be_given_refuses_the_launcher(world, tmp_path, change, why):
    daemon, client, host = world
    change(host)
    with pytest.raises(RuntimeError, match=why):
        launcher(client, host, tmp_path / "scratch")
    with pytest.raises(RuntimeError, match="recorded no NVIDIA driver"):
        launcher(client, host, tmp_path / "scratch", driver=None)
    assert daemon.create_bodies == []


def test_a_container_with_no_gpu_names_none_so_an_image_cannot_choose(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path, owner="harness"))
    gate = play_gate(daemon, host, handle, marker=MARKER)
    handle.wait(10)
    gate.join()
    b = body(daemon, handle)
    assert "NVIDIA_VISIBLE_DEVICES=void" in b["Env"] and "DeviceRequests" not in b["HostConfig"]
    assert "OomScoreAdj" not in b["HostConfig"] and "--oom-score-adj" not in b["Entrypoint"]


def test_a_container_without_a_ceiling_has_the_runners_own_process_limit(world, tmp_path):
    """A Slurm step inherits the runner's ``RLIMIT_NPROC``, and a container is given it too, so the
    build sandbox's spawner makes the commands' user namespace under it as on Slurm; a Docker
    container's own is unlimited. A role with a process ceiling has its ceiling instead."""
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch", nproc_limit=(4000, 5000))
    handle = launch.launch(container_spec(tmp_path, "a"))
    play_gate(daemon, host, handle, marker=MARKER, stay=True).join()
    assert body(daemon, handle)["HostConfig"]["Ulimits"] == [{"Name": "nproc", "Soft": 4000, "Hard": 5000}]
    handle.kill()
    capped = launch.launch(container_spec(tmp_path, "b", nproc=512))
    play_gate(daemon, host, capped, marker=MARKER, stay=True).join()
    assert body(daemon, capped)["HostConfig"]["Ulimits"] == [{"Name": "nproc", "Soft": 512, "Hard": 512}]
    capped.kill()
    soft, hard = resource.getrlimit(resource.RLIMIT_NPROC)
    own = None if resource.RLIM_INFINITY in (soft, hard) else (soft, hard)
    assert launcher(client, host, tmp_path / "scratch").nproc_limit == own


def test_the_gate_is_the_launchers_own_script(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    wait_for(lambda: (handle._gate_dir / "gate").exists(), what="the gate")
    assert (handle._gate_dir / "gate").read_bytes() == GATE.read_bytes()
    assert oct((handle._gate_dir / "gate").stat().st_mode & 0o777) == "0o400"
    assert oct(handle._gate_dir.stat().st_mode & 0o777) == "0o700"
    gate = play_gate(daemon, host, handle, marker=MARKER)
    handle.wait(10)
    gate.join()
    assert not handle._gate_dir.exists()


# -- variables through the gate's FIFO -----------------------------------------------------


def test_variables_go_through_the_fifo_and_nowhere_else(world, tmp_path):
    daemon, client, host = world
    spec = container_spec(tmp_path)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    gate = play_gate(daemon, host, handle, marker=MARKER)
    assert handle.wait(10) == 0
    gate.join()
    assert gate.env == b"API_TOKEN=" + SECRET.encode() + b"\nMODE='it'\"'\"'s fine'\n"
    assert not spec.container.env_file.exists()
    for method, path, query, sent in daemon.requests:
        assert SECRET.encode() not in sent and SECRET not in json.dumps(query)


def test_a_malformed_env_file_is_a_grant_mismatch_and_nothing_is_passed(world, tmp_path):
    daemon, client, host = world
    spec = container_spec(tmp_path)
    spec.container.env_file.write_text("not an assignment\n")
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    assert handle.wait(10) == GRANT_MISMATCH_EXIT
    report = handle.report(1)
    assert any("is malformed" in m for m in report.mismatches)
    assert not spec.container.env_file.exists()


# -- the host-side grant -------------------------------------------------------------------


def test_the_grant_is_measured_from_the_host_and_reported(world, tmp_path):
    daemon, client, host = world
    spec = container_spec(tmp_path, gpus=True, shape=shape(gpus=1, gpu_kind="H200"))
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    gate = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    report = handle.report(10)
    assert handle.started() and handle.started_path.read_text() == f"{JOB}.0\n"
    assert report.mismatches == [] and report.step == f"{JOB}.0" and report.host == "fakehost"
    assert report.gpu_uuids == [GPUS[0]] and report.granted.gpus == 1 and report.granted.gpu_kind == "H200"
    assert report.cpus == [2, 3] and report.granted.cpus == 2 and report.granted.memory_gib == 2
    assert report.swap_max_bytes == 0
    assert Path(report.cgroup_dir) == host.cgroup_of[handle.container_id]
    assert Path(report.cgroup_dir).parent == parent_dir(host.cgroups, "hls-c2-0.slice")
    gate.join()
    host.ended(handle.container_id)
    daemon.exit(handle.container_id, 0)
    assert handle.wait(10) == 0


@pytest.mark.parametrize("override, why", [
    ({"memory.swap.max": "max"}, "lets the container swap"),
    ({"pids.max": "max"}, "holds the container to no tasks"),
    ({"memory.max": str(4 * 2**30)}, "memory_gib: declared 2"),
])
def test_a_grant_that_differs_stops_the_container_before_anything_ran(world, tmp_path, override, why):
    daemon, client, host = world
    host.override.update(override)
    spec = container_spec(tmp_path)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    assert handle.wait(10) == GRANT_MISMATCH_EXIT
    report = handle.report(0)
    assert any(why in m for m in report.mismatches), report.mismatches
    assert any(m == "POST" and p.endswith("/kill") and q["signal"] == ["SIGKILL"] for m, p, q, _ in daemon.requests)
    err = (spec.log_dir / "serve.err")
    handle.close_logs(10)
    assert "refusing to start, the container's grant differs" in err.read_text()
    assert daemon.containers == {}
    assert handle.ended_by_scheduler() is False


def test_gpus_the_devices_did_not_name_are_a_mismatch(world, tmp_path):
    daemon, client, host = world
    host.minors[GPUS[0]] = 7
    del host.minors[GPUS[1]]
    (host.proc).mkdir(exist_ok=True)
    spec = container_spec(tmp_path, gpus=True, shape=shape(gpus=1, gpu_kind="H200"))
    real = host.started

    def with_extra(cid, host_config):
        pid = real(cid, host_config)
        (host.proc / str(pid) / "root" / "dev" / "nvidia9").touch()
        return pid

    host.started = with_extra
    handle = launcher(client, host, tmp_path / "scratch", allocation_gpu_uuids=GPUS[:1]).launch(spec)
    assert handle.wait(10) == GRANT_MISMATCH_EXIT
    assert any("nvidia[9]" in m for m in handle.report(0).mismatches)


def test_a_role_held_to_one_numa_node_is_refused_when_split(world, tmp_path):
    daemon, client, host = world
    for node, cpus in ((0, "0-2"), (1, "3-7")):
        (host.root / "nodes" / f"node{node}").mkdir(parents=True)
        (host.root / "nodes" / f"node{node}" / "cpulist").write_text(cpus + "\n")
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path, one_numa_node=True))
    assert handle.wait(10) == GRANT_MISMATCH_EXIT
    report = handle.report(0)
    assert report.numa_split and report.numa_nodes == [0, 1]
    assert any("NUMA nodes [0, 1]" in m for m in report.mismatches)


def test_a_process_ceiling_the_uid_already_holds_refuses_before_anything_is_created(world, tmp_path):
    daemon, client, host = world
    spec = container_spec(tmp_path, nproc=1)
    handle = launcher(client, host, tmp_path / "scratch", proc=Path("/proc"), allocation_gpu_uuids=()).launch(spec)
    assert handle.wait(10) == PROCESS_CEILING_EXIT
    with pytest.raises(RoleFailed, match="process ceiling of 1"):
        handle.report(0)
    assert daemon.create_bodies == []
    assert not spec.container.env_file.exists()


# -- the end rule ---------------------------------------------------------------------------


def run_to_end(world, tmp_path, **gate) -> DockerRoleHandle:
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, **gate)
    handle.wait(10)
    g.join()
    return handle


def test_a_clean_exit_is_the_roles_own(world, tmp_path):
    handle = run_to_end(world, tmp_path, code=0)
    assert handle.poll() == 0 and not handle.ended_by_scheduler() and handle.end_state() is None


def test_a_nonzero_exit_is_the_roles_own_failure(world, tmp_path):
    handle = run_to_end(world, tmp_path, code=3)
    assert handle.poll() == 3 and not handle.ended_by_scheduler() and handle.end_state() is JobState.FAILED


def test_an_oom_kill_counted_in_the_parent_is_out_of_memory_though_docker_missed_it(world, tmp_path):
    handle = run_to_end(world, tmp_path, code=137, oom_kills=1, oom_flag=False)
    assert handle.poll() == 137 and not handle.ended_by_scheduler()
    assert handle.end_state() is JobState.OUT_OF_MEMORY


def test_docker_s_oom_flag_alone_is_out_of_memory(world, tmp_path):
    handle = run_to_end(world, tmp_path, code=137, oom_flag=True)
    assert handle.end_state() is JobState.OUT_OF_MEMORY


def test_an_unreadable_parent_count_leaves_the_end_unknown_never_not_out_of_memory(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    cid = handle.container_id
    host.ended(cid)
    (handle.parent / "memory.events").unlink()
    daemon.exit(cid, 137)
    assert handle.wait(10) == 137
    assert handle.ended_by_scheduler() is True and handle.end_state() is None


def test_a_container_not_under_its_parent_leaves_the_end_unknown(world, tmp_path):
    daemon, client, host = world
    real = host.started

    def elsewhere(cid, host_config):
        pid = real(cid, {**host_config, "CgroupParent": "system.slice"})
        return pid

    host.started = elsewhere
    handle = run_to_end(world, tmp_path, code=1)
    assert handle.ended_by_scheduler() is True and handle.end_state() is None


def test_a_kill_from_outside_is_the_schedulers(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    host.ended(handle.container_id)
    client.kill_container(handle.container_id, signal.SIGKILL)
    assert handle.wait(10) == 137
    assert handle.ended_by_scheduler() is True and handle.end_state() is JobState.CANCELLED


def test_another_daemon_process_than_at_launch_is_the_schedulers(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    host.identity = DaemonIdentity("engine", 100, 2)
    host.ended(handle.container_id)
    daemon.exit(handle.container_id, 1)
    assert handle.wait(10) == 1
    assert handle.ended_by_scheduler() is True and handle.end_state() is None


def test_a_broken_events_stream_leaves_the_end_unconfirmed(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    handle = launch.launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    launch._events._stream.close()
    host.ended(handle.container_id)
    daemon.exit(handle.container_id, 1)
    assert handle.wait(10) == 1
    assert handle.ended_by_scheduler() is True


def test_a_container_whose_state_is_gone_is_the_schedulers(world, tmp_path, monkeypatch):
    # The fake's removal sends no die event, so the handle stops waiting for one at once.
    monkeypatch.setattr(launcher_module, "DIE_EVENT_WAIT_SEC", 0.5)
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    host.ended(handle.container_id)
    client.remove_container(handle.container_id, force=True)
    assert handle.wait(10) in (137, LOST_STATUS)
    assert handle.ended_by_scheduler() is True


def test_the_time_limit_is_kept_and_is_the_schedulers(world, tmp_path):
    daemon, client, host = world
    spec = container_spec(tmp_path, shape=shape(time_sec=1))
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    assert handle.wait(10) == 137
    assert handle.ended_by_scheduler() is True and handle.end_state() is JobState.TIMEOUT
    handle.close_logs(10)
    assert "reached its time limit of 1s" in (spec.log_dir / "serve.err").read_text()


def test_every_container_is_removed_once_its_state_was_read(world, tmp_path):
    daemon, client, host = world
    run_to_end(world, tmp_path, code=0)
    assert daemon.containers == {}


# -- stopping -------------------------------------------------------------------------------


def test_terminate_sends_sigterm_and_the_end_is_the_runners(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    handle = launch.launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    wait_for(handle.started, what="the start")
    host.ended(handle.container_id)
    launch.terminate([handle])
    assert handle.wait(10) == 128 + signal.SIGTERM
    assert any(p.endswith("/kill") and q["signal"] == ["SIGTERM"] for _, p, q, _ in daemon.requests)
    assert handle.stopped_by_runner and handle.ended_by_scheduler() is False and handle.end_state() is None


def test_kill_sends_sigkill_and_returns_once_ended(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    host.ended(handle.container_id)
    handle.kill()
    assert handle.poll() == 137 and handle.stopped_by_runner and not handle.ended_by_scheduler()


def test_a_role_stopped_before_its_gate_is_released_never_runs(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    handle = launch.launch(container_spec(tmp_path))
    handle.kill()
    assert handle.poll() is not None
    assert daemon.containers == {}


def test_terminate_refuses_a_handle_of_another_launcher(world, tmp_path):
    class Other:
        spec = harness_spec(tmp_path)

    with pytest.raises(TypeError):
        launcher(world[1], world[2], tmp_path / "scratch").terminate([Other()])


# -- logs -----------------------------------------------------------------------------------


def test_logs_go_through_the_fifos_to_the_cappers(world, tmp_path):
    daemon, client, host = world
    spec = container_spec(tmp_path)
    handle = launcher(client, host, tmp_path / "scratch", log_keep_bytes=30).launch(spec)
    g = play_gate(daemon, host, handle, marker=MARKER, out=b"x" * 100, err=b"done\n")
    handle.wait(10)
    g.join()
    handle.close_logs(10)
    assert handle.logs_lost() is None
    out = (spec.log_dir / "serve.out").read_bytes()
    assert out.startswith(b"x" * 30) and b"bytes of this log were dropped" in out
    assert (spec.log_dir / "serve.err").read_bytes().startswith(MARKER.encode())


# -- harness roles --------------------------------------------------------------------------


def test_a_harness_role_runs_the_step_wrapper_in_the_step_image(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    spec = harness_spec(tmp_path, env_file=tmp_path / "scratch" / "roles" / "noop" / "x.env", nproc=900,
                        owner="contestant")
    (tmp_path / "scratch" / "roles" / "noop").mkdir(parents=True)
    handle = launch.launch(spec)
    g = play_gate(daemon, host, handle, stay=True)
    g.join()
    b = body(daemon, handle)
    assert b["Image"] == STEP_IMAGE and g.env == b""
    argv = b["Entrypoint"]
    assert argv[:3] == ["/bin/sh", "/.hls-gate/gate", "--"]
    wrapper = argv[3:argv.index("--", 3)]
    assert wrapper[:3] == [sys.executable, "-m", "hls.substrate.role_exec"]
    assert "--env-file" in wrapper and "--nproc" in wrapper and "--oom-score-adj" in wrapper
    assert argv[argv.index("--", 3) + 1:] == list(spec.argv)
    env = dict(e.split("=", 1) for e in b["Env"])
    assert env["HLS_SUBSTRATE"] == "docker" and env["HLS_STEP_ID"] == f"{JOB}.0" and env["HLS_NOOP"] == "1"
    # The step wrapper in the container finds its step id and limit directory through the factory.
    from hls.substrate.factory import step as factory_step
    assert factory_step(env).step_id(env) == f"{JOB}.0"
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void" and env["PATH"] == "/usr/bin:/bin"
    mounts = {m["Target"]: m for m in b["HostConfig"]["Mounts"]}
    assert mounts[sys.prefix]["ReadOnly"] is True and mounts[str(tmp_path / "store")]["ReadOnly"] is False
    assert mounts[str(tmp_path / "scratch")]["ReadOnly"] is False and mounts["/.hls-gate"]["ReadOnly"] is True
    assert mounts["/tmp"]["Source"] == str(tmp_path / "scratch" / "roles" / "noop" / "tmp")
    # The step wrapper reports its cgroup as the container sees it; the handle rewrites it.
    report = {"host": "fakehost", "step": f"{JOB}.0", "cgroup_dir": "/sys/fs/cgroup", "gpu_uuids": [],
              "cuda_visible_devices": None, "swap_max_bytes": 0, "mismatches": [],
              "granted": {"gpus": 0, "cpus": 2, "memory_gib": 2.0, "disk_free_gib": 10.0}}
    (spec.log_dir / "noop.report.json").write_text(json.dumps(report))
    got = handle.report(5)
    assert got.cgroup_dir == str(host.cgroup_of[handle.container_id])
    assert json.loads((spec.log_dir / "noop.report.json").read_text())["cgroup_dir"] == got.cgroup_dir
    host.ended(handle.container_id)
    daemon.exit(handle.container_id, 0)
    assert handle.wait(10) == 0


@pytest.mark.parametrize("field, value, why", [
    ("step", f"{JOB}.9", "for step"),
    ("cgroup_dir", "/sys/fs/cgroup/x", "reported the cgroup"),
    ("gpu_uuids", [GPUS[1]], "was not given"),
])
def test_a_harness_report_of_another_step_is_refused(world, tmp_path, field, value, why):
    daemon, client, host = world
    spec = harness_spec(tmp_path)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    g = play_gate(daemon, host, handle, stay=True)
    g.join()
    report = {"host": "fakehost", "step": f"{JOB}.0", "cgroup_dir": "/sys/fs/cgroup", "gpu_uuids": [],
              "cuda_visible_devices": None, "swap_max_bytes": 0, "mismatches": [],
              "granted": {"gpus": 0, "cpus": 2, "memory_gib": 2.0, "disk_free_gib": 10.0}, field: value}
    (spec.log_dir / "noop.report.json").write_text(json.dumps(report))
    with pytest.raises(RoleFailed, match=why):
        handle.report(5)
    handle.kill()


def test_a_harness_role_that_ends_without_reporting_fails_its_report(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(harness_spec(tmp_path))
    g = play_gate(daemon, host, handle, code=1)
    g.join()
    with pytest.raises(RoleFailed, match="ended with status 1 before reporting"):
        handle.report(5)


def test_a_harness_role_may_not_set_what_the_launcher_sets(world, tmp_path):
    daemon, client, host = world
    for name in ("PATH", "HLS_STEP_ID", "HLS_SUBSTRATE", "NVIDIA_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        assert reserved_variable(name)
        with pytest.raises(ValueError, match="may not set"):
            launcher(client, host, tmp_path / "scratch").launch(harness_spec(tmp_path, env={name: "x"}))
    assert not reserved_variable("HF_HOME")


def test_a_harness_role_whose_logs_are_outside_the_bound_directories_is_refused(world, tmp_path):
    daemon, client, host = world
    with pytest.raises(ValueError, match="not under a directory bound"):
        launcher(client, host, tmp_path / "scratch").launch(harness_spec(tmp_path, log_dir=tmp_path / "elsewhere"))


# -- GPUs, CPUs and slots -------------------------------------------------------------------


def test_exclusive_roles_hold_their_own_gpus_cpus_and_slots_and_a_shared_one_takes_its_targets(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    a = launch.launch(container_spec(tmp_path, "a", gpus=True, shape=shape(gpus=1, gpu_kind="H200")))
    b = launch.launch(container_spec(tmp_path, "b", gpus=True, shape=shape(gpus=1, gpu_kind="H200")))
    gates = [play_gate(daemon, host, h, marker=MARKER, stay=True) for h in (a, b)]
    for g in gates:
        g.join()
    a.report(5)
    shared = launch.launch(container_spec(tmp_path, "s", gpus=True, shape=shape(gpus=1, gpu_kind="H200",
                                                                                 cpus=1)), share_gpus_of=a)
    over = launch.launch(container_spec(tmp_path, "o", overlap=True, shape=shape(cpus=2)))
    gates = [play_gate(daemon, host, h, marker=MARKER, stay=True) for h in (shared, over)]
    for g in gates:
        g.join()
    bodies = {h.spec.name: body(daemon, h)["HostConfig"] for h in (a, b, shared, over)}
    cards = {n: [d["PathInContainer"] for d in bodies[n].get("Devices", ()) if d["PathInContainer"][-1].isdigit()]
             for n in "abso"}
    assert cards == {"a": ["/dev/nvidia0"], "b": ["/dev/nvidia1"], "s": ["/dev/nvidia0"], "o": []}
    assert (bodies["a"]["CpusetCpus"], bodies["b"]["CpusetCpus"]) == ("2,3", "4,5")
    assert (bodies["s"]["CpusetCpus"], bodies["o"]["CpusetCpus"]) == ("2", "2,3")
    assert [bodies[n]["CgroupParent"] for n in "abso"] == [f"hls-c2-{i}.slice" for i in range(4)]
    with pytest.raises(ValueError, match="0 of the allocation's are free"):
        launch.launch(container_spec(tmp_path, "c", gpus=True, shape=shape(gpus=1, gpu_kind="H200")))
    for h in (a, b, shared, over):
        host.ended(h.container_id)
        h.kill()
    # Ended roles give back what they held.
    c = launch.launch(container_spec(tmp_path, "d", gpus=True, shape=shape(gpus=2, gpu_kind="H200")))
    c.kill()
    assert body(daemon, c)["HostConfig"]["CgroupParent"] == "hls-c2-0.slice"
    assert [d["PathInContainer"] for d in body(daemon, c)["HostConfig"]["Devices"]][:2] == ["/dev/nvidia0",
                                                                                           "/dev/nvidia1"]


def test_two_roles_killed_at_their_limit_at_once_each_count_only_their_own(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    a = launch.launch(container_spec(tmp_path, "a"))
    b = launch.launch(container_spec(tmp_path, "b"))
    ga = play_gate(daemon, host, a, marker=MARKER, code=137, oom_kills=1)
    gb = play_gate(daemon, host, b, marker=MARKER, code=1)
    a.wait(10)
    b.wait(10)
    ga.join()
    gb.join()
    assert a.end_state() is JobState.OUT_OF_MEMORY and b.end_state() is JobState.FAILED
    assert launch.allocation_oom_kills() == 1


def test_a_slot_with_a_process_in_it_is_passed_over_and_recorded(world, tmp_path):
    daemon, client, host = world
    busy = parent_dir(host.cgroups, "hls-c2-0.slice") / "docker-other.scope"
    busy.mkdir(parents=True)
    (busy / "cgroup.procs").write_text("777\n")
    launch = launcher(client, host, tmp_path / "scratch")
    handle = launch.launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER)
    handle.wait(10)
    g.join()
    assert body(daemon, handle)["HostConfig"]["CgroupParent"] == "hls-c2-1.slice"
    assert launch.busy_slots == [busy.parent]


def test_the_parent_cgroups_are_named_for_the_daemons_driver(tmp_path):
    assert cgroup_parent("systemd", 4, 2, cgroup_root=tmp_path) == (
        "hls-c4-2.slice", tmp_path / "hls.slice" / "hls-c4.slice" / "hls-c4-2.slice")
    assert cgroup_parent("systemd", 4, None, cgroup_root=tmp_path) == (
        "hls-c4.slice", tmp_path / "hls.slice" / "hls-c4.slice")
    assert cgroup_parent("cgroupfs", 4, 2, cgroup_root=tmp_path) == ("/hls/c4/2", tmp_path / "hls" / "c4" / "2")
    with pytest.raises(ValueError):
        cgroup_parent("none", 4, 2)
    assert oom_kills(tmp_path / "missing", made=False) == 0 and oom_kills(tmp_path / "missing") is None


# -- what is refused ------------------------------------------------------------------------


def open_the_gateway(daemon, host, *, listens: bool = True) -> threading.Thread:
    """Play the gateway's part: once its container started, say its interceptor listens, or end it."""

    def run() -> None:
        wait_for(lambda: any(b["Labels"].get(GATEWAY_LABEL) for b in daemon.create_bodies), what="the gateway")
        body = next(b for b in daemon.create_bodies if b["Labels"].get(GATEWAY_LABEL))
        own = next(m["Source"] for m in body["HostConfig"]["Mounts"] if not m["ReadOnly"])
        cid = next(c for c, v in daemon.containers.items() if v["Config"]["Labels"].get(GATEWAY_LABEL))
        wait_for(lambda: cid in host.pid_of, what="the gateway to start")
        if listens:
            (Path(own) / "listening").write_text(LISTENING)
        else:
            daemon.exit(cid, 1)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread


def sealed_spec(tmp_path: Path, name: str = "serve", *, ingress_port: int | None = None, **over) -> RoleSpec:
    seal = SealPlan(policy=SealPolicy(network=NetworkPolicy()), index=False, ingress_port=ingress_port)
    return container_spec(tmp_path, name, seal=seal, **over)


def test_a_bind_reached_through_a_link_or_showing_the_daemons_socket_is_refused(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real)
    with pytest.raises(ValueError, match="reached through a link"):
        launch.launch(container_spec(tmp_path, "a", mounts=(Mount(tmp_path / "linked", "/data"),)))
    with pytest.raises(ValueError, match="not in normal form"):
        launch.launch(container_spec(tmp_path, "b", mounts=(Mount(real / ".." / "real", "/data"),)))
    # The fake daemon's socket lies in tmp_path: neither it nor the directory holding it is bound.
    with pytest.raises(ValueError, match="would show the Docker daemon's socket"):
        launch.launch(container_spec(tmp_path, "c", mounts=(Mount(tmp_path, "/data"),)))
    with pytest.raises(ValueError, match="would show the Docker daemon's socket"):
        launch.launch(container_spec(tmp_path, "d", mounts=(Mount(daemon.path, "/data"),)))
    assert daemon.create_bodies == []


def test_the_launchers_own_binds_are_their_real_paths_at_the_paths_named(world, tmp_path):
    daemon, client, host = world
    real = tmp_path / "python-3.12.16"
    real.mkdir()
    (tmp_path / "python-3.12").symlink_to(real)
    launch = launcher(client, host, tmp_path / "scratch", code_dirs=(Path(sys.prefix), tmp_path / "python-3.12"))
    handle = launch.launch(harness_spec(tmp_path))
    play_gate(daemon, host, handle, stay=True).join()
    mounts = {m["Target"]: m["Source"] for m in body(daemon, handle)["HostConfig"]["Mounts"]}
    assert mounts[str(tmp_path / "python-3.12")] == str(real)
    handle.kill()


def test_a_stop_while_the_gateway_starts_ends_the_role_at_once(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(sealed_spec(tmp_path))
    wait_for(lambda: any(v["Config"]["Labels"].get(GATEWAY_LABEL) and v["State"]["Running"]
                         for v in daemon.containers.values()), what="the gateway to start")
    began = time.monotonic()
    handle.kill()
    assert handle.poll() == NOT_STARTED_STATUS and time.monotonic() - began < 5
    assert handle.stopped_by_runner and handle.ended_by_scheduler() is False
    assert [b for b in daemon.create_bodies
            if GATEWAY_LABEL not in b["Labels"] and USER_FILES_LABEL not in b["Labels"]] == []
    assert daemon.containers == {}


def test_the_spawner_confinement_widens_seccomp_apparmor_and_proc_for_its_container_alone(world, tmp_path):
    """The agent build's sandbox asks for the ``spawner`` confinement: its container, and not its
    gateway, gets the spawner's seccomp profile, no AppArmor profile and no masked /proc paths,
    and keeps the host user, no capability and ``no-new-privileges``. Every other container keeps
    the daemon's own confinement."""
    daemon, client, host = world
    spec = sealed_spec(tmp_path, confinement="spawner")
    opener = open_the_gateway(daemon, host)
    launch = launcher(client, host, tmp_path / "scratch")
    handle = launch.launch(spec)
    gate = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    opener.join(10)
    gate.join()
    gateway, payload = next(b for b in daemon.create_bodies if GATEWAY_LABEL in b["Labels"]), body(daemon, handle)
    h = payload["HostConfig"]
    assert h["SecurityOpt"] == ["no-new-privileges", f"seccomp={spawner_profile()}", "apparmor=unconfined"]
    assert h["MaskedPaths"] == [] and h["ReadonlyPaths"] == []
    assert h["CapDrop"] == ["ALL"] and h["CapAdd"] == [] and h["Privileged"] is False
    assert payload["User"] == f"{os.getuid()}:{os.getgid()}"
    assert not any(m["Target"] == str(daemon.path) for m in h["Mounts"])
    assert gateway["HostConfig"]["SecurityOpt"] == ["no-new-privileges"]
    assert "MaskedPaths" not in gateway["HostConfig"]
    handle.kill()
    # A container that asks for no confinement keeps the daemon's.
    other = launch.launch(container_spec(tmp_path, "plain"))
    play_gate(daemon, host, other, marker=MARKER, stay=True).join()
    h = body(daemon, other)["HostConfig"]
    assert h["SecurityOpt"] == ["no-new-privileges"] and "MaskedPaths" not in h and "ReadonlyPaths" not in h
    other.kill()


def test_a_sealed_servers_port_is_published_on_the_loopback_alone(world, tmp_path):
    daemon, client, host = world
    spec = sealed_spec(tmp_path, ingress_port=8000)
    opener = open_the_gateway(daemon, host)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    opener.join(10)
    wait_for(lambda: handle.container_id is not None, what="the payload")
    gateway = next(b for b in daemon.create_bodies if GATEWAY_LABEL in b["Labels"])
    assert gateway["HostConfig"]["PortBindings"] == {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]}
    assert spec.container.seal_files.port.read_text().strip() == "49153"
    handle.kill()


def test_a_sealed_container_runs_in_its_gateways_namespace_and_its_ending_is_recorded(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch", memory_gib=4)
    spec = sealed_spec(tmp_path)
    opener = open_the_gateway(daemon, host)
    handle = launch.launch(spec)
    gate = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    opener.join(10)
    gate.join()
    gateway, payload = next(b for b in daemon.create_bodies if GATEWAY_LABEL in b["Labels"]), body(daemon, handle)
    assert gateway["Labels"][GATEWAY_LABEL] == "gateway" and gateway["Labels"]["hls.job"] == JOB
    assert gateway["Labels"]["hls.step"] == handle.step and gateway["Image"] == STEP_IMAGE
    assert sorted(gateway["HostConfig"]["CapAdd"]) == ["NET_ADMIN", "SETGID", "SETUID"]
    assert gateway["HostConfig"]["Memory"] == 2 * 2**30 and gateway["HostConfig"]["CpusetCpus"] == "2,3"
    gateway_id = next(c for c, v in daemon.containers.items() if v["Config"]["Labels"].get(GATEWAY_LABEL))
    assert payload["HostConfig"]["NetworkMode"] == f"container:{gateway_id}"
    assert payload["HostConfig"]["CapAdd"] == [] and payload["User"] != "0:0"
    files = spec.container.seal_files
    assert files.started.exists()
    # Two roles and a gateway would hold 6 of the allocation's 4 GiB.
    with pytest.raises(ValueError, match="hold 4 of its 4 GiB"):
        launch.launch(container_spec(tmp_path, "other"))
    host.ended(handle.container_id)
    daemon.exit(handle.container_id, 3)
    assert handle.wait(10) == 3
    assert payload_ending(files) == PayloadEnding(exit_code=3, interceptor_alive=True)
    assert daemon.containers == {}
    assert handle.end_state() is JobState.FAILED


def test_a_gateway_that_does_not_seal_ends_the_role_before_its_payload_is_made(world, tmp_path):
    daemon, client, host = world
    spec = sealed_spec(tmp_path)
    opener = open_the_gateway(daemon, host, listens=False)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    opener.join(10)
    assert handle.wait(10) == NOT_STARTED_STATUS
    with pytest.raises(RoleFailed, match="the seal's gateway did not seal"):
        handle.report(0)
    assert [b for b in daemon.create_bodies
            if not b["Labels"].get(GATEWAY_LABEL) and USER_FILES_LABEL not in b["Labels"]] == []
    assert payload_ending(spec.container.seal_files) is None and daemon.containers == {}
    handle.close_logs(10)
    assert (spec.log_dir / "serve.err").read_text().startswith("FATAL: docker launcher: the seal's gateway")


def test_a_sealed_payloads_end_is_recorded_when_it_is_seen_not_after_its_events(world, tmp_path, monkeypatch):
    daemon, client, host = world
    slow = launcher_module._Events.through_end

    def late(self, container_id, generation, timeout):
        time.sleep(2)
        return slow(self, container_id, generation, timeout)

    monkeypatch.setattr(launcher_module._Events, "through_end", late)
    spec = sealed_spec(tmp_path)
    opener = open_the_gateway(daemon, host)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    gate = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    opener.join(10)
    gate.join()
    host.ended(handle.container_id)
    ended = time.time()
    daemon.exit(handle.container_id, 0)
    assert handle.wait(10) == 0 and time.time() - ended >= 2
    pending = spec.container.seal_files.ended.with_name("ended.pending")
    assert pending.stat().st_mtime - ended < 1
    assert payload_ending(spec.container.seal_files) == PayloadEnding(exit_code=0, interceptor_alive=True)


def test_a_sealed_role_stopped_before_its_gate_records_no_ending(world, tmp_path):
    daemon, client, host = world
    spec = sealed_spec(tmp_path)
    opener = open_the_gateway(daemon, host)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    opener.join(10)
    wait_for(lambda: handle.container_id is not None and handle.container_id in host.pid_of, what="the payload")
    host.ended(handle.container_id)
    handle.kill()
    assert handle.poll() is not None and payload_ending(spec.container.seal_files) is None
    assert daemon.containers == {}


@pytest.mark.parametrize("target", ["/.hls-gate", "/tmp", "/proc/x", CONTAINER_HOME, "/dev"])
def test_a_mount_on_what_the_launcher_or_runtime_binds_is_refused(world, tmp_path, target):
    daemon, client, host = world
    spec = container_spec(tmp_path, mounts=(Mount(tmp_path, target),))
    with pytest.raises(ValueError, match="bound by the launcher or the runtime"):
        launcher(client, host, tmp_path / "scratch").launch(spec)


def test_a_role_with_gpus_its_container_would_not_see_is_refused(world, tmp_path):
    daemon, client, host = world
    with pytest.raises(ValueError, match="would not see"):
        launcher(client, host, tmp_path / "scratch").launch(
            container_spec(tmp_path, gpus=False, shape=shape(gpus=1, gpu_kind="H200")))


def test_a_container_the_daemon_will_not_create_fails_its_report(world, tmp_path):
    daemon, client, host = world
    # The role's own container, not the helper its user files are read through.
    daemon.fail("POST", r"/v1\.44/containers/create", 400, "invalid mount config",
                when=lambda body: USER_FILES_LABEL.encode() not in body)
    spec = container_spec(tmp_path)
    handle = launcher(client, host, tmp_path / "scratch").launch(spec)
    assert handle.wait(10) == NOT_STARTED_STATUS
    with pytest.raises(RoleFailed, match="did not create its container"):
        handle.report(0)
    handle.close_logs(10)
    assert (spec.log_dir / "serve.err").read_text().startswith("FATAL: docker launcher:")
    assert handle.ended_by_scheduler() is True


def test_the_launcher_never_runs_containers_as_root(world, tmp_path):
    with pytest.raises(ValueError, match="never run as root"):
        launcher(world[1], world[2], tmp_path / "scratch", uid=0)


# -- the stopper, the step's reads and the code directories ---------------------------------


def test_the_stopper_ends_and_removes_the_jobs_containers_and_no_others(world, tmp_path):
    daemon, client, host = world
    handle = launcher(client, host, tmp_path / "scratch").launch(container_spec(tmp_path))
    g = play_gate(daemon, host, handle, marker=MARKER, stay=True)
    g.join()
    other = daemon.add_image("other:1")
    del other
    from hls.substrate.docker.api import ContainerConfig
    foreign = client.create_container(ContainerConfig(image="other:1", argv=("x",), user="1:1",
                                                      labels={"hls.job": "999"}, network_mode="none", cpuset=(0,),
                                                      memory_bytes=1, pids_limit=1))
    # What the job's image roles left under its name goes too; another job's name stays.
    from hls.substrate.docker.images import job_leftovers, job_repository
    daemon.add_image(f"{job_repository(JOB)}:{'a' * 64}")
    daemon.add_image(f"{job_repository('999')}:{'a' * 64}")
    daemon.volumes[f"buildx_buildkit_hls-job-{JOB}-build0_state"] = set()
    host.ended(handle.container_id)
    DockerStopper(socket_path=daemon.path).stop(JOB)
    assert list(daemon.containers) == [foreign]
    assert job_leftovers(client, JOB) == [] and job_leftovers(client, "999") == [f"{job_repository('999')}:{'a' * 64}"]
    handle.wait(10)


def test_a_step_reads_its_id_and_its_cgroup(tmp_path):
    assert step.step_id({"HLS_STEP_ID": "12.3"}) == "12.3"
    for bad in ({}, {"HLS_STEP_ID": "12"}, {"HLS_STEP_ID": "a.b"}):
        with pytest.raises(RuntimeError):
            step.step_id(bad)
    (tmp_path / "memory.max").write_text("max\n")
    assert step.step_cgroup("/", tmp_path) == tmp_path
    with pytest.raises(RuntimeError, match="no memory.max"):
        step.step_cgroup("/x", tmp_path)
    with pytest.raises(RuntimeError, match="not an absolute"):
        step.step_cgroup("/../etc", tmp_path)


def test_code_dirs_hold_the_checkout_the_environment_and_its_interpreter(tmp_path):
    venv = tmp_path / "checkout" / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /opt/python/cpython-3.12/bin\nversion = 3.12\n")
    assert code_dirs(venv / "bin" / "python", tmp_path / "checkout") == (tmp_path / "checkout",
                                                                       Path("/opt/python/cpython-3.12"))
    with pytest.raises(ValueError, match="not the interpreter of a virtual environment"):
        code_dirs(Path("/usr/bin/python3"), tmp_path)


def test_only_a_harness_role_that_speaks_to_the_daemon_gets_its_socket(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch")
    pin = launch.launch(harness_spec(tmp_path, "pin", daemon=True))
    other = launch.launch(harness_spec(tmp_path, "other"))
    for h in (pin, other):
        play_gate(daemon, host, h, stay=True).join()
    sockets = {h.spec.name: [m for m in body(daemon, h)["HostConfig"]["Mounts"] if m["Target"] == str(daemon.path)]
               for h in (pin, other)}
    assert [m["ReadOnly"] for m in sockets["pin"]] == [False] and sockets["other"] == []
    for h in (pin, other):
        h.kill()
    with pytest.raises(ValueError, match="only a harness role that runs a command"):
        launch.launch(container_spec(tmp_path, "c", owner="harness", daemon=True))
    assert "GroupAdd" not in body(daemon, pin)["HostConfig"]
    try:
        # A socket only its group may write, as a daemon's is (root:docker 0660), is opened through
        # that group, given to the role's container alone.
        daemon.path.chmod(0o670)
        other_user = launcher(client, host, tmp_path / "scratch", uid=os.getuid() + 1, gid=os.getgid() + 1,
                              job_id="1700000001")
        grouped = other_user.launch(harness_spec(tmp_path, "pin3", daemon=True))
        play_gate(daemon, host, grouped, stay=True).join()
        assert body(daemon, grouped)["HostConfig"]["GroupAdd"] == [str(os.getgid())]
        grouped.kill()
        daemon.path.chmod(0o600)
        with pytest.raises(ValueError, match="cannot open the daemon's socket"):
            launcher(client, host, tmp_path / "scratch", uid=os.getuid() + 1).launch(
                harness_spec(tmp_path, "pin2", daemon=True))
    finally:
        daemon.path.chmod(0o755)


def test_the_roles_running_at_once_hold_no_more_memory_than_the_allocation(world, tmp_path):
    daemon, client, host = world
    launch = launcher(client, host, tmp_path / "scratch", memory_gib=3)
    first = launch.launch(container_spec(tmp_path, "a"))
    with pytest.raises(ValueError, match="hold 2 of its 3 GiB"):
        launch.launch(container_spec(tmp_path, "b"))
    first.kill()
    second = launch.launch(container_spec(tmp_path, "c"))
    second.kill()


def test_the_runtime_a_stage_reads_matches_what_the_launcher_does(tmp_path):
    from hls.substrate.docker.launcher import START_MARKER, WORKDIR_DIRS, DockerRuntime

    runtime = DockerRuntime(gateway_memory_gib=3)
    assert runtime.start_marker == START_MARKER == MARKER and runtime.attempt_line is None
    assert runtime.seal_memory_gib == 3
    assert runtime.workdir_dirs == WORKDIR_DIRS == ("tmp", "var_tmp", "home")
    spec = container_spec(tmp_path, mounts=(Mount(tmp_path, "/data"),)).container.spec
    runtime.check(spec)
    for i, mounts in enumerate(((Mount(tmp_path, "/tmp/x"),), (Mount(tmp_path, "/a/../b"),),
                                (Mount(tmp_path, "/d"),) * 2)):
        with pytest.raises(ValueError):
            runtime.check(container_spec(tmp_path, name=f"m{i}", mounts=mounts).container.spec)
    seal = SealPlan(policy=SealPolicy(network=NetworkPolicy()), index=False)
    runtime.check(container_spec(tmp_path, name="s", seal=seal).container.spec)


def _step_archive(path: Path) -> tuple[str, str, str]:
    """A one-image OCI layout archive with no name, as BuildKit writes the step image: its sha256,
    manifest digest and config digest."""
    import hashlib
    import io
    import tarfile

    def digest(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    config = json.dumps({"architecture": "amd64", "os": "linux", "rootfs": {"type": "layers", "diff_ids": []}})
    manifest = json.dumps({"schemaVersion": 2, "config": {"digest": digest(config.encode()), "size": len(config)},
                           "layers": []})
    index = json.dumps({"schemaVersion": 2, "manifests": [{"digest": digest(manifest.encode()),
                                                           "size": len(manifest)}]})
    with tarfile.open(path, "w") as tar:
        for name, text in (("index.json", index), (f"blobs/sha256/{digest(manifest.encode())[7:]}", manifest),
                           (f"blobs/sha256/{digest(config.encode())[7:]}", config)):
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    return hashlib.sha256(path.read_bytes()).hexdigest(), digest(manifest.encode()), digest(config.encode())


@pytest.mark.parametrize("store", ["containerd", "classic"])
def test_the_step_image_is_the_one_the_load_names_on_either_image_store(world, tmp_path, monkeypatch, store):
    daemon, client, host = world
    sha, manifest, config = _step_archive(tmp_path / "step.oci.tar")
    image_id = manifest if store == "containerd" else config
    plain = daemon.add_image

    def add_image(ref, labels=None):
        made = plain(ref, labels)
        daemon.images[image_id] = {**daemon.images.pop(made), "Id": image_id}
        return image_id

    monkeypatch.setattr(daemon, "add_image", add_image)
    got = launcher_module.load_step_image(client, tmp_path / "step.oci.tar", sha256=sha, scratch=tmp_path / "s")
    assert got == image_id
    assert daemon.images[image_id]["RepoTags"] == [f"hls/step:{manifest[7:39]}"]
    assert not (tmp_path / "s" / "step.oci.tar").exists()


def test_a_step_image_the_daemon_names_by_another_digest_or_whose_copy_differs_is_refused(world, tmp_path):
    daemon, client, host = world
    sha, manifest, config = _step_archive(tmp_path / "step.oci.tar")
    with pytest.raises(ValueError, match="neither its manifest"):
        launcher_module.load_step_image(client, tmp_path / "step.oci.tar", sha256=sha, scratch=tmp_path / "s")
    with pytest.raises(ValueError, match="has sha256"):
        launcher_module.load_step_image(client, tmp_path / "step.oci.tar", sha256="0" * 64, scratch=tmp_path / "s")
