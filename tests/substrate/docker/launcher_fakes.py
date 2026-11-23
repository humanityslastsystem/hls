"""A host for the Docker launcher's tests against the fake daemon (``fake_daemon``): each container
the fake starts gets a process id of its own, a ``/proc/<pid>`` and a cgroup under its parent,
in directories of the test, with the limits its create body asked for; and a gate the test runs
in a thread plays the container's part on the FIFOs."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from hls.substrate.docker.api import DaemonIdentity, DockerClient
from hls.substrate.docker.launcher import DockerLauncher, DockerRoleHandle
from hls.substrate.docker.nvidia import CONTROL_DEVICES, MODESET_DEVICE, DriverFile, NvidiaDriver
from tests.substrate.docker import fake_daemon
from tests.substrate.docker.fake_daemon import FakeDocker

JOB = "1700000000"
STEP_IMAGE = "hls/step:test"
ROLE_IMAGE = "hls/role:test"
GPUS = ("GPU-11111111-1111-1111-1111-111111111111", "GPU-22222222-2222-2222-2222-222222222222")
#: The user database of the images the tests run.
IMAGE_PASSWD = b"root:x:0:0:root:/root:/bin/bash\nubuntu:x:1000:1000:Ubuntu:/home/ubuntu:/bin/bash\n"
IMAGE_GROUP = b"root:x:0:\nubuntu:x:1000:\n"
DRIVER_VERSION = "570.211.01"


def parent_dir(cgroup_root: Path, parent: str) -> Path:
    """The directory of the systemd slice *parent*: each dash names a parent slice."""
    stem = parent.removesuffix(".slice").split("-")
    return cgroup_root.joinpath(*["-".join(stem[:i]) + ".slice" for i in range(1, len(stem) + 1)])


@dataclass
class FakeHost:
    """The host side of the fake's containers."""

    root: Path
    proc: Path
    cgroups: Path
    #: The host's /dev, which holds the driver's nodes and ten GPUs' nodes.
    dev: Path
    #: The driver's files as the scheduler records them, under the host's root.
    driver: NvidiaDriver
    #: The minor numbers of the GPU device nodes each new container's /dev shows, by UUID.
    minors: dict[str, int] = field(default_factory=lambda: {u: i for i, u in enumerate(GPUS)})
    #: Limits a test overrides in every new container's cgroup, by file name.
    override: dict[str, str] = field(default_factory=dict)
    #: The cgroup each started container is in, by container ID.
    cgroup_of: dict[str, Path] = field(default_factory=dict)
    pid_of: dict[str, int] = field(default_factory=dict)
    _next_pid: int = 4242
    identity: DaemonIdentity = field(default_factory=lambda: DaemonIdentity("engine", 99, 1))

    def started(self, cid: str, host_config: dict) -> int:
        self._next_pid += 1
        pid = self._next_pid
        # A container made with no parent, a seal's gateway, is under the daemon's default one.
        scope = parent_dir(self.cgroups, host_config.get("CgroupParent", "system.slice")) / f"docker-{cid}.scope"
        scope.mkdir(parents=True)
        memory = host_config["Memory"]
        files = {"memory.max": str(memory), "memory.swap.max": str(host_config["MemorySwap"] - memory),
                 "pids.max": str(host_config["PidsLimit"]), "memory.events": "oom 0\noom_kill 0\n",
                 "cgroup.procs": f"{pid}\n", **self.override}
        for name, text in files.items():
            (scope / name).write_text(text + "\n")
        for d in (scope.parent, scope.parent.parent):
            if not (d / "memory.events").exists():
                (d / "memory.events").write_text("oom 0\noom_kill 0\n")
        (self.proc / str(pid) / "root" / "dev").mkdir(parents=True)
        (self.proc / str(pid) / "cgroup").write_text(f"0::/{scope.relative_to(self.cgroups)}\n")
        for device in host_config.get("Devices") or ():
            (self.proc / str(pid) / "root" / device["PathInContainer"].lstrip("/")).touch()
        self.cgroup_of[cid] = scope
        self.pid_of[cid] = pid
        return pid

    def ended(self, cid: str, *, oom_kills: int = 0) -> None:
        """The kernel's view of a container that ended: its processes gone, its cgroup removed,
        and *oom_kills* counted in every parent above it."""
        scope = self.cgroup_of[cid]
        for d in (scope.parent, scope.parent.parent):
            events = dict(line.split() for line in (d / "memory.events").read_text().splitlines())
            events["oom_kill"] = str(int(events["oom_kill"]) + oom_kills)
            (d / "memory.events").write_text("".join(f"{k} {v}\n" for k, v in events.items()))
        for f in scope.iterdir():
            f.unlink()
        scope.rmdir()

    def affinity(self, pid: int) -> set[int]:
        for cid, p in self.pid_of.items():
            if p == pid:
                cpus = self.cpusets[cid]
                return {int(c) for c in cpus.split(",")}
        raise ProcessLookupError(pid)

    cpusets: dict[str, str] = field(default_factory=dict)

    def nvidia_smi(self, argv, **kwargs) -> subprocess.CompletedProcess:
        """``nvidia-smi --query-gpu=uuid,name,pci.bus_id``, each GPU on bus ``(i + 1) * 0x10``, as a
        driver prints it, and the driver's ``information`` file of each under the host's /proc."""
        lines = []
        for i, (uuid, minor) in enumerate(self.minors.items()):
            bus = (i + 1) * 0x10
            directory = self.proc / "driver" / "nvidia" / "gpus" / f"0000:{bus:02x}:00.0"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "information").write_text(f"Model: \t\t NVIDIA H200\nGPU UUID: \t {uuid}\n"
                                                    f"Device Minor: \t {minor}\n")
            lines.append(f"{uuid}, NVIDIA H200, 00000000:{bus:02X}:00.0")
        return subprocess.CompletedProcess(argv, 0, "\n".join(lines) + "\n", "")


@contextmanager
def fake_world(tmp_path: Path) -> Iterator[tuple[FakeDocker, DockerClient, FakeHost]]:
    """The fake daemon, a client of it and the fake host, with the images the tests run."""
    host = FakeHost(root=tmp_path, proc=tmp_path / "proc", cgroups=tmp_path / "cgroup", dev=tmp_path / "dev",
                    driver=fake_driver(tmp_path / "lib64", tmp_path / "bin"))
    host.proc.mkdir()
    host.cgroups.mkdir()
    host.dev.mkdir()
    for name in (*(f"nvidia{i}" for i in range(10)), *CONTROL_DEVICES, MODESET_DEVICE):
        (host.dev / name).touch()
    (host.proc / "driver" / "nvidia").mkdir(parents=True)
    (host.proc / "driver" / "nvidia" / "version").write_text(
        f"NVRM version: NVIDIA UNIX x86_64 Kernel Module  {DRIVER_VERSION}  Tue Nov 25 16:36:36 UTC 2025\n")
    daemon = FakeDocker(tmp_path / "d.sock")
    routes = fake_daemon._ROUTES
    (at,) = [i for i, (_, handler) in enumerate(routes) if handler is fake_daemon._Handler.post_start]
    pattern, original = routes[at]

    def post_start(handler, query, body, name):
        # The fake's own start, with a process id of the container's own and its host side made
        # before the daemon says it runs, as a real container's is.
        d = handler.daemon
        cid = d._find(name)
        if cid is None:
            return handler._error(404, f"No such container: {name}")
        c = d.containers[cid]
        if c["State"]["Running"]:
            return handler._reply(304)
        host.cpusets[cid] = c["HostConfig"]["CpusetCpus"]
        pid = host.started(cid, c["HostConfig"])
        with d._changed:
            c["State"].update(Status="running", Running=True, Pid=pid, StartedAt=fake_daemon._now())
            c["NetworkSettings"]["Ports"] = c["_ports"]
            d._event("container", "start", cid, {})
        handler._reply(204)

    routes[at] = (pattern, post_start)
    try:
        with daemon.serving():
            for ref in (STEP_IMAGE, ROLE_IMAGE):
                daemon.trees[daemon.add_image(ref)] = {"/etc/passwd": IMAGE_PASSWD, "/etc/group": IMAGE_GROUP}
            with DockerClient(daemon.path, timeout=10.0) as client:
                yield daemon, client, host
    finally:
        routes[at] = (pattern, original)


def fake_driver(lib: Path, bin_dir: Path) -> NvidiaDriver:
    """A driver of libcuda and NVML in *lib* and nvidia-smi in *bin_dir*, as ``read_driver``
    records one."""
    lib.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    for name in ("libcuda.so", "libnvidia-ml.so"):
        (lib / f"{name}.{DRIVER_VERSION}").touch()
    (bin_dir / "nvidia-smi").touch()
    return NvidiaDriver(version=DRIVER_VERSION, libraries=(
        DriverFile(path=lib / f"libcuda.so.{DRIVER_VERSION}",
                   names=("libcuda.so", "libcuda.so.1", f"libcuda.so.{DRIVER_VERSION}")),
        DriverFile(path=lib / f"libnvidia-ml.so.{DRIVER_VERSION}",
                   names=("libnvidia-ml.so.1", f"libnvidia-ml.so.{DRIVER_VERSION}"))),
        programs=(DriverFile(path=bin_dir / "nvidia-smi", names=("nvidia-smi",)),))


def launcher(client: DockerClient, host: FakeHost, scratch: Path, **over) -> DockerLauncher:
    data = host.root / "store"
    data.mkdir(exist_ok=True)
    scratch.mkdir(exist_ok=True)
    base = dict(client=client, job_id=JOB, python=sys.executable, scratch=scratch, step_image=STEP_IMAGE,
                allocation_gpu_uuids=GPUS, cpus=(2, 3, 4, 5, 6, 7), memory_gib=64, code_dirs=(Path(sys.prefix),),
                data_dirs=(data,), pids_limit=4096, labels={"hls.test": "docker-launcher"},
                batch_env={"PATH": "/usr/bin:/bin"}, node="fakehost", proc=host.proc, cgroup_root=host.cgroups,
                numa_root=host.root / "nodes", run=host.nvidia_smi, affinity=host.affinity,
                identity=lambda: host.identity, driver=host.driver, dev=host.dev)
    base.update(over)
    return DockerLauncher(**base)


def wait_for(predicate, timeout: float = 10.0, what: str = "a condition") -> None:
    give_up = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > give_up:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


@dataclass
class Gate:
    """The container's part, played in a thread: what the gate reads and what the role does."""

    env: bytes | None = None
    error: BaseException | None = None
    thread: threading.Thread | None = None

    def join(self) -> None:
        assert self.thread is not None
        self.thread.join(10)
        if self.error is not None:
            raise self.error


def play_gate(daemon: FakeDocker, host: FakeHost, handle: DockerRoleHandle, *, code: int = 0, out: bytes = b"",
              err: bytes = b"", marker: str | None = None, oom_kills: int = 0, oom_flag: bool = False,
              stay: bool = False) -> Gate:
    """Run the gate of *handle*'s container in a thread: read its variables as the gate does, write
    *marker*, *out* and *err*, and end the container with *code* (unless *stay*)."""
    gate = Gate()
    gate_dir = handle._gate_dir

    def run() -> None:
        try:
            wait_for(lambda: handle.container_id is not None and handle.container_id in host.pid_of,
                     what="the container to start")
            out_fd = os.open(gate_dir / "out", os.O_WRONLY)
            err_fd = os.open(gate_dir / "err", os.O_WRONLY)
            with open(gate_dir / "env", "rb") as f:
                gate.env = f.read()
            if marker is not None:
                os.write(err_fd, marker.encode() + b"\n")
            os.write(out_fd, out)
            os.write(err_fd, err)
            os.close(out_fd)
            os.close(err_fd)
            if not stay:
                cid = handle.container_id
                assert cid is not None
                host.ended(cid, oom_kills=oom_kills)
                daemon.exit(cid, code, oom=oom_flag)
        except BaseException as e:  # noqa: BLE001 -- handed to the test by join()
            gate.error = e

    gate.thread = threading.Thread(target=run, daemon=True)
    gate.thread.start()
    return gate
