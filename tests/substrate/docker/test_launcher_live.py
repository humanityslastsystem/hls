"""The Docker launcher against the Docker daemon of the machine the tests run on.

Opt-in, as every ``integration`` test (``HLS_RUN_SLOW=1``), and skipped where no Docker Engine
serving the client's API version answers on the default socket. Container roles run an image the
tests make from the machine's static busybox and load from an OCI archive, so they pull nothing.
Harness roles run the step image, which the tests take from ``HLS_DOCKER_STEP_IMAGE`` (an image
the daemon holds) and are skipped without it. Everything the tests create carries
``hls.test=docker-launcher`` and is removed at the end, also when a test fails; the parent
cgroups the daemon makes for the launcher's slots stay, as they do in production."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import shutil
import tarfile
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

import hls
from hls.spec.resources import ResourceShape
from hls.substrate.base import JobState, RoleSpec
from hls.substrate.cgroup import pids
from hls.substrate.container import ContainerRole, ContainerSpec, Mount, PinnedImage
from hls.substrate.docker.api import ContainerConfig, DockerClient, DockerError, NotFound
from hls.substrate.docker.launcher import DockerLauncher, DockerRoleHandle, code_dirs
from hls.substrate.exits import GRANT_MISMATCH_EXIT
from hls.substrate.secrets import write_env_file
from tests.substrate.docker.test_api_live import _static_elf

pytestmark = pytest.mark.integration

LANE = "docker-launcher"
#: The BuildKit image (release 0.35.1, by its index digest) the no-op instance test builds the step image with.
BUILDKIT_IMAGE = "moby/buildkit@sha256:baf6138eb8c37644800c7c913fba051e840ebb9ba13eaf92d92ef9a1712d0b4c"
LABELS = {"hls.test": LANE}
MARKER = "hls-container-started"
SECRET = "launcher-live-" + uuid.uuid4().hex
SHA = "b" * 64
APPLETS = ("sh", "sleep", "cat", "tail", "head", "touch", "echo", "env", "yes", "dd", "id", "ls", "wget", "httpd")


@pytest.fixture(scope="module")
def docker() -> Iterator[DockerClient]:
    if os.environ.get("HLS_RUN_SLOW") != "1":
        pytest.skip("live Docker tests are opt-in: set HLS_RUN_SLOW=1")
    client = DockerClient()
    try:
        client.check()
    except DockerError as e:
        client.close()
        pytest.skip(f"no Docker Engine to test against: {e}")
    try:
        yield client
    finally:
        leftover = client.list_containers(labels=LABELS)
        for row in leftover:
            client.remove_container(row.id, force=True)
        client.close()
        assert not leftover, f"containers left by a test: {[r.id for r in leftover]}"


def _add(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    tar.addfile(info, io.BytesIO(data))


def _blob(blobs: dict[str, bytes], data: bytes) -> dict[str, object]:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    blobs[digest] = data
    return {"digest": digest, "size": len(data)}


def build_archive(busybox: Path, out: Path, name: str) -> None:
    """An OCI image layout in one tar, which either image store loads, named *name*: busybox with
    a link for each of ``APPLETS``, a user database of root alone, labelled with this lane's
    label, and an environment that asks for every GPU, which a container must not get."""
    layer_buf = io.BytesIO()
    with tarfile.open(fileobj=layer_buf, mode="w", format=tarfile.PAX_FORMAT) as layer:
        for d in ("bin", "etc", "tmp"):
            info = tarfile.TarInfo(d)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            layer.addfile(info)
        _add(layer, "bin/busybox", busybox.read_bytes(), 0o755)
        # A user database, which every image a role runs holds and the launcher appends to.
        _add(layer, "etc/passwd", b"root:x:0:0:root:/root:/bin/sh\n", 0o644)
        _add(layer, "etc/group", b"root:x:0:\n", 0o644)
        for applet in APPLETS:
            link = tarfile.TarInfo(f"bin/{applet}")
            link.type, link.linkname = tarfile.SYMTYPE, "busybox"
            layer.addfile(link)
    layer_bytes = layer_buf.getvalue()
    blobs: dict[str, bytes] = {}
    layer_desc = {"mediaType": "application/vnd.oci.image.layer.v1.tar", **_blob(blobs, layer_bytes)}
    config = {"architecture": "amd64", "os": "linux",
              "config": {"Env": ["PATH=/bin", "NVIDIA_VISIBLE_DEVICES=all"], "Labels": dict(LABELS)},
              "rootfs": {"type": "layers", "diff_ids": [layer_desc["digest"]]}}
    config_desc = {"mediaType": "application/vnd.oci.image.config.v1+json",
                   **_blob(blobs, json.dumps(config).encode())}
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": config_desc, "layers": [layer_desc]}
    manifest_desc = {"mediaType": "application/vnd.oci.image.manifest.v1+json",
                     **_blob(blobs, json.dumps(manifest).encode())}
    index = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
             "manifests": [{**manifest_desc, "annotations": {"io.containerd.image.name": f"docker.io/{name}"}}]}
    # The classic image store reads manifest.json, and names the image by its RepoTags; the
    # containerd store reads index.json.
    docker_manifest = [{"Config": f"blobs/sha256/{config_desc['digest'].removeprefix('sha256:')}",
                        "RepoTags": [name],
                        "Layers": [f"blobs/sha256/{layer_desc['digest'].removeprefix('sha256:')}"]}]
    with tarfile.open(out, "w") as tar:
        _add(tar, "oci-layout", json.dumps({"imageLayoutVersion": "1.0.0"}).encode())
        _add(tar, "index.json", json.dumps(index).encode())
        _add(tar, "manifest.json", json.dumps(docker_manifest).encode())
        for digest, data in blobs.items():
            _add(tar, f"blobs/sha256/{digest.removeprefix('sha256:')}", data)


@pytest.fixture(scope="module")
def image(docker, tmp_path_factory) -> Iterator[str]:
    found = shutil.which("busybox")
    if found is None or not _static_elf(Path(found).resolve()):
        pytest.skip("no static x86-64 busybox on this machine to make the test image from")
    name = f"hls-test/{LANE}:{uuid.uuid4().hex[:12]}"
    archive = tmp_path_factory.mktemp("image") / "image.oci.tar"
    build_archive(Path(found).resolve(), archive, name)
    docker.load_image(archive, timeout=120)
    info = docker.inspect_image(name)
    try:
        yield info.id
    finally:
        docker.remove_image(name)
        try:
            docker.remove_image(info.id)
        except NotFound:
            pass


def host_nproc_max() -> int:
    """A ``site.nproc_max`` this host holds for a step of one CPU: the floor of every evaluation of
    that size the host could run at once, and no less than the default, as a site that sets none
    gets (``node_nproc_max``)."""
    from hls.substrate.nproc import node_nproc_max

    return node_nproc_max(None, os.cpu_count() or 1)


def _job() -> str:
    return str(time.time_ns() // 1000)


@pytest.fixture
def make(docker, tmp_path) -> Iterator:
    made: list[DockerLauncher] = []

    def make(**over) -> DockerLauncher:
        scratch, store = tmp_path / "scratch", tmp_path / "store"
        scratch.mkdir(exist_ok=True)
        store.mkdir(exist_ok=True)
        checkout = Path(hls.__file__).resolve().parents[2]
        base = dict(client=docker, job_id=_job(), python=sys.executable, scratch=scratch,
                    step_image=os.environ.get("HLS_DOCKER_STEP_IMAGE", "unused"), allocation_gpu_uuids=(),
                    cpus=sorted(os.sched_getaffinity(0)), memory_gib=16,
                    code_dirs=code_dirs(Path(sys.executable), checkout),
                    data_dirs=(store,), pids_limit=4096, labels=LABELS)
        base.update(over)
        launcher = DockerLauncher(**base)
        made.append(launcher)
        return launcher

    yield make
    for launcher in made:
        launcher.close()


def role(tmp_path: Path, image: str, command: tuple[str, ...], *, name: str = "serve", memory_gib: int = 1,
         disk_gib: int = 0, env: dict[str, str] | None = None, mounts=(), cpus: int = 1, time_sec: int = 120,
         owner: str = "contestant") -> RoleSpec:
    role_dir = tmp_path / "scratch" / "roles" / name
    role_dir.mkdir(parents=True, exist_ok=True)
    env_file = role_dir / "container.env"
    env_file.unlink(missing_ok=True)
    write_env_file(env_file, env or {})
    workdir = role_dir / "work"
    workdir.mkdir(exist_ok=True)
    spec = ContainerSpec(image=PinnedImage(ref=image, file_sha256=SHA, check_sha256=SHA), command=command,
                         workdir=workdir, mounts=tuple(mounts))
    return RoleSpec(name=name, argv=(), shape=ResourceShape(cpus=cpus, memory_gib=memory_gib, disk_gib=disk_gib,
                                                            time_sec=time_sec),
                    owner=owner, log_dir=tmp_path / "store" / "logs", start_marker=MARKER,
                    container=ContainerRole(spec=spec, env_file=env_file, seal_files=None))


def _inspect_text(container_id: str) -> str:
    return subprocess.run(["docker", "inspect", container_id], capture_output=True, text=True, check=True).stdout


def _cmdlines() -> list[bytes]:
    out = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                out.append((entry / "cmdline").read_bytes())
            except OSError:
                continue
    return out


def _payload_pid(handle: DockerRoleHandle, comm: str) -> int:
    assert handle.cgroup_dir is not None
    for pid in pids(handle.cgroup_dir):
        try:
            if Path(f"/proc/{pid}/comm").read_text().strip() == comm:
                return pid
        except OSError:
            continue
    raise AssertionError(f"no process {comm} in {handle.cgroup_dir}")


def _wait_running(handle: DockerRoleHandle, comm: str, timeout: float = 30.0) -> int:
    give_up = time.monotonic() + timeout
    while True:
        try:
            return _payload_pid(handle, comm)
        except AssertionError:
            if time.monotonic() > give_up:
                raise
            time.sleep(0.05)


def test_the_grant_check_reads_back_every_limit_and_the_container_holds_its_settings(make, image, tmp_path):
    launcher = make()
    spec = role(tmp_path, image, ("sleep", "30"), cpus=2, memory_gib=1)
    handle = launcher.launch(spec)
    try:
        report = handle.report(60)
        assert report.mismatches == []
        assert report.granted.cpus == 2 and len(report.cpus) == 2 and report.granted.memory_gib == 1
        assert report.swap_max_bytes == 0 and report.granted.gpus == 0
        cgroup = Path(report.cgroup_dir)
        assert cgroup.parent == handle.parent and (cgroup / "memory.max").read_text().strip() == str(2**30)
        assert (cgroup / "pids.max").read_text().strip() == "4096"
        _wait_running(handle, "sleep")
        status = Path(f"/proc/{_payload_pid(handle, 'sleep')}/status").read_text()
        fields = dict(line.split(":", 1) for line in status.splitlines())
        assert fields["Uid"].split()[0] == str(os.getuid()) and fields["CapEff"].strip() == "0000000000000000"
        assert fields["NoNewPrivs"].strip() == "1"
        text = _inspect_text(handle.container_id)
        assert '"ReadonlyRootfs": true' in text and '"CapDrop": [\n                "ALL"' in text
    finally:
        handle.kill()
    assert handle.poll() is not None


def test_a_grant_that_differs_is_refused_and_nothing_of_the_role_runs(make, image, tmp_path):
    out = tmp_path / "store" / "out"
    out.mkdir(parents=True)
    spec = role(tmp_path, image, ("touch", "/out/ran"), disk_gib=10**6, mounts=(Mount(out, "/out", False),))
    handle = make().launch(spec)
    assert handle.wait(60) == GRANT_MISMATCH_EXIT
    assert any(m.startswith("disk_gib") for m in handle.report(0).mismatches)
    assert not (out / "ran").exists()
    handle.close_logs(10)
    assert "the container's grant differs" in (spec.log_dir / "serve.err").read_text()


def test_a_secret_reaches_the_payload_and_neither_docker_inspect_nor_any_argv(make, image, tmp_path):
    out = tmp_path / "store" / "out"
    out.mkdir(parents=True)
    spec = role(tmp_path, image, ("sh", "-c", 'echo "$API_TOKEN" > /out/seen; exec sleep 30'),
                env={"API_TOKEN": SECRET}, mounts=(Mount(out, "/out", False),))
    handle = make().launch(spec)
    try:
        handle.report(60)
        _wait_running(handle, "sleep")
        assert (out / "seen").read_text() == SECRET + "\n"
        assert SECRET not in _inspect_text(handle.container_id)
        assert not [c for c in _cmdlines() if SECRET.encode() in c]
        assert not spec.container.env_file.exists()
    finally:
        handle.kill()


def test_files_the_role_writes_are_the_host_users(make, image, tmp_path):
    out = tmp_path / "store" / "out"
    out.mkdir(parents=True)
    spec = role(tmp_path, image, ("sh", "-c", "touch /out/f /tmp/t \"$HOME/h\""), mounts=(Mount(out, "/out", False),))
    handle = make().launch(spec)
    assert handle.wait(60) == 0
    for path in (out / "f", spec.container.spec.workdir / "tmp" / "t", spec.container.spec.workdir / "home" / "h"):
        assert path.stat().st_uid == os.getuid() and path.stat().st_gid == os.getgid()


def test_logs_are_capped(make, image, tmp_path):
    spec = role(tmp_path, image, ("sh", "-c", "yes 0123456789 | head -c 200000; echo bye >&2"))
    handle = make(log_keep_bytes=1024).launch(spec)
    assert handle.wait(60) == 0
    handle.close_logs(10)
    assert handle.logs_lost() is None
    out = (spec.log_dir / "serve.out").read_bytes()
    assert b"bytes of this log were dropped here" in out and len(out) < 3000
    assert (spec.log_dir / "serve.err").read_text() == f"{MARKER}\nbye\n"


def test_thirty_kills_at_the_memory_limit_are_each_out_of_memory_and_an_outside_kill_is_not(make, image, tmp_path):
    launcher = make()
    seen = []
    for batch in range(6):
        handles = [launcher.launch(role(tmp_path, image, ("tail", "/dev/zero"), name=f"hog{batch}-{i}"))
                   for i in range(5)]
        for h in handles:
            assert h.wait(120) == 137, h.spec.name
            seen.append((h.spec.name, h.ended_by_scheduler(), h.end_state()))
    assert seen == [(name, False, JobState.OUT_OF_MEMORY) for name, _, _ in seen] and len(seen) == 30
    control = launcher.launch(role(tmp_path, image, ("sleep", "60"), name="control"))
    control.report(60)
    os.kill(_wait_running(control, "sleep"), signal.SIGKILL)
    assert control.wait(60) == 137
    assert control.ended_by_scheduler() is False and control.end_state() is JobState.FAILED
    assert launcher.allocation_oom_kills() == 30
    assert launcher.busy_slots == []


def test_two_roles_killed_at_their_limit_at_once_count_only_their_own(make, image, tmp_path):
    launcher = make()
    hog = launcher.launch(role(tmp_path, image, ("tail", "/dev/zero"), name="hog"))
    other = launcher.launch(role(tmp_path, image, ("sh", "-c", "sleep 1; exit 3"), name="other"))
    assert hog.wait(120) == 137 and other.wait(120) == 3
    assert hog.end_state() is JobState.OUT_OF_MEMORY and other.end_state() is JobState.FAILED


def test_a_docker_kill_from_outside_is_the_schedulers(make, image, tmp_path, docker):
    handle = make().launch(role(tmp_path, image, ("sleep", "60")))
    handle.report(60)
    _wait_running(handle, "sleep")
    subprocess.run(["docker", "kill", handle.container_id], check=True, capture_output=True)
    assert handle.wait(60) == 137
    assert handle.ended_by_scheduler() is True and handle.end_state() is JobState.CANCELLED


def test_terminate_and_kill_are_the_runners(make, image, tmp_path):
    launcher = make()
    gentle = launcher.launch(role(tmp_path, image, ("sleep", "60"), name="gentle"))
    hard = launcher.launch(role(tmp_path, image, ("sleep", "60"), name="hard"))
    for h in (gentle, hard):
        h.report(60)
        _wait_running(h, "sleep")
    launcher.terminate([gentle])
    hard.kill()
    assert gentle.wait(60) == 128 + signal.SIGTERM and hard.poll() == 137
    for h in (gentle, hard):
        assert h.stopped_by_runner and h.ended_by_scheduler() is False and h.end_state() is None


def test_nothing_is_left_behind(make, image, tmp_path, docker):
    launcher = make()
    handle = launcher.launch(role(tmp_path, image, ("true",)))
    assert handle.wait(60) == 0
    handle.close_logs(10)
    assert docker.list_containers(labels={"hls.job": launcher.job_id}) == []
    assert not handle._gate_dir.exists()
    with pytest.raises(NotFound):
        docker.inspect_container(handle.container_id)


def test_a_harness_role_runs_the_step_wrapper_in_the_step_image(make, tmp_path):
    if not os.environ.get("HLS_DOCKER_STEP_IMAGE"):
        pytest.skip("set HLS_DOCKER_STEP_IMAGE to a step image the daemon holds to run harness roles")
    try:
        from hls.substrate.factory import STEPS
    except ImportError:
        pytest.skip("this checkout has no substrate factory to route the step wrapper's reads")
    if "docker" not in STEPS:
        pytest.skip("the substrate factory does not route a Docker step")
    out = tmp_path / "store" / "out.json"
    spec = RoleSpec(name="noop", argv=(sys.executable, "-m", "hls.substrate.noop_role", "--out", str(out)),
                    shape=ResourceShape(cpus=2, memory_gib=1, time_sec=120), owner="harness",
                    log_dir=tmp_path / "store" / "logs")
    handle = make().launch(spec)
    report = handle.report(120)
    assert report.mismatches == [] and report.step == f"{handle.step}"
    assert Path(report.cgroup_dir) == handle.cgroup_dir and Path(report.cgroup_dir).parent == handle.parent
    assert handle.wait(120) == 0, (spec.log_dir / "noop.err").read_text()
    assert out.stat().st_uid == os.getuid()


def test_a_noop_stage_runs_on_docker_through_the_orchestrator_the_runner_and_the_stage(docker, tmp_path):
    """The orchestrator plans and submits, an in-process scheduler runs the batch script's runner,
    and the runner's launcher is the Docker launcher: the stage's role runs as a harness container
    in the step image and the instance succeeds."""
    step_image = os.environ.get("HLS_DOCKER_STEP_IMAGE")
    if not step_image:
        pytest.skip("set HLS_DOCKER_STEP_IMAGE to a step image the daemon holds to run harness roles")
    from hls.pipeline.pipelines import noop_one
    from hls.spec.resources import GrantedResources
    from hls.substrate.factory import substrate
    from hls.substrate.grant import Measured
    from hls.substrate.runner import JobEnd, Runner, read_latest_plan
    from tests.pipeline.plans import VERSIONS, noop_plan
    from tests.pipeline.test_end_to_end_local import LocalSlurm, _create, _drive

    cpus = sorted(os.sched_getaffinity(0))[:2]
    launchers: list[DockerLauncher] = []

    class DockerSlurm(LocalSlurm):
        def _run(self, job):
            words = job.request.script.read_text().splitlines()[-1].split()
            plan_dir, k, n = Path(words[4]), int(words[6]), int(words[8])
            plan = read_latest_plan(plan_dir)
            req = plan.allocation(k).request
            granted = GrantedResources(gpus=0, cpus=req.cpus, memory_gib=float(req.memory_gib), disk_free_gib=100.0)

            def make_launcher(gpus, gpu_uuids, scratch):
                checkout = Path(hls.__file__).resolve().parents[2]
                launcher = DockerLauncher(client=docker, job_id=job.job_id, python=str(plan.toolchain.python),
                                          scratch=scratch, step_image=step_image, allocation_gpu_uuids=(),
                                          cpus=cpus[:req.cpus], memory_gib=req.memory_gib,
                                          code_dirs=code_dirs(plan.toolchain.python, checkout),
                                          data_dirs=(plan.settings.site.store_root,), pids_limit=4096, labels=LABELS)
                launchers.append(launcher)
                return launcher

            runner = Runner(plan, k, n, job_id=job.job_id, node="local",
                            read_job_end=lambda: JobEnd(controller=time.time() + req.time_sec, env=None),
                            measure_grant=lambda: Measured(granted=granted, gpu_uuids=[], swap_max_bytes=0),
                            measure_tools=lambda: VERSIONS, make_launcher=make_launcher,
                            node_name=substrate(plan.settings.site).node_name,
                            runtime=substrate(plan.settings.site).runtime(),
                            images=substrate(plan.settings.site).images, checkout_state=lambda path: plan.code)
            return runner.run()

    slurm = DockerSlurm()
    decls = noop_one(gpus=0, gpu_kind="H200", sleep_sec=1, stage_time_sec=600)
    try:
        plan = _drive(_create(noop_plan(tmp_path, decls=decls, site={"nproc_max": host_nproc_max()}), slurm), slurm)
    finally:
        for launcher in launchers:
            launcher.close()
    assert plan.outcome.status == "succeeded", plan.outcome
    assert launchers and docker.list_containers(labels={"hls.job": launchers[0].job_id}) == []


def test_a_sealed_container_reaches_the_network_only_through_its_gateway(make, image, tmp_path, docker):
    """A sealed container runs in its gateway's network namespace: a connection to a domain the
    policy denies is refused by the interceptor, which records it, and the payload's ending is
    recorded with the interceptor alive. The gateway is gone afterwards."""
    step_image = os.environ.get("HLS_DOCKER_STEP_IMAGE")
    if not step_image:
        pytest.skip("set HLS_DOCKER_STEP_IMAGE to a step image the daemon holds to run a seal's gateway")
    from hls.egress.policy import NetworkPolicy, SealPolicy
    from hls.substrate.container import SealPlan
    from hls.substrate.seal import PayloadEnding, payload_ending, policy_refusals, seal_files

    policy = SealPolicy(network=NetworkPolicy.model_validate(
        {"domains": {"mode": "deny", "entries": ["example.com", "*.example.com"]}}))
    base = role(tmp_path, image, ("sh", "-c", "wget -q -T 20 -O /dev/null http://example.com/; echo wget=$? >&2"))
    state = tmp_path / "scratch" / "roles" / "serve" / "seal"
    state.mkdir(mode=0o700)
    files = seal_files(state)
    spec = dataclasses.replace(base, container=dataclasses.replace(
        base.container, spec=dataclasses.replace(base.container.spec, seal=SealPlan(policy=policy, index=False)),
        seal_files=files))
    launcher = make(step_image=step_image)
    handle = launcher.launch(spec)
    assert handle.wait(180) == 0, (spec.log_dir / "serve.err").read_text()
    handle.close_logs(10)
    err = (spec.log_dir / "serve.err").read_text()
    assert err.startswith(MARKER) and "wget=0" not in err, err
    assert payload_ending(files) == PayloadEnding(exit_code=0, interceptor_alive=True)
    assert policy_refusals(files.status) == ("example.com",)
    assert docker.list_containers(labels={"hls.job": launcher.job_id}) == []


def _read_all(conn: socket.socket) -> bytes:
    """Everything the peer sends until it closes: an HTTP/1.0 answer may come in several segments."""
    return b"".join(iter(lambda: conn.recv(4096), b""))


def _other_address() -> str | None:
    """An IPv4 address of this host other than the loopback, the one a route out of it uses."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1: nothing is sent, only a route is chosen
        address = probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()
    return None if address.startswith("127.") else address


def test_a_sealed_server_is_reached_on_the_loopback_and_from_no_other_address(make, image, tmp_path, docker):
    """A sealed server's port is published on the host's loopback alone: the drivers on the host
    reach it there, and a connection to the host's other address on that port is refused."""
    step_image = os.environ.get("HLS_DOCKER_STEP_IMAGE")
    if not step_image:
        pytest.skip("set HLS_DOCKER_STEP_IMAGE to a step image the daemon holds to run a seal's gateway")
    other = _other_address()
    if other is None:
        pytest.skip("this host has no address but the loopback to try the published port on")
    from hls.egress.policy import NetworkPolicy, SealPolicy
    from hls.substrate.container import SealPlan
    from hls.substrate.seal import ingress_port, seal_files

    base = role(tmp_path, image, ("sh", "-c", "echo served > /tmp/index.html; exec httpd -f -p 8000 -h /tmp"))
    state = tmp_path / "scratch" / "roles" / "serve" / "seal"
    state.mkdir(mode=0o700)
    files = seal_files(state)
    seal = SealPlan(policy=SealPolicy(network=NetworkPolicy()), index=False, ingress_port=8000)
    spec = dataclasses.replace(base, container=dataclasses.replace(
        base.container, spec=dataclasses.replace(base.container.spec, seal=seal), seal_files=files))
    handle = make(step_image=step_image).launch(spec)
    try:
        handle.report(120)
        give_up, body, port = time.monotonic() + 60, b"", None
        while time.monotonic() < give_up and b"served" not in body:
            port = ingress_port(files)
            try:
                if port is None:
                    raise OSError("no port yet")
                with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
                    conn.sendall(b"GET /index.html HTTP/1.0\r\n\r\n")
                    body = _read_all(conn)
            except OSError:
                time.sleep(0.5)
        assert port is not None and b"served" in body, body
        with pytest.raises(OSError):
            socket.create_connection((other, port), timeout=5).close()
        # Another container on Docker's default bridge cannot reach the server through the
        # gateway's own address there, while the host's publish still answers.
        [gateway] = docker.list_containers(labels={"hls.step": handle.step, "hls.seal": "gateway"})
        address = subprocess.run(["docker", "inspect", "-f", "{{.NetworkSettings.Networks.bridge.IPAddress}}",
                                  gateway.id], capture_output=True, text=True, check=True).stdout.strip()
        assert address
        peer = docker.create_container(ContainerConfig(
            image=image, argv=("wget", "-q", "-T", "5", "-O", "-", f"http://{address}:8000/index.html"),
            user=f"{os.getuid()}:{os.getgid()}", labels=LABELS, network_mode="bridge",
            cpuset=(sorted(os.sched_getaffinity(0))[0],), memory_bytes=64 * 2**20, pids_limit=64))
        try:
            docker.start_container(peer)
            ended = docker.wait_container(peer, timeout=60)
            assert ended is not None and ended.status_code != 0, ended
        finally:
            docker.remove_container(peer, force=True)
        with socket.create_connection(("127.0.0.1", port), timeout=5) as conn:
            conn.sendall(b"GET /index.html HTTP/1.0\r\n\r\n")
            assert b"served" in _read_all(conn)
    finally:
        handle.kill()
    assert docker.list_containers(labels={"hls.test": LANE}) == []


def test_a_noop_instance_runs_on_docker_from_a_site_file(docker, tmp_path):
    """A no-op instance of a site file whose ``site.substrate`` is docker: the local scheduler places
    the allocation and its supervisor runs the batch script, whose runner builds the step image into
    the fresh store's cache from its recipe, loads it and starts the stage's role as a container; the
    instance succeeds, the job ends COMPLETED with its reservation released, and no container of it
    is left. The build pulls the BuildKit image (``HLS_TEST_BUILDKIT_IMAGE``) and the recipe's
    inputs, so it needs the network."""
    buildkit = os.environ.get("HLS_TEST_BUILDKIT_IMAGE", BUILDKIT_IMAGE)
    secrets = tmp_path / "secrets"
    secrets.mkdir(mode=0o700)
    (secrets / "secrets.toml").write_text("[secrets]\n\n[grants]\n")
    (secrets / "secrets.toml").chmod(0o600)
    (tmp_path / "state").mkdir(mode=0o700)
    (tmp_path / "scratch").mkdir()
    site = tmp_path / "hls.toml"
    site.write_text(f"""[site]
substrate = "docker"
store_root = "{tmp_path / 'store'}"
node_scratch = "{tmp_path / 'scratch'}"
secrets_path = "{secrets / 'secrets.toml'}"
nproc_max = {host_nproc_max()}

[site.docker]
buildkit_image = "{buildkit}"
owner_label = "hls.test={LANE}"

[site.local]
state_dir = "{tmp_path / 'state'}"
kept_memory_gib = 4
""")
    from hls.pipeline import commands
    from hls.substrate.docker.launcher import STEP_IMAGE_REPO, archive_digests
    from hls.substrate.docker.step_image import step_key

    lines: list[str] = []
    try:
        plan = commands.plan_noop(config=site, overrides={}, two_stage=False, gpus=0, gpu_kind="H200", sleep_sec=1,
                                  stage_time_sec=300, fail_attempts=0, job_name="hls-noop")
        code = commands.start(plan, {}, echo=lines.append)
    finally:
        # The runner tags the step image it built and loaded; the tag goes, and the image with it.
        built = list((tmp_path / "store").rglob(f"{step_key()}/image.oci.tar"))
        for archive in built:
            try:
                docker.remove_image(f"{STEP_IMAGE_REPO}:{archive_digests(archive)[0].removeprefix('sha256:')[:32]}")
            except NotFound:
                pass
    assert code == 0 and "succeeded" in lines[-1], lines
    [record] = (tmp_path / "state" / "ended").glob("*.json")
    job = json.loads(record.read_text())
    assert (job["state"], job["exit_code"], job["stopper"], job["released"]) == ("COMPLETED", "0:0", "docker", True)
    [report] = (tmp_path / "store").rglob("cpu.report.json")
    role = json.loads(report.read_text())
    assert role["mismatches"] == [] and role["step"] == f"{job['job_id']}.0"
    assert role["cgroup_dir"].startswith("/sys/fs/cgroup/") and "/hls-c" in role["cgroup_dir"]
    assert docker.list_containers(labels={"hls.job": job["job_id"]}) == []
