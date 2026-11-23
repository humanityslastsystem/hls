"""The Engine API client against the Docker daemon of the machine the tests run on.

Opt-in, as every ``integration`` test (``HLS_RUN_SLOW=1``), and skipped where no Docker Engine
serving ``API_VERSION`` answers on the default socket. The tests make their one image from the
machine's static busybox and load it from an OCI archive, so they pull nothing; everything they
create carries ``hls.test=docker-api`` and a label of its own run, and is removed at the end,
also when a test fails."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import signal
import subprocess
import tarfile
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from hls.substrate.docker.api import (
    BindMount,
    ContainerConfig,
    DockerClient,
    DockerError,
    NotFound,
    PortBinding,
    RequestRefused,
    Ulimit,
)

pytestmark = pytest.mark.integration

LANE = "docker-api"
RUN = uuid.uuid4().hex[:12]
LABELS = {"hls.test": LANE, "hls.test.run": RUN}
MIB = 1 << 20


def _static_elf(path: Path) -> bool:
    """Whether *path* is an x86-64 ELF executable with no program interpreter (PT_INTERP)."""
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        return False
    phoff = int.from_bytes(data[32:40], "little")
    phentsize = int.from_bytes(data[54:56], "little")
    phnum = int.from_bytes(data[56:58], "little")
    types = [int.from_bytes(data[phoff + i * phentsize:phoff + i * phentsize + 4], "little") for i in range(phnum)]
    return 3 not in types


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


def _blob(blobs: dict[str, bytes], data: bytes) -> dict[str, object]:
    digest = "sha256:" + hashlib.sha256(data).hexdigest()
    blobs[digest] = data
    return {"digest": digest, "size": len(data)}


def _add(tar: tarfile.TarFile, name: str, data: bytes, mode: int = 0o644) -> None:
    info = tarfile.TarInfo(name)
    info.size, info.mode, info.mtime = len(data), mode, 0
    tar.addfile(info, io.BytesIO(data))


@dataclass(frozen=True)
class LiveImage:
    name: str
    manifest_digest: str
    config_digest: str
    archive: Path


def build_archive(busybox: Path, out: Path) -> LiveImage:
    """An OCI image layout in one tar: busybox as /bin/sh and /bin/busybox, labelled for this
    run, so its digests are this run's own."""
    layer_buf = io.BytesIO()
    with tarfile.open(fileobj=layer_buf, mode="w", format=tarfile.PAX_FORMAT) as layer:
        for d in ("bin", "etc", "tmp"):
            info = tarfile.TarInfo(d)
            info.type, info.mode = tarfile.DIRTYPE, 0o755
            layer.addfile(info)
        _add(layer, "bin/busybox", busybox.read_bytes(), 0o755)
        link = tarfile.TarInfo("bin/sh")
        link.type, link.linkname = tarfile.SYMTYPE, "busybox"
        layer.addfile(link)
        _add(layer, "etc/hls-test-run", RUN.encode())
    layer_bytes = layer_buf.getvalue()
    blobs: dict[str, bytes] = {}
    layer_desc = {"mediaType": "application/vnd.oci.image.layer.v1.tar", **_blob(blobs, layer_bytes)}
    config = {"architecture": "amd64", "os": "linux",
              "config": {"Env": ["PATH=/bin"], "Labels": dict(LABELS)},
              "rootfs": {"type": "layers", "diff_ids": [layer_desc["digest"]]}}
    config_desc = {"mediaType": "application/vnd.oci.image.config.v1+json",
                   **_blob(blobs, json.dumps(config).encode())}
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": config_desc, "layers": [layer_desc]}
    manifest_desc = {"mediaType": "application/vnd.oci.image.manifest.v1+json",
                     **_blob(blobs, json.dumps(manifest).encode())}
    name = f"docker.io/hls-test/{LANE}:{RUN}"
    index = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json",
             "manifests": [{**manifest_desc, "annotations": {"io.containerd.image.name": name,
                                                             "org.opencontainers.image.ref.name": RUN}}]}
    with tarfile.open(out, "w") as tar:
        _add(tar, "oci-layout", json.dumps({"imageLayoutVersion": "1.0.0"}).encode())
        _add(tar, "index.json", json.dumps(index).encode())
        for digest, data in blobs.items():
            _add(tar, f"blobs/sha256/{digest.removeprefix('sha256:')}", data)
    return LiveImage(name=f"hls-test/{LANE}:{RUN}", manifest_digest=str(manifest_desc["digest"]),
                     config_digest=str(config_desc["digest"]), archive=out)


@pytest.fixture(scope="module")
def image(docker, tmp_path_factory) -> Iterator[LiveImage]:
    found = shutil.which("busybox")
    if found is None or not _static_elf(Path(found).resolve()):
        pytest.skip("no static x86-64 busybox on this machine to make the test image from")
    built = build_archive(Path(found).resolve(), tmp_path_factory.mktemp("image") / "image.oci.tar")
    loaded = docker.load_image(built.archive, timeout=120)
    try:
        assert loaded == (f"docker.io/{built.name}",) or loaded == (built.name,), loaded
        yield built
    finally:
        info = docker.inspect_image(built.name)
        for tag in info.repo_tags:
            docker.remove_image(tag)
        try:
            docker.remove_image(info.id)
        except NotFound:
            pass
        with pytest.raises(NotFound):
            docker.inspect_image(info.id)


def _config(image: LiveImage, argv: tuple[str, ...], **over) -> ContainerConfig:
    cpu = sorted(os.sched_getaffinity(0))[0]
    base = dict(image=image.name, argv=argv, user=f"{os.getuid()}:{os.getgid()}", labels=dict(LABELS),
                network_mode="none", cpuset=(cpu,), memory_bytes=64 * MIB, pids_limit=64)
    base.update(over)
    return ContainerConfig(**base)


@contextmanager
def container(docker: DockerClient, cfg: ContainerConfig) -> Iterator[str]:
    cid = docker.create_container(cfg)
    try:
        yield cid
    finally:
        docker.remove_container(cid, force=True)


def test_the_daemon_is_a_docker_engine_with_an_identity(docker):
    version = docker.check()
    assert "Engine" in version.components
    info = docker.info()
    assert info.id and info.ncpu > 0
    first, second = docker.identity(), docker.identity()
    assert first == second and first.engine_id == info.id and first.start_ticks > 0
    assert Path(f"/proc/{first.pid}/comm").read_text().strip() == "dockerd"
    pidfile = Path("/run/docker.pid")
    if pidfile.is_file():
        assert int(pidfile.read_text()) == first.pid


def test_a_loaded_archive_is_named_labelled_and_identified_by_its_digests(docker, image):
    info = docker.inspect_image(image.name)
    assert info.labels == LABELS
    assert info.id in (image.manifest_digest, image.config_digest)
    if docker.info().containerd_store:
        assert info.id == image.manifest_digest
    docker.tag_image(info.id, repo=f"hls-test/{LANE}-tag", tag=RUN)
    try:
        assert docker.inspect_image(f"hls-test/{LANE}-tag:{RUN}").id == info.id
    finally:
        docker.remove_image(f"hls-test/{LANE}-tag:{RUN}")
    assert docker.inspect_image(image.name).id == info.id


def test_limits_user_and_mounts_hold_inside_the_container(docker, image, tmp_path):
    out = tmp_path / "out"
    ro = tmp_path / "ro"
    out.mkdir()
    ro.mkdir()
    (ro / "given").write_text("x")
    script = ("id -u > /out/uid; id -g > /out/gid; "
              "cat /sys/fs/cgroup/memory.max > /out/memory; cat /sys/fs/cgroup/memory.swap.max > /out/swap; "
              "cat /sys/fs/cgroup/pids.max > /out/pids; cat /sys/fs/cgroup/cpuset.cpus.effective > /out/cpus; "
              "df -k /dev/shm | tail -1 > /out/shm; ulimit -n > /out/nofile; "
              "touch /ro/new 2> /dev/null && echo yes > /out/ro_writable; "
              "touch /rootfile 2> /dev/null && echo yes > /out/root_writable; "
              "echo $HLS_TEST_VAR > /out/env; exit 7")
    cfg = _config(image, ("/bin/sh", "-c", script), shm_size_bytes=16 * MIB,
                  ulimits=(Ulimit("nofile", 256, 256),), env={"HLS_TEST_VAR": "plain"},
                  mounts=(BindMount(out, "/out", read_only=False), BindMount(ro, "/ro", read_only=True)))
    with container(docker, cfg) as cid:
        docker.start_container(cid)
        assert docker.wait_container(cid, timeout=60).status_code == 7
        state = docker.inspect_container(cid).state
        assert (state.status, state.exit_code, state.oom_killed) == ("exited", 7, False)
        assert state.started_at is not None and state.finished_at is not None
    read = {p.name: p.read_text().strip() for p in out.iterdir()}
    assert read["uid"] == str(os.getuid()) and read["gid"] == str(os.getgid())
    assert read["memory"] == str(64 * MIB) and read["swap"] == "0" and read["pids"] == "64"
    assert read["cpus"] == str(cfg.cpuset[0])
    assert read["shm"].split()[1] == str(16 * 1024)
    assert read["nofile"] == "256" and read["env"] == "plain"
    assert "ro_writable" not in read and "root_writable" not in read
    assert all(p.stat().st_uid == os.getuid() for p in out.iterdir())


def test_supplementary_groups_are_the_processs_and_no_others(docker, image, tmp_path):
    """A process the host user owns cannot test reading by group, since a file's owner bits
    decide for its owner; so the test reads the groups back, and the launcher's own test opens
    the socket."""
    extra = sorted(set(os.getgroups()) - {os.getgid(), 0})
    if not extra:
        pytest.skip("the test user has no supplementary group to give the container")
    out = tmp_path / "out"
    out.mkdir()
    mounts = (BindMount(out, "/out", read_only=False),)
    for group_add in ((extra[0],), ()):
        cfg = _config(image, ("/bin/sh", "-c", "id -G > /out/groups"), mounts=mounts, group_add=group_add)
        with container(docker, cfg) as cid:
            docker.start_container(cid)
            assert docker.wait_container(cid, timeout=60).status_code == 0
        groups = sorted(int(g) for g in (out / "groups").read_text().split())
        assert groups == sorted({os.getgid(), *group_add})


def test_kill_and_its_events(docker, image):
    since = time.time_ns()
    with docker.watch_events(since_ns=since, types=["container"]) as stream:
        with container(docker, _config(image, ("/bin/sh", "-c", "sleep 60"))) as cid:
            docker.start_container(cid)
            assert docker.wait_container(cid, timeout=0.5) is None
            docker.kill_container(cid, signal.SIGTERM)
            result = docker.wait_container(cid, timeout=30)
            assert result is not None and result.status_code == 143
            with pytest.raises(RequestRefused) as caught:
                docker.kill_container(cid, signal.SIGKILL)
            assert caught.value.status == 409
            seen = []
            deadline = time.monotonic() + 30
            while "die" not in seen and time.monotonic() < deadline:
                event = stream.get(timeout=1)
                if event is not None and event.actor_id == cid:
                    seen.append(event.action)
                    if event.action == "kill":
                        assert event.attributes["signal"] == "15"
                        assert event.attributes["hls.test.run"] == RUN
        assert seen[:2] == ["create", "start"] and "kill" in seen and seen[-1] == "die"
        window = docker.events(since_ns=since, until_ns=time.time_ns(), containers=[cid], actions=["kill"])
        assert [e.attributes["signal"] for e in window] == ["15"]
    with pytest.raises(NotFound):
        docker.inspect_container(cid)


def test_stop_and_remove_of_a_running_container(docker, image):
    with container(docker, _config(image, ("/bin/sh", "-c", "sleep 60"))) as cid:
        docker.start_container(cid)
        with pytest.raises(RequestRefused) as caught:
            docker.remove_container(cid)
        assert caught.value.status == 409
        assert docker.stop_container(cid, grace=1) is True
        assert docker.stop_container(cid, grace=1) is False
        assert not docker.inspect_container(cid).state.running


def _oom_kills(cgroup: Path) -> int:
    for line in (cgroup / "memory.events").read_text().splitlines():
        name, _, value = line.partition(" ")
        if name == "oom_kill":
            return int(value)
    raise AssertionError(f"no oom_kill in {cgroup}/memory.events")


def test_a_memory_hog_is_killed_and_oomkilled_agrees_with_the_oom_event(docker, image):
    """The kernel counts the kill in the parent cgroup's memory.events, which is hierarchical
    and outlives the container's. Docker sets ``State.OOMKilled`` and sends an ``oom`` event only
    when its runtime reads the container cgroup's count before the cgroup is removed, so a kill
    can go unflagged; the flag and the event agree."""
    cfg = _config(image, ("/bin/sh", "-c", "sleep 1; tail /dev/zero"), memory_bytes=32 * MIB)
    with docker.watch_events(since_ns=time.time_ns(), types=["container"]) as stream:
        with container(docker, cfg) as cid:
            docker.start_container(cid)
            pid = docker.inspect_container(cid).state.pid
            own = Path(f"/proc/{pid}/cgroup").read_text().strip().partition("::")[2]
            parent = Path("/sys/fs/cgroup" + own).parent
            before = _oom_kills(parent)
            result = docker.wait_container(cid, timeout=60)
            assert result is not None and result.status_code == 137
            assert _oom_kills(parent) > before
            actions: list[str] = []
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and not {"oom", "die"} <= set(actions):
                event = stream.get(timeout=0.2)
                if event is not None and event.actor_id == cid:
                    actions.append(event.action)
            assert "die" in actions
            assert docker.inspect_container(cid).state.oom_killed == ("oom" in actions)


def test_a_cgroup_parent_holds_the_container_and_counts_its_oom_kill(docker, image):
    """The parent is ``user.slice``, which every systemd host has, so the test makes no slice: a
    slice systemd makes for a parent the daemon names is not the test user's to remove."""
    if docker.info().cgroup_driver != "systemd":
        pytest.skip("the daemon's cgroup driver is not systemd")
    parent = Path("/sys/fs/cgroup/user.slice")
    cfg = _config(image, ("/bin/sh", "-c", "sleep 1; tail /dev/zero"), memory_bytes=32 * MIB,
                  cgroup_parent="user.slice")
    with container(docker, cfg) as cid:
        before = _oom_kills(parent)
        docker.start_container(cid)
        pid = docker.inspect_container(cid).state.pid
        assert Path(f"/proc/{pid}/cgroup").read_text().strip() == f"0::/user.slice/docker-{cid}.scope"
        result = docker.wait_container(cid, timeout=60)
        assert result is not None and result.status_code == 137
        assert _oom_kills(parent) == before + 1


def test_a_published_port_gets_a_host_port(docker, image):
    cfg = _config(image, ("/bin/sh", "-c", "sleep 30"), network_mode="bridge",
                  port_bindings=(PortBinding(8000, "127.0.0.1"),))
    with container(docker, cfg) as cid:
        docker.start_container(cid)
        ports = docker.inspect_container(cid).ports
        assert list(ports) == ["8000/tcp"]
        ((host_ip, host_port),) = ports["8000/tcp"]
        assert host_ip == "127.0.0.1" and host_port > 0


def test_a_bind_mount_of_a_missing_source_is_refused(docker, image, tmp_path):
    cfg = _config(image, ("/bin/sh", "-c", "true"),
                  mounts=(BindMount(tmp_path / "absent", "/out", read_only=False),))
    with pytest.raises(RequestRefused) as caught:
        docker.remove_container(docker.create_container(cfg), force=True)
    assert caught.value.status == 400 and "does not exist" in caught.value.message
    assert not (tmp_path / "absent").exists()


def test_list_containers_finds_this_runs_containers(docker, image):
    with container(docker, _config(image, ("/bin/sh", "-c", "true"))) as cid:
        assert [r.id for r in docker.list_containers(labels=LABELS)] == [cid]


def test_list_images_finds_this_runs_image_by_its_repository(docker, image):
    repository = f"hls-test/{LANE}-list"
    docker.tag_image(image.name, repo=repository, tag=RUN)
    try:
        rows = docker.list_images(repository=repository)
        assert [r.id for r in rows] == [docker.inspect_image(image.name).id]
        assert rows[0].repo_tags == (f"{repository}:{RUN}",)
        assert docker.list_images(repository=f"{repository}-none") == []
    finally:
        docker.remove_image(f"{repository}:{RUN}")


def test_list_containers_named_finds_a_container_by_the_start_of_its_name(docker, image):
    name = f"hls-test-{LANE}-{RUN}-named"
    cid = docker.create_container(_config(image, ("/bin/sh", "-c", "true")), name=name)
    try:
        assert [r.id for r in docker.list_containers_named(f"hls-test-{LANE}-{RUN}-")] == [cid]
        assert docker.list_containers_named(f"{LANE}-{RUN}-named") == []
    finally:
        docker.remove_container(cid, force=True)


def test_list_and_remove_a_volume(docker):
    name = f"hls-test-{LANE}-{RUN}-volume"
    made = subprocess.run(["docker", "volume", "create", "--label", f"hls.test={LANE}", name],
                          capture_output=True, text=True)
    if made.returncode != 0:
        pytest.skip(f"no docker command to make a volume with: {made.stderr.strip()}")
    try:
        assert docker.list_volumes(f"hls-test-{LANE}-{RUN}-") == [name]
        docker.remove_volume(name)
        assert docker.list_volumes(f"hls-test-{LANE}-{RUN}-") == []
        with pytest.raises(NotFound):
            docker.remove_volume(name)
    finally:
        subprocess.run(["docker", "volume", "rm", "--force", name], capture_output=True)
