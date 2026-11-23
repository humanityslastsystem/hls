"""The Engine API client against a fake daemon on a Unix socket (``fake_daemon``)."""

from __future__ import annotations

import io
import json
import os
import signal
import socket
import tarfile
import threading
import time
from pathlib import Path

import pytest

from hls.substrate.docker import api
from hls.substrate.docker.api import (
    BindMount,
    ContainerConfig,
    DaemonUnreachable,
    Device,
    DockerClient,
    DockerError,
    NotFound,
    PortBinding,
    RequestRefused,
    Ulimit,
    WrongDaemon,
)
from tests.substrate.docker.fake_daemon import FakeDocker

SECRET = "s3cret-value-never-shown"


@pytest.fixture
def fake(tmp_path):
    daemon = FakeDocker(tmp_path / "d.sock")
    with daemon.serving():
        yield daemon


@pytest.fixture
def client(fake):
    with DockerClient(fake.path, timeout=5.0) as c:
        yield c


def config(image: str = "busybox:test", **over) -> ContainerConfig:
    base = dict(image=image, argv=("/bin/sh", "-c", "sleep 1"), user="1000:1000",
                labels={"hls.job": "7", "hls.role": "serve"}, network_mode="host", cpuset=(3, 1, 2),
                memory_bytes=1 << 30, pids_limit=512)
    base.update(over)
    return ContainerConfig(**base)


def created(fake: FakeDocker, client: DockerClient, **over) -> str:
    fake.add_image("busybox:test")
    return client.create_container(config(**over))


# -- the daemon --------------------------------------------------------------------------------

def test_check_accepts_a_docker_engine_and_asks_version_without_an_api_version(fake, client):
    version = client.check()
    assert version.version == "29.5.2" and version.api_version == "1.54"
    assert "Engine" in version.components
    assert fake.requests[-1][1] == "/version"


@pytest.mark.parametrize("change, why", [
    ({"Components": [{"Name": "Podman Engine"}]}, "not served by a Docker Engine"),
    ({"ApiVersion": "1.43"}, "hls needs 1.44"),
    ({"MinAPIVersion": "1.45"}, "hls needs 1.44"),
    ({"Os": "windows"}, "not linux"),
    ({"ApiVersion": "one"}, "cannot read"),
])
def test_check_refuses_another_daemon(fake, client, change, why):
    fake.version.update(change)
    with pytest.raises(WrongDaemon, match=why):
        client.check()


def test_a_missing_socket_is_unreachable(tmp_path):
    with DockerClient(tmp_path / "absent.sock", timeout=2.0) as c:
        for call in (c.version, c.check, c.info, lambda: c.inspect_container("x"),
                     lambda: c.load_image(_archive(tmp_path, None), timeout=2.0)):
            with pytest.raises(DaemonUnreachable, match="cannot be reached"):
                call()


def test_a_socket_nobody_serves_is_unreachable(tmp_path):
    path = tmp_path / "stale.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(path))
    s.close()
    with DockerClient(path, timeout=2.0) as c, pytest.raises(DaemonUnreachable):
        c.info()


def test_a_daemon_that_does_not_answer_in_time_is_unreachable(fake):
    fake.hang("GET", "/v1.44/info", 3.0)
    with DockerClient(fake.path, timeout=0.5) as c, pytest.raises(DaemonUnreachable, match="within 0.5 s"):
        c.info()


def test_a_reply_that_is_not_json_or_not_in_form_is_a_docker_error(fake, client):
    fake.garble("GET", "/v1.44/info")
    with pytest.raises(DockerError, match="not JSON") as caught:
        client.info()
    assert type(caught.value) is DockerError
    fake.garble("GET", "/version", b'{"Version": "29"}')
    with pytest.raises(DockerError, match="not in the form expected"):
        client.version()


def test_proxy_variables_are_not_used(fake, client, monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    with DockerClient(fake.path, timeout=2.0) as c:
        assert c.info().name == "fakehost"


def test_info_reads_the_fields_hls_uses(client):
    info = client.info()
    assert info.id == "db28a0d1-0000-4000-8000-000000000000"
    assert info.containerd_store and info.cgroup_version == "2" and not info.live_restore
    assert info.runtimes == ("nvidia", "runc") and info.ncpu == 8 and info.warnings == ()


def _proc(root: Path, entries: dict[int, tuple[str, int, int]]) -> Path:
    """A fake /proc: pid -> (comm, ppid, start ticks)."""
    for pid, (comm, ppid, ticks) in entries.items():
        (root / str(pid)).mkdir(parents=True)
        fields = ["S", str(ppid)] + ["0"] * 17 + [str(ticks)] + ["0"] * 10
        (root / str(pid) / "stat").write_text(f"{pid} ({comm}) {' '.join(fields)}\n")
    (root / "self").mkdir()
    return root


def test_identity_is_the_listener_when_the_daemon_made_its_socket(fake, client, tmp_path):
    proc = _proc(tmp_path / "proc", {os.getpid(): ("dockerd", 1, 777)})
    ident = client.identity(proc=proc)
    assert ident == api.DaemonIdentity(engine_id=fake.info["ID"], pid=os.getpid(), start_ticks=777)


def test_identity_under_socket_activation_is_the_listeners_dockerd_child(client, tmp_path):
    me = os.getpid()
    proc = _proc(tmp_path / "proc", {me: ("systemd", 0, 1), 900: ("dockerd", me, 5150),
                                     901: ("dockerd", 1, 9), 902: ("containerd", me, 3),
                                     903: ("my) dockerd (x", me, 4)})
    assert client.identity(proc=proc).pid == 900
    assert client.identity(proc=proc).start_ticks == 5150


@pytest.mark.parametrize("children", [0, 2])
def test_identity_refuses_a_daemon_it_cannot_single_out(client, tmp_path, children):
    me = os.getpid()
    entries = {me: ("systemd", 0, 1)}
    entries.update({900 + i: ("dockerd", me, 10 + i) for i in range(children)})
    with pytest.raises(DockerError, match=f"{children} children named 'dockerd'"):
        client.identity(proc=_proc(tmp_path / "proc", entries))


def test_identity_changes_with_the_daemon_process(client, tmp_path):
    me = os.getpid()
    first = client.identity(proc=_proc(tmp_path / "a", {me: ("systemd", 0, 1), 900: ("dockerd", me, 50)}))
    again = client.identity(proc=_proc(tmp_path / "b", {me: ("systemd", 0, 1), 900: ("dockerd", me, 50)}))
    restarted = client.identity(proc=_proc(tmp_path / "c", {me: ("systemd", 0, 1), 900: ("dockerd", me, 99)}))
    assert first == again and first != restarted


def test_daemon_pid_reads_the_real_proc():
    comm = Path(f"/proc/{os.getpid()}/comm").read_text().strip()
    assert api.daemon_pid(os.getpid(), comm=comm) == os.getpid()
    assert api.start_ticks(os.getpid()) > 0


# -- containers --------------------------------------------------------------------------------

def test_create_sends_every_setting_and_the_fixed_hardening(fake, client, tmp_path):
    profile = json.dumps({"defaultAction": "SCMP_ACT_ERRNO", "syscalls": []})
    cfg = config(
        env={"HLS_ROLE": "serve", "TOKEN": SECRET}, working_dir="/work", swap_bytes=0,
        shm_size_bytes=1 << 30, ulimits=(Ulimit("nproc", 4096, 4096),),
        mounts=(BindMount(tmp_path, "/ckpt", read_only=True), BindMount(tmp_path, "/scratch", read_only=False)),
        cap_add=("NET_ADMIN",), seccomp_profile=profile, apparmor_profile="hls-sandbox",
        unmask_system_paths=True,
        devices=(Device(Path("/dev/nvidia3"), "/dev/nvidia3"), Device(Path("/dev/nvidiactl"), "/dev/nvidiactl")),
        oom_score_adj=1000, network_mode="bridge", port_bindings=(PortBinding(8000, "127.0.0.1"),))
    fake.add_image("busybox:test")
    cid = client.create_container(cfg, name="hls-7-serve")
    body = fake.create_bodies[-1]
    assert fake.requests[-1][2] == {"name": ["hls-7-serve"]}
    assert body["Entrypoint"] == ["/bin/sh", "-c", "sleep 1"] and body["Cmd"] == []
    assert body["User"] == "1000:1000" and body["WorkingDir"] == "/work"
    assert body["Env"] == ["HLS_ROLE=serve", f"TOKEN={SECRET}"]
    assert body["Labels"] == {"hls.job": "7", "hls.role": "serve"}
    assert body["ExposedPorts"] == {"8000/tcp": {}}
    host = body["HostConfig"]
    assert host["CapDrop"] == ["ALL"] and host["CapAdd"] == ["NET_ADMIN"] and host["Privileged"] is False
    assert host["Runtime"] == "runc"
    assert host["SecurityOpt"] == ["no-new-privileges", f"seccomp={profile}", "apparmor=hls-sandbox"]
    assert host["MaskedPaths"] == [] and host["ReadonlyPaths"] == []
    assert host["CgroupnsMode"] == "private"
    assert host["CpusetCpus"] == "1,2,3"
    assert host["Memory"] == host["MemorySwap"] == 1 << 30
    assert host["PidsLimit"] == 512 and host["ShmSize"] == 1 << 30 and host["OomScoreAdj"] == 1000
    assert host["Ulimits"] == [{"Name": "nproc", "Soft": 4096, "Hard": 4096}]
    # A GPU is its device nodes, which the container may read and write and not create; no device
    # request, which the NVIDIA runtime's hook would serve.
    assert host["Devices"] == [
        {"PathOnHost": "/dev/nvidia3", "PathInContainer": "/dev/nvidia3", "CgroupPermissions": "rw"},
        {"PathOnHost": "/dev/nvidiactl", "PathInContainer": "/dev/nvidiactl", "CgroupPermissions": "rw"}]
    assert "DeviceRequests" not in host
    ro, rw = host["Mounts"]
    assert ro == {"Type": "bind", "Source": str(tmp_path), "Target": "/ckpt", "ReadOnly": True,
                  "BindOptions": {"Propagation": "rprivate", "NonRecursive": False,
                                  "CreateMountpoint": False, "ReadOnlyForceRecursive": True}}
    assert rw["ReadOnly"] is False and "ReadOnlyForceRecursive" not in rw["BindOptions"]
    assert host["NetworkMode"] == "bridge"
    assert host["PortBindings"] == {"8000/tcp": [{"HostIp": "127.0.0.1", "HostPort": ""}]}
    assert host["LogConfig"] == {"Type": "none", "Config": {}}
    assert host["Init"] is True and host["ReadonlyRootfs"] is True
    assert host["AutoRemove"] is False and host["RestartPolicy"] == {"Name": "no"}
    assert cid in fake.containers


def test_defaults_keep_the_daemons_profiles_and_swap_is_added_to_memory(fake, client):
    created(fake, client, swap_bytes=1 << 20)
    host = fake.create_bodies[-1]["HostConfig"]
    assert host["SecurityOpt"] == ["no-new-privileges"]
    assert host["MemorySwap"] == (1 << 30) + (1 << 20)
    for key in ("MaskedPaths", "ReadonlyPaths", "DeviceRequests", "Devices", "ShmSize", "OomScoreAdj",
                "CgroupParent", "GroupAdd"):
        assert key not in host
    # The daemon's own runc, even where its default runtime is the NVIDIA Container Toolkit's.
    assert host["Runtime"] == "runc"
    assert all(path != "/v1.44/info" for _, path, _, _ in fake.requests)


def test_supplementary_groups_are_sent_as_numeric_strings(fake, client):
    created(fake, client, group_add=(998, 1001))
    assert fake.create_bodies[-1]["HostConfig"]["GroupAdd"] == ["998", "1001"]


@pytest.mark.parametrize("driver, parent", [("systemd", "hls-7-serve.slice"), ("cgroupfs", "/hls/7/serve")])
def test_a_cgroup_parent_in_the_drivers_form_is_sent(fake, client, driver, parent):
    fake.info["CgroupDriver"] = driver
    created(fake, client, cgroup_parent=parent)
    assert fake.create_bodies[-1]["HostConfig"]["CgroupParent"] == parent


@pytest.mark.parametrize("driver, parent", [("systemd", "/hls/7"), ("systemd", "hls"),
                                            ("cgroupfs", "hls-7.slice"), ("none", "hls-7.slice")])
def test_a_cgroup_parent_the_driver_does_not_read_is_refused_before_creating(fake, client, driver, parent):
    fake.info["CgroupDriver"] = driver
    fake.add_image("busybox:test")
    with pytest.raises(ValueError, match=f"cgroup driver is '{driver}'"):
        client.create_container(config(cgroup_parent=parent))
    assert fake.create_bodies == [] and fake.containers == {}


@pytest.mark.parametrize("over, why", [
    ({"user": "root"}, "uid:gid"),
    ({"user": ""}, "uid:gid"),
    ({"argv": ()}, "needs a command"),
    ({"cpuset": ()}, "cpuset"),
    ({"memory_bytes": 0}, "positive memory"),
    ({"pids_limit": 0}, "pids limit"),
    ({"cap_add": ("ALL",)}, "one capability"),
    ({"cap_add": ("CAP_NET_ADMIN",)}, "one capability"),
    ({"apparmor_profile": "unconfined"}, "loaded profile"),
    ({"apparmor_profile": "hls-sandbox", "unconfine_apparmor": True}, "or under none, not both"),
    ({"seccomp_profile": "unconfined"}, "not JSON"),
    ({"seccomp_profile": "[]"}, "not a JSON object"),
    ({"network_mode": "container:../x"}, "network mode"),
    ({"port_bindings": (PortBinding(8000, "127.0.0.1"),)}, "bridge"),
    ({"env": {"A-B": "x"}}, "cannot carry"),
    ({"env": {"A": "x\0y"}}, "cannot carry"),
    ({"image": "busybox;rm"}, "image reference"),
    ({"working_dir": "work"}, "absolute"),
    ({"cgroup_parent": "../x"}, "slice name or a cgroup path"),
    ({"cgroup_parent": "/a/../b"}, "slice name or a cgroup path"),
    ({"cgroup_parent": "/"}, "slice name or a cgroup path"),
    ({"cgroup_parent": "-.slice"}, "slice name or a cgroup path"),
    ({"cgroup_parent": "a--b.slice"}, "slice name or a cgroup path"),
    ({"cgroup_parent": "/hls/x.slice"}, "slice name or a cgroup path"),
    ({"cgroup_parent": "a b"}, "slice name or a cgroup path"),
    ({"group_add": (0,)}, "gids above 0"),
    ({"group_add": (998, 0)}, "gids above 0"),
    ({"group_add": (-1,)}, "gids above 0"),
    ({"group_add": ("998",)}, "gids above 0"),
    ({"group_add": (True,)}, "gids above 0"),
])
def test_config_refuses_what_hls_never_creates(over, why):
    with pytest.raises(ValueError, match=why):
        config(**over)


@pytest.mark.parametrize("source, target", [("dev/nvidia0", "/dev/nvidia0"), ("/dev/nvidia0", "/nvidia0")])
def test_a_device_is_a_host_path_given_under_dev(source, target):
    with pytest.raises(ValueError, match="under /dev"):
        Device(Path(source), target)


def test_a_container_without_apparmor_says_so_by_its_own_field(fake, client):
    """No profile name stands for no AppArmor: ``unconfine_apparmor`` alone sends
    ``apparmor=unconfined``, beside ``no-new-privileges``, which stays."""
    created(fake, client, unconfine_apparmor=True)
    assert fake.create_bodies[-1]["HostConfig"]["SecurityOpt"] == ["no-new-privileges", "apparmor=unconfined"]
    created(fake, client)
    assert fake.create_bodies[-1]["HostConfig"]["SecurityOpt"] == ["no-new-privileges"]


def test_a_bind_mount_needs_absolute_paths():
    with pytest.raises(ValueError, match="absolute"):
        BindMount(Path("rel"), "/x", read_only=True)


def test_no_environment_value_is_in_a_repr():
    cfg = config(env={"TOKEN": SECRET})
    assert SECRET not in repr(cfg)


def test_a_create_that_warns_is_removed_and_refused(fake, client):
    fake.add_image("busybox:test")
    fake.create_warnings = [f"Your kernel does not support swap limit capabilities ({SECRET})"]
    with pytest.raises(RequestRefused, match="warned") as caught:
        client.create_container(config(env={"TOKEN": SECRET}))
    assert caught.value.status == 201
    assert SECRET not in str(caught.value)
    assert fake.containers == {}


def test_a_refused_create_never_shows_an_environment_value(fake, client):
    fake.add_image("busybox:test")
    fake.fail("POST", "/v1.44/containers/create", 400, f"invalid environment: TOKEN={SECRET}")
    with pytest.raises(RequestRefused) as caught:
        client.create_container(config(env={"TOKEN": SECRET}))
    assert caught.value.status == 400
    assert SECRET not in str(caught.value) and SECRET not in caught.value.message
    assert "<redacted>" in caught.value.message


def test_create_from_a_missing_image_is_not_found(client):
    with pytest.raises(NotFound, match="No such image"):
        client.create_container(config(image="absent:1"))


def test_create_with_a_name_in_use_is_refused(fake, client):
    fake.add_image("busybox:test")
    client.create_container(config(), name="one")
    with pytest.raises(RequestRefused) as caught:
        client.create_container(config(), name="one")
    assert caught.value.status == 409


def test_start_inspect_and_ports(fake, client):
    fake.add_image("busybox:test")
    cid = client.create_container(config(network_mode="bridge", port_bindings=(PortBinding(8000, "127.0.0.1"),)))
    before = client.inspect_container(cid)
    assert before.state.status == "created" and before.state.started_at is None and before.ports == {}
    client.start_container(cid)
    info = client.inspect_container(cid)
    assert info.id == cid and info.labels == {"hls.job": "7", "hls.role": "serve"}
    assert info.state.running and info.state.pid == 4242 and info.state.exit_code == 0
    assert info.state.started_at is not None and info.state.finished_at is None
    assert info.ports == {"8000/tcp": (("127.0.0.1", 49153),)}
    with pytest.raises(RequestRefused) as caught:
        client.start_container(cid)
    assert caught.value.status == 304


@pytest.mark.parametrize("call", [
    lambda c: c.start_container("nope"),
    lambda c: c.inspect_container("nope"),
    lambda c: c.wait_container("nope", timeout=1),
    lambda c: c.kill_container("nope", signal.SIGTERM),
    lambda c: c.stop_container("nope", grace=1),
    lambda c: c.remove_container("nope"),
    lambda c: c.inspect_image("nope:1"),
    lambda c: c.tag_image("nope:1", repo="hls/x", tag="y"),
    lambda c: c.remove_image("nope:1"),
])
def test_a_missing_object_is_not_found(client, call):
    with pytest.raises(NotFound):
        call(client)


@pytest.mark.parametrize("call", [
    lambda c: c.inspect_container("../etc"),
    lambda c: c.inspect_image("a b"),
    lambda c: c.tag_image("img", repo="Upper/case", tag="x"),
    lambda c: c.tag_image("img", repo="hls/x", tag="bad tag"),
    lambda c: c.list_containers(labels={}),
])
def test_names_are_checked_before_any_request(fake, client, call):
    with pytest.raises(ValueError):
        call(client)
    assert fake.requests == []


def test_wait_returns_none_while_it_runs_and_the_status_once_it_ends(fake, client):
    cid = created(fake, client)
    client.start_container(cid)
    started = time.monotonic()
    assert client.wait_container(cid, timeout=0.3) is None
    assert time.monotonic() - started < 3
    threading.Timer(0.2, fake.exit, (cid, 3)).start()
    assert client.wait_container(cid, timeout=5) == api.WaitResult(status_code=3, error="")
    assert client.wait_container(cid, timeout=5).status_code == 3


def test_wait_on_a_created_container_returns_at_once(fake, client):
    cid = created(fake, client)
    assert client.wait_container(cid, timeout=5) == api.WaitResult(status_code=0, error="")


def test_kill_signals_and_a_stopped_container_is_refused(fake, client):
    cid = created(fake, client)
    client.start_container(cid)
    client.kill_container(cid, signal.SIGTERM)
    assert fake.requests[-1][2] == {"signal": ["SIGTERM"]}
    state = client.inspect_container(cid).state
    assert not state.running and state.exit_code == 143 and state.finished_at is not None
    with pytest.raises(RequestRefused) as caught:
        client.kill_container(cid, signal.SIGKILL)
    assert caught.value.status == 409


def test_stop_says_whether_it_stopped_anything(fake, client):
    cid = created(fake, client)
    client.start_container(cid)
    assert client.stop_container(cid, grace=2, sig=signal.SIGINT) is True
    assert fake.requests[-1][2] == {"t": ["2"], "signal": ["SIGINT"]}
    assert client.stop_container(cid, grace=2) is False


def test_remove_refuses_a_running_container_unless_forced(fake, client):
    cid = created(fake, client)
    client.start_container(cid)
    with pytest.raises(RequestRefused) as caught:
        client.remove_container(cid)
    assert caught.value.status == 409
    client.remove_container(cid, force=True)
    assert fake.requests[-1][2] == {"force": ["1"], "v": ["1"]}
    assert cid not in fake.containers


def test_list_containers_by_label(fake, client):
    fake.add_image("busybox:test")
    one = client.create_container(config(labels={"hls.test": "a", "hls.job": "1"}))
    client.create_container(config(labels={"hls.test": "b"}))
    rows = client.list_containers(labels={"hls.test": "a"})
    assert [r.id for r in rows] == [one]
    assert rows[0].labels == {"hls.test": "a", "hls.job": "1"} and rows[0].state == "created"
    assert json.loads(fake.requests[-1][2]["filters"][0]) == {"label": ["hls.test=a"]}
    assert fake.requests[-1][2]["all"] == ["1"]


def test_a_refusal_carries_the_status_and_the_daemons_message(fake, client):
    fake.fail("GET", "/v1.44/info", 500, "boom")
    with pytest.raises(RequestRefused) as caught:
        client.info()
    assert caught.value.status == 500 and caught.value.message == "boom"
    fake.fail("POST", "/v1.44/containers/x/start", 404, "gone")
    with pytest.raises(NotFound, match="gone"):
        client.start_container("x")


# -- events ------------------------------------------------------------------------------------

def test_events_in_a_window_filtered_by_container(fake, client):
    since = time.time_ns()
    one = created(fake, client)
    two = client.create_container(config())
    client.start_container(one)
    client.kill_container(one, signal.SIGTERM)
    until = time.time_ns()
    got = client.events(since_ns=since, until_ns=until, containers=[one])
    assert [e.action for e in got] == ["create", "start", "kill", "die"]
    assert all(e.actor_id == one and e.type == "container" for e in got)
    assert got[2].attributes["signal"] == "15" and got[2].attributes["hls.role"] == "serve"
    assert got[0].time_ns >= since
    query = fake.requests[-1][2]
    assert query["since"][0] == f"{since // 10**9}.{since % 10**9:09d}"
    assert json.loads(query["filters"][0]) == {"container": [one]}
    kills = client.events(since_ns=since, until_ns=until, actions=["kill"], types=["container"])
    assert [e.actor_id for e in kills] == [one]
    assert client.events(since_ns=until + 1, until_ns=until + 2, containers=[two]) == []


def test_watch_events_streams_until_closed(fake, client):
    stream = client.watch_events(since_ns=time.time_ns(), actions=["start", "die"])
    try:
        assert stream.get(timeout=0.2) is None
        cid = created(fake, client)
        client.start_container(cid)
        first = stream.get(timeout=5)
        assert first is not None and (first.action, first.actor_id) == ("start", cid)
        fake.exit(cid, 0)
        second = stream.get(timeout=5)
        assert second is not None and second.action == "die" and second.attributes["exitCode"] == "0"
    finally:
        started = time.monotonic()
        stream.close()
        assert time.monotonic() - started < 5
    stream.close()
    with pytest.raises(DockerError, match="was closed") as caught:
        stream.get(timeout=0.1)
    assert type(caught.value) is DockerError


def test_watch_events_raises_when_the_daemon_goes_away(tmp_path):
    fake = FakeDocker(tmp_path / "d.sock")
    with DockerClient(fake.path, timeout=5.0) as c:
        with fake.serving():
            stream = c.watch_events(since_ns=time.time_ns())
        with stream:
            with pytest.raises(DaemonUnreachable, match="events stream ended"):
                stream.get(timeout=5)
            with pytest.raises(DaemonUnreachable):
                stream.get(timeout=0.1)


def test_watch_events_reports_a_refusal(fake, client):
    fake.fail("GET", "/v1.44/events", 400, "bad filter")
    with pytest.raises(RequestRefused, match="bad filter"):
        client.watch_events(since_ns=0)


# -- images ------------------------------------------------------------------------------------

def _archive(root: Path, name: str | None) -> Path:
    """A tar holding an OCI layout's index.json, named by *name* when it is given."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"mediaType": "application/vnd.oci.image.manifest.v1+json",
                "digest": "sha256:" + "ab" * 32, "size": 1}
    if name:
        manifest["annotations"] = {"io.containerd.image.name": name}
    data = json.dumps({"schemaVersion": 2, "manifests": [manifest]}).encode()
    path = root / f"image-{name or 'anon'}.tar".replace("/", "_").replace(":", "_")
    with tarfile.open(path, "w") as tar:
        info = tarfile.TarInfo("index.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return path


def test_load_image_streams_the_archive_and_names_what_it_loaded(fake, client, tmp_path):
    archive = _archive(tmp_path, "hls/step:1")
    assert client.load_image(archive, timeout=10) == ("hls/step:1",)
    method, path, query, body = fake.requests[-1]
    assert (method, path, query) == ("POST", "/v1.44/images/load", {"quiet": ["1"]})
    assert body == archive.read_bytes()
    anonymous = client.load_image(_archive(tmp_path, None), timeout=10)
    assert len(anonymous) == 1 and anonymous[0].startswith("sha256:")


def test_load_image_error_in_the_stream_is_a_refusal(fake, client, tmp_path):
    fake.load_error = "unexpected EOF"
    with pytest.raises(RequestRefused, match="unexpected EOF"):
        client.load_image(_archive(tmp_path, "hls/step:1"), timeout=10)


def test_load_image_that_names_nothing_is_an_error(fake, client, tmp_path):
    fake.garble("POST", "/v1.44/images/load", b"")
    with pytest.raises(DockerError, match="named no image"):
        client.load_image(_archive(tmp_path, "hls/step:1"), timeout=10)


def test_load_image_refused_before_streaming(fake, client, tmp_path):
    fake.fail("POST", "/v1.44/images/load", 500, "no space")
    with pytest.raises(RequestRefused, match="no space"):
        client.load_image(_archive(tmp_path, "hls/step:1"), timeout=10)


def test_inspect_tag_and_remove_an_image(fake, client):
    image_id = fake.add_image("busybox:test", labels={"hls.test": "docker-api"})
    info = client.inspect_image("busybox:test")
    assert info.id == image_id and info.labels == {"hls.test": "docker-api"}
    assert info.repo_tags == ("busybox:test",) and info.os == "linux" and info.size == 1234
    client.tag_image(image_id, repo="hls/step", tag="0123abcd")
    assert fake.requests[-1][2] == {"repo": ["hls/step"], "tag": ["0123abcd"]}
    assert client.inspect_image("hls/step:0123abcd").id == image_id
    client.remove_image("hls/step:0123abcd")
    assert client.inspect_image("busybox:test").repo_tags == ("busybox:test",)
    client.remove_image("busybox:test")
    with pytest.raises(NotFound):
        client.inspect_image(image_id)


def test_list_images_by_repository_keeps_only_names_in_it(fake, client):
    one = fake.add_image("hls/job-7:" + "a" * 64)
    fake.images[one]["RepoTags"].append("hls/job-8:" + "a" * 64)
    fake.add_image("hls/job-70:" + "b" * 64)
    fake.add_image("busybox:test")
    rows = client.list_images(repository="hls/job-7")
    assert [r.id for r in rows] == [one]
    assert rows[0].repo_tags == ("hls/job-7:" + "a" * 64,)
    assert json.loads(fake.requests[-1][2]["filters"][0]) == {"reference": ["hls/job-7"]}
    with pytest.raises(ValueError):
        client.list_images(repository="HLS/Job")


def test_list_containers_named_keeps_only_a_prefix_of_a_name(fake, client):
    fake.add_image("busybox:test")
    one = client.create_container(config(), name="buildx_buildkit_hls-job-7-build-a0")
    client.create_container(config(), name="x-buildx_buildkit_hls-job-7-build-b0")
    rows = client.list_containers_named("buildx_buildkit_hls-job-7-")
    assert [r.id for r in rows] == [one] and rows[0].names == ("buildx_buildkit_hls-job-7-build-a0",)
    assert json.loads(fake.requests[-1][2]["filters"][0]) == {"name": ["buildx_buildkit_hls-job-7-"]}
    with pytest.raises(ValueError):
        client.list_containers_named("../x")


def test_list_and_remove_volumes(fake, client):
    fake.volumes = {"buildx_buildkit_hls-job-7-b0_state": set(), "other_buildx_buildkit_hls-job-7-": set()}
    assert client.list_volumes("buildx_buildkit_hls-job-7-") == ["buildx_buildkit_hls-job-7-b0_state"]
    client.remove_volume("buildx_buildkit_hls-job-7-b0_state")
    assert "buildx_buildkit_hls-job-7-b0_state" not in fake.volumes
    with pytest.raises(NotFound):
        client.remove_volume("buildx_buildkit_hls-job-7-b0_state")
    fake.add_image("busybox:test")
    cid = client.create_container(config())
    fake.volumes["in-use"] = {cid}
    with pytest.raises(RequestRefused) as caught:
        client.remove_volume("in-use")
    assert caught.value.status == 409
