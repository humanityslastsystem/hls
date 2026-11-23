"""The submission image on a Docker host: the install container (``image_role.submit``) built,
committed, exported and removed against a fake daemon of each image store, the tree read
(``image_role.read``), the client calls they make, and what the install may never be: a build on
the privileged builder, a role of a contestant as root."""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import tarfile
from pathlib import Path

import pytest

from hls.package import offline
from hls.substrate.base import RoleSpec
from hls.substrate.container import TREE_ABSENT_EXIT, ContainerSpec, PinnedImage
from hls.substrate.docker import image_role
from hls.substrate.docker.api import DockerClient, NotFound
from hls.substrate.docker.image_role import (INSTALL_CAPABILITIES, LOG_MOUNT, TREE_MOUNT, WHEELS_MOUNT,
                                             archive_image, image_config, install_script, with_config)
from tests.substrate.docker.archives import write_archive
from hls.substrate.docker.images import ROLE_MODULE, DockerImages, job_repository
from hls.stages import package as package_stage
from tests.substrate.docker.fake_daemon import FakeDocker

NONCE = "0123456789abcdef0123456789abcdef"
BASE_ID = "sha256:" + "b" * 64
BASE_CONFIG = {"Env": ["PATH=/opt/venv/bin:/usr/bin:/bin", "VIRTUAL_ENV=/opt/venv"], "Entrypoint": None,
               "Cmd": None, "Labels": {"hls.cuda_ref": "cuda:1@sha256:" + "c" * 64}, "WorkingDir": ""}


@pytest.fixture
def fake(tmp_path):
    daemon = FakeDocker(tmp_path / "d.sock")
    daemon.images[BASE_ID] = {"Id": BASE_ID, "RepoTags": ["hls/job-7:" + "a" * 64], "RepoDigests": [],
                              "Config": BASE_CONFIG, "Architecture": "amd64", "Os": "linux", "Size": 1}
    with daemon.serving():
        yield daemon


def _mount(container: dict, target: str) -> Path:
    return Path(next(m["Source"] for m in container["HostConfig"]["Mounts"] if m["Target"] == target))


def _installs(fake: FakeDocker, *, code: int, out: str, err: str = "") -> None:
    """Each started container writes *out* and *err* to its log FIFOs and exits *code*, as the
    install script does."""

    def run(cid: str, container: dict) -> None:
        log = _mount(container, LOG_MOUNT)
        (log / "out").write_text(out)
        (log / "err").write_text(err)
        fake.exit(cid, code)

    fake.on_start = run


def _submit(tmp_path: Path, fake: FakeDocker, monkeypatch, **over) -> tuple[int, Path]:
    monkeypatch.setattr(image_role, "own_memory", lambda: 8 << 30)
    monkeypatch.setattr(image_role, "own_pids", lambda: 512)
    work = tmp_path / "build"
    work.mkdir(exist_ok=True)
    (tmp_path / "tree").mkdir(exist_ok=True)
    (tmp_path / "wheels").mkdir(exist_ok=True)
    args = {"--socket": str(fake.path), "--base": BASE_ID, "--tree": str(tmp_path / "tree"),
            "--wheels": str(tmp_path / "wheels"), "--project": "subm", "--server": "server.py", "--nonce": NONCE,
            "--job": "7", "--name": "hls/job-7:submission-1", "--out": str(work / "submission.sif"),
            "--digest": str(work / "sha256"), **over}
    return image_role.main(["submit", *[w for pair in args.items() for w in pair]]), work


@pytest.mark.parametrize(("store", "digest"), [("containerd", "manifest_digest"), ("classic", "config_digest")])
def test_the_install_is_committed_with_the_bases_config_exported_and_removed(tmp_path, fake, monkeypatch, capsys,
                                                                             store, digest):
    fake.image_store = store
    status_line = f"hls-offline-install-rc-{NONCE}=0\n"
    _installs(fake, code=0, out=status_line, err="pip said something\n")
    rc, work = _submit(tmp_path, fake, monkeypatch)
    assert rc == 0, capsys.readouterr().err
    printed = capsys.readouterr()
    assert status_line in printed.out and "pip said something" in printed.err
    image = archive_image(work / "submission.sif")
    sha256 = hashlib.sha256((work / "submission.sif").read_bytes()).hexdigest()
    assert (work / "sha256").read_text() == sha256 + "\n"
    [(cid, query, config)] = fake.commits
    assert config == image_config(BASE_CONFIG) == {"Env": BASE_CONFIG["Env"], "Labels": BASE_CONFIG["Labels"]}
    with tarfile.open(work / "submission.sif") as tar:
        stored = json.load(tar.extractfile(f"blobs/sha256/{image.config_digest[7:]}"))
    assert stored["config"] == image_config(BASE_CONFIG)
    assert (query["repo"], query["tag"]) == (["hls/job-7"], ["submission-1"])
    # The container and the committed image are gone; the base stays.
    assert cid not in fake.containers and set(fake.images) == {BASE_ID}
    [create] = fake.create_bodies
    host = create["HostConfig"]
    assert create["User"] == "0:0" and create["Image"] == BASE_ID
    assert host["CapDrop"] == ["ALL"] and host["CapAdd"] == ["DAC_READ_SEARCH"] == list(INSTALL_CAPABILITIES)
    assert host["NetworkMode"] == "none" and host["ReadonlyRootfs"] is False and not host.get("Privileged")
    assert not host.get("Init") and all(m["Target"].startswith("/dev/") for m in host["Mounts"])
    assert "no-new-privileges" in host["SecurityOpt"] and not host.get("Devices")
    assert {m["Target"] for m in host["Mounts"]} == {TREE_MOUNT, WHEELS_MOUNT, LOG_MOUNT}
    assert all(m["ReadOnly"] for m in host["Mounts"])
    assert all("docker.sock" not in m["Source"] for m in host["Mounts"])
    assert (host["Memory"], host["PidsLimit"]) == (8 << 30, 512)
    assert create["Labels"]["hls.job"] == "7"
    assert create["Entrypoint"][:2] == ["/bin/sh", "-c"]
    assert create["Entrypoint"][2] == install_script(project="subm", server="server.py", nonce=NONCE,
                                                     constraints=None)


def test_a_failed_install_commits_nothing_and_leaves_nothing(tmp_path, fake, monkeypatch, capsys):
    _installs(fake, code=1, out=f"hls-offline-install-rc-{NONCE}=1\n")
    rc, work = _submit(tmp_path, fake, monkeypatch)
    assert rc == image_role.FAILED_EXIT
    assert f"hls-offline-install-rc-{NONCE}=1" in capsys.readouterr().out
    assert fake.commits == [] and fake.containers == {} and not (work / "submission.sif").exists()
    assert not (work / "sha256").exists()


def test_the_install_needs_the_roles_own_limits(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.setattr(image_role, "own_memory", lambda: None)
    monkeypatch.setattr(image_role, "own_pids", lambda: 512)
    args = ["submit", "--socket", str(fake.path), "--base", BASE_ID, "--tree", str(tmp_path), "--wheels",
            str(tmp_path), "--project", "subm", "--server", "server.py", "--nonce", NONCE, "--job", "7", "--name",
            "hls/job-7:s", "--out", str(tmp_path / "o"), "--digest", str(tmp_path / "d")]
    assert image_role.main(args) == image_role.FAILED_EXIT and fake.create_bodies == []


def test_the_script_stages_as_the_defs_files_runs_its_post_and_writes_the_runscript():
    post, run = offline.offline_install(project="subm", server="server.py", nonce=NONCE, constraints=None)
    lines = install_script(project="subm", server="server.py", nonce=NONCE, constraints=None).split("\n")
    assert lines[0] == f"exec >{LOG_MOUNT}/out 2>{LOG_MOUNT}/err"
    # apptainer's %files copies with cp -fLr (internal/pkg/build/files/copy.go).
    assert lines[1:3] == [f"cp -fLr {TREE_MOUNT} {offline.STAGE} || exit 1", f"cp -fLr {WHEELS_MOUNT} /wheels || exit 1"]
    assert lines[3:3 + len(post)] == post
    assert lines[3 + len(post):] == ["mkdir -p /.hls || exit 1",
                                     f"printf '%s\\n' '#!/bin/sh' '{run}' > /.hls/runscript || exit 1",
                                     "chmod 0755 /.hls/runscript || exit 1", ""]
    with pytest.raises(ValueError):
        install_script(project="x; rm -rf /", server="server.py", nonce=NONCE, constraints=None)


def _read(tmp_path: Path, fake: FakeDocker, path: str) -> tuple[int, Path]:
    dest = tmp_path / "tree"
    dest.mkdir()
    return image_role.main(["read", "--socket", str(fake.path), "--image", BASE_ID, "--path", path, "--dest",
                            str(dest), "--job", "7"]), dest


def test_a_tree_read_puts_the_images_directory_into_dest_and_leaves_no_container(tmp_path, fake):
    fake.trees[BASE_ID] = {"/workspace": {"a.txt": b"A", "sub/b.txt": b"B"}}
    rc, dest = _read(tmp_path, fake, "/workspace")
    assert rc == 0 and (dest / "a.txt").read_bytes() == b"A" and (dest / "sub" / "b.txt").read_bytes() == b"B"
    assert fake.containers == {} and not list(tmp_path.glob(".tree.tar"))
    assert fake.create_bodies[0]["User"] != "0:0"


def test_a_tree_read_takes_members_named_from_the_root(tmp_path, fake):
    """One image store names an archive's members from the root, ``/workspace/a.txt``."""
    fake.trees[BASE_ID] = {"/workspace": {"a.txt": b"A", "sub/b.txt": b"B"}}
    fake.archive_from_root = True
    rc, dest = _read(tmp_path, fake, "/workspace")
    assert rc == 0 and (dest / "a.txt").read_bytes() == b"A" and (dest / "sub" / "b.txt").read_bytes() == b"B"


def test_a_tree_read_never_copies_the_images_root(tmp_path, fake, capsys):
    fake.trees[BASE_ID] = {"/": {"etc/passwd": b"root"}}
    rc, dest = _read(tmp_path, fake, "/")
    assert rc == image_role.FAILED_EXIT and list(dest.iterdir()) == [] and fake.create_bodies == []
    assert "never its root" in capsys.readouterr().err


def test_a_tree_read_refuses_a_path_the_image_does_not_hold(tmp_path, fake, capsys):
    rc, dest = _read(tmp_path, fake, "/workspace")
    assert rc == TREE_ABSENT_EXIT and list(dest.iterdir()) == []
    assert f"refused: image {BASE_ID} holds no directory at /workspace" in capsys.readouterr().err
    assert fake.containers == {}


def test_the_client_commits_saves_and_reads_archives(tmp_path, fake):
    fake.add_image("busybox:test")
    with DockerClient(fake.path) as client:
        from hls.substrate.docker.api import ContainerConfig
        cid = client.create_container(ContainerConfig(image="busybox:test", argv=("true",), user="1000:1000",
                                                      labels={}, network_mode="none", cpuset=(0,),
                                                      memory_bytes=1 << 20, pids_limit=8))
        image_id = client.commit_container(cid, repo="hls/job-7", tag="t", config={"Env": ["A=1"]})
        assert fake.commits[0][2] == {"Env": ["A=1"]} and fake.images[image_id]["RepoTags"] == ["hls/job-7:t"]
        client.save_image(image_id, tmp_path / "saved.tar", timeout=10)
        assert archive_image(tmp_path / "saved.tar").manifest_digest == image_id
        with pytest.raises(FileExistsError):
            client.save_image(image_id, tmp_path / "saved.tar", timeout=10)
        with pytest.raises(NotFound):
            client.container_archive(cid, "/nowhere", tmp_path / "a.tar", timeout=10)
        assert not (tmp_path / "a.tar").exists() and not (tmp_path / "a.tar.partial").exists()
        assert client.inspect_image("busybox:test").config == {"Labels": {}}


def test_the_docker_runtime_builds_a_submission_by_its_install_program_never_the_builder(tmp_path):
    images = DockerImages(tmp_path, job_id="7", python="/py", buildkit_image="sha256:" + "d" * 64)
    base = PinnedImage(ref=BASE_ID, file_sha256="a" * 64, check_sha256="b" * 64)
    argv, env = images.submission_role(tmp_path / "build" / "submission.sif", digest=tmp_path / "build" / "sha256",
                                       base_file=(tmp_path / "base.tar", "a" * 64), base=base, tree=tmp_path / "t",
                                       wheels=tmp_path / "w", project="subm", server="server.py", nonce=NONCE,
                                       constraints="/opt/hls/constraints.txt")
    assert argv[:4] == ["/py", "-m", ROLE_MODULE, "submit"] and env == {}
    assert argv[argv.index("--base") + 1] == BASE_ID
    assert argv[argv.index("--name") + 1].startswith(job_repository("7") + ":submission-")
    assert "build" not in argv and "--buildkit-image" not in argv
    assert images.tree_path(base, "/workspace") is None
    read, _ = images.tree_role(base, "/workspace", tmp_path / "dest", own=tmp_path)
    assert read[:4] == ["/py", "-m", ROLE_MODULE, "read"] and read[read.index("--path") + 1] == "/workspace"
    for path in ("workspace", "/"):
        with pytest.raises(ValueError):
            images.tree_role(base, path, tmp_path / "dest", own=tmp_path)


def test_the_package_checks_read_an_image_from_its_archive_by_its_pinned_manifest(tmp_path):
    """No image is copied whole out of the daemon: the scan and the integrity check get the archive
    each image was pinned from and the manifest digest it was pinned by."""
    images = DockerImages(tmp_path, job_id="7", python="/py", buildkit_image="sha256:" + "d" * 64)
    base = PinnedImage(ref=BASE_ID, file_sha256="a" * 64, check_sha256="b" * 64)
    assert images.reader_args("base", base, tmp_path / "base.tar") == [
        "--base-archive", str(tmp_path / "base.tar"), "--base-manifest", "b" * 64]
    assert images.reader_args("image", base, tmp_path / "s.sif")[::2] == ["--image-archive", "--image-manifest"]
    # The integrity check leaves the base's tree for the scan, which reads it instead of the layers.
    assert images.reader_args("base", base, tmp_path / "base.tar", listing_out=tmp_path / "l.json")[4:] == [
        "--base-listing-out", str(tmp_path / "l.json")]
    assert images.reader_args("base", base, tmp_path / "base.tar", listing=tmp_path / "l.json")[4:] == [
        "--base-listing", str(tmp_path / "l.json")]
    from hls.substrate.apptainer.images import ApptainerImages
    apptainer = ApptainerImages(tmp_path / "node")
    for listing in ({}, {"listing": tmp_path / "l.json"}, {"listing_out": tmp_path / "l.json"}):
        assert apptainer.reader_args("base", base, tmp_path / "base.sif", **listing) == ["--base", BASE_ID]


def test_the_package_stage_reaches_no_builder_and_renders_no_dockerfile():
    """The submission is built through the image runtime's submission role alone: the stage names
    neither the image build role, which a Docker host runs on its privileged builder, nor a
    Dockerfile of the install."""
    source = inspect.getsource(package_stage)
    assert "build_role" not in source and "render_dockerfile" not in source
    assert not hasattr(offline, "render_offline_dockerfile")
    with pytest.raises(ValueError, match="privileged"):
        DockerImages(Path("/s"), job_id="7", python="/py", buildkit_image="sha256:" + "d" * 64).build_role(
            Path("/s/b/Dockerfile"), Path("/s/b/image.oci.tar"), digest=Path("/s/b/d"), parent=None, login=False,
            contestant=True)


def test_no_role_of_the_launcher_runs_as_root(tmp_path):
    """The install container is made by the install program, not the launcher: no role spec or
    container spec carries a user, and the launcher, which gives every role its own uid, refuses
    uid 0."""
    from tests.substrate.docker.launcher_fakes import fake_world, launcher

    assert not {"user", "uid"} & {f.name for f in dataclasses.fields(RoleSpec)}
    assert not {"user", "uid"} & {f.name for f in dataclasses.fields(ContainerSpec)}
    with fake_world(tmp_path) as (_daemon, client, host):
        with pytest.raises(ValueError, match="never run as root"):
            launcher(client, host, tmp_path / "scratch", uid=0)


def test_a_saved_archive_names_no_image(tmp_path, fake):
    with DockerClient(fake.path) as client:
        from hls.substrate.docker.api import ContainerConfig
        cid = client.create_container(ContainerConfig(image=BASE_ID, argv=("true",), user="1000:1000", labels={},
                                                      network_mode="none", cpuset=(0,), memory_bytes=1 << 20,
                                                      pids_limit=8))
        image_id = client.commit_container(cid, repo="hls/job-7", tag="t", config={})
        client.save_image(image_id, tmp_path / "s.tar", timeout=10)
    with tarfile.open(tmp_path / "s.tar") as tar:
        index = json.load(tar.extractfile("index.json"))
    assert not any(m.get("annotations", {}).get("io.containerd.image.name") for m in index["manifests"])


def test_a_committed_images_configuration_is_replaced_and_its_layers_kept(tmp_path):
    made = write_archive(tmp_path / "a.tar")
    with tarfile.open(made.path) as tar:
        layers_before = json.load(tar.extractfile(f"blobs/sha256/{made.manifest_digest[7:]}"))["layers"]
    image = with_config(made.path, {"Env": ["A=1"], "Labels": {"x": "y"}})
    assert image == archive_image(made.path) and image.config_digest != made.config_digest
    with tarfile.open(made.path) as tar:
        manifest = json.load(tar.extractfile(f"blobs/sha256/{image.manifest_digest[7:]}"))
        config = json.load(tar.extractfile(f"blobs/sha256/{image.config_digest[7:]}"))
        names = tar.getnames()
    assert manifest["layers"] == layers_before and config["config"] == {"Env": ["A=1"], "Labels": {"x": "y"}}
    assert f"blobs/sha256/{made.config_digest[7:]}" not in names and f"blobs/sha256/{made.manifest_digest[7:]}" not in names


def test_an_image_configuration_keeps_the_image_fields_that_are_set():
    reported = {"Hostname": "", "AttachStdout": False, "User": "", "Env": ["A=1"], "Entrypoint": None, "Cmd": [],
                "Labels": {"l": "v"}, "WorkingDir": "/w", "Image": "", "Tty": False}
    assert image_config(reported) == {"Env": ["A=1"], "Labels": {"l": "v"}, "WorkingDir": "/w"}
