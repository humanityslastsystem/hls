"""The image roles' program: an archive's identity, the role's own limits, the fetch before the
program, and what the build and the pin refuse before and after they reach a daemon (a fake one,
``fake_daemon``)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

from hls.store.digest import DIGEST_MISMATCH_EXIT
from hls.substrate.docker import image_role
from hls.substrate.docker.api import DockerClient, RequestRefused
from hls.substrate.docker.image_role import (FETCH_EXEC, ArchiveRefused, archive_image, builder_objects, cpu_list,
                                             layout_image, own_memory)
from tests.substrate.docker.archives import write_archive
from tests.substrate.docker.fake_daemon import FakeDocker


@pytest.fixture
def fake(tmp_path):
    daemon = FakeDocker(tmp_path / "d.sock")
    with daemon.serving():
        yield daemon


def test_an_archive_is_identified_by_its_manifest_and_config(tmp_path):
    made = write_archive(tmp_path / "a.tar")
    image = archive_image(made.path)
    assert (image.manifest_digest, image.config_digest) == (made.manifest_digest, made.config_digest)


def test_an_unpacked_layout_gives_the_same_image(tmp_path):
    made = write_archive(tmp_path / "a.tar")
    with tarfile.open(made.path) as tar:
        tar.extractall(tmp_path / "layout", filter="data")
    assert layout_image(tmp_path / "layout") == archive_image(made.path)


def _drop(name_of):
    def edit(members, index):
        del members[name_of(members, index)]
    return edit


def _blob(digest: str) -> str:
    return f"blobs/sha256/{digest[7:]}"


def _second_manifest(members, index):
    index["manifests"].append(dict(index["manifests"][0]))


def _named(members, index):
    index["manifests"][0]["annotations"]["io.containerd.image.name"] = "docker.io/library/busybox:latest"


def _other_manifest(members, index):
    index["manifests"][0]["digest"] = "sha256:" + "0" * 64


def _changed_manifest(members, index):
    name = _blob(index["manifests"][0]["digest"])
    members[name] = members[name] + b" "


def _index_list(members, index):
    index["manifests"] = {"not": "a list"}


def _not_a_manifest(members, index):
    index["manifests"][0]["mediaType"] = "application/vnd.oci.image.index.v1+json"


def _no_layout(members, index):
    del members["oci-layout"]


def _drop_layer(members, index):
    manifest = json.loads(members[_blob(index["manifests"][0]["digest"])])
    del members[_blob(manifest["layers"][0]["digest"])]


def _drop_config(members, index):
    manifest = json.loads(members[_blob(index["manifests"][0]["digest"])])
    del members[_blob(manifest["config"]["digest"])]


def _no_docker_manifest(members, index):
    del members["manifest.json"]


def _docker_manifest(change):
    def edit(members, index):
        entries = json.loads(members["manifest.json"])
        change(entries)
        members["manifest.json"] = json.dumps(entries).encode()
    return edit


def _other_config(entries):
    entries[0]["Config"] = "blobs/sha256/" + "0" * 64


def _other_layers(entries):
    entries[0]["Layers"] = []


def _tagged(entries):
    entries[0]["RepoTags"] = ["busybox:latest"]


def _two_images(entries):
    entries.append(dict(entries[0]))


@pytest.mark.parametrize(("edit", "words"), [
    (_no_docker_manifest, "holds no manifest.json, which the classic image store loads"),
    (_docker_manifest(_other_config), "names another config or other layers"),
    (_docker_manifest(_other_layers), "names another config or other layers"),
    (_docker_manifest(_tagged), "its manifest.json names the image"),
    (_docker_manifest(_two_images), "names other than one image"),
    (_second_manifest, "names 2 manifests, not one"),
    (_index_list, "names no list of manifests"),
    (_named, "names the image (io.containerd.image.name)"),
    (_other_manifest, "which it does not hold"),
    (_changed_manifest, "does not hash to its name"),
    (_not_a_manifest, "not an image manifest"),
    (_no_layout, "holds no oci-layout file"),
    (_drop_layer, "names layer"),
    (_drop_config, "names config"),
])
def test_an_archive_that_is_not_one_image_is_refused(tmp_path, edit, words):
    made = write_archive(tmp_path / "a.tar", edit=edit)
    with pytest.raises(ArchiveRefused, match=words.replace("(", r"\(").replace(")", r"\)")):
        archive_image(made.path)


def test_a_member_that_appears_twice_is_refused(tmp_path):
    made = write_archive(tmp_path / "a.tar", extra=(("index.json", b"{}"),))
    with pytest.raises(ArchiveRefused, match="member index.json appears twice"):
        archive_image(made.path)


def test_a_file_that_is_no_tar_is_refused(tmp_path):
    path = tmp_path / "a.tar"
    path.write_bytes(b"not a tar file at all" * 100)
    with pytest.raises(ArchiveRefused, match="not a tar file"):
        archive_image(path)


def test_cpu_lists_join_ranges():
    assert cpu_list([3, 0, 1, 2, 6, 8, 9]) == "0-3,6,8-9"
    assert cpu_list([5]) == "5"
    with pytest.raises(ValueError):
        cpu_list([])


def test_the_memory_limit_is_the_least_on_the_path_to_the_namespace_root(tmp_path):
    root = tmp_path / "cg"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "memory.max").write_text("4096\n")
    (root / "a" / "b" / "memory.max").write_text("max\n")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/a/b\n")
    assert own_memory(proc_cgroup=proc, root=root) == 4096
    (root / "a" / "b" / "memory.max").write_text("1024\n")
    assert own_memory(proc_cgroup=proc, root=root) == 1024
    proc.write_text("0::/\n")
    assert own_memory(proc_cgroup=proc, root=root) is None
    (root / "memory.max").write_text("2048\n")
    assert own_memory(proc_cgroup=proc, root=root) == 2048


def test_a_builders_objects_are_named_by_the_builder():
    assert builder_objects("hls-0123-build-s-a") == ("buildx_buildkit_hls-0123-build-s-a0",
                                                     "buildx_buildkit_hls-0123-build-s-a0_state")


def _fetch(tmp_path: Path, data: bytes, sha256: str) -> subprocess.CompletedProcess[str]:
    source, copy = tmp_path / "src", tmp_path / "copy"
    source.write_bytes(data)
    return subprocess.run(["/bin/sh", "-c", FETCH_EXEC, "hls-pin", str(source), str(copy), sha256,
                           "/bin/sh", "-c", 'echo "ran $1"', "prog", "arg"], capture_output=True, text=True)


def test_the_fetch_becomes_the_program_once_the_copy_is_checked(tmp_path):
    data = b"archive bytes"
    done = _fetch(tmp_path, data, hashlib.sha256(data).hexdigest())
    assert (done.returncode, done.stdout) == (0, "ran arg\n")
    assert (tmp_path / "copy").read_bytes() == data


def test_the_fetch_refuses_a_file_of_another_digest_and_runs_nothing(tmp_path):
    done = _fetch(tmp_path, b"archive bytes", "0" * 64)
    assert done.returncode == DIGEST_MISMATCH_EXIT and done.stdout == ""
    assert done.stderr.startswith("refused: file ")
    assert not (tmp_path / "copy").exists()


def _pin_args(tmp_path: Path, fake: FakeDocker, archive: Path) -> list[str]:
    return ["pin", "--source", str(archive), "--copy", str(archive), "--name", "hls/0123456789abcdef:" + "a" * 64,
            "--out", str(tmp_path / "pinned.json"), "--socket", str(fake.path)]


def test_a_pin_refuses_a_daemon_whose_image_is_another(tmp_path, fake, capsys):
    # The fake daemon gives a loaded image an ID of its own, which is neither digest.
    made = write_archive(tmp_path / "copy.tar")
    assert image_role.main(_pin_args(tmp_path, fake, made.path)) == DIGEST_MISMATCH_EXIT
    err = capsys.readouterr().err
    assert err.startswith("refused: the Docker daemon loaded ") and made.manifest_digest in err
    assert len(fake.loaded_bodies) == 1
    assert not any(image["RepoTags"] for image in fake.images.values())
    assert not (tmp_path / "pinned.json").exists()


@pytest.mark.parametrize(("store", "digest"), [("containerd", "manifest_digest"), ("classic", "config_digest")])
def test_a_pin_takes_the_id_each_kind_of_image_store_gives(tmp_path, fake, store, digest):
    # containerd's store names a loaded archive by its manifest, the classic store by its config.
    fake.image_store = store
    made = write_archive(tmp_path / "copy.tar")
    assert image_role.main(_pin_args(tmp_path, fake, made.path)) == 0
    pinned = json.loads((tmp_path / "pinned.json").read_text())
    assert pinned["image_id"] == getattr(made, digest)
    assert (pinned["manifest_digest"], pinned["config_digest"]) == (made.manifest_digest, made.config_digest)
    assert fake.images[getattr(made, digest)]["RepoTags"] == ["hls/0123456789abcdef:" + "a" * 64]
    assert not made.path.exists()


def test_a_classic_store_refuses_an_archive_without_manifest_json(tmp_path, fake):
    fake.image_store = "classic"
    made = write_archive(tmp_path / "copy.tar", edit=_no_docker_manifest)
    with DockerClient(fake.path) as client, pytest.raises(RequestRefused, match="does not contain a manifest.json"):
        client.load_image(made.path, timeout=10)


def test_a_pin_refuses_an_archive_that_is_not_one_image_before_loading_it(tmp_path, fake, capsys):
    made = write_archive(tmp_path / "copy.tar", edit=_second_manifest)
    assert image_role.main(_pin_args(tmp_path, fake, made.path)) == DIGEST_MISMATCH_EXIT
    assert "refused: the image archive " in capsys.readouterr().err
    assert fake.loaded_bodies == []


def test_a_pin_that_reaches_no_daemon_fails_without_refusing(tmp_path, capsys):
    made = write_archive(tmp_path / "copy.tar")
    args = ["pin", "--source", str(made.path), "--copy", str(made.path), "--name", "hls/x:" + "a" * 64,
            "--out", str(tmp_path / "pinned.json"), "--socket", str(tmp_path / "none.sock")]
    assert image_role.main(args) == image_role.FAILED_EXIT
    assert "could not be pinned" in capsys.readouterr().err


def _build_args(tmp_path: Path, *extra: str) -> list[str]:
    return ["build", "--docker", "docker-not-run", "--buildkit-image", "moby/buildkit:test", "--builder", "b",
            "--recipe", str(tmp_path / "Dockerfile"), "--contexts", str(tmp_path / "contexts"),
            "--out", str(tmp_path / "image.oci.tar"), "--digest", str(tmp_path / "image.sha256"),
            "--record", str(tmp_path / "image.json"), *extra]


def test_a_build_needs_its_docker_configuration(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    assert image_role.main(_build_args(tmp_path)) == image_role.FAILED_EXIT
    assert "names no DOCKER_CONFIG" in capsys.readouterr().err


def test_a_build_takes_the_login_out_of_its_environment_and_needs_both_halves(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "config"))
    monkeypatch.setenv("REGISTRY_USERNAME", "someone")
    monkeypatch.delenv("REGISTRY_PASSWORD", raising=False)
    assert image_role.main(_build_args(tmp_path, "--login")) == image_role.FAILED_EXIT
    assert "REGISTRY_PASSWORD is missing or empty" in capsys.readouterr().err
    import os
    assert "REGISTRY_USERNAME" not in os.environ


def test_a_build_without_a_memory_limit_creates_no_builder(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(image_role, "own_memory", lambda: None)
    monkeypatch.setattr(image_role, "_run", lambda *a, **k: pytest.fail("a command ran"))
    assert image_role.main(_build_args(tmp_path)) == image_role.FAILED_EXIT
    assert "no memory limit" in capsys.readouterr().err


def test_a_build_refuses_a_context_named_as_the_parent(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(image_role, "own_memory", lambda: 1 << 30)
    monkeypatch.setattr(image_role, "_run", lambda *a, **k: pytest.fail("a command ran"))
    (tmp_path / "contexts" / "Locks").mkdir(parents=True)
    assert image_role.main(_build_args(tmp_path)) == image_role.FAILED_EXIT
    assert "['Locks']" in capsys.readouterr().err


def test_a_build_refuses_a_parent_archive_that_is_not_one_image(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(image_role, "own_memory", lambda: 1 << 30)
    monkeypatch.setattr(image_role, "_run", lambda *a, **k: pytest.fail("a command ran"))
    made = write_archive(tmp_path / "parent.oci.tar", edit=_other_manifest)
    assert image_role.main(_build_args(tmp_path, "--parent", str(made.path))) == DIGEST_MISMATCH_EXIT
    assert capsys.readouterr().err.startswith("refused: the parent's image archive ")
    assert not (tmp_path / "parent-layout").exists()


def test_a_driver_option_is_one_csv_field():
    assert image_role.driver_opt("cpuset-cpus", "0-7,128-135") == '"cpuset-cpus=0-7,128-135"'
    with pytest.raises(ValueError):
        image_role.driver_opt("image", 'a"b')


def test_a_build_quotes_its_driver_options_and_writes_an_archive_both_stores_load(tmp_path, monkeypatch):
    """The build program end to end with buildx answered by a stand-in: a cpuset of two ranges
    reaches the builder as one option, and the output is the docker exporter's with OCI media types."""
    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "config"))
    monkeypatch.setattr(image_role, "own_memory", lambda: 1 << 30)
    monkeypatch.setattr(image_role.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3, 128, 129})
    calls: list[list[str]] = []

    def run(argv, env, *, stdin=None, capture=False):
        calls.append(list(argv))
        assert env["DOCKER_CONFIG"] == str(tmp_path / "config")
        out = ""
        if argv[1:4] == ["buildx", "inspect", "--bootstrap"]:
            # A boot longer than the status read's timeout leaves the bootstrap's read no version.
            out = 'Error: Get "http://docker/v1.54/containers/b0/json": context deadline exceeded\n'
        elif argv[1:3] == ["buildx", "inspect"]:
            out = "Name: b\nBuildKit version:      v0.35.1\n"
        elif argv[1:3] == ["container", "inspect"]:
            out = json.dumps({"Memory": 1 << 30, "MemorySwap": 1 << 30, "CpusetCpus": "0-3,128-129"})
        elif argv[1:3] == ["buildx", "build"]:
            write_archive(tmp_path / "image.oci.tar")
        return subprocess.CompletedProcess(argv, 0, stdout=out)

    monkeypatch.setattr(image_role, "_run", run)
    assert image_role.main(_build_args(tmp_path)) == 0
    create = next(c for c in calls if c[1:3] == ["buildx", "create"])
    opts = [create[i + 1] for i, word in enumerate(create) if word == "--driver-opt"]
    assert opts == ['"image=moby/buildkit:test"', f'"memory={1 << 30}"', f'"memory-swap={1 << 30}"',
                    '"cpuset-cpus=0-3,128-129"']
    build = next(c for c in calls if c[1:3] == ["buildx", "build"])
    assert f"--output=type=docker,oci-mediatypes=true,dest={tmp_path / 'image.oci.tar'}" in build
    record = json.loads((tmp_path / "image.json").read_text())
    image = archive_image(tmp_path / "image.oci.tar")
    assert (record["manifest_digest"], record["config_digest"]) == (image.manifest_digest, image.config_digest)
    assert record["buildkit_version"] == "v0.35.1"
    assert ["docker-not-run", "buildx", "inspect", "b"] in calls
    assert not any("--timeout" in call for call in calls)
    assert ["docker-not-run", "buildx", "rm", "--force", "b"] in calls
