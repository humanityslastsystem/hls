"""The step image: its recipe pins each of its inputs, and the harness builds it into the store's
cache once per recipe, keyed by the recipe's text."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from hls.store.cache import ContentCache
from hls.store.digest import sha256_file
from hls.store.posix import PosixStore
from hls.substrate.docker import step_image as step
from hls.substrate.docker.api import BindMount, ContainerConfig, DockerClient, DockerError, NotFound
from hls.substrate.docker.launcher import STEP_IMAGE_REPO, archive_digests, load_step_image
from hls.substrate.docker.images import BUILT_FILE, IMAGE_FILE, RECIPE_FILE
from tests.substrate.docker.archives import write_archive

_DIGEST = r"@sha256:[0-9a-f]{64}"


def test_every_image_the_step_image_reads_is_pinned_by_digest():
    text = step.RECIPE.read_text()
    syntax = text.splitlines()[0]
    assert re.fullmatch(rf"# syntax=docker/dockerfile:[0-9.]+{_DIGEST}", syntax), syntax
    froms = re.findall(r"^FROM (\S+)", text, re.MULTILINE)
    assert len(froms) == 3
    for ref in froms:
        assert re.fullmatch(rf"[a-z0-9/._-]+:[A-Za-z0-9][A-Za-z0-9.-]*{_DIGEST}", ref), ref


def test_every_package_the_step_image_installs_is_pinned_by_version_from_a_snapshot():
    text = step.RECIPE.read_text()
    installs = re.findall(r"apt-get install [^\n]*", text)
    assert installs
    for line in installs:
        packages = [w for w in line.split()[2:] if not w.startswith("-")]
        assert packages and all(re.fullmatch(r"[a-z0-9.+-]+=[A-Za-z0-9.+~:-]+", p) for p in packages), line
    assert re.search(r"^snapshot=https://snapshot\.ubuntu\.com/ubuntu/[0-9]{8}T[0-9]{6}Z/$", text, re.MULTILINE)
    # A role whose TLS client reads the system's roots needs the image's own CA store.
    assert "ca-certificates=" in text
    # A submission's checkout role (hls.package.checkout) runs git in the step image.
    assert re.search(r"apt-get install [^\n]* git=", text)
    assert re.search(r"^for tool in [^\n]* git ", text, re.MULTILINE)


def test_the_step_images_key_is_its_recipes():
    assert re.fullmatch(r"step-[0-9a-f]{16}", step.step_key())
    assert step.step_key() == step.step_key()


@pytest.fixture
def cache(tmp_path) -> ContentCache:
    return ContentCache(PosixStore(tmp_path / "store"), lock_ttl_sec=30)


def _fake_build(tmp_path: Path, calls: list[dict], status: int = 0):
    def build_image(**kwargs):
        calls.append(kwargs)
        if status == 0:
            made = write_archive(kwargs["out"])
            kwargs["digest"].write_text(sha256_file(made.path)[0] + "\n")
            kwargs["record"].write_text(json.dumps({"manifest_digest": made.manifest_digest,
                                                    "config_digest": made.config_digest,
                                                    "buildkit_version": "v0.35.1"}))
        return status
    return build_image


def test_a_store_without_the_step_image_builds_it_once_into_the_cache(tmp_path, cache, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(step, "build_image", _fake_build(tmp_path, calls))
    monkeypatch.setattr(step.os, "sched_getaffinity", lambda pid: {0, 1, 4, 5})
    assert step.step_image_file(cache) is None
    archive, sha256 = step.ensure_step_image(cache, socket=Path("/run/d.sock"), buildkit_image="sha256:" + "b" * 64,
                                             work_root=tmp_path / "scratch")
    assert archive.name == IMAGE_FILE and archive.parent.name == step.step_key()
    assert sha256_file(archive)[0] == sha256
    assert sorted(p.name for p in archive.parent.iterdir()) == sorted([BUILT_FILE, "ENTRY.json", IMAGE_FILE,
                                                                       RECIPE_FILE])
    assert (archive.parent / RECIPE_FILE).read_text() == step.RECIPE.read_text()
    [call] = calls
    assert (call["builder"], call["cpus"], call["memory"], call["parent"]) == (
        f"hls-{step.step_key()}", "0-1,4-5", step.STEP_BUILD_MEMORY_GIB << 30, None)
    assert call["env"]["DOCKER_HOST"] == "unix:///run/d.sock" and call["contexts"] == ()
    assert call["env"]["DOCKER_CONFIG"].startswith(str(tmp_path / "scratch"))
    assert list((tmp_path / "scratch").iterdir()) == []
    assert step.step_image_file(cache) == (archive, sha256)
    again = step.ensure_step_image(cache, socket=Path("/run/d.sock"), buildkit_image="sha256:" + "b" * 64,
                                   work_root=tmp_path / "scratch")
    assert again == (archive, sha256) and len(calls) == 1


def test_a_failed_build_publishes_nothing(tmp_path, cache, monkeypatch):
    monkeypatch.setattr(step, "build_image", _fake_build(tmp_path, [], status=1))
    with pytest.raises(RuntimeError, match="the step image could not be built"):
        step.ensure_step_image(cache, socket=Path("/run/d.sock"), buildkit_image="sha256:" + "b" * 64,
                               work_root=tmp_path / "scratch")
    assert step.step_image_file(cache) is None
    assert list((tmp_path / "scratch").iterdir()) == []


def test_a_host_without_the_docker_cli_or_buildx_is_named_in_one_sentence(tmp_path):
    assert "the Docker CLI (no-such-docker-command)" in step.docker_cli_problem("no-such-docker-command")
    no_buildx = tmp_path / "docker"
    no_buildx.write_text('#!/bin/sh\n[ "$1" = --version ] && echo "Docker version 29.5.2" && exit 0\nexit 1\n')
    no_buildx.chmod(0o755)
    problem = step.docker_cli_problem(str(no_buildx))
    assert problem is not None and "buildx plugin" in problem


@pytest.mark.integration
def test_the_step_image_builds_from_its_recipe_with_a_ca_store_and_loads(tmp_path):
    """The recipe built for real into a store's cache on this machine's daemon, loaded as the
    launcher loads it, and run as the host user with no network: it holds the CA bundle at the path
    OpenSSL, Go and the native-roots TLS clients of a role read on Ubuntu, and the Docker CLI with
    buildx. Opt-in (``HLS_RUN_SLOW=1``): it pulls the BuildKit image (``HLS_TEST_BUILDKIT_IMAGE``) and
    the recipe's inputs. The image goes with its tag; the builder and its volume go with the build."""
    if os.environ.get("HLS_RUN_SLOW") != "1":
        pytest.skip("live Docker tests are opt-in: set HLS_RUN_SLOW=1")
    if step.docker_cli_problem() is not None:
        pytest.skip(step.docker_cli_problem())
    buildkit = os.environ.get("HLS_TEST_BUILDKIT_IMAGE",
                              "moby/buildkit@sha256:baf6138eb8c37644800c7c913fba051e840ebb9ba13eaf92d92ef9a1712d0b4c")
    client = DockerClient()
    try:
        client.check()
    except DockerError as e:
        pytest.skip(f"no Docker Engine to test against: {e}")
    cache = ContentCache(PosixStore(tmp_path / "store"), lock_ttl_sec=60)
    archive, sha256 = step.ensure_step_image(cache, socket=client.socket, buildkit_image=buildkit,
                                             work_root=tmp_path / "scratch")
    tag = f"{STEP_IMAGE_REPO}:{archive_digests(archive)[0].removeprefix('sha256:')[:32]}"
    image = load_step_image(client, archive, sha256=sha256, scratch=tmp_path / "scratch")
    out = tmp_path / "out"
    out.mkdir()
    cid = None
    try:
        cid = client.create_container(ContainerConfig(
            image=image, argv=("/bin/sh", "-c", "test -s /etc/ssl/certs/ca-certificates.crt && "
                               "grep -c 'BEGIN CERTIFICATE' /etc/ssl/certs/ca-certificates.crt > /out/certs && "
                               "docker buildx version > /out/buildx && git --version > /out/git"),
            user=f"{os.getuid()}:{os.getgid()}", labels={"hls.test": "docker-images"}, network_mode="none",
            cpuset=(0,), memory_bytes=256 << 20, pids_limit=64,
            mounts=(BindMount(source=out, target="/out", read_only=False),)))
        client.start_container(cid)
        result = client.wait_container(cid, timeout=120)
        assert result is not None and result.status_code == 0
        assert int((out / "certs").read_text()) > 100
        assert " v0.34.0 " in (out / "buildx").read_text()
        assert (out / "git").read_text().startswith("git version 2.43.0")
    finally:
        if cid is not None:
            client.remove_container(cid, force=True)
        try:
            client.remove_image(tag)
        except NotFound:
            pass
        client.close()
    left = subprocess.run(["docker", "volume", "ls", "--quiet", "--filter", f"name=hls-{step.step_key()}"],
                          capture_output=True, text=True).stdout.split()
    assert left == []
