"""Docker images built, pinned and inventoried against the Docker daemon of the machine the tests
run on.

Opt-in, as every ``integration`` test (``HLS_RUN_SLOW=1``), and skipped where no Docker Engine
serving the client's API version answers on the default socket, or the daemon has no buildx. The
images are stand-ins for the catalog's: a base rendered from a ``BaseImage`` declaration on a
small pinned public image (Ubuntu 24.04 with curl), with the catalog's snapshot, uv pin and base
environment, and a layer on it through the parent's OCI layout. They pull that image, the
Dockerfile frontend, Ubuntu packages from the snapshot and wheels from PyPI, and the BuildKit
image when the daemon lacks it (``HLS_TEST_BUILDKIT_IMAGE``). Each role runs on the host in a
systemd scope that holds it to 4 GiB and to four CPUs in two ranges, as a launcher holds a role.

Every image and container the tests make carries ``hls.test=docker-images``; a builder cannot
carry a label, and is found by the names its role fixes. All of them are removed at the end,
also when a test fails, and the tests check that none is left."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from hls.recipes import catalog, pins
from hls.recipes import inventory as inv
from hls.recipes.model import BaseImage, LayerImage
from hls.recipes.render_docker import build_inputs, render_dockerfile, syntax_line
from hls.store.digest import DIGEST_MISMATCH_EXIT, sha256_file
from hls.substrate.container import PinnedImage
from hls.substrate.docker.api import BindMount, ContainerConfig, DockerClient, DockerError, NotFound
from hls.substrate.docker.image_role import archive_image, builder_objects
from hls.substrate.docker.images import (BUILT_FILE, IMAGE_FILE, PARENT_FILE, DockerImages, builder_name, image_name,
                                         job_leftovers, lay_out_build, remove_job_leftovers, reported_build)

pytestmark = pytest.mark.integration

LABEL = "docker-images"
LABELS = {"hls.test": LABEL, "hls.test.run": uuid.uuid4().hex[:12]}
BUILDKIT = os.environ.get("HLS_TEST_BUILDKIT_IMAGE",
                          "moby/buildkit:v0.35.1@sha256:baf6138eb8c37644800c7c913fba051e840ebb9ba13eaf92d92ef9a1712d0b4c")
UBUNTU = "buildpack-deps:24.04-curl@sha256:6b03a2e5a50d3b482003a040045952cfbd7ed8081e1d6ac85f283b6e44fbff27"
#: What a role is held to, as a launcher holds a role's container.
ROLE_MEMORY = "4G"
ROLE_CPUS = "0-1,4-5"
LOCKS = {
    "apt-standin.txt": "python3=3.12.3-0ubuntu2.1\n",
    "pip-standin-base.txt": ("six==1.16.0 \\\n"
                             "    --hash=sha256:8abb2f1d86890a2dfb989f9a77cfcfd3e47c2a354b01111771326f8aa26e0254\n"
                             "pip==24.2 \\\n"
                             "    --hash=sha256:2cd581cf58ab7fcfca4ce8efa6dcacd0de5bf8d0a3eb9ec927e07405f4d9e2a2\n"),
    "pip-standin-layer.txt": ("idna==3.10 \\\n"
                              "    --hash=sha256:946d195a0d259cbba61165e88e65941f16e9b36ea6ddb97f00452bae8b1287d3\n"),
}
BASE = BaseImage(name="standin-base", cuda_ref=UBUNTU, snapshot=pins.SNAPSHOT, apt_lock="apt-standin.txt",
                 pip_lock="pip-standin-base.txt", constraints=["six==1.16.0"], env=catalog.BASE_ENV)
LAYER = LayerImage(name="standin-layer", parent="standin-base", pip_lock="pip-standin-layer.txt")


@pytest.fixture(scope="module")
def docker() -> Iterator[DockerClient]:
    if os.environ.get("HLS_RUN_SLOW") != "1":
        pytest.skip("live Docker tests are opt-in: set HLS_RUN_SLOW=1")
    if shutil.which("docker") is None or shutil.which("systemd-run") is None:
        pytest.skip("no docker command, or no systemd-run to hold a role to its limits")
    client = DockerClient()
    try:
        client.check()
    except DockerError as e:
        client.close()
        pytest.skip(f"no Docker Engine to test against: {e}")
    if subprocess.run(["docker", "buildx", "version"], capture_output=True).returncode != 0:
        client.close()
        pytest.skip("the docker command has no buildx")
    try:
        yield client
    finally:
        leftover = client.list_containers(labels=LABELS)
        for row in leftover:
            client.remove_container(row.id, force=True)
        client.close()
        assert not leftover, f"containers left by a test: {[r.id for r in leftover]}"


@pytest.fixture(scope="module")
def area(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("images")


def _runtime(area: Path, job: str) -> DockerImages:
    scratch = area / job
    scratch.mkdir(exist_ok=True)
    return DockerImages(scratch, job_id=job, python=sys.executable, buildkit_image=BUILDKIT)


def _role(argv: list[str], env: dict[str, str], log: Path) -> int:
    """Run a role's command line in a scope of its own memory and CPUs, its output in *log*."""
    held = ["systemd-run", "--user", "--scope", "--quiet", "--expand-environment=no", "-p", f"MemoryMax={ROLE_MEMORY}",
            "-p", "MemorySwapMax=0",
            "--", "taskset", "-c", ROLE_CPUS, *argv]
    base = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}
    for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        if name in os.environ:
            base[name] = os.environ[name]
    with log.open("w") as out:
        return subprocess.run(held, env={**base, **env}, stdout=out, stderr=subprocess.STDOUT).returncode


def _builder_left(name: str) -> list[str]:
    container_name, volume = builder_objects(name)
    found = []
    names = subprocess.run(["docker", "container", "ls", "--all", "--format", "{{.Names}}"], capture_output=True,
                           text=True, check=True).stdout.split()
    if container_name in names:
        found.append(container_name)
    volumes = subprocess.run(["docker", "volume", "ls", "--quiet"], capture_output=True, text=True,
                             check=True).stdout.split()
    if volume in volumes:
        found.append(volume)
    return found


@dataclass(frozen=True)
class Built:
    work: Path
    archive: Path
    sha256: str


def _build(images: DockerImages, area: Path, decl, *, parent: Built | None = None, dockerfile: str | None = None) -> tuple[int, Path]:
    work = area / f"build-{decl.name}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir()
    locks = area / "locks"
    locks.mkdir(exist_ok=True)
    for name, text in LOCKS.items():
        (locks / name).write_text(text)
    text = dockerfile or render_dockerfile(decl, locks=locks, uv=pins.UV) + f"LABEL hls.test={LABEL}\n"
    recipe = lay_out_build(work, text, locks=build_inputs(decl, locks_dir=locks).locks)
    built_on = None if parent is None else (parent.archive, parent.sha256, work / PARENT_FILE)
    argv, env = images.build_role(recipe, work / IMAGE_FILE, digest=work / "image.sha256", parent=built_on,
                                  login=False, contestant=False)
    rc = _role(argv, env, work / "build.log")
    assert not _builder_left(builder_name(images.job_id, work)), "the build left its builder"
    return rc, work


@pytest.fixture(scope="module")
def built(docker, area) -> dict[str, Built]:
    images = _runtime(area, "9001")
    out: dict[str, Built] = {}
    for decl in (BASE, LAYER):
        rc, work = _build(images, area, decl, parent=out.get("standin-base"))
        assert rc == 0, (work / "build.log").read_text()[-4000:]
        sha256 = (work / "image.sha256").read_text().split()[0]
        assert sha256_file(work / IMAGE_FILE)[0] == sha256
        assert not (work / PARENT_FILE).exists() and not (work / "parent-layout").exists()
        record = reported_build(work / BUILT_FILE)
        assert record is not None and record.buildkit_version.startswith("v")
        image = archive_image(work / IMAGE_FILE)
        assert (record.manifest_digest, record.config_digest) == (image.manifest_digest, image.config_digest)
        out[decl.name] = Built(work=work, archive=work / IMAGE_FILE, sha256=sha256)
    return out


@pytest.fixture(scope="module")
def pins_made(docker) -> Iterator[list[tuple[DockerImages, PinnedImage]]]:
    made: list[tuple[DockerImages, PinnedImage]] = []
    try:
        yield made
    finally:
        for images, image in made:
            images.remove(image)
        for _, image in made:
            with pytest.raises(NotFound):
                docker.inspect_image(image.ref)


def _pin(images: DockerImages, area: Path, archive: Path, sha256: str, label: str) -> tuple[int, Path]:
    own = images.scratch / f"pin-{label}"
    shutil.rmtree(own, ignore_errors=True)
    own.mkdir()
    argv, env = images.pin_role(archive, sha256=sha256, own=own)
    return _role(argv, env, own / "pin.log"), own


def _pinned(images: DockerImages, area: Path, b: Built, label: str, made: list) -> PinnedImage:
    rc, own = _pin(images, area, b.archive, b.sha256, label)
    assert rc == 0, (own / "pin.log").read_text()[-4000:]
    image = images.pinned(own, sha256=b.sha256, contestant=False)
    made.append((images, image))
    assert not (own / IMAGE_FILE).exists()
    return image


def _inventory(docker: DockerClient, image: PinnedImage, out: Path) -> inv.Inventory:
    """Run ``inventory.sh`` in a container of *image*, as the images stage's inventory role does:
    the script bound read-only, the container as the host user with no network."""
    out.mkdir(parents=True, exist_ok=True)
    config = ContainerConfig(
        image=image.ref, argv=("/bin/sh", "-c", f"/bin/sh {inv.INVENTORY_TARGET} > /out/inventory.txt"),
        user=f"{os.getuid()}:{os.getgid()}", labels=LABELS, network_mode="none", cpuset=(0, 1, 2, 3),
        memory_bytes=1 << 30, pids_limit=256,
        mounts=(BindMount(source=inv.SCRIPT, target=inv.INVENTORY_TARGET, read_only=True),
                BindMount(source=out, target="/out", read_only=False)))
    cid = docker.create_container(config)
    try:
        docker.start_container(cid)
        result = docker.wait_container(cid, timeout=600)
        assert result is not None and result.status_code == 0
    finally:
        docker.remove_container(cid, force=True)
    parsed = inv.parse_inventory((out / "inventory.txt").read_text())
    (out / "inventory.json").write_text(parsed.model_dump_json(indent=2))
    return parsed


def test_base_and_layer_build_pin_and_inventory(docker, area, built, pins_made):
    images = _runtime(area, "9001")
    base = _pinned(images, area, built["standin-base"], "base", pins_made)
    layer = _pinned(images, area, built["standin-layer"], "layer", pins_made)
    for image, b in ((base, built["standin-base"]), (layer, built["standin-layer"])):
        record = reported_build(b.work / BUILT_FILE)
        assert record is not None and image.ref in (record.manifest_digest, record.config_digest)
        assert docker.inspect_image(image.ref).labels.get("hls.test") == LABEL
    base_inv = _inventory(docker, base, area / "inventory-base")
    layer_inv = _inventory(docker, layer, area / "inventory-layer")
    assert base_inv.apt["python3"] == "3.12.3-0ubuntu2.1"
    assert base_inv.pip["six"] == "1.16.0" and base_inv.pip["pip"] == "24.2"
    assert "idna" not in base_inv.pip and layer_inv.pip["idna"] == "3.10" and layer_inv.pip["six"] == "1.16.0"
    assert {"VIRTUAL_ENV", "UV_CONSTRAINT", "PIP_CONSTRAINT"} <= set(layer_inv.env)
    assert not inv.version_mismatches(layer_inv, BASE.constraints)


def test_one_archive_pinned_by_two_jobs_is_one_image_until_both_remove_it(docker, area, built, pins_made):
    b = built["standin-base"]
    one, two = _runtime(area, "9002"), _runtime(area, "9003")
    first = _pinned(one, area, b, "twice", pins_made)
    second = _pinned(two, area, b, "twice", pins_made)
    assert first.ref == second.ref and first.check_sha256 == second.check_sha256
    tags = set(docker.inspect_image(first.ref).repo_tags)
    assert {":".join(image_name(one.job_id, b.sha256)), ":".join(image_name(two.job_id, b.sha256))} <= tags
    pins_made.remove((one, first))
    one.remove(first)
    assert docker.inspect_image(second.ref).id == second.ref
    assert one.left(b.sha256, contestant=False) is None
    left = two.left(b.sha256, contestant=True)
    assert isinstance(left, PinnedImage) and left.ref == second.ref and left.contestant


def test_a_changed_byte_is_refused_before_anything_loads(docker, area, built):
    b = built["standin-base"]
    changed = area / "changed.oci.tar"
    shutil.copyfile(b.archive, changed)
    with changed.open("r+b") as f:
        f.seek(changed.stat().st_size // 2)
        byte = f.read(1)
        f.seek(-1, io.SEEK_CUR)
        f.write(bytes([byte[0] ^ 1]))
    images = _runtime(area, "9004")
    rc, own = _pin(images, area, changed, b.sha256, "changed")
    assert rc == DIGEST_MISMATCH_EXIT
    assert f"refused: file {changed} has sha256" in (own / "pin.log").read_text()
    with pytest.raises(ValueError, match="ended without loading"):
        images.pinned(own, sha256=b.sha256, contestant=False)
    assert images.left(b.sha256, contestant=False) is None


def _rewrite_index(archive: Path, out: Path, manifest_digest: str) -> None:
    with tarfile.open(archive) as src, tarfile.open(out, "w") as dst:
        for member in src.getmembers():
            data = src.extractfile(member).read() if member.isfile() else None
            if member.name == "index.json":
                index = json.loads(data)
                index["manifests"][0]["digest"] = manifest_digest
                data = json.dumps(index).encode()
                member.size = len(data)
            dst.addfile(member, io.BytesIO(data) if data is not None else None)


def test_an_archive_whose_index_names_another_manifest_is_refused(docker, area, built):
    b, other = built["standin-base"], built["standin-layer"]
    forged = area / "forged.oci.tar"
    _rewrite_index(b.archive, forged, archive_image(other.archive).manifest_digest)
    sha256 = sha256_file(forged)[0]
    images = _runtime(area, "9005")
    rc, own = _pin(images, area, forged, sha256, "forged")
    assert rc == DIGEST_MISMATCH_EXIT
    log = (own / "pin.log").read_text()
    assert f"refused: the image archive {forged}: it names manifest" in log and "which it does not hold" in log
    with pytest.raises(ValueError, match="ended without loading"):
        images.pinned(own, sha256=sha256, contestant=False)
    with pytest.raises(NotFound):
        docker.inspect_image(":".join(image_name(images.job_id, sha256)))


def test_a_failed_build_leaves_no_builder_even_one_an_earlier_role_left(docker, area):
    images = _runtime(area, "9006")
    work = area / "build-standin-broken"
    work.mkdir(exist_ok=True)
    name = builder_name(images.job_id, work)
    earlier = area / "earlier-config"
    earlier.mkdir(exist_ok=True)
    made = subprocess.run(["docker", "buildx", "create", "--name", name, "--driver", "docker-container",
                           "--driver-opt", f"image={BUILDKIT}", "--bootstrap"],
                          env={**os.environ, "DOCKER_CONFIG": str(earlier)}, capture_output=True, text=True)
    assert made.returncode == 0, made.stderr
    assert len(_builder_left(name)) == 2
    broken = f"{syntax_line()}\nFROM {UBUNTU}\nRUN exit 3\nLABEL hls.test={LABEL}\n"
    rc, work = _build(images, area, BASE.model_copy(update={"name": "standin-broken"}), dockerfile=broken)
    assert rc not in (0, DIGEST_MISMATCH_EXIT)
    assert "docker buildx build exited" in (work / "build.log").read_text()
    assert not (work / IMAGE_FILE).exists() and not (work / "image.sha256").exists()
    assert not _builder_left(name)


def test_a_jobs_end_removes_what_it_left_and_keeps_an_image_another_job_names(docker, area, built, pins_made):
    b = built["standin-base"]
    ended, running = _runtime(area, "9007"), _runtime(area, "9008")
    _pinned(ended, area, b, "left", pins_made)
    kept = _pinned(running, area, b, "kept", pins_made)
    name = builder_name(ended.job_id, area / "build-killed")
    made = subprocess.run(["docker", "buildx", "create", "--name", name, "--driver", "docker-container",
                           "--driver-opt", f"image={BUILDKIT}", "--bootstrap"],
                          env={**os.environ, "DOCKER_CONFIG": str(area / "killed-config")}, capture_output=True,
                          text=True)
    assert made.returncode == 0, made.stderr
    assert len(job_leftovers(docker, ended.job_id)) == 3
    remove_job_leftovers(docker, ended.job_id)
    assert job_leftovers(docker, ended.job_id) == [] and not _builder_left(name)
    own = ":".join(image_name(running.job_id, b.sha256))
    assert own in docker.inspect_image(kept.ref).repo_tags
    assert job_leftovers(docker, running.job_id) == [own]
    remove_job_leftovers(docker, running.job_id)
    assert job_leftovers(docker, running.job_id) == []
    # The image goes with its last name; another test's job may still name it.
    try:
        names = docker.inspect_image(kept.ref).repo_tags
    except NotFound:
        names = ()
    assert not any(name.startswith(("hls/job-9007:", "hls/job-9008:")) for name in names)
