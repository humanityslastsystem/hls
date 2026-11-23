"""The Docker image runtime: the roles' command lines, the names a job's images carry, the build's
layout and records, and what ``pinned``, ``left``, ``discard`` and ``remove`` do with a daemon's
images (a fake daemon, ``fake_daemon``)."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from hls.recipes import catalog, pins
from hls.recipes.render import TASKS_DIR
from hls.recipes.render import recipe_key as def_recipe_key
from hls.recipes.render_docker import build_inputs, dockerfile_key, render_dockerfile
from hls.stages.images import ImageMeta, image_keys
from hls.substrate.apptainer.images import ApptainerImages
from hls.substrate.container import PinnedImage
from hls.substrate.docker.api import ContainerConfig, DockerClient
from hls.substrate.docker.image_role import FETCH_EXEC, BuiltImage, Pinned
from hls.substrate.docker.images import (BUILT_FILE, CONTEXTS, IMAGE_FILE, PINNED_FILE, RECIPE_FILE, RECORDS,
                                         ROLE_MODULE, RUNSCRIPT, DockerImages, builder_name, image_name,
                                         job_leftovers, job_repository, lay_out_build, remove_job_leftovers,
                                         reported_build)
from hls.store.digest import sha256_file
from tests.substrate.docker.archives import write_archive
from tests.substrate.docker.fake_daemon import FakeDocker

SHA = "a" * 64
MANIFEST = "sha256:" + "b" * 64
CONFIG = "sha256:" + "c" * 64
PYTHON = "/opt/toolchain/bin/python"


@pytest.fixture
def fake(tmp_path):
    daemon = FakeDocker(tmp_path / "d.sock")
    with daemon.serving():
        yield daemon


def _images(tmp_path: Path, fake: FakeDocker | None = None, job: str = "7") -> DockerImages:
    scratch = tmp_path / job
    scratch.mkdir(exist_ok=True)
    socket = fake.path if fake is not None else Path("/var/run/docker.sock")
    return DockerImages(scratch, job_id=job, python=PYTHON, buildkit_image="sha256:" + "d" * 64, socket=socket)


def _add(fake: FakeDocker, image_id: str, *names: str) -> None:
    fake.images[image_id] = {"Id": image_id, "RepoTags": list(names), "RepoDigests": [], "Config": {"Labels": {}},
                             "Architecture": "amd64", "Os": "linux", "Size": 1}


def test_the_runtime_names_its_file_and_runscript():
    assert DockerImages.image_file == IMAGE_FILE == "image.oci.tar"
    assert DockerImages.runscript == RUNSCRIPT == "/.hls/runscript"


def test_a_jobs_image_names_are_its_own():
    assert image_name("7", SHA) == ("hls/job-7", SHA) and job_repository("7") == "hls/job-7"
    assert image_name("7", SHA) != image_name("8", SHA)
    with pytest.raises(ValueError):
        image_name("7", "A" * 64)
    with pytest.raises(ValueError):
        image_name("Job 7", SHA)


def test_the_pin_role_fetches_then_runs_the_pin_program(tmp_path):
    images = _images(tmp_path)
    own = tmp_path / "pin-x"
    own.mkdir()
    store = tmp_path / "store" / IMAGE_FILE
    argv, env = images.pin_role(store, sha256=SHA, own=own)
    assert argv[:7] == ["/bin/sh", "-c", FETCH_EXEC, "hls-pin", str(store), str(own / IMAGE_FILE), SHA]
    assert argv[7:11] == [PYTHON, "-m", ROLE_MODULE, "pin"]
    assert argv[argv.index("--name") + 1] == ":".join(image_name(images.job_id, SHA))
    assert argv[argv.index("--out") + 1] == str(own / PINNED_FILE)
    assert argv[argv.index("--socket") + 1] == "/var/run/docker.sock"
    assert env == {}
    assert stat.S_IMODE((images.scratch / RECORDS).stat().st_mode) == 0o700


@pytest.mark.parametrize("path", ["relative/image.oci.tar", "/a,b/image.oci.tar", "/a/../b", "/a@b/image.oci.tar"])
def test_a_path_buildx_cannot_read_back_is_refused(tmp_path, path):
    own = tmp_path / "pin-x"
    own.mkdir()
    with pytest.raises(ValueError):
        _images(tmp_path).pin_role(Path(path), sha256=SHA, own=own)


def test_the_build_role_of_a_base_runs_the_build_program(tmp_path):
    images = _images(tmp_path)
    own = tmp_path / "build-s-a"
    own.mkdir()
    argv, env = images.build_role(own / RECIPE_FILE, own / IMAGE_FILE, digest=own / "image.sha256", parent=None,
                                  login=False, contestant=False)
    assert argv[:4] == [PYTHON, "-m", ROLE_MODULE, "build"]
    assert argv[argv.index("--builder") + 1] == builder_name(images.job_id, own)
    assert argv[argv.index("--contexts") + 1] == str(own / CONTEXTS)
    assert argv[argv.index("--record") + 1] == str(own / BUILT_FILE)
    assert argv[argv.index("--buildkit-image") + 1] == "sha256:" + "d" * 64
    assert "--login" not in argv and "--parent" not in argv
    assert env == {"DOCKER_CONFIG": str(own / "docker-config"), "DOCKER_HOST": "unix:///var/run/docker.sock"}
    assert stat.S_IMODE((own / "docker-config").stat().st_mode) == 0o700


def test_the_build_role_of_a_layer_fetches_its_parent_first(tmp_path):
    images = _images(tmp_path)
    own = tmp_path / "build-layer"
    own.mkdir()
    stored, copy = tmp_path / "store" / IMAGE_FILE, own / "parent.oci.tar"
    argv, _ = images.build_role(own / RECIPE_FILE, own / IMAGE_FILE, digest=own / "image.sha256",
                                parent=(stored, SHA, copy), login=True, contestant=False)
    assert argv[:7] == ["/bin/sh", "-c", FETCH_EXEC, "hls-build", str(stored), str(copy), SHA]
    assert argv[7:11] == [PYTHON, "-m", ROLE_MODULE, "build"]
    assert argv[argv.index("--parent") + 1] == str(copy) and "--login" in argv


def test_the_build_role_refuses_a_build_that_reads_a_contestants_files(tmp_path):
    own = tmp_path / "build-submission"
    own.mkdir()
    with pytest.raises(ValueError, match="privileged container, so it builds only images from the catalog's own"):
        _images(tmp_path).build_role(own / RECIPE_FILE, own / IMAGE_FILE, digest=own / "image.sha256", parent=None,
                                     login=False, contestant=True)
    assert not (own / "docker-config").exists()


def test_a_builders_name_is_fixed_by_the_job_and_the_role(tmp_path):
    assert builder_name("7", tmp_path / "build-s-a.2") == "hls-job-7-build-s-a.2"
    with pytest.raises(ValueError):
        builder_name("7", tmp_path / "Build A")


def test_a_buildkit_image_that_is_no_reference_is_refused(tmp_path):
    with pytest.raises(ValueError):
        DockerImages(tmp_path, job_id="7", python=PYTHON, buildkit_image="moby/buildkit:v1,memory=1")


def test_a_build_is_laid_out_with_its_lock_files_only(tmp_path):
    locks = tmp_path / "locks"
    locks.mkdir()
    (locks / "apt-x.txt").write_text("a=1\n")
    (locks / "other.txt").write_text("not copied\n")
    work = tmp_path / "work"
    work.mkdir()
    recipe = lay_out_build(work, "FROM x\n", locks={"apt-x.txt": locks / "apt-x.txt"})
    assert recipe == work / RECIPE_FILE and recipe.read_text() == "FROM x\n"
    assert [p.name for p in (work / CONTEXTS).iterdir()] == ["locks"]
    assert [p.name for p in (work / CONTEXTS / "locks").iterdir()] == ["apt-x.txt"]
    (tmp_path / "w2").mkdir()
    with pytest.raises(ValueError):
        lay_out_build(tmp_path / "w2", "FROM x\n", locks={"../x": locks / "apt-x.txt"})
    assert list((tmp_path / "w2").iterdir()) == []


def test_a_builds_record_is_read_back_or_none(tmp_path):
    built = BuiltImage(manifest_digest=MANIFEST, config_digest=CONFIG, buildkit_version="v0.35.1")
    (tmp_path / BUILT_FILE).write_text(built.model_dump_json())
    assert reported_build(tmp_path / BUILT_FILE) == built
    (tmp_path / BUILT_FILE).write_text('{"manifest_digest": "x"}')
    assert reported_build(tmp_path / BUILT_FILE) is None
    assert reported_build(tmp_path / "missing") is None


def _pin_done(images: DockerImages, fake: FakeDocker, own: Path, image_id: str = MANIFEST) -> str:
    """What a pin role leaves: the daemon's image under the job's name and its record in *own*."""
    own.mkdir(exist_ok=True)
    images.pin_role(own.parent / "store.oci.tar", sha256=SHA, own=own)
    name = ":".join(image_name(images.job_id, SHA))
    _add(fake, image_id, name)
    (own / PINNED_FILE).write_text(Pinned(name=name, image_id=image_id, manifest_digest=MANIFEST,
                                          config_digest=CONFIG).model_dump_json())
    return name


def test_pinned_records_the_image_and_left_finds_it_again(tmp_path, fake):
    images = _images(tmp_path, fake)
    _pin_done(images, fake, tmp_path / "pin-x")
    image = images.pinned(tmp_path / "pin-x", sha256=SHA, contestant=True)
    assert (image.ref, image.file_sha256, image.check_sha256, image.contestant) == (MANIFEST, SHA, MANIFEST[7:], True)
    assert (images.scratch / RECORDS / f"{SHA}.json").exists()
    again = _images(tmp_path, fake).left(SHA, contestant=False)
    assert again == PinnedImage(ref=MANIFEST, file_sha256=SHA, check_sha256=MANIFEST[7:])


def test_a_pin_role_that_wrote_nothing_pinned_nothing_and_its_name_goes(tmp_path, fake):
    images = _images(tmp_path, fake)
    name = _pin_done(images, fake, tmp_path / "pin-x")
    (tmp_path / "pin-x" / PINNED_FILE).unlink()
    with pytest.raises(ValueError, match="^ended without loading "):
        images.pinned(tmp_path / "pin-x", sha256=SHA, contestant=False)
    assert not any(name in i["RepoTags"] for i in fake.images.values())
    assert images.left(SHA, contestant=False) is None


def test_a_name_that_reaches_another_image_is_not_pinned(tmp_path, fake):
    images = _images(tmp_path, fake)
    name = _pin_done(images, fake, tmp_path / "pin-x")
    fake.images[MANIFEST]["RepoTags"].remove(name)
    _add(fake, "sha256:" + "f" * 64, name)
    with pytest.raises(ValueError, match="not at the " + MANIFEST):
        images.pinned(tmp_path / "pin-x", sha256=SHA, contestant=False)
    assert not (images.scratch / RECORDS / f"{SHA}.json").exists()


def test_left_says_why_an_image_cannot_be_reused_and_remove_left_clears_it(tmp_path, fake):
    images = _images(tmp_path, fake)
    name = _pin_done(images, fake, tmp_path / "pin-x")
    images.pinned(tmp_path / "pin-x", sha256=SHA, contestant=False)
    fake.images[MANIFEST]["RepoTags"].remove(name)
    assert images.left(SHA, contestant=False) == f"the Docker daemon holds no image {name}, which a pin recorded"
    images.remove_left(SHA)
    assert images.left(SHA, contestant=False) is None
    _add(fake, MANIFEST, name)
    assert "which no record of a pin names" in images.left(SHA, contestant=False)
    images.remove_left(SHA)
    assert MANIFEST not in fake.images
    with pytest.raises(ValueError):
        images.left("x", contestant=False)


def test_remove_takes_only_the_jobs_own_name(tmp_path, fake):
    one, two = _images(tmp_path, fake, "7"), _images(tmp_path, fake, "8")
    _pin_done(one, fake, tmp_path / "7" / "pin-x")
    first = one.pinned(tmp_path / "7" / "pin-x", sha256=SHA, contestant=False)
    other = ":".join(image_name(two.job_id, SHA))
    fake.images[MANIFEST]["RepoTags"].append(other)
    one.remove(first)
    assert fake.images[MANIFEST]["RepoTags"] == [other]
    assert not (one.scratch / RECORDS / f"{SHA}.json").exists()
    fake.images[MANIFEST]["RepoTags"].remove(other)
    fake.images[MANIFEST]["RepoTags"].append(":".join(image_name(one.job_id, SHA)))
    one.remove(first)
    assert MANIFEST not in fake.images


def test_a_daemon_that_refuses_a_removal_raises_oserror(tmp_path, fake):
    images = _images(tmp_path, fake)
    _pin_done(images, fake, tmp_path / "pin-x")
    image = images.pinned(tmp_path / "pin-x", sha256=SHA, contestant=False)
    fake.fail("DELETE", r"/v1\.44/images/.*", 409, "conflict: unable to remove repository reference (must force)")
    with pytest.raises(OSError, match="could not be removed"):
        images.remove(image)
    assert (images.scratch / RECORDS / f"{SHA}.json").exists()


def test_discard_removes_the_name_a_failed_pin_role_gave(tmp_path, fake):
    images = _images(tmp_path, fake)
    name = _pin_done(images, fake, tmp_path / "pin-x")
    images.discard(tmp_path / "pin-x")
    assert MANIFEST not in fake.images
    images.discard(tmp_path / "pin-x")
    assert name


def _builder(client: DockerClient, name: str) -> str:
    config = ContainerConfig(image="moby/buildkit:test", argv=("buildkitd",), user="1000:1000", labels={},
                             network_mode="bridge", cpuset=(0,), memory_bytes=1 << 30, pids_limit=64)
    return client.create_container(config, name=name)


def test_the_jobs_end_removes_its_builders_volumes_and_names_and_no_other_jobs(fake):
    fake.add_image("moby/buildkit:test")
    other_sha = "e" * 64
    _add(fake, MANIFEST, f"hls/job-7:{SHA}", f"hls/job-8:{SHA}")
    _add(fake, CONFIG, f"hls/job-7:{other_sha}")
    _add(fake, "sha256:" + "9" * 64, f"hls/job-70:{SHA}")
    fake.volumes = {"buildx_buildkit_hls-job-7-build-a0_state": set(), "buildx_buildkit_hls-job-70-build-a0_state": set()}
    with DockerClient(fake.path) as client:
        mine = _builder(client, "buildx_buildkit_hls-job-7-build-a0")
        theirs = _builder(client, "buildx_buildkit_hls-job-70-build-a0")
        assert sorted(job_leftovers(client, "7")) == sorted(["buildx_buildkit_hls-job-7-build-a0",
                                                             "buildx_buildkit_hls-job-7-build-a0_state",
                                                             f"hls/job-7:{SHA}", f"hls/job-7:{other_sha}"])
        remove_job_leftovers(client, "7")
        assert job_leftovers(client, "7") == []
    # The image job 8 also names stays, under job 8's name only; the one only job 7 named is gone.
    assert fake.images[MANIFEST]["RepoTags"] == [f"hls/job-8:{SHA}"]
    assert CONFIG not in fake.images
    assert fake.images["sha256:" + "9" * 64]["RepoTags"] == [f"hls/job-70:{SHA}"]
    assert mine not in fake.containers and theirs in fake.containers
    assert list(fake.volumes) == ["buildx_buildkit_hls-job-70-build-a0_state"]


def test_the_jobs_end_fails_naming_what_the_daemon_keeps(fake):
    _add(fake, MANIFEST, f"hls/job-7:{SHA}")
    fake.fail("DELETE", r"/v1\.44/images/.*", 409, "conflict: image is being used by a stopped container")
    with DockerClient(fake.path) as client, pytest.raises(RuntimeError, match=f"hls/job-7:{SHA} \\(409\\)"):
        remove_job_leftovers(client, "7")
    assert fake.images[MANIFEST]["RepoTags"] == [f"hls/job-7:{SHA}"]


def test_the_recipe_is_a_dockerfile_with_its_locks_context_and_its_key_the_dockerfiles(tmp_path):
    images = _images(tmp_path)
    base, layer = catalog.BASES["s-a"], catalog.LAYERS["s-a-tf-5-13-1"]
    assert (images.recipe_file, images.parent_file) == (RECIPE_FILE, "parent.oci.tar")
    for decl in (base, layer):
        work = tmp_path / f"build-{decl.name}"
        work.mkdir()
        text = images.write_recipe(decl, work, locks_dir=pins.LOCKS_DIR, uv=pins.UV, tasks_dir=TASKS_DIR)
        assert text == render_dockerfile(decl, locks=pins.LOCKS_DIR, uv=pins.UV) == (work / RECIPE_FILE).read_text()
        wanted = sorted(build_inputs(decl, locks_dir=pins.LOCKS_DIR).locks)
        assert sorted(p.name for p in (work / CONTEXTS / "locks").iterdir()) == wanted
    keys = image_keys([base, layer], images.recipe_key)
    assert keys["s-a"] == dockerfile_key(base, parent_key=None, locks_dir=pins.LOCKS_DIR, uv=pins.UV)
    assert keys != image_keys([base, layer], ApptainerImages.recipe_key)


def test_a_tasks_server_layer_is_laid_out_with_its_locks_and_files_in_their_modes(tmp_path):
    """A task's server layer on the bare CUDA base: its build context holds its lock and the
    server source it copies in, from the task's directory, each with its file's mode, which COPY
    gives the file in the image."""
    images = _images(tmp_path)
    decls = {d.name: d for d in catalog.task_images("multimodal_asr").decls}
    layer = decls["multimodal-asr-baseline"]
    assert layer.server and layer.files and decls[layer.parent].pip_lock is None
    work = tmp_path / "build-baseline"
    work.mkdir()
    text = images.write_recipe(layer, work, locks_dir=pins.LOCKS_DIR, uv=pins.UV, tasks_dir=TASKS_DIR)
    assert text == render_dockerfile(layer, locks=pins.LOCKS_DIR, uv=pins.UV) == (work / RECIPE_FILE).read_text()
    context = work / CONTEXTS / "locks"
    own = TASKS_DIR / "multimodal_asr"
    sources = {layer.pip_lock: own / "locks" / layer.pip_lock,
               **{Path(f).name: own / f for f in layer.files.values()}}
    assert sorted(p.name for p in context.iterdir()) == sorted(sources)
    for name, source in sources.items():
        copy = context / name
        assert copy.read_bytes() == source.read_bytes()
        assert stat.S_IMODE(copy.stat().st_mode) == stat.S_IMODE(source.stat().st_mode)


def test_the_apptainer_runtime_keys_and_writes_the_def_as_the_stage_did(tmp_path):
    images = ApptainerImages(tmp_path)
    layer = catalog.LAYERS["s-a-tf-5-13-1"]
    assert image_keys([catalog.BASES["s-a"], layer], images.recipe_key) == image_keys([catalog.BASES["s-a"], layer],
                                                                                      def_recipe_key)
    work = tmp_path / "build-layer"
    work.mkdir()
    text = images.write_recipe(layer, work, locks_dir=pins.LOCKS_DIR, uv=pins.UV, tasks_dir=TASKS_DIR)
    assert (work / "recipe.def").read_text() == text and f"From: {work / 'parent.sif'}" in text


def test_a_docker_build_records_its_archive_and_a_build_without_a_record_is_refused(tmp_path):
    images = _images(tmp_path)
    with pytest.raises(ValueError, match="^ended without recording the image it built"):
        images.build_record(tmp_path, "")
    built = BuiltImage(manifest_digest=MANIFEST, config_digest=CONFIG, buildkit_version="v0.35.1")
    (tmp_path / BUILT_FILE).write_text(built.model_dump_json())
    record = images.build_record(tmp_path, "")
    assert record == {"build_mode": None, "oci": built.model_dump()}
    meta = ImageMeta(name="s-a", kind="base", key="s-a-0123", parent=None, parent_key=None,
                     **record)
    assert '"oci": {' in meta.model_dump_json(indent=2)


def test_an_apptainer_images_meta_json_holds_no_oci_field():
    meta = ImageMeta(name="s-a", kind="base", key="s-a-0123", parent=None, parent_key=None,
                     **ApptainerImages(Path("/x")).build_record(Path("/x"), ""))
    assert set(meta.model_dump()) == {"name", "kind", "key", "parent", "parent_key", "build_mode"}
    assert '"oci"' not in meta.model_dump_json(indent=2)


@pytest.mark.parametrize(("store", "digest"), [("containerd", "manifest_digest"), ("classic", "config_digest")])
def test_an_archive_pins_and_is_found_again_in_either_kind_of_image_store(tmp_path, fake, store, digest):
    """The pin role's real command line against a daemon of each store kind: the image is the
    daemon's ID for it, the identity checked against it is the manifest digest in both."""
    fake.image_store = store
    made = write_archive(tmp_path / "store.oci.tar")
    sha256 = sha256_file(made.path)[0]
    scratch = tmp_path / "7"
    scratch.mkdir()
    images = DockerImages(scratch, job_id="7", python=sys.executable, buildkit_image="sha256:" + "d" * 64,
                          socket=fake.path)
    own = scratch / "pin-x"
    own.mkdir()
    argv, env = images.pin_role(made.path, sha256=sha256, own=own)
    done = subprocess.run(argv, env={**os.environ, **env}, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    image = images.pinned(own, sha256=sha256, contestant=False)
    assert (image.ref, image.check_sha256) == (getattr(made, digest), made.manifest_digest[7:])
    again = DockerImages(scratch, job_id="7", python=sys.executable, buildkit_image="sha256:" + "d" * 64,
                         socket=fake.path).left(sha256, contestant=False)
    assert again == image
    images.remove(image)
    assert getattr(made, digest) not in fake.images
