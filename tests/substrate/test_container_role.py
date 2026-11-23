"""The container interface's types: a container role, a role spec that carries one, and the
Apptainer image runtime's roles."""

from __future__ import annotations

from pathlib import Path

import pytest

from hls.egress.policy import NetworkPolicy, SealPolicy
from hls.spec.resources import ResourceShape
from hls.substrate.apptainer.images import BUILD_SHELL, EXTRACT_SHELL, RUNSCRIPT, ApptainerImages
from hls.substrate.base import RoleSpec
from hls.substrate.container import ContainerRole, ContainerSpec, ImageRuntime, PinnedImage, SealPlan
from hls.substrate.sandbox import IMAGE_FILE, sandbox_paths
from hls.substrate.seal import seal_files
from tests.substrate.sandboxes import stand_in

SHA = "a" * 64
SHAPE = ResourceShape(gpus=0, cpus=1, memory_gib=1, disk_gib=0, time_sec=60)


def _spec(tmp_path: Path, *, seal: SealPlan | None = None) -> ContainerSpec:
    return ContainerSpec(image=stand_in(tmp_path / "image"), command=("true",), workdir=tmp_path / "wd", seal=seal)


def _sealed(tmp_path: Path) -> ContainerSpec:
    return _spec(tmp_path, seal=SealPlan(policy=SealPolicy(network=NetworkPolicy()), index=False))


def test_a_container_role_has_seal_files_exactly_when_its_container_is_sealed(tmp_path):
    files = seal_files(tmp_path / "seal")
    ContainerRole(spec=_spec(tmp_path), env_file=tmp_path / "env", seal_files=None)
    ContainerRole(spec=_sealed(tmp_path), env_file=tmp_path / "env", seal_files=files, ucounts={"max_user_namespaces": 2})
    with pytest.raises(ValueError, match="seal files exactly when"):
        ContainerRole(spec=_spec(tmp_path), env_file=tmp_path / "env", seal_files=files)
    with pytest.raises(ValueError, match="seal files exactly when"):
        ContainerRole(spec=_sealed(tmp_path), env_file=tmp_path / "env", seal_files=None)
    with pytest.raises(ValueError, match="only a sealed container"):
        ContainerRole(spec=_spec(tmp_path), env_file=tmp_path / "env", seal_files=None, ucounts={"max_user_namespaces": 2})


def test_a_container_role_spec_leaves_its_command_environment_and_env_file_to_the_launcher(tmp_path):
    role = ContainerRole(spec=_spec(tmp_path), env_file=tmp_path / "env", seal_files=None)
    spec = RoleSpec(name="srv", argv=(), shape=SHAPE, owner="harness", log_dir=tmp_path, container=role)
    assert spec.container is role
    for given in ({"argv": ("true",)}, {"env": {"A": "1"}}, {"env_file": tmp_path / "env"},
                  {"env_file": tmp_path / "env", "env_fd": True}):
        with pytest.raises(ValueError, match="are the launcher's"):
            RoleSpec(**{"name": "srv", "argv": (), "shape": SHAPE, "owner": "harness", "log_dir": tmp_path,
                        "container": role, **given})


def test_a_pinned_image_names_what_it_runs_from():
    with pytest.raises(ValueError, match="needs a reference"):
        PinnedImage(ref="", file_sha256=SHA, check_sha256=SHA)


def test_apptainer_pins_an_image_by_extracting_its_checked_copy(tmp_path):
    images = ApptainerImages(tmp_path / "scratch", apptainer="/opt/apptainer/bin/apptainer")
    assert isinstance(images, ImageRuntime)
    assert (images.image_file, images.runscript) == (IMAGE_FILE, RUNSCRIPT) == ("image.sif", "/.singularity.d/runscript")
    own = tmp_path / "scratch" / "roles" / "extract-aaaaaaaaaaaa"
    own.mkdir(parents=True)
    argv, env = images.pin_role(Path("/store/image.sif"), sha256=SHA, own=own)
    root, manifest = sandbox_paths(tmp_path / "scratch", SHA)
    dest = Path(argv[7])
    assert argv[:7] == ["/bin/sh", "-c", EXTRACT_SHELL, "hls-extract", "/store/image.sif", SHA,
                        "/opt/apptainer/bin/apptainer"]
    assert dest.parent == root.parent and dest.name.startswith(f".{own.name}-") and argv[8] == str(own / IMAGE_FILE)
    assert env == {"APPTAINER_CACHEDIR": str(own / "cache"), "APPTAINER_TMPDIR": str(own / "tmp-build")}
    assert (own / "tmp-build").is_dir()
    with pytest.raises(ValueError, match="without extracting /store/image.sif into a directory"):
        images.pinned(own, sha256=SHA, contestant=False)
    (own / "tmp-build").rmdir()
    images.pin_role(Path("/store/image.sif"), sha256=SHA, own=own)
    Path(images._pending[own][0], "etc").mkdir(parents=True)
    pinned = images.pinned(own, sha256=SHA, contestant=True)
    assert (pinned.ref, pinned.file_sha256, pinned.contestant) == (str(root), SHA, True) and manifest.is_file()
    assert images.left(SHA, contestant=False) == PinnedImage(ref=str(root), file_sha256=SHA,
                                                             check_sha256=pinned.check_sha256)
    images.remove(pinned)
    assert images.left(SHA, contestant=False) is None


def test_apptainer_refuses_an_extraction_path_it_cannot_pass_exactly(tmp_path):
    images = ApptainerImages(tmp_path / "scratch")
    own = tmp_path / "scratch" / "roles" / "x"
    own.mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot be passed to apptainer exactly"):
        images.pin_role(Path("/store/a b.sif"), sha256=SHA, own=own)


def test_apptainer_builds_a_recipe_with_its_layer_cache_in_the_role_directory(tmp_path):
    work = tmp_path / "build-a"
    work.mkdir()
    argv, env = ApptainerImages(tmp_path).build_role(work / "recipe.def", work / IMAGE_FILE, digest=work / "image.sha256",
                                                     parent=None, login=False, contestant=False)
    assert argv == ["/bin/sh", "-c", BUILD_SHELL, "hls-build", "apptainer", str(work / IMAGE_FILE),
                    str(work / "recipe.def"), str(work / "image.sha256"), "", "", ""]
    assert env == {"APPTAINER_CACHEDIR": str(work / "cache"), "APPTAINER_TMPDIR": str(work / "build-tmp")}
    assert (work / "build-tmp").is_dir()


def test_only_a_harness_role_speaks_to_the_container_daemon(tmp_path):
    assert RoleSpec(name="pin", argv=("true",), shape=SHAPE, owner="harness", log_dir=tmp_path, daemon=True).daemon
    with pytest.raises(ValueError, match="only a harness role"):
        RoleSpec(name="pin", argv=("true",), shape=SHAPE, owner="contestant", log_dir=tmp_path, daemon=True)
