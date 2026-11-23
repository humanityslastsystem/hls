"""A container's device nodes (``ContainerSpec.devices``): the one path under /dev the runtime
binds, the node's own at the same path and writable, after the declared mounts. A spec without
one renders the command line it renders with none."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from hls.substrate.apptainer.argv import ApptainerRuntime
from hls.substrate.container import ContainerSpec, Mount
from tests.substrate.sandboxes import stand_in

ROOT = Path("/node/scratch/sandboxes/" + "ab" * 32)
WORK = Path("/node/scratch/roles/driver/wd")
ENV = "/dev/fd/3"


def _spec(**changes) -> ContainerSpec:
    fields = {"image": stand_in(ROOT), "command": ("python", "-m", "hls.drive.role"), "workdir": WORK,
              "mounts": (Mount(Path("/site/images"), "/hls/sandboxes/images"),
                         Mount(Path("/site/overlays"), "/hls/sandboxes/overlays", read_only=False))}
    return ContainerSpec(**{**fields, **changes})


def test_a_device_is_bound_writable_at_its_own_path_after_the_mounts_and_nothing_else_changes():
    runtime = ApptainerRuntime()
    plain = runtime.argv(_spec(), env_file=ENV)
    argv = runtime.argv(_spec(devices=("/dev/fuse",)), env_file=ENV)
    at = plain.index("--env-file")
    assert plain[at - 4:at] == ["--bind", "/site/images:/hls/sandboxes/images:ro",
                                "--bind", "/site/overlays:/hls/sandboxes/overlays"]
    assert argv == [*plain[:at], "--bind", "/dev/fuse:/dev/fuse", *plain[at:]]


def test_a_spec_without_devices_renders_as_one_that_names_none():
    spec = _spec()
    assert spec.devices == ()
    runtime = ApptainerRuntime()
    assert runtime.argv(spec, env_file=ENV) == runtime.argv(dataclasses.replace(spec, devices=()), env_file=ENV)
    assert not [a for a in runtime.argv(spec, env_file=ENV) if a.startswith("/dev/") and a != ENV]


@pytest.mark.parametrize("device, match", [
    ("/dev", "a node under /dev"),
    ("/dev/", "a node under /dev"),
    ("dev/fuse", "a node under /dev"),
    ("/tmp/fuse", "a node under /dev"),
    ("/dev/../etc/passwd", "a node under /dev"),
    ("//dev/fuse", "a node under /dev"),
    ("/dev/fu se", "cannot be passed to apptainer exactly"),
    ("/dev/fuse:/x", "cannot be passed to apptainer exactly"),
])
def test_a_device_the_runtime_cannot_bind_exactly_is_refused(device, match):
    with pytest.raises(ValueError, match=match):
        ApptainerRuntime().argv(_spec(devices=(device,)), env_file=ENV)


def test_a_device_named_twice_is_refused():
    with pytest.raises(ValueError, match="named twice"):
        ApptainerRuntime().argv(_spec(devices=("/dev/fuse", "/dev/fuse")), env_file=ENV)


def test_a_mount_under_dev_is_still_refused_beside_a_device():
    with pytest.raises(ValueError, match="under /dev"):
        ApptainerRuntime().argv(_spec(mounts=(Mount(Path("/dev/fuse"), "/dev/fuse", read_only=False),),
                                      devices=("/dev/fuse",)), env_file=ENV)
