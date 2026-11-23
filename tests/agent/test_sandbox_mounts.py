"""The exec server's files reach the sandbox the way a contestant's container may see host
paths: copied into the sandbox role's own node-local directory and bound from there, the
programs read-only, the control directory writable. The same files bound from anywhere else are
refused before anything launches."""

from __future__ import annotations

import re
import shutil
import stat
import sys
from pathlib import Path

import pytest

from hls.agent import exec_protocol as protocol
from hls.substrate.container import ContainerSpec, Mount
from tests.pipeline.test_stage_context_images import SHAPE, _ctx, _image_pin
from tests.substrate import fake_apptainer

AGENT = Path(protocol.__file__).parent


@pytest.fixture
def apptainer(tmp_path, monkeypatch):
    """The fake apptainer on PATH."""
    bin_dir = tmp_path / "bin"
    fake_apptainer.install(bin_dir)
    monkeypatch.setenv("PATH", f"{bin_dir}:{Path(sys.executable).parent}:/usr/bin:/bin")
    for knob in ("FAKE_APPTAINER_FAIL", "FAKE_EXTRACT_FAIL", "FAKE_EXTRACT_LOG", "FAKE_TREE_GIB"):
        monkeypatch.delenv(knob, raising=False)
    return bin_dir


def _stage_files(directory: Path) -> tuple[Path, Path]:
    """The exec directory with the three files and the 0700 control directory, under *directory*."""
    exec_dir, control = directory / "exec", directory / "control"
    exec_dir.mkdir(parents=True)
    for name in protocol.SANDBOX_FILES:
        shutil.copyfile(AGENT / name, exec_dir / name)
    control.mkdir(mode=0o700)
    return exec_dir, control


def _spec(ctx, image, exec_dir: Path, control: Path) -> ContainerSpec:
    return ContainerSpec(image=image, command=protocol.server_command(shm_bytes=1 << 30, uid=1000, gid=1000, cli_launches=("exec claude ",)),
                         workdir=ctx.role_dir("sandbox") / "wd",
                         mounts=(Mount(control, protocol.CONTROL_TARGET, read_only=False),
                                 Mount(exec_dir, protocol.EXEC_TARGET)))


def test_the_exec_files_and_the_control_directory_are_bound_from_the_sandbox_roles_own_directory(tmp_path,
                                                                                                  apptainer):
    ctx = _ctx(tmp_path)
    image = _image_pin(ctx, tmp_path)
    exec_dir, control = _stage_files(ctx.role_dir("sandbox"))
    assert stat.S_IMODE(control.stat().st_mode) == 0o700
    ctx.start_container("sandbox", _spec(ctx, image, exec_dir, control), SHAPE, owner="contestant",
                        env={protocol.TOKEN_ENV: "t" * 32})
    [role] = ctx.launcher.launched[-1:]
    binds = [role.argv[i + 1] for i, a in enumerate(role.argv) if a == "--bind"]
    assert f"{exec_dir}:{protocol.EXEC_TARGET}:ro" in binds
    assert f"{control}:{protocol.CONTROL_TARGET}" in binds
    # Every path of the server's command line is one of the two binds, or the image's python.
    command = protocol.server_command(shm_bytes=1 << 30, uid=1000, gid=1000, cli_launches=("exec claude ",))
    assert role.argv[-len(command):] == command
    paths = [a for a in command if a.startswith("/")]
    assert all(p == protocol.SANDBOX_PYTHON or p.startswith((protocol.EXEC_TARGET + "/", protocol.CONTROL_TARGET))
               for p in paths)
    ctx.wait_roles()


@pytest.mark.parametrize("which", ["exec", "control"])
@pytest.mark.parametrize("where", ["checkout", "attempt", "other_role", "package"])
def test_the_same_files_bound_from_outside_the_roles_own_directory_are_refused(tmp_path, apptainer, which, where):
    ctx = _ctx(tmp_path)
    image = _image_pin(ctx, tmp_path)
    exec_dir, control = _stage_files(ctx.role_dir("sandbox"))
    elsewhere = {"checkout": ctx.plan.code.checkout / "src" / "hls" / "agent", "attempt": ctx.attempt_dir / which,
                 "other_role": ctx.role_dir("driver") / which, "package": AGENT}[where]
    reason = {"checkout": "is in the checkout", "attempt": "is in the attempt directory"}.get(
        where, "is not in the role's own directory")
    if which == "exec":
        exec_dir = elsewhere
    else:
        control = elsewhere
    target = protocol.EXEC_TARGET if which == "exec" else protocol.CONTROL_TARGET
    with pytest.raises(ValueError, match=re.escape(f"mount {target} source {elsewhere} {reason}")):
        ctx.start_container("sandbox", _spec(ctx, image, exec_dir, control), SHAPE, owner="contestant")
    assert [r.name for r in ctx.launcher.launched] == [f"extract-{image.file_sha256[:12]}"]
