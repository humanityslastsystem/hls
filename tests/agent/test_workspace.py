"""The sandbox's node-local layout: its directories and modes, the programs it copies in, the
mounts the sandbox and the harness containers get, and the writable directories its disk caps."""

from __future__ import annotations

import os
import stat
import subprocess
import sys

import pytest

from hls.agent import commit, exec_protocol, workspace
from hls.agent.workspace import SandboxLayout
from hls.budget import mcp


def test_the_layout_makes_every_directory_with_the_channel_and_control_directories_private(tmp_path):
    layout = SandboxLayout.create(tmp_path / "role")
    for d in (layout.workspace, layout.agent_logs, layout.logs / "preflight", layout.task / "files",
              layout.task / "skills", layout.installed, layout.workdir, layout.exec_dir, layout.programs,
              layout.harbor):
        assert d.is_dir(), d
    for d in (layout.control, layout.channel):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d


def test_the_programs_are_copies_of_this_package_and_the_mcp_server_is_executable(tmp_path):
    layout = SandboxLayout.create(tmp_path / "role")
    here = os.path.dirname(exec_protocol.__file__)
    for name in exec_protocol.SANDBOX_FILES:
        assert (layout.exec_dir / name).read_bytes() == open(os.path.join(here, name), "rb").read()
    assert sorted(p.name for p in layout.programs.iterdir()) == ["commit.py"]
    assert (layout.programs / "commit.py").read_bytes() == open(commit.__file__, "rb").read()
    assert layout.mcp_file.read_text() == mcp.executable_text(exec_protocol.SANDBOX_PYTHON)
    assert layout.mcp_file.read_text().splitlines()[0] == f"#!{exec_protocol.SANDBOX_PYTHON} {mcp.SHEBANG_FLAGS}"
    assert os.access(layout.mcp_file, os.X_OK)
    # The copy runs on a bare interpreter, as the image's runs it.
    done = subprocess.run([sys.executable, "-I", str(layout.programs / "commit.py"), "--help"],
                          capture_output=True, text=True, env={})
    assert done.returncode == 0, done.stderr


def test_a_layout_over_an_earlier_one_is_refused(tmp_path):
    SandboxLayout.create(tmp_path / "role")
    with pytest.raises(FileExistsError):
        SandboxLayout.create(tmp_path / "role")


def test_a_leftover_workspace_alone_is_refused(tmp_path):
    """The stage clears the role's directory first; a workspace left there would hand an earlier
    attempt's files to this one."""
    (tmp_path / "role" / "workspace").mkdir(parents=True)
    (tmp_path / "role" / "workspace" / "stale.txt").write_text("from an earlier attempt")
    with pytest.raises(FileExistsError):
        SandboxLayout.create(tmp_path / "role")


def test_the_sandbox_mounts_and_their_access(tmp_path):
    layout = SandboxLayout.create(tmp_path / "role")
    mounts = {m.target: (m.source, m.read_only) for m in layout.mounts()}
    assert mounts == {
        "/workspace": (layout.workspace, False), "/logs": (layout.logs, False), "/task": (layout.task, True),
        "/installed-agent": (layout.installed, False), "/run/hls": (layout.channel, True),
        "/run/hls-control": (layout.control, False), "/opt/hls-exec": (layout.exec_dir, True),
        "/opt/hls-budget/get-budget-mcp": (layout.mcp_file, True)}
    assert mounts["/run/hls"][0] / mcp.SOCKET_NAME == layout.budget_socket
    assert mounts["/run/hls-control"][0] / exec_protocol.SOCKET_NAME == layout.socket
    # Every source is under the role's own directory: node-local, as a contestant's must be.
    assert all(source.is_relative_to(layout.root) for source, _ in mounts.values())
    assert [(m.target, m.read_only) for m in layout.program_mounts()] == [(commit.COMMIT_TARGET, True)]


def test_the_sandboxs_writable_directories_hold_nothing_the_harness_writes(tmp_path):
    """The disk cap walks the workdir and the writable mounts; none of them holds or lies in a
    path the harness writes while the sandbox runs, so the harness's files never count against
    the contestant, and the contestant writes nothing beside them."""
    layout = SandboxLayout.create(tmp_path / "role")
    assert layout.writable() == (layout.workdir, layout.workspace, layout.logs, layout.installed, layout.control)
    for written in layout.writable():
        for harness in layout.harness_written():
            assert not written.is_relative_to(harness) and not harness.is_relative_to(written), (written, harness)
    mounted = {m.source for m in layout.mounts()}
    assert {layout.channel, layout.exec_dir, layout.mcp_file} <= mounted
    assert all(m.read_only for m in layout.mounts() if m.source in {layout.channel, layout.exec_dir, layout.mcp_file})
    assert not ({layout.programs, layout.harbor} & mounted)
    assert workspace.AGENT_WRITABLE == ("/workspace", "/tmp", "/var/tmp", "$HOME")


def test_harbors_transfers_reach_only_the_writable_shared_directories(tmp_path):
    layout = SandboxLayout.create(tmp_path / "role")
    assert layout.transfer_paths() == {"/workspace": layout.workspace, "/logs": layout.logs, "/task": layout.task,
                                       "/installed-agent": layout.installed}
    # Harbor writes the agent's logs to /logs/agent, which the /logs transfer puts here.
    assert layout.agent_logs == layout.logs / "agent"
