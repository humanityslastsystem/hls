"""The build sandbox's node-local layout: the host directory behind every path the sandbox sees.

Everything the sandbox can write lives on node-local disk in the sandbox role's own 0700
directory, so the contestant's work outlives the container and the stage reads it from the
host after the container has stopped. ``SandboxLayout`` names each directory and file once and
is the one source of the container paths the stage, the harbor environment and the commit
point use:

| Container path | Host path under the role directory | Access |
|---|---|---|
| ``/workspace`` | ``workspace`` | read-write: the agent's repository and working tree |
| ``/logs`` | ``logs`` | read-write: the agent's logs at ``/logs/agent``, the preflight's at ``/logs/preflight`` |
| ``/task`` | ``task`` | read-only: the task's files (``/task/files``) and skills (``/task/skills``) |
| ``/installed-agent`` | ``installed`` | read-write: harbor's setup writes here, and the image's root is read-only |
| ``/run/hls`` | ``channel`` (0700) | read-only: the budget service's socket |
| ``/run/hls-control`` | ``control`` (0700) | read-write: the exec server's socket and ready file, hidden from every command |
| ``/opt/hls-exec`` | ``exec`` | read-only: the exec server, its spawner and their protocol |
| ``/opt/hls-budget/get-budget-mcp`` | ``mcp/get-budget-mcp`` | read-only: the budget MCP server, executable |
| ``/tmp``, ``/var/tmp``, ``$HOME`` | ``wd`` | read-write: the container's workdir |

The sandbox's writable directories are ``/workspace``, ``/logs``, ``/installed-agent``, the
control directory and the workdir (``writable``), and its declared disk caps what they hold
together. None of them holds or lies in a directory the harness writes into while the sandbox
runs: the channel, the exec files, the MCP server, the programs and harbor's own directory
(``harness_written``). So nothing the harness writes counts against the contestant's cap, and
nothing the contestant writes lands beside a file the harness reads back.

The commit and snapshot containers get ``/workspace`` read-only, an out directory and the commit
program at ``COMMIT_TARGET``. The runner process writes only the small files here: the three exec
files, the MCP server and the commit program, copied from this package; the image's
``/workspace``, the task's files and its skills are copied by roles."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from hls.agent import commit, exec_protocol
from hls.budget import mcp
from hls.substrate.container import Mount

#: The agent's working tree and repository.
WORKSPACE = "/workspace"
#: The logs directory; harbor's adapters write the agent's under ``/logs/agent``.
LOGS = "/logs"
#: The preflight's configuration and working directories, apart from the build's transcript.
PREFLIGHT = "/logs/preflight"
#: The seal probe's inbound check's ready and done files (``hls.agent.probe.inbound_check``).
SEAL_PROBE = "/logs/seal-probe"
#: The task's files and skills.
TASK = "/task"
#: Where a task's skills go, one directory per skill.
SKILLS = "/task/skills"
#: Harbor's setup directory.
INSTALLED = "/installed-agent"
#: The directories harbor's environment may copy files into and out of, each a host directory.
TRANSFER_DIRS = (WORKSPACE, LOGS, TASK, INSTALLED)
#: The paths the brief tells the agent it may write: its working tree and its container's own
#: /tmp, /var/tmp and $HOME, which the workdir backs.
AGENT_WRITABLE = (WORKSPACE, "/tmp", "/var/tmp", "$HOME")
#: The executable budget MCP server's file name.
MCP_FILE = "get-budget-mcp"


@dataclass(frozen=True)
class SandboxLayout:
    """The sandbox role's node-local directories and files under *root*."""

    #: The sandbox role's directory.
    root: Path

    @property
    def workspace(self) -> Path:
        """The host side of ``/workspace``."""
        return self.root / "workspace"

    @property
    def logs(self) -> Path:
        """The host side of ``/logs``."""
        return self.root / "logs"

    @property
    def seal_probe(self) -> Path:
        """The host side of ``SEAL_PROBE``."""
        return self.logs / "seal-probe"

    @property
    def agent_logs(self) -> Path:
        """The host side of ``/logs/agent``, harbor's ``logs_dir``: the agent's stream, its session
        transcripts and the brief harbor persists."""
        return self.logs / "agent"

    @property
    def task(self) -> Path:
        """The host side of ``/task``."""
        return self.root / "task"

    @property
    def installed(self) -> Path:
        """The host side of ``/installed-agent``."""
        return self.root / "installed"

    @property
    def workdir(self) -> Path:
        """The container's workdir, behind its /tmp, /var/tmp and $HOME."""
        return self.root / "wd"

    @property
    def control(self) -> Path:
        """The exec server's control directory, 0700."""
        return self.root / "control"

    @property
    def ready(self) -> Path:
        """The exec server's ready file, which names the spawner it serves from
        (``exec_protocol.READY_NAME``)."""
        return self.control / exec_protocol.READY_NAME

    @property
    def channel(self) -> Path:
        """The budget service's channel directory, 0700."""
        return self.root / "channel"

    @property
    def exec_dir(self) -> Path:
        """The exec server, its spawner and their protocol."""
        return self.root / "exec"

    @property
    def programs(self) -> Path:
        """The commit program, which the commit and snapshot containers bind."""
        return self.root / "programs"

    @property
    def mcp_file(self) -> Path:
        """The executable budget MCP server."""
        return self.root / "mcp" / MCP_FILE

    @property
    def harbor(self) -> Path:
        """Harbor's own bookkeeping directory for the environment, never mounted."""
        return self.root / "harbor"

    @property
    def socket(self) -> Path:
        """The exec server's socket, as the runner reaches it."""
        return self.control / exec_protocol.SOCKET_NAME

    @property
    def budget_socket(self) -> Path:
        """The budget service's socket, as the runner binds it."""
        return self.channel / mcp.SOCKET_NAME

    def host_path(self, container_dir: str) -> Path:
        """The host directory behind one of ``TRANSFER_DIRS``."""
        hosts = {WORKSPACE: self.workspace, LOGS: self.logs, TASK: self.task, INSTALLED: self.installed}
        return hosts[container_dir]

    @classmethod
    def create(cls, root: Path) -> SandboxLayout:
        """Make the layout under *root*, which must hold none of it yet: fresh directories,
        the channel and control directories 0700, and the programs copied from this package."""
        layout = cls(Path(root))
        for d in (layout.workspace, layout.agent_logs, layout.logs / "preflight", layout.seal_probe,
                  layout.task / "files",
                  layout.task / "skills", layout.installed, layout.workdir, layout.exec_dir, layout.programs,
                  layout.mcp_file.parent, layout.harbor):
            d.mkdir(parents=True)
        for d in (layout.control, layout.channel):
            d.mkdir(mode=0o700)
            d.chmod(0o700)
        here = Path(exec_protocol.__file__).parent
        for name in exec_protocol.SANDBOX_FILES:
            shutil.copyfile(here / name, layout.exec_dir / name)
        shutil.copyfile(commit.__file__, layout.programs / Path(commit.COMMIT_TARGET).name)
        layout.mcp_file.write_text(mcp.executable_text(exec_protocol.SANDBOX_PYTHON))
        layout.mcp_file.chmod(0o755)
        return layout

    def mounts(self) -> tuple[Mount, ...]:
        """The sandbox container's own mounts; the data mounts come on top."""
        return (Mount(self.workspace, WORKSPACE, read_only=False), Mount(self.logs, LOGS, read_only=False),
                Mount(self.task, TASK), Mount(self.installed, INSTALLED, read_only=False),
                Mount(self.channel, mcp.CHANNEL_TARGET),
                Mount(self.control, exec_protocol.CONTROL_TARGET, read_only=False),
                Mount(self.exec_dir, exec_protocol.EXEC_TARGET), Mount(self.mcp_file, mcp.MCP_TARGET))

    def writable(self) -> tuple[Path, ...]:
        """The host directories the sandbox can write: the workdir and each writable mount's
        source. Its declared disk caps what they hold."""
        return (self.workdir, *(m.source for m in self.mounts() if not m.read_only))

    def harness_written(self) -> tuple[Path, ...]:
        """The host paths the harness writes into while the sandbox runs, which no writable
        directory of the sandbox may hold or lie in."""
        return (self.channel, self.exec_dir, self.mcp_file.parent, self.programs, self.harbor)

    def transfer_paths(self) -> dict[str, Path]:
        """Container directory to host directory, for harbor's environment's file transfers."""
        return {d: self.host_path(d) for d in TRANSFER_DIRS}

    def program_mounts(self) -> tuple[Mount, ...]:
        """The commit program, read-only, for a harness container."""
        return (Mount(self.programs / Path(commit.COMMIT_TARGET).name, commit.COMMIT_TARGET),)
