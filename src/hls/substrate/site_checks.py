"""What the runner checks of its node before any stage runs, beside the grant: that the node lays
out cgroups as every role step needs, and, for an allocation with a stage that seals a container
(``Stage.seals_containers``), that the network seal can run: unprivileged user namespaces, the
pasta program and the firewall tools. Each problem says what was found, what is required and what
setting or change of the site fixes it. A problem of the node is retried, with the node named so
the retry can go elsewhere, as a grant mismatch is; a wrong setting of the site is not."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from hls.substrate.apptainer.egress import SEAL_TOOL_PATH_VAR
from hls.substrate.cgroup import CGROUP_ROOT, job_dir, own_cgroup, step_dir

#: The directories the seal finds its tools in when the runner's environment names none
#: (``SEAL_TOOL_PATH_VAR``), as ``transparent_seal.sh`` searches them.
SEAL_TOOL_DIRS = "/usr/sbin:/sbin"
#: The tools the seal resolves there: the firewall's for each address family, and sysctl.
SEAL_TOOLS = ("iptables", "ip6tables", "sysctl")
#: Seconds the user namespace probe may take; it makes one namespace and exits.
USERNS_PROBE_SEC = 30
#: The probe: an unprivileged user namespace with a network namespace in it, as pasta makes. It
#: uses no capability inside them, since an AppArmor policy that withholds those does so per
#: program, and pasta's own profile may grant them.
USERNS_PROBE = "import os; os.unshare(os.CLONE_NEWUSER | os.CLONE_NEWNET)"
#: The kernel settings that turn unprivileged user namespaces off, with the value that does.
USERNS_SWITCHES = {"user/max_user_namespaces": "0", "kernel/unprivileged_userns_clone": "0"}
#: What fixes a node whose cgroups are not laid out as Slurm's cgroup v2 plugin lays them out.
_CGROUP_FIX = ("hls reads each job's and step's limits from cgroup v2 as Slurm's cgroup/v2 plugin lays them out, "
               "job_<id>/step_<n>/user: run the compute nodes on the unified cgroup v2 hierarchy, and set "
               "CgroupPlugin=cgroup/v2 in cgroup.conf with ProctrackType=proctrack/cgroup and "
               "TaskPlugin=task/cgroup in slurm.conf")


@dataclass(frozen=True)
class SiteProblem:
    """Why a node cannot run what an allocation asks of it."""

    #: What was found, what is required and what fixes it.
    reason: str
    #: Whether another attempt, on another node, may get past it: False for a setting of the site.
    retryable: bool = True


def cgroup_problem(own: Callable[[], str] = own_cgroup, root: Path = CGROUP_ROOT) -> SiteProblem | None:
    """Why this process's cgroups are not cgroup v2 laid out by Slurm's cgroup/v2 plugin, as the
    runner reads its job's limits and every role step its own (``hls.substrate.cgroup``), or None:
    no cgroup v2 entry (*own* raises), no cgroup v2 mount at *root*, a cgroup outside a
    ``job_<id>`` or ``step_<n>/user`` directory, or a job with no ``memory.max``."""
    try:
        cgroup = own()
    except (OSError, RuntimeError) as e:
        return SiteProblem(f"{e}. {_CGROUP_FIX}")
    if not (root / "cgroup.controllers").is_file():
        return SiteProblem(f"{root} is not a cgroup v2 mount: it has no cgroup.controllers. {_CGROUP_FIX}")
    for where, find in (("job_<id>", job_dir), ("step_<n>/user", step_dir)):
        try:
            find(cgroup, root)
        except RuntimeError:
            return SiteProblem(f"this process's cgroup {cgroup} lies in no {where} directory. {_CGROUP_FIX}")
    job = job_dir(cgroup, root)
    if not (job / "memory.max").is_file():
        return SiteProblem(f"the job's cgroup {job} has no memory.max, so its memory limit cannot be read: Slurm does "
                           "not enable the memory controller for jobs. Set ConstrainRAMSpace=yes in cgroup.conf")
    return None


def pasta_problem(pasta_path: Path | None, environ: Mapping[str, str]) -> SiteProblem | None:
    """Why the seal could not run pasta on this node, or None: *pasta_path* (``site.pasta_path``)
    is no executable file, or, unset, *environ*'s PATH, which every role step gets, holds none."""
    if pasta_path is not None:
        if pasta_path.is_file() and os.access(pasta_path, os.X_OK):
            return None
        found = "is not an executable file" if pasta_path.exists() else "does not exist"
        return SiteProblem(f"site.pasta_path {pasta_path} {found} on this node, and the network seal runs it. Build "
                           "pasta with hls build-pasta into a directory the compute nodes share and set "
                           "site.pasta_path to the binary", retryable=False)
    path = environ.get("PATH", "")
    if shutil.which("pasta", path=path) is not None:
        return None
    return SiteProblem(f"site.pasta_path is unset and this node's PATH ({path}) holds no pasta, which the network seal "
                       "runs. Install the passt package on the compute nodes, or build pasta with "
                       "hls build-pasta and set site.pasta_path to the binary")


def seal_tools_problem(environ: Mapping[str, str]) -> SiteProblem | None:
    """Why the seal would not find its tools (``SEAL_TOOLS``) where it looks for them, the
    directories *environ*'s ``SEAL_TOOL_PATH_VAR`` names or else ``SEAL_TOOL_DIRS``, or None."""
    dirs = environ.get(SEAL_TOOL_PATH_VAR) or SEAL_TOOL_DIRS
    missing = [tool for tool in SEAL_TOOLS
               if not any(os.access(Path(d) / tool, os.X_OK) for d in dirs.split(":") if d)]
    if not missing:
        return None
    return SiteProblem(f"the network seal runs {', '.join(SEAL_TOOLS)} from {dirs}, and {', '.join(missing)} "
                       f"{'is' if len(missing) == 1 else 'are'} not there on this node. Install them on the compute "
                       "nodes: the iptables package carries iptables and ip6tables, procps sysctl")


def userns_problem(run: Callable[..., subprocess.CompletedProcess] = subprocess.run,
                   proc_sys: Path = Path("/proc/sys")) -> SiteProblem | None:
    """Why this node does not give this user an unprivileged user namespace with a network
    namespace in it, as the seal's pasta and the build sandbox's spawner make, or None. The reason
    names each kernel setting that turns them off (``USERNS_SWITCHES``)."""
    try:
        done = run([sys.executable, "-I", "-c", USERNS_PROBE], capture_output=True, text=True,
                   timeout=USERNS_PROBE_SEC)
    except (OSError, subprocess.TimeoutExpired) as e:
        return SiteProblem(f"the user namespace probe could not run: {type(e).__name__}: {e}")
    if done.returncode == 0:
        return None
    error = next((line for line in reversed(done.stderr.splitlines()) if line.strip()), f"exit {done.returncode}")
    off = []
    for name, value in USERNS_SWITCHES.items():
        try:
            if (proc_sys / name).read_text().strip() == value:
                off.append(f"{name.replace('/', '.')} = {value}")
        except OSError:
            continue
    if not off:
        return SiteProblem(f"this node refused an unprivileged user namespace with a network namespace in it "
                           f"({error}), which the network seal and the build sandbox make. Allow this user "
                           "unprivileged user namespaces on the compute nodes")
    return SiteProblem(f"this node refuses unprivileged user namespaces ({error}; {', '.join(off)}), which the network "
                       "seal and the build sandbox make. Allow them on the compute nodes: user.max_user_namespaces "
                       "above 0, and kernel.unprivileged_userns_clone = 1 where the kernel has it")


def seal_problems(pasta_path: Path | None, environ: Mapping[str, str],
                  userns: Callable[[], SiteProblem | None] = userns_problem) -> list[SiteProblem]:
    """Every reason this node could not run the network seal: pasta (``pasta_problem``), the
    firewall tools (``seal_tools_problem``) and user namespaces (``userns_problem``)."""
    found = (pasta_problem(pasta_path, environ), seal_tools_problem(environ), userns())
    return [p for p in found if p is not None]
