"""The runner's checks of its node: each refusal says what was found, what is required and what
fixes it, and a node set up as the substrate needs passes."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from hls.substrate.cgroup import own_cgroup
from hls.substrate.site_checks import (SEAL_TOOLS, SiteProblem, cgroup_problem, pasta_problem, seal_problems,
                                       seal_tools_problem, userns_problem)

BATCH = "/system.slice/slurmstepd.scope/job_100/step_batch/user/task_0"


def _node(tmp_path: Path, cgroup_lines: str, *, v2: bool = True,
          memory: bool = True) -> tuple[Callable[[], str], Path]:
    """This process's cgroup lookup reading *cgroup_lines*, and a cgroup root with the job's
    directory, as cgroup v2 when *v2*, its memory controller on when *memory*."""
    tmp_path.mkdir(exist_ok=True)
    proc = tmp_path / "cgroup"
    proc.write_text(cgroup_lines)
    root = tmp_path / "sys-fs-cgroup"
    root.mkdir()
    if v2:
        (root / "cgroup.controllers").write_text("cpuset cpu io memory pids\n")
    job = root / "system.slice/slurmstepd.scope/job_100"
    job.mkdir(parents=True)
    if memory:
        (job / "memory.max").write_text("85899345920\n")
    return lambda: own_cgroup(proc), root


def test_slurms_cgroup_v2_layout_passes(tmp_path):
    assert cgroup_problem(*_node(tmp_path, f"0::{BATCH}\n")) is None


def test_a_cgroup_v1_node_is_refused_naming_what_it_found_and_the_plugin_to_set(tmp_path):
    problem = cgroup_problem(*_node(tmp_path, "12:memory:/slurm/uid_1/job_100/step_batch\n11:cpuset:/slurm\n",
                                    v2=False))
    # A node's set-up: retried, the node named, so the retry can go elsewhere.
    assert problem is not None and problem.retryable
    assert "has no cgroup v2 entry ('0::'), only cgroup v1 entries such as '12:memory:/slurm/uid_1/job_100/step_batch'" \
        in problem.reason
    assert "CgroupPlugin=cgroup/v2" in problem.reason and "job_<id>/step_<n>/user" in problem.reason


def test_a_hybrid_node_whose_unified_mount_is_elsewhere_is_refused(tmp_path):
    problem = cgroup_problem(*_node(tmp_path, f"4:memory:/slurm/job_100\n0::{BATCH}\n", v2=False))
    assert problem is not None and "is not a cgroup v2 mount" in problem.reason


def test_a_cgroup_outside_slurms_job_and_step_directories_is_refused(tmp_path):
    outside = cgroup_problem(*_node(tmp_path / "a", "0::/user.slice/user-1000.slice/session-3.scope\n"))
    assert outside is not None and "lies in no job_<id> directory" in outside.reason
    no_step = cgroup_problem(*_node(tmp_path / "b", "0::/system.slice/slurmstepd.scope/job_100/task_0\n"))
    assert no_step is not None and "lies in no step_<n>/user directory" in no_step.reason


def test_a_job_without_the_memory_controller_is_refused_naming_the_setting(tmp_path):
    problem = cgroup_problem(*_node(tmp_path, f"0::{BATCH}\n", memory=False))
    assert problem is not None and "ConstrainRAMSpace=yes" in problem.reason


def _executable(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_pasta_is_found_at_the_sites_path_or_on_path(tmp_path):
    built = _executable(tmp_path / "pasta-build" / "pasta")
    assert pasta_problem(built, {"PATH": "/nowhere"}) is None
    assert pasta_problem(None, {"PATH": f"/nowhere:{built.parent}"}) is None


def test_a_missing_pasta_names_the_setting_and_how_to_build_one(tmp_path):
    missing = pasta_problem(tmp_path / "pasta", {"PATH": "/usr/bin"})
    assert missing is not None and not missing.retryable
    assert missing.reason.startswith(f"site.pasta_path {tmp_path / 'pasta'} does not exist on this node")
    assert "hls build-pasta" in missing.reason
    (tmp_path / "plain").write_text("")
    assert "is not an executable file" in pasta_problem(tmp_path / "plain", {}).reason
    unset = pasta_problem(None, {"PATH": str(tmp_path / "empty")})
    assert unset is not None and unset.reason.startswith("site.pasta_path is unset and this node's PATH")
    assert "passt package" in unset.reason


def test_the_seal_tools_are_looked_for_where_the_seal_looks(tmp_path):
    for tool in SEAL_TOOLS:
        _executable(tmp_path / "sbin" / tool)
    assert seal_tools_problem({"HLS_SEAL_TOOL_PATH": str(tmp_path / "sbin")}) is None
    os.remove(tmp_path / "sbin" / "ip6tables")
    problem = seal_tools_problem({"HLS_SEAL_TOOL_PATH": f"{tmp_path / 'none'}:{tmp_path / 'sbin'}"})
    assert problem is not None and problem.retryable
    assert "ip6tables is not there on this node" in problem.reason and "iptables package" in problem.reason


def _ran(code: int, stderr: str = ""):
    return lambda argv, **kw: subprocess.CompletedProcess(argv, code, "", stderr)


def test_user_namespaces_the_kernel_turns_off_are_refused_naming_the_switch(tmp_path):
    proc_sys = tmp_path / "sys"
    (proc_sys / "user").mkdir(parents=True)
    (proc_sys / "user" / "max_user_namespaces").write_text("0\n")
    assert userns_problem(_ran(0), proc_sys) is None
    problem = userns_problem(_ran(1, "PermissionError: [Errno 28] No space left on device\n"), proc_sys)
    assert problem is not None and problem.reason.startswith("this node refuses unprivileged user namespaces")
    assert "user.max_user_namespaces = 0" in problem.reason and "No space left on device" in problem.reason


def test_a_refused_namespace_no_switch_explains_or_a_probe_that_did_not_run_is_still_refused(tmp_path):
    problem = userns_problem(_ran(1, "OSError: [Errno 12] Cannot allocate memory\n"), tmp_path)
    assert problem is not None and problem.retryable and "Cannot allocate memory" in problem.reason

    def hung(argv, **kw):
        raise subprocess.TimeoutExpired(argv, 30)

    assert userns_problem(hung, tmp_path).retryable


def test_this_hosts_probe_runs():
    # The real probe on this host: it passes, or it is refused with a reason.
    problem = userns_problem()
    assert problem is None or problem.reason


def test_seal_problems_gathers_each_one(tmp_path):
    found = seal_problems(None, {"PATH": str(tmp_path), "HLS_SEAL_TOOL_PATH": str(tmp_path)},
                          userns=lambda: SiteProblem("no namespaces"))
    assert [p.reason.split(" ", 1)[0] for p in found] == ["site.pasta_path", "the", "no"]
