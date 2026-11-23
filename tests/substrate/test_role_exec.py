from __future__ import annotations

import dataclasses
import fcntl
import os
import resource
import signal
import stat
import subprocess
import sys

import pytest

from hls.spec.resources import GrantedResources, ResourceShape
from hls.substrate import role_exec
from hls.substrate.base import RoleReport, parse_cpulist
from hls.substrate.factory import step
from hls.substrate.grant import Measured
from hls.substrate.nproc import uid_tasks
from hls.substrate.secrets import parse_env_text, write_env_file

ONE = ResourceShape(gpus=1, gpu_kind="H200", cpus=1, memory_gib=2, time_sec=1)
U1, U2 = "GPU-aaaaaaaa-0000-0000-0000-000000000001", "GPU-aaaaaaaa-0000-0000-0000-000000000002"


def _measured(gpus: int, uuids: list[str], *, cpus: int = 1, memory_gib: float = 2.0,
              problems: list[str] | None = None) -> Measured:
    granted = GrantedResources(gpus=gpus, gpu_kind="H200" if gpus else None, cpus=cpus, memory_gib=memory_gib,
                               disk_free_gib=100.0)
    return Measured(granted=granted, gpu_uuids=uuids, swap_max_bytes=None, problems=problems or [])


def test_an_exact_slice_passes():
    assert role_exec.check_role(ONE, _measured(1, [U1]), shared=None) == []


def test_a_measurement_problem_is_a_refusal_on_its_own():
    assert role_exec.check_role(ONE, _measured(1, [U1], problems=["x"]), shared=None) == ["x"]


def test_a_wider_slice_is_refused():
    problems = role_exec.check_role(ONE, _measured(2, [U1, U2], cpus=4), shared=None)
    assert problems == ["gpus: declared 1, granted 2", "cpus: declared 1, granted 4"]


def test_a_shared_step_counts_only_the_shared_gpus():
    # An overlap step is given every GPU of the allocation and narrowed to the shared ones.
    assert role_exec.check_role(ONE, _measured(2, [U1, U2]), shared=[U2]) == []
    problems = role_exec.check_role(ONE, _measured(2, [U1, U2]), shared=["GPU-elsewhere"])
    assert problems == ["shared GPUs not visible to this step: ['GPU-elsewhere']"]


def test_parse_takes_the_command_after_the_separator():
    args = role_exec._parse(["--report", "/r.json", "--gpus", "0", "--cpus", "1", "--memory-gib", "1",
                             "--disk-gib", "0", "--scratch", "/tmp", "--", "/bin/echo", "--gpus", "x"])
    assert args.command == ["/bin/echo", "--gpus", "x"] and args.gpu_kind is None


class _Execs(list):
    """The argv of each exec, and the program each one named."""

    def __init__(self) -> None:
        super().__init__()
        self.programs: list[str] = []


def _nodes(tmp_path, layout: dict[int, str]):
    """A fake ``/sys/devices/system/node``: one ``node<N>/cpulist`` per entry of *layout*."""
    root = tmp_path / "sys-node"
    for node, cpulist in layout.items():
        (root / f"node{node}").mkdir(parents=True)
        (root / f"node{node}" / "cpulist").write_text(cpulist + "\n")
    (root / "online").write_text("0-1\n")
    return root


@pytest.fixture
def fake_step(monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "own_cgroup", lambda: "/system.slice/slurmstepd.scope/job_5/step_2/user/task_0")
    here = dataclasses.replace(step({}), step_cgroup=lambda cg, root: tmp_path / "step_2" / "user")
    monkeypatch.setattr(role_exec, "step", lambda environ: here)
    # main() changes its own environment before it execs; keep that inside the test. step_env
    # builds a step's environment from Slurm's variables and the role's own, so it carries none
    # of the variables the wrapper removes, and neither does the copy, whatever an earlier test
    # left in this process: what a test sees removed is what it set.
    monkeypatch.setattr(role_exec.os, "environ",
                        {k: v for k, v in os.environ.items() if not role_exec.dropped(k, container=True)})
    monkeypatch.setenv("SLURM_JOB_ID", "5")
    monkeypatch.setenv("SLURM_STEP_ID", "2")
    # A score main() sets would stay on the test process, which could not lower it again.
    monkeypatch.setattr(role_exec, "OOM_SCORE_ADJ_PATH", tmp_path / "oom_score_adj")
    # The step's CPUs and the node's NUMA layout are the test's own: one CPU on a one-node
    # machine unless a test says otherwise.
    monkeypatch.setattr(role_exec, "NODE_ROOT", _nodes(tmp_path, {0: "0-127"}))
    monkeypatch.setattr(role_exec, "own_cpus", lambda: [0])
    execs = _Execs()

    def fake_execv(path, argv):
        execs.programs.append(path)
        execs.append(argv)

    monkeypatch.setattr(role_exec.os, "execv", fake_execv)
    return execs


@pytest.fixture(autouse=True)
def hang_guard():
    """Turns a test that would block forever on a pipe or a named pipe into a failure."""

    def on_alarm(signum, frame):
        raise TimeoutError("the test blocked for 10 seconds")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(10)
    yield
    signal.alarm(0)
    signal.signal(signal.SIGALRM, previous)


SLICE = ["--gpus", "1", "--gpu-kind", "H200", "--cpus", "1", "--memory-gib", "2", "--disk-gib", "0",
         "--scratch", "/tmp"]
ARGS = [*SLICE, "--", "/bin/true"]


def test_main_reports_then_becomes_the_command(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), *ARGS]) == 0
    report = RoleReport.model_validate_json(out.read_bytes())
    assert report.step == "5.2" and report.gpu_uuids == [U1] and report.mismatches == []
    assert report.cgroup_dir == str(tmp_path / "step_2" / "user")
    assert fake_step == [["/bin/true"]]


def test_the_wrapper_says_its_step_started_before_it_measures_the_grant(fake_step, monkeypatch, tmp_path):
    out = tmp_path / "gpu0.report.json"
    marker = tmp_path / "gpu0.started"
    seen = []

    def measure(**kw):
        seen.append(marker.read_text())
        return _measured(1, [U1])

    monkeypatch.setattr(role_exec, "measure", measure)
    assert role_exec.main(["--report", str(out), *ARGS]) == 0
    assert seen == ["5.2\n"] and role_exec.started_path(out) == marker


def test_the_env_file_is_taken_before_the_start_marker_is_written(fake_step, monkeypatch, tmp_path):
    # A marker that cannot be written fails the wrapper; the secrets file is gone by then.
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    env_file = tmp_path / "role.env"
    write_env_file(env_file, {"HF_TOKEN": "s3cret"})
    logs = tmp_path / "logs"
    logs.mkdir()
    logs.chmod(0o555)
    try:
        with pytest.raises(OSError):
            role_exec.main(["--report", str(logs / "gpu0.report.json"), "--env-file", str(env_file), *ARGS])
    finally:
        logs.chmod(0o755)
    assert not env_file.exists() and fake_step == []


def test_a_step_refused_for_its_process_ceiling_has_still_started(fake_step, monkeypatch, tmp_path):
    out = tmp_path / "gpu0.report.json"
    monkeypatch.setattr(role_exec, "uid_tasks", lambda uid: 5000)
    assert role_exec.main(["--report", str(out), *SLICE, "--nproc", "5000", "--", "/bin/true"]) == \
        role_exec.PROCESS_CEILING_EXIT
    assert (tmp_path / "gpu0.started").read_text() == "5.2\n" and not out.exists()


def test_main_refuses_a_mismatched_slice(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(2, [U1, U2]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), *ARGS]) == role_exec.GRANT_MISMATCH_EXIT
    assert RoleReport.model_validate_json(out.read_bytes()).mismatches == ["gpus: declared 1, granted 2"]
    assert fake_step == []


def test_main_narrows_a_shared_step_to_the_shared_uuids(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(2, [U1, U2]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    out = tmp_path / "overlap.report.json"
    assert role_exec.main(["--report", str(out), "--share-gpus", U2, *ARGS]) == 0
    assert RoleReport.model_validate_json(out.read_bytes()).cuda_visible_devices == U2
    assert role_exec.os.environ["CUDA_VISIBLE_DEVICES"] == U2
    assert role_exec.os.environ["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == U2


def _mismatches(path) -> list[str]:
    return RoleReport.model_validate_json(path.read_bytes()).mismatches


def test_a_contestant_role_gets_its_oom_score_adj_before_it_starts(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "srv.report.json"
    assert role_exec.main(["--report", str(out), *SLICE, "--oom-score-adj", "1000", "--", "/bin/true"]) == 0
    assert (tmp_path / "oom_score_adj").read_text() == "1000\n" and fake_step == [["/bin/true"]]
    (tmp_path / "oom_score_adj").unlink()
    assert role_exec.main(["--report", str(out), *ARGS]) == 0
    assert not (tmp_path / "oom_score_adj").exists()


def test_a_score_the_wrapper_cannot_set_refuses_the_role(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    monkeypatch.setattr(role_exec, "OOM_SCORE_ADJ_PATH", tmp_path / "missing" / "oom_score_adj")
    out = tmp_path / "srv.report.json"
    assert role_exec.main(["--report", str(out), *SLICE, "--oom-score-adj", "1000", "--", "/bin/true"]) == \
        role_exec.GRANT_MISMATCH_EXIT
    [why] = _mismatches(out)
    assert why.startswith("oom_score_adj 1000 could not be set: ") and fake_step == []


def test_the_score_is_the_process_own_and_what_it_execs_inherits_it():
    # The real file, in a child, since a raised score cannot be lowered without privilege.
    code = ("import os, sys; from hls.substrate.role_exec import CONTESTANT_OOM_SCORE_ADJ, set_oom_score_adj; "
            "assert set_oom_score_adj(CONTESTANT_OOM_SCORE_ADJ) is None; "
            "os.execv('/bin/cat', ['cat', '/proc/self/oom_score_adj'])")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert (result.returncode, result.stdout) == (0, "1000\n"), result.stderr


def test_a_role_gets_its_process_ceiling_before_it_starts(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    monkeypatch.setattr(role_exec, "uid_tasks", lambda uid: 300)
    ceilings: list[int] = []
    # The test process's own limit is left alone: a hard limit lowered cannot be raised again.
    monkeypatch.setattr(role_exec, "set_process_limit", lambda n: ceilings.append(n))
    out = tmp_path / "srv.report.json"
    assert role_exec.main(["--report", str(out), *SLICE, "--nproc", "5000", "--", "/bin/true"]) == 0
    assert ceilings == [5000] and fake_step == [["/bin/true"]]
    assert role_exec.main(["--report", str(out), *ARGS]) == 0
    assert ceilings == [5000]
    monkeypatch.setattr(role_exec, "set_process_limit", lambda n: "RLIMIT_NPROC 5000 could not be set: denied")
    assert role_exec.main(["--report", str(out), *SLICE, "--nproc", "5000", "--", "/bin/true"]) == \
        role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == ["RLIMIT_NPROC 5000 could not be set: denied"]


@pytest.mark.parametrize("held, refused", [(5000, True), (5001, True), (4999, False)])
def test_a_uid_already_at_the_ceiling_refuses_the_role_before_it_forks_anything(fake_step, monkeypatch, tmp_path,
                                                                                 capsys, held, refused):
    # The ceiling bounds every task of the uid on the node, so a step whose uid already holds as
    # many has no fork left: not even the grant's measurement, which runs nvidia-smi.
    monkeypatch.setattr(role_exec, "uid_tasks", lambda uid: held)
    measured: list[int] = []
    monkeypatch.setattr(role_exec, "measure", lambda **kw: measured.append(1) or _measured(1, [U1]))
    ceilings: list[int] = []
    monkeypatch.setattr(role_exec, "set_process_limit", lambda n: ceilings.append(n))
    out = tmp_path / "srv.report.json"
    code = role_exec.main(["--report", str(out), *SLICE, "--nproc", "5000", "--", "/bin/true"])
    if refused:
        assert code == role_exec.PROCESS_CEILING_EXIT and not out.exists()
        assert measured == [] and ceilings == [] and fake_step == []
        assert f"holds {held} tasks" in capsys.readouterr().err
    else:
        assert code == 0 and ceilings == [5000] and fake_step == [["/bin/true"]]


def test_a_count_of_the_uids_tasks_that_fails_starts_the_step_under_its_ceiling(fake_step, monkeypatch, tmp_path,
                                                                                 capsys):
    # The check only spares a start that could fork nothing; the step's end is judged by the pool
    # counted when it is reported. A count that fails, whatever it raises, never fails the launch.
    def boom(uid):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(role_exec, "uid_tasks", boom)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    ceilings: list[int] = []
    monkeypatch.setattr(role_exec, "set_process_limit", lambda n: ceilings.append(n))
    code = role_exec.main(["--report", str(tmp_path / "srv.report.json"), *SLICE, "--nproc", "5000", "--", "/bin/true"])
    assert code == 0 and ceilings == [5000] and fake_step == [["/bin/true"]]
    assert "could not be counted" in capsys.readouterr().err


@pytest.mark.skipif(os.getuid() == 0, reason="RLIMIT_NPROC does not bind root")
def test_a_wrapper_given_a_ceiling_its_uid_already_holds_exits_before_it_forks(tmp_path):
    held = uid_tasks(os.getuid())
    out = tmp_path / "srv.report.json"
    done = subprocess.run([sys.executable, "-m", "hls.substrate.role_exec", "--report", str(out), *SLICE,
                           "--nproc", str(held // 2), "--", "/bin/true"], capture_output=True, text=True, timeout=60)
    assert done.returncode == role_exec.PROCESS_CEILING_EXIT and not out.exists()
    assert "process ceiling" in done.stderr


def test_the_process_ceiling_is_the_process_own_and_what_it_execs_inherits_it():
    # A ceiling this host's hard limit allows, whatever that is.
    hard = resource.getrlimit(resource.RLIMIT_NPROC)[1]
    ceiling = 4321 if hard == resource.RLIM_INFINITY else min(4321, hard)
    code = ("import os, sys; from hls.substrate.role_exec import set_process_limit; "
            f"assert set_process_limit({ceiling}) is None; "
            "os.execv(sys.executable, [sys.executable, '-c', "
            "'import resource; print(resource.getrlimit(resource.RLIMIT_NPROC))'])")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert (result.returncode, result.stdout) == (0, f"({ceiling}, {ceiling})\n"), result.stderr


TWO_SOCKET_NODES = {0: "0-63", 1: "64-127"}
# Eight CPUs a job was given: four on each socket.
SPLIT_CPUS = [16, 17, 18, 19, 76, 77, 80, 81]
SLICE8 = ["--gpus", "1", "--gpu-kind", "H200", "--cpus", "8", "--memory-gib", "2", "--disk-gib", "0",
          "--scratch", "/tmp"]


@pytest.mark.parametrize("cpus, nodes, split", [
    (SPLIT_CPUS, [0, 1], True),
    ([0, 1, 2, 3, 124, 125, 126, 127], [0, 1], True),
    (list(range(24, 32)), [0], False),
    (list(range(64, 72)), [1], False),
    # As many CPUs as one node holds, over two nodes: split.
    (list(range(32, 96)), [0, 1], True),
    # A role that needs more CPUs than one node holds spans both by its own size.
    (list(range(128)), [0, 1], False),
    (list(range(60, 130)), [0, 1], False),
])
def test_a_placement_is_split_when_its_cpus_span_nodes_but_would_fit_on_one(tmp_path, cpus, nodes, split):
    found = role_exec.placement(cpus, node_root=_nodes(tmp_path, TWO_SOCKET_NODES))
    assert (found.cpus, found.numa_nodes, found.split) == (sorted(cpus), nodes, split)


def test_a_machine_with_one_node_never_has_a_split_placement(tmp_path):
    found = role_exec.placement([0, 5, 111], node_root=_nodes(tmp_path, {0: "0-111"}))
    assert (found.numa_nodes, found.split) == ([0], False)


def test_a_machine_that_lists_no_nodes_has_none_and_no_split(tmp_path):
    found = role_exec.placement([0, 1], node_root=tmp_path / "absent")
    assert (found.numa_nodes, found.split) == ([], False)


def test_a_node_whose_cpu_list_cannot_be_read_counts_as_no_node(tmp_path):
    root = _nodes(tmp_path, {0: "0-63"})
    (root / "node1").mkdir()
    (root / "node2").mkdir()
    (root / "node2" / "cpulist").write_text("garbage\n")
    found = role_exec.placement([0, 1, 64], node_root=root)
    assert (found.numa_nodes, found.split) == ([0], False)


def test_cpulists_are_read_as_the_kernel_writes_them():
    assert parse_cpulist("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
    assert parse_cpulist("\n") == set()


@pytest.mark.parametrize("cpus, nodes, split", [
    (SPLIT_CPUS, [0, 1], True),
    (list(range(64, 72)), [1], False),
])
def test_the_report_says_where_the_steps_cpus_lie(fake_step, monkeypatch, tmp_path, cpus, nodes, split):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1], cpus=8))
    monkeypatch.setattr(role_exec, "NODE_ROOT", _nodes(tmp_path / "alpha", TWO_SOCKET_NODES))
    monkeypatch.setattr(role_exec, "own_cpus", lambda: list(cpus))
    out = tmp_path / "srv.report.json"
    assert role_exec.main(["--report", str(out), *SLICE8, "--", "/bin/true"]) == 0
    report = RoleReport.model_validate_json(out.read_bytes())
    assert (report.cpus, report.numa_nodes, report.numa_split) == (cpus, nodes, split)
    # Where the CPUs lie is a record, never a refusal: the role starts either way.
    assert report.mismatches == [] and fake_step == [["/bin/true"]]


def test_a_role_held_to_one_node_is_refused_on_split_cpus_and_the_refusal_names_them(fake_step, monkeypatch,
                                                                                      tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1], cpus=8))
    monkeypatch.setattr(role_exec, "NODE_ROOT", _nodes(tmp_path / "alpha", TWO_SOCKET_NODES))
    monkeypatch.setattr(role_exec, "own_cpus", lambda: list(SPLIT_CPUS))
    out = tmp_path / "srv.report.json"
    code = role_exec.main(["--report", str(out), *SLICE8, "--one-numa-node", "--", "/bin/true"])
    assert code == role_exec.GRANT_MISMATCH_EXIT and fake_step == []
    assert _mismatches(out) == [f"cpus {SPLIT_CPUS} lie on NUMA nodes [0, 1], though one node holds as many; "
                                "this role runs on one node (site.evaluation_socket_tasks)"]


@pytest.mark.parametrize("flags", [["--one-numa-node"], []], ids=["held", "not-held"])
def test_a_role_on_one_node_starts_whether_or_not_it_is_held_to_one(fake_step, monkeypatch, tmp_path, flags):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1], cpus=8))
    monkeypatch.setattr(role_exec, "NODE_ROOT", _nodes(tmp_path / "alpha", TWO_SOCKET_NODES))
    monkeypatch.setattr(role_exec, "own_cpus", lambda: list(range(64, 72)))
    out = tmp_path / "srv.report.json"
    assert role_exec.main(["--report", str(out), *SLICE8, *flags, "--", "/bin/true"]) == 0
    assert fake_step == [["/bin/true"]]


def test_the_steps_own_cpus_are_its_affinity():
    assert role_exec.own_cpus() == sorted(os.sched_getaffinity(0))


def test_a_bare_command_is_found_on_the_steps_path(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    tool = tmp_path / "bin" / "tool"
    tool.parent.mkdir()
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setenv("PATH", str(tool.parent))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), *SLICE, "--", "tool", "--flag"]) == 0
    assert fake_step.programs == [str(tool)] and fake_step == [["tool", "--flag"]]


def _not_executable(tmp_path):
    plain = tmp_path / "plain"
    plain.write_text("x")
    plain.chmod(0o644)
    return str(plain)


@pytest.mark.parametrize("command", [
    lambda tmp_path: str(tmp_path / "missing"),
    lambda tmp_path: str(tmp_path),
    _not_executable,
    lambda tmp_path: "no-such-tool",
])
def test_a_command_that_cannot_run_is_refused_before_the_role_starts(fake_step, monkeypatch, tmp_path, command):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    out = tmp_path / "gpu0.report.json"
    cmd = command(tmp_path)
    assert role_exec.main(["--report", str(out), *SLICE, "--", cmd]) == role_exec.GRANT_MISMATCH_EXIT
    assert len(_mismatches(out)) == 1 and "not an executable file" in _mismatches(out)[0]
    assert fake_step == []


TOKEN = "hf_abcdefghijklmnop"


def _env_file(tmp_path):
    path = tmp_path / "roles" / "gpu0" / "secrets.env"
    path.parent.mkdir(parents=True)
    write_env_file(path, {"HF_TOKEN": TOKEN})
    return path


def test_an_env_file_is_gone_before_the_grant_is_measured_and_reaches_the_command(fake_step, monkeypatch,
                                                                                    tmp_path):
    env_file = _env_file(tmp_path)

    def measure(**kw):
        assert not env_file.exists()
        return _measured(1, [U1])

    monkeypatch.setattr(role_exec, "measure", measure)
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *ARGS]) == 0
    assert role_exec.os.environ["HF_TOKEN"] == TOKEN and fake_step == [["/bin/true"]]
    assert TOKEN not in out.read_text()


def test_an_env_file_is_gone_when_the_role_is_refused_and_reaches_nothing(fake_step, monkeypatch, tmp_path):
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(2, [U1, U2]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *ARGS]) == \
        role_exec.GRANT_MISMATCH_EXIT
    assert not env_file.exists() and role_exec.os.environ.get("HF_TOKEN") is None and fake_step == []


def test_a_missing_env_file_refuses_the_role(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    missing = tmp_path / "absent.env"
    assert role_exec.main(["--report", str(out), "--env-file", str(missing), *ARGS]) == \
        role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == [f"env file {missing} cannot be read: FileNotFoundError"]
    assert fake_step == []


@pytest.mark.skipif(os.getuid() == 0, reason="root reads a mode 0 file")
def test_an_unreadable_env_file_refuses_the_role_and_is_gone(fake_step, monkeypatch, tmp_path):
    env_file = _env_file(tmp_path)
    env_file.chmod(0)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *ARGS]) == \
        role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == [f"env file {env_file} cannot be read: PermissionError"]
    assert not env_file.exists() and fake_step == []


def _read_path(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def test_env_fd_hands_the_command_a_pipe_that_holds_the_file_once(fake_step, monkeypatch, tmp_path):
    # A free descriptor stands in for 3, which the test runner itself may be using.
    monkeypatch.setattr(role_exec, "ENV_FD", 250)
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    seen = {}

    def fake_execv(path, argv):
        # What apptainer does with --env-file /dev/fd/<n>: open the path and read it, once. The
        # starter and the container's init inherit the descriptor after that.
        seen["fifo"] = stat.S_ISFIFO(os.fstat(role_exec.ENV_FD).st_mode)
        seen["text"] = _read_path(f"/dev/fd/{role_exec.ENV_FD}").decode()
        seen["later_path"] = _read_path(f"/dev/fd/{role_exec.ENV_FD}")
        seen["later_inherited"] = os.read(role_exec.ENV_FD, 4096)
        seen["inheritable"] = os.get_inheritable(role_exec.ENV_FD)
        fake_step.append(argv)

    monkeypatch.setattr(role_exec.os, "execv", fake_execv)
    out = tmp_path / "gpu0.report.json"
    try:
        assert role_exec.main(["--report", str(out), "--env-file", str(env_file), "--env-fd", *ARGS]) == 0
    finally:
        os.close(250)
    assert not env_file.exists()
    assert parse_env_text(seen["text"]) == {"HF_TOKEN": TOKEN} and seen["inheritable"] and seen["fifo"]
    assert seen["later_path"] == b"" and seen["later_inherited"] == b""
    assert role_exec.os.environ.get("HF_TOKEN") is None
    assert role_exec.ENV_FD_PATH == "/dev/fd/3"


@pytest.mark.parametrize("names", [["TERM"], ["http_proxy"], ["HTTPS_PROXY"], ["ALL_PROXY", "TERM", "socks_proxy"]])
def test_a_container_step_loses_what_cleanenv_would_pass_and_the_report_names_it(fake_step, monkeypatch, tmp_path,
                                                                                  names):
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    for name in names:
        monkeypatch.setenv(name, "http://proxy.example:3128")
    seen = {}

    def fake_execv(path, argv):
        seen["env"] = dict(role_exec.os.environ)
        fake_step.append(argv)

    monkeypatch.setattr(role_exec.os, "execv", fake_execv)
    monkeypatch.setattr(role_exec, "ENV_FD", 250)
    out = tmp_path / "gpu0.report.json"
    try:
        assert role_exec.main(["--report", str(out), "--env-file", str(env_file), "--env-fd", *ARGS]) == 0
    finally:
        os.close(250)
    report = RoleReport.model_validate_json(out.read_bytes())
    assert report.mismatches == [] and report.stripped_env == sorted(names)
    assert fake_step == [["/bin/true"]] and not set(names) & set(seen["env"])


#: What a site's TaskProlog may export into every step after the launcher built its environment:
#: apptainer's own options, one of which adds binds past ``--no-mount bind-paths``, in both
#: spellings, and variables apptainer copies into the container past ``--cleanenv``.
TASK_PROLOG = {"APPTAINER_BIND": "/shared", "APPTAINER_CACHEDIR": "/shared/u/.apptainer",
               "SINGULARITY_BINDPATH": "/shared", "APPTAINERENV_X": "from-the-prolog",
               "SINGULARITYENV_Y": "from-the-prolog"}


def test_a_container_step_loses_apptainers_own_variables_and_then_gets_its_gpus(fake_step, monkeypatch, tmp_path):
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for name, value in TASK_PROLOG.items():
        monkeypatch.setenv(name, value)
    # A forwarded GPU list the site set gives way to the step's own.
    monkeypatch.setenv("APPTAINERENV_CUDA_VISIBLE_DEVICES", "0,1,2,3")
    seen = {}

    def fake_execv(path, argv):
        seen["env"] = dict(role_exec.os.environ)
        fake_step.append(argv)

    monkeypatch.setattr(role_exec.os, "execv", fake_execv)
    monkeypatch.setattr(role_exec, "ENV_FD", 250)
    out = tmp_path / "gpu0.report.json"
    try:
        assert role_exec.main(["--report", str(out), "--env-file", str(env_file), "--env-fd", *ARGS]) == 0
    finally:
        os.close(250)
    report = RoleReport.model_validate_json(out.read_bytes())
    assert report.mismatches == [] and report.stripped_env == sorted([*TASK_PROLOG, "APPTAINERENV_CUDA_VISIBLE_DEVICES"])
    assert fake_step == [["/bin/true"]] and not set(TASK_PROLOG) & set(seen["env"])
    assert seen["env"]["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "0"


def test_a_host_role_keeps_term_and_the_proxy_variables(fake_step, monkeypatch, tmp_path):
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    monkeypatch.setenv("TERM", "xterm")
    monkeypatch.setenv("https_proxy", "http://proxy.example:3128")
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *ARGS]) == 0
    report = RoleReport.model_validate_json(out.read_bytes())
    assert report.mismatches == [] and report.stripped_env == [] and fake_step == [["/bin/true"]]
    assert role_exec.os.environ["TERM"] == "xterm" and role_exec.os.environ["https_proxy"]


def test_parse_refuses_env_fd_without_an_env_file():
    with pytest.raises(SystemExit):
        role_exec._parse(["--report", "/r.json", "--env-fd", *ARGS])


def test_the_steps_gpus_are_forwarded_past_a_clean_container_environment(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), *ARGS]) == 0
    assert role_exec.os.environ["APPTAINERENV_CUDA_VISIBLE_DEVICES"] == "0"


def test_no_gpus_means_no_forwarded_gpus(fake_step, monkeypatch, tmp_path):
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(0, []))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setenv("APPTAINERENV_CUDA_VISIBLE_DEVICES", "stale")
    out = tmp_path / "cpu.report.json"
    cpu = ["--gpus", "0", "--cpus", "1", "--memory-gib", "2", "--disk-gib", "0", "--scratch", "/tmp", "--",
           "/bin/true"]
    assert role_exec.main(["--report", str(out), *cpu]) == 0
    assert role_exec.os.environ.get("APPTAINERENV_CUDA_VISIBLE_DEVICES") is None


def test_a_pipe_already_on_the_descriptor_stays_there_for_the_command(monkeypatch, tmp_path):
    # Once the wrapper has closed the file, the pipe's read end can land on descriptor 3 itself.
    path = tmp_path / "secrets.env"
    write_env_file(path, {"HF_TOKEN": TOKEN})
    fd, problem = role_exec.take_env_file(path)
    assert problem is None and stat.S_ISFIFO(os.fstat(fd).st_mode)
    monkeypatch.setattr(role_exec, "ENV_FD", fd)
    try:
        role_exec._hand_over(fd, on_fd=True)
        assert os.get_inheritable(fd) and parse_env_text(os.read(fd, 4096).decode()) == {"HF_TOKEN": TOKEN}
        assert os.read(fd, 4096) == b""
    finally:
        os.close(fd)


@pytest.mark.parametrize("flags", [[], ["--env-fd"]], ids=["environment", "descriptor"])
def test_an_env_file_of_exactly_the_pipe_capacity_is_handed_over_whole(fake_step, monkeypatch, tmp_path, flags):
    monkeypatch.setattr(role_exec, "ENV_FD", 250)
    env_file = tmp_path / "secrets.env"
    # One assignment, A='xxx...', of exactly 64 KiB with its newline.
    env_file.write_bytes(b"A='" + b"x" * (65536 - 5) + b"'\n")
    env_file.chmod(0o600)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    # A read may return less than was asked for. One bounded read would lose the rest.
    real_read = os.read
    monkeypatch.setattr(role_exec.os, "read", lambda fd, n: real_read(fd, min(n, 4096)))
    seen = {}

    def fake_execv(path, argv):
        if flags:
            seen["bytes"] = _read_path(f"/dev/fd/{role_exec.ENV_FD}")

    monkeypatch.setattr(role_exec.os, "execv", fake_execv)
    try:
        assert role_exec.main(["--report", str(tmp_path / "r.json"), "--env-file", str(env_file), *flags,
                               *ARGS]) == 0
    finally:
        if flags:
            os.close(250)
    assert role_exec.MAX_ENV_BYTES == 65536
    if flags:
        assert len(seen["bytes"]) == 65536 and parse_env_text(seen["bytes"].decode()) == {"A": "x" * (65536 - 5)}
    else:
        assert role_exec.os.environ["A"] == "x" * (65536 - 5)


@pytest.mark.parametrize("flags", [[], ["--env-fd"]], ids=["environment", "descriptor"])
@pytest.mark.parametrize("raw, suffix", [
    (b"A='x'\nnot an assignment\n", "is malformed: line 2 is not one NAME=value assignment"),
    (b"HF_TOKEN='fake-value-0123' B='y'\n", "is malformed: line 1 is not one NAME=value assignment"),
    (b"HF_TOKEN='fake-value-0123\n", "is malformed: line 1 is not one NAME=value assignment"),
    (b"HF_TOKEN='\xff\xfe'\n", "is malformed: not UTF-8 text"),
    (b"A='" + b"x" * 65532 + b"'\n", "is larger than 65536 bytes"),
])
def test_an_env_file_that_cannot_be_handed_over_refuses_the_role_in_both_modes(fake_step, monkeypatch, tmp_path,
                                                                                flags, raw, suffix):
    env_file = tmp_path / "secrets.env"
    env_file.write_bytes(raw)
    env_file.chmod(0o600)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *flags, *ARGS]) == \
        role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == [f"env file {env_file} {suffix}"]
    assert "fake-value" not in out.read_text()
    assert not env_file.exists() and role_exec.os.environ.get("HF_TOKEN") is None and fake_step == []


@pytest.mark.parametrize("flags", [[], ["--env-fd"]], ids=["environment", "descriptor"])
def test_a_pipe_too_small_for_the_file_refuses_the_role_instead_of_hanging(fake_step, monkeypatch, tmp_path, flags):
    # The kernel gives a user with too many pipes one page each; shrink this one the same way.
    real_pipe = os.pipe

    def small_pipe():
        read_end, write_end = real_pipe()
        fcntl.fcntl(write_end, fcntl.F_SETPIPE_SZ, 4096)
        return read_end, write_end

    monkeypatch.setattr(role_exec.os, "pipe", small_pipe)
    env_file = tmp_path / "secrets.env"
    env_file.write_bytes(b"A='" + b"x" * 8000 + b"'\n")
    env_file.chmod(0o600)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *flags, *ARGS]) == \
        role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == [f"env file {env_file} does not fit in a pipe"]
    assert not env_file.exists() and fake_step == []


def test_a_directory_at_the_env_file_path_refuses_the_role(fake_step, monkeypatch, tmp_path):
    env_dir = tmp_path / "secrets.env"
    env_dir.mkdir()
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_dir), *ARGS]) == role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == [f"env file {env_dir} is not a regular file"]
    assert env_dir.is_dir() and fake_step == []


def test_a_named_pipe_at_the_env_file_path_refuses_the_role_without_waiting(fake_step, monkeypatch, tmp_path):
    fifo = tmp_path / "secrets.env"
    os.mkfifo(fifo)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(fifo), *ARGS]) == role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == [f"env file {fifo} is not a regular file"]
    assert not fifo.exists() and fake_step == []


@pytest.mark.skipif(os.getuid() == 0, reason="root removes files from a read-only directory")
def test_an_env_file_that_cannot_be_removed_refuses_the_role(fake_step, monkeypatch, tmp_path):
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    env_file.parent.chmod(0o500)
    try:
        assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *ARGS]) == \
            role_exec.GRANT_MISMATCH_EXIT
    finally:
        env_file.parent.chmod(0o700)
    assert _mismatches(out) == [f"env file {env_file} cannot be removed: PermissionError"]
    assert role_exec.os.environ.get("HF_TOKEN") is None and fake_step == []


def test_an_env_file_that_is_a_link_is_refused_and_only_the_link_is_removed(fake_step, monkeypatch, tmp_path):
    target = tmp_path / "elsewhere.env"
    write_env_file(target, {"HF_TOKEN": TOKEN})
    link = tmp_path / "secrets.env"
    link.symlink_to(target)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(link), *ARGS]) == role_exec.GRANT_MISMATCH_EXIT
    assert _mismatches(out) == [f"env file {link} cannot be read: OSError"]
    assert not link.is_symlink() and target.exists()
    assert role_exec.os.environ.get("HF_TOKEN") is None and fake_step == []


@pytest.mark.parametrize("flags", [[], ["--env-fd"]], ids=["environment", "descriptor"])
def test_a_refused_role_closes_the_env_pipe(fake_step, monkeypatch, tmp_path, flags):
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(2, [U1, U2]))
    pipes = []
    real_pipe = os.pipe

    def recording_pipe():
        ends = real_pipe()
        pipes.append(ends)
        return ends

    monkeypatch.setattr(role_exec.os, "pipe", recording_pipe)
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *flags, *ARGS]) == \
        role_exec.GRANT_MISMATCH_EXIT
    assert len(pipes) == 1
    for end in pipes[0]:
        with pytest.raises(OSError):
            os.fstat(end)


# What a site's task prolog adds to every step.
SITE_ADDED = {"BASH_ENV": "/etc/site/bash_env", "APPTAINER_CACHEDIR": "/node/cache", "XDG_RUNTIME_DIR": "/run/user/1",
              "SRUN_CPUS_PER_TASK": "1", "APPTAINER_TMPDIR": "/node/tmp", "APPTAINER_BIND": "/shared",
              "SINGULARITY_BIND": "/shared", "APPTAINERENV_PLANTED": "1", "SINGULARITYENV_PLANTED": "1",
              "APPTAINERENV_CUDA_VISIBLE_DEVICES": "GPU-not-the-steps"}
# What the wrapper removes from every role of those. A container role's step also loses the
# cache and build directories, as it loses every variable apptainer reads.
DROPPED = ["APPTAINERENV_CUDA_VISIBLE_DEVICES", "APPTAINERENV_PLANTED", "APPTAINER_BIND", "BASH_ENV",
           "SINGULARITYENV_PLANTED", "SINGULARITY_BIND"]
KEPT = ["APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"]
LOCATIONS = ["XDG_RUNTIME_DIR", "SRUN_CPUS_PER_TASK"]


@pytest.mark.parametrize("flags", [[], ["--env-fd"]], ids=["host-role", "container-role"])
def test_what_a_site_prolog_adds_that_would_change_a_role_is_removed_and_named(fake_step, monkeypatch, tmp_path,
                                                                              flags):
    env_file = _env_file(tmp_path)
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    for name, value in SITE_ADDED.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", U1)
    seen = {}

    def fake_execv(path, argv):
        seen["env"] = dict(role_exec.os.environ)

    monkeypatch.setattr(role_exec.os, "execv", fake_execv)
    monkeypatch.setattr(role_exec, "ENV_FD", 250)
    out = tmp_path / "gpu0.report.json"
    try:
        assert role_exec.main(["--report", str(out), "--env-file", str(env_file), *flags, *ARGS]) == 0
    finally:
        if flags:
            os.close(250)
    report = RoleReport.model_validate_json(out.read_bytes())
    container = bool(flags)
    removed = sorted(DROPPED + KEPT) if container else DROPPED
    assert report.stripped_env == removed
    env = seen["env"]
    assert not set(removed) - {role_exec.CONTAINER_GPUS_VAR} & set(env)
    # What only says where things are stays, and the GPUs the container sees are the step's own.
    stays = LOCATIONS if container else KEPT + LOCATIONS
    assert {k: env[k] for k in stays} == {k: SITE_ADDED[k] for k in stays}
    assert env[role_exec.CONTAINER_GPUS_VAR] == U1


def test_a_host_role_without_an_env_file_loses_what_a_site_prolog_adds_too(fake_step, monkeypatch, tmp_path):
    # Most harness host roles take no env file; the site's prolog reaches their steps all the same.
    monkeypatch.setattr(role_exec, "measure", lambda **kw: _measured(1, [U1]))
    for name, value in SITE_ADDED.items():
        monkeypatch.setenv(name, value)
    out = tmp_path / "gpu0.report.json"
    assert role_exec.main(["--report", str(out), *ARGS]) == 0
    assert RoleReport.model_validate_json(out.read_bytes()).stripped_env == DROPPED
    assert not set(DROPPED) - {role_exec.CONTAINER_GPUS_VAR} & set(role_exec.os.environ)
    assert fake_step == [["/bin/true"]]


@pytest.mark.parametrize("name, container, gone", [
    ("BASH_ENV", False, True), ("BASH_ENV", True, True),
    ("APPTAINER_BIND", False, True), ("SINGULARITY_NV", True, True),
    ("APPTAINERENV_X", True, True), ("SINGULARITYENV_X", False, True),
    ("APPTAINER_CACHEDIR", False, False), ("APPTAINER_TMPDIR", False, False),
    ("APPTAINER_CACHEDIR", True, True), ("APPTAINER_TMPDIR", True, True), ("SINGULARITY_CACHEDIR", False, True),
    ("TERM", False, False), ("TERM", True, True), ("https_proxy", True, True), ("https_proxy", False, False),
    ("HLS_UPSTREAM_PROXY_URL", True, False), ("XDG_RUNTIME_DIR", True, False), ("APPTAINER", True, False),
])
def test_which_variables_the_wrapper_removes(name, container, gone):
    assert role_exec.dropped(name, container=container) is gone
