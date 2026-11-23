"""The build sandbox's probe: each check's parser against synthesized command output, the command
builders, the host-side checks against a synthesized /proc, and the whole set of in-sandbox
checks run through the real exec server: without a namespace, where the checks that need one
report not ok and the rest pass, and in real namespaces where the machine allows them."""

from __future__ import annotations

import asyncio
import builtins
import contextlib
import errno
import http.client
import io
import json
import os
import resource
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from hls.agent import exec_protocol as protocol
from hls.agent import probe
from hls.agent.memory import ReadyPids, SplitSample
from hls.egress import seal_probe
from tests.agent.probe_answers import NAMESPACE_CHECKS
from tests.agent.test_sandbox_attack import require_real_namespaces, require_user_namespaces

AGENT = Path(protocol.__file__).parent
TOKEN = "tok-0123456789abcdef"
ROUTE_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_MAX_OUTPUT_TOKENS")
#: A process ceiling this machine's own hard limit allows the spawner to give each command.
_HARD = resource.getrlimit(resource.RLIMIT_NPROC)[1]
NPROC = 4096 if _HARD == resource.RLIM_INFINITY else min(4096, _HARD)
#: The probe's PID 1 checks lower PID 1's limits. Without a namespace PID 1 is the machine's own
#: init, which an unprivileged user cannot touch; root could.
NOT_ROOT = pytest.mark.skipif(os.getuid() == 0, reason="the PID 1 checks would reach the machine's init as root")


@dataclass
class R:
    """What ``SandboxEnvironment.exec`` returns, for the parsers."""

    stdout: str | None = None
    return_code: int = 0


#: The limits a sandbox's commands' user namespace gets in these tests.
UCOUNTS = {"max_user_namespaces": 32, "max_inotify_instances": 6}


def _inputs(**changes) -> probe.ProbeInputs:
    base = dict(uid=1000, gid=1000, shm_bytes=8 * 2**30, gpus=1, control_target="/run/hls-control",
                socket_target="/run/hls-control/exec.sock", mcp_target="/opt/hls-budget/get-budget-mcp",
                route_host="api.anthropic.com", refused_host="hls-seal-probe.invalid", dial="192.0.2.1:80",
                route_vars=ROUTE_VARS, command_nproc=8192, nproc_accounting="uid", ucounts=UCOUNTS,
                python=protocol.SANDBOX_PYTHON)
    return probe.ProbeInputs(**{**base, **changes})


def _check(name: str, *results: R, inputs: probe.ProbeInputs | None = None) -> probe.ProbeCheck:
    [c] = [c for c in (*probe.SANDBOX_CHECKS, probe.ACCOUNTING_CHECK) if c.name == name]
    return c.parse(list(results), inputs or _inputs())


# ---------------------------------------------------------------------------
# The parsers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("out, ok", [("3.12.1", True), ("3.14.0", True), ("3.11.9", False), ("junk", False)])
def test_python_version(out, ok):
    assert _check("python-version", R(out)).ok is ok


@pytest.mark.parametrize("out, ok", [("1000\n1000", True), ("0\n0", False), ("1000\n1001", False)])
def test_uid_gid(out, ok):
    assert _check("uid-gid", R(out)).ok is ok


@pytest.mark.parametrize("out, ok", [("0000000000000000", True), ("000001ffffffffff", False), ("", False)])
def test_cap_eff(out, ok):
    assert _check("cap-eff", R(out)).ok is ok


@pytest.mark.parametrize("out, rc, ok", [
    ("1 /usr/bin/python3 -I /opt/hls-exec/spawner.py --fd 4\n7 bash -c x", 0, True),  # PID 1 is the reaper
    ("1 /usr/bin/python3 -I /opt/hls-exec/spawner.py --fd 4\n5 /usr/bin/python3 -I /opt/hls-exec/spawner.py", 0,
     False),
    ("1 init\n12 /usr/bin/python3 -I /opt/hls-exec/exec_server.py --socket x", 0, False),
    ("", 0, False), ("1 init", 1, False),  # the processes were never listed
])
def test_ps_isolation(out, rc, ok):
    assert _check("ps-isolation", R(out, rc)).ok is ok


@pytest.mark.parametrize("out, ok", [("1000", True), ("0", False), ("500", False)])
def test_oom_score_adj(out, ok):
    assert _check("oom-score-adj", R(out)).ok is ok


def test_the_oom_score_the_probe_expects_is_the_step_wrappers(monkeypatch):
    """The probe checks the score the step wrapper gives a contestant step's processes, read from
    the wrapper's own constant, and the in-sandbox protocol's copy, which may import nothing of the
    harness, is the same figure."""
    from hls.agent import exec_protocol
    from hls.substrate import role_exec

    assert exec_protocol.OOM_SCORE_ADJ == role_exec.CONTESTANT_OOM_SCORE_ADJ
    monkeypatch.setattr(role_exec, "CONTESTANT_OOM_SCORE_ADJ", 900)
    check = _check("oom-score-adj", R("900"))
    assert check.ok and check.expected == "900"


@pytest.mark.parametrize("out, ok", [
    ("mem=refused\nenviron=refused", True), ("mem=read\nenviron=refused", False),
    ("mem=refused\nenviron=read", False), ("", False),
    ("mem=errno:2\nenviron=errno:2", False),  # ENOENT: PID 1 absent proves nothing
])
def test_pid1_is_not_dumpable(out, ok):
    assert _check("pid1-non-dumpable", R(out)).ok is ok


def test_a_route_variable_in_a_commands_environment_is_named_never_its_value():
    clean = _check("route-env-absent", R("HOME\nPATH\nTORCH_CUDA_ARCH_LIST"))
    assert (clean.ok, clean.observed) == (True, "none")
    leaked = _check("route-env-absent", R("ANTHROPIC_BASE_URL\nANTHROPIC_API_KEY\nPATH"))
    assert (leaked.ok, leaked.observed) == (False, "ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL")
    assert _check("route-env-absent", R("PATH", 1)).ok is False  # the names were never read


@pytest.mark.parametrize("out, ok, observed", [
    ("unreadable", True, "unreadable"), ("readable\nHOME\nPATH", True, "readable, none"),
    ("readable\nANTHROPIC_API_KEY\nPATH", False, "ANTHROPIC_API_KEY"), ("", False, "no answer"),
    ("errno:2", False, "no answer"),  # ENOENT: not a permission refusal, so not "unreadable"
])
def test_pid1s_environment_is_unreadable_or_holds_no_route_variable(out, ok, observed):
    check = _check("pid1-route-env", R(out))
    assert (check.ok, check.observed) == (ok, observed)


def _run_c_code(command: str, open_error: int | None) -> str:
    """Run the ``python3 -I -c <code>`` the probe command would, in-process: exec the code with
    ``builtins.open`` raising ``OSError(open_error)`` (None leaves it), capturing what it prints.
    The code reads only ``/proc/1/...``, so a patched ``open`` covers every file it touches."""
    parts = shlex.split(command)
    code = parts[parts.index("-c") + 1]
    real_open = builtins.open
    out = io.StringIO()

    def fake_open(*a, **k):
        if open_error is not None:
            raise OSError(open_error, os.strerror(open_error))
        return real_open(*a, **k)

    with contextlib.redirect_stdout(out):
        exec(compile(code, "<probe>", "exec"), {"open": fake_open, "__builtins__": builtins})
    return out.getvalue()


def test_the_recovery_check_knows_a_fresh_pid1_by_its_start_time_not_its_namespaces_inode():
    # The kernel reuses a freed namespace's inode number, so a fresh namespace may carry the
    # dead one's: PID 1's start time tells them apart.
    reused = _check("pid1-recovery", R("pid:[4026535418] 232720891"), R("", -9), R("pid:[4026535418] 232722130"))
    assert reused.ok, reused
    survived = _check("pid1-recovery", R("pid:[4026535418] 232720891"), R("survived"),
                      R("pid:[4026535418] 232720891"))
    assert not survived.ok and "survived" in survived.observed
    [check] = [c for c in probe.SANDBOX_CHECKS if c.name == "pid1-recovery"]
    seen, _, again = check.commands(_inputs())
    assert "/proc/1/stat" in seen and seen == again


@pytest.mark.parametrize("name", ["pid1-non-dumpable", "pid1-route-env"])
@pytest.mark.parametrize("err, ok", [(errno.EACCES, True), (errno.EPERM, True), (errno.ENOENT, False),
                                     (errno.ESRCH, False)])
def test_the_pid1_commands_pass_only_on_a_permission_refusal(name, err, ok):
    """The pid1-non-dumpable and pid1-route-env commands read PID 1's files, which the kernel
    refuses with EACCES/EPERM when it is not dumpable or its environment is sealed. A file that is
    simply not there (ENOENT), or a process that is gone (ESRCH), proves nothing, so the check
    must not pass on it. The command's own ``-c`` code is run with ``open`` raising each
    errno, and its output fed to the parser."""
    [check] = [c for c in probe.SANDBOX_CHECKS if c.name == name]
    output = _run_c_code(check.commands(_inputs())[0], err)
    assert check.parse([R(output.strip())], _inputs()).ok is ok
    assert _check("command-nproc", R("8192\n8192")).ok is True
    assert _check("command-nproc", R("8192\n65536")).ok is False  # the hard limit not lowered
    assert _check("command-nproc", R("unlimited\nunlimited")).ok is False
    # Without a ceiling the commands keep the spawner's own limit, recorded.
    free = _check("command-nproc", R("4096\n65536"), inputs=_inputs(command_nproc=None))
    assert (free.ok, free.observed) == (True, "4096 65536")


def test_dev_shm_is_the_spawners_capped_tmpfs_and_takes_a_write():
    """The cap is the sandbox's whole memory, so the probe never writes up to it: the
    write would end the sandbox for its memory first. It reads the tmpfs's size, which the
    kernel enforces, and writes one page."""
    cap = _inputs().shm_bytes
    assert _check("dev-shm", R(f"size={cap}\nwrite=ok")).ok is True
    assert _check("dev-shm", R(f"size={cap}\nwrite=failed")).ok is False  # not writable
    assert _check("dev-shm", R("size=137438953472\nwrite=ok")).ok is False  # the host's /dev/shm
    [check] = [c for c in probe.SANDBOX_CHECKS if c.name == "dev-shm"]
    [command] = check.commands(_inputs())
    assert "bs=4096 count=1" in command and str(cap) not in command


DENIED = "ready read=denied:13 write=denied:13 create=denied:13"


@pytest.mark.parametrize("out, ok, observed", [
    (f"__listed__\n{DENIED}", True, "empty"),  # covered: an empty tmpfs over the control dir
    (f"unreadable\n{DENIED}", True, "unreadable (mode 000)"),  # covered: the tmpfs is mode 000, EACCES
    (f"exec.sock\n__listed__\n{DENIED}", False, "exec.sock"),  # the socket is visible: not covered
    (f"exec.sock\nother\n__listed__\n{DENIED}", False, "exec.sock,other"),
    (f"errno:2\n{DENIED}", False, "errno:2"),  # ENOENT: the directory is not there, not covered
    ("", False, "no answer; no ready-file answer"),
    # The ready file names the reaper the memory split trusts: a command must not read it, open it
    # for writing, or create a file beside it (a rename over it would need that).
    ("unreadable\nready read=opened write=denied:13 create=denied:13", False,
     "unreadable (mode 000); ready read=opened write=denied:13 create=denied:13"),
    ("unreadable\nready read=denied:13 write=opened create=denied:13", False,
     "unreadable (mode 000); ready read=denied:13 write=opened create=denied:13"),
    ("unreadable\nready read=denied:2 write=denied:30 create=opened", False,
     "unreadable (mode 000); ready read=denied:2 write=denied:30 create=opened"),
    ("unreadable", False, "unreadable (mode 000); no ready-file answer"),
])
def test_control_covered(out, ok, observed):
    check = _check("control-covered", R(out))
    assert (check.ok, check.observed) == (ok, observed)


@pytest.mark.parametrize("out, ok", [("refused", True), ("reached", False)])
def test_control_socket(out, ok):
    assert _check("control-socket", R(out)).ok is ok


def test_gpu_count():
    assert _check("gpu-count", R("GPU 0: H200\nGPU 1: H200"), inputs=_inputs(gpus=2)).ok is True
    assert _check("gpu-count", R("GPU 0: H200"), inputs=_inputs(gpus=2)).ok is False
    assert _check("gpu-count", R(""), inputs=_inputs(gpus=0)).ok is True


def test_budget_tool():
    good = json.dumps({"result": {"content": [{"text": json.dumps({"dimensions": {"wallclock": {}}})}]}})
    assert _check("budget-tool", R(good)).ok is True
    assert _check("budget-tool", R('{"error": "unavailable"}')).ok is False


def test_a_limit_set_on_pid1_must_not_reach_a_later_command():
    assert _check("limits-not-inherited", R("1024\n4096"), R("set"), R("1024\n4096")).ok is True
    assert _check("limits-not-inherited", R("1024\n4096"), R("set"), R("4\n4")).ok is False  # inherited
    # A limit PID 1 refused proves nothing about inheritance.
    refused = _check("limits-not-inherited", R("1024\n4096"), R("refused PermissionError"), R("1024\n4096"))
    assert refused.ok is False and "refused PermissionError" in refused.observed


def test_pid1_ended_by_its_cpu_limit_is_replaced_by_a_fresh_namespace():
    old, fresh = R("pid:[4026532001]"), R("pid:[4026532077]")
    assert _check("pid1-recovery", old, R(None, -9), fresh).ok is True
    assert _check("pid1-recovery", old, R("survived"), R("pid:[4026532001]")).ok is False  # the same namespace
    assert _check("pid1-recovery", old, R("refused", 1), R("pid:[4026532001]")).ok is False
    assert _check("pid1-recovery", old, R(None, -9), R(None, 126)).ok is False  # the next command never ran


def test_the_sealed_route_checks():
    assert _check("route-reachable", R("ok TLSv1.3", 0)).ok is True
    assert _check("route-reachable", R("fail ConnectionRefusedError", 1)).ok is False
    assert _check("route-reachable", R("fail SSLError", 0)).ok is False  # exit 0 but no handshake
    assert _check("unlisted-refused", R("refused")).ok is True
    for seen in ("answered", "unreached ConnectionRefusedError", "timeout", ""):
        assert _check("unlisted-refused", R(seen)).ok is False, seen
    nothing = _check("unlisted-refused", inputs=_inputs(refused_host=None))  # the policy refuses no host
    assert (nothing.ok, nothing.observed) == (True, "the policy refuses no host")
    assert _check("index-reachable", R("", 0)).ok is True
    assert _check("index-reachable", R("", 1)).ok is False


def test_the_accounting_check_reads_the_forks_its_command_made_against_the_sites_accounting():
    space = _inputs(nproc_accounting="user_namespace")
    assert _check("nproc-accounting", R("forks=0")).ok is True
    measured = _check("nproc-accounting", R("forks=511"), inputs=space)
    assert (measured.ok, measured.observed, measured.expected) == (
        True, "forks=511 held=600 limit=512 accounting=user_namespace", "accounting=user_namespace")
    # The site and the kernel disagree, either way.
    assert _check("nproc-accounting", R("forks=511")).ok is False
    assert _check("nproc-accounting", R("forks=0"), inputs=space).ok is False
    unclear = _check("nproc-accounting", R("forks=100"), inputs=space)
    assert unclear.ok is False and unclear.observed.endswith("accounting=unclear")
    for seen in ("error=PermissionError:[Errno 1] Operation not permitted", "error=no user namespace was made", "",
                 "forks=x"):
        assert _check("nproc-accounting", R(seen)).ok is False, seen


@pytest.mark.parametrize("seen, reading", [
    ("forks=0", "uid"), ("forks=256", "user_namespace"), ("forks=511", "user_namespace"),
    # Between them the reading is unclear; an error, a fork refused otherwise than EAGAIN among
    # them, is no reading at all.
    ("forks=1", None), ("forks=255", None), ("error=OSError:[Errno 12] Cannot allocate memory", None),
    ("error=no user namespace was made", None), ("", None), ("forks=x", None)])
def test_only_no_fork_or_half_the_limit_reads_clearly_as_one_accounting(seen, reading):
    assert probe.read_accounting(R(seen)) == reading
    check = _check("nproc-accounting", R(seen))
    assert check.observed.endswith(f"accounting={reading or 'unclear'}")


def test_the_accounting_command_measures_this_hosts_kernel():
    # Run as a sandbox command would run it, on this host: a determinate answer, whichever the
    # kernel's.
    require_user_namespaces()
    check = probe.ACCOUNTING_CHECK
    [command] = check.commands(_inputs(python=sys.executable))
    done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
    seen = done.stdout.strip()
    observed = check.parse([R(seen, done.returncode)], _inputs()).observed
    assert done.returncode == 0 and seen.startswith("forks="), (seen, done.stderr)
    assert observed.endswith(("accounting=uid", "accounting=user_namespace")), observed


def test_the_ucount_check_passes_only_when_the_commands_namespace_has_every_limit_given():
    seen = "max_cgroup_namespaces 2147483647\nmax_inotify_instances 6\nmax_user_namespaces 32\n"
    check = _check("ucount-ceilings", R(seen))
    assert (check.ok, check.observed, check.expected) == (
        True, "max_inotify_instances=6 max_user_namespaces=32", "max_inotify_instances=6 max_user_namespaces=32")
    # The host's own values, as a command outside the namespace reads them, or a limit missing.
    assert _check("ucount-ceilings", R("max_inotify_instances 128\nmax_user_namespaces 4124579\n")).ok is False
    missing = _check("ucount-ceilings", R("max_user_namespaces 32\n"))
    assert missing.ok is False and "max_inotify_instances=absent" in missing.observed
    assert _check("ucount-ceilings", R(""), inputs=_inputs(ucounts={})).ok is False


@pytest.mark.parametrize("seen, ok", [("EACCES", True), ("EPERM", True), ("written", False), ("", False),
                                      ("EROFS", False)])
def test_a_command_that_raised_its_own_limits_fails_the_lock_check(seen, ok):
    assert _check("ucount-locked", R(seen)).ok is ok


def test_the_command_builders_name_the_targets_and_leave_nothing_behind():
    inputs = _inputs(python="/opt/the sandbox's/python3")
    cmds = {c.name: c.commands(inputs) for c in (*probe.SANDBOX_CHECKS, probe.ACCOUNTING_CHECK)}
    # Every interpreter command runs the sandbox's own interpreter, never a path of its own.
    for name in ("python-version", "control-covered", "control-socket", "route-reachable", "unlisted-refused",
                 "nproc-accounting", "ucount-locked"):
        assert cmds[name][0].startswith(shlex.quote(inputs.python) + " -I -c "), name
    assert "oom_score_adj" in cmds["oom-score-adj"][0]
    assert inputs.control_target in cmds["control-covered"][0]
    assert inputs.socket_target in cmds["control-socket"][0]
    assert inputs.mcp_target in cmds["budget-tool"][0]
    assert inputs.route_host in cmds["route-reachable"][0]
    assert inputs.refused_host in cmds["unlisted-refused"][0] and inputs.dial in cmds["unlisted-refused"][0]
    assert probe.INDEX_PACKAGE in cmds["index-reachable"][0] and "--no-cache-dir" in cmds["index-reachable"][0]
    assert 'rm -rf "$d"' in cmds["index-reachable"][0] and 'rm -f "$f"' in cmds["dev-shm"][0]
    # The index check uses the agent's own pip from PATH, never the system interpreter, which the
    # agent cap installs no pip for.
    assert "pip download" in cmds["index-reachable"][0]
    assert "/usr/bin/python3" not in cmds["index-reachable"][0] and "python3 -m pip" not in cmds["index-reachable"][0]
    [refused] = [c for c in probe.SANDBOX_CHECKS if c.name == "unlisted-refused"]
    assert _inputs(refused_host=None) and refused.commands(_inputs(refused_host=None)) == ()
    assert [c.name for c in probe.SANDBOX_CHECKS if c.sealed] == ["route-reachable", "unlisted-refused",
                                                                  "icmp-held", "index-reachable"]
    # The ICMP check runs the drive's seal probe's own program, as code: the sandbox has no checkout.
    icmp = shlex.split(cmds["icmp-held"][0])
    assert icmp[:3] == [inputs.python, "-I", "-c"] and icmp[4:] == ["icmp"]
    assert icmp[3] == Path(seal_probe.__file__).read_text()


@pytest.mark.parametrize("out, ok", [
    ('{"icmp": {"passed": true, "sec": 0.1, "detail": "no echo left: the ping echo was refused"}}', True),
    ('noise first\n{"icmp": {"passed": true, "sec": 0.1, "detail": "no echo left"}}', True),  # its last line
    ('{"icmp": {"passed": false, "sec": 2.0, "detail": "a ping echo to 10.0.0.1 left the seal, unanswered"}}', False),
    ('{"icmp": {"passed": "yes", "sec": 0.1, "detail": ""}}', False), ("Traceback", False), ("", False)])
def test_icmp_held(out, ok):
    check = _check("icmp-held", R(out))
    assert check.ok is ok and check.name == "icmp-held"


def test_a_run_keeps_the_checks_done_before_the_sandbox_ended():
    calls = []

    async def ending(command, timeout_sec=None):
        calls.append(command)
        if len(calls) == 3:
            raise ConnectionError("the sandbox ended")
        return R("1000 1000" if len(calls) == 2 else "3.12.1")

    done: list[probe.ProbeCheck] = []
    with pytest.raises(ConnectionError):
        asyncio.run(probe.run_sandbox_checks(ending, _inputs(), which="unsealed", timeout_sec=5, done=done))
    assert [c.name for c in done] == ["python-version", "uid-gid"] and all(c.ok for c in done)


def test_a_run_of_one_kind_runs_only_that_kind_each_command_with_the_timeout():
    seen: list[tuple[str, int]] = []

    async def answer(command, timeout_sec=None):
        seen.append((command, timeout_sec))
        return R("")

    sealed = asyncio.run(probe.run_sandbox_checks(answer, _inputs(), which="sealed", timeout_sec=20))
    assert [c.name for c in sealed] == ["route-reachable", "unlisted-refused", "icmp-held", "index-reachable"]
    assert len(seen) == 4 and {t for _, t in seen} == {20}
    unsealed = asyncio.run(probe.run_sandbox_checks(answer, _inputs(), which="unsealed", timeout_sec=120))
    assert not {c.name for c in unsealed} & {"route-reachable", "unlisted-refused", "icmp-held", "index-reachable"}
    every = asyncio.run(probe.run_sandbox_checks(answer, _inputs(), which="all", timeout_sec=120))
    assert [c.name for c in every] == [c.name for c in probe.SANDBOX_CHECKS]


# ---------------------------------------------------------------------------
# The host-side checks
# ---------------------------------------------------------------------------


def test_owner_and_readiness_and_record():
    assert probe.owner_check(1000, 1000).ok is True
    assert probe.owner_check(1001, 1000).ok is False
    assert probe.owner_check(None, 1000).ok is False and probe.owner_check(None, 1000).observed == "missing"
    assert probe.readiness_check(10.0, 300.0).ok is True
    assert probe.readiness_check(400.0, 300.0).ok is False
    rec = probe.record([probe.ProbeCheck(name="a", ok=True, observed="", expected=""),
                        probe.ProbeCheck(name="b", ok=False, observed="", expected="")])
    assert rec.all_ok is False and len(rec.checks) == 2 and rec.oom is None
    assert probe.record([probe.ProbeCheck(name="a", ok=True, observed="", expected="")]).all_ok is True


@pytest.mark.parametrize("ready, ok, observed", [
    (probe.ReadyLine(spawner_pid=12, reaper_pid=13, score=0), True, "oom_score_adj=0"),
    (probe.ReadyLine(spawner_pid=12, reaper_pid=13, score=1000), False, "oom_score_adj=1000"),
    (probe.ReadyLine(spawner_pid=12, reaper_pid=13, score=None), False, "oom_score_adj=unknown"),
    (None, False, "no ready line"),
])
def test_the_servers_ready_record_says_it_holds_score_0(ready, ok, observed):
    check = probe.server_score_check(ready)
    assert (check.ok, check.observed) == (ok, observed)


def _record(spawner_pid: int, reaper_pid: int, score: int | None = 0) -> bytes:
    return protocol.ready_record(spawner_pid=spawner_pid, reaper_pid=reaper_pid, score=score)


def test_the_ready_file_names_a_pair_only_from_one_whole_record(tmp_path):
    ready = tmp_path / "ready"
    assert probe.read_ready(ready) is None                                   # not made yet
    ready.write_bytes(b"\0" * protocol.READY_BYTES)                          # allocated, no record yet
    assert probe.read_ready(ready) is None
    old, new = _record(15, 17), _record(40, 42)
    ready.write_bytes(old)
    assert probe.read_ready(ready) == probe.ReadyLine(spawner_pid=15, reaper_pid=17, score=0)
    # Caught mid-rewrite: the start of the new record and the rest of the old. A cut that leaves
    # one record whole (the two share their first bytes) reads as that record; every other reads
    # as no pair, never as a wrong one.
    whole = {old: probe.read_ready(ready), new: probe.ReadyLine(spawner_pid=40, reaper_pid=42, score=0)}
    text = max(old.index(b"\n"), new.index(b"\n")) + 1
    torn = 0
    for cut in range(1, text):
        mixed = new[:cut] + old[cut:]
        ready.write_bytes(mixed)
        assert probe.read_ready(ready) == whole.get(mixed), cut
        torn += mixed not in whole
    # Every cut between the records' first and last differing bytes leaves a torn record.
    differ = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert torn == differ[-1] - differ[0] > 40
    flipped = bytearray(new)
    flipped[new.index(b"spawner_pid=") + len("spawner_pid=")] ^= 0x01                # 40 reads as 50
    ready.write_bytes(bytes(flipped))
    assert probe.read_ready(ready) is None
    ready.write_bytes(new[:-1] + b"x")                                       # garbage in the padding
    assert probe.read_ready(ready) is None
    ready.write_bytes(new[:100])                                             # cut short
    assert probe.read_ready(ready) is None
    ready.write_bytes(_record(40, 42, score=None))                           # a score the server could not read
    assert probe.read_ready(ready) == probe.ReadyLine(spawner_pid=40, reaper_pid=42, score=None)


def _step(tmp_path: Path, processes: list[tuple]) -> tuple[Path, Path]:
    """A step cgroup and a /proc of *processes*: (host pid, NSpid, name, score, fd targets), and
    the parent's host pid as a sixth element where the process names one."""
    cgroup, proc = tmp_path / "cg", tmp_path / "proc"
    (cgroup / "task_0").mkdir(parents=True)
    (cgroup / "cgroup.procs").write_text("\n".join(str(p[0]) for p in processes[:2]) + "\n")
    (cgroup / "task_0" / "cgroup.procs").write_text("\n".join(str(p[0]) for p in processes[2:]) + "\n")
    for pid, nspid, name, score, fds, *parent in processes:
        d = proc / str(pid)
        (d / "fd").mkdir(parents=True)
        ppid = "".join(f"PPid:\t{x}\n" for x in parent)
        (d / "status").write_text(f"Name:\t{name}\n{ppid}NSpid:\t{nspid}\nVmRSS:\t4 kB\n")
        (d / "comm").write_text(name + "\n")
        (d / "oom_score_adj").write_text(score + "\n")
        for n, target in enumerate(fds):
            (d / "fd" / str(n)).symlink_to(target)
    return cgroup, proc


#: A sealed sandbox's step: the starter and pasta at the base, the server and the spawner (pid 7
#: in the container) one below, the reaper (pid 8 there, PID 1 of the commands) and a command
#: two below.
STEP = [(11, "11", "starter", "1000", []), (12, "12", "pasta", "1000", []),
        (13, "13\t1", "python3", "0", []), (14, "14\t7", "python3", "0", ["socket:[99]"]),
        (15, "15\t8\t1", "python3", "0", ["/dev/null", "pipe:[1]", "pipe:[2]"], 14),
        (16, "16\t9\t2", "bash", "1000", ["/dev/null", "pipe:[3]", "pipe:[3]"])]
READY = probe.ReadyLine(spawner_pid=7, reaper_pid=8, score=0)


def test_the_spawner_the_reaper_and_their_depth_are_read_through_the_steps_processes(tmp_path):
    cgroup, proc = _step(tmp_path, STEP)
    score = probe.spawner_score_check(READY, cgroup, proc=proc)
    assert (score.ok, score.observed) == (True, "oom_score_adj=0")
    depth = probe.depth_check(READY, cgroup, proc=proc)
    assert (depth.ok, depth.observed) == (True, "the reaper at the step's base + 2")
    fds = probe.reaper_socket_check(READY, cgroup, proc=proc)
    assert (fds.ok, fds.observed) == (True, "0:/dev/null 1:pipe:[1] 2:pipe:[2]")


def test_the_spawner_holds_the_runners_hard_process_limit_its_commands_namespace_counts_against(tmp_path):
    cgroup, proc = _step(tmp_path, STEP)
    limits = ("Limit                     Soft Limit           Hard Limit           Units     \n"
              "Max processes             {soft}              {hard}              processes \n")
    (proc / "14" / "limits").write_text(limits.format(soft=4124579, hard=4124579))
    held = probe.spawner_limits_check(READY, cgroup, 4124579, proc=proc)
    assert (held.ok, held.observed) == (True, "Max processes 4124579 4124579")
    # A soft limit the step lowered, or a hard one not the runner's, is not the parent level's.
    (proc / "14" / "limits").write_text(limits.format(soft=1024, hard=4124579))
    assert not probe.spawner_limits_check(READY, cgroup, 4124579, proc=proc).ok
    (proc / "14" / "limits").write_text(limits.format(soft="unlimited", hard="unlimited"))
    assert probe.spawner_limits_check(READY, cgroup, None, proc=proc).ok
    assert probe.spawner_limits_check(READY, None, 4124579).observed == "no step processes"
    (proc / "14" / "limits").unlink()
    assert probe.spawner_limits_check(READY, cgroup, 4124579, proc=proc).observed == "unreadable: FileNotFoundError"


def test_a_spawner_at_1000_or_a_reaper_holding_a_socket_fails(tmp_path):
    bad = [*STEP[:3], (14, "14\t7", "python3", "1000", []), (15, "15\t8\t1", "python3", "0", ["socket:[5]"], 14)]
    cgroup, proc = _step(tmp_path, bad)
    assert probe.spawner_score_check(READY, cgroup, proc=proc).observed == "oom_score_adj=1000"
    assert not probe.spawner_score_check(READY, cgroup, proc=proc).ok
    fds = probe.reaper_socket_check(READY, cgroup, proc=proc)
    assert (fds.ok, fds.observed) == (False, "0:socket:[5]")


def test_a_reaper_at_the_spawners_own_depth_or_no_step_fails(tmp_path):
    # The reaper at the spawner's depth, not one below: the commands' namespace is the container's,
    # so they share the exec server's. The check reads the spawner one above, never a fixed depth.
    shallow = [*STEP[:4], (15, "15\t1", "python3", "0", [])]
    cgroup, proc = _step(tmp_path, shallow)
    shallow_ready = probe.ReadyLine(spawner_pid=7, reaper_pid=15, score=0)
    bad = probe.depth_check(shallow_ready, cgroup, proc=proc)
    assert (bad.ok, bad.observed) == (False, "no reaper one namespace below the serving spawner")
    for check in (probe.spawner_score_check, probe.depth_check, probe.reaper_socket_check):
        assert check(READY, None).observed == "no step processes"
        assert check(None, cgroup, proc=proc).observed == "no ready line"
        assert not check(READY, None).ok


#: A seal as dumped from a probe build: pasta makes a PID namespace of its own, so
#: the interceptor and apptainer's starter sit one below the base, the init shim, the exec server
#: (pid 13 in the container) and the spawner (pid 15) two below, and the reaper (pid 17 there, PID
#: 1 of the commands) and a command three below. The host checks read by the ready line's pids,
#: never a fixed depth, so they hold here as in the no-pasta layout.
LIVE_STEP = [(20, "20", "bash", "1000", []), (21, "21", "pasta", "1000", []),
             (22, "22\t5", "python3", "1000", []), (23, "23\t6", "starter", "1000", []),
             (24, "24\t9\t1", "python3", "0", []),
             (25, "25\t10\t13", "python3", "0", ["socket:[99]"]),
             (26, "26\t11\t15", "python3", "0", []),
             (27, "27\t12\t17\t1", "python3", "0", ["/dev/null", "pipe:[1]", "pipe:[2]"], 26),
             (28, "28\t13\t18\t2", "bash", "1000", ["/dev/null", "pipe:[3]", "pipe:[3]"])]
READY_LIVE = probe.ReadyLine(spawner_pid=15, reaper_pid=17, score=0)


def test_the_host_checks_hold_on_the_live_pasta_layout_three_below_the_base(tmp_path):
    cgroup, proc = _step(tmp_path, LIVE_STEP)
    score = probe.spawner_score_check(READY_LIVE, cgroup, proc=proc)
    assert (score.ok, score.observed) == (True, "oom_score_adj=0")
    depth = probe.depth_check(READY_LIVE, cgroup, proc=proc)
    assert (depth.ok, depth.observed) == (True, "the reaper at the step's base + 3")
    fds = probe.reaper_socket_check(READY_LIVE, cgroup, proc=proc)
    assert (fds.ok, fds.observed) == (True, "0:/dev/null 1:pipe:[1] 2:pipe:[2]")
    # The base processes, at the pasta layer, are the ones the outside measurement records.
    outside = probe.outside_check(READY_LIVE, cgroup, proc=proc)
    assert (outside.ok, outside.observed) == (
        True, "bash:1000:4096 pasta:1000:4096 python3:1000:4096 starter:1000:4096")


#: What a command can plant in the live layout to look like the serving pair: one command holding
#: the spawner's pid (15) in the commands' namespace, and below it a PID 1 whose pid there is the
#: reaper's (17), at 1000 and holding a socket. Listed before the real pair.
PLANTED = [(29, "29\t14\t19\t15", "python3", "1000", [], 28),
           (30, "30\t15\t20\t17\t1", "python3", "1000", ["socket:[7]"], 29)]


def test_the_host_checks_read_the_real_pair_past_a_planted_one(tmp_path):
    cgroup, proc = _step(tmp_path, [*LIVE_STEP[:2], *PLANTED, *LIVE_STEP[2:]])
    assert probe.depth_check(READY_LIVE, cgroup, proc=proc).observed == "the reaper at the step's base + 3"
    assert probe.spawner_score_check(READY_LIVE, cgroup, proc=proc).observed == "oom_score_adj=0"
    fds = probe.reaper_socket_check(READY_LIVE, cgroup, proc=proc)
    assert (fds.ok, fds.observed) == (True, "0:/dev/null 1:pipe:[1] 2:pipe:[2]")


def test_the_ready_source_reads_the_file_fresh_on_every_call(tmp_path):
    # Nothing is kept between calls: each sample and each kill sees the pair the server last
    # wrote, so once a fresh spawner is named no call asks for the old one.
    ready = tmp_path / "ready"
    assert probe.ready_pids(ready) is None                     # the server has not written it yet
    ready.write_bytes(_record(15, 17))
    assert probe.ready_pids(ready) == ReadyPids(spawner_pid=15, reaper_pid=17)
    ready.write_bytes(_record(40, 42))
    assert probe.ready_pids(ready) == ReadyPids(spawner_pid=40, reaper_pid=42)
    ready.write_bytes(_record(15, 17))
    assert probe.ready_pids(ready) == ReadyPids(spawner_pid=15, reaper_pid=17)
    ready.write_bytes(b"\0" * protocol.READY_BYTES)
    assert probe.ready_pids(ready) is None


def test_the_policys_record_of_the_refusal_is_required():
    assert probe.refusal_recorded_check(("hls-seal-probe.invalid",), "hls-seal-probe.invalid").ok is True
    assert probe.refusal_recorded_check((), "hls-seal-probe.invalid").ok is False
    assert probe.refusal_recorded_check(None, "hls-seal-probe.invalid").observed == "no interceptor status"
    assert probe.refusal_recorded_check(None, None).ok is True  # the policy refuses no host


def test_the_env_file_holds_no_route_variable_and_no_secrets_value_and_names_never_a_value():
    grants, secrets = {"ANTHROPIC_API_KEY"}, ["sk-test-0123456789abcdef"]
    clean = probe.env_file_check({"HLS_EXEC_TOKEN": "tok", "HF_HOME": "/data"}, grants, secrets)
    assert (clean.ok, clean.observed) == (True, "clean")
    named = probe.env_file_check({"ANTHROPIC_API_KEY": "x", "HF_HOME": "/data"}, grants, secrets)
    assert (named.ok, named.observed) == (False, "ANTHROPIC_API_KEY")
    holding = probe.env_file_check({"NOTE": "key=sk-test-0123456789abcdef;"}, grants, secrets)
    assert (holding.ok, holding.observed) == (False, "NOTE") and secrets[0] not in holding.model_dump_json()


def test_the_kernels_pid_max_is_recorded_and_the_ceiling_must_bind_below_it(tmp_path):
    # The commands' ceiling contains a fork storm only while it is below the node's pid_max; one at
    # or above it (a hard-limit headroom of twice pid_max) would let the commands
    # take every pid of the node first.
    pid_max = tmp_path / "pid_max"
    pid_max.write_text("4194304\n")
    under = probe.pid_max_check(131_072, path=pid_max)
    assert (under.name, under.ok, under.observed) == ("pid-max", True, "pid_max=4194304 ceiling=131072")
    over = probe.pid_max_check(9_279_297, path=pid_max)
    assert (over.ok, over.observed) == (False, "pid_max=4194304 ceiling=9279297")
    assert probe.pid_max_check(4_194_304, path=pid_max).ok is False
    missing = probe.pid_max_check(131_072, path=tmp_path / "gone")
    assert (missing.ok, missing.observed) == (False, "pid_max unreadable ceiling=131072")


def test_the_split_is_a_measurement_that_passes():
    sample = SplitSample(at=1.0, total=10, tmpfs=1, harness=3, contestant=6, oom_kills=0)
    assert probe.split_check(sample).observed == "harness=3 contestant=6 tmpfs=1"
    assert probe.split_check(None).ok and probe.split_check(None).observed == "no sample"


def test_every_process_outside_the_container_is_at_the_contestants_score(tmp_path):
    """The processes above the serving spawner's PID namespace (the starter and pasta here) are
    recorded with their scores and resident bytes, and each must hold 1000; the exec server and the
    spawner in the container lower theirs to 0 and are not among them."""
    cgroup, proc = _step(tmp_path, STEP)
    outside = probe.outside_check(READY, cgroup, proc=proc)
    assert (outside.ok, outside.observed) == (True, "starter:1000:4096 pasta:1000:4096")
    # A name that is not UTF-8 is recorded with its bytes replaced, never a crash of the probe.
    (proc / "12" / "comm").write_bytes(b"pas\xffta\n")
    assert probe.outside_check(READY, cgroup, proc=proc).observed == "starter:1000:4096 pas\ufffdta:1000:4096"


@pytest.mark.parametrize("score", ["0", "999", "-1000"])
def test_a_process_outside_the_container_below_the_contestants_score_fails(tmp_path, score):
    cgroup, proc = _step(tmp_path, [STEP[0], (12, "12", "pasta", score, []), *STEP[2:]])
    outside = probe.outside_check(READY, cgroup, proc=proc)
    assert (outside.ok, outside.observed) == (False, f"starter:1000:4096 pasta:{score}:4096")


def test_an_unreadable_score_or_no_serving_pair_fails_the_outside_check(tmp_path):
    cgroup, proc = _step(tmp_path, STEP)
    (proc / "11" / "oom_score_adj").unlink()
    unread = probe.outside_check(READY, cgroup, proc=proc)
    assert (unread.ok, unread.observed) == (False, "11:unreadable: FileNotFoundError pasta:1000:4096")
    assert (probe.outside_check(READY, None).ok, probe.outside_check(READY, None).observed) == (
        False, "no step processes")
    assert probe.outside_check(None, cgroup, proc=proc).observed == "no ready line"


#: A Docker sandbox's step: the container is the whole step, so its init (docker-init), the exec
#: server and the spawner (pid 7 in the container) are at the base, the reaper (pid 8 there) and a
#: command one below; the seal's gateway is a container of its own.
DOCKER_STEP = [(41, "41\t1", "docker-init", "0", []), (42, "42\t6", "python3", "0", []),
               (43, "43\t7", "python3", "0", ["socket:[99]"], 42),
               (44, "44\t8\t1", "python3", "0", ["/dev/null", "pipe:[1]", "pipe:[2]"], 43),
               (45, "45\t9\t2", "bash", "1000", [], 44)]


def test_on_docker_no_process_of_the_step_is_outside_the_container(tmp_path):
    cgroup, proc = _step(tmp_path, DOCKER_STEP)
    outside = probe.outside_check(READY, cgroup, proc=proc)
    assert (outside.ok, outside.observed) == (True, "none outside the container")
    assert probe.depth_check(READY, cgroup, proc=proc).observed == "the reaper at the step's base + 1"


# ---------------------------------------------------------------------------
# Through the real exec server
# ---------------------------------------------------------------------------


class _Unix(http.client.HTTPConnection):
    def __init__(self, path: Path) -> None:
        super().__init__("control", timeout=180)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._path))


def _serve(root: Path, *, namespaces: bool, nproc: int | None = None,
           ucounts: dict[str, int] | None = None) -> subprocess.Popen:
    sock = root / protocol.SOCKET_NAME
    env = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", protocol.TOKEN_ENV: TOKEN}
    argv = [sys.executable, "-I", str(AGENT / "exec_server.py"), "--socket", str(sock), "--spawner",
            str(AGENT / "spawner.py"), "--shm-bytes", str(1 << 20), "--uid", str(os.getuid()), "--gid",
            str(os.getgid()), "--hide", str(root), "--cli-launch", "exec claude "]
    argv += ([] if namespaces else ["--no-namespaces"]) + ([] if nproc is None else ["--command-nproc", str(nproc)])
    argv += [x for name, limit in (ucounts or {}).items() for x in ("--ucount", f"{name}={limit}")]
    proc = subprocess.Popen(argv, env=env, stdin=subprocess.PIPE, stderr=(root / "err").open("w"))
    deadline = time.monotonic() + 20
    while not sock.exists():
        if proc.poll() is not None or time.monotonic() > deadline:
            proc.kill()
            raise AssertionError((root / "err").read_text())
        time.sleep(0.02)
    return proc


def _checks(root: Path, inputs: probe.ProbeInputs) -> dict[str, probe.ProbeCheck]:
    sock = root / protocol.SOCKET_NAME

    async def run(command, timeout_sec=120):
        def post():
            conn = _Unix(sock)
            body = json.dumps({"command": command, "timeout_sec": timeout_sec}).encode()
            conn.request("POST", "/exec", body, {protocol.TOKEN_HEADER: TOKEN, "Content-Type": "application/json"})
            data = json.loads(conn.getresponse().read())
            conn.close()
            return R(data["stdout"], data["return_code"])
        return await asyncio.to_thread(post)

    return {c.name: c for c in asyncio.run(probe.run_sandbox_checks(run, inputs, which="unsealed", timeout_sec=120))}


def _stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _nvidia_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    except OSError:
        return 0
    return len([line for line in out.stdout.splitlines() if line.startswith("GPU")]) if out.returncode == 0 else 0


@NOT_ROOT
def test_the_checks_that_need_a_namespace_fail_without_one_and_the_rest_pass(tmp_path_factory):
    root = tmp_path_factory.mktemp("c")
    proc = _serve(root, namespaces=False, nproc=NPROC, ucounts=UCOUNTS)
    try:
        # The control directory and socket are the server's real ones, so without the cover they
        # are seen. The sandbox's interpreter is the one the server was launched with.
        inputs = _inputs(uid=os.getuid(), gid=os.getgid(), shm_bytes=1 << 20, control_target=str(root),
                         socket_target=str(root / protocol.SOCKET_NAME), gpus=_nvidia_count(), command_nproc=NPROC,
                         python=sys.executable)
        checks = _checks(root, inputs)
    finally:
        _stop(proc)
    not_ok = {name for name, c in checks.items() if not c.ok}
    # Without a namespace a command's PID 1 is the machine's: an init of another user keeps the
    # command from setting its limits, while a container's init that runs as this user, as a Docker
    # container's does, takes them, and no later command inherits them either way.
    need = set(NAMESPACE_CHECKS) - ({"limits-not-inherited"} if os.stat("/proc/1").st_uid == os.getuid() else set())
    assert need <= not_ok, checks
    # These hold without a namespace too. (The PID 1 checks and cap-eff depend on the host.)
    for name in ("python-version", "uid-gid", "oom-score-adj", "route-env-absent", "command-nproc", "gpu-count"):
        assert checks[name].ok, checks[name]
    assert not {c.name for c in probe.SANDBOX_CHECKS if c.sealed} & set(checks)


def test_every_isolation_check_passes_in_real_namespaces(tmp_path_factory):
    """The exec server in its production mode, the spawner making its own namespaces: every
    in-sandbox check that needs no container, budget service or GPU passes."""
    require_real_namespaces()
    root = tmp_path_factory.mktemp("c")
    proc = _serve(root, namespaces=True, nproc=NPROC, ucounts=UCOUNTS)
    try:
        inputs = _inputs(uid=os.getuid(), gid=os.getgid(), shm_bytes=1 << 20, control_target=str(root),
                         socket_target=str(root / protocol.SOCKET_NAME), gpus=_nvidia_count(), command_nproc=NPROC,
                         python=sys.executable)
        checks = _checks(root, inputs)
    finally:
        _stop(proc)
    # The budget service; and the site's accounting, which this host's kernel need not share.
    needs_more = {"budget-tool", "nproc-accounting"}
    failed = {name: c for name, c in checks.items() if not c.ok and name not in needs_more}
    assert not failed, failed


def test_the_build_knocks_before_it_makes_the_done_file_and_starts_from_no_old_files(tmp_path, monkeypatch):
    """The sandbox's hold keeps its ports until the done file appears, so the stage knocks first
    and only then makes it: each knock finds the ports held. Files an earlier attempt left are
    gone before the hold starts, or an old ready file would let the knocks run before the ports
    are bound. The ports start at the floor of pasta's fresh namespace, where the hold binds
    unprivileged."""
    scratch = tmp_path / "seal-probe"
    scratch.mkdir()
    (scratch / "ready").touch()
    (scratch / "done").touch()
    seen: dict[str, object] = {}
    picked: list[tuple[frozenset[int], int]] = []

    def pick(*, avoid, floor=0):
        picked.append((frozenset(avoid), floor))
        return 20001, 20002

    def knock(addresses, tcp, udp):
        seen["done_at_knock"] = (scratch / "done").exists()
        return tuple(seal_probe.Knock(a, proto, port, "refused") for proto, port in (("tcp", tcp), ("udp", udp))
                     for a in addresses)

    monkeypatch.setattr(seal_probe, "inbound_ports", pick)
    monkeypatch.setattr(seal_probe, "node_addresses", lambda: ("127.0.0.1", "10.0.0.5"))
    monkeypatch.setattr(seal_probe, "knock_inbound", knock)

    async def hold(command, timeout_sec=None):
        # The hold as a working seal runs it: ready once bound, then held until the done file.
        seen["old_files"] = ((scratch / "ready").exists(), (scratch / "done").exists())
        (scratch / "ready").touch()
        while not (scratch / "done").exists():
            await asyncio.sleep(0.01)
        return R(json.dumps({"inbound": {"passed": True, "sec": 0.1, "detail": "nothing reached"}, "reached": []}))

    check = asyncio.run(probe.inbound_check(hold, _inputs(), avoid={8888, 8899}, scratch=scratch,
                                            target="/logs/seal-probe", timeout_sec=5))
    assert check.ok and seen == {"old_files": (False, False), "done_at_knock": False}
    assert picked == [(frozenset({8888, 8899}), seal_probe.NAMESPACE_PORT_FLOOR)]
    assert list(scratch.iterdir()) == []
