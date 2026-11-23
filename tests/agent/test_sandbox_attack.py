"""The spawner's reaper against real nested namespaces: a command that ends PID 1 of its own
namespace, or lowers its resource limits, loses only the contestant's own commands; the server
keeps serving, from a fresh namespace, and a later command inherits none of the limits a command
set on PID 1. These run the exec server in its production mode (the spawner makes the user, mount
and PID namespaces itself).

Skipped where unprivileged user namespaces or the mounts the spawner makes are refused, as on a
workstation under apparmor's ``apparmor_restrict_unprivileged_userns`` (the spawner cannot write
``/proc/self/setgroups`` even with a full capability set under bwrap); there the attack's effect
is pinned by ``test_spawner``'s reaper-death path and ``test_exec_server``'s recovery. CI lifts the
restriction before the tests (``.github/workflows/test.yml``), and under CI (``CI`` set) a runner
that still refuses them fails these tests instead of skipping them (``require_real_namespaces``),
so no automatic run reports them green unchecked."""

from __future__ import annotations

import functools
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from hls.agent import exec_protocol as protocol

AGENT = Path(protocol.__file__).parent
TOKEN = "tok-0123456789abcdef"

_PROBE = (
    "import os,sys\n"
    "pid=os.fork()\n"
    "if pid==0:\n"
    " try:\n"
    "  os.unshare(os.CLONE_NEWUSER)\n"
    "  open('/proc/self/setgroups','w').write('deny')\n"
    "  open('/proc/self/uid_map','w').write(f'1234 {os.getuid()} 1\\n')\n"
    "  open('/proc/self/gid_map','w').write(f'5678 {os.getgid()} 1\\n')\n"
    "  os.unshare(os.CLONE_NEWNS); os.unshare(os.CLONE_NEWPID)\n"
    "  os._exit(0)\n"
    " except OSError:\n"
    "  os._exit(1)\n"
    "sys.exit(os.waitstatus_to_exitcode(os.waitpid(pid,0)[1]))\n")


@functools.cache
def _real_namespaces() -> bool:
    if os.getuid() == 0:
        return False  # the spawner refuses to map a command to uid 0
    return subprocess.run([sys.executable, "-I", "-c", _PROBE], capture_output=True).returncode == 0


def require_real_namespaces() -> None:
    """Skip the calling test where unprivileged user namespaces or the spawner's mounts are refused,
    or, under CI (the ``CI`` variable set, as GitHub Actions sets it), fail it: the workflow lifts
    the restriction, so a runner that still refuses them must not report the isolation tests green."""
    if _real_namespaces():
        return
    why = "unprivileged user namespaces / mounts are refused here (or the tests run as root)"
    if os.environ.get("CI"):
        pytest.fail(f"{why}; CI must run the sandbox's isolation tests, so the runner must allow them")
    pytest.skip(why)


@functools.cache
def _user_namespaces() -> bool:
    return subprocess.run([sys.executable, "-I", "-c", "import os; os.unshare(os.CLONE_NEWUSER)"],
                          capture_output=True).returncode == 0


def require_user_namespaces() -> None:
    """Skip the calling test where an unprivileged process cannot make a user namespace, or, under
    CI, fail it, as ``require_real_namespaces`` does: the probe's accounting command makes one to
    measure how the kernel counts its tasks."""
    if _user_namespaces():
        return
    why = "this host refuses unprivileged user namespaces, which the accounting command measures in"
    if os.environ.get("CI"):
        pytest.fail(f"{why}; CI must run the probe's checks of the kernel, so the runner must allow them")
    pytest.skip(why)


@pytest.fixture(autouse=True)
def _namespaces() -> None:
    require_real_namespaces()


class _Unix(http.client.HTTPConnection):
    def __init__(self, path: Path, timeout: float = 60.0) -> None:
        super().__init__("control", timeout=timeout)
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(str(self._path))


class _Server:
    """The real exec server, its spawner making its own namespaces."""

    def __init__(self, root: Path, *extra: str) -> None:
        self.socket = root / protocol.SOCKET_NAME
        self.err = root / "server.err"
        self.proc = subprocess.Popen(
            [sys.executable, "-I", str(AGENT / "exec_server.py"), "--socket", str(self.socket), "--spawner",
             str(AGENT / "spawner.py"), "--shm-bytes", str(8 << 20), "--uid", str(os.getuid()),
             "--gid", str(os.getgid()), "--hide", str(root), "--cli-launch", "exec claude ", *extra],
            env={"PATH": "/usr/bin:/bin", protocol.TOKEN_ENV: TOKEN}, stdin=subprocess.PIPE,
            stderr=self.err.open("w"))
        deadline = time.monotonic() + 20
        while not self.socket.exists():
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.stop()
                # This module runs only where the spawner makes its namespaces (``pytestmark``),
                # so a server that does not come up here failed, whatever it failed on.
                pytest.fail(f"the server did not come up: {self.err.read_text()}")
            time.sleep(0.02)

    def run(self, command: str, timeout: float = 60.0) -> dict:
        conn = _Unix(self.socket, timeout)
        conn.request("POST", "/exec", body=json.dumps({"command": command}).encode(),
                     headers={"X-HLS-Token": TOKEN, "Content-Type": "application/json"})
        resp = conn.getresponse()
        out = json.loads(resp.read())
        conn.close()
        return out

    def log(self) -> list[str]:
        return self.err.read_text().splitlines()

    def spawner_pid(self) -> int:
        line = next(x for x in self.log() if x.startswith("hls-exec: ready "))
        return int(re.search(r"spawner_pid=(\d+)", line).group(1))

    def ready_count(self) -> int:
        return sum(1 for x in self.log() if x.startswith("hls-exec: ready "))

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()


@pytest.fixture
def server(short_dir):
    s = _Server(short_dir)
    yield s
    s.stop()


#: An inotify instance a command opens, or a user namespace it makes, until refused: how many it
#: got and the error that refused the next.
_UNTIL_REFUSED = (
    "import ctypes, errno, os, signal, sys\n"
    "libc = ctypes.CDLL(None, use_errno=True)\n"
    "kind, made, why = sys.argv[1], [], 'none'\n"
    "while len(made) < 64:\n"
    " if kind == 'inotify':\n"
    "  fd = libc.inotify_init1(0)\n"
    "  if fd < 0:\n"
    "   why = errno.errorcode[ctypes.get_errno()]\n"
    "   break\n"
    "  made.append(fd)\n"
    "  continue\n"
    " r, w = os.pipe()\n"
    " pid = os.fork()\n"
    " if pid == 0:\n"
    "  os.close(r)\n"
    "  try:\n"
    "   os.unshare(os.CLONE_NEWUSER)\n"
    "   os.write(w, b'k')\n"
    "  except OSError as e:\n"
    "   os.write(w, errno.errorcode[e.errno].encode())\n"
    "   os._exit(0)\n"
    "  while True:\n"
    "   signal.pause()\n"
    " os.close(w)\n"
    " got = os.read(r, 16)\n"
    " os.close(r)\n"
    " if got != b'k':\n"
    "  os.waitpid(pid, 0)\n"
    "  why = got.decode()\n"
    "  break\n"
    " made.append(pid)\n"
    "if kind != 'inotify':\n"
    " for pid in made:\n"
    "  os.kill(pid, signal.SIGKILL)\n"
    "  os.waitpid(pid, 0)\n"
    "print(len(made), why, flush=True)\n")


@pytest.fixture
def limited(short_dir):
    s = _Server(short_dir, "--ucount", "max_inotify_instances=4", "--ucount", "max_user_namespaces=3")
    yield s
    s.stop()


def test_a_commands_namespace_has_the_limits_it_was_given(limited):
    out = limited.run("cat /proc/sys/user/max_inotify_instances /proc/sys/user/max_user_namespaces")["stdout"]
    assert out.split() == ["4", "3"]


@pytest.mark.parametrize("kind, limit, refusal", [("inotify", 4, "EMFILE"), ("userns", 3, "ENOSPC")])
def test_a_command_is_refused_past_its_namespaces_limit_and_the_harness_is_not(limited, kind, limit, refusal):
    import shlex

    command = f"{shlex.quote(sys.executable)} -I -c {shlex.quote(_UNTIL_REFUSED)} {kind}"
    assert limited.run(command)["stdout"] == f"{limit} {refusal}"
    # What it made goes with it: the kernel frees a user namespace, and its count, in deferred
    # work, so within a few seconds the next command gets as many again. The harness, counted
    # outside the commands' namespace, was never short.
    deadline = time.monotonic() + 10
    while (again := limited.run(command)["stdout"]) != f"{limit} {refusal}" and time.monotonic() < deadline:
        time.sleep(0.5)
    assert again == f"{limit} {refusal}"
    assert limited.run("echo next")["stdout"] == "next"


def test_a_command_sees_only_its_namespaces_processes_not_the_harness(server):
    pids = {int(p) for p in server.run("ls -d /proc/[0-9]* | xargs -n1 basename")["stdout"].split()}
    own = int(server.run("echo $$")["stdout"])
    assert 1 in pids, "PID 1 is the reaper"
    assert server.spawner_pid() not in pids, "the serving process has no pid in the command's namespace"
    # The only processes are PID 1 and the command's own; nothing of the harness is addressable.
    assert pids <= {1} | set(range(2, own + 50))


def test_a_command_cannot_see_the_server_or_the_control_socket(server):
    # The control directory is covered, so the token-gated socket the server bound there cannot be
    # reached from inside: the command looks for it where it is on the host.
    assert server.socket.is_socket()
    assert server.run(f"test -S {server.socket} && echo found || echo hidden")["stdout"] == "hidden"


def test_a_command_can_neither_read_nor_replace_the_ready_file(server):
    """The ready file names the reaper the memory split trusts; the control directory's cover keeps
    a command from reading it, writing it, or renaming a file of its own over it."""
    ready = server.socket.parent / protocol.READY_NAME
    named = ready.read_text()
    out = server.run(f"cat {ready}; echo forged > {ready}; echo forged > /tmp/r && mv /tmp/r {ready}; "
                     f"touch {ready}.x; echo done")["stdout"]
    assert out.endswith("done") and "spawner_pid" not in out
    assert ready.read_text() == named and not ready.with_name(f"{ready.name}.x").exists()


def test_ending_pid1_by_its_cpu_limit_loses_only_the_contestants_commands_and_the_server_recovers(server):
    results: list[dict] = []
    inflight = threading.Thread(target=lambda: results.append(server.run("sleep 40")))
    inflight.start()
    time.sleep(0.5)
    # Lower PID 1's CPU hard limit, then burn its CPU until the kernel's RLIMIT_CPU SIGKILL (which
    # init protection does not drop) ends it. A capless command may do this because it shares PID
    # 1's uid.
    server.run("prlimit --pid 1 --cpu=1:1; timeout 20 sh -c 'while :; do kill -CHLD 1; done'; echo done")
    inflight.join(25)
    assert results == [{"stdout": None, "stderr": None, "return_code": -int(signal.SIGKILL)}]
    assert server.proc.poll() is None, "the server kept serving"
    assert server.run("echo recovered")["stdout"] == "recovered"
    assert server.ready_count() == 2, "a fresh namespace served the next command"
    assert any("reaper ended" in line for line in server.log())


def test_a_limit_a_command_sets_on_pid1_is_not_inherited_by_a_later_command(server):
    """Each command is forked by the spawner, so it inherits the spawner's limits, never PID 1's."""
    before = int(server.run("ulimit -n")["stdout"])
    assert before > 4
    server.run("prlimit --pid 1 --nofile=4:4; echo lowered")
    assert int(server.run("ulimit -n")["stdout"]) == before, "the later command keeps the spawner's limit"


# --- what a command can do to hide its open files --------------------------------------------------

#: A command that prints what its prctl calls return: PR_SET_DUMPABLE to 0 (and why it failed),
#: PR_GET_DUMPABLE, PR_SET_DUMPABLE to 1 and PR_GET_NO_NEW_PRIVS.
_PRCTL = ("import ctypes, os; libc = ctypes.CDLL(None, use_errno=True); u = ctypes.c_ulong; "
          "r = libc.prctl(4, u(0), u(0), u(0), u(0)); e = ctypes.get_errno(); "
          "print(r, os.strerror(e) if r else '-', libc.prctl(3, u(0), u(0), u(0), u(0)), "
          "libc.prctl(4, u(1), u(0), u(0), u(0)), libc.prctl(39, u(0), u(0), u(0), u(0)))")


def test_a_command_in_its_namespaces_cannot_make_itself_non_dumpable(server):
    """In the spawner's own namespaces, every command runs under its seccomp filter and with
    no_new_privs: prctl(PR_SET_DUMPABLE, 0) fails with EPERM, and the command stays dumpable."""
    import shlex

    out = server.run(f"{shlex.quote(sys.executable)} -I -c {shlex.quote(_PRCTL)}")
    assert out["return_code"] == 0, out
    assert out["stdout"].split() == ["-1", "Operation", "not", "permitted", "1", "0", "1"]


def _descendants(pid: int) -> list[int]:
    """*pid* and every process below it, from each task's ``children``."""
    found, pending = [], [pid]
    while pending:
        p = pending.pop()
        found.append(p)
        for task in Path(f"/proc/{p}/task").glob("*"):
            try:
                pending += [int(c) for c in (task / "children").read_text().split()]
            except OSError:
                continue
    return found


def test_a_command_running_a_program_it_cannot_read_is_past_the_disk_cap_and_the_reaper_never_is(
        server, short_dir, tmp_path_factory):
    """A command that runs a program from a file it cannot read becomes non-dumpable without
    prctl, so the host cannot list its descriptors: still so at the next check, the disk watch
    counts it as past its role's cap, naming it. The reaper is non-dumpable too, and outside a seal
    its descriptors cannot be listed either: the stage names it (``harness_pids``) and it is never
    judged."""
    import shlex

    from hls.agent import memory, probe
    from hls.substrate import memory_watch

    d = tmp_path_factory.mktemp("hide")
    marker = "4243"
    out = server.run(f"cp /bin/sleep {shlex.quote(str(d))}/xsleep && chmod 0111 {shlex.quote(str(d))}/xsleep && "
                     f"({shlex.quote(str(d))}/xsleep {marker} >/dev/null 2>&1 &); echo started")
    assert out["stdout"] == "started", out
    procs = _descendants(server.proc.pid)
    hidden = next(p for p in procs if Path(f"/proc/{p}/cmdline").read_bytes() == f"{d}/xsleep\0{marker}\0".encode())
    try:
        with pytest.raises(PermissionError):
            os.listdir(f"/proc/{hidden}/fd")
        cg = d / "cg"
        cg.mkdir()
        (cg / "cgroup.procs").write_text("".join(f"{p}\n" for p in procs))
        ready = probe.ready_pids(short_dir / protocol.READY_NAME)
        harness = memory.harness_pids(cg, ready)
        assert harness is not None and hidden not in harness
        first = memory_watch.held_unlinked(cg, [d], harness_pids=harness)
        assert [pid for pid, _ in first.unlistable] == [hidden]
        with pytest.raises(memory_watch.Unmeasured) as caught:
            memory_watch.held_unlinked(cg, [d], harness_pids=harness, suspects=first.unlistable)
        assert caught.value.why == "hidden" and f"process {hidden} ({d}/xsleep" in str(caught.value)
        # Unnamed, the reaper itself would be judged: outside a seal its descriptors are hidden too.
        [reaper] = harness
        (cg / "cgroup.procs").write_text(f"{reaper}\n")
        unnamed = memory_watch.held_unlinked(cg, [d], harness_pids=())
        with pytest.raises(memory_watch.Unmeasured, match=f"process {reaper} "):
            memory_watch.held_unlinked(cg, [d], harness_pids=(), suspects=unnamed.unlistable)
        assert memory_watch.held_unlinked(cg, [d], harness_pids=harness,
                                          suspects=unnamed.unlistable) == memory_watch.Held(0, frozenset())
    finally:
        os.kill(hidden, signal.SIGKILL)


def test_a_commands_child_that_exited_unreaped_is_never_judged(server, short_dir, tmp_path_factory):
    """A command's child that has exited and is not yet reaped has no memory left, and the kernel
    gives its descriptors' directory to root: an agent's CLI forks all the time, so the disk watch
    meets such children. A zombie holds no descriptors: however many checks find it, it is
    not judged."""
    import shlex

    from hls.agent import memory, probe
    from hls.substrate import memory_watch

    d = tmp_path_factory.mktemp("zombie")
    marker = "4244"
    parent = ("import os, sys, time\n"
              "if os.fork() == 0:\n"
              "    os._exit(0)\n"
              "time.sleep(int(sys.argv[1]))\n")
    out = server.run(f"({shlex.quote(sys.executable)} -I -c {shlex.quote(parent)} {marker} >/dev/null 2>&1 &); "
                     "echo started")
    assert out["stdout"] == "started", out
    deadline = time.monotonic() + 10
    while True:
        procs = _descendants(server.proc.pid)
        holder = next((p for p in procs if Path(f"/proc/{p}/cmdline").read_bytes().endswith(f"\0{marker}\0".encode())),
                      None)
        zombies = [p for p in procs if Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[0] == "Z"]
        if holder is not None and zombies:
            break
        assert time.monotonic() < deadline, "the command's child never showed as a zombie"
        time.sleep(0.05)
    try:
        [zombie] = zombies
        with pytest.raises(PermissionError):
            os.listdir(f"/proc/{zombie}/fd")
        cg = d / "cg"
        cg.mkdir()
        (cg / "cgroup.procs").write_text("".join(f"{p}\n" for p in procs))
        harness = memory.harness_pids(cg, probe.ready_pids(short_dir / protocol.READY_NAME))
        first = memory_watch.held_unlinked(cg, [d], harness_pids=harness)
        assert first.unlistable == frozenset()
        assert memory_watch.held_unlinked(cg, [d], harness_pids=harness, suspects=first.unlistable).unlistable == frozenset()
    finally:
        os.kill(holder, signal.SIGKILL)
