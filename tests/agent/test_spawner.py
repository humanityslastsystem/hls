"""The spawner: the namespace setup it would make (against a recording kernel, since unprivileged
user namespaces are refused on this machine), and the serving loop run for real without them."""

from __future__ import annotations

import os
import select
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

import hls.agent.spawner as spawner
from hls.agent import exec_protocol as protocol
import errno
from tests.agent.libc_signals import LIBC_RESERVED

SPAWNER = Path(spawner.__file__)
BASH_OWN = {"PWD", "SHLVL", "_"}


class _Kernel:
    """Records every call the namespace setup makes."""

    def __init__(self, *, fork_returns: int = 0, wait_returns: int = 0, fail_mount_on: str | None = None,
                 on_wait=None) -> None:
        self.calls: list[tuple] = []
        self.fork_returns = fork_returns
        self.wait_returns = wait_returns
        self.fail_mount_on = fail_mount_on
        self.on_wait = on_wait

    def unshare(self, flags):
        self.calls.append(("unshare", flags))

    def write(self, path, text):
        self.calls.append(("write", path, text))

    def mount(self, source, target, fstype, flags, data):
        self.calls.append(("mount", source, target, fstype, flags, data))
        if target == self.fail_mount_on:
            raise OSError(1, "Operation not permitted", target)

    def prctl(self, option, value):
        self.calls.append(("prctl", option, value))

    def execv(self, path, argv):
        self.calls.append(("execv", path, argv))

    def fork(self):
        self.calls.append(("fork",))
        return self.fork_returns

    def wait(self, pid):
        self.calls.append(("wait", pid))
        if self.on_wait is not None:
            self.on_wait()
        return self.wait_returns


def test_the_user_namespace_maps_the_ids_then_a_mount_namespace_follows():
    kernel = _Kernel()
    spawner.enter_user_and_mount_namespaces(kernel, uid=1234, gid=5678)
    assert kernel.calls == [
        ("unshare", os.CLONE_NEWUSER),
        ("write", "/proc/self/setgroups", "deny"),
        ("write", "/proc/self/uid_map", f"1234 {os.getuid()} 1\n"),
        ("write", "/proc/self/gid_map", f"5678 {os.getgid()} 1\n"),
        ("unshare", os.CLONE_NEWNS),
    ]


@pytest.mark.parametrize("ids", [{"uid": 0, "gid": 5678}, {"uid": 1234, "gid": 0}], ids=["uid0", "gid0"])
def test_commands_never_run_as_id_0_inside_the_namespace(ids):
    """As uid 0 of the namespace a command would keep its capabilities across execve and could
    remount /dev/shm or uncover the control directory."""
    kernel = _Kernel()
    with pytest.raises(spawner.SpawnerError, match="uid or gid 0"):
        spawner.enter_user_and_mount_namespaces(kernel, **ids)
    assert kernel.calls == []


def test_the_mount_namespace_caps_dev_shm_and_covers_the_hidden_paths_but_leaves_proc_to_the_reaper():
    kernel = _Kernel()
    spawner.set_up_mounts(kernel, shm_bytes=2**30, hide=["/run/hls-control", "/run/other"])
    covered = spawner.MS_RDONLY | spawner.MS_NOSUID | spawner.MS_NODEV | spawner.MS_NOEXEC
    assert kernel.calls == [
        ("mount", "none", "/", None, spawner.MS_REC | spawner.MS_PRIVATE, None),
        ("mount", "tmpfs", "/dev/shm", "tmpfs", spawner.MS_NOSUID | spawner.MS_NODEV, f"size={2**30},mode=1777"),
        ("mount", "tmpfs", "/run/hls-control", "tmpfs", covered, "size=4k,mode=000"),
        ("mount", "tmpfs", "/run/other", "tmpfs", covered, "size=4k,mode=000"),
    ]


def test_entering_the_pid_namespace_unshares_pid_then_forks():
    kernel = _Kernel(fork_returns=4242)
    assert spawner.enter_pid_namespace(kernel) == 4242
    assert kernel.calls == [("unshare", os.CLONE_NEWPID), ("fork",)]


def test_the_reaper_limits_the_commands_counts_mounts_its_own_proc_and_is_not_dumpable():
    # The limits go to the commands' user namespace's own files before any command runs, through
    # the /proc the reaper inherited: one mounted in the commands' own namespaces refuses them.
    kernel = _Kernel()
    spawner.reaper_set_up(kernel, {"max_user_namespaces": 64, "max_inotify_instances": 8})
    assert kernel.calls == [
        ("write", "/proc/sys/user/max_user_namespaces", "64"),
        ("write", "/proc/sys/user/max_inotify_instances", "8"),
        ("mount", "proc", "/proc", "proc", spawner.MS_NOSUID | spawner.MS_NODEV | spawner.MS_NOEXEC, None),
        ("prctl", spawner.PR_SET_DUMPABLE, 0),
    ]


def test_a_limit_the_reaper_cannot_write_keeps_it_from_coming_up():
    class Refused(_Kernel):
        def write(self, path, text):
            super().write(path, text)
            raise OSError(13, "Permission denied", path)

    kernel = Refused()
    with pytest.raises(spawner.SpawnerError, match="cannot limit the commands' max_inotify_instances to 8: "
                                                   "Permission denied"):
        spawner.reaper_set_up(kernel, {"max_inotify_instances": 8})
    assert [c[0] for c in kernel.calls] == ["write"], "nothing after the refused limit"


@pytest.mark.parametrize("given", ["max_user_namespaces=0", "max_user_namespaces=-1", "max_user_namespaces=x",
                                   "../../kernel/pid_max=5", "max_user_namespaces", "=5", "MAX_X=5"])
def test_a_ucount_that_is_not_a_proc_sys_user_name_and_a_positive_limit_is_refused(given, capsys):
    with pytest.raises(SystemExit):
        spawner._parse(_args(3, "--ucount", given))
    assert "--ucount takes max_<count>=<positive limit>" in capsys.readouterr().err


def test_ucounts_are_parsed_into_names_and_limits():
    args = spawner._parse(_args(3, "--ucount", "max_user_namespaces=64", "--ucount", "max_inotify_instances=8"))
    assert dict(args.ucount) == {"max_user_namespaces": 64, "max_inotify_instances": 8}


def _pair():
    ours, theirs = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    return ours, theirs


class _ForkRefused(_Kernel):
    """A kernel whose every fork the commands' namespace refuses as a dying one does (ENOMEM),
    with its PID 1 exiting or not."""

    def __init__(self, *, reaper_exiting: bool) -> None:
        super().__init__()
        self.reaper_exiting = reaper_exiting

    def fork(self):
        self.calls.append(("fork",))
        raise OSError(errno.ENOMEM, "Cannot allocate memory")

    def exiting(self, pid):
        self.calls.append(("exiting", pid))
        return self.reaper_exiting


@pytest.mark.parametrize("exiting, reply", [(True, {"op": "gone", "id": 7}),
                                            (False, {"op": "exit", "id": 7, "status": 126, "signal": None})],
                         ids=["namespace-ending", "out-of-memory"])
def test_a_fork_the_commands_namespace_refused_as_its_pid1_ends_sends_the_command_to_a_fresh_one(exiting, reply):
    # Between PID 1's end and the spawner reaping it, the kernel refuses every fork into the
    # namespace with ENOMEM. That command never started: the server sends it to the fresh
    # spawner ("gone"). A refusal while PID 1 lives is the command's own: it could not start.
    ours, theirs = _pair()
    kernel = _ForkRefused(reaper_exiting=exiting)
    serving = spawner.Spawner(theirs, 4242, 1, kernel=kernel)
    r, w = os.pipe()
    serving.spawn({"op": "spawn", "id": 7, "argv": ["/bin/true"], "env": {}}, [w])
    message, _ = protocol.recv_message(ours)
    assert message == reply and ("exiting", 4242) in kernel.calls
    assert serving.commands == {}
    os.close(r)


def test_a_pid1_is_exiting_while_its_stat_flags_say_so(tmp_path):
    stat = tmp_path / "17" / "stat"
    stat.parent.mkdir()
    stat.write_text("17 (python3 (x)) S 1 17 17 0 -1 4194564 10 0 0 0 0 0 0 0 20 0 1 0 100 0 0\n")
    assert spawner.Kernel().exiting(17, proc=tmp_path) is True
    stat.write_text("17 (python3) S 1 17 17 0 -1 4194560 10 0 0 0 0 0 0 0 20 0 1 0 100 0 0\n")
    assert spawner.Kernel().exiting(17, proc=tmp_path) is False
    stat.unlink()
    assert spawner.Kernel().exiting(17, proc=tmp_path) is True, "a PID 1 already gone has ended"


def _args(fd: int, *extra: str) -> list[str]:
    return ["--fd", str(fd), "--shm-bytes", "1048576", "--uid", "1234", "--gid", "5678", "--hide", "/run/hls-control",
            *extra]


#: ``/proc/self/uid_map`` as uid 0 of a user namespace the host user 1045 made, as under pasta.
OWNERS_UID_MAP = "         0       1045          1\n"
#: ``/proc/self/uid_map`` in the host's own user namespace, as in a Docker container.
HOST_UID_MAP = "         0          0 4294967295\n"


def test_the_hosts_own_user_namespace_is_told_by_its_uid_map():
    assert spawner.in_host_user_namespace(HOST_UID_MAP)
    assert not spawner.in_host_user_namespace(OWNERS_UID_MAP)
    assert not spawner.in_host_user_namespace("         0          0 4294967295\n      5000  5000 1\n")


def test_in_the_hosts_own_user_namespace_the_spawner_executes_itself_again_as_root_of_one_it_makes(tmp_path):
    """Where its container runs in the host's own user namespace (Docker), the spawner first makes
    a namespace whose uid and gid 0 are its own ids and executes this file again in it with its own
    arguments and channel; under pasta's namespace (Slurm) it goes straight on."""
    uid_map = tmp_path / "uid_map"
    uid_map.write_text(HOST_UID_MAP)
    ours, theirs = _pair()
    fd = theirs.detach()
    kernel = _Kernel()
    assert spawner.main(_args(fd), kernel=kernel, uid_map=uid_map) == 1
    assert kernel.calls == [
        ("unshare", os.CLONE_NEWUSER),
        ("write", "/proc/self/setgroups", "deny"),
        ("write", "/proc/self/uid_map", f"0 {os.getuid()} 1\n"),
        ("write", "/proc/self/gid_map", f"0 {os.getgid()} 1\n"),
        ("execv", sys.executable, [sys.executable, "-I", str(SPAWNER.resolve()), *_args(fd)]),
    ]
    uid_map.write_text(OWNERS_UID_MAP)
    ours, theirs = _pair()
    kernel = _Kernel(fork_returns=4242)
    assert spawner.main(_args(theirs.detach()), kernel=kernel, uid_map=uid_map) == 1
    assert "execv" not in [c[0] for c in kernel.calls]
    assert kernel.calls[:3] == [("unshare", os.CLONE_NEWUSER), ("write", "/proc/self/setgroups", "deny"),
                                ("write", "/proc/self/uid_map", f"1234 {os.getuid()} 1\n")]


def test_a_mount_that_fails_ends_the_spawner_naming_it(capsys, tmp_path):
    """The mount namespace is set up before any reaper is forked, so a mount that fails ends the
    spawner before it serves anything."""
    ours, theirs = _pair()
    ours.shutdown(socket.SHUT_WR)
    kernel = _Kernel(fork_returns=0, fail_mount_on="/dev/shm")
    uid_map = tmp_path / "uid_map"
    uid_map.write_text(OWNERS_UID_MAP)
    assert spawner.main(_args(theirs.detach()), kernel=kernel, uid_map=uid_map) == 1
    assert "cannot mount tmpfs on /dev/shm" in capsys.readouterr().err
    assert ours.recv(1) == b"", "the channel closed without a ready message"


def test_a_non_positive_dev_shm_cap_is_refused():
    with pytest.raises(SystemExit):
        spawner.main(["--fd", "0", "--shm-bytes", "0", "--uid", "1", "--gid", "1"])


# The serving loop, run for real without namespaces.


class _Running:
    """A spawner process and our end of its channel. With *prelude*, the spawner runs as a
    program that first runs that code, with the spawner module imported, and then calls its
    ``main`` with ``KERNEL`` if the prelude defines one."""

    def __init__(self, tmp_path: Path, *, prelude: str | None = None, left_open: tuple[int, ...] = ()) -> None:
        self.sock, theirs = _pair()
        self.err = tmp_path / "spawner.err"
        program = [str(SPAWNER)] if prelude is None else ["-c", _wrapped(prelude)]
        # *left_open* are descriptors the spawner's parent leaves open to it, as a launcher might.
        self.proc = subprocess.Popen([sys.executable, "-I", *program, *_args(theirs.fileno(), "--no-namespaces")],
                                     pass_fds=(theirs.fileno(), *left_open), env={}, stdin=subprocess.PIPE,
                                     stderr=self.err.open("w"))
        theirs.close()
        self.sock.settimeout(30)
        self.ready, _ = protocol.recv_message(self.sock)

    def spawn(self, ident: int, script: str, *, env: dict[str, str] | None = None, cwd: str | None = None):
        r, w = os.pipe()
        protocol.send_message(self.sock, {"op": "spawn", "id": ident, "argv": ["/bin/bash", "-c", script],
                                          "cwd": cwd, "env": env if env is not None else {"PATH": "/usr/bin:/bin"}},
                              fds=(w,))
        os.close(w)
        return r

    def next(self) -> dict:
        message, _ = protocol.recv_message(self.sock)
        return message

    def run(self, ident: int, script: str, **kw) -> tuple[dict, dict, str]:
        r = self.spawn(ident, script, **kw)
        started = self.next()
        out = _read_all(r).decode()
        return started, self.next(), out

    def reaper_pid(self) -> int:
        return self.ready["reaper_pid"]

    def close(self) -> int:
        self.sock.close()
        return self.proc.wait(10)


def _wrapped(prelude: str) -> str:
    return (f"import sys\nsys.path.insert(0, {str(SPAWNER.parent)!r})\nimport spawner\nKERNEL = None\n{prelude}\n"
            "sys.exit(spawner.main(sys.argv[1:], kernel=KERNEL))\n")


def _read_all(fd: int, timeout: float = 30.0) -> bytes:
    """Everything written to *fd* until its last writer closes it; fails rather than hangs."""
    deadline, chunks = time.monotonic() + timeout, []
    try:
        while True:
            ready, _, _ = select.select([fd], [], [], max(0.0, deadline - time.monotonic()))
            assert ready, "the command's output was never closed"
            chunk = os.read(fd, 65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


@pytest.fixture
def running(tmp_path):
    r = _Running(tmp_path)
    yield r
    if r.proc.poll() is None:
        r.proc.kill()
        r.proc.wait()


def _children(pid: int) -> list[int]:
    return [int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]


def _stat(pid: int) -> list[str] | None:
    """The fields of process *pid*'s ``/proc/<pid>/stat`` after its command name, from its state
    on; None once it is gone. A process that ends between the open and the read fails the read
    with ESRCH, so a scan of /proc meets that under load."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return text.rsplit(")", 1)[1].split()


def _state(pid: int) -> str | None:
    fields = _stat(pid)
    return None if fields is None else fields[0]


def _ppid(pid: int) -> int | None:
    fields = _stat(pid)
    return None if fields is None else int(fields[1])


def _pgrp(pid: int) -> int | None:
    fields = _stat(pid)
    return None if fields is None else int(fields[2])


def _alive(pid: int) -> bool:
    return _state(pid) not in (None, "Z")


def test_ready_names_the_spawner_its_pid_namespace_and_the_reaper(running):
    """The reaper is a child of the spawner; without namespaces they share the one PID namespace."""
    [reaper] = _children(running.proc.pid)
    assert running.ready == {"op": "ready", "pid": running.proc.pid, "reaper_pid": reaper,
                             "pid_ns": os.stat(f"/proc/{running.proc.pid}/ns/pid").st_ino}


def _signals(pid: int) -> tuple[int, int]:
    """The masks of the signals process *pid* has a handler for, but the C library's own
    (``LIBC_RESERVED``), and of those it blocks."""
    status = dict(line.split(":", 1) for line in Path(f"/proc/{pid}/status").read_text().splitlines())
    return int(status["SigCgt"], 16) & ~LIBC_RESERVED, int(status["SigBlk"], 16)


def test_the_reaper_handles_no_signal(running):
    """PID 1's kernel protection drops every signal a command sends it that it neither handles nor
    blocks. Python handles SIGINT by default, so the reaper puts it, and the other catchable
    killers, back to the default; only an uncatchable SIGKILL can end it, which just starts a
    fresh namespace. SIGCHLD it blocks rather than handles, so one a command sends only ends a
    wait for a child early."""
    assert _signals(running.reaper_pid())[0] == 0


def test_the_reaper_drops_every_handler_it_inherits_from_the_spawner(tmp_path):
    """The reaper is a fork of the spawner, so it starts with whatever handlers the spawner has.
    It puts each catchable killer back to the default, so no signal a command sends runs harness
    code in PID 1."""
    killers = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
    handlers = (f"import signal\nfor s in {[int(k) for k in killers]}:\n"
                "    signal.signal(s, lambda *_: sys.exit(1))\n")
    running = _Running(tmp_path, prelude=handlers)
    mask = sum(1 << (k - 1) for k in killers)
    try:
        assert _signals(running.proc.pid)[0] & mask == mask, "the spawner itself handles them"
        assert _signals(running.reaper_pid())[0] == 0
    finally:
        running.proc.kill()
        running.proc.wait()


def _full_pipe() -> tuple[int, int, int]:
    """A pipe with no room left, so a write to it waits until the read end is drained. Returns
    both ends and the number of bytes that filled it."""
    r, w = os.pipe()
    os.set_blocking(w, False)
    filled = 0
    try:
        while True:
            filled += os.write(w, b"\0" * 65536)
    except BlockingIOError:
        pass
    os.set_blocking(w, True)
    return r, w, filled


def test_the_reaper_sets_up_its_signals_before_it_reports_ready():
    """Ready means set up: the spawner starts commands once the reaper reports ready, and the
    tests read its signals then. Its ready pipe is full here, so the report waits until the test
    drains it: the reaper handles no signal and blocks SIGCHLD alone while its report is still
    held. (Not later: while it waits for a SIGCHLD, the kernel shows that one unblocked.)"""
    set_up = (0, 1 << (signal.SIGCHLD - 1))
    ready_r, ready_w, filled = _full_pipe()
    program = (f"import sys\nsys.path.insert(0, {str(SPAWNER.parent)!r})\nimport spawner\n"
               f"spawner._reaper_main(spawner.Kernel(), {ready_w}, fake=True, ucounts=dict())\n")
    proc = subprocess.Popen([sys.executable, "-I", "-c", program], pass_fds=(ready_w,), env={})
    os.close(ready_w)
    try:
        deadline = time.monotonic() + 10
        while (signals := _signals(proc.pid)) != set_up and time.monotonic() < deadline:
            time.sleep(0.01)
        report = _read_all(ready_r)[filled:]
    finally:
        proc.kill()
        proc.wait()
    assert signals == set_up, "the reaper reported ready before its signals were set up"
    assert report == str(os.stat("/proc/self/ns/pid").st_ino).encode(), "the held write was the ready report"


def _ended_child_prelude(path: Path, hook: str) -> str:
    """A prelude in which *hook* calls ``ended_child()`` in the reaper: it gives the reaper a child
    that has ended and is not yet reaped, and writes the child's pid to *path*. Without namespaces
    no orphan comes to the reaper, so this stands in for one."""
    part = f"{path}.part"
    return ("import os\n"
            "def ended_child():\n"
            "    child = os.fork()\n"
            "    if child == 0:\n"
            "        os._exit(0)\n"
            "    os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)\n"
            f"    with open({part!r}, 'w') as f:\n"
            "        f.write(str(child))\n"
            f"    os.rename({part!r}, {str(path)!r})\n"
            f"{hook}")


#: Gives the reaper its ended child at its first step, before it has set up any signal.
BEFORE_SET_UP = ("class Kernel(spawner.Kernel):\n"
                 "    def prctl(self, option, value):\n"
                 "        super().prctl(option, value)\n"
                 "        if option == spawner.PR_SET_PDEATHSIG:\n"
                 "            ended_child()\n"
                 "KERNEL = Kernel()\n")

#: Gives the reaper its ended child the first time it finds it has no child, before it waits for
#: one: the moment an orphan can come to PID 1 and end in the namespaces.
BEFORE_THE_WAIT = ("spawner_pid, real_waitpid, given = os.getpid(), os.waitpid, []\n"
                   "def waitpid(pid, options):\n"
                   "    try:\n"
                   "        return real_waitpid(pid, options)\n"
                   "    except ChildProcessError:\n"
                   "        if os.getpid() != spawner_pid and not given:\n"
                   "            given.append(True)\n"
                   "            ended_child()\n"
                   "        raise\n"
                   "os.waitpid = waitpid\n")


def _wait_reaped(path: Path, parent: int) -> None:
    """Wait until the child of *parent* whose pid is written to *path* has been reaped."""
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, "the reaper was never given its child"
        time.sleep(0.01)
    child = int(path.read_text())
    while _state(child) == "Z" and _ppid(child) == parent:
        assert time.monotonic() < deadline, "the reaper left a child that ended a zombie"
        time.sleep(0.01)


@pytest.mark.parametrize("hook", [BEFORE_SET_UP, BEFORE_THE_WAIT], ids=["before-set-up", "before-the-wait"])
def test_a_child_that_ends_while_the_reaper_is_not_waiting_is_reaped(tmp_path, hook):
    """An orphan can come to PID 1 and end at any moment, also while the reaper is not waiting
    for one: before its signals are set up, or after it found it had no child and before it
    waits. The reaper still reaps it, and does not wait for a SIGCHLD that came already."""
    child = tmp_path / "child"
    running = _Running(tmp_path, prelude=_ended_child_prelude(child, hook))
    try:
        _wait_reaped(child, running.reaper_pid())
    finally:
        running.proc.kill()
        running.proc.wait()


def test_the_reaper_holds_no_descriptor_but_its_three(tmp_path):
    """Not the spawner's end of the channel, which it would otherwise keep from the fork: PID 1 is
    the one harness process a command can name, so it holds nothing a command could use. The
    reaper is made dumpable here only so this test may list its descriptors."""
    dumpable = ("class Kernel(spawner.Kernel):\n"
                "    def prctl(self, option, value):\n"
                "        if option != spawner.PR_SET_DUMPABLE:\n"
                "            super().prctl(option, value)\n"
                "KERNEL = Kernel()\n")
    running = _Running(tmp_path, prelude=dumpable)
    try:
        reaper = running.reaper_pid()
        assert sorted(os.listdir(f"/proc/{reaper}/fd"), key=int) == ["0", "1", "2"]
    finally:
        running.proc.kill()
        running.proc.wait()


def test_a_command_is_started_its_output_piped_and_its_status_reported(running):
    started, ended, out = running.run(7, "echo out; echo err >&2; exit 3")
    assert started["op"] == "started" and started["id"] == 7 and started["pid"] > 0
    assert ended == {"op": "exit", "id": 7, "status": 3, "signal": None}
    assert out == "out\nerr\n"


def test_a_command_a_signal_ends_is_reported_with_the_signal(running):
    _, ended, _ = running.run(1, "kill -9 $$")
    assert ended == {"op": "exit", "id": 1, "status": None, "signal": signal.SIGKILL}


def test_the_command_gets_exactly_the_environment_sent(running):
    _, _, out = running.run(1, "exec env -0", env={"ONLY": "1", "PATH": "/usr/bin:/bin"})
    env = dict(kv.split("=", 1) for kv in out.split("\0") if kv)
    assert {k: v for k, v in env.items() if k not in BASH_OWN} == {"ONLY": "1", "PATH": "/usr/bin:/bin"}


def test_the_command_runs_in_its_own_session_with_oom_score_adj_1000_and_stdin_null(running):
    _, _, out = running.run(1, "cat; cat /proc/self/oom_score_adj; echo $$; awk '{print $6}' /proc/$$/stat")
    adj, pid, sid = out.split()
    assert adj == str(protocol.OOM_SCORE_ADJ)
    assert sid == pid, "the command leads a session of its own"
    own = Path(f"/proc/{running.proc.pid}/oom_score_adj").read_text().strip()
    assert own == Path("/proc/self/oom_score_adj").read_text().strip(), "the spawner keeps its own score"


def test_sigpipe_and_sigxfsz_are_at_their_defaults_in_the_command(running):
    _, _, out = running.run(1, "grep SigIgn /proc/$$/status")
    ignored = int(out.split()[1], 16)
    for sig in (signal.SIGPIPE, signal.SIGXFSZ):
        assert not ignored & (1 << (sig - 1)), f"{sig.name} is ignored in the command"


def test_a_command_inherits_no_descriptor_its_spawners_parent_left_open(tmp_path):
    """The spawner closes every descriptor above 2 before it runs a command, whatever its own
    parent left open. Read from outside once the command has become ``sleep``, which opens
    nothing itself (a shell may: bash opens the SSSD cache through NSS on a host that has one)."""
    planted_path = tmp_path / "left-open"
    planted = os.open(planted_path, os.O_RDONLY | os.O_CREAT, 0o600)
    os.set_inheritable(planted, True)
    running = _Running(tmp_path, left_open=(planted,))
    os.close(planted)
    pid = None
    try:
        held = {os.readlink(f"/proc/{running.proc.pid}/fd/{fd}") for fd in os.listdir(f"/proc/{running.proc.pid}/fd")}
        assert str(planted_path) in held, "the spawner was given the descriptor"
        r = running.spawn(1, "exec sleep 30")
        pid = running.next()["pid"]
        # Read once it is blocked in its sleep, past exec and the loader's opening of its
        # libraries, which a cmdline that already says "sleep" does not show.
        deadline = time.monotonic() + 10
        while "nanosleep" not in Path(f"/proc/{pid}/wchan").read_text():
            assert time.monotonic() < deadline, "the command never blocked in its sleep"
            time.sleep(0.01)
        assert Path(f"/proc/{pid}/cmdline").read_bytes() == b"sleep\0" + b"30\0"
        fds = {fd: os.readlink(f"/proc/{pid}/fd/{fd}") for fd in os.listdir(f"/proc/{pid}/fd")}
        assert sorted(fds, key=int) == ["0", "1", "2"], fds
        assert str(planted_path) not in fds.values()
        os.close(r)
    finally:
        if pid is not None:
            os.kill(pid, signal.SIGKILL)
        running.proc.kill()
        running.proc.wait()


def test_the_command_starts_in_the_working_directory_asked_for(running, tmp_path):
    _, _, out = running.run(1, "pwd", cwd=str(tmp_path))
    assert out.strip() == str(tmp_path)


def test_a_working_directory_that_cannot_be_entered_ends_the_command_with_127(running):
    _, ended, out = running.run(1, "pwd", cwd="/no/such/dir")
    assert ended["status"] == 127 and "cannot start the command" in out


def test_orphans_come_to_the_spawner_and_are_reaped_without_a_report(running):
    _, ended, out = running.run(1, "sleep 1 >/dev/null 2>&1 & echo $!; exit 0")
    orphan = int(out)
    assert ended["status"] == 0
    assert _ppid(orphan) == running.proc.pid, "the orphan was handed to the spawner"
    deadline = time.monotonic() + 5
    while Path(f"/proc/{orphan}").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not Path(f"/proc/{orphan}").exists(), "the orphan was reaped, not left a zombie"
    _, again, _ = running.run(2, "true")
    assert again == {"op": "exit", "id": 2, "status": 0, "signal": None}, "nothing was reported for the orphan"


def test_closing_the_channel_kills_every_commands_process_group_and_ends_the_spawner(running):
    r = running.spawn(1, "sleep 60 & sleep 60")
    started = running.next()
    deadline = time.monotonic() + 5
    group: list[int] = []
    while len(group) < 3 and time.monotonic() < deadline:
        group = [int(p) for p in os.listdir("/proc") if p.isdigit() and _pgrp(int(p)) == started["pid"]]
        time.sleep(0.05)
    assert len(group) == 3, "bash and its two sleeps"
    assert running.close() == 0
    deadline = time.monotonic() + 5
    while any(_alive(p) for p in group) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not any(_alive(p) for p in group)
    os.close(r)


def test_when_the_reaper_dies_inflight_commands_are_reported_killed_and_the_spawner_exits_recoverable(running):
    """A command that ends PID 1 (here the reaper is killed directly, since without namespaces a
    command cannot reach it) loses only its own commands: each in-flight one is reported killed by
    SIGKILL, and the spawner exits a code that tells the server to serve the next from a fresh
    namespace, not a sandbox-ended code."""
    r = running.spawn(1, "sleep 60")
    started = running.next()
    assert _alive(started["pid"])
    os.kill(running.reaper_pid(), signal.SIGKILL)
    assert running.next() == {"op": "exit", "id": 1, "status": None, "signal": int(signal.SIGKILL)}
    assert running.proc.wait(10) == protocol.REAPER_GONE_EXIT
    deadline = time.monotonic() + 5
    while _alive(started["pid"]) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(started["pid"]), "the command was killed with the reaper"
    os.close(r)


#: A prelude that holds each command's child for a second after the fork, before it makes its own
#: session, as a loaded machine can: the window in which the command is started and its process
#: group is not yet one to signal.
SLOW_CHILD = ("real = spawner._child\n"
              "def _child(*args, **kw):\n"
              "    import time\n"
              "    time.sleep(1.0)\n"
              "    real(*args, **kw)\n"
              "spawner._child = _child\n")


def test_a_command_the_reaper_dies_under_before_it_has_its_own_session_is_still_killed(tmp_path):
    """The started message goes out as soon as the command is forked, before the child has made
    its own session. A reaper that dies in that window still loses its command: the spawner kills
    the command itself as well as its process group, so it reports the command killed at once,
    never waiting for it to end on its own."""
    running = _Running(tmp_path, prelude=SLOW_CHILD)
    try:
        r = running.spawn(1, "sleep 60")
        started = running.next()
        os.kill(running.reaper_pid(), signal.SIGKILL)
        assert running.next() == {"op": "exit", "id": 1, "status": None, "signal": int(signal.SIGKILL)}
        assert running.proc.wait(10) == protocol.REAPER_GONE_EXIT
        assert not _alive(started["pid"]), "the command was killed with the reaper"
        os.close(r)
    finally:
        if running.proc.poll() is None:
            running.proc.kill()
            running.proc.wait()


def test_a_command_started_just_before_the_server_left_is_killed_with_the_rest(tmp_path):
    """The server leaving ends the spawner, which kills every command it started, one still
    without its own session among them, and exits promptly."""
    running = _Running(tmp_path, prelude=SLOW_CHILD)
    try:
        r = running.spawn(1, "sleep 60")
        started = running.next()
        assert running.close() == 0
        assert not _alive(started["pid"]), "the command was killed when the server left"
        os.close(r)
    finally:
        if running.proc.poll() is None:
            running.proc.kill()
            running.proc.wait()


def _wait_state(pid: int, state: str) -> None:
    deadline = time.monotonic() + 5
    while _state(pid) != state:
        assert time.monotonic() < deadline, f"process {pid} never reached state {state}"
        time.sleep(0.01)


def test_a_command_that_ended_beside_the_reaper_keeps_its_own_status(running):
    """A command that ended on its own just as the reaper died, both collected in one pass of the
    spawner's reaping, is reported with its own status, not as killed with the namespace. The
    spawner is stopped while both end, so one pass finds both."""
    r = running.spawn(1, "sleep 0.5; exit 3")
    started = running.next()
    os.kill(running.proc.pid, signal.SIGSTOP)
    try:
        _wait_state(started["pid"], "Z")
        os.kill(running.reaper_pid(), signal.SIGKILL)
        _wait_state(running.reaper_pid(), "Z")
    finally:
        os.kill(running.proc.pid, signal.SIGCONT)
    assert running.next() == {"op": "exit", "id": 1, "status": 3, "signal": None}
    assert running.proc.wait(10) == protocol.REAPER_GONE_EXIT
    os.close(r)


def test_the_reaper_dies_with_the_spawner(running):
    """PDEATHSIG ties the reaper to the spawner, so it never outlives the process that serves it."""
    reaper = running.reaper_pid()
    running.proc.kill()
    running.proc.wait()
    deadline = time.monotonic() + 5
    while _alive(reaper) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(reaper)


def test_a_spawn_without_its_output_descriptor_ends_the_spawner(running):
    protocol.send_message(running.sock, {"op": "spawn", "id": 1, "argv": ["/bin/true"], "cwd": None, "env": {}})
    assert running.proc.wait(10) == 1
    assert "came with 0 descriptors" in running.err.read_text()


def test_an_unknown_message_ends_the_spawner(running):
    protocol.send_message(running.sock, {"op": "kill", "id": 1})
    assert running.proc.wait(10) == 1
    assert "unknown message 'kill'" in running.err.read_text()


# --- the commands' seccomp filter -----------------------------------------------------------------

#: A command that prints what its prctl calls return: PR_SET_DUMPABLE to 0 (and why it failed),
#: PR_GET_DUMPABLE, PR_SET_DUMPABLE to 1 and PR_GET_NO_NEW_PRIVS.
_PRCTL = ("import ctypes, os; libc = ctypes.CDLL(None, use_errno=True); u = ctypes.c_ulong; "
          "r = libc.prctl(4, u(0), u(0), u(0), u(0)); e = ctypes.get_errno(); "
          "print(r, os.strerror(e) if r else '-', libc.prctl(3, u(0), u(0), u(0), u(0)), "
          "libc.prctl(4, u(1), u(0), u(0), u(0)), libc.prctl(39, u(0), u(0), u(0), u(0)))")


def test_a_command_cannot_make_itself_non_dumpable_and_runs_with_no_new_privileges(running):
    """A non-dumpable process hides its open files from the host's disk count. Every command runs
    under the spawner's seccomp filter, which refuses prctl(PR_SET_DUMPABLE, 0) with EPERM and
    lets every other call through, prctl's other uses among them; and with no_new_privs, which
    the filter needs and which keeps an exec from gaining what would make the process
    non-dumpable too."""
    _, ended, out = running.run(1, f"{sys.executable} -I -c {shlex.quote(_PRCTL)}")
    assert ended["status"] == 0, out
    assert out.split() == ["-1", "Operation", "not", "permitted", "1", "0", "1"]


def test_the_filter_has_numbers_for_the_machines_the_sites_run():
    for machine in ("x86_64", "aarch64"):
        program = spawner.dumpable_filter(machine)
        assert program and len(program) % 8 == 0
    with pytest.raises(spawner.SpawnerError, match="sparc64"):
        spawner.dumpable_filter("sparc64")


def test_a_spawner_whose_filter_the_kernel_refuses_never_serves(tmp_path):
    """A filter the kernel will not take leaves commands able to hide their files: the spawner
    ends before it says it is ready, the machinery's failure, which the server reads as such."""
    refusing = ("class Kernel(spawner.Kernel):\n"
                "    def seccomp(self, program):\n"
                "        raise OSError(22, 'Invalid argument')\n"
                "KERNEL = Kernel()\n")
    mine, theirs = _pair()
    proc = subprocess.Popen([sys.executable, "-I", "-c", _wrapped(refusing), *_args(theirs.fileno(), "--no-namespaces")],
                            pass_fds=(theirs.fileno(),), env={}, stderr=subprocess.PIPE)
    theirs.close()
    assert proc.wait(30) == 1
    assert b"seccomp" in proc.stderr.read()
    # It closed the channel without a ready message.
    mine.settimeout(5)
    assert protocol.recv_message(mine) is None
    mine.close()
