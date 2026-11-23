"""The spawner: it serves from outside the commands' PID namespace and forks each one into it.

The exec server starts it once, as ``/usr/bin/python3 -I spawner.py --fd N --shm-bytes B --uid U
--gid G [--hide PATH]... [--command-nproc C] [--ucount NAME=LIMIT]...``, with one end of a
socketpair on descriptor N and an
empty environment.
With namespaces (always, outside tests) it:
0. runs as uid 0 of a user namespace whose uid and gid 0 are the host user's. Under the seal on
   Slurm the container already does: it runs as uid 0 of pasta's namespace. A Docker container
   runs in the host's own namespace, and there the spawner makes that namespace first, maps its
   own ids to 0 in it and executes this file again in it (``enter_owners_namespace``), keeping
   its descriptors. The kernel makes a non-dumpable process's ``/proc/<pid>`` files the property
   of uid 0 of the namespace its memory belongs to, the host's root when that is the host's own
   namespace: so only this way does the reaper, forked from the spawner, show the host user its
   descriptors (the probe's ``reaper-no-socket``) on Docker as on Slurm;
1. enters a new user namespace that maps the ids it has in the container to *U* and *G*. The
   stage passes the host user's ids, never 0, so a command is an ordinary user inside the
   namespace and keeps no capability across ``execve``, also under the seal, where the container
   runs as uid 0 of pasta's namespace, and a mount namespace of its own;
2. sets up the mount namespace: private propagation, a tmpfs of *B* bytes on /dev/shm, and an
   empty read-only tmpfs over each ``--hide`` path (the control directory). Not /proc, which
   needs the new PID namespace;
3. unshares a PID namespace and forks a bare reaper. The reaper is PID 1 of that namespace; it
   closes every descriptor it inherited but the standard three (the channel among them), writes
   each ``--ucount NAME=LIMIT`` to ``/proc/sys/user/NAME``, the commands' user namespace's own
   limit of that per-uid count, which it may set holding every capability there
   (``hls.substrate.ucounts``), mounts a /proc of its own (so a command sees only the namespace's
   processes and none of the harness), becomes non-dumpable, hands its PID namespace's inode
   back, and then only reaps, so a contestant that ends it loses nothing but its own commands. A
   limit it cannot write keeps it from coming up.

Once its reaper is up, the spawner sets no_new_privs on itself and installs a seccomp filter
(``dumpable_filter``) that every command it forks inherits and none can remove: it refuses
``prctl(PR_SET_DUMPABLE, 0)`` with EPERM and lets every other call through. A non-dumpable
process hides its open files from the host's disk count, and no_new_privs keeps an exec from
gaining what would make one non-dumpable too. A command that runs a program from a file it cannot
read is non-dumpable all the same; the host counts such a process's files as past its role's disk
cap (``hls.substrate.memory_watch``). The reaper is forked before the filter,
and makes itself non-dumpable. A filter the kernel refuses ends the spawner before it says it is
ready.

The spawner itself stays in the container's PID namespace (``unshare`` moves only its children),
so no command can see, signal or limit it: a command's own PID namespace holds PID 1 (the reaper)
and the sibling commands, nothing of the harness. The spawner forks every command after the
unshare, so each lands in the reaper's namespace and inherits the spawner's own resource limits,
never the reaper's. It reaps its command children; the reaper reaps orphans reparented to PID 1.

Then it serves messages (``exec_protocol.send_message``) until the server closes the channel:
- ``{"op": "spawn", "id", "argv", "cwd", "env"}`` with one descriptor, the write end of the
  command's output pipe. The command starts in a session of its own with ``oom_score_adj``
  1000, stdin from /dev/null, stdout and stderr on the pipe, SIGPIPE and SIGXFSZ at their
  defaults (as ``subprocess`` leaves them), no other descriptor, and exactly the environment
  sent. With ``--command-nproc C`` its ``RLIMIT_NPROC``, soft and hard, is C, set in the child
  before it becomes the command, so the command's forks fail past the ceiling
  (``hls.substrate.nproc``) while the spawner keeps its own limit and forks the next. The reply is ``{"op": "started", "id", "pid"}``, then, once it is reaped,
  ``{"op": "exit", "id", "status", "signal"}``: its exit status, or the number of the signal
  that ended it.
When the reaper dies, the kernel kills every command in its namespace; the spawner reports each
in-flight command as killed by SIGKILL and exits ``REAPER_GONE_EXIT``, so the server can serve the
next command from a fresh namespace (a second ``unshare(CLONE_NEWPID)`` from one process is
refused, so a fresh namespace needs a fresh spawner). On start it sends ``{"op": "ready", "pid",
"pid_ns", "reaper_pid"}``, where *pid_ns* is the inode of the reaper's PID namespace.

``--no-namespaces`` skips the namespaces and makes the spawner the subreaper of its commands, with
a bare child standing in for the reaper: it exists for the tests, on machines that refuse
unprivileged user namespaces."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import re
import resource
import selectors
import signal
import struct
import socket
import sys
from pathlib import Path

if __package__:
    from hls.agent import exec_protocol as protocol
else:
    # Run as a file in the sandbox, where hls is not installed: the protocol file sits beside
    # this one in the same read-only directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import exec_protocol as protocol

# Mount flags and prctl options, from <sys/mount.h> and <linux/prctl.h>.
MS_RDONLY, MS_NOSUID, MS_NODEV, MS_NOEXEC = 0x1, 0x2, 0x4, 0x8
MS_REC, MS_PRIVATE = 0x4000, 0x40000
PR_SET_DUMPABLE, PR_SET_PDEATHSIG, PR_SET_CHILD_SUBREAPER = 4, 1, 36
# prctl options and the seccomp mode, from <linux/prctl.h> and <linux/seccomp.h>.
PR_SET_NO_NEW_PRIVS, PR_SET_SECCOMP, SECCOMP_MODE_FILTER = 38, 22, 2
#: For each machine (``os.uname().machine``) the commands may run on: the audit arch of its own
#: system calls and prctl's number among them, then the same for the older ABI a process there can
#: still call the kernel through (i386 on x86_64, arm on aarch64), from <linux/audit.h> and each
#: ABI's syscall table.
PRCTL_NUMBERS = {"x86_64": ((0xC000003E, 157), (0x40000003, 172)),
                 "aarch64": ((0xC00000B7, 167), (0x40000028, 172))}
#: The error ``prctl(PR_SET_DUMPABLE, 0)`` fails with under the commands' filter, which the brief
#: names (``hls.prompt.sandbox_brief_section``).
DUMPABLE_REFUSAL = errno.EPERM
#: The flag of a task on its way out, from <linux/sched.h>, in /proc/<pid>/stat's ninth field.
PF_EXITING = 0x4
#: The mount options of the tmpfs that covers a hidden path.
HIDE_OPTIONS = "size=4k,mode=000"
#: Signals the reaper puts back to their default, so the kernel's PID 1 protection drops them
#: when a command sends them: only an uncatchable SIGKILL can end the reaper, and that only
#: costs the contestant its own commands.
REAPER_DEFAULT_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT)
#: The names a ``--ucount`` may give: a file of /proc/sys/user, never a path.
UCOUNT_NAME = re.compile(r"max_[a-z]+(_[a-z]+)*")


class SpawnerError(RuntimeError):
    """The namespace could not be set up, or the reaper did not come up."""


class _SockFprog(ctypes.Structure):
    """``struct sock_fprog``: a classic BPF program's length in instructions and its address."""

    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


def dumpable_filter(machine: str) -> bytes:
    """The seccomp program every command runs under, for *machine*: ``prctl(PR_SET_DUMPABLE, 0)``
    fails with EPERM, through the machine's own system calls or its older ABI, and every other call
    runs as it would. The x32 ABI's numbers are x86_64's with bit 30 set, so the native number is
    compared with that bit cleared. prctl's option is an int, so only its low word is compared.
    Raise SpawnerError for a machine with no numbers here."""
    try:
        (native_arch, native_nr), (compat_arch, compat_nr) = PRCTL_NUMBERS[machine]
    except KeyError:
        raise SpawnerError(f"the commands' seccomp filter has no numbers for machine {machine!r}") from None
    # Classic BPF opcodes, the offsets in struct seccomp_data and the filter's two answers.
    load, jump_eq, and_k, ret = 0x20, 0x15, 0x54, 0x06
    nr, arch, arg0, arg1 = 0, 4, 16, 24
    allow, eperm = 0x7FFF0000, 0x00050000 | DUMPABLE_REFUSAL
    # Each instruction: opcode, operand, and for a jump the labels it goes to when equal and not.
    program: list[tuple] = [
        ("arch", load, arch), (None, jump_eq, native_arch, "native", "compat"),
        ("compat", jump_eq, compat_arch, "compat_nr", "allow"),
        ("native", load, nr), (None, and_k, 0xBFFFFFFF), (None, jump_eq, native_nr, "check", "allow"),
        ("compat_nr", load, nr), (None, jump_eq, compat_nr, "check", "allow"),
        ("check", load, arg0), (None, jump_eq, PR_SET_DUMPABLE, "value", "allow"),
        ("value", load, arg1), (None, jump_eq, 0, "deny", "allow"),
        ("deny", ret, eperm), ("allow", ret, allow),
    ]
    at = {step[0]: i for i, step in enumerate(program) if step[0] is not None}
    code = b""
    for i, (_, op, k, *targets) in enumerate(program):
        jt, jf = (at[t] - i - 1 for t in targets) if targets else (0, 0)
        code += struct.pack("=HBBI", op, jt, jf, k)
    return code


def _c(text: str | None) -> bytes | None:
    return None if text is None else text.encode()


class Kernel:
    """The system calls the namespace setup makes, so tests can record them."""

    def __init__(self) -> None:
        self._libc = ctypes.CDLL(None, use_errno=True)

    def unshare(self, flags: int) -> None:
        os.unshare(flags)

    def write(self, path: str, text: str) -> None:
        with open(path, "w") as f:
            f.write(text)

    def mount(self, source: str, target: str, fstype: str | None, flags: int, data: str | None) -> None:
        if self._libc.mount(_c(source), _c(target), _c(fstype), ctypes.c_ulong(flags), _c(data)) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err), target)

    def execv(self, path: str, argv: list[str]) -> None:
        os.execv(path, argv)

    def prctl(self, option: int, value: int) -> None:
        if self._libc.prctl(option, ctypes.c_ulong(value), 0, 0, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

    def seccomp(self, program: bytes) -> None:
        """Install the seccomp *program* on this process, for it and every child it forks; the
        caller has set no_new_privs."""
        buf = ctypes.create_string_buffer(program, len(program))
        fprog = _SockFprog(len(program) // 8, ctypes.cast(buf, ctypes.c_void_p))
        if self._libc.prctl(PR_SET_SECCOMP, ctypes.c_ulong(SECCOMP_MODE_FILTER), ctypes.byref(fprog),
                            ctypes.c_ulong(0), ctypes.c_ulong(0)) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))

    def fork(self) -> int:
        return os.fork()

    def exiting(self, pid: int, *, proc: Path = Path("/proc")) -> bool:
        """Whether process *pid* (the reaper) is ending, or gone: its ``/proc/<pid>/stat`` flags
        hold ``PF_EXITING``, or it has no stat any more."""
        try:
            text = (proc / str(pid) / "stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return True
        # Fields after the name, which may hold spaces and parentheses: state first, flags seventh.
        return bool(int(text.rsplit(")", 1)[1].split()[6]) & PF_EXITING)


#: ``/proc/self/uid_map`` in the host's own user namespace: every uid, each as itself.
HOST_UID_MAP = ["0", "0", "4294967295"]


def in_host_user_namespace(uid_map: str) -> bool:
    """Whether *uid_map*, this process's ``/proc/self/uid_map``, is the host's own namespace's."""
    lines = uid_map.split("\n")
    return lines[0].split() == HOST_UID_MAP and not any(line.strip() for line in lines[1:])


def enter_owners_namespace(kernel: Kernel, *, fd: int, argv: list[str]) -> None:
    """Step 0, in the host's own user namespace: a user namespace whose uid and gid 0 are this
    process's ids, then this file executed again in it with *argv*, the channel on descriptor *fd*
    kept, as uid 0 there with every capability of it. Never returns."""
    outer_uid, outer_gid = os.getuid(), os.getgid()
    kernel.unshare(os.CLONE_NEWUSER)
    # An unprivileged process may write its gid map only after giving up setgroups.
    kernel.write("/proc/self/setgroups", "deny")
    kernel.write("/proc/self/uid_map", f"0 {outer_uid} 1\n")
    kernel.write("/proc/self/gid_map", f"0 {outer_gid} 1\n")
    os.set_inheritable(fd, True)
    kernel.execv(sys.executable, [sys.executable, "-I", str(Path(__file__).resolve()), *argv])
    raise SpawnerError("the spawner could not execute itself in its own user namespace")


def enter_user_and_mount_namespaces(kernel: Kernel, *, uid: int, gid: int) -> None:
    """Step 1: the user namespace, mapping this process's ids to *uid* and *gid*, then a mount
    namespace of its own. The PID namespace comes later, just before the reaper is forked."""
    if uid == 0 or gid == 0:
        raise SpawnerError("commands may not run as uid or gid 0 inside the namespace")
    outer_uid, outer_gid = os.getuid(), os.getgid()
    kernel.unshare(os.CLONE_NEWUSER)
    # An unprivileged process may write its gid map only after giving up setgroups.
    kernel.write("/proc/self/setgroups", "deny")
    kernel.write("/proc/self/uid_map", f"{uid} {outer_uid} 1\n")
    kernel.write("/proc/self/gid_map", f"{gid} {outer_gid} 1\n")
    kernel.unshare(os.CLONE_NEWNS)


def set_up_mounts(kernel: Kernel, *, shm_bytes: int, hide: list[str]) -> None:
    """Step 2, in the mount namespace: private mounts, the capped /dev/shm, and the hidden paths
    covered. The reaper mounts /proc, since that needs the new PID namespace."""
    steps = [("none", "/", None, MS_REC | MS_PRIVATE, None),
             ("tmpfs", "/dev/shm", "tmpfs", MS_NOSUID | MS_NODEV, f"size={shm_bytes},mode=1777")]
    steps += [("tmpfs", path, "tmpfs", MS_RDONLY | MS_NOSUID | MS_NODEV | MS_NOEXEC, HIDE_OPTIONS) for path in hide]
    for source, target, fstype, flags, data in steps:
        try:
            kernel.mount(source, target, fstype, flags, data)
        except OSError as e:
            raise SpawnerError(f"cannot mount {fstype or 'private propagation'} on {target}: {e.strerror}") from e


def enter_pid_namespace(kernel: Kernel) -> int:
    """Step 3's unshare and fork: the PID namespace, whose PID 1 is the forked child. Returns the
    child's pid in the spawner and 0 in the child (the reaper)."""
    kernel.unshare(os.CLONE_NEWPID)
    return kernel.fork()


def reaper_set_up(kernel: Kernel, ucounts: dict[str, int]) -> None:
    """What the reaper does as PID 1 before it reaps: each of *ucounts* written to the commands'
    user namespace's own ``/proc/sys/user`` file, which binds every command there; a /proc of the
    namespace's own, so a command sees only the namespace's processes; and non-dumpable, so a
    command cannot trace it or read its memory. The limits go through the /proc it inherited,
    before its own is mounted: through a /proc mounted in the commands' own namespaces the kernel
    refuses the write (EACCES under el8, where the inherited one takes it)."""
    for name, limit in ucounts.items():
        try:
            kernel.write(f"/proc/sys/user/{name}", str(limit))
        except OSError as e:
            raise SpawnerError(f"cannot limit the commands' {name} to {limit}: {e.strerror}") from e
    try:
        kernel.mount("proc", "/proc", "proc", MS_NOSUID | MS_NODEV | MS_NOEXEC, None)
    except OSError as e:
        raise SpawnerError(f"cannot mount proc on /proc: {e.strerror}") from e
    kernel.prctl(PR_SET_DUMPABLE, 0)


def _reaper_main(kernel: Kernel, ready_w: int, *, fake: bool, ucounts: dict[str, int]) -> None:
    """The bare reaper, PID 1 of the commands' namespace. It sets itself up, hands its PID
    namespace's inode back over *ready_w*, and then only reaps. Never returns."""
    try:
        # Die when the spawner does, so a reaper never outlives the process that serves its
        # namespace (including the bare stand-in used without namespaces).
        kernel.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL))
        # Keep only the standard three and the ready pipe, never the spawner's end of the
        # channel, which the fork copied: the reaper is the one harness process a command can name.
        os.closerange(3, ready_w)
        os.closerange(ready_w + 1, os.sysconf("SC_OPEN_MAX"))
        if fake:
            # No namespace here: a plain child stands in for PID 1 so the serving loop and the
            # recovery can be tested where unprivileged user namespaces are refused.
            kernel.prctl(PR_SET_DUMPABLE, 0)
        else:
            reaper_set_up(kernel, ucounts)
        pid_ns = os.stat("/proc/self/ns/pid").st_ino
        for sig in REAPER_DEFAULT_SIGNALS:
            signal.signal(sig, signal.SIG_DFL)
        # SIGCHLD is blocked and never handled, the only signal blocked: one that comes while the
        # reaper is not waiting stays pending and ends its next wait at once.
        signal.pthread_sigmask(signal.SIG_SETMASK, {signal.SIGCHLD})
        os.write(ready_w, str(pid_ns).encode())
        os.close(ready_w)
    except BaseException as e:  # noqa: BLE001 -- the reaper must never return into the serving code
        os.write(2, f"{protocol.LOG_PREFIX}reaper: {e}\n".encode(errors="replace"))
        os._exit(1)
    while True:
        try:
            os.waitpid(-1, 0)
        except ChildProcessError:
            signal.sigwait({signal.SIGCHLD})
        except OSError:
            os._exit(0)


def _fork_reaper(kernel: Kernel, *, fake: bool, ucounts: dict[str, int]) -> tuple[int, int]:
    """Fork the reaper, which writes *ucounts* (``reaper_set_up``), and return its pid and the
    inode of its PID namespace. Raise SpawnerError if it did not come up."""
    ready_r, ready_w = os.pipe()
    pid = kernel.fork() if fake else enter_pid_namespace(kernel)
    if pid == 0:
        os.close(ready_r)
        _reaper_main(kernel, ready_w, fake=fake, ucounts=ucounts)
    os.close(ready_w)
    try:
        data = b""
        while chunk := os.read(ready_r, 64):
            data += chunk
    finally:
        os.close(ready_r)
    if not data:
        raise SpawnerError("the reaper did not come up")
    return pid, int(data)


def install_command_filter(kernel: Kernel) -> None:
    """Set no_new_privs on this process and install the commands' seccomp filter
    (``dumpable_filter``), which every command forked after inherits. Raise SpawnerError when
    either is refused."""
    try:
        kernel.prctl(PR_SET_NO_NEW_PRIVS, 1)
        kernel.seccomp(dumpable_filter(os.uname().machine))
    except OSError as e:
        raise SpawnerError(f"the commands' seccomp filter could not be installed: {e}") from e


def _child(argv: list[str], cwd: str | None, env: dict[str, str], out: int, nproc: int | None) -> None:
    """In the forked child: become the command, under the process ceiling *nproc* when one is
    given. Never returns."""
    try:
        os.setsid()
        if nproc is not None:
            resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))
        with open(protocol.OWN_OOM_SCORE, "w") as f:
            f.write(str(protocol.OOM_SCORE_ADJ))
        # Python ignores these, and an ignored signal stays ignored across execve.
        for sig in (signal.SIGPIPE, signal.SIGXFSZ):
            signal.signal(sig, signal.SIG_DFL)
        null = os.open(os.devnull, os.O_RDONLY)
        os.dup2(null, 0)
        os.dup2(out, 1)
        os.dup2(out, 2)
        os.closerange(3, os.sysconf("SC_OPEN_MAX"))
        if cwd:
            os.chdir(cwd)
        os.execve(argv[0], argv, env)
    except BaseException as e:  # noqa: BLE001 -- the child must never return into the spawner's loop
        os.write(2, f"{protocol.LOG_PREFIX}cannot start the command: {e}\n".encode(errors="replace"))
    os._exit(127)


def _kill_command(pid: int) -> None:
    """SIGKILL the command *pid* and its process group. The command itself first: one forked just
    now may not have made its own session yet, so its group is not yet one to signal, and killed
    before it does it starts nothing. One already in its own session leaves its group behind it,
    which the second signal reaches."""
    for send in (os.kill, os.killpg):
        try:
            send(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


class Spawner:
    """The serving loop: forks each command into the reaper's namespace and reports how it ended,
    and tells the server when the reaper is gone so a fresh namespace can serve the next command."""

    def __init__(self, sock: socket.socket, reaper_pid: int, pid_ns: int, *, nproc: int | None = None,
                 kernel: Kernel | None = None) -> None:
        #: The channel to the exec server.
        self.sock = sock
        #: The system calls a spawn makes.
        self.kernel = kernel or Kernel()
        #: Every command's process ceiling (``RLIMIT_NPROC``); None leaves the spawner's own.
        self.nproc = nproc
        #: The reaper's pid, in the spawner's own namespace.
        self.reaper_pid = reaper_pid
        #: The inode of the reaper's PID namespace, for the ready message.
        self.pid_ns = pid_ns
        #: Pid of each running command to its request id.
        self.commands: dict[int, int] = {}
        #: Set once the reaper has ended, which ends the serving loop.
        self.reaper_dead = False

    def spawn(self, message: dict, fds: list[int]) -> None:
        if len(fds) != 1:
            for fd in fds:
                os.close(fd)
            raise SpawnerError(f"spawn {message.get('id')} came with {len(fds)} descriptors, not 1")
        out = fds[0]
        try:
            pid = self.kernel.fork()
        except OSError as e:
            if e.errno == errno.ENOMEM and self.kernel.exiting(self.reaper_pid):
                # The reaper has ended and its namespace takes no new process until the spawner
                # reaps it: the command never started, and the server runs it in a fresh one.
                os.close(out)
                protocol.send_message(self.sock, {"op": "gone", "id": message["id"]})
                return
            os.write(out, f"{protocol.LOG_PREFIX}cannot fork the command: {e}\n".encode())
            os.close(out)
            protocol.send_message(self.sock, {"op": "exit", "id": message["id"], "status": 126, "signal": None})
            return
        if pid == 0:
            _child(message["argv"], message.get("cwd"), message["env"], out, self.nproc)
        os.close(out)
        self.commands[pid] = message["id"]
        protocol.send_message(self.sock, {"op": "started", "id": message["id"], "pid": pid})

    def reap(self) -> None:
        """Collect every ended child and tell the server how each command ended. Note when the
        reaper is the one that ended."""
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                return
            if pid == 0:
                return
            if pid == self.reaper_pid:
                self.reaper_dead = True
                continue
            ident = self.commands.pop(pid, None)
            if ident is None:
                continue
            code = os.waitstatus_to_exitcode(status)
            exit_status, sig = (None, -code) if code < 0 else (code, None)
            protocol.send_message(self.sock, {"op": "exit", "id": ident, "status": exit_status, "signal": sig})

    def handle_reaper_gone(self) -> None:
        """The reaper ended, so its namespace is gone: report each command still in flight as
        killed by SIGKILL, as the kernel kills them with the namespace."""
        for pid, ident in list(self.commands.items()):
            _kill_command(pid)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            protocol.send_message(self.sock, {"op": "exit", "id": ident, "status": None, "signal": int(signal.SIGKILL)})
        self.commands.clear()
        print(f"{protocol.LOG_PREFIX}reaper ended; its commands were killed, the next runs in a fresh namespace",
              file=sys.stderr, flush=True)

    def kill_all(self) -> None:
        """Kill every running command and its process group, and collect the commands."""
        for pid in list(self.commands):
            _kill_command(pid)
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
        self.commands.clear()

    def serve(self) -> int:
        wake_r, wake_w = os.pipe()
        os.set_blocking(wake_r, False)
        os.set_blocking(wake_w, False)
        old_handler = signal.signal(signal.SIGCHLD, lambda _sig, _frame: None)
        old_wakeup = signal.set_wakeup_fd(wake_w, warn_on_full_buffer=False)
        sel = selectors.DefaultSelector()
        sel.register(self.sock, selectors.EVENT_READ, "control")
        sel.register(wake_r, selectors.EVENT_READ, "child")
        protocol.send_message(self.sock, {"op": "ready", "pid": os.getpid(), "pid_ns": self.pid_ns,
                                          "reaper_pid": self.reaper_pid})
        try:
            while True:
                for key, _ in sel.select():
                    if key.data == "child":
                        while True:
                            try:
                                if not os.read(wake_r, 4096):
                                    break
                            except BlockingIOError:
                                break
                        self.reap()
                        if self.reaper_dead:
                            self.handle_reaper_gone()
                            return protocol.REAPER_GONE_EXIT
                        continue
                    received = protocol.recv_message(self.sock)
                    if received is None:
                        return 0
                    message, fds = received
                    if message.get("op") != "spawn":
                        raise SpawnerError(f"unknown message {message.get('op')!r}")
                    self.spawn(message, fds)
        finally:
            # The server is gone or the loop failed: nothing is reported any more.
            self.kill_all()
            signal.set_wakeup_fd(old_wakeup)
            signal.signal(signal.SIGCHLD, old_handler)
            sel.close()
            os.close(wake_r)
            os.close(wake_w)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="spawner.py")
    p.add_argument("--fd", type=int, required=True)
    p.add_argument("--shm-bytes", type=int, required=True)
    p.add_argument("--uid", type=int, required=True)
    p.add_argument("--gid", type=int, required=True)
    p.add_argument("--hide", action="append", default=[])
    p.add_argument("--command-nproc", type=int)
    p.add_argument("--ucount", action="append", default=[], type=_ucount)
    p.add_argument("--no-namespaces", action="store_true")
    args = p.parse_args(argv)
    if args.shm_bytes <= 0:
        p.error("--shm-bytes must be positive")
    if args.command_nproc is not None and args.command_nproc <= 0:
        p.error("--command-nproc must be positive")
    return args


def _ucount(text: str) -> tuple[str, int]:
    """A ``--ucount NAME=LIMIT``: a /proc/sys/user file's name and a positive limit."""
    name, _, limit = text.partition("=")
    if not UCOUNT_NAME.fullmatch(name) or not limit.isdecimal() or int(limit) <= 0:
        raise argparse.ArgumentTypeError(f"--ucount takes max_<count>=<positive limit>, not {text!r}")
    return name, int(limit)


def main(argv: list[str] | None = None, *, kernel: Kernel | None = None,
         uid_map: Path = Path("/proc/self/uid_map")) -> int:
    given = sys.argv[1:] if argv is None else argv
    args = _parse(given)
    sock = socket.socket(fileno=args.fd)
    kernel = kernel or Kernel()
    try:
        if args.no_namespaces:
            kernel.prctl(PR_SET_CHILD_SUBREAPER, 1)
            reaper_pid, pid_ns = _fork_reaper(kernel, fake=True, ucounts={})
        else:
            if in_host_user_namespace(uid_map.read_text()):
                enter_owners_namespace(kernel, fd=args.fd, argv=given)
            enter_user_and_mount_namespaces(kernel, uid=args.uid, gid=args.gid)
            set_up_mounts(kernel, shm_bytes=args.shm_bytes, hide=args.hide)
            reaper_pid, pid_ns = _fork_reaper(kernel, fake=False, ucounts=dict(args.ucount))
        # After the reaper, which makes itself non-dumpable; before any command.
        install_command_filter(kernel)
        return Spawner(sock, reaper_pid, pid_ns, nproc=args.command_nproc, kernel=kernel).serve()
    except (SpawnerError, OSError, ConnectionError, ValueError) as e:
        print(f"{protocol.LOG_PREFIX}spawner: {e}", file=sys.stderr, flush=True)
        return 1
    finally:
        sock.close()


if __name__ == "__main__":
    sys.exit(main())
