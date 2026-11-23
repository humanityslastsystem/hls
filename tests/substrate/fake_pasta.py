"""Stand-ins for what the network seal runs on the host, on machines without pasta: ``pasta``
itself, ``iptables``, ``ip6tables`` and ``sysctl``, and the interpreter that scans a host port and
runs the interceptor. With them, the real ``transparent_seal.sh`` runs end to end in a test,
in the test's own network namespace.

``install(dir)`` writes the five programs into *dir*. Each fake ``pasta`` call takes the next
line of ``$FAKE_PASTA_PLAN`` (``run`` once the file runs out) and appends one JSON line to
``$FAKE_PASTA_LOG`` with its own flags (everything before ``--``), which ``calls`` reads back and
``forwards`` reads the port forwards of:
- ``run``: run the seal's in-namespace block as given, as real pasta would after setting up
  the namespace;
- ``pass``: skip that block and do only what a working one does: write the interceptor's
  status file (``$FAKE_PROXY_ERRORS`` upstream failures, default 0, of them
  ``$FAKE_PROXY_MID_STREAM`` while a file streamed, default 0, the pages it refused for
  their size, the JSON list ``$FAKE_PROXY_OVERSIZED``, default none, the hosts it refused by its
  policy, the JSON list ``$FAKE_PROXY_POLICY_REFUSED``, default none, the fetches it refused past
  its cap, ``$FAKE_PROXY_REFUSED``, default 0, and a
  heartbeat ``$FAKE_PROXY_HEARTBEAT_AGE`` seconds old, default 0) and the started file,
  hand the payload the env file that came on standard input as descriptor 3 when
  ``HLS_SEAL_ENV_FD=3``, with standard input emptied, run the payload, and record its ending
  in ``$HLS_SEAL_ENDED_FILE.pending`` with the interceptor alive. For tests that need no
  listener. Every process of a real step dies with the step, and a local launcher stops only a
  role's first process, the seal's bash, so once its own parent has ended the fake kills the
  payload and exits 137 as well;
- ``pass_exit:<status>``: as ``pass``, but end with that status rather than the payload's, as a
  seal whose own last step failed after the payload would;
- ``die_after:<seconds>``: run the block as ``run`` does, but end with status 137 that many
  seconds after the payload started, leaving the block running, as real pasta killed by
  SIGKILL does (with the block's own status when it ends before the payload starts);
- ``fail:<status>``: exit with that status without running anything, as pasta does when it
  cannot open /dev/net/tun or bind a forwarded port.

Like real pasta, which closes every descriptor it inherited above stderr when it starts
(``isolate_fds``), a fake ``pasta`` closes them before it runs anything.

What a fake ``pasta`` runs gets ``FAKE_PASTA_NAMESPACE=1`` in its environment, standing for
the namespace real pasta would have made, so a test can tell what ran inside the seal, and
``FAKE_NETNS_RULES``, a file standing for that namespace's firewall, made afresh for each call
(``netns-<pid>.rules`` in the fakes' own directory): the fake ``iptables`` appends each rule it
installs there, one JSON line, and ``pass`` writes the ICMP rule a working block installs.
``icmp_dropped`` reads whether the namespace drops its ICMP, which a stand-in for what runs
inside it honours. Likewise ``FAKE_NETNS_SYSCTLS`` (``netns-<pid>.sysctls``) stands for the
namespace's own sysctls, a JSON object, which the fake ``sysctl`` reads and writes:
``-q -w <key>=<value>`` sets one, ``-n <key>`` prints it, a ``user.max_*`` key the namespace never
set reading as 2147483647, a fresh user namespace's own value. With ``$FAKE_SYSCTL_STUCK`` set a
write succeeds and changes nothing; with ``$FAKE_SYSCTL_FAIL`` set it fails as the kernel refuses
one; with ``$FAKE_SYSCTL_LOG`` set every call appends its arguments there, one JSON line. ``pass``
writes the limits of ``$HLS_SEAL_UCOUNTS`` as a working block does.

The fake interpreter, called as ``python -c <code>``, prints port 40000 + n for its n-th call
(counted in ``$FAKE_PYTHON_COUNTER``); called as ``python -m hls.egress.proxy`` it is the
interceptor, by ``$FAKE_INTERCEPTOR`` (a name the step wrapper keeps, unlike one ending in
``_proxy``):
- ``serve`` (the default): write its status file, once more when the payload starts, and listen on
  the seal's ports until killed;
- ``die``: exit at once;
- ``die_first``: exit at once on its first start (counted in ``$FAKE_INTERCEPTOR_STARTS``), serve
  on every later one;
- ``quit_after:<seconds>``: serve, then exit that many seconds after the payload started.
Called as ``python -m <module>`` for any other module of ``hls``, a harness role's (the cache's
copy role, the package stage's roles), it runs the test's own interpreter with the same
arguments, so the role runs as it is. Called any other way it prints that the call was
unexpected and exits 2, so a seal that ran an unexpected program fails loudly.

With ``$FAKE_PIPE_INODE`` set, the fake ``iptables``, ``ip6tables`` and interpreter append their
name to ``$FAKE_PIPE_HOLDERS`` when one of their descriptors is the pipe with that inode: the
env file's, which only the payload may hold. With ``$FAKE_IPTABLES_LOG`` set, the fake
``iptables`` appends its arguments to that file, one JSON line per rule, and with
``$FAKE_IP6TABLES_LOG`` set the fake ``ip6tables`` does the same there, so a test reads the
firewall the seal installed; with ``$FAKE_INTERCEPTOR_ENV`` set, the fake interceptor writes
there the ``HLS_PROXY_OWN_DNS`` it was started with, and with ``$FAKE_INTERCEPTOR_VARIABLES`` set,
every ``HLS_*`` variable it was started with, as one JSON object."""

from __future__ import annotations

import json
import os
import shlex
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

_STUB = "#!/bin/sh\nFAKE_BIN_DIR={directory} exec {python} {script} {role} \"$@\"\n"


def install(directory: Path) -> Path:
    """Write ``pasta``, ``iptables``, ``ip6tables`` and ``python`` into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for role in ("pasta", "iptables", "ip6tables", "sysctl", "python"):
        path = directory / role
        path.write_text(_STUB.format(directory=shlex.quote(str(directory)), python=shlex.quote(sys.executable),
                                     script=shlex.quote(__file__), role=role))
        path.chmod(0o755)
    return directory


def calls(log: Path) -> list[list[str]]:
    """The flags of every fake pasta call, in order."""
    return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []


#: pasta's forward options: into the namespace over TCP (``-t``) and UDP (``-u``), and out of it
#: to the host's loopback over TCP (``-T``) and UDP (``-U``).
FORWARD_OPTIONS = ("-t", "-u", "-T", "-U")
#: The forwards of a seal that forwards nothing either way.
NO_FORWARDS = {option: ["none"] for option in FORWARD_OPTIONS}
#: pasta's other flags the seal passes, none with a value.
SEAL_FLAGS = ("--config-net", "-f", "--no-map-gw", "--no-icmp")


def forwards(flags: list[str]) -> dict[str, list[str]]:
    """What each forward option of one pasta call (``calls``) names, in the order given. An
    option the call does not name is absent; pasta then forwards every port bound on that side.
    Any other flag than ``SEAL_FLAGS`` fails the test that reads it: a long form, an ``=`` value
    or an attached one could forward what this does not see."""
    named: dict[str, list[str]] = {}
    i = 0
    while i < len(flags):
        if flags[i] in FORWARD_OPTIONS:
            named.setdefault(flags[i], []).append(flags[i + 1])
            i += 2
        elif flags[i] in SEAL_FLAGS:
            i += 1
        else:
            raise AssertionError(f"pasta flag {flags[i]!r} of {flags} is neither a forward nor a seal flag")
    return named


def _netns_rules() -> list[list[str]]:
    state = Path(os.environ["FAKE_NETNS_RULES"])
    return [json.loads(line) for line in state.read_text().splitlines()] if state.exists() else []


#: The rule of the seal's firewall that refuses every ICMP packet its namespace sends.
ICMP_DROP = ["-A", "OUTPUT", "-p", "icmp", "-j", "DROP"]


#: What a ``user.max_*`` key reads in a user namespace that never set it: the kernel's default.
_UNSET_USER_LIMIT = "2147483647"


def _sysctls() -> dict[str, str]:
    state = Path(os.environ["FAKE_NETNS_SYSCTLS"])
    return json.loads(state.read_text()) if state.exists() else {}


def _sysctl(argv: list[str]) -> int:
    log = os.environ.get("FAKE_SYSCTL_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(argv) + "\n")
    if "FAKE_NETNS_SYSCTLS" not in os.environ:
        print("fake sysctl: called outside a fake pasta namespace", file=sys.stderr)
        return 2
    if argv[:2] == ["-q", "-w"] and len(argv) == 3 and "=" in argv[2]:
        key, value = argv[2].split("=", 1)
        if os.environ.get("FAKE_SYSCTL_FAIL"):
            print(f'sysctl: permission denied on key "{key}"', file=sys.stderr)
            return 255
        if not os.environ.get("FAKE_SYSCTL_STUCK"):
            Path(os.environ["FAKE_NETNS_SYSCTLS"]).write_text(json.dumps({**_sysctls(), key: value}))
        return 0
    if argv[:1] == ["-n"] and len(argv) == 2 and (argv[1] in _sysctls() or argv[1].startswith("user.max_")):
        print(_sysctls().get(argv[1], _UNSET_USER_LIMIT))
        return 0
    print(f"fake sysctl: unexpected call {argv!r}", file=sys.stderr)
    return 2


def icmp_dropped() -> bool:
    """Whether the namespace of the fake pasta call this runs under drops the ICMP it sends."""
    return ICMP_DROP in _netns_rules()


def _write_status(errors: int, *, first: bool = True) -> None:
    """Write the interceptor's status as the real one does (``write_status_file``): the whole
    file the *first* time, then in place over it."""
    path = os.environ.get("HLS_PROXY_STATUS_FILE")
    if path:
        from hls.egress.proxy import STATUS_FILE_BYTES, write_status_file

        mid_stream = int(os.environ.get("FAKE_PROXY_MID_STREAM", "0"))
        oversized = json.loads(os.environ.get("FAKE_PROXY_OVERSIZED", "[]"))
        fetches_refused = int(os.environ.get("FAKE_PROXY_REFUSED", "0"))
        policy_refused = json.loads(os.environ.get("FAKE_PROXY_POLICY_REFUSED", "[]"))
        heartbeat = time.time() - float(os.environ.get("FAKE_PROXY_HEARTBEAT_AGE", "0"))
        write_status_file(path, {"upstream_errors": errors, "mid_stream_errors": mid_stream,
                                 "oversized_pages": oversized, "policy_refused": policy_refused,
                                 "fetches_refused": fetches_refused, "heartbeat": heartbeat},
                          0 if first else STATUS_FILE_BYTES)


def _count(var: str) -> int:
    """Add one to the counter in the file ``$var`` names, and return the new count."""
    counter = Path(os.environ[var])
    n = (int(counter.read_text()) if counter.exists() else 0) + 1
    counter.write_text(str(n))
    return n


def _note_pipe_holder(role: str) -> None:
    inode, log = os.environ.get("FAKE_PIPE_INODE"), os.environ.get("FAKE_PIPE_HOLDERS")
    if not inode or not log:
        return
    for fd in os.listdir("/proc/self/fd"):
        try:
            st = os.stat(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if stat.S_ISFIFO(st.st_mode) and st.st_ino == int(inode):
            with open(log, "a") as f:
                f.write(f"{role}\n")
            return


def _started() -> bool:
    return Path(os.environ["HLS_SEAL_STARTED_FILE"]).exists()


def _next_action() -> str:
    plan = os.environ.get("FAKE_PASTA_PLAN")
    if not plan or not Path(plan).exists():
        return "run"
    lines = Path(plan).read_text().splitlines()
    if not lines:
        return "run"
    Path(plan).write_text("".join(f"{line}\n" for line in lines[1:]))
    return lines[0].strip()


def _run_with_parent(payload: list[str], *, pass_fds: tuple[int, ...]) -> int:
    """Run *payload* and return its status as ``subprocess.call`` does. Once this program's
    parent has ended, kill the payload and exit 137, as the processes of a stopped step end."""
    parent = os.getppid()
    proc = subprocess.Popen(payload, pass_fds=pass_fds)
    while True:
        try:
            return proc.wait(0.1)
        except subprocess.TimeoutExpired:
            pass
        if os.getppid() != parent:
            proc.kill()
            proc.wait()
            os._exit(137)


def _pass(payload: list[str]) -> int:
    """What a working in-namespace block does around *payload*, without its listeners."""
    _write_status(int(os.environ.get("FAKE_PROXY_ERRORS", "0")))
    handed = os.environ.get("HLS_SEAL_ENV_FD") == "3"
    if handed:
        os.dup2(0, 3)
        devnull = os.open(os.devnull, os.O_RDONLY)
        os.dup2(devnull, 0)
        os.close(devnull)
    Path(os.environ["HLS_SEAL_STARTED_FILE"]).touch()
    rc = _run_with_parent(payload, pass_fds=(3,) if handed else ())
    if handed:
        os.close(3)
    rc = rc if rc >= 0 else 128 - rc
    ended = os.environ.get("HLS_SEAL_ENDED_FILE")
    if ended:
        # In place over the record the seal made before the payload, as its block writes it.
        fd = os.open(f"{ended}.pending", os.O_WRONLY | os.O_CREAT, 0o600)
        try:
            os.pwrite(fd, (json.dumps({"exit": rc, "interceptor": True}) + "\n").encode(), 0)
        finally:
            os.close(fd)
    return rc


def _pasta(argv: list[str]) -> int:
    cut = argv.index("--")
    log = os.environ.get("FAKE_PASTA_LOG")
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(argv[:cut]) + "\n")
    action = _next_action()
    if action.startswith("fail:"):
        return int(action.split(":", 1)[1])
    rest = argv[cut + 1:]
    os.closerange(3, os.sysconf("SC_OPEN_MAX"))
    os.environ["FAKE_PASTA_NAMESPACE"] = "1"
    # A fresh namespace each call, with no firewall yet.
    state = Path(os.environ["FAKE_BIN_DIR"]) / f"netns-{os.getpid()}.rules"
    state.unlink(missing_ok=True)
    os.environ["FAKE_NETNS_RULES"] = str(state)
    sysctls = Path(os.environ["FAKE_BIN_DIR"]) / f"netns-{os.getpid()}.sysctls"
    sysctls.unlink(missing_ok=True)
    os.environ["FAKE_NETNS_SYSCTLS"] = str(sysctls)
    if action == "pass" or action.startswith("pass_exit:"):
        # A working block drops the namespace's ICMP before the payload starts.
        state.write_text(json.dumps(ICMP_DROP) + "\n")
        # And writes its role's limits.
        pairs = os.environ.get("HLS_SEAL_UCOUNTS", "").split()
        sysctls.write_text(json.dumps({f"user.{p.split('=', 1)[0]}": p.split("=", 1)[1] for p in pairs}))
        # rest is: bash -c <in-namespace block> _seal <payload...>
        rc = _pass(rest[4:])
        return int(action.split(":", 1)[1]) if action.startswith("pass_exit:") else rc
    if action.startswith("die_after:"):
        child = os.fork()
        if child == 0:
            os.execvp(rest[0], rest)
        while not _started():
            done, status = os.waitpid(child, os.WNOHANG)
            if done:
                return os.waitstatus_to_exitcode(status)
            time.sleep(0.02)
        time.sleep(float(action.split(":", 1)[1]))
        os._exit(137)
    os.execvp(rest[0], rest)
    return 0  # not reached


def _python(argv: list[str]) -> int:
    if argv[:1] == ["-c"]:
        print(40000 + _count("FAKE_PYTHON_COUNTER"))
        return 0
    if argv[:1] == ["-m"] and len(argv) > 1 and argv[1].startswith("hls.") and argv[1] != "hls.egress.proxy":
        # Any other module of hls is a harness role's, which the real interpreter runs.
        os.execv(sys.executable, [sys.executable, *argv])
    if argv[:2] != ["-m", "hls.egress.proxy"]:
        print(f"fake python: unexpected call {argv!r}", file=sys.stderr)
        return 2
    mode = os.environ.get("FAKE_INTERCEPTOR", "serve")
    if os.environ.get("FAKE_INTERCEPTOR_ENV"):
        Path(os.environ["FAKE_INTERCEPTOR_ENV"]).write_text(os.environ.get("HLS_PROXY_OWN_DNS", ""))
    if os.environ.get("FAKE_INTERCEPTOR_VARIABLES"):
        Path(os.environ["FAKE_INTERCEPTOR_VARIABLES"]).write_text(
            json.dumps({name: value for name, value in os.environ.items() if name.startswith("HLS_")}))
    if mode == "die" or (mode == "die_first" and _count("FAKE_INTERCEPTOR_STARTS") == 1):
        return 3
    _write_status(0)
    listeners = []
    for var in ("HLS_PROXY_TRANSPARENT_PORT", "HLS_PROXY_PORT"):
        if os.environ.get(var):
            s = socket.socket()
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", int(os.environ[var])))
            s.listen()
            listeners.append(s)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    if not mode.startswith("quit_after:"):
        # One heartbeat once the payload runs, as the real interceptor writes every few seconds.
        while not _started():
            time.sleep(0.02)
        _write_status(0, first=False)
        while True:
            time.sleep(1)
    while not _started():
        time.sleep(0.02)
    time.sleep(float(mode.split(":", 1)[1]))
    return 0


def main(argv: list[str]) -> int:
    role, rest = argv[0], argv[1:]
    if role != "pasta":
        _note_pipe_holder(role)
    if role == "pasta":
        return _pasta(rest)
    if role == "python":
        return _python(rest)
    if role == "sysctl":
        return _sysctl(rest)
    # iptables and ip6tables: every rule installs, an IPv4 one in the namespace's firewall too.
    log = os.environ.get({"iptables": "FAKE_IPTABLES_LOG", "ip6tables": "FAKE_IP6TABLES_LOG"}[role])
    if log:
        with open(log, "a") as f:
            f.write(json.dumps(rest) + "\n")
    if role == "iptables" and os.environ.get("FAKE_NETNS_RULES"):
        with open(os.environ["FAKE_NETNS_RULES"], "a") as f:
            f.write(json.dumps(rest) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
