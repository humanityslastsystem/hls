"""The build sandbox's probe: the isolation, credential and route checks an operator runs before
builds, the seal checks every build runs before its agent, and the live check's
own tool.

The agent_build stage starts the sandbox exactly as a build does, then runs these checks through
the exec server, as contestant commands. Each check records what it saw against what it should
have seen (``ProbeCheck``), so a sandbox that is not isolated or a route that is not reachable
shows up as a failed check rather than as a build that quietly ran without a protection. No check
prints a secret: a credential check names the variables it found, never a value, and the stage
scrubs every observed value before it records it.

Three kinds of check:
- *in the sandbox, unsealed* (``SANDBOX_CHECKS`` without ``sealed``): the interpreter, the uid and
  gid, no effective capabilities, a process view that shows none of the harness (the namespace's
  own PID 1, the reaper, is a fork of the spawner and so shows its command line; the server and
  the spawner itself must not show at all), ``oom_score_adj`` 1000 on the command, PID 1's memory
  and environment unreadable or free of route variables, no route variable in the command's own
  environment, the process ceiling the commands run under and how the kernel counts it, the
  per-uid count limits of the commands' namespace and that a command cannot raise them, a capped
  ``/dev/shm``, a covered
  control directory whose ready file a command can neither read, open for writing, nor replace,
  and an unreachable control socket, the GPU count, the budget MCP server
  answering through its socket, a limit set on PID 1 that a later command does not inherit, and
  a PID 1 a command ended by its CPU limit replaced by a fresh namespace for the next command.
  Probe commands go through the exec server without harbor's ``extra_env``, so a route variable
  in their environment came from the sandbox itself;
- *in the sandbox, under the seal* (``sealed``): a TLS connection to the arm's route host
  succeeds, a request for a host the seal's policy refuses, sent to an address only the seal's
  redirect takes, is closed unanswered, no ICMP echo leaves (``icmp-held``), and ``pip download``
  through the pull-through index succeeds; and, with the stage's knocks (``inbound_check``), no
  port a sandbox command binds is reached from any address of the node (``inbound-closed``). The
  last two run the drive's seal probe's own code (``hls.egress.seal_probe``) on the sandbox's
  interpreter. Every build runs these before its preflight (the seal probe);
- *host side, by the stage:* the exec server bound its socket within its budget, the file a
  sandbox command made in ``/workspace`` is owned by the host user, the server's ready line, in
  its ready file (the one the memory split reads, ``ready_pids``), says it holds score 0, the
  seal's interceptor recorded the refused host by its policy (a connection closed unanswered is
  no proof alone, since an interceptor that lets every host through closes it too when the host
  does not resolve), the sandbox's env file holds no route variable and no secret's value, the
  node's ``pid_max`` above the commands' process ceiling, and, read through the step's own
  processes by the ready file's pids
  (``hls.agent.memory.find_reaper``), the spawner's score, the reaper's depth (the commands' PID
  1 one namespace below the spawner; how far below the step's base, which the seal's layers
  decide, is recorded) and the reaper's descriptors (no socket); and
  two measurements that always pass: the step's memory split, and the score and resident memory
  of each process of the step outside the container.

Together they cover each property a live sandbox must show: ``python-version``; ``nspid-depth``;
``uid-gid`` and ``workspace-owner``; ``cap-eff``; ``ps-isolation``; ``dev-shm``;
``control-covered`` and ``control-socket``; ``pid1-recovery``; ``pid1-non-dumpable``;
``oom-score-adj``; ``exec-server-ready``; ``server-oom-score`` and ``spawner-oom-score``; the
sandbox starting at all, its binds made on the read-only tree; ``outside-scores``; for the
reaper, ``limits-not-inherited`` and ``reaper-no-socket``; and for the process ceiling,
``command-nproc`` and ``nproc-accounting``.

A command sees its PID namespace through the reaper's own ``/proc``, where its ``NSpid`` line has
one entry, so the depth is read host side, from the reaper's place among the step's processes.
A command cannot end PID 1 by a signal: the kernel drops every signal a process sends to its own
namespace's init, SIGKILL included, unless init handles it. It can lower PID 1's CPU limit,
which the uid they share allows, and make it work until the kernel's RLIMIT_CPU kill, which that
protection does not drop; ``pid1-recovery`` does that.

The in-sandbox checks of what a nested namespace provides only pass inside one, and the
host-side checks that read the sandbox step's own processes only pass in a real step. ``cap-eff``
is not among the first: an unprivileged host already has no effective capabilities, so it passes
without a namespace too, while under the seal the payload is uid 0 of pasta's user namespace and
the nested namespace is what drops its capabilities.

``--probe-oom`` adds one more: a memory hog runs in the sandbox, and the stage records what ended
it (``OomProbe``), which sets the OOM rule's reserve."""

from __future__ import annotations

import asyncio
import math
import os
import shlex
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from hls.agent import exec_protocol
from hls.agent.memory import ProcessMemory, ReadyPids, SplitSample, find_reaper, read_process, step_processes
from hls.egress import seal_probe
from hls.substrate.cgroup import pids
from hls.substrate import role_exec
from hls.substrate.nproc import Accounting

#: The file a sandbox command makes in the workspace, read back host-side for its owner.
WORKSPACE_PROBE = ".hls-probe-owner"
#: A small package the sealed index check downloads.
INDEX_PACKAGE = "six"
#: Python 3.12 is the floor: ``os.unshare`` and the spawner need it.
MIN_PYTHON = (3, 12)
#: Seconds the refusal check waits for its connection and for a reply.
REFUSAL_WAIT_SEC = 15
#: Seconds the TLS check waits for its connection.
TLS_WAIT_SEC = 10
#: Seconds between the stage's looks for the inbound check's ready file.
READY_POLL_SEC = 0.1
#: The inbound check's name (``inbound_check``).
INBOUND_CHECK = "inbound-closed"


class ProbeCheck(BaseModel):
    """One check and what it found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The check's name.
    name: str
    #: Whether what was seen is what should have been seen.
    ok: bool
    #: What the check saw, for the record; never a secret's value.
    observed: str
    #: What a passing check sees.
    expected: str


class OomProbe(BaseModel):
    """What ended the ``--probe-oom`` memory hog, and the split samples taken while it grew."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Whether the sandbox still ran once the hog had ended.
    survived: bool
    #: The hog's exit status as the exec server returned it (minus the signal for one a signal
    #: ended); None when the exec channel failed.
    exit_code: int | None
    #: The exception the exec channel failed with, by its type's name; None when the hog's exec
    #: returned.
    error: str | None
    #: ``oom_kill`` events of the sandbox step's cgroup in its last split sample.
    oom_kills: int
    #: The runner's memory kills in the sandbox's step: of single contestant processes, and of
    #: the whole step.
    runner_kills: int
    #: The largest resident memory of the harness's processes in any split sample, in bytes: the
    #: measurement ``harness_reserve_gib`` is set from.
    harness_max_bytes: int
    #: The split samples taken while the sandbox ran.
    samples: int


class ProbeRecord(BaseModel):
    """``probe.json``: every check, and the OOM probe when it ran."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Each check, in the order they ran.
    checks: list[ProbeCheck]
    #: Whether every check passed.
    all_ok: bool
    #: The ``--probe-oom`` result; None when it did not run.
    oom: OomProbe | None = None
    #: The roles' judgement of the sandbox's ending when it ended during a stop build, as
    #: ``{fault_class, ending, reason}`` (scrubbed); None when the sandbox was stopped as planned.
    sandbox_ending: dict[str, str] | None = None


@dataclass(frozen=True)
class ProbeInputs:
    """What a passing sandbox looks like, for the in-sandbox checks to compare against."""

    #: The host uid the sandbox's commands should run as.
    uid: int
    #: The host gid the sandbox's commands should run as.
    gid: int
    #: The cap of ``/dev/shm`` in bytes.
    shm_bytes: int
    #: The GPUs the sandbox should see.
    gpus: int
    #: Where the control directory is bound (it should be covered and empty).
    control_target: str
    #: The exec socket's path in the container (a connection to it should be refused).
    socket_target: str
    #: The budget MCP server's path in the container.
    mcp_target: str
    #: The arm's route host, opened under the seal.
    route_host: str
    #: A host the seal's policy refuses, asked for under the seal; None when the policy refuses
    #: no host, and the refusal check is not made.
    refused_host: str | None
    #: ``ADDR:PORT`` the refusal check connects to: an address only the seal's redirect to its
    #: interceptor takes.
    dial: str
    #: The variables the arm's route reaches the CLI by (its grants and its plain settings),
    #: which no sandbox command and no PID 1 may hold.
    route_vars: tuple[str, ...]
    #: The process ceiling each command runs under (``RLIMIT_NPROC``, soft and hard); None when
    #: there is none.
    command_nproc: int | None
    #: How the site says the kernel counts it (``site.nproc_accounting``), which
    #: ``nproc-accounting`` measures.
    nproc_accounting: Accounting
    #: The limit the commands' user namespace should have in each ``/proc/sys/user`` file
    #: (``hls.substrate.ucounts``), by its name.
    ucounts: Mapping[str, int]
    #: The interpreter the sandbox's own programs run on, the image's
    #: (``exec_protocol.SANDBOX_PYTHON``): the probe's interpreter commands run it, and
    #: ``python-version`` reads its version.
    python: str


@dataclass(frozen=True)
class SandboxCheck:
    """One check made by sandbox commands."""

    #: The check's name.
    name: str
    #: The commands it runs, in order, given the inputs; none for a check that has nothing to do.
    commands: Callable[[ProbeInputs], tuple[str, ...]]
    #: The check from the commands' results, in order, and the inputs.
    parse: Callable[[Sequence[Any], ProbeInputs], ProbeCheck]
    #: True for a check that needs the seal: one of the seal probe's.
    sealed: bool = False


def _lines(result: Any) -> list[str]:
    return (result.stdout or "").splitlines()


def _first(result: Any) -> str:
    return (_lines(result) or [""])[0].strip()


def _last(result: Any) -> str:
    return next((line.strip() for line in reversed(_lines(result)) if line.strip()), "")


def _python(inputs: ProbeInputs, code: str, *args: str) -> str:
    """A command that runs *code* on the sandbox's interpreter (``ProbeInputs.python``), isolated,
    with *args*."""
    return " ".join([shlex.quote(inputs.python), "-I", "-c", shlex.quote(code), *map(shlex.quote, args)])


def _check_python(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    seen = _first(results[0])
    try:
        ok = tuple(int(part) for part in seen.split(".")) >= MIN_PYTHON
    except ValueError:
        ok = False
    return ProbeCheck(name="python-version", ok=ok, observed=seen, expected=">= 3.12")


def _check_uid_gid(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    seen, want = " ".join(_lines(results[0])), f"{inputs.uid} {inputs.gid}"
    return ProbeCheck(name="uid-gid", ok=seen == want, observed=seen, expected=want)


def _check_cap(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    seen = _first(results[0])
    return ProbeCheck(name="cap-eff", ok=seen == "0000000000000000", observed=seen, expected="0000000000000000")


def _check_ps(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    listed = [line.split(" ", 1) for line in _lines(results[0]) if line.strip()]
    harness = sorted({pid for pid, *args in listed if pid != "1" and args
                      and ("exec_server.py" in args[0] or "spawner.py" in args[0])})
    expected = "the exec server and the spawner are not listed; PID 1 is the reaper"
    if results[0].return_code != 0 or not listed:
        return ProbeCheck(name="ps-isolation", ok=False, observed="the processes were not listed", expected=expected)
    return ProbeCheck(name="ps-isolation", ok=not harness,
                      observed=f"harness visible as pids {', '.join(harness)}" if harness else
                      f"{len(listed)} processes, none of the harness", expected=expected)


def _check_oom_adj(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    seen = _first(results[0])
    return ProbeCheck(name="oom-score-adj", ok=seen == str(role_exec.CONTESTANT_OOM_SCORE_ADJ), observed=seen,
                      expected=str(role_exec.CONTESTANT_OOM_SCORE_ADJ))


def _check_dumpable(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    seen = " ".join(_lines(results[0]))
    return ProbeCheck(name="pid1-non-dumpable", ok=seen == "mem=refused environ=refused", observed=seen,
                      expected="mem=refused environ=refused")


def _route_names(names: Sequence[str], inputs: ProbeInputs) -> list[str]:
    return sorted(set(names) & set(inputs.route_vars))


def _check_pid1_env(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    lines = _lines(results[0])
    expected = "PID 1's environment unreadable, or without a route variable"
    if lines[:1] == ["unreadable"]:
        return ProbeCheck(name="pid1-route-env", ok=True, observed="unreadable", expected=expected)
    if lines[:1] != ["readable"]:
        return ProbeCheck(name="pid1-route-env", ok=False, observed="no answer", expected=expected)
    found = _route_names(lines[1:], inputs)
    return ProbeCheck(name="pid1-route-env", ok=not found, observed=", ".join(found) or "readable, none",
                      expected=expected)


def _check_own_env(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    found = _route_names(_lines(results[0]), inputs)
    read = results[0].return_code == 0
    return ProbeCheck(name="route-env-absent", ok=read and not found,
                      observed=", ".join(found) or ("none" if read else "unread"),
                      expected="no route variable in a command's environment")


def _check_nproc(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    seen = " ".join(_lines(results[0]))
    if inputs.command_nproc is None:
        # No ceiling: the commands keep the spawner's own limit, recorded as a measurement.
        return ProbeCheck(name="command-nproc", ok=results[0].return_code == 0, observed=seen,
                          expected="the spawner's own limit (no ceiling)")
    want = f"{inputs.command_nproc} {inputs.command_nproc}"
    return ProbeCheck(name="command-nproc", ok=seen == want, observed=seen, expected=want)


def read_accounting(result: Any) -> Accounting | None:
    """How the kernel counts the commands' process ceiling, as ``nproc-accounting``'s command
    *result* shows it: ``uid`` when it forked nothing, ``user_namespace`` when it forked at least
    ``ACCOUNTING_CLEAR``; None when it shows neither, a count between them (unclear) or no count
    at all (an error: a fork refused otherwise than by the limit, such as a transient ENOMEM, no
    namespace made, no answer). Only a reading that is not None says which the kernel uses."""
    seen = _first(result)
    if not seen.startswith("forks="):
        return None
    try:
        forks = int(seen.removeprefix("forks="))
    except ValueError:
        return None
    return "uid" if forks == 0 else "user_namespace" if forks >= ACCOUNTING_CLEAR else None


def _check_accounting(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    seen = _first(results[0])
    measured = read_accounting(results[0])
    observed = f"{seen or 'nothing'} held={ACCOUNTING_HELD} limit={ACCOUNTING_LIMIT} accounting={measured or 'unclear'}"
    return ProbeCheck(name="nproc-accounting", ok=measured == inputs.nproc_accounting, observed=observed,
                      expected=f"accounting={inputs.nproc_accounting}")


def _check_ucounts(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    seen = dict(line.split(" ", 1) for line in _lines(results[0]) if " " in line)
    want = " ".join(f"{name}={limit}" for name, limit in sorted(inputs.ucounts.items()))
    got = " ".join(f"{name}={seen.get(name, 'absent')}" for name in sorted(inputs.ucounts))
    return ProbeCheck(name="ucount-ceilings", ok=bool(inputs.ucounts) and got == want, observed=got or "none",
                      expected=want or "a limit for each per-uid count")


def _check_ucount_locked(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    seen = _first(results[0])
    return ProbeCheck(name="ucount-locked", ok=seen in ("EACCES", "EPERM"), observed=seen or "nothing",
                      expected="a command's write to its namespace's /proc/sys/user refused (EACCES)")


def _check_shm(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    fields = dict(part.split("=", 1) for part in _lines(results[0]) if "=" in part)
    size, write = fields.get("size", ""), fields.get("write", "")
    return ProbeCheck(name="dev-shm", ok=size == str(inputs.shm_bytes) and write == "ok",
                      observed=f"size={size} write={write}", expected=f"size={inputs.shm_bytes} write=ok")


def _check_control(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    lines = _lines(results[0])
    expected = ("no entry: covered empty or unreadable (mode 000); the ready file neither read, opened for "
                "writing, nor replaceable")
    ready = next((line for line in lines if line.startswith("ready ")), "")
    fields = dict(part.split("=", 1) for part in ready.split()[1:] if "=" in part)
    sealed = all(fields.get(k, "").startswith("denied") for k in ("read", "write", "create"))
    lines = [line for line in lines if not line.startswith("ready ")]
    if lines[:1] == ["unreadable"]:
        listed, ok = "unreadable (mode 000)", True
    elif "__listed__" not in lines:
        listed, ok = (lines or ["no answer"])[0], False
    else:
        entries = [line for line in lines if line and line != "__listed__"]
        listed, ok = ",".join(entries) or "empty", not entries
    observed = listed if sealed else f"{listed}; {ready or 'no ready-file answer'}"
    return ProbeCheck(name="control-covered", ok=ok and sealed, observed=observed, expected=expected)


def _check_socket(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    seen = _first(results[0])
    return ProbeCheck(name="control-socket", ok=seen == "refused", observed=seen, expected="refused")


def _check_gpus(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    seen = len([line for line in _lines(results[0]) if line.startswith("GPU")])
    return ProbeCheck(name="gpu-count", ok=seen == inputs.gpus, observed=str(seen), expected=str(inputs.gpus))


def _check_mcp(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    ok = "wallclock" in (results[0].stdout or "")
    return ProbeCheck(name="budget-tool", ok=ok, observed="answered" if ok else "no wallclock",
                      expected="get_budget returns a wall-clock budget")


def _check_limits(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    before, setting, after = (" ".join(_lines(r)) for r in results)
    ok = setting == "set" and before == after and before != ""
    return ProbeCheck(name="limits-not-inherited", ok=ok, observed=f"before={before} pid1:{setting} after={after}",
                      expected="PID 1's open-files limit set to 4, and a later command keeps the spawner's")


def _check_recovery(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    old, ended, fresh = results
    first, last = _first(old), _first(fresh)
    ok = fresh.return_code == 0 and first.startswith("pid:") and last.startswith("pid:") and first != last
    return ProbeCheck(name="pid1-recovery", ok=ok,
                      observed=f"{first or 'none'} -> {last or 'none'} (the ending command: rc={ended.return_code} "
                               f"{_first(ended)})",
                      expected="a command that ends PID 1 by its CPU limit, and the next command in a fresh "
                               "PID namespace")


def _reached(result: Any) -> bool:
    """Whether a TLS check connected: the command prints ``ok <version>`` on a handshake."""
    return result.return_code == 0 and _first(result).split(" ", 1)[0] == "ok"


def _check_route(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    return ProbeCheck(name="route-reachable", ok=_reached(results[0]), observed=_first(results[0])[:80],
                      expected="a TLS connection to the route's host succeeds")


def _check_refused(results: Sequence[Any], inputs: ProbeInputs) -> ProbeCheck:
    expected = "a request for a host the policy refuses is closed unanswered"
    if inputs.refused_host is None:
        return ProbeCheck(name="unlisted-refused", ok=True, observed="the policy refuses no host", expected=expected)
    seen = _first(results[0])
    return ProbeCheck(name="unlisted-refused", ok=seen == "refused", observed=seen[:80], expected=expected)


def _check_index(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    return ProbeCheck(name="index-reachable", ok=results[0].return_code == 0, observed=f"rc={results[0].return_code}",
                      expected=f"pip downloads {INDEX_PACKAGE} through the pull-through index")


# The commands. Each runs through the exec server, so as a contestant command. Every one that
# reads or writes a file of its own leaves nothing behind.

def _python_version_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    return (_python(inputs, 'import sys; print("%d.%d.%d" % sys.version_info[:3])'),)


def _uid_gid_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    return ("printf '%s %s\\n' \"$(id -u)\" \"$(id -g)\"",)


def _cap_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    return ("awk '/^CapEff:/{print $2}' /proc/self/status",)


def _ps_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # Every process this command's /proc lists, by pid and command line; one that ends meanwhile
    # is skipped.
    code = ("import os\n"
            "for p in sorted(os.listdir('/proc')):\n"
            " if p.isdigit():\n"
            "  try:\n   args = open('/proc/' + p + '/cmdline', 'rb').read()\n"
            "  except OSError:\n   continue\n"
            "  print(p, args.replace(bytes(1), b' ').decode(errors='replace').strip())\n")
    return (_python(inputs, code),)


def _oom_adj_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    return ("cat /proc/self/oom_score_adj",)


def _dumpable_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # The kernel checks access when each file is opened; nothing is read. Only a permission
    # refusal (EACCES/EPERM) is a pass: a file that is not there (ENOENT) proves nothing.
    code = ("import errno\n"
            "for f in ('mem', 'environ'):\n"
            " try:\n  open('/proc/1/' + f, 'rb').close(); print(f + '=read')\n"
            " except OSError as e:\n"
            "  print(f + '=refused' if e.errno in (errno.EACCES, errno.EPERM) else f + '=errno:%s' % e.errno)\n")
    return (_python(inputs, code),)


def _pid1_env_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # The names alone, when PID 1's environment can be read at all. Only a permission refusal is
    # "unreadable"; any other error is reported so the check does not pass on it.
    code = ("import errno\n"
            "try:\n data = open('/proc/1/environ', 'rb').read()\n"
            "except OSError as e:\n"
            " print('unreadable' if e.errno in (errno.EACCES, errno.EPERM) else 'errno:%s' % e.errno)\n"
            "else:\n print('readable')\n"
            " print('\\n'.join(sorted({e.split(b'=', 1)[0].decode(errors='replace') "
            "for e in data.split(bytes(1)) if e})))\n")
    return (_python(inputs, code),)


def _own_env_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # The names alone: a value never leaves the sandbox.
    return (_python(inputs, 'import os; print("\\n".join(sorted(os.environ)))'),)


def _nproc_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    return ("ulimit -Su; ulimit -Hu",)


#: The tasks ``nproc-accounting`` holds in its command and the limit it forks under inside a user
#: namespace it makes, as the spawner sets the commands' ceiling inside the one it made: more
#: held than the limit, so a kernel that counts every task of the uid allows no fork, and one that
#: counts the namespace's own allows all but the forking process itself: an el8 kernel allows 0
#: forks, an el9 one 511.
ACCOUNTING_HELD, ACCOUNTING_LIMIT = 600, 512
#: The fewest forks that read as ``user_namespace``: half the limit. A count from 1 up to it reads
#: as neither accounting (``read_accounting``).
ACCOUNTING_CLEAR = ACCOUNTING_LIMIT // 2


def _accounting_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # The threads hold the uid's tasks; a child, single-threaded as unshare needs, makes the
    # namespace, lowers its limit and forks until refused, then ends what it forked. Its count
    # stands only once its own user namespace is seen to differ from the one it was forked in.
    code = ("import errno, os, resource, signal, sys, threading, time, warnings\n"
            "warnings.simplefilter('ignore')\n"
            "held, limit = int(sys.argv[1]), int(sys.argv[2])\n"
            "hold = threading.Event()\n"
            "for _ in range(held):\n"
            " threading.Thread(target=hold.wait, daemon=True).start()\n"
            "outer = os.readlink('/proc/self/ns/user')\n"
            "r, w = os.pipe()\n"
            "pid = os.fork()\n"
            "if pid == 0:\n"
            " os.close(r)\n"
            " made = []\n"
            " try:\n"
            "  os.unshare(os.CLONE_NEWUSER)\n"
            "  resource.setrlimit(resource.RLIMIT_NPROC, (limit, limit))\n"
            "  while len(made) < 2 * limit:\n"
            "   try:\n"
            "    child = os.fork()\n"
            "   except OSError as e:\n"
            "    if e.errno != errno.EAGAIN:\n"
            "     raise\n"
            "    break\n"
            "   if child == 0:\n"
            "    time.sleep(60)\n"
            "    os._exit(0)\n"
            "   made.append(child)\n"
            "  made_ns = os.readlink('/proc/self/ns/user') != outer\n"
            "  out = 'forks=%d' % len(made) if made_ns else 'error=no user namespace was made'\n"
            " except Exception as e:\n"
            "  out = 'error=%s:%s' % (type(e).__name__, e)\n"
            " for c in made:\n"
            "  os.kill(c, signal.SIGKILL)\n"
            " for c in made:\n"
            "  os.waitpid(c, 0)\n"
            " os.write(w, out.encode())\n"
            " os._exit(0)\n"
            "os.close(w)\n"
            "out = os.read(r, 4096).decode()\n"
            "os.waitpid(pid, 0)\n"
            "hold.set()\n"
            "print(out)\n")
    return (_python(inputs, code, str(ACCOUNTING_HELD), str(ACCOUNTING_LIMIT)),)


def _ucounts_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    # The commands' namespace's own limits, as a command reads them.
    return ('for f in /proc/sys/user/max_*; do printf "%s %s\\n" "${f##*/}" "$(cat "$f")"; done',)


def _ucount_locked_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # A command holds no capability in its namespace, so it cannot raise the limits it was given.
    code = ("import errno\n"
            "try:\n"
            " open('/proc/sys/user/max_inotify_instances', 'w').write('2147483647')\n"
            " print('written')\n"
            "except OSError as e:\n"
            " print(errno.errorcode.get(e.errno, e.errno))\n")
    return (_python(inputs, code),)


def _shm_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    # The cap is the sandbox's whole memory, so a write up to it would end the sandbox for its
    # memory first: the probe reads the size the kernel enforces and writes one page.
    return ("f=$(mktemp -p /dev/shm hls-probe.XXXXXX) || exit 1; "
            "printf 'size=%s\\n' \"$(df -B1 --output=size /dev/shm | tail -n1 | tr -d ' ')\"; "
            "if dd if=/dev/zero of=\"$f\" bs=4096 count=1 >/dev/null 2>&1; "
            "then echo write=ok; else echo write=failed; fi; rm -f \"$f\"",)


def _control_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # The control directory is covered by an empty read-only tmpfs of mode 000, so a command
    # either lists it empty or cannot read it at all (EACCES/EPERM): both hide the socket. A
    # directory that is not there (ENOENT) or lists entries is not covered. The sentinel tells a
    # directory listed empty from one the interpreter could not reach at all. The command then
    # tries the server's ready file, which names the reaper the memory split trusts: to read it,
    # to open it for writing (nothing is written), and to create a file beside it, as a rename
    # over it would need; each must be denied. A file it did create is removed.
    code = ("import errno, os, sys\n"
            "d = sys.argv[1]\n"
            "try:\n"
            " entries = os.listdir(d)\n"
            "except OSError as e:\n"
            " print('unreadable' if e.errno in (errno.EACCES, errno.EPERM) else 'errno:%s' % e.errno)\n"
            "else:\n"
            " [print(name) for name in entries]\n print('__listed__')\n"
            "def attempt(path, flags):\n"
            " try:\n"
            "  fd = os.open(path, flags, 0o600)\n"
            " except OSError as e:\n"
            "  return 'denied:%s' % e.errno\n"
            " os.close(fd)\n"
            " return 'opened'\n"
            "ready = os.path.join(d, sys.argv[2])\n"
            "beside = os.path.join(d, '.hls-probe-%d' % os.getpid())\n"
            "made = attempt(beside, os.O_WRONLY | os.O_CREAT | os.O_EXCL)\n"
            "if made == 'opened':\n"
            " os.unlink(beside)\n"
            "print('ready read=%s write=%s create=%s' % (attempt(ready, os.O_RDONLY), attempt(ready, os.O_WRONLY),"
            " made))\n")
    return (_python(inputs, code, inputs.control_target, exec_protocol.READY_NAME),)


def _socket_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    code = ("import socket, sys\n"
            "s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)\n"
            "try:\n s.connect(sys.argv[1]); print('reached')\n"
            "except OSError:\n print('refused')\n")
    return (_python(inputs, code, inputs.socket_target),)


def _gpu_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    return ("nvidia-smi -L 2>/dev/null || true",)


def _mcp_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    init = '{"jsonrpc":"2.0","id":1,"method":"initialize"}'
    call = '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_budget"}}'
    return (f"test -x {shlex.quote(inputs.mcp_target)} && "
            f"printf '%s\\n%s\\n' {shlex.quote(init)} {shlex.quote(call)} | {shlex.quote(inputs.mcp_target)}",)


def _limits_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    read = "ulimit -Sn; ulimit -Hn"
    lower = ("import resource\n"
             "try:\n resource.prlimit(1, resource.RLIMIT_NOFILE, (4, 4)); print('set')\n"
             "except OSError as e:\n print('refused', type(e).__name__)\n")
    return (read, _python(inputs, lower), read)


def _recovery_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    # PID 1 is known by its namespace and its start time: the kernel reuses a freed namespace's
    # inode number, so a fresh namespace may carry the one before it's.
    ns = "echo \"$(readlink /proc/self/ns/pid) $(cut -d' ' -f22 /proc/1/stat)\""
    # PID 1's CPU limit lowered to a second, then it is kept working until the kernel ends it:
    # four churners for up to 90 s leave it orphans to reap, each a grandchild that exits at once.
    # A signal sent to it would not do: el8 drops a signal a command sends to its
    # namespace's init while init waits for it in sigwait. Once PID 1 ends, the kernel ends this
    # command with it. A limit it refused leaves PID 1 alone.
    lower = "import resource; resource.prlimit(1, resource.RLIMIT_CPU, (1, 1))"
    churn = ("import os\n"
             "while True:\n"
             " try:\n"
             "  if os.fork() == 0:\n"
             "   if os.fork() == 0:\n"
             "    os._exit(0)\n"
             "   os._exit(0)\n"
             "  os.wait()\n"
             " except OSError:\n"
             "  pass\n")
    end = (f"{_python(inputs, lower)} 2>/dev/null || {{ echo refused; exit 1; }}; "
           f"for i in 1 2 3 4; do timeout 90 {_python(inputs, churn)} & done; wait; echo survived")
    return (ns, end, ns)


def _tls_cmd(inputs: ProbeInputs, host: str) -> str:
    code = ("import socket, ssl, sys\n"
            "h = sys.argv[1]\n"
            "ctx = ssl.create_default_context()\n"
            "ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE\n"
            "try:\n"
            f" c = ctx.wrap_socket(socket.create_connection((h, 443), {TLS_WAIT_SEC}), server_hostname=h)\n"
            " print('ok', c.version()); c.close()\n"
            "except Exception as e:\n print('fail', type(e).__name__)\n")
    return _python(inputs, code, host)


def _route_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    return (_tls_cmd(inputs, inputs.route_host),)


def _refused_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    if inputs.refused_host is None:
        return ()
    # As the drive stage's seal probe does (``hls.drive.seal_probe.refusal_check``): ask for the
    # host at an address only the seal's redirect takes, and pass when the connection is closed,
    # or reset, without a byte of reply.
    code = ("import socket, sys\n"
            "dial, host = sys.argv[1], sys.argv[2]\n"
            "addr, _, port = dial.rpartition(':')\n"
            "try:\n"
            f" c = socket.create_connection((addr, int(port)), {REFUSAL_WAIT_SEC})\n"
            "except OSError as e:\n print('unreached', type(e).__name__); sys.exit(0)\n"
            "try:\n"
            " c.sendall(('GET / HTTP/1.1\\r\\nHost: %s\\r\\nConnection: close\\r\\n\\r\\n' % host).encode())\n"
            " reply = c.recv(1)\n"
            "except ConnectionResetError:\n reply = b''\n"
            "except TimeoutError:\n print('timeout'); sys.exit(0)\n"
            "except OSError as e:\n print('failed', type(e).__name__); sys.exit(0)\n"
            "print('answered' if reply else 'refused')\n")
    return (_python(inputs, code, inputs.dial, inputs.refused_host),)


def _probe_code() -> str:
    """The seal probe's program (``hls.egress.seal_probe``), as ``-c`` code: the sandbox has no
    checkout to run it from."""
    return Path(seal_probe.__file__).read_text()


def _icmp_cmd(inputs: ProbeInputs) -> tuple[str, ...]:
    return (_python(inputs, _probe_code(), "icmp"),)


def _check_icmp(results: Sequence[Any], _inputs: ProbeInputs) -> ProbeCheck:
    expected = "no ICMP echo leaves the seal"
    check = seal_probe.read_check(_last(results[0]), "icmp")
    if check is None:
        return ProbeCheck(name="icmp-held", ok=False, observed=f"no check: {_last(results[0])[:80]}",
                          expected=expected)
    return ProbeCheck(name="icmp-held", ok=check.passed, observed=check.detail, expected=expected)


def _index_cmd(_i: ProbeInputs) -> tuple[str, ...]:
    # The agent's own pip, resolved from the sandbox PATH (the agent cap installs pip only in its
    # venv, never for the system ``/usr/bin/python3``), so the check exercises the path the agent
    # uses through the pull-through index. Into a directory of its own, without pip's cache, and
    # removed after: the agent of a full build starts after this and finds nothing of it.
    return ("d=$(mktemp -d) || exit 1; pip download --no-deps --no-cache-dir "
            f"--disable-pip-version-check -q -d \"$d\" {INDEX_PACKAGE} >/dev/null 2>&1; rc=$?; rm -rf \"$d\"; "
            "exit $rc",)


#: Every in-sandbox check, in the order they run; the sealed ones are the seal probe.
SANDBOX_CHECKS: tuple[SandboxCheck, ...] = (
    SandboxCheck("python-version", _python_version_cmd, _check_python),
    SandboxCheck("uid-gid", _uid_gid_cmd, _check_uid_gid),
    SandboxCheck("cap-eff", _cap_cmd, _check_cap),
    SandboxCheck("ps-isolation", _ps_cmd, _check_ps),
    SandboxCheck("oom-score-adj", _oom_adj_cmd, _check_oom_adj),
    SandboxCheck("pid1-non-dumpable", _dumpable_cmd, _check_dumpable),
    SandboxCheck("pid1-route-env", _pid1_env_cmd, _check_pid1_env),
    SandboxCheck("route-env-absent", _own_env_cmd, _check_own_env),
    SandboxCheck("command-nproc", _nproc_cmd, _check_nproc),
    SandboxCheck("ucount-ceilings", _ucounts_cmd, _check_ucounts),
    SandboxCheck("ucount-locked", _ucount_locked_cmd, _check_ucount_locked),
    SandboxCheck("dev-shm", _shm_cmd, _check_shm),
    SandboxCheck("control-covered", _control_cmd, _check_control),
    SandboxCheck("control-socket", _socket_cmd, _check_socket),
    SandboxCheck("gpu-count", _gpu_cmd, _check_gpus),
    SandboxCheck("budget-tool", _mcp_cmd, _check_mcp),
    SandboxCheck("limits-not-inherited", _limits_cmd, _check_limits),
    SandboxCheck("pid1-recovery", _recovery_cmd, _check_recovery),
    SandboxCheck("route-reachable", _route_cmd, _check_route, sealed=True),
    SandboxCheck("unlisted-refused", _refused_cmd, _check_refused, sealed=True),
    SandboxCheck("icmp-held", _icmp_cmd, _check_icmp, sealed=True),
    SandboxCheck("index-reachable", _index_cmd, _check_index, sealed=True),
)

#: How the kernel counts the commands' process ceiling, checked against the site's accounting:
#: run with the seal probe, before the clock, in every build (``hls.stages.agent_build``), since a
#: site whose setting is not its kernel's would judge starved commands by the wrong level.
ACCOUNTING_CHECK = SandboxCheck("nproc-accounting", _accounting_cmd, _check_accounting)


Exec = Callable[..., Awaitable[Any]]


async def run_sandbox_checks(exec_: Exec, inputs: ProbeInputs, *, which: Literal["all", "sealed", "unsealed"],
                             timeout_sec: int, done: list[ProbeCheck] | None = None) -> list[ProbeCheck]:
    """Run each in-sandbox check of *which* kind through *exec_* (``SandboxEnvironment.exec``),
    each command with *timeout_sec*, and return them, appending each to *done* as it ends, so a
    caller keeps the checks that ran before a sandbox ended. A lost sandbox raises through
    *exec_*."""
    checks = [] if done is None else done
    for check in SANDBOX_CHECKS:
        if which != "all" and check.sealed != (which == "sealed"):
            continue
        results = [await exec_(command, timeout_sec=timeout_sec) for command in check.commands(inputs)]
        checks.append(check.parse(results, inputs))
    return checks


async def inbound_check(exec_: Exec, inputs: ProbeInputs, *, avoid: Collection[int], scratch: Path, target: str,
                        timeout_sec: int) -> ProbeCheck:
    """The build's inbound check, through *exec_*: a sandbox command (``seal_probe.hold``) binds a
    TCP and a UDP port the seal must not forward (``seal_probe.inbound_ports``, none of *avoid*,
    the ports the seal's namespace holds already, and none below ``NAMESPACE_PORT_FLOOR``, since
    it binds them unprivileged in pasta's fresh namespace) and holds them. Once it has made its ready file
    in *scratch* (*target* in the sandbox), the stage knocks on both from every address of the
    node, and only then makes the done file, so every knock found them held; the command closes
    them and says what reached them. It passes when every knock found its port closed and nothing
    reached it; a knock that found one open names the seal or the node (``knock_failures``).
    The command waits *timeout_sec* for the done file; the files are removed after. A lost
    sandbox raises through *exec_*."""
    expected = "no port a sandbox command binds is reached from any address of the node"

    def failed(observed: str) -> ProbeCheck:
        return ProbeCheck(name=INBOUND_CHECK, ok=False, observed=observed, expected=expected)

    try:
        tcp, udp = seal_probe.inbound_ports(avoid=avoid, floor=seal_probe.NAMESPACE_PORT_FLOOR)
    except OSError as e:
        return failed(str(e))
    ready, done = scratch / "ready", scratch / "done"
    # An old ready file would let the knocks run before the ports are bound.
    ready.unlink(missing_ok=True)
    done.unlink(missing_ok=True)
    command = _python(inputs, _probe_code(), "hold", "--inbound-tcp", str(tcp), "--inbound-udp", str(udp),
                      "--ready", f"{target}/ready", "--done", f"{target}/done", "--check-sec", str(timeout_sec))
    addresses = seal_probe.node_addresses()
    knocks: tuple[seal_probe.Knock, ...] = ()
    held = asyncio.ensure_future(exec_(command, timeout_sec=timeout_sec + math.ceil(seal_probe.INBOUND_SETTLE_SEC)))
    try:
        while not ready.exists() and not held.done():
            await asyncio.sleep(READY_POLL_SEC)
        if ready.exists():
            knocks = await asyncio.to_thread(seal_probe.knock_inbound, addresses, tcp, udp)
        done.touch()
        result = await held
    finally:
        if not held.done():
            held.cancel()
        ready.unlink(missing_ok=True)
        done.unlink(missing_ok=True)
    answer = seal_probe.held(_last(result))
    if answer is None:
        return failed(f"the command held no ports: {_last(result)[:80]}")
    check, reached = answer
    failures = [] if check.passed else [check.detail]
    failures += seal_probe.knock_failures(addresses, knocks, reached)
    if failures:
        return failed("; ".join(failures))
    return ProbeCheck(name=INBOUND_CHECK, ok=True, expected=expected,
                      observed=f"{check.detail}; refused from {', '.join(addresses)}")


def owner_check(path_uid: int | None, uid: int) -> ProbeCheck:
    """Whether the file a sandbox command made in the workspace is owned by the host user *uid*.
    *path_uid* is its owner host-side, or None when it was not made."""
    seen = "missing" if path_uid is None else str(path_uid)
    return ProbeCheck(name="workspace-owner", ok=path_uid == uid, observed=seen, expected=str(uid))


def readiness_check(start_sec: float, budget_sec: float) -> ProbeCheck:
    """Whether the exec server bound its socket within its budget."""
    return ProbeCheck(name="exec-server-ready", ok=start_sec <= budget_sec, observed=f"{start_sec:.1f}s",
                      expected=f"<= {budget_sec:.0f}s")


@dataclass(frozen=True)
class ReadyLine:
    """The exec server's ready record for the spawner it serves from (``exec_protocol.READY_NAME``)."""

    #: The spawner's pid, in the container's PID namespace.
    spawner_pid: int
    #: The reaper's pid, in the container's PID namespace.
    reaper_pid: int
    #: The ``oom_score_adj`` the server gave itself, and so the spawner and the reaper; None when
    #: it could not read it.
    score: int | None

    @property
    def pids(self) -> ReadyPids:
        """The spawner's and the reaper's pids, as the memory split finds them by."""
        return ReadyPids(spawner_pid=self.spawner_pid, reaper_pid=self.reaper_pid)


def read_ready(path: Path) -> ReadyLine | None:
    """The exec server's ready record in the file *path* (``exec_protocol.READY_NAME``), read now;
    None when it is not there or cannot be read, holds no record yet, or was caught mid-write
    (``exec_protocol.parse_ready_record``): a torn record names no pair rather than a wrong one."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    record = exec_protocol.parse_ready_record(data)
    if record is None:
        return None
    return ReadyLine(spawner_pid=record["spawner_pid"], reaper_pid=record["reaper_pid"], score=record["score"])


def ready_pids(path: Path) -> ReadyPids | None:
    """The serving spawner's and reaper's pids, from the ready file *path* read fresh on every
    call, never kept: the memory split's source (``StageContext.start_container``'s
    *ready_source*). None until the server has written a whole record."""
    ready = read_ready(path)
    return ready.pids if ready is not None else None


def server_score_check(ready: ReadyLine | None) -> ProbeCheck:
    """Whether the exec server's ready record *ready* says it holds ``oom_score_adj`` 0 in a step
    the wrapper raised to ``role_exec.CONTESTANT_OOM_SCORE_ADJ``."""
    if ready is None:
        seen = "no ready line"
    else:
        seen = f"oom_score_adj={'unknown' if ready.score is None else ready.score}"
    return ProbeCheck(name="server-oom-score", ok=ready is not None and ready.score == 0, observed=seen,
                      expected="oom_score_adj=0")


def _step_readable(cgroup_dir: Path | None, proc: Path) -> list[ProcessMemory]:
    """Each process of the step that can be read; [] when there is no step. Unlike the memory
    split, a process that cannot be read is left out: the checks only look for the harness's own."""
    if cgroup_dir is None:
        return []
    found = []
    for pid in pids(cgroup_dir):
        try:
            process = read_process(proc, pid)
        except (OSError, ValueError):
            continue
        if process is not None:
            found.append(process)
    return found


def _serving(ready: ReadyLine | None, cgroup_dir: Path | None,
             proc: Path) -> tuple[tuple[ProcessMemory, ProcessMemory] | None, list[ProcessMemory], str]:
    """The serving spawner and reaper among the step's processes, found by the pids the ready
    record *ready* names as the memory split finds them (``hls.agent.memory.find_reaper``), never
    by a fixed depth: the seal nests them in pasta's and apptainer's PID namespaces. Also the
    step's readable processes, and why the pair is None when it is."""
    if ready is None:
        return None, [], "no ready line"
    found = _step_readable(cgroup_dir, proc)
    if not found:
        return None, [], "no step processes"
    pair = find_reaper(found, ready.pids)
    if pair is None:
        return None, found, "no reaper one namespace below the serving spawner"
    return pair, found, ""


def spawner_score_check(ready: ReadyLine | None, cgroup_dir: Path | None, *, proc: Path = Path("/proc")) -> ProbeCheck:
    """Whether the serving spawner holds ``oom_score_adj`` 0, read host side through the step's
    processes, since no command can see it."""
    pair, _, seen = _serving(ready, cgroup_dir, proc)
    if pair is not None:
        try:
            seen = f"oom_score_adj={(proc / str(pair[0].pid) / 'oom_score_adj').read_text().strip()}"
        except OSError as e:
            seen = f"unreadable: {type(e).__name__}"
    return ProbeCheck(name="spawner-oom-score", ok=seen == "oom_score_adj=0", observed=seen, expected="oom_score_adj=0")


def spawner_limits_check(ready: ReadyLine | None, cgroup_dir: Path | None, hard: int | None, *,
                         proc: Path = Path("/proc")) -> ProbeCheck:
    """Whether the serving spawner, which made the commands' user namespace, holds the runner's
    hard ``RLIMIT_NPROC`` *hard* (None for unlimited) as its own soft and hard limit, read host side
    from its ``/proc/<pid>/limits``: the limit that namespace's parent levels count every task of
    the uid against (``hls.substrate.nproc.CeilingRule.parent_limit``)."""
    want = "unlimited" if hard is None else str(hard)
    expected = f"Max processes {want} {want}"
    pair, _, seen = _serving(ready, cgroup_dir, proc)
    if pair is not None:
        try:
            lines = (proc / str(pair[0].pid) / "limits").read_text().splitlines()
            fields = next(line for line in lines if line.startswith("Max processes")).split()
            seen = f"Max processes {fields[2]} {fields[3]}"
        except (OSError, StopIteration, IndexError) as e:
            seen = f"unreadable: {type(e).__name__}"
    return ProbeCheck(name="spawner-limits", ok=seen == expected, observed=seen, expected=expected)


def depth_check(ready: ReadyLine | None, cgroup_dir: Path | None, *, proc: Path = Path("/proc")) -> ProbeCheck:
    """Whether the commands run in a PID namespace nested one below the serving spawner's (the
    container's): the reaper, PID 1 of the commands' namespace, is found one namespace below the
    spawner by the ready record's pids (``hls.agent.memory.find_reaper``). A reaper at the spawner's
    own depth would leave the commands in the exec server's namespace, and is not found. Records
    how many namespaces below the step's base the commands run, which the seal's layers (pasta,
    apptainer, the spawner's own) decide: the depth the OOM split keys the contestant off."""
    expected = "the commands' PID 1 one namespace below the serving spawner"
    pair, found, why = _serving(ready, cgroup_dir, proc)
    if pair is None:
        return ProbeCheck(name="nspid-depth", ok=False, observed=why, expected=expected)
    below = pair[1].depth - min(p.depth for p in found)
    return ProbeCheck(name="nspid-depth", ok=True, observed=f"the reaper at the step's base + {below}",
                      expected=expected)


def reaper_socket_check(ready: ReadyLine | None, cgroup_dir: Path | None, *, proc: Path = Path("/proc")) -> ProbeCheck:
    """Whether the serving reaper, the one harness process a command can name, holds no socket:
    its descriptors, read host side, are its standard three."""
    pair, _, why = _serving(ready, cgroup_dir, proc)
    expected = "no socket among the reaper's descriptors"
    if pair is None:
        return ProbeCheck(name="reaper-no-socket", ok=False, observed=why, expected=expected)
    pid = pair[1].pid
    try:
        fds = {int(fd): os.readlink(proc / str(pid) / "fd" / fd) for fd in os.listdir(proc / str(pid) / "fd")}
    except OSError as e:
        return ProbeCheck(name="reaper-no-socket", ok=False, observed=f"unreadable: {type(e).__name__}",
                          expected=expected)
    sockets = [fd for fd, target in fds.items() if target.startswith("socket:")]
    return ProbeCheck(name="reaper-no-socket", ok=not sockets,
                      observed=" ".join(f"{fd}:{fds[fd]}" for fd in sorted(fds)) or "none", expected=expected)


def refusal_recorded_check(refused: Collection[str] | None, host: str | None) -> ProbeCheck:
    """Whether the seal's interceptor recorded that its policy refused *host*, from the hosts it
    recorded (*refused*; None when it wrote no valid status). Passes with nothing to record when
    the policy refuses no host (*host* None)."""
    expected = "the seal's interceptor recorded the refused host by its policy"
    if host is None:
        return ProbeCheck(name="refusal-recorded", ok=True, observed="the policy refuses no host", expected=expected)
    if refused is None:
        return ProbeCheck(name="refusal-recorded", ok=False, observed="no interceptor status", expected=expected)
    return ProbeCheck(name="refusal-recorded", ok=host in refused,
                      observed="recorded" if host in refused else f"{len(refused)} other hosts recorded",
                      expected=expected)


def env_file_check(env: Mapping[str, str], route_vars: Collection[str], secrets: Collection[str]) -> ProbeCheck:
    """Whether the sandbox's env file, *env* as the stage wrote it, holds no route variable of the
    arm (*route_vars*: its grants and its plain settings) and no secret's value (*secrets*). It
    names the variables that do, never a value."""
    bad = sorted(k for k, v in env.items() if k in route_vars or any(s and s in v for s in secrets))
    return ProbeCheck(name="env-file-clean", ok=not bad, observed=", ".join(bad) or "clean",
                      expected="no route variable and no secret's value")


def split_check(sample: SplitSample | None) -> ProbeCheck:
    """One split sample of the step as a check that always passes: a measurement for the record."""
    observed = (f"harness={sample.harness} contestant={sample.contestant} tmpfs={sample.tmpfs}"
                if sample is not None else "no sample")
    return ProbeCheck(name="memory-split", ok=True, observed=observed,
                      expected="the harness and contestant resident memory of the step")


def pid_max_check(ceiling: int, *, path: Path = Path("/proc/sys/kernel/pid_max")) -> ProbeCheck:
    """The node's ``pid_max`` beside *ceiling*, the process ceiling every command runs under
    (``hls.substrate.nproc``): it passes when the ceiling is below ``pid_max``, so a fork storm in
    the commands meets its ceiling before it can take every pid of the node."""
    try:
        pid_max = int(path.read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return ProbeCheck(name="pid-max", ok=False, observed=f"pid_max unreadable ceiling={ceiling}",
                          expected="the commands' ceiling below the node's pid_max")
    return ProbeCheck(name="pid-max", ok=ceiling < pid_max, observed=f"pid_max={pid_max} ceiling={ceiling}",
                      expected="the commands' ceiling below the node's pid_max")


#: The namespace kinds ``harness-ucounts`` counts.
_NS_KINDS = ("user", "mnt", "pid", "net", "ipc", "uts", "cgroup")


def harness_ucounts_check(cgroup_dir: Path | None, *, proc: Path = Path("/proc")) -> ProbeCheck:
    """The namespaces of each kind the step's processes hold, and the inotify instances each
    process name holds, as a check that always passes: the measurement the per-evaluation
    reserves of ``hls.substrate.ucounts`` are for. A process the host may not read (the
    non-dumpable reaper) counts nothing."""
    spaces: dict[str, set[str]] = {kind: set() for kind in _NS_KINDS}
    inotify: dict[str, int] = {}
    read = False
    if cgroup_dir is not None:
        try:
            processes = step_processes(cgroup_dir, proc=proc)
        except (OSError, ValueError):
            processes = []
        for p in processes:
            base = proc / str(p.pid)
            for kind in _NS_KINDS:
                try:
                    spaces[kind].add(os.readlink(base / "ns" / kind))
                except OSError:
                    continue
            try:
                held = sum(1 for fd in os.listdir(base / "fd") if os.readlink(base / "fd" / fd) == "anon_inode:inotify")
            except OSError:
                continue
            read = True
            if held:
                try:
                    # A process names itself; a name that is not UTF-8 is kept with its bytes replaced.
                    name = (base / "comm").read_bytes().decode(errors="replace").strip()
                except OSError:
                    name = "unnamed"
                inotify[name] = inotify.get(name, 0) + held
    counts = " ".join(f"{kind}={len(seen)}" for kind, seen in spaces.items())
    held_by = ",".join(f"{name}:{n}" for name, n in sorted(inotify.items())) or "none"
    return ProbeCheck(name="harness-ucounts", ok=True, observed=f"{counts} inotify={held_by}" if read else "unread",
                      expected="a measurement")


def outside_check(ready: ReadyLine | None, cgroup_dir: Path | None, *, proc: Path = Path("/proc")) -> ProbeCheck:
    """Whether every process of the step outside the sandbox's container holds
    ``role_exec.CONTESTANT_OOM_SCORE_ADJ``, the score the step wrapper gives the whole step and only
    the harness in the container lowers, recording each one's name, score and resident memory
    for the OOM rule's reserve. Outside the container is a PID namespace above the serving
    spawner's (``_serving``): apptainer's starter and, sealed, pasta, the seal's shell and the
    interceptor. A Docker container is the whole step, its seal's gateway a container of its
    own, so there none is outside. It fails when the serving pair cannot be found or a score
    cannot be read."""
    expected = f"each at oom_score_adj {role_exec.CONTESTANT_OOM_SCORE_ADJ}, with its resident bytes"
    pair, found, why = _serving(ready, cgroup_dir, proc)
    if pair is None:
        return ProbeCheck(name="outside-scores", ok=False, observed=why, expected=expected)
    seen, ok = [], True
    for p in sorted((p for p in found if p.depth < pair[0].depth), key=lambda p: p.pid):
        try:
            # A process names itself; a name that is not UTF-8 is recorded with its bytes replaced.
            name = (proc / str(p.pid) / "comm").read_bytes().decode(errors="replace").strip()
            score = (proc / str(p.pid) / "oom_score_adj").read_text().strip()
        except OSError as e:
            seen.append(f"{p.pid}:unreadable: {type(e).__name__}")
            ok = False
            continue
        ok = ok and score == str(role_exec.CONTESTANT_OOM_SCORE_ADJ)
        seen.append(f"{name}:{score}:{p.rss}")
    return ProbeCheck(name="outside-scores", ok=ok, observed=" ".join(seen) or "none outside the container",
                      expected=expected)


def record(checks: list[ProbeCheck], *, oom: OomProbe | None = None) -> ProbeRecord:
    """The record of *checks*, with the OOM probe's result."""
    return ProbeRecord(checks=checks, all_ok=all(c.ok for c in checks), oom=oom)
