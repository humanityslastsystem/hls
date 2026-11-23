"""The wrapper every role step runs first. It says that its step started, measures the step's
grant, writes the report, refuses a slice that differs from the role's declaration or a command
that is not an executable file, and then becomes the role's command.

The launcher builds the command line::

    python -m hls.substrate.role_exec --report <file> --gpus N [--gpu-kind K] --cpus C
        --memory-gib M --disk-gib D --scratch <dir> [--share-gpus <uuid,...>]
        [--env-file <file> [--env-fd]] [--oom-score-adj A] [--nproc N] [--one-numa-node] -- <argv...>

Once it has taken its env file (below), the wrapper writes its step's id into the start marker
beside the report (``started_path``). That is the first sign of the step the runner can see
without asking the scheduler, so the runner's wait for the scheduler to start the step ends
there and its shorter wait for the report begins (``hls.pipeline.stage``). No code of the role
has run by then.

``--oom-score-adj`` is for a contestant's role: the wrapper sets its own ``oom_score_adj`` to
A before anything else of the role runs, and the command and everything it starts, a
container's processes among them, inherit it. An OOM killer that chooses among more
processes than the role's then takes the contestant's first. It is a
protection, which a contestant's process can lower again, not evidence of whose limit was
reached. A value the wrapper cannot set refuses the role.

``--nproc`` sets the step's soft and hard ``RLIMIT_NPROC`` to N, before anything else of the
role runs, so the command and every process it starts, a container's among them, inherit it:
the count of its uid's tasks on the node past which none of them may fork
(``hls.substrate.nproc``). Set here, before any user namespace is made, it counts every task
of the uid on the node on any kernel, a sealed server's included: each namespace made under it,
pasta's among them, records it as the limit of its parent level, whatever
``site.nproc_accounting`` says of the build sandbox's commands. The wrapper first counts those
tasks; when they already number N or more, the step could fork nothing, the grant's measurement
among it, so the wrapper writes no report, says so on stderr and exits ``PROCESS_CEILING_EXIT``
without forking. A limit the wrapper cannot set refuses the role.

The report names the CPUs the step may use, the NUMA nodes they lie on, and whether they are
split (``placement``): on more than one node although one of those nodes holds as many CPUs as
the step has, as when Slurm gives eight CPUs four on each socket of a node whose free cores are
spread. Where a step's CPUs lie changes how fast a single-threaded hot path runs,
so a stage's record says which a role had. A node directory whose CPU list cannot be read
counts as no node. ``--one-numa-node`` refuses a split step, naming its CPUs and their nodes
(``split_refusal``): a scored server on a site with ``site.evaluation_socket_tasks``, whose
allocation asked for its CPUs as a task on one socket. Without it a split step is recorded and
starts.

``--share-gpus`` is for the one deliberate overlap. That step is given every GPU of the
allocation, and the wrapper sets CUDA_VISIBLE_DEVICES to the shared role's UUIDs inside the
step, after Slurm has set its own value.

``--env-file`` names the role's env file, which holds its secrets. The wrapper's first act,
once the interpreter has imported it and its arguments are parsed, is to open the file, unlink
it and read it, so the file is gone whether the role starts or not, and it refuses a file that
is over 64 KiB or is not ``NAME='value'`` lines. A wrapper that fails before that, in its
imports or its arguments, leaves the file in the role's node-local directory: the stage removes
it when it reports the role, and nothing does after a SIGKILL of the runner. With ``--env-fd``
the content is put in a pipe whose write end is then closed, and the read end is descriptor 3
for the command: ``apptainer --env-file /dev/fd/3`` reads it once, and every copy of the
descriptor the container's processes inherit after that reads end of file. An unlinked file
left open on descriptor 3 would stay readable through /proc/<pid>/fd/3 for as long as the
container runs. Without ``--env-fd`` the variables join the command's environment. Either way
the values never touch argv or a file anyone else can open.

A container started with ``--cleanenv`` sees none of the step's environment, so the wrapper
also exports ``APPTAINERENV_CUDA_VISIBLE_DEVICES`` with the step's final CUDA_VISIBLE_DEVICES:
the container then sees exactly the step's GPUs.

A site's TaskProlog adds variables to every step after the launcher built its environment
(such as APPTAINER_CACHEDIR, BASH_ENV, XDG_RUNTIME_DIR and SRUN_CPUS_PER_TASK). Before
it forwards the GPUs, the wrapper removes those that would change what a role runs
(``dropped``), and the report names every variable it removed:
- from every role, BASH_ENV, which makes every non-interactive bash, such as the network
  seal's, read a file of the site's before its command; and apptainer's settings and the
  variables it copies into a container (``RUNTIME_PREFIXES``: ``APPTAINER_BIND`` adds binds
  that ``--no-mount bind-paths`` does not drop, ``APPTAINERENV_X`` sets X inside), but for
  APPTAINER_CACHEDIR and APPTAINER_TMPDIR (``APPTAINER_KEPT``), which only say where a host
  role's apptainer keeps its cache and its build files;
- from a container role, with ``--env-fd``, which is how a container takes its variables, also
  those two and every other variable apptainer reads from the environment it starts in
  (``read_by_runtime``): ``--cleanenv`` still passes TERM and the proxy variables."""

from __future__ import annotations

import argparse
import os
import resource
import shutil
import socket
import stat
import sys
from collections.abc import Iterable, MutableMapping
from dataclasses import dataclass
from pathlib import Path

from hls.spec.resources import ResourceShape, check_grant
from hls.substrate.base import RoleReport, parse_cpulist
from hls.substrate.cgroup import CGROUP_ROOT, own_cgroup
from hls.substrate.exits import GRANT_MISMATCH_EXIT, PROCESS_CEILING_EXIT
from hls.substrate.factory import step
from hls.substrate.grant import Measured, measure
from hls.substrate.nproc import uid_tasks
from hls.substrate.secrets import parse_env_text

#: Largest env file the wrapper takes: the capacity a pipe has by default, so the whole content
#: is written into one before anything reads it.
MAX_ENV_BYTES = 65536
#: Descriptor the env file's pipe is handed to the command on, with ``--env-fd``.
ENV_FD = 3
#: The path a command opens to read that descriptor.
ENV_FD_PATH = "/dev/fd/3"
#: The variable apptainer copies into a container as CUDA_VISIBLE_DEVICES, past --cleanenv.
CONTAINER_GPUS_VAR = "APPTAINERENV_CUDA_VISIBLE_DEVICES"
#: The ``oom_score_adj`` a contestant's role starts with, the highest there is.
CONTESTANT_OOM_SCORE_ADJ = 1000
#: Where a process sets its own ``oom_score_adj``. Raising it needs no privilege.
OOM_SCORE_ADJ_PATH = Path("/proc/self/oom_score_adj")
#: The startup file variable bash reads before a non-interactive command.
STARTUP_FILE_VAR = "BASH_ENV"
#: The apptainer variables a host role keeps: where apptainer keeps its cache and its build
#: files. A container role's step loses them too. The launcher lets a role declare them
#: (``hls.substrate.slurm.launcher.step_env``).
APPTAINER_KEPT = frozenset({"APPTAINER_CACHEDIR", "APPTAINER_TMPDIR"})
#: Where the kernel lists the CPUs of each NUMA node, one ``node<N>/cpulist`` per node.
NODE_ROOT = Path("/sys/devices/system/node")


@dataclass(frozen=True)
class Placement:
    """Where a step's CPUs lie."""

    #: The CPUs, ascending.
    cpus: list[int]
    #: The NUMA nodes they lie on, ascending; empty when the kernel lists no node.
    numa_nodes: list[int]
    #: True when they lie on more than one node although one of those nodes holds as many CPUs.
    split: bool


def placement(cpus: Iterable[int], *, node_root: Path) -> Placement:
    """Where *cpus* lie among the NUMA nodes *node_root* lists."""
    mine = sorted(set(cpus))
    nodes: dict[int, set[int]] = {}
    for path in node_root.glob("node[0-9]*"):
        if path.name[len("node"):].isdecimal():
            try:
                nodes[int(path.name[len("node"):])] = parse_cpulist((path / "cpulist").read_text())
            except (OSError, ValueError):
                continue
    spanned = sorted(n for n, members in nodes.items() if members & set(mine))
    split = len(spanned) > 1 and max(len(nodes[n]) for n in spanned) >= len(mine)
    return Placement(cpus=mine, numa_nodes=spanned, split=split)


def split_refusal(where: Placement) -> str | None:
    """Why a role held to one NUMA node (``--one-numa-node``) may not start at *where*, naming
    its CPUs and their nodes; None when they are not split."""
    if not where.split:
        return None
    return (f"cpus {where.cpus} lie on NUMA nodes {where.numa_nodes}, though one node holds as many; this role runs "
            "on one node (site.evaluation_socket_tasks)")


def own_cpus() -> list[int]:
    """The CPUs this process may run on."""
    return sorted(os.sched_getaffinity(0))


def check_role(declared: ResourceShape, measured: Measured, *, shared: list[str] | None) -> list[str]:
    """Every way *measured* differs from *declared*. For a shared step, the GPUs that count
    are the *shared* UUIDs, each of which the step must be able to open."""
    problems = list(measured.problems)
    granted = measured.granted
    if shared is not None:
        missing = [u for u in shared if u not in measured.gpu_uuids]
        if missing:
            problems.append(f"shared GPUs not visible to this step: {missing}")
        granted = granted.model_copy(update={"gpus": len(shared)})
    problems += [f"{m.field}: declared {m.declared}, granted {m.granted}" for m in check_grant(declared, granted)]
    return problems


def _resolve_command(command: str) -> str | None:
    """The file *command* would run: itself when it names a path, else the first match on
    the step's PATH. None unless that is a regular file this process may execute."""
    found = command if "/" in command else shutil.which(command, path=os.environ.get("PATH", os.defpath))
    if found is not None and os.path.isfile(found) and os.access(found, os.X_OK):
        return found
    return None


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.substrate.role_exec")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--gpus", type=int, required=True)
    p.add_argument("--gpu-kind")
    p.add_argument("--cpus", type=int, required=True)
    p.add_argument("--memory-gib", type=int, required=True)
    p.add_argument("--disk-gib", type=int, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    p.add_argument("--share-gpus")
    p.add_argument("--env-file", type=Path)
    p.add_argument("--env-fd", action="store_true")
    p.add_argument("--oom-score-adj", type=int)
    p.add_argument("--nproc", type=int)
    p.add_argument("--one-numa-node", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        p.error("no command after --")
    if args.env_fd and args.env_file is None:
        p.error("--env-fd needs --env-file")
    return args


def _read_limited(fd: int) -> bytes:
    """What *fd* holds from where it stands, up to ``MAX_ENV_BYTES`` bytes and one more if
    there is more, which tells the caller the content is too large."""
    data = b""
    while len(data) <= MAX_ENV_BYTES:
        chunk = os.read(fd, MAX_ENV_BYTES + 1 - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _problem_with(data: bytes) -> str | None:
    """Why *data* is not an env file the wrapper passes on, or None. Never quotes the content."""
    if len(data) > MAX_ENV_BYTES:
        return f"is larger than {MAX_ENV_BYTES} bytes"
    try:
        parse_env_text(data.decode())
    except UnicodeDecodeError:
        return "is malformed: not UTF-8 text"
    except ValueError as e:
        return f"is malformed: {e}"
    return None


def _pipe_holding(data: bytes) -> tuple[int | None, str | None]:
    """A pipe that holds *data* and has no writer: its read end, or None and what stopped it.
    The write does not wait, so a pipe that cannot hold all of *data* (the kernel gives one
    page to a user who has too many pipes) is an error and not a partial or a stuck write."""
    try:
        read_end, write_end = os.pipe()
    except OSError as e:
        return None, f"cannot be handed over: {type(e).__name__}"
    try:
        os.set_blocking(write_end, False)
        try:
            written = os.write(write_end, data)
        except BlockingIOError:
            written = 0
    finally:
        os.close(write_end)
    if written != len(data):
        os.close(read_end)
        return None, "does not fit in a pipe"
    return read_end, None


def take_env_file(path: Path) -> tuple[int | None, str | None]:
    """Open *path*, unlink it at once, and move what it holds into a pipe. Return the pipe's
    read end, or None and why the file cannot be used: unreadable, not a regular file, not
    removable, over ``MAX_ENV_BYTES``, not UTF-8, not ``NAME='value'`` lines, or not fitting in
    a pipe. It is unlinked even when it cannot be used, so nothing is left behind on a refusal
    (a directory cannot be, and stays). The pipe has no writer, so whoever has read its
    content once, and every copy of the descriptor made after that, reads end of file."""
    problem = None
    try:
        # O_NONBLOCK keeps a named pipe at the path from holding the open; it does nothing to a regular file.
        file_fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except OSError as e:
        file_fd, problem = None, f"cannot be read: {type(e).__name__}"
    else:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            os.close(file_fd)
            file_fd, problem = None, "is not a regular file"
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        problem = problem or f"cannot be removed: {type(e).__name__}"
    data = b""
    if file_fd is not None:
        try:
            if problem is None:
                data = _read_limited(file_fd)
        except OSError as e:
            problem = f"cannot be read: {type(e).__name__}"
        finally:
            os.close(file_fd)
    if problem is None:
        problem = _problem_with(data)
    pipe_fd = None
    if problem is None:
        pipe_fd, problem = _pipe_holding(data)
    return pipe_fd, None if problem is None else f"env file {path} {problem}"


def dropped(name: str, *, container: bool) -> bool:
    """Whether the wrapper removes variable *name* from a role's environment before the role
    starts: BASH_ENV, and apptainer's variables but those in ``APPTAINER_KEPT``, from every
    role; from a *container* role, BASH_ENV and every variable apptainer reads
    (``Step.runtime_reads``)."""
    if name == STARTUP_FILE_VAR:
        return True
    here = step(os.environ)
    if container:
        return here.runtime_reads(name)
    return here.runtime_setting(name) and name not in APPTAINER_KEPT


def drop_site_variables(environ: MutableMapping[str, str], *, container: bool) -> list[str]:
    """Remove from *environ* every variable ``dropped`` names, and return their names, sorted."""
    names = sorted(k for k in environ if dropped(k, container=container))
    for name in names:
        del environ[name]
    return names


def set_oom_score_adj(value: int) -> str | None:
    """Set this process's ``oom_score_adj`` to *value*, which the command it becomes and every
    process that starts inherit; why that failed, or None."""
    try:
        OOM_SCORE_ADJ_PATH.write_text(f"{value}\n")
    except OSError as e:
        return f"oom_score_adj {value} could not be set: {e}"
    return None


def set_process_limit(ceiling: int) -> str | None:
    """Set this process's soft and hard ``RLIMIT_NPROC`` to *ceiling*, which the command it
    becomes and every process that starts inherit; why that failed, or None."""
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (ceiling, ceiling))
    except (OSError, ValueError) as e:
        return f"RLIMIT_NPROC {ceiling} could not be set: {e}"
    return None


def started_path(report: Path) -> Path:
    """The start marker of the step whose grant report is *report*: ``<role>.started`` beside
    ``<role>.report.json``."""
    return report.with_name(report.name.removesuffix(".report.json") + ".started")


def _write_whole(path: Path, text: str) -> None:
    """Write *text* to *path* so a reader sees all of it or nothing."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_report(path: Path, report: RoleReport) -> None:
    """Write *report* so a reader sees all of it or nothing."""
    _write_whole(path, report.model_dump_json(indent=2))


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    # The env file goes first, before anything else can fail or wait.
    env_fd, env_problem = take_env_file(args.env_file) if args.env_file is not None else (None, None)
    here = step(os.environ)
    step_id = here.step_id(os.environ)
    _write_whole(started_path(args.report), f"{step_id}\n")
    oom_problem = set_oom_score_adj(args.oom_score_adj) if args.oom_score_adj is not None else None
    if args.nproc is not None:
        try:
            held: int | None = uid_tasks(os.getuid())
        except Exception as e:  # noqa: BLE001 -- the check only spares a start that could fork nothing
            # The step's end is judged by the pool counted when it is reported, so a count that
            # fails here starts the step; it never fails the launch.
            print(f"role_exec: the uid's tasks could not be counted ({type(e).__name__}: {e}); starting the step "
                  "without that check", file=sys.stderr)
            held = None
        if held is not None and held >= args.nproc:
            if env_fd is not None:
                os.close(env_fd)
            print(f"role_exec: refusing to start: uid {os.getuid()} holds {held} tasks on this node, as many as "
                  f"the step's process ceiling of {args.nproc} or more, so the step could fork nothing",
                  file=sys.stderr)
            return PROCESS_CEILING_EXIT
    nproc_problem = set_process_limit(args.nproc) if args.nproc is not None else None
    declared = ResourceShape(gpus=args.gpus, gpu_kind=args.gpu_kind, cpus=args.cpus, memory_gib=args.memory_gib,
                             disk_gib=args.disk_gib, time_sec=1)
    shared = args.share_gpus.split(",") if args.share_gpus else None
    if shared is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(shared)
    # Whatever the site added to the step after the launcher built it goes before the GPUs are
    # forwarded, which would otherwise be removed with it.
    stripped = drop_site_variables(os.environ, container=args.env_fd)
    gpus = os.environ.get("CUDA_VISIBLE_DEVICES")
    if gpus is None:
        os.environ.pop(CONTAINER_GPUS_VAR, None)
    else:
        os.environ[CONTAINER_GPUS_VAR] = gpus
    limit_dir = here.step_cgroup(own_cgroup(), CGROUP_ROOT)
    measured = measure(limit_dir=limit_dir, scratch=args.scratch)
    mismatches = check_role(declared, measured, shared=shared)
    # execv does not search PATH, and a command that cannot start would otherwise be reported
    # as a role that ran.
    program = _resolve_command(args.command[0])
    if program is None:
        mismatches.append(f"command {args.command[0]!r} is not an executable file")
    if env_problem is not None:
        mismatches.append(env_problem)
    if oom_problem is not None:
        mismatches.append(oom_problem)
    if nproc_problem is not None:
        mismatches.append(nproc_problem)
    where = placement(own_cpus(), node_root=NODE_ROOT)
    refusal = split_refusal(where) if args.one_numa_node else None
    if refusal is not None:
        mismatches.append(refusal)
    report = RoleReport(
        host=socket.gethostname(),
        step=step_id,
        cgroup_dir=str(limit_dir),
        gpu_uuids=measured.gpu_uuids,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        granted=measured.granted,
        swap_max_bytes=measured.swap_max_bytes,
        mismatches=mismatches,
        stripped_env=stripped,
        cpus=where.cpus,
        numa_nodes=where.numa_nodes,
        numa_split=where.split,
    )
    write_report(args.report, report)
    if mismatches:
        if env_fd is not None:
            os.close(env_fd)
        print(f"role_exec: refusing to start, the step's grant differs: {'; '.join(mismatches)}", file=sys.stderr)
        return GRANT_MISMATCH_EXIT
    if env_fd is not None:
        _hand_over(env_fd, on_fd=args.env_fd)
    os.execv(program, args.command)
    return 0  # not reached


def _hand_over(fd: int, *, on_fd: bool) -> None:
    """Give the command the env file's content, held by the pipe open on *fd*: that pipe on
    ``ENV_FD``, or its variables in the command's environment."""
    if on_fd:
        if fd == ENV_FD:
            os.set_inheritable(fd, True)
        else:
            # dup2 makes the copy inheritable, so it survives the exec.
            os.dup2(fd, ENV_FD)
            os.close(fd)
        return
    try:
        data = _read_limited(fd)
    finally:
        os.close(fd)
    os.environ.update(parse_env_text(data.decode()))


if __name__ == "__main__":
    sys.exit(main())
