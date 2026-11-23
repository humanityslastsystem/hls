"""``Scheduler`` on Slurm: sbatch to submit, squeue and sacct to observe, scontrol to rewire
a pending dependency, and scancel by job id.

Slurm behavior the scheduler relies on:
- A submission carries a unique ``--comment`` tag. When sbatch's answer is lost, the job is
  looked up by that tag in squeue before anything is submitted again. Accounting stores
  comments only where ``AccountingStoreFlags`` has ``job_comment``, so it is not asked.
- A dependent is submitted with ``--kill-on-invalid-dep=no``. When its parent fails it stays
  pending as DependencyNeverSatisfied until it is rewired or cancelled, whatever the site's
  default is.
- A singleton submission adds Slurm's ``singleton`` to its dependency, after any afterok, so
  jobs of one name run one at a time. It needs no ``--kill-on-invalid-dep``: a singleton
  dependency is never invalid.
- squeue answers for queued and running jobs; sacct for jobs that left the queue. A job
  neither reports on is UNKNOWN, which counts as alive.
- Times are epoch seconds: the query tools run with ``SLURM_TIME_FORMAT=%s``.
- Jobs are cancelled one id at a time; nothing here cancels jobs by name or user. Only the
  queue is read per user: squeue lists this user's jobs (``--me``), and each one is then
  matched by id or by its tag."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping, Sequence

from hls.substrate.base import (DependencyRefused, JobEnd, JobInfo, JobState, Refused, Scheduler,
                                SchedulerUnavailable, SubmitRequest, SubmitUnknown)
from hls.substrate.slurm.commands import CommandResult, SlurmCommands
from hls.substrate.slurm.resources import SlurmPlacement, placement_args

log = logging.getLogger(__name__)

_JOB_ID = re.compile(r"[0-9]+")
#: The refusal of a plan made on a host whose PATH holds no ``sbatch``.
NO_SBATCH = ("sbatch is not on this host's PATH: run hls on a host of the Slurm cluster, "
             'or set site.substrate = "docker" on a Docker host')
# squeue fields: job id, state, pending reason, start time, comment. The comment is last, so a
# "|" in someone else's comment cannot shift the other fields.
_SQUEUE_FORMAT = "%i|%T|%r|%S|%k"
# States Slurm prints that are not JobState names, mapped to the one that means the same here.
_STATE_ALIASES = {
    "CONFIGURING": JobState.RUNNING,
    "COMPLETING": JobState.RUNNING,
    "SUSPENDED": JobState.RUNNING,
    "STOPPED": JobState.RUNNING,
    "SIGNALING": JobState.RUNNING,
    "STAGE_OUT": JobState.RUNNING,
    "RESIZING": JobState.RUNNING,
    "REQUEUED": JobState.PENDING,
    "REQUEUE_FED": JobState.PENDING,
    "REQUEUE_HOLD": JobState.PENDING,
    "RESV_DEL_HOLD": JobState.PENDING,
    "SPECIAL_EXIT": JobState.FAILED,
    "REVOKED": JobState.CANCELLED,
}
# What scancel prints for a job that is already over; that counts as cancelled.
_GONE = re.compile(r"invalid job id|already completing or completed|job has already finished", re.IGNORECASE)
# What sbatch and scontrol print when a job's dependency is invalid, such as a purged job id.
_DEPENDENCY = re.compile(r"job dependency problem", re.IGNORECASE)
# Fields of ``scontrol show job``, each read from the line Slurm prints it on, which starts with
# it: the job name, which is printed first, cannot then speak for them.
_END_TIME = re.compile(r"^[ \t]*StartTime=\S+[ \t]+EndTime=(\S+)", re.M)
_STATE_LINE = re.compile(r"^[ \t]*JobState=(\S+)[ \t]+Reason=(\S+)", re.M)
_EPOCH_SECONDS = re.compile(r"[0-9]+")


def check_job_id(job_id: str) -> str:
    """Return *job_id* if it is a plain numeric Slurm job id, else raise ValueError. That
    keeps a malformed id from ever reaching a command line as an option."""
    if not _JOB_ID.fullmatch(job_id):
        raise ValueError(f"not a Slurm job id: {job_id!r}")
    return job_id


def normalize_state(raw: str) -> JobState:
    """The JobState for a state as squeue or sacct prints it, such as "CANCELLED by 1234"
    or "COMPLETING"; UNKNOWN for anything unrecognized."""
    word = raw.strip().split(" ", 1)[0].rstrip("+")
    if word in JobState.__members__:
        return JobState(word)
    return _STATE_ALIASES.get(word, JobState.UNKNOWN)


def parse_slurm_time(text: str) -> float | None:
    """A time as Slurm prints it under ``SLURM_TIME_FORMAT=%s``, integer POSIX seconds; None
    for "N/A", "Unknown", "None", an empty field and anything else that is not a time."""
    text = text.strip()
    return float(text) if _EPOCH_SECONDS.fullmatch(text) else None


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _why(r: CommandResult) -> str:
    """One line on why a command failed: what it printed, else how it ended."""
    tool = os.path.basename(r.argv[0])
    return _one_line(r.stderr) or (f"{tool} timed out" if r.returncode is None else f"{tool} exited {r.returncode}")


def _refusal(r: CommandResult) -> Refused:
    """The error for a command the scheduler answered and refused."""
    return DependencyRefused(_why(r)) if _DEPENDENCY.search(r.stderr) else Refused(_why(r))


def _job_info(job_id: str, state: str, *, start: str, reason: str | None = None, exit_code: str = "") -> JobInfo:
    """A JobInfo from the printed fields. A pending job's start field is only an estimate,
    and sacct prints an exit code of 0:0 for a job that has not ended, so each is kept
    only where it means something."""
    normalized = normalize_state(state)
    return JobInfo(
        job_id,
        normalized,
        reason=reason,
        exit_code=(exit_code or None) if normalized.terminal else None,
        started_at=None if normalized is JobState.PENDING else parse_slurm_time(start),
    )


#: The runner's job log in a submission's directory, as sbatch's ``--output`` names it.
JOB_LOG = "slurm-%j.out"


def is_job_log(name: str) -> bool:
    """Whether the file *name* in a submission's directory is the runner's job log (``JOB_LOG``)."""
    return name.startswith("slurm-") and name.endswith(".out")


def _sbatch_args(request: SubmitRequest) -> list[str]:
    """What sbatch is given for *request* after its mode: the allocation's arguments
    (``placement_args``), with its output in ``JOB_LOG`` in the request's output directory,
    its tag as ``--comment``, its dependency (an ``afterok`` and Slurm's ``singleton``),
    ``--kill-on-invalid-dep=no`` beside an ``afterok``, and the batch script. Raise TypeError for
    a placement of another substrate."""
    if not isinstance(request.placement, SlurmPlacement):
        raise TypeError(f"Slurm cannot place an allocation by {type(request.placement).__name__}")
    output = request.output_dir / JOB_LOG
    args = [*placement_args(request.shape, request.site, request.placement, job_name=request.job_name, output=output),
            f"--comment={request.tag}"]
    dependency = [f"afterok:{check_job_id(request.after)}"] if request.after is not None else []
    if request.singleton:
        dependency.append("singleton")
    if dependency:
        args.append(f"--dependency={','.join(dependency)}")
    if request.after is not None:
        args.append("--kill-on-invalid-dep=no")
    return [*args, str(request.script)]


def submit_argv(request: SubmitRequest) -> tuple[str, ...]:
    """The sbatch command line that submits *request*, printing the job id alone
    (``--parsable``): what ``SlurmScheduler.submit`` runs."""
    return ("sbatch", "--parsable", *_sbatch_args(request))


class SlurmScheduler(Scheduler):
    """Slurm through its command-line tools."""

    def __init__(self, commands: SlurmCommands) -> None:
        self.commands = commands

    def submit_argv(self, request: SubmitRequest) -> tuple[str, ...]:
        return submit_argv(request)

    def submit(self, request: SubmitRequest) -> str:
        r = self.commands(*self.submit_argv(request))
        if r.ok:
            job_id = r.stdout.strip().split(";", 1)[0]
            if _JOB_ID.fullmatch(job_id):
                return job_id
            raise SubmitUnknown(f"sbatch printed {r.stdout.strip()!r}, not a job id")
        if r.unreachable:
            raise SubmitUnknown(_why(r))
        raise _refusal(r)

    def test_only(self, request: SubmitRequest) -> str:
        """sbatch's answer to *request* without submitting it: its start estimate, or Refused,
        which a host without Slurm's ``sbatch`` also gets, so a plan made there is voided with
        that reason before anything is submitted."""
        try:
            r = self.commands("sbatch", "--test-only", *_sbatch_args(request))
        except FileNotFoundError:
            raise Refused(NO_SBATCH) from None
        if r.ok:
            return _one_line(r.stderr) or _one_line(r.stdout)
        if r.unreachable:
            raise SchedulerUnavailable(_why(r))
        raise _refusal(r)

    def _queued(self) -> dict[str, tuple[JobInfo, str]] | None:
        """This user's queued and running jobs, each with its comment; None when squeue
        did not answer."""
        r = self.commands("squeue", "--me", "--all", "--noheader", f"--format={_SQUEUE_FORMAT}")
        if not r.ok:
            return None
        out: dict[str, tuple[JobInfo, str]] = {}
        for line in r.stdout.splitlines():
            parts = line.split("|", 4)
            if len(parts) != 5 or not _JOB_ID.fullmatch(parts[0].strip()):
                continue
            job_id, state, reason, start, comment = (p.strip() for p in parts)
            reason = None if reason in ("", "None", "(null)") else reason
            info = _job_info(job_id, state, reason=reason, start=start)
            out[job_id] = (info, comment)
        return out

    def find_by_tags(self, tags: Sequence[str]) -> dict[str, list[str] | None]:
        wanted = list(dict.fromkeys(tags))
        if not wanted:
            return {}
        queued = self._queued()
        if queued is None:
            return {tag: None for tag in wanted}
        return {tag: sorted((job_id for job_id, (_, comment) in queued.items() if comment == tag), key=int)
                for tag in wanted}

    def states(self, job_ids: Sequence[str]) -> dict[str, JobInfo]:
        ids = [check_job_id(j) for j in dict.fromkeys(job_ids)]
        out = {j: JobInfo(j, JobState.UNKNOWN) for j in ids}
        if not ids:
            return out
        queued = self._queued()
        rest = ids
        if queued is not None:
            for j in ids:
                if j in queued:
                    out[j] = queued[j][0]
            rest = [j for j in ids if j not in queued]
        if rest:
            r = self.commands("sacct", "--noheader", "--parsable2", "--allocations",
                              f"--jobs={','.join(rest)}", "--format=JobID,State,ExitCode,Start")
            if r.ok:
                for line in r.stdout.splitlines():
                    parts = line.split("|")
                    if len(parts) >= 4 and parts[0] in rest:
                        out[parts[0]] = _job_info(parts[0], parts[1], exit_code=parts[2], start=parts[3])
        return out

    def rewire(self, job_id: str, after: str) -> None:
        r = self.commands("scontrol", "update", f"JobId={check_job_id(job_id)}",
                          f"Dependency=afterok:{check_job_id(after)}")
        if r.ok:
            return
        if r.unreachable:
            raise SchedulerUnavailable(_why(r))
        raise _refusal(r)

    def cancel(self, job_id: str) -> None:
        r = self.commands("scancel", check_job_id(job_id))
        if r.ok:
            return
        if r.unreachable:
            raise SchedulerUnavailable(_why(r))
        if _GONE.search(r.stderr):
            return
        raise Refused(_why(r))


def ended_jobs(scheduler: Scheduler, job_ids: Sequence[str]) -> set[str]:
    """Which of *job_ids* the controller reports ended (``JobState.terminal``), each asked for by
    its exact id (``Scheduler.states``). A job neither squeue nor sacct reports on is UNKNOWN,
    which counts as alive."""
    return {job_id for job_id, info in scheduler.states(job_ids).items() if info.state.terminal}


def job_end_time(commands: SlurmCommands, job_id: str) -> float | None:
    """The job's EndTime from ``scontrol show job`` as POSIX seconds; None when it is
    unknown or the controller did not answer."""
    r = commands("scontrol", "show", "job", check_job_id(job_id))
    m = _END_TIME.search(r.stdout) if r.ok else None
    return None if m is None else parse_slurm_time(m.group(1))


def job_not_ready(commands: SlurmCommands, job_id: str) -> str | None:
    """Why job *job_id* may not start a step yet, from ``scontrol show job``: ``JobState=<state>``
    while it is not RUNNING (CONFIGURING while its nodes boot), ``Reason=Prolog`` while the
    controller still runs its Prolog, which it can for half a minute after the batch script
    started; None once it is RUNNING outside its Prolog. Raise SchedulerUnavailable when the
    controller did not answer or printed no line of JobState and Reason."""
    r = commands("scontrol", "show", "job", check_job_id(job_id))
    line = _STATE_LINE.search(r.stdout) if r.ok else None
    if line is None:
        raise SchedulerUnavailable(_why(r) if not r.ok else "scontrol printed no JobState")
    state, reason = line.groups()
    if state != "RUNNING":
        return f"JobState={state}"
    return "Reason=Prolog" if reason == "Prolog" else None


def own_job_id(environ: Mapping[str, str]) -> str:
    """The id of the job whose batch step *environ* is: ``SLURM_JOB_ID``, which Slurm sets in every
    job. Raise KeyError outside a job."""
    return environ["SLURM_JOB_ID"]


def node_name(hostname: str, environ: Mapping[str, str]) -> str:
    """The name Slurm knows this node by, which ``--exclude`` takes: ``SLURMD_NODENAME`` from
    *environ*, which slurmd sets in every job, else *hostname* up to its first dot."""
    return environ.get("SLURMD_NODENAME") or hostname.split(".", 1)[0]


def read_job_end(job_id: str, commands: SlurmCommands, environ: Mapping[str, str]) -> JobEnd:
    """Both sources of job *job_id*'s EndTime: ``scontrol show job`` run through *commands*,
    whose timeout bounds the wait, and ``SLURM_JOB_END_TIME`` in *environ*. A scontrol that
    cannot be run counts as a controller that did not answer."""
    raw = environ.get("SLURM_JOB_END_TIME")
    env = float(raw) if raw and raw.isdecimal() else None
    try:
        controller = job_end_time(commands, job_id)
    except OSError as e:
        log.warning("scontrol could not be run: %s", e)
        controller = None
    return JobEnd(controller=controller, env=env)
