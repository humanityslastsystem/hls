"""The driver role: drive a served system with a task's own driver, write its trace on the node's
disk as the drive goes, and publish the trace to the store.

    python -m hls.drive.role --task-ref FILE --artifact-url URL --local DIR --out DIR
        --outputs-out REPORT --stream-out RECORD --request-timeout-sec G --trace-cap-bytes N [--split test]
        [--drive-timeout-sec S] [--start-by T] [--nofile-soft N] [--reply-max-bytes N]
        [--memory-limit-bytes N] [--measured-baseline]

``FILE`` holds a ``TaskRef`` as JSON. The role rebuilds the task with ``load_task``, builds
its driver whole with ``driver_for(task.agent, ...)`` and the task's GPU kind, and drives it
in this process with ``Driver.run``, passing the task's contract and seed and the task's
tenants with their verifiers stripped. The drive is ``Driver``'s own loop, its
garbage-collector pause, stall watchdog, drive wall and connection pool
included. Before it drives, the role raises its soft ``RLIMIT_NOFILE`` to ``N``
(``NOFILE_SOFT`` unless the stage names another); the limit it drives under, which a sandbox
that refuses ``setrlimit`` leaves at the inherited one, is in the trace's driver description
(``nofile_soft``) and the role's log. Its root logger, and so httpx, stays at WARNING.

Every request gets ``G`` seconds past the drive's last scheduled arrival (the deadline's grace,
``Driver.request_timeout_sec``), which the stage always names. The drive's wall is ``S`` from the
drive's own start, however late the scheduler started the role's step: a drive is never shortened
by the harness's delays. ``--start-by``, a POSIX time,
is the latest the drive may start for that wall and the stage's margin after it to fit before
the allocation ends. A drive that would start later does not start: the role writes no trace,
says why on a last ``refused:`` line on stderr and exits ``LATE_START_EXIT``, which the stage
records as infra (``hls.stages.deploy_drive``).

Each event the driver records goes, as it is recorded, to the drive's trace writer
(``hls.drive.stream``), which writes it into the trace's room in ``--local`` ``DIR``, on the
node's disk, which the stage made before the server started (``hls.trace.store.make_room``):
the role holds no reply past its own request. What the trace may hold is the task's cap,
``--trace-cap-bytes`` (``hls.drive.bounds``, which the stage takes from the plan): the role counts
every reply from ``URL`` as the driver's client reads it, and every request to it as the client
sends it, and fails, as the contestant's request, every reply once the trace written and the
replies and requests not yet written reach the cap, and any reply
past ``--reply-max-bytes`` (``REPLY_MAX_BYTES`` unless the stage names another)
(``hls.drive.replies``). The drive asks ``URL`` for uncompressed replies, and a compressed one
fails. With ``--memory-limit-bytes``, the driver's declared memory, the role also samples its own
memory about once a second, and once that passes 85% of it every further reply fails
(``hls.drive.memory``). The driver records each failure as the contestant's failed request,
and the drive goes on as the driver decides. ``--measured-baseline`` says the artifact is the
task's own measured baseline, the reference's server rather than a contestant's: none of its
replies is counted or refused, and there is no memory guard. A normal drive's trace is the one an
uncounted drive writes, byte for byte. The role's last log line says what it counted, the most
it held, the peak memory it sampled, whether the guard fired, and how far its trace writer fell
behind the drive.

Once the drive has returned and its last event is written, the writer finishes the trace, and the
role copies it from the room into the new directory ``--out`` ``DIR`` (the attempt's
``outputs/trace``, on the store) with ``hls.trace.store.publish_trace``, checking every file
against what the writer recorded and writing ``END.json`` last. A trace has one writer, so an
``--out`` ``DIR`` already there is refused before the drive starts, and a room that was not made
fails the writer. The role then reports every file of the trace,
keyed by its path under ``--out`` ``DIR``'s parent, the outputs directory
(``trace/chunk-000000.jsonl.gz``, ``trace/summary.jsonl.gz``, ``trace/END.json``), in ``REPORT``
for the stage's manifest. The digests are the writer's own, counted from the bytes as it wrote
them, and checked as they were copied. Last, it writes in ``RECORD`` how its trace writer kept up
with the drive (``hls.drive.stream.TraceStreamRecord``), which the stage puts in the drive's
record. Both files are written in place, the stage having set them aside before the server
started.

The role exits 0 once ``END.json`` is published, the trace reported and the writer's record
written. A drive that raises, a trace writer that fails (``TraceStreamFailed``), a trace that
cannot be published, or a report that cannot be written, exits 1 with the traceback on stderr,
and, once its trace writer had started, its record in ``RECORD`` with why the writer failed, if
it did, for the stage's evidence. That is an infra failure of the
role, never a shorter trace: ``END.json`` is written last, so a reader never takes a partly
written trace for a drive. A contestant's server that dies during the drive is not one: its
requests fail, the trace records them, and the role exits 0.

The trace's ``ending`` says how the drive stopped, read from what the driver stamped on the
turns it did not complete: the drive wall, the circuit breaker (or the stall watchdog, which
stamps the breaker's string), or neither (drained). A drive that ended on its task's scheduled
stop is drained: no turn records a fault, and each request it cut carries
``hls.spec.workload.CUT_EXTRA``. The trace's driver description names the stop and the stall
budget where the task declares them. Each session a trajectory driver dropped on a dead sandbox
is in ``END.json`` with its reason (``hls.trace.store.TraceEnd.dropped``), which the drive's
record carries with the rest of ``END.json``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import resource
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from hls.artifact.artifact_client import trace_growth
from hls.core.trace import TraceEvent
from hls.drive.bounds import REPLY_MAX_BYTES
from hls.drive.memory import MemoryGuard, memory_source
from hls.drive.replies import ReplyGate, counting_replies
from hls.drive.stream import TraceStream
from hls.obs.logsetup import configure_logging
from hls.spec.drivers import (ABANDONED_ERROR, ABORTED_ERROR, WALL_ERROR, Driver, TrajectorySessionDriver,
                              driver_for)
from hls.store.manifest import report_written_outputs
from hls.taskref import TaskRef, load_task
from hls.trace.store import Ending, publish_trace
from hls.util import write_in_place

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec
    from hls.spec.tenant import Tenant

# Named, not ``__name__``: run as a program the module is ``__main__``, which no hls logging
# handler covers.
log = logging.getLogger("hls.drive.role")

#: The soft open-file limit the role raises itself to before it drives, unless the stage
#: names another (``_raise_nofile``: clamped to the hard limit, never lowered). Container
#: stacks default the soft limit to 1024, which caps the open-loop driver at about 1,000
#: concurrent sockets: above the SLO knee it sheds load client-side as EMFILE connect faults
#: instead of loading the server. 65536 is at least eight times the requests in flight at the SLO
#: knee, far under systemd's 524288 hard limit.
NOFILE_SOFT = 65536
#: The exit status of a role whose drive would have started after ``--start-by``, and so did not.
LATE_START_EXIT = 4


def _raise_nofile(target: int) -> int:
    """Raise the process's soft RLIMIT_NOFILE to ``target``, clamped to the hard
    limit and never lowered (``NOFILE_SOFT`` says why a drive needs it), and return the
    soft limit the process is left with. Raising the soft limit toward the hard limit is
    the contract systemd documents for processes that need more file descriptors.

    Best-effort: an exotic sandbox may refuse ``setrlimit`` -- say so and continue,
    the drive still works up to the inherited limit, which is then the one returned."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return soft
    new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if new_soft <= soft:
        return soft
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError) as e:
        print(f"[drive role] could not raise RLIMIT_NOFILE {soft} -> {new_soft}: {e}", file=sys.stderr)
    return resource.getrlimit(resource.RLIMIT_NOFILE)[0]


def _limit_text(limit: int) -> str:
    """A soft limit as the trace's driver description and the log say it."""
    return "unlimited" if limit == resource.RLIM_INFINITY else str(limit)


def too_late(start_by: float | None, *, now: float) -> str | None:
    """Why a drive starting at *now* does not start: it is past *start_by*, a POSIX time; None
    when it may start, and always without a *start_by*."""
    if start_by is None or now <= start_by:
        return None
    return (f"the drive would start {now - start_by:.1f}s after {start_by:.3f}, the latest start at which its full "
            "wall and the stage's margin fit before the allocation ends, so it was not started")


def stripped_tenants(task: TaskSpec) -> list[Tenant]:
    """The task's tenants without their verifiers: the driver only generates load, and
    verification-only dependencies need not import here."""
    return [t.model_copy(update={"verifiers": []}) for t in task.effective_tenants()]


class EndingWatch:
    """How the drive stopped, read from the error each recorded event carries: ``wall`` when a
    turn carries the drive wall's error, ``aborted`` when one carries the circuit breaker's (sent
    or unsent), else ``drained``. The wall wins, since it bounds the drive."""

    def __init__(self) -> None:
        self._wall = False
        self._aborted = False

    def note(self, ev: TraceEvent) -> None:
        """Read the error *ev* carries."""
        self._wall = self._wall or ev.error == WALL_ERROR
        self._aborted = self._aborted or ev.error in (ABORTED_ERROR, ABANDONED_ERROR)

    @property
    def ending(self) -> Ending:
        if self._wall:
            return "wall"
        return "aborted" if self._aborted else "drained"


def build_driver(task: TaskSpec, *, request_timeout_sec: float, drive_timeout_sec: float | None) -> Driver:
    """The task's driver, built whole (``driver_for``) with the drive's grace and wall."""
    return driver_for(task.agent, max_context_len=task.max_context_len,
                      chat_template_kwargs=task.model.chat_template_kwargs,
                      request_timeout_sec=request_timeout_sec, drive_timeout_sec=drive_timeout_sec,
                      gpu_kind=task.hardware.accelerator.kind)


def driver_description(driver: Driver, *, split: str, nofile: int) -> dict[str, str]:
    """What the trace says of *driver*: its class, the split, its grace and wall and the soft
    open-file limit it drives under, and the scheduled stop and stall budget of a trajectory
    driver whose task declares them."""
    description = {"class": type(driver).__name__, "split": split,
                   "request_timeout_sec": str(driver.request_timeout_sec),
                   "drive_timeout_sec": str(driver.drive_timeout_sec), "nofile_soft": _limit_text(nofile)}
    stop = driver.scheduled_stop if isinstance(driver, TrajectorySessionDriver) else None
    if stop is not None:
        description.update(wall_clock_sec=str(stop.wall_clock_sec), cut_grace_sec=str(stop.cut_grace_sec))
    if driver.stall_timeout_sec is not None:
        description["stall_timeout_sec"] = str(driver.stall_timeout_sec)
    return description


async def drive(task: TaskSpec, driver: Driver, *, artifact_url: str, split: str,
                on_event: Callable[[TraceEvent], None], gate: ReplyGate | None,
                guard: MemoryGuard | None = None) -> None:
    """Drive *artifact_url* with *driver* in this process, with the task's contract, seed and
    stripped tenants, handing *on_event* each event as it is recorded. Every reply from
    *artifact_url* is counted against *gate*; None counts nothing. *guard*, when given, samples
    the driver's memory while the drive runs."""
    with counting_replies(artifact_url, gate), guard or contextlib.nullcontext():
        await driver.run(artifact_url, split=split, tenants=stripped_tenants(task), contract=task.contract,
                         seed=task.seed, on_event=on_event)


def _positive(text: str) -> int:
    """A byte count the stage passes: a whole number above zero."""
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive number of bytes")
    return value


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.drive.role")
    p.add_argument("--task-ref", type=Path, required=True)
    p.add_argument("--artifact-url", required=True)
    p.add_argument("--local", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    p.add_argument("--stream-out", type=Path, required=True)
    p.add_argument("--split", default="test", choices=("dev", "test"))
    p.add_argument("--request-timeout-sec", type=float, required=True)
    p.add_argument("--drive-timeout-sec", type=float)
    p.add_argument("--start-by", type=float)
    p.add_argument("--nofile-soft", type=int, default=NOFILE_SOFT)
    p.add_argument("--reply-max-bytes", type=_positive, default=REPLY_MAX_BYTES)
    p.add_argument("--trace-cap-bytes", type=_positive, required=True)
    p.add_argument("--memory-limit-bytes", type=_positive)
    p.add_argument("--measured-baseline", action="store_true")
    return p.parse_args(argv)


def _gate(args: argparse.Namespace, task: TaskSpec, stream: TraceStream) -> ReplyGate | None:
    """What the drive's replies are counted against: None for the task's measured baseline,
    else the stage's figures, at the trace growth of the task's contract."""
    if args.measured_baseline:
        return None
    return ReplyGate(reply_max_bytes=args.reply_max_bytes, trace_cap_bytes=args.trace_cap_bytes,
                     growth=trace_growth(task.contract), written=stream.written)


def _guard(args: argparse.Namespace, gate: ReplyGate | None) -> MemoryGuard | None:
    """The memory guard over *gate*: None when nothing is counted or no limit was named."""
    if gate is None or args.memory_limit_bytes is None:
        return None
    return MemoryGuard(gate, args.memory_limit_bytes, memory_source())


def _sink(stream: TraceStream, gate: ReplyGate | None, watch: EndingWatch) -> Callable[[TraceEvent], None]:
    """What the driver hands each recorded event to, on its event loop: the ending watch, then
    the trace's writer, with the reply bytes and the requests the recording task held."""
    def on_event(ev: TraceEvent) -> None:
        watch.note(ev)
        held = gate.detach() if gate is not None else None
        number = stream.put(ev)
        if gate is not None and held is not None:
            gate.queued(number, held)
    return on_event


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    # Every request in flight holds a socket, so a step's inherited soft limit (often 1,024)
    # would fail the requests past it as connect faults charged to the contestant.
    nofile = _raise_nofile(args.nofile_soft)
    # The root logger stays at WARNING, so httpx writes no line per request on the drive's
    # event loop.
    configure_logging()
    if nofile != resource.RLIM_INFINITY and nofile < args.nofile_soft:
        log.warning("the drive runs under a soft open-file limit %d, below the %d asked: requests in flight "
                    "past it fail as connect faults", nofile, args.nofile_soft)
    else:
        log.info("the drive runs under a soft open-file limit %s", _limit_text(nofile))
    stream: TraceStream | None = None
    try:
        if os.path.lexists(args.out):
            raise FileExistsError(f"{args.out} exists: a trace has one writer, and the drive was not started")
        ref = TaskRef.model_validate_json(args.task_ref.read_bytes())
        task = load_task(ref)
        late = too_late(args.start_by, now=time.time())
        if late is not None:
            print(f"refused: {late}", file=sys.stderr)
            return LATE_START_EXIT
        driver = build_driver(task, request_timeout_sec=args.request_timeout_sec,
                              drive_timeout_sec=args.drive_timeout_sec)
        description = driver_description(driver, split=args.split, nofile=nofile)
        watch = EndingWatch()
        with TraceStream(args.local, driver=description) as stream:
            gate = _gate(args, task, stream)
            guard = _guard(args, gate)
            asyncio.run(drive(task, driver, artifact_url=args.artifact_url, split=args.split,
                              on_event=_sink(stream, gate, watch), gate=gate, guard=guard))
            written = stream.finish(watch.ending, driver.dropped_sessions)
        publish_trace(args.local, args.out, written)
        report_written_outputs(args.outputs_out, args.out.parent,
                               {args.out / name: record for name, record in written.files.items()})
        kept_up = stream.record()
        write_in_place(args.stream_out, kept_up.model_dump_json().encode())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        if stream is not None:
            # For the stage's evidence: why the writer failed, if it did, and how far behind it fell.
            with contextlib.suppress(OSError):
                write_in_place(args.stream_out, stream.record().model_dump_json().encode())
        print("drive role failed; no complete trace was written and reported", file=sys.stderr)
        return 1
    end = written.end
    counted = ("replies not counted" if gate is None
               else f"{gate.total:,} reply bytes, at most {gate.held_peak:,} held at once, and at most "
                    f"{gate.held_requests_peak:,} bytes of requests, against a {gate.trace_cap_bytes:,}-byte trace cap")
    memory = "memory guard off" if guard is None else guard.summary()
    log.info("drive ended %s: %d events in %d chunks; %s; %s; %s", end.ending, end.events, len(end.chunks), counted,
             memory, kept_up.summary())
    if end.dropped:
        log.warning("the drive dropped %d session(s) on a dead sandbox, named in END.json: %s", len(end.dropped),
                    ", ".join(d.session_id for d in end.dropped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
