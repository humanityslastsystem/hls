"""The verify role: run a task's verifiers over a drive's trace on the store and write the verdict.

    python -m hls.verify.role --task-ref FILE --trace DIR --out DIR --outputs-out REPORT
        [--eval-id terminal] [--calibration FILE]
        [--answers DIR --writer NAME --gpu-kind KIND --reference-image SHA256]

``FILE`` holds a ``TaskRef`` as JSON. The role rebuilds the task with its calibration
injected from ``--calibration``, the copy of the file the plan pinned, as ``hls run`` and
``hls grade`` assemble it, and reads the summary of the trace in
``DIR`` (``hls.trace.store.read_summary``), checked against ``END.json``. When any request
failed, the run fails whatever the verifiers say, and the role writes the failed verdict from
the summary alone (``hls.verify.run.ungraded_verdict``), reading none of the trace's chunks:
a drive that reached its trace cap failed every reply after it. Otherwise it runs the
verification core, ``hls.verify.run.verify_trace``, over the stored trace
(``hls.trace.store.StoredTrace``), which the role never holds whole: each verifier takes the
events from a view that reads them from the store on each pass, checking each chunk against
``END.json`` before it yields an event of it. The reference engines are loaded in this process
on the role's own GPUs, on an engine host the role creates (``hls.verify.inprocess.EngineHost``).
Either way it writes into the out directory, which must be empty: ``verdict.json`` and one
``replies.<index>.<Verifier>.jsonl.gz`` per graded verdict with per-event detail, and
``verify.json`` (``VerifyRecord``), which names the reference engines
loaded and the stack each reported, the reference answers it read and added, the calibration
file the verifiers were given (``calibration_blob``), and the versions of Python and of the
libraries the verification ran with (``library_versions``): the role runs the verifiers in the
reference image's Python, so a comparison with another environment can see where the two
differ. Last it reports every file it wrote, keyed by its name in the out directory, in
``REPORT`` for the stage's manifest (``hls.store.manifest.report_outputs``).

With ``--answers``, the corpus answers of every reference that caches them are kept in that
directory, the task's area of the store (``hls.verify.answers``), for the GPU kind and the
reference image named by the other three options, which come together or not at all (exit 2
otherwise, argparse's). A reference whose every answer is there is never loaded. Without it
every reference answers afresh.

The exit status says whether a retry can help; every failure here is the harness's:
- 0: a verdict was written, passing or not, and reported;
- 3 (``REFUSED_EXIT``): a deterministic refusal, a ``HarnessFault`` (an empty trace, a label
  metric over unlabeled rows), a ``WiringError`` or a ``CalibrationError`` (no calibration, a
  stale one, one from other silicon, a reference stack that drifted from it). A retry repeats
  it, so the stage records it as infra that is not retried. The reason is one stderr line
  starting with ``REFUSED_PREFIX``;
- 1: anything else, with the traceback on stderr: a summary, or a chunk of a trace whose
  every request was answered, that does not match its ``END.json``, a report that cannot be
  written, and an engine that failed to load or to stop,
  whatever it raised (``EngineFailure``); and an engine the engine host gave up on, ending the
  role at once (``GIVE_UP``): one whose process ended or that reports itself dead, or one whose
  release did not return within the host's bound, after every thread's stack, each with one
  last stderr line starting with ``ENGINE_FAULT_PREFIX``. Loading and stopping engines on the
  role's own GPUs is work a retry can do again. Infra, retried."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import logging
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from hls.core.capabilities import StackReport
from hls.core.engine import Engine
from hls.core.faults import HarnessFault
from hls.core.trace import Trace
from hls.core.verdict import Verdict
from hls.obs.logsetup import configure_logging
from hls.spec.taskspec import TaskSpec
from hls.store.manifest import report_outputs
from hls.store.posix import PosixStore
from hls.taskref import LoadedTask, MeasuredPaths, TaskRef, read_task
from hls.trace.store import StoredTrace
from hls.verify.answers import StoreAnswers
from hls.verify.base import WiringError
from hls.verify.calibration import CalibrationError
from hls.verify.inprocess import EngineHost, GiveUp, Release
from hls.verify.run import ungraded_verdict, verify_trace, write_verdict

# Named, not ``__name__``: run as a program the module is ``__main__``, which no hls logging
# handler covers.
log = logging.getLogger("hls.verify.role")

#: Exit status of a deterministic refusal.
REFUSED_EXIT = 3
#: Start of the stderr line that gives a refusal's reason.
REFUSED_PREFIX = "hls-verify-refused: "
#: Start of the last stderr line of a role whose engine host gave up on an engine.
ENGINE_FAULT_PREFIX = "hls-verify-engine-fault: "
#: How the engine host of the verify role and of the oracle fill gives up: exit 1, retried as infra.
GIVE_UP = GiveUp(exit_code=1, prefix=ENGINE_FAULT_PREFIX)
#: The file the role writes beside the verdict.
VERIFY_FILE = "verify.json"
#: The exceptions that are deterministic refusals.
REFUSALS = (HarnessFault, WiringError, CalibrationError)
#: Distributions whose versions ``verify.json`` records whether or not the verification
#: imported them: the libraries a verifier's numbers can depend on.
LIBRARIES = ("numpy", "transformers", "soxr", "torch")


class EngineFailure(RuntimeError):
    """A reference engine failed to load or to stop. Its cause is kept as ``__cause__``; its
    type is not one of ``REFUSALS``, so the role exits 1 and the stage retries."""


class EngineRecord(BaseModel):
    """One reference engine the role loaded."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Its name in load order, over every load of the verify: "reference", then "reference1", ...
    role: str
    #: The engine's class.
    engine: str
    #: What its ``stack_report`` returned once loaded; None for an engine without one, or
    #: one whose report failed.
    stack: dict[str, Any] | None
    #: Why the report failed; None when it did not.
    stack_error: str | None


class CalibrationBlob(BaseModel):
    """The calibration file a task's verifiers were given. Where the plan found it is the plan's
    pin of it (``Plan.measured``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The id git commits its bytes under (``hls.taskref.blob_id``).
    blob_sha: str
    #: The artifact it holds (``calibration_kind``): "flips" or "report".
    kind: str
    #: The workload fingerprint it was measured for.
    workload_fingerprint: str


def calibration_blob(loaded: LoadedTask) -> CalibrationBlob | None:
    """The calibration file the task *loaded* gave its verifiers, as the load read it; None for a
    task none of whose verifiers needs one."""
    if loaded.calibration is None:
        return None
    return CalibrationBlob(blob_sha=loaded.calibration.blob, kind=loaded.calibration.value.kind,
                           workload_fingerprint=loaded.calibration.value.workload_fingerprint)


class VerifyRecord(BaseModel):
    """``verify.json``: what the verification ran on."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The eval the verdict belongs to.
    eval_id: str
    #: The reference engines loaded, in load order.
    engines: list[EngineRecord]
    #: Each reference answers file read, by its key in the task's answers area, to its sha256
    #: (``StoreAnswers.read``); empty without an area.
    answers_read: dict[str, str] = Field(default_factory=dict)
    #: Each reference answers file this verify added, by its key in the task's answers area, to
    #: its sha256 (``StoreAnswers.written``); empty without an area.
    answers_written: dict[str, str] = Field(default_factory=dict)
    #: The version of Python (``python``) and of each library the verification ran with, by
    #: distribution name (``library_versions``); empty when no verification ran.
    libraries: dict[str, str | None]
    #: The calibration file the verifiers were given (``calibration_blob``); None for a task none
    #: of whose verifiers needs one, and when no verification ran.
    calibration: CalibrationBlob | None = None


class _RecordingHost(EngineHost):
    """The engine host, recording each engine it loads, the name it is given in load order
    (``reference``, ``reference1``, ...) and the stack it reports, and raising every failure to
    load or stop an engine as an ``EngineFailure``."""

    def __init__(self) -> None:
        super().__init__(give_up=GIVE_UP)
        #: The engines loaded, in load order, as (name, class name, stack, stack error).
        self.loaded: list[tuple[str, str, dict[str, Any] | None, str | None]] = []

    async def acquire(self, engine: Engine, *, name: str | None = None) -> tuple[Engine, Release]:
        cls = type(engine).__name__
        try:
            proxy, release = await super().acquire(engine, name=name)
        except Exception as e:
            raise EngineFailure(f"{cls} failed to load: {type(e).__name__}: {e}") from e
        stack, error = None, None
        if isinstance(engine, StackReport):
            try:
                stack = await engine.stack_report()
            except Exception as e:  # noqa: BLE001 -- a report is a record; a gate that needs one asks
                error = f"{type(e).__name__}: {e}"
        n = len(self.loaded)
        self.loaded.append(("reference" if n == 0 else f"reference{n}", cls, stack, error))

        async def stop() -> None:
            try:
                await release()
            except Exception as e:
                raise EngineFailure(f"{cls} failed to stop: {type(e).__name__}: {e}") from e

        return proxy, stop

    async def teardown(self) -> None:
        try:
            await super().teardown()
        except Exception as e:
            raise EngineFailure(f"a reference engine failed to stop: {type(e).__name__}: {e}") from e


def library_versions() -> dict[str, str | None]:
    """The version of Python, under ``python``, then of each of ``LIBRARIES`` (None for one that
    is not installed) and of every installed distribution with a module loaded in this
    process, so every library the task, its verifiers and its engines imported, by
    distribution name. Versions are read from the installed metadata, importing nothing."""
    owners = importlib.metadata.packages_distributions()
    loaded = {d for name in {m.partition(".")[0] for m in list(sys.modules)} for d in owners.get(name, ())}
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for distribution in sorted(set(LIBRARIES) | loaded):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _require_empty(out: Path) -> None:
    """Raise RuntimeError unless the out directory *out* is empty."""
    held = sorted(p.name for p in out.iterdir())
    if held:
        raise RuntimeError(f"the out directory {out} already holds {held}; the verify role writes into an "
                           "empty one")


def _write(out: Path, verdict: Verdict, per_verifier_results: list, *, eval_id: str, engines: list[EngineRecord],
           answers: StoreAnswers | None, calibration: CalibrationBlob | None) -> VerifyRecord:
    """Write ``verdict.json`` and ``verify.json``, naming the reference *engines* loaded, into
    *out*, and return the record."""
    write_verdict(out, verdict, per_verifier_results)
    record = VerifyRecord(
        eval_id=eval_id, engines=engines,
        answers_read=answers.read if answers is not None else {},
        answers_written=answers.written if answers is not None else {},
        libraries=library_versions(), calibration=calibration)
    (out / VERIFY_FILE).write_text(record.model_dump_json(indent=2))
    return record


async def verify(task: TaskSpec, trace: Trace | StoredTrace, *, out: Path, eval_id: str,
                 answers: StoreAnswers | None, calibration: CalibrationBlob | None = None) -> VerifyRecord:
    """Verify the whole *trace*, held in memory or stored, against *task* with the engines in
    this process, serving and
    keeping the corpus answers of caching references in *answers* (None: every reference
    answers afresh), and write ``verdict.json``, the replies files and ``verify.json`` into
    *out*, which must be empty; ``verify.json`` names the *calibration* file *task* was given."""
    _require_empty(out)
    host = _RecordingHost()
    async with host:
        verdict, per_verifier_results = await verify_trace(task, trace, eval_dir=out, host=host, answers=answers)
    engines = [EngineRecord(role=role, engine=engine, stack=stack, stack_error=error)
               for role, engine, stack, error in host.loaded]
    record = _write(out, verdict, per_verifier_results, eval_id=eval_id, engines=engines, answers=answers,
                    calibration=calibration)
    log.info("verified %d events with %d reference engines", len(trace.events), len(record.engines))
    return record


async def verify_stored(task: TaskSpec, trace_dir: Path, *, out: Path, eval_id: str,
                        answers: StoreAnswers | None, calibration: CalibrationBlob | None = None) -> VerifyRecord:
    """``verify`` for the trace stored in *trace_dir*, deciding from its summary whether it is
    graded. When any request failed, the verdict is ``ungraded_verdict``'s of the summary's
    events, which is the whole trace's, since a summary event holds every field it reads; no
    chunk is read and no engine loads, and the files written are the ones ``verify`` writes
    for the whole trace. Otherwise the stored trace is verified (``StoredTrace``), never held."""
    _require_empty(out)
    stored = StoredTrace(trace_dir)
    ungraded = ungraded_verdict(task, stored.summary())
    if ungraded is None:
        return await verify(task, stored, out=out, eval_id=eval_id, answers=answers, calibration=calibration)
    verdict, per_verifier_results = ungraded
    record = _write(out, verdict, per_verifier_results, eval_id=eval_id, engines=[], answers=answers,
                    calibration=calibration)
    log.info("%s; no chunk of the trace was read", verdict.diagnostics["reason"])
    return record


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.verify.role")
    p.add_argument("--task-ref", type=Path, required=True)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    p.add_argument("--eval-id", default="terminal")
    p.add_argument("--calibration", type=Path)
    p.add_argument("--answers", type=Path)
    p.add_argument("--writer")
    p.add_argument("--gpu-kind")
    p.add_argument("--reference-image")
    args = p.parse_args(argv)
    given = [v is not None for v in (args.answers, args.writer, args.gpu_kind, args.reference_image)]
    if any(given) and not all(given):
        p.error("--answers, --writer, --gpu-kind and --reference-image come together or not at all")
    return args


def _answers(args: argparse.Namespace) -> StoreAnswers | None:
    """The answers area the options name, or None without one."""
    if args.answers is None:
        return None
    return StoreAnswers(PosixStore(args.answers), gpu_kind=args.gpu_kind, image=args.reference_image,
                        writer=args.writer)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    configure_logging()
    try:
        ref = TaskRef.model_validate_json(args.task_ref.read_bytes())
        loaded = read_task(ref, calibration=True, files=MeasuredPaths(calibration=args.calibration))
        task, calibration = loaded.task, calibration_blob(loaded)
        asyncio.run(verify_stored(task, args.trace, out=args.out, eval_id=args.eval_id, answers=_answers(args),
                                  calibration=calibration))
        # The out directory was empty, so every file in it now is one this role wrote.
        report_outputs(args.outputs_out, args.out, *sorted(args.out.iterdir()))
    except REFUSALS as e:
        reason = " ".join(str(e).split())
        print(f"{REFUSED_PREFIX}{type(e).__name__}: {reason}", file=sys.stderr)
        return REFUSED_EXIT
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("verify role failed; no verdict was reported", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
