"""The score role: score a drive against its verdict with the task's scorer, and record the
baseline credit the score was computed against.

    python -m hls.score.role --task-ref FILE --drive DIR --verdict FILE --out DIR --outputs-out FILE
        [--credit FILE] [--package FILE]

``--task-ref`` names a file holding a ``TaskRef`` as JSON. The role rebuilds the task with its
baseline credit injected from ``--credit``, the copy of the ``baseline.json`` the plan pinned
(``read_task(credit=True)``, which reads the file once and names the bytes it parsed), as
``hls run`` and ``hls grade`` assemble it. ``--drive`` is the drive stage's outputs
directory: its ``drive.json``, and ``trace/`` when the drive served.
- A served drive is scored from the trace's summary, which every scorer reads exactly as
  it reads the full trace: ``task.scorer.score(task.effective_tenants(), summary, verdict,
  task.hardware)``.
- A drive that did not serve scores a zero whose one reason is its outcome. An invalid
  artifact's zero carries why its submission did not build: the message, the failed step's
  exit status and the end of its output, read from the ``package.json`` that ``--package``
  names. A server that crashed or never became ready
  carries the drive's reason.

Into the ``--out`` directory, the attempt's outputs directory, it writes ``score.json``
(``Score.model_dump_json(indent=2)``), and ``score-provenance.json``
(``ScoreProvenance``): the credit B the scorer was given, the git blob id of the
``baseline.json`` it came from, and the id of the campaign that measured it. Whether
that file is the one the plan pinned is the stage's check, not the role's. It then reports both
files' digests and sizes, keyed by their paths under ``--out``, in ``--outputs-out``
(``report_outputs``), which is how the stage's manifest records them.

The exit status says whose fault a failure is and whether a retry can help:
- 0: both files were written and reported;
- 3 (``REFUSED_EXIT``): a deterministic refusal, with one stderr line starting with
  ``REFUSED_PREFIX``: a baseline credit that is missing, malformed, or measured for another
  workload, load, GPU kind or baseline server (``BaselineCreditError``), or an invalid
  artifact without its package record or with one it cannot read (``PackageOutcome``, the fields
  a grading reads, which every package record holds). A retry repeats it;
- 1: anything else, with the traceback on stderr, such as a summary that does not match its
  ``END.json``. Retried."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import traceback
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from hls.baseline_credit import BaselineCreditError
from hls.core.score import Score
from hls.core.verdict import Verdict
from hls.drive.record import DRIVE_FILE, TRACE_DIR, DriveRecord
from hls.package.result import PackageOutcome
from hls.spec.resources import BaselineHardware
from hls.store.digest import Sha256Hex
from hls.store.manifest import report_outputs
from hls.taskref import LoadedTask, MeasuredPaths, TaskRef, read_task
from hls.trace.store import read_summary
from hls.util import validation_errors

#: Exit status of a deterministic refusal.
REFUSED_EXIT = 3
#: Start of the stderr line that gives a refusal's reason.
REFUSED_PREFIX = "hls-score-refused: "
#: The files the role writes.
SCORE_FILE, PROVENANCE_FILE = "score.json", "score-provenance.json"


class Refused(Exception):
    """The role's inputs cannot be scored, and a retry would read the same ones."""


class CreditRecord(BaseModel):
    """The baseline credit a score was computed against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: B: the mean log credit of the campaign's drives, which the scorer was given.
    log_credit: float
    #: The git blob id of the ``baseline.json`` B was read from.
    blob_sha: str
    #: The campaign's id: SHA-256 of its identity (the task module, the workload
    #: fingerprint, the load, the GPU kind and the baseline server) as canonical JSON.
    campaign_id: Sha256Hex
    #: The hardware records of the allocations the credit's drives ran on, as the credit records
    #: them, blank fields included; None for a credit that records none.
    hardware: list[BaselineHardware] | None = None


class ScoreProvenance(BaseModel):
    """``score-provenance.json``: what the score was computed against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The baseline credit; None for a scorer that scores against no baseline.
    baseline: CreditRecord | None


def credit_record(loaded: LoadedTask) -> CreditRecord | None:
    """The baseline credit the task *loaded* with its credit gave its scorer, as the load read it;
    None when its scorer scores against no baseline."""
    if loaded.credit is None:
        return None
    credit = loaded.credit.value
    identity = json.dumps(credit.campaign.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return CreditRecord(log_credit=credit.log_credit_mean, blob_sha=loaded.credit.blob,
                        campaign_id=hashlib.sha256(identity.encode()).hexdigest(), hardware=credit.hardware)


def invalid_artifact_score(result: PackageOutcome) -> Score:
    """The zero of a submission that did not build: its package record's message, and the
    failed step's exit status and the end of its output when it records them."""
    diagnostics: dict = {"message": result.reason}
    if result.returncode is not None:
        diagnostics["returncode"] = result.returncode
    if result.log_tail:
        diagnostics["log"] = result.log_tail
    return Score.zero(reasons=("invalid_artifact",), diagnostics=diagnostics)


def read_verdict(path: Path) -> Verdict:
    """The verdict ``verdict.json`` holds, as the scorers read it."""
    doc = json.loads(path.read_text())
    return Verdict(passed=doc["aggregate_passed"], per_event=doc["per_event"], failure_mode=doc["failure_mode"],
                   diagnostics=doc["diagnostics"])


def score(task, drive_dir: Path, verdict: Path, package: Path | None) -> Score:
    """The score of the drive in *drive_dir* against the verdict in *verdict*."""
    record = DriveRecord.model_validate_json((drive_dir / DRIVE_FILE).read_bytes())
    if record.outcome == "invalid_artifact":
        if package is None:
            raise Refused("an invalid artifact is scored from its package record, and no --package was given")
        try:
            result = PackageOutcome.model_validate_json(package.read_bytes())
        except ValidationError as e:
            raise Refused(f"{package} is not a package record: {validation_errors(e)}") from None
        return invalid_artifact_score(result)
    if record.outcome != "served":
        return Score.zero(reasons=(record.outcome,), diagnostics={"reason": record.reason})
    summary = read_summary(drive_dir / TRACE_DIR)
    return asyncio.run(task.scorer.score(task.effective_tenants(), summary, read_verdict(verdict), task.hardware))


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.score.role")
    p.add_argument("--task-ref", type=Path, required=True)
    p.add_argument("--drive", type=Path, required=True)
    p.add_argument("--verdict", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    p.add_argument("--credit", type=Path)
    p.add_argument("--package", type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        ref = TaskRef.model_validate_json(args.task_ref.read_bytes())
        loaded = read_task(ref, credit=True, files=MeasuredPaths(credit=args.credit))
        result = score(loaded.task, args.drive, args.verdict, args.package)
        provenance = ScoreProvenance(baseline=credit_record(loaded))
    except (BaselineCreditError, Refused) as e:
        reason = " ".join(str(e).split())
        print(f"{REFUSED_PREFIX}{type(e).__name__}: {reason}", file=sys.stderr)
        return REFUSED_EXIT
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("score role failed; no score was written", file=sys.stderr)
        return 1
    files = (args.out / SCORE_FILE, args.out / PROVENANCE_FILE)
    try:
        files[0].write_bytes(result.model_dump_json(indent=2).encode())
        files[1].write_bytes(provenance.model_dump_json(indent=2).encode())
        report_outputs(args.outputs_out, args.out, *files)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("score role failed writing its files", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
