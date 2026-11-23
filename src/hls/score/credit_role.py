"""The credit role: one drive of a task's measured baseline, scored with no baseline credit and
the assume-correct verdict.

    python -m hls.score.credit_role --task-ref FILE --drive DIR --out DIR --outputs-out FILE

``--task-ref`` names a file holding a ``TaskRef`` as JSON. The role rebuilds the task without a
baseline credit (``load_task``), so its scorer reports the drive's own mean log credit L and
nothing derived from a B: the measurement a baseline credit is the mean of. ``--drive`` is the
drive stage's outputs directory, its ``drive.json`` and ``trace/``. The trace is scored from its
summary, which every scorer reads exactly as the full trace, against ``passthrough_verdict``:
the baseline is the task's own reference server, so its replies are trusted for content, and a
request that errored or was never answered earns nothing.

The drive's times come from its record, which holds no time the server started. The drive is
dated from the driver's start less the seconds the server took to open both readiness gates,
counted from its command's start (``ArtifactRecord.ready_after_command_sec``), and lasts those
seconds and the drive's. That date is later than the server's start by the wait
between its readiness and the driver's start (the stage's check that the drive's wall fits and
the driver step's own start), which the duration leaves out. Nothing in B reads either field;
the candidate lists its drives by date.

It writes ``credit-drive.json`` (``CreditDrive``) into ``--out``, the attempt's outputs
directory, and reports its digest and size in ``--outputs-out`` (``report_outputs``).

Exit status:
- 0: the file was written and reported;
- 3 (``REFUSED_EXIT``): the drive is no clean baseline drive, with one stderr line starting with
  ``REFUSED_PREFIX``: the task's scorer scores against no baseline, or the task declares no
  baseline server; the drive did not serve, sent no request or scored a zero; the scorer
  measured no credit or no unit count. A retry reads the same drive;
- 1: anything else, with the traceback on stderr. Retried."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from hls.core.verdict import passthrough_verdict
from hls.drive.record import DRIVE_FILE, TRACE_DIR, DriveRecord
from hls.store.digest import Sha256Hex
from hls.store.manifest import report_outputs
from hls.taskref import TaskRef, load_task
from hls.trace.store import read_summary

#: Exit status of a deterministic refusal.
REFUSED_EXIT = 3
#: Start of the stderr line that gives a refusal's reason.
REFUSED_PREFIX = "hls-credit-refused: "
#: The file the role writes.
CREDIT_DRIVE_FILE = "credit-drive.json"


class Refused(Exception):
    """The drive is no clean drive of the task's baseline, and a retry would read the same one."""


class CreditDrive(BaseModel):
    """``credit-drive.json``: one drive of the baseline, as a baseline credit pools it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: When the drive began, ISO-8601 UTC: the driver's start less the seconds the server took to
    #: open its readiness gates (the module docstring says how that differs from the server's start).
    started_at: str
    #: Seconds of the server's readiness and of the drive, without the wait between them.
    duration_sec: float
    #: Requests the drive sent, answered or not. Informational: a session's turn count follows
    #: the server's own replies.
    n_requests: int
    #: Units the scorer credited, sessions or requests by the SLO's grain: what drives of one
    #: campaign are pooled by.
    n_units: int
    #: The drive's mean log credit L: its estimate of B.
    mean_log_credit: float
    #: The sha256 of the image file that served.
    image_sha256: Sha256Hex
    #: The scorer's whole score of the drive, with no B, as ``score.json`` would hold it.
    score: dict[str, Any]


def measure(task, drive_dir: Path) -> CreditDrive:
    """The credit of the drive in *drive_dir* for *task*, loaded without a baseline credit.
    Raise Refused for a drive no baseline credit may pool."""
    if not task.scorer.scores_against_baseline():
        raise Refused(f"scorer {type(task.scorer).__name__} scores against no baseline, so there is no baseline "
                      "credit to measure")
    if task.baseline is None:
        raise Refused(f"task {task.id} declares no baseline server to measure")
    record = DriveRecord.model_validate_json((drive_dir / DRIVE_FILE).read_bytes())
    if record.outcome != "served":
        raise Refused(f"the baseline's drive did not serve ({record.outcome}: {record.reason}), so it measures "
                      "nothing")
    summary = read_summary(drive_dir / TRACE_DIR)
    if not summary.events:
        raise Refused("the baseline's drive sent no request, so it measures nothing")
    score = asyncio.run(task.scorer.score(task.effective_tenants(), summary, passthrough_verdict(summary),
                                          task.hardware))
    if score.zero_reasons:
        raise Refused(f"the baseline's drive scored zero ({', '.join(score.zero_reasons)}): the reference server did "
                      "not serve the graded workload cleanly")
    credit, units = task.scorer.measured_credit(score), task.scorer.measured_units(score)
    if credit is None or units is None:
        missing = "credit" if credit is None else "unit count"
        raise Refused(f"scorer {type(task.scorer).__name__} measured no {missing} on the drive")
    # A served drive's record names its server's readiness time and the drive's times
    # (``DriveRecord``), and the credit stage refuses a record that does not parse.
    artifact = record.artifact
    assert artifact is not None and artifact.ready_after_command_sec is not None
    assert record.drive_started_at is not None and record.drive_ended_at is not None
    ready = artifact.ready_after_command_sec
    started = record.drive_started_at - ready
    return CreditDrive(started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(),
                       duration_sec=ready + (record.drive_ended_at - record.drive_started_at),
                       n_requests=len(summary.events), n_units=units, mean_log_credit=credit,
                       image_sha256=artifact.image_sha256, score=json.loads(score.model_dump_json()))


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.score.credit_role")
    p.add_argument("--task-ref", type=Path, required=True)
    p.add_argument("--drive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        ref = TaskRef.model_validate_json(args.task_ref.read_bytes())
        record = measure(load_task(ref), args.drive)
    except Refused as e:
        reason = " ".join(str(e).split())
        print(f"{REFUSED_PREFIX}{reason}", file=sys.stderr)
        return REFUSED_EXIT
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("credit role failed; nothing was written", file=sys.stderr)
        return 1
    path = args.out / CREDIT_DRIVE_FILE
    try:
        path.write_bytes(record.model_dump_json(indent=2).encode())
        report_outputs(args.outputs_out, args.out, path)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("credit role failed writing its file", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
