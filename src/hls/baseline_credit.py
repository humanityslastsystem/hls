"""Baseline credit: the measured constant a speedup-scored task subtracts.

A speedup-scoring scorer reports ``S = exp(L - B)``, where ``L`` is the graded
run's mean log credit over the units its SLO scores (sessions at session grain,
requests at request grain) and ``B`` is the same quantity for the task's own
reference server. ``B`` is not authored by hand -- it is measured, by driving
the task's ``Baseline`` on the exact graded workload several times
(``hls baseline-credit <module>``), each drive on a fresh deploy, into a ``baseline.json``
that the measurement writes into the store (``hls.measured``).

This module is the fail-loud bridge between that file and the TaskSpec whose
scorer needs it, mirroring ``hls.verify.calibration``: ``read_baseline_credit``
refuses a stale, differently-loaded or GPU-mismatched artifact outright, and
``credited`` threads ``B`` into the (frozen) TaskSpec's scorer
without mutating the original tree.

``CampaignIdentity`` is what the file is keyed by, because ``B`` depends on more
than the prompts:

  - ``workload_fingerprint`` -- the same hash a calibration is keyed by. A change
    to the prompts, the model, or the split sizes invalidates ``B``.
  - ``LoadRecord`` -- the arrival processes, think-times and SLO budgets the
    baseline was driven under. The workload fingerprint deliberately excludes
    all three (see ``hls.verify.calibration.workload_payload``), and every one of
    them moves ``B``: load sets the queueing the reference meets, and the SLO is
    the yardstick each latency is divided by.
  - the GPU kind and the reference server itself.

The drives are measured by the ``hls baseline-credit`` pipeline (``hls.pipeline.credit``):
each one a stage's drive of the task's measured baseline image, scored by
``hls.score.credit_role``, and the drives pooled by ``credit_from_drives`` in its fold stage,
which writes it with ``baseline_credit_bytes``.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError, model_validator

from hls.core.poly import Spec, dump_spec
from hls.measured import BASELINE_CREDIT_FILE
from hls.spec.datasets import Split
from hls.spec.resources import BaselineHardware, HardwareRecord
from hls.verify.calibration import workload_fingerprint

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec

#: Drives a measurement must take before it may be written. A credit of one drive records no
#: spread between drives (``BaselineCredit.log_credit_sd`` is None), and says so; a careful
#: measurement takes several (``hls baseline-credit --drives N``).
MIN_DRIVES = 1

#: Largest fraction by which one drive's unit count may fall short of the campaign's
#: largest and still be pooled. Every drive of one campaign serves the same fixed set
#: of units, so a shortfall past this is a unit the drive lost rather than run-to-run
#: variation (see ``credit_from_drives``).
_UNIT_COUNT_TOLERANCE = 0.05

#: What a refusal names as its fix.
_FIX = "measure it again with `hls baseline-credit <module>`, which writes it into the store"


class BaselineCreditError(Exception):
    """Raised whenever a task that grades on a measured baseline credit cannot
    safely proceed: no baseline.json, one measured on a different workload, load
    or GPU kind, or a measurement that could not produce a credit at all. Every
    raise site names the fix (``hls baseline-credit <module>``, ``_FIX``)."""


class BaselineDrive(Spec):
    """One drive of the baseline: a fresh deploy, one clocked serving of the
    graded workload, and the mean log credit it earned."""

    #: ISO-8601 UTC timestamp the drive began: the driver's start less the seconds the server
    #: took to open its readiness gates (``hls.score.credit_role``).
    started_at: str
    #: Wall-clock seconds of the drive: the server's readiness and the drive, without the wait
    #: between them.
    duration_sec: float
    #: Requests the drive served. Informational only. At session grain a trajectory's
    #: turn count is chosen by the server's own replies, so two drives of the same
    #: sessions legitimately differ here.
    n_requests: int
    #: Units the drive's scorer credited: sessions at session grain, requests at
    #: request grain. The set of units is fixed by the workload draw, so this is what
    #: drives are pooled by. None where a drive's record holds none, which
    #: ``read_baseline_credit`` accepts and ``credit_from_drives`` refuses.
    n_units: int | None = None
    #: Mean over those units of ``log(1/r)``, ``r`` being each unit's SLO-normalized
    #: latency: this drive's estimate of ``B``.
    mean_log_credit: float


class BaselineRecord(Spec):
    """The reference server as it was driven: which image served, and how.

    ``B`` is a claim about this exact server, so both halves matter -- a
    different image and a different serve flag are each a different operating
    point. Both halves are authored task config, so it is compared whole against the
    task about to be graded (``check_campaign``), so a retuned baseline cannot keep a
    stale ``B``."""

    #: The reference image the task declares (``Baseline.image``): the authored string,
    #: which a retuned task changes, not the digest of the file a drive served.
    image: str
    #: The extra launch args the baseline was served with
    #: (``Baseline.serve_command``); host and port are derived from the contract
    #: and are not part of it.
    serve_command: list[str]


class TenantLoad(Spec):
    """One tenant's traffic timing and SLO budget, as the baseline met them."""

    #: The tenant id the entry belongs to.
    id: str
    #: Its arrival process, serialized (``ArrivalProcess``).
    arrival: dict
    #: Its think-time between turns of one session (``InterTurnDelay``).
    inter_turn_delay: dict
    #: Its service-level objective, which sets the budget every latency is
    #: divided by to give the ``r`` the credit is a log of.
    slo: dict


class LoadRecord(Spec):
    """What the baseline was driven at, beyond the prompts themselves.

    Everything here is deliberately absent from the workload fingerprint, which
    identifies the population of prompts and not the pace they are served at.
    ``B`` depends on the pace and on the yardstick, so the two identities
    together are what make a stored credit valid for a task."""

    #: The dataset split served. Always the test split -- ``B`` must be measured
    #: on the same traffic a graded run serves -- and recorded so the file says
    #: so rather than leaving a reader to assume it.
    split: str
    #: One entry per effective tenant, in the task's own tenant order.
    tenants: list[TenantLoad]


def credit_stats(credits: list[float]) -> tuple[float, float | None, float | None]:
    """``(mean, sample sd, standard error)`` of a campaign's per-drive credits. One drive has no
    spread between drives, so its sd and standard error are None rather than a number."""
    if not credits:
        raise BaselineCreditError("no drives to summarize")
    mean = statistics.fmean(credits)
    if len(credits) == 1:
        return mean, None, None
    sd = statistics.stdev(credits)
    return mean, sd, sd / math.sqrt(len(credits))


class CampaignIdentity(Spec):
    """What a set of drives is a measurement of: the task, the traffic, the load,
    the hardware and the reference server.

    One comparable unit rather than five loose fields, because it is compared
    whole: ``read_baseline_credit`` holds it to the task about to be graded, and
    the fold stage builds it once for every drive it pools.

    Fields:
        task:                  Dotted module path the credit was measured for
                               (e.g. ``"tasks.multitenant_slo.task"``).
        workload_fingerprint:  The ``workload_fingerprint(task)`` hash the
                               measurement is valid for (the prompts, the model,
                               the split sizes).
        load:                  The arrival processes, think-times, SLO budgets
                               and split the drives ran under. See ``LoadRecord``.
        gpu_kind:              The accelerator kind the drives ran on. A
                               reference server's latency is hardware-specific.
        baseline:              The reference server as deployed.
    """

    task: str
    workload_fingerprint: str
    load: LoadRecord
    gpu_kind: str
    baseline: BaselineRecord


class BaselineCredit(Spec):
    """The full record of one baseline-credit measurement for one task folder.

    The three summaries are computed from ``drives`` on every read and are not
    stored: ``B`` multiplies every score by ``exp(-B)``, and an edited mean would
    be a silent global rescale of the leaderboard. The file holds the drives, so
    there is no summary in it to edit, and a file that carries one is refused
    like any key the model has no field for.

    Fields:
        campaign:              What the drives measured, and what
                               ``read_baseline_credit`` gates a run on.
        drives:                Every drive, ordered by start time.
        created_at:            ISO-8601 timestamp the artifact was built.
        hardware:              The distinct hardware records of the allocations the drives
                               ran on, sorted; more than one when the drives ran on
                               different hardware. A credit the harness measures records
                               every field its drives' grant checks measured; a record may
                               leave fields blank, which a grading is not compared on, and
                               the file may record none. It never
                               moves ``B``.
    """

    campaign: CampaignIdentity
    drives: list[BaselineDrive]
    created_at: str
    hardware: list[BaselineHardware] | None = None

    @model_validator(mode="after")
    def _check_drives(self) -> "BaselineCredit":
        if len(self.drives) < MIN_DRIVES:
            raise BaselineCreditError(
                f"a baseline credit needs at least {MIN_DRIVES} drive, got {len(self.drives)}; {_FIX}")
        if self.hardware is not None and (not self.hardware or self.hardware != _distinct(self.hardware)):
            raise ValueError("a baseline credit's hardware is its drives' distinct records, sorted, at least one")
        return self

    @property
    def log_credit_mean(self) -> float:
        """``B`` itself: the mean of the drives' credits."""
        return credit_stats([d.mean_log_credit for d in self.drives])[0]

    @property
    def log_credit_sd(self) -> float | None:
        """Sample standard deviation across drives -- the drive-to-drive spread of
        the reference itself; None for a credit of one drive, which has none."""
        return credit_stats([d.mean_log_credit for d in self.drives])[1]

    @property
    def log_credit_se(self) -> float | None:
        """``sd / sqrt(n)``: how well ``B`` is pinned. A score is ``exp(L - B)``, so
        an error of ``e`` in ``B`` is a factor of ``exp(e)`` on every score. None for a
        credit of one drive."""
        return credit_stats([d.mean_log_credit for d in self.drives])[2]


def spread_said(bc: "BaselineCredit") -> str:
    """The spread of *bc*'s drives, for people: its sd and standard error, or that one drive has
    no spread between drives."""
    if bc.log_credit_sd is None or bc.log_credit_se is None:
        return f"{len(bc.drives)} drive, so no spread between drives"
    return f"sd {bc.log_credit_sd:.4f}, se {bc.log_credit_se:.4f}, {len(bc.drives)} drives"


def load_record(task: "TaskSpec") -> LoadRecord:
    """The ``LoadRecord`` for ``task`` as a graded run would drive it: every
    effective tenant's arrival, think-time and SLO, on the test split."""
    return LoadRecord(
        split=Split.TEST.value,
        tenants=[
            TenantLoad(id=t.id,
                       arrival=dump_spec(t.workload.arrival),
                       inter_turn_delay=dump_spec(t.workload.inter_turn_delay),
                       slo=dump_spec(t.workload.slo))
            for t in task.effective_tenants()
        ],
    )


def _load_diff(left: LoadRecord, right: LoadRecord) -> list[str]:
    """One line per way the measured load *left* differs from the declared *right*, so a
    refusal names what changed instead of printing both records whole (four tenants of
    arrivals, think-times and budgets is a wall of JSON a reader has to diff by eye).

    Total: two records that are unequal always produce at least one line, since a
    difference no field comparison catches is a difference in tenant order."""
    a, b = "measured", "declared"
    lines: list[str] = []
    if left.split != right.split:
        lines.append(f"split: {a} {left.split!r}, {b} {right.split!r}")
    was = {t.id: t for t in left.tenants}
    now = {t.id: t for t in right.tenants}
    if dropped := sorted(set(was) - set(now)):
        lines.append(f"tenants only on the {a} side: {', '.join(dropped)}")
    if fresh := sorted(set(now) - set(was)):
        lines.append(f"tenants only on the {b} side: {', '.join(fresh)}")
    for tenant_id, right_tenant in now.items():
        left_tenant = was.get(tenant_id)
        if left_tenant is None:
            continue
        for field in ("arrival", "inter_turn_delay", "slo"):
            mine, theirs = getattr(left_tenant, field), getattr(right_tenant, field)
            if mine != theirs:
                lines.append(
                    f"{tenant_id}.{field}: {a} {json.dumps(mine, sort_keys=True)}, "
                    f"{b} {json.dumps(theirs, sort_keys=True)}")
    if not lines:
        lines.append(f"the tenants changed order: {a} {[t.id for t in left.tenants]}, "
                     f"{b} {[t.id for t in right.tenants]}")
    return lines


def _baseline_diff(left: BaselineRecord, right: BaselineRecord) -> list[str]:
    """One line per half of the reference server that moved from the measured *left* to the
    declared *right*, so a refusal names the image or the serve command rather than printing
    both records whole.

    Two unequal records always produce at least one line: the record has these two
    fields and nothing else."""
    a, b = "measured", "declared"
    lines: list[str] = []
    if left.image != right.image:
        lines.append(f"image: {a} {left.image!r}, {b} {right.image!r}")
    if left.serve_command != right.serve_command:
        lines.append(f"serve_command: {a} {json.dumps(left.serve_command)}, "
                     f"{b} {json.dumps(right.serve_command)}")
    return lines


def check_campaign(campaign: CampaignIdentity, task: "TaskSpec", where: str) -> None:
    """Refuse a campaign that measured something other than ``task``.

    The four refusals a credit can earn against the task it is about to grade,
    each naming what differs and the fix: a different workload, a different load,
    a different GPU kind, or a different reference server. ``where`` names what is
    being checked (the file) so the message points at it.

    The reference server is checked here like the rest, because both halves of
    ``BaselineRecord`` are authored task config: swapping a serve flag moves the
    queueing the reference meets and so moves ``B``, which is a factor on every
    score the leaderboard carries."""
    fingerprint = workload_fingerprint(task)
    if campaign.workload_fingerprint != fingerprint:
        raise BaselineCreditError(
            f"stale baseline credit in {where}: it was measured for "
            f"workload_fingerprint={campaign.workload_fingerprint!r} but this task's "
            f"current fingerprint is {fingerprint!r}; {_FIX}")
    want_load = load_record(task)
    if campaign.load != want_load:
        changed = "\n  ".join(_load_diff(campaign.load, want_load))
        raise BaselineCreditError(
            f"baseline credit in {where} was measured under a different load than this "
            f"task now declares, so its credit is not comparable:\n  {changed}\n{_FIX}")
    gpu_kind = task.hardware.accelerator.kind
    if campaign.gpu_kind != gpu_kind:
        raise BaselineCreditError(
            f"baseline credit in {where} was measured on gpu_kind={campaign.gpu_kind!r} "
            f"but this task's hardware.accelerator.kind={gpu_kind!r}; a reference "
            f"server's latency is hardware-specific -- {_FIX}, on the target GPU")
    if task.baseline is None:
        raise BaselineCreditError(
            f"baseline credit in {where} was measured against a reference server, but "
            f"this task declares no baseline to hold it to; a task that scores as a "
            f"speedup must declare the Baseline its credit is measured on")
    want_baseline = BaselineRecord(image=task.baseline.image,
                                   serve_command=list(task.baseline.serve_command))
    if campaign.baseline != want_baseline:
        changed = "\n  ".join(_baseline_diff(campaign.baseline, want_baseline))
        raise BaselineCreditError(
            f"baseline credit in {where} was measured on a different reference server "
            f"than this task now declares, so its credit is not comparable:\n  {changed}\n{_FIX}")


def read_baseline_credit(path: Path, task: "TaskSpec") -> tuple[BaselineCredit, bytes]:
    """Read and validate the baseline credit file at *path* against ``task``, once: the credit and
    the bytes it was parsed from, which a score names by their blob id.

    Fails loud (``BaselineCreditError``, always naming the fix) rather than ever
    grading against a credit measured on something else: the file is missing or
    does not fit the model, or its campaign identity does not match the task
    (``check_campaign``).
    """
    path = Path(path)
    if not path.exists():
        raise BaselineCreditError(
            f"no baseline credit at {path}; this task scores against a "
            f"measured baseline credit -- {_FIX}")
    data = path.read_bytes()
    try:
        bc = BaselineCredit.model_validate_json(data)
    except ValidationError as exc:
        # Malformed JSON and JSON that does not fit the model both arrive here as a
        # pydantic ValidationError (pydantic-core parses the JSON itself).
        raise BaselineCreditError(
            f"{path} is not a valid baseline credit file ({exc}); {_FIX}") from exc
    check_campaign(bc.campaign, task, str(path))
    return bc, data


def credited(task: "TaskSpec", path: Path | None) -> tuple["TaskSpec", tuple[BaselineCredit, bytes] | None]:
    """Return ``task`` with the measured credit threaded into its scorer, never
    mutating the input, and the credit with the bytes of the file it was parsed from
    (``read_baseline_credit``); None for those when the scorer grades against no baseline.

    Returns ``task`` unchanged (same object) when its scorer does not grade against a
    baseline. Otherwise requires the file's *path* (``None`` raises), reads the file once, and
    rebuilds the frozen tree with ``model_copy``, which validates the copy
    (``Spec.model_copy``), so the task passes its wiring checks again; threading ``B`` into
    the scorer changes no field they read."""
    if not task.scorer.scores_against_baseline():
        return task, None
    if path is None:
        raise BaselineCreditError(
            f"task scores against a measured baseline credit but no {BASELINE_CREDIT_FILE} was given; {_FIX}")
    bc, data = read_baseline_credit(path, task)
    return task.model_copy(
        update={"scorer": task.scorer.with_baseline_credit(bc.log_credit_mean)}), (bc, data)


def _distinct(records: list[BaselineHardware]) -> list[BaselineHardware]:
    """*records* without repeats, sorted by their JSON."""
    return sorted(set(records), key=lambda h: h.model_dump_json())


def baseline_credit_bytes(bc: BaselineCredit) -> bytes:
    """The bytes of ``baseline.json`` for ``bc``: its JSON-mode dump with an indent of two,
    without ``hardware`` when it records none and without a hardware record's blank fields. The
    fold stage writes the credit with them."""
    data = bc.model_dump(mode="json")
    if bc.hardware is None:
        del data["hardware"]
    else:
        data["hardware"] = [record.model_dump(mode="json", exclude_none=True) for record in bc.hardware]
    return json.dumps(data, indent=2).encode()


def credit_from_drives(campaign: CampaignIdentity, drives: list[BaselineDrive],
                       hardware: list[HardwareRecord] | None = None) -> BaselineCredit:
    """Build the artifact from a set of drives of one campaign: the fold stage of
    ``hls baseline-credit`` builds the credit here. Drives are ordered by start time,
    so the file reads as a chronology however many allocations drove them.

    Refuses a drive that credited far fewer units than its siblings. The workload draw
    fixes the set of units every drive of one campaign serves: the same sessions at
    session grain, the same requests at request grain. A shortfall past
    ``_UNIT_COUNT_TOLERANCE`` therefore means a drive lost a unit, and averaging it in
    would shift ``B`` for a reason the file does not record.

    Request counts are not compared. At session grain a trajectory takes as many turns
    as the server's own replies call for, so the same sessions can come to different request
    counts between drives with every session answered."""
    if not drives:
        # Checked before the unit-count comparison below, which has no largest to
        # measure against, so an empty campaign lands on this refusal rather than on a
        # bare ValueError out of max().
        raise BaselineCreditError(
            f"no drives to build a baseline credit from; a campaign needs at least "
            f"{MIN_DRIVES}; {_FIX}")
    ordered = sorted(drives, key=lambda d: d.started_at)
    if uncounted := [d for d in ordered if d.n_units is None]:
        # Every drive the credit role measures carries its unit count, so a drive arriving
        # here without one was built by hand.
        raise BaselineCreditError(
            f"{len(uncounted)} of these {len(ordered)} drives record no unit count, so "
            f"there is nothing to check they served the same workload; {_FIX}")
    most = max(d.n_units for d in ordered)
    short = [d for d in ordered if d.n_units < (1.0 - _UNIT_COUNT_TOLERANCE) * most]
    if short:
        served = ", ".join(f"{d.started_at}: {d.n_units} units over {d.n_requests} "
                           f"requests" for d in ordered)
        raise BaselineCreditError(
            f"these drives credited too different numbers of units to be drives of "
            f"one workload (more than {_UNIT_COUNT_TOLERANCE:.0%} below the largest), "
            f"so they cannot be pooled ({served})")
    return BaselineCredit(campaign=campaign, drives=ordered,
                          created_at=datetime.now(timezone.utc).isoformat(),
                          hardware=(_distinct([BaselineHardware.of(record) for record in hardware])
                                    if hardware is not None else None))
