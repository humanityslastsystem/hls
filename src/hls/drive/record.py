"""``drive.json``: what a drive of a served artifact did, as the deploy_drive stage publishes it
(``hls.stages.deploy_drive``) and the verify stage and the score and credit roles read it, with
the names of the drive's outputs they find it and its trace by."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, model_validator

from hls.drive.stream import TraceStreamRecord
from hls.spec.resources import ArtifactPlacement
from hls.store.digest import Sha256Hex
from hls.store.engine_cache import EngineCacheRecord
from hls.substrate.faults import Ending
from hls.substrate.nproc import ProcessCeiling
from hls.trace.store import END_FILE, TraceEnd

#: The output holding the ``DriveRecord``.
DRIVE_FILE = "drive.json"
#: The attempt output directory the driver writes the trace into.
TRACE_DIR = "trace"
#: The trace's last file, by its path under the attempt's outputs: the output the driver reports
#: and every reader finds the trace by.
TRACE_END = f"{TRACE_DIR}/{END_FILE}"


class ArtifactRecord(BaseModel):
    """What the artifact role did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Digest of the image file it ran.
    image_sha256: Sha256Hex
    #: How its role ended on its own; None when the stage stopped it.
    ending: Ending | None
    #: Its exit status when it ended on its own.
    exit_code: int | None
    #: Seconds from its step's start (``StageContext.role_started_at``) until the stage saw its
    #: command start (``StageContext.command_started_at``): the time its runtime took, which the
    #: stage allows ``DeployDriveParams.runtime_start_sec`` for. None when either was not seen.
    command_started_after_sec: float | None
    #: Seconds from its command's start (its start marker, ``StageContext.command_started_at``)
    #: until both readiness gates opened, the time its readiness budgets count; None when they
    #: never did.
    ready_after_command_sec: float | None
    #: Where its step ran: a split one ran on a site without ``site.evaluation_socket_tasks``,
    #: which records it and does not refuse it. None when its step reported nothing.
    placement: ArtifactPlacement | None = None
    #: The engine cache entry its server started from: for the task's
    #: measured baseline, the entry its warm stage left, by key and by the ENTRY.json digest of the
    #: entry its cache was a checked copy of. None for a server that started with an empty engine
    #: cache, as every other artifact's does.
    engine_cache: EngineCacheRecord | None


class DriverRecord(BaseModel):
    """What the driver role did."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: How its role ended.
    ending: Ending
    #: Its exit status.
    exit_code: int | None
    #: The drive wall it was given, in seconds from the drive's own start.
    drive_timeout_sec: float
    #: How its trace writer kept up with the drive, by the role's own record.
    trace_stream: TraceStreamRecord


class SealProbeRecord(BaseModel):
    """The seal probe that ran before the artifact, every check passed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The host the refusal check asked for, which the policy refuses; None when the policy
    #: refuses no host, so the check was not made.
    refused_host: str | None
    #: Seconds the refusal check took; None when it was not made.
    refusal_sec: float | None
    #: Seconds the ingress check took: from the probe's listening on the artifact's port until
    #: it answered the stage's request through the seal's host port.
    ingress_sec: float
    #: Seconds the check took that read, inside the seal, the limits on the uid's per-namespace
    #: counts its seal wrote, found each as the stage gave it, and could not raise one.
    ucounts_sec: float
    #: Seconds the check took that read the probe's own capability sets and found each empty.
    caps_sec: float
    #: Seconds the ICMP check took, which found that no echo could leave the seal.
    icmp_sec: float
    #: The TCP port the probe bound in its seal besides the artifact's, which the stage found
    #: closed from every address of the node and which nothing reached.
    inbound_tcp: int
    #: The UDP port the probe bound in its seal, which nothing reached from any address of the
    #: node and from which no answer came.
    inbound_udp: int
    #: The node's addresses the stage knocked on them from, loopback among them.
    inbound_addresses: tuple[str, ...]
    #: Seconds the inbound check took: from the stage's first request answered to its second.
    inbound_sec: float
    #: Seconds from the probe's start in its seal to its role's end.
    sec: float


class DriveRecord(BaseModel):
    """``drive.json``: how the artifact was served and driven."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``served``, or why there is no drive to verify: ``invalid_artifact`` (the package said
    #: so), ``crashed`` (the server ended before it was ready) or ``not_ready`` (it still ran,
    #: but was not ready within the readiness budget).
    outcome: Literal["served", "invalid_artifact", "crashed", "not_ready"]
    #: Why it was not served, for the contestant's record; set exactly when not served.
    reason: str | None
    #: The probe of the artifact's seal; None for an invalid artifact, which never ran.
    seal_probe: SealProbeRecord | None
    #: The process ceiling the artifact's step ran under (``hls.substrate.nproc``); None for an
    #: invalid artifact, which never ran.
    process_ceiling: ProcessCeiling | None
    #: The artifact role; None for an invalid artifact, which never ran.
    artifact: ArtifactRecord | None
    #: The driver role; set exactly when served.
    driver: DriverRecord | None
    #: The URL the server was probed and driven at; None when it never had one.
    url: str | None
    #: When the driver started, in POSIX seconds; set exactly when served.
    drive_started_at: float | None
    #: When the driver ended, in POSIX seconds; set exactly when served.
    drive_ended_at: float | None
    #: The trace's ``END.json``; set exactly when served.
    trace: TraceEnd | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        served = self.outcome == "served"
        if served != (self.reason is None):
            raise ValueError("a drive that did not serve says why, and only such a drive")
        if served != (None not in (self.driver, self.drive_started_at, self.drive_ended_at, self.trace)):
            raise ValueError("a served drive names its driver, its times and its trace, and only a served "
                             "one")
        ran = self.outcome != "invalid_artifact"
        if not ran == (self.artifact is not None) == (self.seal_probe is not None) == (
                self.process_ceiling is not None):
            raise ValueError("every artifact but an invalid one ran, under its process ceiling, after its seal "
                             "was probed")
        if served and self.artifact is not None and self.artifact.ready_after_command_sec is None:
            raise ValueError("a served drive's server opened its readiness gates, so its record says when")
        return self
