"""The deploy_drive stage: serve the contestant's artifact, drive it with the task's own driver,
and publish how it went.

The stage runs in the runner and stays light; the work is done by roles:
1. it reads the package stage's ``package.json``. A submission that did not build
   (``invalid_artifact``) is a result: the stage publishes ``drive.json`` saying so, starts no
   role, and verify and score write a zero. A measured artifact (``ImageArtifact``) is an image
   of this instance's ``images.json`` instead, and a published submission image graded again
   (``PublishedArtifact``) a file on the cluster's shared filesystem named by its digest;
2. before it starts any role, it copies to the node every cache entry the two roles mount
   (``data_mounts`` through ``materialize``; the driver's tool bundle whole) and pins what the
   containers run, each extracted on the node from an image file whose digest its producer
   recorded: the submission image from its place in the store with ``pin_file`` against the
   digest ``package.json`` records, which the package manifest must record too, marked as an
   image holding the contestant's code; a published submission image from its file against the
   digest the plan names, marked so too, whose refusal a retry would meet again; or the measured
   image; and the harness image, each with
   ``node_image`` against the entry digest ``images.json`` records. A copy or an extraction runs
   alone, so none can wait until a server runs. The extracted trees are
   read-only to every container, and the stage's disk declares them (``PINNED_IMAGES`` of
   ``KEPT_DISK_GIB``). A contestant's container sees nothing else: no store path, no checkout,
   no shared filesystem. The server's engine cache is made then too (``_Drive.engine_cache``):
   for the task's measured baseline, a writable copy of the engine cache
   entry its warm stage left (``engine_cache_owner``, ``hls.stages.baseline_warm``), keyed by the
   baseline's image, start, GPUs, weights, build environment and seal (``served_cache_key``),
   each file checked against the entry; for any other artifact, an empty directory, so a
   contestant's own build and JIT time count against its readiness budget;
3. ``seal-probe`` (harness), just before the artifact: a container on the harness image sealed
   exactly as the artifact will be, its limits on the uid's per-namespace counts and its
   capabilities included, which shows that it holds no capability, that those limits read inside
   the seal as the stage gave them and cannot be raised there, that a host the
   policy refuses is refused, as the seal's interceptor records in its status file
   (``policy_refusals``), that the artifact's port is reached through the seal's host port,
   that no ICMP echo leaves the seal, and that no other port the container binds, a TCP and a
   UDP one below the node's ephemeral range, is reached from any address of the node
   (``hls.egress.seal_probe``). A probe that fails, or that runs past
   ``SEAL_PROBE_SEC`` once its seal started it, is infra and retried, and no artifact starts.
   Then every file the driver will
   need on the node is made (``prepare_driver``), so a server that fills the node's disk leaves
   the driver nothing to make there;
4. ``artifact`` (contestant): the image in a sealed container (``entrypoint``, then ``--host
   0.0.0.0 --port <port>``, then ``serve_command``). The seal enforces the artifact's runtime
   policy (None is the air gap) and runs it in a network namespace of its own, where it
   listens on ``port`` whatever other servers on the node listen on, and forwards a free host
   port to it. The container mounts the artifact's data (``data_mounts``) and its node-local
   engine cache at ``/engine-cache``; its /tmp, /var/tmp and $HOME are its workdir, also on the
   node's disk. The workdir and the engine cache are all it can write, capped together at
   ``artifact_shape.disk_gib``. Its variables are the image's runtime variables, overridden by
   the cache variables, the mounts' variables (``HF_HUB_OFFLINE=1`` among them),
   ``TORCH_CUDA_ARCH_LIST`` for the stage's GPU kind, then the seal's client variables. No
   secret reaches it. Its step runs under a process ceiling (``hls.substrate.nproc``, recorded
   in ``drive.json`` with the headroom and the cap it came from): every task of the uid on the
   node draws on one pool, and the ceiling keeps the reserve of each evaluation's harness out of
   the contestant's reach, and the step under ``site.nproc_max`` where the uid's hard limit
   leaves more. A ceiling or a cap under its floor is infra, not retried, before any role starts. Where its step's CPUs lie is in
   ``drive.json`` (``ArtifactPlacement``). On a site with ``site.evaluation_socket_tasks`` the
   allocation asked for the artifact's CPUs as one of two tasks with one per socket
   (``hls.pipeline.planner.socket_tasks``), and a step whose CPUs Slurm still split over NUMA
   nodes is refused before the server starts (``RoleSpec.one_numa_node``): its role ends
   ``grant_mismatch``, infra and retried, its CPUs and nodes in the evidence. A site without it
   records a split and serves it;
5. readiness: the host port the seal wrote, then the start of the server's command (its start
   marker, ``StageContext.wait_command_started``), which only the stage's deadline bounds, then
   ``wait_healthy`` and ``wait_inference_ready`` against ``http://127.0.0.1:<port>``, each with
   the full ``readiness_timeout_sec``, the first counted from the command's start,
   stopped at once when the artifact's role ends. The stage's deadline is held for both budgets
   and the margin first, from now and again from the command's start, as for the drive's wall
   below, so a staging that ran long does not cut the wait for the command or a gate;
6. ``driver`` (harness): ``python -m hls.drive.role`` on the harness image, with the checkout
   read-only at ``/opt/hls-src``, the task reference, the driver's data mounts, the trace's room
   on the node's disk read-write (``LOCAL``: made before the artifact starts, at the most the
   trace takes, every block given then, ``hls.trace.store.make_room``), where the role writes the
   drive's trace in place as the drive goes, so a server that fills the node's disk takes none of
   it, the attempt's ``outputs/`` directory read-write, where the role publishes ``trace/`` once
   the drive is over, and the role's own report of the outputs it wrote and record of how its
   trace writer kept up with the drive (``STREAM_RECORD``). For a task whose tool runner starts
   containers of its own (``sandbox_paths``), it also gets the site's sandbox stores at the paths
   the runner declares, read-only unless the runner writes them, and the device node a rootless
   runtime mounts an image through (``sandbox_binds``), each checked on the node before the
   artifact starts, and the checkout's harbor on its path (``HARBOR_SRC``); no other container
   gets them. The driver's shape
   declares the room's disk (``hls.drive.bounds.trace_disk_bytes``), and the margin after the
   wall covers publishing the trace to the store (``hls.pipeline.pipelines.TRACE_COPY_SEC``). The
   stage removes the room once its roles are stopped, however it ends (``_Drive.drop_room``). Its
   wall (``drive_timeout_sec``: the task's own, or
   ``OPEN_DRIVE_SEC`` for a task that declares none) counts from the drive's own start, and a drive
   always gets all of it or does not start: the harness's delays never shorten it. When the
   driver is started, the stage's deadline is held to fit the wall and ``wall_margin_sec`` after
   it (``StageContext.hold_until``), and it grows by the time the driver's step then takes to
   start (``StageContext.deadline``), so the drive's own wall fires before the deadline would kill
   it, and the faulted tail is recorded as a scored result. The deadline grows no further than the
   allocation's end, though. The server is not started when its two readiness budgets, the wall
   and the margin do not fit before that end; once it is ready, the driver is not started when
   the wall and the margin no longer fit, and a driver whose step starts too late for them does
   not drive (``hls.drive.role``, ``--start-by``). Each time the stage is infra, retried, with
   each role's wait for its step in the evidence (``no_room``); a server already started is
   stopped first, and its own ending decides when it is the contestant's fault, or adds its own
   evidence when it is infra. The role holds the trace to the task's ``trace_cap_bytes``, the
   server's replies to ``reply_max_bytes`` each, and itself to 85% of the driver's declared
   memory, which the stage always passes so the role's memory guard is on
   (``hls.drive.bounds``, ``hls.drive.replies``, ``hls.drive.memory``); a reply past any of them
   fails as the contestant's request. A drive of the task's measured baseline
   (``measured_baseline``) counts nothing and has no guard. With ``driver_overlaps``
   its step shares the artifact's CPUs;
7. the stage waits for the driver while the artifact serves, and waits for an artifact that
   ends first at once. Whenever the artifact ends at its disk cap, before the drive, during it
   or at its stop, its writable directories are removed before anything else runs
   (``_Drive.cleared``). Once the driver ended, the stage stops the artifact
   (``StageContext.stop_role``: a planned stop, no fault), records the trace's ``END.json`` by
   the driver's own report of it, and its trace writer's record (``DriverRecord.trace_stream``),
   and publishes ``drive.json``. Output ``trace`` names the
   trace's ``END.json``, whose digest pins every chunk and the summary; readers take its
   directory. The runner reads ``END.json`` alone (``read_end``), never a chunk: the verify role
   checks each chunk against its digest as it reads it.

Whose ending it is (the contestant's only when
certain) is the roles' to decide (``stage_fault``):
- the artifact's own exit after its start marker, without an OOM kill, is a result: before it
  was ready, ``crashed``; during the drive, the drive's failed requests stand and the drive is
  published as ``served``. A server still running but not ready when a gate's full budget ran
  out is ``not_ready``; one that died while the gate ran crashed, which a last look at its
  role tells. Each is published and scored, and only once the role's start marker is on its
  stderr (``StageContext.command_started``): the seal's started file says only that the
  runtime was launched, so a container whose command never started, however it ended, is
  infra. The exit status and stderr of a contestant's server decide nothing else: a server
  that prints ``refused:`` and exits 3 has crashed like any other;
- an artifact role that reached its own memory limit for certain, wrote past its disk cap, or
  whose step reached its pids limit within ``OOM_EVIDENCE_MAX_AGE_SEC`` of its end or its stop
  (``within_evidence_age``), is a contestant fault: the stage raises it, publishes
  nothing, and nothing retries it, unless another role failed as infra beside it, which leaves
  the stage infra (``stage_fault``);
- a server that crashed or died during the drive, with any status, 0 among them, or was not
  ready, while the uid's other tasks on the node alone filled its process pool, counted once its
  step was gone, was starved by them (``starvation_fault``, asked through ``scored_end_fault``):
  infra, retried, with the count and the walks' last sample in the evidence; and so is one whose
  count failed, which leaves starvation not ruled out;
- everything else is infra and publishes nothing: an artifact role that the runtime failed
  before its command started (``runtime_failed``: a seal that never started its payload), that
  the scheduler ended, that did not launch or report its grant, that got another grant than
  declared, that the allocation stopped, or that an OOM kill not certainly its own ended; a
  seal that wrote no host port; a seal probe that failed; a trace's room the node's disk could
  not give its blocks to, which no server has started beside; a server whose container had not
  started its command by the stage's deadline; a readiness gate the stage's time cut short of its
  budget; a driver or an extraction or copy role that did not exit 0, whatever its
  stderr says (the driver's carries what the server's replies said), a driver's fault carrying
  its own record of why its trace writer failed, when it wrote one (``_Drive.with_stream``); a
  driver that reported no trace, one whose ``END.json`` names files that are not there, or no
  record of its trace writer; an image file whose digest is
  not the recorded one; a measured baseline's engine cache entry that its warm stage left none of,
  left under another key than the drive's (``engine_cache.owner_entry``), that is corrupt or
  published again since, or whose copy differs from it (``StageContext.seed``).
  A task module that does not import, or a plan whose inputs are missing or disagree, is infra
  too, and not retryable.
Once a role's contestant fault is decided, the stage raises the roles' fault and never one of
its own (``StageContext.contestant_fault``).

The runner imports the task module for its contract alone (the readiness probe). Nothing
else of the task runs here: the driver rebuilds the task inside its own container."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import posixpath
import stat
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, ClassVar, Literal, Self

import httpx
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator

from hls.drive.bounds import trace_event_bytes
from hls.drive.readiness import NotReady, NotServing, wait_healthy, wait_inference_ready
from hls.drive.record import (DRIVE_FILE, TRACE_DIR, TRACE_END, ArtifactRecord, DriveRecord, DriverRecord,
                              SealProbeRecord)
from hls.drive.role import LATE_START_EXIT, NOFILE_SOFT
from hls.drive.stream import TraceStreamRecord
from hls.egress.policy import PROBE_DIAL, AccessList, NetworkPolicy, SealPolicy, refused_host
from hls.egress.seal_probe import (CHECK_SEC, Check, inbound_ports, knock_failures, knock_inbound, node_addresses,
                                   read_report)
from hls.package.result import PackageOutcome
from hls.pipeline.stage import Stage, StageContext, StageFault, register, scored_end_fault, stage_fault
from hls.provision.model import Provisioned, WeightsRef
from hls.provision.mounts import DataCopies, data_mounts
from hls.settings import StageName, sandbox_store_problems
from hls.spec.resources import ArtifactPlacement, ResourceShape, concurrent
from hls.spec.tools import DEVICE, HostPath
from hls.stages import engine_cache
from hls.stages.engine_cache import ENGINE_CACHE, engine_build_env, engine_cache_key
from hls.stages.inputs import dev_files
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.package import IMAGES_OUTPUT, PACKAGE_ADDS_BYTES, SUBMISSION_OUTPUT
from hls.store.base import check_segment
from hls.store.digest import Sha256Hex, sha256_bytes
from hls.store.engine_cache import EngineCacheRecord
from hls.store.manifest import Manifest, ManifestKey, OutputRef
from hls.substrate.seal import INGRESS_WAIT_SEC, ingress_port, namespace_ports, policy_refusals
from hls.substrate.factory import substrate
from hls.substrate.ucounts import SEAL_UCOUNTS_VAR, ucounts_env
from hls.substrate.base import FaultRecord, RoleHandle
from hls.substrate.container import ContainerSpec, Mount, PinnedImage, SealPlan
from hls.substrate.faults import RoleOutcome, classify_role
from hls.substrate.memory_watch import Unmeasured, disk_usage, remove_written
from hls.substrate.nproc import ProcessCeiling
from hls.taskref import TaskRef, load_task
from hls.trace.store import TraceIncomplete, make_room, read_end, room_bytes
from hls.util import remove_tree, validation_errors

#: The role names: the server, its driver, and the probe of the seal the server runs behind.
ARTIFACT, DRIVER, SEAL_PROBE = "artifact", "driver", "seal-probe"
#: Where the checkout, the task reference, the outputs directory and the driver's report of
#: the outputs it wrote mount in the driver's container.
CHECKOUT, TASK_REF = "/opt/hls-src", "/hls/task-ref.json"
OUT, OUTPUTS_REPORT = "/hls/out", "/hls/outputs.json"
#: Where the trace's room mounts in the driver's container, from the role's ``LOCAL_DIR`` on the
#: node's disk; the role writes the drive's trace into it as the drive goes.
LOCAL, LOCAL_DIR = "/hls/local", "trace-local"
#: Where the driver's caches mount in its container (``driver_cache_env``), from the attempt's
#: ``DRIVER_CACHE_DIR`` on the store's filesystem.
DRIVER_CACHE, DRIVER_CACHE_DIR = "/hls/cache", "driver-cache"
#: The seal probe's program in the checkout, which the harness image's interpreter runs as a
#: script in isolated mode: it uses the standard library alone (``hls.egress.seal_probe``).
PROBE_SCRIPT = f"{CHECKOUT}/src/hls/egress/seal_probe.py"
#: Where the seal probe's directory for its report mounts in its container.
PROBE_OUT = "/hls/probe"
#: Seconds the seal probe may take from its start in its seal to its end; each of its checks has
#: the seal probe's own check time within that (``hls.egress.seal_probe.CHECK_SEC``). The probe
#: takes up to 6.08 s in a seal that forwards its inbound ports and lets ICMP out, where every wait
#: of it runs; this is five times that, rounded up to a multiple of 5 s.
SEAL_PROBE_SEC = 35.0
#: Seconds between reads of a seal's port file, and the longest wait for it.
PORT_POLL_SEC, PORT_WAIT_SEC = 0.5, INGRESS_WAIT_SEC
#: Bytes of the seal probe's answer the stage reads at most.
_PROBE_ANSWER_BYTES = 4096
#: Bytes the driver's report of its outputs is given on the node before the artifact starts,
#: which it then writes in place (``hls.store.manifest``). A record takes about 130 bytes and a
#: trace chunk holds 64 MiB, so this holds the report of a trace of hundreds of gigabytes.
OUTPUTS_REPORT_BYTES = 2**20
#: Where the driver writes the record of how its trace writer kept up
#: (``hls.drive.stream.TraceStreamRecord``) in its container, and that file's name in its role
#: directory.
STREAM_RECORD, STREAM_RECORD_FILE = "/hls/stream.json", "stream.json"
#: Bytes that record is given on the node before the artifact starts, which the driver then
#: writes in place: its failure of at most ``FAILURE_CHARS`` characters, each at most the 6 bytes
#: of a control character's escape in its JSON, 12,000 bytes, and its three figures and their
#: names, under 200.
STREAM_RECORD_BYTES = 2**14
#: Seconds between looks at the artifact while the stage waits for the driver.
DRIVE_POLL_SEC = 0.5
#: The artifact's network when the stage names none: no domain at all.
AIR_GAP = NetworkPolicy(domains=AccessList(mode="allow", entries=[]))
#: Images the stage extracts on the node, which live for this stage: its context removes them
#: when it closes. The artifact's and the harness image, each with ``KEPT_DISK_GIB`` of the
#: stage's disk.
PINNED_IMAGES = 2
#: Bytes in a GiB, for the driver's declared memory.
_GIB = 2**30
# Names of the package and provision stages' outputs this stage reads.
_PACKAGE, _PROVISIONED = "package", "provisioned"
#: The device nodes a driver's tool runner may be given (``HostPath`` with ``DEVICE``): the one a
#: rootless runtime mounts a squashfs image through. The sandboxes the driver starts can open
#: whatever it is given.
SANDBOX_DEVICES = ("/dev/fuse",)
#: Where the driver's own mounts go in its container, besides its data mounts; no sandbox store
#: may lie at, under or over one of them.
DRIVER_TARGETS = (CHECKOUT, TASK_REF, LOCAL, OUT, OUTPUTS_REPORT, STREAM_RECORD, DRIVER_CACHE)
#: The checkout's harbor in the driver's container, on the path of a driver whose tool runner
#: launches its sandboxes through harbor: the launch then runs the harbor the plan pins, whose
#: fields the runner sets, rather than a harbor its image may carry.
HARBOR_SRC = f"{CHECKOUT}/harbor/src"

log = logging.getLogger(__name__)


def _meets(a: str, b: str) -> bool:
    """Whether container paths *a* and *b* are one path or one lies under the other, so a bind on
    one would hide or replace the other."""
    pa, pb = PurePosixPath(a), PurePosixPath(b)
    return pa.is_relative_to(pb) or pb.is_relative_to(pa)


def sandbox_problems(paths: Sequence[HostPath], *, beside: Sequence[str] = ()) -> list[str]:
    """Why a role cannot be given the host paths *paths* of its tool runner beside mounts at the
    container paths *beside*: a path not in normal form, a device node not among
    ``SANDBOX_DEVICES`` or declared read-only, which no runtime binds, and a path at, under or over
    another of *paths* or one of *beside*, whose bind would hide the other."""
    problems = []
    for p in paths:
        if posixpath.normpath(p.path) != p.path or p.path.startswith("//"):
            problems.append(f"host path {p.path!r} is not in normal form")
        elif p.store == DEVICE and p.path not in SANDBOX_DEVICES:
            problems.append(f"device {p.path} is not one a driver is given ({', '.join(SANDBOX_DEVICES)})")
        elif p.store == DEVICE and p.read_only:
            problems.append(f"device {p.path} is declared read-only; a device is bound writable")
    for i, p in enumerate(paths):
        met = [q.path for q in paths[i + 1:] if _meets(p.path, q.path)] + [t for t in beside if _meets(p.path, t)]
        if met:
            problems.append(f"host path {p.path} meets {', '.join(met)}")
    return problems


def sandbox_binds(paths: Sequence[HostPath], stores: Mapping[str, Path], *,
                  beside: Sequence[str] = ()) -> tuple[tuple[Mount, ...], tuple[str, ...]]:
    """What a harness role is given whose tool runner starts containers of its own and declares
    *paths* (``ToolRunner.host_paths``), beside its other mounts at the container paths *beside*:
    each store's path bound from the directory the site names for that store (*stores*,
    ``Settings.sandbox_stores``) at the path the runner declares, read-only unless the runner
    writes it, and each device node as a device (``ContainerSpec.devices``). Every container the
    role starts can reach these binds. Raise ValueError naming each store the site does not name,
    and for any of ``sandbox_problems``."""
    problems = sandbox_problems(paths, beside=beside)
    unnamed = sorted({p.store for p in paths if p.store != DEVICE and p.store not in stores})
    if unnamed:
        problems.append(f"the site names no sandbox store {', '.join(unnamed)} (hls.toml's [sandbox_stores])")
    if problems:
        raise ValueError("; ".join(problems))
    mounts = tuple(Mount(stores[p.store], p.path, read_only=p.read_only) for p in paths if p.store != DEVICE)
    return mounts, tuple(p.path for p in paths if p.store == DEVICE)


def is_device(path: str) -> bool:
    """Whether *path* is a character device on this node, as the devices a driver is given are."""
    try:
        return stat.S_ISCHR(os.stat(path).st_mode)
    except OSError:
        return False


def driver_cache_env() -> dict[str, str]:
    """The variables that point every cache the driver's tools write at ``DRIVER_CACHE``, on the
    store's filesystem, so a download during the drive never needs the node's disk, which the
    server may have filled: the hub's root and its hub and datasets caches by name (tokenizers
    and models load through them), the XDG cache root, torch's hub, Inductor's and Triton's
    compile caches, and tiktoken's encodings, which would go to the temporary directory."""
    c = DRIVER_CACHE
    return {"HF_HOME": f"{c}/hf", "HF_HUB_CACHE": f"{c}/hf/hub", "HF_DATASETS_CACHE": f"{c}/hf/datasets",
            "XDG_CACHE_HOME": f"{c}/xdg", "TORCH_HOME": f"{c}/torch", "TORCHINDUCTOR_CACHE_DIR": f"{c}/inductor",
            "TRITON_CACHE_DIR": f"{c}/triton", "TIKTOKEN_CACHE_DIR": f"{c}/tiktoken"}


def artifact_env(*, runtime_env: dict[str, str], mounts_env: dict[str, str],
                 gpu_kind: str | None) -> dict[str, str]:
    """The artifact container's variables, later sources winning: the image's runtime
    variables, the mounts (which forbid hub downloads), and the build environment of a GPU of
    *gpu_kind* (``engine_build_env``): the engine cache's variables and the compute capability an
    extension built at serve time compiles for. The build environment comes last, so it is the
    one a measured baseline's engine cache key digests (``served_cache_key``)."""
    return {**runtime_env, **mounts_env, **engine_build_env(gpu_kind)}


class PackagedArtifact(BaseModel):
    """The submission image a package stage built."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Discriminator.
    kind: Literal["package"] = "package"
    #: The package stage of this instance whose manifest names the image.
    stage: StageName = "package"
    #: A pinned package manifest of another instance (through ``StagePlan.inputs``), read
    #: instead of this instance's; None to read this instance's ``stage``.
    manifest: ManifestKey | None = None


class ImageArtifact(BaseModel):
    """An artifact image the images stage built: a measured baseline or a prebuilt artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Discriminator.
    kind: Literal["image"] = "image"
    #: Its name in this instance's ``images.json``.
    image: Annotated[str, AfterValidator(check_segment)]
    #: The command that starts its server, before ``--host`` and ``--port``.
    entrypoint: list[str] = Field(min_length=1)
    #: Variables the image's server needs, below the stage's own.
    runtime_env: dict[str, str] = Field(default_factory=dict)


def _absolute_path(path: str) -> str:
    if not Path(path).is_absolute():
        raise ValueError(f"a published image's file is named by its absolute path, got {path!r}")
    return path


class PublishedArtifact(BaseModel):
    """A published submission image graded again by the digest of its file: read off this
    cluster's shared filesystem by the stage's extraction role, which refuses any other bytes,
    extracted as a contestant's tree and served by its runscript, as a package's image is."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Discriminator.
    kind: Literal["published"] = "published"
    #: The image file's sha256.
    sha256: Sha256Hex
    #: The file's absolute path on this cluster's shared filesystem.
    source: Annotated[str, AfterValidator(_absolute_path)]


def extracted_disk_gib(artifact: PackagedArtifact | ImageArtifact | PublishedArtifact) -> int:
    """Node-local disk, in GiB, the images a drive of *artifact* extracts hold: ``KEPT_DISK_GIB``
    for each of its ``PINNED_IMAGES``, and for a submission image, a package's or a published
    one, what a package adds to its base (``PACKAGE_ADDS_BYTES``) twice, since the extraction
    holds its copy of the file and its tree at once and each may carry those bytes."""
    adds = 0 if isinstance(artifact, ImageArtifact) else -(-PACKAGE_ADDS_BYTES // 2**30)
    return PINNED_IMAGES * KEPT_DISK_GIB + 2 * adds


class ServedParams(DataCopies):
    """How a stage serves an artifact: what it serves, from which images and data, behind which
    seal and in which shape, and the time it allows before the server's load counts. A drive
    (``DeployDriveParams``) and a measured baseline's warm stage
    (``hls.stages.baseline_warm.BaselineWarmParams``) serve it alike."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, read here for its contract and rebuilt by a drive's driver role.
    task: TaskRef
    #: The artifact to serve.
    artifact: PackagedArtifact | ImageArtifact | PublishedArtifact = Field(discriminator="kind")
    #: The images stage of this instance whose ``images.json`` names a measured artifact and the
    #: harness image.
    images_stage: StageName = "images"
    #: The provision stage of this instance whose ``provisioned.json`` names the mounts.
    provision_stage: StageName = "provision"
    #: The port the server listens on inside its seal.
    port: int = Field(gt=0, lt=65536)
    #: Arguments after ``--host`` and ``--port``.
    serve_command: list[str] = Field(default_factory=list)
    #: The artifact's runtime network policy (``NetworkPolicy`` JSON); None is the air gap.
    network: dict[str, Any] | None = None
    #: The server role's shape, a drive's artifact role's. A drive's server writes at most its
    #: ``disk_gib`` to its workdir and engine cache together.
    artifact_shape: ResourceShape
    #: Seconds the stage allows the server's runtime to start its command before its load counts
    #: (``hls.pipeline.pipelines.RUNTIME_START_SEC``), which the stage's time covers (its kind's
    #: ``check``); a drive's server starts only when what follows fits before the allocation's end
    #: (``_Drive.room_for_readiness``).
    runtime_start_sec: int = Field(ge=0)
    #: Seconds the stage allows before the server starts for its copies and extractions, and a
    #: drive's for its seal probe too (``hls.pipeline.pipelines.STAGING_SEC`` and
    #: ``SEAL_PROBE_SEC``), which the stage's time covers (its kind's ``check``).
    staging_sec: int = Field(ge=0)

    @model_validator(mode="after")
    def _network(self) -> Self:
        if self.network is not None:
            NetworkPolicy.model_validate(self.network)
        return self


def served_seal(params: ServedParams) -> SealPlan:
    """The seal *params*'s server runs behind: its runtime network policy, the air gap when it
    names none (``AIR_GAP``), no package index, and a host port forwarded to its port."""
    policy = NetworkPolicy.model_validate(params.network) if params.network is not None else AIR_GAP
    return SealPlan(policy=SealPolicy(network=policy), index=False, ingress_port=params.port)


def served_cache_key(params: ServedParams, image_sha256: str, weights: WeightsRef | None) -> str:
    """The key of the engine cache entry of the measured server *params* serves from the image whose
    file has the sha256 *image_sha256*, loading *weights* (None: none were provisioned)
    (``engine_cache_key``): its config is how it starts, the catalog's entrypoint and variables
    and the serve command after them, with its GPU count and its weights by repo and revision; its
    GPU kind is its shape's and its network its seal's (``served_seal``). A drive of the baseline
    and its warm stage compute it from the same parameters. Raise ValueError for a server that is no
    measured image's, or that runs on no GPU kind."""
    artifact, shape = params.artifact, params.artifact_shape
    if not isinstance(artifact, ImageArtifact) or shape.gpu_kind is None:
        raise ValueError("only a measured image served on GPUs has an engine cache entry")
    config = json.dumps({"entrypoint": artifact.entrypoint, "runtime_env": artifact.runtime_env,
                         "serve_command": params.serve_command, "gpus": shape.gpus,
                         "weights": None if weights is None else {"repo": weights.repo, "revision": weights.revision}},
                        sort_keys=True)
    return engine_cache_key(image_sha256, shape.gpu_kind, config, seal=served_seal(params))


class DeployDriveParams(ServedParams):
    """What the stage serves and drives, and how long it may take."""

    #: The name in this instance's ``images.json`` of the harness image the driver and the
    #: seal probe run on.
    harness_image: Annotated[str, AfterValidator(check_segment)]
    #: The driver role's shape; GPUs only when the task's driver assets ask for them.
    driver_shape: ResourceShape
    #: True runs the driver on the artifact's CPUs (``RoleSpec.overlap``): the stage holds the artifact's CPUs, and the driver's
    #: memory beside the artifact's. Such a driver takes no GPU and no more CPUs than the
    #: artifact has.
    driver_overlaps: bool = False
    #: True oversubscribes the overlapping driver's memory on the artifact's
    #: (``StageContext.start_container``'s *share_memory_of*): the stage holds the artifact's
    #: memory and not the driver's beside it, the driver keeps its own step limit and its memory
    #: guard, and the job's limit bounds the two together. Left out of the dump when false.
    driver_memory_shared: bool = Field(default=False, exclude_if=lambda shared: not shared)
    #: Seconds each readiness gate may take, health and then inference
    #: (``hls.pipeline.pipelines.READINESS_TIMEOUT_SEC``): the first from the server's command's
    #: start, the second from the first's end.
    readiness_timeout_sec: int = Field(gt=0)
    #: The drive deadline's grace the driver is built with (``hls.pipeline.pipelines.REQUEST_TIMEOUT_SEC``):
    #: every request must complete within this many seconds after the drive's last scheduled arrival.
    request_timeout_sec: float = Field(gt=0)
    #: The drive's wall in seconds from the drive's own start: the task's own (an agent seam's
    #: ``drive_wall_sec()``), or ``hls.pipeline.pipelines.OPEN_DRIVE_SEC`` for a task that declares
    #: none. A drive gets all of it or does not start.
    drive_timeout_sec: float = Field(gt=0)
    #: The split the driver drives.
    split: Literal["dev", "test"] = "test"
    #: Seconds of the stage's time kept after the drive's wall, for finishing, copying and
    #: recording the trace and stopping the artifact: the planner's
    #: ``hls.pipeline.pipelines.DRIVE_MARGIN_SEC`` and ``TRACE_COPY_SEC``.
    wall_margin_sec: int = Field(ge=0)
    #: The most bytes one reply from the server may carry (``hls.drive.bounds.REPLY_MAX_BYTES``
    #: in every evaluation); a longer one fails as the contestant's request.
    reply_max_bytes: int = Field(gt=0)
    #: The cap on the JSON of the drive's trace's events, the one the task declares
    #: (``DriveBounds.trace_cap_bytes``); every reply that would carry the trace past it fails as
    #: the contestant's request.
    trace_cap_bytes: int = Field(gt=0)
    #: The most bytes of requests the drive's trace holds beside the replies, the one the task
    #: declares (``DriveBounds.trace_request_bytes``). The stage sizes the trace's room by it and
    #: the cap (``hls.drive.bounds.trace_event_bytes``).
    trace_request_bytes: int = Field(gt=0)
    #: True for a drive of the task's own measured baseline, the reference's server: the driver
    #: counts none of its replies and has no memory guard, and the server starts from the engine
    #: cache entry ``engine_cache_owner`` left.
    measured_baseline: StrictBool
    #: For a drive of the measured baseline, the warm stage of this instance that owns its server's
    #: engine cache entry (``hls.stages.baseline_warm``), whose ``engine-cache.json`` names the entry
    #: the server's cache is a checked copy of. None for every other
    #: artifact, whose server starts with an empty engine cache.
    engine_cache_owner: StageName | None
    #: What the task's tool runner needs inside the driver's container (``ToolRunner.host_paths``),
    #: which the stage binds from the site's sandbox stores (``sandbox_binds``). Empty for a
    #: runner that needs nothing, and then left out of the plan.
    sandbox_paths: tuple[HostPath, ...] = Field(default=(), exclude_if=lambda paths: not paths)

    @model_validator(mode="after")
    def _checked(self) -> Self:
        problems = sandbox_problems(self.sandbox_paths, beside=DRIVER_TARGETS)
        if problems:
            raise ValueError("the driver's sandbox paths: " + "; ".join(problems))
        if self.measured_baseline != (self.engine_cache_owner is not None):
            raise ValueError("a drive of the measured baseline, and only one, starts from its warm stage's engine "
                             "cache entry (engine_cache_owner)")
        if self.measured_baseline and (not isinstance(self.artifact, ImageArtifact)
                                       or self.artifact_shape.gpu_kind is None):
            raise ValueError("the measured baseline is a measured image served on GPUs")
        if self.driver_overlaps and self.driver_shape.gpus:
            raise ValueError("an overlapping driver takes no GPU: its step would share the artifact's")
        if self.driver_overlaps and self.driver_shape.cpus > self.artifact_shape.cpus:
            raise ValueError(f"an overlapping driver runs on the artifact's {self.artifact_shape.cpus} CPUs, "
                             f"so it cannot take {self.driver_shape.cpus}")
        if self.driver_memory_shared and not self.driver_overlaps:
            raise ValueError("only an overlapping driver shares the artifact's memory")
        return self


@dataclass(frozen=True)
class _DriverFiles:
    """What the driver's container needs on the node, made before the artifact starts
    (``_Drive.prepare_driver``)."""

    #: Its workdir, with the directories the runtime makes there.
    workdir: Path
    #: Its mounts.
    mounts: tuple[Mount, ...]
    #: Its variables, whose env file is written.
    env: dict[str, str]
    #: The device nodes it is given (``ContainerSpec.devices``).
    devices: tuple[str, ...] = ()


def _own(outcome: RoleOutcome) -> RoleOutcome | None:
    """*outcome* when the role ended on its own; None when the stage stopped it as planned."""
    return None if outcome.ending == "planned_stop" else outcome


class ServedAttempt:
    """One attempt of a stage that serves an artifact (``ServedParams``), as a drive and a measured
    baseline's warm stage do: its inputs read, its image pinned and its data copied, and its server
    started sealed, with the engine cache it was given, on the host port its seal forwards."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None or ctx.runtime is None or ctx.images is None or ctx.cache is None:
            raise RuntimeError(f"stage {ctx.stage.name} serves an artifact, so it needs node-local scratch, a "
                               "container runtime, an image runtime and the cache")
        self.ctx = ctx
        self.params: ServedParams = ctx.params
        #: Roles started and not yet waited for or stopped, by name.
        self.running: dict[str, RoleHandle] = {}
        #: The per-uid count limits the server's seal and its probe's write, the ones a graded
        #: server is held to: its build's recorded figures, or the declaration's for an artifact no
        #: build stated them for (``StageContext.grading_ucounts``), once read.
        self.held: dict[str, int] | None = None

    # inputs

    def _manifest(self, key: str) -> Manifest:
        manifest = self.ctx.inputs.get(key)
        if manifest is None:
            raise StageFault.infra(f"the stage was planned without input manifest {key}", retryable=False)
        return manifest

    def _record(self, manifest: Manifest, output: str) -> bytes:
        ref = manifest.outputs.get(output)
        if ref is None:
            raise StageFault.infra(f"{manifest.instance} stage {manifest.stage} published no {output}", retryable=False)
        data = self.ctx.store.read_bytes(ref.key)
        if sha256_bytes(data) != ref.sha256:
            raise StageFault.infra(f"{ref.key} does not match its manifest", retryable=False)
        return data

    def provisioned(self) -> Provisioned:
        manifest = self._manifest(self.ctx.plan.layout.manifest(self.params.provision_stage))
        return Provisioned.model_validate_json(self._record(manifest, _PROVISIONED))

    def image_entries(self) -> dict[str, tuple[str, str]]:
        """Each image of this instance's ``images.json`` by name: its cache key and entry
        digest."""
        manifest = self._manifest(self.ctx.plan.layout.manifest(self.params.images_stage))
        data = self._record(manifest, IMAGES_OUTPUT)
        try:
            return {name: (check_segment(entry["key"]), entry["entry_sha256"])
                    for name, entry in json.loads(data).items()}
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            raise StageFault.infra(f"{manifest.outputs[IMAGES_OUTPUT].key} does not name each image's key and entry "
                                   "digest", retryable=False) from e

    def catalog_image(self, entries: dict[str, tuple[str, str]], name: str) -> PinnedImage:
        """Image *name* of ``images.json``, extracted on the node and pinned to the entry digest
        recorded there."""
        if name not in entries:
            raise StageFault.infra(f"this instance's images stage published no image {name}", retryable=False)
        key, expect = entries[name]
        try:
            return self.ctx.node_image(key, expect)
        except ValueError as e:
            raise StageFault.infra(f"images.json records no digest for image {name}: {e}", retryable=False) from e

    def image_sha256(self, entries: dict[str, tuple[str, str]], name: str) -> str:
        """The sha256 of image *name*'s file, as its entry of ``images.json`` records it: read off
        the entry's manifest, pinned to the entry digest recorded there, with nothing extracted."""
        if name not in entries:
            raise StageFault.infra(f"this instance's images stage published no image {name}", retryable=False)
        key, expect = entries[name]
        assert self.ctx.images is not None
        try:
            return self.ctx.entry_file("images", key, expect, self.ctx.images.image_file)[1]
        except ValueError as e:
            raise StageFault.infra(f"images.json records no digest for image {name}: {e}", retryable=False) from e

    def data(self, role: str, provisioned: Provisioned) -> tuple[list[Mount], dict[str, str]]:
        """The data mounts of the mount wall's role *role* (``hls.provision.mounts``) with their
        variables, each cache entry copied to the node once (``materialize``), with the dev
        reference the plan pins beside each tenant's dev rows (``hls.provision.mounts.dev_view``),
        its view in the role's own directory. The copies must be
        those the plan declared (``data_copies``), or the server, which the node's start check
        passed for the allocation's disk, could find its scratch short: a copy the plan did not
        declare, or one that takes more (an entry published again since the plan), is infra not
        retried, before it is copied (``data_mounts``)."""
        try:
            files = dev_files(self.ctx)
            return data_mounts(role, provisioned, self.ctx.materialize, declared=self.params.data_copies,
                               dev_files=files, views=self.ctx.role_dir(role) / "dev" if files else None)
        except ValueError as e:
            raise StageFault.infra(f"the {role} role's mounts: {e}", retryable=False) from e
        except OSError as e:
            raise StageFault.infra(f"the {role} role's view of the dev rows: {e}", retryable=True) from e

    # roles

    def _timed(self, shape: ResourceShape) -> ResourceShape:
        """*shape* for the stage's own time. A role's shape declares only the stage's default
        time, which a site's ``[resources]`` table may change."""
        return shape.model_copy(update={"time_sec": self.ctx.stage.shape.time_sec})

    def _start(self, name: str, spec: ContainerSpec, shape: ResourceShape,
               owner: Literal["harness", "contestant"], env: dict[str, str], *,
               overlap: bool = False, one_numa_node: bool = False,
               share_memory_of: RoleHandle | None = None) -> RoleHandle:
        # A seal that serves, the artifact's and its probe's, writes the limits a graded server is
        # held to.
        held = self.held if spec.seal is not None and spec.seal.ingress_port is not None else None
        handle = self.ctx.start_container(name, spec, self._timed(shape), owner=owner, env=env,
                                          overlap=overlap, one_numa_node=one_numa_node, seal_ucounts=held,
                                          share_memory_of=share_memory_of)
        self.running[name] = handle
        return handle

    def wait(self, name: str) -> RoleOutcome:
        """How role *name* ended, once it has."""
        self.running.pop(name, None)
        [outcome] = self.ctx.wait_roles(name)
        return outcome

    def stop(self, name: str) -> RoleOutcome:
        """How role *name* ended, once the stage stopped it as planned (``StageContext.stop_role``)."""
        self.running.pop(name, None)
        return self.ctx.stop_role(name)

    def host_port(self, name: str, handle: RoleHandle) -> int | None:
        """The host port the seal of sealed role *name* forwards; None when the role ended, or
        the stage was told to stop, first."""
        ctx = self.ctx
        files = ctx.seal_files(name)
        assert files is not None
        give_up = min(ctx.clock() + PORT_WAIT_SEC, ctx.deadline)
        while (port := ingress_port(files)) is None:
            if handle.poll() is not None or ctx.stopping:
                return None
            if ctx.clock() >= give_up:
                raise StageFault.infra(f"the {name} role's seal wrote no host port in {PORT_WAIT_SEC:.0f}s",
                                       retryable=True, role=name)
            time.sleep(PORT_POLL_SEC)
        return port

    def start_server(self, name: str, image: PinnedImage, entrypoint: list[str], runtime_env: dict[str, str],
                     data: tuple[list[Mount], dict[str, str]], cache: Path, *,
                     owner: Literal["harness", "contestant"], one_numa_node: bool = False) -> RoleHandle:
        """Start the server as role *name*, *owner*'s, in the artifact's shape: *image* running
        *entrypoint*, then ``--host 0.0.0.0 --port <port>`` and the serve command, in a fresh
        workdir in the role's directory, sealed (``served_seal``) and holding no capability, with
        *data* mounted and the engine cache *cache* writable at ``ENGINE_CACHE``, its variables the
        image's *runtime_env* below the stage's own (``artifact_env``). *one_numa_node* refuses a
        step whose CPUs Slurm split over NUMA nodes (``RoleSpec.one_numa_node``)."""
        p = self.params
        workdir = self.ctx.role_dir(name) / "wd"
        workdir.mkdir()
        mounts, mounts_env = data
        spec = ContainerSpec(
            image=image, command=(*entrypoint, "--host", "0.0.0.0", "--port", str(p.port), *p.serve_command),
            workdir=workdir, mounts=(*mounts, Mount(cache, ENGINE_CACHE, read_only=False)),
            gpus=p.artifact_shape.gpus > 0, seal=served_seal(p), contestant_payload=True)
        env = artifact_env(runtime_env=runtime_env, mounts_env=mounts_env, gpu_kind=p.artifact_shape.gpu_kind)
        return self._start(name, spec, p.artifact_shape, owner, env, one_numa_node=one_numa_node)

    def wait_ready(self, name: str, handle: RoleHandle, url: str, contract: Any, *, budget_sec: int, shared: bool,
                   margin_sec: int) -> Literal["ready", "ended", "not_ready"]:
        """Wait for the server of role *name* to start its command, then pass both readiness gates
        at *url*; ``ended`` when its role ended first, or the stage was told to stop, ``not_ready``
        when a gate's budget ran out while it ran. Each gate may take *budget_sec*, the first from
        the command's start (its start marker, ``StageContext.wait_command_started``) and the
        second from the first's end; with *shared*, both gates together take *budget_sec* from the
        command's start. The time the runtime takes to start the container is the harness's, and
        only the stage's deadline bounds it, so a command that has not started by then is infra. A
        gate checks the role before each probe; when one runs out, the role is looked at once more,
        so a server that died during the gate's last probe ended rather than was not ready. The
        stage's deadline is held for the budget and *margin_sec* after it
        (``StageContext.hold_until``), first from now and again from the command's start, so copies
        and extractions that ran long do not cut the wait the allocation still has room for; a gate
        the allocation's end cuts all the same is infra, retried."""
        ctx = self.ctx
        total = budget_sec if shared else 2 * budget_sec
        ctx.hold_until(ctx.clock() + total + margin_sec)

        def alive() -> bool:
            return handle.poll() is None and not ctx.stopping

        began = ctx.wait_command_started(name)
        if began is None:
            if alive():
                raise StageFault.infra(f"the {name}'s container did not start its command before the stage's deadline "
                                       "(no start marker on its stderr)", retryable=True)
            return "ended"
        ctx.hold_until(began + total + margin_sec)
        gates = (lambda t: wait_healthy(url, timeout_sec=t, alive=alive),
                 lambda t: wait_inference_ready(url, contract, timeout_sec=t, alive=alive))
        start = began
        for gate in gates:
            counted_from = began if shared else start
            until = min(counted_from + budget_sec, ctx.deadline - margin_sec)
            try:
                asyncio.run(gate(max(until - ctx.clock(), 0.0)))
            except NotServing:
                return "ended"
            except NotReady as e:
                if not alive():
                    return "ended"
                if until < counted_from + budget_sec:
                    cut = max(until - counted_from, 0)
                    what = "the readiness gates" if shared else "a readiness gate"
                    raise StageFault.infra(f"the stage's time cut {what} to {cut:.0f}s of {'their' if shared else 'its'} "
                                           f"{budget_sec}s budget", retryable=True) from e
                return "not_ready"
            start = ctx.clock()
        return "ready"

    def no_room(self, reason: str, *, served: RoleOutcome | None = None) -> StageFault:
        """The fault of a server the allocation's end left no room for: infra, retried, with each
        role's wait for its step to start and the end it keeps to as evidence. *served* is how the
        server ended, once stopped, when it was started: its own fault decides when it is the
        contestant's, and an infra fault of its own joins the reason and the evidence."""
        ctx = self.ctx
        own = stage_fault([served]) if served is not None else None
        if own is not None and own.fault_class == "contestant":
            return StageFault(own)
        evidence = dict(own.evidence) if own is not None else {}
        evidence |= {f"launch_wait.{name}": f"{sec:.1f}" for name, sec in ctx.launch_waits().items()}
        evidence["deadline_cap"] = str(ctx.deadline_cap)
        if own is not None:
            reason = f"{reason}; {own.reason}"
        return StageFault.infra(reason, retryable=own is None or own.retryable, **evidence)

    def stop_all(self) -> None:
        """Stop every role still running."""
        for name in list(self.running):
            self.stop(name)


class _Drive(ServedAttempt):
    """One deploy_drive attempt."""

    def __init__(self, ctx: StageContext) -> None:
        super().__init__(ctx)
        self.params: DeployDriveParams = ctx.params
        #: The artifact step's process ceiling, once computed.
        self.ceiling: ProcessCeiling | None = None
        #: The artifact's writable directories on the node, once it started.
        self.written: tuple[Path, ...] = ()
        #: The trace's room on the node (``LOCAL_DIR``), once the stage began to make it.
        self.room: Path | None = None
        #: The engine cache entry the server started from, once its cache was made
        #: (``engine_cache``); None for one that started empty.
        self.cache_record: EngineCacheRecord | None = None

    # inputs

    def package(self, artifact: PackagedArtifact) -> tuple[Manifest, PackageOutcome]:
        """The package manifest the drive serves and what a grading reads of its ``package.json``
        (``PackageOutcome``), checked against the manifest: a package another package stage
        published, this instance's or another's, is served by those fields. Raise
        infra, not retried, for a record that is not its manifest's or cannot be read."""
        manifest = self._manifest(artifact.manifest or self.ctx.plan.layout.manifest(artifact.stage))
        data = self._record(manifest, _PACKAGE)
        try:
            return manifest, PackageOutcome.model_validate_json(data)
        except ValidationError as e:
            raise StageFault.infra(f"{manifest.outputs[_PACKAGE].key} is not a package record: {validation_errors(e)}",
                                   retryable=False) from None

    @staticmethod
    def submission(manifest: Manifest, result: PackageOutcome) -> OutputRef:
        """The package manifest's record of the submission image, which must be the image and
        digest ``package.json`` names."""
        ref = manifest.outputs.get(SUBMISSION_OUTPUT)
        if ref is None or ref.key != result.image or ref.sha256 != result.sif_sha256:
            raise StageFault.infra("the package manifest's submission image is not the one package.json names",
                                   retryable=False)
        return ref

    def engine_cache(self, image: PinnedImage, provisioned: Provisioned) -> tuple[Path, EngineCacheRecord | None]:
        """The server's engine cache, made before any role starts, and its record.
        A measured baseline's is a writable copy, made for this attempt
        with each file checked, of the entry its warm stage left (``engine_cache_owner``), under the
        key of the baseline served from *image* with *provisioned*'s weights
        (``served_cache_key``), the record naming the key and the entry's digest. Any other
        artifact's starts empty, with no record, so its own build and JIT time count against its
        readiness budget. Raise infra, not retried, for a warm stage that left no entry, and as
        ``StageContext.seed`` does for its entry."""
        p, ctx = self.params, self.ctx
        if p.engine_cache_owner is None:
            cache, _ = engine_cache.prepare(ctx, ARTIFACT, None, pinned=None)
            return cache, None
        key = served_cache_key(p, image.file_sha256, provisioned.weights)
        pinned = engine_cache.owner_entry(ctx, p.engine_cache_owner, key)
        if pinned is None:
            raise StageFault.infra(f"the measured baseline's warm stage {p.engine_cache_owner} left no engine cache "
                                   "entry to start its server from", retryable=False)
        cache, seeded = engine_cache.prepare(ctx, ARTIFACT, key, pinned=pinned)
        return cache, EngineCacheRecord(key=key, seeded_from=seeded)

    def mounts(self, provisioned: Provisioned) -> tuple[tuple[list[Mount], dict[str, str]],
                                                         tuple[list[Mount], dict[str, str]]]:
        """The artifact's and the driver's data mounts with their variables (``data``)."""
        return self.data(ARTIFACT, provisioned), self.data(DRIVER, provisioned)

    # roles

    def probe_seal(self, harness: PinnedImage, seal: SealPlan, policy: NetworkPolicy) -> SealProbeRecord:
        """Run the seal probe (``hls.egress.seal_probe``) sealed as the artifact will be, send its
        ingress check the request it answers, knock on its inbound ports from every address of
        the node, and only then send it a second request, which it answers once it has closed
        them, so every knock found them held; raise infra, retryable, unless every check passed
        within ``SEAL_PROBE_SEC`` of its start in its seal. The refusal
        check passes only when the seal's interceptor also recorded that its policy refused the
        host (``policy_refusals``): a connection closed unanswered is no proof, since an
        interceptor that lets every host through closes it too when the host does not resolve.
        The caps check passes when the probe, which holds what the artifact will hold, holds no
        capability (``ContainerSpec.contestant_payload``). The ucounts check passes when the probe
        read, inside its seal, each limit on the uid's per-namespace counts that the seal was given
        (``StageContext.grading_ucounts``) and that the site's substrate holds a sealed role to
        (``Substrate.served_ucounts``; a Docker seal holds it to none), as given, and could not
        raise one, so the limits the artifact runs under are verified in place before it starts.
        The ICMP check passes when
        no echo the probe tried to send left the seal. The inbound check
        passes only when every knock found its port closed (``Knock.refused``) and the probe saw
        nothing reach either port; a knock that found one open names the seal when the probe saw
        it arrive, the node when it did not (``knock_failures``). The ``drive.seal_probe`` event
        records each check either way."""
        ctx, p = self.ctx, self.params
        role = ctx.role_dir(SEAL_PROBE)
        out, workdir = role / "out", role / "wd"
        out.mkdir()
        workdir.mkdir()
        host = refused_host(policy)
        try:
            tcp_in, udp_in = inbound_ports(avoid=namespace_ports(seal))
        except OSError as e:
            raise StageFault.infra(f"the seal probe has no inbound ports: {e}", retryable=True, role=SEAL_PROBE) from e
        # The limits the seal writes in its user namespace, which the probe reads back from inside it:
        # the value of the seal's own variable (``StageContext.start_container``), of the counts the
        # site's substrate holds a sealed role to.
        assert self.held is not None
        enforced = substrate(ctx.plan.settings.site).served_ucounts
        limits = ucounts_env({k: v for k, v in self.held.items() if k in enforced})[SEAL_UCOUNTS_VAR]
        command = ["python", "-I", PROBE_SCRIPT, "served", "--port", str(p.port), "--inbound-tcp", str(tcp_in),
                   "--inbound-udp", str(udp_in), "--out", f"{PROBE_OUT}/report.json", "--check-sec",
                   str(CHECK_SEC), "--ucounts", limits]
        if host is not None:
            command += ["--refused-host", host, "--dial", PROBE_DIAL]
        mounts = (Mount(ctx.plan.code.checkout, CHECKOUT), Mount(out, PROBE_OUT, read_only=False))
        spec = ContainerSpec(image=harness, command=tuple(command), workdir=workdir, mounts=mounts, seal=seal,
                             contestant_payload=True)
        # It runs alone, before the artifact, so the driver's shape without a GPU fits.
        shape = p.driver_shape.model_copy(update={"gpus": 0, "gpu_kind": None})
        handle = self._start(SEAL_PROBE, spec, shape, "harness", {})
        port = self.host_port(SEAL_PROBE, handle)
        started = ctx.clock()
        give_up = min(started + SEAL_PROBE_SEC, ctx.deadline)
        answered = port is not None and self._knock(handle, port, give_up)
        addresses = node_addresses()
        knocks = knock_inbound(addresses, tcp_in, udp_in) if answered else ()
        closed = answered and self._knock(handle, port, give_up)
        if not ctx.stopping and handle.wait(max(0.0, give_up - ctx.clock())) is None:
            self.stop(SEAL_PROBE)
            raise StageFault.infra(f"the seal probe took more than {SEAL_PROBE_SEC:.0f}s", retryable=True,
                                   role=SEAL_PROBE)
        outcome = self.wait(SEAL_PROBE)
        sec = round(ctx.clock() - started, 3)
        report = read_report(out / "report.json")
        refusal, ingress, icmp, inbound, ucounts, caps = (
            (None, None, None, None, None, None) if report is None
            else (report.refusal, report.ingress, report.icmp, report.inbound, report.ucounts, report.caps))
        files = ctx.seal_files(SEAL_PROBE)
        assert files is not None
        by_policy = None if host is None else host in (policy_refusals(files.status) or ())
        failures = []
        fault = stage_fault([outcome])
        if fault is not None:
            failures.append(fault.reason)
        if report is None:
            failures.append("it wrote no report of its checks")
        else:
            if host is not None and refusal is None:
                failures.append("it made no refusal check")
            failures += [f"{name}: {check.detail}" for name, check in (
                ("caps", caps), ("ucounts", ucounts), ("refusal", refusal), ("ingress", ingress), ("icmp", icmp),
                ("inbound", inbound)) if check is not None and not check.passed]
        if by_policy is False:
            failures.append(f"the seal recorded no refusal of {host} by its policy")
        if not answered:
            failures.append("the stage's request through the seal's host port was not answered with its path")
        elif not closed:
            failures.append("the stage's second request through the seal's host port was not answered "
                            "with its path")
        failures += knock_failures(addresses, knocks, () if report is None else report.reached)

        def detail(check: Check | None) -> str | None:
            return None if check is None else check.detail

        def took(check: Check | None) -> float | None:
            return None if check is None else check.sec

        ctx.events.emit("drive.seal_probe", passed=not failures, refused_host=host, refused_by_policy=by_policy,
                        refusal=detail(refusal), refusal_sec=took(refusal), ingress=detail(ingress),
                        ingress_sec=took(ingress), answered=answered, icmp=detail(icmp), icmp_sec=took(icmp),
                        inbound=detail(inbound), inbound_sec=took(inbound), ucounts=detail(ucounts),
                        ucounts_sec=took(ucounts), caps=detail(caps), caps_sec=took(caps),
                        inbound_knocks=[{"address": k.address, "proto": k.proto, "port": k.port,
                                         "outcome": k.outcome} for k in knocks],
                        sec=sec, ending=outcome.ending, exit_code=outcome.exit_code)
        if failures or report is None:
            raise StageFault.infra("the seal probe failed: " + "; ".join(failures), retryable=True, role=SEAL_PROBE,
                                   ending=outcome.ending, exit_code=str(outcome.exit_code))
        return SealProbeRecord(refused_host=host, refusal_sec=took(refusal), ingress_sec=report.ingress.sec,
                               icmp_sec=report.icmp.sec, ucounts_sec=report.ucounts.sec, caps_sec=report.caps.sec,
                               inbound_tcp=tcp_in, inbound_udp=udp_in, inbound_addresses=addresses,
                               inbound_sec=report.inbound.sec, sec=sec)

    def _knock(self, handle: RoleHandle, port: int, give_up: float) -> bool:
        """Send the seal probe a request through the seal's host port *port* until it answers or
        *give_up*; whether it answered with the request's own path, a word only this request
        holds."""
        ctx = self.ctx
        word = uuid.uuid4().hex
        with httpx.Client(trust_env=False) as client:
            while ctx.clock() < give_up and handle.poll() is None and not ctx.stopping:
                try:
                    with client.stream("GET", f"http://127.0.0.1:{port}/{word}",
                                       timeout=max(0.1, min(CHECK_SEC, give_up - ctx.clock()))) as reply:
                        body = b""
                        for chunk in reply.iter_bytes():
                            body += chunk
                            if len(body) >= _PROBE_ANSWER_BYTES:
                                break
                        return reply.status_code == 200 and body == word.encode()
                except httpx.HTTPError:
                    time.sleep(PORT_POLL_SEC)
        return False

    def start_artifact(self, image: PinnedImage, entrypoint: list[str], runtime_env: dict[str, str],
                       data: tuple[list[Mount], dict[str, str]], cache: Path) -> RoleHandle:
        """Start the artifact's server as the contestant's role (``start_server``) with the engine
        cache *cache* (``engine_cache``): its workdir and that cache are what it writes."""
        self.written = (self.ctx.role_dir(ARTIFACT) / "wd", cache)
        # On a site that asked for the server's CPUs as one task on one socket, a step Slurm still
        # split over two sockets is refused (``site.evaluation_socket_tasks``). The context gives
        # it the stage's process ceiling (``StageContext.process_ceiling``).
        return self.start_server(ARTIFACT, image, entrypoint, runtime_env, data, cache, owner="contestant",
                                 one_numa_node=self.ctx.plan.settings.site.evaluation_socket_tasks)

    def room_for_readiness(self) -> None:
        """Raise ``no_room`` before the server starts when the time allowed for its runtime's
        start (``runtime_start_sec``), its two readiness budgets, the drive's whole wall and the
        margin after it do not fit before the allocation's end (``StageContext.deadline_cap``):
        the stage would spend the server's readiness on a drive it could not drive whole."""
        p, ctx = self.params, self.ctx
        need = p.runtime_start_sec + 2 * p.readiness_timeout_sec + p.drive_timeout_sec + p.wall_margin_sec
        if ctx.deadline_cap is not None and ctx.clock() + need > ctx.deadline_cap:
            raise self.no_room(f"the {p.runtime_start_sec}s allowed for the server's runtime to start its command, its "
                               f"two {p.readiness_timeout_sec}s readiness budgets, the drive's full wall of "
                               f"{p.drive_timeout_sec:.0f}s and the {p.wall_margin_sec}s margin after it do not fit "
                               "before the allocation's end, so the server was not started")

    def hold_drive(self) -> bool:
        """Whether the stage's deadline holds the drive's whole wall (``drive_timeout_sec``) and the
        margin after it from now, grown for them up to the allocation's end
        (``StageContext.hold_until``)."""
        p = self.params
        return self.ctx.hold_until(self.ctx.clock() + p.drive_timeout_sec + p.wall_margin_sec)

    def prepare_driver(self, data: tuple[list[Mount], dict[str, str]]) -> _DriverFiles:
        """Make every file the driver's container needs on the node, before the artifact starts:
        a server that fills the node's disk then leaves the driver nothing to make there. Those
        are its role directory with its step's ``tmp``, the task reference, its workdir with the
        directories the runtime makes there, its env file (``prepare_container_env``), its
        report of the outputs it wrote and its record of how its trace writer kept up, given
        ``OUTPUTS_REPORT_BYTES`` and ``STREAM_RECORD_BYTES`` of the disk, which it writes in place,
        and the trace's room (``LOCAL_DIR``, ``hls.trace.store.make_room``), at
        the most a trace within the task's bounds takes, every block given now, where the drive's
        trace is written in place as the drive goes. The trace is published to the attempt's
        outputs directory and the caches of the driver's tools go to the attempt's
        ``DRIVER_CACHE_DIR`` (``driver_cache_env``), both on the store's filesystem."""
        ctx, p = self.ctx, self.params
        assert ctx.runtime is not None
        role = ctx.role_dir(DRIVER)
        ref = role / "task-ref.json"
        ref.write_text(p.task.model_dump_json())
        workdir = role / "wd"
        for part in ctx.runtime.workdir_dirs:
            (workdir / part).mkdir(parents=True)
        local = role / LOCAL_DIR
        local.mkdir()
        self.room = local
        event_bytes = trace_event_bytes(p.trace_cap_bytes, p.trace_request_bytes)
        try:
            make_room(local, event_bytes=event_bytes)
        except OSError as e:
            # Before any server: the node's disk, not the contestant. ``run`` removes what was made.
            size = sum(room_bytes(event_bytes).values())
            raise StageFault.infra(f"the trace's room of {size:,} bytes could not be made on the node's disk: "
                                   f"{type(e).__name__}: {e}", retryable=True, room=str(local),
                                   room_bytes=str(size)) from e
        report, stream = ctx.outputs_report(DRIVER), role / STREAM_RECORD_FILE
        for path, size in ((report, OUTPUTS_REPORT_BYTES), (stream, STREAM_RECORD_BYTES)):
            # Filled rather than zeroed: a filesystem that compresses would keep zeros as a hole.
            with open(path, "wb") as f:
                f.write(b" " * size)
                f.flush()
                os.fsync(f.fileno())
        outputs = ctx.outputs_dir
        outputs.mkdir(parents=True, exist_ok=True)
        caches = ctx.attempt_dir / DRIVER_CACHE_DIR
        caches.mkdir(parents=True, exist_ok=True)
        mounts = (*data[0], Mount(ctx.plan.code.checkout, CHECKOUT), Mount(ref, TASK_REF),
                  Mount(local, LOCAL, read_only=False), Mount(outputs, OUT, read_only=False),
                  Mount(report, OUTPUTS_REPORT, read_only=False), Mount(stream, STREAM_RECORD, read_only=False),
                  Mount(caches, DRIVER_CACHE, read_only=False))
        stores, devices = self.sandbox(mounts)
        path = f"{CHECKOUT}/src:{CHECKOUT}" + (f":{HARBOR_SRC}" if p.sandbox_paths else "")
        # The driver runs with HF_HUB_OFFLINE=0: a tool's encoder may still load from the hub.
        env = {**data[1], **driver_cache_env(), "PYTHONPATH": path, "HF_HUB_OFFLINE": "0"}
        ctx.prepare_container_env(DRIVER, env)
        return _DriverFiles(workdir=workdir, mounts=(*mounts, *stores), env=env, devices=devices)

    def sandbox(self, beside: Sequence[Mount]) -> tuple[tuple[Mount, ...], tuple[str, ...]]:
        """The driver's binds of the site's sandbox stores and its device nodes, beside its mounts
        *beside* (``sandbox_binds``); none for a task whose tool runner declares no host paths.
        Each is checked on the node before any server starts: a store that, with its links followed
        here, is not apart from the store, the secrets file's directory, the checkout or another
        store (``sandbox_store_problems``), or that is no directory here, is infra, not retried,
        since every attempt reads the same settings, and a device the node lacks is infra,
        retried."""
        paths = self.params.sandbox_paths
        if not paths:
            return (), ()
        plan = self.ctx.plan
        try:
            stores, devices = sandbox_binds(paths, plan.settings.sandbox_stores, beside=[m.target for m in beside])
        except ValueError as e:
            raise StageFault.infra(f"the driver's sandbox stores: {e}", retryable=False) from e
        # Again on this node, with its links followed: a store must stand apart from the store, the
        # secrets file's directory, the checkout and the other stores here too.
        apart = sandbox_store_problems({p.store: plan.settings.sandbox_stores[p.store] for p in paths
                                        if p.store != DEVICE},
                                       store_root=plan.settings.site.store_root,
                                       secrets_path=plan.settings.site.secrets_path, checkout=plan.code.checkout,
                                       resolve=True)
        if apart:
            raise StageFault.infra(f"the driver's sandbox stores on this node: {'; '.join(apart)}", retryable=False)
        for m in stores:
            if not m.source.is_dir():
                raise StageFault.infra(f"the sandbox store bound at {m.target}, {m.source}, is no directory on this "
                                       "node", retryable=False, store=str(m.source))
        for device in devices:
            if not is_device(device):
                raise StageFault.infra(f"this node has no device {device} for the driver's sandboxes", retryable=True,
                                       device=device)
        return stores, devices

    def latest_drive_start(self, wall: float) -> float | None:
        """The latest time the drive may start for its full *wall* and the margin to fit before
        the allocation's end (``StageContext.deadline_cap``); None when there is no end to keep
        to."""
        cap = self.ctx.deadline_cap
        return None if cap is None else cap - wall - self.params.wall_margin_sec

    def start_driver(self, image: PinnedImage, files: _DriverFiles, url: str, wall: float) -> RoleHandle:
        """Start the driver, whose drive has *wall* seconds from its own start, or does not start
        (``latest_drive_start``)."""
        p = self.params
        # The driver's declared memory always goes, so the role's memory guard is always on.
        command = ["python", "-m", "hls.drive.role", "--task-ref", TASK_REF, "--artifact-url", url,
                   "--local", LOCAL, "--out", f"{OUT}/{TRACE_DIR}", "--outputs-out", OUTPUTS_REPORT,
                   "--stream-out", STREAM_RECORD,
                   "--split", p.split, "--request-timeout-sec", str(p.request_timeout_sec),
                   "--drive-timeout-sec", str(wall), "--nofile-soft", str(NOFILE_SOFT),
                   "--reply-max-bytes", str(p.reply_max_bytes), "--trace-cap-bytes", str(p.trace_cap_bytes),
                   "--memory-limit-bytes", str(p.driver_shape.memory_gib * _GIB)]
        if p.measured_baseline:
            command.append("--measured-baseline")
        start_by = self.latest_drive_start(wall)
        if start_by is not None:
            command += ["--start-by", f"{start_by:.3f}"]
        spec = ContainerSpec(image=image, command=tuple(command), workdir=files.workdir, mounts=files.mounts,
                             gpus=p.driver_shape.gpus > 0, devices=files.devices)
        # Oversubscribed on the artifact's memory where the allocation says so; its own step limit,
        # and so its guard, stay its own memory.
        shared = self.running.get(ARTIFACT) if p.driver_memory_shared else None
        if p.driver_memory_shared and shared is None:
            raise StageFault.infra("the driver shares the artifact's memory, and the artifact is not running",
                                   retryable=True)
        return self._start(DRIVER, spec, p.driver_shape, "harness", files.env, overlap=p.driver_overlaps,
                           share_memory_of=shared)

    def drive(self, artifact: RoleHandle, driver: RoleHandle) -> tuple[RoleOutcome, RoleOutcome | None]:
        """Wait for the driver while the artifact serves; return how the driver ended, and how
        the artifact did when it ended first. An artifact that ends first is waited for at once
        (``cleared``)."""
        ctx = self.ctx
        ended = None
        while driver.poll() is None and not ctx.stopping and ctx.clock() < ctx.deadline:
            if artifact.poll() is not None:
                ended = self.cleared(self.wait(ARTIFACT))
                break
            driver.wait(DRIVE_POLL_SEC)
        return self.wait(DRIVER), ended

    def cleared(self, outcome: RoleOutcome) -> RoleOutcome:
        """*outcome*, the artifact's ending, once an artifact that wrote past its disk cap has its
        writable directories removed: its step's processes are gone, and the bytes it left on
        the node's disk would leave none for the driver or anything else. The
        ``drive.artifact_cleared`` event records it; a removal that fails is logged in it, and
        the artifact's fault stands."""
        if outcome.ending != "disk_limit":
            return outcome
        error = None
        try:
            remove_written(self.written)
        except OSError as e:
            error = f"{type(e).__name__}: {e}"
            log.error("the artifact's writable directories were not all removed: %s", error)
        self.ctx.events.emit("drive.artifact_cleared", paths=[str(path) for path in self.written], error=error)
        return outcome

    # results

    def stream_record(self) -> TraceStreamRecord | None:
        """The driver's record of how its trace writer kept up (``STREAM_RECORD``), read once its
        role has ended; None when it wrote none, or none that parses."""
        try:
            return TraceStreamRecord.model_validate_json((self.ctx.role_dir(DRIVER) / STREAM_RECORD_FILE).read_bytes())
        except (OSError, ValidationError):
            return None

    def with_stream(self, fault: FaultRecord, driven: RoleOutcome) -> FaultRecord:
        """*fault*, when the driver's own end is among its faults and the driver wrote its record
        of its trace writer, with that record, scrubbed: why the writer failed, if it did, in the
        reason (its last line) and the evidence (``driver.trace_stream.failure``), and how far
        behind the drive it fell in the evidence."""
        record = self.stream_record() if classify_role(driven) is not None else None
        if record is None:
            return fault
        evidence = {**fault.evidence, "driver.trace_stream.backlog_peak": str(record.backlog_peak),
                    "driver.trace_stream.lag_max_sec": f"{record.lag_max_sec:.3f}",
                    "driver.trace_stream.space_wait_max_sec": f"{record.space_wait_max_sec:.1f}"}
        reason = fault.reason
        if record.failure is not None:
            failure = self.ctx.scrubber.scrub(record.failure)
            evidence["driver.trace_stream.failure"] = failure
            reason = f"{reason}; its trace writer failed: {failure.strip().splitlines()[-1]}"
        return fault.model_copy(update={"reason": reason, "evidence": evidence})

    def publish(self, record: DriveRecord, *, trace: str | None = None) -> dict[str, str]:
        self.ctx.events.emit("drive_result", outcome=record.outcome, reason=record.reason)
        outputs = {"drive": self.ctx.write_output(DRIVE_FILE, record.model_dump_json(indent=2).encode())}
        if trace is not None:
            outputs["trace"] = trace
        return outputs

    def placement(self) -> ArtifactPlacement | None:
        """Where the artifact's step ran, from its step wrapper's report; None without one."""
        report = self.ctx.report(ARTIFACT)
        if report is None:
            return None
        return ArtifactPlacement(cpus=report.cpus, numa_nodes=report.numa_nodes, numa_split=report.numa_split)

    def command_started_after(self) -> float | None:
        """Seconds from the artifact's step's start until the stage saw its command start
        (``ArtifactRecord.command_started_after_sec``); None when either was not seen."""
        ctx = self.ctx
        began, step = ctx.command_started_at(ARTIFACT), ctx.role_started_at(ARTIFACT)
        return None if began is None or step is None else began - step

    def _not_served(self, outcome: Literal["crashed", "not_ready"], reason: str, sha: str, url: str | None,
                    ended: RoleOutcome | None, seal_probe: SealProbeRecord) -> dict[str, str]:
        artifact = ArtifactRecord(image_sha256=sha, ending=None if ended is None else ended.ending,
                                  exit_code=None if ended is None else ended.exit_code,
                                  command_started_after_sec=self.command_started_after(),
                                  ready_after_command_sec=None, placement=self.placement(),
                                  engine_cache=self.cache_record)
        return self.publish(DriveRecord(outcome=outcome, reason=reason, seal_probe=seal_probe,
                                        process_ceiling=self.ceiling, artifact=artifact, driver=None, url=url,
                                        drive_started_at=None, drive_ended_at=None, trace=None))

    def started(self, outcome: RoleOutcome) -> None:
        """Raise infra unless the artifact's command started, before the stage blames the
        server for how its start went: the seal's started file says only that the runtime was
        launched. *outcome* is how the role ended."""
        if not self.ctx.command_started(ARTIFACT):
            raise StageFault.infra("the artifact's container never started its command (no start marker on its "
                                   f"stderr); it ended {outcome.ending} with status {outcome.exit_code}",
                                   retryable=True)

    def crashed(self, sha: str, url: str | None, seal_probe: SealProbeRecord) -> dict[str, str]:
        """The artifact's role ended before it was ready, or the stage was told to stop: the
        contestant's crash, or the roles' fault, or infra when its command never started. A crash
        is scored as a failure whatever its status, 0 among them, so whether others starved it is
        asked here (``scored_end_fault``)."""
        outcome = self.cleared(self.wait(ARTIFACT))
        fault = scored_end_fault(outcome)
        if fault is not None:
            raise StageFault(fault)
        self.started(outcome)
        reason = f"the server exited with status {outcome.exit_code} before it was ready"
        return self._not_served("crashed", reason, sha, url, outcome, seal_probe)

    # the stage

    def run(self) -> dict[str, str]:
        """``_run``, then stop every role still running, and remove the driver's caches
        (``drop_caches``) and the trace's room (``drop_room``) however the stage ended. Once a
        contestant fault is decided, the stage raises it whatever else failed: it is read before
        the roles are stopped, so a stop that fails cannot lose it, and again after, when
        stopping them may have decided it."""
        try:
            return self._run_and_stop()
        finally:
            self.drop_caches()
            self.drop_room()

    def _run_and_stop(self) -> dict[str, str]:
        try:
            outputs = self._run()
        except Exception as e:
            decided = self.ctx.contestant_fault()
            try:
                self.stop_all()
            except Exception:
                log.exception("stopping the stage's roles failed")
            decided = self.ctx.contestant_fault() or decided
            if decided is not None and not (isinstance(e, StageFault) and e.fault == decided):
                raise StageFault(decided) from e
            raise
        self.stop_all()
        return outputs

    def drop_caches(self) -> None:
        """Remove the attempt's ``DRIVER_CACHE_DIR``, which nothing reads once the stage ended:
        it is no output, and every attempt makes its own. The ``drive.driver_cache`` event
        records the bytes it held first (None when the walk did not measure them) and whether it
        went. A failure is logged and recorded, never raised: the stage's own ending stands."""
        caches = self.ctx.attempt_dir / DRIVER_CACHE_DIR
        if not caches.exists():
            return
        try:
            held: int | None = disk_usage([caches])
        except (Unmeasured, OSError):
            log.exception("the driver's caches in %s could not be measured", caches)
            held = None
        error = None
        try:
            remove_tree(caches)
        except OSError as e:
            error = f"{type(e).__name__}: {e}"
            log.error("the driver's caches in %s were not all removed: %s", caches, error)
        try:
            self.ctx.events.emit("drive.driver_cache", path=str(caches), bytes=held, removed=error is None,
                                 error=error)
        except OSError:
            log.exception("the removal of the driver's caches could not be recorded")

    def drop_room(self) -> None:
        """Remove the trace's room (``LOCAL_DIR``), which nothing reads once the driver's role
        has ended: the role published the trace from it, or failed. It may hold the most the
        trace takes, which the stages after this one on the node may need. The
        ``drive.trace_room`` event records whether it went. A failure is logged and recorded,
        never raised: the stage's own ending stands."""
        room = self.room
        if room is None:
            return
        error = None
        try:
            remove_tree(room)
        except OSError as e:
            error = f"{type(e).__name__}: {e}"
            log.error("the trace's room in %s was not all removed: %s", room, error)
        try:
            self.ctx.events.emit("drive.trace_room", path=str(room), removed=error is None, error=error)
        except OSError:
            log.exception("the removal of the trace's room could not be recorded")

    def _run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        submission = None
        # The figures the build's brief stated, which its package carries; none for a measured image.
        recorded = None
        if isinstance(p.artifact, PackagedArtifact):
            manifest, result = self.package(p.artifact)
            recorded = result.served_ucounts
            if result.status == "invalid_artifact":
                return self.publish(DriveRecord(outcome="invalid_artifact", reason=result.reason,
                                                seal_probe=None, process_ceiling=None, artifact=None, driver=None,
                                                url=None, drive_started_at=None, drive_ended_at=None, trace=None))
            submission = self.submission(manifest, result)
        provisioned = self.provisioned()
        entries = self.image_entries()
        try:
            contract = load_task(p.task).contract
        except Exception as e:
            raise StageFault.infra(f"task {p.task.module} does not load: {type(e).__name__}: {e}",
                                   retryable=False) from e
        # Read before any role starts: a ceiling under its floor is infra, not retried; a node that
        # cannot hold its graded servers' declared per-uid limits is infra, retried.
        self.ceiling = self.ctx.process_ceiling()
        self.held = self.ctx.grading_ucounts(p.artifact_shape, recorded)
        # Every copy and extraction runs alone, so all of them come before the first role.
        artifact_data, driver_data = self.mounts(provisioned)
        if submission is not None:
            # Read from the store by the extraction role alone, which refuses bytes that are not
            # the recorded ones; the tree holds the contestant's code.
            image = ctx.pin_file(ctx.store.path(submission.key), sha256=submission.sha256, contestant=True)
            # The package stage wrote the image's runscript where the image runtime keeps it.
            entrypoint, runtime_env = [ctx.images.runscript], {}
        elif isinstance(p.artifact, PublishedArtifact):
            # Read off the shared filesystem by the extraction role alone, which refuses bytes
            # without the named digest; a retry would read the same file, so the refusal is final.
            # The tree holds the contestant's code.
            image = ctx.pin_file(Path(p.artifact.source), sha256=p.artifact.sha256, contestant=True, stored=True)
            entrypoint, runtime_env = [ctx.images.runscript], {}
        else:
            assert isinstance(p.artifact, ImageArtifact)
            image = self.catalog_image(entries, p.artifact.image)
            entrypoint, runtime_env = p.artifact.entrypoint, p.artifact.runtime_env
        harness = self.catalog_image(entries, p.harness_image)
        sha = image.file_sha256
        cache, self.cache_record = self.engine_cache(image, provisioned)
        seal = served_seal(p)
        seal_probe = self.probe_seal(harness, seal, seal.policy.network)

        driver_files = self.prepare_driver(driver_data)
        self.room_for_readiness()
        handle = self.start_artifact(image, entrypoint, runtime_env, artifact_data, cache)
        port = self.host_port(ARTIFACT, handle)
        if port is None:
            return self.crashed(sha, None, seal_probe)
        url = f"http://127.0.0.1:{port}"
        state = self.wait_ready(ARTIFACT, handle, url, contract, budget_sec=p.readiness_timeout_sec, shared=False,
                                margin_sec=p.wall_margin_sec)
        if state == "ended":
            return self.crashed(sha, url, seal_probe)
        if state == "not_ready":
            stopped = self.cleared(self.stop(ARTIFACT))
            fault = scored_end_fault(stopped)
            if fault is not None:
                raise StageFault(fault)
            reason = (f"the server ran but was not ready within the {p.readiness_timeout_sec}s readiness "
                      "budget")
            return self._not_served("not_ready", reason, sha, url, _own(stopped), seal_probe)
        began = ctx.command_started_at(ARTIFACT)
        assert began is not None
        ready_after = ctx.clock() - began
        started_after = self.command_started_after()
        ctx.events.emit("artifact_ready", url=url, after_command_sec=round(ready_after, 3),
                        command_started_after_sec=None if started_after is None else round(started_after, 3))

        wall = p.drive_timeout_sec
        if not self.hold_drive():
            raise self.no_room(f"the drive's full wall of {wall:.0f}s and the {p.wall_margin_sec}s margin after it do "
                               "not fit before the allocation's end", served=self.cleared(self.stop(ARTIFACT)))
        driver = self.start_driver(harness, driver_files, url, wall)
        # The drive starts with the driver's step, after its wait for the step.
        drive_started = ctx.role_started_at(DRIVER) or ctx.clock()
        driven, ended = self.drive(handle, driver)
        drive_ended = ctx.clock()
        served = ended if ended is not None else self.cleared(self.stop(ARTIFACT))
        late = ctx.refusal(DRIVER, [driven], (LATE_START_EXIT,))
        if late is not None:
            # The driver drove nothing, so only the artifact's own ending can decide besides it.
            raise self.no_room(f"the driver started too late to drive its full wall of {wall:.0f}s: {late.reason}",
                               served=served)
        # A server that ended on its own before the stage stopped it, seen during the drive or only
        # at its stop, is scored on the replies it gave, whatever its status: a failing end, whose
        # starvation is asked of a status of 0 too.
        fault = stage_fault([served, driven]) or (scored_end_fault(served) if served.ending == "exited" else None)
        if fault is not None:
            raise StageFault(self.with_stream(fault, driven))
        trace = ctx.reported_output(TRACE_END, DRIVER)
        try:
            end = read_end(ctx.outputs_dir / TRACE_DIR)
        except TraceIncomplete as e:
            raise StageFault.infra(f"the driver exited 0 without a complete trace: {e}", retryable=True) from e
        stream = self.stream_record()
        if stream is None:
            raise StageFault.infra("the driver exited 0 with no record of its trace writer", retryable=True)
        own = _own(served)
        artifact = ArtifactRecord(image_sha256=sha, ending=None if own is None else own.ending,
                                  exit_code=None if own is None else own.exit_code,
                                  command_started_after_sec=started_after, ready_after_command_sec=ready_after,
                                  placement=self.placement(), engine_cache=self.cache_record)
        record = DriveRecord(outcome="served", reason=None, seal_probe=seal_probe,
                             process_ceiling=self.ceiling, artifact=artifact,
                             driver=DriverRecord(ending=driven.ending, exit_code=driven.exit_code,
                                                 drive_timeout_sec=wall, trace_stream=stream),
                             url=url, drive_started_at=drive_started, drive_ended_at=drive_ended, trace=end)
        return self.publish(record, trace=trace)


@register
class DeployDriveStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "deploy_drive"
    params_model: ClassVar[type[BaseModel]] = DeployDriveParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    #: The artifact is the contestant's server, so every file of an attempt holds the
    #: contestant's bytes, the trace of its replies and the driver's logs of them included
    #: (``Stage.runs_contestant_code``).
    runs_contestant_code: ClassVar[bool] = True

    @classmethod
    def scored_server_cpus(cls, params: BaseModel) -> int:
        """The CPUs of the server whose drive is scored, the artifact's, which a site with
        ``site.evaluation_socket_tasks`` asks for as one task on one socket
        (``hls.pipeline.planner.socket_tasks``)."""
        assert isinstance(params, DeployDriveParams)
        return params.artifact_shape.cpus

    @classmethod
    def sandbox_stores(cls, params: BaseModel) -> tuple[str, ...]:
        """The sandbox stores the driver binds (``DeployDriveParams.sandbox_paths``)."""
        assert isinstance(params, DeployDriveParams)
        return tuple(dict.fromkeys(p.store for p in params.sandbox_paths if p.store != DEVICE))

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: a copy of the weights, the rows and the full tool
        bundle (the artifact mounts no tool bundle; ``hls.provision.mounts``), the two
        extractions, the seal probe, the artifact and the driver, and for a drive of the measured
        baseline the copy that seeds its server's engine cache (``engine_cache_owner``; with no
        *params*, counted)."""
        assert params is None or isinstance(params, DeployDriveParams)
        return 9 if params is None or params.engine_cache_owner is not None else 8

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        """The stage's resources are its roles' together, an overlapping driver's CPUs being
        the artifact's and its memory, where it is oversubscribed (``driver_memory_shared``),
        the artifact's too, and in memory the runner's beside them, in disk the images it extracts
        (``extracted_disk_gib``)
        and the copies of its roles' data (``data_gib``) beside them; a site may change only its
        time, which every role runs for (``_Drive._timed``). An oversubscribed driver's own memory
        must still fit the stage beside the runner's. The artifact declares the disk that caps its
        writes, and the time covers both readiness budgets, the drive wall and the margin."""
        assert isinstance(params, DeployDriveParams)
        artifact, driver = params.artifact_shape, params.driver_shape
        roles = concurrent([artifact, driver])
        extracted = extracted_disk_gib(params.artifact)
        memory = artifact.memory_gib if params.driver_memory_shared else roles.memory_gib
        if params.driver_memory_shared and driver.memory_gib + runner_memory_gib > shape.memory_gib:
            raise ValueError(f"the driver's {driver.memory_gib} GiB, which shares the artifact's memory, and the "
                             f"runner's {runner_memory_gib} GiB do not fit in the stage's {shape.memory_gib} GiB")
        want = roles.model_copy(update={"cpus": artifact.cpus if params.driver_overlaps else roles.cpus,
                                        "memory_gib": memory + runner_memory_gib,
                                        "disk_gib": roles.disk_gib + extracted + params.data_gib,
                                        "time_sec": shape.time_sec})
        if shape != want:
            raise ValueError(f"the stage's resources must be the artifact's and the driver's together, with "
                             f"{runner_memory_gib} GiB more memory for the runner, "
                             f"{extracted} GiB more disk for the images it extracts and "
                             f"{params.data_gib} GiB for the copies of their data, {want}, in all but time; they "
                             f"are {shape}")
        if artifact.disk_gib < 1:
            raise ValueError("the artifact role must declare the disk its workdir and engine cache may fill "
                             "(artifact_shape.disk_gib)")
        need = (params.staging_sec + params.runtime_start_sec + 2 * params.readiness_timeout_sec
                + params.drive_timeout_sec + params.wall_margin_sec)
        if shape.time_sec < need:
            raise ValueError(f"the stage's time {shape.time_sec}s does not cover its floor of {need:.0f}s: the staging, "
                             "the runtime's start, two readiness budgets, the drive's wall and the margin")

    @classmethod
    def seals_containers(cls, params: BaseModel | None) -> bool:
        return True

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Drive(ctx).run()
