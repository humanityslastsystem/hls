"""The agent_build stage: an arm's agent builds a submission in a sealed sandbox, on the runner's
clock, and the stage bundles what it committed.

The stage runs in the runner and stays light; every copy, the commit point and the transcript are
roles. Its one long role is the sandbox, a contestant container on the agent cap whose only
harness process is the exec server; every command the agent runs goes through it to a spawner in
a nested namespace (``hls.agent.exec_server``). In order:
1. It refuses to start when the runner's environment holds a variable harbor would read in place
   of the arm's settings (``hls.agent.harnesses.harness_env_leaks``), when it holds no value of
   the route's base URL secret, and when the commands' process ceiling, or the site's cap on it
   (``site.nproc_max``), is under its floor (``hls.substrate.nproc``).
2. It lays out the sandbox's node-local directories (``hls.agent.workspace``), pins the agent cap
   that ``sandbox_image`` names in this instance's ``images.json`` (``node_image``), and copies the
   agent's data to the node (``materialize``). A role seeds the workspace with what the image holds
   at ``/workspace``, which the workspace's bind would hide, and a role copies in the task's
   uploaded files and skills.
3. It renders the brief with the sandbox's own figures (``brief``, ``hls.prompt``) and writes
   ``brief.md`` and ``brief_provenance.json``.
4. It makes every node-local file of the collection's roles before the sandbox runs
   (``prepare_collection``), so a contestant that fills the node's disk leaves them nothing to make.
5. It starts the budget service (``hls.budget.service``) and the sandbox, sealed with the
   producer's network policy plus the arm's route host (``seal_policy``) and the package index. Its
   env file holds no secret: the route's credential, base URL and output limit are the agent's
   ``extra_env``, which harbor sends over the token-gated exec socket (``build_agent``).
   ``agent_env.json`` names each variable and where it came from, never a value, and every record
   of harbor's loggers is scrubbed while the build runs. A producer that runs the reference
   service also has the task's measured baseline image pinned, the service's socket directory
   bound read-only into the sandbox with ``service_harness.SOCKET_ENV_VAR`` naming the socket, and
   the service started on the sandbox's GPUs once the sandbox has reported; it is stopped right
   after the sandbox and never decides how the build ends (``hls.stages.reference_service``).
6. It waits for the exec server (``command_started``, ``server_answered``), probes the seal as
   contestant commands (``hls.agent.probe``), then runs the preflight
   (``hls.agent.preflight``) and harbor's ``setup``.
7. The agent's clock starts (``WallClock.start``) and harbor's ``run`` drives the agent. Beside it
   the stage enforces the hard limit, reports the sandbox once it has ended, emits
   ``build.heartbeat`` with what the health check read (``hls.agent.health``), stops re-entry for a
   derailed session, and snapshots ``main`` every ``snapshot_interval_sec``; the newest two are kept
   as evidence until the commit point bundles main (``drop_snapshots``).
8. When the agent's part ends it stops the sandbox and then the snapshot, waits for every process
   of the sandbox to be gone, and runs the commit point (``hls.agent.commit``), which writes the
   source bundle or the record that there is none. The transcript role then copies the agent's logs
   with every secret value replaced (``hls.agent.transcript``), and the extraction role has harbor
   read the trajectory and spend from that copy (``hls.agent.extract``): the runner never parses the
   agent's logs. A failed extraction is recorded and the build goes on.

Whose fault an ending is. In one line: a contestant end ends the build;
collection and the commit point are the harness's work, and a failure there is infra; the
contestant scores zero only when nothing is gradable, or for an end on the sandbox's own memory
under a plan that does not grade one.
- The contestant's evidence comes first: a rise of the sandbox step's pids limit count ends the
  build as ``contestant_pids``, and a memory kill the runner made in it ends a lost or failed build
  as ``contestant_oom``, whatever harbor raised; either counts only within
  ``OOM_EVIDENCE_MAX_AGE_SEC`` of the failure it would explain (``within_evidence_age``).
- The agent's own end, the CLI's nonzero exit and the hard limit are normal ends: the stage stops
  the sandbox and collects. A harbor command that fails after the clock started also ends the
  agent's part (``harbor_command_failed``), since the agent's leftover processes can kill harbor's
  commands. Either failure is the contestant's only once the CLI has been launched since the clock
  started (``cli_launches``); before that it is infra, retried. Either is also infra, retried, when
  the uid's other tasks fill the commands' process pool or cannot be counted (``starvation_fault``).
- The egress interceptor must have reported within ``hls.stages.package.HEARTBEAT_STALE_SEC`` when
  the agent's part ended (``interceptor_stale``); otherwise the end is infra, retried.
- A sandbox that ends on its own during the agent's part is judged by its roles
  (``classify_build_end``): an OOM certain at its own limit with the harness within its reserve, a
  write past its disk cap, or its pids limit reached is the contestant's end, and the build is
  graded on its last commit; any other end is infra and retried, as is any end before the clock
  started. An end on the sandbox's own memory scores zero at once, with nothing graded, under a
  plan whose ``run.grade_after_sandbox_oom`` is false (``sandbox_oom_zero``). After a write past
  the disk cap, the sandbox's writable directories but the repository and the logs are removed
  before the commit point runs. A lost exec channel is read from the environment
  (``SandboxEnvironment.lost``), never from what harbor raised, since harbor can turn it into the
  agent's own nonzero exit.
- A commit point that finds no commit on main, or a repository unreadable because of what the
  contestant made, writes ``no_submission``, which the package stage publishes as invalid_artifact;
  after a contestant end it is the contestant's zero (``decides_contestant_faults``). Any other
  failure of the commit point is the harness's: it runs up to ``COMMIT_TRIES`` times within the
  collection allowance (``commit_deadline``), and when no try writes, the
  attempt ends infra that is not retried and the build is void. Every ``.tmp-*`` a killed try left
  in the outputs is removed before each try and before the attempt ends (``clear_partials``).
- Once the commit point has written, every later failure is infra that is not retried, declared at
  once (``StageContext.declare_no_retry``) so that a runner killed outright ends with it too.
- Every other failure before the clock starts is infra; a preflight that read back another model,
  window or output limit is not retried, the others are.

A probe or preflight-only build (``stop_after``, ``hls run --probe-only`` or ``--preflight-only``)
runs steps 1 to 5, then the whole probe (``hls.agent.probe``), the preflight for a preflight-only
build and, with ``probe_oom``, a memory hog. It starts no clock and runs no agent, snapshot or
commit point, and publishes ``probe.json`` beside ``build.json``. A failed check is a not-ok record
for the operator, never a fault (``stop_build``).

The stage protocol is synchronous: ``run`` drives the build's own event loop with
``asyncio.run``. The budget service has its own thread and never waits on that loop, so a busy
loop cannot delay an answer to ``get_budget``.

Harbor is imported only when a build runs (``_Build.run``, through ``hls.agent.harnesses`` and
``hls.agent.environment``): the role programs and the runner import ``hls.stages.registry``,
which loads this module, inside images that have no harbor."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import logging
import math
import os
import secrets
import socket
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from hls.agent import exec_protocol, workspace
from hls.agent import probe as probe_mod
from hls.agent.commit import (CHAIN_BYTES_MAX, NO_MAIN_EXIT, PARTIAL_DIR, UNREADABLE_EXIT, CommitRecord,
                              commit_command, prune_snapshots, snapshot_name)
from hls.agent.extract import TRAJECTORY_FILE, TRAJECTORY_MAX_BYTES, ExtractionRecord, Usage
from hls.agent.health import check as check_health
from hls.agent.preflight import PreflightResult, preflight
from hls.agent.transcript import COPY_DIR, INDEX_FILE
from hls.agent.uploads import UploadRefused, checked_upload, task_skills, upload_destination
from hls.arms import ArmRecord, adapter_for, route_of
from hls.budget.mcp import MCP_TARGET
from hls.budget.service import BudgetService, WallClock
from hls.core.model import Hardware, gb_of_gib
from hls.drive import bounds
from hls.egress.policy import PROBE_DIAL, NetworkPolicy, SealPolicy, refused_host
from hls.egress.seal_probe import CHECK_SEC
from hls.package.wheels import COMPUTE_CAPABILITY
from hls.pipeline.plan import Plan
from hls.pipeline.stage import (LOG_CLOSE_SEC, STOP_GRACE_SEC, Stage, StageContext, StageFault, register,
                                stage_fault)
from hls.prompt import (blocked_packages_of, budget_brief_section, drive_bounds_brief_section,
                        packaging_brief_section, sandbox_brief_section, serving_brief_section, uploads_brief_section)
from hls.provision.model import Provisioned
from hls.provision.mounts import WEIGHTS_TARGET, DataCopies, data_mounts
from hls.reference import service_harness as rs
from hls.settings import SiteSettings, StageName
from hls.spec.resources import ResourceShape, concurrent
from hls.spec.submission import SubmissionContract
from hls.stages.engine_cache import ENGINE_CACHE
from hls.stages.images import KEPT_DISK_GIB
from hls.stages.inputs import dev_files, measured_task
from hls.stages.package import (IMAGES_OUTPUT, NO_SUBMISSION_OUTPUT, SERVED_UCOUNTS_OUTPUT, SOURCE_BUNDLE_OUTPUT,
                                PipLimits, deny_list, interceptor_stale)
from hls.stages.reference_service import SERVICE_SOCKET, BuildService, ReferenceServiceParams, shape_problems
from hls.stages.score import host_role_env
from hls.store.base import check_segment
from hls.store.digest import Sha256Hex, check_sha256_hex, sha256_bytes
from hls.store.manifest import Manifest
from hls.substrate.seal import client_env, namespace_ports, policy_refusals
from hls.substrate.base import FaultRecord, RoleHandle
from hls.substrate.container import TREE_ABSENT_EXIT, ContainerSpec, ImageRuntime, Mount, PinnedImage, SealPlan
from hls.substrate.faults import (OOM_EVIDENCE_MAX_AGE_SEC, RoleOutcome, SharedMemoryEvidence, SplitEvidence,
                                  classify_shared_build_end, infra_fault, starvation_fault, within_evidence_age)
from hls.substrate.memory_watch import remove_written
from hls.substrate.factory import substrate
from hls.substrate.nproc import Accounting, ProcessCeiling
from hls.substrate.secret_forms import registry_forms
from hls.substrate.ucounts import commands_limits
from hls.substrate.secrets import Scrubber
from hls.taskref import TaskRef, load_task, task_dir
from hls.util import MAX_SOCKET_PATH_BYTES, remove_tree

if TYPE_CHECKING:
    from harbor.skills import ResolvedSkill

    from hls.agent.environment import SandboxEnvironment

log = logging.getLogger(__name__)

#: The sandbox role.
SANDBOX = "sandbox"
#: The commit point's role; its later tries are ``commit-2`` and ``commit-3``.
COMMIT = "commit"
#: The role that copies the agent's logs out.
TRANSCRIPT = "transcript"
#: The role in which harbor reads the agent's trajectory and spend from that copy
#: (``hls.agent.extract``).
EXTRACT = "extract"
#: Seconds the extraction role may run before it is stopped and the trajectory and spend recorded
#: as unavailable. Harbor reads the largest codex session the role's memory (``extract_shape``)
#: admits in about 65 s; this is about nine times that.
EXTRACT_SEC = 600.0
#: Bytes of the extraction role's record the stage reads at most; the record is a few hundred, and
#: a longer one leaves the trajectory and spend unavailable.
EXTRACT_RECORD_MAX_BYTES = 2**16
#: The role that copies the agent cap's ``/workspace`` into the sandbox's workspace.
SEED = "seed-workspace"
#: The role that copies the task's uploaded folder and its skills into the sandbox's directories.
UPLOADS = "uploads"
#: The bundle and the no-submission record the commit point writes, as outputs.
SOURCE_BUNDLE_FILE, NO_SUBMISSION_FILE = "source.bundle", "no_submission.json"
#: The build's record, as an output, and that output's name.
BUILD_FILE, BUILD_OUTPUT = "build.json", "build"
#: The variables of the sandbox's env file and of harbor's exec requests, by name and where
#: each came from, as an output.
AGENT_ENV_FILE = "agent_env.json"
#: The brief and where each of its sections came from, as outputs.
BRIEF_FILE, BRIEF_PROVENANCE_FILE = "brief.md", "brief_provenance.json"
#: The file of the record of the figures the brief states for a graded server
#: (``hls.substrate.ucounts.ServedUcounts``), the ``served_ucounts`` output, which the package
#: carries to the grading.
SERVED_UCOUNTS_FILE = "served_ucounts.json"
#: Where the transcript's copy goes among the outputs.
TRANSCRIPT_DIR = "transcript"
#: The output name of harbor's trajectory of the agent's run, ``transcript/trajectory.json``.
TRAJECTORY_OUTPUT = "trajectory"
#: The attempt's directory, on the store, of each commit point try's record and fresh repository.
COMMIT_RECORDS = "commit-points"
#: Where a harness container's out directory, its record directory and its outputs report mount.
OUT, RECORD, OUTPUTS_REPORT = "/hls/out", "/hls/record", "/hls/outputs.json"
#: Bytes a role's outputs report is given on the node before the sandbox starts, which it then
#: writes in place: one record takes about 130 bytes.
OUTPUTS_REPORT_BYTES = 2**16
#: Seconds between the stage's checks while the agent runs (the hard limit, the sandbox, the
#: snapshots) and while it waits for a role.
TICK_SEC = 1.0
#: Seconds a sandbox whose exec channel was lost gets to end on its own before the stage stops it.
#: Ten of the stage's checks (``TICK_SEC``): room for an ending container to be reaped, short
#: beside the stop that follows.
LOST_WAIT_SEC = 10.0
#: Seconds the stage waits for every process of a sandbox that ended to be gone before the commit
#: point runs: a minute for the kernel to reap a step's processes, which the stop before it has
#: already signalled.
GONE_WAIT_SEC = 60.0
#: Runs of the commit point on the same workspace before its failure ends the attempt: the first,
#: and each later one only while time is left before the collection's last steps
#: (``commit_deadline``).
COMMIT_TRIES = 3
#: Seconds between two runs of the commit point: a pause for a passing fault of the node or the
#: shared filesystem to clear, small beside one run's time.
COMMIT_PAUSE_SEC = 10.0
#: Seconds past the time a commit try was handed (``--seconds``, counted from its command's start)
#: that the stage lets it run before it ends the try as out of time: room for the program to stop
#: the git command its time ran out in and write its record. Half a minute is many times what a
#: process's end and a small record's write take.
COMMIT_WIND_DOWN_SEC = 30
#: Seconds a role the stage stops as planned may take to be gone and reported: its grace after
#: SIGTERM and the wait after SIGKILL (``StageContext.stop_role``), then the wait for its capped
#: logs (``LOG_CLOSE_SEC``).
STOPPED_ROLE_SEC = 2 * STOP_GRACE_SEC + LOG_CLOSE_SEC
#: Seconds the sandbox's log files take to be scrubbed in place when it is reported: it logs its
#: secrets scrubbed (``scrub_logs``), so each of its two log files is rewritten with every value
#: replaced, read and written on the store once. Each is kept to the head and tail of
#: LOG_KEEP_BYTES, so up to 2 x LOG_KEEP_BYTES a file. A store scrubs about 40 MiB/s, so this is a
#: wide margin, to cover a slow store.
SANDBOX_SCRUB_SEC = 300
#: Seconds the collection gives the transcript role's copy of the agent's logs, up to
#: ``hls.agent.transcript.MAX_BYTES`` read off the node and written to the store scrubbed. At the
#: store's scrub rate above, a copy at that 8 GiB bound takes about 205 s; ten minutes is about
#: three times that.
TRANSCRIPT_COPY_SEC = 600
#: Seconds the collection keeps, once the commit point has written, for what follows it: the
#: transcript role (``TRANSCRIPT_COPY_SEC`` and its logs' ``LOG_CLOSE_SEC``), then the extraction
#: role for at most ``EXTRACT_SEC`` and, past it, its stop (``STOPPED_ROLE_SEC``). The commit point
#: ends this long before the stage's deadline (``commit_deadline``).
AFTER_COMMIT_SEC = math.ceil(TRANSCRIPT_COPY_SEC + LOG_CLOSE_SEC + EXTRACT_SEC + STOPPED_ROLE_SEC)
#: Seconds a probe or preflight-only build's stage keeps after its hard limit, the whole of its
#: collection: a sandbox whose channel was lost gets ``LOST_WAIT_SEC`` to end, and the sandbox is
#: stopped (``STOPPED_ROLE_SEC``) and its logs scrubbed in place when it is reported
#: (``SANDBOX_SCRUB_SEC``). It runs no commit point.
STOP_BUILD_COLLECT_SEC = math.ceil(LOST_WAIT_SEC + STOPPED_ROLE_SEC + SANDBOX_SCRUB_SEC)
#: Seconds of a full build's collection that are not the commit point's, the fixed steps around
#: it: the sandbox's end and stop with its logs' scrub (``STOP_BUILD_COLLECT_SEC``), the snapshot's
#: stop (``STOPPED_ROLE_SEC``), the wait for the sandbox's processes to be gone (``GONE_WAIT_SEC``),
#: and what follows the commit point (``AFTER_COMMIT_SEC``). The commit point gets the rest of
#: ``run.collect_allowance_sec``.
COLLECT_STEPS_SEC = math.ceil(STOP_BUILD_COLLECT_SEC + STOPPED_ROLE_SEC + GONE_WAIT_SEC + AFTER_COMMIT_SEC)
#: The least time ``run.collect_allowance_sec`` must leave the commit point beside the fixed steps
#: (``COLLECT_STEPS_SEC``); a plan whose allowance leaves less is refused. Bundling a 16 GiB
#: repository takes about 180 s on the commit role's core; ten minutes is three times that.
COMMIT_MIN_SEC = 600


def commit_point_sec(collect_allowance_sec: int) -> int:
    """The least time a full build's commit point gets of the collection allowance
    *collect_allowance_sec*: what the collection's fixed steps leave (``COLLECT_STEPS_SEC``),
    however long they take, since the commit point gets the time left before
    ``AFTER_COMMIT_SEC`` and they bound the steps before it. The brief states it."""
    return collect_allowance_sec - COLLECT_STEPS_SEC


def collection_sec(stop_after: Literal["probe", "preflight"] | None, collect_allowance_sec: int) -> int:
    """Seconds a build's stage keeps after the agent's hard limit: the run's
    *collect_allowance_sec* for a full build, ``STOP_BUILD_COLLECT_SEC`` for a probe or
    preflight-only build (*stop_after*), which runs no commit point."""
    return STOP_BUILD_COLLECT_SEC if stop_after is not None else collect_allowance_sec


#: Seconds each command of the rest of a stop build's probe may run: four times the seal probe's
#: check time (``CHECK_SEC``), the bound of the sealed checks, since the rest read files and limits
#: in the sandbox and start no network wait.
PROBE_CHECK_SEC = 4 * CHECK_SEC
#: The ``--probe-oom`` memory hog: a sandbox command that allocates until something ends it.
OOM_HOG = f"{exec_protocol.SANDBOX_PYTHON} -I -c 'b = []\nwhile True: b.append(bytearray(64 * 2**20))'"
#: Seconds the memory hog may run: ten minutes, several times what allocating a sandbox's whole
#: memory 64 MiB at a time can take, so only a hog nothing ended runs past it.
OOM_PROBE_SEC = 600
#: The probe's record, as an output of a stop build, and that output's name.
PROBE_FILE, PROBE_OUTPUT = "probe.json", "probe"
#: Where a graded server may write: its container's /tmp, /var/tmp and $HOME, and the engine
#: cache the drive stage mounts. ``/workspace`` is read-only there.
SERVED_WRITABLE = ("/tmp", "/var/tmp", "$HOME", ENGINE_CACHE)
#: Copies of cache entries one attempt makes at most: the weights, the rows and one tool bundle,
#: as the drive stage counts its own (``hls.stages.deploy_drive``).
DATA_COPIES = 3
#: The sources of the env file's variables in ``agent_env.json``; a request's variable is
#: ``plain`` or a grant, ``grant:<secret name>``.
Source = Literal["plain", "harness", "seal"]
#: How the agent's part of a build ended, when the stage collects: the agent's own end, the CLI's
#: nonzero exit, the hard limit, a failed harbor command after the clock started, or the
#: sandbox's end on the contestant's fault: its memory, its step's pids limit or its disk; or, for
#: a probe or preflight-only build, the stop after it (``stop_after``).
Ending = Literal["agent_ended", "agent_nonzero_exit", "hard_limit", "harbor_command_failed", "contestant_oom",
                 "contestant_pids", "disk_limit", "probe", "preflight"]
# The ends of the agent's part that are failures, which others filling the uid's process pool
# may have caused (``_Build._unless_starved``).
STARVABLE_ENDINGS: frozenset[Ending] = frozenset({"agent_nonzero_exit", "harbor_command_failed"})
# Bytes of the sandbox's log read as one line at most: a start line of the exec server is its
# fields, a working directory (a path, at most 4,096 bytes) and 120 characters of its command.
_LOG_LINE_BYTES = 64 * 2**10
# A LogRecord's own attributes, which the scrubbing of harbor's records leaves to the message.
_RECORD_FIELDS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}
_GIB = 2**30
# Reads a commit point's record (``CommitRecord``, a dataclass) from its JSON.
_COMMIT_RECORD = TypeAdapter(CommitRecord)


def commit_role(n: int) -> str:
    """The role of the commit point's *n*-th try, counted from 1."""
    return COMMIT if n == 1 else f"{COMMIT}-{n}"


class _Uncollected(Exception):
    """The commit point wrote nothing: the fault the attempt ends with, the last try's record and
    every try."""

    def __init__(self, fault: StageFault, record: CommitRecord | None, tries: list[CommitTry]) -> None:
        super().__init__(fault.fault.reason)
        #: The fault the attempt ends with.
        self.fault = fault
        #: The last try's commit record; None when it wrote none that can be read.
        self.record = record
        #: Every try, in order.
        self.tries = tries


class AgentBuildParams(DataCopies):
    """What one agent build runs, on what, and for how long."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, as the roles and the brief rebuild it.
    task: TaskRef
    #: The arm as the plan resolved it: harness, model, route, window, output limit, effort.
    arm: ArmRecord
    #: Why this arm differs from the task's earlier rollouts of the same arm name; None when it
    #: does not (the planner refuses a difference without one).
    arm_change: str | None = None
    #: The agent cap's name in this instance's ``images.json``: the image the sandbox, the commit
    #: point and the snapshots run from.
    sandbox_image: Annotated[str, AfterValidator(check_segment)]
    #: The packages the base under the sandbox holds for every install, name to version, which
    #: the brief states (``catalog.held_versions``); None for a sandbox that holds none.
    sandbox_held: dict[str, str] | None
    #: The images stage of this instance whose ``images.json`` names the agent cap.
    images_stage: StageName = "images"
    #: The provision stage of this instance whose ``provisioned.json`` names the agent's data.
    provision_stage: StageName = "provision"
    #: The sandbox role's shape: what the brief tells the agent it has. Its disk caps what the
    #: sandbox's writable directories hold and the repository it submits
    #: (``repository_max_bytes``).
    sandbox_shape: ResourceShape
    #: The shape a graded server of this build runs in, the artifact role's of its evaluation
    #: (``hls.pipeline.pipelines.artifact_shape``, deploy_drive's ``artifact_shape``): the serving
    #: brief states its GPUs, CPUs, memory and writable space, and its per-uid count limits
    #: (``StageContext.served_ucounts``), which the grading seal writes.
    served_shape: ResourceShape
    #: The shape of each harness role beside the sandbox or after it but the commit point: a
    #: snapshot, a copy, the transcript copy. No GPUs and no disk.
    snapshot_shape: ResourceShape
    #: The commit point's shape: its memory is the commit point's resident peak and a quarter more
    #: (``hls.pipeline.pipelines.COMMIT_SHAPE``). No GPUs and no disk; it runs once the sandbox
    #: has stopped.
    commit_shape: ResourceShape
    #: The extraction role's shape (``hls.agent.extract``): its memory is the limit on what
    #: harbor's reading of the agent's logs may hold (``hls.pipeline.pipelines.EXTRACT_SHAPE``). No
    #: GPUs and no disk; it runs once the sandbox has stopped.
    extract_shape: ResourceShape
    #: Resident memory the harness's processes in the sandbox step may hold, in GiB: the OOM
    #: rule's reserve. The harness peaks at about 157.6 MiB while a memory hog fills the sandbox,
    #: so 4 GiB holds it about 25 times over.
    harness_reserve_gib: float = Field(default=4.0, gt=0)
    #: The wall-clock budget the agent is told, in seconds; None when the task sets none (then no
    #: goal, no re-entry and no budget tool).
    soft_sec: int | None = Field(default=None, gt=0)
    #: The wall-clock limit the stage enforces from the same start, in seconds.
    hard_sec: int = Field(gt=0)
    #: The task's turn cap, passed to the CLI; None for none.
    max_turns: int | None = Field(default=None, gt=0)
    #: Seconds the stage's time allows, after its staging, for everything before the agent's clock
    #: starts: the sandbox's start, the workspace's seed and uploads, the seal probe, the preflight
    #: and harbor's setup: about ten times what a preflight-only build, which does all of it,
    #: takes for its whole stage (about 6 minutes), since a build that runs out of it is refused as
    #: infra.
    setup_allowance_sec: int = Field(default=3600, gt=0)
    #: Seconds the stage's time allows before its setup for what it copies off the shared
    #: filesystem: the agent's data and the agent cap's file, which it extracts; a fixed allowance
    #: (``hls.pipeline.pipelines.STAGING_SEC``).
    staging_sec: int = Field(ge=0)
    #: Seconds a full build's stage keeps after the agent's hard limit to collect the build, the
    #: run's ``run.collect_allowance_sec``, which the brief states: the fixed steps
    #: (``COLLECT_STEPS_SEC``) and the commit point, which gets the rest. A collection that does not
    #: end within it is infra that is not retried, and the build is void. The
    #: stage refuses one that leaves the commit point less than ``COMMIT_MIN_SEC``. A probe or
    #: preflight-only build keeps ``STOP_BUILD_COLLECT_SEC`` instead (``collection_sec``).
    collect_allowance_sec: int = Field(gt=0)
    #: Seconds between snapshots of the workspace's main.
    snapshot_interval_sec: int = Field(default=1800, gt=0)
    #: The producer's ``NetworkPolicy``, as JSON: what the sandbox's seal lets through, before the
    #: arm's route host is added. Its package-deny rule is what the sandbox's index refuses.
    network: dict[str, Any]
    #: What each pip step of the package this build's bundle goes to may use, which the brief's
    #: packaging section states: the plan's ``hls.stages.package.pip_limits`` of the package
    #: stage's declared shape, the figures that stage holds its steps to. A stop build, which
    #: packages nothing, states those a package of its task would get.
    package_pip: PipLimits
    #: ``probe`` starts the sandbox, runs the probe and stops, with no agent; ``preflight`` also
    #: runs the arm's preflight; None is a full build.
    stop_after: Literal["probe", "preflight"] | None = None
    #: With a probe, also run a memory hog and record what ended it (``--probe-oom``).
    probe_oom: bool = False
    #: The in-build reference service beside the sandbox, for a producer that runs one
    #: (``AgentProducerSpec.reference_service``, ``hls.stages.reference_service``); None for one that
    #: does not, and left out of the dump then.
    reference_service: ReferenceServiceParams | None = Field(default=None, exclude_if=lambda s: s is None)

    @property
    def shm_gib(self) -> int:
        """The cap of the sandbox's /dev/shm, in GiB: the sandbox's whole memory. Its pages count
        against that memory anyway, so the cap sets no limit the memory does not."""
        return self.sandbox_shape.memory_gib

    @property
    def repository_max_bytes(self) -> int:
        """The most bytes the repository's objects may inflate to together, and main's bundle may
        hold, at a commit point or a snapshot (``hls.agent.commit``): the sandbox's writable disk
        cap, which the brief states as the repository's limit too."""
        return self.sandbox_shape.disk_gib * _GIB

    @property
    def collection_sec(self) -> int:
        """Seconds the stage keeps after the agent's hard limit (``collection_sec``)."""
        return collection_sec(self.stop_after, self.collect_allowance_sec)

    @model_validator(mode="after")
    def _checked(self) -> AgentBuildParams:
        NetworkPolicy.model_validate(self.network)
        if self.soft_sec is not None and self.soft_sec > self.hard_sec:
            raise ValueError(f"soft_sec {self.soft_sec} is over hard_sec {self.hard_sec}")
        if self.probe_oom and self.stop_after != "probe":
            raise ValueError("probe_oom runs only with stop_after='probe'")
        return self


class CommitTry(BaseModel):
    """One run of the commit point."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The role that ran it.
    role: str
    #: How the role ended.
    ending: str
    #: Its exit status; None when it never exited.
    exit_code: int | None
    #: What it wrote: the bundle, or the record that there is nothing to package; None when it
    #: failed.
    wrote: Literal["bundle", "no_submission"] | None
    #: Why it failed, scrubbed: the role's fault and its last line on stderr; None when it wrote.
    reason: str | None


class PidsCounts(BaseModel):
    """The ``max`` count of the sandbox step's ``pids.events``: how often its pids limit was
    reached."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: When the sandbox started; None when it could not be read.
    at_start: int | None
    #: When the agent's part ended; None when it could not be read, or the sandbox had ended
    #: first (its outcome then carries its count).
    at_end: int | None
    #: When the count was first read at its latest rise (POSIX time, the runner's), as the
    #: contestant's evidence dates it (``StageContext.pids_hits``); None when it never rose, could
    #: not be read, or the sandbox had ended first.
    rose_at: float | None


class SealProbe(BaseModel):
    """The probe of the sandbox's seal every build runs before its agent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Each check, in the order it ran: the route reached, a refused host refused, the
    #: interceptor's record of that refusal, the index reached.
    checks: list[probe_mod.ProbeCheck]
    #: Whether every check passed.
    passed: bool
    #: How the node's kernel counts the commands' process ceiling, as ``nproc-accounting`` read it
    #: (``probe.read_accounting``); None when its reading was unclear or failed, which says
    #: nothing of the kernel.
    accounting: Accounting | None
    #: Seconds the probe took, from its first command to its last check.
    sec: float


class Extraction(BaseModel):
    """Harbor's reading of the agent's run from the transcript's copy, in the extraction role
    (``hls.agent.extract``), or why it read nothing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The agent's spend harbor read; None when it read none, or the role failed.
    usage: Usage | None
    #: The sha256 of harbor's trajectory, the ``trajectory`` output, as the role reported it; None
    #: when harbor wrote none, or the role failed.
    trajectory_sha256: Sha256Hex | None
    #: Why the trajectory and spend are unavailable: the role could not start, ran out of its
    #: memory, failed or ran past ``EXTRACT_SEC``, with what it said, scrubbed; None when it ran
    #: to its end.
    unavailable: str | None
    #: Why the trajectory alone is unavailable while the spend was read: it took more than
    #: ``TRAJECTORY_MAX_BYTES``, which the role withheld or the stage refused as reported; None
    #: otherwise.
    trajectory_unavailable: str | None


class BuildRecord(BaseModel):
    """``build.json``: what the build ran and how it ended."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: ArmRecord
    #: The declared reason the arm differs from earlier rollouts; None when it does not.
    arm_change: str | None
    #: The sha256 of the agent cap's image file, which the sandbox ran.
    sandbox_image: str
    #: The tree digest (``hls.agent.uploads.tree_digest``) of the task's uploaded folder, which
    #: the sandbox received, checked against the one the task commits; None for a task that
    #: uploads none.
    upload_sha256: str | None
    #: The wall-clock budget the agent was told, in seconds; None when it was told none.
    soft_sec: int | None
    #: The wall-clock limit the stage enforced, in seconds.
    hard_sec: int
    #: The turn cap; None for none.
    max_turns: int | None
    #: The probe of the sandbox's seal, before the preflight.
    seal_probe: SealProbe
    #: The preflight's readback; None for a probe-only build, which runs none.
    preflight: PreflightResult | None
    #: When the agent's clock started (POSIX time, the runner's); for a probe or preflight-only
    #: build, which starts no clock, when its probe started.
    started_at: float
    #: When the agent's part ended (POSIX time, the runner's); for a probe or preflight-only
    #: build, when its probe ended.
    ended_at: float
    #: Seconds on the agent's clock when its part ended.
    agent_sec: float
    #: How the agent's part ended.
    ending: Ending
    #: Sessions harbor resumed.
    reentries: int
    #: Sessions harbor let end.
    sessions_ended: int
    #: The health check's signal when it stopped re-entry; None when it never did.
    derailed: str | None
    #: The commit point's record; None when it wrote none that can be read, and for a probe or
    #: preflight-only build, which runs none.
    commit: CommitRecord | None
    #: Each run of the commit point, in order.
    commit_tries: list[CommitTry]
    #: The sandbox's memory split when it ended (``SplitEvidence``: its harness reserve, its last
    #: OOM and the split sampled before it); None when its step never reported.
    memory: SplitEvidence | None
    #: The contestant fault the agent's part ended on, with its evidence, for a build that ended
    #: ``contestant_oom``, ``contestant_pids`` or ``disk_limit``; None for every other end. For a
    #: ``contestant_oom`` end the plan does not grade, the zero the stage declared
    #: (``sandbox_oom_zero``).
    fault: FaultRecord | None
    #: The rule the build ran under for a sandbox that ends on its own memory, the plan's
    #: ``run.grade_after_sandbox_oom`` (``grades_after_sandbox_oom``): true
    #: when such a build is graded on its last commit; false when it
    #: scores zero with nothing graded, and this record is then the attempt's evidence, never an
    #: output. Recorded whatever the build's end.
    grade_after_sandbox_oom: bool
    #: The process ceiling the sandbox's commands ran under (``hls.substrate.nproc``).
    process_ceiling: ProcessCeiling
    #: How often the sandbox step's pids limit was reached, when it started and when the agent's
    #: part ended.
    pids_max_events: PidsCounts
    #: Harbor's reading of the agent's trajectory and spend after the commit point; None for a
    #: probe or preflight-only build, which runs no agent, and in the evidence of an attempt whose
    #: commit point wrote nothing.
    extraction: Extraction | None
    #: The probe's record of a probe or preflight-only build; None for a full build.
    probe: probe_mod.ProbeRecord | None = None
    #: For a sandbox that ended on an OOM when the job's own limit was reached, beside the in-build
    #: reference service, what the service, the runner and the sandbox held when its end was judged,
    #: which the shared rule read (``classify_shared_build_end``); None otherwise, and left out of
    #: the record then.
    shared_memory: SharedMemoryEvidence | None = Field(default=None, exclude_if=lambda s: s is None)


def budget_params(producer: Any) -> dict[str, int | None]:
    """``soft_sec``, ``hard_sec`` and ``max_turns`` from a task's producer budgets. Raise
    ValueError for a budget the runner cannot keep: a soft turn count, which no tool reports."""
    hard, soft = producer.hard_budget, producer.soft_budget
    if soft is not None and soft.turns is not None:
        raise ValueError("a soft turn budget has no tool that reports it; the task must not set one")
    return {"soft_sec": soft.wallclock_sec if soft is not None else None, "hard_sec": hard.wallclock_sec,
            "max_turns": hard.turns}


def brief_hardware(graded: Hardware, shape: ResourceShape) -> Hardware:
    """The task's hardware *graded* as the brief's hardware block states it for a sandbox of
    *shape*: its accelerators, and the sandbox's CPUs, memory and writable
    disk. The block states decimal GB where the shape is in GiB, rounded down so neither
    overstates the other (``gb_of_gib``)."""
    return graded.model_copy(update={"memory_gb": gb_of_gib(shape.memory_gib), "cpu_count": shape.cpus,
                                     "storage_gb": gb_of_gib(shape.disk_gib)})


def arm_route_secrets(arm: ArmRecord) -> frozenset[str]:
    """The names of the secrets of *arm*'s route, its credential and its base URL: what an agent of
    the arm is granted, and so what a server it built may hold."""
    return frozenset(adapter_for(arm).grants(arm.route_table).values())


def _ipv4s(host: str) -> list[str]:
    """*host*'s IPv4 addresses now; the host itself for an address; [] when it does not resolve."""
    try:
        ipaddress.ip_address(host)
        return [host]
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None, family=socket.AF_INET, type=socket.SOCK_STREAM)
    except OSError:
        return []
    return sorted({str(info[4][0]) for info in infos})


def seal_policy(network: NetworkPolicy, host: str, *,
                resolve: Callable[[str], list[str]] | None = None) -> SealPolicy:
    """The sandbox's seal: *network* with the arm's route *host* allowed (in an allow list), and
    the host vouched for with its current IPv4 addresses (*resolve*, by default the node's
    resolver), so the interceptor dials only those for it. No other host is opened, whatever keys
    the sandbox holds."""
    resolve = resolve or _ipv4s
    domains = network.domains
    entries = list(domains.entries)
    if domains.mode == "allow" and host not in entries:
        entries.append(host)
    vouched = tuple(dict.fromkeys([host, *resolve(host)]))
    return SealPolicy(network=network.model_copy(update={"domains": domains.model_copy(update={"entries": entries})}),
                      trusted_internal_hosts=vouched)


def skill_folders(producer: Any, base: Path, checkout: Path) -> list[ResolvedSkill]:
    """The skills the producer's ``skills_dir`` names, from the task module's directory *base*
    when relative, judged by the rule a scored command's plan applies too
    (``hls.agent.uploads.task_skills``); none for a producer that names no folder. Raise infra,
    not retried, when the folder lies outside the checkout *checkout*, is no directory, or is
    neither a skill nor a folder of skills: the task's own folder is at fault, and a retry would
    read the same one."""
    if producer.skills_dir is None:
        return []
    try:
        return task_skills(producer.skills_dir, base, checkout)
    except UploadRefused as e:
        raise StageFault.infra(f"producer.skills_dir: {e}", retryable=False) from e


def upload_folder(producer: Any, base: Path, checkout: Path) -> tuple[list[Path], str] | None:
    """The producer's uploaded folders (``upload_dir``) as host directories, with the tree digest
    of their merged view, judged by the rule a scored command's plan applies too
    (``hls.agent.uploads.checked_upload``); None for a producer that uploads none. Raise infra,
    not retried, when a folder lies outside the checkout *checkout* or is no directory, holds a
    link or an entry that is neither a file nor a directory, when two hold the same path, or when
    the merged view has another digest than the one the task commits (``upload_sha256``): the
    contestant would be handed material the task does not name."""
    if producer.upload_dir is None:
        return None
    try:
        return checked_upload(producer.upload_dir, base, checkout, producer.upload_sha256)
    except UploadRefused as e:
        raise StageFault.infra(f"producer.upload_dir: {e}", retryable=False) from e


def snapshot_events(n: int, ending: str, exit_code: int | None, record: CommitRecord | None, pruned: list[str],
                    *, scrub: Callable[[str], str]) -> list[tuple[str, dict[str, Any]]]:
    """The events of snapshot *n*'s end, from how its role ended, its commit record and the
    snapshots pruned: ``build.snapshot``, and ``build.snapshot_over_limit`` beside it when the
    commit program found main past one of its limits, which keeps no bundle (a bundle past
    the package's limit is no submission at the commit point). The reason is the program's, which
    *scrub* cleans of every secret value. A snapshot is evidence: neither event stops the build."""
    status = record.status if record is not None else None
    events = [("build.snapshot", {"n": n, "ending": ending, "exit_code": exit_code, "status": status,
                                  "commit": record.commit if record is not None else None, "pruned": pruned})]
    if record is not None and status == "over_limit":
        events.append(("build.snapshot_over_limit", {"n": n, "reason": scrub(record.reason or "")}))
    return events


def _after_commit(reason: str, kind: str, evidence: dict[str, str]) -> FaultRecord:
    """The infra fault, never retried, of a failure after the commit point wrote *kind*."""
    return infra_fault(f"the build failed after its commit point, which kept its {kind} in the attempt's outputs: "
                       f"{reason}", retryable=False, **{**evidence, "commit_point": kind})


def grades_after_sandbox_oom(plan: Plan) -> bool:
    """Whether an agent build of *plan* whose sandbox ended on its own memory (``contestant_oom``)
    is graded on its last commit: the plan's ``run.grade_after_sandbox_oom``, the setting as it was
    when the instance was planned, never the site's file now. The build's
    end, its brief and its record, and the records' harness-conditions table, read the rule here
    alone."""
    return plan.settings.run.grade_after_sandbox_oom


def sandbox_oom_zero(fault: FaultRecord) -> FaultRecord:
    """The contestant's zero of a build whose sandbox ended on its own memory, by its contestant
    *fault*, under a plan that does not grade such a build (``grades_after_sandbox_oom`` false):
    the fault with the rule named in its reason and its evidence, which the
    instance's failed records carry into its zero."""
    return fault.model_copy(update={
        "reason": f"{fault.reason}; run.grade_after_sandbox_oom is false, so the build scores zero with nothing graded",
        "evidence": {**fault.evidence, "grade_after_sandbox_oom": "false"}})


class _ScrubRecords(logging.Filter):
    """Scrubs a record before any handler sees it: its message with its arguments, every extra,
    rendered to text where it is not a string, and its traceback and stack."""

    def __init__(self, scrubber: Scrubber) -> None:
        super().__init__()
        #: Replaces the values the stage holds.
        self.scrubber = scrubber

    def filter(self, record: logging.LogRecord) -> bool:
        scrub = self.scrubber.scrub
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 -- a record whose arguments do not fit its message
            message = f"{record.msg!r} {record.args!r}"
        record.msg, record.args = scrub(message), None
        for key, value in list(vars(record).items()):
            if key in _RECORD_FIELDS:
                continue
            text = value if isinstance(value, str) else str(value)
            scrubbed = scrub(text)
            if scrubbed != text:
                record.__dict__[key] = scrubbed
        if record.exc_info:
            record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = scrub(record.exc_text)
        if record.stack_info:
            record.stack_info = scrub(record.stack_info)
        return True


def _harbors(name: str) -> bool:
    return name == "harbor" or name.startswith("harbor.")


@contextmanager
def scrubbed_harbor_records(scrubber: Scrubber) -> Iterator[None]:
    """While the block runs, every record of harbor's loggers is scrubbed by *scrubber* when it is
    made, before any handler sees it: harbor logs a command's output and errors as extras, which
    can quote what the agent printed. A logger's filters see only the records it makes itself,
    and harbor makes loggers as it goes, so every harbor logger gets the filter, those made
    during the block too (the logger class is swapped for the block)."""
    scrub = _ScrubRecords(scrubber)
    manager = logging.Logger.manager
    old_class = logging.getLoggerClass()

    class _Scrubbing(old_class):  # type: ignore[misc, valid-type]
        def __init__(self, name: str, level: int = logging.NOTSET) -> None:
            super().__init__(name, level)
            if _harbors(name):
                self.addFilter(scrub)

    logging.getLogger("harbor").addFilter(scrub)
    for name, logger in list(manager.loggerDict.items()):
        if _harbors(name) and isinstance(logger, logging.Logger):
            logger.addFilter(scrub)
    logging.setLoggerClass(_Scrubbing)
    try:
        yield
    finally:
        logging.setLoggerClass(old_class)
        for name, logger in list(manager.loggerDict.items()):
            if isinstance(logger, logging.Logger):
                logger.removeFilter(scrub)


class _Counter:
    """The build's events, and how many of harbor's re-entry events there were."""

    def __init__(self, ctx: StageContext) -> None:
        self.ctx = ctx
        #: ``build.agent_reentered`` events so far.
        self.reentries = 0
        #: ``build.agent_session_ended`` events so far.
        self.sessions_ended = 0

    def emit(self, kind: str, **fields: Any) -> None:
        if kind == "build.agent_reentered":
            self.reentries += 1
        elif kind == "build.agent_session_ended":
            self.sessions_ended += 1
        self.ctx.events.emit(kind, **fields)



def seed_source(images: ImageRuntime, image: PinnedImage) -> Path | None:
    """Where *images*, a runtime that holds the agent cap *image* as a host tree
    (``ImageRuntime.tree_path``), holds its ``/workspace`` to seed the sandbox's workspace from;
    None when there is nothing to copy: no directory there, a link, or an empty directory. Raise
    StageFault (infra, not retried) when the image's tree cannot be read, and ValueError for a
    runtime that holds no host tree."""
    root = images.tree_path(image, "/")
    if root is None:
        raise ValueError("the image runtime holds no image as a host tree; its own role copies /workspace out")
    if root.is_symlink() or not root.is_dir():
        raise StageFault.infra(f"the agent cap's tree {root} is no directory, so the sandbox's workspace cannot be "
                               "seeded from it", retryable=False)
    baked = images.tree_path(image, workspace.WORKSPACE)
    assert baked is not None
    if baked.is_symlink() or not baked.is_dir() or not any(baked.iterdir()):
        return None
    return baked


class _Build:
    """One agent_build attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None or ctx.runtime is None or ctx.cache is None:
            raise RuntimeError("agent_build needs node-local scratch, a container runtime and the cache")
        self.ctx = ctx
        self.params: AgentBuildParams = ctx.params
        #: The attempt's outputs directory.
        self.out = ctx.output_path(BUILD_FILE).parent
        #: Whether a sandbox's end on its own memory is graded on the last commit, by the plan's
        #: rule (``grades_after_sandbox_oom``).
        self.oom_graded = grades_after_sandbox_oom(ctx.plan)
        #: Roles started and not yet waited for or stopped, by name.
        self.running: dict[str, RoleHandle] = {}
        #: The events, counted.
        self.events = _Counter(ctx)
        #: The health check's signal once it stopped re-entry; None until then.
        self.derailed: str | None = None
        #: What harbor's agent records of its run, filled after it.
        self.context: Any = None
        #: The sandbox's outcome once the stage reported it; None before.
        self.sandbox_outcome: RoleOutcome | None = None
        #: The sandbox step's pids limit count when it started, and when the agent's part ended.
        self.pids = PidsCounts(at_start=None, at_end=None, rose_at=None)
        #: The launches of the agent's CLI the sandbox's exec server had logged when the agent's
        #: clock started (``cli_launches``); None before, or when its log could not be read.
        self.launches_at_clock: int | None = None
        #: When the runtime started the sandbox's command, its exec server, by the stage's clock
        #: (``command_started``); None until then, or when the sandbox ended or the stage stopped
        #: first.
        self.command_began: float | None = None
        #: The secrets the transcript role scrubs, by the variable each reaches it in, to the
        #: secret's name.
        self.scrubbed: dict[str, str] = {}
        #: The key of what the commit point wrote, the bundle or the no-submission record, by its
        #: own report; None until it wrote one.
        self.collected: str | None = None
        #: The host the sandbox's seal opens for the arm's route (``route_host``); empty until
        #: the build reads it.
        self.host = ""
        #: The process ceiling the sandbox's commands run under; None until the build reads it.
        self.ceiling: ProcessCeiling | None = None
        #: The limit the sandbox's commands' user namespace gets in each per-uid kernel count, by
        #: its ``/proc/sys/user`` file's name (``StageContext.ucount_limits``); empty until the
        #: build reads them.
        self.ucounts: dict[str, int] = {}
        #: The sandbox's env file as the stage wrote it, for the probe's check of it; empty until
        #: the sandbox starts.
        self.sandbox_env: dict[str, str] = {}
        #: The sandbox's seal, whose namespace's own ports the seal probe's inbound check avoids;
        #: None until the sandbox starts.
        self.sandbox_seal: SealPlan | None = None
        #: The sandbox's layout, where the inbound check keeps its files; None until the sandbox
        #: starts.
        self.sandbox_layout: workspace.SandboxLayout | None = None
        #: The probe of the sandbox's seal, once it ran.
        self.seal: SealProbe | None = None
        #: The in-build reference service, for a build that runs one; None until the layout is
        #: made, and for a build without one.
        self.service: BuildService | None = None
        #: What the service, the runner and the sandbox held when the sandbox's own end was judged
        #: beside the service (``_sandbox_fault``); None when it was not.
        self.shared_memory: SharedMemoryEvidence | None = None
        self._snapshots = 0
        self._snapshot_due = 0.0
        #: The snapshot role whose step is being started in a worker thread, its name and the
        #: future of its launch; None when none is in flight.
        self._snapshot_launch: tuple[str, Future[RoleHandle]] | None = None

    # roles

    def _shape(self, shape: ResourceShape) -> ResourceShape:
        """*shape* for the stage's own time, which a site's ``[resources]`` table may change."""
        return shape.model_copy(update={"time_sec": self.ctx.stage.shape.time_sec})

    def _host(self, name: str, argv: list[str], *, env: dict[str, str] | None = None,
              secret_env: tuple[str, ...] = (), daemon: bool = False) -> RoleHandle:
        """Start host harness role *name* with a harness role's shape, on CPUs the sandbox's step
        holds (``overlap``). One given secrets logs into the attempt, scrubbed in place when it
        is reported (``scrub_logs``), never onto the node's disk, which the agent may have
        filled. With *daemon* it speaks to the node's container daemon (``RoleSpec.daemon``)."""
        handle = self.ctx.start_role(name, argv, self._shape(self.params.snapshot_shape), owner="harness", env=env,
                                     secret_env=secret_env, overlap=True, scrub_logs=bool(secret_env), daemon=daemon)
        self.running[name] = handle
        return handle

    def _container(self, name: str, spec: ContainerSpec, shape: ResourceShape) -> RoleHandle:
        """Start harness container *name* on CPUs the sandbox's step holds (``overlap``)."""
        handle = self.ctx.start_container(name, spec, self._shape(shape), owner="harness", overlap=True)
        self.running[name] = handle
        return handle

    def _wait(self, name: str, command_sec: float | None = None) -> RoleOutcome:
        """Wait for role *name* alone and return how it ended. The stage watches the role until
        it ends, and only then has the context report it: the context stops every role it waits
        for once another has faulted, and a snapshot's failure must not stop a later role. A
        shutdown or the stage's deadline still ends the role, through the context's own wait, and
        with *command_sec* so does its command's running that long since the stage saw it start
        (``StageContext.wait_roles``): the role ends ``time_limit`` either way."""
        handle = self.running.pop(name)
        ctx = self.ctx

        def due() -> float:
            began = None if command_sec is None else ctx.command_started_at(name)
            return ctx.deadline if began is None else min(ctx.deadline, began + command_sec)

        while handle.poll() is None and not ctx.stopping and ctx.clock() < due():
            handle.wait(TICK_SEC)
        [outcome] = ctx.wait_roles(name, command_sec=command_sec)
        return outcome

    def _stop(self, name: str) -> RoleOutcome:
        """Stop role *name* as planned, or report its own end if it ended first."""
        self.running.pop(name, None)
        return self.ctx.stop_role(name)

    def _stop_all(self) -> None:
        if self.service is not None:
            self.service.stop()
        for name in list(self.running):
            self._stop(name)

    @staticmethod
    def _check(outcome: RoleOutcome) -> None:
        """Raise the fault of a role that had to succeed."""
        fault = stage_fault([outcome])
        if fault is not None:
            raise StageFault(fault)

    # inputs

    def _manifest(self, stage: str) -> Manifest:
        key = self.ctx.plan.layout.manifest(stage)
        manifest = self.ctx.inputs.get(key)
        if manifest is None:
            raise StageFault.infra(f"agent_build reads this instance's {stage} manifest {key}; the stage was planned "
                                   "without it", retryable=False)
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
        """This instance's provision record, checked against its manifest."""
        return Provisioned.model_validate_json(self._record(self._manifest(self.params.provision_stage),
                                                            "provisioned"))

    def image(self) -> PinnedImage:
        """The agent cap, extracted on the node and pinned to the entry digest this instance's
        ``images.json`` records for it; the digest of its file is recorded for the manifest."""
        pinned = self._node_image(self.params.sandbox_image)
        self.ctx.record_image(SANDBOX, pinned.file_sha256)
        return pinned

    def service_image(self) -> PinnedImage | None:
        """The reference service's image, the task's measured baseline image, pinned as the agent
        cap is; None for a build without the service."""
        planned = self.params.reference_service
        return None if planned is None else self._node_image(planned.image)

    def _node_image(self, name: str) -> PinnedImage:
        """Image *name* of this instance's ``images.json``, extracted on the node and pinned to the
        entry digest it records."""
        manifest = self._manifest(self.params.images_stage)
        try:
            entry = json.loads(self._record(manifest, IMAGES_OUTPUT))[name]
            key, expect = check_segment(entry["key"]), check_sha256_hex(entry["entry_sha256"])
        except (ValueError, KeyError, TypeError) as e:
            raise StageFault.infra(f"{manifest.outputs[IMAGES_OUTPUT].key} names no image {name} with a key "
                                   "and an entry digest", retryable=False) from e
        return self.ctx.node_image(key, expect)

    def artifact(self) -> Any:
        """The task module's ``ARTIFACT``: its producer's uploads, skills and contract, the brief's
        template, and whether the build runs the reference service, which the brief states only
        then. A producer and a plan that disagree on the service are refused, infra not retried,
        before any role: a brief that promises the service never runs without it."""
        import importlib

        artifact = getattr(importlib.import_module(self.params.task.module), "ARTIFACT", None)
        producer = getattr(artifact, "producer", None)
        if producer is None:
            raise StageFault.infra(f"{self.params.task.module} declares no ARTIFACT with a producer", retryable=False)
        planned = self.params.reference_service is not None
        if producer.reference_service != planned:
            asks = "asks for" if producer.reference_service else "runs no"
            plans = "launches one" if planned else "launches none"
            raise StageFault.infra(f"{self.params.task.module}'s producer {asks} reference service beside the build, "
                                   f"and the stage's plan {plans}", retryable=False)
        return artifact

    def service_engine(self, task: Any) -> rs.ServiceEngine | None:
        """The engine the reference service builds, the task's pinned vLLM reference's
        (``service_harness.service_engine_for``); None for a build without the service. A task
        without exactly one, or one that does not span the GPUs the service shares, is refused,
        infra not retried, before any role."""
        from hls.verify.base import WiringError

        planned = self.params.reference_service
        if planned is None:
            return None
        try:
            engine = rs.service_engine_for(task)
        except WiringError as e:
            raise StageFault.infra(str(e), retryable=False) from e
        if engine.tensor_parallel_size != planned.shape.gpus:
            raise StageFault.infra(f"the reference service's engine spans {engine.tensor_parallel_size} GPUs, and it "
                                   f"shares the sandbox's {planned.shape.gpus}", retryable=False)
        return engine

    def route_host(self) -> str:
        """The host the sandbox's seal opens for the arm's route: that of the route's base URL,
        whose value the stage holds as its grant, or the harness's own host when the route names
        none. A base URL grant the stage holds no value for is infra that is not retried, before
        any role: the harness's own host would be opened in its place. The value is never shown."""
        p, ctx = self.params, self.ctx
        adapter = adapter_for(p.arm)
        if p.arm.route_table.base_url is None:
            return adapter.route_host(None)
        try:
            value = ctx.secrets[adapter.base_url_var]
        except KeyError:
            raise StageFault.infra(f"arm {p.arm.name}: the stage holds no value for {adapter.base_url_var}, the "
                                   f"route's base URL secret {p.arm.route_table.base_url}", retryable=False) from None
        try:
            return adapter.route_host(value)
        except ValueError as e:
            raise StageFault.infra(f"arm {p.arm.name}: {e}", retryable=False) from None

    # the sandbox's files

    def layout(self) -> workspace.SandboxLayout:
        """The sandbox's layout, made in its role's own directory, which the context made
        afresh. A node-local scratch so deep that a socket's path would not fit the kernel's
        limit is refused first."""
        root = self.ctx.role_dir(SANDBOX)
        planned = workspace.SandboxLayout(root)
        sockets = [planned.socket, planned.budget_socket]
        if self.params.reference_service is not None:
            sockets.append(root / SERVICE_SOCKET)
        long = [f"{p} ({len(os.fsencode(p))} bytes)" for p in sockets if len(os.fsencode(p)) > MAX_SOCKET_PATH_BYTES]
        if long:
            raise StageFault.infra(f"the site's node-local scratch is too deep for the sandbox's sockets: "
                                   f"{', '.join(long)}, over the {MAX_SOCKET_PATH_BYTES} bytes a socket's path may "
                                   "hold; set site.node_scratch to a shorter path", retryable=False)
        return workspace.SandboxLayout.create(root)

    def seed(self, layout: workspace.SandboxLayout, image: PinnedImage) -> None:
        """Copy what the agent cap holds at ``/workspace`` into the sandbox's workspace, by a
        role, before anything else lands there: the workspace's bind hides the image's own. An
        image runtime that holds the image as a host tree has it read there (``seed_source``), by
        ``cp -a``, which keeps modes and times, so a build tree baked into the image stays
        current; another writes it into the workspace by a role of its own
        (``ImageRuntime.tree_role``). An image with no directory at ``/workspace``, or an empty
        one, copies nothing; one whose tree cannot be read, or that the runtime's role cannot
        read, is refused (infra)."""
        images = self.ctx.images
        assert images is not None
        started = monotonic()
        if images.tree_path(image, "/") is None:
            own = self.ctx.role_dir(SEED)
            argv, env = images.tree_role(image, workspace.WORKSPACE, layout.workspace, own=own)
            self._host(SEED, argv, env=env, daemon=True)
            outcome = self._wait(SEED)
            # The image holds no directory at /workspace: there is nothing to copy.
            if self.ctx.refusal(SEED, [outcome], (TREE_ABSENT_EXIT,)) is not None:
                return
            self._check(outcome)
            self.events.emit("build.workspace_seeded", sec=round(monotonic() - started, 3))
            return
        baked = seed_source(images, image)
        if baked is None:
            return
        self._host(SEED, ["cp", "-a", "--", f"{baked}/.", str(layout.workspace)])
        self._check(self._wait(SEED))
        self.events.emit("build.workspace_seeded", sec=round(monotonic() - started, 3))

    def uploads(self, layout: workspace.SandboxLayout, artifact: Any) -> tuple[str | None, str | None]:
        """Copy the task's uploaded folders and skills into the sandbox's directories, by one
        role, once the folders' merged view is the one the task commits (``upload_folder``) and
        lands where the agent sees it (``hls.agent.uploads.upload_destination``), each folder
        copied into that one destination, and the skills folder is one skill or a folder of them
        (``skill_folders``); return the sandbox directory of the skills, or None without any, and
        the merged view's tree digest, or None without an upload. A task folder or destination
        refused is infra, not retried, before any role."""
        producer, base, checkout = artifact.producer, task_dir(self.params.task), self.ctx.plan.code.checkout
        copies: list[tuple[Path, Path]] = []
        checked = upload_folder(producer, base, checkout)
        uploads, upload_sha256 = checked if checked is not None else ([], None)
        if uploads:
            try:
                under, below = upload_destination(producer.upload_dest)
            except UploadRefused as e:
                raise StageFault.infra(f"producer.upload_dest: {e}", retryable=False) from e
            copies += [(upload, layout.host_path(under) / below) for upload in uploads]
        skills = skill_folders(producer, base, checkout)
        copies += [(s.source, layout.task / "skills" / s.name) for s in skills]
        if not copies:
            return None, None
        for _, dest in copies:
            dest.mkdir(parents=True, exist_ok=True)
        script = 'while [ "$#" -gt 1 ]; do cp -rL -- "$1/." "$2" || exit 1; shift 2; done'
        self._host(UPLOADS, ["/bin/sh", "-c", script, UPLOADS, *(str(x) for pair in copies for x in pair)])
        self._check(self._wait(UPLOADS))
        return (workspace.SKILLS if skills else None), upload_sha256

    def brief(self, task: Any, artifact: Any) -> str:
        """Render and write the brief with the prompt module (``hls.prompt``), with the sandbox role's own
        figures: the hardware's memory, CPUs and storage are its declared ones, and the
        budget section states the wall-clock budget the agent is told and where it reads what is
        left (``budget_brief_section``), the sandbox's figures with its disk cap, the paths it may
        write, the process ceiling its commands run under and the plan's rule for a sandbox that
        ends on its memory (``oom_graded``), then the figures a graded server
        gets (``served_shape``, the shape its grading runs it in) and the paths it may write, and
        what a graded drive holds its replies to: the task's trace cap and the cap on one reply
        (``drive_bounds_brief_section``), read off the task and ``hls.drive.bounds``. The
        submission section is the git repository's for the producer's server file, the only
        submission the package stage builds; the packaging section states the time the harness has to
        collect the repository after the hard limit (``collect_allowance_sec``, a stop build's too,
        which states what a full build of its plan gets) and the commit point's least part of it
        (``commit_point_sec``), the package's limits and the pip steps' disk and memory as the plan
        states them (``package_pip``), with the versions the agent cap's base holds. The ceiling is
        read before the brief (``process_ceiling``), and the
        brief states the value the spawner gives every command; the graded server's per-uid count
        limits are the ones its grading seal writes (``StageContext.served_ucounts``), each of the
        counts the site's substrate holds a graded server to (``Substrate.served_ucounts``)."""
        if self.ceiling is None:
            raise RuntimeError("the brief states the commands' process ceiling, which is read before it")

        producer, p = artifact.producer, self.params
        shape = p.sandbox_shape
        graded = task.hardware
        task = task.model_copy(update={"hardware": brief_hardware(graded, shape)})
        budget = budget_brief_section(p.soft_sec) if p.soft_sec is not None else ""
        sandbox = sandbox_brief_section(memory_gib=shape.memory_gib, harness_gib=p.harness_reserve_gib,
                                        shm_gib=p.shm_gib, cpus=shape.cpus, disk_gib=shape.disk_gib,
                                        chain_gib=CHAIN_BYTES_MAX // _GIB,
                                        writable=workspace.AGENT_WRITABLE, command_nproc=self.ceiling.ceiling,
                                        nproc_accounting=self.ctx.nproc_accounting().accounting,
                                        ucounts=self.ucounts, grade_after_sandbox_oom=self.oom_graded)
        served = p.served_shape
        # Computed once, here, and recorded beside the brief: the grading seal writes these.
        held = self.ctx.served_ucounts(served)
        self.ctx.write_output(SERVED_UCOUNTS_FILE, held.model_dump_json(indent=2).encode())
        enforced = substrate(self.ctx.plan.settings.site).served_ucounts
        serving = serving_brief_section(gpus=served.gpus, gpu_kind=graded.accelerator.kind, cpus=served.cpus,
                                        memory_gib=served.memory_gib, writable_gib=served.disk_gib,
                                        served_writable=SERVED_WRITABLE,
                                        ucounts={k: v for k, v in held.figures().items() if k in enforced})
        # A task that declares no drive bounds cannot be graded, so its brief states none.
        drive = (drive_bounds_brief_section(trace_cap_bytes=task.drive_bounds.trace_cap_bytes,
                                            reply_max_bytes=bounds.REPLY_MAX_BYTES,
                                            verify_limited=task.drive_bounds.verify_limited)
                 if task.drive_bounds is not None else "")
        text, provenance = artifact.brief.render_with_provenance(
            taskspec=task, server_path=f"{workspace.WORKSPACE}/{producer.server_rel_path}",
            submission_section=SubmissionContract(server_filename=producer.server_rel_path).brief_section(),
            packaging_section=packaging_brief_section(held=p.sandbox_held,
                                                      pip_disk_gib=p.package_pip.disk_gib,
                                                      pip_memory_gib=p.package_pip.memory_gib,
                                                      collect_sec=p.collect_allowance_sec,
                                                      bundle_sec=commit_point_sec(p.collect_allowance_sec)),
            budget_section="\n\n".join(s for s in (budget, sandbox, serving, drive) if s),
            uploads_section=(uploads_brief_section(producer.upload_brief, producer.upload_dest)
                             if producer.upload_dir is not None else ""),
            blocked_packages=blocked_packages_of(NetworkPolicy.model_validate(p.network)),
            reference_service=producer.reference_service)
        self.ctx.write_output(BRIEF_FILE, text.encode())
        self.ctx.write_output(BRIEF_PROVENANCE_FILE, provenance.model_dump_json(indent=2).encode())
        return text

    # the sandbox

    def data(self, provisioned: Provisioned) -> tuple[list[Mount], dict[str, str]]:
        """The agent role's data mounts and their variables, each cache entry copied to the node
        now (``materialize``), with the dev reference the plan pins beside each tenant's dev rows
        (``hls.provision.mounts.dev_view``). The copies must be those the plan declared (``data_copies``), or the
        sandbox, which the node's start check passed for the allocation's disk, could find its
        scratch short: a copy the plan did not declare, or one that takes more (an entry published
        again since the plan), is infra not retried, before anything is copied (``data_mounts``)."""
        try:
            files = dev_files(self.ctx)
            return data_mounts("agent", provisioned, self.ctx.materialize, declared=self.params.data_copies,
                               dev_files=files, views=self.ctx.role_dir(SANDBOX) / "dev" if files else None)
        except ValueError as e:
            raise StageFault.infra(f"the agent's data cannot be mounted: {e}", retryable=False) from e
        except OSError as e:
            raise StageFault.infra(f"the agent's view of the dev rows: {e}", retryable=True) from e

    @property
    def records(self) -> Path:
        """The attempt's directory of the commit point's records, on the store."""
        return self.ctx.attempt_dir / COMMIT_RECORDS

    @property
    def extract_record(self) -> Path:
        """Where the extraction role writes its record, in the attempt on the store."""
        return self.ctx.attempt_dir / EXTRACT / "record.json"

    def prepare_collection(self) -> None:
        """Make every node-local file of the commit point's tries, of the transcript role and of
        the extraction role before the sandbox starts: their role directories with their steps'
        ``tmp`` and their log cappers' error files (``StageContext.role_dir``), the commit
        containers' workdirs with the directories the runtime makes there and their env files,
        the transcript role's env file, and each one's outputs report, given
        ``OUTPUTS_REPORT_BYTES`` of the disk, which it writes in place. A contestant that fills
        the node's disk then leaves them nothing to make there; their logs and reports, their
        records, their fresh repositories and their outputs go to the store, the transcript
        role's logs scrubbed in place (``_host``). The snapshots are not prepared: a snapshot's
        failure is only an event."""
        ctx = self.ctx
        assert ctx.runtime is not None
        for n in range(1, COMMIT_TRIES + 1):
            name = commit_role(n)
            workdir = ctx.role_dir(name) / "wd"
            for part in ctx.runtime.workdir_dirs:
                (workdir / part).mkdir(parents=True)
            (self.records / name / "tmp").mkdir(parents=True)
            self._preallocate(ctx.outputs_report(name))
            ctx.prepare_container_env(name, {})
        self.scrubbed = {var: ctx.stage.grants.get(var, var) for var in sorted(ctx.secrets)}
        if self.scrubbed:
            ctx.prepare_role_env(TRANSCRIPT, tuple(self.scrubbed))
        else:
            ctx.role_dir(TRANSCRIPT)
        self._preallocate(ctx.outputs_report(TRANSCRIPT))
        self._preallocate(ctx.outputs_report(EXTRACT))
        self.extract_record.parent.mkdir()

    @staticmethod
    def _preallocate(path: Path) -> None:
        # Filled rather than zeroed: a filesystem that compresses would keep zeros as a hole.
        with open(path, "wb") as f:
            f.write(b" " * OUTPUTS_REPORT_BYTES)
            f.flush()
            os.fsync(f.fileno())

    def start_sandbox(self, layout: workspace.SandboxLayout, image: PinnedImage, data: list[Mount],
                      data_env: dict[str, str], token: str, host: str,
                      ceiling: ProcessCeiling) -> tuple[RoleHandle, dict[str, Source]]:
        """Start the sandbox with the agent's *data* mounts and their variables *data_env*, its
        seal opening the route's *host*, its commands under *ceiling*, under the ``spawner``
        confinement its spawner's namespaces need, which no other container of any stage asks
        for; return its handle and where each variable of its env file came from. The env file
        holds no secret: the arm's route reaches the CLI over the exec channel."""
        p = self.params
        env = {**data_env, exec_protocol.TOKEN_ENV: token}
        if p.sandbox_shape.gpu_kind is not None:
            env["TORCH_CUDA_ARCH_LIST"] = COMPUTE_CAPABILITY[p.sandbox_shape.gpu_kind]
        # The reference service's socket and its path, the sandbox's only part of the service.
        service = (self.service.mount(),) if self.service is not None else ()
        if self.service is not None:
            env.update(self.service.env())
        network = NetworkPolicy.model_validate(p.network)
        policy = seal_policy(network, host)
        seal = SealPlan(policy=policy, index=True, deny_list=deny_list(network))
        command = exec_protocol.server_command(shm_bytes=p.shm_gib * _GIB, uid=os.getuid(), gid=os.getgid(),
                                               cli_launches=(adapter_for(p.arm).cli_launch,),
                                               command_nproc=ceiling.ceiling, ucounts=self.ucounts)
        spec = ContainerSpec(image=image, command=command, workdir=layout.workdir,
                             mounts=(*layout.mounts(), *data, *service), gpus=p.sandbox_shape.gpus > 0,
                             cwd=workspace.WORKSPACE, seal=seal, confinement="spawner")
        handle = self.ctx.start_container(SANDBOX, spec, self._shape(p.sandbox_shape), owner="contestant", env=env,
                                          harness_reserve_gib=p.harness_reserve_gib, command_nproc=ceiling.ceiling,
                                          scrub_logs=True, ready_source=partial(probe_mod.ready_pids, layout.ready))
        self.running[SANDBOX] = handle
        self.sandbox_env = env
        self.sandbox_seal, self.sandbox_layout = seal, layout
        hits = self.ctx.pids_hits(SANDBOX)
        self.pids = PidsCounts(at_start=hits.count if hits is not None else None, at_end=None, rose_at=None)
        sources: dict[str, Source] = {k: "plain" for k in env}
        sources[exec_protocol.TOKEN_ENV] = "harness"
        sources.update({k: "seal" for k in client_env(seal)})
        return handle, sources

    def _stop_service_on_the_way_out(self) -> None:
        """Stop the reference service, if the end of the agent's part has not, before the stage
        goes on or raises. A failure of the stop is logged: the attempt's own end stands, and the
        context's close stops every role left."""
        if self.service is None:
            return
        try:
            self.service.stop()
        except Exception:  # noqa: BLE001 -- the context's close is the backstop
            log.exception("the reference service could not be stopped; the stage's close stops it")

    def start_service(self, sandbox: RoleHandle, image: PinnedImage | None, data: list[Mount]) -> None:
        """Start the reference service on the GPUs of the *sandbox*, once it has reported with the
        grant it declared, on its pinned *image*, with the weights of the agent's *data*
        (``BuildService.start``). A sandbox that did not start cleanly starts no service: its own
        end decides the build."""
        if self.service is None:
            return
        assert image is not None and self.params.reference_service is not None
        report = self.ctx.report(SANDBOX)
        if report is None or report.mismatches:
            self.events.emit("build.reference_service_unreachable", where="start", startup_sec=None,
                             reason="the sandbox did not start with its grant, so the service was not started")
            return
        self.service.start(sandbox, image, data, self.ctx.plan.code.checkout,
                           self._shape(self.params.reference_service.shape))

    def agent_env(self, container: dict[str, Source], agent: Any) -> None:
        """Write ``agent_env.json``: the name and source of each variable of the sandbox's env
        file (``container``) and of those harbor sends with the exec requests (``exec``): the
        arm's route, a grant as ``grant:<its secret's name>``, its plain settings, and the
        variables harbor itself sets in *agent*'s launch (``harness``, the adapter's
        ``launch_env``). Never a value."""
        p = self.params
        adapter = adapter_for(p.arm)
        requests: dict[str, str] = dict.fromkeys(adapter.launch_env(agent), "harness")
        requests.update({**{var: f"grant:{name}" for var, name in adapter.grants(p.arm.route_table).items()},
                         **dict.fromkeys(adapter.plain_env(p.arm), "plain")})
        record = {"container": dict(sorted(container.items())), "exec": dict(sorted(requests.items()))}
        self.ctx.write_output(AGENT_ENV_FILE, json.dumps(record, indent=2).encode())

    # the contestant's evidence

    def _memory_kill_times(self) -> list[float]:
        """When the runner killed in the sandbox's step so far (POSIX time, the watch's): each of
        its kills of single contestant processes, and its kill of the whole step."""
        ctx = self.ctx
        report = ctx.report(SANDBOX)
        if ctx.watch is None or report is None:
            return []
        breach = ctx.watch.breach(report.step)
        return [k.at for k in ctx.watch.kills(report.step)] + ([breach.at] if breach is not None else [])

    def _evidence(self, failed_at: float) -> FaultRecord | None:
        """The contestant's evidence for a failure seen at *failed_at*, read while the sandbox
        still runs, before any ending rule: its step's pids limit count risen since it started,
        or the runner's memory kills in it, as the contestant fault the agent's part then ended
        on; None when it shows nothing. Only evidence within ``OOM_EVIDENCE_MAX_AGE_SEC`` of the
        failure counts (``within_evidence_age``), as the build rule bounds an OOM's: a fork bomb
        or a memory kill the sandbox survived long before says nothing about a later failure,
        which is then judged without it. The count and when it rose go into ``pids``."""
        hits = self.ctx.pids_hits(SANDBOX)
        start = self.pids.at_start or 0
        self.pids = self.pids.model_copy(update={"at_end": hits.count if hits is not None else None,
                                                 "rose_at": hits.rose_at if hits is not None else None})
        if hits is not None and hits.count > start and within_evidence_age(hits.rose_at, failed_at):
            return FaultRecord(fault_class="contestant", retryable=False, evidence={
                "roles": SANDBOX, "ending": "contestant_pids", "pids_at_start": str(self.pids.at_start),
                "pids_at_end": str(hits.count), "pids_rose_at": str(hits.rose_at), "failed_at": str(failed_at)},
                reason=f"build sandbox {SANDBOX} reached its step's process limit {hits.count - start} times, the "
                       f"last {failed_at - (hits.rose_at or failed_at):.0f} s before its failure")
        kills = [at for at in self._memory_kill_times() if within_evidence_age(at, failed_at)]
        if kills:
            return FaultRecord(fault_class="contestant", retryable=False,
                               evidence={"roles": SANDBOX, "ending": "contestant_oom", "memory_kills": str(len(kills)),
                                         "last_kill_at": str(max(kills)), "failed_at": str(failed_at)},
                               reason=f"the runner's memory watch ended processes of build sandbox {SANDBOX} "
                                      f"{len(kills)} times at its memory limit within {OOM_EVIDENCE_MAX_AGE_SEC:.0f} s "
                                      "before its failure")
        return None

    # the agent's part

    async def drive(self, env: SandboxEnvironment, agent: Any, clock: WallClock, brief: str, sandbox: RoleHandle,
                    layout: workspace.SandboxLayout,
                    image: PinnedImage) -> tuple[str, BaseException | None, float]:
        """Run the agent until its part ends; return how (``agent_ended``, ``hard_limit``,
        ``sandbox_ended``, ``stopping``), what ``run`` raised and when the drive saw its part end
        (POSIX time), before it waits for anything: the time the contestant's evidence is read
        against (``_evidence``). A sandbox seen to have ended is reported at once, before
        anything else of that tick, and so is one that ended as harbor's run did (a lost channel
        ends the run first), so its end's time is its own."""
        from harbor.models.agent.context import AgentContext

        ctx, p = self.ctx, self.params
        self.context = AgentContext()
        task = asyncio.ensure_future(agent.run(brief, env, self.context))
        # The reference service's readiness and its probe from the sandbox, beside the agent.
        watch = asyncio.ensure_future(self.service.watch(env)) if self.service is not None else None
        beat = ctx.plan.settings.run.heartbeat_sec
        next_beat = monotonic() + beat
        self._snapshot_due = monotonic() + p.snapshot_interval_sec
        how, ended_at = "agent_ended", None
        try:
            while not task.done():
                if sandbox.poll() is not None:
                    ended_at = time.time()
                    self._report_sandbox()
                    how = "sandbox_ended"
                    break
                if clock.hard_remaining_sec() <= 0:
                    how = "hard_limit"
                    break
                if ctx.stopping:
                    how = "stopping"
                    break
                now = monotonic()
                if now >= next_beat:
                    self.heartbeat(layout, clock)
                    next_beat = now + beat
                self.snapshots(layout, image, now)
                if self.service is not None:
                    self.service.tick()
                await asyncio.wait({task}, timeout=TICK_SEC)
        finally:
            if ended_at is None:
                ended_at = time.time()
            if watch is not None:
                watch.cancel()
                [watched] = await asyncio.gather(watch, return_exceptions=True)
                if isinstance(watched, Exception):
                    # The watch only records; its own failure must never end the build.
                    self.events.emit("error", where="reference_service",
                                     msg=self.ctx.scrubber.scrub(repr(watched))[:300])
            if not task.done():
                task.cancel()
            done = await asyncio.gather(task, return_exceptions=True)
            self._report_if_ended(env, sandbox)
        error = done[0] if isinstance(done[0], BaseException) else None
        if isinstance(error, asyncio.CancelledError) and how != "agent_ended":
            error = None
        return how, error, ended_at

    def _report_if_ended(self, env: SandboxEnvironment, sandbox: RoleHandle) -> None:
        """Report the sandbox at once when it has ended, before harbor's environment is closed;
        when its exec channel was lost, give it ``LOST_WAIT_SEC`` to end first, since a channel
        lost with the sandbox ends harbor's run before the container has ended."""
        if env.lost is not None and sandbox.poll() is None:
            sandbox.wait(LOST_WAIT_SEC)
        if sandbox.poll() is not None:
            self._report_sandbox()

    def _report_sandbox(self) -> RoleOutcome:
        """Have the context report the sandbox, which has ended, once."""
        if self.sandbox_outcome is None:
            self.running.pop(SANDBOX, None)
            [self.sandbox_outcome] = self.ctx.wait_roles(SANDBOX)
        return self.sandbox_outcome

    def heartbeat(self, layout: workspace.SandboxLayout, clock: WallClock) -> None:
        """Emit ``build.heartbeat``, and stop re-entry for a session the health check calls derailed.
        Reading the agent's files must never end the build: whatever the check raises is an
        ``error`` event, and the drive goes on."""
        try:
            health = check_health(layout.agent_logs, adapter_for(self.params.arm))
        except Exception as e:  # noqa: BLE001 -- the agent wrote what the check reads
            self.events.emit("error", where="health", msg=self.ctx.scrubber.scrub(repr(e))[:300])
            return
        self.events.emit("build.heartbeat", log_bytes=health.log_bytes, last_turn_at=health.last_turn_at,
                         turns=health.turns, remaining_sec=round(clock.remaining_sec(), 1),
                         hard_remaining_sec=round(clock.hard_remaining_sec(), 1))
        if health.derailed and self.derailed is None:
            self.derailed = health.signal
            clock.stop_reentry()
            self.events.emit("build.agent_derailed", signal=health.signal, count=health.count, window=health.turns)

    def _reap_snapshot(self, *, stop: bool) -> None:
        """Report the running snapshot when it has ended, or stop it with *stop*; remove the
        partial bundle a snapshot killed while it bundled leaves in the snapshots, which its
        program cannot remove once killed; then keep the newest two and emit its events
        (``snapshot_events``): ``build.snapshot``, and ``build.snapshot_over_limit`` for one past
        a limit."""
        name = f"snapshot-{self._snapshots}"
        if name not in self.running or (not stop and self.running[name].poll() is None):
            return
        outcome = self._stop(name)
        for unfinished in self.snapshot_dir.glob(".tmp-*"):
            remove_tree(unfinished)
        record = self._commit_record(self.ctx.role_dir(name) / "record" / "commit.json")
        pruned = prune_snapshots(self.snapshot_dir)
        for kind, fields in snapshot_events(self._snapshots, outcome.ending, outcome.exit_code, record, pruned,
                                            scrub=self.ctx.scrubber.scrub):
            self.events.emit(kind, **fields)

    def snapshots(self, layout: workspace.SandboxLayout, image: PinnedImage, now: float) -> None:
        """Reap a finished snapshot, collect a finished launch, and start the next launch when it
        is due and none runs or is launching. Starting a step waits for the scheduler to grant it
        (``site.step_launch_allowance_sec``), so the launch runs
        in a worker thread (``_launch_snapshot``): harbor's I/O and the hard-limit check never wait
        on it. One that cannot start, its directories refused on a disk the agent filled or its
        container or step refused, is skipped: the ``build.snapshot_skipped`` event and a warning
        say why, and the next is tried when due. A snapshot runs to the stage's deadline at most
        (``--seconds``); the end of the agent's part stops it before then."""
        self._reap_snapshot(stop=False)
        self._collect_snapshot_launch(wait=False)
        if (now < self._snapshot_due or self._snapshot_launch is not None
                or any(n.startswith("snapshot-") for n in self.running)):
            return
        self._snapshots += 1
        self._snapshot_due = now + self.params.snapshot_interval_sec
        name = f"snapshot-{self._snapshots}"
        try:
            record = self.ctx.role_dir(name) / "record"
            record.mkdir()
            wd = self.ctx.role_dir(name) / "wd"
            wd.mkdir()
            command = commit_command(workspace=workspace.WORKSPACE, out=f"{OUT}/{snapshot_name(self._snapshots)}",
                                     max_bytes=self.params.repository_max_bytes,
                                     seconds=max(1, math.floor(self.ctx.deadline - self.ctx.clock())),
                                     record=f"{RECORD}/commit.json")
            mounts = (Mount(layout.workspace, workspace.WORKSPACE), Mount(self.snapshot_dir, OUT, read_only=False),
                      Mount(record, RECORD, read_only=False), *layout.program_mounts())
            spec = ContainerSpec(image=image, command=command, workdir=wd, mounts=mounts)
        except Exception as e:  # noqa: BLE001 -- a snapshot's failure is only an event
            self._snapshot_skipped(name, e)
            return
        self._snapshot_launch = (name, self._launch_snapshot(name, spec))

    def _launch_snapshot(self, name: str, spec: ContainerSpec) -> Future[RoleHandle]:
        """Start snapshot *name*'s step off the stage's thread (``StageContext.start_off_thread``);
        the future of its handle. Until the stage collects it (``_collect_snapshot_launch``) the
        context refuses every other start, pin, hold and close on the stage's thread, so the
        scheduler's grant of the step never stalls the drive and the two threads never race."""
        return self.ctx.start_off_thread(name, lambda: self._snapshot_container(name, spec))

    def _snapshot_container(self, name: str, spec: ContainerSpec) -> RoleHandle:
        """Start the snapshot container and return its handle, tracking nothing: it runs in the
        off-thread launch's worker, so only the stage's thread takes the handle into ``running``
        (``_collect_snapshot_launch``)."""
        return self.ctx.start_container(name, spec, self._shape(self.params.snapshot_shape),
                                        owner="harness", overlap=True)

    def _collect_snapshot_launch(self, *, wait: bool) -> None:
        """Take the handle of a snapshot launch that has finished into ``running``, or skip the
        snapshot when its launch failed. With *wait*, wait for a launch still in flight, as the
        agent's part ends: once the sandbox is stopped (``_stopped``), so no agent works on while
        the scheduler grants the snapshot's step, and before the stage stops its roles or ends;
        without, leave it for a later tick."""
        if self._snapshot_launch is None or not (wait or self._snapshot_launch[1].done()):
            return
        name, launch = self._snapshot_launch
        self._snapshot_launch = None
        try:
            self.running[name] = launch.result()
        except Exception as e:  # noqa: BLE001 -- a snapshot's failure is only an event
            self._snapshot_skipped(name, e)

    def _snapshot_skipped(self, name: str, error: BaseException) -> None:
        """Record that snapshot *name* was not started; its failure is only an event."""
        scrubbed = self.ctx.scrubber.scrub(f"{type(error).__name__}: {error}")[:500]
        log.warning("snapshot %d could not be started and is skipped: %s", self._snapshots, scrubbed)
        self.events.emit("build.snapshot_skipped", n=self._snapshots, error=scrubbed)

    def drop_snapshots(self) -> None:
        """Remove the attempt's snapshots once the commit point has bundled main: they are
        evidence of the build in progress, which matters only when the commit point bundles
        nothing, so an attempt that collected a bundle stores the bundle alone. The
        ``build.snapshots_dropped`` event names those removed. A failure to remove one is the
        ``build.snapshots_left`` event, with the paths left in the attempt and the first error, and
        the attempt stands: a snapshot is quarantined and finalize reads none of it whether or not
        it stays (``AgentBuildStage.unread_files``)."""
        directory = self.snapshot_dir
        dropped: list[str] = []
        left: list[str] = []
        error: str | None = None
        for name in sorted(path.name for path in directory.glob("main-*.bundle")):
            try:
                (directory / name).unlink()
            except OSError as e:
                # Relative to the attempt, as the quarantine names it.
                left.append(f"{directory.name}/{name}")
                if error is None:
                    error = self.ctx.scrubber.scrub(f"{type(e).__name__}: {e}")[:500]
            else:
                dropped.append(name)
        self.events.emit("build.snapshots_dropped", names=dropped)
        if left:
            self.events.emit("build.snapshots_left", paths=left, error=error)

    @property
    def snapshot_dir(self) -> Path:
        """The attempt's ``snapshots/``, evidence beside its outputs, never one of them."""
        path = self.ctx.attempt_dir / "snapshots"
        path.mkdir(exist_ok=True)
        return path

    @staticmethod
    def _commit_record(path: Path) -> CommitRecord | None:
        """The commit point's record a try or a snapshot wrote at *path*; None when there is none,
        or it cannot be read as one: the harness wrote it, and a role killed outright may leave
        it partial."""
        try:
            return _COMMIT_RECORD.validate_json(path.read_bytes())
        except (OSError, ValueError):
            return None

    def room_for_the_agent(self) -> None:
        """Raise infra, retried, before the agent starts when the agent's hard limit and the
        collection after it (the commit point's tries and the transcript copy) do not fit before
        the allocation's end (``StageContext.deadline_cap``): a build whose allocation lost time
        to slow launches would run the agent for hours and then be cut at the end, an infra end
        that charges the contestant nothing but wastes its GPU time and API spend. It holds the
        stage's deadline for that whole span as well, as deploy_drive's ``hold_drive`` does, so
        the collection after the hard limit is not cut short. The launch waits and the cap go in
        the evidence, as deploy_drive's ``room_for_readiness`` records them."""
        ctx, p = self.ctx, self.params
        need = p.hard_sec + p.collection_sec
        if not ctx.hold_until(ctx.clock() + need):
            evidence = {f"launch_wait.{name}": f"{sec:.1f}" for name, sec in ctx.launch_waits().items()}
            evidence["deadline_cap"] = str(ctx.deadline_cap)
            raise StageFault.infra(f"the agent's {p.hard_sec}s hard limit and the {p.collection_sec}s "
                                   "collection after it do not fit before the allocation's end, so the agent was not "
                                   "started", retryable=True, **evidence)

    def command_started(self, sandbox: RoleHandle) -> float | None:
        """Wait for the runtime to start the sandbox's command, its exec server
        (``StageContext.wait_command_started``, its start marker), and return when, by the stage's
        clock; the runtime's and the seal's start before it are the harness's, bounded by the
        stage's deadline alone. None when the sandbox ended, or the stage was told to stop, first:
        the wait for the server then sees its end. Raise infra, retried, when the deadline came
        first."""
        began = self.ctx.wait_command_started(SANDBOX)
        if began is None and sandbox.poll() is None and not self.ctx.stopping:
            raise StageFault.infra("the sandbox's container did not start its command before the stage's deadline "
                                   "(no start marker on its stderr)", retryable=True)
        return began

    async def server_answered(self, env: SandboxEnvironment) -> None:
        """Wait for the sandbox's exec server to answer (``SandboxEnvironment.start``), the wait
        counted from the runtime's start of its command (``command_began``)."""
        began = self.command_began
        await env.start(elapsed_sec=0.0 if began is None else max(self.ctx.clock() - began, 0.0))

    async def setup(self, env: SandboxEnvironment, agent: Any, clock: WallClock) -> tuple[float, PreflightResult]:
        """Everything before the agent's clock: the sandbox's answer, the probe of its seal,
        the preflight, harbor's setup. A failure here is infra; a preflight that
        read back another model, window or output limit is not retried, nor is an accounting check
        that read the kernel clearly counting otherwise than ``site.nproc_accounting`` says
        (``SealProbe.accounting``), while one that read nothing clear is retried."""
        ctx = self.ctx
        try:
            await self.server_answered(env)
            seal = await self.seal_probe(env)
            [accounting] = [c for c in seal.checks if c.name == probe_mod.ACCOUNTING_CHECK.name]
            chosen = ctx.nproc_accounting()
            if not accounting.ok and seal.accounting is not None and not chosen.from_site:
                raise StageFault.infra(f"this node's kernel release {chosen.kernel} says its kernel counts the "
                                       f"commands' processes as {chosen.accounting!r}, but the nproc-accounting check "
                                       f"read {seal.accounting!r} ({accounting.observed}); set "
                                       f'site.nproc_accounting = "{seal.accounting}"', retryable=False)
            if not accounting.ok and seal.accounting is not None:
                # The kernel clearly counts otherwise than the site says: every build of the site
                # would be judged by the wrong level, so a retry would only repeat it.
                raise StageFault.infra("site.nproc_accounting does not match this node's kernel: "
                                       f"{accounting.observed}, expected {accounting.expected}", retryable=False)
            if not accounting.ok:
                # No clear reading says nothing of the site's setting; another attempt may read one.
                raise StageFault.infra("the nproc-accounting check gave no clear reading of how this node's "
                                       "kernel counts the commands' processes: "
                                       f"{accounting.observed}, expected {accounting.expected}", retryable=True)
            if not seal.passed:
                failed = "; ".join(f"{c.name}: {c.observed}" for c in seal.checks if not c.ok)
                raise StageFault.infra(f"the seal probe failed, so the sandbox may not be sealed as planned: {failed}",
                                       retryable=True, seal_probe_sec=str(seal.sec))
            pre = await preflight(env, agent, self.params.arm, config_dir=f"{workspace.PREFLIGHT}/config",
                                  cwd=f"{workspace.PREFLIGHT}/work")
            pre = pre.model_copy(update={"reason": ctx.scrubber.scrub(pre.reason)})
            self.events.emit("build.preflight", **pre.model_dump())
            if not pre.ok:
                raise StageFault.infra(f"the preflight failed: {pre.reason}", retryable=pre.retryable)
            await agent.setup(env)
            # Last, just before the clock: the agent's wall and the collection after it must fit
            # before the allocation's end, and hold the stage's deadline for them.
            self.room_for_the_agent()
        except StageFault:
            raise
        except Exception as e:  # noqa: BLE001 -- before the clock every failure is the harness's
            raise StageFault.infra(f"the sandbox's setup failed: {type(e).__name__}: "
                                   f"{ctx.scrubber.scrub(str(e))[:500]}", retryable=True) from e
        # The preflight's launch, before the clock, is no launch of the agent's part.
        self.launches_at_clock = self.cli_launches()
        clock.start()
        self.events.emit("build.started", task_id=ctx.plan.task, reasoning_effort=self.params.arm.effort,
                         arm=self.params.arm.name)
        return time.time(), pre

    async def session(self, env: SandboxEnvironment, agent: Any, clock: WallClock, brief: str, sandbox: RoleHandle,
                      layout: workspace.SandboxLayout,
                      image: PinnedImage,
                      ) -> tuple[float | None, PreflightResult | None, str, BaseException | None, float]:
        """The setup and the agent's part, on one event loop, since harbor's environment holds a
        client of it: when the clock started, the preflight, how the agent's part ended, what
        ``run`` raised and when its end was seen (``drive``). A setup that failed once the sandbox
        had ended or lost its channel is the sandbox's end, ``sandbox_ended``, with no clock and
        no preflight, seen when the setup failed."""
        try:
            try:
                started_at, pre = await self.setup(env, agent, clock)
            except Exception as e:
                failed_at = time.time()
                self._report_if_ended(env, sandbox)
                if env.lost is not None or sandbox.poll() is not None:
                    return None, None, "sandbox_ended", e, failed_at
                raise
            how, error, ended_at = await self.drive(env, agent, clock, brief, sandbox, layout, image)
            return started_at, pre, how, error, ended_at
        finally:
            await env.stop()

    # the end of the agent's part

    def end(self, how: str, error: BaseException | None, env: SandboxEnvironment, sandbox: RoleHandle, *,
            ended_at: float) -> tuple[Ending, RoleOutcome, FaultRecord | None]:
        """How the agent's part ended at *ended_at*, the sandbox's outcome and the contestant
        fault it ended on, if any, once the snapshot and the sandbox are stopped; or the fault the
        end is. The contestant's evidence comes first (``_evidence``, read against *ended_at*); a
        sandbox that ended or lost its channel is judged by ``sandbox_ended``."""
        if how == "sandbox_ended" or env.lost is not None or sandbox.poll() is not None:
            return self.sandbox_ended(env, ended_at)
        evidence = self._evidence(ended_at)
        if evidence is not None and evidence.evidence["ending"] == "contestant_pids":
            return self._contestant_end(*self._stopped(evidence))
        try:
            ending = self.ending(how, error)
        except StageFault:
            if how == "stopping" or evidence is None:
                raise
            return self._contestant_end(*self._stopped(evidence))
        self.interceptor_answered(ended_at)
        outcome, fault = self._stopped(None)
        if fault is None:
            self._unless_unlaunched(ending)
            self._unless_starved(ending, outcome)
            return ending, outcome, None
        if fault.fault_class != "contestant":
            raise StageFault(fault)
        # The sandbox ended on its own just before the stop: its own end stands.
        return self._contestant_end(outcome, fault)

    def cli_launches(self) -> int | None:
        """How many launches of the agent's CLI the sandbox's exec server has logged so far: its
        start lines that say their command launches it (``exec_protocol.parse_start_line``), in the
        sandbox role's log, which the contestant cannot write. A line the log lost is not counted.
        None when the log cannot be read."""
        count = 0
        try:
            with open(self.ctx.attempt_dir / "logs" / f"{SANDBOX}.err", "rb") as f:
                while line := f.readline(_LOG_LINE_BYTES):
                    start = exec_protocol.parse_start_line(line.decode("utf-8", "replace").rstrip("\n"))
                    count += start is not None and start["cli"]
        except OSError as e:
            log.warning("the sandbox's log could not be read for the CLI's launches: %s", e)
            return None
        return count

    def _unless_unlaunched(self, ending: Ending) -> None:
        """Raise infra, retried, when the agent's part ended as *ending*, a failure
        (``STARVABLE_ENDINGS``), with no launch of the agent's CLI since the clock started, as the
        exec server logged them once the sandbox was stopped (``cli_launches``), or when they could
        not be counted: a harbor command that failed then, or a launch the server never started,
        failed before any contestant process ran."""
        if ending not in STARVABLE_ENDINGS:
            return
        before, after = self.launches_at_clock, self.cli_launches()
        if before is not None and after is not None and after > before:
            return
        raise StageFault.infra("the agent's part ended in a failure before the agent's CLI was launched in it, so "
                               "no contestant process ran", retryable=True, agent_ending=ending,
                               cli_launches_at_clock=str(before), cli_launches=str(after))

    @staticmethod
    def _unless_starved(ending: Ending, outcome: RoleOutcome) -> None:
        """Raise infra, retried, when the agent's part ended as *ending*, a failure
        (``STARVABLE_ENDINGS``), while the uid's other tasks, counted once the sandbox's step was
        gone, filled its commands' process pool, or when they could not be counted then
        (``starvation_fault`` of the sandbox's *outcome*): the agent's forks may have failed for
        what others started. A normal end is graded whatever they held."""
        if ending not in STARVABLE_ENDINGS:
            return
        starved = starvation_fault(outcome)
        if starved is not None:
            # Its evidence named as a stage's roles' fault names it, with the agent's end.
            evidence = {"roles": outcome.name, **{f"{outcome.name}.{k}": v for k, v in starved.evidence.items()},
                        "agent_ending": ending}
            raise StageFault(starved.model_copy(update={"evidence": evidence}))

    def _stopped(self, evidence: FaultRecord | None) -> tuple[RoleOutcome, FaultRecord | None]:
        """Stop the sandbox as planned, then the snapshot, once a launch of one in flight is
        collected; the sandbox's outcome and its fault: its roles' when that is the contestant's,
        else the contestant's *evidence* read before the stop when there is some, else its roles'.
        The sandbox goes first, so the agent's processes end at the end of its part, the hard
        limit among them, never while a snapshot's step is still being granted. The reference
        service, whose one client is the agent, is stopped right after the sandbox."""
        outcome = self._stop(SANDBOX)
        # Judged before the service's stop: the shared rule reads it as it ran.
        fault = self._sandbox_fault(outcome)
        if self.service is not None:
            self.service.stop()
        self._collect_snapshot_launch(wait=True)
        self._reap_snapshot(stop=True)
        if evidence is not None and (fault is None or fault.fault_class != "contestant"):
            return outcome, evidence
        return outcome, fault

    def _sandbox_fault(self, outcome: RoleOutcome) -> FaultRecord | None:
        """The fault of the sandbox's own end *outcome*, by its roles (``stage_fault``). Beside the
        reference service, whose memory is oversubscribed on the sandbox's, the shared rule decides
        it (``classify_shared_build_end``) from what the service, the runner and the sandbox held,
        read while the service still runs; that evidence goes into the build's record
        (``BuildRecord.shared_memory``)."""
        fault = stage_fault([outcome])
        # Only an OOM when the job's own limit was reached is the shared rule's to read.
        if (self.service is None or self.service.handle is None or outcome.split is None
                or outcome.split.oom_at is None or outcome.job_oom is not True):
            return fault
        report = self.ctx.report(SANDBOX)
        self.shared_memory = self.service.memory_evidence(outcome.split.oom_at,
                                                          report.step if report is not None else None)
        shared = classify_shared_build_end(outcome, self.shared_memory)
        if shared is None:
            return None
        # Named as a stage's roles' fault names them.
        return shared.model_copy(update={"evidence": {"roles": outcome.name, **{
            f"{outcome.name}.{k}": v for k, v in shared.evidence.items()}}})

    def interceptor_answered(self, ended_at: float) -> None:
        """Raise infra, retried, unless the sandbox's egress interceptor, which carries the
        agent's route, reported within the package stage's ``HEARTBEAT_STALE_SEC`` of *ended_at*, when the agent's
        part ended: otherwise the harness may have cut the agent off from its model."""
        status = self.ctx.seal_status(SANDBOX)
        if status is None:
            raise StageFault.infra("the sandbox's egress interceptor recorded no status, so the agent may never have "
                                   "reached its model", retryable=True)
        if interceptor_stale(status, ended_at):
            raise StageFault.infra("the sandbox's egress interceptor had not reported for "
                                   f"{ended_at - status.heartbeat:.0f} s when the agent's part ended, so the agent may "
                                   "have been cut off from its model", retryable=True, heartbeat=str(status.heartbeat))

    def _contestant_end(self, outcome: RoleOutcome,
                        fault: FaultRecord | None) -> tuple[Ending, RoleOutcome, FaultRecord]:
        """The build's ending for a sandbox the contestant ended, by its contestant *fault*: its
        disk or its step's pids limit by its own ending, the contestant's evidence read while it
        ran for one the stage stopped as planned, and its memory otherwise."""
        assert fault is not None and fault.fault_class == "contestant", "a contestant end needs a contestant fault"
        ending: Ending
        if outcome.ending == "disk_limit":
            ending = "disk_limit"
        elif outcome.ending == "contestant_pids":
            ending = "contestant_pids"
        elif outcome.ending == "planned_stop" and fault.evidence.get("ending") == "contestant_pids":
            ending = "contestant_pids"
        else:
            ending = "contestant_oom"
        self.events.emit("build.agent_interrupted", reason=ending, msg=fault.reason)
        return ending, outcome, fault

    def ending(self, how: str, error: BaseException | None) -> Ending:
        """How the agent's part ended, from how the drive ended and what harbor's ``run``
        raised, with the sandbox still running and its channel alive; or the infra fault it is:
        the allocation's shutdown, or any exception of harbor's run that is no failed command."""
        from harbor.agents.installed.base import NonZeroAgentExitCodeError

        if how == "stopping":
            raise StageFault.infra("the allocation is shutting down", retryable=True)
        if how == "hard_limit":
            self.events.emit("build.agent_interrupted", reason="build_timeout", build_timeout_sec=self.params.hard_sec)
            return "hard_limit"
        if error is None:
            return "agent_ended"
        text = self.ctx.scrubber.scrub(str(error))[:500]
        if isinstance(error, NonZeroAgentExitCodeError):
            command = str(error).split("\n", 1)[0]
            if exec_protocol.is_cli_launch(command, (adapter_for(self.params.arm).cli_launch,)):
                self.events.emit("build.agent_interrupted", reason="agent_nonzero_exit", msg=text)
                return "agent_nonzero_exit"
            self.events.emit("build.agent_interrupted", reason="harbor_command_failed",
                             command=self.ctx.scrubber.scrub(command)[:500])
            return "harbor_command_failed"
        raise StageFault.infra(f"harbor's run failed: {type(error).__name__}: {text}", retryable=True)

    def sandbox_ended(self, env: SandboxEnvironment, failed_at: float) -> tuple[Ending, RoleOutcome, FaultRecord]:
        """The ending of a sandbox that ended, or whose exec channel was lost, before the agent's
        part did, once the snapshot is stopped: the contestant's when its roles or the
        contestant's evidence say so; raise its fault otherwise. A sandbox whose channel was lost
        while its container still runs ``LOST_WAIT_SEC`` later (``_report_if_ended`` gave it that
        long) is stopped, and the loss, seen at *failed_at*, is infra unless the contestant's
        evidence, read before the stop against that time (``_evidence``), shows its end."""
        if self.sandbox_outcome is not None:
            outcome = self.sandbox_outcome
            fault = self._sandbox_fault(outcome)
        elif self.running[SANDBOX].poll() is None:
            outcome, fault = self._stopped(self._evidence(failed_at))
            fault = fault or infra_fault(f"the sandbox's exec channel was lost while its container ran: {env.lost}",
                                         retryable=True, roles=SANDBOX)
        else:
            outcome = self._report_sandbox()
            fault = self._sandbox_fault(outcome)
        if fault is None:
            fault = infra_fault(f"the sandbox ended before the agent's part did, {outcome.ending} with status "
                                f"{outcome.exit_code}", retryable=True, roles=SANDBOX)
        self._collect_snapshot_launch(wait=True)
        self._reap_snapshot(stop=True)
        self._stop_all()
        if fault.fault_class != "contestant":
            raise StageFault(fault)
        return self._contestant_end(outcome, fault)

    def clear_breach(self, layout: workspace.SandboxLayout) -> None:
        """Remove what the sandbox wrote past its disk cap, once its processes are gone, before the
        commit point or the transcript role writes anything: every writable directory of it but
        the workspace's repository, which the commit point bundles, and the logs, which the
        transcript role copies. The ``build.sandbox_cleared`` event records it; a removal that
        fails is logged in it, and the contestant's end stands."""
        kept = layout.workspace / ".git"
        doomed = [d for d in layout.writable() if d not in (layout.workspace, layout.logs)]
        error = None
        try:
            names = os.listdir(layout.workspace) if layout.workspace.is_dir() else []
            doomed += [layout.workspace / name for name in names if layout.workspace / name != kept]
            remove_written(doomed)
        except OSError as e:
            error = f"{type(e).__name__}: {e}"
            log.error("the sandbox's writable directories were not all removed: %s", error)
        self.events.emit("build.sandbox_cleared", paths=[str(p) for p in doomed], error=error)

    def wait_gone(self, deadline_sec: float = GONE_WAIT_SEC) -> None:
        """Wait until no process of the sandbox's step is left: its cgroup tree lists none, or is
        gone. Files the contestant held open unlinked, or made with ``O_TMPFILE``, which no name
        leads to, free their space only then. Raise infra, retried, when some are still there after
        *deadline_sec*."""
        report = self.ctx.report(SANDBOX)
        if report is None:
            return
        cgroup = Path(report.cgroup_dir)
        give_up = time.monotonic() + deadline_sec
        while True:
            left = _cgroup_procs(cgroup)
            if not left:
                return
            if time.monotonic() >= give_up or self.ctx.stopping:
                raise StageFault.infra(f"{len(left)} processes of the sandbox's step were still there "
                                       f"{deadline_sec:.0f} s after it ended", retryable=True,
                                       processes=",".join(map(str, sorted(left)[:20])))
            time.sleep(min(TICK_SEC, 0.2))

    # collection

    def commit_deadline(self) -> float:
        """When, by the stage's clock, the commit point's tries must have ended: ``AFTER_COMMIT_SEC``
        before the stage's deadline, which ``room_for_the_agent`` held for the hard limit and the
        collection allowance and which grows by each role's launch wait, so the transcript copy
        and the extraction keep their time after it. Each try is handed the time left before it
        (``commit``)."""
        return self.ctx.deadline - AFTER_COMMIT_SEC

    def commit(self, layout: workspace.SandboxLayout, image: PinnedImage, name: str,
               seconds: int) -> tuple[CommitTry, CommitRecord | None]:
        """One run of the commit point, as role *name*, with *seconds* for all of its work: bundle
        the workspace's main into the outputs, or record that there is nothing to package (no
        commit on main, or a repository it cannot read because of what the contestant made). The
        stage ends a try still running ``COMMIT_WIND_DOWN_SEC`` past its *seconds*, counted from
        its command's start, so a read that hangs ends with the try. Return the try and the commit
        record, empty when it wrote none. What it wrote is recorded by its own report. What an
        earlier try left in the outputs is cleared first (``clear_partials``)."""
        ctx = self.ctx
        self.clear_partials()
        record = self.records / name
        command = commit_command(workspace=workspace.WORKSPACE, out=f"{OUT}/{SOURCE_BUNDLE_FILE}",
                                 max_bytes=self.params.repository_max_bytes, seconds=seconds,
                                 no_submission=f"{OUT}/{NO_SUBMISSION_FILE}", record=f"{RECORD}/commit.json",
                                 outputs_out=OUTPUTS_REPORT, tmp=f"{RECORD}/tmp")
        mounts = (Mount(layout.workspace, workspace.WORKSPACE), Mount(self.out, OUT, read_only=False),
                  Mount(record, RECORD, read_only=False),
                  Mount(ctx.outputs_report(name), OUTPUTS_REPORT, read_only=False), *layout.program_mounts())
        self._container(name, ContainerSpec(image=image, command=command, workdir=ctx.role_dir(name) / "wd",
                                            mounts=mounts), self.params.commit_shape)
        outcome = self._wait(name, command_sec=seconds + COMMIT_WIND_DOWN_SEC)
        found = self._commit_record(record / "commit.json")
        wrote: Literal["bundle", "no_submission"] | None = None
        if outcome.ending == "exited" and outcome.exit_code == 0 and (self.out / SOURCE_BUNDLE_FILE).is_file():
            wrote = "bundle"
        elif outcome.ending == "exited" and outcome.exit_code in (NO_MAIN_EXIT, UNREADABLE_EXIT) and \
                (self.out / NO_SUBMISSION_FILE).is_file():
            wrote = "no_submission"
        reason = None
        if wrote is not None:
            try:
                self.collected = ctx.reported_output(SOURCE_BUNDLE_FILE if wrote == "bundle" else NO_SUBMISSION_FILE,
                                                     name)
            except StageFault as e:
                wrote, reason = None, ctx.scrubber.scrub(e.fault.reason)[:500]
        else:
            fault = stage_fault([outcome])
            if outcome.ending == "time_limit":
                why = f"it was still running past the {seconds} s it was given, and the stage ended it"
            else:
                why = fault.reason if fault is not None else f"it ended with status {outcome.exit_code}"
            said = self._last_line(ctx.attempt_dir / "logs" / f"{name}.err")
            reason = ctx.scrubber.scrub(f"{why}; no bundle or no-submission record" +
                                        (f"; it said: {said}" if said else ""))[:500]
        return CommitTry(role=name, ending=outcome.ending, exit_code=outcome.exit_code, wrote=wrote,
                         reason=reason), found

    def clear_partials(self) -> None:
        """Remove every ``.tmp-*`` in the outputs: a commit point killed outright leaves its
        partial bundle there (``PARTIAL_DIR``), history that no record names and no scan reads."""
        for path in self.out.glob(".tmp-*"):
            remove_tree(path)

    @staticmethod
    def _last_line(path: Path) -> str | None:
        """The last line of the end of *path*; None when it has none."""
        try:
            with open(path, "rb") as f:
                f.seek(max(0, f.seek(0, os.SEEK_END) - 4096))
                lines = f.read().decode(errors="replace").strip().splitlines()
        except OSError:
            return None
        return lines[-1] if lines else None

    def collect(self, layout: workspace.SandboxLayout, image: PinnedImage,
                ending: Ending) -> tuple[Literal["bundle", "none"], CommitRecord | None, list[CommitTry]]:
        """Run the commit point once the agent's part has ended (*ending*), up to
        ``COMMIT_TRIES`` times on the same workspace, ``COMMIT_PAUSE_SEC`` apart, while it fails:
        it is the harness's work, and a failure of it is never the contestant's. Each try gets the
        time left before ``commit_deadline``, which it runs within (``--seconds``); a later try
        starts only while at least a second is left. The agent's clock is
        over, so the tries spend none of its budget. Return what it wrote, its record and every
        try. Raise ``_Uncollected`` with infra that is retried when the allocation
        shuts down first, and with infra that is not once the tries are spent or no time is left,
        a try that ran out of it among them: a new attempt is a new job, without this workspace,
        and could only run the agent again, so the build is void. Whatever a try killed outright
        left in the outputs goes before either (``clear_partials``)."""
        try:
            return self._collect(layout, image, ending)
        except BaseException:
            self._clear_partials_on_the_way_out()
            raise

    def _clear_partials_on_the_way_out(self) -> None:
        """``clear_partials`` as an attempt ends, with its bundle collected or without one; a
        failure to remove one is an event, since the attempt's own end stands and what is left
        lies where finalize reads nothing (``AgentBuildStage.unread_files``)."""
        try:
            self.clear_partials()
        except OSError as e:
            self.events.emit("build.partials_left", error=self.ctx.scrubber.scrub(f"{type(e).__name__}: {e}")[:500])

    def _collect(self, layout: workspace.SandboxLayout, image: PinnedImage,
                 ending: Ending) -> tuple[Literal["bundle", "none"], CommitRecord | None, list[CommitTry]]:
        """``collect``'s tries, as it describes them."""
        ctx = self.ctx
        tries: list[CommitTry] = []
        record: CommitRecord | None = None
        short = ""
        for n in range(1, COMMIT_TRIES + 1):
            if n > 1:
                time.sleep(COMMIT_PAUSE_SEC)
            left = self.commit_deadline() - ctx.clock()
            if left < 1:
                self.events.emit("build.commit_point_out_of_time", tries=len(tries), left_sec=left)
                short = (f"; no time was left for {'a later try' if tries else 'it'} before the collection's last "
                         f"steps, {AFTER_COMMIT_SEC} s before the stage's deadline")
                break
            tried, record = self.commit(layout, image, commit_role(n), math.floor(left))
            tries.append(tried)
            self.events.emit("build.commit_point", n=n, wrote=tried.wrote, ending=tried.ending,
                             exit_code=tried.exit_code, reason=tried.reason)
            if tried.wrote is not None:
                return ("bundle" if tried.wrote == "bundle" else "none"), record, tries
            if ctx.stopping:
                raise _Uncollected(StageFault.infra("the allocation is shutting down after the agent's part ended "
                                                    f"({ending}), before the commit point wrote anything: "
                                                    f"{tried.reason}", retryable=True,
                                                    agent_ending=ending), record, tries)
        snapshots = sorted(path.name for path in self.snapshot_dir.glob("main-*.bundle"))
        kept = ", ".join(f"snapshots/{name}" for name in snapshots) or "none"
        what = (f"failed {len(tries)} time{'s' if len(tries) > 1 else ''}" if tries else "did not run")
        last = f" Its last failure: {tries[-1].reason}." if tries else ""
        raise _Uncollected(StageFault.infra(
            f"the commit point {what} after the agent's part ended ({ending}){short}; a new attempt could only run "
            f"the agent again.{last} Snapshots kept as evidence: {kept}", retryable=False, agent_ending=ending,
            snapshots=kept, commit_tries=" | ".join(f"{c.role}: {c.reason}" for c in tries)[:2000]), record, tries)

    def evidence(self, record: BaseModel, file: str = BUILD_FILE) -> None:
        """Write *record* into the attempt as *file*, beside its snapshots: the evidence of an
        attempt that ends without its outputs, never an output. A failure to write it is an event;
        the attempt's own fault stands."""
        ctx = self.ctx
        try:
            (ctx.attempt_dir / file).write_bytes(ctx.scrubber.scrub(record.model_dump_json(indent=2)).encode())
        except Exception as e:  # noqa: BLE001 -- evidence never replaces the attempt's fault
            self.events.emit("build.evidence_unwritten", error=ctx.scrubber.scrub(f"{type(e).__name__}: {e}")[:500])

    def transcript(self, layout: workspace.SandboxLayout) -> tuple[Path, str]:
        """Copy the agent's logs into the outputs by the transcript role, with every value the
        stage holds replaced; return the copy's directory (``COPY_DIR``, beside the index) and the
        index's key, recorded by the role's own report. The values reach the role only through
        its 0600 env file, written before the sandbox started; its argv names each variable and
        secret, never a value. A granted value is named by its secret; any other value the stage
        holds is keyed by its secret's name already (``StageContext.secrets``)."""
        ctx = self.ctx
        out = self.out / TRANSCRIPT_DIR
        out.mkdir(exist_ok=True)
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.agent.transcript", "--in", str(layout.agent_logs),
                "--out", str(out), "--outputs-out", str(ctx.outputs_report(TRANSCRIPT)),
                *(x for var, name in self.scrubbed.items() for x in ("--scrub", f"{var}={name}"))]
        self._host(TRANSCRIPT, argv, env=host_role_env(ctx), secret_env=tuple(self.scrubbed))
        self._check(self._wait(TRANSCRIPT))
        if not (out / INDEX_FILE).is_file():
            raise StageFault.infra("the transcript copy wrote no index", retryable=True)
        return out / COPY_DIR, ctx.reported_output(f"{TRANSCRIPT_DIR}/{INDEX_FILE}", TRANSCRIPT)

    def extract(self, copy: Path) -> tuple[Extraction, str | None]:
        """Harbor's reading of the agent's trajectory and spend from the transcript's *copy*,
        regular files only with the secrets replaced, by the extraction role (``hls.agent.extract``)
        on its own memory (``extract_shape``), for at most ``EXTRACT_SEC``: harbor's readers hold a
        session whole, and a crafted one many times its size, so the runner never parses the
        agent's logs. Return what it read, and the key of its trajectory, written beside the index
        where no name the agent gives a file of its logs can meet it and recorded by the role's
        own report; None when there is none. A role that could not start, ran out of its memory,
        failed or ran past its time leaves the trajectory and spend unavailable, with why (the
        ``build.extraction_unavailable`` event), and the build goes on: the submission was taken
        at the commit point, so the extraction never fails the build. A trajectory past
        ``TRAJECTORY_MAX_BYTES``, which the role withholds and the stage refuses as reported, is
        unavailable alone (the ``build.trajectory_unavailable`` event), and the spend is kept.
        What the role's record says is read, at most ``EXTRACT_RECORD_MAX_BYTES`` of it, never the
        agent's files."""
        ctx, p = self.ctx, self.params
        out = copy.parent
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.agent.extract", "--harness", p.arm.harness,
                "--model", adapter_for(p.arm).model_string(p.arm), "--logs", str(copy), "--out", str(out),
                "--record", str(self.extract_record), "--outputs-out", str(ctx.outputs_report(EXTRACT))]
        unavailable, read = self._extraction(argv)
        trajectory, partial = None, None
        if read is not None and read.trajectory:
            try:
                trajectory = ctx.reported_output(f"{TRANSCRIPT_DIR}/{TRAJECTORY_FILE}", EXTRACT)
            except StageFault as e:
                unavailable, read = f"the extraction role's trajectory is not reported: {e.fault.reason}", None
        if trajectory is not None and (size := ctx.output_ref(trajectory).size) > TRAJECTORY_MAX_BYTES:
            trajectory = None
            partial = f"harbor's trajectory took {size} bytes, past the {TRAJECTORY_MAX_BYTES} the build keeps"
        elif read is not None:
            partial = read.trajectory_withheld
        if unavailable is not None:
            unavailable = ctx.scrubber.scrub(unavailable)[:500]
            self.events.emit("build.extraction_unavailable", reason=unavailable)
            extraction = Extraction(usage=None, trajectory_sha256=None, unavailable=unavailable,
                                    trajectory_unavailable=None)
        else:
            assert read is not None
            if partial is not None:
                self.events.emit("build.trajectory_unavailable", reason=partial)
            sha = ctx.output_ref(trajectory).sha256 if trajectory is not None else None
            extraction = Extraction(usage=read.usage, trajectory_sha256=sha, unavailable=None,
                                    trajectory_unavailable=partial)
        if trajectory is None:
            # What harbor wrote of a trajectory before the role ended, or past its bound, is no output.
            (out / TRAJECTORY_FILE).unlink(missing_ok=True)
        usage = extraction.usage
        self.events.emit("build.finished", cost_usd=usage.cost_usd if usage else None,
                         n_input_tokens=usage.n_input_tokens if usage else None,
                         n_output_tokens=usage.n_output_tokens if usage else None)
        return extraction, trajectory

    def _extraction(self, argv: list[str]) -> tuple[str | None, ExtractionRecord | None]:
        """Run the extraction role with *argv* for at most ``EXTRACT_SEC``, and return why its
        reading is unavailable, or the record it wrote: a start the context or the scheduler
        refused (unless the stage is stopping, which ends the build anyway), a role that ran past
        its time, one that ended with a fault, and a record that cannot be read or is longer than
        ``EXTRACT_RECORD_MAX_BYTES`` are each unavailable, with why."""
        ctx, p = self.ctx, self.params
        try:
            handle = ctx.start_role(EXTRACT, argv, self._shape(p.extract_shape), owner="harness",
                                    env=host_role_env(ctx), overlap=True)
        except StageFault as e:
            if ctx.stopping:
                raise
            return f"the extraction role could not start: {e.fault.reason}", None
        self.running[EXTRACT] = handle
        give_up = monotonic() + EXTRACT_SEC
        while handle.poll() is None and not ctx.stopping and monotonic() < give_up:
            handle.wait(TICK_SEC)
        late = handle.poll() is None and not ctx.stopping
        outcome = self._stop(EXTRACT) if late else self._wait(EXTRACT)
        unavailable, read = None, None
        if late:
            unavailable = f"the extraction role ran past its {EXTRACT_SEC:g} s and was stopped"
        elif (fault := stage_fault([outcome])) is not None:
            said = self._last_line(ctx.attempt_dir / "logs" / f"{EXTRACT}.err")
            unavailable = (f"the extraction role ended {outcome.ending} with status {outcome.exit_code}: {fault.reason}"
                           + (f"; it said: {said}" if said else ""))
        else:
            try:
                with open(self.extract_record, "rb") as f:
                    raw = f.read(EXTRACT_RECORD_MAX_BYTES + 1)
                if len(raw) > EXTRACT_RECORD_MAX_BYTES:
                    unavailable = f"the extraction role's record is longer than {EXTRACT_RECORD_MAX_BYTES} bytes"
                else:
                    read = ExtractionRecord.model_validate_json(raw)
            except (OSError, ValueError) as e:
                unavailable = f"the extraction role wrote no record that can be read: {type(e).__name__}"
        return unavailable, read

    # the probe: the seal's before every build's agent, and the whole of it in a stop build

    def probe_inputs(self) -> probe_mod.ProbeInputs:
        """What a passing sandbox should look like, for the in-sandbox checks: the host user's
        ids, the sandbox's /dev/shm cap and GPUs, the paths its commands must not reach and the
        budget server's, the route's host (``host``) and a host the seal's policy refuses, the
        route's variables, and the commands' process ceiling (``ceiling``)."""
        p = self.params
        policy = seal_policy(NetworkPolicy.model_validate(p.network), self.host, resolve=lambda host: [])
        return probe_mod.ProbeInputs(
            uid=os.getuid(), gid=os.getgid(), shm_bytes=p.shm_gib * _GIB, gpus=p.sandbox_shape.gpus,
            control_target=exec_protocol.CONTROL_TARGET,
            socket_target=f"{exec_protocol.CONTROL_TARGET}/{exec_protocol.SOCKET_NAME}", mcp_target=MCP_TARGET,
            route_host=self.host, refused_host=refused_host(policy.network), dial=PROBE_DIAL, route_vars=self.route_vars(),
            command_nproc=self.ceiling.ceiling if self.ceiling is not None else None,
            nproc_accounting=self.ctx.nproc_accounting().accounting,
            ucounts=self.ucounts, python=exec_protocol.SANDBOX_PYTHON)

    def route_vars(self) -> tuple[str, ...]:
        """The variables the arm's route reaches the CLI by, its grants and its plain settings,
        which harbor sends with each of the agent's requests and nothing else may hold."""
        arm = self.params.arm
        adapter = adapter_for(arm)
        return tuple(sorted({*adapter.grants(arm.route_table), *adapter.plain_env(arm)}))

    async def seal_probe(self, env: SandboxEnvironment) -> SealProbe:
        """Probe the sandbox's seal as contestant commands, the seal probe's ``CHECK_SEC`` each
        (``hls.egress.seal_probe``), the inbound check's command with the ports' settle on top: the route's
        host reached, a host the policy refuses refused, no ICMP echo out, and the index reached;
        how the kernel counts the commands' process ceiling against the site's accounting
        (``probe.ACCOUNTING_CHECK``); then read host side that the seal's interceptor recorded that
        refusal by its policy; then knock on two ports a sandbox command holds from every address
        of the node (``probe.inbound_check``). The ``build.seal_probe`` event records each check and
        the seconds it took. A lost sandbox raises through the exec."""
        started = monotonic()
        inputs = self.probe_inputs()
        checks = await probe_mod.run_sandbox_checks(env.exec, inputs, which="sealed", timeout_sec=CHECK_SEC)
        accounting = probe_mod.ACCOUNTING_CHECK
        results = [await env.exec(command, timeout_sec=CHECK_SEC) for command in accounting.commands(inputs)]
        checks.append(accounting.parse(results, inputs))
        measured = probe_mod.read_accounting(results[0])
        files = self.ctx.seal_files(SANDBOX)
        recorded = probe_mod.refusal_recorded_check(policy_refusals(files.status) if files is not None else None,
                                                    inputs.refused_host)
        after = next((i + 1 for i, c in enumerate(checks) if c.name == "unlisted-refused"), len(checks))
        checks.insert(after, recorded)
        assert self.sandbox_seal is not None and self.sandbox_layout is not None
        checks.append(await probe_mod.inbound_check(
            env.exec, inputs, avoid=namespace_ports(self.sandbox_seal), scratch=self.sandbox_layout.seal_probe,
            target=workspace.SEAL_PROBE, timeout_sec=CHECK_SEC))
        checks = [self._scrubbed(c) for c in checks]
        seal = SealProbe(checks=checks, passed=all(c.ok for c in checks), accounting=measured,
                         sec=round(monotonic() - started, 3))
        self.seal = seal
        self.events.emit("build.seal_probe", passed=seal.passed, sec=seal.sec, accounting=seal.accounting,
                         checks=[c.model_dump() for c in seal.checks])
        return seal

    def _scrubbed(self, check: probe_mod.ProbeCheck) -> probe_mod.ProbeCheck:
        """*check* with what it saw and expected scrubbed of every value the stage holds, a URL
        secret's host among them, so no record names a gateway."""
        scrub = self.ctx.scrubber.scrub
        return check.model_copy(update={"observed": scrub(check.observed), "expected": scrub(check.expected)})

    async def workspace_owner(self, env: SandboxEnvironment, layout: workspace.SandboxLayout) -> int | None:
        """Make a file in the workspace from a sandbox command and return its owner's uid host-side;
        None when the command or the read failed."""
        try:
            done = await env.exec(f"touch {workspace.WORKSPACE}/{probe_mod.WORKSPACE_PROBE}",
                                  timeout_sec=PROBE_CHECK_SEC)
            if done.return_code != 0:
                return None
            return (layout.workspace / probe_mod.WORKSPACE_PROBE).stat().st_uid
        except OSError:
            return None

    def host_checks(self, layout: workspace.SandboxLayout) -> list[probe_mod.ProbeCheck]:
        """The probe's checks the stage makes on the host: from the exec server's ready file, the
        one the memory split reads, the server's own score, and through the step's processes the
        spawner's score, the reaper's depth and its descriptors; the sandbox's env file; the
        node's ``pid_max`` beside the commands' process ceiling; the step's memory split, sampled
        now; and the processes of the step outside the container."""
        ctx = self.ctx
        if self.ceiling is None:
            raise RuntimeError("the probe reads the commands' process ceiling, which is read before the sandbox")
        ready = probe_mod.read_ready(layout.ready)
        report = ctx.report(SANDBOX)
        cgroup = Path(report.cgroup_dir) if report is not None else None
        samples = ctx.sample_split(SANDBOX)
        return [probe_mod.server_score_check(ready), probe_mod.spawner_score_check(ready, cgroup),
                probe_mod.depth_check(ready, cgroup), probe_mod.reaper_socket_check(ready, cgroup),
                probe_mod.env_file_check(self.sandbox_env, self.route_vars(), list(ctx.secrets.values())),
                probe_mod.pid_max_check(self.ceiling.ceiling), probe_mod.harness_ucounts_check(cgroup),
                probe_mod.spawner_limits_check(ready, cgroup, self.ceiling.hard),
                probe_mod.split_check(samples[-1] if samples else None), probe_mod.outside_check(ready, cgroup)]

    async def oom_probe(self, env: SandboxEnvironment, sandbox: RoleHandle) -> probe_mod.OomProbe:
        """Run the memory hog in the sandbox, as a contestant command, and record what ended it:
        whether the sandbox survived, the hog's exit status, the step's OOM kills, the runner's
        kills in the step, and the largest resident memory the harness held in the split samples,
        the last taken once the hog ended: the measurement ``harness_reserve_gib`` is set from
        (``--probe-oom``). A lost channel is the sandbox's own end, which its roles judge."""
        exit_code: int | None = None
        error: str | None = None
        try:
            exit_code = (await env.exec(OOM_HOG, timeout_sec=OOM_PROBE_SEC)).return_code
        except Exception as e:  # noqa: BLE001 -- the sandbox's end, judged by its roles after the probe
            error = type(e).__name__
        samples = self.ctx.sample_split(SANDBOX)
        oom = probe_mod.OomProbe(survived=sandbox.poll() is None, exit_code=exit_code, error=error,
                                 oom_kills=samples[-1].oom_kills if samples else 0,
                                 runner_kills=len(self._memory_kill_times()),
                                 harness_max_bytes=max((s.harness for s in samples), default=0),
                                 samples=len(samples))
        self.events.emit("build.oom_probe", **oom.model_dump())
        return oom

    async def probe_session(self, env: SandboxEnvironment, agent: Any,
                            layout: workspace.SandboxLayout, sandbox: RoleHandle,
                            ) -> tuple[probe_mod.ProbeRecord, PreflightResult | None, BaseException | None]:
        """A stop build's part, on one event loop: the sandbox's answer, the seal probe as every
        build runs it, the rest of the checks, the preflight for a preflight-only build and the
        OOM probe for ``probe_oom``. No agent clock starts and nothing is committed. A failed
        check is a record, not a fault: the operator reads it. Return the record of every check
        that ran, the preflight, and what failed, if anything: the sandbox may have ended."""
        p = self.params
        checks: list[probe_mod.ProbeCheck] = []
        pre: PreflightResult | None = None
        oom: probe_mod.OomProbe | None = None
        error: BaseException | None = None
        started = monotonic()
        try:
            await self.server_answered(env)
            checks.append(probe_mod.readiness_check(monotonic() - started, env.start_timeout_sec))
            checks += (await self.seal_probe(env)).checks
            inputs = self.probe_inputs()
            await probe_mod.run_sandbox_checks(env.exec, inputs, which="unsealed", timeout_sec=PROBE_CHECK_SEC,
                                               done=checks)
            checks.append(probe_mod.owner_check(await self.workspace_owner(env, layout), inputs.uid))
            checks += self.host_checks(layout)
            if p.stop_after == "preflight":
                pre = await preflight(env, agent, p.arm, config_dir=f"{workspace.PREFLIGHT}/config",
                                      cwd=f"{workspace.PREFLIGHT}/work")
                pre = pre.model_copy(update={"reason": self.ctx.scrubber.scrub(pre.reason)})
                self.events.emit("build.preflight", **pre.model_dump())
            if p.probe_oom:
                oom = await self.oom_probe(env, sandbox)
        except Exception as e:  # noqa: BLE001 -- the caller judges it with the sandbox's state
            error = e
        finally:
            await env.stop()
        checks = [self._scrubbed(c) for c in checks]
        for check in checks:
            self.events.emit("build.probe_check", name=check.name, ok=check.ok, observed=check.observed[:200])
        return probe_mod.record(checks, oom=oom), pre, error

    def stop_build(self, env: SandboxEnvironment, agent: Any, sandbox: RoleHandle, layout: workspace.SandboxLayout,
                   image: PinnedImage, upload_sha256: str | None) -> dict[str, str]:
        """A probe or preflight-only build (``stop_after``): run the probe, then stop the sandbox
        as a build's normal end does, and publish ``probe.json`` and ``build.json``, which records
        the tree digest of the folder uploaded into the sandbox (*upload_sha256*). A stop build
        never reports a score and never declares a contestant fault: a sandbox
        that ended on its own, whose channel was lost, or that ended just as the stage stopped it,
        is judged by its roles, and that judgement is recorded in ``probe.json``, which the stage
        writes into the attempt as evidence; the attempt then ends infra that is not retried, a
        probe outcome, whatever the judgement's class, so ``hls run`` prints the probe's result
        rather than a zero. A probe whose own code failed while the
        sandbox lives is infra, retried."""
        ctx = self.ctx
        started_at = time.time()
        probed, pre, error = asyncio.run(self.probe_session(env, agent, layout, sandbox))
        # Seen before the wait for a lost channel's sandbox to end: the time its evidence is read against.
        ended_at = time.time()
        self._report_if_ended(env, sandbox)
        if env.lost is not None or sandbox.poll() is not None:
            fault = self._ended_fault(env, ended_at)
        else:
            if ctx.stopping:
                raise StageFault.infra("the allocation is shutting down", retryable=True)
            if error is not None:
                raise StageFault.infra(f"the probe failed: {type(error).__name__}: "
                                       f"{ctx.scrubber.scrub(str(error))[:500]}", retryable=True)
            outcome, stopped = self._stopped(None)
            if stopped is None:
                return self.publish_stop(image, pre, probed, outcome, started_at, upload_sha256)
            # The sandbox ended by itself just as the stage stopped it: its own end stands.
            self._stop_all()
            fault = stopped
        judged = {"fault_class": fault.fault_class, "ending": str(fault.evidence.get("ending", "")),
                  "reason": ctx.scrubber.scrub(fault.reason)[:500]}
        self.evidence(probed.model_copy(update={"sandbox_ending": judged}), PROBE_FILE)
        raise StageFault.infra(f"the sandbox ended during the {self.params.stop_after} build, judged "
                               f"{judged['fault_class']} ({judged['ending'] or 'no ending'}): {judged['reason']}",
                               retryable=False, roles=SANDBOX, **{f"sandbox.{k}": v for k, v in judged.items()})

    def _ended_fault(self, env: SandboxEnvironment, ended_at: float) -> FaultRecord:
        """The roles' judgement of a sandbox that ended during a stop build, seen at *ended_at*,
        whatever its class, for the record; ``sandbox_ended`` stops the other roles and either
        returns the contestant fault or raises the infra one."""
        try:
            _, _, fault = self.sandbox_ended(env, ended_at)
        except StageFault as e:
            fault = e.fault
        return fault

    def publish_stop(self, image: PinnedImage, pre: PreflightResult | None, probed: probe_mod.ProbeRecord,
                     stopped: RoleOutcome, started_at: float, upload_sha256: str | None) -> dict[str, str]:
        """Write ``probe.json`` and ``build.json`` for a stop build, and return the outputs; the
        record names the tree digest of the folder the task uploaded (*upload_sha256*)."""
        ctx, p = self.ctx, self.params
        assert p.stop_after is not None and self.seal is not None and self.ceiling is not None
        outputs = {"brief": ctx.output_key(BRIEF_FILE), "brief_provenance": ctx.output_key(BRIEF_PROVENANCE_FILE),
                   SERVED_UCOUNTS_OUTPUT: ctx.output_key(SERVED_UCOUNTS_FILE),
                   "agent_env": ctx.output_key(AGENT_ENV_FILE),
                   PROBE_OUTPUT: ctx.write_output(PROBE_FILE,
                                                  ctx.scrubber.scrub(probed.model_dump_json(indent=2)).encode())}
        build = BuildRecord(arm=p.arm, arm_change=p.arm_change, sandbox_image=image.file_sha256,
                            upload_sha256=upload_sha256, soft_sec=p.soft_sec, hard_sec=p.hard_sec,
                            max_turns=p.max_turns, seal_probe=self.seal, preflight=pre,
                            started_at=started_at, ended_at=time.time(), agent_sec=0.0, ending=p.stop_after,
                            reentries=0, sessions_ended=0, derailed=None, commit=None, commit_tries=[],
                            memory=stopped.split, fault=None, grade_after_sandbox_oom=self.oom_graded,
                            process_ceiling=self.ceiling, pids_max_events=self.pids, extraction=None, probe=probed)
        outputs[BUILD_OUTPUT] = ctx.write_output(BUILD_FILE,
                                                 ctx.scrubber.scrub(build.model_dump_json(indent=2)).encode())
        return outputs

    # the stage

    def run(self) -> dict[str, str]:
        """The build, with every record of harbor's loggers scrubbed while it runs."""
        with scrubbed_harbor_records(self.ctx.scrubber):
            return self._run()

    def _run(self) -> dict[str, str]:
        from hls.agent.environment import SandboxEnvironment
        from hls.agent.harnesses import build_agent, harness_env_leaks

        ctx, p = self.ctx, self.params
        leaks = harness_env_leaks(os.environ)
        if leaks:
            raise StageFault.infra(f"the runner's environment holds {leaks}, which harbor would read in place of the "
                                   "arm's settings", retryable=False)
        artifact = self.artifact()
        # A stop build's plan pins what the store or the task's defaults held, and its brief states what
        # was pinned; a scored build's plan pins every file the task loads.
        pins = ctx.plan.measured
        task = measured_task(ctx, p.task, calibration=p.stop_after is None or pins.calibration is not None,
                             credit=p.stop_after is None or pins.credit is not None)
        engine = self.service_engine(task)
        provisioned = self.provisioned()
        host = self.host = self.route_host()
        # Read before any role starts: a ceiling under its floor is infra, not retried.
        ceiling = self.ceiling = self.ctx.process_ceiling()
        # Read before any role starts too: a limit under its floor is infra, not retried.
        self.ucounts = commands_limits(self.ctx.ucount_limits())
        layout = self.layout()
        if engine is not None:
            assert p.reference_service is not None
            self.service = BuildService(ctx, p.reference_service, engine, layout.root, self.events.emit)
            self.service.prepare()
        image = self.image()
        service_image = self.service_image()
        data, data_env = self.data(provisioned)
        if self.service is not None and not any(m.target == WEIGHTS_TARGET for m in data):
            raise StageFault.infra("the reference service loads the task model's weights, and the agent's data mounts "
                                   "none", retryable=False)
        self.seed(layout, image)
        skills, upload_sha256 = self.uploads(layout, artifact)
        brief = self.brief(task, artifact)
        clock = WallClock(limit_sec=p.soft_sec or p.hard_sec, hard_sec=p.hard_sec, clock=monotonic)
        # The budget server is registered when a full build has a soft budget. A full build runs
        # with its own (``p.soft_sec``); a stop build runs with none of its own (the planner gives
        # it none), so it reads the task's producer budget, the one a full build's agent would use,
        # and its probe and preflight check the same server.
        wants_budget = (p.soft_sec is not None if p.stop_after is None
                        else budget_params(artifact.producer)["soft_sec"] is not None)
        try:
            agent = build_agent(p.arm, logs_dir=layout.agent_logs, clock=clock, mcp=wants_budget,
                                skills_dir=skills, goal=p.soft_sec is not None, max_turns=p.max_turns,
                                emit=self.events.emit, secrets=ctx.secrets)
        except ValueError as e:
            raise StageFault.infra(f"the arm's agent cannot be built: {e}", retryable=False) from e
        self.prepare_collection()
        service = BudgetService(clock, layout.budget_socket)
        service.start()
        try:
            token = secrets.token_hex(32)
            sandbox, sources = self.start_sandbox(layout, image, data, data_env, token, host, ceiling)
            self.agent_env(sources, agent)
            self.start_service(sandbox, service_image, data)
            env = SandboxEnvironment(socket=layout.socket, token=token, paths=layout.transfer_paths(),
                                     root=layout.harbor, gpus=p.sandbox_shape.gpus,
                                     alive=lambda: sandbox.poll() is None,
                                     secret_forms=registry_forms(ctx.secrets))
            self.command_began = self.command_started(sandbox)
            if p.stop_after is not None:
                return self.stop_build(env, agent, sandbox, layout, image, upload_sha256)
            started_at, pre, how, error, ended_at = asyncio.run(self.session(env, agent, clock, brief, sandbox,
                                                                              layout, image))
            # The agent's clock closes with its part: collection spends none of it.
            agent_sec = round(clock.elapsed_sec(), 3)
            if started_at is None or pre is None:
                # The sandbox ended before the agent's clock started: the agent never ran, so
                # whatever its roles say, the end is the harness's. An infra end is raised as
                # its roles' fault; a contestant one as infra naming it.
                _, _, decided = self.sandbox_ended(env, ended_at)
                raise StageFault.infra("the sandbox ended before the agent's clock started, so the agent never ran: "
                                       f"{decided.reason}", retryable=True,
                                       **{f"sandbox_end.{k}": v for k, v in decided.evidence.items()})
            ending, stopped, decided = self.end(how, error, env, sandbox, ended_at=ended_at)
        finally:
            # A launch the end never collected, as when it raised first, is waited for here, so the
            # context holds its role when it closes and stops it.
            self._collect_snapshot_launch(wait=True)
            service.stop()
            self._stop_service_on_the_way_out()

        assert self.seal is not None, "the seal was probed before the clock started"
        seal = self.seal

        def build_record(commit: CommitRecord | None, tries: list[CommitTry], extraction: Extraction | None,
                         fault: FaultRecord | None) -> BuildRecord:
            return BuildRecord(arm=p.arm, arm_change=p.arm_change, sandbox_image=image.file_sha256,
                               upload_sha256=upload_sha256, soft_sec=p.soft_sec, hard_sec=p.hard_sec,
                               max_turns=p.max_turns, seal_probe=seal, preflight=pre,
                               started_at=started_at, ended_at=ended_at, agent_sec=agent_sec, ending=ending,
                               reentries=self.events.reentries, sessions_ended=self.events.sessions_ended,
                               derailed=self.derailed, commit=commit, commit_tries=tries,
                               memory=stopped.split, fault=fault, grade_after_sandbox_oom=self.oom_graded,
                               process_ceiling=ceiling, pids_max_events=self.pids, extraction=extraction,
                               shared_memory=self.shared_memory)

        if ending == "contestant_oom" and not self.oom_graded:
            # The plan's rule scores a sandbox's end on its own memory zero:
            # the commit point never runs, so nothing is graded. The record is evidence.
            assert decided is not None, "a contestant end carries its fault"
            zero = ctx.declare_contestant_fault(sandbox_oom_zero(decided))
            self.evidence(build_record(None, [], None, zero))
            raise StageFault(zero)
        self.wait_gone()
        if ending == "disk_limit":
            self.clear_breach(layout)

        try:
            kind, record, tries = self.collect(layout, image, ending)
        except _Uncollected as e:
            # The record shows the agent's end and every try even when nothing was collected.
            self.evidence(build_record(e.record, e.tries, None, decided))
            raise e.fault from None
        # A retry from here would run the whole session again. The declaration is on the store
        # before anything else runs, so an end the stage never reports (a SIGTERM, the runner's own
        # failure, its OOM kill, the node's failure) ends the attempt with it too.
        ctx.declare_no_retry(_after_commit("the attempt ended before the stage did: the allocation's shutdown, the "
                                           "runner's own failure or its death", kind, {}))
        if kind == "none" and decided is not None:
            # The contestant's end left nothing gradable: its zero, which the stage declares.
            raise StageFault(ctx.declare_contestant_fault(decided))
        try:
            ctx.events.emit("build.source_collected", status=record.status if record else None,
                            commit=record.commit if record else None,
                            reason=ctx.scrubber.scrub(record.reason or "") if record else "")
            if kind == "bundle":
                self.drop_snapshots()
            copy, index = self.transcript(layout)
            extraction, trajectory = self.extract(copy)
            assert self.collected is not None, "the commit point wrote what it reported"
            outputs = {"brief": ctx.output_key(BRIEF_FILE), "brief_provenance": ctx.output_key(BRIEF_PROVENANCE_FILE),
                       SERVED_UCOUNTS_OUTPUT: ctx.output_key(SERVED_UCOUNTS_FILE),
                       "agent_env": ctx.output_key(AGENT_ENV_FILE), "transcript": index,
                       (SOURCE_BUNDLE_OUTPUT if kind == "bundle" else NO_SUBMISSION_OUTPUT): self.collected}
            if trajectory is not None:
                outputs[TRAJECTORY_OUTPUT] = trajectory
            build = build_record(record, tries, extraction, decided)
            outputs[BUILD_OUTPUT] = ctx.write_output(BUILD_FILE,
                                                     ctx.scrubber.scrub(build.model_dump_json(indent=2)).encode())
            self._clear_partials_on_the_way_out()
            return outputs
        except StageFault as e:
            raise StageFault(_after_commit(e.fault.reason, kind, dict(e.fault.evidence))) from e
        except Exception as e:  # noqa: BLE001 -- past the commit point no failure may rebuild the session
            raise StageFault(_after_commit(f"{type(e).__name__}: {ctx.scrubber.scrub(str(e))[:500]}", kind,
                                           {})) from e


def _cgroup_procs(cgroup: Path) -> set[int]:
    """The processes every ``cgroup.procs`` of the cgroup tree *cgroup* lists; none when the tree
    is gone or lists none."""
    found: set[int] = set()
    try:
        files = sorted(cgroup.rglob("cgroup.procs")) if cgroup.is_dir() else []
    except OSError:
        return found
    for path in files:
        try:
            found |= {int(line) for line in path.read_text().split() if line.isdigit()}
        except OSError:
            continue
    return found


@register
class AgentBuildStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "agent_build"
    params_model: ClassVar[type[BaseModel]] = AgentBuildParams
    #: A contestant end of the sandbox does not end the stage by itself: the stage collects the
    #: agent's last commit, and declares the contestant's zero only when nothing is gradable, or
    #: for an end on the sandbox's own memory under a plan that does not grade one.
    decides_contestant_faults: ClassVar[bool] = True
    #: The sandbox runs the agent's code, so every file of an attempt holds bytes the agent
    #: chose, the harness roles' logs and records included (``Stage.runs_contestant_code``).
    runs_contestant_code: ClassVar[bool] = True
    #: The stage's grants are its arm's route, its credential and base URL, which the agent's CLI
    #: holds in the sandbox: a contestant can hold them.
    grants_reach_contestant: ClassVar[bool] = True
    #: Outputs whose bytes the agent wrote: the copy of its logs and its source bundle, read as
    #: stored by a secret scan that reads them (``Stage.raw_outputs``).
    raw_outputs: ClassVar[tuple[str, ...]] = (f"{TRANSCRIPT_DIR}/", SOURCE_BUNDLE_FILE)
    #: The source bundle, a partial one and the snapshots are git packs, compressed: finalize's
    #: scan as stored cannot see inside them, so finalize keeps them out of any export unless a scan
    #: of the bundle's objects read them, which no scan does for a partial one or a snapshot
    #: (``Stage.compressed_files``).
    compressed_files: ClassVar[tuple[str, ...]] = (f"outputs/{SOURCE_BUNDLE_FILE}", f"outputs/{PARTIAL_DIR}/",
                                                   "snapshots/")
    #: Finalize reads none of them, each as large as the sandbox's writable cap: the source_scan
    #: stage's scan of the bundle's objects covers the bundle; a partial bundle, which a commit
    #: point killed with its stage leaves, and a snapshot, present when the commit point bundled
    #: nothing or when its removal failed (``build.snapshots_left``), no scan reads inside, so each
    #: is quarantined (``Stage.unread_files``).
    unread_files: ClassVar[tuple[str, ...]] = compressed_files
    #: A commit point killed outright leaves its partial bundle in the outputs, and a failed attempt
    #: every output it wrote: each file there the published manifest does not hold may hold the
    #: agent's history compressed (``Stage.unpublished_containers``).
    unpublished_containers: ClassVar[tuple[str, ...]] = ("outputs/",)
    #: The transcript's copy is published whole through its index (``Stage.published_dirs``).
    published_dirs: ClassVar[tuple[str, ...]] = (f"outputs/{TRANSCRIPT_DIR}/",)
    #: One retry after an infra fault: a retried build starts from scratch with its whole budget.
    default_retries: ClassVar[int] = 1

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the agent cap's extraction, the copies of its data,
        the workspace's seed, the uploads, the sandbox, a snapshot for every interval of the hard
        limit, the commit point's tries, the transcript copy and harbor's reading of it; and with
        the reference service, its image's extraction and its role."""
        assert isinstance(params, AgentBuildParams)
        snapshots = params.hard_sec // params.snapshot_interval_sec + 1
        service = 2 if params.reference_service is not None else 0
        return 1 + DATA_COPIES + 1 + 1 + 1 + snapshots + COMMIT_TRIES + 1 + 1 + service

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        """The stage's resources are the sandbox's and one harness role's together, with the
        runner's memory beside them and, in disk, the agent cap's extracted tree
        (``KEPT_DISK_GIB``) and the copies of the agent's data (``data_gib``); a site may change
        only its time. The harness role runs on CPUs the
        sandbox's step holds (``overlap``), so the stage's CPUs are the sandbox's, and no more
        than it has; the commit point and the extraction role run once the sandbox has stopped,
        each with the runner's memory beside it. The harness roles ask for no GPU and no disk and
        run on CPUs the sandbox holds; the sandbox declares the disk that caps its writes; the
        collection allowance holds its fixed steps and ``COMMIT_MIN_SEC`` for the commit point, a
        stop build's too, since the brief states it; the time covers the setup allowance, the
        staging, the hard limit and the collection after it (``collection_sec``); the harness
        reserve fits in the sandbox's memory; the producer's network names the packages the
        sandbox's index refuses. A reference service shares every GPU, the CPUs and the memory of
        the sandbox, on no more CPUs than it has, and the stage holds the disk of its image's tree
        and its workdir beside the rest (``hls.stages.reference_service.shape_problems``). That the arm's
        harness can run it as recorded is the record's own rule (``HarnessAdapter.check``), refused
        when the parameters are read."""
        assert isinstance(params, AgentBuildParams)
        p = params
        for field, harness in (("snapshot_shape", p.snapshot_shape), ("commit_shape", p.commit_shape),
                               ("extract_shape", p.extract_shape)):
            if harness.gpus or harness.disk_gib:
                raise ValueError(f"the harness roles beside the sandbox ask for no GPUs and no disk ({field})")
            if harness.cpus > p.sandbox_shape.cpus:
                raise ValueError(f"the harness roles ({field}: {harness.cpus} CPUs) run on CPUs the sandbox holds "
                                 f"({p.sandbox_shape.cpus})")
        if p.sandbox_shape.disk_gib < 1:
            raise ValueError("the sandbox must declare the disk its writable directories may fill "
                             "(sandbox_shape.disk_gib)")
        roles = concurrent([p.sandbox_shape, p.snapshot_shape])
        want = roles.model_copy(update={"cpus": p.sandbox_shape.cpus,
                                        "memory_gib": roles.memory_gib + runner_memory_gib,
                                        "disk_gib": roles.disk_gib + KEPT_DISK_GIB + p.data_gib,
                                        "time_sec": shape.time_sec})
        service = ""
        if p.reference_service is not None:
            problems = shape_problems(p.reference_service.shape, p.sandbox_shape)
            if problems:
                raise ValueError("; ".join(problems))
            # Oversubscribed on the sandbox's memory, its own limit must still fit the job's.
            if p.reference_service.shape.memory_gib + runner_memory_gib > shape.memory_gib:
                raise ValueError(f"the reference service's {p.reference_service.shape.memory_gib} GiB and the runner's "
                                 f"{runner_memory_gib} GiB do not fit in the stage's {shape.memory_gib} GiB")
            from hls.verify.base import WiringError

            try:
                engine = rs.service_engine_for(load_task(p.task))
            except WiringError as e:
                raise ValueError(str(e)) from e
            if engine.tensor_parallel_size != p.reference_service.shape.gpus:
                raise ValueError(f"the reference service's engine spans {engine.tensor_parallel_size} GPUs, and it "
                                 f"shares the sandbox's {p.reference_service.shape.gpus}")
            # The service shares the sandbox's GPUs, CPUs and memory; its image's tree and its
            # workdir take disk of their own.
            held = p.reference_service.shape
            want = want.model_copy(update={"disk_gib": want.disk_gib + KEPT_DISK_GIB + held.disk_gib})
            service = f", and the reference service's image's tree and {held.disk_gib} GiB of its disk"
        if shape != want:
            raise ValueError(f"the stage's resources must be the sandbox's, with a harness role's memory, "
                             f"{runner_memory_gib} GiB more memory for the runner, {KEPT_DISK_GIB} GiB more disk "
                             f"for the agent cap's tree and {p.data_gib} GiB for the copies of its data{service}, "
                             f"{want}, in all but time; they are {shape}")
        for what, after in (("the commit point", p.commit_shape), ("the extraction role", p.extract_shape)):
            if after.memory_gib + runner_memory_gib > shape.memory_gib:
                raise ValueError(f"{what}'s {after.memory_gib} GiB and the runner's {runner_memory_gib} do not fit "
                                 f"in the stage's {shape.memory_gib} GiB")
        least = COLLECT_STEPS_SEC + COMMIT_MIN_SEC
        if commit_point_sec(p.collect_allowance_sec) < COMMIT_MIN_SEC:
            raise ValueError(f"the collection allowance (run.collect_allowance_sec) is {p.collect_allowance_sec}s, "
                             f"under the {least}s a build's collection needs: {COLLECT_STEPS_SEC}s for its fixed "
                             f"steps (the sandbox's and the snapshot's stops, the sandbox's log scrub, the wait for "
                             f"its processes to be gone, the transcript copy and the extraction) and at least "
                             f"{COMMIT_MIN_SEC}s for the commit point; set it to {least} or more")
        floor = p.setup_allowance_sec + p.staging_sec + p.hard_sec + p.collection_sec
        if shape.time_sec < floor:
            raise ValueError(f"the agent_build stage's time {shape.time_sec}s is under its floor of {floor}s: the "
                             "setup allowance, the staging, the hard limit and the collection after it")
        if p.harness_reserve_gib >= p.sandbox_shape.memory_gib:
            raise ValueError(f"the harness reserve ({p.harness_reserve_gib} GiB) must be under the sandbox's memory "
                             f"({p.sandbox_shape.memory_gib} GiB)")
        deny_list(NetworkPolicy.model_validate(p.network))

    @classmethod
    def declared_grants(cls, params: BaseModel, settings: Any) -> dict[str, str]:
        """The arm's route as the variables its harness reads, its credential and base URL, by
        the names of their secrets. Raise ValueError for a record whose route
        table *settings* do not declare (``hls.arms.route_of``)."""
        p = AgentBuildParams.model_validate(params) if not isinstance(params, AgentBuildParams) else params
        return adapter_for(p.arm).grants(route_of(settings, p.arm))

    @classmethod
    def seals_containers(cls, params: BaseModel | None) -> bool:
        return True

    @classmethod
    def check_site(cls, shape: ResourceShape, params: Any, *, site: SiteSettings) -> None:
        """The sandbox's spawner makes the commands' user, mount and PID namespaces: a host where
        *site* runs its allocations that the planner can tell would refuse them is refused here,
        before anything is submitted (``Substrate.spawner_problem``)."""
        problem = substrate(site).spawner_problem()
        if problem is not None:
            raise ValueError(f"the build sandbox cannot run on this site: {problem.reason}")

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Build(ctx).run()
