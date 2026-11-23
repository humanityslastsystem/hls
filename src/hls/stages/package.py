"""The package stage: a contestant's source bundle into the sealed submission image a grading
serves, or the published result that the submission does not build.

The stage runs in the runner and stays light; every step is a role:
1. the submission base is pinned on the node (``StageContext.node_image``) by the entry digest
   this instance's images stage recorded for it in ``images.json``, before any other role
   starts: a pin starts a role of its own and waits for it, so the stage pins what it can
   before its first role. It copies no cache entry;
2. ``checkout`` (harness, host side): the bundle's ``main`` into a node-local tree with a
   hardened git (``hls.package.checkout``), once git's own listing shows the tree within the
   checkout's limits (depth, path length, paths, bytes, no symlink). The bundle is the copy of
   the operator's named bundle that ``hls package`` made in the instance before planning it
   (``params.bundle``, with the digest the operator stated, ``params.bundle_sha256``) when given,
   else the ``source_bundle`` output of the agent_build manifest the plan pins
   (``params.build_manifest``, a grading that packages a rollout's source again) or of this
   instance's ``agent_build``. The role checks the bundle's size before any read and its digest
   before any verdict: a bundle whose size is not the one the plan or its manifest records is
   refused unread; one over the byte limit is the submission's by its size alone, unread, which
   ``hls package`` refuses for a named bundle before planning. Any other bundle it reads once where
   it lies in the store, hashing it as it streams it into git's clone, with no copy on the node,
   and checks it against the stated or the manifest's digest before any verdict. The source_scan
   stage (``hls.stages.source_scan``)
   has already read this bundle, every object of it and its tree, for every value of the site's
   secrets file, and published what it found before this stage runs any of the contestant's code;
3. the stage checks the tree: a ``pyproject.toml`` that is a regular file of at most
   ``PYPROJECT_MAX_BYTES`` and parses as TOML, the submission contract (``pyproject.toml`` and
   the server file), the declared dependencies against the producer's package rule, and a valid
   ``[project].name``;
4. ``dry-run`` and ``wheel`` (contestant, sealed, on the submission base): the two pip steps
   that run the project's build hooks (``hls.package.wheels``). Each sees only its own copy of
   the tree, made by a harness role (``<step>-copy``, ``cp -a``), its own temporary directory
   and, for the wheel step, its wheels directory, all in its own node-local role directory, and
   reaches the network only through the seal and its package index, which refuses the
   producer's own deny list; the seal also refuses the step's own DNS. Once a step has ended, a
   harness role (``<step>-clean``) gives the owner back every permission on all the step could
   write, never through a link, then removes its copy and its temporary directory, so no two
   steps' writes are on the node at once and nothing the step locked fails a later read of its
   wheels;
5. ``wheel-check`` (harness, host side), once the stage has found only the project's wheel and
   one wheel per pin there: every wheel read whole with bounded memory
   (``hls.package.wheel_check``), so each entry pip or the integrity check reads whole is the
   size it declares: a wheel that is not a readable zip, declares more than the install may add,
   holds a metadata entry or a module over its limit or an entry that is not the size and CRC it
   declares, or wheels that together name more paths than the install may write, are the
   submission's;
6. ``build`` (harness, host side): the offline build of the image from the submission base and
   the wheels (``hls.package.offline``). It reads the base's image file from its entry in the
   store once, into its own directory, and builds from that copy only if it has the digest the
   entry records. It runs no contestant code, installs what the contestant supplied only into
   the base's venv and ``/workspace``, under the base's constraints file (``params.constraints``,
   none for a published base), and reports the new image file's digest and the bytes the
   install added to the venv, which ``INSTALL_MAX_GIB`` bounds;
7. the new image is pinned on the node by that digest (``pin_file``), once no other role runs,
   and ``integrity`` (harness, host side) compares its system paths with the base's, byte for
   byte (``hls.package.integrity``); once they are the base's, ``install-scan`` reads what the
   install added to the image (step 9);
8. ``publish`` (harness, host side, ``hls.package.publish``) copies the image to the attempt's
   output ``submission.<suffixes>``, named after the image runtime's file (``submission_file``:
   ``submission.sif`` on Apptainer), and reports the copy's digest, which must be the one the
   image was pinned by;
9. ``install-scan`` (harness, host side, ``hls.package.scan``): the image is squashfs, compressed,
   so a secret's value in it is not in the bytes finalize's own scan reads. The role reads every
   file under the extracted image's venv that the base does not hold as it is, the wheels' entries
   and the bytecode pip compiled among them, for every form of every secret of the site's secrets
   file (``hls.substrate.secret_forms``), and writes ``package-scan.json`` (``PackageScan``, area
   ``install``): each file a form was found in, by its path in the image, with the form's label,
   never a value or a host, up to a bound for each label, past which it records the area and how
   many of its files hold the form, for finalize to keep the whole area out of any export. It runs
   right after the integrity check exited 0: a role that ends with a fault of its own makes the
   context stop every role started after it. The values
   reach the role only in its 0600 env file. On a site whose secrets file holds no secret nothing
   is read for one: the stage writes a record that searched no secret and read nothing, and starts
   no role.

The stage sets ``needs_secret_registry``: the runner hands it every value of the site's secrets
file, which the scan needs, and a stage the runner did not hand the registry refuses to run,
before any role. The stage runs contestant code, so every file it writes is one a contestant's
bytes reach, and ``package-scan.json`` needs no declaration of its own. It declares its image
compressed (``compressed_files``): finalize keeps the image out of any export unless this scan
and source_scan's read it.

An agent_build that left nothing to package (no commit on main, or a repository its commit point
could not read because of what the contestant made in it) publishes ``no_submission`` in place of
the bundle (``hls.agent.commit``). The stage then publishes that as the submission's result,
scrubbed, before it pins the base or starts any role. A read of the workspace that the machine
failed never comes here: agent_build tries its commit point again.

Every output a role writes reaches the manifest by that role's own report of its sha256 and size
(``StageContext.reported_output``): ``source.json`` by ``checkout``, ``integrity.json`` by
``integrity``, the submission image by ``publish``, ``package-scan.json`` by ``install-scan``. The
stage writes only its small records,
``install-report.json`` and ``package.json`` (``write_output``). The runner reads no byte of the
bundle or an image, and of what the roles wrote only the small records it decides by: the
harness roles' ``source.json``, ``check.json`` and ``integrity.json``, and the tree's
``pyproject.toml`` and the dry run's report, held to ``PYPROJECT_MAX_BYTES`` and
``REPORT_MAX_BYTES`` and to being regular files (``pyproject_problem`` judges the first from
``lstat`` before anything opens it; the report is judged by ``lstat`` too, then read through one
descriptor that follows no link and waits on no pipe). It reads no wheel. Only the
submission chose those two files' bytes, so any way they fail to parse is its own, a
``pyproject.toml`` nested deeper than the parser recurses, a report nested deeper than
``REPORT_MAX_DEPTH`` (refused before it is parsed) or either too big to hold among them: each
parse site says so
(``hls.egress.prescan``, ``project_name``, ``read_report``, ``report_pins``), and none of it
reaches the runner as an error of the stage's, which would be retried. An error reading what
the submission produced (those two files, the wheels and their directory) is its own only when
the permissions it left caused it (``PERMISSION_ERRNOS``: EACCES, EPERM); any other (EIO,
ESTALE, ENOSPC and the rest) is the node's, a retryable fault with the errno in its reason.

Whose failure it is (the contestant's only when
certain):
- a build that left nothing to package (``no_submission``),
  a bundle without ``main``, a bundle git does not read back, every byte it read proven the
  bundle's (``hls.package.checkout.bundle_verdict``), a bundle or a tree past the checkout's
  limits (a tree that holds any symlink among them), a tree that breaks the contract or whose
  ``pyproject.toml`` is not a regular file, past its size, kept out by its permissions or does
  not parse, a dry-run report that is missing, passes its size, is not a regular file, is kept
  out by its permissions or does not parse as pip's, a pip step whose payload exited nonzero (by
  the seal's own record, with its interceptor alive) while the index was not out (``index_outage``) and its interceptor
  answering when the payload ended (``interceptor_stale``, ``step_ended_at``), its reason naming
  each simple page the index refused for its size and the fetches it refused past its cap
  (``oversized_reason``), a pip step still
  running once the time both may take together is spent (``PIP_STEPS_SEC``, in their commands'
  own time, ``PipBudget``, which the stage's time holds) while the index counted no upstream
  failure and its interceptor answered, a wheel output
  that is not plain wheel files, holds anything but the project's wheel and one wheel per pin
  (``wheels_problem``) or is kept out by its permissions, a wheel the wheel check refuses, an
  install that adds more than ``INSTALL_MAX_GIB`` to the venv, an offline install that fails by
  the build's own status line, and system paths that differ from the base's only as
  installing the submission's wheels changed them are the submission's. The stage publishes
  ``package.json`` with status ``invalid_artifact`` and the reason; a grading scores it zero and
  nothing retries it;
- a role's own fault is the roles' to decide (``stage_fault``): a contestant step that ran out
  of memory with its own limit certainly reached is a contestant fault, which ends the stage
  and is never retried, and every other role fault is infra, among them a copy of a tree that
  failed (the checkout held the tree to its limits, so a copy cannot fail on its account). Once
  a contestant fault is decided, the stage raises it and nothing of its own;
- everything else is infra and raises a fault of the stage's own: a pip step cut by the
  stage's deadline before the steps' limit, a pip step that fails or reaches that limit while the
  index was out, wrote no count or had stopped answering, an error
  of the node reading what the submission produced, an offline build that fails outside its
  install or runs out of memory (past the wheel check, everything it reads whole is bounded),
  an image file or a bundle that is not the one its record or its operator's stated digest
  names, inputs that contradict the plan, and system paths that differ in a way installing the
  submission's wheels does not account for.

The index's evidence comes from the status file each container's interceptor writes on the
host side of the seal, which no container can reach; the files are cleared before every
launch, so the count starts at zero for each step. A step's status and stderr are its own
code's to choose, so no exit status or ``refused:`` line of a pip step is ever read as a
refusal (``StageContext.refusal`` reads harness roles only): whose failure a step is comes from
the harness's evidence alone, the role's ending, the seal's record and the index's count."""

from __future__ import annotations

import bisect
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, ClassVar, Literal, NamedTuple, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from hls.agent.commit import NoSubmission
from hls.egress.policy import NetworkPolicy, SealPolicy, pep503_normalize
from hls.egress.prescan import prescan
from hls.egress.proxy import UPSTREAM_FETCHES_MAX
from hls.package.checkout import (CHECKOUT_MEMORY_GIB, CHECKOUT_SEC, PYPROJECT_MAX_BYTES, TREE_MAX_BYTES,
                                  SourceRecord, checkout_disk_gib)
from hls.package.integrity import DIFFERS_EXIT, IntegrityRecord
from hls.package.offline import (
    INSTALL_MAX_GIB,
    install_failed,
    installed_bytes,
    new_nonce,
    project_name,
)
from hls.package.result import PackageResult
from hls.package.wheel_check import WheelCheck
from hls.package.wheels import (
    BUILD_TMP,
    PIP_STEPS_SEC,
    REPORT_MAX_BYTES,
    REPORT_MAX_DEPTH,
    container_env,
    dry_run_command,
    missing_requirements,
    pin_name,
    wheel_command,
)
from hls.pipeline.stage import Stage, StageContext, StageFault, register, scored_end_fault, stage_fault
from hls.settings import StageName, TaskName
from hls.spec.resources import GpuKind, ResourceShape
from hls.spec.submission import PERMISSION_ERRNOS, SubmissionContract, SubmissionError
from hls.stages.images import BUILD_DISK_GIB, KEPT_DISK_GIB
from hls.stages.provision import role_shape
from hls.store.base import check_key, check_segment
from hls.store.digest import DIGEST_MISMATCH_EXIT, Sha256Hex, reported_sha256, sha256_bytes
from hls.store.manifest import Manifest, ManifestKey, read_outputs_report
from hls.substrate.seal import IndexStatus, SealFiles, payload_ended_at
from hls.substrate.container import ContainerSpec, Mount, PinnedImage, SealPlan
from hls.substrate.factory import image_files
from hls.substrate.faults import RoleOutcome
from hls.substrate.nproc import ProcessCeiling
from hls.substrate.secret_forms import FORMS
from hls.substrate.secret_scan import PACKAGE_SCAN_FILE, PackageScan
from hls.substrate.ucounts import ServedUcounts
from hls.util import load_json, validation_errors

#: The stage whose output this stage reads the bundle from when ``params.bundle`` is not set.
AGENT_BUILD_STAGE = "agent_build"
#: The name of that stage's bundle output.
SOURCE_BUNDLE_OUTPUT = "source_bundle"
#: The name of the output that stage publishes in place of a bundle when there is nothing to
#: package (``hls.agent.commit.NoSubmission``).
NO_SUBMISSION_OUTPUT = "no_submission"
#: The agent_build output that records the figures its brief stated for its graded server
#: (``hls.substrate.ucounts.ServedUcounts``), which ``package.json`` carries to the grading.
SERVED_UCOUNTS_OUTPUT = "served_ucounts"
#: The images stage's output naming each image's key, entry digest and file digest.
IMAGES_OUTPUT = "images"
def submission_file(image_file: str) -> str:
    """The attempt output holding the built image on a substrate whose image runtime keeps an image
    in the file *image_file* (``ImageRuntime.image_file``): ``submission`` with that file's
    suffixes, ``submission.sif`` beside Apptainer's ``image.sif``."""
    return "submission" + "".join(PurePosixPath(image_file).suffixes)


#: The attempt output holding the built image on each substrate (``submission_file``), which the
#: stage's kind declares to finalize for an instance of any of them.
SUBMISSION_FILES = tuple(sorted({submission_file(name) for name in image_files().values()}))
#: The name of that output in the stage's manifest, by which a drive and a planner read it.
SUBMISSION_OUTPUT = "submission"
#: The attempt output recording what the stage built (``PackageResult``).
PACKAGE_FILE = "package.json"
#: The attempt output of the integrity check.
INTEGRITY_FILE = "integrity.json"
#: The role that scans what the install added to the image for the secrets' values
#: (``hls.package.scan``).
SCAN_ROLE = "install-scan"
#: The most a package adds to its submission base's image file and to the tree extracted from it:
#: what the install adds to the venv, which this stage holds to ``INSTALL_MAX_GIB``, and the
#: committed tree, which the checkout holds to ``TREE_MAX_BYTES``.
PACKAGE_ADDS_BYTES = INSTALL_MAX_GIB * 2**30 + TREE_MAX_BYTES
#: Node-local disk the checked-out tree holds at most, in GiB: ``TREE_MAX_BYTES``, rounded up.
TREE_DISK_GIB = -(-TREE_MAX_BYTES // 2**30)
#: The least node-local disk a pip step's container may write, in GiB: its copy of the tree at the
#: tree's limit and wheels that declare the most the install may add (``INSTALL_MAX_GIB``), so a
#: submission within those limits, which the brief states, never runs out of disk for them. Its
#: temporary files get what the stage's declared disk gives beyond this (``pip_limits``).
PIP_DISK_FLOOR_GIB = TREE_DISK_GIB + INSTALL_MAX_GIB
#: Node-local disk for the work at its least: the checked-out tree, and one pip step's writes at
#: their floor. A step's copy and temporary files are removed when it ends, so the two steps'
#: writes never coexist.
WORK_DISK_GIB = TREE_DISK_GIB + PIP_DISK_FLOOR_GIB
#: Free node-local disk the stage needs once the checkout has ended: a build's unpacking and its
#: image file, the base's and the submission's extracted trees, which live for this stage and go
#: when it closes, the work above, and three more times the most the install may add to the venv,
#: which is in the built image file, the submission's extraction copy and its tree at once.
BUILD_STEPS_DISK_GIB = BUILD_DISK_GIB + 2 * KEPT_DISK_GIB + WORK_DISK_GIB + 3 * INSTALL_MAX_GIB
#: Node-local disk the harness holds beside a pip step's writes at its most, in GiB: everything
#: the stage's disk holds once the checkout has ended but those writes. The checked-out tree and
#: the base's extracted tree are on the node while a step runs; the build's unpacking, its image
#: file and the submission's tree come after it, beside the wheels it left. Whatever a stage
#: declares beyond this is each step's (``pip_limits``), the checkout's own room at a larger cap
#: among it, since the clone is gone by then.
HARNESS_DISK_GIB = BUILD_STEPS_DISK_GIB - PIP_DISK_FLOOR_GIB
#: Seconds the stage allows beside the pip steps and the checkout, for the build, the
#: extractions, the integrity check and the publish.
REST_SEC = 2 * 3600


def min_disk_gib(disk_cap_gib: int) -> int:
    """Free node-local disk the stage needs for a bundle of a build whose writable disk cap is
    *disk_cap_gib*: the checkout runs once the submission base is extracted and before anything
    else, so the base's tree and the checkout's disk at the cap
    (``hls.package.checkout.checkout_disk_gib``), or ``BUILD_STEPS_DISK_GIB`` once it has ended,
    whichever is more."""
    return max(BUILD_STEPS_DISK_GIB, KEPT_DISK_GIB + checkout_disk_gib(disk_cap_gib * 2**30))


#: The least time the stage may declare: the pip steps' four hours (``PIP_STEPS_SEC``), the
#: checkout's clone (``hls.package.checkout.CHECKOUT_SEC``), which runs to the stage's deadline,
#: and ``REST_SEC``, all fixed allowances.
MIN_TIME_SEC = PIP_STEPS_SEC + CHECKOUT_SEC + REST_SEC
#: Seconds the index's interceptor may have gone without writing its status when its step's
#: payload ends before the step's failure is not certainly the submission's: it writes every few
#: seconds while it runs (``hls.egress.proxy``), so a minute without one means it stopped
#: answering.
HEARTBEAT_STALE_SEC = 60
#: Bytes from the end of a failed step's output kept in the record.
LOG_TAIL_BYTES = 8192
#: Bytes from the end of each of the offline build's two streams searched for its status lines,
#: which ``%post`` prints after pip has exited: only the harness's own steps print after them.
BUILD_LOG_BYTES = 4 * 2**20
#: Characters of why the dry run's report cannot be used kept in an invalid package's reason: a
#: pin the report names can be as long as the report.
REPORT_REASON_CHARS = 1000
#: Where the contestant's tree is inside the build containers and the image.
WORKSPACE = "/workspace"
#: Where the wheel step writes the wheels.
WHEELS_OUT = "/wheels-out"
#: Differing paths named in an invalid package's reason or an infra fault.
NAMED_PATHS = 5
#: Names of what the wheel step left that an invalid package's reason gives, the first by name,
#: beside how many there are: a build hook can leave millions, and the reason is published.
NAMED_ENTRIES = 10
#: Characters of each of those names kept in the reason.
NAME_CHARS = 200
# A wheel's file name: the distribution, the version, an optional build tag, then its three tags.
_WHEEL_FILE = re.compile(r"([^-]+)-[^-]+(?:-[^-]+)?-[^-]+-[^-]+-[^-]+\.whl")
#: The step of a clean role, given a count *n* and then paths: give the owner back every
#: permission on each path and what lies under it, never through a link, then remove all the
#: paths but the first *n*, which the stage reads next.
CLEAN_SHELL = 'keep=$1; shift; chmod -R u+rwX -- "$@" && shift "$keep" && rm -rf -- "$@"'
# Whose failure a step is: the submission's ("invalid"), ours ("infra"), or none ("ok").
Verdict = Literal["ok", "invalid", "infra"]


def _node_fault(what: str, error: OSError) -> StageFault:
    """The retryable fault of the node's *error* reading *what*, which the submission produced
    but whose permissions do not explain the error, with its errno."""
    return StageFault.infra(f"{what} cannot be read: {error}", retryable=True, errno=str(error.errno))


class PipLimits(BaseModel):
    """What each of a package's two pip steps may use, which the build's brief states and the
    stage holds the steps to; a step past either is stopped, a contestant fault."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Node-local disk the step's container may write, in GiB: its copy of the tree, its temporary
    #: files and the wheels together.
    disk_gib: int = Field(gt=0)
    #: Memory the step's processes may hold, in GiB.
    memory_gib: int = Field(gt=0)


def pip_limits(shape: ResourceShape, runner_memory_gib: int) -> PipLimits:
    """What each pip step of a package stage that declares *shape* gets, its runner keeping
    *runner_memory_gib*: the disk the stage declares beyond what the harness holds beside a step
    (``HARNESS_DISK_GIB``), and the memory each of the stage's roles gets (``role_shape``). One
    owner of the figures: the planner states them to the stage and to the build's brief, and the
    stage's check refuses a declared shape that gives less. Raise ValueError when the runner's
    reserve leaves the roles no memory."""
    return PipLimits(disk_gib=shape.disk_gib - HARNESS_DISK_GIB,
                     memory_gib=role_shape(shape, runner_memory_gib).memory_gib)


class StoredBundle(BaseModel):
    """A source bundle its operator named, as ``hls package`` copied it into the instance before
    planning it (``InstanceLayout.source_bundle``), hashing it as it copied and checking it
    against the digest the operator stated. The stages read this copy, never the operator's file,
    on every substrate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Store key of the copy.
    key: Annotated[str, AfterValidator(check_key)]
    #: Its size in bytes as copied, which the checkout checks before it reads a byte.
    size: int = Field(ge=0)


class PackageParams(BaseModel):
    """What the package stage builds, and from what."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task the submission is for, recorded in ``package.json``.
    task: TaskName
    #: What each pip step may use, as the plan states it: the figures the build's brief stated
    #: (``pip_limits`` of the package shape its plan declared, which a rebuild reads from the pinned
    #: build's plan, ``AgentBuildParams.package_pip``), or for a named bundle, which no brief stated
    #: figures for, ``pip_limits`` of the stage's declared shape. The stage holds the steps to
    #: exactly these, and its shape must give them (``PackageStage.check``).
    pip: PipLimits
    #: The catalog name of the submission base image (base plus the task's layer), whose key and
    #: entry digest this instance's images stage records in ``images.json``.
    submission_base: Annotated[str, AfterValidator(check_segment)]
    #: The submission base's constraints file, an absolute path in the image, which the offline
    #: install applies (``hls.recipes.catalog.TaskImages.submission_constraints``); None for a
    #: base that holds none, as a published image does.
    constraints: str | None
    #: The stage of this instance whose ``images.json`` names the submission base.
    images_stage: StageName = "images"
    #: The producer's network policy (``NetworkPolicy`` JSON): what the seal lets the pip
    #: steps reach, and the package rule the declared dependencies are checked against.
    network: dict[str, Any]
    #: The run's GPU kind, whose compute capability a CUDA extension is compiled for; None for
    #: a task without GPUs.
    gpu_kind: GpuKind | None = None
    #: The server file the image runs, relative to the tree.
    server_filename: str = "server.py"
    #: The source bundle its operator named, as ``hls package`` copied it into the store; None to
    #: read an agent_build output (``build_manifest``).
    bundle: StoredBundle | None = None
    #: The sha256 the operator states for ``bundle``, which the copy had when it was made and
    #: which what the checkout reads of it must have; None exactly when ``bundle`` is, since the
    #: agent_build manifest records its own.
    bundle_sha256: Sha256Hex | None = None
    #: Another instance's agent_build manifest, pinned in the stage's inputs, whose bundle (or its
    #: record that there is none) is packaged: a grading that packages a rollout's source again.
    #: None to read this instance's agent_build stage.
    build_manifest: ManifestKey | None = None
    #: The writable disk cap of the build the bundle is from, in GiB
    #: (``hls.pipeline.pipelines.sandbox_disk_gib``): the most bytes the bundle, and its objects
    #: inflated, may hold, as the commit point held the repository to it.
    disk_cap_gib: int = Field(gt=0)

    @model_validator(mode="after")
    def _checked(self) -> Self:
        NetworkPolicy.model_validate(self.network)
        check_named_bundle(self.bundle, self.bundle_sha256, self.build_manifest)
        if self.constraints is not None and not Path(self.constraints).is_absolute():
            raise ValueError(f"constraints {self.constraints!r} must be an absolute path in the image")
        return self


def check_named_bundle(bundle: StoredBundle | None, bundle_sha256: str | None,
                       build_manifest: str | None = None) -> None:
    """Raise ValueError unless *bundle* is None, with no *bundle_sha256*, or a stored copy with the
    sha256 its operator states, and unless at most one of a named *bundle* and a pinned
    agent_build manifest (*build_manifest*) says what to read."""
    if bundle is not None and bundle_sha256 is None:
        raise ValueError(f"a named bundle needs the sha256 its operator states; bundle {bundle.key} has none")
    if bundle is None and bundle_sha256 is not None:
        raise ValueError("bundle_sha256 is the digest of a named bundle, and no bundle is named")
    if bundle is not None and build_manifest is not None:
        raise ValueError("a stage reads a named bundle or an agent_build manifest's, not both")


def deny_list(policy: NetworkPolicy) -> tuple[str, ...]:
    """The package names *policy*'s producer declares denied, which the pip steps' seal must
    refuse: its package-deny rule's entries. Raise ValueError for a policy with no package-deny
    rule, or one that names nothing: the steps' index would let any package through."""
    rule = policy.packages
    if rule is None or rule.mode != "deny" or not rule.entries:
        raise ValueError("the producer's network has no package-deny rule naming a package; the pip steps run "
                         "behind a seal with an index, which must refuse the producer's own deny list")
    return tuple(rule.entries)


def index_outage(errors: IndexStatus) -> bool:
    """Whether the index failed a step on its own account, by its interceptor's count *errors*:
    its upstream failed at all. The interceptor counts only what the step cannot bring about: a
    failure while a file streams counts only when the step had read all it was sent promptly
    (``hls.egress.proxy``), since one that stalls its download can leave the upstream waiting
    until it gives up."""
    return errors.total > 0


def oversized_reason(status: IndexStatus | None) -> str:
    """The end of an invalid package's reason naming each simple page the index refused for its
    size, with what was read of it and the cap, and the fetches it refused past the cap on one
    step's fetches (``UPSTREAM_FETCHES_MAX``); empty when it refused none. A page's size is
    PyPI's doing, but a dependency the index will not serve fails the package all the same."""
    if status is None:
        return ""
    pages = "".join(f"; the package index refused the simple page {page.page} for its size: more than its cap of "
                    f"{page.cap} bytes ({page.read} read before it stopped)" for page in status.oversized)
    fetches = (f"; the package index refused {status.fetches_refused} fetches from its upstream past the "
               f"{UPSTREAM_FETCHES_MAX} one step may make") if status.fetches_refused else ""
    return pages + fetches


def interceptor_stale(status: IndexStatus, ended_at: float) -> bool:
    """Whether the index's interceptor, by its *status*, had stopped answering when its step
    ended at *ended_at* (``step_ended_at``): its heartbeat is more than ``HEARTBEAT_STALE_SEC``
    older."""
    return ended_at - status.heartbeat > HEARTBEAT_STALE_SEC


def step_ended_at(files: SealFiles | None, seen: float) -> float:
    """When a sealed pip step ended, for judging its interceptor's heartbeat: when its payload
    ended, by the seal's record of it (``payload_ended_at``), since the seal stops the
    interceptor right then; else *seen*, when the stage saw the step end, for a step with no
    record (stopped at its limit, its payload never started, or its seal ended first). The
    stage may see a step end long after: reporting it waits on its logs and the scheduler."""
    ended = None if files is None else payload_ended_at(files)
    return seen if ended is None else ended


class PipBudget:
    """The time the two pip steps may run together, counted in their commands' own time: from
    when the stage saw a step's command start (``StageContext.command_started_at``) to when its
    payload ended (``step_ended_at``), not the runtime's start of its container nor the harness
    roles' between the steps. The dry run may take all of it, and the wheel step what the dry
    run left."""

    def __init__(self, seconds: float) -> None:
        #: Seconds not yet spent: what the next step's command may run.
        self.left = seconds

    def spend(self, started: float, ended: float) -> None:
        """Take the time of a command that ran from *started* to *ended*; none for one first
        seen started after it ended."""
        self.left = max(0.0, self.left - max(0.0, ended - started))


def contestant_verdict(outcome: RoleOutcome, *, upstream: IndexStatus | None, whole_limit: bool,
                       ended_at: float) -> Verdict:
    """Whose failure a contestant pip step is, when no role fault was decided for its ending
    *outcome* (``stage_fault``, which already judged the seal's record of the payload's own
    ending). *upstream* is the index's count at the end, None when its interceptor wrote none;
    *whole_limit* says the step had all the pip steps' budget left to it (``PipBudget``) before
    the stage's deadline; the step ended at *ended_at* (``step_ended_at``). A step stopped at
    that limit spent it in the submission's build hooks unless the index may have failed it
    meanwhile (``_index_failed``: its upstream failed, it wrote no count, or its interceptor
    stopped answering): the submission's then, as a step that failed."""
    if outcome.ending == "time_limit" and whole_limit and outcome.oom_kills == 0:
        return "infra" if _index_failed(upstream, ended_at) else "invalid"
    if outcome.ending != "exited" or outcome.oom_kills > 0:
        return "infra"
    if outcome.exit_code == 0:
        return "ok"
    return "infra" if _index_failed(upstream, ended_at) else "invalid"


def _index_failed(upstream: IndexStatus | None, ended_at: float) -> bool:
    """Whether the index may have failed a step that ended at *ended_at*, by its status
    *upstream*: it wrote none, its upstream failed (``index_outage``), or its interceptor had
    stopped answering (``interceptor_stale``)."""
    return upstream is None or index_outage(upstream) or interceptor_stale(upstream, ended_at)


def build_verdict(outcome: RoleOutcome, log: str, nonce: str) -> Verdict:
    """Whose failure the offline build's ending *outcome* is, given its output *log* from a
    ``.def`` rendered with *nonce*: the submission's when the install failed by the build's own
    status line or added more than ``INSTALL_MAX_GIB`` to the venv, which ends the build; none
    for a build that exited 0 otherwise; ours for any other ending."""
    if outcome.ending != "exited" or outcome.oom_kills > 0:
        return "infra"
    added = installed_bytes(log, nonce)
    if install_failed(log, nonce) or (added is not None and added > INSTALL_MAX_GIB * 2**30):
        return "invalid"
    return "ok" if outcome.exit_code == 0 else "infra"


def integrity_verdict(record: IntegrityRecord) -> Verdict:
    """Whose failure the integrity check's *record* is: none when the system paths are the
    base's; the submission's when installing its wheels accounts for every difference; ours
    else."""
    if record.differences == 0:
        return "ok"
    return "invalid" if record.unattributed == 0 else "infra"


def _tail(path: Path, size: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, os.path.getsize(path) - size))
            return f.read().decode(errors="replace")
    except FileNotFoundError:
        return ""


class _Bundle(NamedTuple):
    """The source bundle a package reads, and what its record says it is."""

    #: Where the file is in the store: the copy of the operator's named bundle, or the
    #: agent_build output.
    path: Path
    #: The sha256 it must have: the operator's stated one, or the manifest's.
    sha256: str
    #: The size it must have: the copy's as the plan records it, or the manifest's.
    size: int


def instance_manifest(ctx: StageContext, stage: str) -> Manifest | None:
    """This instance's manifest of *stage* among *ctx*'s inputs; None when there is none."""
    found = [m for m in ctx.inputs.values() if m.stage == stage and m.instance == ctx.plan.instance]
    return found[0] if len(found) == 1 else None


def built_manifest(ctx: StageContext, build_manifest: str | None) -> Manifest | None:
    """The agent_build manifest whose bundle, or record that there is none, a stage reads: the one
    *build_manifest* pins among the stage's inputs, which the runner checked against the plan's
    digest, else this instance's; None when this instance has none. Raise StageFault, infra not
    retried, for a pinned manifest that is not among the inputs or is not an agent_build's."""
    if build_manifest is None:
        return instance_manifest(ctx, AGENT_BUILD_STAGE)
    pinned = ctx.inputs.get(build_manifest)
    if pinned is None:
        raise StageFault.infra(f"the agent_build manifest {build_manifest} the plan pins is not among the stage's "
                               "inputs", retryable=False)
    if pinned.stage != AGENT_BUILD_STAGE:
        raise StageFault.infra(f"{build_manifest} is the manifest of stage {pinned.stage}, not of an agent_build",
                               retryable=False)
    return pinned


def source_bundle(ctx: StageContext, bundle: StoredBundle | None, bundle_sha256: str | None,
                  build_manifest: str | None = None) -> _Bundle:
    """The source bundle and what it must be: for a named *bundle* its copy in the store, the
    sha256 its operator stated (*bundle_sha256*) and the size the plan records, else the sha256
    and size the agent_build manifest records for its output, the one *build_manifest* pins or
    this instance's (``built_manifest``). The checkout role reads the file and checks it; this
    process reads none of it."""
    if bundle is not None:
        assert bundle_sha256 is not None  # check_named_bundle requires it with a named bundle
        path = ctx.store.path(bundle.key)
        if not path.is_file():
            raise StageFault.infra(f"the copy {bundle.key} of the source bundle does not exist", retryable=False)
        return _Bundle(path, bundle_sha256, bundle.size)
    built = built_manifest(ctx, build_manifest)
    if built is None or SOURCE_BUNDLE_OUTPUT not in built.outputs:
        raise StageFault.infra("no source bundle: params.bundle is unset and agent_build published none",
                               retryable=False)
    ref = built.outputs[SOURCE_BUNDLE_OUTPUT]
    return _Bundle(ctx.store.path(ref.key), ref.sha256, ref.size)


def served_record(ctx: StageContext, bundle: StoredBundle | None,
                  build_manifest: str | None = None) -> ServedUcounts | None:
    """The figures the agent_build the stage reads (``built_manifest``) stated in its brief for its
    graded server, its ``served_ucounts`` output read by the digest its manifest records; None for
    a named *bundle*, which no brief of a build stated figures for, and for a rollout's build the
    plan pins (*build_manifest*) that published none: the grading computes
    the figures for either (``StageContext.grading_ucounts``). The record is the harness's own, so
    this instance's build without one, or one that is not the recorded bytes or does not parse, is
    infra that is not retried."""
    if bundle is not None:
        return None
    built = built_manifest(ctx, build_manifest)
    if build_manifest is not None and built is not None and SERVED_UCOUNTS_OUTPUT not in built.outputs:
        return None
    if built is None or SERVED_UCOUNTS_OUTPUT not in built.outputs:
        raise StageFault.infra(f"agent_build published no {SERVED_UCOUNTS_OUTPUT} record of the figures its "
                               "brief stated", retryable=False)
    ref = built.outputs[SERVED_UCOUNTS_OUTPUT]
    data = ctx.store.read_bytes(ref.key)
    digest = sha256_bytes(data)
    if digest != ref.sha256:
        raise StageFault.infra(f"{ref.key} does not match its manifest: {digest} is not {ref.sha256}",
                               retryable=False)
    try:
        return ServedUcounts.model_validate_json(data)
    except ValidationError as e:
        raise StageFault.infra(f"{ref.key} is not a {SERVED_UCOUNTS_OUTPUT} record: {validation_errors(e)}",
                               retryable=False) from None


def left_nothing(ctx: StageContext, bundle: StoredBundle | None, build_manifest: str | None = None) -> str | None:
    """Why the agent_build the stage reads (``built_manifest``) left nothing to package, scrubbed,
    or None when it published a bundle, or when *bundle* names one. The record is the commit
    point's, a harness program's, so one that contradicts its build is infra that is not
    retried."""
    if bundle is not None:
        return None
    built = built_manifest(ctx, build_manifest)
    if built is None or NO_SUBMISSION_OUTPUT not in built.outputs:
        return None
    if SOURCE_BUNDLE_OUTPUT in built.outputs:
        raise StageFault.infra("agent_build published both a source bundle and no_submission", retryable=False)
    ref = built.outputs[NO_SUBMISSION_OUTPUT]
    data = ctx.store.read_bytes(ref.key)
    digest = sha256_bytes(data)
    if digest != ref.sha256:
        raise StageFault.infra(f"{ref.key} does not match its manifest: {digest} is not {ref.sha256}", retryable=False)
    try:
        record = NoSubmission.from_json(data)
    except ValueError as e:
        raise StageFault.infra(f"{ref.key} is not a no_submission record: {e}", retryable=False) from None
    return ctx.scrubber.scrub(f"the build left nothing to package: {record.reason}")


def run_checkout(ctx: StageContext, bundle: _Bundle, shape: ResourceShape, *, named: bool, max_bytes: int,
                 scan: tuple[list[str], Path] | None = None) -> tuple[SourceRecord, Path]:
    """Run the checkout role on *bundle*, of *shape*: it checks the file's size against the
    recorded one before any read, and its digest before any verdict, holds it and its
    objects inflated to *max_bytes*, the build's writable disk cap, and reads it once where it
    lies, streaming it into its clone as it hashes it (``hls.package.checkout``), its git running
    to the stage's deadline, which ends the role as infra. With *scan*,
    the names of the secrets to find, which the role gets in its env file, and where it writes
    its record, it also reads every object of the clone first. A file that is not the recorded or
    stated (*named*) bundle is refused by the role, infra that is not retried, since a retry would
    read the same file. A read that went wrong on the way is retried. Return the record and where
    the tree is."""
    tree = ctx.role_dir("checkout") / "tree"
    out = ctx.output_path("source.json")
    argv = [str(ctx.plan.toolchain.python), "-m", "hls.package.checkout", "--bundle", str(bundle.path),
            "--dest", str(tree), "--out", str(out), "--outputs-out", str(ctx.outputs_report("checkout")),
            "--size", str(bundle.size), "--expect", bundle.sha256,
            "--max-bytes", str(max_bytes)]
    names: list[str] = []
    if scan is not None:
        names, scan_out = scan
        argv += ["--names", ",".join(names), "--scan-out", str(scan_out)]
    ctx.start_role("checkout", argv, shape, owner="harness", secret_env=names)
    outcomes = ctx.wait_roles()
    refused = ctx.refusal("checkout", outcomes, (DIGEST_MISMATCH_EXIT,))
    if refused is not None:
        what = "is not the sha256 stated for it" if named else "does not match its manifest"
        raise StageFault.infra(f"the source bundle {what}: {refused.reason}", retryable=False, role="checkout")
    fault = stage_fault(outcomes)
    if fault is not None:
        raise StageFault(fault)
    return SourceRecord.model_validate_json(out.read_text()), tree


class ForeignReport(Exception):
    """The dry run's report is not what pip writes, a plain, readable file of bounded size that
    holds its installation report. Only the step's container wrote there, so it is the
    submission's doing."""


def pyproject_problem(tree: Path) -> str | None:
    """Why this process may not read *tree*'s ``pyproject.toml``, judged from ``lstat`` alone
    before anything opens it, or None: permissions in the tree that keep this process out
    (``PERMISSION_ERRNOS``), a link (never followed, within the tree too), anything but a regular
    file, or a file over ``PYPROJECT_MAX_BYTES``. A missing file is None, for the submission
    contract to report. Any other error is the node's, and raised."""
    try:
        st = os.lstat(tree / "pyproject.toml")
    except FileNotFoundError:
        return None
    except OSError as e:
        if e.errno not in PERMISSION_ERRNOS:
            raise
        return f"pyproject.toml cannot be read: {e}"
    if stat.S_ISLNK(st.st_mode):
        return "pyproject.toml is a symlink"
    if not stat.S_ISREG(st.st_mode):
        return "pyproject.toml is not a regular file"
    if st.st_size > PYPROJECT_MAX_BYTES:
        return f"pyproject.toml holds {st.st_size} bytes, more than the {PYPROJECT_MAX_BYTES} allowed"
    return None


def read_report(path: Path) -> bytes:
    """The dry run's installation report at *path*, judged by ``lstat`` before anything opens
    it, then read through one descriptor that follows no link and never waits on a pipe. Raises
    ``ForeignReport`` for what the step left that pip never writes: no report, a link, anything
    but a regular file (a directory, a pipe, a socket), a file over ``REPORT_MAX_BYTES``, or
    permissions that keep this process out (``PERMISSION_ERRNOS``). Any other error reading it is
    the node's, and raised."""
    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            raise ForeignReport("the dry run's report.json is a symlink")
        if not stat.S_ISREG(st.st_mode):
            raise ForeignReport("the dry run's report.json is not a regular file")
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            # The descriptor's own file is the one read.
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ForeignReport("the dry run's report.json is not a regular file")
            with os.fdopen(fd, "rb", closefd=False) as f:
                # Read one byte past the limit at most, whatever the file's size says.
                data = f.read(REPORT_MAX_BYTES + 1)
        finally:
            os.close(fd)
    except FileNotFoundError as e:
        raise ForeignReport("the dry run wrote no report.json") from e
    except OSError as e:
        if e.errno not in PERMISSION_ERRNOS:
            raise
        raise ForeignReport(f"the dry run's report.json cannot be read: {e}") from e
    if len(data) > REPORT_MAX_BYTES:
        raise ForeignReport(f"the dry run's report.json holds more than the {REPORT_MAX_BYTES} bytes allowed")
    return data


def report_pins(data: bytes) -> list[str]:
    """The pins of what the dry run's report *data* says the install adds
    (``missing_requirements``). Raises ``ForeignReport`` for any way the report fails to parse as
    pip's: not UTF-8, not JSON, not pip's shape, nested deeper than ``REPORT_MAX_DEPTH`` (refused
    before the parse, so the parser never recurses past it) or too big to hold."""
    try:
        return missing_requirements(load_json(data, max_depth=REPORT_MAX_DEPTH), WORKSPACE)
    except (ValueError, MemoryError) as e:
        why = type(e).__name__ + (f": {e}" if str(e) else "")
        raise ForeignReport(f"the dry run's report.json cannot be used: {why[:REPORT_REASON_CHARS]}") from e


class _Listing:
    """Names of one kind found in the wheel step's output, for a reason: how many, and the first
    ``NAMED_ENTRIES`` by name, held as they are found, so a listing of millions costs no more."""

    def __init__(self) -> None:
        #: How many names were found.
        self.count = 0
        #: The first ``NAMED_ENTRIES`` of them by name, sorted.
        self.first: list[str] = []

    def add(self, name: str) -> None:
        self.count += 1
        bisect.insort(self.first, name)
        del self.first[NAMED_ENTRIES:]

    def reason(self, what: str) -> str | None:
        """That the wheel step left *what*, with the count and the first names, each cut at
        ``NAME_CHARS``; None when none was found."""
        if not self.count:
            return None
        names = [name[:NAME_CHARS] for name in self.first]
        listed = (f"{self.count} in all: {names}" if self.count <= len(names)
                  else f"{self.count} in all; the first {len(names)} by name: {names}")
        return f"the wheel step left {what} ({listed})"


def wheels_problem(wheels: Path, project: str, pins: list[str]) -> str | None:
    """Why the wheel step's output *wheels* may not go into the image, or None: entries that are
    not plain wheel files, wheels that are neither the project *project*'s nor one of *pins*', by
    the distribution's normalized name in the file's name, more than one wheel of a distribution,
    no wheel of the project, or permissions the step left that keep this process from listing
    them (``PERMISSION_ERRNOS``). The offline build copies the directory into the image, so a link
    there could carry a host file into it, and installs from it, where anything but a wheel would
    have to be built: the build hooks may write any file there. And pip installs only what the
    project needs, while the wheel check and the image hold every wheel there, so a build hook
    could leave wheels of any number of entries that only the harness reads. Each reason gives a
    count and the first names (``_Listing``). Any other error listing it is the node's, and
    raised."""
    allowed = {pep503_normalize(project), *(pep503_normalize(pin_name(pin)) for pin in pins)}
    odd, stray, twice = _Listing(), _Listing(), _Listing()
    # Each distribution's wheel; None once a second one was found.
    seen: dict[str, str | None] = {}
    try:
        with os.scandir(wheels) as entries:
            for e in entries:
                if e.is_symlink() or not e.is_file(follow_symlinks=False) or not e.name.endswith(".whl"):
                    odd.add(e.name)
                    continue
                parsed = _WHEEL_FILE.fullmatch(e.name)
                dist = None if parsed is None else pep503_normalize(parsed.group(1))
                if dist not in allowed:
                    stray.add(e.name)
                elif dist not in seen:
                    seen[dist] = e.name
                else:
                    first, seen[dist] = seen[dist], None
                    if first is not None:
                        twice.add(first)
                    twice.add(e.name)
    except OSError as e:
        if e.errno not in PERMISSION_ERRNOS:
            raise
        return f"the wheel step's output cannot be read: {e}"
    return (odd.reason("entries that are not plain wheel files")
            or stray.reason("wheels that are neither the project's nor a pinned dependency's")
            or twice.reason("more than one wheel of a distribution")
            or (None if pep503_normalize(project) in seen else f"the wheel step left no wheel of the project {project}"))


class _Package:
    """One package attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None or ctx.runtime is None or ctx.cache is None:
            raise RuntimeError("the package stage needs node-local scratch, a container runtime and the cache")
        self.ctx = ctx
        self.params: PackageParams = ctx.params
        self.policy = NetworkPolicy.model_validate(self.params.network)
        #: The package names the producer declares denied, which the pip steps' seal refuses.
        self.deny_list = deny_list(self.policy)
        #: Each harness role's shape: the stage's, less the runner's memory, and no disk
        #: (``role_shape``); the roles run one at a time. The stage's declared disk, checked when
        #: the job starts, is the one budget for the trees, the copies, the wheels and the build.
        self.shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
        #: Each pip step's shape: a harness role's, with the disk its container may write and the
        #: memory it may hold that the plan states (``params.pip``), which the brief stated and the
        #: stage's declared shape gives (``check``). The step before it left only its outputs on
        #: the node, so the budget leaves that much free when it starts.
        self.step_shape = self.shape.model_copy(update={"disk_gib": self.params.pip.disk_gib,
                                                        "memory_gib": self.params.pip.memory_gib})
        #: The time the two pip steps may still run together.
        self.pip_budget = PipBudget(PIP_STEPS_SEC)
        #: The process ceiling the pip steps run under, the stage's (``StageContext.process_ceiling``),
        #: once the first one starts; ``package.json`` records it.
        self.ceiling: ProcessCeiling | None = None
        #: Output name to store key, for what this attempt wrote.
        self.outputs: dict[str, str] = {}
        #: Digest of the bundle read; None before one is read, when it was refused for its size
        #: before it was read, and when there is none.
        self.bundle_sha256: str | None = None
        #: The commit built.
        self.commit: str | None = None
        #: The figures the build's brief stated for its graded server, which ``package.json``
        #: carries (``served_record``): read before any role or result; None for a named bundle and
        #: a pinned rollout's build that published none.
        self.served: ServedUcounts | None = None

    # inputs

    def _manifest(self, stage: str) -> Manifest | None:
        """This instance's manifest of *stage* among the inputs; None when there is none."""
        return instance_manifest(self.ctx, stage)

    def base_entry(self) -> tuple[str, str]:
        """The submission base's cache key and entry digest, from this instance's
        ``images.json``, which must be the file its manifest records."""
        p = self.params
        images = self._manifest(p.images_stage)
        if images is None or IMAGES_OUTPUT not in images.outputs:
            raise StageFault.infra(f"no {IMAGES_OUTPUT} output of this instance's {p.images_stage} stage names the "
                                   "submission base", retryable=False)
        ref = images.outputs[IMAGES_OUTPUT]
        data = self.ctx.store.read_bytes(ref.key)
        if sha256_bytes(data) != ref.sha256:
            raise StageFault.infra(f"{ref.key} does not match its manifest", retryable=False)
        try:
            entry = json.loads(data)[p.submission_base]
            key, expect = check_segment(entry["key"]), entry["entry_sha256"]
        except (ValueError, KeyError, TypeError) as e:
            raise StageFault.infra(f"{ref.key} names no image {p.submission_base} with a key and an entry digest",
                                   retryable=False) from e
        return key, expect

    # roles

    def _harness_role(self, name: str, argv: list[str]) -> None:
        """Run host role *name*; a harness role that ended any way but exiting 0 is a fault of
        the roles (``stage_fault``), raised here."""
        self.ctx.start_role(name, argv, self.shape, owner="harness")
        fault = stage_fault(self.ctx.wait_roles())
        if fault is not None:
            raise StageFault(fault)

    def checkout(self, bundle: _Bundle) -> tuple[SourceRecord, Path]:
        """Run the checkout role on *bundle* (``run_checkout``), with no scan: the source_scan
        stage read this bundle before this stage ran any of the contestant's code."""
        record, tree = run_checkout(self.ctx, bundle, self.shape, named=self.params.bundle is not None,
                                    max_bytes=self.params.disk_cap_gib * 2**30)
        self.outputs["source"] = self.ctx.reported_output("source.json", "checkout")
        self.bundle_sha256 = record.bundle_sha256
        return record, tree

    def contestant_step(self, name: str, command: list[str], mounts: tuple[Mount, ...], base: PinnedImage, *,
                        failed: str) -> tuple[RoleOutcome, str | None]:
        """Run pip step *name* in a sealed contestant container on the submission base, stopped
        once its command has run for what is left of the pip steps' budget (``PipBudget``),
        counted from when the stage saw the command start, so the runtime's start of the
        container costs the step none of it; and say why the package is invalid when the step's
        failure is the submission's, else None: *failed*, what its failing exit means, with the
        pages its index refused for their size (``oversized_reason``), or that it reached its
        limit. A role fault is raised as the roles decided it (``stage_fault``); any other
        failure that is not the submission's raises an infra fault of the stage's."""
        ctx = self.ctx
        workdir = ctx.role_dir(name) / "wd"
        workdir.mkdir(exist_ok=True)
        seal = SealPlan(policy=SealPolicy(network=self.policy), index=True, deny_list=self.deny_list)
        spec = ContainerSpec(image=base, command=tuple(command), workdir=workdir, mounts=mounts, cwd=WORKSPACE,
                             seal=seal, contestant_payload=True)
        env = container_env(gpu_kind=self.params.gpu_kind, cpus=self.shape.cpus)
        # The submission's build backends are its code: the step runs under the stage's process
        # ceiling, which the context gives every contestant container.
        self.ceiling = ctx.process_ceiling()
        ctx.start_container(name, spec, self.step_shape, owner="contestant", env=env)
        left = self.pip_budget.left
        [outcome] = ctx.wait_roles(command_sec=left)
        ended = ctx.clock()
        began = ctx.command_started_at(name)
        payload_end = step_ended_at(ctx.seal_files(name), ended)
        if began is not None:
            self.pip_budget.spend(began, payload_end)
        fault = stage_fault([outcome])
        if fault is not None:
            raise StageFault(fault)
        # The seal's files were cleared at launch, so the count is this step's alone.
        after = ctx.seal_status(name)
        whole_limit = began is not None and began + left <= ctx.deadline
        verdict = contestant_verdict(outcome, upstream=after, whole_limit=whole_limit, ended_at=payload_end)
        timed_out = f"package step {name} did not end within the {PIP_STEPS_SEC} s both pip steps may take together"
        if verdict == "ok":
            return outcome, None
        # A failing end the uid's other tasks starved, or that they could not be counted at, the
        # step cut at its time limit among them, is infra before any verdict could read it as the
        # submission's.
        starved = scored_end_fault(outcome)
        if starved is not None:
            raise StageFault(starved)
        if verdict == "invalid":
            if outcome.ending == "time_limit":
                return outcome, f"{timed_out} while its package index counted no upstream failure" + oversized_reason(
                    after)
            return outcome, failed + oversized_reason(after)
        if outcome.ending == "time_limit" and not whole_limit:
            raise StageFault.infra(f"package step {name} was cut by the stage's deadline before the pip steps' limit",
                                   retryable=True, ending=outcome.ending)
        if outcome.ending not in ("exited", "time_limit"):
            raise StageFault.infra(f"package step {name} ended {outcome.ending.replace('_', ' ')}", retryable=True,
                                   ending=outcome.ending)
        how = timed_out if outcome.ending == "time_limit" else f"package step {name} failed with status {outcome.exit_code}"
        if after is None:
            raise StageFault.infra(f"{how} and its package index wrote no count, so the failure is not certainly the "
                                   "submission's", retryable=True, ending=outcome.ending)
        if interceptor_stale(after, payload_end):
            raise StageFault.infra(f"{how} while its package index's interceptor had not reported for "
                                   f"{payload_end - after.heartbeat:.0f} s, so the failure is not certainly the "
                                   "submission's", retryable=True, ending=outcome.ending)
        raise StageFault.infra(f"{how} while the package index failed upstream {after.total} times, {after.mid_stream} "
                               "of them mid-stream", retryable=True, ending=outcome.ending,
                               upstream_errors=str(after.total), mid_stream_errors=str(after.mid_stream))

    def _step_dirs(self, name: str, tree: Path) -> tuple[Path, Path]:
        """A fresh copy of the checked-out *tree* and an empty temporary directory, in role
        *name*'s own directory, for its container to mount writable. A harness role copies the
        tree (``cp -a``), so the runner reads none of it; the checkout held the tree to its
        limits, so a copy that fails failed for the harness."""
        root = self.ctx.role_dir(name)
        workspace, tmp = root / "workspace", root / "build-tmp"
        self._harness_role(f"{name}-copy", ["cp", "-a", "--", str(tree), str(workspace)])
        tmp.mkdir()
        return workspace, tmp

    def _clean(self, name: str, workspace: Path, tmp: Path, *, keep: tuple[Path, ...] = ()) -> None:
        """Remove pip step *name*'s copy of the tree and its temporary directory once it has
        ended, by a harness role, so the next step's writes never meet them on the node. The
        step's container wrote there as it liked, so the role first gives the owner back every
        permission (``CLEAN_SHELL``), also on what the step wrote that the stage keeps, *keep*,
        before anything reads it."""
        self._harness_role(f"{name}-clean", ["/bin/sh", "-c", CLEAN_SHELL, "hls-clean", str(len(keep)),
                                             *map(str, keep), str(workspace), str(tmp)])

    # results

    def _log_tail(self, name: str) -> str:
        logs = self.ctx.attempt_dir / "logs"
        text = _tail(logs / f"{name}.err", LOG_TAIL_BYTES) + _tail(logs / f"{name}.out", LOG_TAIL_BYTES)
        return self.ctx.scrubber.scrub(text)

    def scan_install(self, base: tuple[PinnedImage, Path], built: tuple[PinnedImage, Path]) -> str:
        """Scan every entry the install added to the image *built* over *base*, each with the image
        file it was pinned from, for every form of each secret of the secrets file
        (``hls.package.scan``, each image as ``ImageRuntime.reader_args`` hands it) and return the
        key of ``package-scan.json``. On a site whose secrets file holds no secret nothing is read
        for one: the stage writes a record that read nothing and starts no role. A failure of the
        role is a fault of the roles, raised here."""
        ctx = self.ctx
        names = sorted(ctx.secrets)
        out = ctx.output_path(PACKAGE_SCAN_FILE)
        if not names:
            ctx.events.emit("package_scan_skipped", reason="the secrets file holds no secret")
            record = PackageScan(secrets=[], forms=list(FORMS), findings=[], read=[])
            return ctx.write_output(PACKAGE_SCAN_FILE, record.model_dump_json(indent=2).encode())
        assert ctx.images is not None
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.package.scan", "--names", ",".join(names),
                *ctx.images.reader_args("base", *base, listing=self.base_listing()),
                *ctx.images.reader_args("image", *built), "--out", str(out),
                "--outputs-out", str(ctx.outputs_report(SCAN_ROLE))]
        ctx.start_role(SCAN_ROLE, argv, self.shape, owner="harness", secret_env=names)
        fault = stage_fault(ctx.wait_roles())
        if fault is not None:
            raise StageFault(fault)
        return ctx.reported_output(PACKAGE_SCAN_FILE, SCAN_ROLE)

    def _publish(self, result: PackageResult) -> dict[str, str]:
        result = result.model_copy(update={"process_ceiling": self.ceiling})
        self.ctx.events.emit("package_result", status=result.status, reason=result.reason, commit=result.commit)
        self.outputs["package"] = self.ctx.write_output(PACKAGE_FILE, result.model_dump_json(indent=2).encode())
        return self.outputs

    def invalid(self, reason: str, outcome: RoleOutcome | None = None) -> dict[str, str]:
        return self._publish(PackageResult(
            task=self.params.task, status="invalid_artifact", bundle_sha256=self.bundle_sha256, commit=self.commit,
            served_ucounts=self.served, reason=reason, returncode=None if outcome is None else outcome.exit_code,
            log_tail=None if outcome is None else self._log_tail(outcome.name)))

    def tree_problem(self, tree: Path) -> str | None:
        """Why *tree* cannot be built, or None. An error reading pyproject.toml that the tree's
        permissions do not explain is the node's: a retryable fault, raised here."""
        try:
            problem = pyproject_problem(tree)
            if problem is not None:
                return problem
            try:
                SubmissionContract(server_filename=self.params.server_filename).validate(tree)
                denied = prescan(tree, self.policy)
            except SubmissionError as e:
                return str(e)
            if denied:
                return f"pyproject declares packages not permitted by policy: {denied}"
            if project_name(tree) is None:
                return ("pyproject.toml declares no valid [project].name; the harness installs the project "
                        "by its name from the wheels it built")
        except OSError as e:
            raise _node_fault("the tree's pyproject.toml", e) from e
        return None

    # the stage

    def run(self) -> dict[str, str]:
        ctx = self.ctx
        if not ctx.secret_registry:
            raise StageFault.infra("package runs only with the secrets registry, which the runner did not hand it: it "
                                   "could not scan what the submission's compressed files hide from finalize",
                                   retryable=False)
        nothing = left_nothing(ctx, self.params.bundle, self.params.build_manifest)
        bundle = (source_bundle(ctx, self.params.bundle, self.params.bundle_sha256, self.params.build_manifest)
                  if nothing is None else None)
        self.served = served_record(ctx, self.params.bundle, self.params.build_manifest)
        if nothing is not None:
            return self.invalid(nothing)
        assert bundle is not None
        key, expect = self.base_entry()
        # Pinned before any other role starts; the build reads the same entry's file itself.
        base = ctx.node_image(key, expect)
        assert ctx.images is not None
        base_file = ctx.entry_file("images", key, expect, ctx.images.image_file)
        source, tree = self.checkout(bundle)
        self.commit = source.commit
        if source.status == "no_main":
            return self.invalid("the bundle has no commit on main; only the latest commit on main is evaluated")
        if source.status == "over_limit":
            return self.invalid(f"the submission is past the package's limits: {source.reason}")
        problem = self.tree_problem(tree)
        if problem is not None:
            return self.invalid(problem)
        project = project_name(tree)
        assert project is not None  # tree_problem refused a tree without one
        workspace, tmp = self._step_dirs("dry-run", tree)
        mounts = (Mount(workspace, WORKSPACE, read_only=False), Mount(tmp, BUILD_TMP, read_only=False))
        outcome, problem = self.contestant_step(
            "dry-run", dry_run_command(WORKSPACE, f"{BUILD_TMP}/report.json"), mounts, base,
            failed="the submission's dependencies did not resolve against the submission base")
        if problem is not None:
            return self.invalid(problem, outcome)
        try:
            data = read_report(tmp / "report.json")
            pins = report_pins(data)
        except ForeignReport as e:
            return self.invalid(str(e), outcome)
        except OSError as e:
            raise _node_fault("the dry run's report.json", e) from e
        # The report's own bytes: writing it out again could fail where reading it did not.
        self.outputs["install_report"] = ctx.write_output("install-report.json", data)
        self._clean("dry-run", workspace, tmp)
        workspace, tmp = self._step_dirs("wheel", tree)
        wheels = ctx.role_dir("wheel") / "wheels"
        wheels.mkdir()
        (tmp / "pins.txt").write_text("".join(f"{pin}\n" for pin in pins))
        mounts = (Mount(workspace, WORKSPACE, read_only=False), Mount(tmp, BUILD_TMP, read_only=False),
                  Mount(wheels, WHEELS_OUT, read_only=False))
        outcome, problem = self.contestant_step(
            "wheel", wheel_command(WORKSPACE, f"{BUILD_TMP}/pins.txt", WHEELS_OUT), mounts, base,
            failed="the submission's dependencies did not build into wheels")
        if problem is not None:
            return self.invalid(problem, outcome)
        self._clean("wheel", workspace, tmp, keep=(wheels,))
        try:
            problem = wheels_problem(wheels, project, pins)
        except OSError as e:
            raise _node_fault("the wheel step's output", e) from e
        if problem is None:
            problem = self.wheel_check(wheels)
        if problem is not None:
            return self.invalid(problem)
        return self.offline(base, base_file, tree, wheels, project, pins)

    def wheel_check(self, wheels: Path) -> str | None:
        """Read every wheel in *wheels* whole, by the ``wheel-check`` role, before the build
        installs them (``hls.package.wheel_check``). The reason the submission is invalid, or
        None; the node's error reading a wheel is a retryable fault, raised here."""
        ctx = self.ctx
        out = ctx.role_dir("wheel-check") / "check.json"
        self._harness_role("wheel-check", [str(ctx.plan.toolchain.python), "-m", "hls.package.wheel_check",
                                           "--wheels", str(wheels), "--out", str(out)])
        record = WheelCheck.model_validate_json(out.read_text())
        if record.node_error is not None:
            raise StageFault.infra(f"the wheel check could not read the wheels: {record.node_error}", retryable=True,
                                   errno=str(record.node_errno))
        return record.problem

    def offline(self, base: PinnedImage, base_file: tuple[Path, str], tree: Path, wheels: Path, project: str,
                pins: list[str]) -> dict[str, str]:
        """Build the submission image from the base's image file *base_file* (its path in the
        store and the sha256 its entry records), the checkout's *tree* and the *wheels*,
        installing the project *project*, then check it and publish it."""
        ctx = self.ctx
        assert ctx.images is not None
        build_dir = ctx.role_dir("build")
        name = submission_file(ctx.images.image_file)
        image, digest = build_dir / name, build_dir / "sha256"
        # Drawn now, after every step that ran contestant code has ended: no text pip echoes
        # from the submission's wheels can carry it (``hls.package.offline``).
        nonce = new_nonce()
        argv, env = ctx.images.submission_role(image, digest=digest, base_file=base_file, base=base, tree=tree,
                                               wheels=wheels, project=project, server=self.params.server_filename,
                                               nonce=nonce, constraints=self.params.constraints)
        ctx.start_role("build", argv, self.shape, owner="harness", env=env, daemon=True)
        outcomes = ctx.wait_roles()
        [outcome] = outcomes
        refused = ctx.refusal("build", outcomes, (DIGEST_MISMATCH_EXIT,))
        if refused is not None:
            # The entry's file is not the one it records, and a retry would read it again.
            raise StageFault.infra(refused.reason, retryable=False, role="build")
        logs = ctx.attempt_dir / "logs"
        log = _tail(logs / "build.out", BUILD_LOG_BYTES) + "\n" + _tail(logs / "build.err", BUILD_LOG_BYTES)
        verdict = build_verdict(outcome, log, nonce)
        added = installed_bytes(log, nonce)
        if verdict == "invalid" and install_failed(log, nonce):
            return self.invalid("the submission's wheels did not install into the submission base", outcome)
        if verdict == "invalid":
            return self.invalid(f"the submission's install adds {added} bytes to /opt/venv, more than the "
                                f"{INSTALL_MAX_GIB} GiB allowed", outcome)
        fault = stage_fault([outcome])
        if fault is not None:
            raise StageFault(fault)
        sha256 = reported_sha256(digest)
        if verdict == "infra" or not image.is_file() or sha256 is None:
            raise StageFault.infra(f"the offline build failed outside the install with status {outcome.exit_code}",
                                   retryable=True)
        if added is None:
            raise StageFault.infra("the offline build reported no size of what its install added", retryable=True)
        built = ctx.pin_file(image, sha256=sha256, contestant=True)
        problem = self.integrity((base, base_file[0]), (built, image), wheels)
        if problem is not None:
            return self.invalid(problem)
        self.outputs["package_scan"] = self.scan_install((base, base_file[0]), (built, image))
        out = ctx.output_path(name)
        report = ctx.outputs_report("publish")
        self._harness_role("publish", [str(ctx.plan.toolchain.python), "-m", "hls.package.publish", "--image",
                                       str(image), "--out", str(out), "--outputs-out", str(report)])
        self.outputs[SUBMISSION_OUTPUT] = ctx.reported_output(name, "publish")
        copied = (read_outputs_report(report) or {})[name].sha256
        if copied != built.file_sha256:
            raise StageFault.infra(f"the copy of the submission image at {out} has sha256 {copied}, not the "
                                   f"{built.file_sha256} it was pinned by", retryable=True)
        ctx.record_image("submission", built.file_sha256)
        return self._publish(PackageResult(
            task=self.params.task, status="built", bundle_sha256=self.bundle_sha256, commit=self.commit,
            image=self.outputs[SUBMISSION_OUTPUT], sif_sha256=built.file_sha256, wheels=pins,
            served_ucounts=self.served))

    def base_listing(self) -> Path:
        """Where the integrity check leaves the base's tree for the install's scan, on a runtime
        whose readers read an image from its archive (``ImageRuntime.reader_args``), so the base is
        read once: in the integrity role's own directory, which the stage removes."""
        return self.ctx.role_dir("integrity") / "base-listing.json"

    def integrity(self, base: tuple[PinnedImage, Path], built: tuple[PinnedImage, Path], wheels: Path) -> str | None:
        """Check *built*'s system paths against *base*'s, each with the image file it was pinned
        from (``hls.package.integrity``, each image as ``ImageRuntime.reader_args`` hands it). None when
        they are the base's; the reason the package is invalid when installing the submission's
        wheels accounts for every difference; an infra fault, not retryable, for any other."""
        ctx = self.ctx
        out = ctx.output_path(INTEGRITY_FILE)
        assert ctx.images is not None
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.package.integrity",
                *ctx.images.reader_args("base", *base, listing_out=self.base_listing()),
                *ctx.images.reader_args("image", *built), "--wheels", str(wheels), "--out", str(out), "--outputs-out",
                str(ctx.outputs_report("integrity"))]
        ctx.start_role("integrity", argv, self.shape, owner="harness")
        [outcome] = ctx.wait_roles()
        if outcome.ending != "exited" or outcome.exit_code not in (0, DIFFERS_EXIT):
            fault = stage_fault([outcome])
            if fault is not None:
                raise StageFault(fault)
            raise StageFault.infra(f"the integrity check ended {outcome.ending} with status {outcome.exit_code}",
                                   retryable=True)
        try:
            record = IntegrityRecord.model_validate_json(out.read_text())
        except OSError as e:
            raise StageFault.infra(f"the integrity check's record cannot be read: {e}", retryable=True) from e
        except ValidationError as e:
            raise StageFault.infra(f"the integrity check's record cannot be read: {validation_errors(e)}",
                                   retryable=True) from None
        self.outputs["integrity"] = ctx.reported_output(INTEGRITY_FILE, "integrity")
        verdict = integrity_verdict(record)
        if verdict == "ok" and outcome.exit_code == 0:
            return None
        named = ", ".join(d.path for d in record.listed[:NAMED_PATHS])
        if verdict == "invalid":
            wheels_named = sorted({w for d in record.listed for w in d.wheels})
            return (f"the submission's wheels {wheels_named} change {record.differences} system paths of the image, "
                    f"which must be the submission base's: {named}")
        raise StageFault.infra("the submission image's system paths differ from the submission base's at "
                               f"{record.differences} paths, {record.unattributed} of them not accounted for by "
                               f"installing the submission's wheels: {named}", retryable=False,
                               differences=str(record.differences), unattributed=str(record.unattributed))


@register
class PackageStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "package"
    params_model: ClassVar[type[BaseModel]] = PackageParams
    #: The submission image holds the contestant's code and what it installed, read as stored by
    #: finalize's secret scan (``Stage.raw_outputs``).
    raw_outputs: ClassVar[tuple[str, ...]] = SUBMISSION_FILES
    #: The pip steps run the contestant's build, so every file of an attempt holds the
    #: contestant's bytes, the records and logs that name what it wrote and installed included
    #: (``Stage.runs_contestant_code``).
    runs_contestant_code: ClassVar[bool] = True
    #: The scan of what the install added searches for every value of the secrets file.
    needs_secret_registry: ClassVar[bool] = True
    #: The submission image is squashfs, compressed: finalize's scan as stored cannot see inside it,
    #: so finalize keeps it out of any export unless this stage's and source_scan's scans read it
    #: (``Stage.compressed_files``).
    compressed_files: ClassVar[tuple[str, ...]] = tuple(f"outputs/{name}" for name in SUBMISSION_FILES)
    #: Nothing else of its attempts is such a container (``Stage.unpublished_containers``,
    #: ``Stage.published_dirs``).
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the checkout, the base's extraction, the dry run and the
        wheel step each with its copy and its clean-up, wheel-check, the build, its extraction,
        integrity, the install's scan and publish."""
        return 14

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        if shape.gpus:
            raise ValueError("package runs on the host side and in CPU containers and asks for no GPU")
        assert isinstance(params, PackageParams)
        disk = min_disk_gib(params.disk_cap_gib)
        if shape.disk_gib < disk:
            raise ValueError(f"package needs at least {disk} GiB of node-local disk, its checkout's at the build's "
                             f"{params.disk_cap_gib} GiB cap or its build's; the stage declares {shape.disk_gib}")
        if shape.time_sec < MIN_TIME_SEC:
            # A pip step the stage's deadline cuts before the steps' limit is never the submission's.
            raise ValueError(f"package needs at least {MIN_TIME_SEC} s of time, the {PIP_STEPS_SEC} s the pip steps "
                             f"may take together, the checkout's {CHECKOUT_SEC} s and the rest of the package's "
                             f"{REST_SEC} s; the stage declares {shape.time_sec}")
        # One role at a time, of the stage's shape less the runner's memory, as the checkout needs.
        role = role_shape(shape, runner_memory_gib)
        if role.memory_gib < CHECKOUT_MEMORY_GIB:
            raise ValueError(f"package's checkout needs {CHECKOUT_MEMORY_GIB} GiB of memory; the stage's "
                             f"{shape.memory_gib} GiB less the runner's {runner_memory_gib} gives its role "
                             f"{role.memory_gib}")
        # The brief stated the pip steps' limits; a shape the site changed may give more, never less.
        given = pip_limits(shape, runner_memory_gib)
        if given.disk_gib < params.pip.disk_gib or given.memory_gib < params.pip.memory_gib:
            raise ValueError(f"the plan states {params.pip.disk_gib} GiB of disk and {params.pip.memory_gib} GiB of "
                             f"memory for each pip step, and the stage's shape gives {given.disk_gib} and "
                             f"{given.memory_gib}")
        # The pip steps' seal has an index, so the producer must declare a deny list for it.
        deny_list(NetworkPolicy.model_validate(params.network))

    @classmethod
    def seals_containers(cls, params: BaseModel | None) -> bool:
        return True

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Package(ctx).run()
