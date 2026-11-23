"""The finalize stage: one eval's records in the shapes the viewer and ``hls grade`` read, a
scan of every file of the instance for a secret's value, and the instance's index row.

The stage runs in the runner and reads only small files. In order:
1. It reads this instance's score manifest and, through the score stage's parameters, its
   drive and verify manifests, and checks ``drive.json``, ``verdict.json`` and ``score.json``
   against the digests they recorded.
2. It writes the records, built from those files and the manifests:
   - ``grading.json``, a ``GradingRecord`` around the eval's ``EvalEpisode``. ``checkpoint``
     is the digest of the artifact image the drive served ("" for an invalid artifact, which
     never ran); ``started_at`` is the drive stage's start and ``finished_at`` the score
     stage's end; ``trace_path`` is the store key of the trace's directory, which holds
     its ``END.json``, chunks and summary ("" when the drive did not serve);
     ``verdict_summary`` holds the score's ``gate_passed``, the verdict's ``aggregate_passed``
     and the verifiers' ``diagnostics``, and ``score`` the serialized Score;
     ``container_logs`` maps the file name of each role log of the drive, verify and score
     attempts to its key. The provenance is this instance's ``RolloutProvenance``: the
     task as the roles rebuilt it, calibration and baseline credit included (``spec``), every
     image digest the instance's manifests record, the commits, the overrides and the drive's
     artifact;
   - for a rollout, ``rollout.json`` (``RolloutRecord``). Its status is ``invalid_artifact``
     for a submission that did not build and ``ok`` for every scored eval, a zero included. A
     rollout whose agent built its submission (``FinalizeParams.build_stage``) records the build
     (``RolloutBuild``): its ``build.json`` as written, the keys of its outputs, the package
     stage's ``package.json`` and the agent's usage, each read against its manifest;
   - on both records, the harness-conditions table (``hls.obs.conditions``): every stage's
     attempts and the one that counted, every allocation and what it retried, the commits, the
     credit the score role and the calibration file the verify role recorded, the images, where
     the scored server ran, and the arm and its budgets.
   - for a grading that packaged the rollout's source again (``FinalizeParams.package_stage``,
     ``--artifact.rebuild``), ``grading.json`` holds the rebuilt package's ``package.json``.
   A build-only rollout (``score_stage`` None, ``hls run --build-only``) grades nothing: it
   writes ``rollout.json`` alone, with status ``built``, or ``invalid_artifact`` when its
   package did not build, and its index row holds no score.
   Each record is written with ``model_dump_json(indent=2)``, which writes a non-finite number
   as null just as ``json_safe`` does, and with every value of the site's secrets file replaced
   by its name (``ctx.scrubber``).
3. It runs the secret scan role, ``hls.substrate.secret_scan``, on the host, over every file
   under the instance present when it runs (``scan_lists``): plan revisions, statuses and
   events, every attempt's outputs and logs, failed attempts' too, the runner's job logs and
   the records it just wrote, and each published output by name, so one that is gone fails
   the scan. Logs, job logs and the outputs a stage's kind declares raw
   (``Stage.raw_outputs``) are read as stored; the rest as stored and, when it is gzip,
   decompressed too (see ``hls.substrate.secret_scan``). A gzip stream cut short, such as a replies
   file a killed attempt left, is searched as far as it goes, and recorded
   (``scan_cut_streams``). A link is never followed and a special file never opened: each is
   recorded as not read (``scan_unread_entries``), and an export leaves it behind as no regular
   file. The values reach the role only in its env file. The role searches
   every form of each secret (``hls.substrate.secret_forms``): its value, and the host of a value
   that is an http(s) URL when it identifies a machine, in any case, in each file's bytes and in
   its path. It also reads every ``package-scan.json`` (``PackageScan``) this instance
   published or its drive pinned: the source_scan stage's, of the source bundle's objects and the
   checked-out tree, and the package stage's, of what the install added to the image, each read
   while it was plain. The submission image is scanned here as stored, so what its compression
   hides is not seen here; those scans are what see inside it. An agent build's source bundle,
   the partial one a commit point killed with its stage leaves and its snapshots, each as large as
   the build's writable disk cap, are not read here at all (``Stage.unread_files``), only their
   paths searched: the bundle's objects are what the source scan's record read, and a bundle no
   record read is quarantined whole; a partial bundle and a snapshot, left only when the commit
   point bundled nothing, no scan reads inside, so each is always quarantined whole and never
   exported, listed by its path as a quarantined file is. A record
   lists a secret's files up to a bound and names the rest only by area and count, and the
   bundle's objects only by
   count: each such area stands
   found in whole (``area_holders``), as the area in the image (``<record>::/workspace/`` for the
   tree, ``<record>::/opt/venv/`` for what the install added), the source bundle the scan read,
   and the images built from it, so a file the record does not name is never exported; the
   objects of a bundle outside the store, as ``hls package --bundle`` names one, by the record
   (``OUTSIDE_BUNDLE``). And a container no published scan read inside, a source bundle, a
   snapshot or a submission image an attempt left, or a file an agent build left in its outputs
   unpublished (``Stage.compressed_files``, ``Stage.unpublished_containers``;
   ``unscanned_containers``), is quarantined with no secret named, since nothing says it holds
   none. A record reads a container only when it searched every secret the site's file holds
   today, in every form (``PackageScan.forms``). Each form found is judged by the shared rule
   (``judge``), which an export of the instance applies too, and no fault or record names a value
   or a host:
   - a value is judged by its secret first, then by its file. A secret is judged by its value,
     since the bytes found say nothing of the name they came under: two names that hold one
     value, as one URL serving two routes, are one secret here.
     - a secret no contestant could hold, one that the plan grants to no stage that hands its
       grants to its contestant and that is no route secret of the arm whose agent built the
       graded package, nor holds the value of one (``contestant_facing``,
       ``FinalizeParams.built_arm``: the agent held them, and its server may echo them; the plan
       and the arm say so whether or not the site's file still holds them), is the harness's leak
       wherever it is found, and the instance is refused as infra that is not retried. Nothing is
       published. A grading of a package whose inputs reach an agent build must name that build's
       arm, as its ``build.json`` records it (``check_built_arm``);
     - a secret a contestant could hold, found only in files a contestant's bytes reach (any file
       of an attempt of a stage whose kind runs a contestant's code,
       ``Stage.runs_contestant_code``, such as the package's and the drive's, whose harness logs
       and records echo the names and bytes the contestant chose; in another stage, a raw output,
       an output its kind declares a contestant's, ``Stage.contestant_outputs``, such as the
       verdict and the score, or the stdout or stderr of a harness role its kind says echoes
       them; a file the package's scan named): the files are quarantined, named in
       ``quarantine.json`` and the index row so that no export carries them, and the instance is
       published and scored. Found in a file
       the harness alone wrote, it is the harness's leak and refuses;
   - a host, which grants no access and may be public, found in a file a contestant's bytes reach
     (as above), quarantines the file, and never refuses or changes the score; found in a file
     the harness alone wrote, which the harness scrubs of every form, it is the harness's leak
     and refuses.
   Any other failure of the role is retried.
4. Last, it writes the instance's index row (``index_row``): the fields of an
   ``index.jsonl`` line but ``backend``, which a SIF digest leaves always null; ``scanned``
   (true), ``quarantined`` (the first ``SHOWN_QUARANTINED`` files quarantined) and
   ``quarantined_count``; and ``arm``, the arm that built the graded submission, with its model
   as ``agent_implementer_model``, as its own create-only file:
   ``runs/<task>/index/<rollout>.json``, or ``<rollout>+<grading>.json`` for a grading; a
   rollout id never holds ``+``. An evaluation (``hls grade --package``, ``--measured`` or
   ``--artifact.image``), a grading under a cache warmer, goes to the warmers' own index,
   ``warmers/<task>/index/<warmer>+<grading>.json``, so that no evaluation is listed among the
   task's runs. No two instances append to one file.
   The row is not one of the stage's outputs,
   which lie in its attempt's own directory as the runner requires; its key follows from the
   instance (``index_key``). A row that already exists is refused, as infra that is not
   retried, and left as it is. Publish-once keeps a finalized instance from reaching that:
   the runner never runs a stage whose manifest is published, and an attempt writes its row
   only after everything else succeeded, right before the runner publishes. A row is found
   only when an attempt was killed, or its allocation stopped, in that last moment, or when
   another instance's ids name the same file; either way a second row would hide which
   record the row belongs to.

The stage sets ``needs_secret_registry``: the runner loads the site's secrets file for it
although it has no grants, and hands it every value in ``ctx.secrets`` under its secret's
name, since the scrubber and the scan need them all. Without the file the allocation stops
before the stage starts, as infra that is not retried, and a stage the runner did not hand the
registry refuses to run. Its time is a fixed allowance (``FINALIZE_SEC``), and the scan runs to
the stage's deadline, which the scan's wait for its step has grown (``run_scan``): a scan still
running then ends the stage as infra.

A failed instance, one that a stage's contestant fault ended (Outcome ``failed``: the
contestant's scored result, never retried), runs no later stage, so this stage never runs for
it. Its records come from the orchestrator instead, right before it writes the plan revision
that ends the instance (``failed_records``): the zero ``score.json`` whose diagnostics name the
stage and the fault, for an instance that grades the ``grading.json`` around it, and for a
rollout ``rollout.json`` (with its build when its agent_build published), each under the
instance's ``failed/`` directory with the harness-conditions table, and the index row in its
usual place with the failed stage and the fault's reason in the two columns this stage leaves
null, and ``scanned`` false: no scan read the instance's files, so an export scans them first.
The provenance holds the task's spec, loaded from the plan's checkout as this stage loads it;
when the task does not load, the spec is empty and the zero's diagnostics name the error's type
(``spec_error``), and the zero stands. They are built from the plan, the attempt statuses and the
published manifests, all small, and scrubbed with every value of the site's secrets file, which
must be usable as it must for this stage. Each is create-only, so an orchestrator that takes the
instance over keeps what an earlier one wrote. Every instance whose plan holds a finalize stage
gets them; a cache warmer or a no-op instance does not."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, NamedTuple, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, ValidationError, model_validator

from hls.arms import ArmRecord
from hls.core.poly import dump_spec
from hls.core.score import Score
from hls.drive.record import DRIVE_FILE, TRACE_END, DriveRecord
from hls.obs.conditions import ArmNote, BudgetNote, HarnessConditions, harness_conditions
from hls.obs.record import (EvalEpisode, GradingRecord, RolloutBuild, RolloutProvenance, RolloutRecord,
                            finite_number, index_columns)
from hls.package.result import PackageResult
from hls.pipeline.plan import AttemptStatus, Failure, Plan, RecordsRefused, StagePlan
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_class, stage_fault
from hls.recipes.model import VENV
from hls.score.role import PROVENANCE_FILE, SCORE_FILE, ScoreProvenance
from hls.settings import StageName
from hls.spec.resources import ArtifactPlacement, ResourceShape
from hls.stages.agent_build import (BUILD_FILE, AgentBuildParams, AgentBuildStage, arm_route_secrets, budget_params,
                                   grades_after_sandbox_oom)
from hls.stages.deploy_drive import DeployDriveParams, ImageArtifact, PackagedArtifact, PublishedArtifact
from hls.stages.package import AGENT_BUILD_STAGE, PACKAGE_FILE, SOURCE_BUNDLE_OUTPUT, SUBMISSION_FILES, WORKSPACE
from hls.stages.provision import role_shape
from hls.stages.inputs import input_manifest, output, read_checked, read_output
from hls.stages.score import VERDICT_FILE, ScoreParams, drive_record, host_role_env
from hls.store.base import AlreadyExists, NotFound, Store, StoreError, check_key, check_segment
from hls.store.digest import sha256_bytes
from hls.store.layout import GRADINGS_DIR, ROLLOUTS, WARMERS, InstanceLayout
from hls.store.manifest import Manifest, read_manifest
from hls.store.posix import PosixStore
from hls.substrate.factory import substrate
from hls.substrate.secret_forms import FORMS, form_text, label_form
from hls.substrate.secret_scan import (CUT_PREFIX, FOUND_PREFIX, PACKAGE_SCAN_FILE, READ_PREFIX, UNREAD_PREFIX,
                                       PackageScan, PackageScanArea)
from hls.substrate.secrets import Scrubber, load_secrets
from hls.measured import load_pinned
from hls.taskref import TaskRef, first_on_import_path
from hls.util import validation_errors, walk_tree
from hls.verify.role import VERIFY_FILE, VerifyRecord

#: The scan role's name, and the file in its directory where it reports what it found.
SCAN, FOUND_FILE = "secret-scan", "found.jsonl"
#: The records the stage writes; the third only when the scan quarantined a file.
GRADING_FILE, ROLLOUT_FILE, QUARANTINE_FILE = "grading.json", "rollout.json", "quarantine.json"
#: The role logs a grading record names, by suffix.
LOG_SUFFIXES = (".out", ".err")
#: Files a scan fault names before it counts the rest.
SHOWN_HITS = 10
#: Quarantined files the index row names before it counts the rest, and cut streams an event does.
SHOWN_QUARANTINED = 20
#: The directory, under a failed instance's root, that holds the records the orchestrator
#: writes for it.
FAILED_DIR = "failed"
#: Where each area of a package scan lies in the submission image (``PackageScanArea.area``): the
#: checked-out tree's copy, and the venv the install added its files to. The bundle's objects lie
#: in no path of it.
SCAN_AREA_PATHS = {"tree": f"{WORKSPACE}/", "install": f"{VENV}/"}
#: What a record's bundle area is named by when the bundle its scan read is not in the store, as
#: one ``hls package --bundle`` names: ``<record>::<this>``, so a value found there is judged.
OUTSIDE_BUNDLE = "(the objects of a bundle outside the store)"
#: The reason name a failed instance's zero carries.
FAILED_REASON = "contestant_fault"
#: Seconds finalize is given, a fixed allowance: its own reads and records and the secret scan of
#: every file of the instance, which runs to the stage's deadline. The scan of a drive at
#: multimodal_tts's trace cap takes about 1,500 s, so the hour holds two such drives' traces; a
#: scan the stage's deadline ends is infra.
FINALIZE_SEC = 3600


def contestant_facing(plan: Plan, registry: Mapping[str, str], built_arm: ArmRecord | None = None) -> frozenset[str]:
    """The secrets a contestant could hold, by name: each one *plan* grants to a stage whose kind
    hands its grants to the contestant it runs (``Stage.grants_reach_contestant``), as the
    agent build hands its route to the agent; in a grading of an agent's package, the route
    secrets of *built_arm*, the arm that built it (``arm_route_secrets``), which the agent held and
    may have baked into its server; and every other name of *registry*, the
    site's file today, each name to its value, that holds the same value as one of these does
    there. The plan's grants and the arm decide, whatever *registry* holds: a record written
    while the site's file held a route that has been retired since, a package scan's among them,
    names a secret the contestant held. A scan finds a value's bytes, which say nothing of the name
    they came under, and the registry may hold one value under two names, as one URL can
    serve two routes. A stage's grants otherwise reach its harness roles only and are refused in a
    container (``StageContext.start_container``), so a secret granted to package or deploy_drive
    is not one. A value of any other secret found in an instance's files is the harness's own leak
    wherever it is; one of these may have come from the contestant, and is quarantined where a
    contestant's bytes can reach. The values are compared here in memory and never leave it."""
    granted: set[str] = set(arm_route_secrets(built_arm)) if built_arm is not None else set()
    for stage in plan.stages:
        try:
            cls = stage_class(stage.kind)
        except ValueError:
            continue
        if cls.grants_reach_contestant:
            granted |= set(stage.grants.values())
    held = {registry[name] for name in granted if name in registry}
    return frozenset(granted) | {name for name, value in registry.items() if value in held}


def _images(manifest: Manifest) -> list[str]:
    """The submission image *manifest*'s stage published, if any."""
    return [r.key for r in manifest.outputs.values() if r.key.endswith(tuple(f"/outputs/{n}" for n in SUBMISSION_FILES))]


def _bundles(manifests: list[Manifest]) -> list[str]:
    """The source bundle each of *manifests* published, if any."""
    return [m.outputs[SOURCE_BUNDLE_OUTPUT].key for m in manifests if SOURCE_BUNDLE_OUTPUT in m.outputs]


def tree_files(root: Path, *, skip: Path | None = None) -> tuple[list[Path], list[Path]]:
    """Every regular file under the directory *root*, and every other entry that is not a
    directory (a link, a device, a FIFO, a socket), each sorted, by a walk that reads no file,
    follows no link and leaves out the directory *skip* (``walk_tree``); nothing when *root* is
    not a directory. A directory that cannot be listed raises, as it fails the scan."""
    files: list[Path] = []
    others: list[Path] = []
    if root.is_dir() and not root.is_symlink():
        for entry in walk_tree(root, skip=() if skip is None else (skip,)):
            if entry.kind == "file":
                files.append(entry.path)
            elif entry.kind != "directory":
                others.append(entry.path)
    return files, others


def _files(path: Path, *, whole: bool) -> list[Path]:
    """The regular file at *path*, or with *whole* every regular file under the directory at it
    (``tree_files``), sorted; none for anything else, a link never followed."""
    if whole:
        return tree_files(path)[0]
    return [path] if path.is_file() and not path.is_symlink() else []


def shown_key(key: str) -> str:
    """*key* as text a record can hold: a byte of a name that is not UTF-8 as its escape."""
    return key.encode("utf-8", "surrogateescape").decode("utf-8", "backslashreplace")


def _prefixed(outputs: Path, prefix: str) -> list[Path]:
    """What *prefix* names under the outputs directory *outputs*: the directory a prefix ending
    with "/" names, else each entry of its directory whose name starts with its last part."""
    if prefix.endswith("/"):
        found = outputs / prefix.rstrip("/")
        return [found] if found.is_dir() else []
    parent, _, start = prefix.rpartition("/")
    folder = outputs / parent if parent else outputs
    return sorted(p for p in folder.iterdir() if p.name.startswith(start)) if folder.is_dir() else []


# The scan and its judgment, which the finalize stage and an export (``hls.stages.export``) share.

#: Joins a package scan's record and the path in the submission image of a file it found a value
#: in, into the name of that finding.
PACKAGE_SCAN_SEP = "::"


def _kind_of(plan: Plan, stage: str) -> type[Stage] | None:
    """The class of the kind of *plan*'s *stage*; None for a stage the plan does not name or a
    kind this process does not know."""
    kind = next((s.kind for s in plan.stages if s.name == stage), None)
    try:
        return stage_class(kind) if kind is not None else None
    except ValueError:
        return None


def declared_outputs(plan: Plan, stage: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The raw outputs, and those with the other contestant outputs, that the kind of *plan*'s
    *stage* declares (``Stage.raw_outputs``, ``Stage.contestant_outputs``); none for a stage the
    plan does not name or a kind this process does not know."""
    cls = _kind_of(plan, stage)
    if cls is None:
        return (), ()
    return cls.raw_outputs, cls.raw_outputs + cls.contestant_outputs


def contestant_file(plan: Plan, key: str) -> bool:
    """Whether the file at store key *key* of *plan*'s instance holds bytes a contestant chose:
    any file of an attempt of a stage whose kind runs a contestant's code
    (``Stage.runs_contestant_code``), whose harness logs and records echo the names and bytes the
    contestant chose; or, in another stage, an output its kind declares (``declared_outputs``),
    or the stdout or stderr of a harness role its kind says echoes them
    (``Stage.contestant_log_roles``). Every kind that starts a contestant role sets
    ``runs_contestant_code``, so no rule here reads a role's owner."""
    prefix = f"{plan.instance}/stages/"
    if not key.startswith(prefix):
        return False
    parts = key[len(prefix):].split("/")
    if len(parts) < 4 or parts[1] != "attempts":
        return False
    cls = _kind_of(plan, parts[0])
    if cls is None:
        return False
    if cls.runs_contestant_code:
        return True
    if len(parts) < 5:
        return False
    stage, area, rest = parts[0], parts[3], "/".join(parts[4:])
    if area == "outputs":
        return any(rest.startswith(p) for p in declared_outputs(plan, stage)[1])
    if area == "logs" and len(parts) == 5:
        role, _, suffix = rest.rpartition(".")
        return suffix in ("out", "err") and role in cls.contestant_log_roles
    return False


def _attempts_of(store: Store, plan: Plan, stage: str) -> list[int]:
    """The claimed attempts of *plan*'s *stage*; none for a name that is not a stage's."""
    try:
        return plan.layout.attempts(store, stage)
    except ValueError:
        return []


def scan_lists(store: PosixStore, plan: Plan, published: Sequence[Manifest], *,
               skip_gradings: bool) -> tuple[list[Path], list[Path]]:
    """What a scan of *plan*'s instance reads: every file under it present when the scan runs,
    its plan revisions, events, statuses, records and every attempt's outputs and logs, failed
    attempts' too, and each output of *published* by name, so one that is gone fails the scan;
    with *skip_gradings*, not the gradings under it, which are instances of their own. The
    second list is read as stored: every role log, each submission's job log, and the outputs a
    stage's kind declares raw (``Stage.raw_outputs``)."""
    layout, root = plan.layout, plan.instance
    listed = [store.path(f"{root}/{child}") for child in store.list_children(root)
              if not (skip_gradings and child == GRADINGS_DIR)]
    listed += [store.path(r.key) for m in published for r in m.outputs.values()]
    raw: list[Path] = []
    for stage in store.list_children(f"{root}/stages"):
        for n in _attempts_of(store, plan, stage):
            attempt = store.path(layout.attempt(stage, n))
            if (attempt / "logs").is_dir():
                raw.append(attempt / "logs")
            for prefix in declared_outputs(plan, stage)[0]:
                raw += _prefixed(attempt / "outputs", prefix)
    for k in store.list_children(f"{root}/allocations"):
        for n in store.list_children(f"{root}/allocations/{k}/submissions"):
            submission = f"{root}/allocations/{k}/submissions/{n}"
            raw += [store.path(f"{submission}/{name}") for name in store.list_children(submission)
                    if substrate(plan.settings.site).is_job_log(name)]
    return listed, raw


def unread_paths(store: PosixStore, plan: Plan) -> list[Path]:
    """The files and directories of *plan*'s instance's attempts that finalize's scan does not read
    (``Stage.unread_files``), each that is there: containers a package scan reads inside, or that
    are quarantined whole (``unscanned_containers``). A stage of a kind this process does not know
    has none: its every file is read, and quarantined whole as well."""
    found: list[Path] = []
    for stage in store.list_children(f"{plan.instance}/stages"):
        kind = _kind_of(plan, stage)
        unread = kind.unread_files if kind is not None else ()
        for n in _attempts_of(store, plan, stage):
            attempt = store.path(plan.layout.attempt(stage, n))
            found += [path for prefix in unread if os.path.lexists(path := attempt / prefix.rstrip("/"))]
    return found


def store_key(root: Path, path: Path) -> str:
    """The store key of the file at *path* in the store at *root*; the path itself when it is not
    in the store."""
    return path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)


class ScanReport(NamedTuple):
    """What the scan role reported (``hls.substrate.secret_scan``), each file by its store key
    (``store_key``)."""

    #: Each file it found a secret's form in, with the form's label (``SecretForm.label``).
    found: list[tuple[str, str]]
    #: Each gzip stream it found cut short.
    cut: list[str]
    #: Each file it read, with the sha256 of its bytes as stored, every one of which it searched.
    read: dict[str, str]
    #: Each link and special file it found and did not read: a link is never followed and a
    #: special file never opened, and no export carries either.
    unread: list[str]


def read_scan_report(report: Path, root: Path) -> ScanReport | None:
    """What the scan role reported in *report*, of files in the store at *root*; None when the
    report cannot be read."""
    try:
        lines = report.read_text(errors="surrogateescape").splitlines()

        def entries(prefix: str) -> list[dict[str, Any]]:
            return [json.loads(line[len(prefix):]) for line in lines if line.startswith(prefix)]

        return ScanReport(found=[(store_key(root, Path(h["file"])), h["secret"]) for h in entries(FOUND_PREFIX)],
                          cut=[store_key(root, Path(c["file"])) for c in entries(CUT_PREFIX)],
                          read={store_key(root, Path(r["file"])): str(r["sha256"]) for r in entries(READ_PREFIX)},
                          unread=[store_key(root, Path(u["file"])) for u in entries(UNREAD_PREFIX)])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def run_scan(ctx: StageContext, listed: Sequence[Path], raw: Sequence[Path], *, role: str = SCAN,
             unread: Sequence[Path] = ()) -> ScanReport:
    """Run the secret scan role *role* over *listed* and, read as stored, *raw*, reading nothing of
    *unread* but its paths, with every value of the stage's registry in its 0600 env file, and
    return its report (``ScanReport``): each
    file it found a secret's form in (``hls.substrate.secret_forms``), with the form's label, and
    each file it read, with the sha256 of the bytes it searched. The scan runs to the stage's
    deadline, which ends it as infra. Each gzip stream the role found cut
    short is a ``scan_cut_streams`` event, and each link and special file it did not read a
    ``scan_unread_entries`` event. A registry that holds no secret has nothing to find.
    Raise the fault a failure of the role is, and an infra fault that is not retried when the role
    ended well and its report cannot be read, since nothing then says what it found."""
    names = sorted(ctx.secrets)
    if not names:
        ctx.events.emit("secret_scan_skipped", reason="the secrets file holds no secret")
        return ScanReport(found=[], cut=[], read={}, unread=[])
    folder = ctx.role_dir(role)
    listing, stored, skipped = folder / "scan-list.txt", folder / "scan-raw.txt", folder / "scan-unread.txt"
    listing.write_text("".join(f"{path}\n" for path in listed))
    stored.write_text("".join(f"{path}\n" for path in raw))
    skipped.write_text("".join(f"{path}\n" for path in unread))
    argv = [str(ctx.plan.toolchain.python), "-m", "hls.substrate.secret_scan", "--names", ",".join(names),
            "--list", str(listing), "--raw", str(stored), "--unread", str(skipped),
            "--report", str(folder / FOUND_FILE)]
    shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
    ctx.start_role(role, argv, shape, env=host_role_env(ctx), secret_env=names, owner="harness")
    [outcome] = ctx.wait_roles(role)
    fault = stage_fault([outcome])
    if fault is not None:
        raise StageFault(fault)
    report = read_scan_report(folder / FOUND_FILE, ctx.store.root)
    if report is None:
        raise StageFault.infra("the scan ended, and its report of what it found could not be read", retryable=False)
    if report.cut:
        shown = [ctx.scrubber.scrub(shown_key(key)) for key in report.cut[:SHOWN_QUARANTINED]]
        ctx.events.emit("scan_cut_streams", count=len(report.cut), files=shown)
    if report.unread:
        shown = [ctx.scrubber.scrub(shown_key(key)) for key in report.unread[:SHOWN_QUARANTINED]]
        ctx.events.emit("scan_unread_entries", count=len(report.unread), files=shown)
    return report


def judge(findings: Sequence[tuple[str, str, bool]],
          facing: Collection[str]) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """Judge each secret's form found (``hls.substrate.secret_forms``), given as its file's store
    key, the form's label and whether a contestant's bytes reach the file (``contestant_file``):
    - a host, which grants no access and may be public, is quarantined in a contestant's file;
    - a value is judged by its secret first: one *facing* names, which a contestant could hold
      (``contestant_facing``), is quarantined in a contestant's file;
    - any other find is the harness's leak: a host in a file the harness alone wrote, which the
      harness scrubs of every form, or a value no contestant could hold or in such a file.
    Return the leaks, in the order found, and every file something was found in, by its key, with
    the labels found there, which are the files quarantined when there is no leak."""
    def ours(label: str, theirs: bool) -> bool:
        name, form = label_form(label)
        return not theirs or (form == "value" and name not in facing)

    quarantined: dict[str, set[str]] = {}
    for key, label, _ in findings:
        quarantined.setdefault(key, set()).add(label)
    return ([(key, label) for key, label, theirs in findings if ours(label, theirs)],
            {key: sorted(labels) for key, labels in sorted(quarantined.items())})


def leak_fault(ours: Sequence[tuple[str, str]], scrubber: Scrubber) -> StageFault:
    """The fault the harness's leaks *ours* (each a file's key and the form's label) are: infra
    and not retried, naming the first ``SHOWN_HITS`` files with what was found in each
    (``form_text``) and counting the rest, scrubbed."""
    named = [f"{shown_key(key)} ({form_text(label)})" for key, label in ours[:SHOWN_HITS]]
    more = f" and {len(ours) - SHOWN_HITS} more" if len(ours) > SHOWN_HITS else ""
    reason = f"the harness leaked secret values into published files: {', '.join(named)}{more}"
    return StageFault.infra(scrubber.scrub(reason), retryable=False)


def pinned_manifests(store: Store, manifest: Manifest) -> list[Manifest]:
    """The manifests *manifest*'s stage read (its ``inputs``), each checked against the digest it
    recorded. Raise an infra fault that is not retried for one that is gone or is not that one."""
    found = []
    for key, digest in sorted(manifest.inputs.items()):
        try:
            data = store.read_bytes(key)
        except NotFound as e:
            raise StageFault.infra(f"the manifest {key} that {manifest.stage} read is gone", retryable=False) from e
        if sha256_bytes(data) != digest:
            raise StageFault.infra(f"{key} is not the manifest {manifest.stage} read", retryable=False)
        found.append(Manifest.model_validate_json(data))
    return found


def facing_secrets(plan: Plan, registry: Mapping[str, str]) -> frozenset[str]:
    """The secrets of *registry* a contestant of *plan*'s instance could hold
    (``contestant_facing``): those its stages hand their contestants, and in a grading of an
    agent's package the route secrets of the arm that built it (``FinalizeParams.built_arm``).
    Finalize and an export of the instance judge its files by them."""
    params = finalize_params(plan)
    return contestant_facing(plan, registry, params.built_arm if params is not None else None)


def graded_drive(plan: Plan) -> tuple[StagePlan, DeployDriveParams] | None:
    """*plan*'s drive stage, the one its finalize stage's score stage names, with its parameters;
    None for a plan that grades nothing: one without a finalize stage, such as a warmer's, or a
    build-only rollout. Raise ValueError for a plan that names a stage it does not hold."""
    params = finalize_params(plan)
    if params is None or params.score_stage is None:
        return None
    try:
        score = ScoreParams.model_validate(plan.stage(params.score_stage).params)
        stage = plan.stage(score.drive_stage)
    except KeyError as e:
        raise ValueError(f"{plan.instance}: the plan has no stage {e.args[0]}, which its finalize stage names") from e
    return stage, DeployDriveParams.model_validate(stage.params)


def pinned_package(store: Store, pins: Mapping[str, str], key: str) -> Manifest:
    """The package manifest of another instance at *key* that a drive served, checked against the
    digest it was pinned by in *pins*, the drive stage's inputs in the plan, which the runner held
    the drive's reads to. Raise an infra fault that is not retried for one that is gone, that no
    pin names, or that is not the one pinned."""
    pinned = pins.get(key)
    try:
        data = store.read_bytes(key)
    except NotFound as e:
        raise StageFault.infra(f"the package manifest {key} is gone", retryable=False) from e
    if pinned is None or sha256_bytes(data) != pinned:
        raise StageFault.infra(f"{key} is not the package manifest the drive read", retryable=False)
    return Manifest.model_validate_json(data)


def scan_manifests(store: Store, plan: Plan, published: Sequence[Manifest]) -> list[Manifest]:
    """The manifests whose package scans *plan*'s instance answers for: its own *published* ones,
    and the package manifest of another instance its drive pinned (``pinned_package``), with the
    manifests that package read (``pinned_manifests``: source_scan's among them). An instance that
    grades nothing (``graded_drive``), or serves an image or a package of its own, pins none. Raise
    an infra fault that is not retried for a pinned manifest that is gone or is not the one its
    reader read, and ValueError for a plan that names a stage it does not hold."""
    drive = graded_drive(plan)
    artifact = drive[1].artifact if drive is not None else None
    if drive is None or not isinstance(artifact, PackagedArtifact) or artifact.manifest is None:
        return list(published)
    package = pinned_package(store, drive[0].inputs, artifact.manifest)
    return [*published, package, *pinned_manifests(store, package)]


def check_built_arm(store: Store, plan: Plan, published: Sequence[Manifest]) -> None:
    """Refuse a grading whose ``built_arm`` is not the arm that built the package it graded: when
    the inputs of the package its drive served (another instance's it pinned, or this instance's
    own among *published*, a rebuild's) reach an agent build, its ``build.json``, read against its
    manifest, names the arm, and the grading must name the same one; when they reach none, no arm
    built the package and the grading names none. The arm decides which secrets the scan counts
    the contestant's (``facing_secrets``), so a wrong one is infra that is not retried. An image
    graded by its digest has no package to read: the arm it names is the operator's
    (``--built-arm``). A rebuild whose package never published graded none, unless its drive
    published, which then served a package no manifest names: infra that is not retried. The
    drive's parameters and pins come from the plan, so a failed grading, whose drive may not have
    published, is checked too; any other instance passes."""
    params = finalize_params(plan)
    drive = graded_drive(plan)
    if params is None or params.kind != "grading" or drive is None:
        return
    stage, drive_params = drive
    artifact = drive_params.artifact
    if not isinstance(artifact, PackagedArtifact):
        return
    if artifact.manifest is not None:
        package = pinned_package(store, stage.inputs, artifact.manifest)
    else:
        found = next((m for m in published if m.stage == artifact.stage), None)
        if found is None:
            if any(m.stage == stage.name for m in published):
                raise StageFault.infra(f"the drive served the package {artifact.stage} made, which published no "
                                       "manifest", retryable=False)
            return
        package = found
    builds = [m for m in pinned_manifests(store, package) if m.stage == AGENT_BUILD_STAGE]
    if not builds and params.built_arm is not None:
        raise StageFault.infra(f"built_arm {params.built_arm.name} names an arm, and the graded package's inputs reach "
                               "no agent build: no arm built it", retryable=False)
    for build in builds:
        ref = output(build, BUILD_FILE)
        try:
            arm = ArmRecord.model_validate(json.loads(read_checked(store, ref))["arm"])
        except (ValueError, KeyError, TypeError) as e:
            raise StageFault.infra(f"{ref.key} names no arm: {type(e).__name__}", retryable=False) from e
        if params.built_arm is None:
            raise StageFault.infra(f"the graded package was built by the agent build {build.instance}, so the grading "
                                   f"names that build's arm {arm.name} in built_arm, which decides the secrets its "
                                   "server may hold", retryable=False)
        if params.built_arm != arm:
            raise StageFault.infra(f"built_arm {params.built_arm.name} is not the arm {arm.name} that built the graded "
                                   f"package ({ref.key})", retryable=False)


def package_scan(store: Store, manifest: Manifest) -> tuple[str, PackageScan] | None:
    """*manifest*'s ``package-scan.json``, its key and record, checked against the manifest; None
    when it records none. Raise an infra fault that is not retried for a record that is not the
    published one or not a package scan."""
    ref = next((r for r in manifest.outputs.values() if r.key.endswith(f"/outputs/{PACKAGE_SCAN_FILE}")), None)
    if ref is None:
        return None
    try:
        return ref.key, PackageScan.model_validate_json(read_checked(store, ref))
    except ValidationError as e:
        raise StageFault.infra(f"{ref.key} is not a package scan ({PACKAGE_SCAN_FILE}): {validation_errors(e)}",
                               retryable=False) from None


def area_holders(store: Store, manifest: Manifest, manifests: Sequence[Manifest], scan_key: str,
                 area: PackageScanArea) -> list[str]:
    """Everything that holds the files of *area*, which the record of *manifest* at *scan_key*
    names only by count: the area in the image, ``<scan_key>::<path>/`` (``SCAN_AREA_PATHS``; the
    bundle's objects have no path there); for the install (the package's record), the image
    *manifest* built; for the tree and the bundle's objects (source_scan's record), the source
    bundle *manifest* read, and for the tree the images of *manifests* whose stage read
    *manifest*'s. The objects of a bundle the store does not hold, as ``hls package --bundle``
    names one, are named ``<scan_key>::OUTSIDE_BUNDLE``, so what was found there is judged all
    the same."""
    held = [f"{scan_key}{PACKAGE_SCAN_SEP}{SCAN_AREA_PATHS[area.area]}"] if area.area in SCAN_AREA_PATHS else []
    if area.area == "install":
        return held + _images(manifest)
    held += _bundles(pinned_manifests(store, manifest))
    if area.area == "tree":
        key = InstanceLayout(manifest.instance).manifest(manifest.stage)
        held += [image for m in manifests if key in m.inputs for image in _images(m)]
    return held or [f"{scan_key}{PACKAGE_SCAN_SEP}{OUTSIDE_BUNDLE}"]


def package_scan_findings(store: Store, manifests: Iterable[Manifest]) -> list[tuple[str, str]]:
    """What the scans of the submission's own files found (``PackageScan``), for each of
    *manifests* that records a ``package-scan.json``: the source_scan stage's, of the bundle's
    objects and the checked-out tree, and the package stage's, of what the install added. Each
    finding is named as the record's key and the path the image holds the file at, joined by
    ``PACKAGE_SCAN_SEP``, with the label of the form found; for each area a form was found in
    that the record names only by count, everything that holds the area (``area_holders``). Raise an infra fault that is not retried for a record
    that is not the published one or not a package scan, or a manifest a scan read that is not the
    one it pinned."""
    manifests = list(manifests)
    findings: list[tuple[str, str]] = []
    for manifest in manifests:
        scanned = package_scan(store, manifest)
        if scanned is None:
            continue
        scan_key, record = scanned
        findings += [(f"{scan_key}{PACKAGE_SCAN_SEP}{f.path}", f.secret) for f in record.findings]
        for area in record.areas:
            findings += [(held, area.secret) for held in area_holders(store, manifest, manifests, scan_key, area)]
    return findings


def instance_containers(store: PosixStore, plan: Plan, manifests: Sequence[Manifest]) -> list[str]:
    """The store key of every file of *plan*'s instance's attempts whose bytes are a contestant's,
    compressed, as each stage's kind declares them (``Stage.compressed_files``), and of every file
    under the directories a kind declares (``Stage.unpublished_containers``) that its published
    manifest among *manifests* does not hold: neither an output it names nor a file under a
    directory it publishes whole (``Stage.published_dirs``, the transcript). Every kind declares
    all three (``register`` refuses one that does not). Nothing declares them for a directory of a
    stage the plan does not name, or of a kind this process does not know, which no runner of the
    plan writes: every file of its attempts is one, so the scan and the export fail closed."""
    root = plan.instance
    found: list[str] = []
    for stage in store.list_children(f"{root}/stages"):
        kind = _kind_of(plan, stage)
        if kind is None:
            for n in _attempts_of(store, plan, stage):
                attempt = store.path(plan.layout.attempt(stage, n))
                found += [store_key(store.root, p) for p in _files(attempt, whole=True)]
            continue
        declared = kind.compressed_files
        unpublished = kind.unpublished_containers
        manifest = next((m for m in manifests if m.instance == root and m.stage == stage), None)
        held = {r.key for r in manifest.outputs.values()} if manifest is not None else set()
        for n in _attempts_of(store, plan, stage):
            attempt_key = plan.layout.attempt(stage, n)
            attempt = store.path(attempt_key)
            for prefix in declared:
                found += [store_key(store.root, p)
                          for p in _files(attempt / prefix.rstrip("/"), whole=prefix.endswith("/"))]
            whole = [f"{attempt_key}/{d}" for d in kind.published_dirs
                     if manifest is not None and manifest.attempt == n]
            for prefix in unpublished:
                keys = [store_key(store.root, p)
                        for p in _files(attempt / prefix.rstrip("/"), whole=prefix.endswith("/"))]
                found += [key for key in keys if key not in held and not any(key.startswith(d) for d in whole)]
    return found


def unscanned_containers(store: PosixStore, plan: Plan, manifests: Sequence[Manifest],
                         registry: Iterable[str]) -> list[str]:
    """Every container no published scan read inside, among *plan*'s instance's containers
    (``instance_containers``) and the image of a package of another instance among *manifests*
    (one a grading's drive pinned). A record reads for this site only when it searched every
    secret of *registry*, the site's file today, in every form (``FORMS``): a scan run where the
    file held fewer, or none (its record reads no area), or that searched fewer forms, read
    nothing for the rest. A record that names no forms it searched is no
    package scan, and refused, infra and not retried (``package_scan``). A source bundle is read
    when such a record read the bundle's objects, and a submission image when its package's record
    read what the install added and a record the package pinned read its tree. Snapshots and what
    an attempt left unpublished no scan reads."""
    site = set(registry)

    def full(found: tuple[str, PackageScan] | None) -> PackageScan | None:
        if found is None or not set(found[1].secrets) >= site or not set(found[1].forms) >= set(FORMS):
            return None
        return found[1]

    covered: set[str] = set()
    for manifest in manifests:
        record = full(package_scan(store, manifest))
        if record is None:
            continue
        if "bundle" in record.read:
            covered |= set(_bundles(pinned_manifests(store, manifest)))
        if "install" in record.read and any(
                tree is not None and "tree" in tree.read
                for tree in (full(package_scan(store, m)) for m in pinned_manifests(store, manifest))):
            covered |= set(_images(manifest))
    containers = instance_containers(store, plan, manifests) + [
        image for m in manifests if m.instance != plan.instance for image in _images(m)]
    return [key for key in dict.fromkeys(containers) if key not in covered]


class FinalizeParams(BaseModel):
    """What the finalize stage records."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task, rebuilt for the provenance's ``spec``.
    task: TaskRef
    #: The eval the records describe: "terminal" for a rollout's own, the grading id for a
    #: grading.
    eval_id: Annotated[str, AfterValidator(check_segment)] = "terminal"
    #: Whether the instance is a rollout or a grading of one.
    kind: Literal["rollout", "grading"]
    #: The stage of this instance whose score is recorded; its parameters name the drive and
    #: verify stages. None for a build-only rollout, which grades nothing (``hls run
    #: --build-only``).
    score_stage: StageName | None = "score"
    #: The agent_build stage of this rollout whose ``build.json`` and outputs its build record
    #: holds; None for a rollout that built nothing, and for every grading.
    build_stage: StageName | None = None
    #: The package stage of this instance whose ``package.json`` the build record holds and a
    #: build-only rollout's status comes from; None for an instance that packages nothing.
    package_stage: StageName | None = None
    #: The arm whose agent built the package this grading grades, names only: its records name it,
    #: and its route secrets are contestant-facing (``contestant_facing``). None for a rollout,
    #: whose arm is its own build's, and for a package no agent built.
    built_arm: ArmRecord | None = None

    @model_validator(mode="after")
    def _recordable(self) -> Self:
        if self.kind == "grading" and self.score_stage is None:
            raise ValueError("a grading grades: it names its score stage")
        if self.kind == "grading" and self.build_stage is not None:
            raise ValueError("only a rollout records a build; a grading names the arm that built its package in "
                             "built_arm")
        if self.score_stage is None and self.package_stage is None:
            raise ValueError("a rollout that grades nothing records its package: it names its package stage")
        if self.kind == "rollout" and self.built_arm is not None:
            raise ValueError("built_arm is a grading's: a rollout's arm is its own build's")
        return self


def index_key(instance: str, kind: str) -> tuple[str, str, str | None]:
    """The key of *instance*'s index row, its rollout id and its grading id (None for a
    rollout). A rollout is ``runs/<task>/<rollout>``. A grading is one of a rollout's, or an
    evaluation under a cache warmer (``warmers/<task>/<warmer>/gradings/<grading>``, from
    ``hls grade --package``, ``--measured`` or ``--artifact.image``), whose row goes to the
    warmers' own index; the warmer stands in for the rollout. Raise ValueError for any other instance
    (``InstanceLayout.parse``)."""
    key = InstanceLayout.parse(instance)
    if not (kind == "rollout" and key.area == ROLLOUTS and key.grading_id is None
            or kind == "grading" and key.grading_id is not None):
        raise ValueError(f"{instance} is not a {kind} instance under {ROLLOUTS}/"
                         + (f" or {WARMERS}/" if kind == "grading" else ""))
    return key.index_row, key.id, key.grading_id


def record_mode(kind: str,
                artifact: PackagedArtifact | ImageArtifact | PublishedArtifact) -> Literal["original", "adopt", "rebuild"]:
    """How the graded artifact came to be served: built by this rollout, re-deployed from a
    published image, or rebuilt by this grading from the rollout's source."""
    if kind == "rollout":
        return "original"
    if isinstance(artifact, PackagedArtifact) and artifact.manifest is None:
        return "rebuild"
    return "adopt"


def build_params(plan: Plan, stage: str | None) -> AgentBuildParams | None:
    """The parameters of *plan*'s agent build *stage*; None for no stage, or for a stage of another
    kind, such as a test's stand-in. Raise KeyError for a stage the plan does not have."""
    if stage is None:
        return None
    found = plan.stage(stage)
    if found.kind != AgentBuildStage.kind:
        return None
    return AgentBuildParams.model_validate(found.params)


def arm_note(plan: Plan, params: FinalizeParams) -> ArmNote | None:
    """The arm *plan*'s records name: the rollout's own agent build's, with its declared change, or
    the arm that built the package a grading grades (``built_arm``); None for a submission that no
    agent built."""
    build = build_params(plan, params.build_stage)
    if build is not None:
        return ArmNote(record=build.arm, arm_change=build.arm_change)
    if params.built_arm is not None:
        return ArmNote(record=params.built_arm, arm_change=None)
    return None


def budget_note(params: AgentBuildParams, task_module: str) -> BudgetNote:
    """The budgets the build of *params* ran under, and whether they are not the task's own: the
    budgets of the producer of *task_module*'s ``ARTIFACT`` (``budget_params``). Whether they are is
    unknown (None) when the module declares no producer whose budgets the runner keeps."""
    ran = {"soft_sec": params.soft_sec, "hard_sec": params.hard_sec, "max_turns": params.max_turns}
    try:
        own = budget_params(importlib.import_module(task_module).ARTIFACT.producer)
    except (ImportError, AttributeError, ValueError):
        return BudgetNote(**ran, overridden=None)
    return BudgetNote(**ran, overridden=own != ran)


def sandbox_oom_rule(plan: Plan, build: AgentBuildParams | None) -> bool | None:
    """The rule *plan*'s agent build, whose parameters are *build*, ran under for a sandbox that
    ends on its own memory (``grades_after_sandbox_oom``); None for an
    instance that ran no build."""
    return grades_after_sandbox_oom(plan) if build is not None else None


def rollout_build(manifest: Manifest, build: dict[str, Any], package: dict[str, Any] | None) -> RolloutBuild:
    """A rollout's build record from its agent_build *manifest*, the ``build.json`` that manifest
    published (*build*) and the package stage's ``package.json`` (*package*). The usage is the
    spend the extraction role read (``extraction.usage``)."""
    extraction, upload = build.get("extraction"), build.get("upload_sha256")
    usage = extraction.get("usage") if isinstance(extraction, dict) else None
    return RolloutBuild(build=build, outputs={name: ref.key for name, ref in manifest.outputs.items()},
                        package=package, started_at=finite_number(build.get("started_at")),
                        ended_at=finite_number(build.get("ended_at")),
                        usage=usage if isinstance(usage, dict) else None,
                        upload_sha256=upload if isinstance(upload, str) else None)


def placement_note(drive: DriveRecord) -> ArtifactPlacement | None:
    """Where *drive*'s artifact step ran; None when it did not run or its step reported nothing."""
    return drive.artifact.placement if drive.artifact is not None else None


def index_row(*, rollout_id: str, grading_id: str | None, mode: str, created_at: float, finished_at: float | None,
              status: str, score: float | None, gate_passed: bool | None, cost_usd: float | None,
              image_digests: Mapping[str, str], failed_stage: str | None, fault: str | None, scanned: bool,
              quarantined: Sequence[str], spec: Mapping[str, Any], arm: ArmRecord | None) -> dict[str, Any]:
    """An instance's index row: finalize's, and the orchestrator's for a failed instance. Its
    columns, in order, are an ``index.jsonl`` line's but ``backend``, which a
    checkpoint that is a SIF digest leaves always null; the failed stage and its fault; the scan's
    three; and the arm, with the columns read from the record after it (``index_columns``).
    *quarantined* is every file the scan quarantined, of which the row names the first
    ``SHOWN_QUARANTINED`` and counts the rest; *arm* is the arm that built the graded submission."""
    return {"rollout_id": rollout_id, "grading_id": grading_id, "mode": mode, "created_at": created_at,
            "finished_at": finished_at, "status": status, "score": score, "gate_passed": gate_passed,
            "cost_usd": cost_usd, "image_digests": dict(image_digests), "failed_stage": failed_stage, "fault": fault,
            "scanned": scanned, "quarantined": list(quarantined)[:SHOWN_QUARANTINED],
            "quarantined_count": len(quarantined), "arm": arm.name if arm is not None else None,
            **index_columns(spec, arm.model if arm is not None else None)}


class _Finalize:
    """One finalize attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None:
            raise RuntimeError("the finalize stage needs node-local scratch")
        self.ctx = ctx
        self.params: FinalizeParams = ctx.params

    # inputs

    def published(self) -> list[Manifest]:
        """Every published manifest of this instance, in plan order."""
        ctx = self.ctx
        order = {s.name: i for i, s in enumerate(ctx.plan.stages)}
        names = sorted(ctx.store.list_children(f"{ctx.plan.instance}/stages"),
                       key=lambda n: (order.get(n, len(order)), n))
        found = [read_manifest(ctx.store, ctx.plan.layout, name) for name in names]
        return [m for m, _ in filter(None, found)]

    def logs(self, manifest: Manifest) -> dict[str, str]:
        """The file name of each role log of *manifest*'s attempt, to its key."""
        prefix = f"{self.ctx.plan.layout.attempt(manifest.stage, manifest.attempt)}/logs"
        names = self.ctx.store.list_children(prefix)
        return {name: f"{prefix}/{name}" for name in names if name.endswith(LOG_SUFFIXES)}

    # records

    def write(self, file: str, record: BaseModel) -> str:
        """Write *record* to output *file*, non-finite numbers as null and every secret value
        replaced, and return its key."""
        text = self.ctx.scrubber.scrub(record.model_dump_json(indent=2))
        return self.ctx.write_output(file, text.encode())

    def load(self) -> Any:
        """The task as the roles rebuilt it, its calibration and baseline credit included."""
        p = self.params
        try:
            return load_pinned(p.task, self.ctx.plan.measured, calibration=True, credit=True,
                               checkout=self.ctx.plan.code.checkout, store_root=self.ctx.plan.store_root)
        except Exception as e:
            raise StageFault.infra(f"task {p.task.module} does not load: {type(e).__name__}: {e}",
                                   retryable=False) from e

    def provenance(self, task: Any, drive: DeployDriveParams | None, published: list[Manifest],
                   arm: ArmRecord | None) -> RolloutProvenance:
        """What the instance ran: *task* as its roles rebuilt it, every image digest of *published*,
        the code, the artifact *drive* served (none for a build-only rollout, which drove nothing)
        and the reasoning effort of the *arm* that built it."""
        ctx, p = self.ctx, self.params
        images: dict[str, str] = {}
        for m in published:
            images |= m.provenance.image_digests
        code = ctx.plan.code
        return RolloutProvenance(task_id=ctx.plan.task, split=drive.split if drive is not None else "test",
                                 spec=dump_spec(task), image_digests=images, hls_commit=code.hls_commit,
                                 harbor_commit=code.harbor_commit, rollout_overrides=dict(p.task.overrides),
                                 artifact=drive.artifact.model_dump(mode="json") if drive is not None else {},
                                 reasoning_effort=arm.effort if arm is not None else None)

    def credit(self, score: Manifest) -> dict[str, Any] | None:
        """The baseline credit the *score* stage's role recorded it scored against
        (``score-provenance.json``, checked against its manifest); None for a scorer that scores
        against none."""
        ref = output(score, PROVENANCE_FILE)
        try:
            baseline = ScoreProvenance.model_validate_json(read_output(self.ctx, ref)).baseline
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a score provenance: {validation_errors(e)}",
                                   retryable=False) from None
        return baseline.model_dump(mode="json") if baseline is not None else None

    def calibration(self, verify: Manifest) -> dict[str, Any] | None:
        """The calibration file the *verify* stage's role recorded it gave the verifiers
        (``verify.json``, checked against its manifest); None for a task none of whose verifiers
        needs one, and for a drive with nothing to verify."""
        ref = output(verify, VERIFY_FILE)
        try:
            blob = VerifyRecord.model_validate_json(read_output(self.ctx, ref)).calibration
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a verify record: {validation_errors(e)}",
                                   retryable=False) from None
        return blob.model_dump(mode="json") if blob is not None else None

    def conditions(self, published: list[Manifest], credit: dict[str, Any] | None,
                   calibration: dict[str, Any] | None, arm: ArmNote | None,
                   placement: ArtifactPlacement | None, drive_stage: str | None) -> HarnessConditions:
        """The instance's harness conditions now (``hls.obs.conditions``), with the hardware the
        counted attempt of *drive_stage* ran on."""
        ctx, p = self.ctx, self.params
        build = build_params(ctx.plan, p.build_stage)
        return harness_conditions(ctx.store, ctx.plan, published=published, credit=credit, calibration=calibration,
                                  arm=arm,
                                  budget=budget_note(build, p.task.module) if build is not None else None,
                                  placement=placement, drive_stage=drive_stage,
                                  grade_after_sandbox_oom=sandbox_oom_rule(ctx.plan, build))

    def package(self) -> dict[str, Any] | None:
        """``package.json`` of this instance's package stage as written, checked against its
        manifest; None for an instance that names none."""
        ctx, p = self.ctx, self.params
        if p.package_stage is None:
            return None
        try:
            record = json.loads(read_output(ctx, output(input_manifest(ctx, p.package_stage), PACKAGE_FILE)))
        except ValueError as e:
            raise StageFault.infra(f"the package's record does not parse: {e}", retryable=False) from e
        if not isinstance(record, dict):
            raise StageFault.infra(f"{PACKAGE_FILE} holds a JSON object", retryable=False)
        return record

    def build(self) -> RolloutBuild | None:
        """The rollout's build record, from its agent_build and package manifests, each file
        checked against its manifest; None for an instance that records no build."""
        ctx, p = self.ctx, self.params
        if p.build_stage is None:
            return None
        manifest = input_manifest(ctx, p.build_stage)
        try:
            record = json.loads(read_output(ctx, output(manifest, BUILD_FILE)))
        except ValueError as e:
            raise StageFault.infra(f"the build's record does not parse: {e}", retryable=False) from e
        if not isinstance(record, dict):
            raise StageFault.infra(f"{BUILD_FILE} holds a JSON object", retryable=False)
        return rollout_build(manifest, record, self.package())

    def write_row(self, index: str, row: dict[str, Any]) -> None:
        """Create the instance's index row *row* at *index*, scrubbed. A row already there is infra
        that is not retried (see the module docstring)."""
        ctx = self.ctx
        try:
            ctx.store.create(index, ctx.scrubber.scrub(json.dumps(row, allow_nan=False) + "\n").encode())
        except AlreadyExists as e:
            raise StageFault.infra(f"the index row {index} already exists; an earlier attempt of this instance, or "
                                   "another instance of the same ids, wrote it", retryable=False) from e

    def record_built(self, index: str, rollout_id: str, published: list[Manifest], task: Any,
                     arm: ArmNote | None) -> dict[str, str]:
        """The records of a build-only rollout, which grades nothing: ``rollout.json`` with status
        ``built``, or ``invalid_artifact`` when its package did not build, the scan and the index
        row, with no score."""
        ctx, p = self.ctx, self.params
        assert p.package_stage is not None  # FinalizeParams names one when no score stage is named
        ref = output(input_manifest(ctx, p.package_stage), PACKAGE_FILE)
        try:
            result = PackageResult.model_validate_json(read_output(ctx, ref))
        except ValidationError as e:
            raise StageFault.infra(f"{ref.key} is not a package record: {validation_errors(e)}",
                                   retryable=False) from None
        status = "built" if result.status == "built" else "invalid_artifact"
        record = arm.record if arm is not None else None
        provenance = self.provenance(task, None, published, record)
        build = self.build()
        finished = ctx.clock()
        outputs = {"rollout": self.write(ROLLOUT_FILE, RolloutRecord(
            rollout_id=rollout_id, task_id=ctx.plan.task, created_at=ctx.plan.created_at, finished_at=finished,
            status=status, config=provenance, build=build,
            conditions=self.conditions(published, None, None, arm, None, None)))}
        quarantined = self.scan(published)
        if quarantined:
            outputs["quarantine"] = self.write_quarantine(quarantined)
        self.write_row(index, index_row(
            rollout_id=rollout_id, grading_id=None, mode="original", created_at=ctx.plan.created_at,
            finished_at=finished, status=status, score=None, gate_passed=None,
            cost_usd=None if build is None else build.cost_usd,
            image_digests=provenance.image_digests, failed_stage=None, fault=None, scanned=True,
            quarantined=list(quarantined), spec=provenance.spec, arm=record))
        return outputs

    # the scan

    def scan(self, published: list[Manifest]) -> dict[str, list[str]]:
        """Read what each package's own scan found (``package_scan_findings``) in the manifests
        the instance answers for (``scan_manifests``), scan every file under the instance
        (``scan_lists``, ``run_scan``), then judge each form of a secret found (``judge``): a host
        is quarantined where a contestant's bytes reach (``contestant_file``); a value no
        contestant could hold (``facing_secrets``) is the harness's leak wherever it is found, and
        one a contestant could hold is quarantined where a contestant's bytes reach; anything
        found in a file the harness alone wrote is the harness's leak. Return the files
        quarantined, each as a record shows its key (``shown_key``), with the labels of the forms
        found in it, every name that holds a value found among them, and each container no
        published scan read inside (``unscanned_containers``) with none; raise the fault a leak
        is, infra and not retried, and the fault any failure of the role is."""
        ctx = self.ctx
        manifests = scan_manifests(ctx.store, ctx.plan, published)
        findings = [(key, label, True) for key, label in package_scan_findings(ctx.store, manifests)]
        listed, raw = scan_lists(ctx.store, ctx.plan, published, skip_gradings=self.params.kind == "rollout")
        unread = unread_paths(ctx.store, ctx.plan)
        findings += [(key, label, contestant_file(ctx.plan, key))
                     for key, label in run_scan(ctx, listed, raw, unread=unread).found]
        ours, quarantined = judge(findings, facing_secrets(ctx.plan, ctx.secrets))
        if ours:
            raise leak_fault(ours, ctx.scrubber)
        shown: dict[str, set[str]] = {}
        for key, labels in quarantined.items():
            shown.setdefault(shown_key(key), set()).update(labels)
        for key in unscanned_containers(ctx.store, ctx.plan, manifests, ctx.secrets):
            shown.setdefault(shown_key(key), set())
        if shown:
            ctx.events.emit("quarantined", files=len(shown))
        return {key: sorted(labels) for key, labels in sorted(shown.items())}

    def write_quarantine(self, quarantined: dict[str, list[str]]) -> str:
        """Write ``quarantine.json``: the files a value was found in that hold a contestant's
        bytes, each with the names of the secrets found in it, scrubbed; they are kept out of any
        export of the instance."""
        text = json.dumps({"files": quarantined}, indent=2, sort_keys=True)
        return self.ctx.write_output(QUARANTINE_FILE, self.ctx.scrubber.scrub(text).encode())

    # the stage

    def run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        if not ctx.secret_registry:
            raise StageFault.infra("finalize runs only with the secrets registry, which the runner did not hand it: "
                                   "it would write records it cannot scrub and skip the secret scan", retryable=False)
        try:
            index, rollout_id, grading_id = index_key(ctx.plan.instance, p.kind)
        except ValueError as e:
            raise StageFault.infra(str(e), retryable=False) from e
        if grading_id is not None and p.eval_id != grading_id:
            raise StageFault.infra(f"a grading's eval is named by its grading id {grading_id}, not {p.eval_id}",
                                   retryable=False)
        try:
            arm = arm_note(ctx.plan, p)
        except (KeyError, ValidationError) as e:
            raise StageFault.infra(f"the plan's agent build {p.build_stage} cannot be read: {type(e).__name__}",
                                   retryable=False) from e
        task = self.load()
        published = self.published()
        if p.score_stage is None:
            return self.record_built(index, rollout_id, published, task, arm)
        try:
            score_params = ScoreParams.model_validate(ctx.plan.stage(p.score_stage).params)
            drive_params = DeployDriveParams.model_validate(ctx.plan.stage(score_params.drive_stage).params)
        except KeyError as e:
            raise StageFault.infra(f"the plan has no stage {e.args[0]}, which finalize records", retryable=False) from e
        drive_m = input_manifest(ctx, score_params.drive_stage)
        verify_m = input_manifest(ctx, score_params.verify_stage)
        score_m = input_manifest(ctx, p.score_stage)
        if p.kind == "grading":
            check_built_arm(ctx.store, ctx.plan, published)
        drive: DriveRecord = drive_record(ctx, output(drive_m, DRIVE_FILE))
        verdict = json.loads(read_output(ctx, output(verify_m, VERDICT_FILE)))
        score = json.loads(read_output(ctx, output(score_m, SCORE_FILE)))
        record = arm.record if arm is not None else None
        provenance = self.provenance(task, drive_params, published, record)
        conditions = self.conditions(published, self.credit(score_m), self.calibration(verify_m), arm,
                                     placement_note(drive), score_params.drive_stage)
        logs = {name: key for m in (drive_m, verify_m, score_m) for name, key in self.logs(m).items()}
        trace = output(drive_m, TRACE_END).key.rsplit("/", 1)[0] if drive.outcome == "served" else ""
        episode = EvalEpisode(
            eval_id=p.eval_id, triggered_by="harness" if p.kind == "rollout" else "regrade",
            checkpoint=drive.artifact.image_sha256 if drive.artifact is not None else "",
            started_at=drive_m.times.started_at, finished_at=score_m.times.ended_at,
            trace_path=trace,
            verdict_summary={"gate_passed": score["gate_passed"], "aggregate_passed": verdict["aggregate_passed"],
                             "diagnostics": verdict["diagnostics"]},
            score=score, container_logs=logs)
        mode = record_mode(p.kind, drive_params.artifact)
        outputs = {"grading": self.write(GRADING_FILE, GradingRecord(
            episode=episode, provenance=provenance, source_rollout=rollout_id, mode=mode, conditions=conditions,
            package=self.package() if p.kind == "grading" else None))}
        status = "invalid_artifact" if drive.outcome == "invalid_artifact" else "ok"
        build = self.build()
        finished = ctx.clock()
        if p.kind == "rollout":
            outputs["rollout"] = self.write(ROLLOUT_FILE, RolloutRecord(
                rollout_id=rollout_id, task_id=ctx.plan.task, created_at=ctx.plan.created_at, finished_at=finished,
                status=status, config=provenance, build=build, conditions=conditions))
        quarantined = self.scan(published)
        if quarantined:
            outputs["quarantine"] = self.write_quarantine(quarantined)
        self.write_row(index, index_row(
            rollout_id=rollout_id, grading_id=grading_id, mode=mode, created_at=ctx.plan.created_at,
            finished_at=finished, status=status, score=score.get("value"), gate_passed=score["gate_passed"],
            cost_usd=None if build is None else build.cost_usd, image_digests=provenance.image_digests,
            failed_stage=None, fault=None, scanned=True, quarantined=list(quarantined), spec=provenance.spec,
            arm=record))
        return outputs


@register
class FinalizeStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "finalize"
    params_model: ClassVar[type[BaseModel]] = FinalizeParams
    # No file of its attempts holds a contestant's bytes compressed (``Stage.compressed_files``,
    # ``Stage.unpublished_containers``, ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    #: The runner loads the secrets file for this stage although it has no grants.
    needs_secret_registry: ClassVar[bool] = True

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the secret scan."""
        return 1

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        assert isinstance(params, FinalizeParams)
        if shape.gpus:
            raise ValueError(f"the finalize stage writes records and uses no GPU; its shape asks for {shape.gpus}")
        role_shape(shape, runner_memory_gib)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Finalize(ctx).run()


# A failed instance's records


def failed_key(layout: InstanceLayout, name: str) -> str:
    """The store key of the failed-instance record *name* of the instance *layout* names."""
    return check_key(f"{layout.root}/{FAILED_DIR}/{check_segment(name)}")


def finalize_params(plan: Plan) -> FinalizeParams | None:
    """The parameters of *plan*'s finalize stage; None for a plan without one, whose instance
    records no score (a cache warmer, a no-op instance)."""
    stage = next((s for s in plan.stages if s.kind == FinalizeStage.kind), None)
    return None if stage is None else FinalizeParams.model_validate(stage.params)


def ending_failure(plan: Plan) -> Failure:
    """The allocation failure whose contestant fault ended *plan*'s instance failed."""
    if plan.outcome is None or plan.outcome.status != "failed":
        raise ValueError(f"{plan.instance} did not end failed")
    found = [a.failure for a in plan.allocations if a.failure is not None and a.failure.fault == plan.outcome.fault]
    if not found:
        raise ValueError(f"{plan.instance} ended failed, but no allocation failed with its fault")
    return found[0]


def _attempt_times(store: Store, plan: Plan, stage: str) -> tuple[float, float | None]:
    """When *stage*'s last attempt started and ended, from its status; the plan's creation and
    no end when it has none (an allocation that failed before it claimed the attempt)."""
    layout = plan.layout
    for n in reversed(layout.attempts(store, stage)):
        try:
            status = AttemptStatus.model_validate_json(store.read_bytes(layout.attempt_file(stage, n, "status.json")))
        except (NotFound, ValidationError):
            continue
        return status.started_at, status.ended_at
    return plan.created_at, None


def failed_task(plan: Plan, params: FinalizeParams) -> tuple[Any, str | None]:
    """The task of *plan*'s failed instance as finalize loads it, from the plan's checkout, with
    its calibration and baseline credit; or None and the type of the error that stopped it, which
    the zero's diagnostics name in place of the spec. The type alone: a message could quote what
    the plan holds."""
    with first_on_import_path(plan.code.checkout):
        try:
            return load_pinned(params.task, plan.measured, calibration=True, credit=True,
                               checkout=plan.code.checkout, store_root=plan.store_root), None
        except Exception as e:  # noqa: BLE001 -- the zero stands without its spec, and says why
            return None, type(e).__name__


def _published_output(store: Store, manifest: Manifest, file: str) -> bytes:
    """The bytes of *manifest*'s output *file*, checked against the digest it recorded."""
    ref = output(manifest, file)
    data = store.read_bytes(ref.key)
    if sha256_bytes(data) != ref.sha256:
        raise ValueError(f"{ref.key} does not match its manifest")
    return data


def _failed_build(store: Store, params: FinalizeParams, published: Sequence[Manifest]) -> RolloutBuild | None:
    """The build record of a failed rollout whose agent_build published, with ``package.json`` when
    its package stage published too; None otherwise."""
    by_stage = {m.stage: m for m in published}
    build = by_stage.get(params.build_stage) if params.build_stage is not None else None
    if build is None:
        return None
    package = by_stage.get(params.package_stage) if params.package_stage is not None else None
    return rollout_build(build, json.loads(_published_output(store, build, BUILD_FILE)),
                         json.loads(_published_output(store, package, PACKAGE_FILE)) if package is not None else None)


def _failed_placement(store: Store, drive_stage: str | None,
                      published: Sequence[Manifest]) -> ArtifactPlacement | None:
    """Where the artifact of a failed instance's drive ran, when the drive published its record
    before a later stage failed; None otherwise. The placement is the record's account of how the
    instance ran, which no zero depends on, so a drive that published no record gives none."""
    drive = next((m for m in published if m.stage == drive_stage), None)
    if drive is None or not any(r.key.endswith(f"/outputs/{DRIVE_FILE}") for r in drive.outputs.values()):
        return None
    return placement_note(DriveRecord.model_validate_json(_published_output(store, drive, DRIVE_FILE)))


def record_failed(store: Store, plan: Plan, params: FinalizeParams, *, scrubber: Scrubber) -> dict[str, str]:
    """Write the records of *plan*'s failed instance, whose finalize stage has *params*, and
    return each one's name and key: the zero ``score.json`` naming the failed stage and its
    fault; for an instance that grades, the ``grading.json`` around it; for a rollout,
    ``rollout.json``, with its build when its agent_build published; each under ``failed/`` and
    with the harness-conditions table; and the index row, in the finalize stage's place for it.
    The provenance holds the task's spec, loaded from the plan's checkout as finalize loads it, or
    none and the error's type among the zero's diagnostics (``spec_error``). Every record has every
    value *scrubber* knows replaced and is created only if it is not there yet, so a record an
    earlier orchestrator of the instance wrote stands. An instance that has no place for an index
    row (``index_key``) gets the records without one."""
    failure = ending_failure(plan)
    fault, layout = failure.fault, plan.layout
    published = [m for m, _ in filter(None, (read_manifest(store, layout, s.name) for s in plan.stages))]
    images: dict[str, str] = {}
    for m in published:
        images |= m.provenance.image_digests
    drive_stage = next((s for s in plan.stages if s.kind == "deploy_drive"), None)
    drive = DeployDriveParams.model_validate(drive_stage.params) if drive_stage is not None else None
    task, spec_error = failed_task(plan, params)
    note = arm_note(plan, params)
    arm = note.record if note is not None else None
    provenance = RolloutProvenance(task_id=plan.task, split=drive.split if drive is not None else "test",
                                   spec=dump_spec(task) if task is not None else {}, image_digests=images,
                                   hls_commit=plan.code.hls_commit, harbor_commit=plan.code.harbor_commit,
                                   rollout_overrides=dict(params.task.overrides),
                                   artifact=drive.artifact.model_dump(mode="json") if drive is not None else {},
                                   reasoning_effort=arm.effort if arm is not None else None)
    diagnostics: dict[str, Any] = {"stage": failure.stage, "reason": fault.reason, "evidence": dict(fault.evidence)}
    if spec_error is not None:
        diagnostics["spec_error"] = spec_error
    score = Score.zero(reasons=(FAILED_REASON,), diagnostics=diagnostics)
    started, ended = _attempt_times(store, plan, failure.stage)
    try:
        index, rollout_id, grading_id = index_key(plan.instance, params.kind)
    except ValueError:
        index, rollout_id, grading_id = None, layout.root.split("/")[-1], None
    mode = "original" if params.kind == "rollout" else "adopt"
    if drive is not None:
        mode = record_mode(params.kind, drive.artifact)
    build = _failed_build(store, params, published)
    build_stage = build_params(plan, params.build_stage)
    if build_stage is None:
        budget = None
    elif task is None:
        # The task's module did not load, so its own budgets cannot be read either.
        budget = BudgetNote(soft_sec=build_stage.soft_sec, hard_sec=build_stage.hard_sec,
                            max_turns=build_stage.max_turns, overridden=None)
    else:
        with first_on_import_path(plan.code.checkout):
            budget = budget_note(build_stage, params.task.module)
    # No stage after verify can fail as the contestant's, so a failed instance published no verify
    # record and no score: no calibration was given to verifiers and no credit scored against.
    conditions = harness_conditions(store, plan, published=published, credit=None,
                                    calibration=None, arm=note, budget=budget,
                                    placement=_failed_placement(store, drive_stage.name if drive_stage else None,
                                                                published),
                                    drive_stage=drive_stage.name if drive_stage else None,
                                    grade_after_sandbox_oom=sandbox_oom_rule(plan, build_stage))
    records: dict[str, tuple[str, BaseModel]] = {"score": (SCORE_FILE, score)}
    if params.score_stage is not None:
        episode = EvalEpisode(eval_id=params.eval_id,
                              triggered_by="harness" if params.kind == "rollout" else "regrade",
                              checkpoint="", started_at=started, finished_at=ended, trace_path="",
                              verdict_summary={"gate_passed": False, "aggregate_passed": False, "diagnostics": {}},
                              score=score.model_dump(mode="json"))
        records["grading"] = (GRADING_FILE, GradingRecord(episode=episode, provenance=provenance,
                                                          source_rollout=rollout_id, mode=mode,
                                                          conditions=conditions))
    if params.kind == "rollout":
        records["rollout"] = (ROLLOUT_FILE, RolloutRecord(rollout_id=rollout_id, task_id=plan.task,
                                                          created_at=plan.created_at, finished_at=ended, status="ok",
                                                          config=provenance, build=build, conditions=conditions))
    out = {name: _create(store, failed_key(layout, file), scrubber.scrub(record.model_dump_json(indent=2)))
           for name, (file, record) in records.items()}
    if index is not None:
        row = index_row(rollout_id=rollout_id, grading_id=grading_id, mode=mode, created_at=plan.created_at,
                        finished_at=ended, status="ok", score=score.value, gate_passed=False,
                        cost_usd=None if build is None else build.cost_usd, image_digests=images,
                        failed_stage=failure.stage, fault=fault.reason, scanned=False, quarantined=[],
                        spec=provenance.spec, arm=arm)
        out["index"] = _create(store, index, scrubber.scrub(json.dumps(row, allow_nan=False) + "\n"))
    return out


def _create(store: Store, key: str, text: str) -> str:
    """Create *key* with *text* unless it is already there, and return *key*."""
    try:
        store.create(key, text.encode())
    except AlreadyExists:
        pass
    return key


def failed_records(store: Store, plan: Plan) -> dict[str, str]:
    """``record_failed`` for *plan*'s failed instance when its plan has a finalize stage, with
    a scrubber for every value of the site's secrets file; {} for a plan without one. The file
    must be usable, as for the finalize stage: SecretsError otherwise, and nothing is written."""
    params = finalize_params(plan)
    if params is None:
        return {}
    scrubber = Scrubber.from_secrets(load_secrets(plan.settings.site.secrets_path))
    try:
        return record_failed(store, plan, params, scrubber=scrubber)
    except (StoreError, OSError):
        raise
    except Exception as e:
        # Its message can quote what the plan holds; the refusal carries it scrubbed.
        raise RecordsRefused(scrubber.scrub(f"{type(e).__name__}: {e}")) from None
