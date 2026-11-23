"""The export stage: finished instances' publishable files, copied out of the store once a secret
scan of every file of each has cleared them (``hls export``).

The stage runs in the runner, on the CPU partition, and moves no file's bytes itself. Its
parameters name the instances, each finished and of the plan's task (``hls.pipeline.export``
names a rollout's or a warmer's gradings beside it), and the destination, a directory that does
not exist yet. In order:
1. For each instance it reads its latest plan and its published manifests and chooses what
   leaves (``select``): the latest plan revision; each published stage's manifest and every file
   of the attempt it counted; for an instance a contestant fault ended, the last attempt of the
   stage whose fault it was, the evidence of its zero; everything under ``failed/``; and the
   instance's index row. What stays behind is named with why: an earlier plan revision, an
   attempt the instance did not count (only the attempt each stage counted is published), the
   lease, the orchestrators' and the runners' records, a temporary file the store is writing
   (``STORE_TEMP``), anything else, and an entry that is neither a regular file nor a directory.
   Any other dot-named file is judged and copied as any other, since the scan reads dot-named
   files too. The walk reads directory entries only and follows no link.
2. It runs finalize's scan role once (``hls.stages.finalize.run_scan``) over every file of every
   instance, as finalize lists them (``scan_lists``), each instance's index row, and with
   ``oracle`` the task's answers area. The role searches every form of each secret of the site's
   file (``hls.substrate.secret_forms``: its value, and the host of a value that is an http(s)
   URL when it identifies a machine, in any case), and reports each file it read with the sha256
   of the bytes it searched (``ScanReport.read``). Each form found is judged by finalize's rule
   (``judge``) with the plan of the instance its file belongs to (``owner_of``): a host is
   quarantined in a file a contestant's bytes reach (``contestant_file``); a value of a secret a
   contestant of that instance could hold (``facing_secrets``: one its plan grants a contestant
   and, in a grading of an agent's package, the route secrets of the arm that built it, which
   ``check_built_arm`` holds to the build's record, failed gradings included, whether or not the
   site's file still holds them) is quarantined in such a file; anything else is the harness's
   leak, and so is anything found in the answers area or in a file of no instance, which no
   contestant writes. Each package scan's record (``package-scan.json``) of every package
   manifest an instance published or pinned (``scan_manifests``) is judged the same way, each
   finding one in a contestant's file. A file's store key, which the export would
   publish as the copy's path and in ``EXPORT.json``, is judged as its bytes are: the role finds
   a form in every path below each one it lists, and the rest of a key, the instance's root and
   the task, is in the instance's plan, which the role reads. A leak anywhere ends the stage,
   infra and not retried, with nothing copied.
3. What leaves is what step 1 chose less what is quarantined: what the export's own judgment
   quarantined, what the instance's finalize quarantined (``quarantine.json``, read against its
   manifest, whatever the secrets file holds now; an entry with no secret is a container no scan
   read inside), every container of that instance that the published scans did not read inside
   for every secret the registry holds now in every form (``unscanned_containers``: a submission
   image, a source bundle or a snapshot, and a file an agent build left in its outputs
   unpublished, the transcript's published files apart), and, for a finding of a package scan,
   every container of that instance (``instance_containers``), whose compression hid the file
   from every scan as stored; a form a scan named only by its area stands found in the whole
   area, and in the objects of a bundle outside the store (``OUTSIDE_BUNDLE``) when the store
   holds no bundle it read. Finalize's record is scrubbed, so its keys are matched with their
   redactions standing for any text of one character or more (``RecordMatcher``), in one pass
   over the key matched; two paths that match alike both stay behind.
4. The stage opens the destination's directory once, each component of its path without
   following a link (``open_directory``), and makes, checks, renames and removes everything beside
   the destination through that descriptor. It makes a directory there, named for this export and
   attempt, and in it the directory the copy goes into, each through a descriptor of the
   directory it is made in and held open (``make_partial``), and claims the destination by making
   it, an empty directory (``claim_destination``), each where nothing is; it records both in the
   attempt's ``CLAIM_FILE`` (``ClaimRecord``) before anything is copied. A harness role
   (``hls.store.export_copy``) copies each file into the copy's directory, which it opens with no
   link followed and copies into only while it is the directory the stage made, and reports each
   copy's sha256, read back from what landed, and its size. Each copy must have the digest the
   scan reported for the file it searched, so what leaves is what was searched; another means
   the store's file changed after the scan, infra and retried. A copy of a published output must
   also have the digest its manifest recorded; another means the store's file changed after it
   was published, infra and not retried. The stage writes ``EXPORT.json`` (``ExportRecord``) into
   the copy and as its own output, renames the copy onto its claim (``place``), which replaces
   only a directory still empty: whatever another writer put at the destination in the meantime
   makes the export fail and stays as it is; and removes the directory it made for the copy.

The destination mirrors the store: each file is at ``<destination>/<its store key>``, the index
row at its own key, and the answers at ``oracle/<task>/...``.

Every failure here is the harness's. A scan role that fails is retried; one that ended well and
left no report that can be read is not, and neither is a leak, an instance that is gone, still
running or whose records contradict their manifests, a grading whose arm is not the one that
built its package, a registry that holds no secret, a destination that exists or whose directory
a link leads to, or one another writer changed after the export claimed it, since a retry reads
the same files. A copy role that fails, whose report does not name every file it was given, or
whose copy is not what the scan read, is retried. Once the stage has made its directories, a
failure removes the one it made for the copy, with what was copied into it, so nothing a failed
export copied stays outside the store, and the claim while it is still the empty directory the
export made: each is removed only while it is the directory the attempt made (``Made``), never one
another writer put in its place. An attempt that dies before it can remove them, its runner
killed, leaves them as its record names them: the export's next attempt removes that copy and
takes the empty claim for its own. One that dies between making them and recording them leaves
them unrecorded, and every retry refuses the destination as existing. One that dies after the
rename, before the runner publishes the stage, leaves a complete export at the destination, whose
``EXPORT.json`` says what it holds, while the export's instance ends failed: every retry refuses
the destination as existing. It leaves the directory it made for the copy too, empty, which the
rename changed: no retry can tell it from a directory another writer put there, so each leaves it
and records it (``export_partial_left``)."""

from __future__ import annotations

import errno
import os
import re
import shutil
import stat
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError, model_validator

from hls.pipeline.plan import Plan, PlanStore
from hls.pipeline.stage import Stage, StageContext, StageFault, register, stage_fault
from hls.spec.resources import ResourceShape
from hls.stages.finalize import (FAILED_DIR, PACKAGE_SCAN_SEP, QUARANTINE_FILE, FinalizeStage,
                                 check_built_arm, contestant_file, ending_failure, facing_secrets,
                                 finalize_params, index_key, instance_containers, judge, leak_fault, package_scan,
                                 package_scan_findings, run_scan, scan_lists, scan_manifests, shown_key,
                                 tree_files, unscanned_containers)
from hls.stages.provision import role_shape
from hls.stages.inputs import read_checked
from hls.stages.score import host_role_env
from hls.store.base import NotFound, StoreError, check_key
from hls.store.export_copy import COPY_FLAGS, DIR_FLAGS, Made, open_directory, read_copy_report
from hls.store.layout import GRADINGS_DIR, MANIFEST_FILE, InstanceLayout
from hls.store.manifest import Manifest, read_manifest
from hls.store.posix import PosixStore
from hls.substrate.secret_forms import SecretForm, form_text, registry_forms
from hls.substrate.secret_scan import named_in
from hls.util import validation_errors
from hls.verify.answers import answers_area

COPY = "export-copy"
#: The record the stage writes into the destination and as its output, and the output's name.
EXPORT_FILE, EXPORT_OUTPUT = "EXPORT.json", "export"
#: The record of the directories an export attempt made outside the store, in the attempt's
#: directory (``ClaimRecord``).
CLAIM_FILE = "destination-claim.json"
#: Why an export leaves behind a container of the submission no package scan covers.
UNSCANNED_WHY = "unscanned: no package scan covers it"
#: The entries of an instance's root that hold the orchestrators' and the runners' records.
WORKING_RECORDS = ("lease", "orchestrator", "allocations", "cancel-request.json")
#: The output finalize names its quarantine record by (``FinalizeStage``'s outputs).
QUARANTINE_OUTPUT = QUARANTINE_FILE.removesuffix(".json")
#: How the store names a temporary file it writes beside its destination
#: (``hls.store.posix._temp_beside``): the one kind of dot-named file an export leaves behind.
STORE_TEMP = re.compile(r"\.tmp-.+-[0-9a-f]{32}", re.DOTALL)
#: A redaction a scrubbed record holds in place of a secret's form (``SecretForm.marker``).
_REDACTION = re.compile(r"<redacted:[^<>]+>")


class ExportParams(BaseModel):
    """What the export copies out, and where to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The instances whose files leave, by their root keys: each finished and of the plan's task.
    #: A rollout's gradings are instances of their own, named beside it.
    instances: list[Annotated[str, AfterValidator(check_key)]] = Field(min_length=1)
    #: Whether the task's reference answers, ``oracle/<task>/``, leave too.
    oracle: bool = False
    #: Absolute path of the directory the files are copied into; it must not exist yet.
    destination: str

    @model_validator(mode="after")
    def _checked(self) -> ExportParams:
        if len(set(self.instances)) != len(self.instances):
            raise ValueError("an instance is named twice")
        if not Path(self.destination).is_absolute():
            raise ValueError(f"the destination {self.destination} is not an absolute path")
        return self


class CopiedFile(BaseModel):
    """A file the export copied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Its store key as a record shows it (``shown_key``), which is also its path under the
    #: destination.
    path: str
    #: The sha256 of the copy, read back from what landed.
    sha256: str
    #: Its size in bytes.
    size: int = Field(ge=0)


class LeftBehind(BaseModel):
    """A file of an instance the export did not copy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Its store key as a record shows it; for a file finalize quarantined, the key its record
    #: holds, scrubbed when it was written.
    path: str
    #: Why it stayed behind.
    why: str


class InstanceExport(BaseModel):
    """What the export did with one instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The instance's root key.
    instance: str
    #: How it ended (``Outcome.status``).
    outcome: str
    #: Whether its finalize published, so that its own scan read it too; a failed instance's
    #: files were first scanned by this export.
    finalized: bool
    #: Each file copied, in key order.
    copied: list[CopiedFile]
    #: Each file left behind, with why, in key order.
    left_behind: list[LeftBehind]


class ExportRecord(BaseModel):
    """``EXPORT.json``: what an export copied and what it left behind. Store keys and secrets'
    names, never a value or a host."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The export's own instance.
    export: str
    #: The hls commit the export ran from.
    hls_commit: str
    #: The names of the secrets the scan searched for, in every form.
    secrets_searched: list[str]
    #: Each instance, in the order the parameters name them.
    instances: list[InstanceExport]
    #: The task's reference answers copied; None when they were not asked for.
    oracle: list[CopiedFile] | None


class _Quarantine(BaseModel):
    """``quarantine.json`` as finalize writes it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Each file quarantined, by its key as a record shows it, with the labels of the forms found
    #: in it; none for a container no scan read inside.
    files: dict[str, list[str]]


@dataclass(frozen=True)
class Selection:
    """One instance's files, as ``select`` chose them."""

    #: The instance's latest plan.
    plan: Plan
    #: How it ended (``Outcome.status``).
    outcome: str
    #: Every manifest published under it, in plan order.
    published: tuple[Manifest, ...]
    #: Whether its finalize published.
    finalized: bool
    #: The store keys of the files that leave unless they are quarantined, in key order.
    leaving: tuple[str, ...]
    #: The store key of each file that stays behind, with why.
    behind: Mapping[str, str]
    #: The key of its index row, which is among ``leaving``; None for an instance that has none.
    row: str | None


@dataclass(frozen=True)
class Judgment:
    """What the export's own judgment found in one instance (``_Export.judge_all``), each by the
    store key it was found under with the labels of the forms found there, when nothing was the
    harness's leak."""

    #: The files a scan found something in, and each container a package scan's record says holds
    #: an area it names only by count (``area_holders``).
    files: Mapping[str, list[str]]
    #: The leaving files whose store keys hold a form (``path_forms``), which says why they stay
    #: behind.
    paths: Mapping[str, list[str]]
    #: What the package scans found, by the path in the submission image (an area's path ending
    #: with "/", or ``OUTSIDE_BUNDLE``).
    inside: Mapping[str, list[str]]


def behind_why(rest: str, counted: Mapping[str, int], latest: str) -> str | None:
    """Why the file at *rest*, its path under its instance's root, stays behind; None when it
    leaves. *counted* is each stage's counted attempt, *latest* the latest plan revision's path."""
    parts = rest.split("/")
    if STORE_TEMP.fullmatch(parts[-1]):
        return "a temporary file of the store"
    top = parts[0]
    if top == "plan":
        return None if rest == latest else "an earlier revision of the plan"
    if top == FAILED_DIR:
        return None
    if top in WORKING_RECORDS:
        return "a working record of the orchestrators or the runners"
    if top == "stages" and len(parts) == 3 and parts[2] == MANIFEST_FILE:
        return None
    if top == "stages" and len(parts) >= 5 and parts[2] == "attempts":
        if str(counted.get(parts[1])) == parts[3]:
            return None
        return f"attempt {parts[3]} of {parts[1]}, which the instance did not count"
    return "not a record an export publishes"


def select(store: PosixStore, plan: Plan) -> Selection:
    """What of *plan*'s instance leaves and what stays behind, with why (step 1 of the module
    docstring). Raise ValueError for an instance that has not ended."""
    layout, root = plan.layout, plan.instance
    if plan.outcome is None:
        raise ValueError(f"{root} has not ended")
    order = {s.name: i for i, s in enumerate(plan.stages)}
    names = sorted(store.list_children(f"{root}/stages"), key=lambda n: (order.get(n, len(order)), n))
    published = tuple(m for m, _ in filter(None, (read_manifest(store, layout, n) for n in names)))
    counted = {m.stage: m.attempt for m in published}
    if plan.outcome.status == "failed":
        stage = ending_failure(plan).stage
        tried = layout.attempts(store, stage)
        if tried:
            counted.setdefault(stage, tried[-1])
    latest = layout.plan_revision(plan.revision)[len(root) + 1:]
    files, others = tree_files(store.path(root), skip=store.path(f"{root}/{GRADINGS_DIR}"))
    leaving: list[str] = []
    behind: dict[str, str] = {}
    for path in files:
        key = path.relative_to(store.root).as_posix()
        why = behind_why(key[len(root) + 1:], counted, latest)
        if why is None:
            leaving.append(key)
        else:
            behind[key] = why
    for path in others:
        behind[path.relative_to(store.root).as_posix()] = "not a regular file: no export follows a link"
    row = None
    params = finalize_params(plan)
    if params is not None:
        try:
            index: str | None = index_key(root, params.kind)[0]
        except ValueError:
            index = None
        if index is not None and store.exists(index):
            row = index
            leaving.append(index)
    finalize_stages = {s.name for s in plan.stages if s.kind == FinalizeStage.kind}
    return Selection(plan=plan, outcome=plan.outcome.status, published=published,
                     finalized=any(m.stage in finalize_stages for m in published), leaving=tuple(sorted(leaving)),
                     behind=dict(sorted(behind.items())), row=row)


def finalize_quarantine(store: PosixStore, plan: Plan, published: Sequence[Manifest]) -> dict[str, list[str]]:
    """The files *plan*'s finalize quarantined, by their keys as its record shows them, with the
    labels of the forms found in each, none for a container no scan read inside
    (``quarantine.json``, read against its manifest); none when it quarantined none or never
    published. Raise an infra fault that is not retried for a record that is not the published
    one or does not parse."""
    names = {s.name for s in plan.stages if s.kind == FinalizeStage.kind}
    manifest = next((m for m in published if m.stage in names), None)
    if manifest is None or QUARANTINE_OUTPUT not in manifest.outputs:
        return {}
    ref = manifest.outputs[QUARANTINE_OUTPUT]
    try:
        return _Quarantine.model_validate_json(read_checked(store, ref)).files
    except ValidationError as e:
        raise StageFault.infra(f"{ref.key} is not a quarantine record: {validation_errors(e)}",
                               retryable=False) from None


def inside_image(key: str, records: Collection[str]) -> str | None:
    """The path in the submission image that *key*, a name a package scan's finding was given
    (``package_scan_findings``), stands for: what follows the record's key and
    ``PACKAGE_SCAN_SEP``, when the key is one of *records*, the package-scan records' keys as a
    record shows them; None for a key that names a file of the store, as a container that holds an
    area a record names only by count."""
    record, sep, path = key.partition(PACKAGE_SCAN_SEP)
    return path if sep and record in records else None


def scan_records(store: PosixStore, manifests: Iterable[Manifest]) -> set[str]:
    """The keys of the package-scan records (``package-scan.json``) of *manifests*, as a record
    shows them (``shown_key``), each checked against its manifest (``package_scan``)."""
    return {shown_key(found[0]) for found in (package_scan(store, m) for m in manifests) if found is not None}


def owner_of(key: str, selections: Sequence[Selection]) -> Selection | None:
    """The selection of the instance whose files hold *key*: the one whose root is the longest
    prefix of it, since a grading lies under its rollout's root, or whose index row it is; None
    for a key of no instance, such as the answers area's."""
    for s in selections:
        if s.row == key:
            return s
    owners = [s for s in selections if key.startswith(f"{s.plan.instance}/")]
    return max(owners, key=lambda s: len(s.plan.instance), default=None)


def path_forms(key: str, forms: Sequence[SecretForm]) -> list[str]:
    """The labels of the secrets' *forms* that *key*, a file's store key, holds (``named_in``), as
    the scan role finds them in a path: a path a contestant chose, such as an agent's log named
    after its key, would publish what it holds as surely as the file's bytes would."""
    return sorted(named_in(Path(key), forms))


class RecordMatcher:
    """The keys a key of a scrubbed record stands for: the key itself, each redaction in it
    (``<redacted:NAME>``, ``<redacted:NAME.host>``) standing for any text of one character or
    more, since the form it replaced may not be one the secrets file holds now. A key is matched in
    one pass, piece after piece: the record's literal pieces between its redactions are found in
    order, each as early as it can be, which leaves the most room for the rest, so no key a
    contestant names can make the match backtrack."""

    def __init__(self, key: str) -> None:
        #: The record's key.
        self.key = key
        self._pieces = _REDACTION.split(key)

    def matches(self, text: str) -> bool:
        """Whether *text*, a key as a record shows it (``shown_key``), is one the record's key
        stands for."""
        if len(self._pieces) == 1:
            return text == self.key
        first, *middle, last = self._pieces
        end = len(text) - len(last)
        if not (text.startswith(first) and text.endswith(last)):
            return False
        at = len(first)
        for piece in middle:
            # One character at least for the redaction before the piece, and for the one after.
            found = text.find(piece, at + 1, end - 1)
            if found < 0:
                return False
            at = found + len(piece)
        return at < end


def _held(labels: Iterable[str]) -> str:
    """The forms *labels* name, as a record says them (``form_text``)."""
    return ", ".join(form_text(label) for label in sorted(labels))


class ClaimRecord(BaseModel):
    """``CLAIM_FILE``: the directories an export attempt made outside the store, written into its
    attempt's directory before anything is copied, so the export's next attempt can tell them from
    another writer's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The destination, the empty directory the attempt claimed (``claim_destination``) or took
    #: from an earlier attempt.
    claim: Made
    #: The directory beside the destination it made for its copy (``make_partial``).
    partial: Made


def _entry(parent: int, name: str) -> os.stat_result | None:
    """What ``lstat`` says of *name* in the directory open at *parent*; None when nothing is
    there."""
    try:
        return os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _open_at(parent: int, name: str) -> int | None:
    """A descriptor of the directory *name* in the directory open at *parent*, never through a link;
    None when nothing is there, or a link or anything but a directory is."""
    try:
        return os.open(name, DIR_FLAGS, dir_fd=parent)
    except FileNotFoundError:
        return None
    except OSError as e:
        if e.errno in (errno.ENOTDIR, errno.ELOOP):
            return None
        raise


def claim_destination(parent: int, name: str) -> Made:
    """Claim the destination *name* in the directory open at *parent* by making it, an empty
    directory, where nothing is (``mkdir`` fails for anything there, a link included), and return
    what tells it from a directory another writer could put in its place. Raise FileExistsError
    when something is there."""
    os.mkdir(name, 0o755, dir_fd=parent)
    return Made.of(os.stat(name, dir_fd=parent, follow_symlinks=False))


def _is_claim(parent: int, name: str, claim: Made) -> bool:
    """Whether *name* in the directory open at *parent* is the directory *claim*."""
    found = _entry(parent, name)
    return found is not None and stat.S_ISDIR(found.st_mode) and Made.of(found) == claim


def empty_claim(parent: int, name: str, claim: Made) -> bool:
    """Whether *name* in the directory open at *parent* is still *claim* (``claim_destination``)
    and empty, so an attempt of the export that made it may copy for it."""
    if not _is_claim(parent, name, claim):
        return False
    fd = _open_at(parent, name)
    if fd is None:
        return False
    try:
        with os.scandir(fd) as entries:
            return Made.of(os.fstat(fd)) == claim and next(entries, None) is None
    finally:
        os.close(fd)


def release_claim(parent: int, name: str, claim: Made) -> bool:
    """Remove *name* in the directory open at *parent* when it is still *claim*
    (``claim_destination``) and empty, and return whether it was removed: a directory another
    writer filled or put in its place, a file, or nothing at all, stays as it is. Raise OSError
    for a removal that fails otherwise."""
    if not _is_claim(parent, name, claim):
        return False
    try:
        os.rmdir(name, dir_fd=parent)
    except OSError as e:
        if e.errno in (errno.ENOTEMPTY, errno.EEXIST):
            return False
        raise
    return True


@dataclass
class Partial:
    """The directory an export attempt made beside the destination for its copy, and in it the
    directory the copy goes into, each held open: a descriptor holds its directory's inode while
    it is open, so no directory another writer puts in its place is taken for it."""

    #: Its name in the destination's directory.
    name: str
    #: A descriptor of it.
    fd: int
    #: What tells it from a directory another writer could put in its place, once the copy's
    #: directory is made in it; nothing the attempt does before its rename changes it.
    made: Made
    #: The name of the copy's directory in it: the destination's own.
    inner: str
    #: A descriptor of the copy's directory.
    copy: int
    #: The copy's directory as it was made, before anything was copied into it: the copy role
    #: copies into no other (``--to-made``).
    copy_made: Made

    def close(self) -> None:
        """Close both descriptors."""
        try:
            os.close(self.copy)
        finally:
            os.close(self.fd)


def make_partial(parent: int, name: str, inner: str) -> Partial:
    """Make *name* where nothing is in the directory open at *parent*, a directory for its owner
    alone, and in it the directory *inner* the copy goes into, each through a descriptor of the
    directory it is made in, and return both held open (``Partial``). Raise FileExistsError when
    something is at *name*."""
    os.mkdir(name, 0o700, dir_fd=parent)
    fd = os.open(name, DIR_FLAGS, dir_fd=parent)
    try:
        os.mkdir(inner, 0o755, dir_fd=fd)
        copy = os.open(inner, DIR_FLAGS, dir_fd=fd)
    except BaseException:
        try:
            remove_open(parent, name, fd)
        finally:
            os.close(fd)
        raise
    return Partial(name=name, fd=fd, made=Made.of(os.fstat(fd)), inner=inner, copy=copy,
                   copy_made=Made.of(os.fstat(copy)))


def _empty(fd: int) -> None:
    """Remove every entry of the directory open at *fd*, through descriptors, never following a
    link."""
    with os.scandir(fd) as found:
        entries = list(found)
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            shutil.rmtree(entry.name, dir_fd=fd)
        else:
            os.unlink(entry.name, dir_fd=fd)


def remove_open(parent: int, name: str, fd: int) -> bool:
    """Remove everything in the directory open at *fd*, then the directory itself while *name* in
    the directory open at *parent* still names it, and return whether it was removed. A descriptor
    holds its directory's inode while it is open, so no other directory at *name* compares
    alike."""
    _empty(fd)
    held, there = os.fstat(fd), _entry(parent, name)
    if there is None or (there.st_dev, there.st_ino) != (held.st_dev, held.st_ino):
        return False
    os.rmdir(name, dir_fd=parent)
    return True


def remove_made(parent: int, name: str, made: Made) -> bool:
    """Remove *name* in the directory open at *parent*, with everything in it, when it is the
    directory *made* (``remove_open``), and return whether it was removed: anything else there, or
    nothing, stays as it is."""
    fd = _open_at(parent, name)
    if fd is None:
        return False
    try:
        return Made.of(os.fstat(fd)) == made and remove_open(parent, name, fd)
    finally:
        os.close(fd)


class _Export:
    """One export attempt."""

    def __init__(self, ctx: StageContext) -> None:
        if ctx.scratch is None:
            raise RuntimeError("the export stage needs node-local scratch")
        self.ctx = ctx
        self.params: ExportParams = ctx.params

    def selection(self, instance: str) -> Selection:
        """What of *instance* leaves (``select``), once a grading's ``built_arm`` is shown to be the
        arm that built the package it graded (``check_built_arm``), a failed grading's too. Raise
        an infra fault that is not retried for an instance with no plan, of another task, still
        running, whose records cannot be read, or whose arm is not the one that built its
        package."""
        ctx = self.ctx
        try:
            plan = PlanStore(ctx.store, InstanceLayout(instance)).latest()
            if plan is None:
                raise StageFault.infra(f"instance {instance} has no plan in the store", retryable=False)
            if plan.task != ctx.plan.task:
                raise StageFault.infra(f"instance {instance} is of task {plan.task}, not {ctx.plan.task}",
                                       retryable=False)
            if plan.outcome is None:
                raise StageFault.infra(f"instance {instance} has not ended", retryable=False)
            selection = select(ctx.store, plan)
            check_built_arm(ctx.store, plan, selection.published)
            return selection
        except (ValueError, StoreError) as e:
            reason = f"the records of {instance} cannot be read: {type(e).__name__}: {e}"
            raise StageFault.infra(ctx.scrubber.scrub(reason), retryable=False) from e

    def oracle_files(self) -> list[str]:
        """The store keys of the task's reference answers but the store's temporary files. Raise
        an infra fault that is not retried when the store holds none."""
        ctx = self.ctx
        area = answers_area(ctx.plan.task)
        if not ctx.store.path(area).is_dir():
            raise StageFault.infra(f"{ctx.plan.task} has no reference answers in the store ({area})", retryable=False)
        files, _ = tree_files(ctx.store.path(area))
        return [p.relative_to(ctx.store.root).as_posix() for p in files if not STORE_TEMP.fullmatch(p.name)]

    def judge_all(self, selections: Sequence[Selection], hits: Sequence[tuple[str, str]]) -> list[Judgment]:
        """Judge every form the scan found in a file or its path (*hits*) and every package scan's
        finding, instance by instance, by finalize's rule (``judge``) with the instance's own plan;
        whatever is found in a file of no instance, such as the answers area's, is a leak. Return
        each instance's ``Judgment``, with the forms each leaving file's store key holds
        (``path_forms``) for why it stays behind. Raise the fault the leaks are, when there is
        one."""
        ctx = self.ctx
        forms = registry_forms(ctx.secrets)
        found: dict[str, list[tuple[str, str, bool]]] = {s.plan.instance: [] for s in selections}
        ours: list[tuple[str, str]] = []
        for key, label in hits:
            owner = owner_of(key, selections)
            if owner is None:
                ours.append((key, label))
            else:
                found[owner.plan.instance].append((key, label, contestant_file(owner.plan, key)))
        judged = []
        for s in selections:
            facing = facing_secrets(s.plan, ctx.secrets)
            manifests = scan_manifests(ctx.store, s.plan, s.published)
            records = scan_records(ctx.store, manifests)
            scanned = package_scan_findings(ctx.store, manifests)
            leaks, held = judge([*found[s.plan.instance], *((key, label, True) for key, label in scanned)], facing)
            ours += leaks
            paths = {key: labels for key in s.leaving if (labels := path_forms(key, forms))}
            files: dict[str, list[str]] = {}
            inside: dict[str, list[str]] = {}
            for key, labels in held.items():
                path = inside_image(shown_key(key), records)
                if path is None:
                    files[key] = labels
                else:
                    inside[path] = sorted({*inside.get(path, ()), *labels})
            judged.append(Judgment(files=files, paths=paths, inside=inside))
        if ours:
            raise leak_fault(sorted(ours), ctx.scrubber)
        return judged

    def outcome_of(self, selection: Selection,
                   judgment: Judgment) -> tuple[list[str], dict[str, str], dict[str, str]]:
        """What of *selection* is copied; every file left behind with why: what ``select`` left, what
        the export's *judgment* quarantined for what a file holds or what its key holds, what the
        instance's finalize quarantined (an entry with no secret marks a container no scan read
        inside), every container the published scans did not read inside for every secret the
        registry holds now in every form (``unscanned_containers``), and every container
        (``instance_containers``) when a package's scan found something in a file of it, an area
        past the scan's bound shown as ``every file of <area> (over_cap)`` and the objects of a
        bundle outside the store by ``OUTSIDE_BUNDLE``'s text; and, for each file a key of
        finalize's record matched, that key, which is how ``EXPORT.json`` shows the file. Finalize
        writes ``quarantine.json`` scrubbed, and the secrets file may have changed since, so each
        candidate is compared as a record shows it (``shown_key``) with each record key, whose
        redactions stand for any text (``RecordMatcher``). Two paths that compare alike are both
        left behind, the safe direction."""
        ctx = self.ctx
        manifests = scan_manifests(ctx.store, selection.plan, selection.published)
        recorded = finalize_quarantine(ctx.store, selection.plan, selection.published)
        records = scan_records(ctx.store, manifests)
        inside = {path: set(labels) for path, labels in judgment.inside.items()}
        for key, labels in recorded.items():
            path = inside_image(key, records)
            if path is not None:
                inside.setdefault(path, set()).update(labels)
        where = ", ".join(f"every file of {path} (over_cap)" if path.endswith("/") else path
                          for path in sorted(inside))
        clause = (f"a container of the submission, where the package's scan found "
                  f"{_held(set().union(*inside.values()))} in {where}") if inside else None
        boxes = set(instance_containers(ctx.store, selection.plan, manifests))
        unread = set(unscanned_containers(ctx.store, selection.plan, manifests, ctx.secrets))
        matchers = [(RecordMatcher(record), labels) for record, labels in recorded.items()]
        behind = dict(selection.behind)
        shown_as: dict[str, str] = {}
        copied = []
        for key in selection.leaving:
            shown = shown_key(key)
            held = set(judgment.files.get(key, ()))
            unscanned = key in unread
            for matcher, labels in matchers:
                if matcher.matches(shown):
                    held |= set(labels)
                    unscanned = unscanned or not labels
                    shown_as.setdefault(key, matcher.key)
            why = [f"holds {_held(held)}"] if held else []
            if key in judgment.paths:
                why.append(f"its path holds {_held(judgment.paths[key])}")
            if unscanned:
                why.append(UNSCANNED_WHY)
            container = [clause] if clause is not None and key in boxes else []
            if why:
                behind[key] = f"quarantined: {'; '.join(why + container)}"
            elif container:
                behind[key] = container[0]
            else:
                copied.append(key)
        return copied, behind, shown_as

    def copy(self, keys: Sequence[str], into: Path, made: Made) -> dict[str, tuple[str, int]]:
        """Copy *keys* through the copy role into the directory at *into*, which must be the one
        *made* names, and return what it reported, each key's copy's sha256 and size. Raise the
        fault a failure of the role is, and an infra fault that is retried when the report does not
        name exactly the files the role was given."""
        ctx = self.ctx
        folder = ctx.role_dir(COPY)
        listing, report = folder / "copy-list", folder / "copied.jsonl"
        listing.write_bytes(b"".join(os.fsencode(key) + b"\0" for key in keys))
        argv = [str(ctx.plan.toolchain.python), "-m", "hls.store.export_copy", "--root", str(ctx.store.root),
                "--list", str(listing), "--to", str(into), "--to-made", f"{made.inode}:{made.ctime_ns}",
                "--report", str(report)]
        shape = role_shape(ctx.stage.shape, ctx.plan.settings.run.runner_memory_gib)
        ctx.start_role(COPY, argv, shape, env=host_role_env(ctx), owner="harness")
        [outcome] = ctx.wait_roles(COPY)
        fault = stage_fault([outcome])
        if fault is not None:
            raise StageFault(fault)
        copies = read_copy_report(report)
        if copies is None or set(copies) != set(keys):
            got = 0 if copies is None else len(set(copies) & set(keys))
            raise StageFault.infra(f"the copy role reported {got} of the {len(keys)} files it was given",
                                   retryable=True)
        return copies

    def check_copies(self, selections: Sequence[Selection], copies: Mapping[str, tuple[str, int]],
                     read: Mapping[str, str]) -> None:
        """Hold each of *copies* to what it was copied from: the digest the scan reported for the
        file it searched (*read*), and for a published output the digest its manifest recorded.
        Raise an infra fault that is not retried for a copy of a published output whose digest is
        not its manifest's, or of a file the scan did not read; one that is retried for a copy whose
        digest is not the scan's, which a retry scans and copies anew."""
        scrub = self.ctx.scrubber.scrub
        recorded = {ref.key: ref.sha256 for s in selections for m in s.published for ref in m.outputs.values()}
        for key, (sha256, _) in sorted(copies.items()):
            if key in recorded and recorded[key] != sha256:
                raise StageFault.infra(scrub(f"the store's {shown_key(key)} holds other bytes than its manifest "
                                       "recorded"), retryable=False)
            scanned = read.get(key)
            if scanned is None:
                raise StageFault.infra(scrub(f"the scan did not read the store's {shown_key(key)}, which the export "
                                       "would copy"), retryable=False)
            if scanned != sha256:
                raise StageFault.infra(scrub(f"the store's {shown_key(key)} is not the file the scan read: it changed "
                                       "since"), retryable=True)

    def write(self, selections: Sequence[Selection], outcomes: Sequence[tuple[list[str], dict[str, str],
                                                                              dict[str, str]]],
              copies: Mapping[str, tuple[str, int]], oracle: Sequence[str] | None, into: int) -> str:
        """Write ``EXPORT.json`` (``ExportRecord``), scrubbed, into the copy's directory, open at
        *into*, and as the stage's output, and return the output's key."""
        ctx = self.ctx

        def files(keys: Iterable[str]) -> list[CopiedFile]:
            return [CopiedFile(path=shown_key(k), sha256=copies[k][0], size=copies[k][1]) for k in keys]

        def left(behind: Mapping[str, str], shown_as: Mapping[str, str]) -> list[LeftBehind]:
            return [LeftBehind(path=shown_as.get(k) or shown_key(k), why=why)
                    for k, why in sorted(behind.items())]

        exported = [InstanceExport(instance=s.plan.instance, outcome=s.outcome, finalized=s.finalized,
                                   copied=files(copied), left_behind=left(behind, shown_as))
                    for s, (copied, behind, shown_as) in zip(selections, outcomes, strict=True)]
        record = ExportRecord(export=ctx.plan.instance, hls_commit=ctx.plan.code.hls_commit,
                              secrets_searched=sorted(ctx.secrets), instances=exported,
                              oracle=files(oracle) if oracle is not None else None)
        data = ctx.scrubber.scrub(record.model_dump_json(indent=2)).encode()
        with os.fdopen(os.open(EXPORT_FILE, COPY_FLAGS, 0o644, dir_fd=into), "wb") as written:
            written.write(data)
        return ctx.write_output(EXPORT_FILE, data)

    def run(self) -> dict[str, str]:
        ctx, p = self.ctx, self.params
        if not ctx.secret_registry:
            raise StageFault.infra("the export runs only with the secrets registry, which the runner did not hand it: "
                                   "it would copy files no scan had read", retryable=False)
        if not ctx.secrets:
            raise StageFault.infra("the secrets file holds no secret: an export searches every file for the values "
                                   "of the secrets file, and with none it would copy files no scan had read",
                                   retryable=False)
        destination = Path(p.destination)
        try:
            parent = open_directory(destination.parent)
        except OSError as e:
            raise StageFault.infra(f"the destination's directory {destination.parent} cannot be opened without "
                                   f"following a link: {e.strerror}", retryable=False) from e
        try:
            return self.export(parent, destination)
        finally:
            os.close(parent)

    def export(self, parent: int, destination: Path) -> dict[str, str]:
        """Export into *destination*, whose directory is open at *parent*, making, checking,
        renaming and removing every directory beside it through that descriptor (the module
        docstring's steps)."""
        ctx, p = self.ctx, self.params
        earlier = self.earlier_claims()
        for attempt, record in earlier:
            self.discard(parent, self.partial_name(attempt), record.partial)
        adopted = next((record.claim for _, record in earlier
                        if empty_claim(parent, destination.name, record.claim)), None)
        if adopted is None and _entry(parent, destination.name) is not None:
            raise StageFault.infra(f"the destination {destination} exists", retryable=False)
        selections = [self.selection(instance) for instance in p.instances]
        oracle = self.oracle_files() if p.oracle else None
        listed: list[Path] = []
        raw: list[Path] = []
        for s in selections:
            more, stored = scan_lists(ctx.store, s.plan, s.published, skip_gradings=True)
            listed += [*more, *([ctx.store.path(s.row)] if s.row is not None else [])]
            raw += stored
        if oracle is not None:
            listed.append(ctx.store.path(answers_area(ctx.plan.task)))
        scanned = run_scan(ctx, listed, raw)
        judged = self.judge_all(selections, scanned.found)
        outcomes = [self.outcome_of(s, j) for s, j in zip(selections, judged, strict=True)]
        keys = [key for copied, _, _ in outcomes for key in copied] + list(oracle or [])
        name = self.partial_name(ctx.attempt)
        try:
            partial = make_partial(parent, name, destination.name)
        except FileExistsError as e:
            raise StageFault.infra(f"{destination.parent / name}, where this attempt copies to, exists already",
                                   retryable=False) from e
        try:
            claim = self.claim(parent, destination, adopted)
            try:
                self.record(ClaimRecord(claim=claim, partial=partial.made))
                copies = self.copy(keys, destination.parent / name / partial.inner, partial.copy_made)
                self.check_copies(selections, copies, scanned.read)
                out = self.write(selections, outcomes, copies, oracle, partial.copy)
                self.place(parent, partial, destination)
            except BaseException:
                self.release(parent, destination, claim)
                raise
        finally:
            try:
                self.discard_open(parent, partial)
            finally:
                partial.close()
        ctx.events.emit("exported", instances=len(selections), files=len(copies), destination=str(destination))
        return {EXPORT_OUTPUT: out}

    def partial_name(self, attempt: int) -> str:
        """The name of the directory beside the destination that attempt *attempt* of this export
        makes for its copy (``make_partial``)."""
        export_id = self.ctx.plan.instance.rsplit("/", 1)[1]
        return f".{Path(self.params.destination).name}.{export_id}.{attempt}"

    def earlier_claims(self) -> list[tuple[int, ClaimRecord]]:
        """The record of the directories each earlier attempt of this export made
        (``ClaimRecord``), with its attempt's number. Raise an infra fault that is not retried for a
        record that cannot be read."""
        ctx = self.ctx
        layout, stage = ctx.plan.layout, ctx.stage.name
        found = []
        for attempt in layout.attempts(ctx.store, stage):
            if attempt >= ctx.attempt:
                continue
            key = layout.attempt_file(stage, attempt, CLAIM_FILE)
            try:
                found.append((attempt, ClaimRecord.model_validate_json(ctx.store.read_bytes(key))))
            except NotFound:
                continue
            except ValidationError as e:
                raise StageFault.infra(f"{key} is not a record of the directories an export attempt made: "
                                       f"{validation_errors(e)}", retryable=False) from None
        return found

    def claim(self, parent: int, destination: Path, adopted: Made | None) -> Made:
        """The claim this attempt copies for, *destination* in the directory open at *parent*:
        *adopted*, the claim an earlier attempt of this export made, while it is still that empty
        directory (``empty_claim``), else a new one (``claim_destination``). Raise an infra fault
        that is not retried when anything else is at *destination*."""
        if adopted is not None and empty_claim(parent, destination.name, adopted):
            self.ctx.events.emit("export_claim_taken", path=str(destination))
            return adopted
        try:
            return claim_destination(parent, destination.name)
        except FileExistsError as e:
            raise StageFault.infra(f"the destination {destination} exists", retryable=False) from e

    def record(self, record: ClaimRecord) -> None:
        """Write *record*, the directories this attempt made, into its attempt's directory
        (``CLAIM_FILE``)."""
        ctx = self.ctx
        ctx.store.write_bytes(ctx.plan.layout.attempt_file(ctx.stage.name, ctx.attempt, CLAIM_FILE),
                              record.model_dump_json().encode())

    def place(self, parent: int, partial: Partial, destination: Path) -> None:
        """Rename the finished copy in *partial* onto *destination*, the empty directory the export
        claimed in the directory open at *parent*, through both descriptors. Raise an infra fault
        that is not retried when another writer put files in the claim or replaced it with a file,
        which both stay as they are, and for any other failure of the rename."""
        # A plain rename is safe here: rename(2) replaces a directory only when it is empty, so it
        # replaces the claim and fails on anything another writer put there (EEXIST or ENOTEMPTY,
        # ENOTDIR for a file), never overwriting it. renameat2 with RENAME_NOREPLACE is not used:
        # GPFS refuses that flag with EINVAL even onto a free name, where xfs and ext4 accept it; a
        # plain rename on GPFS replaces an empty claim and refuses a filled one with EEXIST.
        try:
            os.rename(partial.inner, destination.name, src_dir_fd=partial.fd, dst_dir_fd=parent)
        except OSError as e:
            if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise StageFault.infra(f"another writer put files in the destination {destination} after the export "
                                       "claimed it; they are kept, and nothing was renamed", retryable=False) from e
            if e.errno == errno.ENOTDIR:
                raise StageFault.infra(f"another writer replaced the destination {destination} with a file after the "
                                       "export claimed it; it is kept, and nothing was renamed", retryable=False) from e
            raise StageFault.infra(f"the export could not become {destination}: {e.strerror}", retryable=False) from e

    def release(self, parent: int, destination: Path, claim: Made) -> None:
        """Remove *destination*, in the directory open at *parent*, when it is still *claim*, the
        empty directory this attempt made (``release_claim``). A claim the export cannot remove is
        recorded (``export_claim_left``)."""
        try:
            release_claim(parent, destination.name, claim)
        except OSError as e:
            self.ctx.events.emit("export_claim_left", path=str(destination), error=type(e).__name__)

    def discard_open(self, parent: int, partial: Partial) -> None:
        """Remove the directory this attempt made for its copy, *partial*, with whatever is in it
        (``remove_open``), so nothing a failed export copied, whose bytes may not be the ones the
        scan read, stays outside the store. The descriptor it holds tells it from any directory at
        its name that is not it, which is another writer's (``_discard``)."""
        self._discard(parent, partial.name, lambda: remove_open(parent, partial.name, partial.fd),
                      "another writer's")

    def discard(self, parent: int, name: str, made: Made) -> None:
        """Remove the directory *name* an earlier attempt of this export made for its copy, while it
        is the one *made* says it made (``remove_made``, ``_discard``). One that is not is another
        writer's, or the attempt's own once the rename of its copy onto the destination changed
        it: the two look alike."""
        self._discard(parent, name, lambda: remove_made(parent, name, made),
                      "not the directory its attempt recorded: another writer's, or the attempt's own after its "
                      "copy was renamed onto the destination")

    def _discard(self, parent: int, name: str, remove: Callable[[], bool], why: str) -> None:
        """Run *remove*, which removes *name* in the directory open at *parent* when it is the
        directory an attempt made and says whether it did. A removal that fails, and a directory
        left at *name*, with *why* it was, are recorded (``export_partial_left``), leaving the
        failure that ended the attempt, if any, to be raised."""
        path = str(Path(self.params.destination).parent / name)
        try:
            removed = remove()
        except OSError as e:
            self.ctx.events.emit("export_partial_left", path=path, error=type(e).__name__)
            return
        if not removed and _entry(parent, name) is not None:
            self.ctx.events.emit("export_partial_left", path=path, why=why)


@register
class ExportStage(Stage):
    """See the module docstring."""

    kind: ClassVar[str] = "export"
    params_model: ClassVar[type[BaseModel]] = ExportParams
    # It copies files out of the store and keeps no copy in its attempts: no file of them holds a
    # contestant's bytes compressed (``Stage.compressed_files``, ``Stage.unpublished_containers``,
    # ``Stage.published_dirs``).
    compressed_files: ClassVar[tuple[str, ...]] = ()
    unpublished_containers: ClassVar[tuple[str, ...]] = ()
    published_dirs: ClassVar[tuple[str, ...]] = ()
    #: The runner loads the secrets file for this stage although it has no grants: the scan
    #: searches for every value of it.
    needs_secret_registry: ClassVar[bool] = True

    @classmethod
    def max_roles(cls, params: BaseModel | None) -> int:
        """The most roles one attempt starts: the scan, then the copy."""
        return 2

    @classmethod
    def check(cls, shape: ResourceShape, params: BaseModel, *, runner_memory_gib: int) -> None:
        """The stage reads and copies files: it asks for no GPU, and each of its roles, one after
        the other, has the stage's shape less the runner's memory (``role_shape``)."""
        assert isinstance(params, ExportParams)
        if shape.gpus:
            raise ValueError(f"the export stage reads and copies files and uses no GPU; its shape asks for "
                             f"{shape.gpus}")
        role_shape(shape, runner_memory_gib)

    def run(self, ctx: StageContext) -> dict[str, str]:
        return _Export(ctx).run()
