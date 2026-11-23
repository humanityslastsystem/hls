"""Plan-time checks of the scored commands.

A scored command's instance ends with a scored result or a measurement: ``hls run``
(its probe and preflight-only builds excepted), ``hls grade``, ``hls baseline-credit`` and
``hls calibrate``. Each calls these checks on the login node once it has imported the task
module and before it builds the plan, so a refusal is a ``PlanError``: one line on stderr and
exit 1, with nothing submitted. The checks run read-only git commands in the checkout and read
the store; none writes anything or loads a model.

- ``check_committed_tree``: the checkout and its harbor submodule hold no uncommitted change,
  and harbor is at the commit the checkout's HEAD pins. The plan records the commits,
  and a scored result names code anyone can check out again.
- ``check_defaults_committed``: a default baseline credit or calibration the plan pins
  (``hls.measured``) is the file committed at HEAD, byte for byte, by the blob id the plan pins
  it under, the id the score role records for the credit it scored against.
- ``check_upload_folder``: each folder a build uploads into its sandbox (``upload_dir``), which
  a commit may not hold, resolves to a directory of the checkout with no link in it, no two hold
  the same path, their merged view has the tree digest the task pins (``upload_sha256``), and
  they land under ``/task`` or ``/workspace`` (``upload_dest``). They are the rules the build applies when it starts
  (``hls.agent.uploads.checked_upload``, ``upload_destination``), applied before an allocation
  is asked for.
- ``check_skills_folder``: the folder of skills a build copies into its sandbox
  (``skills_dir``) resolves to a directory of the checkout that is one skill or a folder of
  skills, by the rule the build applies when it starts (``hls.agent.uploads.task_skills``).
- ``check_oracle_coverage``: a grading of a task whose verifiers keep their reference answers
  in the store's answers area finds an answer there to every prompt its verify will ask.
  The verify role computes an answer the area lacks inside the grading, so each such
  grading would get its own oracle vintage
  and pay the reference's time again. The prompts are the test rows the drive sends, a
  split of hundreds of megabytes, so the check reads none: it reads the question set a
  completed oracle fill recorded (``hls.verify.answers.QuestionSet``), every prompt
  ``hls.verify.fill.corpus_prompts`` builds from the same rows entry and the same declared
  shaping, as a tenant, a request id and a digest. Each question is looked up as the verify
  role looks a prompt up (``StoreAnswers.answered``, under ``unanswered``), in the area of the
  plan's task id, under the key of the GPU kind and the reference image the verify role runs
  on. ``hls.verify.run.may_deploy`` asks the same of a drive's trace, whose prompts are some or
  all of those, so where every engine of a task is a caching reference asked only for corpus
  answers, a task that passes this check loads no engine in any of its gradings' verifies.
  Another engine loads whatever the area holds, which this check does not judge. Answers no
  completed fill recorded, such as a verify's own, are no evidence: without the set nothing says
  they cover the rows.
- ``arm_lines``: the arm a build runs, as the plan resolved it, for the command to print before
  it submits anything.

Every refusal names files, instances and secrets' names, never a value."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from hls.agent.toolchain_references.assemble import ASSEMBLE_COMMAND, LIBRARY_DIR
from hls.agent.uploads import UploadRefused, checked_upload, task_skills, upload_destination
from hls.arms import ArmRecord
from hls.measured import MeasuredFiles
from hls.pipeline.checkout import READ_GIT_TIMEOUT_SEC, CodeState
from hls.pipeline.planner import PlanError
from hls.provision.role import Refused, resolve
from hls.stages.verify import writer_instance
from hls.store.base import StoreError
from hls.store.posix import PosixStore
from hls.taskref import TaskRef, task_dir
from hls.util import validation_errors
from hls.verify.answers import StoreAnswers, answer_key, answers_area, questions_name
from hls.verify.fill import caching_references, question_scopes, question_shape

#: The writer name the check reads the answers area under. It adds no file: only
#: ``StoreAnswers.bank`` writes, and the check never calls it.
_READER = "preflight"


def _git(checkout: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    """How a read-only git command in *checkout* ended. ``--no-optional-locks`` keeps git from
    refreshing the index, so nothing in the checkout is written. Raise PlanError when git does
    not finish within ``READ_GIT_TIMEOUT_SEC``."""
    try:
        return subprocess.run(["git", "--no-optional-locks", "-C", str(checkout), *args], capture_output=True,
                              timeout=READ_GIT_TIMEOUT_SEC, check=False)
    except subprocess.TimeoutExpired:
        raise PlanError(f"git {args[0]} in {checkout} did not finish in {READ_GIT_TIMEOUT_SEC} s") from None


def _head_entry(root: Path, path: str) -> tuple[str, str] | None:
    """The type and object id of the entry at *path*, from the checkout *root*, in HEAD's tree
    (``blob`` for a file, ``commit`` for a submodule's pointer); None when HEAD holds no entry
    there. A ``HEAD:<path>`` revision cannot say which: git reads everything after the colon as
    the path, so a ``^{blob}`` suffix names no entry."""
    listed = _git(root, "ls-tree", "-z", "HEAD", "--", path)
    entries = [e for e in listed.stdout.split(b"\0") if e] if listed.returncode == 0 else []
    if len(entries) != 1:
        return None
    meta, _, name = entries[0].partition(b"\t")
    _, kind, obj = meta.decode().split(" ")
    return (kind, obj) if name.decode() == path else None


def _head_blob(root: Path, path: str) -> str | None:
    """The id of the file at *path*, from the checkout *root*, in HEAD's tree; None when HEAD
    holds no file there, a directory or a submodule among them."""
    entry = _head_entry(root, path)
    return entry[1] if entry is not None and entry[0] == "blob" else None


def check_committed_tree(code: CodeState) -> None:
    """Refuse to plan a scored instance from a checkout with uncommitted changes:
    a path that differs from HEAD, or an untracked file git does not ignore,
    in the checkout or its harbor submodule (``CodeState.dirty_diff_sha256``), and a harbor
    checked out at another commit than the one HEAD's tree pins it at, or pinned at none."""
    if code.dirty_diff_sha256 is not None:
        raise PlanError(f"the checkout {code.checkout} or its harbor submodule has uncommitted changes (dirty diff "
                        f"{code.dirty_diff_sha256[:12]}): a scored instance runs from committed code; "
                        "commit them, or plan from a clean checkout")
    pinned = _head_entry(code.checkout, "harbor")
    if pinned is None or pinned[0] != "commit":
        raise PlanError(f"the checkout {code.checkout} pins no harbor commit at HEAD, so the harbor commit "
                        f"{code.harbor_commit[:12]} is no committed code")
    if pinned[1] != code.harbor_commit:
        raise PlanError(f"the checkout {code.checkout} pins harbor at {pinned[1][:12]}, but its harbor submodule is at "
                        f"{code.harbor_commit[:12]}: a scored instance runs from committed code; "
                        "commit the pointer, or run git submodule update")


def check_defaults_committed(checkout: Path, files: MeasuredFiles) -> None:
    """Refuse a task's default *files* pins that is not committed at HEAD of *checkout* as the bytes
    the plan pinned: a scored instance loads a default only as the commit its
    plan records holds it. A store file is pinned by its bytes alone. Raise
    PlanError when the checkout has no HEAD commit, when a file is not a blob at HEAD, or when the
    blob at HEAD is not the one pinned. The caller checks that the tree is clean first
    (``check_committed_tree``); this check compares the bytes as well, so it holds alone."""
    root = checkout.resolve()
    defaults = [pin for pin in files.pins().values() if pin.source == "defaults"]
    if defaults and _git(root, "rev-parse", "--verify", "--quiet", "HEAD").returncode != 0:
        raise PlanError(f"the checkout {root} has no HEAD commit, so the task's defaults its roles load "
                        "cannot be committed files")
    for pin in defaults:
        blob = _head_blob(root, pin.path)
        if blob is None:
            raise PlanError(f"{pin.path} is not committed at HEAD of {root}: a scored instance loads only a "
                            "committed default")
        if blob != pin.blob:
            raise PlanError(f"{pin.path} in the checkout is blob {pin.blob[:12]}, not the {blob[:12]} committed at "
                            "HEAD: a scored instance loads only a committed default")


@dataclass(frozen=True)
class UploadFolder:
    """The folders a build uploads into its sandbox, merged at one destination, as the task pins
    them."""

    #: Their paths from the checkout's root, in the task's order.
    paths: tuple[str, ...]
    #: The tree digest of their merged view (``hls.agent.uploads.merged_digest``), the one the
    #: task commits.
    sha256: str


def check_upload_folder(checkout: Path, task_module: str, producer: Any) -> UploadFolder | None:
    """The folders *producer* (the task's ``ARTIFACT.producer``) uploads into every build's
    sandbox (``upload_dir``, each from *task_module*'s directory when relative), as they lie in
    *checkout*; None for a producer that uploads none. A folder may be no part of the commit, so
    this is what the build would hand the agent. They are judged by the rule the agent_build
    stage applies when the build starts (``hls.agent.uploads.checked_upload``), here before an
    allocation is asked for: raise PlanError when a folder resolves outside the checkout or to
    no directory, holds a link or an entry that is neither a file nor a directory, when two hold
    the same path, or when their merged view has another tree digest than the one the task pins
    (``upload_sha256``), and when they would land under neither ``/task`` nor ``/workspace``
    (``upload_dest``, ``upload_destination``). A refusal of a task that uploads the toolchain
    library (``LIBRARY_DIR``) ends with the command that assembles it."""
    if producer.upload_dir is None:
        return None
    root = checkout.resolve()
    try:
        folders, digest = checked_upload(producer.upload_dir, task_dir(TaskRef(module=task_module)), root,
                                         producer.upload_sha256)
    except UploadRefused as e:
        named = [PurePosixPath(os.path.normpath(PurePosixPath(*task_module.split(".")[:-1]) / value))
                 for value in producer.upload_dir]
        then = f"; run {ASSEMBLE_COMMAND}" if PurePosixPath(LIBRARY_DIR.as_posix()) in named else ""
        raise PlanError(f"{task_module} uploads {', '.join(producer.upload_dir)} (producer.upload_dir) into every "
                        f"build's sandbox, and {e}: copy in the reference documents the task pins by "
                        f"producer.upload_sha256, which are not committed{then}") from None
    try:
        upload_destination(producer.upload_dest)
    except UploadRefused as e:
        raise PlanError(f"{task_module} copies its upload folder to {producer.upload_dest} (producer.upload_dest), "
                        f"and {e}") from None
    return UploadFolder(paths=tuple(folder.relative_to(root).as_posix() for folder in folders), sha256=digest)


def check_skills_folder(checkout: Path, task_module: str, producer: Any) -> list[str]:
    """The names of the skills *producer* (the task's ``ARTIFACT.producer``) copies into every
    build's sandbox from its ``skills_dir``, named from *task_module*'s directory when
    relative, as the folder lies in *checkout*; none for a producer that names no folder. The
    folder is judged by the rule the agent_build stage applies when the build starts
    (``hls.agent.uploads.task_skills``), here before an allocation is asked for: raise
    PlanError when it resolves outside the checkout or to no directory, or is neither a skill
    nor a folder of skills, naming each directory in it that holds no ``SKILL.md``."""
    if producer.skills_dir is None:
        return []
    try:
        skills = task_skills(producer.skills_dir, task_dir(TaskRef(module=task_module)), checkout.resolve())
    except UploadRefused as e:
        raise PlanError(f"{task_module} names {producer.skills_dir} as its skills (producer.skills_dir), which every "
                        f"build copies into its sandbox, and {e}") from None
    return [s.name for s in skills]


class AnswersMissing(PlanError):
    """A grading's reference answers are not all in the store's answers area, which an
    ``hls oracle fill`` of the task writes."""


@dataclass(frozen=True)
class OracleCoverage:
    """The answers a grading's verify finds in the store's answers area: one to every question a
    completed fill recorded for the rows it will ask about."""

    #: The key of the rows entry whose test rows were asked (``hls.provision.model.rows_key``).
    rows: str
    #: The fingerprints of the task's caching references, one per answer key, in the task's order.
    fingerprints: tuple[str, ...]
    #: The prompts the question sets of those keys hold, together, every one of them answered.
    prompts: int
    #: The answers files the answers came from.
    files: int
    #: Who wrote those files, sorted: the instance whose attempt each writer is (an oracle fill's,
    #: or a grading's that banked answers its verify computed), or a writer's own name when no
    #: attempt was given it (``hls.stages.verify.writer_instance``).
    writers: tuple[str, ...]


class _AnswersLog(logging.Filter):
    """Holds back every record the answers area's reader logs and keeps its warnings' text, so
    the only line the check puts on stderr is its refusal, which names them."""

    def __init__(self) -> None:
        super().__init__()
        #: Each warning's message, on one line, in the order logged.
        self.warnings: list[str] = []

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            self.warnings.append(" ".join(record.getMessage().split()))
        return False


@contextmanager
def _answers_log() -> Iterator[_AnswersLog]:
    """The records of ``hls.verify.answers``, held back while the block runs."""
    logger = logging.getLogger(StoreAnswers.__module__)
    held = _AnswersLog()
    logger.addFilter(held)
    try:
        yield held
    finally:
        logger.removeFilter(held)


def _rows_key(task_id: str, task: Any) -> str:
    """The key of the rows entry a grading of *task*, whose id is *task_id*, provisions, resolved
    from its declaration alone (``hls.provision.role.resolve``) as the provision stage resolves
    it. Raise PlanError when the task does not resolve or no tenant of it has a data source."""
    try:
        key = resolve(task).rows_key
    except Refused as e:
        raise PlanError(f"{task_id}'s provisioning does not resolve: {' '.join(str(e).split())}") from None
    if key is None:
        raise PlanError(f"{task_id}'s verifiers keep reference answers in the store's answers area, but no tenant of "
                        "it has a data source, so its verify asks nothing a fill could answer")
    return key


def task_flags(task_ref: TaskRef) -> list[str]:
    """*task_ref*'s task overrides as a command line gives them, in path order: each
    ``--task.<path>`` and its value, a bool as ``true`` or ``false``."""
    return [part for path, value in sorted(task_ref.overrides.items())
            for part in (f"--{path}", json.dumps(value) if isinstance(value, bool) else str(value))]


def _fill_command(task_ref: TaskRef) -> str:
    """The ``hls oracle fill`` command that fills the answers a grading of *task_ref* asks for: its
    module with its task overrides, each as a command line gives it (``task_flags``)."""
    return shlex.join(["hls", "oracle", "fill", task_ref.module, *task_flags(task_ref)])


def check_oracle_coverage(store: PosixStore, *, task_ref: TaskRef, task_id: str, task: Any, gpu_kind: str,
                          reference_sha256: str | None) -> OracleCoverage | None:
    """The answers a grading of *task*, its ``TASK`` with the plan's task overrides, finds in
    the answers area of *task_id*, the id the plan names the task by (``Plan.task``), whose area
    the verify stage mounts, when its verify role keys them by *gpu_kind* (the GPU kind of the
    role's ``reference_shape``, ``hls.stages.verify.NO_GPU`` for a role with none) and the
    reference image whose file has sha256 *reference_sha256*. *task_ref* is the grading's task
    reference, its module and its task overrides, which a refusal names the fill of: a fill run
    without the overrides answered the bare task's rows, under another key. None for a task with
    no caching
    reference (``caching_references``), and for a *reference_sha256* of None (a built image,
    whose file is not known before the images stage runs; the caller warns).

    No row is read. For each answer key (``question_scopes``) the check reads the question set a
    completed fill recorded there for the rows entry the grading provisions (``_rows_key``) and
    the prompts the task shapes from its rows (``question_shape``): every question its verify may
    ask of that key. Each question is looked up as the verify role looks a prompt up
    (``StoreAnswers.answered``, the reader ``unanswered`` uses), which reads and skips the area's
    files as the verify role does. Raise AnswersMissing when a key holds no such set (no fill
    completed for these rows) and when a question has no answer, naming the first key that lacks one, how many it lacks, the first such question, how many
    files the key holds and the area's files the reader skipped; raise PlanError when a set cannot be read."""
    if not caching_references(task) or reference_sha256 is None:
        return None
    rows = _rows_key(task_id, task)
    area = answers_area(task_id)
    fingerprints, asked, gaps = [], 0, []
    with _answers_log() as logged:
        answers = StoreAnswers(PosixStore(store.path(area)), gpu_kind=gpu_kind, image=reference_sha256,
                               writer=_READER)
        for fingerprint, tenants in question_scopes(task).items():
            shape = question_shape(task, tenants)
            key = answer_key(fingerprint, gpu_kind, reference_sha256)
            where, name = f"{area}/{key}", questions_name(rows, shape)
            try:
                recorded = answers.question_set(fingerprint, rows=rows, shape=shape)
            except ValidationError as e:
                raise PlanError(f"{where}/{name} cannot be read as a question set: "
                                f"{validation_errors(e)}") from None
            except (StoreError, OSError, ValueError) as e:
                raise PlanError(f"{where}/{name} cannot be read as a question set: {type(e).__name__}: "
                                f"{' '.join(str(e).split())}") from None
            if recorded is None:
                raise AnswersMissing(f"{task_id}'s verifiers keep reference answers in the store's answers area, and "
                                     f"there is no completed fill for these rows: {where} holds no {name}, the "
                                     f"question set an oracle fill writes once it has banked an answer to every one. "
                                     f"Run {_fill_command(task_ref)} first")
            fingerprints.append(fingerprint)
            asked += len(recorded.questions)
            missing = [q for q in recorded.questions if not answers.answered(fingerprint, q.request_id, q.digest)]
            if missing:
                gaps.append((fingerprint, key, len(missing), len(recorded.questions), missing[0]))
    if gaps:
        fingerprint, key, missing, of, first = gaps[0]
        files = sum(path.startswith(f"{key}/") for path in answers.read)
        more = f"; {len(gaps) - 1} more references lack answers too" if len(gaps) > 1 else ""
        skipped = (f"; answers files the verify role skips as unreadable: {len(logged.warnings)}, the first: "
                   f"{logged.warnings[0]}" if logged.warnings else "")
        raise AnswersMissing(f"{task_id}'s verify would compute reference answers inside the grading: {missing} of "
                             f"the {of} questions a completed fill recorded for reference {fingerprint} have no "
                             f"answer in {area}/{key} ({files} answers files) for {gpu_kind} GPUs and reference image "
                             f"{reference_sha256[:12]}, {first.tenant}/{first.request_id} first{more}{skipped}. Run "
                             f"{_fill_command(task_ref)} first")
    writers = sorted({writer_instance(w) or w for w in answers.read_writers.values()})
    return OracleCoverage(rows=rows, fingerprints=tuple(fingerprints), prompts=asked, files=len(answers.read),
                          writers=tuple(writers))


def arm_lines(arm: ArmRecord, *, soft_sec: int | None, hard_sec: int, arm_change: str | None) -> list[str]:
    """The arm a build runs as the plan resolved it, for the command to print before it submits:
    its harness, model, window, output limit and effort; its route by name, with the
    names of the secrets its credential and base URL come from; the wall budget the agent is
    told and the one it is held to; and the declared reason it differs from its last build,
    when one was given. The record holds names only, so no line can hold a value."""
    route = arm.route_table
    base = f"base URL from secret {route.base_url}" if route.base_url is not None else "the harness's own provider"
    effort = f"effort {arm.effort}" if arm.effort is not None else "effort unset (the harness's default)"
    told = f"told {soft_sec:,} s" if soft_sec is not None else "told no budget"
    lines = [f"arm {arm.name}: {arm.harness}, model {arm.model}, context window {arm.context_window:,} tokens, "
             f"output limit {arm.output_limit:,} tokens, {effort}",
             f"route {arm.route}: credential from secret {route.credential}, {base}, auth {route.auth}",
             f"budget: {told}, held to {hard_sec:,} s"]
    if arm_change is not None:
        lines.append(f"declared change from its last build: {arm_change}")
    return lines
