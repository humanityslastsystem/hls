"""The task a role works on, named the way a plan can carry it: a task module and the
``task.*`` overrides the operator gave.

A plan holds JSON, never live objects, so a stage cannot pass its roles a ``TaskSpec``.
It passes a ``TaskRef`` instead, and each role program (the driver, the verifier, the
scorer) rebuilds the task with ``load_task`` from the plan's checkout, which is bound
read-only and checked against the plan's commits before every stage. The
result is the module's ``TASK`` with the overrides applied (``apply_task_overrides``, the one
application every role and the provision role share), then, when asked, the calibration and
the baseline credit injected from the files the role was handed (``MeasuredPaths``), in that
order. The files are measurements, found where ``hls.measured`` keeps them.

``blob_sha`` names a file by the id git stores it under, which is how a score records the
baseline credit it was computed against. ``read_task`` reads each of the two files once, and
names the blob id of the bytes it parsed, so a record of what a role was given never reads the
file again."""

from __future__ import annotations

import hashlib
import importlib
import re
import sys
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from hls.override import apply_overrides
from hls.provision.hf import git_blob_sha1

if TYPE_CHECKING:
    from hls.baseline_credit import BaselineCredit
    from hls.spec.taskspec import TaskSpec
    from hls.verify.calibration import Calibration, ReportCalibration

#: A dotted Python module name, importable from the checkout root.
MODULE_PATTERN = r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*"
# An override key: "task." then a dotted path of field names.
_OVERRIDE = re.compile(r"task(\.[A-Za-z_][A-Za-z0-9_]*)+")
#: The fields of a task no override may set: its drive bounds (``hls.drive.bounds``), which limit
#: what a scored drive may make the harness hold, so only the task module declares them; and its
#: provision roles' memory (``TaskSpec.provision_memory_gib``), which a site changes by its
#: ``[resources.<task>.provision]`` table, as it changes any stage's shape, never by changing the task.
FIXED_TASK_FIELDS = frozenset({"drive_bounds", "provision_memory_gib"})


def check_task_overrides(keys: Iterable[str]) -> None:
    """Raise ValueError naming the *keys* that are no ``task.<field>[.<field>...]`` path, or that
    set a field no override may set (``FIXED_TASK_FIELDS``)."""
    keys = list(keys)
    bad = sorted(k for k in keys if not _OVERRIDE.fullmatch(k))
    if bad:
        raise ValueError(f"overrides must be task.<field>[.<field>...] paths, got {bad}")
    fixed = sorted(k for k in keys if k.split(".")[1] in FIXED_TASK_FIELDS)
    if fixed:
        raise ValueError(f"overrides may not set {', '.join(fixed)}: only the task module declares "
                         f"{', '.join(sorted(FIXED_TASK_FIELDS))}")


class TaskRef(BaseModel):
    """A task module and the overrides applied to its ``TASK``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task module, such as ``tasks.multitenant_slo.task``.
    module: str = Field(pattern=f"^{MODULE_PATTERN}$")
    #: Dotted ``task.*`` path to value, as the CLI's ``--task.*`` flags give them, typed.
    overrides: dict[str, Any] = Field(default_factory=dict)

    @field_validator("overrides")
    @classmethod
    def _task_paths(cls, value: dict[str, Any]) -> dict[str, Any]:
        check_task_overrides(value)
        return value


@contextmanager
def first_on_import_path(root: Path) -> Iterator[None]:
    """*root*, a checkout, at the front of the import path while the block runs, so that its task
    modules import from it; left where it is when it is on the path already."""
    entry = str(root)
    added = entry not in sys.path
    if added:
        sys.path.insert(0, entry)
    try:
        yield
    finally:
        if added:
            sys.path.remove(entry)


def _module(ref: TaskRef) -> ModuleType:
    return importlib.import_module(ref.module)


def task_dir(ref: TaskRef) -> Path:
    """The directory the task module lives in. A namespace package has no file and so no such
    directory."""
    file = getattr(_module(ref), "__file__", None)
    if file is None:
        raise ValueError(f"module {ref.module!r} has no __file__, so it has no task directory")
    return Path(file).resolve().parent


@dataclass(frozen=True)
class MeasuredPaths:
    """The measured files a load injects, each the path of a file on this machine; None where
    the role was handed none."""

    #: The calibration file (``calibration.json``).
    calibration: Path | None = None
    #: The baseline credit file (``baseline.json``).
    credit: Path | None = None


@dataclass(frozen=True)
class ReadFile[T]:
    """A measured file as the load read it, once."""

    #: What its bytes parsed to.
    value: T
    #: The id git commits those bytes under (``blob_id``).
    blob: str


@dataclass(frozen=True)
class LoadedTask:
    """A task as ``read_task`` assembled it, and the committed files it was given."""

    #: The module's ``TASK`` with the overrides applied, and the calibration and the baseline
    #: credit injected when asked.
    task: TaskSpec
    #: The calibration its verifiers were given, as ``calibration.json`` held it; None when none
    #: was asked for or none of its verifiers needs one.
    calibration: ReadFile[Calibration | ReportCalibration] | None
    #: The baseline credit its scorer was given, as ``baseline.json`` held it; None when none was
    #: asked for or its scorer scores against none.
    credit: ReadFile[BaselineCredit] | None


class UnknownOverride(ValueError):
    """A task override names no field of the task."""


def apply_task_overrides(task: TaskSpec, overrides: Mapping[str, Any]) -> TaskSpec:
    """*task* with *overrides* applied, each a dotted ``task.*`` path to its typed value, as a
    ``TaskRef`` holds them; *task* itself when there are none. Raise ValueError for a key that is
    no ``task.<field>`` path or sets a field no override may (``check_task_overrides``),
    UnknownOverride for an override that names no field, and pydantic's ValidationError for a value
    its field refuses."""
    check_task_overrides(overrides)
    try:
        return apply_overrides(task, {k[len("task."):]: v for k, v in overrides.items()})
    except KeyError as e:
        raise UnknownOverride(f"overrides {sorted(overrides)} name no field of the task: {e.args[0]}") from e


def load_task(ref: TaskRef, *, calibration: bool = False, credit: bool = False,
              files: MeasuredPaths = MeasuredPaths()) -> TaskSpec:
    """The ``TaskSpec`` *ref* names (``read_task``)."""
    return read_task(ref, calibration=calibration, credit=credit, files=files).task


def read_task(ref: TaskRef, *, calibration: bool = False, credit: bool = False,
              files: MeasuredPaths = MeasuredPaths()) -> LoadedTask:
    """The ``TaskSpec`` *ref* names: the module's ``TASK`` with the overrides applied
    (``apply_task_overrides``), then the calibration and the baseline credit injected when asked,
    from the files *files* names, each file read once and named by the blob id of its bytes. A task
    that needs a file *files* does not name is refused, as its reader refuses a missing file.
    Raise ValueError for a module with no ``TASK`` or an override that names no field."""
    task = getattr(_module(ref), "TASK", None)
    if task is None:
        raise ValueError(f"module {ref.module!r} declares no TASK")
    try:
        task = apply_task_overrides(task, ref.overrides)
    except UnknownOverride as e:
        raise UnknownOverride(f"{ref.module}: {e}") from e
    calibrated_by = credited_by = None
    if calibration:
        from hls.verify.calibration import calibrated

        task, read = calibrated(task, files.calibration)
        calibrated_by = ReadFile(value=read[0], blob=blob_id(read[1])) if read is not None else None
    if credit:
        from hls.baseline_credit import credited

        task, read = credited(task, files.credit)
        credited_by = ReadFile(value=read[0], blob=blob_id(read[1])) if read is not None else None
    return LoadedTask(task=task, calibration=calibrated_by, credit=credited_by)


def blob_id(data: bytes) -> str:
    """The id a file holding *data* is committed under, as ``blob_sha`` gives it for a file."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def blob_sha(path: Path) -> str:
    """The id git commits the file at *path* under (``git hash-object``): the SHA-1 of
    ``"blob <size>\\0"`` and the bytes. A file git stores through no filter, as a calibration
    or a credit file is, has this id in the commit's tree."""
    return git_blob_sha1(path)
