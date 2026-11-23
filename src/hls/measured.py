"""A task's measurements: the baseline credit its scorer subtracts, the calibration its gate
reads and the dev reference its contestant is shown, found where they are kept and pinned by the
plan that uses them.

They are outputs of measurements, not part of the task's design, and no task module names a path
to them. Two places hold them:

- the store's measured area: each measuring command writes what it measured there when it
  succeeds (``hls baseline-credit``: ``baseline.json``, with the hardware its drives ran on;
  ``hls calibrate``: ``calibration.json``, with the hardware its servings ran on;
  ``hls oracle fill``: ``dev_reference.json``), as ``measured/<task id>/<blob>/<file>``, never
  replaced, and as ``measured/<task id>/<file>``, the store's current file, which the next
  measurement replaces (``write_measured``);
- the task's defaults, ``tasks/<task>/defaults/<file>`` beside the task module: what the task's
  authors measured, shipped with the task as data.

``find`` looks for each file a task loads (``needed_files``) in that order and takes the first
that fits the task: one its reader accepts for the task as it is now, with the run's overrides
(``read_baseline_credit``: the workload fingerprint, the load, the GPU kind and the reference
server; ``read_calibration``: the gate's kind, the workload fingerprint, the GPU kind and the
measurement's own checks; ``read_dev_reference``: the dev rows, the reference and the metrics it
was produced on). A file that does not fit is not used, and why is said. When neither fits, the
file is measured first (``hls run``, ``hls.pipeline.prepare``).

A plan pins the files it found (``MeasuredFiles``): where each is and the blob id of its bytes,
a store file by its ``measured/<task id>/<blob>/<file>`` key, so a later measurement leaves the
pinned bytes in place. Every stage that loads one, or hands one to a role, reads it from there
once and refuses other bytes (``read_pinned``, ``load_pinned``), so nothing loads a file other
than the one the plan recorded, and the records name each by its pin.

A run uses the measured files it finds with the images its store builds from the task's recipes.
Its record names each file it used, and when one is the task's default, the run says which in its
record and when it ends (``hls.obs.conditions.measured_note``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hls.store.base import check_segment
from hls.store.posix import PosixStore
from hls.taskref import MeasuredPaths, TaskRef, blob_id, task_dir

#: The baseline credit's file name (``hls.baseline_credit``).
BASELINE_CREDIT_FILE = "baseline.json"
#: The calibration's file name (``hls.verify.calibration``).
CALIBRATION_FILE = "calibration.json"
#: The store's area of measurements, one directory per task id.
MEASURED_AREA = "measured"
#: The directory of a task's defaults, beside its task module.
DEFAULTS_DIR = "defaults"
#: The dev split's reference answers a task's brief discloses, among its defaults.
DEV_REFERENCE_FILE = "dev_reference.json"
#: What each measured file is, for people.
WHAT = {BASELINE_CREDIT_FILE: "baseline credit", CALIBRATION_FILE: "calibration", DEV_REFERENCE_FILE: "dev reference"}
#: The command that measures each file.
MEASURES = {BASELINE_CREDIT_FILE: "hls baseline-credit", CALIBRATION_FILE: "hls calibrate",
            DEV_REFERENCE_FILE: "hls oracle fill"}


def store_key(task_id: str, name: str) -> str:
    """The store key of task *task_id*'s current measured file *name*: the one a plan looks for,
    which the next measurement replaces."""
    if name not in WHAT:
        raise ValueError(f"{name!r} is not a measured file: one of {sorted(WHAT)}")
    return f"{MEASURED_AREA}/{check_segment(task_id)}/{name}"


def pinned_key(task_id: str, name: str, blob: str) -> str:
    """The store key of task *task_id*'s measured file *name* whose bytes have blob id *blob*,
    which a plan pins: written by the measurement beside the current file and never replaced."""
    return f"{MEASURED_AREA}/{check_segment(task_id)}/{check_segment(blob)}/{store_key(task_id, name).rsplit('/', 1)[1]}"


def write_measured(store: PosixStore, task_id: str, name: str, data: bytes) -> str:
    """Write *data*, what a measurement of task *task_id* made, as its measured file *name*: first
    at ``pinned_key``, unless those bytes are there already, then as the store's current file
    (``store_key``), which it replaces. The current file's key."""
    pinned = pinned_key(task_id, name, blob_id(data))
    if not store.exists(pinned):
        store.write_bytes(pinned, data)
    key = store_key(task_id, name)
    store.write_bytes(key, data)
    return key


def defaults_path(ref: TaskRef, name: str) -> Path:
    """Where the task module of *ref* holds its default *name*: ``defaults/<name>`` beside it."""
    return task_dir(ref) / DEFAULTS_DIR / name


def needed_files(task: Any) -> list[str]:
    """The measured files *task*, its ``TASK`` with a plan's overrides, loads, in the order they are
    found and measured: the baseline credit when its scorer scores against one, then the
    calibration when its verifiers need one (``calibration_kind``), then the dev reference when a
    data source declares one (``hls.verify.dev_reference.declaring_tenants``)."""
    from hls.verify.calibration import calibration_kind
    from hls.verify.dev_reference import declaring_tenants

    names = []
    if task.scorer.scores_against_baseline():
        names.append(BASELINE_CREDIT_FILE)
    if calibration_kind(task) is not None:
        names.append(CALIBRATION_FILE)
    if declaring_tenants(task):
        names.append(DEV_REFERENCE_FILE)
    return names


class MeasuredFile(BaseModel):
    """A measured file a plan uses, pinned."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Where it is kept: the store's measured area, or the task's defaults.
    source: Literal["store", "defaults"]
    #: Its store key, or its path from the checkout's root.
    path: str
    #: The blob id of its bytes (``hls.taskref.blob_id``), which records name it by.
    blob: str = Field(pattern=r"^[0-9a-f]{40}$")

    @model_validator(mode="after")
    def _within(self) -> "MeasuredFile":
        """A store file is its bytes' key in the measured area (``pinned_key``); a default is a
        relative path that ends in ``defaults/<name>`` and climbs out of nothing."""
        parts = self.path.split("/")
        name = parts[-1]
        if self.source == "store":
            if len(parts) != 4 or parts[0] != MEASURED_AREA or self.path != pinned_key(parts[1], name, self.blob):
                raise ValueError(f"{self.path!r} is not the key of blob {self.blob} in the store's measured area")
        elif (name not in WHAT or len(parts) < 3 or parts[-2] != DEFAULTS_DIR or self.path.startswith("/")
              or any(p in ("", ".", "..") for p in parts)):
            raise ValueError(f"{self.path!r} is not a task's default, a relative path ending in "
                             f"{DEFAULTS_DIR}/<{' or '.join(sorted(WHAT))}>")
        return self


class MeasuredFiles(BaseModel):
    """The measured files a plan's roles load; None where the task loads none."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The baseline credit (``baseline.json``).
    credit: MeasuredFile | None = None
    #: The calibration (``calibration.json``).
    calibration: MeasuredFile | None = None
    #: The dev reference (``dev_reference.json``), which stages publish beside the dev rows.
    dev_reference: MeasuredFile | None = None

    def pins(self) -> dict[str, MeasuredFile]:
        """Each pinned file by its name."""
        found = {BASELINE_CREDIT_FILE: self.credit, CALIBRATION_FILE: self.calibration,
                 DEV_REFERENCE_FILE: self.dev_reference}
        return {name: pin for name, pin in found.items() if pin is not None}


def local_path(pin: MeasuredFile, *, checkout: Path, store_root: Path) -> Path:
    """The file *pin* names on this machine: under *store_root* for a store file, under the plan's
    *checkout* for a default."""
    return (store_root if pin.source == "store" else checkout) / pin.path


def read_pinned(pin: MeasuredFile, *, checkout: Path, store_root: Path) -> bytes:
    """The bytes of the file *pin* names (``local_path``). Raise ValueError when they are not the
    bytes the plan pinned, or the file is gone."""
    path = local_path(pin, checkout=checkout, store_root=store_root)
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ValueError(f"{pin.path}, which the plan pinned, cannot be read: {type(e).__name__}") from None
    if blob_id(data) != pin.blob:
        raise ValueError(f"{pin.path} is blob {blob_id(data)[:12]}, not the {pin.blob[:12]} the plan pinned: it "
                         "changed since the plan was made")
    return data


def write_pinned(files: MeasuredFiles, directory: Path, *, checkout: Path, store_root: Path) -> MeasuredPaths:
    """Copy each file *files* pins, checked (``read_pinned``), into *directory* under its own name,
    and return the copies' paths, which a role loads: a copy cannot change under the role."""
    directory.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, pin in files.pins().items():
        paths[name] = directory / name
        paths[name].write_bytes(read_pinned(pin, checkout=checkout, store_root=store_root))
    return MeasuredPaths(calibration=paths.get(CALIBRATION_FILE), credit=paths.get(BASELINE_CREDIT_FILE))


def pinned_dev_files(files: MeasuredFiles, *, checkout: Path, store_root: Path) -> dict[str, dict[str, str]]:
    """The files the dev reference *files* pins publishes beside each declaring tenant's dev rows,
    by tenant id, each name to its text; empty when *files* pins none. The plan held the record
    to the task when it pinned it. Raise ValueError as ``read_pinned`` does, and for bytes that are
    not a dev reference record."""
    from hls.verify.dev_reference import DevReference

    if files.dev_reference is None:
        return {}
    data = read_pinned(files.dev_reference, checkout=checkout, store_root=store_root)
    record = DevReference.model_validate_json(data)
    return {tenant: {f.name: f.text} for tenant, f in record.tenants.items()}


class PinMismatch(ValueError):
    """A file a plan pinned is gone, or holds other bytes than the plan pinned."""


def load_pinned(ref: TaskRef, files: MeasuredFiles, *, calibration: bool, credit: bool, checkout: Path,
                store_root: Path) -> Any:
    """The task *ref* names with the calibration and the baseline credit *files* pins injected
    when asked (``hls.taskref.read_task``), each read once and held to the blob id the plan pinned:
    the bytes checked are the bytes loaded. Raise PinMismatch when a pinned file is gone or its
    bytes are not the pinned ones, and what ``read_task`` raises."""
    from hls.taskref import read_task

    pins = files.pins()
    for pin in pins.values():
        if not local_path(pin, checkout=checkout, store_root=store_root).is_file():
            raise PinMismatch(f"{pin.path}, which the plan pinned, is not there")
    paths = {name: local_path(pin, checkout=checkout, store_root=store_root) for name, pin in pins.items()}
    loaded = read_task(ref, calibration=calibration, credit=credit,
                       files=MeasuredPaths(calibration=paths.get(CALIBRATION_FILE),
                                           credit=paths.get(BASELINE_CREDIT_FILE)))
    for name, read in ((CALIBRATION_FILE, loaded.calibration), (BASELINE_CREDIT_FILE, loaded.credit)):
        pin = pins.get(name)
        if read is not None and pin is not None and read.blob != pin.blob:
            raise PinMismatch(f"{pin.path} is blob {read.blob[:12]}, not the {pin.blob[:12]} the plan pinned: it "
                              "changed since the plan was made")
    return loaded.task


def fit(name: str, path: Path, task: Any) -> tuple[str | None, bytes]:
    """Whether the measured file *name* at *path* fits *task*, read once by its reader: why it does
    not, as the reader refuses it, or None when it fits, and the bytes the reader read (empty
    when it read none)."""
    from hls.baseline_credit import BaselineCreditError, read_baseline_credit
    from hls.verify.calibration import CalibrationError, calibration_kind, read_calibration
    from hls.verify.dev_reference import DevReferenceError, read_dev_reference

    try:
        if name == BASELINE_CREDIT_FILE:
            _, data = read_baseline_credit(path, task)
        elif name == DEV_REFERENCE_FILE:
            _, data = read_dev_reference(path, task)
        else:
            kind = calibration_kind(task)
            assert kind is not None
            _, data = read_calibration(path, task, kind)
    except (BaselineCreditError, CalibrationError, DevReferenceError) as e:
        return " ".join(str(e).split()), b""
    return None, data


@dataclass(frozen=True)
class Found:
    """Where a measured file was looked for, and what was found."""

    #: The file's name.
    name: str
    #: The first file that fits the task; None when none does.
    pin: MeasuredFile | None
    #: Why each file that was there was not used, one line each.
    refused: list[str] = field(default_factory=list)

    def said(self) -> str:
        """One line for people: the file used and why each file before it was not, or why none
        was."""
        what = WHAT[self.name]
        if self.pin is not None:
            where = "the store's" if self.pin.source == "store" else "the task's defaults"
            passed = "".join(f"; not {refusal}" for refusal in self.refused)
            return f"{what}: {where} {self.pin.path} (blob {self.pin.blob[:12]}){passed}"
        why = "; ".join(self.refused) if self.refused else "neither the store nor the task's defaults hold one"
        return f"{what}: none fits the task ({why})"


def find(name: str, *, task_ref: TaskRef, task: Any, task_id: str, checkout: Path, store: PosixStore) -> Found:
    """The measured file *name* for *task*, the task *task_ref* names with its overrides applied:
    the store's current file when it fits, else the task's default when it fits (``fit``), each
    read once and pinned by the blob id of the bytes the fit was judged on; or no pin, with why
    each file that was there was not used. A store file is pinned by its bytes' key
    (``pinned_key``), which the measurement wrote beside it (``write_measured``); a current file
    whose bytes are not there is not used. *checkout* is the checkout the task module is imported
    from."""
    refused = []
    key = store_key(task_id, name)
    if store.exists(key):
        problem, data = fit(name, store.path(key), task)
        if problem is None:
            pinned = pinned_key(task_id, name, blob_id(data))
            if store.exists(pinned):
                return Found(name=name, pin=MeasuredFile(source="store", path=pinned, blob=blob_id(data)))
            problem = f"{pinned}, which the measurement that wrote it writes first, is not there"
        refused.append(f"store {key}: {problem}")
    default = defaults_path(task_ref, name)
    if default.is_file():
        where = default.relative_to(checkout.resolve()).as_posix()
        problem, data = fit(name, default, task)
        if problem is None:
            return Found(name=name, pin=MeasuredFile(source="defaults", path=where, blob=blob_id(data)),
                         refused=refused)
        refused.append(f"defaults {where}: {problem}")
    return Found(name=name, pin=None, refused=refused)


def measured_files(found: list[Found]) -> MeasuredFiles:
    """The pins of *found*, each of whose files fits."""
    pins = {f.name: f.pin for f in found}
    if None in pins.values():
        missing = sorted(name for name, pin in pins.items() if pin is None)
        raise ValueError(f"no {' or '.join(missing)} fits the task")
    return MeasuredFiles(credit=pins.get(BASELINE_CREDIT_FILE), calibration=pins.get(CALIBRATION_FILE),
                         dev_reference=pins.get(DEV_REFERENCE_FILE))
