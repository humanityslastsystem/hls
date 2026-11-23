"""Where each instance keeps its plan, lease, allocation status and stage outputs.

An instance is one pipeline execution: a rollout, a grading under a rollout, a
baseline-credit measurement, a calibration, a cache warmer or an export. Each owns a disjoint
subtree, so a build-only rollout and any number of later gradings never write the same key."""

from __future__ import annotations

import datetime as _dt
import re
import secrets
from dataclasses import dataclass

from hls.store.base import AlreadyExists, Store, check_key, check_segment

_STAGE = re.compile(r"[a-z][a-z0-9_]*")
# The directory name of an attempt: ASCII digits, no sign, no leading zero.
_ATTEMPT_NAME = re.compile(r"[1-9][0-9]*")
#: File name of a stage's manifest inside its stage directory.
MANIFEST_FILE = "MANIFEST.json"
#: The area every rollout lives in (``InstanceLayout.rollout``). An area is the first segment of
#: each of its instances' roots, ``<area>/<task>/<id>``.
ROLLOUTS = "runs"
#: The area of baseline-credit measurements (``InstanceLayout.credit``).
CREDITS = "credits"
#: The area of calibrations (``InstanceLayout.calibration``).
CALIBRATIONS = "calibrations"
#: The area of cache warmers (``InstanceLayout.warmer``).
WARMERS = "warmers"
#: The area of exports (``InstanceLayout.export``).
EXPORTS = "exports"
#: Every area an instance lives in.
AREAS = (ROLLOUTS, CREDITS, CALIBRATIONS, WARMERS, EXPORTS)
#: The areas whose instances hold gradings of their own: a rollout's, and a cache warmer's
#: evaluations (``hls grade --package``, ``--measured`` or ``--artifact.image``).
GRADED_AREAS = (ROLLOUTS, WARMERS)
#: The directory under an instance of a graded area that holds its gradings, each an instance of
#: its own (``InstanceLayout.grading``).
GRADINGS_DIR = "gradings"
#: A task's index directory in an area, beside the task's instances there: one row file per
#: instance (``InstanceKey.index_row``). No instance id may take its name.
INDEX_DIR = "index"


def _index(value: int, minimum: int, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{what} must be an integer >= {minimum}, got {value!r}")
    return value


def check_stage(name: str) -> str:
    """Return *name* if it is a valid stage name (lowercase snake case), else raise
    ValueError."""
    if not _STAGE.fullmatch(name):
        raise ValueError(f"invalid stage name {name!r}")
    return name


def check_manifest_key(key: str) -> str:
    """Return *key* if it is a valid store key of some instance's stage manifest, one that
    ends in ``/stages/<stage>/MANIFEST.json`` below a non-empty root, else raise
    ValueError."""
    parts = check_key(key).split("/")
    if len(parts) < 4 or parts[-3] != "stages" or parts[-1] != MANIFEST_FILE:
        raise ValueError(f"{key!r} is not a stage manifest key")
    check_stage(parts[-2])
    return key


def new_rollout_id() -> str:
    """A new instance id: a compact UTC stamp and six hex digits, so ids sort by creation and
    two made in the same second differ."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{secrets.token_hex(3)}"


def new_grading_id() -> str:
    """A grading's id: a new instance id with a ``g-`` prefix, so gradings sort by creation
    and never collide with a run's own grading, ``terminal``."""
    return f"g-{new_rollout_id()}"


def task_index(area: str, task: str) -> str:
    """The key of task *task*'s index directory in *area*, one of ``AREAS``. Raise ValueError
    for another area or a task id that is not one key segment."""
    if area not in AREAS:
        raise ValueError(f"{area!r} is not a store area: {', '.join(AREAS)}")
    return f"{area}/{check_segment(task)}/{INDEX_DIR}"


class NotAnInstance(ValueError):
    """A key that is not an instance's root as ``InstanceLayout``'s constructors build it."""

    def __init__(self, key: str, why: str) -> None:
        super().__init__(f"{key} is not an instance's root: {why}")
        #: The key read.
        self.key = key
        #: Why it is no instance's root, as one clause.
        self.why = why


@dataclass(frozen=True)
class InstanceKey:
    """An instance's root read back into what its constructor took (``InstanceLayout.parse``)."""

    #: The area the instance lives in, one of ``AREAS``.
    area: str
    #: The id of the instance's task.
    task: str
    #: The instance's id in its area; for a grading, the id of the instance it lies under.
    id: str
    #: A grading's own id; None for an instance that is not a grading.
    grading_id: str | None

    @property
    def index_row(self) -> str:
        """The key of the instance's row in its task's index: ``<id>.json``, or for a grading
        ``<id>+<grading id>.json``, so a grading's row lies beside its instance's and no two
        instances share one."""
        name = self.id if self.grading_id is None else f"{self.id}+{self.grading_id}"
        return f"{task_index(self.area, self.task)}/{name}.json"


@dataclass(frozen=True)
class InstanceLayout:
    """Store keys for one instance."""

    #: Key prefix of the instance, e.g. ``runs/<task>/<rollout-id>``.
    root: str

    def __post_init__(self) -> None:
        check_key(self.root)

    @classmethod
    def rollout(cls, task: str, rollout_id: str) -> InstanceLayout:
        """Each id is one key segment, so no id can reach into another instance's subtree."""
        return cls(f"{ROLLOUTS}/{check_segment(task)}/{check_segment(rollout_id)}")

    @classmethod
    def credit(cls, task: str, credit_id: str) -> InstanceLayout:
        return cls(f"{CREDITS}/{check_segment(task)}/{check_segment(credit_id)}")

    @classmethod
    def calibration(cls, task: str, calibration_id: str) -> InstanceLayout:
        return cls(f"{CALIBRATIONS}/{check_segment(task)}/{check_segment(calibration_id)}")

    @classmethod
    def warmer(cls, task: str, warmer_id: str) -> InstanceLayout:
        """A cache warmer (``hls provision``, ``hls images build``), apart from the task's
        rollouts, so warming the cache never adds a run under ``runs/``."""
        return cls(f"{WARMERS}/{check_segment(task)}/{check_segment(warmer_id)}")

    @classmethod
    def export(cls, task: str, export_id: str) -> InstanceLayout:
        """An export of a task's finished instances (``hls export``), apart from its runs and its
        warmers, so exporting adds no run under ``runs/``."""
        return cls(f"{EXPORTS}/{check_segment(task)}/{check_segment(export_id)}")

    @classmethod
    def parse(cls, key: str) -> InstanceKey:
        """The parts of *key*, an instance's root as the constructors build it:
        ``<area>/<task>/<id>``, or a grading's, ``<root>/gradings/<grading id>`` under an
        instance of a graded area. Raise NotAnInstance for any other key, and for an id that is
        the name of its task's index directory or holds "+", which joins a grading's id to its
        instance's in its index row's name, so either could name another instance's row."""
        try:
            parts = check_key(key).split("/")
        except ValueError:
            raise NotAnInstance(key, "it is not a store key") from None
        graded = len(parts) == 5 and parts[0] in GRADED_AREAS and parts[3] == GRADINGS_DIR
        if parts[0] not in AREAS or not (len(parts) == 3 or graded):
            raise NotAnInstance(key, f"a root is <area>/<task>/<id>, the area one of {', '.join(AREAS)}, or a "
                                     f"grading's, <root>/{GRADINGS_DIR}/<id> under one of {', '.join(GRADED_AREAS)}")
        if parts[2] == INDEX_DIR:
            raise NotAnInstance(key, f"its id is {INDEX_DIR}, the name of its task's index directory")
        if "+" in parts[2]:
            raise NotAnInstance(key, "its id holds +, which joins a grading's id to its instance's in the name of "
                                     "the grading's index row")
        return InstanceKey(area=parts[0], task=parts[1], id=parts[2], grading_id=parts[4] if graded else None)

    def grading(self, grading_id: str) -> InstanceLayout:
        """A grading instance under this rollout."""
        return InstanceLayout(f"{self.root}/{GRADINGS_DIR}/{check_segment(grading_id)}")

    @property
    def plan(self) -> str:
        """Directory of the plan's revisions. Each revision is a whole plan in its own
        create-only file, so a writer that lost the lease cannot overwrite a newer one."""
        return f"{self.root}/plan"

    @property
    def source_bundle(self) -> str:
        """The copy of the source bundle its operator named, made by ``hls package`` before it
        plans the instance. The instance's stages read the copy, never the operator's file."""
        return f"{self.root}/source.bundle"

    def plan_revision(self, n: int) -> str:
        return f"{self.plan}/rev-{_index(n, 1, 'revision'):08d}.json"

    @property
    def lease(self) -> str:
        return f"{self.root}/lease"

    @property
    def cancel_request(self) -> str:
        """Created by ``hls cancel`` while another orchestrator holds the lease."""
        return f"{self.root}/cancel-request.json"

    def submission(self, k: int, n: int) -> str:
        """Directory of submission *n* of allocation *k*: its batch script, its Slurm
        output and its runner's status. Each submission is one Slurm job and one writer."""
        k = _index(k, 0, "allocation")
        return f"{self.root}/allocations/{k}/submissions/{_index(n, 1, 'submission')}"

    def submission_file(self, k: int, n: int, name: str) -> str:
        return check_key(f"{self.submission(k, n)}/{name}")

    def allocation_status(self, k: int, n: int) -> str:
        return self.submission_file(k, n, "status.json")

    def orchestrator_file(self, generation: int, name: str) -> str:
        """A file written only by the orchestrator holding lease *generation*."""
        gen = _index(generation, 1, "generation")
        return check_key(f"{self.root}/orchestrator/gen-{gen:08d}/{name}")

    def stage(self, stage: str) -> str:
        return f"{self.root}/stages/{check_stage(stage)}"

    def manifest(self, stage: str) -> str:
        return f"{self.stage(stage)}/{MANIFEST_FILE}"

    def attempt(self, stage: str, n: int) -> str:
        return f"{self.stage(stage)}/attempts/{_index(n, 1, 'attempt')}"

    def attempt_file(self, stage: str, n: int, name: str) -> str:
        return check_key(f"{self.attempt(stage, n)}/{name}")

    def attempts(self, store: Store, stage: str) -> list[int]:
        """Claimed attempt numbers of *stage*, ascending. A name that is not a positive
        decimal number as ``attempt`` writes it is not an attempt and is skipped."""
        names = store.list_children(f"{self.stage(stage)}/attempts")
        return sorted(int(name) for name in names if _ATTEMPT_NAME.fullmatch(name))

    def claim_attempt(self, store: Store, stage: str) -> int:
        """Claim the next attempt number of *stage*. Concurrent claimers each get a
        different number."""
        n = max(self.attempts(store, stage), default=0) + 1
        while True:
            try:
                store.create(self.attempt_file(stage, n, "CLAIM"), b"")
                return n
            except AlreadyExists:
                n += 1
