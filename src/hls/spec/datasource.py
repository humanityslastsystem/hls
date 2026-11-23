"""DataSource -- logical data needs declared on the TaskSpec (the config plane).

Provisioning resolves these to physical mountable units. SplitSpec sizes the dev and
test splits; ``partition_by_hash`` assigns each corpus row to one of them by a hash
of the row's own identity (``Dataset.row_id``), so membership is a property of the
row and never of a draw. The dataset (identity + rendering) is the single Dataset
config; there is no separate DatasetRef.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

from pydantic import Field, model_validator

from hls.core.poly import Poly, Spec
from hls.spec.datasets import Dataset


class SplitSize(Poly):
    """How big a split is, as a tagged type so ``1`` (one item) can never be
    confused with ``1.0`` (the whole corpus). Subclass per policy (Count, Fraction);
    ``resolve(n)`` turns the size into a concrete item count for a corpus of ``n``.
    The ``kind`` discriminator serializes into provenance like the rest of the
    config plane.
    """

    def resolve(self, n: int) -> int:
        raise NotImplementedError


class Count(SplitSize):
    """An absolute item count (>= 1)."""

    value: int = Field(ge=1)

    def resolve(self, n: int) -> int:
        return self.value


class Fraction(SplitSize):
    """A fraction of the corpus, in the open interval (0, 1)."""

    value: float = Field(gt=0.0, lt=1.0)

    def resolve(self, n: int) -> int:
        return int(round(n * self.value))


class SplitSpec(Spec):
    dev: Count | Fraction
    test: Count | Fraction


class DataSource(Spec):
    dataset: Dataset
    split: SplitSpec
    #: When set, the dev split is drawn from this corpus and the test split from
    #: ``dataset``'s corpus -- disjointness holds by corpus identity (e.g. GSM8K train vs
    #: GSM8K-Platinum test), not by a shared partition. Each corpus is ranked alone, so
    #: the ``dev+test <= n`` coupling of ``partition_by_hash`` does not apply and
    #: ``split.dev`` may cover the entire dev corpus. Only honored on the multi-tenant
    #: provisioning path.
    dev_dataset: Dataset | None = None
    #: Reference text to publish beside the dev rows, as ``{filename: contents}``.
    #: For a prompt element that a contestant cannot recover from the rows: the rows
    #: hold raw corpus fields, and anything the task's own template puts around them
    #: exists only in task code. A shared few-shot prefix is the case this is for --
    #: its presence is not inferable from a bare question, so a contestant measuring
    #: prefill on the dev split measures a prompt the graded traffic never sends.
    #: Content stays in the task; provisioning only writes what it is handed. Declared
    #: once per source, beside the split it publishes to, and dev only: the test split
    #: is withheld. Disclosure to the contestant rather than traffic, so it is part of
    #: the provisioning digest and not of the workload identity.
    dev_reference: dict[str, str] = Field(default_factory=dict)
    #: The name a measured dev reference is published under beside the dev rows, such as
    #: ``oracle_dev.json``; None for a source that publishes none. The file is the task's
    #: reference answering the dev rows, each answer with its score under the gate's metric, so
    #: a contestant can see what its own server has to keep. It is a measurement, never task
    #: code and not part of the rows entry: each stage whose roles see the dev rows publishes it
    #: beside them (``hls.verify.dev_reference``).
    measured_dev_reference: str | None = Field(default=None, exclude_if=lambda v: v is None)
    #: Rows removed from the served workload, named by their own id
    #: (``Dataset.row_id``), sorted with no repeats. The drive plan drops the listed
    #: sessions from the loaded split before the arrival schedule is drawn
    #: (``hls.spec.drivers.remove_listed_rows``); a multi-turn session goes whole, since
    #: its later prompts are its earlier replies. A row not in the split is a no-op.
    #: Part of the workload identity (``hls.verify.calibration.workload_payload``), not
    #: of the provisioning digest: a task lists here the rows its calibration's own
    #: serving stack ran to the output cap (``hls.calibrate.capped_rows``), and a new
    #: list is a new calibration.
    removed_rows: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_removed_rows(self) -> "DataSource":
        if list(self.removed_rows) != sorted(set(self.removed_rows)):
            raise ValueError(
                "DataSource.removed_rows must be sorted and free of repeats, so the list "
                "has one form and the workload fingerprint one value.")
        return self


#: The form of a turn's request id: ``<row id>.<turn index>``. A row id may not end
#: this way, or a turn id could not be split back into its row.
_TURN_ID = re.compile(r"\.\d+$")


class PartitionError(ValueError):
    """The rows cannot be partitioned as declared: their ids are not unique or not well
    formed, or the split sizes exceed the corpus or resolve a split to no row. Partitioning the
    same corpus again fails the same way."""


def partition_by_hash(
    ids: Sequence[str], *, dev: Count | Fraction | None = None,
    test: Count | Fraction | None = None,
    pinned_test: Sequence[str] = (),
) -> tuple[list[int], list[int]]:
    """Assign corpus rows to the dev and test splits by a hash of each row's own id.

    Rows are ranked by the SHA-256 of their id (``Dataset.row_id``). Dev is the ``dev``
    lowest-ranked rows and test the ``test`` highest-ranked; a side left ``None`` is
    not assigned (an empty list comes back), which is how a source with its own dev
    corpus ranks each corpus alone. Membership is therefore a function of the row's id
    and the split sizes only: a row that leaves the corpus is replaced from the
    unassigned middle of the ranking and no other row changes sides. With ``Fraction``
    sizes the resolved count itself can move by one row when the corpus size does.
    Nothing here is seeded; the task seed drives only genuine draws (arrival schedules,
    think times). Returns the dev and test row positions, each in ascending hash order.

    ``pinned_test`` names rows the corpus puts in test by declaration
    (``Dataset.forced_split``). They are held out of the ranking and lead the test
    split, so the ``test`` count is the hashed pool's and the pinned rows sit on top of
    it. For a corpus with a handful of rows that must all be graded -- a few deep
    sessions among many shallow ones -- a hash gives no such guarantee and its answer
    moves whenever the corpus does. They lead rather than trail because a row's
    position in the split is its arrival slot. Empty, the default, reproduces the
    ranking and the order exactly.

    Ids must be unique, non-empty and not end in ``.<digits>`` (the turn suffix a
    request id carries), so every row has one name and that name maps back to its row;
    PartitionError says which rule the rows break, or that the split sizes ask for more
    rows than the corpus has or resolve a split that is asked for to no row.
    """
    n = len(ids)
    if len(set(ids)) != n:
        seen: set[str] = set()
        dupes = sorted({i for i in ids if i in seen or seen.add(i)})[:5]
        raise PartitionError(f"row ids must be unique; repeated: {dupes}")
    bad = [i for i in ids if not i or _TURN_ID.search(i)]
    if bad:
        raise PartitionError(
            f"row ids must be non-empty and not end in '.<digits>' (the turn suffix of a "
            f"request id): {bad[:5]}")
    pinned = set(pinned_test)
    unknown = sorted(pinned - set(ids))
    if unknown:
        raise PartitionError(f"pinned_test names rows this corpus does not have: {unknown[:5]}")
    pinned_idx = [i for i, rid in enumerate(ids) if rid in pinned]
    hashed = [i for i in range(n) if ids[i] not in pinned]
    # Counts resolve against the whole corpus, so a Fraction means the same share of the
    # same corpus whether or not any row is pinned; what shrinks is the pool they are
    # drawn from.
    n_dev = dev.resolve(n) if dev is not None else 0
    n_test = test.resolve(n) if test is not None else 0
    if n_dev + n_test > len(hashed):
        if not pinned_idx:
            raise PartitionError(f"split dev({n_dev}) + test({n_test}) exceeds corpus size {n}")
        raise PartitionError(f"split dev({n_dev}) + test({n_test}) exceeds the {len(hashed)} rows left "
                             f"to hash (corpus size {n}, {len(pinned_idx)} pinned to test)")
    for name, size, count in (("dev", dev, n_dev), ("test", test, n_test)):
        if size is not None and count == 0:
            # A split asked for that holds no row is a task graded, or tuned, on nothing.
            raise PartitionError(f"{name} split resolves to 0 rows of a corpus of {n} rows ({size!r})")
    ranked = sorted(hashed, key=lambda i: hashlib.sha256(ids[i].encode()).hexdigest())
    # Pinned rows lead the test split, and the hashed ones keep their ascending hash order
    # behind them.
    return ranked[:n_dev], pinned_idx + (ranked[len(ranked) - n_test:] if n_test else [])
