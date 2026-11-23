"""The workload's removal lists, and the one place their file format lives.

Which rows this workload does not serve is data, not code: one list per tenant of the
rows the calibration's own serving stack ran to the output guard, named by each row's
own id (``Dataset.row_id``) and applied by the drive plan before the arrival schedule
is drawn (``DataSource.removed_rows``). ``task.py`` reads the lists here so the ids
and their derivation live in one auditable file, and ``hls capped-requests --write``
reaches ``add_removed_rows`` here to add newly observed rows -- the command knows
which rows capped, this module knows where they are written and in what shape.

The file holds one entry per tenant::

    {"<tenant>": {"derivation": "<how the ids were obtained>",
                  "ids": ["<row id>", ...]}}

Each list is validated as it loads: ids sorted and free of repeats, so the workload
fingerprint has one value. The derivation strings are documentation for a reader, and
are written by hand rather than by the command.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping

_PATH = Path(__file__).with_name("removed_requests.json")


@lru_cache(maxsize=1)
def _load() -> dict:
    return json.loads(_PATH.read_text())


def removed_rows(tenant: str) -> tuple[str, ...]:
    """The rows removed from ``tenant``'s workload, sorted."""
    ids = _load()[tenant]["ids"]
    if list(ids) != sorted(set(ids)):
        raise ValueError(
            f"removed_requests.json {tenant} ids must be sorted and free of repeats, so "
            f"the workload fingerprint has one value")
    return tuple(ids)


def add_removed_rows(
    rows_by_tenant: Mapping[str, Iterable[str]],
) -> tuple[Path, dict[str, int]]:
    """Add rows to each tenant's list, in place, and return the file and how many rows
    each tenant gained.

    The list only grows: a row once observed at the cap stays removed, so the caller's
    rows are unioned with the ids already listed. The derivation strings are written by
    hand and left untouched; a tenant absent from the file is refused rather than
    invented, since a list with no derivation says nothing about where its ids came
    from."""
    doc = json.loads(_PATH.read_text())
    unknown = sorted(set(rows_by_tenant) - set(doc))
    if unknown:
        raise ValueError(
            f"{_PATH.name} carries no entry for tenant(s) {unknown} (known: "
            f"{sorted(doc)}); add one, with its derivation, before recording rows for it")
    added: dict[str, int] = {}
    for tenant, rows in rows_by_tenant.items():
        merged = sorted(set(doc[tenant]["ids"]) | set(rows))
        added[tenant] = len(merged) - len(doc[tenant]["ids"])
        doc[tenant]["ids"] = merged
    _PATH.write_text(json.dumps(doc, indent=2) + "\n")
    _load.cache_clear()
    return _PATH, added
