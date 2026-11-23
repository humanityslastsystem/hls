"""Dataset rows: which tenants a task provisions, the digest that keys their rows, and the
writer that partitions each corpus into dev and test rows.

Rows are assigned to a split by a hash of their own identity
(``hls.spec.datasource.partition_by_hash``); the task seed plays no part."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from hls.spec.datasource import PartitionError, partition_by_hash

log = logging.getLogger(__name__)

#: Fetches a data source's corpus: ``(data_source) -> datasets.Dataset``.
CorpusFor = Callable[[Any], Any]
#: Fetches an explicit dataset's corpus: ``(dataset) -> datasets.Dataset``.
CorpusOf = Callable[[Any], Any]


def tenant_subpath(tenant_id: str) -> str:
    """The subpath a tenant's unit occupies, under the provisioning root on the host and
    under ``/data`` in a container: ``tenants/<id>`` for one tenant or for five."""
    return f"tenants/{tenant_id}"


def tenant_mount_path(tenant_id: str, split: str) -> str:
    """The container path a tenant's *split* rows mount at, e.g. ``/data/tenants/t0/dev``.

    Every name for that path -- the bind mounts and the brief the agent reads -- comes
    from here, so what is mounted and what is documented cannot drift apart."""
    return f"/data/{tenant_subpath(tenant_id)}/{split}"


def entry_rows(entry: Path, tenants: Sequence[str] | None = None) -> list[dict]:
    """Every row of both splits of each of *tenants*, every tenant when None, that the rows
    entry at *entry* holds (``write_rows``)."""
    names = sorted(p.name for p in (entry / "tenants").iterdir() if p.is_dir()) if tenants is None else tenants
    rows: list[dict] = []
    for tenant in names:
        for split in ("dev", "test"):
            path = entry / tenant_subpath(tenant) / split / "rows.json"
            if path.is_file():
                rows += json.loads(path.read_text())
    return rows


def write_dev_reference(ds, dev_dir: Path) -> None:
    """Publish a data source's ``dev_reference`` files beside its dev rows.

    The rows are raw corpus fields; whatever the task's prompt template wraps around
    them lives in task code and is invisible on disk, so a prompt element a contestant
    cannot infer from a row has to be handed over explicitly. Written from the task's
    own constant, so it cannot drift from what the driver sends."""
    for name, contents in (getattr(ds, "dev_reference", None) or {}).items():
        (dev_dir / name).write_text(contents)


def dev_reference_digest(ds) -> str | None:
    """Digest of a data source's ``dev_reference`` files, or None when it publishes none.

    Part of the provisioning identity: the files are written only on a provisioning
    miss, so without this a changed or newly added reference file would leave a
    cached provisioning serving the old contents."""
    ref = getattr(ds, "dev_reference", None) or {}
    if not ref:
        return None
    h = hashlib.sha256()
    for name in sorted(ref):
        for part in (name, ref[name]):
            h.update(part.encode())
            h.update(b"\0")
    return h.hexdigest()[:16]


def provisioning_digest(tenant_units: list[dict]) -> str:
    """Content-addressed digest of one provisioning's rows: every unit's identity, sorted by
    unit id so authoring order does not matter. The model is no input: the rows are read off
    the datasets alone, with no tokenizer, chat template or other file of the model.
    ``tenant_units`` holds dicts with keys unit_id, corpus_id, dev_corpus_id,
    dev_reference_digest, dev, test, removed_rows, mount_subpath; the two corpus
    digests are ``None`` for a unit without a dev corpus or reference files.

    Split sizes are serialized via ``model_dump_json``, a stable canonical form for a
    frozen pydantic model, rather than ``__str__``. The removal list is hashed rather
    than spelled out, since it runs to hundreds of ids."""
    parts = []
    for u in sorted(tenant_units, key=lambda x: x["unit_id"]):
        removed = hashlib.sha256("\n".join(u["removed_rows"]).encode()).hexdigest()[:16]
        parts.append(
            f"{u['unit_id']}:{u['corpus_id']}:devcorpus={u['dev_corpus_id']}"
            f":devref={u['dev_reference_digest']}"
            f":dev={u['dev'].model_dump_json()}:test={u['test'].model_dump_json()}"
            f":removed={removed}"
            f":sub={u['mount_subpath']}"
        )
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def unit_descriptions(task) -> list[dict]:
    """One description per tenant that has a data source, in the shape
    ``provisioning_digest`` and ``write_rows`` read. Tenants without one are skipped."""
    out = []
    for t in task.effective_tenants():
        if t.data_source is None:
            continue
        ds = t.data_source
        out.append({
            "unit_id": t.id,
            "data_source": ds,
            "corpus_id": ds.dataset.corpus_id(),
            # A per-split dev corpus (gsm8k train -> dev, platinum -> test); None for
            # the common single-corpus partition. getattr keeps duck-typed data_source
            # doubles (which omit the field) working.
            "dev_dataset": (dev_ds := getattr(ds, "dev_dataset", None)),
            "dev_corpus_id": dev_ds.corpus_id() if dev_ds is not None else None,
            "dev_reference_digest": dev_reference_digest(ds),
            "dev": ds.split.dev,
            "test": ds.split.test,
            # The task's removal list, applied to the dev rows. getattr for the same
            # duck-typed doubles.
            "removed_rows": tuple(getattr(ds, "removed_rows", ())),
            "mount_subpath": tenant_subpath(t.id),
        })
    return out


def row_ids(dataset, corpus) -> list[str]:
    """Every corpus row's identity (``Dataset.row_id``), in corpus order; what the
    partition hashes."""
    return [dataset.row_id(row) for row in corpus]


def pinned_test(dataset, corpus) -> list[str]:
    """The rows this corpus declares into the test split (``Dataset.forced_split``), read
    off the rows rather than their ids, because the declaration is a property of the row.
    Empty for every dataset that does not override it."""
    return [dataset.row_id(row) for row in corpus if dataset.forced_split(row) == "test"]


def drop_listed_rows(rows, dataset, removed: tuple[str, ...], *, unit_id: str):
    """The dev partition minus the task's removal list (``DataSource.removed_rows``),
    by ``Dataset.row_id``.

    The list names rows whose reply ran to the output cap under the calibration's own
    serving stack. The driver drops them from the test split as it draws a schedule,
    the one place test rows become traffic; dev rows become traffic only in the
    contestant's hands, so the copy on disk is the one place to screen them, and a
    contestant then tunes against the same workload shape it is graded on. The list
    holds the dev rows a calibration serving ran to the cap."""
    if not removed:
        return rows
    listed = set(removed)
    keep = [i for i, rid in enumerate(row_ids(dataset, rows)) if rid not in listed]
    if len(keep) < len(rows):
        log.info("provisioning: unit %s -- %d of %d dev rows dropped by the removal "
                 "list", unit_id, len(rows) - len(keep), len(rows))
    return rows.select(keep)


def dedup(ids: list[str]) -> list[str]:
    """First-seen order, no repeats: two tenants over one corpus name a row once."""
    return list(dict.fromkeys(ids))


def dev_digest(dev_row_ids: list[str], test_row_ids: list[str]) -> str:
    """Digest of the whole partition, carried in the dev bundle's key.

    Dev because the bundle is built from the dev rows. Test as well because a builder
    also checks that the dev bundle reveals nothing about a test row, and that check has
    to run again when the test rows move even if the dev rows do not -- raising a task's
    test count does exactly that."""
    h = hashlib.sha256()
    for ids in (dev_row_ids, test_row_ids):
        for rid in sorted(ids):
            h.update(rid.encode())
            h.update(b"\0")
        h.update(b"\1")   # separates the two groups, so a row changing sides changes the digest
    return h.hexdigest()[:12]


def declared_assets(task):
    """The task's ``ToolAssets``, or None when it declares no agent or no assets.
    getattr keeps duck-typed task doubles (which omit the fields) working."""
    return getattr(getattr(task, "agent", None), "assets", None)


def write_rows(unit_descs: list[dict], base: Path, *, corpus_for: CorpusFor,
               corpus_of: CorpusOf) -> tuple[list[str], list[str]]:
    """Write every unit's ``<mount_subpath>/{dev,test}/rows.json`` and dev reference files
    under *base*, and return the (dev, test) row ids in write order.

    Corpora sharing one ``corpus_id()`` are fetched once, by *corpus_for* (a data source's
    corpus) or *corpus_of* (a ``dev_dataset``'s); the caller binds whatever credential they
    need. A unit with its own dev corpus ranks each corpus alone, so either split may cover
    its whole corpus. The dev split is screened by the removal list before anything reads
    it, so the rows written and the ids returned agree. A PartitionError names the tenant."""
    corpus_cache: dict[str, Any] = {}
    dev_ids: list[str] = []
    test_ids: list[str] = []
    for u in unit_descs:
        ds = u["data_source"]
        cid = u["corpus_id"]
        if cid not in corpus_cache:
            log.info("fetching corpus %s", cid)
            corpus_cache[cid] = corpus_for(ds)
        corpus = corpus_cache[cid]
        try:
            if u["dev_dataset"] is not None:
                dev_ds, dev_cid = u["dev_dataset"], u["dev_corpus_id"]
                if dev_cid not in corpus_cache:
                    log.info("fetching dev corpus %s", dev_cid)
                    corpus_cache[dev_cid] = corpus_of(dev_ds)
                dev_corpus = corpus_cache[dev_cid]
                dev_idx, _ = partition_by_hash(row_ids(dev_ds, dev_corpus), dev=u["dev"])
                _, test_idx = partition_by_hash(row_ids(ds.dataset, corpus), test=u["test"],
                                                pinned_test=pinned_test(ds.dataset, corpus))
                dev_rows, test_rows = dev_corpus.select(dev_idx), corpus.select(test_idx)
            else:
                dev_idx, test_idx = partition_by_hash(row_ids(ds.dataset, corpus), dev=u["dev"], test=u["test"],
                                                      pinned_test=pinned_test(ds.dataset, corpus))
                dev_rows, test_rows = corpus.select(dev_idx), corpus.select(test_idx)
        except PartitionError as e:
            raise PartitionError(f"tenant {u['unit_id']}: {e}") from e
        screened_by = u["dev_dataset"] if u["dev_dataset"] is not None else ds.dataset
        dev_rows = drop_listed_rows(dev_rows, screened_by, u["removed_rows"], unit_id=u["unit_id"])
        dev_list, test_list = dev_rows.to_list(), test_rows.to_list()
        dev_ids += [(u["dev_dataset"] or ds.dataset).row_id(r) for r in dev_list]
        test_ids += [ds.dataset.row_id(r) for r in test_list]
        dev_dir = base / u["mount_subpath"] / "dev"
        test_dir = base / u["mount_subpath"] / "test"
        dev_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        (dev_dir / "rows.json").write_text(json.dumps(dev_list))
        (test_dir / "rows.json").write_text(json.dumps(test_list))
        write_dev_reference(ds, dev_dir)
    return dev_ids, test_ids
