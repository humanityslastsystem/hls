"""The mount wall: which provisioned data each role's container may see, and where.

One convention for every role: the task model's weights at ``/data/weights``, each tenant's
rows at ``/data/tenants/<id>/<split>`` for the splits the role may see, and a tool bundle at
``/data/tools/<name>`` in the variant the role may load. The wall holds by construction:
what a role may not see is not mounted, so the test rows, and a bundle built from them, are
absent from the agent, the build sandbox and the artifact.

The artifact role mounts the dev rows too. The brief tells the contestant the dev rows are at
that path and the build sandbox has them, so a server may read them at serve time.
A tenant's dev reference, a measurement kept apart from the rows entry
(``hls.verify.dev_reference``), is published beside its dev rows: such a tenant's dev split mounts
from a view that holds a copy of the rows and the reference's file (``dev_view``), in the role's
own directory, which its stage declares among its copies (``dev_view_key``, ``dev_view_bytes``).

Every role also gets ``HF_HUB_OFFLINE=1`` and ``HF_DATASETS_OFFLINE=1``, so an engine never
downloads at run time, and ``HLS_WEIGHTS_DIR`` when it mounts the weights. The rows entry's
``ids.json``, which names the test rows, is never mounted: only its split directories are.
Each entry is copied to the node pinned to the ``ENTRY.json`` digest provision recorded, so
an entry published again under the same key is refused rather than mounted.

A stage declares the copies its roles make (``DataCopies``), and a role's entries are copied only
when its stage declared each of them with room for it. A copy lasts the job, and a later stage of
the job that mounts the same entry reuses it, so an allocation holds each entry its stages copy
once (``hls.pipeline.planner.allocation_request``).

The wall also holds where a container starts, whatever mounts a stage asks for:
``StageContext`` refuses a contestant's mount of anything in a cache entry copy that
``contestant_mount_problem`` names."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, NonNegativeInt

from hls.provision.model import Provisioned, RowsRef, ToolRef, WeightsRef, is_dev_tool_key
from hls.provision.rows import tenant_mount_path, tenant_subpath
from hls.store.cache import COPY_BLOCK_BYTES
from hls.substrate.container import Mount

#: Where the task model's weights mount.
WEIGHTS_TARGET = "/data/weights"
#: Under which tool bundles mount, one directory each.
TOOLS_ROOT = "/data/tools"
#: What each role may mount: the weights, and which splits of each tenant's rows. The
#: driver is trusted harness code and drives whichever split the run asks for; the
#: reference gets its inputs from the verify stage and mounts no rows. The oracle fill
#: (``hls.verify.fill``) runs the reference's engines on the test rows themselves, ahead of
#: the gradings, and on the dev rows for the dev reference (``hls.verify.dev_reference``), so it
#: mounts what the reference does and both splits. A calibration's capture and score roles
#: (``hls.verify.calibration_roles``) mount the weights and the test rows: they drive the test
#: split's sessions and key every captured reply against them.
ROLE_UNITS: dict[str, tuple[str, ...]] = {
    "artifact": ("weights", "dev"),
    "agent": ("weights", "dev"),
    "build": ("weights", "dev"),
    "driver": ("weights", "dev", "test"),
    "reference": ("weights",),
    "oracle": ("weights", "dev", "test"),
    "calibration": ("weights", "test"),
}
#: Which tool bundle variant each role loads. The artifact loads none; the reference loads the
#: full one, and so do the oracle fill, which runs the same references, and a calibration's
#: roles, whose pool members run the task's tools.
ROLE_TOOLS: dict[str, str] = {"agent": "dev", "build": "dev", "driver": "full", "reference": "full",
                              "oracle": "full", "calibration": "full"}
#: Variables every role gets, so no engine downloads at run time.
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}

#: The node-local copy of a cache entry: ``(kind, key, entry_sha256) -> path``, refusing an
#: entry whose ENTRY.json digest is not *entry_sha256*. Stages pass ``StageContext.materialize``.
Local = Callable[[str, str, str], Path]
#: A provisioned entry's record.
Ref = WeightsRef | RowsRef | ToolRef
#: What a stage declares a dev view under among its copies (``dev_view_key``).
DEV_VIEW = "dev-view"


def dev_view_key(role: str, tenant: str) -> str:
    """The name a stage declares role *role*'s view of tenant *tenant*'s dev split under among
    its copies (``DataCopies``)."""
    return f"{DEV_VIEW}/{role}/{tenant}"


def dev_view_bytes(rows: RowsRef, tenant: str, files: Mapping[str, str]) -> int:
    """The disk a view of tenant *tenant*'s dev split in *rows* takes (``dev_view``): the copy of
    the split as the provision record states it (``RowsRef.dev_disk_bytes``), and each of *files*
    in whole ``COPY_BLOCK_BYTES`` blocks with one block for its name. Raise ValueError when the
    record states no size for the tenant's dev split."""
    if tenant not in rows.dev_disk_bytes:
        raise ValueError(f"the provision record of {rows.key} states no size of tenant {tenant}'s dev split, which "
                         "its view copies: provision the task again")
    block = COPY_BLOCK_BYTES
    return rows.dev_disk_bytes[tenant] + sum(block * (-(-len(text.encode()) // block) + 1) for text in files.values())


def _views(role: str, provisioned: Provisioned,
           dev_files: Mapping[str, Mapping[str, str]]) -> dict[str, int]:
    """The dev views role *role* makes from *provisioned*, by the name its stage declares each
    under (``dev_view_key``), to the disk each takes (``dev_view_bytes``): one for each tenant of
    *dev_files* whose dev split the role sees. Raise ValueError as ``dev_view_bytes`` does."""
    rows = provisioned.rows
    if rows is None or "dev" not in ROLE_UNITS.get(role, ()):
        return {}
    return {dev_view_key(role, tenant): dev_view_bytes(rows, tenant, files)
            for tenant, files in dev_files.items() if tenant in rows.tenants}


def _entries(role: str, provisioned: Provisioned, *, weights: bool) -> list[tuple[str, Ref]]:
    """The cache entries role *role* mounts from *provisioned*, each with its kind, in mount
    order: the weights, the rows when it sees a split of them, and the tool bundle of its
    variant. Raise ValueError as ``data_mounts`` does."""
    if role not in ROLE_UNITS:
        raise ValueError(f"unknown role {role!r}; expected one of {sorted(ROLE_UNITS)}")
    units = ROLE_UNITS[role]
    found: list[tuple[str, Ref]] = []
    if "weights" in units and weights:
        if provisioned.weights is None:
            raise ValueError(f"role {role} mounts the task model's weights, which were not provisioned")
        found.append(("weights", provisioned.weights))
    if provisioned.rows is not None and any(s in units for s in ("dev", "test")):
        found.append(("rows", provisioned.rows))
    found += [("tools", tool) for tool in provisioned.tools if tool.variant == ROLE_TOOLS.get(role)]
    return found


def dev_view(dev_rows: Path, view: Path, files: Mapping[str, str]) -> Path:
    """*view*, made afresh: a copy of every file of the dev rows directory *dev_rows* at its own
    relative path, and beside them each of *files*, its name written with its text. A role that
    sees the tenant's dev split mounts it in place of the rows entry's directory, read-only. The
    dev split is small, and a copy leaves the verified node-local copy of the rows entry as it
    was. Raise ValueError for a name of *files* the dev rows already hold, and OSError for a file
    that cannot be copied or written."""
    clash = sorted(name for name in files if (dev_rows / name).exists())
    if clash:
        raise ValueError(f"the dev rows {dev_rows} already hold {clash}, which the dev reference publishes")
    if view.exists():
        shutil.rmtree(view)
    view.mkdir(parents=True)
    for path in sorted(dev_rows.rglob("*")):
        target = view / path.relative_to(dev_rows)
        if path.is_dir():
            target.mkdir(exist_ok=True)
        else:
            shutil.copyfile(path, target)
    for name, text in files.items():
        (view / name).write_text(text)
    return view


def data_mounts(role: str, provisioned: Provisioned, local: Local, *, declared: Mapping[str, int],
                weights: bool = True, dev_files: Mapping[str, Mapping[str, str]] | None = None,
                views: Path | None = None) -> tuple[list[Mount], dict[str, str]]:
    """The read-only mounts and the variables role *role* gets from *provisioned*.
    *local(kind, key, entry_sha256)* is the node-local copy of a cache entry, pinned to the
    digest *provisioned* recorded for it; it is called only for the entries the role mounts,
    so a stage copies to the node only what its role sees, and only once each of them is among
    the copies its stage *declared* with room for it (``DataCopies.data_copies``): the node's
    start check passed the disk of the declared copies, so one that is not declared, or takes more
    than declared (an entry published again since the plan), could find the node's disk short.
    Without *weights*, a role that may mount the task model's weights does not: the verify stage's
    role when no reference engine of it can load (``hls.verify.run.may_deploy``). *dev_files*
    holds, by tenant id, the dev reference files published beside that tenant's dev rows; a role
    that sees the dev split mounts such a tenant's from a view under *views* (``dev_view``).
    Raise ValueError for an unknown role, for a role that needs the task model's weights when none
    were provisioned, for a copy or a view its stage did not declare room for, before anything is
    copied, and for *dev_files* without *views*."""
    env = dict(OFFLINE_ENV)
    mounts: list[Mount] = []
    dev_files = dev_files or {}
    if dev_files and views is None:
        raise ValueError("dev reference files to publish beside the dev rows need a directory for their views")
    splits = [s for s in ("dev", "test") if s in ROLE_UNITS.get(role, ())]
    entries = _entries(role, provisioned, weights=weights)
    problem = _copies_problem({**{f"{kind}/{ref.key}": ref.disk_bytes for kind, ref in entries},
                               **_views(role, provisioned, dev_files)}, declared)
    if problem is not None:
        raise ValueError(f"{problem}: plan the instance again")
    for kind, ref in entries:
        copy = local(kind, ref.key, ref.entry_sha256)
        if isinstance(ref, WeightsRef):
            env["HLS_WEIGHTS_DIR"] = WEIGHTS_TARGET
            mounts.append(Mount(copy, WEIGHTS_TARGET))
        elif isinstance(ref, RowsRef):
            for tenant in ref.tenants:
                for split in splits:
                    source = copy / tenant_subpath(tenant) / split
                    if split == "dev" and tenant in dev_files:
                        assert views is not None
                        source = dev_view(source, views / tenant, dev_files[tenant])
                    mounts.append(Mount(source, tenant_mount_path(tenant, split)))
        else:
            mounts.append(Mount(copy, f"{TOOLS_ROOT}/{ref.name}"))
    return mounts, env


def _copies_problem(copies: Mapping[str, int], declared: Mapping[str, int]) -> str | None:
    """Why the copies *copies* (``copied_entries``) do not fit the copies *declared*: those not
    among them, and those that take more bytes than declared; None when they fit."""
    problems = []
    missing = [entry for entry in copies if entry not in declared]
    if missing:
        problems.append(f"the plan made no room on the node for a copy of {', '.join(missing)}")
    problems += [f"the copy of {entry} takes {size} bytes, more than the {declared[entry]} the plan made room for "
                 "(the entry was published again since)"
                 for entry, size in copies.items() if entry in declared and size > declared[entry]]
    return "; ".join(problems) or None


def copied_entries(roles: Iterable[str], provisioned: Provisioned, *, weights: bool = True,
                   dev_files: Mapping[str, Mapping[str, str]] | None = None) -> dict[str, int]:
    """The node-local copies of the cache entries roles *roles* of one stage mount from
    *provisioned*, ``<kind>/<key>`` of each to the bytes its copy takes as the provision record
    states it (``disk_bytes``): each entry once, as a stage copies an entry once whichever of its
    roles mounts it (``StageContext.materialize``); and each role's dev views of the tenants of
    *dev_files*, the dev reference files the plan pins (``dev_view_key``, ``dev_view_bytes``). The
    one derivation of a stage's copies: the planner declares them (``DataCopies``) from the cache's
    provision record (``hls.stages.provision.cached_provision``), and ``data_mounts`` holds the
    stage's own record to them before it copies anything. *weights* as for ``data_mounts``. Raise
    ValueError as it does."""
    roles = list(roles)
    copies = {f"{kind}/{ref.key}": ref.disk_bytes for role in roles
              for kind, ref in _entries(role, provisioned, weights=weights)}
    for role in roles:
        copies.update(_views(role, provisioned, dev_files or {}))
    return copies


def copies_gib(copies: Mapping[str, int]) -> int:
    """The disk, in whole GiB, that the node-local copies *copies* (``copied_entries``) take
    together."""
    return -(-sum(copies.values()) // 2**30)


class DataCopies(BaseModel):
    """The parameters of a stage whose roles mount the task's provisioned data: the node-local
    copies it makes."""

    #: The cache entries the stage's roles copy to the node, ``<kind>/<key>`` of each to the bytes
    #: its copy takes (``copied_entries`` of the provision record the plan read from the cache,
    #: ``hls.stages.provision.cached_provision``), and the dev views its roles make
    #: (``dev_view_key``). The stage's disk holds them (``data_gib``) beside
    #: the rest it needs; an allocation holds each entry its stages copy once, since a copy lasts
    #: the job; and the stage copies no entry that is not among them (``data_mounts``).
    data_copies: dict[str, NonNegativeInt]

    @property
    def data_gib(self) -> int:
        """GiB of node-local disk the stage's copies take (``copies_gib``)."""
        return copies_gib(self.data_copies)


def contestant_mount_problem(kind: str, key: str, rel: PurePosixPath) -> str | None:
    """Why a contestant's container may not mount the path *rel* inside the node-local copy
    of cache entry *kind*/*key*, worded to follow "is in"; None when it may. A contestant may
    see a weights entry whole, a rows entry's ``tenants/<id>/dev`` directories and what is in
    them (never the test split, nor ``ids.json``, which names the test rows), and a tool
    bundle's dev variant; no other kind of entry."""
    if kind == "weights":
        return None
    if kind == "rows":
        parts = rel.parts
        # A tenant's rows lie at tenant_subpath(<id>) = tenants/<id>, one directory per split.
        if len(parts) >= 3 and (parts[0], parts[2]) == ("tenants", "dev"):
            return None
        return f"rows entry {key}, where a contestant may see only tenants/<id>/dev"
    if kind == "tools":
        return None if is_dev_tool_key(key) else f"tools entry {key}, which is not a dev bundle"
    return f"{kind} entry {key}, a kind no contestant may mount"
