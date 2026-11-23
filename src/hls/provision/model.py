"""Cache keys of provisioned data, and the records the provision stage reads and publishes.

Three kinds of cache entry come out of provisioning:
- ``weights/<org>+<name>@<sha>``: a model snapshot at one hub commit. The key holds the
  resolved commit, never a branch name, so an entry never changes meaning when the branch
  moves. ``+`` stands for ``/`` because a key is one store segment and the hub forbids
  ``+`` in repo ids, so the mapping cannot collide.
- ``rows/rows-<16hex>``: every tenant's dev and test rows, keyed by the digest
  ``hls.provision.rows.provisioning_digest`` computes, so the same task partitions the same rows.
- ``tools/<name>-<identity>-full`` and ``tools/<name>-<identity>-dev-<12hex>``: the two
  variants of a task's tool bundle; the dev one is keyed by the partition too.

``Resolved`` is what the resolve role works out before anything is fetched. ``Provisioned``
is the stage's output, ``provisioned.json``: each entry's key, the digest of its ``ENTRY.json``,
which later stages materialize and pin, and the disk its node-local copy takes
(``hls.store.cache.copy_disk_bytes``), which the stages that copy it declare."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from hls.store.base import check_segment
from hls.store.digest import Sha256Hex

# A hub repo id: "<name>" or "<org>/<name>".
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
# A hub commit id.
_REVISION = re.compile(r"[0-9a-f]{40}")
#: The form of a rows key (``rows_key``), as a regular expression.
ROWS_KEY_PATTERN = r"rows-[0-9a-f]{16}"
_ROWS_KEY = re.compile(ROWS_KEY_PATTERN)
_DEV_DIGEST = re.compile(r"[0-9a-f]{12}")
# The key of a dev bundle, as ``tool_key`` makes it; a full one ends in "-full" instead.
_DEV_TOOL_KEY = re.compile(r".+-dev-[0-9a-f]{12}")


def check_repo(repo: str) -> str:
    """Return *repo* if it is a hub repo id, else raise ValueError."""
    if not _REPO.fullmatch(repo):
        raise ValueError(f"not a hub repo id: {repo!r}")
    return repo


def is_revision(value: str) -> bool:
    """Whether *value* is a hub commit id (40 lowercase hex), not a branch or tag name."""
    return bool(_REVISION.fullmatch(value))


def check_revision(revision: str) -> str:
    """Return *revision* if it is a hub commit id, else raise ValueError."""
    if not is_revision(revision):
        raise ValueError(f"not a hub commit id: {revision!r}")
    return revision


def weights_key(repo: str, revision: str) -> str:
    """The cache key of *repo*'s snapshot at commit *revision*."""
    return check_segment(f"{check_repo(repo).replace('/', '+')}@{check_revision(revision)}")


def rows_key(digest: str) -> str:
    """The cache key of the rows whose provisioning digest is *digest*."""
    return check_rows_key(f"rows-{digest}")


def check_rows_key(key: str) -> str:
    """Return *key* if it is a rows key, else raise ValueError."""
    if not _ROWS_KEY.fullmatch(key):
        raise ValueError(f"not a rows key: {key!r}")
    return key


def check_dev_digest(digest: str) -> str:
    """Return *digest* if it is a partition digest (12 lowercase hex), else raise ValueError."""
    if not _DEV_DIGEST.fullmatch(digest):
        raise ValueError(f"not a partition digest: {digest!r}")
    return digest


def tool_key(name: str, identity: str, variant: Literal["full", "dev"], dev_digest: str | None = None) -> str:
    """The cache key of one variant of a tool bundle. The dev variant needs *dev_digest*,
    the digest of the partition it was built from; the full one takes none."""
    if (variant == "dev") != (dev_digest is not None):
        raise ValueError("a dev bundle key needs the partition digest, and a full one takes none")
    suffix = "full" if dev_digest is None else f"dev-{check_dev_digest(dev_digest)}"
    return check_segment(f"{check_segment(name)}-{check_segment(identity)}-{suffix}")


def is_dev_tool_key(key: str) -> bool:
    """Whether *key* is the cache key of a tool bundle's dev variant: one that ends in
    ``-dev-<12hex>``, where the full variant's ends in ``-full``."""
    return bool(_DEV_TOOL_KEY.fullmatch(key))


Repo = Annotated[str, AfterValidator(check_repo)]
Revision = Annotated[str, AfterValidator(check_revision)]
DevDigest = Annotated[str, AfterValidator(check_dev_digest)]


class ModelRevision(BaseModel):
    """A model repo at one commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Hub repo id.
    repo: Repo
    #: Commit id.
    revision: Revision


class ToolSource(BaseModel):
    """A task's tool bundle as its assets declare it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``ToolAssets.name``, which is also its mount name.
    name: str
    #: ``ToolAssets.identity()``, the digest of its pinned sources.
    identity: str


class Resolved(BaseModel):
    """What a task needs provisioned, with every revision resolved to a commit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task model.
    weights: ModelRevision
    #: Key of the rows entry; None when no tenant has a data source.
    rows_key: str | None
    #: Ids of the tenants with a data source, in task order.
    tenants: list[str]
    #: The tool bundle; None when the task declares no assets.
    tools: ToolSource | None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if self.rows_key is not None:
            check_rows_key(self.rows_key)
        if (self.rows_key is None) != (not self.tenants):
            raise ValueError("a rows key goes with at least one tenant, and only then")
        if self.tools is not None and self.rows_key is None:
            raise ValueError("a tool bundle needs the rows it is partitioned by")
        return self


class WeightsRef(BaseModel):
    """A published weights entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Hub repo id.
    repo: Repo
    #: Commit id.
    revision: Revision
    #: Cache key, ``weights_key(repo, revision)``.
    key: str
    #: Digest of the entry's ENTRY.json.
    entry_sha256: Sha256Hex
    #: Bytes of disk a node-local copy of the entry takes at most (``copy_disk_bytes``).
    disk_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _key(self) -> Self:
        if self.key != weights_key(self.repo, self.revision):
            raise ValueError(f"key {self.key!r} is not the key of {self.repo}@{self.revision}")
        return self


class RowsRef(BaseModel):
    """A published rows entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Cache key, ``rows-<16hex>``.
    key: Annotated[str, AfterValidator(check_rows_key)]
    #: Digest of the entry's ENTRY.json.
    entry_sha256: Sha256Hex
    #: Bytes of disk a node-local copy of the entry takes at most (``copy_disk_bytes``).
    disk_bytes: int = Field(ge=0)
    #: Tenant ids whose rows it holds, under ``tenants/<id>/{dev,test}``.
    tenants: list[str]
    #: Bytes of disk a copy of each tenant's dev split takes, by tenant id
    #: (``subtree_disk_bytes``): what a dev view copies (``hls.provision.mounts.dev_view``).
    dev_disk_bytes: dict[str, int] = Field(default_factory=dict)


class ToolRef(BaseModel):
    """A published tool bundle variant."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``ToolAssets.name``.
    name: str
    #: ``ToolAssets.identity()``.
    identity: str
    #: Which variant: ``dev`` for roles that may see only the dev rows, ``full`` for the rest.
    variant: Literal["full", "dev"]
    #: Digest of the partition a ``dev`` bundle was built from; None for ``full``.
    dev_digest: DevDigest | None
    #: Cache key, ``tool_key(name, identity, variant, dev_digest)``.
    key: str
    #: Digest of the entry's ENTRY.json.
    entry_sha256: Sha256Hex
    #: Bytes of disk a node-local copy of the entry takes at most (``copy_disk_bytes``).
    disk_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def _key(self) -> Self:
        if self.key != tool_key(self.name, self.identity, self.variant, self.dev_digest):
            raise ValueError(f"key {self.key!r} is not the key of the {self.variant} bundle "
                             f"{self.name} ({self.identity})")
        return self


class Provisioned(BaseModel):
    """The provision stage's output: every entry it published or found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The task model; None only for a provisioning asked for rows alone.
    weights: WeightsRef | None
    #: The rows; None when no tenant has a data source.
    rows: RowsRef | None
    #: Tool bundle variants, both or none.
    tools: list[ToolRef]

    @model_validator(mode="after")
    def _tools(self) -> Self:
        if self.tools and self.rows is None:
            raise ValueError("a tool bundle needs the rows it is partitioned by")
        if sorted(t.variant for t in self.tools) not in ([], ["dev", "full"]):
            raise ValueError("a tool bundle is provisioned as both variants or not at all")
        return self
