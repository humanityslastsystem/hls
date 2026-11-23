"""An image archive read as trees from its layers, never extracted: how the package's checks read
a submission image and its base on a substrate whose image files are OCI image archives
(Docker's), where no image is a tree on the host.

An archive is an OCI image layout in a tar: ``index.json`` naming one manifest, the manifest
naming a config and layers, and every one of them a blob under ``blobs/sha256``. Each digest on
the way is checked, so what is read is the image that was pinned: the manifest's against the
digest the pin recorded, the config's against the manifest's, and each layer's uncompressed bytes
against the config's ``rootfs.diff_ids`` as they stream (``read_layer``). A layer is a tar, plain
or gzip-compressed; it is applied as the OCI image spec applies a changeset (``apply``):
``.wh.<name>`` removes ``<name>`` of the layers below with everything under it, ``.wh..wh..opq``
empties its directory of what the layers below hold there, and an entry replaces the one below
it, a non-directory over a directory with everything under that.

A name in a layer is data. It is never joined to a host path, and a name that climbs out of the
image (``..``), a hard link to no file the layer holds before it, an entry whose directory no layer
holds, and a layer of a kind this module does not read are refused (``LayerError``).

The submission image the package stage builds on such a substrate is its base's layers with one
more, the install's (``installed_layer`` proves it of the two archives' configs), so what the
install added or changed is that layer alone, and the base's trees come from the base's own
archive (``base_and_image``). The first check that reads the base leaves its tree, with the
digest of the manifest it read, as a listing (``write_listing``); the next reads that instead of
the archive's layers (``read_listing``), refusing one of another manifest."""

from __future__ import annotations

import gzip
import hashlib
import json
import stat
import tarfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from hls.util import EntryKind

#: The layer media types read, each with whether its tar is gzip-compressed.
LAYER_TYPES = {
    "application/vnd.oci.image.layer.v1.tar": False,
    "application/vnd.oci.image.layer.v1.tar+gzip": True,
    "application/vnd.docker.image.rootfs.diff.tar": False,
    "application/vnd.docker.image.rootfs.diff.tar.gzip": True,
}
#: The name of a whiteout's prefix and of the marker that makes a directory opaque.
WHITEOUT, OPAQUE = ".wh.", ".wh..wh..opq"
# Bytes read at a time.
_CHUNK = 1 << 20

__all__ = ["LAYER_TYPES", "Change", "Image", "Layer", "LayerError", "Meta", "Tree", "apply", "base_and_image",
           "installed_layer", "open_image", "read_layer", "read_listing", "write_listing"]


class LayerError(ValueError):
    """An archive or a layer that is not the image it must be, or not one this module reads."""


@dataclass(frozen=True)
class Meta:
    """An entry of an image, as its layer's tar header records it."""

    #: ``file``, ``directory``, ``link`` or ``special`` (``hls.util.EntryKind``).
    kind: EntryKind
    #: The permission bits, set-user-ID, set-group-ID and sticky among them.
    mode: int
    #: The size of a file; 0 for anything else.
    size: int = 0
    #: Where a link points; None for anything else.
    target: str | None = None
    #: The sha256 of a file's bytes; None for anything else.
    sha256: str | None = None


@dataclass(frozen=True)
class Change:
    """One member of a layer: an entry at *path* (``meta``), or a whiteout of what the layers
    below hold at *path* (``meta`` None): the entry with everything under it, or with *opaque*
    only what is under the directory."""

    #: The path from the image's root, without a leading slash; ``""`` is the root.
    path: str
    meta: Meta | None
    opaque: bool = False


@dataclass(frozen=True)
class Layer:
    """A layer of an image: its blob in the archive, its media type and its diff_id."""

    blob: str
    media_type: str
    diff_id: str


@dataclass(frozen=True)
class Image:
    """An image of an archive, its digests checked (``open_image``)."""

    archive: Path
    manifest_digest: str
    layers: tuple[Layer, ...]


@dataclass
class Tree:
    """The entries of an image after some of its layers, by path from its root (``""``, a
    directory), with each directory's names."""

    entries: dict[str, Meta] = field(default_factory=lambda: {"": Meta("directory", 0o755)})
    names: dict[str, set[str]] = field(default_factory=lambda: {"": set()})

    def get(self, path: str) -> Meta | None:
        return self.entries.get(path)

    def copy(self) -> Tree:
        return Tree(dict(self.entries), {k: set(v) for k, v in self.names.items()})

    def _drop_below(self, path: str) -> None:
        pending = [path]
        while pending:
            here = pending.pop()
            for name in self.names.pop(here, set()):
                child = _join(here, name)
                del self.entries[child]
                pending.append(child)

    def remove(self, path: str) -> None:
        """Remove the entry at *path* with everything under it; nothing when there is none."""
        if path not in self.entries or path == "":
            return
        self._drop_below(path)
        del self.entries[path]
        self.names[_parent(path)].discard(_name(path))

    def clear(self, path: str) -> None:
        """Remove everything under the directory at *path*, keeping the directory."""
        if path in self.names:
            self._drop_below(path)
            self.names[path] = set()

    def put(self, path: str, meta: Meta) -> None:
        """Set the entry at *path*: a directory over a directory keeps what is under it; anything
        else replaces what was there with everything under it. Raise LayerError when the layers
        hold no directory at *path*'s parent."""
        if path == "":
            if meta.kind != "directory":
                raise LayerError("a layer holds a root that is no directory")
            self.entries[""] = meta
            return
        parent = _parent(path)
        if parent not in self.names:
            raise LayerError(f"a layer holds /{path}, and no layer holds a directory at /{parent}")
        was = self.entries.get(path)
        if was is not None and not (was.kind == "directory" and meta.kind == "directory"):
            self.remove(path)
        self.entries[path] = meta
        self.names[parent].add(_name(path))
        if meta.kind == "directory":
            self.names.setdefault(path, set())


def _join(parent: str, name: str) -> str:
    return f"{parent}/{name}" if parent else name


def _parent(path: str) -> str:
    return path.rpartition("/")[0]


def _name(path: str) -> str:
    return path.rpartition("/")[2]


def _member_path(name: str) -> str:
    """The path from the image's root a tar member's *name* means: no leading slash, no empty or
    ``.`` component. Raise LayerError for a name that climbs (``..``)."""
    parts = [p for p in name.split("/") if p not in ("", ".")]
    if ".." in parts:
        raise LayerError(f"a layer names {name!r}, which climbs out of the image")
    return "/".join(parts)


def _blob(digest: object) -> str:
    if not isinstance(digest, str):
        raise LayerError(f"the archive names a blob by {digest!r}, which is no sha256 digest")
    algorithm, _, hexdigest = digest.partition(":")
    if algorithm != "sha256" or len(hexdigest) != 64 or not all(c in "0123456789abcdef" for c in hexdigest):
        raise LayerError(f"the archive names a blob by {digest!r}, which is no sha256 digest")
    return f"blobs/sha256/{hexdigest}"


def _read(tar: tarfile.TarFile, name: str) -> bytes:
    try:
        member = tar.getmember(name)
    except KeyError:
        raise LayerError(f"the archive holds no {name}") from None
    f = tar.extractfile(member) if member.isfile() else None
    if f is None:
        raise LayerError(f"the archive's {name} is no file")
    return f.read()


def _checked(tar: tarfile.TarFile, digest: object) -> bytes:
    data = _read(tar, _blob(digest))
    if "sha256:" + hashlib.sha256(data).hexdigest() != digest:
        raise LayerError(f"the archive's blob {digest} holds other bytes")
    return data


def open_image(archive: Path, manifest_digest: str) -> Image:
    """The image of *archive* whose manifest has the digest *manifest_digest* (``sha256:<hex>``):
    its manifest and config read and checked, its layers listed. Raise LayerError for an archive
    that names another manifest or several, a blob that is not the one its digest names, or a
    layer of a kind not read (``LAYER_TYPES``)."""
    with tarfile.open(archive) as tar:
        index = json.loads(_read(tar, "index.json"))
        manifests = index.get("manifests") or []
        if len(manifests) != 1 or manifests[0].get("digest") != manifest_digest:
            raise LayerError(f"the archive {archive} names {[m.get('digest') for m in manifests]}, not the manifest "
                             f"{manifest_digest} alone")
        manifest = json.loads(_checked(tar, manifest_digest))
        config = json.loads(_checked(tar, (manifest.get("config") or {}).get("digest")))
    diff_ids = (config.get("rootfs") or {}).get("diff_ids")
    layers = manifest.get("layers")
    if not isinstance(diff_ids, list) or not isinstance(layers, list) or len(diff_ids) != len(layers):
        raise LayerError(f"the archive {archive}'s manifest names {len(layers or [])} layers and its config "
                         f"{len(diff_ids or [])} diff_ids")
    read: list[Layer] = []
    for layer, diff_id in zip(layers, diff_ids, strict=True):
        if layer.get("mediaType") not in LAYER_TYPES:
            raise LayerError(f"the archive {archive} holds a layer of type {layer.get('mediaType')!r}, which is "
                             "not read")
        _blob(diff_id)
        read.append(Layer(_blob(layer["digest"]), layer["mediaType"], diff_id))
    return Image(archive, manifest_digest, tuple(read))


class _Hashing:
    """A stream that hashes every byte read from the stream it wraps."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.digest.update(data)
        return data


class _Member:
    """A regular file's bytes in a layer, hashed as they are read, by its reader or after it."""

    def __init__(self, raw: BinaryIO) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.size = 0

    def read(self, size: int = -1) -> bytes:
        data = self.raw.read(size)
        self.digest.update(data)
        self.size += len(data)
        return data

    def finish(self) -> str:
        while self.read(_CHUNK):
            pass
        return self.digest.hexdigest()


def _kind(member: tarfile.TarInfo) -> EntryKind:
    if member.isreg():
        return "file"
    if member.isdir():
        return "directory"
    if member.issym():
        return "link"
    return "special"


def read_layer(image: Image, layer: Layer, *,
               reader: Callable[[str, BinaryIO], None] | None = None) -> list[Change]:
    """The members of *layer* of *image*, in the order of its tar, each regular file's bytes
    hashed, and the whole layer's uncompressed bytes checked against its diff_id. *reader*, when
    given, is handed each regular file's path and its bytes as a stream, to read as far as it
    likes before the file is hashed to its end. Raise LayerError for a layer that is not the one
    its diff_id names, a name that climbs, or a hard link to no file the layer holds before it."""
    gzipped = LAYER_TYPES[layer.media_type]
    changes: list[Change] = []
    files: dict[str, Meta] = {}
    with tarfile.open(image.archive) as outer:
        try:
            blob = outer.getmember(layer.blob)
        except KeyError:
            raise LayerError(f"the archive {image.archive} holds no {layer.blob}") from None
        raw = outer.extractfile(blob) if blob.isfile() else None
        if raw is None:
            raise LayerError(f"the archive {image.archive}'s {layer.blob} is no file")
        stream: BinaryIO = gzip.GzipFile(fileobj=raw, mode="rb") if gzipped else raw  # type: ignore[assignment]
        hashing = _Hashing(stream)
        try:
            with tarfile.open(fileobj=hashing, mode="r|", encoding="utf-8",  # type: ignore[call-overload]
                              errors="surrogateescape") as tar:
                for member in tar:
                    changes.append(_change(tar, member, files, reader))
            while hashing.read(_CHUNK):
                pass
        except (OSError, EOFError, tarfile.TarError) as e:
            raise LayerError(f"the layer {layer.diff_id} of {image.archive} cannot be read: {e}") from e
    if "sha256:" + hashing.digest.hexdigest() != layer.diff_id:
        raise LayerError(f"the layer {layer.blob} of {image.archive} is not the layer {layer.diff_id}")
    return changes


def _change(tar: tarfile.TarFile, member: tarfile.TarInfo, files: dict[str, Meta],
            reader: Callable[[str, BinaryIO], None] | None) -> Change:
    path = _member_path(member.name)
    name = _name(path)
    if name == OPAQUE:
        return Change(_parent(path), None, opaque=True)
    if name.startswith(WHITEOUT):
        return Change(_join(_parent(path), name[len(WHITEOUT):]), None)
    mode = stat.S_IMODE(member.mode)
    if member.islnk():
        target = files.get(_member_path(member.linkname))
        if target is None:
            raise LayerError(f"a layer's hard link /{path} names {member.linkname!r}, no file before it")
        meta = target
    elif member.isreg():
        f = tar.extractfile(member)
        assert f is not None
        data = _Member(f)
        if reader is not None:
            reader(path, data)  # type: ignore[arg-type]
        sha256 = data.finish()
        meta = Meta("file", mode, data.size, sha256=sha256)
        files[path] = meta
    else:
        kind = _kind(member)
        meta = Meta(kind, mode, target=member.linkname if kind == "link" else None)
    return Change(path, meta)


def apply(tree: Tree, changes: Iterable[Change]) -> None:
    """Apply one layer's *changes* to *tree*, the layers below it: its whiteouts first, which
    remove what the layers below hold and none of the layer's own entries, then its entries,
    directories before what they hold."""
    changes = list(changes)
    for change in changes:
        if change.meta is None and change.opaque:
            tree.clear(change.path)
        elif change.meta is None:
            tree.remove(change.path)
    entries = [c for c in changes if c.meta is not None]
    for change in sorted(entries, key=lambda c: c.path.count("/") if c.path else -1):
        assert change.meta is not None
        tree.put(change.path, change.meta)


def installed_layer(base: Image, image: Image) -> Layer:
    """The one layer *image* holds over *base*: proved by the image's layers being the base's,
    diff_id for diff_id, with one more. The diff_ids, not the blobs, are compared: one image store
    exports a layer gzip-compressed and another plain, and the diff_id names its bytes either way.
    Raise LayerError for an image that is not its base and one layer more."""
    below = tuple(layer.diff_id for layer in image.layers[:-1])
    if not image.layers or below != tuple(layer.diff_id for layer in base.layers):
        raise LayerError(f"the image of {image.archive} is not its base {base.manifest_digest} and one layer more")
    return image.layers[-1]


def base_and_image(base: Image, image: Image, *, reader: Callable[[str, BinaryIO], None] | None = None,
                   base_tree: Tree | None = None) -> tuple[Tree, Tree, list[Change]]:
    """The trees of *base* and of *image*, which is *base* and one layer more (``installed_layer``),
    and that layer's changes: its file bytes handed to *reader* as ``read_layer`` hands them. The
    base's tree is *base_tree*, its listing an earlier check left (``read_listing``), or else comes
    from its own archive, every layer of it; the image's is the base's with the one layer applied,
    which is the only part of the image's archive read."""
    changes = read_layer(image, installed_layer(base, image), reader=reader)
    tree = base_tree
    if tree is None:
        tree = Tree()
        for layer in base.layers:
            apply(tree, read_layer(base, layer))
    built = tree.copy()
    apply(built, changes)
    return tree, built, changes


def write_listing(path: Path, tree: Tree, manifest_digest: str) -> None:
    """Leave *tree*, the tree of the image whose manifest has the digest *manifest_digest*, at
    *path* for a later check (``read_listing``): JSON of the digest and each entry, by path, with
    its kind, mode, size, target and sha256. It is written beside *path* and renamed into place,
    so a listing cut short is never read."""
    entries = [[p, m.kind, m.mode, m.size, m.target, m.sha256] for p, m in tree.entries.items()]
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps({"manifest": manifest_digest, "entries": entries}))
    partial.replace(path)


def read_listing(path: Path, manifest_digest: str) -> Tree:
    """The tree a listing at *path* holds (``write_listing``). Raise LayerError for a listing of
    another manifest than *manifest_digest*, or one that is not a tree: an entry whose directory it
    lacks, or a field of another type."""
    try:
        listing = json.loads(path.read_text())
        if listing["manifest"] != manifest_digest:
            raise LayerError(f"the listing {path} is of the image {listing['manifest']}, not {manifest_digest}")
        tree = Tree()
        for p, kind, mode, size, target, sha256 in sorted(listing["entries"], key=lambda e: e[0].count("/")):
            if kind not in ("file", "directory", "link", "special") or not isinstance(mode, int) or (
                    not isinstance(size, int)):
                raise LayerError(f"the listing {path} holds {p!r} as {kind!r}")
            tree.put(p, Meta(kind, mode, size, target, sha256))
    except (OSError, ValueError, KeyError, TypeError) as e:
        if isinstance(e, LayerError):
            raise
        raise LayerError(f"the listing {path} cannot be read: {e}") from e
    return tree
