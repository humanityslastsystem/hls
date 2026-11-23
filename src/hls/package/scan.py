"""The scan for secrets in what compression hides from finalize's scan.

A submission image is a squashfs file and a source bundle a git pack, both compressed, so a
secret's value the agent committed or a build hook wrote is not in their stored bytes, which are
all finalize's own scan reads (``hls.substrate.secret_scan``). This role searches every form of each
secret (``hls.substrate.secret_forms``: its value, and the host of a value that is an http(s) URL
when it identifies a machine, in any ASCII case), as finalize's scan does, in what was plain on the
node:
- every entry of the checked-out tree (``hls.package.checkout``), the bundle's ``main`` that the
  image holds at ``/workspace``, names starting with a dot among them (the source_scan stage,
  ``hls.stages.source_scan``; the checkout itself reads every object of the bundle);
- once the image is built, every entry under the image's venv (``VENV``) that the base does not
  hold as it is: what the offline install added, the wheels' entries and what pip wrote of its
  own, the bytecode it compiles among it (the package stage).
The walk is ``hls.util.walk_tree``'s, and no entry is passed over: a form is found in every
entry's path as the image holds it (``named_in``), directories, links and special files among
them; in a regular file's bytes, read as they are stored, never decompressed (``scan_file``); and
in the target a link holds, which the image carries as it is. A link is never followed and a
device, a FIFO or a socket never opened, and a venv whose real path leads outside the extracted
image is not read: a contestant's link must not make the harness read a file of the host's. Both
sources are bounded by the package's limits: the tree by the checkout's (``TREE_MAX_BYTES``,
``TREE_MAX_ENTRIES``), what the install added by ``INSTALL_MAX_GIB``; the comparison reads the
base's venv too.

    python -m hls.package.scan --names NAME[,NAME...] [--tree DIR --tree-at PATH]
        [--base DIR --image DIR | --base-archive FILE --base-manifest HEX [--base-listing FILE]
         --image-archive FILE --image-manifest HEX] [--extend FILE] --out FILE --outputs-out R

The values come from the role's environment, one variable per name, which the stage hands over in
the role's 0600 env file; argv carries the names only. *PATH* is where the image holds the tree.
``--base`` and ``--image`` are the extracted base and submission images. On a substrate whose
image files are OCI image archives, which are no trees on the host, the archives are given
instead, each with the bare hex sha256 of the manifest it was pinned by, and the image is proved
the base and one layer more, the install's (``hls.package.layers``): only that layer's entries can
be what the install added, so only its bytes are read of the image, and the base's entries come
from the base's archive (``layer_entries``), or from ``--base-listing``, the base's tree the
integrity check left, which must be of the base's manifest. ``--extend`` names a
record an earlier run wrote for the same names and forms, whose findings come first: the checkout
writes the record of the bundle's objects, and the tree's scan adds to it. The role writes
``package-scan.json`` (``PackageScan``) to *FILE*: the forms it searched, the areas it read, each
entry a form was found in, by its path in the image with every form in it replaced by its marker
(``<redacted:name>``, ``<redacted:name.host>``), and the form's label, never a value or a host, at
most ``FINDINGS_PER_SECRET`` files for each label, so that the record finalize reads stays small
whatever a contestant wrote; past that bound it records the area and how many of its files hold
the form (``PackageScanArea``), for finalize to keep the whole area out of any export. It reports
the file's sha256 and size in ``R`` (``report_outputs``) for the stage's manifest, and prints how
many files it lists and the areas it names by count. Exit 0 whatever it found; 1 for anything
else, such as a tree it cannot walk, a file it cannot read or a record to extend that is not one
for these names and forms, with the traceback on stderr."""

from __future__ import annotations

import argparse
import io
import json
import os
import stat
import sys
import traceback
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from hls.package import layers
from hls.recipes.model import VENV
from hls.store.manifest import report_outputs
from hls.substrate.secret_forms import FORMS, SecretForm, registry_forms
from hls.substrate.secret_scan import (CHUNK_BYTES, PackageScan, PackageScanArea, PackageScanFinding, ScanArea,
                                       find, named_in, scan_file)
from hls.substrate.secrets import Scrubber
from hls.util import EntryKind, walk_tree

#: Files listed for each label at most, the value's and the host's of one secret each on its own.
#: A contestant can put a form in every file it makes, up to the tree's 200,000 and the install's
#: paths; any one of them decides how finalize judges the form, and past the bound the record holds
#: the area and a count instead, which finalize keeps out of any export whole, so the record it
#: reads stays small.
FINDINGS_PER_SECRET = 1000
#: Characters of a file's path a record keeps, enough for every path of the tree within the
#: checkout's limits (1,024 bytes after ``/workspace/``), so that each finding, and so the record,
#: is bounded whatever path the install wrote.
FINDING_PATH_CHARS = 1100
# Bytes compared at a time.
_COMPARE_BYTES = 1 << 20

__all__ = ["CHUNK_BYTES", "FINDINGS_PER_SECRET", "FINDING_PATH_CHARS", "Entry", "Searched", "added_entries",
           "layer_entries", "scan", "tree_entries"]


@dataclass(frozen=True)
class Searched:
    """An entry read from an image archive, which has no path on the host: the labels of the forms
    its bytes, or its link's target, hold, found as its layer was read (``layer_entries``)."""

    labels: frozenset[str]


#: An entry of an area a scan reads: its path on the host, or what was found in it as it was read
#: from an image archive (``Searched``); its path in the image; and its kind.
Entry = tuple[Path | Searched, str, EntryKind]


def tree_entries(tree: Path, at: str) -> Iterator[Entry]:
    """Every entry under *tree*, names starting with a dot among them, in the order of their
    paths, with its path in the image, which holds the tree at *at* (``walk_tree``: links are
    never followed, and a directory that cannot be listed raises)."""
    for entry in walk_tree(tree):
        yield entry.path, str(PurePosixPath(at) / entry.rel), entry.kind


def _same_file(a: Path, b: Path) -> bool:
    """Whether the regular files *a* and *b* hold the same bytes."""
    if os.stat(a).st_size != os.stat(b).st_size:
        return False
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            chunk = fa.read(_COMPARE_BYTES)
            if chunk != fb.read(_COMPARE_BYTES):
                return False
            if not chunk:
                return True


def _base_holds(host: Path, kind: EntryKind, theirs: Path, real: Path) -> bool:
    """Whether the base holds the image's entry *host*, of *kind*, as it is, at *theirs*, which is
    there only when its real path is *real*: a regular file with the same bytes, a directory, or a
    link to the same target. A special file is never the base's: the base's venv holds none, and
    one has no bytes to compare."""
    try:
        if Path(os.path.realpath(theirs.parent)) != real.parent:
            return False
        mine, st = os.lstat(host), os.lstat(theirs)
    except (FileNotFoundError, NotADirectoryError):
        return False
    if stat.S_IFMT(mine.st_mode) != stat.S_IFMT(st.st_mode):
        return False
    if kind == "file":
        return _same_file(host, theirs)
    if kind == "link":
        return os.readlink(host) == os.readlink(theirs)
    return kind == "directory"


def added_entries(base: Path, image: Path) -> Iterator[Entry]:
    """Every entry under the extracted *image*'s venv (``VENV``) that the extracted *base* does
    not hold as it is (``_base_holds``), in the order of their paths, with its path in the image:
    what the offline install added or changed. Links are never followed, in the image or in the
    base (``walk_tree``), a venv whose real path leads outside *image* holds nothing of the image's,
    and a directory that cannot be listed raises."""
    root, base_root = Path(os.path.realpath(image)), Path(os.path.realpath(base))
    venv = image / VENV.lstrip("/")
    if not Path(os.path.realpath(venv)).is_relative_to(root):
        return
    for entry in walk_tree(venv):
        rel = entry.path.relative_to(image)
        if not _base_holds(entry.path, entry.kind, base / rel, base_root / rel):
            yield entry.path, "/" + rel.as_posix(), entry.kind


def _layer_base_holds(base: layers.Tree, path: str, meta: layers.Meta) -> bool:
    """Whether *base* holds the entry *meta* at *path* as it is, as ``_base_holds`` asks of an
    extracted base: there only when every directory on the way is one and none is a link; a
    regular file with the same bytes, a directory, or a link to the same target."""
    parts = path.split("/")
    for i in range(1, len(parts)):
        above = base.get("/".join(parts[:i]))
        if above is None or above.kind != "directory":
            return False
    theirs = base.get(path)
    if theirs is None or theirs.kind != meta.kind:
        return False
    if meta.kind == "file":
        return theirs.sha256 == meta.sha256
    if meta.kind == "link":
        return theirs.target == meta.target
    return meta.kind == "directory"


def layer_entries(base: layers.Image, image: layers.Image, forms: Sequence[SecretForm], *,
                  base_tree: layers.Tree | None = None) -> list[Entry]:
    """Every entry under the venv (``VENV``) of *image*, which is *base* and one layer more, that
    *base* does not hold as it is, in the order of their paths, as ``added_entries`` finds them in
    extracted images: an entry the layer leaves alone is the base's, so only the layer's entries
    are compared, against the base's tree from the base's archive (``layers.base_and_image``). The
    bytes of the layer's files are searched for *forms* as the layer is read, and a link's target
    when it is listed, so each entry comes as ``Searched``. Raise ValueError for an
    image whose venv is not a directory with none but directories on the way, and LayerError
    (``layers``) for archives that are not these images. *base_tree*, when given, is the base's
    tree, read from its listing instead of its layers."""
    venv = VENV.strip("/")
    found: dict[str, frozenset[str]] = {}

    def reader(path: str, stream: BinaryIO) -> None:
        # Every file of the layer: a hard link under the venv may name one outside it.
        found[path] = frozenset(find(stream, forms))

    before, after, changes = layers.base_and_image(base, image, reader=reader, base_tree=base_tree)
    parts = venv.split("/")
    for i in range(1, len(parts) + 1):
        meta = after.get("/".join(parts[:i]))
        if meta is None or meta.kind != "directory":
            raise ValueError(f"the image holds no directory at /{'/'.join(parts[:i])}")
    by_bytes = {c.meta.sha256: found[c.path] for c in changes if c.meta is not None and c.path in found}
    entries: list[Entry] = []
    for path in sorted({c.path for c in changes if c.meta is not None and c.path.startswith(venv + "/")},
                       key=lambda p: p.split("/")):
        meta = after.get(path)
        assert meta is not None  # the layer's own entry, which no layer above it hides
        if _layer_base_holds(before, path, meta):
            continue
        if meta.kind == "file":
            labels = by_bytes[meta.sha256]
        elif meta.kind == "link":
            assert meta.target is not None
            labels = frozenset(find(io.BytesIO(os.fsencode(meta.target)), forms))
        else:
            labels = frozenset()
        entries.append((Searched(labels), "/" + path, meta.kind))
    return entries


def _redacted(path: str, scrubber: Scrubber) -> str:
    """*path* with every form of every secret in it replaced by its marker (``Scrubber``), a host in
    any case, so a record never holds a value or a host, cut to its first ``FINDING_PATH_CHARS``
    characters."""
    return os.fsdecode(scrubber.scrub_bytes(os.fsencode(path)))[:FINDING_PATH_CHARS]


def _held(host: Path | Searched, at: str, kind: EntryKind, forms: Sequence[SecretForm]) -> set[str]:
    """The labels of *forms* an entry holds, at *host* and *at* in the image: in its path, always;
    in its bytes as stored, for a regular file; in its target, for a link, which the image carries
    as it is; for an entry read from an image archive, what was found in those as it was read. A
    special file holds nothing else and is never opened."""
    labels = named_in(Path(at), forms)
    if isinstance(host, Searched):
        return labels | host.labels
    if kind == "file":
        labels |= scan_file(host, forms, decompress=False)
    elif kind == "link":
        labels |= find(io.BytesIO(os.readlink(os.fsencode(host))), forms)
    return labels


def _found(forms: Sequence[SecretForm], scrubber: Scrubber, sources: Iterable[Entry]) -> Iterator[tuple[str, str]]:
    """Each (path in the image, every form in it redacted; label) of the entries of *sources* that
    hold one of *forms* (``_held``)."""
    for host, at, kind in sources:
        for label in sorted(_held(host, at, kind, forms)):
            yield _redacted(at, scrubber), label


def scan(registry: Mapping[str, str], *sources: tuple[ScanArea, Iterable[Entry]],
         extend: PackageScan | None = None) -> PackageScan:
    """The record of where every form of each secret of *registry*, secret name to value, is found
    in the entries of *sources*, each an area with its entries, after what *extend*, an earlier
    record for the same names and forms, holds: the areas read, and at most ``FINDINGS_PER_SECRET``
    entries for each label, *extend*'s among them, in the order the sources give them; for each
    area in which a label's entries are not all listed, the area and how many of them hold the form.
    Raise ValueError for an *extend* that searched other names or other forms."""
    forms, scrubber = registry_forms(registry), Scrubber(registry)
    findings: list[PackageScanFinding] = []
    areas: list[PackageScanArea] = []
    read: list[ScanArea] = []
    listed: dict[str, int] = {}
    if extend is not None:
        if extend.secrets != sorted(registry):
            raise ValueError(f"the record to extend searched {extend.secrets}, not {sorted(registry)}")
        if extend.forms != list(FORMS):
            raise ValueError(f"the record to extend searched the forms {extend.forms}, not {list(FORMS)}")
        findings += extend.findings
        areas += extend.areas
        read += extend.read
        for finding in extend.findings:
            listed[finding.secret] = listed.get(finding.secret, 0) + 1
    for area, source in sources:
        found: dict[str, int] = {}
        kept: dict[str, int] = {}
        for path, label in _found(forms, scrubber, source):
            found[label] = found.get(label, 0) + 1
            if listed.get(label, 0) < FINDINGS_PER_SECRET:
                listed[label] = listed.get(label, 0) + 1
                kept[label] = kept.get(label, 0) + 1
                findings.append(PackageScanFinding(path=path, secret=label))
        areas += [PackageScanArea(secret=label, area=area, count=count)
                  for label, count in sorted(found.items()) if count > kept.get(label, 0)]
        read.append(area)
    return PackageScan(secrets=sorted(registry), forms=list(FORMS), read=read, findings=findings, areas=areas)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.package.scan")
    p.add_argument("--names", required=True)
    p.add_argument("--tree", type=Path)
    p.add_argument("--tree-at")
    p.add_argument("--base", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--base-archive", type=Path)
    p.add_argument("--base-manifest")
    p.add_argument("--image-archive", type=Path)
    p.add_argument("--image-manifest")
    p.add_argument("--base-listing", type=Path)
    p.add_argument("--extend", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    args = p.parse_args(argv)
    if (args.tree is None) != (args.tree_at is None):
        p.error("--tree and --tree-at go together")
    if (args.base is None) != (args.image is None):
        p.error("--base and --image go together")
    archives = (args.base_archive, args.base_manifest, args.image_archive, args.image_manifest)
    if archives.count(None) not in (0, 4):
        p.error("--base-archive, --base-manifest, --image-archive and --image-manifest go together")
    if args.image is not None and args.image_archive is not None:
        p.error("give an image as its tree or as its archive, not both")
    if args.base_listing is not None and args.base_archive is None:
        p.error("--base-listing is the listing of the base's archive, and goes with it")
    if args.tree is None and args.image is None and args.image_archive is None:
        p.error("give a tree, an image or both to scan")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        registry = {name: os.environ[name] for name in args.names.split(",")}
        sources: list[tuple[ScanArea, Iterable[Entry]]] = []
        if args.tree is not None:
            if not args.tree.is_dir() or args.tree.is_symlink():
                raise FileNotFoundError(f"{args.tree} is not the checked-out tree")
            sources.append(("tree", tree_entries(args.tree, args.tree_at)))
        if args.image is not None:
            sources.append(("install", added_entries(args.base, args.image)))
        elif args.image_archive is not None:
            base = layers.open_image(args.base_archive, "sha256:" + args.base_manifest)
            image = layers.open_image(args.image_archive, "sha256:" + args.image_manifest)
            listed = None if args.base_listing is None else layers.read_listing(args.base_listing,
                                                                                  base.manifest_digest)
            sources.append(("install", layer_entries(base, image, registry_forms(registry), base_tree=listed)))
        extend = None if args.extend is None else PackageScan.model_validate_json(args.extend.read_bytes())
        record = scan(registry, *sources, extend=extend)
        tmp = args.out.with_name(f".{args.out.name}.tmp")
        tmp.write_text(record.model_dump_json(indent=2))
        os.replace(tmp, args.out)
        report_outputs(args.outputs_out, args.out.parent, args.out)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("package scan failed before it read every file", file=sys.stderr)
        return 1
    print(json.dumps({"findings": len(record.findings), "areas": [a.model_dump() for a in record.areas]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
