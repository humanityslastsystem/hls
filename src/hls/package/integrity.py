"""Whether a submission image's system paths are its base's, byte for byte.

Apptainer starts every container through the image's own ``/bin/sh`` and runs the image's
``/.singularity.d`` action and environment scripts with it before anything the harness owns,
and that shell prints the container's start marker (``hls.substrate.apptainer.argv``). A
submission that could change those files could end its container nonzero before the marker
and pass its own failure off as the runtime's, an infra fault that is retried. So the offline
build installs everything the submission supplies into the base's venv, ``/opt/venv``, and runs
no submission code (``hls.package.offline``), and this check proves the result. After the
build, the submission image's ``/bin``, ``/sbin``, ``/usr``, every top-level ``lib*``, ``/etc``
and ``/.singularity.d`` must hold exactly what the base's do: the same entries, of the same
type and permission bits, with the same bytes and the same link targets. Owners and times are
not compared: both trees are extracted without privileges, so every entry belongs to the job's
user, and a build sets times of its own. ``BUILD_RECORDS`` may differ: the files of
``/.singularity.d`` that record the build itself, the run script the offline ``.def`` declares,
the ``.def`` and the labels; and so may anything under ``BUILD_HISTORY``, where apptainer
moves the base's ``.def``. No container start runs any of them, since the runtime always names
its command.

A difference is the submission's only when pip, installing the wheels the submission's build
produced, made it (``install_targets``), and only the wheels pip installed count: those whose
``.dist-info`` directory the image's venv holds and the base's did not (``installed``). A wheel
pip left out wrote nothing, so only its list of entries is read. pip writes a file for each
entry of a wheel, where the venv's install scheme puts it, and one for each console or GUI
script a wheel declares, in the venv's ``bin``. The write lands where the kernel takes it in the image: through every link the
base holds on the way, and through a last link only when that leads nowhere, since pip removes
an existing entry before it writes (``write_target``). So a difference is pip's only where
the image holds a file and the base held no directory there, or the image holds a directory
pip created above such a file and the base held nothing there. Nothing else in the build writes there: its own steps write ``/workspace``, ``/wheels``
and the build records. Any other difference is not certainly the submission's: the harness,
apptainer, or something nobody foresaw changed the image, so the stage reports it as infra.

    python -m hls.package.integrity --base DIR --image DIR --wheels DIR --out FILE --outputs-out R
    python -m hls.package.integrity --base-archive FILE --base-manifest HEX --image-archive FILE
        --image-manifest HEX [--base-listing-out FILE] --wheels DIR --out FILE --outputs-out R

Both directories are images the stage context pinned (``HostTree``). On a substrate whose image
files are OCI image archives, no image is a tree on the host: the check reads the two archives
instead, each by the manifest digest it was pinned by, the image proved to be the base and one
layer more, the install's (``hls.package.layers``). The base's tree then comes from the base's
archive and the image's is the base's with that layer applied (``ArchiveTree``), so the bytes
read of the image are the install's alone, and an entry's mode is its tar header's, as the image
holds it. ``--base-listing-out`` leaves the base's tree there for the scan, which then reads no
layer of the base again (``layers.write_listing``). Exit 0: the system paths are the base's;
``DIFFERS_EXIT``: they are not, and *FILE* says how (``IntegrityRecord``); either way *FILE*'s
sha256 and size are reported in ``R`` (``report_outputs``) for the stage's manifest. Any other
exit is a failure of the check itself (an entry it cannot read, a missing tree), with the
traceback on stderr."""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import posixpath
import stat
import sys
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

from pydantic import BaseModel, ConfigDict

from hls.package import layers
from hls.package.wheel_check import is_entry_points, unreadable_zip
from hls.recipes.pins import PYTHON_VERSION
from hls.recipes.render import VENV
from hls.store.manifest import report_outputs

#: The top-level directories compared, besides every top-level name that starts with ``lib``.
SYSTEM_DIRS = ("bin", "sbin", "usr", "etc", ".singularity.d")
#: The files of the build's own record, which may differ from the base's.
BUILD_RECORDS = frozenset({".singularity.d/runscript", ".singularity.d/Singularity", ".singularity.d/labels.json"})
#: The directory of the definitions an image was bootstrapped from, which may differ with
#: everything under it.
BUILD_HISTORY = ".singularity.d/bootstrap_history"
#: Exit status when the system paths differ.
DIFFERS_EXIT = 4
#: Differences listed in the record; the counts cover them all.
MAX_LISTED = 200
#: Links followed while resolving one install destination through the base.
MAX_LINKS = 40
#: The entry-point groups pip writes a script for.
SCRIPT_GROUPS = ("console_scripts", "gui_scripts")
# Bytes read at a time while hashing.
_CHUNK = 1 << 20

Change = Literal["added", "removed", "type", "mode", "target", "content"]


class Difference(BaseModel):
    """One way the submission image's system paths differ from the base's."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The path in the image, from its root: ``/usr/bin/sh``.
    path: str
    #: What differs.
    change: Change
    #: The two sides, for people: ``0o755 -> 0o777``, ``file -> link``.
    detail: str
    #: The submission's wheels whose install accounts for the difference, by file name; empty
    #: when none does.
    wheels: list[str]


class IntegrityRecord(BaseModel):
    """What the check compared and found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The top-level names compared.
    compared: list[str]
    #: Entries under them in the base.
    entries: int
    #: How many differences there are.
    differences: int
    #: How many of them no submission wheel's install accounts for.
    unattributed: int
    #: The first ``MAX_LISTED`` differences, by path.
    listed: list[Difference]


class Targets(NamedTuple):
    """Where installing the submission's wheels writes, each path to the wheels (file names)
    that write there."""

    #: Files pip writes.
    files: dict[str, set[str]]
    #: Directories above them, which pip creates when they are missing.
    dirs: dict[str, set[str]]


class _Entry(NamedTuple):
    """An entry as ``lstat`` sees it."""

    #: ``file``, ``dir``, ``link`` or ``special``.
    kind: str
    #: The permission bits.
    mode: int
    #: The size of a file; 0 for anything else.
    size: int
    #: Where a link points; None for anything else.
    target: str | None


def _kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "dir"
    if stat.S_ISLNK(mode):
        return "link"
    return "special"


def _entry(path: Path) -> _Entry:
    st = os.lstat(path)
    kind = _kind(st.st_mode)
    return _Entry(kind, stat.S_IMODE(st.st_mode), st.st_size if kind == "file" else 0,
                  os.readlink(path) if kind == "link" else None)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


class Tree(Protocol):
    """An image's tree as the check reads it, each path from the image's root without a leading
    slash, ``""`` the root itself."""

    def names(self, rel: str) -> list[str]:
        """The names the directory at *rel* holds, as ``os.listdir`` lists them."""

    def entry(self, rel: str) -> _Entry:
        """The entry at *rel* as ``lstat`` sees it; the OSError ``lstat`` raises when there is none."""

    def lexists(self, rel: str) -> bool:
        """Whether there is an entry at *rel*, as ``os.path.lexists`` says."""

    def is_symlink(self, rel: str) -> bool:
        """Whether the entry at *rel* is a link, as ``Path.is_symlink`` says."""

    def readlink(self, rel: str) -> str:
        """Where the link at *rel* points."""

    def sha256(self, rel: str) -> str:
        """The sha256 of the bytes of the file at *rel*."""


class HostTree:
    """An image extracted on the host: a pinned image on a substrate that extracts its images."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def names(self, rel: str) -> list[str]:
        return os.listdir(self.root / rel)

    def entry(self, rel: str) -> _Entry:
        return _entry(self.root / rel)

    def lexists(self, rel: str) -> bool:
        return os.path.lexists(self.root / rel)

    def is_symlink(self, rel: str) -> bool:
        return (self.root / rel).is_symlink()

    def readlink(self, rel: str) -> str:
        return os.readlink(self.root / rel)

    def sha256(self, rel: str) -> str:
        return _sha256(self.root / rel)


#: The check's kind of each kind of an image archive's entries.
_KINDS = {"file": "file", "directory": "dir", "link": "link", "special": "special"}


class ArchiveTree:
    """An image read from its archive's layers (``hls.package.layers``). A path resolves as the
    kernel resolves it in the image: a link on the way to the last name is followed inside the
    image, an absolute one from the image's root, at most ``MAX_LINKS`` of them."""

    def __init__(self, tree: layers.Tree) -> None:
        self.tree = tree

    def _real(self, parts: list[str]) -> str:
        """The path of the directory *parts* name, with every link on the way followed."""
        done: list[str] = []
        pending = list(parts)
        hops = 0
        while pending:
            part = pending.pop(0)
            if part == ".":
                continue
            if part == "..":
                if done:
                    done.pop()
                continue
            here = "/".join([*done, part])
            meta = self.tree.get(here)
            if meta is None:
                raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "/" + here)
            if meta.kind == "link":
                hops += 1
                if hops > MAX_LINKS:
                    raise OSError(errno.ELOOP, os.strerror(errno.ELOOP), "/" + here)
                assert meta.target is not None
                if meta.target.startswith("/"):
                    done = []
                pending = [p for p in meta.target.split("/") if p] + pending
                continue
            if meta.kind != "directory":
                raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), "/" + here)
            done.append(part)
        return "/".join(done)

    def _meta(self, rel: str) -> layers.Meta:
        parts = [p for p in rel.split("/") if p]
        if not parts or parts[-1] in (".", ".."):
            path = self._real(parts)
        else:
            parent = self._real(parts[:-1])
            path = f"{parent}/{parts[-1]}" if parent else parts[-1]
        meta = self.tree.get(path)
        if meta is None:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), "/" + rel)
        return meta

    def names(self, rel: str) -> list[str]:
        path = self._real([p for p in rel.split("/") if p])
        if path not in self.tree.names:
            raise NotADirectoryError(errno.ENOTDIR, os.strerror(errno.ENOTDIR), "/" + rel)
        return list(self.tree.names[path])

    def entry(self, rel: str) -> _Entry:
        meta = self._meta(rel)
        return _Entry(_KINDS[meta.kind], meta.mode, meta.size if meta.kind == "file" else 0, meta.target)

    def lexists(self, rel: str) -> bool:
        try:
            self._meta(rel)
        except OSError:
            return False
        return True

    def is_symlink(self, rel: str) -> bool:
        try:
            return self._meta(rel).kind == "link"
        except OSError:
            return False

    def readlink(self, rel: str) -> str:
        meta = self._meta(rel)
        if meta.target is None:
            raise OSError(errno.EINVAL, os.strerror(errno.EINVAL), "/" + rel)
        return meta.target

    def sha256(self, rel: str) -> str:
        meta = self._meta(rel)
        if meta.sha256 is None:
            raise IsADirectoryError(errno.EISDIR, "not a regular file", "/" + rel)
        return meta.sha256


def compared_names(base: Tree, image: Tree) -> list[str]:
    """The top-level names compared: ``SYSTEM_DIRS`` and every ``lib*`` name of either tree."""
    libs = {name for root in (base, image) for name in root.names("") if name.startswith("lib")}
    return sorted(set(SYSTEM_DIRS) | libs)


def walk(root: Tree, names: list[str]) -> dict[str, _Entry]:
    """Every entry at or under the top-level *names* of *root*, by path from the root (no
    leading slash). A link is recorded, never followed."""
    out: dict[str, _Entry] = {}

    def visit(rel: str) -> None:
        entry = root.entry(rel)
        out[rel] = entry
        if entry.kind == "dir":
            for child in sorted(root.names(rel)):
                visit(f"{rel}/{child}")

    for name in names:
        if root.lexists(name):
            visit(name)
    return out


class _Found(NamedTuple):
    """One difference, with what each tree holds at its path."""

    #: The path from the root, without the leading slash.
    rel: str
    #: What differs.
    change: Change
    #: The two sides, for people.
    detail: str
    #: The kind of the base's entry there; None when it holds none.
    before: str | None
    #: The kind of the image's entry there; None when it holds none.
    after: str | None


def _change(a: _Entry | None, b: _Entry | None, base: Tree, image: Tree, rel: str) -> tuple[Change, str] | None:
    if a is None:
        assert b is not None
        return "added", f"a {b.kind} the base does not hold"
    if b is None:
        return "removed", f"the base's {a.kind} is gone"
    if a.kind != b.kind:
        return "type", f"{a.kind} -> {b.kind}"
    if a.mode != b.mode:
        return "mode", f"{a.mode:#o} -> {b.mode:#o}"
    if a.target != b.target:
        return "target", f"{a.target} -> {b.target}"
    if a.kind == "file" and (a.size != b.size or base.sha256(rel) != image.sha256(rel)):
        return "content", f"{a.size} bytes -> {b.size} bytes"
    return None


def differences(base: Tree, image: Tree, names: list[str]) -> Iterator[_Found]:
    """Each way the entries at or under *names* differ between *base* and *image*, by path,
    except ``BUILD_RECORDS`` and ``BUILD_HISTORY``."""
    old, new = walk(base, names), walk(image, names)
    for rel in sorted(old.keys() | new.keys()):
        if rel in BUILD_RECORDS or rel == BUILD_HISTORY or rel.startswith(BUILD_HISTORY + "/"):
            continue
        a, b = old.get(rel), new.get(rel)
        found = _change(a, b, base, image, rel)
        if found is not None:
            yield _Found(rel, *found, None if a is None else a.kind, None if b is None else b.kind)


def scheme(dist: str) -> dict[str, str]:
    """Where pip installs each part of the wheel of distribution *dist* in the base's venv."""
    site = f"{VENV}/lib/python{PYTHON_VERSION}/site-packages"
    return {"purelib": site, "platlib": site, "scripts": f"{VENV}/bin", "data": VENV,
            "headers": f"{VENV}/include/site/python{PYTHON_VERSION}/{dist}"}


def destination(member: str, dist: str) -> str | None:
    """The absolute path pip installs the wheel entry *member* of *dist* at; None for a
    directory entry. ``..`` is resolved as text, so an entry that climbs out of its scheme names
    the place it would land."""
    if member.endswith("/"):
        return None
    parts = member.split("/")
    places = scheme(dist)
    if len(parts) > 2 and parts[0].endswith(".data") and parts[1] in places:
        return posixpath.normpath(posixpath.join(places[parts[1]], *parts[2:]))
    return posixpath.normpath(posixpath.join(places["purelib"], member))


def through_links(base: Tree, path: str) -> str | None:
    """*path* (absolute, in the image) with every link the base holds along it resolved, inside
    the base, as the kernel would resolve it in the image; None past ``MAX_LINKS`` links, where
    the kernel refuses the path."""
    parts = [p for p in path.split("/") if p]
    done: list[str] = []
    hops = 0
    while parts:
        part = parts.pop(0)
        if part == "..":
            if done:
                done.pop()
            continue
        if part == ".":
            continue
        here = "/".join([*done, part])
        if base.is_symlink(here):
            hops += 1
            if hops > MAX_LINKS:
                return None
            target = base.readlink(here)
            if target.startswith("/"):
                done = []
            parts = [p for p in target.split("/") if p] + parts
            continue
        done.append(part)
    return "/" + "/".join(done)


def write_target(base: Tree, folder: str, name: str) -> str:
    """Where pip's write of the file *name* in *folder* (a directory of the image, its links
    resolved) lands. pip removes an existing entry before it writes, so a link the base holds
    there is replaced when it leads to an entry and written through when it leads nowhere."""
    where = posixpath.join(folder, name)
    if base.is_symlink(where.lstrip("/")):
        target = through_links(base, where)
        if target is not None and not base.lexists(target.lstrip("/")):
            return target
    return where


def script_names(text: str) -> list[str]:
    """The console and GUI scripts an ``entry_points.txt`` declares, read the way
    ``importlib.metadata`` reads one, which is how pip reads a wheel's."""
    names: list[str] = []
    group = None
    for line in map(str.strip, text.splitlines()):
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            group = line.strip("[]")
        elif group in SCRIPT_GROUPS and "=" in line:
            names.append(line.split("=", 1)[0].strip())
    return names


def installed(base: Tree, image: Tree, members: list[str]) -> bool:
    """Whether pip installed the wheel whose entries are *members* in *image*: the image holds
    the wheel's one ``.dist-info`` directory in the venv's site-packages, where *base* held
    nothing. pip installs no wheel with no such directory or several, and finds a distribution
    whose ``.dist-info`` the base holds installed already."""
    tops = {member.split("/", 1)[0] for member in members}
    found = [top for top in tops if top.endswith(".dist-info")]
    site = through_links(base, scheme("")["purelib"])
    if len(found) != 1 or site is None:
        return False
    where = f"{site.lstrip('/')}/{found[0]}"
    try:
        made = image.entry(where).kind == "dir"
    except FileNotFoundError:
        return False
    return made and not base.lexists(where)


def _wheel_contents(path: str, base: Tree, image: Tree) -> tuple[list[str], list[str]] | None:
    """The entries of the wheel at *path*, and the text of each ``.dist-info/entry_points.txt``
    among them; None when pip did not install it in *image* over *base* (``installed``), whose
    entry points are then never read."""
    with zipfile.ZipFile(path) as z:
        members = z.namelist()
        if not installed(base, image, members):
            return None
        return members, [z.read(m).decode() for m in members if is_entry_points(m)]


def _writes(members: list[str], entry_points: list[str], dist: str) -> list[str]:
    """Each path, in the image before any link is followed, that pip writes a file at when it
    installs a wheel of *dist* holding *members*: its entries, and the scripts its
    *entry_points* files declare."""
    paths = [dest for member in members if (dest := destination(member, dist)) is not None]
    for text in entry_points:
        # Joined as os.path.join joins them for pip's script maker: an absolute name replaces
        # the folder, and the kernel resolves any ".." after the links.
        paths += [posixpath.join(f"{VENV}/bin", name) for name in script_names(text)]
    return paths


def install_targets(wheels: Path, base: Tree, image: Tree) -> Targets:
    """Where installing the wheels in *wheels* that pip installed in *image* over *base*
    (``installed``) writes in the image. A file that is not a readable wheel writes nothing
    here: pip installs no wheel it cannot read. An error of the node reading a wheel, one neither
    the wheel nor the permissions the wheel step left explain (``unreadable_zip``), is raised."""
    targets = Targets({}, {})
    for item in sorted(os.scandir(wheels), key=lambda e: e.name):
        if not item.name.endswith(".whl") or not item.is_file(follow_symlinks=False):
            continue
        dist = item.name.split("-")[0]
        try:
            contents = _wheel_contents(item.path, base, image)
        except OSError as e:
            # The node's error fails the check; one the wheel itself or its permissions explain
            # leaves it unreadable.
            if not unreadable_zip(e):
                raise
            continue
        except Exception:  # noqa: BLE001
            # Only the wheel step wrote the file, so whatever way it fails to read is its own,
            # never a failure of this check.
            continue
        if contents is None:
            continue
        members, entry_points = contents
        for path in _writes(members, entry_points, dist):
            folder = through_links(base, posixpath.dirname(path))
            if folder is None:
                continue
            targets.files.setdefault(write_target(base, folder, posixpath.basename(path)), set()).add(item.name)
            while folder != "/":
                targets.dirs.setdefault(folder, set()).add(item.name)
                folder = posixpath.dirname(folder)
    return targets


def _writers(targets: Targets, found: _Found) -> list[str]:
    """The wheels whose install accounts for *found*: a file pip wrote where the base held no
    directory, or a directory it created where the base held nothing."""
    path = f"/{found.rel}"
    if found.after == "file" and found.before != "dir":
        return sorted(targets.files.get(path, ()))
    if found.after == "dir" and found.before is None:
        return sorted(targets.dirs.get(path, ()))
    return []


def check(base: Tree, image: Tree, wheels: Path) -> IntegrityRecord:
    """Compare *image*'s system paths with *base*'s and attribute each difference to the wheels
    in *wheels* whose install accounts for it."""
    names = compared_names(base, image)
    targets = install_targets(wheels, base, image)
    found = [Difference(path=f"/{d.rel}", change=d.change, detail=d.detail, wheels=_writers(targets, d))
             for d in differences(base, image, names)]
    return IntegrityRecord(compared=names, entries=len(walk(base, names)), differences=len(found),
                           unattributed=sum(1 for d in found if not d.wheels), listed=found[:MAX_LISTED])


def archive_trees(base_archive: Path, base_manifest: str, image_archive: Path, image_manifest: str, *,
                  listing_out: Path | None = None) -> tuple[ArchiveTree, ArchiveTree]:
    """The trees of the base and the image of two archives, each by the bare hex sha256 of the
    manifest it was pinned by, the image the base and one layer more (``layers.base_and_image``);
    with *listing_out*, the base's tree is left there (``layers.write_listing``)."""
    base = layers.open_image(base_archive, "sha256:" + base_manifest)
    image = layers.open_image(image_archive, "sha256:" + image_manifest)
    before, after, _ = layers.base_and_image(base, image)
    if listing_out is not None:
        layers.write_listing(listing_out, before, base.manifest_digest)
    return ArchiveTree(before), ArchiveTree(after)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls.package.integrity")
    p.add_argument("--base", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--base-archive", type=Path)
    p.add_argument("--base-manifest")
    p.add_argument("--image-archive", type=Path)
    p.add_argument("--image-manifest")
    p.add_argument("--base-listing-out", type=Path)
    p.add_argument("--wheels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    archives = (args.base_archive, args.base_manifest, args.image_archive, args.image_manifest)
    base: Tree
    image: Tree
    if args.base is not None and args.image is not None and archives == (None,) * 4 and (
            args.base_listing_out is None):
        for tree in (args.base, args.image):
            if not tree.is_dir() or tree.is_symlink():
                raise SystemExit(f"{tree} is not an extracted image")
        base, image = HostTree(args.base), HostTree(args.image)
    elif args.base is None and args.image is None and None not in archives:
        base, image = archive_trees(*archives, listing_out=args.base_listing_out)
    else:
        p.error("give --base and --image, or --base-archive, --base-manifest, --image-archive and --image-manifest "
                "with --base-listing-out or without")
    record = check(base, image, args.wheels)
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_text(record.model_dump_json(indent=2))
    os.replace(tmp, args.out)
    report_outputs(args.outputs_out, args.out.parent, args.out)
    return DIFFERS_EXIT if record.differences else 0


if __name__ == "__main__":
    sys.exit(main())
