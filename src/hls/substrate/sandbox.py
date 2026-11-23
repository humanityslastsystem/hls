"""Images extracted on the node, and the manifest that pins each extraction.

A container runs from a directory: its image file extracted into the job's node-local scratch,
at ``sandboxes/<sha256 of the image file>/``, for the stage that pins it. The tree lives as
long as that stage: its context removes it when it closes, so each stage's declared disk holds
exactly its own trees, and a later stage of the job extracts the image again, a minute or more
for a large image. Within a stage, a second
pin of the image returns the tree the stage pinned. A tree a stage finds that it did not pin
was left by a context that could not remove it; it is reused only after a check against the
manifest written right after its extraction, at ``sandboxes/<sha256>.json``, outside the
directory. The check costs one ``lstat`` per entry, a second or two for such an image; anything
it finds re-extracts the image.

The manifest records, for every entry under the directory, its path, its ``st_mode`` (type and
permissions), its size, its modification and change times in nanoseconds, its inode number and,
for a link, its target. So a file added, removed, replaced, renamed, resized, made a link or
given other permissions changes the walk, and so does a file whose bytes were rewritten in
place: the write moves its change time, which no unprivileged process can set back. What the
check cannot catch is a change below the filesystem (a disk that returns other bytes), or one
by root, who can set the clock, since it never reads a file's bytes. Reading them would take
tens of seconds at every reuse of a 25 GB tree, a good part of extracting again, and would
catch only those."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from hls.store.digest import Sha256Hex, check_sha256_hex, sha256_bytes
from hls.util import remove_tree

#: The file an images cache entry holds its image in.
IMAGE_FILE = "image.sif"
#: The directory, in the job's node-local scratch, that holds the extracted images.
SANDBOXES = "sandboxes"


class TreeEntry(NamedTuple):
    """One entry under an extracted directory, as ``lstat`` saw it."""

    #: The path relative to the directory, "/"-separated; "." for the directory itself.
    path: str
    #: ``st_mode``: the file type and its permission bits.
    mode: int
    #: ``st_size``.
    size: int
    #: ``st_mtime_ns``.
    mtime_ns: int
    #: ``st_ctime_ns``, which only the kernel sets.
    ctime_ns: int
    #: ``st_ino``.
    ino: int
    #: Where a link points; None for anything else.
    target: str | None


class TreeManifest(BaseModel):
    """What an extracted directory held right after its extraction."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The sha256 of the image file it was extracted from.
    sif_sha256: Sha256Hex
    #: Every entry, the directory itself first, then sorted by path.
    entries: list[TreeEntry]


def sandbox_paths(scratch: Path, sif_sha256: str) -> tuple[Path, Path]:
    """The extracted directory of the image file whose bare hex sha256 is *sif_sha256*, and its
    manifest, in the job's node-local *scratch*. Raise ValueError for a digest that is not bare
    hex."""
    root = scratch / SANDBOXES
    return root / check_sha256_hex(sif_sha256), root / f"{sif_sha256}.json"


def scan_tree(root: Path) -> list[TreeEntry]:
    """Every entry under *root*, *root* itself first and then sorted by path. Links are
    recorded, never followed."""
    st = os.lstat(root)
    out = [TreeEntry(".", st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, None)]
    pending = [(root, "")]
    while pending:
        directory, rel = pending.pop()
        with os.scandir(directory) as it:
            for e in it:
                path = f"{rel}{e.name}"
                st = e.stat(follow_symlinks=False)
                target = os.readlink(e.path) if stat.S_ISLNK(st.st_mode) else None
                out.append(TreeEntry(path, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino,
                                     target))
                if stat.S_ISDIR(st.st_mode):
                    pending.append((Path(e.path), f"{path}/"))
    out[1:] = sorted(out[1:], key=lambda e: e.path)
    return out


def _manifest_bytes(manifest: TreeManifest) -> bytes:
    return json.dumps(manifest.model_dump(mode="json"), separators=(",", ":")).encode()


def write_manifest(root: Path, sif_sha256: str, path: Path) -> str:
    """Scan *root*, extracted from the image file whose sha256 is *sif_sha256*, write its
    manifest to *path* and return the manifest's sha256. A manifest cut short by a crash does
    not parse, so the next check extracts the image again."""
    data = _manifest_bytes(TreeManifest(sif_sha256=sif_sha256, entries=scan_tree(root)))
    path.write_bytes(data)
    return sha256_bytes(data)


def tree_problem(root: Path, path: Path, sif_sha256: str) -> tuple[str | None, str | None]:
    """Whether *root* still holds exactly what the manifest at *path* recorded for the image
    file whose sha256 is *sif_sha256*: (None, the manifest's sha256) when it does, else (why
    not, None)."""
    try:
        data = path.read_bytes()
        manifest = TreeManifest.model_validate_json(data)
    except (OSError, ValueError) as e:
        return f"its manifest {path} cannot be read: {e}", None
    if manifest.sif_sha256 != sif_sha256:
        return f"its manifest {path} is of image file {manifest.sif_sha256}", None
    try:
        now = scan_tree(root)
    except OSError as e:
        return f"{root} cannot be walked: {e}", None
    if now == manifest.entries:
        return None, sha256_bytes(data)
    recorded = {e.path: e for e in manifest.entries}
    seen = {e.path: e for e in now}
    changed = sorted(p for p in recorded.keys() & seen.keys() if recorded[p] != seen[p])
    added, removed = sorted(seen.keys() - recorded.keys()), sorted(recorded.keys() - seen.keys())
    parts = [f"{what} {names[:3]}" + (f" and {len(names) - 3} more" if len(names) > 3 else "")
             for what, names in (("changed", changed), ("added", added), ("removed", removed)) if names]
    return f"{root} differs from its manifest: " + "; ".join(parts), None


def make_removable(root: Path) -> None:
    """Give the owner read, write and search on the directory *root* and every directory under
    it, without following a link. An image's directories can lack the owner's write bit, which
    ``hls.util.remove_tree`` needs to empty them: a context removes its trees with it when it
    closes, and the runner the job's scratch when the job ends. The container sees the bits,
    never a write: its root is read-only."""
    os.chmod(root, stat.S_IMODE(os.lstat(root).st_mode) | stat.S_IRWXU)
    # Each directory before the walk enters it, which it cannot without read and search.
    for directory, dirs, _ in os.walk(root):
        for d in dirs:
            child = Path(directory) / d
            if not os.path.islink(child):
                os.chmod(child, stat.S_IMODE(os.lstat(child).st_mode) | stat.S_IRWXU)


def renew_directory_times(root: Path) -> None:
    """Give the directory *root* and every directory under it the current time as its access
    and modification time, without following a link; files keep theirs. An extracted tree
    carries the times its image was built with, and a node's periodic ``systemd-tmpfiles
    --clean`` (often daily, ``q /tmp 1777 root root 10d``) removes from a scratch under /tmp
    each empty directory whose access and modification times are both older than its age: it can
    take the empty directories of an extracted tree, /tmp and /var/tmp among
    them, and a container started from it then has no /tmp to bind its workdir on. Reading a directory
    renews its access time on a relatime mount only, so the walk that writes the manifest does
    not cover a noatime one. Files need nothing: the clean also weighs their change time, which
    the extraction set. Run after ``make_removable``, which lets the walk enter every directory."""
    os.utime(root)
    for directory, dirs, _ in os.walk(root):
        for d in dirs:
            child = Path(directory) / d
            if not os.path.islink(child):
                os.utime(child)


def remove_extracted(path: Path) -> None:
    """Remove the extracted directory at *path*, or whatever else is there, without following
    a link, after ``make_removable``: a failed extraction leaves its directories as the image
    has them. ``hls.util.remove_tree`` walks without recursing, so any depth is removed."""
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode):
        make_removable(path)
    remove_tree(path)
