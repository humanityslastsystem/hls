"""The cache's role program: the node-local copy of a cache entry, made and checked on the node by
a harness role step, so the runner reads no byte of an entry.

::

    python -m hls.store.cache_role copy   --entry DIR --expect SHA256 --dest DIR --digests-out FILE
    python -m hls.store.cache_role verify --copy DIR --expect SHA256 --digests-out FILE
    python -m hls.store.cache_role fill   --source DIR --dest DIR --digests-out FILE

``copy`` reads the entry in the store once. It copies the entry's ``ENTRY.json`` into *dest*, a
directory it makes, and goes on only if the copy's sha256 is *expect*, the digest the entry's
producer recorded, so the manifest it copies by is the one the stage pinned. It holds the
entry's tree against the manifest by names, sizes and types, then copies each file the
manifest lists and hashes the copy, the bytes that landed, and once they check gives the copy
the modification time its record holds (the store file's own time is never read, so a touch
in the store changes no copy). Every file, ``ENTRY.json`` the first, is copied by one rule: a
copy whose digest is not the recorded one sends the role back to the file in the store, only
then. ``verify`` hashes an existing copy, its ``ENTRY.json`` first, against the manifest, and
holds each file's time against its record, as a stage does that reuses a copy it cannot trust by
its stamp. Each writes the sha256 of every file it read back, ``ENTRY.json`` among them, to
*FILE* outside the copy (``write_digests``); the stage keeps the copy only when those and the
names, sizes and times it finds are the manifest's (``ContentCache.materialize``).

A refusal ends the role with ``DIGEST_MISMATCH_EXIT`` and a last ``refused: <why>`` line on
stderr. ``copy`` refuses the entry in the store, which a retry would read again: an
``ENTRY.json`` that is not *expect*, a file missing, unexpected, or of another size or
sha256 than its record, and anything the cache's walk refuses (``walk_files``: links, hard
links, special files, reserved names, depth). A file whose copy differs from its record, the
``ENTRY.json`` among them, is read again in the store only then, to tell a bad entry from a
copy that went wrong on the way: when the entry's file has its record, the copy was short or
wrong (``CopyFailed``); so was a copy that kept another time than the one it was given. The
role then ends with ``COPY_FAILED_EXIT`` and a ``copy failed:`` line, and any error reading or
writing ends it otherwise; a retry copies anew either way. ``verify``
refuses the copy, whatever of it differs from the manifest, a file's time among it; the stage
retries that, since a new attempt copies the entry afresh.

``fill`` copies a node-local tree a role wrote, such as an engine's caches, into *dest*, an
entry's empty staging directory (``ContentCache.populate``), each file keeping its modification
time, which the cache records, and hashes each copy, the bytes that landed, for the cache to
record. It refuses a tree the cache would not publish, before any file is copied (``walk_files``:
links, hard links, special files, reserved names, depth), and a copy that did not keep its time
is a copy gone wrong (``COPY_FAILED_EXIT``)."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

from pydantic import ValidationError

from hls.store.cache import (ENTRY_FILE, VERIFIED_FILE, CacheCorrupt, EntryManifest, hash_tree, mtime_problem,
                             walk_files, write_digests)
from hls.store.digest import (COPY_FAILED_EXIT, DIGEST_MISMATCH_EXIT, check_sha256_hex, sha256_bytes,
                              sha256_file)
from hls.util import validation_errors


class Refused(Exception):
    """What the role read differs from the manifest it was given: the entry in the store, for
    ``copy``, or the copy, for ``verify``."""


class CopyFailed(Exception):
    """A copy differs from the file it was made from, which is sound: the copy went wrong, its
    bytes or the time it was given."""


def _plain_bytes(path: Path, what: str) -> bytes:
    """The bytes of the file *path*, called *what*, read once. Raise Refused unless it is a plain
    file."""
    try:
        plain = stat.S_ISREG(os.lstat(path).st_mode)
    except FileNotFoundError:
        plain = False
    if not plain:
        raise Refused(f"{what} is not a regular file")
    return path.read_bytes()


def _parsed(data: bytes, what: str) -> EntryManifest:
    """The entry manifest *data*, the bytes of the file called *what*. Raise Refused when they do
    not parse."""
    try:
        return EntryManifest.model_validate_json(data)
    except ValidationError as e:
        raise Refused(f"{what} is not a valid entry manifest: {validation_errors(e)}") from None


def _tree_problems(manifest: EntryManifest, root: Path, *, skip: set[str]) -> list[str]:
    """Every way the tree *root* differs from *manifest* by names and sizes. Raise Refused for
    what the cache's walk refuses."""
    try:
        have = {f.path: f.size for f in walk_files(root, skip=skip)}
    except CacheCorrupt as e:
        raise Refused(str(e)) from e
    want = {f.path: f.size for f in manifest.files}
    return ([f"missing {p}" for p in sorted(want.keys() - have.keys())]
            + [f"unexpected {p}" for p in sorted(have.keys() - want.keys())]
            + [f"changed {p}" for p in sorted(want.keys() & have.keys()) if want[p] != have[p]])


def _set_mtime(path: Path, mtime_ns: int) -> int:
    """Give the file *path* the modification time *mtime_ns*, keeping its access time, and
    return the time that landed, which a filesystem keeping coarser times would round."""
    os.utime(path, ns=(os.lstat(path).st_atime_ns, mtime_ns), follow_symlinks=False)
    return os.lstat(path).st_mtime_ns


def copy_entry(entry: Path, expect: str, dest: Path) -> dict[str, str]:
    """Copy the entry in the directory *entry*, whose ``ENTRY.json`` must have the sha256
    *expect*, into the new directory *dest*, and return the sha256 of each file of the copy, read
    back once written, ``ENTRY.json`` among them. Each file of the copy is given the modification
    time its record holds once its bytes check. Raise Refused for an entry that differs from its
    manifest, CopyFailed for a copy that differs from a sound entry's file or kept another time
    than its record's, and FileExistsError when *dest* exists."""
    check_sha256_hex(expect)
    dest.mkdir()
    source = entry / ENTRY_FILE
    if not stat.S_ISREG(os.lstat(source).st_mode):
        raise Refused(f"the entry's {ENTRY_FILE} is not a regular file")
    shutil.copyfile(source, dest / ENTRY_FILE)
    data = _plain_bytes(dest / ENTRY_FILE, f"the copy of the entry's {ENTRY_FILE}")
    if sha256_bytes(data) != expect:
        stored = sha256_file(source)[0]
        if stored != expect:
            raise Refused(f"the entry's {ENTRY_FILE} has sha256 {stored}, not the {expect} its producer "
                          "recorded")
        raise CopyFailed(f"the copy of {ENTRY_FILE} differs from the entry's file, which has the recorded "
                         f"sha256 {expect}: {len(data)} bytes with sha256 {sha256_bytes(data)}")
    manifest = _parsed(data, f"the entry's {ENTRY_FILE}")
    problems = _tree_problems(manifest, entry, skip={ENTRY_FILE})
    if problems:
        raise Refused("; ".join(problems))
    digests = {ENTRY_FILE: expect}
    for record in manifest.files:
        target = dest / record.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(entry / record.path, target)
        got = sha256_file(target)
        if got != (record.sha256, record.size):
            if sha256_file(entry / record.path) != (record.sha256, record.size):
                raise Refused(f"changed {record.path}")
            raise CopyFailed(f"the copy of {record.path} differs from the entry's file, which matches its "
                             f"record: {got[1]} bytes with sha256 {got[0]}, not {record.size} with "
                             f"{record.sha256}")
        landed = _set_mtime(target, record.mtime_ns)
        if landed != record.mtime_ns:
            # The entry is sound and the bytes landed: only the copy's time went wrong.
            raise CopyFailed(mtime_problem(record.path, landed, record.mtime_ns))
        digests[record.path] = got[0]
    return digests


def verify_copy(copy: Path, expect: str) -> dict[str, str]:
    """Hash every file of the node-local *copy*, its ``ENTRY.json`` first, which must have the
    sha256 *expect*, against the manifest, holding each file's modification time against its
    record, and return each file's sha256, ``ENTRY.json`` among them. The copy's stamp is passed
    over. Raise Refused for whatever differs."""
    check_sha256_hex(expect)
    data = _plain_bytes(copy / ENTRY_FILE, f"its {ENTRY_FILE}")
    if sha256_bytes(data) != expect:
        raise Refused(f"its {ENTRY_FILE} has sha256 {sha256_bytes(data)}, not the {expect} its producer "
                      "recorded")
    manifest = _parsed(data, f"its {ENTRY_FILE}")
    problems = _tree_problems(manifest, copy, skip={ENTRY_FILE, VERIFIED_FILE})
    if problems:
        raise Refused("; ".join(problems))
    digests = {ENTRY_FILE: expect}
    for record in manifest.files:
        got = sha256_file(copy / record.path)
        if got != (record.sha256, record.size):
            raise Refused(f"changed {record.path}")
        mtime_ns = os.lstat(copy / record.path).st_mtime_ns
        if mtime_ns != record.mtime_ns:
            raise Refused(mtime_problem(record.path, mtime_ns, record.mtime_ns))
        digests[record.path] = got[0]
    return digests


def fill_entry(source: Path, dest: Path) -> dict[str, str]:
    """Copy every file of the tree *source* into *dest*, an existing empty directory, by the same
    paths and with the modification time each has in *source*, which the cache then records, and
    return the sha256 of each copy, read back once written and synced (``hash_tree``). Raise
    Refused for a tree the cache would refuse to publish, before any file is copied, CopyFailed
    for a copy that did not keep its file's time, and FileExistsError when *dest* holds
    anything."""
    if any(dest.iterdir()):
        raise FileExistsError(f"{dest} holds files already; a fill copies into an empty directory")
    try:
        files = walk_files(source)
    except CacheCorrupt as e:
        raise Refused(str(e)) from e
    for f in files:
        target = dest / f.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / f.path, target)
        landed = _set_mtime(target, f.mtime_ns)
        if landed != f.mtime_ns:
            raise CopyFailed(f"the copy of {f.path} kept mtime {landed} ns, not the {f.mtime_ns} of the tree's file")
    return hash_tree(dest)


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.store.cache_role")
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("copy")
    c.add_argument("--entry", type=Path, required=True)
    c.add_argument("--expect", required=True)
    c.add_argument("--dest", type=Path, required=True)
    c.add_argument("--digests-out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--copy", type=Path, required=True)
    v.add_argument("--expect", required=True)
    v.add_argument("--digests-out", type=Path, required=True)
    f = sub.add_parser("fill")
    f.add_argument("--source", type=Path, required=True)
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--digests-out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    target = args.copy if args.command == "verify" else args.dest
    if args.digests_out.resolve().is_relative_to(target.resolve()):
        raise ValueError(f"the report {args.digests_out} lies inside the copy {target}")
    try:
        if args.command == "copy":
            digests = copy_entry(args.entry, args.expect, args.dest)
        elif args.command == "fill":
            digests = fill_entry(args.source, args.dest)
        else:
            digests = verify_copy(args.copy, args.expect)
    except Refused as e:
        print(f"refused: {' '.join(str(e).split())}", file=sys.stderr)
        return DIGEST_MISMATCH_EXIT
    except CopyFailed as e:
        print(f"copy failed: {' '.join(str(e).split())}", file=sys.stderr)
        return COPY_FAILED_EXIT
    write_digests(args.digests_out, digests)
    return 0


if __name__ == "__main__":
    sys.exit(main())
