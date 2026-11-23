"""The export's copy role (``hls export``, ``hls.stages.export``): it copies the files an export
publishes out of the store, into the directory the stage made for the copy beside the
destination.

    python -m hls.store.export_copy --root STORE --list FILE --to DIR --to-made INODE:CTIME_NS --report FILE

copies each store key ``FILE`` lists, NUL-separated (a name may hold a newline, never a NUL), from
under ``STORE`` to the same key under ``DIR``, the directory the stage made for the copy, and
writes one JSON object per copy to ``--report``: the key, the copy's sha256 and its size. It opens
``DIR`` once, each component of its path without following a link, and copies nothing unless it
is the directory ``--to-made`` names (``Made``), as the stage made it. Each directory of a key is
opened without following a link, under the store and under ``DIR``. A key that is not a relative
path inside the store, a source that is a link or not a regular file, a copy that is already
there, and a ``DIR`` that is not the stage's fail the program (exit 1, the reason on stderr); a
copy whose bytes differ from what was read fails it with ``COPY_FAILED_EXIT``, which a retry
copies anew.

What tells a directory the export made from one another writer put in its place (``Made``), and
how a directory outside the store is opened, each component without following a link
(``open_directory``), are the stage's as well as this role's."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import traceback
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from hls.store.digest import COPY_FAILED_EXIT

#: Bytes the copy role reads at a time.
CHUNK_BYTES = 1 << 20
#: Opens a directory: never through a link.
DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
#: Makes a file: only a new one, never through a link.
COPY_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
# Opens a source file: never through a link, and without blocking on a fifo.
_SOURCE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class Made(BaseModel):
    """A directory an export attempt made outside the store, by what tells it from one another
    writer put in its place: its inode and its change time once the attempt finished making it,
    which nothing the attempt does later changes. A filesystem may give another directory the
    inode once this one is gone, never an earlier change time. The device is left out: a cluster
    filesystem may number it otherwise on each node, and a retry may run on another."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Its inode number.
    inode: int
    #: Its change time, in nanoseconds.
    ctime_ns: int

    @classmethod
    def of(cls, found: os.stat_result) -> Made:
        """The directory ``stat`` described as *found*."""
        return cls(inode=found.st_ino, ctime_ns=found.st_ctime_ns)


def open_directory(path: Path) -> int:
    """A descriptor of the directory at *path*, an absolute path, each component of it opened from
    the root without following a link. Raise OSError for a link or anything but a directory on the
    way (ELOOP, ENOTDIR), or nothing there, and ValueError for a relative path."""
    if not path.is_absolute():
        raise ValueError(f"{path} is not an absolute path")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        for name in path.parts[1:]:
            below = os.open(name, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = below
    except BaseException:
        os.close(fd)
        raise
    return fd


def read_copy_report(report: Path) -> dict[str, tuple[str, int]] | None:
    """Each copy the copy role reported in *report*, its store key to its sha256 and size; None
    when the report cannot be read."""
    try:
        copies = {}
        for line in report.read_text(errors="surrogateescape").splitlines():
            entry = json.loads(line)
            copies[entry["key"]] = (str(entry["sha256"]), int(entry["size"]))
        return copies
    except (OSError, ValueError, KeyError, TypeError):
        return None


class CopyFailed(Exception):
    """A copy's bytes differ from what was read: it went wrong on the way, and a retry copies anew."""


def _parts(key: str) -> list[str]:
    """The names of *key*'s path, if it is a relative path with no empty, "." or ".." part; else
    raise ValueError."""
    parts = key.split("/")
    if not key or any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"{key!r} is not a relative path inside the store")
    return parts


@contextmanager
def _directory(base: int, names: Sequence[str], *, make: bool) -> Iterator[int]:
    """A descriptor of the directory *names* lead to under the directory open at *base*, each
    opened without following a link, each made first when *make* holds."""
    with ExitStack() as opened:
        fd = base
        for name in names:
            if make:
                try:
                    os.mkdir(name, 0o755, dir_fd=fd)
                except FileExistsError:
                    pass
            fd = os.open(name, DIR_FLAGS, dir_fd=fd)
            opened.callback(os.close, fd)
        yield fd


def _read_back(fd: int) -> tuple[str, int]:
    """The sha256 and size of the open file *fd*, read from its start."""
    digest, size = hashlib.sha256(), 0
    os.lseek(fd, 0, os.SEEK_SET)
    while chunk := os.read(fd, CHUNK_BYTES):
        digest.update(chunk)
        size += len(chunk)
    return digest.hexdigest(), size


def copy_file(key: str, root: int, to: int) -> tuple[str, int]:
    """Copy the store's *key* from under the store's root, open at *root*, to the same key under
    the directory open at *to*: read the source once,
    no link followed anywhere on its path, into a copy that must not exist yet, made with no link
    followed either, sync it, and return the copy's sha256, read back from what landed, and its
    size. Raise ValueError for a key that is not a relative path inside the store or a source that
    is not a regular file, OSError for a link or a copy already there, and CopyFailed when the
    copy's bytes differ from what was read."""
    *folders, name = _parts(key)
    read = hashlib.sha256()
    with _directory(root, folders, make=False) as source_dir, _directory(to, folders, make=True) as copy_dir:
        source = os.open(name, _SOURCE_FLAGS, dir_fd=source_dir)
        try:
            if not stat.S_ISREG(os.fstat(source).st_mode):
                raise ValueError(f"{key} is not a regular file")
            copy = os.open(name, COPY_FLAGS, 0o644, dir_fd=copy_dir)
            try:
                while chunk := os.read(source, CHUNK_BYTES):
                    os.write(copy, chunk)
                    read.update(chunk)
                os.fsync(copy)
            finally:
                os.close(copy)
        finally:
            os.close(source)
        landed = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=copy_dir)
        try:
            sha256, size = _read_back(landed)
        finally:
            os.close(landed)
    if sha256 != read.hexdigest():
        raise CopyFailed(f"the copy of {key} has sha256 {sha256}, not the {read.hexdigest()} read from the store")
    return sha256, size


def _made_arg(text: str) -> Made:
    """``--to-made``: a directory's inode and change time, ``INODE:CTIME_NS``."""
    inode, sep, ctime_ns = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError(f"{text!r} is not INODE:CTIME_NS")
    try:
        return Made(inode=int(inode), ctime_ns=int(ctime_ns))
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"{text!r} is not INODE:CTIME_NS") from e


def _parse(argv: Sequence[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.store.export_copy")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--to", type=Path, required=True)
    p.add_argument("--to-made", type=_made_arg, required=True)
    p.add_argument("--report", type=Path, required=True)
    return p.parse_args(list(argv))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    try:
        keys = [os.fsdecode(k) for k in args.list.read_bytes().split(b"\0") if k]
        with ExitStack() as held:
            to = open_directory(args.to)
            held.callback(os.close, to)
            if Made.of(os.fstat(to)) != args.to_made:
                raise ValueError(f"{args.to} is not the directory the export made for its copy")
            root = os.open(args.root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            held.callback(os.close, root)
            with open(args.report, "w") as report:
                for key in keys:
                    sha256, size = copy_file(key, root, to)
                    report.write(json.dumps({"key": key, "sha256": sha256, "size": size}) + "\n")
    except CopyFailed as e:
        print(f"copy failed: {e}", file=sys.stderr)
        return COPY_FAILED_EXIT
    except Exception as e:  # noqa: BLE001 -- the role's failure, reported on stderr
        traceback.print_exc(file=sys.stderr)
        print(f"the export's copy failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
