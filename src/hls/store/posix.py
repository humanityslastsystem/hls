"""The store on a POSIX filesystem shared by login and compute nodes (such as GPFS).

Writes go to a dotfile temp in the destination directory and are then published by
``os.replace`` (replace) or ``os.link`` (create-only). ``link`` fails if the name
exists, which makes create-only atomic across nodes without ``flock``."""

from __future__ import annotations

import errno
import os
import stat
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from hls.store.base import AlreadyExists, NotFound, Store, StoreError, check_key, check_segment

#: Directory under the root that holds private staging directories. Its leading dot keeps
#: it out of ``list_keys`` and out of reach of any key.
STAGING = ".staging"
# What opening or unlinking a path raises when no object is there: nothing at the path,
# a directory at the path, or a file where a parent directory should be.
_NO_OBJECT = (FileNotFoundError, IsADirectoryError, NotADirectoryError)


def _temp_beside(dest: Path) -> Path:
    return dest.parent / f".tmp-{dest.name}-{uuid.uuid4().hex}"


def _unlink_quietly(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _write_synced(path: Path, data: bytes) -> None:
    with open(path, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


class PosixStore(Store):
    """A store rooted at a directory. Only regular files are objects: a key naming a
    directory holds no object, as with an S3 prefix."""

    def __init__(self, root: Path) -> None:
        #: Directory every key is relative to.
        self.root = Path(root)

    def path(self, key: str) -> Path:
        """The filesystem path of *key*."""
        return self.root / check_key(key)

    def read_bytes(self, key: str) -> bytes:
        try:
            return self.path(key).read_bytes()
        except _NO_OBJECT as e:
            raise NotFound(key) from e

    def write_bytes(self, key: str, data: bytes) -> None:
        dest = self.path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = _temp_beside(dest)
        try:
            _write_synced(tmp, data)
            os.replace(tmp, dest)
        finally:
            _unlink_quietly(tmp)

    def create(self, key: str, data: bytes) -> None:
        dest = self.path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = _temp_beside(dest)
        try:
            _write_synced(tmp, data)
            try:
                os.link(tmp, dest)
            except FileExistsError as e:
                raise AlreadyExists(key) from e
        finally:
            _unlink_quietly(tmp)

    def exists(self, key: str) -> bool:
        return self.path(key).is_file()

    def size(self, key: str) -> int:
        """The size of the file at *key*, from ``lstat``: a link or a directory there is no object."""
        try:
            st = os.lstat(self.path(key))
        except (FileNotFoundError, NotADirectoryError) as e:
            raise NotFound(key) from e
        if not stat.S_ISREG(st.st_mode):
            raise StoreError(f"{key} is not a regular file")
        return st.st_size

    def list_keys(self, prefix: str) -> list[str]:
        base = self.path(prefix)
        if base.is_file():
            return [prefix]
        if not base.is_dir():
            return []
        out = []
        for p in base.rglob("*"):
            rel = p.relative_to(self.root).as_posix()
            if p.is_file() and not any(part.startswith(".") for part in rel.split("/")):
                out.append(rel)
        return sorted(out)

    def list_children(self, prefix: str) -> list[str]:
        try:
            with os.scandir(self.path(prefix)) as entries:
                return sorted(e.name for e in entries if not e.name.startswith("."))
        except (FileNotFoundError, NotADirectoryError):
            return []

    def delete(self, key: str) -> None:
        # A directory at the key holds no object, so it counts as missing and is left alone.
        try:
            self.path(key).unlink()
        except _NO_OBJECT:
            pass

    @contextmanager
    def open_write(self, key: str, *, exclusive: bool = False) -> Iterator[BinaryIO]:
        dest = self.path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = _temp_beside(dest)
        f = open(tmp, "wb")
        try:
            yield f
            f.flush()
            os.fsync(f.fileno())
            f.close()
            if exclusive:
                try:
                    os.link(tmp, dest)
                except FileExistsError as e:
                    raise AlreadyExists(key) from e
            else:
                os.replace(tmp, dest)
        finally:
            if not f.closed:
                f.close()
            _unlink_quietly(tmp)

    @contextmanager
    def open_read(self, key: str) -> Iterator[BinaryIO]:
        try:
            f = open(self.path(key), "rb")
        except _NO_OBJECT as e:
            raise NotFound(key) from e
        with f:
            yield f

    def staging_dir(self, area: str) -> Path:
        """A new, empty private directory to fill before ``publish_dir``. It lives under
        the store root, on the same filesystem, so the publish is a rename. *area* is one
        key segment, so the directory cannot land outside ``STAGING``."""
        d = self.root / STAGING / check_segment(area) / uuid.uuid4().hex
        d.mkdir(parents=True)
        return d

    def publish_dir(self, src: Path, key: str) -> None:
        """Move a fully written directory to *key* in one rename; AlreadyExists if
        *key* is taken. The caller must not touch *src* afterwards."""
        dest = self.path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            raise AlreadyExists(key)
        try:
            os.rename(src, dest)
        except OSError as e:
            if e.errno in (errno.EEXIST, errno.ENOTEMPTY):
                raise AlreadyExists(key) from e
            raise
