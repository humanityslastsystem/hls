"""The wheels the wheel step left, read whole before the offline build installs them.

A zip declares each entry's size and CRC in its list of entries (its central directory), but
only decompressing the entry proves them. pip, installing a wheel, reads its ``.dist-info``
entries (``RECORD``, ``METADATA``, ``WHEEL``, ``entry_points.txt``) whole and streams the rest
out, and the integrity check reads ``entry_points.txt`` whole. A wheel step could leave a wheel
whose entries unpack to far more than they declare, or a metadata entry of gigabytes: the
offline build or the integrity check would then run out of memory, which reads as the
harness's failure and is retried. So before the build this role reads every wheel in the
wheels directory with bounded memory, and the submission is invalid when a wheel:
- is not a zip this role can read, or lists its entries in more than ``DIRECTORY_MAX_BYTES``,
  which zipfile reads into memory whole;
- takes what every wheel's entries declare together past ``INSTALL_MAX_BYTES``, the most the
  install may add to the venv;
- holds a ``.dist-info`` entry that declares more than ``METADATA_MAX_BYTES``;
- holds a module, an entry whose name ends ``.py``, that declares more than
  ``MODULE_MAX_BYTES``: pip compiles every module it installs, held whole, in the offline build,
  which would otherwise run out of memory there as the harness's failure;
- holds an entry packed by any method but stored or deflate (``PACKING_METHODS``). zipfile
  bounds what one read decompresses only for deflate: a bzip2 or lzma entry unpacks a whole
  compressed chunk into memory at once, here and in pip's extraction alike, so an honest entry
  of gibibytes would run either out of memory;
- holds an entry whose bytes are not the size and CRC it declares. Each entry is decompressed
  a chunk at a time, and the read stops one byte past its declared size;
- takes the paths every wheel's install writes together past ``PATHS_MAX``: each entry, each
  directory above an entry, counted once across the wheels, and each console or GUI script
  their entry points can declare, one per "=" in a ``.dist-info``'s ``entry_points.txt``. To
  count the directories at a bounded cost, an entry may be nested no deeper than a path of the
  submission's tree (``DEPTH_MAX``), and no name in its path may be longer than a filesystem
  holds (``NAME_MAX_BYTES``), which pip could not write anyway.
What the entries declare, their sizes and methods and the paths they name, is checked for every
wheel before any entry is decompressed; the scripts are counted as their entries are read. A
reason names an entry by its first ``ENTRY_NAME_CHARS`` characters.

Past the check every entry has the size it declares, so whatever pip or the integrity check
reads whole is bounded, and a build that then runs out of memory failed for the harness. The
paths bound what pip writes and the integrity check attributes, and the tree the runner records
when it extracts the built image: its base, the submission's tree (``TREE_MAX_ENTRIES``) and
the install, which adds to its paths at most a bytecode file or a ``__pycache__`` per path and
four records per wheel it installs, each wheel of at least three paths: under 3.4 times
``PATHS_MAX`` in all.

    python -m hls.package.wheel_check --wheels DIR --out FILE

writes the verdict to *FILE* (``WheelCheck``) and exits 0. An error reading a wheel is the
submission's when the wheel itself or the permissions the wheel step left explain it
(``unreadable_zip``); any other is the node's, recorded in ``node_error`` with its errno for
the stage to raise as a retryable fault. Any other exit is a failure of the check itself."""

from __future__ import annotations

import argparse
import copy
import errno
import os
import sys
import zipfile
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Self, TypeVar

from pydantic import BaseModel, ConfigDict, model_validator

from hls.package.checkout import TREE_MAX_DEPTH, TREE_MAX_NAME_BYTES
from hls.package.offline import INSTALL_MAX_GIB
from hls.spec.submission import PERMISSION_ERRNOS

#: The most bytes one ``.dist-info`` entry of a wheel may declare: pip reads the metadata
#: entries whole, and no real one comes near it.
METADATA_MAX_BYTES = 16 * 2**20
#: The most bytes a wheel may list its entries in: zipfile reads the list whole and keeps an
#: object per entry, near 200 MB at this size, and a real wheel's list is a few MB.
DIRECTORY_MAX_BYTES = 16 * 2**20
#: The most bytes every wheel's entries may declare together: as many as the install may add to
#: the venv (``INSTALL_MAX_GIB``).
INSTALL_MAX_BYTES = INSTALL_MAX_GIB * 2**30
#: The methods a wheel's entries may be packed by: stored, and deflate, the one compression
#: zipfile decompresses a bounded amount of at a time.
PACKING_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
#: The most paths every wheel's install may write together: each entry, each directory above one
#: and each script the entry points declare: room for the wheels of a large machine-learning
#: environment.
PATHS_MAX = 500_000
#: The most names a wheel entry's path may hold: as many as a path of the submission's tree.
DEPTH_MAX = TREE_MAX_DEPTH
#: The longest one name of a wheel entry's path may be, in bytes: the filesystem's own limit.
NAME_MAX_BYTES = TREE_MAX_NAME_BYTES
#: The most bytes one module, an entry whose name ends ``.py``, may declare: pip compiles every
#: module it installs to bytecode in the offline build, holding the module and its syntax tree
#: whole.
MODULE_MAX_BYTES = 128 * 2**20
#: Characters of an entry's name a reason gives: a name may be as long as a zip allows.
ENTRY_NAME_CHARS = 200
#: Bytes decompressed at a time.
CHUNK_BYTES = 2**20
_T = TypeVar("_T")


class WheelCheck(BaseModel):
    """What the check found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Why the wheels may not be installed, naming the wheel and its entry; None when every
    #: wheel reads as it declares, or when the node stopped the check.
    problem: str | None = None
    #: The node's error that stopped the check, naming the wheel; None unless it stopped it.
    node_error: str | None = None
    #: The errno of ``node_error``; set exactly with it.
    node_errno: int | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        if (self.node_error is None) != (self.node_errno is None):
            raise ValueError("a node error comes with its errno, and only a node error")
        if self.problem is not None and self.node_error is not None:
            raise ValueError("the check found a problem or the node stopped it, not both")
        return self


class _Refused(Exception):
    """A wheel that may not be installed, with the reason."""


def unreadable_zip(error: OSError) -> bool:
    """Whether *error*, raised reading a zip the submission produced, is the submission's: the
    permissions it left (``PERMISSION_ERRNOS``), a decompressor's own data error, which carries
    no errno, or a seek to an offset the zip's own records make invalid (EINVAL). Any other
    error is the node's."""
    return error.errno is None or error.errno == errno.EINVAL or error.errno in PERMISSION_ERRNOS


def _entries(f: BinaryIO, wheel: str) -> list[zipfile.ZipInfo]:
    """The entries the wheel *wheel*, open as *f*, lists, once its list is within
    ``DIRECTORY_MAX_BYTES``."""
    # zipfile's own reader of the end record, so the bound is on exactly what ZipFile then reads.
    end = zipfile._EndRecData(f)
    listed = None if end is None else end[zipfile._ECD_SIZE]
    if listed is not None and listed > DIRECTORY_MAX_BYTES:
        raise _Refused(f"the wheel {wheel} lists its entries in {listed} bytes, more than the "
                       f"{DIRECTORY_MAX_BYTES} allowed")
    with zipfile.ZipFile(f) as z:
        return z.infolist()


def is_entry_points(member: str) -> bool:
    """Whether the wheel entry *member* is a ``.dist-info``'s ``entry_points.txt``, where pip
    reads the scripts it writes."""
    top, _, rest = member.partition("/")
    return top.endswith(".dist-info") and rest == "entry_points.txt"


def _entry(info: zipfile.ZipInfo, wheel: str) -> str:
    """The entry *info* of the wheel *wheel*, for a reason: its name cut at
    ``ENTRY_NAME_CHARS``."""
    return f"the entry {info.filename[:ENTRY_NAME_CHARS]!r} of the wheel {wheel}"


class _Totals:
    """What the wheels read so far declare together."""

    def __init__(self) -> None:
        #: The bytes their entries declare.
        self.bytes = 0
        #: The paths their install writes, as far as counted (``PATHS_MAX``).
        self.paths = 0
        # Each directory above an entry so far, as a tree of names, so each counts once.
        self._dirs: dict[str, dict] = {}

    def declared(self, entries: list[zipfile.ZipInfo], wheel: str) -> None:
        """Add the wheel *wheel*'s *entries*. Raise ``_Refused`` for an entry packed by a method
        outside ``PACKING_METHODS``, a ``.dist-info`` entry past ``METADATA_MAX_BYTES``, every
        entry's bytes together past ``INSTALL_MAX_BYTES``, an entry nested past ``DEPTH_MAX`` or
        holding a name past ``NAME_MAX_BYTES``, or the paths past ``PATHS_MAX``."""
        for info in entries:
            what = _entry(info, wheel)
            if info.compress_type not in PACKING_METHODS:
                method = zipfile.compressor_names.get(info.compress_type, "unknown")
                raise _Refused(f"{what} is packed by method {info.compress_type} ({method}), not stored or deflated")
            self.bytes += info.file_size
            if self.bytes > INSTALL_MAX_BYTES:
                raise _Refused(f"the wheels declare {self.bytes} bytes in all by the entry "
                               f"{info.filename[:ENTRY_NAME_CHARS]!r} of {wheel}, more than the {INSTALL_MAX_GIB} GiB "
                               "the install may add")
            if info.filename.split("/")[0].endswith(".dist-info") and info.file_size > METADATA_MAX_BYTES:
                raise _Refused(f"{what} declares {info.file_size} bytes, more than the {METADATA_MAX_BYTES} a "
                               "wheel's metadata may hold")
            if info.filename.endswith(".py") and info.file_size > MODULE_MAX_BYTES:
                raise _Refused(f"{what} declares {info.file_size} bytes, more than the {MODULE_MAX_BYTES} a module "
                               "may hold: pip compiles each module it installs, held whole")
            self._path(info.filename, what)

    def _path(self, member: str, what: str) -> None:
        """Count the entry *member* and each directory above it that no entry so far had."""
        path = member.rstrip("/")
        depth = path.count("/") + 1
        if depth > DEPTH_MAX:
            raise _Refused(f"{what} is nested {depth} levels deep, more than the {DEPTH_MAX} allowed")
        names = path.split("/")
        longest = max(len(name.encode("utf-8", "surrogatepass")) for name in names)
        if longest > NAME_MAX_BYTES:
            raise _Refused(f"{what} holds a name of {longest} bytes, more than the {NAME_MAX_BYTES} a filesystem "
                           "holds")
        self.count(1, what)
        node = self._dirs
        for name in names[:-1]:
            below = node.get(name)
            if below is None:
                below = node[name] = {}
                self.count(1, what)
            node = below

    def count(self, paths: int, what: str) -> None:
        """Add *paths* written for *what*; raise ``_Refused`` once all of them pass
        ``PATHS_MAX``."""
        self.paths += paths
        if self.paths > PATHS_MAX:
            raise _Refused(f"the wheels name more than the {PATHS_MAX} paths their install may write, by {what}: "
                           "every entry, each directory above one, and each script their entry points declare")


def _unpacked(z: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, int, int]:
    """The bytes the entry *info* of *z* decompresses to, up to one past its declared size,
    their CRC, and how many "=" they hold, read a chunk at a time."""
    probe = copy.copy(info)
    probe.file_size = info.file_size + 1
    # zipfile checks an entry's CRC only when it has one; this role checks it after the size.
    del probe.CRC
    size, crc, equals = 0, 0, 0
    with z.open(probe) as member:
        while chunk := member.read(CHUNK_BYTES):
            size, crc, equals = size + len(chunk), zlib.crc32(chunk, crc), equals + chunk.count(b"=")
    return size, crc, equals


def _unpack(f: BinaryIO, wheel: str, totals: _Totals) -> None:
    """Decompress every entry of the wheel *wheel*, open as *f*, and add the scripts its entry
    points can declare to *totals*; raise ``_Refused`` for an entry whose bytes are not the size
    and CRC it declares, or that does not unpack, and as ``_Totals.count`` does."""
    with zipfile.ZipFile(f) as z:
        for info in z.infolist():
            what = _entry(info, wheel)
            try:
                size, crc, equals = _unpacked(z, info)
            except OSError as e:
                if not unreadable_zip(e):
                    raise
                raise _Refused(f"{what} cannot be unpacked: {e}") from e
            except Exception as e:  # noqa: BLE001
                # zipfile and its decompressors raise many kinds on bytes they cannot read.
                raise _Refused(f"{what} cannot be unpacked: {type(e).__name__}: {e}") from e
            if size > info.file_size:
                raise _Refused(f"{what} unpacks to more than the {info.file_size} bytes it declares")
            if size < info.file_size:
                raise _Refused(f"{what} unpacks to {size} bytes, not the {info.file_size} bytes it declares")
            if crc != info.CRC:
                raise _Refused(f"{what} unpacks to CRC {crc:08x}, not the {info.CRC:08x} it declares")
            if is_entry_points(info.filename):
                # Each entry point takes an "=", whatever its group and however its lines end.
                totals.count(equals, what)


def _read(wheels: Path, name: str, step: Callable[[BinaryIO, str], _T]) -> _T:
    """Run *step* on the wheel *name* in *wheels*, open; raise ``_Refused`` when the wheel
    cannot be read on the submission's account, and OSError for the node's."""
    try:
        with open(wheels / name, "rb") as f:
            return step(f, name)
    except OSError as e:
        if not unreadable_zip(e):
            raise
        raise _Refused(f"the wheel {name} cannot be read: {e}") from e
    except _Refused:
        raise
    except Exception as e:  # noqa: BLE001
        raise _Refused(f"the wheel {name} is not a zip this check can read: {type(e).__name__}: {e}") from e


def check_wheels(wheels: Path) -> WheelCheck:
    """Read every wheel in the directory *wheels*, in name order: first what each declares,
    then every entry's bytes."""
    try:
        names = sorted(os.listdir(wheels))
    except OSError as e:
        if e.errno not in PERMISSION_ERRNOS:
            return WheelCheck(node_error=f"the wheels directory: {e}", node_errno=e.errno)
        return WheelCheck(problem=f"the wheels directory cannot be listed: {e}")
    name = ""
    try:
        totals = _Totals()
        for name in names:
            totals.declared(_read(wheels, name, _entries), name)
        for name in names:
            _read(wheels, name, lambda f, wheel: _unpack(f, wheel, totals))
    except _Refused as e:
        return WheelCheck(problem=str(e))
    except OSError as e:
        return WheelCheck(node_error=f"the wheel {name}: {e}", node_errno=e.errno)
    return WheelCheck()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls.package.wheel_check")
    p.add_argument("--wheels", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    record = check_wheels(args.wheels)
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_text(record.model_dump_json(indent=2))
    os.replace(tmp, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
