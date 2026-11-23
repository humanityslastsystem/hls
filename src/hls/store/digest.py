"""SHA-256 helpers that stream, so a 50 GiB file is never held in memory, and the one
digest format the store records: 64 lowercase hex characters with no ``sha256:``
prefix."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Annotated, BinaryIO

from pydantic import AfterValidator

#: Bytes read per step when hashing a stream.
CHUNK = 8 * 1024 * 1024
#: Exit status of a harness role that refused a file whose sha256 is not the one recorded for
#: it, having printed a line starting ``refused:`` on stderr: an image file it was to extract or
#: build on, or a file of a cache entry it was to copy or check. Nothing is made from the file.
DIGEST_MISMATCH_EXIT = 65
#: Exit status of a harness role whose copy of a file differs from the file's recorded sha256
#: while the file itself, read again, has it: the copy went wrong on the way, and a retry copies
#: again. The role prints a line starting ``copy failed:`` and no ``refused:`` line. A role that
#: copies a recorded file reads the source again only on such a mismatch, and refuses the file
#: (``DIGEST_MISMATCH_EXIT``) only when the source differs too.
COPY_FAILED_EXIT = 75
_HEX64 = re.compile(r"[0-9a-f]{64}")


def check_sha256_hex(value: str) -> str:
    """Return *value* if it is a bare hex SHA-256, else raise ValueError. The whole
    string must match, so a prefix or a trailing newline is rejected."""
    if not _HEX64.fullmatch(value):
        raise ValueError(f"not a bare hex SHA-256: {value!r}")
    return value


#: A SHA-256 digest as recorded in manifests, cache entries and provenance.
Sha256Hex = Annotated[str, AfterValidator(check_sha256_hex)]


def sha256_bytes(data: bytes) -> str:
    """Hex SHA-256 of *data*."""
    return hashlib.sha256(data).hexdigest()


def sha256_stream(f: BinaryIO) -> tuple[str, int]:
    """Hex SHA-256 and size in bytes of everything left in *f*, read ``CHUNK`` bytes at a time."""
    h = hashlib.sha256()
    size = 0
    while chunk := f.read(CHUNK):
        h.update(chunk)
        size += len(chunk)
    return h.hexdigest(), size


def sha256_file(path: Path) -> tuple[str, int]:
    """Hex SHA-256 and size in bytes of the file at *path*."""
    with open(path, "rb") as f:
        return sha256_stream(f)


def reported_sha256(path: Path) -> str | None:
    """The sha256 a role's ``sha256sum`` wrote to *path*; None when the file is missing or holds
    no bare hex digest first."""
    try:
        words = path.read_text(errors="replace").split()
        return check_sha256_hex(words[0]) if words else None
    except (OSError, ValueError):
        return None
