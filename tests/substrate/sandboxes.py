"""Pinned images for tests that do not test pinning itself.

``stand_in`` is a ``PinnedImage`` for tests of a runtime's command line, which reads only the
extracted directory's path. ``pre_extract`` leaves, in a job's node-local scratch, the extracted
directory and manifest that ``StageContext`` would leave for an image file, as the fake
apptainer's ``build --sandbox`` makes it; a context's ``pin_file`` of that file then reuses it,
as it reuses a tree an earlier stage could not remove, and starts no extraction role, so a
test's roles are only its own. The context removes it when it closes, and ``image_file`` makes
it again for the next. ``install_read_log`` logs which files a role's ``cp`` and ``sha256sum``
read."""

from __future__ import annotations

import hashlib
import shlex
import shutil
from pathlib import Path

from hls.store.digest import sha256_file
from hls.substrate.container import PinnedImage
from hls.substrate.sandbox import sandbox_paths, write_manifest

#: What ``image_file`` writes when the file is missing.
IMAGE_BYTES = b"fake image\n"
#: Its sha256, which a context pins it by, as the role that wrote it would report it.
IMAGE_SHA256 = hashlib.sha256(IMAGE_BYTES).hexdigest()


def stand_in(root: Path) -> PinnedImage:
    """A pinned image whose extracted directory is *root*."""
    return PinnedImage(ref=str(root), file_sha256="0" * 64, check_sha256="1" * 64)


def pre_extract(scratch: Path, image_file: Path) -> Path:
    """Extract *image_file* into the job scratch *scratch* as the fake apptainer would, unless
    a manifest is there already, and return the extracted directory."""
    root, manifest = sandbox_paths(scratch, sha256_file(image_file)[0])
    if not manifest.exists():
        for d in ("bin", "etc", "usr/bin", "tmp"):
            (root / d).mkdir(parents=True, exist_ok=True)
        (root / ".fake-image").write_bytes(image_file.read_bytes())
        (root / ".fake-source").write_text(str(image_file))
        write_manifest(root, sha256_file(image_file)[0], manifest)
    return root


def install_read_log(directory: Path, log: Path) -> None:
    """Write into *directory* a ``cp`` and a ``sha256sum`` that append to *log* one line for
    each file they read, ``cp <source>``, ``sha256sum <file>`` for a file named as an argument
    and ``sha256sum < <file>`` for its standard input, then run the real ones."""
    directory.mkdir(parents=True, exist_ok=True)
    real_cp = shutil.which("cp", path="/usr/bin:/bin")
    real_sum = shutil.which("sha256sum", path="/usr/bin:/bin")
    q = shlex.quote(str(log))
    (directory / "cp").write_text(f'#!/bin/sh\n[ "$1" = -- ] && echo "cp $2" >> {q} || echo "cp $1" >> {q}\n'
                                  f'exec {real_cp} "$@"\n')
    (directory / "sha256sum").write_text(
        f'#!/bin/sh\nif [ $# -eq 0 ]; then echo "sha256sum < $(readlink /proc/self/fd/0)" >> {q}; '
        f'else for f in "$@"; do echo "sha256sum $f" >> {q}; done; fi\nexec {real_sum} "$@"\n')
    for name in ("cp", "sha256sum"):
        (directory / name).chmod(0o755)


def image_file(scratch: Path) -> Path:
    """A small image file in the job scratch *scratch*, extracted ahead (``pre_extract``)."""
    path = scratch / "images-in" / "img.sif"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(IMAGE_BYTES)
    pre_extract(scratch, path)
    return path
