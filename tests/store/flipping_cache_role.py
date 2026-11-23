"""What a test runs in place of ``python -m hls.store.cache_role``: the real role program, whose
copies land one byte other than they read, as a copy that went wrong on the way would. It flips
every file of the entry but its manifest, or, with ``FLIP_ONLY`` set to a file name, only the
files of that name, such as ``ENTRY.json``. With ``HIDE_SOURCE`` set it then makes the source of
each flipped copy unreadable, so the role's second read of it fails, as a read error on the
shared filesystem would."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from hls.store import cache_role
from hls.store.cache import ENTRY_FILE

_copyfile = shutil.copyfile
_ONLY = os.environ.get("FLIP_ONLY")
_HIDE = bool(os.environ.get("HIDE_SOURCE"))


def _flipping(src, dst, **kw):
    _copyfile(src, dst, **kw)
    name = Path(dst).name
    if (name == _ONLY) if _ONLY else (name != ENTRY_FILE):
        with open(dst, "r+b") as f:
            first = f.read(1)
            f.seek(0)
            f.write(bytes([first[0] ^ 1]) if first else b"x")
        if _HIDE:
            os.chmod(src, 0)
    return dst


cache_role.shutil.copyfile = _flipping
sys.exit(cache_role.main(sys.argv[1:]))
