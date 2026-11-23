"""Stands in for the export's copy role (``python -m hls.store.export_copy``): it copies as the real
role does, then plays another writer at the destination the export claimed, ``DESTINATION`` in its
environment, as ``RACE`` there says: ``fill`` puts a file of its own in the claim, ``file``
replaces the claim with a file, ``swap`` replaces it with an empty directory of its own, after
which the role fails, and ``swap-partial`` does that to the directory the export made for its
copy, the one ``--to`` names is in, with a file of its own in it (``swap-partial-empty``: none).
With ``recorded`` it fails before it copies unless the attempt's record, ``CLAIM_RECORD`` in its
environment, names the claim and that directory."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

from hls.store.export_copy import main

destination = Path(os.environ["DESTINATION"])
partial = Path(sys.argv[sys.argv.index("--to") + 1]).parent
race = os.environ["RACE"]
if race == "recorded":
    record = json.loads(Path(os.environ["CLAIM_RECORD"]).read_text())
    if (record["claim"]["inode"], record["partial"]["inode"]) != (os.lstat(destination).st_ino,
                                                                  os.lstat(partial).st_ino):
        sys.exit(3)
code = main(sys.argv[1:])
if race == "fill":
    (destination / "theirs.txt").write_text("another writer's\n")
elif race == "file":
    destination.rmdir()
    destination.write_text("another writer's file\n")
elif race == "swap":
    destination.rmdir()
    destination.mkdir()
    sys.exit(1)
elif race in ("swap-partial", "swap-partial-empty"):
    shutil.rmtree(partial)
    partial.mkdir()
    if race == "swap-partial":
        (partial / "theirs.txt").write_text("another writer's\n")
    sys.exit(1)
sys.exit(code)
