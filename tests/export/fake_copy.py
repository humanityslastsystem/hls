"""Stands in for the export's copy role (``python -m hls.store.export_copy``): it copies one file into
the directory the stage made for it and reports that it copied nothing, as a copy role that lost
files would."""

from __future__ import annotations

import sys
from pathlib import Path

args = sys.argv[1:]
to = Path(args[args.index("--to") + 1])
(to / "a-copy-it-made.txt").write_text("copied, then lost track of\n")
Path(args[args.index("--report") + 1]).write_text("")
