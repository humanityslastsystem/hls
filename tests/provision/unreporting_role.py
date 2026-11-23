"""What a test runs in place of ``python -m hls.provision.role``: the role as ``fake_role`` runs it,
whose digest report is then gone, as a role that wrote its files but died before it reported."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

report = Path(sys.argv[sys.argv.index("--digests-out") + 1]) if "--digests-out" in sys.argv else None
try:
    runpy.run_path(str(Path(__file__).with_name("fake_role.py")), run_name="__main__")
except SystemExit as done:
    if report is not None:
        report.unlink(missing_ok=True)
    raise SystemExit(done.code) from None
