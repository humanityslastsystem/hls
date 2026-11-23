"""What a test runs in place of ``python -m hls.store.cache_role``: the real role program, whose
report then names another digest for one file of the copy than the bytes it checked, as a role
with a fault in its report would."""

from __future__ import annotations

import sys
from pathlib import Path

from hls.store import cache_role
from hls.store.cache import ENTRY_FILE, read_digests, write_digests

code = cache_role.main(sys.argv[1:])
report = Path(sys.argv[sys.argv.index("--digests-out") + 1])
digests = read_digests(report) or {}
name = next(p for p in sorted(digests) if p != ENTRY_FILE)
write_digests(report, {**digests, name: "0" * 64})
sys.exit(code)
