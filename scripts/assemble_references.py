"""Assemble ``tasks/assets/references``, the offline toolchain documentation that multitenant_slo,
tab_completion and multimodal_tts upload into every build's sandbox, from the public sources
``hls.agent.toolchain_references`` pins, and check it against the tree digest the tasks pin.

    uv run python scripts/assemble_references.py [--dest DIR] [--cache DIR]

The library replaces its destination whole, so a destination that holds anything but a library
this script wrote is refused; one that already has the pinned digest is left as it is. Exit 0
when the destination is the pinned library, 1 when the library cannot be assembled or the one
assembled has another digest, after naming each file that differs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hls.agent.toolchain_references import assemble

#: The checkout this script belongs to.
ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assemble_references.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=ROOT / assemble.LIBRARY_DIR,
                        help="the library folder, replaced whole (default: tasks/assets/references in this "
                             "checkout, where the tasks' upload_dir points)")
    parser.add_argument("--cache", type=Path, default=assemble.DEFAULT_CACHE,
                        help=f"where clones and downloads are kept between runs (default: {assemble.DEFAULT_CACHE})")
    args = parser.parse_args(argv)
    try:
        exact = assemble.assemble_library(args.dest.expanduser().absolute(), args.cache.expanduser().absolute(),
                                          build=assemble.assemble)
    except assemble.AssemblyError as e:
        print(f"assemble_references.py: cannot assemble the library: {e}", file=sys.stderr)
        return 1
    return 0 if exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
