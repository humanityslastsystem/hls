"""The package stage's publish role: the built submission image into the attempt's outputs.

    python -m hls.package.publish --image FILE --out FILE --outputs-out R

copies the image file to *out* (under a temporary name, renamed into place whole), then reads
the copy back once and reports its sha256 and size in ``R`` (``report_outputs``): the stage
records the output by that digest, which must be the one it pinned the image by, and the runner
reads none of its bytes."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from hls.store.manifest import report_outputs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls.package.publish")
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    shutil.copyfile(args.image, tmp)
    os.replace(tmp, args.out)
    report_outputs(args.outputs_out, args.out.parent, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
