"""The byte cap on every role's stdout and stderr.

    python -m hls.substrate.log_cap --keep N FILE

A contestant's code writes its own logs' bytes, and as many as it likes, and a harness role
such as the driver can log what a contestant's replies said. Unbounded, logs would fill the
store and make every later reader of the instance, the secret scan among them, take as long as
the contestant wants (a forced timeout is infra, retried and voided, which a contestant must
never be able to choose). So every role's step writes its stdout and stderr into a pipe each,
and this program, one per stream, keeps the first ``--keep`` bytes and the last ``--keep``
bytes of what came through it in ``FILE``, opened for appending. A role's findings, digests and
records go to files of their own, never only to its stdout.

The first bytes are written through as they arrive, so a start marker or the runtime's first
lines are in the file at once. The last bytes are held in memory, in a ring of ``--keep``
bytes allocated only once the first ones are written, and written when the pipe ends, after a
line that says how many bytes were dropped between the two (``dropped_line``); a stream no
longer than twice ``--keep`` is kept whole, with no such line. A capper that is killed loses
what its ring held, never what it already wrote. The capper knows no secret: a value its cut
went through is backed off from when the log is scrubbed (``cuts``,
``hls.substrate.secrets.Scrubber.scrub_file``), which drops the piece of it on either side of
the line and counts those bytes among the dropped.

A capper ignores SIGTERM, SIGINT and SIGHUP (``IGNORED_SIGNALS``) and reads until its pipe
ends, which happens when its writer and everything that inherited the pipe have ended. A job's
cancel signals every process of the batch step, the cappers among them, before the runner has
stopped the roles: a capper that ended then would lose its log's tail, and its role would end
``log_lost`` while the role itself was not yet stopped. Only SIGKILL ends it early. It sets the
signals aside once its interpreter has imported this module, so one that comes in those first
milliseconds still ends it.

``LOG_KEEP_BYTES`` is the figure the runner uses: 64 MiB at each end, which keeps a server log
of tens of MB whole. A capper at that figure peaks at about 79 MiB resident (the 64 MiB ring and
the interpreter), so a role's two take at most 158 MiB of the runner's reserve
while it runs."""

from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, NamedTuple

#: Bytes a contestant role's stdout and stderr each keep at their start and at their end.
LOG_KEEP_BYTES = 64 * 2**20
#: The signals a running capper ignores, so that only its pipe's end, or SIGKILL, ends it.
IGNORED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
#: Bytes read from the pipe at a time.
_READ_BYTES = 2**20


def dropped_line(dropped: int, keep: int) -> bytes:
    """The line written between a log's first and last *keep* bytes once *dropped* bytes
    between them were not kept. A scrub that backs off from the cut keeps fewer, so it says at
    most."""
    return (f"\n[hls: {dropped} bytes of this log were dropped here; at most its first and last {keep} "
            "bytes are kept]\n").encode()


#: ``dropped_line`` as found in a log.
DROPPED = re.compile(rb"\n\[hls: ([0-9]+) bytes of this log were dropped here; at most its first and last "
                     rb"([0-9]+) bytes are kept\]\n")
#: Bytes a ``dropped_line`` takes at most: its text and two counts of twenty digits.
DROPPED_LINE_MAX_BYTES = len(dropped_line(10**20 - 1, 10**20 - 1))


class Cut(NamedTuple):
    """Where a log says it dropped bytes: its ``dropped_line``."""

    #: The line's first byte, its newline.
    start: int
    #: The byte after its last.
    end: int
    #: The bytes it says were dropped.
    dropped: int
    #: The bytes it says are kept at most at either end.
    keep: int


def cuts(f: BinaryIO, *, chunk_bytes: int) -> list[Cut]:
    """Every ``dropped_line`` in the log open as *f*, read *chunk_bytes* at a time from its
    start."""
    found: list[Cut] = []
    f.seek(0)
    base, data = 0, b""
    while chunk := f.read(chunk_bytes):
        data += chunk
        searched = 0
        for m in DROPPED.finditer(data):
            found.append(Cut(start=base + m.start(), end=base + m.end(), dropped=int(m[1]), keep=int(m[2])))
            searched = m.end()
        # A line the chunk's end cut short is searched again with the next chunk.
        rest = max(searched, len(data) - DROPPED_LINE_MAX_BYTES)
        base, data = base + rest, data[rest:]
    return found


def cap(read: Callable[[int], bytes], out: Path, *, keep: int) -> None:
    """Copy what *read* gives, a read of at most so many bytes each call and b"" at the end, into
    *out* (appended), keeping the first and last *keep* bytes and saying how many were dropped
    between them."""
    with open(out, "ab", buffering=0) as f:
        head = 0
        #: The last bytes past the head, ``keep`` of them once full; ``at`` is where the next goes.
        ring: bytearray | None = None
        at = 0
        tail = 0
        while chunk := read(_READ_BYTES):
            if head < keep:
                take = chunk[:keep - head]
                f.write(take)
                head += len(take)
                chunk = chunk[len(take):]
                if not chunk:
                    continue
            if ring is None:
                ring = bytearray(keep)
            view = memoryview(chunk)[-keep:]
            first = min(len(view), keep - at)
            ring[at:at + first] = view[:first]
            ring[:len(view) - first] = view[first:]
            at = (at + len(view)) % keep
            tail += len(chunk)
        if ring is None:
            return
        if tail > keep:
            f.write(dropped_line(tail - keep, keep))
        # Written from the ring itself, so the end holds no second copy of it.
        view = memoryview(ring)
        if tail >= keep:
            f.write(view[at:])
        f.write(view[:at])


def errors_file(role_dir: Path, suffix: str) -> Path:
    """Where the capper of a role's log ending in *suffix* (``.out`` or ``.err``) appends its own
    errors: a file in the role's node-local directory *role_dir*. The stage context makes it
    with the directory, so a launch opens a file already there."""
    return role_dir / f"log-cap.{suffix.lstrip('.')}.err"


def spawn(out: Path, *, keep: int, python: str, errors: Path,
          popen=subprocess.Popen) -> tuple[int, subprocess.Popen]:
    """Start a capper of *out* run by *python* through *popen*, and return the write end of its
    pipe, which the caller hands the writer and then closes, and the capper's process. The
    capper holds only the pipe's read end, and this process none of it; its own stderr goes to
    *errors*, appended. *out* exists, as the file a step writes into directly does, before the
    capper starts."""
    with open(out, "ab"):
        pass
    read_end, write_end = os.pipe()
    try:
        with open(errors, "ab") as err:
            proc = popen([python, "-m", "hls.substrate.log_cap", "--keep", str(keep), str(out)], stdin=read_end,
                         stdout=subprocess.DEVNULL, stderr=err, start_new_session=True)
    except BaseException:
        os.close(write_end)
        raise
    finally:
        os.close(read_end)
    return write_end, proc


def lost(cappers: tuple[subprocess.Popen, ...] | list[subprocess.Popen]) -> str | None:
    """Why the logs *cappers* wrote are not whole: a capper that ended other than by its pipe
    ending, nonzero or by a signal; None when every one exited 0 or still runs. Its writer then
    got a broken pipe, which may have ended the role whatever its own code did."""
    ended = [f"the capper of {Path(c.args[-1]).name} exited {c.returncode}" for c in cappers
             if c.returncode not in (0, None)]
    return "; ".join(ended) or None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls.substrate.log_cap")
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("file", type=Path)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.keep < 1:
        p.error("--keep must be at least 1")
    stdin = sys.stdin.fileno()
    cap(lambda n: os.read(stdin, n), args.file, keep=args.keep)
    return 0


if __name__ == "__main__":
    for signum in IGNORED_SIGNALS:
        signal.signal(signum, signal.SIG_IGN)
    sys.exit(main())
