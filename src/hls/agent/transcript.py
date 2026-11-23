"""The build's transcript: the agent's logs copied out of the sandbox, with the stage's secrets
removed.

The agent_build stage runs this program as a host harness role once the sandbox has stopped,
with the agent's node-local logs as ``--in`` and the attempt's ``outputs/transcript`` as
``--out``::

    python -m hls.agent.transcript --in DIR --out DIR [--scrub VAR=NAME]...

Each ``--scrub`` names a variable of the role's environment, whose value the stage hands over in
the role's 0600 env file as one of its granted secrets, never on the command line, and the
secret's name. Each form of each such secret (``hls.substrate.secret_forms``, the runner's
scrubber's) is replaced as the bytes are copied: the value with ``<redacted:NAME>``, since the
sandbox held the arm's credential and an agent that printed its environment would otherwise put
it in the store; and the host of a value that is an http(s) URL, a route's base URL, when the host
identifies a machine, with ``<redacted:NAME.host>``, in whatever case it was printed, since a host
is case-insensitive and a CLI may print the address it dialled in either case. Forms are replaced
as the runner's scrubber replaces them, through the same primitive (``ByteScrubber``): longest
first, and one split across two reads is replaced whole.

The agent wrote every file here, and the program runs on the host, so it never follows a link:
``--in`` and every directory below it are opened relative to their parent without following a
symlink, a symlink is neither followed nor kept, and a device, fifo or socket is skipped without
blocking on it. Only regular files are copied, each keeping its relative path under
``--out``'s ``COPY_DIR``. The index and its temporary file are written beside that directory, in
``--out`` itself, so no name the agent gives an entry of its logs can meet them.

The transcript is evidence and is never scored, and the program runs after the commit point, so
nothing the agent put in its logs may fail it: a failure here would end a graded build as infra
that is not retried. So the walk keeps its own stack instead of recursing, and every entry it
cannot use is skipped and recorded, never raised: one it cannot open or read (with the errno,
``--in`` itself included), a symlink, a special file, a directory deeper than ``MAX_DEPTH``, a
path longer than ``MAX_PATH_BYTES``, a file that would take the copy past ``MAX_BYTES``, and
every name past the first ``MAX_ENTRIES`` the copy's listings hand out (``hls.agent.listing``),
so it never holds more names than that, however many the agent made. Only a failure to write the copy, the
machine's, ends the program.

``MAX_BYTES`` bounds what the program writes, the copy and its index together, since finalize's
scan must read both in its time: a file counts by the bytes written, scrubbed, which can be more
than the agent wrote (a short host replaced by its longer name), so one whose scrubbed bytes pass
the room left is removed and skipped; every entry of the index counts as it will be written
(``_Index``); and once the next entry would not fit, the walk stops with a last entry saying so,
for which room is kept from the start.

``INDEX.json`` is written last: each copied file's path relative to ``COPY_DIR``, its size and
sha256, and each skipped entry with why. It is the stage's one output for the transcript, and it pins every file
of the copy by its digest. With ``--outputs-out FILE`` the program reports the index's sha256
and size, keyed by its path under the outputs directory (``--out``'s name, then ``INDEX.json``:
``--out`` is a directory of the attempt's outputs directory), as
``hls.store.manifest.report_outputs`` records them, so the stage publishes it by its writer's
digest. It writes FILE in place and cuts it to the record's length, so a report file the stage
made larger beforehand takes no new space on a disk the agent may have filled. The program
imports only ``hls.substrate.secret_forms`` and ``hls.agent.listing`` beyond the standard library,
and those modules use only the standard library.

Exit 0 once ``INDEX.json`` is written and reported; 1 on anything else, with the reason on
stderr."""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

from hls.agent.listing import first_entries
from hls.substrate.secret_forms import ByteScrubber, PastLimit, SecretForm, forms_of

#: The index the program writes last, in ``--out``.
INDEX_FILE = "INDEX.json"
#: The directory of ``--out`` the copy goes into, apart from the index.
COPY_DIR = "logs"
#: Directory levels below ``--in`` the copy enters.
MAX_DEPTH = 64
#: Bytes of an entry's path below ``--in``, so that every copied path can be made under ``--out``.
MAX_PATH_BYTES = 1024
#: Bytes the copy and its index hold together.
MAX_BYTES = 8 << 30
#: Entries of the logs the copy looks at.
MAX_ENTRIES = 200_000
# Opens a directory below ``--in``, and ``--in`` itself: never through a symlink.
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# Opens a file: never through a symlink, and without blocking on a fifo.
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
# What opening a path through a symlink, or a directory that became a file, fails with.
_NOT_FOLLOWED = (errno.ELOOP, errno.ENOTDIR)


def scrubber_for(pairs: list[str], environ: dict[str, str]) -> ByteScrubber:
    """What replaces every form of the secret *NAME* each ``VAR=NAME`` of *pairs* names, which the
    variable holds in *environ* (``hls.substrate.secret_forms.forms_of``: its value, and the host
    of a value that is a URL when the host identifies a machine), the pairs' forms in their order,
    each pair's value before its host. A variable that is unset or empty is an error: the stage
    names only variables it put in the env file."""
    forms: list[SecretForm] = []
    for pair in pairs:
        var, sep, name = pair.partition("=")
        if not sep or not var or not name:
            raise ValueError(f"--scrub takes VAR=NAME, got {pair!r}")
        value = environ.get(var, "")
        if not value:
            raise ValueError(f"--scrub {var}: the variable is not set")
        forms += forms_of(name, value)
    return ByteScrubber(forms)


class Unreadable(Exception):
    """Reading one of the agent's files failed: that entry is skipped, never the copy."""

    def __init__(self, error: OSError) -> None:
        super().__init__(str(error))
        #: The read's error.
        self.error = error


def copy_scrubbed(src_fd: int, dst: Path, scrubber: ByteScrubber, *, limit: int | None = None) -> tuple[int, str]:
    """Copy the open file *src_fd* to *dst*, scrubbed (``ByteScrubber.copy``, so a form split
    between two reads is replaced whole); return the size and sha256 written. A failed read raises
    ``Unreadable``, a failed write OSError, and scrubbed bytes past *limit*, when given,
    ``PastLimit``, with no more than *limit* written: the file's scrubbed bytes passed the room the
    copy had left for it, and that entry is skipped, never the copy."""

    def read(size: int) -> bytes:
        try:
            return os.read(src_fd, size)
        except OSError as e:
            raise Unreadable(e) from None

    with open(dst, "wb") as out:
        return scrubber.copy(read, out.write, limit=limit)


def _why(error: OSError) -> str:
    """Why an entry that could not be opened or read was skipped, by its errno's name."""
    if error.errno in _NOT_FOLLOWED:
        return "symlink"
    return f"unreadable: {errno.errorcode.get(error.errno or 0, 'unknown error')}"


def _entry_bytes(entry: dict) -> int:
    """The most bytes *entry* adds to the index as ``write_index`` writes it: its lines indented
    two levels down, and the separator before it."""
    return len(json.dumps(entry, indent=2)) + 4 * (len(entry) + 2) + 4


#: The most digits a file's size takes in the index.
_SIZE_DIGITS = 20


class _Index:
    """The index as the walk builds it, and the bytes the copy and the index will hold together
    once written (``used``), never past ``MAX_BYTES``: room for the entry that ends a walk the
    bound stopped (``stop``) is kept from the start."""

    def __init__(self) -> None:
        #: Each copied file, in walk order.
        self.files: list[dict] = []
        #: Each skipped entry with why, in walk order.
        self.skipped: list[dict] = []
        #: The entry that ends a walk the bound stopped.
        self.stop = {"path": ".", "why": f"the copy and its index reached {MAX_BYTES} bytes: the rest is not read"}
        #: The bytes the copy written so far and the index as it stands will hold.
        self.used = len(json.dumps({"files": [], "skipped": []}, indent=2)) + _entry_bytes(self.stop)
        #: Whether ``stop`` ended the walk.
        self.stopped = False

    def room(self, path: str) -> int:
        """The bytes a file at *path* may still be written in, its index entry's bytes counted."""
        entry = {"path": path, "size": 10 ** _SIZE_DIGITS - 1, "sha256": "0" * 64}
        return MAX_BYTES - self.used - _entry_bytes(entry)

    def add_file(self, path: str, size: int, sha256: str) -> None:
        """Name the file written at *path*, which ``room`` left space for."""
        entry = {"path": path, "size": size, "sha256": sha256}
        self.files.append(entry)
        self.used += size + _entry_bytes(entry)

    def add_skipped(self, path: str, why: str) -> bool:
        """Name the entry at *path* skipped for *why*, and return True; or, when its entry would
        take the index past the bound, end the walk with ``stop`` and return False."""
        entry = {"path": path, "why": why}
        if self.stopped:
            return False
        if self.used + _entry_bytes(entry) > MAX_BYTES:
            self.skipped.append(self.stop)
            self.stopped = True
            return False
        self.skipped.append(entry)
        self.used += _entry_bytes(entry)
        return True

    def record(self) -> dict:
        """The index as ``write_index`` writes it."""
        return {"files": self.files, "skipped": self.skipped}


def copy_tree(src: Path, dst: Path, scrubber: ByteScrubber) -> dict:
    """Copy the regular files below the directory *src* into *dst*, scrubbed, depth first in
    name order, and return the index. Every entry it cannot use is skipped and recorded, *src*
    itself included: a symlink, a special file, one it cannot open or read, one past
    ``MAX_DEPTH`` or ``MAX_PATH_BYTES``, one whose bytes, scrubbed, the room left under
    ``MAX_BYTES`` cannot hold, and everything past ``MAX_ENTRIES``; the walk stops once the index
    could not name the next entry within ``MAX_BYTES`` (``_Index``). Only a failure to write into
    *dst* raises."""
    index = _Index()
    try:
        top = os.open(src, _DIR_FLAGS)
    except OSError as e:
        index.add_skipped(".", _why(e))
        return index.record()
    # Names every listing has handed out so far: the copy reads at most MAX_ENTRIES of them in all,
    # so it never holds more, however many the agent made.
    listed = [0]
    # The open directories from the top down to the one being copied, each with its path below
    # src, its depth and the names it still holds, last name first. Every descriptor still on it
    # is closed on the way out.
    frames = [(top, "", 0, _names(top, "", index, listed))]
    try:
        while frames:
            fd, rel, depth, names = frames[-1]
            if not names:
                frames.pop()
                os.close(fd)
                continue
            name = names.pop()
            path = f"{rel}/{name}" if rel else name
            if len(os.fsencode(path)) > MAX_PATH_BYTES:
                if not index.add_skipped(path, f"its path is longer than {MAX_PATH_BYTES} bytes"):
                    break
                continue
            opened = _open(fd, name, path, depth, index)
            if index.stopped:
                break
            if opened is None:
                continue
            handle, is_dir, size = opened
            if is_dir:
                # Made one level at a time as the walk enters it, so making it never recurses.
                frames.append((handle, path, depth + 1, _names(handle, path, index, listed)))
                (dst / path).mkdir(exist_ok=True)
                if index.stopped:
                    break
                continue
            try:
                room = index.room(path)
                why = None
                if size > room:
                    why = f"its {size} bytes would take the copy past {MAX_BYTES}"
                else:
                    os.set_blocking(handle, True)
                    try:
                        written, sha = copy_scrubbed(handle, dst / path, scrubber, limit=room)
                    except Unreadable as e:
                        why = _why(e.error)
                    except PastLimit:
                        why = f"its bytes, scrubbed, would take the copy past {MAX_BYTES}"
                    if why is not None:
                        (dst / path).unlink(missing_ok=True)
                if why is None:
                    index.add_file(path, written, sha)
                elif not index.add_skipped(path, why):
                    break
            finally:
                os.close(handle)
    finally:
        for fd, _, _, _ in frames:
            os.close(fd)
    return index.record()


def _names(fd: int, rel: str, index: _Index, listed: list[int]) -> list[str]:
    """The names in the directory open at *fd*, *rel* below the top, last name first, among the
    first the copy may still read: ``MAX_ENTRIES`` less those every listing has handed out so far
    (*listed*, a one-element counter this adds to), read one at a time (``first_entries``). A
    directory with more is recorded in *index* as skipped past them, and one that cannot be
    listed as skipped with why."""
    try:
        entries, more = first_entries(fd, max(0, MAX_ENTRIES - listed[0]))
    except OSError as e:
        index.add_skipped(rel or ".", _why(e))
        return []
    listed[0] += len(entries)
    if more:
        index.add_skipped(rel or ".", f"past the first {MAX_ENTRIES} entries: the rest is not read")
    return sorted((entry.name for entry in entries), reverse=True)


def _open(fd: int, name: str, path: str, depth: int, index: _Index) -> tuple[int, bool, int] | None:
    """Open the entry *name* of the directory open at *fd*, *depth* below the top, without
    following a link or blocking, and return its descriptor, whether it is a directory and its
    size; None, with the entry skipped in *index* and why, for one the copy does not use. The
    type is read before opening, since opening a socket fails outright."""
    try:
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
    except OSError as e:
        index.add_skipped(path, _why(e))
        return None
    if stat.S_ISLNK(info.st_mode):
        index.add_skipped(path, "symlink")
        return None
    is_dir = stat.S_ISDIR(info.st_mode)
    if not is_dir and not stat.S_ISREG(info.st_mode):
        index.add_skipped(path, "not a regular file")
        return None
    if is_dir and depth + 1 > MAX_DEPTH:
        index.add_skipped(path, f"deeper than {MAX_DEPTH} levels")
        return None
    try:
        handle = os.open(name, _DIR_FLAGS if is_dir else _FILE_FLAGS, dir_fd=fd)
    except OSError as e:
        index.add_skipped(path, _why(e))
        return None
    try:
        now = os.fstat(handle)
    except OSError as e:
        os.close(handle)
        index.add_skipped(path, _why(e))
        return None
    if stat.S_ISDIR(now.st_mode) != is_dir or not (is_dir or stat.S_ISREG(now.st_mode)):
        # It changed between the look and the open: whatever it is now, the copy does not use it.
        os.close(handle)
        index.add_skipped(path, "not a regular file")
        return None
    return handle, is_dir, now.st_size


def write_index(out: Path, index: dict) -> tuple[str, int]:
    """Write *index* as ``INDEX.json`` in *out*, whole and synced, through a name beside it; return
    the sha256 and size of the bytes written."""
    data = json.dumps(index, indent=2).encode()
    partial = out / f".tmp-{INDEX_FILE}"
    with open(partial, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, out / INDEX_FILE)
    return hashlib.sha256(data).hexdigest(), len(data)


def report_index(report: Path, name: str, sha256: str, size: int) -> None:
    """Report the index, output *name*, with its *sha256* and *size* in *report*, written in place
    and cut to the record's length (``hls.store.manifest``)."""
    data = json.dumps({name: {"sha256": sha256, "size": size}}, indent=1).encode()
    fd = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.ftruncate(fd, len(data))
        os.fsync(fd)
    finally:
        os.close(fd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hls.agent.transcript", description="Copy the agent's logs, scrubbed.")
    ap.add_argument("--in", dest="src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scrub", action="append", default=[])
    ap.add_argument("--outputs-out", type=Path)
    args = ap.parse_args(argv)
    try:
        scrubber = scrubber_for(args.scrub, dict(os.environ))
        (args.out / COPY_DIR).mkdir(parents=True, exist_ok=True)
        index = copy_tree(args.src, args.out / COPY_DIR, scrubber)
        sha256, size = write_index(args.out, index)
        if args.outputs_out is not None:
            report_index(args.outputs_out, f"{args.out.name}/{INDEX_FILE}", sha256, size)
    except (OSError, ValueError) as e:
        print(f"hls-transcript: {e}", file=sys.stderr)
        return 1
    print(f"hls-transcript: {len(index['files'])} files, {len(index['skipped'])} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
