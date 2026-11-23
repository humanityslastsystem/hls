"""The secret scan role: find every form of every secret in an instance's published files.

    python -m hls.substrate.secret_scan --names NAME[,NAME...] --list FILE --report FILE [--raw FILE]
        [--unread FILE]

The values come from the role's environment, one variable per name, which the finalize stage
hands over in the role's 0600 env file; argv carries the names only. Each secret is searched in
every one of its forms (``hls.substrate.secret_forms``): its value as given and, for a value that
is an http(s) URL whose host identifies a machine, the host in any ASCII case. Each ``FILE`` lists
files and directories to scan, one path per line. Every regular file in them, and below each
directory, is read once, as a stream, and each read window overlaps the last by the longest form's
length, so a form split between two reads is still found. The walk is ``hls.util.walk_tree``'s: a
link is never followed, since its target is either in the scanned tree, and scanned there, or
outside the instance's outputs, and a device, a FIFO or a socket is never opened; each is reported
as not read (below), and an export never carries one. A file is read whatever its name: a
contestant's dot-files and dot-directories are its bytes like any other, and so is the dot-named
temp file of a writer that was killed, such as a trace's chunk the driver had not renamed yet. A
store's temporary file (``.tmp-<name>-<32 hex>``) needs no rule of its own: one its writer renamed
or removed while the scan ran is gone, and skipped as any file the walk found that is gone by the
time it is read (below); one a dead writer left holds what that writer was writing, and is read
like any file.

Every file's bytes are searched as stored. A file of ``--list`` that starts with gzip's magic
bytes is decompressed too, whatever its name, member after member, and the plain bytes are
searched as well: the harness writes traces and replies compressed, and their readers decompress
them, while a member's header (its file name, comment and extra field) is stored plain. A stream
cut short (one a writer that was killed left, or one with a corrupt block) does not fail the
scan: what it decompressed to before the cut is searched, with the whole file as stored, and the
report names the file as cut. A file of ``--raw``, or below a directory ``--raw`` names, is
searched as stored and never decompressed, even when ``--list`` holds it too: a role's log is
shown as it is, and a contestant writes its own log's and outputs' bytes, which read as gzip
could fail every scan (two magic bytes, a stream cut short) or expand without bound.

A file of ``--unread``, or below a directory it names, is not read at all, even when the other
lists hold it: a container whose bytes a caller has read otherwise or keeps out of every export
whole (``hls.stages.finalize``). Its path is searched like any other, and no ``READ_PREFIX``
line names it. A file it names itself must be there.

A directory that cannot be listed fails the scan, since the files below it would go unread. A
file the walk found that is gone by the time it is read is skipped: it is no longer published.
A path a list names itself must be there.

A form is found in a path too: every entry under the lists, files, directories and links, the
ones whose names start with a dot among them, is a finding of each form its path holds from
the listed path's own name down, so a file a contestant named after a value is quarantined or
refused as one holding it would be. A file below a directory so named holds the form in its
path too. What lies above a listed path, the store's root among it, is the site's and not
searched.

Each file read is one line of ``--report``, ``READ_PREFIX`` and a JSON object naming the file and
the ``sha256`` of its bytes as stored, every one of which was searched: whoever copies the file
later holds its copy to that digest, so the bytes copied are the bytes searched. Each form found is
one ``FOUND_PREFIX`` line, a JSON object naming the file and, as ``secret``, the form's label
(``SecretForm.label``: the secret's name for its value, ``<name>.host`` for its host), never the
value or the host; each stream cut short is one ``CUT_PREFIX`` line naming the file; each link or
special file is one ``UNREAD_PREFIX`` line, a JSON object naming it and, as ``kind``, ``link`` or
``special``. The lines go to a file of the role's own, not to its stdout, which the runner keeps
only the first and last bytes of (``hls.substrate.log_cap``); stdout says only how many of each
there were. The exit status is 0 once every file was read and the report written, whatever was
found: the findings are in the report alone, since a harness role's nonzero exit is a fault that
stops every role started after it. It is 1 for anything else, such as a listed file that is gone
or a directory that cannot be listed, with the traceback on stderr."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
import zlib
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import BinaryIO, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field

from hls.substrate.secret_forms import Form, SecretForm, registry_forms
from hls.util import EntryKind, entry_kind, walk_tree

#: Start of each report line that names a form found.
FOUND_PREFIX = "hls-secret-found: "
#: Start of each report line that names a gzip stream cut short.
CUT_PREFIX = "hls-secret-scan-cut: "
#: Start of each report line that names a file read, with the sha256 of its bytes as stored.
READ_PREFIX = "hls-secret-scan-read: "
#: Start of each report line that names a link or a special file the scan did not read.
UNREAD_PREFIX = "hls-secret-scan-unread: "
#: The name of the package stage's output that records what its scan of the submission's own
#: files found (``PackageScan``).
PACKAGE_SCAN_FILE = "package-scan.json"
#: Bytes read at a time.
CHUNK_BYTES = 1 << 20
#: The first bytes of a gzip member.
GZIP_MAGIC = b"\x1f\x8b"
#: The most plain bytes a gzip stream is decompressed to at a time, so a member that expands
#: without bound is searched a piece at a time.
_PLAIN_PIECE_BYTES = 1 << 20


#: What one form is searched by: its label, its bytes, lowered for a form found in any case, and
#: whether it is.
_Needle = tuple[str, bytes, bool]


def _needles(forms: Sequence[SecretForm]) -> list[_Needle]:
    """What each of *forms* is searched by."""
    return [(f.label, f.text.encode().lower() if f.any_case else f.text.encode(), f.any_case) for f in forms]


def _held(needles: list[_Needle], data: bytes) -> set[str]:
    """The labels of *needles* that *data* holds: a value as given, a host in any ASCII case."""
    folded = data.lower() if any(any_case for _, _, any_case in needles) else data
    return {label for label, needle, any_case in needles if needle in (folded if any_case else data)}


class _Finder:
    """The labels of forms found in bytes fed to it in order, block after block: each block is
    searched with the end of the bytes before it, as long as the longest form, so a form split
    between two blocks is still found."""

    def __init__(self, forms: Sequence[SecretForm]) -> None:
        self._needles = _needles(forms)
        self._keep = max((len(needle) for _, needle, _ in self._needles), default=0)
        self._tail = b""
        #: The labels found so far.
        self.found: set[str] = set()

    def feed(self, block: bytes) -> None:
        window = self._tail + block
        self.found |= _held(self._needles, window)
        self._tail = window[-self._keep:] if self._keep else b""


def find(stream: BinaryIO, forms: Sequence[SecretForm], *, chunk: int = CHUNK_BYTES) -> set[str]:
    """The labels of *forms* that occur in *stream*, read *chunk* bytes at a time: a value as given,
    a host in any ASCII case."""
    finder = _Finder(forms)
    while block := stream.read(chunk):
        finder.feed(block)
    return finder.found


class PackageScanFinding(BaseModel):
    """A form of a secret a scan of the submission's plain files found in one of them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The file's path as the submission image holds it, such as ``/workspace/server.py``, with
    #: each form of each secret in it replaced by its marker (``<redacted:name>``,
    #: ``<redacted:name.host>``).
    path: str
    #: The label of the form it holds (``SecretForm.label``): the secret's name for its value,
    #: ``<name>.host`` for its host.
    secret: str


#: The areas of a submission a scan reads (``hls.package.scan``): ``bundle``, every object of the
#: source bundle, its history, messages and tags among them; ``tree``, the checked-out tree, which
#: the image holds at ``/workspace``; ``install``, the files the offline install added to the image.
ScanArea = Literal["bundle", "tree", "install"]


class PackageScanArea(BaseModel):
    """A form of a secret found in an area that a record does not list file by file: in more of the
    tree's or the install's files than it lists (``hls.package.scan.FINDINGS_PER_SECRET``), or in
    any object of the bundle, which has no path in the image. The area and how many of its files or
    objects hold it, never their paths. Finalize keeps the whole area out of any export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The label of the form found (``SecretForm.label``).
    secret: str
    #: The area its files or objects are in.
    area: ScanArea
    #: How many files or objects of the area hold it, the listed ones among them.
    count: int = Field(ge=1)


class PackageScan(BaseModel):
    """``package-scan.json``: what a scan of the submission's own files found while they were plain,
    before a bundle or an image compressed them: the source_scan stage's, of the bundle's objects
    and the checked-out tree, or the package stage's, of what the install added. Paths and the
    labels of secrets' forms, never a value or a host. Finalize treats each finding as it treats a
    form its own scan found, keeps each area in ``areas`` out of any export whole, and counts a
    container as read only when a record says it read the container's areas and searched every
    secret the site's file holds, in every form (``hls.stages.finalize``)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: The names of the secrets the scan searched for.
    secrets: list[str]
    #: The forms of each secret the scan searched (``hls.substrate.secret_forms.FORMS``): finalize
    #: and the export count a record as covering a container only when it searched every one.
    forms: list[Form]
    #: The areas the scan read, whole.
    read: list[ScanArea]
    #: Each file a form was found in, with the form's label, up to the scan's bound for each label.
    findings: list[PackageScanFinding]
    #: Each area a form was found in that ``findings`` does not list file by file.
    areas: list[PackageScanArea] = Field(default_factory=list)


class _Gunzip:
    """The plain bytes of the gzip members of a stream fed to it in order, as far as they go. A
    stream that ends inside a member, or holds a block zlib refuses, ends its plain bytes there and
    sets ``cut``."""

    def __init__(self) -> None:
        #: True once the stream turned out to be cut short.
        self.cut = False
        self._member = zlib.decompressobj(wbits=31)

    def feed(self, data: bytes) -> Iterator[bytes]:
        """The plain bytes *data*, the stored bytes that follow those fed before, decompress to,
        at most ``_PLAIN_PIECE_BYTES`` at a time."""
        while data and not self.cut:
            if self._member.eof:
                self._member = zlib.decompressobj(wbits=31)
            member = self._member
            try:
                plain = member.decompress(data, _PLAIN_PIECE_BYTES)
            except zlib.error:
                self.cut = True
                return
            if plain:
                yield plain
            data = member.unused_data if member.eof else member.unconsumed_tail

    def close(self) -> bytes:
        """The plain bytes the last member still held once the stream ended, and ``cut`` set when
        it ended inside that member."""
        if self.cut or self._member.eof:
            return b""
        try:
            rest = self._member.flush()
        except zlib.error:
            rest = b""
        self.cut = not self._member.eof
        return rest


class Scanned(NamedTuple):
    """What a scan of one file found (``scan_stream``)."""

    #: The labels of the forms found in it.
    labels: set[str]
    #: Whether it is a gzip stream cut short.
    cut: bool
    #: The sha256 of its bytes as stored, every one of which was searched.
    sha256: str


def scan_stream(path: Path, forms: Sequence[SecretForm], *, chunk: int = CHUNK_BYTES,
                decompress: bool = True) -> Scanned:
    """Read the file at *path* once, *chunk* bytes at a time, and search it for *forms*
    (``find``): its bytes as stored, and, when it is gzip and *decompress* holds, the plain bytes
    it decompresses to as far as they go. Give the labels found, whether it is a gzip stream cut
    short, and the sha256 of its bytes as stored."""
    stored, plain = _Finder(forms), _Finder(forms)
    digest = hashlib.sha256()
    gunzip: _Gunzip | None = None
    with open(path, "rb") as raw:
        head = raw.read(len(GZIP_MAGIC))
        if decompress and head == GZIP_MAGIC:
            gunzip = _Gunzip()
        block = head
        while block:
            digest.update(block)
            stored.feed(block)
            for piece in gunzip.feed(block) if gunzip is not None else ():
                plain.feed(piece)
            block = raw.read(chunk)
    if gunzip is not None:
        plain.feed(gunzip.close())
    return Scanned(labels=stored.found | plain.found, cut=gunzip is not None and gunzip.cut,
                   sha256=digest.hexdigest())


def scan_file(path: Path, forms: Sequence[SecretForm], *, chunk: int = CHUNK_BYTES,
              decompress: bool = True) -> set[str]:
    """The labels of *forms* in the file at *path* (``scan_stream``)."""
    return scan_stream(path, forms, chunk=chunk, decompress=decompress).labels


def _walk(path: Path) -> Iterator[Path]:
    """Every regular file at *path* and below it, whatever its name (``walk_tree``: never through
    a link). Raise FileNotFoundError for a *path* that is neither a file, a directory nor a link,
    and OSError for a directory below it that cannot be listed."""
    if path.is_symlink():
        return
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is neither a file nor a directory")
    yield from (entry.path for entry in walk_tree(path) if entry.kind == "file")


def files(paths: list[Path]) -> Iterator[Path]:
    """Every regular file of *paths* and below each directory among them, once each, in
    order. A listed path that does not exist raises FileNotFoundError."""
    seen: set[Path] = set()
    for path in paths:
        for file in _walk(path):
            if file not in seen:
                seen.add(file)
                yield file


def entries(paths: list[Path]) -> Iterator[tuple[Path, Path, EntryKind]]:
    """Each of *paths* and every entry below each directory among them, once each, with its path
    from the listed path's own name down and its kind (``walk_tree``): files, directories, links
    and special files, names starting with a dot among them, links never followed. A listed path
    that is gone, or a directory that cannot be listed, raises."""
    seen: set[str] = set()
    for path in paths:
        kind = entry_kind(os.lstat(path).st_mode)
        below = ((e.path, e.kind) for e in walk_tree(path)) if kind == "directory" else ()
        for entry, its_kind in ((path, kind), *below):
            if _same(entry) not in seen:
                seen.add(_same(entry))
                yield entry, entry.relative_to(path.parent), its_kind


def named_in(path: Path, forms: Sequence[SecretForm]) -> set[str]:
    """The labels of *forms* that *path* holds: a value as given, a host in any ASCII case."""
    return _held(_needles(forms), os.fsencode(path))


def _listed(listing: Path | None) -> list[Path]:
    """The paths *listing* names, one per line; none without a listing."""
    return [] if listing is None else [Path(line) for line in listing.read_text().splitlines() if line]


def _parse(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="hls.substrate.secret_scan")
    p.add_argument("--names", required=True)
    p.add_argument("--list", type=Path, required=True)
    p.add_argument("--raw", type=Path)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--unread", type=Path)
    return p.parse_args(argv)


def _same(path: Path) -> str:
    """*path* in one spelling, so that two lists naming one file name it alike."""
    return os.path.normpath(os.path.abspath(path))


def main(argv: list[str] | None = None) -> int:
    args = _parse(sys.argv[1:] if argv is None else argv)
    found, where, cut, unread = 0, set(), 0, 0
    try:
        forms = registry_forms({name: os.environ[name] for name in args.names.split(",")})
        report = open(args.report, "w")
        listed, raw_listed, unread_paths = _listed(args.list), _listed(args.raw), _listed(args.unread)
        named = {_same(p) for p in (*listed, *raw_listed)}
        gone = [p for p in unread_paths if not os.path.lexists(p)]
        if gone:
            raise FileNotFoundError(f"{gone[0]} is listed unread and is not there")
        skipped = {_same(f) for f in files(unread_paths)}
        raw = [f for f in files(raw_listed) if _same(f) not in skipped]
        stored = {_same(f) for f in raw} | skipped
        passes = [(f, True) for f in files(listed) if _same(f) not in stored] + [(f, False) for f in raw]
        with report:
            for file, decompress in passes:
                try:
                    scanned = scan_stream(file, forms, decompress=decompress)
                except FileNotFoundError:
                    if _same(file) in named:
                        raise
                    continue
                report.write(READ_PREFIX + json.dumps({"file": str(file), "sha256": scanned.sha256}) + "\n")
                if scanned.cut:
                    cut += 1
                    report.write(CUT_PREFIX + json.dumps({"file": str(file)}) + "\n")
                for label in sorted(scanned.labels):
                    found += 1
                    where.add(file)
                    report.write(FOUND_PREFIX + json.dumps({"file": str(file), "secret": label}) + "\n")
            for entry, below, kind in entries([*listed, *raw_listed]):
                for label in sorted(named_in(below, forms)):
                    found += 1
                    where.add(entry)
                    hit = {"file": os.fsdecode(entry), "secret": label}
                    report.write(FOUND_PREFIX + json.dumps(hit) + "\n")
                if kind in ("link", "special"):
                    unread += 1
                    report.write(UNREAD_PREFIX + json.dumps({"file": os.fsdecode(entry), "kind": kind}) + "\n")
    except Exception:
        traceback.print_exc(file=sys.stderr)
        print("secret scan failed before it read every file", file=sys.stderr)
        return 1
    files_found = f"{len(where)} file{'' if len(where) == 1 else 's'}"
    print(f"{found} value{'' if found == 1 else 's'} found in {files_found}; {cut} stream"
          f"{'' if cut == 1 else 's'} cut short; {unread} link{'' if unread == 1 else 's'} or special file"
          f"{'' if unread == 1 else 's'} not read")
    return 0


if __name__ == "__main__":
    sys.exit(main())
