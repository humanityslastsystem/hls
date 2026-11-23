"""Generic, domain-free helpers. Lives here so domain modules don't accrete top-of-file
utility functions."""

from __future__ import annotations

import contextlib
import gzip
import io
import json
import math
import os
import re
import stat
import tomllib
from collections.abc import Collection, Iterator
from pathlib import Path
from typing import Any, Literal, NamedTuple, TextIO

from pydantic import ValidationError

#: Longest UNIX socket path the kernel binds, in bytes: ``sun_path`` holds 108, its NUL included.
MAX_SOCKET_PATH_BYTES = 107

#: How many of a validation's errors ``validation_errors`` names; the rest it counts, so that a
#: file wrong throughout still makes a refusal of one line.
SHOWN_VALIDATION_ERRORS = 5


def validation_errors(error: ValidationError) -> str:
    """The errors of *error*, each by its type and where it is, such as ``int_parsing
    inner.items.1``, joined by ``"; "``: the first ``SHOWN_VALIDATION_ERRORS``, then how many more.
    Never the input values, which a refusal must not print: a record's field may hold a secret, and
    a file that does not parse is shown nothing of. A missing required field is named with the
    hint that this version of hls requires it."""
    named = []
    for e in error.errors(include_url=False, include_context=False, include_input=False)[:SHOWN_VALIDATION_ERRORS]:
        where = ".".join(str(part) for part in e["loc"])
        named.append(f"{e['type']} {where}" if where else f"{e['type']} at the top level")
        if e["type"] == "missing":
            named[-1] += " (required by this version of hls)"
    more = error.error_count() - len(named)
    return "; ".join([*named, f"and {more} more"] if more else named)


#: Where a ``tomllib.TOMLDecodeError``'s message says the error is, at its end.
_TOML_POSITION = re.compile(r"\((at line [0-9]+, column [0-9]+|at end of document)\)$")


class TomlError(ValueError):
    """Bytes that do not load as TOML. Its message quotes nothing of them."""


def load_toml(data: bytes) -> dict[str, Any]:
    """*data* loaded as TOML. Raise TomlError, with no cause or context, saying why not: "not UTF-8
    text", or "not valid TOML" with the position the decoder names. The decoder's own words can
    quote a character of a value, so only its position is kept; too deep a nesting
    (RecursionError), too long an integer (a plain ValueError) or any other error it raises has
    none."""
    try:
        return tomllib.loads(data.decode())
    except UnicodeDecodeError:
        problem = "not UTF-8 text"
    except Exception as e:
        at = _TOML_POSITION.search(str(e)) if isinstance(e, tomllib.TOMLDecodeError) else None
        problem = "not valid TOML" + (f" ({at.group(1)})" if at is not None else "")
    raise TomlError(problem)


class JsonDepthError(ValueError):
    """JSON text nested deeper than its reader allows, refused before it was parsed."""


def _json_depth(data: bytes) -> int:
    """The deepest nesting of objects and arrays in the UTF-8 JSON text *data*, brackets inside a
    string, after an escaped quote too, not counted. It reads the bytes in one pass and never
    recurses."""
    depth = deepest = 0
    in_string = escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # a backslash escapes the byte after it
                escaped = True
            elif byte == 0x22:
                in_string = False
        elif byte == 0x22:
            in_string = True
        elif byte in b"[{":
            depth += 1
            deepest = max(deepest, depth)
        elif byte in b"]}":
            depth -= 1
    return deepest


def load_json(data: bytes, *, max_depth: int) -> Any:
    """*data* loaded as UTF-8 JSON. One nested deeper than *max_depth* raises JsonDepthError before
    it is parsed: the parser recurses once a level, and from Python 3.14 as deep as the C stack
    allows, so what an unbounded text costs and whether it parses would depend on the interpreter
    and the stack. The parse reads the text decoded as UTF-8, the encoding the depth was counted
    in: ``json.loads`` of bytes also reads UTF-16 and UTF-32, whose code units can hold a quote's
    byte and so hide brackets from the count. Any other way *data* fails raises as decoding or
    ``json.loads`` raises it, a ValueError or MemoryError."""
    if _json_depth(data) > max_depth:
        raise JsonDepthError(f"nested deeper than {max_depth} levels")
    return json.loads(data.decode())


def json_safe(obj: Any) -> Any:
    """Coerce a value to STRICT-JSON-native types so it can always be serialized.

    Dicts/lists recurse; ``str``/``bool``/``int``/``None`` pass through unchanged;
    a ``float`` passes through unless it is ``inf``/``-inf``/``nan`` -- those are not
    valid RFC-8259 JSON (``json.dumps`` emits the bare literals ``Infinity``/``NaN``,
    which strict consumers reject), so they are coerced to ``None``. Anything else
    (e.g. a numpy/torch scalar) becomes ``float`` when possible (re-checking for
    inf/nan), else ``str``. Used to keep the verdict, the harness-conditions table and
    the raw provenance writes serializable whatever a producer emits."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    return f if math.isfinite(f) else None


def write_in_place(path: Path, data: bytes) -> None:
    """Write *data* into the file *path* from its first byte, made if it is not there, then cut
    the file to it. A file a stage made larger beforehand, on a node disk a contestant may have
    filled since, so takes no new space."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.ftruncate(fd, len(data))
    finally:
        os.close(fd)


@contextlib.contextmanager
def gzip_text_writer(path: Path) -> Iterator[TextIO]:
    """A UTF-8 text stream that writes the new file *path* as one gzip member whose header holds
    no file name and a modification time of zero, so the same text always makes the same bytes:
    a harness record's digest then says whether its content changed."""
    with open(path, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as packed, \
            io.TextIOWrapper(packed, encoding="utf-8") as text:
        yield text


def remove_tree(path: Path, *, ignore_errors: bool = False) -> None:
    """Remove *path*: a directory with everything under it, or a file or a link itself. A link
    is never followed. It walks with a list, not by recursing, so a tree of any depth is removed;
    ``shutil.rmtree`` before Python 3.12.4 recurses once a level and raises RecursionError on a
    tree deeper than the interpreter's limit, which a role can make in a directory it writes.
    Nothing at *path* is not an error. An OSError is raised, or with *ignore_errors* passed over,
    leaving what could not be removed."""

    def attempt(op, target) -> bool:
        try:
            op(target)
        except FileNotFoundError:
            pass
        except OSError:
            if not ignore_errors:
                raise
            return False
        return True

    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(mode):
        attempt(os.unlink, path)
        return
    # Each directory is listed when it is first taken off the list and removed when it comes
    # off a second time, after everything under it.
    pending: list[tuple[str, bool]] = [(os.fspath(path), False)]
    while pending:
        directory, listed = pending.pop()
        if listed:
            attempt(os.rmdir, directory)
            continue
        pending.append((directory, True))
        try:
            with os.scandir(directory) as entries:
                children = [(e.path, e.is_dir(follow_symlinks=False)) for e in entries]
        except FileNotFoundError:
            continue
        except OSError:
            if not ignore_errors:
                raise
            continue
        for child, is_dir in children:
            if is_dir:
                pending.append((child, False))
            else:
                attempt(os.unlink, child)


#: What an entry of a tree is (``entry_kind``): a regular file, a directory, a symbolic link, or a
#: special file, which is a device, a FIFO or a socket.
EntryKind = Literal["file", "directory", "link", "special"]


def entry_kind(mode: int) -> EntryKind:
    """The kind of an entry whose ``lstat`` gives *mode*."""
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "link"
    return "special"


class WalkEntry(NamedTuple):
    """An entry ``walk_tree`` found below its root."""

    #: Its path: the root's, joined with ``rel``.
    path: Path
    #: Its path below the root, "/"-separated.
    rel: str
    #: What it is (``entry_kind``).
    kind: EntryKind
    #: Its ``lstat``, taken when its directory was listed.
    stat: os.stat_result


def walk_tree(root: Path, *, skip: Collection[Path] = ()) -> Iterator[WalkEntry]:
    """Every entry below the directory *root*, each once, with its kind and its ``lstat``, in the
    order ``sorted`` gives their paths: a directory's entries by name, and everything below a
    directory before the directory's next sibling. It reads no file and follows no link, and walks
    with a list, not by recursing, so a tree of any depth is walked. One rule says what each entry
    is, and every entry is yielded, so no caller passes one over without deciding what it means:
    - a regular file (``file``) is for the caller to read;
    - a directory (``directory``) is yielded, then entered, unless it is among *skip*, which leaves
      it out with everything below it, unlisted;
    - a symbolic link (``link``) is never followed, to a file or to a directory: what it points to
      is either below *root*, and found there, or not the tree's;
    - a device, a FIFO or a socket (``special``) is never opened: a FIFO blocks its reader, and a
      device hands over the host's bytes;
    - a directory that cannot be listed, for want of permission or because it is gone since its own
      directory was listed, raises the OSError its listing raised: what lies below it would go
      unread, and every caller fails closed on it;
    - an entry gone between its directory's listing and its ``lstat`` raises the
      FileNotFoundError its ``lstat`` raised: every tree a caller walks holds still while it is
      walked, so an entry that goes means the tree changed under the walk, and it would go unread.
    *root* itself is listed as given, a link there followed as ``os.scandir`` follows one: a
    caller that must not follow a link at *root* looks at it first. A *root* that is no directory
    raises as ``os.scandir`` does.

    What each caller reads, and what it does with the rest:
    - ``hls.stages.finalize.tree_files`` (finalize's list of what it scans and its containers, an
      export's count of the instance's bytes and its choice of what leaves): the files; links and
      special files are the others, which an export neither counts nor carries as no regular
      files, and finalize does not list for its scan;
    - ``hls.substrate.secret_scan`` (finalize's and an export's scan): the files' bytes, and every
      entry's path; each link and special file is reported as not read, which finalize names in an
      event, and an export never carries one;
    - ``hls.package.scan`` (the checked-out tree and what the install added): the files' bytes,
      every entry's path, and the target a link holds, which the image carries as it is; a special
      file holds nothing else;
    - ``hls.store.cache.walk_files`` (the cache's check of a fill, an entry or a copy): refuses a
      link and a special file (``CacheCorrupt``);
    - ``hls.store.cache.hash_tree`` (a role's digests of what it wrote): the files; a link or a
      special file is left to ``walk_files``, which refuses the tree that holds it;
    - ``hls.agent.uploads.tree_digest`` (a task's uploaded folder): refuses a link and a special
      file, and a directory it cannot list (``TreeDigestError``)."""
    left_out = {os.fspath(path) for path in skip}
    # The entries still to yield of each directory from *root* down to the one being walked, last
    # name first.
    pending = [_listed(root, "")]
    while pending:
        if not pending[-1]:
            pending.pop()
            continue
        entry = pending[-1].pop()
        if os.fspath(entry.path) in left_out:
            continue
        yield entry
        if entry.kind == "directory":
            pending.append(_listed(entry.path, f"{entry.rel}/"))


def _listed(directory: Path, prefix: str) -> list[WalkEntry]:
    """The entries of *directory*, last name first, each with its path below the walk's root
    (*prefix* and its name); FileNotFoundError for one gone before its ``lstat``."""
    with os.scandir(directory) as it:
        found = sorted(it, key=lambda e: e.name, reverse=True)
    out = []
    for e in found:
        st = e.stat(follow_symlinks=False)
        out.append(WalkEntry(Path(e.path), prefix + e.name, entry_kind(st.st_mode), st))
    return out
