"""The content-addressed cache for images, weights and datasets.

Entries live at ``cache/<kind>/<key>/`` in the store and never change once published.
The first run that misses takes the entry's lock and fills a private staging directory
under ``.staging/cache/<kind>/<key>/``. It refuses anything in the fill output that is
not a plain file or directory, runs the caller's check against the files, records each file
in ``ENTRY.json`` with the sha256 the role that wrote it reported, syncs the tree's
directories and publishes it with one rename. Every
other run waits while the holder keeps its lock renewed and then reads the published
entry, so concurrent runs download once and never see a half-written entry; a run that has
a deadline, or is told to stop, gives up waiting then instead. A holder
that died is taken over, and the next holder moves whatever staging a dead fill left aside
and removes it. ``materialize`` gives each job its own verified copy on node-local disk.

The process that runs this, the runner, reads no file of an entry, a fill or a copy but the
cache's own ``ENTRY.json`` and stamp: data moves only through roles. A fill's
roles hash and sync each file they write (``hash_tree``) and report the digests
(``write_digests``), which ``populate`` takes through its *digests*; the cache scans names,
sizes, types and modification times alone (``scan_dir``). A copy is made and checked on the node
by a role of the cache's own program (``hls.store.cache_role``), which ``materialize`` is given to
run, and ``seed`` too, for a writable copy a role then writes into as its own.

File times. Each file's record holds its modification time (``FileRecord.mtime_ns``), so the
entry's digest pins it, and every copy gives its file that time back: an engine's JIT rebuilds
with ninja, and Python reuses a ``.pyc``, by comparing file times, so a copy with fresh times
(its sources written after their outputs) would build again at every load. A fill keeps the
times of the tree its producer left; a copy sets each file's recorded time once its bytes check;
the copy is refused when a time differs, by its role and again here. Regular files only:
directories are not recorded, and a copy's directories carry the time it made them, since
neither ninja nor the .pyc check reads a directory's time, and a node's periodic tmpfiles clean
removes old empty directories from a scratch under /tmp; links never reach an entry
(``walk_files``). An entry whose records carry no times is taken for absent
(``lookup``) and replaced by the next fill."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from hls.store.base import AlreadyExists, StoreError, check_segment
from hls.store.digest import CHUNK, Sha256Hex, check_sha256_hex, sha256_bytes
from hls.store.lock import GenerationLock, Holder, LockHandle, LockHeld, LockLost, Liveness
from hls.store.posix import STAGING, PosixStore
from hls.util import remove_tree, validation_errors, walk_tree

#: Name of the manifest file inside every entry directory.
ENTRY_FILE = "ENTRY.json"
#: Name of the marker a node-local copy gets once it matches its manifest (``_Stamp``).
VERIFIED_FILE = ".verified"
#: Names the cache writes into entries and copies itself, so a fill may not produce them.
RESERVED_NAMES = frozenset({ENTRY_FILE, VERIFIED_FILE})
#: Deepest directory nesting a fill or an entry may have. Weights, images and datasets are a few
#: levels deep. Every walk of a fill, an entry or a copy refuses a deeper tree (``walk_files``), so
#: the walks that recurse, such as ``os.walk`` and making a copy's directories, stay far from the
#: interpreter's recursion limit; every removal walks without recursing.
MAX_DEPTH = 64
#: The unit the node's scratch allocates a file's bytes and a directory's names in, at most: the
#: 4 KiB block of the filesystems a site's node-local scratch can be (xfs and ext4 by default,
#: and tmpfs, whose page it is). ``copy_disk_bytes`` counts a node-local copy's disk in it.
COPY_BLOCK_BYTES = 4096


class CacheCorrupt(StoreError):
    """An entry, a local copy or a fill's output does not match what the cache allows."""


class WaitEnded(StoreError):
    """``populate`` stopped waiting for another filler: its caller is stopping, or its
    deadline passed. Nothing was filled or published. The message names the filler that held
    the entry's lock at the last look."""


class FileRecord(BaseModel):
    """One file of an entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Path relative to the entry directory, "/"-separated.
    path: str
    #: Size in bytes.
    size: int = Field(ge=0)
    #: Hex SHA-256.
    sha256: Sha256Hex
    #: Modification time in ns of the file the fill staged, which keeps the time the tree's
    #: producer left (``hls.store.cache_role`` ``fill``). Every copy out of the store gives its
    #: file this time: a JIT's ninja build and Python's .pyc check decide by it whether an output
    #: is older than its source.
    mtime_ns: int


class EntryManifest(BaseModel):
    """What an entry holds, written last into the staging directory before publish."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Entry kind, e.g. "weights" or "images".
    kind: str
    #: Entry key, e.g. "Qwen+Qwen3.6-27B@<revision>".
    key: str
    #: Every file, sorted by path.
    files: list[FileRecord]
    #: Sum of file sizes.
    total_size: int = Field(ge=0)
    #: Publish time.
    created_at: float
    #: Where the content came from (repo, revision, recipe digest), for people reading it.
    source: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class CacheEntry:
    """A published entry."""

    kind: str
    key: str
    path: Path
    manifest: EntryManifest
    #: Hex SHA-256 of the ``ENTRY.json`` bytes that ``lookup`` read, which the manifest was parsed from.
    manifest_sha256: Sha256Hex
    #: Size of those bytes, which a node-local copy holds as its own ENTRY.json.
    manifest_size: int


@dataclass(frozen=True)
class _Untimed:
    """An entry whose file records carry no mtimes: ``lookup`` takes it for absent,
    and the next fill replaces it (``ContentCache._retire_untimed``)."""

    path: Path


def _lacks_every_mtime(error: ValidationError, loose: object) -> bool:
    """Whether *error*, raised parsing the entry manifest *loose* (its JSON, as plain values), says
    only that its file records lack their ``mtime_ns``, and every record does. The cache writes
    no manifest with some records timed, nor one with a time that is not an integer, so either is
    corrupt."""
    files = loose.get("files") if isinstance(loose, dict) else None
    if not isinstance(files, list) or not files:
        return False
    errors = error.errors()
    untimed = {e["loc"][1] for e in errors
               if e["type"] == "missing" and len(e["loc"]) == 3 and e["loc"][0] == "files" and e["loc"][2] == "mtime_ns"}
    return len(untimed) == len(errors) and untimed == set(range(len(files)))


def _loose_json(raw: bytes) -> object:
    """*raw* parsed as plain JSON values; None when it does not parse."""
    try:
        return json.loads(raw)
    except (ValueError, RecursionError):
        return None


@dataclass(frozen=True)
class Populated:
    """What ``populate`` returns."""

    #: The published entry.
    entry: CacheEntry
    #: True when this call published it; False when it was there already, or another filler
    #: published the same entry first.
    published: bool


#: Called with the fill's files as scanned, each with the digest its role reported, before
#: anything is published. Raising aborts the fill.
FillCheck = Callable[[list[FileRecord]], None]
#: How ``materialize`` makes or checks a node-local copy through a role: called with the
#: copy's directory, which for a copy does not exist yet, it runs the role, which copies the
#: entry there or checks the copy that is there, and returns the sha256 the role reported for
#: each file it read back, ``ENTRY.json`` among them. It raises when the role refused or failed.
CopyRole = Callable[[Path], Mapping[str, str]]


def entry_sha256(entry: CacheEntry) -> str:
    """Hex SHA-256 of the entry's ``ENTRY.json`` bytes as ``lookup`` read them: the digest a
    manifest pins the entry by. It is not read again, so it names the manifest this entry was
    parsed from even if the file changes afterwards."""
    return entry.manifest_sha256


def copy_disk_bytes(entry: CacheEntry) -> int:
    """The disk a node-local copy of *entry* takes at most (``materialize``): every file of its
    manifest and its ENTRY.json, each in whole ``COPY_BLOCK_BYTES`` blocks, its stamp in one block,
    and one block for the name of every file and directory of the copy, its own directory
    included, which bounds what its directories' entries take. The provision record states it of
    each entry (``hls.provision.model``), and the stages that copy data declare it as disk."""
    block = COPY_BLOCK_BYTES
    files = entry.manifest.files
    dirs = {PurePosixPath(f.path).parent for f in files} | {PurePosixPath(".")}
    for d in list(dirs):
        dirs.update(d.parents)
    data = sum(-(-f.size // block) for f in files) + -(-entry.manifest_size // block) + 1
    names = len(files) + 2 + len(dirs)
    return block * (data + names)


def subtree_disk_bytes(entry: CacheEntry, directory: str) -> int:
    """The disk a copy of the files of *entry* under its *directory* takes, counted as
    ``copy_disk_bytes`` counts a whole entry: every file in whole ``COPY_BLOCK_BYTES`` blocks, and
    one block for the name of every file and directory of the copy, its own directory included.
    The provision record states it of each tenant's dev split, which a dev view copies
    (``hls.provision.mounts.dev_view``)."""
    block = COPY_BLOCK_BYTES
    top = PurePosixPath(directory)
    files = [f for f in entry.manifest.files if top in PurePosixPath(f.path).parents]
    dirs = {top}
    for f in files:
        dirs.update(d for d in PurePosixPath(f.path).parents if top in d.parents)
    return block * (sum(-(-f.size // block) for f in files) + len(files) + len(dirs))


def copies_root(dest_root: Path, job_id: str) -> Path:
    """The directory ``materialize`` puts every node-local copy of job *job_id* under:
    ``<dest_root>/<job_id>``, holding ``<kind>/<key>`` for each entry."""
    return Path(dest_root) / check_segment(job_id)


class WalkedFile(NamedTuple):
    """A regular file ``walk_files`` found, as its directory entry describes it."""

    #: Path relative to the walked root, "/"-separated.
    path: str
    #: Size in bytes.
    size: int
    #: Modification time in ns.
    mtime_ns: int


def walk_files(root: Path, *, skip: Collection[str] = ()) -> list[WalkedFile]:
    """Every file under *root* with its size and modification time, sorted by path, from the
    directory entries alone: no file is opened. Names in *skip* are passed over at the top level
    only, where the cache keeps its own markers, and only if they are plain files.

    Raise CacheCorrupt for a symlink, for anything that is neither a regular file nor a
    directory, for a file with other hard links, for a reserved name anywhere else in the
    tree and for a tree nested deeper than ``MAX_DEPTH``. A symlink would publish a pointer
    out of the store, a hard link would let a writer outside the store change the entry's
    bytes, and a reserved name would be mistaken for the cache's own. It walks with a list, not
    by recursing (``walk_tree``), so a tree of any depth is refused rather than overflowing: a
    directory too deep is refused before it is listed."""
    out: list[WalkedFile] = []
    for e in walk_tree(root):
        name, top = e.path.name, "/" not in e.rel
        if top and name in skip:
            _require_plain_file(root, e.rel, e.stat)
            continue
        if name in RESERVED_NAMES:
            raise CacheCorrupt(f"{root}: {e.rel} uses a reserved name")
        if e.kind == "link":
            raise CacheCorrupt(f"{root}: {e.rel} is a symlink")
        if e.kind == "directory":
            if e.rel.count("/") + 1 > MAX_DEPTH:
                raise CacheCorrupt(f"{root}: {e.rel} is nested deeper than {MAX_DEPTH} levels")
        elif e.kind == "file":
            _require_plain_file(root, e.rel, e.stat)
            out.append(WalkedFile(e.rel, e.stat.st_size, e.stat.st_mtime_ns))
        else:
            raise CacheCorrupt(f"{root}: {e.rel} is not a regular file or directory")
    return sorted(out)


def scan_dir(root: Path, *, digests: Mapping[str, str], skip: Collection[str] = ()) -> list[FileRecord]:
    """Every file under *root* (``walk_files``), sorted by path, each recorded with its size and
    modification time and the sha256 the role that wrote it reported in *digests*, by path. No
    file is opened: the runner reads no file of a fill or a copy.

    Raise CacheCorrupt as ``walk_files`` does, and for a file whose digest was not reported, a
    reported digest that is not a bare hex sha256, or one of a path that is not a file of the
    tree."""
    for path, digest in digests.items():
        try:
            check_sha256_hex(digest)
        except ValueError as e:
            raise CacheCorrupt(f"{root}: the digest reported for {path}: {e}") from e
    files = walk_files(root, skip=skip)
    unreported = [f.path for f in files if f.path not in digests]
    if unreported:
        raise CacheCorrupt(f"{root}: no digest was reported for {', '.join(unreported)}")
    stray = sorted(set(digests) - {f.path for f in files})
    if stray:
        raise CacheCorrupt(f"{root}: a role reported a digest for {', '.join(stray)}, which is not a file of "
                           "the tree")
    return [FileRecord(path=f.path, size=f.size, sha256=digests[f.path], mtime_ns=f.mtime_ns) for f in files]


def mtime_problem(path: str, found: int, recorded: int) -> str:
    """What is wrong with the copy of file *path*, which has the mtime *found* where its record
    holds *recorded*, both in ns."""
    return f"{path} has mtime {found} ns, not the recorded {recorded}"


def hash_tree(root: Path) -> dict[str, str]:
    """For a role that wrote the tree *root*: the sha256 of every regular file under it, by
    "/"-separated path, of the bytes on disk after the write, each file synced once it is hashed.
    What it reports is what landed, so a role that reports it hashes no byte it only meant to
    write. A link is never followed and anything that is not a regular file is never opened (a
    FIFO would block, ``walk_tree``). Neither is refused here: the cache's scan of the same tree
    (``walk_files``) refuses it, and the tree's depth, as a refused fill, which is not retried,
    where the role failing here would be a fault that is. A directory that cannot be listed
    raises."""
    return {e.rel: hash_file(e.path)[0] for e in walk_tree(root) if e.kind == "file"}


def hash_file(path: Path) -> tuple[str, int]:
    """For a role that wrote the file *path*: the sha256 and size of the bytes on disk, read back
    through, and the file synced to disk. A link is not followed."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        h = hashlib.sha256()
        size = 0
        while chunk := os.read(fd, CHUNK):
            h.update(chunk)
            size += len(chunk)
        os.fsync(fd)
        return h.hexdigest(), size
    finally:
        os.close(fd)


def write_file(path: Path, data: bytes) -> str:
    """Write *data* to the new file *path*, sync it and return its sha256: how the process that
    runs a fill writes a small file of its own making into the staging directory and reports
    it, as a role reports what it wrote, without reading it back. Raise FileExistsError when
    something is at *path* already."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(fd, view):]
        os.fsync(fd)
    finally:
        os.close(fd)
    return sha256_bytes(data)


def write_digests(path: Path, digests: Mapping[str, str]) -> None:
    """Write a role's report *digests*, each file's sha256 by its path, to *path* as JSON, for
    the stage that started the role to read (``read_digests``)."""
    path.write_text(json.dumps(dict(sorted(digests.items())), indent=1))


def read_digests(path: Path) -> dict[str, str] | None:
    """The report a role wrote to *path* with ``write_digests``; None when there is none, or
    when it is not a JSON object of bare hex sha256 strings by path."""
    try:
        got = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(got, dict) or not all(isinstance(d, str) for d in got.values()):
        return None
    try:
        return {p: check_sha256_hex(d) for p, d in got.items()}
    except ValueError:
        return None


def _require_plain_file(root: Path, path: str, st: os.stat_result) -> None:
    """Raise CacheCorrupt unless *st* is a regular file with no other hard link."""
    if not stat.S_ISREG(st.st_mode):
        raise CacheCorrupt(f"{root}: {path} is not a regular file")
    if st.st_nlink != 1:
        raise CacheCorrupt(f"{root}: {path} has {st.st_nlink} hard links")


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_dirs(root: Path) -> None:
    """fsync every directory under *root*, deepest first, then *root* itself, so the names are
    on disk before a rename makes the tree visible. Every file in it was synced by whoever wrote
    it: a role (``hash_tree``) or the fill's own process (``write_file``), so none is opened."""
    for d, dirs, _ in os.walk(root, topdown=False):
        for name in dirs:
            _sync(Path(d) / name)
    _sync(root)


def _diff(expected: list[FileRecord], actual: list[FileRecord]) -> list[str]:
    """Every way the files *actual* differ from the records *expected*: by name, by size or
    digest, and by modification time."""
    want = {f.path: f for f in expected}
    have = {f.path: f for f in actual}
    problems = [f"missing {p}" for p in sorted(want.keys() - have.keys())]
    problems += [f"unexpected {p}" for p in sorted(have.keys() - want.keys())]
    for p in sorted(want.keys() & have.keys()):
        if (want[p].size, want[p].sha256) != (have[p].size, have[p].sha256):
            problems.append(f"changed {p}")
        elif want[p].mtime_ns != have[p].mtime_ns:
            problems.append(mtime_problem(p, have[p].mtime_ns, want[p].mtime_ns))
    return problems


class _Stamp(BaseModel):
    """What a node-local copy's ``VERIFIED_FILE`` says: the job that verified it, against which
    entry, and how its tree looked then."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Digest of the ENTRY.json the copy was verified against.
    entry_sha256: Sha256Hex
    #: The job whose ``materialize`` verified it.
    job_id: str
    #: The copy's ``_fingerprint`` right after it was verified.
    fingerprint: Sha256Hex


def _fingerprint(root: Path) -> str:
    """Digest of every path under *root* but the marker, in sorted order, each with its type,
    size, inode, mtime and ctime as lstat gives them. A write, a new, removed or replaced file
    and a link change it, and so does a touch; setting a file's mtime back does not hide a write,
    since nothing sets ctime. Costs one lstat per path, where a check of the bytes reads them all."""
    h = hashlib.sha256()
    for directory, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for name in sorted(dirnames + filenames):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if rel == VERIFIED_FILE:
                continue
            st = os.lstat(path)
            h.update(f"{rel}\0{stat.S_IFMT(st.st_mode)}\0{st.st_size}\0{st.st_ino}\0{st.st_mtime_ns}\0"
                     f"{st.st_ctime_ns}\n".encode())
    return h.hexdigest()


def _read_stamp(copy: Path) -> _Stamp | None:
    """The stamp of *copy*; None when it has none, or one that does not parse."""
    marker = copy / VERIFIED_FILE
    try:
        return _Stamp.model_validate_json(marker.read_bytes()) if marker.is_file() and not marker.is_symlink() else None
    except (OSError, ValidationError):
        return None


def _write_stamp(copy: Path, entry: CacheEntry, job_id: str) -> None:
    """Stamp *copy*, just verified against *entry* by job *job_id*, with its fingerprint now; a
    reader sees the old stamp or the new one."""
    stamp = _Stamp(entry_sha256=entry.manifest_sha256, job_id=job_id, fingerprint=_fingerprint(copy))
    tmp = copy / f"{VERIFIED_FILE}.tmp"
    tmp.write_text(stamp.model_dump_json())
    os.replace(tmp, copy / VERIFIED_FILE)


def _copy_problems(entry: CacheEntry, copy: Path, reported: Mapping[str, str], *,
                   skip: Collection[str]) -> list[str]:
    """Every way the node-local *copy* differs from *entry*, as the role that made or checked it
    reported each file's sha256 in *reported*: each file's name, size, digest and modification
    time against the manifest, the names in *skip* passed over at the top, and its ENTRY.json,
    which must be a plain file the role reported with the digest of the bytes *entry* was read
    from. Raise CacheCorrupt as ``scan_dir`` does."""
    digests = dict(reported)
    manifest_sha256 = digests.pop(ENTRY_FILE, None)
    problems = _diff(entry.manifest.files, scan_dir(copy, skip=skip, digests=digests))
    try:
        plain = stat.S_ISREG(os.lstat(copy / ENTRY_FILE).st_mode)
    except FileNotFoundError:
        plain = False
    if not plain or manifest_sha256 != entry.manifest_sha256:
        problems.append(f"its {ENTRY_FILE} is not the entry's")
    return problems


class ContentCache:
    """The cache area of a POSIX store."""

    def __init__(
        self,
        store: PosixStore,
        *,
        lock_ttl_sec: float = 120.0,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        #: Store the cache lives in.
        self.store = store
        #: TTL of entry locks; the keep-alive renews at a quarter of it.
        self.lock_ttl_sec = lock_ttl_sec
        #: Time source, for entry times and the locks.
        self.clock = clock
        #: Sleep used between looks while another filler holds the lock.
        self.sleep = sleep

    def _entry_key(self, kind: str, key: str) -> str:
        return f"cache/{check_segment(kind)}/{check_segment(key)}"

    def _staging_root(self, kind: str, key: str) -> Path:
        """Where fills of one entry stage. Only the lock holder writes or sweeps here."""
        return self.store.root / STAGING / "cache" / check_segment(kind) / check_segment(key)

    def lookup(self, kind: str, key: str) -> CacheEntry | None:
        """The published entry, or None: when nothing is at the entry's place, and when the entry
        there has file records without their files' mtimes, which no copy
        could give back, so no stage may use it and the next ``populate`` replaces it. Raise
        CacheCorrupt for anything else at the entry's place that is not a plain directory with a
        valid manifest of that kind and key: publishing renames a complete directory, so nothing
        legitimate looks like that. The entry carries the digest of the manifest bytes read
        here."""
        found = self._find(kind, key)
        return found if isinstance(found, CacheEntry) else None

    def _find(self, kind: str, key: str) -> CacheEntry | _Untimed | None:
        """What is at the place of entry *kind*/*key*: the published entry, an entry whose file
        records carry no mtimes (``_Untimed``), or None when nothing is. Raise CacheCorrupt as
        ``lookup`` says."""
        return self._read_place(self.store.path(self._entry_key(kind, key)), kind, key)

    def _read_place(self, path: Path, kind: str, key: str) -> CacheEntry | _Untimed | None:
        """What the directory *path* holds as entry *kind*/*key*: the entry, an entry of it whose
        file records carry no mtimes (``_Untimed``), or None when nothing is at *path*. Raise
        CacheCorrupt for anything else, an untimed manifest of another entry among it."""
        if not os.path.lexists(path):
            return None
        if path.is_symlink():
            raise CacheCorrupt(f"{path} is a symlink")
        if not path.is_dir():
            raise CacheCorrupt(f"{path} is not a directory")
        mf = path / ENTRY_FILE
        if mf.is_symlink():
            raise CacheCorrupt(f"{mf} is a symlink")
        if not mf.is_file():
            raise CacheCorrupt(f"{path} is published with no ENTRY.json")
        raw = mf.read_bytes()
        try:
            manifest = EntryManifest.model_validate_json(raw)
        except ValidationError as e:
            loose = _loose_json(raw)
            if not _lacks_every_mtime(e, loose):
                raise CacheCorrupt(f"{mf} is not a valid entry manifest: {validation_errors(e)}") from None
            assert isinstance(loose, dict)
            if (loose.get("kind"), loose.get("key")) != (kind, key):
                raise CacheCorrupt(f"{path} holds the entry {loose.get('kind')}/{loose.get('key')}") from None
            return _Untimed(path)
        if (manifest.kind, manifest.key) != (kind, key):
            raise CacheCorrupt(f"{path} holds the entry {manifest.kind}/{manifest.key}")
        return CacheEntry(kind, key, path, manifest, sha256_bytes(raw), len(raw))

    def _retire_untimed(self, kind: str, key: str, root: Path, handle: LockHandle) -> None:
        """Under the lock *handle* holds, just confirmed, with the new entry staged and checked:
        when the entry at the place of *kind*/*key* is one whose file records carry no mtimes,
        which the new entry replaces, move it aside into *root*, the entry's staging root, and
        remove it. It goes only now, so a fill may read it (an import whose only file on the
        cluster is the entry's own), and a failed fill leaves it. A removal cut short leaves it in
        *root* for the next holder's sweep.

        A holder that lost its lock between its look and its move may have moved what a successor
        published since: what was moved is read again, and anything but an untimed entry of
        *kind*/*key* is put back and the holder's loss reported (LockLost; CacheCorrupt when the
        lock still holds, which nothing legitimate brings about, or when it cannot be put back,
        which leaves it in *root*). Nothing at the place to move means another holder retired it:
        that holder took the lock over, which *handle* reports, unless the lock holds. What was
        moved having vanished before it is read back means a holder of the lock swept it: the
        loss is reported, or CacheCorrupt raised when the lock still holds, and nothing is
        published."""
        place = self.store.path(self._entry_key(kind, key))
        if not isinstance(self._find(kind, key), _Untimed):
            return
        aside = root / f".retired-{uuid.uuid4().hex}"
        try:
            os.rename(place, aside)
        except FileNotFoundError:
            handle.confirm()
            return
        try:
            moved: CacheEntry | _Untimed | None | CacheCorrupt = self._read_place(aside, kind, key)
        except CacheCorrupt as e:
            moved = e
        if isinstance(moved, _Untimed):
            remove_tree(aside, ignore_errors=True)
            return
        if moved is None:
            # Only a holder of the lock sweeps the staging root, so whatever was moved is gone
            # and this holder has lost the lock, which confirm() reports.
            handle.confirm()
            raise CacheCorrupt(f"what was moved aside from {place} to be retired vanished, under the lock")
        try:
            os.rename(aside, place)
        except OSError as e:
            raise CacheCorrupt(f"{place} changed while it was retired, and what was there is left at {aside}: "
                               f"{e}") from e
        handle.confirm()
        raise CacheCorrupt(f"{place} changed while it was retired, under the lock; it was put back")

    def populate(
        self,
        kind: str,
        key: str,
        fill: Callable[[Path], None],
        *,
        holder: Holder,
        digests: Callable[[], Mapping[str, str]],
        alive: Liveness | None = None,
        check: FillCheck | None = None,
        poll_sec: float = 5.0,
        source: dict[str, str] | None = None,
        deadline: float | None = None,
        stopping: Callable[[], bool] | None = None,
        on_wait: Callable[[Holder], None] | None = None,
    ) -> Populated:
        """Return the entry, filling it with *fill* (which writes into the directory it is
        given) if nobody has published it yet, and whether this call published it.
        *digests*, called after *fill*, gives the sha256 of every file the fill wrote, by path,
        as the role that wrote each reported it, having hashed what landed and synced it
        (``hash_tree``), or as the fill's own process wrote it (``write_file``). The entry records
        them, and the cache opens none of those files: it scans names, sizes, types and
        modification times only (``scan_dir``), recording each file's time as the fill left it,
        and a file with no reported digest is refused. An entry at the place whose records carry
        no times, which ``lookup`` takes for absent, stays there while the fill runs, which may
        read it, and is moved aside and removed under the lock just before the new entry is
        published.

        While another filler holds the entry's lock and keeps it renewed, this waits, looking
        for the published entry every *poll_sec*. It waits until the time *deadline* (by the
        cache's clock; never sleeping past it) or until *stopping* says to stop, checked
        before each pause, and then raises WaitEnded; with neither, it waits for as long as
        the holder renews. *on_wait* is called with each holder this call starts waiting for.
        A holder that released the lock, stopped renewing for a TTL, or that *alive* reports
        dead is taken over. *check*, when given, sees the fill's output before anything is
        published."""
        hit = self.lookup(kind, key)
        if hit is not None:
            return Populated(hit, published=False)
        what = f"{check_segment(kind)}/{check_segment(key)}"
        lock = GenerationLock(self.store, f"cache-locks/{what}", ttl_sec=self.lock_ttl_sec, clock=self.clock)
        waited_for: Holder | None = None
        while True:
            try:
                handle = lock.try_acquire(holder, alive=alive)
                break
            except LockHeld as held:
                hit = self.lookup(kind, key)
                if hit is not None:
                    return Populated(hit, published=False)
                other = held.record.holder
                if other != waited_for and on_wait is not None:
                    on_wait(other)
                waited_for = other
                self._pause(what, other, poll_sec, deadline, stopping)
        published = False
        try:
            with handle, handle.keep_alive(interval_sec=self.lock_ttl_sec / 4):
                if self.lookup(kind, key) is None:
                    published = self._fill_and_publish(kind, key, fill, digests, handle=handle, check=check,
                                                       source=source)
        except LockLost:
            # Lost after the publish, or beaten by an identical fill: the entry stands.
            hit = self.lookup(kind, key)
            if hit is None:
                raise
            return Populated(hit, published=published)
        found = self.lookup(kind, key)
        if found is None:
            raise CacheCorrupt(f"{kind}/{key} was published but is not there")
        return Populated(found, published=published)

    def _pause(self, what: str, other: Holder, poll_sec: float, deadline: float | None,
               stopping: Callable[[], bool] | None) -> None:
        """Sleep before the next look at entry *what*, which *other* is filling: *poll_sec*,
        or less where *deadline* comes first. Raise WaitEnded instead once *stopping* says to
        stop or the deadline has passed."""
        by = f"{other.purpose} on {other.host} pid {other.pid}"
        if stopping is not None and stopping():
            raise WaitEnded(f"{what}: stopped while {by} fills it")
        pause = poll_sec
        if deadline is not None:
            left = deadline - self.clock()
            if left <= 0:
                raise WaitEnded(f"{what}: still filled by {by} at the deadline")
            pause = min(pause, left)
        self.sleep(pause)

    def _sweep(self, root: Path) -> None:
        """Remove what dead fills left under *root*. Each leftover is renamed aside first: a
        holder that lost its lock may still rename its staging tree into place, and a tree
        being deleted in place would then be emptied after it became the published entry.
        Once renamed, the name that holder renames from is gone, so its rename is refused."""
        if not root.exists():
            return
        for leftover in sorted(root.iterdir()):
            aside = root / f".swept-{uuid.uuid4().hex}"
            try:
                os.rename(leftover, aside)
            except FileNotFoundError:
                continue
            remove_tree(aside, ignore_errors=True)

    def _fill_and_publish(self, kind: str, key: str, fill: Callable[[Path], None],
                          digests: Callable[[], Mapping[str, str]], *, handle: LockHandle,
                          check: FillCheck | None, source: dict[str, str] | None) -> bool:
        """Under the held lock: sweep what dead fills left, fill a fresh staging directory, check
        it, write its manifest, sync its directories, retire an entry at the place whose records
        carry no times (``_retire_untimed``) and publish the new one. Return False when another
        filler had published the entry first."""
        root = self._staging_root(kind, key)
        self._sweep(root)
        staging = root / uuid.uuid4().hex
        staging.mkdir(parents=True)
        entry_key = self._entry_key(kind, key)
        try:
            fill(staging)
            files = scan_dir(staging, digests=digests())
            if check is not None:
                check(files)
            manifest = EntryManifest(kind=kind, key=key, files=files, total_size=sum(f.size for f in files),
                                     created_at=self.clock(), source=source or {})
            write_file(staging / ENTRY_FILE, manifest.model_dump_json(indent=2).encode())
            _sync_dirs(staging)
            # A fill whose lock was lost must not publish; confirm() rereads the lock.
            handle.confirm()
            self._retire_untimed(kind, key, root, handle)
            dest = self.store.path(entry_key)
            # The directories publish_dir creates need their names made durable as well.
            created = [d for d in (dest.parent, dest.parent.parent) if not d.exists()]
            try:
                self.store.publish_dir(staging, entry_key)
            except AlreadyExists:
                # Another filler published the same content first; ours is a duplicate.
                remove_tree(staging, ignore_errors=True)
                return False
            _sync(dest.parent)
            for d in created:
                _sync(d.parent)
            return True
        except BaseException as error:
            remove_tree(staging, ignore_errors=True)
            if isinstance(error, Exception) and not isinstance(error, LockLost):
                # A holder that lost its lock may have had its staging swept from under it, and
                # what it then fails with says nothing about the fill. Report the loss.
                try:
                    handle.confirm()
                except LockLost as lost:
                    raise lost from error
                except Exception:
                    # The store could not answer; the error the fill hit is still the one to report.
                    pass
            raise

    def materialize(self, entry: CacheEntry, dest_root: Path, job_id: str, *, copy: CopyRole,
                    verify: CopyRole) -> Path:
        """A verified copy of *entry* at ``<dest_root>/<job_id>/<kind>/<key>``: its files are
        the manifest's, and its ENTRY.json is the one *entry* was read from. Once verified, the
        copy is stamped with *job_id* and a stat fingerprint of its tree (``_fingerprint``).

        Every byte is read by a role, never here. *copy* runs the role that copies the entry
        into a fresh directory beside the copy's place and checks each file it wrote against the
        manifest (``hls.store.cache_role``), giving each file its recorded time; the copy is kept,
        stamped and renamed into place only when the names, sizes, types and times found here,
        with the digests the role reported, are the manifest's (``_copy_problems``), and is
        removed otherwise.

        A copy stamped for this entry is reused. It is trusted without reading a byte when this
        job stamped it and its fingerprint still matches. Otherwise, a file whose change time
        moved or another job's stamp, *verify* runs the role that hashes every file of the copy
        against the manifest and holds its time against the record: a copy that still matches is
        stamped anew and reused, and one that changed since, a touched file among them, is
        refused (by the role, or CacheCorrupt here), since something wrote into it.
        Anything else at that path is replaced by a fresh copy. Whatever *copy* or *verify*
        raises, such as the role's refusal, is raised as it is.

        The job's copies lie under ``copies_root``, a directory made 0700, as the role
        directories are, so no other user of the node can read them whatever the copy role's modes.

        One process per job and node may call this, and the runner is that process: two
        callers sharing a job id could replace each other's copy while it is in use."""
        root = copies_root(dest_root, job_id)
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir(mode=0o700, exist_ok=True)
        root.chmod(0o700)
        dest = root / entry.kind / entry.key
        stamp = _read_stamp(dest)
        if stamp is not None and stamp.entry_sha256 == entry.manifest_sha256:
            if stamp.job_id == job_id and stamp.fingerprint == _fingerprint(dest):
                return dest
            problems = _copy_problems(entry, dest, verify(dest), skip={ENTRY_FILE, VERIFIED_FILE})
            if problems:
                raise CacheCorrupt(f"copy of {entry.kind}/{entry.key} changed since it was verified: "
                                   + "; ".join(problems))
            _write_stamp(dest, entry, job_id)
            return dest
        remove_tree(dest)
        self._checked_copy(entry, dest, copy, stamp=job_id)
        return dest

    def seed(self, entry: CacheEntry, dest: Path, *, copy: CopyRole) -> None:
        """A writable copy of *entry* made at *dest*, where nothing may be, for a role to write into
        as its own: when it is made its files are the manifest's and its ENTRY.json is the one
        *entry* was read from, and nothing vouches for it after. *copy* runs the role that copies
        the entry into a fresh directory beside *dest* and checks each file it wrote, as for
        ``materialize``; the copy is renamed into place only when the names, sizes, types and times
        found here, with the digests the role reported, are the manifest's, and is removed otherwise
        (CacheCorrupt). It is never stamped, so no stage reuses it as a copy of the entry. Raise
        FileExistsError when something is at *dest*, and whatever *copy* raises as it is."""
        if os.path.lexists(dest):
            raise FileExistsError(f"{dest} exists; a seed is made only where nothing is")
        self._checked_copy(entry, dest, copy, stamp=None)

    def _checked_copy(self, entry: CacheEntry, dest: Path, copy: CopyRole, *, stamp: str | None) -> None:
        """Make *dest* a copy of *entry* through the role *copy* runs in a fresh sibling directory,
        renamed into place once the role's report and the tree found here are the manifest's, and
        stamped for the job *stamp* first when one is given. Raise CacheCorrupt for a copy that is
        not the manifest's, which is removed."""
        tmp = dest.parent / f".tmp-{entry.key}-{uuid.uuid4().hex}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            problems = _copy_problems(entry, tmp, copy(tmp), skip={ENTRY_FILE})
            if problems:
                raise CacheCorrupt(f"copy of {entry.kind}/{entry.key}: " + "; ".join(problems))
            if stamp is not None:
                _write_stamp(tmp, entry, stamp)
            os.rename(tmp, dest)
        finally:
            remove_tree(tmp, ignore_errors=True)
