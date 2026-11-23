"""The commit point of an agent build: the ``main`` branch of the sandbox's workspace as a git
bundle, the submission its package builds from.

Once the sandbox has stopped, the agent_build stage runs this program in a harness container on
the sandbox image. The container sees the workspace read-only and the directory the bundle goes
to, nothing else of the host, and no network. The repository it reads was written by the
contestant, and git runs whatever a repository's configuration names (hooks, filters, an ssh
command, a promisor remote to fetch missing objects from), so git never opens that repository:
- the program reads ``main`` itself, from the loose ref or ``packed-refs``, following symbolic
  refs inside ``refs/``;
- it makes a fresh bare repository whose configuration git writes, points that repository's
  ``objects/info/alternates`` at the workspace's object store, sets ``refs/heads/main`` there and
  bundles it. git reads only object files from the workspace: no configuration, hook, ref
  database or index of the contestant's, and no commit-graph, multi-pack-index, bitmap or
  reverse index of its object stores (``GIT_FLAGS``). git's own settings are switched off
  (``CLONE_ENV``) and HOME is fresh;
- git trusts the object files it reads: a pack's index or a loose object's file can name one
  object at another's bytes, and a pack's entry can hold a deflate stream or a delta git cannot
  read, so the bundle git makes may not read back. The program writes the bundle once and reads
  nothing back; the package's checkout, the first to read it, proves every byte it gives git has
  the digest this program reported, so git's verdict on the bundle there is the submission's
  (``hls.package.checkout.receive``);
- a repository that is a link, a gitfile, a linked worktree's directory, keeps refs in reftable
  or uses SHA-256, whose objects, refs or alternates lead out of the workspace, whose links
  loop, or that holds anything but directories, regular files and links among its objects and
  refs, is refused as unreadable. Alternates inside the workspace are followed, as git follows
  them, and their object stores are held to the same rules.
Only ``main`` is read, since only the latest commit on main is evaluated.

Every file of the contestant's this program reads itself is read a bounded line at a time
(``LINE_MAX_BYTES``), without waiting on a pipe, from files of bounded size
(``LISTING_MAX_BYTES``); and every reason or branch name it records is bounded and valid UTF-8.
Past such a bound the repository is the contestant's unreadable one. The walk of the object
stores and refs follows at most ``ALTERNATES_MAX`` stores and visits at most
``walk_max_entries()`` entries, room for every object git is given as a loose file; past either
the repository is over the limit. What the contestant wrote can neither exhaust the container's
memory or time, hang it nor make a record that cannot be read, any of which would make the
contestant's work our failure.

The stage runs ``commit_command(...)``: the image's ``/usr/bin/python3 -I`` on a read-only copy
of this file at ``COMMIT_TARGET``. The program uses the stdlib only, because importing ``hls``
would import the whole package.

    commit.py --workspace DIR --out FILE --max-bytes N --seconds S [--no-submission FILE]
              [--record FILE] [--outputs-out FILE] [--tmp DIR]

``--max-bytes`` is the build's writable disk cap, the sandbox's (``hls.stages.agent_build``): the
most bytes the repository's objects may inflate to together and main's bundle may hold. The
bundle is packed with no delta search and no compression (``GIT_FLAGS``), so it holds about the
objects' own bytes, and its time is linear in them. ``--seconds`` is the time the program has,
counted from its start: what the stage's collection leaves the commit point before its last steps
must start, or for a snapshot what is left of the stage's time (``hls.stages.agent_build``). The
program's own reads of the repository spend it, and each git command runs only within what is
left of it (``_left``): a git still running then is ended, and the try is infra.

Exit status, which the stage reads together with the files:
- 0: FILE holds a bundle of ``main``;
- 5 (``NO_MAIN_EXIT``): the workspace has no repository, or no commit on main. A result, the
  contestant's; ``--no-submission`` is written;
- 4 (``UNREADABLE_EXIT``): the repository cannot be read because of what the contestant made:
  refused as above, or main's history has an object that is missing or corrupt, or main names
  something that is not a commit, or a file or symbolic link of main's tree, as git lists it once
  the bundle is written, names an object that is not a blob (``blob_sizes``), which git's plumbing
  can write and git's checkout cannot, or a file of it is refused to the program (EACCES or EPERM,
  ``REFUSED_ERRNOS``): the commit container reads as the workspace's owner, so only permissions
  the contestant set refuse it. Or the walk passed its bounds, or the repository's objects lie in
  more packs than git is given (``PACKS_MAX``), or the repository holds more
  objects than git is given to bundle (``OBJECTS_MAX``), or an object larger than git is given
  (``NONBLOB_MAX_BYTES`` for a commit, tree or tag, which git loads whole; ``DELTA_MAX_BYTES``
  for one stored as a delta; ``OBJECT_MAX_BYTES`` for any), or objects that inflate past
  ``--max-bytes`` together, or deltas whose rebuilding reads more than ``CHAIN_BYTES_MAX``
  together, or tree objects that hold more than ``TREE_BYTES_MAX`` together, whose entries git
  walks whether or not the blobs they name are there, each size read from the object's header
  alone; or main's tree, as git lists it once the bundle is written, lists more than
  ``TREE_MAX_ENTRIES`` paths or ``TREE_MAX_DIRECTORIES`` directories, each counted wherever it
  appears, or a path longer than ``TREE_MAX_PATH_BYTES`` (``listing_problem``), the listing read
  as git writes it and git ended at the first bound passed, so a tree whose subtrees repeat,
  which lists billions of paths from a few objects, is refused here before any later step lists
  it; or main's bundle holds more than ``--max-bytes``, the package's own limit on a source
  bundle (a bundle past the package's limit is no submission at the commit
  point: the program writes git's bundle through a byte budget, stops git one byte past the limit
  and keeps none of it) (kind ``over_limit``). The contestant's; ``--no-submission`` holds the reason,
  which names the errno or the bound;
- 74 (``IO_ERROR_EXIT``, EX_IOERR): an operating-system error the contestant cannot cause: EIO,
  ESTALE, ENOSPC or EDQUOT, or a refused path outside the workspace, such as an output. stderr
  and the record name the errno. Infra: the stage tries again on the same workspace;
- 1: anything else, such as exhausted memory, a git that hung, was killed or ran out of
  ``--seconds``, a workspace path that is not a directory, or an output the store holds at another
  size than the bytes written: infra.

``--record`` (``CommitRecord``) is written on exits 0, 4 and 5, and on 74 when it can be.
``--outputs-out`` reports the sha256 and size of the output written, the bundle or the
no-submission record, keyed by its name, as ``hls.store.manifest.report_written_outputs`` does
for a file directly in the attempt's outputs directory, so the stage publishes it by its writer's
digest: the bytes the program wrote, hashed as it wrote them (the bundle as git streamed it),
never read back. Once they are synced the file must hold that many bytes on
the store, or the try is infra, the store's fault (``_landed``).
The periodic snapshots (the agent_build stage's ``snapshot_interval_sec``) run the same program into
``snapshots/``; ``prune_snapshots`` keeps the newest ``SNAPSHOT_KEEP``."""

from __future__ import annotations

import argparse
import contextlib
import math
import errno
import hashlib
import json
import os
import re
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import zlib
from collections.abc import Callable, Iterable, Iterator
from typing import BinaryIO
from dataclasses import asdict, dataclass
from pathlib import Path

#: The variables that switch every git configuration of the host off; the same as
#: ``hls.package.checkout.CLONE_ENV``, which a test pins.
CLONE_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}
#: Settings on every git command line, for the fresh repository: no transport but local files,
#: no hooks, no automatic gc, and packing with no delta search (``pack.window=0``) and no
#: compression (``pack.compression=0``): what git does with each object is then read it, inflated,
#: and write it, so the bundle's time is linear in the objects' bytes whatever they hold, and the
#: bundle holds about those bytes. Object ids and content are unchanged. A delta the contestant's
#: own pack holds is reused as it is, never computed by us; with no window, git's settings of the
#: delta search (its threads, its window's memory, its cache) have nothing to bound. git reads no
#: commit-graph, multi-pack-index, pack bitmap or pack reverse index of the object stores it is
#: given: the contestant writes those, and git trusts them for what the objects hold (a
#: commit-graph names each commit's tree, which git's walk takes from it without reading the
#: commit), where the objects themselves and each pack's index, which the program reads too, are
#: enough. A pack's index can still name one object at another's bytes; the package's checkout
#: reads every bundle as git reads it and refuses one that does not read back as the submission's
#: (``hls.package.checkout.receive``).
GIT_FLAGS = ("-c", "protocol.allow=never", "-c", "protocol.file.allow=always", "-c", "core.hooksPath=/dev/null",
             "-c", "gc.auto=0", "-c", "pack.window=0", "-c", "pack.compression=0", "-c", "core.commitGraph=false",
             "-c", "core.multiPackIndex=false", "-c", "pack.useBitmaps=false", "-c", "pack.readReverseIndex=false")
#: The branch a submission is read from.
BRANCH = "main"
#: Seconds git's listing of main's tree may take, here and at the package's checkout
#: (``listing_problem``): twice its slowest at its bounds at the commit point (git 2.43.0 on one
#: AMD EPYC 9534 core): 22.54 s for 200,000 paths of up to 1,008 bytes in 199,020
#: directories, the most a listing within the bounds writes, each next tree git reads in the next
#: of ``PACKS_MAX`` packs, so git searches every pack for each, beside 5,000,000 other files in
#: the pack directory, which each git command reads. Here the listing gets the lesser
#: of this and the time left (``_left``): a listing that hangs ends its try as infra while the
#: collection may still have time for another, rather than taking all of it.
TREE_LIST_SEC = math.ceil(2 * 22.54)
#: How git's check of the objects main's tree names (``blob_sizes``) asks for each one's kind and
#: size: ``git cat-file`` fed their ids, a line each.
KIND_CHECK_FORMAT = "--batch-check=%(objecttype) %(objectsize)"
#: The longest line git's check writes for an id: the id and `` missing`` for one it cannot find.
KIND_LINE_MAX_BYTES = 64
#: Seconds git's check of the objects main's tree names may take, here and at the package's
#: checkout (``blob_sizes``): twice its slowest at the bounds at the commit point (git 2.43.0 on
#: one AMD EPYC 9534 core): 121.22 s for 200,000 files, each a blob 3,872 to 4,094 deltas
#: deep in one of 900 chains of reference deltas, as many as ``CHAIN_BYTES_MAX`` lets through (67.9
#: GB of chain bytes), git finding each one's kind at its chain's root; each next file's blob in
#: the next of ``PACKS_MAX`` packs, so git searches every pack for each; 5,000,000 other files in
#: the pack directory. Here the
#: check gets the lesser of this and the time left, as the listing does (``TREE_LIST_SEC``).
KIND_CHECK_SEC = math.ceil(2 * 121.22)
#: Exit status for a workspace with no repository or no commit on main.
NO_MAIN_EXIT = 5
#: Exit status for a repository that cannot be read because of its content.
UNREADABLE_EXIT = 4
#: Exit status for an operating-system error the contestant cannot cause (EX_IOERR).
IO_ERROR_EXIT = 74
#: Words in git's messages that name an errno, each with the errno's name.
ERRNO_WORDS = {"Input/output error": "EIO", "Stale file handle": "ESTALE", "No space left on device": "ENOSPC",
               "Disk quota exceeded": "EDQUOT", "Permission denied": "EACCES", "Operation not permitted": "EPERM"}
#: The errnos of a refused path: the contestant's when the path is in the workspace.
REFUSED_ERRNOS = ("EACCES", "EPERM")
#: Where the stage binds a read-only copy of this file in the commit container.
COMMIT_TARGET = "/opt/hls-agent/commit.py"
#: The directory beside a bundle's output, in the same directory, where git's bundle is written
#: until it is whole and then moved into place: a program killed outright leaves it there, under
#: one name its caller declares and clears (``hls.stages.agent_build``).
PARTIAL_DIR = ".tmp-partial"
#: Snapshots kept as evidence of a build in progress.
SNAPSHOT_KEEP = 2
#: The most bytes of git's messages read for a failure's reason, here and at the package's
#: checkout: their last ones (``git_messages``). git writes a few lines when it fails; a repository
#: can make it write more.
MESSAGE_BYTES = 4000
#: Words in git's message that mean the machine, not the repository, failed.
RESOURCE_ERRORS = ("No space left on device", "Disk quota exceeded", "Cannot allocate memory", "Out of memory",
                   "Resource temporarily unavailable")
#: The most bytes of one line read from a file of the repository that git reads as lines: a
#: loose ref, a line of ``packed-refs`` or of an alternates file. git writes none near it.
LINE_MAX_BYTES = 64 * 2**10
#: The most bytes of a file of the repository read line by line, as often as a lookup needs:
#: ``packed-refs`` or an alternates file.
LISTING_MAX_BYTES = 16 * 2**20
#: The most object stores followed through alternates, the workspace's own among them; each
#: one's alternates file is scanned. Past it the repository is over the limit.
ALTERNATES_MAX = 10
#: The most objects the bundle's git is given, loose and packed, across the workspace's object
#: store and every alternate one: ten revisions of a tree at the package's 200,000-path limit
#: are about 2,000,000.
OBJECTS_MAX = 5_000_000
#: The most packs git is given, over every object store the repository's objects lie in, its
#: alternates' among them: each a pack index with its pack beside it, the packs git opens
#: (``_packs``). git looks an object up in its packs most recently used first, so objects that
#: cycle through the packs in the order git reads them have git search every pack for each, and
#: each git command's own setup reads every pack's name: 70,000 empty packs beside one commit take
#: git's listing of main past its time (git 2.53). In a repository whose packs only git's
#: automatic gc writes, gc folds every pack into one once there are more than
#: ``gc.autoPackLimit``, 50. This is ten times that. Past it the repository is over the limit,
#: the contestant's.
PACKS_MAX = 500
#: The entries the walks of the object stores and refs visit beside the objects themselves: the
#: refs, and the directories and pack files of every store followed (256 fan-out directories
#: each). git writes far fewer; the walk's bound is ``walk_max_entries()``.
WALK_ROOM_ENTRIES = 1_000_000
#: The most bytes of a loose object, or of a delta in a pack, read to find its header: git
#: writes the header within its first few dozen.
HEADER_INPUT_BYTES = 4096
#: The most pack entries the size scan reads, across every pack: each entry is read once in its
#: pack's pass and at most once more while a delta's kind is found through its chain, since
#: every kind found is kept; so twice the most objects git is given.
SCAN_MAX_READS = 2 * OBJECTS_MAX
#: The largest commit, tree or tag git is given, which git loads whole. A tree at the package's
#: limits is at most 200,000 entries of a mode, a name of up to 255 bytes and a 20-byte id, about
#: 57 MB; a commit or tag is its message.
NONBLOB_MAX_BYTES = 64 * 2**20
#: The largest object of any kind git is given: the package's limit on a whole tree's bytes, past
#: which no checkout of it is accepted.
OBJECT_MAX_BYTES = 4 * 2**30
#: The largest object stored as a delta git is given, which git rebuilds whole when it cannot
#: copy the delta: git's own big-file threshold, above which it never stores a delta.
DELTA_MAX_BYTES = 512 * 2**20
#: Bytes read from git's output at a time.
CHUNK_BYTES = 2**20
#: The most bytes the repository's tree objects may hold together, over the whole object store.
#: git's bundle walks every entry of every tree reachable from main, and the package's scan reads
#: them again; a tree entry may name a blob that is not in the store, and such an entry is no
#: object, so it escapes ``object_count`` and the inflated bound (its blob is never there to
#: inflate) while git still reads it. One 64 MiB tree holds about 2.3 million entries (29 bytes
#: each name one byte long), and on the commit role's CPU git bundle walks them at 3.97 s per MiB
#: of tree, 115 s a million entries (an AMD EPYC 9534, git
#: 2.43.0): 128 MiB of trees take about 508 s, while
#: forty such trees (2.5 GiB of trees, about 92 million missing-blob entries), which pass
#: ``OBJECTS_MAX`` (80 objects) and the inflated bound (no blob to inflate), would take git nearly
#: three hours. The history
#: ``OBJECTS_MAX`` is sized for, ten loose revisions of a tree at the package's 200,000-path
#: limit, holds about 61 MiB of trees, so this is about twice that. Past it the repository is
#: over the limit, the contestant's.
TREE_BYTES_MAX = 128 * 2**20
#: The most paths main's tree may track as git lists them (``git ls-tree -r``): every file,
#: symlink and submodule, each counted wherever its directory appears, since a tree may name one
#: subtree many times and git lists, and the package's checkout writes, each place. A tree of ten
#: directories that each name the one below, nine deep, lists ten billion paths from eleven
#: objects.
TREE_MAX_ENTRIES = 200_000
#: The most directories main's tree may hold, each counted wherever it appears, an empty one (the
#: empty tree, which git's plumbing can name and no checkout creates) among them: git's listing
#: and checkout visit every one, so a tree whose subtrees repeat over the empty tree lists no path
#: and still keeps git walking past any time.
TREE_MAX_DIRECTORIES = 200_000
#: The longest a path of main's tree may be, in bytes, a directory's among them: what one record
#: of git's listing holds, and what its checkout builds for each place it visits.
TREE_MAX_PATH_BYTES = 1024
#: git's settings for a command that reads main's trees wherever they appear (its listing here,
#: and the package's listing and checkout): a cache of delta bases as large as the repository's
#: trees may be together (``TREE_BYTES_MAX``), so a tree stored as a delta is rebuilt from its base
#: once and not at each place it is read. git's default, 96 MiB, is under two of the largest trees
#: a repository may hold: with bases larger together than it, each read rebuilds a tree from its
#: chain.
TREE_CACHE = ("-c", f"core.deltaBaseCacheLimit={TREE_BYTES_MAX}")
#: The most bytes a record of git's listing (``git ls-tree -r -t -z``) holds before its path: a
#: mode of six octal digits, a kind of at most six letters (``commit``), a 40-digit object id, the
#: two spaces between them and the tab.
LISTING_META_MAX_BYTES = 6 + 6 + 40 + 2 + 1
#: The most bytes rebuilding every object the repository stores as a delta may read together, over
#: the whole object store, each delta counted with its chain: the object it rebuilds and its
#: delta's own bytes, on the same for its base, down to a whole object. git rebuilds a delta whose
#: base it does not bundle, and the package's scan rebuilds every delta it reads; a chain's every
#: delta is rebuilt from its root when the objects are larger than git's 96 MiB cache of bases, so
#: the work grows with the chains, not with what the objects inflate to. git 2.43.0's bundle
#: rebuilds deltas at up to 4.22 s per GiB of chain bytes on one AMD EPYC 9534 core (deltas of
#: one-byte copies), so the bound of 64 GiB takes it about 270 s. Past it the repository is over
#: the limit, the contestant's, which the brief states.
CHAIN_BYTES_MAX = 64 * 2**30
#: The most branch names a record holds.
BRANCHES_MAX = 1000
#: The most branch names a no-submission reason lists.
BRANCHES_NAMED = 10
#: The longest a reason may be, in characters.
REASON_MAX_CHARS = 2000
#: The most characters of a name the contestant chose that a reason quotes, so the reason's
#: own words survive.
NAME_SHOWN_CHARS = 200
#: The longest ref name a symbolic ref may lead to, in bytes.
REF_MAX_BYTES = 1024
#: The longest part of that name, in bytes: no file could have a longer name.
REF_PART_MAX_BYTES = 255
#: How deep git itself follows alternates of alternates, and symbolic refs.
_DEPTH = 5
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REF = re.compile(r"refs/[A-Za-z0-9._@+-]+(?:/[A-Za-z0-9._@+-]+)*")
_SNAPSHOT = re.compile(r"main-([0-9]{6})\.bundle")
#: A loose object's fan-out directory, and its file's name there.
_LOOSE_DIR = re.compile(r"[0-9a-f]{2}")
_LOOSE_NAME = re.compile(r"[0-9a-f]{38}")
#: Pack entry types of whole objects, as git numbers them, and of the two kinds of delta.
_PACK_KINDS = {1: "commit", 2: "tree", 3: "blob", 4: "tag"}
_OFS_DELTA, _REF_DELTA = 6, 7
#: The longest delta chain git writes (the ceiling of its pack.depth); a deeper one, or one that
#: loops, is not one git made.
_DELTA_DEPTH_MAX = 4095
#: How a pack scan keeps an entry's chain depth, at most ``_DELTA_DEPTH_MAX``, and where its chain
#: bytes start, in the one number it keeps for the entry (``_PackScan.kinds``).
_DEPTH_MASK, _CHAIN_SHIFT = 0xFFF, 15
#: The first bytes of a pack index of version 2 or later; a version 1 index starts with its
#: fan-out table.
_INDEX_MAGIC = b"\xfftOc"
#: An absolute path in one of git's messages.
_PATH = re.compile(r"(?:^|(?<=[\s'\"(]))/[^\s'\":]+")
#: What makes the workspace's repository unreadable to the program, the contestant's no
#: submission (``UNREADABLE_EXIT``), by a name for each: every refusal names one
#: (``_unreadable``), and the brief states them all (``hls.prompt``).
UNREADABLE = {
    "link": "the workspace's `.git` is a symbolic link",
    "gitfile": "the workspace's `.git` is a file that points at a repository elsewhere",
    "worktree": "the workspace's `.git` is a linked worktree's directory",
    "reftable": "the repository keeps its refs in reftable format",
    "sha256": "the repository names its objects with SHA-256",
    "outside": "the repository's objects, refs or alternates, or a link among them, lead outside the workspace",
    "loop": "symbolic links among the repository's objects or refs loop",
    "special": "the repository's objects or refs hold anything but directories, regular files and links",
    "unwritten": (f"a file of the repository holds what git never writes: a ref, `packed-refs` or alternates file "
                  f"of more than {LISTING_MAX_BYTES // 2**20} MiB or with a line of more than "
                  f"{LINE_MAX_BYTES // 2**10} KiB, an object or pack index git cannot read, or a delta chain "
                  f"deeper than {_DELTA_DEPTH_MAX:,}"),
    "main": (f"`main` is a symbolic ref to something that is not a ref, is reached through more than {_DEPTH} "
             "symbolic refs, or holds no object name"),
    "history": "`main`'s history has an object that is missing or corrupt, or `main` names something that is not "
               "a commit",
    "kind": "`main`'s tree names a file or symbolic link whose object is not a blob but a tree, a commit or a tag, "
            "which git's plumbing can write and no checkout can read",
    "refused": "a file of the repository refuses to be read, by the permissions you set",
}


def _text(text: str) -> str:
    """*text* as a record holds it: valid UTF-8, with any byte that was not as an escape, and at
    most ``REASON_MAX_CHARS`` long."""
    clean = text.encode("utf-8", "backslashreplace").decode("utf-8")
    return clean if len(clean) <= REASON_MAX_CHARS else clean[:REASON_MAX_CHARS - 3] + "..."


@dataclass(frozen=True)
class NoSubmission:
    """``no_submission.json``: why the build left nothing to package."""

    #: ``no_main`` (no repository or no commit on main), ``unreadable``, or ``over_limit`` (more
    #: objects, a larger one, more inflated bytes, more chain bytes or more tree bytes than git is
    #: given to bundle, more entries or object stores than the walk visits, or a bundle past
    #: ``--max-bytes``).
    kind: str
    #: One line for the package result and the build record.
    reason: str
    #: When it was found (POSIX time).
    at: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: bytes | str) -> NoSubmission:
        """Parse a record; raise ValueError for anything else."""
        try:
            raw = json.loads(data)
        except (ValueError, RecursionError) as e:
            raise ValueError(f"not JSON: {e}") from None
        if not isinstance(raw, dict) or set(raw) != {"kind", "reason", "at"}:
            raise ValueError("expected exactly the keys kind, reason and at")
        if raw["kind"] not in ("no_main", "unreadable", "over_limit") or not isinstance(raw["reason"], str) \
                or isinstance(raw["at"], bool) or not isinstance(raw["at"], (int, float)):
            raise ValueError("kind must be no_main, unreadable or over_limit, reason a string and at a number")
        if raw["reason"] != _text(raw["reason"]):
            raise ValueError("reason must be valid UTF-8 of at most REASON_MAX_CHARS characters")
        return cls(kind=raw["kind"], reason=raw["reason"], at=float(raw["at"]))


@dataclass(frozen=True)
class CommitRecord:
    """``--record``: what the commit point found."""

    #: ``ok``, ``no_main``, ``unreadable``, ``over_limit`` or ``io_error``.
    status: str
    #: The commit on main that was bundled; None without a bundle.
    commit: str | None
    #: Why there is no bundle; None with one.
    reason: str | None
    #: The repository's branches, sorted: at most ``BRANCHES_MAX`` of them, the first found;
    #: [] when they could not be read.
    branches: list[str]
    #: When the program finished (POSIX time).
    at: float


class _NoSubmission(Exception):
    def __init__(self, kind: str, reason: str, branches: list[str] | None = None) -> None:
        reason = _text(reason)
        super().__init__(reason)
        self.kind, self.reason, self.branches = kind, reason, branches or []


class _Infra(Exception):
    pass


class _IoError(Exception):
    """An operating-system error the contestant cannot cause (``IO_ERROR_EXIT``)."""

    def __init__(self, errno_name: str, what: str) -> None:
        super().__init__(_text(f"{errno_name}: {what}"))
        #: The errno, by name.
        self.errno_name = errno_name


def _unreadable(cause: str, reason: str) -> _NoSubmission:
    """The contestant's no submission for a repository the program cannot read, for *reason*,
    under *cause*, one the brief states (``UNREADABLE``)."""
    if cause not in UNREADABLE:
        raise ValueError(f"{cause!r} is not a cause the brief states (UNREADABLE)")
    return _NoSubmission("unreadable", reason)


def commit_command(*, workspace: str, out: str, max_bytes: int, seconds: int, no_submission: str | None = None,
                   record: str | None = None, outputs_out: str | None = None,
                   tmp: str | None = None) -> tuple[str, ...]:
    """The command the stage runs in the commit container, with container paths, holding the
    repository to *max_bytes*, the build's writable disk cap, with *seconds* for all of its work
    from its start (``--seconds``). *tmp* is where the program makes its fresh repository; the
    container's own temporary directory when None."""
    argv = ["/usr/bin/python3", "-I", COMMIT_TARGET, "--workspace", workspace, "--out", out, "--max-bytes",
            str(max_bytes), "--seconds", str(seconds)]
    if no_submission is not None:
        argv += ["--no-submission", no_submission]
    if record is not None:
        argv += ["--record", record]
    if outputs_out is not None:
        argv += ["--outputs-out", outputs_out]
    if tmp is not None:
        argv += ["--tmp", tmp]
    return tuple(argv)


def _left(deadline: float, what: str) -> float:
    """Seconds left before *deadline*, a time of the monotonic clock, for *what* to run in; raise
    ``_Infra`` naming *what* when none are."""
    left = deadline - time.monotonic()
    if left <= 0:
        raise _Infra(f"no time was left for {what}: the program's --seconds were spent")
    return left


def git_env(home: Path) -> dict[str, str]:
    """The whole environment git runs with: PATH from this process, a fresh HOME, and the
    variables that switch the host's configuration off."""
    return {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home), "LANG": "C.UTF-8", **CLONE_ENV}


def git_messages(path: Path) -> str:
    """The last ``MESSAGE_BYTES`` of the messages git wrote to the file *path*, read from there
    alone, as text with any byte that is not UTF-8 replaced."""
    with open(path, "rb") as f:
        f.seek(max(0, f.seek(0, os.SEEK_END) - MESSAGE_BYTES))
        return f.read(MESSAGE_BYTES).decode("utf-8", errors="replace")


def git_failure(returncode: int, stderr: str) -> str | None:
    """git's last message line when its failure is the repository's, or None when the machine
    failed it: killed by a signal, or out of disk or memory."""
    if returncode < 0 or any(word in stderr for word in RESOURCE_ERRORS):
        return None
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    return lines[-1] if lines else f"git exited with status {returncode}"


def errno_failure(stderr: str, workspace: Path) -> tuple[str, bool] | None:
    """The errno, by name, that git's *stderr* names, and whether it is the contestant's: a path
    in *workspace* that git was refused (``REFUSED_ERRNOS``). Every other errno, and a refused
    path outside the workspace, is the machine's, and the machine's wins over the contestant's.
    A refused line that names no path names a workspace object by its id, since git reads no
    other objects. None when *stderr* names no errno."""
    root = Path(os.path.realpath(workspace))
    refused = None
    for line in stderr.splitlines():
        name = next((n for words, n in ERRNO_WORDS.items() if words in line), None)
        if name is None:
            continue
        if name not in REFUSED_ERRNOS or any(not _lexically_inside(path, root) for path in _PATH.findall(line)):
            return name, False
        refused = refused or name
    return None if refused is None else (refused, True)


def _verb(args: list[str]) -> str:
    """The git command *args* run, past git's own options (``-C <dir>`` and ``-c <setting>``)."""
    at = 0
    while args[at] in ("-C", "-c"):
        at += 2
    return args[at]


def _git(args: list[str], env: dict[str, str], seconds: float) -> subprocess.CompletedProcess[str]:
    """Run ``git <GIT_FLAGS> <args>`` in its own session, killing the whole session when it
    outlives *seconds*, the time it is given (``_left``). Each command run this way writes a line
    or two; its messages can carry the contestant's bytes, so they are read as UTF-8 with any other
    byte replaced."""
    p = subprocess.Popen(["git", *GIT_FLAGS, *args], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace", start_new_session=True)
    try:
        out, err = p.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.communicate()
        raise _Infra(f"git {_verb(args)} took longer than the {seconds:.0f} s it was given") from None
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


def _objects_failure(args: list[str], returncode: int, stderr: str, workspace: Path) -> Exception:
    """What the failure of a git command of the fresh repository that read *workspace*'s objects
    is: unreadable when the objects or their permissions caused it, ``_IoError`` for an errno of
    the machine's, ``_Infra`` for anything else."""
    tail = stderr.strip()[-500:]
    found = errno_failure(stderr, workspace)
    if found is not None and not found[1]:
        return _IoError(found[0], f"git {_verb(args)} failed: {tail}")
    if found is not None:
        return _unreadable("refused", f"{found[0]}: git {_verb(args)} was refused a file of the workspace's "
                                      "repository, "
                           f"whose permissions the contestant set: {tail}")
    why = git_failure(returncode, stderr)
    if why is None:
        return _Infra(f"git {_verb(args)} failed with status {returncode}: {tail}")
    return _unreadable("history", f"main's history cannot be read from the workspace's repository: {why}")


def _on_objects(args: list[str], env: dict[str, str], workspace: Path, seconds: float) -> None:
    """Run a git command of the fresh repository that reads *workspace*'s objects, within
    *seconds*, and raise what its failure is (``_objects_failure``)."""
    p = _git(args, env, seconds)
    if p.returncode != 0:
        raise _objects_failure(args, p.returncode, p.stderr, workspace)


def _output(fd: int, *, deadline: float, what: str, seconds: float) -> Iterator[bytes]:
    """What the pipe *fd* carries, as it arrives, until it ends. Raise ``_Infra``, naming *what*
    and the *seconds* it had, once the monotonic clock passes *deadline*."""
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            raise _Infra(f"{what} took longer than the {seconds:.0f} s it was given")
        ready, _, _ = select.select([fd], [], [], left)
        if not ready:
            continue
        chunk = os.read(fd, CHUNK_BYTES)
        if not chunk:
            return
        yield chunk


def _copy_bounded(fd: int, dest: BinaryIO, *, limit: int, deadline: float, what: str, seconds: float,
                  streamed: hashlib._Hash | None = None) -> int:
    """Copy what the pipe *fd* carries into *dest*, hashed into *streamed* as written when it is
    given, until it ends, or until one byte past *limit* is written; return the bytes written,
    never more than *limit* and one. Raise ``_Infra``, naming *what* and the *seconds* it had, once
    the monotonic clock passes *deadline*."""
    written = 0
    for chunk in _output(fd, deadline=deadline, what=what, seconds=seconds):
        kept = chunk[:limit + 1 - written]
        dest.write(kept)
        if streamed is not None:
            streamed.update(kept)
        written += len(kept)
        if written > limit:
            break
    return written


def listing_problem(chunks: Iterable[bytes], keep: Callable[[bytes, bytes], None] | None = None) -> str | None:
    """The first bound main's tree passes as git lists it (``git ls-tree -r -t -z``), read from
    *chunks* of git's output in order; None when the listing ends
    within them. A record that is not a directory's (kind ``tree``) is a path, counted against
    ``TREE_MAX_ENTRIES``; a directory's is counted against ``TREE_MAX_DIRECTORIES``; each wherever
    it appears, and either's path is held to ``TREE_MAX_PATH_BYTES``. A record is held only until
    its end, or until it is longer than one within the bound can be, so a listing of any length is
    read in bounded memory and its git can be ended at the first bound passed. Each path goes to
    *keep*, as its record's metadata and its path."""
    paths = directories = 0
    held = b""
    for chunk in chunks:
        *records, held = (held + chunk).split(b"\0")
        for record in records:
            meta, _, path = record.partition(b"\t")
            if len(path) > TREE_MAX_PATH_BYTES:
                return _too_long(path)
            if meta.split(b" ")[1:2] == [b"tree"]:
                directories += 1
                if directories > TREE_MAX_DIRECTORIES:
                    return f"the tree holds more than the {TREE_MAX_DIRECTORIES} directories allowed"
                continue
            paths += 1
            if paths > TREE_MAX_ENTRIES:
                return f"the tree tracks more than the {TREE_MAX_ENTRIES} paths allowed"
            if keep is not None:
                keep(meta, path)
        if len(held) > LISTING_META_MAX_BYTES + TREE_MAX_PATH_BYTES:
            return _too_long(held.partition(b"\t")[2])
    return None


def _too_long(path: bytes) -> str:
    """The reason a tree tracking the path that starts *path* passes ``TREE_MAX_PATH_BYTES``."""
    return (f"the tree tracks a path longer than the {TREE_MAX_PATH_BYTES} bytes allowed: "
            f"{path[:NAME_SHOWN_CHARS].decode(errors='replace')}")


class FileIds:
    """The entries of main's tree whose mode makes them a file or a symbolic link, gathered from
    its listing as it streams (``listing_problem``'s *keep*) for git's check of their objects
    (``blob_sizes``): each one's id, a line of the file git reads, and the start of its path, in
    the listing's order. git's listing prints each entry's kind from its mode alone, so an entry
    may name a tree, a commit or a tag as a file; git's bundle packs that object by its own kind,
    and the package's checkout cannot write it. A submodule's entry names a commit of another
    repository, which no checkout reads, so its id is never looked up: each id git cannot find
    costs it a rescan of every pack directory."""

    def __init__(self, ids: BinaryIO) -> None:
        #: Where each id is written, a line each.
        self.ids = ids
        #: The first ``NAME_SHOWN_CHARS`` bytes of each one's path.
        self.paths: list[bytes] = []

    def __call__(self, meta: bytes, path: bytes) -> None:
        _, kind, oid = meta.split(b" ")[:3]
        if kind == b"blob":
            self.ids.write(oid + b"\n")
            self.paths.append(path[:NAME_SHOWN_CHARS])


def blob_sizes(chunks: Iterable[bytes], paths: list[bytes]) -> list[int] | str:
    """The size of each object git's check (``git cat-file`` with ``KIND_CHECK_FORMAT``) reports,
    read from *chunks* of its output in order, a line for each id of ``FileIds`` it was fed, whose
    paths are *paths*; or, in place of the sizes, the reason naming the first of those entries whose
    object is not a blob. Fewer sizes than *paths* when git's output ends early, which its exit
    status then explains. Raise ValueError for output git never writes for such ids: a line past
    ``KIND_LINE_MAX_BYTES``, one more than *paths*, or a size that is not a number."""
    sizes: list[int] = []
    held = b""
    for chunk in chunks:
        *lines, held = (held + chunk).split(b"\n")
        for line in lines:
            if len(sizes) == len(paths):
                raise ValueError(f"a line past the {len(paths)} ids it was fed: {line[:KIND_LINE_MAX_BYTES]!r}")
            kind, _, size = line.partition(b" ")
            if kind != b"blob":
                found = "missing" if size == b"missing" else f"a {kind.decode(errors='replace')}"
                return (f"main's tree names {paths[len(sizes)].decode(errors='replace')} as a file or symbolic link, "
                        f"but its object is {found}")
            sizes.append(int(size))
        if len(held) > KIND_LINE_MAX_BYTES:
            raise ValueError(f"a line longer than {KIND_LINE_MAX_BYTES} bytes: {held[:KIND_LINE_MAX_BYTES]!r}")
    return sizes


def _streamed(args: list[str], env: dict[str, str], workspace: Path, *, errors: Path, seconds: float,
              read: Callable[[Iterator[bytes]], str | None], stdin: BinaryIO | None = None) -> str | None:
    """What *read* finds in the output of git on *args* in the fresh repository, handed to it as
    it arrives: a reason, upon which git's session is ended, or None once git has written all and
    ended. git reads *stdin*, nothing when None; its failures are judged as ``_on_objects`` judges
    them, its messages going to the file *errors*. Past *seconds* the try is infra, and so is
    output git never writes (*read* raising ValueError)."""
    deadline = time.monotonic() + seconds
    what = f"git {_verb(args)}"
    with open(errors, "wb") as err:
        p = subprocess.Popen(["git", *GIT_FLAGS, *args], env=env, stdout=subprocess.PIPE, stderr=err,
                             stdin=subprocess.DEVNULL if stdin is None else stdin, start_new_session=True)
        assert p.stdout is not None
        try:
            try:
                problem = read(_output(p.stdout.fileno(), deadline=deadline, what=what, seconds=seconds))
            except ValueError as e:
                raise _Infra(f"{what} wrote what it never writes: {e}") from None
            if problem is not None:
                return problem
            try:
                returncode = p.wait(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise _Infra(f"{what} took longer than the {seconds:.0f} s it was given") from None
        finally:
            # Past a bound, out of time or failed here: git's session never outlives the read.
            if p.poll() is None:
                _kill(p)
            p.stdout.close()
    if returncode != 0:
        raise _objects_failure(args, returncode, git_messages(errors), workspace)
    return None


def _check_tree(fresh: Path, env: dict[str, str], work: Path, workspace: Path, deadline: float) -> None:
    """Hold main's tree, as git lists it in the fresh repository, to the listing's bounds
    (``listing_problem``), git's session ended at the first one passed, which is over_limit; then
    every file and symbolic link of it to naming a blob, as git's check of their objects reads them
    (``blob_sizes``): one that names another kind is unreadable. git's failures are judged as
    ``_on_objects`` judges them; the listing past ``TREE_LIST_SEC``, the check past
    ``KIND_CHECK_SEC``, or either past *deadline*, the program's end (``_left``), makes the try
    infra. git's messages and the ids the check reads are files in *work*."""
    ids = work / "listing.ids"
    with open(ids, "wb") as written:
        files = FileIds(written)
        problem = _streamed([*TREE_CACHE, "-C", str(fresh), "ls-tree", "-r", "-t", "-z", f"refs/heads/{BRANCH}"],
                            env, workspace, errors=work / "listing.err",
                            seconds=min(TREE_LIST_SEC, _left(deadline, "git ls-tree")),
                            read=lambda chunks: listing_problem(chunks, files))
    if problem is not None:
        raise _NoSubmission("over_limit", problem)
    checked: list[int] | str = []

    def check(chunks: Iterator[bytes]) -> str | None:
        nonlocal checked
        checked = blob_sizes(chunks, files.paths)
        return checked if isinstance(checked, str) else None

    with open(ids, "rb") as fed:
        problem = _streamed(["-C", str(fresh), "cat-file", KIND_CHECK_FORMAT], env, workspace,
                            errors=work / "kinds.err", seconds=min(KIND_CHECK_SEC, _left(deadline, "git cat-file")),
                            read=check, stdin=fed)
    if problem is not None:
        raise _unreadable("kind", problem)
    if len(checked) != len(files.paths):
        raise _Infra(f"git cat-file reported {len(checked)} of the {len(files.paths)} objects it was asked for")


def _kill(p: subprocess.Popen) -> None:
    """End *p*'s whole session and reap it."""
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    p.wait()


def _bundle(fresh: Path, out: Path, env: dict[str, str], workspace: Path, errors: Path, max_bytes: int,
            seconds: float, streamed: hashlib._Hash | None = None) -> int:
    """Write the bundle of the fresh repository's main into *out* from git's standard output,
    hashed into *streamed* as written when it is given (``_copy_bounded``), through a byte budget
    (a bundle past the package's limit is no submission at the commit
    point): one byte past *max_bytes*, git's session is ended and the bundle is the contestant's
    ``over_limit``. git's failures are judged as ``_on_objects`` judges them; its messages go to
    the file *errors*, so a full pipe never stalls it. Past *seconds*, the time left it
    (``_left``), the try is infra. Once synced, *out* must hold every byte written (``_landed``). Return the bytes
    written. The caller removes *out* unless the whole bundle was written."""
    args = ["-C", str(fresh), "bundle", "create", "-", f"refs/heads/{BRANCH}"]
    deadline = time.monotonic() + seconds
    with open(errors, "wb") as err, open(out, "wb") as dest:
        p = subprocess.Popen(["git", *GIT_FLAGS, *args], env=env, stdout=subprocess.PIPE, stderr=err,
                             stdin=subprocess.DEVNULL, start_new_session=True)
        assert p.stdout is not None
        try:
            written = _copy_bounded(p.stdout.fileno(), dest, limit=max_bytes, deadline=deadline,
                                    what=f"git {_verb(args)}", seconds=seconds, streamed=streamed)
            if written > max_bytes:
                raise _NoSubmission("over_limit", f"main's bundle holds more than {max_bytes} bytes, the most "
                                    "a source bundle may hold; the commit point stopped writing it there and kept "
                                    "none of it")
            try:
                returncode = p.wait(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                raise _Infra(f"git {_verb(args)} took longer than the {seconds:.0f} s it was given") from None
        except BaseException:
            # Past the limit, out of time or failed here: git's session never outlives the copy.
            if p.poll() is None:
                _kill(p)
            raise
        finally:
            p.stdout.close()
        dest.flush()
        os.fsync(dest.fileno())
    if returncode != 0:
        raise _objects_failure(args, returncode, git_messages(errors), workspace)
    _landed(out, written, "main's bundle")
    return written


def _landed(path: Path, size: int, what: str) -> None:
    """Raise ``_Infra`` unless the synced file *path*, *what* the program wrote, holds the *size*
    bytes it wrote as the store reports the file: one of another size lost or gained bytes on the
    way, the store's fault. The bytes themselves are never read back; the outputs report gives the
    digest of the bytes written, which every reader checks before it uses them."""
    held = os.lstat(path).st_size
    if held != size:
        raise _Infra(f"the store holds {held} bytes of {what}, not the {size} the program wrote: the store changed it")


def _lstat(path: Path) -> os.stat_result | None:
    """*path*'s own status, not following a link; None when nothing is there. Any other error is
    raised, for ``commit`` to split by errno: an error of the machine's must never read as a
    missing repository or part, which would be the contestant's zero."""
    try:
        return os.lstat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _stat(path: Path) -> os.stat_result | None:
    """*path*'s status, following links; None when nothing is there, as ``_lstat``."""
    try:
        return os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None


def _resolve(path: Path) -> Path:
    """Where *path* leads with every link followed, as git opens it; a part that is missing is
    kept as written. Raise unreadable for links that loop: only the contestant's links make
    them."""
    try:
        return Path(os.path.realpath(path, strict=True))
    except OSError as e:
        if e.errno == errno.ELOOP:
            raise _unreadable("loop", "the repository holds symbolic links that loop, at "
                              f"{str(path)[:NAME_SHOWN_CHARS]}") from None
    return Path(os.path.realpath(path))


def _inside(path: Path, root: Path) -> bool:
    return _resolve(path).is_relative_to(root)


def _lexically_inside(path: str, root: Path) -> bool:
    """Whether *path*, from a message, lies in *root*, resolved as far as it can be without
    raising."""
    return Path(os.path.realpath(path)).is_relative_to(root)


def _shown(path: Path, root: Path) -> str:
    """*path* as a reason names it: under the repository's directory if it lies there, else
    under the workspace."""
    for base in (root / ".git", root):
        if path.is_relative_to(base):
            return path.relative_to(base).as_posix()
    return str(path)


def walk_max_entries() -> int:
    """The most entries the walks of the object stores and refs visit, together: every object git
    is given (``OBJECTS_MAX``) as a loose file, and ``WALK_ROOM_ENTRIES`` beside them, so the
    count of objects is the bound a repository of loose objects meets first. Directories side by
    side, the walk's worst shape, take about 17 us an entry on one AMD EPYC 9534 core, so the
    bound's 6,000,000 entries take about 102 s."""
    return OBJECTS_MAX + WALK_ROOM_ENTRIES


class _Budget:
    """The entries the walks of one repository may still visit (``walk_max_entries``)."""

    def __init__(self) -> None:
        #: The most entries the walks visit.
        self.bound = walk_max_entries()
        #: Entries left to visit.
        self.left = self.bound

    def take(self) -> None:
        """Count one entry; raise over_limit for one past the bound."""
        if self.left <= 0:
            raise _NoSubmission("over_limit", f"the repository's object stores and refs hold more than {self.bound} "
                                "entries, more than the harness walks")
        self.left -= 1


def _walk(directory: Path, root: Path, budget: _Budget) -> None:
    """Walk *directory*, following the links that stay in *root*, and raise unreadable at the
    first entry git must not read: a link that leads outside *root*, or anything but a directory,
    a regular file or a link to one. Every entry visited is counted against *budget*. The walk
    keeps its own stack of directories and holds one directory's entries at a time; a link's
    target is walked once, which ends any cycle. A directory it cannot list raises, for
    ``commit`` to split by errno."""
    start = str(_resolve(directory))
    followed, stack = {start}, [start]
    while stack:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                budget.take()
                if entry.is_symlink():
                    path = Path(entry.path)
                    target = _resolve(path)
                    if not target.is_relative_to(root):
                        raise _unreadable("outside", f"the repository's {_shown(path, root)} links outside the "
                                                     "workspace")
                    st = _stat(path)
                    if st is not None and stat.S_ISDIR(st.st_mode):
                        if str(target) not in followed:
                            followed.add(str(target))
                            stack.append(str(target))
                    elif st is not None and not stat.S_ISREG(st.st_mode):
                        raise _unreadable("special", f"the repository's {_shown(path, root)} links to something "
                                                     "that is not a regular file")
                elif entry.is_dir(follow_symlinks=False):
                    stack.append(entry.path)
                elif not entry.is_file(follow_symlinks=False):
                    raise _unreadable("special", f"the repository's {_shown(Path(entry.path), root)} is not a "
                                                 "regular file")


def _open_regular(path: Path, what: str) -> BinaryIO:
    """*path* open for reading, never waiting on a pipe. Raise unreadable for anything but a
    regular file: git writes nothing else here."""
    f = os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK), "rb")
    if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
        f.close()
        raise _unreadable("special", f"the repository's {what} is not a regular file")
    return f


def _lines(path: Path, what: str) -> Iterator[bytes]:
    """Each line of the regular file *path*, without its line break, read a bounded line at a
    time (``_open_regular``). Raise unreadable for a file of more than ``LISTING_MAX_BYTES``, or
    a line of more than ``LINE_MAX_BYTES``."""
    with _open_regular(path, what) as f:
        size = os.fstat(f.fileno()).st_size
        if size > LISTING_MAX_BYTES:
            raise _unreadable("unwritten", f"the repository's {what} holds {size} bytes, more than "
                                           f"{LISTING_MAX_BYTES} bytes "
                              "the harness reads")
        while line := f.readline(LINE_MAX_BYTES + 1):
            if len(line) > LINE_MAX_BYTES:
                raise _unreadable("unwritten", f"the repository's {what} holds a line of more than "
                                               f"{LINE_MAX_BYTES} bytes, "
                                  "which git never writes")
            yield line.rstrip(b"\r\n")


def _object_stores(objects: Path, root: Path) -> list[Path]:
    """*objects* and every alternate object directory reachable from it, as git follows them.
    Raise unreadable for one that lies outside *root*, and over_limit for more than
    ``ALTERNATES_MAX``. An alternates entry is a path's bytes, resolved as git reads them."""
    stores: list[Path] = []
    todo = [(_resolve(objects), 0)]
    while todo:
        store, depth = todo.pop()
        if store in stores:
            continue
        stores.append(store)
        listing = store / "info" / "alternates"
        if depth >= _DEPTH or _lstat(listing) is None:
            continue
        if not _inside(listing, root):
            raise _unreadable("outside", "the repository's alternates file is outside the workspace")
        for line in _lines(listing, "alternates file"):
            entry = os.fsdecode(line.strip())
            if entry and not entry.startswith("#"):
                target = _resolve(store / entry)
                if not target.is_relative_to(root):
                    raise _unreadable("outside", f"the repository borrows objects from outside the workspace "
                                                 "(alternates: "
                                      f"{entry[:NAME_SHOWN_CHARS]})")
                if len(stores) + len(todo) >= ALTERNATES_MAX:
                    raise _NoSubmission("over_limit", f"the repository borrows objects from more than {ALTERNATES_MAX} "
                                        "object directories")
                todo.append((target, depth + 1))
    return stores


def _index_header(path: Path, what: str) -> tuple[int, int]:
    """The pack index *path*'s version and the objects it lists, its fan-out table's last entry,
    read through ``_open_regular`` and no further than the table. Raise unreadable for an index
    git refuses: shorter than its table, of an unknown version, with a fan-out that decreases, or
    of a size its count does not give."""
    with _open_regular(path, what) as f:
        size = os.fstat(f.fileno()).st_size
        head = f.read(8 + 256 * 4)
    version = 1
    if head[:4] == _INDEX_MAGIC:
        version, head = int.from_bytes(head[4:8], "big"), head[8:]
    table = [int.from_bytes(head[i:i + 4], "big") for i in range(0, 256 * 4, 4)] if len(head) >= 256 * 4 else []
    count = table[-1] if table else 0
    if version == 2:
        least = 8 + 256 * 4 + count * (20 + 4 + 4) + 20 + 20
        # Each object may have a 64-bit offset besides its 32-bit one, but the first.
        fits = least <= size <= least + max(count - 1, 0) * 8
    else:
        fits = size == 256 * 4 + count * (4 + 20) + 20 + 20
    if version not in (1, 2) or not table or any(a > b for a, b in zip(table, table[1:])) or not fits:
        raise _unreadable("unwritten", f"the repository's {what} is not a pack index git reads")
    return version, count


def _loose_directories(store: Path) -> list[str]:
    """*store*'s fan-out directories of loose objects."""
    with os.scandir(store) as entries:
        return [e.path for e in entries if _LOOSE_DIR.fullmatch(e.name) and e.is_dir()]


def _packs(store: Path) -> list[tuple[Path, Path]]:
    """*store*'s pack indexes, each with its pack: an index with no pack beside it git never
    opens."""
    directory = store / "pack"
    found = _stat(directory)
    if found is None or not stat.S_ISDIR(found.st_mode):
        return []
    with os.scandir(directory) as entries:
        names = [e.name.removesuffix(".idx") for e in entries if e.name.endswith(".idx")]
    return [(directory / f"{n}.idx", directory / f"{n}.pack") for n in names
            if _stat(directory / f"{n}.pack") is not None]


def _stores(stores: list[Path]) -> list[Path]:
    """Those of *stores* that are directories."""
    return [s for s in stores if (found := _stat(s)) is not None and stat.S_ISDIR(found.st_mode)]


def object_count(stores: list[Path], root: Path) -> int:
    """The objects *stores* hold, as git finds them: the loose ones in each store's fan-out
    directories, and each pack index's count (``_index_header``) for an index with its pack
    beside it. An object kept twice is counted twice."""
    held = 0
    for store in _stores(stores):
        for directory in _loose_directories(store):
            with os.scandir(directory) as entries:
                held += sum(1 for e in entries if _LOOSE_NAME.fullmatch(e.name))
        for index, _ in _packs(store):
            held += _index_header(index, _shown(index, root))[1]
    return held


def _inflated_start(raw: bytes, want: int, more: bool, what: Callable[[], str]) -> bytes | None:
    """Up to *want* bytes the zlib stream that starts with *raw* inflates to, inflating no more.
    None when it is not a stream git can inflate: git fails at the same bytes, before it
    allocates anything. *more* says the stream goes on past *raw*; then fewer bytes mean empty
    blocks that git would inflate through to a header this program never reads, which is the
    contestant's unreadable repository, *what* naming the object."""
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(raw, want)
    except zlib.error:
        return None
    if len(out) < want and not inflater.eof and more:
        raise _unreadable("unwritten", f"the repository's {what()} does not show its header within the "
                                       f"{HEADER_INPUT_BYTES} "
                          "bytes git writes it in")
    return out


def _loose_header(path: str, what: Callable[[], str]) -> tuple[str, int] | None:
    """The kind and size the header of the loose object at *path* declares, read as
    ``_open_regular`` reads, through a bare descriptor since there may be millions; None for an
    object git cannot read either."""
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise _unreadable("special", f"the repository's {what()} is not a regular file")
        raw = os.read(fd, HEADER_INPUT_BYTES + 1)
    finally:
        os.close(fd)
    out = _inflated_start(raw[:HEADER_INPUT_BYTES], 64, len(raw) > HEADER_INPUT_BYTES, what)
    if out is None or b"\0" not in out:
        return None
    kind, _, size = out[:out.index(b"\0")].partition(b" ")
    if kind not in (b"blob", b"tree", b"commit", b"tag") or not size.isdigit():
        return None
    return kind.decode(), int(size)


def _varint(data: bytes, at: int) -> tuple[int, int] | None:
    """The little-endian base-128 number at *data*[*at*:] and where it ends; None past the data
    or past 64 bits."""
    value = shift = 0
    while at < len(data) and shift <= 63:
        c = data[at]
        value |= (c & 0x7F) << shift
        at, shift = at + 1, shift + 7
        if not c & 0x80:
            return value, at
    return None


def _pack_entry(fd: int, offset: int) -> tuple[int, int, int | bytes, int] | None:
    """The pack entry at *offset* of the pack open as *fd*: its type, the size its header
    declares, its delta's base (the base's offset for an offset delta, its id for a reference
    delta, else 0), and where its zlib stream starts. None for an entry git cannot read either."""
    head = os.pread(fd, 64, offset)
    if not head:
        return None
    kind, size, shift, at = (head[0] >> 4) & 7, head[0] & 0x0F, 4, 1
    c = head[0]
    while c & 0x80:
        if at >= len(head) or shift > 63:
            return None
        c = head[at]
        size, shift, at = size | (c & 0x7F) << shift, shift + 7, at + 1
    if kind == _OFS_DELTA:
        if at >= len(head):
            return None
        c, at = head[at], at + 1
        back = c & 0x7F
        while c & 0x80:
            if at >= len(head) or back >= 1 << 56:
                return None
            c, at = head[at], at + 1
            back = ((back + 1) << 7) | (c & 0x7F)
        if back <= 0 or back >= offset:
            return None
        return kind, size, offset - back, offset + at
    if kind == _REF_DELTA:
        if len(head) < at + 20:
            return None
        return kind, size, head[at:at + 20], offset + at + 20
    if kind not in _PACK_KINDS:
        return None
    return kind, size, 0, offset + at


def _delta_result(fd: int, start: int, what: str) -> int | None:
    """The size the delta whose zlib stream starts at *start* declares for its result, the size
    git allocates to rebuild the object; None for a delta git cannot read either."""
    raw = os.pread(fd, HEADER_INPUT_BYTES + 1, start)
    out = _inflated_start(raw[:HEADER_INPUT_BYTES], 20, len(raw) > HEADER_INPUT_BYTES, lambda: what)
    base = _varint(out, 0) if out is not None else None
    result = _varint(out, base[1]) if base is not None and out is not None else None
    return None if result is None else result[0]


def _offset_of(fd: int, version: int, count: int, oid: bytes) -> int | None:
    """Where the pack entry of the object *oid* starts, by the index open as *fd*; None when the
    index does not list it. A binary search over the index's ids, within the fan-out's range."""
    table = 8 if version == 2 else 0
    lo = int.from_bytes(os.pread(fd, 4, table + 4 * (oid[0] - 1)), "big") if oid[0] else 0
    hi = int.from_bytes(os.pread(fd, 4, table + 4 * oid[0]), "big")
    ids, width = (table + 256 * 4, 20) if version == 2 else (256 * 4 + 4, 24)
    while lo < hi:
        mid = (lo + hi) // 2
        found = os.pread(fd, 20, ids + mid * width)
        if found == oid:
            return _index_offset(fd, version, count, mid)
        lo, hi = (mid + 1, hi) if found < oid else (lo, mid)
    return None


def _index_offset(fd: int, version: int, count: int, n: int) -> int | None:
    """The pack offset the index open as *fd* lists for its *n*-th object; None for one git
    cannot read either."""
    if version == 1:
        return int.from_bytes(os.pread(fd, 4, 256 * 4 + n * 24), "big")
    value = int.from_bytes(os.pread(fd, 4, 8 + 256 * 4 + count * 24 + n * 4), "big")
    if not value & 0x80000000:
        return value
    large = os.pread(fd, 8, 8 + 256 * 4 + count * 28 + (value & 0x7FFFFFFF) * 8)
    return int.from_bytes(large, "big") if len(large) == 8 else None


def _index_offsets(fd: int, version: int, count: int) -> Iterator[int | None]:
    """Each pack offset the index open as *fd* lists, a bounded chunk of its table at a time."""
    step = 1 << 16
    for first in range(0, count, step):
        n = min(step, count - first)
        if version == 1:
            chunk = os.pread(fd, n * 24, 256 * 4 + first * 24)
            yield from (int.from_bytes(chunk[i:i + 4], "big") for i in range(0, len(chunk) - 23, 24))
            continue
        chunk = os.pread(fd, n * 4, 8 + 256 * 4 + count * 24 + first * 4)
        for i in range(0, len(chunk) - 3, 4):
            value = int.from_bytes(chunk[i:i + 4], "big")
            yield value if not value & 0x80000000 else _index_offset(fd, version, count, first + i // 4)


class _Reads:
    """The pack entries the size scan may still read, across every pack (``SCAN_MAX_READS``)."""

    def __init__(self) -> None:
        #: Entries left to read.
        self.left = SCAN_MAX_READS

    def take(self) -> None:
        """Count one entry read; raise over_limit for one past the bound."""
        if self.left <= 0:
            raise _NoSubmission("over_limit", f"sizing the repository's objects needs more than {SCAN_MAX_READS} "
                                              "pack entries read, more than the harness reads")
        self.left -= 1


class _PackScan:
    """The size scan of one pack: its open files, the reads it shares with every pack's scan, and
    what it has found, kept so that no chain of deltas is followed twice."""

    def __init__(self, pack: int, index: int, version: int, count: int, what: str, reads: _Reads) -> None:
        #: The pack, open.
        self.pack = pack
        #: Its index, open.
        self.index = index
        #: The index's version.
        self.version = version
        #: The objects the index lists.
        self.count = count
        #: The pack, as a reason names it.
        self.what = what
        #: The reads left to the whole scan.
        self.reads = reads
        #: Each entry whose kind was found, by offset: the kind's code (0 when a base is not in
        #: this pack) in its low three bits, its depth in its chain in the twelve above them, and
        #: its chain bytes above those (``_CHAIN_SHIFT``).
        self.kinds: dict[int, int] = {}
        #: Each base id looked up, to its entry's offset; None when the index does not list it.
        self.offsets: dict[bytes, int | None] = {}

    def entry(self, offset: int) -> tuple[int, int, int | bytes, int] | None:
        """The pack entry at *offset* (``_pack_entry``), counted against the scan's reads."""
        self.reads.take()
        return _pack_entry(self.pack, offset)

    def offset_of(self, oid: bytes) -> int | None:
        """Where the entry of the object *oid* starts (``_offset_of``), looked up once."""
        if oid not in self.offsets:
            self.offsets[oid] = _offset_of(self.index, self.version, self.count, oid)
        return self.offsets[oid]

    def kind(self, offset: int, entry: tuple[int, int, int | bytes, int], rebuilt: int) -> tuple[str | None, int]:
        """The kind of the object whose entry *entry* is at *offset*, of *rebuilt* bytes: its own,
        or its chain's root's, followed from base to base as far as a kind already found; and its
        chain bytes, what rebuilding it reads: its own bytes, a delta's own too, on its base's chain
        bytes, down to a whole object (a base outside the pack counts nothing). Every entry passed
        is kept with its kind, depth and chain bytes, so none is read for this twice; a delta passed
        has its result read once more (``_delta_result``). The kind is None when a base is not in
        this pack. Raise unreadable for a chain deeper than git writes, or one that loops."""
        trail: list[tuple[int, int]] = []
        code, declared, base, start = entry
        at, own = offset, rebuilt + (declared if code in (_OFS_DELTA, _REF_DELTA) else 0)
        while code in (_OFS_DELTA, _REF_DELTA):
            trail.append((at, own))
            if len(trail) > _DELTA_DEPTH_MAX:
                raise self.too_deep()
            following = base if code == _OFS_DELTA else self.offset_of(base)
            known = None if following is None else self.kinds.get(following)
            if known is not None:
                code, depth, below = known & 7, (known >> 3) & _DEPTH_MASK, known >> _CHAIN_SHIFT
                break
            found = None if following is None else self.entry(following)
            if found is None:
                code, depth, below = 0, 0, 0
                break
            code, declared, base, start = found
            at = following
            if code in (_OFS_DELTA, _REF_DELTA):
                result = _delta_result(self.pack, start, self.what)
                own = (result if result is not None else declared) + declared
            else:
                own = declared
        else:
            depth, below = 0, own
            self.kinds[at] = code | below << _CHAIN_SHIFT
        if depth + len(trail) > _DELTA_DEPTH_MAX:
            raise self.too_deep()
        for steps, (passed, cost) in enumerate(reversed(trail), start=1):
            below += cost
            self.kinds[passed] = code | (depth + steps) << 3 | below << _CHAIN_SHIFT
        return _PACK_KINDS.get(code), below

    def too_deep(self) -> _NoSubmission:
        return _unreadable("unwritten", f"the repository's {self.what} holds a delta chain longer than the "
                                        f"{_DELTA_DEPTH_MAX} "
                           "git writes, or one that loops")


def _size_problem(size: int, delta: bool, kind_of: Callable[[], str | None],
                  where: Callable[[], str]) -> str | None:
    """Why git must not be given an object of *size*, stored as a delta or not; None when it
    may. *kind_of* gives its kind (None when its delta's base is not in its pack, read as the
    stricter) and *where* names it; each is asked only when needed, since kinds are found
    through a delta's bases."""
    if size > OBJECT_MAX_BYTES:
        return (f"the repository holds a {_kind_shown(kind_of())} of {size} bytes ({where()}), more than "
                f"{OBJECT_MAX_BYTES} bytes the harness gives git for one object")
    if delta and size > DELTA_MAX_BYTES:
        return (f"the repository holds a {_kind_shown(kind_of())} of {size} bytes stored as a delta ({where()}), "
                f"more than {DELTA_MAX_BYTES} bytes git ever stores as one")
    if size > NONBLOB_MAX_BYTES and (kind := kind_of()) != "blob":
        return (f"the repository holds a {_kind_shown(kind)} of {size} bytes ({where()}), more than "
                f"{NONBLOB_MAX_BYTES} bytes the harness gives git for one commit, tree or tag, which git loads "
                "whole")
    return None


def _kind_shown(kind: str | None) -> str:
    return kind or "object whose kind its delta's base does not show"


def object_sizes(stores: list[Path], root: Path, max_bytes: int) -> str | tuple[int, int, int]:
    """What the objects *stores* hold inflate to together, what their tree objects hold and what
    rebuilding their deltas reads (their chain bytes, ``_PackScan.kind``), or why git must not be
    given one of them (``_size_problem``), all of them, which inflate past *max_bytes* together or
    whose deltas' chain bytes pass ``CHAIN_BYTES_MAX``, or their tree objects, which
    hold more than ``TREE_BYTES_MAX`` together, each size read from the object's header alone: a
    loose object's first bytes, inflated as far as its header; a packed one's entry header,
    reached through its index, and for a delta the start of its data, which declares the size of
    the object it rebuilds, the size it counts as. A tree's bytes are those of the tree it
    rebuilds, counted whether or not its entries name objects that are there: git walks them all.
    A delta whose base is not in its own pack shows no kind (git writes none: a thin pack is
    completed when fetched), and counts as a tree, the stricter reading."""
    reads = _Reads()
    inflated = tree_bytes = chain_bytes = 0
    for store in _stores(stores):
        for directory in _loose_directories(store):
            with os.scandir(directory) as entries:
                paths = [e.path for e in entries if _LOOSE_NAME.fullmatch(e.name)]
            for path in paths:
                named = lambda path=path: _shown(Path(path), root)  # noqa: E731
                header = _loose_header(path, named)
                if header is None:
                    continue
                kind, size = header
                inflated += size
                if kind == "tree":
                    tree_bytes += size
                problem = _size_problem(size, False, lambda kind=kind: kind, named)
                if problem is not None:
                    return problem
        for index_path, pack_path in _packs(store):
            what = _shown(pack_path, root)
            version, count = _index_header(index_path, _shown(index_path, root))
            with _open_regular(index_path, _shown(index_path, root)) as index, _open_regular(pack_path, what) as pack:
                scan = _PackScan(pack.fileno(), index.fileno(), version, count, what, reads)
                for offset in _index_offsets(index.fileno(), version, count):
                    entry = None if offset is None else scan.entry(offset)
                    if entry is None:
                        continue
                    delta = entry[0] in (_OFS_DELTA, _REF_DELTA)
                    result = _delta_result(pack.fileno(), entry[3], what) if delta else entry[1]
                    size = max(entry[1], result or 0)
                    rebuilt = result if result is not None else entry[1]
                    inflated += rebuilt
                    kind, chain = scan.kind(offset, entry, rebuilt)
                    if delta:
                        chain_bytes += chain
                    # A delta whose base is outside its pack shows no kind: read as the stricter.
                    if kind in ("tree", None):
                        tree_bytes += size
                    problem = _size_problem(size, delta, lambda kind=kind: kind,
                                            lambda offset=offset: f"{what} at {offset}")
                    if problem is not None:
                        return problem
    if tree_bytes > TREE_BYTES_MAX:
        return (f"the repository's tree objects hold {tree_bytes} bytes together, more than "
                f"{TREE_BYTES_MAX} bytes the harness gives git to walk and bundle")
    if inflated > max_bytes:
        return (f"the repository's objects inflate to {inflated} bytes together, more than "
                f"{max_bytes} bytes, the build's writable disk cap, the harness gives git and the package's scan")
    if chain_bytes > CHAIN_BYTES_MAX:
        return (f"rebuilding the repository's deltas reads {chain_bytes} bytes together, more than the "
                f"{CHAIN_BYTES_MAX} bytes the harness gives git and the package's scan")
    return inflated, tree_bytes, chain_bytes


def check_repository(workspace: Path, max_bytes: int) -> tuple[str, str] | None:
    """(kind, reason) when *workspace*'s repository is absent, must not be read, or holds more
    packs than ``PACKS_MAX``, more objects than ``OBJECTS_MAX``, one larger than git is given,
    objects that inflate past *max_bytes* together, deltas whose rebuilding reads past
    ``CHAIN_BYTES_MAX`` together, or tree objects past ``TREE_BYTES_MAX`` together
    (``object_sizes``); None when its objects and refs lie in the workspace within the bounds.
    An error of the filesystem's is raised."""
    root = Path(os.path.realpath(workspace))
    git = root / ".git"
    found = _lstat(git)
    if found is None:
        return "no_main", "the workspace has no git repository; only the latest commit on main is evaluated"
    try:
        if stat.S_ISLNK(found.st_mode):
            raise _unreadable("link", "the workspace's .git is a symbolic link, so its history is not in the workspace")
        if not stat.S_ISDIR(found.st_mode):
            raise _unreadable("gitfile", "the workspace's .git is a file that points at a repository elsewhere, so "
                                          "its history is not in the workspace")
        if _lstat(git / "commondir") is not None:
            raise _unreadable("worktree", "the workspace's .git is a linked worktree's directory, whose history "
                                           "lives elsewhere")
        if _lstat(git / "reftable") is not None:
            raise _unreadable("reftable", "the repository keeps its refs in reftable format, which the harness does "
                                           "not read")
        for part in ("objects", "refs", "packed-refs"):
            if _lstat(git / part) is not None and not _inside(git / part, root):
                raise _unreadable("outside", f"the repository's {part} lead outside the workspace")
        budget = _Budget()
        for part in ("objects", "refs"):
            st = _stat(git / part)
            if st is not None and stat.S_ISDIR(st.st_mode):
                _walk(git / part, root, budget)
        stores = _object_stores(git / "objects", root)
        for store in stores[1:]:
            st = _stat(store)
            if st is not None and stat.S_ISDIR(st.st_mode):
                _walk(store, root, budget)
        packs = sum(len(_packs(store)) for store in _stores(stores))
        if packs > PACKS_MAX:
            return "over_limit", (f"the repository's objects lie in {packs} packs, more than the {PACKS_MAX} the "
                                  "harness reads")
        held = object_count(stores, root)
        if held > OBJECTS_MAX:
            return "over_limit", (f"the repository holds {held} objects, more than {OBJECTS_MAX} objects the harness "
                                  "bundles")
        sized = object_sizes(stores, root, max_bytes)
        if isinstance(sized, str):
            return "over_limit", sized
    except _NoSubmission as e:
        return e.kind, e.reason
    return None


def _packed_ref(git: Path, name: str) -> str | None:
    """The object ``packed-refs`` records for the ref *name*; None when it records none."""
    if _lstat(git / "packed-refs") is None:
        return None
    wanted = os.fsencode(name)
    for line in _lines(git / "packed-refs", "packed-refs"):
        sha, _, ref = line.strip().partition(b" ")
        if ref == wanted and sha and not sha.startswith((b"#", b"^")):
            return sha.decode(errors="replace")
    return None


def _loose_ref(git: Path, name: str) -> str | None:
    """What the loose ref *name* holds; None when there is no such file."""
    st = _stat(git / name)
    if st is None or stat.S_ISDIR(st.st_mode):
        return None
    with _open_regular(git / name, name) as f:
        data = f.read(LINE_MAX_BYTES + 1)
    if len(data) > LINE_MAX_BYTES:
        raise _unreadable("unwritten", f"the repository's {name} holds more than {LINE_MAX_BYTES} bytes, which "
                                       "git never "
                          "writes")
    return data.decode(errors="replace").strip()


def branches(git: Path) -> list[str]:
    """The repository's branch names, loose and packed, sorted: at most ``BRANCHES_MAX`` of them,
    the first found."""
    names: set[str] = set()
    heads = git / "refs" / "heads"
    found = _stat(heads)
    stack = [heads] if found is not None and stat.S_ISDIR(found.st_mode) else []
    while stack and len(names) < BRANCHES_MAX:
        with os.scandir(stack.pop()) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                else:
                    names.add(_text(Path(entry.path).relative_to(heads).as_posix()))
                if len(names) >= BRANCHES_MAX:
                    break
    if len(names) < BRANCHES_MAX and _lstat(git / "packed-refs") is not None:
        for line in _lines(git / "packed-refs", "packed-refs"):
            sha, _, ref = line.strip().partition(b" ")
            if ref.startswith(b"refs/heads/") and sha and not sha.startswith((b"#", b"^")):
                names.add(_text(os.fsdecode(ref.removeprefix(b"refs/heads/"))))
                if len(names) >= BRANCHES_MAX:
                    break
    return sorted(names)


def main_commit(git: Path) -> str | None:
    """The object ``main`` names, read from its loose ref or ``packed-refs`` and following
    symbolic refs as git does; None when there is no main. Raise unreadable for a ref git would
    not write."""
    name = f"refs/heads/{BRANCH}"
    for _ in range(_DEPTH):
        value = _loose_ref(git, name)
        if value is None:
            value = _packed_ref(git, name)
        if value is None:
            return None
        if value.startswith("ref:"):
            name = value.removeprefix("ref:").strip()
            parts = name.split("/")
            if not _REF.fullmatch(name) or ".." in parts or len(name.encode()) > REF_MAX_BYTES \
                    or any(len(part.encode()) > REF_PART_MAX_BYTES for part in parts):
                raise _unreadable("main", f"main is a symbolic ref to {name[:NAME_SHOWN_CHARS]!r}, which is not "
                                          "a ref")
            continue
        if _SHA256.fullmatch(value):
            raise _unreadable("sha256", "the repository uses SHA-256 object names, which the harness does not read")
        if not _SHA1.fullmatch(value):
            raise _unreadable("main", f"main holds {value[:NAME_SHOWN_CHARS]!r}, which is not an object name")
        return value
    raise _unreadable("main", f"main is a chain of more than {_DEPTH} symbolic refs")


def _named(names: list[str]) -> str:
    """*names*, for a reason: at most ``BRANCHES_NAMED`` of them, and how many more."""
    shown = ", ".join(names[:BRANCHES_NAMED])
    more = len(names) - BRANCHES_NAMED
    if more > 0:
        shown += f" and {'at least ' if len(names) >= BRANCHES_MAX else ''}{more} more"
    return shown


def commit(workspace: Path, out: Path, *, tmp: Path, max_bytes: int, seconds: int,
           outputs_out: Path | None = None) -> CommitRecord:
    """Bundle *workspace*'s main into *out*, working in a fresh directory under *tmp*, never more
    than *max_bytes*, which bounds the repository's inflated objects too (``check_repository``,
    ``_bundle``), all of it within *seconds* from now: the reads of the repository spend them, and
    each git command runs within what is left (``_left``). The bundle is written to the store once
    and never read back. With *outputs_out*, it is hashed as it is written, and its outputs report
    gives that digest and the bytes written (``_report``), which the store must hold once synced
    (``_landed``). Raise ``_NoSubmission`` for a workspace with nothing to bundle or a bundle past
    the limit, ``_IoError`` for an errno of the machine's, ``_Infra`` for anything else the
    repository did not cause, running out of *seconds* among it."""
    deadline = time.monotonic() + seconds
    if workspace.is_symlink() or not workspace.is_dir():
        raise _Infra(f"the workspace {workspace} is not a directory")
    root = Path(os.path.realpath(workspace))
    git = root / ".git"
    try:
        checked = check_repository(workspace, max_bytes)
        if checked is not None:
            raise _NoSubmission(*checked)
        names = branches(git)
        sha = main_commit(git)
    except OSError as e:
        name = errno.errorcode.get(e.errno or 0, "EIO")
        in_workspace = isinstance(e.filename, str) and _lexically_inside(e.filename, root)
        if name in REFUSED_ERRNOS and in_workspace:
            raise _unreadable("refused", f"{name}: the workspace's repository cannot be read, its permissions, "
                                         "which the "
                              f"contestant set, refuse it: {e}") from e
        raise _IoError(name, f"reading the workspace's repository failed: {e}") from e
    if sha is None:
        listed = f"its branches are {_named(names)}" if names else "it has no commits"
        raise _NoSubmission("no_main", f"the workspace's repository has no commit on main ({listed}); only the "
                            f"latest commit on main is evaluated", names)
    work = Path(tempfile.mkdtemp(prefix="hls-commit-", dir=tmp))
    try:
        env = git_env(work)
        fresh = work / "main.git"
        p = _git(["init", "--quiet", "--bare", "--template=", str(fresh)], env, _left(deadline, "git init"))
        if p.returncode != 0:
            raise _Infra(f"git init failed with status {p.returncode}: {p.stderr.strip()[-500:]}")
        (fresh / "objects" / "info").mkdir(parents=True, exist_ok=True)
        (fresh / "objects" / "info" / "alternates").write_bytes(os.fsencode(_resolve(git / "objects")) + b"\n")
        try:
            _on_objects(["-C", str(fresh), "update-ref", f"refs/heads/{BRANCH}", sha], env, root,
                        _left(deadline, "git update-ref"))
        except _NoSubmission as e:
            raise _NoSubmission(e.kind, e.reason, names) from None
        (out.parent / PARTIAL_DIR).mkdir(parents=True, exist_ok=True)
        partial = out.parent / PARTIAL_DIR / f"{out.name}-{os.getpid()}"
        try:
            try:
                streamed = None if outputs_out is None else hashlib.sha256()
                written = _bundle(fresh, partial, env, root, work / "bundle.err", max_bytes,
                                  _left(deadline, "git bundle"), streamed)
                # git's bundle read every object of main's history, so its tree lists; a tree that
                # lists past a bound, or names an object as a file that is not one, is never kept.
                _check_tree(fresh, env, work, root, deadline)
            except _NoSubmission as e:
                raise _NoSubmission(e.kind, e.reason, names) from None
            os.replace(partial, out)
            if outputs_out is not None:
                assert streamed is not None
                _report(outputs_out, out.name, streamed.hexdigest(), written)
        finally:
            # A bundle cut short, by a failure or by the limit, is never kept, nor its directory
            # once empty.
            partial.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                partial.parent.rmdir()
        return CommitRecord(status="ok", commit=sha, reason=None, branches=names, at=time.time())
    finally:
        shutil.rmtree(work, ignore_errors=True)


def snapshot_name(n: int) -> str:
    """The file name of snapshot *n*, counted from 1."""
    if n < 1:
        raise ValueError(f"snapshots are counted from 1, got {n}")
    return f"main-{n:06d}.bundle"


def snapshot_numbers(directory: Path) -> list[int]:
    """The numbers of the snapshots in *directory*, ascending."""
    if not directory.is_dir():
        return []
    return sorted(int(m.group(1)) for p in directory.iterdir() if (m := _SNAPSHOT.fullmatch(p.name)))


def prune_snapshots(directory: Path, keep: int = SNAPSHOT_KEEP) -> list[str]:
    """Delete every snapshot in *directory* but the *keep* highest-numbered; return the names
    deleted. Numbers, not times, say which is newest."""
    doomed = snapshot_numbers(directory)[:-keep] if keep > 0 else snapshot_numbers(directory)
    for n in doomed:
        (directory / snapshot_name(n)).unlink(missing_ok=True)
    return [snapshot_name(n) for n in doomed]


def _write(path: Path, text: str) -> None:
    """Write *path* whole, through a temporary name beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.parent / f".tmp-{path.name}-{os.getpid()}"
    partial.write_text(text)
    os.replace(partial, path)


def _write_output(path: Path, data: bytes) -> None:
    """Write the output *path* whole, through a temporary name beside it, synced, and hold the
    store to every byte of *data* (``_landed``): its outputs report gives the digest of *data*,
    never a read of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.parent / f".tmp-{path.name}-{os.getpid()}"
    with open(partial, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _landed(partial, len(data), path.name)
    os.replace(partial, path)


def _report(report: Path, name: str, digest: str, size: int) -> None:
    """Report the output *name*'s sha256, *digest*, and its *size* in *report*, keyed by its name:
    the record ``hls.store.manifest.report_written_outputs`` writes for an output its writer hashed
    as it wrote it. *report* is written in place and cut to the record's length, as that module
    writes it, so a report file the stage made larger beforehand, on a node-local disk the agent
    may have filled, takes no new space."""
    data = json.dumps({name: {"sha256": digest, "size": size}}, indent=1).encode()
    out = os.open(report, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(data)
        while view:
            view = view[os.write(out, view):]
        os.ftruncate(out, len(data))
        os.fsync(out)
    finally:
        os.close(out)


def _positive(text: str) -> int:
    """An argument that is a count of at least one."""
    count = int(text)
    if count < 1:
        raise ValueError(f"{count} is not positive")
    return count


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="commit.py", description="Bundle the workspace's main branch.")
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-submission", type=Path)
    ap.add_argument("--record", type=Path)
    ap.add_argument("--outputs-out", type=Path)
    ap.add_argument("--tmp", type=Path, default=Path(tempfile.gettempdir()))
    ap.add_argument("--max-bytes", type=_positive, required=True)
    ap.add_argument("--seconds", type=_positive, required=True)
    args = ap.parse_args(argv)
    if args.outputs_out is not None and args.no_submission is not None \
            and os.path.abspath(args.no_submission.parent) != os.path.abspath(args.out.parent):
        ap.error("--outputs-out keys each output by its name, so --no-submission must be beside the bundle")
    try:
        try:
            # The bundle's outputs report is written as it is kept, from its hash as git wrote it.
            record = commit(args.workspace, args.out, tmp=args.tmp, max_bytes=args.max_bytes,
                            seconds=args.seconds, outputs_out=args.outputs_out)
            code = 0
        except _NoSubmission as e:
            now = time.time()
            record = CommitRecord(status=e.kind, commit=None, reason=e.reason, branches=e.branches, at=now)
            written = args.no_submission
            if written is not None:
                data = NoSubmission(kind=e.kind, reason=e.reason, at=now).to_json().encode()
                _write_output(written, data)
                if args.outputs_out is not None:
                    _report(args.outputs_out, written.name, hashlib.sha256(data).hexdigest(), len(data))
            code = NO_MAIN_EXIT if e.kind == "no_main" else UNREADABLE_EXIT
        if args.record is not None:
            _write(args.record, json.dumps(asdict(record), indent=2))
    except _Infra as e:
        print(f"hls-commit: infra: {_text(str(e))}", file=sys.stderr)
        return 1
    except (_IoError, OSError) as e:
        failed = e if isinstance(e, _IoError) else _IoError(errno.errorcode.get(e.errno or 0, "EIO"), str(e))
        print(f"hls-commit: io-error: {failed}", file=sys.stderr)
        if args.record is not None:
            try:
                _write(args.record, json.dumps(asdict(CommitRecord(status="io_error", commit=None, reason=str(failed),
                                                                   branches=[], at=time.time())), indent=2))
            except OSError:
                pass  # stderr names the errno all the same
        return IO_ERROR_EXIT
    print(f"hls-commit: {record.status} {record.commit or record.reason}")
    return code


if __name__ == "__main__":
    sys.exit(main())
