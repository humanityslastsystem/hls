"""The source tree a package builds from: the ``main`` branch of the contestant's git bundle,
cloned on the host into node-local disk.

The clone runs host git on contestant data, so it is hardened against what a repository can make
git do:
- only a bundle file is cloned, into a repository this role makes with no template. A bundle
  carries refs and objects and nothing else: no hooks, no repository config, no
  ``objects/info/alternates`` and no gitfile, so the clone can neither run contestant code nor
  read another repository on the host;
- git runs with a minimal environment: no ``GIT_*`` variable of the host's, the system and
  global configuration switched off (``GIT_CONFIG_NOSYSTEM=1``, ``GIT_CONFIG_GLOBAL=/dev/null``)
  and a fresh HOME, so no hooks path, filter driver, URL rewrite or LFS smudge of the host's
  applies;
- submodules are never cloned.

Before any file is written, the commit's tree is read from git's own listing and held to the
limits below. Every later step copies the tree, mounts it or builds it into the image, and
within these limits none of that can fail on the tree's account: a tree past them is the
contestant's (``over_limit``), and a step that fails on a tree within them failed for the
harness. The limits sit far above any submission's source, which carries code and no weights.
They include the names git's checkout refuses to write (``git_refuses``), which a tree built
with git's plumbing can hold, and paths that are not UTF-8, which these records cannot hold.
The listing is read as git writes it and held, as it streams, to the bounds the commit point
holds main's tree to (``hls.agent.commit.listing_problem``): its paths and its directories,
each counted wherever it appears, and the length of each path. A tree may name one subtree many
times, so a few objects can list billions of paths or directories; git is ended at the first
bound passed, and neither the listing nor git's walk is ever held whole. git's listing prints
each entry's kind from its mode alone, so the object of every file and symbolic link is then read
by git's check (``hls.agent.commit.blob_sizes``), which gives each one's size: one that is not a
blob, which git's checkout cannot write and a bundle from outside the store may hold, is the
contestant's too.

git's checkout writes every file as its blob holds it, whatever the tree's ``.gitattributes``
asks (``CHECKOUT_ATTRIBUTES``), so the bytes the limits count are the bytes written; the brief
says so.

A tree may hold no symlink (mode 120000 in the listing). The image build's ``%files`` copy
follows every link it meets, so a link, or a chain of links each of which leads inside the
tree, could carry a host file into the image, and links to directories could multiply the
tree's bytes far past its limit.

    python -m hls.package.checkout --bundle B --dest D --out source.json --outputs-out R
        --size BYTES --expect SHA256 --max-bytes N [--names NAME[,NAME...] --scan-out FILE]

checks the bundle's size before any read and its digest before any verdict. Its git commands run
to the stage's deadline, which ends the role, as infra, when one is still running then; only the
tree's listing and the check of its objects have times of their own (``listed_tree``). Every
bundle comes
with the digest and the size it must have, *SHA256* and *BYTES*: the ones the agent_build
manifest records, or for the copy ``hls package`` made of an operator's named bundle, the digest
the operator states and the copy's size, which the plan records. A file whose size is not *BYTES*
is not the recorded bundle: the role refuses it (a ``refused:`` line and ``DIGEST_MISMATCH_EXIT``)
before reading a byte. A bundle of more than *N* bytes, the build's writable disk cap, which the
commit point held the repository to (``hls.agent.commit``) and ``hls package`` holds a named
bundle to, is the contestant's (``over_limit``) by its size alone, unread and never cloned, since
the clone would hold it whole on the node. Any other bundle is read once where it lies, with no
copy on the node (``receive``): every byte is
hashed as it is handed to git's ``index-pack``, which writes the clone's pack as it reads it, so
git never opens the bundle and gets only the bytes the role hashed. No verdict is given, and
nothing reads an object of it, until the whole bundle has *SHA256*: one with another digest is
refused, its clone removed, and when the bundle where it lies still has *SHA256* the read went
wrong on the way (``COPY_FAILED_EXIT``), which a retry reads again. It writes the outcome to
``source.json``, with the bundle's sha256 as above:
``ok`` with the commit and the tracked
files; ``no_main`` when the bundle has no ``main`` branch (the contestant committed nothing
there); ``over_limit`` with the limit the bundle or its tree passes, and nothing checked out. It
reports the record's sha256 and size in ``R`` (``report_outputs``) for the stage's manifest.
git's verdict on the content of a bundle it indexes or whose history it walks
(``bundle_verdict``: an object missing or of another kind, a deflate stream or a delta it cannot
read), every byte it was given having the bundle's digest, is the bundle's own: ``over_limit``,
the verdict in the reason. The commit point writes its bundle once and never reads it back, and
git trusts the object files it bundles from, which the contestant wrote, so a bundle the commit
point kept may give one, and so may one from outside the store; this role is the first to read
either as git reads it. Any other failure of git itself, such as a file that is no bundle, exits
nonzero, which the stage counts as infra.

With ``--names`` the role also scans for every form of each secret (``hls.substrate.secret_forms``:
its value, and the host of a value that is an http(s) URL when it identifies a machine, in any
ASCII case), whose values come
from its environment, one variable per name, in its 0600 env file. Once the bundle is cloned, and
before the limits decide anything about the tree, it streams every object of the clone through the
scan (``scan_objects``): every blob, tree, commit and tag of the bundle's whole history, as git
stores it before compression. A key the agent committed and removed later, one in a commit message
or a tag, or one in a tree past the limits is found there, though the bundle's stored bytes are
compressed and no tree of it is checked out. The role writes the count of objects each form was
found in, by the form's label, to ``FILE`` (``PackageScan``, area ``bundle``, never a value or a
host), which it reports beside ``source.json``. A bundle it never clones, one without ``main`` or
past the byte limit, gets no scan: nothing claims it was read.

A bundle's objects can inflate far past its bytes, and the scan reads them inflated. The commit
point holds an agent build's repository to the same cap, *N*, before git touches it
(``hls.agent.commit``); a bundle from outside the store, as ``hls package --bundle`` names one, is
held to it here. Before the scan, and with or without one, the role reads the kind and size git
records for every object of the clone (``object_bytes``); objects that inflate past *N* together,
or tree objects past the commit point's ``TREE_BYTES_MAX`` together, are the contestant's
(``over_limit``, the size in the reason), with nothing checked out and no scan."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
import re
import signal
import subprocess
import sys
import threading
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict

from hls.agent.commit import (ERRNO_WORDS, KIND_CHECK_FORMAT, KIND_CHECK_SEC, MESSAGE_BYTES, TREE_BYTES_MAX,
                              TREE_CACHE, TREE_LIST_SEC, FileIds, blob_sizes, git_failure, git_messages,
                              listing_problem)
from hls.store.digest import COPY_FAILED_EXIT, DIGEST_MISMATCH_EXIT, Sha256Hex, check_sha256_hex, sha256_file
from hls.store.manifest import report_outputs
from hls.substrate.secret_forms import FORMS, SecretForm, registry_forms
from hls.substrate.secret_scan import PackageScan, PackageScanArea, find
from hls.util import remove_tree

#: The variables that switch every git configuration of the host off.
CLONE_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"}
#: The branch a submission is read from.
BRANCH = "main"
#: git's verdicts on the objects of a bundle it indexes and on main (git 2.43): index-pack's on an
#: object a link names that the pack does not hold, one of another kind than its link names, a
#: link it cannot follow, an id another content has (builtin/index-pack.c), and update-ref's on
#: main's commit missing or not a commit (refs/files-backend.c).
OBJECT_VERDICTS = re.compile(r"did not receive expected object [0-9a-f]{40}"
                             r"|object [0-9a-f]{40}: expected type [a-z]+, found [a-z]+"
                             r"|object type mismatch at [0-9a-f]{40}"
                             r"|Not all child objects of [0-9a-f]{40} are reachable"
                             r"|SHA1 COLLISION FOUND WITH [0-9a-f]{40} !"
                             r"|trying to write ref '[^'\n]*' with nonexistent object [0-9a-f]{40}"
                             r"|trying to write non-commit object [0-9a-f]{40} to branch '[^'\n]*'")
#: index-pack's verdicts on the data of the pack it indexes (git 2.43, builtin/index-pack.c): an
#: object's deflate stream it cannot inflate, a delta it cannot apply or whose base lies outside
#: the pack, an object of no kind, an object's bytes ending early. git reuses an entry of the
#: contestant's packs as stored when it bundles it, checking only the checksum the pack's index
#: gives it, which the contestant wrote too, so the commit point's bundle can carry them.
DATA_VERDICTS = re.compile(r"pack has bad object at offset \d+: (?:inflate returned -?\d+"
                           r"|offset value overflow for delta base object"
                           r"|delta base offset is out of bound|unknown object type \d+"
                           r"|failed to apply delta)"
                           r"|serious inflate inconsistency"
                           r"|premature end of pack file, \d+ bytes? missing"
                           r"|early EOF")
#: git's verdicts on a bundle's history when it walks main's history whole once the bundle is
#: indexed (``git rev-list --objects``; git 2.43: builtin/rev-list.c, list-objects.c, revision.c,
#: commit.c): an object main's history names that the bundle does not hold, or one of another kind
#: than its entry names.
HISTORY_VERDICTS = re.compile(r"missing (?:blob|tree|commit|tag) object '[0-9a-f]{40}'"
                              r"|bad tree object [0-9a-f]{40}"
                              r"|entry '[^'\n]*' in tree [0-9a-f]{40} has (?:tree|blob) mode, but is not a (?:tree|blob)"
                              r"|unable to load root tree for commit [0-9a-f]{40}"
                              r"|unable to parse commit [0-9a-f]{40}"
                              r"|bad object [0-9a-f]{40}"
                              r"|Could not read [0-9a-f]{40}")
#: Node-local disk, in GiB, for git's own index and temporary files beside the pack and the tree.
GIT_SCRATCH_GIB = 4
#: The worst peak of the checkout role's anonymous memory, in GiB, the role and the git it runs
#: summed, over the shapes the commit point's bounds let through
#: (host git 2.43.7, the role on two CPUs): three 4 GiB blobs in main's history
#: (``hls.agent.commit.OBJECT_MAX_BYTES``), each with a blob of 512 MiB less 64 KiB on main stored
#: as a delta on it (``DELTA_MAX_BYTES``), 8.444 GiB. ``index-pack`` rebuilds them
#: on one thread, holding one base and the object it rebuilds; git's object scan and its checkout
#: hold the base they last rebuilt from in their cache while they read the next, two bases and the
#: object, at any setting of the cache (``core.deltaBaseCacheLimit`` 0 too).
#: Anonymous memory is what a role's limit holds:
#: the pack's file pages git maps are reclaimed first.
CHECKOUT_PEAK_GIB = 8.444
#: The checkout role's memory, in GiB: ``CHECKOUT_PEAK_GIB`` and a quarter more, rounded up, as
#: the commit point's is sized (``hls.pipeline.pipelines.COMMIT_SHAPE``), ceil(8.444 x 1.25) = 11.
#: The source_scan and package stages each give their roles at least this.
CHECKOUT_MEMORY_GIB = math.ceil(CHECKOUT_PEAK_GIB * 1.25)
#: Seconds the source_scan and package stages each allow the checkout role's clone of the bundle,
#: with the object scan in source_scan's, beside the tree's listing, check and checkout, which
#: each stage allows apart: a fixed allowance. A clone takes up to 12.3 s per GiB of large
#: objects and the object scan 4.97 s per GiB, on one thread of an AMD EPYC 9534 (host git
#: 2.43.7), so an hour holds a clone and scan of about 200 GiB of large objects before the
#: store's reads; at the commit point's object and delta-chain bounds (``hls.agent.commit``), whose
#: own terms take about 1,450 s, about 115 GiB. A clone or scan the stage's deadline ends is infra.
CHECKOUT_SEC = 3600
#: Bytes of the bundle read at a time.
READ_BYTES = 2**20
#: The most bytes of a bundle's header the role reads: its signature, its capabilities, its
#: prerequisites and its refs, a line each. A commit point's names one ref; this is far past any
#: bundle's.
HEADER_MAX_BYTES = 16 * 2**20
#: The first line of a bundle git writes: version 2, or version 3, which names its capabilities.
_SIGNATURES = (b"# v2 git bundle\n", b"# v3 git bundle\n")
#: A ref line of a bundle's header: an object id and the ref's name.
_REF_LINE = re.compile(rb"([0-9a-f]{40}) (refs/[^\n]+)")
#: The most path components a tracked path may have.
TREE_MAX_DEPTH = 64
#: The longest one name of a tracked path may be, in bytes: the filesystem's own limit, past
#: which the checkout cannot create the file.
TREE_MAX_NAME_BYTES = 255
#: The most bytes the tree's files may hold together.
TREE_MAX_BYTES = 4 * 2**30
#: The largest ``pyproject.toml`` the tree may hold, which the package stage parses: far above any
#: project's, and a bound on what the runner reads of the tree.
PYPROJECT_MAX_BYTES = 2**20
#: The attributes of the checkout's own (its repository's ``info/attributes``), which outrank any
#: ``.gitattributes`` of the tree: no conversion of what git writes, ``text`` unset and ``eol``
#: unspecified (no end-of-line conversion; with ``text`` unset git never reads ``eol``), no
#: ``ident``, no filter and no working-tree encoding. Every file is then written as its blob holds
#: it, the size the limits count, where the tree's own attributes could have git write 47 bytes for
#: every 4 of ``$Id$`` or re-encode a file in memory (git holds five times a file it re-encodes as
#: UTF-32). The brief names each (``unapplied_attributes``).
CHECKOUT_ATTRIBUTES = "* -text !eol -ident !filter !working-tree-encoding\n"


class SourceRecord(BaseModel):
    """What the checkout found."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: ``ok``; ``no_main`` when the bundle has no ``main`` branch; ``over_limit`` when the
    #: bundle, its objects inflated or the commit's tree passes a limit, the tree names an object
    #: that is not a blob as a file or symbolic link, or git refuses the bundle's content, and
    #: nothing was checked out.
    status: Literal["ok", "no_main", "over_limit"]
    #: The sha256 of the bundle the role cloned; None when the bundle was refused for its size,
    #: unread.
    bundle_sha256: Sha256Hex | None
    #: The commit read; None without ``main``.
    commit: str | None
    #: Every tracked path of the commit, sorted; empty unless ``ok``.
    files: list[str]
    #: The refs the bundle holds.
    heads: list[str]
    #: The limit the bundle, its objects or the tree passes, with its figure, the entry whose
    #: object is not a blob, or git's verdict on the bundle's content; None unless ``over_limit``.
    reason: str | None = None


def unapplied_attributes() -> list[str]:
    """The attributes of a tree's ``.gitattributes`` the checkout never applies, as its own
    attributes switch them off (``CHECKOUT_ATTRIBUTES``), in their order: what the brief names."""
    return [setting.lstrip("-!") for setting in CHECKOUT_ATTRIBUTES.split()[1:]]


def git_env(home: Path) -> dict[str, str]:
    """The whole environment git runs with: PATH from this process, a fresh HOME, and the
    variables that switch the host's configuration off."""
    return {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(home), "LANG": "C.UTF-8", **CLONE_ENV}


class GitFailed(Exception):
    """A git the role ran failed or outlived its time; the message names it. The role ends with
    it, nothing of its clone left, which the stage counts as infra."""


def bundle_verdict(returncode: int, stderr: str) -> str | None:
    """git's verdict on the content of a bundle it read, when its failure is one: on an object
    (``OBJECT_VERDICTS``), on the pack's data (``DATA_VERDICTS``) or, walking main's history, on an
    object it names (``HISTORY_VERDICTS``). None for a git killed or starved, one whose messages
    name an errno, or one that ended on anything else: each the machine's
    (``hls.agent.commit.git_failure``). The bundle's own only when every byte git read is proven
    the bundle's, which ``receive`` does."""
    if any(words in stderr for words in ERRNO_WORDS) or git_failure(returncode, stderr) is None:
        return None
    found = OBJECT_VERDICTS.search(stderr) or DATA_VERDICTS.search(stderr) or HISTORY_VERDICTS.search(stderr)
    return None if found is None else found.group(0)


class Unreadable(Exception):
    """git's verdict on the content of a bundle whose every byte the role gave git has the digest
    the bundle must have (``bundle_verdict``): the bundle's own, an invalid package
    (``over_limit``)."""

    def __init__(self, verdict: str, refs: dict[str, str]) -> None:
        super().__init__(verdict)
        #: git's verdict.
        self.verdict = verdict
        #: The refs the bundle's header names.
        self.refs = refs


def _verb(argv: list[str]) -> str:
    """The git command *argv* runs, past git's own options (``-C`` and ``-c`` with their values)."""
    at = 1
    while argv[at] in ("-C", "-c"):
        at += 2
    return argv[at]


def _end(p: subprocess.Popen) -> None:
    """End *p*'s session unless it has ended."""
    if p.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(p.pid, signal.SIGKILL)


class _Git:
    """One git of the role's, on *argv* with *env* in its own session, reading the file *stdin*
    when given: its output is the
    caller's to read as it streams (``with _Git(...) as out``), and its messages are drained as git
    writes them, the last ``MESSAGE_BYTES`` kept, so a full pipe of them never stalls it. A git
    that outlives *timeout_sec*, when it is given, has its session ended, and so does one the
    caller stops early (``stop``), past a bound of what it read; it is reaped as the block ends.
    Without *timeout_sec* git runs until the stage's deadline ends the role. ``check`` then raises
    ``GitFailed`` for a git that ran past its time, or one that failed and was not stopped."""

    def __init__(self, argv: list[str], env: dict[str, str], timeout_sec: float | None = None,
                 stdin: BinaryIO | None = None) -> None:
        #: The command and its environment.
        self.argv, self.env = argv, env
        #: What git reads; nothing when None.
        self.stdin = stdin
        #: Seconds git may take; None for no time of its own.
        self.timeout_sec = timeout_sec
        #: The last ``MESSAGE_BYTES`` of git's messages.
        self.messages = b""
        #: Whether git outlived its time and was ended.
        self.expired = False
        #: Whether the caller ended git early.
        self.stopped = False
        #: git, once started.
        self.process: subprocess.Popen | None = None
        #: The thread that drains git's messages.
        self._drain: threading.Thread | None = None
        #: The timer that ends git past its time; None without one.
        self._timer: threading.Timer | None = None

    def __enter__(self) -> BinaryIO:
        p = self.process = subprocess.Popen(self.argv, env=self.env,
                                            stdin=subprocess.DEVNULL if self.stdin is None else self.stdin,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        self._drain = threading.Thread(target=self._read_messages, args=(p.stderr,), daemon=True)
        self._drain.start()
        if self.timeout_sec is not None:
            self._timer = threading.Timer(self.timeout_sec, self._expire)
            self._timer.start()
        assert p.stdout is not None
        return p.stdout

    def _read_messages(self, stream: BinaryIO) -> None:
        while chunk := stream.read1(READ_BYTES):
            self.messages = (self.messages + chunk)[-MESSAGE_BYTES:]

    def _expire(self) -> None:
        assert self.process is not None
        self.expired = True
        _end(self.process)

    def stop(self) -> None:
        """End git now: what it wrote so far decided the read."""
        assert self.process is not None
        self.stopped = True
        _end(self.process)

    def __exit__(self, kind: type[BaseException] | None, error: BaseException | None,
                 trace: TracebackType | None) -> None:
        assert self.process is not None and self._drain is not None
        if kind is not None:
            _end(self.process)
        # A git that closed its output but never exits is ended by the timer, or by the stage's
        # deadline without one, still running here.
        self.process.wait()
        if self._timer is not None:
            self._timer.cancel()
        self._drain.join()
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                stream.close()

    def check(self) -> None:
        """Raise ``GitFailed`` for a git that outlived its time, or one that failed unless the
        caller stopped it."""
        assert self.process is not None
        if self.expired:
            raise GitFailed(f"git {_verb(self.argv)} took longer than {self.timeout_sec} s")
        if not self.stopped and self.process.returncode != 0:
            raise GitFailed(f"git {_verb(self.argv)} failed with status {self.process.returncode}: "
                            f"{self.messages.decode(errors='replace').strip()}")


def _git(argv: list[str], env: dict[str, str]) -> bytes:
    """Run git on *argv* (``_Git``), with no time of its own, and return its output: each command
    the role runs this way prints a commit id at most."""
    run = _Git(argv, env)
    with run as out:
        output = out.read()
    run.check()
    return output


class TreeEntry(NamedTuple):
    """One tracked path, as git lists it."""

    #: The path; a byte that is not UTF-8 is kept as ``os.fsdecode`` keeps it, a lone surrogate.
    path: str
    #: The bytes of its file or link target; 0 for a submodule.
    size: int
    #: Whether it is a symlink (mode 120000).
    link: bool


def tree_entries(listed: list[tuple[bytes, bytes]], sizes: list[int]) -> list[TreeEntry]:
    """The tracked paths of the records of ``git ls-tree -r -t -z`` past its directories, each its
    metadata (a mode, a kind and an object id) and its path (*listed*, in the listing's order),
    with the size git's check of the objects gives each file and link, in the same order
    (*sizes*, ``hls.agent.commit.blob_sizes``); a submodule's, whose object is never read, is 0."""
    left = iter(sizes)
    entries = []
    for meta, path in listed:
        mode, kind = meta.split(b" ")[:2]
        entries.append(TreeEntry(path.decode(errors="surrogateescape"), next(left) if kind == b"blob" else 0,
                                 mode == b"120000"))
    return entries


def listed_tree(repo: Path, env: dict[str, str]) -> list[TreeEntry] | str:
    """Every tracked path of *repo*'s HEAD as git lists it (``git ls-tree -r -t -z``), read as it
    streams, with each file's and link's size as git's check of its object reads it
    (``hls.agent.commit.blob_sizes``); or the first bound of ``hls.agent.commit.listing_problem``
    the listing passes, with git ended there, so a tree whose subtrees repeat is never listed
    whole, or the first file or link whose object is not a blob, which git's checkout could not
    write. A git that fails, or one past its time (the listing's
    ``hls.agent.commit.TREE_LIST_SEC``, the check's ``hls.agent.commit.KIND_CHECK_SEC``), raises
    ``GitFailed``."""
    listed: list[tuple[bytes, bytes]] = []
    ids = repo / ".git" / "listing.ids"
    with open(ids, "wb") as written:
        files = FileIds(written)

        def keep(meta: bytes, path: bytes) -> None:
            files(meta, path)
            listed.append((meta, path))

        run = _Git(["git", *TREE_CACHE, "-C", str(repo), "ls-tree", "-r", "-t", "-z", "HEAD"], env, TREE_LIST_SEC)
        with run as out:
            problem = listing_problem(iter(lambda: out.read1(READ_BYTES), b""), keep)
            if problem is not None:
                run.stop()
        run.check()
    if problem is not None:
        return problem
    with open(ids, "rb") as fed:
        run = _Git(["git", "-C", str(repo), "cat-file", KIND_CHECK_FORMAT], env, KIND_CHECK_SEC, stdin=fed)
        with run as out:
            try:
                sizes = blob_sizes(iter(lambda: out.read1(READ_BYTES), b""), files.paths)
            except ValueError as e:
                raise GitFailed(f"git cat-file wrote what it never writes: {e}") from None
            if isinstance(sizes, str):
                run.stop()
        run.check()
    if isinstance(sizes, str):
        return sizes
    if len(sizes) != len(files.paths):
        raise GitFailed(f"git cat-file reported {len(sizes)} of the {len(files.paths)} objects it was asked for")
    return tree_entries(listed, sizes)


#: A name git takes for its own directory on NTFS: ``.git`` or its short name ``git~1``, in any
#: case, then only dots and spaces up to the name's end, a backslash or a colon (the start of an
#: alternate data stream).
_NTFS_DOTGIT = re.compile(r"(?:\.git|git~1)[. ]*(?:[/\\:]|\Z)", re.ASCII | re.IGNORECASE)


def git_refuses(path: str) -> bool:
    """Whether git's checkout refuses to write the file *path*: git's own ``verify_path`` as it
    runs on Linux with the default configuration, ``core.protectNTFS`` on and
    ``core.protectHFS`` off, which is how the role runs git. It refuses an empty, ``.`` or
    ``..`` name, and a name NTFS would read as ``.git``, checked again after each backslash but
    a name's first character. The names git refuses only on a symlink's path never reach it: the
    limits refuse every symlink first."""
    start = 0
    for name in path.split("/"):
        if name in ("", ".", ".."):
            return True
        for at in [start] + [start + i + 1 for i, c in enumerate(name) if c == "\\" and i > 0]:
            if _NTFS_DOTGIT.match(path, at):
                return True
        start += len(name) + 1
    return False


def limit_problem(entries: list[TreeEntry]) -> str | None:
    """The first of the limits the tree of *entries* passes, with its figure, or None: those its
    listing's read does not hold it to as it streams (``listed_tree``), which bounds how many
    entries there are and how long a path is."""
    for e in entries:
        try:
            e.path.encode()
        except UnicodeEncodeError:
            return f"the tree tracks a path that is not UTF-8: {os.fsencode(e.path)[:200]!r}"
    deepest = max((e.path for e in entries), key=lambda p: p.count("/"), default="")
    if deepest.count("/") + 1 > TREE_MAX_DEPTH:
        return (f"the tree nests {deepest.count('/') + 1} levels deep, more than the {TREE_MAX_DEPTH} allowed: "
                f"{deepest[:200]}")
    widest = max((name for e in entries for name in e.path.split("/")), key=lambda n: len(n.encode()), default="")
    if len(widest.encode()) > TREE_MAX_NAME_BYTES:
        return (f"the tree tracks a name of {len(widest.encode())} bytes, longer than the {TREE_MAX_NAME_BYTES} "
                f"allowed: {widest[:200]}")
    link = next((e.path for e in entries if e.link), None)
    if link is not None:
        return f"the tree tracks a symlink, which a submission may not hold: {link[:200]!r}"
    refused = next((e.path for e in entries if git_refuses(e.path)), None)
    if refused is not None:
        return f"the tree tracks a path git refuses to check out: {refused[:200]!r}"
    total = sum(e.size for e in entries)
    if total > TREE_MAX_BYTES:
        return f"the tree holds {total} bytes, more than the {TREE_MAX_BYTES} allowed"
    return None


class _Object:
    """The next *size* bytes of *stream*, read as a stream of their own."""

    def __init__(self, stream: BinaryIO, size: int) -> None:
        self.stream, self.left = stream, size

    def read(self, n: int) -> bytes:
        data = self.stream.read(min(n, self.left))
        self.left -= len(data)
        return data


def object_bytes(repo: Path, env: dict[str, str]) -> tuple[int, int]:
    """The bytes every object of *repo*'s object store holds inflated, together, and those of its
    tree objects, as git records each one's kind and size (``git cat-file --batch-all-objects
    --batch-check``), read without inflating any (``_Git``)."""
    argv = ["git", "-C", str(repo), "cat-file", "--batch-all-objects", "--batch-check=%(objecttype) %(objectsize)",
            "--unordered"]
    total = trees = 0
    run = _Git(argv, env)
    with run as out:
        for line in out:
            kind, size = line.split()
            total += int(size)
            if kind == b"tree":
                trees += int(size)
    run.check()
    return total, trees


def scan_objects(repo: Path, env: dict[str, str], forms: Sequence[SecretForm]) -> dict[str, int]:
    """How many objects of *repo*'s object store hold each of *forms*, by its label: every blob,
    tree, commit and tag, each read whole as git stores it before compression (``git cat-file
    --batch-all-objects --batch``), which rebuilds every delta (``_Git``)."""
    argv = ["git", "-C", str(repo), "cat-file", "--batch-all-objects", "--batch", "--unordered"]
    counts: dict[str, int] = {}
    run = _Git(argv, env)
    with run as out:
        while header := out.readline():
            size = int(header.split()[2])
            obj = _Object(out, size)
            for label in find(obj, forms):
                counts[label] = counts.get(label, 0) + 1
            while obj.read(1 << 20):
                pass
            out.read(1)
    run.check()
    return counts


def _write_scan(out: Path, registry: Mapping[str, str], counts: dict[str, int]) -> None:
    """Write the record of the bundle's objects to *out*: the secrets of *registry* and the forms
    searched, the area read and each label's count."""
    record = PackageScan(secrets=sorted(registry), forms=list(FORMS), read=["bundle"], findings=[],
                         areas=[PackageScanArea(secret=label, area="bundle", count=count)
                                for label, count in sorted(counts.items())])
    tmp = out.with_name(f".{out.name}.tmp")
    tmp.write_text(record.model_dump_json(indent=2))
    os.replace(tmp, out)


def checkout_disk_gib(max_bytes: int) -> int:
    """The node-local disk, in GiB, a checkout of a bundle of at most *max_bytes* takes at once:
    its clone's pack, as large as the bundle since the bundle is not compressed, the checked-out
    tree (``TREE_MAX_BYTES``) and git's own files (``GIT_SCRATCH_GIB``). The bundle is streamed
    from where it lies, with no copy on the node."""
    return math.ceil((max_bytes + TREE_MAX_BYTES) / 2**30) + GIT_SCRATCH_GIB


class Mismatch(Exception):
    """What was read of the bundle does not have the digest it must have. Whose fault that is
    the file where it lies decides, read again (``main``)."""


def _chunks(fd: int, path: Path) -> Iterator[bytes]:
    """The bytes of the bundle *path*, open as *fd*, ``READ_BYTES`` at a time."""
    while chunk := os.read(fd, READ_BYTES):
        yield chunk


def _header(chunks: Iterator[bytes], hasher: hashlib._Hash) -> tuple[dict[str, str] | str, bytes]:
    """The refs a bundle's header names (ref to id), read from *chunks*, each hashed into
    *hasher*, and the bytes read past the header; in place of the refs, why the header is not one
    git could clone: not a bundle, past ``HEADER_MAX_BYTES``, or naming prerequisites, which no
    commit point's has."""
    held = b""
    for chunk in chunks:
        hasher.update(chunk)
        held += chunk
        if b"\n\n" in held or len(held) > HEADER_MAX_BYTES:
            break
    end = held.find(b"\n\n")
    if end < 0 or end > HEADER_MAX_BYTES:
        return "the bundle has no header git could read within its first bytes", b""
    lines = held[:end + 1].split(b"\n")[:-1]
    if not lines or lines[0] + b"\n" not in _SIGNATURES:
        return "the file is not a git bundle of version 2 or 3", b""
    refs: dict[str, str] = {}
    for line in lines[1:]:
        if line == b"@object-format=sha1":
            continue
        if line.startswith(b"-"):
            return "the bundle names prerequisite commits, which a clone cannot have", b""
        found = _REF_LINE.fullmatch(line)
        if found is None:
            return f"the bundle's header holds a line git does not write: {line[:200]!r}", b""
        refs[os.fsdecode(found.group(2))] = found.group(1).decode()
    return refs, held[end + 2:]


def _feed(stdin: BinaryIO, chunk: bytes | None) -> bool:
    """Write *chunk* to git's standard input *stdin*, or close it for None; False when git has
    stopped reading, which its exit status then says why."""
    try:
        if chunk is None:
            stdin.close()
        else:
            stdin.write(chunk)
    except BrokenPipeError:
        return False
    return True


def receive(bundle: Path, dest: Path, env: dict[str, str], expect: str) -> dict[str, str]:
    """Read *bundle* once where it lies, hashing every byte, into a fresh repository at *dest*:
    the refs its header names (ref to id), returned, and its pack, handed to git's
    ``index-pack`` as it is read, so git never opens the bundle and indexes only the bytes hashed.
    A bundle without ``main`` is hashed to its end and never indexed. Raise ``Mismatch``, with
    *dest* removed, unless the whole bundle has the sha256 *expect*: nothing of it is read before.
    Then ``main`` is set, its history checked whole and HEAD made to name it. git's verdict on the
    bundle's content there, the bundle's digest proven, raises ``Unreadable``, the bundle's own. A
    header git could not clone ends the role, and a git that fails otherwise raises ``GitFailed``;
    the stage counts both as infra, as it does the role still running at its deadline.
    ``index-pack`` rebuilds the bundle's deltas on
    one thread: by default it runs one for every two CPUs the node has online, whatever the role's,
    and each holds a delta's base whole, so the role's memory would grow with the node's CPUs and
    the bases a contestant packed."""
    hasher = hashlib.sha256()
    fd = os.open(bundle, os.O_RDONLY)
    try:
        chunks = _chunks(fd, bundle)
        refs, rest = _header(chunks, hasher)
        main = f"refs/heads/{BRANCH}"
        if isinstance(refs, str) or main not in refs:
            # Nothing to index; the digest still decides first, so a file that is not the bundle
            # is refused whatever it holds.
            for chunk in chunks:
                hasher.update(chunk)
            if hasher.hexdigest() != expect:
                raise Mismatch
            if isinstance(refs, str):
                sys.exit(refs)
            return refs
        _git(["git", "init", "--quiet", "--template=", str(dest)], env)
        errors = dest / ".git" / "index-pack.err"
        with open(errors, "wb") as err:
            p = subprocess.Popen(["git", "-C", str(dest), "index-pack", "--stdin", "--threads=1"], env=env,
                                 stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
            assert p.stdin is not None
            try:
                feeding = _feed(p.stdin, rest)
                for chunk in chunks:
                    hasher.update(chunk)
                    feeding = feeding and _feed(p.stdin, chunk)
                _feed(p.stdin, None)
                indexed = p.wait()
            finally:
                if p.poll() is None:
                    p.kill()
                    p.wait()
        if hasher.hexdigest() != expect:
            remove_tree(dest)
            raise Mismatch
        if indexed != 0:
            message = git_messages(errors).strip()
            verdict = bundle_verdict(indexed, message)
            if verdict is not None:
                raise Unreadable(verdict, refs)
            raise GitFailed(f"git index-pack failed with status {indexed}: {message}")
        errors.unlink()
    finally:
        os.close(fd)
    for argv in (["git", "-C", str(dest), "update-ref", main, refs[main]],
                 ["git", "-C", str(dest), "rev-list", "--objects", "--quiet", main]):
        verdict = _verdict_of(argv, env)
        if verdict is not None:
            raise Unreadable(verdict, refs)
    _git(["git", "-C", str(dest), "symbolic-ref", "HEAD", main], env)
    return refs


def _verdict_of(argv: list[str], env: dict[str, str]) -> str | None:
    """Run git on *argv* as ``_git`` does, its output discarded: git's verdict on the content of
    the bundle when it fails with one (``bundle_verdict``), else None once it
    succeeds; any other failure raises ``GitFailed``."""
    run = _Git(argv, env)
    with run as out:
        out.read()
    assert run.process is not None
    if run.process.returncode != 0:
        verdict = bundle_verdict(run.process.returncode, run.messages.decode(errors="replace"))
        if verdict is not None:
            return verdict
    run.check()
    return None


def checkout(bundle: Path, dest: Path, home: Path, *, bundle_sha256: str, max_bytes: int,
             scan: tuple[Mapping[str, str], Path] | None = None) -> SourceRecord:
    """Clone *bundle*'s ``main`` into *dest*, hold its objects inflated to *max_bytes* and then its
    tree to the limits, check it out and drop its ``.git``. A bundle past a limit leaves nothing at
    *dest*. *bundle_sha256* is the bundle's digest, which the record carries. With *scan*, the
    secrets to find (name to value), each in every form, and where to write the record, every
    object of the clone is scanned once their inflated size is within its limit and before the
    tree's limits decide (``scan_objects``). The bundle is read once where it lies (``receive``);
    raise ``Mismatch`` when it does not have *bundle_sha256*. A git command that fails, or the
    listing or the check of the tree's objects past its own time (``listed_tree``), ends the role,
    with nothing of the clone left, which the stage counts as infra; every other git command runs
    until the stage's deadline ends the role."""
    try:
        return _checked_out(bundle, dest, git_env(home), bundle_sha256=bundle_sha256, max_bytes=max_bytes,
                            scan=scan)
    except GitFailed as e:
        remove_tree(dest, ignore_errors=True)
        sys.exit(str(e))


def _checked_out(bundle: Path, dest: Path, env: dict[str, str], *, bundle_sha256: str, max_bytes: int,
                 scan: tuple[Mapping[str, str], Path] | None) -> SourceRecord:
    """``checkout``'s steps, git running with *env*."""
    try:
        refs = receive(bundle, dest, env, bundle_sha256)
    except Unreadable as e:
        remove_tree(dest)
        return SourceRecord(status="over_limit", bundle_sha256=bundle_sha256, commit=e.refs.get(f"refs/heads/{BRANCH}"),
                            files=[], heads=sorted(e.refs),
                            reason=f"the bundle does not read back as git reads it: {e.verdict}")
    heads = sorted(refs)
    if f"refs/heads/{BRANCH}" not in refs:
        return SourceRecord(status="no_main", bundle_sha256=bundle_sha256, commit=None, files=[], heads=heads)
    commit = _git(["git", "-C", str(dest), "rev-parse", "HEAD"], env).decode().strip()

    def over(reason: str) -> SourceRecord:
        remove_tree(dest)
        return SourceRecord(status="over_limit", bundle_sha256=bundle_sha256, commit=commit, files=[], heads=heads,
                            reason=reason)

    inflated, trees = object_bytes(dest, env)
    if inflated > max_bytes:
        return over(f"the bundle's objects inflate to {inflated} bytes, more than the {max_bytes} bytes the scan "
                    "reads")
    if trees > TREE_BYTES_MAX:
        return over(f"the bundle's tree objects hold {trees} bytes together, more than the {TREE_BYTES_MAX} allowed")
    if scan is not None:
        registry, out = scan
        _write_scan(out, registry, scan_objects(dest, env, registry_forms(registry)))
    entries = listed_tree(dest, env)
    if isinstance(entries, str):
        return over(entries)
    problem = limit_problem(entries)
    if problem is not None:
        return over(problem)
    (dest / ".git" / "info").mkdir(exist_ok=True)
    (dest / ".git" / "info" / "attributes").write_text(CHECKOUT_ATTRIBUTES)
    # git's checkout reads a directory's tree object again wherever the directory appears.
    _git(["git", *TREE_CACHE, "-C", str(dest), "reset", "--quiet", "--hard", "HEAD"], env)
    remove_tree(dest / ".git")
    return SourceRecord(status="ok", bundle_sha256=bundle_sha256, commit=commit,
                        files=sorted(e.path for e in entries), heads=heads)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hls.package.checkout")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--outputs-out", type=Path, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--expect", type=check_sha256_hex, required=True)
    p.add_argument("--names")
    p.add_argument("--scan-out", type=Path)
    p.add_argument("--max-bytes", type=int, required=True)
    args = p.parse_args(sys.argv[1:] if argv is None else argv)
    if args.max_bytes < 1:
        p.error("--max-bytes must be positive")
    if (args.names is None) != (args.scan_out is None):
        p.error("--names and --scan-out go together")
    size = os.stat(args.bundle).st_size
    if size != args.size:
        print(f"refused: file {args.bundle} holds {size} bytes, not the {args.size} its record names", file=sys.stderr)
        return DIGEST_MISMATCH_EXIT
    if size > args.max_bytes:
        record = SourceRecord(status="over_limit", bundle_sha256=None, commit=None, files=[], heads=[],
                              reason=f"the source bundle holds {size} bytes, more than the {args.max_bytes} allowed")
    else:
        home = args.dest.parent / f".{args.dest.name}-home"
        home.mkdir(parents=True, exist_ok=True)
        scan = None
        if args.names is not None:
            scan = ({name: os.environ[name] for name in args.names.split(",")}, args.scan_out)
        try:
            record = checkout(args.bundle, args.dest, home, bundle_sha256=args.expect, max_bytes=args.max_bytes,
                              scan=scan)
        except Mismatch:
            # The file itself decides whose: a store file that is not the recorded bundle, or a
            # read gone wrong on the way, which a retry reads again.
            lies = sha256_file(args.bundle)[0]
            if lies == args.expect:
                print(f"read failed: what was read of file {args.bundle} differs from it", file=sys.stderr)
                return COPY_FAILED_EXIT
            print(f"refused: file {args.bundle} has sha256 {lies}, not {args.expect}", file=sys.stderr)
            return DIGEST_MISMATCH_EXIT
        finally:
            remove_tree(home, ignore_errors=True)
    tmp = args.out.with_name(f".{args.out.name}.tmp")
    tmp.write_text(record.model_dump_json(indent=2))
    os.replace(tmp, args.out)
    scanned = [args.scan_out] if args.scan_out is not None and args.scan_out.exists() else []
    report_outputs(args.outputs_out, args.out.parent, args.out, *scanned)
    return 0


if __name__ == "__main__":
    sys.exit(main())
