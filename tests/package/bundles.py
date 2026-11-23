"""Git bundles for the package tests, made the way a build's commit point makes one: a
repository whose ``main`` holds the submission, bundled with ``git bundle create``."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import zlib
from pathlib import Path
from typing import Any

from hls.util import remove_tree

#: A minimal valid submission: a pyproject that names the project and one dependency.
PYPROJECT = """[project]
name = "subm"
version = "0.1"
dependencies = ["httpx>=0.27"]
"""
#: Its server.
SERVER = "print('serving')\n"
GOOD = {"pyproject.toml": PYPROJECT, "server.py": SERVER}
#: The symlinks a submission might hold, each as files besides ``GOOD``, links (path to target)
#: and the first link git lists: a chain whose links each lead inside the tree as
#: text while ``t`` resolves through ``x/y/s`` to ``secret`` beside the tree, a plain link to a
#: file of the tree, a link to a directory of it, and a link that leads nowhere.
SYMLINKS = {
    "a-chain-that-leaves-through-a-link": ({"a/f": "x"}, {"x/y/s": "../../a", "t": "x/y/s/../../secret"}, "t"),
    "a-link-in-the-tree": ({}, {"inside": "server.py"}, "inside"),
    "a-link-to-a-directory": ({"pkg/mod.py": ""}, {"lib": "pkg"}, "lib"),
    "a-dangling-link": ({}, {"dangling": "missing/file"}, "dangling"),
}

# Automatic maintenance is off: since git 2.47 a commit of many objects starts it detached, and
# it would still be writing into the repository while the helper removes it.
_ENV = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_AUTHOR_NAME": "a",
        "GIT_AUTHOR_EMAIL": "a@x", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@x",
        "GIT_CONFIG_COUNT": "2", "GIT_CONFIG_KEY_0": "maintenance.auto", "GIT_CONFIG_VALUE_0": "false",
        "GIT_CONFIG_KEY_1": "gc.auto", "GIT_CONFIG_VALUE_1": "0"}


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, check=True, capture_output=True,
                          env={"PATH": os.environ["PATH"], "HOME": str(repo), **_ENV}).stdout.decode().strip()


def _mkdirs(directory: Path) -> None:
    """Create *directory* and its missing parents one level at a time: ``Path.mkdir`` with
    ``parents`` recurses once a level and fails on a tree deeper than the interpreter's limit."""
    missing = []
    while not directory.exists():
        missing.append(directory)
        directory = directory.parent
    for path in reversed(missing):
        path.mkdir()


def make_bundle(root: Path, files: dict[str, str], *, branch: str = "main",
                links: dict[str, str] | None = None, name: str = "source.bundle") -> Path:
    """A bundle of one commit on *branch* holding *files* and the symlinks *links* (path to
    target), written to ``root/name``. The repository it was made in is removed, without
    recursion, so no tree deeper than the interpreter's limit is left for pytest's own cleanup."""
    repo = root / f"{name}.repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", branch)
    for path, text in files.items():
        _mkdirs((repo / path).parent)
        (repo / path).write_text(text)
    for path, target in (links or {}).items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, repo / path)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "submission")
    bundle = root / name
    _git(repo, "bundle", "create", str(bundle), branch)
    remove_tree(repo)
    return bundle


def stored(store_root: Path, key: str, bundle: Path) -> dict[str, Any]:
    """*bundle* copied to *key* in the store at *store_root*, as ``hls package`` copies a named
    bundle into its instance, and the plan's record of the copy (``StoredBundle``)."""
    copy = store_root / key
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle, copy)
    return {"key": key, "size": copy.stat().st_size}


def make_tree_bundle(root: Path, paths: dict[str, bool], *, name: str = "source.bundle") -> Path:
    """A bundle of one commit on ``main`` whose tree git's plumbing (``mktree``) writes entry by
    entry, so it may hold names ``git add`` refuses. Each of *paths* holds one line, as a symlink
    when its value is true; a name is written as ``os.fsencode`` gives it, so a lone surrogate
    stands for a byte that is not UTF-8."""
    repo = root / f"{name}.repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    blob = _git(repo, "hash-object", "-w", "--stdin", stdin=b"x\n")
    top: dict = {}
    for path, link in paths.items():
        *dirs, leaf = path.split("/")
        folder = top
        for part in dirs:
            folder = folder.setdefault(part, {})
        folder[leaf] = "120000" if link else "100644"

    def mktree(folder: dict) -> str:
        lines = [f"040000 tree {mktree(v)}" if isinstance(v, dict) else f"{v} blob {blob}" for v in folder.values()]
        return _git(repo, "mktree", "-z", stdin=b"".join(
            line.encode() + b"\t" + os.fsencode(n) + b"\0" for line, n in zip(lines, folder)))

    commit = _git(repo, "commit-tree", mktree(top), "-m", "submission")
    _git(repo, "update-ref", "refs/heads/main", commit)
    bundle = root / name
    _git(repo, "bundle", "create", str(bundle), "main")
    remove_tree(repo)
    return bundle


def make_history_bundle(root: Path, commits: list[tuple[dict[str, str | None], str]], *,
                        tag: tuple[str, str] | None = None, links: dict[str, str] | None = None,
                        name: str = "source.bundle") -> Path:
    """A bundle of ``main`` made of *commits*, each the files it writes (path to text, None to
    remove) and its message, the last commit also holding the symlinks *links*, with an annotated
    tag *tag* (its name and message) on the last commit when given, written to ``root/name``."""
    repo = root / f"{name}.repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    for i, (files, message) in enumerate(commits):
        for path, text in files.items():
            if text is None:
                (repo / path).unlink()
                continue
            _mkdirs((repo / path).parent)
            (repo / path).write_text(text)
        if i == len(commits) - 1:
            for path, target in (links or {}).items():
                os.symlink(target, repo / path)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    refs = ["main"]
    if tag is not None:
        _git(repo, "tag", "-a", tag[0], "-m", tag[1])
        refs.append(tag[0])
    bundle = root / name
    _git(repo, "bundle", "create", str(bundle), *refs)
    remove_tree(repo)
    return bundle


def make_stored_bundle(root: Path, files: dict[str, bytes], *, name: str = "source.bundle") -> Path:
    """A bundle of one commit on ``main`` whose tree holds each of *files* (a name to its bytes)
    at its top, each blob written as given (``hash-object --no-filters``), whatever attributes
    the tree sets: so a tree can hold a file its own ``.gitattributes`` would have git convert."""
    repo = root / f"{name}.repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    entries = b""
    for path, data in sorted(files.items()):
        blob = _git(repo, "hash-object", "-w", "--no-filters", "--stdin", stdin=data)
        entries += f"100644 blob {blob}\t{path}".encode() + b"\0"
    commit = _git(repo, "commit-tree", _git(repo, "mktree", "-z", stdin=entries), "-m", "submission")
    _git(repo, "update-ref", "refs/heads/main", commit)
    bundle = root / name
    _git(repo, "bundle", "create", str(bundle), "main")
    remove_tree(repo)
    return bundle


def repeated_tree(repo: Path, levels: int, width: int, *, files: bool = True) -> str:
    """Commit, as ``main`` of the repository *repo*, a tree that repeats its subtrees: *levels*
    trees, each of *width* directories that all name the one below, over a bottom tree of *width*
    files of one line (or, without *files*, the empty tree, which git's plumbing writes and no
    checkout creates). It lists ``width ** (levels + 1)`` paths (none without *files*) in
    ``width + width ** 2 + ... + width ** levels`` directories, from ``levels + 2`` objects: a git
    bomb. Return the commit."""
    blob = _git(repo, "hash-object", "-w", "--stdin", stdin=b"x\n")
    entries = b"".join(f"100644 blob {blob}\tf{i}".encode() + b"\0" for i in range(width)) if files else b""
    tree = _git(repo, "mktree", "-z", stdin=entries)
    for _ in range(levels):
        tree = _git(repo, "mktree", "-z", stdin=b"".join(f"040000 tree {tree}\td{i}".encode() + b"\0"
                                                         for i in range(width)))
    commit = _git(repo, "commit-tree", tree, "-m", "submission")
    _git(repo, "update-ref", "refs/heads/main", commit)
    return commit


def make_repeated_bundle(root: Path, levels: int, width: int, *, files: bool = True,
                         name: str = "source.bundle") -> Path:
    """A bundle of ``main`` holding one commit of ``repeated_tree``, written to ``root/name``."""
    repo = root / f"{name}.repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    repeated_tree(repo, levels, width, files=files)
    bundle = root / name
    _git(repo, "bundle", "create", str(bundle), "main")
    remove_tree(repo)
    return bundle


#: The objects an entry of ``kind_tree`` may name: made for it and reached through it alone, or
#: (``main-``) one main's history also holds as its own kind; ``missing`` is no object at all.
KIND_TARGETS = ("blob", "tree", "commit", "tag", "missing", "main-blob", "main-tree", "main-commit")


def kind_tree(repo: Path, mode: str, target: str) -> str:
    """Commit, as ``main`` of a repository made at *repo*, a commit over ``GOOD`` whose child's
    tree holds ``GOOD`` and ``y.py``, an entry of *mode* that names an object of *target*
    (``KIND_TARGETS``): what git's plumbing writes and no porcelain does, the tree's bytes hashed as
    given (``hash-object --literally``). Return the commit."""
    _git(repo, "init", "-q", "-b", "main")
    blobs = {path: _git(repo, "hash-object", "-w", "--stdin", stdin=text.encode()) for path, text in GOOD.items()}
    good = _git(repo, "mktree", "-z", stdin=b"".join(f"100644 blob {o}\t{p}".encode() + b"\0"
                                                     for p, o in sorted(blobs.items())))
    first = _git(repo, "commit-tree", good, "-m", "first")
    lone = _git(repo, "hash-object", "-w", "--stdin", stdin=b"lone\n")
    lone_tree = _git(repo, "mktree", "-z", stdin=f"100644 blob {lone}\tg".encode() + b"\0")
    tag = f"object {lone}\ntype blob\ntag t\ntagger a <a@x> 1 +0000\n\nt\n".encode()
    names = {"blob": lone, "tree": lone_tree, "commit": _git(repo, "commit-tree", lone_tree, "-m", "lone"),
             "tag": _git(repo, "mktag", stdin=tag), "missing": "1" * 40, "main-blob": blobs["server.py"],
             "main-tree": good, "main-commit": first}
    entries = sorted([*((p.encode(), b"100644", o) for p, o in blobs.items()), (b"y.py", mode.encode(), names[target])])
    raw = b"".join(m + b" " + p + b"\0" + bytes.fromhex(o) for p, m, o in entries)
    tree = _git(repo, "hash-object", "-t", "tree", "--literally", "-w", "--stdin", stdin=raw)
    commit = _git(repo, "commit-tree", tree, "-p", first, "-m", "submission")
    _git(repo, "update-ref", "refs/heads/main", commit)
    return commit


def pack_all(repo: Path) -> Path:
    """Pack every object of the repository *repo* into one pack by its id, with no reverse index,
    drop the loose copies, and return the pack's index."""
    objects = repo / ".git" / "objects"
    loose = [f"{d.name}{f.name}" for d in objects.iterdir() if len(d.name) == 2 for f in d.iterdir()]
    _git(repo, "pack-objects", "-q", str(objects / "pack" / "pack"), stdin="".join(f"{o}\n" for o in loose).encode())
    _git(repo, "prune-packed")
    for rev in (objects / "pack").glob("*.rev"):
        rev.unlink()
    [index] = (objects / "pack").glob("*.idx")
    return index


def _write_index(index: Path, data: bytearray) -> None:
    data[-20:] = hashlib.sha1(bytes(data[:-20])).digest()
    index.chmod(0o644)
    index.write_bytes(bytes(data))


def _index_ids(data: bytearray) -> tuple[int, list[bytes]]:
    """Where a version 2 pack index's ids start, and its ids."""
    count = int.from_bytes(data[8 + 255 * 4:8 + 256 * 4], "big")
    at = 8 + 256 * 4
    return at, [bytes(data[at + 20 * i:at + 20 * (i + 1)]) for i in range(count)]


def swap_in_index(index: Path, a: str, b: str) -> None:
    """Swap the objects *a* and *b* in the pack index *index*: each named at the other's offset,
    with the other's checksum. git then reads b's bytes wherever a is named, and a bundle git makes
    carries them in a's place."""
    data = bytearray(index.read_bytes())
    ids_at, ids = _index_ids(data)
    at_a, at_b = ids.index(bytes.fromhex(a)), ids.index(bytes.fromhex(b))
    for table in (ids_at + 20 * len(ids), ids_at + 24 * len(ids)):
        here, there = slice(table + 4 * at_a, table + 4 * (at_a + 1)), slice(table + 4 * at_b, table + 4 * (at_b + 1))
        data[here], data[there] = data[there], data[here]
    _write_index(index, data)


def break_stream(index: Path, blob: str) -> None:
    """Break, halfway, the deflate stream of the object *blob* in the pack beside the index
    *index*, and give the broken bytes their own checksum in the index and the pack its own: git
    reuses the entry as stored when it bundles it, checking only that checksum, and only inflating
    it finds it broken."""
    pack = index.with_suffix(".pack")
    data, packed = bytearray(index.read_bytes()), bytearray(pack.read_bytes())
    ids_at, ids = _index_ids(data)
    count = len(ids)
    offsets = [int.from_bytes(data[ids_at + 24 * count + 4 * i:ids_at + 24 * count + 4 * (i + 1)], "big")
               for i in range(count)]
    at = ids.index(bytes.fromhex(blob))
    start = offsets[at]
    end = min((o for o in offsets if o > start), default=len(packed) - 20)
    assert end - start > 1000, "the object's stream is long enough to break halfway"
    packed[(start + end) // 2] ^= 0xFF
    packed[-20:] = hashlib.sha1(bytes(packed[:-20])).digest()
    pack.chmod(0o644)
    pack.write_bytes(bytes(packed))
    crc = ids_at + 20 * count + 4 * at
    data[crc:crc + 4] = zlib.crc32(bytes(packed[start:end])).to_bytes(4, "big")
    data[-40:-20] = packed[-20:]
    _write_index(index, data)


def _entry_data(packed: bytearray, start: int) -> tuple[int, int]:
    """Where the data of the pack entry at *start* begins, past its header and, for a delta, its
    base's name or offset; and the entry's kind, as git numbers it."""
    kind, at = (packed[start] >> 4) & 7, start
    while packed[at] & 0x80:
        at += 1
    at += 1
    if kind == 7:
        at += 20
    elif kind == 6:
        while packed[at] & 0x80:
            at += 1
        at += 1
    return at, kind


def break_delta(index: Path, oid: str) -> None:
    """Make the first instruction of the delta the object *oid* is stored as, in the pack beside
    the index *index*, written with no compression, a copy from far past the end of its base. The
    entry stays a whole deflate stream, its adler32, its checksum in the index and the pack's own
    recomputed: git reuses it as stored when it bundles it, and only applying the delta finds it
    broken."""
    pack = index.with_suffix(".pack")
    data, packed = bytearray(index.read_bytes()), bytearray(pack.read_bytes())
    ids_at, ids = _index_ids(data)
    count = len(ids)
    offsets = [int.from_bytes(data[ids_at + 24 * count + 4 * i:ids_at + 24 * count + 4 * (i + 1)], "big")
               for i in range(count)]
    at = ids.index(bytes.fromhex(oid))
    start = offsets[at]
    end = min((o for o in offsets if o > start), default=len(packed) - 20)
    begins, kind = _entry_data(packed, start)
    assert kind in (6, 7), "the object is stored as a delta"
    delta = zlib.decompress(bytes(packed[begins:end]))
    # A stream with no compression holds the delta as it is, then its adler32.
    found = packed.find(delta, begins, end)
    assert found > begins and found + len(delta) <= end - 4, "the delta is stored uncompressed"
    op = 0
    for _ in range(2):  # the base's size and the result's
        while delta[op] & 0x80:
            op += 1
        op += 1
    assert delta[op] & 0x80, "the delta starts with a copy"
    broken = bytearray(delta)
    # A copy of 255 bytes from offset 0xff000000, its offset's top byte and its size's low byte.
    broken[op:op + 3] = bytes([0x80 | 0x08 | 0x10, 0xFF, 0xFF])
    packed[found:found + len(delta)] = broken
    packed[end - 4:end] = zlib.adler32(bytes(broken)).to_bytes(4, "big")
    packed[-20:] = hashlib.sha1(bytes(packed[:-20])).digest()
    pack.chmod(0o644)
    pack.write_bytes(bytes(packed))
    crc = ids_at + 20 * count + 4 * at
    data[crc:crc + 4] = zlib.crc32(bytes(packed[start:end])).to_bytes(4, "big")
    data[-40:-20] = packed[-20:]
    _write_index(index, data)


#: A file long enough that its deflate stream can be broken halfway (``break_stream``).
LONG_TEXT = "".join(f"line {n} of a file long enough to deflate in many blocks\n" for n in range(4000))
#: The ways ``lying_repo`` makes a repository's object store lie, git's bundle of its main
#: carrying the lie, each with git's verdict on that bundle when the package's checkout reads it
#: (``hls.package.checkout.bundle_verdict``), as a pattern.
LIES = {
    "broken": r"pack has bad object at offset \d+: inflate returned -?\d+",
    "swapped": r"missing blob object '[0-9a-f]{40}'",
    "loose": r"missing blob object '[0-9a-f]{40}'",
    "subtree": r"bad tree object [0-9a-f]{40}",
    "parent": r"Could not read [0-9a-f]{40}",
    "delta": r"pack has bad object at offset \d+: failed to apply delta",
}


def lying_repo(repo: Path, lie: str) -> None:
    """Make, at *repo*, a repository whose object store lies as *lie* names (``LIES``), git's
    bundle of its main carrying the lie, since git trusts the object files it bundles from:
    ``broken``, main's ``data.txt`` blob's deflate stream broken halfway (``break_stream``);
    ``swapped``, that blob swapped in the pack's index with a blob nothing reaches
    (``swap_in_index``), so git reads the other's bytes in its place; ``loose``, the same with
    loose objects, that blob's file holding the other's; ``subtree``, main's ``pkg`` tree swapped
    in the pack's index with a tree nothing reaches; ``parent``, main's parent commit swapped in
    the pack's index with a commit nothing reaches; ``delta``, ``data.txt`` edited in a second
    commit and packed with no compression as a delta on its other version, whose first instruction
    is broken (``break_delta``)."""
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    for path, text in {**GOOD, "data.txt": LONG_TEXT, "pkg/mod.py": "x = 1\n"}.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "submission")
    if lie in ("parent", "delta"):
        first = _git(repo, "rev-parse", "HEAD")
        (repo / "data.txt").write_text(LONG_TEXT.replace("line 2000 of", "LINE 2000 OF"))
        _git(repo, "commit", "-q", "-am", "more")
    if lie == "loose":
        blob = _git(repo, "rev-parse", "HEAD:data.txt")
        other = _git(repo, "hash-object", "-w", "--stdin", stdin=b"a blob nothing reaches\n")
        objects = repo / ".git" / "objects"
        (objects / blob[:2] / blob[2:]).chmod(0o644)
        (objects / blob[:2] / blob[2:]).write_bytes((objects / other[:2] / other[2:]).read_bytes())
        return
    if lie == "delta":
        objects = repo / ".git" / "objects"
        loose = [f"{d.name}{f.name}" for d in objects.iterdir() if len(d.name) == 2 for f in d.iterdir()]
        _git(repo, "-c", "pack.compression=0", "pack-objects", "-q", "--window=10", str(objects / "pack" / "pack"),
             stdin="".join(f"{o}\n" for o in loose).encode())
        _git(repo, "prune-packed")
        for rev in (objects / "pack").glob("*.rev"):
            rev.unlink()
        [index] = (objects / "pack").glob("*.idx")
        versions = {_git(repo, "rev-parse", f"{rev}:data.txt") for rev in ("HEAD", "HEAD~1")}
        stored = _git(repo, "verify-pack", "-v", str(index)).splitlines()
        [delta] = [line.split()[0] for line in stored if line.split()[0] in versions and len(line.split()) > 5]
        break_delta(index, delta)
        return
    if lie == "broken":
        break_stream(pack_all(repo), _git(repo, "rev-parse", "HEAD:data.txt"))
        return
    if lie == "swapped":
        named = _git(repo, "rev-parse", "HEAD:data.txt")
        other = _git(repo, "hash-object", "-w", "--stdin", stdin=b"a blob nothing reaches\n")
    elif lie == "subtree":
        named = _git(repo, "rev-parse", "HEAD:pkg")
        blob = _git(repo, "hash-object", "-w", "--stdin", stdin=b"y = 2\n")
        other = _git(repo, "mktree", stdin=f"100644 blob {blob}\tother.py\n".encode())
    else:
        named = first
        other = _git(repo, "commit-tree", _git(repo, "rev-parse", "HEAD^{tree}"), "-m", "nothing reaches it")
    swap_in_index(pack_all(repo), named, other)


#: Each bundle ``make_lying_bundle`` makes, with git's verdict on it as the package's checkout
#: reads it: one git made from each store ``LIES`` names, and ``main``, whose header names a commit
#: its pack does not hold.
BUNDLE_LIES = {**LIES, "main": r"trying to write ref 'refs/heads/main' with nonexistent object 1{40}"}


def make_lying_bundle(root: Path, lie: str, *, name: str = "source.bundle") -> Path:
    """A bundle of ``main``, as ``git bundle create`` with no delta search makes it from a
    repository whose object store lies (``lying_repo``), or, for ``main``, a bundle of a good
    repository whose header names as main a commit its pack does not hold: what a bundle from
    outside the store may be. Written to ``root/name``."""
    repo = root / f"{name}.repo"
    bundle = root / name
    if lie == "main":
        repo.mkdir(parents=True)
        _git(repo, "init", "-q", "-b", "main")
        for path, text in GOOD.items():
            (repo / path).write_text(text)
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", "submission")
        _git(repo, "bundle", "create", str(bundle), "main")
        held = _git(repo, "rev-parse", "main")
        bundle.write_bytes(bundle.read_bytes().replace(held.encode(), b"1" * 40, 1))
    else:
        lying_repo(repo, lie)
        _git(repo, "-c", "pack.window=0", "bundle", "create", str(bundle), "main")
    remove_tree(repo)
    return bundle


def make_kind_bundle(root: Path, mode: str, target: str, *, name: str = "source.bundle") -> Path:
    """A bundle of ``main`` holding ``kind_tree``'s commits, made by ``git bundle create`` with no
    commit point's check, as a bundle from outside the store may be, written to ``root/name``."""
    repo = root / f"{name}.repo"
    repo.mkdir(parents=True)
    kind_tree(repo, mode, target)
    bundle = root / name
    _git(repo, "bundle", "create", str(bundle), "main")
    remove_tree(repo)
    return bundle


#: Each bundle the package's checkout refuses as the submission's on its content, with the reason
#: its record gives, as a pattern: git's verdict on one that does not read back (``BUNDLE_LIES``),
#: and ``kind``, a tree naming a tree as a file (``make_kind_bundle``), which git's check of the
#: objects the tree names finds.
REFUSED = {**{lie: f"the bundle does not read back as git reads it: {verdict}" for lie, verdict in BUNDLE_LIES.items()},
           "kind": "main's tree names y\\.py as a file or symbolic link, but its object is a tree"}


def make_refused_bundle(root: Path, refused: str) -> Path:
    """The bundle ``REFUSED`` names as *refused*, written to ``root/source.bundle``."""
    return make_kind_bundle(root, "100644", "tree") if refused == "kind" else make_lying_bundle(root, refused)
