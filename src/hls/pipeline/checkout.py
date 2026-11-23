"""The code an instance runs: the checkout's hls and harbor commits and a digest of any
uncommitted change.

The orchestrator runs from its checkout on the shared filesystem, and every allocation's
runner runs from the same checkout. The plan records this state when it is made. The runner
compares it again before each stage and refuses to run a stage from a checkout that changed.
A checkout whose index hides a tracked file from ``git status``
(assume-unchanged, skip-worktree) has no state to record, and is refused.

Harness roles see the checkout, so a copy of a secrets file inside it would reach them.
``secrets_copies`` finds every such copy, ignored by git or not."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from hls.store.digest import Sha256Hex, sha256_file

#: Seconds one read-only git command in the checkout may take: the reading of its state here,
#: the runner's check before each stage, and the planner's checks of the files a scored instance
#: loads (``hls.pipeline.preflight``). A chosen bound and not a fit: far beyond what
#: ``git status`` and ``git ls-tree`` take, so only a git that hangs is cut.
READ_GIT_TIMEOUT_SEC = 300


class CodeState(BaseModel):
    """A checkout's commits and uncommitted changes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Absolute path of the hls checkout.
    checkout: Path
    #: Its HEAD commit.
    hls_commit: str
    #: HEAD commit of its harbor submodule.
    harbor_commit: str
    #: Digest of the working tree's content at every path that differs from HEAD or is
    #: untracked and not ignored, in either repository; None when both are clean. The
    #: content hashing (file bytes, executable bit, symlink target, deletion) does not depend
    #: on git's configuration or version. The set of paths follows git's own view of the
    #: repository, so ``core.fileMode`` and the ignore rules apply.
    dirty_diff_sha256: Sha256Hex | None


def _git(repo: Path, *args: str) -> bytes:
    """Run a read-only git command in *repo*, within ``READ_GIT_TIMEOUT_SEC``.
    ``--no-optional-locks`` keeps ``git status`` from taking the index lock to refresh the index,
    so the checkout is not written."""
    return subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, check=True,
                          timeout=READ_GIT_TIMEOUT_SEC).stdout


def _paths(repo: Path) -> set[bytes]:
    """Paths of *repo* that differ from HEAD, in the index or the working tree, or are
    untracked and not ignored, as ``git status`` lists them. Submodules are left out; harbor
    is recorded separately. Renames are reported as a deletion and an addition, so every
    record is one path. A path staged and then reverted in the working tree is still
    listed and hashed by its content."""
    status = _git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=all",
                  "--no-renames")
    # Each record is two status letters, a space and the path.
    return {record[3:] for record in status.split(b"\0") if record}


def _entry(repo: Path, rel: bytes) -> bytes:
    """What *rel* holds in the working tree now: ``path NUL kind NUL value NUL``."""
    path = repo / os.fsdecode(rel)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        kind, value = b"deleted", b""
    else:
        if stat.S_ISLNK(mode):
            kind, value = b"symlink", os.fsencode(os.readlink(path))
        elif stat.S_ISREG(mode):
            kind = b"file"
            value = (b"executable " if mode & stat.S_IXUSR else b"plain ") + sha256_file(path)[0].encode()
        else:
            kind, value = b"other", b""
    return rel + b"\0" + kind + b"\0" + value + b"\0"


def _entries(repo: Path, prefix: bytes) -> list[bytes]:
    return [prefix + _entry(repo, rel) for rel in _paths(repo)]


def _check_nothing_hidden(repo: Path) -> None:
    """Raise ValueError when a tracked file of *repo* is marked assume-unchanged or
    skip-worktree: ``git ls-files -v`` tags it in lowercase or with ``S``. ``git status``
    passes over such a file whatever it holds, so ``_paths`` would too, and its changes would
    reach no dirty digest."""
    listed = _git(repo, "ls-files", "-v", "-z")
    hidden = [record[2:] for record in listed.split(b"\0") if record[:1].islower() or record[:1] == b"S"]
    if hidden:
        raise ValueError(f"{repo}: {len(hidden)} tracked files are marked assume-unchanged or skip-worktree, "
                         f"{os.fsdecode(hidden[0])} first, so git status and the dirty digest pass over their changes; "
                         "clear the marks with git update-index --no-assume-unchanged --no-skip-worktree")


def _toplevel(path: Path) -> Path | None:
    """The top directory of the repository *path* belongs to; None when it is in none."""
    try:
        out = _git(path, "rev-parse", "--show-toplevel")
    except subprocess.CalledProcessError:
        return None
    return Path(os.fsdecode(out.rstrip(b"\n"))).resolve()


#: Name prefix of the secrets file and of every copy of it (``.secrets.toml.bak-1``, ``~``).
SECRETS_FILE_PREFIX = ".secrets.toml"


def secrets_copies(checkout: Path) -> list[Path]:
    """Every entry under *checkout* whose name starts with ``.secrets.toml``, whatever its
    type, sorted. A directory with such a name counts and is searched too. The search goes
    into ``.git`` but not into its ``objects`` directory, which holds only hashed names.
    Symlinks are reported but not followed, so a link out of the checkout is not searched."""
    found = []
    pending = [Path(checkout)]
    while pending:
        directory = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name.startswith(SECRETS_FILE_PREFIX):
                    found.append(Path(entry.path))
                if entry.is_dir(follow_symlinks=False) and not (entry.name == "objects" and directory.name == ".git"):
                    pending.append(Path(entry.path))
    return sorted(found)


def code_state(checkout: Path) -> CodeState:
    """The state of *checkout* and of its ``harbor`` submodule now. Raise ValueError when
    harbor is not a repository of its own, as in a fresh worktree where the submodule was
    never initialized, and when either repository holds a tracked file marked assume-unchanged
    or skip-worktree, whose changes no digest would record (``_check_nothing_hidden``)."""
    checkout = checkout.resolve()
    harbor = checkout / "harbor"
    if _toplevel(harbor) != harbor.resolve():
        raise ValueError(f"{harbor}: the harbor submodule is not initialized")
    for repo in (checkout, harbor):
        _check_nothing_hidden(repo)
    entries = sorted(_entries(checkout, b"") + _entries(harbor, b"harbor/"))
    dirty = hashlib.sha256(b"".join(entries)).hexdigest() if entries else None
    return CodeState(
        checkout=checkout,
        hls_commit=_git(checkout, "rev-parse", "HEAD").decode().strip(),
        harbor_commit=_git(harbor, "rev-parse", "HEAD").decode().strip(),
        dirty_diff_sha256=dirty,
    )
