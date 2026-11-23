"""The tree digest of a task's uploaded folders: the value a task module commits beside its
``upload_dir`` (``AgentProducerSpec.upload_sha256``), the digest of the view the folders make
merged at one destination, and the one the build checks before it copies them into the sandbox,
so every checkout hands a contestant the same material.

Each of a task's folders and where it lands is judged by one rule here, both when a scored
command plans a build (``hls.pipeline.preflight``) and when the build starts
(``hls.stages.agent_build``), so the two cannot disagree:
- ``checked_upload``: each uploaded folder is named from the task module's directory, may be
  reached through links and ``..`` as long as it resolves to a directory inside the checkout,
  and holds no link; no path is held by two of them; and their merged view has the tree digest
  the task pins (``merged_digest``);
- ``upload_destination``: it lands under ``/task`` or ``/workspace``, the sandbox directories
  the agent sees;
- ``task_skills``: the skills folder is named and resolved as the uploaded one is, and is one
  skill or a folder of skills by harbor's rule."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from hls.agent import workspace
from hls.prompt import UPLOAD_DEST
from hls.store.digest import sha256_stream
from hls.util import walk_tree

if TYPE_CHECKING:
    from harbor.skills import ResolvedSkill


class TreeDigestError(ValueError):
    """A folder whose digest cannot be taken: no directory, a link in it, or an entry that is
    neither a file nor a directory."""


class UploadRefused(ValueError):
    """A task folder a build may not hand the agent (``task_folder``, ``checked_upload``)."""


def _file_sha256(path: Path) -> str:
    """The sha256 of the bytes of *path*, which the walk found to be a regular file. It is opened
    without following a link and without waiting, so a pipe or a link that took its place since
    is neither waited on nor followed."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as f:
        return sha256_stream(f)[0]


def _held_mode(st_mode: int) -> int:
    """A file's mode as git holds it: 755 when its owner may execute it, 644 otherwise. A
    checkout writes a file's other bits from its umask, so they say nothing of the folder."""
    return 0o755 if st_mode & stat.S_IXUSR else 0o644


def _tree(root: Path) -> tuple[dict[str, str], set[str]]:
    """The digest line of every file under *root* by its relative POSIX path, one each
    (``tree_digest``), and the relative path of every directory under it. Raise TreeDigestError
    as ``tree_digest`` does."""
    root = Path(root)
    if root.is_symlink():
        raise TreeDigestError(f"{root} is a link")
    if not root.is_dir():
        raise TreeDigestError(f"{root} is not a directory")
    try:
        entries = list(walk_tree(root))
    except OSError as e:
        raise TreeDigestError(f"{e.filename} cannot be listed: {e.strerror}") from e
    files: dict[str, str] = {}
    dirs: set[str] = set()
    for entry in entries:
        if entry.kind == "link":
            raise TreeDigestError(f"{entry.path} is a link")
        if entry.kind == "special":
            raise TreeDigestError(f"{entry.path} is neither a file nor a directory")
        if entry.kind == "file":
            files[entry.rel] = f"{entry.rel}\0{_held_mode(entry.stat.st_mode):o}\0{_file_sha256(entry.path)}\n"
        else:
            dirs.add(entry.rel)
    return files, dirs


def _digest(lines: list[str]) -> str:
    return hashlib.sha256("".join(sorted(lines)).encode("utf-8", "surrogateescape")).hexdigest()


def tree_digest(root: Path) -> str:
    """The sha256 over every file under *root*, in the order of their relative POSIX paths, of
    one line each: ``<path>\\0<mode in octal>\\0<sha256 of the bytes>\\n``, the mode as git holds
    it (``_held_mode``), so a folder a commit holds has one digest whatever umask checked it out.
    A directory adds nothing of its own, so an empty one changes nothing. Raise TreeDigestError when *root*
    is a link or no directory, when anything under it is a link or neither a file nor a
    directory (``walk_tree``): a link would hand over whatever it points to when the folder is
    copied; and when a directory under it cannot be listed, since its files would be left out
    unseen. Each entry is judged by its ``lstat`` before anything opens it, so a special file is
    refused without being opened."""
    return _digest(list(_tree(root)[0].values()))


def _merged(roots: Sequence[Path]) -> dict[str, tuple[Path, str]]:
    """Every file of the view *roots* make merged, by its relative POSIX path: the folder that
    holds it and its digest line. Raise TreeDigestError as ``merged_digest`` does."""
    held: dict[str, tuple[Path, str]] = {}
    view: dict[str, tuple[Path, str]] = {}
    for root in roots:
        files, dirs = _tree(root)
        for rel, kind in [*((rel, "file") for rel in files), *((rel, "dir") for rel in dirs)]:
            first = held.get(rel)
            if first is not None and "file" in (first[1], kind):
                raise TreeDigestError(f"{rel} is held by both {first[0]} and {root}")
            held.setdefault(rel, (Path(root), kind))
        view.update({rel: (Path(root), line) for rel, line in files.items()})
    return view


def merged_digest(roots: Sequence[Path]) -> str:
    """The tree digest of the view *roots* make when each is copied into one directory: the
    ``tree_digest`` of a directory that holds every file of each at its own relative path, so the
    merged digest of one folder is its own ``tree_digest``. Directories merge. Raise
    TreeDigestError as ``tree_digest`` does for each, and for a path two of them hold as anything
    but a directory in both: the copy of the second would replace the first's file, or put a file
    where the first has a directory."""
    return _digest([line for _, line in _merged(roots).values()])


def merged_view(roots: Sequence[Path]) -> dict[str, Path]:
    """Every file of the view *roots* make merged (``merged_digest``), by its relative POSIX path,
    to the file in the folder that holds it. Raise TreeDigestError as ``merged_digest`` does."""
    return {rel: root / rel for rel, (root, _) in _merged(roots).items()}


def task_folder(value: str, base: Path, checkout: Path) -> Path:
    """The directory a producer folder (``upload_dir``, ``skills_dir``) names, resolved: an
    absolute *value*, after ``~`` is expanded, as it is, and a relative one from the task
    module's directory *base*. Links and ``..`` on the way are followed, so the folder may be
    named through either, since the task's author names it. Raise UploadRefused when it
    resolves outside the checkout *checkout*, whose copy of the task is the one that runs, or
    to no directory."""
    path = Path(value).expanduser()
    path = (path if path.is_absolute() else base / path).resolve()
    if not path.is_relative_to(checkout.resolve()):
        raise UploadRefused(f"{path} lies outside the checkout {checkout}")
    if not path.is_dir():
        raise UploadRefused(f"{path} is not a directory")
    return path


def checked_upload(values: Sequence[str], base: Path, checkout: Path,
                   pinned: str | None) -> tuple[list[Path], str]:
    """The folders a build uploads into its sandbox, merged at one destination (``upload_dir``
    *values*, each by ``task_folder``), resolved, and the tree digest of their merged view
    (``merged_digest``). Raise UploadRefused when one lies outside the checkout or is no
    directory, holds a link or an entry that is neither a file nor a directory, when two hold
    the same path, or when the merged view has another tree digest than *pinned*, the one the
    task commits (``upload_sha256``): the agent would be handed material the task does not name."""
    folders = [task_folder(value, base, checkout) for value in values]
    try:
        digest = merged_digest(folders)
    except TreeDigestError as e:
        raise UploadRefused(str(e)) from e
    if digest != pinned:
        named = folders[0] if len(folders) == 1 else ", ".join(str(f) for f in folders)
        raise UploadRefused(f"{named} has tree digest {digest}, and the task commits {pinned}")
    return folders, digest


def upload_destination(value: str | None) -> tuple[str, PurePosixPath]:
    """Where in the sandbox a build copies the uploaded folder: *value* (``upload_dest``), or
    ``UPLOAD_DEST`` for none, as the sandbox directory it lies under, ``workspace.TASK`` or
    ``workspace.WORKSPACE``, and its path below that directory. Raise UploadRefused for a
    destination under neither, or one that names ``..``: the sandbox binds only those two for
    the agent's files, and a copy that climbs out of them would land on the host."""
    dest = PurePosixPath(value or UPLOAD_DEST)
    under = next((d for d in (workspace.TASK, workspace.WORKSPACE) if dest.is_relative_to(d)), None)
    if under is None or ".." in dest.parts:
        raise UploadRefused(f"{dest} is neither under {workspace.TASK} nor under {workspace.WORKSPACE}")
    return under, dest.relative_to(under)


def task_skills(value: str, base: Path, checkout: Path) -> list[ResolvedSkill]:
    """The skills a build copies into its sandbox from the folder *value* names
    (``skills_dir``, ``task_folder``), each under its own name, by the rule harbor's harnesses
    read skills by (``harbor.skills.resolve_skills``): the folder is one skill, holding
    ``SKILL.md``, or a folder each of whose directories, hidden ones aside, is one. Raise
    UploadRefused when the folder lies outside the checkout *checkout* or is no directory, or is
    neither a skill nor a folder of skills, naming each directory in it that holds no
    ``SKILL.md``: the agent would be handed a folder its harness does not read as skills."""
    from harbor.skills import resolve_skills  # harbor is imported only once a build is planned or runs

    folder = task_folder(value, base, checkout)
    try:
        return resolve_skills([folder])
    except (OSError, ValueError) as e:
        raise UploadRefused(str(e)) from e
