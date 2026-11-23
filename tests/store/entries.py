"""Cache entries for tests, and a record of which files the test process reads.

``populate`` fills an entry as a stage does with a fill role: the role's files are hashed the
way such a role reports them (``hash_tree``), here by the test, which stands in for the role.

``recording_reads`` lists every regular file this process opens for reading while it is
active, through an audit hook (``sys.addaudithook``), so a test can show that the runner's own
process never reads a cache entry's files or a stage's outputs: ``entry_reads`` keeps the reads
under the given roots, but the cache's own ``ENTRY.json`` and stamp, which it reads to pin an
entry and to trust a copy. Two reads the hook cannot place are kept whatever the roots:
- an open of a relative path that names no file from the working directory, such as one made
  under a directory descriptor (``os.open(name, dir_fd=...)``), whose audit event carries no
  descriptor; it is recorded as ``UNRESOLVED`` and the path;
- any process this one starts, by ``subprocess``, ``os.exec*``, ``os.posix_spawn``, ``os.spawn*``,
  ``os.system`` or ``os.fork``, but a role step a launcher starts (``launches``): a child's reads
  are outside any hook of this process. It is recorded as ``SPAWNED`` and what was run."""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

from hls.store.cache import RESERVED_NAMES, ContentCache, Populated, hash_tree
from hls.substrate.apptainer.images import ApptainerImages

#: The prefix of a read the hook could not resolve to a path.
UNRESOLVED = "<unresolved> "
#: The prefix of a process this one started outside a launcher.
SPAWNED = "<spawned> "
#: Audit events of a process starting another, or a copy of itself.
_SPAWNS = frozenset({"subprocess.Popen", "os.exec", "os.posix_spawn", "os.spawn", "os.system", "os.fork",
                     "os.forkpty"})
#: The reads while a recording is active; None when none is.
_READS: list[str] | None = None
#: The code of each function whose processes are role steps (``launches``).
_LAUNCHERS: set = set()


def launches(func: Callable) -> Callable:
    """Mark *func* as starting role steps, as a launcher does: a process started while it runs
    is not recorded. Returns *func*, so it serves as a decorator."""
    _LAUNCHERS.add(func.__code__)
    return func


def _in_launcher() -> bool:
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code in _LAUNCHERS:
            return True
        frame = frame.f_back
    return False


def _audit(event: str, args: tuple) -> None:
    if _READS is None:
        return
    if event in _SPAWNS:
        if not _in_launcher():
            _READS.append(SPAWNED + f"{event} {args!r}")
        return
    if event != "open" or len(args) != 3:
        return
    path, _, flags = args
    if not isinstance(path, (str, bytes, os.PathLike)) or not isinstance(flags, int):
        return
    if flags & os.O_ACCMODE == os.O_WRONLY:
        return
    name = os.fsdecode(path)
    # A directory is opened to sync it, which reads nothing of a file.
    if os.path.isfile(name):
        _READS.append(os.path.abspath(name))
    elif not os.path.isabs(name) and not os.path.isdir(name) and not flags & os.O_DIRECTORY:
        _READS.append(UNRESOLVED + name)


sys.addaudithook(_audit)


@contextmanager
def recording_reads() -> Iterator[list[str]]:
    """The regular files this process opens for reading from now until the block ends, with the
    reads and processes the hook cannot place (``UNRESOLVED``, ``SPAWNED``)."""
    global _READS
    _READS = []
    try:
        yield _READS
    finally:
        _READS = None


def entry_reads(reads: Iterable[str], roots: Iterable[Path]) -> list[str]:
    """The paths of *reads* that lie under one of *roots*, the cache's own files left out, and
    every read and process the hook could not place."""
    resolved = [Path(r).resolve() for r in roots]

    def kept(p: str) -> bool:
        if p.startswith((UNRESOLVED, SPAWNED)):
            return True
        path = Path(p)
        return path.name not in RESERVED_NAMES and any(path.resolve().is_relative_to(r) for r in resolved)

    return [p for p in reads if kept(p)]


def populate(cache: ContentCache, kind: str, key: str, fill: Callable[[Path], None], **kw) -> Populated:
    """``cache.populate`` with *fill*, each file it wrote reported as a fill role reports it."""
    staged: list[Path] = []

    def wrapped(staging: Path) -> None:
        staged.append(staging)
        fill(staging)

    return cache.populate(kind, key, wrapped, digests=lambda: hash_tree(staged[-1]), **kw)


#: The bytes of the stand-in image file ``hold_images`` publishes.
STAND_IN_IMAGE = b"image"


def hold_images(store_root: Path, keys: Iterable[str]) -> None:
    """Publish in the cache of the store at *store_root* an images entry under each of *keys* it
    does not hold yet, holding a stand-in image file (``STAND_IN_IMAGE``), as an images warmer
    would have."""
    from hls.stages.images import IMAGE_FILE
    from hls.store.lock import Holder
    from hls.store.posix import PosixStore

    cache = ContentCache(PosixStore(store_root))
    for key in keys:
        populate(cache, "images", key, lambda d: (d / IMAGE_FILE).write_bytes(STAND_IN_IMAGE),
                 holder=Holder.here("test"))


def hold_measured(store_root: Path, catalog_task: str) -> dict[str, str]:
    """Publish in the cache of the store at *store_root* an images entry for each image catalog
    task *catalog_task* declares for itself that it does not hold yet, as the task's images warmer
    (``hls images build``) would have; return the key of every one of them by name."""
    from hls.recipes import catalog
    from hls.stages.images import image_keys

    images = catalog.task_images(catalog_task)
    keys = image_keys(images.decls, ApptainerImages.recipe_key)
    measured = {name: keys[name] for name in images.measured.values()}
    hold_images(store_root, measured.values())
    return measured


def _register_launchers() -> None:
    """The local launcher, which starts every role step of the tests' stages."""
    from tests.pipeline.fakes import LocalLauncher

    launches(LocalLauncher.launch)


_register_launchers()
