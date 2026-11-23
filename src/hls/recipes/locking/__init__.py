"""The tools that write the image recipes' pins,
behind ``hls images``: ``lock-apt``, ``lock-pip`` and ``lock-npm`` regenerate a lock under
``hls.recipes.pins.LOCKS_DIR`` from its inputs, and ``base-digest`` reads the digest
``pins.CUDA_BASE`` holds. Each refuses, with ``LockRefused``, a lock an image may not hold. They
read public registries and package indexes and run on a maintainer's machine, not in an
allocation."""


import re
from urllib.parse import urlsplit

#: A lock's name: one file name of the locks directory, with no separator and no leading dot.
_LOCK_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class LockRefused(ValueError):
    """A lock or pin that cannot be written or read as asked, with the reason."""


def lock_name(name: str) -> str:
    """*name*, a lock's name; raise LockRefused for one that is not a plain file name, so nothing
    is read or written outside the locks directory."""
    if not _LOCK_NAME.fullmatch(name):
        raise LockRefused(f"{name!r} is no lock name: a lock is named by one file name of the locks "
                          "directory, such as apt-base or base-s-a")
    return name


def unreachable(url: str, error: Exception) -> LockRefused:
    """The refusal for *url* that could not be read, naming its host and the error."""
    return LockRefused(f"{urlsplit(url).hostname} could not be read ({url}): {type(error).__name__}: {error}")
