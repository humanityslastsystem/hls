"""Loaded by every Python program a fake container runs (``tests.drive.fake_container``): it
points the driver's dataset root at the container's fake root, where the real container has
its ``/data`` mounts.

With ``FAKE_FULL_NODE=<dir>`` in its environment, a program also stands on a node whose disk is
full under that directory: whatever would take new space there (a file or directory made, a
link or a new name, a file emptied or appended to) fails with ENOSPC, judged by where its links
lead, and each attempt is appended to the file ``FAKE_FULL_NODE_LOG`` names. A write in place
into a file already there takes no new space and goes through, and a directory made where a name
already is fails as the kernel fails it, with EEXIST before any space is asked for."""

import errno
import os
import sys
from pathlib import Path

_root = os.environ.get("FAKE_CONTAINER_ROOT")
if _root:
    try:
        import hls.spec.datasets as _datasets
    except ImportError:
        pass
    else:
        _datasets.MOUNT_ROOT = Path(_root) / "data"

_full = os.environ.get("FAKE_FULL_NODE")
if _full:
    _FULL = os.path.realpath(_full)
    _LOG = os.environ["FAKE_FULL_NODE_LOG"]
    _MAKES = ("os.mkdir", "os.symlink", "os.link", "os.rename", "os.replace")

    def _taking_space(event: str, args: tuple) -> list:
        """The paths the audited call *event* would take new space at."""
        if event == "open" and len(args) == 3 and isinstance(args[2], int):
            path, flags = args[0], args[2]
            if isinstance(path, int) or not flags & (os.O_WRONLY | os.O_RDWR):
                return []
            if flags & (os.O_TRUNC | os.O_APPEND) or (flags & os.O_CREAT and not os.path.lexists(path)):
                return [path]
            return []
        if event == "os.mkdir":
            return [] if os.path.lexists(args[0]) else [args[0]]
        if event in _MAKES:
            return [args[1]]
        return []

    def _refuse_on_the_full_node(event: str, args: tuple) -> None:
        for path in _taking_space(event, args):
            if not isinstance(path, (str, bytes, os.PathLike)):
                continue
            where = os.path.realpath(os.path.abspath(os.fsdecode(path)))
            if where == _FULL or where.startswith(_FULL + os.sep):
                with open(_LOG, "a") as log:
                    log.write(f"{os.getpid()} {event} {where}\n")
                raise OSError(errno.ENOSPC, "No space left on device", where)

    sys.addaudithook(_refuse_on_the_full_node)
