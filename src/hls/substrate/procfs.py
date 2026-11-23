"""A process's ``/proc/<pid>/status``, read one way across the codebase.

A process names itself: its ``Name`` line is whatever it set (``prctl(PR_SET_NAME)``, or the name
of the file it ran), any user's or a contestant's, which need not be UTF-8 and can hold a carriage
return; the kernel escapes only a newline in it. So the status is read as bytes and split at
newlines only, and only the fields a reader asks for are kept, their values as the kernel wrote
them: no name can stop a reader or forge a line of the status (``status_fields``). The memory
split (``hls.agent.memory.read_process``) and the uid's task count (``hls.substrate.nproc``) read
it here."""

from __future__ import annotations

from collections.abc import Collection
from pathlib import Path


def status_fields(proc: Path, pid: int | str, keys: Collection[bytes]) -> dict[bytes, bytes] | None:
    """The values of *keys* in ``<proc>/<pid>/status``, each the bytes after its colon,
    stripped; a key the status lacks is absent. None when the process has ended. Raise OSError
    when its status is there but cannot be read."""
    try:
        raw = (proc / str(pid) / "status").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    found: dict[bytes, bytes] = {}
    for line in raw.split(b"\n"):
        key, sep, value = line.partition(b":")
        if sep and key in keys:
            found[key] = value.strip()
    return found
