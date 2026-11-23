"""Bounded listings of directories the agent writes.

The transcript copy (``hls.agent.transcript``) and the runner's health check
(``hls.agent.health``) list the agent's log directories, where the agent can make millions of
names: a listing read whole, or sorted, before a bound applied could hold them all, ending the
transcript role after the commit point or stalling the runner's loop. ``first_entries`` hands
out a directory's entries one at a time, as the kernel lists them, and stops at its bound, saying
whether more were left; neither caller ever holds more names than its bound.

The module uses only the standard library: the transcript role imports it."""

from __future__ import annotations

import os


def first_entries(directory: int | str, limit: int) -> tuple[list[os.DirEntry], bool]:
    """At most the first *limit* entries of *directory*, an open descriptor or a path, in the
    order the filesystem lists them, and whether it held more: one entry more is read to tell.
    None of the rest is read. Raise OSError as ``os.scandir`` does, and ValueError for a negative
    *limit*."""
    if limit < 0:
        raise ValueError(f"a listing's bound must not be negative, not {limit}")
    entries: list[os.DirEntry] = []
    with os.scandir(directory) as it:
        for entry in it:
            if len(entries) == limit:
                return entries, True
            entries.append(entry)
    return entries, False
