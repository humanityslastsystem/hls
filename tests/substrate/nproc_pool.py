"""The kernel's ``RLIMIT_NPROC`` check: the limit is no share
of what a step started but a threshold on every task of the uid on the node. A fork by a process
whose limit is L succeeds only while the uid's tasks, whoever started them, number fewer than L;
a process that started nothing is refused once the uid holds L."""

from __future__ import annotations


class UidPool:
    """Every task of one uid on one node."""

    def __init__(self, tasks: int = 0) -> None:
        #: The uid's tasks on the node, processes and threads.
        self.tasks = tasks

    def fork(self, limit: int | None) -> bool:
        """Start one task from a process whose ``RLIMIT_NPROC`` is *limit* (None for unlimited);
        whether the kernel let it."""
        if limit is not None and self.tasks >= limit:
            return False
        self.tasks += 1
        return True

    def grow(self, limit: int | None, wanted: int) -> int:
        """Start up to *wanted* tasks from a process whose limit is *limit*; how many started."""
        started = 0
        while started < wanted and self.fork(limit):
            started += 1
        return started
