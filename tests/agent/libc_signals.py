"""The signals the C library keeps for itself, which the tests that read a process's handlers set
aside."""

from __future__ import annotations

import signal

#: The signals from 32 to just below ``SIGRTMIN``: older glibc (el8's 2.28) handles its NPTL
#: signals, SIGCANCEL and SIGSETXID, in every process from the start, and no harness code runs from
#: those handlers. Read from ``signal.SIGRTMIN``, never typed.
LIBC_RESERVED = sum(1 << (s - 1) for s in range(32, signal.SIGRTMIN))
