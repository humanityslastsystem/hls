"""The signals the handler-reading tests set aside are only the C library's own."""

from __future__ import annotations

import signal

from tests.agent.libc_signals import LIBC_RESERVED


def test_the_signals_set_aside_are_only_the_c_librarys_own():
    # Every standard signal (1 to 31) stays checked; only the libc-reserved realtime ones are set aside.
    assert LIBC_RESERVED & ((1 << 31) - 1) == 0
    assert all(LIBC_RESERVED >> (s - 1) & 1 for s in range(32, signal.SIGRTMIN))
    assert not LIBC_RESERVED >> (signal.SIGRTMIN - 1) & 1
