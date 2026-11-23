"""pidfds: a handle on one process, through which a signal cannot reach any other process.

Python builds differ in whether they wrap the two system calls (``os.pidfd_open`` and
``signal.pidfd_send_signal``); where a build has no wrapper the call goes through the C library's
``syscall``, so every build hls runs on gets the same behaviour from the same kernel.
"""
from __future__ import annotations

import ctypes
import errno
import os
import signal

# The pidfd system calls' numbers, the same on every architecture hls runs on.
_SYS_PIDFD_SEND_SIGNAL = 424
_SYS_PIDFD_OPEN = 434
# The C library, for the system calls a Python build has no wrapper for.
_LIBC = ctypes.CDLL(None, use_errno=True)


def _syscall(number: int, *args: object) -> int:
    """Linux system call *number*; OSError with its errno when it fails."""
    result = _LIBC.syscall(ctypes.c_long(number), *args)
    if result < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))
    return result


def pidfd_open(pid: int) -> int | None:
    """A pidfd of *pid*, for the caller to close; None where the kernel has no pidfds. Raise
    ProcessLookupError once the process has ended."""
    try:
        if hasattr(os, "pidfd_open"):
            return os.pidfd_open(pid)
        return _syscall(_SYS_PIDFD_OPEN, ctypes.c_int(pid), ctypes.c_uint(0))
    except OSError as e:
        if e.errno == errno.ENOSYS:
            return None
        raise


def pidfd_send_signal(pidfd: int, sig: int) -> None:
    """Send *sig* to the process *pidfd* holds; ProcessLookupError once it has exited."""
    if hasattr(signal, "pidfd_send_signal"):
        signal.pidfd_send_signal(pidfd, sig)
        return
    _syscall(_SYS_PIDFD_SEND_SIGNAL, ctypes.c_int(pidfd), ctypes.c_int(sig), None, ctypes.c_uint(0))
