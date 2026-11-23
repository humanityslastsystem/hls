from __future__ import annotations

import errno
import os
import signal
import subprocess
import sys

import pytest

from hls.substrate import pidfd


def test_a_kernel_without_pidfds_reads_as_none_and_an_ended_process_as_gone(monkeypatch):
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    answer = [errno.ENOSYS]

    def syscall(number, *args):
        raise OSError(answer[0], os.strerror(answer[0]))

    monkeypatch.setattr(pidfd, "_syscall", syscall)
    assert pidfd.pidfd_open(os.getpid()) is None
    answer[0] = errno.ESRCH
    with pytest.raises(ProcessLookupError):
        pidfd.pidfd_open(os.getpid())


def test_a_python_without_the_wrappers_opens_and_signals_through_the_system_calls(monkeypatch):
    """A Python build with no ``os.pidfd_open`` or ``signal.pidfd_send_signal`` gets a pidfd that
    ends the process it holds, as one with them does."""
    monkeypatch.delattr(os, "pidfd_open", raising=False)
    monkeypatch.delattr(signal, "pidfd_send_signal", raising=False)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        fd = pidfd.pidfd_open(child.pid)
        assert fd is not None
        try:
            pidfd.pidfd_send_signal(fd, signal.SIGKILL)
        finally:
            os.close(fd)
        assert child.wait(10) == -signal.SIGKILL
    finally:
        child.kill()
        child.wait()
    with pytest.raises(ProcessLookupError):
        pidfd.pidfd_open(child.pid)
