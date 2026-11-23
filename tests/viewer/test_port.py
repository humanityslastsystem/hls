"""``hls view``'s choice of a port: the first free one of a range, or none when every one is taken."""

from __future__ import annotations

import socket

from hls.viewer.build import find_available_port


def test_find_available_port_returns_int_in_range():
    """find_available_port must return an integer within [start, end]."""
    result = find_available_port("127.0.0.1", 19000, 19100)
    assert result is not None, "expected a free port in range 19000-19100"
    assert 19000 <= result <= 19100


def test_find_available_port_none_when_all_taken():
    """When no port of the range is free, there is none to return. A port a socket listens on
    cannot be bound again, address reuse or not."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        occupied = s.getsockname()[1]
        assert find_available_port("127.0.0.1", occupied, occupied) is None
