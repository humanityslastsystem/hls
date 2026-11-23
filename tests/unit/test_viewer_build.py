"""``hls.viewer.build.find_available_port``: the first port of a range that binds, or None."""

from __future__ import annotations

import socket


def test_find_available_port_returns_int_in_range():
    """find_available_port must return an integer within [start, end]."""
    from hls.viewer.build import find_available_port

    result = find_available_port("127.0.0.1", 19000, 19100)
    assert result is not None, "expected a free port in range 19000-19100"
    assert 19000 <= result <= 19100


def test_find_available_port_none_when_a_server_listens_on_the_only_port():
    """A port a server listens on never binds again, with or without SO_REUSEADDR, so the range
    holding only that port has none free."""
    from hls.viewer.build import find_available_port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        occupied = s.getsockname()[1]
        assert find_available_port("127.0.0.1", occupied, occupied) is None
