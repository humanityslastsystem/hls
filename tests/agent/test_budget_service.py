from __future__ import annotations

import errno
import json
import socket
import stat
import threading
import time
from pathlib import Path

import pytest

import hls.budget.service as service
from hls.budget.service import NOTE, BudgetService, WallClock, answer
from hls.util import MAX_SOCKET_PATH_BYTES


class _Clock:
    """A clock the test moves by hand."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def _clock(limit=3600, hard=4000, now=1000.0):
    c = _Clock(now)
    return WallClock(limit_sec=limit, hard_sec=hard, clock=c), c


@pytest.fixture
def channel(short_dir):
    d = short_dir / "ch"
    d.mkdir(mode=0o700)
    return d


def _ask(path: Path, *lines: bytes) -> list[dict]:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5)
        conn.connect(str(path))
        replies = []
        with conn.makefile("rb") as reader:
            for line in lines:
                conn.sendall(line)
                replies.append(json.loads(reader.readline()))
        return replies


def _closed(conn: socket.socket) -> bool:
    """True when the peer closed *conn*: an end of file, or a reset when it closed with our
    request unread."""
    try:
        return conn.recv(100) == b""
    except ConnectionResetError:
        return True


def _refused(conn: socket.socket, request: bytes) -> bool:
    """True when the peer closed *conn* instead of answering *request*, whether it closed before
    the request was sent (a broken pipe or a reset on the send) or after (``_closed``)."""
    try:
        conn.sendall(request)
    except (BrokenPipeError, ConnectionResetError):
        return True
    return _closed(conn)


@pytest.fixture
def served(channel):
    clock, moved = _clock()
    svc = BudgetService(clock, channel / "budget.sock")
    svc.start()
    yield svc, clock, moved
    svc.stop()


# the clock


def test_the_clock_is_monotonic_unless_a_test_supplies_one():
    """A step of the node's wall clock must move nothing."""
    assert WallClock(limit_sec=1, hard_sec=1).clock is time.monotonic


def test_the_clock_starts_once_and_counts_from_its_start():
    clock, moved = _clock()
    assert not clock.started and clock.elapsed_sec() == 0.0 and clock.remaining_sec() == 3600
    moved.now += 50
    clock.start()
    assert clock.started and clock.elapsed_sec() == 0.0
    moved.now += 120.5
    assert clock.elapsed_sec() == 120.5
    assert clock.remaining_sec() == 3600 - 120.5
    assert clock.hard_remaining_sec() == 4000 - 120.5
    with pytest.raises(RuntimeError, match="already started"):
        clock.start()


def test_past_the_budget_the_remainder_goes_negative():
    clock, moved = _clock(limit=100, hard=200)
    clock.start()
    moved.now += 130
    assert clock.remaining_sec() == -30
    assert clock.hard_remaining_sec() == 70


def test_a_failing_clock_reads_as_no_time_left_and_never_raises():
    calls = iter([0.0])

    def clock():
        try:
            return next(calls)
        except StopIteration:
            raise OSError("clock gone") from None

    wall = WallClock(limit_sec=3600, hard_sec=3600, clock=clock)
    wall.start()
    assert wall.remaining_sec() == 0.0


def test_stopped_reentry_reports_no_time_left_while_the_agent_still_reads_the_clock():
    clock, moved = _clock()
    clock.start()
    moved.now += 10
    clock.stop_reentry()
    assert clock.remaining_sec() == 0.0
    assert clock.response()["dimensions"]["wallclock"]["remaining"] == 3590


@pytest.mark.parametrize(("limit", "hard"), [(0, 10), (-5, 10), (100, 99)])
def test_a_clock_whose_limits_make_no_sense_is_refused(limit, hard):
    with pytest.raises(ValueError):
        WallClock(limit_sec=limit, hard_sec=hard)


def test_the_response_is_the_old_tools_restricted_to_the_wall_clock_in_whole_seconds():
    clock, moved = _clock(limit=3600)
    assert clock.response() == {"dimensions": {"wallclock": {"unit": "seconds", "limit": 3600, "spent": 0,
                                                             "remaining": 3600, "over_budget": False}},
                                "note": NOTE}
    clock.start()
    moved.now += 3599.9
    wall = clock.response()["dimensions"]["wallclock"]
    assert (wall["spent"], wall["remaining"], wall["over_budget"]) == (3599, 1, False)
    assert all(isinstance(wall[k], int) for k in ("limit", "spent", "remaining"))
    moved.now += 0.1
    wall = clock.response()["dimensions"]["wallclock"]
    assert (wall["spent"], wall["remaining"], wall["over_budget"]) == (3600, 0, True)
    moved.now += 100
    assert clock.response()["dimensions"]["wallclock"]["remaining"] == -100


def test_the_answer_reads_nothing_but_the_requests_name():
    clock, _ = _clock()
    assert answer(b'{"op": "get_budget"}\n', clock) == clock.response()
    assert answer(b'{"op": "get_budget", "limit": 1}\n', clock) == clock.response()
    for line in (b"[1]\n", b'{"op": "set_budget"}\n', b'"get_budget"\n', b"{nope\n"):
        assert set(answer(line, clock)) == {"error"}


def test_a_request_nested_past_the_bound_is_refused_before_it_is_parsed():
    """A request is one flat object. One nested deeper than ``MAX_REQUEST_DEPTH`` is refused
    before the parse, whose recursion a deep enough line would exhaust, whatever bounds the line;
    brackets inside a string, an escaped quote's among them, are no nesting."""
    clock, _ = _clock()

    def nested(depth: int) -> bytes:
        inner = depth - 1
        extra = b', "x": ' + b"[" * inner + b"]" * inner if inner else b""
        return b'{"op": "get_budget"' + extra + b"}\n"

    assert answer(nested(service.MAX_REQUEST_DEPTH), clock) == clock.response()
    too_deep = {"error": f"a request nests no deeper than {service.MAX_REQUEST_DEPTH}"}
    assert answer(nested(service.MAX_REQUEST_DEPTH + 1), clock) == too_deep
    for line in (b"[" * 200_000, b"{" * 200_000, b'{"a": ' * 200_000):
        assert answer(line, clock) == too_deep
    for text in (b"[" * 200, b'\\"' + b"{" * 200):
        assert answer(b'{"op": "get_budget", "x": "' + text + b'"}\n', clock) == clock.response()


def test_a_request_is_utf8_the_encoding_its_nesting_was_counted_in():
    """``json.loads`` of bytes would read UTF-16 too, whose code units can hold a quote's byte and
    so hide a nesting from the count: the request in UTF-16 is refused."""
    clock, _ = _clock()
    assert answer('{"op": "get_budget"}\n'.encode("utf-16-le"), clock) == {
        "error": "a request is one JSON object per line"}


# the socket


def test_the_service_answers_get_budget_on_its_socket(served):
    svc, clock, moved = served
    clock.start()
    moved.now += 42
    [reply] = _ask(svc.socket_path, b'{"op": "get_budget"}\n')
    assert reply == clock.response()
    assert reply["dimensions"]["wallclock"]["spent"] == 42


def test_the_socket_is_the_owners_only(served):
    svc, _, _ = served
    assert stat.S_ISSOCK(svc.socket_path.stat().st_mode)
    assert stat.S_IMODE(svc.socket_path.stat().st_mode) == 0o600


def test_another_request_is_refused_and_the_connection_stays_open(served):
    svc, clock, _ = served
    first, second, third = _ask(svc.socket_path, b'{"op": "stop"}\n', b"not json\n", b'{"op": "get_budget"}\n')
    assert "error" in first and "error" in second
    assert third == clock.response()


def test_a_line_too_long_closes_the_connection(served):
    svc, _, _ = served
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(5)
        conn.connect(str(svc.socket_path))
        conn.sendall(b"x" * (service.MAX_LINE_BYTES + 1))
        with conn.makefile("rb") as reader:
            assert json.loads(reader.readline()) == {"error": "request line too long"}
            assert reader.readline() == b""


def test_a_line_at_the_limit_is_read(served):
    svc, clock, _ = served
    request = b'{"op": "get_budget"}'
    line = request + b" " * (service.MAX_LINE_BYTES - len(request)) + b"\n"
    [reply] = _ask(svc.socket_path, line)
    assert reply == clock.response()


def test_connections_beyond_the_cap_are_closed_and_the_service_keeps_serving(channel, monkeypatch):
    monkeypatch.setattr(service, "MAX_CONNECTIONS", 2)
    clock, _ = _clock()
    svc = BudgetService(clock, channel / "budget.sock")
    svc.start()
    held = []
    try:
        for _ in range(2):
            conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            conn.connect(str(svc.socket_path))
            held.append(conn)
        time.sleep(0.3)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as extra:
            extra.settimeout(5)
            extra.connect(str(svc.socket_path))
            assert _refused(extra, b'{"op": "get_budget"}\n')
        held.pop().close()
        time.sleep(0.3)
        [reply] = _ask(svc.socket_path, b'{"op": "get_budget"}\n')
        assert reply == clock.response()
    finally:
        for conn in held:
            conn.close()
        svc.stop()


def test_a_connection_thread_that_cannot_start_is_dropped_and_the_service_keeps_serving(channel, monkeypatch):
    """A process limit in the sandbox can refuse a new thread. That connection is closed and its slot
    freed; the accept loop lives on, and the next connection is served."""
    monkeypatch.setattr(service, "MAX_CONNECTIONS", 1)
    clock, _ = _clock()
    svc = BudgetService(clock, channel / "budget.sock")
    svc.start()
    real_start, refused = threading.Thread.start, []

    def start(self):
        if self.name == "hls-budget-conn" and not refused:
            refused.append(self)
            raise RuntimeError("can't start new thread")
        real_start(self)

    monkeypatch.setattr(threading.Thread, "start", start)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as first:
            first.settimeout(2)
            first.connect(str(svc.socket_path))
            assert _refused(first, b'{"op": "get_budget"}\n')
        assert len(refused) == 1
        # with one slot, a slot the refused connection kept would close this one too
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as second:
            second.settimeout(2)
            second.connect(str(svc.socket_path))
            second.sendall(b'{"op": "get_budget"}\n')
            with second.makefile("rb") as reader:
                assert json.loads(reader.readline()) == clock.response()
    finally:
        svc.stop()


def test_an_accept_that_fails_is_retried_after_a_pause_and_then_served(channel, monkeypatch):
    real_accept, failures = socket.socket.accept, []

    def accept(self):
        if len(failures) < 3:
            failures.append(1)
            raise OSError(errno.EMFILE, "Too many open files")
        return real_accept(self)

    monkeypatch.setattr(socket.socket, "accept", accept)
    clock, _ = _clock()
    svc = BudgetService(clock, channel / "budget.sock")
    svc.start()
    try:
        [reply] = _ask(svc.socket_path, b'{"op": "get_budget"}\n')
        assert reply == clock.response() and len(failures) == 3
    finally:
        svc.stop()


def test_an_accept_that_keeps_failing_does_not_spin(channel, monkeypatch):
    """EMFILE repeats at once, so a loop that retried without a pause would burn a core and
    flood the log."""
    calls = []

    def accept(self):
        calls.append(1)
        raise OSError(errno.EMFILE, "Too many open files")

    monkeypatch.setattr(socket.socket, "accept", accept)
    monkeypatch.setattr(service.log, "exception", lambda *a, **kw: None)
    clock, _ = _clock()
    svc = BudgetService(clock, channel / "budget.sock")
    window = 0.6
    svc.start()
    try:
        time.sleep(window)
        seen = len(calls)
    finally:
        svc.stop()
    assert 2 <= seen <= window / service._ACCEPT_ERROR_PAUSE_SEC + 2


def test_an_idle_connection_is_dropped(channel, monkeypatch):
    monkeypatch.setattr(service, "IDLE_TIMEOUT_SEC", 0.3)
    clock, _ = _clock()
    svc = BudgetService(clock, channel / "budget.sock")
    svc.start()
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(5)
            conn.connect(str(svc.socket_path))
            assert _closed(conn)
    finally:
        svc.stop()


def test_many_clients_at_once_all_get_answers(served):
    svc, clock, _ = served
    replies: list[dict] = []

    def one():
        replies.extend(_ask(svc.socket_path, b'{"op": "get_budget"}\n'))

    threads = [threading.Thread(target=one) for _ in range(service.MAX_CONNECTIONS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert replies == [clock.response()] * service.MAX_CONNECTIONS


def test_stop_removes_the_socket(channel):
    clock, _ = _clock()
    svc = BudgetService(clock, channel / "budget.sock")
    svc.start()
    svc.stop()
    assert not svc.socket_path.exists()
    with pytest.raises(OSError), socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(str(svc.socket_path))


def test_a_taken_path_is_refused_and_left_alone(channel):
    taken = channel / "budget.sock"
    taken.write_text("someone else's")
    clock, _ = _clock()
    with pytest.raises(FileExistsError):
        BudgetService(clock, taken).start()
    assert taken.read_text() == "someone else's"


def test_stopping_a_service_that_never_bound_leaves_the_path_alone(channel):
    taken = channel / "budget.sock"
    taken.write_text("someone else's")
    clock, _ = _clock()
    svc = BudgetService(clock, taken)
    with pytest.raises(FileExistsError):
        svc.start()
    svc.stop()
    assert taken.read_text() == "someone else's"


def test_a_path_taken_after_the_check_is_left_alone(channel, monkeypatch):
    """bind refuses a path that appeared between the check and the bind; what is there is
    not ours to remove."""
    taken = channel / "budget.sock"
    taken.write_text("someone else's")
    monkeypatch.setattr(Path, "exists", lambda self, **kw: False)
    monkeypatch.setattr(Path, "is_symlink", lambda self: False)
    clock, _ = _clock()
    with pytest.raises(OSError):
        BudgetService(clock, taken).start()
    monkeypatch.undo()
    assert taken.read_text() == "someone else's"


def test_a_socket_path_the_kernel_cannot_hold_is_refused(short_dir):
    clock, _ = _clock()
    longest = short_dir / ("s" * (MAX_SOCKET_PATH_BYTES - len(str(short_dir)) - 1))
    assert len(str(longest).encode()) == MAX_SOCKET_PATH_BYTES
    BudgetService(clock, longest)
    with pytest.raises(ValueError, match="longer than"):
        BudgetService(clock, Path(str(longest) + "s"))
