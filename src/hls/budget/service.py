"""The build's wall-clock budget, kept by the runner.

The agent's budget is measured on the runner's clock, from the moment the agent_build stage
hands control to the agent: after the sandbox is up, the task files are in, the preflight
passed and the harness's setup returned. Nothing the agent can write moves it: it reads no
transcript timestamp, which the agent can write.

``WallClock`` is the clock. The stage starts it, reads ``hard_remaining_sec`` to enforce the
hard limit, and hands ``remaining_sec`` to Claude Code's session re-entry, which calls it on
the build's event loop after every exit, so it is a plain computation that never raises.

``BudgetService`` answers the agent. It serves a UNIX socket in the node-local channel
directory that the sandbox sees read-only at ``/run/hls``, and the in-sandbox MCP server
(``hls.budget.mcp``) asks it on every ``get_budget`` call. It answers one request, with the
wall clock alone: API cost has no source the agent cannot write, so it is not reported. Anything
in the sandbox can reach the
socket, so the service reads nothing from a request beyond its name, caps its connections,
line length and a request's nesting, and drops idle connections."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from collections.abc import Callable
from pathlib import Path

from hls.util import MAX_SOCKET_PATH_BYTES, JsonDepthError, load_json

log = logging.getLogger(__name__)

#: The note every response carries.
NOTE = "Your budget for this build. Pace your work to stay within it."
#: The one request the service answers.
GET_BUDGET = "get_budget"
#: Longest request line, in bytes, newline excluded. A longer one closes the connection.
MAX_LINE_BYTES = 4096
#: Connections served at once; one more is closed as soon as it is accepted.
MAX_CONNECTIONS = 16
#: Seconds a connection may go without sending anything before it is closed.
IDLE_TIMEOUT_SEC = 30.0
#: Deepest nesting of objects and arrays a request line may hold: a request is one flat object
#: (``hls.budget.mcp`` sends ``{"op": "get_budget"}``). The line is checked before it is parsed,
#: as the parse recurses once a level and a deep enough line would exhaust the recursion limit,
#: whatever bounds the line's length.
MAX_REQUEST_DEPTH = 1
# Seconds between checks of the stop flag while waiting for a connection.
_ACCEPT_POLL_SEC = 0.2
# Seconds to wait after an accept fails, so EMFILE, which repeats at once, does not spin the loop.
_ACCEPT_ERROR_PAUSE_SEC = 0.1


class WallClock:
    """The agent's wall clock: started once, read by the stage, the service and re-entry."""

    def __init__(self, *, limit_sec: int, hard_sec: int, clock: Callable[[], float] = time.monotonic) -> None:
        if limit_sec <= 0:
            raise ValueError(f"limit_sec must be positive, got {limit_sec}")
        if hard_sec < limit_sec:
            raise ValueError(f"hard_sec ({hard_sec}) must be at least limit_sec ({limit_sec})")
        #: The budget the agent is told: the arm's wall-clock budget, in seconds.
        self.limit_sec = limit_sec
        #: The limit the stage enforces, from the same start, in seconds.
        self.hard_sec = hard_sec
        #: Time source; monotonic, so a change of the node's clock moves nothing.
        self.clock = clock
        self._started_at: float | None = None
        self._reentry_stopped = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the clock: the agent has control from now. A second call raises."""
        with self._lock:
            if self._started_at is not None:
                raise RuntimeError("the agent's clock was already started")
            self._started_at = self.clock()

    @property
    def started(self) -> bool:
        """True once ``start`` was called."""
        return self._started_at is not None

    def elapsed_sec(self) -> float:
        """Seconds since the start; 0 before it."""
        started = self._started_at
        return 0.0 if started is None else self.clock() - started

    def remaining_sec(self) -> float:
        """Seconds of the agent's budget left, for session re-entry: negative past the
        budget, and 0 once re-entry was stopped. It never raises: a clock that fails reads
        as no time left, so a session is not resumed on a budget nobody can measure, and the
        hard limit still ends the build."""
        if self._reentry_stopped:
            return 0.0
        try:
            return self.limit_sec - self.elapsed_sec()
        except Exception:  # noqa: BLE001 -- re-entry calls this on the event loop and must not fail the build
            log.exception("the agent's clock could not be read; re-entry sees no time left")
            return 0.0

    def hard_remaining_sec(self) -> float:
        """Seconds until the hard limit."""
        return self.hard_sec - self.elapsed_sec()

    def stop_reentry(self) -> None:
        """Make ``remaining_sec`` report 0 from now on, so no session is resumed again. The
        agent's own ``get_budget`` keeps reading the clock."""
        self._reentry_stopped = True

    def response(self) -> dict:
        """What ``get_budget`` returns: the wall-clock dimension alone, in whole seconds."""
        spent = int(self.elapsed_sec())
        return {"dimensions": {"wallclock": {"unit": "seconds", "limit": self.limit_sec, "spent": spent,
                                             "remaining": self.limit_sec - spent,
                                             "over_budget": spent >= self.limit_sec}},
                "note": NOTE}


def answer(line: bytes, clock: WallClock) -> dict:
    """The reply to one request line, UTF-8 JSON; an error for one nested deeper than
    ``MAX_REQUEST_DEPTH``, before it is parsed (``load_json``)."""
    try:
        request = load_json(line, max_depth=MAX_REQUEST_DEPTH)
    except JsonDepthError:
        return {"error": f"a request nests no deeper than {MAX_REQUEST_DEPTH}"}
    except ValueError:
        return {"error": "a request is one JSON object per line"}
    if not isinstance(request, dict) or request.get("op") != GET_BUDGET:
        return {"error": f"the only request is {{\"op\": \"{GET_BUDGET}\"}}"}
    return clock.response()


class BudgetService:
    """Serves ``get_budget`` from *clock* on the UNIX socket *socket_path*."""

    def __init__(self, clock: WallClock, socket_path: Path) -> None:
        if len(os.fsencode(socket_path)) > MAX_SOCKET_PATH_BYTES:
            raise ValueError(f"socket path {socket_path} is longer than {MAX_SOCKET_PATH_BYTES} bytes")
        #: The clock the answers come from.
        self.clock = clock
        #: Where the socket is bound, inside the 0700 channel directory.
        self.socket_path = Path(socket_path)
        self._listener: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def start(self) -> None:
        """Bind the socket, mode 0600, and serve in a daemon thread. Refuse a path that is
        already taken: the channel directory is fresh, so anything there is not ours."""
        if self._listener is not None:
            raise RuntimeError("the budget service is already serving")
        if self.socket_path.exists() or self.socket_path.is_symlink():
            raise FileExistsError(f"{self.socket_path} already exists")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
        except BaseException:
            # bind refuses a taken path, so a failed bind leaves whatever is there alone
            listener.close()
            raise
        try:
            self.socket_path.chmod(0o600)
            listener.listen(MAX_CONNECTIONS)
            listener.settimeout(_ACCEPT_POLL_SEC)
        except BaseException:
            listener.close()
            self.socket_path.unlink(missing_ok=True)
            raise
        self._listener = listener
        self._thread = threading.Thread(target=self._serve, args=(listener,), name="hls-budget-service",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop accepting, close the socket and remove it. Connections in flight finish on
        their own threads. A service that never bound its socket leaves the path alone."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(_ACCEPT_POLL_SEC * 10)
        if self._listener is not None:
            self._listener.close()
            self._listener = None
            self.socket_path.unlink(missing_ok=True)

    def _serve(self, listener: socket.socket) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = listener.accept()
            except TimeoutError:
                continue
            except OSError:
                if self._stop.is_set():
                    return
                log.exception("the budget service could not accept a connection")
                self._stop.wait(_ACCEPT_ERROR_PAUSE_SEC)
                continue
            if not self._slots.acquire(blocking=False):
                conn.close()
                continue
            try:
                threading.Thread(target=self._connection, args=(conn,), name="hls-budget-conn",
                                 daemon=True).start()
            except RuntimeError:
                # No thread could start (a process limit in the sandbox). This connection is
                # lost; the loop must live on, or every later get_budget would time out.
                log.exception("the budget service could not start a thread for a connection")
                self._slots.release()
                conn.close()

    def _connection(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(IDLE_TIMEOUT_SEC)
            with conn, conn.makefile("rb") as reader:
                while not self._stop.is_set():
                    line = reader.readline(MAX_LINE_BYTES + 1)
                    if not line:
                        return
                    if len(line.rstrip(b"\n")) > MAX_LINE_BYTES:
                        conn.sendall(json.dumps({"error": "request line too long"}).encode() + b"\n")
                        return
                    if not line.strip():
                        continue
                    conn.sendall(json.dumps(answer(line, self.clock)).encode() + b"\n")
        except OSError:
            return
        finally:
            self._slots.release()
