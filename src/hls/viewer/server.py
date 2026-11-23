"""HLS viewer HTTP layer -- create_app.

Every request needs the session's token, which ``hls view`` makes fresh at each start and prints
in the viewer's URL, as Jupyter does: other users of a shared node can reach the loopback port,
but cannot read anything through it. The URL carries the token as ``?token=``; the response
keeps it in an HTTP-only, same-site cookie named by the viewer's port, which the page's later
requests carry. No page of another origin may read a response: the viewer grants no CORS. The
frontend's dev server proxies ``/api`` to the viewer, so its pages are same-origin; open
``/api/health?token=<token>`` through it once to set the cookie. The files endpoint serves what a contestant wrote so that no browser runs it: under a sandbox, with
no type sniffing, and a type a browser would run sent as plain text."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import re
import secrets
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from hls.obs.record import TERMINAL_STATUSES
from hls.pipeline.follow import EventTail, TailEvent
from hls.viewer.models import FileInfo, PaginatedResponse, RolloutSummary, TraceSummary
from hls.viewer.scanner import INSTANCE_ENDINGS, UNSEALED, RolloutScanner, ViewRefused, check_viewing_root

logger = logging.getLogger(__name__)

#: The query parameter that carries the session's token in the URL ``hls view`` prints.
TOKEN_PARAM = "token"
#: The fewest characters a session token may have; ``new_token`` makes 43.
TOKEN_MIN_CHARS = 32
#: Headers on every response of the files endpoint: the browser runs nothing a file holds (a
#: sandbox without scripts) and takes the type sent, never one it guesses from the bytes.
FILE_HEADERS = {"Content-Security-Policy": "sandbox", "X-Content-Type-Options": "nosniff"}
#: The files endpoint's paths: the listing and each file.
_FILES_PATH = re.compile(r"/api/rollouts/[^/]+/[^/]+/files(/|$)")

# Status values that mean the rollout is already finished: the record schema's, how a store
# instance that ended without a record ended (its plan's outcome), and a published rollout that
# was never sealed. Taken from the schemas rather than
# listed here: a status this set misses makes the rollout list report a seal as one more phase
# change. The scanner's "running" and "stalled" are not finished: a stalled store instance has
# no orchestrator, but its allocations may still run, and hls resume takes it over.
_DONE_STATUSES: frozenset[str] = TERMINAL_STATUSES | INSTANCE_ENDINGS | {UNSEALED}
# Polls of a store instance's logs one events stream makes, half a second apart.
_STORE_EVENT_POLLS = 600
# The last event of a published log, on which the page ends its live follow and reloads the record;
# a store instance's stream sends it once the instance has ended.
_CLOSING_EVENT = "rollout.finished"

# Poll interval for disconnect checks within a sleep period.
_DISCONNECT_POLL_SEC: float = 0.05


def new_token() -> str:
    """A fresh session token: 32 random bytes, URL-safe."""
    return secrets.token_urlsafe(32)


def viewer_url(host: str, port: int, token: str) -> str:
    """The URL that opens the viewer at *host*:*port* in the session of *token*."""
    return f"http://{host}:{port}/?{TOKEN_PARAM}={token}"


def token_cookie(port: int) -> str:
    """The cookie the session's token is kept in, once a request brought it, by the viewer on
    *port*: a browser keys cookies by host alone, so two viewers on one host would otherwise
    evict each other's."""
    return f"hls_viewer_token_{port}"


def _is_token(given: str | None, token: str) -> bool:
    """Whether *given* is the session's *token*, compared in constant time."""
    return given is not None and hmac.compare_digest(given.encode(), token.encode())


class _TokenGate:
    """Serves a request only when it carries the session's token, in the query or the cookie;
    any other request is refused with 403 before it reaches a route. A response to a request
    that brought the token in its query sets the cookie."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        #: The application behind the gate.
        self.app = app
        #: The session's token.
        self.token = token

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        name = token_cookie((scope.get("server") or ("", 0))[1])
        if _is_token(request.query_params.get(TOKEN_PARAM), self.token):
            cookie = f"{name}={self.token}; HttpOnly; Path=/; SameSite=strict"

            async def send_with_cookie(message: Message) -> None:
                if message["type"] == "http.response.start":
                    MutableHeaders(scope=message).append("set-cookie", cookie)
                await send(message)

            await self.app(scope, receive, send_with_cookie)
        elif _is_token(request.cookies.get(name), self.token):
            await self.app(scope, receive, send)
        elif scope["type"] == "http":
            refused = JSONResponse({"detail": "this viewer needs its session token: open the URL hls view printed"},
                                   status_code=403)
            await refused(scope, receive, send)
        else:
            await send({"type": "websocket.close", "code": 1008})


class _FileHeaders:
    """Adds ``FILE_HEADERS`` to every response of the files endpoint, refusals included."""

    def __init__(self, app: ASGIApp) -> None:
        #: The application behind it.
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not _FILES_PATH.match(scope["path"]):
            await self.app(scope, receive, send)
            return

        async def send_guarded(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in FILE_HEADERS.items():
                    headers[name] = value
            await send(message)

        await self.app(scope, receive, send_guarded)


def _page_event(found: TailEvent) -> dict[str, Any]:
    """A store instance's event in the shape the viewer's page reads from a published log:
    ``type`` its kind, ``ts`` its time, ``data`` its other fields and the log it came from as
    ``source``; and ``offset``, where its line begins in that log, which with ``source`` names the
    event on every connection."""
    fields = {k: v for k, v in found.event.items() if k not in ("at", "kind")}
    return {"type": found.event.get("kind", ""), "ts": found.event.get("at"), "offset": found.offset,
            "data": {**fields, "source": found.source}}


async def _store_events(scanner: RolloutScanner, task: str, rollout_id: str, request: Request) -> Any:
    """A store instance's events, merged from every log it keeps (``EventTail``), in the page's
    shape, until the instance has ended; each event once on this connection, each log read from its
    first line, so a reconnecting page is sent what it holds again and keeps each event once by its
    log and offset. Whether it ended is read before the poll, so the events written before its
    orchestrator let go are sent too; then one closing event, which names no line and which the
    page ends its live follow on (``_CLOSING_EVENT``). A log a finalize scan quarantined is left
    out from the poll after its list appears: a stage log of a contestant's code can hold what the
    contestant chose."""
    tail = EventTail(scanner.rollout_dir(task, rollout_id))
    for _ in range(_STORE_EVENT_POLLS):
        if await request.is_disconnected():
            return
        ended = await asyncio.to_thread(scanner.instance_ended, task, rollout_id)
        quarantined = await asyncio.to_thread(scanner.quarantined, task, rollout_id)
        for found in await asyncio.to_thread(tail.poll, quarantined):
            yield f"data: {json.dumps(_page_event(found))}\n\n"
        if ended:
            closing = {"type": _CLOSING_EVENT, "ts": time.time(), "data": {"source": "viewer"}}
            yield f"data: {json.dumps(closing)}\n\n"
            return
        if await _sleep_or_disconnect(0.5, request):
            return


async def _sleep_or_disconnect(seconds: float, request: Request) -> bool:
    """Sleep for *seconds*, waking every _DISCONNECT_POLL_SEC to check disconnect.

    Returns True if the client disconnected before the sleep elapsed.
    """
    elapsed = 0.0
    while elapsed < seconds:
        chunk = min(_DISCONNECT_POLL_SEC, seconds - elapsed)
        await asyncio.sleep(chunk)
        elapsed += chunk
        if await request.is_disconnected():
            return True
    return False


def create_app(
    runs_dir: Path,
    *,
    token: str,
    static_dir: Path | None = None,
) -> FastAPI:
    """Build and return the HLS viewer FastAPI application.

    Parameters
    ----------
    runs_dir:
        Root directory containing ``<task>/<rollout_id>/`` trees. A root that is, lies in or
        holds the store's answers area is refused (``check_viewing_root``, ViewRefused).
    token:
        The session's token (``new_token``), which every request must carry; ValueError for
        one shorter than ``TOKEN_MIN_CHARS``.
    static_dir:
        Optional directory containing built frontend assets (index.html, assets/, fonts/).
        When provided and the directory exists, static assets are mounted and a
        catch-all route serves ``index.html`` for client-side routing.
    """
    if len(token) < TOKEN_MIN_CHARS:
        raise ValueError(f"the viewer's session token must have at least {TOKEN_MIN_CHARS} characters")
    runs_dir = Path(runs_dir)
    check_viewing_root(runs_dir)
    scanner = RolloutScanner(runs_dir)

    app = FastAPI(title="HLS Viewer", docs_url=None, redoc_url=None)
    # The file headers wrap the gate, so its refusals carry them too.
    app.add_middleware(_TokenGate, token=token)
    app.add_middleware(_FileHeaders)

    @app.exception_handler(ViewRefused)
    async def refused(request: Request, exc: ViewRefused) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=403)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Rollout list
    # ------------------------------------------------------------------

    @app.get("/api/rollouts")
    def list_rollouts(
        page: int = Query(1, ge=1),
        page_size: int = Query(100, ge=1, le=200),
    ) -> PaginatedResponse[RolloutSummary]:
        rows = scanner.list_rollouts()
        total = len(rows)
        start = (page - 1) * page_size
        page_rows = rows[start : start + page_size]
        total_pages = math.ceil(total / page_size) if total > 0 else 0
        items = [RolloutSummary(**row) for row in page_rows]
        return PaginatedResponse(
            items=items,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
        )

    # ------------------------------------------------------------------
    # Rollout stream SSE  (MUST be registered before /{task}/{rollout_id})
    # ------------------------------------------------------------------

    @app.get("/api/rollouts/stream")
    def stream_rollouts(request: Request) -> StreamingResponse:
        async def gen() -> Any:
            seen: dict[str, str] = {}
            for _ in range(3600):
                if await request.is_disconnected():
                    return
                rows = await asyncio.to_thread(scanner.list_rollouts)
                for row in rows:
                    key = f"{row.get('task_id')}/{row.get('rollout_id')}"
                    status = row.get("status") or ""
                    phase = row.get("phase") or ""
                    sig = f"{status}|{phase}"
                    if key not in seen:
                        kind = "create"
                    elif seen[key] != sig:
                        kind = "seal" if status in _DONE_STATUSES else "phase"
                    else:
                        continue
                    seen[key] = sig
                    payload = json.dumps(
                        {
                            "kind": kind,
                            "task_id": row.get("task_id"),
                            "rollout_id": row.get("rollout_id"),
                            "status": status,
                            "phase": phase,
                        }
                    )
                    yield f"data: {payload}\n\n"
                if await _sleep_or_disconnect(1.0, request):
                    return

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Single rollout endpoints
    # ------------------------------------------------------------------

    @app.get("/api/rollouts/{task}/{rollout_id}")
    def get_rollout(task: str, rollout_id: str) -> Any:
        rec = scanner.get_rollout(task, rollout_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="rollout not found")
        return rec

    @app.get("/api/rollouts/{task}/{rollout_id}/config")
    def get_rollout_config(task: str, rollout_id: str) -> Any:
        try:
            config = scanner.get_config(task, rollout_id)
        except ViewRefused:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if config is None:
            raise HTTPException(status_code=404, detail="config not found")
        return config

    @app.get("/api/rollouts/{task}/{rollout_id}/trajectory")
    def get_trajectory(task: str, rollout_id: str) -> Any:
        rec = scanner.get_rollout(task, rollout_id)
        if rec is None:
            raise HTTPException(status_code=404, detail="rollout not found")
        traj = scanner.get_trajectory(task, rollout_id)
        if traj is None:
            raise HTTPException(status_code=404, detail="trajectory not found")
        return traj

    # ------------------------------------------------------------------
    # Trace endpoints
    # ------------------------------------------------------------------

    @app.get("/api/rollouts/{task}/{rollout_id}/evals/{eval_id}/trace")
    def get_trace_page(
        task: str,
        rollout_id: str,
        eval_id: str,
        offset: int = Query(0, ge=0),
        limit: int = Query(100, ge=1),
        only: str | None = Query(None),
    ) -> Any:
        page = scanner.trace_page(task, rollout_id, eval_id, offset=offset, limit=limit, only=only)
        return page

    @app.get("/api/rollouts/{task}/{rollout_id}/evals/{eval_id}/trace/summary")
    def get_trace_summary(task: str, rollout_id: str, eval_id: str) -> Any:
        summary = scanner.trace_summary(task, rollout_id, eval_id)
        if summary is None:
            return TraceSummary(total=0)
        return summary

    @app.get("/api/rollouts/{task}/{rollout_id}/evals/{eval_id}/trace/row/{request_id}")
    def get_trace_row(task: str, rollout_id: str, eval_id: str, request_id: str) -> Any:
        row = scanner.trace_row(task, rollout_id, eval_id, request_id)
        if row is None:
            raise HTTPException(status_code=404, detail="trace row not found")
        return row

    @app.get("/api/rollouts/{task}/{rollout_id}/evals/{eval_id}/timeline")
    def get_timeline(task: str, rollout_id: str, eval_id: str) -> Any:
        payload = scanner.timeline(task, rollout_id, eval_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="trace not found")
        return payload

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    @app.get("/api/rollouts/{task}/{rollout_id}/files")
    def list_files(task: str, rollout_id: str) -> list[FileInfo]:
        try:
            rdir = scanner.rollout_dir(task, rollout_id)
        except ViewRefused:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not rdir.is_dir():
            raise HTTPException(status_code=404, detail="rollout not found")
        raw = scanner.list_files(task, rollout_id)
        return [FileInfo(**entry) for entry in raw]

    @app.get("/api/rollouts/{task}/{rollout_id}/files/{file_path:path}")
    def read_file(task: str, rollout_id: str, file_path: str) -> Any:
        try:
            full_path, media_type = scanner.resolve_file(task, rollout_id, file_path)
        except ViewRefused:
            raise
        except ValueError as exc:
            msg = str(exc)
            if "not found" in msg:
                raise HTTPException(status_code=404, detail=msg) from exc
            raise HTTPException(status_code=400, detail=msg) from exc

        # Stream from disk (not read fully into memory) -- a multi-hundred-MB
        # trace.json must not be buffered per request. resolve_file() has already
        # validated file_path is confined to the rollout dir; hand that same safe
        # path straight to FileResponse without any further path manipulation.
        return FileResponse(full_path, media_type=media_type)

    # ------------------------------------------------------------------
    # Events SSE (a store instance's logs, merged)
    # ------------------------------------------------------------------

    @app.get("/api/rollouts/{task}/{rollout_id}/events")
    def stream_events(task: str, rollout_id: str, request: Request) -> StreamingResponse:
        # A published rollout's log is never appended to again; it is among its files.
        try:
            store = scanner.is_store(task, rollout_id)
        except ViewRefused:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not store:
            raise HTTPException(status_code=404, detail="no event stream: not a store instance")
        return StreamingResponse(_store_events(scanner, task, rollout_id, request),
                                 media_type="text/event-stream")

    # ------------------------------------------------------------------
    # Static frontend (optional)
    # ------------------------------------------------------------------

    if static_dir is not None:
        sd = Path(static_dir)
        if sd.exists():
            assets_dir = sd / "assets"
            if assets_dir.exists():
                app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

            fonts_dir = sd / "fonts"
            if fonts_dir.exists():
                app.mount("/fonts", StaticFiles(directory=str(fonts_dir)), name="fonts")

            index_html = sd / "index.html"

            @app.get("/{path:path}")
            def spa_catch_all(path: str) -> FileResponse:
                if path.startswith("api/"):
                    raise HTTPException(status_code=404, detail="not found")
                return FileResponse(str(index_html))

    return app
