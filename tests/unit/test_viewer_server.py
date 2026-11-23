"""Tests for hls.viewer.server -- create_app HTTP layer."""

from __future__ import annotations

import threading
from pathlib import Path

import httpx
import pytest
import uvicorn
from fastapi.testclient import TestClient

from hls.viewer.server import create_app, token_cookie

# Committed fixture rollout (a real sealed sample_task run) so these HTTP tests are
# deterministic on CI -- the repo-root runs/ is gitignored (local-only).
RUNS = Path(__file__).resolve().parents[1] / "fixtures" / "sample_runs"

SAMPLE_TASK = "sample_task"
SAMPLE_ID = "20260602T075745Z-4363e2"
EVAL_ID = "terminal"
#: The session token these tests' viewers run with.
TOKEN = "unit-session-token-" + "0" * 24


def _client(runs: Path, **kwargs) -> TestClient:
    """A client that holds the session's token, as a browser does once it opened the printed URL."""
    return TestClient(create_app(runs, token=TOKEN, **kwargs), cookies={token_cookie(80): TOKEN})


@pytest.fixture
def client() -> TestClient:
    return _client(RUNS)


# ---------------------------------------------------------------------------
# /api/health
# ---------------------------------------------------------------------------


def test_health(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /api/config
# ---------------------------------------------------------------------------


def test_the_viewer_serves_no_config(client: TestClient) -> None:
    assert client.get("/api/config").status_code == 404


# ---------------------------------------------------------------------------
# /api/rollouts
# ---------------------------------------------------------------------------


def test_list_rollouts_shape(client: TestClient) -> None:
    resp = client.get("/api/rollouts")
    assert resp.status_code == 200
    data = resp.json()
    assert "items" in data
    assert "total" in data


def test_list_rollouts_pagination(client: TestClient) -> None:
    resp = client.get("/api/rollouts?page=1&page_size=2")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) <= 2
    assert data["page_size"] == 2


def test_list_rollouts_page_size_capped(client: TestClient) -> None:
    # page_size > 200 should return 422
    resp = client.get("/api/rollouts?page_size=201")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}
# ---------------------------------------------------------------------------


def test_get_real_rollout_200(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}")
    assert resp.status_code == 200


def test_get_missing_rollout_404(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/nonexistent-id")
    assert resp.status_code == 404


def test_real_rollout_episode_comes_from_its_sealed_grading(client: TestClient) -> None:
    """This reads evals/terminal/grading.json -- the one rule the reader has. Served off
    the inline episodes these records also carry, it would pass here and lose every
    score on a record without them."""
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}")
    episodes = resp.json()["evals"]
    assert [e["eval_id"] for e in episodes] == [EVAL_ID]
    assert episodes[0]["score"]["value"] == pytest.approx(0.5333196170637383)


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/trajectory
# ---------------------------------------------------------------------------


def test_get_trajectory_200_or_404(client: TestClient) -> None:
    # SAMPLE_ID is a sealed status:ok rollout with build/trajectory.atif.json -- must be 200.
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/trajectory")
    assert resp.status_code == 200
    body = resp.json()
    assert body is None or isinstance(body, dict)


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/evals/{eval_id}/trace
# ---------------------------------------------------------------------------


def test_trace_page_200(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/evals/{EVAL_ID}/trace?offset=0&limit=1")
    assert resp.status_code == 200
    data = resp.json()
    assert "rows" in data
    assert isinstance(data["rows"], list)


def test_trace_page_has_total(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/evals/{EVAL_ID}/trace")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data
    assert "offset" in data
    assert "limit" in data


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/evals/{eval_id}/trace/summary
# ---------------------------------------------------------------------------


def test_trace_summary_200(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/evals/{EVAL_ID}/trace/summary")
    assert resp.status_code == 200
    data = resp.json()
    assert "total" in data


def test_trace_summary_missing_eval_not_500(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/evals/nonexistent/trace/summary")
    # must be 200 (empty) or 404, never 500
    assert resp.status_code in (200, 404)


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/files
# ---------------------------------------------------------------------------


def test_list_files_200(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/files")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, list)
    if data:
        assert "path" in data[0]
        assert "name" in data[0]
        assert "is_dir" in data[0]


def test_list_files_missing_rollout_404(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/nonexistent-id/files")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/files/{file_path}
# ---------------------------------------------------------------------------


def test_read_rollout_json_file(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/files/rollout.json")
    assert resp.status_code == 200


def test_read_missing_file_404(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/files/does_not_exist.json")
    assert resp.status_code == 404


def test_read_file_traversal_400(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/files/../../etc/passwd")
    # Either blocked with 400 (caught ValueError) or 404 (FastAPI path normalisation);
    # must never be 200 with a real /etc/passwd.
    assert resp.status_code in (400, 404, 422)


# ---------------------------------------------------------------------------
# /api/rollouts/stream -- route ordering guard
# ---------------------------------------------------------------------------


def test_stream_route_not_treated_as_task_name() -> None:
    """/api/rollouts/stream resolves to the stream handler (not captured as a task),
    emits SSE, and the connection is closable without hanging."""
    app = create_app(RUNS, token=TOKEN)

    # Run the app on a real TCP port so that closing the HTTP connection
    # is visible to the server and triggers the disconnect path.
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

    # Start uvicorn in a background thread; it signals `started` once the port
    # is bound so we know what port to connect to.
    started = threading.Event()
    original_startup = server.startup

    async def _patched_startup(sockets: object = None) -> None:
        await original_startup(sockets=sockets)
        started.set()

    server.startup = _patched_startup  # type: ignore[method-assign]

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started.wait(timeout=10)

    # Determine the bound port.
    port = server.servers[0].sockets[0].getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"

    try:
        # trust_env=False: this test dials its OWN ephemeral 127.0.0.1 server; an
        # ambient http_proxy (whose no_proxy lists "localhost" but not
        # "127.0.0.1") would otherwise detour the request through the proxy -> 503.
        with httpx.Client(base_url=base_url, timeout=10.0, trust_env=False,
                          cookies={token_cookie(port): TOKEN}) as hc:
            with hc.stream("GET", "/api/rollouts/stream") as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers.get("content-type", "")
                # Read until the first data: line, then close.
                # Closing the real TCP connection signals disconnect to the generator.
                for line in resp.iter_lines():
                    if line.startswith("data:"):
                        break
    finally:
        server.should_exit = True
        thread.join(timeout=5)


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/events -- a store instance's alone
# ---------------------------------------------------------------------------


def test_events_of_a_published_rollout_404(client: TestClient) -> None:
    """Nothing appends to a published rollout's events.jsonl, so it has no event
    stream; the log is among its files."""
    assert client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/events").status_code == 404
    assert client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/files/events.jsonl").status_code == 200


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/config -- malformed JSON -> 400
# ---------------------------------------------------------------------------


def test_get_rollout_config_malformed_json_400(tmp_path: Path) -> None:
    """Malformed config.json must yield 400, not 500."""
    task = "bad_task"
    rid = "bad-rollout"
    rollout_dir = tmp_path / task / rid
    rollout_dir.mkdir(parents=True)
    (rollout_dir / "config.json").write_text("{ not valid json }")
    c = _client(tmp_path)
    resp = c.get(f"/api/rollouts/{task}/{rid}/config")
    assert resp.status_code == 400
    assert "not valid JSON" in resp.json().get("detail", "")


# ---------------------------------------------------------------------------
# /api/rollouts/{task}/{rollout_id}/evals/{eval_id}/timeline
# ---------------------------------------------------------------------------


def test_timeline_200(client: TestClient) -> None:
    resp = client.get(f"/api/rollouts/{SAMPLE_TASK}/{SAMPLE_ID}/evals/{EVAL_ID}/timeline")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) >= {"t0", "duration", "tenants", "requests"}
    assert isinstance(data["tenants"], list)
    assert isinstance(data["requests"], list)
    if data["requests"]:
        assert set(data["requests"][0]) >= {"ti", "sid", "turn", "send", "recv", "err"}


def test_timeline_404_unknown(client: TestClient) -> None:
    resp = client.get("/api/rollouts/nope/nope/evals/terminal/timeline")
    assert resp.status_code == 404


def test_every_sealed_status_counts_as_finished() -> None:
    """The rollout list's stream calls a status change a seal when the new status says the
    rollout is done. A hand-listed set silently drops a status added later, and the list
    calls that seal a phase change. A store instance that ended without a record (void,
    cancelled) is done too, and so is a published rollout that was never sealed."""
    from hls.obs.record import TERMINAL_STATUSES
    from hls.viewer.scanner import INSTANCE_ENDINGS, UNSEALED
    from hls.viewer.server import _DONE_STATUSES

    assert _DONE_STATUSES == TERMINAL_STATUSES | INSTANCE_ENDINGS | {UNSEALED}
    assert INSTANCE_ENDINGS == {"succeeded", "failed", "void", "cancelled"}
    assert {"invalid_artifact", "void"} <= _DONE_STATUSES
    assert not {"running", "stalled"} & _DONE_STATUSES
