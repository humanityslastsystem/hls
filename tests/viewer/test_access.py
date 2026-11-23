"""Who the viewer serves and what it never serves: every request needs the session's token, and
no page of another origin reads a response; a file a finalize scan quarantined is neither
listed nor served, nor read into a view or the event stream; a file is served so that no
browser runs what it holds; and the store's answers area is never reachable, from a viewing
root that is or holds it or through a path below one."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hls.core.trace import Trace
from hls.pipeline.plan import Outcome
from hls.substrate.base import FaultRecord
from hls.viewer import scanner as scanner_mod
from hls.viewer.scanner import ACTIVE_SUFFIXES, RolloutScanner, ViewRefused
from hls.viewer.server import create_app, token_cookie
from tests.trace.traces import drive_trace
from tests.viewer.published_fixture import published_runs
from tests.viewer.store_fixture import (CLIENT_PORT, CREATED, GRADING, ROLLOUT, ROLLOUT_STAGES, TASK, TOKEN,
                                        append_events, dumps, grading_key, grading_record, rollout_key,
                                        rollout_record, sealed_rollout, viewer_client, write_bytes, write_plan)

TRACE = drive_trace()
GRADING_TRACE = Trace(events=drive_trace().events[:5])
#: What a contestant's server echoed the arm's relay key into, under the rollout's directory: a
#: log, a file whose own name holds the key, and a log of the rollout's grading.
LEAK = "stages/deploy_drive/attempts/1/logs/artifact.stdout"
NAMED = "stages/deploy_drive/attempts/1/logs/sk-ARM-ROUTE-SECRET.txt"
GRADING_LEAK = f"gradings/{GRADING}/stages/deploy_drive/attempts/1/logs/artifact.stdout"
#: A log beside them that holds no value.
CLEAN = "stages/deploy_drive/attempts/1/logs/artifact.stderr"
#: Where finalize writes the list, under an instance's directory.
QUARANTINE = "stages/finalize/attempts/1/outputs/quarantine.json"


def _quarantined(tmp_path: Path, *, extra: dict[str, list[str]] | None = None) -> Path:
    """A sealed rollout and its grading whose scans quarantined the files above. Each list holds
    the files' store keys as finalize writes them: scrubbed, so a key that holds the value holds
    its marker instead."""
    store = tmp_path / "store"
    sealed_rollout(store, tmp_path, trace=TRACE, grading_trace=GRADING_TRACE)
    key = rollout_key()
    for rel in (LEAK, NAMED, GRADING_LEAK):
        write_bytes(store, f"{key}/{rel}", b"relay key: sk-ARM-ROUTE-SECRET\n")
    write_bytes(store, f"{key}/{CLEAN}", b"listening on 0.0.0.0:8000\n")
    files = {f"{key}/{LEAK}": ["ARM_RELAY_KEY"],
             f"{key}/stages/deploy_drive/attempts/1/logs/<redacted:ARM_RELAY_KEY>.txt": ["ARM_RELAY_KEY"],
             **(extra or {})}
    write_bytes(store, f"{key}/{QUARANTINE}", json.dumps({"files": files}, indent=2).encode())
    write_bytes(store, f"{grading_key()}/{QUARANTINE}",
                json.dumps({"files": {f"{grading_key()}/stages/deploy_drive/attempts/1/logs/artifact.stdout":
                                      ["ARM_RELAY_KEY"]}}).encode())
    return store


def test_every_request_needs_the_session_token(tmp_path):
    published_runs(tmp_path, TRACE)
    app = create_app(tmp_path, token=TOKEN)
    anonymous = TestClient(app)
    for url in ("/api/health", "/api/rollouts", f"/api/rollouts/{TASK}/{ROLLOUT}",
                f"/api/rollouts/{TASK}/{ROLLOUT}/files", f"/api/rollouts/{TASK}/{ROLLOUT}/files/rollout.json", "/"):
        assert anonymous.get(url).status_code == 403, url
    for wrong in (TOKEN[:-1] + "1", TOKEN[:-1], "", TOKEN + "0"):
        assert anonymous.get("/api/rollouts", params={"token": wrong}).status_code == 403, wrong
        assert TestClient(app, cookies={token_cookie(CLIENT_PORT): wrong}).get("/api/rollouts").status_code == 403
    # The URL hls view prints carries the token; the response keeps it in a cookie, which the
    # page's later requests carry.
    opened = anonymous.get("/api/rollouts", params={"token": TOKEN})
    assert opened.status_code == 200
    assert anonymous.get(f"/api/rollouts/{TASK}/{ROLLOUT}/files/rollout.json").status_code == 200


def test_a_token_too_short_to_guess_is_refused(tmp_path):
    with pytest.raises(ValueError, match="token"):
        create_app(tmp_path, token="short")


def test_a_file_a_scan_quarantined_is_neither_listed_nor_served(tmp_path):
    store = _quarantined(tmp_path)
    client = viewer_client(store / "runs")
    base = f"/api/rollouts/{TASK}/{ROLLOUT}/files"
    for rel in (LEAK, NAMED, GRADING_LEAK):
        refused = client.get(f"{base}/{rel}")
        assert refused.status_code == 403, rel
        assert "sk-ARM" not in refused.text
    assert client.get(f"{base}/{CLEAN}").status_code == 200
    listed = {entry["path"] for entry in client.get(base).json()}
    assert not {LEAK, NAMED, GRADING_LEAK} & listed
    assert {CLEAN, QUARANTINE} <= listed
    # A link to a quarantined file is refused, and left out of the listing, as the file is.
    (store / rollout_key() / "stages/deploy_drive/attempts/1/alias").symlink_to(store / rollout_key() / LEAK)
    assert client.get(f"{base}/stages/deploy_drive/attempts/1/alias").status_code == 403
    assert "stages/deploy_drive/attempts/1/alias" not in {entry["path"] for entry in client.get(base).json()}
    with pytest.raises(ViewRefused):
        RolloutScanner(store / "runs").resolve_file(TASK, ROLLOUT, LEAK)


def test_a_quarantined_trace_reads_as_no_trace(tmp_path):
    chunk = "stages/deploy_drive/attempts/1/outputs/trace/chunk-000000.jsonl.gz"
    store = _quarantined(tmp_path, extra={f"{rollout_key()}/{chunk}": ["ARM_RELAY_KEY"]})
    scanner = RolloutScanner(store / "runs")
    assert scanner.trace_summary(TASK, ROLLOUT, "terminal") is None
    assert scanner.trace_page(TASK, ROLLOUT, "terminal").total == 0
    # The grading's own drive was not quarantined.
    assert scanner.trace_summary(TASK, ROLLOUT, GRADING).total == len(GRADING_TRACE.events)


def test_an_unreadable_quarantine_list_refuses_every_file(tmp_path):
    store = _quarantined(tmp_path)
    write_bytes(store, f"{grading_key()}/{QUARANTINE}", b'{"files": {"runs/')
    client = viewer_client(store / "runs")
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/files/{CLEAN}").status_code == 403
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/files").json() == []


def test_a_viewing_root_that_is_or_holds_the_answers_area_is_refused(tmp_path):
    store = tmp_path / "store"
    sealed_rollout(store, tmp_path, trace=TRACE, grading_trace=GRADING_TRACE)
    write_bytes(store, f"oracle/{TASK}/{'a' * 64}/w1.1.json", b'{"answers": "withheld"}')
    for root in (store, store / "oracle", store / "oracle" / TASK):
        with pytest.raises(ViewRefused, match="answers"):
            create_app(root, token=TOKEN)
    assert viewer_client(store / "runs").get("/api/rollouts").status_code == 200


def test_the_answers_area_below_a_viewing_root_is_never_served(tmp_path):
    """Pointed above the store, the viewer reaches the area only through a rollout directory or
    a file path, and refuses both."""
    above = tmp_path / "above"
    store = above / "site" / "store"
    sealed_rollout(store, tmp_path, trace=TRACE, grading_trace=GRADING_TRACE)
    answers = f"oracle/{TASK}/{'a' * 64}/w1.1.json"
    write_bytes(store, answers, b'{"answers": "withheld"}')
    client = viewer_client(above)
    assert client.get(f"/api/rollouts/site/store/files/{answers}").status_code == 403
    assert not any(entry["path"].startswith("oracle")
                   for entry in client.get("/api/rollouts/site/store/files").json())
    client = viewer_client(above / "site")
    assert client.get(f"/api/rollouts/store/oracle/files/{TASK}/{'a' * 64}/w1.1.json").status_code == 403
    with pytest.raises(ViewRefused):
        RolloutScanner(above / "site").get_rollout("store", "oracle")


def test_no_other_origin_reads_the_api(tmp_path):
    """A page on another port of the same host, which any user of a shared node can serve, gets
    no CORS grant, nor does any other origin: the frontend's dev server proxies the API, so its
    pages are of the viewer's own origin."""
    published_runs(tmp_path, TRACE)
    client = viewer_client(tmp_path)
    for origin in ("http://localhost:5173", "http://localhost:9999", "http://127.0.0.1:8081", "null"):
        assert "access-control-allow-origin" not in client.get("/api/rollouts", headers={"Origin": origin}).headers
        asked = client.options("/api/rollouts", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
        assert "access-control-allow-origin" not in asked.headers


#: Files a contestant could name, each with what a browser would run: the active types are sent
#: as text; the rest keep their own type.
ACTIVE = {"page.html": "<script>fetch('/api/rollouts')</script>", "page.htm": "<script>1</script>",
          "logo.svg": '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>',
          "feed.xml": '<?xml-stylesheet href="x.xsl"?><a/>', "doc.xhtml": "<html><script>1</script></html>",
          "app.js": "alert(1)", "mod.mjs": "export default 1",
          # Not among the endings, but named an HTML or XML type.
          "page.shtml": "<script>1</script>", "style.xsl": "<xsl:stylesheet/>"}
INERT = {"notes.txt": ("plain", "text/plain"), "data.json": ("{}", "application/json")}
#: Where the contestant's files lie in the instance.
SOURCE = "stages/package/attempts/1/outputs/source"


def test_a_contestants_active_file_is_served_inert(tmp_path):
    """Every response of the files endpoint forbids scripts (a sandbox) and type sniffing, and a
    type a browser runs is sent as plain text."""
    store = tmp_path / "store"
    sealed_rollout(store, tmp_path, trace=TRACE, grading_trace=GRADING_TRACE)
    for name, body in {**ACTIVE, **{n: b for n, (b, _) in INERT.items()}}.items():
        write_bytes(store, f"{rollout_key()}/{SOURCE}/{name}", body.encode())
    client = viewer_client(store / "runs")
    base = f"/api/rollouts/{TASK}/{ROLLOUT}/files"

    def guarded(response) -> bool:
        return (response.headers.get("content-security-policy"), response.headers.get("x-content-type-options")) \
            == ("sandbox", "nosniff")

    for name, body in ACTIVE.items():
        served = client.get(f"{base}/{SOURCE}/{name}")
        assert (served.status_code, served.text) == (200, body), name
        assert served.headers["content-type"].split(";")[0] == "text/plain", name
        assert guarded(served), name
    for name, (body, media) in INERT.items():
        served = client.get(f"{base}/{SOURCE}/{name}")
        assert served.headers["content-type"].split(";")[0] == media and guarded(served), name
    # The listing, a missing file, a path out of the rollout: every response of the endpoint.
    escape = client.get(f"{base}/%2e%2e/%2e%2e/%2e%2e/%2e%2e/etc/passwd")
    assert escape.status_code == 400
    anonymous = TestClient(create_app(store / "runs", token=TOKEN)).get(f"{base}/{SOURCE}/page.html")
    assert anonymous.status_code == 403
    for response in (client.get(base), client.get(f"{base}/{SOURCE}/missing.html"), escape, anonymous):
        assert guarded(response), response.url
    assert RolloutScanner(store / "runs").resolve_file(TASK, ROLLOUT, f"{SOURCE}/logo.svg")[1] == "text/plain"


def test_an_active_ending_is_served_as_text_whatever_the_type_table_says(tmp_path, monkeypatch):
    """The system's type table may not know an ending, or name it otherwise; the endings alone
    send such a file as text."""
    store = tmp_path / "store"
    sealed_rollout(store, tmp_path, trace=TRACE, grading_trace=GRADING_TRACE)
    endings = sorted(ACTIVE_SUFFIXES)
    for ending in endings:
        write_bytes(store, f"{rollout_key()}/{SOURCE}/file{ending}", b"<script>1</script>")
    monkeypatch.setattr(scanner_mod.mimetypes, "guess_type", lambda name, strict=True: ("image/png", None))
    scanner = RolloutScanner(store / "runs")
    assert [scanner.resolve_file(TASK, ROLLOUT, f"{SOURCE}/file{e}")[1] for e in endings] == ["text/plain"] * len(endings)


def test_each_viewer_on_a_host_keeps_its_own_cookie(tmp_path):
    """A browser keys cookies by host alone, so the cookie is named by the viewer's port: a second
    viewer on the same host neither evicts the first one's nor accepts it."""
    published_runs(tmp_path, TRACE)
    app = create_app(tmp_path, token=TOKEN)
    opened = TestClient(app, base_url="http://localhost:8081").get("/api/rollouts", params={"token": TOKEN})
    assert opened.headers["set-cookie"].startswith(f"{token_cookie(8081)}={TOKEN};")
    assert token_cookie(8081) != token_cookie(8082)
    assert TestClient(app, base_url="http://localhost:8081",
                      cookies={token_cookie(8082): TOKEN}).get("/api/rollouts").status_code == 403
    assert TestClient(app, base_url="http://localhost:8081",
                      cookies={token_cookie(8081): TOKEN}).get("/api/rollouts").status_code == 200


def test_a_quarantined_event_log_is_left_out_of_the_event_stream(tmp_path):
    """The drive stage's own log can carry what the contestant chose (a cleared path's name);
    once finalize quarantined it, a stream of the instance leaves it out."""
    store = _quarantined(tmp_path, extra={f"{rollout_key()}/stages/deploy_drive/attempts/1/events.jsonl": ["K"]})
    key = rollout_key()
    append_events(store, f"{key}/stages/deploy_drive/attempts/1/events.jsonl",
                  [{"at": CREATED + 2, "kind": "drive.artifact_cleared", "paths": ["/w/sk-LEAK"]}])
    append_events(store, f"{key}/stages/verify/attempts/1/events.jsonl",
                  [{"at": CREATED + 5, "kind": "role_started", "role": "verify"}])
    append_events(store, f"{key}/orchestrator/gen-00000001/events.jsonl",
                  [{"at": CREATED + 9, "kind": "instance_ended", "status": "succeeded"}])
    client = viewer_client(store / "runs")
    body = client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/events").text
    sent = [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]
    assert "sk-LEAK" not in body
    assert [e["type"] for e in sent] == ["role_started", "instance_ended", "rollout.finished"]
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/files/stages/deploy_drive/attempts/1/events.jsonl"
                      ).status_code == 403


def test_quarantined_records_outputs_and_trajectory_are_not_read(tmp_path):
    """Finalize's records, the verdict and the agent's trajectory, once quarantined, are not read
    into any view: the record is made from the plan, its own grading is left out, and the config,
    the trajectory and the verdict's figures are absent."""
    key = rollout_key()
    trajectory = "stages/agent_build/attempts/1/outputs/transcript/trajectory.json"
    store = _quarantined(tmp_path, extra={f"{key}/{rel}": ["K"] for rel in (
        "stages/finalize/attempts/1/outputs/rollout.json", "stages/finalize/attempts/1/outputs/grading.json",
        "stages/verify/attempts/1/outputs/verdict.json", trajectory)})
    write_bytes(store, f"{key}/{trajectory}", json.dumps({"steps": [{"secret": "sk-LEAK"}]}).encode())
    scanner = RolloutScanner(store / "runs")
    rec = scanner.get_rollout(TASK, ROLLOUT)
    assert (rec["status"], rec["build"], [e["eval_id"] for e in rec["evals"]]) == ("succeeded", None, [GRADING])
    assert scanner.get_config(TASK, ROLLOUT) is None
    assert scanner.get_trajectory(TASK, ROLLOUT) is None
    summary = scanner.trace_summary(TASK, ROLLOUT, "terminal")
    assert (summary.total, summary.gate_passed) == (len(TRACE.events), None)
    client = viewer_client(store / "runs")
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/config").status_code == 404
    assert client.get(f"/api/rollouts/{TASK}/{ROLLOUT}/trajectory").status_code == 404


def test_a_quarantined_failed_record_is_not_read(tmp_path):
    """The orchestrator's copies under ``failed/`` are judged as finalize's records are."""
    store = tmp_path / "store"
    key = rollout_key()
    fault = FaultRecord(fault_class="contestant", reason="contestant role wrote past its disk cap", retryable=False)
    write_plan(store, key, ROLLOUT_STAGES, tmp=tmp_path,
               outcome=Outcome(status="failed", reason=fault.reason, at=CREATED + 700, fault=fault))
    write_bytes(store, f"{key}/failed/rollout.json", dumps(rollout_record()))
    write_bytes(store, f"{key}/failed/grading.json", dumps(grading_record("terminal", 0.0)))
    write_bytes(store, f"{key}/{QUARANTINE}", json.dumps({"files": {f"{key}/failed/grading.json": ["K"]}}).encode())
    rec = RolloutScanner(store / "runs").get_rollout(TASK, ROLLOUT)
    assert (rec["status"], rec["evals"]) == ("ok", [])
