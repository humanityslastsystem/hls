"""A snapshot's events: ``build.snapshot`` for every snapshot, and ``build.snapshot_over_limit``
beside it for one the commit program found past a limit, of which it kept no bundle. A
snapshot is evidence: neither event stops the build."""

from __future__ import annotations

from hls.agent.commit import CommitRecord
from hls.stages.agent_build import snapshot_events


def _record(status: str, commit: str | None, reason: str | None) -> CommitRecord:
    """The commit program's record of a snapshot, as it writes it."""
    return CommitRecord(status=status, commit=commit, reason=reason, branches=["main"], at=1000.0)


def test_a_snapshot_reports_its_end_and_what_was_pruned():
    events = snapshot_events(3, "exited", 0, _record("ok", "c" * 40, None), ["main-000001.bundle"], scrub=str)
    assert events == [("build.snapshot", {"n": 3, "ending": "exited", "exit_code": 0, "status": "ok",
                                          "commit": "c" * 40, "pruned": ["main-000001.bundle"]})]


def test_a_snapshot_past_a_limit_is_reported_scrubbed_beside_its_end():
    record = _record("over_limit", None, "main's bundle holds more than 4 bytes sk-live")
    events = snapshot_events(2, "exited", 4, record, [], scrub=lambda text: text.replace("sk-live", "<redacted:k>"))
    assert [kind for kind, _ in events] == ["build.snapshot", "build.snapshot_over_limit"]
    assert events[0][1]["status"] == "over_limit"
    assert events[1][1] == {"n": 2, "reason": "main's bundle holds more than 4 bytes <redacted:k>"}


def test_a_snapshot_without_a_record_reports_its_end_alone():
    assert snapshot_events(1, "killed", None, None, [], scrub=str) == [
        ("build.snapshot", {"n": 1, "ending": "killed", "exit_code": None, "status": None, "commit": None,
                            "pruned": []})]
