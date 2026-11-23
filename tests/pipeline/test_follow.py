"""``hls follow``: an instance's events as their writers append them, each once and in time
order, until the instance has ended and no orchestrator holds it; a torn last line is waited
for and never parsed."""

from __future__ import annotations

import json
from pathlib import Path

from hls.pipeline.follow import EventTail, follow, lease_held, log_source
from hls.store.lock import Holder, LockRecord
from hls.store.posix import PosixStore
from tests.viewer.store_fixture import (CREATED, ROLLOUT_STAGES, SUCCEEDED, append_events, append_raw,
                                        rollout_key, write_bytes, write_lease, write_plan)

NOW = CREATED + 1000.0
INSTANCE = rollout_key()
ORCH = f"{INSTANCE}/orchestrator/gen-00000001/events.jsonl"
RUNNER = f"{INSTANCE}/allocations/0/submissions/1/events.jsonl"
VERIFY = f"{INSTANCE}/stages/verify/attempts/1/events.jsonl"
AGENT_BUILD = f"{INSTANCE}/stages/agent_build/attempts/1/events.jsonl"
ENDED = f"{INSTANCE}: succeeded: every stage published"


def _rest(line: str) -> str:
    """A printed event without its date and time."""
    return line.split(" ", 2)[2]


def _follow(store: Path, lines: list[str], *, sleep=lambda sec: None) -> int:
    return follow(INSTANCE, store=PosixStore(store), echo=lines.append, sleep=sleep, clock=lambda: NOW,
                  poll_sec=0.01)


def _end(store: Path, tmp_path: Path) -> None:
    """What the orchestrator does last: the revision that ends the instance, then the lease let go."""
    write_plan(store, INSTANCE, ROLLOUT_STAGES, outcome=SUCCEEDED, revision=2, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW, released=True)


def test_follow_prints_each_event_once_in_time_order(tmp_path):
    store = tmp_path / "store"
    write_plan(store, INSTANCE, ROLLOUT_STAGES, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW)
    append_events(store, ORCH, [{"at": CREATED + 1, "kind": "instance_created", "revision": 1}])
    append_events(store, VERIFY, [{"at": CREATED + 5, "kind": "role_started", "role": "verify"}])
    append_events(store, RUNNER, [{"at": CREATED + 3, "kind": "stage_started", "stage": "verify"}])
    sleeps: list[float] = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        append_events(store, VERIFY, [{"at": CREATED + 7, "kind": "role_ended", "role": "verify",
                                       "exit_code": 0}])
        append_events(store, ORCH, [{"at": CREATED + 9, "kind": "instance_ended", "status": "succeeded"}])
        _end(store, tmp_path)

    lines: list[str] = []
    assert _follow(store, lines, sleep=sleep) == 0
    assert sleeps == [0.01]
    assert [_rest(line) for line in lines[:-1]] == [
        "orchestrator gen 1: instance_created revision=1",
        "allocation 0 submission 1: stage_started stage=verify",
        "stage verify attempt 1: role_started role=verify",
        "stage verify attempt 1: role_ended exit_code=0 role=verify",
        "orchestrator gen 1: instance_ended status=succeeded"]
    assert lines[-1] == ENDED


def test_follow_stops_when_the_instance_ended(tmp_path):
    store = tmp_path / "store"
    write_plan(store, INSTANCE, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW, released=True)
    append_events(store, ORCH, [{"at": CREATED + 9, "kind": "instance_ended", "status": "succeeded"}])
    lines: list[str] = []

    def sleep(sec: float) -> None:
        raise AssertionError("an ended instance is not polled again")

    assert _follow(store, lines, sleep=sleep) == 0
    assert [_rest(lines[0]), lines[1]] == ["orchestrator gen 1: instance_ended status=succeeded", ENDED]


def test_follow_keeps_following_while_an_orchestrator_holds_the_lease(tmp_path):
    store = tmp_path / "store"
    # The outcome is written, and the orchestrator has yet to write its last events and let go.
    write_plan(store, INSTANCE, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW)
    sleeps: list[float] = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        append_events(store, ORCH, [{"at": CREATED + 9, "kind": "failed_records_written", "records": {}}])
        write_lease(store, INSTANCE, heartbeat_at=NOW, released=True)

    lines: list[str] = []
    assert _follow(store, lines, sleep=sleep) == 0
    assert len(sleeps) == 1
    assert [_rest(line) for line in lines[:-1]] == ["orchestrator gen 1: failed_records_written records={}"]


def test_follow_waits_for_a_torn_line(tmp_path):
    store = tmp_path / "store"
    write_plan(store, INSTANCE, ROLLOUT_STAGES, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW)
    whole = json.dumps({"at": CREATED + 5, "kind": "role_started", "role": "verify"}, sort_keys=True)
    append_raw(store, VERIFY, whole[:20])
    sleeps: list[float] = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) == 1:
            # The rest of the line lands, a whole JSON object, but its newline not yet.
            append_raw(store, VERIFY, whole[20:])
        else:
            append_raw(store, VERIFY, "\n")
            _end(store, tmp_path)

    lines: list[str] = []
    assert _follow(store, lines, sleep=sleep) == 0
    assert len(sleeps) == 2
    # Printed once, after its newline; never from the bytes before it.
    assert [_rest(line) for line in lines[:-1]] == ["stage verify attempt 1: role_started role=verify"]


def test_each_event_names_where_its_line_begins_in_its_log(tmp_path):
    """An event's offset is where its line begins in its log, in bytes, a line read torn and then
    whole among them; with its log it names the event once, so another reader of the same logs, as
    the viewer's stream is on each connection, names each event as the first did."""
    lines = [json.dumps({"at": CREATED + n, "kind": "role_started", "role": f"r{n}"}, sort_keys=True) + "\n"
             for n in range(3)]
    append_raw(tmp_path, VERIFY, lines[0] + lines[1][:9])
    tail = EventTail(tmp_path / INSTANCE)
    first = tail.poll()
    assert [e.offset for e in first] == [0]
    append_raw(tmp_path, VERIFY, lines[1][9:] + lines[2])
    append_events(tmp_path, ORCH, [{"at": CREATED + 1.5, "kind": "instance_created", "revision": 1}])
    later = tail.poll()
    assert [(e.source, e.offset) for e in later] == [
        ("stage verify attempt 1", len(lines[0])), ("orchestrator gen 1", 0),
        ("stage verify attempt 1", len(lines[0]) + len(lines[1]))]
    again = EventTail(tmp_path / INSTANCE).poll()
    assert [(e.source, e.offset, e.event) for e in again] == [
        (e.source, e.offset, e.event) for e in first + later]


def test_a_line_still_torn_when_the_instance_ended_is_named_never_parsed(tmp_path):
    store = tmp_path / "store"
    write_plan(store, INSTANCE, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW, released=True)
    append_raw(store, VERIFY, '{"at": 1.0, "kind": "role_star')
    lines: list[str] = []
    assert _follow(store, lines) == 0
    assert lines == ["stage verify attempt 1: its last line was never finished and is not shown", ENDED]


def test_follow_says_once_that_no_orchestrator_holds_a_running_instance(tmp_path):
    store = tmp_path / "store"
    write_plan(store, INSTANCE, ROLLOUT_STAGES, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW - 1000)  # renewed last 1,000 s ago, past its 120 s TTL
    sleeps: list[float] = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        if len(sleeps) == 3:
            _end(store, tmp_path)

    lines: list[str] = []
    assert _follow(store, lines, sleep=sleep) == 0
    assert lines == [f"{INSTANCE}: no orchestrator holds it; hls resume {INSTANCE} takes it over", ENDED]


def test_follow_of_no_instance_exits_2(tmp_path):
    lines: list[str] = []
    assert follow("runs/multitenant_slo/nope", store=PosixStore(tmp_path), echo=lines.append) == 2
    assert lines == [f"no such instance under {tmp_path}"]
    lines.clear()
    assert follow("runs/../etc", store=PosixStore(tmp_path), echo=lines.append) == 2
    assert lines == ["'runs/../etc' is not an instance's store key"]


def test_a_lease_holds_until_released_or_past_its_ttl():
    holder = Holder(host="login01", pid=1, job_id=None, purpose="orchestrator")
    record = LockRecord(generation=1, holder=holder, acquired_at=0.0, heartbeat_at=100.0, ttl_sec=120.0)
    assert lease_held(record, now=220.0) is True
    assert lease_held(record, now=220.5) is False
    assert lease_held(record.model_copy(update={"released": True}), now=100.0) is False
    assert lease_held(None, now=100.0) is False


def test_only_the_instances_logs_are_tailed(tmp_path):
    assert log_source(("orchestrator", "gen-00000002", "events.jsonl")) == "orchestrator gen 2"
    assert log_source(("allocations", "1", "submissions", "3", "events.jsonl")) == "allocation 1 submission 3"
    assert log_source(("stages", "verify", "attempts", "2", "events.jsonl")) == "stage verify attempt 2"
    assert log_source(("stages", "verify", "attempts", "2", "outputs", "events.jsonl")) is None
    assert log_source(("orchestrator", "gen-x", "events.jsonl")) is None
    # A line that is not a JSON object is left out, as read_events leaves it.
    root = tmp_path / "instance"
    path = root / "stages" / "verify" / "attempts" / "1" / "events.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('[1, 2]\nnot json\n{"at": 2.0, "kind": "ok"}\n')
    assert [f.event["kind"] for f in EventTail(root).poll()] == ["ok"]


def test_a_lease_follow_cannot_read_counts_as_held(tmp_path):
    """The instance has an outcome, but its lease is a torn file: follow keeps following until
    the lease reads as let go, rather than leaving before the orchestrator's last events."""
    store = tmp_path / "store"
    write_plan(store, INSTANCE, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_bytes(store, f"{INSTANCE}/lease/gen-00000001.json", b'{"generation": 1, "hol')
    sleeps: list[float] = []

    def sleep(sec: float) -> None:
        sleeps.append(sec)
        write_lease(store, INSTANCE, heartbeat_at=NOW, released=True)

    lines: list[str] = []
    assert _follow(store, lines, sleep=sleep) == 0
    assert (len(sleeps), lines) == (1, [ENDED])


def test_follow_escapes_what_a_contestant_could_have_written(tmp_path):
    """An agent's output reaches its build's events: a newline in it would print a forged line,
    and an escape sequence would drive the operator's terminal. Text that holds either is
    printed as JSON, one line, its control characters escaped."""
    store = tmp_path / "store"
    write_plan(store, INSTANCE, ROLLOUT_STAGES, outcome=SUCCEEDED, tmp=tmp_path)
    write_lease(store, INSTANCE, heartbeat_at=NOW, released=True)
    msg = "exit 1\n2000-01-01 12:00:00 orchestrator gen 1: instance_ended status=succeeded\x1b[2J"
    append_events(store, AGENT_BUILD, [{"at": CREATED + 5, "kind": "build.agent_interrupted", "msg": msg},
                                       {"at": CREATED + 6, "kind": "odd\x1b]0;title\x07", "na\nme": "plain"}])
    lines: list[str] = []
    assert _follow(store, lines) == 0
    shown = lines[:-1]
    assert not any(c in line for line in shown for c in "\n\x1b\x07")
    assert [_rest(line) for line in shown] == [
        f"stage agent_build attempt 1: build.agent_interrupted msg={json.dumps(msg)}",
        'stage agent_build attempt 1: "odd\\u001b]0;title\\u0007" "na\\nme"=plain']
