from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import uvicorn
from fastapi import FastAPI

import hls.spec.datasets as datasets_mod
import hls.store.cache as cache_mod
import hls.store.manifest as manifest_mod
from hls.artifact.artifact_client import HTTPArtifactClient
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse, Trace, TraceEvent
from hls.drive import role
from hls.drive import stream as stream_mod
from hls.drive.stream import TraceStreamRecord
from hls.pipeline.pipelines import OPEN_DRIVE_SEC, REQUEST_TIMEOUT_SEC, drive_wall, drive_wall_sec
from hls.spec.datasource import Count, DataSource, SplitSpec
from hls.spec.drivers import ABANDONED_ERROR, ABORTED_ERROR, WALL_ERROR, Driver, TrajectorySessionDriver
from hls.spec.workload import DeterministicArrival, NoDelay, NoSLO, Uncapped, Workload
from hls.store.manifest import ReportedOutput, read_outputs_report
from hls.taskref import TaskRef, load_task
from hls.trace import store
from hls.trace.store import read_end, read_trace, write_trace
from tests.support.inline_dataset import InlineMultiTurnDataset
from tests.support.virtual_clock import virtual_clock

SAMPLE = TaskRef(module="tasks.sample_task.task")


def _rows(n: int) -> list[dict]:
    return [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4, "subject": "x", "source_index": i}
            for i in range(n)]


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    """sample_task's single tenant's rows where the driver reads them, /data/tenants/t0."""
    for split, n in (("dev", 5), ("test", 10)):
        d = tmp_path / "data" / "tenants" / "t0" / split
        d.mkdir(parents=True)
        (d / "rows.json").write_text(json.dumps(_rows(n)))
    monkeypatch.setattr(datasets_mod, "MOUNT_ROOT", tmp_path / "data")
    return tmp_path / "data"


def _app(seen: list[dict]) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def _health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        seen.append(body)
        return {"id": "c", "choices": [{"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop",
                                        "token_ids": [33]}], "usage": {"completion_tokens": 1}}

    return app


@pytest.fixture
def server():
    """A real HTTP server on a free local port, speaking sample_task's chat contract."""
    seen: list[dict] = []
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(_app(seen), log_level="warning"))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", seen
    srv.should_exit = True
    thread.join(5)
    sock.close()


# A server process that answers its first three requests and is then killed with the rest in
# flight, as a contestant's server that crashes mid-drive is: its sockets close at once.
_DYING = '''
import os, signal, sys
import uvicorn
from fastapi import FastAPI

app = FastAPI()
seen = []


@app.post("/v1/chat/completions")
async def chat(body: dict):
    seen.append(body)
    if len(seen) > 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return {"id": "c", "choices": [{"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop",
                                    "token_ids": [33]}], "usage": {"completion_tokens": 1}}


uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="warning")
'''


@pytest.fixture
def dying_server(tmp_path):
    script = tmp_path / "dying.py"
    script.write_text(_DYING)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    proc = subprocess.Popen([sys.executable, str(script), str(port)])
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
            break
        except OSError:
            time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", proc
    if proc.poll() is None:
        proc.kill()
    proc.wait(10)


def _ref_file(tmp_path: Path, ref: TaskRef = SAMPLE) -> Path:
    path = tmp_path / "task-ref.json"
    path.write_text(ref.model_dump_json())
    return path


def _report(tmp_path: Path) -> Path:
    """Where the stage tells the role to report its outputs: a file in the role's own directory."""
    path = tmp_path / "role" / "outputs.json"
    path.parent.mkdir(exist_ok=True)
    return path


def _stream_report(tmp_path: Path) -> Path:
    """Where the stage tells the role to write its trace writer's record: a file in the role's
    own directory."""
    path = tmp_path / "role" / "stream.json"
    path.parent.mkdir(exist_ok=True)
    return path


#: The bytes of events the room ``_room`` makes takes: more than a sample drive writes.
ROOM_EVENT_BYTES = 2**20


def _room(local: Path) -> Path:
    """*local*, made with a trace room as the stage makes it before the server starts."""
    local.mkdir(parents=True)
    store.make_room(local, event_bytes=ROOM_EVENT_BYTES)
    return local


def _args(tmp_path: Path, url: str, out: Path, *extra: str, ref: TaskRef = SAMPLE,
          grace: float = REQUEST_TIMEOUT_SEC, local: Path | None = None) -> list[str]:
    if local is None:
        local = tmp_path / "role" / "local"
        if not local.exists():
            _room(local)
    return ["--task-ref", str(_ref_file(tmp_path, ref)), "--artifact-url", url, "--local", str(local),
            "--out", str(out), "--outputs-out", str(_report(tmp_path)), "--stream-out", str(_stream_report(tmp_path)),
            "--request-timeout-sec", str(grace), "--trace-cap-bytes", str(2**40), *extra]


def _drive_in_process(task, url: str, *, request_timeout_sec: float = REQUEST_TIMEOUT_SEC,
                      drive_timeout_sec: float | None = None) -> Trace:
    """*task*'s drive of *url*, as the role drives it, its events collected in the order recorded."""
    events: list[TraceEvent] = []
    driver = role.build_driver(task, request_timeout_sec=request_timeout_sec, drive_timeout_sec=drive_timeout_sec)
    asyncio.run(role.drive(task, driver, artifact_url=url, split="test", on_event=events.append, gate=None))
    return Trace(events=events)


def test_the_role_drives_only_with_the_grace_the_stage_names(capsys):
    """The stage always names the drive's grace; the role has none of its own."""
    with pytest.raises(SystemExit):
        role._parse(["--task-ref", "r", "--artifact-url", "http://h:1", "--local", "l", "--out", "o",
                     "--outputs-out", "p", "--stream-out", "s", "--trace-cap-bytes", "5"])
    assert "--request-timeout-sec" in capsys.readouterr().err


def test_the_role_records_how_its_trace_writer_kept_up_and_says_it_last(tmp_path, mounted, server, capsys):
    """The stage reads the record into the drive's record; the role's last line says it too."""
    url, _ = server
    assert role.main(_args(tmp_path, url, tmp_path / "outputs" / "trace")) == 0
    record = TraceStreamRecord.model_validate_json(_stream_report(tmp_path).read_bytes())
    assert record.failure is None and 1 <= record.backlog_peak <= 10
    assert 0 < record.lag_max_sec < 60 and record.space_wait_max_sec == 0
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert "drive ended drained" in last and record.summary() in last
    assert f"at most {record.backlog_peak} events behind" in record.summary()


def test_the_sessions_the_driver_dropped_are_in_the_published_end_json(tmp_path, mounted, server, monkeypatch):
    """What the driver dropped on a dead sandbox goes into ``END.json``, which the drive's record
    carries; a drive that dropped nothing has no such key (the test above)."""
    from hls.core.trace import DroppedSession

    dropped = (DroppedSession(session_id="r3", tenant_id="t0", reason="sandbox gone: session='r3' restarts=5"),)
    real_run = Driver.run

    async def dropping(self, *a, **k):
        await real_run(self, *a, **k)
        self.dropped_sessions = dropped

    monkeypatch.setattr(Driver, "run", dropping)
    url, _ = server
    out = tmp_path / "outputs" / "trace"
    assert role.main(_args(tmp_path, url, out)) == 0
    assert tuple(read_end(out).dropped) == dropped


def test_the_role_drives_the_server_and_writes_a_complete_trace(tmp_path, mounted, server):
    url, seen = server
    out = tmp_path / "outputs" / "trace"
    assert role.main(_args(tmp_path, url, out)) == 0
    end = read_end(out)
    assert (end.events, end.ending) == (10, "drained") and len(seen) == 10
    assert end.driver["class"] == "Driver" and end.driver["split"] == "test"
    assert end.dropped == [] and b"dropped" not in (out / "END.json").read_bytes()
    trace = read_trace(out)
    assert all(ev.response is not None and ev.error is None for ev in trace.events)
    assert {ev.tenant_id for ev in trace.events} == {"t0"}


def test_a_server_that_dies_mid_drive_is_failed_requests_and_the_role_still_exits_0(tmp_path, mounted,
                                                                                      dying_server):
    """A contestant's server that dies is the contestant's result, never the driver's failure:
    a nonzero exit here would be read as the harness role's fault and retried as infra."""
    url, proc = dying_server
    out = tmp_path / "outputs" / "trace"
    started = time.time()
    assert role.main(_args(tmp_path, url, out, grace=60)) == 0
    assert time.time() - started < 30, "the dead server's requests failed at once, not by timing out"
    assert proc.wait(10) == -9, "the server was meant to be killed during the drive"
    trace = read_trace(out)
    assert read_end(out).events == len(trace.events) == 10
    answered = [ev for ev in trace.events if ev.response is not None and ev.error is None]
    failed = [ev for ev in trace.events if ev.error is not None]
    assert len(answered) == 3 and len(failed) == 7


def test_the_trace_on_the_store_reads_back_as_the_driver_recorded_it(tmp_path, mounted, server):
    url, _ = server
    trace = _drive_in_process(load_task(SAMPLE), url)
    write_trace(trace, tmp_path / "trace", ending="drained", driver={})
    back = read_trace(tmp_path / "trace")
    assert [ev.model_dump_json() for ev in back.events] == [ev.model_dump_json() for ev in trace.events]


def test_the_drive_role_reports_every_trace_file_from_its_writer_and_reads_none_back(tmp_path, mounted, server,
                                                                                     monkeypatch):
    """A trace can reach its cap, tens of GiB, so its digests come from the bytes the writer wrote,
    checked as the trace is copied to the store: the role reports each file under the outputs
    directory without hashing any of them again."""
    url, _ = server

    def refuse(path, *a, **k):
        raise AssertionError(f"{path} was read back to hash it")

    monkeypatch.setattr(cache_mod, "hash_file", refuse)
    monkeypatch.setattr(manifest_mod, "hash_file", refuse)
    # One event per chunk, so the report must name ten chunks.
    monkeypatch.setattr(role, "TraceStream", functools.partial(stream_mod.TraceStream, chunk_bytes=1))
    out = tmp_path / "outputs" / "trace"
    assert role.main(_args(tmp_path, url, out)) == 0
    end = read_end(out)
    assert len(end.chunks) == 10
    files = sorted(out.iterdir())
    assert {p.name for p in files} == {"END.json", "summary.jsonl.gz", *(c.name for c in end.chunks)}
    assert read_outputs_report(_report(tmp_path)) == {
        f"trace/{p.name}": ReportedOutput(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), size=p.stat().st_size)
        for p in files}


def test_the_drive_role_refuses_an_existing_trace_directory(tmp_path, mounted, server, capsys):
    """A trace has one writer: a directory already there on the store is not driven into, and
    what it holds is left as it was and reported nowhere."""
    url, seen = server
    there = tmp_path / "outputs" / "trace"
    there.mkdir(parents=True)
    (there / "chunk-000000.jsonl.gz").write_bytes(b"left over")
    assert role.main(_args(tmp_path, url, there)) == 1
    assert "exists" in capsys.readouterr().err
    assert seen == [], "the role refused before it drove anything"
    assert [p.name for p in there.iterdir()] == ["chunk-000000.jsonl.gz"]
    assert (there / "chunk-000000.jsonl.gz").read_bytes() == b"left over"
    assert not _report(tmp_path).exists()


def test_a_drive_may_start_until_its_latest_start_and_without_one_always():
    assert role.too_late(None, now=10**10) is None
    assert role.too_late(1000.0, now=1000.0) is None
    assert "1.5s after 1000.000" in role.too_late(1000.0, now=1001.5)


def test_a_drive_past_its_latest_start_does_not_start(tmp_path, monkeypatch, capsys):
    async def never(self, *a, **k):
        raise AssertionError("the drive was started")

    monkeypatch.setattr(Driver, "run", never)
    out = tmp_path / "outputs" / "trace"
    late = f"{time.time() - 5:.3f}"
    args = _args(tmp_path, "http://127.0.0.1:9", out, "--drive-timeout-sec", "600", "--start-by", late)
    assert role.main(args) == role.LATE_START_EXIT
    assert capsys.readouterr().err.strip().splitlines()[-1].startswith("refused: the drive would start")
    assert not out.exists() and not _report(tmp_path).exists()
    room = tmp_path / "role" / "local"
    assert {p.name: p.stat().st_size for p in room.iterdir()} == store.room_bytes(ROOM_EVENT_BYTES)


def test_the_role_builds_the_driver_whole_and_drives_it_with_the_tasks_arguments(tmp_path, monkeypatch):
    calls: dict = {}

    async def fake_run(self, artifact_url, *, split, tenants, contract, seed, on_event):
        calls.update(driver=self, url=artifact_url, split=split, tenants=tenants, contract=contract,
                     seed=seed, on_event=on_event)

    monkeypatch.setattr(Driver, "run", fake_run)
    task = load_task(SAMPLE.model_copy(update={"overrides": {"task.seed": 1234}}))
    sink = [].append
    driver = role.build_driver(task, request_timeout_sec=7.0, drive_timeout_sec=900.0)
    asyncio.run(role.drive(task, driver, artifact_url="http://h:1", split="test", on_event=sink, gate=None))
    assert calls["on_event"] is sink
    d = calls["driver"]
    assert type(d) is Driver and not {"tenants", "seed", "contract"} & set(vars(d))
    assert (d.request_timeout_sec, d.drive_timeout_sec) == (7.0, 900.0)
    assert (calls["url"], calls["split"]) == ("http://h:1", "test")
    assert calls["seed"] == task.seed == 1234 and calls["contract"] == task.contract
    # The call gets the task's tenants with their verifiers stripped.
    expected = [t.model_copy(update={"verifiers": []}) for t in task.effective_tenants()]
    assert [t.model_dump_json() for t in calls["tenants"]] == [t.model_dump_json() for t in expected]
    assert all(t.verifiers == [] for t in calls["tenants"])


#: When the late request of ``test_an_open_drives_wall_ends_a_late_request_before_its_grace``
#: arrives, in seconds after the first.
_LATE_ARRIVAL_SEC = 600.0


class _AnswersOnlyTheFirst(HTTPArtifactClient):
    """The harness's HTTP client with its POST replaced: the first request is answered after a
    second, and every later one never is. Built without ``__init__``, as the drive parity's
    recording client is."""

    #: How many requests were posted.
    posted: int

    async def _post(self, json_body: dict) -> dict:
        self.posted += 1
        await asyncio.sleep(1.0 if self.posted == 1 else 100 * REQUEST_TIMEOUT_SEC)
        return {"id": "c", "choices": [{"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop",
                                        "token_ids": [33]}], "usage": {"completion_tokens": 1}}


def _answers_only_the_first(contract, base_url, request_timeout_sec=None, stall_timeout_sec=None):
    client = HTTPArtifactClient.__new__(_AnswersOnlyTheFirst)
    client.contract, client.base_url = contract, base_url.rstrip("/")
    client.request_timeout_sec, client.posted = request_timeout_sec, 0
    return client


def test_an_open_drives_wall_ends_a_late_request_before_its_grace():
    """A task that declares no wall is driven under ``OPEN_DRIVE_SEC`` from the drive's start,
    and a request gets ``REQUEST_TIMEOUT_SEC`` past the last arrival only while the wall allows.
    On an open single-turn drive the wall comes first: a request sent at the last arrival and
    never answered is cut by the wall, short of its grace. Driven with the values the planner
    gives the drive stage, on a virtual clock whose time 0 is the drive's start."""
    sample = load_task(SAMPLE)
    [t0] = sample.effective_tenants()
    two_sessions = InlineMultiTurnDataset(conversations=[("c0", [("Q0", "A")]), ("c1", [("Q1", "A")])])
    tenant = t0.model_copy(update={
        "data_source": DataSource(dataset=two_sessions, split=SplitSpec(dev=Count(value=1), test=Count(value=2))),
        "workload": Workload(arrival=DeterministicArrival(gap=_LATE_ARRIVAL_SEC), inter_turn_delay=NoDelay(),
                             output_length_cap=Uncapped(), slo=NoSLO())})
    task = sample.model_copy(update={"tenants": [tenant], "data_source": None, "workload": None, "verifiers": None})
    assert drive_wall(task) is None and drive_wall_sec(task) == OPEN_DRIVE_SEC
    with virtual_clock() as runner:
        with mock.patch("hls.artifact.artifact_client.make_artifact_client", _answers_only_the_first):
            events: list[TraceEvent] = []
            driver = role.build_driver(task, request_timeout_sec=REQUEST_TIMEOUT_SEC,
                                       drive_timeout_sec=drive_wall_sec(task))
            runner.run(role.drive(task, driver, artifact_url="http://127.0.0.1:1", split="test",
                                  on_event=events.append, gate=None))
    first, late = sorted(events, key=lambda ev: ev.t_send)
    assert first.error is None and late.t_send == _LATE_ARRIVAL_SEC
    assert late.error == WALL_ERROR and late.failure.t_abandoned == OPEN_DRIVE_SEC
    assert late.failure.t_abandoned < late.t_send + REQUEST_TIMEOUT_SEC


def test_a_task_with_an_agent_seam_gets_the_trajectory_driver(monkeypatch):
    from tasks.deepresearch import task as dr

    calls: dict = {}

    async def fake_run(self, artifact_url, *, split, tenants, contract, seed, on_event):
        calls["driver"] = self

    monkeypatch.setattr(Driver, "run", fake_run)
    driver = role.build_driver(dr.TASK, request_timeout_sec=REQUEST_TIMEOUT_SEC, drive_timeout_sec=None)
    asyncio.run(role.drive(dr.TASK, driver, artifact_url="http://h:1", split="test", on_event=[].append,
                           gate=None))
    assert isinstance(calls["driver"], TrajectorySessionDriver)
    assert calls["driver"].drive_timeout_sec == dr.TASK.agent.drive_timeout_sec
    assert calls["driver"].scheduled_stop is None and calls["driver"].stall_timeout_sec is None
    # Its retriever asks for a GPU, of the task's kind.
    assert calls["driver"].assets_gpu and calls["driver"].gpu_kind == dr.TASK.hardware.accelerator.kind


def test_the_driver_description_is_unchanged_for_a_task_without_a_stop():
    """The trace's driver description names the class, the split, the grace, the wall and the
    open-file limit, and nothing more for a task that declares no scheduled stop and no stall
    budget."""
    from tasks.deepresearch import task as dr

    for task in (load_task(SAMPLE), dr.TASK):
        driver = role.build_driver(task, request_timeout_sec=REQUEST_TIMEOUT_SEC, drive_timeout_sec=drive_wall_sec(task))
        assert set(role.driver_description(driver, split="test", nofile=1024)) == {
            "class", "split", "request_timeout_sec", "drive_timeout_sec", "nofile_soft"}


def test_the_driver_description_names_the_stop_and_the_stall_budget_where_declared():
    from tasks.agentic_coding import task as ac

    driver = role.build_driver(ac.TASK, request_timeout_sec=REQUEST_TIMEOUT_SEC, drive_timeout_sec=None)
    described = role.driver_description(driver, split="test", nofile=65536)
    assert described["drive_timeout_sec"] == str(drive_wall_sec(ac.TASK))
    assert (described["wall_clock_sec"], described["cut_grace_sec"], described["stall_timeout_sec"]) == (
        str(ac.DRIVE_WALL_CLOCK_SEC), str(ac.DRIVE_CUT_GRACE_SEC), str(ac.STALL_TIMEOUT_SEC))
    store._checked_driver(described)


def test_the_roles_own_process_serializes_no_event_while_the_drive_runs(tmp_path, mounted, server, monkeypatch):
    """The role's process, whose event loop sends every request and stamps every time, turns no
    event into JSON during the drive: its writer process does."""
    url, _ = server
    driving = {"now": False}
    real_run = Driver.run

    async def watched(self, *a, **k):
        driving["now"] = True
        try:
            return await real_run(self, *a, **k)
        finally:
            driving["now"] = False

    def refuse(self, *a, **k):
        raise AssertionError("a trace event was serialized during the drive")

    real_dump_json, real_dump = TraceEvent.model_dump_json, TraceEvent.model_dump
    monkeypatch.setattr(Driver, "run", watched)
    monkeypatch.setattr(TraceEvent, "model_dump_json",
                        lambda self, *a, **k: refuse(self) if driving["now"] else real_dump_json(self, *a, **k))
    monkeypatch.setattr(TraceEvent, "model_dump",
                        lambda self, *a, **k: refuse(self) if driving["now"] else real_dump(self, *a, **k))
    out = tmp_path / "trace"
    assert role.main(_args(tmp_path, url, out)) == 0
    assert read_end(out).events == 10


@pytest.mark.parametrize("where", ["out", "local", "room"])
def test_a_trace_that_cannot_be_written_is_an_infra_exit_with_no_trace(tmp_path, mounted, server, capsys, where):
    """On the store, where the trace is published, or on the node, where it is written as the
    drive goes: into no directory, or a directory the stage made no room in."""
    url, _ = server
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    out, local = tmp_path / "outputs" / "trace", _room(tmp_path / "local")
    if where == "out":
        out = blocker / "trace"
    elif where == "local":
        local = blocker / "local"
    else:
        local = tmp_path / "empty"
        local.mkdir()
    assert role.main(_args(tmp_path, url, out, local=local)) == 1
    assert "no complete trace was written" in capsys.readouterr().err
    assert not _report(tmp_path).exists()
    # The record of the trace's writer says why it failed, for the stage's evidence; one that
    # wrote the whole trace failed nothing.
    record = TraceStreamRecord.model_validate_json(_stream_report(tmp_path).read_bytes())
    if where == "out":
        assert record.failure is None
    else:
        assert record.failure.startswith("the trace writer failed") and (
            "FileNotFoundError" in record.failure or "NotADirectoryError" in record.failure), record.failure


def test_a_drive_that_raises_is_an_infra_exit(tmp_path, mounted, monkeypatch, capsys):
    async def broken(self, *a, **k):
        raise RuntimeError("driver blew up")

    monkeypatch.setattr(Driver, "run", broken)
    out = tmp_path / "trace"
    assert role.main(_args(tmp_path, "http://h:1", out)) == 1
    assert "driver blew up" in capsys.readouterr().err and not out.exists()
    assert not _report(tmp_path).exists()


def test_an_unknown_task_override_is_an_infra_exit(tmp_path, capsys):
    ref = TaskRef(module="tasks.sample_task.task", overrides={"task.no_such_field": 1})
    assert role.main(_args(tmp_path, "http://h:1", tmp_path / "trace", ref=ref)) == 1
    assert "no_such_field" in capsys.readouterr().err


def test_the_open_file_limit_is_the_roles_own_unless_the_stage_names_one(tmp_path, monkeypatch):
    """The role raises its soft RLIMIT_NOFILE before the drive: to ``NOFILE_SOFT``, or to the
    value the stage passes."""
    raised: list[int] = []

    async def broken(self, *a, **k):
        raise RuntimeError("stop here")

    def raise_nofile(target: int) -> int:
        raised.append(target)
        return target

    monkeypatch.setattr(role, "_raise_nofile", raise_nofile)
    monkeypatch.setattr(Driver, "run", broken)
    role.main(_args(tmp_path, "http://h:1", tmp_path / "a"))
    role.main(_args(tmp_path, "http://h:1", tmp_path / "b", "--nofile-soft", "4096"))
    assert raised == [role.NOFILE_SOFT, 4096] and raised[0] == 65536


# The role as a program: run in its own process, as its step runs it, with sample_task's
# rows at the directory given first, and the soft open-file limit given second ("-" leaves
# it), as a step that inherited it would start. It prints what it saw during the drive on
# stdout.
_ROLE_PROGRAM = '''
import json, logging, resource, sys
from pathlib import Path

import hls.spec.datasets as datasets
from hls.spec.drivers import Driver

datasets.MOUNT_ROOT = Path(sys.argv[1])
if sys.argv[2] != "-":
    resource.setrlimit(resource.RLIMIT_NOFILE, (int(sys.argv[2]), resource.getrlimit(resource.RLIMIT_NOFILE)[1]))
seen = {"driving": False, "httpx_records": [], "soft_nofile": None}


class _Seen(logging.Handler):
    def emit(self, record):
        if seen["driving"]:
            seen["httpx_records"].append(record.getMessage())


for name in ("httpx", "httpcore"):
    logging.getLogger(name).addHandler(_Seen())
real_run = Driver.run


async def watched(self, *a, **k):
    seen["driving"] = True
    seen["soft_nofile"] = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    try:
        return await real_run(self, *a, **k)
    finally:
        seen["driving"] = False


Driver.run = watched
from hls.drive import role

code = role.main(sys.argv[3:])
del seen["driving"]
print(json.dumps(seen))
sys.exit(code)
'''


def _run_role_program(tmp_path: Path, rows: int, url: str, *, soft_nofile: int | None = None):
    data = tmp_path / "data"
    for split, n in (("dev", 5), ("test", rows)):
        d = data / "tenants" / "t0" / split
        d.mkdir(parents=True)
        (d / "rows.json").write_text(json.dumps(_rows(n)))
    program = tmp_path / "role_program.py"
    program.write_text(_ROLE_PROGRAM)
    repo = Path(__file__).resolve().parents[2]
    out = tmp_path / "outputs" / "trace"
    soft = "-" if soft_nofile is None else str(soft_nofile)
    proc = subprocess.run([sys.executable, str(program), str(data), soft,
                           *_args(tmp_path, url, out, grace=60)],
                          cwd=repo, env={**os.environ, "PYTHONPATH": f"{repo / 'src'}:{repo}"},
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.splitlines()[-1]), out, proc.stderr, resource.getrlimit(resource.RLIMIT_NOFILE)[1]


@pytest.fixture
def slow_server():
    """A server that holds every request for a second, so an all-at-once drive keeps every
    request open at the same time."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        await asyncio.sleep(1.0)
        return {"id": "c", "choices": [{"message": {"role": "assistant", "content": "B"}, "finish_reason": "stop",
                                        "token_ids": [33]}], "usage": {"completion_tokens": 1}}

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    srv = uvicorn.Server(uvicorn.Config(app, log_level="warning", backlog=4096))
    thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.time() + 10
    while not srv.started and time.time() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{sock.getsockname()[1]}"
    srv.should_exit = True
    thread.join(5)
    sock.close()


def test_a_drive_past_the_inherited_open_file_limit_is_served_whole(tmp_path, slow_server):
    """A step may inherit a low soft limit on open files. Each request in flight holds a
    socket, so a role that kept it would fail the requests past it as connect faults charged
    to the contestant; the role raises it before the drive."""
    seen, out, _, hard = _run_role_program(tmp_path, 300, slow_server, soft_nofile=128)
    assert seen["soft_nofile"] == min(role.NOFILE_SOFT, hard)
    assert read_end(out).driver["nofile_soft"] == str(min(role.NOFILE_SOFT, hard))
    trace = read_trace(out)
    assert len(trace.events) == 300 and [ev.error for ev in trace.events if ev.error] == []
    assert read_end(out).ending == "drained"


def test_no_httpx_record_is_emitted_while_the_drive_runs(tmp_path, server):
    """The driver writes no line per request: its process keeps the root logger, and so
    httpx, at WARNING. A line per request would be written on the measured event loop, and on
    a streamed reply inside its time to first token."""
    url, seen_requests = server
    seen, out, stderr, _ = _run_role_program(tmp_path, 10, url)
    assert len(seen_requests) == 10 and read_end(out).events == 10
    assert seen["httpx_records"] == [] and "HTTP Request" not in stderr
    assert "drive ended drained" in stderr


# The role run as its step runs it, ``python -m hls.drive.role``, so that its module is
# ``__main__``, with sample_task's rows at the directory given first.
_AS_MAIN = '''
import runpy, sys
from pathlib import Path

import hls.spec.datasets as datasets

datasets.MOUNT_ROOT = Path(sys.argv[1])
sys.argv = ["hls.drive.role", *sys.argv[2:]]
runpy.run_module("hls.drive.role", run_name="__main__", alter_sys=True)
'''


def test_the_roles_own_log_lines_reach_its_stderr_when_it_runs_as_a_program(tmp_path, mounted, server):
    url, _ = server
    program = tmp_path / "as_main.py"
    program.write_text(_AS_MAIN)
    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.run([sys.executable, str(program), str(mounted),
                           *_args(tmp_path, url, tmp_path / "outputs" / "trace", grace=60)],
                          cwd=repo, env={**os.environ, "PYTHONPATH": f"{repo / 'src'}:{repo}"},
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "drive ended drained: 10 events" in proc.stderr


def _event(i: int, error: str | None = None) -> TraceEvent:
    env = Envelope(request_id=f"s{i}.0", contents=[TextPayload(text="q")], metadata={"tenant_id": "t0"})
    if error is not None:
        return TraceEvent(request_id=f"s{i}.0", input=env, t_send=float(i), error=error)
    out = Envelope(request_id=f"s{i}.0", contents=[TextPayload(text="a")])
    return TraceEvent(request_id=f"s{i}.0", input=env, t_send=float(i), t_recv=i + 0.5,
                      response=ArtifactResponse(output=out))


@pytest.mark.parametrize("errors, ending", [
    ([None, None], "drained"),
    ([None, WALL_ERROR], "wall"),
    ([None, ABORTED_ERROR], "aborted"),
    ([ABANDONED_ERROR, None], "aborted"),
    ([ABORTED_ERROR, WALL_ERROR], "wall"),
    ([None, "server error 500"], "drained"),
])
def test_the_ending_is_read_from_the_stamped_errors(errors, ending):
    watch = role.EndingWatch()
    for i, e in enumerate(errors):
        watch.note(_event(i, e))
    assert watch.ending == ending


# The open-file limit tests mutate this test process's own RLIMIT_NOFILE, always within
# [current-or-lower, hard], which an unprivileged process may do freely, and restore it after.
@pytest.fixture
def restore_nofile():
    """Snapshot RLIMIT_NOFILE and restore it after the test."""
    saved = resource.getrlimit(resource.RLIMIT_NOFILE)
    yield saved
    resource.setrlimit(resource.RLIMIT_NOFILE, saved)


def test_raise_nofile_raises_a_lowered_soft_limit(restore_nofile):
    soft, hard = restore_nofile
    lowered = min(soft, 512)
    resource.setrlimit(resource.RLIMIT_NOFILE, (lowered, hard))
    role._raise_nofile(soft)
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (soft, hard)


def test_raise_nofile_clamps_the_target_to_the_hard_limit(restore_nofile):
    soft, hard = restore_nofile
    if hard == resource.RLIM_INFINITY:
        pytest.skip("hard limit is unlimited: nothing to clamp against")
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(soft, 512), hard))
    role._raise_nofile(hard + 1_000_000)  # beyond the hard limit: must clamp, not crash
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (hard, hard)


def _refusing_setrlimit() -> SimpleNamespace:
    """The ``resource`` module as an exotic sandbox gives it to the role: ``setrlimit`` refused."""
    def refused(*a):
        raise OSError(1, "Operation not permitted")

    return SimpleNamespace(getrlimit=resource.getrlimit, setrlimit=refused, RLIMIT_NOFILE=resource.RLIMIT_NOFILE,
                           RLIM_INFINITY=resource.RLIM_INFINITY)


def test_raise_nofile_returns_the_soft_limit_it_left(restore_nofile, monkeypatch):
    soft, hard = restore_nofile
    lowered = min(soft, 512)
    resource.setrlimit(resource.RLIMIT_NOFILE, (lowered, hard))
    assert role._raise_nofile(soft) == soft
    assert role._raise_nofile(1) == soft
    resource.setrlimit(resource.RLIMIT_NOFILE, (lowered, hard))

    monkeypatch.setattr(role, "resource", _refusing_setrlimit())
    assert role._raise_nofile(soft) == lowered


def test_a_drive_whose_limit_was_not_raised_records_and_logs_the_one_it_drove_under(
        tmp_path, mounted, server, restore_nofile, monkeypatch, capsys):
    """An exotic sandbox may refuse ``setrlimit``; the drive goes on under the inherited soft
    limit, and the trace's driver description and the role's log say which, so connect faults
    past it are not read as the contestant's without the limit beside them."""
    url, _ = server
    soft, hard = restore_nofile
    lowered = min(soft, 512)
    resource.setrlimit(resource.RLIMIT_NOFILE, (lowered, hard))

    monkeypatch.setattr(role, "resource", _refusing_setrlimit())
    out = tmp_path / "outputs" / "trace"
    assert role.main(_args(tmp_path, url, out)) == 0
    assert read_end(out).driver["nofile_soft"] == str(lowered)
    err = capsys.readouterr().err
    assert f"soft open-file limit {lowered}, below the {role.NOFILE_SOFT} asked" in err


def test_raise_nofile_never_lowers(restore_nofile):
    soft, hard = restore_nofile
    role._raise_nofile(1)  # far below the current soft limit
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == (soft, hard)
