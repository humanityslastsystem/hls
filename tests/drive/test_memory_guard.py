"""The driver role's memory guard, the second layer of its reply gate: once the driver's own
memory passes 85% of its declared memory, every further reply fails as the contestant's."""
from __future__ import annotations

import asyncio
import json
import os
import socket
import threading
import time
import tracemalloc
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

import hls.spec.datasets as datasets_mod
from hls.core.trace import FailureReason
from hls.drive import memory as memory_mod
from hls.drive import role
from hls.drive.memory import GUARD_SHARE, SAMPLE_SEC, MemoryGuard, MemorySource, memory_source
from hls.drive.replies import ReplyGate, ReplyOverBounds
from hls.trace.store import read_trace
from tests.drive.test_replies import STREAM, Server, _drive

#: What a reply the guard fails records.
WHY = "reply past the drive's bounds: the driver's memory passed 85% of its"


class _Values:
    """A memory source that reads the next of *values*, then the last one again."""

    def __init__(self, *values: int) -> None:
        self.values = list(values)
        self.reads = 0

    def __call__(self) -> int:
        self.reads += 1
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def _gate() -> ReplyGate:
    return ReplyGate(reply_max_bytes=10**9, trace_cap_bytes=10**12, growth=1.0, written=lambda: (0, 0))


# --- the guard ----------------------------------------------------------------------------------

def test_the_guard_closes_the_gate_once_memory_passes_85_percent_and_not_at_it():
    gate = _gate()
    guard = MemoryGuard(gate, 1000, MemorySource("test", _Values(100, 850, 851, 900)))
    assert GUARD_SHARE == 0.85 and guard.threshold == 850
    guard.check()
    guard.check()
    assert gate.closed is None and guard.fired_at is None
    gate.admit()
    gate.count(10, 10)
    guard.check()
    assert guard.fired_at == 851 and gate.closed == "the driver's memory passed 85% of its 1,000 bytes"
    guard.check()
    assert guard.peak == 900 and guard.fired_at == 851, "the guard fires once, at the first sample past it"


def test_a_gate_the_guard_closed_fails_every_later_reply():
    gate = _gate()
    gate.count(5, 5)
    gate.close("the driver's memory passed 85% of its 1,000 bytes")
    with pytest.raises(ReplyOverBounds, match="the driver's memory passed 85% of its 1,000 bytes"):
        gate.admit()
    with pytest.raises(ReplyOverBounds, match="the driver's memory passed 85% of its 1,000 bytes"):
        gate.count(10, 5)


def test_the_guard_samples_while_the_drive_runs_and_stops_with_it():
    values = _Values(1)
    guard = MemoryGuard(_gate(), 10**6, MemorySource("test", values), interval=0.01)
    assert SAMPLE_SEC == 1.0 and MemoryGuard(_gate(), 1, MemorySource("t", values)).interval == SAMPLE_SEC
    with guard:
        time.sleep(0.3)
        assert any(t.name == "drive-memory-guard" for t in threading.enumerate())
    during = values.reads
    assert during >= 10
    time.sleep(0.1)
    assert values.reads == during and not any(t.name == "drive-memory-guard" for t in threading.enumerate())


def test_a_source_that_fails_falls_back_to_the_processs_rss():
    def broken() -> int:
        raise OSError("the step's cgroup is gone")

    guard = MemoryGuard(_gate(), 10**12, MemorySource("step cgroup", broken))
    guard.check()
    assert guard.peak > 0 and guard.source.name == "RSS"


# --- what memory is read ------------------------------------------------------------------------

def _step(tmp_path: Path, files: dict[str, str]) -> tuple[Path, Path]:
    cgroup = "/system.slice/slurmstepd.scope/job_7/step_2/user/task_0"
    proc = tmp_path / "cgroup"
    proc.write_text(f"0::{cgroup}\n")
    root = tmp_path / "fs"
    step = root / "system.slice/slurmstepd.scope/job_7/step_2/user"
    (step / "task_0").mkdir(parents=True)
    for name, text in files.items():
        (step / name).write_text(text)
    return proc, root


def test_the_memory_is_the_steps_footprint_as_the_runners_memory_watch_reads_it(tmp_path):
    """memory.current and swap, less the page cache and slab the kernel reclaims first."""
    proc, root = _step(tmp_path, {"memory.current": "1000\n", "memory.swap.current": "20\n",
                                  "memory.stat": "file 300\nshmem 100\nslab_reclaimable 50\n",
                                  "memory.events": "oom 0\noom_kill 0\n"})
    source = memory_source(proc_cgroup=proc, root=root)
    assert source.name == "step cgroup" and source.read() == 1000 + 20 - (300 - 100) - 50


def test_outside_a_readable_step_the_memory_is_the_processs_rss(tmp_path):
    not_a_step = tmp_path / "cgroup"
    not_a_step.write_text("0::/user.slice/user-1000.slice/session-1.scope\n")
    unreadable, root = _step(tmp_path, {})
    rss = int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    for proc in (not_a_step, unreadable, tmp_path / "missing"):
        source = memory_source(proc_cgroup=proc, root=root)
        assert source.name == "RSS" and 0.5 * rss < source.read() < 2 * rss


# --- the role -----------------------------------------------------------------------------------

@pytest.fixture
def forty(tmp_path, monkeypatch):
    """sample_task's rows where its driver reads them: forty test questions."""
    for split, n in (("dev", 2), ("test", 40)):
        d = tmp_path / "data" / "tenants" / "t0" / split
        d.mkdir(parents=True)
        (d / "rows.json").write_text(json.dumps(
            [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": 0, "subject": "x", "source_index": i}
             for i in range(n)]))
    monkeypatch.setattr(datasets_mod, "MOUNT_ROOT", tmp_path / "data")


class _Traced:
    """Python's allocations since the first read, as tracemalloc counts them."""

    def __init__(self) -> None:
        self.base: int | None = None

    def __call__(self) -> int:
        current = tracemalloc.get_traced_memory()[0]
        if self.base is None:
            self.base = current
        return current - self.base


class _DenseStreams:
    """A server whose every stream reply sends *frames* frames of *ids* token ids of four bytes
    each, *gap* seconds apart: arriving all at once, its forty streams are in flight together,
    each holding its parsed frames until it ends."""

    def __init__(self, *, frames: int, ids: int, gap: float) -> None:
        app = FastAPI()

        @app.get("/health")
        async def _health():
            return {"status": "ok"}

        @app.post("/v1/chat/completions")
        async def _chat(request: Request):
            await request.json()

            async def body():
                for _ in range(frames):
                    frame = {"id": "c", "choices": [{"index": 0, "delta": {"content": "B"}, "token_ids": [300] * ids,
                                                     "finish_reason": None}]}
                    yield f"data: {json.dumps(frame)}\n\n"
                    await asyncio.sleep(gap)
                yield f"data: {json.dumps({'id': 'c', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(body(), media_type="text/event-stream")

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.sock = sock
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.srv = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self.thread = threading.Thread(target=self.srv.run, kwargs={"sockets": [sock]}, daemon=True)

    def __enter__(self) -> "_DenseStreams":
        self.thread.start()
        deadline = time.time() + 10
        while not self.srv.started and time.time() < deadline:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc) -> None:
        self.srv.should_exit = True
        self.thread.join(5)
        self.sock.close()


def test_densely_spelled_streams_in_flight_trip_the_guard_before_the_driver_reaches_its_memory(
        tmp_path, forty, monkeypatch, capsys):
    """Forty streams in flight at once, each frame 5,000 token ids of four bytes, which the
    decoder holds as Python ints until each stream ends: well within the drive's trace cap, far past
    what their wire bytes suggest. The guard fails every reply once the driver's memory passes 85%
    of its limit, and the driver's memory never reaches the limit."""
    limit = 32 * 2**20
    traced = _Traced()
    monkeypatch.setattr(memory_mod, "SAMPLE_SEC", 0.01)
    monkeypatch.setattr(role, "memory_source", lambda: MemorySource("traced", traced))
    tracemalloc.start()
    try:
        with _DenseStreams(frames=20, ids=5_000, gap=0.05) as srv:
            out = _drive(tmp_path, srv.url, "--memory-limit-bytes", str(limit), ref=STREAM)
        peak = tracemalloc.get_traced_memory()[1] - traced.base
    finally:
        tracemalloc.stop()
    events = read_trace(out).events
    failed = [ev for ev in events if ev.error is not None]
    assert len(events) == 40 and failed
    for ev in failed:
        assert ev.error == f"protocol: {WHY} {limit:,} bytes" and ev.failure.reason is FailureReason.PROTOCOL
    assert peak < limit
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert f"of {limit:,} bytes (traced), guard fired at" in last


def test_a_normal_drive_never_trips_the_guard(tmp_path, forty, capsys):
    with Server(lambda i: "B", stagger=0) as srv:
        out = _drive(tmp_path, srv.url, "--memory-limit-bytes", str(16 * 2**30))
    assert [ev.error for ev in read_trace(out).events] == [None] * 40
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert "memory peak" in last and f"of {16 * 2**30:,} bytes" in last and "guard not fired" in last
    assert not any(t.name == "drive-memory-guard" for t in threading.enumerate())


def test_without_a_memory_limit_the_guard_is_off_and_says_so(tmp_path, forty, capsys):
    with Server(lambda i: "B", stagger=0) as srv:
        _drive(tmp_path, srv.url)
    assert "; memory guard off; the trace writer was" in capsys.readouterr().err.strip().splitlines()[-1]


def test_a_drive_of_the_measured_baseline_has_no_guard(tmp_path, forty, monkeypatch):
    monkeypatch.setattr(role, "memory_source", lambda: MemorySource("full", lambda: 10**12))
    with Server(lambda i: "B", stagger=0) as srv:
        out = _drive(tmp_path, srv.url, "--measured-baseline", "--memory-limit-bytes", "1000")
    assert [ev.error for ev in read_trace(out).events] == [None] * 40


def test_the_memory_limit_is_a_positive_number_of_bytes(capsys):
    base = ["--task-ref", "r", "--artifact-url", "http://h:1", "--local", "l", "--out", "o", "--outputs-out", "p",
            "--stream-out", "s", "--request-timeout-sec", "60", "--trace-cap-bytes", "5"]
    assert role._parse(base).memory_limit_bytes is None
    assert role._parse([*base, "--memory-limit-bytes", "5"]).memory_limit_bytes == 5
    with pytest.raises(SystemExit):
        role._parse([*base, "--memory-limit-bytes", "0"])
    assert "--memory-limit-bytes: 0 is not a positive number of bytes" in capsys.readouterr().err
