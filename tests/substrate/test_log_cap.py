from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hls.pipeline.events import EventLog, read_events
from hls.spec.resources import ResourceShape
from hls.store.posix import PosixStore
from hls.pipeline.stage import StageContext
from hls.substrate.faults import classify_role
from hls.substrate.log_cap import cap, dropped_line, spawn
from tests.pipeline.fakes import LocalLauncher
from tests.pipeline.plans import noop_plan

KEEP = 1000


def _capped(tmp_path: Path, data: bytes, keep: int = KEEP, chunks: int = 7) -> bytes:
    """What the capper leaves in its file once *data* went through it in *chunks* writes."""
    out = tmp_path / "role.out"
    writer, proc = spawn(out, keep=keep, python=sys.executable, errors=tmp_path / "cap.err")
    step = max(1, len(data) // chunks)
    with os.fdopen(writer, "wb") as pipe:
        for i in range(0, len(data), step):
            pipe.write(data[i:i + step])
    assert proc.wait(30) == 0
    return out.read_bytes()


@pytest.mark.parametrize("size", [0, 1, KEEP, KEEP + 1, 2 * KEEP])
def test_a_log_within_both_ends_is_kept_whole(tmp_path, size):
    data = bytes(i % 251 for i in range(size))
    assert _capped(tmp_path, data) == data


@pytest.mark.parametrize("size", [2 * KEEP + 1, 3 * KEEP + 17, 50 * KEEP])
def test_a_longer_log_keeps_its_first_and_last_bytes_and_says_what_it_dropped(tmp_path, size):
    data = bytes(i % 251 for i in range(size))
    dropped = size - 2 * KEEP
    assert _capped(tmp_path, data) == data[:KEEP] + dropped_line(dropped, KEEP) + data[-KEEP:]
    assert dropped_line(dropped, KEEP).startswith(b"\n[hls: ") and str(dropped).encode() in dropped_line(dropped, KEEP)


def test_the_head_is_written_through_as_it_comes(tmp_path):
    out = tmp_path / "role.err"
    writer, proc = spawn(out, keep=KEEP, python=sys.executable, errors=tmp_path / "cap.err")
    with os.fdopen(writer, "wb", buffering=0) as pipe:
        pipe.write(b"HLS-START\n")
        deadline = time.time() + 10
        while out.read_bytes() != b"HLS-START\n" and time.time() < deadline:
            time.sleep(0.02)
        assert out.read_bytes() == b"HLS-START\n" and proc.poll() is None
    assert proc.wait(30) == 0


def test_the_capper_appends_to_a_log_a_relaunch_left(tmp_path):
    out = tmp_path / "role.out"
    out.write_bytes(b"earlier launch\n")
    writer, proc = spawn(out, keep=KEEP, python=sys.executable, errors=tmp_path / "cap.err")
    os.write(writer, b"this launch\n")
    os.close(writer)
    assert proc.wait(30) == 0 and out.read_bytes() == b"earlier launch\nthis launch\n"


def test_the_capper_holds_no_other_descriptor_of_its_parent(tmp_path):
    read_end, write_end = os.pipe()
    try:
        writer, proc = spawn(tmp_path / "role.out", keep=KEEP, python=sys.executable, errors=tmp_path / "cap.err")
        os.close(writer)
        assert proc.wait(30) == 0
        # The capper did not hold the other pipe's write end, or this read would block.
        os.close(write_end)
        assert os.read(read_end, 1) == b""
    finally:
        os.close(read_end)


def _context(tmp_path: Path, launcher: LocalLauncher) -> StageContext:
    plan = noop_plan(tmp_path)
    store = PosixStore(plan.store_root)
    events = EventLog(store.path(plan.layout.attempt_file("noop_a", 1, "events.jsonl")))
    return StageContext(store=store, plan=plan, stage=plan.stage("noop_a"), attempt=1, launcher=launcher,
                        watch=None, events=events, inputs={}, deadline=time.time() + 60,
                        scratch=tmp_path / "node" / "hls-100", job_id="100")


#: Writes 3 * KEEP bytes of a's to stdout and b's to stderr, then exits.
FLOOD = [sys.executable, "-c", f"import sys; sys.stdout.write('a' * {3 * KEEP}); sys.stderr.write('b' * {3 * KEEP})"]


def test_every_roles_logs_are_capped_and_whole_once_it_is_reported(tmp_path):
    ctx = _context(tmp_path, LocalLauncher(gpus=0, log_keep_bytes=KEEP))
    ctx.start_role("flood", FLOOD, ResourceShape(cpus=1, memory_gib=1, time_sec=60), owner="contestant")
    ctx.start_role("helper", FLOOD, ResourceShape(cpus=1, memory_gib=1, time_sec=60), owner="harness")
    ctx.wait_roles()
    logs = ctx.attempt_dir / "logs"
    line = dropped_line(KEEP, KEEP)
    assert (logs / "flood.out").read_bytes() == b"a" * KEEP + line + b"a" * KEEP
    assert (logs / "flood.err").read_bytes() == b"b" * KEEP + line + b"b" * KEEP
    # A harness role, such as a driver logging what replies said, is capped the same way.
    assert (logs / "helper.out").read_bytes() == b"a" * KEEP + line + b"a" * KEEP
    assert (logs / "helper.err").read_bytes() == b"b" * KEEP + line + b"b" * KEEP
    owners = {e["role"]: e["owner"] for e in read_events(ctx.attempt_dir / "events.jsonl")
              if e["kind"] == "role_started"}
    assert owners == {"flood": "contestant", "helper": "harness"}


def test_the_capper_reads_its_pipe_to_the_end_whatever_its_writer_does(tmp_path):
    out = tmp_path / "role.out"
    writer, proc = spawn(out, keep=KEEP, python=sys.executable, errors=tmp_path / "cap.err")
    child = subprocess.Popen([sys.executable, "-c", "import os; os.write(1, b'x' * 5000); os._exit(9)"],
                             stdout=writer)
    os.close(writer)
    assert child.wait(30) == 9 and proc.wait(30) == 0
    assert out.read_bytes() == b"x" * KEEP + dropped_line(3000, KEEP) + b"x" * KEEP


@pytest.mark.parametrize("pieces", [[300] * 12, [999, 1, 1000, 1001, 7], [2500, 13, 13, 13]])
def test_the_last_bytes_keep_their_order_however_they_arrive(tmp_path, pieces):
    data = bytes(i % 251 for i in range(sum(pieces)))
    reads = []
    at = 0
    for n in pieces:
        reads.append(data[at:at + n])
        at += n
    reads.reverse()
    out = tmp_path / "role.out"
    cap(lambda n: reads.pop() if reads else b"", out, keep=KEEP)
    assert out.read_bytes() == data[:KEEP] + dropped_line(len(data) - 2 * KEEP, KEEP) + data[-KEEP:]



@pytest.mark.parametrize("owner", ["harness", "contestant"])
def test_a_role_whose_capper_died_ends_as_infra(tmp_path, owner):
    ctx = _context(tmp_path, LocalLauncher(gpus=0, log_keep_bytes=KEEP))
    slow = [sys.executable, "-c", "import sys, time; time.sleep(1); print('late'); sys.stdout.flush()"]
    handle = ctx.start_role("slow", slow, ResourceShape(cpus=1, memory_gib=1, time_sec=60), owner=owner)
    capper = handle.log_cappers[0]
    capper.kill()
    capper.wait(10)
    [outcome] = ctx.wait_roles()
    assert outcome.ending == "log_lost" and f"exited {capper.returncode}" in outcome.detail
    fault = classify_role(outcome)
    assert (fault.fault_class, fault.retryable) == ("infra", True)


def _holds(path: Path, data: bytes) -> None:
    """Wait until the file at *path* holds *data*, which a capper writes through as it comes."""
    deadline = time.time() + 10
    while path.read_bytes() != data and time.time() < deadline:
        time.sleep(0.02)
    assert path.read_bytes() == data


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_a_signalled_capper_keeps_capturing_until_its_writer_is_done(tmp_path, signum):
    # A job's cancel signals every process of the batch step, its cappers among them.
    out = tmp_path / "role.out"
    writer, proc = spawn(out, keep=KEEP, python=sys.executable, errors=tmp_path / "cap.err")
    with os.fdopen(writer, "wb", buffering=0) as pipe:
        pipe.write(b"before\n")
        _holds(out, b"before\n")
        proc.send_signal(signum)
        pipe.write(b"after\n" * (3 * KEEP // 6))
    assert proc.wait(30) == 0
    data = b"before\n" + b"after\n" * (3 * KEEP // 6)
    assert out.read_bytes() == data[:KEEP] + dropped_line(len(data) - 2 * KEEP, KEEP) + data[-KEEP:]


def test_a_role_whose_cappers_were_signalled_ends_its_own_way(tmp_path):
    ctx = _context(tmp_path, LocalLauncher(gpus=0, log_keep_bytes=KEEP))
    script = ("import sys, time\nfor f in (sys.stdout, sys.stderr): print('first', file=f, flush=True)\n"
              "time.sleep(1)\nfor f in (sys.stdout, sys.stderr): print('last', file=f, flush=True)\n")
    handle = ctx.start_role("talker", [sys.executable, "-c", script], ResourceShape(cpus=1, memory_gib=1, time_sec=60),
                            owner="contestant")
    logs = ctx.attempt_dir / "logs"
    for suffix in ("out", "err"):
        _holds(logs / f"talker.{suffix}", b"first\n")
    for capper in handle.log_cappers:
        capper.send_signal(signal.SIGTERM)
    [outcome] = ctx.wait_roles()
    assert (outcome.ending, outcome.exit_code) == ("exited", 0) and classify_role(outcome) is None
    assert [(logs / f"talker.{s}").read_bytes() for s in ("out", "err")] == [b"first\nlast\n"] * 2


def test_a_host_process_that_imports_the_module_keeps_its_signals():
    # The child sets each signal to its default before the import: an ignored signal is
    # inherited, so this process, which imported the module, could otherwise hide a change.
    code = ("import signal\nsigs = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)\n"
            "for s in sigs:\n    signal.signal(s, signal.SIG_DFL)\n"
            "import hls.substrate.log_cap\n"
            "print([signal.getsignal(s) == signal.SIG_DFL for s in sigs])\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[True, True, True]"


def test_a_cappers_errors_go_to_a_file_beside_it(tmp_path):
    out = tmp_path / "role.out"
    errors = tmp_path / "role.out.cap.err"
    writer, proc = spawn(out, keep=0, python=sys.executable, errors=errors)
    os.close(writer)
    assert proc.wait(30) != 0 and "--keep must be at least 1" in errors.read_text()
