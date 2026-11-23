"""The reference-scoring service's request handling, with a stub engine and no GPU."""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

from hls.reference import service
from hls.reference import service_harness as rs
from hls.reference.service import (RequestError, Service, UnixHTTPServer,
                                   parse_score_request, position_record)
from hls.util import remove_tree

VOCAB = 1000
GIB = 1 << 30


class _Entry:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


def _row(token_id: int, pos: int) -> dict:
    """A deterministic prompt-logprob row naming ``token_id`` plus 21 alternatives."""
    row = {token_id: _Entry(-5.0 - (token_id % 3) * 0.1)}
    for k in range(21):
        alt = (token_id + 7 + k * 13) % VOCAB
        row.setdefault(alt, _Entry(-0.5 - k * 0.2 - pos * 1e-4))
    return row


class _StubEngine:
    def __init__(self, *, fail_wake: bool = False) -> None:
        self.calls: list[str] = []
        self.fail_wake = fail_wake

    def rows(self, token_ids):
        self.calls.append("rows")
        return [None] + [_row(t, i) for i, t in enumerate(token_ids) if i > 0]

    def sleep(self):
        self.calls.append("sleep")

    def wake(self):
        self.calls.append("wake")
        if self.fail_wake:
            raise RuntimeError("CUDA out of memory")

    def shutdown(self):
        self.calls.append("shutdown")


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _service(engine, *, free_asleep=100 * GIB, clock=None, gpus=2):
    """A loaded service whose engine used 80 GiB per device awake and 2 GiB asleep."""
    total = 140 * GIB
    free = {"now": [total - 80 * GIB] * gpus}

    def memory():
        return [{"index": i, "free_bytes": f, "total_bytes": total}
                for i, f in enumerate(free["now"])]

    svc = Service(engine_factory=lambda: engine, vocab_size=VOCAB, max_model_len=64,
                  top_logprobs=21, idle_sleep_sec=120, gpu_memory=memory,
                  config={"k": "v"}, clock=clock or _Clock())
    real_sleep = engine.sleep

    def sleep_and_free():
        real_sleep()
        free["now"] = [total - 2 * GIB] * gpus

    engine.sleep = sleep_and_free
    svc.load()
    free["now"] = [free_asleep] * gpus
    return svc, free


@pytest.fixture
def short_dir():
    """A short directory for a unix socket, whose path the kernel holds to 107 bytes."""
    d = Path(tempfile.mkdtemp(prefix="rs"))
    yield d
    remove_tree(d, ignore_errors=True)


def test_parse_score_request_accepts_a_well_formed_body():
    assert parse_score_request(
        {"prompt_token_ids": [1, 2, 3], "continuation_token_ids": [4],
         "prompt_positions": [1, 2]}, vocab_size=VOCAB, max_model_len=64) == (
        [1, 2, 3], [4], [1, 2])


@pytest.mark.parametrize("body, needle", [
    ({"prompt_token_ids": [1, True], "continuation_token_ids": [2]}, "non-integer"),
    ({"prompt_token_ids": [1, VOCAB], "continuation_token_ids": [2]}, "outside the vocabulary"),
    ({"prompt_token_ids": [1], "continuation_token_ids": [-1]}, "outside the vocabulary"),
    ({"prompt_token_ids": [], "continuation_token_ids": [2]}, "empty"),
    ({"prompt_token_ids": [1, 2]}, "nothing to score"),
    ({"prompt_token_ids": [1, 2], "prompt_positions": [0]}, "prompt_positions must lie"),
    ({"prompt_token_ids": [1, 2], "prompt_positions": [2]}, "prompt_positions must lie"),
    ({"prompt_token_ids": [1, 2], "prompt_positions": [1, 1]}, "at most 1 fit a prompt of 2 tokens"),
    ({"prompt_token_ids": [1] * 60, "continuation_token_ids": [2] * 4}, "exceed the limit"),
    ({"prompt_token_ids": [1], "continuation_token_ids": [2], "top_k": 5}, "unknown fields"),
    ([1, 2], "JSON object"),
])
def test_parse_score_request_refuses_with_400(body, needle):
    with pytest.raises(RequestError) as exc:
        parse_score_request(body, vocab_size=VOCAB, max_model_len=64)
    assert exc.value.status == 400
    assert needle in exc.value.body["error"]


def test_the_limit_is_one_under_the_context_length():
    """The brief states the limit as the window less one token."""
    assert parse_score_request({"prompt_token_ids": [1] * 60, "continuation_token_ids": [2] * 3},
                               vocab_size=VOCAB, max_model_len=64)[1] == [2] * 3
    with pytest.raises(RequestError, match="64 tokens exceed the limit of 63"):
        parse_score_request({"prompt_token_ids": [1] * 60, "continuation_token_ids": [2] * 4},
                            vocab_size=VOCAB, max_model_len=64)


def test_position_record_ranks_by_logprob_then_id_and_drops_the_extra_entry():
    row = {t: _Entry(-1.0 - t * 0.01) for t in range(30)}
    row[500] = _Entry(-9.0)                     # the occupying token, outside the top 21
    row[2] = _Entry(row[1].logprob)             # a tie, broken on the id
    rec = position_record(row, 500, 21)
    assert rec["token_id"] == 500 and rec["logprob"] == -9.0
    ids = [t for t, _ in rec["top21"]]
    assert len(ids) == 21 and 500 not in ids
    assert ids[:3] == [0, 1, 2]


def test_position_record_refuses_a_row_without_the_occupying_token():
    with pytest.raises(RuntimeError, match="does not name that token"):
        position_record({1: _Entry(-0.1)}, 7, 21)


def test_load_ends_asleep_with_the_measured_wake_footprint():
    engine = _StubEngine()
    svc, _ = _service(engine)
    assert svc.state == "asleep" and svc.phase is None
    assert engine.calls == ["sleep"]
    assert svc.awake_used_bytes == [80 * GIB] * 2
    assert svc.asleep_used_bytes == [2 * GIB] * 2
    assert svc.wake_need_bytes == [78 * GIB + service.WAKE_MARGIN_BYTES] * 2


def test_the_start_s_first_sleep_still_reports_starting():
    """The load ends with a sleep of tens of seconds; the service is not ready until it
    is over, so /status must not report "sleeping" (which a caller reads as built)."""
    engine = _StubEngine()
    seen = []
    svc = None

    def sleep():
        seen.append((svc.state, svc.phase))

    engine.sleep = sleep
    svc = Service(engine_factory=lambda: engine, vocab_size=VOCAB, max_model_len=64,
                  top_logprobs=21, idle_sleep_sec=120, gpu_memory=lambda: [], config={})
    svc.load()
    assert seen == [("starting", "loading")] and svc.state == "asleep"


def test_the_vocabulary_width_comes_from_the_engine_when_not_given():
    engine = _StubEngine()
    engine.vocab_size = VOCAB
    svc = Service(engine_factory=lambda: engine, vocab_size=None, max_model_len=64,
                  top_logprobs=21, idle_sleep_sec=120, gpu_memory=lambda: [], config={})
    svc.load()
    assert svc.score({"prompt_token_ids": [1],
                      "continuation_token_ids": [VOCAB - 1]})["n_continuation"] == 1
    with pytest.raises(RequestError, match="outside the vocabulary"):
        svc.score({"prompt_token_ids": [1], "continuation_token_ids": [VOCAB]})


def test_score_wakes_an_asleep_engine_and_returns_forced_rows():
    engine = _StubEngine()
    svc, _ = _service(engine)
    out = svc.score({"prompt_token_ids": [5, 6, 7], "continuation_token_ids": [8, 9],
                     "prompt_positions": [2]})
    assert engine.calls == ["sleep", "wake", "rows"]
    assert out["woke"] is True and svc.state == "awake"
    assert [r["token_id"] for r in out["continuation"]] == [8, 9]
    assert out["continuation"][0] == position_record(_row(8, 3), 8, 21)
    assert out["prompt"] == [{"position": 2, **position_record(_row(7, 2), 7, 21)}]
    assert sorted(out) == ["continuation", "n_continuation", "n_prompt", "prompt", "score_sec", "wake_sec",
                           "woke"]
    again = svc.score({"prompt_token_ids": [5], "continuation_token_ids": [8]})
    assert again["woke"] is False and engine.calls[-1] == "rows"
    assert "prompt" not in again


def test_score_answers_507_without_waking_when_a_device_lacks_room():
    engine = _StubEngine()
    svc, free = _service(engine, free_asleep=100 * GIB)
    free["now"] = [100 * GIB, 40 * GIB]
    with pytest.raises(RequestError) as exc:
        svc.score({"prompt_token_ids": [5], "continuation_token_ids": [8]})
    assert exc.value.status == 507
    body = exc.value.body
    assert body["gpu_free_bytes"] == [100 * GIB, 40 * GIB]
    assert body["wake_need_bytes"] == svc.wake_need_bytes
    assert "Stop or shrink your own server" in body["instruction"]
    assert "wake" not in engine.calls and svc.state == "asleep"


def test_a_wake_that_fails_goes_back_to_sleep_and_answers_507():
    engine = _StubEngine(fail_wake=True)
    svc, _ = _service(engine)
    with pytest.raises(RequestError) as exc:
        svc.wake()
    assert exc.value.status == 507 and "wake failed" in exc.value.body["error"]
    assert engine.calls == ["sleep", "wake", "sleep"] and svc.state == "asleep"


def test_idle_engine_goes_back_to_sleep_after_idle_sleep_sec():
    clock = _Clock()
    engine = _StubEngine()
    svc, _ = _service(engine, clock=clock)
    svc.wake()
    clock.now += 119
    svc.sleep_if_idle()
    assert svc.state == "awake"
    clock.now += 2
    svc.sleep_if_idle()
    assert svc.state == "asleep" and engine.calls[-1] == "sleep"


def test_a_score_counts_as_use_for_the_idle_sleep():
    clock = _Clock()
    engine = _StubEngine()
    svc, _ = _service(engine, clock=clock)
    svc.score({"prompt_token_ids": [5], "continuation_token_ids": [8]})
    clock.now += 100
    svc.score({"prompt_token_ids": [5], "continuation_token_ids": [8]})
    clock.now += 100
    svc.sleep_if_idle()
    assert svc.state == "awake"
    clock.now += 21
    svc.sleep_if_idle()
    assert svc.state == "asleep"


def test_requests_before_load_or_after_failure_answer_503_with_the_state():
    clock = _Clock()
    svc = Service(engine_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
                  vocab_size=VOCAB, max_model_len=64, top_logprobs=21, idle_sleep_sec=1,
                  gpu_memory=lambda: [], config={}, clock=clock)
    assert svc.status()["state"] == "starting"
    clock.now += 42.0
    for call in (svc.wake, svc.sleep,
                 lambda: svc.score({"prompt_token_ids": [1], "continuation_token_ids": [2]})):
        with pytest.raises(RequestError) as exc:
            call()
        assert exc.value.status == 503
        assert exc.value.body["state"] == "starting"
        assert exc.value.body["phase"] == "loading"
        assert exc.value.body["starting_sec"] == 42.0
    svc.load()
    assert svc.state == "failed" and svc.phase is None
    assert svc.error.startswith("engine load failed") and "boom" in svc.error
    with pytest.raises(RequestError) as exc:
        svc.score({"prompt_token_ids": [1], "continuation_token_ids": [2]})
    assert exc.value.status == 503
    assert exc.value.body["state"] == "failed" and "boom" in exc.value.body["error"]


def test_status_names_every_field_the_brief_states():
    svc, _ = _service(_StubEngine())
    status = svc.status()
    for field in ("state", "gpu_free_bytes", "wake_need_bytes", "last_wake_sec", "idle_sleep_sec"):
        assert field in status, field
    assert status["idle_sleep_sec"] == 120


# ---------------------------------------------------------------------------
# Memory: a shortage is an answer, never a crash
# ---------------------------------------------------------------------------


class _Short(_StubEngine):
    """A stub engine whose sleep fails while ``host_short`` holds, as a pinned copy of the
    weights that the operating system refuses."""

    def __init__(self) -> None:
        super().__init__()
        self.host_short = False

    def sleep(self):
        self.calls.append("sleep")
        if self.host_short:
            raise RuntimeError("CUDA error: out of memory (cudaHostAlloc of the weights' copy)")


def _gpus(free_gib, *, total_gib=140):
    return lambda: [{"index": i, "free_bytes": int(f * GIB), "total_bytes": total_gib * GIB}
                    for i, f in enumerate(free_gib)]


@pytest.mark.parametrize("exc, operation, kind", [
    (MemoryError(), "load", "host"),
    (OSError(12, "Cannot allocate memory"), "load", "host"),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "load", "gpu"),
    (RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"), "wake", "gpu"),
    (RuntimeError("CUDA error: out of memory"), "sleep", "host"),
    (ValueError("Free memory on device (12.0/140.0 GiB) on startup is less than desired GPU memory "
                "utilization (0.9, 126.0 GiB)."), "load", "gpu"),
    (ValueError("No available memory for the cache blocks. Try increasing `gpu_memory_utilization`"), "load", "gpu"),
    (RuntimeError("pin_memory failed"), "sleep", "host"),
    (RuntimeError("Engine core initialization failed. See root cause above."), "load", None),
    (ValueError("no Glm5Next in registry"), "load", None),
])
def test_memory_shortage_names_the_memory_the_engines_error_says_was_short(exc, operation, kind):
    assert service.memory_shortage(exc, operation=operation) == kind


def test_a_shortage_is_read_through_the_errors_an_error_was_raised_from():
    try:
        try:
            raise MemoryError()
        except MemoryError as inner:
            raise RuntimeError("worker failed") from inner
    except RuntimeError as outer:
        assert service.memory_shortage(outer, operation="load") == "host"


def test_a_load_the_cards_lack_room_for_leaves_it_unloaded_until_a_wake_loads_it_again():
    """vLLM starts only with its fraction of every card free. Short of it, the service builds
    nothing, says which cards and by how much, and answers 507 until a wake loads it again."""
    engine = _StubEngine()
    free = {"now": [100, 140]}
    svc = Service(engine_factory=lambda: engine, vocab_size=VOCAB, max_model_len=64, top_logprobs=21,
                  idle_sleep_sec=120, gpu_memory=lambda: _gpus(free["now"])(), config={}, load_fraction=0.9)
    svc.load()
    assert svc.state == "unloaded" and engine.calls == []
    short = svc.memory_error
    assert (short["memory"], short["operation"]) == ("gpu", "load")
    assert short["gpu_free_bytes"] == [100 * GIB, 140 * GIB] and short["gpu_need_bytes"] == [126 * GIB] * 2
    assert svc.status()["memory_error"] == short
    with pytest.raises(RequestError) as exc:
        svc.score({"prompt_token_ids": [1], "continuation_token_ids": [2]})
    assert exc.value.status == 507 and exc.value.body["state"] == "unloaded"
    assert exc.value.body["memory"] == "gpu" and exc.value.body["gpu_need_bytes"] == [126 * GIB] * 2
    assert exc.value.body["instruction"] == ("Stop or shrink your own server so that every device has at least "
                                             "gpu_need_bytes free, then POST /wake to load the reference again.")
    assert svc.sleep()["state"] == "unloaded"
    # Still short: the wake loads again, in the background, and finds the same.
    assert svc.wake()["state"] in ("starting", "unloaded")
    deadline = time.monotonic() + 5
    while svc.state == "starting":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert svc.state == "unloaded"
    free["now"] = [140, 140]
    assert svc.wake()["state"] in ("starting", "asleep")
    while svc.state == "starting":
        assert time.monotonic() < deadline
        time.sleep(0.01)
    assert svc.state == "asleep" and svc.memory_error is None and engine.calls == ["sleep"]


@pytest.mark.parametrize("raised, state", [
    (MemoryError(), "unloaded"),
    (RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB"), "unloaded"),
    (RuntimeError("Engine core initialization failed. See root cause above."), "failed"),
])
def test_an_engine_that_fails_to_build_for_lack_of_memory_leaves_it_unloaded_and_otherwise_failed(raised, state):
    def factory():
        raise raised

    svc = Service(engine_factory=factory, vocab_size=VOCAB, max_model_len=64, top_logprobs=21,
                  idle_sleep_sec=120, gpu_memory=_gpus([140]), config={}, load_fraction=0.9)
    svc.load()
    assert svc.state == state
    if state == "unloaded":
        assert svc.memory_error["error"] == repr(raised) and svc.error is None
    else:
        assert svc.memory_error is None and svc.error.startswith("engine load failed")


def test_a_first_sleep_short_of_host_memory_leaves_the_engine_awake_and_a_later_sleep_measures_it():
    engine = _Short()
    engine.host_short = True
    svc = Service(engine_factory=lambda: engine, vocab_size=VOCAB, max_model_len=64, top_logprobs=21,
                  idle_sleep_sec=120, gpu_memory=_gpus([10]), config={})
    svc.load()
    assert svc.state == "awake" and svc.startup_sec is not None and svc.wake_need_bytes is None
    assert (svc.memory_error["memory"], svc.memory_error["operation"]) == ("host", "sleep")
    with pytest.raises(RequestError) as exc:
        svc.sleep()
    assert exc.value.status == 507 and exc.value.body["memory"] == "host"
    assert exc.value.body["instruction"] == "Release host memory your processes hold, then retry."
    assert svc.state == "awake"
    # It scores while awake.
    assert svc.score({"prompt_token_ids": [1], "continuation_token_ids": [2]})["woke"] is False
    engine.host_short = False
    assert svc.sleep()["state"] == "asleep" and svc.memory_error is None
    assert svc.wake_need_bytes == [service.WAKE_MARGIN_BYTES]


def test_an_idle_sleep_short_of_host_memory_stays_awake_and_tries_again_after_another_idle_period():
    clock = _Clock()
    engine = _Short()
    svc, _ = _service(engine, clock=clock)
    svc.wake()
    engine.host_short = True
    clock.now += 121
    svc.sleep_if_idle()
    assert svc.state == "awake" and svc.memory_error["memory"] == "host"
    tried = engine.calls.count("sleep")
    clock.now += 60
    svc.sleep_if_idle()
    assert engine.calls.count("sleep") == tried
    engine.host_short = False
    clock.now += 61
    svc.sleep_if_idle()
    assert svc.state == "asleep" and svc.memory_error is None


def test_a_sleep_that_fails_otherwise_leaves_the_service_failed():
    engine = _StubEngine()
    svc, _ = _service(engine)
    svc.wake()

    def broken():
        raise RuntimeError("worker died")

    engine.sleep = broken
    with pytest.raises(RequestError) as exc:
        svc.sleep()
    assert exc.value.status == 503 and svc.state == "failed" and "worker died" in svc.error


def test_a_wake_the_cards_lack_room_for_names_the_gpu():
    engine = _StubEngine()
    svc, free = _service(engine)
    free["now"] = [1 * GIB] * 2
    with pytest.raises(RequestError) as exc:
        svc.wake()
    assert exc.value.status == 507 and exc.value.body["memory"] == "gpu"
    assert svc.status()["memory_error"]["gpu_need_bytes"] == svc.wake_need_bytes


def test_shutdown_ends_a_built_engine_and_is_a_no_op_before_one_exists():
    engine = _StubEngine()
    svc, _free = _service(engine)
    svc.shutdown()
    assert engine.calls[-1] == "shutdown"
    unbuilt = Service(engine_factory=lambda: engine, vocab_size=VOCAB, max_model_len=64,
                      top_logprobs=21, idle_sleep_sec=1, gpu_memory=lambda: [], config={})
    unbuilt.shutdown()  # nothing built yet: returns without touching an engine


def _http(path: str, method: str, route: str, body=None) -> tuple[int, dict]:
    data = b"" if body is None else (body if isinstance(body, bytes)
                                     else json.dumps(body).encode())
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(path)
        s.sendall(f"{method} {route} HTTP/1.0\r\nContent-Length: {len(data)}\r\n\r\n"
                  .encode() + data)
        raw = b""
        while chunk := s.recv(65536):
            raw += chunk
    head, _, payload = raw.partition(b"\r\n\r\n")
    return int(head.split()[1]), json.loads(payload)


@pytest.fixture
def served(short_dir):
    engine = _StubEngine()
    svc, free = _service(engine)
    sock = str(short_dir / "ref.sock")
    server = UnixHTTPServer(sock, svc)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield sock, svc, free
    server.shutdown()
    server.server_close()


def test_http_routes_over_the_unix_socket(served):
    sock, svc, free = served
    assert os.stat(sock).st_mode & 0o777 == 0o666
    code, status = _http(sock, "GET", "/status")
    assert code == 200 and status["state"] == "asleep"
    assert status["gpu_free_bytes"] == [100 * GIB] * 2 and status["engine"] == {"k": "v"}
    code, out = _http(sock, "POST", "/score",
                      {"prompt_token_ids": [5, 6], "continuation_token_ids": [8]})
    assert code == 200 and out["woke"] is True and len(out["continuation"]) == 1
    code, out = _http(sock, "POST", "/sleep")
    assert code == 200 and out["state"] == "asleep"
    free["now"] = [1 * GIB] * 2
    code, out = _http(sock, "POST", "/wake")
    assert code == 507 and out["gpu_free_bytes"] == [1 * GIB] * 2
    assert _http(sock, "POST", "/score", b"{not json")[0] == 400
    assert _http(sock, "GET", "/nope")[0] == 404
    assert _http(sock, "GET", "/score")[0] == 405


def test_a_body_over_the_limit_answers_413(served):
    sock, _, _ = served
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(sock)
        s.sendall(f"POST /score HTTP/1.0\r\nContent-Length: {service.MAX_BODY_BYTES + 1}\r\n\r\n".encode())
        raw = b""
        while chunk := s.recv(65536):
            raw += chunk
    assert raw.split()[1] == b"413"
    assert service.MAX_BODY_BYTES == 64 * 1024 * 1024


@pytest.mark.parametrize("stated", ["-1", f"-{64 * 1024 * 1024 + 1}", "many"])
def test_a_content_length_that_is_no_length_answers_400_without_reading_a_body(served, stated):
    # A negative length handed to ``read`` takes the stream to its end, so the cap would
    # not hold. The client here keeps its side open and sends a body the answer must not
    # wait for: a server that read to the end of the stream would never answer.
    sock, svc, _ = served
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(10)
        s.connect(sock)
        s.sendall(f"POST /score HTTP/1.0\r\nContent-Length: {stated}\r\n\r\n".encode()
                  + b'{"prompt_token_ids": [5, 6], "continuation_token_ids": [8]}')
        raw = b""
        while chunk := s.recv(65536):
            raw += chunk
    head, _, payload = raw.partition(b"\r\n\r\n")
    assert head.split()[1] == b"400"
    assert json.loads(payload) == {"error": "Content-Length must be a non-negative integer"}
    assert svc.state == "asleep"


def test_a_score_returns_no_more_prompt_records_than_the_prompt_has_positions(served):
    # One record per entry of ``prompt_positions``: a list longer than the prompt's
    # positions can only repeat, and its reply would grow with the list alone.
    sock, svc, _ = served
    code, out = _http(sock, "POST", "/score",
                      {"prompt_token_ids": [5, 6], "prompt_positions": [1] * 200_000})
    assert code == 400
    assert out == {"error": "prompt_positions has 200000 entries; at most 1 fit a prompt of "
                            "2 tokens"}
    assert svc.state == "asleep"
    # Every position of a prompt, once each, is the longest list served.
    code, out = _http(sock, "POST", "/score",
                      {"prompt_token_ids": [5, 6, 7, 8], "prompt_positions": [1, 2, 3]})
    assert code == 200 and [r["position"] for r in out["prompt"]] == [1, 2, 3]
    code, out = _http(sock, "POST", "/score",
                      {"prompt_token_ids": [5, 6, 7, 8], "prompt_positions": [1, 2, 3, 3]})
    assert code == 400


def test_the_status_client_reads_the_live_service(served):
    sock, _, _ = served
    status = rs.get_status(sock)
    assert status["state"] == "asleep" and status["engine"] == {"k": "v"}


def _entry_point(argv: list[str]) -> subprocess.Popen:
    return subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


def test_sigterm_closes_the_socket_and_exits_zero_while_the_engine_starts(tmp_path, short_dir):
    """The real entry point, with an engine that cannot build here (no GPU, no vLLM, or
    both): SIGTERM must still end the process promptly, cleanly and without its socket."""
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": VOCAB}))
    sock = str(short_dir / "ref.sock")
    proc = _entry_point([sys.executable, service.__file__, "--socket", sock, "--weights", str(tmp_path),
                         "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.9"])
    try:
        deadline = time.monotonic() + 30
        while not os.path.exists(sock):
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        code, status = _http(sock, "GET", "/status")
        assert code == 200 and status["state"] in ("starting", "failed")
        t0 = time.monotonic()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=service.SHUTDOWN_GRACE_SEC + 5) == 0
        assert time.monotonic() - t0 < service.SHUTDOWN_GRACE_SEC + 5
        assert not os.path.exists(sock)
    finally:
        if proc.poll() is None:
            proc.kill()


def test_the_harness_command_line_starts_the_entry_point(tmp_path, short_dir):
    """The real entry point, started with the harness's own command line: it answers on its
    socket at once, and the engine load, which cannot finish here (no GPU, no vLLM, or both),
    is reported in phase loading and then as a failed engine load."""
    (tmp_path / "config.json").write_text(json.dumps({"vocab_size": VOCAB}))
    sock = str(short_dir / "ref.sock")
    engine = rs.ServiceEngine(tensor_parallel_size=1, max_model_len=None,
                              gpu_memory_utilization=0.9, dtype="auto",
                              enforce_eager=False, batch_invariant=False)
    argv = engine.argv(weights=str(tmp_path))
    assert argv[:2] == ["python3", rs.SCRIPT_DEST]
    argv[:2] = [sys.executable, service.__file__]
    argv[argv.index("--socket") + 1] = sock
    proc = _entry_point(argv)
    try:
        deadline = time.monotonic() + 30
        while not os.path.exists(sock):
            assert proc.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        while (status := _http(sock, "GET", "/status")[1])["state"] == "starting":
            assert status["phase"] == "loading" and time.monotonic() < deadline
            time.sleep(0.1)
        assert status["state"] == "failed" and status["error"].startswith("engine load failed")
        assert status["engine"]["top_logprobs"] == 21 and status["idle_sleep_sec"] == 120
        code, body = _http(sock, "POST", "/score", {"prompt_token_ids": [1], "continuation_token_ids": [2]})
        assert code == 503 and body["state"] == "failed"
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=service.SHUTDOWN_GRACE_SEC + 5)
            except subprocess.TimeoutExpired:
                proc.kill()


def test_the_service_script_imports_nothing_of_hls():
    """It is mounted alone into a container that has no hls."""
    text = Path(service.__file__).read_text()
    assert "import hls" not in text and "from hls" not in text


def test_vocab_size_reads_the_text_config_first(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(
        {"vocab_size": 5, "text_config": {"vocab_size": 154880}}))
    assert service.vocab_size_of(str(tmp_path)) == 154880
