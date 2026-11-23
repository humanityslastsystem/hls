"""hls.reference.service_harness: the engine a task's service builds, and the two ways the
harness reads its ``/status`` (from the host, and from inside the build sandbox)."""
from __future__ import annotations

import http.server
import os
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

import pytest

from hls.core.decode import Greedy
from hls.reference import service_harness as rs
from hls.reference.references import HFReference, VLLMReference
from hls.util import remove_tree
from hls.verify.base import WiringError

_REF = VLLMReference(decode=Greedy(), tensor_parallel_size=8, max_model_len=4096)


def _task(*refs, tenants=1):
    verifiers = [SimpleNamespace(reference=r) for r in refs]
    return SimpleNamespace(id="t", effective_tenants=lambda: [
        SimpleNamespace(verifiers=verifiers) for _ in range(tenants)])


def test_the_engine_is_the_references_field_for_field():
    ref = VLLMReference(decode=Greedy(), tensor_parallel_size=4, max_model_len=8192,
                        gpu_memory_utilization=0.8, dtype="bfloat16", enforce_eager=True,
                        batch_invariant=True)
    engine = rs.service_engine_for(_task(ref))
    assert engine == rs.ServiceEngine(tensor_parallel_size=4, max_model_len=8192,
                                      gpu_memory_utilization=0.8, dtype="bfloat16",
                                      enforce_eager=True, batch_invariant=True)
    argv = engine.argv(weights="/data/weights")
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert "--enforce-eager" in argv and "--batch-invariant" in argv


def test_an_unset_context_length_is_left_to_vllm():
    argv = rs.ServiceEngine.from_reference(
        VLLMReference(decode=Greedy())).argv(weights="/data/weights")
    assert "--max-model-len" not in argv
    assert "--enforce-eager" not in argv and "--batch-invariant" not in argv


def test_the_command_line_is_the_services_own():
    assert rs.ServiceEngine.from_reference(_REF).argv(weights="/data/weights") == [
        "python3", "/opt/hls-reference-service/service.py", "--socket", "/run/hls-reference/ref.sock",
        "--weights", "/data/weights", "--tensor-parallel-size", "8", "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9", "--dtype", "auto", "--top-logprobs", "21", "--idle-sleep-sec", "120",
    ]


def test_the_same_reference_on_several_verifiers_or_tenants_is_one_engine():
    assert rs.service_engine_for(_task(_REF, _REF.model_copy(), tenants=2)) == \
        rs.ServiceEngine.from_reference(_REF)


@pytest.mark.parametrize("task", [
    _task(),
    _task(HFReference(decode=Greedy())),
    _task(_REF, _REF.model_copy(update={"max_model_len": 2048})),
], ids=["none", "hf-only", "two-different"])
def test_anything_but_one_vllm_engine_refuses(task):
    with pytest.raises(WiringError, match="VLLMReference"):
        rs.service_engine_for(task)


def test_agentic_coding_gets_its_graders_engine():
    from tasks.agentic_coding.task import TASK

    assert rs.service_engine_for(TASK) == rs.ServiceEngine(
        tensor_parallel_size=8, max_model_len=1048576, gpu_memory_utilization=0.9,
        dtype="auto", enforce_eager=False, batch_invariant=False)


def test_the_scoring_service_runs_the_grader_engine_at_the_brief_width():
    # The harness launches the service with the grader's reference engine and the report
    # width the brief states, and the service names its list after that width.
    from hls.reference.service import position_record
    from hls.reference.service_harness import service_engine_for
    from hls.verify.report import TOP_LOGPROBS_REQUESTED
    from tasks.agentic_coding import task as task_mod

    argv = service_engine_for(task_mod.TASK).argv(weights="/data/weights")
    flag = {a: b for a, b in zip(argv, argv[1:])}
    assert flag["--top-logprobs"] == str(TOP_LOGPROBS_REQUESTED) == "21"
    assert flag["--tensor-parallel-size"] == "8"
    assert flag["--max-model-len"] == str(task_mod.MAX_CONTEXT_LEN)
    assert flag["--gpu-memory-utilization"] == "0.9"

    class _E:
        def __init__(self, lp):
            self.logprob = lp

    row = {t: _E(-float(t)) for t in range(TOP_LOGPROBS_REQUESTED + 3)}
    assert "top21" in position_record(row, 0, TOP_LOGPROBS_REQUESTED)


def test_the_protocol_facts_the_brief_states():
    assert (rs.SOCKET_DIR, rs.SOCKET_PATH, rs.SOCKET_ENV_VAR) == (
        "/run/hls-reference", "/run/hls-reference/ref.sock", "HLS_REFERENCE_SOCKET")
    assert (rs.IDLE_SLEEP_SEC, rs.READY_TIMEOUT_SEC) == (120, 1800.0)
    assert (rs.STARTING_STATE, rs.READY_STATE, rs.FAILED_STATE) == ("starting", "asleep", "failed")
    assert rs.READY_STATES == {"asleep", "waking", "awake", "sleeping"}


# ---------------------------------------------------------------------------
# /status, from the host and from the sandbox
# ---------------------------------------------------------------------------


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def __init__(self, path, code, body, *, drip_sec=0.0):
        self.code, self.body, self.drip_sec = code, body, drip_sec
        super().__init__(path, _Handler)


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        assert self.path == "/status"
        self.send_response(self.server.code)
        self.send_header("Content-Length", str(len(self.server.body)))
        self.end_headers()
        if self.server.drip_sec:
            try:
                for b in self.server.body:
                    self.wfile.write(bytes([b]))
                    self.wfile.flush()
                    time.sleep(self.server.drip_sec)
            except BrokenPipeError:
                pass  # the client gave up, as the test means it to
        else:
            self.wfile.write(self.server.body)

    def log_message(self, *args):
        pass


@pytest.fixture
def serve():
    """Start a status server on a fresh short socket path; yields a starter."""
    started = []
    d = tempfile.mkdtemp(prefix="hs")

    def _start(code=200, body=b'{"state": "asleep"}', **kw):
        path = os.path.join(d, f"s{len(started)}.sock")
        srv = _Server(path, code, body, **kw)
        threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01},
                         daemon=True).start()
        started.append(srv)
        return path

    yield _start
    for srv in started:
        srv.shutdown()
        srv.server_close()
    remove_tree(d, ignore_errors=True)


def test_get_status_reads_the_reply(serve):
    path = serve(body=b'{"state": "starting", "phase": "loading"}')
    assert rs.get_status(path) == {"state": "starting", "phase": "loading"}


def test_get_status_refuses_a_reply_without_a_state(serve):
    with pytest.raises(ValueError, match="state"):
        rs.get_status(serve(body=b'{"ok": true}'))
    with pytest.raises(ValueError, match="not JSON"):
        rs.get_status(serve(code=500, body=b"Internal Server Error"))


def test_get_status_on_a_missing_socket_is_an_oserror(tmp_path):
    with pytest.raises(OSError):
        rs.get_status(str(tmp_path / "absent.sock"))


def test_get_status_is_bounded_in_time_whatever_the_server_does(serve):
    """A reply that trickles in byte by byte still ends the exchange at its timeout."""
    path = serve(body=b'{"state": "asleep", "pad": "' + b"x" * 200 + b'"}', drip_sec=0.05)
    t0 = time.monotonic()
    with pytest.raises(OSError):
        rs.get_status(path, timeout=0.5)
    assert time.monotonic() - t0 < 3


def test_get_status_is_bounded_in_size(serve, monkeypatch):
    monkeypatch.setattr(rs, "STATUS_MAX_BYTES", 64)
    with pytest.raises(OSError, match="more than 64 bytes"):
        rs.get_status(serve(body=b'{"state": "asleep", "pad": "' + b"x" * 200 + b'"}'))


def test_the_probe_command_runs_and_its_output_parses(serve, monkeypatch):
    """The in-sandbox probe is a python one-liner; run it for real against a socket and
    parse what it prints the way the build does."""
    path = serve(body=b'{"state": "asleep", "gpus": 8}')
    monkeypatch.setattr(rs, "SOCKET_PATH", path)
    res = subprocess.run(rs.probe_command(python=sys.executable, timeout=5), shell=True,
                         capture_output=True, text=True, timeout=30)
    assert res.returncode == 0, res.stderr
    assert rs.read_probe(res.stdout) == {"state": "asleep", "gpus": 8}


def test_the_probe_fails_loudly_on_a_missing_socket(monkeypatch, tmp_path):
    monkeypatch.setattr(rs, "SOCKET_PATH", str(tmp_path / "absent.sock"))
    res = subprocess.run(rs.probe_command(python=sys.executable, timeout=5), shell=True,
                         capture_output=True, text=True, timeout=30)
    assert res.returncode != 0 and "FileNotFoundError" in res.stderr


def test_read_probe_refuses_what_is_not_base64():
    with pytest.raises(ValueError, match="base64"):
        rs.read_probe("not base64 !")


def test_parse_http_response_rejects_what_is_not_http():
    assert rs.parse_http_response(b"HTTP/1.0 200 OK\r\nA: b\r\n\r\nbody") == (200, b"body")
    for raw in (b"", b"garbage", b"HTTP/1.0 200 OK\r\n"):
        with pytest.raises(ValueError):
            rs.parse_http_response(raw)
