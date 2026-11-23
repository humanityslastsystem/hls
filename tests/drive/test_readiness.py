from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import socket
import threading
import time
import tracemalloc

import httpx
import pytest
import uvicorn
import zstandard
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

import hls.drive.readiness as readiness
from hls.drive.readiness import NotReady, NotServing, ReadinessError, wait_healthy, wait_inference_ready
from hls.spec.contracts import RequestResponseContract


class _Resp:
    """A streamed reply in no content coding: its status and its body, *text*, or *body* as JSON."""

    encoding = "utf-8"

    def __init__(self, status_code, text="", body=None):
        self.status_code = status_code
        self.headers = httpx.Headers()
        self._content = (text if body is None else json.dumps(body)).encode()

    async def aiter_raw(self):
        yield self._content


class _StubClient:
    """Stands in for httpx.AsyncClient: streams a scripted response sequence (the last one
    repeats) and counts the probes."""

    def __init__(self, responses, calls):
        self._responses = list(responses)
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @contextlib.asynccontextmanager
    async def stream(self, method, url, json=None, headers=None):
        self._calls.append(1)
        yield self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]


@pytest.fixture
def stub(monkeypatch):
    calls: list[int] = []

    def install(responses):
        monkeypatch.setattr(readiness.httpx, "AsyncClient", lambda **kw: _StubClient(responses, calls))
        real_sleep = asyncio.sleep
        monkeypatch.setattr(readiness.asyncio, "sleep", lambda *_a, **_k: real_sleep(0))
        return calls

    return install


def _contract() -> RequestResponseContract:
    return RequestResponseContract(path="/v1/chat/completions", input_modalities=frozenset({"text"}),
                                   output_modalities=frozenset({"text"}))


def test_health_waits_through_503s_until_200(stub):
    calls = stub([_Resp(503, "loading"), _Resp(503, ""), _Resp(200)])
    asyncio.run(wait_healthy("http://x", timeout_sec=30.0))
    assert len(calls) == 3


def test_a_server_that_exited_stops_the_gate_before_any_probe(stub):
    calls = stub([_Resp(503)])
    with pytest.raises(NotServing, match="exited before it was ready"):
        asyncio.run(wait_healthy("http://x", timeout_sec=30.0, alive=lambda: False))
    assert calls == []


def test_a_server_that_exits_while_loading_stops_the_gate_at_the_next_probe(stub):
    calls = stub([_Resp(503, "loading")])
    seen: list[int] = []

    def alive() -> bool:
        seen.append(1)
        return len(seen) <= 2

    with pytest.raises(NotServing):
        asyncio.run(wait_healthy("http://x", timeout_sec=30.0, alive=alive))
    assert len(calls) == 2


def test_a_timeout_is_not_ready_and_names_the_last_body(stub):
    stub([_Resp(503, '{"status": "loading"}')])
    with pytest.raises(NotReady, match=r"last response body.*loading") as caught:
        asyncio.run(wait_healthy("http://x", timeout_sec=0.05))
    assert isinstance(caught.value, ReadinessError)


@pytest.mark.parametrize("statuses, ready", [
    ([404, 200], True),
    ([405, 422, 400], True),
    ([503, 302], True),
    ([422], False),
    ([500], False),
])
def test_the_inference_gate_opens_below_400_and_on_a_bare_400(stub, statuses, ready):
    stub([_Resp(s) for s in statuses])
    if ready:
        asyncio.run(wait_inference_ready("http://x", _contract(), timeout_sec=30.0))
    else:
        with pytest.raises(NotReady, match="inference endpoint not ready"):
            asyncio.run(wait_inference_ready("http://x", _contract(), timeout_sec=0.05))


def test_the_inference_gate_stops_when_the_server_exits(stub):
    stub([_Resp(404)])
    with pytest.raises(NotServing):
        asyncio.run(wait_inference_ready("http://x", _contract(), timeout_sec=30.0, alive=lambda: False))


# The gates against real servers on a free local port: what a contestant's server answers
# must neither keep a gate past its deadline, nor grow the probing process's memory, nor end
# the gate with anything but NotReady or NotServing.


@pytest.fixture
def serve(monkeypatch):
    """Start apps on real local servers; the gates poll every 0.1 s."""
    monkeypatch.setattr(readiness, "HEALTH_POLL_SEC", 0.1)
    monkeypatch.setattr(readiness, "INFERENCE_POLL_SEC", 0.1)
    running = []

    def start(app: FastAPI) -> str:
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        srv = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        thread = threading.Thread(target=srv.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while not srv.started and time.time() < deadline:
            time.sleep(0.05)
        running.append((srv, thread, sock))
        return f"http://127.0.0.1:{sock.getsockname()[1]}"

    yield start
    for srv, thread, sock in running:
        srv.should_exit = True
        thread.join(5)
        sock.close()


def _streaming(status: int, chunk: bytes, every_sec: float) -> FastAPI:
    """An app whose health and inference routes answer *status* with a body that never ends:
    *chunk* every *every_sec* seconds."""
    app = FastAPI()

    async def endless():
        while True:
            yield chunk
            await asyncio.sleep(every_sec)

    @app.get("/health")
    async def _health():
        return StreamingResponse(endless(), status_code=status, media_type="text/plain")

    @app.post("/v1/chat/completions")
    async def _chat():
        return StreamingResponse(endless(), status_code=status, media_type="text/plain")

    return app


def _gate(gate: str, url: str, timeout_sec: float):
    if gate == "health":
        return wait_healthy(url, timeout_sec=timeout_sec, alive=lambda: True)
    return wait_inference_ready(url, _contract(), timeout_sec=timeout_sec, alive=lambda: True)


async def _bounded(gate, outer_sec: float) -> str:
    """Run *gate*, cut off at *outer_sec*: a gate still running then has outlived its deadline."""
    try:
        await asyncio.wait_for(gate, outer_sec)
    except NotReady:
        return "NotReady"
    except TimeoutError:
        return "still probing"
    return "ready"


@pytest.mark.parametrize("gate", ["health", "inference"])
def test_an_endless_503_body_ends_the_gate_on_time_with_bounded_memory(serve, gate):
    """A body that never ends is read only as far as its first bytes: the gate ends NotReady
    at its deadline, and the probing process holds no more of the body than that."""
    url = serve(_streaming(503, b"x" * 65536, 0.001))
    tracemalloc.start()
    started = time.monotonic()
    try:
        outcome = asyncio.run(_bounded(_gate(gate, url, 2.0), 10.0))
        took = time.monotonic() - started
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert (outcome, took < 4.0) == ("NotReady", True), (outcome, took)
    assert peak < 8 * 2**20, f"the gate held {peak / 2**20:.1f} MiB"


@pytest.mark.parametrize("gate", ["health", "inference"])
def test_a_body_that_drips_forever_ends_at_the_client_timeout(serve, gate, monkeypatch):
    """A byte at a time, each well inside the client's read timeout, never ends a probe by
    itself: the whole probe is bounded by the client timeout, so the gate ends NotReady one
    client timeout after the probe started."""
    monkeypatch.setattr(readiness, "HEALTH_PROBE_TIMEOUT_SEC", 1.5)
    monkeypatch.setattr(readiness, "INFERENCE_PROBE_TIMEOUT_SEC", 1.5)
    url = serve(_streaming(503, b"x", 0.2))
    started = time.monotonic()
    outcome = asyncio.run(_bounded(_gate(gate, url, 0.5), 10.0))
    took = time.monotonic() - started
    assert (outcome, 1.4 < took < 3.0) == ("NotReady", True), (outcome, took)


@pytest.mark.parametrize("gate", ["health", "inference"])
def test_a_probe_answered_after_the_deadline_within_the_client_timeout_passes(serve, gate):
    """A probe started before the deadline counts if its answer comes within the
    client timeout, so a server that becomes ready right at its deadline passes."""
    app = FastAPI()

    @app.get("/health")
    async def _health():
        await asyncio.sleep(1.0)
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def _chat():
        await asyncio.sleep(1.0)
        return {"id": "c"}

    url = serve(app)
    started = time.monotonic()
    asyncio.run(_gate(gate, url, 0.3))
    assert time.monotonic() - started >= 0.9


@pytest.mark.parametrize("gate", ["health", "inference"])
@pytest.mark.parametrize("status", [503, 200])
def test_a_reply_whose_body_cannot_be_decoded_is_not_ready_yet(serve, gate, status):
    """A body that claims gzip and is not cannot be read, and the gate never takes a reply
    it cannot read for ready: not yet, until the deadline ends the gate NotReady."""
    app = FastAPI()

    def bad() -> Response:
        return Response(content=b"not gzip at all", status_code=status, headers={"Content-Encoding": "gzip"})

    @app.get("/health")
    async def _health():
        return bad()

    @app.post("/v1/chat/completions")
    async def _chat():
        return bad()

    url = serve(app)
    with pytest.raises(NotReady):
        asyncio.run(_gate(gate, url, 1.0))


def _coded(encoding: str, plain: bytes) -> bytes:
    """*plain* in the content coding *encoding*, codings listed in the order they were applied."""
    for coding in (c.strip() for c in encoding.split(",")):
        plain = gzip.compress(plain) if coding == "gzip" else zstandard.ZstdCompressor().compress(plain)
    return plain


#: Bytes of zeros the compressed bodies below stand for: a few KiB on the wire.
EXPANDS_TO = 256 * 2**20


@pytest.mark.parametrize("gate", ["health", "inference"])
@pytest.mark.parametrize("status", [503, 200])
@pytest.mark.parametrize("encoding", ["zstd", "gzip", "zstd, zstd"])
def test_a_compressed_reply_is_refused_at_its_headers_and_never_decoded(serve, monkeypatch, gate, status, encoding):
    """The probes ask for an uncompressed reply. A server that answers in a content coding all the
    same is refused at its headers, before any of its body is read: a few KiB that expand to any size
    are never expanded in the probing process, and the gate waits on, NotReady by its deadline, a 200
    so coded included."""
    def decoded(self, data):
        raise AssertionError("a compressed body was decoded")

    monkeypatch.setattr(httpx._decoders.ZStandardDecoder, "decode", decoded)
    monkeypatch.setattr(httpx._decoders.GZipDecoder, "decode", decoded)
    body = _coded(encoding, bytes(EXPANDS_TO))
    assert len(body) < 2**20
    app = FastAPI()

    def coded() -> Response:
        return Response(content=body, status_code=status, headers={"Content-Encoding": encoding})

    @app.get("/health")
    async def _health():
        return coded()

    @app.post("/v1/chat/completions")
    async def _chat():
        return coded()

    url = serve(app)
    with pytest.raises(NotReady):
        asyncio.run(_gate(gate, url, 1.0))


@pytest.mark.parametrize("gate", ["health", "inference"])
def test_a_body_that_expands_without_bound_costs_the_probing_process_nothing(serve, gate):
    """A zstd body of zeros that expands to 256 MiB, answered to every probe: the gate ends NotReady
    holding no more than a probe's bound of it."""
    body = _coded("zstd", bytes(EXPANDS_TO))
    app = FastAPI()

    @app.get("/health")
    async def _health():
        return Response(content=body, status_code=503, headers={"Content-Encoding": "zstd"})

    @app.post("/v1/chat/completions")
    async def _chat():
        return Response(content=body, status_code=503, headers={"Content-Encoding": "zstd"})

    url = serve(app)
    tracemalloc.start()
    try:
        outcome = asyncio.run(_bounded(_gate(gate, url, 1.0), 10.0))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert outcome == "NotReady"
    assert peak < 8 * 2**20, f"the gate held {peak / 2**20:.1f} MiB"


def test_the_probes_ask_for_an_uncompressed_reply(serve):
    """Both gates send ``Accept-Encoding: identity``, so a server that honours it never codes a
    reply, and one that does not is refused (above)."""
    asked: list[str | None] = []
    app = FastAPI()

    @app.get("/health")
    async def _health(request: Request):
        asked.append(request.headers.get("accept-encoding"))
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def _chat(request: Request):
        asked.append(request.headers.get("accept-encoding"))
        return {"id": "c"}

    url = serve(app)
    asyncio.run(wait_healthy(url, timeout_sec=3.0, alive=lambda: True))
    asyncio.run(wait_inference_ready(url, _contract(), timeout_sec=3.0, alive=lambda: True))
    assert asked == ["identity", "identity"]


@pytest.mark.parametrize("charset", ["rot_13", "base64", "idna"])
def test_a_body_in_a_charset_that_is_not_text_is_not_ready_yet(serve, charset):
    """A server names the charset its body is read in. One that is not a text encoding still
    leaves the gate waiting for a 200, and the deadline ends it NotReady."""
    app = FastAPI()

    @app.get("/health")
    async def _health():
        return Response(content=b"\xffloading", status_code=503,
                        headers={"Content-Type": f"text/plain; charset={charset}"})

    url = serve(app)
    with pytest.raises(NotReady, match="last response body"):
        asyncio.run(wait_healthy(url, timeout_sec=1.0, alive=lambda: True))


def test_a_ready_server_passes_both_gates(serve):
    app = FastAPI()
    loads = {"n": 0}

    @app.get("/health")
    async def _health():
        loads["n"] += 1
        if loads["n"] < 3:
            return Response(content=b"loading", status_code=503)
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        return {"id": "c", "choices": [{"message": {"role": "assistant", "content": "B"}}]}

    url = serve(app)
    asyncio.run(wait_healthy(url, timeout_sec=10.0, alive=lambda: True))
    assert loads["n"] == 3
    asyncio.run(wait_inference_ready(url, _contract(), timeout_sec=10.0, alive=lambda: True))


def test_the_inference_gate_posts_the_contracts_own_probe_body(serve):
    """The inference gate posts the contract protocol's own ``probe_request()`` to the
    contract's path, so the gate means the same thing for every wire protocol."""
    from hls.spec.wire import OpenAIChatProtocol

    class _MarkedProtocol(OpenAIChatProtocol):
        def probe_request(self) -> dict:
            return {"ping": "marked-probe"}

    seen: list[tuple[str, bytes]] = []
    app = FastAPI()

    @app.post("/custom/infer")
    async def _infer(request: Request):
        seen.append((request.url.path, await request.body()))
        return {"ok": True}

    url = serve(app)
    contract = RequestResponseContract(path="/custom/infer", input_modalities=frozenset({"text"}),
                                       output_modalities=frozenset({"text"}), protocol=_MarkedProtocol())
    asyncio.run(wait_inference_ready(url, contract, timeout_sec=10.0, alive=lambda: True))
    assert seen and all(path == "/custom/infer" for path, _ in seen)
    assert json.loads(seen[0][1]) == {"ping": "marked-probe"}


def test_the_probes_ignore_proxy_variables(serve, monkeypatch):
    """Both gates reach the served address directly: a proxy named in the environment, here one
    that refuses every connection, is never used for a node-internal probe."""
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")
    app = FastAPI()

    @app.get("/health")
    async def _health():
        return {"status": "ok"}

    @app.post("/v1/chat/completions")
    async def _chat():
        return {"id": "c"}

    url = serve(app)
    asyncio.run(wait_healthy(url, timeout_sec=3.0, alive=lambda: True))
    asyncio.run(wait_inference_ready(url, _contract(), timeout_sec=3.0, alive=lambda: True))
