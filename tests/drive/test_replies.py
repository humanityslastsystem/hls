"""The drive role's gate on the contestant's replies: the cap on one reply, the cap on the drive's
trace with the replies it holds unwritten counted at their most and the requests at the JSON their
events take, and how a reply past either is recorded; the uncompressed replies every drive asks
for; and that a normal drive's trace is the one
an uncounted drive writes."""
from __future__ import annotations

import asyncio
import base64
import gzip
import json
import re
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import uvicorn
import zstandard
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

import hls.artifact.artifact_client as artifact_client_mod
import hls.spec.datasets as datasets_mod
import hls.spec.drivers as drivers_mod
from hls.artifact.artifact_client import HTTPArtifactClient, trace_growth
from hls.core.faults import ArtifactFault
from hls.core.payloads import Envelope, ImagePayload, TextPayload
from hls.core.trace import FailureReason, RequestAttempt, TraceEvent
from hls.drive import role
from hls.drive.bounds import REPLY_MAX_BYTES, REQUEST_EVENT_BYTES, request_trace_bytes
from hls.drive.replies import ReplyGate, ReplyOverBounds, _CountedResponse, counting_replies
from hls.spec.drivers import stamp_request, user_message
from hls.taskref import TaskRef, load_task
from hls.trace import store
from hls.trace.store import END_FILE, iter_events, read_end, read_trace

SAMPLE = TaskRef(module="tasks.sample_task.task")
STREAM = TaskRef(module="tests.drive.stream_task")
#: Seconds between two questions' replies, so the replies arrive in question order.
STAGGER_SEC = 0.2
#: What prefixes every failure the gate records.
WHY = "reply past the drive's bounds"
#: The trace cap a drive gets in these tests unless it names another: far past any of their traces.
ROOMY_CAP = 2**40


def _rows(n: int) -> list[dict]:
    return [{"question": f"q{i}", "choices": ["a", "b", "c", "d"], "answer": i % 4, "subject": "x", "source_index": i}
            for i in range(n)]


@pytest.fixture
def mounted(tmp_path, monkeypatch):
    """sample_task's rows where its driver reads them: ten test questions, q0 to q9."""
    for split, n in (("dev", 5), ("test", 10)):
        d = tmp_path / "data" / "tenants" / "t0" / split
        d.mkdir(parents=True)
        (d / "rows.json").write_text(json.dumps(_rows(n)))
    monkeypatch.setattr(datasets_mod, "MOUNT_ROOT", tmp_path / "data")


def _question(text: str) -> int:
    """The index of the sample_task question *text* asks."""
    return int(re.search(r"Question: q(\d+)", text).group(1))


def _chat_reply(text: str, token_ids: list[int] | None = None) -> dict:
    return {"id": "c", "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop",
                                    "token_ids": token_ids or [33]}], "usage": {"completion_tokens": 1}}


def _stream_reply(pieces: list[str]):
    for i, piece in enumerate(pieces):
        frame = {"id": "c", "choices": [{"index": 0, "delta": {"content": piece}, "token_ids": [40 + i],
                                         "finish_reason": None}]}
        yield f"data: {json.dumps(frame)}\n\n"
    yield f"data: {json.dumps({'id': 'c', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


class Server:
    """A real HTTP server on a free local port speaking sample_task's chat contract, unary or
    streamed as the request asks. Question i is answered after ``i * stagger`` seconds with
    ``text(i)``; ``bodies`` keeps each answered question's reply bytes as sent. ``/health``
    answers ``{"status": "ok"}`` plus ``health_padding`` characters of padding. With
    ``encoding`` a unary reply is sent in that content coding whatever the request accepts;
    ``accepted`` keeps each chat request's Accept-Encoding, and ``received`` the bytes of each
    question's request body. A unary reply carries ``token_ids(i)`` when given, else one id."""

    def __init__(self, text, stagger: float = STAGGER_SEC, health_padding: int = 0,
                 encoding: str | None = None, token_ids=None) -> None:
        self.text = text
        self.stagger = stagger
        self.bodies: dict[int, bytes] = {}
        self.accepted: list[str | None] = []
        self.received: dict[int, int] = {}
        app = FastAPI()

        @app.get("/health")
        async def _health():
            return {"status": "ok", **({"padding": "x" * health_padding} if health_padding else {})}

        @app.get("/empty")
        async def _empty():
            return Response(status_code=204)

        @app.post("/v1/chat/completions")
        async def _chat(request: Request):
            body = await request.json()
            self.accepted.append(request.headers.get("accept-encoding"))
            i = _question(json.dumps(body["messages"]))
            self.received[i] = len(await request.body())
            await asyncio.sleep(i * self.stagger)
            if encoding is not None:
                plain = JSONResponse(_chat_reply(self.text(i))).body
                coded = {"gzip": gzip.compress, "zstd": zstandard.ZstdCompressor().compress,
                         "identity": bytes}[encoding](plain)
                self.bodies[i] = coded
                return Response(coded, media_type="application/json", headers={"content-encoding": encoding})
            if body.get("stream"):
                pieces = [self.text(i)[k:k + 100] for k in range(0, len(self.text(i)), 100)]
                frames = list(_stream_reply(pieces))
                self.bodies[i] = "".join(frames).encode()
                return StreamingResponse(iter(frames), media_type="text/event-stream")
            reply = JSONResponse(_chat_reply(self.text(i), token_ids(i) if token_ids else None))
            self.bodies[i] = reply.body
            return reply

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.sock = sock
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.srv = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        self.thread = threading.Thread(target=self.srv.run, kwargs={"sockets": [sock]}, daemon=True)

    def __enter__(self) -> "Server":
        self.thread.start()
        deadline = time.time() + 10
        while not self.srv.started and time.time() < deadline:
            time.sleep(0.05)
        return self

    def __exit__(self, *exc) -> None:
        self.srv.should_exit = True
        self.thread.join(5)
        self.sock.close()


#: What the role's own warm-up GET /health reads from the fake server.
HEALTH_BYTES = len(b'{"status":"ok"}')


def _drive(tmp_path: Path, url: str, *extra: str, ref: TaskRef = SAMPLE, name: str = "trace",
           cap: int = ROOMY_CAP) -> Path:
    """Run the driver role in this process against *url* with *extra* flags and the trace cap
    *cap*; the trace's directory on the store. The role must exit 0."""
    ref_file = tmp_path / f"{name}-ref.json"
    ref_file.write_text(ref.model_dump_json())
    out = tmp_path / name / "outputs" / "trace"
    report = tmp_path / name / "role" / "outputs.json"
    report.parent.mkdir(parents=True)
    local = tmp_path / name / "local"
    local.mkdir()
    store.make_room(local, event_bytes=2**20)
    argv = ["--task-ref", str(ref_file), "--artifact-url", url, "--local", str(local),
            "--out", str(out), "--outputs-out", str(report), "--stream-out", str(report.with_name("stream.json")),
            "--request-timeout-sec", "60",
            "--trace-cap-bytes", str(cap), *extra]
    assert role.main(argv) == 0
    return out


def _by_question(out: Path) -> dict:
    """The trace's events by the question each asked."""
    return {_question(ev.input.model_dump_json()): ev for ev in read_trace(out).events}


def _gate(*, reply_max: int = 10**9, cap: int = 10**12, growth: float = 1.0, written=lambda: (0, 0)) -> ReplyGate:
    return ReplyGate(reply_max_bytes=reply_max, trace_cap_bytes=cap, growth=growth, written=written)


# --- one reply over the cap -------------------------------------------------------------------

def test_a_reply_over_the_cap_fails_as_the_contestants_request_and_the_trace_says_why(tmp_path, mounted):
    """One reply past the cap on one reply fails that request alone: the driver records it
    as a protocol failure with the reason in its error, and the drive goes on."""
    with Server(lambda i: "B" * (5000 if i == 4 else 1)) as srv:
        out = _drive(tmp_path, srv.url, "--reply-max-bytes", "4000")
    events = _by_question(out)
    assert len(events) == 10
    big = events[4]
    assert big.response is None and big.error.startswith(f"protocol: {WHY}")
    assert "4,000 bytes" in big.error
    assert big.failure.reason is FailureReason.PROTOCOL and big.t_recv is None
    assert all(events[i].response is not None and events[i].error is None for i in range(10) if i != 4)


def test_a_streamed_reply_over_the_cap_keeps_what_arrived_before_it_as_the_partial_output(tmp_path, mounted):
    text = "".join(f"{k:04d}" for k in range(2000))                      # 8,000 characters
    with Server(lambda i: text if i == 2 else "B") as srv:
        out = _drive(tmp_path, srv.url, "--reply-max-bytes", "6000", ref=STREAM)
    events = _by_question(out)
    failed = events[2]
    assert failed.error.startswith(f"protocol: {WHY}") and failed.failure.reason is FailureReason.PROTOCOL
    partial = failed.failure.observed.partial
    assert partial is not None and 0 < len(partial.text) < len(text) and text.startswith(partial.text)
    assert [ev.error for q, ev in events.items() if q != 2] == [None] * 9


def test_a_warm_up_reply_over_the_cap_does_not_end_the_drive(tmp_path, mounted):
    """The client warms its pool with a GET of /health and tolerates any transport error
    there; a /health reply past the cap is one, and the drive goes on."""
    with Server(lambda i: "B", stagger=0, health_padding=5000) as srv:
        out = _drive(tmp_path, srv.url, "--reply-max-bytes", "4000")
    assert [ev.error for ev in read_trace(out).events] == [None] * 10


def test_a_reply_read_once_is_counted_once():
    """A body already read is replayed from memory if it is iterated again, and not counted
    again."""
    gate = _gate()
    raw = httpx.Response(200, stream=httpx.ByteStream(b"x" * 1000))
    counted = _CountedResponse(raw, gate)

    async def read_twice() -> bytes:
        await counted.aread()
        return b"".join([chunk async for chunk in counted.aiter_bytes()])

    assert asyncio.run(read_twice()) == b"x" * 1000
    assert gate.total == 1000


def test_a_reply_of_exactly_the_cap_is_held_and_one_byte_more_is_not(tmp_path):
    """The cap is on the bytes the client reads, after any content decoding: an
    uncompressed reply's wire bytes."""
    with Server(lambda i: "x" * 3000, stagger=0) as srv:
        size = len(JSONResponse(_chat_reply("x" * 3000)).body)
        assert _submit(srv.url, _gate(reply_max=size)) is None
        failed = _submit(srv.url, _gate(reply_max=size - 1))
        assert failed.startswith(f"protocol: {WHY}")


# --- the trace cap ---------------------------------------------------------------------------------

def test_the_gate_counts_the_trace_written_and_the_replies_held_at_their_growth():
    """A reply's bytes are held from their reading until the writer reports its event written,
    and counted at the protocol's growth beside the trace written so far: the gate closes once
    the two reach the cap, and stays closed."""
    written = [0, 0]
    gate = _gate(cap=1000, growth=10.0, written=lambda: tuple(written))

    async def go() -> None:
        async def recorded(number: int, n: int) -> None:
            gate.count(n, n)
            gate.queued(number, gate.detach())

        await asyncio.create_task(recorded(0, 40))
        assert gate.held == 40, "held until the writer reports it written"
        written[:] = [1, 300]
        gate.admit()
        assert gate.held == 0
        # 300 written and 69 held at 10 trace bytes a byte: 990, under the cap.
        await asyncio.create_task(recorded(1, 69))
        gate.admit()
        assert gate.held == 69 and gate.closed is None
        with pytest.raises(ReplyOverBounds, match="trace reached its cap of 1,000 bytes"):
            await asyncio.create_task(recorded(2, 1))
        written[:] = [3, 0]
        with pytest.raises(ReplyOverBounds, match="trace reached its cap"):
            gate.admit()

    asyncio.run(go())


def test_what_a_task_holds_is_released_when_it_ends():
    """Bytes no recorded event took, as the warm-up's or a reply whose event another task records,
    are released when the task that read them ends."""
    gate = _gate()

    async def go() -> None:
        async def reader() -> None:
            gate.count(25, 25)

        await asyncio.create_task(reader())
        await asyncio.sleep(0)
        assert gate.held == 0 and gate.total == 25 and gate.held_peak == 25

    asyncio.run(go())


#: A request as the drive loop sends one: sample_task's first question, stamped.
QUESTION = stamp_request(Envelope(request_id="q0.0", contents=[TextPayload(text="Question: q0 \x7f \u00e9")]),
                         [{"role": "user", "content": "Question: q0 \x7f \u00e9"}], tenant_id="t0", session_id="q0",
                         turn_idx=0, max_tokens=4)


def test_a_request_is_held_at_its_events_json_from_its_send_until_its_event_is_written():
    """The JSON a request's event takes beside its reply (``request_trace_bytes``) is held from its
    send, charged to the task that sends it, and released as its reply's bytes are: once the writer
    reports the event the task recorded written, or when the task ends with nothing recorded."""
    written = [0, 0]
    gate = _gate(written=lambda: tuple(written))
    charge = request_trace_bytes(QUESTION)

    async def go() -> None:
        async def recorded(number: int) -> None:
            gate.sent(QUESTION)
            gate.count(10, 10)
            gate.queued(number, gate.detach())

        async def unrecorded() -> None:
            gate.sent(QUESTION)
            assert gate.held_requests == charge

        await asyncio.create_task(recorded(0))
        assert (gate.held, gate.held_requests) == (10, charge), "held until the writer reports it written"
        written[:] = [1, 100]
        gate.admit()
        assert (gate.held, gate.held_requests) == (0, 0)
        await asyncio.create_task(unrecorded())
        await asyncio.sleep(0)
        assert gate.held_requests == 0, "released when the task that sent it ended"
        assert gate.held_requests_peak == charge

    asyncio.run(go())


def test_the_gate_closes_on_the_requests_it_holds_beside_the_replies():
    """Written, the replies held at their growth and the requests held at their events' JSON
    together reach the cap: the next reply fails."""
    charge = request_trace_bytes(QUESTION)
    gate = _gate(cap=100 + 10 * 6 + 2 * charge, growth=6.0, written=lambda: (1, 100))

    async def go() -> None:
        async def sending() -> None:
            gate.sent(QUESTION)
            gate.count(9, 9)
            gate.admit()
            gate.sent(QUESTION)
            gate.admit()
            with pytest.raises(ReplyOverBounds, match=f"100 written, 10 reply bytes and {2 * charge:,} bytes of "
                                                      "requests held"):
                gate.count(10, 1)

        await asyncio.create_task(sending())

    asyncio.run(go())


def test_every_request_to_the_artifact_is_held_as_the_client_sends_it(tmp_path):
    """A request the client sends to the artifact is held at its event's JSON until the task that
    sent it records an event or ends; one to another origin, or one that is not a request of the
    client, as its warm-up, holds nothing."""
    contract = load_task(SAMPLE).contract
    with Server(lambda i: "B", stagger=0) as artifact, Server(lambda i: "B", stagger=0) as other:
        gate = _gate()

        async def go(url: str) -> int:
            client = HTTPArtifactClient(contract, url)
            try:
                await client.warmup()
                await client.submit(QUESTION, attempt=RequestAttempt())
                return gate.held_requests
            finally:
                await client.aclose()

        with counting_replies(artifact.url, gate):
            assert asyncio.run(go(other.url)) == 0
            assert asyncio.run(go(artifact.url)) == request_trace_bytes(QUESTION)
            assert gate.held_requests == 0, "released when the task that sent it ended"


def test_a_drive_past_its_trace_cap_fails_every_reply_from_the_crossing_and_none_before(tmp_path, mounted):
    """The replies arrive in question order, each long after the last one's event is written,
    and all ten requests are sent at once. The cap lets q2 through, read with q0 and q1 written
    and q2 to q9 in flight, and is reached by q3, read with q0 to q2 written and q3 to q9 in
    flight: q3 fails, and so does every reply after it, while q0 to q2 stay answered. Each event
    written takes more trace than its request was held at, so each read counts more."""
    text = "B" * 20_000
    with Server(lambda i: text, stagger=0.5) as srv:
        roomy = _drive(tmp_path, srv.url, name="roomy")
    lines = _event_bytes(roomy)
    growth = trace_growth(load_task(SAMPLE).contract)
    # The warm-up's reply, held by the drive's own task, and the question's being read.
    held = HEALTH_BYTES + len(JSONResponse(_chat_reply(text)).body)
    # Each question's request while in flight; the warm-up is not one of the client's requests.
    sent = _by_question(roomy)
    requests = [request_trace_bytes(sent[i].input) for i in range(10)]
    in_flight = sum(requests[2:])
    step = lines[2] - requests[2]
    assert step > 0
    # Half a step past what q2's read counts; half a step short of what q3's counts.
    cap = sum(lines[:2]) + int(growth * held) + in_flight + step // 2
    with Server(lambda i: text, stagger=0.5) as srv:
        out = _drive(tmp_path, srv.url, name="capped", cap=cap)
    events = _by_question(out)
    assert [events[i].error is None and events[i].response is not None for i in range(10)] == [True] * 3 + [False] * 7
    for i in range(3, 10):
        assert events[i].error.startswith(f"protocol: {WHY}: the drive's trace reached its cap of {cap:,} bytes")
        assert events[i].failure.reason is FailureReason.PROTOCOL


#: The multimodal_omni task, whose requests carry their image or clip as base64.
OMNI = TaskRef(module="tasks.multimodal_omni.task")
#: Where the omni tests' artifact serves; nothing listens there, since their transport answers.
OMNI_URL = "http://127.0.0.1:9"
#: An omni server slower than the drive's arrivals: 689 requests outstanding at the server, and
#: 1,056 events of 368.7 MB of trace written.
OMNI_OUTSTANDING, OMNI_WRITTEN = 689, (1_056, 368_700_000)
#: A 349,184-character base64 image, the size of the omni drive's images.
OMNI_IMAGE = base64.b64encode(bytes(range(256)) * 1023).decode()


def _omni_turn(i: int) -> Envelope:
    """Question *i* of multimodal_omni's image arm as the drive loop sends it: ``OMNI_IMAGE`` and a
    question, stamped (``stamp_request``)."""
    turn = Envelope(request_id=f"vqa{i}.0", contents=[ImagePayload(image_b64=OMNI_IMAGE, width=640, height=480),
                                                      TextPayload(text=f"What is in picture {i}? Answer in one word.")])
    return stamp_request(turn, [user_message(turn)], tenant_id="vqa", session_id=f"vqa{i}", turn_idx=0,
                         max_tokens=256)


def _omni_gate(written: tuple[int, int]) -> tuple[ReplyGate, int]:
    """The gate the drive role builds for multimodal_omni, at its own cap, over a trace of which
    *written* is written; and the cap."""
    task = load_task(OMNI)
    cap = task.drive_bounds.trace_cap_bytes
    args = role._parse(["--task-ref", "r", "--artifact-url", OMNI_URL, "--local", "l", "--out", "o",
                        "--outputs-out", "p", "--stream-out", "s", "--request-timeout-sec", "60",
                        "--trace-cap-bytes", str(cap)])
    return role._gate(args, task, SimpleNamespace(written=lambda: written)), cap


def _omni_drive(monkeypatch, gate: ReplyGate, requests: list[Envelope]) -> tuple[list, int]:
    """Send *requests* at once through the drive's client, counted against *gate*, to a server that
    answers none until it has all of them: each request's reply or failure, and what the gate held
    for requests once all were at the server."""
    reply = {"id": "c", "choices": [{"message": {"role": "assistant", "content": "bus"}, "finish_reason": "stop"}],
             "usage": {"completion_tokens": 1}}
    state: dict = {"arrived": 0}

    async def server(self, request: httpx.Request) -> httpx.Response:
        state["arrived"] += 1
        if state["arrived"] == len(requests):
            state["held"] = gate.held_requests
            state["everyone"].set()
        await state["everyone"].wait()
        return httpx.Response(200, json=reply, request=request)

    async def go() -> list:
        state["everyone"] = asyncio.Event()
        client = HTTPArtifactClient(load_task(OMNI).contract, OMNI_URL)
        try:
            return await asyncio.gather(*(client.submit(env) for env in requests), return_exceptions=True)
        finally:
            await client.aclose()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", server)
    with counting_replies(OMNI_URL, gate):
        replies = asyncio.run(go())
    return replies, state["held"]


def test_hundreds_of_omni_requests_at_a_slow_server_leave_the_gate_open_far_below_its_cap(monkeypatch):
    """multimodal_omni's requests carry their media as base64, which the trace writes as it was sent.
    A server slower than the drive's arrivals has hundreds of them outstanding at once, each held
    at the JSON its event will take: the gate stays open, and every reply is read, while the trace
    with every one of them written stays far below the task's cap."""
    gate, cap = _omni_gate(OMNI_WRITTEN)
    requests = [_omni_turn(i) for i in range(OMNI_OUTSTANDING)]
    replies, held = _omni_drive(monkeypatch, gate, requests)
    assert [r for r in replies if isinstance(r, BaseException)] == [] and gate.closed is None
    # Each request in an event of nothing else; the gate holds its event's own fields at their longest.
    alone = sum(store.event_line_bytes(TraceEvent(request_id=env.request_id, input=env, t_send=0.0))
                for env in requests)
    assert alone - len(requests) * REQUEST_EVENT_BYTES <= held <= alone + len(requests) * REQUEST_EVENT_BYTES
    assert OMNI_WRITTEN[1] + alone < cap / 4


def test_omni_requests_whose_events_reach_the_cap_close_the_gate(monkeypatch):
    """Held at their events' JSON, the requests at the server still count against the cap: once the
    trace written and the requests held reach it, every reply fails."""
    requests = [_omni_turn(i) for i in range(20)]
    held = sum(request_trace_bytes(env) for env in requests)
    gate, _ = _omni_gate((1_056, load_task(OMNI).drive_bounds.trace_cap_bytes - held))
    replies, at_server = _omni_drive(monkeypatch, gate, requests)
    assert at_server == held
    assert all(isinstance(r, ArtifactFault) and str(r).startswith(f"{WHY}: the drive's trace reached its cap")
               for r in replies)


def test_a_reply_after_the_gate_closed_fails_even_with_no_body(tmp_path):
    """Every reply after the gate closed fails as it arrives, before any of its body is read."""
    with Server(lambda i: "B", stagger=0) as srv:
        gate = _gate()
        gate.close("closed for the test")
        with counting_replies(srv.url, gate):
            async def empty() -> None:
                async with httpx.AsyncClient(trust_env=False) as cli:
                    await cli.get(srv.url + "/empty")

            with pytest.raises(ReplyOverBounds, match="closed for the test"):
                asyncio.run(empty())


def test_a_refused_reply_gives_its_connection_back(tmp_path):
    """A reply refused before its body is read is closed, so its connection does not stay
    taken: on a one-connection pool the next request still goes out and is refused too."""
    with Server(lambda i: "B", stagger=0) as srv:
        gate = _gate()
        gate.close("closed for the test")
        with counting_replies(srv.url, gate):
            async def two_heads() -> None:
                limits = httpx.Limits(max_connections=1)
                async with httpx.AsyncClient(trust_env=False, limits=limits, timeout=httpx.Timeout(5.0)) as cli:
                    for _ in range(2):
                        with pytest.raises(ReplyOverBounds, match="closed for the test"):
                            await cli.get(srv.url + "/health")

            asyncio.run(two_heads())


def test_the_role_says_what_it_counted(tmp_path, mounted, capsys):
    with Server(lambda i: "B", stagger=0) as srv:
        _drive(tmp_path, srv.url, cap=10**9)
    last = capsys.readouterr().err.strip().splitlines()[-1]
    assert "reply bytes, at most" in last and "held at once, and at most" in last
    assert "bytes of requests, against a 1,000,000,000-byte trace cap" in last


def test_the_role_defaults_are_the_cap_on_one_reply_and_the_trace_cap_is_named():
    base = ["--task-ref", "r", "--artifact-url", "http://h:1", "--local", "l", "--out", "o", "--outputs-out", "p",
            "--stream-out", "s", "--request-timeout-sec", "60"]
    with pytest.raises(SystemExit):
        role._parse(base)
    args = role._parse([*base, "--trace-cap-bytes", "7"])
    assert (args.reply_max_bytes, args.trace_cap_bytes, args.measured_baseline) == (REPLY_MAX_BYTES, 7, False)


@pytest.mark.parametrize("flag", ["--reply-max-bytes", "--trace-cap-bytes"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_a_bound_that_is_not_positive_is_refused(tmp_path, flag, value, capsys):
    base = ["--task-ref", "r", "--artifact-url", "http://h:1", "--local", "l", "--out", "o", "--outputs-out", "p",
            "--stream-out", "s", "--request-timeout-sec", "60", "--trace-cap-bytes", "5"]
    args = role._parse([*base, flag, "1"])
    assert getattr(args, flag[2:].replace("-", "_")) == 1
    with pytest.raises(SystemExit):
        role._parse([*base, flag, value])
    assert f"{flag}: {value} is not a positive number of bytes" in capsys.readouterr().err


def _event_bytes(out: Path) -> list[int]:
    """The bytes of each event's line in the trace's chunks, in emission order."""
    from hls.trace.store import _event_line
    return [len(_event_line(ev)) for ev in iter_events(out)]


# --- compressed replies ----------------------------------------------------------------------

#: What a compressed reply's failure records.
COMPRESSED = "a compressed reply, not asked for"


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_a_compressed_reply_fails_as_the_contestants_request(tmp_path, mounted, encoding):
    """The drive asks the artifact for uncompressed replies: a reply in another content coding
    fails as the contestant's request, whatever its size."""
    with Server(lambda i: "B", stagger=0, encoding=encoding) as srv:
        out = _drive(tmp_path, srv.url)
    assert srv.accepted == ["identity"] * 10
    events = read_trace(out).events
    assert len(events) == 10
    for ev in events:
        assert ev.response is None and ev.error == f"protocol: {COMPRESSED}: content-encoding {encoding}"
        assert ev.failure.reason is FailureReason.PROTOCOL


@pytest.mark.parametrize("encoding", ["gzip", "zstd"])
def test_a_compressed_reply_is_refused_before_any_of_its_body_is_decoded(monkeypatch, encoding):
    def refuse(self, data):
        raise AssertionError("a compressed body was decoded")

    monkeypatch.setattr(httpx._decoders.GZipDecoder, "decode", refuse)
    monkeypatch.setattr(httpx._decoders.ZStandardDecoder, "decode", refuse)
    with Server(lambda i: "B" * 100_000, stagger=0, encoding=encoding) as srv:
        gate = _gate()
        assert _submit(srv.url, gate) == f"protocol: {COMPRESSED}: content-encoding {encoding}"
    assert gate.total == 0


def test_a_reply_that_names_the_identity_coding_is_read(tmp_path, mounted):
    with Server(lambda i: "B", stagger=0, encoding="identity") as srv:
        out = _drive(tmp_path, srv.url)
    assert [ev.error for ev in read_trace(out).events] == [None] * 10


def test_only_the_artifacts_requests_ask_for_uncompressed_replies(tmp_path):
    with Server(lambda i: "B", stagger=0) as artifact, Server(lambda i: "B", stagger=0) as other:
        gate = _gate()
        assert _submit(artifact.url, gate, artifact_url=artifact.url) is None
        assert _submit(other.url, gate, artifact_url=artifact.url) is None
    assert artifact.accepted == ["identity"] and other.accepted != ["identity"]


# --- only the contestant's replies count ------------------------------------------------------

def test_a_drive_of_the_measured_baseline_counts_none_of_its_replies(tmp_path, mounted):
    """The task's own reference server is not the contestant's: none of its replies is
    counted, so neither cap can fail one."""
    with Server(lambda i: "B" * 2000, stagger=0) as srv:
        out = _drive(tmp_path, srv.url, "--measured-baseline", "--reply-max-bytes", "10", cap=10)
    assert [ev.error for ev in read_trace(out).events] == [None] * 10


def test_replies_from_another_server_than_the_artifact_are_not_counted(tmp_path):
    with Server(lambda i: "B" * 2000, stagger=0) as artifact, Server(lambda i: "B" * 5000, stagger=0) as other:
        gate = _gate(reply_max=3000)
        assert _submit(other.url, gate, artifact_url=artifact.url) is None
        assert gate.total == 0
        assert _submit(artifact.url, gate, artifact_url=artifact.url) is None
        assert gate.total == len(artifact.bodies[0])


def test_the_counting_is_removed_when_the_drive_ends(tmp_path):
    real = (httpx.AsyncHTTPTransport.handle_async_request, HTTPArtifactClient.submit)

    def now() -> tuple:
        return httpx.AsyncHTTPTransport.handle_async_request, HTTPArtifactClient.submit

    with Server(lambda i: "B", stagger=0) as srv:
        with counting_replies(srv.url, _gate(reply_max=1, cap=1)):
            assert all(a is not b for a, b in zip(now(), real))
        assert now() == real
        with pytest.raises(RuntimeError), counting_replies(srv.url, _gate(reply_max=1, cap=1)):
            raise RuntimeError("the drive raised")
        assert now() == real


# --- a normal drive is unchanged ----------------------------------------------------------------

def test_a_normal_drive_writes_the_same_trace_byte_for_byte_with_and_without_the_counting(tmp_path, mounted,
                                                                                            monkeypatch):
    """The counting only reads the bytes the client reads: a drive within its bounds
    writes exactly the trace an uncounted drive writes. The clock is held still and the
    replies arrive in question order, so the two drives can be compared byte for byte."""
    monkeypatch.setattr(drivers_mod, "harness_now", lambda: 1000.0)
    with Server(lambda i: f"answer {i} " * (i + 1)) as srv:
        counted = _drive(tmp_path, srv.url, name="counted")
        uncounted = _drive(tmp_path, srv.url, "--measured-baseline", name="uncounted")
    _assert_same_trace(counted, uncounted)
    assert len(read_trace(counted).events) == 10


def test_a_streamed_normal_drive_writes_the_same_trace_with_and_without_the_counting(tmp_path, mounted, monkeypatch):
    monkeypatch.setattr(drivers_mod, "harness_now", lambda: 1000.0)
    # The client stamps each frame's arrival; held still too, in that module alone.
    monkeypatch.setattr(artifact_client_mod, "time", SimpleNamespace(monotonic=lambda: 2000.0))
    with Server(lambda i: f"answer {i} " * (40 * i + 1)) as srv:
        counted = _drive(tmp_path, srv.url, ref=STREAM, name="counted")
        uncounted = _drive(tmp_path, srv.url, "--measured-baseline", ref=STREAM, name="uncounted")
    _assert_same_trace(counted, uncounted)
    assert [ev.error for ev in read_trace(counted).events] == [None] * 10


def _assert_same_trace(a: Path, b: Path) -> None:
    """Every file of trace *a* is byte for byte *b*'s, and the two END.json differ only in
    when their writer finished."""
    files = sorted(p.name for p in a.iterdir())
    assert files == sorted(p.name for p in b.iterdir()) and len(files) >= 3
    for name in files:
        if name != END_FILE:
            assert (a / name).read_bytes() == (b / name).read_bytes(), name
    assert read_end(a).model_copy(update={"written_at": 0.0}) == read_end(b).model_copy(update={"written_at": 0.0})


# --- helpers ----------------------------------------------------------------------------------------

def _submit(url: str, gate: ReplyGate, *, artifact_url: str | None = None) -> str | None:
    """One sample_task request to *url* through the client, counted as the role counts
    the drive of *artifact_url* (by default *url*): None when it is answered, else the fault
    string the driver records."""
    contract = load_task(SAMPLE).contract
    env = Envelope(request_id="q0.0", contents=[TextPayload(text="Question: q0")], metadata={"max_tokens": 4})

    async def go() -> str | None:
        client = HTTPArtifactClient(contract, url)
        try:
            await client.submit(env, attempt=RequestAttempt())
            return None
        except ArtifactFault as e:
            return f"{e.category}: {e}"
        finally:
            await client.aclose()

    with counting_replies(artifact_url or url, gate):
        return asyncio.run(go())
