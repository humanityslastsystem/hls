import asyncio
import sys

import httpx
import pytest
from unittest.mock import AsyncMock, patch

from hls.artifact.artifact_client import (HTTPArtifactClient, _MAX_FAULT_BODY_CHARS,
                                          _status_fault_message)
from hls.core.faults import ArtifactProtocolError, ArtifactTransportError
from hls.spec.contracts import RequestResponseContract
from hls.spec.wire import OpenAIChatProtocol
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import ArtifactResponse


def _contract(**kw):
    return RequestResponseContract(
        path="/v1/chat/completions",
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}), **kw)


def test_unary_submit_returns_artifact_response():
    client = HTTPArtifactClient(
        contract=_contract(protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")
    openai_json = {
        "id": "chatcmpl-xyz",
        "choices": [{"message": {"role": "assistant", "content": "B"},
                     "finish_reason": "stop", "token_ids": [42]}],
        "usage": {"completion_tokens": 1},
    }
    with patch.object(client, "_post", new=AsyncMock(return_value=openai_json)):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        resp = asyncio.run(client.submit(env))

    assert isinstance(resp, ArtifactResponse)
    assert resp.t_first_packet is None                       # unary
    assert resp.output.request_id == "r1"                    # preserved, not "chatcmpl-xyz"
    assert resp.output.first_of("text").text == "B"
    assert resp.output.first_of("text").tokens == [42]
    assert resp.output.metadata["completion_tokens"] == 1


def test_streaming_submit_reassembles_and_times():
    client = HTTPArtifactClient(
        contract=_contract(transport="http-sse",
                           protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")

    chunks = [
        {"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906], "finish_reason": None}]},
        {"choices": [{"delta": {"content": "!"}, "token_ids": [999], "finish_reason": "stop"}]},
        {"choices": [], "usage": {"completion_tokens": 2}},
    ]

    async def fake_sse(_body):
        for c in chunks:
            yield c

    with patch.object(client, "_post_sse", new=fake_sse):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        resp = asyncio.run(client.submit(env))

    assert resp.output.first_of("text").text == "Hi!"
    assert resp.output.first_of("text").tokens == [9906, 999]
    assert resp.output.metadata["completion_tokens"] == 2
    assert resp.t_first_packet is not None
    assert len(resp.packet_timestamps) == 2


async def _alines(items):
    for it in items:
        yield it


async def _collect(agen):
    return [x async for x in agen]


def test_frame_sse_parses_data_skips_noise_and_stops_at_done():
    raw = [
        ": keep-alive comment",                                  # SSE comment -- skip
        "event: message",                                        # non-data field -- skip
        "",                                                      # blank separator -- skip
        'data: {"choices":[{"delta":{"content":"Hi"}}]}',        # data with leading space
        "",
        'data:{"choices":[{"delta":{"content":"!"}}]}',          # data WITHOUT leading space
        "data: [DONE]",                                          # terminator -- stop here
        'data: {"choices":[{"delta":{"content":"after"}}]}',     # after DONE -- must be ignored
    ]
    out = asyncio.run(_collect(HTTPArtifactClient._frame_sse(_alines(raw))))
    assert [c["choices"][0]["delta"]["content"] for c in out] == ["Hi", "!"]


def _http_req():
    return httpx.Request("POST", "http://localhost:9000/v1/chat/completions")


def test_submit_wraps_http_status_error_as_transport_fault():
    client = HTTPArtifactClient(contract=_contract(), base_url="http://localhost:9000")
    exc = httpx.HTTPStatusError(
        "503", request=_http_req(), response=httpx.Response(503, request=_http_req()))
    with patch.object(client, "_post", new=AsyncMock(side_effect=exc)):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        with pytest.raises(ArtifactTransportError):
            asyncio.run(client.submit(env))


def test_status_fault_quotes_the_servers_own_reason_not_just_the_code():
    """The status code says a request was rejected; only the body says WHY. vLLM states a
    context overflow there, and that text is what a diagnosis reads -- a bare "artifact
    returned 400" is indistinguishable from a bad param or an unknown model."""
    body = ('{"object":"error","message":"This model\'s maximum context length is 8192 '
            'tokens. However, you requested 9200 tokens","type":"BadRequestError"}')
    resp = httpx.Response(400, request=_http_req(), text=body)
    msg = _status_fault_message(resp)
    assert msg.startswith("artifact returned 400: ")
    assert "maximum context length is 8192 tokens" in msg      # verbatim
    assert "you requested 9200 tokens" in msg


def test_status_fault_bounds_a_huge_body_and_flattens_it_to_one_line():
    """A 5xx can return a whole HTML page; this string lands on every failed TraceEvent."""
    resp = httpx.Response(500, request=_http_req(), text="<html>\n  " + "x" * 5000)
    msg = _status_fault_message(resp)
    assert "\n" not in msg
    assert len(msg) < _MAX_FAULT_BODY_CHARS + 60      # code prefix + bounded body + marker
    assert msg.endswith("...")


def test_status_fault_falls_back_to_the_bare_code_for_an_empty_body():
    resp = httpx.Response(503, request=_http_req(), text="")
    assert _status_fault_message(resp) == "artifact returned 503"


def test_status_fault_survives_a_body_that_was_never_read():
    """An unread streamed body raises ResponseNotRead on .text; degrade to the code rather
    than turning a served fault into a harness crash."""
    resp = httpx.Response(400, request=_http_req(),
                          stream=httpx.AsyncByteStream())  # never read
    assert _status_fault_message(resp) == "artifact returned 400"


class _UnreadStream(httpx.AsyncByteStream):
    """A response body that is NOT pre-buffered -- what a real socket gives `client.stream`.

    `httpx.Response(..., text=...)` sets `.content` eagerly, so `.text` works even on the
    streaming path and a test built on it passes with OR without the eager read. Only an
    unread stream reproduces the ResponseNotRead that loses the server's
    reason in production, which is the whole behavior under test.
    """

    def __init__(self, data: bytes) -> None:
        self._data = data

    async def __aiter__(self):
        yield self._data


def test_streaming_submit_reports_the_error_body_of_a_rejected_stream():
    """THE path that matters for a streaming task: raise_for_status() on a STREAMED
    response does not read the body, and the stream closes when _post_sse's context exits
    -- so unless the body is read eagerly on the error path, the server's reason is gone by
    the time the fault is built and every rejection collapses to "artifact returned 400"."""
    body = ('{"object":"error","message":"This model\'s maximum context length is 8192 '
            'tokens. However, you requested 9200 tokens"}').encode()

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, stream=_UnreadStream(body),
                              headers={"content-length": str(len(body))})

    client = HTTPArtifactClient(
        contract=_contract(transport="http-sse", protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
    with pytest.raises(ArtifactTransportError) as ei:
        asyncio.run(client.submit(env))
    assert "maximum context length is 8192 tokens" in str(ei.value)
    assert "you requested 9200 tokens" in str(ei.value)


def test_streaming_submit_still_streams_a_healthy_response_incrementally():
    """The eager read is error-path ONLY: a 200 stream must still be consumed chunk by
    chunk, or the TTFT/TPOT timings this benchmark exists to measure would collapse to the
    moment the whole body finished arriving."""
    sse = ('data: {"choices":[{"delta":{"content":"Hi"},"token_ids":[5,6],'
           '"finish_reason":null}]}\n\n'
           'data: {"choices":[{"delta":{"content":"!"},"token_ids":[7],'
           '"finish_reason":"stop"}]}\n\n'
           'data: [DONE]\n\n')

    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse,
                              headers={"content-type": "text/event-stream"})

    client = HTTPArtifactClient(
        contract=_contract(transport="http-sse", protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
    resp = asyncio.run(client.submit(env))
    assert resp.output.first_of("text").text == "Hi!"
    assert resp.t_first_packet is not None


def test_submit_wraps_timeout_as_transport_fault():
    client = HTTPArtifactClient(contract=_contract(), base_url="http://localhost:9000")
    with patch.object(client, "_post", new=AsyncMock(side_effect=httpx.TimeoutException("t"))):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        with pytest.raises(ArtifactTransportError):
            asyncio.run(client.submit(env))


def test_submit_wraps_connect_error_as_transport_fault():
    client = HTTPArtifactClient(contract=_contract(), base_url="http://localhost:9000")
    with patch.object(client, "_post", new=AsyncMock(side_effect=httpx.ConnectError("refused"))):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        with pytest.raises(ArtifactTransportError):
            asyncio.run(client.submit(env))


def _served(body: str, *, transport: str = "http"):
    """Submit one request to a server that answers 200 with ``body``."""
    def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    client = HTTPArtifactClient(
        contract=_contract(transport=transport, protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    return asyncio.run(client.submit(Envelope(request_id="r1",
                                              contents=[TextPayload(text="Q?")])))


#: Bodies Python's JSON reader refuses, each the server's own output. Two are valid
#: JSON: an integer longer than the interpreter's 4,300-digit conversion limit raises
#: a plain ValueError, and nesting past the recursion limit raises RecursionError.
_UNREADABLE_JSON = {
    "not JSON": "<html>Bad Gateway</html>",
    "an integer of 5,000 digits": '{"choices": [], "id": ' + "9" * 5000 + "}",
    "nesting 100,000 deep": "[" * 100_000 + "]" * 100_000,
}


@pytest.mark.parametrize("body", list(_UNREADABLE_JSON.values()), ids=list(_UNREADABLE_JSON))
def test_an_unreadable_body_fails_its_own_request(body):
    """Outside the fault taxonomy, one such reply would end the whole drive."""
    with pytest.raises(ArtifactProtocolError):
        _served(body)


@pytest.mark.parametrize("body", list(_UNREADABLE_JSON.values()), ids=list(_UNREADABLE_JSON))
def test_an_unreadable_stream_frame_fails_its_own_request(body):
    with pytest.raises(ArtifactProtocolError):
        _served(f"data: {body}\n\ndata: [DONE]\n\n", transport="http-sse")


def test_submit_propagates_protocol_error_from_decode():
    client = HTTPArtifactClient(contract=_contract(), base_url="http://localhost:9000")
    with patch.object(client, "_post", new=AsyncMock(return_value={})):  # no choices
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        with pytest.raises(ArtifactProtocolError):
            asyncio.run(client.submit(env))


def test_request_timeout_sec_explicit():
    client = HTTPArtifactClient(
        contract=_contract(), base_url="http://x", request_timeout_sec=5.0)
    assert client.request_timeout_sec == 5.0


def test_request_timeout_sec_default_fallback():
    from hls.artifact.artifact_client import REQUEST_TIMEOUT_SEC
    client = HTTPArtifactClient(contract=_contract(), base_url="http://x")
    assert client.request_timeout_sec == REQUEST_TIMEOUT_SEC


def test_make_artifact_client_threads_timeout():
    from hls.artifact.artifact_client import make_artifact_client
    client = make_artifact_client(_contract(), "http://x", request_timeout_sec=9.0)
    assert client.request_timeout_sec == 9.0


def test_shared_client_pool_is_unbounded():
    """Open-loop invariant: the shared client must NEVER queue requests
    client-side (a bounded pool converts the workload to closed-loop above the
    cap and livelocks httpcore's queued-request rescan under load). Pins the
    pool's max_connections=None via httpx internals -- brittle on an httpx
    major bump by design: if this breaks, re-verify the invariant survived."""
    client = HTTPArtifactClient(contract=_contract(), base_url="http://x")
    pool = client._client._transport._pool
    # httpcore normalizes max_connections=None to sys.maxsize.
    assert pool._max_connections in (None, sys.maxsize)


def test_warmup_gets_health_on_shared_client():
    # warmup() must establish a pooled keep-alive connection on the SAME shared
    # client the load reuses -- a GET to the artifact origin's /health (same
    # scheme+host+port as the inference POSTs -> same httpx connection pool).
    client = HTTPArtifactClient(contract=_contract(), base_url="http://localhost:9000")
    with patch.object(client._client, "get",
                      new=AsyncMock(return_value=httpx.Response(200))) as g:
        asyncio.run(client.warmup())
    g.assert_awaited_once()
    called = g.await_args
    url = called.args[0] if called.args else called.kwargs.get("url")
    assert url == "http://localhost:9000/health"


def test_warmup_swallows_transport_error():
    # Best-effort: a server that refuses the warmup must not fail the run -- it
    # degrades to a cold pool (the first real request pays the handshake).
    client = HTTPArtifactClient(contract=_contract(), base_url="http://localhost:9000")
    with patch.object(client._client, "get",
                      new=AsyncMock(side_effect=httpx.ConnectError("refused"))) as g:
        asyncio.run(client.warmup())  # must not raise
    g.assert_awaited_once()


def test_artifact_client_abc_warmup_is_noop():
    # A minimal ABC subclass that only defines submit inherits a no-op warmup().
    from hls.artifact.artifact_client import ArtifactClient

    class _Min(ArtifactClient):
        async def submit(self, env, *, attempt=None):  # pragma: no cover - never called
            ...

    assert asyncio.run(_Min().warmup()) is None


def test_streaming_mid_stream_transport_break_raises_transport_error():
    # An httpx.ReadError mid-stream must surface as ArtifactTransportError,
    # not escape raw. This pins the transport-break classification for the streaming path.
    client = HTTPArtifactClient(
        contract=_contract(transport="http-sse", protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")

    async def dropping_sse(_body):
        yield {"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]}
        raise httpx.ReadError("connection dropped")

    with patch.object(client, "_post_sse", new=dropping_sse):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        with pytest.raises(ArtifactTransportError):
            asyncio.run(client.submit(env))


# ---------------------------------------------------------------------------
# RequestAttempt: what the transport records about a request that fails
# ---------------------------------------------------------------------------


def test_attempt_records_the_status_of_a_served_rejection():
    """A status code is what separates a server that answered "no" from a connection
    that never delivered an answer, and the driver classifies the failure on it."""
    from hls.core.trace import RequestAttempt

    client = HTTPArtifactClient(contract=_contract(), base_url="http://localhost:9000")
    exc = httpx.HTTPStatusError(
        "503", request=_http_req(), response=httpx.Response(503, request=_http_req()))
    attempt = RequestAttempt()
    with patch.object(client, "_post", new=AsyncMock(side_effect=exc)):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        with pytest.raises(ArtifactTransportError):
            asyncio.run(client.submit(env, attempt=attempt))

    assert attempt.status_code == 503
    assert attempt.n_frames == 0            # it never streamed anything
    assert attempt.partial is None


def test_attempt_records_the_frames_a_stream_sent_before_it_broke():
    from hls.core.trace import RequestAttempt

    client = HTTPArtifactClient(
        contract=_contract(transport="http-sse", protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")

    async def dropping_sse(_body):
        yield {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906],
                            "finish_reason": None}]}
        raise httpx.ReadError("connection dropped")

    attempt = RequestAttempt()
    with patch.object(client, "_post_sse", new=dropping_sse):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        with pytest.raises(ArtifactTransportError):
            asyncio.run(client.submit(env, attempt=attempt))

    assert attempt.n_frames == 1
    assert attempt.t_first_frame is not None
    assert attempt.status_code is None       # the 200 was already committed
    assert attempt.partial.token_ids == [9906]
    assert attempt.partial.text == "Hi"


def test_attempt_survives_the_caller_cancelling_the_submit():
    """The case with no other record: on a drive-deadline miss the submit is cancelled,
    so every frame it collected dies with its own coroutine. The attempt is the caller's
    reference into that frame, which is why it is a parameter and not a return value."""
    from hls.core.trace import RequestAttempt

    client = HTTPArtifactClient(
        contract=_contract(transport="http-sse", protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")

    async def stalling_sse(_body):
        yield {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906],
                            "finish_reason": None}]}
        await asyncio.sleep(3600)

    attempt = RequestAttempt()
    env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])

    async def _drive_and_give_up():
        with patch.object(client, "_post_sse", new=stalling_sse):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(client.submit(env, attempt=attempt), timeout=0.1)

    asyncio.run(_drive_and_give_up())

    assert attempt.n_frames == 1
    assert attempt.partial.token_ids == [9906]


def test_submit_without_an_attempt_still_succeeds():
    """``attempt`` is the caller's optional recorder, not a dependency of sending: an
    in-process caller that cannot observe frames passes nothing."""
    client = HTTPArtifactClient(
        contract=_contract(protocol=OpenAIChatProtocol()),
        base_url="http://localhost:9000")
    body = {"choices": [{"message": {"role": "assistant", "content": "B"},
                         "finish_reason": "stop", "token_ids": [42]}]}
    with patch.object(client, "_post", new=AsyncMock(return_value=body)):
        env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])
        resp = asyncio.run(client.submit(env))
    assert resp.output.first_of("text").text == "B"


def test_artifact_client_disables_trust_env():
    """The client to a served artifact ignores any proxy the environment names."""
    c = HTTPArtifactClient(_contract(protocol=OpenAIChatProtocol()), "http://node:8000")
    assert c._client.trust_env is False


# ---------------------------------------------------------------------------
# The mid-stream stall (``AgentSpec.stall_timeout_sec``)
# ---------------------------------------------------------------------------
#
# A stream that delivered tokens and then stopped delivering them is a failed request. The
# silence before a first token is not: that is prefill plus queueing, which the latency ratio
# already prices, so the rule arms on the first token and never before. These run on a
# virtual clock, so a budget of seconds costs no real time and load cannot stretch a gap.


def _stall_client(stall_timeout_sec=2.0):
    return HTTPArtifactClient(contract=_contract(transport="http-sse", protocol=OpenAIChatProtocol()),
                              base_url="http://localhost:9000", stall_timeout_sec=stall_timeout_sec)


def _token_frame(tid, finish=None):
    return {"choices": [{"delta": {"content": "x"}, "token_ids": [tid], "finish_reason": finish}]}


def _on_virtual_clock(coro_fn):
    from tests.support.virtual_clock import virtual_clock

    with virtual_clock() as runner:
        return runner.run(coro_fn())


def _run_stream(client, frames, attempt=None):
    """Submit against the SSE generator function *frames*, on a virtual clock."""
    env = Envelope(request_id="r1", contents=[TextPayload(text="Q?")])

    async def go():
        with patch.object(client, "_post_sse", new=frames):
            return await client.submit(env, attempt=attempt)

    return _on_virtual_clock(go)


async def _one_token_then_nothing(_body):
    yield _token_frame(1)
    await asyncio.sleep(3600)


def test_a_stream_that_stops_after_a_token_fails_as_a_stall_of_its_own_class():
    """Not a transport error: the connection never broke, and reporting it as one would send
    the reader looking for a dropped socket that is still open."""
    from hls.core.faults import ArtifactStallError

    with pytest.raises(ArtifactStallError, match="no token for 2s after the stream had delivered 1") as caught:
        _run_stream(_stall_client(), _one_token_then_nothing)
    assert not isinstance(caught.value, ArtifactTransportError)
    assert caught.value.category == "stall"


def test_the_stall_fires_at_the_budget_after_the_last_token():
    from hls.core.faults import ArtifactStallError

    seen: dict = {}

    async def go():
        client = _stall_client(stall_timeout_sec=2.0)
        loop = asyncio.get_running_loop()
        with patch.object(client, "_post_sse", new=_one_token_then_nothing):
            with pytest.raises(ArtifactStallError):
                await client.submit(Envelope(request_id="r1", contents=[TextPayload(text="Q?")]))
        seen["t"] = loop.time()

    _on_virtual_clock(go)
    assert seen["t"] == pytest.approx(2.0)


def test_a_long_silence_before_the_first_token_is_not_a_stall():
    """Prefill on a long prompt under load takes as long as it takes, and the ratio prices it."""
    async def slow_prefill(_body):
        await asyncio.sleep(60.0)
        yield _token_frame(1)
        yield _token_frame(2, finish="stop")

    resp = _run_stream(_stall_client(), slow_prefill)
    assert resp.output.first_of("text").tokens == [1, 2]


def test_gaps_just_under_the_budget_complete():
    async def steady(_body):
        for tid in (1, 2, 3):
            await asyncio.sleep(1.9)
            yield _token_frame(tid)
        yield _token_frame(4, finish="stop")

    resp = _run_stream(_stall_client(), steady)
    assert resp.output.first_of("text").tokens == [1, 2, 3, 4]


def test_a_stall_deep_into_a_long_reply_still_fails():
    """The rule is about the gap, not about how much was served before it."""
    from hls.core.faults import ArtifactStallError

    async def many_then_nothing(_body):
        for tid in range(1000):
            yield _token_frame(tid)
        await asyncio.sleep(3600)

    with pytest.raises(ArtifactStallError, match="delivered 1000"):
        _run_stream(_stall_client(), many_then_nothing)


def test_frames_carrying_no_tokens_do_not_re_arm_the_stall():
    """A server keeping the connection warm through a stuck decode is stalled: the budget runs
    from the last token, not the last frame. The heartbeats never stop, so a budget that
    re-armed on any frame would never end the request; the outer bound fails the test then."""
    from hls.core.faults import ArtifactStallError

    async def heartbeats(_body):
        yield _token_frame(1)
        while True:
            await asyncio.sleep(0.5)
            yield {"choices": []}

    async def go():
        client = _stall_client()
        with patch.object(client, "_post_sse", new=heartbeats):
            with pytest.raises(ArtifactStallError):
                await asyncio.wait_for(client.submit(Envelope(request_id="r1", contents=[TextPayload(text="Q?")])),
                                       timeout=60.0)

    _on_virtual_clock(go)


def test_no_stall_budget_leaves_a_silent_stream_to_the_callers_deadline():
    """A task that declares none is served as before: nothing here ends the request."""
    async def go():
        client = _stall_client(stall_timeout_sec=None)
        with patch.object(client, "_post_sse", new=_one_token_then_nothing):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(client.submit(Envelope(request_id="r1", contents=[TextPayload(text="Q?")])),
                                       timeout=600.0)

    _on_virtual_clock(go)


def test_a_stalled_request_keeps_what_it_streamed():
    """The record a cancelled submit leaves: the fault is the verdict, the partial the evidence
    for how far the server got."""
    from hls.core.faults import ArtifactStallError
    from hls.core.trace import RequestAttempt

    async def two_then_nothing(_body):
        yield _token_frame(7)
        yield _token_frame(8)
        await asyncio.sleep(3600)

    attempt = RequestAttempt()
    with pytest.raises(ArtifactStallError):
        _run_stream(_stall_client(), two_then_nothing, attempt=attempt)
    assert attempt.n_frames == 2
    assert attempt.partial.token_ids == [7, 8]
    assert len(attempt.partial.packet_timestamps) == 2


def test_a_stalled_stream_is_closed():
    """The stall ends the request, so the stream it read from is closed rather than left open
    on a server that may still be holding it."""
    from hls.core.faults import ArtifactStallError

    closed: list[bool] = []

    async def one_token_then_nothing(_body):
        try:
            yield _token_frame(1)
            await asyncio.sleep(3600)
        finally:
            closed.append(True)

    with pytest.raises(ArtifactStallError):
        _run_stream(_stall_client(), one_token_then_nothing)
    assert closed == [True]


def test_the_make_function_threads_the_stall_budget():
    from hls.artifact.artifact_client import make_artifact_client

    assert make_artifact_client(_contract(), "http://x", stall_timeout_sec=120.0).stall_timeout_sec == 120.0
    assert make_artifact_client(_contract(), "http://x").stall_timeout_sec is None


# ---------------------------------------------------------------------------
# A connection the peer resets
# ---------------------------------------------------------------------------
#
# The drive role runs the driver on asyncio's own event loop (``asyncio.run``). There a
# connection the peer resets, whether as it opens or while it sits idle in the pool, reaches
# the client as an httpx error, so it fails one request as a transport fault and never the
# drive. These pin that against a real socket.


_SSE_FRAMES = [
    {"choices": [{"delta": {"content": "Hi"}, "token_ids": [9906], "finish_reason": "stop"}]},
    {"choices": [], "usage": {"completion_tokens": 1}},
]


async def _serve(reader, writer):
    """A keep-alive HTTP/1.1 server: 200 on GET, one short SSE reply on POST."""
    import json

    try:
        while True:
            head = await reader.readuntil(b"\r\n\r\n")
            length = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    length = int(line.split(b":", 1)[1])
            await reader.readexactly(length)
            if head.startswith(b"GET"):
                writer.write(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\n\r\nok")
            else:
                writer.write(b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\n"
                             b"transfer-encoding: chunked\r\n\r\n")
                for frame in [*(f"data: {json.dumps(f)}\n\n" for f in _SSE_FRAMES), "data: [DONE]\n\n"]:
                    data = frame.encode()
                    writer.write(b"%x\r\n%s\r\n" % (len(data), data))
                writer.write(b"0\r\n\r\n")
            await writer.drain()
    except (asyncio.IncompleteReadError, ConnectionError):
        writer.close()


class _RecordingBackend:
    """The pool's real network backend, keeping every stream it opens so a test can reset one;
    ``reset_on_connect`` resets each connection as soon as it is made."""

    def __init__(self, reset_on_connect: bool = False):
        import httpcore

        self._inner = httpcore.AnyIOBackend()
        self.reset_on_connect = reset_on_connect
        self.streams: list = []

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        stream = await self._inner.connect_tcp(host, port, timeout=timeout, local_address=local_address,
                                               socket_options=socket_options)
        self.streams.append(stream)
        if self.reset_on_connect:
            await _reset(stream)
        return stream

    async def sleep(self, seconds):
        await self._inner.sleep(seconds)


async def _reset(stream) -> None:
    """Abort the transport under an httpcore stream and let the loop deliver it."""
    stream._stream._transport.abort()
    await asyncio.sleep(0)


def _pool_backend(monkeypatch, backend):
    """Make every connection pool built after this use *backend* as its network."""
    import httpcore._async.connection_pool as pool_module

    monkeypatch.setattr(pool_module, "AutoBackend", lambda: backend)


def _sse_client(port):
    return HTTPArtifactClient(contract=_contract(transport="http-sse", protocol=OpenAIChatProtocol()),
                              base_url=f"http://127.0.0.1:{port}")


def test_a_connection_reset_as_it_opens_is_a_transport_fault(monkeypatch):
    _pool_backend(monkeypatch, _RecordingBackend(reset_on_connect=True))

    async def go():
        server = await asyncio.start_server(_serve, "127.0.0.1", 0)
        client = _sse_client(server.sockets[0].getsockname()[1])
        try:
            with pytest.raises(ArtifactTransportError):
                await client.submit(Envelope(request_id="r1", contents=[TextPayload(text="Q?")]))
        finally:
            await client.aclose()
            server.close()

    asyncio.run(go())


def test_a_reset_idle_connection_does_not_break_the_next_request(monkeypatch):
    """The pool drops a pooled connection the peer reset while it sat idle and serves the next
    request on a fresh one: the server is still up, so a failure here would charge a healthy
    artifact for the harness's own client."""
    backend = _RecordingBackend()
    _pool_backend(monkeypatch, backend)

    async def go():
        server = await asyncio.start_server(_serve, "127.0.0.1", 0)
        client = _sse_client(server.sockets[0].getsockname()[1])
        try:
            await client.warmup()
            assert len(backend.streams) == 1
            await _reset(backend.streams[0])
            return await client.submit(Envelope(request_id="r1", contents=[TextPayload(text="Q?")]))
        finally:
            await client.aclose()
            server.close()

    resp = asyncio.run(go())
    assert resp.output.first_of("text").text == "Hi"
    assert len(backend.streams) == 2
