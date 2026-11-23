"""The load generator retires idle connections before the served stack does.

httpx and uvicorn both default to a 5-second idle timeout, so a pooled connection that
has gone quiet is retired by BOTH ends at the same instant. When the client wins that
race it hands a request to a socket the server has already closed, which surfaces as
``transport: ReadError('')``. These tasks zero a run on any failed request, so one of
those discards an entire drive.

A drive that begins after the pool has sat idle, as after an off-clock warm-up, is exposed
to that race on its first requests.

Retiring first costs nothing: the client opens a fresh connection, which is what it would
have done anyway had it noticed the close.
"""

from __future__ import annotations

import httpx

from hls.artifact.artifact_client import _KEEPALIVE_EXPIRY_SEC

#: uvicorn's ``--timeout-keep-alive`` default, and so the idle timeout of every server
#: these tasks serve behind.
_UVICORN_KEEP_ALIVE_SEC = 5.0


def test_we_retire_idle_connections_before_the_server_does():
    assert _KEEPALIVE_EXPIRY_SEC < _UVICORN_KEEP_ALIVE_SEC, (
        f"the client holds idle connections for {_KEEPALIVE_EXPIRY_SEC}s and the server "
        f"drops them at {_UVICORN_KEEP_ALIVE_SEC}s; the client has to let go first")


def test_the_margin_is_not_a_hair():
    """Half the server's timeout or better. A margin of a few hundred milliseconds would
    still lose the race whenever a drive's first request lands inside it."""
    assert _KEEPALIVE_EXPIRY_SEC <= _UVICORN_KEEP_ALIVE_SEC / 2


def test_httpx_default_would_tie_with_the_server():
    """Why this needs setting at all: left alone, httpx picks exactly the server's
    number. If httpx ever changes its default this test is the thing that notices."""
    assert httpx.Limits().keepalive_expiry == _UVICORN_KEEP_ALIVE_SEC


def test_the_client_actually_applies_it():
    """A constant nothing reads would be worse than no constant, so this reaches into
    the live connection pool rather than trusting the call site."""
    from hls.artifact.artifact_client import HTTPArtifactClient
    from hls.spec.contracts import RequestResponseContract
    from hls.spec.wire import OpenAIChatProtocol

    contract = RequestResponseContract(
        path="/v1/chat/completions", input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}), transport="http",
        protocol=OpenAIChatProtocol(model="m"))
    client = HTTPArtifactClient(contract=contract, base_url="http://127.0.0.1:9999")
    assert client._client._transport._pool._keepalive_expiry == _KEEPALIVE_EXPIRY_SEC
