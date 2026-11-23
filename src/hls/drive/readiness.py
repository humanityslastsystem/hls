"""The two readiness gates a served system passes before it is driven.

1. ``wait_healthy``: ``GET <base_url>/health`` once a second, with a 5 s client timeout,
   until it answers 200.
2. ``wait_inference_ready``: ``POST`` the contract's own probe body to the contract's path
   every 2 s, with a 10 s client timeout, until it answers below 400 or exactly 400. A 404,
   405, 422 or any other 4xx means the route or model is not registered yet (vLLM answers
   ``/health`` 200 for a while before its completions route exists), and a 5xx or a
   transport error means the server is not up yet; both keep waiting.

A stage passes ``alive``, a callable that says whether the server's role still runs, and a gate
stops at once with ``NotServing`` when it does not, instead of polling a dead server out to the
full timeout. It is checked before every probe.

What a server answers cannot hold a gate or the process running it, the runner: each probe is
bounded as a whole by its client timeout, asks for an uncompressed reply, refuses one in any
content coding at its headers and reads no more than the first ``PROBE_BODY_BYTES`` of the body as
they came over the connection (``hls.drive.reading``), and any httpx error (that refusal among
them) means not ready yet. So a gate ends with ``NotReady`` by its deadline plus one client timeout
and one poll interval, whatever the reply. A probe started before the deadline counts if its
answer comes within the client timeout, even after the deadline, and one that ends in time and
reads cleanly is judged by the rules above on its status.

The probes use ``trust_env=False``: they target a node-internal address, and an ambient
HTTP proxy would answer for it with a 502."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx

from hls.drive.reading import IDENTITY, read_raw, refuse_coding

if TYPE_CHECKING:
    from hls.spec.contracts import ArtifactContract

#: The path the health gate probes.
HEALTH_PATH = "/health"
#: Seconds between health probes.
HEALTH_POLL_SEC = 1.0
#: Each health probe's client timeout, in seconds.
HEALTH_PROBE_TIMEOUT_SEC = 5.0
#: Seconds between inference probes.
INFERENCE_POLL_SEC = 2.0
#: Each inference probe's client timeout, in seconds.
INFERENCE_PROBE_TIMEOUT_SEC = 10.0
#: Bytes of a reply's body a probe reads at most; the rest is never read.
PROBE_BODY_BYTES = 64 * 1024


class ReadinessError(Exception):
    """A served system did not become ready."""


class NotReady(ReadinessError):
    """The deadline passed before the gate opened."""


class NotServing(ReadinessError):
    """The server's role ended while the gate waited."""


async def _probe(cli: httpx.AsyncClient, method: str, url: str, *, within_sec: float,
                 payload: object = None) -> tuple[int, bytes, str] | None:
    """One probe: the reply's status, the first ``PROBE_BODY_BYTES`` of its body, and the text
    encoding of the body. None when the server is not up yet: any httpx error, a reply in a
    content coding among them, or no whole reply within *within_sec*."""
    try:
        async with asyncio.timeout(within_sec):
            async with cli.stream(method, url, json=payload, headers=IDENTITY) as resp:
                refuse_coding(resp)
                body = await read_raw(resp, PROBE_BODY_BYTES)
                return resp.status_code, body, resp.encoding or "utf-8"
    except (httpx.HTTPError, TimeoutError):
        return None


def _text(raw: bytes, encoding: str) -> str:
    """*raw* as text in the reply's *encoding*, or as UTF-8 when the server named a codec that
    does not decode bytes to text, which a server may name for any reply."""
    try:
        return raw.decode(encoding, errors="replace")
    except (LookupError, ValueError):
        return raw.decode("utf-8", errors="replace")


def _check_alive(alive: Callable[[], bool] | None, url: str) -> None:
    if alive is not None and not alive():
        raise NotServing(f"the server at {url} exited before it was ready")


async def wait_healthy(base_url: str, *, timeout_sec: float, alive: Callable[[], bool] | None = None) -> None:
    """Poll ``base_url + HEALTH_PATH`` until it answers 200. Raise ``NotReady`` at the deadline,
    carrying the start of the last body seen, and ``NotServing`` as soon as ``alive()`` is
    False."""
    url = base_url + HEALTH_PATH
    deadline = time.monotonic() + timeout_sec
    last_body = ""
    async with httpx.AsyncClient(timeout=httpx.Timeout(HEALTH_PROBE_TIMEOUT_SEC), trust_env=False) as cli:
        while time.monotonic() < deadline:
            _check_alive(alive, url)
            reply = await _probe(cli, "GET", url, within_sec=HEALTH_PROBE_TIMEOUT_SEC)
            if reply is not None:
                status, raw, encoding = reply
                if status == 200:
                    return
                body = _text(raw, encoding).strip()
                if body:
                    last_body = body
            await asyncio.sleep(HEALTH_POLL_SEC)
    raise NotReady(f"service not healthy at {url} in {timeout_sec}s"
                   + (f"; last response body: {last_body!r}" if last_body else ""))


async def wait_inference_ready(base_url: str, contract: ArtifactContract, *, timeout_sec: float,
                               alive: Callable[[], bool] | None = None) -> None:
    """Probe the contract's inference path with its own probe body until the server serves
    it: any status below 400, or a bare 400 (the route is registered and parsed the body).
    Raise ``NotReady`` at the deadline and ``NotServing`` as soon as ``alive()`` is False."""
    url = base_url + contract.path
    probe_body = contract.protocol.probe_request()
    deadline = time.monotonic() + timeout_sec
    async with httpx.AsyncClient(timeout=httpx.Timeout(INFERENCE_PROBE_TIMEOUT_SEC), trust_env=False) as cli:
        while time.monotonic() < deadline:
            _check_alive(alive, url)
            reply = await _probe(cli, "POST", url, within_sec=INFERENCE_PROBE_TIMEOUT_SEC, payload=probe_body)
            if reply is not None and reply[0] <= 400:
                return
            await asyncio.sleep(INFERENCE_POLL_SEC)
    raise NotReady(f"inference endpoint not ready at {url} in {timeout_sec}s")
