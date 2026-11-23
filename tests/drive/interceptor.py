"""The seal's interceptor (``hls.egress.proxy``) for the deploy_drive tests, run in the test's own
process on the loopback, where the seal probe's refusal check dials it.

With no redirect to read a connection's original destination from, it takes the one the probe
dials in a real seal (``deploy_drive.PROBE_DIAL``). It enforces *policy* as the seal's
interceptor does and writes its status to *status_file*, the file of the probe's seal, so the
stage reads what it refused as it reads a real seal's record."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

from hls.egress.policy import NetworkPolicy
from hls.egress.proxy import EgressProxy


class Interceptor:
    """A running interceptor; ``close`` stops it."""

    def __init__(self, status_file: Path, policy: NetworkPolicy) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        proxy = EgressProxy(policy.model_dump(mode="json"), host="127.0.0.1", port=None, transparent_port=0,
                            original_dst=lambda sock: ("192.0.2.1", 80), status_file=str(status_file))
        self._server = asyncio.run_coroutine_threadsafe(proxy.start_transparent(), self._loop).result(10)
        #: The proxy, for what it refused.
        self.proxy = proxy

    @property
    def address(self) -> str:
        """``host:port`` the probe dials."""
        host, port = self._server.sockets[0].getsockname()[:2]
        return f"{host}:{port}"

    def close(self) -> None:
        async def stop() -> None:
            self._server.close()
            await self._server.wait_closed()

        asyncio.run_coroutine_threadsafe(stop(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)
        self._loop.close()
