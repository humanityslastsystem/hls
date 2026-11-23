"""How the harness reads a reply from a served system.

Every harness client of a server the contestant built reads its replies this way: the readiness
gates (``hls.drive.readiness``), which run in the runner, and the drive role's client
(``hls.drive.replies``).

- Each request asks for an uncompressed reply: ``Accept-Encoding: identity`` (``IDENTITY``).
- A reply in any other content coding is refused at its headers, before any of its body is read
  (``refuse_coding``, which raises ``CompressedReply``). A body of a few KiB can expand to any
  size, as a zstd frame of zeros does, and a content decoder expands each piece whole before its
  reader can count it, so no coded body is ever handed to one.
- A body is read as the bytes that came over the connection, up to the caller's bound
  (``read_raw``): the readiness probes read at most their first bytes, and the drive counts every
  byte against its budget as it reads them.

``CompressedReply`` is a ``RefusedReply``: an ``ArtifactProtocolError``, which the drive records as
the request's failure, the contestant's, and an ``httpx.TransportError``, which a readiness probe
reads as a server not ready yet."""

from __future__ import annotations

import httpx

from hls.core.faults import ArtifactProtocolError

#: The request header every harness client of a served system sends, so its replies come
#: uncompressed.
IDENTITY = {"Accept-Encoding": "identity"}
#: What a compressed reply's refusal says.
COMPRESSED = "a compressed reply, not asked for"


class RefusedReply(ArtifactProtocolError, httpx.TransportError):
    """A reply from a served system the harness does not read."""


class CompressedReply(RefusedReply):
    """A reply in a content coding other than identity, which no harness client asks for."""


def refuse_coding(response: httpx.Response) -> None:
    """Raise ``CompressedReply`` for a reply in any content coding other than identity."""
    codings = [c.strip().lower() for c in response.headers.get_list("content-encoding", split_commas=True)]
    if any(c not in ("", "identity") for c in codings):
        raise CompressedReply(f"{COMPRESSED}: content-encoding {', '.join(codings)}")


async def read_raw(response: httpx.Response, max_bytes: int) -> bytes:
    """The first *max_bytes* of streamed *response*'s body, as they came over the connection and
    never through a content decoder; the rest is never read. The caller has refused a coded reply
    at its headers first (``refuse_coding``)."""
    body = bytearray()
    async for chunk in response.aiter_raw():
        body += chunk[:max_bytes - len(body)]
        if len(body) >= max_bytes:
            break
    return bytes(body)
