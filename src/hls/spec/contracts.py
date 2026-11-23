"""L1 -- ArtifactContract subtypes.

The contract is the trust boundary: what the implementer's HTTP server must expose.
It declares the path + modalities and *holds* a WireProtocol that owns the wire
codec and the build-brief contributions. RequestResponseContract is the one implemented;
DuplexSessionContract is a stub that raises on instantiation.
"""

from __future__ import annotations
from typing import Literal
from pydantic import Field, field_serializer

from hls.core.poly import Poly
from hls.spec.wire import OpenAIChatProtocol, WireProtocol


class ArtifactContract(Poly):
    """The trust boundary between the harness and the implementer's server (L1).

    Declares the HTTP `path` and the accepted input/output `modalities`. The harness
    renders the contract into the build brief. Subclass per interaction shape."""

    path: str
    input_modalities: frozenset[str]
    output_modalities: frozenset[str]
    port: int = 8000

    @field_serializer("input_modalities", "output_modalities")
    def _ser_modalities(self, v: frozenset[str]) -> list[str]:
        # frozenset has no stable iteration order across processes (PYTHONHASHSEED),
        # which would make the serialized provenance / any content digest of this
        # contract non-deterministic. Emit a sorted list; pydantic coerces it back
        # to frozenset on load, so the round-trip is preserved.
        return sorted(v)

    def body_schema(self) -> dict:  # pragma: no cover - default
        return {}

    def prompt_fields(self) -> dict:  # pragma: no cover - overridden by leaves
        return {
            "path": self.path,
            "input_modalities": sorted(self.input_modalities),
            "output_modalities": sorted(self.output_modalities),
            "port": self.port,
        }


class RequestResponseContract(ArtifactContract):
    """A server that answers each request with one reply, whole or streamed."""

    #: "http" for a whole reply, "http-sse" for one streamed as server-sent events.
    transport: Literal["http", "http-sse"] = "http"
    #: The wire protocol of the requests and replies, which owns their codec and the brief's
    #: description of them.
    protocol: WireProtocol = Field(default_factory=OpenAIChatProtocol)

    def body_schema(self) -> dict:
        return self.protocol.body_schema()

    def brief_section(self) -> str:
        return self.protocol.brief_section(path=self.path,
                                           streaming=self.transport == "http-sse")

    def prompt_fields(self) -> dict:
        return {**super().prompt_fields(), "transport": self.transport}


# A stub, importable but not wired.
class DuplexSessionContract(ArtifactContract):
    def __init__(self, **_):  # pragma: no cover
        raise NotImplementedError("DuplexSessionContract is not implemented")
