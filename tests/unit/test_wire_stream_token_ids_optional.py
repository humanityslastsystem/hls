"""A protocol that does not require token ids must not require them when streaming.

``emits_token_ids`` is the marker that says whether a contract's responses carry integer
token ids. When it is off, ``_flags`` stops sending ``return_token_ids`` and the brief
stops documenting the field, so the server is never asked for it. Both decodes honour
that: a streaming decode that mandated ids on every content-bearing delta would fault
every reply to a request that asked for none.

The marker lives on the shared OpenAI protocol, so any contract may set it, on either
transport.
"""

from __future__ import annotations

import pytest

from hls.core.faults import ArtifactProtocolError
from hls.spec.wire import OmniChatProtocol, OpenAIChatProtocol


def _delta(text, *, token_ids=None, finish_reason=None):
    """One SSE chunk carrying a content delta, with or without its token ids."""
    choice: dict = {"delta": {"content": text}, "finish_reason": finish_reason}
    if token_ids is not None:
        choice["token_ids"] = token_ids
    return {"choices": [choice]}


def test_a_stream_without_token_ids_decodes_when_the_protocol_asks_for_none():
    """The whole point of the marker: an omni server streams text and no ids, and that is
    a complete reply, not a protocol fault."""
    chunks = [(0.5, _delta("Hel")), (0.7, _delta("lo")),
              (0.9, _delta("", finish_reason="stop"))]

    resp = OmniChatProtocol().decode_stream(chunks, "r1")

    payload = resp.output.contents[0]
    assert payload.text == "Hello"
    assert payload.tokens is None
    assert payload.finish_reason == "stop"


def test_a_stream_without_token_ids_still_reports_when_its_packets_arrived():
    """Without ids to count, each content delta is the packet. Otherwise the reply has no
    first-packet time and no packet timestamps, and a TTFT/TPOT SLO has nothing to read --
    every request would be unscoreable rather than merely id-less."""
    chunks = [(0.5, _delta("Hel")), (0.7, _delta("lo")),
              (0.9, _delta("", finish_reason="stop"))]

    resp = OmniChatProtocol().decode_stream(chunks, "r1")

    assert resp.t_first_packet == 0.5
    assert resp.packet_timestamps == [0.5, 0.7]
    assert resp.packet_timestamps[0] == resp.t_first_packet


def test_an_empty_stream_is_not_refused_for_carrying_no_token_ids():
    """The zero-token refusal is about a server that stripped the ids it was asked for.
    A protocol that asked for none cannot have stripped anything, so pinned stop ids do
    not make its empty reply a fault."""
    proto = OmniChatProtocol(stop_token_ids=(151645,))
    chunks = [(0.5, _delta("", finish_reason="stop"))]

    resp = proto.decode_stream(chunks, "r1")

    assert resp.output.contents[0].text == ""


def test_a_protocol_that_does_require_ids_still_faults_without_them():
    """The check is narrowed by the marker, not removed. A chat contract asks for ids on
    every request, so a content delta that drops them still leaves tokens misaligned with
    text and still fails the request."""
    chunks = [(0.5, _delta("Hello")), (0.9, _delta("", finish_reason="stop"))]

    with pytest.raises(ArtifactProtocolError, match="without token_ids"):
        OpenAIChatProtocol().decode_stream(chunks, "r1")


def test_a_protocol_that_requires_ids_keys_its_packets_off_them():
    """Unchanged where ids are required: a leading ids-only chunk is a real packet
    arrival, and one timestamp is recorded per token rather than per delta."""
    chunks = [(0.4, {"choices": [{"delta": {}, "token_ids": [1]}]}),
              (0.6, _delta("Hi", token_ids=[2, 3])),
              (0.9, _delta("", finish_reason="stop"))]

    resp = OpenAIChatProtocol().decode_stream(chunks, "r1")

    assert resp.t_first_packet == 0.4
    assert resp.packet_timestamps == [0.4, 0.6, 0.6]
    assert resp.output.contents[0].tokens == [1, 2, 3]
