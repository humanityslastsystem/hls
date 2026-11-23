"""A reply carrying no emitted tokens, on both decode paths.

It is a fault only where the brief published the rule. Both brief templates put "a
response with an empty `token_ids` array is unverifiable and fails the run" inside the
pinned-stop-ids block, together with the sentence that a reply ending on its first
token is one token and not zero. A protocol that pins no stop ids never shows its
contestant either sentence, so a zero-token reply decodes there instead of failing the
request.
"""

import pytest

from hls.core.faults import ArtifactProtocolError
from hls.spec.wire import OpenAIChatProtocol, OpenAICompletionProtocol

PROTOCOLS = [OpenAIChatProtocol, OpenAICompletionProtocol]


def _unary(protocol_cls, token_ids):
    choice = {"token_ids": token_ids, "finish_reason": "stop"}
    if protocol_cls is OpenAIChatProtocol:
        choice["message"] = {"content": "hi"}
    else:
        choice["text"] = "hi"
    return {"choices": [choice]}


def _stream(protocol_cls, token_ids):
    """One terminal frame carrying no content, with or without token ids."""
    choice: dict = {"finish_reason": "stop"}
    choice["delta" if protocol_cls is OpenAIChatProtocol else "text"] = (
        {} if protocol_cls is OpenAIChatProtocol else "")
    if token_ids:
        choice["token_ids"] = token_ids
    return [(0.0, {"choices": [choice]})]


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
def test_empty_token_ids_faults_when_the_contract_pins_stop_ids(protocol_cls):
    p = protocol_cls(stop_token_ids=(0,))
    with pytest.raises(ArtifactProtocolError, match="empty choices\\[0\\].token_ids"):
        p.decode_response(_unary(protocol_cls, []), "r1")
    with pytest.raises(ArtifactProtocolError, match="no token_ids"):
        p.decode_stream(_stream(protocol_cls, []), "r1")


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
def test_a_zero_token_reply_decodes_when_no_stop_ids_are_pinned(protocol_cls):
    """The rule is unpublished there, so enforcing it would zero a run over a reply
    whose contestant was told only that `token_ids` is mandatory -- the reasoning
    INCOMPLETE_FINISH_REASONS already applies to finish_reason labels."""
    p = protocol_cls()
    env = p.decode_response(_unary(protocol_cls, []), "r1")
    assert env.contents[0].tokens == []
    assert p.decode_stream(_stream(protocol_cls, []), "r1").output.contents[0].tokens is None


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
@pytest.mark.parametrize("stop_token_ids", [None, (0,)])
def test_absent_token_ids_still_reports_as_missing(protocol_cls, stop_token_ids):
    """An absent array is a missing mandatory field, which every brief publishes."""
    body = _unary(protocol_cls, [])
    del body["choices"][0]["token_ids"]
    with pytest.raises(ArtifactProtocolError, match="missing choices\\[0\\].token_ids"):
        protocol_cls(stop_token_ids=stop_token_ids).decode_response(body, "r1")


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
def test_a_single_token_reply_decodes(protocol_cls):
    """A reply whose only token is the end-of-turn id is one token, not zero."""
    env = protocol_cls(stop_token_ids=(0,)).decode_response(_unary(protocol_cls, [0]), "r1")
    assert env.contents[0].tokens == [0]
