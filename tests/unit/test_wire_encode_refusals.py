"""What the request encoders refuse, and as whose fault.

Encoding reads an envelope this harness built, so a request it cannot encode is our
fault or the task's: a HarnessFault, which invalidates the run, and never an
ArtifactFault, which would record a failed request against a contestant that has not
been sent anything yet.
"""

import pytest

from hls.core.faults import ArtifactFault, HarnessFault
from hls.core.payloads import Envelope, TextPayload
from hls.spec.wire import OpenAIChatProtocol, OpenAICompletionProtocol

PROTOCOLS = [OpenAIChatProtocol, OpenAICompletionProtocol]


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
def test_an_envelope_with_no_conversation_is_our_fault(protocol_cls):
    """No message list and no text payload leaves nothing to send. messages_of raises
    TypeError on it; a bare TypeError says nothing about whose fault it is."""
    env = Envelope(request_id="r1", contents=[], metadata={})
    with pytest.raises(HarnessFault, match="no text payload"):
        protocol_cls().encode_request(env)
    with pytest.raises(HarnessFault):
        protocol_cls().encode_request(env, stream=True)


@pytest.mark.parametrize("protocol_cls", PROTOCOLS)
def test_that_refusal_is_not_charged_to_the_artifact(protocol_cls):
    env = Envelope(request_id="r1", contents=[], metadata={})
    with pytest.raises(BaseException) as e:
        protocol_cls().encode_request(env)
    assert not isinstance(e.value, ArtifactFault)


def test_raw_conditioning_refuses_a_conversation_it_cannot_send():
    """A raw prompt is the model input verbatim, so only the last message rides on the
    wire. A tenant whose turns are not pre-rendered stitches its history into the same
    field, and it would be dropped without a word."""
    env = Envelope(
        request_id="r1", contents=[TextPayload(text="u2")],
        metadata={"messages": [{"role": "user", "content": "u1"},
                               {"role": "assistant", "content": "a1"},
                               {"role": "user", "content": "u2"}]})
    with pytest.raises(HarnessFault, match="raw conditioning"):
        OpenAICompletionProtocol().encode_request(env)
    # The same conversation is ordinary traffic for a chat protocol, which sends it whole.
    assert len(OpenAIChatProtocol().encode_request(env)["messages"]) == 3


def test_raw_conditioning_sends_a_single_message_conversation():
    """What every tab_completion request is: one message, pre-rendered by the session
    dataset or stitched as a single-turn tenant's only turn."""
    env = Envelope(request_id="r1", contents=[TextPayload(text="fallback")],
                   metadata={"messages": [{"role": "user", "content": "STEP"}]})
    assert OpenAICompletionProtocol().encode_request(env)["prompt"] == "STEP"
