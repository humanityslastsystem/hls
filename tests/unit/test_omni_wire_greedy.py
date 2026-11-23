"""omni's wire protocol must pin greedy decode, because its brief already promises it.

The rendered omni brief says "Decode greedily (argmax, no sampling)" -- it inherits the
base brief's decoding block -- and the oracle decodes greedily. A request body of just
{model, messages, max_tokens} leaves a server to fill the rest from the checkpoint's
`generation_config`, and a `generation_config` repetition_penalty of 1.05 applies to
greedy decode on all three serving stacks.

These pin that the request carries the decode the brief promises.
"""

from __future__ import annotations

from hls.core.payloads import Envelope, TextPayload
from hls.spec.wire import OmniChatProtocol

#: Every knob that has to be neutral for a decode to be greedy.
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": -1,
          "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0}


def _body():
    proto = OmniChatProtocol(model="m")
    env = Envelope(request_id="r0", contents=[TextPayload(text="what is this?")],
                   metadata={"max_tokens": 32})
    return proto.encode_request(env)


def test_the_request_pins_greedy_decode():
    body = _body()

    for key, want in GREEDY.items():
        assert key in body, f"the omni request does not pin {key!r}, so the stack decides"
        assert body[key] == want, f"{key} is {body[key]!r}, expected {want!r}"


def test_it_matches_the_reference_engines_greedy_pin():
    """Two spellings of 'greedy' that drift apart is worse than one that is wrong."""
    from hls.reference.engines import _GREEDY_SAMPLING

    body = _body()
    for key, want in _GREEDY_SAMPLING.items():
        assert body.get(key) == want, (
            f"wire pins {key}={body.get(key)!r} but the reference engines pin {want!r}")


def test_the_rest_of_the_request_is_unchanged():
    body = _body()

    assert body["model"] == "m"
    assert body["messages"][0]["role"] == "user"
    assert body["max_tokens"] == 32


def test_streaming_requests_pin_it_too():
    proto = OmniChatProtocol(model="m")
    env = Envelope(request_id="r0", contents=[TextPayload(text="x")],
                   metadata={"max_tokens": 8})

    body = proto.encode_request(env, stream=True)

    assert body["temperature"] == 0.0 and body["top_p"] == 1.0
