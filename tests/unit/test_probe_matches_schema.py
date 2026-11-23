"""Every protocol's readiness probe must be a valid instance of the body it publishes.

The probe is the second-stage readiness check: after /health returns 200 the substrate
sends probe_request() and treats any response under 500 as ready. A contestant builds its
request model from the brief, which is rendered from body_schema()/encode_request(). So a
probe that omits a field those declare is rejected by a conforming server -- and the gate
reads that rejection as "route not registered yet" and polls until the deploy times out.

AudioTranscriptionProtocol and TTSStreamProtocol build their probe through
encode_request. These tests hold the line for the protocols that hand-write one, and for
any added later.
"""

from __future__ import annotations

import pytest

from hls.core.payloads import Envelope, TextPayload
from hls.spec.wire import (AudioTranscriptionProtocol, OmniChatProtocol, OpenAIChatProtocol,
                           TTSStreamProtocol)

#: One configured instance per protocol. Non-default values where a field is optional, so
#: a probe that drops it is caught rather than matching the default by luck.
PROTOCOLS = [
    pytest.param(AudioTranscriptionProtocol(model="m", language="en"), id="transcription"),
    pytest.param(TTSStreamProtocol(model="m", voice="Ryan", return_token_ids=True), id="tts"),
    pytest.param(OpenAIChatProtocol(model="m"), id="chat"),
    pytest.param(OmniChatProtocol(model="m"), id="omni"),
]


def _encoded(proto) -> dict:
    """What the protocol sends for a real request, as the contestant's model sees it."""
    env = Envelope(request_id="r", contents=[TextPayload(text="hello")],
                   metadata={"max_tokens": 8})
    return proto.encode_request(env)


@pytest.mark.parametrize("proto", PROTOCOLS)
def test_probe_carries_every_field_the_encoder_sends(proto):
    """A field the encoder always sends is one a strict request model may mark required,
    so the probe has to carry it too."""
    missing = set(_encoded(proto)) - set(proto.probe_request())
    # max_tokens is per-request and optional in every schema; the probe sets its own.
    missing.discard("max_tokens")
    assert not missing, (
        f"{type(proto).__name__}.probe_request() omits {sorted(missing)}, which "
        f"encode_request() always sends. A server that validates the documented body "
        f"rejects the probe, and the readiness gate reads that as an unregistered route.")


@pytest.mark.parametrize("proto", PROTOCOLS)
def test_probe_names_the_served_model(proto):
    """Not a placeholder. Stock vLLM 404s a request whose `model` does not match
    --served-model-name, and the gate treats 404 as not-yet-ready, so a placeholder makes
    a healthy server look permanently unready."""
    assert proto.probe_request().get("model") == proto.model


@pytest.mark.parametrize("proto", PROTOCOLS)
def test_probe_sends_no_field_the_schema_does_not_declare(proto):
    """The reverse drift: a probe-only field a strict model would reject as unexpected."""
    extra = set(proto.probe_request()) - set(_encoded(proto)) - {"max_tokens", "stream"}
    assert not extra, (
        f"{type(proto).__name__}.probe_request() sends {sorted(extra)}, which "
        f"encode_request() never sends and the brief therefore never documents.")
