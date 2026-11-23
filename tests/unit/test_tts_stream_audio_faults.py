"""Bad audio from a TTS contestant fails its own request, not the whole drive.

The assembly after the per-chunk loop in ``TTSStreamProtocol.decode_stream`` is where the
frames are actually read: base64 decoded, WAV headers parsed, sample rates compared. Every
failure there raises its own exception type -- binascii.Error, wave.Error, ValueError --
none of which is an ArtifactProtocolError, so unguarded, a single malformed reply would
escape the fault taxonomy as an unclassified harness error. A stream that switches codec
rate partway through fails its own request.
"""

from __future__ import annotations

import base64
import io
import json
import wave

import numpy as np
import pytest

from hls.core.audio import encode_wav_b64
from hls.core.faults import ArtifactProtocolError
from hls.spec.wire import TTSStreamProtocol


def _tone(n: int = 160, *, sample_rate: int = 16_000) -> str:
    """A short well-formed base64 PCM16 WAV frame."""
    return encode_wav_b64(np.zeros(n, dtype=np.float32), sample_rate)


def _truncated_24bit() -> str:
    """A 24-bit WAV whose data chunk is one byte short of its own header.

    The widening path reads three bytes per sample, so a byte count that is not a
    multiple of three raises ValueError out of numpy's reshape."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(3)
        w.setframerate(16_000)
        w.writeframes(b"\x01\x02\x03\x04\x05\x06")
    return base64.b64encode(bytes(bytearray(buf.getvalue())[:-1])).decode("ascii")


def _frames(*audio_b64: str) -> list[tuple[float, dict]]:
    return [(0.1 * (i + 1), {"audio": a}) for i, a in enumerate(audio_b64)]


def test_a_clean_stream_still_assembles():
    """The guard must not change what a conforming server gets: two frames concatenate
    into one clip at their own rate."""
    resp = TTSStreamProtocol().decode_stream(_frames(_tone(), _tone()), "r1")

    payload = resp.output.contents[0]
    assert payload.sample_rate == 16_000
    assert payload.duration_sec == pytest.approx(320 / 16_000)
    assert resp.t_first_packet == pytest.approx(0.1)


@pytest.mark.parametrize("label,frames", [
    ("not base64", ["!!!not base64!!!"]),
    ("not a wav", [base64.b64encode(b"hello world").decode("ascii")]),
    ("rate change mid-stream", [_tone(sample_rate=16_000), _tone(sample_rate=24_000)]),
    ("24-bit frame cut short", [_truncated_24bit()]),
])
def test_bad_audio_fails_one_request_as_a_protocol_fault(label, frames):
    """Each of these raises a different exception type out of the audio helpers. All of
    them are the contestant's output, so all of them must arrive as ArtifactProtocolError
    -- the fault the harness charges to one request."""
    with pytest.raises(ArtifactProtocolError):
        TTSStreamProtocol().decode_stream(_frames(*frames), "r1")


@pytest.mark.parametrize("audio", [
    "!!!not base64!!!",
    base64.b64encode(b"hello world").decode("ascii"),
])
def test_bad_audio_in_a_unary_reply_is_a_protocol_fault_too(audio):
    """The unary path reads the same contestant bytes to get the clip's duration, so
    the same malformed reply must fail its request there as well."""
    with pytest.raises(ArtifactProtocolError):
        TTSStreamProtocol().decode_response({"audio": audio}, "r1")


def test_a_sample_rate_past_integer_range_fails_one_request():
    """JSON's ``1e400`` and ``Infinity`` both read as float infinity, which ``int``
    refuses with OverflowError rather than ValueError. Outside the guards, one reply
    stating such a rate would end the whole drive as a harness error."""
    reply = json.loads('{"audio": "%s", "sample_rate": 1e400}' % _tone())
    with pytest.raises(ArtifactProtocolError):
        TTSStreamProtocol().decode_stream([(0.1, reply)], "r1")
    with pytest.raises(ArtifactProtocolError):
        TTSStreamProtocol().decode_response(reply, "r1")
