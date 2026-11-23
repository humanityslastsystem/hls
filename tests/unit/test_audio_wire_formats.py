"""`decode_wav_b64` against WAV headers a contestant might actually send.

The wire contract is mono PCM16, and `encode_wav_b64` always writes that, so the oracle's
audio is never anything else. A contestant's can be. Reading the frame bytes as int16 mono
whatever the header says would decode stereo as interleaved mono at twice its true length,
and the judge would transcribe nonsense.

These pin that every supported width and channel count decodes to the SAME waveform, so
the format a contestant happens to pack in cannot move its correctness score.
"""

from __future__ import annotations

import base64
import io
import logging
import wave

import numpy as np

from hls.core.audio import decode_wav_b64, encode_wav_b64

SR = 16_000


def _tone(seconds: float = 0.25, freq: float = 440.0) -> np.ndarray:
    t = np.arange(int(SR * seconds), dtype=np.float32) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _wav_b64(samples: np.ndarray, *, channels: int, width: int) -> str:
    """Pack a mono float waveform as a WAV with the given channel count and width."""
    if width == 1:                                   # unsigned, centred on 128
        data = np.clip(np.round(samples * 128.0) + 128.0, 0, 255).astype(np.uint8)
    elif width == 2:
        data = (samples * 32767.0).astype(np.int16)
    elif width == 3:
        as32 = (samples * 8388607.0).astype(np.int32)
        data = as32.view(np.uint8).reshape(-1, 4)[:, :3].reshape(-1)  # low 3 bytes, LE
    elif width == 4:
        data = (samples * 2147483647.0).astype(np.int32)
    else:
        raise AssertionError(width)

    if channels > 1 and width != 3:
        data = np.repeat(data, channels)             # same signal in every channel
    elif channels > 1:
        data = np.tile(data.reshape(-1, 3), (1, channels)).reshape(-1)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(SR)
        w.writeframes(data.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_the_contract_format_is_unchanged():
    """Mono PCM16 is what the oracle sends and must decode exactly."""
    tone = _tone()
    arr, sr = decode_wav_b64(encode_wav_b64(tone, SR))

    assert sr == SR
    assert len(arr) == len(tone)
    assert np.allclose(arr, tone, atol=1e-4)


def test_stereo_decodes_to_the_right_LENGTH_not_double():
    """Interleaved stereo read as mono would come back at 2x length, and the judge would
    then hear a clip twice as long as the words in it."""
    tone = _tone()
    arr, _ = decode_wav_b64(_wav_b64(tone, channels=2, width=2))

    assert len(arr) == len(tone), (
        f"stereo decoded to {len(arr)} samples for a {len(tone)}-sample signal; "
        "a 2x length means the channels were read as consecutive samples")
    assert np.allclose(arr, tone, atol=1e-3)


def test_every_supported_width_decodes_to_the_same_waveform():
    """A contestant's choice of sample width must not move its correctness score."""
    tone = _tone()
    for width, tol in ((1, 2e-2), (2, 1e-4), (3, 1e-4), (4, 1e-4)):
        arr, _ = decode_wav_b64(_wav_b64(tone, channels=1, width=width))
        assert len(arr) == len(tone), f"width {width} gave {len(arr)} samples"
        assert np.allclose(arr, tone, atol=tol), f"width {width} decoded wrong"


def test_stereo_at_every_width_also_downmixes():
    tone = _tone()
    for width, tol in ((1, 2e-2), (2, 1e-3), (3, 1e-3), (4, 1e-3)):
        arr, _ = decode_wav_b64(_wav_b64(tone, channels=2, width=width))
        assert len(arr) == len(tone), f"stereo width {width} gave {len(arr)} samples"
        assert np.allclose(arr, tone, atol=tol)


def test_a_contract_departure_is_logged_not_swallowed(caplog):
    """It must not raise -- every caller is inside verification, where an exception takes
    down a tenant's whole grading -- but it must not be silent either."""
    with caplog.at_level(logging.WARNING, logger="hls.core.audio"):
        decode_wav_b64(_wav_b64(_tone(), channels=2, width=2))

    assert "2-channel 16-bit" in caplog.text
    assert "mono PCM16" in caplog.text


def test_the_contract_format_logs_nothing(caplog):
    with caplog.at_level(logging.WARNING, logger="hls.core.audio"):
        decode_wav_b64(encode_wav_b64(_tone(), SR))

    assert caplog.text == ""


def test_an_unconvertible_width_says_so():
    """`wave` only admits 1-4 byte widths, so this is reached through a hand-built
    header; it is the one case that raises, because there is nothing sane to return."""
    b64 = _wav_b64(_tone(0.01), channels=1, width=2)
    raw = bytearray(base64.b64decode(b64))
    raw[34] = 40                                     # bits-per-sample field -> 5 bytes
    try:
        decode_wav_b64(base64.b64encode(bytes(raw)).decode("ascii"))
    except (ValueError, wave.Error) as e:
        assert "sample width" in str(e) or "width" in str(e).lower()
    else:
        raise AssertionError("an unsupported width decoded silently")


# -- the rate recorded alongside a clip must be the clip's own --

def test_sample_rate_of_b64_reads_the_header():
    from hls.core.audio import sample_rate_of_b64

    for rate in (8_000, 16_000, 22_050, 24_000, 44_100, 48_000):
        assert sample_rate_of_b64(encode_wav_b64(_tone(0.02), rate)) == rate


def test_a_tts_stream_records_its_own_rate_not_the_default():
    """A server that streams 24 kHz audio without repeating `sample_rate` in every chunk
    has its own rate written into its trace, not the 16 kHz default."""
    from hls.spec.wire import TTSStreamProtocol

    rate, frame = 24_000, _tone(0.05)
    chunks = [(0.1 * i, {"audio": encode_wav_b64(frame, rate)}) for i in range(3)]

    resp = TTSStreamProtocol().decode_stream(chunks, "r0")

    payload = resp.output.first_of("audio")
    assert payload.sample_rate == rate, (
        f"trace recorded {payload.sample_rate} Hz for {rate} Hz audio")
    # Derived from the rate, not hardcoded: the point is that the two AGREE. A duration
    # computed against 16 kHz while the samples are 24 kHz is the same bug wearing a hat.
    assert abs(payload.duration_sec - 3 * len(frame) / rate) < 1e-6
