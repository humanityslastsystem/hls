"""L0 -- Audio (de)serialization for the Envelope wire.

Audio rides inside the JSON Envelope as a base64 WAV (PCM16, mono). This module is
the one place that converts between a numpy waveform and that base64 string, so the
dataset (encode), the reference (decode), and the artifact server (decode) all agree
on the byte layout. numpy is imported lazily so importing the spec layer stays
torch/numpy-free on the host.
"""

from __future__ import annotations
import base64
import io
import logging
import wave
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

log = logging.getLogger(__name__)

DEFAULT_SAMPLE_RATE = 16_000


def encode_wav_b64(array: "np.ndarray", sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """Encode a mono waveform (float in [-1, 1] or int16) as a base64 PCM16 WAV."""
    import numpy as np

    a = np.asarray(array)
    if a.ndim > 1:
        a = a.reshape(-1)  # downmix/flatten to mono
    if np.issubdtype(a.dtype, np.floating):
        a = np.clip(a, -1.0, 1.0)
        a = (a * 32767.0).round().astype(np.int16)
    else:
        a = a.astype(np.int16)
    return pcm16_wav_b64(a.tobytes(), sample_rate)


def pcm16_wav_b64(pcm: bytes, sample_rate: int = DEFAULT_SAMPLE_RATE) -> str:
    """The mono PCM16 samples *pcm* as a base64 WAV, the wire's audio format. Of no samples it is
    the 44-byte header alone, the least WAV the decoder reads."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)  # int16
        w.setframerate(int(sample_rate))
        w.writeframes(pcm)
    return base64.b64encode(buf.getvalue()).decode("ascii")


#: Full-scale divisor per WAV sample width, so a decoded waveform lands in [-1, 1]
#: whatever the contestant packed. 3-byte samples are unpacked by hand below (numpy has
#: no 24-bit integer dtype).
_FULL_SCALE = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}


def decode_wav_b64(b64: str) -> tuple["np.ndarray", int]:
    """Decode a base64 WAV back to (float32 mono waveform in [-1, 1], sr).

    The wire contract is mono PCM16 and `encode_wav_b64` always writes that, so the
    oracle's audio is always in that form. A contestant can send something else. Read as
    int16 mono whatever the header says, stereo PCM16 decodes as interleaved mono at twice
    its true length and 24-bit as garbage, with nothing raised, so a packaging mistake
    scores as a correctness failure with nothing in the logs pointing at the real cause.

    So the header is honoured: any supported width is converted and multi-channel audio is
    downmixed, which is what `encode_wav_b64` and `decode_audio_bytes` already do on the
    paths they own. A contestant is not penalised for packaging, and it still pays for the
    extra bytes in its own latency. The departure from the contract is logged rather than
    raised, because every caller of this is inside verification, where an exception would
    take down a whole tenant's grading instead of one request.
    """
    import numpy as np

    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as w:
        sr = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        frames = w.readframes(w.getnframes())

    if width not in _FULL_SCALE:
        raise ValueError(
            f"decode_wav_b64: unsupported sample width {width} bytes; the wire contract "
            "is mono PCM16 and only 8/16/24/32-bit integer PCM can be converted")
    if channels != 1 or width != 2:
        log.warning(
            "audio payload is %d-channel %d-bit, not the mono PCM16 the wire contract "
            "specifies; decoding it anyway (downmixing to mono). This costs the sender "
            "bandwidth and latency, so fix the encoder.", channels, width * 8)

    if width == 3:
        # No 24-bit dtype: widen each 3-byte little-endian sample into int32 by placing
        # it in the HIGH bytes, which sign-extends for free, then scale for the shift.
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        widened = np.zeros((len(b), 4), dtype=np.uint8)
        widened[:, 1:] = b
        a = widened.view(np.int32).reshape(-1).astype(np.float32) / 256.0
    else:
        a = np.frombuffer(frames, dtype={1: np.uint8, 2: np.int16,
                                         4: np.int32}[width]).astype(np.float32)
        if width == 1:
            a = a - 128.0                     # 8-bit WAV is UNSIGNED, centred on 128
    a = a / _FULL_SCALE[width]

    if channels > 1:
        usable = (len(a) // channels) * channels
        a = a[:usable].reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(a, dtype=np.float32), sr


def decode_audio_bytes(raw: bytes, target_sample_rate: int = DEFAULT_SAMPLE_RATE
                       ) -> tuple["np.ndarray", int]:
    """Decode encoded audio (FLAC/WAV/MP3...) bytes to a mono float32 waveform at
    `target_sample_rate`, torch-free (soundfile + soxr). This is the harness-host
    path for dataset clips at PROVISION time -- it keeps audio loading off
    torch/torchcodec so the host stays torch-free (the model only runs inside the
    artifact/reference containers)."""
    import numpy as np
    import soundfile as sf

    arr, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if arr.ndim > 1:                       # downmix to mono
        arr = arr.mean(axis=1)
    if sr != target_sample_rate:
        import soxr
        arr = soxr.resample(arr, sr, target_sample_rate)
        sr = target_sample_rate
    return np.ascontiguousarray(arr, dtype=np.float32), sr


def concat_wav_b64(b64_frames: list[str]) -> tuple[str, float]:
    """Concatenate base64 WAV frames into one base64 WAV; return (b64, duration_sec).
    Used by a streaming codec to assemble streamed audio frames into the full
    waveform. Empty input yields a zero-length clip. All frames must share one sample
    rate (a single stream's codec rate is fixed); a mixed rate is a bug, not something
    to silently mis-time, so it raises."""
    import numpy as np

    if not b64_frames:
        return encode_wav_b64(np.zeros(0, dtype=np.float32), DEFAULT_SAMPLE_RATE), 0.0
    parts, sr0 = [], None
    for fr in b64_frames:
        arr, sr = decode_wav_b64(fr)
        if sr0 is None:
            sr0 = sr
        elif sr != sr0:
            raise ValueError(
                f"concat_wav_b64: mixed sample rates in one stream ({sr0} vs {sr})")
        parts.append(arr)
    full = np.concatenate(parts)
    return encode_wav_b64(full, sr0), len(full) / float(sr0 or 1)


def duration_sec_of_b64(b64: str) -> float:
    """Audio duration in seconds, read from the WAV header without decoding samples."""
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as w:
        return w.getnframes() / float(w.getframerate() or 1)


def sample_rate_of_b64(b64: str) -> int:
    """Sample rate, read from the WAV header without decoding samples.

    The audio itself is the authority on its own rate. A caller that records a rate
    alongside a clip should read it from here rather than carry a default, which would
    record 16 kHz for 24 kHz speech."""
    raw = base64.b64decode(b64)
    with wave.open(io.BytesIO(raw), "rb") as w:
        return w.getframerate()
