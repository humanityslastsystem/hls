"""The ASR oracle transcribes a clip longer than 30 s whole.

Whisper reads a 30 s window. Called bare, its feature extractor pads or cuts every clip to
that window (3,000 mel frames), so the tail of a longer clip is never heard and every word
in it is scored as a deletion. Handing generate() the full features is not enough on its
own either: whisper-large-v3's generation config sets ``return_timestamps`` false, and
transformers refuses input over 3,000 frames unless timestamps are on, because the
sequential long-form path needs them to stitch consecutive windows. So above the window
the extractor must keep the full clip and generate() must be asked for timestamps.

9 of 2,620 LibriSpeech test.clean clips exceed 30 s, the longest at 34.96 s; the oracle
transcribes all of them.

The oracle is driven with a fake processor and model that follow those two rules, so
these tests check what it hands Whisper, not how its source is spelled.
"""

from __future__ import annotations

import contextlib
import sys
import types

import numpy as np
import pytest

from hls.core.audio import encode_wav_b64

SR = 16_000
#: Mel frames per second of audio (hop 160 at 16 kHz), and the window in frames.
FRAMES_PER_SEC = 100
WINDOW_FRAMES = 3_000


# ---- a Whisper that follows the two rules ------------------------------------

class _Features(dict):
    """What the processor returns: the mel frames it kept."""

    def to(self, *args, **kwargs):
        return self


class _Row(list):
    def tolist(self):
        return list(self)


class _FakeProcessor:
    """WhisperFeatureExtractor's length rule: by default it pads or truncates to the
    3,000-frame window; with ``truncation=False, padding="longest"`` it keeps every frame."""

    def __call__(self, array, *, sampling_rate, return_tensors, **kwargs):
        n = int(np.ceil(len(array) / SR * FRAMES_PER_SEC))
        keep_all = kwargs.get("truncation") is False and kwargs.get("padding") == "longest"
        return _Features(n_frames=n if keep_all else WINDOW_FRAMES,
                         attention_mask=kwargs.get("return_attention_mask", False))

    def batch_decode(self, out, skip_special_tokens):
        return [" the whole clip"]


class _FakeWhisper:
    """generate()'s rule for whisper-large-v3, whose generation config sets
    ``return_timestamps`` false: more than 3,000 frames without timestamps is refused."""

    dtype = "bfloat16"

    def __init__(self):
        self.frames: list[int] = []

    def generate(self, *, n_frames, attention_mask, **kwargs):
        if n_frames > WINDOW_FRAMES and not kwargs.get("return_timestamps"):
            # transformers' WhisperGenerationMixin._set_return_timestamps
            raise ValueError("You have passed more than 3000 mel input features "
                             "(> 30 seconds) which automatically enables long-form "
                             "generation which requires the model to predict timestamp "
                             "tokens. Please either pass `return_timestamps=True` or make "
                             "sure to pass no more than 3000 mel input features.")
        self.frames.append(n_frames)
        return [_Row([50258, 400, 50257])]


def _clip(seconds: float) -> str:
    return encode_wav_b64(np.zeros(int(seconds * SR), dtype=np.float32), SR)


def _no_grad_torch() -> types.ModuleType:
    """The one torch call the oracle makes around generate()."""
    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    return torch


@pytest.mark.parametrize("seconds", [29.0, 31.0, 34.96])
def test_the_oracle_transcribes_the_whole_clip(monkeypatch, seconds):
    from hls.core.decode import Greedy
    from hls.core.model import ModelSpec
    from hls.reference.asr import WhisperASREngine

    monkeypatch.setitem(sys.modules, "torch", _no_grad_torch())
    engine = WhisperASREngine(
        model_spec=ModelSpec(base_checkpoint="openai/whisper-large-v3", arch="EncDec",
                             chat_format="whisper"),
        decode=Greedy(), language="en")
    model = _FakeWhisper()
    engine._processor, engine._model, engine._device = _FakeProcessor(), model, "cpu"
    text, _ = engine._transcribe(_clip(seconds), 440)
    assert text == "the whole clip"
    assert model.frames == [max(WINDOW_FRAMES, int(np.ceil(seconds * FRAMES_PER_SEC)))]
