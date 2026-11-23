"""L3 -- Whisper ASR reference (encoder-decoder speech-to-text oracle).

The multimodal-asr oracle. ``WhisperReference`` is the pure-config ``Reference`` a verifier
owns; ``WhisperASREngine`` is its 1:1 runtime ``Engine``, which provides the
``Generate`` capability (audio Envelope -> text Envelope) and loads the Whisper model
lazily on first generate, from the mounted ``$HLS_WEIGHTS_DIR`` snapshot. teacher_force
is not meaningful for ASR, so the engine provides only ``Generate``.

Mirrors the ``HFReference -> HFEngine`` split: the config is serialized into
provenance; the engine loads in the verifying role's process, on its engine host, and its
capability is a method call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hls.core.capabilities import Generate
from hls.core.decode import DecodeConfig, Sample
from hls.core.engine import Engine
from hls.core.model import ModelSpec
from hls.core.payloads import AudioPayload, Envelope, TextPayload
from hls.reference.engines import ReferenceEngine
from hls.reference.references import Reference

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec


class WhisperASREngine(ReferenceEngine, Generate):
    """HF Whisper (AutoModelForSpeechSeq2Seq) greedy transcription engine.

    Reads each prompt's ``AudioPayload``, decodes the base64 WAV (core.audio), runs
    greedy decode pinned to ``language``/``asr_task`` (so it matches a contestant
    whose Whisper is configured the same way), and returns a ``TextPayload``
    transcript carrying the emitted decoder token ids. Loads lazily on first
    generate so import + TaskSpec construction stay cheap and torch-free."""

    def __init__(self, *, model_spec: ModelSpec, decode: DecodeConfig,
                 max_new_tokens: int = 440, language: str | None = "english",
                 asr_task: str = "transcribe", gpu_kind: str | None = None) -> None:
        super().__init__(model_spec=model_spec, decode=decode, gpu_kind=gpu_kind)
        self.max_new_tokens = max_new_tokens
        self.language = language
        self.asr_task = asr_task
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore[import-untyped]
        from transformers import (  # type: ignore[import-untyped]
            AutoModelForSpeechSeq2Seq, AutoProcessor)
        if torch.cuda.is_available():
            self._device, dtype = "cuda", torch.bfloat16
        else:
            self._device, dtype = "cpu", torch.float32
        # Shared source policy (ReferenceEngine._weights_source): the mounted snapshot.
        src = self._weights_source()
        self._processor = AutoProcessor.from_pretrained(src, local_files_only=True)
        # Load then cast via .to(dtype): version-robust across transformers (the
        # from_pretrained dtype kwarg was torch_dtype in <5.0, dtype in >=5.0).
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(src, local_files_only=True)
        self._model.to(device=self._device, dtype=dtype).eval()

    _WHISPER_SR = 16_000
    #: Seconds of audio Whisper's encoder reads in one pass. A clip longer than this
    #: needs the sequential long-form path; a shorter one needs the window-padded input
    #: the short-form path expects. Named so a check can move it and compare both paths.
    _LONGFORM_SEC = 30

    def _transcribe(self, audio_b64: str, max_new_tokens: int) -> tuple[str, list[int]]:
        import torch  # type: ignore[import-untyped]
        from hls.core.audio import decode_wav_b64
        array, sr = decode_wav_b64(audio_b64)
        if sr != self._WHISPER_SR:
            # Whisper's feature extractor REQUIRES 16 kHz, and a corpus may be encoded
            # at another rate (HFAudioDataset.target_sample_rate) -- resample rather
            # than error. soxr when available (the reference images ship it), linear
            # interp as the dependency-free fallback.
            import numpy as np
            try:
                import soxr  # type: ignore[import-untyped]
                array = soxr.resample(array, sr, self._WHISPER_SR)
            except ImportError:
                n_out = int(round(len(array) * self._WHISPER_SR / sr))
                array = np.interp(np.linspace(0.0, len(array) - 1.0, n_out),
                                  np.arange(len(array)), array).astype(np.float32)
            sr = self._WHISPER_SR
        # Audio past Whisper's 30 s window is otherwise discarded: the feature extractor
        # defaults to padding="max_length" at 30 s, so every word after the cut is scored
        # as a deletion. The oracle then sets a bar below what the model can do, and a
        # contestant that handles long audio correctly is measured against a reference
        # that does not.
        #
        # The long-form arguments apply only above the window. They keep the full
        # features and hand generate() an attention mask, which is what sends it down
        # Whisper's sequential long-form path. Below the window the default pad to 3000
        # frames is what the short-form path requires, and "longest" would hand the
        # encoder a shorter input than it expects.
        long_form = len(array) > self._WHISPER_SR * self._LONGFORM_SEC
        feats = self._processor(  # type: ignore[union-attr]
            array, sampling_rate=sr, return_tensors="pt",
            **({"truncation": False, "padding": "longest",
                "return_attention_mask": True} if long_form else {}))
        inputs = feats.to(self._device, dtype=self._model.dtype)  # type: ignore[union-attr]
        gen_kwargs: dict = {"max_new_tokens": max_new_tokens, "do_sample": False,
                            "num_beams": 1}
        if long_form:
            # Whisper's sequential long-form path stitches consecutive 30 s windows and
            # needs timestamp tokens to know where each one ends; generate() refuses the
            # long input without this. They are special tokens, so skip_special_tokens
            # keeps them out of the transcript that gets scored.
            gen_kwargs["return_timestamps"] = True
        if self.language is not None:
            gen_kwargs["language"] = self.language
            gen_kwargs["task"] = self.asr_task
        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)  # type: ignore[union-attr]
        token_ids = out[0].tolist()
        text = self._processor.batch_decode(  # type: ignore[union-attr]
            out, skip_special_tokens=True)[0].strip()
        return text, token_ids

    async def generate(self, prompts: list[Envelope]) -> list[Envelope]:
        if isinstance(self.decode, Sample):
            raise NotImplementedError(
                "WhisperASREngine supports only greedy decode; Sample is not implemented.")
        if not prompts:
            return []
        self._load()
        outs: list[Envelope] = []
        for prompt in prompts:
            audio = prompt.first_of("audio")
            if not isinstance(audio, AudioPayload):
                raise TypeError(
                    f"WhisperASREngine expects an AudioPayload, got {type(audio)!r}")
            # Honor the per-request decode cap the protocol forwards (metadata['max_tokens'])
            # so the oracle and the artifact decode under IDENTICAL budgets; fall back to the
            # configured default when the prompt carries none.
            req_cap = prompt.metadata.get("max_tokens")
            max_new = int(req_cap) if isinstance(req_cap, int) and req_cap > 0 else self.max_new_tokens
            text, token_ids = self._transcribe(audio.audio_b64, max_new)
            outs.append(Envelope(
                request_id=prompt.request_id,
                contents=[TextPayload(text=text, tokens=token_ids, finish_reason="stop")],
                metadata={"source": "reference"}))
        return outs


class WhisperReference(Reference):
    """Config for the greedy Whisper transcription oracle (1:1 with WhisperASREngine).

    The model comes from ``task.model``, so the oracle always matches the task's own
    Whisper. ``max_new_tokens``/``language``/``asr_task``
    pin the decode to match the contestant server's configuration."""

    engine_cls: ClassVar[type[Engine]] = WhisperASREngine
    max_new_tokens: int = 440
    language: str | None = "english"
    asr_task: str = "transcribe"

    def build_engine(self, task: "TaskSpec"):
        # gpu_kind: pragmatic default (task's kind), NOT task semantics -- mirrors
        # HFReference/VLLMReference so the oracle asks for the declared accelerator (see
        # ReferenceEngine.__init__).
        return WhisperASREngine(model_spec=task.model, decode=self.decode,
                                max_new_tokens=self.max_new_tokens, language=self.language,
                                asr_task=self.asr_task,
                                gpu_kind=task.hardware.accelerator.kind)
