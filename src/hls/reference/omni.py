"""L3 -- Qwen3-Omni mixed-modality reference (image/audio/text -> text oracle).

The multimodal-omni oracle. ``Qwen3OmniReference`` is the pure-config ``Reference`` a verifier
owns; ``Qwen3OmniEngine`` is its 1:1 runtime ``Engine``, which provides ``Generate``
(an Envelope carrying an image and/or an audio clip plus a text instruction -> a text
Envelope) and loads the model lazily on first generate. teacher_force is not provided:
the omni gate is a task-level metric comparison, not a token-level one.

A modality-specific engine exists because the shared ``engines._messages_of`` reads a
prompt only from ``metadata['messages']`` or a ``TextPayload``, and raises otherwise --
so HFEngine and VLLMEngine structurally cannot carry the image and audio payloads this
task's requests are made of.

Mirrors the ``HFReference -> HFEngine`` split: the config is serialized into
provenance; the engine loads in the verifying role's process, on its engine host, and its
capability is a method call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from hls.core.capabilities import Generate
from hls.core.decode import DecodeConfig, Sample
from hls.core.engine import Engine
from hls.core.faults import HarnessFault
from hls.core.model import ModelSpec
from hls.core.payloads import AudioPayload, Envelope, ImagePayload, TextPayload
from hls.reference.engines import ReferenceEngine
from hls.reference.references import Reference

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec


class Qwen3OmniEngine(ReferenceEngine, Generate):
    """HF transformers Qwen3-Omni greedy engine for mixed-modality chat.

    Reads each prompt's payloads off the Envelope -- ImagePayload (base64 encoded
    image file bytes), AudioPayload (base64 PCM16 WAV) and TextPayload (the
    instruction) -- renders them as Qwen3-Omni chat content parts, decodes greedily
    in text-output mode, and returns a ``TextPayload`` answer. Loads lazily on first
    generate so import + TaskSpec construction stay cheap and torch-free.

    The model class comes from the checkpoint's own ``config.architectures`` rather
    than a hardcoded name: an omni checkpoint's body lives under a multimodal key
    layout that a text-only auto-class would build a differently-shaped tree for,
    load nothing into, and leave randomly initialized.

    The generate call follows the documented Qwen3-Omni pattern, and
    ``tasks/multimodal_omni/server.py`` makes the same calls so oracle and server do not
    decode differently."""

    def __init__(self, *, model_spec: ModelSpec, decode: DecodeConfig,
                 max_new_tokens: int = 256, sample_rate: int = 16_000,
                 batch_size: int = 8, gpu_kind: str | None = None) -> None:
        super().__init__(model_spec=model_spec, decode=decode, gpu_kind=gpu_kind)
        #: Default decode budget, used when a prompt carries no ``max_tokens``.
        self.max_new_tokens = max_new_tokens
        #: The sample rate the processor's audio front end expects. Clips arriving at
        #: any other rate are refused rather than resampled (see ``_audio_array``).
        self.sample_rate = sample_rate
        #: Prompts per generate call. The oracle answers a whole rollout while verify
        #: waits, so it is a throughput job; 8 keeps peak activation memory reasonable
        #: when a batch mixes images and 30-second audio clips.
        self.batch_size = batch_size
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import transformers  # type: ignore[import-untyped]
        from transformers import AutoConfig, AutoProcessor  # type: ignore[import-untyped]
        # Shared source policy (ReferenceEngine._weights_source): the mounted snapshot.
        src = self._weights_source()
        config = AutoConfig.from_pretrained(src, local_files_only=True)
        arch = (getattr(config, "architectures", None) or [""])[0]
        model_cls = getattr(transformers, arch, None)
        if model_cls is None:
            raise RuntimeError(
                f"transformers {transformers.__version__} has no class {arch!r}, the "
                f"architecture {src!r} declares. Loading an omni checkpoint through a "
                f"different class leaves its body randomly initialized, so the oracle "
                f"refuses rather than scoring with weights it cannot account for.")
        self._processor = AutoProcessor.from_pretrained(src, local_files_only=True)
        # dtype="auto" honors the checkpoint's stored precision; device_map="auto"
        # shards across whatever cards the reference slot was given.
        self._model = model_cls.from_pretrained(
            src, dtype="auto", device_map="auto", local_files_only=True).eval()
        # The task contract is text output only. Left enabled, the speech-output head
        # runs on every generate: it costs memory the decode needs and raises inside
        # the talker's own assembly on text-only inputs.
        if hasattr(self._model, "disable_talker"):
            self._model.disable_talker()

    def _audio_array(self, payload: AudioPayload):
        """The clip's waveform, refusing a rate the processor would misread.

        The front end resamples nothing: a clip fed at the wrong rate is silently
        heard at the wrong speed and pitch, which reads as a bad transcript rather
        than as a misconfiguration. The dataset pins the rate it encodes at, so a
        mismatch here is a task-authoring error and says so."""
        from hls.core.audio import decode_wav_b64
        array, sr = decode_wav_b64(payload.audio_b64)
        if sr != self.sample_rate:
            raise ValueError(
                f"Qwen3OmniEngine expects {self.sample_rate} Hz audio, got {sr} Hz. "
                f"Set the audio dataset's target_sample_rate to match the reference.")
        return array

    def _content_parts(self, prompt: Envelope) -> list[dict]:
        """One prompt's payloads as Qwen3-Omni chat content parts.

        Images decode to PIL, audio to a waveform array, text rides as-is. Order
        follows the Envelope so the oracle sees the parts in the order the contestant's
        request carried them."""
        import io

        from PIL import Image as PILImage  # type: ignore[import-untyped]
        from hls.core.image import decode_image_bytes

        parts: list[dict] = []
        for p in prompt.contents:
            if isinstance(p, ImagePayload):
                img = PILImage.open(io.BytesIO(decode_image_bytes(p.image_b64))).convert("RGB")
                parts.append({"type": "image", "image": img})
            elif isinstance(p, AudioPayload):
                parts.append({"type": "audio", "audio": self._audio_array(p)})
            elif isinstance(p, TextPayload):
                parts.append({"type": "text", "text": p.text or ""})
        if not parts:
            raise TypeError(
                f"Qwen3OmniEngine got an empty prompt for request "
                f"{prompt.request_id!r}: no image, audio or text payload to answer.")
        return parts

    def _answer_batch(self, prompts: list[Envelope], max_new: int) -> list[str]:
        """Greedy text-output forward pass for a batch of prompts.

        The oracle is a throughput job, not a latency one: it answers a whole rollout
        while the verify phase waits, so it runs the largest batch that fits rather than
        one prompt at a time. On an H200 a batch of 8 answers 13.3x faster than one
        prompt at a time (14.9 s against 198 s per answer).

        Batched matmuls reassociate floating-point work and greedy decode turns a near-tie
        into a different token, so ``batch_size`` is part of what this oracle answers and
        is serialized on the Reference rather than left as a private default.

        This oracle feeds an end-metric gate comparing mean scores over hundreds of
        requests, not a token-level gate comparing log-probabilities. A token-level
        verifier must never be handed a batched reference.

        Padding is forced LEFT. These are decoder-only generations, so right padding
        would put pad tokens between the prompt and the first generated token and the
        continuations would not line up with their prompts."""
        import torch  # type: ignore[import-untyped]
        from qwen_omni_utils import process_mm_info  # type: ignore[import-untyped]

        convos = [[{"role": "user", "content": self._content_parts(p)}] for p in prompts]
        texts = [self._processor.apply_chat_template(  # type: ignore[union-attr]
            c, add_generation_prompt=True, tokenize=False) for c in convos]
        audio: list = []
        images: list = []
        videos: list = []
        for convo in convos:
            a, i, v = process_mm_info(convo, use_audio_in_video=False)
            audio += a or []
            images += i or []
            videos += v or []

        prev = getattr(self._processor.tokenizer, "padding_side", None)
        self._processor.tokenizer.padding_side = "left"  # type: ignore[union-attr]
        try:
            inputs = self._processor(  # type: ignore[union-attr]
                text=texts, audio=audio or None, images=images or None,
                videos=videos or None, return_tensors="pt", padding=True,
                use_audio_in_video=False)
        finally:
            if prev is not None:
                self._processor.tokenizer.padding_side = prev  # type: ignore[union-attr]

        inputs = inputs.to(self._model.device).to(self._model.dtype)  # type: ignore[union-attr]
        with torch.no_grad():
            out = self._model.generate(  # type: ignore[union-attr]
                **inputs, do_sample=False, thinker_return_dict_in_generate=True,
                thinker_max_new_tokens=max_new, use_audio_in_video=False,
                return_audio=False)
        # Text-output generate returns either the thinker's GenerateOutput or a
        # (text, audio) pair whose first element is it.
        text_out = out[0] if isinstance(out, (tuple, list)) else out
        seqs = text_out.sequences if hasattr(text_out, "sequences") else text_out
        gen = seqs[:, inputs["input_ids"].shape[1]:]
        decoded = self._processor.batch_decode(  # type: ignore[union-attr]
            gen, skip_special_tokens=True)
        if len(decoded) != len(prompts):
            raise HarnessFault(
                f"Qwen3OmniEngine asked for {len(prompts)} answers and decoded "
                f"{len(decoded)}, so answers can no longer be matched to the requests "
                "that asked for them.")
        return [d.strip() for d in decoded]

    async def generate(self, prompts: list[Envelope]) -> list[Envelope]:
        if isinstance(self.decode, Sample):
            raise NotImplementedError(
                "Qwen3OmniEngine supports only greedy decode; Sample is not implemented.")
        if not prompts:
            return []
        self._load()

        def cap_of(prompt: Envelope) -> int:
            # Honor the per-request decode cap the protocol forwards
            # (metadata['max_tokens']) so the oracle and the artifact decode under
            # IDENTICAL budgets; fall back to the configured default when absent.
            req_cap = prompt.metadata.get("max_tokens")
            return (int(req_cap) if isinstance(req_cap, int) and req_cap > 0
                    else self.max_new_tokens)

        # Group by decode cap before batching: one generate call takes a single
        # max_new_tokens, and letting a batch inherit the largest cap in it would give
        # some requests a budget their own request did not ask for.
        order: dict[int, list[int]] = {}
        for idx, prompt in enumerate(prompts):
            order.setdefault(cap_of(prompt), []).append(idx)

        answers: list[str | None] = [None] * len(prompts)
        for max_new, idxs in order.items():
            for start in range(0, len(idxs), self.batch_size):
                chunk = idxs[start:start + self.batch_size]
                for idx, text in zip(chunk, self._answer_batch(
                        [prompts[i] for i in chunk], max_new)):
                    answers[idx] = text

        return [Envelope(request_id=p.request_id,
                         contents=[TextPayload(text=answers[i] or "",
                                               finish_reason="stop")],
                         metadata={"source": "reference"})
                for i, p in enumerate(prompts)]


class Qwen3OmniReference(Reference):
    """Config for the greedy Qwen3-Omni oracle (1:1 with ``Qwen3OmniEngine``).

    The model comes from ``task.model``, so the oracle is always the task's own omni
    checkpoint. ``max_new_tokens`` and ``sample_rate`` pin the decode budget and the
    expected audio rate to the task's own workload cap and dataset encoding."""

    engine_cls: ClassVar[type[Engine]] = Qwen3OmniEngine
    #: Decode budget for a prompt that carries no per-request ``max_tokens``.
    max_new_tokens: int = 256
    #: Sample rate the engine requires of every audio payload (see the engine).
    sample_rate: int = 16_000
    #: Prompts per generate call. Serialized because it bounds peak activation memory,
    #: so the value that fits one arena's card is a property of the run rather than a
    #: default worth hiding in the engine.
    batch_size: int = 8

    def build_engine(self, task: "TaskSpec"):
        # gpu_kind: pragmatic default (task's kind), NOT task semantics -- mirrors
        # HFReference/VLLMReference so the oracle asks for the declared accelerator (see
        # ReferenceEngine.__init__).
        return Qwen3OmniEngine(model_spec=task.model, decode=self.decode,
                               max_new_tokens=self.max_new_tokens,
                               sample_rate=self.sample_rate,
                               batch_size=self.batch_size,
                               gpu_kind=task.hardware.accelerator.kind)
