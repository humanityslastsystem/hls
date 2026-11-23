"""L3 -- Qwen3-TTS reference (text-to-speech oracle).

The multimodal-tts oracle. ``Qwen3TTSReference`` is the pure-config ``Reference`` a
verifier owns; ``Qwen3TTSEngine`` is its 1:1 runtime ``Engine``. The engine provides
three capabilities and loads the model lazily on first use, from the mounted
``$HLS_WEIGHTS_DIR`` snapshot:

- ``ReferenceLogprobs``: price a served code matrix under this checkpoint's own
  sampling distribution, position by position, which is the gate this task ships.
- ``DecodeCodes``: run the codec decoder alone, so a waveform-fidelity gate can ask
  what the served codes are supposed to sound like.
- ``StackReport``: the torch stack it runs on.

The oracle never synthesizes: both gates read what the contestant served. Transcription
is not meaningful for a TTS model, so the engine provides no ``Generate`` either.

Mirrors the ``WhisperReference -> WhisperASREngine`` split in ``asr.py``: the config is
serialized into provenance; the engine loads in the verifying role's process, on its engine
host, and its capabilities are method calls.

**What "the model's own decode" means here.** ``generate_custom_voice`` fills every
decode argument the caller leaves ``None`` from the checkpoint's generation config, so
an oracle that names none of them is silently accepting temperature 0.9, top-k 50 and a
repetition penalty of 1.05, without any of it being written down. Every argument is
named here instead, in ``_generate_kwargs``,
and the same values are written into the model's own defaults at load, so the decode is
a thing the task declares rather than a thing the package happens to default to.

The decode is the plain softmax at temperature 1 on all sixteen groups, with no top-k,
no top-p and no repetition penalty. Two structural rules survive because they are the
model's own shape rather than a sampling preference: the special ids are never emitted,
and the end of speech cannot be chosen before frame 2.

It samples rather than taking the argmax. Taking the argmax on all sixteen codebooks --
the decode a token gate would most like to have -- stops the model terminating: a
frame's input embedding is the sum of all sixteen of its code embeddings, so a
degenerate residual predictor feeds the generator its own loop. So the gate measures a
sampled decode rather than pretending otherwise: the server reports the distribution it drew
each code from, and this reference reprices every one of the sixteen groups and the end
of speech under the same rules generation applied. What the codes turn into is graded
separately, by the waveform they have to decode to.

Nothing truncates, and that is deliberate. A truncated row prices some codes at exactly
zero, and a served code priced at zero makes the divergence between two implementations
infinite no matter how small the real disagreement was. Under the full softmax every
code carries positive probability, so every disagreement is a finite number.
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator, Sequence

from hls.core.capabilities import DecodeCodes, ReferenceLogprobs
from hls.core.decode import DecodeConfig, Sample
from hls.core.engine import Engine
from hls.core.faults import HarnessFault
from hls.core.model import ModelSpec
from hls.core.payloads import AudioPayload, Envelope
from hls.reference.engines import ReferenceEngine, TorchStackReport
from hls.reference.references import Reference

if TYPE_CHECKING:
    from hls.spec.taskspec import TaskSpec


#: Repetition penalty applied to codebook 0, on both sides of the gate: the oracle
#: generates under it, and the reference reproduces it when it prices a served code
#: (``_processed_logprobs``).
#:
#: Off. The contract is the plain softmax, so no processor reshapes the distribution a
#: code is drawn from beyond the two structural rules this module keeps, the suppressed
#: special ids and the opening end-of-speech block. A penalty is a fourth number a
#: contestant has to reproduce exactly, and it buys nothing the gate needs.
REPETITION_PENALTY: float = 1.0

#: Whether the sub-talker samples its fifteen codebooks rather than taking the argmax.
#:
#: Not the free choice it looks like. On this checkpoint a greedy sub-talker drives the
#: talker into non-terminating repetition, because a frame's input embedding is the sum
#: of all sixteen of its code embeddings, so a degenerate sub-talker feeds the talker its
#: own loop.
#:
#: The gate grades the distribution rather than the draw: it compares what the server
#: says it sampled from against what this reference computes, at all sixteen groups, so
#: a sampled group is held to account without an argmax.
#:
#: It reads a report rather than a code for a second reason. Replaying generation on
#: its own recorded hidden state does not redraw every code it drew, because one
#: bfloat16 tie changes every step after it. A teacher-forced forward cannot compound a
#: tie, so it prices a reported distribution rather than reproducing drawn ids.
SUBTALKER_SAMPLE: bool = True

#: Sampling shape for the residual codebooks, used only when ``SUBTALKER_SAMPLE``.
#: Temperature 1 and no truncation, which is the contract's plain softmax. Named here
#: rather than inherited because the checkpoint's generation config supplies 0.9 and 50
#: to anything that leaves them unset, and a number the package supplies silently is a
#: rule nobody declared.
SUBTALKER_TEMPERATURE: float = 1.0
SUBTALKER_TOP_K: int = 0

#: Sampling shape for the FIRST codebook, which the talker draws. The same plain softmax
#: as the residual groups, and the same reason for naming it rather than inheriting it.
#:
#: Top-k is 0, meaning no truncation, and that is the load-bearing half. Truncation puts
#: hard zeros in the reference's row, and a served code the reference prices at zero
#: makes the divergence between two implementations infinite however small the real
#: disagreement was. Pricing every code under the full softmax leaves the row positive
#: everywhere, so a disagreement is always a finite number the calibrated ceiling can rank.
TEMPERATURE: float = 1.0
TOP_K: int = 0

#: Decimals the reference rounds its log-probabilities to. The reply carries twenty floats
#: a position; six decimals of a log-probability is a part in a million of an atom's mass,
#: far below the rounding the served report itself arrives with. Changing it changes
#: what a calibration measures.
_REPORT_DECIMALS: int = 6

#: The codec decoder's output rate, in Hz. Fixed by the 12 Hz tokenizer v2 decoder
#: (1,920 samples per codec frame), not a choice. ``_decode`` refuses a decoder that
#: reports anything else rather than letting a rate change pass as a fidelity failure.
CODEC_SAMPLE_RATE: int = 24_000

#: Text-side mode the talker is conditioned in. ``True`` puts the whole sentence in the
#: prefix; ``False`` simulates streaming text input, which conditions each frame on a
#: growing prefix and so is a different decode. Pinned because the two produce
#: different codes for the same sentence.
NON_STREAMING_MODE: bool = True

#: Positions at the start of a generation where the codec end-of-speech id is blocked
#: (HF ``min_new_tokens``, as ``Qwen3TTSForConditionalGeneration.generate`` sets it).
#: Teacher forcing must block it at the same positions or a two-frame reply would be
#: scored against a distribution generation never saw.
_MIN_NEW_TOKENS: int = 2

#: How many of the top talker ids ``generate`` suppresses on codebook 0 (every id in
#: ``[vocab_size - _SPECIAL_ID_SPAN, vocab_size)`` except the end-of-speech id).
_SPECIAL_ID_SPAN: int = 1024


class _PrefixCaptured(Exception):
    """Raised by the capture hook once the talker's prefix tensors exist, to stop
    ``generate`` before it decodes anything."""

    def __init__(self, kwargs: dict) -> None:
        super().__init__("talker prefix captured")
        self.kwargs = kwargs


@contextlib.contextmanager
def _hooked_talker_generate(network: Any, hook: Callable[[Callable, dict], Any]
                            ) -> Iterator[None]:
    """Temporarily route ``network.talker.generate`` through ``hook(original, kwargs)``.

    ``Qwen3TTSForConditionalGeneration.generate`` builds the talker prefix and then
    calls ``self.talker.generate(**kwargs)`` with keyword arguments only, forwarding
    none of its own. That call is the only seam that reaches those tensors, so reading
    the prefix means intercepting it. Restored on the way out, including on an
    exception, so a captured prefix never leaves the model patched."""
    original = network.talker.generate

    def wrapper(**kwargs):
        return hook(original, kwargs)

    network.talker.generate = wrapper
    try:
        yield
    finally:
        network.talker.generate = original


def _frame_embeds(talker: Any, codes: Any) -> Any:
    """The talker's input embedding for each frame: the sum of its per-group code
    embeddings.

    Codebook 0 is embedded by the talker's own input embedding, groups 1..15 by the
    sub-talker's per-group tables. The concatenate-then-sum shape matches the reduction
    generation itself performs, so the summation order is generation's and the bf16
    rounding is the same.

    ``codes`` is ``[T, G]`` int64; the result is ``[T, H]``."""
    parts = [talker.get_input_embeddings()(codes[:, 0:1])]
    group_embeds = talker.code_predictor.get_input_embeddings()
    for i in range(1, codes.shape[1]):
        parts.append(group_embeds[i - 1](codes[:, i:i + 1]))
    import torch  # type: ignore[import-untyped]
    return torch.cat(parts, dim=1).sum(1)


def _trailing_term(prefix: dict, n_frames: int) -> Any:
    """The text-side term generation adds to each frame's embedding: the next
    unconsumed text hidden state while the text lasts, the text-pad embedding after.

    Under the pinned non-streaming mode the whole sentence is already in the prefix and
    this trailing run is short, but it is not empty, so leaving it out shifts every
    frame's conditioning. ``[n_frames, H]``."""
    import torch  # type: ignore[import-untyped]
    trailing = prefix["trailing_text_hidden"]
    pad = prefix["tts_pad_embed"]
    available = trailing.shape[1]
    rows = [trailing[0, j] if j < available else pad[0, 0] for j in range(n_frames)]
    return torch.stack(rows, dim=0)


def _processed_logprobs(rows: Any, codes0: Any, *, suppress: Any,
                        repetition_penalty: float, eos_id: int,
                        temperature: float, top_k: int,
                        min_new_tokens: int = _MIN_NEW_TOKENS) -> Any:
    """Codebook-0 log-probabilities at every position, after the processing generation
    applies before it picks a code.

    Three processors, in the order ``generate`` builds them: the special ids are
    suppressed to ``-inf`` (all but the end-of-speech id), the repetition penalty is
    applied to every codebook-0 id already emitted at that position, and the
    end-of-speech id is blocked for the first ``min_new_tokens`` positions. A verifier
    that skipped them would compare a contestant against a distribution no generation
    ever sampled from.

    ``rows`` is ``[T + 1, V]`` raw head output and ``codes0`` the ``[T]`` served
    codebook-0 sequence; row ``j`` predicts frame ``j``, so its penalty history is
    ``codes0[:j]`` and the last row is the end-of-speech position. The processors
    commute here: suppression writes ``-inf``, which the penalty's multiply and divide
    both preserve, and the end-of-speech id is never in the history because generation
    stops the moment it is drawn.

    ``temperature`` and ``top_k`` are the sampling transforms that follow, applied by
    ``_sampling_logprobs`` in generation's order."""
    import torch  # type: ignore[import-untyped]
    rows = rows.to(torch.float32).clone()
    rows[:, suppress] = -math.inf
    if repetition_penalty != 1.0:
        n_steps, vocab = rows.shape
        hits = torch.zeros((n_steps, vocab), device=rows.device, dtype=torch.float32)
        hits[1:].scatter_(1, codes0.view(-1, 1), 1.0)
        seen = hits.cumsum(0) > 0
        # HF's rule: a positive logit is divided by the penalty and a negative one
        # multiplied, so both move toward -inf.
        penalized = torch.where(rows > 0, rows / repetition_penalty,
                                rows * repetition_penalty)
        rows = torch.where(seen, penalized, rows)
    rows[:min_new_tokens, eos_id] = -math.inf
    return _sampling_logprobs(rows, temperature=temperature, top_k=top_k)


def _sampling_logprobs(rows: Any, *, temperature: float, top_k: int) -> Any:
    """Log-probabilities of one block of raw logits after the sampling transforms, in
    the order generation applies them: temperature, then top-k, then the softmax.

    Order matters and is not the order that reads most naturally. Dividing by the
    temperature before truncating means the truncation keeps the same k ids it would
    have kept anyway -- scaling is monotone -- but the renormalization afterwards is
    over the scaled values, which is a different distribution from truncating first and
    scaling second. Generation does it this way, so a reference that did it the other
    way would grade an honest server against a distribution nothing sampled from.

    Ids outside the kept k get minus infinity rather than being dropped, so the row
    stays the model's full vocabulary and a report naming a truncated id is scored at
    the zero probability the sampler really gave it.

    Under this task's contract neither transform does anything: temperature is 1, so the
    division is a no-op, and ``top_k`` is 0, so the truncation branch is skipped and this
    is a plain log-softmax of the rows. The general form is kept because the argument is
    what makes that a statement about the pins rather than about this function."""
    import torch  # type: ignore[import-untyped]
    rows = rows.to(torch.float32) / temperature
    if 0 < top_k < rows.shape[-1]:
        kept = rows.topk(top_k, dim=-1).values[..., -1:]
        rows = rows.masked_fill(rows < kept, -math.inf)
    return torch.log_softmax(rows, dim=-1)


#: Elements a row of the masked block in ``_gather_and_tail`` is padded to, 512 bytes of
#: float32: the boundary the CUDA caching allocator starts every tensor on, so each row starts
#: where a masked copy of that row alone would. Defence in depth only: this checkpoint's
#: 3,072- and 2,048-wide rows are already multiples of it, and a misaligned row changes no
#: tail, since ``logsumexp`` sums a fresh temporary of the row rather than the row.
_ROW_ALIGN: int = 128


def _gather_and_tail(rows: Any, ids: Sequence[Sequence[int]]
                     ) -> tuple[list[list[float]], list[float]]:
    """Each position's reported log-probabilities and the log mass outside them, for a
    block of positions priced from rows of one vocabulary.

    ``rows`` is ``[n, V]``; row ``i`` is the distribution at the position whose served
    report fixed the partition ``ids[i]``. A pad slot carries a negative id, gathers minus
    infinity and holds no mass. The tail is summed in log space over everything else
    rather than derived as ``log(1 - sum)``, which cancels to noise on a peaked row where
    the reported ids hold nearly all of it.

    The block reaches the host in two reads, whatever its size. Every read of a device
    value waits for the device, and reading each reported id on its own costs one such wait
    per id. The numbers are bit for bit the ones pricing each position on its own gives:
    the gather copies values, the mask covers the same ids, and each tail is one
    ``logsumexp`` over one contiguous row of the vocabulary. One reduction over the whole
    block would sum in another order and move rounded tails on a GPU.

    Rounded to ``_REPORT_DECIMALS``."""
    import torch  # type: ignore[import-untyped]
    n, vocab = int(rows.shape[0]), int(rows.shape[-1])
    if n == 0:
        return [], []
    width = max(len(r) for r in ids)
    # A pad, and a slot a shorter row lacks, points one past the vocabulary, at a column
    # the mask carries only to absorb it.
    index = torch.tensor([[t if t >= 0 else vocab for t in r] + [vocab] * (width - len(r))
                          for r in ids], dtype=torch.long).to(rows.device)
    mask = torch.zeros((n, vocab + 1), dtype=torch.bool, device=rows.device)
    mask.scatter_(1, index, True)
    stride = -(-vocab // _ROW_ALIGN) * _ROW_ALIGN
    masked = torch.empty((n, stride), dtype=rows.dtype, device=rows.device)[:, :vocab]
    masked.copy_(rows).masked_fill_(mask[:, :vocab], -math.inf)
    values = rows.gather(1, index.clamp(max=vocab - 1)).tolist()
    tails = torch.stack([torch.logsumexp(masked[i], dim=-1) for i in range(n)]).tolist()
    gathered = [[round(v, _REPORT_DECIMALS) if t >= 0 else -math.inf for v, t in zip(vals, r)]
                for vals, r in zip(values, ids)]
    return gathered, [round(t, _REPORT_DECIMALS) for t in tails]


class Qwen3TTSEngine(ReferenceEngine, DecodeCodes, ReferenceLogprobs, TorchStackReport):
    """Qwen3-TTS reference engine, pinned to the task's decode (1:1 with
    ``Qwen3TTSReference``).

    Prices the codes a server reported, position by position, and runs the codec
    decoder on the codes it served. Loads lazily on first use so import and TaskSpec
    construction stay cheap and torch-free.

    Loaded through the ``qwen_tts`` package rather than a transformers auto-class: that
    is the only interface published for this checkpoint, and the same one the task's own
    servers use, so the oracle and the contestant run the same code path. The talker's
    conditioning is built by the package's own prefix construction (``_talker_prefix``)
    under the pinned decode arguments (``_generate_kwargs``), which are passed per call
    as well as written into the model's own defaults at load, because per-call arguments
    win over defaults and a code path that reaches the model another way must not
    quietly sample.

    ``Sample`` is the decode this task declares, so the gate is ``ReferenceLogprobs``:
    the server reports the distribution it drew each code from and this engine reprices
    it. There is no argmax comparison, because a sampler had no reason to pick one.

    Codec decoding is batched (``batch_size``). Scoring a report runs one utterance at a
    time, which is the variant that is provably equal to any padded batch.
    """

    def __init__(self, *, model_spec: ModelSpec, decode: DecodeConfig,
                 speaker: str = "Ryan", language: str = "English",
                 instruct: str = "", repetition_penalty: float = REPETITION_PENALTY,
                 subtalker_sample: bool = SUBTALKER_SAMPLE,
                 temperature: float = TEMPERATURE, top_k: int = TOP_K,
                 subtalker_temperature: float = SUBTALKER_TEMPERATURE,
                 subtalker_top_k: int = SUBTALKER_TOP_K, batch_size: int = 16, dtype: str | None = None,
                 gpu_kind: str | None = None) -> None:
        super().__init__(model_spec=model_spec, decode=decode, gpu_kind=gpu_kind)
        #: CustomVoice speaker every scored utterance is conditioned on.
        self.speaker = speaker
        #: Language the talker is conditioned on, in the spelling qwen_tts expects.
        self.language = language
        #: Style instruction prepended to the conditioning. Empty is the pinned value:
        #: an instruction changes the codes for the same sentence, so it is part of the
        #: conditioning every scored position is priced under.
        self.instruct = instruct
        #: Repetition penalty on codebook 0, used by generation and reproduced when a
        #: served code is priced. See ``REPETITION_PENALTY``.
        self.repetition_penalty = repetition_penalty
        #: Whether the sub-talker samples its fifteen codebooks rather than taking the
        #: argmax. See ``SUBTALKER_SAMPLE``: on this checkpoint a greedy sub-talker
        #: drives the talker into non-terminating repetition, so this is not the free
        #: choice it looks like.
        self.subtalker_sample = subtalker_sample
        #: Temperature the first codebook samples at.
        self.temperature = temperature
        #: Top-k the first codebook samples from.
        self.top_k = top_k
        #: Temperature the residual codebooks sample at. Read only when they sample.
        self.subtalker_temperature = subtalker_temperature
        #: Top-k the residual codebooks sample from. Read only when they sample.
        self.subtalker_top_k = subtalker_top_k
        #: Code matrices per codec-decoder call. Bounds peak activation memory, which
        #: grows with the longest utterance in the batch.
        self.batch_size = batch_size
        #: Torch dtype name to load in, or None for the device default (bfloat16 on
        #: cuda, float32 on cpu). A float32 load is the anchor a calibration preflight
        #: compares the bf16 instrument against.
        self.dtype = dtype

    @property
    def sample(self) -> bool:
        """Whether the first codebook is drawn rather than taken at the argmax.

        Read off the harness decode config, which is where a task declares that its
        contract is a sampled one, rather than duplicated as a field. The sampling SHAPE
        is a field, because the harness's ``Sample`` deliberately carries a temperature
        and nothing else: it excludes top-k so that truncation is a deviation its report
        gate can detect. This checkpoint's own decode truncates, so for this task the
        truncation is part of the contract rather than a departure from it, and it lives
        in the fields the reference serializes. See ``reference_logprobs``."""
        return isinstance(self.decode, Sample)

    def _check_decode(self) -> None:
        """Refuse a decode whose temperature disagrees with the declared one.

        Two places name the temperature -- the harness decode config and this engine's
        own field, which the brief renders from -- and they must not be able to differ.
        A contestant reading one number while the reference scores against the other
        would be graded on a distribution nobody told it about."""
        if self.sample and abs(float(self.decode.temperature) - self.temperature) > 1e-9:
            raise HarnessFault(
                f"Qwen3TTSEngine's decode declares temperature "
                f"{self.decode.temperature} but the reference declares "
                f"{self.temperature}; the brief renders one of these and the gate scores "
                f"against the other, so they cannot differ.")

    # -- loading ------------------------------------------------------------------

    def _torch_dtype(self, torch: Any) -> Any:
        """The dtype to load in: the declared one, else bfloat16 on cuda and float32 on
        cpu. An unknown name is refused rather than silently falling back, since the
        whole point of the knob is to say which numeric instrument is running."""
        if self.dtype is None:
            return torch.bfloat16 if self._device == "cuda:0" else torch.float32
        resolved = getattr(torch, self.dtype, None)
        if not isinstance(resolved, torch.dtype):
            raise HarnessFault(
                f"Qwen3TTSEngine was asked for dtype {self.dtype!r}, which is not a "
                f"torch dtype; use e.g. 'bfloat16' or 'float32'.")
        return resolved

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # type: ignore[import-untyped]
        from qwen_tts import Qwen3TTSModel  # type: ignore[import-untyped]
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Shared source policy (ReferenceEngine._weights_source): the mounted snapshot.
        # qwen_tts reads model_info from the hub for a repo id, which a sealed container
        # refuses, so the snapshot is also the only path that works.
        src = self._weights_source()
        self._model = Qwen3TTSModel.from_pretrained(
            src, device_map=self._device, dtype=self._torch_dtype(torch))
        # Qwen3TTSModel is a plain wrapper, not an nn.Module: it holds the network on
        # `.model`, a processor, and the `generate_defaults` mapping that its generate
        # entry points merge their per-call kwargs over. Pin the decode there too, so a
        # code path that reaches the model without the per-call arguments cannot
        # quietly sample. Refuse a wrapper without the mapping rather than claim a
        # pinned decode nothing enforces.
        defaults = getattr(self._model, "generate_defaults", None)
        if not isinstance(defaults, dict):
            raise HarnessFault(
                "Qwen3TTSEngine found no generate_defaults mapping on the loaded "
                "Qwen3TTSModel, which is where its generate entry points read their "
                "decoding from, so the oracle's pinned decode would be unenforced.")
        defaults.update(self._generate_kwargs())
        self._model.model.eval()

    def _generate_kwargs(self) -> dict:
        """The decode every talker call is pinned to.

        ``do_sample`` covers the first codebook and follows the declared decode (on
        under ``Sample``); ``subtalker_dosample`` covers the other fifteen; the
        repetition penalty is the only other processor that touches the first
        codebook's distribution. A sampling shape is named only for the codebooks that
        actually sample, so a greedy configuration carries no temperature it does not
        read."""
        pinned = {"do_sample": self.sample, "subtalker_dosample": self.subtalker_sample,
                  "repetition_penalty": self.repetition_penalty}
        if self.sample:
            pinned["temperature"] = self.temperature
            pinned["top_k"] = self.top_k
        if self.subtalker_sample:
            pinned["subtalker_temperature"] = self.subtalker_temperature
            pinned["subtalker_top_k"] = self.subtalker_top_k
        return pinned

    # -- conditioning and codec decode ---------------------------------------------

    def _instruct_ids(self) -> Any:
        """The tokenized style instruction, or None when there is none. ``None`` is what
        the packaged path passes for an empty instruction, and it is not the same as
        conditioning on an empty instruction block."""
        if not self.instruct:
            return None
        return self._model._tokenize_texts(
            [self._model._build_instruct_text(self.instruct)])[0]

    def _decode(self, codes_list: list[Any]) -> tuple[list[Any], int]:
        """Run the codec decoder over a batch of code matrices: mono float32 waveforms
        and the rate they share.

        The rate is checked rather than trusted: a decoder reporting anything but
        ``CODEC_SAMPLE_RATE`` would make every waveform comparison downstream a
        resampling artifact, which is a broken instrument and not a contestant's fault."""
        import numpy as np
        wavs, sr = self._model.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in codes_list])
        if int(sr) != CODEC_SAMPLE_RATE:
            raise HarnessFault(
                f"the Qwen3-TTS codec decoder reported {int(sr)} Hz, but this reference "
                f"is built on its fixed {CODEC_SAMPLE_RATE} Hz output; a rate change "
                f"means a different decoder than the one the gate was measured on.")
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs], int(sr)

    # -- codec decode as its own capability ---------------------------------------

    async def decode_codes(self, codes_list: list[Any]) -> list[Any]:
        """The codec decoder's waveform for each served code matrix.

        Each entry is a sequence of rows, one per codec frame, each row one id per
        codebook group in generation order, exactly as a contestant sent them. The
        waveforms come back as float32 numpy arrays at ``CODEC_SAMPLE_RATE``.

        No talker runs: this is the decoder alone, which is what makes it a fair
        comparison for the waveform a contestant streamed.

        Shape and range are checked before anything touches the decoder, and a violation
        raises. The wire protocol already bounds both on the way in, so reaching this
        with a bad matrix is a harness bug rather than a contestant's -- but the failure
        mode if it were not checked is an out-of-range id in the decoder's embedding
        gather, a device-side assert that poisons the CUDA context for the rest of the
        run, which is too expensive to leave to a second line of defence."""
        if not codes_list:
            return []
        import asyncio
        await asyncio.to_thread(self._load)
        return await asyncio.to_thread(self._decode_codes_blocking, codes_list)

    def _decode_codes_blocking(self, codes_list: list[Any]) -> list[Any]:
        """Blocking half of ``decode_codes`` (run via ``to_thread``)."""
        import numpy as np
        import torch  # type: ignore[import-untyped]
        out: list[Any] = []
        size = self._codec_codebook_size()
        for i in range(0, len(codes_list), self.batch_size):
            chunk = []
            for j, rows in enumerate(codes_list[i:i + self.batch_size]):
                where = f"decode_codes item {i + j}"
                checked = self._checked_codes(rows, where)
                bad = [v for row in checked for v in row if not 0 <= v < size]
                if bad:
                    raise ValueError(
                        f"{where}: {len(bad)} code(s) outside this codec's "
                        f"[0, {size}) codebooks, first {bad[0]}; the wire protocol "
                        f"bounds them on the way in, so a matrix that reaches the "
                        f"decoder out of range is a harness bug.")
                chunk.append(checked)
            tensors = [torch.as_tensor(rows, dtype=torch.long, device=self._device)
                       for rows in chunk]
            with torch.no_grad():
                wavs, _sr = self._decode(tensors)
            out.extend(np.asarray(w, dtype=np.float32) for w in wavs)
        return out

    # -- the sampling gate's reference distributions --------------------------------

    async def reference_logprobs(self, items: list[Envelope]) -> list[dict]:
        """The reference's log-probabilities at the ids each item's report names, for
        every position of a served utterance.

        An item is one served utterance: ``metadata['text']`` is the sentence the request
        asked for, the item's ``AudioPayload.codec_codes`` is what the server said, and
        ``metadata['report_ids'][j]`` is the ids its report named at position j. Position
        order is row-major over ``(frame, group 0..15)`` and then the end-of-speech
        decision, so an utterance of ``T`` frames has ``16T + 1`` positions.

        Returns ``HFEngine.reference_logprobs``'s dict exactly -- ``p_on_q``,
        ``p_tail_lp``, ``p_eot_lp``, ``n_ctx``, ``eot_ids``, ``max_context`` -- because
        the gate reading it is that engine's gate, unchanged.

        **Where this departs from the text engine, and why.** That one applies the
        temperature and nothing else, and refuses outright to score a decode declaring a
        penalty, because two serving stacks disagree about whether a reported
        distribution is the penalized one. Here there is no such ambiguity: one package
        serves this checkpoint, and its generation runs the first codebook through the
        processors ``_processed_logprobs`` reproduces (the suppressed special ids, the
        repetition penalty and the opening end-of-speech block) before the temperature
        and top-k transforms, so the reference applies all of them rather than skipping
        any. Under this task's pins the penalty is 1, the temperature 1 and the top-k 0,
        so what remains is the two structural rules over a plain softmax.

        The end-of-speech price is the model's own at every position. At a residual
        group it is the mass the sub-talker puts on an id that is not in its vocabulary,
        which is zero, and the log of that is minus infinity. No mask and no special
        case: the end-of-turn count multiplies by a probability of zero there and the
        position contributes nothing to what it counts.

        Inputs are validated before the model loads, so the raise paths run on a
        torch-free host."""
        if not items:
            return []
        self._check_decode()
        prepared: list[tuple[str, list[list[int]], list[list[int]]]] = []
        for item in items:
            text = item.metadata.get("text")
            if not isinstance(text, str):
                raise ValueError(
                    f"reference_logprobs item {item.request_id!r} carries no "
                    f"metadata['text'] (got {text!r}); the sentence the server was asked "
                    f"to say is the conditioning every position is scored under, so "
                    f"there is no fallback.")
            ap = item.first_of("audio")
            if not isinstance(ap, AudioPayload) or ap.codec_codes is None:
                raise TypeError(
                    f"reference_logprobs item {item.request_id!r} must carry the served "
                    f"codec codes in its audio payload (AudioPayload.codec_codes); none "
                    f"present.")
            codes = [list(row) for row in ap.codec_codes]
            report_ids = item.metadata.get("report_ids")
            want = sum(len(row) for row in codes) + 1
            if not isinstance(report_ids, list) or len(report_ids) != want:
                raise ValueError(
                    f"reference_logprobs item {item.request_id!r} carries "
                    f"{len(report_ids) if isinstance(report_ids, list) else None} rows "
                    f"of metadata['report_ids'] for {want} positions; the reference "
                    f"scores one row of reported ids per position, which is every "
                    f"codebook of every frame and then the end of speech.")
            prepared.append((text, codes, [list(r) for r in report_ids]))
        import asyncio
        await asyncio.to_thread(self._load)
        return await asyncio.to_thread(self._reference_logprobs_prepared, prepared)

    def _reference_logprobs_prepared(
        self, prepared: list[tuple[str, list[list[int]], list[list[int]]]]
    ) -> list[dict]:
        """Blocking per-item report-scoring forward passes (run via ``to_thread``).
        See ``reference_logprobs`` for the returned shape."""
        import torch  # type: ignore[import-untyped]
        tc = self._talker_config()
        eos_id = int(tc.codec_eos_token_id)
        talker_vocab = int(tc.vocab_size)
        sub_vocab = self._codec_codebook_size()
        suppress = torch.as_tensor(self._suppress_ids(), dtype=torch.long,
                                   device=self._device)
        # The checkpoint's own facts, the two the caller's ending rule needs and cannot
        # learn on a torch-free host. This model declares no text context window: what
        # bounds a reply is the frame budget the request stamped, which the caller
        # already holds, so None is the honest answer rather than a borrowed number.
        facts = {"eot_ids": [eos_id], "max_context": None}
        out: list[dict] = []
        for index, (text, codes, report_ids) in enumerate(prepared):
            checked = self._checked_codes(codes, f"reference_logprobs item {index}")
            if not checked:
                out.append({"p_on_q": [], "p_tail_lp": [], "p_eot_lp": [],
                            "n_ctx": 0, **facts})
                continue
            if not self._in_vocabulary(checked):
                out.append({"invalid": "a served code lies outside this checkpoint's "
                                       "codebooks"})
                continue
            bad = self._reported_outside(report_ids, len(checked[0]), talker_vocab,
                                         sub_vocab)
            if bad is not None:
                out.append({"invalid": bad})
                continue
            with torch.no_grad():
                out.append({**self._score_report(text, checked, report_ids,
                                                 eos_id=eos_id, suppress=suppress),
                            **facts})
        return out

    def _reported_outside(self, report_ids: list[list[int]], n_groups: int,
                          talker_vocab: int, sub_vocab: int) -> str | None:
        """The first reported id that no distribution at its position covers, or None.

        Range-checked before the model is touched, for the reason the served codes are:
        an id outside a vocabulary reaches a gather, and a device-side assert there
        sticky-poisons the CUDA context for the rest of the run. The bound depends on
        the position: the first codebook is drawn from the talker's whole vocabulary,
        including the special range generation suppresses, while the residual groups are
        drawn from the acoustic codebook alone. Pad slots carry a negative id and are
        skipped; they hold no mass."""
        for pos, row in enumerate(report_ids):
            limit = talker_vocab if pos % n_groups == 0 or pos == len(report_ids) - 1 \
                else sub_vocab
            for tid in row:
                if tid >= limit:
                    return (f"position {pos} reports id {tid}, outside the "
                            f"{limit}-id vocabulary that position is drawn from")
        return None

    def _score_report(self, text: str, codes: list[list[int]],
                      report_ids: list[list[int]], *, eos_id: int,
                      suppress: Any) -> dict:
        """One utterance's reference rows: the log-probability at each reported id, the
        log mass outside them, and the log price of ending the utterance, at every
        position."""
        import torch  # type: ignore[import-untyped]
        network = self._model.model
        n_frames, n_groups = len(codes), len(codes[0])
        prefix = self._talker_prefix(text)
        embeds, mask = prefix["inputs_embeds"], prefix["attention_mask"]
        n_prefix = int(embeds.shape[1])
        codes_t = torch.as_tensor(codes, dtype=torch.long, device=embeds.device)

        frames = _frame_embeds(network.talker, codes_t) + _trailing_term(prefix, n_frames)
        full = torch.cat([embeds, frames.unsqueeze(0).to(embeds.dtype)], dim=1)
        full_mask = torch.cat(
            [mask, torch.ones((1, n_frames), dtype=mask.dtype, device=mask.device)],
            dim=1)
        result = network.talker(inputs_embeds=full, attention_mask=full_mask,
                                use_cache=False, output_hidden_states=True)
        # Rows n_prefix-1 onward predict frame 0's first codebook, then each later
        # frame's, and finally the end of speech: n_frames + 1 rows.
        head = result.logits[0][n_prefix - 1: n_prefix + n_frames]
        first = _processed_logprobs(
            head, codes_t[:, 0], suppress=suppress,
            repetition_penalty=self.repetition_penalty, eos_id=eos_id,
            temperature=self.temperature, top_k=self.top_k)

        # The sub-talker is teacher-forced on the served codes, so group g of a frame is
        # conditioned on that frame's codes 0..g-1 exactly as generation conditioned it.
        hidden = result.hidden_states[0][-1][0]
        sub_hidden = hidden[n_prefix - 1: n_prefix - 1 + n_frames]
        sub_logits, _loss = network.talker.forward_sub_talker_finetune(codes_t, sub_hidden)
        residual = _sampling_logprobs(sub_logits, temperature=self.subtalker_temperature,
                                      top_k=self.subtalker_top_k)

        # Each position is priced from one row of one of two blocks: a frame's first
        # codebook and the end-of-speech decision from ``first`` (row j at position
        # j * n_groups, the last row at the last position), group g of frame f from
        # ``residual[f, g - 1]``. A block is priced at once and reaches the host whole.
        n_positions = len(report_ids)
        first_at = [frame * n_groups for frame in range(n_frames + 1)]
        residual_at = [frame * n_groups + group
                       for frame in range(n_frames) for group in range(1, n_groups)]
        p_on_q: list = [None] * n_positions
        p_tail_lp: list = [None] * n_positions
        p_eot_lp: list = [None] * n_positions
        for block, at in ((first, first_at),
                          (residual.reshape(-1, residual.shape[-1]), residual_at)):
            gathered, tails = _gather_and_tail(block, [report_ids[pos] for pos in at])
            # The model's own price for ending here. Zero at a residual group, whose
            # vocabulary does not contain the end-of-speech id at all, which in log
            # space is minus infinity and prices that position out of the count.
            eot = (block[:, eos_id].tolist() if block.shape[-1] > eos_id
                   else [-math.inf] * len(at))
            for pos, g, t, e in zip(at, gathered, tails, eot, strict=True):
                p_on_q[pos], p_tail_lp[pos], p_eot_lp[pos] = g, t, e
        return {"p_on_q": p_on_q, "p_tail_lp": p_tail_lp, "p_eot_lp": p_eot_lp,
                "n_ctx": n_prefix}

    def _talker_config(self) -> Any:
        """The loaded checkpoint's talker config: vocabulary sizes, the group count and
        the end-of-speech id that every scored position is measured against."""
        return self._model.model.config.talker_config

    def _suppress_ids(self) -> list[int]:
        """The codebook-0 ids ``generate`` suppresses: the top block of the talker
        vocabulary, minus the end-of-speech id, which is the one special the model is
        allowed to choose."""
        tc = self._talker_config()
        eos = int(tc.codec_eos_token_id)
        return [i for i in range(int(tc.vocab_size) - _SPECIAL_ID_SPAN,
                                 int(tc.vocab_size)) if i != eos]

    def _checked_codes(self, rows: Any, where: str) -> list[list[int]]:
        """A served code matrix as plain ints, with its shape checked against the
        checkpoint's group count.

        A ragged matrix has no defined position layout, so it is refused here rather
        than scored: the caller's own contract check is what a contestant is graded on,
        and reaching this with a ragged matrix is a harness bug. Ids are range-checked
        by the callers that can express an out-of-range id as a verdict."""
        n_groups = int(self._talker_config().num_code_groups)
        out: list[list[int]] = []
        for i, row in enumerate(rows):
            vals = [int(v) for v in row]
            if len(vals) != n_groups:
                raise ValueError(
                    f"{where}: codec frame {i} carries {len(vals)} ids but this "
                    f"checkpoint generates {n_groups} per frame; the position layout is "
                    f"undefined for a ragged code matrix.")
            out.append(vals)
        return out

    def _codec_codebook_size(self) -> int:
        """Ids per codebook in this codec: the sub-talker's vocabulary, which is the
        acoustic code space every group draws from. The talker's own vocabulary is
        larger because it also holds the specials only it can emit, so it is not the
        bound a decoder's input has to satisfy."""
        return int(self._model.model.talker.code_predictor.config.vocab_size)

    def _in_vocabulary(self, codes: list[list[int]]) -> bool:
        """Whether every served id is one this checkpoint could have emitted: codebook 0
        against the talker's vocabulary, the rest against the sub-talker's.

        Checked before anything touches the model. An id outside an embedding table
        fires a device-side assert in the gather, which sticky-poisons the CUDA context
        and turns one unfaithful reply into a harness error for the rest of the run;
        a negative id would silently wrap through Python indexing instead."""
        talker_vocab = int(self._talker_config().vocab_size)
        sub_vocab = self._codec_codebook_size()
        return all(0 <= row[0] < talker_vocab
                   and all(0 <= v < sub_vocab for v in row[1:])
                   for row in codes)

    def _talker_prefix(self, text: str) -> dict:
        """The tensors generation would have handed the talker for this sentence:
        ``inputs_embeds``, ``attention_mask``, ``trailing_text_hidden`` and
        ``tts_pad_embed``.

        Built by running the model's own prefix construction and stopping the moment it
        is finished, rather than by reassembling it from the role embeddings, codec
        prefill ids, speaker embedding and text-side embeds by hand. Reassembling it
        would be a second implementation of a dozen constants that the checkpoint is
        free to change, and a prefix that is subtly wrong reads as a contestant's
        infidelity rather than as a harness bug. Costs one prefix build and no
        decoding."""
        def hook(_original, kwargs):
            raise _PrefixCaptured(kwargs)

        model = self._model
        with _hooked_talker_generate(model.model, hook):
            try:
                model.model.generate(
                    input_ids=model._tokenize_texts(
                        [model._build_assistant_text(text)]),
                    instruct_ids=[self._instruct_ids()],
                    languages=[self.language], speakers=[self.speaker],
                    non_streaming_mode=NON_STREAMING_MODE,
                    **model._merge_generate_kwargs(**self._generate_kwargs()))
            except _PrefixCaptured as captured:
                return captured.kwargs
        raise HarnessFault(
            "Qwen3-TTS generation returned without ever calling talker.generate, so the "
            "talker prefix could not be captured; the packaged model's generate path "
            "changed shape, so the conditioning a scored position is priced under cannot "
            "be reproduced.")


class Qwen3TTSReference(Reference):
    """Config for the pinned Qwen3-TTS oracle (1:1 with ``Qwen3TTSEngine``).

    The model identity is ``task.model``, so the oracle runs the task's own TTS
    checkpoint from the task weight mount and needs no separate provisioning.

    Every field here moves what is measured, which is why each is serialized into
    provenance rather than left as an engine default: the voice, the language and the
    instruction are the conditioning every served position is priced under, and the
    sampling fields shape the distribution it is priced against."""

    engine_cls: ClassVar[type[Engine]] = Qwen3TTSEngine
    #: CustomVoice speaker every scored utterance is conditioned on.
    speaker: str = "Ryan"
    #: Language the talker is conditioned on, in the spelling qwen_tts expects.
    language: str = "English"
    #: Style instruction for the voice. Empty is the pinned value.
    instruct: str = ""
    #: Repetition penalty on codebook 0, the one a served code is priced under.
    repetition_penalty: float = REPETITION_PENALTY
    #: Whether the sub-talker samples its fifteen codebooks. See ``SUBTALKER_SAMPLE``.
    subtalker_sample: bool = SUBTALKER_SAMPLE
    #: Temperature the first codebook samples at.
    temperature: float = TEMPERATURE
    #: Top-k the first codebook samples from.
    top_k: int = TOP_K
    #: Temperature the residual codebooks sample at, when they sample.
    subtalker_temperature: float = SUBTALKER_TEMPERATURE
    #: Top-k the residual codebooks sample from, when they sample.
    subtalker_top_k: int = SUBTALKER_TOP_K
    #: Code matrices per codec-decoder call. Serialized because it bounds peak
    #: activation memory, so the value that fits one arena's card is a property of the
    #: run, not a default.
    batch_size: int = 16
    #: Torch dtype name to load in, or None for the device default (bfloat16 on cuda).
    #: Only a preflight anchor sets it; the graded oracle runs the device default.
    dtype: str | None = None

    def build_engine(self, task: "TaskSpec"):
        # gpu_kind: pragmatic default (task's kind), NOT task semantics -- mirrors
        # HFReference/WhisperReference (see ReferenceEngine.__init__).
        return Qwen3TTSEngine(model_spec=task.model, decode=self.decode,
                              speaker=self.speaker, language=self.language,
                              instruct=self.instruct,
                              repetition_penalty=self.repetition_penalty,
                              subtalker_sample=self.subtalker_sample,
                              temperature=self.temperature, top_k=self.top_k,
                              subtalker_temperature=self.subtalker_temperature,
                              subtalker_top_k=self.subtalker_top_k,
                              batch_size=self.batch_size, dtype=self.dtype,
                              gpu_kind=task.hardware.accelerator.kind)
