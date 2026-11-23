"""Tests for the TTS waveform gate, and for what the two gates ask of the wire.

``CodecFidelityVerifier``'s measurement and each of its fail-closed counts are tested
directly. ``TtsReportFaithfulnessVerifier`` inherits every rule from
the text sampling gate; what is tested here is what it declares to ``check_wiring``, the
one rule whose units it changes, the ending check, which counts positions where a
speech request's cap counts frames, and its own row builder, which states that cap in
positions and refuses a reply whose report it cannot line up with the codes.

Everything runs on the torch-free host. The reference is a fake proxy handing back a
scripted waveform per code matrix. The log-mel comparison itself is real numpy, so the
numbers below are the numbers the deployed gate computes.
"""

from __future__ import annotations

import asyncio
import math

import numpy as np
import pytest

from hls.core.audio import encode_wav_b64
from hls.core.capabilities import DecodeCodes
from hls.core.decode import Sample
from hls.core.payloads import AudioPayload, CodecPosition, Envelope, TextPayload
from hls.core.trace import ArtifactResponse, TraceEvent
from hls.reference.tts import (CODEC_SAMPLE_RATE, TEMPERATURE,
                               Qwen3TTSReference)
from hls.verify.base import EngineAccessor, WiringError, check_wiring
from hls.verify.report import TOP_LOGPROBS_REQUESTED
from hls.verify.tts_faithfulness import (CODEC_SAMPLES_PER_FRAME,
                                         LOGMEL_PSNR_FLOOR_DB,
                                         CodecFidelityVerifier,
                                         TtsReportFaithfulnessVerifier, log_mel,
                                         logmel_psnr_db, time_domain_snr_db)

#: Codebook groups per frame in these fixtures. Four rather than the checkpoint's
#: sixteen, so a gap vector stays readable; the position arithmetic is the same
#: function either way.
GROUPS = 4


def _ref() -> Qwen3TTSReference:
    """The oracle config as the task declares it: sampled, at the checkpoint's own
    temperature. The report gate refuses a greedy reference at wiring time, so this
    has to be the real decode rather than a convenient one."""
    return Qwen3TTSReference(decode=Sample(temperature=TEMPERATURE))


def _speech(seconds: float = 1.0, f0: float = 150.0, seed: int = 0) -> np.ndarray:
    """A broadband, speech-shaped probe: a harmonic stack under an amplitude envelope.
    Broadband matters -- a pure tone leaves most mel bands on the log floor, where every
    comparison agrees for the wrong reason."""
    rng = np.random.default_rng(seed)
    n = int(CODEC_SAMPLE_RATE * seconds)
    t = np.arange(n) / CODEC_SAMPLE_RATE
    wav = sum((1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28))
              for k in range(1, 24))
    env = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    return (wav / np.abs(wav).max() * env * 0.8).astype(np.float32)


def _codes(n_frames: int, start: int = 1) -> list[list[int]]:
    return [[(start + f * GROUPS + g) % 2048 for g in range(GROUPS)]
            for f in range(n_frames)]


def _event(rid: str, *, codes=None, defect=None, wav=None, rate=CODEC_SAMPLE_RATE,
           text="say this", audio=True, max_tokens=1_000_000) -> TraceEvent:
    """One served TTS request: the sentence in, the audio and its codes out."""
    contents = []
    if audio:
        b64 = encode_wav_b64(_speech() if wav is None else wav, rate)
        contents.append(AudioPayload(audio_b64=b64, sample_rate=rate,
                                     codec_codes=codes, codec_codes_defect=defect))
    out = Envelope(request_id=rid, contents=contents)
    inp = Envelope(request_id=rid, contents=[TextPayload(text=text)],
                   metadata={"tenant_id": "t0", "max_tokens": max_tokens})
    return TraceEvent(request_id=rid, input=inp, response=ArtifactResponse(output=out),
                      t_send=0.0, t_recv=1.0)


class _CodecProxy:
    """The reference's codec decoder: one waveform per code matrix, from a scripted
    map keyed by the matrix itself so a test can decide what each reply "should" be."""

    def __init__(self, waveforms):
        self.waveforms = waveforms
        self.asked = None

    async def decode_codes(self, codes_list):
        self.asked = codes_list
        return [self.waveforms[i] for i in range(len(codes_list))]


def _codec_accessor(events, waveforms):
    proxy = _CodecProxy(waveforms)
    return proxy, EngineAccessor(artifact_events=events,
                                 cap_to_provider={DecodeCodes: proxy.decode_codes})


# ---- the codec gate ---------------------------------------------------------

def _codec_gate(**kwargs) -> CodecFidelityVerifier:
    defaults = dict(min_logmel_psnr_db=20.0, max_length_delta_frames=2)
    defaults.update(kwargs)
    return CodecFidelityVerifier(reference=_ref(), **defaults)


def _report_gate(**kwargs) -> TtsReportFaithfulnessVerifier:
    """The shipped faithfulness gate, uncalibrated, for the checks that read only its
    declared wiring and modality."""
    return TtsReportFaithfulnessVerifier(reference=_ref(), **kwargs)


def test_the_same_waveform_passes_and_a_different_one_does_not():
    """Same codes through the same decoder is the honest case, and it is the identical
    signal. A contestant that streamed something else is what the gate exists to see."""
    honest, other = _speech(seed=1), _speech(seed=2, f0=210.0)
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=honest)]
    _proxy, acc = _codec_accessor(events, [honest])
    assert asyncio.run(gate.verify(acc)).passed is True

    events = [_event("r0", codes=_codes(3), wav=other)]
    _proxy, acc = _codec_accessor(events, [honest])
    verdict = asyncio.run(gate.verify(acc))
    assert verdict.passed is False
    assert "psnr_below_floor" in verdict.diagnostics["rejections"]


def test_the_gate_is_every_utterance_not_the_average():
    """One substituted utterance among clean ones fails the run: a mean would let a
    contestant swap a fraction of its replies and stay above any floor."""
    honest = _speech(seed=3)
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=honest),
              _event("r1", codes=_codes(3, start=99), wav=_speech(seed=4, f0=260.0)),
              _event("r2", codes=_codes(3, start=7), wav=honest)]
    _proxy, acc = _codec_accessor(events, [honest, honest, honest])
    verdict = asyncio.run(gate.verify(acc))
    assert verdict.passed is False
    assert verdict.diagnostics["n_below_floor"] == 1
    assert verdict.per_event == {"r0": True, "r1": False, "r2": True}


def test_a_length_difference_is_its_own_check():
    """A signal that is merely longer is not distorted, so length is checked in codec
    frames rather than folded into the fidelity number. Three frames over a two-frame
    tolerance rejects on length alone."""
    honest = _speech(seed=5)
    padded = np.concatenate([honest, np.zeros(3 * CODEC_SAMPLES_PER_FRAME,
                                              dtype=np.float32)])
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=padded)]
    _proxy, acc = _codec_accessor(events, [honest])
    verdict = asyncio.run(gate.verify(acc))
    assert verdict.passed is False
    assert "length_mismatch" in verdict.diagnostics["rejections"]
    assert verdict.event_diagnostics["r0"]["length_delta_frames"] == pytest.approx(3.0)


def test_a_length_difference_inside_the_tolerance_is_not_a_rejection():
    """A decoder that pads or trims its last chunk differently moves by about a frame,
    and that is not what the gate is looking for."""
    honest = _speech(seed=6)
    trimmed = honest[:-CODEC_SAMPLES_PER_FRAME]
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=trimmed)]
    _proxy, acc = _codec_accessor(events, [honest])
    assert asyncio.run(gate.verify(acc)).passed is True


def test_a_reply_at_another_sample_rate_is_compared_rather_than_dropped():
    """The contract pins the codec's rate, so a reply at another one is already
    off-contract. Resampling it means the verdict reports the fidelity the contestant
    achieved instead of an artifact of the mismatch."""
    honest = _speech(seed=7)
    import soxr
    at_16k = np.asarray(soxr.resample(honest, CODEC_SAMPLE_RATE, 16_000),
                        dtype=np.float32)
    gate = _codec_gate(min_logmel_psnr_db=-math.inf)
    events = [_event("r0", codes=_codes(3), wav=at_16k, rate=16_000)]
    _proxy, acc = _codec_accessor(events, [honest])
    verdict = asyncio.run(gate.verify(acc))
    # Back at the codec's rate the lengths line up again, so nothing rejects on length.
    assert "length_mismatch" not in verdict.diagnostics["rejections"]
    assert math.isfinite(verdict.event_diagnostics["r0"]["logmel_psnr_db"])


def test_every_unverifiable_reply_has_its_own_count():
    honest = _speech(seed=8)
    gate = _codec_gate()
    events = [_event("r0", codes=None, defect="response carried no codec_codes"),
              _event("r1", codes=[]),
              _event("r2", audio=False)]
    _proxy, acc = _codec_accessor(events, [])
    verdict = asyncio.run(gate.verify(acc))
    assert verdict.passed is False
    assert set(verdict.diagnostics["rejections"]) == {
        "no_codec_codes", "empty_codec_codes", "no_audio", "no_comparable_replies"}
    assert verdict.per_event == {"r0": False, "r1": False, "r2": False}
    assert verdict.event_diagnostics["r0"]["codec_codes_defect"] == (
        "response carried no codec_codes")
    assert honest is not None


def test_the_per_event_maps_are_keyed_the_way_the_runner_looks_an_event_up():
    """A per-tenant verifier's maps are read under ``request_id`` and a pooled one's
    under ``event_key``. Keying them the other way is invisible in a verdict and silently
    drops every per-event result on the floor."""
    from hls.verify.run import _verdict_key
    honest = _speech(seed=10)
    gate = _codec_gate()
    assert gate.pooled is False
    events = [_event("r0", codes=_codes(3), wav=honest)]
    _proxy, acc = _codec_accessor(events, [honest])
    verdict = asyncio.run(gate.verify(acc))
    expected = {_verdict_key(events[0], gate.pooled)}
    assert set(verdict.per_event) == expected
    assert set(verdict.event_diagnostics) == expected


def test_a_trace_with_nothing_comparable_cannot_pass():
    """Every rule is vacuous on an empty comparison set, so a pass would rest on
    nothing at all."""
    gate = _codec_gate()
    _proxy, acc = _codec_accessor([], [])
    verdict = asyncio.run(gate.verify(acc))
    assert verdict.passed is False
    assert "no_comparable_replies" in verdict.diagnostics["rejections"]


def test_the_codec_gate_asks_the_reference_uncached():
    honest = _speech(seed=9)
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=honest)]
    proxy = _CodecProxy([honest])
    calls: list[str] = []

    async def cached(codes_list):
        calls.append("cached")
        return await proxy.decode_codes(codes_list)

    async def uncached(codes_list):
        calls.append("uncached")
        return await proxy.decode_codes(codes_list)

    acc = EngineAccessor(artifact_events=events,
                         cap_to_provider={DecodeCodes: cached},
                         cap_to_uncached={DecodeCodes: uncached})
    asyncio.run(gate.verify(acc))
    assert calls == ["uncached"]


# ---- what the codec gate holds and computes, whatever a reply claims ------------------


def test_a_reply_past_its_frame_cap_is_not_decoded_and_fails_on_its_own_count():
    """A reply that served more frames than its request's cap ignored the cap it was sent,
    which the report gate fails the trace on. The codec gate does not decode it either, so a
    server that ignores the cap does not choose how much the decoder holds; it names it."""
    honest = _speech(seed=11)
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=honest, max_tokens=2),
              _event("r1", codes=_codes(3, start=5), wav=honest)]
    proxy, acc = _codec_accessor(events, [honest])
    verdict = asyncio.run(gate.verify(acc))
    assert proxy.asked == [_codes(3, start=5)]
    assert verdict.passed is False
    assert "over_cap" in verdict.diagnostics["rejections"]
    assert (verdict.diagnostics["n_over_cap"], verdict.diagnostics["over_cap_events"]) == (1, ["r0"])
    assert verdict.per_event == {"r0": False, "r1": True}
    assert verdict.event_diagnostics["r0"] == {"over_cap": {"frames": 3, "cap": 2}}


def test_a_reply_whose_stated_rate_would_resample_it_far_past_its_codes_fails_on_length_unresampled(monkeypatch):
    """A WAV header may state any rate. Brought to the codec's rate, 4,800 samples stated at
    1 Hz would be 115 million: past the length tolerance however the resampler rounds, so the
    reply fails on length without being resampled, and nothing that size is ever made."""
    import soxr

    honest = _speech(seed=12)

    def refuse(*args, **kwargs):
        raise AssertionError("the gate resampled a reply it had already failed on length")

    monkeypatch.setattr(soxr, "resample", refuse)
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=honest[:4800], rate=1)]
    _proxy, acc = _codec_accessor(events, [honest[:3 * CODEC_SAMPLES_PER_FRAME]])
    verdict = asyncio.run(gate.verify(acc))
    assert verdict.passed is False
    assert verdict.diagnostics["rejections"] == ["length_mismatch"]
    diag = verdict.event_diagnostics["r0"]
    assert diag["length_delta_frames"] == pytest.approx((4800 * CODEC_SAMPLE_RATE - 3 * CODEC_SAMPLES_PER_FRAME)
                                                        / CODEC_SAMPLES_PER_FRAME)
    assert diag["served_samples"] == 4800 * CODEC_SAMPLE_RATE and diag["passed"] is False
    assert "logmel_psnr_db" not in diag


def test_a_reply_at_another_rate_within_the_length_tolerance_is_still_resampled_and_compared():
    """Only a reply that would fail on length anyway goes unresampled: one that would land
    within the tolerance is resampled and graded on its fidelity, as before."""
    import soxr

    honest = _speech(seed=13)
    at_16k = np.asarray(soxr.resample(honest, CODEC_SAMPLE_RATE, 16_000), dtype=np.float32)
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=at_16k, rate=16_000)]
    _proxy, acc = _codec_accessor(events, [honest])
    verdict = asyncio.run(gate.verify(acc))
    assert math.isfinite(verdict.event_diagnostics["r0"]["logmel_psnr_db"])
    assert "length_mismatch" not in verdict.diagnostics["rejections"]


class _Replies:
    """*n* replies, each with a large report and one frame of audio, made afresh on every
    pass and held by nothing but the reader, as a stored trace's view gives them."""

    def __init__(self, n: int, report_positions: int) -> None:
        self.n, self.report_positions = n, report_positions

    def __len__(self) -> int:
        return self.n

    def __iter__(self):
        frame = _speech(seconds=CODEC_SAMPLES_PER_FRAME / CODEC_SAMPLE_RATE, seed=14)
        for i in range(self.n):
            ev = _event(f"r{i}", codes=_codes(1, start=i), wav=frame)
            ap = ev.response.output.contents[0]
            report = [CodecPosition(token_id=1, logprob=-0.5, ids=[1, 2], lps=[-0.5, -1.0])] * self.report_positions
            yield ev.model_copy(update={"response": ArtifactResponse(output=Envelope(
                request_id=ev.request_id, contents=[ap.model_copy(update={"codec_logprobs": report})]))})


def test_the_codec_gate_keeps_no_reply_whole():
    """It keeps of each reply only what it compares: its key, its audio and its codes, never
    the event, whose report it does not read. Forty replies, each with a report of 100,000
    positions (an 800 KB list): holding them would take 32 MB."""
    import tracemalloc

    frame = _speech(seconds=CODEC_SAMPLES_PER_FRAME / CODEC_SAMPLE_RATE, seed=14)
    gate = _codec_gate(min_logmel_psnr_db=-math.inf)
    # One pass first, so what the gate imports and caches on its first reply is not counted.
    asyncio.run(gate.verify(_codec_accessor(_Replies(1, report_positions=1), [frame])[1]))
    _proxy, acc = _codec_accessor(_Replies(40, report_positions=100_000), [frame] * 40)
    tracemalloc.start()
    try:
        verdict = asyncio.run(gate.verify(acc))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert verdict.diagnostics["n_compared"] == 40
    assert peak < 8 * 2**20


def test_the_floor_sits_between_the_measured_honest_and_cheating_populations():
    """The two numbers the codec validation measured, over the utterances the talker
    finishes. A floor outside them is either failing honest decoders or admitting
    substituted audio, and there is no way to tell which from the gate's own output."""
    # Measured under the decode this task pins, over all 64 sentences.
    worst_honest, best_negative = 32.04, 26.28
    assert best_negative < LOGMEL_PSNR_FLOOR_DB < worst_honest
    # Within a fifth of a dB of the midpoint rather than hugging either edge.
    assert abs(LOGMEL_PSNR_FLOOR_DB - (worst_honest + best_negative) / 2) < 0.25
    # And a wide margin over the nearest negative anyone has an incentive to attempt,
    # which is the trimmed leading silence rather than the gain.
    assert LOGMEL_PSNR_FLOOR_DB - 21.37 > 7.0


def test_the_verdict_records_time_domain_snr_without_gating_on_it():
    """The two measurements answer different questions, so a verdict carries both. A
    gain change is the closest thing to honest this gate sees, and it is exactly the
    case log-mel is blind to and time-domain is severe on."""
    honest = _speech(seed=21)
    gate = _codec_gate()
    events = [_event("r0", codes=_codes(3), wav=honest)]
    _proxy, acc = _codec_accessor(events, [honest])
    verdict = asyncio.run(gate.verify(acc))
    assert verdict.passed is True
    # Not infinite: the reply's audio rode the wire as 16-bit PCM, so what comes back
    # differs from the oracle's float waveform by the quantization step and nothing
    # else. That is what an honest reply looks like on this measurement.
    assert verdict.event_diagnostics["r0"]["time_domain_snr_db"] > 60.0


def test_the_two_waveform_measurements_disagree_on_a_gain_change():
    """Recording both is only worth anything if they can differ, and this is the case
    that made it worth doing: log-mel reads a constant gain as a small offset while
    time-domain reads it as total loss."""
    honest = _speech(seed=22)
    # Halved rather than doubled: the same constant offset in log-mel space, with no
    # clipping to confound it, since this probe already peaks near full scale.
    quieter = (honest * 0.5).astype(np.float32)
    assert logmel_psnr_db(honest, quieter) > 20.0
    assert time_domain_snr_db(honest, quieter) < 7.0
    # And the other way round: a small additive noise is nearly invisible in the time
    # domain and obvious in log-mel.
    noisy = (honest + 1e-3 * np.random.default_rng(5).standard_normal(
        honest.size)).astype(np.float32)
    assert time_domain_snr_db(honest, noisy) > 40.0
    assert logmel_psnr_db(honest, noisy) < 20.0


def test_time_domain_snr_handles_the_degenerate_pairs():
    wav = _speech(seed=23)
    assert time_domain_snr_db(wav, wav) == math.inf
    assert time_domain_snr_db(np.zeros_like(wav), wav) == -math.inf
    assert time_domain_snr_db(wav, np.zeros(0, dtype=np.float32)) == -math.inf


# ---- the log-mel measurement ------------------------------------------------

def _htk_filterbank(n_fft: int = 1024, n_mels: int = 80,
                    sr: int = CODEC_SAMPLE_RATE) -> np.ndarray:
    """The HTK-scale triangular filterbank, written out here from its definition.

    A second implementation on purpose. The floor the gate enforces is measured by a
    separate script through torchaudio or librosa, and what makes those two numbers
    comparable is that all three build this same bank: HTK mel spacing
    (``2595*log10(1+f/700)``), triangles in hertz between consecutive mel-spaced edges,
    and no area normalization. Comparing the shipped bank against a library that shares
    its code would prove nothing, so this derives it independently."""
    def to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    edges = to_hz(np.linspace(to_mel(0.0), to_mel(sr / 2), n_mels + 2))
    freqs = np.linspace(0.0, sr / 2, n_fft // 2 + 1)
    bank = np.zeros((n_fft // 2 + 1, n_mels), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = edges[m], edges[m + 1], edges[m + 2]
        rising = (freqs - lo) / (mid - lo)
        falling = (hi - freqs) / (hi - mid)
        bank[:, m] = np.maximum(0.0, np.minimum(rising, falling))
    return bank


def test_the_shipped_filterbank_is_the_htk_one_the_floor_was_measured_with():
    from transformers.audio_utils import mel_filter_bank
    shipped = mel_filter_bank(num_frequency_bins=513, num_mel_filters=80,
                              min_frequency=0.0, max_frequency=CODEC_SAMPLE_RATE / 2,
                              sampling_rate=CODEC_SAMPLE_RATE, norm=None,
                              mel_scale="htk")
    assert np.allclose(shipped, _htk_filterbank(), atol=1e-6)
    # And the check has teeth: the Slaney-scale bank the same call can build is a
    # different bank, so an agreement above is a real agreement and not a tautology.
    slaney = mel_filter_bank(num_frequency_bins=513, num_mel_filters=80,
                             min_frequency=0.0, max_frequency=CODEC_SAMPLE_RATE / 2,
                             sampling_rate=CODEC_SAMPLE_RATE, norm="slaney",
                             mel_scale="slaney")
    assert not np.allclose(slaney, _htk_filterbank(), atol=1e-6)


def _reference_log_mel(wav: np.ndarray) -> np.ndarray:
    """The whole log-mel front end written out from its definition, ``[frames, mels]``.

    A centred, reflect-padded short-time Fourier transform under a periodic Hann
    window, squared to power, through the HTK bank above, floored and logged. Second
    implementation for the same reason as the bank: the number this has to agree with
    is measured elsewhere, by a different library."""
    n_fft, hop = 1024, 256
    window = np.hanning(n_fft + 1)[:n_fft]          # periodic, as an STFT wants
    padded = np.pad(wav, n_fft // 2, mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    frames = np.stack([padded[i * hop: i * hop + n_fft] * window
                       for i in range(n_frames)])
    power = np.abs(np.fft.rfft(frames, n=n_fft, axis=-1)) ** 2
    return np.log10(np.maximum(power @ _htk_filterbank(), 1e-5))


def test_log_mel_is_the_power_spectrogram_the_floor_was_measured_on():
    """Power rather than magnitude, and the bank applied after squaring rather than
    before. Both choices move every error the comparison reports, so both are checked
    against an implementation that shares no code with the one that ships."""
    wav = _speech(seed=20)
    mine, theirs = log_mel(wav), _reference_log_mel(wav)
    n = min(mine.shape[0], theirs.shape[0])
    assert n > 50
    assert np.allclose(mine[:n], theirs[:n], atol=2e-3)
    # Teeth: squaring after the bank instead of before is a different front end, and
    # this comparison sees the difference.
    n_fft, hop = 1024, 256
    from transformers.audio_utils import spectrogram, window_function
    magnitude = spectrogram(wav, window=window_function(n_fft, "hann"),
                            frame_length=n_fft, hop_length=hop, fft_length=n_fft,
                            power=1.0, center=True, mel_filters=_htk_filterbank(),
                            mel_floor=1e-5, log_mel=None).T
    assert not np.allclose(mine[:n], 2.0 * np.log10(np.maximum(magnitude[:n], 1e-5)),
                           atol=2e-3)


def test_log_mel_is_frames_by_mels_and_floored():
    mel = log_mel(_speech(seconds=0.5))
    assert mel.ndim == 2 and mel.shape[1] == 80
    assert mel.min() >= math.log10(1e-5) - 1e-6


def test_an_identical_signal_has_no_error_at_all():
    wav = _speech(seed=11)
    assert logmel_psnr_db(wav, wav) == math.inf


def test_the_measure_falls_as_the_perturbation_grows():
    """Monotone in the size of the difference, which is what lets a single floor
    separate a decoder's own numerical noise from a substituted signal."""
    wav = _speech(seed=12)
    rng = np.random.default_rng(3)
    scores = [logmel_psnr_db(wav, wav + s * rng.standard_normal(wav.size).astype(
        np.float32)) for s in (1e-5, 1e-4, 1e-3, 1e-2)]
    assert scores == sorted(scores, reverse=True)


def test_comparing_against_silence_is_worse_than_any_perturbation():
    wav = _speech(seed=13)
    silence = logmel_psnr_db(wav, np.zeros_like(wav))
    noisy = logmel_psnr_db(wav, wav + 1e-3 * np.random.default_rng(4).standard_normal(
        wav.size).astype(np.float32))
    assert silence < noisy


def test_a_reference_with_no_dynamic_range_scores_the_floor():
    """Nothing to compare against: digital silence has no log-mel range, so a peak-
    relative ratio against it is undefined and the answer is the worst one, never a
    pass."""
    assert logmel_psnr_db(np.zeros(24_000, dtype=np.float32),
                          _speech(seed=14)) == -math.inf


def test_only_the_shared_leading_frames_are_compared():
    """Length is the other check's business. Truncating the served signal must not
    change the fidelity of the part that is there."""
    wav = _speech(seed=15)
    full = logmel_psnr_db(wav, wav + 1e-3)
    half = logmel_psnr_db(wav, (wav + 1e-3)[: wav.size // 2])
    assert full == pytest.approx(half, rel=0.25)


# ---- wiring -----------------------------------------------------------------

class _Proto:
    """The contract markers check_wiring reads, as a task's protocol carries them.

    The report fields default to what the shipped gate demands, so a test that varies
    only ``codec_codes`` is measuring the codes requirement and not tripping over the
    logprob one on the way."""

    def __init__(self, *, token_ids=False, codec_codes=False,
                 return_logprobs=True, top_logprobs=TOP_LOGPROBS_REQUESTED):
        self.emits_token_ids = token_ids
        self.emits_codec_codes = codec_codes
        self.return_logprobs = return_logprobs
        self.top_logprobs = top_logprobs


class _Contract:
    def __init__(self, protocol, modalities=frozenset({"audio"})):
        self.protocol = protocol
        self.output_modalities = modalities


def test_both_gates_need_a_contract_that_asks_for_the_codes():
    """Without the request on the wire every reply would arrive with no codes and fail
    closed, reporting a contestant defect that is really a mis-wired task."""
    for gate in (_report_gate(), _codec_gate()):
        with pytest.raises(WiringError, match="codec codes"):
            check_wiring([gate], _Contract(_Proto(token_ids=True)))
        check_wiring([gate], _Contract(_Proto(codec_codes=True)))


def test_both_gates_accept_audio_output():
    for gate in (_report_gate(), _codec_gate()):
        assert gate.accepts == frozenset({"audio"})
        assert gate.requires_codec_codes is True


# ---- the ending rule reads the cap in the unit its positions are counted in ----

class _EndingProbe:
    """The ending rule alone, with the two inputs it reads: the event and the item.

    Built against the base class rather than a subclass, because the rule is the base
    class's and the point is that it is unchanged."""

    def __init__(self, finish: str | None):
        self.finish = finish

    def run(self, *, served_positions: int, event_cap: int, item_cap: int,
            ends_with_eot: bool = True):
        from hls.verify.verifiers import ReportFaithfulnessVerifier
        gate = ReportFaithfulnessVerifier(reference=_ref())
        gate._finish_reason = lambda _ev, f=self.finish: f   # type: ignore[assignment]
        ev = _event("r0", codes=_codes(2))
        ev = ev.model_copy(update={"input": ev.input.model_copy(
            update={"metadata": {**ev.input.metadata, "max_tokens": event_cap}})})
        item = Envelope(request_id="r0", contents=[TextPayload(text="")],
                        metadata={"max_tokens": item_cap})
        return gate._ending_violation(ev, item, [1] * served_positions, 0, None,
                                      ends_with_eot)


def test_the_ending_rule_is_unchanged_for_text_where_the_two_caps_agree():
    """For text a position is a token and the item copies the request's cap across, so
    both readings are the same number and the rule behaves exactly as before."""
    probe = _EndingProbe("length")
    # Spent the whole budget: no violation.
    assert probe.run(served_positions=64, event_cap=64, item_cap=64) is None
    # Stopped short of it: caught.
    reason = probe.run(served_positions=10, event_cap=64, item_cap=64)
    assert reason is not None and "10 of 64" in reason


#: The checkpoint's end-of-speech id, which a reply's last reported position names.
EOS = 2150


def _position(token_id: int, at: int) -> CodecPosition:
    """One conforming report entry: the served code holds nearly all the mass, and the
    rest of the contract's width is distinct ids small enough to keep it under one."""
    others = [10_000 + at * 100 + k for k in range(TOP_LOGPROBS_REQUESTED - 1)]
    return CodecPosition(token_id, -0.1, [token_id, *others],
                         [-0.1] + [-7.0] * len(others))


def _reported_event(n_frames: int = 3, *, max_tokens: int = 1_000_000,
                    finish: str | None = None, codes=None, logprobs=None) -> TraceEvent:
    """A served TTS request whose reply carries ``n_frames`` frames of codes and a
    conforming report on them: one entry per code, frames outermost, then the end of
    speech. ``codes`` or ``logprobs`` replaces that half of the reply to break it."""
    codes = _codes(n_frames) if codes is None else codes
    if logprobs is None:
        served = [c for row in codes for c in row] + [EOS]
        logprobs = [_position(t, i) for i, t in enumerate(served)]
    out = Envelope(request_id="r0", contents=[AudioPayload(
        audio_b64="", sample_rate=CODEC_SAMPLE_RATE, codec_codes=codes,
        codec_logprobs=logprobs, finish_reason=finish)])
    inp = Envelope(request_id="r0", contents=[TextPayload(text="say this")],
                   metadata={"tenant_id": "t0", "max_tokens": max_tokens})
    return TraceEvent(request_id="r0", input=inp, response=ArtifactResponse(output=out),
                      t_send=0.0, t_recv=1.0)


def _ending(ev: TraceEvent) -> str | None:
    """The ending rule on one reply, with the item the TTS gate builds for it."""
    gate = _report_gate()
    built = gate._rows_for(ev)
    assert not isinstance(built, str), built
    _rows, item = built
    return gate._ending_violation(ev, item, item.first_of("text").tokens, 0, None,
                                  ends_with_eot=False)


def test_the_item_states_the_request_cap_in_positions():
    """A request caps frames, and a reply that spent the cap carries one position per
    code of every frame, then the end of speech. The item has to state the cap in those
    positions. The reply's own position count in its place would say every reply had
    reached its cap."""
    _rows, item = _report_gate()._rows_for(_reported_event(3, max_tokens=1280))

    assert item.metadata["max_tokens"] == 1280 * GROUPS + 1


def test_a_speech_reply_claiming_it_ran_out_of_budget_after_stopping_short_is_caught():
    """A speech reply's positions outnumber its frames, so reading
    the request's frame cap would compare 1,601 positions against a 1,280-frame budget,
    conclude the budget was spent, and exempt a server that stopped at under a third of
    it. Reading the reply's own position count as the cap would exempt every reply. The
    gate's item states the cap in positions, so the comparison is between two numbers of
    the same kind."""
    frames, frame_cap = 400, 1280
    served, position_cap = frames * GROUPS + 1, frame_cap * GROUPS + 1
    assert served > frame_cap, "the unit confusion this guards against"

    caught = _ending(_reported_event(frames, max_tokens=frame_cap, finish="length"))

    assert caught is not None and f"{served} of {position_cap}" in caught
    # A reply that really did spend the budget is not caught.
    assert _ending(_reported_event(frames, max_tokens=frames, finish="length")) is None


# ---- the TTS gate's own row builder refuses what it cannot line up ----------

def test_a_reply_past_its_frame_cap_is_refused():
    """The wire does not know the request's cap, so this is the only place a reply
    longer than its budget is caught. A reply exactly at the cap is not past it."""
    gate = _report_gate()

    assert gate._rows_for(_reported_event(3, max_tokens=2)) == (
        "served 3 frames past its cap of 2")
    assert not isinstance(gate._rows_for(_reported_event(3, max_tokens=3)), str)


def test_frames_of_different_widths_are_refused():
    """A position is one group of one frame, so ragged frames have no position count
    to check the report against."""
    ev = _reported_event(codes=[[1, 2, 3, 4], [5, 6, 7]])

    assert _report_gate()._rows_for(ev) == "its codec frames are not all one width: [3, 4]"


def test_a_report_that_does_not_cover_every_position_is_refused_in_the_replys_terms():
    """A report missing its end-of-speech entry is short by one position. The refusal
    names frames and groups, the units the reply was served in."""
    served = [c for row in _codes(3) for c in row]
    ev = _reported_event(3, logprobs=[_position(t, i) for i, t in enumerate(served)])

    assert _report_gate()._rows_for(ev) == (
        f"{3 * GROUPS} reported positions for {3 * GROUPS + 1} served positions "
        f"(3 frames of {GROUPS} groups, then the end of speech)")


@pytest.mark.parametrize("finish", [None, "stop", "length"])
def test_the_finish_reason_is_the_one_the_audio_payload_reports(finish):
    """The ending rule reads it, and a speech reply carries it on its audio payload."""
    assert _report_gate()._finish_reason(_reported_event(2, finish=finish)) == finish
