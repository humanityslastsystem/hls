"""TTS correctness: is the served speech the checkpoint's own speech?

Two gates, both about the codec codes a contestant's reply carries. Between them they
close the loop from the model's decisions to the bytes that were streamed:

``TtsReportFaithfulnessVerifier``
    Grades the per-position distributions the server reports against the ones this
    checkpoint computes for the same codes. Same rules, same statistics and same
    calibration artifact as the text sampling gate (``ReportFaithfulnessVerifier``,
    which it subclasses); only what a "position" is changes, from an emitted token to
    one code group of one codec frame, plus one position for the end of speech. A
    reply of ``T`` frames is ``16T + 1`` positions.

``CodecFidelityVerifier``
    Decodes those same codes with the oracle's own codec decoder and compares the
    result to the waveform the contestant actually streamed. Same codes and the same
    decoder, so the two signals are frame-aligned by construction and a log-mel PSNR
    floor separates "decoded these codes" from "sent something else".

Why intelligibility is not enough. Round-trip word error rate asks only whether a
listener can make out the sentence, which is a property of any competent speech
synthesizer, so such a gate cannot see that a different model answered.
Faithfulness to the pinned checkpoint's own decode is the property the arena pays for,
and it is what the two gates here measure. The first says the contestant drew its codes
from the distributions this checkpoint would have drawn from; the second says the audio
it shipped is those codes decoded.

**Why two and not one.** The first gate constrains the codes and says nothing about the
bytes: a server could pick honest codes and stream something else entirely. The second
constrains the bytes against the codes and says nothing about whether the codes were the
model's: it would accept an honest decode of codes from any other synthesizer. Neither
subsumes the other, and a contestant has to satisfy both at once.

A sampled decode is what makes the first gate a distribution test rather than an argmax
test. Taking the argmax on all sixteen code groups stops this checkpoint terminating, so
the decode samples, so there is no single code any position is obliged to have produced.
Scoring the reported distribution covers all sixteen groups anyway, including the
fifteen an argmax gate could not score.

Both fail closed. A reply with no codes, with malformed codes, with no report or with no
audio is graded as wrong, never skipped: the wire contract requires all of it, so a
reply missing any is unverifiable evidence rather than an exemption.
"""

from __future__ import annotations

import math
from typing import Any, ClassVar

from pydantic import Field

from hls.core.capabilities import DecodeCodes
from hls.core.payloads import AudioPayload, Envelope, TextPayload
from hls.core.verdict import Verdict
from hls.reference.references import Reference
from hls.reference.tts import CODEC_SAMPLE_RATE
from hls.verify.base import ArtifactRollout, EngineAccessor, Verifier
from hls.verify.flips import past_cap
from hls.verify.report import reply_report
from hls.verify.verifiers import SHOWN_EVENTS, ReportFaithfulnessVerifier

#: Codec frame rate, in frames per second. The 12 Hz tokenizer v2 emits one frame of
#: parallel codebook ids at this rate, which is what turns a sample count into a frame
#: count for the length check.
CODEC_FRAME_RATE_HZ: float = 12.5

#: Waveform samples one codec frame decodes to, at ``CODEC_SAMPLE_RATE``.
CODEC_SAMPLES_PER_FRAME: int = int(round(CODEC_SAMPLE_RATE / CODEC_FRAME_RATE_HZ))

#: Log-mel analysis the waveform comparison runs on. A mel scale rather than a raw
#: spectrogram because the comparison should weigh the parts of the spectrum speech
#: carries information in, and log because codec artifacts are multiplicative. The
#: window is ~43 ms at 24 kHz with a ~11 ms hop, so each codec frame (80 ms) spans
#: several analysis frames and a fidelity failure cannot hide between them.
LOGMEL_N_FFT: int = 1024
LOGMEL_HOP: int = 256
LOGMEL_N_MELS: int = 80

#: Floor applied inside the log, in linear mel-energy units. Bounds how far a silent
#: band can drag the comparison: without it, log of a near-zero band is an arbitrarily
#: large negative number and two decodes of the same codes would look far apart wherever
#: they are both silent.
LOGMEL_FLOOR: float = 1e-5

#: Per-utterance log-mel PSNR floor, in dB, under the decode this task pins: the least
#: fidelity a reply's streamed audio may have against the oracle's decode of its codes.
#: It sits about midway between the worst honest decoder variant (chunked streaming
#: decode) and the closest audio altered after the decode.
LOGMEL_PSNR_FLOOR_DB: float = 29.0

#: Tolerated difference in length between the decoded codes and the streamed waveform,
#: in codec frames. Two frames is 160 ms: enough for a decoder that pads or trims its
#: final chunk differently, far short of the length change any resampling, trimming or
#: time-stretching produces.
MAX_LENGTH_DELTA_FRAMES: int = 2


def log_mel(wav: Any, *, sample_rate: int = CODEC_SAMPLE_RATE,
            n_fft: int = LOGMEL_N_FFT, hop: int = LOGMEL_HOP,
            n_mels: int = LOGMEL_N_MELS, floor: float = LOGMEL_FLOOR) -> Any:
    """One waveform's log-mel spectrogram, ``[n_frames, n_mels]``.

    A POWER mel spectrogram on the HTK mel scale with UNNORMALIZED triangular filters,
    floored at ``floor`` and taken to base-10 log, over a centred Hann window. Each of
    those four choices changes the numbers, so each is pinned here: this is the function
    the codec floor was measured with and the function the deployed gate grades with,
    and a floor measured under one convention says nothing about a comparison run under
    another.

    Computed through transformers' own numpy audio utilities, so the verifying host
    needs no torch and no extra dependency. The construction is the standard one that
    ``torchaudio.transforms.MelSpectrogram(power=2.0, norm=None, mel_scale="htk")`` and
    ``librosa.feature.melspectrogram(power=2.0, htk=True, norm=None)`` also build, which
    is what lets a measurement taken with either of those be read against this."""
    import numpy as np
    from transformers.audio_utils import (mel_filter_bank, spectrogram,
                                          window_function)

    filters = mel_filter_bank(
        num_frequency_bins=n_fft // 2 + 1, num_mel_filters=n_mels,
        min_frequency=0.0, max_frequency=sample_rate / 2.0,
        sampling_rate=sample_rate, norm=None, mel_scale="htk")
    mel = spectrogram(
        np.asarray(wav, dtype=np.float32), window=window_function(n_fft, "hann"),
        frame_length=n_fft, hop_length=hop, fft_length=n_fft, power=2.0,
        center=True, mel_filters=filters, mel_floor=floor, log_mel=None)
    # spectrogram returns [n_mels, n_frames]; frames-first reads more naturally
    # everywhere the two signals are aligned by frame.
    return np.log10(np.maximum(np.asarray(mel, dtype=np.float32), floor)).T


def logmel_psnr_db(reference: Any, served: Any, *,
                   sample_rate: int = CODEC_SAMPLE_RATE) -> float:
    """Peak-signal-to-noise ratio between two waveforms' log-mel spectrograms, in dB.

    The peak is the reference's own log-mel range (its max minus its min), so the ratio
    is read against how much dynamic range this particular utterance has rather than
    against a constant: a quiet sentence and a loud one are then held to the same
    relative bar. The two spectrograms are compared frame by frame from the start over
    the frames they share; a length difference is a separate check, because a signal
    that is merely longer should not be scored as distorted.

    Returns ``+inf`` for a bit-identical pair (no error at all) and ``-inf`` when the
    reference has no dynamic range to speak of but the two still differ, which is the
    degenerate case of comparing against silence. Both are decisions a floor can be
    compared against, so neither needs a special case at the call site. A measurement
    that reports the same two cases as "undefined" is reporting the same two facts: the
    ratio is undefined in both, and what differs is only which way a gate should read
    it, which is a gate's question and not the measurement's."""
    import numpy as np

    ref_mel = log_mel(reference, sample_rate=sample_rate)
    got_mel = log_mel(served, sample_rate=sample_rate)
    n = min(ref_mel.shape[0], got_mel.shape[0])
    if n == 0:
        return -math.inf
    diff = ref_mel[:n].astype(np.float64) - got_mel[:n].astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        return math.inf
    peak = float(ref_mel[:n].max() - ref_mel[:n].min())
    if peak <= 0.0:
        return -math.inf
    return 10.0 * math.log10(peak * peak / mse)


def time_domain_snr_db(reference: Any, served: Any) -> float:
    """Signal-to-noise ratio between two waveforms in the time domain, in dB, over the
    samples they share.

    Recorded per utterance and never gated. A constant gain is an additive offset in
    log-mel space, and this ratio is severe on one. It is as severe on a sample-level
    shift that leaves the audio sounding identical, which this gate does not reject, so
    it stays a diagnostic. A verdict that carries both numbers lets a reader tell those
    two cases apart.

    ``+inf`` for a bit-identical pair; ``-inf`` when the reference is silent and the
    served signal is not."""
    import numpy as np

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    got = np.asarray(served, dtype=np.float64).reshape(-1)
    n = min(ref.size, got.size)
    if n == 0:
        return -math.inf
    signal = float(np.sum(ref[:n] * ref[:n]))
    noise = float(np.sum((ref[:n] - got[:n]) ** 2))
    if noise == 0.0:
        return math.inf
    if signal == 0.0:
        return -math.inf
    return 10.0 * math.log10(signal / noise)


def _resampled(wav: Any, sample_rate: int, target: int) -> Any:
    """``wav`` at ``target`` Hz, resampled only when it is not already there.

    The contract pins the codec's own rate, so a reply at another rate is already
    off-contract; resampling it rather than refusing it means the verdict reports the
    fidelity the contestant actually achieved instead of an artifact of the mismatch."""
    if sample_rate == target:
        return wav
    import numpy as np
    import soxr
    return np.asarray(soxr.resample(wav, sample_rate, target), dtype=np.float32)


def _resampled_samples(n_samples: int, sample_rate: int, target: int) -> float:
    """How many samples ``_resampled`` makes of *n_samples* at *sample_rate*: the length at
    ``target`` Hz, which the resampler rounds to within half a sample (soxr), so a length
    check can be read before any of them is made."""
    return n_samples * target / sample_rate


class TtsReportFaithfulnessVerifier(ReportFaithfulnessVerifier):
    """The sampling gate, over a speech reply's codec report.

    Every check, threshold, contract rule and refusal is
    ``ReportFaithfulnessVerifier``'s, unchanged. What this class supplies is where the
    facts live: a speech reply carries its served codes and its report on the audio
    payload rather than the text one, and its finish reason there too. Nothing here
    decides anything about faithfulness.

    **A position is one codebook of one frame.** A reply of ``T`` frames has ``16T + 1``
    of them, row-major over ``(frame, group)`` and then the end-of-speech decision, which
    is the order the wire protocol fixes on ``AudioPayload.codec_logprobs``. Under the
    checkpoint's own decode every one of those is sampled, which is why the gate is this
    one: there is no argmax for a sampled position to be faithful to, and grading a
    served code against one would measure nothing.

    **The end-of-speech price needs no special case.** At a residual group the
    distribution is the sub-talker's, over a vocabulary that does not contain the
    end-of-speech id, so the reference prices ending there at zero. The count multiplies
    by that zero and the position contributes nothing, which is the arithmetic doing what
    a hand-written mask would have done, without the mask.

    The waveform is not this gate's business. That the audio a contestant streamed is the
    decode of the codes it reported is ``CodecFidelityVerifier``, and between them the
    two close the loop from what the model chose to what the listener received.
    """

    accepts: ClassVar[frozenset[str]] = frozenset({"audio"})
    requires_codec_codes: ClassVar[bool] = True

    def _rows_for(self, ev) -> tuple[Any, Envelope] | str:
        """One event's partition and the item the reference scores it against, or the
        reason its report violates the contract.

        The served sequence is every code of every frame in order, then the code the
        reply says it ended on. That last id comes off the report's own final entry
        rather than from a constant here: the verifier has no way to know the
        checkpoint's end-of-speech id before it has asked the reference, and the base
        class already refuses a reference whose end-of-turn ids disagree with the
        calibration's. A reply naming something else there is caught by the ending rule,
        which sees a last token that is not one of the checkpoint's.

        The item carries the codes rather than the waveform, and the sentence rather than
        a conversation, because that is what this reference reproduces a position from.
        The request's stamped cap counts frames, and the item states it in positions,
        ``cap * groups + 1``, the count a reply that spent the cap carries. The base
        class's at-the-budget test then compares two numbers in the same unit."""
        out = ev.response.output if ev.response is not None else None
        ap = out.first_of("audio") if out is not None else None
        if not isinstance(ap, AudioPayload) or ap.codec_codes is None:
            return ("the reply carries no codec codes"
                    + (f" ({ap.codec_codes_defect})"
                       if isinstance(ap, AudioPayload) and ap.codec_codes_defect else ""))
        if not ap.codec_codes:
            return "the reply served no codec frames"
        if ap.codec_logprobs is None:
            return "the reply carries no reported logprobs for its codes"
        codes = [list(row) for row in ap.codec_codes]
        widths = {len(row) for row in codes}
        if len(widths) != 1:
            return f"its codec frames are not all one width: {sorted(widths)}"
        n_groups = widths.pop()
        want = len(codes) * n_groups + 1
        if len(ap.codec_logprobs) != want:
            return (f"{len(ap.codec_logprobs)} reported positions for {want} served "
                    f"positions ({len(codes)} frames of {n_groups} groups, then the end "
                    f"of speech)")
        end_id = ap.codec_logprobs[-1].token_id
        if end_id is None:
            return "its last reported position names no token id, so what it ended on " \
                   "is unstated"
        served = [c for row in codes for c in row] + [int(end_id)]
        cap = ev.input.metadata.get("max_tokens")
        if isinstance(cap, int) and past_cap(len(codes), cap):
            return f"served {len(codes)} frames past its cap of {cap}"
        # reply_report reads a payload's `logprobs` and nothing else, so the report is
        # handed over on a carrier rather than by teaching it a second payload type.
        # model_construct: the carrier is not re-validated, so the packed positions go
        # through as they are rather than being rebuilt as TokenLogprob models.
        rows = reply_report(TextPayload.model_construct(text="", logprobs=ap.codec_logprobs),
                            served, width=self.report_width)
        if isinstance(rows, str):
            return rows
        if not isinstance(cap, int) or cap <= 0:
            raise ValueError(
                f"TtsReportFaithfulnessVerifier needs the request's stamped output cap "
                f"(metadata['max_tokens']) on event {ev.event_key!r}; got {cap!r}. The "
                f"drive loop stamps it on every request "
                f"(hls.spec.drivers.stamp_request).")
        tp = ev.input.first_of("text")
        if tp is None or not getattr(tp, "text", ""):
            raise ValueError(
                f"TtsReportFaithfulnessVerifier found no sentence on event "
                f"{ev.event_key!r}: the request carried no text payload, so there is no "
                f"conditioning to score its codes under")
        item = Envelope(
            request_id=ev.request_id,
            # The served positions ride as the item's tokens because that is where the
            # shared reduction reads them; the codes ride beside them because that is
            # what the reference conditions its forward on.
            contents=[TextPayload(text="", tokens=served),
                      AudioPayload(audio_b64="", sample_rate=CODEC_SAMPLE_RATE,
                                   codec_codes=codes)],
            metadata={"text": tp.text, "report_ids": rows.q_ids,
                      "max_tokens": cap * n_groups + 1})
        return rows, item

    def _finish_reason(self, ev) -> str | None:
        """The reply's own reported finish reason, off the audio payload."""
        ap = ev.response.output.first_of("audio") if ev.response is not None else None
        return ap.finish_reason if isinstance(ap, AudioPayload) else None


class CodecFidelityVerifier(Verifier):
    """The streamed waveform must be the served codes, decoded.

    For each reply, the oracle's codec decoder is run over the codes the reply carries
    and the result is compared to the audio the contestant actually streamed. Two
    checks, both per utterance, both fail-closed, and the run passes only when every
    utterance passes both:

      Length: the two signals may differ by at most ``max_length_delta_frames`` codec
              frames. A decoder that pads its last chunk differently moves by a frame
              or so; trimming silence, resampling or speaking faster moves by far more.

      Fidelity: the log-mel PSNR of the streamed audio against the decoded codes must
              reach ``min_logmel_psnr_db``. Same codes through the same decoder, so two
              honest runs differ only by the decoder's own numerical noise and sit far
              above any floor; audio that came from somewhere else sits far below.

    This is the half of the TTS gate that no amount of token-level faithfulness covers.
    A contestant can report the codes the checkpoint would have chosen and stream
    whatever audio it likes -- a cheaper synthesizer's output, the same speech sped up,
    a cached clip -- and the token gate would see nothing wrong. Tying the bytes to the
    codes is what makes the reported codes binding.

    What fails closed, each on its own count in the verdict: a reply with no audio
    payload, a reply whose codes are absent or were rejected on the wire, a reply with
    zero frames, a reply past its request's frame cap, which is never decoded, and a
    comparison that produced a non-finite number. None of those is a number below a
    threshold, so none of them is scored as one.

    What the gate holds is bounded by the workload, not by what a reply claims: of each
    reply it keeps its key, its audio and its codes, never the event; a reply past its
    frame cap is not decoded; and a reply whose header states a rate that would make its
    audio longer than the length check allows fails on length before it is resampled.
    Each changes only the order of the checks: the report gate fails a trace with a reply
    past its cap, and the length check fails the other reply after resampling as before.

    The decoder is reached through the ``DecodeCodes`` capability on the task's own TTS
    oracle, uncached: the codes are one contestant's own, so no other arm could ever be
    served a stored answer.

    Per-tenant, not pooled: the floor is a per-utterance bound with no pooled statistic
    behind it, so a tenant's slice is graded on its own and every event carries its own
    verdict. That is also what keys the per-event maps on ``request_id`` -- the key the
    runner looks an event up under for a per-tenant verifier, unique within the slice
    this ever sees (``hls.verify.run._verdict_key``).
    """

    #: The TTS oracle whose codec decoder defines what the codes are supposed to sound
    #: like. The task's own checkpoint, so this is the same reference the faithfulness
    #: gate prices its reports against.
    reference: Reference
    #: Per-utterance log-mel PSNR floor, in dB. The default's derivation is on
    #: ``LOGMEL_PSNR_FLOOR_DB``. A task should still pass it explicitly rather than
    #: inherit it, so that a re-measurement on another checkpoint or corpus has one
    #: obvious place to land and the brief has something to render.
    min_logmel_psnr_db: float = LOGMEL_PSNR_FLOOR_DB
    #: Tolerated length difference between the decoded codes and the streamed audio, in
    #: codec frames. ge=0: a negative tolerance would reject every reply including an
    #: exact one.
    max_length_delta_frames: int = Field(default=MAX_LENGTH_DELTA_FRAMES, ge=0)
    requires = frozenset({DecodeCodes})
    accepts: ClassVar[frozenset[str]] = frozenset({"audio"})
    requires_codec_codes: ClassVar[bool] = True

    async def verify(self, engine: EngineAccessor) -> Verdict:
        import numpy as np
        from hls.core.audio import decode_wav_b64

        events = engine[ArtifactRollout]
        # Split the trace into the replies that can be compared at all and the ones that
        # cannot. The second group never reaches the decoder: there is nothing to decode,
        # and each of them is its own rejection below. Of a comparable reply only its key,
        # its audio as streamed, at the rate its header states, and its codes are kept, never
        # the event, whose report this gate does not read.
        comparable: list[tuple[str, Any, int, list[list[int]]]] = []
        no_audio: list[str] = []
        no_codes: list[dict] = []
        empty_codes: list[str] = []
        over_cap: list[tuple[str, int, int]] = []
        for ev in events:
            key = ev.request_id
            ap = ev.response.output.first_of("audio")
            if not isinstance(ap, AudioPayload) or not ap.audio_b64:
                no_audio.append(key)
                continue
            if ap.codec_codes is None:
                no_codes.append({"event": key, "defect": ap.codec_codes_defect})
                continue
            if not ap.codec_codes:
                empty_codes.append(key)
                continue
            # A reply past its request's frame cap ignored the cap it was sent, which the
            # report gate fails the trace on (``TtsReportFaithfulnessVerifier._rows_for``).
            # It is failed here before the decoder sees it, as the token gate fails one
            # before the reference does: the order changes, never a verdict, and a server
            # that ignores the cap does not choose how much the decoder holds.
            cap = ev.input.metadata.get("max_tokens")
            if isinstance(cap, int) and past_cap(len(ap.codec_codes), cap):
                over_cap.append((key, len(ap.codec_codes), cap))
                continue
            served, rate = decode_wav_b64(ap.audio_b64)
            comparable.append((key, served, rate, [list(row) for row in ap.codec_codes]))

        decoded = (await engine.uncached(DecodeCodes)([c for _, _, _, c in comparable])
                   if comparable else [])

        per_event: dict[str, bool] = {}
        event_diagnostics: dict[str, dict] = {}
        below_floor: list[str] = []
        wrong_length: list[str] = []
        non_finite: list[str] = []
        worst_psnr = math.inf
        for (key, served, rate, _codes), oracle in zip(comparable, decoded):
            oracle = np.asarray(oracle, dtype=np.float32).reshape(-1)
            # The length check is read before the audio is resampled: one whose resampled
            # length would be past the tolerance by more than the resampler's rounding fails
            # on length as it would after resampling, without the resampling, whose output a
            # header stating a tiny rate can make arbitrarily large.
            at_rate = _resampled_samples(len(served), rate, CODEC_SAMPLE_RATE)
            if abs(at_rate - len(oracle)) / CODEC_SAMPLES_PER_FRAME > self.max_length_delta_frames + 1 / CODEC_SAMPLES_PER_FRAME:
                wrong_length.append(key)
                per_event[key] = False
                event_diagnostics[key] = {
                    "length_delta_frames": abs(at_rate - len(oracle)) / CODEC_SAMPLES_PER_FRAME,
                    # What the streamed audio would be at the codec's rate, not made.
                    "served_samples": round(at_rate),
                    "decoded_samples": int(len(oracle)),
                    "passed": False,
                }
                continue
            served = _resampled(served, rate, CODEC_SAMPLE_RATE)
            delta_frames = abs(len(served) - len(oracle)) / CODEC_SAMPLES_PER_FRAME
            psnr = logmel_psnr_db(oracle, served)
            length_ok = delta_frames <= self.max_length_delta_frames
            # NaN loses every comparison, so it is tested for rather than compared: an
            # unnoticed one would read as a pass on a broken measurement.
            fidelity_ok = not math.isnan(psnr) and psnr >= self.min_logmel_psnr_db
            if math.isnan(psnr):
                non_finite.append(key)
            elif not fidelity_ok:
                below_floor.append(key)
            if not length_ok:
                wrong_length.append(key)
            if not math.isnan(psnr):
                worst_psnr = min(worst_psnr, psnr)
            per_event[key] = length_ok and fidelity_ok
            event_diagnostics[key] = {
                "logmel_psnr_db": psnr,
                # Recorded, never gated: see time_domain_snr_db. It is what tells a
                # reader whether a borderline utterance differs in level or in content.
                "time_domain_snr_db": time_domain_snr_db(oracle, served),
                "length_delta_frames": delta_frames,
                "served_samples": int(len(served)),
                "decoded_samples": int(len(oracle)),
                "passed": per_event[key],
            }
        for key in no_audio + empty_codes:
            per_event[key] = False
        for entry in no_codes:
            per_event[entry["event"]] = False
            event_diagnostics[entry["event"]] = {"codec_codes_defect": entry["defect"]}
        for key, frames, cap in over_cap:
            per_event[key] = False
            event_diagnostics[key] = {"over_cap": {"frames": frames, "cap": cap}}

        rejections = [
            name for name, fired in (
                ("psnr_below_floor", bool(below_floor)),
                ("length_mismatch", bool(wrong_length)),
                ("non_finite_psnr", bool(non_finite)),
                ("no_audio", bool(no_audio)),
                ("no_codec_codes", bool(no_codes)),
                ("empty_codec_codes", bool(empty_codes)),
                ("over_cap", bool(over_cap)),
                # An empty trace is not evidence of fidelity. Every rule above is
                # vacuous on it, so a pass would rest on nothing at all.
                ("no_comparable_replies", not comparable),
            ) if fired
        ]
        return Verdict(
            passed=not rejections,
            per_event=per_event,
            diagnostics={
                # Which rules rejected, in the order they are documented. Empty on a
                # pass; the counts below say by how much.
                "rejections": rejections,
                "n_compared": len(comparable),
                "min_logmel_psnr_db": self.min_logmel_psnr_db,
                "max_length_delta_frames": self.max_length_delta_frames,
                # The worst utterance's fidelity, which is what the floor is read
                # against. inf when nothing was compared.
                "worst_logmel_psnr_db": worst_psnr,
                "n_below_floor": len(below_floor),
                "n_wrong_length": len(wrong_length),
                "n_non_finite": len(non_finite),
                "n_no_audio": len(no_audio),
                "n_no_codec_codes": len(no_codes),
                "n_empty_codec_codes": len(empty_codes),
                "n_over_cap": len(over_cap),
                # A handful of names per failure class, so a verdict points at replies
                # to look at without carrying the whole trace.
                "below_floor_events": below_floor[:SHOWN_EVENTS],
                "wrong_length_events": wrong_length[:SHOWN_EVENTS],
                "no_codec_codes_events": no_codes[:SHOWN_EVENTS],
                "over_cap_events": [key for key, _, _ in over_cap[:SHOWN_EVENTS]],
            },
            event_diagnostics=event_diagnostics,
        )


__all__ = ["CODEC_FRAME_RATE_HZ", "TtsReportFaithfulnessVerifier", "CODEC_SAMPLES_PER_FRAME", "LOGMEL_FLOOR",
           "LOGMEL_HOP", "LOGMEL_N_FFT", "LOGMEL_N_MELS", "LOGMEL_PSNR_FLOOR_DB",
           "MAX_LENGTH_DELTA_FRAMES", "CodecFidelityVerifier", "log_mel",
           "logmel_psnr_db", "time_domain_snr_db"]
