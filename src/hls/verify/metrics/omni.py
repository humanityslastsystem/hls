"""Omni mixed-modality correctness metric (multimodal-omni).

The omni workload interleaves image-input (VQA) and audio-input (ASR or S2TT) requests
on one server. The two arms are scored by different measures, both in [0, 1]:

  - audio input  -> per ``audio_measure``: ``1 - WER`` vs a transcript GT (ASR) or
    smoothed sentence **BLEU** vs a translation GT (S2TT -- the FLEURS speech-translation
    setup the Qwen3-Omni report evaluates);
  - image input  -> SQuAD-style word-level **F1** between the answer and the reference
    answer (VQA answers are short; F1 is the standard partial-credit measure).

Those numbers are not comparable to each other, so ``MixedModalityMatchMetric`` gates
one arm per instance: a task declares two verifiers, one per arm, and requires both.
Averaging the arms into a single floor instead lets a strong image arm pay for an
abandoned audio one.

The modality is read from ``metadata["input_modality"]`` (stamped by MixedModalityDataset)
with a fallback to sniffing the input envelope's payloads.
"""

from __future__ import annotations

from typing import Literal

from hls.core.faults import HarnessFault
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric
from hls.verify.metrics.asr import normalize_transcript, word_error_rate


def sentence_bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Smoothed sentence-level BLEU in [0, 1], torch-free (like ``word_f1``).

    The S2TT measure (the Qwen3-Omni report scores FLEURS speech translation with
    BLEU). Exact unigram precision + add-one smoothing for n>=2 (Lin & Och), and the
    standard brevity penalty. Sentence-level (mean over events), not corpus BLEU --
    close for same-scale comparisons; documented so nobody reads the two
    interchangeably against published corpus-BLEU numbers.

    Both sides are folded through ``normalize_transcript`` first, the normaliser WER
    already uses. BLEU compares tokens, so without it a reference translation writing
    "403" and a decode saying "four hundred and three" share no n-gram and the sentence
    scores near zero for agreeing."""
    import math

    pred = normalize_transcript(prediction).split()
    ref = normalize_transcript(reference).split()
    if not pred or not ref:
        return 1.0 if (not pred and not ref) else 0.0
    log_p_sum = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams: dict[tuple, int] = {}
        for i in range(len(pred) - n + 1):
            g = tuple(pred[i:i + n])
            pred_ngrams[g] = pred_ngrams.get(g, 0) + 1
        ref_ngrams: dict[tuple, int] = {}
        for i in range(len(ref) - n + 1):
            g = tuple(ref[i:i + n])
            ref_ngrams[g] = ref_ngrams.get(g, 0) + 1
        total = max(len(pred) - n + 1, 0)
        matched = sum(min(c, ref_ngrams.get(g, 0)) for g, c in pred_ngrams.items())
        if n == 1:
            if matched == 0:
                return 0.0                       # no unigram overlap: BLEU is 0
            p_n = matched / total
        else:
            p_n = (matched + 1.0) / (total + 1.0)  # add-one smoothing (short sentences)
        log_p_sum += math.log(p_n)
    bp = 1.0 if len(pred) >= len(ref) else math.exp(1.0 - len(ref) / len(pred))
    return bp * math.exp(log_p_sum / max_n)


def word_f1(prediction: str, reference: str) -> float:
    """SQuAD-style token (word) F1 between two normalized strings, in [0, 1]."""
    pred = normalize_transcript(prediction).split()
    ref = normalize_transcript(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    ref_counts: dict[str, int] = {}
    for w in ref:
        ref_counts[w] = ref_counts.get(w, 0) + 1
    overlap = 0
    seen: dict[str, int] = {}
    for w in pred:
        seen[w] = seen.get(w, 0) + 1
        if seen[w] <= ref_counts.get(w, 0):
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 2 * precision * recall / (precision + recall)


def _text_of(env: Envelope | None) -> str:
    tp = env.first_of("text") if env is not None else None
    return (tp.text or "") if isinstance(tp, TextPayload) else ""


def strip_think(text: str) -> str:
    """Drop ``<think>...</think>`` chain-of-thought blocks before scoring.

    Thinking-variant models (e.g. Qwen3-*-Thinking) prefix every reply with CoT;
    scoring the raw text buries the short answer under hundreds of reasoning words
    (F1 ~ 0 for a correct reply). An UNTERMINATED ``<think>`` (the output cap cut
    the reply mid-thought) drops everything from the tag -- there is no answer to
    salvage. Applied uniformly, so thinking and non-thinking contestants meet the
    same bar on the answer text itself."""
    out, pos = [], 0
    while True:
        start = text.find("<think>", pos)
        if start == -1:
            out.append(text[pos:])
            break
        out.append(text[pos:start])
        end = text.find("</think>", start)
        if end == -1:
            break
        pos = end + len("</think>")
    return "".join(out).strip()


def _input_modality(event: TraceEvent) -> str:
    """image | audio -- from the stamped metadata, else sniffed off the input payloads."""
    tagged = event.input.metadata.get("input_modality") if event.input else None
    if tagged in ("image", "audio"):
        return tagged
    mods = event.input.modalities() if event.input else frozenset()
    if "audio" in mods:
        return "audio"
    return "image"        # default to the F1 scorer (text-vs-text)


def _not_this_arm(modality: str) -> PerEventVerdict:
    """The entry a per-arm metric records for an event belonging to the other arm.

    Passed with no score: the arm's gate has no opinion on it, and the runner ANDs the
    verifiers' per-event maps, so the sibling arm's gate is what decides the event."""
    return PerEventVerdict(passed=True, score=None,
                           rationale=f"{modality} event, graded by the {modality} arm",
                           diagnostics={"modality": modality, "graded": False})


class MixedModalityMatchMetric(RolloutMetric):
    """Correctness for ONE input-modality arm of the omni mix.

    ``arm="image"`` reduces over the image-input (VQA) events by word F1; ``arm="audio"``
    reduces over the audio-input events by ``audio_measure``. A task declares one
    verifier per arm and requires both, so neither arm's score can subsidize the other.

    The reduced value is the mean over this arm's events only, but ``score`` returns a
    per-event entry for every event it was handed. The runner reads each verifier's
    per-event map as an opinion on every event that verifier covers and treats a missing
    key as a failure, so dropping the other arm's events would zero their goodput
    credit. They are marked passed with no score, which leaves them to the sibling
    arm's gate.

    An arm with no events raises rather than reducing an empty mean, so a run that
    served (or kept) nothing on one modality cannot clear that modality's gate on no
    evidence."""

    #: Which arm this instance gates: "image" (VQA) or "audio" (ASR or S2TT).
    arm: Literal["image", "audio"]
    #: How the audio arm is scored: "wer" (1 - WER vs a transcript GT -- the ASR
    #: shape) or "bleu" (smoothed sentence BLEU vs a translation GT -- the S2TT shape
    #: the Qwen3-Omni report evaluates on FLEURS). Unused when ``arm="image"``.
    audio_measure: Literal["wer", "bleu"] = "wer"
    #: Per-event pass bar on this arm's own scale. It labels each event pass/fail in
    #: the verdict record; a goodput-counting scorer then credits only the passing
    #: ones, while a capacity scorer reads the run-level gate alone. Sentence-BLEU 0.3
    #: is a strong translation while word-F1 0.3 is a bad VQA answer, so each arm's
    #: verifier sets its own.
    pass_threshold: float = 0.5

    def score(self, events: list[TraceEvent]) -> MetricResult:
        """Reduce this arm's events to a mean, carrying per-event detail for all of them."""
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault(
                "MixedModalityMatchMetric needs ground-truth labels; the dataset provides none")
        mine = [ev for ev in events if _input_modality(ev) == self.arm]
        if not mine:
            raise HarnessFault(
                f"MixedModalityMatchMetric(arm={self.arm!r}) was handed {len(events)} "
                f"events and none of them are {self.arm}-input. The omni workload mixes "
                f"both modalities by construction, so an empty arm means the corpus or "
                f"the rollout lost one -- there is nothing to grade this arm on")
        per_event: dict[str, PerEventVerdict] = {}
        for ev in events:
            modality = _input_modality(ev)
            per_event[ev.request_id] = (self.score_one(ev) if modality == self.arm
                                        else _not_this_arm(modality))
        vals = [per_event[ev.request_id].score or 0.0 for ev in mine]
        return MetricResult(value=sum(vals) / len(vals), per_event=per_event)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        """Score one event on this arm's measure, against ``pass_threshold``.

        The measure follows ``arm``, not the event, so the two can never disagree;
        ``score`` only ever hands it events of its own arm."""
        pred = strip_think(_text_of(event.response.output if event.response else None))
        ref_text = str(event.ground_truth) if event.ground_truth is not None else ""
        if self.arm == "audio":
            if self.audio_measure == "bleu":
                score = sentence_bleu(pred, ref_text)
                diag = {"modality": "audio", "bleu": score}
            else:
                wer = word_error_rate(pred, ref_text)
                score = 1.0 - wer
                diag = {"modality": "audio", "wer": wer}
        else:
            score = word_f1(pred, ref_text)
            diag = {"modality": "image", "f1": score}
        diag.update({"pred": pred[:160], "ref": ref_text[:160]})
        return PerEventVerdict(passed=score >= self.pass_threshold, score=score,
                               rationale=f"{self.arm} score={score:.3f}", diagnostics=diag)
