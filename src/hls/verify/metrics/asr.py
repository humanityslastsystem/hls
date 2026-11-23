"""ASR correctness metric: word error rate.

``word_error_rate`` is the shared primitive (Levenshtein over words normalized by
``normalize_transcript``, the convention this corpus's published scores use).
``WERMetric`` is the ``RolloutMetric``: per event it scores ``1 - WER`` of the output
transcript against the dataset ground-truth transcript and reduces by mean (mean
accuracy). Pair it with ``MetricVerifier`` (absolute floor) or, for reference-relative
grading, ``ReferenceRelativeVerifier`` with a Whisper oracle (artifact mean accuracy within
``tol`` of the oracle's) -- a streaming-engine contestant need not byte-match the
oracle, only be about as accurate.
"""

from __future__ import annotations

from typing import Sequence

from hls.core.faults import HarnessFault
from hls.core.payloads import Envelope, TextPayload
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric

_std_normalizer = None


def normalize_transcript(s: str) -> str:
    """Normalize a transcript the way this corpus's published word error rates do.

    The only normalizer these measures use. WER here and the token-overlap measures in
    ``hls.verify.metrics.omni`` (word F1, BLEU) all fold both sides through it, so the
    arms of a mixed-modality gate are on one scale; a second one would put them on two.

    Speech corpora write numbers out ("two hundred and fifty thousand dollars") while a
    modern recognizer emits them formatted ("$250,000"). Comparing those as plain tokens
    charges a correct transcript six edits, and rewards a server for spelling numbers
    back out, which is a scoring artifact and not a transcription difference. Whisper's
    EnglishTextNormalizer folds both spellings to one form, along with contractions and
    casing, and is what open-asr-leaderboard scores with -- so this keeps these scores
    comparable to the published ones for the same corpus.

    The spelling map is empty, so British and American spellings of the same word stay
    distinct ("humour" against "humor"). Populating it needs a file from the tokenizer
    repo, which the scoring path cannot count on reaching.
    """
    global _std_normalizer
    if _std_normalizer is None:
        # Imported here rather than at module scope: transformers is a heavyweight
        # import and most callers of this module never score a transcript.
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        _std_normalizer = EnglishTextNormalizer({})
    return _std_normalizer(s or "")


def _levenshtein(a: Sequence, b: Sequence) -> int:
    """Edit distance between two sequences (words)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_error_rate(prediction: str, reference: str) -> float:
    """WER = edit distance(words) / |reference words|; 0 when both empty, 1 when only
    the reference is empty-but-prediction-nonempty."""
    ref = normalize_transcript(reference).split()
    pred = normalize_transcript(prediction).split()
    if not ref:
        return 0.0 if not pred else 1.0
    return _levenshtein(pred, ref) / len(ref)


def _text_of(env: Envelope | None) -> str:
    tp = env.first_of("text") if env is not None else None
    return (tp.text or "") if isinstance(tp, TextPayload) else ""


class WERMetric(RolloutMetric):
    """Per-event ``score = 1 - WER`` of the output transcript vs the dataset
    ground-truth transcript; reduced by mean (mean accuracy).

    The score is not floored at 0. WER exceeds 1 when a reply has more words than its
    reference, so a reply that loops scores below zero and costs the mean everything it
    got wrong. The briefs state accuracy as plain ``1 - WER``.

    A per-event ``passed`` requires ``score >= pass_threshold`` (default 0.5, i.e.
    WER <= 0.5) so a near-garbage transcript sharing a word or two with the reference
    isn't credited full goodput (per-event ``passed`` feeds the scorer's goodput count).
    Tasks tune the bar via ``pass_threshold``."""

    #: Minimum per-event accuracy (1 - WER) for the event to count as passed/goodput.
    pass_threshold: float = 0.5

    def score(self, events: list[TraceEvent]) -> MetricResult:
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault(
                "WERMetric needs ground-truth transcripts; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        pred = _text_of(out)
        ref_text = str(event.ground_truth) if event.ground_truth is not None else ""
        wer = word_error_rate(pred, ref_text)
        score = 1.0 - wer
        return PerEventVerdict(
            passed=score >= self.pass_threshold,
            score=score,
            rationale=f"wer={wer:.3f}",
            diagnostics={"wer": wer, "pred": pred[:160], "ref": ref_text[:160]})
