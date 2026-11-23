"""CoQA (Reddy et al., 2019) official scoring and the metric that emits it.

The headline CoQA score is a heuristic token-level F1 between the prediction and
the human reference answer(s), after SQuAD-style normalization (lowercase, strip
punctuation, drop the articles a/an/the, collapse whitespace). When a turn has
more than one human reference, the official score is the leave-one-out average:
for each reference, exclude it and take the best F1 against the remaining ones,
then average. With a single reference it reduces to plain token F1.

The math (`coqa_f1`) is ported from the CoQA evaluation script (evaluate-v1.0.py).
`CoqaF1Metric` wraps it as a dataset-based RolloutMetric (no oracle/engine).
"""

from __future__ import annotations
import re
import string
from collections import Counter

from hls.core.faults import HarnessFault
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def normalize_answer(s: str) -> str:
    """Lowercase, remove punctuation, drop a/an/the, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES_RE.sub(" ", s)
    return " ".join(s.split())


def _get_tokens(s: str) -> list[str]:
    return normalize_answer(s).split() if s else []


def _f1(prediction: str, ground_truth: str) -> float:
    pred_toks = _get_tokens(prediction)
    gold_toks = _get_tokens(ground_truth)
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is empty, F1 is 1 iff both are empty (official behavior).
        return float(gold_toks == pred_toks)
    num_same = sum((Counter(gold_toks) & Counter(pred_toks)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def coqa_f1(prediction: str, references: list[str]) -> float:
    """Official CoQA turn F1 of `prediction` against one or more `references`.

    >1 reference: leave-one-out average (the human-comparable scoring). Exactly
    one reference: plain token F1. No references: 0.0.
    """
    if not references:
        return 0.0
    if len(references) == 1:
        return _f1(prediction, references[0])
    total = 0.0
    for i in range(len(references)):
        others = references[:i] + references[i + 1 :]
        total += max(_f1(prediction, r) for r in others)
    return total / len(references)


class CoqaF1Metric(RolloutMetric):
    """CoQA's official heuristic token-F1 (see `coqa_f1`) as a per-event metric. The
    per-event score is the turn F1; MetricVerifier reduces by mean -> the headline CoQA
    number. The ground truth is a reference answer string or a sequence of them."""

    def score(
        self, events: list[TraceEvent]
    ) -> MetricResult:  # inline the no-label raise (no LabelMetric base)
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("CoqaF1Metric needs ground-truth labels; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        pred = (tp.text or "") if tp is not None else ""
        v = event.ground_truth
        refs = [str(x) for x in v] if isinstance(v, (list, tuple)) else ([str(v)] if v is not None else [])
        f1 = coqa_f1(pred, refs)
        return PerEventVerdict(
            passed=f1 > 0.0, score=f1, rationale=f"f1={f1:.4g}", diagnostics={"f1": f1, "refs": refs}
        )
