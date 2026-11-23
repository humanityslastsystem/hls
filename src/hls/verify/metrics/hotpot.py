"""HotpotQA EM / token-F1 on the short answer string.

Ported from the official ``hotpot_evaluate_v1.py`` (hotpotqa/hotpot). EM and the
SQuAD-style ``normalize_answer`` are byte-identical to CoQA's, so ``normalize_answer``
is shared; but HotpotQA's F1 is NOT CoQA's -- it forces F1 to 0 when either side is a
``yes``/``no``/``noanswer`` answer that does not match exactly, and (unlike CoQA) does
not credit both-empty as 1. So F1 is implemented here, faithful to the official script,
rather than reusing ``coqa._f1``.
"""
from __future__ import annotations

from collections import Counter

from hls.core.faults import HarnessFault
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric
from hls.verify.metrics.coqa import normalize_answer

_YESNO = ("yes", "no", "noanswer")


def _hotpot_f1(pred: str, gold: str) -> float:
    """Token-F1 faithful to hotpot_evaluate_v1.py:26-46.

    Differs from CoQA's F1 in two ways the official script requires: a yes/no/noanswer
    answer that does not exactly match the other side scores 0, and there is no
    both-empty -> 1 credit (empty overlap always scores 0).
    """
    np_ = normalize_answer(pred)
    ng = normalize_answer(gold)
    if np_ in _YESNO and np_ != ng:
        return 0.0
    if ng in _YESNO and np_ != ng:
        return 0.0
    pred_toks = np_.split()
    gold_toks = ng.split()
    num_same = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


class HotpotEmF1(RolloutMetric):
    """Per-event score is token-F1 (mean = headline F1); diagnostics carry EM."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("HotpotEmF1 needs ground-truth answers; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        pred = (tp.text or "") if tp is not None else ""
        gold = str(event.ground_truth) if event.ground_truth is not None else ""
        f1 = _hotpot_f1(pred, gold)
        em = 1.0 if normalize_answer(pred) == normalize_answer(gold) else 0.0
        return PerEventVerdict(
            passed=f1 > 0.0, score=f1, rationale=f"f1={f1:.4g} em={em:.0f}",
            diagnostics={"f1": f1, "em": em, "pred": pred[:160], "gold": gold[:160]})
