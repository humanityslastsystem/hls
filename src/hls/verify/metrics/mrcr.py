"""MRCRRatio -- openai/mrcr's official grader: the response must begin with the
per-item random prepend string; the score is difflib.SequenceMatcher.ratio()
between response and gold answer, both with the prepend stripped."""
from __future__ import annotations

from difflib import SequenceMatcher

from hls.core.faults import HarnessFault
from hls.core.trace import TraceEvent
from hls.core.verdict import PerEventVerdict
from hls.verify.base import RolloutMetric


class MRCRRatio(RolloutMetric):
    """Per-event score in [0, 1]: 0.0 when the response doesn't start with the
    per-item random prepend string, else difflib.SequenceMatcher.ratio() between
    response and gold answer, both with the prepend stripped."""

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        gt = event.ground_truth
        if not isinstance(gt, dict) or "answer" not in gt or "prepend" not in gt:
            raise HarnessFault("MRCRRatio needs ground truth {'answer', 'prepend'}")
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        response = (tp.text or "") if tp is not None else ""
        prepend, answer = str(gt["prepend"]), str(gt["answer"])
        if not response.startswith(prepend):
            return PerEventVerdict(passed=False, score=0.0, diagnostics={"prepend_missing": True})
        ratio = SequenceMatcher(
            None, response.removeprefix(prepend), answer.removeprefix(prepend)).ratio()
        return PerEventVerdict(passed=ratio > 0.0, score=float(ratio), diagnostics={"ratio": ratio})
