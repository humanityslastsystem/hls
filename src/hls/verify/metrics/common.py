"""Generic, benchmark-agnostic correctness metrics.

These are reusable primitives -- answer-letter matching, substring matching --
usable across many datasets. Benchmark-specific metrics (e.g. CoQA) live in their
own sibling module; all are re-exported from ``hls.verify.metrics``.
"""

from __future__ import annotations

import re

from hls.core.faults import HarnessFault
from hls.core.payloads import Envelope
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric

_LETTER_RE = re.compile(r"\b([A-D])\b")


def _letter(env: Envelope | None) -> str | None:
    if env is None:
        return None
    tp = env.first_of("text")
    text = (tp.text or "") if tp is not None else ""
    m = _LETTER_RE.search(text.strip().upper())
    return m.group(1) if m else None


class LetterMatchMetric(RolloutMetric):
    """Extract the first A-D letter and compare to the dataset label."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("LetterMatchMetric needs ground-truth labels; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        pred = _letter(out)
        v = event.ground_truth
        expected = str(v).strip().upper()[:1] if v is not None else None
        ok = pred is not None and pred == expected
        return PerEventVerdict(
            passed=ok,
            score=1.0 if ok else 0.0,
            rationale=f"predicted={pred!r}, expected={expected!r}",
            diagnostics={"predicted": pred, "expected": expected},
        )


class ContainsMatchMetric(RolloutMetric):
    """Pass when the dataset label string appears (case-insensitive) in the output text."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("ContainsMatchMetric needs ground-truth labels; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        text = (tp.text or "") if tp is not None else ""
        v = event.ground_truth
        needle = str(v).strip().lower() if v is not None else ""
        ok = needle != "" and needle in text.lower()
        return PerEventVerdict(
            passed=ok,
            score=1.0 if ok else 0.0,
            rationale=f"needle={needle!r} present={ok}",
            diagnostics={"needle": needle, "got": text[:160]},
        )
