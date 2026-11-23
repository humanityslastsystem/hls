"""MMLU-Pro letter-match: extract the model's final stated answer letter (A-J)
and exact-match it against the gold letter. Mirrors gsm8k.py's structure
(extract-and-compare + HarnessFault guard on all-missing ground truth)."""
from __future__ import annotations

import re

from hls.core.faults import HarnessFault
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric

# "the answer is (D)" / "The answer is D" / "answer is: D" -- case-insensitive,
# parens optional. Models restate their conclusion mid-reasoning, so callers take
# the last match, not the first.
_ANSWER_IS_RE = re.compile(r"answer\s+is\s*:?\s*\(?\s*([A-J])\s*\)?", re.IGNORECASE)


def extract_letter(text: str | None) -> str | None:
    """The model's stated answer letter (A-J).

    Prefers the last "answer is (X)" match. Falls back to a bare reply (the
    entire stripped text is a single A-J letter, e.g. "D"), not a scan for any
    standalone A-J token anywhere in the text: "I" is a common English pronoun
    and would be misread as the letter I in ordinary prose (e.g. "I think
    it's option four" must fail closed, not match "I"). Returns None when
    nothing parses."""
    if not text:
        return None
    matches = _ANSWER_IS_RE.findall(text)
    if matches:
        return matches[-1].upper()
    stripped = text.strip()
    if re.fullmatch(r"[A-J]", stripped, re.IGNORECASE):
        return stripped.upper()
    return None


class MMLUProMatch(RolloutMetric):
    """1.0 when the model's stated answer letter equals the gold letter, else 0.0."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("MMLUProMatch needs ground-truth answer letters; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        pred = extract_letter((tp.text or "") if tp is not None else "")
        gold = str(event.ground_truth).strip().upper() if event.ground_truth is not None else None
        ok = pred is not None and gold is not None and pred == gold
        return PerEventVerdict(
            passed=ok, score=1.0 if ok else 0.0,
            rationale=f"pred={pred!r} gold={gold!r}",
            diagnostics={"pred": pred, "gold": gold})
