"""GSM8K exact-match, faithful to lm-eval-harness flexible-extract.

Extract the final answer (the last number in the completion, after a ``####`` marker
if one is present) and compare prediction to ground truth as NORMALIZED STRINGS -- the
same rule lm-eval's ``exact_match`` uses after its flexible-extract filter, and what
OpenAI's ``is_correct`` uses. Comparison is string, NOT numeric: matching the official
metric means ``42.0`` != ``42`` and ``007`` != ``7`` (a model that emits a
decimal-formatted or zero-padded integer is scored wrong, exactly as upstream does).
Numeric equality would over-credit against every official source.
The extraction already strips commas and (via the regex) never captures ``$`` or a
bare trailing ``.``, so the extracted string is already lm-eval-normalized.
"""
from __future__ import annotations

import re

from hls.core.faults import HarnessFault
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric

_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """The canonical final number: after the last '####' marker if present, else
    the last number in the string. Commas stripped; sign and decimals preserved."""
    if not text:
        return None
    segment = text.split("####")[-1]
    matches = _NUM_RE.findall(segment) or _NUM_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


class Gsm8kExactMatch(RolloutMetric):
    """1.0 when the prediction's final number equals the ground truth's, else 0.0."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("Gsm8kExactMatch needs ground-truth answers; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        pred = extract_final_number((tp.text or "") if tp is not None else "")
        gold = extract_final_number(str(event.ground_truth) if event.ground_truth is not None else "")
        ok = pred is not None and gold is not None and pred == gold
        return PerEventVerdict(
            passed=ok, score=1.0 if ok else 0.0,
            rationale=f"pred={pred!r} gold={gold!r}",
            diagnostics={"pred": pred, "gold": gold})
