"""IFEvalConstraint -- instruction-following correctness via the self-contained
checker registry. Ground truth carries {instruction_id_list, kwargs}. Per-event
pass iff EVERY instruction is followed. Unsupported instruction ids fail closed
(the event fails, with a diagnostic), never raising."""
from __future__ import annotations

import json

from hls.core.faults import HarnessFault
from hls.core.trace import TraceEvent
from hls.core.verdict import MetricResult, PerEventVerdict
from hls.verify.base import RolloutMetric
from hls.verify.metrics.ifeval.registry import INSTRUCTION_DICT


def _spec(gt) -> dict:
    if isinstance(gt, str):
        gt = json.loads(gt)
    if not isinstance(gt, dict):
        raise HarnessFault(
            f"IFEvalConstraint ground truth must be a dict or JSON object, got {type(gt).__name__}")
    return gt


class IFEvalConstraint(RolloutMetric):
    """Pass iff the response follows every instruction in ground_truth's
    instruction_id_list (using that instruction's kwargs). Unsupported ids fail closed."""

    def score(self, events: list[TraceEvent]) -> MetricResult:
        if all(ev.ground_truth is None for ev in events):
            raise HarnessFault("IFEvalConstraint needs the constraint spec; the dataset provides none")
        return super().score(events)

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        response = (tp.text or "") if tp is not None else ""
        spec = _spec(event.ground_truth)
        ids = list(spec.get("instruction_id_list", []))
        kwargs_list = list(spec.get("kwargs", [{}] * len(ids)))
        results: list[bool] = []
        unsupported: list[str] = []
        for i, iid in enumerate(ids):
            kw = kwargs_list[i] if i < len(kwargs_list) else {}
            fn = INSTRUCTION_DICT.get(iid)
            if fn is None:
                unsupported.append(iid)
                results.append(False)            # fail closed, do not raise
                continue
            clean = {k: v for k, v in (kw or {}).items() if v is not None}
            try:
                results.append(bool(response.strip()) and bool(fn(response, **clean)))
            except Exception:                     # a malformed kwarg fails the instruction, not the run
                results.append(False)
        passed = bool(results) and all(results)
        return PerEventVerdict(
            passed=passed, score=1.0 if passed else 0.0,
            rationale=f"followed {sum(results)}/{len(results)} instructions"
                      + (f"; unsupported={unsupported}" if unsupported else ""),
            diagnostics={"instruction_id_list": ids, "followed": results,
                         "unsupported": unsupported})
