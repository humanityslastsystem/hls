"""BFCLAstMatch -- BFCL v3 AST-category grader (pure Python, no execution).
Logic adapted from gorilla's bfcl_eval ast_checker (Apache-2.0):
parse the model's "[f(a=1), g(b='x')]" reply with ast.parse, then match the
call multiset against possible_answer allowed-value lists."""
from __future__ import annotations

import ast

from hls.core.faults import HarnessFault
from hls.core.trace import TraceEvent
from hls.core.verdict import PerEventVerdict
from hls.verify.base import RolloutMetric


def _dotted(node) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    raise ValueError("unsupported callable expression")


def parse_calls(text: str) -> list[tuple[str, dict]] | None:
    """'[f(a=1), m.g(b="x")]' -> [("f", {"a": 1}), ("m.g", {"b": "x"})]; None if unparseable."""
    text = text.strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return None
    try:
        tree = ast.parse(text[start:end + 1], mode="eval")
        if not isinstance(tree.body, ast.List):
            return None
        calls = []
        for el in tree.body.elts:
            if not isinstance(el, ast.Call):
                return None
            kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in el.keywords
                      if kw.arg is not None}
            calls.append((_dotted(el.func), kwargs))
        return calls
    except (SyntaxError, ValueError):
        return None


def _norm(v):
    return " ".join(str(v).lower().split()) if isinstance(v, str) else v


def _value_ok(got, allowed: list) -> bool:
    """Whether ``got`` matches any entry of a gold ``possible_answer`` value list.

    ``""`` is a normal entry here, matched by an empty-string emission. BFCL's ``""``
    sentinel says the parameter may be omitted or supplied as the empty/default value;
    omission is the other half, handled by ``_call_ok``. Skipping ``""`` in this
    comparison would score a reply that emits the permitted empty value (``unit=''``
    against gold ``["", "celsius"]``) as wrong."""
    for a in allowed:
        if isinstance(a, list) and isinstance(got, list):
            if len(a) == len(got) and all(_value_ok(g, [x]) for g, x in zip(got, a)):
                return True
        elif _norm(got) == _norm(a):
            return True
        elif isinstance(a, (int, float)) and isinstance(got, (int, float)) and float(a) == float(got):
            return True
    return False


def _call_ok(name: str, kwargs: dict, gt_call: dict) -> bool:
    """Whether one emitted call matches one gold call: same name, every gold parameter
    either present with an allowed value or absent-and-optional, and no parameter the
    gold call does not list. ``""`` among a parameter's allowed values is BFCL's
    optionality marker -- it permits omitting the parameter, and (via ``_value_ok``)
    passing the empty value."""
    (gt_name, params), = gt_call.items()
    if name != gt_name:
        return False
    for p, allowed in params.items():
        optional = "" in allowed
        if p not in kwargs:
            if not optional:
                return False
        elif not _value_ok(kwargs[p], allowed):
            return False
    return all(k in params for k in kwargs)   # no hallucinated params


class BFCLAstMatch(RolloutMetric):
    """Per-event 1.0/0.0: the model's parsed call list must multiset-match the
    gold ``possible_answer`` calls (order-insensitive), each call's name exact
    and every parameter matching one of its allowed values -- where a parameter
    whose allowed list contains ``""`` may equally be omitted."""

    def score_one(self, event: TraceEvent) -> PerEventVerdict:
        gt = event.ground_truth
        if not isinstance(gt, dict) or "ground_truth" not in gt:
            raise HarnessFault("BFCLAstMatch needs ground truth {'ground_truth', 'functions'}")
        out = event.response.output if event.response else None
        tp = out.first_of("text") if out is not None else None
        calls = parse_calls((tp.text or "") if tp is not None else "")
        gt_calls = list(gt["ground_truth"])
        if calls is None or len(calls) != len(gt_calls):
            return PerEventVerdict(passed=False, score=0.0,
                                   diagnostics={"parse_failed": calls is None})
        remaining = list(gt_calls)            # order-insensitive greedy matching
        for name, kwargs in calls:
            m = next((g for g in remaining if _call_ok(name, kwargs, g)), None)
            if m is None:
                return PerEventVerdict(passed=False, score=0.0, diagnostics={"unmatched_call": name})
            remaining.remove(m)
        return PerEventVerdict(passed=True, score=1.0, diagnostics={})
