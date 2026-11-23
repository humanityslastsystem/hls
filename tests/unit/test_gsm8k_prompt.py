"""Assertions on the frozen GSM8K 8-shot CoT prefix (tasks/multitenant_slo/prompts.py).

The exemplars are the canonical Wei et al. (2201.11903, Table 20) 8-shot chain of
thought, sourced verbatim from lm-evaluation-harness. Pins: exactly 8 Q/A blocks,
each answer ends in "The answer is N.", and the Qwen token count brackets the
prefix's 715 tokens (Qwen/Qwen3.6-27B -- the served model, add_special_tokens=False).
The band's job is detecting fetch truncation/corruption.
The token-count assertion is skipped when no local Qwen tokenizer cache is
available (offline unit runs).
"""
from __future__ import annotations

import pytest


def _prompt():
    from tasks.multitenant_slo.prompts import GSM8K_8SHOT
    return GSM8K_8SHOT


def test_exactly_eight_question_blocks():
    assert _prompt().count("Question:") == 8


def test_each_block_has_answer_and_final_answer_phrase():
    text = _prompt()
    assert text.count("Answer:") == 8
    # Every exemplar states its result in the canonical "The answer is N." form.
    assert text.count("The answer is") == 8


def test_ends_with_blank_line_separator():
    # Trailing "\n\n" lets the per-item template append "Question: ...\nAnswer:"
    # as a clean 9th block.
    assert _prompt().endswith("\n\n")


def test_token_count_in_measured_band():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    except Exception:  # no local tokenizer cache (offline) -> skip
        pytest.skip("Qwen tokenizer not available offline")
    n = len(tok(_prompt(), add_special_tokens=False)["input_ids"])
    # 715 tokens; the band brackets that (fetch-truncation/corruption detector).
    assert 650 <= n <= 780, f"8-shot token count {n} outside measured band [650, 780]"
