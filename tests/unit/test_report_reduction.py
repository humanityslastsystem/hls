"""Grading and calibration reduce a served reply to the same record.

The deployed sampling gate scores a contestant's trace and ``hls calibrate`` scores a
trusted stack's captures, and the ceiling the first reads is the one the second
measured. Everything after the reference answers is the same reduction -- the facts a
reply's ending is read against, the ending rule, the position statistics and the record
they reduce to -- so a rule that reached only one side would grade a trace against a
ceiling measured under a different rule. These tests drive both entry points over the
same replies and hold their records to each other.

The reports and the reference come from ``test_report_verifier``; what the statistics
mean is pinned there and in ``test_report``.
"""

from __future__ import annotations

import asyncio

import pytest

from hls.calibrate import _captured_event, _score_report_group, _without_report
from hls.core.payloads import Envelope
from tests.unit.test_report_verifier import (
    _EOT,
    _MAX_CONTEXT,
    _Proxy,
    _acc,
    _draw,
    _event,
    _honest_reply,
    _verifier,
)


def _replies() -> list:
    """Four replies covering what the two sides could disagree about: an ordinary one,
    one that served its whole budget, one that stopped short of it with the context
    window full, and one whose ending violates the contract."""
    return [
        _event("r0", _draw(40) + [_EOT]),
        _event("r1", _draw(8) + [_EOT], finish_reason="length", max_tokens=9),
        _event("r2", _draw(8) + [_EOT], finish_reason="length"),
        _event("r3", _draw(8) + [7]),
    ]


def _reference_replies(items) -> list[dict]:
    """The reference's honest answer to whatever each item served.

    ``r2`` comes back with its context window full, which is the one exemption from the
    ending rule a reply that stopped short of its cap has -- and it is read against the
    window in ``facts``, not against the reply."""
    out = []
    for item in items:
        served = item.first_of("text").tokens
        n_ctx = _MAX_CONTEXT - len(served) if item.request_id == "r2" else 32
        out.append(_honest_reply(served, n_ctx=n_ctx))
    return out


def _grading_batch(verifier, events) -> list[tuple]:
    """The batch ``verify`` hands its scoring pass, built as ``verify`` builds it."""
    batch = []
    for ev in events:
        rows, item = verifier._rows_for(ev)
        batch.append((ev, item, rows))
    return batch


def _calibration_group(verifier, events) -> list[tuple]:
    """The same replies as ``score_report_capture`` holds them: read back from a
    capture, with the report released once the rows exist."""
    group = []
    for ev in events:
        captured = _captured_event(Envelope(request_id=ev.request_id,
                                            metadata=dict(ev.input.metadata),
                                            contents=list(ev.response.output.contents)))
        rows, item = verifier._rows_for(captured)
        group.append((_without_report(captured), item, rows))
    return group


@pytest.mark.parametrize("declared", [None, 1_000_000])
def test_the_two_paths_reduce_the_same_replies_to_the_same_records(declared):
    """One batch of replies, one reference, two callers: the records have to agree
    field for field, and so do the facts the two accumulate.

    ``declared`` is the context window the caller seeds -- ``None`` where the task
    declares none, which is where the reference's own window is read instead. It
    decides whether the reply that stopped short of its cap is exempt, so both sides
    reading the same window is part of what this pins.
    """
    verifier = _verifier()
    events = _replies()

    grading_facts = {"eot_ids": set(), "max_context": declared}
    graded = asyncio.run(verifier._score_batch(
        _acc(events, _Proxy(_reference_replies)),
        _grading_batch(verifier, events), grading_facts))

    calibration_facts = {"max_context": declared}
    calibrated = asyncio.run(_score_report_group(
        verifier, _Proxy(_reference_replies),
        _calibration_group(verifier, events), calibration_facts, "vllm", 0))

    records = [record for _key, _reason, record, _short in graded]
    assert records == list(calibrated)
    assert grading_facts == calibration_facts
    assert grading_facts["eot_ids"] == {_EOT}
    # The batch covers all four cases, so a rule that drifted on any of them bites.
    assert [r.event_key for r in records] == ["t0:r0", "t0:r1", "t0:r2", "t0:r3"]
    assert [r.at_cap for r in records] == [False, True, False, False]
    assert [r.violation is None for r in records] == [
        True, True, declared is None, False]
