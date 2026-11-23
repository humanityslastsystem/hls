"""Cut replies under the sampling gate.

A request the drive's scheduled stop cut ends where the harness stopped listening, not
where the server stopped talking. Everything it did serve is still evidence, so it is
graded like any reply; its ending is not, so it is out of the contract's two ending rules
and out of the end-of-turn count entirely. These tests drive the real ``verify`` body
through the fake engine of ``test_report_verifier``.
"""

from hls.core.trace import TraceEvent
from tests.unit.test_report_verifier import (_EOT, _Proxy, _draw, _event, _honest_reply,
                                             _run, _verifier, _violation)


def _cut_event(req: str, served: list[int], **kwargs) -> TraceEvent:
    """A cut reply as the driver records one: the tokens that arrived, no
    finish_reason, and the cut flag."""
    from hls.spec.workload import CUT_EXTRA

    ev = _event(req, served, finish_reason=None, **kwargs)
    return ev.model_copy(update={"extras": {CUT_EXTRA: True}})


def test_a_cut_reply_is_graded_on_the_tokens_it_streamed():
    """Every position it served is a draw the server made and the gate can read."""
    verdict = _run(_verifier(), [_cut_event("r0", _draw(200, seed=1))])
    assert verdict.passed is True
    assert verdict.diagnostics["n_violations"] == 0
    assert verdict.diagnostics["n_tokens"] == 200


def test_a_cut_reply_is_not_an_ending_the_contract_can_check():
    """It reports no finish reason and its last token is an ordinary one, which on any
    other reply is the contract violation that fails the trace. Here the harness chose
    where it stopped."""
    verdict = _run(_verifier(), [_cut_event("r0", _draw(20, seed=1))])
    assert verdict.diagnostics["violations"] == []


def test_an_uncut_reply_with_no_finish_reason_still_violates_the_contract():
    """The exemption is the cut's alone: a server that simply stops reporting how its
    replies ended must not inherit it."""
    ev = _event("r0", _draw(20, seed=1), finish_reason=None)
    assert "not one of the checkpoint's end-of-turn ids" in \
        _violation(_run(_verifier(), [ev]))


def test_a_cut_reply_contributes_nothing_to_the_end_of_turn_count():
    """It did not end, so neither the endings it served nor the prices of its
    positions are evidence about where this server ends turns. Out of both sides of
    the count: a position priced but never at risk of ending would understate the
    null."""
    whole = _event("r0", _draw(40, seed=1) + [_EOT])
    cut = _cut_event("r1", _draw(200, seed=2))

    d = _run(_verifier(), [whole, cut]).diagnostics

    assert d["n_tokens"] == 241               # the cut reply is graded
    assert d["end_of_turn"]["priced_positions"] == 40   # but priced at none of it


def test_a_cut_reply_that_happens_to_end_on_an_end_of_turn_id_is_still_not_counted():
    """The count asks where a server chose to end. A cut reply chose nothing, so a
    last token that happens to be an end-of-turn id is not an ending it made."""
    def replies(items):
        return [_honest_reply(item.first_of("text").tokens, cut=True)
                for item in items]

    cut = _cut_event("r0", _draw(40, seed=1) + [_EOT])
    d = _run(_verifier(), [cut], _Proxy(replies)).diagnostics

    assert "end_of_turn_count" not in d["rejections"]
    assert d["end_of_turn"]["observed"] == 0
    assert d["end_of_turn"]["priced_positions"] == 0


def test_a_cut_reply_that_streamed_nothing_is_not_a_violation():
    """A request cut before its first token reports nothing and is graded by nothing,
    which is not the same as a server exempting itself with an empty reply."""
    whole = _event("r0", _draw(40, seed=1) + [_EOT])
    cut = _cut_event("r1", [], tokens=None, logprobs=[])

    verdict = _run(_verifier(), [whole, cut])

    assert verdict.passed is True
    assert verdict.diagnostics["n_violations"] == 0
    assert verdict.diagnostics["n_tokens"] == 41


def test_an_uncut_empty_reply_still_violates_the_contract():
    """The other half of that exemption: an empty reply the harness did not cut is the
    artifact taking itself out of the gate."""
    ev = _event("r0", [], tokens=None, logprobs=[])
    assert "served no tokens" in _violation(_run(_verifier(), [ev]))
