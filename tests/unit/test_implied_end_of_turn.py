"""The implied end-of-turn rule (``ReportFaithfulnessVerifier.implied_end_of_turn``).

A normal stop whose last served token is not an end-of-turn id is a contract violation
by default. Under the rule it is scored as if the server had served one more token, the
end-of-turn id the reference rates highest at the next position. That position counts in
the end-of-turn count as a served ending at the reference's price there, and in nothing
that reads the report. These tests drive the real ``verify`` body through the fake engine
of ``test_report_verifier``, then the two engines' stop rows and the power of the count
against truncation at the size of a real trace.
"""

import math

import numpy as np
import pytest

from hls.verify.calibration import CalibrationError
from hls.verify.report import (poisson_binomial_upper_tail, position_stats,
                               reply_uniforms, score_reply)
from tests.unit.calibration_fixtures import end_of_turn_check, make_report_calibration
from tests.unit.test_report_cut_replies import _cut_event
from tests.unit.test_report_verifier import (_EOT, _IDS, _Proxy, _draw, _event,
                                             _honest_reply, _run, _verifier)

#: A tool-call close, standing for the last token of a reply that stopped one token short
#: of its end-of-turn id. Any id the fixture report can serve that is not an end-of-turn
#: id will do.
_TOOL_CLOSE = _IDS[2]


def _with_stop_rows(ending: float, *, eot_ids=(_EOT,), per_id=None, shift=0.0):
    """Reference replies that agree with the report, plus a stop row on every item that
    asks for one.

    ``ending`` is the reference's probability of ending the turn at the position after
    the last served token, split evenly over ``eot_ids`` unless ``per_id`` gives each
    id's own probability."""
    probs = list(per_id) if per_id is not None else [ending / len(eot_ids)] * len(eot_ids)

    def replies(items):
        out = []
        for item in items:
            r = _honest_reply(item.first_of("text").tokens, shift=shift)
            r["eot_ids"] = list(eot_ids)
            if item.metadata.get("stop_row"):
                lps = [math.log(p) for p in probs]
                r["p_stop_lp"] = lps
                r["p_stop_eot_lp"] = float(np.logaddexp.reduce(lps))
            out.append(r)
        return out

    return replies


def _short_stops(n: int, *, first_seed: int = 10) -> list:
    """Replies that close a tool call and report a normal stop without serving the
    end-of-turn id that follows it."""
    return [_event(f"s{i}", _draw(100, seed=first_seed + i) + [_TOOL_CLOSE])
            for i in range(n)]


def _whole(n: int) -> list:
    """Replies that end on the end-of-turn id where the reference wants to end."""
    return [_event(f"r{i}", _draw(100, seed=i) + [_EOT]) for i in range(n)]


# --- test 1: a stop one token short passes under the rule, and only under it -----------


def test_a_stop_one_token_short_of_its_end_of_turn_passes_and_is_counted():
    events = _whole(3) + _short_stops(3)
    proxy = _Proxy(_with_stop_rows(0.999))
    verdict = _run(_verifier(implied_end_of_turn=True), events, proxy)
    d = verdict.diagnostics
    assert verdict.passed is True, d["rejections"]
    assert d["n_implied_eot"] == 3
    assert d["n_violations"] == 0
    for i in range(3):
        rec = verdict.event_diagnostics[f"t0:s{i}"]
        assert rec["implied_eot"] == _EOT
        assert rec["ends_with_eot"] is False
        # The implied ending sits one past the last served position.
        assert rec["eot_positions"] == [rec["n_tokens"]]
    for i in range(3):
        assert "implied_eot" not in verdict.event_diagnostics[f"t0:r{i}"]
    # Priced at 0.999, far above eps, so the endings are not evidence of anything.
    assert d["end_of_turn"]["observed"] == 0
    # The stop row is asked for on every normal stop, since whether the last token is
    # an end-of-turn id is known only once the reference names the ids.
    asked = [it.metadata.get("stop_row") for batch in proxy.batches for it in batch]
    assert asked == [True] * 6


def test_the_same_stops_fail_with_the_rule_off():
    events = _whole(3) + _short_stops(3)
    proxy = _Proxy(_with_stop_rows(0.999))
    verdict = _run(_verifier(), events, proxy)
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["rejections"] == ["contract_violations"]
    assert d["n_violations"] == 3
    assert "is not one of the checkpoint's end-of-turn ids" in d["violations"][0]["reason"]
    assert "n_implied_eot" not in d
    assert all("stop_row" not in it.metadata for batch in proxy.batches for it in batch)


# --- test 2: truncation is caught by the end-of-turn count ----------------------------


def test_truncated_replies_fail_through_the_end_of_turn_count():
    """Cut mid-text where the reference gives ending about one in a million: each
    implied ending is a priced position the server ended at, against an expected count
    near zero. Nothing else fires, since every token served is a faithful draw."""
    events = _whole(3) + _short_stops(3)
    verdict = _run(_verifier(implied_end_of_turn=True), events,
                   _Proxy(_with_stop_rows(1e-6)))
    d = verdict.diagnostics
    assert verdict.passed is False
    assert d["rejections"] == ["end_of_turn_count"]
    assert d["n_implied_eot"] == 3
    eot = d["end_of_turn"]
    assert eot["observed"] == 3
    assert eot["p_value"] <= d["alpha"]
    # Every served position and the three implied ones are priced.
    assert eot["priced_positions"] == d["n_tokens"] - 3 + 3


def test_the_implied_position_is_priced_by_the_mass_of_every_end_of_turn_id():
    """Check 4 prices a position by the reference's mass on all the end-of-turn ids,
    so three ids at 0.04 each are a price of 0.12, above eps, even though the one the
    rule picks is at 0.04 on its own."""
    ids = (_EOT, 11, 12)
    cal = make_report_calibration(end_of_turn=end_of_turn_check(["vllm"]).model_copy(
        update={"ids": list(ids)}))
    verdict = _run(_verifier(cal, implied_end_of_turn=True), _short_stops(3),
                   _Proxy(_with_stop_rows(0.12, eot_ids=ids)))
    eot = verdict.diagnostics["end_of_turn"]
    assert eot["eps"] == 0.1
    assert eot["observed"] == 0
    assert eot["priced_positions"] == verdict.diagnostics["n_tokens"]


def test_the_power_of_the_count_at_the_size_of_a_graded_trace():
    """How many truncated replies the count needs to fire at the verifier's alpha, on a
    trace the size of agentic_coding's calibration serving: 2,525,539 priced positions
    and an expected 2.114 endings among them (``calibration.json``, the sglang member).

    The critical count at that null is 7, so the answer depends on how many priced
    endings the rest of the trace already served. From the trusted stack's own 3, four
    truncations fire and three do not; from none, it takes seven."""
    n, lam, alpha = 2_525_539, 2.1143698904289034, 0.01
    background = np.full(n, lam / n)
    trunc = 1e-4                            # well below eps, where the cut landed

    def p_value(baseline: int, k: int) -> float:
        return poisson_binomial_upper_tail(
            np.concatenate([background, np.full(k, trunc)]), baseline + k)

    assert p_value(3, 4) <= alpha < p_value(3, 3)
    assert p_value(0, 7) <= alpha < p_value(0, 6)
    critical = next(c for c in range(20)
                    if poisson_binomial_upper_tail(background, c) <= alpha)
    assert critical == 7


# --- test 3: the implied token is the argmax id and enters no report statistic -------


def test_the_implied_token_is_the_end_of_turn_id_the_reference_rates_highest():
    ids = (_EOT, 11, 12)
    cal = make_report_calibration(end_of_turn=end_of_turn_check(["vllm"]).model_copy(
        update={"ids": list(ids)}))
    verdict = _run(_verifier(cal, implied_end_of_turn=True), _short_stops(1),
                   _Proxy(_with_stop_rows(0, eot_ids=ids, per_id=(0.2, 0.7, 0.05))))
    assert verdict.passed is True, verdict.diagnostics["rejections"]
    assert verdict.event_diagnostics["t0:s0"]["implied_eot"] == 11


def test_the_implied_token_enters_no_report_statistic():
    """The same served tokens scored twice: once with an implied ending, once cut, which
    grades the tokens and reads no ending. Every statistic over the report agrees; only
    the end-of-turn count sees the implied position."""
    served = {f"r{i}": _draw(300, seed=4 + i) + [_TOOL_CLOSE] for i in range(2)}
    replies = _with_stop_rows(1e-6, shift=0.3)
    implied = _run(_verifier(implied_end_of_turn=True),
                   [_event(k, t) for k, t in served.items()], _Proxy(replies)).diagnostics
    cut = _run(_verifier(implied_end_of_turn=True),
               [_cut_event(k, t) for k, t in served.items()], _Proxy(replies)).diagnostics
    for key in ("n_tokens", "mean_kl", "se_kl", "cvm_omega2", "cvm_p", "z", "n_moment",
                "n_served_outside", "outside_rate"):
        assert implied[key] == cut[key], key
    n = sum(len(t) for t in served.values())
    assert implied["n_tokens"] == n and implied["mean_kl"] > 0
    assert implied["n_implied_eot"] == 2 and cut["n_implied_eot"] == 0
    assert implied["end_of_turn"]["priced_positions"] == n + 2
    assert implied["end_of_turn"]["observed"] == 2
    assert cut["end_of_turn"]["priced_positions"] == 0


def test_the_record_carries_the_implied_position_only_in_its_end_of_turn_fields():
    """The reduction itself: the report arrays stay one per served token, and the
    end-of-turn arrays gain the implied position."""
    served = _draw(20, seed=2) + [_TOOL_CLOSE]
    from hls.verify.report import reply_report
    from tests.unit.test_report_verifier import _report_entry
    from hls.core.payloads import TextPayload

    tp = TextPayload(text="x", tokens=served, logprobs=[_report_entry(t) for t in served])
    rows = reply_report(tp, served)
    ref = _honest_reply(served)
    stats = position_stats(served=served, q_ids=rows.q_ids, q_lp=rows.q_lp,
                           q_emit=rows.q_emit, p_on_q=ref["p_on_q"],
                           p_tail_lp=ref["p_tail_lp"], p_eot_lp=ref["p_eot_lp"],
                           eot_ids={_EOT}, v=reply_uniforms("t0:r0", len(served)))
    plain = score_reply(stats, event_key="t0:r0", n_served=len(served),
                        finish_reason="stop", ends_with_eot=False, at_cap=False)
    implied = score_reply(stats, event_key="t0:r0", n_served=len(served),
                          finish_reason="stop", ends_with_eot=False, at_cap=False,
                          implied_eot=(_EOT, math.log(0.5)))
    assert implied.u == plain.u and implied.kl_sum == plain.kl_sum
    assert implied.n_tokens == plain.n_tokens == len(served)
    assert implied.p_eot_lp == plain.p_eot_lp + (math.log(0.5),)
    assert implied.eot_positions == plain.eot_positions + (len(served),)
    assert implied.implied_eot == _EOT and plain.implied_eot is None
    assert "implied_eot" not in plain.to_json()
    assert type(implied).from_json(implied.to_json()) == implied
    with pytest.raises(ValueError, match="did not end"):
        score_reply(stats, event_key="t0:r0", n_served=len(served), finish_reason=None,
                    ends_with_eot=False, at_cap=False, count_end_of_turn=False,
                    implied_eot=(_EOT, 0.0))
    # A price that is not one is refused where it is scored, naming the reply.
    for bad in (math.nan, math.inf):
        with pytest.raises(ValueError, match="'t0:r0': the reference priced the implied end of turn"):
            score_reply(stats, event_key="t0:r0", n_served=len(served), finish_reason="stop",
                        ends_with_eot=False, at_cap=False, implied_eot=(_EOT, bad))
    assert score_reply(stats, event_key="t0:r0", n_served=len(served), finish_reason="stop",
                       ends_with_eot=False, at_cap=False,
                       implied_eot=(_EOT, -math.inf)).p_eot_lp[-1] == -math.inf


# --- test 4: what the rule leaves alone ----------------------------------------------


def test_a_stop_that_served_its_end_of_turn_implies_nothing():
    verdict = _run(_verifier(implied_end_of_turn=True), _whole(2),
                   _Proxy(_with_stop_rows(1e-6)))
    assert verdict.passed is True
    assert verdict.diagnostics["n_implied_eot"] == 0
    assert all("implied_eot" not in r for r in verdict.event_diagnostics.values())


def test_a_length_stop_short_of_its_cap_still_violates():
    ev = _event("r0", _draw(20), finish_reason="length", max_tokens=8192)
    proxy = _Proxy(_with_stop_rows(0.999))
    verdict = _run(_verifier(implied_end_of_turn=True), [ev], proxy)
    assert "finish_reason='length' after 20 of 8192 tokens" in \
        verdict.diagnostics["violations"][0]["reason"]
    assert "stop_row" not in proxy.batches[0][0].metadata


def test_a_length_stop_at_its_cap_implies_nothing():
    served = _draw(20)
    verdict = _run(_verifier(implied_end_of_turn=True),
                   [_event("r0", served, finish_reason="length", max_tokens=len(served))],
                   _Proxy(_with_stop_rows(1e-6)))
    assert verdict.passed is True
    assert verdict.diagnostics["n_implied_eot"] == 0
    assert verdict.diagnostics["end_of_turn"]["observed"] == 0


def test_a_cut_reply_implies_nothing_and_asks_for_no_stop_row():
    proxy = _Proxy(_with_stop_rows(1e-6))
    verdict = _run(_verifier(implied_end_of_turn=True),
                   [_cut_event("r0", _draw(40, seed=1))], proxy)
    assert verdict.passed is True
    assert verdict.diagnostics["n_implied_eot"] == 0
    assert "stop_row" not in proxy.batches[0][0].metadata
    assert verdict.event_diagnostics["t0:r0"]["eot_positions"] == []


def test_an_empty_reply_still_violates():
    ev = _event("r0", [], tokens=[], logprobs=[])
    verdict = _run(_verifier(implied_end_of_turn=True), [ev], _Proxy(_with_stop_rows(1)))
    assert verdict.diagnostics["violations"][0]["reason"] == "the reply served no tokens"


def test_an_uncut_reply_with_no_finish_reason_still_violates():
    """The rule completes a claimed normal stop and nothing else."""
    ev = _event("r0", _draw(20, seed=1) + [_TOOL_CLOSE], finish_reason=None)
    verdict = _run(_verifier(implied_end_of_turn=True), [ev],
                   _Proxy(_with_stop_rows(0.999)))
    assert "reports finish_reason=None" in verdict.diagnostics["violations"][0]["reason"]


def test_a_reference_that_returns_no_stop_row_is_a_harness_error():
    """The item asked for the row, so its absence is the engine failing the request,
    not the server; it raises rather than grading the reply either way."""
    def replies(items):
        return [_honest_reply(item.first_of("text").tokens) for item in items]

    with pytest.raises(RuntimeError, match="returned no stop row"):
        _run(_verifier(implied_end_of_turn=True), _short_stops(1), _Proxy(replies))


def test_the_rule_still_refuses_a_reference_with_other_end_of_turn_ids():
    with pytest.raises(CalibrationError, match="end-of-turn ids"):
        _run(_verifier(implied_end_of_turn=True), _short_stops(1),
             _Proxy(_with_stop_rows(0.999, eot_ids=(_EOT, 77))))


# --- the engines' stop rows ------------------------------------------------------------


def test_the_hf_engine_scores_the_stop_row_off_the_row_past_the_reply():
    torch = pytest.importorskip("torch")
    from hls.core.decode import Sample
    from tests.unit.test_references import _stub_engine

    rows = [[0.0] * 16 for _ in range(3)]
    rows[0][1] = 2.0
    rows[1][2] = 1.0
    rows[2][9] = 3.0                        # the model wants to end after the reply
    rows[2][11] = 1.0
    eng = _stub_engine(rows, eos_ids=(9, 11), decode=Sample(temperature=1.0))
    item = ([{"role": "user", "content": "q"}], None, False, [1, 2], [[1, 3], [2, 5]])
    plain, stopped = eng._reference_logprobs_prepared(
        [(*item, False), (*item, True)], 1.0)
    assert "p_stop_lp" not in plain
    for key in ("p_on_q", "p_tail_lp", "p_eot_lp", "n_ctx"):
        assert stopped[key] == plain[key], key
    lp = torch.log_softmax(torch.tensor(rows, dtype=torch.float32), dim=-1).double()
    assert stopped["p_stop_lp"] == pytest.approx([lp[2, 9].item(), lp[2, 11].item()],
                                                 abs=1e-6)
    assert stopped["p_stop_eot_lp"] == pytest.approx(
        torch.logsumexp(lp[2, [9, 11]], dim=0).item(), abs=1e-6)


@pytest.fixture
def fresh_capture(monkeypatch):
    """A capture slot of the test's own, as ``test_vllm_report`` gives each of its."""
    pytest.importorskip("torch")
    from hls.reference import vllm_report
    monkeypatch.setattr(vllm_report, "CAPTURE", vllm_report.ReportCapture())


def test_the_vllm_capture_reduces_the_prompts_last_row_as_the_stop_row(fresh_capture):
    """The row that predicts the first generated token is the stop row. It is reduced
    only when asked for, through the same head, and the reply's rows are unchanged."""
    torch = pytest.importorskip("torch")
    from hls.reference import vllm_report
    from hls.reference.vllm_report import REPORT_DECIMALS
    from tests.unit.test_vllm_report import VOCAB, armed_runner, run_step

    runner = armed_runner(4)
    vllm_report.CAPTURE.arm([[1]] * 4, [0, 2], True)
    assert run_step(runner, prompt_len=100, computed=0, scheduled=100) == [
        95, 96, 97, 98, 99]
    got = vllm_report.CAPTURE.take()
    assert len(got["p_tail_lp"]) == 4
    lp = torch.log_softmax(-torch.arange(VOCAB, dtype=torch.float32), dim=-1)
    assert got["p_stop_lp"] == [round(lp[0].item(), REPORT_DECIMALS),
                                round(lp[2].item(), REPORT_DECIMALS)]
    assert got["p_stop_eot_lp"] == round(torch.logsumexp(lp[[0, 2]], dim=0).item(),
                                         REPORT_DECIMALS)


def test_the_vllm_stop_row_arrives_with_the_prefills_last_chunk(fresh_capture):
    from hls.reference import vllm_report
    from tests.unit.test_vllm_report import armed_runner, run_step

    runner = armed_runner(4)
    vllm_report.CAPTURE.arm([[1]] * 4, [0], True)
    assert run_step(runner, prompt_len=100, computed=0, scheduled=97) == [95, 96]
    assert run_step(runner, prompt_len=100, computed=97, scheduled=3) == [97, 98, 99]
    got = vllm_report.CAPTURE.take()
    assert len(got["p_tail_lp"]) == 4 and len(got["p_stop_lp"]) == 1


def test_the_vllm_engine_arms_the_stop_row_and_refuses_a_capture_without_it(monkeypatch):
    from tests.unit.test_vllm_reference import _capture_rows, _report_engine, _ReportLLM

    stopped = {**_capture_rows(2), "p_stop_lp": [-0.1], "p_stop_eot_lp": -0.1}
    llm = _ReportLLM(rows=stopped)
    eng = _report_engine(monkeypatch, llm)
    assert eng._score_one([1, 2, 3], [4, 5], [[4], [5]], [0], True) == stopped
    assert llm.calls[0] == ("hls_arm_report", ([[4], [5]], [0], True))
    # The same prompt either way: the stop row is a row vLLM already computes.
    assert llm.prompt == [1, 2, 3, 4, 5]
    lost = _ReportLLM(rows=_capture_rows(2))
    eng = _report_engine(monkeypatch, lost)
    with pytest.raises(RuntimeError, match="reduced none"):
        eng._score_one([1], [4, 5], [[4], [5]], [0], True)
