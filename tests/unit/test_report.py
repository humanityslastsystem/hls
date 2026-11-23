"""The sampling gate's measurement: the partition, the statistics, the records.

The reduction is shared by grading and by calibration, so these tests pin the numbers
themselves rather than a verdict: a hand case per branch of the partition rule, hand
cases for each statistic, the two exact nulls checked by simulation, the end-of-turn
count's tail checked against brute-force enumeration, and -- where torch is installed --
agreement with a torch reference implementation.
"""

import itertools
import math
import warnings

import numpy as np
import pytest

from hls.verify import report

from hls.core.payloads import TextPayload, TokenLogprob, TopLogprob
from hls.verify.report import (
    EOT_EPS_GRID,
    MASS_TOL,
    PAD_ID,
    TOP_LOGPROBS_KEPT,
    TOP_LOGPROBS_REQUESTED,
    PooledReport,
    ReplyReport,
    ReportRows,
    cramer_von_mises_uniform,
    end_of_turn_count,
    kept_width,
    partition_row,
    poisson_binomial_upper_tail,
    position_stats,
    reply_report,
    reply_uniforms,
    score_reply,
)


def _entry(served: int, alts: list[tuple[int, float]], emitted: float | None = None
           ) -> TokenLogprob:
    """One reported position: the served token's id and log-probability, and the
    alternatives the server reported as (id, logprob) pairs."""
    lp = emitted if emitted is not None else dict(alts).get(served, -20.0)
    return TokenLogprob(
        token="t", token_id=served, logprob=lp,
        top_logprobs=[TopLogprob(token=f"a{i}", token_id=i_, logprob=v)
                      for i, (i_, v) in enumerate(alts)])


def _alts(n: int, start: int = 0, step: float = -0.5) -> list[tuple[int, float]]:
    """``n`` distinct alternatives with strictly decreasing log-probabilities.

    The partition rule reads the order of these values and never their mass, so the
    default step is chosen for readable rank tests rather than for a report that adds
    up; a test that goes through ``reply_report`` wants ``_dist_alts``."""
    return [(start + i, step * (i + 1)) for i in range(n)]


def _dist_alts(n: int, start: int = 0) -> list[tuple[int, float]]:
    """``n`` distinct alternatives whose probabilities sum to well under one, which is
    what a reply's report has to do at every position."""
    return _alts(n, start=start, step=-2.0)


# --- the partition -------------------------------------------------------------


def test_partition_keeps_the_entries_above_the_threshold_rank():
    """21 distinct alternatives, all distinct values: the atoms are the 19 strictly
    above the twentieth largest, which is what the padding width is sized for."""
    alts = _alts(TOP_LOGPROBS_REQUESTED)
    ids, lps, n_distinct = partition_row(_entry(0, alts), served=0)
    assert n_distinct == TOP_LOGPROBS_REQUESTED
    assert len(ids) == len(lps) == TOP_LOGPROBS_KEPT
    assert ids == [i for i, _ in alts[:TOP_LOGPROBS_KEPT]]
    assert PAD_ID not in ids


def test_partition_drops_the_served_token_vllm_appended():
    """vLLM sends the served token followed by its top-k. When the served token ranks
    below the requested width it arrives as the strictly smallest entry, and dropping it
    is what makes the partition the same on either serving stack."""
    alts = _alts(TOP_LOGPROBS_REQUESTED - 1, start=1) + [(999, -50.0)]
    ids, lps, n_distinct = partition_row(_entry(999, alts, emitted=-50.0), served=999)
    assert n_distinct == TOP_LOGPROBS_REQUESTED
    assert 999 not in ids
    # With the appended entry gone exactly 20 remain, so the threshold is the smallest
    # of them and the atoms are the 19 above it.
    assert len(ids) == TOP_LOGPROBS_KEPT
    assert lps == sorted(lps, reverse=True)


def test_partition_keeps_a_served_token_that_is_not_the_smallest():
    """SGLang reports its true top-k, so the served token's entry is a real one; it is
    dropped only when it sits strictly below every other entry."""
    alts = _alts(TOP_LOGPROBS_REQUESTED, start=1)
    served = alts[3][0]
    ids, _, _ = partition_row(_entry(served, alts), served=served)
    assert served in ids


def test_partition_tails_a_tie_group_at_the_threshold_whole():
    """A tie at the threshold value is tailed entirely: keeping part of it would make
    the partition depend on which member the wire happened to list first, and a report
    built around the served token can move that."""
    alts = _alts(17, start=1) + [(100, -9.0), (101, -9.0), (102, -9.0), (103, -9.0)]
    ids, lps, _ = partition_row(_entry(1, alts), served=1)
    # The twentieth largest value is -9.0, so all four tied entries are outside.
    assert {100, 101, 102, 103}.isdisjoint(ids)
    assert ids[:17] == [i for i, _ in alts[:17]]
    assert ids[17:] == [PAD_ID, PAD_ID]
    assert lps[17:] == [float("-inf"), float("-inf")]


def test_partition_orders_ties_above_the_threshold_by_id():
    """Above the threshold, order is log-probability descending then id ascending, so
    two engines listing a tie group differently produce the same row."""
    alts = [(7, -1.0), (3, -1.0)] + _alts(19, start=10, step=-2.0)
    ids, _, _ = partition_row(_entry(3, alts), served=3)
    assert ids[:2] == [3, 7]


def test_partition_refuses_too_few_distinct_alternatives():
    """Twenty distinct values are what the threshold rank needs; below that there is no
    draw-independent threshold to cut at, so the position is a contract violation."""
    reason = partition_row(_entry(0, _alts(19)), served=0)
    assert isinstance(reason, str)
    assert "19 usable alternatives (19 distinct as reported)" in reason
    assert "fewer than the 20 the partition needs" in reason


def test_the_refusal_counts_what_is_left_after_the_served_entry_is_dropped():
    """The message has to name the usable count, not the reported one.

    Nineteen alternatives plus the served token strictly below all of them is vLLM's
    appended entry: twenty distinct ids arrive, the appended one is deleted, and
    nineteen are left. Quoting only what arrived would read as "carries 20 distinct
    alternatives, fewer than the 20 the partition needs", which tells a contestant
    nothing about what to change."""
    reason = partition_row(_entry(0, _alts(19, start=1) + [(0, -50.0)]), served=0)
    assert reason == ("carries 19 usable alternatives (20 distinct as reported), "
                      "fewer than the 20 the partition needs")


def test_a_wider_report_moves_the_threshold_rank_and_the_row_width():
    """A gate that asks for 65 alternatives keeps the 63 strictly above the sixty-fourth
    largest, and pads rows to 63. The rule is the default's with the rank moved."""
    alts = _alts(65)
    ids, lps, n_distinct = partition_row(_entry(0, alts), served=0, width=65)
    assert n_distinct == 65
    assert len(ids) == len(lps) == kept_width(65) == 63
    assert ids == [i for i, _ in alts[:63]]
    assert PAD_ID not in ids


def test_a_wider_report_refuses_a_default_width_report():
    """A server that sends the text tasks' 21 alternatives to a gate that asks for 65
    cannot fix that gate's threshold, so the position is a violation, not a row."""
    reason = partition_row(_entry(0, _alts(TOP_LOGPROBS_REQUESTED)), served=0, width=65)
    assert isinstance(reason, str)
    assert "fewer than the 64 the partition needs" in reason


def test_the_default_width_is_the_text_tasks_21():
    """Leaving the width unstated is the 21-wide partition every text task calibrated
    on, row for row."""
    alts = _alts(30)
    assert (partition_row(_entry(0, alts), served=0)
            == partition_row(_entry(0, alts), served=0, width=TOP_LOGPROBS_REQUESTED))
    assert len(partition_row(_entry(0, alts), served=0)[0]) == TOP_LOGPROBS_KEPT


def test_reply_report_carries_the_width_to_every_position():
    """``reply_report`` builds every row at the width it is asked for, and fails the
    reply at the first position that cannot fill it."""
    served = [0, 1]
    wide = TextPayload(text="", tokens=served,
                       logprobs=[_entry(0, _dist_alts(65)), _entry(1, _dist_alts(65))])
    rows = reply_report(wide, served, width=65)
    assert not isinstance(rows, str)
    assert all(len(r) == 63 for r in rows.q_ids)
    narrow = TextPayload(text="", tokens=served,
                         logprobs=[_entry(0, _dist_alts(65)), _entry(1, _dist_alts(21))])
    reason = reply_report(narrow, served, width=65)
    assert isinstance(reason, str) and reason.startswith("position 1 ")


def test_partition_deduplicates_before_counting():
    """A server that concatenates the served token onto its top-k without deduplicating
    has reported one alternative twice, not two: 21 entries over 20 distinct ids count
    as 20, which is exactly the threshold rank."""
    alts = _alts(20) + [(0, -0.5)]
    ids, _, n_distinct = partition_row(_entry(0, alts), served=0)
    assert n_distinct == 20
    assert len(ids) == TOP_LOGPROBS_KEPT
    assert len(set(ids)) == len(ids)
    # One fewer distinct id and the threshold rank is not on the wire at all.
    reason = partition_row(_entry(0, _alts(19) + [(0, -0.5)]), served=0)
    assert isinstance(reason, str)
    assert "19 usable alternatives (19 distinct as reported)" in reason


def test_partition_refuses_an_alternative_without_a_token_id():
    alts = _alts(TOP_LOGPROBS_REQUESTED)
    entry = _entry(0, alts)
    entry = TokenLogprob(token="t", token_id=0, logprob=-0.5,
                         top_logprobs=[TopLogprob(token="x", token_id=None, logprob=-1.0)])
    assert isinstance(partition_row(entry, served=0), str)


# --- the reply-level report ----------------------------------------------------


def _payload(n_positions: int, served: list[int]) -> TextPayload:
    """A reply whose every position reports a full, well-formed row."""
    entries = [_entry(served[j], _dist_alts(TOP_LOGPROBS_REQUESTED, start=served[j]))
               for j in range(n_positions)]
    return TextPayload(text="x", tokens=served, logprobs=entries)


def test_reply_report_builds_parallel_rows():
    served = [3, 4, 5]
    rows = reply_report(_payload(3, served), served)
    assert len(rows.q_ids) == len(rows.q_lp) == len(rows.q_emit) == 3
    assert rows.n_short == 0
    assert all(len(r) == TOP_LOGPROBS_KEPT for r in rows.q_ids)


def test_reply_report_names_a_missing_report():
    payload = TextPayload(text="x", tokens=[1, 2])
    assert reply_report(payload, [1, 2]) == "the reply carries no reported logprobs"


def test_reply_report_names_a_short_report():
    served = [1, 2, 3]
    payload = _payload(2, served[:2])
    reason = reply_report(payload, served)
    assert isinstance(reason, str) and "2 reported positions for 3 served tokens" in reason


def test_reply_report_names_a_long_report():
    """Extra positions are as much a mismatch as missing ones: the server is describing
    positions it did not serve, so the report is not the one the emission came from."""
    served = [1, 2, 3]
    payload = _payload(4, served + [4])
    reason = reply_report(payload, served)
    assert isinstance(reason, str) and "4 reported positions for 3 served tokens" in reason


def test_reply_report_names_a_position_reporting_another_token():
    """The report has to be the one the emission came from, which is the premise of the
    whole check, so a disagreement fails the reply rather than being scored around."""
    served = [1, 2]
    payload = _payload(2, served)
    bad = TokenLogprob(token="t", token_id=99, logprob=-0.5,
                       top_logprobs=payload.logprobs[1].top_logprobs)
    payload = TextPayload(text="x", tokens=served,
                          logprobs=[payload.logprobs[0], bad])
    reason = reply_report(payload, served)
    assert isinstance(reason, str) and "reports token 99, served 2" in reason


def test_reply_report_counts_short_positions_without_failing():
    """A report one alternative narrower than the contract asked for still has the
    threshold rank on the wire, so the partition holds and the position is only
    counted."""
    served = [1, 2]
    full = _payload(2, served)
    narrow = _entry(2, _dist_alts(TOP_LOGPROBS_REQUESTED - 1, start=2))
    payload = TextPayload(text="x", tokens=served, logprobs=[full.logprobs[0], narrow])
    rows = reply_report(payload, served)
    assert rows.n_short == 1


# --- the report has to be a distribution ---------------------------------------


def _mass_alts(total: float) -> list[tuple[int, float]]:
    """A full-width report whose kept atoms carry ``total`` of probability between them.

    The atoms are a halving sequence scaled to that total, distinct so the partition
    keeps exactly them, followed by the two entries the threshold rank needs at
    probabilities far below the smallest of them."""
    w = [2.0 ** -i for i in range(TOP_LOGPROBS_KEPT)]
    scale = total / sum(w)
    alts = [(i, math.log(scale * wi)) for i, wi in enumerate(w)]
    return alts + [(500 + j, math.log(1e-9 * (j + 1)))
                   for j in range(TOP_LOGPROBS_REQUESTED - TOP_LOGPROBS_KEPT)]


def _one_position(served: int, alts: list[tuple[int, float]],
                  emitted: float | None = None):
    """``reply_report`` on a one-position reply carrying that report."""
    payload = TextPayload(text="x", tokens=[served],
                          logprobs=[_entry(served, alts, emitted)])
    return reply_report(payload, [served])


def test_reply_report_fails_a_position_reporting_more_than_unit_mass():
    """A position cannot hold more than one of probability, so a report that says it
    does is not a report of that position's distribution and the reply fails on it --
    for the same reason a wrong token id or a wrong position count fails it."""
    reason = _one_position(3, _mass_alts(1.5))
    assert reason == ("position 0 reports 1.5 of probability over its alternatives, "
                      "more than the one a distribution has")


def test_reply_report_keeps_a_report_a_hair_over_unit_mass():
    """Log-probabilities arrive rounded, so mass a little over one is the wire rather
    than a server's claim, and the position is measured with an empty tail."""
    rows = _one_position(0, _mass_alts(1.0 + MASS_TOL / 2))
    assert isinstance(rows, ReportRows)
    assert len(rows.q_ids) == 1 and len(rows.q_ids[0]) == TOP_LOGPROBS_KEPT


def test_reply_report_fails_a_served_token_outside_the_atoms_at_probability_zero():
    """The atoms already hold the whole of the position's mass and the served token is
    not one of them, so the report says the server drew a token it gave no probability.
    Read off the arithmetic: a log-probability of -800 passes every check on the value
    and still exponentiates to zero."""
    reason = _one_position(900, _mass_alts(1.0 + 2e-5), emitted=-800.0)
    assert reason == "position 0 reports the token it served at probability zero"


def test_reply_report_fails_a_served_token_inside_the_atoms_at_probability_zero():
    """The same claim with the served token listed among the alternatives, which is the
    reading the mass alone does not reach: the atoms leave a tail, so nothing about the
    row's total is wrong, and only the served token's own cell is empty.

    That cell is what the transform is drawn across, so a report shaped like this fixes
    its own transform -- a cell of no width puts it exactly at the mass below the cell,
    whatever uniform the verifier drew -- while the divergence charges the atom
    nothing. A sampler cannot draw a token it gives no probability, inside the
    alternatives or outside them."""
    atoms = [(i, math.log(0.05)) for i in range(18)]
    reason = _one_position(900, atoms + [(900, -800.0), (901, -900.0), (902, -901.0)])
    assert reason == "position 0 reports the token it served at probability zero"


def test_the_distribution_check_names_the_position():
    """A trace carries millions of positions, so a contestant reading the verdict needs
    the one that failed."""
    served = [1, 2, 3]
    entries = [_entry(t, _dist_alts(TOP_LOGPROBS_REQUESTED, start=t)) for t in served]
    entries[2] = _entry(3, _mass_alts(2.0))
    reason = reply_report(TextPayload(text="x", tokens=served, logprobs=entries), served)
    assert reason.startswith("position 2 reports 2 of probability")


# --- the reported values have to be log-probabilities ---------------------------


def test_reply_report_fails_a_served_token_logprob_that_is_not_a_number():
    """A value that is not a number compares false against every bound, so the mass
    arithmetic would pass it through and a statistic would carry it. The reply fails on
    the value instead."""
    reason = _one_position(900, _mass_alts(0.5), emitted=float("nan"))
    assert reason == "position 0 reports a log-probability that is not a number: nan"


def test_reply_report_fails_alternatives_that_are_not_numbers():
    """A report whose every alternative is not a number describes nothing, and it is the
    shape that reads best if it is let through: no value survives the partition's
    comparisons, so the row comes out empty, its tail holds everything, and the
    divergence and the transform both read as a report in perfect agreement with the
    reference. Only the served-outside rate would see it."""
    alts = [(i, float("nan")) for i in range(TOP_LOGPROBS_REQUESTED)]
    reason = _one_position(900, alts, emitted=math.log(0.5))
    assert reason == "position 0 reports a log-probability that is not a number: nan"


def test_alternatives_at_minus_infinity_are_atoms_of_no_mass():
    """A stack that masks tokens before it samples reports minus infinity for every
    token it masked, and a top-k wider than the surviving set carries some of them. That
    is a claim of zero probability, which the arithmetic reads as an atom of no mass, so
    the report is read rather than refused; the masked entries fall below the threshold
    rank and out of the partition."""
    alts = _dist_alts(TOP_LOGPROBS_REQUESTED)
    alts[-3:] = [(i_, float("-inf")) for i_, _ in alts[-3:]]
    rows = _one_position(0, alts)
    assert isinstance(rows, ReportRows)
    assert all(i_ not in rows.q_ids[0] for i_, _ in alts[-3:])


def test_reply_report_fails_a_log_probability_above_zero():
    """One token cannot hold more probability than a whole position, so a positive
    log-probability is not one. It is read off the value rather than off the mass it
    makes: a large enough value overflows the exponential, and a trace carries millions
    of positions to overflow it at."""
    alts = _dist_alts(TOP_LOGPROBS_REQUESTED)
    alts[0] = (alts[0][0], 800.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reason = _one_position(3, alts, emitted=-2.0)
    assert reason == "position 0 reports a log-probability above zero: 800"


def test_a_log_probability_a_hair_above_zero_is_the_wire_rounding():
    """Values arrive rounded, so one a little over zero is the wire rather than a claim
    of more than unit mass, on the same slack a whole row's mass is given."""
    alts = [(0, 5e-5)] + [(i, -100.0) for i in range(1, TOP_LOGPROBS_REQUESTED)]
    rows = _one_position(0, alts)
    assert isinstance(rows, ReportRows)
    assert rows.q_ids[0][0] == 0


# --- the per-position statistics ----------------------------------------------


def _rows(atoms: list[list[tuple[int, float]]], width: int = 3):
    """Partition rows from per-position (id, log-probability) atoms, padded to
    ``width``."""
    ids = [[i for i, _ in row] + [PAD_ID] * (width - len(row)) for row in atoms]
    lps = [[v for _, v in row] + [float("-inf")] * (width - len(row)) for row in atoms]
    return np.array(ids, dtype=np.int64), np.array(lps, dtype=np.float64)


def _eot(n: int) -> dict:
    """The end-of-turn inputs of ``n`` positions that never end a turn: a reference
    price far below any threshold, and a checkpoint whose end-of-turn id none of these
    fixtures serve. The three report statistics read neither, so a test about them
    states them once here."""
    return {"p_eot_lp": [math.log(1e-9)] * n, "eot_ids": (9999,)}


def _self_reference(q_lp: np.ndarray, q_tail: np.ndarray):
    """A reference that agrees with the report exactly, so the divergence is zero and a
    test can read the other statistics on their own."""
    tail = np.where(q_tail > 0, np.log(np.maximum(q_tail, 1e-300)), -np.inf)
    return q_lp, tail


def test_served_inside_the_atoms():
    """The served token is an atom: its cell is that atom's mass, the transform lands
    inside it, and the moment reads the atom's own log-probability."""
    q = [0.5, 0.3, 0.2]
    atoms = [[(10, math.log(q[0])), (11, math.log(q[1])), (12, math.log(q[2]))]]
    q_ids, q_lp = _rows(atoms)
    p_on_q, p_tail = _self_reference(q_lp, np.array([0.0]))
    st = position_stats(served=[11], q_ids=q_ids, q_lp=q_lp,
                        q_emit=[math.log(q[1])], p_on_q=p_on_q, p_tail_lp=p_tail,
                        v=[0.25], **_eot(1))
    h = -sum(p * math.log(p) for p in q)
    assert st["outside"] == [False]
    assert st["kl_qp"][0] == pytest.approx(0.0, abs=1e-12)
    # Cells in report order are [0.5) for id 10 and [0.5, 0.8) for id 11.
    assert st["u_q"][0] == pytest.approx(0.5 + 0.25 * 0.3)
    assert st["s_q"][0] == pytest.approx(math.log(0.3) + h)
    assert st["var_q"][0] == pytest.approx(
        sum(p * (math.log(p) + h) ** 2 for p in q))


def test_served_outside_the_atoms_lands_in_the_tail():
    """A served token the report did not list falls in the tail interval, which begins
    where the atoms end, and its moment reads the tail's log-mass."""
    q = [0.5, 0.3]
    atoms = [[(10, math.log(q[0])), (11, math.log(q[1]))]]
    q_ids, q_lp = _rows(atoms)
    p_on_q, p_tail = _self_reference(q_lp, np.array([0.2]))
    st = position_stats(served=[77], q_ids=q_ids, q_lp=q_lp,
                        q_emit=[math.log(0.05)], p_on_q=p_on_q, p_tail_lp=p_tail,
                        v=[0.5], **_eot(1))
    assert st["outside"] == [True]
    assert st["u_q"][0] == pytest.approx(0.8 + 0.5 * 0.2)
    h = -sum(p * math.log(p) for p in q) - 0.2 * math.log(0.2)
    assert st["s_q"][0] == pytest.approx(math.log(0.2) + h)


def test_a_peaked_report_floors_the_tail_at_the_served_probability():
    """A report whose atoms round to unit mass has still reported the served token's own
    probability, so an outside token is charged what the report says it was worth rather
    than failing the reply."""
    atoms = [[(10, math.log(1.0))]]
    q_ids, q_lp = _rows(atoms, width=1)
    st = position_stats(served=[77], q_ids=q_ids, q_lp=q_lp,
                        q_emit=[math.log(1e-9)],
                        p_on_q=np.array([[math.log(0.9)]]),
                        p_tail_lp=[math.log(0.1)], v=[0.5], **_eot(1))
    assert st["outside"] == [True]
    # The tail is the floor, 1e-9, so the moment charges that.
    h = -1e-9 * math.log(1e-9)
    assert st["s_q"][0] == pytest.approx(math.log(1e-9) + h, rel=1e-9)
    assert np.isfinite(st["kl_qp"][0])


def test_mass_just_over_one_is_rounding_and_is_measured():
    """Log-probabilities arrive rounded, so a report a hair over unit mass is measured
    with an empty tail."""
    over = 1.0 + MASS_TOL / 2
    atoms = [[(10, math.log(over))]]
    q_ids, q_lp = _rows(atoms, width=1)
    st = position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[math.log(over)],
                        p_on_q=np.array([[0.0]]), p_tail_lp=[-np.inf], v=[0.5],
                        **_eot(1))
    assert np.isfinite(st["kl_qp"][0])


def test_padded_slots_carry_no_mass():
    """Every padded slot carries the same id and no probability: they are empty slots
    rather than atoms, so the tail is what the one real atom leaves."""
    atoms = [[(10, math.log(0.6))]]
    q_ids, q_lp = _rows(atoms, width=4)
    p_on_q, p_tail = _self_reference(q_lp, np.array([0.4]))
    st = position_stats(served=[77], q_ids=q_ids, q_lp=q_lp, q_emit=[math.log(0.4)],
                        p_on_q=p_on_q, p_tail_lp=p_tail, v=[0.5], **_eot(1))
    # The served token is outside the one atom, so its cell is the whole 0.4 tail.
    assert st["u_q"][0] == pytest.approx(0.6 + 0.5 * 0.4)


def test_position_stats_refuses_a_row_that_is_not_a_distribution():
    """``reply_report`` fails a reply carrying such a row before anything is reduced, so
    meeting one here means a caller reduced a report nothing checked -- which fails loud
    rather than returning a statistic over numbers that are not a distribution."""
    atoms = [[(10, math.log(0.7)), (11, math.log(0.7))]]
    q_ids, q_lp = _rows(atoms, width=2)
    p_on_q, p_tail = _self_reference(q_lp, np.array([0.0]))
    with pytest.raises(ValueError, match="position 0 reports 1.4 of probability"):
        position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[math.log(0.7)],
                       p_on_q=p_on_q, p_tail_lp=p_tail, v=[0.5], **_eot(1))


def test_position_stats_refuses_a_row_that_repeats_an_id():
    """The mass is summed over a row's slots, so a repeated id is one atom counted once
    per copy: the report would claim mass it never reported, and the served token's cell
    would be the first copy's while the rest inflated the atoms around it.
    ``partition_row`` drops duplicates, so this is a row that came from somewhere else.
    """
    atoms = [[(10, math.log(0.3)), (10, math.log(0.3)), (11, math.log(0.2))]]
    q_ids, q_lp = _rows(atoms)
    p_on_q, p_tail = _self_reference(q_lp, np.array([0.2]))
    with pytest.raises(ValueError, match="position 0 reports one id twice"):
        position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[math.log(0.3)],
                       p_on_q=p_on_q, p_tail_lp=p_tail, v=[0.5], **_eot(1))


def test_position_stats_measures_a_padded_row_without_reading_its_padding_as_repeats():
    """Every padded slot carries the same id, which is what padding is, so the repeated
    id the reduction refuses is one among the atoms."""
    atoms = [[(10, math.log(0.6))]]
    q_ids, q_lp = _rows(atoms, width=4)
    p_on_q, p_tail = _self_reference(q_lp, np.array([0.4]))
    st = position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[math.log(0.6)],
                        p_on_q=p_on_q, p_tail_lp=p_tail, v=[0.5], **_eot(1))
    assert st["outside"] == [False]


def test_position_stats_refuses_a_served_token_of_no_probability():
    """The reduction reads the same arithmetic the contract check does, so it refuses
    the same rows: here the served token is an atom the report gives no probability, so
    its cell has no width and the transform across it would be fixed rather than
    randomized."""
    atoms = [[(10, -800.0), (11, math.log(0.5))]]
    q_ids, q_lp = _rows(atoms, width=2)
    p_on_q, p_tail = _self_reference(q_lp, np.array([0.5]))
    with pytest.raises(ValueError,
                       match="position 0 reports the token it served at probability"):
        position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[-800.0],
                       p_on_q=p_on_q, p_tail_lp=p_tail, v=[0.5], **_eot(1))


def test_position_stats_measures_every_position_it_is_handed():
    """Nothing is excluded: a report that got past ``reply_report`` is a
    distribution at every position, so no statistic is NaN and there is no mask saying
    which ones to skip."""
    st = position_stats(**_honest_draw(500, seed=12))
    assert "refused" not in st
    for key in ("kl_qp", "u_q", "s_q", "var_q"):
        assert not np.isnan(st[key]).any(), key
        assert len(st[key]) == 500


def test_the_reduction_never_meets_a_row_the_contract_check_accepted():
    """One arithmetic decides which reports fail and which are measured, so the
    reduction's raise is unreachable from a report that got past ``reply_report``. Two
    transcriptions of it would each be right about their own rows and wrong about the
    ones between them, where the gap is a crash on one side and an unmeasured position
    on the other.

    The reports here straddle every boundary that arithmetic has: masses either side of
    the rounding slack, a served token that is an atom or outside them, a served token
    reported so low its probability underflows, and values that are not
    log-probabilities at all."""
    rng = np.random.default_rng(3)
    cases = [(900, [(i, float("nan")) for i in range(TOP_LOGPROBS_REQUESTED)], -0.7),
             (3, [(0, 800.0)] + _dist_alts(TOP_LOGPROBS_REQUESTED - 1, start=1), -2.0),
             (900, [(i, math.log(0.05)) for i in range(18)]
              + [(900, -800.0), (901, -900.0), (902, -901.0)], -800.0)]
    for _ in range(200):
        total = float(rng.uniform(0.9, 1.1))
        outside = bool(rng.random() < 0.5)
        served = 900 if outside else int(rng.integers(0, TOP_LOGPROBS_KEPT))
        emitted = float(rng.choice([-800.0, math.log(1e-6), -3.0]))
        cases.append((served, _mass_alts(total), emitted if outside else None))
    measured = failed = 0
    for served, alts, emitted in cases:
        rows = _one_position(served, alts, emitted=emitted)
        if isinstance(rows, str):
            failed += 1
            continue
        measured += 1
        q_lp = np.asarray(rows.q_lp, dtype=np.float64)
        st = position_stats(served=[served], q_ids=np.asarray(rows.q_ids, np.int64),
                            q_lp=q_lp, q_emit=rows.q_emit, p_on_q=q_lp,
                            p_tail_lp=[math.log(1e-3)], v=[0.5], **_eot(1))
        assert np.isfinite(st["u_q"]).all()
    assert measured and failed, "the reports have to exercise both outcomes"


def test_divergence_reads_both_the_atoms_and_the_tail():
    """The coarsened divergence is the atom terms plus the tail's, which is what makes
    it a lower bound on the full-vocabulary one and identical on both sides of the
    gate."""
    q = [0.5, 0.3]
    p = [0.4, 0.4]
    atoms = [[(10, math.log(q[0])), (11, math.log(q[1]))]]
    q_ids, q_lp = _rows(atoms, width=2)
    st = position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[math.log(q[0])],
                        p_on_q=np.array([[math.log(p[0]), math.log(p[1])]]),
                        p_tail_lp=[math.log(0.2)], v=[0.5], **_eot(1))
    want = (q[0] * math.log(q[0] / p[0]) + q[1] * math.log(q[1] / p[1])
            + 0.2 * math.log(0.2 / 0.2))
    assert st["kl_qp"][0] == pytest.approx(want)


def test_divergence_is_infinite_where_the_reference_leaves_no_tail():
    """A report that puts mass where the reference puts none diverges without bound
    there, and saying so is the point: the verdict fails on it."""
    atoms = [[(10, math.log(0.6))]]
    q_ids, q_lp = _rows(atoms, width=1)
    st = position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[math.log(0.6)],
                        p_on_q=np.array([[0.0]]), p_tail_lp=[-np.inf], v=[0.5],
                        **_eot(1))
    assert st["kl_qp"][0] == math.inf


def test_position_stats_refuses_mismatched_shapes():
    q_ids, q_lp = _rows([[(10, -0.5)]], width=2)
    with pytest.raises(ValueError, match="one partition row per served token"):
        position_stats(served=[10, 11], q_ids=q_ids, q_lp=q_lp, q_emit=[-0.5, -0.5],
                       p_on_q=q_lp, p_tail_lp=[-1.0, -1.0], v=[0.5, 0.5], **_eot(2))
    with pytest.raises(ValueError, match="q_lp"):
        position_stats(served=[10], q_ids=q_ids, q_lp=np.zeros((1, 3)), q_emit=[-0.5],
                       p_on_q=q_lp, p_tail_lp=[-1.0], v=[0.5], **_eot(1))
    with pytest.raises(ValueError, match="q_emit"):
        position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[-0.5, -0.5],
                       p_on_q=q_lp, p_tail_lp=[-1.0], v=[0.5], **_eot(1))
    with pytest.raises(ValueError, match="p_eot_lp"):
        position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[-0.5],
                       p_on_q=q_lp, p_tail_lp=[-1.0], v=[0.5], **_eot(2))
    with pytest.raises(ValueError, match="p_on_q"):
        position_stats(served=[10], q_ids=q_ids, q_lp=q_lp, q_emit=[-0.5],
                       p_on_q=np.zeros((1, 3)), p_tail_lp=[-1.0], v=[0.5], **_eot(1))


# --- the two exact nulls -------------------------------------------------------


def _honest_draw(n: int, seed: int):
    """``n`` positions of one fixed report, with the served token drawn from it.

    Five atoms holding 0.9 and a tail holding 0.1; a draw that lands in the tail serves
    an id the report never listed, which is the outside branch of every statistic."""
    probs = np.array([0.35, 0.25, 0.15, 0.1, 0.05])
    tail = 1.0 - probs.sum()
    ids = np.array([10, 11, 12, 13, 14], dtype=np.int64)
    rng = np.random.default_rng(seed)
    cell = rng.choice(len(probs) + 1, size=n, p=list(probs) + [tail])
    served = np.where(cell < len(probs), ids[np.minimum(cell, len(probs) - 1)], 900)
    q_ids = np.tile(ids, (n, 1))
    q_lp = np.tile(np.log(probs), (n, 1))
    q_emit = np.where(cell < len(probs), q_lp[:, 0], math.log(tail / 50))
    q_emit = np.where(cell < len(probs),
                      np.log(probs)[np.minimum(cell, len(probs) - 1)],
                      math.log(tail / 50))
    p_on_q, p_tail = q_lp, np.full(n, math.log(tail))
    return dict(served=served, q_ids=q_ids, q_lp=q_lp, q_emit=q_emit,
                p_on_q=p_on_q, p_tail_lp=p_tail, v=rng.random(n), **_eot(n))


def test_the_transform_is_uniform_under_an_honest_sampler():
    """The randomized transform of the served token under its own report is uniform on
    [0, 1) whatever the report is, so a correct draw passes the uniformity test."""
    st = position_stats(**_honest_draw(20_000, seed=7))
    u = st["u_q"]
    assert u.min() >= 0.0 and u.max() < 1.0
    _, p = cramer_von_mises_uniform(u)
    assert p > 0.01


def test_the_transform_is_not_uniform_when_the_draw_is_not_the_report():
    """Reporting one distribution and serving from another is what the test exists to
    catch: here the server always serves its reported mode."""
    draw = _honest_draw(5_000, seed=8)
    draw["served"] = np.full(len(draw["served"]), 10, dtype=np.int64)
    draw["q_emit"] = np.full(len(draw["served"]), draw["q_lp"][0, 0])
    st = position_stats(**draw)
    _, p = cramer_von_mises_uniform(st["u_q"])
    assert p < 1e-6


def test_the_log_score_moment_is_zero_mean_under_an_honest_sampler():
    """Both of the moment's terms come from the report alone, so its mean is exactly
    zero under a faithful draw and the deviate it forms is standard normal."""
    n = 40_000
    st = position_stats(**_honest_draw(n, seed=9))
    ok = np.isfinite(st["var_q"]) & (st["var_q"] > 0)
    z = st["s_q"][ok].sum() / math.sqrt(st["var_q"][ok].sum())
    assert abs(z) < 3.0


def test_the_log_score_moment_moves_when_the_draw_is_peaked():
    """A server that serves likelier tokens than it reported produces a positive
    deviate, which is the second reading of the same null."""
    draw = _honest_draw(5_000, seed=10)
    draw["served"] = np.full(len(draw["served"]), 10, dtype=np.int64)
    draw["q_emit"] = np.full(len(draw["served"]), draw["q_lp"][0, 0])
    st = position_stats(**draw)
    ok = np.isfinite(st["var_q"]) & (st["var_q"] > 0)
    z = st["s_q"][ok].sum() / math.sqrt(st["var_q"][ok].sum())
    assert z > 10.0


def test_reply_uniforms_are_deterministic_per_event_and_differ_across_events():
    a = reply_uniforms("t0:r0", 16)
    assert np.array_equal(a, reply_uniforms("t0:r0", 16))
    assert not np.array_equal(a, reply_uniforms("t0:r1", 16))
    assert a.min() >= 0.0 and a.max() < 1.0


# --- the records ---------------------------------------------------------------


def _record(event_key="t0:r0", n=4, kl=0.5, seed=1) -> ReplyReport:
    draw = _honest_draw(n, seed=seed)
    st = position_stats(**draw)
    # Shift the divergence to a chosen per-position value so the pooled arithmetic is
    # readable; the reduction's own divergence on a self-consistent reference is zero.
    st["kl_qp"] = st["kl_qp"] + kl
    return score_reply(st, event_key=event_key, n_served=n, finish_reason="stop",
                       ends_with_eot=True, at_cap=False)


def test_reply_report_round_trips():
    rec = _record()
    again = ReplyReport.from_json(rec.to_json())
    assert again == rec


def test_a_stored_record_carrying_a_key_this_reduction_dropped_still_loads():
    """A record in a calibration capture store can carry fields this reduction does not
    count. Reading one back reads the keys it needs and leaves the rest alone."""
    rec = _record()
    assert ReplyReport.from_json({**rec.to_json(), "n_refused": 0}) == rec


def test_score_reply_refuses_a_reduction_of_the_wrong_length():
    st = position_stats(**_honest_draw(3, seed=2))
    with pytest.raises(ValueError, match="3 positions for 4 served tokens"):
        score_reply(st, event_key="t0:r0", n_served=4, finish_reason="stop",
                    ends_with_eot=True, at_cap=False)


def test_pooled_report_means_over_the_measured_positions():
    pooled = PooledReport()
    pooled.extend([_record("t0:r0", n=4, kl=0.5, seed=1),
                   _record("t0:r1", n=6, kl=1.0, seed=2)])
    assert pooled.n_replies == 2 and pooled.n_tokens == 10
    assert pooled.mean_kl == pytest.approx((4 * 0.5 + 6 * 1.0) / 10)


def test_the_pool_refuses_a_record_that_measured_fewer_positions_than_it_served():
    """Every position of a measured reply is measured, so a record with fewer
    transforms than served tokens was reduced by something else: its divergence total
    and its position count would be over different sets, and a mean over the two would
    mean nothing."""
    rec = _record("t0:r0", n=4)
    short = ReplyReport(**{**rec.to_json(), "u": rec.u[:2],
                           "p_eot_lp": list(rec.p_eot_lp),
                           "eot_positions": rec.eot_positions})
    with pytest.raises(ValueError, match="2 transforms for 4 served tokens"):
        PooledReport().add(short)


def test_pooled_report_counts_a_violation_and_nothing_else():
    """A reply whose report violates the contract was never measured, so it moves the
    violation count and no other total."""
    pooled = PooledReport()
    pooled.add(_record("t0:r0", n=4, kl=0.5))
    pooled.add(ReplyReport(event_key="t0:r1", n_tokens=7, n_outside=0,
                           kl_sum=0.0, s_sum=0.0, var_sum=0.0, n_moment=0, u=(),
                           p_eot_lp=(), eot_positions=(),
                           finish_reason="stop", ends_with_eot=False, at_cap=False,
                           violation="the reply carries no reported logprobs"))
    assert pooled.n_replies == 2 and pooled.n_violations == 1
    assert pooled.n_tokens == 4


def test_pooled_standard_error_is_clustered_by_reply():
    """Positions inside one reply are one piece of evidence, not many: a measurement
    whose replies differ from each other has a larger clustered error than the
    independent-position one, and the gate reads the clustered one."""
    per_reply = [0.2, 0.9, 0.4, 1.3]
    n_each = 50
    pooled = PooledReport()
    for i, kl in enumerate(per_reply):
        pooled.add(ReplyReport(event_key=f"t0:r{i}", n_tokens=n_each, n_outside=0, kl_sum=kl * n_each, s_sum=0.0,
                               var_sum=0.0, n_moment=0, u=tuple([0.5] * n_each),
                               p_eot_lp=(), eot_positions=(),
                               finish_reason="stop", ends_with_eot=True, at_cap=False,
                               violation=None))
    values = np.repeat(per_reply, n_each)
    n = len(values)
    residual = values - values.mean()
    se_iid = math.sqrt((residual ** 2).sum() * n / (n - 1)) / n
    assert pooled.mean_kl == pytest.approx(values.mean())
    assert pooled.se_kl > se_iid * 5


def _counted(event_key: str, n_tokens: int, n_outside: int) -> ReplyReport:
    """A measured record carrying only what the served-outside rate reads."""
    return ReplyReport(event_key=event_key, n_tokens=n_tokens, n_outside=n_outside, kl_sum=0.0, s_sum=0.0, var_sum=0.0,
                       n_moment=0, u=tuple([0.5] * n_tokens), p_eot_lp=(),
                       eot_positions=(), finish_reason="stop", ends_with_eot=True,
                       at_cap=False, violation=None)


def test_pooled_outside_rate_is_over_the_served_positions():
    """The rate is the served-outside count over every served position of the measured
    replies; a violated reply, never measured, is in neither."""
    pooled = PooledReport()
    pooled.extend([_counted("t0:r0", 40, 2), _counted("t0:r1", 60, 3)])
    pooled.add(ReplyReport(event_key="t0:r2", n_tokens=7, n_outside=7,
                           kl_sum=0.0, s_sum=0.0, var_sum=0.0, n_moment=0, u=(),
                           p_eot_lp=(), eot_positions=(), finish_reason="stop",
                           ends_with_eot=False, at_cap=False,
                           violation="the reply carries no reported logprobs"))
    assert pooled.n_outside == 5 and pooled.n_tokens == 100
    assert pooled.outside_rate == pytest.approx(0.05)


def test_pooled_outside_error_is_clustered_by_reply():
    """Replies that differ in their rate carry a larger clustered error than the
    independent-position one, as the divergence's does."""
    per_reply = [0, 10, 2, 14]
    n_each = 50
    pooled = PooledReport()
    for i, k in enumerate(per_reply):
        pooled.add(_counted(f"t0:r{i}", n_each, k))
    rate = sum(per_reply) / (n_each * len(per_reply))
    n = n_each * len(per_reply)
    se_iid = math.sqrt(rate * (1 - rate) / n)
    assert pooled.outside_rate == pytest.approx(rate)
    assert pooled.se_outside > se_iid * 2


def test_pooled_outside_rate_is_empty_before_anything_is_added():
    pooled = PooledReport()
    assert math.isnan(pooled.outside_rate) and math.isnan(pooled.se_outside)
    pooled.add(_counted("t0:r0", 10, 1))
    assert pooled.outside_rate == pytest.approx(0.1)
    assert math.isnan(pooled.se_outside)


def test_pooled_standard_error_needs_two_replies():
    """One reply leaves nothing to cluster, so there is no error to report rather than
    a misleadingly small one."""
    pooled = PooledReport()
    pooled.add(_record())
    assert math.isnan(pooled.se_kl)


def test_pooled_report_is_empty_before_anything_is_added():
    pooled = PooledReport()
    assert pooled.n_tokens == 0
    assert math.isnan(pooled.mean_kl) and math.isnan(pooled.z)
    assert all(math.isnan(x) for x in pooled.uniformity())


# --- the Cramer-von Mises test -------------------------------------------------


def test_cvm_matches_the_reference_implementation():
    """Values from scipy.stats.cramervonmises, which this is a port of: the statistic
    bit for bit and the p-value to the accuracy of the expansion behind it."""
    rng = np.random.default_rng(0)
    cases = [(rng.random(20), 0.09485412265288656, 0.6154781350529503),
             (rng.random(200), 0.34604356195541003, 0.1007250130971824),
             (rng.random(5000) ** 1.15, 8.521797816177282, 9.249745414052768e-11),
             (rng.random(50) * 0.8, 0.3249812547428769, 0.11491273686148662)]
    for u, want_stat, want_p in cases:
        stat, p = cramer_von_mises_uniform(u)
        assert stat == pytest.approx(want_stat, rel=1e-12)
        assert p == pytest.approx(want_p, rel=1e-4)


def test_cvm_p_value_stays_in_range():
    """The expansion overshoots by parts in 1e12 near an endpoint; a verdict must not
    publish a probability outside [0, 1]."""
    stat, p = cramer_von_mises_uniform(np.linspace(0.001, 0.999, 37))
    assert 0.0 <= p <= 1.0 and p == pytest.approx(1.0, abs=1e-9)


def test_cvm_needs_two_values():
    assert all(math.isnan(x) for x in cramer_von_mises_uniform(np.array([0.5])))


#: The distribution function at statistics up to just under the first one where a correction
#: series without an overflow guard overflows (about 4,177), for sample sizes from 20 to a TTS
#: trace's 5,066,400 positions.
CVM_GOLDEN = [
    (0.005, 20, -5.698194904733337e-11),
    (0.02, 20, 0.0022699060648161476),
    (0.05, 20, 0.11839068286079314),
    (0.1, 20, 0.4105633427772819),
    (0.2, 20, 0.7313929752242903),
    (0.3461, 20, 0.8999296806482441),
    (0.4614, 20, 0.9510571777390807),
    (0.7434, 20, 0.9907800081536796),
    (1.0, 20, 0.9979307098119743),
    (1.1679, 20, 0.9992264977017649),
    (2.0, 20, 0.9999965411085067),
    (5.0, 20, 0.9999999999694221),
    (20.0, 20, 1.0),
    (100.0, 20, 1.0),
    (1000.0, 20, 1.0),
    (4000.0, 20, 1.0),
    (4176.0, 20, 1.0),
    (0.005, 200, 1.4104030378092624e-11),
    (0.02, 200, 0.0029275434779233063),
    (0.05, 200, 0.12318623034886526),
    (0.1, 200, 0.41467023971161293),
    (0.2, 200, 0.7324159100357317),
    (0.3461, 200, 0.8993106834693563),
    (0.4614, 200, 0.9501160339257345),
    (0.7434, 200, 0.990075037743512),
    (1.0, 200, 0.9975786640190771),
    (1.1679, 200, 0.999022849928608),
    (2.0, 200, 0.9999881514482135),
    (5.0, 200, 0.9999999999531574),
    (20.0, 200, 0.9999999991752363),
    (100.0, 200, 1.0),
    (1000.0, 200, 1.0),
    (4000.0, 200, 1.0),
    (4176.0, 200, 1.0),
    (0.005, 5000, 2.1686534850138065e-11),
    (0.02, 5000, 0.0029976914686547364),
    (0.05, 5000, 0.12369775541425962),
    (0.1, 5000, 0.41510830871794163),
    (0.2, 5000, 0.7325250230822855),
    (0.3461, 5000, 0.8992446571036082),
    (0.4614, 5000, 0.950015645252311),
    (0.7434, 5000, 0.9899998408997608),
    (1.0, 5000, 0.9975411124678348),
    (1.1679, 5000, 0.9990011274994715),
    (2.0, 5000, 0.9999872565511156),
    (5.0, 5000, 0.9999999999514223),
    (20.0, 5000, 0.9999999991527732),
    (100.0, 5000, 0.9999999740548858),
    (1000.0, 5000, 0.9999997441699449),
    (4000.0, 5000, 1.0),
    (4176.0, 5000, 1.0),
    (0.005, 1000000, 2.2000892848041613e-11),
    (0.02, 1000000, 0.0030005996874371437),
    (0.05, 1000000, 0.12371896239092908),
    (0.1, 1000000, 0.415126470328829),
    (0.2, 1000000, 0.7325295467273405),
    (0.3461, 1000000, 0.8992419197605281),
    (0.4614, 1000000, 0.9500114833052251),
    (0.7434, 1000000, 0.9899967233639467),
    (1.0, 1000000, 0.9975395556431061),
    (1.1679, 1000000, 0.9990002269237633),
    (2.0, 1000000, 0.9999872194501733),
    (5.0, 1000000, 0.9999999999513504),
    (20.0, 1000000, 0.9999999991518419),
    (100.0, 1000000, 0.999999974046022),
    (1000.0, 1000000, 0.9999997441222641),
    (4000.0, 1000000, 0.9999994522965321),
    (4176.0, 1000000, 0.9999994394606562),
    (0.005, 5066400, 2.2002160739451405e-11),
    (0.02, 5066400, 0.0030006114170754434),
    (0.05, 5066400, 0.12371904792444095),
    (0.1, 5066400, 0.4151265435795513),
    (0.2, 5066400, 0.7325295649724328),
    (0.3461, 5066400, 0.899241908720078),
    (0.4614, 5066400, 0.9500114665189592),
    (0.7434, 5066400, 0.9899967107900759),
    (1.0, 5066400, 0.9975395493640081),
    (1.1679, 5066400, 0.9990002232914962),
    (2.0, 5066400, 0.9999872193005351),
    (5.0, 5066400, 0.9999999999513501),
    (20.0, 5066400, 0.999999999151838),
    (100.0, 5066400, 0.9999999740459863),
    (1000.0, 5066400, 0.9999997441220718),
    (4000.0, 5066400, 0.9999994522961254),
    (4176.0, 5066400, 0.9999994394602058),
]


#: How far a value of the distribution function may lie from its golden one on another host: numpy's
#: exponentials and their kin take another code path on another CPU's instructions, which moves a
#: value by a few ulps of one, while a change of the code moves it by at least the 1e-7 its series
#: is cut at.
CVM_ROUNDING = 1e-14


def test_the_distribution_function_is_unchanged_wherever_its_series_converged():
    """To within the host's float rounding (``CVM_ROUNDING``): the overflow guard moves no value
    where the correction series converges.

    ``CVM_GOLDEN`` is a frozen record, not regenerated: each value is the distribution function
    the Csorgo-Faraway expansion gives (``_cvm_cdf``) with its correction series summed with no
    overflow guard (``_CVM_CORRECTION_TERMS``), which at these statistics converges before it
    overflows. A change to the expansion that moves a value past rounding fails here; the values
    change only with a reviewed change to the expansion."""
    for x, n, value in CVM_GOLDEN:
        assert abs(report._cvm_cdf(x, n) - value) <= CVM_ROUNDING, (x, n)


@pytest.mark.parametrize("u", [np.full(200_000, 0.5), np.linspace(0, 1, 200_000) ** 2],
                         ids=["one-value", "squared"])
def test_a_statistic_past_the_expansions_range_has_p_zero_and_raises_nothing(u):
    """A gross departure from uniformity on a long trace puts the statistic in the
    thousands, past where the expansion's series converge within the terms float64 can hold.
    The distribution function is one within rounding
    there, so the p-value is zero, and the check fails rather than the verify role."""
    omega2, p = cramer_von_mises_uniform(u)
    assert omega2 > 4_177 and p == 0.0


# --- the end-of-turn count's exact tail ----------------------------------------


def _brute_force_tail(p: list[float], k: int) -> float:
    """``P(X >= k)`` by enumerating every outcome of the trials. Exponential in the
    trial count, so it is a check on the real thing rather than an implementation."""
    total = 0.0
    for bits in itertools.product((0, 1), repeat=len(p)):
        if sum(bits) >= k:
            term = 1.0
            for b, pi in zip(bits, p):
                term *= pi if b else 1.0 - pi
            total += term
    return total


def test_the_exact_tail_matches_brute_force_enumeration():
    """Against the definition, at every count a small trial set can reach, on
    probabilities that are all different -- the unequal-probability case is the whole
    reason the law is Poisson-binomial rather than binomial."""
    rng = np.random.default_rng(4)
    for n in (1, 2, 5, 9, 12):
        p = list(rng.random(n) * 0.9)
        for k in range(0, n + 2):
            assert poisson_binomial_upper_tail(np.array(p), k) == pytest.approx(
                _brute_force_tail(p, k), abs=1e-12), (n, k)


def test_the_exact_tail_matches_the_poisson_limit_for_many_tiny_trials():
    """Many small probabilities summing to a moderate mean is the regime a graded trace
    sits in, and there the law is close to Poisson at that mean -- which is where the
    check's spread of about the square root of the expectation comes from."""
    n, lam = 200_000, 5.0
    p = np.full(n, lam / n)
    for k in (1, 3, 5, 8, 12):
        want = 1.0 - sum(math.exp(-lam) * lam ** j / math.factorial(j)
                         for j in range(k))
        assert poisson_binomial_upper_tail(p, k) == pytest.approx(want, abs=1e-5)


def test_the_exact_tail_at_its_edges():
    p = np.array([0.3, 0.4])
    assert poisson_binomial_upper_tail(p, 0) == 1.0
    assert poisson_binomial_upper_tail(p, 3) == 0.0  # more endings than positions
    assert poisson_binomial_upper_tail(np.array([]), 1) == 0.0
    # A certain trial needs no special case: it moves the whole law up by one.
    certain = np.array([1.0, 0.25])
    assert poisson_binomial_upper_tail(certain, 1) == pytest.approx(1.0)
    assert poisson_binomial_upper_tail(certain, 2) == pytest.approx(0.25)


def test_the_exact_tail_refuses_a_probability_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        poisson_binomial_upper_tail(np.array([0.5, 1.5]), 1)


# --- the end-of-turn count -----------------------------------------------------


def test_the_count_reads_only_the_positions_the_threshold_prices():
    """A position priced above the threshold is not evidence, whatever the server
    served there: that is a position where the model itself was ending the turn."""
    prices = np.log(np.array([0.9, 1e-3, 1e-3, 1e-3]))
    served_eot = np.array([True, True, False, False])
    count = end_of_turn_count(prices, served_eot, eps=0.01)
    assert count.n_priced == 3
    assert count.expected == pytest.approx(3e-3)
    assert count.observed == 1
    assert count.p_value == pytest.approx(1 - (1 - 1e-3) ** 3)
    assert count.to_json()["priced_positions"] == 3


def test_the_count_is_one_sided():
    """Ending fewer turns than the reference expects is not a deviation, so a
    measurement under its expectation is nowhere near rejection."""
    prices = np.log(np.full(1000, 0.005))
    count = end_of_turn_count(prices, np.zeros(1000, dtype=bool), eps=0.01)
    assert count.expected == pytest.approx(5.0)
    assert count.observed == 0 and count.p_value == 1.0


def test_the_count_refuses_prices_and_endings_of_different_lengths():
    with pytest.raises(ValueError, match="one price per served position"):
        end_of_turn_count(np.zeros(3), np.zeros(2, dtype=bool), eps=0.1)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_the_count_refuses_a_price_that_is_not_one(bad):
    """A NaN compares false with the threshold, so it would drop its position from both
    sides of the count and turn a served ending there into no evidence at all."""
    prices = np.log(np.array([1e-3, 1e-3, 1e-3]))
    prices[1] = bad
    with pytest.raises(ValueError, match="1 of 3 are NaN or plus infinity"):
        end_of_turn_count(prices, np.array([False, True, False]), eps=0.01)


def test_the_count_prices_minus_infinity_as_a_probability_of_zero():
    prices = np.array([-math.inf, math.log(1e-3)])
    count = end_of_turn_count(prices, np.array([False, False]), eps=0.01)
    assert count.n_priced == 2 and count.expected == pytest.approx(1e-3)


def test_the_published_thresholds_are_ascending_probabilities():
    assert list(EOT_EPS_GRID) == sorted(EOT_EPS_GRID)
    assert 0 < min(EOT_EPS_GRID) and max(EOT_EPS_GRID) < 1


def test_a_record_prices_every_served_position():
    """The count reads the reference's price and the served id rather than the report,
    so the record carries one price per served position and the indices where the reply
    ended its turn."""
    draw = _honest_draw(4, seed=5)
    draw["served"] = np.array([10, 11, 12, 13], dtype=np.int64)
    draw["p_eot_lp"] = [math.log(1e-4)] * 4
    draw["eot_ids"] = (12,)
    st = position_stats(**draw)
    rec = score_reply(st, event_key="t0:r0", n_served=4, finish_reason="stop",
                      ends_with_eot=False, at_cap=False)
    assert len(rec.u) == 4
    assert len(rec.p_eot_lp) == 4 and rec.eot_positions == (2,)
    assert ReplyReport.from_json(rec.to_json()) == rec


def _priced_record(event_key: str, *, prices, endings) -> ReplyReport:
    """A record carrying what the end-of-turn count reads and nothing else: one
    reference price per served position, and the positions the reply ended a turn at."""
    return ReplyReport(event_key=event_key, n_tokens=len(prices), n_outside=0, kl_sum=0.0, s_sum=0.0, var_sum=0.0, n_moment=0,
                       u=tuple([0.5] * len(prices)),
                       p_eot_lp=tuple(math.log(p) for p in prices),
                       eot_positions=tuple(endings), finish_reason="stop",
                       ends_with_eot=True, at_cap=False, violation=None)


def test_the_stored_form_keeps_what_the_summary_drops():
    """A calibration chooses its thresholds after every serving is scored, so its
    stored records are re-read and have to carry the per-position values. The summary
    a graded run persists is the same record without them."""
    rec = _priced_record("t0:r0", prices=[1e-3, 0.5], endings=(1,))
    full, summary = rec.to_json(), rec.summary_json()
    assert full["u"] == [0.5, 0.5] and len(full["p_eot_lp"]) == 2
    assert "u" not in summary and "p_eot_lp" not in summary
    assert summary == {k: v for k, v in full.items() if k not in ("u", "p_eot_lp")}
    assert ReplyReport.from_json(full) == rec


def test_the_pooled_count_applies_the_threshold_rather_than_the_record():
    """The threshold is applied here and not at scoring time, which is what lets a
    calibration search thresholds over records it has already scored."""
    pooled = PooledReport()
    pooled.extend([_priced_record("t0:r0", prices=[1e-3, 1e-3, 0.5], endings=(2,)),
                   _priced_record("t0:r1", prices=[1e-3, 0.02], endings=(0,))])
    assert pooled.expected_end_of_turns(0.01) == pytest.approx(3e-3)
    tight = pooled.end_of_turn(0.01)
    assert (tight.n_priced, tight.observed) == (3, 1)
    assert tight.expected == pytest.approx(3e-3)
    # A wider threshold prices one more position, so the same replies carry a larger
    # expectation; the ending at 0.5 stays outside both.
    wide = pooled.end_of_turn(0.1)
    assert (wide.n_priced, wide.observed) == (4, 1)
    assert wide.expected == pytest.approx(3e-3 + 0.02)


def test_a_violated_reply_is_outside_the_count_too():
    """A reply whose report violates the contract was never measured at all, and the
    trace fails on the violation rather than on a count taken over it."""
    pooled = PooledReport()
    pooled.add(ReplyReport(event_key="t0:r0", n_tokens=7, n_outside=0,
                           kl_sum=0.0, s_sum=0.0, var_sum=0.0, n_moment=0, u=(),
                           p_eot_lp=(), eot_positions=(), finish_reason="stop",
                           ends_with_eot=False, at_cap=False, violation="no report"))
    count = pooled.end_of_turn(0.1)
    assert (count.n_priced, count.expected, count.observed) == (0, 0.0, 0)
    assert count.p_value == 1.0
