#!/usr/bin/env python3
"""Score your server's report against the reference's, the way the gate scores it.

    python3 selfcheck.py reference.jsonl mine.jsonl
    python3 selfcheck.py --self-test

YOUR file holds one JSON object per line:

    {"id": "<prompt id from prompts.jsonl>",
     "finish_reason": "stop",
     "prompt_positions": [64, 128, ...],
     "prompt_forced": [{"token_id": 811, "logprob": -2.1, "top21": [[811, -2.1], ...]},
                       ...],
     "tokens": [{"token_id": 12345,
                 "logprob": -0.31,
                 "top21": [[12345, -0.31], [678, -1.9], ...]},
                ...]}

``tokens`` is one entry per position of the reference's continuation, in order:
``token_id`` is the reference's served token at that position, ``logprob`` is the
log-probability YOUR server assigns it there, and ``top21`` is your server's 21 most
probable tokens at that position with their log-probabilities. Your file is therefore
teacher-forced on the reference's continuation, which is what lets it be compared
position by position without a second GPU.

``prompt_forced`` is optional and is not part of the gate. It is your prefill's own
output at the prompt positions the reference sampled, which lets a divergence be
placed: if your prefill already disagrees at prompt positions, nothing about your
decode kernel is the cause.

THE REFERENCE FILE NAMES TWO PATHS
reference.jsonl gives every generated position a ``decode`` and a ``forced`` object of
that same shape. ``decode`` is what the reference streamed while it sampled the
continuation. ``forced`` is what the reference says about that same continuation when
the continuation is teacher-forced back through it. vLLM prefills and decodes with
different kernels, so the two are not equal. The gate teacher-forces a contestant's
served tokens, so ``forced`` is the one it reads and the one this script scores you
against by default; ``--against decode`` scores the other. The reference's own
decode-against-forced divergence is printed on every run: it is the floor, and no
implementation can be asked to sit below it.

One of the gate's five checks is computable this way, and this script computes that one
and no others:

  divergence      mean_kl over positions, against the ceiling below.

The other four read a null about YOUR sampler drawing YOUR served tokens: rank
uniformity, the log-score moment, the end-of-turn count, and the served-outside rate.
A teacher-forced file has no tokens your sampler drew -- the tokens are the reference's
-- so none of the four can be read from it. The outside rate is still printed, because
it says how often your head disagrees with the reference's about the token the
reference drew, but it is not the gate's check 5 and nothing here fails on it.

One more thing the divergence number is not: the gate compares a DECODE report against
the forced reference, and its ceiling is sized for that. Your teacher-forced file is a
prefill against a prefill, which is a much closer comparison. Passing here at a number
near the ceiling would be a bad sign, not a near miss.

Stdlib only, on purpose: it has to run inside a sandbox with no numeric stack.

WHAT THIS REPRODUCES, AND WHERE IT CANNOT
-----------------------------------------

The gate's reference is a full-vocabulary teacher-forcing pass, so it can quote its own
log-probability at any token id your report names. This file's reference is 21 entries
per position. The two agree exactly at a position where every atom of your partition is
one of the reference's reported ids, and that is the ordinary case for a faithful
report, because both sides' heads are the same tokens. Where an atom of yours is not
among the reference's 21, its reference probability is unknown; it is bounded above by
the smallest probability the reference did report, and this script substitutes that
bound, which makes the position's divergence a LOWER bound on the gate's. Positions
scored that way are counted and printed as ``uncovered``. A run with ``uncovered`` at
zero is the gate's own arithmetic; a run with it above zero reads low, so treat the
number as a floor.

One further difference, on every position: the gate sums the reference's tail mass in
log space over the vocabulary outside your atoms, while 21 entries admit only
``log1p(-sum)`` over the atoms. The two differ once the atoms hold all but about 1e-12
of the mass, where this file's tail is the less accurate of the two.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants, from the task's shipped calibration.json and the verifier's defaults.
# ---------------------------------------------------------------------------

#: Alternatives the contract asks for at every position.
#: hls/verify/report.py:124 (TOP_LOGPROBS_REQUESTED).
TOP_LOGPROBS_REQUESTED = 21

#: Width of a partition row. hls/verify/report.py:129 (TOP_LOGPROBS_KEPT).
TOP_LOGPROBS_KEPT = TOP_LOGPROBS_REQUESTED - 2

#: Id of a padded slot. hls/core/capabilities.py:39 (PAD_ID).
PAD_ID = -1

#: Slack a row's reported mass gets over one. hls/verify/report.py:134 (MASS_TOL).
MASS_TOL = 1e-4

#: The largest log-probability a report may carry. hls/verify/report.py:139.
MAX_LOGPROB = math.log1p(MASS_TOL)

#: The divergence a trusted implementation of this checkpoint already shows against the
#: pinned reference, as the one-sided 95% upper limit of the calibration measurement.
#: tasks/agentic_coding/defaults/calibration.json, key "kappa".
KAPPA = 0.06349320767592255

#: The rate at which a trusted stack's served token falls outside its own reported
#: atoms. calibration.json, key "outside_ceiling".
OUTSIDE_CEILING = 0.0023599434639501567

#: Headroom over both calibrated ceilings.
#: hls/verify/verifiers.py:695 (ReportFaithfulnessVerifier.tolerance, default 2.0).
TOLERANCE = 2.0

#: What a trace is actually held to.
KL_CEILING = TOLERANCE * KAPPA                 # 1.269864e-01 nats/token
OUTSIDE_LIMIT = TOLERANCE * OUTSIDE_CEILING    # 4.719887e-03, i.e. 0.47 percent

#: Smallest positive float64, the floor the tail's log is taken off.
#: hls/verify/report.py:550 (_TINY).
TINY = 5e-324


# ---------------------------------------------------------------------------
# The partition. A port of hls/verify/report.py:195 (partition_row) and the value
# check it calls, hls/verify/report.py:243 (_unreadable_values).
# ---------------------------------------------------------------------------


def unreadable_values(lps, q_emit):
    """Why one position's reported values cannot all be read as log-probabilities, or
    None when they can. hls/verify/report.py:243.

    A value that is not a number passes every bound it is compared against, and a value
    above zero is more mass than a position has. Values a hair above zero are the wire
    rounding and are allowed on the slack the mass check gives a whole row."""
    for value in (*lps, q_emit):
        if value != value:
            return "reports a log-probability that is not a number: nan"
        if value > MAX_LOGPROB:
            return f"reports a log-probability above zero: {value:.6g}"
    return None


def partition_row(top, q_emit, served):
    """The atoms to keep at one position, padded to ``TOP_LOGPROBS_KEPT``, and how many
    distinct alternatives arrived; or the reason the entry supplies no partition.
    A port of hls/verify/report.py:195.

    ``top`` is the reported ``[[token_id, logprob], ...]``. Duplicates are dropped in
    arrival order. An entry for the served token that sits strictly below every other
    entry is vLLM's appended one and is dropped too. The atoms are then the entries
    strictly above the 20th largest of what remains, ordered by log-probability
    descending and then by id ascending, with a tie group at that value tailed whole.
    Nothing here reads which token was served, which is what makes the partition
    independent of the draw."""
    ids, lps = [], []
    for alt in top:
        tid, lp = alt[0], alt[1]
        if not isinstance(tid, int) or isinstance(tid, bool):
            return "an alternative arrived without an integer token id"
        if tid not in ids:
            ids.append(tid)
            lps.append(float(lp))
    bad = unreadable_values(lps, q_emit)
    if bad is not None:
        return bad
    n_distinct = len(ids)
    if served in ids:
        j = ids.index(served)
        others = [lp for k, lp in enumerate(lps) if k != j]
        if others and lps[j] < min(others):
            del ids[j]
            del lps[j]
    threshold_rank = TOP_LOGPROBS_REQUESTED - 1
    if len(ids) < threshold_rank:
        return (f"carries {len(ids)} usable alternatives ({n_distinct} distinct as "
                f"reported), fewer than the {threshold_rank} the partition needs")
    order = sorted(range(len(ids)), key=lambda j: (-lps[j], ids[j]))
    threshold = lps[order[threshold_rank - 1]]
    keep = [j for j in order if lps[j] > threshold]
    pad = TOP_LOGPROBS_KEPT - len(keep)
    return ([ids[j] for j in keep] + [PAD_ID] * pad,
            [lps[j] for j in keep] + [float("-inf")] * pad,
            n_distinct)


# ---------------------------------------------------------------------------
# The mass arithmetic. A port of hls/verify/report.py:356 (_row_mass) and
# hls/verify/report.py:388 (_mass_violation), one position at a time.
# ---------------------------------------------------------------------------


class RowMass:
    """What one position's partition says about its own mass.

    ``q`` is each atom's probability, zero in a padded slot; ``q_sum`` their total;
    ``inside`` whether the served token is an atom; ``col`` which atom it is;
    ``q_tail`` the mass left outside the atoms; ``width`` the served token's own cell,
    its atom's mass inside and the tail outside."""

    __slots__ = ("q", "q_sum", "inside", "col", "total", "q_tail", "width")

    def __init__(self, q_ids, q_lp, q_emit, served):
        self.q = [math.exp(lp) if tid != PAD_ID else 0.0
                  for tid, lp in zip(q_ids, q_lp)]
        self.q_sum = math.fsum(self.q)
        self.col = next((j for j, tid in enumerate(q_ids)
                         if tid != PAD_ID and tid == served), None)
        self.inside = self.col is not None
        emit = math.exp(q_emit)
        # A report a hair over unit mass is rounding, so the tail floors at zero. Where
        # the served token is outside the atoms it floors at that token's own reported
        # probability instead: the server has reported both numbers and the smaller one
        # is the served token's. hls/verify/report.py:379.
        tail = max(1.0 - self.q_sum, 0.0)
        self.q_tail = tail if self.inside else max(tail, emit)
        self.total = self.q_sum + (0.0 if self.inside else emit)
        self.width = self.q[self.col] if self.inside else self.q_tail


def mass_violation(mass, pos):
    """Why this position's report is not a distribution, or None when it is.
    A port of hls/verify/report.py:388.

    Two ways to fail. The mass may exceed one by more than the wire's rounding slack.
    Or the served token's cell may have no width, which is a token no sampler could
    have drawn from what was reported."""
    if mass.total > 1.0 + MASS_TOL:
        return (f"position {pos} reports {mass.total:.6g} of probability over its "
                f"alternatives, more than the one a distribution has")
    if mass.width <= 0.0:
        return f"position {pos} reports the token it served at probability zero"
    return None


# ---------------------------------------------------------------------------
# The divergence. A port of the kl_qp half of hls/verify/report.py:417
# (position_stats), which is
#
#     sum_i q_i (log q_i - log p_i)  +  q_tail (log q_tail - log p_tail)
#
# with Q your report and P the reference's, both coarsened onto YOUR partition:
# hls/verify/report.py:443-445 names the formula and hls/verify/report.py:539-541 is
# the arithmetic. Q is the reported distribution and P the reference's, in that order;
# the divergence is of the report FROM the reference, so it is finite wherever your
# report puts mass and the reference does not put none.
# ---------------------------------------------------------------------------


def reference_row(entry):
    """The reference's log-probabilities at one position, and the ceiling that applies
    to a token it did not report.

    The mapping folds the served token's own entry in beside the alternatives: the
    serving stack reports it whether or not it ranks inside the top 21, and its
    probability is as much a fact about the reference's distribution as any
    alternative's.

    The ceiling is NOT the smallest value in that mapping. A token the reference did
    not report ranked below its true top entries, so its probability is at most the
    smallest of THOSE -- and the served token's entry is not one of them when the
    served token ranked below the width, which is the case the serving stack appends it
    for. Taking the mapping's minimum there would hand back the served token's own
    probability, which can be orders of magnitude below the true threshold, and the
    substituted term would then read HIGH rather than low. So the served token's entry
    is dropped first when it sits strictly below every other, which is the same rule
    ``partition_row`` drops it by (hls/verify/report.py:222)."""
    row = {}
    for alt in entry["top21"]:
        row.setdefault(int(alt[0]), float(alt[1]))
    served = int(entry["token_id"])
    row.setdefault(served, float(entry["logprob"]))
    ranked = dict(row)
    if served in ranked and len(ranked) > 1:
        others = [lp for tid, lp in ranked.items() if tid != served]
        if ranked[served] < min(others):
            del ranked[served]
    return row, min(ranked.values())


def position_kl(q_ids, q_lp, mass, ref_row, ref_floor):
    """This position's divergence, and whether the reference covered every atom.

    An atom the reference did not report has a probability at most the smallest one it
    ranked, since it ranked below all of those. Substituting that ceiling makes the
    atom's term ``q (log q - log p_ceiling) <= q (log q - log p_true)``, so the
    position's value is a lower bound on the gate's rather than a guess at it.

    Also returns how much of your own probability sits on atoms the reference did not
    price. That mass is what the substitution's error is weighted by, so it says how
    much the number below can be off; the count of affected positions does not, because
    a boundary atom carrying 1e-6 of your mass moves nothing."""
    covered = True
    uncovered_mass = 0.0
    terms = []
    named = []
    for tid, lp, q in zip(q_ids, q_lp, mass.q):
        if q <= 0.0:
            # An atom the report gives no mass contributes nothing, and its log is an
            # infinity that would otherwise turn the term into a NaN.
            # hls/verify/report.py:562 (_atom_sum).
            continue
        p_lp = ref_row.get(tid)
        if p_lp is None:
            covered = False
            uncovered_mass += q
            p_lp = ref_floor
        terms.append(q * (lp - p_lp))
        named.append(tid)
    if mass.q_tail > 0.0:
        log_q_tail = math.log(max(mass.q_tail, TINY))
        # The reference mass outside your atoms. The gate sums it in log space over the
        # whole vocabulary; 21 entries admit only the complement of what is named here.
        inside_p = math.fsum(math.exp(ref_row[t]) for t in named if t in ref_row)
        p_tail = 1.0 - inside_p
        if p_tail <= 0.0:
            # Your atoms account for every unit of the reference's reported mass and
            # then some. The reference still has a tail -- a vocabulary is larger than
            # 21 tokens -- so this is the arithmetic running out of resolution, not a
            # reference of no tail. Floored, and the position is reported uncovered.
            covered = False
            p_tail = TINY
        terms.append(mass.q_tail * (log_q_tail - math.log(p_tail)))
    return math.fsum(terms), covered, uncovered_mass


# ---------------------------------------------------------------------------
# Per-reply and pooled reduction. A port of hls/verify/report.py:686 (score_reply),
# hls/verify/report.py:841 (PooledReport) and hls/verify/report.py:1024
# (_clustered_se).
# ---------------------------------------------------------------------------


class Reply:
    """One prompt's measurement: its divergence total, its position count and how often
    its served token fell outside its own atoms."""

    __slots__ = ("id", "n_tokens", "n_outside", "kl_sum", "n_uncovered",
                 "uncovered_mass", "n_short")

    def __init__(self, ident):
        self.id = ident
        self.n_tokens = 0
        self.n_outside = 0
        self.kl_sum = 0.0
        self.n_uncovered = 0
        self.uncovered_mass = 0.0
        self.n_short = 0


def ref_side(record, which):
    """The reference's flat per-position entries on one of its two paths.

    reference.jsonl carries both at every generated position. ``decode`` is what the
    reference streamed while it was sampling; ``forced`` is what it says about that
    same continuation when the continuation is teacher-forced back through it. The gate
    reads the forced path, because teacher-forcing a contestant's served tokens is what
    the gate does, so ``forced`` is the default everywhere in this program."""
    if which not in ("forced", "decode"):
        raise SystemExit(f"unknown reference path {which!r}: use forced or decode")
    entries = []
    for pos, tok in enumerate(record.get("tokens") or []):
        side = tok.get(which)
        if side is None:
            raise SystemExit(
                f"{record.get('id')!r} position {pos} carries no {which!r} entry; "
                f"every generated position in reference.jsonl has both")
        entries.append(side)
    return entries


def score_positions(ident, mine_entries, ref_entries, what):
    """Measure one list of positions, or name the contract violation that stops it
    being measured.

    Returns ``(Reply, None)`` or ``(None, reason)``. The contract checks are the gate's
    own, in the gate's order: a report that does not cover the reference's positions
    exactly, a position naming a token other than the one there, a position with too few
    distinct alternatives to fix the partition's threshold, a value no log-probability
    takes, and a position whose numbers are not a distribution.
    hls/verify/report.py:276 (reply_report).

    ``what`` names the positions in the messages, because the same arithmetic scores
    generated positions and prompt positions and a reader has to know which failed."""
    served = [int(e["token_id"]) for e in ref_entries]
    if mine_entries is None:
        return None, f"carries no reported logprobs at {what} positions"
    if len(mine_entries) != len(served):
        return None, (f"{len(mine_entries)} reported {what} positions for "
                      f"{len(served)} in the reference")
    out = Reply(ident)
    for pos, (entry, ref_entry) in enumerate(zip(mine_entries, ref_entries)):
        if int(entry["token_id"]) != served[pos]:
            return None, (f"{what} position {pos} reports token "
                          f"{entry['token_id']}, the reference has {served[pos]}")
        q_emit = float(entry["logprob"])
        row = partition_row(entry["top21"], q_emit, served[pos])
        if isinstance(row, str):
            return None, f"position {pos} {row}"
        q_ids, q_lp, n_distinct = row
        mass = RowMass(q_ids, q_lp, q_emit, served[pos])
        bad = mass_violation(mass, pos)
        if bad is not None:
            return None, bad
        ref_row, ref_floor = reference_row(ref_entry)
        kl, covered, uncovered_mass = position_kl(q_ids, q_lp, mass, ref_row, ref_floor)
        out.n_tokens += 1
        out.kl_sum += kl
        out.n_outside += not mass.inside
        out.n_uncovered += not covered
        out.uncovered_mass += uncovered_mass
        out.n_short += n_distinct < TOP_LOGPROBS_REQUESTED
    return out, None


def score_reply(ident, mine, ref, against="forced"):
    """One prompt's generated positions: your report against the reference's.

    ``against`` picks which of the reference's two paths your report is held to. The
    gate uses ``forced``; ``decode`` is available so the two can be told apart."""
    return score_positions(ident, mine.get("tokens"), ref_side(ref, against),
                           "generated")


def score_prompt(ident, mine, ref):
    """One prompt's sampled PROMPT positions: your prefill against the reference's.

    Nothing has been sampled at these positions -- they are the prompt's own tokens,
    and the distribution at each is what the model says about the token that is already
    there. That makes them a check of the prefill alone, before any decode kernel has
    run. The gate never reads them; they are here so a contestant can find out whether
    a divergence is in its prefill or in its decode.

    The positions have to be the same positions: a file that sampled the prompt
    somewhere else is measuring different rows, which would read as agreement or
    disagreement that means nothing."""
    ref_positions = ref.get("prompt_positions")
    if not ref_positions:
        return None, "the reference carries no prompt positions"
    mine_positions = mine.get("prompt_positions")
    if mine_positions is None:
        return None, "carries no prompt_positions"
    if list(mine_positions) != list(ref_positions):
        return None, (f"reports {len(mine_positions)} prompt positions where the "
                      f"reference names {len(ref_positions)}, or names different ones")
    return score_positions(ident, mine.get("prompt_forced"),
                           ref.get("prompt_forced") or [], "prompt")


def clustered_se(by_reply, mean, n):
    """The standard error of a per-position mean whose positions come in replies.
    A port of hls/verify/report.py:1024.

    A reply served from a slightly wrong distribution is wrong all the way down, so its
    positions carry one piece of evidence rather than a thousand. Each reply's residual
    is its total against the pooled mean times its count; the residuals are squared,
    summed, scaled by the finite-cluster factor and divided by the position count."""
    if len(by_reply) < 2 or not n:
        return float("nan")
    residuals = [total - cnt * mean for total, cnt in by_reply]
    clusters = len(by_reply)
    return math.sqrt(math.fsum(r * r for r in residuals) * clusters / (clusters - 1)) / n


class Pooled:
    """Sum of ``Reply`` records over one run. hls/verify/report.py:841."""

    def __init__(self):
        self.replies = []
        self.violations = []

    def add(self, reply):
        self.replies.append(reply)

    @property
    def n_tokens(self):
        return sum(r.n_tokens for r in self.replies)

    @property
    def kl_sum(self):
        return math.fsum(r.kl_sum for r in self.replies)

    @property
    def mean_kl(self):
        return self.kl_sum / self.n_tokens if self.n_tokens else float("nan")

    @property
    def se_kl(self):
        return clustered_se([(r.kl_sum, r.n_tokens) for r in self.replies if r.n_tokens],
                            self.mean_kl, self.n_tokens)

    @property
    def n_outside(self):
        return sum(r.n_outside for r in self.replies)

    @property
    def outside_rate(self):
        return self.n_outside / self.n_tokens if self.n_tokens else float("nan")

    @property
    def se_outside(self):
        return clustered_se(
            [(float(r.n_outside), r.n_tokens) for r in self.replies if r.n_tokens],
            self.outside_rate, self.n_tokens)

    @property
    def n_uncovered(self):
        return sum(r.n_uncovered for r in self.replies)

    @property
    def uncovered_mass(self):
        """Your probability sitting on atoms the reference never priced, per position.

        The substitution's error is weighted by exactly this, so it is the number that
        says whether ``n_uncovered`` matters. A rate of 1e-5 means the divergence above
        is a lower bound that reads low by at most about that much times the log-ratio
        it stands in for, which is nothing against a ceiling of 0.127."""
        total = math.fsum(r.uncovered_mass for r in self.replies)
        return total / self.n_tokens if self.n_tokens else float("nan")

    @property
    def n_short(self):
        return sum(r.n_short for r in self.replies)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def load(path):
    """One JSONL file as a mapping from id to record, refusing a repeated id."""
    out = {}
    with Path(path).open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                raise SystemExit(f"{path}:{lineno}: not JSON: {exc}")
            ident = rec.get("id")
            if ident is None:
                raise SystemExit(f"{path}:{lineno}: record carries no id")
            if ident in out:
                raise SystemExit(f"{path}:{lineno}: id {ident!r} appears twice")
            out[ident] = rec
    return out


def reference_outside(reference, side):
    """The rate at which the reference's own rows on one path do not name the token
    that path's position holds.

    Scored exactly the way your file is scored, with the reference standing in for you.
    It is the number your own outside rate has to be read against: the continuation was
    sampled on the decode path, so on the forced path the sampled token is sometimes not
    among the prefill's most probable, and that is a property of the two kernels rather
    than of anybody's sampler."""
    pooled = Pooled()
    for ident, rec in reference.items():
        reply, reason = score_reply(ident, {"tokens": ref_side(rec, side)}, rec, side)
        if reason is None:
            pooled.add(reply)
    return pooled


def reference_gap(reference):
    """The reference's own decode-vs-forced divergence: pooled, and per prompt.

    The same arithmetic as everything else here, with the reference's decode rows put
    where a contestant's report would go and its forced rows as the reference. Both
    come from the same weights on the same node in the same hour; what separates them
    is that vLLM prefills and decodes with different kernels. So this is the floor. A
    contestant whose divergence is at this level is as close as the reference is to
    itself, and no implementation can be asked to do better."""
    pooled = Pooled()
    per_prompt = []
    for ident, rec in reference.items():
        reply, reason = score_reply(ident, {"tokens": ref_side(rec, "decode")},
                                    rec, "forced")
        if reason is not None:
            pooled.violations.append((ident, reason))
            continue
        pooled.add(reply)
        per_prompt.append((ident, reply.kl_sum / reply.n_tokens, reply.n_tokens))
    return pooled, per_prompt


def run(reference_path, mine_path, against="forced"):
    """Score one file against the reference and print the two checks. Returns the exit
    status: 0 when both pass, 1 when either fails or the file violates the contract."""
    reference = load(reference_path)
    mine = load(mine_path)

    missing = sorted(set(reference) - set(mine))
    extra = sorted(set(mine) - set(reference))
    pooled = Pooled()
    prompt_pooled = Pooled()
    for ident in reference:
        if ident not in mine:
            continue
        reply, reason = score_reply(ident, mine[ident], reference[ident], against)
        if reason is not None:
            pooled.violations.append((ident, reason))
            continue
        pooled.add(reply)
        # The prefill check is optional: a file that carries no prompt_forced is still
        # a complete answer to the gate, which never reads prompt positions.
        if mine[ident].get("prompt_forced") is not None:
            prompt_reply, prompt_reason = score_prompt(ident, mine[ident],
                                                       reference[ident])
            if prompt_reason is not None:
                prompt_pooled.violations.append((ident, prompt_reason))
            else:
                prompt_pooled.add(prompt_reply)

    n = pooled.n_tokens
    mean_kl, outside = pooled.mean_kl, pooled.outside_rate
    print(f"prompts scored          {len(pooled.replies)} of {len(reference)}")
    print(f"positions               {n}")
    if missing:
        print(f"missing from your file  {len(missing)}: {', '.join(missing[:5])}"
              + (" ..." if len(missing) > 5 else ""))
    if extra:
        print(f"not in the reference    {len(extra)}: {', '.join(extra[:5])}"
              + (" ..." if len(extra) > 5 else ""))
    print()
    print("check 1, divergence")
    print(f"  mean_kl               {mean_kl:.6g} +/- {pooled.se_kl:.3g} nats/token")
    print(f"  ceiling               {KL_CEILING:.6g}  "
          f"(tolerance {TOLERANCE:g} x kappa {KAPPA:.6g})")
    print(f"  utilization           {mean_kl / KL_CEILING:.1%}" if n else "")
    print()
    # Check 5 is NOT computable from a teacher-forced file, for the same reason the
    # other three are not. The gate asks whether the token a server SERVED is among the
    # atoms that server REPORTED at the same position -- a property of one sampler
    # against its own distribution. Here the token came from the reference's sampler and
    # the distribution is yours, so a rate above the limit says the two disagree about
    # the head, not that your sampler drew outside its own report. The reference's own
    # forced rows, scored the same way, sit above the limit for exactly that reason.
    ref_outside = reference_outside(reference, against)
    print("served outside (NOT the gate's check on a teacher-forced file)")
    print(f"  outside_rate          {outside:.6g} +/- {pooled.se_outside:.3g}")
    print(f"  the reference's own {against} rows, scored the same way  "
          f"{ref_outside.outside_rate:.6g}")
    print(f"  the gate's limit, for scale  {OUTSIDE_LIMIT:.6g}  "
          f"(tolerance {TOLERANCE:g} x outside_ceiling {OUTSIDE_CEILING:.6g})")
    print()
    print(f"positions the reference did not cover  {pooled.n_uncovered}"
          + ("  (mean_kl above is a LOWER bound; see the module docstring)"
             if pooled.n_uncovered else ""))
    print(f"  your mass on unpriced atoms, per position  {pooled.uncovered_mass:.3g}")
    print(f"positions reporting fewer than {TOP_LOGPROBS_REQUESTED} alternatives  "
          f"{pooled.n_short}")

    # The prefill check. Diagnostic: the gate never reads prompt positions, so nothing
    # here can fail the run. It exists to tell a prefill fault from a decode fault.
    if prompt_pooled.replies or prompt_pooled.violations:
        print()
        print("prefill (prompt positions, NOT part of the gate)")
        print(f"  prompts               {len(prompt_pooled.replies)}")
        print(f"  positions             {prompt_pooled.n_tokens}")
        print(f"  mean_kl               {prompt_pooled.mean_kl:.6g} "
              f"+/- {prompt_pooled.se_kl:.3g} nats/token")
        print(f"  positions the reference did not cover  "
              f"{prompt_pooled.n_uncovered}")
        for ident, reason in prompt_pooled.violations[:5]:
            print(f"  not measured: {ident}: {reason}")

    # The floor, from the reference against itself.
    gap, per_prompt = reference_gap(reference)
    print()
    print("the reference's own decode-vs-forced gap (the kernel-noise floor)")
    print(f"  mean_kl               {gap.mean_kl:.6g} "
          f"+/- {gap.se_kl:.3g} nats/token over {gap.n_tokens} positions")
    print(f"  as a share of the ceiling  {gap.mean_kl / KL_CEILING:.2%}")
    worst = sorted(per_prompt, key=lambda r: -r[1])[:5]
    for ident, kl, npos in worst:
        print(f"    {ident:<44} {kl:.6g} over {npos} positions")

    rejections = []
    if pooled.violations:
        rejections.append("contract_violations")
    if n == 0:
        rejections.append("no_measured_positions")
    if mean_kl > KL_CEILING:
        rejections.append("divergence")
    if pooled.violations:
        print()
        print(f"contract violations: {len(pooled.violations)}")
        for ident, reason in pooled.violations[:5]:
            print(f"  {ident}: {reason}")
    print()
    if rejections:
        print("FAIL: " + ", ".join(rejections))
        return 1
    print("PASS on the one check this file can read. Rank uniformity, the log-score "
          "moment, the end-of-turn count and the served-outside rate all read a "
          "sampler against its own report; a teacher-forced file holds no tokens your "
          "sampler drew, so none of the four were checked.")
    return 0


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _synthetic(seed=7, n_positions=64, n_replies=3, n_prompt=8):
    """Records of plausible shape, built without numpy or a model.

    Each position gets a decaying set of 21 log-probabilities that sum under one, so
    every row has a tail, and the occupying token is drawn from the atoms the partition
    keeps -- the 19 most probable -- so a faithful report of this traffic has an
    outside rate of exactly zero and the equality check can assert one.

    The records come out in a CONTESTANT's shape: flat ``tokens`` plus ``prompt_forced``
    at ``prompt_positions``. ``_as_reference`` puts one in the reference's two-sided
    shape, so the same generator feeds both sides of every case below."""
    import random
    rng = random.Random(seed)

    def row():
        ids = rng.sample(range(1000, 9000), TOP_LOGPROBS_REQUESTED)
        weights = sorted((rng.random() ** 2 for _ in ids), reverse=True)
        total = math.fsum(weights) / rng.uniform(0.80, 0.995)
        top = [[i, math.log(w / total)] for i, w in zip(ids, weights)]
        here = rng.choice(ids[:TOP_LOGPROBS_KEPT])
        return {"token_id": here,
                "logprob": dict((i, lp) for i, lp in top)[here],
                "top21": top}

    records = []
    for r in range(n_replies):
        records.append({
            "id": f"synthetic-{r}", "finish_reason": "stop",
            "prompt_positions": [64 * (k + 1) for k in range(n_prompt)],
            "prompt_forced": [row() for _ in range(n_prompt)],
            "tokens": [row() for _ in range(n_positions)]})
    return records


def _as_reference(flat, decode=None):
    """One flat record in reference.jsonl's two-sided shape.

    The reference names both paths at every generated position; a contestant's file
    names one. Given no separate decode rows the two sides are the same record, which
    is the case where the kernel-noise floor is exactly zero -- so a self-test that
    asserts the floor is zero is asserting something about the arithmetic and not about
    the data."""
    forced = flat["tokens"]
    dec = forced if decode is None else decode
    rec = {k: v for k, v in flat.items() if k != "tokens"}
    rec["tokens"] = [{"decode": d, "forced": f} for d, f in zip(dec, forced)]
    return rec


def self_test():
    """A file equal to the reference scores KL 0 and coverage 0, and a perturbed one
    does not. The second half is what makes the first mean anything: a scorer that
    returned zero unconditionally would pass the equality check alone."""
    import tempfile
    records = _synthetic()
    reference = [_as_reference(r) for r in records]
    with tempfile.TemporaryDirectory() as tmp:
        ref = Path(tmp) / "reference.jsonl"
        ref.write_text("".join(json.dumps(r) + "\n" for r in reference))
        mine_path = Path(tmp) / "mine.jsonl"
        mine_path.write_text("".join(json.dumps(r) + "\n" for r in records))

        pooled = Pooled()
        for rec, ref_rec in zip(records, reference):
            reply, reason = score_reply(rec["id"], rec, ref_rec)
            assert reason is None, reason
            pooled.add(reply)
        assert pooled.n_tokens == 64 * 3, pooled.n_tokens
        assert abs(pooled.mean_kl) < 1e-12, f"identical file scored {pooled.mean_kl}"
        assert pooled.n_uncovered == 0, pooled.n_uncovered
        print(f"identical file:  mean_kl {pooled.mean_kl:.3g}, "
              f"uncovered {pooled.n_uncovered}  OK")

        # The prompt positions are scored by the same arithmetic, and an identical file
        # scores zero there too.
        prompt_pooled = Pooled()
        for rec, ref_rec in zip(records, reference):
            reply, reason = score_prompt(rec["id"], rec, ref_rec)
            assert reason is None, reason
            prompt_pooled.add(reply)
        assert prompt_pooled.n_tokens == 8 * 3, prompt_pooled.n_tokens
        assert abs(prompt_pooled.mean_kl) < 1e-12, prompt_pooled.mean_kl
        print(f"identical prefill: mean_kl {prompt_pooled.mean_kl:.3g} over "
              f"{prompt_pooled.n_tokens} prompt positions  OK")

        # A file that sampled the prompt somewhere else is refused rather than scored.
        moved_positions = json.loads(json.dumps(records[0]))
        moved_positions["prompt_positions"][0] += 1
        reply, reason = score_prompt("moved", moved_positions, reference[0])
        assert reply is None and reason and "different ones" in reason, reason
        print(f"moved prompt positions: refused  OK")

        # With decode and forced the same record, the floor is exactly zero; with them
        # different it is not. The second half is what makes the first mean anything.
        gap, per_prompt = reference_gap({r["id"]: r for r in reference})
        assert abs(gap.mean_kl) < 1e-12, gap.mean_kl
        assert len(per_prompt) == len(reference), per_prompt
        shifted = _as_reference(records[0], decode=[
            {"token_id": e["token_id"], "logprob": e["logprob"] - 1e-3,
             "top21": [[i, lp - 1e-3] for i, lp in e["top21"]]}
            for e in records[0]["tokens"]])
        gap2, _ = reference_gap({"shifted": shifted})
        assert gap2.mean_kl > 0.0, gap2.mean_kl
        print(f"decode-vs-forced floor: 0 when equal, {gap2.mean_kl:.3g} when not  OK")

        # Perturbed: every log-probability moved, renormalized so the row is still a
        # distribution. The divergence has to be strictly positive and the arithmetic
        # has to notice.
        moved = []
        for rec in records:
            tokens = []
            for e in rec["tokens"]:
                shifted = [[i, lp * 0.85] for i, lp in e["top21"]]
                norm = math.fsum(math.exp(lp) for _i, lp in shifted)
                scale = math.log(min(1.0, 0.999 / norm))
                shifted = [[i, lp + scale] for i, lp in shifted]
                by_id = dict((i, lp) for i, lp in shifted)
                tokens.append({"token_id": e["token_id"],
                               "logprob": by_id[e["token_id"]], "top21": shifted})
            moved.append({**rec, "tokens": tokens})
        pooled2 = Pooled()
        for rec, mrec in zip(reference, moved):
            reply, reason = score_reply(rec["id"], mrec, rec)
            assert reason is None, reason
            pooled2.add(reply)
        assert pooled2.mean_kl > 1e-6, f"perturbed file scored {pooled2.mean_kl}"
        print(f"perturbed file:  mean_kl {pooled2.mean_kl:.4g}  OK (strictly positive)")

        # A row one alternative short cannot fix the partition's threshold, and the
        # contract check has to say so rather than scoring 20 atoms.
        short = json.loads(json.dumps(records[0]))
        short["tokens"][0]["top21"] = short["tokens"][0]["top21"][:19]
        reply, reason = score_reply("short", short, reference[0])
        assert reply is None and reason and "fewer than the 20" in reason, reason
        print(f"short row:       refused, {reason}  OK")

        # A row claiming more than unit mass is not a distribution. The 19 atoms have
        # to be strictly above the 20th value or the partition tails the whole tie
        # group and keeps nothing, which is the rule and not a defect: a row of 21
        # equal values is a row whose atoms carry no mass at all.
        heavy = json.loads(json.dumps(records[0]))
        ids = [i for i, _lp in heavy["tokens"][0]["top21"]]
        heavy["tokens"][0]["top21"] = [
            [i, math.log(0.1 if k < TOP_LOGPROBS_KEPT else 0.001)]
            for k, i in enumerate(ids)]
        heavy["tokens"][0]["token_id"] = ids[0]
        heavy["tokens"][0]["logprob"] = math.log(0.1)
        ref_heavy = json.loads(json.dumps(records[0]))
        ref_heavy["tokens"][0]["token_id"] = ids[0]
        reply, reason = score_reply("heavy", heavy, _as_reference(ref_heavy))
        assert reply is None and reason and "more than the one" in reason, reason
        print(f"over-unit row:   refused, {reason}  OK")

        # A row of 21 equal values keeps no atom, which is the tie rule rather than a
        # violation: the whole group sits at the threshold and is tailed. Checked so a
        # future edit to partition_row that "fixes" this is caught here.
        flat = json.loads(json.dumps(records[0]))
        ref_flat = json.loads(json.dumps(records[0]))
        flat_ids = [i for i, _lp in flat["tokens"][0]["top21"]]
        for rec in (flat, ref_flat):
            rec["tokens"][0]["top21"] = [[i, math.log(1.0 / 64)] for i in flat_ids]
            rec["tokens"][0]["token_id"] = flat_ids[0]
            rec["tokens"][0]["logprob"] = math.log(1.0 / 64)
        reply, reason = score_reply("flat", flat, _as_reference(ref_flat))
        assert reason is None, reason
        assert reply.n_outside >= 1, reply.n_outside
        print(f"tie at threshold: kept no atom, position counted outside  OK")

        # The served token outside the atoms is what check 5 counts. Rebuilt with a
        # served token the report never names, so every position is outside.
        outside = json.loads(json.dumps(records[0]))
        ref_outside = json.loads(json.dumps(records[0]))
        for e, re_ in zip(outside["tokens"], ref_outside["tokens"]):
            novel = 99991
            e["token_id"] = novel
            e["logprob"] = math.log(1e-9)
            re_["token_id"] = novel
            re_["logprob"] = math.log(1e-9)
        reply, reason = score_reply("outside", outside, _as_reference(ref_outside))
        assert reason is None, reason
        assert reply.n_outside == reply.n_tokens, (reply.n_outside, reply.n_tokens)
        print(f"all-outside:     outside_rate 1.0 over {reply.n_tokens} positions  OK")

        # The ceiling for an unreported token is the smallest of the reference's RANKED
        # entries, not the smallest value in its row. When the reference's own served
        # token ranked below its 21, its entry rides along at a probability far under
        # the ranking threshold, and using that as the ceiling makes the substituted
        # term read HIGH -- turning the lower bound into no bound at all. Caught by
        # comparing this file against the gate's own reduction, so it is pinned here.
        entry = {"token_id": 4242, "logprob": math.log(1e-12),
                 "top21": [[i, math.log(0.04)] for i in range(500, 521)]}
        _row, floor = reference_row(entry)
        assert abs(floor - math.log(0.04)) < 1e-12, (
            f"floor {floor} came from the served token, not from the ranked entries")
        # And when the served token IS one of the ranked entries, nothing is dropped.
        entry2 = {"token_id": 500, "logprob": math.log(0.04),
                  "top21": [[i, math.log(0.04)] for i in range(500, 521)]}
        row2, floor2 = reference_row(entry2)
        assert len(row2) == 21 and abs(floor2 - math.log(0.04)) < 1e-12, (row2, floor2)
        print(f"unreported ceiling: {math.exp(floor):.4g}, not the served 1e-12  OK")

        # The clustered error is the gate's, so a hand-computable case pins it.
        se = clustered_se([(2.0, 2), (0.0, 2)], 0.5, 4)
        want = math.sqrt((1.0 ** 2 + 1.0 ** 2) * 2 / 1) / 4
        assert abs(se - want) < 1e-15, (se, want)
        print(f"clustered se:    {se:.6g} == {want:.6g}  OK")

        # And the ceilings the numbers are read against.
        assert abs(KL_CEILING - 0.1269864153518451) < 1e-15, KL_CEILING
        assert abs(OUTSIDE_LIMIT - 0.0047198869279003134) < 1e-18, OUTSIDE_LIMIT
        print(f"ceilings:        kl {KL_CEILING:.6e}, outside {OUTSIDE_LIMIT:.6e}  OK")

        status = run(ref, mine_path)
        assert status == 0, status
    print("\nself-test passed")
    return 0


def main():
    ap = argparse.ArgumentParser(
        description="Score a report against the reference the way the gate does.")
    ap.add_argument("reference", nargs="?", help="reference.jsonl as shipped")
    ap.add_argument("mine", nargs="?", help="your own file, same shape")
    ap.add_argument("--self-test", action="store_true",
                    help="check this script against cases with known answers")
    ap.add_argument("--against", choices=("forced", "decode"), default="forced",
                    help="which of the reference's two paths to score against. "
                         "forced (the default) is the one the gate uses: the reference "
                         "teacher-forced on the served continuation. decode is what the "
                         "reference streamed while sampling it.")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not args.reference or not args.mine:
        ap.error("both reference.jsonl and your own file are required")
    return run(args.reference, args.mine, args.against)


if __name__ == "__main__":
    sys.exit(main())
