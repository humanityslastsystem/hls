"""What the sampling gate measures: a served reply's own report of what it sampled.

Under a sampled decode there is no single right token, so a server cannot be graded by
comparing its tokens against a reference's. It is graded instead on a *report*: at every
position it emits, the log-probability it assigned to the token it served and to that
position's most probable alternatives, with token ids. Three statistics read that report,
and together they pin both halves of the claim -- that the reported distribution is the
model's, and that the served tokens came out of the reported distribution.

Calibration measures a trusted engine's servings to find the clean divergence ceiling
(``hls.calibrate``); the deployed verifier measures a contestant's trace the same way to
grade against it (``hls.verify.verifiers.ReportFaithfulnessVerifier``). Both sides must
compute the identical summary, so the reduction is defined here, once, and both call it.
Pure numpy: no torch, and nothing in this module touches a model.

The partition
-------------

Everything is computed on a coarsening of the vocabulary into a few reported atoms and
one tail atom holding all the rest. Both divergences on that coarsening lower-bound the
full-vocabulary one, and both sides use the same coarsening, so the bound never has to be
tightened.

The rule that chooses the atoms never looks at the served token, and that is the whole
point of it. If it did, the served token's own placement would carry information about
the draw, and the two exact checks would stop being exact. The rule is forced by how the
two serving stacks build their reported lists: SGLang sends its true top-k, while vLLM
sends the served token followed by its top-k, deduplicated and cut to the requested
width -- so when the served token ranks below that width, the last true entry is
displaced by it. A rule that simply kept "the largest K" would therefore be
draw-dependent at an exact tie across the keep boundary.

So: duplicates are dropped, an entry for the served token that sits strictly below every
other entry is vLLM's appended one and is dropped, and the atoms are the entries whose
log-probability is strictly greater than the ``w - 1``-th largest of what remains, where
``w`` is the report width the contract asks for. A tie group at that threshold value is
tailed whole. With the default 21 requested, the threshold is the twentieth value,
present on the wire whatever the draw, and the partition holds at most 19 atoms. Rows
are padded to that width with ``PAD_ID``, which carries no mass.

The rule fixes the partition before the draw, but the reporter still chooses what is in
it. A server that reports rare tokens as its alternatives, leaves the served token out of
them and serves whatever it likes puts every served token in the tail: the coarsened
divergence is near zero because the atoms carry no mass, the transform is uniform
because the tail interval is nearly the whole unit interval, and the moment vanishes
with the atoms' mass. What gives it away is how often the served token falls outside the
atoms. A faithful report puts it there at the model's own tail mass past this width,
which depends on the modality and is one of the quantities calibration measures. That
attack puts it there at nearly every position, and the measured rate separates the two
while the honest rate is well under one. A task whose tail past the default width is
heavier asks for a wider report
(``ReportFaithfulnessVerifier.report_width``), which moves the threshold rank and leaves
every statistic below as it is. So the served-outside rate is measured on the trusted
stacks beside the divergence and a trace is held to it (``PooledReport.outside_rate``).
Once the served token has to be an atom at all but that rate, the transform forces the
reported tail down to match, the atoms have to carry the mass, and the divergence reads
it.

The three statistics
--------------------

``kl_qp``  The divergence of the reported distribution from the reference's, on the
           partition. Its mean over a trace is compared against a calibrated ceiling:
           two correct implementations of one checkpoint disagree by a measurable amount,
           and the ceiling is that amount. This is the test that sees a wrong
           distribution -- a different quantization, a broken kernel, the wrong weights.

``u_q``    The randomized probability integral transform of the served token under the
           reported distribution (Dunn and Smyth 1996). Exactly uniform on [0, 1) under
           an honest sampler, at any reply length and under any dependence between
           positions, so non-uniformity is a verdict rather than an asymptotic
           approximation of one -- even though the Cramer-von Mises p-value that reads
           it is asymptotic. This is the test that sees a server reporting one
           distribution and sampling from another: truncating, reweighting, or serving
           greedily while reporting a full softmax.

``s_q``    ``log q(served) + H(q)`` on the partition, whose mean is exactly zero under
           the same null, with ``var_q`` its per-position variance. Both terms come from
           the report and from nothing else, so ``z = sum(s_q) / sqrt(sum(var_q))`` is
           standard normal with nothing estimated from the data. It reads the same null
           as ``u_q`` through a different functional, and catches shifts in the served
           tokens' typicality that leave the rank distribution close to uniform.

The verifier's uniforms (``reply_uniforms``) are drawn from the reply's event key, so
they are deterministic, independent of the contestant, and identical in grading and in
calibration.

The end-of-turn count
---------------------

A fourth statistic, and the only one that does not read the report at all. Each of the
three above is diluted by the trace's length, so a deviation confined to a few hundred
positions -- ending replies early, which saves the tokens after them -- is invisible in
all three at one trace. The count is not: the reference prices ending the turn at every
position the submission served, and among the positions it prices at or below a
calibrated threshold ``eps``, the number of end-of-turn tokens the submission actually
served is compared with the number those prices predict.

Under a faithful sampler, ending the turn at a position is a Bernoulli trial at the
reference's own probability, so the count is Poisson-binomial with those means and
``poisson_binomial_upper_tail`` is its p-value, with nothing calibrated and nothing
estimated from the data. The prices are predictable rather than fixed in advance: each
one follows from the prefix the submission has already served, and a reply ends where
its own trial fires. So the tail is the law of the count given the prices the trace
realized; the Poisson tail at the same mean is the larger of the two and bounds it from
above, which is the statement that does not condition on them.

The threshold only decides which positions are cheap enough to be evidence; it is
chosen by declaring the expected count (``EOT_EPS_GRID``, and ``hls.calibrate``).
One-sided: serving fewer end-of-turn tokens than expected is not a deviation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from hls.core.capabilities import PAD_ID
from hls.core.payloads import TextPayload, TokenLogprob

#: Alternatives the contract asks the server for at every position, unless a gate
#: declares a wider report (``ReportFaithfulnessVerifier.report_width``). One more than
#: the partition keeps, so the keep threshold is a value that is on the wire whatever
#: the draw (see the module docstring).
TOP_LOGPROBS_REQUESTED: int = 21

#: Width of a partition row at the default report width: at most this many atoms, the
#: entries strictly above the ``TOP_LOGPROBS_REQUESTED - 1``-th largest reported value.
#: Rows shorter than this are padded with ``PAD_ID``. A report of width ``w`` keeps rows
#: of ``w - 2`` (``kept_width``).
TOP_LOGPROBS_KEPT: int = TOP_LOGPROBS_REQUESTED - 2


def kept_width(width: int) -> int:
    """Partition row width for a report that asks for ``width`` alternatives."""
    return width - 2


#: How far a report's mass may exceed one before the position violates the contract.
#: The server's log-probabilities are rounded on the wire, so an exact sum is not
#: available.
MASS_TOL: float = 1e-4

#: The largest log-probability a report may carry. A probability is at most one, so a
#: log-probability is at most zero, and the slack above it is the rounding the mass
#: check already allows -- spent here on a single token rather than on a whole row.
_MAX_LOGPROB: float = math.log1p(MASS_TOL)

#: Decimals the per-position uniforms are recorded to. They are read only through a
#: distributional test over many positions, and the record carries one per served token.
_U_DECIMALS: int = 6

#: Candidate thresholds on the reference's end-of-turn probability, ascending. A
#: calibration picks the largest of these at which the expected count over the task's
#: own traffic stays under the declared maximum. A fixed grid rather than a solved
#: value: the threshold is published to contestants, and a handful of round numbers is
#: what a brief can state and a reader can recognize.
EOT_EPS_GRID: tuple[float, ...] = (1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1, 0.3)

__all__ = [
    "EOT_EPS_GRID",
    "MASS_TOL",
    "PAD_ID",
    "TOP_LOGPROBS_KEPT",
    "TOP_LOGPROBS_REQUESTED",
    "EndOfTurnCount",
    "kept_width",
    "PooledReport",
    "ReplyReport",
    "ReportRows",
    "cramer_von_mises_uniform",
    "end_of_turn_count",
    "partition_row",
    "poisson_binomial_upper_tail",
    "position_stats",
    "reply_report",
    "reply_uniforms",
    "score_reply",
]


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReportRows:
    """One reply's report as parallel rows, ready for ``position_stats``.

    ``q_ids`` and ``q_lp`` are ``n_served`` rows of ``kept_width(width)`` atoms, padded
    with ``PAD_ID`` / minus infinity; ``q_emit`` is the served token's own reported
    log-probability at each position. ``n_short`` counts positions whose report carried
    fewer distinct alternatives than the contract asked for, which is a description of
    the serving stack rather than a violation -- the partition is still well defined
    there, because it needs only the threshold rank."""

    q_ids: list[list[int]]
    q_lp: list[list[float]]
    q_emit: list[float]
    n_short: int


def partition_row(entry: TokenLogprob, served: int, *,
                  width: int = TOP_LOGPROBS_REQUESTED) -> tuple[list[int], list[float], int] | str:
    """The atoms to keep at one position, padded to ``kept_width(width)``, and how many
    distinct alternatives arrived; or the reason the entry cannot supply a partition.
    ``width`` is the number of alternatives the contract asks for.

    Every reported value is read as a log-probability first: the alternatives' and the
    served token's own. Duplicates go next, since a server may concatenate the served
    token onto its top-k without deduplicating. A served-token entry strictly below
    every other entry is vLLM's appended one and goes after that. The atoms are then the
    entries strictly above the ``width - 1``-th largest remaining
    value, ordered by log-probability descending then id ascending; a tie group at that
    value is tailed whole. Nothing here reads which token was served, which is what
    makes the partition draw-independent.

    The ids that come back are distinct, which is the precondition the reduction
    downstream reads them under (``position_stats``)."""
    ids: list[int] = []
    lps: list[float] = []
    for alt in entry.top_logprobs or []:
        if alt.token_id is None:
            return "an alternative arrived without a token id"
        if alt.token_id not in ids:
            ids.append(alt.token_id)
            lps.append(alt.logprob)
    unreadable = _unreadable_values(lps, entry.logprob)
    if unreadable is not None:
        return unreadable
    n_distinct = len(ids)
    if served in ids:
        j = ids.index(served)
        others = [lp for k, lp in enumerate(lps) if k != j]
        if others and lps[j] < min(others):
            del ids[j], lps[j]
    threshold_rank = width - 1
    if len(ids) < threshold_rank:
        # The usable count, since the served token's entry may have been dropped just
        # above. Quoting only what arrived reads as a contradiction at the boundary --
        # "carries 20 distinct alternatives, fewer than the 20 the partition needs".
        return (f"carries {len(ids)} usable alternatives ({n_distinct} distinct as "
                f"reported), fewer than the {threshold_rank} the partition needs")
    order = sorted(range(len(ids)), key=lambda j: (-lps[j], ids[j]))
    threshold = lps[order[threshold_rank - 1]]
    keep = [j for j in order if lps[j] > threshold]
    pad = kept_width(width) - len(keep)
    return ([ids[j] for j in keep] + [PAD_ID] * pad,
            [lps[j] for j in keep] + [float("-inf")] * pad,
            n_distinct)


def _unreadable_values(lps: Sequence[float], q_emit: float) -> str | None:
    """Why one position's reported values cannot all be read as log-probabilities,
    naming the first that cannot, or ``None`` when they can.

    The wire's own type admits anything a float holds, including values no probability
    has. A value that is not a number compares false against every bound, so it passes
    each check the mass arithmetic makes and lands in a statistic instead; a value above
    zero exponentiates to more mass than a position has, and far enough above it
    overflows. Both are read off the values here, before any of that arithmetic runs.

    Values a hair above zero are the wire rounding its log-probabilities and are
    allowed, on the same slack the mass check gives a whole row. Minus infinity is
    allowed too: it is a claim of zero probability, which a stack that masks tokens
    before sampling makes for every token it masked, and the mass arithmetic reads it as
    an atom of no mass. Made about the served token it is a different claim, and the
    mass check is what answers it (``_mass_violation``).

    A sum that is not a number holds a value that is not, and the largest value bounds
    every other, so the row is read whole first and value by value only once that has
    failed. A reply carries a row per served token and one more value per row than the
    report width, which is a loop worth not running."""
    total = sum(lps) + q_emit
    if (total == total and q_emit <= _MAX_LOGPROB
            and max(lps, default=0.0) <= _MAX_LOGPROB):
        return None
    for value in (*lps, q_emit):
        if value != value:
            return "reports a log-probability that is not a number: nan"
        if value > _MAX_LOGPROB:
            return f"reports a log-probability above zero: {value:.6g}"
    return None


def reply_report(payload: TextPayload, served: Sequence[int], *,
                 width: int = TOP_LOGPROBS_REQUESTED) -> ReportRows | str:
    """One reply's report as parallel rows, or the reason it violates the contract.
    ``width`` is the number of alternatives the contract asks for at every position.

    Exactly one of the two comes back. A report that does not cover the served tokens
    exactly, or whose entry at a position names a different token than the one served, is
    not the report the emission came from -- which is the premise of the whole check --
    so the reply fails rather than being scored on the part that lines up. Extra
    positions fail for the same reason as missing ones: a server describing positions it
    did not serve is describing a different decode, which is what a reply truncated after
    the report was taken looks like. Numbers that are not a distribution over the
    position fail for that reason too (``_mass_violation``), which is one pass over the
    assembled rows rather than a check per position."""
    n = len(served)
    if payload.logprobs is None:
        return "the reply carries no reported logprobs"
    if len(payload.logprobs) != n:
        return f"{len(payload.logprobs)} reported positions for {n} served tokens"
    q_ids: list[list[int]] = []
    q_lp: list[list[float]] = []
    q_emit: list[float] = []
    n_short = 0
    for pos in range(n):
        entry = payload.logprobs[pos]
        if entry.token_id is not None and entry.token_id != served[pos]:
            return f"position {pos} reports token {entry.token_id}, served {served[pos]}"
        row = partition_row(entry, served[pos], width=width)
        if isinstance(row, str):
            return f"position {pos} {row}"
        n_short += row[2] < width
        q_ids.append(row[0])
        q_lp.append(row[1])
        q_emit.append(entry.logprob)
    shape = (n, kept_width(width))
    bad = _mass_violation(_row_mass(
        np.asarray(q_ids, dtype=np.int64).reshape(shape),
        np.asarray(q_lp, dtype=np.float64).reshape(shape),
        np.asarray(q_emit, dtype=np.float64),
        np.asarray(served, dtype=np.int64)))
    if bad is not None:
        return bad
    return ReportRows(q_ids=q_ids, q_lp=q_lp, q_emit=q_emit, n_short=n_short)


@dataclass(frozen=True)
class _RowMass:
    """What one reply's partition rows say about their own mass, per position.

    Two readers share it: the contract check that fails a reply whose report is not a
    distribution (``_mass_violation``), and the reduction that measures one whose report
    is (``position_stats``). One arithmetic rather than two, because the check decides
    exactly which rows the reduction is ever handed, and a second transcription of it
    would decide something slightly different.

    Every field is an ``[n]`` array over the reply's positions, except ``valid`` and
    ``q``, which are ``[n, kept_width(width)]`` over the atoms.

    Attributes:
        valid:  the slot holds an atom rather than padding.
        q:      each atom's reported probability, zero in a padded slot.
        q_sum:  the atoms' mass.
        inside: the served token is one of the atoms.
        col:    the column its atom sits in; arbitrary, and unused, where it is not one.
        total:  the mass the report accounts for: its atoms, plus the served token's own
                entry when that token is outside them.
        q_tail: the mass left outside the atoms.
        width:  the width of the served token's own cell -- its atom's mass inside, the
                tail outside. The transform is drawn across this, so a cell of no width
                is a token the report says could not have been served.
    """

    valid: np.ndarray
    q: np.ndarray
    q_sum: np.ndarray
    inside: np.ndarray
    col: np.ndarray
    total: np.ndarray
    q_tail: np.ndarray
    width: np.ndarray


def _row_mass(q_ids: np.ndarray, q_lp: np.ndarray, q_emit: np.ndarray,
              served: np.ndarray) -> _RowMass:
    """The mass arithmetic of one reply's partition rows, vectorized over its positions.

    A padded slot (id ``PAD_ID``, log-probability minus infinity) is an atom of no mass:
    it matches no served token and adds nothing to any sum. A report a hair over unit
    mass is rounding, so the tail floors at zero rather than going negative. Where the
    served token is outside the atoms the tail is floored at that token's own reported
    probability instead -- a server whose top-k rounds to unit mass and then serves a
    token of probability 1e-12 has reported both numbers, and the smaller one is the
    served token's.

    The rows are expected to carry distinct ids per position and values that are
    log-probabilities, both of which ``partition_row`` establishes: a repeated id would
    have its mass counted twice here, and a value that is not a number would spread from
    one position's atom to that position's mass and tail, where it compares false
    against every bound below."""
    valid = q_ids != PAD_ID
    q = np.where(valid, np.exp(q_lp), 0.0)
    q_sum = q.sum(axis=1)
    match = valid & (q_ids == served[:, None])
    inside = match.any(axis=1)
    emit = np.exp(q_emit)
    q_tail = np.maximum(1.0 - q_sum, 0.0)
    q_tail = np.where(inside, q_tail, np.maximum(q_tail, emit))
    col = match.argmax(axis=1)
    q_at = q[np.arange(len(q_ids)), col]
    return _RowMass(valid=valid, q=q, q_sum=q_sum, inside=inside, col=col,
                    total=q_sum + np.where(inside, 0.0, emit), q_tail=q_tail,
                    width=np.where(inside, q_at, q_tail))


def _mass_violation(mass: _RowMass) -> str | None:
    """Why a reply's report is not a distribution at one of its positions, naming the
    first such position, or ``None`` when every position's is one.

    Two ways a position fails. Reported values are rounded on the wire, so the mass is
    allowed ``MASS_TOL`` of slack over one, and past that the row claims more
    probability than a position has. And the served token's cell has to have width:
    a token the report gives no probability, whether it is one of the atoms or sits in
    a tail the atoms leave nothing of, is a token no sampler could have drawn from what
    was reported.

    Both are read off the arithmetic rather than off the reported values themselves,
    because a log-probability far below the float range passes every check on the value
    and still exponentiates to zero."""
    bad = np.flatnonzero((mass.total > 1.0 + MASS_TOL) | (mass.width <= 0.0))
    if not bad.size:
        return None
    pos = int(bad[0])
    if mass.total[pos] > 1.0 + MASS_TOL:
        return (f"position {pos} reports {mass.total[pos]:.6g} of probability over its "
                f"alternatives, more than the one a distribution has")
    return f"position {pos} reports the token it served at probability zero"


# ---------------------------------------------------------------------------
# The per-position statistics
# ---------------------------------------------------------------------------


def position_stats(*, served: Sequence[int], q_ids, q_lp, q_emit,
                   p_on_q, p_tail_lp, p_eot_lp, eot_ids: Iterable[int],
                   v) -> dict[str, np.ndarray]:
    """Reduce one reply's report against the reference rows at the same positions.

    Every argument is parallel over the reply's ``n`` served positions, and passing them
    by keyword is enforced because they are same-shaped arrays whose order a caller
    could silently get wrong:

    - ``served``: the served token ids.
    - ``q_ids``, ``q_lp``: the partition, ``[n, kept_width(width)]``, padded with
      ``PAD_ID`` / minus infinity. A row's ids have to be distinct, which is what
      ``partition_row`` builds and what the mass here is summed as; a row that repeats
      one is refused rather than measured with that atom counted twice.
    - ``q_emit``: the served token's own reported log-probability.
    - ``p_on_q``: the reference's log-probabilities at ``q_ids``
      (``ReferenceLogprobs``), minus infinity in padded slots.
    - ``p_tail_lp``: the log of the reference mass outside ``q_ids``.
    - ``p_eot_lp``: the log of the reference's mass on the end-of-turn ids, over the
      whole vocabulary.
    - ``eot_ids``: the checkpoint's end-of-turn ids.
    - ``v``: uniforms on [0, 1), one per position (``reply_uniforms``).

    Returns float64 ``[n]`` arrays plus two boolean ``[n]`` masks:

    - ``kl_qp``: the divergence of the report from the reference on the partition,
      ``sum_i q_i (log q_i - log p_i) + q_tail (log q_tail - log p_tail)``.
    - ``u_q``: the randomized probability integral transform of the served token under
      the report, uniform on [0, 1) iff the server sampled what it reported. A served
      token outside the atoms falls in the tail interval.
    - ``s_q``: ``log q(served) + H(q)`` on the partition, zero-mean under the same null.
      ``log q(served)`` is the served token's own atom, or the tail's log-mass when the
      token is not an atom; both terms come from the report and from nothing else, so
      the mean is exactly zero rather than zero up to what the wire's per-token field
      says.
    - ``var_q``: the variance of ``s_q`` under the report, its standard error.
    - ``outside``: the served token is not one of the atoms.
    - ``p_eot_lp``: ``p_eot_lp`` back, validated against the position count.
    - ``served_eot``: the served token is one of the checkpoint's end-of-turn ids.

    The last two are the end-of-turn count's evidence, and they read the reference's
    price and the served id rather than the report.

    Raises ``ValueError`` on a shape mismatch, on a row that repeats an id, or on a row
    whose numbers are not a distribution (``_mass_violation``, which ``reply_report``
    fails a reply for before it gets here, reading the same arithmetic), and
    ``RuntimeError`` when a transform lands outside [0, 1], which is an arithmetic fault
    rather than a report to clamp."""
    served = np.asarray(served, dtype=np.int64)
    n = len(served)
    q_ids = np.asarray(q_ids, dtype=np.int64)
    if q_ids.ndim != 2 or len(q_ids) != n:
        raise ValueError(
            f"q_ids must be one partition row per served token ({n} rows), got shape "
            f"{q_ids.shape}")
    q_lp = np.asarray(q_lp, dtype=np.float64)
    if q_lp.shape != q_ids.shape:
        raise ValueError(f"q_lp must have the partition's shape {q_ids.shape}, got "
                         f"{q_lp.shape}")
    for name, arr in (("q_emit", q_emit), ("p_tail_lp", p_tail_lp),
                      ("p_eot_lp", p_eot_lp), ("v", v)):
        if len(arr) != n:
            raise ValueError(f"{name} must hold {n} values, got {len(arr)}")
    q_emit = np.asarray(q_emit, dtype=np.float64)
    p_tail_lp = np.asarray(p_tail_lp, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    p_on_q = np.asarray(p_on_q, dtype=np.float64)
    if p_on_q.shape != q_ids.shape:
        raise ValueError(f"p_on_q must have the partition's shape {q_ids.shape}, got "
                         f"{p_on_q.shape}")

    # The mass below is summed over a row's ids as if they were distinct, which is what
    # partition_row builds: one repeated id is one atom whose probability would be
    # counted once per copy. Padded slots all carry PAD_ID and repeat by design.
    ordered = np.sort(q_ids, axis=1)
    repeated = np.flatnonzero(((np.diff(ordered, axis=1) == 0)
                               & (ordered[:, 1:] != PAD_ID)).any(axis=1))
    if repeated.size:
        raise ValueError(
            f"position {int(repeated[0])} reports one id twice, so its mass would be "
            f"counted twice; rows reach this reduction deduplicated (partition_row)")
    mass = _row_mass(q_ids, q_lp, q_emit, served)
    # A row that is not a distribution has no reduction, and reply_report fails the
    # reply for one before it reaches this arithmetic, so meeting one here means a
    # caller reduced a report nothing checked.
    bad = _mass_violation(mass)
    if bad is not None:
        raise ValueError(
            f"the report cannot be reduced: {bad}; reply_report fails a reply carrying "
            f"such a position rather than reducing it")
    valid, q, q_sum = mass.valid, mass.q, mass.q_sum
    inside, q_tail, width = mass.inside, mass.q_tail, mass.width
    has_tail = q_tail > 0.0
    # The log is floored off zero because both branches of a where() are evaluated; the
    # substitute is only ever multiplied by a q_tail of zero.
    log_q_tail = np.where(has_tail, np.log(np.maximum(q_tail, _TINY)), 0.0)

    rows = np.arange(n)
    lq_at = q_lp[rows, mass.col]
    before = valid & ((q_lp > lq_at[:, None])
                      | ((q_lp == lq_at[:, None]) & (q_ids < served[:, None])))
    # Atoms in report order (log-probability descending, ties by id ascending) with the
    # tail last as one interval: the mass strictly before the served token's cell. The
    # cell's own width came off the same arithmetic as the check above.
    f_lt = np.where(inside, (q * before).sum(axis=1), q_sum)

    raw = f_lt + v * width
    # The intervals of a report this reduction accepted tile [0, q_sum] and nothing
    # else, so a transform outside that is arithmetic gone wrong, not a report to clamp.
    if raw.size and (raw.min() < -1e-6 or raw.max() > 1.0 + MASS_TOL):
        raise RuntimeError(f"u_q outside [0, 1]: min {raw.min()}, max {raw.max()}")
    # Rounding can put the transform a hair past 1; the test needs [0, 1).
    u_q = np.clip(raw, 0.0, 1.0 - 1e-12)

    has_mass = q > 0.0
    h_q = -_atom_sum(q, q_lp) - q_tail * log_q_tail
    s_q = np.where(inside, lq_at, log_q_tail) + h_q
    var_q = (_atom_sum(q, (_finite(q_lp, has_mass) + h_q[:, None]) ** 2)
             + q_tail * (log_q_tail + h_q) ** 2)
    # The log-ratio per atom, differenced only where the report gives the atom mass: a
    # padded atom is minus infinity in both rows, and that difference is not a number.
    # The tail's reference log-mass is substituted the same way where there is no tail.
    ratio = _finite(q_lp, has_mass) - _finite(p_on_q, has_mass)
    kl_qp = _atom_sum(q, ratio) + np.where(
        has_tail, q_tail * (log_q_tail - _finite(p_tail_lp, has_tail)), 0.0)

    return {"kl_qp": kl_qp, "u_q": u_q, "s_q": s_q, "var_q": var_q,
            "outside": ~inside,
            "p_eot_lp": np.asarray(p_eot_lp, dtype=np.float64),
            "served_eot": np.isin(served, np.fromiter(eot_ids, dtype=np.int64))}


#: Smallest positive float64, the floor the tail's log is taken off.
_TINY: float = float(np.finfo(np.float64).tiny)


def _finite(a: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``a`` with the values outside ``keep`` replaced by zero.

    The replaced entries are always multiplied by a zero weight afterwards. Replacing
    them before the arithmetic rather than after is what keeps an infinity from turning
    a dropped term into a NaN that no later mask can remove."""
    return np.where(keep, a, 0.0)


def _atom_sum(w: np.ndarray, term: np.ndarray) -> np.ndarray:
    """Sum ``w * term`` over the atoms that carry mass, one value per row.

    An atom the report gives no mass contributes nothing to any of these sums, but its
    ``term`` is an infinity, and zero times an infinity is not a number."""
    keep = w > 0.0
    return (np.where(keep, w, 0.0) * _finite(term, keep)).sum(axis=1)


def reply_uniforms(event_key: str, n: int) -> np.ndarray:
    """The ``n`` uniforms on [0, 1) that randomize one reply's transforms.

    Drawn from the reply's own event key, so they are the same in grading and in
    calibration, the same on a re-score of a persisted trace, and outside the
    contestant's control -- a server that could predict them could place the served
    token inside its reported cell to make the transforms look uniform."""
    seed = int.from_bytes(hashlib.sha256(event_key.encode()).digest()[:8], "big")
    return np.random.default_rng(seed).random(n)


# ---------------------------------------------------------------------------
# The per-reply record and the pooled measurement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplyReport:
    """One served reply's measurement, in the form both sides persist per event.

    Sums rather than per-position values, except the transforms: the divergence's mean
    and its reply-clustered standard error both need only a reply's total and its count,
    and the moment test needs only the two sums, while the uniformity test needs the
    values themselves. Every served position is measured, so ``n_tokens == len(u)`` on
    any record this module produced.

    A reply with a ``violation`` was never measured: its report does not describe the
    emission it came with, so every measured count below is zero and the trace fails on
    the violation itself. What the reply said about itself is still recorded."""

    #: ``hls.core.trace.event_key`` of the request.
    event_key: str
    #: Served tokens, every one of them measured on a record with no ``violation``.
    n_tokens: int
    #: Positions whose served token was not one of the reported atoms.
    n_outside: int
    #: Sum of the coarsened divergence over the served positions.
    kl_sum: float
    #: Sum of the log-score moment over the positions that carry one.
    s_sum: float
    #: Sum of that moment's variance over the same positions.
    var_sum: float
    #: Positions the two moment sums are over: those with a finite, strictly positive
    #: variance. A position whose report is a point mass with no tail has both terms
    #: exactly zero and contributes to neither, so the count belongs beside them.
    n_moment: int
    #: The per-position transforms, one per served token, to ``_U_DECIMALS``.
    u: tuple[float, ...]
    #: Per served position, the log of the reference's probability that the model ends
    #: its turn there, as the reference reported it. Kept per position rather than
    #: reduced, so the count can be recomputed at any threshold from a stored record.
    #: A reply with an ``implied_eot`` has one more entry, for the position after its
    #: last served token.
    p_eot_lp: tuple[float, ...]
    #: Indices into ``p_eot_lp`` of the positions whose served token is one of the
    #: checkpoint's end-of-turn ids. An index list rather than a mask: a reply ends the
    #: turn once or not at all, so this is a handful of numbers against thousands.
    eot_positions: tuple[int, ...]
    #: The reply's own reported finish reason, recorded for audit. The ending rule is
    #: enforced by the verifier, which knows the request's cap and the checkpoint's
    #: end-of-turn ids; nothing here reads this.
    finish_reason: str | None
    #: Whether the reply's last served id is one of the checkpoint's end-of-turn ids.
    ends_with_eot: bool
    #: Whether the served length reached the ``max_tokens`` the harness stamped.
    at_cap: bool
    #: Why the reply's report violates the contract, or ``None`` when it does not.
    violation: str | None
    #: The end-of-turn id this reply is scored as having served one position past its
    #: last token, or ``None``. Set by the verifier's implied end-of-turn rule for a
    #: normal stop that did not serve one
    #: (``ReportFaithfulnessVerifier.implied_end_of_turn``). The position counts in the
    #: end-of-turn count and in nothing else, since the server reported nothing there.
    implied_eot: int | None = None

    def to_json(self) -> dict:
        """Plain-JSON form, one key per field.

        Carries the two per-position arrays, which is what a calibration's stored
        servings need: their thresholds are chosen after every serving is scored, so
        the pooled uniformity and the end-of-turn count are recomputed from these
        records. A graded run persists ``summary_json`` instead."""
        return {**self.summary_json(), "u": list(self.u),
                "p_eot_lp": list(self.p_eot_lp)}

    def summary_json(self) -> dict:
        """The same record without ``u`` and ``p_eot_lp``.

        Both are one float per position, so on a graded drive they are the largest
        thing a verdict would carry -- roughly two floats per served token, against a
        few dozen bytes for everything else here -- and nothing reads them back: a
        graded trace is measured once, at the deployed thresholds, and the statistics
        over both arrays are already in the verdict's trace-level diagnostics. On a
        reply that ended on its own, neither length is lost: both are ``n_tokens``, and
        ``p_eot_lp`` one more when ``implied_eot`` is set. A reply the harness cut
        mid-stream keeps its served count and its ``u``, but its end-of-turn fields are
        empty, since it chose no ending to count. A reply that was never measured
        carries both arrays empty: either ``violation`` names why, or it was cut before
        its first token and ``n_tokens`` is zero.

        ``implied_eot`` appears only when it is set."""
        out = {"event_key": self.event_key, "n_tokens": self.n_tokens,
               "n_outside": self.n_outside,
               "kl_sum": self.kl_sum, "s_sum": self.s_sum, "var_sum": self.var_sum,
               "n_moment": self.n_moment,
               "eot_positions": list(self.eot_positions),
               "finish_reason": self.finish_reason,
               "ends_with_eot": self.ends_with_eot, "at_cap": self.at_cap,
               "violation": self.violation}
        if self.implied_eot is not None:
            out["implied_eot"] = self.implied_eot
        return out

    @classmethod
    def from_json(cls, d: dict) -> "ReplyReport":
        """Inverse of ``to_json``, reading the keys it needs and ignoring any other. A
        ``summary_json`` record
        does not round-trip: its two arrays are gone, and reading it back as empty would
        report a reply that measured no positions."""
        return cls(event_key=d["event_key"], n_tokens=d["n_tokens"],
                   n_outside=d["n_outside"],
                   kl_sum=d["kl_sum"], s_sum=d["s_sum"], var_sum=d["var_sum"],
                   n_moment=d["n_moment"], u=tuple(d["u"]),
                   p_eot_lp=tuple(d["p_eot_lp"]),
                   eot_positions=tuple(d["eot_positions"]),
                   finish_reason=d["finish_reason"], ends_with_eot=d["ends_with_eot"],
                   at_cap=d["at_cap"], violation=d["violation"],
                   implied_eot=d.get("implied_eot"))


def score_reply(stats: dict[str, np.ndarray], *, event_key: str, n_served: int,
                finish_reason: str | None, ends_with_eot: bool,
                at_cap: bool, count_end_of_turn: bool = True,
                implied_eot: tuple[int, float] | None = None) -> ReplyReport:
    """Reduce one reply's ``position_stats`` output to its ``ReplyReport``.

    Every served position enters the divergence sum, because a report that could not be
    read as a distribution at one of them failed the whole reply upstream
    (``reply_report``). The moment sums are narrower, over the positions that carry a
    variance, because the moment test's numerator and denominator have to be over the
    same set.

    ``count_end_of_turn`` is False for a reply that did not end, one the harness cut
    mid-stream, which leaves the end-of-turn fields empty. The count asks where a
    server chose to end its turns, and such a reply chose nothing: its served endings
    are not endings it made, and pricing its positions would put trials in the null
    that were never at risk of firing. Out of both sides, and out of the record, so a
    threshold searched over stored records reads it the same way.

    ``implied_eot`` is the end-of-turn id a normal stop is scored as having served at
    the position after its last token, with the reference's log end-of-turn mass
    there. That position joins the end-of-turn count as a served ending, priced like
    any other, and stays out of every report statistic: the server reported nothing
    there."""
    u_q = stats["u_q"]
    if len(u_q) != n_served:
        raise ValueError(
            f"event {event_key!r}: the reduction returned {len(u_q)} positions for "
            f"{n_served} served tokens; the report is scored one position per token")
    var_q, s_q = stats["var_q"], stats["s_q"]
    moment = np.isfinite(var_q) & np.isfinite(s_q) & (var_q > 0)
    if implied_eot is not None and not count_end_of_turn:
        raise ValueError(
            f"event {event_key!r}: an end-of-turn was implied for a reply that did not "
            f"end; a cut reply has no ending to complete")
    p_eot_lp = stats["p_eot_lp"].tolist() if count_end_of_turn else []
    eot_positions = ([int(i) for i in np.flatnonzero(stats["served_eot"])]
                     if count_end_of_turn else [])
    if implied_eot is not None:
        price = float(implied_eot[1])
        if math.isnan(price) or price == math.inf:
            raise ValueError(
                f"event {event_key!r}: the reference priced the implied end of turn at {price}; "
                f"an end-of-turn price is a finite log-probability or minus infinity")
        p_eot_lp.append(price)
        eot_positions.append(n_served)
    return ReplyReport(
        event_key=event_key, n_tokens=n_served,
        n_outside=int(stats["outside"].sum()),
        kl_sum=float(stats["kl_qp"].sum()),
        s_sum=float(s_q[moment].sum()), var_sum=float(var_q[moment].sum()),
        n_moment=int(moment.sum()),
        u=tuple(round(x, _U_DECIMALS) for x in u_q),
        p_eot_lp=tuple(p_eot_lp), eot_positions=tuple(eot_positions),
        finish_reason=finish_reason, ends_with_eot=ends_with_eot, at_cap=at_cap,
        violation=None,
        implied_eot=int(implied_eot[0]) if implied_eot is not None else None)


@dataclass(frozen=True)
class EndOfTurnCount:
    """How often a measurement ended a turn where the reference did not expect it.

    A position is *priced* when the reference gives the checkpoint's end-of-turn ids a
    probability of at most ``eps`` there. Over the priced positions, ``expected`` is the
    sum of those probabilities -- the number of end-of-turn tokens a faithful sampler
    would serve among them -- and ``observed`` is the number it actually served.

    ``p_value`` is ``P(X >= observed)`` under the null that each priced position was an
    independent Bernoulli trial at the reference's own probability, whose sum is
    Poisson-binomial. Nothing is estimated from the data and nothing is calibrated;
    ``eps`` only decides which positions are cheap enough to be evidence. The prices are
    predictable rather than fixed in advance -- each follows from the prefix already
    served, and a reply ends where its trial fires -- so this is the law of the count
    given the prices the measurement realized, which the Poisson tail at the same mean
    bounds from above."""

    eps: float
    n_priced: int
    expected: float
    observed: int
    p_value: float

    def to_json(self) -> dict:
        """Plain-JSON form, one key per field, with ``n_priced`` spelled out as
        ``priced_positions`` for a reader of the verdict."""
        return {"eps": self.eps, "priced_positions": self.n_priced,
                "expected": self.expected, "observed": self.observed,
                "p_value": self.p_value}


def end_of_turn_count(p_eot_lp: np.ndarray, served_eot: np.ndarray, *,
                      eps: float) -> EndOfTurnCount:
    """The end-of-turn count at threshold ``eps`` over positions pooled from any number
    of replies.

    ``p_eot_lp`` is the reference's log end-of-turn probability at each position and
    ``served_eot`` whether the server ended its turn there; both are parallel over every
    served position of the measurement (``ReplyReport``)."""
    p_eot_lp = np.asarray(p_eot_lp, dtype=np.float64)
    served_eot = np.asarray(served_eot, dtype=bool)
    if p_eot_lp.shape != served_eot.shape:
        raise ValueError(
            f"the end-of-turn count needs one price per served position: got "
            f"{p_eot_lp.shape} prices for {served_eot.shape} served flags")
    # A NaN compares false with any threshold, so it would leave its position out of both
    # sides of the count; minus infinity is a probability of zero, which is a price.
    unpriceable = np.isnan(p_eot_lp) | (p_eot_lp == np.inf)
    if unpriceable.any():
        raise ValueError(
            f"the end-of-turn count needs a finite log-probability or minus infinity at every "
            f"position: {int(unpriceable.sum())} of {p_eot_lp.size} are NaN or plus infinity")
    # Compared in log space, so a price far below float range still selects correctly.
    priced = p_eot_lp <= math.log(eps)
    p = np.exp(p_eot_lp[priced])
    observed = int((served_eot & priced).sum())
    return EndOfTurnCount(
        eps=eps, n_priced=int(priced.sum()), expected=float(p.sum()),
        observed=observed, p_value=poisson_binomial_upper_tail(p, observed))


def poisson_binomial_upper_tail(p: np.ndarray, k: int) -> float:
    """``P(X >= k)`` where ``X`` is a sum of independent Bernoulli variables with the
    probabilities ``p``, which need not be equal.

    Exact in the probabilities it is given, in float64 and with no distributional
    approximation: the law of ``X`` is the coefficient list of
    ``prod_i ((1 - p_i) + p_i z)``, and the upper tail is one minus the first ``k``
    coefficients. That those are the trials' own probabilities is the caller's premise
    rather than this function's (``end_of_turn_count``). The product is taken pairwise
    up a binary tree with every multiplication truncated at degree ``k``, which costs
    about ``1.5 * len(p) * k`` multiply-adds -- linear in the trace at the counts this
    is read at, since a faithful trace's ``k`` is a handful.

    Every coefficient is a sum of products of non-negative numbers, so nothing cancels
    while the product is built. The one subtraction is the last, and it fixes the
    resolution: the head sums to nearly 1 at any tail worth rejecting, so the result is
    accurate to about ``1e-15`` absolute and a returned value at that scale means "below
    what float64 can separate from zero" rather than a measured probability. Four
    orders of magnitude under any test size, so no verdict turns on it.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 1:
        raise ValueError(f"the trial probabilities must be a flat array, got shape "
                         f"{p.shape}")
    if p.size and (p.min() < 0.0 or p.max() > 1.0):
        raise ValueError(f"the trial probabilities must lie in [0, 1], got "
                         f"[{p.min()}, {p.max()}]")
    if k <= 0:
        return 1.0
    if k > p.size:
        return 0.0  # the sum of n trials cannot reach k > n
    return float(min(max(1.0 - _pgf_head(p, k).sum(), 0.0), 1.0))


def _pgf_head(p: np.ndarray, width: int) -> np.ndarray:
    """The first ``width`` coefficients of ``prod_i ((1 - p_i) + p_i z)``.

    One row per trial to start with, then repeated pairwise multiplication of the rows,
    each product truncated to ``width`` terms because nothing above them is read. The
    row count halves every pass, so the loop runs ``log2(len(p))`` times whatever the
    trial count; an odd count is padded with the polynomial 1, which multiplies to no
    effect."""
    polys = np.stack([1.0 - p, p], axis=1)
    while polys.shape[0] > 1:
        if polys.shape[0] % 2:
            unit = np.zeros((1, polys.shape[1]))
            unit[0, 0] = 1.0
            polys = np.concatenate([polys, unit])
        a, b = polys[0::2], polys[1::2]
        out = np.zeros((a.shape[0], min(a.shape[1] + b.shape[1] - 1, width)))
        for j in range(min(a.shape[1], out.shape[1])):
            m = min(b.shape[1], out.shape[1] - j)
            out[:, j:j + m] += a[:, j:j + 1] * b[:, :m]
        polys = out
    # A single trial never enters the loop, so its row can still be wider than asked
    # for; every other row was truncated on the way up.
    kept = min(polys.shape[1], width)
    head = np.zeros(width)
    head[:kept] = polys[0, :kept]
    return head


class PooledReport:
    """Sum of ``ReplyReport`` records over one measurement.

    Construct, :meth:`add` each reply's record, then read the totals, the three report
    statistics and the end-of-turn count. Records carrying a violation are counted in
    ``n_violations`` and contribute to nothing else.

    Attributes:
        n_replies:    records added, violations included.
        n_violations: records whose report violated the contract.
        n_tokens:     served tokens over the measured replies, which is also the number
                      of positions the divergence is summed over: every position of a
                      measured reply is measured.
        n_outside:    positions whose served token was not a reported atom; the
                      numerator of ``outside_rate``.
        n_moment:     positions that entered the two moment sums.
        kl_sum:       the divergence summed over ``n_tokens`` positions.
    """

    def __init__(self) -> None:
        self.n_replies = 0
        self.n_violations = 0
        self.n_tokens = 0
        self.n_outside = 0
        self.n_moment = 0
        self.kl_sum = 0.0
        self._s_sum = 0.0
        self._var_sum = 0.0
        # Per reply, its divergence total and its served count. The clustered standard
        # error needs each reply's residual sum, which is exactly this pair against the
        # pooled mean, so the individual values never have to be kept.
        self._by_reply: list[tuple[float, int]] = []
        # Per reply, its served-outside count and its served count, for the same
        # clustered error on the outside rate.
        self._outside_by_reply: list[tuple[int, int]] = []
        self._u: list[np.ndarray] = []
        # Per reply, its end-of-turn prices and where it ended its turn. Kept per
        # position because the count's threshold is applied here rather than at
        # scoring time: the calibration searches thresholds over records it has
        # already scored, and the verifier is handed one.
        self._p_eot_lp: list[np.ndarray] = []
        self._served_eot: list[np.ndarray] = []

    def add(self, record: ReplyReport) -> None:
        """Count one reply's record into the totals.

        A measured record carries one transform per served token (``score_reply``).
        One that does not was reduced by something else, and its divergence total and
        its position count would then be over different sets of positions, so it fails
        here rather than moving a mean that would mean nothing."""
        self.n_replies += 1
        if record.violation is not None:
            self.n_violations += 1
            return
        if len(record.u) != record.n_tokens:
            raise ValueError(
                f"event {record.event_key!r}: {len(record.u)} transforms for "
                f"{record.n_tokens} served tokens; a measured reply carries one "
                f"transform per served token")
        self.n_tokens += record.n_tokens
        self.n_outside += record.n_outside
        self.n_moment += record.n_moment
        self.kl_sum += record.kl_sum
        self._s_sum += record.s_sum
        self._var_sum += record.var_sum
        if record.n_tokens:
            # A reply that served nothing carries no evidence about either mean, so it
            # is not a cluster; counting it would only inflate the finite-cluster factor.
            self._by_reply.append((record.kl_sum, record.n_tokens))
            self._outside_by_reply.append((record.n_outside, record.n_tokens))
            self._u.append(np.asarray(record.u, dtype=np.float64))
        if record.p_eot_lp:
            served_eot = np.zeros(len(record.p_eot_lp), dtype=bool)
            served_eot[list(record.eot_positions)] = True
            self._p_eot_lp.append(np.asarray(record.p_eot_lp, dtype=np.float64))
            self._served_eot.append(served_eot)

    def extend(self, records: Iterable[ReplyReport]) -> None:
        """``add`` every record."""
        for r in records:
            self.add(r)

    @property
    def mean_kl(self) -> float:
        """The coarsened divergence per measured position; NaN on no measured
        position."""
        return self.kl_sum / self.n_tokens if self.n_tokens else float("nan")

    @property
    def se_kl(self) -> float:
        """The standard error of ``mean_kl``, clustered by reply.

        A reply served from a slightly wrong distribution is wrong all the way down, so
        its positions carry one piece of evidence rather than a thousand, and an error
        that assumes otherwise is too small by whatever the within-reply correlation is.
        The residuals are summed per reply and the usual finite-cluster correction
        applied; with fewer than two replies there is nothing to cluster and the error
        is NaN."""
        return _clustered_se(self._by_reply, self.mean_kl, self.n_tokens)

    @property
    def outside_rate(self) -> float:
        """The fraction of served positions whose token was not one of the reported
        atoms, over the replies that were measured; NaN with no served position.

        Under a faithful report this is the mass the report leaves outside its atoms,
        which is the model's own tail on the task's traffic. It is the one thing about
        the partition the reporter can choose, so calibration measures it on trusted
        stacks and the verifier holds a trace to it (module docstring, The partition).
        """
        return self.n_outside / self.n_tokens if self.n_tokens else float("nan")

    @property
    def se_outside(self) -> float:
        """The standard error of ``outside_rate``, clustered by reply as ``se_kl`` is
        and for the same reason; NaN with fewer than two measured replies."""
        return _clustered_se(self._outside_by_reply, self.outside_rate, self.n_tokens)

    @property
    def z(self) -> float:
        """The log-score moment as a standard normal deviate, or NaN with no positions
        behind it. Nothing is estimated from the data, so the null is exact up to the
        central limit theorem over positions."""
        if not self.n_moment or self._var_sum <= 0:
            return float("nan")
        return float(self._s_sum / math.sqrt(self._var_sum))

    @property
    def s_sum(self) -> float:
        """The log-score moment summed over ``n_moment`` positions, ``z``'s numerator."""
        return self._s_sum

    @property
    def var_sum(self) -> float:
        """That moment's summed variance, the square of ``z``'s denominator."""
        return self._var_sum

    def uniformity(self) -> tuple[float, float]:
        """The Cramer-von Mises statistic and p-value of every transform pooled.

        One sort over the pooled positions, which at a long trace is millions of them,
        so call it once and keep the result."""
        if not self.n_tokens:
            return float("nan"), float("nan")
        return cramer_von_mises_uniform(np.concatenate(self._u))

    def expected_end_of_turns(self, eps: float) -> float:
        """How many end-of-turn tokens the reference expects among the positions it
        prices at or below ``eps``.

        The threshold search reads this and nothing else, so it stops before the exact
        tail, which is the expensive half of the count (``end_of_turn_count``)."""
        if not self._p_eot_lp:
            return 0.0
        prices = np.concatenate(self._p_eot_lp)
        return float(np.exp(prices[prices <= math.log(eps)]).sum())

    def end_of_turn(self, eps: float) -> EndOfTurnCount:
        """The end-of-turn count at threshold ``eps`` over every reply pooled.

        A measurement with no priced position has an expected count of zero and no
        evidence against the null, so its p-value is 1."""
        if not self._p_eot_lp:
            return EndOfTurnCount(eps=eps, n_priced=0, expected=0.0, observed=0,
                                  p_value=1.0)
        return end_of_turn_count(np.concatenate(self._p_eot_lp),
                                 np.concatenate(self._served_eot), eps=eps)


# ---------------------------------------------------------------------------
# The Cramer-von Mises test for uniformity
# ---------------------------------------------------------------------------
#
# scipy.stats.cramervonmises is the same test, and this is a port of it rather than a
# variant: the harness's dependency set has no scipy (it would pull a compiled numeric
# stack into an orchestrator that otherwise needs none), and the statistic is read
# against a fixed significance level, so an approximation of the p-value with its own
# error budget is not good enough. The statistic is exact. The p-value is the asymptotic
# expansion of Csorgo and Faraway (1996), equations 1.2 and 1.8 -- the one place this
# check is not exact -- and it agrees with scipy's to within 1e-14 over the range where
# the expansion itself is accurate.


def _clustered_se(by_reply: Sequence[tuple[float, int]], mean: float,
                  n: int) -> float:
    """The standard error of a per-position mean whose positions come in replies.

    ``by_reply`` holds each reply's total and its position count, ``mean`` the pooled
    mean over ``n`` positions. Each reply's residual is its total against ``mean``
    times its count; the residuals are squared, summed, scaled by the finite-cluster
    factor and divided by ``n``. NaN with fewer than two replies or no position, since
    one cluster leaves nothing to cluster."""
    if len(by_reply) < 2 or not n:
        return float("nan")
    residual = np.array([total - count * mean for total, count in by_reply])
    clusters = len(by_reply)
    correction = clusters / (clusters - 1)
    return float(np.sqrt((residual ** 2).sum() * correction) / n)


def cramer_von_mises_uniform(u: np.ndarray) -> tuple[float, float]:
    """The Cramer-von Mises statistic and p-value for uniformity on [0, 1).

    ``omega^2 = 1/(12N) + sum_i ((2i-1)/(2N) - u_(i))^2`` on the sorted values. One sort
    and one vectorized pass, not the quadratic form, because a graded trace carries
    millions of positions.

    The p-value is clamped into [0, 1]. The expansion behind it overshoots by parts in
    1e12 where the true value is within rounding of an endpoint, and a verdict should
    not publish a probability outside its own range."""
    u = np.sort(np.asarray(u, dtype=np.float64))
    n = len(u)
    if n < 2:
        return float("nan"), float("nan")
    i = np.arange(1, n + 1)
    omega2 = float(1.0 / (12 * n) + np.sum(((2 * i - 1) / (2 * n) - u) ** 2))
    return omega2, min(max(1.0 - _cvm_cdf(omega2, n), 0.0), 1.0)


#: Trapezoid step for the scaled Bessel integrals below. The integrand is analytic in a
#: strip of half-width pi/2 around the real axis, where the trapezoid rule's error falls
#: off as exp(-pi^2 / step), so this is already far inside float64.
_BESSEL_STEP: float = 0.05


def _kve(nu: float, z: np.ndarray) -> np.ndarray:
    """``exp(z) * K_nu(z)``, the exponentially scaled modified Bessel function of the
    second kind, for strictly positive ``z``.

    Evaluated as ``integral over t of exp(-z (cosh t - 1)) cosh(nu t)``, which is the
    standard integral representation of ``K_nu`` rescaled so the value stays of order
    one at every ``z``: the unscaled function overflows at small ``z`` and underflows at
    large, and the series it appears in multiplies it by a matching exponential either
    way. The integrand decays double-exponentially in ``t``, so a trapezoid rule over a
    truncated half-line converges geometrically."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(z.shape, dtype=np.float64)
    flat, res = z.reshape(-1), out.reshape(-1)
    for i, zi in enumerate(flat):
        if not zi > 0.0:
            raise ValueError(f"the scaled Bessel integral needs z > 0, got {zi}")
        # Truncate where the integrand has fallen below the float64 floor, allowing for
        # the cosh(nu t) the exponential is fighting.
        t_max = math.acosh(1.0 + (830.0 + nu * 45.0) / zi) + 5.0
        t = np.arange(0, int(math.ceil(t_max / _BESSEL_STEP)) + 1) * _BESSEL_STEP
        f = np.exp(-zi * (np.cosh(t) - 1.0)) * np.cosh(nu * t)
        f[0] *= 0.5  # the trapezoid's endpoint, the integrand being even about 0
        res[i] = _BESSEL_STEP * f.sum()
    return out


def _cvm_cdf_asymptotic(x: np.ndarray) -> np.ndarray:
    """The limiting distribution function of the statistic (Csorgo and Faraway 1996,
    equation 1.2), summed until the terms fall below 1e-7.

    Not accurate for ``x`` much above 2, where the value is within rounding of one and
    the p-value it gives is within rounding of zero."""
    x = np.asarray(x, dtype=np.float64)
    total = np.zeros(x.shape, dtype=np.float64)
    for k in range(_CVM_MAX_TERMS):
        y = 4 * k + 1
        q = y ** 2 / (16 * x)
        scale = math.exp(math.lgamma(k + 0.5) - math.lgamma(k + 1.0))
        term = (scale / (math.pi ** 1.5 * np.sqrt(x)) * math.sqrt(y)
                * np.exp(-2.0 * q) * _kve(0.25, q))
        total = total + term
        if np.all(np.abs(term) < 1e-7):
            break
    return total


def _cvm_cdf(x: float, n: int) -> float:
    """The distribution function of the statistic at sample size ``n``.

    The limiting value with the first finite-sample correction of Csorgo and Faraway
    (1996), equation 1.8. The statistic's support is ``[1/(12n), n/3]``, and the
    expansion holds strictly inside it: at or below the lower end the value is zero and
    at or above the upper end it is one, which is where a p-value of exactly 1 or 0
    comes from. It is one too where the correction's series does not converge within the
    terms float64 can hold (``_cvm_correction``): there the statistic is in the
    thousands, far past the expansion's range and past any value a uniform sample reaches
    (the limiting 0.999 quantile is 1.17), so the p-value is zero."""
    if x <= 1.0 / (12 * n):
        return 0.0
    if x >= n / 3.0:
        return 1.0
    grid = np.array([x], dtype=np.float64)
    correction = _cvm_correction(grid)
    if correction is None:
        return 1.0
    value = (_cvm_cdf_asymptotic(grid) * (1 + 1.0 / (12 * n))
             + correction / n)
    return float(value[0])


#: Terms of the limiting Csorgo-Faraway series to sum before giving up. It converges in a
#: handful of terms over the range the expansion is accurate on; the bound only keeps a
#: pathological argument from looping.
_CVM_MAX_TERMS: int = 200
#: Terms of the correction series float64 can hold: its term ``k`` carries
#: ``Gamma(k + 3/2)``, which passes float64's largest value at ``k = 171``
#: (``Gamma(172.5)`` is about ``1.6e310``), so the terms ``k = 0 .. 170`` are all it can sum.
_CVM_CORRECTION_TERMS: int = 171


def _cvm_correction(x: np.ndarray) -> np.ndarray | None:
    """The order ``1/n`` correction to the limiting distribution function (Csorgo and
    Faraway 1996, equation 1.8), less its own limiting term, which
    ``_cvm_cdf`` folds into the leading factor instead so the series above is evaluated
    once. None when the series has not converged within ``_CVM_CORRECTION_TERMS`` terms,
    which happens for a statistic above about 4,177, far past the range the expansion is
    accurate on."""
    x = np.asarray(x, dtype=np.float64)
    sx = 2 * np.sqrt(x)
    y1, y2 = x ** 0.75, x ** 1.25
    total = np.zeros(x.shape, dtype=np.float64)
    for k in range(_CVM_CORRECTION_TERMS):
        m = 2 * k + 1
        g_half = math.exp(math.lgamma(k + 0.5))
        g_three_half = math.exp(math.lgamma(k + 1.5))
        a = (m * g_half * _cvm_e2((4 * k + 3) / sx) / (9 * y1)
             + g_half * _cvm_e3((4 * k + 1) / sx) / (72 * y2)
             + 2 * (m + 2) * g_three_half * _cvm_e3((4 * k + 5) / sx) / (12 * y2)
             + 7 * m * g_half * _cvm_e2((4 * k + 1) / sx) / (144 * y1)
             + 7 * m * g_half * _cvm_e2((4 * k + 5) / sx) / (144 * y1))
        term = -a / (math.pi * math.exp(math.lgamma(k + 1.0)))
        total = total + term
        if np.all(np.abs(term) < 1e-7):
            return total
    return None


def _cvm_e2(y: np.ndarray) -> np.ndarray:
    """The first Bessel combination of the correction series' terms."""
    z = y ** 2 / 4
    return (np.exp(-2.0 * z) * (y / 2) ** 1.5
            * (_kve(0.25, z) + _kve(0.75, z)) / math.sqrt(math.pi))


def _cvm_e3(y: np.ndarray) -> np.ndarray:
    """The second Bessel combination of the correction series' terms."""
    z = y ** 2 / 4
    return (np.exp(-2.0 * z) / math.sqrt(math.pi) * (y / 2) ** 2.5
            * (2 * _kve(0.25, z) + 3 * _kve(0.75, z) - _kve(1.25, z)))
