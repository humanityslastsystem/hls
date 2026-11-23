"""Poisson arithmetic for the flip-count curve rule, in log space and without scipy.

The curve rule models flips at a gap level as one Poisson arrival process per token
(see ``hls.verify.calibration.CurveSide`` for the model and how it is measured).
Calibration needs an upper confidence limit on that rate from a count and an exposure
(``garwood_upper``); grading needs the count at which a trace leaves the calibrated
envelope (``flip_count_threshold``) and the shortest trace on which the companion rate
rule can carry its false-alarm guarantee (``rate_rule_min_tokens``). All of them reduce
to the Poisson tail, so they live together here.
"""

from __future__ import annotations

import math

# Once the running log-sum exceeds a new term by this margin, every remaining term
# (they decay at least geometrically past the Poisson mode) is below double
# precision's resolution of the sum: exp(-41.5) ~ 1e-18.
_LOG_NEGLIGIBLE = math.log(1e-18)

# Both series in _reg_gamma_p converge in tens of iterations over the range used here;
# the cap only bounds a hang if they are ever called far outside it.
_MAX_ITERS = 10_000

# Doublings allowed when a search brackets its answer from below. Each doubling is a
# factor of 2 on a float, so a few dozen cover every input these functions see and 200
# stops short of overflowing to infinity (which would poison the bracket rather than
# raise).
_MAX_DOUBLINGS = 200

# Relative width at which a converging series or bisection stops.
_TOL = 1e-16


def _logaddexp(a: float, b: float) -> float:
    """``log(exp(a) + exp(b))`` without leaving log space, larger operand first so the
    exponential that is actually evaluated can only underflow, never overflow."""
    hi, lo = (a, b) if a >= b else (b, a)
    if hi == float("-inf"):
        return hi
    return hi + math.log1p(math.exp(lo - hi))


def _psi(u: float) -> float:
    """The Poisson large-deviation rate function ``u*ln(u) - u + 1``, for ``u >= 1``.

    Zero at ``u = 1`` and strictly increasing above it (its derivative is ``ln u``),
    which is what makes the Chernoff bound ``P(Poisson(m) >= u*m) <= exp(-m*psi(u))``
    tighten monotonically as the excess ratio ``u`` grows. Two callers here rely on
    that monotonicity to turn "the tail is small enough at this point" into "and at
    every point past it", which is what bounds their searches.
    """
    return u * math.log(u) - u + 1.0


def _certain_tail_count(mu: float, log_p: float) -> int:
    """A count ``k`` for which ``P(Poisson(mu) > k) <= exp(log_p)`` is certain, obtained
    without summing anything.

    By the Chernoff bound above, ``P(X >= x) <= exp(-mu * _psi(x/mu))`` for ``x > mu``.
    Bisecting ``_psi`` for the excess ratio ``u`` at which ``mu * _psi(u) >= -log_p``
    and returning ``ceil(mu*u)`` therefore names a count whose tail is under budget.
    It is an upper bound on the smallest such count, not that count itself -- its use
    is to give a walk somewhere safe to start from.

    Requires ``mu > 0`` and ``log_p < 0``.
    """
    target = -log_p / mu
    lo, hi = 1.0, 2.0
    for _ in range(_MAX_DOUBLINGS):
        if _psi(hi) >= target:
            break
        lo, hi = hi, hi * 2.0
    else:
        raise RuntimeError(
            f"could not bracket a Chernoff excess ratio for mu={mu}, log_p={log_p}")
    # Invariant: _psi(hi) >= target, so ceil(mu*hi) is safe at every step; bisection
    # only tightens it. 200 halvings take the bracket far below one count.
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _psi(mid) >= target:
            hi = mid
        else:
            lo = mid
    return math.ceil(mu * hi)


def poisson_log_sf(n: int, mu: float) -> float:
    """log P(X > n) for X ~ Poisson(mu), summed in log space.

    Each pmf term is formed as ``-mu + j*log(mu) - lgamma(j+1)`` and terms are
    accumulated with logaddexp, so neither factorials nor tiny tail masses ever leave
    log space. Summation starts at ``j = n+1`` (the survival tail itself, never 1-CDF,
    so a sf of 1e-300 keeps full relative precision) and stops once past the mode a
    term can no longer move the accumulated sum at double precision.

    Edge cases: ``n < 0`` is the whole mass (log 1 = 0.0); ``mu == 0`` has an empty
    tail (-inf).
    """
    if n < 0:
        return 0.0
    if mu < 0 or math.isnan(mu):
        raise ValueError(f"Poisson mean must be >= 0, got {mu}")
    if mu == 0.0:
        return float("-inf")
    log_mu = math.log(mu)
    term = -mu + (n + 1) * log_mu - math.lgamma(n + 2)
    total = term
    # Bounded by construction rather than by a hang guard: from j = 2*mu onward every
    # term is at most half its predecessor, and no term ever exceeds the running sum it
    # is added to, so at most 64 further terms are needed to fall the 1e-18 below that
    # sum which ends the walk (2^-64 < 1e-18).
    j_stop = max(n + 1, math.ceil(2.0 * mu)) + 64
    for j in range(n + 2, j_stop + 1):
        term += log_mu - math.log(j)  # log pmf(j) = log pmf(j-1) + log(mu/j)
        total = _logaddexp(total, term)
        # Past the mode (j > mu) terms decrease strictly, so once one is
        # negligible against the sum the remaining geometric tail is too.
        if j > mu and term < total + _LOG_NEGLIGIBLE:
            return total
    raise RuntimeError(f"Poisson tail sum did not converge (n={n}, mu={mu})")


def flip_count_threshold(lambda_up: float, n_tokens: int, delta: float) -> int:
    """k(c, N): the smallest count the clean envelope rejects beyond.

    ``k = min{ k : P(Poisson(lambda_up * n_tokens) > k) <= delta }``. The verifier
    rejects a trace whose observed count at this level exceeds k (strict >): observing
    exactly k is the last count still inside the envelope.

    Found by walking down from a count whose tail is certainly under budget
    (``_certain_tail_count``), accumulating one pmf term per step. Descending is what
    makes this both cheap and stable: every step adds a term to a growing tail (never
    subtracts near-equal tails, and never forms 1-CDF), and the walk touches only the
    counts between the answer and its starting bound -- ``O(sqrt(mu))`` of them --
    rather than re-summing the whole tail once per candidate count.
    """
    if n_tokens < 0:
        raise ValueError(f"n_tokens must be >= 0 (got {n_tokens})")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    mu = lambda_up * n_tokens
    log_delta = math.log(delta)
    if mu <= 0.0:
        # No expected flips at all: the tail above every count is empty, so the
        # envelope admits nothing and any observation (strict >) rejects.
        return 0
    log_mu = math.log(mu)
    # Start where the tail is negligible even relative to delta, so treating
    # P(X > k_top) as 0 below perturbs no comparison against delta.
    k_top = _certain_tail_count(mu, log_delta + _LOG_NEGLIGIBLE)
    log_pmf = -mu + k_top * log_mu - math.lgamma(k_top + 1.0)  # log P(X = k_top)
    log_tail = float("-inf")  # log P(X > k_top)
    for k in range(k_top - 1, -1, -1):
        log_tail = _logaddexp(log_tail, log_pmf)  # P(X > k) = P(X > k+1) + P(X = k+1)
        if log_tail > log_delta:
            # k is the largest count whose tail is still over budget (the tail only
            # grows as k falls), so k+1 is the smallest count that is not.
            return k + 1
        log_pmf += math.log(k + 1) - log_mu  # log P(X = k) from log P(X = k+1)
    # Every count down to 0 stayed inside the budget.
    return 0


def rate_rule_min_tokens(q_max: float, delta: float, headroom: float) -> int:
    """The shortest trace at and above which the rate rule holds its nominal
    false-alarm rate, in teacher-forced tokens.

    A clean trace of ``N`` tokens carries about ``Poisson(q_max * N)`` flips, and the
    rule rejects above ``headroom * q_max``. Its false-alarm probability is therefore
    ``g(lam) = P(Poisson(lam) > floor(headroom * lam))`` with ``lam = q_max * N``,
    which for small ``lam`` sits far above the budget the rule was sized for.

    ``g`` is a sawtooth, not a decreasing function, which is why this is not
    "the first ``lam`` that clears ``delta``". Write ``m = floor(headroom * lam)``.
    Across an interval where ``m`` holds still, ``g`` rises with ``lam`` (a fixed count
    threshold against a growing mean); it drops only where ``m`` steps. So a ``lam``
    that clears ``delta`` can be followed by longer traces that do not, and a floor
    taken at the first crossing would license traces the rule cannot certify.

    Indexing by ``m`` instead of by ``lam`` makes the answer exact. On ``m``'s interval
    ``[m/headroom, (m+1)/headroom)`` the supremum of ``g`` is its right-hand limit
    ``P(Poisson((m+1)/headroom) > m)``, approached but not attained, so the whole
    interval is inside budget exactly when that limit is. The floor is then the right
    end of the last violating interval -- minimal, because any ``lam`` below it lies
    inside a violating interval and so has violating lengths above it.

    The scan is finite by Chernoff rather than by a streak heuristic: at that
    right-hand limit the excess ratio is exactly ``headroom``, so the limit is at most
    ``exp(-(m+1)/headroom * _psi(headroom))``, which is under ``delta`` for every ``m``
    with ``m + 1 >= headroom * ln(1/delta) / _psi(headroom)``. No interval past that
    horizon can violate, so scanning up to it is exhaustive.

    The result depends on ``headroom``: a tighter one leaves less room for Poisson
    noise, so it needs more expected flips before the guarantee holds (at
    delta=1.25e-4, a 2x headroom needs lam = 17.5 expected flips and a 1.5x headroom
    needs lam = 62.67). That is why the floor is computed where the headroom is known
    -- on the verifier -- rather than baked into a calibration, which several verifiers
    may grade against at different headrooms.
    """
    if not q_max > 0:
        raise ValueError(f"q_max must be > 0 (got {q_max})")
    if not 0.0 < delta < 1.0:
        raise ValueError(f"delta must be in (0, 1), got {delta}")
    if not headroom > 1.0:
        raise ValueError(
            f"headroom must be > 1 (got {headroom}); at or below 1 the rule rejects the "
            f"clean rate itself and no trace length gives it a false-alarm guarantee")
    log_delta = math.log(delta)
    horizon = math.ceil(headroom * -log_delta / _psi(headroom))
    last_violating = -1
    for m in range(horizon + 1):
        if poisson_log_sf(m, (m + 1) / headroom) > log_delta:
            last_violating = m
    # -1 (nothing violated) gives lam_min = 0: the rule carries its guarantee at every
    # length, so there is no floor.
    lam_min = (last_violating + 1) / headroom
    return math.ceil(lam_min / q_max)


#: One-sided false-coverage probability of every Garwood limit in the design: the
#: complement of the 97.5% upper endpoint of a conventional two-sided 95% interval.
#: The single name for that choice -- ``garwood_upper`` bounds each level's event rate
#: at it, and ``hls.verify.calibration.exposure_target`` reads the same number as ``ln(1/alpha)``
#: when it sizes a campaign against a level that observed nothing. Dimensionless.
GARWOOD_ALPHA: float = 0.025


def garwood_upper(events: int, exposure: int, alpha: float = GARWOOD_ALPHA) -> float:
    """The exact (Garwood) upper confidence limit on a Poisson rate per unit exposure.

    ``lambda_up = chi2_quantile(1 - alpha, 2*events + 2) / (2 * exposure)``, the rate
    whose Poisson tail puts exactly ``alpha`` mass at or below ``events``. Exact rather
    than asymptotic, so a level that observed zero events still gets a finite ceiling
    (``ln(1/alpha)/exposure``) instead of the useless 0 a point estimate would give.
    """
    if events < 0:
        raise ValueError(f"events must be >= 0 (got {events})")
    if exposure <= 0:
        raise ValueError(f"exposure must be > 0 (got {exposure})")
    return chi2_quantile(1.0 - alpha, 2 * events + 2) / (2.0 * exposure)


def chi2_quantile(p: float, df: float) -> float:
    """The chi-square quantile: the x with CDF(x; df) = P(df/2, x/2) = p.

    Bisection over a doubling bracket, 200 halvings, giving ~1e-14 relative -- far
    below every tolerance that consumes it.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = df / 2.0
    lo, hi = 0.0, max(df, 1.0)
    for _ in range(_MAX_DOUBLINGS):
        if _reg_gamma_p(a, hi / 2.0) >= p:
            break
        hi *= 2.0
    else:
        raise RuntimeError(
            f"chi2_quantile could not bracket the quantile (p={p}, df={df})")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _reg_gamma_p(a, mid / 2.0) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _reg_gamma_p(a: float, x: float) -> float:
    """The regularized lower incomplete gamma P(a, x) = gamma(a, x) / Gamma(a).

    Series expansion for x < a+1, continued fraction for the complement otherwise --
    the classic numerically stable split, each expansion used only where it converges
    fast and monotonically.
    """
    if a <= 0 or x < 0:
        raise ValueError(f"P(a, x) needs a > 0, x >= 0 (got a={a}, x={x})")
    if x == 0.0:
        return 0.0
    log_prefactor = -x + a * math.log(x) - math.lgamma(a)
    if x < a + 1.0:
        # P(a,x) = e^-x x^a / Gamma(a) * sum_{n>=0} x^n / (a(a+1)...(a+n)).
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(_MAX_ITERS):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * _TOL:
                return total * math.exp(log_prefactor)
        raise RuntimeError(f"gamma series did not converge (a={a}, x={x})")
    # Modified Lentz continued fraction for Q(a,x); P = 1 - Q.
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITERS):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        h *= step
        if abs(step - 1.0) < _TOL:
            return 1.0 - h * math.exp(log_prefactor)
    raise RuntimeError(f"gamma continued fraction did not converge (a={a}, x={x})")
