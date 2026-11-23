"""Tests for hls.verify.poisson -- the curve rule's Poisson arithmetic.

Three functions, all pure and all scipy-free: the Poisson survival function in log
space, the rejection count it implies at a given trace length, and the exact Garwood
upper confidence limit on a rate. Each is checked against a closed form where one
exists, so the tests do not merely re-run the implementation.
"""

from __future__ import annotations

import math

import pytest

from hls.verify.poisson import (
    chi2_quantile,
    flip_count_threshold,
    garwood_upper,
    poisson_log_sf,
    rate_rule_min_tokens,
)

# The multitenant_slo artifact's deployed parameters, used to check the trace floor
# against the configuration that actually ships.
_SHIPPED_Q_MAX = 0.0021852664967043248
_SHIPPED_DELTA = 1.25e-4  # delta_family 1e-3 split across 8 rungs


# ---- poisson_log_sf -----------------------------------------------------------


def test_poisson_sf_whole_mass_below_zero():
    # P(X > n) = 1 for any n < 0 (the whole distribution sits above).
    assert poisson_log_sf(-1, 3.0) == 0.0


def test_poisson_sf_empty_tail_at_mu_zero():
    assert poisson_log_sf(0, 0.0) == float("-inf")


def test_poisson_sf_matches_closed_forms():
    # P(X > 0) = 1 - e^{-mu}, exactly.
    for mu in (0.01, 1.0, 3.6889, 20.0):
        assert poisson_log_sf(0, mu) == pytest.approx(math.log(-math.expm1(-mu)), abs=1e-13)
    # P(X > 2) for mu=1.5, against the finite complement sum.
    direct = 1.0 - math.exp(-1.5) * (1.0 + 1.5 + 1.5 ** 2 / 2.0)
    assert math.exp(poisson_log_sf(2, 1.5)) == pytest.approx(direct, rel=1e-12)


def test_poisson_sf_deep_tail_keeps_relative_precision():
    # The reason for summing the tail directly instead of 1-CDF: a ~1e-50 tail
    # would be annihilated by cancellation, but the log-space sum keeps it. The
    # exact tail at mu=1 is sum_{j>40} e^{-1}/j! -- small enough to compare
    # against a directly-summed reference (each term is ~1e-50, well inside
    # double range even though 1-CDF would return exactly 0).
    got = poisson_log_sf(40, 1.0)
    want = math.log(sum(math.exp(-1.0 - math.lgamma(j + 1)) for j in range(41, 100)))
    assert got == pytest.approx(want, rel=1e-12)
    assert math.isfinite(got) and got < -100


def test_poisson_sf_rejects_negative_mu():
    with pytest.raises(ValueError):
        poisson_log_sf(1, -0.5)


# ---- flip_count_threshold ------------------------------------------------------


def test_threshold_is_the_poisson_tail_quantile():
    # k is the smallest count whose exceedance probability is within budget, so the
    # tail at k must clear delta and the one at k-1 must not.
    lam, n, delta = 2e-5, 100_000, 1.25e-4
    k = flip_count_threshold(lam, n, delta)
    assert poisson_log_sf(k, lam * n) <= math.log(delta)
    assert poisson_log_sf(k - 1, lam * n) > math.log(delta)


def test_threshold_grows_with_trace_length():
    lam = 2e-5
    ks = [flip_count_threshold(lam, n, 1.25e-4) for n in (10_000, 100_000, 1_000_000)]
    assert ks == sorted(ks) and ks[0] < ks[-1]


def test_threshold_at_zero_trace_admits_nothing():
    # mu = 0: a clean trace of no tokens carries no flips, so observing ANY (strict >)
    # must reject.
    assert flip_count_threshold(1e-6, 0, 1e-4) == 0


def test_threshold_rejects_bad_delta():
    with pytest.raises(ValueError):
        flip_count_threshold(1e-6, 100, 0.0)
    with pytest.raises(ValueError):
        flip_count_threshold(1e-6, 100, 1.0)


def _threshold_by_direct_search(mu: float, delta: float) -> int:
    """The definition, evaluated the slow way: walk k up, re-summing the whole tail at
    every candidate. Quadratic in mu, so only usable at moderate mu -- which is exactly
    what makes it an independent check on the fast descending walk."""
    log_delta = math.log(delta)
    k = 0
    while poisson_log_sf(k, mu) > log_delta:
        k += 1
    return k


@pytest.mark.parametrize("mu", [0.0, 0.01, 1.0, 17.0, 60.0, 250.0, 1000.0])
@pytest.mark.parametrize("delta", [1e-3, 1.25e-4, 1e-6, 0.5])
def test_threshold_matches_a_direct_tail_search(mu, delta):
    """The descending walk answers exactly what the definition does, across means
    spanning four orders of magnitude and budgets spanning three."""
    assert flip_count_threshold(mu, 1, delta) == _threshold_by_direct_search(mu, delta)


@pytest.mark.parametrize("mu", [2e5, 2e6])
def test_threshold_stays_minimal_at_large_mu(mu):
    """Beyond the direct search's reach (it is quadratic), check the defining property
    instead: k clears the budget and k-1 does not. Runs in milliseconds, which is
    itself the point -- the same query walked from scratch per candidate does not."""
    delta = 1.25e-4
    k = flip_count_threshold(mu, 1, delta)
    assert poisson_log_sf(k, mu) <= math.log(delta)
    assert poisson_log_sf(k - 1, mu) > math.log(delta)


def test_poisson_sf_converges_at_large_mu():
    """The tail walk's term budget is sized from mu, not fixed, so a large mean still
    converges rather than tripping the non-convergence guard."""
    got = poisson_log_sf(2_005_181, 2e6)
    assert math.isfinite(got) and got < math.log(1.25e-4)


# ---- rate_rule_min_tokens ------------------------------------------------------


def _rate_rule_false_alarm(n_tokens: int, q_max: float, headroom: float) -> float:
    """log P(a clean trace of ``n_tokens`` tokens trips the rate rule)."""
    lam = q_max * n_tokens
    return poisson_log_sf(math.floor(headroom * lam), lam)


@pytest.mark.parametrize("headroom", [1.5, 2.0, 3.0])
def test_floor_holds_at_every_length_above_it(headroom):
    """The property the floor claims: not merely that the guarantee holds AT the floor,
    but that it holds at every longer trace. P(Poisson(lam) > floor(h*lam)) rises with
    lam wherever floor(h*lam) holds still, so a floor taken at the first crossing would
    admit longer traces that violate again."""
    n = rate_rule_min_tokens(_SHIPPED_Q_MAX, _SHIPPED_DELTA, headroom)
    log_delta = math.log(_SHIPPED_DELTA)
    # Sampled with a stride coprime to nothing in particular, over a span wide enough to
    # cross many floor(h*lam) steps.
    for length in range(n, n + 400_000, 89):
        assert _rate_rule_false_alarm(length, _SHIPPED_Q_MAX, headroom) <= log_delta


@pytest.mark.parametrize("headroom", [1.5, 2.0, 3.0])
def test_floor_is_minimal(headroom):
    """One token shorter and the guarantee genuinely fails, so the floor is not merely
    safe but the shortest safe length."""
    n = rate_rule_min_tokens(_SHIPPED_Q_MAX, _SHIPPED_DELTA, headroom)
    assert (_rate_rule_false_alarm(n - 1, _SHIPPED_Q_MAX, headroom)
            > math.log(_SHIPPED_DELTA))


def test_floor_at_the_shipped_config():
    """The deployed multitenant_slo gate's floor, pinned: 1.5x headroom over the
    artifact's q_max needs lam = 94/1.5 = 62.67 expected clean flips."""
    assert rate_rule_min_tokens(_SHIPPED_Q_MAX, _SHIPPED_DELTA, 1.5) == 28_677
    assert rate_rule_min_tokens(_SHIPPED_Q_MAX, _SHIPPED_DELTA, 2.0) == 8_009


def test_floor_tightens_with_headroom():
    """A tighter headroom leaves less room for Poisson noise, so it needs a longer
    trace -- which is why the floor lives with the headroom on the verifier."""
    floors = [rate_rule_min_tokens(_SHIPPED_Q_MAX, _SHIPPED_DELTA, h)
              for h in (3.0, 2.0, 1.5, 1.2)]
    assert floors == sorted(floors)


def test_floor_rejects_a_headroom_that_cannot_be_certified():
    with pytest.raises(ValueError):
        rate_rule_min_tokens(_SHIPPED_Q_MAX, _SHIPPED_DELTA, 1.0)


def test_garwood_fixture_matches_the_e0_closed_form():
    # E=0: chi2_{2, 0.975}/2 = -ln(0.025), so lambda_up = -ln(0.025)/T exactly.
    t = 2_930_232.0
    assert garwood_upper(0, t) == pytest.approx(-math.log(0.025) / t, rel=1e-12)

# ---- garwood_upper / chi2_quantile --------------------------------------------


def test_chi2_quantile_matches_the_df2_closed_form():
    # chi-square with 2 df is Exp(2), so ppf(p, 2) = -2 ln(1-p) exactly.
    for p in (0.025, 0.5, 0.975):
        assert chi2_quantile(p, 2) == pytest.approx(-2.0 * math.log(1.0 - p), rel=1e-12)


def test_garwood_is_the_rate_whose_tail_puts_conf_above_the_count():
    """The defining property, checked at every even df the curve rule uses: the limit
    is exactly the rate with P(Poisson(lambda*T) > e) = 0.975."""
    for e in (0, 1, 2, 5, 11, 47, 176, 1024):
        lam = garwood_upper(e, 1)
        assert poisson_log_sf(e, lam) == pytest.approx(math.log(0.975), abs=1e-9)


def test_garwood_at_zero_events_is_finite_and_scales_with_exposure():
    """A level that saw nothing still gets a usable ceiling -- which is the whole
    reason the curve uses an upper limit rather than the point rate."""
    t = 2_930_232.0
    assert garwood_upper(0, t) == pytest.approx(-math.log(0.025) / t, rel=1e-12)
    assert garwood_upper(0, 2 * t) == pytest.approx(garwood_upper(0, t) / 2, rel=1e-12)


def test_garwood_rises_with_the_observed_count():
    t = 1_000_000
    limits = [garwood_upper(e, t) for e in (0, 1, 5, 40)]
    assert limits == sorted(limits)


def test_garwood_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        garwood_upper(-1, 1000)
    with pytest.raises(ValueError):
        garwood_upper(0, 0)
