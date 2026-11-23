import math

import pytest

from hls.spec.workload import (
    DeterministicArrival, Poisson, Gamma, Uniform, Normal, ConstantInterTurn,
    GammaInterTurn,
)


def test_deterministic_arrival_is_cumulative():
    sched = DeterministicArrival(gap=2.0).schedule(4, seed=0)
    assert sched == [0.0, 2.0, 4.0, 6.0]


def test_schedule_is_nondecreasing_and_starts_at_zero():
    for arr in (Poisson(rate_per_sec=2.0), Gamma(shape=0.5, scale=2.0),
                Uniform(low=0.0, high=1.0), Normal(mean=0.5, std=0.2)):
        sched = arr.schedule(20, seed=7)
        assert sched[0] == 0.0
        assert all(b >= a for a, b in zip(sched, sched[1:])), arr
        assert len(sched) == 20


def test_schedule_is_deterministic_under_seed():
    a = Poisson(rate_per_sec=3.0).schedule(10, seed=123)
    b = Poisson(rate_per_sec=3.0).schedule(10, seed=123)
    c = Poisson(rate_per_sec=3.0).schedule(10, seed=124)
    assert a == b
    assert a != c


@pytest.mark.parametrize("arr", [
    DeterministicArrival(gap=1.5),
    Poisson(rate_per_sec=3.0),
    Gamma(shape=2.0, scale=0.5),
    Uniform(low=0.1, high=2.0),
    Normal(mean=1.0, std=0.3),
], ids=["deterministic", "poisson", "gamma", "uniform", "normal"])
def test_every_arrival_family_is_reproducible_under_seed(arr):
    """Same (n, seed) -> byte-identical schedule for EVERY family. They all draw from
    random.Random(seed), so this also holds across processes (no hash() / PYTHONHASHSEED)."""
    assert arr.schedule(15, seed=4242) == arr.schedule(15, seed=4242)


@pytest.mark.parametrize("arr", [
    Poisson(rate_per_sec=3.0),
    Gamma(shape=2.0, scale=0.5),
    Uniform(low=0.1, high=2.0),
    Normal(mean=1.0, std=0.3),
], ids=["poisson", "gamma", "uniform", "normal"])
def test_stochastic_arrival_family_is_seed_sensitive(arr):
    """A different seed yields a different schedule for the stochastic families.
    DeterministicArrival is excluded -- it ignores the seed by construction."""
    assert arr.schedule(15, seed=4242) != arr.schedule(15, seed=4243)


def test_poisson_mean_gap_is_about_one_over_rate():
    sched = Poisson(rate_per_sec=5.0).schedule(2001, seed=1)
    gaps = [b - a for a, b in zip(sched, sched[1:])]
    assert math.isclose(sum(gaps) / len(gaps), 1 / 5.0, rel_tol=0.15)


def test_arrival_validation():
    with pytest.raises(ValueError):
        Poisson(rate_per_sec=0.0)
    with pytest.raises(ValueError):
        Gamma(shape=0.0, scale=1.0)
    with pytest.raises(ValueError):
        Uniform(low=1.0, high=1.0)
    with pytest.raises(ValueError):
        Normal(mean=-1.0, std=1.0)


def test_constant_inter_turn_delay():
    d = ConstantInterTurn(gap=1.5)
    assert d.delay(1, seed=(0, 0)) == 1.5
    assert d.delay(5, seed=(9, 9)) == 1.5


def test_gamma_inter_turn_is_deterministic_under_seed():
    d = GammaInterTurn(shape=0.4, scale=280.0, max_gap=1800.0)
    assert d.delay(1, seed=(0, 0)) == d.delay(1, seed=(0, 0))     # same (turn, seed) -> same draw


def test_gamma_inter_turn_varies_by_turn_and_session():
    d = GammaInterTurn(shape=0.4, scale=280.0, max_gap=1800.0)
    assert d.delay(1, seed=(0, 0)) != d.delay(2, seed=(0, 0))     # turn folded into the stream
    assert d.delay(1, seed=(0, 0)) != d.delay(1, seed=(0, 1))     # session folded in


def test_gamma_inter_turn_clamps_at_max_gap():
    d = GammaInterTurn(shape=0.4, scale=280.0, max_gap=10.0)      # tiny cap
    assert all(d.delay(k, seed=(7, k)) <= 10.0 for k in range(1, 200))
    assert any(d.delay(k, seed=(7, k)) == 10.0 for k in range(1, 200))   # cap actually bites


def test_gamma_inter_turn_cv_matches_one_over_sqrt_shape():
    import statistics
    shape = 0.4
    d = GammaInterTurn(shape=shape, scale=280.0, max_gap=1e12)    # effectively untruncated
    xs = [d.delay(1, seed=(s, 0)) for s in range(20000)]
    cv = statistics.pstdev(xs) / statistics.mean(xs)
    assert cv == pytest.approx(1.0 / shape ** 0.5, rel=0.05)      # CV = 1/sqrt(shape)


def test_gamma_inter_turn_validation():
    with pytest.raises(ValueError):
        GammaInterTurn(shape=0.0, scale=1.0, max_gap=1.0)
    with pytest.raises(ValueError):
        GammaInterTurn(shape=1.0, scale=0.0, max_gap=1.0)
    with pytest.raises(ValueError):
        GammaInterTurn(shape=1.0, scale=1.0, max_gap=0.0)


def test_gamma_inter_turn_describe():
    assert "gamma" in GammaInterTurn(shape=0.4, scale=280.0, max_gap=1800.0).describe()


def test_mean_rate_is_sessions_per_second_for_every_family():
    """``mean_rate`` is the reciprocal of each family's mean gap, in sessions per
    second; the identity arrival has none."""
    from hls.spec.workload import (AllAtOnce, DeterministicArrival, Gamma, Normal,
                                   Poisson, Uniform)
    assert AllAtOnce().mean_rate() is None
    assert DeterministicArrival(gap=4.0).mean_rate() == 0.25
    assert Poisson(rate_per_sec=2.0).mean_rate() == 2.0
    assert abs(Gamma(shape=0.444444, scale=3.37838).mean_rate() - 0.666) < 1e-3
    assert Uniform(low=1.0, high=3.0).mean_rate() == 0.5
    assert Normal(mean=5.0, std=1.0).mean_rate() == 0.2
    assert Normal(mean=0.0, std=1.0).mean_rate() is None


# --- how long a schedule takes to place every session ----------------------
#
# A trajectory drive stops issuing at a declared instant, and a session scheduled
# past it is never opened. So the arrival process, the session count and the issuing
# window are one relationship, and nothing related them until these.

def test_a_schedules_span_is_the_wait_for_its_last_session():
    """The quantity the issuing window has to cover. Session 0 leaves at t=0, so ``n``
    sessions are placed by ``n - 1`` gaps."""
    from hls.spec.workload import AllAtOnce, DeterministicArrival

    assert DeterministicArrival(gap=2.0).schedule_span_sec(4) == 6.0
    assert AllAtOnce().schedule_span_sec(100) == 0.0
    assert DeterministicArrival(gap=2.0).schedule_span_sec(1) == 0.0


def test_the_span_matches_the_draw_it_predicts():
    """Against the schedule itself, which is what the driver actually sleeps on."""
    from hls.spec.workload import Poisson

    arr = Poisson(rate_per_sec=0.5)
    drawn = [max(arr.schedule(200, seed=s)) for s in range(40)]
    mean_span = arr.schedule_span_sec(200)
    assert abs(sum(drawn) / len(drawn) - mean_span) < 0.1 * mean_span


def test_headroom_widens_the_span_a_stochastic_family_needs():
    """A mean span leaves half the draws over the line. ``sigmas`` is the margin that
    buys a window the draw actually fits inside, and a deterministic family needs
    none because its draw has no spread."""
    from hls.spec.workload import DeterministicArrival, Poisson

    arr = Poisson(rate_per_sec=0.5)
    assert arr.schedule_span_sec(200, sigmas=3.0) > arr.schedule_span_sec(200)
    assert all(max(arr.schedule(200, seed=s)) <= arr.schedule_span_sec(200, sigmas=3.0)
               for s in range(40))
    fixed = DeterministicArrival(gap=2.0)
    assert fixed.schedule_span_sec(4, sigmas=3.0) == fixed.schedule_span_sec(4)


def test_a_poisson_process_can_be_asked_to_fit_a_window():
    """The inverse, which is how a task declares a rate instead of guessing one: give
    it the sessions and the window it has to place them in."""
    from hls.spec.workload import Poisson

    arr = Poisson.covering(44, window_sec=2700.0)
    assert arr.schedule_span_sec(44, sigmas=3.0) <= 2700.0
    assert all(max(arr.schedule(44, seed=s)) <= 2700.0 for s in range(40))


def test_fitting_a_window_refuses_what_cannot_be_placed():
    from hls.spec.workload import Poisson

    with pytest.raises(ValueError, match="window_sec"):
        Poisson.covering(44, window_sec=0.0)
    with pytest.raises(ValueError, match="n_sessions"):
        Poisson.covering(0, window_sec=2700.0)
