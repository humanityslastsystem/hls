"""Unit tests for the Poisson arrival process."""

from __future__ import annotations

import pytest

from hls.spec.workload import AllAtOnce, Poisson


def test_rejects_non_positive_rate():
    # Poisson is a pydantic Poly, so a bad rate raises a ValidationError (a
    # ValueError subclass) carrying the validator's message.
    with pytest.raises(ValueError, match="rate_per_sec must be > 0"):
        Poisson(rate_per_sec=0.0)
    with pytest.raises(ValueError, match="rate_per_sec must be > 0"):
        Poisson(rate_per_sec=-1.0)


def test_schedule_is_strictly_increasing():
    """Cumulative arrivals: each offset > the previous."""
    offsets = Poisson(rate_per_sec=10.0).schedule(20, seed=42)
    assert all(offsets[i + 1] > offsets[i] for i in range(len(offsets) - 1))


def test_schedule_length_matches_n_items():
    assert len(Poisson(rate_per_sec=1.0).schedule(0, seed=0)) == 0
    assert len(Poisson(rate_per_sec=1.0).schedule(7, seed=0)) == 7
    assert len(Poisson(rate_per_sec=1.0).schedule(1000, seed=0)) == 1000


def test_schedule_is_seeded_reproducibly():
    a = Poisson(rate_per_sec=5.0).schedule(50, seed=123)
    b = Poisson(rate_per_sec=5.0).schedule(50, seed=123)
    c = Poisson(rate_per_sec=5.0).schedule(50, seed=456)
    assert a == b
    assert a != c


def test_mean_inter_arrival_matches_one_over_rate():
    """For a large enough N, the mean inter-arrival time converges to 1/rate."""
    rate = 4.0
    n = 5000
    offsets = Poisson(rate_per_sec=rate).schedule(n, seed=7)
    inter = [offsets[0]] + [offsets[i + 1] - offsets[i] for i in range(n - 1)]
    mean = sum(inter) / n
    # 1/rate = 0.25; tolerance loose enough for n=5000 sample noise.
    assert abs(mean - 1.0 / rate) < 0.02


def test_all_at_once_remains_zero():
    """AllAtOnce is unchanged -- sanity check the sibling arrival shape."""
    assert AllAtOnce().schedule(5, seed=0) == [0.0, 0.0, 0.0, 0.0, 0.0]
