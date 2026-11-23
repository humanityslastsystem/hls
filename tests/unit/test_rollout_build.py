"""A rollout build record's cost: the one rule finalize's index row and the viewer's list row both
read a build's cost by."""

from __future__ import annotations

import pytest

from hls.obs.record import RolloutBuild, finite_number

#: A build record with every field but its usage.
BUILD = {"build": {}, "outputs": {}, "package": None, "started_at": 300.0, "ended_at": 400.0,
         "upload_sha256": None}


@pytest.mark.parametrize(("usage", "cost"), [
    ({"cost_usd": 12.5}, 12.5),
    ({"cost_usd": 3}, 3.0),
    ({"cost_usd": 0}, 0.0),
    ({"cost_usd": None}, None),
    ({}, None),
    (None, None),
    ({"cost_usd": float("nan")}, None),
    ({"cost_usd": float("inf")}, None),
    ({"cost_usd": True}, None),
    ({"cost_usd": "12.5"}, None),
])
def test_a_builds_cost_is_its_usages_cost_when_that_is_a_finite_number(usage, cost):
    assert RolloutBuild.model_validate({**BUILD, "usage": usage}).cost_usd == cost


@pytest.mark.parametrize(("value", "number"), [
    (1.5, 1.5), (2, 2.0), (False, None), (float("-inf"), None), (None, None), ([1], None),
])
def test_a_finite_number_is_a_finite_int_or_float_but_no_bool(value, number):
    assert finite_number(value) == number
