"""Testy usuwania marży, value i stawek."""
from __future__ import annotations

import numpy as np
import pytest

from market import (fair_probs, fair_proportional, fair_shin, flat_stake, is_value,
                    kelly_fraction, kelly_stake, overround, value)

ODDS = np.array([
    [1.19, 6.77, 14.19],
    [2.10, 3.40, 3.60],
    [8.37, 4.93, 1.36],
    [1.80, 2.05, 9.0],
])


@pytest.mark.parametrize("method", ["shin", "proportional"])
def test_margin_removal_sums_to_one(method):
    p = fair_probs(ODDS, method)
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)
    assert (p > 0).all() and (p < 1).all()


def test_two_way_market_sums_to_one():
    p = fair_shin(np.array([1.90, 1.95]))
    assert p.sum() == pytest.approx(1.0)


def test_proportional_formula():
    o = np.array([2.0, 3.5, 4.0])
    pi = 1 / o
    assert np.allclose(fair_proportional(o), pi / pi.sum())


def test_shin_corrects_favourite_longshot_bias():
    """Shin przypisuje faworytowi więcej, a outsiderowi mniej niż metoda proporcjonalna."""
    o = np.array([1.25, 6.0, 12.0])
    prop, shin = fair_proportional(o), fair_shin(o)
    assert shin[0] > prop[0]
    assert shin[2] < prop[2]


def test_shin_without_margin_returns_implied():
    fair = np.array([0.5, 0.3, 0.2])
    p, z = fair_shin(1 / fair, return_z=True)
    assert np.allclose(p, fair) and z == pytest.approx(0.0, abs=1e-9)


def test_shin_z_positive_with_margin():
    _, z = fair_shin(ODDS, return_z=True)
    assert (z > 0).all() and (z < 0.2).all()


def test_overround():
    assert overround(np.array([1.9, 1.9])) == pytest.approx(2 / 1.9)


def test_value_and_threshold():
    assert value(0.55, 2.0) == pytest.approx(0.10)
    assert is_value(0.55, 2.0, 0.05)
    assert not is_value(0.52, 2.0, 0.05)  # value 4% < próg 5%


def test_kelly():
    # f* = (p*o - 1)/(o - 1) = (0.55*2 - 1)/1 = 0.10
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.40, 2.0) == 0.0
    # 1/4 Kelly = 2.5% -> obcięte do 2% bankrolla
    assert kelly_stake(0.55, 2.0, 1000, fraction=0.25, max_pct=0.02) == pytest.approx(20.0)
    # 1/4 Kelly z f* = 4% -> 1%
    assert kelly_stake(0.52, 2.0, 1000, fraction=0.25, max_pct=0.02) == pytest.approx(10.0)


def test_flat_stake():
    s = flat_stake(np.array([0.55, 0.50]), np.array([2.0, 2.0]), stake=10, threshold=0.05)
    assert list(s) == [10.0, 0.0]
