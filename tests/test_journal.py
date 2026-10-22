"""Testy dziennika typów."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

import journal as j


@pytest.fixture
def conn(tmp_path):
    return j.connect(tmp_path / "j.sqlite")


def _bet(**kw):
    base = {"match_date": "2026-10-25", "league": "E0", "home": "Arsenal", "away": "Chelsea",
            "market": "1X2", "selection": "1", "odds": 2.0, "stake": 20.0, "p_model": 0.55}
    return {**base, **kw}


@pytest.mark.parametrize("market,sel,hg,ag,expected", [
    ("1X2", "1", 2, 1, "won"), ("1X2", "X", 2, 1, "lost"), ("1X2", "X", 1, 1, "won"),
    ("1X2", "2", 0, 3, "won"), ("O/U 2.5", "Over", 2, 1, "won"), ("O/U 2.5", "Under", 2, 1, "lost"),
    ("O/U 1.5", "Under", 1, 0, "won"), ("O/U 3.5", "Over", 2, 2, "won"),
    ("BTTS", "Tak", 1, 1, "won"), ("BTTS", "Nie", 1, 1, "lost"), ("BTTS", "Nie", 3, 0, "won"),
])
def test_settle_outcome(market, sel, hg, ag, expected):
    assert j.settle_outcome(market, sel, hg, ag) == expected


def test_budget_blocks_bets_over_limit(conn):
    now = datetime(2026, 10, 20, 12)
    j.add_bet(conn, _bet(stake=150), monthly_limit=200, now=now)
    with pytest.raises(j.BudgetExceeded):
        j.add_bet(conn, _bet(stake=60), monthly_limit=200, now=now)
    j.add_bet(conn, _bet(stake=50), monthly_limit=200, now=now)  # dokładnie do limitu
    st = j.budget_status(conn, 200, day=now.date())
    assert st["blocked"] and st["remaining"] == 0
    # nowy miesiąc = nowy budżet
    assert not j.budget_status(conn, 200, day=datetime(2026, 11, 1).date())["blocked"]


def test_invalid_bets_rejected(conn):
    with pytest.raises(ValueError):
        j.add_bet(conn, _bet(odds=1.0), 1000)
    with pytest.raises(ValueError):
        j.add_bet(conn, _bet(market="1X2", selection="Over"), 1000)


def test_auto_settlement_and_clv(conn):
    j.add_bet(conn, _bet(selection="1", odds=2.2, stake=10), 1000)
    j.add_bet(conn, _bet(home="Leeds", away="Wolves", market="O/U 2.5", selection="Over", odds=1.9, stake=10), 1000)
    j.add_bet(conn, _bet(home="Hull", away="Everton", stake=10), 1000)  # brak wyniku -> zostaje pending
    results = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-26", "2026-10-25"]),
        "home": ["Arsenal", "Leeds"], "away": ["Chelsea", "Wolves"], "hg": [2, 1], "ag": [0, 1],
        "closing_odds_h": [2.0, 2.5], "closing_odds_d": [3.5, 3.3], "closing_odds_a": [4.0, 2.9],
        "closing_odds_over25": [1.8, 2.0], "closing_odds_under25": [2.0, 1.8],
    })
    assert j.settle_pending(conn, {"E0": results}) == 2
    df = j.list_bets(conn).set_index("home")
    assert df.loc["Arsenal", "status"] == "won" and df.loc["Arsenal", "profit"] == pytest.approx(12.0)
    assert df.loc["Arsenal", "clv"] == pytest.approx(2.2 / 2.0 - 1)
    assert df.loc["Leeds", "status"] == "lost" and df.loc["Leeds", "profit"] == pytest.approx(-10.0)
    assert df.loc["Hull", "status"] == "pending"
    s = j.stats(j.list_bets(conn))
    assert s["n_settled"] == 2 and s["profit"] == pytest.approx(2.0)
    assert s["roi"] == pytest.approx(2.0 / 20.0)


def test_void_bet_frees_budget(conn):
    now = datetime(2026, 10, 20)
    bid = j.add_bet(conn, _bet(stake=200), monthly_limit=200, now=now)
    j.void_bet(conn, bid)
    assert j.month_staked(conn, now.date()) == 0
