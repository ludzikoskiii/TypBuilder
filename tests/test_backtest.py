"""Testy backtestu - przede wszystkim brak wycieku danych z przyszłości."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest
from backtest import (block_cutoffs, bootstrap_ci, brier, log_loss, max_drawdown, run_backtest,
                      simulate_bets, walk_forward_predict)
from config import load_config
from model import DixonColes, score_matrix, simulate_matches

N_TEAMS = 12


@pytest.fixture(scope="module")
def league() -> pd.DataFrame:
    """Symulowana liga: 4 sezony, kursy z marżą 5% wokół prawdziwych prawdopodobieństw."""
    rng = np.random.default_rng(7)
    att = rng.normal(0, 0.3, N_TEAMS)
    dfn = rng.normal(0, 0.3, N_TEAMS)
    att -= att.mean()
    dfn -= dfn.mean()
    frames = []
    for k, year in enumerate(range(2020, 2024)):
        s = simulate_matches(att, dfn, 0.15, 0.25, -0.1, n_rounds=2, seed=100 + k, start=f"{year}-08-01")
        s["season"] = f"{year}/{year + 1}"
        frames.append(s)
    df = pd.concat(frames, ignore_index=True)
    idx = {f"T{k:02d}": k for k in range(N_TEAMS)}
    probs = []
    for h, a in zip(df["home"], df["away"]):
        i, j = idx[h], idx[a]
        m = score_matrix(np.exp(0.15 + 0.25 + att[i] - dfn[j]), np.exp(0.15 + att[j] - dfn[i]), -0.1)
        probs.append([np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum(), m[np.add.outer(range(11), range(11)) > 2.5].sum()])
    p = np.array(probs)
    noise = rng.normal(0, 0.03, (len(df), 3))
    pn = np.clip(p[:, :3] * np.exp(noise), 0.02, None)
    pn /= pn.sum(axis=1, keepdims=True)
    df[["odds_h", "odds_d", "odds_a"]] = 1 / (pn * 1.05)
    df[["closing_odds_h", "closing_odds_d", "closing_odds_a"]] = 1 / (p[:, :3] * 1.04)
    df["odds_over25"], df["odds_under25"] = 1 / (p[:, 3] * 1.05), 1 / ((1 - p[:, 3]) * 1.05)
    df["closing_odds_over25"], df["closing_odds_under25"] = np.nan, np.nan
    df["odds_are_closing"] = False
    df["league"] = "SIM"
    return df


@pytest.fixture
def cfg() -> dict:
    c = load_config()
    c["model"]["xi_grid"] = [0.0, 0.003]
    c["backtest"]["bootstrap_samples"] = 300
    return c


class SpyDC(DixonColes):
    """DixonColes, który zapisuje, jakie dane widział przy treningu i co przewidywał."""
    log: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    def fit(self, matches, ref_date=None, warm_start=None):
        self._seen_max = pd.Timestamp(matches["date"].max())
        return super().fit(matches, ref_date, warm_start)

    def predict_many(self, fixtures):
        SpyDC.log.append((self._seen_max, pd.Timestamp(fixtures["date"].min())))
        return super().predict_many(fixtures)


def test_no_future_data_in_training(league, monkeypatch):
    """Każda predykcja korzysta z modelu trenowanego tylko na meczach sprzed przewidywanych."""
    SpyDC.log = []
    monkeypatch.setattr(backtest, "DixonColes", SpyDC)
    pred = walk_forward_predict(league, pd.Timestamp("2022-08-01"), pd.Timestamp("2024-08-01"),
                                {"xi": 0.002}, refit_days=3, min_train=50)
    assert len(SpyDC.log) > 20
    for seen_max, first_pred in SpyDC.log:
        assert seen_max < first_pred
    assert (pred["train_max_date"] < pred["date"]).all()
    assert (pred["cutoff"] <= pred["date"]).all()


def test_predictions_do_not_change_when_future_is_altered(league):
    """Zmiana wyników z przyszłości nie może wpłynąć na wcześniejsze predykcje."""
    start, end = pd.Timestamp("2022-08-01"), pd.Timestamp("2024-08-01")
    split = pd.Timestamp("2023-03-01")
    base = walk_forward_predict(league, start, end, {"xi": 0.002}, min_train=50)
    poisoned = league.copy()
    future = poisoned["date"] >= split
    rng = np.random.default_rng(0)
    poisoned.loc[future, "hg"] = rng.integers(0, 8, future.sum())
    poisoned.loc[future, "ag"] = rng.integers(0, 8, future.sum())
    alt = walk_forward_predict(poisoned, start, end, {"xi": 0.002}, min_train=50)
    before = base["date"] < split
    cols = ["p_home", "p_draw", "p_away", "p_over25"]
    assert np.allclose(base.loc[before, cols], alt.loc[before, cols], atol=1e-12)
    # kontrola czułości testu: predykcje po zmianie danych faktycznie się różnią
    after = base["date"] > split + pd.Timedelta(days=7)
    assert not np.allclose(base.loc[after, cols], alt.loc[after, cols])


def test_xi_tuning_uses_only_pre_test_data(league, cfg, monkeypatch):
    SpyDC.log = []
    monkeypatch.setattr(backtest, "DixonColes", SpyDC)
    cfg["backtest"]["validation_seasons"] = 1
    res = run_backtest(league, ["2023/2024"], cfg, league="SIM", tune=True)
    test_start = league.loc[league["season"] == "2023/2024", "date"].min()
    assert res.validation_seasons == ["2022/2023"]
    assert res.xi_table is not None and len(res.xi_table) == 2
    for seen_max, first_pred in SpyDC.log:
        assert seen_max < first_pred
    # pierwsze wywołania to strojenie xi - wszystkie przed okresem testowym
    tuning_calls = SpyDC.log[: len(SpyDC.log) - res.predictions["cutoff"].nunique()]
    assert tuning_calls and all(fp < test_start for _, fp in tuning_calls)


def test_block_cutoffs():
    dates = pd.Series(pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-09",
                                      "2024-01-10", "2024-01-12"]))
    assert block_cutoffs(dates, 3) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-01-09"),
                                       pd.Timestamp("2024-01-12")]


def test_full_backtest_runs_and_gives_verdict(league, cfg):
    res = run_backtest(league, ["2022/2023", "2023/2024"], cfg, league="SIM", league_name="Symulacja",
                       tune=False)
    rep = res.reports["1X2"]
    assert {"Model DC"} <= set(rep.metrics["zrodlo"])
    assert rep.betting["n_bets"] > 0
    lo, hi = rep.betting["roi_ci"]
    assert lo <= rep.betting["roi_flat"] <= hi
    assert "Symulacja" in res.verdict
    assert any(w in res.verdict for w in ("NIE pokonuje", "NIEJEDNOZNACZNY", "POKONAŁ"))
    assert "OU25" in res.reports
    assert rep.clv is not None


def test_metrics():
    p = np.array([[0.5, 0.3, 0.2], [0.1, 0.2, 0.7]])
    y = np.array([0, 2])
    assert np.allclose(log_loss(p, y), [-np.log(0.5), -np.log(0.7)])
    assert np.allclose(brier(p, y), [0.25 + 0.09 + 0.04, 0.01 + 0.04 + 0.09])


def test_bootstrap_ci_and_drawdown():
    rng = np.random.default_rng(0)
    lo, hi = bootstrap_ci(rng.normal(0.1, 1, 2000), rng, 1000)
    assert lo < 0.1 < hi
    assert max_drawdown(np.array([1, 3, 0, 2, -1])) == pytest.approx(4.0)


def test_simulate_bets_kelly_and_flat(cfg):
    bets = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                         "home": ["A", "B", "C"], "odds": [2.0, 3.0, 2.0],
                         "p_model": [0.6, 0.4, 0.6], "won": [True, False, False], "value": [0.2, 0.2, 0.2]})
    cfg["staking"].update({"flat_stake": 10, "bankroll": 1000, "kelly_fraction": 0.25, "max_stake_pct": 0.02})
    out, s = simulate_bets(bets, cfg)
    assert s["profit_flat"] == pytest.approx(10 - 10 - 10)
    # dzień 1: obie stawki od bankrolla 1000, 1/4 Kelly obcięte do 2% -> 20 i 20
    assert out["stake_kelly"].iloc[0] == pytest.approx(20) and out["stake_kelly"].iloc[1] == pytest.approx(20)
    # dzień 2: bankroll 1000 + 20 - 20 = 1000 -> 20
    assert out["stake_kelly"].iloc[2] == pytest.approx(20)
    assert s["end_bankroll_kelly"] == pytest.approx(980)
