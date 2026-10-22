"""Testy modelu Dixona-Colesa."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

from model import (DixonColes, markets_from_matrix, score_matrices, score_matrix,
                   simulate_matches, tau, time_weights)

N_TEAMS = 20


@pytest.fixture(scope="module")
def truth() -> dict:
    rng = np.random.default_rng(1)
    att = rng.normal(0, 0.3, N_TEAMS)
    dfn = rng.normal(0, 0.3, N_TEAMS)
    return {"att": att - att.mean(), "def": dfn - dfn.mean(), "mu": 0.15, "home": 0.27, "rho": -0.12}


@pytest.fixture(scope="module")
def simulated(truth) -> pd.DataFrame:
    return simulate_matches(truth["att"], truth["def"], truth["mu"], truth["home"], truth["rho"],
                            n_rounds=6, seed=3)


@pytest.fixture(scope="module")
def fitted(simulated) -> DixonColes:
    return DixonColes(xi=0.0, ridge=0.0, min_matches=0).fit(simulated)


# ------------------------------------------------------------------ prawdopodobieństwa


@pytest.mark.parametrize("lam,mu,rho", [(1.5, 1.1, -0.1), (0.4, 3.2, 0.05), (2.8, 0.3, -0.2), (1.0, 1.0, 0.0)])
def test_probabilities_sum_to_one(lam, mu, rho):
    m = score_matrix(lam, mu, rho)
    assert m.sum() == pytest.approx(1.0, abs=1e-12)
    assert (m >= 0).all()
    mk = markets_from_matrix(m)
    assert mk["p_home"] + mk["p_draw"] + mk["p_away"] == pytest.approx(1.0, abs=1e-12)
    for line in ("15", "25", "35"):
        assert mk[f"over{line}"] + mk[f"under{line}"] == pytest.approx(1.0)
    assert mk["btts_yes"] + mk["btts_no"] == pytest.approx(1.0)


def test_predict_many_matches_single_and_sums_to_one(fitted):
    fx = pd.DataFrame({"home": ["T00", "T05", "T10"], "away": ["T01", "T19", "T03"]})
    pm = fitted.predict_many(fx)
    assert np.allclose(pm[["p_home", "p_draw", "p_away"]].sum(axis=1), 1.0)
    single = fitted.predict("T05", "T19").markets
    assert pm.loc[1, "p_home"] == pytest.approx(single["p_home"])
    assert pm.loc[1, "p_over25"] == pytest.approx(single["over25"])


# ------------------------------------------------------------------ korekta rho


def test_rho_correction_matches_paper_formula():
    lam, mu, rho = 1.4, 1.1, -0.13
    m = score_matrix(lam, mu, rho, max_goals=15)
    indep = np.outer(poisson.pmf(np.arange(16), lam), poisson.pmf(np.arange(16), mu))
    expected = {(0, 0): 1 - lam * mu * rho, (0, 1): 1 + lam * rho, (1, 0): 1 + mu * rho, (1, 1): 1 - rho}
    for (x, y), factor in expected.items():
        assert m[x, y] == pytest.approx(factor * indep[x, y], rel=1e-9)
        assert tau(np.array([x]), np.array([y]), lam, mu, rho)[0] == pytest.approx(factor)
    # pozostałe wyniki bez zmian
    mask = np.ones_like(m, dtype=bool)
    mask[:2, :2] = False
    assert np.allclose(m[mask], indep[mask] / indep.sum(), rtol=1e-9)
    # korekta nie zmienia sumy prawdopodobieństw (przed obcięciem macierzy)
    adj = sum((f - 1) * indep[c] for c, f in expected.items())
    assert adj == pytest.approx(0.0, abs=1e-15)


def test_rho_zero_is_independent_poisson():
    m = score_matrix(1.3, 0.9, 0.0, max_goals=15)
    indep = np.outer(poisson.pmf(np.arange(16), 1.3), poisson.pmf(np.arange(16), 0.9))
    assert np.allclose(m, indep / indep.sum())


def test_negative_rho_increases_draws_00_11():
    base, neg = score_matrix(1.3, 1.1, 0.0), score_matrix(1.3, 1.1, -0.15)
    assert neg[0, 0] > base[0, 0] and neg[1, 1] > base[1, 1]
    assert neg[1, 0] < base[1, 0] and neg[0, 1] < base[0, 1]


def test_vectorized_matrices_equal_single():
    lam, mu = np.array([1.2, 2.5]), np.array([0.8, 1.7])
    ms = score_matrices(lam, mu, -0.1)
    for k in range(2):
        assert np.allclose(ms[k], score_matrix(lam[k], mu[k], -0.1))


# ------------------------------------------------------------------ estymacja


def test_recovers_known_parameters(truth, fitted):
    est_att = np.array([fitted.team_params(f"T{k:02d}")[0] for k in range(N_TEAMS)])
    est_def = np.array([fitted.team_params(f"T{k:02d}")[1] for k in range(N_TEAMS)])
    assert fitted.converged
    assert np.corrcoef(est_att, truth["att"])[0, 1] > 0.93
    assert np.corrcoef(est_def, truth["def"])[0, 1] > 0.93
    assert np.abs(est_att - truth["att"]).max() < 0.25
    assert np.abs(est_def - truth["def"]).max() < 0.25
    assert fitted.home_adv == pytest.approx(truth["home"], abs=0.06)
    assert fitted.mu == pytest.approx(truth["mu"], abs=0.06)
    assert fitted.rho == pytest.approx(truth["rho"], abs=0.07)


def test_identifiability_constraint(fitted):
    assert fitted.attack.sum() == pytest.approx(0.0, abs=1e-9)
    assert fitted.defence.sum() == pytest.approx(0.0, abs=1e-9)


def test_analytic_gradient_matches_numeric(simulated):
    m = DixonColes(ridge=1.5)
    teams = sorted(set(simulated["home"]) | set(simulated["away"]))
    idx = {t: k for k, t in enumerate(teams)}
    hi, ai = simulated["home"].map(idx).to_numpy(), simulated["away"].map(idx).to_numpy()
    x, y = simulated["hg"].to_numpy(float), simulated["ag"].to_numpy(float)
    w = np.random.default_rng(0).uniform(0.2, 1.0, len(simulated))
    n = len(teams)
    rng = np.random.default_rng(5)
    th = rng.normal(0, 0.15, 2 * n + 1)
    th[-1] = -0.08
    prior = (rng.uniform(0, 5, n), rng.normal(0, 0.2, n), rng.normal(0, 0.2, n))
    f = lambda t: m._nll(t, hi, ai, x, y, w, n, 0.0, prior)[0]  # noqa: E731
    grad = m._nll(th, hi, ai, x, y, w, n, 0.0, prior)[1]
    h = 1e-5
    num = np.array([(f(th + e) - f(th - e)) / (2 * h) for e in np.eye(len(th)) * h])
    assert np.allclose(grad, num, atol=1e-5)


def test_time_weights():
    dates = pd.Series(pd.to_datetime(["2026-01-01", "2025-01-01"]))
    w = time_weights(dates, pd.Timestamp("2026-01-01"), 0.0019)
    assert w[0] == pytest.approx(1.0)
    assert w[1] == pytest.approx(np.exp(-0.0019 * 365))


def test_time_weighting_tracks_recent_form(truth):
    """Drużyna, która nagle się wzmocniła, przy xi > 0 jest oceniana wyżej niż przy xi = 0."""
    early = simulate_matches(truth["att"], truth["def"], truth["mu"], truth["home"], truth["rho"],
                             n_rounds=2, seed=11, start="2020-08-01")
    att2 = truth["att"].copy()
    att2[0] += 0.8
    late = simulate_matches(att2, truth["def"], truth["mu"], truth["home"], truth["rho"],
                            n_rounds=1, seed=12, start="2022-08-01")
    both = pd.concat([early, late], ignore_index=True)
    flat = DixonColes(xi=0.0, min_matches=0).fit(both).team_params("T00")[0]
    decayed = DixonColes(xi=0.005, min_matches=0).fit(both).team_params("T00")[0]
    assert decayed > flat


# ------------------------------------------------------------------ fallback


def test_unknown_team_gets_weak_team_fallback(fitted, simulated):
    model = DixonColes(xi=0.0, min_matches=5, fallback_bottom_k=3).fit(simulated)
    att, dfn, is_fb = model.team_params("Beniaminek")
    assert is_fb
    strength = model.attack + model.defence
    weakest = np.argsort(strength)[:3]
    assert att == pytest.approx(model.attack[weakest].mean())
    assert dfn == pytest.approx(model.defence[weakest].mean())
    p = model.predict("Beniaminek", "T00")
    assert p.home_fallback and not p.away_fallback


def test_team_with_few_matches_is_shrunk_to_fallback_and_flagged(simulated):
    extra = pd.DataFrame({"date": [pd.Timestamp("2030-01-01"), pd.Timestamp("2030-01-08")],
                          "home": ["Nowy", "T01"], "away": ["T00", "Nowy"], "hg": [5, 0], "ag": [0, 4]})
    both = pd.concat([simulated, extra], ignore_index=True)
    model = DixonColes(xi=0.0, min_matches=6).fit(both)
    no_prior = DixonColes(xi=0.0, min_matches=0).fit(both)
    att, _, is_fb = model.team_params("Nowy")
    assert is_fb and model.low_data["Nowy"] == pytest.approx(1 - 2 / 6)
    assert not no_prior.team_params("Nowy")[2]
    # prior ściąga ocenę beniaminka w stronę fallbacku, ale dane nadal ją podnoszą
    assert model.fallback_att < att < no_prior.team_params("Nowy")[0] - 0.2
    assert "Nowy" in set(model.ratings().query("fallback > 0")["druzyna"])
    # drużyny z pełną historią zmieniają się tylko o przesunięcie z ograniczenia sum(att)=0
    assert abs(model.team_params("T05")[0] - no_prior.team_params("T05")[0]) < 0.05
