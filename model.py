"""Model Dixona-Colesa (1997) z wagą czasową.

Parametryzacja (log-skala):
    log lambda_home = mu + gamma + att[home] - def[away]
    log lambda_away = mu + att[away] - def[home]
gdzie att = siła ataku, def = siła obrony (więcej = lepsza obrona), gamma = przewaga
gospodarzy, mu = poziom bramek ligi. Identyfikowalność: sum(att) = 0, sum(def) = 0.

Rozkład wyniku: P(x, y) = tau(x, y) * Pois(x; lambda) * Pois(y; lambda_away), gdzie
tau koryguje wyniki 0:0, 1:0, 0:1, 1:1 parametrem rho (wzór (4.2) z pracy):
    tau(0,0) = 1 - lambda*mu*rho,  tau(0,1) = 1 + lambda*rho,
    tau(1,0) = 1 + mu*rho,         tau(1,1) = 1 - rho,   w pozostałych przypadkach 1.

Estymacja: maksymalizacja ważonej log-wiarygodności, wagi exp(-xi * dni_od_meczu).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import poisson


# --------------------------------------------------------------------------- wzory


def tau(x: np.ndarray, y: np.ndarray, lam: np.ndarray, mu: np.ndarray, rho: float) -> np.ndarray:
    """Czynnik korekcyjny Dixona-Colesa dla wyników (x, y) (wektorowo)."""
    x, y = np.asarray(x), np.asarray(y)
    lam, mu = np.broadcast_to(lam, x.shape), np.broadcast_to(mu, x.shape)
    out = np.ones(np.broadcast(x, y).shape, dtype=float)
    out = np.where((x == 0) & (y == 0), 1.0 - lam * mu * rho, out)
    out = np.where((x == 0) & (y == 1), 1.0 + lam * rho, out)
    out = np.where((x == 1) & (y == 0), 1.0 + mu * rho, out)
    out = np.where((x == 1) & (y == 1), 1.0 - rho, out)
    return out


def score_matrix(lam: float, mu: float, rho: float, max_goals: int = 10) -> np.ndarray:
    """Macierz P(gospodarze = i, goście = j), i, j = 0..max_goals, znormalizowana do 1."""
    goals = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(goals, lam), poisson.pmf(goals, mu))
    m[0, 0] *= 1.0 - lam * mu * rho
    m[0, 1] *= 1.0 + lam * rho
    m[1, 0] *= 1.0 + mu * rho
    m[1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum()


def score_matrices(lam: np.ndarray, mu: np.ndarray, rho: float, max_goals: int = 10) -> np.ndarray:
    """Wektorowa wersja score_matrix: zwraca tablicę (N, G, G)."""
    goals = np.arange(max_goals + 1)
    ph = poisson.pmf(goals[None, :], np.asarray(lam)[:, None])
    pa = poisson.pmf(goals[None, :], np.asarray(mu)[:, None])
    m = ph[:, :, None] * pa[:, None, :]
    m[:, 0, 0] *= 1.0 - lam * mu * rho
    m[:, 0, 1] *= 1.0 + lam * rho
    m[:, 1, 0] *= 1.0 + mu * rho
    m[:, 1, 1] *= 1.0 - rho
    m = np.clip(m, 0.0, None)
    return m / m.sum(axis=(1, 2), keepdims=True)


def markets_from_matrix(m: np.ndarray, top_n: int = 5) -> dict[str, Any]:
    """Prawdopodobieństwa rynków z macierzy wyników: 1X2, O/U 1.5/2.5/3.5, BTTS, top wyniki."""
    g = m.shape[0]
    i, j = np.indices(m.shape)
    total = i + j
    out: dict[str, Any] = {
        "p_home": float(np.tril(m, -1).sum()),
        "p_draw": float(np.trace(m)),
        "p_away": float(np.triu(m, 1).sum()),
        "btts_yes": float(m[1:, 1:].sum()),
        "exp_home_goals": float((i * m).sum()),
        "exp_away_goals": float((j * m).sum()),
    }
    out["btts_no"] = 1.0 - out["btts_yes"]
    for line in (1.5, 2.5, 3.5):
        over = float(m[total > line].sum())
        key = str(line).replace(".", "")
        out[f"over{key}"] = over
        out[f"under{key}"] = 1.0 - over
    flat = np.argsort(m, axis=None)[::-1][:top_n]
    out["top_scores"] = [(int(k // g), int(k % g), float(m.flat[k])) for k in flat]
    return out


def matrices_to_markets(ms: np.ndarray) -> pd.DataFrame:
    """Wektorowo: 1X2, O/U 2.5 i BTTS dla tablicy macierzy (N, G, G)."""
    g = ms.shape[1]
    i, j = np.indices((g, g))
    lower, upper = i > j, i < j
    return pd.DataFrame({
        "p_home": (ms * lower).sum(axis=(1, 2)),
        "p_draw": np.trace(ms, axis1=1, axis2=2),
        "p_away": (ms * upper).sum(axis=(1, 2)),
        "p_over25": (ms * (i + j > 2.5)).sum(axis=(1, 2)),
        "p_btts": ms[:, 1:, 1:].sum(axis=(1, 2)),
    })


def time_weights(dates: pd.Series, ref_date: pd.Timestamp, xi: float) -> np.ndarray:
    """Wagi exp(-xi * dni od meczu do ref_date)."""
    days = (pd.Timestamp(ref_date) - pd.to_datetime(dates)).dt.days.to_numpy(dtype=float)
    return np.exp(-xi * np.clip(days, 0, None))


# --------------------------------------------------------------------------- model


@dataclass
class MatchPrediction:
    """Wynik modelu dla jednego meczu."""
    home: str
    away: str
    lam_home: float
    lam_away: float
    matrix: np.ndarray
    markets: dict[str, Any]
    home_fallback: bool = False
    away_fallback: bool = False


@dataclass
class DixonColes:
    """Model Dixona-Colesa. Użycie: DixonColes(xi=...).fit(mecze).predict('A', 'B')."""
    xi: float = 0.0019
    max_goals: int = 10
    ridge: float = 1.0
    min_matches: int = 6
    fallback_bottom_k: int = 3
    fallback_strength: float = 10.0
    rho_bounds: tuple[float, float] = (-0.25, 0.25)

    # parametry po dopasowaniu
    teams: list[str] = field(default_factory=list)
    attack: np.ndarray = field(default_factory=lambda: np.zeros(0))
    defence: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mu: float = 0.0
    home_adv: float = 0.0
    rho: float = 0.0
    n_matches: dict[str, int] = field(default_factory=dict)
    low_data: dict[str, float] = field(default_factory=dict)  # drużyna -> siła priora fallbacku (0..1]
    fallback_att: float = 0.0
    fallback_def: float = 0.0
    ref_date: pd.Timestamp | None = None
    train_max_date: pd.Timestamp | None = None
    converged: bool = False
    loglik: float = float("nan")

    @classmethod
    def from_config(cls, cfg: dict[str, Any], **overrides: Any) -> "DixonColes":
        m = cfg.get("model", {})
        keys = ("xi", "max_goals", "ridge", "min_matches", "fallback_bottom_k", "fallback_strength")
        kwargs = {k: m[k] for k in keys if k in m}
        kwargs.update(overrides)
        return cls(**kwargs)

    # ---------------------------------------------------------------- estymacja

    @staticmethod
    def _unpack(theta: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        a_free, d_free = theta[: n - 1], theta[n - 1: 2 * n - 2]
        att = np.append(a_free, -a_free.sum())
        dfn = np.append(d_free, -d_free.sum())
        mu, home, rho = theta[2 * n - 2], theta[2 * n - 1], theta[2 * n]
        return att, dfn, mu, home, rho

    def _nll(self, theta: np.ndarray, hi: np.ndarray, ai: np.ndarray, x: np.ndarray,
             y: np.ndarray, w: np.ndarray, n: int, const: float,
             prior: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None) -> tuple[float, np.ndarray]:
        """Ujemna ważona log-wiarygodność + kary L2 oraz jej gradient analityczny.

        prior = (kappa, att0, def0): dodatkowa kara 0.5*kappa*((att-att0)^2 + (def-def0)^2)
        per drużyna (prior fallbacku dla drużyn z małą liczbą meczów).
        """
        att, dfn, mu, home, rho = self._unpack(theta, n)
        eta_h = mu + home + att[hi] - dfn[ai]
        eta_a = mu + att[ai] - dfn[hi]
        lam, lmb = np.exp(eta_h), np.exp(eta_a)

        m00 = (x == 0) & (y == 0)
        m01 = (x == 0) & (y == 1)
        m10 = (x == 1) & (y == 0)
        m11 = (x == 1) & (y == 1)
        t = np.ones_like(lam)
        t[m00] = 1.0 - lam[m00] * lmb[m00] * rho
        t[m01] = 1.0 + lam[m01] * rho
        t[m10] = 1.0 + lmb[m10] * rho
        t[m11] = 1.0 - rho
        t = np.maximum(t, 1e-10)

        ll = x * eta_h - lam + y * eta_a - lmb + np.log(t)
        nll = -(w * ll).sum() + const + 0.5 * self.ridge * (att @ att + dfn @ dfn)

        # pochodne log(tau) po eta_h, eta_a, rho
        dt_h = np.zeros_like(lam)
        dt_a = np.zeros_like(lam)
        dt_r = np.zeros_like(lam)
        dt_h[m00] = -lam[m00] * lmb[m00] * rho
        dt_a[m00] = -lam[m00] * lmb[m00] * rho
        dt_r[m00] = -lam[m00] * lmb[m00]
        dt_h[m01] = lam[m01] * rho
        dt_r[m01] = lam[m01]
        dt_a[m10] = lmb[m10] * rho
        dt_r[m10] = lmb[m10]
        dt_r[m11] = -1.0
        g_h = w * ((x - lam) + dt_h / t)
        g_a = w * ((y - lmb) + dt_a / t)
        g_r = (w * dt_r / t).sum()

        g_att = np.bincount(hi, g_h, n) + np.bincount(ai, g_a, n)
        g_def = -np.bincount(ai, g_h, n) - np.bincount(hi, g_a, n)
        g_att = -g_att + self.ridge * att
        g_def = -g_def + self.ridge * dfn
        if prior is not None:
            kappa, att0, def0 = prior
            nll += 0.5 * (kappa * ((att - att0) ** 2 + (dfn - def0) ** 2)).sum()
            g_att += kappa * (att - att0)
            g_def += kappa * (dfn - def0)
        grad = np.concatenate([
            g_att[:-1] - g_att[-1],
            g_def[:-1] - g_def[-1],
            [-(g_h.sum() + g_a.sum()), -g_h.sum(), -g_r],
        ])
        return nll, grad

    def fit(self, matches: pd.DataFrame, ref_date: pd.Timestamp | None = None,
            warm_start: "DixonColes | None" = None) -> "DixonColes":
        """Dopasuj model do meczów (kolumny: date, home, away, hg, ag).

        ref_date: data, względem której liczone są wagi czasowe (domyślnie dzień po
        ostatnim meczu). Model widzi WYŁĄCZNIE przekazane mecze.
        """
        df = matches.dropna(subset=["hg", "ag"])
        if df.empty:
            raise ValueError("Brak meczów do dopasowania modelu")
        self.train_max_date = pd.Timestamp(df["date"].max())
        self.ref_date = pd.Timestamp(ref_date) if ref_date is not None else self.train_max_date + pd.Timedelta(days=1)
        teams = sorted(set(df["home"]) | set(df["away"]))
        n = len(teams)
        idx = {t: k for k, t in enumerate(teams)}
        hi = df["home"].map(idx).to_numpy()
        ai = df["away"].map(idx).to_numpy()
        x = df["hg"].to_numpy(dtype=float)
        y = df["ag"].to_numpy(dtype=float)
        w = time_weights(df["date"], self.ref_date, self.xi)
        const = float((w * (gammaln(x + 1) + gammaln(y + 1))).sum())

        theta0 = np.zeros(2 * n + 1)
        mean_goals = max((x.mean() + y.mean()) / 2, 0.1)
        theta0[2 * n - 2] = np.log(mean_goals)
        theta0[2 * n - 1] = 0.25
        theta0[2 * n] = -0.05
        if warm_start is not None and warm_start.teams:
            prev = dict(zip(warm_start.teams, zip(warm_start.attack, warm_start.defence)))
            att0 = np.array([prev.get(t, (0.0, 0.0))[0] for t in teams])
            def0 = np.array([prev.get(t, (0.0, 0.0))[1] for t in teams])
            att0 -= att0.mean()
            def0 -= def0.mean()
            theta0[: n - 1] = att0[:-1]
            theta0[n - 1: 2 * n - 2] = def0[:-1]
            theta0[2 * n - 2: 2 * n + 1] = [warm_start.mu, warm_start.home_adv,
                                            np.clip(warm_start.rho, *self.rho_bounds)]

        bounds = [(None, None)] * (2 * n) + [self.rho_bounds]
        args = (hi, ai, x, y, w, n, const)
        res = minimize(self._nll, theta0, args=args + (None,), jac=True,
                       method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
        counts = pd.concat([df["home"], df["away"]]).value_counts()
        self.n_matches = {t: int(counts.get(t, 0)) for t in teams}
        self.teams = teams
        self._set_params(res)
        self._set_fallback()

        # Drużyny z małą liczbą meczów: drugi przebieg z priorem wokół fallbacku,
        # którego siła maleje liniowo do zera przy min_matches meczach.
        share = np.array([max(0.0, 1.0 - self.n_matches[t] / self.min_matches) if self.min_matches else 0.0
                          for t in teams])
        self.low_data = {t: float(sh) for t, sh in zip(teams, share) if sh > 0}
        if self.low_data:
            kappa = self.fallback_strength * share
            prior = (kappa, np.where(share > 0, self.fallback_att, 0.0),
                     np.where(share > 0, self.fallback_def, 0.0))
            res = minimize(self._nll, res.x, args=args + (prior,), jac=True,
                           method="L-BFGS-B", bounds=bounds, options={"maxiter": 2000})
            self._set_params(res)
            self._set_fallback()
        return self

    def _set_params(self, res: Any) -> None:
        att, dfn, mu, home, rho = self._unpack(res.x, len(self.teams))
        self.attack, self.defence = att, dfn
        self.mu, self.home_adv, self.rho = float(mu), float(home), float(rho)
        self.converged = bool(res.success)
        self.loglik = float(-res.fun)

    def _set_fallback(self) -> None:
        """Fallback = średnia `fallback_bottom_k` najsłabszych drużyn z wystarczającą historią."""
        enough = [k for k, t in enumerate(self.teams) if self.n_matches[t] >= max(self.min_matches, 1)]
        if not enough:
            self.fallback_att = self.fallback_def = 0.0
            return
        strength = self.attack[enough] + self.defence[enough]
        k = min(self.fallback_bottom_k, len(enough))
        weakest = np.array(enough)[np.argsort(strength)[:k]]
        self.fallback_att = float(self.attack[weakest].mean())
        self.fallback_def = float(self.defence[weakest].mean())

    # ---------------------------------------------------------------- predykcja

    def team_params(self, team: str) -> tuple[float, float, bool]:
        """(atak, obrona, czy_fallback). Drużyna bez historii dostaje fallback."""
        if team in self.teams:
            k = self.teams.index(team)
            return float(self.attack[k]), float(self.defence[k]), team in self.low_data
        return self.fallback_att, self.fallback_def, True

    def expected_goals(self, home: str, away: str) -> tuple[float, float]:
        ah, dh, _ = self.team_params(home)
        aa, da, _ = self.team_params(away)
        return (float(np.exp(self.mu + self.home_adv + ah - da)),
                float(np.exp(self.mu + aa - dh)))

    def predict(self, home: str, away: str) -> MatchPrediction:
        """Pełna predykcja meczu: macierz wyników i rynki."""
        lam, lmb = self.expected_goals(home, away)
        m = score_matrix(lam, lmb, self.rho, self.max_goals)
        return MatchPrediction(home, away, lam, lmb, m, markets_from_matrix(m),
                               self.team_params(home)[2], self.team_params(away)[2])

    def predict_many(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """Wektorowa predykcja dla wielu meczów (kolumny home, away)."""
        if fixtures.empty:
            return pd.DataFrame(columns=["lam_home", "lam_away", "p_home", "p_draw", "p_away",
                                         "p_over25", "p_btts", "home_fallback", "away_fallback"])
        hp = [self.team_params(t) for t in fixtures["home"]]
        ap = [self.team_params(t) for t in fixtures["away"]]
        ah, dh, fh = map(np.array, zip(*hp))
        aa, da, fa = map(np.array, zip(*ap))
        lam = np.exp(self.mu + self.home_adv + ah - da)
        lmb = np.exp(self.mu + aa - dh)
        out = matrices_to_markets(score_matrices(lam, lmb, self.rho, self.max_goals))
        out.insert(0, "lam_home", lam)
        out.insert(1, "lam_away", lmb)
        out["home_fallback"] = fh.astype(bool)
        out["away_fallback"] = fa.astype(bool)
        out.index = fixtures.index
        return out

    def ratings(self) -> pd.DataFrame:
        """Tabela siły drużyn: atak, obrona, siła ogólna, liczba meczów, fallback."""
        df = pd.DataFrame({
            "druzyna": self.teams,
            "atak": self.attack,
            "obrona": self.defence,
            "mecze": [self.n_matches[t] for t in self.teams],
            "fallback": [self.low_data.get(t, 0.0) for t in self.teams],
        })
        df["sila"] = df["atak"] + df["obrona"]
        # oczekiwane gole w meczu z przeciętnym rywalem na neutralnym terenie
        df["gole_zdobyte"] = np.exp(self.mu + df["atak"])
        df["gole_stracone"] = np.exp(self.mu - df["obrona"])
        return df.sort_values("sila", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- symulacja


def simulate_matches(attack: np.ndarray, defence: np.ndarray, mu: float, home_adv: float,
                     rho: float, n_rounds: int = 2, start: str = "2020-08-01",
                     seed: int = 0, max_goals: int = 10) -> pd.DataFrame:
    """Symuluj mecze (każdy z każdym, n_rounds razy u siebie) z rozkładu Dixona-Colesa."""
    rng = np.random.default_rng(seed)
    n = len(attack)
    teams = [f"T{k:02d}" for k in range(n)]
    pairs = [(h, a) for h in range(n) for a in range(n) if h != a]
    rows = []
    day = pd.Timestamp(start)
    for r in range(n_rounds):
        order = rng.permutation(len(pairs))
        for k, p in enumerate(order):
            h, a = pairs[p]
            lam = np.exp(mu + home_adv + attack[h] - defence[a])
            lmb = np.exp(mu + attack[a] - defence[h])
            m = score_matrix(lam, lmb, rho, max_goals)
            cell = rng.choice(m.size, p=m.ravel())
            rows.append((day + pd.Timedelta(days=int((r * len(pairs) + k) // (n // 2)) * 3.5),
                         teams[h], teams[a], cell // m.shape[0], cell % m.shape[0]))
    return pd.DataFrame(rows, columns=["date", "home", "away", "hg", "ag"])
