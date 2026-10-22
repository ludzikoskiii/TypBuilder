"""Walk-forward backtest modelu Dixona-Colesa względem rynku.

Zasada: każdy mecz testowy jest przewidywany modelem dopasowanym WYŁĄCZNIE do meczów
rozegranych przed datą odcięcia (cutoff) jego bloku. Bloki zaczynają się co
`refit_days` dni (domyślnie 3, czyli osobno kolejka weekendowa i środkowa tygodnia).
Strojenie xi odbywa się na sezonach walidacyjnych przed pierwszym sezonem testowym.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from config import load_config
from data import load_league
from market import fair_probs, kelly_stake
from model import DixonColes

ProgressFn = Callable[[float, str], None]


# --------------------------------------------------------------------------- rynki


@dataclass(frozen=True)
class MarketSpec:
    key: str
    name: str
    outcomes: tuple[str, ...]
    model_cols: tuple[str, ...]
    odds_cols: tuple[str, ...]
    closing_cols: tuple[str, ...]


MARKETS: dict[str, MarketSpec] = {
    "1X2": MarketSpec("1X2", "1X2", ("1", "X", "2"), ("p_home", "p_draw", "p_away"),
                      ("odds_h", "odds_d", "odds_a"),
                      ("closing_odds_h", "closing_odds_d", "closing_odds_a")),
    "OU25": MarketSpec("OU25", "Over/Under 2.5", ("Over 2.5", "Under 2.5"), ("p_over25", "p_under25"),
                       ("odds_over25", "odds_under25"),
                       ("closing_odds_over25", "closing_odds_under25")),
}


def outcome_index(df: pd.DataFrame, market: str) -> np.ndarray:
    """Indeks faktycznego wyniku w danym rynku (np. 0=1, 1=X, 2=2)."""
    hg, ag = df["hg"].to_numpy(), df["ag"].to_numpy()
    if market == "1X2":
        return np.where(hg > ag, 0, np.where(hg == ag, 1, 2))
    if market == "OU25":
        return np.where(hg + ag > 2.5, 0, 1)
    raise ValueError(market)


# --------------------------------------------------------------------------- walk-forward


def block_cutoffs(dates: pd.Series, refit_days: int) -> list[pd.Timestamp]:
    """Daty rozpoczęcia bloków: nowy blok, gdy minęło >= refit_days od początku poprzedniego."""
    cutoffs: list[pd.Timestamp] = []
    for d in sorted(pd.to_datetime(dates).unique()):
        d = pd.Timestamp(d)
        if not cutoffs or (d - cutoffs[-1]).days >= refit_days:
            cutoffs.append(d)
    return cutoffs


def walk_forward_predict(matches: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                         model_kwargs: dict[str, Any], train_days: int = 1095,
                         refit_days: int = 3, min_train: int = 150,
                         progress: ProgressFn | None = None) -> pd.DataFrame:
    """Predykcje out-of-sample dla meczów z [start, end).

    Dla każdego bloku model jest trenowany na meczach z [cutoff - train_days, cutoff).
    Zwraca kopię meczów testowych z kolumnami predykcji, `cutoff` i `train_max_date`.
    """
    m = matches.sort_values("date").reset_index(drop=True)
    test = m[(m["date"] >= start) & (m["date"] < end)]
    if test.empty:
        return pd.DataFrame()
    cutoffs = block_cutoffs(test["date"], refit_days)
    out: list[pd.DataFrame] = []
    prev: DixonColes | None = None
    for i, cut in enumerate(cutoffs):
        nxt = cutoffs[i + 1] if i + 1 < len(cutoffs) else pd.Timestamp(end)
        block = test[(test["date"] >= cut) & (test["date"] < nxt)]
        train = m[(m["date"] < cut) & (m["date"] >= cut - pd.Timedelta(days=train_days))]
        if len(train) < min_train or block.empty:
            continue
        # Twarda gwarancja braku wycieku danych z przyszłości.
        if train["date"].max() >= block["date"].min():
            raise AssertionError("Wyciek danych: trening zawiera mecze z okresu testowego")
        model = DixonColes(**model_kwargs).fit(train, ref_date=cut, warm_start=prev)
        prev = model
        pred = model.predict_many(block)
        res = block.copy()
        for col in pred.columns:
            res[col] = pred[col].to_numpy()
        res["p_under25"] = 1.0 - res["p_over25"]
        res["cutoff"] = cut
        res["train_max_date"] = train["date"].max()
        res["n_train"] = len(train)
        out.append(res)
        if progress:
            progress((i + 1) / len(cutoffs), f"Blok {i + 1}/{len(cutoffs)} ({cut:%Y-%m-%d})")
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def log_loss(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Log-loss per mecz: -log p(faktyczny wynik)."""
    return -np.log(np.clip(p[np.arange(len(y)), y], 1e-15, 1.0))


def brier(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Wieloklasowy Brier score per mecz: sum_k (p_k - 1[y=k])^2."""
    onehot = np.eye(p.shape[1])[y]
    return ((p - onehot) ** 2).sum(axis=1)


def rps(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ranked probability score (dla rynków uporządkowanych, np. 1X2)."""
    onehot = np.eye(p.shape[1])[y]
    cp, co = np.cumsum(p, axis=1)[:, :-1], np.cumsum(onehot, axis=1)[:, :-1]
    return ((cp - co) ** 2).sum(axis=1) / (p.shape[1] - 1)


def tune_xi(matches: pd.DataFrame, val_start: pd.Timestamp, val_end: pd.Timestamp,
            grid: list[float], model_kwargs: dict[str, Any], train_days: int = 1095,
            refit_days: int = 7, progress: ProgressFn | None = None) -> tuple[float, pd.DataFrame]:
    """Wybierz xi minimalizujące log-loss 1X2 w walk-forward na okresie walidacyjnym."""
    rows = []
    for k, xi in enumerate(grid):
        kw = {**model_kwargs, "xi": xi}
        pred = walk_forward_predict(matches, val_start, val_end, kw, train_days, refit_days)
        if pred.empty:
            continue
        y = outcome_index(pred, "1X2")
        p = pred[list(MARKETS["1X2"].model_cols)].to_numpy()
        rows.append({"xi": xi, "log_loss": log_loss(p, y).mean(), "brier": brier(p, y).mean(),
                     "mecze": len(pred)})
        if progress:
            progress((k + 1) / len(grid), f"Strojenie xi: {xi}")
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Brak danych walidacyjnych do strojenia xi")
    best = float(table.loc[table["log_loss"].idxmin(), "xi"])
    return best, table


# --------------------------------------------------------------------------- ewaluacja


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000,
                 stat: Callable[[np.ndarray], np.ndarray] | None = None,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentylowy przedział ufności bootstrap dla średniej (lub innej statystyki po osi 1)."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    sample = values[idx]
    stats = stat(sample) if stat else sample.mean(axis=1)
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def calibration_table(p: np.ndarray, y: np.ndarray, source: str, bins: int = 10) -> pd.DataFrame:
    """Kalibracja: wszystkie wyniki rynku razem, przedziały prawdopodobieństwa co 1/bins."""
    onehot = np.eye(p.shape[1])[y]
    pf, of = p.ravel(), onehot.ravel()
    edges = np.linspace(0, 1, bins + 1)
    b = np.clip(np.digitize(pf, edges) - 1, 0, bins - 1)
    df = pd.DataFrame({"bin": b, "p": pf, "o": of})
    g = df.groupby("bin").agg(p_srednie=("p", "mean"), czestosc=("o", "mean"), n=("o", "size")).reset_index()
    g["zrodlo"] = source
    return g


def max_drawdown(curve: np.ndarray, start: float = 0.0) -> float:
    """Największy spadek od szczytu (w jednostkach krzywej)."""
    full = np.concatenate([[start], np.asarray(curve, dtype=float)])
    return float((np.maximum.accumulate(full) - full).max())


@dataclass
class MarketReport:
    """Wyniki backtestu dla jednego rynku."""
    spec: MarketSpec
    n_matches: int
    metrics: pd.DataFrame
    ll_diff: float
    ll_diff_ci: tuple[float, float]
    calibration: pd.DataFrame
    bets: pd.DataFrame
    betting: dict[str, Any]
    clv: dict[str, Any] | None
    by_season: pd.DataFrame
    by_type: pd.DataFrame
    odds_are_closing: bool


def simulate_bets(bets: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Symulacja stawek płaskich i ułamkowego Kelly'ego (stawki dnia liczone od bankrolla z rana)."""
    st = cfg["staking"]
    bets = bets.sort_values(["date", "home"]).reset_index(drop=True)
    flat = float(st["flat_stake"])
    bets["stake_flat"] = flat
    bets["profit_flat"] = np.where(bets["won"], flat * (bets["odds"] - 1), -flat)
    bankroll = float(st["bankroll"])
    stakes = np.zeros(len(bets))
    after = np.zeros(len(bets))
    for _, idx in bets.groupby("date").groups.items():
        idx = np.asarray(idx)
        s = kelly_stake(bets.loc[idx, "p_model"].to_numpy(), bets.loc[idx, "odds"].to_numpy(),
                        bankroll, st["kelly_fraction"], st["max_stake_pct"])
        won = bets.loc[idx, "won"].to_numpy()
        pnl = np.where(won, s * (bets.loc[idx, "odds"].to_numpy() - 1), -s)
        stakes[idx] = s
        bankroll += pnl.sum()
        after[idx] = bankroll
    bets["stake_kelly"] = stakes
    bets["profit_kelly"] = np.where(bets["won"], stakes * (bets["odds"] - 1), -stakes)
    bets["bankroll_kelly"] = after
    bets["cum_profit_flat"] = bets["profit_flat"].cumsum()
    n = len(bets)
    summary = {
        "n_bets": n,
        "hit_rate": float(bets["won"].mean()) if n else float("nan"),
        "avg_odds": float(bets["odds"].mean()) if n else float("nan"),
        "avg_value": float(bets["value"].mean()) if n else float("nan"),
        "staked_flat": flat * n,
        "profit_flat": float(bets["profit_flat"].sum()),
        "roi_flat": float(bets["profit_flat"].sum() / (flat * n)) if n else float("nan"),
        "max_dd_flat": max_drawdown(bets["cum_profit_flat"].to_numpy()) if n else 0.0,
        "start_bankroll": float(st["bankroll"]),
        "end_bankroll_kelly": float(bankroll),
        "staked_kelly": float(stakes.sum()),
        "roi_kelly": float(bets["profit_kelly"].sum() / stakes.sum()) if stakes.sum() > 0 else float("nan"),
    }
    if n:
        curve = np.concatenate([[st["bankroll"]], after])
        peak = np.maximum.accumulate(curve)
        summary["max_dd_kelly_pct"] = float(((peak - curve) / peak).max())
    else:
        summary["max_dd_kelly_pct"] = 0.0
    return bets, summary


def evaluate_market(pred: pd.DataFrame, spec: MarketSpec, cfg: dict[str, Any],
                    rng: np.random.Generator) -> MarketReport | None:
    """Metryki model vs rynek, kalibracja, symulacja zakładów value i CLV dla jednego rynku."""
    odds_all = pred[list(spec.odds_cols)].to_numpy(dtype=float)
    ok = np.isfinite(odds_all).all(axis=1) & (odds_all > 1).all(axis=1)
    if ok.sum() < 50:
        return None
    df = pred[ok].reset_index(drop=True)
    odds = odds_all[ok]
    y = outcome_index(df, spec.key)
    p_model = df[list(spec.model_cols)].to_numpy(dtype=float)
    method = cfg["market"]["margin_method"]
    p_mkt = fair_probs(odds, method)
    p_prop = fair_probs(odds, "proportional" if method == "shin" else "shin")
    nb = cfg["backtest"]["bootstrap_samples"]
    odds_are_closing = bool(df["odds_are_closing"].all())

    def row(name: str, p: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
        pp, yy = (p, y) if mask is None else (p[mask], y[mask])
        r = {"zrodlo": name, "mecze": len(yy), "log_loss": log_loss(pp, yy).mean(),
             "brier": brier(pp, yy).mean()}
        if spec.key == "1X2":
            r["rps"] = rps(pp, yy).mean()
        return r

    other = "proporcjonalnie" if method == "shin" else "Shin"
    mname = "Shin" if method == "shin" else "proporcjonalnie"
    label_mkt = f"Rynek ({'zamknięcie, ' if odds_are_closing else ''}{mname})"
    rows = [row("Model DC", p_model), row(label_mkt, p_mkt),
            row(f"Rynek ({'zamknięcie, ' if odds_are_closing else ''}{other})", p_prop)]

    clo = df[list(spec.closing_cols)].to_numpy(dtype=float)
    has_clo = np.isfinite(clo).all(axis=1) & (clo > 1).all(axis=1)
    p_clo = np.full_like(p_mkt, np.nan)
    if has_clo.any():
        p_clo[has_clo] = fair_probs(clo[has_clo], method)
    if has_clo.sum() >= 50 and not odds_are_closing:
        if has_clo.all():
            rows.append(row(f"Rynek zamknięcia ({mname})", p_clo))
        else:
            rows.append(row("Model DC (mecze z kursem zamkn.)", p_model, has_clo))
            rows.append(row(f"Rynek zamknięcia ({mname})", p_clo, has_clo))
    metrics = pd.DataFrame(rows)

    diff = log_loss(p_model, y) - log_loss(p_mkt, y)
    ll_ci = bootstrap_ci(diff, rng, nb)

    calib = pd.concat([calibration_table(p_model, y, "Model DC"),
                       calibration_table(p_mkt, y, "Rynek")], ignore_index=True)

    # --- zakłady value: najwyżej jeden typ na mecz (największe value powyżej progu)
    bt = cfg["backtest"]
    thr = cfg["market"]["value_threshold"]
    val = p_model * odds - 1.0
    eligible = (val > thr) & (odds >= bt["min_odds"]) & (odds <= bt["max_odds"])
    val_masked = np.where(eligible, val, -np.inf)
    pick = val_masked.argmax(axis=1)
    has_bet = eligible.any(axis=1)
    rows_b = np.where(has_bet)[0]
    sel = pick[rows_b]
    bets = pd.DataFrame({
        "date": df.loc[rows_b, "date"].to_numpy(),
        "season": df.loc[rows_b, "season"].to_numpy(),
        "home": df.loc[rows_b, "home"].to_numpy(),
        "away": df.loc[rows_b, "away"].to_numpy(),
        "wynik": (df.loc[rows_b, "hg"].astype(str) + ":" + df.loc[rows_b, "ag"].astype(str)).to_numpy(),
        "typ": [spec.outcomes[s] for s in sel],
        "odds": odds[rows_b, sel],
        "p_model": p_model[rows_b, sel],
        "p_market": p_mkt[rows_b, sel],
        "value": val[rows_b, sel],
        "won": y[rows_b] == sel,
        "closing_odds": np.where(has_clo[rows_b], clo[rows_b, sel], np.nan),
        "p_closing": p_clo[rows_b, sel],
    })
    bets, betting = simulate_bets(bets, cfg)
    if len(bets) >= 2:
        betting["roi_ci"] = bootstrap_ci(bets["profit_flat"].to_numpy() / cfg["staking"]["flat_stake"], rng, nb)
    else:
        betting["roi_ci"] = (float("nan"), float("nan"))

    # --- CLV: czy kurs w momencie typu był lepszy od kursu zamknięcia
    clv: dict[str, Any] | None = None
    if not odds_are_closing:
        cb = bets[bets["closing_odds"].notna()]
        if len(cb) >= 2:
            ev_close = (cb["odds"] * cb["p_closing"] - 1).to_numpy()
            clv = {
                "n": len(cb),
                "mean_clv": float(ev_close.mean()),
                "clv_ci": bootstrap_ci(ev_close, rng, nb),
                "beat_close_pct": float((cb["odds"] > cb["closing_odds"]).mean()),
                "mean_price_move": float((cb["odds"] / cb["closing_odds"] - 1).mean()),
            }
        bets["clv"] = bets["odds"] * bets["p_closing"] - 1

    by_type = (bets.groupby("typ")
               .agg(zaklady=("won", "size"), trafnosc=("won", "mean"), sr_kurs=("odds", "mean"),
                    sr_p_model=("p_model", "mean"), sr_p_rynek=("p_market", "mean"),
                    zysk=("profit_flat", "sum"))
               .reset_index())
    by_type["roi"] = by_type["zysk"] / (by_type["zaklady"] * cfg["staking"]["flat_stake"])

    seasons = []
    for season, g in df.groupby("season"):
        mask = (df["season"] == season).to_numpy()
        sb = bets[bets["season"] == season]
        seasons.append({
            "sezon": season, "mecze": int(mask.sum()),
            "log_loss_model": log_loss(p_model[mask], y[mask]).mean(),
            "log_loss_rynek": log_loss(p_mkt[mask], y[mask]).mean(),
            "zaklady": len(sb),
            "roi_flat": sb["profit_flat"].sum() / (len(sb) * cfg["staking"]["flat_stake"]) if len(sb) else np.nan,
        })
    return MarketReport(spec, len(df), metrics, float(diff.mean()), ll_ci, calib, bets, betting, clv,
                        pd.DataFrame(seasons), by_type, odds_are_closing)


# --------------------------------------------------------------------------- werdykt


def _pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


def market_verdict(rep: MarketReport) -> tuple[str, str]:
    """(status, uzasadnienie) dla jednego rynku. status: 'TAK' / 'NIEJEDNOZNACZNY' / 'NIE'."""
    d, (lo, hi) = rep.ll_diff, rep.ll_diff_ci
    ci = f"95% CI różnicy [{lo:+.4f}, {hi:+.4f}]"
    if hi < 0:
        ll_txt, ll_good = f"log-loss lepszy od rynku o {abs(d):.4f}, istotnie ({ci})", True
    elif lo > 0:
        ll_txt, ll_good = f"log-loss gorszy od rynku o {d:.4f}, istotnie ({ci})", False
    else:
        ll_txt, ll_good = (f"log-loss {'gorszy' if d > 0 else 'lepszy'} od rynku o {abs(d):.4f}, "
                           f"ale różnica nieistotna ({ci} obejmuje zero)"), False
    b = rep.betting
    if b["n_bets"] == 0:
        roi_txt, roi_good = "brak zakładów value", False
    else:
        rlo, rhi = b["roi_ci"]
        where = "powyżej zera" if rlo > 0 else ("poniżej zera" if rhi < 0 else "obejmuje zero")
        roi_txt = (f"ROI {_pct(b['roi_flat'])} w {b['n_bets']} zakładach, "
                   f"95% CI [{_pct(rlo)}, {_pct(rhi)}] {where}")
        roi_good = rlo > 0
    if ll_good and roi_good:
        status = "TAK"
    elif ll_good or roi_good:
        status = "NIEJEDNOZNACZNY"
    else:
        status = "NIE"
    return status, f"{ll_txt}; {roi_txt}"


@dataclass
class BacktestResult:
    league: str
    league_name: str
    test_seasons: list[str]
    validation_seasons: list[str]
    xi: float
    xi_table: pd.DataFrame | None
    predictions: pd.DataFrame
    reports: dict[str, MarketReport]
    verdict: str = ""
    notes: list[str] = field(default_factory=list)


def build_verdict(res: BacktestResult) -> tuple[str, list[str]]:
    """Werdykt tekstowy - zła wiadomość jest podawana wprost."""
    notes: list[str] = []
    main = res.reports.get("1X2")
    if main is None:
        return "Brak wystarczających danych z kursami 1X2 - werdykt niemożliwy.", notes
    status, why = market_verdict(main)
    name = res.league_name
    if status == "TAK":
        verdict = (f"Model POKONAŁ rynek 1X2 w lidze {name} w tej próbie ({why}). "
                   f"To wynik historyczny - nie gwarantuje zysku w przyszłości.")
    elif status == "NIEJEDNOZNACZNY":
        verdict = (f"Wynik NIEJEDNOZNACZNY w lidze {name} ({why}). Jedna miara wygląda dobrze, "
                   f"druga nie - nie ma podstaw, by twierdzić, że model ma przewagę.")
    else:
        verdict = f"Model NIE pokonuje rynku 1X2 w lidze {name} ({why})."
    if main.clv is not None:
        c = main.clv
        notes.append(f"CLV (1X2, {c['n']} zakładów): kurs w momencie typu vs kurs zamknięcia średnio "
                     f"{_pct(c['mean_price_move'])}, lepszy od zamknięcia w {c['beat_close_pct'] * 100:.0f}% "
                     f"zakładów. Wartość oczekiwana względem zamknięcia bez marży: {_pct(c['mean_clv'])} "
                     f"(95% CI [{_pct(c['clv_ci'][0])}, {_pct(c['clv_ci'][1])}]) - ujemna oznacza, że rynek "
                     f"przed startem meczu poszedł przeciw typom lub marża zjada przewagę.")
    elif main.odds_are_closing:
        notes.append("CLV niedostępne: plik zawiera tylko kursy zamknięcia, więc zakłady symulowano "
                     "po kursach zamknięcia (najbardziej efektywny moment rynku - trudniejszy test).")
    ou = res.reports.get("OU25")
    if ou is not None:
        st, why_ou = market_verdict(ou)
        notes.append(f"Over/Under 2.5: {'model lepszy' if st == 'TAK' else 'niejednoznacznie' if st == 'NIEJEDNOZNACZNY' else 'model nie pokonuje rynku'} ({why_ou}).")
    if main.betting["n_bets"] and main.betting["n_bets"] < 200:
        notes.append(f"Mała próba ({main.betting['n_bets']} zakładów) - przedział ROI jest szeroki.")
    fb = res.predictions[["home_fallback", "away_fallback"]].any(axis=1).mean()
    if fb > 0:
        notes.append(f"{fb * 100:.1f}% meczów testowych z udziałem beniaminka/drużyny z małą liczbą "
                     f"meczów (fallback).")
    return verdict, notes


def run_backtest(matches: pd.DataFrame, test_seasons: list[str], cfg: dict[str, Any] | None = None,
                 league: str = "", league_name: str = "", tune: bool = True, xi: float | None = None,
                 progress: ProgressFn | None = None) -> BacktestResult:
    """Pełny backtest na przekazanych meczach: strojenie xi -> walk-forward -> metryki -> werdykt."""
    cfg = cfg or load_config()
    mcfg, bcfg = cfg["model"], cfg["backtest"]
    rng = np.random.default_rng(bcfg.get("seed", 42))
    matches = matches.sort_values("date").reset_index(drop=True)
    all_seasons = sorted(matches["season"].unique())
    test_seasons = sorted(test_seasons)
    test = matches[matches["season"].isin(test_seasons)]
    if test.empty:
        raise ValueError("Brak meczów w wybranych sezonach testowych")
    test_start, test_end = test["date"].min(), test["date"].max() + pd.Timedelta(days=1)
    first = all_seasons.index(test_seasons[0])
    val_seasons = all_seasons[max(0, first - bcfg.get("validation_seasons", 1)): first]
    model_kwargs = {k: mcfg[k] for k in ("max_goals", "ridge", "min_matches", "fallback_bottom_k",
                                         "fallback_strength") if k in mcfg}
    train_days = mcfg.get("train_days", 1095)
    notes: list[str] = []

    xi_table = None
    if xi is None:
        xi = mcfg["xi"]
        if tune and val_seasons:
            val = matches[matches["season"].isin(val_seasons)]
            # okres walidacyjny kończy się przed pierwszym meczem testowym
            val_end = min(val["date"].max() + pd.Timedelta(days=1), test_start)
            try:
                xi, xi_table = tune_xi(matches[matches["date"] < test_start], val["date"].min(), val_end,
                                       mcfg["xi_grid"], model_kwargs, train_days,
                                       progress=(lambda f, t: progress(0.4 * f, t)) if progress else None)
            except ValueError as exc:
                notes.append(f"Strojenie xi pominięte: {exc}")
        elif tune:
            notes.append("Brak sezonu walidacyjnego przed testem - xi z konfiguracji.")

    pred = walk_forward_predict(matches, test_start, test_end, {**model_kwargs, "xi": xi}, train_days,
                                bcfg.get("refit_days", 3),
                                progress=(lambda f, t: progress(0.4 + 0.55 * f, t)) if progress else None)
    if pred.empty:
        raise ValueError("Za mało danych treningowych przed okresem testowym")
    reports = {}
    for key, spec in MARKETS.items():
        rep = evaluate_market(pred, spec, cfg, rng)
        if rep is not None:
            reports[key] = rep
    res = BacktestResult(league, league_name or league, test_seasons, val_seasons, float(xi), xi_table,
                         pred, reports)
    res.verdict, extra = build_verdict(res)
    res.notes = notes + extra
    if progress:
        progress(1.0, "Gotowe")
    return res


def run_league_backtest(league: str, cfg: dict[str, Any] | None = None, n_test_seasons: int = 3,
                        test_seasons: list[str] | None = None, tune_xi: bool = True,
                        offline: bool = False, progress: ProgressFn | None = None) -> BacktestResult:
    """Wczytaj dane ligi i uruchom backtest na `n_test_seasons` ostatnich sezonach."""
    cfg = cfg or load_config()
    matches = load_league(league, cfg, offline=offline)
    seasons = sorted(matches["season"].unique())
    chosen = test_seasons or seasons[-n_test_seasons:]
    return run_backtest(matches, chosen, cfg, league=league, league_name=cfg["leagues"][league]["name"],
                        tune=tune_xi, progress=progress)


# --------------------------------------------------------------------------- raport


def format_report(res: BacktestResult) -> str:
    """Raport tekstowy (CLI)."""
    lines = [
        "=" * 100,
        f"BACKTEST: {res.league_name} ({res.league}) | sezony testowe: {', '.join(res.test_seasons)}",
        f"Walidacja xi: {', '.join(res.validation_seasons) or 'brak'} | wybrane xi = {res.xi}",
    ]
    if res.xi_table is not None:
        lines.append("Strojenie xi (log-loss 1X2 na walidacji):")
        lines.append(res.xi_table.round(5).to_string(index=False))
    for rep in res.reports.values():
        b = rep.betting
        lines += ["", f"--- Rynek {rep.spec.name} ({rep.n_matches} meczów) ---",
                  rep.metrics.round(4).to_string(index=False),
                  f"Różnica log-loss (model - rynek): {rep.ll_diff:+.4f}, "
                  f"95% CI [{rep.ll_diff_ci[0]:+.4f}, {rep.ll_diff_ci[1]:+.4f}]",
                  f"Zakłady value: {b['n_bets']} | trafność {b['hit_rate'] * 100:.1f}% | śr. kurs "
                  f"{b['avg_odds']:.2f} | śr. value {b['avg_value'] * 100:.1f}%" if b["n_bets"] else
                  "Zakłady value: 0"]
        if b["n_bets"]:
            lines += [
                f"  Płasko: ROI {_pct(b['roi_flat'])}, 95% CI [{_pct(b['roi_ci'][0])}, {_pct(b['roi_ci'][1])}], "
                f"zysk {b['profit_flat']:+.1f}, max obsunięcie {b['max_dd_flat']:.1f}",
                f"  Kelly: bankroll {b['start_bankroll']:.0f} -> {b['end_bankroll_kelly']:.1f}, "
                f"ROI {_pct(b['roi_kelly'])}, max obsunięcie {b['max_dd_kelly_pct'] * 100:.1f}%",
            ]
        if rep.clv:
            c = rep.clv
            lines.append(f"  CLV: ruch kursu {_pct(c['mean_price_move'])}, lepszy kurs niż zamknięcie: "
                         f"{c['beat_close_pct'] * 100:.0f}%, EV vs zamknięcie bez marży {_pct(c['mean_clv'])} "
                         f"(95% CI [{_pct(c['clv_ci'][0])}, {_pct(c['clv_ci'][1])}])")
        if b["n_bets"]:
            lines.append("Zakłady wg typu:")
            lines.append(rep.by_type.round(3).to_string(index=False))
        lines.append("Per sezon:")
        lines.append(rep.by_season.round(4).to_string(index=False))
    lines += ["", "WERDYKT: " + res.verdict] + [f"  * {n}" for n in res.notes]
    return "\n".join(lines)
