"""Typer DC - lokalna aplikacja Streamlit do analizy meczów modelem Dixona-Colesa.

Uruchomienie:  streamlit run app.py
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

import charts
import journal
from backtest import BacktestResult, load_summary, market_verdict, run_backtest, save_summary
from config import load_config
from data import data_freshness, load_fixtures, load_league
from market import fair_probs, kelly_stake
from model import DixonColes

st.set_page_config(page_title="Typer DC", page_icon="⚽", layout="wide")

CFG = load_config()
CUR = CFG["budget"].get("currency", "PLN")
VALUE_THR = CFG["market"]["value_threshold"]


# --------------------------------------------------------------------------- pomocnicze


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except AttributeError:
        return False


def pct(x: float, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "–"
    return f"{x * 100:+.1f}%" if signed else f"{x * 100:.1f}%"


def money(x: float) -> str:
    return f"{x:,.2f} {CUR}".replace(",", " ")


@st.cache_data(ttl=3600, show_spinner="Wczytywanie danych…")
def get_results(league: str, token: int) -> pd.DataFrame:
    return load_league(league, load_config())


@st.cache_data(ttl=1800, show_spinner=False)
def get_fixtures(league: str, token: int) -> pd.DataFrame:
    try:
        return load_fixtures(league, load_config(), results=get_results(league, token))
    except Exception:  # brak sieci i brak cache - mecze nadchodzące są opcjonalne
        return pd.DataFrame()


@st.cache_resource(show_spinner="Dopasowywanie modelu Dixona-Colesa…")
def get_model(league: str, xi: float, token: int) -> DixonColes:
    cfg = load_config()
    df = get_results(league, token)
    ref = pd.Timestamp(date.today())
    train = df[(df["date"] < ref) & (df["date"] >= ref - pd.Timedelta(days=cfg["model"]["train_days"]))]
    if train.empty:  # np. przerwa między sezonami dłuższa niż okno
        train = df[df["date"] < ref]
    return DixonColes.from_config(cfg, xi=xi).fit(train, ref_date=ref)


@st.cache_resource
def get_journal(path: str):
    return journal.connect(path)


def current_teams(results: pd.DataFrame, fixtures: pd.DataFrame) -> list[str]:
    last = results[results["season"] == results["season"].max()]
    teams = set(last["home"]) | set(last["away"])
    if not fixtures.empty:
        teams |= set(fixtures["home"]) | set(fixtures["away"])
    return sorted(teams)


def fallback_note(model: DixonColes, team: str) -> str | None:
    if team not in model.teams:
        return (f"⚠️ **{team}**: brak meczów w oknie treningowym – parametry = fallback "
                f"(średnia {model.fallback_bottom_k} najsłabszych drużyn ligi).")
    if team in model.low_data:
        return (f"⚠️ **{team}**: tylko {model.n_matches[team]} mecz(e/ów) w oknie – parametry ściągane "
                f"w stronę fallbacku (średnia najsłabszych drużyn). Prognoza mniej pewna.")
    return None


def render_footer() -> None:
    bg, ink, border = ("#0d0d0d", "#c3c2b7", "#2c2c2a") if is_dark() else ("#f9f9f7", "#52514e", "#e1e0d9")
    st.markdown(f"""
<style>
  .block-container {{ padding-bottom: 5.5rem; }}
  .typer-footer {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000005;
    padding: 8px 16px; font-size: 0.8rem; line-height: 1.35; text-align: center;
    background: {bg}; color: {ink}; border-top: 1px solid {border}; }}
</style>
<div class="typer-footer">
  <b>Typer DC to narzędzie analityczne, nie doradztwo.</b> Model nie gwarantuje zysku – w backtestach
  zwykle nie pokonuje rynku. Graj wyłącznie u legalnych bukmacherów (w Polsce: z zezwoleniem Ministra
  Finansów), tylko od 18 lat i tylko w ramach budżetu, którego stratę możesz zaakceptować.
</div>""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- sidebar


def sidebar() -> tuple[str, float, int]:
    st.sidebar.title("⚽ Typer DC")
    st.sidebar.caption("Model Dixona-Colesa vs rynek – uczciwa analiza, bez obietnic.")
    leagues = CFG["leagues"]
    league = st.sidebar.selectbox("Liga", list(leagues), format_func=lambda k: f"{leagues[k]['name']} ({k})",
                                  key="league")
    token = st.session_state.setdefault("refresh_token", 0)
    summary = load_summary(league, CFG)
    default_xi = float(summary["xi"]) if summary and "xi" in summary else float(CFG["model"]["xi"])
    xi = st.sidebar.number_input(
        "ξ – waga czasowa (na dzień)", min_value=0.0, max_value=0.02, value=default_xi, step=0.0005,
        format="%.4f", key=f"xi_{league}",
        help="Waga meczu = exp(-ξ · dni). Domyślnie wartość wybrana w ostatnim strojeniu (zakładka Backtest), "
             "a bez backtestu – z config.yaml.")
    if summary and "xi" in summary:
        st.sidebar.caption(f"ξ z ostatniego strojenia: {summary['xi']}")
    if st.sidebar.button("🔄 Odśwież dane z football-data.co.uk", width="stretch"):
        cfg = load_config()
        cfg["data"]["max_age_hours"] = 0
        with st.spinner("Pobieranie…"):
            try:
                df = load_league(league, cfg)
                load_fixtures(league, cfg, results=df)
                st.session_state["refresh_token"] = token + 1
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:
                st.sidebar.error(f"Nie udało się pobrać danych: {exc}")
    fresh = data_freshness(league, CFG)
    if fresh:
        st.sidebar.caption(f"Dane w cache z: {fresh:%Y-%m-%d %H:%M}")
    return league, float(xi), st.session_state["refresh_token"]


# --------------------------------------------------------------------------- zakładka 1: mecze


def honesty_banner(league: str) -> None:
    s = load_summary(league, CFG)
    if not s or "status" not in s:
        st.info("ℹ️ Dla tej ligi nie ma jeszcze backtestu. Uruchom go w zakładce **Backtest**, zanim "
                "zaczniesz ufać oznaczeniom value.")
    elif s["status"] == "NIE":
        st.warning(f"⚠️ Ostatni backtest ({s['run_at'][:10]}, sezony {', '.join(s['test_seasons'])}): "
                   f"**model NIE pokonał rynku w tej lidze** (ROI {pct(s['roi'], True)}, 95% CI "
                   f"[{pct(s['roi_ci'][0], True)}, {pct(s['roi_ci'][1], True)}]). Oznaczenie „value” to "
                   f"różnica zdań model–rynek, a nie przewaga.")
    elif s["status"] == "NIEJEDNOZNACZNY":
        st.warning(f"⚠️ Ostatni backtest ({s['run_at'][:10]}): wynik **niejednoznaczny** – brak dowodu "
                   f"przewagi modelu w tej lidze.")
    else:
        st.success(f"✅ Ostatni backtest ({s['run_at'][:10]}): model pokonał rynek w próbie historycznej. "
                   f"To nie gwarantuje wyników w przyszłości.")


def fixtures_table(fx: pd.DataFrame, model: DixonColes) -> pd.DataFrame:
    pm = model.predict_many(fx)
    out = pd.DataFrame({
        "Data": fx["date"].dt.strftime("%Y-%m-%d") + " " + fx["time"].fillna("").astype(str).str[:5],
        "Gospodarz": fx["home"], "Gość": fx["away"],
        "1 model": pm["p_home"] * 100, "X model": pm["p_draw"] * 100, "2 model": pm["p_away"] * 100,
    })
    odds = fx[["odds_h", "odds_d", "odds_a"]].to_numpy(dtype=float)
    has = np.isfinite(odds).all(axis=1) & (odds > 1).all(axis=1)
    mk = np.full_like(odds, np.nan)
    if has.any():
        mk[has] = fair_probs(odds[has], CFG["market"]["margin_method"])
    out["1 rynek"], out["X rynek"], out["2 rynek"] = mk[:, 0] * 100, mk[:, 1] * 100, mk[:, 2] * 100
    out["Kursy 1/X/2"] = [f"{o[0]:.2f} / {o[1]:.2f} / {o[2]:.2f}" if h else "–" for o, h in zip(odds, has)]
    p = pm[["p_home", "p_draw", "p_away"]].to_numpy()
    val = p * odds - 1
    labels = []
    for k in range(len(fx)):
        if not has[k]:
            labels.append("brak kursów")
            continue
        best = int(np.nanargmax(val[k]))
        labels.append(f"✅ {'1X2'[best]} ({val[k, best] * 100:+.1f}%)" if val[k, best] > VALUE_THR else "–")
    out["Value 1X2"] = labels
    out["Over 2.5 model"] = pm["p_over25"] * 100
    out["BTTS model"] = pm["p_btts"] * 100
    out["Uwagi"] = ["⚠️ beniaminek / mało danych" if (a or b) else "" for a, b in
                    zip(pm["home_fallback"], pm["away_fallback"])]
    return out


def odds_editor_rows(markets: dict[str, Any], fixture: pd.Series | None) -> pd.DataFrame:
    def fx_odds(col: str) -> float:
        if fixture is None or col not in fixture.index or pd.isna(fixture[col]):
            return np.nan
        return float(fixture[col])

    rows = [
        ("1X2", "1", markets["p_home"], fx_odds("odds_h")), ("1X2", "X", markets["p_draw"], fx_odds("odds_d")),
        ("1X2", "2", markets["p_away"], fx_odds("odds_a")),
        ("O/U 1.5", "Over", markets["over15"], np.nan), ("O/U 1.5", "Under", markets["under15"], np.nan),
        ("O/U 2.5", "Over", markets["over25"], fx_odds("odds_over25")),
        ("O/U 2.5", "Under", markets["under25"], fx_odds("odds_under25")),
        ("O/U 3.5", "Over", markets["over35"], np.nan), ("O/U 3.5", "Under", markets["under35"], np.nan),
        ("BTTS", "Tak", markets["btts_yes"], np.nan), ("BTTS", "Nie", markets["btts_no"], np.nan),
    ]
    df = pd.DataFrame(rows, columns=["Rynek", "Typ", "P model", "Kurs"])
    df["Kurs fair"] = 1 / df["P model"]
    df["P model"] = df["P model"] * 100
    return df[["Rynek", "Typ", "P model", "Kurs fair", "Kurs"]]


def value_analysis(edited: pd.DataFrame) -> pd.DataFrame:
    st_cfg = CFG["staking"]
    df = edited.copy()
    df["p"] = df["P model"] / 100
    df["P rynek"] = np.nan
    for market, g in df.groupby("Rynek"):
        o = g["Kurs"].to_numpy(dtype=float)
        if np.isfinite(o).all() and (o > 1).all():
            df.loc[g.index, "P rynek"] = fair_probs(o, CFG["market"]["margin_method"]) * 100
    df = df[df["Kurs"].notna() & (df["Kurs"] > 1)]
    df["Value"] = (df["p"] * df["Kurs"] - 1) * 100
    df["Value?"] = np.where(df["Value"] > VALUE_THR * 100, "✅ value", "–")
    df["Kelly 1/4"] = kelly_stake(df["p"].to_numpy(), df["Kurs"].to_numpy(), st_cfg["bankroll"],
                                  st_cfg["kelly_fraction"], st_cfg["max_stake_pct"])
    df["Płasko"] = np.where(df["Value"] > VALUE_THR * 100, st_cfg["flat_stake"], 0.0)
    return df[["Rynek", "Typ", "P model", "P rynek", "Kurs", "Value", "Value?", "Kelly 1/4", "Płasko"]]


def tab_matches(league: str, xi: float, token: int) -> None:
    results = get_results(league, token)
    model = get_model(league, xi, token)
    fx = get_fixtures(league, token)
    honesty_banner(league)

    st.subheader("Nadchodzące mecze")
    if fx.empty:
        st.info("Brak meczów tej ligi w aktualnych plikach fixtures football-data.co.uk (pliki są "
                "uzupełniane zwykle kilka dni przed kolejką). Możesz przeanalizować dowolną parę drużyn "
                "poniżej, wpisując kursy ręcznie.")
    else:
        tbl = fixtures_table(fx, model)
        pc = st.column_config.NumberColumn(format="%.1f%%")
        st.dataframe(tbl, hide_index=True, column_config={c: pc for c in tbl.columns if "model" in c or "rynek" in c})
        st.caption(f"Rynek = prawdopodobieństwa z kursów (średnia rynkowa) po usunięciu marży metodą "
                   f"{CFG['market']['margin_method']}. Value = p_model · kurs − 1 > {VALUE_THR:.0%}.")

    st.subheader("Szczegóły meczu")
    manual = "✏️ Wybierz drużyny ręcznie"
    options = [f"{r.date:%Y-%m-%d} {r.home} – {r.away}" for r in fx.itertuples()] + [manual]
    choice = st.selectbox("Mecz", options, key=f"match_{league}")
    teams = current_teams(results, fx)
    fixture = None
    if choice == manual:
        c1, c2 = st.columns(2)
        home = c1.selectbox("Gospodarz", teams, key=f"mh_{league}")
        away = c2.selectbox("Gość", teams, index=min(1, len(teams) - 1), key=f"ma_{league}")
        if home == away:
            st.error("Wybierz dwie różne drużyny.")
            return
    else:
        fixture = fx.iloc[options.index(choice)]
        home, away = fixture["home"], fixture["away"]

    pred = model.predict(home, away)
    mk = pred.markets
    for note in filter(None, (fallback_note(model, home), fallback_note(model, away))):
        st.warning(note)

    c = st.columns(5)
    c[0].metric(f"xG {home}", f"{pred.lam_home:.2f}")
    c[1].metric(f"xG {away}", f"{pred.lam_away:.2f}")
    c[2].metric("1", pct(mk["p_home"]))
    c[3].metric("X", pct(mk["p_draw"]))
    c[4].metric("2", pct(mk["p_away"]))

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(charts.score_heatmap(pred.matrix, home, away, is_dark()), key=f"hm_{league}")
    with right:
        st.markdown("**Najbardziej prawdopodobne wyniki**")
        st.dataframe(pd.DataFrame([{"Wynik": f"{h}:{a}", "P": p * 100} for h, a, p in mk["top_scores"]]),
                     hide_index=True, column_config={"P": st.column_config.NumberColumn(format="%.1f%%")})
        st.caption(f"ρ = {model.rho:+.3f} (korekta 0:0, 1:0, 0:1, 1:1), przewaga gospodarzy "
                   f"×{np.exp(model.home_adv):.2f} goli.")

    st.markdown("**Rynki – wpisz kursy swojego bukmachera** (prefill: średnia rynkowa z pliku fixtures, jeśli jest)")
    editor_key = f"odds_{league}_{home}_{away}"
    edited = st.data_editor(
        odds_editor_rows(mk, fixture), hide_index=True, key=editor_key, disabled=["Rynek", "Typ", "P model", "Kurs fair"],
        column_config={"P model": st.column_config.NumberColumn(format="%.1f%%"),
                       "Kurs fair": st.column_config.NumberColumn(format="%.2f", help="1 / p_model"),
                       "Kurs": st.column_config.NumberColumn(min_value=1.01, max_value=1000.0, step=0.01, format="%.2f")})
    va = value_analysis(edited)
    if va.empty:
        st.caption("Wpisz kursy, aby zobaczyć value i sugerowaną stawkę.")
    else:
        st.dataframe(va, hide_index=True, column_config={
            "P model": st.column_config.NumberColumn(format="%.1f%%"),
            "P rynek": st.column_config.NumberColumn(format="%.1f%%", help="z kursów rynku po usunięciu marży (gdy wpisano komplet kursów rynku)"),
            "Kurs": st.column_config.NumberColumn(format="%.2f"),
            "Value": st.column_config.NumberColumn(format="%+.1f%%"),
            "Kelly 1/4": st.column_config.NumberColumn(format=f"%.2f {CUR}", help="1/4 Kelly, maks. 2% bankrolla z config.yaml"),
            "Płasko": st.column_config.NumberColumn(format=f"%.2f {CUR}"),
        })
        st.caption(f"Bankroll do wyliczeń: {money(CFG['staking']['bankroll'])} (config.yaml). Value z modelu, "
                   f"który w backteście nie pokonuje rynku, to głównie szum – patrz zakładka Backtest.")


# --------------------------------------------------------------------------- zakładka 2: ranking


def tab_ratings(league: str, xi: float, token: int) -> None:
    results = get_results(league, token)
    model = get_model(league, xi, token)
    fx = get_fixtures(league, token)
    c = st.columns(5)
    c[0].metric("Przewaga gospodarzy", f"×{np.exp(model.home_adv):.2f}", help="mnożnik oczekiwanych goli gospodarzy")
    c[1].metric("ρ (Dixon-Coles)", f"{model.rho:+.3f}")
    c[2].metric("Gole/mecz",  f"{np.exp(model.mu) * (1 + np.exp(model.home_adv)):.2f}")
    c[3].metric("ξ", f"{model.xi:.4f}")
    c[4].metric("Mecze w treningu", f"{sum(model.n_matches.values()) // 2}")
    st.caption(f"Model dopasowany na meczach z ostatnich {CFG['model']['train_days']} dni "
               f"(do {model.train_max_date:%Y-%m-%d}), wagi exp(−ξ·dni).")

    only_current = st.toggle("Tylko drużyny bieżącego sezonu", value=True, key=f"cur_{league}")
    r = model.ratings()
    if only_current:
        r = r[r["druzyna"].isin(current_teams(results, fx))]
    tbl = r.assign(**{"Uwagi": [f"⚠️ mało danych ({m} meczów)" if fb > 0 else "" for m, fb in zip(r["mecze"], r["fallback"])]})
    st.dataframe(
        tbl[["druzyna", "sila", "atak", "obrona", "gole_zdobyte", "gole_stracone", "mecze", "Uwagi"]],
        hide_index=True,
        column_config={
            "druzyna": "Drużyna",
            "sila": st.column_config.NumberColumn("Siła", format="%+.3f", help="atak + obrona"),
            "atak": st.column_config.NumberColumn("Atak", format="%+.3f"),
            "obrona": st.column_config.NumberColumn("Obrona", format="%+.3f", help="więcej = lepsza obrona"),
            "gole_zdobyte": st.column_config.NumberColumn("Gole zdobyte*", format="%.2f"),
            "gole_stracone": st.column_config.NumberColumn("Gole stracone*", format="%.2f"),
            "mecze": st.column_config.NumberColumn("Mecze w oknie"),
        })
    st.caption("\\* oczekiwane gole na mecz z przeciętnym rywalem na neutralnym boisku. "
               "⚠️ = beniaminek lub mało meczów – parametry ściągane w stronę średniej najsłabszych drużyn.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.ratings_bar(r, is_dark()), key=f"rb_{league}")
    with right:
        st.plotly_chart(charts.ratings_scatter(r, is_dark()), key=f"rs_{league}")


# --------------------------------------------------------------------------- zakładka 3: backtest


@st.cache_resource
def backtest_store() -> dict[tuple, BacktestResult]:
    return {}


def cached_backtest(league: str, seasons: tuple[str, ...], tune: bool, threshold: float, method: str,
                    xi: float | None, token: int, progress=None) -> BacktestResult:
    """Backtest z pamięcią wyników. Nie używa st.cache_data, bo ten odtwarza przy trafieniu
    w cache wywołania paska postępu, który powstał poza funkcją (CacheReplayClosureError)."""
    store = backtest_store()
    key = (league, seasons, tune, threshold, method, xi, token)
    if key not in store:
        cfg = load_config()
        cfg["market"]["value_threshold"] = threshold
        cfg["market"]["margin_method"] = method
        matches = get_results(league, token)
        store[key] = run_backtest(matches, list(seasons), cfg, league=league,
                                  league_name=cfg["leagues"][league]["name"], tune=tune, xi=xi, progress=progress)
        while len(store) > 20:
            store.pop(next(iter(store)))
    return store[key]


def verdict_box(res: BacktestResult) -> None:
    main = res.reports.get("1X2")
    status = market_verdict(main)[0] if main else "NIE"
    box = {"TAK": st.success, "NIEJEDNOZNACZNY": st.warning, "NIE": st.error}[status]
    icon = {"TAK": "✅", "NIEJEDNOZNACZNY": "⚠️", "NIE": "❌"}[status]
    box(f"{icon} **Werdykt:** {res.verdict}")
    for n in res.notes:
        st.markdown(f"- {n}")


def market_section(rep, res: BacktestResult, key: str) -> None:
    b = rep.betting
    c = st.columns(5)
    c[0].metric("Δ log-loss (model − rynek)", f"{rep.ll_diff:+.4f}",
                help=f"95% CI [{rep.ll_diff_ci[0]:+.4f}, {rep.ll_diff_ci[1]:+.4f}]. Dodatnia = model gorszy.")
    c[1].metric("Zakłady value", f"{b['n_bets']}", help=f"z {rep.n_matches} meczów")
    c[2].metric("ROI (płasko)", pct(b["roi_flat"], True),
                help=f"95% CI [{pct(b['roi_ci'][0], True)}, {pct(b['roi_ci'][1], True)}] (bootstrap)")
    c[3].metric("Max obsunięcie (płasko)", f"{b['max_dd_flat']:.0f} {CUR}")
    c[4].metric("Bankroll Kelly 1/4", f"{b['end_bankroll_kelly']:.0f}",
                delta=f"{b['end_bankroll_kelly'] - b['start_bankroll']:+.0f}",
                help=f"start {b['start_bankroll']:.0f}, max obsunięcie {b['max_dd_kelly_pct'] * 100:.1f}%")
    if b["n_bets"]:
        st.caption(f"95% przedział ufności ROI: [{pct(b['roi_ci'][0], True)}, {pct(b['roi_ci'][1], True)}] · "
                   f"trafność {pct(b['hit_rate'])} · średni kurs {b['avg_odds']:.2f}")
    if rep.clv:
        cl = rep.clv
        st.caption(f"CLV: kurs przy typie vs zamknięcie {pct(cl['mean_price_move'], True)}, lepszy od zamknięcia w "
                   f"{cl['beat_close_pct'] * 100:.0f}% zakładów; EV względem zamknięcia bez marży "
                   f"{pct(cl['mean_clv'], True)} (95% CI [{pct(cl['clv_ci'][0], True)}, {pct(cl['clv_ci'][1], True)}]).")

    num = st.column_config.NumberColumn(format="%.4f")
    st.dataframe(rep.metrics, hide_index=True, column_config={"log_loss": num, "brier": num, "rps": num})
    left, right = st.columns(2)
    with left:
        st.plotly_chart(charts.calibration_chart(rep.calibration, is_dark()), key=f"cal_{key}")
    with right:
        if b["n_bets"]:
            bets = rep.bets
            st.plotly_chart(charts.line_chart(bets["date"], bets["cum_profit_flat"],
                                              f"Skumulowany zysk – stawka płaska {CFG['staking']['flat_stake']:.0f}",
                                              f"zysk ({CUR})", is_dark(), baseline=0.0), key=f"pf_{key}")
            st.plotly_chart(charts.line_chart(bets["date"], bets["bankroll_kelly"],
                                              "Bankroll – 1/4 Kelly (maks. 2% na zakład)", "bankroll", is_dark(),
                                              baseline=b["start_bankroll"]), key=f"kb_{key}")
    st.markdown("**Per sezon**")
    st.dataframe(rep.by_season, hide_index=True, column_config={
        "log_loss_model": num, "log_loss_rynek": num,
        "roi_flat": st.column_config.NumberColumn("ROI", format="percent")})
    if b["n_bets"]:
        st.markdown("**Zakłady wg typu**")
        st.dataframe(rep.by_type, hide_index=True, column_config={
            "trafnosc": st.column_config.NumberColumn(format="percent"),
            "sr_p_model": st.column_config.NumberColumn(format="percent"),
            "sr_p_rynek": st.column_config.NumberColumn(format="percent"),
            "roi": st.column_config.NumberColumn(format="percent"),
            "sr_kurs": st.column_config.NumberColumn(format="%.2f"), "zysk": st.column_config.NumberColumn(format="%.1f")})
        with st.expander(f"Wszystkie zakłady ({b['n_bets']})"):
            cols = ["date", "home", "away", "wynik", "typ", "odds", "p_model", "p_market", "value", "won",
                    "closing_odds", "profit_flat", "stake_kelly", "bankroll_kelly"]
            st.dataframe(rep.bets[cols], hide_index=True)
            st.download_button("Pobierz CSV", rep.bets.to_csv(index=False).encode("utf-8"),
                               file_name=f"backtest_{res.league}_{rep.spec.key}.csv", key=f"dl_{key}")


def tab_backtest(league: str, token: int) -> None:
    st.markdown("Walk-forward: każdy mecz przewidywany modelem trenowanym **wyłącznie** na meczach sprzed "
                "jego daty; ξ strojone na sezonach poprzedzających test. Porównanie z prawdopodobieństwami "
                "rynku bez marży.")
    leagues = CFG["leagues"]
    c1, c2 = st.columns([1, 2])
    bt_league = c1.selectbox("Liga", list(leagues), index=list(leagues).index(league),
                             format_func=lambda k: leagues[k]["name"], key="bt_league")
    results = get_results(bt_league, token)
    seasons = sorted(results["season"].unique())
    chosen = c2.multiselect("Sezony testowe", seasons, default=seasons[-3:], key=f"bt_seasons_{bt_league}")
    c3, c4, c5 = st.columns(3)
    tune = c3.checkbox("Strój ξ na sezonach walidacyjnych", value=True, key="bt_tune")
    thr = c4.slider("Próg value", 0.0, 0.30, float(VALUE_THR), 0.01, format="%.2f", key="bt_thr")
    method = c5.selectbox("Usuwanie marży", ["shin", "proportional"],
                          index=0 if CFG["market"]["margin_method"] == "shin" else 1, key="bt_method")
    if chosen and seasons.index(sorted(chosen)[0]) == 0:
        st.warning("Pierwszy wybrany sezon to najstarszy w danych – model nie ma na czym trenować przed nim. "
                   "Zostaw co najmniej 1–2 wcześniejsze sezony (zwiększ `seasons_back` w config.yaml).")
    if st.button("▶️ Uruchom backtest", type="primary", disabled=not chosen):
        bar = st.progress(0.0, text="Start…")
        try:
            res = cached_backtest(bt_league, tuple(sorted(chosen)), tune, thr, method,
                                  None if tune else float(CFG["model"]["xi"]), token,
                                  progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
        except ValueError as exc:
            bar.empty()
            st.error(f"Backtest nieudany: {exc}")
            return
        bar.empty()
        save_summary(res, CFG)
        st.session_state.setdefault("bt_results", {})[bt_league] = res

    res: BacktestResult | None = st.session_state.get("bt_results", {}).get(bt_league)
    if res is None:
        st.info("Wybierz sezony i uruchom backtest (PL: ~20–40 s przy strojeniu ξ).")
        return
    st.divider()
    st.markdown(f"### {res.league_name}: {', '.join(res.test_seasons)}")
    verdict_box(res)
    st.caption(f"Sezony walidacyjne: {', '.join(res.validation_seasons) or 'brak'} · użyte ξ = {res.xi}")
    tabs = st.tabs([f"Rynek {r.spec.name}" for r in res.reports.values()] + ["Strojenie ξ"])
    for t, (key, rep) in zip(tabs, res.reports.items()):
        with t:
            market_section(rep, res, f"{res.league}_{key}")
    with tabs[-1]:
        if res.xi_table is not None:
            st.plotly_chart(charts.xi_chart(res.xi_table, res.xi, is_dark()), key=f"xichart_{res.league}")
            st.dataframe(res.xi_table, hide_index=True)
        else:
            st.caption("ξ nie było strojone w tym przebiegu.")


# --------------------------------------------------------------------------- zakładka 4: dziennik


def tab_journal(league: str, xi: float, token: int) -> None:
    conn = get_journal(CFG["journal"]["db_path"])
    bets = journal.list_bets(conn)
    pending_leagues = sorted(set(bets.loc[bets["status"] == "pending", "league"]))
    if pending_leagues:
        res_by_league = {}
        for lg in pending_leagues:
            if lg in CFG["leagues"]:
                try:
                    res_by_league[lg] = get_results(lg, token)
                except Exception:
                    pass
        n = journal.settle_pending(conn, res_by_league)
        if n:
            st.toast(f"Rozliczono automatycznie {n} typ(y/ów).")
            bets = journal.list_bets(conn)

    bcfg = CFG["budget"]
    status = journal.budget_status(conn, bcfg["monthly_limit"], bcfg.get("warn_at_pct", 0.8))
    st.subheader(f"Budżet – {date.today():%m.%Y}")
    c = st.columns(3)
    c[0].metric("Postawiono w tym miesiącu", money(status["spent"]))
    c[1].metric("Limit miesięczny", money(status["limit"]))
    c[2].metric("Pozostało", money(status["remaining"]))
    st.progress(min(status["pct"], 1.0), text=f"{status['pct'] * 100:.0f}% budżetu")
    if status["blocked"]:
        st.error("🛑 **Miesięczny budżet wyczerpany. Dodawanie typów jest zablokowane do końca miesiąca.** "
                 "Limit zmienisz tylko w config.yaml – zastanów się dwa razy, zanim to zrobisz.")
    elif status["warn"]:
        st.warning(f"⚠️ Wykorzystano {status['pct'] * 100:.0f}% miesięcznego budżetu.")

    st.subheader("Dodaj typ")
    leagues = CFG["leagues"]
    lg = st.selectbox("Liga", list(leagues), index=list(leagues).index(league),
                      format_func=lambda k: leagues[k]["name"], key="j_league")
    results, fx = get_results(lg, token), get_fixtures(lg, token)
    mode = st.segmented_control("Mecz", ["Nadchodzący z listy", "Ręcznie"], default="Ręcznie" if fx.empty else "Nadchodzący z listy",
                                key="j_mode")
    if mode == "Nadchodzący z listy" and not fx.empty:
        opts = [f"{r.date:%Y-%m-%d} {r.home} – {r.away}" for r in fx.itertuples()]
        pick = st.selectbox("Mecz", opts, key="j_fx")
        row = fx.iloc[opts.index(pick)]
        home, away, mdate = row["home"], row["away"], row["date"].date()
    else:
        if mode == "Nadchodzący z listy":
            st.caption("Brak nadchodzących meczów w pliku fixtures – wpisz mecz ręcznie.")
        teams = current_teams(results, fx)
        c1, c2, c3 = st.columns(3)
        home = c1.selectbox("Gospodarz", teams, key="j_home")
        away = c2.selectbox("Gość", teams, index=min(1, len(teams) - 1), key="j_away")
        mdate = c3.date_input("Data meczu", value=date.today(), key="j_date")
    c1, c2, c3, c4 = st.columns(4)
    market = c1.selectbox("Rynek", list(journal.MARKETS), key="j_market")
    sel = c2.selectbox("Typ", journal.MARKETS[market], key=f"j_sel_{market}")
    odds = c3.number_input("Kurs", min_value=1.01, max_value=1000.0, value=2.0, step=0.01, key="j_odds")
    max_stake = max(status["remaining"], 0.01)
    stake = c4.number_input(f"Stawka ({CUR})", min_value=0.01, value=min(float(CFG["staking"]["flat_stake"]), max_stake),
                            step=1.0, key="j_stake")
    note = st.text_input("Notatka (opcjonalnie)", key="j_note")

    p_model = None
    if home != away:
        m = get_model(lg, xi, token)
        p_model = float(m.predict(home, away).markets[journal.MODEL_KEYS[(market, sel)]])
        v = p_model * odds - 1
        k = kelly_stake(p_model, odds, CFG["staking"]["bankroll"], CFG["staking"]["kelly_fraction"],
                        CFG["staking"]["max_stake_pct"])
        st.caption(f"Model: p = {p_model:.1%} (kurs fair {1 / p_model:.2f}) · value {v * 100:+.1f}% "
                   f"{'✅' if v > VALUE_THR else ''} · sugestia 1/4 Kelly: {money(float(k))}")
        for n in filter(None, (fallback_note(m, home), fallback_note(m, away))):
            st.caption(n)
    over = stake > status["remaining"] + 1e-9
    if over and not status["blocked"]:
        st.error(f"Stawka przekracza pozostały budżet ({money(status['remaining'])}).")
    if st.button("➕ Dodaj typ do dziennika", type="primary",
                 disabled=status["blocked"] or over or home == away):
        try:
            journal.add_bet(conn, {"match_date": mdate, "league": lg, "home": home, "away": away, "market": market,
                                   "selection": sel, "odds": odds, "stake": stake, "p_model": p_model,
                                   "note": note or None}, bcfg["monthly_limit"])
            st.success("Dodano typ.")
            st.rerun()
        except journal.BudgetExceeded as exc:
            st.error(f"🛑 {exc}")
        except ValueError as exc:
            st.error(str(exc))

    st.subheader("Twoje typy")
    if bets.empty:
        st.info("Dziennik jest pusty.")
        return
    s = journal.stats(bets)
    c = st.columns(6)
    c[0].metric("Rozliczone / oczekujące", f"{s['n_settled']} / {s['n_pending']}")
    c[1].metric("Postawiono (rozliczone)", money(s["staked"]))
    c[2].metric("Zysk", money(s["profit"]))
    c[3].metric("ROI", pct(s["roi"], True))
    c[4].metric("Trafność", pct(s["hit_rate"]))
    c[5].metric("Średni CLV", pct(s["mean_clv"], True), help=f"kurs / kurs zamknięcia − 1, n = {s['n_clv']}")
    if s["n_settled"] and s["n_settled"] < 100:
        st.caption(f"Tylko {s['n_settled']} rozliczonych typów – ROI na tak małej próbie to w dużej mierze "
                   f"przypadek. CLV jest szybszym sygnałem, czy typy mają sens.")
    status_pl = {"pending": "⏳ oczekuje", "won": "✅ wygrany", "lost": "❌ przegrany", "void": "↩️ zwrot"}
    show = bets.assign(status=bets["status"].map(status_pl),
                       wynik=[f"{int(h)}:{int(a)}" if pd.notna(h) else "" for h, a in zip(bets["hg"], bets["ag"])])
    st.dataframe(show[["id", "match_date", "league", "home", "away", "market", "selection", "odds", "stake",
                       "p_model", "status", "wynik", "profit", "closing_odds", "clv", "note"]],
                 hide_index=True, column_config={
                     "match_date": "Data", "league": "Liga", "home": "Gospodarz", "away": "Gość",
                     "market": "Rynek", "selection": "Typ", "odds": st.column_config.NumberColumn("Kurs", format="%.2f"),
                     "stake": st.column_config.NumberColumn("Stawka", format="%.2f"),
                     "p_model": st.column_config.NumberColumn("p model", format="percent"),
                     "profit": st.column_config.NumberColumn("Zysk", format="%+.2f"),
                     "closing_odds": st.column_config.NumberColumn("Kurs zamkn.", format="%.2f"),
                     "clv": st.column_config.NumberColumn("CLV", format="percent")})
    settled = bets[bets["status"].isin(["won", "lost"])].sort_values(["match_date", "id"])
    if len(settled) >= 2:
        curve = settled.assign(date=pd.to_datetime(settled["match_date"]), cum=settled["profit"].cumsum())
        st.plotly_chart(charts.line_chart(curve["date"], curve["cum"], "Skumulowany zysk z dziennika",
                                          f"zysk ({CUR})", is_dark(), baseline=0.0), key="j_curve")
    with st.expander("Zarządzaj typem (kurs zamknięcia, zwrot, usunięcie)"):
        bid = st.selectbox("Typ (id)", bets["id"].tolist(),
                           format_func=lambda i: (lambda r: f"#{i} {r.home} – {r.away} {r.market} {r.selection} @ {r.odds}")(
                               bets.set_index("id").loc[i]), key="j_manage")
        c1, c2, c3 = st.columns(3)
        clo = c1.number_input("Kurs zamknięcia", min_value=1.01, value=2.0, step=0.01, key="j_close")
        if c1.button("Zapisz kurs zamknięcia"):
            journal.set_closing_odds(conn, int(bid), clo)
            st.rerun()
        if c2.button("Oznacz jako zwrot (mecz odwołany)"):
            journal.void_bet(conn, int(bid))
            st.rerun()
        if c3.button("🗑️ Usuń typ"):
            journal.delete_bet(conn, int(bid))
            st.rerun()


# --------------------------------------------------------------------------- main


def main() -> None:
    league, xi, token = sidebar()
    st.title(f"{CFG['leagues'][league]['name']}")
    try:
        results = get_results(league, token)
    except Exception as exc:
        st.error(f"Nie udało się wczytać danych ligi ({exc}). Sprawdź połączenie z internetem.")
        render_footer()
        return
    if results.empty:
        st.error("Brak danych dla tej ligi.")
        render_footer()
        return
    st.caption(f"{len(results)} meczów w danych · ostatni: {results['date'].max():%Y-%m-%d} · "
               f"dzisiaj: {datetime.now():%Y-%m-%d}")
    t1, t2, t3, t4 = st.tabs(["📅 Mecze", "📊 Ranking siły", "🧪 Backtest", "📒 Dziennik"])
    with t1:
        tab_matches(league, xi, token)
    with t2:
        tab_ratings(league, xi, token)
    with t3:
        tab_backtest(league, token)
    with t4:
        tab_journal(league, xi, token)
    render_footer()


main()
