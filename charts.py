"""Wykresy Plotly dla UI (paleta zwalidowana pod CVD, osobne odcienie dla trybu ciemnego)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

PALETTE = {
    "light": {"s1": "#2a78d6", "s2": "#eb6834", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "surface": "#fcfcfb",
              "seq": ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]},
    "dark": {"s1": "#3987e5", "s2": "#d95926", "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "surface": "#1a1a19",
             "seq": ["#1f2733", "#104281", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef", "#cde2fb"]},
}


def _c(dark: bool) -> dict:
    return PALETTE["dark" if dark else "light"]


def _layout(fig: go.Figure, dark: bool, title: str, height: int = 380, **kw) -> go.Figure:
    c = _c(dark)
    fig.update_layout(
        title={"text": title, "font": {"size": 15, "color": c["ink"]}, "x": 0, "xanchor": "left"},
        height=height, margin={"l": 10, "r": 10, "t": 48, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "system-ui, -apple-system, 'Segoe UI', sans-serif", "color": c["ink2"], "size": 12},
        hoverlabel={"bgcolor": c["surface"], "font": {"color": c["ink"]}, "bordercolor": c["axis"]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.0, "xanchor": "right", "x": 1,
                "font": {"color": c["ink2"]}},
        **kw,
    )
    axis = {"gridcolor": c["grid"], "linecolor": c["axis"], "zerolinecolor": c["axis"], "gridwidth": 1,
            "tickfont": {"color": c["muted"]}, "title": {"font": {"color": c["ink2"]}}}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def score_heatmap(matrix: np.ndarray, home: str, away: str, dark: bool = False, show: int = 6) -> go.Figure:
    """Heatmapa prawdopodobieństw wyników (gospodarze w wierszach, goście w kolumnach)."""
    c = _c(dark)
    m = matrix[: show + 1, : show + 1] * 100
    scale = [[k / (len(c["seq"]) - 1), col] for k, col in enumerate(c["seq"])]
    text = np.where(m >= 2.0, np.char.mod("%.1f", m), "")
    zmax = max(m.max(), 1e-9)
    # kolor tekstu zależny od jasności komórki
    light_cell = (m / zmax) < 0.45
    fig = go.Figure(go.Heatmap(
        z=m, x=[str(k) for k in range(show + 1)], y=[str(k) for k in range(show + 1)],
        colorscale=scale, zmin=0, zmax=zmax, xgap=2, ygap=2,
        colorbar={"title": {"text": "%", "font": {"color": c["ink2"]}}, "tickfont": {"color": c["muted"]},
                  "thickness": 10},
        hovertemplate=f"{home} %{{y}} : %{{x}} {away}<br>P = %{{z:.2f}}%<extra></extra>",
    ))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if text[i, j]:
                dark_text = light_cell[i, j] != dark
                fig.add_annotation(x=str(j), y=str(i), text=text[i, j], showarrow=False,
                                   font={"size": 11, "color": "#0b0b0b" if dark_text else "#f0efec"})
    _layout(fig, dark, f"Rozkład wyników: {home} – {away}", height=420)
    fig.update_xaxes(title_text=f"Gole: {away}", showgrid=False)
    fig.update_yaxes(title_text=f"Gole: {home}", autorange="reversed", showgrid=False)
    return fig


def ratings_bar(r: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Siła ogólna drużyn (atak + obrona) - jedna seria, jeden kolor."""
    c = _c(dark)
    r = r.sort_values("sila")
    labels = [f"{t} ⚠" if fb > 0 else t for t, fb in zip(r["druzyna"], r["fallback"])]
    fig = go.Figure(go.Bar(
        x=r["sila"], y=labels, orientation="h", marker={"color": c["s1"], "cornerradius": 4},
        customdata=np.stack([r["atak"], r["obrona"], r["mecze"]], axis=1),
        hovertemplate="<b>%{y}</b><br>siła %{x:+.3f}<br>atak %{customdata[0]:+.3f}"
                      "<br>obrona %{customdata[1]:+.3f}<br>mecze %{customdata[2]}<extra></extra>",
    ))
    _layout(fig, dark, "Siła ogólna (atak + obrona, 0 = średnia ligi)", height=max(360, 24 * len(r) + 80),
            bargap=0.25)
    fig.update_xaxes(zeroline=True, zerolinewidth=1)
    return fig


def ratings_scatter(r: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Atak vs obrona; prawy górny róg = najlepsze drużyny."""
    c = _c(dark)
    # podpisy tylko dla skrajnych drużyn i beniaminków, reszta w tooltipie
    order = r.sort_values("sila")["druzyna"]
    labelled = set(order.head(4)) | set(order.tail(4)) | set(r.loc[r["fallback"] > 0, "druzyna"])
    fig = go.Figure(go.Scatter(
        x=r["atak"], y=r["obrona"], mode="markers+text", customdata=r["druzyna"],
        text=[t if t in labelled else "" for t in r["druzyna"]], textposition="top center",
        textfont={"size": 11, "color": c["ink2"]},
        marker={"size": 10, "color": c["s1"], "line": {"width": 2, "color": c["surface"]}},
        hovertemplate="<b>%{customdata}</b><br>atak %{x:+.3f}<br>obrona %{y:+.3f}<extra></extra>",
    ))
    _layout(fig, dark, "Atak vs obrona (więcej = lepiej)", height=520)
    fig.update_xaxes(title_text="atak", zeroline=True)
    fig.update_yaxes(title_text="obrona", zeroline=True)
    return fig


def calibration_chart(calib: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Kalibracja: przewidywane prawdopodobieństwo vs częstość, model i rynek."""
    c = _c(dark)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": c["muted"], "width": 1},
                             name="idealna kalibracja", hoverinfo="skip"))
    for src, col in (("Model DC", c["s1"]), ("Rynek", c["s2"])):
        d = calib[(calib["zrodlo"] == src) & (calib["n"] >= 10)]
        fig.add_trace(go.Scatter(
            x=d["p_srednie"], y=d["czestosc"], mode="lines+markers", name=src,
            line={"color": col, "width": 2}, marker={"size": 9, "color": col, "line": {"width": 2, "color": c["surface"]}},
            customdata=d["n"],
            hovertemplate=f"{src}<br>przewidywane %{{x:.1%}}<br>faktycznie %{{y:.1%}}<br>n = %{{customdata}}<extra></extra>",
        ))
    _layout(fig, dark, "Kalibracja (przedziały co 10 pp, min. 10 obserwacji)", height=420)
    fig.update_xaxes(title_text="przewidywane prawdopodobieństwo", range=[0, 1], tickformat=".0%")
    fig.update_yaxes(title_text="faktyczna częstość", range=[0, 1], tickformat=".0%")
    return fig


def line_chart(x: pd.Series, y: pd.Series, title: str, y_title: str, dark: bool = False,
               baseline: float | None = None, y_fmt: str = ",.1f", hover_name: str = "") -> go.Figure:
    """Pojedyncza seria w czasie (bez drugiej osi Y)."""
    c = _c(dark)
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines", line={"color": c["s1"], "width": 2}, name=hover_name,
                               hovertemplate=f"%{{x|%Y-%m-%d}}<br>{y_title}: %{{y:{y_fmt}}}<extra></extra>"))
    if baseline is not None:
        fig.add_hline(y=baseline, line={"color": c["axis"], "width": 1})
    if len(y):
        fig.add_annotation(x=x.iloc[-1], y=y.iloc[-1], text=f"{y.iloc[-1]:{y_fmt}}", showarrow=False,
                           xanchor="left", xshift=6, font={"color": c["ink"], "size": 12})
    _layout(fig, dark, title, height=340, hovermode="x unified", showlegend=False)
    fig.update_yaxes(title_text=y_title)
    return fig


def xi_chart(table: pd.DataFrame, best: float, dark: bool = False) -> go.Figure:
    """Log-loss na walidacji w funkcji xi."""
    c = _c(dark)
    fig = go.Figure(go.Scatter(x=table["xi"], y=table["log_loss"], mode="lines+markers",
                               line={"color": c["s1"], "width": 2}, marker={"size": 8, "color": c["s1"]},
                               hovertemplate="xi = %{x}<br>log-loss %{y:.5f}<extra></extra>"))
    b = table[table["xi"] == best]
    fig.add_trace(go.Scatter(x=b["xi"], y=b["log_loss"], mode="markers+text", text=[f"wybrane xi = {best}"],
                             textposition="top center", textfont={"color": c["ink"]},
                             marker={"size": 12, "color": c["s2"], "line": {"width": 2, "color": c["surface"]}},
                             hoverinfo="skip"))
    _layout(fig, dark, "Strojenie xi na sezonach walidacyjnych", height=320, showlegend=False)
    fig.update_xaxes(title_text="xi (na dzień)")
    fig.update_yaxes(title_text="log-loss 1X2")
    return fig
