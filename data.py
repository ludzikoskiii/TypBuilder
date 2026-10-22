"""Pobieranie, cache i ujednolicanie danych z football-data.co.uk.

Ujednolicony format wyników (jeden wiersz = jeden rozegrany mecz):
date, time, league, season, home, away, hg, ag,
odds_h/d/a (kursy przedmeczowe), closing_odds_h/d/a (kursy zamknięcia),
odds_over25/under25, closing_odds_over25/under25,
odds_source, closing_source, odds_are_closing.

Uwaga o formatach (stan na sezon 2026/27):
- Ligi główne (mmz4281/<sezon>/<kod>.csv) mają kursy przedmeczowe (np. AvgH)
  i zamknięcia (np. AvgCH). Kolumny Pinnacle (PSH, PSCH) zniknęły od 2026/27.
- Pliki "extra leagues" (new/<kod>.csv, np. POL.csv) zawierają WYŁĄCZNIE kursy
  zamknięcia (PSCH, MaxCH, AvgCH, BFECH, B365CH). Dla nich odds_* = closing_odds_*
  i odds_are_closing = True, więc CLV nie da się policzyć.
"""
from __future__ import annotations

import io
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests

from config import load_config

log = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "date", "time", "league", "season", "home", "away", "hg", "ag",
    "odds_h", "odds_d", "odds_a",
    "closing_odds_h", "closing_odds_d", "closing_odds_a",
    "odds_over25", "odds_under25", "closing_odds_over25", "closing_odds_under25",
    "odds_source", "closing_source", "odds_are_closing",
]
FIXTURE_COLUMNS = [
    "date", "time", "league", "home", "away",
    "odds_h", "odds_d", "odds_a", "odds_over25", "odds_under25", "odds_source",
]

# Warianty nazw spotykane w plikach football-data (wariant -> nazwa kanoniczna).
DEFAULT_TEAM_ALIASES: dict[str, str] = {
    "Gornik Z.": "Gornik Zabrze",
    "Ruch": "Ruch Chorzow",
    "Wisla Krakow": "Wisla",
    "Nottingham Forest": "Nott'm Forest",
    "Nottingham": "Nott'm Forest",
    "Manchester United": "Man United",
    "Manchester Utd": "Man United",
    "Manchester City": "Man City",
    "Sheffield Utd": "Sheffield United",
    "Wolverhampton": "Wolves",
}

USER_AGENT = "TyperDC/1.0 (+lokalne narzedzie analityczne)"


# --------------------------------------------------------------------------- sezony


def current_season_start(today: date | None = None) -> int:
    """Rok startu bieżącego sezonu (sezon europejski zaczyna się w lipcu)."""
    today = today or date.today()
    return today.year if today.month >= 7 else today.year - 1


def season_code(start_year: int) -> str:
    """2025 -> '2526' (kod katalogu mmz4281)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2025 -> '2025/2026' (format kolumny Season w plikach extra)."""
    return f"{start_year}/{start_year + 1}"


def recent_season_starts(n: int, today: date | None = None) -> list[int]:
    """Lata startu n ostatnich sezonów (rosnąco, łącznie z bieżącym)."""
    cur = current_season_start(today)
    return list(range(cur - n + 1, cur + 1))


# --------------------------------------------------------------------------- pobieranie


@dataclass
class FetchResult:
    path: Path
    downloaded: bool
    stale: bool = False


def fetch(url: str, path: Path, max_age_hours: float, immutable: bool = False,
          timeout: float = 30.0) -> FetchResult:
    """Pobierz plik do cache, chyba że kopia jest świeża.

    immutable=True oznacza plik, który się nie zmienia (zakończony sezon) - wtedy
    istniejąca kopia jest zawsze używana. Gdy pobranie się nie uda, a w cache jest
    stara kopia, zostanie ona użyta (stale=True).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        age_h = (time.time() - path.stat().st_mtime) / 3600
        if immutable or age_h < max_age_hours:
            return FetchResult(path, downloaded=False)
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        body = resp.content
        head = body[:200].decode("utf-8-sig", errors="ignore")
        if "," not in head or "<html" in head.lower():
            raise ValueError(f"Odpowiedź nie wygląda na CSV: {url}")
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(body)
        tmp.replace(path)
        return FetchResult(path, downloaded=True)
    except (requests.RequestException, ValueError) as exc:
        if path.exists() and path.stat().st_size > 0:
            log.warning("Nie udało się pobrać %s (%s) - używam kopii z cache.", url, exc)
            return FetchResult(path, downloaded=False, stale=True)
        raise


def read_csv_any(path: Path) -> pd.DataFrame:
    """Wczytaj CSV football-data (BOM UTF-8 w nowych plikach, latin-1 w starych)."""
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    df = pd.read_csv(io.StringIO(text), on_bad_lines="skip", low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.dropna(how="all")


# --------------------------------------------------------------------------- nazwy drużyn


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def build_alias_map(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Mapa aliasów z kluczami znormalizowanymi (małe litery, bez ogonków)."""
    aliases = {**DEFAULT_TEAM_ALIASES, **(extra or {})}
    return {_strip_accents(k).strip().lower(): v for k, v in aliases.items()}


def normalize_team(name: Any, alias_map: dict[str, str]) -> str:
    """Ujednolić nazwę drużyny: usuń białe znaki/ogonki i zastosuj aliasy."""
    if not isinstance(name, str):
        return name
    clean = " ".join(_strip_accents(name).split())
    return alias_map.get(clean.lower(), clean)


# --------------------------------------------------------------------------- parsowanie


def parse_dates(s: pd.Series) -> pd.Series:
    """Daty dd/mm/yyyy (nowe pliki) lub dd/mm/yy (stare pliki)."""
    s = s.astype(str).str.strip()
    out = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = out.isna()
    if short.any():
        out[short] = pd.to_datetime(s[short], format="%d/%m/%y", errors="coerce")
    return out


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def pick_odds(df: pd.DataFrame, prefixes: Iterable[str],
              suffixes: tuple[str, ...] = ("H", "D", "A")) -> tuple[pd.DataFrame, pd.Series]:
    """Wybierz kursy z pierwszego źródła (wg kolejności), które ma komplet kursów > 1.

    Zwraca (DataFrame z kolumnami = suffixes, Series z nazwą użytego źródła).
    """
    out = pd.DataFrame(np.nan, index=df.index, columns=list(suffixes))
    src = pd.Series(pd.NA, index=df.index, dtype="object")
    for prefix in prefixes:
        cols = [f"{prefix}{s}" for s in suffixes]
        if not all(c in df.columns for c in cols):
            continue
        block = pd.concat([_num(df, c) for c in cols], axis=1)
        block.columns = list(suffixes)
        ok = block.notna().all(axis=1) & (block > 1.0).all(axis=1) & src.isna()
        out.loc[ok] = block.loc[ok].values
        src.loc[ok] = prefix
    return out, src


def _ou_prefix(prefix: str) -> str:
    # W kolumnach over/under Pinnacle ma prefiks "P" zamiast "PS".
    return {"PS": "P", "PSC": "PC"}.get(prefix, prefix)


def pick_ou25(df: pd.DataFrame, prefixes: Iterable[str]) -> pd.DataFrame:
    """Kursy over/under 2.5 z pierwszego dostępnego źródła."""
    out, _ = pick_odds(df, [_ou_prefix(p) for p in prefixes], suffixes=(">2.5", "<2.5"))
    return out.rename(columns={">2.5": "over", "<2.5": "under"})


def parse_main(raw: pd.DataFrame, league: str, season: str, cfg: dict[str, Any],
               alias_map: dict[str, str]) -> pd.DataFrame:
    """Ujednolić plik ligi głównej (format mmz4281)."""
    dcfg = cfg["data"]
    df = pd.DataFrame(index=raw.index)
    df["date"] = parse_dates(raw["Date"])
    df["time"] = raw["Time"].astype(str) if "Time" in raw.columns else ""
    df["league"] = league
    df["season"] = season
    df["home"] = raw["HomeTeam"].map(lambda n: normalize_team(n, alias_map))
    df["away"] = raw["AwayTeam"].map(lambda n: normalize_team(n, alias_map))
    df["hg"] = _num(raw, "FTHG")
    df["ag"] = _num(raw, "FTAG")
    pre, pre_src = pick_odds(raw, dcfg["odds_priority"])
    clo, clo_src = pick_odds(raw, dcfg["closing_priority"])
    df[["odds_h", "odds_d", "odds_a"]] = pre.values
    df[["closing_odds_h", "closing_odds_d", "closing_odds_a"]] = clo.values
    ou = pick_ou25(raw, dcfg["odds_priority"])
    ouc = pick_ou25(raw, dcfg["closing_priority"])
    df["odds_over25"], df["odds_under25"] = ou["over"], ou["under"]
    df["closing_odds_over25"], df["closing_odds_under25"] = ouc["over"], ouc["under"]
    df["odds_source"] = pre_src
    df["closing_source"] = clo_src
    df["odds_are_closing"] = False
    return df


def parse_extra(raw: pd.DataFrame, league: str, cfg: dict[str, Any],
                alias_map: dict[str, str], lcfg: dict[str, Any]) -> pd.DataFrame:
    """Ujednolić zbiorczy plik "extra league" (np. POL.csv) - tylko kursy zamknięcia."""
    dcfg = cfg["data"]
    if "League" in raw.columns and lcfg.get("league"):
        raw = raw[raw["League"].astype(str).str.strip() == lcfg["league"]]
    df = pd.DataFrame(index=raw.index)
    df["date"] = parse_dates(raw["Date"])
    df["time"] = raw["Time"].astype(str) if "Time" in raw.columns else ""
    df["league"] = league
    df["season"] = raw["Season"].astype(str).str.strip()
    df["home"] = raw["Home"].map(lambda n: normalize_team(n, alias_map))
    df["away"] = raw["Away"].map(lambda n: normalize_team(n, alias_map))
    df["hg"] = _num(raw, "HG")
    df["ag"] = _num(raw, "AG")
    clo, clo_src = pick_odds(raw, dcfg["closing_priority"])
    # Starsze wersje plików extra miały kursy zamknięcia bez litery "C" (np. PH, AvgH).
    legacy, legacy_src = pick_odds(raw, [p.replace("PS", "P") for p in dcfg["odds_priority"]])
    missing = clo_src.isna()
    clo.loc[missing] = legacy.loc[missing].values
    clo_src.loc[missing] = legacy_src.loc[missing]
    df[["closing_odds_h", "closing_odds_d", "closing_odds_a"]] = clo.values
    df[["odds_h", "odds_d", "odds_a"]] = clo.values
    for col in ("odds_over25", "odds_under25", "closing_odds_over25", "closing_odds_under25"):
        df[col] = np.nan
    df["odds_source"] = clo_src
    df["closing_source"] = clo_src
    df["odds_are_closing"] = True
    return df


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["date", "home", "away", "hg", "ag"]).copy()
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df["date"] = df["date"].dt.normalize()
    df = df.drop_duplicates(subset=["date", "home", "away"], keep="last")
    return df[RESULT_COLUMNS].sort_values(["date", "time", "home"]).reset_index(drop=True)


# --------------------------------------------------------------------------- API


def league_config(league: str, cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        return cfg["leagues"][league]
    except KeyError as exc:
        raise KeyError(f"Nieznana liga '{league}' - dodaj ją do config.yaml") from exc


def load_league(league: str, cfg: dict[str, Any] | None = None, seasons_back: int | None = None,
                today: date | None = None, offline: bool = False) -> pd.DataFrame:
    """Wczytaj (pobierając w razie potrzeby) ujednolicone wyniki ligi.

    offline=True: używaj wyłącznie plików z cache (bez sieci).
    """
    cfg = cfg or load_config()
    dcfg = cfg["data"]
    lcfg = league_config(league, cfg)
    n = seasons_back or dcfg["seasons_back"]
    cache = Path(dcfg["cache_dir"])
    alias_map = build_alias_map(dcfg.get("team_aliases"))
    max_age = float("inf") if offline else dcfg["max_age_hours"]
    starts = recent_season_starts(n, today)
    frames: list[pd.DataFrame] = []

    if lcfg["type"] == "main":
        cur = current_season_start(today)
        for start in starts:
            code = season_code(start)
            url = f"{dcfg['base_url']}/mmz4281/{code}/{lcfg['code']}.csv"
            path = cache / f"{lcfg['code']}_{code}.csv"
            try:
                res = fetch(url, path, max_age, immutable=start < cur)
            except (requests.RequestException, ValueError) as exc:
                if start == cur:
                    log.warning("Brak pliku bieżącego sezonu %s: %s", code, exc)
                    continue
                raise
            raw = read_csv_any(res.path)
            if raw.empty or "HomeTeam" not in raw.columns:
                continue
            frames.append(parse_main(raw, league, season_label(start), cfg, alias_map))
    elif lcfg["type"] == "extra":
        url = f"{dcfg['base_url']}/new/{lcfg['code']}.csv"
        res = fetch(url, cache / f"extra_{lcfg['code']}.csv", max_age)
        df = parse_extra(read_csv_any(res.path), league, cfg, alias_map, lcfg)
        wanted = {season_label(s) for s in starts} | {str(y) for s in starts for y in (s, s + 1)}
        frames.append(df[df["season"].isin(wanted)])
    else:
        raise ValueError(f"Nieznany typ ligi: {lcfg['type']}")

    if not frames:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return _finalize(pd.concat(frames, ignore_index=True))


def load_fixtures(league: str, cfg: dict[str, Any] | None = None, results: pd.DataFrame | None = None,
                  today: date | None = None, offline: bool = False) -> pd.DataFrame:
    """Nadchodzące mecze ligi z fixtures.csv / new_league_fixtures.csv (z kursami, jeśli są)."""
    cfg = cfg or load_config()
    dcfg = cfg["data"]
    lcfg = league_config(league, cfg)
    alias_map = build_alias_map(dcfg.get("team_aliases"))
    cache = Path(dcfg["cache_dir"])
    max_age = float("inf") if offline else dcfg["max_age_hours"]
    today_ts = pd.Timestamp(today or date.today())

    if lcfg["type"] == "main":
        res = fetch(f"{dcfg['base_url']}/fixtures.csv", cache / "fixtures.csv", max_age)
        raw = read_csv_any(res.path)
        raw = raw[raw.get("Div", pd.Series(dtype=str)).astype(str).str.strip() == lcfg["code"]]
        home_col, away_col = "HomeTeam", "AwayTeam"
    else:
        res = fetch(f"{dcfg['base_url']}/new_league_fixtures.csv",
                    cache / "new_league_fixtures.csv", max_age)
        raw = read_csv_any(res.path)
        mask = pd.Series(True, index=raw.index)
        if "Country" in raw.columns and lcfg.get("country"):
            mask &= raw["Country"].astype(str).str.strip() == lcfg["country"]
        if "League" in raw.columns and lcfg.get("league"):
            mask &= raw["League"].astype(str).str.strip() == lcfg["league"]
        raw = raw[mask]
        home_col, away_col = "Home", "Away"

    if raw.empty:
        return pd.DataFrame(columns=FIXTURE_COLUMNS)

    df = pd.DataFrame(index=raw.index)
    df["date"] = parse_dates(raw["Date"]).dt.normalize()
    df["time"] = raw["Time"].astype(str) if "Time" in raw.columns else ""
    df["league"] = league
    df["home"] = raw[home_col].map(lambda n: normalize_team(n, alias_map))
    df["away"] = raw[away_col].map(lambda n: normalize_team(n, alias_map))
    pre, src = pick_odds(raw, dcfg["odds_priority"])
    df[["odds_h", "odds_d", "odds_a"]] = pre.values
    ou = pick_ou25(raw, dcfg["odds_priority"])
    df["odds_over25"], df["odds_under25"] = ou["over"], ou["under"]
    df["odds_source"] = src
    df = df[df["date"] >= today_ts]
    if results is not None and not results.empty:
        played = set(zip(results["date"], results["home"], results["away"]))
        df = df[[(d, h, a) not in played for d, h, a in zip(df["date"], df["home"], df["away"])]]
    return df[FIXTURE_COLUMNS].sort_values(["date", "time", "home"]).reset_index(drop=True)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Podsumowanie danych per sezon: liczba meczów, drużyn, gole, pokrycie kursów."""
    if df.empty:
        return pd.DataFrame()
    g = df.groupby("season", sort=True)
    out = pd.DataFrame({
        "mecze": g.size(),
        "druzyny": g.apply(lambda x: len(set(x["home"]) | set(x["away"])), include_groups=False),
        "od": g["date"].min().dt.strftime("%Y-%m-%d"),
        "do": g["date"].max().dt.strftime("%Y-%m-%d"),
        "gole/mecz": g.apply(lambda x: (x["hg"] + x["ag"]).mean(), include_groups=False).round(2),
        "% 1": g.apply(lambda x: (x["hg"] > x["ag"]).mean() * 100, include_groups=False).round(1),
        "% X": g.apply(lambda x: (x["hg"] == x["ag"]).mean() * 100, include_groups=False).round(1),
        "% 2": g.apply(lambda x: (x["hg"] < x["ag"]).mean() * 100, include_groups=False).round(1),
        "kursy %": g["odds_h"].apply(lambda s: s.notna().mean() * 100).round(1),
        "zamkn. %": g["closing_odds_h"].apply(lambda s: s.notna().mean() * 100).round(1),
        "zrodlo": g["odds_source"].agg(lambda s: ",".join(sorted(s.dropna().unique()))),
        "tylko zamkn.": g["odds_are_closing"].any(),
    })
    return out


def data_freshness(league: str, cfg: dict[str, Any] | None = None) -> datetime | None:
    """Czas modyfikacji najnowszego pliku ligi w cache (do wyświetlenia w UI)."""
    cfg = cfg or load_config()
    lcfg = league_config(league, cfg)
    cache = Path(cfg["data"]["cache_dir"])
    pattern = f"{lcfg['code']}_*.csv" if lcfg["type"] == "main" else f"extra_{lcfg['code']}.csv"
    files = list(cache.glob(pattern))
    if not files:
        return None
    return datetime.fromtimestamp(max(f.stat().st_mtime for f in files))
