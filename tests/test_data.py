"""Testy parsowania i cache danych (bez sieci)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import data
from config import load_config

MAIN_CSV = """﻿Div,Date,Time,HomeTeam,AwayTeam,FTHG,FTAG,FTR,PSH,PSD,PSA,AvgH,AvgD,AvgA,AvgCH,AvgCD,AvgCA,Avg>2.5,Avg<2.5,AvgC>2.5,AvgC<2.5
E0,21/08/2026,20:00,Arsenal,Coventry,3,0,H,1.2,7,14,1.19,6.77,14.19,1.2,6.72,14,1.55,2.38,1.55,2.37
E0,22/08/2026,12:30,Hull,Manchester United,2,0,H,,,,8.37,4.93,1.36,8.49,5.11,1.34,1.64,2.2,1.6,2.3
E0,23/08/2026,14:00,Leeds,Wolves,1,1,D,2.1,3.4,3.6,,,,,,,,,,
"""

EXTRA_CSV = """﻿Country,League,Season,Date,Time,Home,Away,HG,AG,Res,PSCH,PSCD,PSCA,AvgCH,AvgCD,AvgCA,B365CH,B365CD,B365CA
Poland,Ekstraklasa,2012/2013,17/08/2012,19:30,Piast Gliwice,Gornik Z.,1,2,A,2.72,3.12,2.98,2.56,3.05,2.78,,,
Poland,Ekstraklasa,2026/2027,18/10/2026,11:15,Gornik Zabrze,Piast Gliwice,1,1,D,,,,1.54,3.98,5.11,1.55,3.9,5.25
Poland,Ekstraklasa,2026/2027,19/10/2026,18:00,Wisla Plock,Radomiak Radom,,,,,,,1.8,3.62,3.8,,,
"""


@pytest.fixture
def cfg() -> dict:
    return load_config()


def _write(tmp_path: Path, name: str, text: str) -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_main_unifies_columns_and_odds_priority(tmp_path, cfg):
    cfg["data"]["odds_priority"] = ["PS", "Avg"]
    raw = data.read_csv_any(_write(tmp_path, "e0.csv", MAIN_CSV))
    df = data._finalize(data.parse_main(raw, "E0", "2026/2027", cfg, data.build_alias_map()))
    assert list(df.columns) == data.RESULT_COLUMNS
    assert df.loc[0, "odds_source"] == "PS" and df.loc[0, "odds_h"] == 1.2
    # Brak Pinnacle -> spadek na średnią
    row = df[df["home"] == "Hull"].iloc[0]
    assert row["odds_source"] == "Avg" and row["odds_a"] == 1.36
    assert row["away"] == "Man United"  # alias
    assert row["closing_odds_a"] == 1.34 and row["odds_over25"] == 1.64
    # Brak kursów zamknięcia/OU -> NaN, a nie błąd
    leeds = df[df["home"] == "Leeds"].iloc[0]
    assert np.isnan(leeds["closing_odds_h"]) and np.isnan(leeds["odds_over25"])
    assert df["date"].iloc[0] == pd.Timestamp("2026-08-21")


def test_parse_extra_uses_closing_odds_and_aliases(tmp_path, cfg):
    raw = data.read_csv_any(_write(tmp_path, "pol.csv", EXTRA_CSV))
    lcfg = cfg["leagues"]["POL"]
    df = data._finalize(data.parse_extra(raw, "POL", cfg, data.build_alias_map(), lcfg))
    assert len(df) == 2  # mecz bez wyniku odrzucony
    assert set(df["home"]) | set(df["away"]) == {"Piast Gliwice", "Gornik Zabrze"}
    assert df["odds_are_closing"].all()
    assert (df["odds_h"] == df["closing_odds_h"]).all()
    assert df.iloc[0]["season"] == "2012/2013"


def test_missing_odds_columns_give_nan(tmp_path, cfg):
    text = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE0,01/09/19,A,B,1,0\n"
    raw = data.read_csv_any(_write(tmp_path, "old.csv", text))
    df = data._finalize(data.parse_main(raw, "E0", "2019/2020", cfg, data.build_alias_map()))
    assert len(df) == 1 and df["odds_h"].isna().all()
    assert df["date"].iloc[0] == pd.Timestamp("2019-09-01")  # format dd/mm/yy


def test_normalize_team():
    amap = data.build_alias_map({"Legia Warszawa": "Legia"})
    assert data.normalize_team("  Legia   Warszawa ", amap) == "Legia"
    assert data.normalize_team("Górnik Z.", amap) == "Gornik Zabrze"
    assert data.normalize_team("Arsenal", amap) == "Arsenal"


def test_fetch_uses_fresh_cache(tmp_path, monkeypatch):
    path = _write(tmp_path, "f.csv", "a,b\n1,2\n")
    calls = []
    monkeypatch.setattr(data.requests, "get", lambda *a, **k: calls.append(a))
    res = data.fetch("http://x/f.csv", path, max_age_hours=1)
    assert not res.downloaded and calls == []


def test_fetch_falls_back_to_stale_cache(tmp_path, monkeypatch):
    path = _write(tmp_path, "f.csv", "a,b\n1,2\n")
    old = time.time() - 48 * 3600
    os.utime(path, (old, old))

    def boom(*a, **k):
        raise data.requests.ConnectionError("offline")

    monkeypatch.setattr(data.requests, "get", boom)
    res = data.fetch("http://x/f.csv", path, max_age_hours=1)
    assert res.stale and path.read_text().startswith("a,b")


def test_season_helpers():
    from datetime import date
    assert data.current_season_start(date(2026, 10, 22)) == 2026
    assert data.current_season_start(date(2027, 3, 1)) == 2026
    assert data.season_code(2026) == "2627"
    assert data.recent_season_starts(3, date(2026, 10, 22)) == [2024, 2025, 2026]
