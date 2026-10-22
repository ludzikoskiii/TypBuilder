"""Dziennik typów w SQLite: dodawanie, limit budżetu, automatyczne rozliczanie, ROI i CLV."""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bets (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    match_date   TEXT NOT NULL,
    league       TEXT NOT NULL,
    home         TEXT NOT NULL,
    away         TEXT NOT NULL,
    market       TEXT NOT NULL,
    selection    TEXT NOT NULL,
    odds         REAL NOT NULL CHECK (odds > 1),
    stake        REAL NOT NULL CHECK (stake > 0),
    p_model      REAL,
    closing_odds REAL,
    status       TEXT NOT NULL DEFAULT 'pending',
    hg           INTEGER,
    ag           INTEGER,
    profit       REAL,
    settled_at   TEXT,
    note         TEXT
);
"""

# rynek -> dozwolone typy
MARKETS: dict[str, tuple[str, ...]] = {
    "1X2": ("1", "X", "2"),
    "O/U 1.5": ("Over", "Under"),
    "O/U 2.5": ("Over", "Under"),
    "O/U 3.5": ("Over", "Under"),
    "BTTS": ("Tak", "Nie"),
}

# (rynek, typ) -> klucz prawdopodobieństwa w model.markets_from_matrix
MODEL_KEYS: dict[tuple[str, str], str] = {
    ("1X2", "1"): "p_home", ("1X2", "X"): "p_draw", ("1X2", "2"): "p_away",
    ("O/U 1.5", "Over"): "over15", ("O/U 1.5", "Under"): "under15",
    ("O/U 2.5", "Over"): "over25", ("O/U 2.5", "Under"): "under25",
    ("O/U 3.5", "Over"): "over35", ("O/U 3.5", "Under"): "under35",
    ("BTTS", "Tak"): "btts_yes", ("BTTS", "Nie"): "btts_no",
}

# (rynek, typ) -> kolumna kursu zamknięcia w danych (do automatycznego CLV)
CLOSING_COLUMNS: dict[tuple[str, str], str] = {
    ("1X2", "1"): "closing_odds_h", ("1X2", "X"): "closing_odds_d", ("1X2", "2"): "closing_odds_a",
    ("O/U 2.5", "Over"): "closing_odds_over25", ("O/U 2.5", "Under"): "closing_odds_under25",
}


class BudgetExceeded(Exception):
    """Dodanie typu przekroczyłoby miesięczny budżet."""


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Otwórz (i w razie potrzeby utwórz) bazę dziennika."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _month_bounds(day: date) -> tuple[str, str]:
    start = day.replace(day=1)
    end = start.replace(year=start.year + 1, month=1) if start.month == 12 else start.replace(month=start.month + 1)
    return start.isoformat(), end.isoformat()


def month_staked(conn: sqlite3.Connection, day: date | None = None) -> float:
    """Suma stawek typów dodanych w miesiącu kalendarzowym `day` (bez typów anulowanych)."""
    lo, hi = _month_bounds(day or date.today())
    row = conn.execute("SELECT COALESCE(SUM(stake), 0) FROM bets WHERE created_at >= ? AND created_at < ? "
                       "AND status != 'void'", (lo, hi)).fetchone()
    return float(row[0])


def budget_status(conn: sqlite3.Connection, limit: float, warn_at: float = 0.8,
                  day: date | None = None) -> dict[str, Any]:
    """Stan miesięcznego budżetu: wydane, pozostało, czy ostrzegać / blokować."""
    spent = month_staked(conn, day)
    pct = spent / limit if limit > 0 else 1.0
    return {"spent": spent, "limit": limit, "remaining": max(limit - spent, 0.0), "pct": pct,
            "warn": pct >= warn_at, "blocked": spent >= limit}


def settle_outcome(market: str, selection: str, hg: int, ag: int) -> str:
    """'won' / 'lost' dla typu przy wyniku hg:ag."""
    if market == "1X2":
        actual = "1" if hg > ag else ("X" if hg == ag else "2")
        return "won" if selection == actual else "lost"
    if market.startswith("O/U"):
        line = float(market.split()[1])
        over = hg + ag > line
        return "won" if (selection == "Over") == over else "lost"
    if market == "BTTS":
        both = hg > 0 and ag > 0
        return "won" if (selection == "Tak") == both else "lost"
    raise ValueError(f"Nieznany rynek: {market}")


def add_bet(conn: sqlite3.Connection, bet: dict[str, Any], monthly_limit: float,
            now: datetime | None = None) -> int:
    """Dodaj typ. Rzuca BudgetExceeded, gdy stawka przekroczyłaby miesięczny limit."""
    now = now or datetime.now()
    if bet["market"] not in MARKETS or bet["selection"] not in MARKETS[bet["market"]]:
        raise ValueError(f"Niepoprawny rynek/typ: {bet['market']} {bet['selection']}")
    if not bet.get("odds") or bet["odds"] <= 1:
        raise ValueError("Kurs musi być większy od 1")
    if not bet.get("stake") or bet["stake"] <= 0:
        raise ValueError("Stawka musi być dodatnia")
    spent = month_staked(conn, now.date())
    if spent + bet["stake"] > monthly_limit + 1e-9:
        raise BudgetExceeded(f"Limit miesięczny {monthly_limit:.2f} - wydano już {spent:.2f}, "
                             f"stawka {bet['stake']:.2f} go przekracza.")
    cur = conn.execute(
        "INSERT INTO bets (created_at, match_date, league, home, away, market, selection, odds, stake, "
        "p_model, closing_odds, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (now.isoformat(timespec="seconds"), pd.Timestamp(bet["match_date"]).date().isoformat(),
         bet["league"], bet["home"], bet["away"], bet["market"], bet["selection"], float(bet["odds"]),
         float(bet["stake"]), bet.get("p_model"), bet.get("closing_odds"), bet.get("note")))
    conn.commit()
    return int(cur.lastrowid)


def list_bets(conn: sqlite3.Connection) -> pd.DataFrame:
    """Wszystkie typy (najnowsze na górze) z kolumną CLV = kurs / kurs zamknięcia - 1."""
    df = pd.read_sql_query("SELECT * FROM bets ORDER BY match_date DESC, id DESC", conn)
    df["clv"] = df["odds"] / df["closing_odds"] - 1
    return df


def delete_bet(conn: sqlite3.Connection, bet_id: int) -> None:
    conn.execute("DELETE FROM bets WHERE id = ?", (bet_id,))
    conn.commit()


def set_closing_odds(conn: sqlite3.Connection, bet_id: int, odds: float) -> None:
    conn.execute("UPDATE bets SET closing_odds = ? WHERE id = ?", (float(odds), bet_id))
    conn.commit()


def void_bet(conn: sqlite3.Connection, bet_id: int) -> None:
    """Anuluj typ (np. mecz odwołany) - zwrot stawki, nie liczy się do ROI ani budżetu."""
    conn.execute("UPDATE bets SET status = 'void', profit = 0, settled_at = ? WHERE id = ?",
                 (datetime.now().isoformat(timespec="seconds"), bet_id))
    conn.commit()


def find_result(results: pd.DataFrame, home: str, away: str, match_date: str,
                tolerance_days: int = 3) -> pd.Series | None:
    """Znajdź rozegrany mecz (te same drużyny, data +/- tolerance_days)."""
    if results is None or results.empty:
        return None
    d = pd.Timestamp(match_date)
    cand = results[(results["home"] == home) & (results["away"] == away)
                   & ((results["date"] - d).abs() <= pd.Timedelta(days=tolerance_days))]
    return cand.iloc[0] if len(cand) else None


def settle_pending(conn: sqlite3.Connection, results_by_league: dict[str, pd.DataFrame]) -> int:
    """Rozlicz oczekujące typy na podstawie wyników; uzupełnij brakujący kurs zamknięcia z danych."""
    pending = conn.execute("SELECT * FROM bets WHERE status = 'pending'").fetchall()
    settled = 0
    for b in pending:
        res = find_result(results_by_league.get(b["league"]), b["home"], b["away"], b["match_date"])
        if res is None:
            continue
        hg, ag = int(res["hg"]), int(res["ag"])
        status = settle_outcome(b["market"], b["selection"], hg, ag)
        profit = b["stake"] * (b["odds"] - 1) if status == "won" else -b["stake"]
        closing = b["closing_odds"]
        col = CLOSING_COLUMNS.get((b["market"], b["selection"]))
        if closing is None and col and col in res.index and pd.notna(res[col]):
            closing = float(res[col])
        conn.execute("UPDATE bets SET status=?, hg=?, ag=?, profit=?, closing_odds=?, settled_at=? WHERE id=?",
                     (status, hg, ag, profit, closing, datetime.now().isoformat(timespec="seconds"), b["id"]))
        settled += 1
    conn.commit()
    return settled


def stats(df: pd.DataFrame) -> dict[str, Any]:
    """Statystyki dziennika: liczba typów, zysk, ROI, trafność, średni CLV."""
    done = df[df["status"].isin(["won", "lost"])]
    staked = float(done["stake"].sum())
    profit = float(done["profit"].sum())
    clv = df["clv"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "n_total": len(df),
        "n_settled": len(done),
        "n_pending": int((df["status"] == "pending").sum()),
        "pending_stake": float(df.loc[df["status"] == "pending", "stake"].sum()),
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked > 0 else float("nan"),
        "hit_rate": float((done["status"] == "won").mean()) if len(done) else float("nan"),
        "avg_odds": float(done["odds"].mean()) if len(done) else float("nan"),
        "mean_clv": float(clv.mean()) if len(clv) else float("nan"),
        "n_clv": len(clv),
        "beat_close_pct": float((clv > 0).mean()) if len(clv) else float("nan"),
    }
