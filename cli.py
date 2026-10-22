"""Narzędzia wiersza poleceń: pobieranie danych, podsumowanie, backtest.

Przykłady:
    python cli.py summary E0 POL
    python cli.py backtest E0 --seasons 3
"""
from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from config import load_config
from data import load_fixtures, load_league, summarize


def cmd_summary(args: argparse.Namespace) -> None:
    cfg = load_config()
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    for league in args.leagues:
        df = load_league(league, cfg, offline=args.offline)
        name = cfg["leagues"][league]["name"]
        print(f"\n=== {name} ({league}) - {len(df)} meczów ===")
        print(summarize(df).to_string())
        teams = sorted(set(df["home"]) | set(df["away"]))
        print(f"Drużyny ({len(teams)}): {', '.join(teams)}")
        missing = df["odds_h"].isna().sum()
        print(f"Mecze bez kursów: {missing}")
        fx = load_fixtures(league, cfg, results=df, offline=args.offline)
        print(f"Nadchodzące mecze w pliku fixtures: {len(fx)}")
        if not fx.empty:
            print(fx.head(15).to_string(index=False))


def cmd_backtest(args: argparse.Namespace) -> None:
    from backtest import run_league_backtest, format_report

    cfg = load_config()
    for league in args.leagues:
        res = run_league_backtest(league, cfg, n_test_seasons=args.seasons,
                                  tune_xi=not args.no_tune, offline=args.offline)
        print(format_report(res))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Typer DC - narzędzia CLI")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary", help="Pobierz dane i pokaż podsumowanie")
    s.add_argument("leagues", nargs="+")
    s.add_argument("--offline", action="store_true")
    s.set_defaults(func=cmd_summary)
    b = sub.add_parser("backtest", help="Walk-forward backtest i werdykt")
    b.add_argument("leagues", nargs="+")
    b.add_argument("--seasons", type=int, default=3, help="liczba sezonów testowych")
    b.add_argument("--no-tune", action="store_true", help="nie strój xi")
    b.add_argument("--offline", action="store_true")
    b.set_defaults(func=cmd_backtest)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
