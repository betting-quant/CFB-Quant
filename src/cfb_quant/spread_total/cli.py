"""CFB-Quant spread/total research CLI.

Commands: validate and forecast. Does not submit bets or claim proven EV.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import (audit_game_pairs, build_game_errors, prior_errors, simulate_scores, stable_seed,
                   fair_summary, settle, expected_roi, empirical_crps)


def _save(df: pd.DataFrame, path: str) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    print(f"Saved: {dest} ({len(df)} rows)")


def _load_oof(path: str, model: str) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing points OOF file: {source}. Run the oof_points step first.")
    frame = pd.read_csv(source) if source.suffix.lower() == ".csv" else pd.read_parquet(source)
    audit = audit_game_pairs(frame, model=model)
    _save(audit, "reports/spread_total_pair_audit_v1.csv")
    counts = audit.status.value_counts()
    total = len(audit)
    n_good = int(counts.get("complete", 0))
    n_bad = total - n_good
    print(f"PAIR AUDIT: {n_good:,}/{total:,} complete games; {n_bad:,} incomplete; "
          f"coverage={100*n_good/total:.1f}%")
    print("By status:", counts.to_dict())
    if n_bad:
        print("Incomplete game rows excluded, NOT imputed. "
              "Review reports/spread_total_pair_audit_v1.csv")
        print("Incomplete by season:")
        print(audit.loc[audit.status.ne("complete")].groupby(["season", "status"]).size().to_string())
    if n_good / total < 0.70:
        raise ValueError("<70% of games have two-side predictions; investigate source data before validation")
    return build_game_errors(frame, model=model, allow_incomplete=True)


def _score_events(dist, actual_margin: float, actual_total: float) -> dict[str, float]:
    out = {}
    for line in (-7.5, 0.5, 7.5):
        key = f"home_margin_gt_{str(line).replace('.', 'p').replace('-', 'm')}"
        p = float(np.mean(dist.margin > line))
        observed = float(actual_margin > line)
        out[f"brier_{key}"] = (p - observed) ** 2
    for line in (45.5, 55.5, 65.5):
        key = f"total_over_{str(line).replace('.', 'p')}"
        p = float(np.mean(dist.total > line))
        observed = float(actual_total > line)
        out[f"brier_{key}"] = (p - observed) ** 2
    return out


def validate(args: argparse.Namespace) -> None:
    games = _load_oof(args.oof, args.model)
    output = []
    seasons = sorted(int(v) for v in games.season.unique())
    for year in seasons:
        if year < args.start_year:
            continue
        test = games.loc[games.season.eq(year)]
        try:
            train = prior_errors(games, before_season=year,
                                 min_games=args.min_history_games, min_seasons=args.min_history_seasons)
        except ValueError as exc:
            print(f"SKIP {year}: {exc}")
            continue
        for r in test.itertuples(index=False):
            dist = simulate_scores(r.home_pred, r.away_pred, train,
                                   n=args.samples, seed=stable_seed(r.game_id, args.seed))
            pred = fair_summary(dist)
            m = float(r.actual_margin)
            t = float(r.actual_total)
            output.append({
                "game_id": r.game_id, "season": year, "week": r.week,
                "home_team": r.home_team, "away_team": r.away_team,
                "actual_margin": m, "actual_total": t,
                "original_margin": r.pred_margin, "original_total": r.pred_total,
                **pred,
                "margin_abs_error": abs(pred["fair_home_margin_mean"] - m),
                "total_abs_error": abs(pred["fair_total_mean"] - t),
                "crps_margin": empirical_crps(dist.margin, m),
                "crps_total": empirical_crps(dist.total, t),
                "margin_80_covered": float(np.quantile(dist.margin, .10) <= m <= np.quantile(dist.margin, .90)),
                "total_80_covered": float(np.quantile(dist.total, .10) <= t <= np.quantile(dist.total, .90)),
                "original_margin_abs_error": abs(r.pred_margin - m),
                "original_total_abs_error": abs(r.pred_total - t),
                "prior_games": len(train),
                "prior_max_season": int(train.season.max()),
                **_score_events(dist, m, t),
            })
        print(f"Validated year={year} games={len(test)} residuals={len(train)} prior_max_season={int(train.season.max())}")
    if not output:
        raise RuntimeError("No historical validation games. Need additional historical OOF seasons.")
    result = pd.DataFrame(output)
    assert (result.prior_max_season < result.season).all()
    _save(result, args.out)
    metrics = ["margin_abs_error", "total_abs_error", "crps_margin", "crps_total",
               "margin_80_covered", "total_80_covered", "original_margin_abs_error", "original_total_abs_error"]
    metrics += [c for c in result if c.startswith("brier_")]
    print("\nVALIDATION METRICS (illustrative only until upstream inputs audited):")
    print(result.groupby("season")[metrics].mean().round(3).to_string())
    print("\nNo profitable betting claim can be made from these statistics alone.")
    print("CAUTION: 2025 was used during development of the existing CFB-Quant model; it is not a pristine untouched test.")


def _read_live_games(path: str, as_of_utc: pd.Timestamp, season: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"game_id", "start_date", "home_team", "away_team", "home_points", "away_points"}
    if required.difference(df.columns):
        raise ValueError(f"Live games CSV is missing columns: {sorted(required.difference(df.columns))}")
    if df.game_id.duplicated().any():
        raise ValueError("Duplicate live game IDs")
    starts = pd.to_datetime(df.start_date, errors="coerce", utc=True)
    if starts.isna().any():
        raise ValueError("All games must have valid kickoff timestamps")
    if (starts <= as_of_utc).any():
        raise ValueError("A selected game started at/before as-of. Pregame-only pipeline refuses it.")
    if (starts.dt.year != season).any():
        raise ValueError("Game kickoff years disagree with specified --season")
    df["start_date_utc"] = starts
    for col in ["home_points", "away_points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["home_points", "away_points"]].isna().any().any():
        raise ValueError("Non-numeric live point projections")
    if not np.isfinite(df[["home_points", "away_points"]].to_numpy()).all():
        raise ValueError("Non-finite live projections")
    return df


def _read_markets(path: str, *, game_ids: set[str], as_of_utc: pd.Timestamp,
                  starts: dict[str, pd.Timestamp]) -> pd.DataFrame:
    df = pd.read_csv(path)
    req = {"game_id", "market", "selection", "line", "american_odds", "bookmaker", "quoted_at_utc"}
    if req.difference(df.columns):
        raise ValueError(f"Market CSV missing columns: {sorted(req.difference(df.columns))}")
    df["game_id"] = df["game_id"].astype(str)
    if not df["game_id"].isin(game_ids).all():
        raise ValueError("Market CSV has game IDs not present in live forecasts")
    times = pd.to_datetime(df.quoted_at_utc, errors="coerce", utc=True)
    if times.isna().any() or (times > as_of_utc).any():
        raise ValueError("All odds need valid timestamps no later than --as-of")
    if any(times.iloc[i] >= starts[id_] for i, id_ in enumerate(df.game_id)):
        raise ValueError("Market quote at or after kickoff")
    df["quoted_at_utc"] = times
    df["line"] = pd.to_numeric(df["line"], errors="raise")
    df["american_odds"] = pd.to_numeric(df["american_odds"], errors="raise")
    return df


def forecast(args: argparse.Namespace) -> None:
    if not args.as_of:
        raise ValueError("An explicit --as-of timestamp is REQUIRED for a pregame report")
    as_of_utc = pd.to_datetime(args.as_of, utc=True, errors="raise")
    if pd.isna(as_of_utc):
        raise ValueError("Invalid --as-of")
    if not pd.Timestamp(args.as_of).tzinfo:
        raise ValueError("--as-of must contain a timezone, e.g. 2026-10-10T08:00:00-06:00")
    history = _load_oof(args.oof, args.model)
    prior = prior_errors(history, before_season=args.season,
                         min_games=args.min_history_games, min_seasons=args.min_history_seasons)
    games = _read_live_games(args.games, as_of_utc, args.season)
    starts = dict(zip(games.game_id.astype(str), games.start_date_utc))
    results, sims = [], {}
    for r in games.itertuples(index=False):
        game_id = str(r.game_id)
        dist = simulate_scores(r.home_points, r.away_points, prior,
                               n=args.samples, seed=stable_seed(game_id, args.seed))
        sims[game_id] = dist
        results.append({"game_id": game_id, "home_team": r.home_team,
                        "away_team": r.away_team, "start_date_utc": r.start_date_utc.isoformat(),
                        "raw_home_points": float(r.home_points), "raw_away_points": float(r.away_points),
                        "raw_total": float(r.home_points+r.away_points),
                        "raw_home_margin": float(r.home_points-r.away_points),
                        **fair_summary(dist), "residual_games": len(prior),
                        "residual_end_year": int(prior.season.max()),
                        "as_of_utc": as_of_utc.isoformat(),
                        "status": "RESEARCH_ONLY_NOT_CALIBRATED"})
    _save(pd.DataFrame(results), args.out)
    if args.markets:
        quotes = _read_markets(args.markets, game_ids=set(sims), as_of_utc=as_of_utc, starts=starts)
        market_rows = []
        for r in quotes.itertuples(index=False):
            w, p, l = settle(sims[str(r.game_id)], market=str(r.market),
                             selection=str(r.selection), line=float(r.line))
            roi = expected_roi(w, p, l, int(r.american_odds))
            stale_hours = (as_of_utc-r.quoted_at_utc).total_seconds()/3600
            market_rows.append({"game_id": r.game_id, "market": r.market, "selection": r.selection,
                                "line": r.line, "american_odds": int(r.american_odds),
                                "bookmaker": r.bookmaker, "quoted_at_utc": r.quoted_at_utc.isoformat(),
                                "quote_age_hours": round(stale_hours, 2),
                                "p_win": w, "p_push": p, "p_lose": l, "estimated_roi": roi,
                                "status": "STALE_RECHECK_ODDS" if stale_hours > args.max_quote_hours else "RESEARCH_ONLY_NOT_CALIBRATED"})
        markets_out = args.markets_out or str(Path(args.out).with_name(Path(args.out).stem + "_market_review.csv"))
        _save(pd.DataFrame(market_rows), markets_out)
    print("RESEARCH-ONLY: Outputs require validated pregame data, distribution calibration, and timestamped market tracking.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Research-only team spreads/totals probability upgrade")
    sub = ap.add_subparsers(dest="mode", required=True)
    for action in ("validate", "forecast"):
        sp = sub.add_parser(action)
        sp.add_argument("--oof", default="data/processed/cfb_spread_total_points_oof_v1.parquet")
        sp.add_argument("--model", default="ridge")
        sp.add_argument("--min-history-games", type=int, default=300)
        sp.add_argument("--min-history-seasons", type=int, default=3)
        sp.add_argument("--samples", type=int, default=10000)
        sp.add_argument("--seed", type=int, default=731)
    v = sub.choices["validate"]
    v.add_argument("--start-year", type=int, default=2019)
    v.add_argument("--out", default="reports/spread_total_historical_validation_v1.csv")
    f = sub.choices["forecast"]
    f.add_argument("--season", required=True, type=int)
    f.add_argument("--as-of", required=True)
    f.add_argument("--games", required=True)
    f.add_argument("--markets")
    f.add_argument("--out", default="reports/spread_total_forecast_v1.csv")
    f.add_argument("--markets-out")
    f.add_argument("--max-quote-hours", type=float, default=6.0)
    args = ap.parse_args()
    (validate if args.mode == "validate" else forecast)(args)

if __name__ == "__main__":
    main()
