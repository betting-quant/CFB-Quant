"""CFB-Quant historical schedule chronology audit (read-only for model artifacts).

Usage (from existing CFB-Quant root):
    python audit_schedule_chronology_v2.py
    python audit_schedule_chronology_v2.py --offline

Caches schedule JSON in data/raw/cfbd/chronology_v2/ when a CFBD request is
necessary. Does NOT modify model inputs, historical predictions, or weights.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

API_URL = "https://api.collegefootballdata.com/games"
ORDER = {"regular": 0, "postseason": 1, "spring_regular": 2, "spring_postseason": 3}


def _dotenv_key(root: Path) -> str | None:
    key = os.environ.get("CFBD_API_KEY") or os.environ.get("COLLEGE_FOOTBALL_DATA_API_KEY")
    if key:
        return key.strip()
    env_path = root / ".env"
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() in {"CFBD_API_KEY", "COLLEGE_FOOTBALL_DATA_API_KEY"}:
            return value.strip().strip('"\'') or None
    return None


def load_games(root: Path, year: int, token: str | None, offline: bool = False) -> tuple[list[dict], str]:
    archived = root / f"data/raw/cfbd/chronology_v2/games_{year}.json"
    existing = root / f"data/raw/cfbd/games/games_{year}.json"
    for path in (archived, existing):
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(data, list) and all(isinstance(x, dict) for x in data):
                return data, str(path)
            raise ValueError(f"Schedule cache is not a list of games: {path}")
    if offline:
        raise FileNotFoundError(f"No local schedule for {year}; run without --offline to fetch it")
    if not token:
        raise RuntimeError(
            f"Schedule missing for {year}, and no CFBD_API_KEY was found in the environment or .env. "
            "Do not paste your key in chat; put CFBD_API_KEY=... in the local .env."
        )
    try:
        r = requests.get(
            API_URL,
            params={"year": year, "seasonType": "both"},
            headers={"Authorization": f"Bearer {token}"},
            timeout=40,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"CFBD schedule request failed for {year}: {type(exc).__name__} (HTTP {getattr(getattr(exc, 'response', None), 'status_code', 'unknown')})") from exc
    if not isinstance(data, list) or not data:
        raise RuntimeError(f"CFBD returned empty or unexpected games response for {year}")
    archived.parent.mkdir(parents=True, exist_ok=True)
    archived.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data, f"CFBD API (archived to {archived})"


def schedules_to_frame(by_year: dict[int, list[dict]]) -> pd.DataFrame:
    games = []
    for year, raw_games in by_year.items():
        for g in raw_games:
            if "id" not in g:
                continue
            games.append({
                "game_id": str(g["id"]),
                "schedule_season": year,
                "kickoff_utc": pd.to_datetime(g.get("startDate"), utc=True, errors="coerce"),
                "home_team_cfbd": g.get("homeTeam"),
                "away_team_cfbd": g.get("awayTeam"),
                "kickoff_tbd": bool(g.get("startTimeTBD", False)),
                "scheduled_week": g.get("week"),
                "scheduled_season_type": g.get("seasonType"),
            })
    if not games:
        raise ValueError("No games found in schedule sources")
    result = pd.DataFrame(games)
    # Never silently choose one conflicting game ID/date.
    conflict = result.groupby("game_id")["kickoff_utc"].nunique(dropna=True)
    if (conflict > 1).any():
        raise ValueError(f"Conflicting kickoff timestamps for game IDs: {conflict[conflict>1].index.tolist()[:10]}")
    result = result.drop_duplicates("game_id").reset_index(drop=True)
    return result


def audit_chronology(features: pd.DataFrame, schedule: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    required = {"game_id", "season", "season_type", "week", "team", "opponent", "home_away"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")
    if features.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate (game_id, team) rows in features; resolve before chronology audit")
    feature = features[list(required)].copy()
    feature["game_id"] = feature["game_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    combined = feature.merge(schedule, on="game_id", how="left", validate="many_to_one", indicator=True)
    if len(combined) != len(features):
        raise ValueError("Merging schedules changed feature row counts")
    combined["_order"] = (
        combined.season_type.fillna("").astype(str).str.lower().str.strip().map(ORDER).fillna(9).astype(int)
    )
    combined["_week_num"] = pd.to_numeric(combined.week, errors="coerce")
    combined["kickoff_utc"] = pd.to_datetime(combined["kickoff_utc"], utc=True, errors="coerce")
    old_sort = ["season", "_order", "_week_num", "game_id", "team"]
    old = combined.sort_values(old_sort, kind="mergesort").copy()
    old["original_rank"] = old.groupby(["season", "team"], dropna=False).cumcount() + 1
    old["previous_kickoff_utc"] = old.groupby(["season", "team"], dropna=False).kickoff_utc.shift()
    old["previous_game_id"] = old.groupby(["season", "team"], dropna=False).game_id.shift()
    old["previous_opponent"] = old.groupby(["season", "team"], dropna=False).opponent.shift()
    inversions = old.loc[
        old.kickoff_utc.notna() & old.previous_kickoff_utc.notna()
        & (old.kickoff_utc < old.previous_kickoff_utc),
        ["season", "team", "game_id", "opponent", "week", "season_type", "kickoff_utc",
         "previous_game_id", "previous_opponent", "previous_kickoff_utc"],
    ].sort_values(["season", "team", "kickoff_utc"]).reset_index(drop=True)
    # Exact rank comparison only for teams with complete timestamp coverage.
    grp = old.groupby(["season", "team"], dropna=False)
    old["complete_team_schedule"] = grp.kickoff_utc.transform(lambda s: s.notna().all())
    ranked = old.loc[old.complete_team_schedule].sort_values(
        ["season", "team", "kickoff_utc", "game_id"], kind="mergesort"
    ).copy()
    ranked["chronological_rank"] = ranked.groupby(["season", "team"], dropna=False).cumcount() + 1
    wrong_rank = ranked.loc[ranked.original_rank != ranked.chronological_rank, [
        "season", "team", "game_id", "opponent", "season_type", "week", "kickoff_utc",
        "original_rank", "chronological_rank",
    ]].sort_values(["season", "team", "chronological_rank"]).reset_index(drop=True)
    # Detect side-of-game schedule-name mismatches; don't modify names or prediction data.
    side = combined.home_away.fillna("").str.lower()
    expected_team = np.where(side.eq("home"), combined.home_team_cfbd, combined.away_team_cfbd)
    wrong_team = combined.loc[combined.kickoff_utc.notna() & (
        pd.Series(expected_team, index=combined.index).fillna("").str.casefold().str.strip()
        != combined.team.fillna("").astype(str).str.casefold().str.strip()
    ), ["game_id", "season", "team", "home_away", "home_team_cfbd", "away_team_cfbd"]].copy()
    unmatched = combined.loc[combined.kickoff_utc.isna(), [
        "game_id", "season", "week", "season_type", "team", "opponent", "home_away"
    ]].copy()
    count_games = int(combined.game_id.nunique())
    matched_games = int(combined.loc[combined.kickoff_utc.notna(), "game_id"].nunique())
    tbd_games = int(combined.loc[combined.kickoff_tbd.eq(True), "game_id"].nunique())
    summary = {
        "feature_games": count_games,
        "matched_games": matched_games,
        "missing_games": count_games - matched_games,
        "coverage_pct": round(100 * matched_games / max(count_games, 1), 3),
        "games_with_kickoff_tbd_flag": tbd_games,
        "adjacent_team_order_inversions": len(inversions),
        "team_game_rows_out_of_chronological_rank": len(wrong_rank),
        "team_name_or_side_mismatches": len(wrong_team),
        "read_only": True,
        "upstream_source_snapshot_verified": False,
    }
    return combined, summary, {
        "team_order_inversions": inversions,
        "misranked_team_games": wrong_rank,
        "unmatched_team_games": unmatched,
        "schedule_team_mismatches": wrong_team,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=".")
    p.add_argument("--features", default="data/processed/cfb_matchup_features_v0_1.parquet")
    p.add_argument("--output", default="reports/spread_total_schedule_chronology_v2")
    p.add_argument("--offline", action="store_true", help="Use local schedule caches only")
    args = p.parse_args()
    root = Path(args.root)
    feat_path = root / args.features
    if not feat_path.is_file():
        p.error(f"Historical feature dataset not found: {feat_path}")
    feat = pd.read_parquet(feat_path)
    years = sorted(pd.to_numeric(feat.season, errors="raise").astype(int).unique())
    key = _dotenv_key(root)
    schedules: dict[int, list[dict]] = {}
    sources = {}
    print(f"Historical seasons: {years[0]}–{years[-1]} ({len(years)} years)")
    for year in years:
        games, source = load_games(root, year, key, offline=args.offline)
        schedules[year] = games
        sources[str(year)] = source
        print(f"  {year}: {len(games)} schedule games from {source}")
    schedule = schedules_to_frame(schedules)
    merged, summary, reports = audit_chronology(feat, schedule)
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)
    for label, frame in reports.items():
        frame.to_csv(out / f"{label}.csv", index=False)
    (merged[["game_id", "season", "team", "opponent", "home_away", "week", "season_type", "kickoff_utc"]]
     .sort_values(["season", "game_id", "team"])
     .to_csv(out / "team_game_kickoff_map.csv", index=False))
    summary["schedule_sources"] = sources
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSCHEDULE CHRONOLOGY AUDIT — MODEL DATA UNCHANGED")
    for field in ["feature_games", "matched_games", "missing_games", "coverage_pct", "games_with_kickoff_tbd_flag", "adjacent_team_order_inversions", "team_game_rows_out_of_chronological_rank", "team_name_or_side_mismatches"]:
        print(f"  {field}: {summary[field]}")
    if summary["missing_games"] or summary["adjacent_team_order_inversions"] or summary["team_game_rows_out_of_chronological_rank"]:
        print("ACTION REQUIRED: Repair game chronology and regenerate historical features before trusting previous backtests.")
    else:
        print("Ordering appears consistent for timestamped games. This does NOT verify original source availability or eliminate all leakage.")
    print(f"Reports saved to: {out}")


if __name__ == "__main__":
    main()
