"""Read-only preflight for existing CFB-Quant spread/total matchup features.

This is a data-quality audit, NOT proof that pregame features are leak-free.
It does not train models, use odds, call an API, or modify feature datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FORBIDDEN_EXACT = {
    "points", "opponent_points", "pass_attempts", "rush_attempts",
    "total_plays", "passing_yards", "rushing_yards", "possession_seconds",
}
KEYS = {"game_id", "season", "week", "team", "opponent", "home_away"}
POSSIBLE_TIME_COLUMNS = (
    "game_start_utc", "kickoff_utc", "start_date", "start_time", "start_date_utc", "scheduled_start_utc",
)
SNAPSHOT_COLUMNS = ("retrieved_at_utc", "as_of_utc", "available_at_utc", "source_snapshot_utc")


def audit_frame(frame: pd.DataFrame, manifest: dict) -> dict:
    missing_keys = KEYS.difference(frame.columns)
    if missing_keys:
        raise ValueError(f"Missing required matchup keys: {sorted(missing_keys)}")
    features = manifest.get("feature_columns")
    if not isinstance(features, list) or not features:
        raise ValueError("Manifest requires nonempty feature_columns list")
    if len(set(features)) != len(features):
        raise ValueError("Duplicate feature names in manifest")
    if len(frame) == 0:
        raise ValueError("Feature dataset is empty")
    if frame[list(KEYS)].isna().any().any():
        raise ValueError("Missing team/game/season/week or home_away key")
    missing_features = sorted(set(features).difference(frame.columns))
    if missing_features:
        raise ValueError(f"Manifest references absent features: {missing_features[:25]}")
    leaked_direct_features = sorted(set(features).intersection(FORBIDDEN_EXACT))
    if leaked_direct_features:
        raise ValueError(f"Current-game scoring targets selected as features: {leaked_direct_features}")
    if frame.duplicated(["game_id", "team"]).any():
        raise ValueError("Duplicate team-game rows in matchup feature file")
    home_away = frame["home_away"].astype(str).str.lower()
    if not home_away.isin(["home", "away"]).all():
        raise ValueError("Invalid home_away values")

    df = frame.copy()
    df["home_away"] = home_away
    numeric = df[features].apply(pd.to_numeric, errors="coerce")
    finite = numeric.replace([np.inf, -np.inf], np.nan)
    missing = finite.isna()
    summary_features = pd.DataFrame({
        "feature": features,
        "missing_pct_all": missing.mean(axis=0).to_numpy() * 100,
        "n_non_null": (~missing).sum(axis=0).to_numpy(),
        "n_distinct_non_null": finite.nunique(dropna=True).to_numpy(),
    })
    summary_features["feature_family"] = summary_features.feature.str.extract(r"^(off_|def_)", expand=False).fillna("context")
    summary_features = summary_features.sort_values(["missing_pct_all", "feature"], ascending=[False, True])

    early = pd.to_numeric(df.week, errors="coerce").le(4)
    if early.isna().any():
        raise ValueError("Unusable week numbers")
    period_rows = []
    for period, mask in (("weeks_1_to_4", early), ("week_5_plus", ~early)):
        sub = missing.loc[mask]
        period_rows.append({
            "period": period,
            "team_games": int(mask.sum()),
            "mean_feature_missing_pct": float(sub.to_numpy().mean() * 100) if len(sub) else np.nan,
            "median_row_missing_pct": float(sub.mean(axis=1).median() * 100) if len(sub) else np.nan,
            "all_features_missing_team_games": int(sub.all(axis=1).sum()),
        })
    periods = pd.DataFrame(period_rows)

    pair = df.groupby("game_id", dropna=False).agg(
        rows=("team", "size"),
        home_count=("home_away", lambda s: int((s == "home").sum())),
        away_count=("home_away", lambda s: int((s == "away").sum())),
    ).reset_index()
    pair["complete_pair"] = (pair.rows.eq(2) & pair.home_count.eq(1) & pair.away_count.eq(1))
    complete = df.loc[df.game_id.isin(pair.loc[pair.complete_pair, "game_id"])]
    h = complete.loc[complete.home_away.eq("home"), ["game_id", "team", "opponent"]]
    a = complete.loc[complete.home_away.eq("away"), ["game_id", "team", "opponent"]]
    joined = h.merge(a, on="game_id", suffixes=("_home", "_away"), validate="one_to_one")
    mismatched = joined.loc[~(
        joined.team_home.astype(str).eq(joined.opponent_away.astype(str)) &
        joined.team_away.astype(str).eq(joined.opponent_home.astype(str))
    )]

    # Matching same-team season/week rows deserve review: week ordering is not a
    # guarantee of kickoff ordering, e.g., rescheduled games or inconsistent data.
    repeated_team_weeks = df.loc[
        df.duplicated(["season", "week", "team"], keep=False),
        ["season", "week", "game_id", "team", "opponent", "home_away"],
    ].sort_values(["season", "week", "team", "game_id"])

    summary = {
        "team_game_rows": len(df),
        "games": int(df.game_id.nunique()),
        "feature_count": len(features),
        "complete_pairs": int(pair.complete_pair.sum()),
        "incomplete_pairs": int((~pair.complete_pair).sum()),
        "opponent_identity_mismatches_on_complete_pairs": len(mismatched),
        "repeated_team_week_rows": len(repeated_team_weeks),
        "no_variation_features": int((summary_features.n_distinct_non_null <= 1).sum()),
        "mostly_missing_features_over_50pct": int((summary_features.missing_pct_all > 50).sum()),
        "available_game_time_columns": [col for col in POSSIBLE_TIME_COLUMNS if col in df.columns],
        "available_snapshot_time_columns": [col for col in SNAPSHOT_COLUMNS if col in df.columns],
        "warnings": [
            "Rolling features and chronological model training do not prove source availability at historical kickoff.",
            "Some FBS/FCS matchups may yield incomplete pairs; FBS-only eligibility must be separately verified.",
            "Comparing with retrospective CFBD ratings can introduce hindsight unless historical snapshots are available.",
            "Do not declare this model betting-ready based on this preflight.",
        ],
    }
    return {
        "summary": summary,
        "feature_missingness": summary_features,
        "period_missingness": periods,
        "game_pair_audit": pair,
        "opponent_mismatches": mismatched,
        "repeated_team_weeks": repeated_team_weeks,
    }


def save_audit(result: dict, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name in ("feature_missingness", "period_missingness", "game_pair_audit", "opponent_mismatches", "repeated_team_weeks"):
        result[name].to_csv(outdir / f"{name}.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(result["summary"], indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only CFB matchup-feature quality preflight")
    ap.add_argument("--features", default="data/processed/cfb_matchup_features_v0_1.parquet")
    ap.add_argument("--manifest", default="reports/cfb_matchup_feature_manifest_v0_1.json")
    ap.add_argument("--outdir", default="reports/spread_total_feature_preflight_v1")
    args = ap.parse_args()
    df = pd.read_parquet(args.features)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    result = audit_frame(df, manifest)
    save_audit(result, Path(args.outdir))
    s = result["summary"]
    print("CFB MATCHUP FEATURE PREFLIGHT — READ ONLY")
    for k, v in s.items():
        if k != "warnings":
            print(f"  {k}: {v}")
    print("Missingness by period:")
    print(result["period_missingness"].round(2).to_string(index=False))
    print("Worst missing features:")
    print(result["feature_missingness"].head(12).round(1).to_string(index=False))
    print(f"Saved to: {args.outdir}")
    print("WARNING: Preflight cannot establish complete point-in-time validity; source timestamps require an independent audit.")


if __name__ == "__main__":
    main()
