from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

PLAYER_DATA = Path("data/processed/cfb_pregame_features_v0_1.parquet")
PLAYER_MANIFEST = Path("reports/cfb_feature_manifest_v0_1.json")
MATCHUP_OOF = Path("data/processed/cfb_matchup_oof_features_v0_1.parquet")

OUTPUT_DATA = Path("data/processed/cfb_pregame_features_matchup_v0_1.parquet")
OUTPUT_MANIFEST = Path("reports/cfb_feature_manifest_matchup_v0_1.json")
COVERAGE_PATH = Path("reports/player_matchup_oof_coverage_v0_1.csv")

WINNER_SOURCE_COLUMNS = {
    "matchup_proj_points": "matchup_ridge_points",
    "matchup_proj_pass_attempts": "matchup_extra_trees_pass_attempts",
    "matchup_proj_rush_attempts": "matchup_hist_gradient_boosting_rush_attempts",
    "matchup_proj_total_plays": "matchup_hist_gradient_boosting_total_plays",
    "matchup_proj_passing_yards": "matchup_hist_gradient_boosting_passing_yards",
    "matchup_proj_rushing_yards": "matchup_ridge_rushing_yards",
    "matchup_proj_possession_seconds": "matchup_hist_gradient_boosting_possession_seconds",
}


def main():
    print("=" * 90)
    print("PLAYER + MATCHUP OOF FEATURE MERGE v0.1")
    print("=" * 90)

    players = pd.read_parquet(PLAYER_DATA)
    matchup = pd.read_parquet(MATCHUP_OOF)
    manifest = json.loads(PLAYER_MANIFEST.read_text(encoding="utf-8"))

    original_rows = len(players)
    original_features = list(manifest["feature_columns"])
    matchup_features = list(WINNER_SOURCE_COLUMNS)

    print(f"Player rows before: {original_rows:,}")
    print(f"Original features: {len(original_features):,}")
    print(f"Matchup OOF team rows: {len(matchup):,}")

    if matchup.duplicated(["game_id", "team"]).any():
        raise RuntimeError("Duplicate matchup team-game rows.")

    missing = [
        c for c in WINNER_SOURCE_COLUMNS.values()
        if c not in matchup.columns
    ]
    if missing:
        raise RuntimeError(f"Missing matchup columns: {missing}")

    compact = matchup[
        ["game_id", "team", *WINNER_SOURCE_COLUMNS.values()]
    ].copy()

    compact = compact.rename(
        columns={
            source: dest
            for dest, source in WINNER_SOURCE_COLUMNS.items()
        }
    )

    merged = players.merge(
        compact,
        on=["game_id", "team"],
        how="left",
        validate="many_to_one",
    )

    if len(merged) != original_rows:
        raise RuntimeError(
            f"Row multiplication: {original_rows} -> {len(merged)}"
        )

    overlap = set(original_features) & set(matchup_features)
    if overlap:
        raise RuntimeError(f"Feature overlap: {sorted(overlap)}")

    enhanced_features = original_features + matchup_features

    if len(enhanced_features) != len(set(enhanced_features)):
        raise RuntimeError("Duplicate enhanced feature names.")

    direct_targets = {
        "pass_attempts",
        "completions",
        "passing_yards",
        "rush_attempts",
        "rushing_yards",
        "receptions",
        "receiving_yards",
    }

    leaks = direct_targets & set(enhanced_features)
    if leaks:
        raise RuntimeError(f"Direct target leakage: {sorted(leaks)}")

    coverage_rows = []

    print()
    print("MATCHUP COVERAGE BY SEASON")
    print("-" * 90)

    for season, frame in merged.groupby("season", sort=True):
        present = frame[matchup_features].notna().any(axis=1)
        coverage = float(present.mean())

        coverage_rows.append({
            "season": int(season),
            "rows": len(frame),
            "rows_with_matchup": int(present.sum()),
            "coverage": coverage,
        })

        print(
            f"{int(season)}: "
            f"{present.sum():,}/{len(frame):,} "
            f"({coverage:.2%})"
        )

    coverage = pd.DataFrame(coverage_rows)

    row_2014 = coverage.loc[coverage["season"].eq(2014)]
    if not row_2014.empty and float(row_2014.iloc[0]["coverage"]) != 0:
        raise RuntimeError("2014 unexpectedly has matchup projections.")

    if not coverage["season"].eq(2025).any():
        raise RuntimeError("2025 holdout missing.")

    merged.to_parquet(OUTPUT_DATA, index=False)

    enhanced_manifest = dict(manifest)
    enhanced_manifest["feature_columns"] = enhanced_features
    enhanced_manifest["feature_count"] = len(enhanced_features)
    enhanced_manifest["matchup_oof_features"] = matchup_features
    enhanced_manifest["matchup_oof_source"] = str(MATCHUP_OOF)
    enhanced_manifest["matchup_model_winner_sources"] = WINNER_SOURCE_COLUMNS
    enhanced_manifest["base_player_feature_manifest"] = str(PLAYER_MANIFEST)
    enhanced_manifest["matchup_oof_method"] = (
        "Chronological season-level OOF matchup predictions; "
        "each prediction season was fit only on prior seasons."
    )

    OUTPUT_MANIFEST.write_text(
        json.dumps(enhanced_manifest, indent=2),
        encoding="utf-8",
    )

    coverage.to_csv(COVERAGE_PATH, index=False)

    print()
    print("=" * 90)
    print("MERGE COMPLETE")
    print("=" * 90)
    print(f"Rows after: {len(merged):,}")
    print(f"Original features: {len(original_features):,}")
    print(f"Matchup features added: {len(matchup_features):,}")
    print(f"Enhanced features: {len(enhanced_features):,}")

    for feature in matchup_features:
        print(f"  {feature}")

    print()
    print(f"Enhanced data: {OUTPUT_DATA}")
    print(f"Enhanced manifest: {OUTPUT_MANIFEST}")
    print(f"Coverage audit: {COVERAGE_PATH}")
    print("Original player files: UNCHANGED")


if __name__ == "__main__":
    main()
