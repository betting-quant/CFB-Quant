from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("reports/market_shortlist_top20_2026_week_4_v0_1.csv")
OUTPUT = Path("reports/market_shortlist_audit_2026_week_4_v0_1.csv")

def main():
    df = pd.read_csv(INPUT)

    df["projection_line_gap"] = df["projection_blended"] - df["line"]

    if "local_residual_std" in df.columns:
        denom = df["local_residual_std"].replace(0, np.nan)
        df["projection_line_gap_z"] = (
            df["projection_line_gap"].abs() / denom
        )
    else:
        df["projection_line_gap_z"] = np.nan

    if "standardized_model_disagreement" not in df.columns:
        if "local_residual_std" in df.columns and "model_disagreement" in df.columns:
            denom = df["local_residual_std"].replace(0, np.nan)
            df["standardized_model_disagreement"] = (
                df["model_disagreement"].abs() / denom
            )
        else:
            df["standardized_model_disagreement"] = np.nan

    flags = []
    audit_status = []

    for _, row in df.iterrows():
        row_flags = []

        games = float(row.get("player_games_before", 0) or 0)
        season_games = float(row.get("player_season_games_before", 0) or 0)
        prob = float(row.get("adjusted_probability", 0.5))

        gap_z = row.get("projection_line_gap_z", np.nan)
        disagreement_z = row.get("standardized_model_disagreement", np.nan)
        neighborhood_ratio = row.get("neighborhood_ratio", np.nan)

        if games < 5:
            row_flags.append("THIN_PLAYER_HISTORY")

        if season_games < 3:
            row_flags.append("THIN_2026_SAMPLE")

        if prob >= 0.85:
            row_flags.append("EXTREME_PROBABILITY")

        if pd.notna(gap_z) and float(gap_z) >= 2.0:
            row_flags.append("EXTREME_LINE_GAP")

        if pd.notna(disagreement_z) and float(disagreement_z) >= 0.75:
            row_flags.append("MODEL_DISAGREEMENT")

        if pd.notna(neighborhood_ratio) and float(neighborhood_ratio) >= 0.25:
            row_flags.append("WEAK_LOCAL_SUPPORT")

        if len(row_flags) >= 2:
            status = "MANUAL_REVIEW"
        elif len(row_flags) == 1:
            status = "CAUTION"
        else:
            status = "CLEAN"

        flags.append("|".join(row_flags))
        audit_status.append(status)

    df["audit_flags"] = flags
    df["audit_status"] = audit_status

    status_rank = {
        "CLEAN": 0,
        "CAUTION": 1,
        "MANUAL_REVIEW": 2,
    }

    df["_audit_rank"] = df["audit_status"].map(status_rank)

    df = df.sort_values(
        ["_audit_rank", "adjusted_ev", "adjusted_probability_edge"],
        ascending=[True, False, False],
    ).drop(columns="_audit_rank").reset_index(drop=True)

    df["audit_rank"] = range(1, len(df) + 1)

    df.to_csv(OUTPUT, index=False)

    print("=" * 145)
    print("CFB TOP-20 SHORTLIST SANITY AUDIT")
    print("=" * 145)
    print()
    print(df["audit_status"].value_counts().to_string())
    print()

    cols = [
        "audit_rank",
        "shortlist_rank",
        "away_team",
        "home_team",
        "player_name_model",
        "target",
        "line",
        "projection_blended",
        "raw_best_side",
        "raw_best_odds",
        "adjusted_probability",
        "adjusted_probability_edge",
        "adjusted_ev",
        "uncertainty_confidence",
        "projection_line_gap_z",
        "standardized_model_disagreement",
        "player_games_before",
        "player_season_games_before",
        "audit_status",
        "audit_flags",
    ]

    cols = [c for c in cols if c in df.columns]

    print(df[cols].round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)
    print()
    print("Audit status is a robustness screen, not an official bet label.")

if __name__ == "__main__":
    main()
