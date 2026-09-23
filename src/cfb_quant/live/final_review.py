from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT = Path("reports/market_shortlist_audit_2026_week_4_v0_1.csv")
OUTPUT = Path("reports/final_review_top10_2026_week_4_v0_1.csv")

def main():
    df = pd.read_csv(INPUT)

    clean = df.loc[df["audit_status"].eq("CLEAN")].copy()

    clean = clean.sort_values(
        [
            "adjusted_ev",
            "adjusted_probability_edge",
            "uncertainty_confidence",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    top10 = clean.head(10).copy()
    top10["final_review_rank"] = range(1, len(top10) + 1)

    top10.to_csv(OUTPUT, index=False)

    print("=" * 145)
    print("CFB WEEK 4 FINAL TOP-10 REVIEW SLATE")
    print("=" * 145)
    print()

    cols = [
        "final_review_rank",
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
        "player_games_before",
        "player_season_games_before",
        "audit_status",
    ]

    cols = [c for c in cols if c in top10.columns]

    print(top10[cols].round(4).to_string(index=False))
    print()
    print("Clean candidates available:", len(clean))
    print("Final review slate:", len(top10))
    print("Saved:", OUTPUT)
    print()
    print("FINAL REVIEW means strongest clean model signals only; not yet official wager status.")

if __name__ == "__main__":
    main()
