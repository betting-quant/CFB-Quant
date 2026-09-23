from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT = Path("reports/final_bet_report_2026_week_4_v0_1.csv")
OUTPUT = Path("reports/final_portfolio_2026_week_4_v0_1.csv")

MAX_TOP_BETS = 5
MAX_PER_GAME = 1

def main():
    df = pd.read_csv(INPUT)

    bets = df.loc[df["decision"].eq("BET")].copy()

    bets = bets.sort_values(
        [
            "adjusted_ev",
            "adjusted_probability_edge",
            "uncertainty_confidence",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected = []
    used_games = set()

    for idx, row in bets.iterrows():
        if len(selected) >= MAX_TOP_BETS:
            break

        game = str(row["game_id"])

        if game in used_games:
            continue

        selected.append(idx)
        used_games.add(game)

    bets["portfolio_status"] = "SECONDARY"
    bets.loc[selected, "portfolio_status"] = "TOP_BET"

    top = bets.loc[bets["portfolio_status"].eq("TOP_BET")].copy()
    top = top.sort_values(
        ["adjusted_ev", "adjusted_probability_edge"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top["portfolio_rank"] = range(1, len(top) + 1)

    # Keep beta staking conservative.
    stake_map = {
        1: 0.70,
        2: 0.70,
        3: 0.60,
        4: 0.60,
        5: 0.50,
    }

    top["portfolio_units"] = top["portfolio_rank"].map(stake_map)
    top["portfolio_dollars"] = top["portfolio_units"] * 25.0

    top.to_csv(OUTPUT, index=False)

    print("=" * 150)
    print("CFB WEEK 4 FINAL PORTFOLIO - TOP 5 BETA")
    print("=" * 150)
    print()

    cols = [
        "portfolio_rank",
        "away_team",
        "home_team",
        "player_name_model",
        "target",
        "raw_best_side",
        "line",
        "projection_blended",
        "raw_best_odds",
        "adjusted_probability",
        "adjusted_probability_edge",
        "adjusted_ev",
        "uncertainty_confidence",
        "max_playable_price_5pct_ev",
        "portfolio_units",
        "portfolio_dollars",
    ]

    cols = [c for c in cols if c in top.columns]

    print(top[cols].round(4).to_string(index=False))
    print()
    print("Total units:", round(top["portfolio_units"].sum(), 2))
    print("Total dollars:", round(top["portfolio_dollars"].sum(), 2))
    print()
    print("Saved:", OUTPUT)
    print()
    print("TOP_BET remains a beta portfolio label, not historically ROI-validated.")

if __name__ == "__main__":
    main()
