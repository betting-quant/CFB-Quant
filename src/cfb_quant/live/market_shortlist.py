from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT = Path("reports/market_role_audit_2026_week_4_v0_2.csv")
OUTPUT = Path("reports/market_shortlist_2026_week_4_v0_1.csv")
TOP_OUTPUT = Path("reports/market_shortlist_top20_2026_week_4_v0_1.csv")

MAX_PER_GAME = 2
MAX_SHORTLIST = 20

def main():
    df = pd.read_csv(INPUT)

    df["decision_status"] = "NOT_SHORTLISTED"
    df["correlation_reason"] = ""

    candidates = df.loc[
        df["post_role_status"].eq("CANDIDATE")
    ].copy()

    # Rank strongest mathematical signals first.
    candidates = candidates.sort_values(
        [
            "adjusted_ev",
            "adjusted_probability_edge",
            "uncertainty_confidence",
        ],
        ascending=[False, False, False],
    ).reset_index()

    selected_indices = []
    selected_players = set()
    game_counts = {}

    for _, row in candidates.iterrows():
        original_index = int(row["index"])
        player = str(row["player_name_model"])
        game = str(row["game_id"])

        if player in selected_players:
            df.at[original_index, "decision_status"] = "CORRELATED_OUT"
            df.at[original_index, "correlation_reason"] = "Lower-ranked prop for same player"
            continue

        if game_counts.get(game, 0) >= MAX_PER_GAME:
            df.at[original_index, "decision_status"] = "CORRELATED_OUT"
            df.at[original_index, "correlation_reason"] = "Game already has maximum shortlisted exposure"
            continue

        if len(selected_indices) >= MAX_SHORTLIST:
            df.at[original_index, "decision_status"] = "BELOW_TOP20"
            df.at[original_index, "correlation_reason"] = "Outside global top-20 shortlist"
            continue

        selected_indices.append(original_index)
        selected_players.add(player)
        game_counts[game] = game_counts.get(game, 0) + 1

        df.at[original_index, "decision_status"] = "SHORTLIST"

    shortlist = df.loc[
        df["decision_status"].eq("SHORTLIST")
    ].copy()

    shortlist = shortlist.sort_values(
        [
            "adjusted_ev",
            "adjusted_probability_edge",
            "uncertainty_confidence",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    shortlist["shortlist_rank"] = range(1, len(shortlist) + 1)

    rank_map = dict(
        zip(
            shortlist["player_name_model"].astype(str) + "|" + shortlist["target"].astype(str),
            shortlist["shortlist_rank"],
        )
    )

    df["shortlist_rank"] = df.apply(
        lambda r: rank_map.get(str(r["player_name_model"]) + "|" + str(r["target"])),
        axis=1,
    )

    df.to_csv(OUTPUT, index=False)
    shortlist.to_csv(TOP_OUTPUT, index=False)

    print("=" * 135)
    print("CFB PLAYER PROP CORRELATION-PRUNED SHORTLIST")
    print("=" * 135)
    print()
    print(df["decision_status"].value_counts().to_string())
    print()

    cols = [
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
        "player_games_before",
        "player_season_games_before",
    ]

    cols = [c for c in cols if c in shortlist.columns]

    print("TOP 20 SHORTLIST")
    print("-" * 135)
    print(shortlist[cols].round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)
    print("Saved:", TOP_OUTPUT)
    print()
    print("SHORTLIST is correlation-pruned review status, not official bet status.")

if __name__ == "__main__":
    main()
