from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MARKET = Path("reports/market_uncertainty_2026_week_4_v0_2.csv")
TEAM = Path("reports/matchup_predictions_2026_week_4.csv")
OUTPUT = Path("reports/market_role_audit_2026_week_4_v0_2.csv")

def safe_div(a, b):
    if pd.isna(a) or pd.isna(b) or float(b) == 0:
        return np.nan
    return float(a) / float(b)

def main():
    market = pd.read_csv(MARKET)
    team = pd.read_csv(TEAM)

    market["team_key"] = market["team"].astype(str).str.strip().str.lower()
    team["team_key"] = team["team"].astype(str).str.strip().str.lower()

    team_cols = [
        "game_id",
        "team_key",
        "proj_pass_attempts",
        "proj_rush_attempts",
        "proj_passing_yards",
        "proj_rushing_yards",
        "proj_total_plays",
        "proj_points",
    ]

    team_small = team[[c for c in team_cols if c in team.columns]].copy()

    out = market.merge(
        team_small,
        on=["game_id", "team_key"],
        how="left",
        validate="many_to_one",
    )

    out["player_team_volume_share"] = np.nan
    out["role_audit_flag"] = "OK"
    out["role_audit_note"] = ""

    for idx, row in out.iterrows():
        target = row["target"]
        projection = float(row["projection_blended"])

        share = np.nan
        flag = "OK"
        note = ""

        if target == "pass_attempts":
            share = safe_div(projection, row.get("proj_pass_attempts"))
            if pd.notna(share) and share > 1.10:
                flag = "RED_FLAG"
                note = "Player pass attempts exceed team pass-attempt projection by >10%."
            elif pd.notna(share) and share < 0.65:
                flag = "REVIEW"
                note = "QB projects for <65% of team pass attempts; possible split-role issue."

        elif target == "completions":
            share = safe_div(projection, row.get("proj_pass_attempts"))
            if pd.notna(share) and share > 0.85:
                flag = "RED_FLAG"
                note = "Projected completions exceed 85% of team pass attempts."

        elif target == "passing_yards":
            share = safe_div(projection, row.get("proj_passing_yards"))
            if pd.notna(share) and share > 1.10:
                flag = "RED_FLAG"
                note = "Player passing yards exceed team passing-yard projection by >10%."
            elif pd.notna(share) and share < 0.65:
                flag = "REVIEW"
                note = "QB accounts for <65% of team passing yards; possible role/split issue."

        elif target == "rush_attempts":
            share = safe_div(projection, row.get("proj_rush_attempts"))
            if pd.notna(share) and share > 0.90:
                flag = "RED_FLAG"
                note = "Player projects for >90% of team rushing attempts."
            elif pd.notna(share) and share > 0.75:
                flag = "REVIEW"
                note = "Player projects for an unusually concentrated rushing-attempt share."

        elif target == "rushing_yards":
            share = safe_div(projection, row.get("proj_rushing_yards"))
            if pd.notna(share) and share > 1.00:
                flag = "RED_FLAG"
                note = "Player rushing yards exceed entire team rushing-yard projection."
            elif pd.notna(share) and share > 0.80:
                flag = "REVIEW"
                note = "Player projects for >80% of team rushing yards."

        elif target == "receiving_yards":
            share = safe_div(projection, row.get("proj_passing_yards"))
            if pd.notna(share) and share > 0.70:
                flag = "RED_FLAG"
                note = "Receiver projects for >70% of team passing yards."
            elif pd.notna(share) and share > 0.50:
                flag = "REVIEW"
                note = "Receiver projects for >50% of team passing yards."

        elif target == "receptions":
            share = safe_div(projection, row.get("proj_pass_attempts"))
            if pd.notna(share) and share > 0.45:
                flag = "RED_FLAG"
                note = "Receiver receptions exceed 45% of team pass-attempt projection."
            elif pd.notna(share) and share > 0.32:
                flag = "REVIEW"
                note = "Receiver projects for a very concentrated share of team pass attempts."

        out.at[idx, "player_team_volume_share"] = share
        out.at[idx, "role_audit_flag"] = flag
        out.at[idx, "role_audit_note"] = note

    def final_screen(row):
        original = row["screen_status"]
        flag = row["role_audit_flag"]

        if original == "PASS":
            return "PASS"
        if flag == "RED_FLAG":
            return "PASS_ROLE_CONFLICT"
        if original == "CANDIDATE" and flag == "REVIEW":
            return "REVIEW"
        return original

    out["post_role_status"] = out.apply(final_screen, axis=1)

    rank = {
        "CANDIDATE": 0,
        "WATCH": 1,
        "REVIEW": 2,
        "PASS_ROLE_CONFLICT": 3,
        "PASS": 4,
    }

    out["_rank"] = out["post_role_status"].map(rank).fillna(9)
    out = out.sort_values(
        ["_rank", "adjusted_ev"],
        ascending=[True, False],
    ).drop(columns=["_rank", "team_key"]).reset_index(drop=True)

    out.to_csv(OUTPUT, index=False)

    print("=" * 135)
    print("CFB PLAYER PROP ROLE + TEAM VOLUME AUDIT")
    print("=" * 135)
    print()
    print(out["post_role_status"].value_counts().to_string())
    print()

    cols = [
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
        "player_team_volume_share",
        "role_audit_flag",
        "role_audit_note",
        "post_role_status",
    ]

    cols = [c for c in cols if c in out.columns]

    print("SURVIVING / REVIEW ROWS")
    print("-" * 135)
    print(
        out.loc[
            out["post_role_status"].isin(["CANDIDATE","WATCH","REVIEW"]),
            cols,
        ]
        .head(75)
        .round(4)
        .to_string(index=False)
    )

    print()
    print("Saved:", OUTPUT)
    print()
    print("This is still a diagnostic screen, not final official-bet classification.")

if __name__ == "__main__":
    main()
