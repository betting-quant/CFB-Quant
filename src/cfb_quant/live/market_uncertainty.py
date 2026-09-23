from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("reports/market_probability_2026_week_4_v0_3_calibrated.csv")
OUTPUT = Path("reports/market_uncertainty_2026_week_4_v0_2.csv")

def decimal_odds(odds):
    odds = float(odds)
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    return 1.0 + odds / 100.0

def expected_value(probability, odds):
    d = decimal_odds(odds)
    return probability * (d - 1.0) - (1.0 - probability)

def probability_to_american(p):
    p = float(np.clip(p, 0.0001, 0.9999))
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p

def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default

def main():
    df = pd.read_csv(INPUT)

    rows = []

    for _, row in df.iterrows():
        raw_p = safe_float(row["raw_best_probability"], 0.5)
        raw_edge = safe_float(row["raw_best_probability_edge"], 0.0)
        raw_ev = safe_float(row["raw_best_ev"], 0.0)
        odds = safe_float(row["raw_best_odds"], -115.0)

        local_std = max(safe_float(row.get("local_residual_std"), 0.0), 1e-6)
        disagreement = abs(safe_float(row.get("model_disagreement"), 0.0))

        player_games = safe_float(row.get("player_games_before"), 0.0)
        season_games = safe_float(row.get("player_season_games_before"), 0.0)

        local_distance = safe_float(
            row.get("local_projection_mean_abs_distance"),
            0.0,
        )

        projection = abs(safe_float(row.get("projection_blended"), 0.0))

        standardized_disagreement = disagreement / local_std

        disagreement_factor = float(
            np.exp(-0.70 * standardized_disagreement)
        )

        career_sample_factor = float(
            min(1.0, np.sqrt(max(player_games, 0.0) / 8.0))
        )

        season_sample_factor = float(
            min(1.0, np.sqrt(max(season_games, 0.0) / 3.0))
        )

        if projection > 0:
            neighborhood_ratio = local_distance / projection
        else:
            neighborhood_ratio = 0.0

        neighborhood_factor = float(
            np.exp(-1.50 * neighborhood_ratio)
        )

        confidence = (
            0.40 * disagreement_factor
            + 0.25 * career_sample_factor
            + 0.25 * season_sample_factor
            + 0.10 * neighborhood_factor
        )

        confidence = float(np.clip(confidence, 0.20, 1.00))

        adjusted_p = 0.50 + (raw_p - 0.50) * confidence
        adjusted_p = float(np.clip(adjusted_p, 0.01, 0.99))

        market_p = safe_float(row["raw_best_no_vig_probability"], 0.5)
        adjusted_edge = adjusted_p - market_p
        adjusted_ev = expected_value(adjusted_p, odds)
        adjusted_fair_odds = probability_to_american(adjusted_p)

        if (
            adjusted_ev >= 0.08
            and adjusted_edge >= 0.06
            and confidence >= 0.60
            and raw_ev > 0
        ):
            screen = "CANDIDATE"
        elif (
            adjusted_ev >= 0.03
            and adjusted_edge >= 0.03
            and confidence >= 0.45
        ):
            screen = "WATCH"
        else:
            screen = "PASS"

        record = row.to_dict()
        record.update({
            "standardized_model_disagreement": standardized_disagreement,
            "disagreement_factor": disagreement_factor,
            "career_sample_factor": career_sample_factor,
            "season_sample_factor": season_sample_factor,
            "neighborhood_ratio": neighborhood_ratio,
            "neighborhood_factor": neighborhood_factor,
            "uncertainty_confidence": confidence,
            "adjusted_probability": adjusted_p,
            "adjusted_probability_edge": adjusted_edge,
            "adjusted_ev": adjusted_ev,
            "adjusted_fair_odds": adjusted_fair_odds,
            "screen_status": screen,
        })

        rows.append(record)

    out = pd.DataFrame(rows)

    rank_map = {"CANDIDATE": 0, "WATCH": 1, "PASS": 2}
    out["_rank"] = out["screen_status"].map(rank_map)

    out = out.sort_values(
        ["_rank", "adjusted_ev", "adjusted_probability_edge"],
        ascending=[True, False, False],
    ).drop(columns="_rank").reset_index(drop=True)

    out.to_csv(OUTPUT, index=False)

    print("=" * 130)
    print("CFB PLAYER PROP UNCERTAINTY SCREEN")
    print("=" * 130)
    print()
    print(out["screen_status"].value_counts().to_string())
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
        "raw_best_probability",
        "uncertainty_confidence",
        "adjusted_probability",
        "adjusted_probability_edge",
        "adjusted_ev",
        "model_disagreement",
        "player_games_before",
        "player_season_games_before",
        "screen_status",
    ]

    cols = [c for c in cols if c in out.columns]

    print("CANDIDATES + WATCH LIST")
    print("-" * 130)
    print(
        out.loc[out["screen_status"].ne("PASS"), cols]
        .head(60)
        .round(4)
        .to_string(index=False)
    )

    print()
    print("Saved:", OUTPUT)
    print()
    print("CANDIDATE is a screening label only. It is not yet an official bet.")

if __name__ == "__main__":
    main()
