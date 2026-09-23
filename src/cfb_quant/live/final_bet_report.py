from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("reports/final_review_top10_2026_week_4_v0_1.csv")
OUTPUT = Path("reports/final_bet_report_2026_week_4_v0_1.csv")

UNIT_DOLLARS = 25.0

def decimal_odds(odds):
    odds = float(odds)
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    return 1.0 + odds / 100.0

def implied_probability(odds):
    odds = float(odds)
    if odds < 0:
        return (-odds) / ((-odds) + 100.0)
    return 100.0 / (odds + 100.0)

def fair_american(p):
    p = float(np.clip(p, 0.0001, 0.9999))
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p

def max_playable_american(p, minimum_ev=0.05):
    p = float(np.clip(p, 0.0001, 0.9999))

    required_decimal = (1.0 + minimum_ev) / p

    if required_decimal <= 1.0:
        return -10000.0

    if required_decimal >= 2.0:
        return 100.0 * (required_decimal - 1.0)

    return -100.0 / (required_decimal - 1.0)

def stake_from_signal(prob, edge, ev, confidence):
    # Conservative beta staking. Do not size solely from raw EV.
    if prob >= 0.78 and edge >= 0.24 and confidence >= 0.96 and ev >= 0.35:
        return 0.80
    if prob >= 0.74 and edge >= 0.20 and confidence >= 0.94 and ev >= 0.25:
        return 0.70
    if prob >= 0.70 and edge >= 0.16 and confidence >= 0.92 and ev >= 0.18:
        return 0.60
    if prob >= 0.66 and edge >= 0.12 and confidence >= 0.90 and ev >= 0.12:
        return 0.50
    return 0.40

def classify(prob, edge, ev, confidence):
    if prob >= 0.70 and edge >= 0.15 and ev >= 0.15 and confidence >= 0.90:
        return "BET"
    if prob >= 0.62 and edge >= 0.08 and ev >= 0.08:
        return "LEAN"
    return "PASS"

def main():
    df = pd.read_csv(INPUT)

    df["decision"] = df.apply(
        lambda r: classify(
            float(r["adjusted_probability"]),
            float(r["adjusted_probability_edge"]),
            float(r["adjusted_ev"]),
            float(r["uncertainty_confidence"]),
        ),
        axis=1,
    )

    df["fair_odds_adjusted"] = df["adjusted_probability"].apply(fair_american)
    df["max_playable_price_5pct_ev"] = df["adjusted_probability"].apply(
        lambda p: max_playable_american(float(p), 0.05)
    )

    df["stake_units"] = df.apply(
        lambda r: stake_from_signal(
            float(r["adjusted_probability"]),
            float(r["adjusted_probability_edge"]),
            float(r["adjusted_ev"]),
            float(r["uncertainty_confidence"]),
        ) if r["decision"] == "BET" else 0.0,
        axis=1,
    )

    df["stake_dollars"] = df["stake_units"] * UNIT_DOLLARS

    df["projection_vs_line"] = (
        df["projection_blended"] - df["line"]
    )

    order = {"BET": 0, "LEAN": 1, "PASS": 2}
    df["_decision_rank"] = df["decision"].map(order)

    df = df.sort_values(
        ["_decision_rank", "adjusted_ev", "adjusted_probability_edge"],
        ascending=[True, False, False],
    ).drop(columns="_decision_rank").reset_index(drop=True)

    df["final_rank"] = range(1, len(df) + 1)

    df.to_csv(OUTPUT, index=False)

    print("=" * 155)
    print("CFB WEEK 4 FINAL BET REPORT - BETA")
    print("=" * 155)
    print()
    print(df["decision"].value_counts().to_string())
    print()

    cols = [
        "final_rank",
        "away_team",
        "home_team",
        "player_name_model",
        "target",
        "raw_best_side",
        "line",
        "projection_blended",
        "projection_vs_line",
        "raw_best_odds",
        "adjusted_probability",
        "adjusted_probability_edge",
        "adjusted_ev",
        "fair_odds_adjusted",
        "max_playable_price_5pct_ev",
        "uncertainty_confidence",
        "decision",
        "stake_units",
        "stake_dollars",
    ]

    cols = [c for c in cols if c in df.columns]

    print(df[cols].round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)
    print()
    print("BET status is beta-model qualification, not historically ROI-validated.")

if __name__ == "__main__":
    main()
