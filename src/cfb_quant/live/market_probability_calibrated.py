from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

INPUT = Path("reports/market_probability_2026_week_4_v0_2.csv")
CALIBRATION_VALIDATION = Path("reports/player_probability_calibrator_validation_2025_v0_1.csv")
CALIBRATORS = Path("models/artifacts/cfb_player_probability_calibrators_v0_1.joblib")
OUTPUT = Path("reports/market_probability_2026_week_4_v0_3_calibrated.csv")

def decimal_odds(odds):
    odds = float(odds)
    if odds < 0:
        return 1.0 + 100.0 / (-odds)
    return 1.0 + odds / 100.0

def expected_value(p, odds):
    d = decimal_odds(odds)
    return p * (d - 1.0) - (1.0 - p)

def probability_to_american(p):
    p = float(np.clip(p, 0.0001, 0.9999))
    if p >= 0.5:
        return -100.0 * p / (1.0 - p)
    return 100.0 * (1.0 - p) / p

def main():
    df = pd.read_csv(INPUT)
    validation = pd.read_csv(CALIBRATION_VALIDATION)
    calibrators = joblib.load(CALIBRATORS)

    # Only use calibration where it improved BOTH held-out Brier and ECE.
    approved = set(
        validation.loc[
            (validation["brier_improvement"] > 0)
            & (validation["ece_improvement"] > 0),
            "target",
        ]
    )

    print("Calibration approved for:", sorted(approved))

    rows = []

    for _, row in df.iterrows():
        target = row["target"]
        raw_over = float(row["model_probability_over"])

        if target in approved:
            calibrated_over = float(
                calibrators[target].predict(np.array([raw_over]))[0]
            )
            calibration_used = True
        else:
            calibrated_over = raw_over
            calibration_used = False

        calibrated_over = float(np.clip(calibrated_over, 0.01, 0.99))
        calibrated_under = 1.0 - calibrated_over

        over_odds = float(row["over_odds"])
        under_odds = float(row["under_odds"])

        no_vig_over = float(row["no_vig_probability_over"])
        no_vig_under = float(row["no_vig_probability_under"])

        edge_over = calibrated_over - no_vig_over
        edge_under = calibrated_under - no_vig_under

        ev_over = expected_value(calibrated_over, over_odds)
        ev_under = expected_value(calibrated_under, under_odds)

        if ev_over >= ev_under:
            best_side = "OVER"
            best_probability = calibrated_over
            best_market_probability = no_vig_over
            best_edge = edge_over
            best_ev = ev_over
            best_odds = over_odds
        else:
            best_side = "UNDER"
            best_probability = calibrated_under
            best_market_probability = no_vig_under
            best_edge = edge_under
            best_ev = ev_under
            best_odds = under_odds

        record = row.to_dict()
        record.update({
            "calibration_used": calibration_used,
            "probability_over_precalibration": raw_over,
            "probability_over_calibrated": calibrated_over,
            "probability_under_calibrated": calibrated_under,
            "calibrated_edge_over": edge_over,
            "calibrated_edge_under": edge_under,
            "calibrated_ev_over": ev_over,
            "calibrated_ev_under": ev_under,
            "calibrated_fair_odds_over": probability_to_american(calibrated_over),
            "calibrated_fair_odds_under": probability_to_american(calibrated_under),
            "raw_best_side": best_side,
            "raw_best_probability": best_probability,
            "raw_best_no_vig_probability": best_market_probability,
            "raw_best_probability_edge": best_edge,
            "raw_best_ev": best_ev,
            "raw_best_odds": best_odds,
        })

        rows.append(record)

    out = pd.DataFrame(rows)
    out = out.sort_values(
        ["raw_best_ev","raw_best_probability_edge"],
        ascending=[False,False],
    ).reset_index(drop=True)

    out.to_csv(OUTPUT, index=False)

    print()
    print("=" * 120)
    print("SELECTIVELY CALIBRATED WEEK 4 MARKET PROBABILITIES")
    print("=" * 120)
    print("Rows:", len(out))
    print("Calibration used:", int(out["calibration_used"].sum()))
    print()

    cols = [
        "player_name_model",
        "target",
        "line",
        "projection_blended",
        "raw_best_side",
        "raw_best_odds",
        "raw_best_probability",
        "raw_best_probability_edge",
        "raw_best_ev",
        "calibration_used",
    ]

    print(out[cols].head(40).round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)

if __name__ == "__main__":
    main()
