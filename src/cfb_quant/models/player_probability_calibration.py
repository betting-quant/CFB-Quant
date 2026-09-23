from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("reports/player_prop_holdout_predictions_2025_v0_1.csv")
OUTPUT = Path("reports/player_probability_calibration_2025_v0_1.csv")
SUMMARY = Path("reports/player_probability_calibration_summary_2025_v0_1.csv")

NEIGHBORS = {
    "pass_attempts": 600,
    "completions": 600,
    "passing_yards": 600,
    "rush_attempts": 1000,
    "rushing_yards": 1000,
    "receptions": 1400,
    "receiving_yards": 1400,
}

OFFSETS = {
    "pass_attempts": [-6.5,-4.5,-2.5,-0.5,1.5,3.5,5.5],
    "completions": [-5.5,-3.5,-1.5,-0.5,1.5,3.5,5.5],
    "passing_yards": [-60.5,-40.5,-20.5,-0.5,19.5,39.5,59.5],
    "rush_attempts": [-5.5,-3.5,-1.5,-0.5,1.5,3.5,5.5],
    "rushing_yards": [-35.5,-25.5,-15.5,-5.5,4.5,14.5,24.5,34.5],
    "receptions": [-2.5,-1.5,-0.5,0.5,1.5,2.5],
    "receiving_yards": [-35.5,-25.5,-15.5,-5.5,4.5,14.5,24.5,34.5],
}

def empirical_over_probability(residuals, threshold):
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    above = n - np.searchsorted(r, threshold, side="right")
    return (above + 0.5) / (n + 1.0)

def main():
    df = pd.read_csv(INPUT)
    df = df.dropna(subset=["target","prediction_blended","actual","residual"]).copy()

    rows = []

    for target, group in df.groupby("target"):
        group = group.reset_index(drop=True)
        k = NEIGHBORS[target]

        for i, row in group.iterrows():
            pool = group.drop(index=i).copy()
            pool["distance"] = np.abs(pool["prediction_blended"] - row["prediction_blended"])
            local = pool.nsmallest(min(k, len(pool)), "distance")
            residuals = local["residual"].to_numpy(dtype=float)

            for offset in OFFSETS[target]:
                line = float(row["prediction_blended"] + offset)
                threshold = line - float(row["prediction_blended"])
                p_over = empirical_over_probability(residuals, threshold)
                actual_over = float(row["actual"] > line)

                rows.append({
                    "target": target,
                    "season": row.get("season"),
                    "week": row.get("week"),
                    "game_id": row.get("game_id"),
                    "player_name": row.get("player_name"),
                    "projection": row["prediction_blended"],
                    "actual": row["actual"],
                    "line": line,
                    "offset": offset,
                    "predicted_over_probability": p_over,
                    "actual_over": actual_over,
                })

    out = pd.DataFrame(rows)
    out["prob_bin"] = pd.cut(
        out["predicted_over_probability"],
        bins=np.arange(0,1.0001,0.05),
        include_lowest=True,
    )

    out.to_csv(OUTPUT, index=False)

    summary = out.groupby(["target","prob_bin"], observed=True).agg(
        rows=("actual_over","size"),
        predicted_probability=("predicted_over_probability","mean"),
        observed_frequency=("actual_over","mean"),
    ).reset_index()

    summary["calibration_error"] = (
        summary["observed_frequency"] - summary["predicted_probability"]
    )

    summary.to_csv(SUMMARY, index=False)

    print("=" * 110)
    print("2025 PLAYER PROP PROBABILITY CALIBRATION")
    print("=" * 110)

    overall = out.groupby("target").apply(
        lambda g: pd.Series({
            "rows": len(g),
            "brier": np.mean((g["predicted_over_probability"] - g["actual_over"]) ** 2),
            "mean_predicted": g["predicted_over_probability"].mean(),
            "mean_observed": g["actual_over"].mean(),
        }),
        include_groups=False,
    ).reset_index()

    print(overall.round(4).to_string(index=False))
    print()
    print("Saved:", OUTPUT)
    print("Saved:", SUMMARY)

if __name__ == "__main__":
    main()
